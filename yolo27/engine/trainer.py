import torch
import torch.nn as nn
from ..loss.assigner import TaskAlignedAssigner
from ..loss.detection_loss import DetectionLoss
from ..loss.segmentation_loss import (
    semantic_segmentation_loss, boundary_loss,
    mask_quality_loss, discriminative_instance_loss
)
from ..loss.depth_3d_loss import scale_invariant_log_depth_loss, loss_3d_detection
from ..loss.consistency_loss import TransformAlignedConsistencyLoss
from ..loss.multitask_loss import AdaptiveTaskLoss
from ..data.augmentation import GeometricMultiViewAugment

class YOLO27MultiTaskTrainer:
    """
    Complete end-to-end multi-task trainer for YOLO27 v0.5.
    Connects:
      Batch -> Augmentation -> Model -> Assigner -> Loss functions -> Adaptive weighting -> Backward -> Optimizer
    Features:
      - Gradient clipping
      - Modality masking for missing GT
      - AMP mixed precision support
      - Multi-view transform-aligned consistency
    """
    def __init__(self, model, optimizer, device="cpu", num_classes: int = 80, clip_grad_norm: float = 10.0, use_amp: bool = False):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.clip_grad_norm = clip_grad_norm
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())

        # Sub-modules
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=num_classes)
        self.det_loss_fn = DetectionLoss(num_classes=num_classes)
        self.consistency_loss_fn = TransformAlignedConsistencyLoss()
        self.augmenter = GeometricMultiViewAugment(p_flip=0.5)

        # Multi-task adaptive loss with uncertainty weighting
        tasks = ["det", "semantic", "instance", "boundary", "mask_quality", "depth", "geometry_3d", "consistency"]
        self.task_loss_module = AdaptiveTaskLoss(tasks).to(device)
        self.optimizer.add_param_group({"params": self.task_loss_module.parameters(), "lr": optimizer.param_groups[0]["lr"]})

    def train_step(self, batch: dict) -> tuple:
        """
        Executes one full training step on a multi-task batch.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        images = batch["images"].to(self.device)
        gt_boxes = batch["gt_boxes"].to(self.device)
        gt_labels = batch["gt_labels"].to(self.device)
        mask_gt = batch["mask_gt"].to(self.device)
        sem_gt = batch["semantic_masks"].to(self.device)
        inst_gt = batch["instance_masks"].to(self.device)
        bound_gt = batch["boundary_maps"].to(self.device)
        task_masks = batch.get("task_masks", {})

        # 1. Primary forward pass
        out = self.model(images)

        # 2. Dynamic target assignment for 2D/3D detection
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pd_scores=torch.sigmoid(out["class_logits"]),
            pd_bboxes=out["pred_boxes"],
            anc_points=out["anchor_points"],
            gt_labels=gt_labels,
            gt_bboxes=gt_boxes,
            mask_gt=mask_gt
        )

        # 3. Compute component losses
        losses = {}

        # 3.1 Detection loss
        loss_det, det_dict = self.det_loss_fn(
            pred_scores=out["class_logits"],
            pred_bboxes=out["pred_boxes"],
            target_scores=target_scores,
            target_bboxes=target_bboxes,
            fg_mask=fg_mask
        )
        losses["det"] = loss_det

        # 3.2 Semantic segmentation loss
        losses["semantic"] = semantic_segmentation_loss(out["semantic_masks"], sem_gt)

        # 3.3 Boundary segmentation loss
        losses["boundary"] = boundary_loss(out["boundary_map"], bound_gt)

        # 3.4 Instance embedding loss
        losses["instance"] = discriminative_instance_loss(out["instance_embeddings"], inst_gt)

        # 3.5 Mask quality loss
        pred_sem_binary = (torch.argmax(out["semantic_masks"], dim=1, keepdim=True) > 0).float()
        gt_sem_binary = (sem_gt.unsqueeze(1) > 0).float()
        losses["mask_quality"] = mask_quality_loss(out["mask_quality"], pred_sem_binary, gt_sem_binary)

        # 3.6 Depth loss (if depth ground truth is available)
        if "depth" in batch and task_masks.get("depth", 0) > 0:
            gt_depth = batch["depth"].to(self.device)
            valid_depth = batch["valid_depth"].to(self.device)
            losses["depth"] = scale_invariant_log_depth_loss(out["depth_map"], gt_depth, valid_depth)
        else:
            losses["depth"] = out["depth_inverse"].sum() * 0.0

        # 3.7 3D geometry loss (if 3D ground truth is available)
        if "gt_3d_xyz" in batch and task_masks.get("geometry_3d", 0) > 0:
            # Positive detections supervision
            pos_mask_3d = fg_mask & (target_gt_idx >= 0)
            if pos_mask_3d.any():
                gt_3d_xyz = batch["gt_3d_xyz"].to(self.device)
                gt_3d_lwh = batch["gt_3d_lwh"].to(self.device)
                gt_3d_yaw = batch["gt_3d_yaw"].to(self.device)

                # Gather corresponding matched GT targets
                matched_xyz = torch.cat([gt_3d_xyz[b, target_gt_idx[b, pos_mask_3d[b]]] for b in range(images.shape[0]) if pos_mask_3d[b].any()], dim=0)
                matched_lwh = torch.cat([gt_3d_lwh[b, target_gt_idx[b, pos_mask_3d[b]]] for b in range(images.shape[0]) if pos_mask_3d[b].any()], dim=0)
                matched_yaw = torch.cat([gt_3d_yaw[b, target_gt_idx[b, pos_mask_3d[b]]] for b in range(images.shape[0]) if pos_mask_3d[b].any()], dim=0)

                pos_pred_xyz = out["pred_xyz"][pos_mask_3d]
                pos_pred_lwh = out["pred_lwh"][pos_mask_3d]
                pos_pred_yaw = out["pred_yaw_sincos"][pos_mask_3d]

                loss_3d, _ = loss_3d_detection(
                    pos_pred_xyz, pos_pred_lwh, pos_pred_yaw,
                    matched_xyz, matched_lwh, matched_yaw,
                    pos_mask=torch.ones(pos_pred_xyz.shape[0], dtype=torch.bool, device=self.device),
                    log_sigma_xyz=out["log_sigma_xyz"],
                    log_sigma_lwh=out["log_sigma_lwh"],
                    log_sigma_yaw=out["log_sigma_yaw"]
                )
                losses["geometry_3d"] = loss_3d
            else:
                losses["geometry_3d"] = out["pred_xyz"].sum() * 0.0
        else:
            losses["geometry_3d"] = out["pred_xyz"].sum() * 0.0

        # 3.8 Consistency loss: generate augmented view with tracked inverse transform
        if task_masks.get("consistency", 1.0) > 0:
            with torch.no_grad():
                aug_images, inv_mat, is_hflip = self.augmenter(images)
            out_aug = self.model(aug_images)
            loss_cons = self.consistency_loss_fn(out, out_aug, inv_mat, is_hflip=is_hflip)
            losses["consistency"] = loss_cons
        else:
            losses["consistency"] = out["class_logits"].sum() * 0.0

        # 4. Total adaptive loss aggregation with active task masks
        total_loss, raw_dict, weighted_dict, weights = self.task_loss_module(losses, task_masks=task_masks)

        # 5. Backward & Optimization
        total_loss.backward()

        # Gradient clipping
        all_params = list(self.model.parameters()) + list(self.task_loss_module.parameters())
        grad_norm = nn.utils.clip_grad_norm_(all_params, max_norm=self.clip_grad_norm)

        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            "raw_losses": raw_dict,
            "weighted_losses": weighted_dict,
            "task_weights": weights
        }
