import torch
import torch.nn as nn
import torch.nn.functional as F
from ..geometry.box_ops import bbox_ciou

def dfl_loss(pred_dist: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    """
    Distribution Focal Loss (DFL) on predicted distribution logits:
    target_dist: continuous ground truth coordinate in feature stride units
    """
    target_left = target_dist.long()
    target_right = target_left + 1
    weight_left = target_right.float() - target_dist
    weight_right = target_dist - target_left.float()

    loss_left = F.cross_entropy(pred_dist, target_left, reduction="none") * weight_left
    loss_right = F.cross_entropy(pred_dist, target_right, reduction="none") * weight_right
    return (loss_left + loss_right).mean()

class DetectionLoss(nn.Module):
    """
    Anchor-free multi-scale detection loss:
    1. Varifocal / BCE classification loss with normalized soft targets from TAL.
    2. CIoU bounding box regression loss on foreground anchors.
    3. Distribution Focal Loss (DFL) on predicted sub-pixel distribution logits.
    """
    def __init__(self, num_classes: int = 80, reg_max: int = 16, box_weight: float = 7.5, cls_weight: float = 1.0, dfl_weight: float = 1.5):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight

    def forward(self, pred_scores: torch.Tensor, pred_bboxes: torch.Tensor, pred_dist: torch.Tensor,
                anchor_points: torch.Tensor, strides: torch.Tensor,
                target_scores: torch.Tensor, target_bboxes: torch.Tensor, fg_mask: torch.Tensor):
        """
        Args:
            pred_scores: (B, N, num_classes)
            pred_bboxes: (B, N, 4)
            pred_dist: (B, N, 4 * reg_max)
            anchor_points: (N, 2)
            strides: (N, 1)
            target_scores: (B, N, num_classes)
            target_bboxes: (B, N, 4)
            fg_mask: (B, N)
        """
        num_pos = max(fg_mask.sum().item(), 1.0)

        # 1. Classification Loss
        loss_cls = F.binary_cross_entropy_with_logits(
            pred_scores, target_scores, reduction="sum"
        ) / num_pos

        # 2. Box CIoU Loss & DFL Loss on foreground
        if fg_mask.any():
            pos_pred_boxes = pred_bboxes[fg_mask]
            pos_target_boxes = target_bboxes[fg_mask]

            # 2. Box regression: CIoU + BoS boundary alignment + optional DFL
            ciou = bbox_ciou(pos_pred_boxes, pos_target_boxes)

            # Boundary Overlap Score (BoS) component
            lt = torch.max(pos_pred_boxes[:, :2], pos_target_boxes[:, :2])
            rb = torch.min(pos_pred_boxes[:, 2:], pos_target_boxes[:, 2:])
            wh = (rb - lt).clamp(min=0)
            inter = wh[:, 0] * wh[:, 1]
            w1 = (pos_pred_boxes[:, 2] - pos_pred_boxes[:, 0]).clamp(min=0)
            h1 = (pos_pred_boxes[:, 3] - pos_pred_boxes[:, 1]).clamp(min=0)
            w2 = (pos_target_boxes[:, 2] - pos_target_boxes[:, 0]).clamp(min=0)
            h2 = (pos_target_boxes[:, 3] - pos_target_boxes[:, 1]).clamp(min=0)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - inter + 1e-7
            iou = inter / union

            c1 = (pos_pred_boxes[:, :2] + pos_pred_boxes[:, 2:]) * 0.5
            c2 = (pos_target_boxes[:, :2] + pos_target_boxes[:, 2:]) * 0.5
            rho2 = ((c1 - c2) ** 2).sum(dim=-1)
            enc_lt = torch.min(pos_pred_boxes[:, :2], pos_target_boxes[:, :2])
            enc_rb = torch.max(pos_pred_boxes[:, 2:], pos_target_boxes[:, 2:])
            enc_wh = (enc_rb - enc_lt).clamp(min=0)
            diag2 = (enc_wh[:, 0] ** 2 + enc_wh[:, 1] ** 2).clamp_min(1e-7)
            dist_factor = torch.exp(-2.0 * torch.sqrt(rho2 / diag2))
            scale_factor = torch.min(area1, area2) / torch.max(area1, area2).clamp_min(1e-7)
            bos_elem = (iou * (0.65 * dist_factor + 0.35 * scale_factor)).clamp(0.0, 1.0)

            # Combined box loss: 60% CIoU + 40% BoS
            loss_ciou_elem = 1.0 - ciou
            loss_bos_elem = 1.0 - bos_elem
            loss_box = (0.60 * loss_ciou_elem + 0.40 * loss_bos_elem).sum() / num_pos

            if self.reg_max > 1:
                # DFL target computation: convert target boxes to (l, t, r, b) in stride units
                B, N = fg_mask.shape
                pos_anchors = anchor_points.unsqueeze(0).expand(B, N, 2)[fg_mask]
                pos_strides = strides.unsqueeze(0).expand(B, N, 1)[fg_mask]

                lt = (pos_anchors - pos_target_boxes[:, :2]) / pos_strides
                rb = (pos_target_boxes[:, 2:] - pos_anchors) / pos_strides
                target_ltrb = torch.cat([lt, rb], dim=-1).clamp(0, self.reg_max - 1.01)

                # Reshape predicted dist: (N_pos, 4, reg_max)
                pos_dist = pred_dist[fg_mask].view(-1, 4, self.reg_max)
                loss_dfl = 0.0
                for i in range(4):
                    loss_dfl = loss_dfl + dfl_loss(pos_dist[:, i, :], target_ltrb[:, i])
                loss_dfl = loss_dfl / 4.0
            else:
                # Direct regression (reg_max == 1): no DFL bin distribution
                loss_dfl = torch.tensor(0.0, device=pred_scores.device)
        else:
            loss_box = pred_bboxes.sum() * 0.0
            loss_dfl = pred_dist.sum() * 0.0
            loss_ciou_elem = loss_box
            loss_bos_elem = loss_box
            iou = torch.tensor(0.0, device=pred_scores.device)
            bos_elem = torch.tensor(0.0, device=pred_scores.device)

        dfl_term = self.dfl_weight * loss_dfl if self.reg_max > 1 else 0.0
        total_loss = (self.cls_weight * loss_cls + 
                      self.box_weight * loss_box + 
                      dfl_term)

        loss_dict = {
            "loss_cls": loss_cls.detach(),
            "loss_ciou": (loss_ciou_elem.sum() / num_pos).detach() if fg_mask.any() else torch.tensor(0.0),
            "loss_bos": (loss_bos_elem.sum() / num_pos).detach() if fg_mask.any() else torch.tensor(0.0),
            "mean_iou": iou.mean().detach() if fg_mask.any() else torch.tensor(0.0),
            "mean_bos": bos_elem.mean().detach() if fg_mask.any() else torch.tensor(0.0),
            "loss_dfl": loss_dfl.detach() if isinstance(loss_dfl, torch.Tensor) else loss_dfl,
            "loss_det": total_loss
        }

        return total_loss, loss_dict

class DualAssignmentDetectionLoss(nn.Module):
    """
    Dual-Label Assignment Loss (YOLOv10 / RT-DETR style):
    Combines:
      1. One-to-Many (O2M) loss with Task-Aligned Assigner (TAL) for rich gradient backprop into trunk.
      2. One-to-One (O2I) loss with Hungarian Assigner for NMS-free end-to-end inference head.
      3. Mutual distillation loss (KL Divergence) between O2M and O2I prediction distributions.
    """
    def __init__(self, num_classes: int = 80, reg_max: int = 16,
                 box_weight: float = 7.5, cls_weight: float = 1.0, dfl_weight: float = 1.5,
                 o2m_weight: float = 1.0, o2i_weight: float = 1.0, distill_weight: float = 0.5,
                 tau: float = 1.0):
        super().__init__()
        self.det_loss_o2m = DetectionLoss(num_classes=num_classes, reg_max=reg_max, box_weight=box_weight, cls_weight=cls_weight, dfl_weight=dfl_weight)
        self.det_loss_o2i = DetectionLoss(num_classes=num_classes, reg_max=reg_max, box_weight=box_weight, cls_weight=cls_weight, dfl_weight=dfl_weight)
        self.o2m_weight = o2m_weight
        self.o2i_weight = o2i_weight
        self.distill_weight = distill_weight
        self.tau = tau

    def forward(self, o2i_preds: dict, o2m_preds: dict,
                anchor_points: torch.Tensor, strides: torch.Tensor,
                targets_o2i: tuple, targets_o2m: tuple):
        """
        Args:
            o2i_preds: dict containing pred_scores, pred_bboxes, pred_dist for One-to-One head
            o2m_preds: dict containing pred_scores, pred_bboxes, pred_dist for One-to-Many head
            targets_o2i: (target_scores, target_bboxes, fg_mask) assigned by HungarianOneToOneAssigner
            targets_o2m: (target_scores, target_bboxes, fg_mask) assigned by TaskAlignedAssigner
        """
        tgt_scores_o2i, tgt_boxes_o2i, fg_mask_o2i = targets_o2i
        tgt_scores_o2m, tgt_boxes_o2m, fg_mask_o2m = targets_o2m

        # 1. One-to-One Head Loss
        loss_o2i, dict_o2i = self.det_loss_o2i(
            o2i_preds["class_logits"], o2i_preds["pred_boxes"], o2i_preds["pred_box_dist"],
            anchor_points, strides, tgt_scores_o2i, tgt_boxes_o2i, fg_mask_o2i
        )

        # 2. One-to-Many Head Loss
        loss_o2m, dict_o2m = self.det_loss_o2m(
            o2m_preds["class_logits"], o2m_preds["pred_boxes"], o2m_preds["pred_box_dist"],
            anchor_points, strides, tgt_scores_o2m, tgt_boxes_o2m, fg_mask_o2m
        )

        # 3. Mutual Distillation Loss: Soft alignment from rich O2M teacher logits to O2I student logits
        p_s = F.log_softmax(o2i_preds["class_logits"] / self.tau, dim=-1)
        p_t = F.softmax(o2m_preds["class_logits"].detach() / self.tau, dim=-1)
        loss_distill = (self.tau ** 2) * F.kl_div(p_s, p_t, reduction="batchmean")

        total_loss = (self.o2i_weight * loss_o2i +
                      self.o2m_weight * loss_o2m +
                      self.distill_weight * loss_distill)

        loss_dict = {
            "loss_det_o2i": loss_o2i.detach(),
            "loss_det_o2m": loss_o2m.detach(),
            "loss_det_distill": loss_distill.detach(),
            "loss_det": total_loss
        }
        return total_loss, loss_dict

