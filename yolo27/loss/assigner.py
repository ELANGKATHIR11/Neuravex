import torch
import torch.nn as nn
import torch.nn.functional as F
from ..geometry.box_ops import box_cxcywh_to_xyxy, box_iou_2d

class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned Assigner (SimOTA / TAL style) for anchor-free multi-scale object detection.
    Aligns classification and box regression alignment metric:
        t = s^alpha * IoU^beta
    Selects top-k anchors per ground-truth bounding box.
    """
    def __init__(self, topk: int = 10, num_classes: int = 80, alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores: torch.Tensor, pd_bboxes: torch.Tensor, 
                anc_points: torch.Tensor, gt_labels: torch.Tensor, gt_bboxes: torch.Tensor,
                mask_gt: torch.Tensor):
        """
        Args:
            pd_scores: (B, N_anchors, num_classes) sigmoid classification scores
            pd_bboxes: (B, N_anchors, 4) predicted xyxy boxes
            anc_points: (N_anchors, 2) anchor centers (cx, cy)
            gt_labels: (B, max_gt, 1) or (B, max_gt) class indices
            gt_bboxes: (B, max_gt, 4) xyxy ground truth boxes
            mask_gt: (B, max_gt, 1) boolean mask indicating valid GT boxes
        Returns:
            target_labels: (B, N_anchors) class target (0 to num_classes-1, or background)
            target_bboxes: (B, N_anchors, 4) aligned box targets
            target_scores: (B, N_anchors, num_classes) soft alignment target scores
            fg_mask: (B, N_anchors) boolean mask of assigned foreground anchors
            target_gt_idx: (B, N_anchors) index of matched GT for 3D/mask supervision
        """
        B, N, C = pd_scores.shape
        max_gt = gt_bboxes.shape[1]

        target_labels = torch.full((B, N), self.num_classes, dtype=torch.long, device=pd_scores.device)
        target_bboxes = torch.zeros((B, N, 4), dtype=torch.float32, device=pd_bboxes.device)
        target_scores = torch.zeros((B, N, C), dtype=torch.float32, device=pd_scores.device)
        fg_mask = torch.zeros((B, N), dtype=torch.bool, device=pd_scores.device)
        target_gt_idx = torch.full((B, N), -1, dtype=torch.long, device=pd_scores.device)

        if max_gt == 0 or not mask_gt.bool().any():
            return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

        for b in range(B):
            n_gt = mask_gt[b].squeeze(-1).sum().item()
            if n_gt == 0:
                continue

            b_gt_boxes = gt_bboxes[b, :int(n_gt)]  # (M, 4)
            b_gt_labels = gt_labels[b, :int(n_gt)].long().squeeze(-1)  # (M,)
            b_pd_boxes = pd_bboxes[b]  # (N, 4)
            b_pd_scores = pd_scores[b]  # (N, C)

            # 1. Check anchor centers inside GT box (in_gts mask)
            x = anc_points[:, 0:1]  # (N, 1)
            y = anc_points[:, 1:2]
            in_x = (x >= b_gt_boxes[:, 0:1].T) & (x <= b_gt_boxes[:, 2:3].T)  # (N, M)
            in_y = (y >= b_gt_boxes[:, 1:2].T) & (y <= b_gt_boxes[:, 3:4].T)
            is_in_gts = (in_x & in_y).T  # (M, N)

            # Fallback if no anchor falls strictly within GT box (e.g. tiny box): use center proximity
            for m in range(int(n_gt)):
                if not is_in_gts[m].any():
                    gt_cx = (b_gt_boxes[m, 0] + b_gt_boxes[m, 2]) * 0.5
                    gt_cy = (b_gt_boxes[m, 1] + b_gt_boxes[m, 3]) * 0.5
                    dist = (anc_points[:, 0] - gt_cx).pow(2) + (anc_points[:, 1] - gt_cy).pow(2)
                    is_in_gts[m, dist.argmin()] = True

            # 2. Pairwise IoU between predicted boxes and GT boxes
            pairwise_iou = box_iou_2d(b_gt_boxes, b_pd_boxes)  # (M, N)

            # 3. Alignment metric: s^alpha * (IoU + 0.1)^beta
            cls_score_gt = b_pd_scores[:, b_gt_labels].T  # (M, N)
            align_metric = (cls_score_gt.clamp_min(self.eps).pow(self.alpha) * 
                            (pairwise_iou + 0.1).pow(self.beta))

            # Filter candidates: must be inside GT box
            align_metric = align_metric * is_in_gts.float()

            # 4. Top-K candidates per GT
            k = min(self.topk, N)
            topk_metrics, topk_idx = torch.topk(align_metric, k=k, dim=-1, largest=True)
            candidate_mask = torch.zeros_like(align_metric, dtype=torch.bool)
            candidate_mask.scatter_(dim=-1, index=topk_idx, value=True)
            candidate_mask = candidate_mask & is_in_gts

            # 5. Resolve conflicts if an anchor is assigned to multiple GTs
            cost_matrix = align_metric * candidate_mask.float()
            max_metric_per_anchor, best_gt_for_anchor = cost_matrix.max(dim=0)  # (N,)

            anchor_assigned = (max_metric_per_anchor > self.eps) & candidate_mask.any(dim=0)
            assigned_indices = anchor_assigned.nonzero(as_tuple=True)[0]

            if len(assigned_indices) > 0:
                matched_gt = best_gt_for_anchor[assigned_indices]
                fg_mask[b, assigned_indices] = True
                target_labels[b, assigned_indices] = b_gt_labels[matched_gt]
                target_bboxes[b, assigned_indices] = b_gt_boxes[matched_gt]
                target_gt_idx[b, assigned_indices] = matched_gt

                soft_scores = (pairwise_iou[matched_gt, assigned_indices]).clamp(0.0, 1.0)
                # Assign soft alignment score
                for idx, gt_cls, s_val in zip(assigned_indices, b_gt_labels[matched_gt], soft_scores):
                    target_scores[b, idx, gt_cls] = max(float(s_val), 0.5)

        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx
