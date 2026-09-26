import torch
import torch.nn as nn
import torch.nn.functional as F
from ..geometry.box_ops import box_cxcywh_to_xyxy, box_iou_2d

class TaskAlignedAssigner(nn.Module):
    """
    Standard Task-Aligned Assigner (TAL) for anchor-free multi-scale object detection.
    Computes alignment metric:
        t = s^alpha * IoU^beta
    Normalizes target alignment scores such that the maximum score per GT is 1.0.
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
            target_labels: (B, N_anchors) class target
            target_bboxes: (B, N_anchors, 4) aligned box targets
            target_scores: (B, N_anchors, num_classes) normalized soft alignment target scores
            fg_mask: (B, N_anchors) boolean mask of assigned foreground anchors
            target_gt_idx: (B, N_anchors) index of matched GT
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

            # 1. Candidate selection: anchor points inside GT boxes
            x = anc_points[:, 0:1]
            y = anc_points[:, 1:2]
            in_x = (x >= b_gt_boxes[:, 0:1].T) & (x <= b_gt_boxes[:, 2:3].T)
            in_y = (y >= b_gt_boxes[:, 1:2].T) & (y <= b_gt_boxes[:, 3:4].T)
            is_in_gts = (in_x & in_y).T  # (M, N)

            # Fallback for tiny/extreme boxes: assign nearest anchor center
            for m in range(int(n_gt)):
                if not is_in_gts[m].any():
                    gt_cx = (b_gt_boxes[m, 0] + b_gt_boxes[m, 2]) * 0.5
                    gt_cy = (b_gt_boxes[m, 1] + b_gt_boxes[m, 3]) * 0.5
                    dist = (anc_points[:, 0] - gt_cx).pow(2) + (anc_points[:, 1] - gt_cy).pow(2)
                    is_in_gts[m, dist.argmin()] = True

            # 2. Pairwise IoU
            pairwise_iou = box_iou_2d(b_gt_boxes, b_pd_boxes)  # (M, N)

            # 3. Alignment metric: s^alpha * IoU^beta
            cls_score_gt = b_pd_scores[:, b_gt_labels].T  # (M, N)
            align_metric = (cls_score_gt.clamp_min(self.eps).pow(self.alpha) * 
                            (pairwise_iou + 0.05).pow(self.beta))

            align_metric = align_metric * is_in_gts.float()

            # 4. Top-K candidates per GT
            k = min(self.topk, N)
            topk_metrics, topk_idx = torch.topk(align_metric, k=k, dim=-1, largest=True)
            candidate_mask = torch.zeros_like(align_metric, dtype=torch.bool)
            candidate_mask.scatter_(dim=-1, index=topk_idx, value=True)
            candidate_mask = candidate_mask & is_in_gts

            # 5. Cost matrix and GT conflict resolution
            cost_matrix = align_metric * candidate_mask.float()
            max_metric_per_anchor, best_gt_for_anchor = cost_matrix.max(dim=0)

            anchor_assigned = (max_metric_per_anchor > self.eps) & candidate_mask.any(dim=0)
            assigned_indices = anchor_assigned.nonzero(as_tuple=True)[0]

            if len(assigned_indices) > 0:
                matched_gt = best_gt_for_anchor[assigned_indices]
                fg_mask[b, assigned_indices] = True
                target_labels[b, assigned_indices] = b_gt_labels[matched_gt]
                target_bboxes[b, assigned_indices] = b_gt_boxes[matched_gt]
                target_gt_idx[b, assigned_indices] = matched_gt

                # Standard target normalization:
                # normalize soft score by max metric for that GT, multiplied by IoU
                for m in range(int(n_gt)):
                    m_anchors = assigned_indices[matched_gt == m]
                    if len(m_anchors) == 0:
                        continue
                    m_metrics = max_metric_per_anchor[m_anchors]
                    m_max = m_metrics.max().clamp_min(self.eps)
                    m_ious = pairwise_iou[m, m_anchors].clamp(0.0, 1.0)
                    norm_scores = (m_metrics / m_max) * m_ious
                    c_id = b_gt_labels[m].item()
                    target_scores[b, m_anchors, c_id] = norm_scores.clamp(0.0, 1.0)

        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

class HungarianOneToOneAssigner(nn.Module):
    """
    Exact One-to-One (O2I) Hungarian Assigner for NMS-Free End-to-End Detection (YOLOv10 / RT-DETR style).
    Performs optimal bipartite matching between predictions and ground-truth boxes.
    Cost:
        C = lambda_cls * C_cls + lambda_l1 * C_l1 + lambda_giou * C_giou
    """
    def __init__(self, num_classes: int = 80, cost_cls: float = 1.0, cost_l1: float = 5.0, cost_giou: float = 2.0, eps: float = 1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.cost_cls = cost_cls
        self.cost_l1 = cost_l1
        self.cost_giou = cost_giou
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores: torch.Tensor, pd_bboxes: torch.Tensor,
                anc_points: torch.Tensor, gt_labels: torch.Tensor, gt_bboxes: torch.Tensor,
                mask_gt: torch.Tensor):
        """
        Args:
            pd_scores: (B, N, num_classes) sigmoid/logits scores
            pd_bboxes: (B, N, 4) in xyxy format
            anc_points: (N, 2) anchor points
            gt_labels: (B, M, 1) or (B, M) class indices
            gt_bboxes: (B, M, 4) in xyxy format
            mask_gt: (B, M, 1) or (B, M) valid ground truth mask
        """
        from scipy.optimize import linear_sum_assignment

        B, N, C = pd_scores.shape
        max_gt = gt_bboxes.shape[1]

        target_labels = torch.full((B, N), self.num_classes, dtype=torch.long, device=pd_scores.device)
        target_bboxes = torch.zeros((B, N, 4), dtype=torch.float32, device=pd_bboxes.device)
        target_scores = torch.zeros((B, N, C), dtype=torch.float32, device=pd_scores.device)
        fg_mask = torch.zeros((B, N), dtype=torch.bool, device=pd_scores.device)
        target_gt_idx = torch.full((B, N), -1, dtype=torch.long, device=pd_scores.device)

        if max_gt == 0 or not mask_gt.bool().any():
            return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

        # Sigmoid prob for matching cost
        probs = pd_scores.sigmoid() if pd_scores.min() < 0.0 or pd_scores.max() > 1.0 else pd_scores

        for b in range(B):
            n_gt = mask_gt[b].reshape(-1).sum().item()
            if n_gt == 0:
                continue

            b_gt_boxes = gt_bboxes[b, :int(n_gt)]  # (M, 4)
            b_gt_labels = gt_labels[b, :int(n_gt)].long().reshape(-1)  # (M,)
            b_pd_boxes = pd_bboxes[b]  # (N, 4)
            b_pd_probs = probs[b]  # (N, C)

            # 1. Classification Cost (Focal cost: -prob for true class)
            # For each GT box m, cost against each anchor n is -prob[n, label_m]
            cls_cost = -b_pd_probs[:, b_gt_labels].T  # (M, N)

            # 2. L1 Distance Cost
            l1_cost = torch.cdist(b_gt_boxes, b_pd_boxes, p=1) / 1000.0  # (M, N) normalized scale

            # 3. IoU Cost (1.0 - IoU)
            pairwise_iou = box_iou_2d(b_gt_boxes, b_pd_boxes, eps=self.eps)  # (M, N)
            giou_cost = 1.0 - pairwise_iou

            # Total cost matrix (M, N)
            total_cost = (self.cost_cls * cls_cost + 
                          self.cost_l1 * l1_cost + 
                          self.cost_giou * giou_cost)

            # Scipy Hungarian assignment
            cost_np = total_cost.detach().cpu().numpy()
            gt_ind, pd_ind = linear_sum_assignment(cost_np)

            if len(pd_ind) > 0:
                pd_ind_tensor = torch.as_tensor(pd_ind, dtype=torch.long, device=pd_scores.device)
                gt_ind_tensor = torch.as_tensor(gt_ind, dtype=torch.long, device=pd_scores.device)

                fg_mask[b, pd_ind_tensor] = True
                target_labels[b, pd_ind_tensor] = b_gt_labels[gt_ind_tensor]
                target_bboxes[b, pd_ind_tensor] = b_gt_boxes[gt_ind_tensor]
                target_gt_idx[b, pd_ind_tensor] = gt_ind_tensor

                for idx, (g_i, p_i) in enumerate(zip(gt_ind, pd_ind)):
                    c_id = b_gt_labels[g_i].item()
                    # Soft score based on matched IoU
                    target_scores[b, p_i, c_id] = pairwise_iou[g_i, p_i].clamp(0.0, 1.0)

        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx

