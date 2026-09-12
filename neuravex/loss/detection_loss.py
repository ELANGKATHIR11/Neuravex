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

            ciou = bbox_ciou(pos_pred_boxes, pos_target_boxes)
            loss_box = (1.0 - ciou).sum() / num_pos

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
            loss_box = pred_bboxes.sum() * 0.0
            loss_dfl = pred_dist.sum() * 0.0

        total_loss = (self.cls_weight * loss_cls + 
                      self.box_weight * loss_box + 
                      self.dfl_weight * loss_dfl)

        loss_dict = {
            "loss_cls": loss_cls.detach(),
            "loss_ciou": loss_box.detach(),
            "loss_dfl": loss_dfl.detach() if isinstance(loss_dfl, torch.Tensor) else loss_dfl,
            "loss_det": total_loss
        }
        return total_loss, loss_dict
