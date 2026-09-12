import torch
import torch.nn as nn
import torch.nn.functional as F
from ..geometry.box_ops import bbox_ciou

class DetectionLoss(nn.Module):
    """
    Anchor-free multi-scale detection loss:
    1. Varifocal / BCE classification loss with soft targets from task-aligned assigner.
    2. CIoU bounding box regression loss on foreground anchors.
    3. Distribution Focal Loss (DFL) or L1 box offset regression.
    """
    def __init__(self, num_classes: int = 80, box_weight: float = 7.5, cls_weight: float = 1.0, dfl_weight: float = 1.5):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight

    def forward(self, pred_scores: torch.Tensor, pred_bboxes: torch.Tensor,
                target_scores: torch.Tensor, target_bboxes: torch.Tensor, fg_mask: torch.Tensor):
        """
        Args:
            pred_scores: (B, N, num_classes) predicted logits
            pred_bboxes: (B, N, 4) predicted xyxy boxes
            target_scores: (B, N, num_classes) aligned target scores
            target_bboxes: (B, N, 4) aligned target xyxy boxes
            fg_mask: (B, N) boolean foreground mask
        Returns:
            loss_total, loss_dict
        """
        num_pos = max(fg_mask.sum().item(), 1.0)

        # 1. Classification Loss (BCE with soft target scores)
        loss_cls = F.binary_cross_entropy_with_logits(
            pred_scores, target_scores, reduction="sum"
        ) / num_pos

        # 2. Box Regression Loss (CIoU) on positive anchors
        if fg_mask.any():
            pos_pred_boxes = pred_bboxes[fg_mask]
            pos_target_boxes = target_bboxes[fg_mask]

            ciou = bbox_ciou(pos_pred_boxes, pos_target_boxes)
            loss_box = (1.0 - ciou).sum() / num_pos

            # L1 box regression penalty for coordinate stability
            loss_l1 = F.l1_loss(pos_pred_boxes, pos_target_boxes, reduction="sum") / (num_pos * 4.0)
        else:
            loss_box = pred_bboxes.sum() * 0.0
            loss_l1 = pred_bboxes.sum() * 0.0

        total_loss = (self.cls_weight * loss_cls + 
                      self.box_weight * loss_box + 
                      self.dfl_weight * loss_l1)

        loss_dict = {
            "loss_cls": loss_cls.detach(),
            "loss_ciou": loss_box.detach(),
            "loss_box_l1": loss_l1.detach(),
            "loss_det": total_loss
        }
        return total_loss, loss_dict
