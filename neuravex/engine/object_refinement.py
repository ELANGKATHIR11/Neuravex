"""
Object-Centric Refinement Subsystem:
Two-stage compute allocation:
1. Fast global detection pass to discover object candidate bounding boxes.
2. Focused ROI Align & refinement head executed ONLY on candidate object patches,
   achieving high precision boundary alignment and contour IoU without full-frame dense cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from typing import Dict, List, Optional, Tuple, Any


class ObjectCentricRefinementHead(nn.Module):
    """
    Lightweight ROI refinement head:
    Takes ROI-aligned feature representations (e.g. 7x7) of detected candidate objects
    and predicts refined boundary deltas: [delta_x1, delta_y1, delta_x2, delta_y2] and contour quality.
    """
    def __init__(self, in_channels: int = 64, roi_size: int = 7):
        super().__init__()
        self.roi_size = roi_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        # Predicts relative bbox adjustment [-0.2, 0.2]
        self.delta_reg = nn.Linear(in_channels, 4)
        # Predicts contour boundary quality score [0, 1]
        self.contour_quality = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, roi_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        roi_feats: (K, C, 7, 7) feature tensor for K candidate ROIs
        Returns:
            deltas: (K, 4) relative coordinate offsets
            quality: (K, 1) contour quality confidence
        """
        h = self.conv(roi_feats)
        deltas = 0.2 * torch.tanh(self.delta_reg(h))
        quality = self.contour_quality(h)
        return deltas, quality


class ObjectCentricRefiner(nn.Module):
    """
    Object-Centric Pipeline:
    - Filters top-K candidate detection boxes above confidence threshold.
    - Extracts ROI aligned features from neck pyramid.
    - Refines bounding boxes and predicts contour stability score only on candidates.
    """
    def __init__(self, in_channels: int = 64, conf_thresh: float = 0.25, max_rois: int = 50):
        super().__init__()
        self.conf_thresh = conf_thresh
        self.max_rois = max_rois
        self.head = ObjectCentricRefinementHead(in_channels=in_channels, roi_size=7)

    def refine_detections(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        spatial_scale: float = 1.0 / 8.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        features: (B, C, H, W) neck feature map (e.g. Q3 at stride 8)
        boxes: (N, 4) in [x1, y1, x2, y2] format
        scores: (N,) confidence scores
        spatial_scale: scale factor from image coords to feature coords
        Returns:
            refined_boxes: (M, 4)
            filtered_scores: (M,)
            qualities: (M, 1)
        """
        if boxes is None or len(boxes) == 0:
            return boxes, scores, torch.empty((0, 1), device=features.device)

        # 1. Filter candidates above confidence threshold
        mask = scores >= self.conf_thresh
        if not mask.any():
            return torch.empty((0, 4), device=boxes.device), torch.empty((0,), device=scores.device), torch.empty((0, 1), device=boxes.device)

        cand_boxes = boxes[mask]
        cand_scores = scores[mask]

        if len(cand_boxes) > self.max_rois:
            top_scores, top_idx = torch.topk(cand_scores, self.max_rois)
            cand_boxes = cand_boxes[top_idx]
            cand_scores = top_scores

        # Format ROIs for roi_align: list of (N, 4) boxes
        roi_list = [cand_boxes]
        roi_feats = ops.roi_align(
            features,
            roi_list,
            output_size=(7, 7),
            spatial_scale=spatial_scale,
            sampling_ratio=2,
            aligned=True
        )

        deltas, qualities = self.head(roi_feats)

        # Apply deltas to refine bounding boxes:
        # width = x2 - x1, height = y2 - y1
        w = (cand_boxes[:, 2] - cand_boxes[:, 0]).clamp(min=1.0)
        h = (cand_boxes[:, 3] - cand_boxes[:, 1]).clamp(min=1.0)

        ref_x1 = cand_boxes[:, 0] + deltas[:, 0] * w
        ref_y1 = cand_boxes[:, 1] + deltas[:, 1] * h
        ref_x2 = cand_boxes[:, 2] + deltas[:, 2] * w
        ref_y2 = cand_boxes[:, 3] + deltas[:, 3] * h

        refined_boxes = torch.stack([ref_x1, ref_y1, ref_x2, ref_y2], dim=-1)
        return refined_boxes, cand_scores, qualities
