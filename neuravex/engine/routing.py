"""
Dynamic Routing Subsystem for Neuravex:
1. Dynamic Resolution Router (320 -> 512 -> 640+ escalation)
2. Dynamic Token Router (Spatial importance gating)
3. Dynamic Layer / Early-Exit Router (Confidence-based depth pruning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


class TokenImportanceRouter(nn.Module):
    """
    Dynamic Token Router:
    Estimates spatial token importance across feature maps, routes top-k or salient
    tokens through deep processing, and allows skipping or cheap-pooling low-value background tokens.
    """
    def __init__(self, in_channels: int, reduction: int = 4, keep_ratio: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.keep_ratio = keep_ratio
        
        # Ultra-lightweight 1x1 conv to estimate spatial importance score [0, 1]
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, max(4, in_channels // reduction), kernel_size=1, bias=False),
            nn.BatchNorm2d(max(4, in_channels // reduction)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, in_channels // reduction), 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, force_full: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x: (B, C, H, W) feature map
        Returns:
            gated_x: (B, C, H, W) masked/gated feature map
            stats: dict containing token retention ratio and mask
        """
        if force_full or not self.training and self.keep_ratio >= 1.0:
            return x, {"retention_ratio": torch.tensor(1.0, device=x.device), "mask": torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)}

        B, C, H, W = x.shape
        importance = self.gate(x)  # (B, 1, H, W)
        
        # Hard or soft top-k gating based on keep_ratio
        flat_importance = importance.view(B, -1)
        k = max(1, int(flat_importance.shape[1] * self.keep_ratio))
        topk_vals, _ = torch.topk(flat_importance, k, dim=-1)
        min_topk = topk_vals[:, -1:].view(B, 1, 1, 1)

        mask = (importance >= min_topk).float()
        # Soft-straight-through for gradient flow during training
        if self.training:
            mask = mask.detach() - importance.detach() + importance

        # Gated features: salient tokens preserved, non-salient tokens attenuated
        gated_x = x * mask
        actual_retention = mask.sum() / mask.numel()

        return gated_x, {
            "retention_ratio": actual_retention,
            "mask": mask,
            "importance": importance
        }


class EarlyExitLayerRouter(nn.Module):
    """
    Dynamic Layer / Depth Router:
    Monitors early-stage classification confidence and entropy.
    If the sample is unambiguous (high confidence, low entropy), early exit is triggered,
    bypassing deeper neck and refinement stages.
    """
    def __init__(self, in_channels: int, num_classes: int, confidence_threshold: float = 0.85):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes
        
        # Lightweight probe head on early neck features
        self.early_probe = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, feat: torch.Tensor) -> Tuple[bool, torch.Tensor, float]:
        """
        feat: (B, C, H, W) intermediate feature tensor (e.g. from P4)
        Returns:
            should_exit: bool indicating if early exit criteria met
            probs: (B, num_classes) probability distribution
            mean_confidence: float max probability
        """
        logits = self.early_probe(feat)
        probs = F.softmax(logits, dim=-1)
        max_prob, _ = torch.max(probs, dim=-1)
        mean_conf = float(max_prob.mean().item())

        # If mean confidence exceeds threshold and not training, trigger early exit
        should_exit = (not self.training) and (mean_conf >= self.confidence_threshold)
        return should_exit, probs, mean_conf


class ResolutionRouter:
    """
    Dynamic Resolution Router:
    Implements adaptive multi-scale evaluation:
    1. Runs fast pass at lower resolution (e.g., 320x320).
    2. Analyzes detection boxes and confidence:
       - If no objects detected, or all detected objects are large (> min_box_ratio) and confident (> conf_thresh),
         returns low-res predictions directly (massive speedup on CPU).
       - If small objects or low-confidence/uncertain detections exist, escalates only ambiguous ROIs
         or full frame to higher resolution (512x512 or 640x640) and fuses predictions.
    """
    def __init__(
        self,
        low_res: int = 320,
        mid_res: int = 512,
        high_res: int = 640,
        conf_escalate_thresh: float = 0.55,
        small_box_area_ratio: float = 0.04
    ):
        self.low_res = low_res
        self.mid_res = mid_res
        self.high_res = high_res
        self.conf_escalate_thresh = conf_escalate_thresh
        self.small_box_area_ratio = small_box_area_ratio

    def should_escalate(self, boxes: torch.Tensor, scores: torch.Tensor, img_hw: Tuple[int, int]) -> Tuple[bool, str]:
        """
        boxes: (N, 4) in [x1, y1, x2, y2] format
        scores: (N,) confidence scores
        img_hw: (H, W) current image resolution
        Returns:
            escalate: bool
            reason: str description
        """
        if boxes is None or len(boxes) == 0:
            return False, "no_candidates"

        # Check for uncertain detections
        uncertain_mask = (scores >= 0.20) & (scores < self.conf_escalate_thresh)
        if uncertain_mask.any():
            return True, "uncertain_predictions"

        # Check for small objects needing higher resolution
        H, W = img_hw
        img_area = float(H * W)
        box_areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        area_ratios = box_areas / max(1.0, img_area)
        
        small_mask = (area_ratios > 0) & (area_ratios < self.small_box_area_ratio) & (scores >= 0.25)
        if small_mask.any():
            return True, "small_objects_detected"

        return False, "sufficient_confidence"

    def run_adaptive_inference(self, model: nn.Module, x: torch.Tensor, tasks: tuple = ("det",), **kwargs) -> Dict[str, any]:
        """
        Executes adaptive 320 -> 512/640 resolution routing.
        """
        B, C, orig_H, orig_W = x.shape
        
        # Step 1: Low-resolution pass (320x320)
        x_low = F.interpolate(x, size=(self.low_res, self.low_res), mode="bilinear", align_corners=False)
        out_low = model(x_low, tasks=tasks, **kwargs)
        
        # Extract predictions for escalation analysis
        pred_boxes = out_low.get("pred_boxes", None)
        class_logits = out_low.get("class_logits", None)

        if pred_boxes is not None and class_logits is not None:
            probs = torch.sigmoid(class_logits)
            max_scores, _ = probs.max(dim=-1)
            # Evaluate escalation criteria on first batch item
            sample_boxes = pred_boxes[0]
            sample_scores = max_scores[0]
            
            escalate, reason = self.should_escalate(sample_boxes, sample_scores, (self.low_res, self.low_res))
            
            if escalate:
                # Step 2: Escalate to mid/high resolution
                target_res = self.high_res if orig_H >= 640 else self.mid_res
                x_high = F.interpolate(x, size=(target_res, target_res), mode="bilinear", align_corners=False)
                out_high = model(x_high, tasks=tasks, **kwargs)
                out_high["resolution_routing"] = {
                    "escalated": True,
                    "reason": reason,
                    "initial_res": self.low_res,
                    "final_res": target_res,
                    "savings_achieved": False
                }
                return out_high

        out_low["resolution_routing"] = {
            "escalated": False,
            "reason": "sufficient_confidence",
            "initial_res": self.low_res,
            "final_res": self.low_res,
            "savings_achieved": True
        }
        return out_low
