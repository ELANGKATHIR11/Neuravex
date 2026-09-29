"""
Confidence and Uncertainty Engine for Neuravex:
Provides:
1. Temperature-scaled classification calibration.
2. Predictive entropy & energy estimation.
3. Bounding box edge variance (DFL dispersion uncertainty).
4. Composite uncertainty metrics driving dynamic routing and escalation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional


class ConfidenceUncertaintyEngine:
    """
    Calibrated Uncertainty & Decision Routing Engine:
    - Calibrates classification logits via temperature scaling.
    - Quantifies spatial bounding box uncertainty from DFL bin dispersion.
    - Calculates epistemic/aleatoric uncertainty scores per detection.
    - Emits escalation triggers when prediction ambiguity exceeds budget thresholds.
    """
    def __init__(
        self,
        temperature: float = 1.15,
        uncertainty_escalation_threshold: float = 0.45,
        reg_max: int = 16
    ):
        self.temperature = temperature
        self.uncertainty_escalation_threshold = uncertainty_escalation_threshold
        self.reg_max = reg_max

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies temperature scaling to raw class logits."""
        return logits / self.temperature

    def compute_classification_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Computes binary entropy per class across detection anchors:
        H(p) = - [p*log(p) + (1-p)*log(1-p)]
        """
        eps = 1e-7
        p = probs.clamp(eps, 1.0 - eps)
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        return entropy.mean(dim=-1)  # (B, N)

    def compute_box_dispersion_variance(self, pred_box_dist: torch.Tensor) -> torch.Tensor:
        """
        Measures dispersion variance across the DFL distribution bins for each of 4 edges (L, T, R, B).
        High variance indicates high spatial boundary ambiguity.
        pred_box_dist: (B, N, 4 * reg_max)
        """
        B, N, _ = pred_box_dist.shape
        # Reshape to (B, N, 4, reg_max)
        dist = pred_box_dist.view(B, N, 4, self.reg_max)
        probs = F.softmax(dist, dim=-1)  # (B, N, 4, reg_max)

        bins = torch.arange(self.reg_max, device=pred_box_dist.device, dtype=pred_box_dist.dtype)
        # Expected value
        mean_val = (probs * bins).sum(dim=-1, keepdim=True)  # (B, N, 4, 1)
        # Variance: E[(X - E[X])^2]
        variance = (probs * (bins - mean_val) ** 2).sum(dim=-1)  # (B, N, 4)
        mean_edge_variance = variance.mean(dim=-1)  # (B, N)
        return mean_edge_variance

    def evaluate_uncertainty(
        self,
        class_logits: torch.Tensor,
        pred_box_dist: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computes calibrated probabilities, entropy, spatial variance,
        and composite uncertainty scores for all candidate detections.
        """
        cal_logits = self.calibrate_logits(class_logits)
        cal_probs = torch.sigmoid(cal_logits)
        max_conf, pred_labels = torch.max(cal_probs, dim=-1)

        cls_entropy = self.compute_classification_entropy(cal_probs)
        box_variance = self.compute_box_dispersion_variance(pred_box_dist)

        # Normalize metrics to [0, 1] range
        norm_entropy = (cls_entropy / math.log(2.0)).clamp(0.0, 1.0)
        norm_box_var = (box_variance / float((self.reg_max ** 2) / 12.0)).clamp(0.0, 1.0)

        # Composite uncertainty score
        composite_uncertainty = 0.5 * norm_entropy + 0.5 * norm_box_var

        # Escalation mask: candidates with moderate confidence but high ambiguity
        escalation_mask = (max_conf >= 0.25) & (composite_uncertainty > self.uncertainty_escalation_threshold)

        return {
            "calibrated_probs": cal_probs,
            "max_confidence": max_conf,
            "predicted_labels": pred_labels,
            "classification_entropy": cls_entropy,
            "box_edge_variance": box_variance,
            "composite_uncertainty": composite_uncertainty,
            "escalation_mask": escalation_mask,
            "requires_escalation": escalation_mask.any().item()
        }
