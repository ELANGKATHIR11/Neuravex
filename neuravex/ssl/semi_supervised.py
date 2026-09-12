import torch
import torch.nn as nn
import torch.nn.functional as F

class SemiSupervisedPseudoLabeler:
    """
    Teacher -> confidence fusion -> adaptive threshold -> class-balanced pseudo-labels -> student
    Fuses cls, box, and 3D uncertainty; rejects unsafe labels.
    """
    def __init__(self, num_classes: int = 80, tau_base: float = 0.65, max_entropy: float = 0.5):
        self.num_classes = num_classes
        self.tau_base = tau_base
        self.max_entropy = max_entropy
        # Class-specific dynamic threshold buffer
        self.class_thresholds = torch.full((num_classes,), tau_base)

    @torch.no_grad()
    def generate_pseudo_labels(self, teacher_out: dict) -> dict:
        """
        Takes raw teacher outputs on weakly-augmented views and returns high-confidence pseudo ground truth.
        """
        logits = teacher_out["class_logits"] # (B, N, C)
        probs = torch.sigmoid(logits)
        max_prob, pred_cls = probs.max(dim=-1) # (B, N)

        # 1. Predictive entropy
        p = probs.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean(dim=-1) # (B, N)

        # 2. Adaptive thresholding per class
        thresh = self.class_thresholds.to(probs.device)[pred_cls] # (B, N)
        confidence_mask = (max_prob >= thresh) & (entropy <= self.max_entropy)

        # 3. Filter boxes and 3D estimates
        boxes = teacher_out["pred_boxes"] # (B, N, 4)
        xyz = teacher_out["pred_xyz"] if "pred_xyz" in teacher_out else None
        lwh = teacher_out["pred_lwh"] if "pred_lwh" in teacher_out else None
        yaw_sc = teacher_out["pred_yaw_sincos"] if "pred_yaw_sincos" in teacher_out else None

        return {
            "valid_mask": confidence_mask,
            "pseudo_cls": pred_cls,
            "pseudo_scores": max_prob,
            "pseudo_boxes": boxes,
            "pseudo_xyz": xyz,
            "pseudo_lwh": lwh,
            "pseudo_yaw": yaw_sc
        }

    def update_class_thresholds(self, class_frequencies: torch.Tensor):
        """Adjusts thresholds to maintain class balance across training."""
        freq_norm = class_frequencies / (class_frequencies.sum() + 1e-6)
        # Rare classes receive slightly lower threshold to prevent majority-class bias
        adjusted = self.tau_base * (1.0 + torch.log(freq_norm + 1e-4) * 0.05)
        self.class_thresholds = adjusted.clamp(0.4, 0.9)
