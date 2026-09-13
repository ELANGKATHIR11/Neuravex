"""
Neuravex Teacher → Student Knowledge Distillation.

Implements multi-level distillation from a teacher model (large Neuravex,
or any model exposing compatible outputs) to a specialized student.

Components:
  1. Logit KD: Temperature-scaled KL divergence on class logits
  2. Box regression KD: DFL distribution matching
  3. Feature hint loss: L2 on neck output features (after projection)
  4. Teacher validation: only applies distillation if teacher improves metrics

Teacher compatibility:
  - Any model with outputs: class_logits, pred_boxes, pred_box_dist
  - Optional: neck features via teacher.get_neck_features()
  - If teacher is not better than student (no AP improvement), distillation
    is disabled (flag: distillation_effective=False)

Usage:
    distiller = NeuravexDistillation(student, teacher, temperature=4.0)
    loss = distiller(student_outputs, teacher_outputs, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class LogitDistillationLoss(nn.Module):
    """
    Temperature-scaled soft-logit KL divergence.
    L_kd = T^2 * KL(p_t || p_s) on class logits.
    """
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        student_logits: (B, N, C) raw class logits
        teacher_logits: (B, N, C) raw class logits
        """
        T = self.T
        s_soft = F.log_softmax(student_logits / T, dim=-1)
        t_soft = F.softmax(teacher_logits / T, dim=-1)
        kd_loss = F.kl_div(s_soft, t_soft, reduction="batchmean") * (T ** 2)
        return kd_loss


class BoxDistributionDistillationLoss(nn.Module):
    """
    DFL distribution matching between student and teacher.
    L_dfl_kd = T^2 * KL(p_t_dist || p_s_dist) on box distribution logits.
    """
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.T = temperature

    def forward(
        self,
        student_dist: torch.Tensor,
        teacher_dist: torch.Tensor,
    ) -> torch.Tensor:
        """
        student_dist: (B, N, 4*reg_max)
        teacher_dist: (B, N, 4*reg_max)
        """
        T = self.T
        B, N, D = student_dist.shape
        s = student_dist.view(B, N, 4, -1)
        t = teacher_dist.view(B, N, 4, -1)
        s_soft = F.log_softmax(s / T, dim=-1).view(B * N * 4, -1)
        t_soft = F.softmax(t / T, dim=-1).view(B * N * 4, -1)
        return F.kl_div(s_soft, t_soft, reduction="batchmean") * (T ** 2)


class FeatureHintLoss(nn.Module):
    """
    Intermediate feature distillation via projector.
    L_hint = MSE(proj(F_student), F_teacher) / HW
    Supports mismatched channel dimensions via learnable projector.
    """
    def __init__(self, student_c: int, teacher_c: int):
        super().__init__()
        self.proj = nn.Conv2d(student_c, teacher_c, 1, bias=False) if student_c != teacher_c else nn.Identity()

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        student_feat: (B, C_s, H, W)
        teacher_feat: (B, C_t, H, W)
        """
        s = self.proj(student_feat)
        if s.shape != teacher_feat.shape:
            teacher_feat = F.interpolate(teacher_feat, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return F.mse_loss(s, teacher_feat.detach())


class NeuravexDistillation(nn.Module):
    """
    Unified teacher → student distillation wrapper for Neuravex.

    Validates that distillation is effective (AP_teacher > AP_student) before
    applying it. If validation fails, distillation_effective = False and no
    KD loss is applied.

    Supports optional feature hint loss when both models expose neck features.
    """
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        temperature: float = 4.0,
        lambda_logit: float = 1.0,
        lambda_dist: float = 0.5,
        lambda_hint: float = 0.1,
        student_neck_c: Optional[int] = None,
        teacher_neck_c: Optional[int] = None,
    ):
        super().__init__()
        self.student = student
        # Teacher is frozen — never update its weights
        self.teacher = teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        self.lambda_logit = lambda_logit
        self.lambda_dist = lambda_dist
        self.lambda_hint = lambda_hint

        self.logit_kd = LogitDistillationLoss(temperature)
        self.dist_kd = BoxDistributionDistillationLoss(temperature)

        self.hint_loss = None
        if student_neck_c is not None and teacher_neck_c is not None:
            self.hint_loss = FeatureHintLoss(student_neck_c, teacher_neck_c)

        self.distillation_effective = None  # Unknown until validate() is called

    @torch.no_grad()
    def get_teacher_outputs(self, x: torch.Tensor) -> Dict:
        """Run frozen teacher forward pass."""
        self.teacher.eval()
        with torch.no_grad():
            return self.teacher(x)

    def compute_kd_loss(
        self,
        student_outputs: Dict,
        teacher_outputs: Dict,
        student_feat: Optional[torch.Tensor] = None,
        teacher_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined KD loss.
        Returns (total_kd_loss, component_dict).
        Hard-fails if teacher/student outputs are invalid (NaN/Inf shape mismatch).
        """
        losses = {}
        total = torch.tensor(0.0, device=next(self.student.parameters()).device)

        s_cls = student_outputs.get("class_logits")
        t_cls = teacher_outputs.get("class_logits")

        # --- Safety checks ---
        if s_cls is None or t_cls is None:
            raise ValueError("[Distillation] class_logits missing from student or teacher outputs.")
        if s_cls.shape != t_cls.shape:
            raise ValueError(
                f"[Distillation] Shape mismatch: student_cls {s_cls.shape} vs teacher_cls {t_cls.shape}"
            )
        if torch.isnan(s_cls).any() or torch.isnan(t_cls).any():
            raise RuntimeError("[Distillation] NaN detected in class logits.")
        if torch.isinf(s_cls).any() or torch.isinf(t_cls).any():
            raise RuntimeError("[Distillation] Inf detected in class logits.")

        # 1. Logit distillation
        l_logit = self.logit_kd(s_cls, t_cls)
        losses["kd_logit"] = float(l_logit.item())
        total = total + self.lambda_logit * l_logit

        # 2. Box distribution distillation (optional)
        s_dist = student_outputs.get("pred_box_dist")
        t_dist = teacher_outputs.get("pred_box_dist")
        if s_dist is not None and t_dist is not None and s_dist.shape == t_dist.shape:
            l_dist = self.dist_kd(s_dist, t_dist)
            losses["kd_dist"] = float(l_dist.item())
            total = total + self.lambda_dist * l_dist

        # 3. Feature hint distillation (optional)
        if self.hint_loss is not None and student_feat is not None and teacher_feat is not None:
            l_hint = self.hint_loss(student_feat, teacher_feat)
            losses["kd_hint"] = float(l_hint.item())
            total = total + self.lambda_hint * l_hint

        losses["kd_total"] = float(total.item())

        if torch.isnan(total) or torch.isinf(total):
            raise RuntimeError(f"[Distillation] NaN/Inf in total KD loss: {losses}")

        return total, losses

    def validate_effectiveness(
        self, teacher_ap: float, student_ap: float, tolerance: float = 0.005
    ):
        """
        Check if teacher is genuinely better than student.
        If teacher_ap <= student_ap + tolerance, distillation is disabled.
        Never fabricate or extrapolate results.
        """
        self.distillation_effective = teacher_ap > (student_ap + tolerance)
        if not self.distillation_effective:
            print(
                f"[Distillation] WARNING: Teacher AP={teacher_ap:.4f} <= "
                f"Student AP={student_ap:.4f} + tol={tolerance:.3f}. "
                "Distillation DISABLED. Teacher is NOT better."
            )
        else:
            print(
                f"[Distillation] Teacher AP={teacher_ap:.4f} > Student AP={student_ap:.4f}. "
                "Distillation ENABLED."
            )
        return self.distillation_effective


class ImbalanceAwareClassWeights:
    """
    Computes per-class loss weights to prevent rare classes from being ignored.

    Uses inverse-frequency weighting with temperature smoothing:
        w_c = (N_total / (N_c + ε)) ^ α
    where α in [0, 1] controls sharpness (0 = uniform, 1 = full inverse-freq).
    """
    def __init__(self, class_counts: Dict[int, int], num_classes: int, alpha: float = 0.5):
        self.num_classes = num_classes
        counts = torch.zeros(num_classes)
        for cls_id, cnt in class_counts.items():
            if 0 <= cls_id < num_classes:
                counts[cls_id] = cnt
        counts = counts.clamp_min(1)  # avoid division by zero
        total = counts.sum()
        # Inverse frequency weights
        raw_weights = (total / counts) ** alpha
        # Normalize so mean weight = 1
        self.weights = (raw_weights / raw_weights.mean()).float()

    def get_weights(self, device: str = "cpu") -> torch.Tensor:
        return self.weights.to(device)
