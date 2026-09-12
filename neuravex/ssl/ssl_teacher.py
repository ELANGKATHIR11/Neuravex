import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class EMATeacher:
    """
    Momentum-updated Exponential Moving Average (EMA) teacher model.
    Teacher network is strictly no-grad and updated exclusively via:
      theta_teacher <- alpha * theta_teacher + (1 - alpha) * theta_student
    """
    def __init__(self, model: nn.Module, alpha: float = 0.999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.alpha = alpha

    @torch.no_grad()
    def update(self, student_model: nn.Module):
        """Updates teacher parameters with student parameters."""
        for p_t, p_s in zip(self.model.parameters(), student_model.parameters()):
            p_t.data.mul_(self.alpha).add_(p_s.data, alpha=1.0 - self.alpha)

        for b_t, b_s in zip(self.model.buffers(), student_model.buffers()):
            b_t.copy_(b_s)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, **kwargs):
        self.model.eval()
        return self.model(x, **kwargs)

class MultiViewSSLLoss(nn.Module):
    """
    Self-Supervised Distillation & Consistency Loss:
      L_ssl = lambda_f * L_f + lambda_m * L_ms + lambda_r * L_mask
    - L_f: Multi-scale feature distillation between teacher and student.
    - L_ms: Contrastive cosine alignment on P3, P4, P5 representations.
    - L_mask: Masked feature patch reconstruction.
    """
    def __init__(self, lambda_f: float = 1.0, lambda_m: float = 0.5, lambda_r: float = 0.5):
        super().__init__()
        self.lambda_f = lambda_f
        self.lambda_m = lambda_m
        self.lambda_r = lambda_r

    def forward(self, student_out: dict, teacher_out: dict, mask_indices: torch.Tensor = None) -> torch.Tensor:
        # 1. Feature distillation across detection anchor predictions
        s_logits = student_out["class_logits"]
        t_logits = teacher_out["class_logits"].detach()
        l_f = F.mse_loss(s_logits, t_logits)

        # 2. Multi-scale contrastive alignment on box distributions
        s_box = student_out["pred_box_dist"]
        t_box = teacher_out["pred_box_dist"].detach()
        # Cosine similarity over distribution bins
        cos_sim = F.cosine_similarity(s_box, t_box, dim=-1)
        l_ms = (1.0 - cos_sim).mean()

        # 3. Masked patch reconstruction
        if mask_indices is not None and "pred_xyz" in student_out and "pred_xyz" in teacher_out:
            s_xyz = student_out["pred_xyz"]
            t_xyz = teacher_out["pred_xyz"].detach()
            l_mask = F.l1_loss(s_xyz[mask_indices], t_xyz[mask_indices])
        else:
            l_mask = torch.tensor(0.0, device=s_logits.device)

        loss = self.lambda_f * l_f + self.lambda_m * l_ms + self.lambda_r * l_mask
        return loss
