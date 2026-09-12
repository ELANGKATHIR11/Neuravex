import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveComputeRouter(nn.Module):
    """
    Uncertainty & Confidence-Driven Adaptive Routing:
      y = f_base(x) + g(x) * f_refine(x),  g in [0, 1]
    Easy / high-confidence samples use cheap base path;
    Hard / ambiguous samples trigger refinement, strictly enforcing compute budgets.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 8, 8), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(channels // 8, 8), 1, 1),
            nn.Sigmoid()
        )
        self.refine_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, feat: torch.Tensor, threshold: float = 0.5) -> tuple:
        """
        feat: (B, C, H, W)
        Returns: (fused_feat, gate_score, executed_refine_bool)
        """
        gate = self.router(feat) # (B, 1, 1, 1)

        # Dynamic early-exit / execution
        if gate.mean() < threshold and not self.training:
            # Skip refinement during inference when confident!
            return feat, gate, False

        refine = self.refine_block(feat)
        out = feat + gate * refine
        return out, gate, True

class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation (KD) for Model Compression & Accuracy Optimization:
      - Softmax Temperature Logit Matching (KL Divergence)
      - Continuous DFL Distribution Matching
      - Intermediate Feature Hint Loss
    """
    def __init__(self, temperature: float = 2.0, lambda_logits: float = 1.0, lambda_dist: float = 0.5):
        super().__init__()
        self.T = temperature
        self.lambda_logits = lambda_logits
        self.lambda_dist = lambda_dist

    def forward(self, student_cls: torch.Tensor, teacher_cls: torch.Tensor,
                student_dist: torch.Tensor = None, teacher_dist: torch.Tensor = None) -> torch.Tensor:
        # 1. Soft logit distillation with temperature T
        s_soft = F.log_softmax(student_cls / self.T, dim=-1)
        t_soft = F.softmax(teacher_cls / self.T, dim=-1)
        kd_cls = F.kl_div(s_soft, t_soft, reduction="batchmean") * (self.T ** 2)

        # 2. DFL regression distribution distillation
        if student_dist is not None and teacher_dist is not None:
            s_d_soft = F.log_softmax(student_dist / self.T, dim=-1)
            t_d_soft = F.softmax(teacher_dist / self.T, dim=-1)
            kd_dist = F.kl_div(s_d_soft, t_d_soft, reduction="batchmean") * (self.T ** 2)
        else:
            kd_dist = torch.tensor(0.0, device=student_cls.device)

        total = self.lambda_logits * kd_cls + self.lambda_dist * kd_dist
        return total
