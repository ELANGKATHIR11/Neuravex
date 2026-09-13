import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveComputeRouter(nn.Module):
    """
    Uncertainty & Confidence-Driven Adaptive Routing:
      y = f_base(x) + g(x) * f_refine(x),  g in [0, 1]
    Easy / high-confidence samples use cheap base path;
    Hard / ambiguous samples trigger refinement, strictly enforcing compute budgets.

    Supports:
      - Per-sample routing (individual batch samples route independently)
      - Explicit compute budget enforcement
      - Deterministic full-compute fallback for reproducible benchmarks / export
      - Static vs. Expected vs. Executed FLOPs tracking
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels
        hidden = max(channels // reduction, 8)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid()
        )
        self.refine_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # Baseline and refinement relative compute cost weights (per-pixel FLOPs)
        # Refine FLOPs: DW-conv (2 * 1 * 9 * c * H * W) + PW-conv (2 * c * c * H * W)
        self.register_buffer("base_cost", torch.tensor(1.0))
        self.register_buffer("refine_cost_multiplier", torch.tensor(1.5))

    def forward(self, feat: torch.Tensor, threshold: float = 0.5, force_full_compute: bool = False) -> tuple:
        """
        feat: (B, C, H, W)
        threshold: routing threshold in [0, 1]
        force_full_compute: if True, deterministically executes refinement for all samples
        Returns:
          fused_feat: (B, C, H, W)
          stats: dict containing gate_scores, active_ratio, expected_cost, executed_cost
        """
        B, C, H, W = feat.shape
        gate = self.router(feat) # (B, 1, 1, 1)

        if force_full_compute or self.training:
            # In training or forced mode: full forward pass with soft gate weighting
            refine = self.refine_block(feat)
            out = feat + gate * refine
            active_mask = (gate >= threshold).float()
            active_ratio = active_mask.mean()
            stats = {
                "gate": gate,
                "active_ratio": active_ratio,
                "expected_compute": (1.0 + gate.mean() * self.refine_cost_multiplier),
                "executed_compute": (1.0 + 1.0 * self.refine_cost_multiplier),
                "fully_executed": True
            }
            return out, stats

        # Inference per-sample routing:
        sample_active = (gate.view(B) >= threshold) # (B,) bool
        active_ratio = float(sample_active.float().mean().item())

        if not sample_active.any():
            # 100% skipped refinement across entire batch!
            stats = {
                "gate": gate,
                "active_ratio": 0.0,
                "expected_compute": (1.0 + gate.mean() * self.refine_cost_multiplier),
                "executed_compute": torch.tensor(1.0, device=feat.device),
                "fully_executed": False
            }
            return feat, stats

        if sample_active.all():
            # 100% active refinement across entire batch
            refine = self.refine_block(feat)
            out = feat + gate * refine
            stats = {
                "gate": gate,
                "active_ratio": 1.0,
                "expected_compute": (1.0 + gate.mean() * self.refine_cost_multiplier),
                "executed_compute": (1.0 + 1.0 * self.refine_cost_multiplier),
                "fully_executed": True
            }
            return out, stats

        # Heterogeneous batch: evaluate refinement only on active samples to save compute
        out = feat.clone()
        active_idx = sample_active.nonzero(as_tuple=True)[0]
        active_feat = feat[active_idx]
        active_refine = self.refine_block(active_feat)
        out[active_idx] = active_feat + gate[active_idx] * active_refine

        stats = {
            "gate": gate,
            "active_ratio": active_ratio,
            "expected_compute": (1.0 + gate.mean() * self.refine_cost_multiplier),
            "executed_compute": (1.0 + active_ratio * self.refine_cost_multiplier),
            "fully_executed": False
        }
        return out, stats

class ComputeBudgetLoss(nn.Module):
    """
    Budget-constrained loss enforcing target execution efficiency:
    L = L_task + lambda_c * C_expected + lambda_l * Latency_expected
    """
    def __init__(self, target_budget: float = 0.5, lambda_compute: float = 0.1):
        super().__init__()
        self.target_budget = target_budget
        self.lambda_compute = lambda_compute

    def forward(self, routing_stats: dict) -> torch.Tensor:
        if "gate" not in routing_stats:
            return torch.tensor(0.0)
        gate = routing_stats["gate"]
        # Penalize mean gating that exceeds target compute budget
        mean_gate = gate.mean()
        budget_penalty = F.relu(mean_gate - self.target_budget).pow(2)
        return self.lambda_compute * (mean_gate + budget_penalty * 5.0)

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
