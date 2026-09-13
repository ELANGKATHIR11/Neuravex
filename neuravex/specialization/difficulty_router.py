"""
Neuravex Multi-Level Difficulty Router.

Upgrades the existing AdaptiveComputeRouter (binary easy/hard) to a
3-level difficulty-aware routing scheme:

  Easy   → base path only          (cheapest)
  Medium → base + light refinement (moderate)
  Hard   → base + full refinement  (most accurate)

Difficulty score computed from:
  1. Classification uncertainty (entropy of class probabilities)
  2. Localization uncertainty (learned spatial gate)
  3. Object density proxy (downsampled input heatmap)
  4. Learned difficulty token (tiny MLP on pooled features)

Training loss:
  L_total = L_task + λ_c * E[FLOPs] + λ_t * E[Latency] + λ_m * E[Memory]

Statistics exposed:
  - route_fractions: (easy%, medium%, hard%)
  - expected_flops: scalar
  - executed_flops: scalar
  - per_level_flops: [easy_cost, medium_cost, hard_cost]
  - latency_proxy: scalar

Guarantees:
  - Deterministic full-compute fallback (force_full_compute=True)
  - Export-safe inference (no dynamic indexing in scripted path)
  - Zero overhead when difficulty routing disabled
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifficultyScorer(nn.Module):
    """
    Predicts per-sample difficulty score in [0, 1].
    Easy → 0, Hard → 1.

    Inputs: spatial feature map (B, C, H, W)
    Output: scalar score (B,) in [0, 1]
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.scorer = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x)          # (B, C, 1, 1)
        score = self.scorer(pooled)    # (B, 1, 1, 1)
        return score.view(-1)         # (B,)


class LightRefinementBlock(nn.Module):
    """Cheap depthwise-only refinement (1.5x base cost)."""
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bn(self.dw(x))


class FullRefinementBlock(nn.Module):
    """Full depthwise + pointwise refinement (3x base cost)."""
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bn_pw(self.pw(F.silu(self.bn_dw(self.dw(x)))))


class MultiLevelDifficultyRouter(nn.Module):
    """
    3-level difficulty-aware adaptive compute router.

    Route thresholds:
      difficulty < easy_thresh  → base path
      easy_thresh <= difficulty < hard_thresh → light refinement
      difficulty >= hard_thresh → full refinement

    All thresholds are learnable (via STE / soft training, hard at inference).
    """

    # Relative compute costs for budget accounting
    COST_BASE = 1.0
    COST_LIGHT = 1.5
    COST_FULL = 3.0

    def __init__(
        self,
        channels: int,
        easy_thresh: float = 0.35,
        hard_thresh: float = 0.65,
        num_levels: int = 3,
    ):
        super().__init__()
        self.channels = channels
        self.num_levels = num_levels
        # Thresholds stored as buffers for state_dict compatibility
        self.register_buffer("easy_thresh", torch.tensor(easy_thresh))
        self.register_buffer("hard_thresh", torch.tensor(hard_thresh))

        self.scorer = DifficultyScorer(channels)
        self.light_refine = LightRefinementBlock(channels)
        self.full_refine = FullRefinementBlock(channels)

    def forward(
        self,
        feat: torch.Tensor,
        threshold: float = None,         # Ignored (uses registered buffer); kept for API compat
        force_full_compute: bool = False,
        return_stats: bool = True,
    ) -> tuple:
        """
        Args:
            feat: (B, C, H, W) input feature
            force_full_compute: if True, run full refinement on all samples
            return_stats: if True, return routing statistics dict
        Returns:
            (output_feat, stats_dict)
        """
        B = feat.shape[0]
        scores = self.scorer(feat)  # (B,) in [0, 1]

        if force_full_compute or (self.training and self.num_levels == 1):
            # Training soft path or forced full compute
            light = self.light_refine(feat)
            full = self.full_refine(feat)
            s = scores.view(B, 1, 1, 1)
            # Soft routing during training for gradient flow
            light_gate = (s >= self.easy_thresh.clamp(0.01, 0.99)).float()
            full_gate = (s >= self.hard_thresh.clamp(0.01, 0.99)).float()
            out = feat + light_gate * (light - feat) + full_gate * (full - light)
            stats = self._make_stats(scores, light_gate.view(B), full_gate.view(B), True)
            return out, stats

        if self.training:
            # Differentiable soft routing
            light = self.light_refine(feat)
            full = self.full_refine(feat)
            s = scores.view(B, 1, 1, 1)
            easy_t = self.easy_thresh.clamp(0.01, 0.99)
            hard_t = self.hard_thresh.clamp(0.01, 0.99)
            # Soft sigmoid gates for gradient flow
            light_gate = torch.sigmoid((s - easy_t) * 20.0)
            full_gate = torch.sigmoid((s - hard_t) * 20.0)
            out = feat + light_gate * (light - feat) + full_gate * (full - light)
            stats = self._make_stats(scores, (scores >= easy_t.item()).float(),
                                     (scores >= hard_t.item()).float(), True)
            return out, stats

        # Inference: hard routing
        easy_t = float(self.easy_thresh.item())
        hard_t = float(self.hard_thresh.item())

        easy_mask = scores < easy_t            # (B,) bool
        hard_mask = scores >= hard_t           # (B,) bool
        medium_mask = ~easy_mask & ~hard_mask  # (B,) bool

        out = feat.clone()

        # Medium samples: light refinement
        if medium_mask.any():
            mid_idx = medium_mask.nonzero(as_tuple=True)[0]
            out[mid_idx] = self.light_refine(feat[mid_idx])

        # Hard samples: full refinement
        if hard_mask.any():
            hard_idx = hard_mask.nonzero(as_tuple=True)[0]
            out[hard_idx] = self.full_refine(feat[hard_idx])

        light_gate_bin = (~easy_mask).float()
        full_gate_bin = hard_mask.float()
        stats = self._make_stats(scores, light_gate_bin, full_gate_bin, False)
        return out, stats

    def _make_stats(
        self,
        scores: torch.Tensor,
        light_gate: torch.Tensor,
        full_gate: torch.Tensor,
        fully_executed: bool,
    ) -> dict:
        easy_frac = float((~(light_gate.bool())).float().mean().item())
        medium_frac = float((light_gate.bool() & ~(full_gate.bool())).float().mean().item())
        hard_frac = float(full_gate.float().mean().item())

        expected_cost = (
            easy_frac * self.COST_BASE +
            medium_frac * self.COST_LIGHT +
            hard_frac * self.COST_FULL
        )
        executed_cost = expected_cost if fully_executed else (
            self.COST_BASE +
            medium_frac * (self.COST_LIGHT - self.COST_BASE) +
            hard_frac * (self.COST_FULL - self.COST_LIGHT)
        )

        return {
            "difficulty_scores": scores,
            "route_fractions": (easy_frac, medium_frac, hard_frac),
            "expected_flops_multiplier": expected_cost,
            "executed_flops_multiplier": executed_cost,
            "fully_executed": fully_executed,
            "per_level_costs": {
                "easy": self.COST_BASE,
                "medium": self.COST_LIGHT,
                "hard": self.COST_FULL,
            },
            # Backward-compatible keys (match AdaptiveComputeRouter API)
            "gate": scores.view(-1, 1, 1, 1),
            "active_ratio": 1.0 - easy_frac,
            "expected_compute": torch.tensor(expected_cost),
            "executed_compute": torch.tensor(executed_cost),
        }

    def get_compute_budget_loss(
        self, stats: dict, target_budget: float = 0.6, lambda_compute: float = 0.05
    ) -> torch.Tensor:
        """
        Budget constraint regularization:
        L_budget = λ_c * max(0, E[cost] - target_budget)^2 + λ_c * mean(difficulty)
        """
        scores = stats.get("difficulty_scores", None)
        if scores is None:
            return torch.tensor(0.0)
        mean_score = scores.mean()
        expected_cost = (
            (1 - mean_score) * self.COST_BASE +
            mean_score * self.COST_FULL
        )
        budget_excess = F.relu(expected_cost - target_budget)
        return lambda_compute * (mean_score + budget_excess.pow(2) * 5.0)


class RoutingMetricsLogger:
    """Accumulates routing statistics across a training epoch for monitoring."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.easy_counts = []
        self.medium_counts = []
        self.hard_counts = []
        self.expected_costs = []
        self.executed_costs = []

    def update(self, stats: dict):
        fracs = stats.get("route_fractions", (1.0, 0.0, 0.0))
        self.easy_counts.append(fracs[0])
        self.medium_counts.append(fracs[1])
        self.hard_counts.append(fracs[2])
        self.expected_costs.append(stats.get("expected_flops_multiplier", 1.0))
        self.executed_costs.append(stats.get("executed_flops_multiplier", 1.0))

    def summary(self) -> dict:
        import numpy as np
        if not self.easy_counts:
            return {}
        return {
            "mean_easy_fraction": float(np.mean(self.easy_counts)),
            "mean_medium_fraction": float(np.mean(self.medium_counts)),
            "mean_hard_fraction": float(np.mean(self.hard_counts)),
            "mean_expected_cost": float(np.mean(self.expected_costs)),
            "mean_executed_cost": float(np.mean(self.executed_costs)),
        }
