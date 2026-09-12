import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaControllerBandit(nn.Module):
    """
    RL / Bandit Meta-Controller for Hyperparameters & Dynamic Resource Allocation.
    CRITICAL RULE: NEVER updates detector neural network weights directly.
    Only predicts bounded, smoothed policies for:
      - Augmentation intensity: p_flip, p_scale in [0.1, 0.9]
      - Task loss weights: [w_det, w_seg, w_depth, w_3d, w_ssl] in [0.2, 3.0]
      - Inference routing threshold: tau in [0.2, 0.8]

    State: [losses, grad_norm, validation_metrics, uncertainty, compute_flops]
    Reward: R = Quality_Gain - lambda * Latency - mu * Memory - nu * Instability
    """
    def __init__(self, state_dim: int = 10, action_dim: int = 7):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Lightweight policy network (1 hidden layer, tiny footprint)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Sigmoid() # Strictly bounded in [0, 1]
        )

        # Baseline value estimate for policy gradient variance reduction
        self.value_head = nn.Linear(32, 1)

        self.register_buffer("running_mean_reward", torch.tensor(0.0))
        self.register_buffer("step_count", torch.tensor(0.0))

    def forward(self, state: torch.Tensor) -> dict:
        """
        state: (B, state_dim)
        Returns: smoothed, bounded actions
        """
        raw_action = self.policy_net(state) # in [0, 1]

        # Action mapping with strict physical boundaries:
        aug_prob = 0.1 + raw_action[..., 0] * 0.8        # [0.1, 0.9]
        w_det = 0.5 + raw_action[..., 1] * 2.0           # [0.5, 2.5]
        w_seg = 0.2 + raw_action[..., 2] * 1.5           # [0.2, 1.7]
        w_depth = 0.2 + raw_action[..., 3] * 1.5         # [0.2, 1.7]
        w_3d = 0.2 + raw_action[..., 4] * 1.5            # [0.2, 1.7]
        w_ssl = 0.1 + raw_action[..., 5] * 1.0           # [0.1, 1.1]
        route_thresh = 0.2 + raw_action[..., 6] * 0.6    # [0.2, 0.8]

        return {
            "aug_prob": aug_prob,
            "w_det": w_det,
            "w_seg": w_seg,
            "w_depth": w_depth,
            "w_3d": w_3d,
            "w_ssl": w_ssl,
            "route_thresh": route_thresh,
            "raw_action": raw_action
        }

    def compute_reward(self, quality_gain: float, latency_ms: float, memory_mb: float, grad_norm: float,
                       lambda_lat: float = 0.01, mu_mem: float = 0.001, nu_instability: float = 0.05) -> float:
        """Computes scalar reward preventing collapse or runaway resource usage."""
        reward = (
            quality_gain
            - lambda_lat * (latency_ms / 100.0)
            - mu_mem * (memory_mb / 1000.0)
            - nu_instability * min(grad_norm / 10.0, 5.0)
        )
        return float(reward)
