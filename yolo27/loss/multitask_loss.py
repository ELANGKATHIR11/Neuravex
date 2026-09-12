import torch
import torch.nn as nn

class AdaptiveTaskLoss(nn.Module):
    """
    Homoscedastic uncertainty loss weighting with safe bounded log variances:
    L = sum_t m_t * (exp(-s_t) * L_t + s_t)
    where s_t in [-4.0, 4.0] ensures stable positive weights exp(-s_t) in [0.018, 54.6].
    """
    def __init__(self, task_names: list, init_weights: dict = None):
        super().__init__()
        self.task_names = list(task_names)
        self.log_vars = nn.ParameterDict()
        for t in self.task_names:
            init_val = 0.0
            if init_weights and t in init_weights:
                init_val = float(init_weights[t])
            self.log_vars[t] = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

    def forward(self, losses: dict, task_masks: dict = None) -> tuple:
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        raw_losses = {}
        weighted_losses = {}
        weights = {}

        num_active_tasks = 0

        for t in self.task_names:
            if t not in losses:
                continue

            loss_val = losses[t]
            raw_losses[t] = loss_val.item() if isinstance(loss_val, torch.Tensor) else float(loss_val)

            # Modality mask
            mask = 1.0
            if task_masks is not None and t in task_masks:
                m = task_masks[t]
                mask = float(m.item() if isinstance(m, torch.Tensor) else m)

            if mask <= 0.0:
                continue

            # Clamped log variance for strict stability
            s = self.log_vars[t].clamp(-4.0, 4.0)
            w = torch.exp(-s)
            weights[t] = w.item()

            weighted_l = w * loss_val + s
            weighted_losses[t] = weighted_l.item() if isinstance(weighted_l, torch.Tensor) else float(weighted_l)
            total_loss = total_loss + weighted_l
            num_active_tasks += 1

        # Safe normalization across active tasks
        if num_active_tasks > 0:
            total_loss = total_loss / float(num_active_tasks)

        return total_loss, raw_losses, weighted_losses, weights
