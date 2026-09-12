import torch
import torch.nn as nn

class AdaptiveTaskLoss(nn.Module):
    """
    Homoscedastic uncertainty loss weighting with task active masks:
    L = sum_t m_t * (exp(-s_t) * L_t + s_t)
    where:
        m_t in {0, 1} is a binary availability mask for task t
        s_t is a learnable log variance parameter
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
        """
        Args:
            losses: dict of {task_name: loss_tensor}
            task_masks: dict of {task_name: 0 or 1 or bool tensor}
        Returns:
            total_loss: scalar tensor
            raw_losses: dict of float values
            weighted_losses: dict of float values
            task_weights: dict of exp(-s_t) float values
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        raw_losses = {}
        weighted_losses = {}
        weights = {}

        for t in self.task_names:
            if t not in losses:
                continue

            loss_val = losses[t]
            raw_losses[t] = loss_val.item() if isinstance(loss_val, torch.Tensor) else float(loss_val)

            # Check task mask (whether modality is present)
            mask = 1.0
            if task_masks is not None and t in task_masks:
                m = task_masks[t]
                mask = float(m.item() if isinstance(m, torch.Tensor) else m)

            if mask <= 0.0:
                continue

            s = self.log_vars[t]
            w = torch.exp(-s)
            weights[t] = w.item()

            weighted_l = w * loss_val + s
            weighted_losses[t] = weighted_l.item() if isinstance(weighted_l, torch.Tensor) else float(weighted_l)
            total_loss = total_loss + weighted_l

        return total_loss, raw_losses, weighted_losses, weights
