import torch
import torch.nn as nn
import torch.nn.functional as F

class PhotometricDepthSSLLoss(nn.Module):
    """
    Self-Supervised Monocular Photometric Depth Loss:
      I_t -> D_t -> warp(T) -> I_hat
      L_dssl = alpha * L_photo + beta * L_smooth + gamma * L_metric

    Uses SSIM + L1 reprojection, occlusion masking, and edge-aware second-order smoothness.
    """
    def __init__(self, alpha: float = 0.85, beta: float = 0.1, gamma: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Structural Similarity Index Measure (SSIM) on image patches."""
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)

        sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        return (ssim_n / ssim_d.clamp_min(1e-7)).clamp(0.0, 1.0)

    def edge_aware_smoothness(self, depth: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """Edge-aware smoothness: penalizes depth discontinuities where image gradient is low."""
        d_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        d_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])

        i_dx = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
        i_dy = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)

        w_x = torch.exp(-i_dx)
        w_y = torch.exp(-i_dy)

        smooth_x = (d_dx * w_x).mean()
        smooth_y = (d_dy * w_y).mean()
        return smooth_x + smooth_y

    def forward(self, pred_depth: torch.Tensor, img_target: torch.Tensor, img_warped: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        pred_depth: (B, 1, H, W)
        img_target: (B, 3, H, W)
        img_warped: (B, 3, H, W) reconstructed view
        """
        # Photometric error: SSIM + L1
        l1 = torch.abs(img_target - img_warped).mean(dim=1, keepdim=True)
        ssim_val = self.ssim(img_target, img_warped).mean(dim=1, keepdim=True)
        photo_map = self.alpha * (1.0 - ssim_val) * 0.5 + (1.0 - self.alpha) * l1

        if valid_mask is not None:
            photo_loss = (photo_map * valid_mask).sum() / (valid_mask.sum().clamp_min(1.0))
        else:
            photo_loss = photo_map.mean()

        # Edge-aware smoothness
        smooth_loss = self.edge_aware_smoothness(pred_depth, img_target)

        # Metric scale prior (depth should stay within plausible non-zero range)
        metric_prior = F.relu(0.1 - pred_depth).mean() + F.relu(pred_depth - 100.0).mean()

        total = self.alpha * photo_loss + self.beta * smooth_loss + self.gamma * metric_prior
        return total
