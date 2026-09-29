import torch
import torch.nn as nn
import torch.nn.functional as F
from ..geometry.camera import CameraIntrinsics

class RelativePoseEstimator(nn.Module):
    """
    Lightweight 6-DoF relative camera pose estimator predicting rotation (axis-angle/euler)
    and translation [R | t] between consecutive frames I_t and I_t+1.
    """
    def __init__(self, in_channels: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        # 6 outputs: 3 rotation angles (rx, ry, rz) + 3 translation components (tx, ty, tz)
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 6)
        )

    def forward(self, img_t: torch.Tensor, img_target: torch.Tensor) -> torch.Tensor:
        """
        img_t: (B, 3, H, W)
        img_target: (B, 3, H, W)
        Returns:
            pose: (B, 6) containing [rx, ry, rz, tx, ty, tz]
        """
        x = torch.cat([img_t, img_target], dim=1)
        feat = self.convs(x).flatten(1)
        pose = self.fc(feat)
        return pose

def euler_angles_to_matrix(rot_vec: torch.Tensor) -> torch.Tensor:
    """
    Converts 3D rotation vector (rx, ry, rz in radians) to 3x3 rotation matrices.
    rot_vec: (B, 3)
    Returns: (B, 3, 3)
    """
    B = rot_vec.shape[0]
    rx, ry, rz = rot_vec[:, 0], rot_vec[:, 1], rot_vec[:, 2]

    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    zeros = torch.zeros_like(rx)
    ones = torch.ones_like(rx)

    # Rx
    Rx = torch.stack([
        ones, zeros, zeros,
        zeros, cx, -sx,
        zeros, sx, cx
    ], dim=1).reshape(B, 3, 3)

    # Ry
    Ry = torch.stack([
        cy, zeros, sy,
        zeros, ones, zeros,
        -sy, zeros, cy
    ], dim=1).reshape(B, 3, 3)

    # Rz
    Rz = torch.stack([
        cz, -sz, zeros,
        sz, cz, zeros,
        zeros, zeros, ones
    ], dim=1).reshape(B, 3, 3)

    # R = Rz * Ry * Rx
    R = torch.bmm(Rz, torch.bmm(Ry, Rx))
    return R

class ViewSynthesisWarp(nn.Module):
    """
    Warps target frame I_target (e.g. I_t+1) back into source frame I_t
    using predicted metric depth map D_t, camera intrinsics K, and relative 6-DoF pose T_t->target.
    
    Formula:
      p_target ~ K * (R * (D_t * K^-1 * p_t) + t)
    """
    def __init__(self):
        super().__init__()

    def forward(self, img_target: torch.Tensor, depth_t: torch.Tensor,
                rot_vec: torch.Tensor, trans_vec: torch.Tensor,
                intrinsics: CameraIntrinsics) -> tuple:
        """
        img_target: (B, 3, H, W)
        depth_t: (B, 1, H, W) or (B, H, W) metric depth in source frame
        rot_vec: (B, 3) relative rotation
        trans_vec: (B, 3) relative translation
        intrinsics: CameraIntrinsics instance
        
        Returns:
            warped_img: (B, 3, H, W)
            valid_mask: (B, 1, H, W) float mask indicating projected pixels within image bounds
        """
        if depth_t.dim() == 3:
            depth_t = depth_t.unsqueeze(1)
        B, _, H, W = img_target.shape
        device = img_target.device
        dtype = img_target.dtype

        # 1. Generate normalized pixel grid
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij"
        )
        x_grid = x_grid.unsqueeze(0).expand(B, -1, -1) # (B, H, W)
        y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)

        # 2. Unproject source pixel coordinates to 3D camera coordinates in frame t
        # X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = Z
        z_t = depth_t.squeeze(1).clamp_min(1e-3)
        x_t = (x_grid - intrinsics.cx) * z_t / intrinsics.fx
        y_t = (y_grid - intrinsics.cy) * z_t / intrinsics.fy

        pts_t = torch.stack([x_t, y_t, z_t], dim=1).reshape(B, 3, H * W) # (B, 3, H*W)

        # 3. Transform 3D coordinates using relative pose: P_target = R * P_t + t
        R = euler_angles_to_matrix(rot_vec) # (B, 3, 3)
        t = trans_vec.unsqueeze(-1) # (B, 3, 1)

        pts_target = torch.bmm(R, pts_t) + t # (B, 3, H*W)
        x_tgt = pts_target[:, 0, :]
        y_tgt = pts_target[:, 1, :]
        z_tgt = pts_target[:, 2, :].clamp_min(1e-3)

        # 4. Project transformed 3D points to pixel coordinates in target frame
        u_tgt = x_tgt * intrinsics.fx / z_tgt + intrinsics.cx
        v_tgt = y_tgt * intrinsics.fy / z_tgt + intrinsics.cy

        # 5. Normalize to [-1, 1] for F.grid_sample
        u_norm = (u_tgt / (W - 1.0)) * 2.0 - 1.0
        v_norm = (v_tgt / (H - 1.0)) * 2.0 - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1).reshape(B, H, W, 2)

        # In-bounds valid mask
        valid_mask = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0) & \
                     (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0) & \
                     (pts_target[:, 2, :].reshape(B, H, W) > 1e-3)
        valid_mask = valid_mask.unsqueeze(1).float()

        # Bilinear sampling
        warped_img = F.grid_sample(img_target, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return warped_img, valid_mask

class PhotometricReconstructionLoss(nn.Module):
    """
    Self-Supervised Monocular Photometric Loss:
      L_photo = alpha * SSIM(I_t, I_hat) + (1 - alpha) * |I_t - I_hat|
    with auto-masking to eliminate moving objects (cattle, cars, dynamic agents)
    and edge-aware second-order depth smoothness.
    """
    def __init__(self, alpha: float = 0.85, beta_smooth: float = 0.1, auto_mask: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta_smooth = beta_smooth
        self.auto_mask = auto_mask

    def ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """SSIM calculation over 3x3 patch windows."""
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
        """Penalize depth gradient where image gradient is low (preserves depth discontinuities at true visual edges)."""
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        inv_depth = 1.0 / depth.clamp_min(1e-3)
        inv_depth = inv_depth / (inv_depth.mean(dim=[2, 3], keepdim=True).clamp_min(1e-6))

        d_dx = torch.abs(inv_depth[:, :, :, :-1] - inv_depth[:, :, :, 1:])
        d_dy = torch.abs(inv_depth[:, :, :-1, :] - inv_depth[:, :, 1:, :])

        i_dx = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
        i_dy = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)

        w_x = torch.exp(-i_dx)
        w_y = torch.exp(-i_dy)

        return (d_dx * w_x).mean() + (d_dy * w_y).mean()

    def forward(self, img_t: torch.Tensor, img_warped: torch.Tensor,
                img_raw_target: torch.Tensor, depth_t: torch.Tensor,
                valid_mask: torch.Tensor = None) -> dict:
        """
        img_t: (B, 3, H, W) source image at time t
        img_warped: (B, 3, H, W) target image warped into source frame t
        img_raw_target: (B, 3, H, W) unwarped target image (for stationary auto-masking)
        depth_t: (B, 1, H, W) predicted depth map at time t
        valid_mask: (B, 1, H, W) boundary validity mask
        """
        # 1. Photometric error map
        l1 = torch.abs(img_t - img_warped).mean(dim=1, keepdim=True)
        ssim_map = self.ssim(img_t, img_warped).mean(dim=1, keepdim=True)
        pe_warped = self.alpha * (1.0 - ssim_map) * 0.5 + (1.0 - self.alpha) * l1

        mask = valid_mask if valid_mask is not None else torch.ones_like(pe_warped)

        # 2. Auto-masking for stationary objects and camera-following motion
        if self.auto_mask:
            l1_unwarped = torch.abs(img_t - img_raw_target).mean(dim=1, keepdim=True)
            ssim_unwarped = self.ssim(img_t, img_raw_target).mean(dim=1, keepdim=True)
            pe_unwarped = self.alpha * (1.0 - ssim_unwarped) * 0.5 + (1.0 - self.alpha) * l1_unwarped
            
            auto_mask = (pe_warped < pe_unwarped).float()
            mask = mask * auto_mask

        photo_loss = (pe_warped * mask).sum() / (mask.sum().clamp_min(1.0))
        smooth_loss = self.edge_aware_smoothness(depth_t, img_t)
        total_loss = photo_loss + self.beta_smooth * smooth_loss

        return {
            "loss_photo": total_loss,
            "reconstruction_error": photo_loss.detach(),
            "smoothness_error": smooth_loss.detach(),
            "active_mask_ratio": mask.mean().detach()
        }

