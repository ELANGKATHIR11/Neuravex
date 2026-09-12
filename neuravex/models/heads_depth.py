import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ConvBNAct
from ..geometry.camera import inverse_to_depth, depth_to_inverse, CameraIntrinsics

class CameraAwareDEM(nn.Module):
    """
    Dense Elevation / Depth Model (DEM).
    Predicts inverse depth rho = 1/Z, recovers metric depth Z = 1/(rho + eps),
    and unprojects dense 3D pointcloud (X, Y, Z) given camera intrinsics K = (fx, fy, cx, cy).
    """
    def __init__(self, in_channels: int):
        super().__init__()
        c = in_channels
        self.conv_f = nn.Sequential(
            ConvBNAct(c * 3, c, 3),
            ConvBNAct(c, c // 2, 3),
            nn.Conv2d(c // 2, 1, 1)
        )

    def forward(self, q3, q4, q5, out_hw: tuple, intrinsics: CameraIntrinsics = None):
        f3 = q3
        f4 = F.interpolate(q4, size=q3.shape[-2:], mode="bilinear", align_corners=False)
        f5 = F.interpolate(q5, size=q3.shape[-2:], mode="bilinear", align_corners=False)
        feat = torch.cat([f3, f4, f5], dim=1)

        raw = self.conv_f(feat)
        # Inverse depth rho is strictly positive
        rho_low = F.softplus(raw) + 1e-4
        rho = F.interpolate(rho_low, size=out_hw, mode="bilinear", align_corners=False)
        
        # Metric depth Z = 1 / rho
        metric_depth = inverse_to_depth(rho)

        out = {
            "depth_inverse": rho,
            "depth_map": metric_depth
        }

        # If camera intrinsics are provided, unproject full 3D pointcloud
        if intrinsics is not None:
            dense_xyz = intrinsics.unproject_depth_map(metric_depth)
            out["dense_xyz"] = dense_xyz

        return out
