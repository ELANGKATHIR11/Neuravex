import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ConvBNAct
from ..geometry.camera import inverse_to_depth, depth_to_inverse, CameraIntrinsics

class CameraAwareDEM(nn.Module):
    """
    Dense Elevation / Metric Depth Model (DEM).
    Predicts:
      1. Inverse depth rho = 1/Z, recovering camera metric depth Z = 1/(rho + eps),
         and unprojecting dense 3D pointcloud (X, Y, Z) given camera intrinsics K.
      2. Normalized metric terrain elevation (h_terrain) explicitly separated:
         terrain elevation != object depth.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        c = in_channels
        # Predicts: channel 0 -> inverse depth rho, channel 1 -> depth confidence / log variance
        self.conv_f = nn.Sequential(
            ConvBNAct(c * 3, c, 3),
            ConvBNAct(c, c // 2, 3),
            nn.Conv2d(c // 2, 2, 1)
        )
        # Dedicated head for metric terrain elevation (e.g. USGS 3DEP / DEM)
        self.conv_terrain = nn.Sequential(
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
        raw_rho = raw[:, 0:1]
        raw_conf = raw[:, 1:2]

        # Inverse depth rho is strictly positive and bounded for float16 AMP stability
        rho_low = F.softplus(raw_rho).clamp(min=1e-3, max=50.0)
        rho = F.interpolate(rho_low, size=out_hw, mode="bilinear", align_corners=False)
        
        # Metric depth Z = 1 / rho (strictly positive metric distance in meters)
        metric_depth = (1.0 / rho).clamp(min=0.02, max=1000.0)

        # Calibrated depth confidence in [0, 1]
        depth_conf_low = torch.sigmoid(raw_conf)
        depth_confidence = F.interpolate(depth_conf_low, size=out_hw, mode="bilinear", align_corners=False)

        # Terrain elevation: separate branch predicting metric surface elevation
        raw_terrain = self.conv_terrain(feat)
        terrain_elevation = F.interpolate(raw_terrain, size=out_hw, mode="bilinear", align_corners=False)

        out = {
            "depth_inverse": rho,
            "depth_map": metric_depth,
            "depth_confidence": depth_confidence,
            "terrain_elevation": terrain_elevation
        }

        # If camera intrinsics are provided, unproject full 3D pointcloud
        if intrinsics is not None:
            dense_xyz = intrinsics.unproject_depth_map(metric_depth)
            out["dense_xyz"] = dense_xyz

        return out
