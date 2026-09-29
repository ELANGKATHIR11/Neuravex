"""
Sensor Calibration & Fusion: Camera Pinhole Models, LiDAR Spherical Sensors,
and Extrinsic Rigid-Body Transformations (Camera <-> LiDAR).
"""

import math
from typing import Optional, Tuple, Dict, Any, Union
import torch
import numpy as np

class CameraIntrinsics:
    """
    Pinhole camera model with focal lengths (fx, fy), principal points (cx, cy),
    and distortion coefficients (k1, k2, p1, p2).
    """
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int = 1920,
        height: int = 1080,
        k1: float = 0.0,
        k2: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
        device: Union[str, torch.device] = "cpu"
    ):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(width)
        self.height = int(height)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.p1 = float(p1)
        self.p2 = float(p2)
        self.device = torch.device(device) if isinstance(device, str) else device

    @property
    def matrix(self) -> torch.Tensor:
        """Returns 3x3 intrinsic matrix K."""
        return torch.tensor([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=self.device)

    def scale(self, scale_x: float, scale_y: float) -> "CameraIntrinsics":
        """Scales intrinsics after image resizing."""
        return CameraIntrinsics(
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
            width=int(self.width * scale_x),
            height=int(self.height * scale_y),
            device=self.device
        )

    def unproject_pixels(self, u: torch.Tensor, v: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Unproject 2D pixel coordinates (u, v) and metric depth Z to 3D camera points (X, Y, Z).
        Returns (..., 3) Tensor in meters.
        """
        u = u.to(self.device).float()
        v = v.to(self.device).float()
        depth = depth.to(self.device).float()

        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return torch.stack([x, y, z], dim=-1)

    def project_points(self, points_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project 3D points (X, Y, Z) into 2D pixel coordinates (u, v) and depth Z.
        Returns:
            u, v, z
        """
        points_3d = points_3d.to(self.device).float()
        x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]
        eps = 1e-6
        z_safe = torch.clamp(z, min=eps)

        u = (x * self.fx / z_safe) + self.cx
        v = (y * self.fy / z_safe) + self.cy
        return u, v, z


class LiDARConfig:
    """
    LiDAR sensor specification and spherical-to-cartesian unprojection model.
    """
    def __init__(
        self,
        num_beams: int = 64,
        horizontal_fov: float = 360.0,
        vertical_fov_up: float = 15.0,
        vertical_fov_down: float = -25.0,
        max_range: float = 120.0,
        min_range: float = 0.5,
        device: Union[str, torch.device] = "cpu"
    ):
        self.num_beams = num_beams
        self.horizontal_fov = horizontal_fov
        self.vertical_fov_up = vertical_fov_up
        self.vertical_fov_down = vertical_fov_down
        self.max_range = float(max_range)
        self.min_range = float(min_range)
        self.device = torch.device(device) if isinstance(device, str) else device

    def spherical_to_cartesian(
        self,
        ranges: torch.Tensor,
        azimuths: torch.Tensor,
        elevations: torch.Tensor
    ) -> torch.Tensor:
        """
        Converts range, azimuth (radians), and elevation (radians) into (X, Y, Z) point cloud.
        """
        ranges = ranges.to(self.device).float()
        azimuths = azimuths.to(self.device).float()
        elevations = elevations.to(self.device).float()

        # Mask invalid ranges
        valid = (ranges >= self.min_range) & (ranges <= self.max_range)
        ranges = torch.where(valid, ranges, torch.zeros_like(ranges))

        x = ranges * torch.cos(elevations) * torch.cos(azimuths)
        y = ranges * torch.cos(elevations) * torch.sin(azimuths)
        z = ranges * torch.sin(elevations)
        return torch.stack([x, y, z], dim=-1)


class SensorFusionEngine:
    """
    Fuses camera optical data (RGB + Depth map) with LiDAR point clouds using
    SE(3) rigid-body transformation (Rotation matrix R, Translation vector T).
    """
    def __init__(
        self,
        camera: CameraIntrinsics,
        lidar: Optional[LiDARConfig] = None,
        r_lidar2cam: Optional[torch.Tensor] = None,
        t_lidar2cam: Optional[torch.Tensor] = None
    ):
        self.camera = camera
        self.lidar = lidar or LiDARConfig(device=camera.device)
        self.device = camera.device

        # Default Extrinsic: Identity rotation, zero translation
        if r_lidar2cam is None:
            self.R = torch.eye(3, dtype=torch.float32, device=self.device)
        else:
            self.R = r_lidar2cam.to(self.device).float()

        if t_lidar2cam is None:
            self.T = torch.zeros(3, dtype=torch.float32, device=self.device)
        else:
            self.T = t_lidar2cam.to(self.device).float()

    def transform_lidar_to_camera(self, lidar_pts: torch.Tensor) -> torch.Tensor:
        """
        Transforms LiDAR points (N, 3) into camera 3D coordinates: P_cam = R * P_lidar + T
        """
        pts = lidar_pts.to(self.device).float()
        return (torch.matmul(pts, self.R.T) + self.T)

    def project_lidar_to_image(self, lidar_pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Projects LiDAR points into camera pixel plane.
        Returns:
            u_px, v_px, valid_mask (within image dimensions and positive depth)
        """
        cam_pts = self.transform_lidar_to_camera(lidar_pts)
        u, v, z = self.camera.project_points(cam_pts)

        valid = (
            (z > 0.1) &
            (u >= 0) & (u < self.camera.width) &
            (v >= 0) & (v < self.camera.height)
        )
        return u, v, valid
