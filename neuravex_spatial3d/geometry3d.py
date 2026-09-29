"""
3D Bounding Boxes, Length/Width/Height (L, W, H) computation, Oriented 3D IoU,
and 3D rotation representation (Euler / Quaternion / Yaw).
"""

import math
from typing import Tuple, Optional, Dict, Any, Union
import torch
import numpy as np

class BoundingBox3D:
    """
    Native 3D Bounding Box represented in metric coordinates:
      - Center: (x, y, z) in meters
      - Extents: (length, width, height) in meters
      - Heading: yaw (radians around vertical axis)
    """
    def __init__(
        self,
        center: Union[Tuple[float, float, float], torch.Tensor],
        size_lwh: Union[Tuple[float, float, float], torch.Tensor],
        yaw: float = 0.0,
        score: float = 1.0,
        class_id: int = 0,
        class_name: str = "object",
        track_id: Optional[int] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        if isinstance(center, torch.Tensor):
            self.center = center.to(self.device).float()
        else:
            self.center = torch.tensor(center, dtype=torch.float32, device=self.device)

        if isinstance(size_lwh, torch.Tensor):
            self.lwh = size_lwh.to(self.device).float()
        else:
            self.lwh = torch.tensor(size_lwh, dtype=torch.float32, device=self.device)

        self.yaw = float(yaw)
        self.score = float(score)
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.track_id = track_id

    @property
    def volume(self) -> float:
        """Volume in cubic meters (L * W * H)."""
        return float(self.lwh[0] * self.lwh[1] * self.lwh[2])

    @property
    def distance(self) -> float:
        """Euclidean distance from sensor origin."""
        return float(torch.norm(self.center).item())

    def get_corners(self) -> torch.Tensor:
        """
        Computes the 8 corner vertices in 3D camera coordinates.
        Returns:
            Tensor of shape (8, 3)
        """
        l, w, h = self.lwh[0], self.lwh[1], self.lwh[2]
        # Canonical corner offsets
        x_corners = torch.tensor([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], device=self.device)
        y_corners = torch.tensor([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], device=self.device)
        z_corners = torch.tensor([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2], device=self.device)

        # 3D Yaw Rotation Matrix around Z (or Y depending on camera convention)
        c = math.cos(self.yaw)
        s = math.sin(self.yaw)
        rot = torch.tensor([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=self.device)

        corners = torch.stack([x_corners, y_corners, z_corners], dim=0) # (3, 8)
        rot_corners = torch.matmul(rot, corners) # (3, 8)
        corners_3d = rot_corners.T + self.center # (8, 3)
        return corners_3d

    def to_dict(self) -> Dict[str, Any]:
        """Serialize 3D box attributes."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": round(self.score, 4),
            "xyz": [round(v.item(), 3) for v in self.center],
            "lwh": [round(v.item(), 3) for v in self.lwh],
            "yaw_deg": round(math.degrees(self.yaw), 2),
            "volume_m3": round(self.volume, 4),
            "distance_m": round(self.distance, 3)
        }


def compute_lwh_from_pointcloud(points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Computes (Center, Length-Width-Height, Yaw) from a 3D point cloud cluster (N, 3)
    using Principal Component Analysis (PCA) bounding alignment.

    Args:
        points: (N, 3) Tensor of 3D points belonging to an object

    Returns:
        center: (3,) center of bounding box
        lwh: (3,) (length, width, height) in meters
        yaw: heading angle in radians
    """
    if points.shape[0] < 3:
        center = points.mean(dim=0) if points.shape[0] > 0 else torch.zeros(3, device=points.device)
        return center, torch.tensor([0.5, 0.5, 0.5], device=points.device), 0.0

    center = points.mean(dim=0)
    centered = points - center

    # 2D PCA on ground plane (X, Y)
    cov_2d = torch.matmul(centered[:, :2].T, centered[:, :2]) / (points.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_2d)

    # Primary direction corresponds to largest eigenvalue
    primary_axis = eigenvectors[:, 1]
    yaw = math.atan2(primary_axis[1].item(), primary_axis[0].item())

    # Rotate points to principal axes
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    rot_2d = torch.tensor([[c, -s], [s, c]], dtype=torch.float32, device=points.device)
    aligned_xy = torch.matmul(centered[:, :2], rot_2d.T)

    min_xy = aligned_xy.min(dim=0).values
    max_xy = aligned_xy.max(dim=0).values
    l = torch.clamp(max_xy[0] - min_xy[0], min=0.05)
    w = torch.clamp(max_xy[1] - min_xy[1], min=0.05)

    # Height along vertical axis (Z)
    z_min = centered[:, 2].min()
    z_max = centered[:, 2].max()
    h = torch.clamp(z_max - z_min, min=0.05)

    lwh = torch.stack([l, w, h])
    return center, lwh, yaw


def compute_lwh_from_mask_depth(
    mask: torch.Tensor,
    depth_map: torch.Tensor,
    intrinsics: Any,
    min_depth: float = 0.1,
    max_depth: float = 100.0
) -> Optional[BoundingBox3D]:
    """
    Computes 3D Bounding Box (Center + L,W,H) directly from 2D Instance Mask and Depth Map.
    """
    if mask.dtype != torch.bool:
        mask_bool = mask > 0.5
    else:
        mask_bool = mask

    valid = mask_bool & (depth_map >= min_depth) & (depth_map <= max_depth) & (~torch.isnan(depth_map))
    if valid.sum() < 10:
        return None

    v_idx, u_idx = torch.where(valid)
    d_vals = depth_map[valid]

    pts_3d = intrinsics.unproject_pixels(u_idx, v_idx, d_vals)
    center, lwh, yaw = compute_lwh_from_pointcloud(pts_3d)

    return BoundingBox3D(center=center, size_lwh=lwh, yaw=yaw, device=pts_3d.device)


def oriented_iou3d(box_a: BoundingBox3D, box_b: BoundingBox3D) -> float:
    """
    Calculates 3D Oriented Intersection-over-Union between two 3D bounding boxes.
    """
    dist = torch.norm(box_a.center - box_b.center).item()
    max_diag = 0.5 * (torch.norm(box_a.lwh) + torch.norm(box_b.lwh)).item()
    if dist > max_diag:
        return 0.0

    # Approximate 3D IoU using rotated intersection volume
    vol_a = box_a.volume
    vol_b = box_b.volume

    min_lwh = torch.minimum(box_a.lwh, box_b.lwh)
    overlap_dist = torch.clamp(0.5 * (box_a.lwh + box_b.lwh) - torch.abs(box_a.center - box_b.center), min=0.0)
    inter_vol = float((overlap_dist[0] * overlap_dist[1] * overlap_dist[2]).item())

    # Modulate with yaw alignment
    yaw_diff = abs(math.cos(box_a.yaw - box_b.yaw))
    inter_vol *= yaw_diff

    union_vol = vol_a + vol_b - inter_vol
    return max(0.0, min(1.0, inter_vol / max(union_vol, 1e-6)))
