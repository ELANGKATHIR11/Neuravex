"""
Native Segmentation to 3D Space: Mask-depth backprojection,
per-instance point cloud extraction, and volumetric isolation.
"""

from typing import Tuple, Dict, Any, List, Optional
import torch
from .sensors import CameraIntrinsics
from .geometry3d import BoundingBox3D, compute_lwh_from_pointcloud

class Mask3DProjector:
    """
    Back-projects 2D binary instance segmentation masks into calibrated 3D point clouds
    and extracts volumetric physical bounding extents.
    """
    def __init__(self, camera: CameraIntrinsics, min_depth: float = 0.1, max_depth: float = 120.0):
        self.camera = camera
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)

    def mask_to_points(
        self,
        mask: torch.Tensor,
        depth_map: torch.Tensor,
        subsample: int = 1
    ) -> torch.Tensor:
        """
        Extracts 3D coordinates for all valid pixels inside a 2D segmentation mask.

        Returns:
            (N, 3) point cloud Tensor in meters.
        """
        if mask.dtype != torch.bool:
            mask_bool = mask > 0.5
        else:
            mask_bool = mask

        valid = mask_bool & (depth_map >= self.min_depth) & (depth_map <= self.max_depth) & (~torch.isnan(depth_map))
        v_idx, u_idx = torch.where(valid)

        if subsample > 1:
            u_idx = u_idx[::subsample]
            v_idx = v_idx[::subsample]

        d_vals = depth_map[v_idx, u_idx]
        return self.camera.unproject_pixels(u_idx, v_idx, d_vals)

    def extract_instance_box3d(
        self,
        mask: torch.Tensor,
        depth_map: torch.Tensor,
        class_id: int = 0,
        class_name: str = "object",
        score: float = 1.0,
        track_id: Optional[int] = None
    ) -> Optional[BoundingBox3D]:
        """
        Generates a 3D Bounding Box from an instance mask and depth map.
        """
        pts = self.mask_to_points(mask, depth_map)
        if pts.shape[0] < 5:
            return None

        center, lwh, yaw = compute_lwh_from_pointcloud(pts)
        return BoundingBox3D(
            center=center,
            size_lwh=lwh,
            yaw=yaw,
            score=score,
            class_id=class_id,
            class_name=class_name,
            track_id=track_id,
            device=str(pts.device)
        )


def robust_mask_unproject(
    mask: torch.Tensor,
    depth_map: torch.Tensor,
    camera: CameraIntrinsics
) -> Tuple[torch.Tensor, BoundingBox3D]:
    """
    Helper function to cleanly unproject a mask into (Pointcloud, BoundingBox3D).
    """
    projector = Mask3DProjector(camera)
    pts = projector.mask_to_points(mask, depth_map)
    box = projector.extract_instance_box3d(mask, depth_map)
    return pts, box
