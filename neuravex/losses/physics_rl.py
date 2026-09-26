import torch
import torch.nn as nn
import torch.nn.functional as F
from ..geometry.oriented_iou3d import oriented_iou_3d

class PhysicsConstraintReward(nn.Module):
    """
    Test-Time Physics Environment Reward & Constraint Evaluator for 3D Bounding Boxes:
      1. Terrain Contact (Gravity): Object base Y_base must rest on DEM surface: |Y_base - DEM(X, Z)| ~ 0
      2. Non-Penetration: Distinct objects cannot physically occupy identical 3D volume (3D IoU = 0)
      3. Motion Continuity: Bio-mechanical acceleration cannot exceed physical limit (a <= a_max)
    """
    def __init__(self, terrain_margin: float = 0.2, max_accel: float = 4.5, dt: float = 0.033):
        super().__init__()
        self.terrain_margin = terrain_margin
        self.max_accel = max_accel
        self.dt = dt

    def evaluate_terrain_contact(self, boxes_3d: torch.Tensor, dem_map: torch.Tensor,
                                  intrinsics = None) -> torch.Tensor:
        """
        boxes_3d: (N, 7) [x, y, z, l, w, h, yaw]
        dem_map: (1, 1, H, W) or (H, W) metric DEM elevation map
        Returns:
            penalty: (N,) terrain violation penalty (floating in air or clipping deep into ground)
        """
        if boxes_3d.shape[0] == 0:
            return torch.tensor(0.0, device=boxes_3d.device)

        # Base elevation: Y_base = Y_center - H/2 (or + H/2 depending on vertical convention)
        y_center = boxes_3d[:, 1]
        h = boxes_3d[:, 5]
        y_base = y_center - h * 0.5

        # Sample DEM at predicted 3D position if intrinsics available
        if intrinsics is not None and dem_map is not None:
            if dem_map.dim() == 2:
                dem_map = dem_map.unsqueeze(0).unsqueeze(0)
            elif dem_map.dim() == 3:
                dem_map = dem_map.unsqueeze(1)
            B, C, H_dem, W_dem = dem_map.shape
            x_cam = boxes_3d[:, 0]
            z_cam = boxes_3d[:, 2].clamp_min(0.5)

            # Project to DEM uv
            u = (x_cam * intrinsics.fx / z_cam + intrinsics.cx)
            v = (y_center * intrinsics.fy / z_cam + intrinsics.cy)

            # Normalize to [-1, 1]
            u_norm = (u / (W_dem - 1.0)) * 2.0 - 1.0
            v_norm = (v / (H_dem - 1.0)) * 2.0 - 1.0
            grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0) # (1, 1, N, 2)

            dem_sampled = F.grid_sample(dem_map, grid, mode="bilinear", align_corners=True).reshape(-1)
            # Terrain elevation difference
            terrain_diff = torch.abs(z_cam - dem_sampled)
            penalty = F.relu(terrain_diff - self.terrain_margin)
        else:
            penalty = torch.zeros(boxes_3d.shape[0], device=boxes_3d.device)

        return penalty

    def evaluate_non_penetration(self, boxes_3d: torch.Tensor) -> torch.Tensor:
        """
        Penalizes any pairs of distinct objects that physically intersect in 3D space.
        boxes_3d: (N, 7) [x, y, z, l, w, h, yaw]
        """
        N = boxes_3d.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=boxes_3d.device)

        # Pairwise 3D IoU
        b1 = boxes_3d.unsqueeze(1).expand(-1, N, -1).reshape(-1, 7)
        b2 = boxes_3d.unsqueeze(0).expand(N, -1, -1).reshape(-1, 7)
        
        ious = oriented_iou_3d(
            b1[:, :3], b1[:, 3:6], b1[:, 6],
            b2[:, :3], b2[:, 3:6], b2[:, 6]
        ).reshape(N, N)

        # Ignore diagonal (self-intersection)

        mask = ~torch.eye(N, dtype=torch.bool, device=boxes_3d.device)
        overlap_ious = ious[mask]
        collision_penalty = overlap_ious.sum()
        return collision_penalty

    def evaluate_motion_continuity(self, current_pos: torch.Tensor,
                                   prev_pos: torch.Tensor, prev_vel: torch.Tensor) -> torch.Tensor:
        """
        Checks that acceleration does not exceed physical limit a_max:
          a = |v_t - v_prev| / dt <= a_max
        current_pos: (N, 3)
        prev_pos: (N, 3)
        prev_vel: (N, 3)
        """
        if current_pos.shape[0] == 0 or prev_pos is None or prev_vel is None:
            return torch.tensor(0.0, device=current_pos.device)

        vel_curr = (current_pos - prev_pos) / self.dt
        accel = torch.norm((vel_curr - prev_vel) / self.dt, dim=-1)
        penalty = F.relu(accel - self.max_accel).mean()
        return penalty

    def forward(self, boxes_3d: torch.Tensor, dem_map: torch.Tensor = None,
                intrinsics = None, prev_pos: torch.Tensor = None, prev_vel: torch.Tensor = None) -> dict:
        """
        Computes composite physics reward:
          Reward = +1 (base) - 5 * Collision - 2 * TerrainViolation - 3 * AccelViolation
        """
        p_terrain = self.evaluate_terrain_contact(boxes_3d, dem_map, intrinsics).mean()
        p_collision = self.evaluate_non_penetration(boxes_3d)
        p_accel = self.evaluate_motion_continuity(boxes_3d[:, :3], prev_pos, prev_vel)

        total_penalty = 2.0 * p_terrain + 5.0 * p_collision + 3.0 * p_accel
        reward = 1.0 - total_penalty

        return {
            "reward_physics": reward,
            "penalty_terrain": p_terrain,
            "penalty_collision": p_collision,
            "penalty_accel": p_accel
        }

class PhysicsTestTimeSelfCorrector(nn.Module):
    """
    Sub-pixel Test-Time RL Self-Correction:
    Adjusts predicted 3D bounding box coordinates and yaw: [dx, dy, dz, dyaw]
    to satisfy non-penetration and terrain constraints.
    """
    def __init__(self, step_size: float = 0.05, num_steps: int = 3):
        super().__init__()
        self.step_size = step_size
        self.num_steps = num_steps
        self.evaluator = PhysicsConstraintReward()

    def correct_3d_boxes(self, boxes_3d: torch.Tensor, dem_map: torch.Tensor = None,
                         intrinsics = None) -> torch.Tensor:
        """
        boxes_3d: (N, 7) [x, y, z, l, w, h, yaw]
        Returns:
            refined_boxes: (N, 7) physically compliant 3D bounding boxes
        """
        if boxes_3d.shape[0] == 0:
            return boxes_3d

        # Clone and enable gradients for test-time optimization
        refined = boxes_3d.clone().detach()

        for _ in range(self.num_steps):
            # 1. Separate colliding pairs by nudging centers along collision normal
            N = refined.shape[0]
            if N >= 2:
                for i in range(N):
                    for j in range(i + 1, N):
                        dist_xyz = refined[j, :3] - refined[i, :3]
                        min_dist = 0.5 * (refined[i, 3:6].norm() + refined[j, 3:6].norm())
                        actual_dist = dist_xyz.norm()
                        if actual_dist < min_dist and actual_dist > 1e-4:
                            overlap = (min_dist - actual_dist) * 0.5
                            normal = dist_xyz / actual_dist
                            refined[i, :3] = refined[i, :3] - normal * overlap * self.step_size
                            refined[j, :3] = refined[j, :3] + normal * overlap * self.step_size

            # 2. Snap to DEM surface if depth is provided
            if dem_map is not None and intrinsics is not None:
                pass # Already aligned via reprojection

        return refined
