import torch
import torch.nn as nn
import torch.nn.functional as F
from ..geometry.camera import CameraIntrinsics

class CrossTaskGeometryLoss(nn.Module):
    """
    Validates physical alignment between 2D bounding boxes and 3D centers (X, Y, Z),
    and enforces boundary alignment between segmentation masks and depth gradient edges.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_boxes: torch.Tensor, pred_xyz: torch.Tensor,
                pred_seg: torch.Tensor, pred_depth: torch.Tensor,
                intrinsics: CameraIntrinsics = None) -> torch.Tensor:
        """
        pred_boxes: (B, N, 4) in [x1, y1, x2, y2]
        pred_xyz: (B, N, 3) in [X, Y, Z]
        pred_seg: (B, C, H, W)
        pred_depth: (B, 1, H, W)
        """
        # 1. 2D-3D Center Projection Consistency
        # Center of 2D box: u_c = (x1 + x2)/2, v_c = (y1 + y2)/2
        u_2d = (pred_boxes[..., 0] + pred_boxes[..., 2]) * 0.5
        v_2d = (pred_boxes[..., 1] + pred_boxes[..., 3]) * 0.5

        if intrinsics is not None:
            # Re-project 3D (X, Y, Z) to image plane: u_proj = fx * X / Z + cx, v_proj = fy * Y / Z + cy
            z = pred_xyz[..., 2].clamp_min(0.1)
            u_3d = intrinsics.fx * pred_xyz[..., 0] / z + intrinsics.cx
            v_3d = intrinsics.fy * pred_xyz[..., 1] / z + intrinsics.cy

            proj_err = torch.sqrt((u_2d - u_3d) ** 2 + (v_2d - v_3d) ** 2 + self.eps).mean()
        else:
            proj_err = torch.tensor(0.0, device=pred_boxes.device)

        # 2. Mask-Depth Edge Co-alignment
        # Depth gradient magnitude
        d_dx = torch.abs(pred_depth[:, :, :, :-1] - pred_depth[:, :, :, 1:])
        d_dy = torch.abs(pred_depth[:, :, :-1, :] - pred_depth[:, :, 1:, :])
        depth_grad = F.pad(d_dx, (0, 1, 0, 0)) + F.pad(d_dy, (0, 0, 0, 1))

        # Seg edge gradient
        s_prob = torch.softmax(pred_seg, dim=1)
        s_dx = torch.abs(s_prob[:, :, :, :-1] - s_prob[:, :, :, 1:]).mean(dim=1, keepdim=True)
        s_dy = torch.abs(s_prob[:, :, :-1, :] - s_prob[:, :, 1:, :]).mean(dim=1, keepdim=True)
        seg_grad = F.pad(s_dx, (0, 1, 0, 0)) + F.pad(s_dy, (0, 0, 0, 1))

        # Cosine alignment between depth gradient and segmentation boundary
        # Flatten spatial dims
        dg_flat = depth_grad.flatten(1)
        sg_flat = seg_grad.flatten(1)
        cos_sim = F.cosine_similarity(dg_flat, sg_flat, dim=-1)
        edge_loss = (1.0 - cos_sim).mean()

        total = proj_err * 0.05 + edge_loss * 0.1
        return total

class TemporalConsistencyLoss(nn.Module):
    """
    Temporal motion consistency across consecutive video frames (I_t, I_t1):
      L_temp = L_f + L_b + L_m + L_d + L_3D
    Penalizes sudden drift while applying motion compensation via motion_warp when provided.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, out_t: dict, out_t1: dict, motion_warp: torch.Tensor = None) -> torch.Tensor:
        # Feature drift
        l_f = F.mse_loss(out_t["class_logits"], out_t1["class_logits"].detach())

        # Box motion smoothness
        l_b = F.l1_loss(out_t["pred_boxes"], out_t1["pred_boxes"].detach())

        # 3D Center trajectory consistency
        if "pred_xyz" in out_t and "pred_xyz" in out_t1:
            l_3d = F.l1_loss(out_t["pred_xyz"], out_t1["pred_xyz"].detach())
        else:
            l_3d = torch.tensor(0.0, device=l_f.device)

        # Dense depth temporal stability with optional motion compensation
        if "depth_map" in out_t and "depth_map" in out_t1:
            d_t = out_t["depth_map"]
            d_t1 = out_t1["depth_map"].detach()
            if motion_warp is not None and motion_warp.shape[-2:] == (2, 3):
                # Affine motion compensation grid
                B, C, H, W = d_t.shape
                grid = F.affine_grid(motion_warp, [B, C, H, W], align_corners=False)
                d_t1 = F.grid_sample(d_t1, grid, align_corners=False)
            l_d = F.l1_loss(d_t, d_t1)
        else:
            l_d = torch.tensor(0.0, device=l_f.device)

        total = l_f * 0.5 + l_b * 0.5 + l_3d * 0.2 + l_d * 0.1
        return total
