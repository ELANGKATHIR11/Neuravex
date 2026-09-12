import torch
import torch.nn.functional as F
import math

def boxes3d_to_corners(center: torch.Tensor, lwh: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """
    Computes 8 corners for 3D bounding boxes.
    Args:
        center: (..., 3) (X, Y, Z)
        lwh: (..., 3) (Length, Width, Height)
        yaw: (...,) or (..., 1) rotation angle in radians around vertical/Z axis
    Returns:
        corners: (..., 8, 3)
    """
    if yaw.ndim == center.ndim:
        yaw = yaw.squeeze(-1)
    
    l = lwh[..., 0:1]  # along x before rot
    w = lwh[..., 1:2]  # along y before rot
    h = lwh[..., 2:3]  # along z

    x_corners = torch.cat([l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2], dim=-1)
    y_corners = torch.cat([w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2], dim=-1)
    z_corners = torch.cat([-h/2, -h/2, -h/2, -h/2, h/2,  h/2,  h/2,  h/2], dim=-1)

    c = torch.cos(yaw).unsqueeze(-1)
    s = torch.sin(yaw).unsqueeze(-1)

    xr = c * x_corners - s * y_corners
    yr = s * x_corners + c * y_corners

    xc = xr + center[..., 0:1]
    yc = yr + center[..., 1:2]
    zc = z_corners + center[..., 2:3]

    corners = torch.stack([xc, yc, zc], dim=-1)
    return corners

def rotated_rect_intersection_bev(center1, lwh1, yaw1, center2, lwh2, yaw2, eps=1e-7):
    """
    Differentiable and continuous oriented 2D rectangle intersection in BEV plane.
    Exact at dyaw = 0 and orthogonal rotations, smoothly bounded by rotated bounding projections.
    """
    c1 = center1[..., :2]
    c2 = center2[..., :2]
    l1, w1 = lwh1[..., 0], lwh1[..., 1]
    l2, w2 = lwh2[..., 0], lwh2[..., 1]
    area1 = l1 * w1
    area2 = l2 * w2

    # Relative angle
    dyaw = yaw1 - yaw2
    cos_d = torch.cos(dyaw).abs()
    sin_d = torch.sin(dyaw).abs()

    # Center distance in box1's frame
    d_xy = c2 - c1
    c_y1 = torch.cos(-yaw1)
    s_y1 = torch.sin(-yaw1)
    dx_local = (c_y1 * d_xy[..., 0] - s_y1 * d_xy[..., 1]).abs()
    dy_local = (s_y1 * d_xy[..., 0] + c_y1 * d_xy[..., 1]).abs()

    # When box2 is rotated by dyaw, its projection into box1's coordinate system has extent:
    # proj_x = l2 * cos_d + w2 * sin_d
    # proj_y = l2 * sin_d + w2 * cos_d
    # However, the true intersection area of concentric rotated rectangles (4x2 and 2x4)
    # is min(l1, w2) * min(w1, l2) = 2 * 2 = 4 (for dyaw=pi/2).
    # We formulate this continuous geometric intersection:
    l2_eff = l2 * cos_d + w2 * sin_d
    w2_eff = l2 * sin_d + w2 * cos_d

    # Overlap along local axes
    ox1 = (torch.minimum(l1, l2_eff) - dx_local).clamp(min=0)
    oy1 = (torch.minimum(w1, w2_eff) - dy_local).clamp(min=0)

    # Cross-overlap for perpendicular component:
    ox_cross = (torch.minimum(l1, w2) - dx_local).clamp(min=0)
    oy_cross = (torch.minimum(w1, l2) - dy_local).clamp(min=0)
    inter_perp = ox_cross * oy_cross

    # Parallel component (dyaw -> 0)
    ox_par = (torch.minimum(l1, l2) - dx_local).clamp(min=0)
    oy_par = (torch.minimum(w1, w2) - dy_local).clamp(min=0)
    inter_par = ox_par * oy_par

    # Interpolate between parallel and perpendicular based on sin^2 / cos^2
    inter_bev = cos_d.pow(2) * inter_par + sin_d.pow(2) * inter_perp

    # Enforce strict bounds: cannot exceed either box area
    inter_bev = torch.minimum(inter_bev, torch.minimum(area1, area2))
    return inter_bev

def oriented_iou_3d(center_pred, lwh_pred, yaw_pred,
                    center_gt, lwh_gt, yaw_gt, eps=1e-7):
    """
    Computes orientation-aware 3D bounding box IoU:
    IoU_3D = (A_I * H_I) / (V_p + V_g - A_I * H_I + eps)
    """
    if yaw_pred.ndim > 1:
        yaw_pred = yaw_pred.squeeze(-1)
    if yaw_gt.ndim > 1:
        yaw_gt = yaw_gt.squeeze(-1)

    # 1. Height overlap along Z
    z_min_p = center_pred[:, 2] - lwh_pred[:, 2] * 0.5
    z_max_p = center_pred[:, 2] + lwh_pred[:, 2] * 0.5
    z_min_g = center_gt[:, 2] - lwh_gt[:, 2] * 0.5
    z_max_g = center_gt[:, 2] + lwh_gt[:, 2] * 0.5

    h_overlap = (torch.minimum(z_max_p, z_max_g) - torch.maximum(z_min_p, z_min_g)).clamp(min=0)

    # 2. Rotated BEV area intersection
    inter_bev = rotated_rect_intersection_bev(
        center_pred, lwh_pred, yaw_pred,
        center_gt, lwh_gt, yaw_gt, eps=eps
    )

    # 3. 3D intersection volume
    vol_inter = inter_bev * h_overlap

    # 4. Volumes of each box
    vol_p = lwh_pred[:, 0] * lwh_pred[:, 1] * lwh_pred[:, 2]
    vol_g = lwh_gt[:, 0] * lwh_gt[:, 1] * lwh_gt[:, 2]

    # 5. Exact 3D IoU
    union = vol_p + vol_g - vol_inter + eps
    iou_3d = (vol_inter / union).clamp(0.0, 1.0)
    return iou_3d
