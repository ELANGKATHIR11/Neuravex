import torch
import torch.nn.functional as F
import math

def boxes3d_to_corners(center: torch.Tensor, lwh: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """
    Computes 8 corners for 3D bounding boxes.
    """
    if yaw.ndim == center.ndim:
        yaw = yaw.squeeze(-1)
    
    l = lwh[..., 0:1]
    w = lwh[..., 1:2]
    h = lwh[..., 2:3]

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

    return torch.stack([xc, yc, zc], dim=-1)

def rotated_rect_intersection_bev(center1, lwh1, yaw1, center2, lwh2, yaw2, eps=1e-7):
    """
    Symmetric, continuous oriented rectangle intersection in BEV plane:
    - Symmetric: inter(A, B) == inter(B, A)
    - Valid range: 0 <= inter <= min(area1, area2)
    - Sensitive to yaw difference, center distance, and aspect ratio.
    """
    c1 = center1[..., :2]
    c2 = center2[..., :2]
    l1, w1 = lwh1[..., 0], lwh1[..., 1]
    l2, w2 = lwh2[..., 0], lwh2[..., 1]
    area1 = (l1 * w1).clamp_min(eps)
    area2 = (l2 * w2).clamp_min(eps)

    # Relative angle modulo pi/2
    dyaw = (yaw1 - yaw2).abs()
    # Normalize dyaw to [0, pi]
    dyaw = dyaw % math.pi
    cos_d = torch.cos(dyaw).abs()
    sin_d = torch.sin(dyaw).abs()

    # Center distance in world frame
    d_xy = c2 - c1
    dist_sq = (d_xy[..., 0]**2 + d_xy[..., 1]**2)
    dist = torch.sqrt(dist_sq + eps)

    # Local projections in both box frames for perfect symmetry
    c_y1, s_y1 = torch.cos(-yaw1), torch.sin(-yaw1)
    dx1 = (c_y1 * d_xy[..., 0] - s_y1 * d_xy[..., 1]).abs()
    dy1 = (s_y1 * d_xy[..., 0] + c_y1 * d_xy[..., 1]).abs()

    c_y2, s_y2 = torch.cos(-yaw2), torch.sin(-yaw2)
    dx2 = (c_y2 * (-d_xy[..., 0]) - s_y2 * (-d_xy[..., 1])).abs()
    dy2 = (s_y2 * (-d_xy[..., 0]) + c_y2 * (-d_xy[..., 1])).abs()

    # Overlaps at aligned and cross orientations
    # Parallel (dyaw=0)
    ox_par1 = (torch.minimum(l1, l2) - dx1).clamp(min=0)
    oy_par1 = (torch.minimum(w1, w2) - dy1).clamp(min=0)
    inter_par1 = ox_par1 * oy_par1

    ox_par2 = (torch.minimum(l1, l2) - dx2).clamp(min=0)
    oy_par2 = (torch.minimum(w1, w2) - dy2).clamp(min=0)
    inter_par2 = ox_par2 * oy_par2
    inter_par = torch.minimum(inter_par1, inter_par2)

    # Perpendicular (dyaw=pi/2)
    ox_cross1 = (torch.minimum(l1, w2) - dx1).clamp(min=0)
    oy_cross1 = (torch.minimum(w1, l2) - dy1).clamp(min=0)
    inter_cross1 = ox_cross1 * oy_cross1

    ox_cross2 = (torch.minimum(l2, w1) - dx2).clamp(min=0)
    oy_cross2 = (torch.minimum(w2, l1) - dy2).clamp(min=0)
    inter_cross2 = ox_cross2 * oy_cross2
    inter_cross = torch.minimum(inter_cross1, inter_cross2)

    # Continuous angular blend
    inter_bev = cos_d.pow(2) * inter_par + sin_d.pow(2) * inter_cross
    inter_bev = torch.minimum(inter_bev, torch.minimum(area1, area2)).clamp(min=0.0)
    return inter_bev

def oriented_iou_3d(center_pred, lwh_pred, yaw_pred,
                    center_gt, lwh_gt, yaw_gt, eps=1e-7):
    """
    Computes orientation-aware 3D bounding box IoU:
    IoU_3D = (A_I * H_I) / (V_p + V_g - A_I * H_I + eps)
    Strictly bounded in [0, 1] and symmetric under argument permutation.
    """
    if yaw_pred.ndim > 1:
        yaw_pred = yaw_pred.squeeze(-1)
    if yaw_gt.ndim > 1:
        yaw_gt = yaw_gt.squeeze(-1)

    # 1. Vertical height overlap along Z
    z_min_p = center_pred[..., 2] - lwh_pred[..., 2] * 0.5
    z_max_p = center_pred[..., 2] + lwh_pred[..., 2] * 0.5
    z_min_g = center_gt[..., 2] - lwh_gt[..., 2] * 0.5
    z_max_g = center_gt[..., 2] + lwh_gt[..., 2] * 0.5

    h_overlap = (torch.minimum(z_max_p, z_max_g) - torch.maximum(z_min_p, z_min_g)).clamp(min=0)

    # 2. Rotated BEV area intersection
    inter_bev = rotated_rect_intersection_bev(
        center_pred, lwh_pred, yaw_pred,
        center_gt, lwh_gt, yaw_gt, eps=eps
    )

    # 3. 3D intersection volume
    vol_inter = inter_bev * h_overlap

    # 4. Box volumes
    vol_p = (lwh_pred[..., 0] * lwh_pred[..., 1] * lwh_pred[..., 2]).clamp_min(eps)
    vol_g = (lwh_gt[..., 0] * lwh_gt[..., 1] * lwh_gt[..., 2]).clamp_min(eps)

    # 5. Union and IoU
    union = vol_p + vol_g - vol_inter + eps
    iou_3d = (vol_inter / union).clamp(0.0, 1.0)
    return iou_3d
