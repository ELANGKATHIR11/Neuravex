import torch
import torch.nn as nn
import torch.nn.functional as F
from ..geometry.oriented_iou3d import oriented_iou_3d

def scale_invariant_log_depth_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Scale-invariant log depth loss (SILog, Eigen et al.):
    L_silog = 1/T sum d_i^2 - 0.5 * (1/T sum d_i)^2
    """
    p = torch.log(pred_depth.clamp_min(eps))
    g = torch.log(gt_depth.clamp_min(eps))
    v = valid_mask.bool()
    if v.sum() == 0:
        return pred_depth.sum() * 0.0

    d = (p - g)[v]
    return (d * d).mean() - 0.5 * (d.mean().pow(2))

def depth_gradient_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """
    Edge-preserving depth gradient loss:
    L_grad = sum (|dx(P) - dx(G)| + |dy(P) - dy(G)|) on valid pixels
    """
    v = valid_mask.float()
    v_x = v[:, :, :, :-1] * v[:, :, :, 1:]
    v_y = v[:, :, :-1, :] * v[:, :, 1:, :]

    dx_p = pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1]
    dx_g = gt_depth[:, :, :, 1:] - gt_depth[:, :, :, :-1]
    dy_p = pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :]
    dy_g = gt_depth[:, :, 1:, :] - gt_depth[:, :, :-1, :]

    loss_x = (torch.abs(dx_p - dx_g) * v_x).sum() / (v_x.sum() + 1e-6)
    loss_y = (torch.abs(dy_p - dy_g) * v_y).sum() / (v_y.sum() + 1e-6)
    return loss_x + loss_y

def comprehensive_metric_depth_loss(pred_depth: torch.Tensor, gt_depth: torch.Tensor, valid_mask: torch.Tensor,
                                    lambda_silog: float = 1.0, lambda_abs: float = 0.5, lambda_grad: float = 0.5) -> torch.Tensor:
    """
    Full metric depth supervision:
    L_depth = lambda_silog * L_silog + lambda_abs * L_abs + lambda_grad * L_grad
    """
    v = valid_mask.bool()
    if v.sum() == 0:
        return pred_depth.sum() * 0.0

    l_silog = scale_invariant_log_depth_loss(pred_depth, gt_depth, valid_mask)
    l_abs = F.l1_loss(pred_depth[v], gt_depth[v])
    l_grad = depth_gradient_loss(pred_depth, gt_depth, valid_mask)

    return lambda_silog * l_silog + lambda_abs * l_abs + lambda_grad * l_grad

def loss_3d_detection(pred_xyz: torch.Tensor, pred_lwh: torch.Tensor, pred_yaw_sincos: torch.Tensor,
                      gt_xyz: torch.Tensor, gt_lwh: torch.Tensor, gt_yaw: torch.Tensor,
                      pos_mask: torch.Tensor,
                      log_sigma_xyz: torch.Tensor = None,
                      log_sigma_lwh: torch.Tensor = None,
                      log_sigma_yaw: torch.Tensor = None) -> tuple:
    """
    Object-specific 3D supervision on assigned positive detections:
    L_3D = L_xyz + L_lwh + L_theta + L_IoU3D
    """
    if pos_mask.sum() == 0:
        zero = (pred_xyz.sum() + pred_lwh.sum() + pred_yaw_sincos.sum()) * 0.0
        return zero, {"loss_xyz": zero, "loss_lwh": zero, "loss_yaw": zero, "loss_iou3d": zero}

    p_xyz = pred_xyz[pos_mask]
    g_xyz = gt_xyz[pos_mask]
    p_lwh = pred_lwh[pos_mask]
    g_lwh = gt_lwh[pos_mask]
    p_yaw_sc = pred_yaw_sincos[pos_mask]
    g_yaw = gt_yaw[pos_mask]
    if g_yaw.ndim == 1:
        g_yaw = g_yaw.unsqueeze(-1)

    # 1. Center XYZ loss
    if log_sigma_xyz is not None:
        sigma_xyz = torch.exp(log_sigma_xyz.clamp(-5.0, 5.0))
        loss_xyz = (F.smooth_l1_loss(p_xyz, g_xyz, reduction="none") / sigma_xyz + log_sigma_xyz).mean()
    else:
        loss_xyz = F.smooth_l1_loss(p_xyz, g_xyz)

    # 2. Dimensions LWH loss in log space
    log_p_lwh = torch.log(p_lwh.clamp_min(1e-4))
    log_g_lwh = torch.log(g_lwh.clamp_min(1e-4))
    if log_sigma_lwh is not None:
        sigma_lwh = torch.exp(log_sigma_lwh.clamp(-5.0, 5.0))
        loss_lwh = (F.smooth_l1_loss(log_p_lwh, log_g_lwh, reduction="none") / sigma_lwh + log_sigma_lwh).mean()
    else:
        loss_lwh = F.smooth_l1_loss(log_p_lwh, log_g_lwh)

    # 3. Periodic Yaw loss with normalized sin/cos
    p_sc_norm = F.normalize(p_yaw_sc, dim=-1)
    g_sc = torch.cat([torch.sin(g_yaw), torch.cos(g_yaw)], dim=-1)
    loss_yaw_cos = (1.0 - (p_sc_norm * g_sc).sum(dim=-1)).mean()
    recovered_pred_yaw = torch.atan2(p_sc_norm[:, 0], p_sc_norm[:, 1])

    # 4. Oriented 3D IoU loss
    iou_3d = oriented_iou_3d(
        p_xyz, p_lwh, recovered_pred_yaw,
        g_xyz, g_lwh, g_yaw.squeeze(-1)
    )
    loss_iou3d = (1.0 - iou_3d).mean()

    total_3d = loss_xyz + loss_lwh + loss_yaw_cos + loss_iou3d
    metrics = {
        "loss_xyz": loss_xyz.detach(),
        "loss_lwh": loss_lwh.detach(),
        "loss_yaw": loss_yaw_cos.detach(),
        "loss_iou3d": loss_iou3d.detach(),
        "mean_iou3d": iou_3d.mean().detach()
    }
    return total_3d, metrics
