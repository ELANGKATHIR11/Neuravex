import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ConvBNAct
from ..geometry.box_ops import box_cxcywh_to_xyxy

class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss (DFL) module for fine-grained sub-pixel coordinate regression.
    Converts reg_max discrete bins into continuous offset coordinates.
    """
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("project", torch.linspace(0, reg_max - 1, reg_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, 4 * reg_max)
        Returns: (B, N, 4) continuous distances
        """
        B, N, C = x.shape
        # Softmax over the reg_max distribution
        x_reshaped = x.view(B, N, 4, self.reg_max).softmax(dim=-1)
        # Expectation E[x] = sum(p_i * i)
        out = (x_reshaped * self.project.to(x.device)).sum(dim=-1)
        return out

class MultiScaleDetectionHead(nn.Module):
    """
    Decoupled multi-scale anchor-free detection and 3D head across P3, P4, P5 (strides: 8, 16, 32).
    Outputs:
      - 2D: class logits, box regression (with DFL distribution bins)
      - 3D: (X, Y, Z) center with camera pinhole geometry (Z=exp(z), X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy),
            (L, W, H) log dimensions, yaw (sin theta, cos theta)
    """
    def __init__(self, in_channels: int, num_classes: int = 80, strides=(8, 16, 32), reg_max: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max
        self.dfl = DistributionFocalLoss(reg_max=reg_max)

        # Multi-scale decoupled convs
        self.cls_convs = nn.ModuleList([
            nn.Sequential(ConvBNAct(in_channels, in_channels, 3), ConvBNAct(in_channels, in_channels, 3))
            for _ in strides
        ])
        self.reg_convs = nn.ModuleList([
            nn.Sequential(ConvBNAct(in_channels, in_channels, 3), ConvBNAct(in_channels, in_channels, 3))
            for _ in strides
        ])

        # Projections per scale
        self.cls_preds = nn.ModuleList([nn.Conv2d(in_channels, num_classes, 1) for _ in strides])
        # Box predicts 4 * reg_max distribution bins
        self.box_preds = nn.ModuleList([nn.Conv2d(in_channels, 4 * reg_max, 1) for _ in strides])
        
        # 3D heads: center offset (3), log_lwh (3), yaw_sincos (2)
        self.pred_3d_xyz = nn.ModuleList([nn.Conv2d(in_channels, 3, 1) for _ in strides])
        self.pred_3d_lwh = nn.ModuleList([nn.Conv2d(in_channels, 3, 1) for _ in strides])
        self.pred_3d_yaw = nn.ModuleList([nn.Conv2d(in_channels, 2, 1) for _ in strides])

        # Bounded uncertainty parameters for 3D
        self.log_sigma_xyz = nn.Parameter(torch.zeros(3))
        self.log_sigma_lwh = nn.Parameter(torch.zeros(3))
        self.log_sigma_yaw = nn.Parameter(torch.zeros(1))

        self._init_biases()

    def _init_biases(self):
        prior_prob = 0.01
        bias_init = float(-torch.log(torch.tensor((1 - prior_prob) / prior_prob)))
        for conv in self.cls_preds:
            nn.init.constant_(conv.bias, bias_init)

    def forward(self, feats: list, intrinsics = None):
        """
        feats: list of [q3, q4, q5]
        """
        all_cls = []
        all_box_dist = []
        all_xyz = []
        all_lwh = []
        all_yaw = []
        all_anchors = []
        all_strides = []

        for i, (f, s) in enumerate(zip(feats, self.strides)):
            B, C, H, W = f.shape
            c_feat = self.cls_convs[i](f)
            r_feat = self.reg_convs[i](f)

            cls_out = self.cls_preds[i](c_feat).permute(0, 2, 3, 1).reshape(B, H * W, self.num_classes)
            box_dist_out = self.box_preds[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, 4 * self.reg_max)

            xyz_out = self.pred_3d_xyz[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, 3)
            lwh_out = self.pred_3d_lwh[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, 3)
            yaw_out = self.pred_3d_yaw[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, 2)

            yv, xv = torch.meshgrid(
                torch.arange(H, device=f.device, dtype=f.dtype),
                torch.arange(W, device=f.device, dtype=f.dtype),
                indexing="ij"
            )
            grid_points = torch.stack([(xv + 0.5) * s, (yv + 0.5) * s], dim=-1).reshape(H * W, 2)
            stride_tensor = torch.full((H * W, 1), s, device=f.device, dtype=f.dtype)

            all_cls.append(cls_out)
            all_box_dist.append(box_dist_out)
            all_xyz.append(xyz_out)
            all_lwh.append(lwh_out)
            all_yaw.append(yaw_out)
            all_anchors.append(grid_points)
            all_strides.append(stride_tensor)

        pred_cls = torch.cat(all_cls, dim=1)
        pred_box_dist = torch.cat(all_box_dist, dim=1)
        pred_xyz_raw = torch.cat(all_xyz, dim=1)
        pred_lwh_raw = torch.cat(all_lwh, dim=1)
        pred_yaw_raw = torch.cat(all_yaw, dim=1)
        anchor_points = torch.cat(all_anchors, dim=0)
        strides_cat = torch.cat(all_strides, dim=0)

        # 1. Decode 2D boxes via DFL integration
        box_ltrb = self.dfl(pred_box_dist) * strides_cat.unsqueeze(0)  # (B, N, 4) in l, t, r, b
        x1 = anchor_points[:, 0:1].unsqueeze(0) - box_ltrb[..., 0:1]
        y1 = anchor_points[:, 1:2].unsqueeze(0) - box_ltrb[..., 1:2]
        x2 = anchor_points[:, 0:1].unsqueeze(0) + box_ltrb[..., 2:3]
        y2 = anchor_points[:, 1:2].unsqueeze(0) + box_ltrb[..., 3:4]
        decoded_boxes = torch.cat([x1, y1, x2, y2], dim=-1)

        # 2. Camera-aware 3D decoding:
        # Z = exp(z_raw) strictly positive; X = (u - cx)*Z/fx, Y = (v - cy)*Z/fy
        z_depth = torch.exp(pred_xyz_raw[..., 2:3].clamp(-4.0, 4.0)) + 0.1
        u_anc = anchor_points[:, 0:1].unsqueeze(0) + pred_xyz_raw[..., 0:1] * strides_cat.unsqueeze(0)
        v_anc = anchor_points[:, 1:2].unsqueeze(0) + pred_xyz_raw[..., 1:2] * strides_cat.unsqueeze(0)

        if intrinsics is not None:
            fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
            x_cam = (u_anc - cx) * z_depth / fx
            y_cam = (v_anc - cy) * z_depth / fy
        else:
            x_cam = u_anc
            y_cam = v_anc

        decoded_xyz = torch.cat([x_cam, y_cam, z_depth], dim=-1)
        decoded_lwh = torch.exp(pred_lwh_raw.clamp(-4.0, 4.0))

        return {
            "class_logits": pred_cls,
            "pred_box_dist": pred_box_dist,
            "pred_boxes": decoded_boxes,
            "pred_xyz": decoded_xyz,
            "pred_lwh": decoded_lwh,
            "pred_yaw_sincos": pred_yaw_raw,
            "anchor_points": anchor_points,
            "strides": strides_cat,
            "log_sigma_xyz": self.log_sigma_xyz.clamp(-5.0, 5.0),
            "log_sigma_lwh": self.log_sigma_lwh.clamp(-5.0, 5.0),
            "log_sigma_yaw": self.log_sigma_yaw.clamp(-5.0, 5.0)
        }
