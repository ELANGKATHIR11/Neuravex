import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ConvBNAct
from ..geometry.box_ops import box_cxcywh_to_xyxy

class MultiScaleDetectionHead(nn.Module):
    """
    Decoupled multi-scale anchor-free detection and 3D head across P3, P4, P5.
    Strides: [8, 16, 32].
    Outputs:
      - 2D: class logits, box regression (cx, cy, w, h in stride/feature scale)
      - 3D: (X, Y, Z) center offsets, (L, W, H) log dimensions, yaw (sin theta, cos theta)
    """
    def __init__(self, in_channels: int, num_classes: int = 80, strides=(8, 16, 32)):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.strides = strides

        # Multi-scale shared heads
        self.cls_convs = nn.ModuleList([
            nn.Sequential(ConvBNAct(in_channels, in_channels, 3), ConvBNAct(in_channels, in_channels, 3))
            for _ in strides
        ])
        self.reg_convs = nn.ModuleList([
            nn.Sequential(ConvBNAct(in_channels, in_channels, 3), ConvBNAct(in_channels, in_channels, 3))
            for _ in strides
        ])

        # Prediction projections per scale
        self.cls_preds = nn.ModuleList([nn.Conv2d(in_channels, num_classes, 1) for _ in strides])
        self.box_preds = nn.ModuleList([nn.Conv2d(in_channels, 4, 1) for _ in strides])
        
        # 3D heads: center offset (3), log_lwh (3), yaw_sincos (2)
        self.pred_3d_xyz = nn.ModuleList([nn.Conv2d(in_channels, 3, 1) for _ in strides])
        self.pred_3d_lwh = nn.ModuleList([nn.Conv2d(in_channels, 3, 1) for _ in strides])
        self.pred_3d_yaw = nn.ModuleList([nn.Conv2d(in_channels, 2, 1) for _ in strides])

        # Uncertainty parameters for 3D outputs
        self.log_sigma_xyz = nn.Parameter(torch.zeros(3))
        self.log_sigma_lwh = nn.Parameter(torch.zeros(3))
        self.log_sigma_yaw = nn.Parameter(torch.zeros(1))

        self._init_biases()

    def _init_biases(self):
        # Prior prob bias init for classification to avoid early instability
        prior_prob = 0.01
        bias_init = float(-torch.log(torch.tensor((1 - prior_prob) / prior_prob)))
        for conv in self.cls_preds:
            nn.init.constant_(conv.bias, bias_init)

    def forward(self, feats: list):
        """
        feats: list of [q3, q4, q5]
        Returns dict of concatenated multi-scale flattened predictions and anchor grids.
        """
        all_cls = []
        all_boxes = []
        all_xyz = []
        all_lwh = []
        all_yaw = []
        all_anchors = []
        all_strides = []

        for i, (f, s) in enumerate(zip(feats, self.strides)):
            B, C, H, W = f.shape
            c_feat = self.cls_convs[i](f)
            r_feat = self.reg_convs[i](f)

            # Class logits: (B, num_classes, H, W) -> (B, H*W, num_classes)
            cls_out = self.cls_preds[i](c_feat).permute(0, 2, 3, 1).reshape(B, H * W, self.num_classes)
            # Box raw predictions: (B, 4, H, W) -> (B, H*W, 4)
            box_raw = self.box_preds[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, 4)

            # 3D predictions
            xyz_out = self.pred_3d_xyz[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, 3)
            lwh_out = self.pred_3d_lwh[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, 3)
            yaw_out = self.pred_3d_yaw[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, 2)

            # Anchor grid generator for this scale
            yv, xv = torch.meshgrid(
                torch.arange(H, device=f.device, dtype=f.dtype),
                torch.arange(W, device=f.device, dtype=f.dtype),
                indexing="ij"
            )
            # Center of the cell in input image coordinates
            grid_points = torch.stack([(xv + 0.5) * s, (yv + 0.5) * s], dim=-1).reshape(H * W, 2)
            stride_tensor = torch.full((H * W, 1), s, device=f.device, dtype=f.dtype)

            all_cls.append(cls_out)
            all_boxes.append(box_raw)
            all_xyz.append(xyz_out)
            all_lwh.append(lwh_out)
            all_yaw.append(yaw_out)
            all_anchors.append(grid_points)
            all_strides.append(stride_tensor)

        # Concatenate across scales
        pred_cls = torch.cat(all_cls, dim=1)           # (B, N_anchors, num_classes)
        pred_box_raw = torch.cat(all_boxes, dim=1)     # (B, N_anchors, 4)
        pred_xyz_raw = torch.cat(all_xyz, dim=1)       # (B, N_anchors, 3)
        pred_lwh_raw = torch.cat(all_lwh, dim=1)       # (B, N_anchors, 3)
        pred_yaw_raw = torch.cat(all_yaw, dim=1)       # (B, N_anchors, 2)
        anchor_points = torch.cat(all_anchors, dim=0)  # (N_anchors, 2)
        strides_cat = torch.cat(all_strides, dim=0)    # (N_anchors, 1)

        # Decode 2D boxes:
        # pred_box_raw contains: (dx, dy, w_rel, h_rel)
        cx = anchor_points[:, 0:1].unsqueeze(0) + pred_box_raw[..., 0:1] * strides_cat.unsqueeze(0)
        cy = anchor_points[:, 1:2].unsqueeze(0) + pred_box_raw[..., 1:2] * strides_cat.unsqueeze(0)
        bw = F.softplus(pred_box_raw[..., 2:3]) * strides_cat.unsqueeze(0)
        bh = F.softplus(pred_box_raw[..., 3:4]) * strides_cat.unsqueeze(0)
        decoded_boxes = box_cxcywh_to_xyxy(torch.cat([cx, cy, bw, bh], dim=-1))

        # Decode 3D boxes:
        # Z center is strictly positive
        pred_z = F.softplus(pred_xyz_raw[..., 2:3]) + 0.1
        pred_x = pred_xyz_raw[..., 0:1]
        pred_y = pred_xyz_raw[..., 1:2]
        decoded_xyz = torch.cat([pred_x, pred_y, pred_z], dim=-1)
        decoded_lwh = torch.exp(pred_lwh_raw.clamp(-4.0, 4.0))  # positive dimensions

        return {
            "class_logits": pred_cls,
            "pred_boxes": decoded_boxes,
            "pred_xyz": decoded_xyz,
            "pred_lwh": decoded_lwh,
            "pred_yaw_sincos": pred_yaw_raw,
            "anchor_points": anchor_points,
            "strides": strides_cat,
            "log_sigma_xyz": self.log_sigma_xyz,
            "log_sigma_lwh": self.log_sigma_lwh,
            "log_sigma_yaw": self.log_sigma_yaw
        }
