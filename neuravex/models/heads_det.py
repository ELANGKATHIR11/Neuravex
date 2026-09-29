import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ConvBNAct
from ..geometry.box_ops import box_cxcywh_to_xyxy
from ..geometry.oriented_iou3d import rotation_6d_to_matrix, matrix_to_yaw

class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss (DFL) module for fine-grained sub-pixel coordinate regression.
    Supports both:
      - reg_max > 1: Discrete distribution bins with expectation E[x] = sum(p_i * i)
      - reg_max == 1: Direct coordinate regression (no DFL projection overhead)
    """
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        if reg_max > 1:
            self.register_buffer("project", torch.linspace(0, reg_max - 1, reg_max))
        else:
            self.register_buffer("project", torch.tensor([1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, 4 * reg_max) or (B, N, 4)
        Returns: (B, N, 4) continuous distances
        """
        if self.reg_max == 1:
            # Direct regression
            return F.relu(x.view(x.shape[0], x.shape[1], 4))

        B, N, C = x.shape
        # Softmax over the reg_max distribution
        x_reshaped = x.view(B, N, 4, self.reg_max).softmax(dim=-1)
        # Expectation E[x] = sum(p_i * i)
        out = (x_reshaped * self.project.to(x.device, dtype=x.dtype)).sum(dim=-1)
        return out

class MultiScaleDetectionHead(nn.Module):
    """
    Decoupled multi-scale anchor-free detection and 3D head across P3, P4, P5 (strides: 8, 16, 32).
    Upgraded for Neuravex 2.0:
      1. Dual-Label Assignment & NMS-Free Head:
         - One-to-Many (O2M) branch with TAL assignment for rich gradient flow during training.
         - One-to-One (O2I) branch with Hungarian matching for zero-NMS deterministic edge inference.
      2. Continuous 6D Orientation Representation (SO(3) via Gram-Schmidt orthogonalization).
      3. Instance Heteroscedastic Uncertainty (per-anchor log_sigma_xyz and log_sigma_lwh).
      4. DEM-to-3D Depth Cross-Gating: Ground-conditioned depth injection from dense metric DEM.
    """
    def __init__(self, in_channels: int, num_classes: int = 80, strides=(8, 16, 32), reg_max: int = 16,
                 rot_dim: int = 6, num_prototypes: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max
        self.rot_dim = rot_dim
        self.num_prototypes = num_prototypes
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

        # Primary One-to-One (O2I) Projections per scale (Active in Training & Deployment)
        self.cls_preds = nn.ModuleList([nn.Conv2d(in_channels, num_classes, 1) for _ in strides])
        self.box_preds = nn.ModuleList([nn.Conv2d(in_channels, 4 * reg_max, 1) for _ in strides])

        # Auxiliary One-to-Many (O2M) Projections per scale (Active ONLY during training)
        self.cls_preds_o2m = nn.ModuleList([nn.Conv2d(in_channels, num_classes, 1) for _ in strides])
        self.box_preds_o2m = nn.ModuleList([nn.Conv2d(in_channels, 4 * reg_max, 1) for _ in strides])

        # 3D heads:
        # Center offset (3), log_lwh (3), continuous 6D rotation (6)
        self.pred_3d_xyz = nn.ModuleList([nn.Conv2d(in_channels, 3, 1) for _ in strides])
        self.pred_3d_lwh = nn.ModuleList([nn.Conv2d(in_channels, 3, 1) for _ in strides])
        self.pred_3d_yaw = nn.ModuleList([nn.Conv2d(in_channels, rot_dim, 1) for _ in strides])

        # Instance-adaptive heteroscedastic uncertainty heads (log variance per anchor):
        # 3 channels for xyz variance, 3 channels for lwh variance
        self.pred_unc_xyz = nn.ModuleList([nn.Conv2d(in_channels, 3, 1) for _ in strides])
        self.pred_unc_lwh = nn.ModuleList([nn.Conv2d(in_channels, 3, 1) for _ in strides])

        # Per-detection mask coefficients for YOLACT-style instance mask assembly:
        # Each anchor predicts num_prototypes coefficients that linearly combine proto_masks
        # mask_i = sigmoid(coeffs_i @ proto_masks)  →  one independent mask per detection
        self.mask_coeff_preds = nn.ModuleList([
            nn.Conv2d(in_channels, num_prototypes, 1) for _ in strides
        ])

        # Optional DEM depth cross-gating projection: aligns DEM depth prior with 3D regression
        self.dem_depth_gate = nn.Sequential(
            ConvBNAct(1, 16, 3),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        # Global fallbacks for backward compatibility
        self.log_sigma_xyz = nn.Parameter(torch.zeros(3))
        self.log_sigma_lwh = nn.Parameter(torch.zeros(3))
        self.log_sigma_yaw = nn.Parameter(torch.zeros(1))

        self._init_biases()

    def _init_biases(self):
        prior_prob = 0.01
        bias_init = float(-torch.log(torch.tensor((1 - prior_prob) / prior_prob)))
        for conv in list(self.cls_preds) + list(self.cls_preds_o2m):
            nn.init.constant_(conv.bias, bias_init)

    def _decode_boxes(self, pred_box_dist, anchor_points, strides_cat):
        box_ltrb = self.dfl(pred_box_dist) * strides_cat.unsqueeze(0)
        x1 = anchor_points[:, 0:1].unsqueeze(0) - box_ltrb[..., 0:1]
        y1 = anchor_points[:, 1:2].unsqueeze(0) - box_ltrb[..., 1:2]
        x2 = anchor_points[:, 0:1].unsqueeze(0) + box_ltrb[..., 2:3]
        y2 = anchor_points[:, 1:2].unsqueeze(0) + box_ltrb[..., 3:4]
        return torch.cat([x1, y1, x2, y2], dim=-1)

    def forward(self, feats: list, intrinsics = None, dem_depth_map: torch.Tensor = None,
                return_o2m: bool = None, compute_3d: bool = True):
        """
        feats: list of [q3, q4, q5]
        intrinsics: CameraIntrinsics for pinhole unprojection
        dem_depth_map: (B, 1, H, W) metric depth map from CameraAwareDEM for cross-gating
        return_o2m: If True (or during training by default), returns auxiliary O2M head predictions
        compute_3d: If False, completely skips 3D and DEM cross-gating for zero-overhead 2D detection
        """
        if return_o2m is None:
            return_o2m = self.training

        all_cls = []
        all_box_dist = []
        all_cls_o2m = []
        all_box_dist_o2m = []
        all_xyz = []
        all_lwh = []
        all_yaw = []
        all_unc_xyz = []
        all_unc_lwh = []
        all_mask_coeffs = []
        all_anchors = []
        all_strides = []
        all_dem_priors = []

        for i, (f, s) in enumerate(zip(feats, self.strides)):
            B, C, H, W = f.shape
            c_feat = self.cls_convs[i](f)
            r_feat = self.reg_convs[i](f)

            # 1. Primary One-to-One Head (O2I)
            cls_out = self.cls_preds[i](c_feat).permute(0, 2, 3, 1).reshape(B, H * W, self.num_classes)
            box_dist_out = self.box_preds[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, 4 * self.reg_max)

            # Per-detection mask coefficients for instance mask assembly
            mask_coeff_out = self.mask_coeff_preds[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, self.num_prototypes)

            all_cls.append(cls_out)
            all_box_dist.append(box_dist_out)
            all_mask_coeffs.append(mask_coeff_out)

            # 2. Auxiliary One-to-Many Head (O2M)
            if return_o2m:
                cls_o2m = self.cls_preds_o2m[i](c_feat).permute(0, 2, 3, 1).reshape(B, H * W, self.num_classes)
                box_dist_o2m = self.box_preds_o2m[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, 4 * self.reg_max)
                all_cls_o2m.append(cls_o2m)
                all_box_dist_o2m.append(box_dist_o2m)

            # 3. 3D Head Predictions & Instance Heteroscedastic Uncertainty (conditional)
            if compute_3d:
                xyz_out = self.pred_3d_xyz[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, 3)
                lwh_out = self.pred_3d_lwh[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, 3)
                yaw_out = self.pred_3d_yaw[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, self.rot_dim)

                unc_xyz_out = self.pred_unc_xyz[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, 3)
                unc_lwh_out = self.pred_unc_lwh[i](r_feat).permute(0, 2, 3, 1).reshape(B, H * W, 3)

                all_xyz.append(xyz_out)
                all_lwh.append(lwh_out)
                all_yaw.append(yaw_out)
                all_unc_xyz.append(unc_xyz_out)
                all_unc_lwh.append(unc_lwh_out)

            # Spatial grid & anchors
            yv, xv = torch.meshgrid(
                torch.arange(H, device=f.device, dtype=f.dtype),
                torch.arange(W, device=f.device, dtype=f.dtype),
                indexing="ij"
            )
            grid_points = torch.stack([(xv + 0.5) * s, (yv + 0.5) * s], dim=-1).reshape(H * W, 2)
            stride_tensor = torch.full((H * W, 1), s, device=f.device, dtype=f.dtype)

            all_anchors.append(grid_points)
            all_strides.append(stride_tensor)

            # DEM depth cross-gating sample at scale
            if compute_3d and dem_depth_map is not None:
                dem_down = F.interpolate(dem_depth_map, size=(H, W), mode="bilinear", align_corners=False)
                gate = self.dem_depth_gate(dem_down)
                dem_prior = (dem_down * gate).permute(0, 2, 3, 1).reshape(B, H * W, 1)
                all_dem_priors.append(dem_prior)

        pred_cls = torch.cat(all_cls, dim=1)
        pred_box_dist = torch.cat(all_box_dist, dim=1)
        anchor_points = torch.cat(all_anchors, dim=0)
        strides_cat = torch.cat(all_strides, dim=0)

        # 1. Decode 2D boxes for primary O2I head
        decoded_boxes = self._decode_boxes(pred_box_dist, anchor_points, strides_cat)

        # 2. Camera-aware 3D decoding with DEM Cross-Gating
        decoded_xyz = None
        decoded_lwh = None
        pred_yaw_sincos = None
        pred_rot_matrix = None
        pred_unc_xyz = None
        pred_unc_lwh = None

        if compute_3d and len(all_xyz) > 0:
            pred_xyz_raw = torch.cat(all_xyz, dim=1)
            pred_lwh_raw = torch.cat(all_lwh, dim=1)
            pred_yaw_raw = torch.cat(all_yaw, dim=1)
            pred_unc_xyz = torch.cat(all_unc_xyz, dim=1)
            pred_unc_lwh = torch.cat(all_unc_lwh, dim=1)

            if len(all_dem_priors) > 0:
                dem_priors_cat = torch.cat(all_dem_priors, dim=1)
                # Physical anchor ray depth: Z = Z_dem * (1.0 + 0.5 * tanh(delta_z))
                delta_z = torch.tanh(pred_xyz_raw[..., 2:3])
                z_depth = (dem_priors_cat * (1.0 + 0.5 * delta_z)).clamp(min=0.05, max=1000.0)
            else:
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

            # Continuous 6D Orientation Matrix & Extracted Yaw
            if self.rot_dim == 6:
                pred_rot_matrix = rotation_6d_to_matrix(pred_yaw_raw)
                extracted_yaw = matrix_to_yaw(pred_rot_matrix).unsqueeze(-1)
                pred_yaw_sincos = torch.cat([torch.sin(extracted_yaw), torch.cos(extracted_yaw)], dim=-1)
            else:
                pred_rot_matrix = None
                pred_yaw_sincos = pred_yaw_raw

        # 4. Energy-Based Unsupervised Density Estimation (OOD Detection)
        temperature = 1.0
        free_energy = -temperature * torch.logsumexp(pred_cls / temperature, dim=-1, keepdim=True)
        ood_score = torch.sigmoid(free_energy - 5.0)

        # Concatenate mask coefficients across all scales
        pred_mask_coeffs = torch.cat(all_mask_coeffs, dim=1) if len(all_mask_coeffs) > 0 else None

        out = {
            # Primary One-to-One Head (Zero-NMS Deployment outputs)
            "class_logits": pred_cls,
            "pred_box_dist": pred_box_dist,
            "pred_boxes": decoded_boxes,
            "pred_mask_coeffs": pred_mask_coeffs,
            "anchor_points": anchor_points,
            "strides": strides_cat,
            "free_energy": free_energy,
            "ood_score": ood_score,
        }

        if compute_3d and decoded_xyz is not None:
            out.update({
                "pred_xyz": decoded_xyz,
                "pred_lwh": decoded_lwh,
                "pred_yaw_sincos": pred_yaw_sincos,
                "pred_rot_matrix": pred_rot_matrix,
                "pred_6d_rot": pred_yaw_raw if self.rot_dim == 6 else None,
                "log_sigma_xyz": pred_unc_xyz.clamp(-5.0, 5.0),
                "log_sigma_lwh": pred_unc_lwh.clamp(-5.0, 5.0),
                "log_sigma_yaw": self.log_sigma_yaw.clamp(-5.0, 5.0)
            })

        # 5. Include O2M head predictions if active
        if return_o2m and len(all_cls_o2m) > 0:
            pred_cls_o2m = torch.cat(all_cls_o2m, dim=1)
            pred_box_dist_o2m = torch.cat(all_box_dist_o2m, dim=1)
            decoded_boxes_o2m = self._decode_boxes(pred_box_dist_o2m, anchor_points, strides_cat)
            out["o2m_preds"] = {
                "class_logits": pred_cls_o2m,
                "pred_box_dist": pred_box_dist_o2m,
                "pred_boxes": decoded_boxes_o2m
            }

        return out


