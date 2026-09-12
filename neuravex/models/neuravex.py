import torch
import torch.nn as nn
from .backbone import Backbone
from .neck import PANetNeck, BidirectionalCrossTaskFusion
from .heads_det import MultiScaleDetectionHead
from .heads_seg import MultiLayerSegmentationHead
from .heads_depth import CameraAwareDEM
from ..geometry.camera import CameraIntrinsics

class Neuravex(nn.Module):
    """
    Neuravex v0.6 — Lightweight Custom Computer Vision Architecture.
    Direct next-generation competitor to the YOLO family, unifying 2D/3D and Dense Multitask CV:
      1. 2D Multi-scale anchor-free detection (P3, P4, P5 at strides 8, 16, 32) with DFL
      2. 3D detection: (X, Y, Z), (L, W, H), yaw (sin theta, cos theta) with pinhole geometry
      3. Semantic segmentation (multiclass raw logits)
      4. Boundary segmentation (raw logits for BCE/Dice)
      5. Instance discriminative embeddings + prototype masks
      6. Mask quality prediction (continuous IoU supervision)
      7. Dense inverse depth & metric depth (DEM)
      8. True bidirectional cross-task feature fusion with dedicated task branches
    """
    def __init__(self, num_classes: int = 80, base_c: int = 48, depth_mul: float = 1.0, seg_embed: int = 16, num_parts: int = 16):
        super().__init__()
        self.num_classes = num_classes
        self.base_c = base_c
        self.depth_mul = depth_mul
        neck_c = base_c * 4

        # Backbone & FPN Neck
        self.backbone = Backbone(base_c=base_c, depth_mul=depth_mul)
        self.neck = PANetNeck(base_c=base_c)

        # Cross-Task Fusion at multi-scale
        self.fusion_p3 = BidirectionalCrossTaskFusion(neck_c)

        # Task Heads
        self.det_head = MultiScaleDetectionHead(in_channels=neck_c, num_classes=num_classes)
        self.seg_head = MultiLayerSegmentationHead(in_channels=neck_c, num_classes=num_classes, embed_dim=seg_embed, num_parts=num_parts)
        self.dem_head = CameraAwareDEM(in_channels=neck_c)

    def forward(self, x: torch.Tensor, intrinsics: CameraIntrinsics = None) -> dict:
        B, _, H, W = x.shape
        out_hw = (H, W)

        # 1. Multi-scale feature extraction
        p3, p4, p5 = self.backbone(x)
        q3, q4, q5 = self.neck(p3, p4, p5)

        # 2. True Bidirectional Cross-Task Fusion across task tokens
        q3_det, q3_seg, q3_dep = self.fusion_p3(q3, q3, q3)

        # 3. Dense depth prediction (consuming q3_dep)
        depth_out = self.dem_head(q3_dep, q4, q5, out_hw, intrinsics=intrinsics)

        # 4. Dense multi-layer segmentation (consuming q3_seg)
        seg_out = self.seg_head(q3_seg, q4, q5, out_hw)

        # 5. Multi-scale detection and 3D prediction (consuming q3_det)
        det_feats = [q3_det, q4, q5]
        det_out = self.det_head(det_feats, intrinsics=intrinsics)

        # Merge outputs into unified dictionary
        outputs = {}
        outputs.update(det_out)
        outputs.update(seg_out)
        outputs.update(depth_out)
        return outputs

# Alias for backward-compatibility
YOLO27 = Neuravex

def build_neuravex(size: str = "medium", num_classes: int = 80) -> Neuravex:
    """
    Factory function for scalable Neuravex variants:
      nano: base_c = 16, depth_mul = 0.33
      small: base_c = 32, depth_mul = 0.67
      medium: base_c = 48, depth_mul = 1.0
      large: base_c = 64, depth_mul = 1.33
    """
    configs = {
        "nano": (16, 0.33),
        "small": (32, 0.67),
        "medium": (48, 1.0),
        "large": (64, 1.33)
    }
    base, d_mul = configs.get(size.lower(), (48, 1.0))
    return Neuravex(num_classes=num_classes, base_c=base, depth_mul=d_mul)

build_yolo27 = build_neuravex
