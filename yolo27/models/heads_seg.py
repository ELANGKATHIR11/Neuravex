import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ConvBNAct, RepBlock

class MultiLayerSegmentationHead(nn.Module):
    """
    Multi-layer dense segmentation head outputting:
      - semantic_masks: raw class logits (B, num_classes, H, W)
      - boundary_map: raw boundary logits (B, 1, H, W)
      - instance_embeddings: unit-normalized discriminative embeddings (B, embed_dim, H, W)
      - mask_quality: raw quality prediction logits (B, 1, H, W)
      - part_masks: optional part segmentation logits (B, parts, H, W)
    """
    def __init__(self, in_channels: int, num_classes: int = 80, embed_dim: int = 16, num_parts: int = 16):
        super().__init__()
        c = in_channels
        self.fuse_conv = nn.Sequential(
            ConvBNAct(c * 3, c, 3),
            RepBlock(c)
        )
        # Raw logit projections
        self.semantic_head = nn.Conv2d(c, num_classes, 1)
        self.boundary_head = nn.Conv2d(c, 1, 1)
        self.instance_head = nn.Conv2d(c, embed_dim, 1)
        self.quality_head = nn.Conv2d(c, 1, 1)
        self.parts_head = nn.Conv2d(c, num_parts, 1)

    def forward(self, q3, q4, q5, out_hw: tuple):
        # Feature pyramid fusion into high-resolution q3 grid
        f3 = q3
        f4 = F.interpolate(q4, size=q3.shape[-2:], mode="bilinear", align_corners=False)
        f5 = F.interpolate(q5, size=q3.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.fuse_conv(torch.cat([f3, f4, f5], dim=1))

        # Output raw logits interpolated to original input image resolution
        sem_logits = F.interpolate(self.semantic_head(feat), size=out_hw, mode="bilinear", align_corners=False)
        bound_logits = F.interpolate(self.boundary_head(feat), size=out_hw, mode="bilinear", align_corners=False)
        inst_raw = F.interpolate(self.instance_head(feat), size=out_hw, mode="bilinear", align_corners=False)
        inst_emb = F.normalize(inst_raw, dim=1)  # L2 normalized embeddings
        quality_logits = F.interpolate(self.quality_head(feat), size=out_hw, mode="bilinear", align_corners=False)
        parts_logits = F.interpolate(self.parts_head(feat), size=out_hw, mode="bilinear", align_corners=False)

        return {
            "semantic_masks": sem_logits,
            "boundary_map": bound_logits,
            "instance_embeddings": inst_emb,
            "mask_quality": quality_logits,
            "part_masks": parts_logits
        }
