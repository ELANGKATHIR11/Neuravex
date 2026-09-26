import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neuravex.models.backbone import Backbone, MaskedMultimodalAutoencoder


def test_masked_multimodal_autoencoder():
    backbone = Backbone(base_c=16, depth_mul=0.33)
    mma = MaskedMultimodalAutoencoder(
        backbone=backbone,
        patch_size=16,
        embed_dim=64,
        rgb_mask_ratio=0.60,
        depth_mask_ratio=0.75
    )

    B, H, W = 2, 64, 64
    rgb = torch.rand(B, 3, H, W)
    depth = torch.rand(B, 1, H, W) * 10.0

    res = mma(rgb, depth)
    assert "loss_mma" in res
    assert "loss_rgb_recon" in res
    assert "loss_depth_recon" in res
    assert res["loss_mma"] > 0.0
    assert not torch.isnan(res["loss_mma"])
