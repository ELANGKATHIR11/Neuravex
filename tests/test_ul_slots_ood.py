import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neuravex.models.slots import SlotAttention, UnsupervisedObjectDiscoveryHead
from neuravex.models.heads_det import MultiScaleDetectionHead


def test_slot_attention():
    slot_attn = SlotAttention(num_slots=4, slot_dim=32, in_features=64, iters=2)
    B, N, C = 2, 16, 64 # e.g. 4x4 spatial patches
    inputs = torch.randn(B, N, C)
    slots, attn = slot_attn(inputs)

    assert slots.shape == (B, 4, 32)
    assert attn.shape == (B, 4, N)
    # Check competition softmax: sum over slots is approx 1
    assert torch.allclose(attn.sum(dim=1), torch.ones(B, N), atol=1e-3)

def test_unsupervised_discovery_head():
    head = UnsupervisedObjectDiscoveryHead(in_channels=64, num_slots=4, slot_dim=32)
    feat = torch.randn(2, 64, 8, 8)
    out = head(feat)

    assert "slots" in out
    assert "slot_masks" in out
    assert "discovered_boxes_3d" in out
    assert "slot_objectness" in out
    assert out["slot_masks"].shape == (2, 4, 8, 8)
    assert out["discovered_boxes_3d"].shape == (2, 4, 6)

def test_free_energy_ood_detector():
    head = MultiScaleDetectionHead(in_channels=64, num_classes=10, strides=(8, 16, 32))
    feats = [
        torch.randn(2, 64, 16, 16),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 64, 4, 4)
    ]
    out = head(feats)

    assert "free_energy" in out
    assert "ood_score" in out
    assert out["free_energy"].shape[0] == 2
    assert out["ood_score"].min() >= 0.0
    assert out["ood_score"].max() <= 1.0
