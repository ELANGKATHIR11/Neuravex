import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neuravex.models.neuravex import Neuravex, build_neuravex
from neuravex.geometry.camera import CameraIntrinsics

def test_neuravex_with_slots_and_rl():
    # Instantiate Neuravex nano with slots enabled
    model = Neuravex(num_classes=10, base_c=16, depth_mul=0.33, enable_slots=True, num_slots=4)
    model.eval()

    intrinsics = CameraIntrinsics(fx=64.0, fy=64.0, cx=64.0, cy=64.0)
    x = torch.randn(2, 3, 128, 128)

    # 1. Forward with active RL compute policy
    out = model(x, intrinsics=intrinsics, use_rl_policy=True)

    # Verify 2D and 3D detection outputs
    assert "class_logits" in out
    assert "pred_boxes" in out
    assert "pred_xyz" in out
    assert "pred_lwh" in out
    assert "pred_yaw_sincos" in out

    # Verify Free-Energy OOD detection scores
    assert "free_energy" in out
    assert "ood_score" in out
    assert out["ood_score"].shape[0] == 2
    assert not torch.isnan(out["free_energy"]).any()

    # Verify RL Policy Action outputs
    assert "rl_action" in out
    assert "rl_log_prob" in out
    assert "rl_value" in out
    assert out["rl_action"].shape == (2,)

    # Verify Slot Attention Unsupervised Object Discovery outputs
    assert "slots" in out
    assert "slot_masks" in out
    assert "discovered_boxes_3d" in out
    assert "slot_objectness" in out
    assert out["slots"].shape == (2, 4, 128)
    assert out["discovered_boxes_3d"].shape == (2, 4, 6)
