import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neuravex.geometry.camera import CameraIntrinsics

from neuravex.losses.photometric import (
    RelativePoseEstimator,
    ViewSynthesisWarp,
    PhotometricReconstructionLoss
)

def test_relative_pose_estimator():
    net = RelativePoseEstimator(in_channels=6, hidden_dim=64)
    img_t = torch.randn(2, 3, 64, 64)
    img_target = torch.randn(2, 3, 64, 64)
    pose = net(img_t, img_target)
    assert pose.shape == (2, 6)
    assert not torch.isnan(pose).any()

def test_view_synthesis_warp():
    warp = ViewSynthesisWarp()
    B, H, W = 2, 64, 64
    img_target = torch.rand(B, 3, H, W)
    depth_t = torch.ones(B, 1, H, W) * 2.0
    rot = torch.zeros(B, 3)
    trans = torch.zeros(B, 3)
    intrinsics = CameraIntrinsics(fx=32.0, fy=32.0, cx=32.0, cy=32.0)

    warped, valid_mask = warp(img_target, depth_t, rot, trans, intrinsics)
    assert warped.shape == (B, 3, H, W)
    assert valid_mask.shape == (B, 1, H, W)
    # Zero motion identity warp check
    assert torch.allclose(warped, img_target, atol=1e-2)

def test_photometric_loss_with_automask():
    loss_fn = PhotometricReconstructionLoss(alpha=0.85, beta_smooth=0.1, auto_mask=True)
    B, H, W = 2, 32, 32
    img_t = torch.rand(B, 3, H, W)
    img_warped = img_t.clone() # perfect reconstruction
    img_raw_target = torch.rand(B, 3, H, W)
    depth_t = torch.ones(B, 1, H, W) * 5.0

    res = loss_fn(img_t, img_warped, img_raw_target, depth_t)
    assert "loss_photo" in res
    assert res["loss_photo"] >= 0.0
    assert not torch.isnan(res["loss_photo"])
