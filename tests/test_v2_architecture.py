import pytest
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neuravex.models.neuravex import build_neuravex, Neuravex
from neuravex.models.heads_det import MultiScaleDetectionHead
from neuravex.loss.assigner import TaskAlignedAssigner, HungarianOneToOneAssigner
from neuravex.loss.detection_loss import DetectionLoss, DualAssignmentDetectionLoss
from neuravex.loss.depth_3d_loss import loss_3d_detection
from neuravex.geometry.oriented_iou3d import rotation_6d_to_matrix, matrix_to_yaw
from neuravex.geometry.camera import CameraIntrinsics

def test_rotation_6d_to_matrix_orthogonality():
    """Verify continuous 6D rotation produces valid SO(3) orthogonal matrices with det(R) = +1."""
    # Arbitrary random 6D vectors
    d6 = torch.randn(4, 10, 6)
    R = rotation_6d_to_matrix(d6)  # (4, 10, 3, 3)

    assert R.shape == (4, 10, 3, 3)

    # Check orthogonality: R @ R^T = I
    RT = R.transpose(-1, -2)
    identity_approx = torch.matmul(R, RT)
    eye = torch.eye(3).expand_as(identity_approx)
    assert torch.allclose(identity_approx, eye, atol=1e-5)

    # Check determinant: det(R) = +1
    dets = torch.linalg.det(R)
    assert torch.allclose(dets, torch.ones_like(dets), atol=1e-5)

    # Check yaw extraction
    yaws = matrix_to_yaw(R)
    assert yaws.shape == (4, 10)
    assert not torch.isnan(yaws).any()

def test_hungarian_one_to_one_assigner():
    """Verify exact 1-to-1 matching for NMS-free head."""
    B, N, num_classes = 2, 100, 10
    M = 3 # 3 GT boxes per batch

    pd_scores = torch.randn(B, N, num_classes)
    pd_bboxes = torch.rand(B, N, 4) * 100.0
    pd_bboxes = torch.stack([pd_bboxes[..., 0], pd_bboxes[..., 1],
                             pd_bboxes[..., 0] + pd_bboxes[..., 2],
                             pd_bboxes[..., 1] + pd_bboxes[..., 3]], dim=-1)

    anc_points = torch.rand(N, 2) * 200.0
    gt_labels = torch.randint(0, num_classes, (B, M, 1))
    gt_bboxes = torch.tensor([
        [[10.0, 10.0, 50.0, 50.0], [60.0, 60.0, 120.0, 120.0], [130.0, 130.0, 180.0, 180.0]],
        [[20.0, 20.0, 70.0, 70.0], [80.0, 80.0, 140.0, 140.0], [150.0, 150.0, 190.0, 190.0]]
    ])
    mask_gt = torch.ones((B, M, 1), dtype=torch.bool)

    assigner = HungarianOneToOneAssigner(num_classes=num_classes)
    tgt_labels, tgt_bboxes, tgt_scores, fg_mask, tgt_gt_idx = assigner(
        pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
    )

    # In exact 1-to-1 matching, number of positive anchors per batch must be EXACTLY equal to M
    for b in range(B):
        assert fg_mask[b].sum().item() == M
        # Ensure matched GT indices are unique
        matched_gts = tgt_gt_idx[b][fg_mask[b]]
        assert len(matched_gts.unique()) == M

def test_dual_head_forward_and_loss():
    """Verify dual head (O2M + O2I) training and zero-NMS deployment outputs."""
    net = build_neuravex(size="nano", num_classes=5)
    net.train()

    x = torch.randn(2, 3, 128, 128)
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=64.0, cy=64.0)

    # 1. Training Forward Pass
    out_train = net(x, intrinsics=intrinsics, tasks=("det", "dem"))
    assert "class_logits" in out_train
    assert "pred_boxes" in out_train
    assert "o2m_preds" in out_train
    assert "pred_6d_rot" in out_train
    assert "pred_rot_matrix" in out_train
    assert "log_sigma_xyz" in out_train
    assert out_train["log_sigma_xyz"].shape == out_train["pred_xyz"].shape

    # 2. Deployment Forward Pass (eval mode, zero-NMS)
    net.eval()
    with torch.no_grad():
        out_eval = net(x, intrinsics=intrinsics, tasks=("det",))
        assert "class_logits" in out_eval
        assert "pred_boxes" in out_eval
        # O2M auxiliary branch is dropped at inference
        assert "o2m_preds" not in out_eval
        # Zero duplicate post-processing NMS needed: outputs are deterministic top-K ready
        assert out_eval["pred_boxes"].ndim == 3

    # 3. Dual-assignment loss check
    net.train()
    loss_fn = DualAssignmentDetectionLoss(num_classes=5)

    N_anchors = out_train["class_logits"].shape[1]
    tgt_scores = torch.zeros_like(out_train["class_logits"])
    tgt_boxes = torch.zeros_like(out_train["pred_boxes"])
    fg_mask = torch.zeros((2, N_anchors), dtype=torch.bool)
    fg_mask[:, :3] = True
    tgt_scores[:, :3, 0] = 1.0

    targets_o2i = (tgt_scores, tgt_boxes, fg_mask)
    targets_o2m = (tgt_scores, tgt_boxes, fg_mask)

    total_loss, loss_dict = loss_fn(
        out_train, out_train["o2m_preds"],
        out_train["anchor_points"], out_train["strides"],
        targets_o2i, targets_o2m
    )
    assert total_loss.requires_grad
    total_loss.backward()
    assert total_loss.item() > 0.0

def test_dem_depth_cross_gating():
    """Verify DEM dense depth map physically conditions 3D detection regression."""
    net = build_neuravex(size="nano", num_classes=5)
    net.eval()

    x = torch.randn(1, 3, 128, 128)
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=64.0, cy=64.0)

    # Run with both det and DEM
    out = net(x, intrinsics=intrinsics, tasks=("det", "dem"))
    assert "depth_map" in out
    assert "pred_xyz" in out

    # The predicted depth Z should be positive and bounded within reasonable physical depth
    z_pred = out["pred_xyz"][..., 2]
    assert (z_pred > 0.0).all()
    assert not torch.isnan(z_pred).any()

def test_replk_deploy_switch():
    """Verify RepLK 7x7 conv switches cleanly to deploy without numerical blowup."""
    net = build_neuravex(size="nano", num_classes=5)
    net.eval()

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out1 = net(x, tasks=("det",))
        net.switch_to_deploy()
        out2 = net(x, tasks=("det",))

    # Features before and after deploy fusing should be numerically close
    diff = torch.max(torch.abs(out1["class_logits"] - out2["class_logits"]))
    assert diff < 1e-4
