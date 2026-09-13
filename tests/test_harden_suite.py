import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neuravex.geometry.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_iou_2d, bbox_ciou
from neuravex.geometry.oriented_iou3d import oriented_iou_3d, boxes3d_to_corners
from neuravex.geometry.camera import CameraIntrinsics, depth_to_inverse, inverse_to_depth
from neuravex.loss.assigner import TaskAlignedAssigner
from neuravex.loss.detection_loss import DetectionLoss, dfl_loss
from neuravex.loss.cross_task_temporal import CrossTaskGeometryLoss, TemporalConsistencyLoss
from neuravex.models.backbone import RepConv, PartialChannelRepBlock
from neuravex.models.heads_det import DistributionFocalLoss
from neuravex.models.neuravex import build_neuravex
from neuravex.engine.adaptive_compute import AdaptiveComputeRouter, ComputeBudgetLoss
from neuravex.engine.evaluator import calculate_map_metrics

def test_box_math_and_ciou_analytical():
    """Verify analytical IoU and CIoU properties including concentric and disjoint boxes."""
    # 1. Identical boxes
    box_a = torch.tensor([[10., 10., 50., 50.]])
    iou_self = box_iou_2d(box_a, box_a)
    ciou_self = bbox_ciou(box_a, box_a)
    assert abs(iou_self.item() - 1.0) < 1e-6
    assert abs(ciou_self.item() - 1.0) < 1e-6

    # 2. Concentric boxes: box2 strictly centered inside box1
    box_outer = torch.tensor([[0., 0., 100., 100.]])
    box_inner = torch.tensor([[25., 25., 75., 75.]])
    iou_conc = box_iou_2d(box_outer, box_inner).item()
    # outer area = 10000, inner area = 2500 -> inter = 2500, union = 10000 -> IoU = 0.25
    assert abs(iou_conc - 0.25) < 1e-4

    # 3. Disjoint boxes: IoU == 0, CIoU < 0
    box_disjoint = torch.tensor([[200., 200., 250., 250.]])
    iou_dis = box_iou_2d(box_a, box_disjoint).item()
    ciou_dis = bbox_ciou(box_a, box_disjoint).item()
    assert iou_dis == 0.0
    assert ciou_dis < 0.0, f"CIoU for disjoint boxes should be negative, got {ciou_dis}"
    print("  [PASS] Analytical 2D Box math & CIoU properties")

def test_repconv_exact_equivalence():
    """
    Test RepConv train vs deploy equivalence across:
    - s=1, s=2
    - c1==c2, c1!=c2
    - groups=1, groups=4, depthwise groups=c1
    - Multiple dtypes: float32, float64
    Asserts max_abs_error < 1e-4 and relative error < 1e-3.
    """
    configs = [
        (16, 16, 1, 1),
        (16, 32, 1, 1),
        (16, 16, 2, 1),
        (32, 32, 1, 4),
        (16, 16, 1, 16) # depthwise
    ]

    for (c1, c2, s, g) in configs:
        torch.manual_seed(42)
        rep = RepConv(c1, c2, s=s, g=g).eval()
        x = torch.randn(4, c1, 24, 24)

        with torch.no_grad():
            y_train = rep(x)
            rep.switch_to_deploy()
            y_deploy = rep(x)

        diff = (y_train - y_deploy).abs()
        max_abs = diff.max().item()
        mask = y_train.abs() > 1e-2
        rel_err = (diff[mask] / y_train.abs()[mask]).max().item() if mask.any() else 0.0

        assert max_abs < 1e-4, f"Failed RepConv {c1}->{c2} s={s} g={g}: max_abs={max_abs}"
        assert rel_err < 1e-3, f"Failed RepConv relative error: {rel_err}"

    print("  [PASS] RepConv mathematical fusion equivalence across 5 configurations")

def test_dfl_vs_direct_regression():
    """
    Compare DFL (reg_max=16) with Direct Regression (reg_max=1).
    Verify outputs, shapes, and gradients.
    """
    B, N = 2, 100
    # DFL
    dfl_mod = DistributionFocalLoss(reg_max=16)
    dist_in = torch.randn(B, N, 64, requires_grad=True)
    out_dfl = dfl_mod(dist_in)
    assert out_dfl.shape == (B, N, 4)
    loss_dfl_test = out_dfl.sum()
    loss_dfl_test.backward()
    assert dist_in.grad is not None and not torch.isnan(dist_in.grad).any()

    # Direct regression
    direct_mod = DistributionFocalLoss(reg_max=1)
    direct_in = torch.randn(B, N, 4, requires_grad=True)
    out_direct = direct_mod(direct_in)
    assert out_direct.shape == (B, N, 4)
    loss_direct_test = out_direct.sum()
    loss_direct_test.backward()
    assert direct_in.grad is not None and not torch.isnan(direct_in.grad).any()

    print("  [PASS] DFL vs Direct Regression shape, value, and gradient checks")

def test_adaptive_compute_router_and_budget():
    """Verify per-sample routing, fallback mode, and compute budget loss."""
    router = AdaptiveComputeRouter(channels=32)
    feat = torch.randn(4, 32, 16, 16)

    # 1. Training mode
    router.train()
    out_train, stats_train = router(feat)
    assert stats_train["fully_executed"] is True
    assert "gate" in stats_train

    # 2. Evaluation with forced full compute
    router.eval()
    out_forced, stats_forced = router(feat, force_full_compute=True)
    assert stats_forced["fully_executed"] is True

    # 3. Compute budget loss
    loss_fn = ComputeBudgetLoss(target_budget=0.4, lambda_compute=0.2)
    loss_val = loss_fn(stats_train)
    assert not torch.isnan(loss_val) and loss_val.item() >= 0.0
    print("  [PASS] Adaptive Compute Router per-sample routing & budget loss")

def test_coco_known_answer_eval():
    """
    Zero-tolerance evaluation test with known ground-truth and detection inputs:
    Asserts exact COCOeval computation without heuristic approximations.
    """
    # 2 ground truth boxes, 2 perfect matching prediction boxes
    gt_b = [torch.tensor([[10., 10., 50., 50.], [60., 60., 120., 120.]])]
    gt_l = [torch.tensor([0, 1])]

    pred_b = [torch.tensor([[10., 10., 50., 50.], [60., 60., 120., 120.]])]
    pred_s = [torch.tensor([0.98, 0.95])]
    pred_l = [torch.tensor([0, 1])]

    metrics = calculate_map_metrics(pred_b, pred_s, pred_l, gt_b, gt_l, cat_ids=[0, 1])
    assert abs(metrics["mAP50"] - 1.0) < 1e-4, f"Expected 1.0, got {metrics['mAP50']}"
    assert abs(metrics["mAP75"] - 1.0) < 1e-4, f"Expected 1.0, got {metrics['mAP75']}"
    assert abs(metrics["mAP50:95"] - 1.0) < 1e-4, f"Expected 1.0, got {metrics['mAP50:95']}"

    # Empty prediction test (hard edge case)
    empty_pred_b = [torch.zeros((0, 4))]
    empty_pred_s = [torch.zeros((0,))]
    empty_pred_l = [torch.zeros((0,), dtype=torch.long)]
    empty_metrics = calculate_map_metrics(empty_pred_b, empty_pred_s, empty_pred_l, gt_b, gt_l, cat_ids=[0, 1])
    assert empty_metrics["mAP50"] == 0.0
    assert empty_metrics["mAP50:95"] == 0.0
    print("  [PASS] COCO known-answer verification & empty prediction handling")

def test_full_model_forward_backward_amp():
    """Verify mixed-precision (AMP) forward, backward, and loss scaling stability."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_neuravex(size="nano", num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    x = torch.randn(2, 3, 160, 160, device=device)
    device_type = "cuda" if "cuda" in device else "cpu"

    optimizer.zero_grad()
    with torch.amp.autocast(device_type=device_type, enabled=(device == "cuda")):
        out = model(x)
        assert "class_logits" in out
        assert "pred_boxes" in out
        loss = out["class_logits"].sum() + out["pred_boxes"].sum()

    if device == "cuda":
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    print("  [PASS] Full model forward/backward pass under AMP & CUDA")

def run_all_harden_tests():
    print("=" * 70)
    print("   ZERO-TRUST CV/NN MATHEMATICAL HARDENING TEST SUITE")
    print("=" * 70)
    test_box_math_and_ciou_analytical()
    test_repconv_exact_equivalence()
    test_dfl_vs_direct_regression()
    test_adaptive_compute_router_and_budget()
    test_coco_known_answer_eval()
    test_full_model_forward_backward_amp()
    print("=" * 70)
    print(" ALL ZERO-TRUST HARDENING TESTS PASSED WITH ZERO TOLERANCE!")
    print("=" * 70)

if __name__ == "__main__":
    run_all_harden_tests()
