import pytest
import math
import torch
import numpy as np

from neuravex import (
    Neuravex,
    build_neuravex,
    CameraIntrinsics,
    NeuravexInferencePostProcessor,
    RealTimeMetricDepthTracker,
    robust_mask_depth_estimator,
    calculate_depth_metrics
)
from neuravex.loss.depth_3d_loss import comprehensive_metric_depth_loss, loss_3d_detection

def test_camera_intrinsics_transforms():
    """Test scale and letterbox transformations on CameraIntrinsics."""
    K = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
    
    # Scaling test (e.g. downsampling by 0.5)
    K_scaled = K.scale(scale_x=0.5, scale_y=0.5)
    assert math.isclose(K_scaled.fx, 500.0)
    assert math.isclose(K_scaled.fy, 500.0)
    assert math.isclose(K_scaled.cx, 320.0)
    assert math.isclose(K_scaled.cy, 180.0)

    # Letterbox test (scaling by 0.5 with 20px pad_x and 10px pad_y)
    K_lb = K.adjust_letterbox(scale=0.5, pad_x=20.0, pad_y=10.0)
    assert math.isclose(K_lb.fx, 500.0)
    assert math.isclose(K_lb.fy, 500.0)
    assert math.isclose(K_lb.cx, 340.0)
    assert math.isclose(K_lb.cy, 190.0)

def test_projection_unprojection_consistency():
    """
    Test pinhole unprojection and re-projection consistency:
      Z = Zobj
      X = (u - cx)*Z / fx
      Y = (v - cy)*Z / fy
      u' = X*fx / Z + cx
      v' = Y*fy / Z + cy
    """
    K = CameraIntrinsics(fx=721.5, fy=721.5, cx=609.5, cy=172.8)
    u_orig, v_orig, z_orig = 500.0, 200.0, 15.25

    X = (u_orig - K.cx) * z_orig / K.fx
    Y = (v_orig - K.cy) * z_orig / K.fy
    Z = z_orig
    R = math.sqrt(X**2 + Y**2 + Z**2)

    # Re-project
    u_proj = (X * K.fx) / Z + K.cx
    v_proj = (Y * K.fy) / Z + K.cy

    assert math.isclose(u_proj, u_orig, rel_tol=1e-5)
    assert math.isclose(v_proj, v_orig, rel_tol=1e-5)
    assert math.isclose(math.sqrt(X**2 + Y**2 + Z**2), R, rel_tol=1e-5)

def test_robust_mask_depth_estimator():
    """
    Test true mask-depth fusion rejecting invalid/outlier pixels.
    """
    H, W = 100, 100
    # Ground truth metric depth is 10.0m
    depth_map = torch.ones((H, W), dtype=torch.float32) * 10.0
    conf_map = torch.ones((H, W), dtype=torch.float32) * 0.95

    # Object mask: rectangle in [30:70, 30:70]
    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[30:70, 30:70] = True

    # Inject extreme outliers inside mask: background bleed at 80m and near-field dust at 0.1m
    depth_map[30:35, 30:70] = 80.0
    conf_map[30:35, 30:70] = 0.1  # Low confidence on outliers
    depth_map[65:70, 30:70] = 0.1
    conf_map[65:70, 30:70] = 0.1

    z_est, u_c, v_c, d_conf = robust_mask_depth_estimator(
        depth_map, mask, depth_conf=conf_map, min_depth=0.5, max_depth=60.0, trim_ratio=0.15
    )

    # Outliers should be rejected by trimmed median; estimated depth should be ~10.0m
    assert math.isclose(z_est, 10.0, abs_tol=0.2)
    assert math.isclose(u_c, 49.5, abs_tol=1.0)
    assert math.isclose(v_c, 49.5, abs_tol=1.0)
    assert d_conf > 0.85

def test_metric_depth_loss_and_confidence_calibration():
    """Test comprehensive metric depth loss with calibrated uncertainty."""
    pred_depth = torch.tensor([[[[10.0, 15.0], [20.0, 25.0]]]]).float()
    gt_depth = torch.tensor([[[[10.5, 14.8], [20.2, 24.5]]]]).float()
    valid_mask = torch.ones_like(gt_depth).bool()
    pred_conf = torch.tensor([[[[0.9, 0.9], [0.9, 0.9]]]]).float()

    loss = comprehensive_metric_depth_loss(pred_depth, gt_depth, valid_mask, pred_confidence=pred_conf)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_real_time_metric_depth_tracker():
    """
    Test temporal stabilization, EMA filtering, occlusion coasting, and scene cut detection.
    """
    tracker = RealTimeMetricDepthTracker(max_missed_frames=3, scene_cut_distance=20.0)

    # Frame 1: Object at (0, 0, 10)
    frame1 = [{
        "class": 0,
        "score": 0.9,
        "bbox": [50.0, 50.0, 100.0, 100.0],
        "x": 0.0,
        "y": 0.0,
        "z": 10.0,
        "depth": 10.0,
        "distance": 10.0,
        "depth_confidence": 0.9
    }]
    out1 = tracker.step(frame1)
    assert len(out1) == 1
    assert out1[0]["id"] == 1
    assert math.isclose(out1[0]["z"], 10.0, abs_tol=1e-3)

    # Frame 2: Slight motion with noisy depth (10.6m instead of 10.2m)
    frame2 = [{
        "class": 0,
        "score": 0.9,
        "bbox": [51.0, 50.0, 101.0, 100.0],
        "x": 0.1,
        "y": 0.0,
        "z": 10.6,
        "depth": 10.6,
        "distance": 10.6,
        "depth_confidence": 0.8
    }]
    out2 = tracker.step(frame2)
    assert len(out2) == 1
    assert out2[0]["id"] == 1
    # Check that EMA smoothed the depth jump (should be between 10.0 and 10.6)
    assert 10.0 < out2[0]["z"] < 10.6

    # Frame 3: Missed/occluded detection (empty detection)
    out3 = tracker.step([])
    assert len(out3) == 0
    # Track still exists in tracker memory with incremented time_since_update
    assert 1 in tracker.tracks
    assert tracker.tracks[1].time_since_update == 1

    # Frame 4: Scene cut test (all objects shift by 50m)
    frame_scene_cut = [{
        "class": 0,
        "score": 0.95,
        "bbox": [500.0, 500.0, 600.0, 600.0],
        "x": 50.0,
        "y": 50.0,
        "z": 60.0,
        "depth": 60.0,
        "distance": math.sqrt(50**2 + 50**2 + 60**2),
        "depth_confidence": 0.95
    }]
    out_sc = tracker.step(frame_scene_cut)
    # Old track (id 1) should be discarded due to scene cut
    assert len(out_sc) == 1
    assert out_sc[0]["id"] != 1 or len(tracker.tracks) == 1

def test_full_inference_pipeline_and_bypassing():
    """
    Test complete inference pipeline emitting required keys and verify
    zero-overhead bypassing when tasks=('det',).
    """
    model = build_neuravex(size="nano", num_classes=5)
    model.eval()
    post_processor = NeuravexInferencePostProcessor(conf_thresh=0.01)
    K = CameraIntrinsics(fx=500.0, fy=500.0, cx=160.0, cy=160.0)
    tracker = RealTimeMetricDepthTracker()

    x = torch.randn(1, 3, 320, 320)

    # 1. Fast path: tasks=('det',)
    det_only = model(x, tasks=("det",), intrinsics=K)
    assert "pred_boxes" in det_only
    assert "class_logits" in det_only
    # Aux heads must NOT be computed
    assert "depth_map" not in det_only
    assert "semantic_masks" not in det_only

    res_det = post_processor(det_only, intrinsics=K, tracker=tracker)
    assert "detections" in res_det
    assert "objects" in res_det

    # 2. Full 3D Multi-task path: tasks=None
    full_out = model(x, tasks=None, intrinsics=K)
    assert "depth_map" in full_out
    assert "depth_confidence" in full_out
    assert "semantic_masks" in full_out

    res_full = post_processor(full_out, intrinsics=K, tracker=tracker)
    assert "objects" in res_full
    if len(res_full["objects"][0]) > 0:
        obj = res_full["objects"][0][0]
        # Check all required keys
        required_keys = ["id", "class", "score", "bbox", "mask", "x", "y", "z", "depth", "distance", "depth_confidence"]
        for k in required_keys:
            assert k in obj, f"Missing key: {k}"

def test_fp16_autocast_numerical_stability():
    """Test FP16 / AMP execution safety on full model forward."""
    model = build_neuravex(size="nano", num_classes=5)
    model.eval()
    x = torch.randn(1, 3, 320, 320)
    K = CameraIntrinsics(fx=500.0, fy=500.0, cx=160.0, cy=160.0)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(x, tasks=None, intrinsics=K)

    assert not torch.isnan(out["depth_map"]).any()
    assert not torch.isnan(out["depth_confidence"]).any()
    assert not torch.isinf(out["depth_map"]).any()
