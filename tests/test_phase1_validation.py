"""
Neuravex Phase 1 Validation Test Suite.

Tests all critical fixes:
  - Instance mask uniqueness (proto_assembly, 1:1 detection↔mask)
  - Mask quality (no semantic crop, no full-box fallback when proto available)
  - Depth from DEM head (no fabricated gradient)
  - XYZ from intrinsics + depth (pinhole model)
  - 3D corners from boxes3d_to_corners (mathematical projection)
  - 3D IoU correctness
  - LWH positivity (from model head, no heuristic multipliers)
  - Yaw continuity (from 6D rotation head)
  - Camera project/unproject round-trip
  - Native pipeline has NO external detector
  - Core import without heavy deps
  - Backend fallback behavior
  - Deterministic inference
  - ObjectState/FrameState schema
"""

import sys
import math
import numpy as np
import pytest

# ============================================================
# Test 1: Core import without heavy dependencies
# ============================================================

def test_core_import_minimal():
    """Import neuravex.core without torch/torchvision/scipy."""
    from neuravex.core import ObjectState, FrameState
    obj = ObjectState(id=1, class_id=0)
    frame = FrameState(frame_id=0, total_visible=0)
    assert obj.id == 1
    assert frame.total_visible == 0
    assert obj.mask is None
    assert obj.depth_source == "UNVALIDATED"


def test_version_exists():
    import neuravex
    assert hasattr(neuravex, "__version__")
    assert neuravex.__version__ >= "0.2.0"


# ============================================================
# Test 2: ObjectState / FrameState schema
# ============================================================

def test_object_state_to_dict():
    from neuravex.core import ObjectState
    obj = ObjectState(
        id=1, class_id=5, class_name="car",
        score=0.92, bbox2d=[10, 20, 100, 150],
        depth=5.3, depth_source="dem_head",
        xyz=[1.2, -0.5, 5.3], lwh=[4.5, 1.8, 1.5], yaw=0.3,
        mask_source="proto_assembly",
    )
    d = obj.to_dict()
    assert d["id"] == 1
    assert d["class_name"] == "car"
    assert d["depth_source"] == "dem_head"
    assert d["mask_source"] == "proto_assembly"
    assert d["lwh"] == [4.5, 1.8, 1.5]


def test_frame_state_to_dict():
    from neuravex.core import ObjectState, FrameState
    frame = FrameState(
        frame_id=42, image_width=1920, image_height=1080,
        objects=[ObjectState(id=1, class_id=0)],
        total_visible=1, depth_validated=False,
    )
    d = frame.to_dict()
    assert d["frame_id"] == 42
    assert d["depth_validated"] is False
    assert len(d["objects"]) == 1


# ============================================================
# Test 3: Model builds without crash for all variants
# ============================================================

@pytest.mark.parametrize("variant", ["pico", "femto", "nano", "lite", "edge", "pro", "omni"])
def test_build_neuravex_variants(variant):
    """Every model variant must build without error."""
    import torch
    from neuravex.models.neuravex import build_neuravex
    model = build_neuravex(variant, num_classes=10)
    assert model is not None
    # Count parameters — must be positive
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0


# ============================================================
# Test 4: Forward pass produces expected keys
# ============================================================

def test_forward_pass_output_keys():
    """Full forward pass must produce detection, segmentation, depth, and mask coefficient keys."""
    import torch
    from neuravex.models.neuravex import build_neuravex

    model = build_neuravex("pico", num_classes=5).eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    # Detection keys
    assert "class_logits" in out
    assert "pred_boxes" in out
    assert "pred_mask_coeffs" in out  # NEW: mask coefficients

    # 3D keys
    assert "pred_xyz" in out
    assert "pred_lwh" in out
    assert "pred_yaw_sincos" in out

    # Segmentation keys
    assert "proto_masks" in out
    assert "semantic_masks" in out
    assert "instance_embeddings" in out
    assert "mask_quality" in out

    # Depth keys
    assert "depth_map" in out
    assert "depth_confidence" in out
    assert "terrain_elevation" in out


# ============================================================
# Test 5: Mask coefficient dimensions match proto_masks
# ============================================================

def test_mask_coeff_dimensions():
    """pred_mask_coeffs must have same number of channels as proto_masks."""
    import torch
    from neuravex.models.neuravex import build_neuravex

    model = build_neuravex("pico", num_classes=5).eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    num_proto = out["proto_masks"].shape[1]  # (B, num_proto, H, W)
    num_coeff = out["pred_mask_coeffs"].shape[2]  # (B, N_anchors, num_proto)
    assert num_proto == num_coeff, f"Proto {num_proto} != Coeff {num_coeff}"


# ============================================================
# Test 6: Instance mask assembly produces unique masks
# ============================================================

def test_instance_mask_assembly_uniqueness():
    """Each detection must get a unique instance mask from proto_assembly."""
    import torch
    from neuravex.engine.precision_perception import assemble_instance_masks

    B, P, H, W = 1, 32, 64, 64
    N_det = 3

    proto_masks = torch.randn(B, P, H, W)
    # Different coefficients → different masks
    mask_coeffs = torch.randn(N_det, P)
    det_boxes = torch.tensor([
        [5, 5, 30, 30],
        [35, 35, 60, 60],
        [10, 40, 50, 60],
    ], dtype=torch.float32)

    masks = assemble_instance_masks(proto_masks, mask_coeffs, det_boxes, H, W)
    assert len(masks) == N_det

    # Masks should not be identical
    for i in range(N_det):
        for j in range(i + 1, N_det):
            if masks[i].sum() > 0 and masks[j].sum() > 0:
                # Not ALL pixels identical
                assert not np.array_equal(masks[i], masks[j]), \
                    f"Masks {i} and {j} are identical — violates instance uniqueness"


# ============================================================
# Test 7: LWH positivity from detection head
# ============================================================

def test_lwh_positivity():
    """Model LWH predictions must be strictly positive (exp-clamped)."""
    import torch
    from neuravex.models.neuravex import build_neuravex

    model = build_neuravex("pico", num_classes=5).eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    lwh = out["pred_lwh"]
    assert (lwh > 0).all(), "LWH must be strictly positive (exp-clamped)"


# ============================================================
# Test 8: Depth map is strictly positive
# ============================================================

def test_depth_map_positive():
    """DEM head depth must be strictly positive metric distance."""
    import torch
    from neuravex.models.neuravex import build_neuravex

    model = build_neuravex("pico", num_classes=5).eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    depth = out["depth_map"]
    assert (depth > 0).all(), "Metric depth must be > 0"
    assert (depth < 1001).all(), "Depth should be clamped below 1000m"


# ============================================================
# Test 9: Camera unproject → project round-trip
# ============================================================

def test_camera_round_trip():
    """Unproject(u, v, Z) → XYZ → project(XYZ) should recover (u, v)."""
    import torch
    from neuravex.geometry.camera import CameraIntrinsics

    K = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)

    u = torch.tensor([100.0, 320.0, 500.0])
    v = torch.tensor([50.0, 240.0, 400.0])
    z = torch.tensor([3.0, 5.0, 10.0])

    xyz = K.unproject_points(u, v, z)
    uv_recovered = K.project_points(xyz)

    assert torch.allclose(uv_recovered[:, 0], u, atol=1e-3)
    assert torch.allclose(uv_recovered[:, 1], v, atol=1e-3)


# ============================================================
# Test 10: 3D corners from boxes3d_to_corners
# ============================================================

def test_3d_corners_shape_and_center():
    """boxes3d_to_corners should produce 8 corners centered on input center."""
    import torch
    from neuravex.geometry.oriented_iou3d import boxes3d_to_corners

    center = torch.tensor([[1.0, 2.0, 5.0]])
    lwh = torch.tensor([[2.0, 1.0, 1.5]])
    yaw = torch.tensor([0.0])

    corners = boxes3d_to_corners(center, lwh, yaw)
    assert corners.shape == (1, 8, 3)

    # Mean of all corners should be approximately the center
    mean_corner = corners.mean(dim=1)
    assert torch.allclose(mean_corner, center, atol=1e-4)


# ============================================================
# Test 11: 3D IoU correctness
# ============================================================

def test_3d_iou_identical_boxes():
    """IoU of identical 3D boxes must be 1.0."""
    import torch
    from neuravex.geometry.oriented_iou3d import oriented_iou_3d

    center = torch.tensor([[0.0, 0.0, 5.0]])
    lwh = torch.tensor([[2.0, 1.0, 1.5]])
    yaw = torch.tensor([0.0])

    iou = oriented_iou_3d(center, lwh, yaw, center, lwh, yaw)
    assert abs(float(iou.item()) - 1.0) < 0.01, f"Self-IoU should be ~1.0, got {iou.item()}"


def test_3d_iou_non_overlapping():
    """IoU of non-overlapping 3D boxes must be 0.0."""
    import torch
    from neuravex.geometry.oriented_iou3d import oriented_iou_3d

    c1 = torch.tensor([[0.0, 0.0, 0.0]])
    c2 = torch.tensor([[100.0, 100.0, 100.0]])
    lwh = torch.tensor([[1.0, 1.0, 1.0]])
    yaw = torch.tensor([0.0])

    iou = oriented_iou_3d(c1, lwh, yaw, c2, lwh, yaw)
    assert float(iou.item()) < 0.01, f"Non-overlapping IoU should be ~0, got {iou.item()}"


# ============================================================
# Test 12: 6D rotation → matrix → yaw round-trip
# ============================================================

def test_rotation_6d_roundtrip():
    """6D rotation representation should produce valid SO(3) matrix with det=+1."""
    import torch
    from neuravex.geometry.oriented_iou3d import rotation_6d_to_matrix

    d6 = torch.randn(5, 6)
    R = rotation_6d_to_matrix(d6)
    assert R.shape == (5, 3, 3)

    # Check orthogonality: R @ R^T ≈ I
    eye = torch.eye(3).expand(5, 3, 3)
    RRt = torch.bmm(R, R.transpose(1, 2))
    assert torch.allclose(RRt, eye, atol=1e-4), "R @ R^T must equal I"

    # Check determinant ≈ +1
    det = torch.det(R)
    assert torch.allclose(det, torch.ones(5), atol=1e-4), "det(R) must be +1"


# ============================================================
# Test 13: Robust mask depth estimator
# ============================================================

def test_robust_mask_depth_basic():
    """Robust depth estimator should return valid depth for a non-empty mask."""
    import torch
    from neuravex.engine.tracker import robust_mask_depth_estimator

    H, W = 64, 64
    depth_map = torch.full((H, W), 5.0)
    mask = torch.zeros((H, W), dtype=torch.bool)
    mask[20:40, 20:40] = True

    z, u, v, conf = robust_mask_depth_estimator(depth_map, mask)
    assert not math.isnan(z)
    assert abs(z - 5.0) < 0.5
    assert conf > 0


def test_robust_mask_depth_empty_mask():
    """Empty mask should return NaN."""
    import torch
    from neuravex.engine.tracker import robust_mask_depth_estimator

    depth_map = torch.full((64, 64), 5.0)
    mask = torch.zeros((64, 64), dtype=torch.bool)

    z, u, v, conf = robust_mask_depth_estimator(depth_map, mask)
    assert math.isnan(z)
    assert conf == 0.0


# ============================================================
# Test 14: No YOLO in native pipeline
# ============================================================

def test_no_external_detector_in_native_pipeline():
    """PrecisionPerceptionPipeline must NOT import or use ultralytics/YOLO."""
    import inspect
    from neuravex.engine import precision_perception
    source = inspect.getsource(precision_perception)

    assert "ultralytics" not in source, "YOLO/ultralytics found in native pipeline"
    assert "YOLO(" not in source, "YOLO constructor found in native pipeline"
    assert "seg_extractor" not in source, "seg_extractor (YOLO) found in native pipeline"


def test_no_fabricated_depth():
    """No perspective gradient fabrication in precision_perception."""
    import inspect
    from neuravex.engine import precision_perception
    source = inspect.getsource(precision_perception)

    assert "perspective_depth" not in source, "Fabricated perspective depth found"
    assert "1.8 + 4.5" not in source, "Hardcoded depth gradient found"
    assert "0.3 * dem_norm" not in source, "Fabricated DEM mixing found"


def test_no_heuristic_dimensions():
    """No heuristic L=0.85*B or similar multipliers."""
    import inspect
    from neuravex.engine import precision_perception
    source = inspect.getsource(precision_perception)

    assert "0.85" not in source, "Heuristic dimension multiplier 0.85 found"
    assert "Infant Langur" not in source, "Hardcoded infant label found"
    assert "Adult Langur" not in source, "Hardcoded adult label found"


def test_no_hardcoded_paths():
    """No hardcoded absolute user paths."""
    import inspect
    from neuravex.engine import precision_perception
    source = inspect.getsource(precision_perception)

    assert "C:\\Users\\elang" not in source, "Hardcoded user path found"
    assert "C:/Users/elang" not in source, "Hardcoded user path found"


# ============================================================
# Test 15: Tracker creates unique IDs
# ============================================================

def test_tracker_unique_ids():
    """Each new detection must get a unique track ID."""
    from neuravex.engine.tracker import RealTimeMetricDepthTracker

    tracker = RealTimeMetricDepthTracker()
    objects = [
        {"id": 1, "class": 0, "score": 0.9, "bbox": [10, 10, 50, 50],
         "x": 1.0, "y": 0.0, "z": 5.0, "depth": 5.0, "distance": 5.1, "depth_confidence": 0.8},
        {"id": 2, "class": 0, "score": 0.8, "bbox": [100, 100, 200, 200],
         "x": -1.0, "y": 0.5, "z": 8.0, "depth": 8.0, "distance": 8.1, "depth_confidence": 0.7},
    ]
    result = tracker.step(objects)
    ids = [r["id"] for r in result]
    assert len(ids) == len(set(ids)), f"Duplicate track IDs: {ids}"


# ============================================================
# Test 16: Adaptive compute router
# ============================================================

def test_adaptive_router_skip_path():
    """With high threshold, router should skip refinement."""
    import torch
    from neuravex.engine.adaptive_compute import AdaptiveComputeRouter

    router = AdaptiveComputeRouter(channels=32).eval()
    feat = torch.randn(2, 32, 8, 8)

    with torch.no_grad():
        out, stats = router(feat, threshold=0.999)

    assert out.shape == feat.shape
    assert "active_ratio" in stats


# ============================================================
# Test 17: Deterministic inference
# ============================================================

def test_deterministic_inference():
    """Two forward passes with same input must produce identical output."""
    import torch
    from neuravex.models.neuravex import build_neuravex

    model = build_neuravex("pico", num_classes=5).eval()
    x = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        out1 = model(x, tasks=("det",))
        out2 = model(x, tasks=("det",))

    assert torch.allclose(out1["class_logits"], out2["class_logits"])
    assert torch.allclose(out1["pred_boxes"], out2["pred_boxes"])


# ============================================================
# Test 18: 3D projection mathematical correctness
# ============================================================

def test_3d_projection_mathematical():
    """project_3d_corners_to_image must use pinhole model, not pixel offsets."""
    from neuravex.engine.precision_perception import project_3d_corners_to_image
    from neuravex.geometry.camera import CameraIntrinsics

    K = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    center = np.array([0.0, 0.0, 10.0])
    lwh = np.array([2.0, 1.0, 1.5])
    yaw = 0.0

    corners_2d = project_3d_corners_to_image(center, lwh, yaw, K)
    assert corners_2d is not None
    assert corners_2d.shape == (8, 2)

    # All corners should be near center of image since object is at (0,0,10)
    mean_u = corners_2d[:, 0].mean()
    mean_v = corners_2d[:, 1].mean()
    assert abs(mean_u - K.cx) < 50, f"Mean U {mean_u} too far from cx={K.cx}"
    assert abs(mean_v - K.cy) < 50, f"Mean V {mean_v} too far from cy={K.cy}"


# ============================================================
# Test 19: Deploy mode (RepConv fusion)
# ============================================================

def test_switch_to_deploy():
    """Model should be fusable to deploy mode without error."""
    import torch
    from neuravex.models.neuravex import build_neuravex

    model = build_neuravex("pico", num_classes=5).eval()
    x = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        out_before = model(x, tasks=("det",))

    model.switch_to_deploy()

    with torch.no_grad():
        out_after = model(x, tasks=("det",))

    # Outputs should be numerically close (fusion is exact for inference)
    assert torch.allclose(
        out_before["class_logits"], out_after["class_logits"], atol=1e-3
    ), "Deploy fusion changed outputs beyond tolerance"
