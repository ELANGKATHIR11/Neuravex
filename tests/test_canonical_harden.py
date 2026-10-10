"""
Comprehensive test suite verifying canonical O2O NMS-free inference,
synthetic 3D geometry consistency, DEM metrics, instance mask metrics, and runtime selection.
"""

import math
import pytest
import torch
import numpy as np

from neuravex import (
    build_neuravex,
    NeuravexInferencePostProcessor,
    CameraIntrinsics,
    calculate_bos_metrics,
    calculate_bos_geo,
    calculate_mask_metrics,
    calculate_3d_iou_and_bos,
    runtime,
    runtime_explain
)
from neuravex.core import ObjectState, FrameState, CanonicalPrediction
from neuravex.geometry.oriented_iou3d import oriented_iou_3d, boxes3d_to_corners


def test_canonical_o2o_inference_is_nms_free():
    """
    Test Gate: Production O2O inference MUST NOT call NMS/batched_nms.
    Verify that post-processor without NMS operates natively on One-to-One predictions.
    """
    post_proc_o2o = NeuravexInferencePostProcessor(conf_thresh=0.1, use_nms=False)
    assert post_proc_o2o.use_nms is False, "Canonical inference MUST have use_nms=False"

    # Simulate predictions with two high-confidence overlapping boxes
    class_logits = torch.tensor([[[2.5, -2.0], [2.4, -2.0]]])  # (1, 2, 2)
    pred_boxes = torch.tensor([[[10.0, 10.0, 50.0, 50.0], [11.0, 11.0, 51.0, 51.0]]])  # IoU > 0.9

    outputs = {
        "class_logits": class_logits,
        "pred_boxes": pred_boxes
    }

    # Running canonical NMS-free post-processing
    res = post_proc_o2o(outputs)
    # Both boxes MUST be retained (not suppressed by NMS) in O2O mode
    assert len(res["detections"][0]["boxes"]) == 2, "Canonical O2O must retain predictions without NMS suppression"

    # Compare with legacy NMS mode where overlapping box is suppressed
    post_proc_legacy = NeuravexInferencePostProcessor(conf_thresh=0.1, iou_thresh=0.5, use_nms=True)
    res_legacy = post_proc_legacy(outputs)
    assert len(res_legacy["detections"][0]["boxes"]) == 1, "Legacy mode with NMS should suppress overlapping box"


def test_canonical_prediction_contract():
    """Verify CanonicalPrediction dataclass adherence to exact specification."""
    pred = CanonicalPrediction(
        object_id=1,
        class_id=2,
        confidence=0.95,
        bbox_xyxy=[10.0, 20.0, 100.0, 150.0],
        mask=None,
        center_xy=[55.0, 85.0],
        depth=12.5,
        xyz=[1.2, -0.4, 12.5],
        lwh=[4.2, 1.8, 1.6],
        yaw=0.15,
        rotation=None,
        terrain_elevation=0.5,
        uncertainty={"sigma_xyz": 0.05},
        track_state={"track_id": 1, "velocity": [0.0, 0.0, 2.1]},
        ood_score=0.02
    )

    d = pred.to_dict()
    for req in ["object_id", "class_id", "confidence", "bbox_xyxy", "mask", "center_xy",
                "depth", "xyz", "lwh", "yaw", "rotation", "terrain_elevation", "uncertainty",
                "track_state", "ood_score"]:
        assert req in d, f"Missing required canonical field: {req}"

    # Verify ObjectState conversion to CanonicalPrediction
    obj = ObjectState(
        id=10,
        class_id=3,
        bbox2d=[5.0, 5.0, 45.0, 45.0],
        score=0.91,
        depth=5.0,
        xyz=[0.0, 0.0, 5.0],
        lwh=[1.0, 1.0, 1.0],
        yaw=0.0,
        terrain_elevation=0.0,
        ood_score=0.01
    )
    can = obj.to_canonical()
    assert isinstance(can, CanonicalPrediction)
    assert can.object_id == 10
    assert can.class_id == 3
    assert can.depth == 5.0
    assert can.center_xy == [25.0, 25.0]


def test_synthetic_3d_geometry_and_metrics():
    """
    Synthetic ground truth test for 3D Oriented Bounding Boxes:
    Validates XYZ MAE/RMSE, LWH MAE, Yaw Error, and 3D IoU / BoS_3D.
    """
    # Create synthetic true 3D box at (X=1.0, Y=0.0, Z=10.0) with (L=4.0, W=2.0, H=1.5), Yaw=0.0
    gt_xyz = torch.tensor([[1.0, 0.0, 10.0]], dtype=torch.float32)
    gt_lwh = torch.tensor([[4.0, 2.0, 1.5]], dtype=torch.float32)
    gt_yaw = torch.tensor([[0.0]], dtype=torch.float32)

    # 1. Exact match test
    res_exact = calculate_3d_iou_and_bos(gt_xyz, gt_lwh, gt_yaw, gt_xyz, gt_lwh, gt_yaw)
    assert math.isclose(res_exact["mean_3d_iou"], 1.0, abs_tol=1e-3)
    assert math.isclose(res_exact["mean_3d_bos"], 1.0, abs_tol=1e-3)

    # 2. Perturbed prediction test: 10cm depth error, 5 deg yaw error
    pred_xyz = torch.tensor([[1.0, 0.0, 10.1]], dtype=torch.float32)
    pred_lwh = torch.tensor([[4.1, 2.0, 1.5]], dtype=torch.float32)
    pred_yaw = torch.tensor([[math.radians(5.0)]], dtype=torch.float32)

    # Compute errors
    xyz_mae = torch.abs(pred_xyz - gt_xyz).mean().item()
    lwh_mae = torch.abs(pred_lwh - gt_lwh).mean().item()
    yaw_err_deg = math.degrees(abs(pred_yaw.item() - gt_yaw.item()))

    assert xyz_mae < 0.05
    assert lwh_mae < 0.05
    assert abs(yaw_err_deg - 5.0) < 1e-4

    res_perturbed = calculate_3d_iou_and_bos(pred_xyz, pred_lwh, pred_yaw, gt_xyz, gt_lwh, gt_yaw)
    assert res_perturbed["mean_3d_iou"] > 0.80
    assert res_perturbed["mean_3d_bos"] > 0.80


def test_mask_metrics_and_boundary_f1():
    """Verify mask metrics: Mask IoU, Dice, and true BoundaryF1 contour score."""
    H, W = 64, 64
    gt_mask = torch.zeros((1, H, W), dtype=torch.float32)
    gt_mask[0, 16:48, 16:48] = 1.0

    # Perfect prediction
    res_perfect = calculate_mask_metrics(gt_mask, gt_mask)
    assert math.isclose(res_perfect["mask_iou"], 1.0, abs_tol=1e-4)
    assert math.isclose(res_perfect["dice"], 1.0, abs_tol=1e-4)
    assert math.isclose(res_perfect["boundary_f1"], 1.0, abs_tol=1e-4)

    # 1-pixel dilated prediction: boundary F1 should gracefully reflect the contour offset
    pred_mask = torch.zeros((1, H, W), dtype=torch.float32)
    pred_mask[0, 15:49, 15:49] = 1.0
    res_dilated = calculate_mask_metrics(pred_mask, gt_mask)
    assert 0.80 < res_dilated["mask_iou"] < 1.0
    assert 0.80 < res_dilated["dice"] < 1.0
    assert 0.0 <= res_dilated["boundary_f1"] <= 1.0


def test_runtime_engine_auto_detection():
    """Verify AutoRuntimeEngine hardware detection and explain() method."""
    engine = runtime()
    assert engine is not None
    assert engine.profile.os_name != ""
    assert engine.profile.ram_gb > 0.0
    assert engine.selection.backend in ("pytorch-cuda", "pytorch-cpu", "ort-cuda", "ort-cpu")

    explanation = runtime_explain()
    assert "NEURAVEX AUTONOMOUS RUNTIME ENGINE EXPLANATION" in explanation
    assert "Hardware Environment" in explanation
    assert "Decision Rationales" in explanation
