"""
Unit and regression tests verifying repository repairs across 3D geometry,
tiled segmentation, counting, and canonical contracts.
"""

import pytest
import torch
import numpy as np

from neuravex_spatial3d.sensors import CameraIntrinsics
from neuravex_spatial3d.geometry3d import BoundingBox3D, compute_lwh_from_pointcloud
from neuravex_spatial3d.counting import CountingZone3D, SpatialCounter
from neuravex.engine.sliced_segmenter import RealTimeSlicedPatchSegmenter


def test_no_synthetic_id_fallback_in_counting():
    """Verify that CountingZone3D rejects boxes with None track_id instead of using id(box)."""
    zone = CountingZone3D("zone_1", "entry_gate", 0.0, 5.0, 0.0, 5.0, 0.0, 10.0)
    
    # Box without track_id
    box_no_id = BoundingBox3D(center=(2.0, 2.0, 5.0), size_lwh=(1.0, 1.0, 1.0), track_id=None)
    result = zone.update(box_no_id)
    assert result is False, "Zone must not register count when track_id is None"
    assert zone.total_count == 0

    # Box with valid track_id
    box_valid = BoundingBox3D(center=(2.0, 2.0, 5.0), size_lwh=(1.0, 1.0, 1.0), track_id=42)
    result = zone.update(box_valid)
    assert result is True, "Zone must register count for valid track_id"
    assert zone.total_count == 1

    # Same box in next frame should not double count
    result_dup = zone.update(box_valid)
    assert result_dup is False, "Duplicate track_id must not increment cumulative zone count"
    assert zone.total_count == 1


def test_sliced_segmenter_uncalibrated_contract():
    """Verify that RealTimeSlicedPatchSegmenter does not fabricate animal heights or alternating yaw."""
    segmenter = RealTimeSlicedPatchSegmenter(slice_size=(320, 320))
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    class DummyDetector:
        def __call__(self, x):
            # Returns one detection: score 0.8, box [50, 50, 150, 150]
            return [{
                "scores": torch.tensor([0.8]),
                "boxes": torch.tensor([[50.0, 50.0, 150.0, 150.0]])
            }]

    detector = DummyDetector()
    results = segmenter.detect_and_segment_tiled(
        img_bgr=dummy_img,
        detector=detector,
        class_name="Langur",
        class_id=4,
        K_intrinsics=None
    )

    assert len(results) > 0
    det = results[0]
    # Geometry must NOT be fabricated from assumed height
    assert det["depth_m"] is None, "Depth must be None when uncalibrated"
    assert det["yaw_deg"] is None, "Yaw must be None when uncalibrated"
    assert det["dimensions"] is None, "Dimensions must be None when uncalibrated"
    assert det["center_xyz"] is None, "3D Center must be None when uncalibrated"
    assert det["geometry_status"] == "UNVALIDATED_NO_CALIBRATION"


def test_pca_bounding_box_computation():
    """Verify that compute_lwh_from_pointcloud accurately computes physical extents from real 3D points."""
    # Synthetic cuboid point cloud: L=2.0 (X), W=1.0 (Y), H=0.5 (Z)
    x = torch.linspace(-1.0, 1.0, 20)
    y = torch.linspace(-0.5, 0.5, 10)
    z = torch.linspace(-0.25, 0.25, 5)
    gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
    pts = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)

    center, lwh, yaw = compute_lwh_from_pointcloud(pts)

    assert torch.allclose(center, torch.zeros(3), atol=1e-2), f"Expected center ~0, got {center}"
    assert abs(lwh[0].item() - 2.0) < 0.1, f"Expected L ~2.0, got {lwh[0].item()}"
    assert abs(lwh[1].item() - 1.0) < 0.1, f"Expected W ~1.0, got {lwh[1].item()}"
    assert abs(lwh[2].item() - 0.5) < 0.1, f"Expected H ~0.5, got {lwh[2].item()}"
