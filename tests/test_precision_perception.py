"""
Test Suite for Precision Perception Pipeline:
Verifies:
1. Pipeline initialization without external detector.
2. Inference on synthetic image without errors.
3. Mathematical 3D wireframe projection via pinhole model.
4. ObjectState/FrameState output structure.
"""

import os
import torch
import numpy as np
import pytest

from neuravex.engine.precision_perception import PrecisionPerceptionPipeline, project_3d_corners_to_image
from neuravex.geometry.camera import CameraIntrinsics


def test_precision_perception_synthetic():
    pipeline = PrecisionPerceptionPipeline(device="cpu", model_size="pico", num_classes=5)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw simple bright rectangle resembling foreground object
    dummy_img[100:300, 150:350] = (200, 200, 200)

    res = pipeline.analyze(dummy_img)
    assert "total_visible" in res or "total_animals_counted" in res
    assert "detected_objects" in res or "objects" in res
    # Pipeline must not crash on synthetic input


def test_3d_wireframe_corners_projection():
    """3D box corners must be mathematically projected via pinhole model."""
    K = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    center = np.array([0.0, 0.0, 5.0])
    lwh = np.array([1.0, 0.5, 0.8])
    yaw = 0.0

    corners = project_3d_corners_to_image(center, lwh, yaw, K)
    assert corners is not None
    assert corners.shape == (8, 2)

    # Center of projected box should be near image center (object at origin, 5m away)
    mean_u = corners[:, 0].mean()
    mean_v = corners[:, 1].mean()
    assert abs(mean_u - K.cx) < 20, f"Mean U={mean_u} not near cx={K.cx}"
    assert abs(mean_v - K.cy) < 20, f"Mean V={mean_v} not near cy={K.cy}"


def test_pipeline_has_no_yolo():
    """The pipeline must NOT import or reference ultralytics/YOLO."""
    import inspect
    from neuravex.engine import precision_perception
    source = inspect.getsource(precision_perception)
    assert "ultralytics" not in source
    assert "YOLO(" not in source
