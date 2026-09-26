import os
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class PipelineConfig:
    # Model parameters
    num_classes: int = 80
    base_c: int = 48
    depth_mul: float = 1.0
    device: str = "cuda"  # Auto-falls back to cpu if cuda unavailable
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45

    # Camera Intrinsics (pinhole model: fx, fy, cx, cy)
    resolution: Tuple[int, int] = (1280, 720)  # (width, height)
    fx: float = 1050.0
    fy: float = 1050.0
    cx: float = 640.0
    cy: float = 360.0

    # Feed source: "webcam", "synthetic", or a file path / rtsp url
    source_type: str = "webcam"  # Default to real live camera feed (auto-falls back to synthetic if camera unavailable)
    camera_index: int = 0
    target_fps: int = 25
    scene_mode: str = "conveyor"  # "conveyor" (apple/produce flow counter) or "pasture" (cattle & human tracking)

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
