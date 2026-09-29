"""
neuravex-spatial3d: High-performance 3D Spatial Perception, Sensor Fusion,
Native DEM, Real-Time Volumetric Geometry, and Spatial Memory Counting SDK.

Supported Features:
  - Native 3D Bounding Boxes (X, Y, Z, Length, Width, Height, Yaw/Pitch/Roll)
  - Camera Pinhole / Fisheye / Depth Intrinsics Calibration
  - Multi-Modal LiDAR Sensor Fusion (Point Cloud unprojection, LiDAR-to-Camera calibration)
  - Native DEM (Digital Elevation Model) terrain surface fitting & elevation mapping
  - Native Instance & Semantic Mask-Guided 3D Unprojection
  - Native Spatial Entity Marking & Persistent Identity Store
  - Native Spatial Flow & Volumetric Counting Engine
  - Real-Time Hardware Acceleration: CUDA, cuDNN, Intel (oneDNN/AVX-512), AMD (ROCm/AVX2)
"""

__version__ = "0.1.0"
__author__ = "ELANGKATHIR11"

from .sensors import CameraIntrinsics, LiDARConfig, SensorFusionEngine
from .geometry3d import (
    BoundingBox3D,
    compute_lwh_from_pointcloud,
    compute_lwh_from_mask_depth,
    oriented_iou3d,
)
from .dem import NativeDEMSurface, compute_elevation_profile
from .segmentation import Mask3DProjector, robust_mask_unproject
from .marking import SpatialMarkerStore, MarkedEntity
from .counting import SpatialCounter, CountingZone3D
from .hardware import DeviceContext, get_optimal_device

__all__ = [
    "CameraIntrinsics",
    "LiDARConfig",
    "SensorFusionEngine",
    "BoundingBox3D",
    "compute_lwh_from_pointcloud",
    "compute_lwh_from_mask_depth",
    "oriented_iou3d",
    "NativeDEMSurface",
    "compute_elevation_profile",
    "Mask3DProjector",
    "robust_mask_unproject",
    "SpatialMarkerStore",
    "MarkedEntity",
    "SpatialCounter",
    "CountingZone3D",
    "DeviceContext",
    "get_optimal_device",
]
