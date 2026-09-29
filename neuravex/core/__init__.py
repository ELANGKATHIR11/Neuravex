"""
Neuravex Unified Output Schema: ObjectState + FrameState.

Every Neuravex inference produces one FrameState per frame, containing a list of ObjectState
records — one per detected object. This is the SINGLE source of truth for all downstream
consumers (tracking, counting, visualization, export, serialization).

Fields that cannot be computed (missing calibration, untrained heads, etc.) are set to None
with a corresponding source/flag indicating why. Neuravex NEVER fabricates data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class ObjectState:
    """Per-object unified perception state. One independent record per detected object."""
    # Identity
    id: int
    class_id: int
    class_name: str = ""
    probabilities: Optional[List[float]] = None  # per-class softmax probabilities

    # 2D Detection
    bbox2d: Optional[List[float]] = None  # [x1, y1, x2, y2] in pixels
    score: float = 0.0  # detection confidence

    # Instance Segmentation
    mask: Optional[np.ndarray] = None  # (H, W) binary uint8 mask
    mask_confidence: float = 0.0
    mask_source: str = "none"  # 'proto_assembly' | 'embedding' | 'box_fallback' | 'none'

    # Metric Depth
    depth: Optional[float] = None  # metric Z in meters
    depth_confidence: float = 0.0
    depth_source: str = "UNVALIDATED"  # 'dem_head' | 'detection_head' | 'UNVALIDATED'

    # 3D Camera Coordinates (meters)
    xyz: Optional[List[float]] = None  # [X, Y, Z] in camera frame

    # 3D Bounding Box
    lwh: Optional[List[float]] = None  # [L, W, H] in meters (positive, from model head)
    yaw: Optional[float] = None  # rotation in radians (from model head)
    bbox3d_corners: Optional[np.ndarray] = None  # (8, 3) world-frame 3D corners
    bbox3d_projected: Optional[np.ndarray] = None  # (8, 2) projected pixel corners
    geometry_confidence: float = 0.0

    # Distance
    distance: Optional[float] = None  # Euclidean distance from camera in meters

    # Tracking (populated when tracker is active)
    track_id: Optional[int] = None
    velocity: Optional[List[float]] = None  # [vx, vy, vz] m/s
    acceleration: Optional[List[float]] = None  # [ax, ay, az] m/s²
    heading: Optional[float] = None  # heading angle in degrees
    track_age: int = 0
    movement_state: str = "unknown"  # 'stationary' | 'walking' | 'running' | 'erratic_turn' | ...

    # Counting (populated when counter is active)
    count_state: Optional[str] = None  # 'pre_zone' | 'in_zone' | 'counted' | 'post_zone'

    # Terrain / DEM (populated when real DEM data is available)
    terrain_elevation: Optional[float] = None  # meters above reference datum
    height_above_ground: Optional[float] = None  # object bottom - terrain elevation

    # Uncertainty
    xyz_uncertainty: Optional[List[float]] = None  # [sigma_x, sigma_y, sigma_z]
    lwh_uncertainty: Optional[List[float]] = None  # [sigma_l, sigma_w, sigma_h]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dictionary, excluding large arrays."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                if v.size <= 24:  # small arrays (corners)
                    d[k] = v.tolist()
                else:
                    d[k] = f"<ndarray shape={v.shape}>"
            else:
                d[k] = v
        return d


@dataclass
class FrameState:
    """Per-frame unified perception output."""
    frame_id: int = 0
    timestamp: float = 0.0
    image_width: int = 0
    image_height: int = 0

    # Objects
    objects: List[ObjectState] = field(default_factory=list)

    # Frame-level dense outputs (optional, large)
    depth_map: Optional[np.ndarray] = None  # (H, W) metric depth
    terrain_map: Optional[np.ndarray] = None  # (H, W) terrain elevation
    depth_confidence_map: Optional[np.ndarray] = None  # (H, W) confidence

    # Counts
    total_visible: int = 0
    total_active_tracks: int = 0
    total_unique_ids: int = 0

    # Validation flags — Neuravex never claims what it cannot prove
    depth_validated: bool = False
    masks_validated: bool = False
    intrinsics_provided: bool = False
    dem_data_available: bool = False

    # Routing / compute stats
    routing_stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dictionary."""
        d = {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "image_size": [self.image_width, self.image_height],
            "total_visible": self.total_visible,
            "depth_validated": self.depth_validated,
            "masks_validated": self.masks_validated,
            "intrinsics_provided": self.intrinsics_provided,
            "objects": [o.to_dict() for o in self.objects],
        }
        if self.routing_stats:
            d["routing_stats"] = {
                k: (float(v) if hasattr(v, 'item') else v)
                for k, v in self.routing_stats.items()
                if k != "gate"
            }
        return d
