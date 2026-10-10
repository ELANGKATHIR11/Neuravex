"""
Neuravex Unified Canonical Output Schema: ObjectState, FrameState, and CanonicalPrediction.

Every Neuravex inference outputs valid CanonicalPrediction records within FrameState.
Fields that cannot be computed or are unrequested default to explicit None.
No incompatible parallel schemas exist across subsystems.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class CanonicalPrediction:
    """
    Contractually specified canonical per-object perception output:
    object_id, class_id, confidence, bbox_xyxy, mask, center_xy, depth,
    xyz, lwh, yaw, rotation, terrain_elevation, uncertainty, track_state, ood_score
    """
    object_id: int
    class_id: int
    confidence: float
    bbox_xyxy: List[float]                                     # [x1, y1, x2, y2]
    mask: Optional[np.ndarray] = None                          # (H, W) binary uint8 mask or None
    center_xy: Optional[List[float]] = None                    # [cx, cy] in pixels
    depth: Optional[float] = None                              # Metric Z in meters
    xyz: Optional[List[float]] = None                          # [X, Y, Z] in camera metric frame
    lwh: Optional[List[float]] = None                          # [L, W, H] physical dimensions
    yaw: Optional[float] = None                                # Rotation heading in radians
    rotation: Optional[List[List[float]]] = None               # 3x3 SO(3) rotation matrix
    terrain_elevation: Optional[float] = None                  # Metric ground elevation
    uncertainty: Optional[Dict[str, float]] = None             # {"sigma_xyz": ..., "sigma_lwh": ...}
    track_state: Optional[Dict[str, Any]] = None               # {"track_id": int, "velocity": [...], "age": int}
    ood_score: Optional[float] = None                          # Free energy out-of-distribution score

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                if v.size <= 24:
                    d[k] = v.tolist()
                else:
                    d[k] = f"<ndarray shape={v.shape}>"
            else:
                d[k] = v
        return d


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
    movement_state: str = "unknown"

    # Counting
    count_state: Optional[str] = None

    # Terrain / DEM
    terrain_elevation: Optional[float] = None
    height_above_ground: Optional[float] = None

    # Uncertainty & OOD
    xyz_uncertainty: Optional[List[float]] = None
    lwh_uncertainty: Optional[List[float]] = None
    ood_score: Optional[float] = None

    # Canonical prediction representation
    canonical: Optional[CanonicalPrediction] = None

    def to_canonical(self) -> CanonicalPrediction:
        """Produce the single canonical contract prediction."""
        cxcy = None
        if self.bbox2d is not None and len(self.bbox2d) == 4:
            cxcy = [(self.bbox2d[0] + self.bbox2d[2]) * 0.5, (self.bbox2d[1] + self.bbox2d[3]) * 0.5]

        unc = {}
        if self.xyz_uncertainty is not None:
            unc["xyz_sigma"] = self.xyz_uncertainty
        if self.lwh_uncertainty is not None:
            unc["lwh_sigma"] = self.lwh_uncertainty

        trk = None
        if self.track_id is not None:
            trk = {
                "track_id": self.track_id,
                "velocity": self.velocity,
                "acceleration": self.acceleration,
                "heading": self.heading,
                "age": self.track_age,
                "movement_state": self.movement_state
            }

        return CanonicalPrediction(
            object_id=self.id,
            class_id=self.class_id,
            confidence=float(self.score),
            bbox_xyxy=self.bbox2d if self.bbox2d is not None else [0.0, 0.0, 0.0, 0.0],
            mask=self.mask,
            center_xy=cxcy,
            depth=self.depth,
            xyz=self.xyz,
            lwh=self.lwh,
            yaw=self.yaw,
            rotation=None,
            terrain_elevation=self.terrain_elevation,
            uncertainty=unc if unc else None,
            track_state=trk,
            ood_score=self.ood_score
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if k == "canonical":
                continue
            if isinstance(v, np.ndarray):
                if v.size <= 24:
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

    # Frame-level dense outputs
    depth_map: Optional[np.ndarray] = None
    terrain_map: Optional[np.ndarray] = None
    depth_confidence_map: Optional[np.ndarray] = None

    # Counts
    total_visible: int = 0
    total_active_tracks: int = 0
    total_unique_ids: int = 0

    # Validation flags
    depth_validated: bool = False
    masks_validated: bool = False
    intrinsics_provided: bool = False
    dem_data_available: bool = False

    # Routing stats
    routing_stats: Optional[Dict[str, Any]] = None

    def canonical_predictions(self) -> List[CanonicalPrediction]:
        """Returns list of CanonicalPrediction objects."""
        return [o.to_canonical() for o in self.objects]

    def to_dict(self) -> Dict[str, Any]:
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
