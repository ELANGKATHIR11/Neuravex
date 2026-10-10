"""
Typed Pydantic Schemas for Neuravex FastAPI Layer.
Conforms strictly to the canonical FrameState, ObjectState, and CanonicalPrediction contracts.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ObjectStateSchema(BaseModel):
    id: int
    class_id: int
    class_name: str = ""
    probabilities: Optional[List[float]] = None
    bbox2d: Optional[List[float]] = None
    score: float = 0.0
    mask_confidence: float = 0.0
    mask_source: str = "none"
    depth: Optional[float] = None
    depth_confidence: float = 0.0
    depth_source: str = "UNVALIDATED"
    xyz: Optional[List[float]] = None
    lwh: Optional[List[float]] = None
    yaw: Optional[float] = None
    geometry_confidence: float = 0.0
    distance: Optional[float] = None
    track_id: Optional[int] = None
    velocity: Optional[List[float]] = None
    heading: Optional[float] = None
    track_age: int = 0
    movement_state: str = "unknown"
    terrain_elevation: Optional[float] = None
    height_above_ground: Optional[float] = None
    ood_score: Optional[float] = None


class FrameStateSchema(BaseModel):
    frame_id: int = 0
    timestamp: float = 0.0
    image_width: int = 0
    image_height: int = 0
    objects: List[ObjectStateSchema] = Field(default_factory=list)
    total_visible: int = 0
    total_active_tracks: int = 0
    total_unique_ids: int = 0
    depth_validated: bool = False
    masks_validated: bool = False
    intrinsics_provided: bool = False
    dem_data_available: bool = False
    routing_stats: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.3.0"
    device: str = "cpu"
    ready: bool = True


class CapabilitiesResponse(BaseModel):
    platform: str
    cuda_available: bool
    device_name: str
    vram_total_mb: float
    supported_models: List[str]
    tasks: List[str]


class AnalyzeRequest(BaseModel):
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    conf_threshold: float = Field(default=0.20, ge=0.01, le=1.0)
    tasks: Optional[List[str]] = Field(default=["det", "seg", "depth"])
    model_size: str = "nano"


class AnalyzeResponse(BaseModel):
    success: bool = True
    frame_state: FrameStateSchema
    fps: float = 0.0
    latency_ms: float = 0.0
    device: str = "cpu"


class TrackingSessionCreate(BaseModel):
    session_id: Optional[str] = None
    model_size: str = "nano"
    max_missed: int = 15
    distance_threshold: float = 2.5


class TrackingSessionResponse(BaseModel):
    session_id: str
    created_at: float
    active_tracks: int = 0
    total_unique_tracked: int = 0


class EntityResponse(BaseModel):
    entity_id: str
    track_id: int
    class_name: str
    last_known_xyz: Optional[List[float]] = None
    last_known_lwh: Optional[List[float]] = None
    trajectory_length: int = 0
