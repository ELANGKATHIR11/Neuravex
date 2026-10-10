"""
Neuravex Decoupled FastAPI Server Layer.
Provides high-performance, non-blocking asynchronous REST endpoints for
computer vision, 3D perception, multi-modal analysis, and session-isolated tracking.
"""

import time
import base64
import uuid
import asyncio
from typing import Dict, Any, Optional
import numpy as np
import cv2
import torch

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    HealthResponse,
    CapabilitiesResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    FrameStateSchema,
    ObjectStateSchema,
    TrackingSessionCreate,
    TrackingSessionResponse,
    EntityResponse,
)

# Active tracking sessions and entities
TRACKING_SESSIONS: Dict[str, Dict[str, Any]] = {}
ENTITIES_STORE: Dict[str, Dict[str, Any]] = {}
LOADED_MODELS: Dict[str, Any] = {}


def get_loaded_model(size: str = "nano"):
    """Fetch or lazy-load model into worker cache."""
    clean_size = size.lower().replace("neuravex-", "").replace("neuravex_", "")
    if clean_size not in LOADED_MODELS:
        from ..models.neuravex import build_neuravex
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = build_neuravex(size=clean_size).to(device).eval()
        LOADED_MODELS[clean_size] = model
    return LOADED_MODELS[clean_size]


def decode_image_bytes(b64_str: str) -> np.ndarray:
    """Safely decode base64 string into BGR numpy image."""
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        return img
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid base64 image data: {str(e)}"
        )


def create_app() -> FastAPI:
    """Factory creating the configured FastAPI application."""
    app = FastAPI(
        title="Neuravex Vision & Spatial 3D API",
        version="0.3.0",
        description="High-performance edge & cloud multi-modal perception REST API.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HealthResponse(
            status="ok",
            version="0.3.0",
            device=device,
            ready=True
        )

    @app.get("/api/v1/health/ready", response_model=HealthResponse)
    async def health_ready():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HealthResponse(
            status="ready",
            version="0.3.0",
            device=device,
            ready=True
        )

    @app.get("/api/v1/capabilities", response_model=CapabilitiesResponse)
    async def capabilities():
        cuda_avail = torch.cuda.is_available()
        vram = 0.0
        device_name = "CPU"
        if cuda_avail:
            device_name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2)

        return CapabilitiesResponse(
            platform=torch.__version__,
            cuda_available=cuda_avail,
            device_name=device_name,
            vram_total_mb=vram,
            supported_models=["pico", "femto", "nano", "lite", "edge", "pro", "omni"],
            tasks=["det", "seg", "depth", "pose", "dem", "tracking", "counting"]
        )

    def _sync_analyze(img_bgr: np.ndarray, model_size: str, conf_thresh: float) -> Dict[str, Any]:
        from ..engine.precision_perception import PrecisionPerceptionPipeline
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = PrecisionPerceptionPipeline(conf_thresh=conf_thresh, device=dev)
        frame_state = pipeline.process(img_bgr)
        return frame_state

    @app.post("/api/v1/vision/analyze", response_model=AnalyzeResponse)
    async def analyze(req: AnalyzeRequest):
        t0 = time.time()
        if not req.image_base64:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Must provide 'image_base64' in request payload."
            )

        img_bgr = decode_image_bytes(req.image_base64)
        h, w = img_bgr.shape[:2]

        # Non-blocking execution in worker thread
        frame_state = await asyncio.to_thread(_sync_analyze, img_bgr, req.model_size, req.conf_threshold)

        latency_ms = (time.time() - t0) * 1000.0
        fps = 1000.0 / max(latency_ms, 1e-3)
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        objs_schema = []
        for o in frame_state.objects:
            objs_schema.append(ObjectStateSchema(
                id=o.id,
                class_id=o.class_id,
                class_name=o.class_name,
                bbox2d=o.bbox2d,
                score=o.score,
                mask_confidence=o.mask_confidence,
                mask_source=o.mask_source,
                depth=o.depth,
                depth_confidence=o.depth_confidence,
                depth_source=o.depth_source,
                xyz=o.xyz,
                lwh=o.lwh,
                yaw=o.yaw,
                geometry_confidence=o.geometry_confidence,
                distance=o.distance,
                track_id=o.track_id,
                ood_score=o.ood_score
            ))

        fs_schema = FrameStateSchema(
            frame_id=frame_state.frame_id,
            timestamp=frame_state.timestamp,
            image_width=w,
            image_height=h,
            objects=objs_schema,
            total_visible=len(objs_schema),
            depth_validated=frame_state.depth_validated,
            masks_validated=frame_state.masks_validated,
            intrinsics_provided=frame_state.intrinsics_provided
        )

        return AnalyzeResponse(
            success=True,
            frame_state=fs_schema,
            fps=round(fps, 1),
            latency_ms=round(latency_ms, 2),
            device=dev
        )

    @app.post("/api/v1/vision/segment", response_model=AnalyzeResponse)
    async def segment(req: AnalyzeRequest):
        req.tasks = ["seg", "det"]
        return await analyze(req)

    @app.post("/api/v1/tracking/sessions", response_model=TrackingSessionResponse)
    async def create_tracking_session(req: TrackingSessionCreate):
        sess_id = req.session_id or str(uuid.uuid4())[:8]
        if sess_id in TRACKING_SESSIONS:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Tracking session '{sess_id}' already exists."
            )

        from neuravex_spatial3d.live import LiveObjectTracker
        TRACKING_SESSIONS[sess_id] = {
            "tracker": LiveObjectTracker(max_cosine_dist=req.distance_threshold, max_missed=req.max_missed),
            "created_at": time.time(),
            "model_size": req.model_size,
            "total_unique": 0
        }

        return TrackingSessionResponse(
            session_id=sess_id,
            created_at=TRACKING_SESSIONS[sess_id]["created_at"],
            active_tracks=0,
            total_unique_tracked=0
        )

    @app.post("/api/v1/tracking/sessions/{session_id}/frames", response_model=AnalyzeResponse)
    async def process_session_frame(session_id: str, req: AnalyzeRequest):
        if session_id not in TRACKING_SESSIONS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tracking session '{session_id}' not found."
            )

        res = await analyze(req)
        tracker = TRACKING_SESSIONS[session_id]["tracker"]

        from neuravex_spatial3d.geometry3d import BoundingBox3D
        boxes_3d = []
        for obj in res.frame_state.objects:
            c = obj.xyz or [0.0, 0.0, 1.0]
            s = obj.lwh or [0.5, 0.5, 0.5]
            b = BoundingBox3D(
                center=tuple(c),
                size_lwh=tuple(s),
                yaw=obj.yaw or 0.0,
                score=obj.score,
                class_id=obj.class_id,
                class_name=obj.class_name
            )
            boxes_3d.append(b)

        tracked = tracker.update(boxes_3d)
        for tb in tracked:
            eid = f"ent_{tb.track_id}"
            ENTITIES_STORE[eid] = {
                "entity_id": eid,
                "track_id": tb.track_id,
                "class_name": tb.class_name,
                "xyz": [round(v.item(), 3) for v in tb.center],
                "lwh": [round(v.item(), 3) for v in tb.lwh],
                "trajectory_len": 1
            }

        TRACKING_SESSIONS[session_id]["total_unique"] = len(tracker.active_tracks)
        return res

    @app.delete("/api/v1/tracking/sessions/{session_id}")
    async def delete_tracking_session(session_id: str):
        if session_id not in TRACKING_SESSIONS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tracking session '{session_id}' not found."
            )
        del TRACKING_SESSIONS[session_id]
        return {"status": "deleted", "session_id": session_id}

    @app.get("/api/v1/entities/{entity_id}", response_model=EntityResponse)
    async def get_entity(entity_id: str):
        if entity_id not in ENTITIES_STORE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Entity '{entity_id}' not found in active spatial memory."
            )
        data = ENTITIES_STORE[entity_id]
        return EntityResponse(
            entity_id=data["entity_id"],
            track_id=data["track_id"],
            class_name=data["class_name"],
            last_known_xyz=data["xyz"],
            last_known_lwh=data["lwh"],
            trajectory_length=data["trajectory_len"]
        )

    return app


app = create_app()
