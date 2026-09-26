import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import PipelineConfig
from .pipeline import NeuravexVisionPipeline
from .memory import memory_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("neuravex.server")

# Global pipeline instance and background broadcast task
pipeline_instance: Optional[NeuravexVisionPipeline] = None
broadcast_task: Optional[asyncio.Task] = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected. Active clients: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket client disconnected. Active clients: %d", len(self.active_connections))

    async def broadcast_json(self, data: Dict[str, Any]):
        if not self.active_connections:
            return
        msg = json.dumps(data)
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(msg)
            except Exception:
                disconnected.append(connection)
        
        for dead_conn in disconnected:
            self.disconnect(dead_conn)

manager = ConnectionManager()

async def frame_broadcaster_loop():
    """Continuous async loop running pipeline frame inference and broadcasting to clients."""
    logger.info("Starting real-time frame broadcaster loop...")
    while True:
        try:
            if pipeline_instance:
                # Run inference step in thread pool to keep asyncio event loop non-blocking
                frame_data = await asyncio.to_thread(pipeline_instance.process_next_frame)
                await manager.broadcast_json(frame_data)
                
                # Check if new events were generated
                if pipeline_instance.history_events:
                    latest_ev = pipeline_instance.history_events[0]
                    # Tag with type: 'event' for frontend listener
                    ev_msg = dict(latest_ev)
                    ev_msg["type"] = "event"
                    # Broadcast only on new events
            
            # Target ~25-30 FPS pacing
            await asyncio.sleep(0.035)
        except asyncio.CancelledError:
            logger.info("Broadcaster loop cancelled.")
            break
        except Exception as e:
            logger.error("Error in broadcaster loop: %s", e, exc_info=True)
            await asyncio.sleep(0.5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance, broadcast_task
    logger.info("Initializing Neuravex Backend Server...")
    config = PipelineConfig()
    pipeline_instance = NeuravexVisionPipeline(config)
    broadcast_task = asyncio.create_task(frame_broadcaster_loop())
    yield
    logger.info("Shutting down Neuravex Backend...")
    if broadcast_task:
        broadcast_task.cancel()
        try:
            await broadcast_task
        except asyncio.CancelledError:
            pass

app = FastAPI(
    title="Neuravex CV Real-Time Backend",
    description="Real-time Computer Vision, Cattle & Human Spatial Tracking, and 3D Perception API",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for local Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= REST Endpoints =================

@app.get("/api/status")
async def get_status():
    if not pipeline_instance:
        return {"status": "offline", "version": "0.1.0", "uptimeSeconds": 0, "activeStreams": 0}
    
    uptime = int(time.time() - pipeline_instance.start_time)
    return {
        "status": "healthy",
        "version": "0.1.0",
        "uptimeSeconds": uptime,
        "activeStreams": len(manager.active_connections),
        "sensorInfo": {
            "model": "Neuravex-Spatial-RGBD",
            "serial": "NX-8820-GPU",
            "baselineMm": 120.0,
            "fovHorizontalDeg": 85.0,
            "fovVerticalDeg": 55.0
        },
        "telemetry": {
            "fps": pipeline_instance.current_fps,
            "latencyMs": pipeline_instance.current_latency_ms,
            "frameTimestamp": int(time.time() * 1000),
            "frameIndex": pipeline_instance.frame_index,
            "resolution": {
                "width": pipeline_instance.config.resolution[0],
                "height": pipeline_instance.config.resolution[1]
            },
            "droppedFrames": pipeline_instance.dropped_frames,
            "modelName": "Neuravex-v0.1-CUDA",
            "pipelineStatus": "running" if pipeline_instance.camera_enabled else "paused",
            "cameraEnabled": pipeline_instance.camera_enabled
        }
    }

@app.get("/api/entities")
async def get_entities():
    if not pipeline_instance:
        return []
    return pipeline_instance.active_entities

@app.get("/api/entities/{entity_id}")
async def get_entity(entity_id: str):
    if not pipeline_instance:
        raise HTTPException(status_code=404, detail="Pipeline not initialized")
    for ent in pipeline_instance.active_entities:
        if ent["id"] == entity_id:
            return ent
    raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

@app.get("/api/history")
async def get_history(entityId: Optional[str] = None):
    if not pipeline_instance:
        return []
    if entityId:
        return [ev for ev in pipeline_instance.history_events if ev["entityId"] == entityId]
    return pipeline_instance.history_events

@app.get("/api/telemetry")
async def get_telemetry():
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline offline")
    return {
        "fps": pipeline_instance.current_fps,
        "latencyMs": pipeline_instance.current_latency_ms,
        "frameTimestamp": int(time.time() * 1000),
        "frameIndex": pipeline_instance.frame_index,
        "resolution": {
            "width": pipeline_instance.config.resolution[0],
            "height": pipeline_instance.config.resolution[1]
        },
        "droppedFrames": pipeline_instance.dropped_frames,
        "modelName": "Neuravex-v0.1-CUDA",
        "pipelineStatus": "running"
    }

class ConfigUpdateRequest(BaseModel):
    source_type: Optional[str] = None
    camera_index: Optional[int] = None
    target_fps: Optional[int] = None
    conf_threshold: Optional[float] = None

class MarkEntityRequest(BaseModel):
    tag: Optional[str] = None
    nickname: Optional[str] = None
    notes: Optional[str] = None

class CameraToggleRequest(BaseModel):
    enabled: Optional[bool] = None

class SceneModeRequest(BaseModel):
    mode: str

class PromptExemplarRequest(BaseModel):
    box: List[float]
    label: Optional[str] = "apple"

@app.get("/api/camera/state")
async def get_camera_state():
    if not pipeline_instance:
        return {"status": "offline", "enabled": False}
    return {
        "status": "ok",
        "enabled": pipeline_instance.camera_enabled,
        "sourceType": pipeline_instance.config.source_type,
        "cameraIndex": pipeline_instance.config.camera_index
    }

@app.post("/api/camera/toggle")
async def toggle_camera(req: Optional[CameraToggleRequest] = None):
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline offline")
    target = req.enabled if (req and req.enabled is not None) else not pipeline_instance.camera_enabled
    new_state = pipeline_instance.set_camera_enabled(target)
    
    # Broadcast camera state change to all connected WebSocket clients
    await manager.broadcast_json({
        "type": "camera_state",
        "enabled": new_state,
        "timestamp": int(time.time() * 1000)
    })
    return {"status": "ok", "enabled": new_state}

@app.post("/api/pipeline/config")
async def update_pipeline_config(req: ConfigUpdateRequest):
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline offline")
    if req.source_type is not None:
        pipeline_instance.config.source_type = req.source_type
        pipeline_instance._init_source()
    if req.conf_threshold is not None:
        pipeline_instance.config.conf_threshold = req.conf_threshold
    return {"status": "ok", "config": pipeline_instance.config.__dict__}

@app.post("/api/entities/{entity_id}/mark")
async def mark_entity(entity_id: str, req: MarkEntityRequest):
    """
    Mark an object by ID to remember it permanently in backend memory with a custom name/tag/notes.
    """
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline offline")

    # Find current entity if active
    ent = next((e for e in pipeline_instance.active_entities if e["id"] == entity_id), None)
    pos3d = ent.get("position3D") if ent else None
    dims3d = ent.get("dimensions3D") if ent else None
    ent_type = ent.get("type") if ent else None

    record = memory_store.mark_entity(
        entity_id=entity_id,
        tag=req.tag,
        nickname=req.nickname,
        notes=req.notes,
        entity_type=ent_type,
        position3D=pos3d,
        dimensions3D=dims3d
    )

    # Broadcast memory update to all connected clients
    await manager.broadcast_json({
        "type": "event",
        "id": f"ev-mark-{int(time.time()*1000)}",
        "entityId": entity_id,
        "entityType": ent_type or "object",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "eventType": "state_change",
        "description": f"Object {entity_id} marked as '{record.get('nickname') or record.get('tag')}'",
        "severity": "info"
    })

    return {"status": "ok", "record": record}

@app.delete("/api/entities/{entity_id}/mark")
async def unmark_entity(entity_id: str):
    """
    Remove an object from persistent memory.
    """
    success = memory_store.unmark_entity(entity_id)
    return {"status": "ok", "unmarked": success}

@app.get("/api/memory")
async def get_marked_memory():
    """
    Get all remembered and marked objects.
    """
    return memory_store.get_all_marked()

@app.get("/api/dem")
async def get_dem_info():
    """
    Get current DEM elevation summary and surface profile.
    """
    if not pipeline_instance:
        return {"status": "offline"}
    return {
        "depthStats": pipeline_instance.latest_depth_stats,
        "routerStats": pipeline_instance.latest_router_stats,
        "resolution": pipeline_instance.config.resolution,
        "intrinsics": {
            "fx": pipeline_instance.config.fx,
            "fy": pipeline_instance.config.fy,
            "cx": pipeline_instance.config.cx,
            "cy": pipeline_instance.config.cy
        }
    }

@app.get("/api/video-intelligence")
async def get_video_intelligence():
    """
    Get current high-level video intelligence narrative:
    What happened, how did it happen, and how did it change over time.
    """
    if not pipeline_instance:
        return {"status": "offline", "videoIntelligence": {}}
    return {
        "status": "ok",
        "videoIntelligence": pipeline_instance.latest_video_intelligence,
        "spatialInteractions": pipeline_instance.latest_spatial_interactions
    }

@app.get("/api/interactions")
async def get_spatial_interactions():
    """
    Get all active pairwise spatial interactions, proximity alerts, and contact durations.
    """
    if not pipeline_instance:
        return []
    return pipeline_instance.latest_spatial_interactions

@app.get("/api/anomalies")
async def get_anomalies():
    """
    Get active anomalies and behavioral shift alerts.
    """
    if not pipeline_instance:
        return []
    return pipeline_instance.latest_video_intelligence.get("activeAnomalies", [])

@app.get("/api/flow/stats")
async def get_flow_stats():
    """Returns real-time conveyor flow statistics, throughput, and produce sizing distribution."""
    if not pipeline_instance:
        return {}
    return pipeline_instance.latest_flow_stats

@app.get("/api/flow/zones")
async def get_flow_zones():
    """Returns configured counting tripwire zones."""
    if not pipeline_instance:
        return []
    return pipeline_instance.counting_zones

@app.post("/api/flow/scene-mode")
async def set_scene_mode_endpoint(req: SceneModeRequest):
    """Switches pipeline scene mode between 'conveyor' and 'pasture'."""
    if not pipeline_instance:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    mode = pipeline_instance.set_scene_mode(req.mode)
    await manager.broadcast_json({
        "type": "scene_mode",
        "mode": mode,
        "timestamp": int(time.time() * 1000)
    })
    return {"status": "ok", "sceneMode": mode}

@app.post("/api/flow/prompt-exemplar")
async def add_prompt_exemplar_endpoint(req: PromptExemplarRequest):
    """Registers a SAM-style prompt exemplar bounding box for prototype distillation."""
    if not pipeline_instance:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    record = pipeline_instance.add_prompt_exemplar(req.box, req.label or "apple")
    await manager.broadcast_json({
        "type": "prompt_exemplar_added",
        "exemplar": record,
        "timestamp": int(time.time() * 1000)
    })
    return record

# ================= WebSocket Endpoint =================

@app.websocket("/ws/cv")
async def websocket_cv_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            try:
                data = json.loads(text)
                # Handle client ping
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": data.get("timestamp", int(time.time() * 1000))
                    }))
                elif data.get("type") == "mark_entity" or data.get("command") == "mark_entity":
                    # Instant real-time marking over WebSocket
                    eid = data.get("entityId")
                    if eid:
                        ent = next((e for e in pipeline_instance.active_entities if e["id"] == eid), None) if pipeline_instance else None
                        record = memory_store.mark_entity(
                            entity_id=eid,
                            tag=data.get("tag"),
                            nickname=data.get("nickname"),
                            notes=data.get("notes"),
                            entity_type=ent.get("type") if ent else None,
                            position3D=ent.get("position3D") if ent else None,
                            dimensions3D=ent.get("dimensions3D") if ent else None
                        )
                        await manager.broadcast_json({
                            "type": "event",
                            "id": f"ev-mark-{int(time.time()*1000)}",
                            "entityId": eid,
                            "entityType": ent.get("type") if ent else "object",
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "eventType": "state_change",
                            "description": f"Object {eid} marked as '{record.get('nickname') or record.get('tag')}'",
                            "severity": "info"
                        })
                elif data.get("type") in ("toggle_camera", "set_camera_state"):
                    if pipeline_instance:
                        req_en = data.get("enabled")
                        target = req_en if req_en is not None else not pipeline_instance.camera_enabled
                        new_state = pipeline_instance.set_camera_enabled(target)
                        await manager.broadcast_json({
                            "type": "camera_state",
                            "enabled": new_state,
                            "timestamp": int(time.time() * 1000)
                        })
                elif data.get("type") == "set_scene_mode":
                    mode = data.get("mode", "conveyor")
                    if pipeline_instance:
                        new_mode = pipeline_instance.set_scene_mode(mode)
                        await manager.broadcast_json({
                            "type": "scene_mode",
                            "mode": new_mode,
                            "timestamp": int(time.time() * 1000)
                        })
                elif data.get("type") == "prompt_exemplar":
                    box = data.get("box", [100.0, 100.0, 200.0, 200.0])
                    lbl = data.get("label", "apple")
                    if pipeline_instance:
                        rec = pipeline_instance.add_prompt_exemplar(box, lbl)
                        await manager.broadcast_json({
                            "type": "prompt_exemplar_added",
                            "exemplar": rec,
                            "timestamp": int(time.time() * 1000)
                        })
                elif data.get("type") == "command":
                    logger.info("Received client command: %s", data)
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning("WebSocket error: %s", e)
        manager.disconnect(websocket)
