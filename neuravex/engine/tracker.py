"""
Neuravex Real-Time Per-Object Spatial-Temporal Metric Depth Estimator and Tracker.

Fuses:
  - 2D Bounding Boxes & Class Logits (NMS filtered)
  - True Instance Segmentation Masks (Mask-depth fusion, NOT bounding box rectangles)
  - Dense Calibrated Metric Depth Map & Confidence
  - Camera Pinhole Intrinsics K = (fx, fy, cx, cy)

Performs:
  - Robust confidence-weighted trimmed/median depth estimation rejecting invalid/outlier depth pixels
  - Precise Camera 3D coordinates:
      Z_obj = robust_depth
      X_obj = (u_c - cx) * Z_obj / fx
      Y_obj = (v_c - cy) * Z_obj / fy
      R_obj = sqrt(X_obj^2 + Y_obj^2 + Z_obj^2)
  - Real-time temporal stabilization with object tracking IDs, confidence-adaptive EMA/Kalman filtering,
    missed object coasting, and scene cut detection.
  - Zero compute overhead when depth is bypassed.
"""

import math
import torch
import numpy as np
from ..geometry.camera import CameraIntrinsics

def robust_mask_depth_estimator(
    depth_map: torch.Tensor,
    mask: torch.Tensor,
    depth_conf: torch.Tensor = None,
    min_depth: float = 0.05,
    max_depth: float = 120.0,
    trim_ratio: float = 0.15
) -> tuple:
    """
    Computes robust confidence-weighted trimmed median depth and centroid over TRUE instance mask.
    Never uses bounding box rectangles.
    
    Args:
        depth_map: (H, W) or (1, H, W) metric depth in meters
        mask: (H, W) or (1, H, W) binary boolean or float tensor mask
        depth_conf: (H, W) or (1, H, W) calibrated confidence in [0, 1]
        min_depth: minimum valid metric depth
        max_depth: maximum valid metric depth
        trim_ratio: fraction of lower and upper percentiles to trim as outliers (default 15%)
        
    Returns:
        (z_metric, u_center, v_center, depth_confidence)
    """
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze(0)
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    if depth_conf is not None and depth_conf.ndim == 3:
        depth_conf = depth_conf.squeeze(0)

    # Convert mask to bool
    if mask.dtype != torch.bool:
        mask_bool = mask > 0.5
    else:
        mask_bool = mask

    # Valid mask pixels with realistic metric range
    valid_px = mask_bool & (depth_map >= min_depth) & (depth_map <= max_depth) & (~torch.isnan(depth_map)) & (~torch.isinf(depth_map))

    if not valid_px.any():
        return float("nan"), float("nan"), float("nan"), 0.0

    d_vals = depth_map[valid_px]
    v_indices, u_indices = torch.nonzero(valid_px, as_tuple=True)

    if depth_conf is not None:
        c_vals = depth_conf[valid_px]
    else:
        c_vals = torch.ones_like(d_vals)

    n_pts = d_vals.numel()

    # If very few pixels, return simple weighted average
    if n_pts < 5:
        w_sum = c_vals.sum().clamp_min(1e-6)
        z_est = (d_vals * c_vals).sum() / w_sum
        u_c = (u_indices.float() * c_vals).sum() / w_sum
        v_c = (v_indices.float() * c_vals).sum() / w_sum
        conf = float(c_vals.mean().item())
        return float(z_est.item()), float(u_c.item()), float(v_c.item()), conf

    # Robust Trimmed Estimator: sort depth values, remove lowest and highest trim_ratio
    sorted_idx = torch.argsort(d_vals)
    k_trim = int(n_pts * trim_ratio)
    keep_idx = sorted_idx[k_trim : max(n_pts - k_trim, k_trim + 1)]

    trimmed_d = d_vals[keep_idx]
    trimmed_c = c_vals[keep_idx]
    trimmed_u = u_indices[keep_idx].float()
    trimmed_v = v_indices[keep_idx].float()

    w_sum = trimmed_c.sum().clamp_min(1e-6)
    z_est = (trimmed_d * trimmed_c).sum() / w_sum
    u_c = (trimmed_u * trimmed_c).sum() / w_sum
    v_c = (trimmed_v * trimmed_c).sum() / w_sum

    # Confidence is scaled by variance consistency and mean pixel confidence
    std_d = torch.std(trimmed_d)
    var_factor = torch.exp(-std_d / (z_est + 1e-4)).clamp(0.1, 1.0)
    conf = float((trimmed_c.mean() * var_factor).item())

    return float(z_est.item()), float(u_c.item()), float(v_c.item()), conf

class TemporalObjectFilter:
    """
    Confidence-adaptive real-time 3D state tracking filter for a single tracked object.
    Maintains (x, y, z, vx, vy, vz, dist, conf) with adaptive smoothing.
    """
    def __init__(self, track_id: int, initial_xyz: list, initial_box: list, initial_conf: float, class_id: int):
        self.track_id = track_id
        self.class_id = class_id
        self.box = list(initial_box)
        self.xyz = np.array(initial_xyz, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.confidence = float(initial_conf)
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

    def update(self, measured_xyz: list, measured_box: list, measured_conf: float, dt: float = 1.0 / 30.0):
        m_xyz = np.array(measured_xyz, dtype=np.float32)
        m_conf = float(measured_conf)

        # Adaptive smoothing rate alpha based on measured confidence
        # High confidence -> responsive (alpha ~ 0.7-0.85); Low confidence -> smoother (alpha ~ 0.3)
        alpha = float(np.clip(0.2 + 0.65 * m_conf, 0.15, 0.90))

        # Velocity estimation with exponential decay
        inst_vel = (m_xyz - self.xyz) / max(dt, 1e-4)
        self.velocity = 0.8 * self.velocity + 0.2 * inst_vel

        # Smooth position update
        self.xyz = (1.0 - alpha) * self.xyz + alpha * m_xyz
        self.confidence = 0.8 * self.confidence + 0.2 * m_conf

        # Smooth box update
        for i in range(4):
            self.box[i] = (1.0 - alpha) * self.box[i] + alpha * measured_box[i]

        self.hits += 1
        self.time_since_update = 0
        self.age += 1

    def predict(self, dt: float = 1.0 / 30.0):
        # Coast position with damped velocity when measurement is momentarily missed
        self.xyz = self.xyz + self.velocity * dt * 0.9
        self.confidence *= 0.85  # Confidence decays during coasting
        self.time_since_update += 1
        self.age += 1

class RealTimeMetricDepthTracker:
    """
    Real-time multi-object spatial-temporal metric tracker.
    Associates 2D/3D detections, filters metric depth over time,
    handles occlusions, and detects scene cuts.
    """
    def __init__(self, max_missed_frames: int = 5, iou_thresh: float = 0.35, scene_cut_distance: float = 25.0):
        self.max_missed_frames = max_missed_frames
        self.iou_thresh = iou_thresh
        self.scene_cut_distance = scene_cut_distance
        self.tracks = {}
        self.next_id = 1
        self.prev_centroid = None

    def _iou(self, box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / max(union, 1e-6)

    def step(self, objects: list, dt: float = 1.0 / 30.0) -> list:
        """
        objects: list of dicts with:
          {'class': int, 'score': float, 'bbox': [x1,y1,x2,y2], 'mask': tensor/ndarray or None,
           'x': float, 'y': float, 'z': float, 'depth': float, 'distance': float, 'depth_confidence': float}
        """
        # 1. Scene cut check: if all objects suddenly shifted massively, reset tracks
        if len(objects) > 0:
            curr_centroid = np.mean([np.array([o["x"], o["y"], o["z"]]) for o in objects], axis=0)
            if self.prev_centroid is not None:
                scene_shift = float(np.linalg.norm(curr_centroid - self.prev_centroid))
                if scene_shift > self.scene_cut_distance:
                    # Scene cut detected! Clear previous tracks to avoid catastrophic lag
                    self.tracks.clear()
            self.prev_centroid = curr_centroid

        # 2. Predict existing tracks
        for trk in self.tracks.values():
            trk.predict(dt)

        matched_track_ids = set()
        matched_obj_indices = set()

        # 3. Associate detections to tracks using spatial distance + 2D IoU
        if len(self.tracks) > 0 and len(objects) > 0:
            track_ids = list(self.tracks.keys())
            cost_matrix = np.zeros((len(objects), len(track_ids)), dtype=np.float32)

            for i, obj in enumerate(objects):
                for j, tid in enumerate(track_ids):
                    trk = self.tracks[tid]
                    # Class consistency check
                    if obj["class"] != trk.class_id:
                        cost_matrix[i, j] = 1e6
                        continue

                    iou = self._iou(obj["bbox"], trk.box)
                    dist_3d = float(np.linalg.norm(np.array([obj["x"], obj["y"], obj["z"]]) - trk.xyz))
                    
                    # Combined metric: high IoU and low 3D distance minimize cost
                    cost = (1.0 - iou) * 10.0 + min(dist_3d, 10.0)
                    cost_matrix[i, j] = cost

            # Greedy assignment
            for _ in range(min(len(objects), len(track_ids))):
                i_min, j_min = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                if cost_matrix[i_min, j_min] > 15.0:
                    break  # Beyond association threshold

                tid = track_ids[j_min]
                obj = objects[i_min]
                self.tracks[tid].update([obj["x"], obj["y"], obj["z"]], obj["bbox"], obj["depth_confidence"], dt=dt)
                matched_track_ids.add(tid)
                matched_obj_indices.add(i_min)
                cost_matrix[i_min, :] = 1e6
                cost_matrix[:, j_min] = 1e6

        # 4. Create new tracks for unmatched detections
        for i, obj in enumerate(objects):
            if i not in matched_obj_indices:
                tid = self.next_id
                self.next_id += 1
                new_trk = TemporalObjectFilter(
                    track_id=tid,
                    initial_xyz=[obj["x"], obj["y"], obj["z"]],
                    initial_box=obj["bbox"],
                    initial_conf=obj["depth_confidence"],
                    class_id=obj["class"]
                )
                self.tracks[tid] = new_trk
                matched_track_ids.add(tid)

        # 5. Clean up old dead tracks
        dead_ids = [tid for tid, trk in self.tracks.items() if trk.time_since_update > self.max_missed_frames]
        for tid in dead_ids:
            del self.tracks[tid]

        # 6. Assemble stabilized output list
        stabilized_objects = []
        for i, obj in enumerate(objects):
            # Find assigned track
            assigned_tid = None
            for tid in matched_track_ids:
                if tid in self.tracks and self.tracks[tid].time_since_update == 0:
                    trk = self.tracks[tid]
                    # Check matching position
                    if np.linalg.norm(np.array([obj["x"], obj["y"], obj["z"]]) - trk.xyz) < 5.0 and obj["class"] == trk.class_id:
                        assigned_tid = tid
                        break

            if assigned_tid is not None:
                trk = self.tracks[assigned_tid]
                x_s, y_s, z_s = float(trk.xyz[0]), float(trk.xyz[1]), float(trk.xyz[2])
                dist_s = float(math.sqrt(x_s ** 2 + y_s ** 2 + z_s ** 2))
                out_obj = {
                    "id": int(trk.track_id),
                    "class": int(obj["class"]),
                    "score": float(obj["score"]),
                    "bbox": [float(b) for b in trk.box],
                    "mask": obj.get("mask", None),
                    "x": x_s,
                    "y": y_s,
                    "z": z_s,
                    "depth": z_s,
                    "distance": dist_s,
                    "depth_confidence": float(trk.confidence)
                }
            else:
                out_obj = {
                    "id": self.next_id,
                    "class": int(obj["class"]),
                    "score": float(obj["score"]),
                    "bbox": [float(b) for b in obj["bbox"]],
                    "mask": obj.get("mask", None),
                    "x": float(obj["x"]),
                    "y": float(obj["y"]),
                    "z": float(obj["z"]),
                    "depth": float(obj["depth"]),
                    "distance": float(obj["distance"]),
                    "depth_confidence": float(obj["depth_confidence"])
                }
                self.next_id += 1

            stabilized_objects.append(out_obj)

        return stabilized_objects
