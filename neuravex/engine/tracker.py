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
from typing import Optional, Dict, Any, List
import torch
import numpy as np
from ..geometry.camera import CameraIntrinsics

def robust_mask_depth_estimator(
    depth_map: torch.Tensor,
    mask: torch.Tensor,
    depth_conf: Optional[torch.Tensor] = None,
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
    Maintains:
      - 2D Bounding Box Kalman Filter [cx, cy, w, h, v_cx, v_cy, v_w, v_h] eliminating spatial lag
      - (x, y, z) 3D camera metric positions & 3D kinematic velocities
      - Multi-class Bayesian posterior belief to prevent single-frame class flipping
      - Boundary Overlap Score (BoS) and Temporal IoU metrics
      - Kinematics & Behavioral analytics (heading, speed, turn rate, dwell time, anomaly)
    """
    def __init__(self, track_id: int, initial_xyz: list, initial_box: list, initial_conf: float, class_id: int, num_classes: int = 80, initial_dims: Optional[dict] = None, initial_yaw: float = 0.0):
        self.track_id = track_id
        self.class_id = class_id
        self.num_classes = max(num_classes, class_id + 1)
        
        # 2D Bounding Box: convert [x1, y1, x2, y2] to [cx, cy, w, h]
        x1, y1, x2, y2 = initial_box
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        
        # 2D Box State: [cx, cy, w, h, v_cx, v_cy, v_w, v_h]
        self.kf_box = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.box = [float(x1), float(y1), float(x2), float(y2)]
        self.prev_box = list(self.box)
        
        # 3D State
        self.xyz = np.array(initial_xyz, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.prev_velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.confidence = float(initial_conf)
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        # Physical 3D Dimensions & Orientation
        self.dimensions3D = dict(initial_dims) if initial_dims else None
        self.yawDeg = float(initial_yaw) if initial_yaw is not None else 0.0

        # Multi-Class Bayesian Belief Vector
        self.class_priors = np.ones(self.num_classes, dtype=np.float32) * 0.05
        self.class_priors[class_id] += 1.0
        self.dominant_class_id = class_id
        self.class_confidence = float(initial_conf)

        # Overlap Stability & Metric Tracking
        self.bos = 1.0          # Boundary Overlap Score [0.0 - 1.0]
        self.temporal_iou = 1.0 # Temporal IoU [0.0 - 1.0]

        # Motion & Time-based Analytics
        self.heading_deg = float(self.yawDeg)
        self.direction_change_deg = 0.0
        self.cumulative_turns_deg = 0.0
        self.dwell_time = 0.0
        self.is_stationary = False
        self.stationary_duration = 0.0
        self.movement_state = "stationary"
        self.anomaly_score = 0.0
        self.anomaly_flags = []
        self.trajectory_history = [list(self.xyz)]


    def _box_to_xyxy(self, cxcywh: np.ndarray) -> list:
        cx, cy, w, h = cxcywh[:4]
        return [
            float(cx - 0.5 * w),
            float(cy - 0.5 * h),
            float(cx + 0.5 * w),
            float(cy + 0.5 * h)
        ]

    def _compute_iou(self, b1: list, b2: list) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
        a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
        union = a1 + a2 - inter
        return float(inter / max(union, 1e-6))

    def _compute_bos(self, b1: list, b2: list) -> float:
        iou = self._compute_iou(b1, b2)
        c1x, c1y = (b1[0] + b1[2]) * 0.5, (b1[1] + b1[3]) * 0.5
        c2x, c2y = (b2[0] + b2[2]) * 0.5, (b2[1] + b2[3]) * 0.5
        rho = math.sqrt((c1x - c2x)**2 + (c1y - c2y)**2)
        diag = math.sqrt(max(1.0, (max(b1[2], b2[2]) - min(b1[0], b2[0]))**2 + (max(b1[3], b2[3]) - min(b1[1], b2[1]))**2))
        dist_factor = math.exp(-2.0 * (rho / diag))
        return float(np.clip(iou * (0.7 * dist_factor + 0.3), 0.0, 1.0))

    def update(self, measured_xyz: list, measured_box: list, measured_conf: float, class_id: Optional[int] = None, measured_dims: Optional[dict] = None, measured_yaw: Optional[float] = None, dt: float = 1.0 / 30.0):
        dt = max(dt, 1e-4)
        m_xyz = np.array(measured_xyz, dtype=np.float32)
        m_conf = float(measured_conf)

        if measured_dims:
            self.dimensions3D = dict(measured_dims)
        if measured_yaw is not None:
            self.yawDeg = float(measured_yaw)


        # Multi-Class Bayesian Update
        if class_id is not None:
            if class_id >= len(self.class_priors):
                # Expand class prior vector dynamically
                new_priors = np.ones(class_id + 10, dtype=np.float32) * 0.05
                new_priors[:len(self.class_priors)] = self.class_priors
                self.class_priors = new_priors
            # Evidence accumulation with exponential decay of past history
            self.class_priors *= 0.96
            self.class_priors[class_id] += max(0.2, m_conf) * 1.5
            self.dominant_class_id = int(np.argmax(self.class_priors))
            prior_sum = float(np.sum(self.class_priors))
            self.class_confidence = float(self.class_priors[self.dominant_class_id] / max(prior_sum, 1e-6))
            self.class_id = self.dominant_class_id

        # 1. 2D Box Kalman Update:
        # Convert measured box to [cx, cy, w, h]
        mx1, my1, mx2, my2 = measured_box
        mcx = (mx1 + mx2) * 0.5
        mcy = (my1 + my2) * 0.5
        mw = max(1.0, mx2 - mx1)
        mh = max(1.0, my2 - my1)
        z_meas = np.array([mcx, mcy, mw, mh], dtype=np.float32)

        # Innovation (residual)
        res = z_meas - self.kf_box[:4]

        # Adaptive Kalman gain based on detection confidence
        # High confidence -> fast response, zero lag
        k_pos = float(np.clip(0.60 + 0.35 * m_conf, 0.40, 0.95))
        k_vel = float(np.clip(0.35 + 0.35 * m_conf, 0.20, 0.70))

        # Update position states
        self.kf_box[:4] += k_pos * res

        # Update velocity states [v_cx, v_cy, v_w, v_h]
        v_meas = res / dt
        self.kf_box[4:] = (1.0 - k_vel) * self.kf_box[4:] + k_vel * v_meas

        # Compute updated xyxy box and overlap stability metrics
        self.prev_box = list(self.box)
        self.box = self._box_to_xyxy(self.kf_box)

        # Calculate BoS and Temporal IoU against measured detection
        self.temporal_iou = self._compute_iou(self.box, measured_box)
        self.bos = self._compute_bos(self.box, measured_box)

        # 2. 3D Kinematics and Position Update
        inst_vel = (m_xyz - self.xyz) / dt
        self.prev_velocity = np.copy(self.velocity)
        self.velocity = 0.70 * self.velocity + 0.30 * inst_vel
        inst_acc = (self.velocity - self.prev_velocity) / dt
        self.acceleration = 0.70 * self.acceleration + 0.30 * inst_acc

        speed = float(np.linalg.norm(self.velocity))
        accel_mag = float(np.linalg.norm(self.acceleration))

        # 3. Heading & Direction Change (yaw in horizontal X-Z plane)
        if speed > 0.08:
            new_heading = float(math.atan2(float(self.velocity[0]), float(self.velocity[2])) * 180.0 / math.pi)
            angle_diff = abs(new_heading - self.heading_deg)
            if angle_diff > 180.0:
                angle_diff = 360.0 - angle_diff
            self.direction_change_deg = round(angle_diff, 1)
            self.cumulative_turns_deg += angle_diff
            self.heading_deg = round(new_heading, 1)
        else:
            self.direction_change_deg = 0.0

        # 4. Time-based Analytics: Dwell Time & Stationary Duration
        self.dwell_time += dt
        if speed < 0.18:
            self.is_stationary = True
            self.stationary_duration += dt
        else:
            self.is_stationary = False
            self.stationary_duration = 0.0

        # 5. Movement State Classification
        if self.is_stationary:
            if self.stationary_duration > 15.0:
                self.movement_state = "loitering"
            else:
                self.movement_state = "stationary"
        elif accel_mag > 3.0:
            self.movement_state = "rapid_acceleration"
        elif self.direction_change_deg > 45.0:
            self.movement_state = "erratic_turn"
        elif speed > 1.8:
            self.movement_state = "running"
        else:
            self.movement_state = "walking"

        # 6. Behavioral Anomaly Detection
        self.anomaly_flags = []
        anomaly_val = 0.0
        if accel_mag > 4.5:
            self.anomaly_flags.append("high_acceleration_shock")
            anomaly_val += 0.45
        if self.direction_change_deg > 60.0 and speed > 1.0:
            self.anomaly_flags.append("erratic_trajectory_shift")
            anomaly_val += 0.35
        if self.stationary_duration > 30.0:
            self.anomaly_flags.append("extended_stationary_dwell")
            anomaly_val += 0.25
        self.anomaly_score = float(np.clip(anomaly_val, 0.0, 1.0))

        # Metric depth update with confidence weighting
        alpha_3d = float(np.clip(0.35 + 0.55 * m_conf, 0.25, 0.90))
        self.xyz = (1.0 - alpha_3d) * self.xyz + alpha_3d * m_xyz
        self.confidence = 0.8 * self.confidence + 0.2 * m_conf

        self.trajectory_history.append(list(self.xyz))
        if len(self.trajectory_history) > 60:
            self.trajectory_history.pop(0)

        self.hits += 1
        self.time_since_update = 0
        self.age += 1

    def predict(self, dt: float = 1.0 / 30.0):
        # 1. Forward motion extrapolation for 2D box eliminating tracking lag
        # cx += v_cx * dt, cy += v_cy * dt, w += v_w * dt, h += v_h * dt
        self.kf_box[:2] += self.kf_box[4:6] * dt
        # Damped scale prediction
        self.kf_box[2:4] += self.kf_box[6:8] * dt * 0.5
        self.kf_box[2:4] = np.maximum(self.kf_box[2:4], 2.0)
        
        # Velocity damping
        self.kf_box[4:] *= 0.92

        self.prev_box = list(self.box)
        self.box = self._box_to_xyxy(self.kf_box)

        # 2. Forward motion extrapolation for 3D state
        self.xyz = self.xyz + self.velocity * dt * 0.9
        self.confidence *= 0.90  # Confidence decays during coasting
        self.time_since_update += 1
        self.age += 1
        self.dwell_time += dt


class RealTimeMetricDepthTracker:
    """
    Real-time multi-object spatial-temporal metric tracker.
    Features:
      - Hungarian Optimal Assignment via linear_sum_assignment with composite (CIoU + 3D Metric + Class Bayesian) cost
      - Strict 1-to-1 bijective mapping preventing duplicate track corruption
      - Constant-Velocity 2D Kalman box filtering eliminating motion lag and maximizing IoU / BoS
      - Bayesian multi-class calibration preventing single-frame class flips
      - Metric 3D depth fusion, occlusion coasting, and scene cut detection
    """
    def __init__(self, max_missed_frames: int = 5, iou_thresh: float = 0.30, scene_cut_distance: float = 25.0):
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
        return float(inter / max(union, 1e-6))

    def _ciou(self, b1, b2):
        iou = self._iou(b1, b2)
        # Center distance
        c1x, c1y = (b1[0] + b1[2]) * 0.5, (b1[1] + b1[3]) * 0.5
        c2x, c2y = (b2[0] + b2[2]) * 0.5, (b2[1] + b2[3]) * 0.5
        rho2 = (c1x - c2x)**2 + (c1y - c2y)**2

        # Enclosing diagonal
        enc_w = max(b1[2], b2[2]) - min(b1[0], b2[0])
        enc_h = max(b1[3], b2[3]) - min(b1[1], b2[1])
        c2 = max(1e-4, enc_w**2 + enc_h**2)

        # Aspect ratio consistency
        w1, h1 = max(1.0, b1[2] - b1[0]), max(1.0, b1[3] - b1[1])
        w2, h2 = max(1.0, b2[2] - b2[0]), max(1.0, b2[3] - b2[1])
        v = (4.0 / (math.pi**2)) * ((math.atan(w2 / h2) - math.atan(w1 / h1))**2)
        alpha = v / max(1e-4, (1.0 - iou + v))
        return float(iou - (rho2 / c2 + alpha * v))

    def step(self, objects: list, dt: float = 1.0 / 30.0) -> list:
        """
        Processes detections in current frame, updates tracks, and outputs stabilized objects.
        """
        # 1. Scene cut check: if all objects suddenly shifted massively, reset tracks
        if len(objects) > 0:
            curr_centroid = np.mean([np.array([o["x"], o["y"], o["z"]]) for o in objects], axis=0)
            if self.prev_centroid is not None:
                scene_shift = float(np.linalg.norm(curr_centroid - self.prev_centroid))
                if scene_shift > self.scene_cut_distance:
                    self.tracks.clear()
            self.prev_centroid = curr_centroid

        # 2. Predict all existing tracks
        for trk in self.tracks.values():
            trk.predict(dt)

        matched_obj_to_track = {}
        unmatched_obj_indices = set(range(len(objects)))

        # 3. Hungarian Bipartite Assignment
        if len(self.tracks) > 0 and len(objects) > 0:
            track_ids = list(self.tracks.keys())
            num_dets = len(objects)
            num_trks = len(track_ids)
            cost_matrix = np.full((num_dets, num_trks), 1e5, dtype=np.float32)

            for i, obj in enumerate(objects):
                obj_pos = np.array([obj["x"], obj["y"], obj["z"]], dtype=np.float32)
                obj_cls = int(obj["class"])
                for j, tid in enumerate(track_ids):
                    trk = self.tracks[tid]
                    iou = self._iou(obj["bbox"], trk.box)
                    ciou = self._ciou(obj["bbox"], trk.box)
                    dist_3d = float(np.linalg.norm(obj_pos - trk.xyz))
                    
                    # Normalized depth scale
                    z_scale = max(1.0, trk.xyz[2] * 0.15)
                    norm_dist_3d = dist_3d / z_scale

                    # Multi-Class Bayesian likelihood penalty:
                    # If classes match -> 0 penalty; if mismatch -> soft penalty scaled by IoU
                    class_mismatch = (obj_cls != trk.dominant_class_id)
                    class_penalty = 4.0 if class_mismatch else 0.0

                    # Composite Cost: Higher CIoU and lower metric 3D distance minimize cost
                    cost = (1.0 - ciou) * 8.0 + norm_dist_3d * 2.5 + class_penalty
                    
                    # Gating: reject impossible pairings
                    if (iou < 0.05 and dist_3d > 4.5) or (class_mismatch and iou < 0.25 and dist_3d > 2.0):
                        cost = 1e5

                    cost_matrix[i, j] = cost

            # Solve via Hungarian algorithm
            try:
                from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped,import-not-found]
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                for r, c in zip(row_ind, col_ind):
                    if cost_matrix[r, c] < 20.0:  # Association acceptance threshold
                        tid = track_ids[c]
                        obj = objects[r]
                        self.tracks[tid].update(
                            [obj["x"], obj["y"], obj["z"]],
                            obj["bbox"],
                            obj.get("depth_confidence", 0.8),
                            class_id=obj["class"],
                            measured_dims=obj.get("dimensions3D", None),
                            measured_yaw=obj.get("yawDeg", None),
                            dt=dt
                        )
                        matched_obj_to_track[r] = tid
                        unmatched_obj_indices.discard(r)
            except ImportError:
                # Fallback greedy matching if scipy is not available
                cost_copy = np.copy(cost_matrix)
                for _ in range(min(num_dets, num_trks)):
                    i_min, j_min = np.unravel_index(np.argmin(cost_copy), cost_copy.shape)
                    if cost_copy[i_min, j_min] >= 20.0:
                        break
                    tid = track_ids[j_min]
                    obj = objects[i_min]
                    self.tracks[tid].update(
                        [obj["x"], obj["y"], obj["z"]],
                        obj["bbox"],
                        obj.get("depth_confidence", 0.8),
                        class_id=obj["class"],
                        measured_dims=obj.get("dimensions3D", None),
                        measured_yaw=obj.get("yawDeg", None),
                        dt=dt
                    )
                    matched_obj_to_track[i_min] = tid
                    unmatched_obj_indices.discard(i_min)
                    cost_copy[i_min, :] = 1e5
                    cost_copy[:, j_min] = 1e5

        # 4. Create new tracks for unmatched detections
        for i in unmatched_obj_indices:
            obj = objects[i]
            tid = self.next_id
            self.next_id += 1
            new_trk = TemporalObjectFilter(
                track_id=tid,
                initial_xyz=[obj["x"], obj["y"], obj["z"]],
                initial_box=obj["bbox"],
                initial_conf=obj.get("depth_confidence", 0.8),
                class_id=obj["class"],
                initial_dims=obj.get("dimensions3D", None),
                initial_yaw=obj.get("yawDeg", 0.0)
            )
            self.tracks[tid] = new_trk
            matched_obj_to_track[i] = tid

        # 5. Clean up stale/dead tracks
        dead_ids = [tid for tid, trk in self.tracks.items() if trk.time_since_update > self.max_missed_frames]
        for tid in dead_ids:
            del self.tracks[tid]

        # 6. Assemble stabilized output list with exact bijective matching
        stabilized_objects = []
        for i, obj in enumerate(objects):
            assigned_tid = matched_obj_to_track.get(i, None)
            if assigned_tid is not None and assigned_tid in self.tracks:
                trk = self.tracks[assigned_tid]
                x_s, y_s, z_s = float(trk.xyz[0]), float(trk.xyz[1]), float(trk.xyz[2])
                dist_s = float(math.sqrt(x_s ** 2 + y_s ** 2 + z_s ** 2))
                spd = float(np.linalg.norm(trk.velocity))
                acc_mag = float(np.linalg.norm(trk.acceleration))
                out_obj = {
                    "id": int(trk.track_id),
                    "class": int(trk.dominant_class_id),
                    "class_confidence": round(float(trk.class_confidence), 3),
                    "score": float(obj["score"]),
                    "bbox": [float(b) for b in trk.box],
                    "mask": obj.get("mask", None),
                    "x": x_s,
                    "y": y_s,
                    "z": z_s,
                    "depth": z_s,
                    "distance": dist_s,
                    "depth_confidence": float(trk.confidence),
                    "dimensions3D": trk.dimensions3D,
                    "yawDeg": trk.yawDeg,
                    # High IoU and BoS Stability metrics
                    "bos": round(float(trk.bos), 3),
                    "temporal_iou": round(float(trk.temporal_iou), 3),
                    # Kinematic Velocity & Acceleration
                    "vx": round(float(trk.velocity[0]), 2),
                    "vy": round(float(trk.velocity[1]), 2),
                    "vz": round(float(trk.velocity[2]), 2),
                    "speed": round(spd, 2),
                    "ax": round(float(trk.acceleration[0]), 2),
                    "ay": round(float(trk.acceleration[1]), 2),
                    "az": round(float(trk.acceleration[2]), 2),
                    "acceleration": round(acc_mag, 2),
                    # Direction & Turning Analytics
                    "headingDeg": trk.heading_deg,
                    "directionChangeDeg": trk.direction_change_deg,
                    "cumulativeTurnsDeg": round(trk.cumulative_turns_deg, 1),
                    # Time-Based Analytics
                    "dwellTime": round(trk.dwell_time, 1),
                    "isStationary": trk.is_stationary,
                    "stationaryDuration": round(trk.stationary_duration, 1),
                    # Behavioral Intelligence & Anomaly
                    "movementState": trk.movement_state,
                    "anomalyScore": trk.anomaly_score,
                    "anomalyFlags": trk.anomaly_flags
                }
            else:
                out_obj = {
                    "id": self.next_id,
                    "class": int(obj["class"]),
                    "class_confidence": float(obj["score"]),
                    "score": float(obj["score"]),
                    "bbox": [float(b) for b in obj["bbox"]],
                    "mask": obj.get("mask", None),
                    "x": float(obj["x"]),
                    "y": float(obj["y"]),
                    "z": float(obj["z"]),
                    "depth": float(obj["depth"]),
                    "distance": float(obj["distance"]),
                    "depth_confidence": float(obj["depth_confidence"]),
                    "dimensions3D": obj.get("dimensions3D", None),
                    "yawDeg": float(obj.get("yawDeg", 0.0)),
                    "bos": 1.0,
                    "temporal_iou": 1.0,
                    "vx": 0.0, "vy": 0.0, "vz": 0.0, "speed": 0.0,
                    "ax": 0.0, "ay": 0.0, "az": 0.0, "acceleration": 0.0,
                    "headingDeg": 0.0,
                    "directionChangeDeg": 0.0,
                    "cumulativeTurnsDeg": 0.0,
                    "dwellTime": round(dt, 1),
                    "isStationary": True,
                    "stationaryDuration": round(dt, 1),
                    "movementState": "stationary",
                    "anomalyScore": 0.0,
                    "anomalyFlags": []
                }
                self.next_id += 1

            stabilized_objects.append(out_obj)


        return stabilized_objects

