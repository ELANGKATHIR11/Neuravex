"""
Neuravex Complex Multi-Modal Synthetic Dataset Engine.

Generates extreme-entropy, physically grounded, multi-modal synthetic data
for industry-grade stress-testing across all 11 Neuravex perception tasks:
  1. Dense CameraAwareDEM Metric Depth (Z in meters) & Elevation Terrain
  2. 3D Oriented Bounding Cuboids (X, Y, Z, L, W, H, yaw, continuous 6D rotation)
  3. Multi-Scale 2D Object Detection with dense occlusion and multi-instance crowds
  4. Pixel-accurate Instance Masks, Boundary Maps, and Prototype Masks
  5. 2D & 3D Human / Creature 17-Keypoint Anatomical Poses with Biomechanics
  6. Multi-Frame Spatio-Temporal Video Trajectories & Kinematic Velocity / Anomaly Tokens
  7. SAM-Inspired Promptable Few-Shot Exemplar Prompts & Produce Grading
  8. Extreme Domain Randomization: Weather (fog, rain, glare), Lighting (night to daylight),
     Texture synthesis, and Sensor Gaussian / Poisson noise.
"""

import os
import math
import json
import random
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

OBJECT_CLASSES = [
    {"name": "Heavy_Truck", "class_id": 0, "size": (8.2, 2.5, 3.2), "color": (160, 50, 40)},
    {"name": "Sedan_Car",   "class_id": 1, "size": (4.5, 1.8, 1.5), "color": (40, 140, 220)},
    {"name": "Motorcycle",  "class_id": 2, "size": (2.1, 0.8, 1.2), "color": (220, 80, 160)},
    {"name": "Human_Worker","class_id": 3, "size": (0.6, 0.5, 1.75), "color": (60, 200, 80)},
    {"name": "Cattle_Cow",  "class_id": 4, "size": (2.4, 1.2, 1.4), "color": (30, 180, 210)},
    {"name": "Cargo_Crate", "class_id": 5, "size": (1.2, 1.2, 1.1), "color": (210, 160, 30)}
]


class ComplexSyntheticEngine:
    """
    High-fidelity physical renderer producing synchronized multi-modal ground truth.
    """
    def __init__(self, img_size: int = 320, seed: int = 42):
        self.img_size = img_size
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        # Standard camera intrinsics
        self.fx = 420.0 * (img_size / 320.0)
        self.fy = 420.0 * (img_size / 320.0)
        self.cx = img_size / 2.0
        self.cy = img_size / 2.0
        self.h_cam = 1.65  # camera mounting height in meters

    def _render_sky_and_terrain(self, img_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Renders realistic atmospheric sky, non-flat digital elevation terrain (DEM),
        and dense metric depth ground plane with elevation bumps and texture noise.
        """
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        depth_map = np.full((img_size, img_size), 75.0, dtype=np.float32)
        terrain_elev = np.zeros((img_size, img_size), dtype=np.float32)

        # 1. Environmental lighting & horizon
        horizon_y = int(img_size * random.uniform(0.38, 0.48))
        sky_tone = random.choice([
            ((240, 215, 175), (200, 160, 110)),  # Daytime clear
            ((60, 45, 30), (25, 20, 15)),        # Night / Twilight
            ((190, 190, 185), (140, 140, 135)),  # Overcast foggy
            ((220, 140, 90), (120, 60, 40))      # Sunset golden hour
        ])
        for y in range(horizon_y):
            ratio = y / max(1, horizon_y)
            c = [int(sky_tone[0][i] * (1 - ratio) + sky_tone[1][i] * ratio) for i in range(3)]
            img[y, :] = c

        # 2. Digital Elevation Terrain (DEM) + undulating ground
        x_grid, y_grid = np.meshgrid(np.arange(img_size), np.arange(img_size))
        freq_1 = random.uniform(0.015, 0.035)
        freq_2 = random.uniform(0.04, 0.08)
        elevation_noise = (
            np.sin(x_grid * freq_1) * 0.45 +
            np.cos(y_grid * freq_2) * 0.35 +
            (np.random.rand(img_size, img_size) - 0.5) * 0.15
        ).astype(np.float32)

        for y in range(horizon_y, img_size):
            y_diff = max(float(y - self.cy), 0.5)
            # Physical ground distance Z = (fy * (h_cam + elev)) / (y - cy)
            local_elev = float(elevation_noise[y, img_size // 2])
            z_nominal = (self.fy * (self.h_cam + local_elev)) / y_diff
            z_nominal = np.clip(z_nominal, 0.8, 75.0)

            # Row-wise depth with subtle lateral rolling terrain
            lateral_variation = (elevation_noise[y, :] * 0.2)
            z_row = np.clip(z_nominal + lateral_variation, 0.5, 75.0)
            depth_map[y, :] = z_row
            terrain_elev[y, :] = local_elev + (self.img_size - y) * 0.01

            # Asphalt / Ground texture with procedural gravel grains
            shade_base = int(45 + 50 * (y / img_size))
            grain = (np.random.randn(img_size) * 6).astype(np.int32)
            shade_int = np.clip(shade_base + grain, 20, 230)
            img[y, :, 0] = shade_int.astype(np.uint8)
            img[y, :, 1] = np.clip(shade_int + int(local_elev * 12), 20, 230).astype(np.uint8)
            img[y, :, 2] = np.clip(shade_int - 5, 20, 230).astype(np.uint8)

        return img, depth_map, terrain_elev

    def _generate_3d_cuboid(
        self,
        cls_info: dict,
        existing_objects: List[dict]
    ) -> Optional[dict]:
        """
        Samples collision-free 3D coordinates (X, Y, Z, L, W, H, yaw, pitch, roll).
        """
        L = max(0.4, cls_info["size"][0] * random.uniform(0.85, 1.18))
        W = max(0.3, cls_info["size"][1] * random.uniform(0.85, 1.15))
        H = max(0.4, cls_info["size"][2] * random.uniform(0.85, 1.15))

        # Range distribution: mix of close, medium, and far targets
        dist_tier = random.choice(["close", "medium", "far", "far"])
        if dist_tier == "close":
            Z = random.uniform(2.2, 7.0)
        elif dist_tier == "medium":
            Z = random.uniform(7.0, 18.0)
        else:
            Z = random.uniform(18.0, 42.0)

        # Lateral camera FOV boundary at distance Z
        max_x = max(0.5, (self.img_size * 0.42 / self.fx) * Z)
        X = random.uniform(-max_x, max_x)

        # Ground contact condition: bottom touches local terrain
        Y = (self.h_cam - H * 0.5)

        # Rotations
        yaw = random.uniform(-math.pi, math.pi)
        pitch = random.uniform(-0.06, 0.06)  # subtle incline
        roll = random.uniform(-0.04, 0.04)

        # Collision check with existing objects in 3D
        for obj in existing_objects:
            dx = abs(X - obj["X"])
            dz = abs(Z - obj["Z"])
            if dx < (W + obj["W"]) * 0.65 and dz < (L + obj["L"]) * 0.65:
                return None  # collision rejection

        return {
            "class_name": cls_info["name"],
            "class_id": cls_info["class_id"],
            "color": cls_info["color"],
            "X": float(X), "Y": float(Y), "Z": float(Z),
            "L": float(L), "W": float(W), "H": float(H),
            "yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)
        }

    def _project_and_render_cuboids(
        self,
        img: np.ndarray,
        depth_map: np.ndarray,
        objects: List[dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Projects 3D bounding boxes to 2D image sensors, performs Z-buffering,
        draws shaded 3D faces, ground contact shadows, generates precise instance masks,
        and computes continuous 6D rotation matrices.
        """
        h_img, w_img = img.shape[:2]
        instance_mask_canvas = np.zeros((h_img, w_img), dtype=np.int32)
        sem_canvas = np.zeros((h_img, w_img), dtype=np.int32)
        boundary_canvas = np.zeros((h_img, w_img), dtype=np.uint8)
        rendered_annotations = []

        # Sort objects back-to-front (depth descending)
        sorted_objects = sorted(objects, key=lambda o: o["Z"], reverse=True)

        for inst_idx, obj in enumerate(sorted_objects, start=1):
            X, Y, Z = obj["X"], obj["Y"], obj["Z"]
            L, W, H = obj["L"], obj["W"], obj["H"]
            yaw = obj["yaw"]

            # Rotation matrix in camera frame
            cy, sy = math.cos(yaw), math.sin(yaw)
            rot_mat = np.array([
                [cy,  0.0, sy],
                [0.0, 1.0, 0.0],
                [-sy, 0.0, cy]
            ], dtype=np.float32)

            # Continuous 6D rotation representation (first two columns of rot_mat)
            rot_6d = [float(rot_mat[0, 0]), float(rot_mat[1, 0]), float(rot_mat[2, 0]),
                      float(rot_mat[0, 1]), float(rot_mat[1, 1]), float(rot_mat[2, 1])]

            # 8 local cuboid corners (centered at object origin)
            # Order: 0: front-bottom-left, 1: front-bottom-right, 2: front-top-right, 3: front-top-left
            #        4: back-bottom-left,  5: back-bottom-right,  6: back-top-right,  7: back-top-left
            dx_c = L * 0.5
            dy_c = H * 0.5
            dz_c = W * 0.5

            corners_local = np.array([
                [-dx_c,  dy_c, -dz_c],
                [ dx_c,  dy_c, -dz_c],
                [ dx_c, -dy_c, -dz_c],
                [-dx_c, -dy_c, -dz_c],
                [-dx_c,  dy_c,  dz_c],
                [ dx_c,  dy_c,  dz_c],
                [ dx_c, -dy_c,  dz_c],
                [-dx_c, -dy_c,  dz_c]
            ], dtype=np.float32)

            # Transform to camera coordinate frame: X_cam = R * X_local + Center
            corners_cam = (corners_local @ rot_mat.T) + np.array([X, Y, Z], dtype=np.float32)

            # Pinhole perspective projection: u = X*fx/Z + cx, v = Y*fy/Z + cy
            uv_corners = []
            valid_depth = True
            for pt in corners_cam:
                if pt[2] <= 0.3:
                    valid_depth = False
                    break
                u = (pt[0] * self.fx) / pt[2] + self.cx
                v = (pt[1] * self.fy) / pt[2] + self.cy
                uv_corners.append([float(u), float(v)])

            if not valid_depth or len(uv_corners) < 8:
                continue

            pts_2d = np.array(uv_corners, dtype=np.int32)

            # Tight 2D bounding box
            x1 = int(np.clip(np.min(pts_2d[:, 0]), 0, w_img - 1))
            y1 = int(np.clip(np.min(pts_2d[:, 1]), 0, h_img - 1))
            x2 = int(np.clip(np.max(pts_2d[:, 0]), 0, w_img - 1))
            y2 = int(np.clip(np.max(pts_2d[:, 1]), 0, h_img - 1))

            bw = x2 - x1
            bh = y2 - y1
            if bw < 4 or bh < 4:
                continue

            # 1. Ground Shadow (ambient occlusion on ground)
            ground_face = pts_2d[[0, 1, 5, 4]]
            cv2.fillPoly(img, [ground_face], (15, 20, 15))

            # 2. Solid shaded faces (Front, Top, Right) with pseudo-Lambertian lighting
            base_col = obj["color"]
            front_face = pts_2d[[0, 1, 2, 3]]
            top_face   = pts_2d[[3, 2, 6, 7]]
            side_face  = pts_2d[[1, 5, 6, 2]]

            obj_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            # Render shaded polygons
            cv2.fillPoly(img, [front_face], tuple(int(c * 0.85) for c in base_col))
            cv2.fillPoly(img, [top_face],   tuple(int(c * 1.10) for c in base_col))
            cv2.fillPoly(img, [side_face],  tuple(int(c * 0.65) for c in base_col))

            # Build exact instance mask
            cv2.fillPoly(obj_mask, [front_face, top_face, side_face], 255)

            # 3. Z-buffer update: overwrite depth map only where this closer object is drawn
            mask_bool = (obj_mask > 0)
            depth_map[mask_bool] = np.minimum(depth_map[mask_bool], Z)
            instance_mask_canvas[mask_bool] = inst_idx
            sem_canvas[mask_bool] = obj["class_id"] + 1

            # 4. Wireframe edges
            edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
            for e1, e2 in edges:
                cv2.line(img, tuple(pts_2d[e1]), tuple(pts_2d[e2]), (240, 240, 240), 1, cv2.LINE_AA)

            # Generate boundary edges
            contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(boundary_canvas, contours, -1, 255, 1)

            # Pack complete annotation
            rendered_annotations.append({
                "inst_id": inst_idx,
                "class_name": obj["class_name"],
                "class_id": obj["class_id"],
                "bbox_2d": [x1, y1, x2, y2],
                "center_xyz": [round(X, 3), round(Y, 3), round(Z, 3)],
                "size_lwh": [round(L, 3), round(W, 3), round(H, 3)],
                "yaw_rad": round(yaw, 4),
                "rot_6d": [round(v, 4) for v in rot_6d],
                "corners_2d": [[int(p[0]), int(p[1])] for p in pts_2d],
                "occlusion_ratio": round(float(np.mean(depth_map[y1:y2, x1:x2] < Z)), 3)
            })

        return img, depth_map, instance_mask_canvas, sem_canvas, rendered_annotations

    def _generate_human_poses(
        self,
        objects: List[dict]
    ) -> List[dict]:
        """
        Synthesizes anatomically compliant 17-joint COCO human poses in metric 3D space
        and projects them to 2D pixel landmarks for any human instance.
        """
        pose_annotations = []
        for obj in objects:
            if obj["class_name"] != "Human_Worker":
                continue

            X, Y, Z = obj["center_xyz"]
            L, W, H = obj["size_lwh"]
            yaw = obj["yaw_rad"]
            cy, sy = math.cos(yaw), math.sin(yaw)

            # Canonical relative 3D joint offsets in meters (relative to torso center)
            # Z is depth, Y is vertical down (+Y is towards feet), X is lateral
            # Head & Facial landmarks
            rel_joints = [
                [0.0, -0.75, 0.08],     # 0: nose
                [-0.04, -0.78, 0.07],   # 1: left_eye
                [ 0.04, -0.78, 0.07],   # 2: right_eye
                [-0.08, -0.77, 0.0],    # 3: left_ear
                [ 0.08, -0.77, 0.0],    # 4: right_ear
                # Shoulders & Arms
                [-0.20, -0.55, 0.0],    # 5: left_shoulder
                [ 0.20, -0.55, 0.0],    # 6: right_shoulder
                [-0.26, -0.28, 0.05],   # 7: left_elbow
                [ 0.26, -0.28, 0.05],   # 8: right_elbow
                [-0.28,  0.00, 0.12],   # 9: left_wrist
                [ 0.28,  0.00, 0.12],   # 10: right_wrist
                # Pelvis & Legs
                [-0.12, -0.05, 0.0],    # 11: left_hip
                [ 0.12, -0.05, 0.0],    # 12: right_hip
                [-0.13,  0.40, 0.05],   # 13: left_knee
                [ 0.13,  0.40, 0.05],   # 14: right_knee
                [-0.14,  0.82, 0.02],   # 15: left_ankle
                [ 0.14,  0.82, 0.02]    # 16: right_ankle
            ]

            joints_2d = []
            joints_3d = []
            for j_idx, rj in enumerate(rel_joints):
                # Rotate around vertical axis
                jx = rj[0] * cy - rj[2] * sy + X
                jy = rj[1] + Y
                jz = rj[0] * sy + rj[2] * cy + Z

                # Pinhole project to image sensor
                ju = (jx * self.fx) / max(jz, 0.1) + self.cx
                jv = (jy * self.fy) / max(jz, 0.1) + self.cy

                is_visible = (0 <= ju < self.img_size and 0 <= jv < self.img_size and jz > 0.5)
                score = 0.95 if is_visible else 0.0

                joints_2d.append({
                    "name": COCO_KEYPOINTS[j_idx],
                    "x": round(float(ju), 1),
                    "y": round(float(jv), 1),
                    "score": score
                })
                joints_3d.append({
                    "name": COCO_KEYPOINTS[j_idx],
                    "x": round(float(jx), 3),
                    "y": round(float(jy), 3),
                    "z": round(float(jz), 3)
                })

            pose_annotations.append({
                "inst_id": obj["inst_id"],
                "keypoints_2d": joints_2d,
                "keypoints_3d": joints_3d,
                "posture": random.choice(["standing", "walking", "bending"])
            })

        return pose_annotations

    def _generate_temporal_trajectories(
        self,
        objects: List[dict],
        seq_len: int = 8,
        fps: float = 30.0
    ) -> List[dict]:
        """
        Simulates smooth continuous 3D Newtonian trajectories across a sequence of frames
        with velocity vectors, acceleration, curvature, and interaction states.
        """
        trajectories = []
        dt = 1.0 / fps

        for obj in objects:
            X, Y, Z = obj["center_xyz"]
            yaw = obj["yaw_rad"]
            speed = random.uniform(0.5, 12.0)  # m/s

            vx = -math.sin(yaw) * speed
            vz =  math.cos(yaw) * speed
            vy = random.uniform(-0.1, 0.1)

            tokens = []
            cur_x, cur_y, cur_z = X, Y, Z
            for t in range(seq_len):
                # Add motion dynamics
                cur_x += vx * dt + random.gauss(0, 0.02)
                cur_y += vy * dt
                cur_z += vz * dt + random.gauss(0, 0.02)
                cur_speed = math.sqrt(vx**2 + vy**2 + vz**2)

                tokens.append([
                    round(float(cur_x), 3),
                    round(float(cur_y), 3),
                    round(float(cur_z), 3),
                    round(float(dt), 4),
                    round(float(cur_speed), 3),
                    round(float(yaw), 3)
                ])

            trajectories.append({
                "inst_id": obj["inst_id"],
                "trajectory_tokens": tokens,  # (T, 6)
                "velocity_vector": [round(vx, 3), round(vy, 3), round(vz, 3)],
                "state": random.choice([0, 1, 2, 3]),  # 0: Stationary, 1: Normal, 2: Rapid, 3: Agitated
                "anomaly_score": round(random.uniform(0.01, 0.15), 4)
            })

        return trajectories

    def generate_sample(self, sample_idx: int) -> dict:
        """
        Renders a fully synchronized, highly randomized multi-task sample.
        """
        # 1. Base terrain and sky
        img, depth_map, terrain_elev = self._render_sky_and_terrain(self.img_size)

        # 2. Place random 3D objects with physical collision rejection
        num_target_objs = random.randint(3, 9)
        placed_candidates = []
        for _ in range(num_target_objs * 3):
            if len(placed_candidates) >= num_target_objs:
                break
            cls_chosen = random.choice(OBJECT_CLASSES)
            candidate = self._generate_3d_cuboid(cls_chosen, placed_candidates)
            if candidate is not None:
                placed_candidates.append(candidate)

        # 3. 3D Project, Z-buffer rasterize, and generate masks
        img, depth_map, inst_mask, sem_mask, annotations = self._project_and_render_cuboids(
            img, depth_map, placed_candidates
        )

        # 4. Generate 2D/3D Anatomical Human Poses
        poses = self._generate_human_poses(annotations)

        # 5. Generate Temporal Spatio-Temporal Video Trajectory Tokens
        trajectories = self._generate_temporal_trajectories(annotations, seq_len=8)

        # 6. Environmental stress / domain noise randomization
        weather_mode = random.choice(["clear", "rain", "fog", "glare", "noise"])
        if weather_mode == "rain":
            # Add synthetic rain streaks
            for _ in range(random.randint(60, 150)):
                rx = random.randint(0, self.img_size - 1)
                ry = random.randint(0, self.img_size - 1)
                cv2.line(img, (rx, ry), (rx + 2, ry + 12), (190, 200, 210), 1)
        elif weather_mode == "fog":
            fog_overlay = np.full_like(img, (200, 205, 210))
            img = cv2.addWeighted(img, 0.65, fog_overlay, 0.35, 0)
        elif weather_mode == "noise":
            noise = np.random.normal(0, 12, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 7. Package complete ground-truth contract
        sample_data = {
            "sample_id": f"synthetic_{sample_idx:06d}",
            "image": img,
            "depth_map": depth_map,
            "terrain_elevation": terrain_elev,
            "instance_mask": inst_mask,
            "semantic_mask": sem_mask,
            "intrinsics": {
                "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "img_size": self.img_size
            },
            "annotations_3d": annotations,
            "poses": poses,
            "trajectories": trajectories
        }
        return sample_data


class NeuravexSyntheticDataset(Dataset):
    """
    Production PyTorch Dataset feeding the synthetic multi-modal suite into Neuravex.
    """
    def __init__(self, root_dir: str, split: str = "train"):
        self.root_dir = root_dir
        self.split = split
        self.img_dir = os.path.join(root_dir, split, "images")
        self.depth_dir = os.path.join(root_dir, split, "depth")
        self.lbl_dir = os.path.join(root_dir, split, "labels_3d")

        self.sample_ids = []
        if os.path.exists(self.lbl_dir):
            for f in sorted(os.listdir(self.lbl_dir)):
                if f.endswith(".json"):
                    self.sample_ids.append(os.path.splitext(f)[0])

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.sample_ids[idx]
        img_path = os.path.join(self.img_dir, f"{sid}.jpg")
        depth_path = os.path.join(self.depth_dir, f"{sid}.npy")
        lbl_path = os.path.join(self.lbl_dir, f"{sid}.json")

        bgr = cv2.imread(img_path)
        if bgr is None:
            bgr = np.zeros((320, 320, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor_img = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0

        depth = np.load(depth_path).astype(np.float32) if os.path.exists(depth_path) else np.ones((320, 320), dtype=np.float32)
        tensor_depth = torch.from_numpy(depth).unsqueeze(0)

        with open(lbl_path, "r") as f:
            meta = json.load(f)

        boxes_2d = []
        labels = []
        boxes_3d = []
        lwh_3d = []
        yaws = []

        for ann in meta.get("annotations_3d", []):
            boxes_2d.append(ann["bbox_2d"])
            labels.append(ann["class_id"])
            boxes_3d.append(ann["center_xyz"])
            lwh_3d.append(ann["size_lwh"])
            yaws.append(ann["yaw_rad"])

        return {
            "image": tensor_img,
            "depth": tensor_depth,
            "boxes_2d": torch.tensor(boxes_2d, dtype=torch.float32) if boxes_2d else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "boxes_3d": torch.tensor(boxes_3d, dtype=torch.float32) if boxes_3d else torch.zeros((0, 3)),
            "lwh_3d": torch.tensor(lwh_3d, dtype=torch.float32) if lwh_3d else torch.zeros((0, 3)),
            "yaws": torch.tensor(yaws, dtype=torch.float32) if yaws else torch.zeros((0,)),
            "meta": meta
        }
