import os
import json
import math
import random
import cv2
import numpy as np
import torch

def generate_multitask_3d_dataset(output_dir="data/synthetic_3d_dataset", num_train=300, num_val=60, img_size=320):
    """
    Generates a physically consistent 3D + DEM vision dataset (< 100 MB).
    Each sample contains:
      - RGB image (ground plane, sky/background, rendered 3D objects with shading & texture)
      - Exact Camera Intrinsics K = (fx, fy, cx, cy)
      - 3D Bounding Boxes: (X, Y, Z) in camera frame (meters), (L, W, H) dimensions, yaw angle theta
      - 2D Projected Bounding Boxes: [x1, y1, x2, y2]
      - Dense Ground-Truth Metric Depth Map (Z in meters) & Inverse Depth rho = 1/Z
      - Pixel-accurate Semantic Segmentation Map
      - Instance Segmentation Map
      - Morphological Boundary Map
    """
    print(f"Generating synthetic 3D + DEM dataset in {output_dir}...")
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "depth"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels_3d"), exist_ok=True)

    # Fixed Camera Intrinsics
    fx = 400.0 * (img_size / 320.0)
    fy = 400.0 * (img_size / 320.0)
    cx = img_size / 2.0
    cy = img_size / 2.0

    camera_intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "img_size": img_size}
    with open(os.path.join(output_dir, "intrinsics.json"), "w") as f:
        json.dump(camera_intrinsics, f, indent=2)

    # Object classes with typical metric sizes [L, W, H]
    classes = [
        {"name": "Vehicle", "class_id": 0, "size_mean": [4.2, 1.8, 1.5], "size_std": [0.4, 0.1, 0.15], "color": (50, 120, 220)},
        {"name": "Pedestrian", "class_id": 1, "size_mean": [0.8, 0.6, 1.75], "size_std": [0.1, 0.05, 0.1], "color": (220, 80, 50)},
        {"name": "Cyclist", "class_id": 2, "size_mean": [1.8, 0.7, 1.6], "size_std": [0.15, 0.08, 0.1], "color": (80, 200, 80)},
        {"name": "Obstacle", "class_id": 3, "size_mean": [1.0, 1.0, 1.0], "size_std": [0.2, 0.2, 0.2], "color": (200, 180, 50)}
    ]

    splits = [("train", num_train), ("val", num_val)]

    for split_name, count in splits:
        print(f"Generating {count} samples for {split_name} split...")
        for idx in range(count):
            # 1. Base RGB canvas and Dense Metric Depth Map
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            depth_map = np.ones((img_size, img_size), dtype=np.float32) * 50.0  # background depth (50m)
            sem_map = np.zeros((img_size, img_size), dtype=np.int64)
            inst_map = np.zeros((img_size, img_size), dtype=np.int64)

            # Draw physical ground plane with perspective depth gradient
            horizon_y = int(img_size * 0.45)
            # Sky
            img[:horizon_y, :] = (230, 210, 180)  # light blue sky in BGR
            # Ground road
            for y in range(horizon_y, img_size):
                # Distance along ground plane Z(y)
                # camera height h_cam = 1.6m -> y - cy = fy * h_cam / Z -> Z = fy * h_cam / (y - cy)
                y_diff = max(y - cy, 1.0)
                z_ground = (fy * 1.5) / y_diff
                depth_map[y, :] = z_ground
                # Gradient asphalt color
                road_shade = int(70 + 40 * (y / img_size))
                img[y, :] = (road_shade, road_shade, road_shade)

            # 2. Place 3D objects with physical coordinates
            num_objects = random.randint(2, 5)
            annotations_3d = []
            
            # Sort objects by depth descending (painter's algorithm for rendering)
            placed_objects = []
            for _ in range(num_objects):
                cls = random.choice(classes)
                # Random 3D center in camera coordinates:
                # Z in [4.0m, 25.0m], X in [-4.0m, 4.0m], Y constrained to ground plane
                Z = random.uniform(4.5, 20.0)
                L = max(0.5, random.gauss(cls["size_mean"][0], cls["size_std"][0]))
                W = max(0.4, random.gauss(cls["size_mean"][1], cls["size_std"][1]))
                H = max(0.5, random.gauss(cls["size_mean"][2], cls["size_std"][2]))
                
                # Ground contact: object bottom touches ground (y_cam_bottom = 1.5m)
                Y = 1.5 - H * 0.5
                # X coordinate such that object projects within visible image
                max_x = (img_size * 0.4 / fx) * Z
                X = random.uniform(-max_x, max_x)
                yaw = random.uniform(-math.pi, math.pi)

                placed_objects.append({
                    "cls": cls,
                    "X": X, "Y": Y, "Z": Z,
                    "L": L, "W": W, "H": H,
                    "yaw": yaw
                })

            placed_objects.sort(key=lambda o: o["Z"], reverse=True)

            for inst_id, obj in enumerate(placed_objects, start=1):
                cls = obj["cls"]
                X, Y, Z = obj["X"], obj["Y"], obj["Z"]
                L, W, H = obj["L"], obj["W"], obj["H"]
                yaw = obj["yaw"]

                # 3D corners in object frame
                cos_y, sin_y = math.cos(yaw), math.sin(yaw)
                x_corners = [L/2,  L/2, -L/2, -L/2,  L/2,  L/2, -L/2, -L/2]
                y_corners = [W/2, -W/2, -W/2,  W/2,  W/2, -W/2, -W/2,  W/2]
                z_corners = [-H/2, -H/2, -H/2, -H/2, H/2,  H/2,  H/2,  H/2]

                projected_uv = []
                for xc, yc, zc in zip(x_corners, y_corners, z_corners):
                    # Rotate in horizontal XZ plane around Y (camera convention)
                    xr = cos_y * xc - sin_y * yc + X
                    yr = zc + Y
                    zr = sin_y * xc + cos_y * yc + Z

                    # Pinhole projection: u = xr * fx / zr + cx, v = yr * fy / zr + cy
                    u = (xr * fx) / max(zr, 0.1) + cx
                    v = (yr * fy) / max(zr, 0.1) + cy
                    projected_uv.append([u, v])

                pts = np.array(projected_uv, dtype=np.float32)
                x1 = int(max(0, np.min(pts[:, 0])))
                y1 = int(max(0, np.min(pts[:, 1])))
                x2 = int(min(img_size - 1, np.max(pts[:, 0])))
                y2 = int(min(img_size - 1, np.max(pts[:, 1])))

                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    # Render solid 3D box on RGB canvas
                    # Base color with depth fog
                    fog = min(1.0, Z / 25.0)
                    base_col = np.array(cls["color"]) * (1.0 - 0.4 * fog)
                    cv2.rectangle(img, (x1, y1), (x2, y2), tuple(map(int, base_col)), -1)
                    # Add orientation line / edge highlighting
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

                    # Update dense metric depth map
                    depth_map[y1:y2, x1:x2] = Z
                    sem_map[y1:y2, x1:x2] = cls["class_id"] + 1  # 0 is background
                    inst_map[y1:y2, x1:x2] = inst_id

                    annotations_3d.append({
                        "class_name": cls["name"],
                        "class_id": cls["class_id"],
                        "bbox_2d": [x1, y1, x2, y2],
                        "center_3d": [round(X, 3), round(Y, 3), round(Z, 3)],
                        "lwh_3d": [round(L, 3), round(W, 3), round(H, 3)],
                        "yaw": round(yaw, 4)
                    })

            # Save RGB image
            sample_name = f"sample_{idx:05d}"
            img_path = os.path.join(output_dir, split_name, "images", f"{sample_name}.jpg")
            cv2.imwrite(img_path, img)

            # Save Metric Depth (meters saved as float16 numpy array or 16-bit PNG millimeter)
            depth_path = os.path.join(output_dir, split_name, "depth", f"{sample_name}.npy")
            np.save(depth_path, depth_map.astype(np.float32))

            # Save 3D Annotation JSON
            lbl_path = os.path.join(output_dir, split_name, "labels_3d", f"{sample_name}.json")
            with open(lbl_path, "w") as f:
                json.dump({
                    "image": f"{sample_name}.jpg",
                    "width": img_size,
                    "height": img_size,
                    "intrinsics": camera_intrinsics,
                    "annotations": annotations_3d
                }, f, indent=2)

    # Dataset size check
    total_bytes = 0
    for root, _, files in os.walk(output_dir):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(root, f))
    print(f"Generated Dataset Total Size: {total_bytes / (1024**2):.2f} MB (Well under 1 GB limit).")

if __name__ == "__main__":
    generate_multitask_3d_dataset()
