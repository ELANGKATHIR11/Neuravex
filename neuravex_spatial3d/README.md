# neuravex-spatial3d

High-Performance Native 3D Spatial Perception, Photo, Image & Video Analysis, Sensor Fusion, Native DEM, and Volumetric Flow Counting SDK.

[![Release](https://img.shields.io/badge/Release-v0.1.0-brightgreen.svg)](https://github.com/ELANGKATHIR11/Neuravex)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![CUDA & cuDNN](https://img.shields.io/badge/CUDA%20%7C%20cuDNN-Accelerated-76b900.svg)](https://nvidia.com)
[![Intel & AMD](https://img.shields.io/badge/Hardware-Intel%20%26%20AMD%20AVX2%2F512-blueviolet.svg)](https://github.com/ELANGKATHIR11/Neuravex)

---

## Capabilities & Full DL Model Integration

1. **Native Photo & Image Analysis (`ImageProcessor` & `NativeSpatialAnalyzer`)**:
   - Accepts JPG, PNG, WebP, BMP, raw NumPy arrays, PIL Images, and PyTorch tensors.
   - Computes 2D classification, metric depth maps, 3D $(L, W, H)$ bounding boxes, and ground DEM elevation profiles.
2. **Native Video Stream Analysis (`VideoStreamProcessor` & `VideoWriter`)**:
   - Processes MP4, AVI, MKV, RTSP/HTTP camera streams, and USB webcams frame-by-frame.
   - Real-time 3D spatial overlay rendering and direct output video encoding.
3. **Native 3D Bounding Boxes**: Metric center $(X, Y, Z)$, physical extents $(Length, Width, Height)$, and heading angles $(Yaw, Pitch, Roll)$ with 8-corner camera vertices computation.
4. **Native Camera Intrinsics**: Full pinhole model $(f_x, f_y, c_x, c_y)$ with lens distortion correction and pixel unprojection.
5. **LiDAR Sensor Fusion**: Spherical-to-cartesian unprojection, extrinsic rigid-body $SE(3)$ transformation ($R, T$), and LiDAR-to-image projection.
6. **Native DEM (Digital Elevation Model)**: Converts metric depth grids into terrain elevation surfaces, computing slope (deg), aspect (deg), and ground plane fitting.
7. **Native 3D Instance Segmentation**: Back-projects 2D masks into 3D point cloud clusters and isolates object volumes without rectangular bounding box leakage.
8. **Native Entity Marking & Spatial Memory**: Persistent identity tracking, 3D trajectory recording, and spatial memory querying.
9. **Native Volumetric Counting**: Virtual 3D zones, tripwires, and cumulative flow counting.
10. **Cross-Platform Compatibility**: Fully compatible with Python 3.11+, NVIDIA CUDA & cuDNN, Intel CPUs (oneDNN/AVX-512), and AMD processors/GPUs (AVX2/ROCm).

---

## Installation

```bash
cd ml_neuravex/neuravex_spatial3d
pip install -e .
```

---

## Quickstart: Photo & Image Analysis

```python
import neuravex_spatial3d as sp3d

# 1. Initialize analyzer with hardware auto-detection (CUDA/cuDNN or Intel/AMD CPU)
analyzer = sp3d.NativeSpatialAnalyzer(model_name="neuravex-nano")

# 2. Analyze any photo / image directly
results = analyzer.analyze_image("sample_photo.jpg", conf_threshold=0.25)
print("Image Latency:", results["latency_ms"], "ms")
print("3D Bounding Boxes (L,W,H):", results["boxes_3d"])
print("DEM Elevation (min/max/mean):", results["dem_metrics"])
```

---

## Quickstart: Real-Time Video Analysis & Stream Processing

```python
import neuravex_spatial3d as sp3d

analyzer = sp3d.NativeSpatialAnalyzer()

# Define a 3D volumetric counting gate (e.g. entry zone)
zone = sp3d.CountingZone3D("gate_north", "Main Gate", x_min=-2.0, x_max=2.0, y_min=-1.0, y_max=3.0, z_min=1.0, z_max=15.0)
analyzer.counter.add_zone(zone)

# Process video file or RTSP stream and save annotated 3D video output
video_report = analyzer.analyze_video(
    video_source="traffic_stream.mp4",  # or 0 for live webcam
    output_path="annotated_output.mp4",
    max_frames=300
)

print("Processed FPS:", video_report["fps"])
print("Cumulative Counts:", video_report["cumulative_counts"])
print("Zone Counts:", video_report["active_zone_counts"])
```

---

## Sensor Fusion & Native DEM Example

```python
import torch
import neuravex_spatial3d as sp3d

cam = sp3d.CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, width=1920, height=1080)
lidar = sp3d.LiDARConfig(num_beams=64, max_range=120.0)
fusion = sp3d.SensorFusionEngine(camera=cam, lidar=lidar)

# Project LiDAR point clouds onto image plane
lidar_pts = torch.randn(1000, 3) + torch.tensor([0, 0, 15.0])
u_proj, v_proj, valid = fusion.project_lidar_to_image(lidar_pts)

# DEM Terrain Surface
dem_grid = torch.randn(200, 200) + 10.0
dem = sp3d.NativeDEMSurface(dem_grid, cell_resolution_m=0.05)
slope, aspect = dem.compute_slopes_and_aspect()
normal, d = dem.fit_ground_plane()
```

---

## Native Live Video Streaming Pipeline (`LiveVideoPipeline`)

Continuous real-time live streaming generator with concurrent 2D/3D boxes, live segmentation, live DEM, physical $L, W, H$ calculation, persistent track ID creation, and spatial memory entity marking:

```python
import cv2
import neuravex_spatial3d as sp3d

# Initialize real-time live video pipeline
pipeline = sp3d.LiveVideoPipeline()

# Stream live video from USB webcam (0), RTSP IP camera, or MP4 file
for result, annotated_frame in pipeline.stream_live(video_source=0, conf_threshold=0.25):
    fps = result["fps"]
    boxes = result["boxes_3d"]
    dem = result["dem_metrics"]
    
    print(f"Live FPS: {fps} | Tracked Entities: {len(boxes)} | Mean DEM Elev: {dem['mean_elevation_m']}m")
    
    for box in boxes:
        print(f"  -> Entity ID: {box['track_id']} | LWH: {box['lwh']} | XYZ: {box['xyz']}")

    # Display real-time 3D dashboard overlay
    if annotated_frame is not None:
        cv2.imshow("Neuravex Native Live 3D Stream", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
```
