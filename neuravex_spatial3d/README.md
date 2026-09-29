# neuravex-spatial3d

High-Performance Native 3D Spatial Perception, Volumetric Bounding Boxes, Sensor Fusion, Native DEM, and Spatial Flow Counting SDK.

[![Release](https://img.shields.io/badge/Release-v0.1.0-brightgreen.svg)](https://github.com/ELANGKATHIR11/Neuravex)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![CUDA & cuDNN](https://img.shields.io/badge/CUDA%20%7C%20cuDNN-Accelerated-76b900.svg)](https://nvidia.com)
[![Intel & AMD](https://img.shields.io/badge/Hardware-Intel%20%26%20AMD%20AVX2%2F512-blueviolet.svg)](https://github.com/ELANGKATHIR11/Neuravex)

---

## Capabilities

1. **Native 3D Bounding Boxes**: Metric center $(X, Y, Z)$, physical extents $(L, W, H)$, and heading angles $(Yaw, Pitch, Roll)$ with 8-corner vertices computation.
2. **Native Camera Intrinsics**: Full pinhole model $(f_x, f_y, c_x, c_y)$ with lens distortion correction and pixel unprojection.
3. **LiDAR Sensor Fusion**: Spherical-to-cartesian unprojection, extrinsic rigid-body $SE(3)$ transformation ($R, T$), and LiDAR-to-image projection.
4. **Native DEM (Digital Elevation Model)**: Converts metric depth grids into terrain elevation surfaces, computing slope (deg), aspect (deg), and ground plane fitting.
5. **Native 3D Instance Segmentation**: Back-projects 2D masks into 3D point cloud clusters and isolates object volumes without rectangular bounding box leakage.
6. **Native Entity Marking & Spatial Memory**: Persistent identity tracking, 3D trajectory recording, and spatial memory querying.
7. **Native Volumetric Counting**: Virtual 3D zones, tripwires, and cumulative flow counting.
8. **Cross-Platform Compatibility**: Fully compatible with Python 3.11+, NVIDIA CUDA & cuDNN, Intel CPUs (oneDNN/AVX-512), and AMD processors/GPUs (AVX2/ROCm).

---

## Installation

```bash
cd ml_neuravex/neuravex_spatial3d
pip install -e .
```

---

## Quick Example

```python
import torch
import neuravex_spatial3d as sp3d

# 1. Hardware context auto-selects CUDA+cuDNN or AVX Intel/AMD
ctx = sp3d.DeviceContext()
print("Hardware:", ctx.backend["device_name"])

# 2. Camera Intrinsics & Unprojection
cam = sp3d.CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0, width=1920, height=1080)
u = torch.tensor([960.0])
v = torch.tensor([540.0])
depth = torch.tensor([4.5])  # 4.5 meters
xyz = cam.unproject_pixels(u, v, depth)

# 3. 3D Bounding Box with L, W, H
box = sp3d.BoundingBox3D(center=xyz[0], size_lwh=(1.8, 0.9, 1.4), yaw=0.15, class_name="vehicle")
print("Box Volume:", box.volume, "m^3")
print("8 Corners:\n", box.get_corners())

# 4. Digital Elevation Model (DEM)
dem_grid = torch.randn(200, 200) + 12.0
dem = sp3d.NativeDEMSurface(dem_grid, cell_resolution_m=0.05)
slope, aspect = dem.compute_slopes_and_aspect()

# 5. Volumetric Counting Zone
counter = sp3d.SpatialCounter()
zone = sp3d.CountingZone3D("gate_1", "North Gate", -2.0, 2.0, -1.0, 3.0, 0.0, 10.0)
counter.add_zone(zone)
report = counter.process_detections([box])
print("Spatial Count Report:", report)
```
