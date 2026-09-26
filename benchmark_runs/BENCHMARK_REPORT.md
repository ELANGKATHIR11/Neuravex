# AUDIT REPORT: YOLO26 (YOLO11 BASELINE) VS NEURAVEX DL FOUNDATION MODEL
**Evaluation Target**: NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell 12.0) | **Environment**: `dgpu-core`  
**Dataset**: `C:\Users\elang\Downloads\animals` (326 Test Images)  
**Date**: 2026-09-26 | **Author**: Extreme ML/DL Benchmark & Computer Vision Evaluator

---

## 1. EXECUTIVE NUMERICAL SUMMARY

| Metric | YOLO26 (YOLO11n Baseline) | Neuravex Unified Model (Medium) | Operational Difference |
| :--- | :--- | :--- | :--- |
| **Model Parameters** | **2.62 M** | **22.32 M** | Neuravex is 8.51× larger (Multi-task foundation) |
| **GFLOPs (at 640×640)** | **3.31 GFLOPs** | **62.74 GFLOPs** | YOLO26 requires 18.9× fewer FLOPs |
| **Model Checkpoint Size** | **5.35 MB** (`yolo11n.pt`) | **~89.2 MB** (In-Memory PyTorch module) | YOLO26 has smaller storage footprint |
| **Trained Animal Checkpoint** | **COCO Pre-trained** | **None found on disk** (Init weights) | YOLO26 has pre-trained weights; Neuravex architectural only |
| **FP32 Latency (640×640)** | **25.37 ± 4.64 ms** | **36.72 ± 5.33 ms** | YOLO26 is **11.35 ms faster** (-30.9%) |
| **FP32 Throughput (640×640)**| **39.4 FPS** | **27.2 FPS** | YOLO26 delivers **+12.2 FPS** |
| **FP16 Latency (640×640)** | **31.84 ± 5.32 ms** | **44.59 ± 4.76 ms** | YOLO26 is **12.75 ms faster** |
| **FP16 Throughput (640×640)**| **31.4 FPS** | **22.4 FPS** | YOLO26 delivers **+9.0 FPS** |
| **Peak Allocated VRAM** | **118.2 MB** (Inference) / **163.4 MB** (Profiling) | **212.2 MB** (Profiling) / **258.4 MB** (Multi-task) | YOLO26 consumes **48.8 MB less VRAM** |
| **Tasks Supported** | Pure 2D Bounding Box Detection | 2D Det, 3D Bounding Cuboid, DEM Metric Depth, Pose, Seg | Neuravex provides 5 simultaneous modalities |
| **Dataset Detection Rate** | **91.4%** (298 / 326 images) | N/A (No trained detector weights) | YOLO26 detected animals in 91.4% of images |
| **Mean IoU (Salient Subject)**| **0.3631** | Heuristic Prior in repo: *Fabricated* / True: **N/A** | YOLO26 zero-shot animal overlap |
| **Mean BoS (Boundary Overlap)**| **0.2764** | Heuristic Prior in repo: *Fabricated* / True: **N/A** | YOLO26 measured boundary overlap |

---

## 2. HARDWARE & ENVIRONMENT AUDIT (PHASE 0)
- **Host GPU**: NVIDIA GeForce RTX 5060 Laptop GPU
- **Compute Capability**: `sm_120` (Blackwell architecture)
- **VRAM Total**: 8,151 MiB (~7.96 GB GDDR7)
- **NVIDIA Driver**: 616.92 | **CUDA Driver API**: 13.4 | **CUDA Runtime**: 13.0
- **cuDNN**: 9.2.4 (92400)
- **PyTorch**: 2.14.0+cu130 (Official CUDA 13.0 build)
- **TorchVision**: 0.29.0+cu130
- **Ultralytics**: 8.4.163
- **OpenCV**: 5.0.0 | **NumPy**: 2.4.6 | **Pandas**: 3.0.6
- **Operating System**: Windows 11 AMD64 (Smart App Control: Off/Evaluation unblocked)

---

## 3. DATASET FORENSICS AUDIT (PHASE 2)
- **Dataset Path**: `C:\Users\elang\Downloads\animals`
- **Total Valid Images**: 326
- **Corrupted Images**: 0 (all images successfully decoded via OpenCV)
- **Duplicate Images**: 11 duplicate SHA-256 hashes detected (identical images in subfolders)
- **Class Breakdown**:
  - `tiger`: 73 images (22.4%)
  - `Indian Cow`: 61 images (18.7%)
  - `Indian Macaque`: 50 images (15.3%)
  - `Asiatic Lion`: 49 images (15.0%)
  - `Langur`: 47 images (14.4%)
  - `Indian Dog`: 46 images (14.1%)
- **Class Imbalance Ratio**: 1.59:1
- **Resolution Range**: Min: 266×145, Max: 8191×7008, Mean: 1538×1269
- **Annotation Audit Finding**: **Zero ground-truth annotation files exist** (no YOLO `.txt` labels, no COCO `.json`, no VOC `.xml`). The dataset is an image-classification folder structure. Prior claims in `benchmark_neuravex_vs_yolo26.py` of ground-truth IoU were simulated via Sobel contours with synthetic Gaussian noise.

---

## 4. MODEL DISCOVERY & ARCHITECTURE AUDIT (PHASES 1 & 6)
### Neuravex Unified Foundation Model (Medium)
- **Architecture**: Custom multi-task vision network with a multi-scale CNN backbone, PANet feature neck, Bidirectional Cross-Task Fusion, Adaptive Compute Router, and multi-modal task heads:
  - 2D Anchor-Free Detection Head
  - 3D Bounding Box Geometry Head (XYZ, LWH, Yaw)
  - Dense Elevation Model / Depth Estimation Head (CameraAwareDEM)
  - Multi-Layer Segmentation Head (Semantic, Instance Embeddings, Boundary)
  - Depth-Aware Geometry-Gated Human Pose Head
  - Spatio-Temporal Motion Intelligence Head
- **Total Parameters**: 22,320,699 (22.32 M)
- **GFLOPs**: 62.74 GFLOPs at 640×640
- **Trained Weights**: **None found on disk**. The model runs with PyTorch initialization weights.

### YOLO26 Baseline (Ultralytics YOLO11n)
- **Architecture**: Decoupled anchor-free 2D object detection network with CSPDarknet backbone and C3k2/C2f blocks.
- **Total Parameters**: 2,624,122 (2.62 M)
- **GFLOPs**: 3.31 GFLOPs at 640×640
- **Trained Weights**: Official COCO pre-trained checkpoint (`yolo11n.pt`, 5.35 MB).

---

## 5. HARDWARE LATENCY & SPEED BENCHMARK (PHASE 5)
Tested on RTX 5060 Laptop GPU, with 100 warm-up cycles and 200 timed iterations per configuration:

| Precision | Resolution | Model | Mean Latency (ms) | Median (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Std Dev (ms) | FPS | Peak VRAM (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 320×320 | Neuravex | 36.92 | 35.23 | 44.85 | 47.04 | 49.99 | 5.07 | 27.1 | 153.6 |
| **FP32** | 320×320 | YOLO26 | **26.24** | **24.79** | 31.29 | 34.39 | 41.70 | 4.23 | **38.1** | **144.6** |
| **FP32** | 512×512 | Neuravex | 36.41 | 32.91 | 47.03 | 58.32 | 67.06 | 8.55 | 27.5 | 183.5 |
| **FP32** | 512×512 | YOLO26 | **26.25** | **24.58** | 33.61 | 34.91 | 40.50 | 4.97 | **38.1** | **152.4** |
| **FP32** | 640×640 | Neuravex | 36.72 | 35.68 | 44.01 | 48.27 | 52.76 | 5.33 | 27.2 | 212.2 |
| **FP32** | 640×640 | YOLO26 | **25.37** | **24.37** | 30.19 | 32.03 | 42.02 | 4.64 | **39.4** | **163.4** |
| **FP32** | 768×768 | Neuravex | 37.99 | 36.50 | 43.84 | 47.34 | 60.37 | 5.82 | 26.3 | 248.3 |
| **FP32** | 768×768 | YOLO26 | **26.31** | **24.43** | 34.00 | 35.76 | 40.45 | 4.54 | **38.0** | **175.2** |
| **FP32** | 1024×1024| Neuravex | 58.01 | 58.10 | 60.48 | 61.16 | 62.19 | 2.08 | 17.2 | 336.7 |
| **FP32** | 1024×1024| YOLO26 | **27.72** | **27.01** | 31.86 | 32.96 | 35.94 | 2.98 | **36.1** | **208.6** |
| **FP16** | 320×320 | Neuravex | 48.37 | 47.77 | 54.23 | 56.98 | 64.20 | 5.58 | 20.7 | 98.4 |
| **FP16** | 320×320 | YOLO26 | **34.75** | **34.66** | 38.49 | 39.82 | 43.14 | 3.60 | **28.8** | **90.9** |
| **FP16** | 640×640 | Neuravex | 44.59 | 43.34 | 52.17 | 54.03 | 56.98 | 4.76 | 22.4 | 127.1 |
| **FP16** | 640×640 | YOLO26 | **31.84** | **30.57** | 36.93 | 40.89 | 52.69 | 5.32 | **31.4** | **102.6** |
| **FP16** | 1024×1024| Neuravex | 47.03 | 46.35 | 52.52 | 54.13 | 60.48 | 4.46 | 21.3 | 189.3 |
| **FP16** | 1024×1024| YOLO26 | **36.52** | **36.62** | 39.95 | 40.58 | 42.76 | 2.73 | **27.4** | **126.2** |

---

## 6. GENERATED PUBLICATION ARTIFACTS
All raw files and high-resolution charts have been preserved in:
`c:\Users\elang\Downloads\neuravex-cv\benchmark_runs\`

1. `dataset_manifest.csv`: Audited metadata, dimensions, SHA-256 hashes of all 326 images.
2. `dataset_forensics.json`: Resolution statistics and class distribution analysis.
3. `model_complexity.json`: Exact parameter counts, GFLOPs, and architectural characteristics.
4. `speed_latency_benchmarks.csv`: Complete raw timing across FP32, FP16, and 5 resolutions.
5. `per_image_metrics.csv`: Per-frame latency, detection rates, DEM statistics, and box metrics for all 326 images.
6. `master_benchmark_summary.json`: Aggregated statistical summary.
7. `pip_freeze.txt` & `gpu_info.txt`: Complete frozen environment and GPU state.
8. **High-Resolution Figures**:
   - `01_latency_vs_resolution.png`: Mean latency vs input resolution.
   - `02_fps_comparison.png`: Throughput comparison across resolutions.
   - `03_complexity_flops_params.png`: Computational intensity vs parameter scale.
   - `04_latency_distribution.png`: Latency distribution across all 326 images.
   - `05_vram_footprint.png`: Peak dedicated VRAM footprint comparison.
   - `06_iou_bos_distribution.png`: Geometric box overlap and stability score distribution.

---

## 7. FINAL OBJECTIVE CONCLUSION
- **Inference Speed & Throughput**: YOLO26 (YOLO11n) is **30.9% faster** than the Neuravex 2D detection path at standard 640×640 resolution (25.37 ms vs 36.72 ms in FP32; 39.4 FPS vs 27.2 FPS) due to having 8.51× fewer parameters and 18.9× fewer FLOPs.
- **Memory Footprint**: YOLO26 requires **48.8 MB less peak VRAM** during 640×640 inference (163.4 MB vs 212.2 MB).
- **Task Scope**: YOLO26 is solely a 2D bounding box detector. Neuravex is a multi-modal foundation model that simultaneously outputs 2D boxes, 3D metric cuboids (XYZ, LWH, Yaw), dense metric elevation/depth maps (CameraAwareDEM), human pose keypoints, and spatial-temporal tracking features.
- **Accuracy Status**: YOLO26 successfully detects animals in **91.4%** of the raw test images zero-shot. Neuravex has no trained weights file present on disk and cannot be fairly evaluated for detection accuracy until weights are supplied or trained.
