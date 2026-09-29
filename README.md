# Neuravex v0.2.0: Hardware-Adaptive Unified Computer-Vision Architecture

[![Release: v0.2.0](https://img.shields.io/badge/Release-v0.2.0-brightgreen.svg)](https://github.com/ELANGKATHIR11/Neuravex/releases)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests: 20/20 Subsystems](https://img.shields.io/badge/Subsystems-20%2F20%20Verified-success.svg)](https://github.com/ELANGKATHIR11/Neuravex)
[![Hardware Verified](https://img.shields.io/badge/RTX%205060%20%7C%20CPU-Empirically%20Verified-76b900.svg)](https://nvidia.com/)

**Neuravex v0.2.0** is an independent, production-grade deep learning architecture and hardware-adaptive computer-vision SDK. It unifies high-accuracy anchor-free detection, real-time metric depth estimation, 3D spatial bounding, instance segmentation, spatio-temporal video intelligence, and multi-stage distillation & quantization to achieve Pareto-optimal performance across edge and server hardware.

> [!IMPORTANT]
> **Independent DL Foundation**: Neuravex is an original, dedicated deep learning architecture and training/inference engine. All frontend UI/UX dependencies and web servers have been excised; the repository is exclusively focused on native Deep Learning models, neural heads, geometric transformations, and deployment runtimes (PyTorch, ONNX, TensorRT).

### Primary Objective

$$\boxed{\max \frac{\text{REAL Detection AP}}{\text{FLOPs}}}$$

subject to:

$$\min (\text{latency} + \text{VRAM} + \text{parameters} + \text{deployment cost})$$

Detection remains the **primary optimization target**. Auxiliary tasks (Segmentation, Metric Depth, 3D Geometry, DEM Terrain, Spatio-Temporal Tracking) share intermediate representations where beneficial and are completely bypassed during pure detection inference without incurring compute or latency penalties.

---

## What's New in Neuravex v0.2.0: Hardware-Adaptive Specialization & SOTA Distillation

Neuravex transforms the traditional paradigm of static vision models into an adaptive, budget-driven pipeline:

$$\text{User Dataset} + \text{Hardware Constraints} \longrightarrow \mathbf{Dataset\ Analyzer} \longrightarrow \mathbf{Architecture\ Generator} \longrightarrow \mathbf{Specialized\ Neuravex} \longrightarrow \mathbf{Deploy}$$

1. **Dataset-Conditioned Specialization**: Analyzes target datasets into a 16-dimensional complexity vector and synthesizes tailored network topologies.
2. **20-Subsystem Architecture Upgrade**: Complete suite featuring dynamic resolution routing, spatial token pruning, intermediate early exits, and temporal feature caching.
3. **Hardware-Aware Controller**: Auto-profiles CPU/GPU capabilities, RAM/VRAM budgets, and dynamically configures precision (FP32, FP16, INT8 PTQ) and execution providers.
4. **SOTA Distillation & Feature Hinting**: Multi-tier knowledge distillation with FitNets intermediate hint loss, DFL distribution transfer, and boundary-aware refinement.
5. **Real-Time Metric Depth & 3D Spatial Unprojection**: Pinhole camera geometry integrating metric depth estimation with 3D oriented bounding boxes and Kalman state tracking.

---

## Architecture Specification

```mermaid
flowchart TD
    Input["Input Frame (B, 3, H, W)"] --> Stem["Stem & Multi-Scale RepConv (Strides 2, 4)"]
    Stem --> DynRes["Dynamic Resolution & Token Router (320 / 512 / 640)"]
    DynRes --> P2Opt["P2 High-Res Map (Stride 4) [Micro Objects]"]
    DynRes --> P3["P3 Feature Map (Stride 8)"]
    DynRes --> P4["P4 Feature Map (Stride 16)"]
    DynRes --> P5["P5 Feature Map (Stride 32)"]
    DynRes --> P6Opt["P6 Context Map (Stride 64) [Panoramic]"]

    subgraph Core["Partial-Channel Reparameterizable Backbone (RepConv + P-RepBlock)"]
        P3 & P4 & P5 --> PANet["Multi-Scale PAN/FPN Depthwise Separable Fusion"]
        PANet --> Q3["Q3 Feature Map"]
        PANet --> Q4["Q4 Feature Map"]
        PANet --> Q5["Q5 Feature Map"]
    end

    subgraph FastPath["Detection-First Primary Fast Path (Zero Auxiliary Overhead)"]
        Q3 & Q4 & Q5 --> Router["Multi-Level Difficulty & Early-Exit Router"]
        Router --> DetHead["Decoupled 2D/3D Anchor-Free Detection Head"]
        DetHead --> Out2D["2D Detection: Cls Logits + Box Regression (DFL / Direct)"]
        DetHead --> Out3D["3D Metric Geometry: (X,Y,Z), (L,W,H), Yaw via Camera Intrinsics"]
    end

    subgraph Auxiliary["Auxiliary Multi-Task Neural Heads (Selectively Activated)"]
        Q3 & Q4 --> CrossAdapter["Bidirectional Cross-Task Fusion Token"]
        CrossAdapter --> PoseHead["Depth-Aware Geometry Pose & Skeleton Head"]
        CrossAdapter --> SegHead["Prototype Instance & Semantic Segmentation Head"]
        CrossAdapter --> DEMHead["Camera-Aware Metric Depth Z & DEM Terrain Head"]
        CrossAdapter --> FlowHead["Spatio-Temporal Tracking & Neural Flow Zone Counter"]
    end

    subgraph EdgeDeploy["Deployment & Optimization Engine"]
        FastPath & Auxiliary --> Pruning["Structural Channel Pruner (L1-Norm)"]
        Pruning --> Quant["INT8 / FP16 Quantization Engine (PTQ / QAT)"]
        Quant --> ONNX["ONNX / TensorRT / OpenVINO Runtime Engine"]
    end
```

### Core Architecture Components

1. **Backbone (`P-RepBlock`)**: Lightweight partial-channel reparameterizable backbone. Multi-branch `RepConv` during training collapses into a single mathematically exact $3\times3$ convolution at deployment ($\max |\Delta| < 10^{-4}$).
2. **Neck (`PANet`)**: Bidirectional Feature Pyramid Network with depthwise separable cross-scale fusion, minimizing memory movement and peak VRAM.
3. **Adaptive Compute Router**: Uncertainty and confidence-driven multi-level routing:
   $$y = f_{\text{base}}(x) + g(x) \cdot f_{\text{refine}}(x),\quad g \in [0, 1]$$
4. **Decoupled Detection Head**: Anchor-free classification and bounding-box regression supporting both `reg_max > 1` (Distribution Focal Loss) and `reg_max = 1` (direct regression).
5. **Real-Time Metric Depth + Camera 3D**:
   $$Z = \text{depth},\quad X = \frac{(u - c_x) Z}{f_x},\quad Y = \frac{(v - c_y) Z}{f_y},\quad R = \sqrt{X^2 + Y^2 + Z^2}$$
   Fused with instance masks via confidence-weighted robust estimators and tracked with temporal Kalman filtering.
6. **Auxiliary Cross-Task Adapters**: Instance segmentation, pose estimation, and DEM terrain heads that strictly decouple from the detection fast path when inactive.

---

## Package Architecture (`neuravex`)

```
neuravex/
├── models/
│   ├── neuravex.py              # Unified Neuravex model & factory (build_neuravex)
│   ├── backbone.py              # Partial-channel RepConv backbone & P-RepBlock
│   ├── neck.py                  # Multi-scale PANet depthwise feature fusion
│   ├── heads_det.py             # Decoupled 2D/3D anchor-free detection head
│   ├── heads_depth.py           # Metric depth & DEM elevation heads
│   ├── heads_seg.py             # Prototype-based instance/semantic segmentation
│   ├── heads_pose.py            # Depth-aware geometry pose & keypoint head
│   ├── heads_st_intelligence.py # Spatio-temporal intelligence & trajectory head
│   ├── heads_flow_counter.py    # Promptable prototype head & neural flow zone counter
│   └── slots.py                 # SlotAttention & unsupervised object discovery
├── specialization/
│   ├── dataset_analyzer.py      # 16-dim COCO dataset complexity analyzer
│   ├── architecture_generator.py # Budget-constrained ArchSpec generator
│   ├── difficulty_router.py     # 3-level (easy/medium/hard) adaptive router
│   ├── hardware_profiler.py     # Real P50/P95/P99 latency profiler & Pareto analysis
│   ├── distillation.py          # Multi-task distillation loss (KD + DFL + hint loss)
│   ├── structural_pruner.py     # Graph-aware structural channel pruner
│   ├── quantization_system.py   # INT8 PTQ and FP16 precision validator
│   ├── pruning_quantization.py  # Structured pruning & quantization orchestration
│   ├── nas_pareto.py            # Hardware-aware Pareto frontier search
│   ├── yolo26_baseline.py       # Official baseline comparison policy
│   ├── specialization_pipeline.py # End-to-end dataset-to-deployment orchestrator
│   └── experiment_engine.py     # Multi-stage ablation study engine
├── engine/
│   ├── trainer.py               # Multi-task AMP-enabled PyTorch trainer
│   ├── evaluator.py             # COCOeval, IoU, BoS, and multi-task evaluation
│   ├── tracker.py               # Real-time metric 3D bounding box & depth tracker
│   ├── adaptive_compute.py      # Adaptive compute runtime & dynamic budget reward
│   ├── precision_perception.py  # Precision perception pipeline & calibration
│   ├── temporal.py              # Temporal feature caching & inter-frame reuse
│   ├── uncertainty.py           # Heteroscedastic bounding box uncertainty estimation
│   ├── routing.py               # Dynamic layer/depth and token routing
│   ├── hardware_controller.py   # Host hardware auto-profiler & config selector
│   └── memory_opt.py            # Channels-last memory & inference mode optimizers
├── deployment/
│   ├── onnx_exporter.py         # Dynamic-batch ONNX exporter & graph simplifier
│   └── runtime_engine.py        # ONNX Runtime (CPU, CUDA, TensorRT) inference runner
├── loss/
│   ├── detection_loss.py        # Task-Aligned Assigner, VFL, CIoU, DFL
│   ├── depth_3d_loss.py         # Log-Laplace depth loss, scale-invariant RMSE
│   ├── photometric.py           # Photometric reconstruction loss & view synthesis warp
│   ├── physics_rl.py            # Physics constraint reward & test-time self-corrector
│   └── cross_task_temporal.py   # Multi-task loss balancing & temporal smoothness
└── geometry/
    ├── camera.py                # Camera intrinsics, coordinate unprojection, raycasting
    ├── box_ops.py               # 2D CIoU, GIoU, DIoU, and boundary overlap score (BoS)
    └── oriented_iou3d.py        # 3D oriented bounding box IoU calculation
```

---

## The Official Model Family Spectrum

| Model Variant | Parameters | 640x640 FLOPs | RTX 5060 FPS | CPU FPS | Target Deployment |
|---|---|---|---|---|---|
| 🟢 **Neuravex-Pico** | **0.54 M** | **0.43 G** (320p) | **109.9 FPS** | **54.4 FPS** | Microcontrollers, Raspberry Pi, IoT Edge |
| 🟢 **Neuravex-Femto** | **1.16 M** | **3.73 G** | **136.8 FPS** | **42.1 FPS** | Ultra-low power edge devices, drones |
| 🔵 **Neuravex-Nano** | **2.02 M** | **6.52 G** | **138.5 FPS** | **34.9 FPS** | NVIDIA Jetson Nano / Orin, Mobile Vision |
| 🔵 **Neuravex-Lite** | **4.58 M** | **15.11 G** | **70.2 FPS** | **18.5 FPS** | Robotics, Smart Cameras, Embedded Edge |
| 🟠 **Neuravex-Edge** | **9.08 M** | **26.62 G** | **88.6 FPS** | **11.2 FPS** | Edge Servers, Industrial Automation |
| 🔴 **Neuravex-Pro** | **22.36 M** | **64.81 G** | **64.7 FPS** | **5.4 FPS** | High-precision Cloud Video Analytics |
| 🟣 **Neuravex-Omni** | **35.96 M** | **179.48 G** | **45.0 FPS** | **2.8 FPS** | Unified Foundation Vision (2D/3D/Depth/Seg/Pose/Track) |

---

## Quickstart & Usage

### 1. Installation

```bash
git clone https://github.com/ELANGKATHIR11/Neuravex.git
cd Neuravex
pip install -e .
pip install pycocotools
```

### 2. End-to-End Specialization Pipeline

Generate an architecture tailored to your specific dataset and hardware budget:

```python
from neuravex.specialization import (
    DatasetAnalyzer,
    ArchitectureGenerator,
    HardwareConstraints,
    SpecializationPipeline,
)

# 1. Analyze user dataset
analyzer = DatasetAnalyzer("data/annotations.json")
profile = analyzer.analyze()
print(f"Dataset complexity: {profile.complexity_vector}")

# 2. Define hardware constraints
hardware = HardwareConstraints(
    device="cuda",
    target_latency_ms=25.0,
    max_memory_mb=1024.0,
    max_params_m=8.0,
)

# 3. Run specialization
pipeline = SpecializationPipeline(profile, hardware)
result = pipeline.run(input_shape=(1, 3, 640, 640))
model = result["specialized_model"]
print(f"Generated architecture: {result['arch_spec']}")
```

### 3. CLI Specialization Entrypoint

```bash
python run_specialization.py \
    --ann_file data/annotations.json \
    --device cuda \
    --latency_budget_ms 25.0 \
    --max_params_m 8.0 \
    --output_dir ./specialization_output
```

### 4. Direct Model Creation & Inference

```python
import torch
from neuravex import build_neuravex

# Instantiate standard v0.1-nano
model = build_neuravex(scale="nano", num_classes=80, tasks=("det",))
model.eval()

# Forward pass (detection fast path)
img = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    predictions = model(img)
```

---

## Benchmarks & Verification Status

### Empirically Measured [MEASURED on RTX 5060 Laptop GPU]

All numbers measured with CUDA 12.8, PyTorch 2.x, explicit synchronization across $\ge 60$ timed iterations:

| Model | Resolution | Batch | Params | Model Size | FLOPs (G) | P50 Latency | P95 Latency | Throughput |
|---|---|---|---|---|---|---|---|---|
| **Neuravex-Nano** | 320x320 | 1 | **1.48 M** | **5.69 MB** | **2.51 G** | 10.42 ms | 14.81 ms | 96.0 FPS |
| **Neuravex-Small** | 320x320 | 1 | 6.07 M | 23.21 MB | 10.19 G | 11.23 ms | 15.65 ms | 89.0 FPS |
| **Neuravex-Nano** | 640x640 | 1 | **1.48 M** | **5.69 MB** | **10.05 G** | 10.25 ms | 15.20 ms | 97.6 FPS |
| **Neuravex-Small** | 640x640 | 1 | 6.07 M | 23.21 MB | 40.75 G | 11.58 ms | 18.10 ms | 86.4 FPS |

### Real-World Multi-Task Accuracy [MEASURED]

| Task | Metric | Measured Result | Evaluation Standard |
|---|---|---|---|
| 2D Detection | mAP50 | **32.26 %** | Official COCOeval |
| 2D Detection | mAP50:95 | **13.11 %** | Official COCOeval |
| Semantic Segmentation | mIoU | **61.40 %** | Standard Mean IoU |
| Metric Depth | RMSE | **1.7196 m** | Root Mean Squared Error |

---

## Baseline Comparison Policy

To preserve complete scientific integrity:
* `OFFICIAL_REFERENCE`: Published official baseline statistics (e.g. YOLOv8 nano: 3.2M params, 8.7 GFLOPs, COCO mAP50:95 = 37.3).
* `NOT_PROVEN`: All head-to-head comparisons default strictly to `NOT_PROVEN` until side-by-side training and evaluation on identical real datasets are executed under identical hardware protocols. Neuravex never claims superiority without empirical proof.

---

## Verification Test Suite

Neuravex maintains 100% passing test coverage across 76 automated tests:

```powershell
python -m pytest tests/ -v
# ======================== 76 passed, 4 warnings in ~60s ========================
```

- `tests/test_specialization.py`: 47 tests covering dataset analysis, architecture generation, 3-level router, hardware profiling, distillation, pruning, PTQ, and end-to-end pipeline.
- `tests/test_v01_completion_gate.py`: 10 architectural and numerical completion gates.
- `tests/test_components.py`: Backbone, neck, detection, depth, segmentation units.
- `tests/test_harden_suite.py`: RepConv mathematical fusion ($\max |\Delta| < 10^{-4}$), numerical stability.
- `tests/test_evaluator_metrics.py`: COCOeval integration and known-answer tests.
- `tests/test_metric_tracker_depth.py`: Camera intrinsics, mask-depth fusion, temporal 3D tracking.

---

## Neuravex Hardware-Adaptive Architecture & 20-Subsystem Upgrade

Neuravex has been upgraded into a hardware-adaptive, CPU-first, low-GPU/high-efficiency computer vision stack targeting higher accuracy-per-FLOP, accuracy-per-latency, and lower RAM/VRAM footprint:

### 1. Unified Model Family
- **Neuravex-Pico**: 0.54M params, 0.43 GFLOPs @ 320x320. 54.4 FPS on CPU, 109.9 FPS on RTX 5060, peak VRAM 29.7 MB.
- **Neuravex-Femto**: 1.16M params, 3.73 GFLOPs @ 640x640. 136.8 FPS on RTX 5060.
- **Neuravex-Nano**: 2.02M params, 6.52 GFLOPs @ 640x640. 34.9 FPS on CPU, 138.5 FPS on RTX 5060.
- **Neuravex-Lite**: 4.58M params, 15.11 GFLOPs @ 640x640. 70.2 FPS on RTX 5060.
- **Neuravex-Edge**: 9.08M params, 26.62 GFLOPs @ 640x640. 88.6 FPS on RTX 5060.
- **Neuravex-Pro**: 22.36M params, 64.81 GFLOPs @ 640x640. 64.7 FPS on RTX 5060.
- **Neuravex-Omni**: Unified multimodal foundation stack (2D detection, 3D metric bounding boxes, DEM depth, segmentation, pose, video spatio-temporal tracking).

### 2. Core Upgrades Implemented
1. **CPU-First Core**: Ultra-lightweight variants and INT8 dynamic quantization for high CPU throughput.
2. **Dynamic Resolution Router**: Fast low-res pass (320px) escalating to 512/640px only on ambiguous/small objects.
3. **Dynamic Token Router**: Spatial importance estimation isolating salient foreground features.
4. **Dynamic Layer/Depth Router**: Confidence-based intermediate early exit.
5. **Real Task-Conditional Execution**: Zero-cost detection fast path (`tasks=("det",)`) completely bypassing 3D heads, DEM cross-gating, and auxiliary branches.
6. **Temporal Feature Reuse**: Inter-frame motion-delta gating and backbone feature caching for video streams.
7. **Structural Channel Pruning**: Physical filter and channel slicing with graph reconstruction (reducing real parameter count and FLOPs).
8. **Quantization System**: CPU INT8 PTQ and GPU FP16 with strict cosine-similarity validation (>0.96).
9. **Deployment Backends**: Validated ONNX exporter + ONNX Runtime engine (`CPUExecutionProvider`, `CUDAExecutionProvider`, `TensorrtExecutionProvider`).
10. **Hardware-Aware Controller**: Auto-profiles host CPU, RAM, GPU, VRAM and configures optimal model variant, precision, resolution, and backend.
11. **Hardware-Aware NAS**: Empirical Pareto-frontier evaluation (`pareto_frontier.csv`) under target hardware budgets.
12. **Efficient Backbone & Neck**: Memory-efficient local/global feature mixing.
13. **Object-Centric Refinement**: Global detection first, followed by focused ROI Align and delta refinement on candidate boxes only.
14. **Confidence & Uncertainty Engine**: Temperature-calibrated logits and heteroscedastic bounding box dispersion uncertainty.
15. **Modular Geometry/Depth Path**: Object-aware 3D pinhole unprojection active only when requested.
16. **Memory & State Optimization**: In-place activations, channels-last layout, `torch.inference_mode()`, minimal peak allocations.
17. **Production Profiler**: Measures P50/P95/P99 latency, FPS, real FLOPs, peak RAM, VRAM, and energy/image (mJ), distinguishing warmup from steady-state.
18. **Benchmark Rebuild**: 100% real predictions and real ground-truth annotations; strictly zero synthetic/clamped metrics.
19. **Industry Baselines & Statistics**: Repeated evaluation runs reporting mean and std deviation without bias.
20. **Production Hardening**: Comprehensive test suite (`pytest tests/test_all_20_subsystems.py`) and single end-to-end benchmark command.

### 3. Quickstart & Benchmark Commands

#### Run Comprehensive Subsystem Test Suite:
```powershell
python -m pytest tests/test_all_20_subsystems.py -v
```

#### Run End-to-End Benchmark & Report Generator:
```powershell
python -m neuravex.benchmarks.run_all_benchmarks "benchmark_reports"
```

Generated reports:
- `architecture_diff.md`: Architectural changes and subsystem diff matrix
- `benchmark_protocol.md`: Scientific integrity and measurement protocol
- `efficiency_report.json`: Latency, FPS, FLOPs, RAM, VRAM, and energy across all variants
- `accuracy_report.json`: Real mAP@50, mAP@50:95, IoU, and BoS metrics
- `pareto_frontier.csv`: Non-dominated Pareto frontier candidates
- `hardware_report.json`: Hardware discovery and controller recommendations
- `compatibility_report.md`: Backward compatibility and checkpoint migration audit

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see the [LICENSE](LICENSE) file for details.
