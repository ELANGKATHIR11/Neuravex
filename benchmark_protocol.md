# Neuravex Production Benchmark Protocol

## 1. Principles & Scientific Integrity Rules
To guarantee reproducible, unmanipulated benchmark comparisons:
1. **Zero Metric Fabrication / Clamping**:
   - Metrics are computed directly from model tensor outputs and real ground-truth bounding box annotations.
   - Clamping, synthetic augmentation of ground-truth boxes, or heuristic Otsu thresholding approximations are strictly forbidden.
2. **Identical Evaluation Settings**:
   - Both Neuravex and baselines are evaluated under identical conditions:
     - Input resolution: 320x320 and 640x640.
     - Batch size: 1 (real-time stream setting).
     - Confidence threshold: 0.25 (standard) / 0.50 (strict).
     - NMS IoU threshold: 0.45.
     - Precision: FP32 on CPU, FP16/FP32 on GPU.
     - Target Hardware: Host CPU and NVIDIA GeForce RTX 5060 Laptop GPU.
3. **Statistical Validity**:
   - All timing benchmarks are performed over multiple repeated runs (minimum 15 steady-state iterations after 5 warmup iterations).
   - Reported metrics include: Mean Latency, P50 (median), P95, P99, Standard Deviation, and Steady-State FPS.
   - Accuracy benchmarks report mean and standard deviation across repeated evaluations.
4. **Honest Baseline Accounting**:
   - If Neuravex falls short of a baseline in a specific operational envelope (e.g. single-task raw CUDA FP16 latency vs Ultralytics YOLO), the gap is reported transparently, and the engineering root cause is documented.

---

## 2. Benchmark Measurement Specifications

### A. Latency & Throughput
- **Warmup Phase**: 5-10 iterations to prime kernel caches, memory allocators, and PyTorch dynamic graphs. All warmup measurements are excluded from timing.
- **CUDA Synchronization**:
  ```python
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)
  start_event.record()
  out = model(input_tensor)
  end_event.record()
  torch.cuda.synchronize()
  latency_ms = start_event.elapsed_time(end_event)
  ```
- **Throughput**: $FPS = \frac{1000.0}{\text{Mean Latency (ms)}}$.

### B. FLOPs & Complexity Accounting
- Analytical counting of multiply-accumulate operations across all Conv2d and Linear layers.
- Parameter count computed via `sum(p.numel() for p in model.parameters())`.
- Model size on disk measured from serialized state dictionaries.

### C. Detection Accuracy (mAP, IoU, BoS)
- Evaluated using standard COCO-compliant evaluation routines (`pycocotools.cocoeval`):
  - **mAP@50**: Average Precision at IoU threshold = 0.50.
  - **mAP@50:95**: Mean Average Precision across 10 IoU thresholds [0.50:0.05:0.95].
  - **Mean IoU**: Intersection over Union between matched True Positive bounding boxes and ground truth.
  - **Boundary Overlap Stability (BoS)**: Multi-metric overlap consistency combining IoU, GIoU, DIoU, and CIoU.

### D. Energy & Power Estimation
- Power is modeled based on device thermal design power (TDP) during active compute:
  - RTX 5060 Laptop GPU: 80W active compute envelope.
  - Host CPU: 35W sustained compute envelope.
- Energy consumed per image:
  $$\text{Energy (mJ)} = \text{Power (Watts)} \times \left(\frac{\text{Mean Latency (ms)}}{1000.0}\right) \times 1000.0 = \text{Power (Watts)} \times \text{Mean Latency (ms)}$$

---

## 3. End-to-End Command Line Reproduction
To execute the complete benchmark suite and regenerate all reports:
```powershell
python -m neuravex.benchmarks.run_all_benchmarks "benchmark_reports"
```
Or execute subsystem unit tests:
```powershell
pytest tests/test_all_20_subsystems.py -v
```
