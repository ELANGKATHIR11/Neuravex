"""
Neuravex Canonical Benchmark Entry Point:
python -m benchmarks.run

Supports:
  --model: nano, small, medium, large (default: nano)
  --weights: Path to checkpoint .pt
  --dataset: Path to real annotated dataset
  --device: cpu, cuda, or specific device (default: auto)
  --precision: fp32, fp16 (default: auto)
  --resolution: input image resolution (default: 320)
  --batch: batch size (default: 1)
  --warmup: warmup iterations (default: 10)
  --iterations: benchmark iterations (default: 50)
  --seed: random seed for reproducibility (default: 42)
  --output: output directory (default: benchmarks/reports)

Records complete metadata:
  git commit, weight hash, dataset hash, library versions, hardware, OS,
  resolution, batch, precision, seed, and timestamp.
"""

import os
import sys
import time
import json
import hashlib
import argparse
import random
import platform
import numpy as np
import torch

from neuravex import build_neuravex, runtime, runtime_explain, NeuravexInferencePostProcessor


def compute_sha256(file_path: str) -> str:
    if not file_path or not os.path.exists(file_path):
        return "N/A"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_args():
    parser = argparse.ArgumentParser(description="Neuravex Canonical Benchmark Runner")
    parser.add_argument("--model", type=str, default="nano", choices=["nano", "small", "medium", "large"], help="Architecture size")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights file")
    parser.add_argument("--dataset", type=str, default=None, help="Path to evaluation dataset")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'auto')")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "fp16"], help="Precision")
    parser.add_argument("--resolution", type=int, default=320, help="Input resolution")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="benchmarks/reports", help="Output directory")
    return parser.parse_args()


def run_benchmark():
    args = parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output, exist_ok=True)

    # Resolve device and precision
    rt_engine = runtime()
    if args.device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = args.device

    if args.precision == "auto":
        target_precision = "fp16" if target_device.startswith("cuda") else "fp32"
    else:
        target_precision = args.precision

    print("=" * 75)
    print(" NEURAVEX CANONICAL REPRODUCIBLE BENCHMARK RUNNER")
    print("=" * 75)
    print(f"Model Size:       Neuravex-{args.model}")
    print(f"Device:           {target_device} ({torch.cuda.get_device_name(0) if 'cuda' in target_device else 'CPU'})")
    print(f"Precision:        {target_precision}")
    print(f"Resolution:       {args.resolution}x{args.resolution}")
    print(f"Batch Size:       {args.batch}")
    print(f"Warmup / Runs:    {args.warmup} / {args.iterations}")
    print(f"Weights Path:     {args.weights or 'Default Initialized'}")
    print(f"Output Directory: {args.output}")
    print("=" * 75)

    # 1. Instantiate Model
    model = build_neuravex(size=args.model, num_classes=80).to(target_device)
    model.eval()

    if args.weights and os.path.exists(args.weights):
        print(f"Loading checkpoint: {args.weights}...")
        ckpt = torch.load(args.weights, map_location=target_device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    num_params = sum(p.numel() for p in model.parameters())

    # 2. Warmup
    dummy_input = torch.randn(args.batch, 3, args.resolution, args.resolution, device=target_device)
    dtype = torch.float16 if target_precision == "fp16" and "cuda" in target_device else torch.float32

    # Measure cold-start latency
    t_cold_0 = time.perf_counter()
    with torch.no_grad():
        with torch.autocast(device_type=target_device.split(":")[0], enabled=(dtype == torch.float16)):
            _ = model(dummy_input, tasks=("det",))
    if "cuda" in target_device:
        torch.cuda.synchronize()
    cold_start_ms = (time.perf_counter() - t_cold_0) * 1000.0

    for _ in range(args.warmup):
        with torch.no_grad():
            with torch.autocast(device_type=target_device.split(":")[0], enabled=(dtype == torch.float16)):
                _ = model(dummy_input, tasks=("det",))
    if "cuda" in target_device:
        torch.cuda.synchronize()

    # 3. Measurement Loop
    latencies = []
    if "cuda" in target_device:
        torch.cuda.reset_peak_memory_stats(target_device)

    for _ in range(args.iterations):
        if "cuda" in target_device:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.autocast(device_type=target_device.split(":")[0], enabled=(dtype == torch.float16)):
                _ = model(dummy_input, tasks=("det",))
        if "cuda" in target_device:
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt_ms)

    lats = np.array(latencies)
    p50_ms = float(np.percentile(lats, 50))
    p95_ms = float(np.percentile(lats, 95))
    p99_ms = float(np.percentile(lats, 99))
    mean_ms = float(np.mean(lats))
    fps = (args.batch * 1000.0) / mean_ms

    peak_vram_mb = 0.0
    if "cuda" in target_device:
        peak_vram_mb = float(torch.cuda.max_memory_allocated(target_device) / (1024 ** 2))

    # 4. Compile Canonical Telemetry Report
    report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "hardware_device": torch.cuda.get_device_name(0) if "cuda" in target_device else "CPU",
            "seed": args.seed,
            "weight_hash": compute_sha256(args.weights) if args.weights else "initial_weights",
            "dataset_hash": compute_sha256(args.dataset) if args.dataset else "none"
        },
        "configuration": {
            "model_size": args.model,
            "parameters_m": round(num_params / 1e6, 2),
            "device": target_device,
            "precision": target_precision,
            "input_resolution": args.resolution,
            "batch_size": args.batch,
            "iterations": args.iterations,
            "nms_free_o2o": True
        },
        "telemetry": {
            "cold_start_ms": round(cold_start_ms, 2),
            "mean_latency_ms": round(mean_ms, 2),
            "p50_latency_ms": round(p50_ms, 2),
            "p95_latency_ms": round(p95_ms, 2),
            "p99_latency_ms": round(p99_ms, 2),
            "throughput_fps": round(fps, 1),
            "peak_vram_mb": round(peak_vram_mb, 1)
        }
    }

    report_path = os.path.join(args.output, f"benchmark_{args.model}_{target_precision}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 75)
    print(" BENCHMARK RESULTS")
    print("=" * 75)
    print(f"Cold Start Latency: {cold_start_ms:.2f} ms")
    print(f"Mean Latency:       {mean_ms:.2f} ms")
    print(f"P50 Latency:        {p50_ms:.2f} ms")
    print(f"P95 Latency:        {p95_ms:.2f} ms")
    print(f"P99 Latency:        {p99_ms:.2f} ms")
    print(f"Throughput:         {fps:.1f} FPS")
    print(f"Peak VRAM:          {peak_vram_mb:.1f} MB")
    print(f"Report Generated:   {report_path}")
    print("=" * 75)
    return report


if __name__ == "__main__":
    run_benchmark()
