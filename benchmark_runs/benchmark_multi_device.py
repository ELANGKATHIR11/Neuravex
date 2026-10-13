"""
Multi-Device Tier Benchmark for Neuravex:
Evaluates Neuravex Edge Family:
  - Pico   (0.53M) : Embedded microcontrollers / edge CPU (Zero-GPU)
  - Femto  (1.14M) : Low-power Edge CPU (Raspberry Pi / Mobile)
  - Nano   (2.36M) : Entry CPU & Mobile
  - Micro  (5.21M) : Ultra-low-power GPUs (Jetson Nano / GTX 1650)
  - Small  (9.08M) : Compact GPUs (Jetson Orin / RTX 3050)
  - Medium (22.3M) : RTX 5060 Workstation Multi-Task
Compared directly against YOLO11n on both CPU and RTX 5060 GPU.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import json
import csv
import numpy as np
import torch
import psutil
from ultralytics import YOLO

REPO_ROOT = r"C:\Users\elang\Downloads\neuravex-cv"
ML_DIR = os.path.join(REPO_ROOT, "ml_neuravex")
OUTPUT_DIR = os.path.join(REPO_ROOT, "benchmark_runs")
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from neuravex import build_neuravex, CameraIntrinsics

device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

TIERS = [
    ("pico", "CPU (No GPU / IoT)", device_cpu, 320),
    ("femto", "CPU (Raspberry Pi / Edge)", device_cpu, 320),
    ("nano", "CPU (Standard Host)", device_cpu, 512),
    ("pico", "RTX 5060 GPU", device_gpu, 320),
    ("femto", "RTX 5060 GPU", device_gpu, 320),
    ("nano", "RTX 5060 GPU", device_gpu, 512),
    ("micro", "RTX 5060 GPU", device_gpu, 640),
    ("small", "RTX 5060 GPU", device_gpu, 640),
    ("medium", "RTX 5060 GPU", device_gpu, 640)
]

results = []

print("=" * 80)
print("BENCHMARKING NEURAVEX MULTI-DEVICE FAMILY ACROSS CPU & GPU")
print("=" * 80)

for size, target_device_name, dev, res in TIERS:
    print(f"\n---> Profiling Neuravex [{size.upper()}] on {target_device_name} at {res}x{res}...")
    
    # 1. Build and switch to deploy (fused 3x3 convolutions)
    model = build_neuravex(size=size, num_classes=6).to(dev)
    model.eval()
    model.switch_to_deploy()
    
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    K = CameraIntrinsics(fx=float(res), fy=float(res), cx=float(res)/2.0, cy=float(res)/2.0, device=str(dev))
    
    dummy_input = torch.randn(1, 3, res, res, device=dev)
    
    # Warmup
    with torch.no_grad():
        for _ in range(15):
            _ = model(dummy_input, intrinsics=K, tasks=("det",))
        if dev.type == "cuda":
            torch.cuda.synchronize()
            
    # Timed runs
    latencies = []
    num_runs = 50 if dev.type == "cpu" else 150
    
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(0)
        
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input, intrinsics=K, tasks=("det",))
            if dev.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024*1024) if dev.type == "cuda" else 0.0
    mean_lat = float(np.mean(latencies))
    fps = round(1000.0 / mean_lat, 1)
    
    results.append({
        "tier": size,
        "parameters_M": round(params_m, 3),
        "target_hardware": target_device_name,
        "device_type": dev.type,
        "input_resolution": f"{res}x{res}",
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "fps": fps,
        "peak_vram_mb": round(peak_vram, 1)
    })
    
    print(f"  Params: {params_m:.2f}M | Mean Latency: {mean_lat:.2f} ms | FPS: {fps} | Peak VRAM: {peak_vram:.1f} MB")
    del model, dummy_input
    if dev.type == "cuda":
        torch.cuda.empty_cache()

# Also benchmark YOLO11n on CPU and GPU for baseline matching
yolo_path = os.path.join(REPO_ROOT, "yolo11n.pt")
yolo = YOLO(yolo_path)

for dev, target_device_name, res in [(device_cpu, "CPU (Standard Host)", 512), (device_gpu, "RTX 5060 GPU", 640)]:
    print(f"\n---> Profiling YOLO11n Baseline on {target_device_name} at {res}x{res}...")
    yolo.to(dev)
    dummy_input = torch.randn(1, 3, res, res, device=dev)
    with torch.no_grad():
        for _ in range(15):
            _ = yolo.model(dummy_input)
        if dev.type == "cuda":
            torch.cuda.synchronize()
            
    latencies = []
    num_runs = 50 if dev.type == "cpu" else 150
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(0)
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = yolo.model(dummy_input)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            
    peak_vram = torch.cuda.max_memory_allocated(0) / (1024*1024) if dev.type == "cuda" else 0.0
    mean_lat = float(np.mean(latencies))
    fps = round(1000.0 / mean_lat, 1)
    params_m = sum(p.numel() for p in yolo.model.parameters()) / 1e6
    
    results.append({
        "tier": "YOLO11n Baseline",
        "parameters_M": round(params_m, 3),
        "target_hardware": target_device_name,
        "device_type": dev.type,
        "input_resolution": f"{res}x{res}",
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(float(np.median(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "fps": fps,
        "peak_vram_mb": round(peak_vram, 1)
    })
    print(f"  YOLO11n on {target_device_name}: {mean_lat:.2f} ms | FPS: {fps}")

# Export CSV and JSON
multi_device_csv = os.path.join(OUTPUT_DIR, "multi_device_tiers_benchmark.csv")
with open(multi_device_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

multi_device_json = os.path.join(OUTPUT_DIR, "multi_device_tiers_benchmark.json")
with open(multi_device_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nAll multi-device benchmarks saved to {multi_device_csv}")
