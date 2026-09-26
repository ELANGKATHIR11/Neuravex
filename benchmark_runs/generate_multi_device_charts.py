import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = r"C:\Users\elang\Downloads\neuravex-cv\benchmark_runs"

# Load multi-device benchmark data
multi_device_csv = os.path.join(OUTPUT_DIR, "multi_device_tiers_benchmark.csv")
with open(multi_device_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    records = list(reader)

plt.style.use("dark_background")

# 1. CPU Latency & FPS: Neuravex Edge Tiers vs YOLO11n
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

cpu_tiers = [r for r in records if r["device_type"] == "cpu"]
labels = [f"{r['tier'].upper()}\n({r['parameters_M']}M)" for r in cpu_tiers]
latencies = [float(r["mean_latency_ms"]) for r in cpu_tiers]
fps_vals = [float(r["fps"]) for r in cpu_tiers]
colors = ['#38bdf8', '#34d399', '#a78bfa', '#f43f5e']

bars1 = ax1.bar(labels, latencies, color=colors, width=0.5, edgecolor='black')
ax1.set_title("CPU Latency (Zero-GPU Embedded / Host)", weight='bold', fontsize=12)
ax1.set_ylabel("Mean Latency (ms)", fontsize=11)
ax1.grid(True, alpha=0.25)
for bar in bars1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + 0.8, f"{y:.1f} ms", ha='center', va='bottom', weight='bold')

bars2 = ax2.bar(labels, fps_vals, color=colors, width=0.5, edgecolor='black')
ax2.set_title("CPU Throughput (FPS)", weight='bold', fontsize=12)
ax2.set_ylabel("Frames Per Second (FPS)", fontsize=11)
ax2.grid(True, alpha=0.25)
for bar in bars2:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, y + 1.2, f"{y:.1f} FPS", ha='center', va='bottom', weight='bold')

plt.suptitle("Neuravex Edge Family vs. YOLO11n on CPU (No GPU Required)", fontsize=14, weight='bold')
plt.tight_layout()
chart_cpu_path = os.path.join(OUTPUT_DIR, "07_cpu_edge_tiers_comparison.png")
fig.savefig(chart_cpu_path)
plt.close(fig)

# 2. Complete Family Scalability on GPU: Parameter Count vs FPS
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

gpu_tiers = [r for r in records if r["device_type"] == "cuda"]
x_params = [float(r["parameters_M"]) for r in gpu_tiers]
y_fps = [float(r["fps"]) for r in gpu_tiers]
labels_gpu = [r["tier"].upper() for r in gpu_tiers]

ax.scatter(x_params, y_fps, s=250, c=['#38bdf8', '#34d399', '#a78bfa', '#fbbf24', '#f97316', '#06b6d4', '#f43f5e'], edgecolor='white')

for i, txt in enumerate(labels_gpu):
    ax.annotate(f"{txt}\n({x_params[i]:.2f}M, {y_fps[i]:.1f} FPS)", (x_params[i] + 0.5, y_fps[i] - 1.2), fontsize=9, weight='bold')

ax.set_title("Neuravex Model Scalability: Parameter Footprint vs. GPU Throughput (RTX 5060)", fontsize=13, weight='bold')
ax.set_xlabel("Parameters (Millions)", fontsize=11)
ax.set_ylabel("Inference Throughput (FPS)", fontsize=11)
ax.grid(True, alpha=0.25)
plt.tight_layout()
chart_gpu_path = os.path.join(OUTPUT_DIR, "08_gpu_family_scalability.png")
fig.savefig(chart_gpu_path)
plt.close(fig)

print("Generated new multi-device comparison charts: 07 & 08.")
