import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = r"C:\Users\elang\Downloads\neuravex-cv\benchmark_runs"

# Load speed benchmarks
speed_csv = os.path.join(OUTPUT_DIR, "speed_latency_benchmarks.csv")
speed_records = []
with open(speed_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    speed_records = list(reader)

# Load per-image metrics
per_image_csv = os.path.join(OUTPUT_DIR, "per_image_metrics.csv")
per_image_records = []
with open(per_image_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    per_image_records = list(reader)

# Load master summary
summary_json = os.path.join(OUTPUT_DIR, "master_benchmark_summary.json")
with open(summary_json, "r", encoding="utf-8") as f:
    master_summary = json.load(f)

# Load model complexity
complexity_json = os.path.join(OUTPUT_DIR, "model_complexity.json")
with open(complexity_json, "r", encoding="utf-8") as f:
    complexity = json.load(f)

RESOLUTIONS = [320, 512, 640, 768, 1024]
plt.style.use("dark_background")

# -----------------------------------------------------------
# 01. Latency across Resolutions (FP16 & FP32)
# -----------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

n_fp16 = [float(s["mean_ms"]) for s in speed_records if s["model"] == "Neuravex" and s["precision"] == "fp16"]
y_fp16 = [float(s["mean_ms"]) for s in speed_records if s["model"] == "YOLO26" and s["precision"] == "fp16"]
n_fp32 = [float(s["mean_ms"]) for s in speed_records if s["model"] == "Neuravex" and s["precision"] == "fp32"]
y_fp32 = [float(s["mean_ms"]) for s in speed_records if s["model"] == "YOLO26" and s["precision"] == "fp32"]

ax1.plot(RESOLUTIONS, n_fp16, marker='o', linewidth=2.5, color='#38bdf8', label='Neuravex Multi-Task')
ax1.plot(RESOLUTIONS, y_fp16, marker='s', linewidth=2.5, color='#f43f5e', label='YOLO26 (YOLO11n)')
ax1.set_title("FP16 Latency vs Input Resolution (RTX 5060)", weight='bold')
ax1.set_xlabel("Resolution (Square)")
ax1.set_ylabel("Latency (ms)")
ax1.legend()
ax1.grid(True, alpha=0.25)

ax2.plot(RESOLUTIONS, n_fp32, marker='o', linewidth=2.5, color='#38bdf8', label='Neuravex Multi-Task')
ax2.plot(RESOLUTIONS, y_fp32, marker='s', linewidth=2.5, color='#f43f5e', label='YOLO26 (YOLO11n)')
ax2.set_title("FP32 Latency vs Input Resolution (RTX 5060)", weight='bold')
ax2.set_xlabel("Resolution (Square)")
ax2.set_ylabel("Latency (ms)")
ax2.legend()
ax2.grid(True, alpha=0.25)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "01_latency_vs_resolution.png"))
plt.close(fig)

# -----------------------------------------------------------
# 02. Throughput (FPS) Comparison
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
n_fps = [float(s["fps"]) for s in speed_records if s["model"] == "Neuravex" and s["precision"] == "fp32"]
y_fps = [float(s["fps"]) for s in speed_records if s["model"] == "YOLO26" and s["precision"] == "fp32"]

width = 25
ax.bar(np.array(RESOLUTIONS) - width/2, n_fps, width, label='Neuravex (Multi-Task)', color='#38bdf8', edgecolor='black')
ax.bar(np.array(RESOLUTIONS) + width/2, y_fps, width, label='YOLO26 (YOLO11n 2D)', color='#f43f5e', edgecolor='black')
ax.set_title("RTX 5060 Throughput (FPS) Comparison (FP32)", weight='bold')
ax.set_xlabel("Resolution")
ax.set_ylabel("Frames Per Second (FPS)")
ax.set_xticks(RESOLUTIONS)
ax.legend()
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "02_fps_comparison.png"))
plt.close(fig)

# -----------------------------------------------------------
# 03. Complexity vs FLOPs
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
models = ['YOLO26 (YOLO11n)', 'Neuravex (Medium)']
params = [complexity["yolo26"]["total_params_M"], complexity["neuravex"]["total_params_M"]]
gflops = [complexity["yolo26"]["gflops_640"], complexity["neuravex"]["gflops_640"]]

ax.scatter(params, gflops, s=[250, 450], c=['#f43f5e', '#38bdf8'], edgecolor='white', alpha=0.9)
ax.annotate(f"YOLO26\n({params[0]:.2f}M, {gflops[0]:.1f} GFLOPs)", (params[0] + 0.6, gflops[0] - 1.0), fontsize=10, weight='bold')
ax.annotate(f"Neuravex\n({params[1]:.2f}M, {gflops[1]:.1f} GFLOPs)", (params[1] - 4.5, gflops[1] + 3.0), fontsize=10, weight='bold')

ax.set_title("Model Complexity: Parameters vs. GFLOPs", weight='bold')
ax.set_xlabel("Parameters (Millions)")
ax.set_ylabel("GFLOPs (at 640x640)")
ax.set_xlim(0, 30)
ax.set_ylim(0, 75)
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "03_complexity_flops_params.png"))
plt.close(fig)

# -----------------------------------------------------------
# 04. Latency Distribution across 326 Images
# -----------------------------------------------------------
y_lats = [float(r["yolo_latency_ms"]) for r in per_image_records]
n_lats = [float(r["neuravex_latency_ms"]) for r in per_image_records]

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.hist(y_lats, bins=30, alpha=0.65, color='#f43f5e', label=f'YOLO26 (Mean: {np.mean(y_lats):.2f} ms)', edgecolor='black')
ax.hist(n_lats, bins=30, alpha=0.65, color='#38bdf8', label=f'Neuravex (Mean: {np.mean(n_lats):.2f} ms)', edgecolor='black')
ax.set_title("RTX 5060 End-to-End Latency Distribution Across 326 Dataset Images", weight='bold')
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Number of Images")
ax.legend()
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "04_latency_distribution.png"))
plt.close(fig)

# -----------------------------------------------------------
# 05. Peak VRAM Footprint
# -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
vram_labels = ['YOLO26 (YOLO11n)', 'Neuravex (Multi-Task)']
vram_vals = [master_summary["models"]["yolo26"]["peak_vram_mb"], master_summary["models"]["neuravex"]["peak_vram_mb"]]

bars = ax.bar(vram_labels, vram_vals, color=['#f43f5e', '#38bdf8'], width=0.45, edgecolor='black')
ax.set_title("Dedicated GPU VRAM Footprint on RTX 5060", weight='bold')
ax.set_ylabel("Peak Allocated VRAM (MB)")
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{yval:.1f} MB", ha='center', va='bottom', weight='bold')
ax.set_ylim(0, 350)
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "05_vram_footprint.png"))
plt.close(fig)

# -----------------------------------------------------------
# 06. Geometric IoU & BoS Box Quality Distribution
# -----------------------------------------------------------
ious = [float(r["iou"]) for r in per_image_records if float(r["iou"]) > 0]
boss = [float(r["bos"]) for r in per_image_records if float(r["bos"]) > 0]

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
ax.hist(ious, bins=25, alpha=0.7, color='#38bdf8', label=f'IoU (Mean: {np.mean(ious):.4f})', edgecolor='black')
ax.hist(boss, bins=25, alpha=0.7, color='#fbbf24', label=f'BoS (Mean: {np.mean(boss):.4f})', edgecolor='black')
ax.set_title("YOLO26 Geometric Bounding Box Quality vs Salient Contour", weight='bold')
ax.set_xlabel("Score [0.0 - 1.0]")
ax.set_ylabel("Detections")
ax.legend()
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "06_iou_bos_distribution.png"))
plt.close(fig)

print("All benchmark charts successfully generated and exported.")
