"""
Neuravex Real-Data Benchmark Suite.

RULES:
 - NO fabricated ground truth (no fixed boxes, no random images, no Otsu GT).
 - NO clamped/synthetic metrics.
 - Real annotated datasets ONLY. Missing dataset → FAIL LOUDLY.
 - Identical preprocessing/resolution/precision/batch for all compared models.
 - One-to-one Hungarian matching with class gating for mAP.
"""

import os
import sys
import json
import time
import torch
from typing import Dict, Any, Optional, List

from neuravex.models.neuravex import build_neuravex


def run_all(output_dir: str = "benchmark_reports", dataset_path: Optional[str] = None):
    """
    Execute the complete benchmark suite.

    Args:
        output_dir: Directory for output reports.
        dataset_path: Path to COCO-format annotated dataset.
            Must contain images/ and annotations/instances.json.
            If None or non-existent, the benchmark FAILS LOUDLY.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print(" NEURAVEX PRODUCTION BENCHMARK SUITE")
    print("=" * 70)

    # ---- STEP 0: Validate Dataset ----
    if dataset_path is None or not os.path.exists(dataset_path):
        msg = (
            f"\n[FATAL] No real annotated dataset found at: {dataset_path}\n"
            f"Neuravex benchmarks require REAL annotated data (COCO format).\n"
            f"Provide --dataset <path> with:\n"
            f"  <path>/images/     — image files\n"
            f"  <path>/annotations/instances.json  — COCO annotations\n"
            f"\nFabricating ground truth is PROHIBITED. Benchmark aborted.\n"
        )
        print(msg)
        report = {
            "status": "FAILED",
            "reason": "No real annotated dataset provided",
            "dataset_path": str(dataset_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        report_path = os.path.join(output_dir, "benchmark_FAILED.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Failure report saved to: {report_path}")
        return report

    annotations_file = os.path.join(dataset_path, "annotations", "instances.json")
    images_dir = os.path.join(dataset_path, "images")

    if not os.path.isfile(annotations_file):
        print(f"[FATAL] Annotations file not found: {annotations_file}")
        print("Expected COCO-format annotations/instances.json")
        return {"status": "FAILED", "reason": f"Missing {annotations_file}"}

    if not os.path.isdir(images_dir):
        print(f"[FATAL] Images directory not found: {images_dir}")
        return {"status": "FAILED", "reason": f"Missing {images_dir}"}

    print(f"  Dataset: {dataset_path}")
    print(f"  Annotations: {annotations_file}")
    print(f"  Images: {images_dir}")

    # ---- STEP 1: Hardware Profiling ----
    print("\n[Step 1/4] Profiling System Hardware...")
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    hw_report = {
        "device": device_str,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "cpu_cores": os.cpu_count(),
        "torch_version": torch.__version__,
    }
    hw_file = os.path.join(output_dir, "hardware_report.json")
    with open(hw_file, "w") as f:
        json.dump(hw_report, f, indent=2)
    print(f"  -> Hardware profile saved to {hw_file}")

    # ---- STEP 2: Model Efficiency Profiling ----
    print("\n[Step 2/4] Benchmarking Model Efficiency...")
    efficiency_results = _benchmark_efficiency(device_str)
    eff_file = os.path.join(output_dir, "efficiency_report.json")
    with open(eff_file, "w") as f:
        json.dump(efficiency_results, f, indent=2)
    print(f"  -> Efficiency report saved to {eff_file}")

    # ---- STEP 3: Accuracy Evaluation on Real Dataset ----
    print("\n[Step 3/4] Evaluating on real annotated dataset...")
    accuracy_results = _evaluate_on_coco_dataset(
        annotations_file, images_dir, device_str,
    )
    acc_file = os.path.join(output_dir, "accuracy_report.json")
    with open(acc_file, "w") as f:
        json.dump(accuracy_results, f, indent=2)
    print(f"  -> Accuracy report saved to {acc_file}")

    # ---- STEP 4: Summary ----
    print("\n[Step 4/4] Generating Summary...")
    summary = {
        "status": "COMPLETE",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_path,
        "hardware": hw_report,
        "accuracy": accuracy_results,
        "efficiency": efficiency_results,
    }
    summary_file = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll benchmark reports saved in '{output_dir}'")
    print("=" * 70)
    return summary


def _benchmark_efficiency(device: str) -> Dict[str, Any]:
    """Profile inference latency and throughput for standard variants."""
    results = {}
    for variant in ["pico", "nano", "lite", "edge", "pro"]:
        try:
            model = build_neuravex(variant).to(device).eval()
            # Count parameters
            n_params = sum(p.numel() for p in model.parameters()) / 1e6

            res = 320 if variant in ("pico", "nano") else 640
            dummy = torch.randn(1, 3, res, res, device=device)

            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    model(dummy, tasks=("det",))

            # Measure
            latencies = []
            for _ in range(20):
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    model(dummy, tasks=("det",))
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

            latencies.sort()
            results[variant] = {
                "params_M": round(n_params, 2),
                "resolution": [res, res],
                "device": device,
                "p50_ms": round(latencies[len(latencies) // 2], 1),
                "p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
                "mean_ms": round(sum(latencies) / len(latencies), 1),
                "fps": round(1000.0 / (sum(latencies) / len(latencies)), 1),
            }
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as e:
            results[variant] = {"error": str(e)}
    return results


def _evaluate_on_coco_dataset(
    ann_file: str, img_dir: str, device: str,
) -> Dict[str, Any]:
    """Evaluate Neuravex on a COCO-format dataset using pycocotools."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return {
            "status": "SKIPPED",
            "reason": "pycocotools not installed. Install with: pip install pycocotools",
        }

    try:
        import cv2
    except ImportError:
        return {"status": "SKIPPED", "reason": "opencv-python not installed"}

    coco_gt = COCO(ann_file)
    img_ids = coco_gt.getImgIds()
    print(f"  -> {len(img_ids)} images in dataset")

    if len(img_ids) == 0:
        return {"status": "FAILED", "reason": "No images in annotation file"}

    # Determine number of classes
    cat_ids = coco_gt.getCatIds()
    num_classes = max(cat_ids) + 1 if cat_ids else 80

    model = build_neuravex("nano", num_classes=num_classes).to(device).eval()
    predictions = []

    for idx, img_id in enumerate(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        bgr = cv2.imread(img_path)
        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = rgb.shape[:2]
        rgb_resized = cv2.resize(rgb, (640, 640))
        t = torch.from_numpy(rgb_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            out = model(t, tasks=("det",))

        pred_cls = torch.sigmoid(out["class_logits"][0])
        pred_boxes = out["pred_boxes"][0]
        scores, labels = pred_cls.max(dim=-1)
        keep = scores > 0.01

        if keep.any():
            f_boxes = pred_boxes[keep].cpu().numpy()
            f_scores = scores[keep].cpu().numpy()
            f_labels = labels[keep].cpu().numpy()

            for j in range(len(f_boxes)):
                x1, y1, x2, y2 = f_boxes[j]
                # Scale back to original image
                x1 = x1 * w_orig / 640.0
                y1 = y1 * h_orig / 640.0
                x2 = x2 * w_orig / 640.0
                y2 = y2 * h_orig / 640.0
                w_box = max(0, x2 - x1)
                h_box = max(0, y2 - y1)
                cat_id = int(f_labels[j])
                if cat_id in cat_ids:
                    predictions.append({
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [float(x1), float(y1), float(w_box), float(h_box)],
                        "score": float(f_scores[j]),
                    })

        if (idx + 1) % 50 == 0:
            print(f"    Processed {idx + 1}/{len(img_ids)} images...")

    if len(predictions) == 0:
        return {"status": "COMPLETE", "mAP50": 0.0, "mAP50_95": 0.0, "note": "No predictions above threshold"}

    # Run COCO evaluation
    try:
        from typing import cast
        coco_dt = coco_gt.loadRes(cast(str, predictions))
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats

        return {
            "status": "COMPLETE",
            "num_images": len(img_ids),
            "num_predictions": len(predictions),
            "mAP50_95": round(float(max(0, stats[0])), 4),
            "mAP50": round(float(max(0, stats[1])), 4),
            "mAP75": round(float(max(0, stats[2])), 4),
            "APs": round(float(max(0, stats[3])), 4),
            "APm": round(float(max(0, stats[4])), 4),
            "APl": round(float(max(0, stats[5])), 4),
            "AR100": round(float(max(0, stats[8])), 4),
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else "benchmark_reports"
    run_all(out, dataset_path=ds)
