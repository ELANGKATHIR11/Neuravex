import os
import sys
import glob
import hashlib
import json
import csv
import cv2
import numpy as np

DATASET_ROOT = r"C:\Users\elang\Downloads\animals"
OUTPUT_DIR = r"C:\Users\elang\Downloads\neuravex-cv\benchmark_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Asiatic Lion", "Indian Cow", "Indian Dog", "Indian Macaque", "Langur", "tiger"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

manifest_records = []
corrupted_files = []
hashes = {}
duplicate_records = []

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

for root, dirs, files in os.walk(DATASET_ROOT):
    for f in sorted(files):
        ext = os.path.splitext(f)[1].lower()
        if ext in image_extensions:
            file_path = os.path.join(root, f)
            parent_dir = os.path.basename(root)
            
            # Determine class
            matched_cls = None
            for c in CLASS_NAMES:
                if c.lower() == parent_dir.lower():
                    matched_cls = c
                    break
            
            # Check file hash
            with open(file_path, 'rb') as fp:
                file_bytes = fp.read()
                file_hash = hashlib.sha256(file_bytes).hexdigest()
                file_size = len(file_bytes)
            
            if file_hash in hashes:
                duplicate_records.append((file_path, hashes[file_hash]))
            else:
                hashes[file_hash] = file_path
                
            # Verify image integrity and dimensions
            img = cv2.imread(file_path)
            if img is None:
                corrupted_files.append(file_path)
                continue
                
            h, w, ch = img.shape
            aspect_ratio = float(w) / float(h)
            
            manifest_records.append({
                "file_path": file_path,
                "relative_path": os.path.relpath(file_path, DATASET_ROOT),
                "file_name": f,
                "class_name": matched_cls if matched_cls else "Unknown",
                "class_id": CLASS_TO_ID.get(matched_cls, -1),
                "width": w,
                "height": h,
                "channels": ch,
                "aspect_ratio": round(aspect_ratio, 4),
                "file_size_bytes": file_size,
                "sha256": file_hash,
                "has_gt_bbox": False,
                "has_gt_mask": False,
                "has_gt_depth": False
            })

# Save CSV manifest
manifest_csv_path = os.path.join(OUTPUT_DIR, "dataset_manifest.csv")
with open(manifest_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(manifest_records[0].keys()))
    writer.writeheader()
    writer.writerows(manifest_records)

# Summary Forensics
class_counts = {}
for r in manifest_records:
    c = r["class_name"]
    class_counts[c] = class_counts.get(c, 0) + 1

resolutions = [(r["width"], r["height"]) for r in manifest_records]
widths = [r["width"] for r in manifest_records]
heights = [r["height"] for r in manifest_records]
ratios = [r["aspect_ratio"] for r in manifest_records]

summary = {
    "dataset_root": DATASET_ROOT,
    "total_images_discovered": len(manifest_records),
    "total_corrupted_images": len(corrupted_files),
    "total_duplicates": len(duplicate_records),
    "class_counts": class_counts,
    "class_imbalance_ratio": max(class_counts.values()) / min(class_counts.values()),
    "min_width": min(widths),
    "max_width": max(widths),
    "mean_width": float(np.mean(widths)),
    "min_height": min(heights),
    "max_height": max(heights),
    "mean_height": float(np.mean(heights)),
    "mean_aspect_ratio": float(np.mean(ratios)),
    "has_manual_gt_annotations": False,
    "annotation_audit_note": "No ground truth annotations (YOLO/COCO/VOC) exist in dataset. Images are organized strictly in classification directories."
}

summary_json_path = os.path.join(OUTPUT_DIR, "dataset_forensics.json")
with open(summary_json_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print(f"Dataset Forensics Complete: {len(manifest_records)} valid images.")
print(f"Class Counts: {class_counts}")
print(f"Duplicate images detected: {len(duplicate_records)}")
print(f"Corrupted files: {len(corrupted_files)}")
