import os
from pathlib import Path
from PIL import Image
from collections import Counter

dataset_root = Path(r"c:\Users\elang\Downloads\neuravex-cv\Dataset")

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
annotation_extensions = {".xml", ".txt", ".json", ".csv"}

print(f"================================================================================")
print(f"DATASET INVENTORY & FORENSICS: {dataset_root}")
print(f"================================================================================\n")

total_images_all = 0
total_bytes_all = 0
folder_summaries = []

for root, dirs, files in os.walk(dataset_root):
    r_path = Path(root)
    rel_path = r_path.relative_to(dataset_root)
    
    img_files = [f for f in files if Path(f).suffix.lower() in image_extensions]
    ann_files = [f for f in files if Path(f).suffix.lower() in annotation_extensions]
    other_files = [f for f in files if Path(f).suffix.lower() not in image_extensions and Path(f).suffix.lower() not in annotation_extensions]
    
    if len(files) > 0:
        total_images_all += len(img_files)
        total_size = sum((r_path / f).stat().st_size for f in files)
        total_bytes_all += total_size
        
        # sample dimensions and aspect ratios
        resolutions = []
        formats = Counter()
        for f in img_files:
            formats[Path(f).suffix.lower()] += 1
            if len(resolutions) < 5:
                try:
                    with Image.open(r_path / f) as im:
                        resolutions.append(im.size)
                except Exception:
                    pass
        
        display_name = str(rel_path) if str(rel_path) != "." else "[ROOT]"
        folder_summaries.append({
            "folder": display_name,
            "total_files": len(files),
            "images": len(img_files),
            "formats": dict(formats),
            "annotations": len(ann_files),
            "other": len(other_files),
            "size_mb": total_size / (1024 * 1024),
            "sample_res": resolutions
        })

print(f"Found {len(folder_summaries)} directories containing data.\n")
for item in folder_summaries:
    print(f"[DIR] Folder: {item['folder']}")
    print(f"   - Total Files: {item['total_files']} | Images: {item['images']} | Annotations: {item['annotations']} | Other: {item['other']}")
    print(f"   - Disk Size: {item['size_mb']:.2f} MB")
    if item['formats']:
        print(f"   - Image Formats: {item['formats']}")
    if item['sample_res']:
        print(f"   - Sample Resolutions (W x H): {item['sample_res']}")
    print("-" * 70)

print(f"\n================================================================================")
print(f"OVERALL SUMMARY:")
print(f"Total Image Files: {total_images_all}")
print(f"Total Dataset Size: {total_bytes_all / (1024 * 1024):.2f} MB ({total_bytes_all / (1024 * 1024 * 1024):.2f} GB)")
print(f"================================================================================")
