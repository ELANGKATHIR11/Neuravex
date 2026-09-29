"""
Production CLI & Module for High-Precision Multi-Modal Perception:
Can be called on any input image or folder to produce:
- Animal Census / Count
- Tight, perspective-accurate 3D Bounding Wireframe Cubes
- Exact Instance Segmentations
- Dense CameraAwareDEM Metric Depth Map & Elevation
- High-contrast, non-overlapping HUD measurement badges
- Structured JSON telemetry export

Usage:
  python analyse_user_image.py --image <path_to_image> --output <path_to_output_vis>
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import json
from neuravex.engine.precision_perception import PrecisionPerceptionPipeline

def main():
    parser = argparse.ArgumentParser(description="Neuravex Precision 3D & DEM Image Perception")
    parser.add_argument("--image", type=str, default=r"C:\Users\elang\.gemini\antigravity-ide\brain\151118fd-e01f-44a3-b131-a0080004762a\.user_uploaded\media_1790523153559.jpg", help="Path to input image")
    parser.add_argument("--output", type=str, default=r"C:\Users\elang\.gemini\antigravity-ide\brain\151118fd-e01f-44a3-b131-a0080004762a\precise_langur_3d_segmentation.png", help="Path to output visualization")
    parser.add_argument("--json", type=str, default=r"C:\Users\elang\.gemini\antigravity-ide\brain\151118fd-e01f-44a3-b131-a0080004762a\precise_langur_data.json", help="Path to output JSON")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")

    args = parser.parse_args()

    pipeline = PrecisionPerceptionPipeline(conf_thresh=args.conf)
    results = pipeline.analyze(args.image, output_vis_path=args.output, output_json_path=args.json)

    print("\n" + "=" * 60)
    print(" NEURAVEX PRECISION 3D & DEM PERCEPTION COMPLETED")
    print("=" * 60)
    print(f"Total Animals Localized: {results['total_animals_counted']}")
    print(f"Visualization Saved: {results['visualization_path']}")
    for obj in results["detected_objects"]:
        c = obj["coordinates_3d"]
        d = obj["dimensions_3d"]
        print(f"  [{obj['id']}] {obj['label']} | Conf: {obj['confidence']:.2f} | Dist: {obj['distance_from_camera_m']:.2f}m")
        print(f"      3D Coords (X, Y, Z): ({c['X_lateral_m']:+.2f}, {c['Y_vertical_m']:+.2f}, {c['Z_depth_m']:.2f}) m")
        print(f"      Dimensions (L x B x H): {d['length_L_m']:.2f}m x {d['breadth_B_m']:.2f}m x {d['height_H_m']:.2f}m")

if __name__ == "__main__":
    main()
