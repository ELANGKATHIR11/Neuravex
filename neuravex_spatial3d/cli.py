"""
neuravex-spatial3d Command-Line Interface (CLI).
Global terminal entrypoints:
  neuravex-spatial3d info             - Inspect hardware, CUDA, cuDNN, Intel/AMD AVX accelerators
  neuravex-spatial3d analyze          - Run native 3D spatial perception on image/photo
  neuravex-spatial3d live             - Run real-time live camera/stream 3D perception dashboard
  neuravex-spatial3d dem              - Generate DEM terrain elevation, slope & aspect from depth map
"""

import sys
import argparse
import json
import cv2
import neuravex_spatial3d as sp3d

def main():
    parser = argparse.ArgumentParser(
        prog="neuravex-spatial3d",
        description="Neuravex Native 3D Spatial Perception & Video Analysis CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- info ---
    subparsers.add_parser("info", help="Inspect device context, CUDA, cuDNN, and AVX acceleration")

    # --- analyze (image/photo) ---
    analyze_p = subparsers.add_parser("analyze", help="Analyze image or photo for 3D boxes, DEM, and LWH dimensions")
    analyze_p.add_argument("--image", "-i", type=str, required=True, help="Path to input photo or image")
    analyze_p.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold")
    analyze_p.add_argument("--output-json", "-o", type=str, default=None, help="Save structured JSON analysis")

    # --- live (video/stream) ---
    live_p = subparsers.add_parser("live", help="Launch live video analysis stream (webcam, RTSP, or video file)")
    live_p.add_argument("--source", "-s", type=str, default="0", help="Video source (0 for webcam, file path, or RTSP URL)")
    live_p.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold")
    live_p.add_argument("--output", "-o", type=str, default=None, help="Save annotated output video")
    live_p.add_argument("--no-gui", action="store_true", help="Run headless without cv2.imshow window")

    args = parser.parse_args()

    if args.command == "info":
        ctx = sp3d.DeviceContext()
        print("=" * 60)
        print("Neuravex Spatial3D Hardware & Execution Context:")
        print(f"  Device:         {ctx.backend['device']}")
        print(f"  Device Name:    {ctx.backend['device_name']}")
        print(f"  Platform:       {ctx.backend['platform']}")
        print(f"  CUDA Available: {ctx.backend['cuda_available']}")
        print(f"  cuDNN Enabled:  {ctx.backend['cudnn_enabled']} (v{ctx.backend['cudnn_version']})")
        print(f"  Memory (GB):    {ctx.backend['memory_gb']} GB")
        print("=" * 60)

    elif args.command == "analyze":
        analyzer = sp3d.NativeSpatialAnalyzer()
        res = analyzer.analyze_image(args.image, conf_threshold=args.conf)
        print("=" * 60)
        print(f"Image Analysis Complete for: {args.image}")
        print(f"  Dimensions:   {res['image_size'][0]} x {res['image_size'][1]}")
        print(f"  Latency:      {res['latency_ms']} ms ({1000.0/max(res['latency_ms'], 1e-3):.1f} FPS)")
        print(f"  Detections:   {res['num_detections']} objects")
        print(f"  DEM Elev:     {res['dem_metrics']['mean_elevation_m']} m")
        for i, b in enumerate(res['boxes_3d']):
            print(f"    [{i+1}] {b['class_name']} (score={b['score']:.2f}) | XYZ: {b['xyz']} m | LWH: {b['lwh']} m | Vol: {b['volume_m3']} m3")
        print("=" * 60)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Saved JSON report to: {args.output_json}")

    elif args.command == "live":
        source = int(args.source) if args.source.isdigit() else args.source
        pipeline = sp3d.LiveVideoPipeline()
        print(f"Starting Neuravex Native Live 3D Stream on source: {source}...")
        try:
            for result, annotated_frame in pipeline.stream_live(video_source=source, conf_threshold=args.conf):
                fps = result["fps"]
                boxes = result["boxes_3d"]
                sys.stdout.write(f"\r[Live 3D] FPS: {fps:5.1f} | Tracked Objects: {len(boxes):2d} | DEM Mean Elev: {result['dem_metrics']['mean_elevation_m']:.2f} m")
                sys.stdout.flush()

                if not args.no_gui and annotated_frame is not None:
                    cv2.imshow("Neuravex Native Live 3D Stream", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            print("\nStream finished.")
        finally:
            cv2.destroyAllWindows()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
