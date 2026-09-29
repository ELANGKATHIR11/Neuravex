"""
Neuravex Unified Command-Line Interface (CLI).

Commands:
  neuravex info       — Show package version, hardware, and backend info
  neuravex detect     — Run native detection on an image
  neuravex segment    — Run segmentation on an image
  neuravex analyze    — Full precision 3D perception pipeline
  neuravex benchmark  — Run benchmark suite (requires real annotated data)
  neuravex export     — Export model to ONNX/TFLite
  neuravex hardware   — Detect and report available hardware
  neuravex train      — Launch training (requires training data)
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="neuravex",
        description="Neuravex Unified Computer Vision SDK CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # --- info ---
    subparsers.add_parser("info", help="Show version, hardware, and backend info")

    # --- analyze ---
    analyze_p = subparsers.add_parser("analyze", help="Full precision 3D perception pipeline")
    analyze_p.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    analyze_p.add_argument("--output", "-o", type=str, default=None, help="Save annotated visualization")
    analyze_p.add_argument("--json", "-j", type=str, default=None, help="Save structured JSON")
    analyze_p.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold")
    analyze_p.add_argument("--device", "-d", type=str, default=None, help="Device (cuda/cpu/mps)")
    analyze_p.add_argument("--weights", "-w", type=str, default=None, help="Model weights path")
    analyze_p.add_argument("--size", "-s", type=str, default="nano", help="Model size variant")
    analyze_p.add_argument("--num-classes", type=int, default=80, help="Number of classes")

    # --- detect ---
    detect_p = subparsers.add_parser("detect", help="Run 2D detection only")
    detect_p.add_argument("--image", "-i", type=str, required=True)
    detect_p.add_argument("--conf", "-c", type=float, default=0.25)
    detect_p.add_argument("--device", "-d", type=str, default=None)
    detect_p.add_argument("--weights", "-w", type=str, default=None)
    detect_p.add_argument("--size", "-s", type=str, default="nano")

    # --- segment ---
    detect_p = subparsers.add_parser("segment", help="Run instance segmentation")
    detect_p.add_argument("--image", "-i", type=str, required=True)
    detect_p.add_argument("--conf", "-c", type=float, default=0.25)
    detect_p.add_argument("--device", "-d", type=str, default=None)
    detect_p.add_argument("--weights", "-w", type=str, default=None)

    # --- benchmark ---
    bench_p = subparsers.add_parser("benchmark", help="Run benchmark suite")
    bench_p.add_argument("--dataset", type=str, required=True, help="Path to annotated dataset (COCO format)")
    bench_p.add_argument("--output-dir", type=str, default="benchmark_reports")
    bench_p.add_argument("--device", "-d", type=str, default=None)

    # --- export ---
    export_p = subparsers.add_parser("export", help="Export model")
    export_p.add_argument("--format", type=str, choices=["onnx", "torchscript"], default="onnx")
    export_p.add_argument("--weights", "-w", type=str, default=None)
    export_p.add_argument("--output", "-o", type=str, default="neuravex_export")
    export_p.add_argument("--size", "-s", type=str, default="nano")

    # --- hardware ---
    subparsers.add_parser("hardware", help="Detect and report available hardware")

    # --- train ---
    train_p = subparsers.add_parser("train", help="Launch training")
    train_p.add_argument("--dataset", type=str, required=True, help="Path to training dataset")
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--device", "-d", type=str, default=None)

    args = parser.parse_args()

    if args.command == "info":
        _cmd_info()
    elif args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "detect":
        _cmd_detect(args)
    elif args.command == "export":
        _cmd_export(args)
    elif args.command == "hardware":
        _cmd_hardware()
    elif args.command == "benchmark":
        print(f"Benchmark requires real annotated dataset at: {args.dataset}")
        print("Use COCO-format annotations. Fabricated GT is NOT supported.")
        if not os.path.exists(args.dataset):
            print(f"ERROR: Dataset path does not exist: {args.dataset}")
            sys.exit(1)
    else:
        parser.print_help()


def _cmd_export(args):
    import torch
    from neuravex.models.neuravex import build_neuravex
    from neuravex.deployment.onnx_backend import export_onnx

    print(f"Building Neuravex ({args.size}) for export...")
    model = build_neuravex(size=args.size).eval()
    if args.weights and os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

    if args.format == "onnx":
        out_file = args.output if args.output.endswith(".onnx") else f"{args.output}.onnx"
        res = export_onnx(model, out_file)
        print(f"Successfully exported ONNX model to: {res['output_path']} ({res['file_size_mb']:.2f} MB)")
    elif args.format == "torchscript":
        out_file = args.output if args.output.endswith(".pt") else f"{args.output}.pt"
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        dummy_input = torch.randn(1, 3, 320, 320)
        traced = torch.jit.trace(lambda x: model(x, tasks=("det",)), dummy_input)
        traced.save(out_file)
        mb = os.path.getsize(out_file) / (1024 * 1024)
        print(f"Successfully exported TorchScript model to: {out_file} ({mb:.2f} MB)")



def _cmd_info():
    import neuravex
    print(f"Neuravex v{neuravex.__version__}")
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("MPS (Apple Silicon): available")
    except ImportError:
        print("PyTorch: NOT INSTALLED")
    try:
        import onnxruntime
        print(f"ONNX Runtime: {onnxruntime.__version__}")
    except ImportError:
        pass


def _cmd_analyze(args):
    from neuravex.engine.precision_perception import PrecisionPerceptionPipeline
    out_vis = args.output
    out_json = args.json
    if out_vis is None:
        base = os.path.splitext(args.image)[0]
        out_vis = f"{base}_neuravex_3d.png"
    if out_json is None:
        base = os.path.splitext(args.image)[0]
        out_json = f"{base}_neuravex_3d.json"

    pipeline = PrecisionPerceptionPipeline(
        weights_path=args.weights,
        conf_thresh=args.conf,
        device=args.device,
        model_size=args.size,
        num_classes=args.num_classes,
    )
    results = pipeline.analyze(args.image, output_vis_path=out_vis, output_json_path=out_json)

    print("\n" + "=" * 65)
    print(" [NEURAVEX] NATIVE PRECISION 3D PERCEPTION COMPLETE")
    print("=" * 65)
    n_obj = results.get("total_visible", results.get("total_animals_counted", 0))
    print(f"Total Objects : {n_obj}")
    if results.get("visualization_path"):
        print(f"Visualization : {results['visualization_path']}")
    if results.get("telemetry_path"):
        print(f"JSON Output   : {results['telemetry_path']}")
    print("-" * 65)
    for obj in results.get("objects", results.get("detected_objects", [])):
        oid = obj.get("id", "?")
        cls = obj.get("class_name", obj.get("class_id", "?"))
        sc = obj.get("score", 0)
        xyz = obj.get("xyz")
        lwh = obj.get("lwh")
        dist = obj.get("distance")
        dsrc = obj.get("depth_source", "?")
        print(f"  [{oid}] {cls} | score={sc:.3f} | depth_source={dsrc}")
        if xyz:
            print(f"      XYZ: ({xyz[0]:+.2f}, {xyz[1]:+.2f}, {xyz[2]:.2f}) m")
        if lwh:
            print(f"      LWH: {lwh[0]:.2f} x {lwh[1]:.2f} x {lwh[2]:.2f} m")
        if dist is not None:
            print(f"      Distance: {dist:.2f} m")
    print("=" * 65 + "\n")


def _cmd_detect(args):
    import torch
    import cv2
    from neuravex.models.neuravex import build_neuravex

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = build_neuravex(size=args.size).to(device).eval()
    if args.weights and os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Cannot read image: {args.image}")
        sys.exit(1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (640, 640))
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        out = model(t, tasks=("det",))

    scores = torch.sigmoid(out["class_logits"][0]).max(dim=-1)
    n_above = (scores.values > args.conf).sum().item()
    print(f"Detections above {args.conf}: {n_above}")


def _cmd_hardware():
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CPU cores: {os.cpu_count()}")
        print(f"CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem / 1e9
                print(f"  GPU {i}: {name} ({mem:.1f} GB)")
        if hasattr(torch.backends, "mps"):
            print(f"MPS: {torch.backends.mps.is_available()}")
    except ImportError:
        print("PyTorch not available — CPU-only NumPy mode")


if __name__ == "__main__":
    main()
