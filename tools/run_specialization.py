#!/usr/bin/env python3
"""
Neuravex Specialization Pipeline — CLI Entry Point.

Run the full dataset-conditioned + hardware-aware + adaptive-compute
specialization pipeline on any COCO-format dataset.

Usage:
    python run_specialization.py \\
        --ann_file data/custom/annotations.json \\
        --img_dir data/custom/images \\
        --num_classes 20 \\
        --device cuda \\
        --img_size 640 \\
        --latency_budget_ms 30.0 \\
        --param_budget_M 20.0 \\
        --run_ablation \\
        --output_dir results/specialization

For YOLO26 competition gate:
    python run_specialization.py \\
        --ann_file data/custom/annotations.json \\
        --yolo26_dataset_path data/custom/dataset.yaml \\
        --compare_yolo26
"""

import argparse
import json
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Neuravex Specialization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Dataset
    parser.add_argument("--ann_file", required=True, help="COCO annotation JSON file")
    parser.add_argument("--img_dir", default=None, help="Image directory (optional, for texture analysis)")
    parser.add_argument("--num_classes", type=int, default=80)
    parser.add_argument("--max_analysis_images", type=int, default=500)

    # Hardware
    parser.add_argument("--device", default="cpu", help="'cpu', 'cuda', 'cuda:0'")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--latency_budget_ms", type=float, default=50.0)
    parser.add_argument("--vram_budget_mb", type=float, default=2000.0)
    parser.add_argument("--param_budget_M", type=float, default=50.0)
    parser.add_argument("--target_fps", type=float, default=30.0)
    parser.add_argument("--ap_target", type=float, default=0.30)

    # Pipeline options
    parser.add_argument("--run_pruning", action="store_true", help="Run channel pruning")
    parser.add_argument("--run_quantization", action="store_true", help="Run PTQ INT8")
    parser.add_argument("--run_ablation", action="store_true", help="Run ablation experiment")

    # YOLO26 comparison
    parser.add_argument("--compare_yolo26", action="store_true", help="Compare vs YOLO26 baseline")
    parser.add_argument("--yolo26_dataset_path", default=None, help="Dataset path for YOLO26 eval")
    parser.add_argument("--yolo26_variant", default="nano", choices=["nano", "small"])

    # Output
    parser.add_argument("--output_dir", default="results/specialization")
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    from neuravex.specialization import (
        DatasetAnalyzer, ArchitectureGenerator, HardwareConstraints,
        SpecializationPipeline, ExperimentEngine,
        OFFICIAL_REFERENCE, YOLO26BaselineRunner,
    )

    print("\n" + "="*70)
    print("  NEURAVEX SPECIALIZATION PIPELINE")
    print("="*70)
    print(f"  Dataset       : {args.ann_file}")
    print(f"  Device        : {device}")
    print(f"  Image size    : {args.img_size}")
    print(f"  Classes       : {args.num_classes}")
    print(f"  Latency budget: {args.latency_budget_ms}ms")
    print(f"  Param budget  : {args.param_budget_M}M")

    # Step A: Dataset Analysis
    print("\n[1] Analyzing dataset...")
    analyzer = DatasetAnalyzer()
    stats = analyzer.analyze(args.ann_file, args.img_dir, max_images=args.max_analysis_images)
    summary = analyzer.summarize(stats)

    print(f"  Classes: {stats.num_classes}")
    print(f"  Images:  {stats.total_images}")
    print(f"  Annotations: {stats.total_annotations}")
    print(f"  Scale emphasis: {summary['recommended_scale_emphasis']}")
    print(f"  Class imbalance (Gini): {summary['class_imbalance_gini']:.4f}")
    print(f"  Object sizes: small={summary['object_size_fractions']['small (<32^2)']:.2%}, "
          f"medium={summary['object_size_fractions']['medium']:.2%}, "
          f"large={summary['object_size_fractions']['large (>=96^2)']:.2%}")

    # Step B: Architecture Generation
    print("\n[2] Generating specialized architecture...")
    constraints = HardwareConstraints(
        device=device,
        vram_budget_mb=args.vram_budget_mb,
        latency_budget_ms=args.latency_budget_ms,
        param_budget_M=args.param_budget_M,
        precision=args.precision,
        target_fps=args.target_fps,
        ap_target=args.ap_target,
    )
    generator = ArchitectureGenerator(img_size=args.img_size)
    spec = generator.generate(stats.complexity_vector, constraints,
                               num_classes=args.num_classes, verbose=True)

    # Step C: Run Ablation (optional)
    if args.run_ablation:
        print("\n[3] Running ablation experiment...")
        engine = ExperimentEngine(device=device, img_size=args.img_size, verbose=True)
        ablation_results = engine.run_ablation(
            ann_file=args.ann_file,
            img_dir=args.img_dir,
            constraints=constraints,
            num_classes=args.num_classes,
            max_analysis_images=args.max_analysis_images,
        )
        ablation_path = os.path.join(args.output_dir, "ablation_results.json")
        engine.save_results(ablation_results, ablation_path)
        print(f"\n  Ablation results saved to: {ablation_path}")
        pareto = engine.pareto_frontier(ablation_results)
        if pareto:
            print(f"  Pareto-optimal stages: {[r.stage_name for r in pareto]}")

    # Step D: Full Pipeline
    print("\n[4] Running full specialization pipeline...")
    pipeline = SpecializationPipeline(device=device, img_size=args.img_size, verbose=True)
    report = pipeline.run(
        ann_file=args.ann_file,
        img_dir=args.img_dir,
        constraints=constraints,
        num_classes=args.num_classes,
        run_pruning=args.run_pruning,
        run_quantization=args.run_quantization,
        yolo26_dataset_path=args.yolo26_dataset_path if args.compare_yolo26 else None,
    )

    # Step E: YOLO26 Baseline Reference
    print("\n[5] YOLO26 Baseline Reference")
    print(f"  OFFICIAL_REFERENCE (YOLOv8 nano, published):")
    nano_ref = OFFICIAL_REFERENCE["variants"]["nano"]
    print(f"    Params: {nano_ref['params_M']}M | GFLOPs: {nano_ref['gflops_640']} | "
          f"COCO AP50:95: {nano_ref['coco_val_ap50_95']}")
    print(f"  Source: {OFFICIAL_REFERENCE['github']}")
    print(f"  Status: NOT_PROVEN (requires same-dataset evaluation for valid comparison)")
    print(f"\n  NOTE: The existing Neuravex backbone is NOT claimed to be YOLO26.")
    print(f"  Innovations not claimed: {OFFICIAL_REFERENCE['NOT_Neuravex_innovations']}")

    # Save report
    report_path = os.path.join(args.output_dir, "specialization_report.json")
    report.save(report_path)

    print("\n" + "="*70)
    print(f"  PIPELINE STATUS: {report.status}")
    if report.errors:
        print(f"  ERRORS: {report.errors}")
    print(f"  Report saved: {report_path}")
    print("="*70 + "\n")

    return 0 if report.status in ("SUCCESS", "NOT_PROVEN") else 1


if __name__ == "__main__":
    sys.exit(main())
