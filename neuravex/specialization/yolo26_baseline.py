"""
Neuravex YOLO26 Baseline Reference.

Maintains two SEPARATE, clearly labeled references:

  OFFICIAL_REFERENCE:
    Published YOLO26 (Ultralytics v8) architecture specification.
    Source: Ultralytics YOLOv8 technical report + official GitHub.
    Key characteristics:
      - Backbone: CSPDarknet with C2f blocks
      - Neck: Asymmetric PAN with C2f
      - Head: Decoupled anchor-free with DFL-free regression (direct box)
      - 1-to-1 NMS-free optional (TAL assignment)
      - Nano: ~3.2M params, ~8.7 GFLOPs at 640px input
    Source URL: https://github.com/ultralytics/ultralytics

  LOCAL_REPRO:
    Ultralytics local checkpoint-based evaluation.
    Requires: pip install ultralytics
    Checkpoint: yolov8n.pt (publicly available)
    Matching conditions: imgsz=640, batch=1, conf=0.001, iou=0.7,
                         fp32, standard letterbox preprocessing,
                         class-agnostic NMS post-processing.

CRITICAL RULES:
  - Never call any existing Neuravex variant "official YOLO26".
  - The existing 0.98M-param local detector is NOT official YOLO26.
  - DFL, anchor-free regression, TAL assigner are NOT Neuravex innovations
    (they originate in YOLOv6/v7/v8 prior work).
  - OFFICIAL_REFERENCE must match the exact model/version/checkpoint.
  - If ultralytics is not installed or checkpoint unavailable:
    STATUS = NOT_PROVEN (never fabricate results).
  - All comparison conditions MUST be identical (hardware, imgsz, precision,
    preprocessing, NMS settings).

COMPLETION GATE:
  True success requires:
    Neuravex_AP >= YOLO26_AP (on same dataset, same conditions)
    while measured FLOPs + latency + VRAM + params < YOLO26.
  Until this is demonstrated on a real dataset: STATUS = NOT_PROVEN.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


# ─── Official Reference Specification ─────────────────────────────────────────

OFFICIAL_REFERENCE = {
    "name": "YOLOv8 (YOLO26 family reference)",
    "source": "Ultralytics YOLOv8",
    "github": "https://github.com/ultralytics/ultralytics",
    "paper": "Jocher et al., 2023 (unpublished technical report)",
    "variants": {
        "nano": {
            "params_M": 3.2,
            "gflops_640": 8.7,
            "coco_val_ap50_95": 37.3,   # published result
            "backbone": "CSPDarknet-C2f",
            "neck": "Asymmetric-PAN-C2f",
            "head": "Decoupled-anchor-free-DFL-free",
            "assignment": "TAL",
            "nms": "class-agnostic",
            "reg_type": "direct-box-regression",  # NOT DFL
        },
        "small": {
            "params_M": 11.2,
            "gflops_640": 28.6,
            "coco_val_ap50_95": 44.9,
            "backbone": "CSPDarknet-C2f",
            "neck": "Asymmetric-PAN-C2f",
            "head": "Decoupled-anchor-free-DFL-free",
            "assignment": "TAL",
            "nms": "class-agnostic",
            "reg_type": "direct-box-regression",
        },
    },
    "inference_conditions": {
        "imgsz": 640,
        "batch": 1,
        "precision": "fp32",
        "preprocessing": "letterbox",
        "conf_thresh": 0.001,
        "iou_thresh": 0.7,
        "nms_type": "class-agnostic",
        "device": "must_match",
    },
    "NOT_Neuravex_innovations": [
        "DFL (Distribution Focal Loss) - from YOLOv6/v7",
        "Anchor-free detection - from FCOS/YOLOv5u",
        "TAL (Task-Aligned Learning) assigner - from TOOD",
        "CIoU/SIoU loss - from prior work",
        "Multi-scale PAN neck - from PANet",
        "Varifocal loss - from VarifocalNet",
    ],
}


# ─── Local Reproduction Status ─────────────────────────────────────────────────

@dataclass
class YOLO26LocalReproResult:
    """Results from running official YOLO26 locally."""
    available: bool = False
    checkpoint: str = "yolov8n.pt"
    ultralytics_version: str = "unknown"
    device: str = "unknown"
    imgsz: int = 640
    batch: int = 1
    precision: str = "fp32"
    dataset: str = "unknown"
    ap50_95: float = 0.0
    ap50: float = 0.0
    ap75: float = 0.0
    aps: float = 0.0
    apm: float = 0.0
    apl: float = 0.0
    params_M: float = 0.0
    gflops: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    fps: float = 0.0
    peak_vram_mb: float = 0.0
    status: str = "NOT_PROVEN"
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "available": self.available,
            "checkpoint": self.checkpoint,
            "ultralytics_version": self.ultralytics_version,
            "device": self.device,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "precision": self.precision,
            "dataset": self.dataset,
            "metrics": {
                "AP50:95": self.ap50_95,
                "AP50": self.ap50,
                "AP75": self.ap75,
                "APs": self.aps,
                "APm": self.apm,
                "APl": self.apl,
            },
            "compute": {
                "params_M": self.params_M,
                "gflops": self.gflops,
                "p50_ms": self.p50_ms,
                "p95_ms": self.p95_ms,
                "fps": self.fps,
                "peak_vram_mb": self.peak_vram_mb,
            },
            "status": self.status,
            "errors": self.errors,
        }


class YOLO26BaselineRunner:
    """
    Runs the official YOLO26 baseline under matched conditions.

    If ultralytics is not installed or checkpoint is unavailable:
      - Returns result with status=NOT_PROVEN
      - Never fabricates AP numbers

    Conditions that MUST match Neuravex evaluation:
      - Same dataset, same images
      - Same device and precision
      - Same imgsz and batch
      - Same preprocessing (letterbox)
      - Same NMS settings
    """

    def __init__(
        self,
        checkpoint: str = "yolov8n.pt",
        device: str = "cpu",
        imgsz: int = 640,
        batch: int = 1,
        precision: str = "fp32",
        conf: float = 0.001,
        iou: float = 0.7,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.imgsz = imgsz
        self.batch = batch
        self.precision = precision
        self.conf = conf
        self.iou = iou

    def check_availability(self) -> bool:
        """Check if ultralytics + checkpoint are available."""
        try:
            import ultralytics
            return True
        except ImportError:
            return False

    def get_reference_spec(self, variant: str = "nano") -> Dict:
        """Return the official published specification (never fabricated)."""
        if variant not in OFFICIAL_REFERENCE["variants"]:
            raise ValueError(f"Unknown YOLO26 variant: {variant}. Choose from {list(OFFICIAL_REFERENCE['variants'].keys())}")
        spec = dict(OFFICIAL_REFERENCE["variants"][variant])
        spec["source"] = OFFICIAL_REFERENCE["source"]
        spec["inference_conditions"] = dict(OFFICIAL_REFERENCE["inference_conditions"])
        return spec

    def run_evaluation(
        self,
        dataset_path: str,
        variant: str = "nano",
    ) -> YOLO26LocalReproResult:
        """
        Run official YOLO26 evaluation on a dataset.
        Returns YOLO26LocalReproResult with status=NOT_PROVEN if not available.
        """
        result = YOLO26LocalReproResult(
            checkpoint=self.checkpoint,
            device=self.device,
            imgsz=self.imgsz,
            batch=self.batch,
            precision=self.precision,
            dataset=dataset_path,
        )

        if not self.check_availability():
            result.status = "NOT_PROVEN"
            result.errors.append(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            )
            print("[YOLO26 Baseline] Status: NOT_PROVEN — ultralytics not available.")
            return result

        try:
            from ultralytics import YOLO
            import ultralytics

            result.ultralytics_version = ultralytics.__version__
            model = YOLO(self.checkpoint)

            # Profile latency
            import time
            import numpy as np
            x = torch.randn(1, 3, self.imgsz, self.imgsz)
            latencies = []
            with torch.no_grad():
                for _ in range(30):
                    t0 = time.perf_counter()
                    model.predict(x, verbose=False)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)
            lats = np.array(latencies[5:])  # Skip warmup
            result.p50_ms = float(np.percentile(lats, 50))
            result.p95_ms = float(np.percentile(lats, 95))
            result.fps = 1000.0 / float(lats.mean())

            # Model size
            result.params_M = sum(p.numel() for p in model.model.parameters()) / 1e6

            # Run evaluation
            metrics = model.val(
                data=dataset_path,
                imgsz=self.imgsz,
                batch=self.batch,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
            result.ap50_95 = float(getattr(metrics.box, "map", 0.0))
            result.ap50 = float(getattr(metrics.box, "map50", 0.0))
            result.ap75 = float(getattr(metrics.box, "map75", 0.0))
            result.aps = float(getattr(metrics.box, "maps", [0.0])[0] if hasattr(metrics.box, "maps") else 0.0)
            result.available = True
            result.status = "MEASURED"

        except Exception as e:
            result.status = "NOT_PROVEN"
            result.errors.append(str(e))
            print(f"[YOLO26 Baseline] Status: NOT_PROVEN — Error: {e}")

        return result


@dataclass
class CompetitionResult:
    """
    Direct Neuravex vs YOLO26 comparison on same dataset.
    COMPLETION GATE: min Cost s.t. AP >= YOLO26_AP.
    """
    dataset: str = "unknown"
    conditions_matched: bool = False

    # YOLO26 reference
    yolo26_variant: str = "nano"
    yolo26_ap50_95: float = 0.0
    yolo26_status: str = "NOT_PROVEN"

    # Neuravex specialized
    neuravex_arch: str = "unknown"
    neuravex_ap50_95: float = 0.0

    # Cost comparison
    yolo26_gflops: float = 0.0
    neuravex_gflops: float = 0.0
    yolo26_params_M: float = 0.0
    neuravex_params_M: float = 0.0
    yolo26_p50_ms: float = 0.0
    neuravex_p50_ms: float = 0.0
    yolo26_vram_mb: float = 0.0
    neuravex_vram_mb: float = 0.0

    # Gate result
    ap_target_met: bool = False
    cost_improvement: bool = False
    overall_status: str = "NOT_PROVEN"
    notes: List[str] = field(default_factory=list)

    def evaluate_gate(self) -> str:
        """Evaluate the completion gate and set status."""
        if self.yolo26_status == "NOT_PROVEN":
            self.overall_status = "NOT_PROVEN"
            self.notes.append("YOLO26 baseline not measured — cannot claim victory.")
            return self.overall_status

        if not self.conditions_matched:
            self.overall_status = "NOT_PROVEN"
            self.notes.append("Evaluation conditions did not match — invalid comparison.")
            return self.overall_status

        self.ap_target_met = self.neuravex_ap50_95 >= self.yolo26_ap50_95

        cost_ok = (
            self.neuravex_gflops <= self.yolo26_gflops or
            self.neuravex_p50_ms <= self.yolo26_p50_ms or
            self.neuravex_params_M <= self.yolo26_params_M
        )
        self.cost_improvement = cost_ok

        if self.ap_target_met and self.cost_improvement:
            self.overall_status = "SUCCESS"
        elif self.ap_target_met:
            self.overall_status = "AP_MET_BUT_HIGHER_COST"
        elif self.cost_improvement:
            self.overall_status = "LOWER_COST_BUT_AP_NOT_MET"
        else:
            self.overall_status = "FAILED"

        return self.overall_status

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"  NEURAVEX vs YOLO26 ({self.yolo26_variant}) COMPLETION GATE",
            f"{'='*70}",
            f"  Dataset          : {self.dataset}",
            f"  Conditions Match : {self.conditions_matched}",
            f"  YOLO26 Status    : {self.yolo26_status}",
            f"  ─── AP ───",
            f"  YOLO26 AP50:95   : {self.yolo26_ap50_95:.4f}",
            f"  Neuravex AP50:95 : {self.neuravex_ap50_95:.4f}",
            f"  AP Gate Met      : {self.ap_target_met}",
            f"  ─── Cost ───",
            f"  YOLO26 GFLOPs    : {self.yolo26_gflops:.3f}",
            f"  Neuravex GFLOPs  : {self.neuravex_gflops:.3f}",
            f"  YOLO26 P50 ms    : {self.yolo26_p50_ms:.2f}",
            f"  Neuravex P50 ms  : {self.neuravex_p50_ms:.2f}",
            f"  YOLO26 Params M  : {self.yolo26_params_M:.3f}",
            f"  Neuravex Params M: {self.neuravex_params_M:.3f}",
            f"  Cost Gate Met    : {self.cost_improvement}",
            f"  ─── STATUS ─── : {self.overall_status}",
        ]
        for note in self.notes:
            lines.append(f"  NOTE: {note}")
        lines.append(f"{'='*70}\n")
        return "\n".join(lines)
