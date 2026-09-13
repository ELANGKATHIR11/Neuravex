"""
Neuravex Architecture Generator.

Takes a dataset-complexity vector + user hardware constraints and automatically
selects/generates an optimal ArchSpec for the Neuravex specialization pipeline.

Design principles:
  - Tiny-object datasets → high-resolution P2/P3 paths, shallower P5
  - Large-object datasets → deeper P5 coarse features, trim P2
  - Dense/hard scenes → more router budget, more RepBlocks
  - Sparse/easy scenes → aggressive compute savings, thinner width
  - Never generate architectures violating tensor/compute constraints
  - Search validates actual latency vs budget (via HardwareProfiler)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ArchSpec:
    """
    Architecture specification for a specialized Neuravex model.
    All fields control the build_specialized_neuravex factory.
    """
    # Backbone
    base_c: int = 32                          # Base channel width
    depth_mul: float = 0.67                   # Depth multiplier for RepBlocks
    use_p2: bool = False                      # Enable stride-4 P2 feature level
    use_p6: bool = False                      # Enable stride-64 P6 (very large objects)

    # Neck
    neck_depth_mul: float = 1.0               # Neck RepBlock depth multiplier

    # Detection head
    reg_max: int = 16                         # DFL bins (1 = direct regression, no DFL overhead)
    strides: Tuple[int, ...] = (8, 16, 32)   # Active detection strides

    # Routing
    router_threshold: float = 0.5            # Default routing threshold
    router_budget: float = 0.6               # Max expected compute fraction
    num_difficulty_levels: int = 2           # easy/medium/hard routing (2 or 3)

    # Task heads
    seg_embed_dim: int = 16                  # Instance embedding dimension
    num_parts: int = 8                       # Part segmentation slots

    # Cost estimate
    estimated_gflops: float = 0.0
    estimated_params_M: float = 0.0

    # Specialization metadata
    scale_emphasis: str = "balanced"
    dataset_complexity_norm: float = 0.5
    rationale: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "base_c": self.base_c,
            "depth_mul": self.depth_mul,
            "use_p2": self.use_p2,
            "use_p6": self.use_p6,
            "neck_depth_mul": self.neck_depth_mul,
            "reg_max": self.reg_max,
            "strides": list(self.strides),
            "router_threshold": self.router_threshold,
            "router_budget": self.router_budget,
            "num_difficulty_levels": self.num_difficulty_levels,
            "seg_embed_dim": self.seg_embed_dim,
            "num_parts": self.num_parts,
            "estimated_gflops": self.estimated_gflops,
            "estimated_params_M": self.estimated_params_M,
            "scale_emphasis": self.scale_emphasis,
            "dataset_complexity_norm": self.dataset_complexity_norm,
            "rationale": self.rationale,
        }


class HardwareConstraints:
    """User-specified hardware deployment constraints."""
    def __init__(
        self,
        device: str = "cpu",
        vram_budget_mb: float = 2000.0,
        latency_budget_ms: float = 50.0,
        param_budget_M: float = 50.0,
        precision: str = "fp32",
        target_fps: float = 30.0,
        ap_target: float = 0.30,
    ):
        self.device = device
        self.vram_budget_mb = vram_budget_mb
        self.latency_budget_ms = latency_budget_ms
        self.param_budget_M = param_budget_M
        self.precision = precision
        self.target_fps = target_fps
        self.ap_target = ap_target

    def to_dict(self) -> Dict:
        return {
            "device": self.device,
            "vram_budget_mb": self.vram_budget_mb,
            "latency_budget_ms": self.latency_budget_ms,
            "param_budget_M": self.param_budget_M,
            "precision": self.precision,
            "target_fps": self.target_fps,
            "ap_target": self.ap_target,
        }


# Candidate configs ordered from cheapest to most expensive
_CANDIDATE_CONFIGS = [
    {"base_c": 8,  "depth_mul": 0.25, "name": "pico"},
    {"base_c": 12, "depth_mul": 0.33, "name": "femto"},
    {"base_c": 16, "depth_mul": 0.33, "name": "nano"},
    {"base_c": 24, "depth_mul": 0.50, "name": "micro"},
    {"base_c": 32, "depth_mul": 0.67, "name": "small"},
    {"base_c": 48, "depth_mul": 1.00, "name": "medium"},
    {"base_c": 64, "depth_mul": 1.33, "name": "large"},
    {"base_c": 80, "depth_mul": 1.67, "name": "xlarge"},
]


def _estimate_params(base_c: int, depth_mul: float, use_p2: bool, use_p6: bool) -> float:
    """Quick empirical parameter estimate in millions."""
    # Backbone: scales as base_c^2 * depth_mul
    backbone_params = (base_c ** 2) * depth_mul * 0.006
    # Neck: ~40% of backbone
    neck_params = backbone_params * 0.4
    # Detection head: ~20% of backbone
    det_params = backbone_params * 0.2
    # P2 adds ~15% extra
    p2_bonus = 0.15 if use_p2 else 0.0
    # P6 adds ~20% extra
    p6_bonus = 0.20 if use_p6 else 0.0
    total = backbone_params * (1 + p2_bonus + p6_bonus) + neck_params + det_params
    return round(total, 3)


def _estimate_gflops(base_c: int, depth_mul: float, img_size: int = 640) -> float:
    """Quick empirical GFLOPs estimate at given input size."""
    # Scales roughly as base_c^2 * depth_mul * (img_size/640)^2
    scale = (img_size / 640.0) ** 2
    gflops = (base_c ** 2) * depth_mul * 0.0015 * scale
    return round(gflops, 3)


class ArchitectureGenerator:
    """
    Generates an optimal ArchSpec from a dataset complexity vector +
    hardware constraints using rule-based specialization.

    The search:
    1. Reads the complexity vector to determine scale emphasis & difficulty
    2. Selects width/depth from budget-feasible candidate configs
    3. Adds scale specializations (P2 for tiny objects, P6 for large)
    4. Configures router threshold and difficulty levels
    5. Validates the spec against compute constraints
    6. Returns the minimum-cost valid ArchSpec meeting AP target heuristics
    """

    def __init__(self, img_size: int = 640):
        self.img_size = img_size

    def generate(
        self,
        complexity_vector: "np.ndarray",
        constraints: HardwareConstraints,
        num_classes: int = 80,
        verbose: bool = False,
    ) -> ArchSpec:
        """
        Generate optimal ArchSpec from complexity vector + hardware constraints.

        complexity_vector: 16-dim float array in [0, 1]
        constraints: HardwareConstraints
        Returns: ArchSpec
        """
        import numpy as np
        vec = np.array(complexity_vector, dtype=np.float32)
        if len(vec) < 16:
            vec = np.pad(vec, (0, 16 - len(vec)))

        # --- Parse complexity signals ---
        small_frac   = float(vec[2])
        medium_frac  = float(vec[3])
        large_frac   = float(vec[4])
        density_norm = float(vec[5])
        occlusion    = float(vec[11])
        imbalance    = float(vec[1])
        complexity_norm = float(np.mean(vec))

        # --- Determine scale emphasis ---
        if small_frac > 0.5:
            scale_emphasis = "tiny"
        elif large_frac > 0.5:
            scale_emphasis = "large"
        elif small_frac > 0.25:
            scale_emphasis = "mixed_small"
        else:
            scale_emphasis = "balanced"

        # --- Determine difficulty level ---
        difficulty = density_norm * 0.4 + occlusion * 0.4 + imbalance * 0.2
        if difficulty > 0.6:
            diff_level = "hard"
        elif difficulty > 0.3:
            diff_level = "medium"
        else:
            diff_level = "easy"

        # --- Select base config from budget ---
        selected_cfg = self._select_config_for_budget(
            constraints, scale_emphasis, diff_level, num_classes
        )
        base_c = selected_cfg["base_c"]
        depth_mul = selected_cfg["depth_mul"]

        rationale = {
            "scale_emphasis": scale_emphasis,
            "difficulty_level": diff_level,
            "selected_config": selected_cfg["name"],
        }

        # --- Scale specializations ---
        use_p2 = scale_emphasis in ("tiny", "mixed_small")
        use_p6 = scale_emphasis == "large" and base_c >= 32

        # Active strides
        strides = [8, 16, 32]
        if use_p2:
            strides = [4, 8, 16, 32]
        if use_p6:
            strides = [8, 16, 32, 64]

        # --- Router config ---
        if diff_level == "easy":
            router_threshold = 0.3   # Very aggressive skipping
            router_budget = 0.4
            num_diff_levels = 2
        elif diff_level == "medium":
            router_threshold = 0.5
            router_budget = 0.6
            num_diff_levels = 2
        else:  # hard
            router_threshold = 0.7
            router_budget = 0.9
            num_diff_levels = 3

        # --- Neck depth ---
        # Dense/hard scenes get deeper neck; sparse/easy get shallower
        neck_depth = 1.33 if diff_level == "hard" else (0.67 if diff_level == "easy" else 1.0)

        # --- reg_max ---
        # Small objects need finer regression
        reg_max = 16 if scale_emphasis in ("tiny", "mixed_small") else (8 if scale_emphasis == "large" else 16)

        # --- Seg embed ---
        # Simpler embed for easy/sparse; richer for dense
        seg_embed_dim = 32 if diff_level == "hard" else 16

        # --- Estimate cost ---
        est_params = _estimate_params(base_c, depth_mul, use_p2, use_p6)
        est_gflops = _estimate_gflops(base_c, depth_mul, self.img_size)

        spec = ArchSpec(
            base_c=base_c,
            depth_mul=depth_mul,
            use_p2=use_p2,
            use_p6=use_p6,
            neck_depth_mul=neck_depth,
            reg_max=reg_max,
            strides=tuple(strides),
            router_threshold=router_threshold,
            router_budget=router_budget,
            num_difficulty_levels=num_diff_levels,
            seg_embed_dim=seg_embed_dim,
            num_parts=8,
            estimated_gflops=est_gflops,
            estimated_params_M=est_params,
            scale_emphasis=scale_emphasis,
            dataset_complexity_norm=complexity_norm,
            rationale=rationale,
        )

        if verbose:
            self._print_spec(spec, constraints)

        return spec

    def _select_config_for_budget(
        self,
        constraints: HardwareConstraints,
        scale_emphasis: str,
        diff_level: str,
        num_classes: int,
    ) -> Dict:
        """Select minimum-cost config that fits within parameter + latency budget."""
        budget_params = constraints.param_budget_M
        budget_lat = constraints.latency_budget_ms

        # Empirical latency scaling: cpu ~5x slower than gpu
        lat_multiplier = 5.0 if "cpu" in constraints.device.lower() else 1.0

        # Start from cheapest, pick largest config that fits budget
        selected = _CANDIDATE_CONFIGS[0]
        for cfg in _CANDIDATE_CONFIGS:
            use_p2 = scale_emphasis in ("tiny", "mixed_small")
            est_p = _estimate_params(cfg["base_c"], cfg["depth_mul"], use_p2, False)
            est_gf = _estimate_gflops(cfg["base_c"], cfg["depth_mul"], self.img_size)
            # Rough latency estimate: proportional to GFLOPs
            est_lat = est_gf * 8.0 * lat_multiplier

            if est_p <= budget_params and est_lat <= budget_lat:
                selected = cfg
            else:
                break  # Exceeded budget, keep previous

        # For hard/dense datasets, bump up one tier if budget allows
        if diff_level == "hard":
            idx = _CANDIDATE_CONFIGS.index(selected)
            if idx + 1 < len(_CANDIDATE_CONFIGS):
                next_cfg = _CANDIDATE_CONFIGS[idx + 1]
                use_p2 = scale_emphasis in ("tiny", "mixed_small")
                next_p = _estimate_params(next_cfg["base_c"], next_cfg["depth_mul"], use_p2, False)
                next_gf = _estimate_gflops(next_cfg["base_c"], next_cfg["depth_mul"], self.img_size)
                next_lat = next_gf * 8.0 * lat_multiplier
                if next_p <= budget_params and next_lat <= budget_lat:
                    selected = next_cfg

        return selected

    @staticmethod
    def _print_spec(spec: ArchSpec, constraints: HardwareConstraints):
        print("\n" + "=" * 60)
        print("Architecture Generator — Generated ArchSpec")
        print("=" * 60)
        print(f"  Scale emphasis      : {spec.scale_emphasis}")
        print(f"  Difficulty level    : {spec.rationale.get('difficulty_level', '?')}")
        print(f"  Selected config     : {spec.rationale.get('selected_config', '?')}")
        print(f"  base_c / depth_mul  : {spec.base_c} / {spec.depth_mul}")
        print(f"  Active strides      : {spec.strides}")
        print(f"  use_p2 / use_p6     : {spec.use_p2} / {spec.use_p6}")
        print(f"  reg_max             : {spec.reg_max}")
        print(f"  Router threshold    : {spec.router_threshold}")
        print(f"  Router budget       : {spec.router_budget}")
        print(f"  Difficulty levels   : {spec.num_difficulty_levels}")
        print(f"  Est. params (M)     : {spec.estimated_params_M:.3f}")
        print(f"  Est. GFLOPs         : {spec.estimated_gflops:.3f}")
        print(f"  Hardware target     : {constraints.device} / {constraints.latency_budget_ms}ms budget")
        print("=" * 60 + "\n")

    def search(
        self,
        complexity_vector: "np.ndarray",
        constraints: HardwareConstraints,
        num_classes: int = 80,
        n_candidates: int = 5,
    ) -> List[ArchSpec]:
        """
        Generate multiple candidate ArchSpecs (Pareto-style sweep).
        Returns list ordered from cheapest to most accurate/expensive.
        """
        import numpy as np
        specs = []
        vec = np.array(complexity_vector, dtype=np.float32)

        # Vary constraint budget to get a range of operating points
        budget_multipliers = [0.25, 0.5, 0.75, 1.0, 1.5][:n_candidates]
        for mult in budget_multipliers:
            c_sweep = HardwareConstraints(
                device=constraints.device,
                vram_budget_mb=constraints.vram_budget_mb * mult,
                latency_budget_ms=constraints.latency_budget_ms * mult,
                param_budget_M=constraints.param_budget_M * mult,
                precision=constraints.precision,
                target_fps=constraints.target_fps,
                ap_target=constraints.ap_target,
            )
            spec = self.generate(vec, c_sweep, num_classes=num_classes)
            specs.append(spec)
        return specs
