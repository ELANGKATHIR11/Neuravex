"""
Neuravex Production Benchmarking Subsystems:
- Hardware-adaptive accurate profiler (P50/P95/P99, FLOPs, RAM, VRAM, Energy)
- Benchmark harness (real data, real annotations, identical settings)
- Industry baseline comparator (Neuravex vs YOLO family with statistical validity)
"""

from .profiler import SystemProfiler, ProfileMetrics
from .benchmark_harness import BenchmarkHarness, BenchmarkResult
from .baseline_comparison import IndustryBaselineComparator

__all__ = [
    "SystemProfiler",
    "ProfileMetrics",
    "BenchmarkHarness",
    "BenchmarkResult",
    "IndustryBaselineComparator"
]
