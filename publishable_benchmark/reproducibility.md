# Reproducibility & Verification Guide
======================================

## Exact Reproduction Commands
```bash
# 1. Activate conda environment
conda activate dgpu-core

# 2. Run the complete SOTA Optimization, Distillation, and Benchmark Pipeline
cd c:/Users/elang/Downloads/neuravex-cv/ml_neuravex
python -u publishable_benchmark/run_sota_optimization.py
```

## Environment Manifest
- **Platform:** Windows 10/11 x86_64
- **Python:** 3.11.15
- **PyTorch:** 2.14.0
- **CUDA Available:** True (NVIDIA GeForce RTX 5060 Laptop GPU)
- **Deterministic Seed:** 42
- **Dataset Split:** Stratified 70% Train (228), 15% Val (49), 15% Held-out Test (49)
- **Dataset Hash Audit:** Recorded in `dataset_manifest.json`
