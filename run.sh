#!/usr/bin/env bash
# run.sh — AQHM-Net experiment launcher
# conda run 25.1.0 crashes on this machine — use the pmqnet python directly.
# Usage:  bash run.sh     (from the Quantum_Images directory)
#
# Epoch guide (METHODOLOGY.md §14.3):
#   --n_epochs 50   : default (CosineAnnealingLR T_max=50, early-stop patience=10)
#   --n_epochs 100  : extended run for small/hard datasets (BreastMNIST, DermaMNIST)

set -e
PYTHON="C:/Users/nisha/miniconda3/envs/pmqnet/python.exe"
OUT="./results"

# ── 0. Debug sanity check ───────────────────────────────────────────────────
echo "====== DEBUG CHECK ======"
$PYTHON run_experiment.py \
    --dataset    mnist \
    --n_epochs   1 \
    --output_dir "$OUT" \
    --debug

# ── 1. MNIST-10 ─────────────────────────────────────────────────────────────
echo "====== MNIST-10 ======"
python run_experiment.py \
    --dataset    mnist \
    --n_epochs   50 \
    --n_runs     10 \
    --output_dir "$OUT"

# ── 2. MNIST-4  (classes 0-3, Wu et al. 2025 comparison) ───────────────────
echo "====== MNIST-4 ======"
python run_experiment.py \
    --dataset      mnist \
    --mnist_classes 0 1 2 3 \
    --n_epochs     50 \
    --n_runs       10 \
    --output_dir   "$OUT"

# ── 3. FashionMNIST-10 ──────────────────────────────────────────────────────
echo "====== FashionMNIST-10 ======"
python run_experiment.py \
    --dataset    fashionmnist \
    --n_epochs   50 \
    --n_runs     10 \
    --output_dir "$OUT"

# ── 4. PathMNIST (9-class RGB histology) ────────────────────────────────────
echo "====== PathMNIST ======"
python run_experiment.py \
    --dataset    pathmnist \
    --n_epochs   50 \
    --n_runs     10 \
    --contrastive \
    --output_dir "$OUT"

# ── 5. DermaMNIST (7-class RGB — hardest, 100 epochs) ───────────────────────
echo "====== DermaMNIST ======"
python run_experiment.py \
    --dataset    dermamnist \
    --n_epochs   100 \
    --n_runs     10 \
    --contrastive \
    --output_dir "$OUT"

# ── 6. PneumoniaMNIST (binary chest X-ray) ──────────────────────────────────
echo "====== PneumoniaMNIST ======"
python run_experiment.py \
    --dataset    pneumoniamnist \
    --n_epochs   50 \
    --n_runs     10 \
    --output_dir "$OUT"

# ── 7. BreastMNIST (binary ultrasound, 546 train — 100 epochs) ──────────────
echo "====== BreastMNIST ======"
python run_experiment.py \
    --dataset    breastmnist \
    --n_epochs   100 \
    --n_runs     10 \
    --output_dir "$OUT"

echo ""
echo "All experiments complete. Results in: $OUT"
