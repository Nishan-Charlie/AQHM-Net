#!/usr/bin/env bash
# Single-seed (n_runs=1) first-signal pass for the parallel-quantum-circuits
# direction. All datasets at K=4; CIFAR also at K=1 as an immediate baseline.
# Ordered fast -> slow so early results land first. Each run is independent
# (one failure does not abort the rest). Multi-seed + full K-ablation comes
# later via --n_quantum_heads {1,2,4,8} --n_runs 10.

PY="C:/Users/nisha/miniconda3/envs/pmqnet/python.exe"
OUT="./results_parallel"
COMMON="--n_runs 1 --n_epochs 50 --output_dir $OUT"

run () {  # $1 = human label, rest = args
  label="$1"; shift
  echo "=================================================================="
  echo ">>> $label  ::  $*"
  echo "=================================================================="
  "$PY" run_experiment.py "$@" $COMMON || echo "!!! FAILED: $label"
}

# Fast medical / small first
run "BreastMNIST  K=4" --dataset breastmnist   --use_focal_loss --use_balanced_sampler --n_quantum_heads 4
run "Pneumonia    K=4" --dataset pneumoniamnist --n_quantum_heads 4
run "DermaMNIST   K=4" --dataset dermamnist    --use_focal_loss --use_balanced_sampler --n_quantum_heads 4
run "MNIST-4      K=4" --dataset mnist --mnist_classes 0 1 2 3 --n_quantum_heads 4

# CIFAR: baseline (K=1) and parallel (K=4) for a direct comparison
run "CIFAR-10  K=1" --dataset cifar10  --n_quantum_heads 1
run "CIFAR-10  K=4" --dataset cifar10  --n_quantum_heads 4
run "CIFAR-100 K=1" --dataset cifar100 --n_quantum_heads 1
run "CIFAR-100 K=4" --dataset cifar100 --n_quantum_heads 4

# Largest last
run "MNIST-10  K=4" --dataset mnist --n_quantum_heads 4
run "PathMNIST K=4" --dataset pathmnist --contrastive --n_quantum_heads 4

echo "ALL DONE"
