#!/usr/bin/env bash
# CIFAR-10 / CIFAR-100 parallel-quantum-circuit K-ablation.
# 300 epochs, early-stopping patience 20, single seed (n_runs=1).
# K = 1, 2, 4, 8 parallel quantum circuits over patch groups.
# Output dirs tagged _kK (K>1); CIFAR-10 K=1 -> ./results_cifar_sweep/cifar10 etc.

PY="C:/Users/nisha/miniconda3/envs/pmqnet/python.exe"
OUT="./results_cifar_sweep"
COMMON="--n_runs 1 --n_epochs 300 --patience 20 --output_dir $OUT"

run () {  # $1 = label, rest = args
  label="$1"; shift
  echo "=================================================================="
  echo ">>> $label  ::  $*  ($(date '+%H:%M:%S'))"
  echo "=================================================================="
  "$PY" run_experiment.py "$@" $COMMON || echo "!!! FAILED: $label"
}

for K in 1 2 4 8; do
  run "CIFAR-10  K=$K"  --dataset cifar10  --n_quantum_heads "$K"
done

for K in 1 2 4 8; do
  run "CIFAR-100 K=$K"  --dataset cifar100 --n_quantum_heads "$K"
done

echo "ALL DONE  ($(date '+%H:%M:%S'))"
