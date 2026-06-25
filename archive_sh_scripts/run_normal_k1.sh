#!/usr/bin/env bash
# Normal (K=1) - CIFAR 100 - 200 epochs

export PYTHONUNBUFFERED=1
PY="/home/e21283/miniconda3/envs/quantum/bin/python"
OUT="./results_cifar100_k1"
COMMON="--dataset cifar100 --n_epochs 200 --n_runs 1 --n_quantum_heads 1 --output_dir $OUT"

echo "Starting Normal (K=1) experiments..."
echo "  - Attention OFF on cuda:0"
echo "  - Attention ON on cuda:1"

# Run concurrently
"$PY" run_experiment.py $COMMON --device cuda:0 > "log_normal_k1_att_off.txt" 2>&1 &
"$PY" run_experiment.py $COMMON --attention_encoding --device cuda:1 > "log_normal_k1_att_on.txt" 2>&1 &

wait
echo "Normal (K=1) experiments completed."
