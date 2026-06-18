#!/usr/bin/env bash
# Medium (K=4) - CIFAR 10 - 200 epochs

export PYTHONUNBUFFERED=1
PY="/home/e21283/miniconda3/envs/quantum/bin/python"
OUT="./results_cifar10_k4"
COMMON="--dataset cifar10 --n_epochs 200 --n_runs 1 --n_quantum_heads 4 --output_dir $OUT"

echo "Starting Medium (K=4) experiments..."
echo "  - Attention OFF on cuda:0"
echo "  - Attention ON on cuda:1"

# Run concurrently
"$PY" run_experiment.py $COMMON --device cuda:0 > "log_medium_k4_att_off.txt" 2>&1 &
"$PY" run_experiment.py $COMMON --attention_encoding --device cuda:1 > "log_medium_k4_att_on.txt" 2>&1 &

wait
echo "Medium (K=4) experiments completed."
