#!/bin/bash

export PYTHONUNBUFFERED=1

echo "========================================================="
echo " STARTING PHASE 3: MEDIUM SCALE EXPERIMENTS"
echo "========================================================="

PY="/home/e21283/miniconda3/envs/quantum/bin/python"
COMMON_ARGS="--dataset cifar10 --n_epochs 200 --n_runs 1 --scale medium --scheduler onecycle --cutmix_alpha 1.0 --mixup_alpha 0.8 --use_randaugment --output_dir ./results_combinations --resume"

# GPU 0 Jobs
(
    echo "[GPU 0] Starting Medium Classical Baseline..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 1 --ablation no_quantum --device cuda:0 > logs_combinations/log_cifar10_medium_classical.txt 2>&1

    echo "[GPU 0] Starting Medium Quantum (k=4, NO attenc)..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 4 --device cuda:0 > logs_combinations/log_cifar10_medium_k4_no_attenc.txt 2>&1
    
    echo "[GPU 0] Starting Medium Quantum (k=8, NO attenc)..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 8 --device cuda:0 > logs_combinations/log_cifar10_medium_k8_no_attenc.txt 2>&1

    echo "[GPU 0] Phase 3 Jobs Finished."
) &

# GPU 1 Jobs
(
    echo "[GPU 1] Starting Medium Quantum (k=4, WITH attenc)..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 4 --attention_encoding --device cuda:1 > logs_combinations/log_cifar10_medium_k4_attenc.txt 2>&1
    
    echo "[GPU 1] Starting Medium Quantum (k=8, WITH attenc)..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 8 --attention_encoding --device cuda:1 > logs_combinations/log_cifar10_medium_k8_attenc.txt 2>&1

    echo "[GPU 1] Phase 3 Jobs Finished."
) &

wait

echo "========================================================="
echo " PHASE 3 COMPLETED "
echo "========================================================="
