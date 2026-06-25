#!/bin/bash

export PYTHONUNBUFFERED=1

echo "========================================================="
echo " STARTING PHASE 2: SMALL SCALE EXPERIMENTS"
echo "========================================================="

PY="/home/e21283/miniconda3/envs/quantum/bin/python"
COMMON_ARGS="--dataset cifar10 --n_epochs 200 --n_runs 1 --scale small --img_size 224 --scheduler onecycle --cutmix_alpha 1.0 --mixup_alpha 0.8 --use_randaugment --output_dir ./results_combinations --resume"

# GPU 0 Jobs
(
    echo "[GPU 0] Starting Small Classical Baseline..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 1 --ablation no_quantum --device cuda:0 > logs_combinations/log_cifar10_small_classical.txt 2>&1

    echo "[GPU 0] Starting Small Quantum (k=4, NO attenc)..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 4 --contrastive --device cuda:0 > logs_combinations/log_cifar10_small_k4_no_attenc.txt 2>&1
    
    echo "[GPU 0] Starting Small Quantum (k=8, NO attenc)..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 8 --contrastive --device cuda:0 > logs_combinations/log_cifar10_small_k8_no_attenc.txt 2>&1

    echo "[GPU 0] Phase 2 Jobs Finished."
) &

# GPU 1 Jobs
(
    echo "[GPU 1] Starting Small Quantum (k=4, WITH attenc)..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 4 --attention_encoding --contrastive --device cuda:1 > logs_combinations/log_cifar10_small_k4_attenc.txt 2>&1
    
    echo "[GPU 1] Starting Small Quantum (k=8, WITH attenc)..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 8 --attention_encoding --contrastive --device cuda:1 > logs_combinations/log_cifar10_small_k8_attenc.txt 2>&1

    echo "[GPU 1] Phase 2 Jobs Finished."
) &

wait

echo "========================================================="
echo " PHASE 2 COMPLETED "
echo "========================================================="
