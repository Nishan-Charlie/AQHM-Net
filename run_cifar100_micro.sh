#!/bin/bash

export PYTHONUNBUFFERED=1

echo "========================================================="
echo " STARTING CIFAR100 MICRO SCALE EXPERIMENTS"
echo "========================================================="

PY="/home/e21283/miniconda3/envs/quantum/bin/python"
COMMON_ARGS="--dataset cifar100 --n_epochs 200 --n_runs 1 --scale micro --img_size 224 --scheduler onecycle --cutmix_alpha 1.0 --mixup_alpha 0.8 --use_randaugment --output_dir ./results_combinations --resume"

# GPU 0 Jobs
(
    echo "[GPU 0] Starting Micro Classical Baseline..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 1 --ablation no_quantum --device cuda:0 > logs_combinations/log_cifar100_micro_classical.txt 2>&1

    echo "[GPU 0] Starting Micro Quantum (k=4, NO attenc)..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 4 --contrastive --device cuda:0 > logs_combinations/log_cifar100_micro_k4_no_attenc.txt 2>&1
    
    echo "[GPU 0] Jobs Finished."
) &

# GPU 1 Jobs
(
    echo "[GPU 1] Starting Micro Quantum (k=4, WITH attenc)..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 4 --attention_encoding --contrastive --device cuda:1 > logs_combinations/log_cifar100_micro_k4_attenc.txt 2>&1
    
    echo "[GPU 1] Jobs Finished."
) &

wait

echo "========================================================="
echo " CIFAR100 MICRO COMPLETED "
echo "========================================================="
