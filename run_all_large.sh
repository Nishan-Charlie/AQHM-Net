#!/bin/bash

export PYTHONUNBUFFERED=1

echo "========================================================="
echo " STARTING ALL DATASETS LARGE SCALE EXPERIMENTS"
echo "========================================================="

PY="/home/e21283/miniconda3/envs/quantum/bin/python"
DATASETS=("cifar10" "cifar100" "dermamnist" "breastmnist" "pathmnist" "pneumoniamnist")

# GPU 0 Jobs
(
    for ds in "${DATASETS[@]}"; do
        COMMON="--dataset $ds --n_epochs 200 --n_runs 1 --scale large --scheduler onecycle --cutmix_alpha 1.0 --mixup_alpha 0.8 --use_randaugment --output_dir ./results_combinations --resume"
        
        echo "[GPU 0] [$ds] Starting Large Classical Baseline..."
        $PY run_experiment.py $COMMON --n_quantum_heads 1 --ablation no_quantum --device cuda:0 > logs_combinations/log_${ds}_large_classical.txt 2>&1

        echo "[GPU 0] [$ds] Starting Large Quantum (k=4, NO attenc)..."
        $PY run_experiment.py $COMMON --n_quantum_heads 4 --contrastive --device cuda:0 > logs_combinations/log_${ds}_large_k4_no_attenc.txt 2>&1
        
        echo "[GPU 0] [$ds] Starting Large Quantum (k=8, NO attenc)..."
        $PY run_experiment.py $COMMON --n_quantum_heads 8 --contrastive --device cuda:0 > logs_combinations/log_${ds}_large_k8_no_attenc.txt 2>&1
    done
    echo "[GPU 0] Jobs Finished."
) &

# GPU 1 Jobs
(
    for ds in "${DATASETS[@]}"; do
        COMMON="--dataset $ds --n_epochs 200 --n_runs 1 --scale large --scheduler onecycle --cutmix_alpha 1.0 --mixup_alpha 0.8 --use_randaugment --output_dir ./results_combinations --resume"
        
        echo "[GPU 1] [$ds] Starting Large Quantum (k=4, WITH attenc)..."
        $PY run_experiment.py $COMMON --n_quantum_heads 4 --attention_encoding --contrastive --device cuda:1 > logs_combinations/log_${ds}_large_k4_attenc.txt 2>&1
        
        echo "[GPU 1] [$ds] Starting Large Quantum (k=8, WITH attenc)..."
        $PY run_experiment.py $COMMON --n_quantum_heads 8 --attention_encoding --contrastive --device cuda:1 > logs_combinations/log_${ds}_large_k8_attenc.txt 2>&1
    done
    echo "[GPU 1] Jobs Finished."
) &

wait

echo "========================================================="
echo " ALL DATASETS LARGE COMPLETED "
echo "========================================================="
