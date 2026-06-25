#!/usr/bin/env bash
# run_k4_k8_combinations.sh
# Exhaustive combination for K=4 and K=8 across datasets and scales.

export PYTHONUNBUFFERED=1
PY="/home/e21283/miniconda3/envs/quantum/bin/python"
EPOCHS=200
RUNS=1

DATASETS=(
    "cifar10"
    "cifar100"
    "mnist"
    "pathmnist"
    "dermamnist"
    "pneumoniamnist"
    "breastmnist"
)

mkdir -p logs_combinations

for DATASET in "${DATASETS[@]}"; do
    echo "========================================================="
    echo " Starting K=4 & K=8 combinations for dataset: $DATASET"
    echo "========================================================="

    # K=4 (Small, Medium, Large)
    echo "[$DATASET] Running AQHM-Net (K=4) - Small, Medium, Large"
    "$PY" run_experiment.py --resume --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --n_quantum_heads 4 --scale small --output_dir ./results_combinations --device cuda:0 \
        > "logs_combinations/log_${DATASET}_aqhm_k4_small.txt" 2>&1 &
        
    "$PY" run_experiment.py --resume --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --n_quantum_heads 4 --scale medium --output_dir ./results_combinations --device cuda:1 \
        > "logs_combinations/log_${DATASET}_aqhm_k4_medium.txt" 2>&1 &
        
    "$PY" run_experiment.py --resume --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --n_quantum_heads 4 --scale large --output_dir ./results_combinations --device cuda:0 \
        > "logs_combinations/log_${DATASET}_aqhm_k4_large.txt" 2>&1 &

    # K=8 (Small, Medium, Large)
    echo "[$DATASET] Running AQHM-Net (K=8) - Small, Medium, Large"
    "$PY" run_experiment.py --resume --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --n_quantum_heads 8 --scale small --output_dir ./results_combinations --device cuda:1 \
        > "logs_combinations/log_${DATASET}_aqhm_k8_small.txt" 2>&1 &
        
    "$PY" run_experiment.py --resume --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --n_quantum_heads 8 --scale medium --output_dir ./results_combinations --device cuda:0 \
        > "logs_combinations/log_${DATASET}_aqhm_k8_medium.txt" 2>&1 &
        
    "$PY" run_experiment.py --resume --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --n_quantum_heads 8 --scale large --output_dir ./results_combinations --device cuda:1 \
        > "logs_combinations/log_${DATASET}_aqhm_k8_large.txt" 2>&1 &
        
    # Wait for all 6 experiments for this dataset to finish before moving to the next dataset
    wait

done

echo "All K=4 and K=8 combinations completed!"
