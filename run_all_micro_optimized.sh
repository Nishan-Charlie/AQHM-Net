#!/usr/bin/env bash
# run_all_micro_optimized.sh
# Run the optimized micro experiment (quantum vs classical) for all remaining datasets.

export PYTHONUNBUFFERED=1
PY="/home/e21283/miniconda3/envs/quantum/bin/python"
EPOCHS=100
RUNS=1

mkdir -p logs_combinations

DATASETS=(
    "cifar100"
    "mnist"
    "pathmnist"
    "dermamnist"
    "pneumoniamnist"
    "breastmnist"
)

echo "========================================================="
echo " Starting Optimized Micro Experiments for all datasets"
echo "========================================================="

for DATASET in "${DATASETS[@]}"; do
    echo "---------------------------------------------------------"
    echo " Starting dataset: $DATASET"
    echo "---------------------------------------------------------"

    # 1. OPTIMIZED QUANTUM (K=4) on cuda:0
    echo "[$DATASET] Running OPTIMIZED AQHM-Net (K=4) on cuda:0"
    "$PY" run_experiment.py --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --scale micro --n_quantum_heads 4 \
        --attention_encoding --contrastive \
        --output_dir ./results_combinations --device cuda:0 \
        > "logs_combinations/log_${DATASET}_optimized_quantum.txt" 2>&1 &

    # 2. NO-QUANTUM ABLATION (Classical only) on cuda:1
    echo "[$DATASET] Running NO-QUANTUM Ablation on cuda:1"
    "$PY" run_experiment.py --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --scale micro --ablation no_quantum \
        --output_dir ./results_combinations --device cuda:1 \
        > "logs_combinations/log_${DATASET}_optimized_noquant.txt" 2>&1 &

    # Wait for both branches of the current dataset to finish before moving to the next dataset
    wait

    echo "[$DATASET] Completed."
done

echo "========================================================="
echo " All datasets completed!"
echo "========================================================="
