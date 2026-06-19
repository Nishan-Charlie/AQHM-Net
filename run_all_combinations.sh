#!/usr/bin/env bash
# run_all_combinations.sh
# Exhaustive combination of experiments across datasets, scales, and baselines.
# Distributes jobs across cuda:0 and cuda:1

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

SCALES=("small" "medium" "large")

# Ensure logs dir exists
mkdir -p logs_combinations

for DATASET in "${DATASETS[@]}"; do
    echo "========================================================="
    echo " Starting combinations for dataset: $DATASET"
    echo "========================================================="

    # 1. AQHM-Net with K=1, Attention OFF (Small, Medium, Large)
    # We can run Small on cuda:0, Medium on cuda:1, wait, then Large on cuda:0
    
    echo "[$DATASET] Running AQHM-Net (K=1, Attention OFF) - Small & Medium"
    "$PY" run_experiment.py --resume --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --n_quantum_heads 1 --scale small --output_dir ./results_combinations --device cuda:0 \
        > "logs_combinations/log_${DATASET}_aqhm_small.txt" 2>&1 &
        
    "$PY" run_experiment.py --resume --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --n_quantum_heads 1 --scale medium --output_dir ./results_combinations --device cuda:1 \
        > "logs_combinations/log_${DATASET}_aqhm_medium.txt" 2>&1 &
        
    wait

    echo "[$DATASET] Running AQHM-Net (K=1, Attention OFF) - Large"
    "$PY" run_experiment.py --resume --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --n_quantum_heads 1 --scale large --output_dir ./results_combinations --device cuda:0 \
        > "logs_combinations/log_${DATASET}_aqhm_large.txt" 2>&1 &
        
    # We pair the Large AQHM-Net with the Small No-Quantum Ablation
    echo "[$DATASET] Running No-Quantum Ablation - Small"
    "$PY" run_experiment.py --resume --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --ablation no_quantum --scale small --output_dir ./results_combinations --device cuda:1 \
        > "logs_combinations/log_${DATASET}_noquant_small.txt" 2>&1 &
        
    wait

    # 2. No-Quantum Ablation (Medium, Large)
    echo "[$DATASET] Running No-Quantum Ablation - Medium & Large"
    "$PY" run_experiment.py --resume --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --ablation no_quantum --scale medium --output_dir ./results_combinations --device cuda:0 \
        > "logs_combinations/log_${DATASET}_noquant_medium.txt" 2>&1 &
        
    "$PY" run_experiment.py --resume --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --ablation no_quantum --scale large --output_dir ./results_combinations --device cuda:1 \
        > "logs_combinations/log_${DATASET}_noquant_large.txt" 2>&1 &
        
    wait

    # 3. ResNet-18 Baseline
    echo "[$DATASET] Running ResNet-18 Baseline"
    "$PY" run_experiment.py --resume --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --model resnet18 --output_dir ./results_combinations --device cuda:0 \
        > "logs_combinations/log_${DATASET}_resnet18.txt" 2>&1 &
        
    wait

done

echo "All combinations completed!"
