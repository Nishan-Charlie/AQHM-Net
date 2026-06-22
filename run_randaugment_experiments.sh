#!/usr/bin/env bash
# run_randaugment_experiments.sh
# Execute experiments with top-tier augmentations (RandAugment, CutMix, MixUp) and OneCycleLR scheduler.
# Distributes jobs across cuda:0 and cuda:1

export PYTHONUNBUFFERED=1
PY="/home/e21283/miniconda3/envs/quantum/bin/python"
EPOCHS=100
RUNS=1

mkdir -p logs_combinations

# Run on remaining datasets: fashionmnist (remaining), mnist (reference), cifar10 (reference)
DATASETS=(
    "fashionmnist"
    "mnist"
    "cifar10"
)

echo "========================================================="
# Print the starting header
echo " Starting RandAugment + OneCycleLR + CutMix Experiments"
echo "========================================================="

for DATASET in "${DATASETS[@]}"; do
    echo "---------------------------------------------------------"
    echo " Starting dataset: $DATASET"
    echo "---------------------------------------------------------"

    # 1. QUANTUM HYBRID (K=4) + RandAugment + CutMix/MixUp + OneCycleLR on cuda:0
    echo "[$DATASET] Running AQHM-Net (K=4, attenc, RandAugment, CutMix, OneCycleLR) on cuda:0"
    "$PY" run_experiment.py --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --scale micro --n_quantum_heads 4 \
        --attention_encoding --contrastive \
        --scheduler onecycle --cutmix_alpha 1.0 --mixup_alpha 0.8 --use_randaugment \
        --output_dir ./results_combinations --device cuda:0 \
        > "logs_combinations/log_${DATASET}_randaugment_quantum.txt" 2>&1 &

    # 2. NO-QUANTUM ABLATION + RandAugment + CutMix/MixUp + OneCycleLR on cuda:1
    echo "[$DATASET] Running NO-QUANTUM Ablation (RandAugment, CutMix, OneCycleLR) on cuda:1"
    "$PY" run_experiment.py --dataset "$DATASET" --n_epochs $EPOCHS --n_runs $RUNS \
        --scale micro --ablation no_quantum \
        --scheduler onecycle --cutmix_alpha 1.0 --mixup_alpha 0.8 --use_randaugment \
        --output_dir ./results_combinations --device cuda:1 \
        > "logs_combinations/log_${DATASET}_randaugment_noquant.txt" 2>&1 &

    # Wait for both branches of the current dataset to finish before moving to the next dataset
    wait

    echo "[$DATASET] Completed."
done

echo "========================================================="
echo " All RandAugment combination experiments completed!"
echo "========================================================="
