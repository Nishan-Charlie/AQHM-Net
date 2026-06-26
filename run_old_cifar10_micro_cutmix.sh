#!/bin/bash

# =========================================================================
# OLD-BACKBONE with CutMix preprocessing for CIFAR-10 at MICRO scale
# =========================================================================

export PYTHONUNBUFFERED=1

echo "========================================================="
echo " OLD-BACKBONE: CIFAR-10 MICRO with CUTMIX"
echo "========================================================="

PY="/home/e21283/miniconda3/envs/quantum/bin/python"
OUT="./results_old_backbone"
LOGS="logs_old_backbone"
mkdir -p "$LOGS"

DATASET="cifar10"
# Added --cutmix_alpha 1.0 for CutMix preprocessing
COMMON_ARGS="--dataset $DATASET --n_epochs 200 --n_runs 1 --scale micro --img_size 28 --scheduler cosine --cutmix_alpha 1.0 --output_dir $OUT --resume"

# classical baseline + quantum without attention encoding
# (
#     echo "[$DATASET] Micro Classical Baseline (no_quantum)..."
#     $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 1 --ablation no_quantum --device cuda:0 > "$LOGS/log_${DATASET}_micro_cutmix_classical.txt" 2>&1

#     echo "[$DATASET] Micro Quantum (k=4, NO attenc)..."
#     $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 4 --contrastive --device cuda:0 > "$LOGS/log_${DATASET}_micro_cutmix_k4_no_attenc.txt" 2>&1

#     echo "[$DATASET] Jobs Finished (Baseline & NO attenc)."
# ) &

# quantum with attention encoding
(
    echo "[$DATASET] Micro Quantum (k=4, WITH attenc)..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 4 --attention_encoding --contrastive --device cuda:0 > "$LOGS/log_${DATASET}_micro_cutmix_k4_attenc.txt" 2>&1

    echo "[$DATASET] Jobs Finished (WITH attenc)."
) &

wait

echo "========================================================="
echo " OLD-BACKBONE CUTMIX MICRO COMPLETED  ->  $OUT"
echo "========================================================="
