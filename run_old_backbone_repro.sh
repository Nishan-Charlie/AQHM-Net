#!/bin/bash

# =========================================================================
# Reproduce the original "MICRO quantum BEATS no-quantum" result.
#
# Uses the OLD (legacy) classical backbone: FC SuperpixelProjector + SE-style
# SSA. That backbone is the DEFAULT now, so we do NOT pass --new_backbone.
# Conditions match the original run that produced the result
# (cifar10_k4_attenc_micro 0.7874 wF1  vs  cifar10_no_quantum_micro 0.7643):
#   scale=micro, img_size=28, scheduler=cosine, NO CutMix/MixUp/RandAugment.
# =========================================================================

export PYTHONUNBUFFERED=1

echo "========================================================="
echo " OLD-BACKBONE REPRODUCTION: micro quantum vs no-quantum"
echo "========================================================="

PY="/home/e21283/miniconda3/envs/quantum/bin/python"
OUT="./results_old_backbone"
LOGS="logs_old_backbone"
mkdir -p "$LOGS"

# Original conditions (28px, cosine, no augmentation). Legacy backbone = default.
COMMON_ARGS="--dataset cifar10 --n_epochs 200 --n_runs 1 --scale micro --img_size 28 --scheduler cosine --output_dir $OUT --resume"

# # GPU 0: classical baseline + quantum without attention encoding
# (
#     echo "[GPU 0] Micro Classical Baseline (no_quantum)..."
#     $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 1 --ablation no_quantum --device cuda:0 > "$LOGS/log_cifar10_micro_classical.txt" 2>&1

#     echo "[GPU 0] Micro Quantum (k=4, NO attenc)..."
#     $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 4 --contrastive --device cuda:0 > "$LOGS/log_cifar10_micro_k4_no_attenc.txt" 2>&1

#     echo "[GPU 0] Jobs Finished."
# ) &

# GPU 1: the winning config — quantum with attention encoding
(
    echo "[GPU 0] Micro Quantum (k=4, WITH attenc) -- the winner..."
    $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 4 --attention_encoding --contrastive --device cuda:0 > "$LOGS/log_cifar10_micro_k4_attenc.txt" 2>&1

    echo "[GPU 0] Jobs Finished."
) &

wait

echo "========================================================="
echo " OLD-BACKBONE REPRODUCTION COMPLETED  ->  $OUT"
echo "========================================================="
