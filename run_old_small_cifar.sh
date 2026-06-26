#!/bin/bash

# =========================================================================
# Reproduce the original "SMALL quantum BEATS no-quantum" result for CIFAR-10 and CIFAR-100.
#
# Uses the OLD (legacy) classical backbone: FC SuperpixelProjector + SE-style
# SSA. That backbone is the DEFAULT now, so we do NOT pass --new_backbone.
# Conditions: scale=small, img_size=28, scheduler=cosine, NO CutMix/MixUp/RandAugment.
# =========================================================================

export PYTHONUNBUFFERED=1

echo "========================================================="
echo " OLD-BACKBONE REPRODUCTION: small quantum vs no-quantum (CIFAR)"
echo "========================================================="

PY="/home/e21283/miniconda3/envs/quantum/bin/python"
OUT="./results_old_backbone"
LOGS="logs_old_backbone"
mkdir -p "$LOGS"

for DATASET in cifar10 cifar100; do
    echo ""
    echo "---------------------------------------------------------"
    echo " Starting runs for dataset: $DATASET"
    echo "---------------------------------------------------------"

    COMMON_ARGS="--dataset $DATASET --n_epochs 200 --n_runs 1 --scale small --img_size 28 --scheduler cosine --output_dir $OUT --resume"

    # classical baseline + quantum without attention encoding
    (
        echo "[$DATASET] Small Classical Baseline (no_quantum)..."
        $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 1 --ablation no_quantum --device cuda:0 > "$LOGS/log_${DATASET}_small_classical.txt" 2>&1

        echo "[$DATASET] Small Quantum (k=4, NO attenc)..."
        $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 4 --contrastive --device cuda:0 > "$LOGS/log_${DATASET}_small_k4_no_attenc.txt" 2>&1

        echo "[$DATASET] Jobs Finished (Baseline & NO attenc)."
    ) &

    # the winning config — quantum with attention encoding
    (
        echo "[$DATASET] Small Quantum (k=4, WITH attenc) -- the winner..."
        $PY run_experiment.py $COMMON_ARGS --n_quantum_heads 4 --attention_encoding --contrastive --device cuda:0 > "$LOGS/log_${DATASET}_small_k4_attenc.txt" 2>&1

        echo "[$DATASET] Jobs Finished (WITH attenc)."
    ) &

    wait
done

echo "========================================================="
echo " OLD-BACKBONE REPRODUCTION COMPLETED  ->  $OUT"
echo "========================================================="
