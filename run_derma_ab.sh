#!/usr/bin/env bash
# DermaMNIST attention-encoding A/B: ON vs OFF, 5 seeds, 50ep, no early stopping.
# Identical settings except --attention_encoding, so it's a clean controlled test.
PY="C:/Users/nisha/miniconda3/envs/pmqnet/python.exe"
OUT="./results_attenc_ab"
BASE="--dataset dermamnist --use_focal_loss --use_balanced_sampler --n_runs 5 --n_epochs 50 --patience 1000 --output_dir $OUT"
echo ">>> START $(date '+%H:%M:%S')"
"$PY" run_experiment.py $BASE                       > "$OUT/derma_off.log" 2>&1 &
"$PY" run_experiment.py $BASE --attention_encoding  > "$OUT/derma_on.log"  2>&1 &
wait
echo ">>> ALL DONE $(date '+%H:%M:%S')"
