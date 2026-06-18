#!/usr/bin/env bash
# DermaMNIST @128px: K x attention-encoding sweep (highest res that fits 8GB VRAM).
# 1 seed, batch 32, K in {1,8}, attention {off,on} = 4 configs, sequential.
PY="C:/Users/nisha/miniconda3/envs/pmqnet/python.exe"
OUT="./results_res128"
BASE="--dataset dermamnist --img_size 128 --use_focal_loss --use_balanced_sampler --n_runs 1 --n_epochs 50 --patience 1000 --batch_size 32 --output_dir $OUT"
run () { echo ">>> $1  ($(date '+%H:%M:%S'))"; shift; "$PY" run_experiment.py $BASE "$@" || echo "!!! FAILED"; }
run "K=1 att OFF"
run "K=1 att ON"  --attention_encoding
run "K=8 att OFF" --n_quantum_heads 8
run "K=8 att ON"  --n_quantum_heads 8 --attention_encoding
echo ">>> ALL DONE ($(date '+%H:%M:%S'))"
