#!/usr/bin/env bash
# CIFAR-10 model-size comparison: small / medium / large, 1 seed each, 100 epochs.
# K=1, attention off -> pure capacity (backbone width/depth + fusion) comparison.
PY="C:/Users/nisha/miniconda3/envs/pmqnet/python.exe"
OUT="./results_scale_k4"
COMMON="--dataset cifar10 --n_runs 1 --base_seed 0 --n_epochs 100 --patience 1000 --output_dir $OUT"
echo ">>> START $(date '+%H:%M:%S')"
"$PY" run_experiment.py $COMMON --scale large --n_quantum_heads 4 > "$OUT/large.log"  2>&1
# "$PY" run_experiment.py $COMMON --scale medium --n_quantum_heads 4> "$OUT/medium.log" 2>&1
# "$PY" run_experiment.py $COMMON --scale small   > "$OUT/small.log"  2>&1
echo ">>> ALL DONE $(date '+%H:%M:%S')"
