#!/usr/bin/env bash
# run_cifar10_r224.sh
export PYTHONUNBUFFERED=1
PY="/home/e21283/miniconda3/envs/quantum/bin/python"

mkdir -p logs_combinations
"$PY" run_experiment.py --dataset cifar10 --n_epochs 200 --n_runs 1 \
    --scale small --n_quantum_heads 4 \
    --scheduler onecycle --cutmix_alpha 1.0 --mixup_alpha 0.8 --use_randaugment \
    --img_size 224 --batch_size 64 --num_workers 4 --output_dir ./results_combinations --device cuda:1 \
    > logs_combinations/log_cifar10_randaugment_k4_small_r224.txt 2>&1
