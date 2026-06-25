#!/usr/bin/env bash
# run_optimized_test.sh
# A quick parallel test on CIFAR-10 to verify if the "micro" backbone,
# boosted VQC LR, attention encoding, and contrastive loss
# grant the Quantum branch a 3-4% advantage over the No-Quantum ablation.

export PYTHONUNBUFFERED=1
PY="/home/e21283/miniconda3/envs/quantum/bin/python"
EPOCHS=100
RUNS=1

mkdir -p logs_combinations

echo "========================================================="
echo " Starting Optimized Quantum Advantage Test on CIFAR-10"
echo "========================================================="

# 1. OPTIMIZED QUANTUM (K=4)
# Features: --scale micro, --n_quantum_heads 4, --attention_encoding, --contrastive
echo "[cifar10] Running OPTIMIZED AQHM-Net (K=4) on cuda:0"
"$PY" run_experiment.py --dataset cifar10 --n_epochs $EPOCHS --n_runs $RUNS \
    --scale micro --n_quantum_heads 4 \
    --attention_encoding --contrastive \
    --output_dir ./results_combinations --device cuda:0 \
    > "logs_combinations/log_cifar10_optimized_quantum.txt" 2>&1 &

# 2. NO-QUANTUM ABLATION (Classical only)
# Features: --scale micro (to ensure a fair apples-to-apples comparison)
echo "[cifar10] Running NO-QUANTUM Ablation on cuda:1"
"$PY" run_experiment.py --dataset cifar10 --n_epochs $EPOCHS --n_runs $RUNS \
    --scale micro --ablation no_quantum \
    --output_dir ./results_combinations --device cuda:1 \
    > "logs_combinations/log_cifar10_optimized_noquant.txt" 2>&1 &

wait

echo "Optimized test completed! Check the logs to see if Quantum outperformed No-Quantum."
