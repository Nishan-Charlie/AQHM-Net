#!/bin/bash

chmod +x run_medmnist_micro.sh run_medmnist_small.sh run_medmnist_medium.sh

echo "========================================================="
echo " STARTING MEDMNIST COMPLETE MULTI-PHASE SUITE"
echo "========================================================="

echo "[1/3] Executing MedMNIST Phase 1 (Micro)..."
./run_medmnist_micro.sh

echo "[2/3] Executing MedMNIST Phase 2 (Small)..."
./run_medmnist_small.sh

echo "[3/3] Executing MedMNIST Phase 3 (Medium)..."
./run_medmnist_medium.sh

echo "========================================================="
echo " ALL MEDMNIST PHASES COMPLETED SUCCESSFULLY "
echo "========================================================="
