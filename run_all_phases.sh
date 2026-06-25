#!/bin/bash

# Make sure all phase scripts are executable
chmod +x run_phase1_micro.sh run_phase2_small.sh run_phase3_medium.sh

echo "========================================================="
echo " STARTING COMPLETE MULTI-PHASE EXPERIMENT SUITE"
echo "========================================================="

echo "[1/3] Executing Phase 1 (Micro)..."
./run_phase1_micro.sh

echo "[2/3] Executing Phase 2 (Small)..."
./run_phase2_small.sh

echo "[3/3] Executing Phase 3 (Medium)..."
./run_phase3_medium.sh

echo "========================================================="
echo " ALL PHASES COMPLETED SUCCESSFULLY "
echo "========================================================="
