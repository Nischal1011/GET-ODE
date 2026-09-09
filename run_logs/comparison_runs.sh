#!/bin/bash
set -x
cd /mnt/c/Users/nsubedi/Documents/LG-ODE

echo "=== [1/3] LG-ODE baseline, extrapolation, springs @ 60% observed ==="
.venv/bin/python run_models.py --dataset-dir data/spring --extrap True --niters 30 --alias springs_extrap_60 \
  > run_logs/springs_extrap_60.log 2>&1
echo "=== [1/3] DONE, exit code $? ==="

echo "=== [2/3] AT-LG-ODE, interpolation, springs @ 60% observed ==="
.venv/bin/python run_models_at.py --dataset-dir data/spring --niters 30 --alias at_springs_interp_60 \
  > run_logs/at_springs_interp_60.log 2>&1
echo "=== [2/3] DONE, exit code $? ==="

echo "=== [3/3] AT-LG-ODE, extrapolation, springs @ 60% observed ==="
.venv/bin/python run_models_at.py --dataset-dir data/spring --extrap True --niters 30 --alias at_springs_extrap_60 \
  > run_logs/at_springs_extrap_60.log 2>&1
echo "=== [3/3] DONE, exit code $? ==="

echo "ALL COMPARISON RUNS COMPLETE"
