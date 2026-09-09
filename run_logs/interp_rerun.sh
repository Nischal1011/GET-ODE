#!/bin/bash
set -x
cd /mnt/c/Users/nsubedi/Documents/LG-ODE

echo "=== [1/2] LG-ODE (patched), interpolation, springs @ 60% observed, 30 epochs ==="
.venv/bin/python run_models.py --dataset-dir data/spring --niters 30 --alias springs_interp_60_patched \
  > run_logs/springs_interp_60_patched.log 2>&1
echo "=== [1/2] DONE, exit code $? ==="

echo "=== [2/2] AT-LG-ODE (patched), interpolation, springs @ 60% observed, 30 epochs ==="
.venv/bin/python run_models_at.py --dataset-dir data/spring --niters 30 --alias at_springs_interp_60_patched \
  > run_logs/at_springs_interp_60_patched.log 2>&1
echo "=== [2/2] DONE, exit code $? ==="

echo "INTERP RERUNS COMPLETE"
