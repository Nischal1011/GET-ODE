#!/bin/bash
set -x
cd /mnt/c/Users/nsubedi/Documents/LG-ODE

echo "=== [1/2] AT-LG-ODE (nonadj-floor=0.3), interpolation, charged @ 60% observed, 30 epochs ==="
.venv/bin/python run_models_at.py --data charged --dataset-dir data/charged --niters 30 --nonadj-floor 0.3 --alias at_charged_interp_60_floor \
  > run_logs/at_charged_interp_60_floor.log 2>&1
echo "=== [1/2] DONE, exit code $? ==="

echo "=== [2/2] AT-LG-ODE (nonadj-floor=0.3), extrapolation, charged @ 60% observed, 30 epochs ==="
.venv/bin/python run_models_at.py --data charged --dataset-dir data/charged --extrap True --niters 30 --nonadj-floor 0.3 --alias at_charged_extrap_60_floor \
  > run_logs/at_charged_extrap_60_floor.log 2>&1
echo "=== [2/2] DONE, exit code $? ==="

echo "CHARGED FLOOR RERUN COMPLETE"
