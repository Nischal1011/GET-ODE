#!/bin/bash
set -x
cd /mnt/c/Users/nsubedi/Documents/LG-ODE

echo "=== [1/6] Corrected LG-ODE, springs, interpolation, 60% observed, 30 epochs ==="
.venv/bin/python run_models_corrected.py --data spring --niters 30 --alias corrected_spring_interp_60 \
  > run_logs/corrected_spring_interp_60.log 2>&1
echo "=== [1/6] DONE exit=$? ==="

echo "=== [2/6] Corrected LG-ODE, springs, extrapolation, 60% observed, 30 epochs ==="
.venv/bin/python run_models_corrected.py --data spring --extrap True --niters 30 --alias corrected_spring_extrap_60 \
  > run_logs/corrected_spring_extrap_60.log 2>&1
echo "=== [2/6] DONE exit=$? ==="

echo "=== [3/6] Corrected LG-ODE, charged, interpolation, 60% observed, 30 epochs ==="
.venv/bin/python run_models_corrected.py --data charged --niters 30 --alias corrected_charged_interp_60 \
  > run_logs/corrected_charged_interp_60.log 2>&1
echo "=== [3/6] DONE exit=$? ==="

echo "=== [4/6] Corrected LG-ODE, charged, extrapolation, 60% observed, 30 epochs ==="
.venv/bin/python run_models_corrected.py --data charged --extrap True --niters 30 --alias corrected_charged_extrap_60 \
  > run_logs/corrected_charged_extrap_60.log 2>&1
echo "=== [4/6] DONE exit=$? ==="

echo "=== [5/6] Corrected LG-ODE, IEEE39, interpolation, 60% observed, 30 epochs ==="
.venv/bin/python run_models_corrected.py --data ieee39 --niters 30 --alias corrected_ieee39_interp_60 \
  > run_logs/corrected_ieee39_interp_60.log 2>&1
echo "=== [5/6] DONE exit=$? ==="

echo "=== [6/6] Corrected LG-ODE, IEEE39, extrapolation, 60% observed, 30 epochs ==="
.venv/bin/python run_models_corrected.py --data ieee39 --extrap True --niters 30 --alias corrected_ieee39_extrap_60 \
  > run_logs/corrected_ieee39_extrap_60.log 2>&1
echo "=== [6/6] DONE exit=$? ==="

echo "CORRECTED LG-ODE BASELINE STAGE 1 COMPLETE"
