#!/bin/bash
set -x
cd /mnt/c/Users/nsubedi/Documents/LG-ODE

# Ablation study: springs, extrapolation, 60% observed, 30 epochs, 2 seeds.
# Seed 1991 for LG-ODE and current-AT (ablation=none) reuse existing results:
#   run_logs/springs_extrap_60.log        (LG-ODE,   seed 1991) = 1.6418e-2
#   run_logs/at_springs_extrap_60.log     (AT none,  seed 1991) = 1.2374e-2
# The 8 runs below fill in everything else.

echo "=== [1/8] LG-ODE, seed 42 ==="
.venv/bin/python run_models.py --dataset-dir data/spring --extrap True --niters 30 --random-seed 42 \
  --alias ablation_lgode_seed42 > run_logs/ablation_lgode_seed42.log 2>&1
echo "=== [1/8] DONE exit=$? ==="

echo "=== [2/8] AT hardmask, seed 1991 ==="
.venv/bin/python run_models_at.py --dataset-dir data/spring --extrap True --niters 30 --random-seed 1991 \
  --ablation hardmask --alias ablation_hardmask_seed1991 > run_logs/ablation_hardmask_seed1991.log 2>&1
echo "=== [2/8] DONE exit=$? ==="

echo "=== [3/8] AT hardmask, seed 42 ==="
.venv/bin/python run_models_at.py --dataset-dir data/spring --extrap True --niters 30 --random-seed 42 \
  --ablation hardmask --alias ablation_hardmask_seed42 > run_logs/ablation_hardmask_seed42.log 2>&1
echo "=== [3/8] DONE exit=$? ==="

echo "=== [4/8] AT constant, seed 1991 ==="
.venv/bin/python run_models_at.py --dataset-dir data/spring --extrap True --niters 30 --random-seed 1991 \
  --ablation constant --alias ablation_constant_seed1991 > run_logs/ablation_constant_seed1991.log 2>&1
echo "=== [4/8] DONE exit=$? ==="

echo "=== [5/8] AT constant, seed 42 ==="
.venv/bin/python run_models_at.py --dataset-dir data/spring --extrap True --niters 30 --random-seed 42 \
  --ablation constant --alias ablation_constant_seed42 > run_logs/ablation_constant_seed42.log 2>&1
echo "=== [5/8] DONE exit=$? ==="

echo "=== [6/8] AT shuffled, seed 1991 ==="
.venv/bin/python run_models_at.py --dataset-dir data/spring --extrap True --niters 30 --random-seed 1991 \
  --ablation shuffled --alias ablation_shuffled_seed1991 > run_logs/ablation_shuffled_seed1991.log 2>&1
echo "=== [6/8] DONE exit=$? ==="

echo "=== [7/8] AT shuffled, seed 42 ==="
.venv/bin/python run_models_at.py --dataset-dir data/spring --extrap True --niters 30 --random-seed 42 \
  --ablation shuffled --alias ablation_shuffled_seed42 > run_logs/ablation_shuffled_seed42.log 2>&1
echo "=== [7/8] DONE exit=$? ==="

echo "=== [8/8] AT current (ablation=none), seed 42 ==="
.venv/bin/python run_models_at.py --dataset-dir data/spring --extrap True --niters 30 --random-seed 42 \
  --ablation none --alias ablation_none_seed42 > run_logs/ablation_none_seed42.log 2>&1
echo "=== [8/8] DONE exit=$? ==="

echo "ABLATION STUDY COMPLETE"
