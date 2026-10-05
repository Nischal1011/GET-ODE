#!/usr/bin/env bash
# Remaining final-config cells, restarted after the WSL restart at 10-02 13:27 killed charged-extrap
# at epoch 16. Identical flags to run_logs/final_config.sh. Two concurrent (~6 GB of 15).
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/final_config_driver.log
FLAGS="--niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --random-seed 1991"
run () { local tag="$1"; shift; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') (restart) ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" $FLAGS --extrap True --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
run fc_charged_extrap --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 &
run fc_ieee39_extrap  --data ieee39  --dataset-dir data/processed/ieee39_gen_subset &
wait
echo "FINAL CONFIG COMPLETE $(date '+%m-%d %H:%M')" >> "$LOG"
