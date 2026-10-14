#!/usr/bin/env bash
# CHANGES.md Part 32: time-reversal consistency with unlabelled horizon extension (x1 labelled
# horizon), on the Part 29 configuration (reversion head + full-horizon loss). Charged seed 1991,
# weights 0.1 and 1.0; the weight is chosen on VALIDATION and then applied to every cell.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 60; done; }
run () { local tag="$1"; shift; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --random-seed 1991"
BASE="--niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --reversion --reversal-ext 1.0"
run tr01_s1991_charged_extrap $BASE --reversal-weight 0.1 $CH &
sleep 240
run tr1_s1991_charged_extrap $BASE --reversal-weight 1.0 $CH &
wait
echo "REVERSAL TEST COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
