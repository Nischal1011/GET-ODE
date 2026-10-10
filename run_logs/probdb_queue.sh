#!/usr/bin/env bash
# CHANGES.md Part 30: probabilistic GIL-ODE v1 (K = 8 particles, learned diffusion, ensemble-mean
# forecast, mean likelihood + fair CRPS, full-horizon loss). Part 30.1: debiased ensemble-mean loss.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 8 ]; do sleep 60; done; }
run () { local tag="$1"; shift; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667"
PROB="--niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --extrap True --horizon-loss full --particles 8 --debias"
for s in 1991 1992 1993; do run probdb_s${s}_charged_extrap $PROB $CH --random-seed $s; done
