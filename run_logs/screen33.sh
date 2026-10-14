#!/usr/bin/env bash
# CHANGES.md Part 33: fast screen of the top-3 ideas on charged-extrap, seed 1991, 15 epochs,
# against a matched reference (Part 29 config, same budget). Judged on validation only.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 60; done; }
run () { local tag="$1"; shift; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --random-seed 1991"
BASE="--niters 15 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --reversion"
a () { run scr_ref_charged $BASE $CH; run scr_moe_charged $BASE --moe 4 $CH; }
b () { sleep 120; run scr_geo_charged $BASE --geo-msg $CH; }
c () { sleep 240; run scr_ctx_charged $BASE --ctx-init $CH; }
a & b & c & wait
echo "SCREEN33 COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
