#!/usr/bin/env bash
# CHANGES.md Part 35: test-shaped training tasks (--resplit), 15-epoch screen, charged seed 1991.
# References: Part 29 config 0.0652 (scr_ref), + MoE-4 0.0596 (scr_moe).
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
guard () {
  while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ] || \
        [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 6000 ]; do sleep 60; done; }
run () { local tag="$1"; shift; guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --random-seed 1991"
BASE="--niters 15 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --reversion"
run scr_resplit_charged $BASE --resplit $CH &
sleep 120; run scr_resplitmoe_charged $BASE --resplit --moe 4 $CH &
wait
echo "SCREEN35 COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
