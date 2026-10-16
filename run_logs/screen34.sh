#!/usr/bin/env bash
# CHANGES.md Part 34: MoE refinements, 15-epoch screen on charged-extrap seed 1991, Part 29 config.
# Reference: scr_moe_charged (MoE-4, 15 epochs) best val 0.0596. Pass: <= 0.0566 (>=5% better).
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
a () { run scr_moe8_charged $BASE --moe 8 $CH; run scr_width_charged $BASE --int-width 238 $CH; }
b () { sleep 120; run scr_moeroute_charged $BASE --moe 4 --moe-route --moe-temp 0.3 --route-weight 10 $CH; }
c () { sleep 240; run scr_edgemoe_charged $BASE --moe-msg 4 --moe-route --moe-temp 0.5 --route-weight 10 $CH; }
a & b & c & wait
echo "SCREEN34 COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
