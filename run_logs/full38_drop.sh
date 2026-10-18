#!/usr/bin/env bash
# CHANGES.md Part 38.1: dynamics dropout (E) passed the paired screen on both seeds. Full
# from-scratch runs on the 5/6 baseline config + --dyn-dropout 0.1, 50 epochs (same budget as
# every baseline), charged-extrap, 3 seeds; two at a time. Guards on RAM and GPU memory.
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
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True"
CFG="--niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --dyn-dropout 0.1"
( run full38_drop_s1991_charged_extrap $CFG $CH --random-seed 1991; run full38_drop_s1993_charged_extrap $CFG $CH --random-seed 1993 ) &
sleep 120; run full38_drop_s1992_charged_extrap $CFG $CH --random-seed 1992 &
wait
echo "FULL38 DROP COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
