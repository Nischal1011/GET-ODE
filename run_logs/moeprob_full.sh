#!/usr/bin/env bash
# CHANGES.md Part 33.2: MoE (K=4) + probabilistic ensemble (K=8 particles), charged-extrap, 3 seeds in
# parallel, on the Part 29 config. RAM guard 6 GB (three jobs ~3 GB each).
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 60; done; }
run () { local tag="$1"; shift; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True"
BASE="--niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --reversion --moe 4 --particles 8"
run moeprob_s1991_charged_extrap $BASE $CH --random-seed 1991 &
sleep 120; run moeprob_s1992_charged_extrap $BASE $CH --random-seed 1992 &
sleep 120; run moeprob_s1993_charged_extrap $BASE $CH --random-seed 1993 &
wait
echo "MOE+PROB COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
