#!/usr/bin/env bash
# Follows moeprob_full.sh: springs and IEEE39 extrapolation with the MoE + probabilistic config,
# 3 seeds each, so the configuration can be applied to every extrapolation cell. 3 parallel jobs.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
until grep -q "MOE+PROB COMPLETE" $LOG; do sleep 120; done
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 60; done; }
run () { local tag="$1"; shift; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
SP="--data spring  --dataset-dir data/spring_subset  --val-fraction 0.166666667 --extrap True"
IE="--data ieee39  --dataset-dir data/processed/ieee39_gen_subset --extrap True"
BASE="--niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --reversion --moe 4 --particles 8"
st () { local s=$1; sleep $(( (s-1991)*120 )); run moeprob_s${s}_spring_extrap $BASE $SP --random-seed $s; run moeprob_s${s}_ieee39_extrap $BASE $IE --random-seed $s; }
st 1991 & st 1992 & st 1993 & wait
echo "MOE+PROB REST COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
