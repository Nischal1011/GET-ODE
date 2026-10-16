#!/usr/bin/env bash
# Part 33.2 rerun: MoE + probabilistic, charged-extrap, two jobs at a time. An unrelated GPU job
# (run_gsm8k.py, ~20 GB) started 10-15 ~11:52 and the three parallel runs died with CUDA driver
# errors at that moment. Each launch waits for >= 6 GB free GPU memory (one job needs ~3.5-4 GB) and >= 6 GB free RAM.
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
BASE="--niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --reversion --moe 4 --particles 8"
s1 () { run moeprob_s1991_charged_extrap $BASE $CH --random-seed 1991; run moeprob_s1993_charged_extrap $BASE $CH --random-seed 1993; }
s2 () { sleep 120; run moeprob_s1992_charged_extrap $BASE $CH --random-seed 1992; }
s1 & s2 & wait
echo "MOE+PROB SEQ COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
