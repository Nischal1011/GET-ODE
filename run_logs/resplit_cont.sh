#!/usr/bin/env bash
# CHANGES.md Part 46: paired test of horizon-randomised training (--resplit). Identical to
# run_logs/anneal_cont.sh (15 annealed epochs from each MoE-4 checkpoint) plus --resplit, so the
# cont_moe_* runs are the exact controls.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/anneal_cont_driver.log
guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 26000 ]; do sleep 60; done; }
run () { local tag="$1"; shift; guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --niters 15 --anneal --resplit"
GIL="--ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --reversion --moe 4"
gil () { run rs_moe_s$1_charged_extrap run_models_gilode.py $GIL $CH --random-seed $1 --init-from "$(grep "^moe $1 " run_logs/eval/stack_ckpts.txt | awk '{print $3}')"; }
gil 1991 &
sleep 120
gil 1992 &
wait
gil 1993
echo "RESPLIT CONT COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
