#!/usr/bin/env bash
# CHANGES.md Part 44: 15-epoch cosine-annealed continuation (5e-4 -> 1e-6) from the trained
# charged-extrap checkpoints, same seeds, for MoE-4 GIL-ODE AND LG-ODE (identical extra budget).
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/anneal_cont_driver.log
guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 26000 ]; do sleep 60; done; }
run () { local tag="$1"; shift; guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --niters 15 --anneal"
GIL="--ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --reversion --moe 4"
gil () { run cont_moe_s$1_charged_extrap run_models_gilode.py $GIL $CH --random-seed $1 --init-from "$(grep "^moe $1 " run_logs/eval/stack_ckpts.txt | awk '{print $3}')"; }
lg () { run cont_lgode_s$1_charged_extrap run_models_corrected.py $CH --random-seed $1 --init-from "$(grep "_extrap_" run_logs/eval/baseline_extrap_ckpts.txt | grep "LG-ODE" | grep charged | sed -n "$2p" | awk '{print $3}')"; }
( gil 1991; lg 1991 1 ) &
sleep 120
( gil 1992; lg 1992 2 ) &
wait
gil 1993; lg 1993 3
echo "ANNEAL CONT COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
