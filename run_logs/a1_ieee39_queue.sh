#!/usr/bin/env bash
# A1 (no lifting, retrained) on IEEE39, the only dataset where lifting is active (lift ratio ~0.28
# vs ~0.003 on springs/charged, run_logs/eval/icml_eval.jsonl). One training stream; RAM guard 8 GB.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 8 ]; do sleep 60; done; }
run () { local tag="$1"; shift; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
IE="--data ieee39 --dataset-dir data/processed/ieee39_gen_subset"
GIL="--niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --ablate no_lift"
for s in 1991 1992 1993; do run a1_s${s}_ieee39_extrap $GIL $IE --extrap True --random-seed $s; done
for s in 1991 1992 1993; do run a1_s${s}_ieee39_interp $GIL $IE --random-seed $s; done
