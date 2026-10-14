#!/usr/bin/env bash
# CHANGES.md Part 31: decisive test of the train/test clock inconsistency on charged-extrap.
# GIL-ODE (Part 30 deterministic best config: reversion head + full-horizon loss) and Corrected
# LG-ODE, both with --align-time, seed 1991. Two streams; RAM guard 8 GB.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 8 ]; do sleep 60; done; }
run () { local tag="$1"; local script="$2"; shift 2; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python $script "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --align-time --random-seed 1991"
GIL="--niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --reversion"
run align_gil_s1991_charged_extrap run_models_gilode.py $GIL $CH &
sleep 180
run align_lgode_s1991_charged_extrap run_models_corrected.py --niters 50 $CH &
wait
echo "ALIGN TEST COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
