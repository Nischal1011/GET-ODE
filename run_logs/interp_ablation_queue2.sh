#!/usr/bin/env bash
# Resume of interp_ablation_queue.sh after the Part 31 pause: the runs not yet done.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 8 ]; do sleep 60; done; }
run () { local tag="$1"; shift; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667"
IE="--data ieee39  --dataset-dir data/processed/ieee39_gen_subset"
GIL="--niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --random-seed 1991"
CORE="--niters 50 --random-seed 1991"
s1 () { run a4_s1991_ieee39_interp $GIL --ablate overwrite $IE; run t5_s1991_charged_interp $CORE $CH; }
s2 () { sleep 120; run a4_s1991_charged_interp $GIL --ablate overwrite $CH; run t5_s1991_ieee39_interp $CORE $IE; }
s1 & s2 & wait
echo "INTERP ABLATION QUEUE COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
