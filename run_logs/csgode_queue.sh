#!/usr/bin/env bash
# CSG-ODE (ICML 2025, reimplemented in lib/csg_ode.py) on all six cells x seeds 1991-1993.
# Same split/epoch/seed commands as every other baseline; paper's Table 8 hyperparameters.
# Three streams; each launch waits for >= 8 GB free RAM.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/csgode_driver.log; : > "$LOG"
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 8 ]; do sleep 60; done; }
run () { local tag="$1"; shift; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_csgode.py "$@" --niters 50 --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
SPV="--data spring  --dataset-dir data/spring_subset  --val-fraction 0.166666667"
CHV="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667"
IEV="--data ieee39  --dataset-dir data/processed/ieee39_gen_subset"
stream () { local s=$1; sleep $(( (s-1991)*90 ))
  run csgode_ieee39_interp_s$s  $IEV                --random-seed $s
  run csgode_ieee39_extrap_s$s  $IEV --extrap True  --random-seed $s
  run csgode_charged_interp_s$s $CHV                --random-seed $s
  run csgode_charged_extrap_s$s $CHV --extrap True  --random-seed $s
  run csgode_spring_interp_s$s  $SPV                --random-seed $s
  run csgode_spring_extrap_s$s  $SPV --extrap True  --random-seed $s
}
stream 1991 & stream 1992 & stream 1993 & wait
echo "CSGODE COMPLETE $(date '+%m-%d %H:%M')" >> "$LOG"
