#!/usr/bin/env bash
# CHANGES.md Part 49: beyond-horizon spread cap (--spread-cap 1.0) on the final mixture config.
# Go/no-go set, in priority order: springs-extrap 1993 (the failed seed), charged-extrap 1991-1993
# (the tie cell), springs-extrap 1991 (a healthy seed must not get worse). Selection on validation.
# Runs alongside final5.sh; polls every 15 s and re-checks RAM 90 s after the guard passes so it does
# not double-launch with a final5 worker (those poll every 60 s). Resumable via FINAL lines.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/final5_driver.log
ram () { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
gpu () { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }
guard () { while true; do
  while [ "$(ram)" -lt 6 ] || [ "$(gpu)" -gt 26000 ]; do sleep 15; done
  sleep 90; [ "$(ram)" -ge 6 ] && [ "$(gpu)" -le 26000 ] && return 0; done; }
GIL="run_models_gilode.py --niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --reversion --moe 4 --spread-cap 1.0 --extrap True"
SPV="--data spring --dataset-dir data/spring_subset --val-fraction 0.166666667"
CHV="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667"
for job in "spring 1993" "charged 1991" "charged 1992" "charged 1993" "spring 1991"; do
  set -- $job; data=$1 seed=$2; tag=spr_s${seed}_${data}_extrap
  grep -q "^FINAL" run_logs/$tag.log 2>/dev/null && continue
  D=$([ $data = spring ] && echo "$SPV" || echo "$CHV")
  guard; t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> $LOG
  .venv/bin/python $GIL $D --random-seed $seed --alias $tag > run_logs/$tag.log 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> $LOG
done
echo "SPREAD TEST COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
