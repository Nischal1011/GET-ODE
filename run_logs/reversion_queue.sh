#!/usr/bin/env bash
# CHANGES.md Part 29: GIL-ODE + mean-reverting forecast head + full-horizon loss on the three
# extrapolation cells, charged first (go/no-go). Stream 2 starts once the T6 IEEE39 runs (the
# matching "no head" ablation) free their slot. Then resumes A1 on IEEE39. RAM guard 8 GB.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 8 ]; do sleep 60; done; }
run () { local tag="$1"; shift; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
SP="--data spring  --dataset-dir data/spring_subset  --val-fraction 0.166666667"
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667"
IE="--data ieee39  --dataset-dir data/processed/ieee39_gen_subset"
GIL="--niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0"
REV="$GIL --extrap True --horizon-loss full --reversion"
stream1 () {
  run rev_s1991_charged_extrap $REV $CH --random-seed 1991
  run rev_s1992_charged_extrap $REV $CH --random-seed 1992
  run rev_s1993_charged_extrap $REV $CH --random-seed 1993
  run rev_s1991_spring_extrap  $REV $SP --random-seed 1991
  run rev_s1993_spring_extrap  $REV $SP --random-seed 1993
  run rev_s1992_ieee39_extrap  $REV $IE --random-seed 1992
}
stream2 () {
  until grep -q "t6_s1993_ieee39_extrap DONE" $LOG; do sleep 120; done
  run rev_s1992_spring_extrap  $REV $SP --random-seed 1992
  run rev_s1991_ieee39_extrap  $REV $IE --random-seed 1991
  run rev_s1993_ieee39_extrap  $REV $IE --random-seed 1993
  A1="$GIL --ablate no_lift"
  for s in 1991 1992 1993; do run a1_s${s}_ieee39_extrap $A1 $IE --extrap True --random-seed $s; done
  for s in 1991 1992 1993; do run a1_s${s}_ieee39_interp $A1 $IE --random-seed $s; done
}
stream1 & stream2 & wait
echo "REVERSION QUEUE COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
