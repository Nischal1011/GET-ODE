#!/usr/bin/env bash
# CHANGES.md Part 49: beyond-horizon spread cap (--spread-cap 1.0) on the final mixture config.
# Go/no-go set, in priority order: springs-extrap 1993 (the failed seed), charged-extrap 1991-1993
# (the tie cell), springs-extrap 1991 (a healthy seed must not get worse). Selection on validation.
# v2 (slot hand-over): both final5 GIL worker bashes are SIGSTOPped by the caller. When the first of
# their running jobs ends (gone or zombie), the other worker is resumed, the finished worker stays
# paused and this script uses its slot for all five runs, then resumes it. v1 polled for free RAM
# and lost every race to workers that relaunch the moment their own job ends.
# Usage: spread_test.sh JOB_A WORKER_A JOB_B WORKER_B
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
JA=$1 WA=$2 JB=$3 WB=$4
LOG=run_logs/final5_driver.log
alive () { local s; s=$(ps -o stat= -p "$1" 2>/dev/null) || return 1; [[ "$s" != Z* ]]; }
while alive $JA && alive $JB; do sleep 15; done
if alive $JA; then MINE=$WB; kill -CONT $WA; else MINE=$WA; kill -CONT $WB; fi
echo "=== spread test takes the slot of worker $MINE $(date '+%m-%d %H:%M') ===" >> $LOG
GIL="run_models_gilode.py --niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --reversion --moe 4 --spread-cap 1.0 --extrap True"
SPV="--data spring --dataset-dir data/spring_subset --val-fraction 0.166666667"
CHV="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667"
for job in "spring 1993" "charged 1991" "charged 1992" "charged 1993" "spring 1991"; do
  set -- $job; data=$1 seed=$2; tag=spr_s${seed}_${data}_extrap
  grep -q "^FINAL" run_logs/$tag.log 2>/dev/null && continue
  D=$([ $data = spring ] && echo "$SPV" || echo "$CHV")
  t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> $LOG
  .venv/bin/python $GIL $D --random-seed $seed --alias $tag > run_logs/$tag.log 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> $LOG
done
echo "SPREAD TEST COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
kill -CONT $MINE
