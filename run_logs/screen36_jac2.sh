#!/usr/bin/env bash
# Part 36 B rerun after the guard/NaN fix: --jac-reg 0.002 on the 5/6 baseline, 15 epochs,
# charged seed 1991; reference scr36_ref (0.0707). Passing (<= 0.95 x ref) auto-launches 3 seeds.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
guard () {
  while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ] || \
        [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 6000 ]; do sleep 60; done; }
run () { local tag="$1"; shift; guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  local rc=$?; echo "=== ${tag} DONE exit=$rc $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; return $rc; }
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True"
CFG="--ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --jac-reg 0.002"
run scr36_jac2_charged --niters 15 $CFG $CH --random-seed 1991 || exit 1
v=$(grep 'Best val' run_logs/scr36_jac2_charged.log | tail -1 | grep -oE 'mse [0-9.]+' | awk '{print $2}')
REF=$(grep 'Best val' run_logs/scr36_ref_charged.log | tail -1 | grep -oE 'mse [0-9.]+' | awk '{print $2}')
ok=$(python3 -c "print(1 if float('$v') <= 0.95*float('$REF') else 0)")
echo "=== SCREEN36 jac2: best val $v vs ref $REF pass=$ok ===" >> $LOG
[ "$ok" = 1 ] || exit 0
until grep -q "FULL36 COMPLETE" $LOG; do sleep 120; done   # do not overlap the other chain's full runs
for s in 1991 1992 1993; do run full36_jac_s${s}_charged_extrap --niters 50 $CFG $CH --random-seed $s & sleep 120; [ $s = 1992 ] && wait; done; wait
echo "FULL36 JAC COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
