#!/usr/bin/env bash
# CHANGES.md Part 36: three drop-in ideas on the VALIDATED 5/6 baseline (fc_* config, no MoE, no
# reversion head), 15-epoch screen, charged-extrap seed 1991, with a matched baseline reference.
#   A --ema 0.99   B --jac-reg 0.002   C --support-aware
# PASS: best val <= 0.95 x reference best val. Passing ideas launch full 50-epoch charged runs,
# 3 seeds (two at a time). Guards: >= 6 GB RAM and >= 6 GB GPU free per launch.
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
CFG="--ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0"
declare -A FLAG=( [ref]="" [ema]="--ema 0.99" [jac]="--jac-reg 0.002" [sup]="--support-aware" )
( run scr36_ref_charged --niters 15 $CFG $CH --random-seed 1991; run scr36_sup_charged --niters 15 $CFG ${FLAG[sup]} $CH --random-seed 1991 ) &
sleep 120; run scr36_ema_charged --niters 15 $CFG ${FLAG[ema]} $CH --random-seed 1991 &
sleep 120; run scr36_jac_charged --niters 15 $CFG ${FLAG[jac]} $CH --random-seed 1991 &
wait
bv () { grep 'Best val' run_logs/scr36_$1_charged.log | tail -1 | grep -oE 'mse [0-9.]+' | awk '{print $2}'; }
REF=$(bv ref); PASS=()
for k in ema jac sup; do
  v=$(bv $k); ok=$(python3 -c "print(1 if '$v' and '$REF' and float('$v') <= 0.95*float('$REF') else 0)")
  echo "=== SCREEN36 ${k}: best val ${v:-none} vs ref ${REF} pass=${ok} ===" >> "$LOG"
  [ "$ok" = 1 ] && PASS+=("$k")
done
echo "SCREEN36 COMPLETE pass=[${PASS[*]}] $(date '+%m-%d %H:%M')" >> $LOG
i=0
for k in "${PASS[@]}"; do for s in 1991 1992 1993; do
  run full36_${k}_s${s}_charged_extrap --niters 50 $CFG ${FLAG[$k]} $CH --random-seed $s &
  i=$((i+1)); if [ $((i % 2)) = 0 ]; then wait; fi; sleep 120
done; done; wait
echo "FULL36 COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
