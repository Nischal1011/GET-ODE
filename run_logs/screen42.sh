#!/usr/bin/env bash
# CHANGES.md Part 42: AGLA launch adapter screen. Head-only (324 params) 5-epoch fine-tunes from
# the 5/6-baseline checkpoints (every base parameter frozen), full-horizon loss, eps in
# {0.05, 0.1}, seeds 1991/1993. Compared with each checkpoint's own validation MSE. Starts after
# the GN oracle; two jobs at a time.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
until grep -q ALLDONE run_logs/eval/gn_oracle.out 2>/dev/null; do sleep 60; done
guard () {
  while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ] || \
        [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 6000 ]; do sleep 60; done; }
run () { local tag="$1"; shift; guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True"
CFG="--niters 5 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --train-only core.launch --lr 3e-3"
declare -A CK=( [1991]=experiments_gilode/experiment_51931_gilode_charged_0.6_extrap_epoch_40_valmse_0.03504239022731781.ckpt
                [1993]=experiments_gilode/experiment_6450_gilode_charged_0.6_extrap_epoch_47_valmse_0.03346292208880186.ckpt )
n=0
for e in 0.05 0.1; do for s in 1991 1993; do
  run ft42_agla${e}_s${s} $CFG --launch-eps $e --init-from ${CK[$s]} $CH --random-seed $s &
  n=$((n+1)); [ $((n % 2)) = 0 ] && wait; sleep 60
done; done; wait
echo "SCREEN42 COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
