#!/usr/bin/env bash
# CHANGES.md Part 38: paired fine-tune screen. Every run continues ALL parameters of the trained
# 5/6-baseline checkpoint for 5 epochs; the control continues with no change. Paired by seed, so
# seed noise mostly cancels. Charged-extrap, seeds 1991 and 1993.
#   D distillation from the 3 baseline seeds (w=1)   E dynamics dropout 0.1   F obs noise 0.01
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/icml_queue_driver.log
guard () {
  while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ] || \
        [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 6000 ]; do sleep 60; done; }
run () { local tag="$1"; shift; guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True"
CFG="--niters 5 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0"
declare -A CK=( [1991]=experiments_gilode/experiment_51931_gilode_charged_0.6_extrap_epoch_40_valmse_0.03504239022731781.ckpt
                [1993]=experiments_gilode/experiment_6450_gilode_charged_0.6_extrap_epoch_47_valmse_0.03346292208880186.ckpt )
TEACH="experiments_gilode/experiment_51931_gilode_charged_0.6_extrap_epoch_40_valmse_0.03504239022731781.ckpt,experiments_gilode/experiment_60073_gilode_charged_0.6_extrap_epoch_46_valmse_0.03752531763166189.ckpt,experiments_gilode/experiment_6450_gilode_charged_0.6_extrap_epoch_47_valmse_0.03346292208880186.ckpt"
declare -A FLAG=( [ctl]="" [dis]="--distill-from $TEACH --distill-weight 1.0" [drop]="--dyn-dropout 0.1" [noise]="--obs-noise 0.01" )
n=0
for s in 1991 1993; do for k in ctl dis drop noise; do
  run ft38_${k}_s${s} $CFG ${FLAG[$k]} --init-from ${CK[$s]} $CH --random-seed $s &
  n=$((n+1)); [ $((n % 3)) = 0 ] && wait; sleep 60
done; done; wait
echo "SCREEN38 COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
