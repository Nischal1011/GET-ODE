#!/usr/bin/env bash
# CHANGES.md Part 40: null-space forecast jitter (NFJ), paired 5-epoch fine-tune from the 5/6
# baseline checkpoints, full-horizon loss, charged-extrap, seeds 1991/1993. sigma 0 = control.
# Then a deterministic VALIDATION eval of every run's best checkpoint (per-step MSE for fifths).
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
CFG="--niters 5 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full"
declare -A CK=( [1991]=experiments_gilode/experiment_51931_gilode_charged_0.6_extrap_epoch_40_valmse_0.03504239022731781.ckpt
                [1993]=experiments_gilode/experiment_6450_gilode_charged_0.6_extrap_epoch_47_valmse_0.03346292208880186.ckpt )
n=0
for s in 1991 1993; do for sg in 0 0.01 0.03 0.05; do
  run ft40_nfj${sg}_s${s} $CFG --nfj-sigma $sg --init-from ${CK[$s]} $CH --random-seed $s &
  n=$((n+1)); [ $((n % 3)) = 0 ] && wait; sleep 60
done; done; wait
for s in 1991 1993; do for sg in 0 0.01 0.03 0.05; do
  L=run_logs/ft40_nfj${sg}_s${s}.log
  id=$(grep -m1 -oE "Experiment [0-9]+" $L | awk '{print $2}')
  ck=$(ls experiments_gilode/experiment_${id}_* | awk -F'valmse_' '{print $2" "$0}' | sort -g | head -1 | cut -d' ' -f2)
  .venv/bin/python eval_gilode.py --ckpt $ck --split val --tag nfj${sg}_s${s} --out run_logs/eval/nfj_val.jsonl 2>&1 | grep -E "Traceback|Error" >> run_logs/eval/nfj_val.err
done; done
echo "SCREEN40 COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
