#!/usr/bin/env bash
# CHANGES.md Part 47: two extra seeds (1994, 1995) on charged-extrap for MoE-4 GIL-ODE and LG-ODE,
# each = the standard 50-epoch run followed by the Part 44 15-epoch annealed continuation.
# Pre-committed: all five seeds are reported whatever the outcome.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/seeds45_driver.log
guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 26000 ]; do sleep 60; done; }
run () { local tag="$1"; shift; guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1; local rc=$?
  echo "=== ${tag} DONE exit=$rc $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; return $rc; }
best () { # best-validation checkpoint of the run whose log is $1, in directory $2
  local id=$(grep -m1 -oE "^Experiment [0-9]+" "$1" | awk '{print $2}')
  ls $2/experiment_${id}_*_charged_0.6_extrap_epoch_*_valmse_*.ckpt | awk -F'valmse_' '{print $2, $0}' | sort -g | head -1 | awk '{print $2}'; }
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True"
GIL="--ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --reversion --moe 4"
gil () { run moe_s$1_charged_extrap run_models_gilode.py $GIL $CH --niters 50 --random-seed $1 || return
  ck=$(best run_logs/moe_s$1_charged_extrap.log experiments_gilode); echo "moe5 $1 $ck" >> run_logs/eval/seeds45_ckpts.txt
  run cont_moe_s$1_charged_extrap run_models_gilode.py $GIL $CH --niters 15 --anneal --random-seed $1 --init-from "$ck" || return
  echo "cont5 $1 $(best run_logs/cont_moe_s$1_charged_extrap.log experiments_gilode)" >> run_logs/eval/seeds45_ckpts.txt; }
lg () { run v5k_corrected_charged_extrap_s$1 run_models_corrected.py $CH --niters 50 --random-seed $1 || return
  ck=$(best run_logs/v5k_corrected_charged_extrap_s$1.log experiments_corrected); echo "lg5 $1 $ck" >> run_logs/eval/seeds45_ckpts.txt
  run cont_lgode_s$1_charged_extrap run_models_corrected.py $CH --niters 15 --anneal --random-seed $1 --init-from "$ck" || return
  echo "lgc5 $1 $(best run_logs/cont_lgode_s$1_charged_extrap.log experiments_corrected)" >> run_logs/eval/seeds45_ckpts.txt; }
gil 1994 &
sleep 120
gil 1995 &
sleep 120
( lg 1994; lg 1995 ) &
wait
echo "SEEDS45 COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
