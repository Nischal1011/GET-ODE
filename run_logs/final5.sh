#!/usr/bin/env bash
# CHANGES.md Part 48: one final configuration on all six cells, five seeds (1991-1995).
#   GIL-ODE: 5/6 baseline flags + --horizon-loss full --reversion --moe 4  (charged-extrap already
#            done in Parts 33 and 47); two workers.
#   Baselines: seeds 1994 and 1995 for every model x cell, commands identical to seeds 1991-1993
#            (LG-ODE charged-extrap already done in Part 47); one worker.
# Resumable: a job whose log already holds a FINAL line is skipped. Stop by PID, never by pattern.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/final5_driver.log
guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 26000 ]; do sleep 60; done; }
run () { local tag="$1"; shift
  grep -q "^FINAL" "run_logs/${tag}.log" 2>/dev/null && return 0
  guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
SPV="--data spring --dataset-dir data/spring_subset --val-fraction 0.166666667"
CHV="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667"
IEV="--data ieee39 --dataset-dir data/processed/ieee39_gen_subset"
dflags () { case $1 in spring) echo "$SPV";; charged) echo "$CHV";; ieee39) echo "$IEV";; esac; }
mflag () { [ "$1" = extrap ] && echo "--extrap True"; }
GIL="run_models_gilode.py --niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --horizon-loss full --reversion --moe 4"

# GIL job list, in priority order: "data mode seed"
: > run_logs/final5_gil_jobs.txt
for cell in "spring extrap" "ieee39 extrap"; do for s in 1991 1992 1993; do echo "$cell $s"; done; done >> run_logs/final5_gil_jobs.txt
for cell in "spring interp" "charged interp" "ieee39 interp"; do for s in 1991 1992 1993; do echo "$cell $s"; done; done >> run_logs/final5_gil_jobs.txt
for cell in "spring extrap" "ieee39 extrap" "spring interp" "charged interp" "ieee39 interp"; do for s in 1994 1995; do echo "$cell $s"; done; done >> run_logs/final5_gil_jobs.txt
gil_worker () { local w=$1 i=0
  while read data mode seed; do i=$((i+1)); [ $((i % 2)) -eq $w ] || continue
    run moe_s${seed}_${data}_${mode} $GIL $(dflags $data) $(mflag $mode) --random-seed $seed
  done < run_logs/final5_gil_jobs.txt; }

base_worker () {
  for mode in extrap interp; do for data in charged spring ieee39; do for s in 1994 1995; do
    D="$(dflags $data) $(mflag $mode) --niters 50 --random-seed $s"
    [ "$data $mode" = "charged extrap" ] || run s5_corrected_${data}_${mode}_s$s run_models_corrected.py $D
    run s5_edgegnn_${data}_${mode}_s$s   run_models_edgegnn.py $D
    run s5_latentode_${data}_${mode}_s$s run_models_latentode.py $D --hidden-dim 180
    run s5_odernn_${data}_${mode}_s$s    run_models_odernn.py $D --hidden-dim 192
    run s5_rnnnri_${data}_${mode}_s$s    run_models_rnnnri.py $D --hidden-dim 120
    run s5_csgode_${data}_${mode}_s$s    run_models_csgode.py $D
  done; done; done; }

gil_worker 1 &
sleep 150
gil_worker 0 &
sleep 150
base_worker &
wait
echo "FINAL5 COMPLETE $(date '+%m-%d %H:%M')" >> $LOG
