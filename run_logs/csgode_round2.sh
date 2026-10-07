#!/usr/bin/env bash
# CSG-ODE round 2.
#  (1) Faithfulness check: full-scale springs (data/spring, 20k trajectories), 60% observation, Euler
#      -- the paper's own setting -- to compare against its reported Table 1/2 values
#      (interp 0.1440, extrap 1.2969, MSE x1e-2). Same command shape as the full-scale LG-ODE
#      reproduction (final2_corrected_spring_*), which landed near LG-ODE's published numbers.
#  (2) RK4 decoder on all six subset cells x three seeds, removing the solver confound (LG-ODE uses
#      RK4 here; the paper's Table 9 shows CSG-ODE improves with a stronger solver).
# Two streams; each launch waits for >= 8 GB free RAM.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/csgode_round2_driver.log; : > "$LOG"
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 8 ]; do sleep 60; done; }
run () { local tag="$1"; shift; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_csgode.py "$@" --niters 50 --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
SPV="--data spring  --dataset-dir data/spring_subset  --val-fraction 0.166666667"
CHV="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667"
IEV="--data ieee39  --dataset-dir data/processed/ieee39_gen_subset"
R="--csg-solver rk4"
streamA () {
  run csgfull_spring_interp_s1991 --data spring --random-seed 1991
  for s in 1991 1992 1993; do run csgrk4_ieee39_interp_s$s $IEV $R --random-seed $s; done
  for s in 1991 1992 1993; do run csgrk4_charged_interp_s$s $CHV $R --random-seed $s; done
  for s in 1991 1992 1993; do run csgrk4_spring_interp_s$s $SPV $R --random-seed $s; done
}
streamB () { sleep 90
  run csgfull_spring_extrap_s1991 --data spring --extrap True --random-seed 1991
  for s in 1991 1992 1993; do run csgrk4_ieee39_extrap_s$s $IEV $R --extrap True --random-seed $s; done
  for s in 1991 1992 1993; do run csgrk4_charged_extrap_s$s $CHV $R --extrap True --random-seed $s; done
  for s in 1991 1992 1993; do run csgrk4_spring_extrap_s$s $SPV $R --extrap True --random-seed $s; done
}
streamA & streamB & wait
echo "CSGODE ROUND2 COMPLETE $(date '+%m-%d %H:%M')" >> "$LOG"
