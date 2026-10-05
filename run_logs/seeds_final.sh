#!/usr/bin/env bash
# Multi-seed confirmation of the final configuration, plus protocol fixes.
#   stream A : GIL final config, seed 1992, all six cells
#   stream B : GIL final config, seed 1993, all six cells
#   stream C : hardest baselines at seeds 1992/1993, and the charged baselines that originally
#              trained on 5,400 trajectories instead of 5,000 (missing --val-fraction), rerun so the
#              training split matches every other run.
# GIL streams wait for the seed-1991 final-config run to finish. Every launch waits for >= 6 GB
# free RAM (WSL has 15 GB; six concurrent jobs crashed it before).
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/seeds_final_driver.log; : > "$LOG"
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 60; done; }
run () { local tag="$1"; shift; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }

SPV="--data spring  --dataset-dir data/spring_subset  --val-fraction 0.166666667"
CHV="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667"
IEV="--data ieee39  --dataset-dir data/processed/ieee39_gen_subset"
GIL="run_models_gilode.py --niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0"

gil_stream () { local s=$1
  until grep -q "FINAL CONFIG COMPLETE" run_logs/final_config_driver.log 2>/dev/null; do sleep 120; done
  sleep $(( (s - 1992) * 300 ))   # stagger the two streams
  run fc_s${s}_spring_interp  $GIL $SPV                --random-seed $s
  run fc_s${s}_spring_extrap  $GIL $SPV --extrap True  --random-seed $s
  run fc_s${s}_charged_extrap $GIL $CHV --extrap True  --random-seed $s
  run fc_s${s}_ieee39_extrap  $GIL $IEV --extrap True  --random-seed $s
  run fc_s${s}_ieee39_interp  $GIL $IEV                --random-seed $s
  run fc_s${s}_charged_interp $GIL $CHV                --random-seed $s
}
baseline_stream () {
  # protocol fixes: charged baselines that trained on 5,400
  for s in 1991 1992 1993; do run v5k_corrected_charged_extrap_s$s run_models_corrected.py $CHV --extrap True --niters 50 --random-seed $s; done
  run v5k_corrected_charged_interp_s1991 run_models_corrected.py $CHV --niters 50 --random-seed 1991
  run v5k_odernn_charged_interp_s1991  run_models_odernn.py $CHV --niters 50 --hidden-dim 192 --random-seed 1991
  run v5k_odernn_charged_extrap_s1991  run_models_odernn.py $CHV --extrap True --niters 50 --hidden-dim 192 --random-seed 1991
  # hardest-baseline seeds (springs-extrap LG-ODE 1992/1993 already exist)
  for s in 1992 1993; do
    run seed_latentode_spring_interp_s$s  run_models_latentode.py $SPV --hidden-dim 180 --niters 50 --random-seed $s
    run seed_rnnnri_charged_interp_s$s    run_models_rnnnri.py   $CHV --hidden-dim 120 --niters 50 --random-seed $s
    run seed_rnnnri_ieee39_interp_s$s     run_models_rnnnri.py   $IEV --hidden-dim 120 --niters 50 --random-seed $s
    run seed_odernn_ieee39_extrap_s$s     run_models_odernn.py   $IEV --extrap True --hidden-dim 192 --niters 50 --random-seed $s
  done
}
gil_stream 1992 & gil_stream 1993 & baseline_stream & wait
echo "SEEDS FINAL COMPLETE $(date '+%m-%d %H:%M')" >> "$LOG"
