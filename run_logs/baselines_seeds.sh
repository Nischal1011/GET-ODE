#!/usr/bin/env bash
# Seeds 1992/1993 for every baseline x cell not already seeded (48 runs), so the paper's main table
# has mean +- sd for all five baselines. Commands mirror the seed-1991 runs exactly (same split,
# capacity, epochs). Three parallel streams, balanced by logged runtimes; each launch waits for
# >= 8 GB free RAM (raised from 6: three concurrent Edge-GNN IEEE39 jobs, ~3.7 GB each, left under 3 GB free).
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/baselines_seeds_driver.log; : > "$LOG"
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 8 ]; do sleep 60; done; }
run () { local tag="$1"; shift; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python "$@" --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
stream0 () {
  sleep 0
  run seed_edgegnn_ieee39_interp_s1993 run_models_edgegnn.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --niters 50 --random-seed 1993
  run seed_corrected_ieee39_interp_s1993 run_models_corrected.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --niters 50 --random-seed 1993
  run seed_corrected_ieee39_extrap_s1992 run_models_corrected.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --extrap True --niters 50 --random-seed 1992
  run seed_odernn_ieee39_interp_s1992 run_models_odernn.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --hidden-dim 192 --niters 50 --random-seed 1992
  run v5k_odernn_charged_interp_s1993 run_models_odernn.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --hidden-dim 192 --niters 50 --random-seed 1993
  run v5k_odernn_charged_extrap_s1993 run_models_odernn.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --hidden-dim 192 --niters 50 --random-seed 1993
  run seed_rnnnri_spring_interp_s1992 run_models_rnnnri.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --hidden-dim 120 --niters 50 --random-seed 1992
  run seed_rnnnri_ieee39_extrap_s1993 run_models_rnnnri.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --extrap True --hidden-dim 120 --niters 50 --random-seed 1993
  run seed_rnnnri_charged_extrap_s1992 run_models_rnnnri.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --hidden-dim 120 --niters 50 --random-seed 1992
  run seed_odernn_spring_extrap_s1993 run_models_odernn.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --extrap True --hidden-dim 192 --niters 50 --random-seed 1993
  run seed_latentode_spring_extrap_s1992 run_models_latentode.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --extrap True --hidden-dim 180 --niters 50 --random-seed 1992
  run seed_latentode_charged_interp_s1993 run_models_latentode.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --hidden-dim 180 --niters 50 --random-seed 1993
  run seed_latentode_charged_extrap_s1992 run_models_latentode.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --hidden-dim 180 --niters 50 --random-seed 1992
  run seed_edgegnn_spring_extrap_s1993 run_models_edgegnn.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --extrap True --niters 50 --random-seed 1993
  run seed_edgegnn_charged_interp_s1992 run_models_edgegnn.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --niters 50 --random-seed 1992
  run seed_corrected_spring_interp_s1993 run_models_corrected.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --niters 50 --random-seed 1993
}
stream1 () {
  sleep 90
  run seed_edgegnn_ieee39_interp_s1992 run_models_edgegnn.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --niters 50 --random-seed 1992
  run seed_corrected_ieee39_interp_s1992 run_models_corrected.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --niters 50 --random-seed 1992
  run v5k_corrected_charged_interp_s1993 run_models_corrected.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --niters 50 --random-seed 1993
  run seed_odernn_ieee39_interp_s1993 run_models_odernn.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --hidden-dim 192 --niters 50 --random-seed 1993
  run seed_latentode_ieee39_interp_s1992 run_models_latentode.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --hidden-dim 180 --niters 50 --random-seed 1992
  run v5k_odernn_charged_extrap_s1992 run_models_odernn.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --hidden-dim 192 --niters 50 --random-seed 1992
  run seed_rnnnri_spring_extrap_s1993 run_models_rnnnri.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --extrap True --hidden-dim 120 --niters 50 --random-seed 1993
  run seed_rnnnri_ieee39_extrap_s1992 run_models_rnnnri.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --extrap True --hidden-dim 120 --niters 50 --random-seed 1992
  run seed_odernn_spring_interp_s1993 run_models_odernn.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --hidden-dim 192 --niters 50 --random-seed 1993
  run seed_odernn_spring_extrap_s1992 run_models_odernn.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --extrap True --hidden-dim 192 --niters 50 --random-seed 1992
  run seed_latentode_ieee39_extrap_s1993 run_models_latentode.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --extrap True --hidden-dim 180 --niters 50 --random-seed 1993
  run seed_latentode_charged_interp_s1992 run_models_latentode.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --hidden-dim 180 --niters 50 --random-seed 1992
  run seed_edgegnn_spring_interp_s1993 run_models_edgegnn.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --niters 50 --random-seed 1993
  run seed_edgegnn_spring_extrap_s1992 run_models_edgegnn.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --extrap True --niters 50 --random-seed 1992
  run seed_edgegnn_charged_extrap_s1993 run_models_edgegnn.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --niters 50 --random-seed 1993
  run seed_corrected_spring_interp_s1992 run_models_corrected.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --niters 50 --random-seed 1992
}
stream2 () {
  sleep 180
  run seed_edgegnn_ieee39_extrap_s1993 run_models_edgegnn.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --extrap True --niters 50 --random-seed 1993
  run seed_edgegnn_ieee39_extrap_s1992 run_models_edgegnn.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --extrap True --niters 50 --random-seed 1992
  run seed_corrected_ieee39_extrap_s1993 run_models_corrected.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --extrap True --niters 50 --random-seed 1993
  run v5k_corrected_charged_interp_s1992 run_models_corrected.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --niters 50 --random-seed 1992
  run seed_latentode_ieee39_interp_s1993 run_models_latentode.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --hidden-dim 180 --niters 50 --random-seed 1993
  run v5k_odernn_charged_interp_s1992 run_models_odernn.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --hidden-dim 192 --niters 50 --random-seed 1992
  run seed_rnnnri_spring_interp_s1993 run_models_rnnnri.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --hidden-dim 120 --niters 50 --random-seed 1993
  run seed_rnnnri_spring_extrap_s1992 run_models_rnnnri.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --extrap True --hidden-dim 120 --niters 50 --random-seed 1992
  run seed_rnnnri_charged_extrap_s1993 run_models_rnnnri.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --hidden-dim 120 --niters 50 --random-seed 1993
  run seed_odernn_spring_interp_s1992 run_models_odernn.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --hidden-dim 192 --niters 50 --random-seed 1992
  run seed_latentode_spring_extrap_s1993 run_models_latentode.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --extrap True --hidden-dim 180 --niters 50 --random-seed 1993
  run seed_latentode_ieee39_extrap_s1992 run_models_latentode.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset --extrap True --hidden-dim 180 --niters 50 --random-seed 1992
  run seed_latentode_charged_extrap_s1993 run_models_latentode.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --hidden-dim 180 --niters 50 --random-seed 1993
  run seed_edgegnn_spring_interp_s1992 run_models_edgegnn.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 --niters 50 --random-seed 1992
  run seed_edgegnn_charged_interp_s1993 run_models_edgegnn.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --niters 50 --random-seed 1993
  run seed_edgegnn_charged_extrap_s1992 run_models_edgegnn.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 --extrap True --niters 50 --random-seed 1992
}
stream0 & stream1 & stream2 & wait
echo "BASELINE SEEDS COMPLETE $(date '+%m-%d %H:%M')" >> "$LOG"
