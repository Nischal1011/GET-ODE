#!/usr/bin/env bash
# FINAL GIL-ODE configuration -- identical flags for every cell:
#   subspace anchoring + null prior + error-controlled rk4 + bidirectional smoother (cold-start) +
#   free-running supervision. The smoother engages only where a target has a future observation
#   (interpolation; bit-identical in extrapolation, verified). Free-running acts only in
#   extrapolation training. Springs-extrap (0.411) already ran under an equivalent configuration.
# Each cell is compared against its hardest baseline. Seed 1991. RAM guard: >=6 GB free per launch.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/final_config_driver.log; : > "$LOG"
ram_guard () { while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do sleep 60; done; }
FLAGS="--niters 50 --ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0 --random-seed 1991"
run () { local tag="$1"; shift; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python run_models_gilode.py "$@" $FLAGS --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"; }
SP="--data spring  --dataset-dir data/spring_subset  --val-fraction 0.166666667"
CH="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667"
IE="--data ieee39  --dataset-dir data/processed/ieee39_gen_subset"
run fc_spring_interp  $SP                  # vs Latent-ODE 0.0170   (fast; confirms trained smoother is stable)
run fc_charged_interp $CH                  # vs RNN-NRI 0.2376      (the uncertain cell)
run fc_ieee39_interp  $IE                  # vs RNN-NRI 0.8625
run fc_charged_extrap $CH --extrap True    # vs LG-ODE 5.403 (charged-extrap baselines still pending)
run fc_ieee39_extrap  $IE --extrap True    # vs ODE-RNN 11.284
echo "FINAL CONFIG COMPLETE $(date '+%m-%d %H:%M')" >> "$LOG"
