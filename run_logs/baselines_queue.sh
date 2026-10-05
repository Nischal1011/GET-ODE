#!/usr/bin/env bash
# The three baselines never run on the subset protocol: Latent-ODE, Edge-GNN, RNN-NRI.
# Seed 1991 to match the existing bars; capacity as in the full-scale runs; commands mirror the
# existing subset bars exactly (springs/charged with --val-fraction 0.166666667, IEEE39 without).
#
# Sequential, running alongside run_logs/overnight.sh -> at most two training jobs at once.
# RAM guard: waits for >= 6 GB available before each launch. Six concurrent jobs (~2.6 GB each)
# exhausted WSL's 15 GB and crashed the VM earlier.
set -u
cd /mnt/c/Users/nsubedi/Documents/LG-ODE
LOG=run_logs/baselines_driver.log
: > "$LOG"

ram_guard () {
  while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 6 ]; do
    echo "    (waiting: <6 GB RAM free $(date '+%H:%M'))" >> "$LOG"; sleep 60
  done
}
run () {  # run <alias> <script> <args...>
  local tag="$1" script="$2"; shift 2; ram_guard; local t0=$(date +%s)
  echo "=== ${tag} START $(date '+%m-%d %H:%M') ===" >> "$LOG"
  .venv/bin/python "$script" "$@" --niters 50 --random-seed 1991 --alias "$tag" > "run_logs/${tag}.log" 2>&1
  echo "=== ${tag} DONE exit=$? $(( ($(date +%s)-t0)/60 ))min ===" >> "$LOG"
}

SPV="--data spring  --dataset-dir data/spring_subset  --val-fraction 0.166666667"
CHV="--data charged --dataset-dir data/charged_subset --val-fraction 0.166666667"
IEV="--data ieee39  --dataset-dir data/processed/ieee39_gen_subset"
LAT="run_models_latentode.py"; EDG="run_models_edgegnn.py"; NRI="run_models_rnnnri.py"
LH="--hidden-dim 180"; NH="--hidden-dim 120"

# 1. springs-interp -- highest risk (full-scale: Latent-ODE 0.014, RNN-NRI 0.026 vs our 0.0654)
run subset_latentode_spring_interp $LAT $SPV $LH
run subset_rnnnri_spring_interp    $NRI $SPV $NH
run subset_edgegnn_spring_interp   $EDG $SPV
# 2-3. IEEE39 -- never checked against these three under the current protocol
run ieee39_subset_latentode_interp $LAT $IEV $LH
run ieee39_subset_rnnnri_interp    $NRI $IEV $NH
run ieee39_subset_edgegnn_interp   $EDG $IEV
run ieee39_subset_latentode_extrap $LAT $IEV $LH --extrap True
run ieee39_subset_rnnnri_extrap    $NRI $IEV $NH --extrap True
run ieee39_subset_edgegnn_extrap   $EDG $IEV --extrap True
# 4. remaining cells
run subset_latentode_spring_extrap $LAT $SPV $LH --extrap True
run subset_rnnnri_spring_extrap    $NRI $SPV $NH --extrap True
run subset_edgegnn_spring_extrap   $EDG $SPV --extrap True
run subset_latentode_charged_interp $LAT $CHV $LH
run subset_rnnnri_charged_interp    $NRI $CHV $NH
run subset_edgegnn_charged_interp   $EDG $CHV
run subset_latentode_charged_extrap $LAT $CHV $LH --extrap True
run subset_rnnnri_charged_extrap    $NRI $CHV $NH --extrap True
run subset_edgegnn_charged_extrap   $EDG $CHV --extrap True
echo "BASELINES QUEUE COMPLETE $(date '+%m-%d %H:%M')" >> "$LOG"
