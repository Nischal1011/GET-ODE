#!/bin/bash
set -x
cd /mnt/c/Users/nsubedi/Documents/LG-ODE

run() {
  local script=$1 data=$2 extrap=$3 alias_name=$4
  local extrap_flag=""
  if [ "$extrap" = "True" ]; then extrap_flag="--extrap True"; fi
  .venv/bin/python "$script" --data "$data" $extrap_flag --niters 30 --alias "$alias_name" \
    > "run_logs/${alias_name}.log" 2>&1
  echo "=== $alias_name DONE exit=$? ==="
}

# Fast datasets first (springs, charged), all 4 baselines, then the slow IEEE39 runs last.
for ds in spring charged; do
  for task in interp extrap; do
    extrap_flag="False"; [ "$task" = "extrap" ] && extrap_flag="True"
    run run_models_odernn.py    "$ds" "$extrap_flag" "odernn_${ds}_${task}_60"
    run run_models_latentode.py "$ds" "$extrap_flag" "latentode_${ds}_${task}_60"
    run run_models_edgegnn.py   "$ds" "$extrap_flag" "edgegnn_${ds}_${task}_60"
    run run_models_rnnnri.py    "$ds" "$extrap_flag" "rnnnri_${ds}_${task}_60"
  done
done

for task in interp extrap; do
  extrap_flag="False"; [ "$task" = "extrap" ] && extrap_flag="True"
  run run_models_odernn.py    "ieee39" "$extrap_flag" "odernn_ieee39_${task}_60"
  run run_models_latentode.py "ieee39" "$extrap_flag" "latentode_ieee39_${task}_60"
  run run_models_edgegnn.py   "ieee39" "$extrap_flag" "edgegnn_ieee39_${task}_60"
  run run_models_rnnnri.py    "ieee39" "$extrap_flag" "rnnnri_ieee39_${task}_60"
done

echo "BASELINE MATRIX COMPLETE"
