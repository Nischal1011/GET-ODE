#!/bin/bash
set -x
cd /mnt/c/Users/nsubedi/Documents/LG-ODE

# Re-runs the 9 extrapolation jobs that a bash variable-shadowing bug in baseline_matrix.sh
# silently ran in interpolation mode instead (run()'s internal `extrap_flag=""` reassignment
# clobbered the outer loop's identically-named global variable, corrupting every call after
# the first -- odernn's extrap runs, called first each iteration, were unaffected and are
# already valid). `local` here prevents the same bug from recurring.
run() {
  local script=$1 data=$2 alias_name=$3
  .venv/bin/python "$script" --data "$data" --extrap True --niters 30 --alias "$alias_name" \
    > "run_logs/${alias_name}.log" 2>&1
  echo "=== $alias_name DONE exit=$? ==="
}

for ds in spring charged; do
  run run_models_latentode.py "$ds" "latentode_${ds}_extrap_60"
  run run_models_edgegnn.py   "$ds" "edgegnn_${ds}_extrap_60"
  run run_models_rnnnri.py    "$ds" "rnnnri_${ds}_extrap_60"
done

run run_models_latentode.py "ieee39" "latentode_ieee39_extrap_60"
run run_models_edgegnn.py   "ieee39" "edgegnn_ieee39_extrap_60"
run run_models_rnnnri.py    "ieee39" "rnnnri_ieee39_extrap_60"

echo "EXTRAP RERUN COMPLETE"
