#!/bin/bash
set -x
cd /mnt/c/Users/nsubedi/Documents/LG-ODE

# `local` on every variable inside run() -- a bash variable-shadowing bug in the earlier
# baseline_matrix.sh (see CHANGES.md Part 10) silently corrupted 9 of 24 runs by omitting this.
run() {
  local ds=$1 extrap=$2 alias_name=$3
  local extrap_flag=""
  if [ "$extrap" = "True" ]; then extrap_flag="--extrap True"; fi
  .venv/bin/python run_models_gilode.py --data "$ds" $extrap_flag --niters 30 --alias "$alias_name" \
    > "run_logs/${alias_name}.log" 2>&1
  echo "=== $alias_name DONE exit=$? ==="
}

run spring  False "gilode_spring_interp_60"
run spring  True  "gilode_spring_extrap_60"
run charged False "gilode_charged_interp_60"
run charged True  "gilode_charged_extrap_60"
run ieee39  False "gilode_ieee39_interp_60"
run ieee39  True  "gilode_ieee39_extrap_60"

echo "GIL-ODE MATRIX COMPLETE"
