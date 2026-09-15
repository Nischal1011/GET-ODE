#!/bin/bash
set -x
cd /mnt/c/Users/nsubedi/Documents/LG-ODE

# v2: alpha warm-started at 0.15 (was 0.0) + 10x LR param group for alpha (lib/gil_ode.py,
# run_models_gilode.py) -- see CHANGES.md for why the v1 hard-zero init stalled alpha's growth
# under long extrapolation horizons. Aliases suffixed _v2 to keep v1 logs/checkpoints for
# before/after comparison.
run() {
  local ds=$1 extrap=$2 alias_name=$3
  local extrap_flag=""
  if [ "$extrap" = "True" ]; then extrap_flag="--extrap True"; fi
  .venv/bin/python run_models_gilode.py --data "$ds" $extrap_flag --niters 30 --alias "$alias_name" \
    > "run_logs/${alias_name}.log" 2>&1
  echo "=== $alias_name DONE exit=$? ==="
}

run spring  False "gilode_spring_interp_v2"
run spring  True  "gilode_spring_extrap_v2"
run charged False "gilode_charged_interp_v2"
run charged True  "gilode_charged_extrap_v2"
run ieee39  False "gilode_ieee39_interp_v2"
run ieee39  True  "gilode_ieee39_extrap_v2"

echo "GIL-ODE V2 MATRIX COMPLETE"
