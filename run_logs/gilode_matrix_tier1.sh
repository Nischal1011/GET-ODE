#!/bin/bash
set -x
cd /mnt/c/Users/nsubedi/Documents/LG-ODE

# Tier 1: phi given a second hidden layer (was 1) + alpha's LR boost annealed 10x->1x over the
# run (was flat 10x) -- see CHANGES.md Part 12. Aliases suffixed _tier1; v1/v2 logs and
# checkpoints are kept for the full before/after record.
run() {
  local ds=$1 extrap=$2 alias_name=$3
  local extrap_flag=""
  if [ "$extrap" = "True" ]; then extrap_flag="--extrap True"; fi
  .venv/bin/python run_models_gilode.py --data "$ds" $extrap_flag --niters 30 --alias "$alias_name" \
    > "run_logs/${alias_name}.log" 2>&1
  echo "=== $alias_name DONE exit=$? ==="
}

run spring  False "gilode_spring_interp_tier1"
run spring  True  "gilode_spring_extrap_tier1"
run charged False "gilode_charged_interp_tier1"
run charged True  "gilode_charged_extrap_tier1"
run ieee39  False "gilode_ieee39_interp_tier1"
run ieee39  True  "gilode_ieee39_extrap_tier1"

echo "GIL-ODE TIER1 MATRIX COMPLETE"
