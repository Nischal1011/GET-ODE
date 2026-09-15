#!/bin/bash
set -x
cd /mnt/c/Users/nsubedi/Documents/LG-ODE

# Final corrected comparison: all 6 model families, 50 epochs (matching the original LG-ODE
# repo's own --niters default, not the 30 used for every prior round -- see CHANGES.md Part 13),
# with the 4 from-scratch baselines (ODE-RNN, Latent-ODE, RNN-NRI, GIL-ODE) capacity-matched to
# Corrected LG-ODE (268,836 params) / Edge-GNN (247,652 params) via --hidden-dim, since LG-ODE's
# own --latents/--rec-dims/--ode-dims are left untouched per the "don't change LG-ODE's
# structure" rule. See CHANGES.md Part 14 for the full rationale and parameter-count table.
run() {
  local script=$1 ds=$2 extrap=$3 alias_name=$4 extra=$5
  local extrap_flag=""
  if [ "$extrap" = "True" ]; then extrap_flag="--extrap True"; fi
  .venv/bin/python "$script" --data "$ds" $extrap_flag --niters 50 $extra --alias "$alias_name" \
    > "run_logs/${alias_name}.log" 2>&1
  echo "=== $alias_name DONE exit=$? ==="
}

for ds in spring charged ieee39; do
  for extrap in False True; do
    tag=$([ "$extrap" = "True" ] && echo extrap || echo interp)
    run run_models_corrected.py  "$ds" "$extrap" "final_corrected_${ds}_${tag}"  ""
    run run_models_odernn.py     "$ds" "$extrap" "final_odernn_${ds}_${tag}"     "--hidden-dim 192"
    run run_models_latentode.py  "$ds" "$extrap" "final_latentode_${ds}_${tag}"  "--hidden-dim 180"
    run run_models_edgegnn.py    "$ds" "$extrap" "final_edgegnn_${ds}_${tag}"    ""
    run run_models_rnnnri.py     "$ds" "$extrap" "final_rnnnri_${ds}_${tag}"     "--hidden-dim 120"
    run run_models_gilode.py     "$ds" "$extrap" "final_gilode_${ds}_${tag}"     "--hidden-dim 152"
  done
done

echo "FINAL MATRIX COMPLETE"
