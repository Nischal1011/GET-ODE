#!/bin/bash
set -x
cd /mnt/c/Users/nsubedi/Documents/LG-ODE

# Full rerun under the Part 19 fixes: springs now uses the full 20k-graph data/spring (was the
# small 1800-graph data/example_data), checkpoint selection is validation-based everywhere (was
# best-on-test), Latent-ODE uses the canonical backward encoder, RNN-NRI uses a full-trajectory
# GRU relation encoder (was mean/std pooling). Every number in the pre-Part-19 RESULTS.md is
# superseded by this. 40 runs total: 6 models x 3 datasets x interp/extrap (36), plus the
# original/uncorrected run_models.py on spring/charged x interp/extrap (4, run_models.py's own
# default dataset path is left untouched -- explicit --dataset-dir used here instead).
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
    run run_models_corrected.py  "$ds" "$extrap" "final2_corrected_${ds}_${tag}"  ""
    run run_models_odernn.py     "$ds" "$extrap" "final2_odernn_${ds}_${tag}"     "--hidden-dim 192"
    run run_models_latentode.py  "$ds" "$extrap" "final2_latentode_${ds}_${tag}"  "--hidden-dim 180"
    run run_models_edgegnn.py    "$ds" "$extrap" "final2_edgegnn_${ds}_${tag}"    ""
    run run_models_rnnnri.py     "$ds" "$extrap" "final2_rnnnri_${ds}_${tag}"     "--hidden-dim 120"
    run run_models_gilode.py     "$ds" "$extrap" "final2_gilode_${ds}_${tag}"     "--hidden-dim 152"
  done
done

# Original, unmodified LG-ODE (run_models.py) -- own default dataset path left untouched,
# full-scale data reached via the pre-existing --dataset-dir override. IEEE39 not applicable
# (no code path in the original script; the dataset didn't exist in the original paper).
for ds in spring charged; do
  for extrap in False True; do
    tag=$([ "$extrap" = "True" ] && echo extrap || echo interp)
    run run_models.py "$ds" "$extrap" "final2_uncorrected_${ds}_${tag}" "--dataset-dir data/${ds}"
  done
done

echo "FINAL2 MATRIX COMPLETE"
