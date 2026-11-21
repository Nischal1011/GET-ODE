#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS_SCRIPT="${ROOT_DIR}/submit_polaris_obs_job.pbs"

NITERS="${NITERS:-100}"
OBS_RATIOS=(${OBS_RATIOS:-0.4 0.6 0.8})
MODES=(${MODES:-interp extrap})
DATASET_LABELS=(${DATASET_LABELS:-dataset1 dataset2 dataset3 dataset4})
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/experiments}"
EXTRA_ARGS_BASE="${EXTRA_ARGS_BASE:---save ${SAVE_DIR}}"

printf '%s\n' "Submitting masked Polaris jobs with labels:"
printf '  %s\n' "${DATASET_LABELS[@]}"
printf '%s\n' "Modes: ${MODES[*]}"
printf '%s\n' "Obs ratios: ${OBS_RATIOS[*]}"
printf '%s\n' "PBS script: ${PBS_SCRIPT}"
printf '\n'

for dataset_label in "${DATASET_LABELS[@]}"; do
  for mode in "${MODES[@]}"; do
    for obs_ratio in "${OBS_RATIOS[@]}"; do
      obs_tag="${obs_ratio/./}"
      job_name="obs_${dataset_label}_${mode}_o${obs_tag}"

      qsub \
        -N "${job_name}" \
        -v "DATASET=${dataset_label},MODE=${mode},OBS_RATIO=${obs_ratio},NITERS=${NITERS},EXTRA_ARGS=${EXTRA_ARGS_BASE}" \
        "${PBS_SCRIPT}"
    done
  done
done
