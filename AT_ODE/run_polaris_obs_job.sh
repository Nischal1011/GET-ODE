#!/usr/bin/env bash
set -euo pipefail

AT_ODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET="${DATASET:-spring}"
MODE="${MODE:-interp}"
OBS_RATIO="${OBS_RATIO:-0.4}"
SEED="${SEED:-1991}"
NITERS="${NITERS:-100}"

MOTION_BATCH_SIZE="${MOTION_BATCH_SIZE:-8}"
PEMS08_BATCH_SIZE="${PEMS08_BATCH_SIZE:-4}"
OTHER_BATCH_SIZE="${OTHER_BATCH_SIZE:-256}"

EXTRA_ARGS_STR="${EXTRA_ARGS:-}"
if [[ -n "$EXTRA_ARGS_STR" ]]; then
  read -r -a EXTRA_ARGS <<< "$EXTRA_ARGS_STR"
fi

cd "$AT_ODE_DIR"

public_dataset="$DATASET"

if [[ "$DATASET" == "dataset3" || "$DATASET" == "motion" ]]; then
  batch_size="$MOTION_BATCH_SIZE"
  optimizer="Adam"
elif [[ "$DATASET" == "dataset4" || "$DATASET" == "pems08" ]]; then
  batch_size="$PEMS08_BATCH_SIZE"
  optimizer="AdamW"
else
  batch_size="$OTHER_BATCH_SIZE"
  optimizer="AdamW"
fi

if [[ "$MODE" == "interp" ]]; then
  extrap_flag="False"
elif [[ "$MODE" == "extrap" ]]; then
  extrap_flag="True"
else
  echo "Unknown MODE=$MODE" >&2
  exit 1
fi

alias="ATODE_${public_dataset}_${OBS_RATIO}_${MODE}"

cmd=(
  "$PYTHON_BIN" run_models_gpu.py
  --data "$DATASET"
  --sample-percent-train "$OBS_RATIO"
  --sample-percent-test "$OBS_RATIO"
  --extrap "$extrap_flag"
  --niters "$NITERS"
  --alias "$alias"
  --z0-encoder GTrans
  --latents 16
  --rec-dims 64
  --ode-dims 128
  --n-heads 1
  --batch-size "$batch_size"
  --lr 5e-4
  --dropout 0.2
  --solver rk4
  --optimizer "$optimizer"
  --random-seed "$SEED"
)
if [[ "$MODE" == "extrap" ]]; then
  cmd+=(--extrap_num 40)
fi
if [[ -n "$EXTRA_ARGS_STR" ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "============================================================"
echo "AT_ODE_DIR=$AT_ODE_DIR"
echo "DATASET=$public_dataset MODE=$MODE OBS_RATIO=$OBS_RATIO NITERS=$NITERS"
echo "Command: ${cmd[*]}"
echo "============================================================"

"${cmd[@]}"
