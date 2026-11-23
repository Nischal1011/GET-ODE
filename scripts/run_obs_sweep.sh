#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AT_ODE_DIR="${AT_ODE_DIR:-$ROOT_DIR/AT_ODE}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET="${DATASET:-spring}"
MODE="${MODE:-interp}"
OBS_RATIO="${OBS_RATIO:-0.4}"

SEED="${SEED:-1991}"
MOTION_BATCH_SIZE="${MOTION_BATCH_SIZE:-8}"
PEMS08_BATCH_SIZE="${PEMS08_BATCH_SIZE:-4}"
OTHER_BATCH_SIZE="${OTHER_BATCH_SIZE:-256}"
NITERS="${NITERS:-100}"

# Extra AT-ODE args can be passed through here, for example:
#   EXTRA_ARGS="--solver rk4 --lr 5e-4"
EXTRA_ARGS_STR="${EXTRA_ARGS:-}"
read -r -a EXTRA_ARGS <<< "${EXTRA_ARGS_STR}"

cd "$AT_ODE_DIR"

echo "AT_ODE_DIR=$AT_ODE_DIR"
echo "DATASET=$DATASET"
echo "MODE=$MODE"
echo "OBS_RATIO=$OBS_RATIO"
echo "PYTHON_BIN=$PYTHON_BIN"

if [[ "$DATASET" == "motion" ]]; then
  batch_size="$MOTION_BATCH_SIZE"
  optimizer="Adam"
elif [[ "$DATASET" == "pems08" ]]; then
  batch_size="$PEMS08_BATCH_SIZE"
  optimizer="AdamW"
else
  batch_size="$OTHER_BATCH_SIZE"
  optimizer="AdamW"
fi

if [[ "$MODE" == "interp" ]]; then
  extrap_flag="False"
  extrap_args=()
elif [[ "$MODE" == "extrap" ]]; then
  extrap_flag="True"
  extrap_args=(--extrap_num 40)
else
  echo "Unknown mode: $MODE" >&2
  exit 1
fi

alias="ATODE_${DATASET}_${OBS_RATIO}_${MODE}"
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
cmd+=("${extrap_args[@]}")
cmd+=("${EXTRA_ARGS[@]}")

echo
echo "============================================================"
echo "Running dataset=$DATASET mode=$MODE obs=$OBS_RATIO niters=$NITERS"
echo "Command: ${cmd[*]}"
echo "============================================================"
"${cmd[@]}"
