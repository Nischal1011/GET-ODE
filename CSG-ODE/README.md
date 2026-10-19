# CSG-ODE baseline

This directory contains two implementations:

- `lib/baselines/csg_ode/` is the isolated, tested reconstruction used for new baseline sweeps.
- The older files directly under `lib/` and `run_models.py` are the original prototype and were left untouched by this reconstruction.

The reconstruction follows the CSG-ODE paper, the official ControlSynth implementation for missing ODE details, canonical AGCRN for the recurrent gate convention, and LG-ODE for likelihood and observation-ratio conventions. See [REPRODUCTION.md](REPRODUCTION.md) for the audit, equation mapping, assumptions, and current validation status.

## Single run

From this directory:

```bash
export WANDB_MODE=online
export WANDB_PROJECT=at-ode

python run_csg_ode.py \
  --model csg_ode \
  --dataset springs \
  --task interp \
  --obs-ratio 0.6 \
  --config configs/csg_ode/springs.yaml \
  --seed 1991
```

`--device auto` selects CUDA, then Apple MPS, then CPU. PEMS08 uses paper batch size `4`.

The current local PEMS08 file contains flow, time-of-day, and day-of-week rather than the paper's flow, speed, and occupancy. Paper-comparison runs therefore stop by default. Replace the source with verified features and add `../data/real/pems08/raw/feature_semantics.json`, following `configs/csg_ode/pems08_feature_semantics.example.json`; its SHA-256 must match the corrected `data.npz`. `--allow-dataset-mismatch` is only for explicitly labeled plumbing tests, and those results are excluded from aggregation.

## Full sweep

Preview the 72 primary commands without launching them:

```bash
python reproduce_csg_ode.py
```

Run all four datasets, interpolation/extrapolation, 0.4/0.6/0.8 ratios, and seeds 1991/42/7:

```bash
python reproduce_csg_ode.py \
  --execute \
  --device auto \
  --wandb-mode online \
  --wandb-project at-ode
```

The driver preflights PEMS08 before launching any run, writes `sweep_manifest.json`, `aggregate.json`, and `aggregate.csv`, and excludes limited-batch or explicitly mismatched-data runs from aggregation. Until corrected PEMS08 features are installed, run the ready datasets with `--datasets springs,charged,motion_walk`.

Current runnable primary sweep:

```bash
python reproduce_csg_ode.py \
  --datasets springs,charged,motion_walk \
  --tasks interp,extrap \
  --obs-ratios 0.4,0.6,0.8 \
  --seeds 1991,42,7 \
  --variants full \
  --device auto \
  --wandb-mode online \
  --wandb-project at-ode \
  --execute
```

To unblock paper-compatible PEMS08 runs, replace `../data/real/pems08/raw/data.npz` with a verified `[T,170,3]` flow/speed/occupancy source, regenerate the irregular files under `../data/real/pems08/pems08`, and create `feature_semantics.json` from the provided example with the exact raw-file SHA-256 and source identifier. Re-run the data audit before adding `pems08` to the matrix.

## Validation

```bash
python audit_csg_data.py --data-root ../data
pytest -q
```

The expensive one-batch overfit checks are opt-in:

```bash
CSG_RUN_SLOW_TESTS=1 pytest -q -m slow
```
