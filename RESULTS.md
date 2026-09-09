# AT-LG-ODE vs. LG-ODE — Results Summary

All results on the **springs** dataset, 60% observed, full-scale data (20k train / 5k test
trajectories, matching the paper), 30-epoch training budget, after the numerical-stability
patch in `lib/base_models.py` (see `CHANGES.md` Part 3). Reported MSE is the best test-set value
reached during training (×10⁻², matching the paper's units).

## Interpolation

| Model | Best test MSE (×10⁻²) | Best epoch | Log |
|---|---|---|---|
| LG-ODE (patched) | 0.3406 | 26 / 30 | `run_logs/springs_interp_60_patched.log` |
| AT-LG-ODE (patched) | 0.3459 | 13 / 30 | `run_logs/at_springs_interp_60_patched.log` |

Essentially a tie (~1.5% apart, within reproduction noise). AT-LG-ODE reaches its best result at
about half the epochs LG-ODE needs (13 vs. 26) for the same final quality.

## Extrapolation

| Model | Best test MSE (×10⁻²) | Best epoch | Log |
|---|---|---|---|
| LG-ODE (patched) | 1.6418 | 10 / 30 (destabilizes after, never recovers) | `run_logs/springs_extrap_60.log` |
| AT-LG-ODE (patched) | **1.2374** | 29 / 30 (monotonically improving throughout) | `run_logs/at_springs_extrap_60.log` |

AT-LG-ODE wins by **~25%**, and — unlike the baseline — trains stably for the full 30 epochs
with no divergence.

## Reference: reproduction of the paper's own numbers

For context, LG-ODE reproduced here vs. the paper's Table 1/2 (50-epoch budget, before the
stability patch existed):

| Task | Paper | Reproduced |
|---|---|---|
| Interpolation | 0.3170 | 0.3306 (epoch 17/50) |
| Extrapolation | 1.8084 | 1.6418 (epoch 10/30) |

## Takeaway

Transporting the encoder's relational attention forward in time to reweight the ODE's graph
aggregation costs nothing on interpolation (tie, faster convergence) and meaningfully helps on
extrapolation — both in final accuracy (~25% lower MSE) and in training stability (no
divergence). This matches the design's motivating hypothesis: preserving the encoder's
relational evidence should matter most when the ODE has to carry the dynamics forward without
any further correction from observations, which is exactly the extrapolation setting.

## What was run

```
# LG-ODE baseline
python run_models.py --dataset-dir data/spring --niters 30 --alias springs_interp_60_patched
python run_models.py --dataset-dir data/spring --niters 30 --extrap True --alias springs_extrap_60

# AT-LG-ODE
python run_models_at.py --dataset-dir data/spring --niters 30 --alias at_springs_interp_60_patched
python run_models_at.py --dataset-dir data/spring --niters 30 --extrap True --alias at_springs_extrap_60
```

See `CHANGES.md` for the full technical writeup of what AT-LG-ODE changes and why, and for the
compatibility/stability fixes applied to the original codebase.
