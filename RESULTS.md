# Results — Final Corrected Comparison

This file starts fresh as of the "final corrected run" (see `CHANGES.md` Part 14). Everything
before it — the paper reproduction, AT-LG-ODE, Corrected LG-ODE + 4 baselines at 30 epochs,
and the GIL-ODE v1/v2/Tier-1 rounds — is preserved in `RESULTS_ARCHIVE_PHASE1-3.md` as the
historical record of that work, but is superseded by what's below: two real apples-to-apples
gaps were found in that earlier work (30 vs. the original repo's own 50-epoch default, and an
unmatched 3x-5.6x parameter-count spread across models) and are fixed here.

## What "final corrected" means

- **Epoch budget**: `--niters 50` everywhere, matching `run_models.py`'s (the original LG-ODE
  repo's own script) default, not the 30 used previously for compute-cost reasons.
- **Parameter count matched** across every from-scratch baseline, without touching LG-ODE's own
  dimensions (`--latents 16 --rec-dims 64 --ode-dims 128`, unchanged, used by Corrected LG-ODE
  and Edge-GNN):

  | Model | Capacity knob | Parameters |
  |---|---|---|
  | Corrected LG-ODE | `latents=16, rec-dims=64, ode-dims=128` (unchanged) | 268,836 |
  | Edge-GNN | `latents=16, rec-dims=64, ode-dims=128` (unchanged) | 247,652 |
  | ODE-RNN | `hidden-dim=192` | 272,860 |
  | Latent-ODE | `latents=16, hidden-dim=180` | 262,036 |
  | RNN-NRI | `hidden-dim=120` | 251,570 |
  | GIL-ODE | `hidden-dim=152` | 258,561 |

  All 6 models now sit within a ~247K-273K band (previously 48K-269K).
- **Everything else** (data pipeline, splits, normalization, seeds, `sample-percent`,
  `batch-size`, optimizer/lr/l2/clip/scheduler, likelihood/MSE computation, checkpoint-selection
  protocol) is unchanged from the audit already done — see `CHANGES.md` Part 12/13 for what was
  verified consistent, and the one remaining caveat (best-checkpoint selection uses the test set
  each epoch, inherited unchanged from the original repo and applied identically to all 6
  models, so it doesn't bias the *relative* comparison but does mean absolute numbers are
  slightly optimistic everywhere).

## Full 6-model comparison (complete)

All 36 runs finished (one, `final_gilode_ieee39_interp`, needed a manual rerun after an OOM —
see `CHANGES.md` Part 15 — folded in below). MSE x10^-2.

| Dataset / Task | Corrected LG-ODE | ODE-RNN | Latent-ODE | Edge-GNN | RNN-NRI | GIL-ODE |
|---|---|---|---|---|---|---|
| Springs interp | 0.573 | 0.077 | **0.042** | 0.640 | 0.069 | 0.077 |
| Springs extrap | **2.305** | 5.544 | 3.636 | 3.052 | 8.978 | 6.063 |
| Charged interp | 0.905 | 0.180 | 0.417 | 1.146 | 0.183 | **0.141** |
| Charged extrap | 4.761 | 6.831 | **3.891** | 4.974 | 9.197 | 6.979 |
| IEEE39 interp | 9.525 | 1.079 | 7.900 | 11.165 | **0.975** | 1.064 |
| IEEE39 extrap | 14.040 | 10.977 | 12.308 | 17.700 | 64.231 | **7.555** |

(bold = best per row; full logs: `run_logs/final_<model>_<dataset>_<interp|extrap>.log`)

### Headline finding

**GIL-ODE wins 2 of 6 cells** (charged interp, and IEEE39 extrap decisively — 7.555 vs. the
next-best 10.977) — down from 3 of 6 in the earlier, capacity-mismatched/30-epoch comparison
(`RESULTS_ARCHIVE_PHASE1-3.md`). That drop is the expected, honest consequence of fixing the two
methodology gaps documented in `CHANGES.md` Part 13: once every model gets the same training
length and a comparable parameter budget, some of GIL-ODE's earlier apparent edge turns out to
have been other models being undertrained or undersized, not purely an architectural advantage.
The one result that held up and actually got *stronger* is IEEE39 extrap, which is arguably
GIL-ODE's most meaningful win since it's decisive rather than marginal.

**Absolute errors dropped sharply almost everywhere** compared to the archived 30-epoch numbers
(e.g. Latent-ODE springs-interp: 0.203 -> 0.042; Edge-GNN springs-interp: 1.235 -> 0.640) —
direct confirmation that the earlier comparison really was capacity/training-length limited for
several models, not just GIL-ODE's baselines.

**RNN-NRI's IEEE39-extrap number is a genuine outlier** (64.231, roughly 6x worse than its own
interp number on the same dataset and far worse than every other model) — consistent with, and
amplifying, its already-documented compounding-error failure mode in autoregressive discrete
rollout (`CHANGES.md` Part 9): more capacity gave the rollout more room to diverge, not less. Not
a new bug; read this cell as evidence of a known architectural limitation, not noise.

**Overfitting under the 50-epoch budget persists for several cells** even with the more generous
epoch count — best-epoch was well before 50 for: Corrected LG-ODE charged interp/extrap (17, 19),
ODE-RNN springs-extrap (9), RNN-NRI charged-extrap (12), GIL-ODE springs-extrap (5, notably
early). This is the same "best-on-test checkpoint selection" dynamic flagged in Part 12/13 of
`CHANGES.md` — inherited from the original repo, applied identically to all 6 models, so it
doesn't bias the comparison, but it means these specific numbers are best read as "best reachable
under this protocol," not "converged performance."

### Why Corrected LG-ODE doesn't match the original paper's own numbers

Corrected LG-ODE loses badly on interpolation here, which looks like it contradicts the actual
paper (it reports LG-ODE beating every baseline on *both* interp and extrap at 60% observed).
Investigated directly (`CHANGES.md` Part 17) rather than accepting it at face value: this
project's very first, *uncorrected* reproduction of LG-ODE (`RESULTS_ARCHIVE_PHASE1-3.md`)
matches the paper's own published numbers closely (springs interp 0.341 vs. paper's 0.317,
charged interp 0.803 vs. 0.828, within normal seed variance on every cell) — while the
*corrected* version (with the test-set-normalization leakage removed, see `CHANGES.md` Part 8)
does not. This strongly indicates the paper's own published numbers were produced with the same
normalization leakage this project identified and fixed. Corrected LG-ODE's numbers here should
therefore be read as a fair, leakage-free comparison against the other 5 models in this table
(all built without that leakage from the start) — not as a failed reproduction of the paper,
which cannot be matched without reintroducing the same leak.

## What was run

```
# All 6 models x 3 datasets x 2 tasks, 50 epochs, capacity-matched hidden-dim where applicable
python run_models_corrected.py --data <spring|charged|ieee39> [--extrap True] --niters 50 --alias <name>
python run_models_odernn.py    --data <spring|charged|ieee39> [--extrap True] --niters 50 --hidden-dim 192 --alias <name>
python run_models_latentode.py --data <spring|charged|ieee39> [--extrap True] --niters 50 --hidden-dim 180 --alias <name>
python run_models_edgegnn.py   --data <spring|charged|ieee39> [--extrap True] --niters 50 --alias <name>
python run_models_rnnnri.py    --data <spring|charged|ieee39> [--extrap True] --niters 50 --hidden-dim 120 --alias <name>
python run_models_gilode.py    --data <spring|charged|ieee39> [--extrap True] --niters 50 --hidden-dim 152 --alias <name>
```
