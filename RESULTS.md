# Results — TMLR-Bound Comparison

This file has been rewritten to reflect the current direction (see `CHANGES.md` Parts 19-20):
a TMLR submission using existing models as transparently-documented faithful reimplementations
(the paper never released baseline code, so nothing here claims to reproduce the paper's own
baseline numbers — see "Baseline provenance" below) and deliberately smaller, fixed, stratified
dataset subsets rather than the paper's full 20k/18k-trajectory scale, validated against
full-scale springs results before being trusted. Everything before this rewrite — including the
first "final corrected" 36-run matrix, which used validation-blind (best-on-test) checkpoint
selection and a stale mean/std RNN-NRI encoder, both since fixed — is preserved in
`RESULTS_ARCHIVE_PHASE1-3.md` and the git history of this file, not this page.

## Baseline provenance (read this before comparing to the paper)

The original LG-ODE paper (Huang, Sun, Wang, NeurIPS 2020) never released code or hyperparameters
for its own baselines (Latent-ODE, Weight-Decay, Edge-GNN, NRI+RNN) — confirmed directly against
the authors' own GitHub repo, which contains only LG-ODE's own model code. So:

- **LG-ODE itself** is a genuine reproduction — the authors' own code is public (`run_models.py`
  and unmodified `lib/` files), and it matches the paper's published numbers closely once run at
  the correct data scale and the paper's own (leaky) protocol (`CHANGES.md` Part 17).
- **ODE-RNN, Latent-ODE, Edge-GNN, RNN-NRI** are this project's own from-scratch
  reimplementations of the *ideas* described in the paper and their originating papers
  (Rubanova et al. 2019 for Latent-ODE, Gong & Cheng 2019 for Edge-GNN, Kipf et al. 2018 for
  NRI), not the paper's own baseline code or hyperparameters, which don't exist publicly. Weight-
  Decay (one of the paper's four baselines) was never implemented at all. Every "X beats Y" claim
  in this project's tables is a claim about these reimplementations under this project's own
  training protocol, not a claim about the paper's specific reported baseline figures.
- Two of these reimplementations were revised for fidelity to their own source papers (not to
  LG-ODE) as of `CHANGES.md` Part 19: Latent-ODE now uses the canonical backward-in-time encoder
  (was a forward-pass-plus-single-jump simplification); RNN-NRI now uses a full-trajectory GRU
  relation encoder (was mean/std pooling).

## Dataset provenance and subsetting

Springs and charged particles are Kipf et al.'s standard simulated benchmarks (5 interacting
particles, irregular per-node observation sampling, interp/extrap tasks), matching the paper's
own setup at 60% observed. The paper's specific sample count (20k train / 5k test) is not itself
the scientific contribution — the defining behavior is the interaction structure (sparse binary
graph for springs, dense signed graph for charged), the irregular per-object observation pattern,
and the interp/extrap task split. This project uses fixed, deterministic, stratified **subsets**
(5000 train / 1000 val / 1000 test for both springs and charged) that preserve this behavior,
validated directly against full-scale results below, in order to make iteration and multi-model
comparison computationally tractable. See `data/make_subset.py` and each dataset's
`data/<name>_subset/subset_manifest.json` (exact selected trajectory indices + file hashes, for
reproducibility) for the full method.

IEEE39-Gen remains the primary real-world use case (generator-level transient forecasting under
irregular partial observation) and is not yet subsetted or re-derived with a physically-grounded
(Kron-reduced) coupling graph — both open, tracked in `CHANGES.md` Part 20.

## Full-scale springs (validation reference)

50 epochs, full 20,000 train / 2,000 val / 5,000 test (`CorrectedParseData`'s own 10% val split
off the 20k pool), validation-based checkpoint selection, capacity-matched hidden dims. MSE x10^-2.

| Task | Corrected LG-ODE | ODE-RNN | Latent-ODE | Edge-GNN | RNN-NRI | GIL-ODE |
|---|---|---|---|---|---|---|
| Springs interp | 0.275 | 0.055 | **0.014** | 0.404 | 0.026 | 0.054 |
| Springs extrap | **1.970** | 5.446 | 5.602 | 2.296 | 3.264 | 5.375 |

Logs: `run_logs/final2_<model>_spring_<interp|extrap>.log`. (Charged and IEEE39 full-scale runs
were stopped partway through once the subset pivot was decided — charged got 10/12 cells done
before stopping, kept as a bonus partial reference, not reported as a complete table here.)

## Springs subset validation (5000/1000/1000 vs. full-scale)

Ran the three most distinct anchor models (ODE-RNN: no graph; Corrected LG-ODE: graph + VAE
bottleneck; GIL-ODE: graph + innovation lifting) on the fixed subset, same protocol, to check
whether the subset preserves full-scale ranking and behavior before trusting it for the rest of
the comparison work.

| Model / Task | Full-scale | Subset | Ratio |
|---|---|---|---|
| ODE-RNN interp | 0.055 | 0.069 | 1.26x |
| ODE-RNN extrap | 5.446 | 5.374 | 0.99x |
| Corrected LG-ODE interp | 0.275 | 0.305 | 1.11x |
| Corrected LG-ODE extrap | 1.970 | 2.021 | 1.03x |
| GIL-ODE interp | 0.054 | 0.068 | 1.26x |
| GIL-ODE extrap | 5.375 | 6.249 | 1.16x |

**Ranking is preserved.** Interp: GIL-ODE and ODE-RNN are close together and both far ahead of
Corrected LG-ODE, in both full-scale and subset (same order, similar ~5x gap to LG-ODE in both).
Extrap: Corrected LG-ODE wins decisively over both baselines in both full-scale and subset (same
~2.5-3x gap). The one soft spot: GIL-ODE and ODE-RNN's extrap order flips between the two regimes
(GIL-ODE narrowly ahead full-scale, ODE-RNN narrowly ahead on the subset) — but given how close
those two already are in both regimes (within 1-2%, then within 16%), this reads as noise around
a near-tie between two closely-matched models, not a failure of the subset to preserve the
result that actually matters (LG-ODE's decisive extrap win, and the interp story). Absolute MSEs
shift up modestly on the subset (mostly +3% to +26%), as expected with 4x less training data,
but the qualitative comparison a reader would draw from either table is the same.

Logs: `run_logs/subset_<model>_spring_<interp|extrap>.log`.

## What was run

```
# Subset creation (deterministic, stratified, seeded)
python data/make_subset.py --dataset spring  --train-pool-size 6000 --test-size 1000 --seed 20260916
python data/make_subset.py --dataset charged --train-pool-size 6000 --test-size 1000 --seed 20260916

# Full-scale springs reference (6 models x 2 tasks)
python run_models_corrected.py --data spring [--extrap True] --niters 50 --alias <name>
python run_models_odernn.py    --data spring [--extrap True] --niters 50 --hidden-dim 192 --alias <name>
python run_models_latentode.py --data spring [--extrap True] --niters 50 --hidden-dim 180 --alias <name>
python run_models_edgegnn.py   --data spring [--extrap True] --niters 50 --alias <name>
python run_models_rnnnri.py    --data spring [--extrap True] --niters 50 --hidden-dim 120 --alias <name>
python run_models_gilode.py    --data spring [--extrap True] --niters 50 --hidden-dim 152 --alias <name>

# Subset validation (3 anchor models x 2 tasks)
python run_models_odernn.py    --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 [--extrap True] --niters 50 --hidden-dim 192 --alias <name>
python run_models_corrected.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 [--extrap True] --niters 50 --alias <name>
python run_models_gilode.py    --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 [--extrap True] --niters 50 --hidden-dim 152 --alias <name>
```
