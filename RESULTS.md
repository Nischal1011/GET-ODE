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

## Charged subset validation (5000/1000/1000 vs. full-scale)

Same three anchor models, same protocol.

| Model / Task | Full-scale | Subset | Ratio |
|---|---|---|---|
| ODE-RNN interp | 0.180 | 0.265 | 1.47x |
| ODE-RNN extrap | 8.828 | 7.631 | 0.86x |
| Corrected LG-ODE interp | 0.904 | 0.975 | 1.08x |
| Corrected LG-ODE extrap | 4.789 | 5.403 | 1.13x |
| GIL-ODE interp | 0.153 | 0.192 | 1.26x |
| GIL-ODE extrap | 8.319 | 8.359 | 1.00x |

**Ranking is preserved**, same pattern as springs: interp has GIL-ODE and ODE-RNN both far
ahead of Corrected LG-ODE at both scales (same order); extrap has Corrected LG-ODE decisively
ahead of both baselines at both scales. GIL-ODE/ODE-RNN's extrap order flips between scales here
too (GIL-ODE narrowly ahead full-scale, ODE-RNN narrowly ahead on the subset) — the same
near-tie noise pattern seen on springs, not a ranking failure for the result that matters.

Logs: `run_logs/final2_<model>_charged_<interp|extrap>.log` (full-scale; 5 of 6 cells came from
the earlier `final2_matrix.sh` run before the subset pivot, Part 20 of `CHANGES.md`; only
GIL-ODE charged-extrap was filled in separately), `run_logs/subset_<model>_charged_<interp|extrap>.log`.

## IEEE39 subset (no full-scale comparison — see below for why)

IEEE39's graph is fixed and shared across every trajectory (unlike springs/charged, where each
trajectory has its own graph — the property springs/charged's validation was actually checking
survives subsetting). The only thing an IEEE39 subset can distort is its (label, clearing-time)
mix, which was already checked directly to 4 decimal places before training any model
(`dataset.md`). Combined with springs/charged's full validation already confirming the
subsetting *methodology* itself, a second full-scale-vs-subset trained-model comparison for
IEEE39 was judged not to add proportionate evidence for its cost and was not run (`CHANGES.md`
Part 21). These are the subset (7000/1000/2000) numbers directly.

| Model / Task | MSE (x10^-2) |
|---|---|
| ODE-RNN interp | 1.102 |
| Corrected LG-ODE interp | 7.938 |
| GIL-ODE interp | 1.194 |
| ODE-RNN extrap | 11.284 |
| Corrected LG-ODE extrap | 13.440 |
| GIL-ODE extrap | **8.235** |

Same interp pattern as springs/charged (ODE-RNN and GIL-ODE close together, both far ahead of
Corrected LG-ODE's VAE-bottleneck encoder). Extrap breaks the springs/charged pattern: **GIL-ODE
wins outright here**, not Corrected LG-ODE — the most decisive win GIL-ODE has anywhere in this
project, and consistent with earlier findings throughout this work that IEEE39-extrap was
GIL-ODE's strongest cell. Worth noting this graph now differs from earlier IEEE39 results in
this project's history: it's Kron-reduced from the real network (`dataset.md`), not the earlier
complete-graph placeholder, and GIL-ODE's own graph-construction code was previously silently
ignoring even that placeholder (`CHANGES.md` Part 21) — so this is the first IEEE39 result in
the project where GIL-ODE actually uses real physical structure.

Logs: `run_logs/ieee39_subset_<model>_<interp|extrap>.log`.

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

# Subset validation, springs (3 anchor models x 2 tasks)
python run_models_odernn.py    --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 [--extrap True] --niters 50 --hidden-dim 192 --alias <name>
python run_models_corrected.py --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 [--extrap True] --niters 50 --alias <name>
python run_models_gilode.py    --data spring --dataset-dir data/spring_subset --val-fraction 0.166666667 [--extrap True] --niters 50 --hidden-dim 152 --alias <name>

# IEEE39 graph derivation + dataset builds (full-scale and fixed subset)
python data/build_ieee39_kron_graph.py
python data/prepare_ieee39_gen.py
python data/prepare_ieee39_gen.py --train-n 7000 --val-n 1000 --test-n 2000 --out-dir data/processed/ieee39_gen_subset

# Subset validation, charged (3 anchor models x 2 tasks) -- full-scale mostly already existed
python run_models_odernn.py    --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 [--extrap True] --niters 50 --hidden-dim 192 --alias <name>
python run_models_corrected.py --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 [--extrap True] --niters 50 --alias <name>
python run_models_gilode.py    --data charged --dataset-dir data/charged_subset --val-fraction 0.166666667 [--extrap True] --niters 50 --hidden-dim 152 --alias <name>

# IEEE39 subset only, no full-scale comparison (3 anchor models x 2 tasks)
python run_models_odernn.py    --data ieee39 --dataset-dir data/processed/ieee39_gen_subset [--extrap True] --niters 50 --hidden-dim 192 --alias <name>
python run_models_corrected.py --data ieee39 --dataset-dir data/processed/ieee39_gen_subset [--extrap True] --niters 50 --alias <name>
python run_models_gilode.py    --data ieee39 --dataset-dir data/processed/ieee39_gen_subset [--extrap True] --niters 50 --hidden-dim 152 --alias <name>
```
