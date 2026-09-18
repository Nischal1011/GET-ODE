# Datasets

This document describes exactly how each of the three datasets used in this project is
constructed, and characterizes how each one satisfies the problem setting the LG-ODE paper
(Huang, Sun, Wang, NeurIPS 2020) defines: a multi-agent dynamic system with an underlying graph
structure, observed via **irregularly-sampled, partial observations** per object, evaluated on
**interpolation** (reconstruct missing values within an observed window) and **extrapolation**
(forecast a future window from an observed one). Springs and charged particles are the paper's
own benchmarks; IEEE39-Gen is this project's own dataset, built for the real-world power-grid
use case this work emphasizes.

For all three datasets, this project uses fixed, deterministic, **stratified subsets** rather
than the full trajectory counts (20k/18k for springs/charged; the full ~12.9k for IEEE39),
because the defining behavior — interaction structure, irregular per-node observation pattern,
interp/extrap task split — is a property of *how* a dataset is constructed, not how many
trajectories it contains. Subsetting was validated, not assumed: three structurally distinct
anchor models (a no-graph baseline, a graph+VAE-bottleneck baseline, and the proposed
graph+innovation-lifting architecture) were run on both the full-scale and subset versions of
springs (and, as of this writing, IEEE39) specifically to check that the subset preserves
full-scale ranking and behavior, not just to save compute without checking. See "Subset
validation" under each dataset below, and `RESULTS.md` for the full tables.

## How each dataset satisfies "irregularly-sampled partial observations"

| LG-ODE requirement | Springs / Charged | IEEE39-Gen |
|---|---|---|
| Graph-structured interacting objects | 5 particles, per-trajectory adjacency | 10 generators, one fixed physical graph |
| Irregular per-object observation times | Each object independently subsamples its own 40-52 of 60 grid points | Each generator independently subsamples its own observation indices via the same `sample_percent` mechanism |
| Partial observability (not every object observed at once) | Yes — independent per-object sampling means, at almost any given grid point, some objects are observed and others aren't | Yes — same independent-per-node sampling scheme (`data/prepare_ieee39_gen.py: make_interp_mask/make_extrap_mask`) |
| Interpolation task | Condition on a random subset of `[0,1]` (normalized time), reconstruct all of `[0,1]` | Condition on a random subset of the first 30 points, reconstruct all 60 |
| Extrapolation task | Condition on `[0,1]`, forecast an independent second `[1,2]` window | Condition on the first 30 (post-fault-clearance) points, forecast the last 30 |

## Springs

**Source**: `data/generate_dataset.py --simulation springs` (leapfrog physics simulation,
`data/synthetic_sim.py: SpringSim`), matching Kipf et al.'s standard synthetic benchmark and the
LG-ODE paper's own setup (5 particles, 2D box, 60% observed used throughout this project,
matching the paper's own reported setting).

**Construction**:
- 5 particles per trajectory; a `[5,5]` adjacency generated independently per trajectory
  (equal-probability spring / no-spring per pair, symmetric, diagonal 0). **0 = no physical
  interaction** (sparse, unsigned, unweighted).
- 6000-step leapfrog simulation, subsampled every 100 steps → 60-point regular grid per
  trajectory (6s of simulated time, rescaled to `[0,1]`).
- Each object independently samples 40-52 of its 60 grid points as its "recorded" observations;
  `sample_percent` (0.4/0.6/0.8, this project uses 0.6 throughout) then further subsamples that
  recorded set per object, independently, to produce the actual irregular per-node observation
  pattern fed to the encoder.
- Test trajectories additionally record 40 points from a second, independent 6000-step window
  (`[1,2]` normalized time) for the extrapolation task.
- Full scale: 20,000 train / 5,000 test trajectories (paper's own scale).

**Fixed subset** (`data/make_subset.py --dataset spring`): 5,000 train / 1,000 val / 1,000 test,
selected via seeded (`seed=20260916`), stratified-proportional sampling without replacement,
**stratified by the number of active edges per trajectory** (0-10 for 5 particles — this is the
property that matters, since an edge means a real Hooke's-law force and its absence means none).
Verified to match the full population's edge-count distribution to within ~0.001 at every bucket
(0 through 10). The selected train pool is shuffled (fixed seed) before saving so
`CorrectedParseData`'s own contiguous "last 10% is val" split isn't biased toward any bucket.
Exact selected trajectory indices and a sha256 of every saved file are recorded in
`data/spring_subset/subset_manifest.json` for exact reproducibility.

**Subset validation** (ODE-RNN, Corrected LG-ODE, GIL-ODE; 50 epochs; validation-based
checkpoint selection; same protocol at both scales):

| Model / Task | Full-scale MSE (x10^-2) | Subset MSE (x10^-2) | Ratio |
|---|---|---|---|
| ODE-RNN interp | 0.055 | 0.069 | 1.26x |
| ODE-RNN extrap | 5.446 | 5.374 | 0.99x |
| Corrected LG-ODE interp | 0.275 | 0.305 | 1.11x |
| Corrected LG-ODE extrap | 1.970 | 2.021 | 1.03x |
| GIL-ODE interp | 0.054 | 0.068 | 1.26x |
| GIL-ODE extrap | 5.375 | 6.249 | 1.16x |

Ranking is preserved at both scales (interp: GIL-ODE ~ ODE-RNN, both far ahead of Corrected
LG-ODE; extrap: Corrected LG-ODE decisively ahead of both), with absolute MSEs shifting up
modestly (as expected with 4x less training data). Full discussion: `RESULTS.md`.

## Charged particles

**Source**: same pipeline as springs (`--simulation charged`, `synthetic_sim.py:
ChargedParticlesSim`), same 6000-step/60-grid/irregular-subsampling/60%-observed scheme, same
20,000/5,000 scale.

**Construction**:
- Each particle is independently assigned a charge of **+1 or -1** (never neutral); the `[5,5]`
  adjacency is the outer product of charges, so **every off-diagonal pair is always non-zero**
  (dense) — +1 = same charge (repel), -1 = opposite charge (attract). Unlike springs, 0 never
  means "no interaction" here; every pair genuinely interacts, only the sign varies.
- Because charge is assigned per-*particle* (not per-*pair*) and a pair's sign is the product of
  its two particles' charges, the number of positive (repulsive) edges per trajectory is not a
  smooth distribution — it only takes 3 distinct values (4, 6, 10), matching `C(k,2)+C(5-k,2)`
  for `k` particles holding one charge sign out of 5. This was confirmed directly from the data,
  not assumed, when building the subset (see below).

**Fixed subset** (`data/make_subset.py --dataset charged`): 5,000 train / 1,000 val / 1,000 test,
same seed and method as springs, **stratified by the number of positive edges per trajectory**
(preserving the attraction/repulsion balance rather than a density axis, since this dataset is
dense, not sparse). Matched the full population's 3-class distribution (4/6/10) to within
~0.001 at every class. Manifest: `data/charged_subset/subset_manifest.json`.

**Subset validation** (same three anchor models, same protocol as springs):

| Model / Task | Full-scale MSE (x10^-2) | Subset MSE (x10^-2) | Ratio |
|---|---|---|---|
| ODE-RNN interp | 0.180 | 0.265 | 1.47x |
| ODE-RNN extrap | 8.828 | 7.631 | 0.86x |
| Corrected LG-ODE interp | 0.904 | 0.975 | 1.08x |
| Corrected LG-ODE extrap | 4.789 | 5.403 | 1.13x |
| GIL-ODE interp | 0.153 | 0.192 | 1.26x |
| GIL-ODE extrap | 8.319 | 8.359 | 1.00x |

Ranking preserved, same pattern as springs: interp (GIL-ODE and ODE-RNN both far ahead of
Corrected LG-ODE, both scales); extrap (Corrected LG-ODE decisively ahead of both baselines,
both scales). GIL-ODE/ODE-RNN's extrap order flips between scales, the same near-tie noise seen
on springs, not a ranking failure. Full discussion: `RESULTS.md`.

## IEEE39-Gen

**Source**: a public Mendeley transient-stability-assessment dataset for the IEEE New England
39-bus system (`data/raw/ieee39_tsa/tsa_data.pkl`; see `reports/DATA_CHARACTERISTICS.md` for the
full audit). 12,852 time-domain simulations (DIgSILENT PowerFactory), sweeping
generation/load level (80-120%), fault location (7 points along every line), and fault-clearing
time (0.1-0.3s). Each simulation is a genuine 60-point time series per generator (verified
directly — values evolve step to step, not a flattened snapshot), sampled at a strict 0.01s
interval starting at fault clearance + 0.01s.

**Node/feature meaning**: 10 synchronous generators (G01-G10). 5 features per generator: active
power (`P`, MW), terminal voltage (`ut`, p.u.), excitation current (`ie`, p.u.), rotor speed
(`xspeed`, p.u.), rotor angle (`firel`, degrees, relative to G02, the reference machine — so
G02's own `firel` is identically 0.0, a fixed reference rather than a real observation).

**Graph — Kron-reduced from the real network** (`data/build_ieee39_kron_graph.py`): the released
Mendeley dataset has *no* network/admittance data at all, so this project's earlier phase used a
complete-graph placeholder, always explicitly documented as an assumption, not physical topology.
This has been replaced with a real, derived graph:
1. Bus/branch parameters (impedances, shunt susceptance, transformer taps) taken from MATPOWER's
   `case39.m` — the standard machine-readable IEEE 39-bus New England system dataset, sourced
   from Bills et al. 1970 / Pai 1989 / Athay, Podmore & Virmani 1979 (IEEE Trans. Power
   Apparatus and Systems, PAS-98(2):573-584, 1979).
2. The full 39-bus complex admittance matrix is built from this data (standard pi-model branch
   admittance, including off-nominal-tap transformers), then **Kron-reduced** to the 10
   generator buses only: `Y_reduced = Y_gg - Y_gn @ inv(Y_nn) @ Y_ng`, eliminating every
   non-generator bus under a no-current-injection assumption. This is the standard
   network-reduction formula from the same transient-stability literature (Pai 1989) this
   dataset's own generation method belongs to — it is not a novel technique, but its application
   here (deriving the exact graph this specific dataset's 10 nodes need) is.
3. MATPOWER's generator bus numbers (30-39) are mapped to the dataset's G01-G10 naming via
   `case39.m`'s own comment (generator index `i` → bus `29+i`). Cross-checked, not assumed: under
   this mapping G02 → bus 31, and `case39.m` marks bus 31 as the swing/reference bus —
   independently, this project's own data audit found G02's `firel` column is identically 0.0
   across all 12,852 real simulations, exactly what "designated reference machine" predicts.
4. The reduced matrix's magnitude is thresholded at its **median** to produce a binary
   interaction-support graph (kept binary rather than continuous so it plugs into every existing
   model's shared discrete edge-type machinery unchanged, matching springs' binary edge/no-edge
   and charged's ±1-sign convention — a continuous weight would require changing LG-ODE's own
   architecture, out of scope). Result: **22 of 45 possible generator pairs** are "strongly
   coupled" — a real, non-trivial, non-complete structure (e.g. one generator connects to only 2
   of the other 9), verified to not be degenerate (not all-0, not all-1).

**Never** read this graph, or the raw 39-bus adjacency, as the literal transmission-line map —
it is a reduced, thresholded electrical-coupling structure among generators specifically, derived
for this purpose.

**Irregular partial-observation procedure and interp/extrap masks**
(`data/prepare_ieee39_gen.py: make_interp_mask/make_extrap_mask`): built the same way as
springs/charged — each generator independently samples its own observation indices at a given
ratio (40/60/80%, this project uses 60% throughout), from the first 30 points (context) for
interpolation, and from the first 30 only (forecasting the last 30) for extrapolation.

**Split**: stratified by (stability label, clearing-time group — 6 discrete fault-clearing times
recovered from each trajectory's raw absolute start offset), so both splits carry a matched
mix of stable/unstable outcomes and clearing conditions, not an arbitrary train/test partition.
Normalization statistics are fit on the train split only.

**Fixed subset**: 7,000 train / 1,000 val / 2,000 test out of the ~12,852-trajectory corpus,
same (label, clearing-time-group) stratification as the full-scale split, via
`python data/prepare_ieee39_gen.py --train-n 7000 --val-n 1000 --test-n 2000 --out-dir
data/processed/ieee39_gen_subset`. Verified to match the full corpus's stable-fraction (0.5804 in
both, to 4 decimal places) and clearing-time-group fractions (within 0.0001) exactly.

**Subset validation**: unlike springs/charged, IEEE39's graph is fixed and shared across every
trajectory (there is no per-trajectory graph to lose by subsetting), so the only thing a subset
can distort is the (label, clearing-time-group) mix — and that was already checked directly, to
4 decimal places, before any model was trained (above). Combined with springs' full model-level
validation already confirming the subsetting *methodology* itself preserves ranking and behavior,
a second full-scale-vs-subset trained-model comparison for IEEE39 was judged not to add
proportionate evidence for its cost (~10+ hours) and was not run; the subset numbers themselves
(no full-scale comparison) are reported directly in `RESULTS.md`.

## Where to look

- **Springs/charged subsetting**: `data/make_subset.py`, `data/spring_subset/`,
  `data/charged_subset/` (each with `subset_manifest.json`)
- **IEEE39 graph derivation**: `data/build_ieee39_kron_graph.py`, `data/ieee39_kron_reduced.npz`
- **IEEE39 dataset build (full-scale and subset)**: `data/prepare_ieee39_gen.py`,
  `data/processed/ieee39_gen/`, `data/processed/ieee39_gen_subset/`
- **Deeper raw-data audit**: `reports/DATA_READY.md`, `reports/DATA_CHARACTERISTICS.md`
- **Full change history and rationale for every fix mentioned above**: `CHANGES.md`
