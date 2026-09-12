# Data characteristics

Companion to `reports/DATA_READY.md`. Purpose: establish exactly what's shared and what
differs across the three datasets before any architecture decision. No architecture is
proposed here.

## Springs

- **Source / generation**: `data/generate_dataset.py --simulation springs` (leapfrog physics
  simulation, `data/synthetic_sim.py: SpringSim`). 5 particles in a 2D box. 6000-step
  simulation (`delta_T=0.001`), subsampled every 100 steps -> a 60-point regular grid per
  trajectory, then independently irregularly subsampled per object (40-52 of the 60 points for
  training; test additionally samples 40 points from a second 6000-step extrapolation window).
- **Trajectory shape**: per object, a variable-length `[n_obs, 4]` array (irregular across
  objects and across trajectories); `n_balls=5` objects per trajectory, `20,000` train /
  `5,000` test trajectories.
- **Node meaning**: particles. **Feature meaning**: `[x, y, vx, vy]` — 2D position and velocity.
- **Temporal resolution/duration**: 60-point regular grid = 6s of simulated time
  (`6000 * delta_T`); times are rescaled to `[0,1]` in the training pipeline. Train conditions
  on `[0,1]`; extrapolation test conditions on `[0,1]` and forecasts `[1,2]` (a second,
  independent 6s window).
- **Graph construction**: one `[5,5]` adjacency generated *independently per trajectory* via
  `SpringSim.generate_static_graph` — each pair is assigned type 0 (no spring) or 1 (spring)
  with equal probability (the middle "weak spring" type has probability 0 in the sampling
  distribution actually used, so despite `_spring_types=[0, 0.5, 1]` the realized values are
  strictly binary). Symmetric, diagonal is 0.
- **Edge value meaning**: **0 means no physical interaction** — non-adjacent particles exert
  exactly zero spring force on each other. **Sparse**, unsigned, unweighted (binary).
- **Topology behavior**: static within a trajectory; independently re-randomized *between*
  trajectories.
- **Interpolation/extrapolation setup**: fully built and exercised throughout this project
  (`run_models.py --extrap True|False`). Interpolation conditions on a random subset of `[0,1]`
  and reconstructs all of `[0,1]`; extrapolation conditions on `[0,1]` and forecasts `[1,2]`.
- **Irregular partial-observation procedure**: per object, `np.random.choice` picks
  `sample_percent * n_obs` of the recorded points, independently per object
  (`lib/new_dataLoader.py: ParseData.split_data`); `sample_percent` is a CLI flag (`0.4/0.6/0.8`
  used in the paper).
- **Normalization/split**: min-max to `[-1,1]` using combined train+test statistics
  (`ParseData.normalize_features`). Only a train/test split exists in the current pipeline —
  **no validation split**.
- **Limitations/quirks**: none beyond what's already documented in `CHANGES.md` (the NumPy
  2.x compatibility fixes and the `base_models.py` posterior-std stability patch, both fixed).

## Charged particles

- **Source / generation**: same pipeline as springs (`--simulation charged`,
  `synthetic_sim.py: ChargedParticlesSim`), same 6000-step/60-grid/irregular-subsampling
  scheme, same `20,000`/`5,000` split, same `[x,y,vx,vy]` feature meaning, same time scaling.
- **Graph construction**: each particle is independently assigned a charge of **+1 or -1**
  (never 0 — `charge_prob=[0.5, 0, 0.5]`, so the "neutral" charge type has probability 0 despite
  being in `_charge_types`); the `[5,5]` adjacency is the outer product `charges @ charges.T`.
  Verified directly against the generated data: every off-diagonal entry is exactly +1 or -1,
  **never 0** — confirmed dense (no disconnected pairs, ever). The diagonal is always **+1**
  (a charge times itself), unlike springs' diagonal of 0 — this matters, see the quirk below.
- **Edge value meaning**: **every non-self pair physically interacts** (Coulomb attraction or
  repulsion) regardless of the sampled value. +1 = same charge (repel), -1 = opposite charge
  (attract). **Dense**, **signed**. Critically: the *processed* relation label fed to the NRI
  ODE function (`(edges+1)/2` cast to int, giving class 0 or 1) does **not** mean "no edge" for
  class 0 — it means "attractive pair." Both classes 0 and 1 correspond to a real, physically
  active interaction; there is no "disconnected" class for this dataset. (Contrast with springs,
  where class 0 genuinely means "no interaction.")
- **Topology behavior**: same as springs — static per trajectory, re-randomized across
  trajectories.
- **Interpolation/extrapolation, irregular-observation procedure, normalization/split**:
  identical mechanics to springs (same shared code path).
- **Limitation / quirk (verified directly, not just suspected)**: the shared temporal
  graph-construction code (`lib/new_dataLoader.py` / `lib/at_new_dataLoader.py`,
  `transfer_one_graph`) builds a node's "same-object" (temporal self-continuity) edges by
  checking `edge[i][i] + 1 == 1`, i.e. it expects the raw adjacency's diagonal to be 0 (true for
  springs). For charged particles the diagonal is always **1** (charge times itself), so
  `edge[i][i] + 1 == 2`, and this check **never fires**. Verified empirically: running the real
  data through `transfer_one_graph`, `edge_same` is **0 for 100% of edges** on charged
  particles, versus **~27%** on springs. In practice this means the temporal encoder (`GTrans`)
  never receives an edge connecting an object's own past observation to its current one for
  charged particles — only cross-object edges — and the "same-object" projection weights
  (`w_k_list_same`, `v_list_same`, etc.) are never exercised for this dataset. This is a
  property of the *existing, unmodified* shared code interacting with this dataset's
  charge-diagonal convention, not something introduced by AT-LG-ODE, and it affects the LG-ODE
  baseline and AT-LG-ODE identically (so it doesn't invalidate the head-to-head comparison in
  `RESULTS.md`), but it does mean charged-particle results shouldn't be read as "the model sees
  the same kind of temporal structure it sees on springs, just with a different graph."

## IEEE 39-bus

- **Source**: `data/raw/ieee39_tsa/tsa_data.pkl` (Mendeley transient-stability-assessment
  dataset; see `data/Dataset for Transient Stability Assessment of IEEE 39-Bus System/` for the
  original zip with `description.txt`). Generated via DIgSILENT PowerFactory + Python
  automation — **not** ANDES. 12,852 time-domain simulations of the IEEE New England 39-bus
  system: generation/load profile swept 80-120% in 5% steps; three-phase short-circuit faults
  applied at 7 locations (0/10/20/50/80/90/100% along every transmission line); fault clearing
  times swept 0.1-0.3s.
- **Verdict on the audit's core question**: **A** — genuine full trajectories, confirmed
  directly (not assumed): `data[i]` is a real `pandas.DataFrame` of shape `(60, 50)` per
  simulation, and the values genuinely evolve step to step (e.g. `G01 P in MW` moves
  724.9 -> 730.0 -> 734.7 -> ... across consecutive rows) — this is **not** a single flattened
  snapshot reshaped to look like a sequence.
- **Trajectory shape**: `(12852, 60, 50)`. **50 = 10 generators x 5 features**, but the column
  order is **feature-major, not generator-major**: columns 0-9 are `G01..G10 P in MW`, 10-19 are
  `G01..G10 ut in p.u.`, 20-29 are `ie`, 30-39 are `xspeed`, 40-49 are `firel`. Reshaping the 50
  columns to `[10, 5]` requires `reshape(5, 10).T`, *not* a naive `reshape(10, 5)` — the latter
  would silently scramble generator identity across features. (This is exactly the trap flagged
  in the original task brief; confirmed by inspecting the actual column names, not assumed.)
- **Node meaning**: the 10 synchronous generators (G01-G10). **Feature meaning**: `P` = active
  power (MW), `ut` = terminal voltage (p.u.), `ie` = excitation current (p.u.), `xspeed` = rotor
  speed (p.u.), `firel` = rotor angle (degrees, **relative to G02**, the designated reference
  machine). Because angles are reference-relative, **G02's own `firel` column is identically
  0.0 for every trajectory** (verified directly) — it carries no information and should be
  treated as a fixed reference, not a real observation, wherever it's used as a feature.
- **Temporal resolution/duration**: 60 points at a strict 0.01s interval (verified: `diff` is
  exactly `0.01` throughout, no gaps). Absolute timestamps are **not** aligned across all
  12,852 samples — there are exactly **6 distinct absolute-time windows** (e.g. `0.11-0.70`,
  `0.16-0.75`, ..., `0.31-0.90`), each internally regular. Per the dataset description, sampling
  starts at fault clearance + 0.01s and runs 0.6s afterward, so the 6 windows correspond to the
  6 fault-clearing times actually used (the varying absolute offset directly encodes clearing
  time). **Practical implication**: build the shared `times` array as *time-since-clearance*
  (i.e. re-baseline every trajectory to start at 0.01), not as the raw absolute index — that
  recovers a single common 60-point grid across all samples and simultaneously recovers a
  legitimate per-sample "clearing time" feature that isn't stored explicitly anywhere else.
- **Missing values**: none found (checked the full first 500 simulations, zero `NaN`s).
- **Generator identities**: preserved and consistent — same 10 `G01..G10` names, same column
  order, in every one of the 12,852 simulations.
- **Graph construction**: **not available in this file at all.** No admittance matrix, no
  reactance/coupling values, nothing describing which generators/buses are electrically near
  each other. The IEEE 39-bus network topology is well-known publicly (39 buses, 10 generators,
  34-46 branches depending on source), but a proper **generator-level reduced admittance/
  coupling matrix** (preserving real/imaginary/magnitude, not thresholded to binary) needs to
  either come from an ANDES (or equivalent) power-flow model, which is not present in this repo
  (`data/raw/andes_ieee39/` does not exist, `andes` is not installed), or be sourced/derived
  some other way. **Do not** substitute the raw 39x39 bus adjacency for the 10-node generator
  graph — they are not the same object and the original task brief is explicit about this.
- **Event / disturbance metadata**: **partially preserved.** The binary stability outcome
  (1=stable, 7460 samples; 0=unstable, 5392 samples, ~42% unstable) is present per sample.
  Clearing time is *indirectly* recoverable (6 discrete values, from the absolute-time offset,
  as above). **Fault location and loading-condition percentage are not recoverable per sample**
  from this file — only the aggregate ranges swept across the whole corpus are documented in
  `description.txt`, not the specific values used for any individual simulation.
- **Topology behavior**: the underlying physical system's topology genuinely changes during a
  full event (pre-fault -> faulted -> post-clearance network states differ), but **the saved
  window only covers the post-clearance period** (fault clearance + 0.01s to +0.6s) — within
  that saved window the network topology is presumably static (post-fault-cleared), though nothing in the
  data itself confirms this; the pre-fault and faulted-network states are not part of this file.
- **Interpolation/extrapolation setup, irregular-observation procedure**: **not implemented.**
  The data is currently fully dense/regular (60/60 points, no masking). This would need to be
  built as a separate preprocessing step (per the original task brief's Phase 4), generating
  fixed observation masks at multiple ratios (40/60/80%) reused across models, with the full
  trajectory retained as the reconstruction/forecast target.
- **Normalization/split**: not done. No train/validation/test split exists; would need to be
  created with no leakage across the split (e.g. by held-out simulation ID), and normalization
  statistics fit on the training split only.
- **Other limitations**: single, fixed 60-step window per trajectory (no longer horizon
  available in this file); no separate validation split; per-sample fault-location/loading
  metadata absent; graph entirely absent; only 10 nodes total (much smaller than 39 buses, and
  smaller than springs/charged's node count is irrelevant since it's a different physical
  system, but worth remembering when comparing statistics like "average degree" across
  datasets).

## Comparison table

| Property | Springs | Charged | IEEE 39 |
|---|---|---|---|
| Node meaning | Particles | Particles | Synchronous generators (10 of 39 buses) |
| State features | Position, velocity (x, y, vx, vy) | Position, velocity (x, y, vx, vy) | Active power, terminal voltage, excitation current, rotor speed, rotor angle |
| Interaction support | Sparse (binary, ~50% edge probability) | Dense (every pair interacts, always) | Unknown / not yet available (needs external topology source) |
| Edge semantics | 0 = no force, 1 = spring exists | Always active; sign (±1) = attract/repel, never "off" | Not available — would need a weighted, signed generator-coupling matrix |
| Dynamics | Deterministic Newtonian physics, no external forcing | Deterministic Coulomb physics, no external forcing | Real disturbance-driven (fault + clearing) transient electromechanical dynamics |
| Topology behavior | Static per trajectory, random across trajectories | Static per trajectory, random across trajectories | Genuinely time-varying at the true-physics level (pre/during/post fault), but only the post-clearance window is captured in this file |
| Main modeling challenge | Learning which of many possible sparse graphs governs a given trajectory from partial observations | Learning a dense, signed interaction structure where "no edge" isn't an option | No graph at all yet; only 6 partially-informative clearing-time groups recoverable; fault location/loading are latent; irregular-observation protocol and split still need to be built |
