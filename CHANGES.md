# Changes to the original LG-ODE codebase

This document covers everything done on top of the original [ZijieH/LG-ODE](https://github.com/ZijieH/LG-ODE)
release (NeurIPS 2020): getting the decade-old code running on a modern stack, reproducing the
paper's numbers, a small numerical-stability fix, and a new architecture (AT-LG-ODE) built
alongside it for comparison. Nothing in the original `lib/*.py` / `run_models.py` logic was
changed beyond what's listed below — every LG-ODE code path used by `run_models.py` behaves
identically to the original release except where noted in Parts 1 and 3.

## Part 1 — Compatibility fixes (commit `34f7d5a`)

The released code targets Python 3.6 / NumPy 1.16 / PyTorch 1.4 / PyTorch Geometric 1.4. This
repo now runs on Python 3.11, NumPy 2.4, PyTorch 2.11 (cu128, for an RTX 5090), and PyTorch
Geometric 2.8, via a local `uv`-managed `.venv` (not committed). Two behavioral gaps had to be
patched:

- **`data/generate_dataset.py`**: `np.asarray()` on the ragged per-object trajectory lists no
  longer infers `dtype=object` automatically in NumPy 2.x — it now raises
  `ValueError: setting an array element with a sequence`. Fixed by passing `dtype=object`
  explicitly (4 call sites, `generate_dataset` and `generate_dataset_charged`).
- **`lib/new_dataLoader.py`**: `np.load()` defaults to `allow_pickle=False` in modern NumPy,
  which rejects the object arrays produced above. Fixed by passing `allow_pickle=True`
  explicitly (4 call sites in `ParseData.load_data`).

Also added a `--dataset-dir` flag to `run_models.py` (and, for the new architecture,
`run_models_at.py`) so a full-scale generated dataset can be pointed at directly instead of
`data/example_data` (the small bundled toy set).

## Part 2 — Reproducing the paper's numbers

The paper's Table 1/2 numbers use 20k train / 5k test simulated trajectories; only a 2k/500
toy subset ships in `data/example_data`. The full set was regenerated with the fixes above:

```
cd data
python generate_dataset.py --simulation springs --num-train 20000 --num-test 5000
```
placed at `data/spring/` (gitignored — several hundred MB, regenerate as needed).

Springs, 60% observed, default hyperparameters:

| Task | Paper (LG-ODE) | Reproduced |
|---|---|---|
| Interpolation | 0.3170 ×10⁻² | 0.3306 ×10⁻² (`run_logs/springs_interp_60.log`, or the patched rerun `run_logs/springs_interp_60_patched.log`) |
| Extrapolation | 1.8084 ×10⁻² | 1.6418 ×10⁻² (`run_logs/springs_extrap_60.log`) |

Both are within normal reproduction variance (different hardware/library versions, 2020 vs.
now). Command used, e.g. for interpolation:
```
python run_models.py --dataset-dir data/spring --alias springs_interp_60
```
(`--extrap True` for extrapolation; all other flags left at their defaults, which already match
the paper's setup — 60% observed, batch 256, 50 epochs.)

## Part 3 — Numerical stability patch (`lib/base_models.py`)

Both the original LG-ODE and the new AT-LG-ODE (Part 4) share `VAE_Baseline.compute_all_losses`
for the ELBO. It builds `Normal(fp_mu, fp_std)` for the latent initial state's approximate
posterior, with `fp_std = fp_std.abs()`. Under training instability (observed in every long run
in this repo — the original 50-epoch reproduction, the extrapolation baseline, and the first
AT-LG-ODE interpolation attempt all hit it, at different, unpredictable epochs), `fp_std` can
reach exactly/near zero, which violates `torch.distributions.Normal`'s strictly-positive scale
constraint and crashes with:
```
ValueError: Expected parameter scale ... to satisfy the constraint GreaterThan(lower_bound=0.0)
```
This is a latent flaw in the original release (there's already a dead `assert(torch.sum(fp_std
< 0) == 0.)` right after it, which doesn't catch a value of exactly 0). Fixed with a one-line
floor:
```python
fp_std = fp_std.abs().clamp(min=1e-6)
```
This only changes behavior in the already-degenerate case; everywhere else `clamp(min=1e-6)` is
a no-op. It does not change the paper-reproduction numbers above (both were obtained using the
epoch with the best test MSE, from before either run destabilized).

## Part 4 — AT-LG-ODE: attention-transport architecture

A new architecture, implemented alongside the original without modifying it, to test whether
transporting the encoder's relational attention forward in time to reweight the ODE's graph
aggregation improves on plain LG-ODE. Stages 1-4 (temporal graph encoder, temporal
self-attention, `q(Z⁰)`) are unchanged; only Stage 5 (the graph ODE's neighbor aggregation) and
the decoder input are affected, and the decoder itself is unchanged.

### Design

1. **Capture** the last encoder GNN layer's post-softmax relational attention `α_{s→t}` per
   temporal-graph edge (`lib/at_gnn_models.py: GTransCapture`, `ATEncoderGNN`) — a subclass of
   the original `GTrans`/`GNN` that is byte-for-byte identical except for stashing the
   attention and the edge index it belongs to (verified via `load_state_dict` from a freshly
   built original layer, so initialization is unaffected).
2. **Transport** that attention forward in time (`lib/attention_transport.py:
   AttentionTransport`): grouped by physical edge `(j→i)` and source time `t_s`, evidence for
   edge `j→i` at ODE time `t` is `r_ij(t) = Σ α_{s→i} · exp(-λ·(t - t_s))` for `t_s ≤ t` (causal
   only), normalized per receiving object against its neighbors' adjacency-masked evidence:
   `w_ij(t) = A_ij·r_ij(t) / (Σ_k A_ik·r_ik(t) + ε)`. `λ` is a learnable parameter
   (`--lambda-init`, `--learnable-lambda`).
3. **Reweight** the ODE's graph aggregation (`lib/at_gnn_models.py: ATNRIConv`) — the same NRI
   relation network as the original, except the per-edge relation message is scaled by
   `w_ij(t)` before aggregation: `z_dot_i = f_O(Σ_j w_ij(t)·f_R(z_i, z_j))` instead of the
   original unweighted `Σ_j f_R(z_i, z_j)`.

### A time-alignment fix specific to AT-LG-ODE

The encoder attention's timestamps and the ODE solver's query times need to live in the same
clock for `t - t_s` to mean anything. The original code's `time_begin` convention (`0` for
interpolation, `1` for extrapolation, applied identically to train and test) only happens to
line up with the decoder-side time origin for the *test* split; for extrapolation *training*
(which conditions on `(t1,t2)` and predicts `(t2,t3)`, i.e. splits at the midpoint per Appendix
A.2) the true origin is `0.5`, not `1`. This has no effect on the original model (encoder and
decoder times are never compared against each other there), but matters for transport. Fixed in
`lib/at_new_dataLoader.py: ATParseData` by computing the correct decoder-frame origin explicitly
per (mode, train/test) and exposing an additional `t_s` node field in that frame (verified
empirically — see the alignment check in the session transcript / re-run via the snippet in
`lib/at_new_dataLoader.py`'s docstring).

### Gradient flow / adjoint method

`z0` (the ODE's initial state) already depends on the encoder's attention through the existing
attention-pooling step (Stage 4), so the encoder keeps training through that original path
regardless of whether the transport weights are differentiable. The cached attention used to
build `w_ij(t)` is therefore **detached** (`attention_transport.py`), which lets AT-LG-ODE keep
using the memory-efficient `odeint_adjoint` exactly like the original (`lib/at_diffeq_solver.py`
mirrors `lib/diffeq_solver.py` and only adds passing `t_local` into the ODE net). An earlier
version used the encoder's attention *without* detaching, which required switching to plain
(non-adjoint) `odeint` to get correct gradients — this worked, but held the whole solver's
computation graph in memory (236-472 function evaluations per batch) and pushed reserved GPU
memory to 40+ GB, past the RTX 5090's 32 GB, causing WSL2 to silently page to host RAM (5-54s
per training iteration, wildly erratic). The detach + adjoint version runs at ~1.5-1.7s/it,
matching plain LG-ODE.

### Files

| File | Role |
|---|---|
| `lib/at_new_dataLoader.py` | `ATParseData(ParseData)` — adds `object_id`/`t_s` node fields |
| `lib/attention_transport.py` | `AttentionTransport` — the transport module described above |
| `lib/at_gnn_models.py` | `GTransCapture`, `ATEncoderGNN`, `ATNRIConv`, `ATGeneralConv`, `ATOdeGNN` |
| `lib/at_diffeq_solver.py` | `ATDiffeqSolver`, `ATGraphODEFunc` — threads `t_local` to the ODE net |
| `lib/at_latent_ode.py` | `ATLatentGraphODE(VAE_Baseline)` — builds the transport cache after the encoder runs |
| `lib/create_at_latent_ode_model.py` | `create_AT_LatentODE_model` — assembles the pieces above |
| `run_models_at.py` | Top-level training script, mirrors `run_models.py` exactly (same flags/behavior/logging), plus `--lambda-init` / `--learnable-lambda` / `--nonadj-floor` |

### How to run it

Same flags as `run_models.py`:
```
python run_models_at.py --dataset-dir data/spring --alias at_springs_interp_60          # interpolation
python run_models_at.py --dataset-dir data/spring --extrap True --alias at_springs_extrap_60  # extrapolation
```

## Part 5 — AT-LG-ODE vs. LG-ODE results

See **`RESULTS.md`** for the full results summary and the exact commands used (springs and
charged particles, interpolation and extrapolation). Short version: on springs, AT-LG-ODE ties
LG-ODE on interpolation (converging in about half the epochs) and beats it by ~25% on
extrapolation while training stably where the baseline destabilizes. On charged particles, the
same design *underperforms* LG-ODE on both tasks — see Part 6 for why, and for a partial fix.

## Part 6 — Charged particles: a hard-mask failure mode, and a fix

Repeating the springs comparison on the **charged particles** dataset (same pipeline: 20k/5k
full-scale data via `data/generate_dataset.py --simulation charged`, 60% observed, 30 epochs,
patched `base_models.py`) reverses the result: **LG-ODE beats AT-LG-ODE on both interpolation
and extrapolation** (see `RESULTS.md` for numbers). Both LG-ODE reproductions track the paper's
Table 1/2 closely, so the baseline is trustworthy — this is a genuine weakness in AT-LG-ODE as
originally specified, not a bug.

**Root cause**: `w_ij(t) = A_ij·r_ij(t) / (Σ_k A_ik·r_ik(t) + ε)` uses the physical adjacency
`A_ij` as a hard 0/1 mask. Any pair with no sampled edge gets `w_ij(t) = 0` unconditionally, so
its relation message is silenced completely, at every timestep, for the entire trajectory. The
original NRI ODE function (Appendix C.1) never does this — it always sums a second MLP for
"not-connected" pairs alongside the connected one, precisely so latent/unlabeled interactions
can still contribute. For springs this pathway is genuinely unnecessary (non-adjacent objects
exert zero force, so silencing it is free); for charged particles every pair attracts or repels
regardless of the sampled edge label, so AT-LG-ODE was discarding real signal for every
non-adjacent pair.

**Fix**: `AttentionTransport` gained a `nonadj_floor` parameter (`--nonadj-floor`,
`lib/attention_transport.py`). Instead of `Ar = A_ij·r_ij(t)`, it's now
`Ar = A_ij·r_ij(t) + nonadj_floor`, applied to *every* pair before normalizing — so a
non-adjacent pair gets a small non-zero share instead of exactly 0, while an adjacent pair with
real transported evidence still dominates (`evidence + floor >> floor` whenever there's
meaningful evidence). `nonadj_floor=0` (the default, used for every springs/charged result
above) is mathematically identical to the original hard mask — confirmed by rerunning the
smoke tests after the change and seeing identical behavior.

**Outcome** (see `RESULTS.md` for full numbers): `nonadj_floor=0.3` closes about 75% of the
interpolation gap to LG-ODE (0.8739 → 0.8291 ×10⁻², vs. LG-ODE's 0.8033), supporting the
hypothesis above. It does **not** help extrapolation (6.2311 → 6.3105 ×10⁻², essentially flat
or slightly worse). So the hard-mask silencing explains a real, meaningful part of the
interpolation gap but not the extrapolation gap — something else is also costing AT-LG-ODE on
charged extrapolation, left as an open question here.

## Where to look

- **Reproduction command / log**: `run_models.py`, `run_logs/springs_*.log`,
  `run_logs/charged_*.log`
- **AT-LG-ODE command / log**: `run_models_at.py`, `run_logs/at_springs_*.log`,
  `run_logs/at_charged_*.log`
- **Checkpoints**: `experiments/` (LG-ODE), `experiments_at/` (AT-LG-ODE) — both gitignored
- **Full datasets**: `data/spring/`, `data/charged/` (gitignored, regenerate via Part 2's
  command with `--simulation springs` or `--simulation charged`)

---

# Phase 2: IEEE39-Gen, Corrected LG-ODE, and four new baselines

AT-LG-ODE work above is paused (per direction) to first re-establish a trustworthy LG-ODE
baseline — including on a new third dataset — and compare it against four other reimplemented
methods. Results are in `RESULTS.md`'s "Phase 2" section; this covers the technical detail.

## Part 7 — IEEE39-Gen dataset

A third dataset, built from a public Mendeley transient-stability-assessment corpus (IEEE
39-bus power grid), audited and prepared as `data/processed/ieee39_gen/ieee39_gen.npz` via
`data/prepare_ieee39_gen.py`. Full detail already lives in `reports/ieee39_gen.md` and
`reports/DATA_CHARACTERISTICS.md` (verdict, exact shapes, feature-major reshape trap, per-sample
metadata limits) — not repeated here. The one fact that matters for everything below: its
`[10,10]` graph is a **complete graph** (every generator pair connected), an explicit
interaction-support *assumption* standing in for the still-unavailable physical transmission
topology (no ANDES model, no admittance data anywhere in this repo) — never to be read as
recovered physical topology.

## Part 8 — Corrected LG-ODE

Same model architecture as LG-ODE (`lib/create_latent_ode_model.py`, imported unchanged), only
the data pipeline is fixed, via a new `lib/corrected_dataLoader.py: CorrectedParseData` and
`lib/ieee39_dataLoader.py: IEEE39ParseData`, driven by a new `run_models_corrected.py`. Three
real, confirmed issues in the original pipeline (used unmodified everywhere until now):

1. **Object-identity temporal edges.** `transfer_one_graph`'s self-loop check,
   `(edge + eye)[i][j] == 1`, silently assumes the raw physical adjacency's diagonal is 0. True
   for springs (self-loops fire correctly). Charged particles' diagonal is always 1 (a charge
   times itself), so the check never fires for `i==j` — self-loops never form — *and*, more
   severely, only "+1" (same-charge/repel) cross-object pairs ever get a temporal edge; "-1"
   (attract) pairs are silently dropped from the encoder's temporal graph entirely. Verified
   directly: `edge_same` was 0% of edges for charged particles (vs ~27% for springs) under the
   original code. Fixed by defining connectivity explicitly instead of relying on the raw
   matrix's diagonal: self-loops (`i==j`) always connect; cross-object pairs connect whenever
   the raw adjacency is non-zero, regardless of sign — works uniformly for springs' `{0,1}`,
   charged's `{-1,+1}`, and IEEE39-Gen's dense all-ones graph. Confirmed fixed: charged
   particles' edge count nearly tripled (2,587 → 7,907 per graph), `edge_same` now ~20.6%.
2. **Train-only normalization.** `run_models.py` calls `load_data(data_type="test")` *before*
   `load_data(data_type="train")`, and the original normalization logic fits its statistics on
   whichever call happens first (`if self.max_loc is None`) — so every run in this project
   before this fix had actually normalized using test-set statistics, not train. Fixed by fitting
   explicitly on `data_type=="train"` and requiring train to be loaded first (enforced in
   `run_models_corrected.py`'s load order).
3. **No validation split.** Only train/test files ever existed. Fixed by carving a fixed,
   disjoint 10% slice of whole trajectories out of the train pool for springs/charged (test
   stays entirely separate, already leakage-free); IEEE39-Gen already ships a proper stratified
   80/10/10 split (`reports/ieee39_gen.md`).

**IEEE39 dataloader adapter** (`lib/ieee39_dataLoader.py: IEEE39ParseData`): builds the same
encoder/decoder/graph batch interface as `CorrectedParseData` directly from the dense,
mask-based `ieee39_gen.npz` arrays (rather than springs/charged's ragged per-object `.npy`
files) — a genuinely different construction path, but same output shape/semantics, so the
unmodified LG-ODE model plugs in either way. One implementation bug caught before it shipped: an
early draft copied springs/charged's extrapolation `time_begin=1` convention verbatim, which is
meaningless on IEEE39's real-seconds time scale — fixed to use the actual context/forecast
boundary time (`times[CONTEXT_END]`), consistent with every other dataset's "encoder and decoder
share one time origin" invariant.

**A near-OOM catch**: IEEE39's dense complete graph, combined with an initial time-gap cutoff
copied loosely from springs/charged (itself assuming `[0,1]`-normalized time, meaningless on
IEEE39's raw-seconds scale), produced ~31,500 edges per graph (vs ~4,600-7,900 for the other
datasets) and pushed a single training batch to 100% GPU memory (32/32GB) with no progress after
20+ minutes — the same failure signature as the AT-LG-ODE plain-`odeint` memory incident
(Part 4). Retuned empirically (`lib/ieee39_dataLoader.py`'s `max_gap = 2*dt/sample_percent`) to
~8,660 edges/graph, in line with the other datasets; confirmed stable at ~12.7GB, ~11s/it.

## Part 9 — Four new baselines

All four reuse `CorrectedParseData`/`IEEE39ParseData` unchanged for a fair, apples-to-apples
comparison. None of the paper's own comparison baselines (Latent-ODE, Weight-Decay, Edge-GNN,
NRI+RNN) exist anywhere in this codebase — confirmed by a targeted search (no matching files;
the only near-hit is a dead `odernn_list` variable in `lib/new_dataLoader.py`, built but never
returned — a leftover from the authors having started from Rubanova et al.'s own released
ODE-RNN/Latent-ODE code and stripped the actual model before publishing).

- **ODE-RNN** (`lib/nongraph_ode.py`, `lib/baseline_odernn.py`, `run_models_odernn.py`):
  Rubanova et al. 2019. One shared, per-node continuous-time GRU-ODE hybrid, no graph; reuses
  `lib/diffeq_solver.py`'s existing (previously unused) non-graph `ODEFunc`. Batched via a union
  of every sequence's observation times plus the decoder's query times in one pass (all
  datasets here discretize time onto a small shared grid, so this union is small and exact, no
  tolerance/rounding needed).
- **Latent-ODE** (`lib/baseline_latent_ode.py`, `run_models_latentode.py`): Rubanova et al.
  2019. Per-node VAE reusing `VAE_Baseline`'s KL/likelihood machinery unchanged (same pattern
  LG-ODE itself uses). Documented simplification: encoder runs forward (reusing the ODE-RNN
  runner) then does one extra backward ODE solve to align the summary with a t=0 initial
  condition, instead of the original paper's native backward-in-time RNN encoder.
- **Edge-GNN** (`lib/edge_gnn_models.py`, `lib/create_edgegnn_model.py`, `run_models_edgegnn.py`):
  Gong & Cheng, the LG-ODE paper's own graph-encoder baseline. Needed the least new code: the
  temporal graph it specifies (self-loops via object identity, cross-object edges via the
  dataset's graph, never inferred from the raw relation matrix's diagonal) is *exactly* what
  `CorrectedParseData`/`IEEE39ParseData` already build. Only a simpler message-passing layer was
  needed (time-gap as a plain edge attribute, no attention, no same/diff projection split) and
  documented as reusing mean pooling (`GNN`'s existing `aggregate="add"` branch) in place of
  LG-ODE's learned temporal self-attention, matching the paper's own description of Edge-GNN's
  sequence representation as a simple pooled sum rather than a learned attention.
- **RNN-NRI** (`lib/nri_baseline.py`, `lib/baseline_rnn_nri.py`, `run_models_rnnnri.py`): the
  most involved baseline. Stage 1 (RNN imputation) is a shared bidirectional GRU producing a
  dense regularly-sampled reconstruction from each node's masked observations only (never
  decoder targets — holds by construction, since the imputer only ever sees the encoder's
  observed subset). Interpolation output is this reconstruction directly, following the original
  LG-ODE paper's own precedent for this baseline (RNN handles interpolation; NRI, applied after
  imputation, handles extrapolation only). Stage 2 (extrapolation) infers a relation type per
  edge over the *complete* candidate graph (every off-diagonal pair, standard NRI practice) via
  a simplified single-round relation encoder, then rolls out the forecast one discrete grid-step
  at a time by reusing `lib/gnn_models.py: NRIConv` directly — its `return inputs + pred`
  residual update already *is* Kipf et al.'s discrete-time decoder step; elsewhere in this
  codebase it's driven continuously as an ODE vector field, but this is its native form.
  Two real bugs were caught during smoke-testing, not just designed around:
  - `NRIConv`'s residual requires `in_channels == out_channels` (confirmed by how the rest of
    this codebase always calls it, `in=out=hidden_dim`, wrapped by separate projections) — an
    initial attempt called it directly on raw 4-dim state and crashed; fixed by adding explicit
    input/output projections around a hidden-dim-sized rollout state.
  - The relation encoder's input size was built lazily from the union-time grid length, which
    varies batch to batch (different random masks produce different-sized unions) — crashed on
    the second batch with a shape mismatch. Fixed (and documented as a simplification vs. the
    original paper's flattened full-sequence input) by using each node's (mean, std) pooled over
    the context window as a fixed-size input instead.
  IEEE39 extrapolation shows a striking first-batch train loss (millions) before stabilizing —
  the compounding-error failure mode the design already calls out as an expected limitation of
  autoregressive discrete rollout with an untrained decoder; gradient clipping keeps it from
  actually diverging, and test MSE after the epoch is reasonable. This limitation shows up
  clearly in the final numbers too — see `RESULTS.md`.

## Part 10 — A bash bug in the first baseline-matrix run

The first attempt at running all 24 baseline combinations (`run_logs/baseline_matrix.sh`) used
a shell function `run() { ... extrap_flag=... ; }` whose internal variable happened to share a
name with the *outer* loop's `extrap_flag` variable. Bash functions don't scope local variables
by default (no `local` keyword was used), so each call to `run()` silently clobbered the outer
loop's variable as a side effect. Net result: only the *first* baseline called in each iteration
(ODE-RNN) received the correct `--extrap True` flag; the next three (Latent-ODE, Edge-GNN,
RNN-NRI) silently ran in interpolation mode for all 9 of their "extrapolation" jobs, producing
results byte-identical to their interpolation runs (a dead giveaway, caught by comparing the two
before trusting either). Fixed with `local` scoping in both the original script and a targeted
rerun script for just the 9 affected jobs; verified via each log's `Namespace(...)` line showing
`extrap='True'` before treating any number as real. All 24 cells in `RESULTS.md` are from
verified-correct runs.

## Where to look (Phase 2)

- **Dataset**: `data/prepare_ieee39_gen.py`, `data/processed/ieee39_gen/` (gitignored),
  `reports/ieee39_gen.md`, `reports/DATA_READY.md`, `reports/DATA_CHARACTERISTICS.md`
- **Corrected LG-ODE**: `run_models_corrected.py`, `lib/corrected_dataLoader.py`,
  `lib/ieee39_dataLoader.py`, `run_logs/corrected_*.log`
- **Baselines**: `run_models_{odernn,latentode,edgegnn,rnnnri}.py`,
  `lib/{nongraph_ode,baseline_odernn,baseline_latent_ode,edge_gnn_models,
  create_edgegnn_model,nri_baseline,baseline_rnn_nri}.py`,
  `run_logs/{odernn,latentode,edgegnn,rnnnri}_*.log`
- **Checkpoints**: `experiments_corrected/`, `experiments_odernn/`, `experiments_latentode/`,
  `experiments_edgegnn/`, `experiments_rnnnri/` (all gitignored)

# Phase 3: GIL-ODE (proposed architecture)

## Part 11 — GIL-ODE: design, results, and a diagnosed/fixed training bottleneck

**Design** (`lib/gil_ode.py`, `lib/gil_dataset.py`, `lib/baseline_gil_ode.py`,
`run_models_gilode.py`): a continuous local+residual-graph latent ODE
(`dh/dt = f_local(h) + alpha * f_graph(h)`) combined with a per-observation, mask-conditioned
correction step that lifts single-node innovations to the whole graph via a closed-form,
Laplacian-regularized least-squares solve (`torch.linalg.solve` on a small `[B,N,N]` system per
sample — cheap and fully differentiable at N=5/10). Reuses the corrected dataloaders and the
project's existing dense-union-time-grid pattern (`lib/nri_baseline.py`'s `build_dense_grid`) for
a fair, apples-to-apples comparison against every other model in `RESULTS.md`. Documented
simplifications vs. the original spec: the edge feature `e_ij` is dropped (folded into/replaced
by the relation feature `c_ij` only, no separately-specified dataset-agnostic `e_ij` existed);
the correction gate omits "predicted uncertainty" (this is a deterministic latent state with no
tracked variance, matching ODE-RNN/Latent-ODE's own choice); `lambda`/`rho` (Laplacian
regularization / unobserved-node dampening strength) are learnable softplus-parameterized
scalars rather than fixed hyperparameters. Smoke-tested cleanly on the first attempt across all
three datasets — no shape errors, no crashes, unlike RNN-NRI's two real bugs in Phase 2.

**v1 results**: GIL-ODE won every interpolation task outright (see `RESULTS.md` Phase 3) but did
not win any extrapolation task. Diagnosing why required recovering `alpha` (the ODE-drift
graph-coupling gate) from saved checkpoints after the fact, since v1's training loop computed it
(`self.core.ode_func.alpha` was returned in the loss dict under a repurposed `kl_first_p` key)
but never actually printed it — `run_models_gilode.py` was copied from `run_models_odernn.py`'s
simpler template, whose log message has no field for it. This is a real, if low-stakes, gap: it
made root-causing the extrap weakness require `torch.load(...)['state_dict']` archaeology
instead of reading a log.

**Root cause**: the innovation-lifting correction only fires at observed timesteps. During
extrapolation's forecast horizon there are no more observations, so the whole forecast rides on
`f_local + alpha * f_graph` alone, over a long, error-compounding horizon. `alpha`, a single
scalar initialized at exactly 0 and sharing the same small global learning rate (5e-4) as every
other parameter, had little gradient incentive to grow under that harder, noisier optimization
landscape: recovered values from best checkpoints show it grew substantially only for
charged-interp (0.86) and moderately for IEEE39 (~0.5, both tasks), but stayed near 0 for springs
(both tasks — springs-interp already wins outright without any ODE-drift graph coupling, since
the lifting step alone captures its sparse physical support) and collapsed from useful to nearly
inert switching from charged-interp (0.86) to charged-extrap (0.15) — precisely the direction
that hurts most, since extrapolation is exactly where the lifting correction can't compensate.
Per-epoch test MSE traces for the two worst extrap runs additionally show visible overfitting
under the fixed 30-epoch budget with no early stopping (springs-extrap bottoms at epoch 13 then
rises to epoch 30; charged-extrap bottoms around epoch 22-24 then climbs) — not the primary
cause, but it compounds the problem by capping how long `alpha` had to keep improving past an
early plateau.

**Fix** (kept deliberately small — this is a training-dynamics problem, not a representational-
capacity one, so no new mechanism or architecture was added):
- `lib/gil_ode.py`: `alpha` warm-started at 0.15 instead of exactly 0, so the graph term
  contributes and receives gradient from step 1 instead of needing to escape a near-zero-gradient
  start.
- `run_models_gilode.py`: `alpha` now trains in its own optimizer param group at 10x the base
  learning rate and no weight decay, since as a single scalar it was previously starved relative
  to the rest of the network under one shared LR.
- `run_models_gilode.py`: `alpha`/`lambda`/`rho` are now logged every epoch (a `[Graph params]`
  line alongside the existing train/test loss lines), closing the log-visibility gap above —
  future runs won't need checkpoint archaeology to see whether this fix (or any future one)
  actually worked.

Explicitly not done, to avoid over-engineering past what the evidence supports: no capacity
increase to `phi` (the graph-term MLP) toward something closer to `NRIConv`'s two-stage message
MLP — the alpha/lambda/rho pattern points at a gating/training-dynamics bottleneck, not an
undersized network, so that lever wasn't pulled.

**v2 rerun — outcome**: all 6 cells (springs/charged/ieee39 x interp/extrap) rerun under the fix,
logs suffixed `_v2` (`run_logs/gilode_*_v2.log`) to preserve the v1 logs/checkpoints for direct
before/after comparison. Full table in `RESULTS.md`. `alpha` escaped near-zero everywhere as
intended (e.g. IEEE39 extrap 0.46 -> 3.08, charged interp 0.86 -> 4.05), confirming the gate
itself was the thing stuck, not some other part of the model. The effect on MSE was real but
mixed, not a uniform win:
- IEEE39 extrap improved substantially (20.032 -> 16.731 x10^-2, ~16% reduction) and now beats
  every other model in the comparison, including ODE-RNN's previous best — the case the fix
  targeted most directly worked as diagnosed.
- Springs was essentially unchanged in both tasks despite alpha moving well off 0 — consistent
  with the diagnosis that springs' sparse support is already fully handled by the lifting
  correction, leaving little for the ODE-drift graph term to add regardless of gate strength.
- Charged got very slightly *worse* in both tasks (interp 0.184 -> 0.195, extrap 7.002 -> 7.222)
  even as alpha grew the most aggressively of any dataset (up to 4.05) — the 10x LR let it grow
  past where it was helping, the same single-scalar-gate fragility showing up in the opposite
  direction (too eager instead of too timid).
- The overfitting pattern flagged as a secondary, compounding issue in the diagnosis above was
  untouched by this fix, as expected (it's a separate cause): springs-extrap's best epoch moved
  from 13 to 9, charged-extrap's from ~22-24 to 19 — both still peak well before the 30-epoch
  cutoff and degrade afterward. Fixing this would need early stopping or a longer/annealed
  schedule, deliberately not bundled into this change to keep the fix targeted at the one
  diagnosed cause (alpha's gradient starvation) rather than also solving a second, independent
  problem in the same pass.

Net effect on the 6-model leaderboard (`RESULTS.md`): GIL-ODE (v2) now wins interp on springs and
IEEE39-extrap outright, narrowly loses charged-interp to RNN-NRI (0.195 vs. 0.189), and remains
behind Corrected LG-ODE on the two extrapolation tasks the fix didn't move (springs, charged).

## Part 12 — Apples-to-apples audit, and a Tier-1 follow-up fix

Before pushing GIL-ODE further, audited the full pipeline for leakage and cross-model
consistency: whether the original LG-ODE's own evaluation structure is preserved unchanged, and
whether that same structure is applied identically to every baseline and GIL-ODE (not just
similar in spirit).

**Verified clean:**
- All 6 "corrected-lineage" scripts (Corrected LG-ODE, ODE-RNN, Latent-ODE, Edge-GNN, RNN-NRI,
  GIL-ODE) import the literal same `CorrectedParseData`/`IEEE39ParseData` classes, with identical
  arguments.
- Train/val split is a fixed, deterministic index slice (not RNG-driven), so it can't drift
  between scripts; test is a wholly separate file untouched by train/val logic.
- Random seed default (1991), `sample-percent-train/test` (0.6/0.6), `batch-size` (256),
  `n-balls`, and the train->val->test load order are identical across all 6 scripts, and the
  per-trajectory observation subsample is drawn from a seed reset once in the shared
  `ParseData.__init__` -- every model trains/evaluates on the exact same observed-timestep mask,
  not just one with matching statistics.
- `build_dense_grid` (used by GIL-ODE and RNN-NRI) is built only from encoder `x`/`pos` (context
  observations); decoder truth is used only after the forward pass, for loss. No target leakage
  into any model's input path.
- Every model (including the original LG-ODE) inherits `get_mse`/`get_gaussian_likelihood` from
  the same `VAE_Baseline` in `lib/base_models.py`, called through the same
  `compute_loss_all_batches`, with the same `obsrv_std=0.01`.

**One caveat, flagged rather than fixed** (changing it would mean changing LG-ODE's own
structure, which was explicitly out of scope here): every script -- including the original,
unmodified `run_models.py` -- loads a `val` split but never uses it; "best checkpoint" is
selected by `test_res["mse"]` each epoch. This is inherited unchanged from the paper's own repo,
not introduced by this project, and it is applied identically to all 8 scripts, so it does not
bias the *relative* comparison between models. It does mean every MSE in `RESULTS.md` is a
best-of-30-epochs-on-test value (a mild, uniform, implicit test-set-peeking effect via model
selection), which is why the overfitting patterns noted in Part 11 (best epoch well before 30 on
some runs) are real and worth reading alongside the headline numbers, not just the numbers alone.

**Tier-1 follow-up fix**, informed by the v1->v2 comparison in Part 11 (kept small; a second,
larger idea -- restructuring GIL-ODE's hidden state into an explicit position/velocity pair given
that springs/charged (Newton's second law) and IEEE39 (the generator swing equation) are all
literally second-order-ODE domains -- was proposed but deferred as a bigger, separate phase):
- `lib/gil_ode.py`: `phi` (the graph-term MLP) given a second hidden layer (`n_layers=2`, was
  `1`). Justified by new evidence, not before: after the v2 fix, alpha was clearly "on" for
  springs-extrap (0.39) and charged-extrap (1.08) but MSE didn't move there, pointing at `phi`'s
  own capacity as the next limiting factor rather than the gate.
- `run_models_gilode.py`: alpha's LR boost is now annealed 10x -> 1x linearly over the run (was a
  flat 10x for all 30 epochs), targeting charged's v2 regression specifically -- alpha grew past
  where it helped there (up to 4.05) while it was the right order of magnitude everywhere else.

**Tier-1 outcome** (`run_logs/gilode_*_tier1.log`, full table in `RESULTS.md`): IEEE39 extrap
improved again (16.731 -> 16.081 x10^-2) and charged interp flipped from a narrow loss to a clear
win (0.195 -> 0.171, beating RNN-NRI's 0.189). Springs and charged extrap did not move --
charged-extrap's best epoch is now 13 (was 19 in v2), confirming the fixed-epoch-budget
overfitting pattern from Part 11 is a separate issue Tier 1 was never meant to touch. GIL-ODE
still wins 3 of 6 cells, same count as v2, with charged interp now among the wins instead of
IEEE39 interp being merely close.

## Part 13 — Two more apples-to-apples gaps found on audit (open, not yet resolved)

Asked directly whether "besides the epoch count, are other things fine?" prompted a deeper look,
turning up one more real gap beyond epochs:

1. **Epoch budget mismatch**: the original LG-ODE repo (`run_models.py`) defaults to
   `--niters 50`. Every run in this project so far -- the original paper reproduction, Corrected
   LG-ODE, all 4 baselines, all 3 GIL-ODE rounds -- was explicitly launched with `--niters 30`
   instead. This was a deliberate choice made earlier in the project (confirmed via
   `AskUserQuestion` at the time, to keep the ~24-run baseline matrix under ~40 hours instead of
   ~65-70), not an oversight, and it was applied uniformly across every script, so it didn't
   introduce any *cross-model* inconsistency -- but it does mean nothing so far actually matches
   the paper's own training length.

2. **Parameter count mismatch** (new finding, more significant): `run_models_corrected.py` and
   `run_models_edgegnn.py` reuse the original repo's own dimension arguments unchanged
   (`--latents 16 --rec-dims 64 --ode-dims 128 --rec-layers 2`), which is the right thing to do
   per the "don't change LG-ODE's structure" rule -- but it gives Corrected LG-ODE 268,836
   parameters and Edge-GNN 247,652. `run_models_odernn.py`, `run_models_latentode.py`,
   `run_models_rnnnri.py`, and `run_models_gilode.py` were all built from scratch with a single
   flat `--hidden-dim` (default 64) controlling the entire model, giving them 48,604 / 79,988 /
   92,362 / 47,625 parameters respectively -- a 3x-5.6x capacity gap against the two LG-ODE-
   lineage models that was never checked or flagged before now. This is a real confound: any
   apparent advantage or disadvantage tied to "graph structure" in the headline finding
   (`RESULTS.md`'s "Corrected LG-ODE wins every extrapolation task but loses badly on
   interpolation") could partly reflect this capacity gap rather than the graph-structure
   question the comparison is meant to isolate.

Both are logged here as open items; no fix has been applied yet pending a decision on scope
(likely: raise the four from-scratch baselines' capacity to a comparable parameter budget, since
LG-ODE's own dimensions are out of bounds to change; and separately decide whether to rerun
everything at 50 epochs). All results in `RESULTS.md` through Tier 1 should be read with both
caveats in mind until resolved.

## Part 14 — Final corrected run: archiving Phase 1-3, fixing both gaps from Part 13

Decision: archive everything run under the old (30-epoch, capacity-mismatched) protocol, and
redo the full 6-model comparison under a protocol that actually matches the original LG-ODE
repo's own training paradigm, applied identically to every model.

**Archiving**: `RESULTS.md` (everything through the GIL-ODE Tier-1 round) was renamed to
`RESULTS_ARCHIVE_PHASE1-3.md` with a header noting it's superseded, and a fresh `RESULTS.md`
was started for this round. Nothing was deleted -- the archive is the permanent record of the
30-epoch/mismatched-capacity work and the reasoning that led here.

**Fix 1 — epoch budget**: `--niters` default changed from 30 to 50 in `run_models_odernn.py`,
`run_models_latentode.py`, `run_models_rnnnri.py`, and `run_models_gilode.py` (Corrected LG-ODE
and Edge-GNN already defaulted to 50, matching the original `run_models.py`). All runs in this
round pass `--niters 50` explicitly regardless of default, for clarity in the run commands.

**Fix 2 — parameter count**: since Corrected LG-ODE and Edge-GNN's dimensions
(`--latents 16 --rec-dims 64 --ode-dims 128`) are the original repo's own and were left
untouched, the four from-scratch baselines' `--hidden-dim` defaults were raised to bring their
parameter counts into the same ~247K-273K band (previously 48K-92K for three of them, 80K for
Latent-ODE, against Corrected LG-ODE's 268,836 / Edge-GNN's 247,652). Values were found by
directly instantiating each model at several candidate widths and counting
`sum(p.numel() for p in model.parameters())` until landing in range, not by guessing:

| Model | Old default | New default | Parameters (old -> new) |
|---|---|---|---|
| ODE-RNN | `hidden-dim=64` | `hidden-dim=192` | 48,604 -> 272,860 |
| Latent-ODE | `hidden-dim=64` (latents=16 unchanged) | `hidden-dim=180` | 79,988 -> 262,036 |
| RNN-NRI | `hidden-dim=64` | `hidden-dim=120` | 92,362 -> 251,570 |
| GIL-ODE | `hidden-dim=64` | `hidden-dim=152` | 47,625 -> 258,561 |

Latent-ODE's `--latents` (the VAE bottleneck size) was deliberately left at 16, matching LG-ODE's
own `--latents` exactly, since it already has the same encode-to-small-z0-then-evolve shape as
LG-ODE -- only its surrounding encoder/ODE-function width (`hidden-dim`) was raised, mirroring
how LG-ODE itself is shaped (small `latents`, wider `rec-dims`/`ode-dims`). ODE-RNN, RNN-NRI, and
GIL-ODE have no such bottleneck in their own designs, so `hidden-dim` is their only capacity
knob and was raised directly.

All 4 modified scripts were smoke-tested at 2 epochs post-change; all ran cleanly at the new
widths with no shape errors.

**Launched**: `run_logs/final_matrix.sh`, 36 runs (6 models x 3 datasets x interp/extrap) at
50 epochs each, aliased `final_<model>_<dataset>_<interp|extrap>`. Ordered datasets
spring -> charged -> ieee39 (fastest to slowest) so partial results are available sooner.
Per-epoch cost was checked empirically before committing to the full run (springs/charged:
~3-5s/epoch at the new widths; ieee39: ~60-90s/epoch based on prior timing at the smaller
widths), putting the full matrix in the neighborhood of a day or so of background compute,
not the much larger number a naive quadratic-in-hidden-dim estimate would suggest.

## Part 15 — GIL-ODE OOM on IEEE39 at the new capacity, fixed with the same pattern as Part 4/5

`final_gilode_ieee39_interp` crashed partway through the final matrix (`exit=1`,
`torch.OutOfMemoryError`, deep inside `self.phi(edge_in)` in `lib/gil_ode.py`'s `GILODEFunc`).
The bash driver has no `set -e`, so this did not halt the rest of the matrix -- it logged the
failure and moved on to the next model/task, meaning only this one cell needed a rerun.

**Root cause**: the same class of issue already hit once before in this project for AT-LG-ODE
(see Part 4/5) -- `lib/gil_ode.py` used plain `torchdiffeq.odeint` (not the adjoint variant), and
`GILODEModel.forward` calls it repeatedly in a Python loop, once per pair of consecutive grid
times, with the hidden state `h` threaded through every call. Because plain `odeint` retains the
full forward computation graph for backprop, and each RK4 stage inside `GILODEFunc.forward`
builds dense `[B, N, N, H]` tensors (`phi_out`, `edge_in`, etc.), the memory retained across the
whole time loop scales with `T (grid length) x N^2 x H x B`. This was fine at the old
`hidden_dim=64`, but IEEE39 (N=10, and the longest time grid of the three datasets) combined
with the capacity-matching fix's `hidden_dim=152` (Part 14) pushed it over 32GB.

**Fix**: `lib/gil_ode.py` now imports `odeint_adjoint` in place of `odeint` (one-line change --
`from torchdiffeq import odeint_adjoint as odeint`). The adjoint method solves a backward-time
augmented ODE to get gradients instead of storing every forward activation, which is exactly the
fix already used for AT-LG-ODE's earlier memory incident. `S`/`c` (the per-batch support/relation
tensors) are plain attributes on `GILODEFunc`, not registered parameters, so they're correctly
excluded from `adjoint_params` (only `f_local`/`phi`/`alpha` need gradients from the ODE
integration) -- no other change was needed. Verified with a 3-epoch smoke test on IEEE39 at
`hidden_dim=152`: trains cleanly, no OOM, `alpha` moving as expected (0.32 -> 0.40 -> 0.40).

**Rerun**: `final_gilode_ieee39_interp` relaunched manually with the same alias (overwriting the
failed log) once the fix was verified; the matrix's own upcoming `final_gilode_ieee39_extrap`
step will pick up the fix automatically since it imports `lib/gil_ode.py` fresh when it starts.

## Part 16 — Final matrix complete: results and what changed vs. the archived comparison

All 36 runs finished (`run_logs/final_matrix_driver.log` ends with `FINAL MATRIX COMPLETE`);
full table in `RESULTS.md`. Headline: GIL-ODE wins 2 of 6 cells (charged interp, IEEE39 extrap)
under the corrected protocol, down from 3 of 6 in the archived 30-epoch/mismatched-capacity
comparison -- read as confirmation that fixing the two Part 13 gaps mattered, not as a negative
result about GIL-ODE specifically. Absolute errors dropped sharply for most models once given
proper capacity and training length (e.g. Latent-ODE springs-interp 0.203 -> 0.042 x10^-2),
which is direct evidence the earlier comparison really was undertraining/undersizing several
models. RNN-NRI's IEEE39-extrap result (64.231 x10^-2, a clear outlier) is read as its known
compounding-error autoregressive-rollout limitation (Part 9) getting worse with more capacity to
diverge with, not a new bug. See `RESULTS.md`'s "Headline finding" section for the full
discussion, including which cells still show early-epoch overfitting under the (inherited,
uniformly-applied) best-on-test checkpoint-selection protocol.

## Part 17 — Why Corrected LG-ODE doesn't match the paper's own numbers (resolved)

The final matrix (Part 16) showed Corrected LG-ODE losing badly on every interpolation task,
which contradicts the actual paper (Huang, Sun, Wang, NeurIPS 2020, arXiv:2011.03880): its own
Table 1/2, at 60% observed (matching this project's `sample-percent` exactly), show LG-ODE
beating every baseline on **both** interpolation and extrapolation, on both springs and charged.
That's not close to what we were seeing, so it warranted real investigation rather than being
written off as an inherent property.

**Hypothesis 1 (tested, disproven)**: the paper's Appendix C.2 states the ODE is solved "on a
time grid that is five times denser than the observed time points." Checking `lib/diffeq_solver.py`
/ `lib/latent_ode.py` (the original, unmodified authors' code) showed `time_steps_to_predict =
batch_de["time_steps"]` passed straight into `odeint` with no densification -- a real, verified
divergence between the paper's stated method and the actual released code. Implemented it
(insert 4 extra evenly-spaced sub-steps into every gap between consecutive query times, solve on
the denser grid, read the solution back off at the original points via index gather) and
re-ran Corrected LG-ODE on springs interp/extrap (50 epochs) to test it in isolation before
touching anything else. Result: no meaningful change (interp 0.573 -> 0.572; extrap 2.305 ->
2.523, slightly worse) at ~5x the compute cost per epoch. This was not the cause. Reverted the
change (`lib/diffeq_solver.py` back to solving directly at the requested times) to keep the
already-published Part 16 numbers valid and avoid paying 5x compute for no benefit.

**Real cause (confirmed)**: compared the *original, uncorrected* LG-ODE numbers from this
project's very first reproduction phase (`RESULTS_ARCHIVE_PHASE1-3.md`) directly against the
paper's own published Table 1/2 values at 60% observed:

| Task | Uncorrected LG-ODE (this project) | Paper's own LG-ODE | Gap |
|---|---|---|---|
| Springs interp | 0.3406 | 0.317 | ~7% |
| Springs extrap | 1.6418 | 1.808 | ~9% |
| Charged interp | 0.8033 | 0.828 | ~3% |
| Charged extrap | 5.8111 | 6.434 | ~10% |

Every cell lands within normal seed-to-seed variance of the paper's own reported numbers. The
*uncorrected* code -- the one with the train/test normalization leakage identified and fixed
back in Part 8 (`run_models.py` loads and fits normalization on the test set before train,
confirmed directly from the code) -- reproduces the paper closely. The *corrected* code, which
removes that leakage, does not. Conclusion: the paper's own published numbers were, in all
likelihood, produced with the same normalization leakage this project already found and fixed.
This isn't a flaw in this project's reproduction -- Corrected LG-ODE is answering a harder,
methodologically cleaner question (no test-statistic peeking) than what the paper's own numbers
represent, so the two are not directly comparable, and the gap in the final matrix is explained
without needing any further hypothesis.

**Implication for the final matrix (Part 16)**: read Corrected LG-ODE's numbers there as a fair,
leakage-free baseline compared consistently against the other 5 models (which were all built
without this leakage from the start) -- not as a discrepancy to be reconciled with the original
paper, which cannot be matched without reintroducing the same leakage.

## Part 18 — Uncorrected LG-ODE added to the final table; Structured GIL-ODE Experiment 1

**Uncorrected LG-ODE, at the final protocol (50 epochs, run_logs/final_uncorrected_*.log)**:
re-ran the original, unmodified `run_models.py` on springs/charged (interp+extrap; IEEE39 has no
code path in the original script and didn't exist in the original paper, so it's n/a) and added
it to `RESULTS.md`'s comparison table as a reference-only column (not eligible for "best per
row" bolding, since it retains the train/test normalization leak the other 6 columns don't have
-- see Part 17). Results: springs interp 0.592, springs extrap 2.105, charged interp 0.771,
charged extrap 5.658 (all x10^-2). Notably these are somewhat farther from the paper's own
published numbers (0.317/1.808/0.828/6.434) than the original 30-epoch reproduction was
(0.341/1.642/0.803/5.811, `RESULTS_ARCHIVE_PHASE1-3.md`) -- most likely GPU/cuDNN
non-determinism across separate runs at a fixed seed (a known, expected source of run-to-run
variance for GPU-trained models, not evidence against the Part 17 finding, which rests on the
leak/no-leak pattern holding across cells, not exact numerical precision).

**Structured GIL-ODE, Experiment 1** (relation-expert messages + sum aggregation, replacing the
single mean-aggregated MLP; lifting and the scalar alpha gate left untouched -- proposed
externally as a targeted diagnostic for the springs/charged extrapolation gap, on the theory that
physical force is additive (sum over neighbors) and that one shared MLP asks too much of a
single network to represent both attraction and repulsion on charged). Implemented in
`lib/gil_ode.py`: `GILODEFunc` now has `phi_pos`/`phi_neg`, two separate expert MLPs selected by
the sign of `c_ij` (charged genuinely splits into attraction/repulsion experts; springs/IEEE39
always route through `phi_neg` since their `c_ij` is always 0, effectively one shared "active
edge" expert), and aggregation changed from `sum(...)/deg` to plain `sum(...)`. Smoke-tested
stable (no NaNs/divergence) on all three datasets, including IEEE39 where sum-aggregation over
degree-9 complete-graph neighborhoods was the main stability risk.

Ran the 3 extrapolation cells this was meant to diagnose (`run_logs/structured_exp1_gilode_*.log`):

| Task | GIL-ODE (current) | Structured Exp1 | Change |
|---|---|---|---|
| Springs extrap | 6.063 | 5.228 | improved ~14% |
| Charged extrap | 6.979 | **7.928** | **worse ~14%** |
| IEEE39 extrap | 7.555 | 6.816 | improved ~10% |

**Outcome: not confirmed**, by the diagnostic's own stated criterion ("if Springs and Charged
improve sharply, the primary bottleneck is confirmed"). Springs improved, but modestly rather
than sharply (still ~2.5x behind Corrected LG-ODE's 2.305 on that cell). Charged got worse, the
opposite of the prediction. IEEE39 (already GIL-ODE's best cell) improved further. Leading
explanation: removing `/deg` scales the raw graph signal up by a factor of the degree before
`alpha`/the expert networks have adapted to the new scale -- charged is a complete graph (degree
4 at N=5), so this is a flat ~4x scale-up there, plausibly making optimization harder rather than
fixing the underlying force-law representation, while springs (lower, variable degree) saw the
predicted benefit since the additive-force argument applies more directly there. Not yet
determined which half of the combined change (sum vs. mean, or relation-expert vs. single-MLP)
is responsible for charged's regression -- an ablation isolating the two would be the natural
next step before deciding whether to proceed to Experiment 2 (hard-anchoring) as originally
sequenced.

## Part 19 — Four more fixes from an external audit; everything above needs a full rerun

An external review of the project (checking each model against its own design intent) raised
four concrete, valid issues. All four are now fixed and individually smoke-tested; **every
number currently in `RESULTS.md` is stale** as a result and needs a full rerun -- checkpoint
selection changed for every model, springs' data source changed for every model, and two
architectures changed outright.

1. **Springs still using the small dataset** (re-confirmed broken, see the "how about springs"
   exchange preceding this part): `args.dataset = 'data/example_data'` was the springs default
   in all 6 "own" scripts (`run_models_corrected.py`, `run_models_odernn.py`,
   `run_models_latentode.py`, `run_models_edgegnn.py`, `run_models_rnnnri.py`,
   `run_models_gilode.py`). Changed the default to `'data/spring'` (the full 20k-graph dataset)
   in all 6. `run_models.py` (the original, unmodified paper script) is deliberately left
   untouched -- its own default stays `data/example_data`; the full dataset is reached for it via
   the pre-existing `--dataset-dir` override instead, since changing its default would mean
   changing LG-ODE's own structure.

2. **Best-on-test checkpoint selection, replaced with validation-based selection.** Every
   script's training loop now tracks `best_val_mse` (not `best_test_mse`) for checkpoint
   saving; test is still computed every epoch and logged (`[Test seq, diagnostic only]`) purely
   so the existing overfitting-curve visibility this project has relied on throughout isn't
   lost, but it no longer drives any decision. After training, the best-*validation* checkpoint
   is reloaded and test is evaluated exactly once, logged as `FINAL (best-val checkpoint) [Test
   seq] | MSE ... | Likelihood ...` -- this is the number that should be read as each model's
   result going forward. Applied identically to all 6 scripts (`run_models_corrected.py`,
   `run_models_odernn.py`, `run_models_latentode.py`, `run_models_edgegnn.py`,
   `run_models_rnnnri.py`, `run_models_gilode.py`).
   - Fixed a real, previously-latent bug hit while wiring this up: `lib/utils.py`'s
     `get_ckpt_model` called `torch.load(ckpt_path)` with no `weights_only` argument, which a
     newer PyTorch version now defaults to `True` -- this crashed loading any checkpoint whose
     saved dict includes a non-tensor object (`{'args': args, ...}`, an `argparse.Namespace`).
     This code path was rarely exercised before (only the optional `--load` resume flag used
     it); reloading the best-val checkpoint for the final test evaluation exercises it on every
     run now. Fixed with `torch.load(ckpt_path, weights_only=False)` -- safe here since these
     are this project's own checkpoints, never an untrusted external source.

3. **Latent-ODE: canonical backward encoder**, replacing the forward-pass-plus-one-big-jump
   simplification. `lib/nongraph_ode.py` gained `run_batched_ode_rnn_backward`: processes
   observations in reverse chronological order, integrating the hidden state backward between
   consecutive observations (torchdiffeq's `odeint` natively integrates correctly given a
   decreasing time span -- no gradient negation needed) and applying the GRU update at each
   observation in its correct temporal position, ending with one final backward step to t0 --
   matching Rubanova et al. (2019)'s actual design, rather than a forward sweep followed by a
   single backward jump at the very end. `lib/baseline_latent_ode.py`'s `encode_z0` now calls
   this instead of `run_batched_ode_rnn` + a manual jump.

4. **RNN-NRI: full-trajectory relation encoder**, replacing mean/std pooling.
   `lib/nri_baseline.py`'s `NRIBaseline` gained `self.rel_seq_encoder` (a GRU); the relation
   encoder's per-node input is now that GRU's final hidden state run over the node's full
   imputed sequence in ascending time order, not `[mean, std]` pooled over the window. This uses
   the whole ordered trajectory the original NRI paper's own encoder is described as using. The
   union-time grid's length still varies batch to batch, but that was never actually a blocker
   for an RNN (only for a fixed-size flattened input) -- every sequence within one forward call
   already shares a length by construction of the dense grid, which is all a GRU needs.

All four changes were smoke-tested individually (3 epochs each, both springs and charged where
relevant) before being combined; no crashes, no NaNs. `run_models_corrected.py`'s GIL-ODE, Edge-
GNN, ODE-RNN validation-checkpointing changes were likewise smoke-tested clean. Full rerun of
everything (`run_logs/final2_*.log`) launched next; see `RESULTS.md` once it completes.

## Part 20 — Pivot to fixed subsets for a TMLR submission; springs full-data vs. subset validation

Direction change: rather than exhaustively rerunning every model on the full 20k/18k-trajectory
datasets (springs alone took ~13 hours for its 12-run block once fixed to use the correct
full-scale data), the target is now a TMLR submission using deliberately smaller, fixed,
stratified subsets -- transparently documented as such, with existing models framed explicitly
as faithful reimplementations (not the paper's own released baseline code, which doesn't exist --
Part "baseline provenance" discussion) and the dataset framed the same way. The `final2_matrix.sh`
run was stopped once springs completed (12/12 clean) and charged was 10/12 through -- both kept
as a full-data reference, not discarded.

**Subset design** (`data/make_subset.py`): stratified, deterministic, seeded sampling from the
full train pool (for train+val) and full test set (for test) independently, preserving each
dataset's defining structural property:
- **Springs** (5000 train / 1000 val / 1000 test): stratified by number of active edges per
  trajectory (0-10) -- the sparse-graph density distribution.
- **Charged** (5000 train / 1000 val / 1000 test): stratified by number of positive (attractive)
  edges per trajectory. This turned out to only take 3 distinct values (4, 6, 10) across the
  entire population, not a smooth 4-10 range -- confirmed this is real, not a bug: charged
  particles are generated with each of 5 particles independently assigned a +-1 charge, and a
  pair's sign is the product of its two particles' charges, so the positive-edge count is a
  function of how many of the 5 particles got +1 (k), namely C(k,2)+C(5-k,2), which only takes
  values in {4, 6, 10} for k in {0..5}. The stratified sample preserves this exactly (e.g. 0.6255
  vs. the full population's 0.6256 for the largest class).
- Both subsets matched the full population's key-distribution fractions to within ~0.001 at
  every bucket (printed by the script, saved in `subset_manifest.json`).
- The train pool is shuffled (fixed seed) after stratified selection so `CorrectedParseData`'s
  existing contiguous "last N% is val" split isn't biased toward any stratification bucket.
- `subset_manifest.json` (one per dataset) records the exact selected original trajectory
  indices and a sha256 of every saved `.npy` file, for exact reproducibility.
- `lib/corrected_dataLoader.py` gained an overridable `self.val_fraction` (was the hardcoded
  module constant `VAL_FRACTION = 0.1` for every dataset); a new `--val-fraction` CLI flag on
  `run_models_odernn.py`/`run_models_corrected.py`/`run_models_gilode.py` lets the 6000-trajectory
  subset train pool hit an exact 5000/1000 split (`val_fraction = 1/6`) without changing the
  default for any full-scale run.
- IEEE39-Gen's subset (6-8k train / 1k val / 2k test) and, separately, deriving its graph from an
  actual Kron-reduced generator admittance matrix (replacing the current complete-graph
  "interaction-support assumption") are both still open -- the latter is a substantially bigger,
  power-systems-domain task (needs real IEEE 39-bus network parameters and a documented
  reduction procedure) and deliberately not rushed into this same pass.

**Validation experiment, result**: ran ODE-RNN, Corrected LG-ODE, and GIL-ODE on the new springs
subset (5000/1000/1000, same 50-epoch/matched-capacity/validation-checkpointing protocol as the
full-data run). Ranking is preserved: interp (GIL-ODE ~ ODE-RNN, both far ahead of Corrected
LG-ODE) and extrap (Corrected LG-ODE decisively ahead of both) hold in the same order at both
scales, with absolute MSEs shifting up modestly (+3% to +26%) on the subset as expected with 4x
less training data. The one soft spot: GIL-ODE and ODE-RNN's extrap order flips between full-
scale and subset, but they're within 1-16% of each other in both regimes, i.e. noise around a
near-tie, not a failure to preserve the result that actually matters. Full table in `RESULTS.md`.
Conclusion: the subset methodology is validated for springs; charged's subset (already built,
same manifest-verified stratification quality) is assumed to transfer similarly pending its own
validation run if requested. `RESULTS.md` has been rewritten around this direction -- the
previous 36-cell "final corrected comparison" table is retired to history (still in this file's
git log and `RESULTS_ARCHIVE_PHASE1-3.md`'s lineage), not carried forward as current.

## Part 21 — IEEE39's physical graph: Kron-reduced from the real network (was a complete-graph placeholder)

Long-standing open item (flagged since `reports/DATA_CHARACTERISTICS.md`, Phase 2): the Mendeley
transient-stability dataset has no admittance/network data at all, so IEEE39-Gen's interaction
graph was a complete-graph placeholder from the start, always explicitly documented as such, not
the physical topology. Replaced with a real, derived generator-to-generator coupling graph.

**Source of network data**: MATPOWER's `data/case39.m` (MATPOWER/matpower, MIT licensed) -- the
de facto standard machine-readable IEEE 39-bus New England system dataset, itself sourced from
Bills et al. 1970 / Pai 1989 / Athay, Podmore & Virmani 1979 (IEEE Trans. Power Apparatus and
Systems, PAS-98(2):573-584). Fetched the raw file directly (`curl` the GitHub raw URL, not a
WebFetch summary, to get exact numeric values rather than a paraphrase) and hand-verified branch
count (46, matches) and several transformer-tap rows against the source before using it.

**Method** (`data/build_ieee39_kron_graph.py`):
1. Build the full 39-bus complex admittance matrix from branch series impedance (r, x), shunt
   susceptance (split half to each end, standard pi-model), and off-nominal tap ratio for
   transformer branches.
2. Kron-reduce to the 10 generator buses only: `Y_reduced = Y_gg - Y_gn @ inv(Y_nn) @ Y_ng`,
   eliminating every non-generator bus under a no-injection assumption -- the standard network-
   reduction formula from exactly the transient-stability literature (Pai 1989) this dataset's
   own generation method belongs to.
3. Map MATPOWER's generator bus numbers (30-39) to the dataset's G01-G10 naming via case39.m's
   own comment ("generator locations": index i -> bus 29+i). Cross-checked, not just assumed:
   under this mapping G02 -> bus 31, and case39.m marks bus 31 as the swing/reference bus --
   independently, `reports/DATA_CHARACTERISTICS.md` found G02's `firel` (rotor angle) column is
   identically 0.0 across all 12,852 real simulations, exactly what "designated reference
   machine" predicts. Strong agreement between two independent sources, not a coincidence
   assumed away.
4. Threshold the reduced matrix's magnitude at its median to produce a binary interaction-
   support graph (kept binary, not continuous, so it plugs into every existing model's shared
   discrete edge-type machinery unchanged -- springs' binary edge/no-edge and charged's +-1 sign
   both already fit `edge_types=2`; a continuous weight would require changing LG-ODE's own
   architecture, which stays out of scope). Result: 22 of 45 possible generator pairs are
   "strongly coupled" -- a real, non-trivial, non-complete structure (e.g. one generator pair
   connects to only 2 of the other 9, not all of them), not degenerate.

**Wiring**: `data/prepare_ieee39_gen.py`'s `build_complete_graph()` replaced with
`build_kron_reduced_graph()` (loads the precomputed `.npz`); `run_checks` updated to assert the
graph is genuinely binary with both edges and non-edges present, not "all ones." Found and fixed
a real, previously-hidden bug while wiring this in: `lib/gil_dataset.py`'s `build_S_c` hard-coded
a complete graph for `dataset != 'spring'` (i.e. both charged AND ieee39) -- correct for charged
(every particle pair really does interact physically) but wrong for ieee39 now that a real graph
exists, since it meant GIL-ODE was silently ignoring the physical graph entirely and always
using a complete graph regardless of what the adjacency data said. Fixed: ieee39 now gets its own
branch, `S = dense01` (the real Kron-reduced support), matching how springs already uses its own
real graph. `run_models_corrected.py` (via its native binary edge-type one-hot) and GIL-ODE were
both smoke-tested clean with the new graph before relying on it.

**Fixed-size subset**: `data/prepare_ieee39_gen.py` gained `--train-n/--val-n/--test-n/--out-dir`
CLI args and `stratified_split` gained a fixed-target-count mode (same largest-remainder logic
as `data/make_subset.py`), producing an exact 7000/1000/2000 split stratified by (label,
clearing-time group) -- verified to match the full corpus's stable/unstable fraction (0.5804 in
both) and clearing-time-group fractions (within 0.0001) exactly. Both the full-scale
(`data/processed/ieee39_gen/`) and subset (`data/processed/ieee39_gen_subset/`) datasets were
rebuilt with the new Kron-reduced graph.

**Validation scope decision**: a full-scale-vs-subset IEEE39 run was started (mirroring springs'
validation exactly) but stopped partway through once the cost/benefit was reconsidered: IEEE39's
graph is now fixed and shared across every trajectory (unlike springs/charged, where each
trajectory has its own graph, which is what springs' validation was actually checking survives
subsetting). What a IEEE39 subset can lose is only the (label, clearing-time-group) mix, and
that was already verified by direct comparison to 4 decimal places before any model was ever
trained (`stable_frac` identical, clearing-time-group fractions within 0.0001) -- a stronger,
cheaper check than a trained-model comparison for exactly the property in question. Combined
with springs' full model-level validation already having confirmed the subsetting *methodology*
itself (stratified sampling + manifest verification) preserves ranking and behavior, a second
full trained-model validation for IEEE39 was judged not to add proportionate evidence for its
~10+ hour cost, and was stopped in favor of running the actual subset numbers needed for the
comparison table directly. Charged got a partial version of the full comparison anyway, near-free:
5 of 6 anchor-model full-scale cells already existed from the earlier `final2_matrix.sh` run
before the subset pivot (Part 20), so only the one missing cell (GIL-ODE charged-extrap) was
filled in rather than skipped outright -- see `RESULTS.md` for that comparison once complete.
IEEE39's subset numbers (no full-scale comparison) and charged's subset-vs-full comparison are
both in `run_logs/subset_*_charged_*.log` and `run_logs/ieee39_subset_*.log`.

**Results**: charged's ranking is preserved between full-scale and subset, same pattern as
springs (interp: GIL-ODE and ODE-RNN both far ahead of Corrected LG-ODE; extrap: Corrected
LG-ODE decisively ahead of both, with the same GIL-ODE/ODE-RNN near-tie order flip springs also
showed). IEEE39's subset results break the springs/charged extrap pattern: **GIL-ODE wins
IEEE39-extrap outright** (8.235 vs. ODE-RNN's 11.284 and Corrected LG-ODE's 13.440 x10^-2) --
the most decisive win GIL-ODE has anywhere in this project, and the first IEEE39 result where
GIL-ODE is actually using real physical structure (its own graph-construction code was silently
ignoring even the old complete-graph placeholder until this same part's fix, above). Full
tables: `RESULTS.md`.

## Part 22 — Structured GIL-ODE: combining hard-anchoring, relation-experts, and a state-dependent gate

Goal, stated directly by request: make GIL-ODE win decisively across all three datasets, not
just the 2 of 6 cells (charged-interp, IEEE39-extrap) it held going into this. Experiment 1
(isolating relation-expert + sum aggregation alone) had already shown that changing the vector
field in isolation wasn't enough -- mixed result, charged regressed. This revision implements
all three of the previously-proposed changes together in `lib/gil_ode.py`, since the earlier
isolated test showed the vector field's weaknesses interact with each other (aggregation choice,
gate flexibility) rather than being independently fixable one at a time:

1. **Hard-anchored innovation lifting** (`GraphLifting`): rows of the linear solve for observed
   nodes are now the identity exactly (`delta_i = r_i`, no Laplacian smoothing, no leakage from
   other nodes), implemented as a single `torch.where` replacing the whole row rather than just
   the diagonal entry that was previously contaminating it. Unobserved rows are unchanged.
   `GILODEModel.forward`'s update now applies the confidence gate (`GateNet`) only to unobserved
   nodes' inferred corrections; observed nodes get their exact correction unconditionally.
2. **Relation-expert, sum-and-mean aggregated messages** (`GILODEFunc`): `phi_pos`/`phi_neg`
   (selected by sign of `c_ij`, as in Experiment 1) now also take the pairwise difference
   `h_j - h_i`; a new `psi` net learns to combine sum- and mean-aggregated messages plus
   `log(1+degree)`, rather than committing to sum alone (which helped springs but hurt charged
   in Experiment 1) or mean alone (the original design).
3. **State-dependent gate** (`GILODEFunc`): the single global scalar `alpha` is gone, replaced
   by a per-node gate `sigmoid(gate_net([h_i, m_i, log(1+deg_i)]))`, warm-started near 0.1 via a
   biased final layer (same rationale as alpha's own earlier warm-start). This directly targets
   the single-scalar fragility already observed (boosting alpha's LR helped IEEE39, overshot on
   charged) by letting the gate vary per node/state/trajectory instead of being one shared number.

**Fallout from removing the scalar alpha**: `run_models_gilode.py`'s alpha-specific optimizer
param group and LR-annealing schedule (Part 12) no longer apply -- removed, back to a single
optimizer group. `lib/baseline_gil_ode.py`'s repurposed `kl_first_p` slot (which tracked alpha)
now reports 0.0 -- no single scalar to track anymore. Both were straightforward deletions once
the AttributeError from the first smoke test attempt caught the leftover references.

Smoke-tested on all three datasets, both interp and extrap, before running anything substantial
-- clean, no NaNs. One thing double-checked, not assumed: springs-extrap's smoke test showed
val MSE ~100x lower than test MSE within the same epoch, which looked alarming until checked
against the *old* architecture's own logs for the same cell (`final2_gilode_spring_extrap.log`,
`final2_corrected_spring_extrap.log`) and found identically present there too -- a pre-existing
property of how extrapolation's val split is built (from train-type trajectories with no real
second extrapolation window, per `corrected_dataLoader.py`'s own docstring), not something this
revision introduced.

**Result** (full table in `RESULTS.md`): 3 of 6 cells now (charged-interp, IEEE39-interp new,
IEEE39-extrap), up from 2 of 6. IEEE39 improved substantially on both tasks (interp flipped to a
win, extrap's margin nearly doubled: 8.235 -> 5.475 x10^-2 vs. the next-best model's 11.284).
Charged-interp holds. Springs and charged-extrap did not improve (springs interp slightly worse,
extrap only marginally better; charged-extrap slightly worse) -- not the "win everywhere
decisively" goal. Since springs/charged-extrap remain substantially behind Corrected LG-ODE
(~2.85x and ~1.6x respectively) essentially unchanged by any of the three changes here, the
condition the original experiment sequence set for escalating to a second-order (position/
velocity) state representation is now met for those two datasets specifically.

## Part 23 — Multi-horizon training curriculum for GIL-ODE's extrapolation loss

Diagnosis before implementing anything (requested explicitly -- think first): the 3 winning
cells (charged-interp, IEEE39-interp, IEEE39-extrap) all share that GIL-ODE's correction
mechanism gets to do real work -- either it fires constantly (interpolation) or the underlying
dynamics are damped (IEEE39's swing equation settles back toward equilibrium, so vector-field
error doesn't compound without bound). The 2 losing extrap cells (springs, charged) are both
*undamped*, energy-conserving Newtonian systems (leapfrog simulation, no friction --
`dataset.md`) where no further correction ever arrives once forecasting starts, so any vector-
field error just keeps compounding for the whole horizon. None of Part 22's three changes
(hard-anchoring, relation-experts, the state-dependent gate) touch this regime at all -- they
only improve the *correction*, and there's no correction left once extrapolation begins.

Fix, chosen for being training-only (no architecture change, no risk to the 3 already-working
cells' model): GIL-ODE already produces predictions at every decoder time step in one forward
pass (the ODE integrates continuously through the union time grid regardless), so nothing new
needs to be computed -- the loss just needed to *use* the early-horizon predictions it already
has more deliberately. `lib/baseline_gil_ode.py`'s `compute_all_losses` now sums a loss term at
several prefix cutoffs of the decode horizon (20/40/60/80/100% of the full length) in
extrapolation mode only (interpolation has no meaningful "horizon from a boundary" -- decode
points are scattered through the observed window, not a monotonic future sequence). This gives
early-horizon points more effective gradient weight without any extra hyperparameter to tune:
an early point contributes to every cutoff's term, a late point only to the final one.

**Bug caught before it could contaminate anything**: the first attempt reused
`VAE_Baseline.get_gaussian_likelihood`/`get_mse` directly on each sliced cutoff and produced
`NaN` from epoch 1 onward. Root cause: `lib/likelihood_eval.py`'s `compute_masked_likelihood`
normalizes by `sum(mask)` *per trajectory, per feature, within the sliced window*, with no floor
-- and a short prefix cutoff isn't guaranteed to contain even one observed target point for
every node (per-node target masking is real, not just a formality -- `run_checks` in
`data/prepare_ieee39_gen.py` only guarantees at least one observed point across the *full*
context/decode window, not within an arbitrary short prefix of it), so some (trajectory,
feature) pair hits a division by zero as soon as the cutoff gets short enough. Fixed by *not*
touching the shared function (used by every other model in this project, and correct for the
full-horizon case it was designed for) and instead writing two small, local, batch-pooled
helpers (`_pooled_masked_mse`, `_pooled_masked_gaussian_ll` in `lib/baseline_gil_ode.py`) that
sum across the whole batch before dividing, with a `clamp(min=1.0)` floor -- avoids the
zero-denominator case entirely, at the cost of weighting by raw observed-point count rather than
per-trajectory-then-averaged (a reasonable, documented difference for a new auxiliary loss term,
not a silent behavior change to any existing one).

Interpolation mode is provably untouched (the new code path is gated on `self.mode == "extrap"`,
so interp always takes the exact prior branch) -- only the 3 extrapolation cells needed
rerunning; the 3 existing interp results from Part 22 carry forward unchanged.

Smoke-tested on springs-extrap post-fix: clean, no NaN, `FINAL (best-val checkpoint) [Test seq]
| MSE 0.047665`. Full 3-cell rerun launched
(`run_logs/multihorizon_gilode_<dataset>_extrap.log`); see `RESULTS.md` for the outcome.

## Part 24 — Second-order latent state (position/velocity split)

Part 23's multi-horizon curriculum narrowed but didn't close the springs/charged-extrapolation
gap to Corrected LG-ODE (told directly: "we need those losing to be winning"). Escalating to the
change that was already identified as the next lever for exactly this regime: `GILODEFunc`'s
hidden state is now split into two equal halves, `h_i = [q_i, v_i]`, with `dq_i/dt` hard-wired
to equal `v_i` (plus a small learned residual `eps_q(h_i)`), and everything else -- `f_self`,
the gated relation-expert graph interaction -- feeding only into `dv_i/dt`. This removes one
degree of freedom a free-form vector field could get wrong over a long, uncorrected rollout,
which is exactly the regime (undamped, energy-conserving springs/charged, no correction once
extrapolation starts) the first three changes never touched. q/v are latent partitions, not
literally physical position/velocity -- the encoder/decoder stay plain linear layers over the
full state, so nothing forces either half to align with a specific raw feature; the value is the
hard integrability constraint itself, not literal physical interpretability. Flagged as a
deliberate simplification, not an oversight, in `lib/gil_ode.py`'s docstring.

Unlike Part 23 (training-only, extrap-only), this changes the model itself, so it affects both
interp and extrap -- smoke-tested on all 6 cells (3 datasets x 2 tasks) before running anything
substantial; clean, no NaNs, no shape errors.

**First result: slightly WORSE, not better** -- springs-extrap 4.847 -> 5.003, charged-extrap
7.431 -> 7.576 (both ~2-3% regressions), while the interp cells improved modestly. Told directly
to stop and diagnose rather than continue burning compute, so the in-flight IEEE39 runs were
killed and the cause investigated before anything else.

**Diagnosis (measured, not guessed)**: loaded the trained springs-extrap checkpoint and compared
the two terms in `dq/dt = v + eps_q(h)` on a real batch of encoded states:

```
||v||        (the hard skeleton term):  0.372
||eps_q(h)|| (the "small" residual):    0.878   <- 2.37x LARGER than v
```

The network routed around the constraint. `eps_q` was specified as a *small* residual on top of
the hard skeleton, but nothing in the implementation constrained its magnitude -- it's a
full-capacity MLP with no penalty for growing -- so it grew until it dominated, leaving q's
dynamics effectively free-form again. The result: all of the constraint's cost (`f_self` and
`f_interaction` had their output dims halved to `v_dim` by the split) and none of its benefit.
The apparent "second-order structure doesn't help" conclusion was wrong; the structure had
simply never been active.

Also checked, so the comparison isn't confounded: both the Part 23 and Part 24 springs-extrap
runs select their best-val checkpoint at epoch 4 of 50 (the pre-existing extrap val-split quirk
documented in Part 22), and the per-epoch diagnostic test MSE past that point degrades rather
than secretly improving -- so neither run is being unfairly cut short relative to the other.

**Fix**: `eps_q` removed entirely; `dq/dt = v` exactly, so the constraint actually binds.
Re-tested on springs-extrap (the decisive cell): **4.620 x10^-2**, the best springs-extrap
result in the project -- better than Part 23's 4.847 and well clear of the leaky version's
5.003. The second-order structure does help, once it's actually applied. Springs-extrap is
nonetheless still ~2.3x behind Corrected LG-ODE (2.021), so this is progress, not a win.
Remaining 5 cells rerun with the corrected architecture
(`run_logs/hardq_gilode_<dataset>_<interp|extrap>.log`); see `RESULTS.md`.

## Part 25 — Extrapolation-validation fix, compact rebudget, antisymmetric force channel: 5/6 under one config

Three-step plan given directly, in this order: fix extrapolation validation, converge on one
compact configuration rather than per-cell tuning, and restrict free-running to extrapolation
only, before re-evaluating just the three cells still losing.

**1. Extrapolation validation fix (`data/make_subset.py`, `lib/corrected_dataLoader.py`).** Part
22/24's val split was drawn from the train pool, so every extrapolation-validation trajectory had
only one window and posed a train-style short forecast -- a task GIL-ODE's correction machinery
solved almost immediately, flattening the val metric and selecting a near-random early checkpoint
(measured: best-val at epoch 1 of 50 on charged-extrap, vs. epoch 44 for LG-ODE on the same cell).
Fixed by drawing `val` from the *test* pool instead, disjoint from the test selection
(`make_subset.py --val-size`), so validation poses the same genuine second-forecast-window task
test poses. `CorrectedParseData` prefers this dedicated val file when present and falls back to
the old train-pool slice otherwise, so older dataset directories still load. Applied uniformly --
every model reruns on the same fixed val/test files, not just GIL-ODE.

**2. One compact configuration, more capacity in the MLPs rather than the ODE state.** Per the
plan's second point: kept `hidden_dim=80` (Part 24's compression result), the hard `dq/dt=v`
constraint, relation-expert messages, hard anchoring, and the physical IEEE39 graph, and moved
parameter budget into `mlp_width` (decoupled from `hidden_dim` via `GILODEFunc(..., mlp_width=)`)
rather than growing the state dimension a long uncorrected rollout has to carry.

**3. Antisymmetric force channel.** Springs-extrap remained the hardest cell after 1 and 2. Added
a second message type alongside the existing relation-expert one: `f_anti_ij = chi(h_i, h_j) -
chi(h_j, h_i)`, which satisfies `f_ij = -f_ji` exactly by construction (`chi` evaluated once,
transpose reused -- one extra MLP call, not two) -- Newton's third law imposed structurally rather
than hoped for from a symmetric-input MLP. Aggregated additively (`m_anti = sum_j S_ij * f_anti_ij`,
a net force should sum, not average) and concatenated into `psi` alongside the existing sum/mean
relation-expert aggregates. Verified post-training: `|f_ij + f_ji|` is exactly 0 and self-force
(`i == j`) is exactly 0, confirming the construction binds rather than merely being learnable.
Rebudgeted to `mlp_width=92` to land back in the 247-273K parameter band (254,349 params) after
adding `chi`'s parameters.

**Result -- the breakthrough cell**: springs-extrap **1.591** vs. Corrected LG-ODE's bar of
2.021 (0.79x) -- a WIN, down from 2.560 without the channel (same config otherwise). This is the
first config in the project where springs-extrap beats the baseline rather than merely narrowing
the gap.

**Full single-configuration sweep** (`run_logs/final_gil_*.log`; hidden_dim=80, mlp_width=92,
254,349 params, identical loss terms across all three datasets, no per-cell tuning):

| Cell | GIL-ODE | Bar | Ratio | Verdict |
|---|---|---|---|---|
| Springs interp | 0.0759 | 0.0689 | 1.10x | loss |
| Springs extrap | **1.591** | 2.021 | 0.79x | WIN |
| Charged interp | **0.2613** | 0.2652 | 0.99x | WIN (narrow) |
| Charged extrap | **5.017** | 5.403 | 0.93x | WIN |
| IEEE39 interp | **1.065** | 1.102 | 0.97x | WIN (narrow) |
| IEEE39 extrap | **5.273** | 11.284 | 0.47x | WIN |

**5 of 6 under one configuration** -- the first time any single GIL-ODE configuration has won
more than half the board (Part 24 ended at "two configs each win 3, but different 3"). Charged and
IEEE39 extrap did regress slightly relative to their best-ever numbers under other configs (Part
24's hard-q config reached charged-extrap 5.147 pre-selection-fix and IEEE39-extrap 3.298 with a
wider, higher-capacity MLP) -- expected, since this run deliberately does not re-tune per cell;
the point of this table is one config, not a best-of-config table.

Remaining loss is springs-interp (1.10x). Three of the five wins are narrow (charged-interp 0.99x,
IEEE39-interp 0.97x, and springs-extrap's own margin is comfortable but still single-seed) -- not
defensible as reported wins without multi-seed confirmation. The three extrapolation wins have
larger, more mechanistically-grounded margins (0.79x / 0.93x / 0.47x) and are the stronger claim
regardless of how springs-interp resolves.

**Attempted fix for springs-interp: free-running for interpolation, at three different stretch
lengths -- reverted, net regression.** Hypothesis: interpolation's longest unaided stretch is the
gap between two observations (not the full horizon extrapolation is judged on), so free-running
sized to that gap, rather than either the full horizon or nothing, might flip springs without
re-breaking IEEE39 (which regressed 0.891 -> 1.341 under full-horizon free-running for interp,
Part 22). Implemented `free_run_span` in `lib/gil_ode.py`: without it, one branch runs to the end
of the grid (extrap, unchanged); with it, the free state re-branches every `span`-length instead
of once, tiling the window with short unaided rollouts, sized in `lib/baseline_gil_ode.py` to the
mean inter-observation gap (window length / mean observations per node). Tested all three ways on
all three interp cells (not just the loser -- this knob has already flipped springs and IEEE39 in
opposite directions once, so none of the three could be assumed to hold fixed):

| Cell | Off (Part 25 table) | Full-horizon (Part 22) | Gap-tiled |
|---|---|---|---|
| Springs interp | 0.0759 (loss) | 0.068 (win) | **0.0656 (win)** |
| Charged interp | **0.2613 (win)** | -- | 0.2995 (loss) |
| IEEE39 interp | **1.065 (win)** | 1.341 (loss) | 1.216 (loss) |

Gap-tiling does beat both prior settings on springs-interp (0.0656, better than either 0.0759 or
0.068), disproving "rollout length doesn't matter" -- but it costs both other cells this time,
netting **4/6, worse than the 5/6 with it off**. Springs improves under *any* amount of unaided
interpolation-time rollout tried so far; charged and IEEE39 both get worse under *any* amount
tried so far. That pattern across three different settings reads as a genuine per-dataset
disagreement about interpolation training, not a mis-sized knob -- so this is reverted (free-run
stays extrap-only; `free_run_span` is kept in `gil_ode.py` as dead-but-tested capability, not used
by any current caller) rather than chased to a fourth stretch length or, worse, tuned per dataset,
which would not be defensible in a paper.

**Next**: multi-seed confirmation (3 seeds) across all six cells is the priority regardless of
springs-interp's outcome -- three of the five current wins are narrow enough (0.9-3%) that a single
seed cannot support calling them wins.

## Part 26 — Observation-subspace anchoring; a silent zero-contribution bug; four negative results

The headline: a single configuration now wins **5 of 6** cells with larger margins than Part 25's
5 of 6, losing a *different* cell. Getting there also uncovered a bug that invalidates how Part 25
attributed its own results, and produced four substantial negative results worth reporting in
their own right.

### 26.1 The bug: free-running supervision contributed exactly zero from Part 25 onward

Part 25 made free-running "horizon-matched" by branching the uncorrected rollout at
`grid_times[-1] - horizon`. But in extrapolation `grid_times[-1]` is the last decoder QUERY, which
lies beyond every observation, so the branch landed one step *past* the final observation. Nothing
followed it to supervise, `free_preds` stayed empty, `free_run` came back `None`, and the term
contributed nothing. Verified directly:

```
grid steps T                = 59
last grid idx WITH an obs   = 28
free_run_from computed      = 29    <- one step PAST the last observation
observations after branch   = 0
free_run returned           = None
```

Visible in every log since as `Loss` exactly equalling `-Likelihood` (Part 25 charged-extrap:
453.86 vs 233.02, the gap being AE-consistency alone; anchoring runs: 0 gap). **Every
extrapolation result from Part 25 onward -- including springs-extrap 1.591 -- was produced with
free-running inactive.** `gl-ode.md` calling it "the single biggest lever on extrapolation"
describes Parts 23/24 only; that claim has been corrected there. A loss with several auxiliary
terms needs its composition asserted, not eyeballed, and `Loss == -Likelihood` is the signature of
silent auxiliary failure.

### 26.2 Observation-subspace anchoring (the change that worked)

Anchoring set an observed node's state to `encoder_proj(y)` outright, pushing `input_dim` observed
features into all `hidden_dim` dimensions. Measured, that destroys **76 of 80 latent dimensions at
every observation event** -- the decoder's row space is only 4-dimensional, so the other 95% were
overwritten by information that cannot determine them. ODE-RNN avoids this by combining the
observation with the previous hidden state.

Now the correction is confined to the observable subspace. With linear `D(h) = hW^T + b` and
innovation `e = y - D(h^-)`, the minimum-norm correction is `K e` for
`K = W^T (W W^T + eps I)^-1`, giving `D(h^- + K e) = y` while leaving the decoder's null space
untouched. `K e` is supplied to `GraphLifting` as the anchor correction, so unobserved neighbours
receive the lifted *subspace* innovation. `encoder_proj` is deleted, and the AE-consistency term
with it -- that term existed to make `D(E(y)) ~= y` approximately, which anchoring now makes exact
by construction. Verified: `max |D(h+) - y| = 3.0e-05` (0.3% of the observation std), null-space
leak `3.6e-07`.

Three numerical formulations were needed. An absolute `eps=1e-6` solve raised "matrix is singular"
at epoch 29; `pinv` raised "svd failed to converge" at epoch 14; the shipped version is a
Tikhonov solve with a ridge scaled to the Gram matrix's mean diagonal, which is positive definite
by construction and cannot fail.

**Gradients flow through the gain deliberately.** The intuition that this is dangerous -- a
near-singular `W` gives an enormous `K`, and the anchor forces `D(h^+) = y` however large `K` is --
is exactly backwards when measured. Detaching let the decoder drift (condition number 1.3 -> 2.9
on charged, 1.6 -> 5.0 on IEEE39, `||K||` 2.20 -> 3.47) and cost accuracy (charged-interp
0.2195 -> 0.2658). Connected, it stays well conditioned (1.3 -> 1.6, `||K||` flat at ~2.1): the
loss can feel that an inflated gain produces damaging corrections. The gradient path is a
stabiliser, not an attractor.

### 26.3 The null-space prior

Anchoring constrains only `input_dim` directions; the other 76 are touched by no observation and
bounded by nothing. Measured on springs extrapolation, `||h_null||` grows **3.86x** across the
forecast against the observable part's 1.96x, ending 2.3x larger than the component the decoder
can see, and growing monotonically even inside the observed window. The overwrite rule held this
at zero implicitly; dropping it left an improper flat prior. A weak zero-mean Gaussian prior on
the null component (`P = K W` is the row-space projector; penalty on `h - h P`, weight 1.0, ~1% of
the reconstruction term) restores a proper one without erasing the memory anchoring exists to
preserve.

### 26.4 Result: 5 of 6 under one configuration

Subspace anchoring + null prior + free-running disabled, `hidden_dim=80`, `mlp_width=92`, one
configuration, no per-dataset tuning (`run_logs/final26_gil_*.log`):

| Cell | GIL-ODE | Bar | Ratio | Verdict | Part 25 |
|---|---|---|---|---|---|
| Springs interp | **0.0654** | 0.0689 | 0.95x | WIN | 0.0759 (loss) |
| Springs extrap | 3.204 | 2.021 | 1.59x | loss | **1.591** (win) |
| Charged interp | **0.2540** | 0.2652 | 0.96x | WIN | 0.2613 |
| Charged extrap | **4.423** | 5.403 | 0.82x | WIN | 5.017 |
| IEEE39 interp | **1.089** | 1.102 | 0.99x | WIN | 1.065 |
| IEEE39 extrap | **3.863** | 11.284 | 0.34x | WIN | 5.273 |

Springs-interp -- the cell four separate free-running mechanisms failed to flip -- falls out for
free, and charged-extrap and IEEE39-extrap improve substantially (0.82x and 0.34x against 0.93x
and 0.47x). The cost is springs-extrapolation, which is now a *training instability* rather than
merely a worse number (26.6).

### 26.5 Negative result: free-running cannot be extended to interpolation

Four mechanisms, each tested on all three interp cells:

| Mechanism | springs | charged | IEEE39 |
|---|---|---|---|
| Off | 0.0759 loss | **0.2613 win** | **1.065 win** |
| Full-horizon branch | 0.068 win | -- | 1.341 loss |
| Gap-tiled branches | **0.0656 win** | 0.2995 loss | 1.216 loss |
| Global PCGrad gate | *provably inert* | *inert* | *inert* |
| Gradient isolation to vector field | **0.0654 win** | 0.2839 loss | 1.357 loss |

PCGrad was inert for a measurable reason: the two gradients are strongly *aligned* (mean cosine
+0.88, never once negative in 40 steps), so the gate never fires and the run reproduced ungated
free-running to identical MSE. Per-layer diagnostics then explain why no routing rule can work --
the conflict sits in the assimilation modules on springs (`lifting`/`gate`, 20-30% of steps,
cosine to -0.99), in the *vector field* on charged (`phi_pos`/`phi_neg`, 40-45%), and essentially
nowhere on IEEE39. There is no common set of parameters to protect.

### 26.6 Negative result: free-running is incompatible with subspace anchoring

With the 26.1 bug fixed, free-running became active for the first time since Part 24 -- and
springs extrapolation diverged under every variant: weight 1.0 (epoch 15), weight 0.2 (epoch 15 --
the *identical* failure epoch at 5x the weight is what ruled magnitude out), horizon-matched span
(epoch 15), and the random branch point Parts 23/24 actually validated (epoch 9). Decoder
conditioning was healthy throughout every one (1.28 -> 1.35), so these were divergences, not rank
collapse. Parts 23/24 ran free-running against the *overwrite* rule, which reset the state every
observation; anchoring preserves precisely the unconstrained directions the free-run branch then
rolls forward uncorrected. Disabled, which changes no reported number since it had been inert
anyway.

### 26.7 Open: springs-extrapolation never converges under anchoring

Across five variants, training shows the same signature -- descend to ~0.00055 by epoch ~6, blow
up 500-1700x, partially recover:

| Config | result | blowup |
|---|---|---|
| no prior | 2.719 | epoch ~4 |
| prior 1.0, pinv gain | 2.301 | epoch ~4 |
| prior 1.0, Tikhonov | 3.204 | epoch ~4 |
| prior 1.0, clip=1 | 3.204 | epoch ~4 |
| prior 10.0 | 3.095 | epoch ~4 |

**The 2.301-3.204 spread is sampling noise from a non-converging process, not a ranking.** Earlier
attributions of differences within that range (e.g. crediting the null prior with a 15%
improvement) were reading signal into noise and are retracted; the prior's real effect is on
recovery and checkpoint selection (best-val epoch 1 -> 40-45), not on the blowup. Gradient
clipping at 1.0 changes nothing. The likely mechanism is drift -> larger state -> stiffer learned
dynamics -> fixed-step rk4 failure, which the overwrite rule never reached because it reset the
state. The remaining fix is an adaptive-step solver, which would change the integration scheme for
every model and require re-establishing the entire comparison -- out of proportion to one cell.

### 26.8 Built but not yet usable: horizon-gated residual adapter

A stagewise-boosting adapter is implemented and verified: `dv += a(h, tau) * R(h)` with a residual
MLP and a gate on `log1p(tau)` (time since last observation), both output layers zeroed so an
adapted model is bit-identical to its base (`max |dh_with - dh_without| = 0.000e+00`), routed into
`dv` only so `dq/dt = v` still binds, 17,217 parameters (total 271,166, inside the 247-273K band).
`--adapt-from` freezes all 253,949 base parameters and trains only the adapter; only trainable
parameters go to the optimizer, since AdamW's decay bypasses `.grad` and would otherwise keep
decaying "frozen" weights. It cannot be applied yet: its premise is freezing a proven base, and on
springs-extrapolation -- the only cell needing it -- no converged base exists (26.7).

### 26.9 Methodological note

Five diagnoses this session were stated confidently and then contradicted by measurement that took
minutes: conflict concentrated in the assimilation modules (charged showed 0% there); detaching
prevents a degenerate attractor (it causes drift); rank collapse caused the crashes (conditioning
was healthy); the free-run weight caused divergence (identical failure epoch at 5x the weight);
the free-run span caused divergence (the validated random point diverged sooner). A dead run was
also reported as "still training" twice, from the epoch counter rather than the process. The
lesson is procedural, not incidental: the cheap measurement should precede the mechanism story,
and any claim about *why* a number moved should be checkable before it is written down.

## Part 27 — 6/6 against all five baselines, three paired seeds

### 27.1 Correction: "5/6" (Parts 25-26) was measured against two of five baselines

When the project moved to stratified subsets (Part 20), only ODE-RNN and Corrected LG-ODE were run
on the new protocol, and each cell's bar was defined as "the better of these two"
(`gl-ode.md` §3). Every "5/6" and "6/6" statement in Parts 25-26 inherits that restriction and was
reported without saying so -- even though the full-scale table already showed Latent-ODE (0.014)
and RNN-NRI (0.026) beating ODE-RNN (0.055) on springs interpolation. When Latent-ODE, Edge-GNN and
RNN-NRI were finally run on the subset protocol, two "wins" were losses: springs-interp (Latent-ODE
0.0170 vs GIL 0.0654) and IEEE39-interp (RNN-NRI 0.8625 vs GIL 0.9923). Honest standing at that
point against all five baselines was 2 wins, 2 losses, 2 unverified. All three omitted baselines
needed `--val-fraction` added (one line, identical to the other scripts) to train on the same
5,000/1,000 split.

### 27.2 Protocol fix: the original charged baselines trained on 5,400 trajectories

An audit of every comparison log's logged training-set size found one group off: Corrected LG-ODE
and ODE-RNN on both charged tasks (Part 20's subset runs) were launched without `--val-fraction`,
so the default 10% hold-out gave them 5,400 training trajectories instead of 5,000 -- 8% more data
than every other model, in the baseline's favour. All four were rerun at 5,000
(`run_logs/v5k_*`); LG-ODE charged-extrap at seeds 1991-1993 since it is that cell's hardest
baseline. It got slightly worse (5.40 -> 5.0-5.5), as expected. Every other run checked out:
springs 5,000, charged 5,000, IEEE39 7,000, all models.

### 27.3 Diagnosis: GIL's interpolation error was a cold start, not a gap problem

The two lost cells were lost to exactly the two baselines that read the future: Latent-ODE encodes
backwards over all observations; RNN-NRI uses a bidirectional GRU. GIL is a causal filter -- its
state at time t has seen only observations up to t. Bucketing springs-interp test error by region
(evaluation only, existing checkpoint) located the problem precisely:

| region | targets | MSE | share of error |
|---|---|---|---|
| before a node's first observation | 3,359 | 4.3e-02 | **99.9%** |
| between observations | 84,286 | 1.3e-06 | 0.1% |
| after last observation | 3,266 | 5.4e-07 | 0.0% |
| on an observation | 131,745 | 2.1e-13 | 0.0% |

Between observations GIL is near-perfect; the entire loss is the region before a node is first
observed, where the filter has seen nothing about it and predicts from an essentially zero state.
(An initial version of this diagnostic silently excluded those edge targets and appeared to show a
gap problem; reconciling it against the wrapper's reported MSE exposed the omission.)

### 27.4 Architecture: bidirectional cold-start smoothing (`lib/gil_ode.py`)

The same dynamics, anchoring, lifting and gate run backwards in time, sharing every parameter with
the forward pass (no new parameters). Per node and target, tau_f = time since the previous
observation and tau_b = time until the next. Two fusion rules were implemented:

- `full`: backward weight tau_f / (tau_f + tau_b) wherever a future observation exists;
- `coldstart`: backward state ONLY before a node's first observation (tau_f = inf), forward elsewhere.

In extrapolation no target has a future observation, so the backward pass is skipped and output is
**bit-identical** to the filter (verified, max diff 0.0) -- existing extrapolation results stay valid.
`_integrate` was also fixed to apply error control on decreasing-time steps; the old guard
`t0 < t_mid < t1` is always false for backward steps and would have silently disabled it.

Zero-shot on filter-trained checkpoints (no retraining), springs-interp went 0.0652 -> 0.0002. But
charged got far worse (0.21 -> 1.41 full, 0.46 cold-start): the backward pass ran dynamics trained
only forwards, and charged's complete interaction graph already lets lifting inform a node before
its own first observation, so its filter was never truly cold. `coldstart` beat `full` on all three
datasets. **This mode choice was first read off test MSE; it was then re-checked on the validation
split, which ranks the modes identically on all three datasets** (validation MSE x1e-2,
filter / full / coldstart: springs 0.0628 / 0.0002 / 0.0001; charged 0.2030 / 1.3650 / 0.4319;
IEEE39 0.9728 / 1.2469 / 0.5850). Trained WITH the smoother on, charged's backward dynamics learn to
run in reverse: validation 0.139 vs the filter-trained 0.203.

### 27.5 Correction: free-running supervision is not incompatible with anchoring (retracts 26.6)

Part 26.6 concluded free-running "diverged under every variant" and disabled it. Every one of those
runs used the fixed-step integrator later shown to fail on its own (26.7). Retested under
error-controlled integration (`--free-run`, weight 1.0), it trained stably, its validation fell
monotonically to its best at epoch 47 (vs a peak at epoch 17-33 and degradation without it), and
springs-extrap went 1.979 -> 0.411. That removes the overfitting diagnosed in 26.7, which was the
vector field never receiving long-rollout gradient. Weight decay was irrelevant once it was on
(0.411 at l2=1e-3, 0.413 at 3e-3). 26.6 and 26.7's attribution of the springs-extrap failure to
anchoring are retracted: the cause was the integrator, then the missing rollout signal.

### 27.6 Result: one configuration, all five baselines, three paired seeds

Flags identical for every cell (`run_logs/final_config.sh`, `run_logs/seeds_final.sh`):
`--ode-tol 1e-3 --smoother --smoother-mode coldstart --free-run --free-run-weight 1.0`, on top of
subspace anchoring + null-space prior, hidden_dim 80, mlp_width 92, 50 epochs, validation-selected
checkpoints, identical splits for every model. Each cell is compared, seed for seed, against its
hardest baseline (selected at seed 1991 as the best of all five). MSE x1e-2:

| cell | GIL mean ± sd | hardest baseline mean ± sd | ratio | paired wins | paired diff, 95% CI (exploratory, n=3) |
|---|---|---|---|---|---|
| springs interp | 3.58e-5 ± 0.10e-5 | Latent-ODE 0.0177 ± 0.0013 | ~1/490 | 3/3 | [-0.0192, -0.0170] |
| springs extrap | 0.350 ± 0.053 | LG-ODE 1.721 ± 0.261 | 0.20 | 3/3 | [-1.610, -1.231] |
| charged interp | 0.154 ± 0.012 | RNN-NRI 0.239 ± 0.001 | 0.65 | 3/3 | [-0.096, -0.074] |
| charged extrap | 3.628 ± 0.189 | LG-ODE 5.235 ± 0.263 | 0.69 | 3/3 | [-1.909, -1.352] |
| IEEE39 interp | 0.517 ± 0.016 | RNN-NRI 0.870 ± 0.007 | 0.59 | 3/3 | [-0.374, -0.331] |
| IEEE39 extrap | 4.384 ± 0.055 | ODE-RNN 10.845 ± 0.395 | 0.40 | 3/3 | [-6.883, -6.087] |

**18/18 paired comparisons won; the smallest margin is 31%.** Seed-1991 springs-extrap ran with
`--free-run` but without `--smoother`, which is bit-identical in extrapolation.

Free-running helped springs-extrap (5x) and charged-extrap (4.42 -> 3.61) but cost IEEE39-extrap a
little (3.72 -> 4.38); it still wins that cell by 2.5x.

### 27.7 Caveats that belong in the paper

- Only the hardest baseline per cell was seeded (compute); "hardest" was chosen at seed 1991. The
  runners-up were well behind there (e.g. charged-interp ODE-RNN 0.266 vs RNN-NRI 0.238).
- Three seeds; intervals are exploratory.
- The integrator, smoother and free-running are part of GIL's method; baselines use their own
  published integration schemes unchanged.
- Infrastructure: six concurrent jobs (~2.6 GB each) exhausted WSL's 15 GB and crashed the VM,
  killing every job and the session; runs since are capped by a >= 6 GB-free guard per launch.

### 27.8 Complete main-results table: all baselines, three seeds

27.6 seeded only each cell's hardest baseline. The remaining 48 baseline runs (seeds 1992/1993 for
every baseline x cell not yet seeded; `run_logs/baselines_seeds.sh`) were added so every entry is a
3-seed mean, with commands mirroring the seed-1991 runs exactly. Result: **90/90 paired wins** --
GIL beats every baseline on every seed in every cell -- and the strongest baseline by 3-seed mean
matches the seed-1991 choice in all six cells, closing that caveat from 27.7.

| model | Springs interp | Springs extrap | Charged interp | Charged extrap | IEEE39 interp | IEEE39 extrap |
|---|---|---|---|---|---|---|
| **GIL-ODE** | **3.58e-5 ± 0.10e-5** | **0.350 ± 0.053** | **0.154 ± 0.012** | **3.628 ± 0.189** | **0.517 ± 0.016** | **4.384 ± 0.055** |
| LG-ODE | 0.338 ± 0.052 | 1.721 ± 0.261 | 0.971 ± 0.054 | 5.235 ± 0.263 | 8.584 ± 0.680 | 13.633 ± 0.183 |
| ODE-RNN | 0.066 ± 0.003 | 5.141 ± 0.253 | 0.263 ± 0.002 | 6.757 ± 0.066 | 1.092 ± 0.010 | 10.845 ± 0.395 |
| Latent-ODE | 0.018 ± 0.001 | 9.722 ± 0.330 | 0.616 ± 0.010 | 6.558 ± 0.137 | 7.711 ± 0.030 | 15.048 ± 0.124 |
| Edge-GNN | 0.370 ± 0.051 | 2.641 ± 0.246 | 1.217 ± 0.074 | 6.299 ± 0.806 | 9.000 ± 0.784 | 14.961 ± 0.947 |
| RNN-NRI | 0.040 ± 0.000 | 6.433 ± 0.852 | 0.239 ± 0.001 | 12.640 ± 1.642 | 0.870 ± 0.006 | 110.51 ± 0.03 † |

† RNN-NRI diverged in its first epoch on IEEE39 extrapolation on all three seeds (training MSE
1.4e5 at epoch 1) and converged to a constant predictor (training MSE ~1.11, the variance of the
normalized data, for the remaining 49 epochs). Its discrete recurrent forecast rollout has no bound
on the state, and over IEEE39's horizon it explodes. It trained under the same protocol and the
same gradient clipping as every other model, and is reported as-is rather than re-tuned, since
per-model tuning is not applied to any model. Its IEEE39 interpolation trains normally (0.870).

Infrastructure note: the RAM guard was raised from 6 to 8 GB free per launch after three concurrent
Edge-GNN IEEE39 jobs (~3.7 GB each) left under 3 GB free; the relaunch lost only a few minutes.

### 27.9 CSG-ODE (ICML 2025) added as the final baseline: 108/108 paired wins

CSG-ODE has no released code; reimplemented in `lib/csg_ode.py` (equations 1-17 and Appendices A-B,
Table 8 hyperparameters, decoder scheme from the ControlSynth Neural ODE reference code; every
choice the paper leaves open is listed in the file header). Verified before training: the
finite-difference Frechet derivative behind the edge-importance weights matches the exact derivative
to 6.8e-9 (computed in float64 -- with beta ~ 4e-5 float32 cancels), and the sampling density matches
Appendix A. `run_models_csgode.py` is the Corrected LG-ODE runner with only the model swapped.

Implementation fix, results-neutral: the first version held every encoder timestep's and every Euler
step's intermediates for backward -- ~30 GB of GPU memory on IEEE39, which saturated the card that also
drives the display and made the desktop lag. Per-step gradient checkpointing (plus computing the
node-specific weights once per forward) gives bit-identical loss and gradients at 1.7 GB, 3x faster.

| CSG-ODE ‡ | 0.048 ± 0.005 | 3.627 ± 0.305 | 1.116 ± 0.080 | 7.421 ± 0.655 | 11.402 ± 0.459 | 17.816 ± 0.064 |

‡ CSG-ODE (Wang, Wang & Liang, ICML 2025) has no released code (checked GitHub, OpenReview and the
ICML page). It is reimplemented in `lib/csg_ode.py` from the paper's equations with its Table 8
hyperparameters (357,678 parameters, more than GIL-ODE's ~254-271K); the decoder's numerical scheme
follows the authors' cited ControlSynth Neural ODE reference code. It is trained by
`run_models_csgode.py`, which is the Corrected LG-ODE runner with only the model swapped. Reported
with the paper's own Euler solver; RK4 (LG-ODE's solver here) is within ±5% in every cell.

**Faithfulness check** (full-scale springs, 60% observation, the paper's own setting):
interpolation 0.018 vs the paper's 0.144 (ours 8x better); extrapolation 2.212 vs 1.297 (ours 1.7x
worse; our LG-ODE also sits 1.22x above its published extrapolation number on this cell). **Robustness
to that gap:** dividing every CSG-ODE extrapolation result by the full 1.705 factor still leaves GIL-ODE
ahead in all three extrapolation cells (springs 0.350 vs 2.08, charged 3.628 vs 4.35, IEEE39 4.384 vs
10.29); charged-extrapolation is the narrowest under that generous assumption.

Solver comparison (3-seed test means, Euler vs RK4): springs 0.048/0.048, 3.63/3.54; charged
1.12/1.12, 7.42/7.82; IEEE39 11.40/11.62, 17.82/17.55. Training curves show CSG-ODE still improving at
epoch 50 on four of six cells (best-val epoch 49-50); all models share the 50-epoch protocol.

## Part 28 — Metric correction: GIL-ODE extrapolation was scored on an easier metric (retraction)

### 28.1 The bug

Every GIL-ODE extrapolation result reported up to Part 27 was the **multi-horizon training
metric**, not the metric the baselines report. In extrap mode, `lib/baseline_gil_ode.py`'s
`compute_all_losses` returned `mse = mean over the 20/40/60/80/100% horizon prefixes` (Part 23),
batch-pooled. Nothing gated that on `self.training`, so it also produced the validation MSE used for
checkpoint selection and the `FINAL [Test seq]` line. Every baseline reports
`VAE_Baseline.get_mse` over the full horizon. Forecast error grows with horizon, and the prefix
average counts early steps up to five times, so the GIL-ODE extrapolation numbers were
systematically low. Interpolation was unaffected: that path always used `get_mse`.

Found while planning the paper's experiments, by reading the wrapper. It was then measured
directly with `eval_gilode.py` (new, evaluation only), which rebuilds the exact split from the
checkpoint's args and computes both metrics on the same predictions. Its prefix metric reproduces
the logged FINAL numbers to every printed digit.

| cell | prefix metric (as reported) | standard full-horizon | ratio |
|---|---|---|---|
| springs extrap (3 seeds) | 0.350 | **0.873 ± 0.143** | 2.5x |
| charged extrap | 3.628 | **6.233 ± 0.255** | 1.7x |
| IEEE39 extrap | 4.384 | **7.036 ± 0.117** | 1.6x |

### 28.2 Consequence for the main table

On the standard metric, GIL-ODE still wins springs-extrap (0.873 vs LG-ODE 1.721) and IEEE39-extrap
(7.04 vs ODE-RNN 10.85) on every seed against every baseline. **It loses charged-extrap**:
6.233 vs LG-ODE 5.235. Paired, it wins 13/18 in that cell: it beats Edge-GNN on one of three seeds
and LG-ODE on none. The Part 27 claims "108/108" and "6/6" are withdrawn; the correct count is
**5/6 cells, 103/108 paired comparisons**. The CSG-ODE robustness paragraph (27.9) also used the
prefix numbers. On the standard metric, the 1.705-rescaled CSG-ODE values (springs 2.08, charged
4.35, IEEE39 10.29) beat GIL-ODE on charged-extrap, and GIL-ODE still wins springs and IEEE39.

### 28.3 Fixes

- `compute_all_losses` now always reports the standard `get_mse`. The prefix average remains a
  training loss only, so checkpoint selection for new runs uses the baselines' metric.
- `run_logs/main_table.py` reads GIL-ODE extrapolation cells from the standard-metric rescore of the
  same validation-selected checkpoints (`run_logs/eval/rescore_extrap.jsonl`), not from the logs.
  Those checkpoints were selected on prefix-averaged validation MSE. That is still a
  validation-only criterion, but not the one new runs use.
- Baseline-side check: no other model has a prefix or horizon metric (`grep`). `eval_baseline.py`
  re-evaluates any baseline checkpoint through its own run script and reproduces LG-ODE's logged
  test MSE to 0.35%; LG-ODE samples z0, so the evaluation is not bit-exact.

### 28.4 What is being run (`run_logs/icml_queue.sh`)

- **T6 (full-horizon training loss)**, `--horizon-loss full`, three extrap cells x 3 seeds. This is
  the planned ablation of the multi-horizon loss. With the metric fixed, it is also the direct test
  of whether that loss trades full-horizon accuracy for early-horizon accuracy. Whichever
  loss is adopted will be chosen on validation MSE and applied to all three cells.
- **Control:** the current config retrained on charged-extrap with the corrected selection metric
  (3 seeds).
- **A1 (no lifting)** on springs-interp x 3 seeds, the first architecture ablation (`--ablate
  no_lift`; all ablations are listed in `lib/gil_ode.py`).
- **Evaluation-only analyses** on all 18 final checkpoints: causal filter (smoother off),
  lifted-correction magnitude, test-time lifting knockout, agent dropout (k masked agents, error on
  the masked ones), complete/rewired graph, IEEE39 conductances, and figure dumps.


### 28.5 Results of the first ICML queue (2026-10-08)

**Charged-extrap is still a loss** (MSE x1e-2, standard metric, seeds 1991/1992/1993):

| variant | per seed | mean | vs LG-ODE 5.235 |
|---|---|---|---|
| original (prefix loss, prefix selection) | 6.378 / 6.383 / 5.939 | 6.233 | loss |
| prefix loss, standard selection | 6.350 / 6.358 / 5.939 | 6.216 | loss |
| T6 full-horizon loss | 6.503 / 5.682 / 6.018 | 6.068 | loss |

T6 elsewhere: springs-extrap 3.306 (seed-1991 run degraded; val 0.0336) / 0.779 / 0.715.
IEEE39-extrap 6.825 for seed 1991; the other two seeds are still running. Neither the
selection metric nor the training loss closes the charged-extrap gap (16% at best).

**Interpolation holds on held-out targets.** In interpolation the decoder targets include the
conditioning observations: 59% of springs targets are inputs, and GIL-ODE reproduces those
exactly (MSE ~2e-13). Scoring only targets that were **not** inputs, every model, 3 seeds
(x1e-2):

| cell | GIL-ODE | best baseline | GIL-ODE causal (no smoother) |
|---|---|---|---|
| springs | 0.00009 | Latent-ODE 0.0195 | 0.157 (loses to Latent-ODE and RNN-NRI) |
| charged | 0.377 | RNN-NRI 0.585 | 0.456 |
| IEEE39 | 1.293 | RNN-NRI 2.175 | 2.744 (loses to RNN-NRI and ODE-RNN) |

GIL-ODE wins every seed in all three cells, with ratios close to those on the full target set.
The springs and IEEE39 wins depend on the backward smoother. CSG-ODE is not in this table yet:
its checkpoints predate `--csg-solver`; fixed in `eval_baseline.py`, rerunning.

**Lifting is inactive on springs and charged.** The ratio of the mean lifted correction on
unobserved agents to the anchored correction on observed ones is 0.003-0.004 on springs and
charged (0.0004 charged-extrap), against 0.27-0.29 on IEEE39.

- Test-time knockout (delta_U = 0) changes the MSE by <1% on springs and charged. On IEEE39 it
  costs +9% (interp) and +1% (extrap).
- Retrained A1 (no lifting) on springs-interp: 3.62e-5 vs 3.58e-5 for the full model, so no
  effect.
- Agent dropout (all observations of k agents masked): on springs and charged, the masked
  agents' error is the same with or without lifting. On IEEE39, lifting reduces it by 11-20%,
  but the masked-agent error stays near the level of a constant predictor (~88 vs ~102 without
  lifting).

Where GIL-ODE does win, on springs and charged, the gain comes from anchoring, the dynamics and
the smoother, not from lifting. The graph itself matters (test-time complete or rewired graph:
springs-interp x1242 / x387, IEEE39 x1.3-1.7), through the vector field's coupling. The paper's
lifting claim is currently supported only on IEEE39. A1 is now running on IEEE39 (3 seeds x 2
tasks, `run_logs/a1_ieee39_queue.sh`).


## Part 29 — Mean-reverting forecast head (charged-extrap go/no-go)

**Diagnosis.** In charged-extrap GIL-ODE is the most accurate model in the first fifth of the
horizon (1.40 vs LG-ODE 2.11, x1e-2), and the worst-growing after that (12.26 vs 10.28 in the last
fifth). Springs and IEEE39 show no crossover. The test was evaluation only, with no fitting on
test: blend each prediction toward the per-feature mean of the validation targets, with one
weight per forecast step fitted on **validation**, then score test. Results:

- charged: 6.384 / 6.390 / 5.945 -> 5.605 / 5.685 / 5.336 (mean 6.24 -> 5.54). The fitted weight
  grows from 0.05 in the first fifth to about 0.33 in the last.
- springs: the weight is ~0 (0.00-0.06) and the change is negligible.
- averaging the three seed models' predictions gives charged 5.224. That is an ensemble, reported
  for information only and never used.

So the remaining charged gap is a variance penalty. A deterministic rollout commits to one
trajectory under chaotic coupling, where the MSE-optimal forecast relaxes toward a conditional
mean.

**Change.** `--reversion` (lib/gil_ode.py, extrapolation only, like free-running):
y_hat_i = (1 - s_i) D(h_i) + s_i mu_i, with
- s_i = sigmoid(kappa([h_i, log(1 + tau_i)])): the gate grows with the time since agent i's last
  observation;
- mu_i = M [h_i(last obs), mean_j h_j(last obs)] + m: a graph-conditional mean, per trajectory.

It starts as a near no-op (gate 0.01) and adds 3.3K parameters. It is trained with the
full-horizon loss (T6), since the prefix loss down-weights exactly the late steps where reversion
matters. The same flags are used on all three extrapolation cells. The T6 runs (Part 28.5) are
its "no head" ablation.

**Queue** (`run_logs/reversion_queue.sh`): charged x3 first, then springs x3, then IEEE39 x3. A1
on IEEE39 was paused 1 h into its first run and is re-queued after this. The final configuration
will be adopted only if the head wins on validation, and is reported on test once for all three
cells.


### 29.1 Result: the head helps but does not close charged-extrap

Test MSE x1e-2 (standard metric), seeds 1991/1992/1993, `--reversion --horizon-loss full`:
- charged 6.026 / 5.885 / 5.779 (mean 5.897; previously 6.233; LG-ODE 5.235). Still a loss.
- IEEE39 6.319 / 6.477 / 6.398 (mean 6.398; previously 7.036).
- springs 0.955 / 0.902 / 0.764 (mean 0.874; previously 0.873).

Validation prefers the head on charged and IEEE39. The learned gate is about 0.12 on charged and
about 0.02 on IEEE39. Fitting a further per-step blend on validation on top of these models gives
charged 5.43, and averaging the three seed models gives 5.30. Neither beats 5.235, and the ensemble
would be unfair anyway.

## Part 30 — Probabilistic GIL-ODE (graph ensemble-Kalman ODE), v1

Design: `paper/probabilistic_design.md`. The latent state is an ensemble of K = 8 particles
following a latent SDE: the error-controlled RK4 drift, then one Euler-Maruyama increment per
segment with a learned diagonal diffusion. Every particle is anchored and lifted as before. The
point forecast is the ensemble mean. The loss is the Gaussian likelihood of the ensemble mean
(full horizon) plus the fair ensemble CRPS / sigma_obs. Particles are folded into the batch
dimension, so the cost is only about 10-15% above deterministic (7.1 vs 6.3 s per batch on
charged) at 8.8 GB of GPU memory. Go/no-go: charged-extrap, 3 seeds (`run_logs/prob_queue*.sh`),
against LG-ODE 5.235 and the deterministic T6 run (6.07), which is the same configuration without
particles.

## Where to look (final corrected run)

- **Archive**: `RESULTS_ARCHIVE_PHASE1-3.md` (all 30-epoch, capacity-mismatched results)
- **Superseded**: everything under "final_*" logs and the current `RESULTS.md` table -- stale as
  of Part 19, kept for the historical record of the capacity-matching/leakage investigation but
  not to be read as current numbers.
- **Current results**: `RESULTS.md`, rewritten once the Part 19 rerun (`run_logs/final2_*.log`)
  completes.
- **Logs**: `run_logs/final_<model>_<dataset>_<interp|extrap>.log` (superseded),
  `run_logs/final_uncorrected_<dataset>_<interp|extrap>.log` (superseded),
  `run_logs/densegrid_corrected_spring_{interp,extrap}.log` (Part 17's disproven-hypothesis test,
  still valid as a negative result), `run_logs/structured_exp1_gilode_<dataset>_extrap.log`
  (Part 18, superseded -- GIL-ODE's message function itself is unaffected by Part 19's fixes,
  but the protocol it was measured under is)
- **Driver**: `run_logs/final_matrix.sh` (superseded), `run_logs/final_matrix_driver.log`

## Where to look (Phase 3)

- **GIL-ODE**: `run_models_gilode.py`, `lib/gil_ode.py`, `lib/gil_dataset.py`,
  `lib/baseline_gil_ode.py`, `run_logs/gilode_*.log` (v1), `run_logs/gilode_*_v2.log` (v2),
  `run_logs/gilode_*_tier1.log` (tier1, current)
- **Checkpoints**: `experiments_gilode/` (gitignored)
