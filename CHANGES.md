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
