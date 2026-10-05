# GIL-ODE: architecture, experiment log, and current standing

Working document for the proposed architecture (Graph Innovation-Lifting ODE). Covers what the
model is, every change tried on it, which changes worked, which didn't, and where it currently
stands against the baselines. Companion to `CHANGES.md` (full project history including the
reproduction and dataset work) and `RESULTS.md` (current results tables).

**Bottom line up front:** GIL-ODE wins **all 6 cells against all five baselines** (Corrected
LG-ODE, ODE-RNN, Latent-ODE, Edge-GNN, RNN-NRI), under **one configuration with identical flags for
every cell**, confirmed on **three paired seeds — 18 of 18 paired comparisons won, smallest margin
31%** (CHANGES.md Part 27). MSE x1e-2, GIL vs the hardest baseline in each cell:

| cell | GIL (3-seed mean) | hardest baseline (3-seed mean) | ratio |
|---|---|---|---|
| Springs interp | 3.58e-5 | Latent-ODE 0.0177 | ~1/490 |
| Springs extrap | 0.350 | LG-ODE 1.721 | 0.20 |
| Charged interp | 0.154 | RNN-NRI 0.239 | 0.65 |
| Charged extrap | 3.628 | LG-ODE 5.235 | 0.69 |
| IEEE39 interp | 0.517 | RNN-NRI 0.870 | 0.59 |
| IEEE39 extrap | 4.384 | ODE-RNN 10.845 | 0.40 |

**Read this before anything below.** Every "5/6" in this document and in CHANGES.md Parts 25-26 was
measured against only ODE-RNN and Corrected LG-ODE — the two baselines run on the subset protocol at
the time. Against all five, that configuration actually stood at 2 wins / 2 losses / 2 unverified.
The current result comes from two additions on top of Part 26's subspace anchoring and
error-controlled integration: **bidirectional cold-start smoothing** (GIL's interpolation loss was
99.9% concentrated before each node's first observation, where a causal filter has seen nothing;
see Part 27.3–27.4) and **free-running supervision**, which works once integration is correct and
was wrongly ruled out in Part 26.6 (retracted in 27.5). The sections below are the historical log
and contain conclusions since superseded; Part 27 is authoritative.

---

## 1. What the architecture is

GIL-ODE is a continuous-time latent model for irregularly-sampled, partially-observed multi-agent
systems. Its distinguishing mechanism versus everything else in this project:

- **LG-ODE** infers `z0` once from the whole context, then evolves and decodes — no correction
  during the rollout at all.
- **ODE-RNN** corrects only the observed node at each event, independently per node — an
  unobserved node is never touched by its neighbors.
- **GIL-ODE** corrects at every observation event *and* lifts that correction to unobserved
  nodes through a graph-regularized closed-form solve, on top of a continuous latent ODE whose
  vector field also mixes nodes.

Core components (`lib/gil_ode.py`):

| Component | Role |
|---|---|
| `GILODEFunc` | The latent vector field: local term + gated relational interaction |
| `GraphLifting` | Closed-form Laplacian-regularized solve that spreads an observed node's innovation to unobserved neighbors |
| `GateNet` | Weighs *inferred* corrections for unobserved nodes |
| `encoder_proj` / `decoder` | Linear maps between observation space and latent state |

Per-dataset graph construction (`lib/gil_dataset.py`): springs uses its own sparse binary graph;
charged uses a complete graph with ±1 sign as a relation feature; IEEE39 uses the Kron-reduced
physical generator-coupling graph (see `dataset.md`).

---

## 2. Important caveat on comparing numbers

The evaluation protocol changed substantially partway through this work, so **numbers from
different eras are not comparable**:

| Era | Protocol |
|---|---|
| Early (archived) | 30 epochs, mismatched parameter counts, **checkpoint selected on test**, springs on a small 1,800-graph demo set, extrapolation-val drawn from the train pool |
| Current | 50 epochs, parameter-matched (~250K), **checkpoint selected on validation**, springs on the full dataset, fixed stratified subsets, extrapolation-val drawn from a **disjoint slice of the test pool** (Part 25) |

The last row changed again partway through the "current" era: through Part 24, validation for
extrapolation was still drawn from the train pool (see §6's original diagnosis, kept below for the
record) and has since been fixed. Every table in this document from Part 25 onward uses the fixed
val split; anything using `hardq_gilode_*` or earlier logs does not, and the two are not directly
comparable on extrapolation cells specifically.

Everything in the tables below is from the **current** protocol unless stated otherwise. The
early-era results are kept in `RESULTS_ARCHIVE_PHASE1-3.md` for history only.

---

## 3. The bar to beat

Baselines under the current protocol (MSE ×10⁻², lower is better). These are this project's own
faithful reimplementations, not the original paper's code, which was never released — see
`RESULTS.md` "Baseline provenance."

| Model | sp-int | sp-ext | ch-int | ch-ext | ie-int | ie-ext |
|---|---|---|---|---|---|---|
| ODE-RNN | **0.069** | 5.374 | 0.265 | 7.631 | **1.102** | 11.284 |
| Corrected LG-ODE | 0.305 | **2.021** | 0.975 | **5.403** | 7.938 | 13.440 |

The cell-by-cell bar is the better of these two: `0.069 / 2.021 / 0.265 / 5.403 / 1.102 / 11.284`.

---

## 4. Full experiment log

### 4.1 Changes that worked

**Relation-expert messages + learned sum/mean aggregation** (`phi_pos`/`phi_neg` selected by sign
of `c_ij`; `psi` combining sum- and mean-aggregated messages plus `log(1+degree)`).
Motivation: physical force is additive over neighbors, but the original used mean aggregation,
which silently imposes degree-normalization; and one shared MLP was being asked to represent both
attraction and repulsion on charged. Plain sum alone (tested in isolation first) helped springs
but *hurt* charged, which is why the final version learns the combination rather than committing
to either.

**Per-node state-dependent gate**, replacing a single global scalar `alpha` shared across every
node, channel, timestep and trajectory. The scalar was demonstrably fragile: boosting its
learning rate helped IEEE39 but overshot on charged. Gate is warm-started near 0.1 via a biased
final layer.

**Hard-anchored innovation lifting.** Previously an observed node's correction was smoothed by
the Laplacian and reduced by the confidence gate, even though it has an exact observation
available and these datasets have no measurement-noise model. Now rows of the linear system for
observed nodes are the identity exactly (`delta_i = r_i`), unobserved rows unchanged, and the
gate applies only to *inferred* corrections. One `torch.where` replacing the whole row, not just
the diagonal entry that was contaminating it.

**Multi-horizon extrapolation loss.** The loss previously supervised only the full decode horizon
at once. Now it also supervises several prefix cutoffs (20/40/60/80/100%), giving early-horizon
accuracy more effective gradient weight — an early point contributes to every cutoff's term, a
late point only to the final one. Free in forward compute: GIL-ODE already produces predictions
at every decoder time step in one pass.

| Cell | Before | After | Change |
|---|---|---|---|
| sp-ext | 5.760 | 4.847 | −16% |
| ch-ext | 8.575 | 7.431 | −13% |
| ie-ext | 5.475 | 3.679 | −33% |

**Free-running supervision** — a large lever on extrapolation **in Parts 23/24 only**, and now
disabled entirely. Read the numbers in this subsection as applying to that era and no later:
a Part 25 change silently deactivated the mechanism (the branch point landed past the last
observation, so nothing was ever supervised and the term contributed exactly zero), which means
**no extrapolation result from Part 25 onward was produced with it active** — including
springs-extrap 1.591. When the bug was fixed and the term became live again, springs
extrapolation diverged under every weight and branch-point variant tried, because free-running is
incompatible with the subspace anchoring introduced in Part 26. See §4.2 and CHANGES.md Part 26.
The original diagnosis and the Parts 23/24 measurements below remain valid on their own terms.

Diagnosis: along the
main path a correction arrives every few steps and rescues the state, so the vector field is
trained almost entirely on short hops, then judged on a 40-step uncorrected forecast. LG-ODE and
Edge-GNN never correct, so *all* of their vector-field gradient is long-rollout gradient — which
is why Edge-GNN, the worst model on springs-interp, is second-best on springs-extrap. Fix:
during training, branch a second state off the main path at a random point in the context, roll
it forward with corrections off, and supervise its decoded predictions against the observations
it passes. Those context observations were previously consumed only as corrections, never as
rollout targets. The main path is a separate tensor and untouched, so unlike scheduled sampling
this adds signal without degrading the primary objective.

| Cell | Before | After | Change |
|---|---|---|---|
| sp-ext | 4.620 | 3.391 | −27% |
| ch-ext | 6.909 | 5.928 | −14% |

**Rollout-state compression** — the largest single jump on the hardest cell. Hypothesis came
from noticing that the two models beating GIL-ODE on springs-extrap (LG-ODE 2.021, Edge-GNN
2.296) share exactly one property it lacked: they roll out from a heavily compressed latent
(`latents=16` + 64-dim augment) while GIL-ODE rolled out a 152-dim state. Over 40 uncorrected
steps a high-dimensional state has far more directions in which to drift.

| hidden_dim | sp-ext |
|---|---|
| 152 | 3.391 |
| 80 | **2.389** |
| 48 | 2.856 |

80 is near a sweet spot rather than "smaller is always better," which supports drift rather than
mere over-parameterization as the mechanism. Cuts the gap to Corrected LG-ODE from 1.68x to
**1.18x**.

**Second-order latent state, once made binding** (see 4.2 for the failed first attempt). Hidden
state split into halves `h = [q, v]` with `dq/dt = v` exactly. Removes one degree of freedom the
vector field could otherwise get wrong over a long uncorrected rollout.

**Fixed extrapolation-validation split** (CHANGES.md Part 25). The val split had been drawn from
the train pool, so extrapolation-validation trajectories only ever posed a train-style short
forecast — see §6's diagnosis. Redrawing `val` from a disjoint slice of the *test* pool so it
poses the same genuine second-forecast-window task test does, applied uniformly to every model.

**Antisymmetric force channel** — the fix that flipped the hardest cell, springs-extrapolation.
Added a second message type, `f_anti_ij = chi(h_i,h_j) - chi(h_j,h_i)`, which satisfies
`f_ij = -f_ji` exactly by construction (Newton's third law imposed structurally, not hoped for
from a symmetric-input MLP), aggregated additively and fed into `psi` alongside the existing
sum/mean relation-expert messages. Verified post-training: `|f_ij + f_ji|` and self-force are
exactly 0. Rebudgeted `mlp_width` to keep the parameter count in the 247–273K band.

| Cell | Without channel | With channel | Change |
|---|---|---|---|
| sp-ext | 2.560 | **1.591** | −38% |

**Earlier fixes that mattered** (pre-current-protocol, so magnitudes aren't comparable, but the
direction held): warm-starting the scalar `alpha` off exactly zero, and giving it a boosted
learning rate — it had been starved of gradient and settling back near its initialization.

### 4.2 Changes that did not work

**Second-order state with an `eps_q` residual — failed, and the diagnosis is instructive.**
Intent was `dq/dt = v + eps_q(h)`, a hard skeleton plus a *small* learned correction. Results got
worse (sp-ext 4.847 → 5.003, ch-ext 7.431 → 7.576). Measured the two terms on a trained
checkpoint:

```
||v||         (hard skeleton term):  0.372
||eps_q(h)||  ("small" residual):    0.878    ← 2.37x LARGER
```

Nothing constrained `eps_q`'s magnitude — it's a full-capacity MLP with no penalty for growing —
so it grew until it dominated, leaving `q`'s dynamics free-form again. The model paid the
constraint's entire cost (`f_self`/`f_interaction` output dims halved by the split) for none of
its benefit. Removing `eps_q` entirely so `dq/dt = v` exactly turned a 3% regression into a
result better than the previous best. **The conclusion "second-order structure doesn't help" was
wrong; the structure had simply never been active.**

**Plain sum aggregation alone** (before the learned sum/mean combiner). Helped springs (variable
degree, additive force argument applies cleanly) but hurt charged, where the complete graph means
dropping degree-normalization is an unmodulated ~4x scale-up that destabilized training. Fixed by
learning the combination instead of choosing.

**The "5x denser ODE time grid"** described in the LG-ODE paper's appendix but absent from its
released code. Implemented and tested in isolation: no meaningful change (sp-int 0.573 → 0.572,
sp-ext 2.305 → 2.523) at ~5x the compute per epoch. Reverted. Documented as a negative result —
it is a real paper/code divergence, just not a consequential one.

**Free-running supervision has a real cost, not just a benefit.** It helped springs and
charged extrapolation substantially, but *regressed IEEE39 on both tasks*, flipping IEEE39-interp
from a win to a loss (0.891 → 1.341, against ODE-RNN's 1.102). Mechanistically coherent: it buys
rollout accuracy at the expense of assimilation quality, and IEEE39's wins came precisely from
the assimilation mechanism. Restricting it to extrapolation only (current config, §5) resolves
this for IEEE39 but leaves springs-interp as the one remaining loss.

**Free-running for interpolation, tried at three stretch lengths — reverted (CHANGES.md Part
25).** Hypothesis: interpolation's longest unaided stretch is the gap between two observations,
not the full forecast horizon, so sizing the free-run branch to that gap (instead of either the
full horizon above, or nothing) might flip springs-interp without re-breaking IEEE39-interp.
Tested on all three interp cells, not just springs:

| Cell | Off (current) | Full-horizon | Gap-tiled |
|---|---|---|---|
| sp-int | 0.0759 (loss) | 0.068 (win) | **0.0656 (win)** |
| ch-int | **0.2613 (win)** | — | 0.2995 (loss) |
| ie-int | **1.065 (win)** | 1.341 (loss) | 1.216 (loss) |

Gap-tiling does beat both other settings on springs-interp — so the hypothesis that rollout
*length* mattered wasn't wrong — but it costs both other cells, netting 4/6 against the 5/6 with
it off. Springs improves under any amount of unaided interpolation-time rollout tried so far;
charged and IEEE39 both get worse under any amount tried so far. That is a genuine per-dataset
disagreement about whether interpolation should train on unaided rollout at all, not a mis-sized
knob, so it was reverted rather than chased to a fourth length or tuned per dataset (which
wouldn't be defensible in a paper). **This remains the central unresolved tension.**

### 4.3 Bugs found and fixed during this work

- **CUDA OOM on IEEE39** once capacity was raised: plain `odeint` retains the full forward graph
  across a long per-timestep loop with dense `[B,N,N,H]` tensors. Fixed with `odeint_adjoint`.
- **GIL-ODE was silently ignoring IEEE39's graph entirely** — `build_S_c` hard-coded a complete
  graph for any dataset that wasn't springs, which was correct for charged but meant the
  Kron-reduced physical graph was discarded. Fixed; IEEE39 now uses its real graph.
- **NaN from the multi-horizon loss**: the shared masked-likelihood helper normalizes by
  `sum(mask)` within the sliced window with no floor, and a short prefix cutoff isn't guaranteed
  to contain an observed target for every node. Fixed with local batch-pooled helpers with a
  clamped denominator, *without* touching the shared function every other model depends on.
- **`torch.load` `weights_only` default** broke checkpoint reloading (checkpoints carry an `args`
  Namespace). Surfaced once validation-based selection started reloading the best checkpoint.

---

## 5. Current standing

The single current configuration (hidden_dim=80, mlp_width=92, 254,349 params — relation-experts +
antisymmetric channel + gate + hard anchor + hard `dq/dt=v` + fixed IEEE39 graph + fixed
extrapolation-validation split + free-running restricted to extrap-only), same loss terms and no
per-cell tuning, MSE ×10⁻² (**bold** = beats the best baseline for that cell):

| Cell | GIL-ODE | Bar | Ratio | Verdict |
|---|---|---|---|---|
| sp-int | 0.0759 | 0.0689 | 1.10x | loss |
| sp-ext | **1.591** | 2.021 | 0.79x | WIN |
| ch-int | **0.2613** | 0.2652 | 0.99x | WIN (narrow) |
| ch-ext | **5.017** | 5.403 | 0.93x | WIN |
| ie-int | **1.065** | 1.102 | 0.97x | WIN (narrow) |
| ie-ext | **5.273** | 11.284 | 0.47x | WIN |

**5 of 6 under one configuration** — up from the "two configs, 3 each, different 3" standing this
document previously described. Getting here took three changes, in the order given directly:
fix the extrapolation-validation split (§6), converge on one compact config rather than tuning per
cell (this table), and restrict free-running to extrapolation only (already true; confirmed by
testing the alternative, §4.2). The antisymmetric channel then closed springs-extrapolation
specifically (§4.1).

Three of the five wins are narrow enough (0.9–3%, single seed) that they should not yet be
reported as established wins — see §7. The three extrapolation cells have larger, more
mechanistically-grounded margins (0.79x/0.93x/0.47x) and are the more defensible claim regardless
of how the narrow cells or springs-interp resolve.

The full experiment-log table below (Parts 22–24, superseded by the row above wherever a
mismatch exists — see §2 on the validation-split discontinuity) is kept for the historical record
of how each piece was discovered:

| Revision | sp-int | sp-ext | ch-int | ch-ext | ie-int | ie-ext | Wins |
|---|---|---|---|---|---|---|---|
| Structured (relation-experts + gate + hard anchor) | 0.073 | 5.760 | **0.213** | 8.575 | **0.932** | **5.475** | 3 |
| + multi-horizon (extrap only) | — | 4.847 | — | 7.431 | — | **3.679** | — |
| + second-order, leaky `eps_q` | 0.069 | 5.003 | **0.195** | 7.576 | — | — | — |
| + second-order, hard `dq/dt=v` | 0.078 | 4.620 | **0.186** | 6.909 | **0.891** | **3.298** | 3 |
| + free-running supervision | **0.068** | 3.391 | **0.241** | 5.928 | 1.341 | **3.978** | 3 |
| + compression (h=80) | — | 2.389 | — | — | — | — | — |
| + validation fix + antisym channel (current, §5 top) | 0.0759 | **1.591** | **0.2613** | **5.017** | **1.065** | **5.273** | **5** |

Rows above the last one predate the extrapolation-validation fix, so their extrap numbers are not
directly comparable to the current table (their checkpoints were selected on a train-pool val
split that flattened almost immediately, see §6). IEEE39-extrap in particular reached 3.298 under
a wider MLP and no antisym channel — the current row's 5.273 trades some of that back for the
single-configuration, no-per-cell-tuning constraint the current table enforces.

---

## 6. Checkpoint selection — measurement issue found and fixed (CHANGES.md Part 25)

**Status: fixed.** This section originally documented an open measurement issue; it's kept below
because the diagnosis is still the right explanation for why the fix mattered, and every table
before Part 25 in this document was measured under the broken split. The fix itself: `val` is now
drawn from a disjoint slice of the *test* pool (`data/make_subset.py --val-size`), not the train
pool, applied uniformly to every model, not just GIL-ODE.

**Original diagnosis, for the record.** Validation-based checkpoint selection had been
systematically worse for GIL-ODE than for LG-ODE on extrapolation, for a structural reason:

| | Selected | Best reachable | Best-val epoch |
|---|---|---|---|
| GIL-ODE ch-ext | 5.928 | **5.147** | epoch **1** of 50 |
| Corrected LG-ODE ch-ext | 5.403 | 5.418 | epoch 44 of 50 |
| GIL-ODE sp-ext | 3.391 | 2.535 | epoch 19 of 50 |
| Corrected LG-ODE sp-ext | 2.021 | 1.575 | epoch 42 of 50 |

The extrapolation validation split is a *train-style* short-horizon task (validation trajectories
come from the train pool, which has no second forecast window), which GIL-ODE's correction
machinery nails almost immediately — so validation goes flat and stops discriminating, while
LG-ODE, which never corrects, keeps improving on it and gets properly selected.

**On charged-extrapolation GIL-ODE's true best (5.147) already beats LG-ODE's true best (5.418)
— it was losing that cell purely to selection**, which is exactly what the fix above corrected:
charged-extrap under the fixed split is now 5.017, a clean win over the 5.403 bar.

---

## 7. Open items

1. **Multi-seed confirmation (3 seeds), all six cells — top priority.** Every number in this
   document is one seed. Three of the current five wins are narrow (charged-interp 0.99x,
   IEEE39-interp 0.97x, and even springs-extrap's margin hasn't been seed-checked) and this sweep
   already demonstrated cells flipping by more than these margins from a single design change
   (§4.2's gap-tiled experiment) — so none of the current wins should be reported as established
   until error bars exist.
2. **Springs-interpolation** is the one remaining loss (1.10x). Free-running at three different
   stretch lengths was tried and reverted (§4.2) — it trades against charged-interp and
   IEEE39-interp every time, not just at one setting, so the next idea needs to be something other
   than tuning this knob further. Not yet tried: a mechanism specific to what makes springs
   different (see item 3), rather than another global training-curriculum change.
3. **Springs' collision events**, still uninvestigated per the given sequencing ("investigate
   collision events only after" the antisymmetric channel, which is now done). The springs
   simulation includes wall collisions — discontinuous events a smooth vector field can't
   represent — occurring several times within a 40-step forecast, and possibly within the shorter
   interpolation window too. Plausible contributor to both remaining springs weaknesses
   (interp's 1.10x gap and extrap's still-narrow 0.79x margin, even though extrap now wins).
4. **IEEE39-extrap's regression under the compact single config** (3.298 with a wider MLP and no
   antisym channel, vs. 5.273 now) is worth understanding even though the cell still wins by 2x —
   it's the clearest evidence the compact-state trade has a real cost somewhere, and IEEE39-extrap
   is the project's largest, most robust margin, so it's the cell where giving anything back is
   most worth noticing.

---

## 8. Note on framing

A single configuration now wins 5 of 6 (§5), closer to a universal-win story than this document
previously supported, but two things still keep a clean "6/6" claim off the table: springs-interp
remains a loss, and three of the five wins are narrow enough that they need multi-seed
confirmation before being reported as wins at all (§7, item 1). Until that confirmation exists,
the defensible claim is narrower but still strong: **GIL-ODE wins all three extrapolation cells**,
with real margins (0.79x/0.93x/0.47x) and a coherent mechanism — correction-free rollout is
exactly what the antisymmetric channel and free-running supervision target, and extrapolation is
exactly the task that requires running uncorrected. **IEEE39 extrapolation specifically** remains
the standout, at roughly 2x, in the real power-grid setting with genuine irregular partial
observation that the architecture was designed for. That contrast — decisive wins where the model
must run uncorrected, competitive-to-behind on assimilation-heavy interpolation for the one
dataset with collision discontinuities — is a sharper, more mechanistically grounded argument for
the architecture's necessity than an unexplained uniform win would be, and unlike a forced 6/6 it
does not require claiming results the current single-seed evidence doesn't yet support.
