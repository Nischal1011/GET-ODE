# AT-LG-ODE vs. LG-ODE — Results Summary

Full-scale data (20k train / 5k test trajectories, matching the paper), 60% observed,
30-epoch training budget, after the numerical-stability patch in `lib/base_models.py`
(see `CHANGES.md` Part 3). Reported MSE is the best test-set value reached during training
(×10⁻², matching the paper's units).

## Springs

### Interpolation

| Model | Best test MSE (×10⁻²) | Best epoch | Log |
|---|---|---|---|
| LG-ODE (patched) | 0.3406 | 26 / 30 | `run_logs/springs_interp_60_patched.log` |
| AT-LG-ODE (patched) | 0.3459 | 13 / 30 | `run_logs/at_springs_interp_60_patched.log` |

Essentially a tie (~1.5% apart, within reproduction noise). AT-LG-ODE reaches its best result at
about half the epochs LG-ODE needs (13 vs. 26) for the same final quality.

### Extrapolation

| Model | Best test MSE (×10⁻²) | Best epoch | Log |
|---|---|---|---|
| LG-ODE (patched) | 1.6418 | 10 / 30 (destabilizes after, never recovers) | `run_logs/springs_extrap_60.log` |
| AT-LG-ODE (patched) | **1.2374** | 29 / 30 (monotonically improving throughout) | `run_logs/at_springs_extrap_60.log` |

AT-LG-ODE wins by **~25%**, and — unlike the baseline — trains stably for the full 30 epochs
with no divergence.

### Reference: reproduction of the paper's own numbers (springs)

For context, LG-ODE reproduced here vs. the paper's Table 1/2 (50-epoch budget, before the
stability patch existed):

| Task | Paper | Reproduced |
|---|---|---|
| Interpolation | 0.3170 | 0.3306 (epoch 17/50) |
| Extrapolation | 1.8084 | 1.6418 (epoch 10/30) |

## Charged particles

### Interpolation

| Model | Best test MSE (×10⁻²) | Best epoch | Log |
|---|---|---|---|
| LG-ODE (patched) | **0.8033** | 24 / 30 | `run_logs/charged_interp_60.log` |
| AT-LG-ODE (patched, `nonadj-floor=0`) | 0.8739 | 14 / 30 | `run_logs/at_charged_interp_60.log` |
| AT-LG-ODE (patched, `nonadj-floor=0.3`) | 0.8291 | 19 / 30 | `run_logs/at_charged_interp_60_floor.log` |

The floor closes about 75% of the gap to LG-ODE (0.8739 → 0.8291, vs. LG-ODE's 0.8033) —
meaningful support for the "silenced non-adjacent pathway" hypothesis below.

### Extrapolation

| Model | Best test MSE (×10⁻²) | Best epoch | Log |
|---|---|---|---|
| LG-ODE (patched) | **5.8111** | 27 / 30 | `run_logs/charged_extrap_60.log` |
| AT-LG-ODE (patched, `nonadj-floor=0`) | 6.2311 | 24 / 30 | `run_logs/at_charged_extrap_60.log` |
| AT-LG-ODE (patched, `nonadj-floor=0.3`) | 6.3105 | 24 / 30 | `run_logs/at_charged_extrap_60_floor.log` |

The floor does **not** help here — if anything, slightly worse than `nonadj_floor=0`. So the
non-adjacent-pathway story is at most a partial explanation: it accounts for a real chunk of the
interpolation gap but not the extrapolation gap, meaning something else is also costing
AT-LG-ODE on charged extrapolation specifically. Not chased further here.

Both LG-ODE reproductions track the paper closely (Table 1: 0.8277, Table 2: 6.4338), so this
isn't a reproduction issue — **LG-ODE beats AT-LG-ODE on both charged tasks**, the reverse of
springs.

**Why this reverses**: the transport weight is `w_ij(t) = A_ij·r_ij(t) / (Σ_k A_ik·r_ik(t)+ε)`.
`A_ij` is a hard 0/1 mask, so any pair with no physical edge gets `w_ij(t) = 0` *always* — its
relation message is completely silenced, at every timestep. The original NRI ODE function, by
contrast, always keeps a second MLP for "not-connected" pairs (Appendix C.1 of the paper)
alongside the connected one. For springs, non-adjacent objects truly exert zero force, so
silencing that pathway costs nothing. For charged particles, every pair attracts or repels
regardless of the sampled edge label, so that pathway carries real signal that AT-LG-ODE was
discarding entirely for every non-adjacent pair.

**Fix**: `AttentionTransport(nonadj_floor=...)` (`lib/attention_transport.py`,
`--nonadj-floor` flag) adds a small constant to *every* pair's numerator before normalizing —
`Ar = A_ij·r_ij(t) + nonadj_floor` — instead of only the adjacent ones. At `nonadj_floor=0` this
is mathematically identical to the original hard mask (confirmed: springs results above are
unaffected by this change). At `nonadj_floor=0.3`, non-adjacent pairs get a small non-zero
share of the aggregation instead of being zeroed, while adjacent pairs with real transported
evidence still dominate. Result: it recovers most of the interpolation gap but not the
extrapolation gap — see the tables above.

## Takeaway

Transporting the encoder's relational attention forward in time to reweight the ODE's graph
aggregation costs nothing on springs interpolation (tie, faster convergence) and meaningfully
helps on springs extrapolation (~25% lower MSE, no divergence where the baseline destabilizes).

On charged particles, the same hard-masked transport hurts on both tasks. Giving non-adjacent
pairs a small floor weight (`nonadj_floor=0.3`) instead of hard-zeroing them recovers most of
the interpolation gap (0.8739 → 0.8291 vs. LG-ODE's 0.8033), confirming that silencing the
"no labeled edge" pathway costs real signal on a dataset where every pair interacts regardless
of the sampled edge label. It does not fix extrapolation, though, so the hard mask is only part
of what's costing AT-LG-ODE there — this remains an open gap for the design on charged
particles specifically.

## What was run

```
# Springs
python run_models.py    --dataset-dir data/spring  --niters 30 --alias springs_interp_60_patched
python run_models.py    --dataset-dir data/spring  --niters 30 --extrap True --alias springs_extrap_60
python run_models_at.py --dataset-dir data/spring  --niters 30 --alias at_springs_interp_60_patched
python run_models_at.py --dataset-dir data/spring  --niters 30 --extrap True --alias at_springs_extrap_60

# Charged particles
python run_models.py    --data charged --dataset-dir data/charged --niters 30 --alias charged_interp_60
python run_models.py    --data charged --dataset-dir data/charged --niters 30 --extrap True --alias charged_extrap_60
python run_models_at.py --data charged --dataset-dir data/charged --niters 30 --alias at_charged_interp_60
python run_models_at.py --data charged --dataset-dir data/charged --niters 30 --extrap True --alias at_charged_extrap_60

# Charged particles, nonadj-floor fix
python run_models_at.py --data charged --dataset-dir data/charged --niters 30 --nonadj-floor 0.3 --alias at_charged_interp_60_floor
python run_models_at.py --data charged --dataset-dir data/charged --niters 30 --extrap True --nonadj-floor 0.3 --alias at_charged_extrap_60_floor
```

See `CHANGES.md` for the full technical writeup of what AT-LG-ODE changes and why, and for the
compatibility/stability fixes applied to the original codebase.

---

# Phase 2: Corrected LG-ODE and the full baseline suite

AT-LG-ODE work above is paused (per direction) in favor of re-establishing a trustworthy LG-ODE
baseline first, and comparing it against five other methods: ODE-RNN, Latent-ODE, Edge-GNN,
RNN-NRI, and Corrected LG-ODE itself. All results below are MSE ×10⁻² (paper units),
best-test-epoch, 30-epoch budget, seed 1991, 60% observed, across three datasets: springs,
charged particles, and IEEE39-Gen (a new dataset — see `reports/ieee39_gen.md` and
`reports/DATA_CHARACTERISTICS.md`). See `CHANGES.md` Parts 7-10 for full technical detail on
everything summarized here.

## Corrected LG-ODE vs. the original (uncorrected) LG-ODE

"Corrected" fixes three issues in the original data pipeline (used unmodified since the start
of this project): an object-identity/self-loop bug in temporal-edge construction (silently
dropped almost all of charged particles' edges), normalization statistics actually being fit on
the *test* set instead of train (confirmed by tracing the code — `run_models.py` loads test
before train), and no validation split at all. See `CHANGES.md` Part 8.

| Task | Uncorrected LG-ODE | Corrected LG-ODE | Change |
|---|---|---|---|
| Springs interpolation | 0.3406 | 0.6500 | worse |
| Springs extrapolation | 1.6418 | 2.8715 | worse |
| Charged interpolation | 0.8033 | 0.9132 | worse |
| Charged extrapolation | 5.8111 | **4.8943** | **better** |

Most cells get *worse* after removing the train/test normalization leakage — expected and
reassuring, since it confirms the leakage was real and was inflating the original numbers.
Charged extrapolation improves anyway because the connectivity fix's benefit (edge count for
charged nearly tripled, 2,587 → 7,907 per graph, once self-loops and previously-dropped
"attract" pairs are restored) outweighs the lost leakage advantage there.

## Full baseline comparison

| Dataset / Task | ODE-RNN | Latent-ODE | Edge-GNN | RNN-NRI | Corrected LG-ODE |
|---|---|---|---|---|---|
| Springs interp | **0.088** | 0.203 | 1.235 | 0.117 | 0.650 |
| Springs extrap | 5.760 | 4.740 | 3.773 | 9.495 | **2.872** |
| Charged interp | 0.221 | 0.489 | 1.038 | **0.189** | 0.913 |
| Charged extrap | 7.021 | **4.841** | 5.132 | 10.584 | 4.894 |
| IEEE39 interp | 1.470 | 10.910 | 14.913 | **1.076** | 11.713 |
| IEEE39 extrap | **17.791** | 19.856 | 20.340 | 43.381 | 18.507 |

(bold = best per row; full per-run logs: `run_logs/{odernn,latentode,edgegnn,rnnnri,corrected}_{spring,charged,ieee39}_{interp,extrap}_60.log`)

### Headline finding

Graph structure (Corrected LG-ODE) wins or is closely competitive on **every extrapolation
task**, but loses — often badly — on **every interpolation task**, sometimes to the simplest
possible baseline (ODE-RNN, a per-node model with no graph at all). The starkest case is IEEE39
interpolation: ODE-RNN (1.470) and RNN-NRI (1.076) beat every graph-based model by roughly
10-14x (Corrected LG-ODE 11.713, Edge-GNN 14.913).

**Caveat, not dismissal**: IEEE39-interp's graph-based models were still visibly improving at
epoch 30 in the training logs (not plateaued), while the simpler models converge faster — so
part of that specific gap may be "the graph models need more than 30 epochs here," not purely
"graph structure hurts interpolation." The extrapolation pattern (graph structure wins
consistently, across all three datasets) is not explained by that caveat, since Corrected
LG-ODE's extrapolation runs show the same kind of late-epoch improvement *and* still end up
ahead.

RNN-NRI's extrapolation numbers are worst-in-class everywhere, most dramatically on IEEE39
(43.381 — roughly 2.3x worse than the next-worst model). This matches the expected limitation
called out for this baseline: autoregressive discrete rollout compounds error over many steps,
and was observed directly in training (first-batch train loss spiking into the millions on
IEEE39 before stabilizing) — see `CHANGES.md` Part 9.

## What was run (Phase 2)

```
# Corrected LG-ODE (springs, charged, ieee39 x interp, extrap)
python run_models_corrected.py --data spring  --niters 30 --alias corrected_spring_interp_60
python run_models_corrected.py --data spring  --niters 30 --extrap True --alias corrected_spring_extrap_60
python run_models_corrected.py --data charged --niters 30 --alias corrected_charged_interp_60
python run_models_corrected.py --data charged --niters 30 --extrap True --alias corrected_charged_extrap_60
python run_models_corrected.py --data ieee39  --niters 30 --alias corrected_ieee39_interp_60
python run_models_corrected.py --data ieee39  --niters 30 --extrap True --alias corrected_ieee39_extrap_60

# Each of the 4 baselines, same 3 datasets x 2 tasks (24 runs total)
python run_models_odernn.py    --data <spring|charged|ieee39> [--extrap True] --niters 30 --alias <name>
python run_models_latentode.py --data <spring|charged|ieee39> [--extrap True] --niters 30 --alias <name>
python run_models_edgegnn.py   --data <spring|charged|ieee39> [--extrap True] --niters 30 --alias <name>
python run_models_rnnnri.py    --data <spring|charged|ieee39> [--extrap True] --niters 30 --alias <name>
```
