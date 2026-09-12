# IEEE39-Gen dataset

Built by `data/prepare_ieee39_gen.py` from `data/raw/ieee39_tsa/tsa_data.pkl`. Output:
`data/processed/ieee39_gen/ieee39_gen.npz`.

## The graph is an assumption, not the physical network

`adjacency` is a **complete graph** on the 10 generators (ones off-diagonal, zero diagonal) —
every generator is allowed to message-pass with every other generator. This is an
**interaction-support assumption**, standing in for the fact that no physical IEEE 39-bus
topology or admittance data is available anywhere in this repo (see
`reports/DATA_CHARACTERISTICS.md` / `reports/DATA_READY.md`: no `andes_ieee39/` model, `andes`
not installed). **It is not the physical transmission-line graph** — the real IEEE 39-bus
network does not have 10 generators all directly wired to each other, and this adjacency
should never be described or cited as if it were derived from the network's actual topology.
If a real generator-level coupling/admittance matrix becomes available later, it should
replace this file's `adjacency`, not extend it.

## Contents of `ieee39_gen.npz`

| Key | Shape | Notes |
|---|---|---|
| `states` | `[12852, 60, 10, 5]` | `[trajectory, time, generator, feature]`, float32, raw units (not normalized) |
| `times` | `[12852, 60]` | seconds, rebased per trajectory to start at 0.0 |
| `adjacency` | `[10, 10]` | complete graph, see above |
| `labels` | `[12852]` | 1 = stable, 0 = unstable (source semantics preserved) |
| `train_idx` / `val_idx` / `test_idx` | variable | whole-trajectory indices, 80/10/10, seed 1991, stratified by (label, clearing-time group), disjoint and covering all 12,852 |
| `feature_mean` / `feature_std` | `[1,1,1,5]` | computed from `states[train_idx]` only |
| `interp_mask_{40,60,80}` | `[12852, 60, 10]` | per (trajectory, time, generator) observed indicator; independent per generator; full trajectory remains the reconstruction target |
| `extrap_mask_{40,60,80}` | `[12852, 60, 10]` | same shape; only times `[0,30)` (context) may be observed, times `[30,60)` (forecast) are always 0 |
| `generator_ids` | `[10]` | `["G01", ..., "G10"]`, consistent ordering with `states`'/`adjacency`'s generator axis |
| `feature_names` | `[5]` | `["P", "ut", "ie", "xspeed", "firel"]` |
| `clearing_time_group` | `[12852]` | which of the 6 distinct clearing times each trajectory belongs to (recovered from the raw absolute timestamps, not stored explicitly in the source data) |

## Known, inherited limitations (see `reports/DATA_CHARACTERISTICS.md` for detail)

- Per-sample fault location and loading-condition percentage are **not** recoverable — only
  the binary stability label and (via `clearing_time_group`) clearing time survive from the
  original generation parameters.
- `G02`'s `firel` (rotor angle) is identically 0.0 in every trajectory (it's the reference
  machine) — real per-generator angle variance exists across the other 9, so training
  normalization statistics are still meaningful, but `G02`'s own angle carries no signal.
- Only the post-fault-clearance window (60 points, 0.01s apart, ~0.6s) is available; pre-fault
  and during-fault dynamics are not part of this data.
