# Data readiness

Audit date: checked directly against the files on disk in this repo (not assumed from
documentation). See `reports/DATA_CHARACTERISTICS.md` for the full writeup per dataset.

| Dataset | Location | Trajectories | Nodes | Features | Time points | Graph available | Status |
|---|---|---|---|---|---|---|---|
| Springs | `data/spring/` | 20,000 train / 5,000 test | 5 particles | 4 (x, y, vx, vy) | 40-52 irregular samples/object (train); +40 extrapolation samples (test); drawn from a 60-point regular grid | Yes — `[5,5]`, binary (0/1), independently random per trajectory | **READY** |
| Charged particles | `data/charged/` | 20,000 train / 5,000 test | 5 particles | 4 (x, y, vx, vy) | Same as springs (40-52 irregular, +40 extrap, 60-point grid) | Yes — `[5,5]`, signed (−1/+1), dense (no zero entries), independently random per trajectory | **READY**, with a caveat (see below) |
| IEEE 39-bus (raw) | `data/raw/ieee39_tsa/tsa_data.pkl` | 12,852, no pre-defined split | 10 generators | 5 (P, ut, ie, xspeed, firel) | 60 regular points @ 0.01s, fully dense (no missing observations) | **No** — no admittance/topology data anywhere in the repo; ANDES path (`data/raw/andes_ieee39/`) does not exist, `andes` package not installed | BLOCKED (raw) |
| IEEE39-Gen (processed) | `data/processed/ieee39_gen/ieee39_gen.npz` | 12,852 (10,282 train / 1,285 val / 1,285 test) | 10 generators | 5 (P, ut, ie, xspeed, firel) | 60 regular points @ 0.01s (rebased to start at 0); interp masks at 40/60/80%; extrap masks (context=first 30, forecast=last 30) at 40/60/80% of context | Yes — `[10,10]` **complete graph** (interaction-support assumption, not the physical network — see `reports/ieee39_gen.md`) | **READY** |

## IEEE39-Gen: what was built, and its one real caveat

`data/prepare_ieee39_gen.py` converts the raw trajectories (verdict **A**, genuine dynamics —
see characteristics doc) into an LG-ODE-style forecasting dataset: correctly un-flattening the
feature-major `[60,50]` columns into `[60,10,5]` (verified against the raw DataFrames, not just
assumed), rebasing time per trajectory, building a stratified 80/10/10 split (by stability
label x clearing-time group, seed 1991), computing train-only normalization statistics, and
generating fixed interpolation and extrapolation observation masks at 40/60/80%. All of this is
now **READY** — see `reports/ieee39_gen.md` for the full field-by-field description and
`data/prepare_ieee39_gen.py`'s own checks (shapes, finite values, monotonic times, disjoint
splits, zero graph diagonal, no extrapolation leakage) which all pass.

The one thing to keep in view: the `[10,10]` graph is a **complete graph** (every generator
pair connected), standing in for the still-missing physical topology — it is an
interaction-support assumption, not a claim about the real IEEE 39-bus network's structure.
None of the underlying blockers from the raw-data audit are actually resolved by this
conversion: there is still no admittance/topology data anywhere in the repo, no ANDES model
present, and per-sample fault-location/loading metadata is still unrecoverable (only the binary
stability label and, indirectly, clearing time survive). If a real generator-level coupling
matrix becomes available later, it should replace `adjacency` in `ieee39_gen.npz`, not be
treated as a separate, better version of it.

## Charged particles caveat

Data is complete and loads correctly, but there's a real quirk in how the *existing, shared*
graph-construction code (`lib/new_dataLoader.py` / `lib/at_new_dataLoader.py`, used unmodified
by both LG-ODE and AT-LG-ODE) handles it: verified directly (see characteristics doc) that
`edge_same` is **always 0** for charged particles — the temporal encoder never constructs
same-object (temporal-continuity) edges for this dataset, only cross-object ones, versus ~27%
same-object edges for springs. This is a data/code-interaction issue, not a data-quality issue
(the data itself has everything needed), so it doesn't block readiness, but it does mean
charged-particle encoder behavior isn't quite what it looks like from the springs case, and
should be kept in mind when comparing results across the two datasets or extending the pipeline
to IEEE 39.
