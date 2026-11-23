# CSG-ODE Reconstruction Report

## Status

The isolated implementation and all four data adapters are ready for baseline training sweeps. Springs, Charged, and Motion-walk can run now; PEMS08 paper-comparison training is deliberately blocked until its incorrect local feature source is replaced. Equation-level tests, checkpoint round trips, and full-architecture CPU forward/backward smoke checks pass for all four dataset shapes and both tasks. This report intentionally does **not** call the numerical results reproduced: the required 50-epoch, three-seed experiment matrix has not been run yet.

Primary sources:

- [CSG-ODE paper and appendix](https://proceedings.mlr.press/v267/wang25dd.html)
- [Official ControlSynth Neural ODE implementation](https://github.com/ContinuumCoder/ControlSynth-Neural-ODE), inspected at `ca08733dd6bfbb38081e9fb17c7134026f89a861`
- [Canonical AGCRN implementation](https://github.com/LeiBAI/AGCRN), inspected at `7fbbf2aeb099242098a3cf482b55cd45d7295c28`
- [AGCRN paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/ce1aad92b939420fc17005e5461e6f48-Paper.pdf)

## Repository Audit

The pre-existing prototype recomputed communicability in float32 during every encoder forward, used a shared `GRUCell` rather than observation-derived node kernels, fed the encoder's adaptive graph to the ODE instead of the physical graph, ignored `augment_dim`, and optimized MSE plus KL rather than the LG-ODE Gaussian ELBO. Its loader only handled Springs, generated order-dependent masks, and selected checkpoints on test loss. Those files remain untouched because they contain local work and may still be useful as a historical prototype.

The new implementation is isolated under `lib/baselines/csg_ode/` and entered through `run_csg_ode.py`; LG-ODE and AT-ODE behavior is not changed.

Implementation inventory:

- `lib/baselines/csg_ode/`: equations, data adapters, training, diagnostics, metrics, checkpointing, and deterministic masks;
- `configs/csg_ode/`: four paper configurations and the checksum-bound PEMS08 feature manifest example;
- `run_csg_ode.py`, `reproduce_csg_ode.py`, and `audit_csg_data.py`: single-run, matrix, aggregation, and audit entry points;
- `tests/test_csg_*.py`: equation, loader, integration, checkpoint, overfit, and sweep-artifact tests;
- `README.md`, `REPRODUCTION.md`, and `reports/`: commands, audit decisions, smoke evidence, and generated data statistics.

## Architecture and Tensor Flow

For a batch with `B` samples, `T` union timestamps, `N` nodes, feature size `F`, embedding size `k`, node-parameter size `q`, hidden size `h=16`, and latent size `d2=16`:

```text
observed_values [B,T,N,F] + masks + normalized times
  -> observation MLP -> E [B,T,N,k]
  -> observed-only temporal mean -> Emean [B,N,k]
  -> phi(Emean) -> sample-dependent Q [B,N,q]

Es,Et [N,F] -> row-softmax(relu(Es Et^T)) -> G [N,N]
Go [B,N,N] -> cached float64 finite difference -> D [B,N,N]
Gmix = G + W1 * D
R [B,T,N] + pair mask -> streamed G_t [B,N,N]

(E_t, H_{t-1}, G_t, Q) -> node-adaptive graph GRU -> H [B,N,h]
H -> (mu,std) [B,N,d2] -> sampled z0
z0 + zero augmentation -> z [B,N,d2+augment]
c0 [B,N,d2+augment] = 0 by default

Euler solve of joint [z,c]:
  z_dot = A0 z + sum_j A_j f_j(MLP_j(z)) + g(c)
  c_dot = relation-aware GNN(z, original physical graph)

first d2 latent channels -> node-wise LG-ODE linear decoder -> predictions [S,B,T_target,N,F]
```

The recurrent encoder streams one graph snapshot at a time. It does not allocate `[B,T,N,N]`, which is important for PEMS08.

## Equation Traceability

| Paper component | Implementation |
|---|---|
| Eq. 1 adaptive adjacency | `adaptive_gcrnn.StaticAdaptiveAdjacency` |
| Eq. 2-3 communicability | `communicability.finite_difference_frechet` and `CommunicabilityCache` |
| Eq. 4 mixed graph | `encoder.CSGEncoder.forward` |
| Appendix-A sampling interval | `density_adjustment.sampling_intervals` |
| Eq. 5-6 density snapshot | `density_adjustment.DensityAdjustedGraph` |
| Eq. 8 observation embedding | `encoder.CSGEncoder.observation_embedding` |
| Eq. 9-10 Emean and Q | `encoder.CSGEncoder._mean_embedding` and `node_representation` |
| Eq. 11 adaptive graph convolution | `adaptive_gcrnn.NodeAdaptiveGraphConv` |
| Eq. 12-13 Gaussian z0 | `encoder.CSGEncoder.posterior` |
| Eq. 14 coupled ODE | `csode_func.ControlSynthGraphODEFunc` |
| Physical relation GNN | `csode_func.RelationAwareControlGNN` |
| Euler/dopri5 integration | `solver.solve_joint_ode` |
| Gaussian ELBO and MSE | `metrics.py` and `model.CSGODE.compute_all_losses` |

## Dataset Audit

The generated detailed audit is in `reports/dataset_audit.json`.

| Dataset | Train/Test | Nodes | Features used | Stored observations | Verified graph |
|---|---:|---:|---:|---:|---|
| Springs | 20,000 / 5,000 | 5 | 4: 2 position + 2 velocity | train 40-52; test 80-91 | symmetric binary, 0-20 entries, variable, zero diagonal |
| Charged | 20,000 / 5,000 | 5 | 4: 2 position + 2 velocity | train 40-52; test 80-91 | dense signed 5x5, 25 active entries, +1 diagonal |
| Motion-walk | 16 / 7 | 29 | 6 stored pose channels | train 30-42; test 70-82 | fixed symmetric binary graph, 56 entries, zero diagonal |
| PEMS08 | 199 / 49 | 170 | 3 stored, but semantics mismatch paper | train 40-52; test 80-92 | fixed symmetric binary graph, 548 entries, zero diagonal |

Important findings:

- Motion now matches the requested 29-joint, 16/7 split. Its `loc` arrays already contain all six paper features; concatenating the derivative `vel` file would incorrectly create 12 features.
- Springs uses binary adjacency for communicability and no-spring/spring relation labels over all 20 off-diagonal NRI candidate pairs in the control GNN.
- Charged uses `-1` and `+1` as two active relation labels and stores five positive self entries. The adapter preserves all 25 physical entries.
- The local PEMS08 adjacency is symmetric. The adapter preserves that source direction convention instead of inventing a directed graph.
- The local PEMS08 raw channels are exactly `[traffic flow, time-of-day, day-of-week]`; channels 1 and 2 are deterministic calendar covariates. The paper requires `[traffic flow, speed, occupancy]`. Primary PEMS08 runs fail closed until corrected data with a checksum-bound `feature_semantics.json` sidecar is supplied. `--allow-dataset-mismatch` creates only a labeled variant result that the paper aggregator ignores.
- Springs includes zero-edge sequences although the paper table states 1-20 directed entries. This discrepancy is preserved and logged.
- PEMS08 processed timestamps are local to each window. The preprocessing source verifies non-overlapping global slices (`raw[:11940]` and the following 5,880 points), but global indices cannot be reconstructed from the processed arrays alone.

## Task and Metric Protocol

The source files are already independently irregularly sampled with the paper's `U(40,52)` or `U(30,42)` process. The configured 0.4/0.6/0.8 ratio is therefore applied once as the verified LG-ODE second-stage subsampling layer. As in LG-ODE, the initial context observation for each node is dropped before second-stage masking. Masks use the public `masking.deterministic_observation_indices` helper and are deterministically keyed by global seed, logical split, original sample index, node index, and ratio, independent of loader/shuffle order.

Interpolation targets all stored context points after the initial-point convention. Extrapolation training conditions on points before the midpoint and targets points at or after it; test conditions on the context segment and targets the stored 40 future observations. Context and target task intervals are normalized independently to `[0,1]`, and no future values enter Emean, Q, graph snapshots, or posterior inference.

The processed irregular files do not retain an unobserved dense ground-truth trajectory. Consequently, “all target points” means all target points retained in these LG-ODE-compatible files, not every original simulator step. This limitation must remain attached to comparisons with paper numbers.

Logged metrics are:

- `mse_lgode`: the existing LG-ODE convention, normalized over time per node/feature and then averaged;
- `mse_all`: element-weighted MSE on all task target entries;
- `mse_unobserved`: interpolation entries hidden from the encoder, or all future entries for extrapolation;
- `mse_table_scaled = 100 * mse_lgode` because the paper tables use MSE x 10^-2;
- Gaussian reconstruction likelihood with fixed standard deviation `0.01` and z0 KL separately.

## Paper Hyperparameters

All primary YAMLs use Adam, learning rate `5e-4`, weight decay `1e-3`, dropout `0.2`, gradient clipping `10`, 50 epochs, `h=16`, `d2=16`, two nonlinear subnetworks, depth one, and Euler. Dataset batch sizes are Springs `256`, Charged `256`, Motion `8`, and PEMS08 `4`. PEMS08 uses `k=16`, `q=8`, width `64`, and no augmentation; the other datasets use `k=64`, `q=32`, width `128`, and augmentation `64`.

The paper's per-epoch LG-ODE KL schedule is retained by default: coefficient zero for the first 10 batches, then `1 - 0.99^(batch_index-10)`. This legacy schedule restarts each epoch and is recorded in checkpoints. `global` and `constant` are explicit alternatives, not primary defaults.

## Ambiguity Ledger

1. The paper omits complete GCRNN gate equations. The implementation uses canonical AGCRN-style reset/update gates with separate node-adaptive gate and candidate pools.
2. Equation 5 does not identify `sigma`. Default is sigmoid; tanh and identity are selectable.
3. Equation 1 uses row-wise softmax (`dim=-1`).
4. W1/W2 default to zero; Xavier is selectable for the bounded validation comparison.
5. Unobserved nodes carry their prior hidden state by default; normal GRU updates are selectable.
6. `c(0)=0` is primary; `GNN(z0)` is selectable.
7. Following the official ControlSynth convention, depth one is `Linear(d,width) -> activation -> Linear(width,d)`; the equation's common tanh is applied to that output.
8. Zero augmentation is appended to z0. The control state has the same augmented size, and only the original 16 latent channels are decoded.
9. The decoder and fixed-variance Gaussian likelihood match LG-ODE; the legacy KL schedule is documented above.
10. Primary comparison uses `mse_lgode`; explicit all-target and unobserved-only metrics are also saved.
11. Primary normalization is split-wise max-absolute; train-statistics max-absolute is selectable and must be reported separately.
12. Charged retains 25 active signed entries including self edges rather than forcing 20 off-diagonal edges.
13. The current Motion files match 29 nodes, so no 31-node fallback is used.
14. PEMS08 paper inconsistencies are retained: interpolation 60% is 0.2827 in the primary table versus 0.2602 in the ablation table; extrapolation 40% is 1.6607 versus 1.7726.

Every checkpoint stores this ledger, the full resolved configuration, dataset audit, optimizer, RNG states, and validation selection metadata.

## Ablations

The run CLI exposes:

- `--ablation-aq`: learned free Q instead of observation-derived Q;
- `--ablation-no-ei`: remove D from Gmix;
- `--ablation-no-g`: set g(c) to zero;
- `--ablation-no-ni`: remove the nonlinear subnetwork sum.

`reproduce_csg_ode.py --variants full,aq,no-ei,no-g,no-ni` creates the ablation matrix. LG-CSODE and SCSG-ODE are intentionally deferred until ordinary CSG-ODE has numerical validation, as requested.

## Validation Completed

Complete test invocation with overfit checks enabled:

```text
51 passed, 1 skipped in 29.03s
```

The only skip is the CUDA/AMP smoke test because CUDA was unavailable. All four one-batch overfit checks pass. Full paper-size forward/backward smoke checks passed on CPU for all datasets in both interpolation and extrapolation modes. Output shapes were:

A two-stage CLI resume smoke test also restored model, optimizer, global RNG, and train-loader RNG state, preserved epoch history, and selected the earlier validation-best checkpoint when the resumed epoch did not improve.

The existing AT-ODE regression suite also remains green (`23 passed, 1 skipped`). LG-ODE contains no test files in this checkout; no LG-ODE or AT-ODE source file was changed by this reconstruction.

| Dataset | Interpolation | Extrapolation |
|---|---|---|
| Springs | `[1,1,59,5,4]` | `[1,1,30,5,4]` |
| Charged | `[1,1,59,5,4]` | `[1,1,30,5,4]` |
| Motion-walk | `[1,1,49,29,6]` | `[1,1,25,29,6]` |
| PEMS08 | `[1,1,59,170,3]` | `[1,1,30,170,3]` |

A one-batch PEMS08 interpolation plumbing profile with batch size 4 and three posterior samples on this CPU measured approximately. It used the incompatible local covariate variant and is not paper performance evidence:

| Stage | Seconds |
|---|---:|
| Data | 0.043 |
| Encoder | 0.166 |
| ODE | 0.508 |
| Decoder | 0.002 |
| Backward + optimizer | 0.806 |

CUDA peak memory was not measured because this environment has no CUDA device. Apple MPS selection is implemented through `--device auto` but was also unavailable in this execution environment.

## Reproduction Commands

Audit and communicability validation:

```bash
python audit_csg_data.py --data-root ../data
python run_csg_ode.py \
  --dataset springs --task interp --obs-ratio 0.6 \
  --config configs/csg_ode/springs.yaml \
  --audit-only --validate-communicability
```

Single primary run with W&B:

```bash
export WANDB_MODE=online
export WANDB_PROJECT=at-ode

python run_csg_ode.py \
  --dataset springs --task extrap --obs-ratio 0.4 \
  --config configs/csg_ode/springs.yaml \
  --seed 1991 --device auto \
  --wandb-mode online --wandb-project at-ode
```

The matrix driver performs dataset preflight before launching its first subprocess. With the current PEMS08 source, use `--datasets springs,charged,motion_walk`; after installing and checksum-labeling the corrected PEMS08 source, the four-dataset command above becomes available.

Complete primary three-seed matrix:

```bash
python reproduce_csg_ode.py \
  --datasets springs,charged,motion_walk,pems08 \
  --tasks interp,extrap \
  --obs-ratios 0.4,0.6,0.8 \
  --seeds 1991,42,7 \
  --variants full \
  --normalization-mode paper_splitwise_maxabs \
  --device auto \
  --wandb-mode online --wandb-project at-ode \
  --execute
```

Run `train_maxabs` as a separate matrix rather than mixing normalization modes.

## Published Targets and Pending Results

The sweep aggregator writes raw and x100 MSE, mean, sample standard deviation, best seed, runtime, paper target, and relative error. It refuses to aggregate limited-batch runs. Until all three seeds finish, the reproduction columns remain incomplete rather than being inferred from smoke MSE.

| Dataset | Task | Paper x100 at 0.4 / 0.6 / 0.8 | Reproduction |
|---|---|---|---|
| Springs | interp | 0.1550 / 0.1440 / 0.1386 | pending |
| Springs | extrap | 1.3495 / 1.2969 / 1.2691 | pending |
| Charged | interp | 0.7947 / 0.7169 / 0.7099 | pending |
| Charged | extrap | 5.5086 / 4.7690 / 4.4966 | pending |
| Motion-walk | interp | 0.0439 / 0.0406 / 0.0400 | pending |
| Motion-walk | extrap | 0.1791 / 0.1539 / 0.1593 | pending |
| PEMS08 | interp | 0.2526 / 0.2827 / 0.3360 | blocked on correct feature data |
| PEMS08 | extrap | 1.6607 / 1.7436 / 1.4937 | blocked on correct feature data |

## Remaining Work Before Claiming Reproduction

- Run the complete 50-epoch, three-seed matrix and the bounded ambiguity validation sweep.
- Replace the local PEMS08 calendar-covariate file with verified flow/speed/occupancy data before running or comparing PEMS08 primary results.
- Capture CUDA/MPS runtime and peak memory on the actual sweep hardware.
- Re-run LG-ODE and AT-ODE through the same deterministic mask helper, or record a mask manifest, before claiming a paired comparison. Their current legacy mask generation is seeded but does not use this adapter's per-sample hash.
- Compare trends for AQ, no-EI, no-g, and no-NI before considering LG-CSODE or SCSG-ODE.
