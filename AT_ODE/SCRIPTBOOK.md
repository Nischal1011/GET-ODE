# AT-ODE Scriptbook

## Additive relational evidence transport

The original AT path is unchanged: GTrans constructs event representations with
receiver-neighborhood softmax attention, the attention coefficients are mapped
to ordered physical pairs, and `ATTransportPosterior` supplies time-dependent
NRI relation probabilities. Select it with `--edge-posterior-source attention`,
which remains the default for checkpoint compatibility.

The evidence path instead uses the final GTrans event embeddings. For every
cross-object temporal edge, `RelationalEvidenceHead` applies an MLP and
`Softplus` to `[h_dst, h_src, delta_t, mask_dst, mask_src]`. These nonnegative
values are never normalized across physical edges. The receiver event time is
the source time because that is when its aggregation makes the relational
evidence available. Evidence enters age bin zero, advances toward older bins
with a CFL-substepped upwind update, decays by relation type, and leaves through
the open oldest-bin boundary. Prior concentration is then added and normalized
only across relation types for each ordered physical edge.

Springs relation labels are `0 = no spring`, `1 = spring`; evidence mode treats
relation 0 as an inactive message. Charged labels are `0 = opposite charges`
and `1 = like charges`; both are active force laws and both NRI message networks
remain enabled. These mappings follow the checked-in simulators and the loader's
`(edge + 1) / 2` categorical conversion. The mechanism represents persistence
and aging of relational evidence under partial observation, not delayed forces.

### Tensor contract

| Quantity | Shape |
| --- | --- |
| Final event embeddings | `[N_event, H_encoder]` |
| Cross-object event evidence | `[E_event_cross, R]` |
| Evidence aggregated by physical pair | `[B, E_physical, R]` |
| Streaming lag concentration | `[B, E_physical, R, K_lag]` |
| Posterior concentration grid | `[T, B, E_physical, R]` |
| `q_probs_grid` | `[T, B, E_physical, R]` |
| `rel_type_provider(t)` | `[B, E_physical, R]` |

`E_physical = N * (N - 1)`. Its order is derived from the generator's
`rel_send` and `rel_rec` matrices, not reconstructed from an assumed row-major
layout. NRI tiles provider output along the batch dimension when
`n_traj_samples > 1`. In evidence mode, `q_probs_grid` is also supplied to
`odeint_adjoint` as a dynamic adjoint tensor so reconstruction gradients reach
the evidence head and final encoder event embeddings.

### Arguments

| Argument | Default | Purpose |
| --- | --- | --- |
| `--edge-posterior-source` | `attention` | `none`, legacy `attention`, new `evidence`, constant-attention ablation, or topology-only posterior |
| `--evidence-hidden-dim` | `64` | Evidence MLP hidden width |
| `--evidence-init-bias` | `-2.0` | Keeps initial evidence modest and near the prior |
| `--evidence-decay-init` | `1.0` | Initial nonnegative relation decay rate |
| `--edge-prior-strength` | `1.0` | Prior concentration scale |
| `--age-kernel` | `uniform` | `uniform`, `exponential`, or `learned` lag readout |
| `--lag-max` | `1.0` | Maximum retained normalized evidence age |
| `--evidence-l1-coef` | `0.0` | Optional evidence-magnitude penalty |
| `--evidence-ablation` | `full` | `full`, `instantaneous`, `shuffle-times`, `shuffle-values`, or `prior-only` |

`--attn_v` retains its implemented meaning as lag-axis transport velocity.
`gumbel_tau` and `gumbel_hard` are consulted only when `build(..., sample=True)`;
the normal `sample=False` training path returns the posterior mean and is
independent of both settings.

This is a Dirichlet-evidence parameterization of a categorical posterior mean,
not a sampled Dirichlet posterior. The current Charged temporal-event builder
also retains its legacy `edge == +1` cross-object filtering. Consequently,
raw `-1` Charged pairs receive their graph prior and an active NRI relation-0
message law, but no direct cross-event evidence. Changing that event graph would
be a separate data-construction ablation and is intentionally outside this patch.

### Controlled Polaris submissions

From the `AT_ODE` directory, submit the complete requested suite with:

```bash
CHECKPOINT_DIR=/path/to/checkpoints/at_ode_evidence \
  ./submit_evidence_ablations.sh
```

This submits Springs extrapolation at 60% observations for seeds 1991 and 42:
LG-ODE (`none`), legacy attention transport, instantaneous evidence, full
evidence transport, shuffled event times, shuffled evidence values, and prior
only. It also submits full evidence transport for Charged interpolation and
extrapolation at 60% for both seeds. Architecture, optimizer, solver, and
trajectory-sample settings continue to come from `run_polaris_obs_job.sh`.

To submit one configuration directly:

```bash
qsub -v 'DATASET=dataset1,MODE=extrap,OBS_RATIO=0.6,NITERS=100,SEED=1991,EXTRA_ARGS=--edge-posterior-source evidence --evidence-ablation full --age-kernel uniform --edge-prior-strength 1.0 --save /path/to/checkpoints/at_ode_evidence' submit_polaris_obs_job.pbs
```

Run the focused verification suite with:

```bash
python -m pytest -q tests/test_evidence_transport.py
```
