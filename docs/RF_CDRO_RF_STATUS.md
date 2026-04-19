# RF / CDRO-RF Status

This note is the shortest current description of the toy-side RF additions.

## Public Naming

- Use `RF` for the public clean rectified-flow baseline.
- Use `CDRO-RF` for the robust RF method.
- Use `Wild-Diffusion` in plots/presentation for the historically named `wdro` method.
- Use `Wild-Diffusion-RF` for `method_version=wdro --training-objective rf`.

## Clean RF Baseline

The public `RF` baseline in the toy stack is a strong two-stage baseline:

1. `rf_stage1`: continuous-time 1-RF training on straight pairs `(x_L, x_R) = (z, x0)` with
   `x_t = (1 - t) x_L + t x_R` and target velocity `d = x_R - x_L`.
2. `rf_reflow`: one frozen one-round reflow refresh, training on pairs
   `(x_L, x_R) = (z, x_hat_teacher(z))`.

Implementation notes:

- Stage-1 timestep law is uniform in continuous `t`.
- Reflow-stage timestep law is controlled by `rf_reflow_t_distribution` and currently defaults to `u_shaped`.
- EMA is kept in both stages.
- The shared EDM warm start is required through `rf_edm_init_ckpt_path`.

## CDRO-RF

`CDRO-RF` keeps the existing CDRO shell as literally as possible, but swaps the EDM reference path for an RF straight path.

For a pair `(x_L, x_R)`, define:

- `d = x_R - x_L`
- `x_ref(t) = x_L + t d`

The controlled rollout uses:

- `x_ctrl^0 = x_L`
- `x_nom^{k+1} = x_ctrl^k + Delta t_k d`
- `x_ctrl^{k+1} = x_nom^{k+1} + delta_k`

The training target stays fixed as `d = x_R - x_L`, and the loss is path-averaged RF regression on the controlled states.

Current staged pair law:

1. robust stage-1 uses standard RF pairs `(z, x0)`
2. robust reflow stage uses frozen one-round reflow pairs `(z, x_hat_teacher(z))`

`CDRO-RF` also requires the shared EDM warm-start checkpoint and skips the toy-side
baseline warmup phase entirely.

## Wild-Diffusion-RF

`Wild-Diffusion-RF` keeps the existing WDRO shell but swaps the continuation loss
to RF after the shared EDM warm start.

Current direct-run behavior:

1. robust RF stage-1 rebuilds the WDRO augmented pool with worst-case samples and
   trains on RF pairs `(z, x_adv)`
2. robust RF reflow freezes the stage-1 teacher and trains on one-round reflow pairs
   `(z, x_hat_teacher(z))`

Implementation notes:

- `method_version=wdro --training-objective rf` is the direct `Wild-Diffusion-RF` path.
- The shared EDM warm start is required through `rf_edm_init_ckpt_path`.
- The shared EDM checkpoint is treated as the warmup prefix, so the in-run WDRO-RF
  budget resumes from that checkpoint and goes straight into RF robust continuation.
- `wdro_warmup_fraction` is ignored for the direct `Wild-Diffusion-RF` path.
- Shared-grid collector support is still deferred; direct `toy/run_toy.py` runs are
  the supported path today.

## Time-Law Matching

CDRO-RF rollout grids are stage-matched to the clean RF timestep law:

- robust RF stage-1 matches clean RF stage-1
- robust RF reflow matches clean RF reflow

The robust rollout now uses a stratified Monte Carlo discretization of the
stage-matched RF time law:

- stage-1 rollout nodes are stratified samples from uniform `t`
- reflow rollout nodes are stratified samples from the configured reflow law
- conditional on the sampled nodes, the RF reference path remains the straight
  path `x_ref(t) = x_L + t d`

Deterministic right-endpoint RF grids remain only as solver/reference grids for
sampling, evaluation, and frozen-teacher reflow pair generation.

## What Is Still Deferred

- shared-grid collector / checkpointed sweep support for `Wild-Diffusion-RF`
- broader paper-facing writeups outside the toy stack
- any claim that docs are the full source of truth for experiment settings

Checkpoint note:

- `toy/scripts/reevaluate_three_method_fids_from_checkpoints.py` can reevaluate FIDs
  for RF checkpoints.
- The extra `edm_clean_probe` is EDM-only and is reported unsupported for RF checkpoints.

Use `metrics.json` artifacts under `toy_outputs/` as the authoritative record of exact run settings.
