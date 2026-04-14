# RF / CDRO-RF Status

This note is the shortest current description of the toy-side RF additions.

## Public Naming

- Use `RF` for the public clean rectified-flow baseline.
- Use `CDRO-RF` for the robust RF method.
- Use `Wild-Diffusion` in plots/presentation for the historically named `wdro` method.
- `Wild-Diffusion-RF` is not implemented for the current milestone.

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
- Optional EDM warm start is exposed through `rf_edm_init_ckpt_path`.

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

- `Wild-Diffusion-RF`
- broader paper-facing writeups outside the toy stack
- any claim that docs are the full source of truth for experiment settings

Use `metrics.json` artifacts under `toy_outputs/` as the authoritative record of exact run settings.
