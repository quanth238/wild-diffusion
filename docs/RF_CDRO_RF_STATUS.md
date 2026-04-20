# RF / CDRO-RF Status

This note is the shortest current description of the toy-side RF additions.

## Public Naming

- Use `RF` for the public clean rectified-flow baseline.
- Use `CDRO-RF` for the robust RF method.
- Use `Wild-Diffusion` in plots/presentation for the historically named `wdro` method.
- Use `Wild-Diffusion-RF` for `method_version=wdro --training-objective rf`.

## Clean RF Baseline

The public `RF` baseline in the toy stack now follows the explicit RF++-style
shared-teacher protocol:

1. start from the same shared EDM checkpoint used by the EDM family
2. use that checkpoint as the frozen teacher
3. train the RF student directly on teacher-generated reflow pairs
   `(x_L, x_R) = (z, x_hat_teacher(z))`

Implementation notes:

- Reflow-stage timestep law is controlled by `rf_reflow_t_distribution` and currently defaults to `u_shaped`.
- EMA is kept for the RF-family student.
- The shared EDM warm start is required through `rf_edm_init_ckpt_path`.
- The shared RF family reflow/teacher grid now defaults to `rf_teacher_n_steps_path=40`.
- The shared RF family eval/FID grid now defaults to `rf_eval_n_steps_path=9`.
- RF-family training now uses one explicit shared-EDM-teacher reflow protocol rather than an in-run
  stage split. `rf_stage1_fraction` and `rf_reflow_start_step` have been removed from the public
  RF launch surfaces.
- Shared-grid RF collection therefore treats RF-family continuation as reflow-only off the shared
  EDM branchpoint, and the family reflow boundary now coincides with that shared warm-start point.

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

Current pair law:

1. use the shared EDM checkpoint as the frozen teacher
2. robustify one explicit reflow student on teacher-generated pairs `(z, x_hat_teacher(z))`

`CDRO-RF` also requires the shared EDM warm-start checkpoint and skips the toy-side
baseline warmup phase entirely.

## Wild-Diffusion-RF

`Wild-Diffusion-RF` keeps the existing WDRO shell but swaps the continuation loss
to RF after the shared EDM warm start.

Current direct-run behavior:

1. use the shared EDM checkpoint as the frozen teacher
2. construct WDRO adversarial reflow pairs against that teacher-generated pair distribution
3. train one explicit reflow student on those robustified pairs

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

Default RF-family step split:

- shared RF teacher / reflow grid: `40`
- shared RF eval / FID grid: `9`
- RF-family training grid (`n_steps_path`) is unified at the shared teacher setting for RF runs,
  so clean RF, Wild-Diffusion-RF, and CDRO-RF now all use `40`.

## What Is Still Deferred

- shared-grid collector / checkpointed sweep support for `Wild-Diffusion-RF`
- broader paper-facing writeups outside the toy stack
- any claim that docs are the full source of truth for experiment settings

Checkpoint note:

- `toy/scripts/reevaluate_three_method_fids_from_checkpoints.py` can reevaluate FIDs
  for RF checkpoints.
- The reevaluator now records `rf_clean_probe` for RF checkpoints and keeps
  `edm_clean_probe` as unsupported there.
- `objective_clean_probe` is the generic row field that selects `edm_clean_probe`
  for EDM-family checkpoints and `rf_clean_probe` for RF-family checkpoints.

Use `metrics.json` artifacts under `toy_outputs/` as the authoritative record of exact run settings.
