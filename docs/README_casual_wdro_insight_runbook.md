# Casual WDRO (Insight-First) Runbook

This document describes the **minimal but critical** implementation to test your idea quickly.
The goal is not full pathwise theory fidelity; the goal is to answer:

- Does multi-time WDRO perturbation improve StageA continuation quality?
- Is the compute overhead acceptable for your workflow?

## 1) What is implemented (and what is intentionally not)

Implemented (core signal):
- Keep existing WILD-Diffusion training pipeline (resume/full-state, interval refresh, evaluation).
- Add a new WDRO attack mode: `casual` (`causal` alias accepted).
- In each WDRO refresh, each clean batch can generate `M` adversarial variants at `M` distinct discrete timesteps.
- Inner optimization uses decoupled perturbations and penalty `0.5 * ||delta||^2`.
- Warmup is preserved through existing WDRO schedule start (`wdro_warmup_ratio`).

Intentionally not implemented (for this insight phase):
- Full pathwise coupling/BPTT across the entire diffusion trajectory.
- Custom time weights `omega_k` and position weights `alpha_k` (implicitly uniform now).
- Exact outer-loop path objective over full trajectories.

## 2) Warmup is included

Your friend's algorithm missed warmup. This implementation keeps warmup via:
- `WDRO_WARMUP_RATIO` and scheduler start before first WDRO refresh.
- For paper-like setting: `WDRO_WARMUP_RATIO=0.2`.

## 3) Practical algorithm used in code

Given clean minibatch `x`:

1. If WDRO refresh trigger is reached (every `m` epochs), generate adversarial buffer:
- Sample `M` distinct timestep bins per sample.
- Map bins to EDM sigma levels using a discretized EDM schedule (`T=WDRO_TIME_BINS`).
- For each selected timestep (or chunk of timesteps):
  - Initialize `delta=0`.
  - Repeat `R=WDRO_K` ascent steps on:
    - `loss(theta; x+delta, t_j) - gamma * 0.5 * ||delta||^2`.
  - Clamp `x+delta` to `[-1,1]`.
- Store all adversarial variants in WDRO buffer + clean images.

2. Between refreshes, reuse saved WDRO buffer exactly as baseline loop does.

3. Outer model update remains unchanged (train on mixed dataset stream).

## 4) Code changes (for audit)

- `train.py`
  - Added CLI:
    - `--wdro-attack-mode single|casual|causal`
    - `--wdro-m-times`
    - `--wdro-time-bins`
    - `--wdro-time-chunk`

- `training/loss.py`
  - `EDMLossWdro` now supports optional fixed-time attack inputs:
    - `sigma_override`
    - `noise_override`

- `training/training_wdro_loop.py`
  - Added WDRO mode parsing and validation.
  - Added multi-time attack path in WDRO refresh block.
  - Added helper functions:
    - EDM sigma grid build
    - distinct timestep sampling
    - fixed-sigma attack
    - multi-time attack
  - Added refresh telemetry for feasibility checks:
    - log line with clean/adv/ratio/time
    - `casual_refresh_metrics.jsonl`

- `scripts/setup_and_train_cifar10.sh`
  - Added env passthrough:
    - `WDRO_ATTACK_MODE`
    - `WDRO_M_TIMES`
    - `WDRO_TIME_BINS`
    - `WDRO_TIME_CHUNK`

## 5) StageA A/B experiment protocol (recommended)

Use the same anchor and seed for fair comparison.

### A) Baseline StageA (single-time WDRO)

Follow existing doc:
- `docs/README_stageA_resume_full_state_40k.md`

Ensure:
- `WDRO_ATTACK_MODE=single`
- Keep all other hyperparameters fixed.

### B) Candidate StageA (casual WDRO)

Use the same command as StageA, only add/override:

```bash
WDRO_ATTACK_MODE=casual \
WDRO_M_TIMES=4 \
WDRO_TIME_BINS=40 \
WDRO_TIME_CHUNK=1 \
WDRO_P_ADV=1.0
```

Recommended first candidate name:
- `EXP_NAME=stageA-casual-plus20-seed0`

## 6) Verification checklist (fast and concrete)

After training starts:

1. Confirm mode in `RUN_DIR/log.txt`:
- contains `attack mode=casual`
- contains `m_times=4, time_bins=40`

2. Confirm resume is true full-state:
- `resume_state_dump` is not `None`
- log contains `Loading training state from ...training-state-040000.pt`

3. Confirm WDRO refresh happened:
- `combined_dataset-*.pt` files appear in run dir.
- `casual_refresh_metrics.jsonl` appears in run dir (for casual mode).

4. Run official eval (50k FID) using existing eval script.

5. Compare baseline vs casual using same protocol:
- final FID
- training stability (no NaN/crash)
- wall-clock overhead

## 7) Interpreting outcome for go/no-go

Continue this direction if most conditions hold:
- FID improves or is comparable with better qualitative robustness.
- No instability (NaN / repeated divergence).
- Overhead is acceptable for your scheduling budget.

Stop or revise if:
- FID degrades consistently across repeated seeds.
- Refresh cost is too high for practical iteration.

## 8) Safe fallback knobs

If OOM/slow:
- Reduce `WDRO_M_TIMES` first (`4 -> 2`).
- Keep `WDRO_TIME_CHUNK=1` for lowest VRAM.
- Reduce `BATCH_GPU` if needed.

If signal is weak:
- Increase `WDRO_M_TIMES` (`2 -> 4`) before changing many other knobs.
- Keep `WDRO_K=5`, `WDRO_STEP_SIZE=0.01`, `WDRO_GAMMA=1.0` fixed for first A/B.

## 9) Alignment to your algorithm (quick truth table)

Matched:
- Alternating structure (inner adversarial update + outer model update).
- Separable perturbations per selected timestep.
- Gradient-ascent update with quadratic penalty.
- Optional projection implemented as clamp `[-1,1]`.
- Warmup stage is preserved.

Approximation used (intentional for insight-first):
- Uses `M` sampled timesteps per sample instead of full `k=0..K`.
- Uses image-space states rather than explicitly storing full diffusion paths.
- Uses implicit uniform weights (no explicit `alpha_k`, `omega_k` yet).

This is enough to test feasibility signal quickly before full theoretical expansion.
