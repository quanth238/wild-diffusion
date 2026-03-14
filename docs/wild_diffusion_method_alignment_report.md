# WILD-Diffusion Detailed Alignment Audit

## Scope and Inputs
This report audits the repository implementation against:
- Method section (equations + Algorithm 1) of `pdfs/9149_WILD_Diffusion_A_WDRO_Ins.pdf`.
- Experiment section and appendix implementation settings (Section 4, Appendix C including Tables 3 and 4).

Focus:
1. End-to-end, step-by-step walkthrough of the current code flow.
2. Full hyperparameter cross-check.
3. Explicit mismatch list between paper and code.

Repository audited:
- `train.py`
- `training/training_wdro_loop.py`
- `training/loss.py`
- `training/networks.py`
- `generate.py`
- `fid.py`

---

## A) Paper Spec Snapshot (What Code Should Match)

### A.1 Method-level requirements from paper
- WDRO objective over Wasserstein ball (Eq. 2), dual/surrogate reformulation (Eq. 9a/9b).
- Surrogate-gradient interpretation (Eq. 10).
- Bi-level interval update:
  - Distribution update every `m` epochs.
  - Inner ascent update (Eq. 11):
    - `x_i^k = x_i^{k-1} + ζ ∇_x [ℓ(θ; x_i^{k-1}, t) - γ c(x_i^{k-1}, x_i^0)]`
    - with transport cost `c(x, x0) = 1/2 ||x-x0||_2^2` from Eq. 7.
- Warmup stage for `S_w` epochs before WDRO updates.
- Paper text explicitly states warmup default is 20% of total epochs.

### A.2 Experiment-level requirements from paper
- Section 4 datasets/backbones:
  - Low-res: CIFAR-10, FFHQ, CelebA-HQ with DDPM++ default (and NCSN++/ADM variants).
  - High-res: LSUN-Church with ADM (paper text also mentions ADM coupled with Stable Diffusion).
- FID protocol:
  - Most diffusion results: FID with 50k generated samples vs full training set.
  - Few-shot Table 2/6: FID with 5k samples.
- Sensitivity defaults (Section 4.1 / Appendix C.2):
  - `m = 20`, `K = 5`, perturbation step size `η = 0.01`, `γ = 1`.
- Appendix C.1/C.2 settings:
  - Table 3 architecture summary.
  - Table 4 training defaults:
    - CIFAR-10: duration 200 Mimg, batch 1024, lr `1e-5` (20%) / `1e-4` (100%).
    - FFHQ & CelebA-HQ: duration 200 Mimg, batch 512, lr `2e-4`.
    - LSUN-Church: duration 200 Mimg, batch 256, lr `1e-4`.

---

## B) Step-by-Step Walkthrough of Current Code Flow

## B.1 CLI and config assembly (`train.py`)
1. Parses CLI options for dataset/model/training.
   - Includes: `--arch`, `--precond`, `--duration`, `--batch`, `--lr`, `--ema`, `--dropout`, `--augment`, `--xflip`, etc.
   - Ref: `train.py:35-69`.
2. Builds config dicts:
   - dataset: `training.dataset.ImageFolderDataset`
   - optimizer: Adam `(lr, betas=(0.9, 0.999), eps=1e-8)`
   - Refs: `train.py:88-93`.
3. Validates dataset resolution/size/labels.
   - Refs: `train.py:95-104`.
4. Selects architecture knobs:
   - `ddpmpp`: `SongUNet`, positional embedding, standard encoder.
   - `ncsnpp`: `SongUNet`, fourier embedding, residual encoder.
   - `adm`: `DhariwalUNet`.
   - Refs: `train.py:107-115`.
5. Selects preconditioning/loss:
   - `wdroedm` -> `EDMPrecond + EDMLossWdro`.
   - Refs: `train.py:122-124`.
6. Applies augmentation setup when `--augment > 0`.
   - Ref: `train.py:131-134`.
7. Converts high-level training options to internal values:
   - `total_kimg = duration * 1000`.
   - `ema_halflife_kimg = ema * 1000`.
   - Refs: `train.py:138-143`.
8. Dispatches to WDRO loop:
   - `training_wdro_loop.training_loop(**c)`.
   - Ref: `train.py:218`.

## B.2 Training initialization (`training/training_wdro_loop.py`)
9. Initializes RNG/CUDA backend and distributed setup assumptions.
   - Ref: `training/training_wdro_loop.py:43-49`.
10. Builds infinite dataloader from original dataset.
    - Ref: `training/training_wdro_loop.py:60-63`.
11. Constructs network and prints module summary.
    - Refs: `training/training_wdro_loop.py:73-82`.
12. Instantiates loss, optimizer, augment pipe, DDP wrapper, EMA copy.
    - Refs: `training/training_wdro_loop.py:85-92`.
13. Optional resume from snapshot/state dump.
    - Refs: `training/training_wdro_loop.py:94-110`.

## B.3 WDRO schedule setup in code
14. Sets WDRO start point as 40% of run:
    - `wdro_start_kimg = int(0.4*total_kimg)`.
    - Ref: `training/training_wdro_loop.py:124`.
15. Sets WDRO refresh interval based on dataset size:
    - `wdro_interval_kimg = int(100 * len(dataset_obj) / 1000)`.
    - Ref: `training/training_wdro_loop.py:131`.
16. Overrides augmentation probability by dataset size bucket:
    - `p_now = 0.12/0.15/0.18`, then `augment_pipe.p = p_now`.
    - Refs: `training/training_wdro_loop.py:134-135`.

## B.4 Main train iteration path
17. If refresh trigger reached (`cur_nimg >= next_wdro_kimg*1000`), WDRO dataset regeneration runs.
    - Refs: `training/training_wdro_loop.py:138-147`.
18. Rank 0 iterates full raw dataset (no shuffle), normalizes to `[-1,1]`.
    - Refs: `training/training_wdro_loop.py:150-159`.
19. For each raw batch, adversarial generation is stochastic with fixed probability `p_adv=0.3`.
    - Refs: `training/training_wdro_loop.py:156-161`.
20. Saves concatenated `(original + optional adversarial)` tensors to disk as `combined_dataset-*.pt`.
    - Refs: `training/training_wdro_loop.py:167-173`.
21. All ranks switch iterator to `TensorDataset(combined_images, combined_labels)`.
    - Refs: `training/training_wdro_loop.py:176-185`, `281-294`.
22. Updates next refresh point by interval.
    - Ref: `training/training_wdro_loop.py:186`.

## B.5 Parameter update path per iteration
23. Draws minibatch from current iterator.
24. Casts images to float32 in current loop path (no `/127.5 - 1` here).
25. Computes `EDMLossWdro`, backprop, optimizer step, EMA step.
    - Refs: `training/training_wdro_loop.py:192-213`.
26. LR schedule is linear warmup to target over `lr_rampup_kimg=10000`.
    - Ref: `training/training_wdro_loop.py:31`, `200-201`.
27. Saves snapshots/state dumps/logs by tick settings.
    - Refs: `training/training_wdro_loop.py:243-266`.

## B.6 Inner adversarial update (`wdro_attack`)
28. Starts from `x_adv = x0 = images`.
29. For `iters` steps (default 2):
    - Computes diffusion loss on `x_adv` via `loss_fn`.
    - Computes penalty `C = mean(||x_adv - images||_2)`.
    - Maximizes `loss_cls.mean() - gamma*C` with ascent step `x_adv += alpha*grad`.
    - Clamps to `[-1,1]`.
    - Refs: `training/training_wdro_loop.py:296-315`.
30. Defaults: `gamma=1.0`, `alpha=1e-3`, `iters=2`, `clamp=(-1,1)`.
    - Ref: `training/training_wdro_loop.py:298`.

## B.7 Loss behavior (`EDMLossWdro`)
31. Every call samples fresh `sigma` (log-normal style) and fresh noise `n`.
32. Computes EDM denoising loss `weight * (D(y+n)-y)^2`.
    - Refs: `training/loss.py:13-20`.

## B.8 Evaluation scripts (for experiment protocol)
33. `generate.py` defaults:
    - seeds `0-63`, steps `35`, batch `64`.
    - Refs: `generate.py:209`, `212`, `214`.
34. `fid.py calc` defaults:
    - `--num 50000`, `--batch 64`.
    - Refs: `fid.py:116`, `118`.
35. Few-shot 5k FID is possible only by manual `--num 5000`; no dedicated few-shot pipeline script.

---

## C) Complete Hyperparameter Inventory and Cross-Check

Status legend:
- `Aligned`: code default/behavior matches paper statement.
- `Configurable`: paper value not default, but can be set through existing CLI without code changes.
- `Mismatch`: current code behavior/default diverges from paper method or stated experiment default.
- `Missing`: paper-reported setup/component not implemented in repository.

## C.1 Core method hyperparameters (WDRO algorithm)

| Hyperparameter | Paper (Method/Section) | Code value/behavior | Location | Status |
|---|---|---|---|---|
| Warmup ratio `Sw/S` | 20% warmup (Sec. 3.1 text) | 40% (`int(0.4*total_kimg)`) | `training/training_wdro_loop.py:124` | Mismatch |
| Interval `m` | default `m=20` (Sec. 4.1, C.2) | `int(100*len(dataset)/1000)` kimg (dataset-dependent) | `training/training_wdro_loop.py:131` | Mismatch |
| Inner steps `K` | default `K=5` (Sec. 4.1, C.2) | `iters=2` | `training/training_wdro_loop.py:298` | Mismatch |
| Inner step size | default `0.01` (Sec. 4.1, C.2) | `alpha=1e-3` | `training/training_wdro_loop.py:298` | Mismatch |
| Penalty `γ` | default `γ=1` (Sec. 4.1, C.2) | `gamma=1.0` | `training/training_wdro_loop.py:298` | Aligned |
| Transport cost `c(x,x0)` | `1/2 ||x-x0||^2` (Eq. 7) | `mean(||x-x0||_2)` (unsquared) | `training/training_wdro_loop.py:310` | Mismatch |
| Per-sample adversarial generation | Algorithm 1 uses all `i=1..n` | stochastic subset with `p_adv=0.3` | `training/training_wdro_loop.py:156-161` | Mismatch |
| Parameter update form | Algorithm 1 uses explicit `ℓ(x_i)+ℓ(x'_i)` per i | merged dataset sampling, not explicit paired term | `training/training_wdro_loop.py:167-197` | Mismatch |
| Fixed `t` within inner loop | Algorithm 1 samples `t` once before K-loop | `loss_fn` re-samples sigma/noise every step | `training/training_wdro_loop.py:303-308`, `training/loss.py:13-18` | Mismatch |
| Worst-case set fixed between intervals | yes (paper) | yes (saved combined dataset until next refresh) | `training/training_wdro_loop.py:172-186` | Aligned |
| WDRO hyperparameters exposed | expected tunable for sensitivity claims | not exposed in CLI (`m`, `K`, `alpha`, `gamma`, `p_adv`) | `train.py` + `training/training_wdro_loop.py` | Mismatch |

## C.2 Training hyperparameters from paper Table 4

| Setting | Paper Table 4 | Code default (`train.py`) | Can match via CLI? | Status |
|---|---|---|---|---|
| Duration | 200 Mimg | 200 Mimg | yes (`--duration`) | Aligned |
| CIFAR minibatch | 1024 | 512 | yes (`--batch`) | Configurable |
| CIFAR lr (20%/100%) | `1e-5` / `1e-4` | `1e-4` | yes (`--lr`) | Configurable |
| FFHQ/CelebA batch | 512 | 512 | yes | Aligned |
| FFHQ/CelebA lr | `2e-4` | `1e-4` | yes (`--lr 2e-4`) | Configurable |
| LSUN batch | 256 | 512 | yes (`--batch 256`) | Configurable |
| LSUN lr | `1e-4` | `1e-4` | yes | Aligned |

Note: code does not auto-select dataset-specific batch/lr; paper table values require manual CLI override per run.

## C.3 Architecture cross-check (paper Table 3 vs code)

| Architecture property | Paper Table 3 | Code | Status |
|---|---|---|---|
| DDPM++ resampling filter | Box | `[1,1]` | Aligned |
| NCSN++ resampling filter | Bilinear | `[1,3,3,1]` | Aligned |
| DDPM++ noise embedding | Positional | `embedding_type='positional'` | Aligned |
| NCSN++ noise embedding | Fourier | `embedding_type='fourier'` | Aligned |
| NCSN++ residual encoder skip | Residual | `encoder_type='residual'` | Aligned |
| Residual blocks per resolution | DDPM++/NCSN++: 4, ADM: 3 | `SongUNet num_blocks=4`, `DhariwalUNet num_blocks=3` | Aligned |
| Attention resolutions | DDPM++/NCSN++ `{16}`, ADM `{32,16,8}` | `SongUNet [16]`, `DhariwalUNet [32,16,8]` | Aligned by architecture defaults |
| Attention heads | DDPM++/NCSN++ 1, ADM 6-9-12 | `SongUNet num_heads=1`; ADM uses `channels_per_head=64` giving multi-head scaling | Aligned in design intent |

Important nuance: training defaults for ADM in `train.py` set `channel_mult=[1,2,3,4]`; whether all listed attention resolutions are instantiated depends on image resolution and effective depth.

## C.4 Additional code hyperparameters (not explicitly specified in paper)

These are active in implementation and affect behavior/reproducibility:
- Optimizer: Adam betas `(0.9, 0.999)`, eps `1e-8` (`train.py:92`).
- EMA half-life default `0.5 Mimg` via `--ema`; EMA rampup ratio `0.05` in loop signature (`training_wdro_loop.py:30`).
- LR rampup fixed `10000 kimg` (`training_wdro_loop.py:31`).
- Augmentation pipeline probabilities and transforms initialized from `--augment` then overridden by dataset-size heuristic `p_now`.
  - `train.py:131-134`, `training_wdro_loop.py:134-135`.
- WDRO stochastic augmentation probability `p_adv=0.3` (`training_wdro_loop.py:156`).
- Attack clamp `[-1,1]` (`training_wdro_loop.py:298,314`).
- Mixed precision default off (`--fp16=False`, `train.py:54`).
- DataLoader workers default `1` (`train.py:58`).
- Logging/snapshot cadence defaults (`tick=50`, `snap=50`, `dump=500`) (`train.py:63-65`).

## C.5 Exhaustive `train.py` option catalog

| Option | Default | Role in code | Paper mention | Status |
|---|---|---|---|---|
| `--outdir` | required | output directory | not method-specific | N/A |
| `--data` | required | dataset path | datasets discussed in Sec. 4 | Aligned intent |
| `--cond` | `False` | class-conditional toggle | both cond/uncond results reported | Configurable |
| `--arch` | `ddpmpp` | selects `ddpmpp/ncsnpp/adm` | all 3 used in paper | Aligned |
| `--precond` | `wdroedm` | selects WDRO or ADV loss path | WDRO focus in paper | Aligned |
| `--duration` | `200` (Mimg) | total training budget | Table 4 uses 200 | Aligned |
| `--batch` | `512` | global minibatch | Table 4 dataset-specific | Configurable |
| `--batch-gpu` | unset | per-GPU cap | not explicitly specified | N/A |
| `--cbase` | unset | override model channels | not specified | N/A |
| `--cres` | unset | override channel multipliers | not specified | N/A |
| `--lr` | `1e-4` | optimizer lr | Table 4 dataset-specific | Configurable |
| `--ema` | `0.5` (Mimg) | EMA half-life | not explicitly tabulated | N/A |
| `--dropout` | `0.13` | network dropout | not explicitly tabulated | N/A |
| `--augment` | `0.12` | augment pipe base prob | paper compares augment methods, no exact default for this pipe | Partially aligned |
| `--xflip` | `False` | dataset xflip | not explicitly tabulated | N/A |
| `--fp16` | `False` | mixed precision | not explicitly tabulated | N/A |
| `--ls` | `1` | loss scaling | not explicitly tabulated | N/A |
| `--bench` | `True` | cuDNN benchmarking | not explicitly tabulated | N/A |
| `--cache` | `True` | dataset CPU cache | not explicitly tabulated | N/A |
| `--workers` | `1` | DataLoader workers | not explicitly tabulated | N/A |
| `--desc` | unset | run name suffix | not relevant to paper method | N/A |
| `--nosubdir` | `False` | output layout | not relevant to paper method | N/A |
| `--tick` | `50` | print interval | not relevant to method equations | N/A |
| `--snap` | `50` | snapshot frequency | not relevant to method equations | N/A |
| `--dump` | `500` | state dump frequency | not relevant to method equations | N/A |
| `--seed` | random | run seed | random subset mention in paper, but not exact seed | Configurable |
| `--transfer` | unset | load pretrained snapshot | paper has pretrained/non-pretrained settings | Configurable |
| `--resume` | unset | resume training state | engineering utility | N/A |
| `--dry-run` | `False` | print config and exit | engineering utility | N/A |

## C.6 Exhaustive WDRO internal constants (not CLI-exposed)

| Internal parameter | Code value | Location | Paper relation | Status |
|---|---|---|---|---|
| `wdro_start_kimg` | `0.4 * total_kimg` | `training/training_wdro_loop.py:124` | warmup ratio should be 20% | Mismatch |
| `wdro_interval_kimg` | `int(100*n/1000)` | `training/training_wdro_loop.py:131` | paper default `m=20` epochs | Mismatch |
| `p_now` augment override | `0.12/0.15/0.18` by dataset size | `training/training_wdro_loop.py:134-135` | not described in paper method | Extra heuristic |
| `p_adv` | `0.3` | `training/training_wdro_loop.py:156` | paper Algorithm 1 perturbs all samples | Mismatch |
| attack `gamma` | `1.0` | `training/training_wdro_loop.py:298` | paper default `γ=1` | Aligned |
| attack step size | `alpha=1e-3` | `training/training_wdro_loop.py:298` | paper default perturbation step `0.01` | Mismatch |
| attack steps | `iters=2` | `training/training_wdro_loop.py:298` | paper default `K=5` | Mismatch |
| attack clamp | `[-1,1]` | `training/training_wdro_loop.py:298` | not in Eq. 11 | Extra heuristic |
| LR rampup budget | `10000 kimg` | `training/training_wdro_loop.py:31` | not explicitly reported | N/A |
| EMA rampup ratio | `0.05` | `training/training_wdro_loop.py:30` | not explicitly reported | N/A |
| Optimizer betas/eps | `(0.9,0.999),1e-8` | `train.py:92` | not explicitly reported | N/A |
| Loss params `P_mean/P_std/sigma_data` | `-1.2/1.2/0.5` | `training/loss.py:7-10` | EDM default-style, not paper-specific WDRO knobs | Partially aligned |

## C.7 Evaluation-script hyperparameter catalog

### `generate.py`

| Option | Default | Paper relation | Status |
|---|---|---|---|
| `--seeds` | `0-63` | paper FID uses 50k generations | Requires override for paper protocol |
| `--batch` | `64` | implementation detail | Configurable |
| `--steps` | `35` | not explicitly fixed in paper text | N/A |
| `--rho` | `7` | EDM sampling default | N/A |
| `--S_churn/S_min/S_max/S_noise` | `0/0/inf/1` | stochastic sampler controls not discussed in paper method | N/A |
| `--solver/disc/schedule/scaling` | unset | ablation sampler switches | N/A |

### `fid.py`

| Option | Default | Paper relation | Status |
|---|---|---|---|
| `calc --num` | `50000` | matches main diffusion FID protocol | Aligned |
| `calc --batch` | `64` | implementation detail | N/A |
| `calc --seed` | `0` | sampling subset control | N/A |
| `ref --batch` | `64` | reference-stat computation detail | N/A |

---

## D) Flagged Mismatches (Method Section)

### D1. Critical: transport penalty is not Eq. (7)
- Paper: `c(x,x0)=1/2||x-x0||^2`.
- Code: unsquared mean L2 norm.
- Impact: changes inner maximization geometry and gradients.
- Refs: `training/training_wdro_loop.py:310-311`.

### D2. High: Algorithm 1 all-sample worst-case generation is reduced to random subset
- Paper Algorithm 1 builds full `D={x_i^K}_{i=1}^n`.
- Code perturbs only with Bernoulli `p_adv=0.3` at batch level.
- Refs: `training/training_wdro_loop.py:156-166`.

### D3. High: update objective differs from Algorithm 1 line 23
- Paper uses explicit paired objective `ℓ(x_i,t)+ℓ(x'_i,t)`.
- Code trains on merged dataset sampling (original + optional adversarial), no explicit pairing.
- Refs: `training/training_wdro_loop.py:167-197`.

### D4. High: warmup default is 40% not 20%
- Paper states 20% warmup.
- Code starts WDRO after 40% progress.
- Refs: `training/training_wdro_loop.py:124`.

### D5. High: interval default departs from `m=20`
- Paper default: `m=20`.
- Code uses dataset-dependent interval equivalent to ~100 epochs (before integer effects), not fixed 20.
- Refs: `training/training_wdro_loop.py:131`.

### D6. High: inner defaults (`K`, step size) differ from paper defaults
- Paper: `K=5`, step size `0.01`, `γ=1`.
- Code: `iters=2`, `alpha=1e-3`, `gamma=1.0`.
- Refs: `training/training_wdro_loop.py:298`.

### D7. Medium: inner-loop target is stochastic across ascent steps
- Paper Algorithm 1 samples `t` once per inner loop.
- Code re-calls stochastic `EDMLossWdro` each step with fresh sigma/noise.
- Refs: `training/training_wdro_loop.py:303-308`, `training/loss.py:13-18`.

### D8. Medium: inconsistent image scaling within WDRO loop phases
- WDRO dataset generation normalizes to `[-1,1]`.
- Main training path in WDRO loop does not normalize incoming iterator batches.
- Refs: `training/training_wdro_loop.py:158`, `training/training_wdro_loop.py:193`.

### D9. Medium: core WDRO knobs are hardcoded (not externally tunable)
- Paper reports sensitivity of `m`, `K`, step size, `γ`.
- Code hardcodes `m` logic, `K`, `alpha`, `p_adv` without CLI exposure.
- Refs: `train.py` options, `training/training_wdro_loop.py:124,131,156,298`.

---

## E) Flagged Mismatches (Experiment Section / Reproducibility)

### E1. Dataset-specific training settings in Table 4 are not encoded as run defaults
- Paper uses dataset-dependent minibatch/lr.
- Code has global defaults (`batch=512`, `lr=1e-4`) and no automatic per-dataset switching.
- Refs: `train.py:43`, `train.py:47`.
- Classification: reproducibility mismatch by default, but manually configurable.

### E2. Reported high-resolution "ADM + Stable Diffusion" setup is not present in repository code
- Code contains EDM/DDPM++/NCSN++/ADM training only; no Stable Diffusion/DreamBooth training implementation.
- Refs: repository file set + `train.py` architecture options.
- Classification: missing experimental component.

### E3. Few-shot GAN and DreamBooth experiment pipelines are missing
- Paper Appendix C.4/C.5 reports GAN-style few-shot and DreamBooth text-to-image experiments.
- Repository has no StyleGAN/DreamBooth codepaths.
- Classification: missing experimental component.

### E4. Data subset protocol (20%/50%/100%) is not scripted in repo
- Paper repeatedly uses random subset fractions.
- Code lacks built-in sampling/splitting utility; must be done externally.
- Classification: reproducibility gap.

### E5. Baseline methods in tables are not implemented in this repo
- Patch Diffusion, DeepCache, DDIM baselines, etc. are not code-integrated.
- Classification: expected for compact release, but not end-to-end reproducible from single repo.

### E6. FID protocol support is mostly aligned but manual
- `fid.py` default `--num=50000` aligns with main diffusion tables.
- Few-shot `5k` requires manual override (`--num 5000`), not dedicated script preset.
- Refs: `fid.py:116`.

---

## F) Confirmed Alignments

1. The repo is built on EDM-style code and supports DDPM++/NCSN++/ADM backbones as paper states.
2. The implementation uses a bi-level pattern with periodic dataset refresh and frequent parameter updates.
3. Gamma-penalized adversarial sample generation exists in code.
4. FID tooling supports 50k-evaluation protocol by default.

---

## G) Bottom-Line Assessment

The codebase is **WDRO-inspired but not a strict implementation of the paper’s Algorithm 1 and Eq. (7)/(11) defaults**. The largest method-level divergences are:
- non-quadratic transport penalty,
- subset-only adversarial generation,
- different warmup/interval schedules,
- different inner-loop defaults (`K`, step size),
- and stochastic inner objective behavior.

At the experiment level, architecture family support is present, but several paper-reported experimental tracks (few-shot GAN, DreamBooth/Stable-Diffusion pipeline, subset protocol automation, baseline integrations) are not included in this repository, limiting full reproduction from code-as-is.
