# Experiment Commands

These are the command sequences used to probe, tune, and compare the toy 2D methods.

## 1. WDRO Reference Budget Probe

Use this to measure the reference `wdro` total transport cost on a small setup.

```powershell
python -m toy_2d.train_wild `
  --dataset two_moons `
  --method wdro `
  --epochs 3 `
  --num-samples 512 `
  --batch-size 128 `
  --eval-every 3 `
  --num-eval-samples 512 `
  --metric-samples 256 `
  --device cpu `
  --seed 0 `
  --lr 0.002 `
  --ema-decay 0.99 `
  --hidden-dim 128 `
  --depth 4 `
  --embedding-dim 32 `
  --sampler-steps 60 `
  --wdro-k 5 `
  --wdro-step-size 0.2 `
  --wdro-gamma 3.0 `
  --wdro-p-adv 1.0 `
  --wdro-warmup-epochs 0 `
  --wdro-refresh-every 1 `
  --outdir toy-runs\_sum_probe_wdro
```

## 2. CDRO Budget Probe

Probe `cdro` over a small grid of `path_steps` and `gamma` values.

```powershell
$pathSteps = 8,16,24
$gammas = 0.03,0.1,0.3,1,3,10
foreach ($t in $pathSteps) {
  foreach ($g in $gammas) {
    $tag = "t${t}_g${g}".Replace('.','p')
    python -m toy_2d.train_wild `
      --dataset two_moons `
      --method cdro `
      --epochs 3 `
      --num-samples 512 `
      --batch-size 128 `
      --eval-every 3 `
      --num-eval-samples 512 `
      --metric-samples 256 `
      --device cpu `
      --seed 0 `
      --lr 0.001 `
      --ema-decay 0.99 `
      --hidden-dim 256 `
      --depth 4 `
      --embedding-dim 64 `
      --sampler-steps 20 `
      --cdro-path-steps $t `
      --cdro-inner-steps 2 `
      --cdro-step-size 0.0005 `
      --cdro-gamma $g `
      --cdro-sigma-schedule karras_grid `
      --outdir "toy-runs\_sum_probe_$tag"
  }
}
```

Optional local step-size probe around the matched-budget region:

```powershell
$steps = 0.0002,0.0005,0.001,0.002
foreach ($s in $steps) {
  $tag = "s${s}".Replace('.','p')
  python -m toy_2d.train_wild `
    --dataset two_moons `
    --method cdro `
    --epochs 3 `
    --num-samples 512 `
    --batch-size 128 `
    --eval-every 3 `
    --num-eval-samples 512 `
    --metric-samples 256 `
    --device cpu `
    --seed 0 `
    --lr 0.001 `
    --ema-decay 0.99 `
    --hidden-dim 256 `
    --depth 4 `
    --embedding-dim 64 `
    --sampler-steps 20 `
    --cdro-path-steps 16 `
    --cdro-inner-steps 2 `
    --cdro-step-size $s `
    --cdro-gamma 10 `
    --cdro-sigma-schedule karras_grid `
    --outdir "toy-runs\_sum_probe_$tag"
}
```

## 3. Coarse CDRO Tuning Sweep

This is the main coarse tuning pass for the accumulated-cost `cdro`.

```powershell
python -m toy_2d.tune_wild `
  --outdir toy-runs\tuning_path_sumcost_tmp `
  --datasets eight_gaussians spiral two_moons `
  --methods cdro `
  --trials 12 `
  --workers 4 `
  --epochs 20 `
  --num-samples 2048 `
  --batch-size 128 `
  --eval-every 5 `
  --num-eval-samples 1024 `
  --metric-samples 512 `
  --seed 0
```

Inspect the ranked trials:

```powershell
Get-Content toy-runs\tuning_path_sumcost_tmp\ranking.json
Get-Content toy-runs\tuning_path_sumcost_tmp\best_config.json
```

## 4. Update Method Configs

After selecting the `cdro` config, update the local comparison config files:
- `toy-runs\tuned_method_configs_v1.json`
- `toy-runs\tuned_method_configs_hq_v1.json`
- `toy-runs\tuned_method_configs_xt_v1.json`
- `toy-runs\tuned_method_configs_xt_hq_v1.json`

Keep `baseline` and `wdro` fixed unless you explicitly retune them.

## 5. Seeded Low-Data Comparison

```powershell
python -m toy_2d.compare_methods `
  --outdir toy-runs\method_table_path_v1 `
  --method-configs toy-runs\tuned_method_configs_v1.json `
  --datasets eight_gaussians spiral two_moons `
  --methods baseline wdro cdro `
  --fractions 0.2 0.5 1.0 `
  --full-samples 2000 `
  --seeds 0 1 2 `
  --workers 6 `
  --cleanup-runs `
  --epochs 50 `
  --batch-size 128 `
  --eval-every 5 `
  --num-eval-samples 2048 `
  --metric-samples 1024
```

## 6. High-Quality 100% Comparison

```powershell
python -m toy_2d.compare_methods `
  --outdir toy-runs\high_quality_100pct_path_v1 `
  --method-configs toy-runs\tuned_method_configs_hq_v1.json `
  --datasets eight_gaussians spiral two_moons `
  --methods baseline wdro cdro `
  --fractions 1.0 `
  --full-samples 4000 `
  --seeds 0 `
  --workers 3 `
  --cleanup-runs `
  --epochs 100 `
  --batch-size 128 `
  --eval-every 10 `
  --num-eval-samples 4096 `
  --metric-samples 2048
```

## 7. Optional Cleanup

Delete temporary probe outputs after inspection:

```powershell
cmd /c rmdir /s /q toy-runs\_sum_probe_wdro
cmd /c rmdir /s /q toy-runs\_sum_probe_t8_g0p03
cmd /c rmdir /s /q toy-runs\_sum_probe_t8_g0p1
cmd /c rmdir /s /q toy-runs\_sum_probe_t8_g0p3
cmd /c rmdir /s /q toy-runs\_sum_probe_t8_g1
cmd /c rmdir /s /q toy-runs\_sum_probe_t8_g3
cmd /c rmdir /s /q toy-runs\_sum_probe_t8_g10
cmd /c rmdir /s /q toy-runs\_sum_probe_t16_g0p03
cmd /c rmdir /s /q toy-runs\_sum_probe_t16_g0p1
cmd /c rmdir /s /q toy-runs\_sum_probe_t16_g0p3
cmd /c rmdir /s /q toy-runs\_sum_probe_t16_g1
cmd /c rmdir /s /q toy-runs\_sum_probe_t16_g3
cmd /c rmdir /s /q toy-runs\_sum_probe_t16_g10
cmd /c rmdir /s /q toy-runs\_sum_probe_t24_g0p03
cmd /c rmdir /s /q toy-runs\_sum_probe_t24_g0p1
cmd /c rmdir /s /q toy-runs\_sum_probe_t24_g0p3
cmd /c rmdir /s /q toy-runs\_sum_probe_t24_g1
cmd /c rmdir /s /q toy-runs\_sum_probe_t24_g3
cmd /c rmdir /s /q toy-runs\_sum_probe_t24_g10
cmd /c rmdir /s /q toy-runs\_sum_probe_s0p0002
cmd /c rmdir /s /q toy-runs\_sum_probe_s0p0005
cmd /c rmdir /s /q toy-runs\_sum_probe_s0p001
cmd /c rmdir /s /q toy-runs\_sum_probe_s0p002
```
