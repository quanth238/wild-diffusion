#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _parse_seeds(seeds_text: str) -> List[int]:
    out = []
    for s in seeds_text.split(","):
        s = s.strip()
        if s:
            out.append(int(s))
    if not out:
        raise ValueError("At least one seed is required.")
    return out


def _run(cmd: List[str], dry_run: bool = False) -> None:
    print("[run]", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _load_metrics(path: Path) -> Dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in obj:
        raise ValueError(f"Invalid metrics payload: {path}")
    return obj["metrics"]


def _safe_mean(xs: List[float]) -> float:
    return mean(xs) if len(xs) > 0 else float("nan")


def _safe_last(xs: List[float]) -> float:
    return xs[-1] if len(xs) > 0 else float("nan")


def _summary_final(summary_obj: Dict) -> float:
    if not isinstance(summary_obj, dict):
        return float("nan")
    val = summary_obj.get("final", None)
    if val is None:
        return float("nan")
    return float(val)


def _trimmed_mean_curve(curves: List[List[float]]) -> List[float]:
    curves = [c for c in curves if len(c) > 0]
    if len(curves) == 0:
        return []
    n = min(len(c) for c in curves)
    out = []
    for i in range(n):
        out.append(float(mean(c[i] for c in curves)))
    return out


def _convergence_step(curve: List[float], ratio: float = 0.9) -> float:
    if len(curve) == 0:
        return float("nan")
    start = float(curve[0])
    end = float(curve[-1])
    progress = start - end
    if progress <= 0.0:
        return float("nan")
    target = start - ratio * progress
    for i, v in enumerate(curve, start=1):
        if float(v) <= target:
            return float(i)
    return float(len(curve))


def _aggregate(rows: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        key = (row["objective"], row["mode"])
        groups.setdefault(key, []).append(row)

    out: Dict[Tuple[str, str], Dict] = {}
    for key, vals in groups.items():
        baseline_finals = [v["baseline_loss_final"] for v in vals]
        proxy_finals = [v["proxy_loss_final"] for v in vals]
        wild_outer_finals = [v["wild_outer_loss_final"] for v in vals if not math.isnan(v["wild_outer_loss_final"])]
        recovery = [v["recovery_ref_terminal"] for v in vals]
        generated_std = [v["generated_global_std"] for v in vals]
        denoise_gap = [v["robust_minus_baseline_denoise_mean"] for v in vals]
        wild_attack_loss = [v["wild_inner_attack_loss_final"] for v in vals if not math.isnan(v["wild_inner_attack_loss_final"])]
        wild_transport = [v["wild_transport_final"] for v in vals if not math.isnan(v["wild_transport_final"])]
        wild_delta = [v["wild_delta_norm_mean_final"] for v in vals if not math.isnan(v["wild_delta_norm_mean_final"])]
        baseline_curve = _trimmed_mean_curve([v["baseline_loss_curve"] for v in vals])
        wild_curve = _trimmed_mean_curve([v["wild_outer_loss_curve"] for v in vals])

        out[key] = {
            "num_runs": len(vals),
            "baseline_loss_final_mean": _safe_mean(baseline_finals),
            "baseline_loss_final_std": pstdev(baseline_finals) if len(vals) > 1 else 0.0,
            "proxy_loss_final_mean": _safe_mean(proxy_finals),
            "proxy_loss_final_std": pstdev(proxy_finals) if len(vals) > 1 else 0.0,
            "wild_outer_loss_final_mean": _safe_mean(wild_outer_finals),
            "recovery_ref_terminal_mean": _safe_mean(recovery),
            "generated_global_std_mean": _safe_mean(generated_std),
            "robust_minus_baseline_denoise_mean": _safe_mean(denoise_gap),
            "wild_inner_attack_loss_final_mean": _safe_mean(wild_attack_loss),
            "wild_transport_final_mean": _safe_mean(wild_transport),
            "wild_delta_norm_mean_final_mean": _safe_mean(wild_delta),
            "baseline_conv90_step": _convergence_step(baseline_curve, ratio=0.9),
            "wild_outer_conv90_step": _convergence_step(wild_curve, ratio=0.9),
        }
    return out


def _plot(
    rows: List[Dict],
    agg: Dict[Tuple[str, str], Dict],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    base_edm = _trimmed_mean_curve([r["baseline_loss_curve"] for r in rows if r["objective"] == "edm" and r["mode"] == "baseline"])
    base_score = _trimmed_mean_curve([r["baseline_loss_curve"] for r in rows if r["objective"] == "score" and r["mode"] == "baseline"])
    wild_edm = _trimmed_mean_curve([r["wild_outer_loss_curve"] for r in rows if r["objective"] == "edm" and r["mode"] == "wild"])
    wild_score = _trimmed_mean_curve([r["wild_outer_loss_curve"] for r in rows if r["objective"] == "score" and r["mode"] == "wild"])

    if base_edm:
        ax00.plot(range(1, len(base_edm) + 1), base_edm, label="baseline-edm", linewidth=2.0)
    if base_score:
        ax00.plot(range(1, len(base_score) + 1), base_score, label="baseline-score", linewidth=2.0)
    ax00.set_title("MNIST Baseline Convergence")
    ax00.set_xlabel("step")
    ax00.set_ylabel("primary objective")
    ax00.grid(alpha=0.25)
    ax00.legend()

    if wild_edm:
        ax01.plot(range(1, len(wild_edm) + 1), wild_edm, label="wild-edm", linewidth=2.0)
    if wild_score:
        ax01.plot(range(1, len(wild_score) + 1), wild_score, label="wild-score", linewidth=2.0)
    ax01.set_title("MNIST WILD Outer-Loss Convergence")
    ax01.set_xlabel("step")
    ax01.set_ylabel("outer objective")
    ax01.grid(alpha=0.25)
    ax01.legend()

    labels = ["edm-base", "edm-wild", "score-base", "score-wild"]
    keys = [("edm", "baseline"), ("edm", "wild"), ("score", "baseline"), ("score", "wild")]
    proxy = [agg[k]["proxy_loss_final_mean"] for k in keys]
    recovery = [agg[k]["recovery_ref_terminal_mean"] for k in keys]
    attack_delta = []
    for k in keys:
        v = agg[k]["wild_delta_norm_mean_final_mean"]
        attack_delta.append(0.0 if math.isnan(v) else float(v))

    x = list(range(len(labels)))
    colors = ["#2b8cbe", "#a6bddb", "#ef8a62", "#fddbc7"]

    ax10.bar(x, proxy, color=colors)
    ax10.set_xticks(x, labels, rotation=20, ha="right")
    ax10.set_title("Final Proxy Weighted Denoise Loss")
    ax10.set_ylabel("lower is better")
    ax10.grid(alpha=0.25, axis="y")

    ax11.bar(x, attack_delta, color=colors, label="wild delta-norm")
    ax11.plot(x, recovery, marker="o", color="#1b7837", label="recovery MSE")
    ax11.set_xticks(x, labels, rotation=20, ha="right")
    ax11.set_title("Attack Strength + Recovery")
    ax11.grid(alpha=0.25, axis="y")
    ax11.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline vs WILD on MNIST for EDM and score objectives.")
    parser.add_argument("--outdir", type=Path, default=Path("toy_outputs_mnist_wild_compare"))
    parser.add_argument("--prefix", type=str, default="mnist_wildcmp")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-steps-path", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--debug-eval-batch", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--image-train-size", type=int, default=5000)
    parser.add_argument("--image-val-size", type=int, default=2000)

    parser.add_argument("--wild-update-interval", type=int, default=20)
    parser.add_argument("--wild-cache-batches", type=int, default=4)
    parser.add_argument("--wild-inner-steps", type=int, default=3)
    parser.add_argument("--wild-step-size", type=float, default=0.05)
    parser.add_argument("--wild-gamma", type=float, default=2.0)
    parser.add_argument("--outer-clean-weight", type=float, default=1.0)
    parser.add_argument("--outer-attack-weight", type=float, default=0.5)
    parser.add_argument("--warmup-clean-steps", type=int, default=80)
    parser.add_argument("--warmup-ramp-steps", type=int, default=80)
    parser.add_argument("--wild-sample-min", type=float, default=-1.0)
    parser.add_argument("--wild-sample-max", type=float, default=1.0)

    parser.add_argument("--run-checks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    args.outdir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[2]
    run_toy = root / "toy" / "run_toy.py"

    common = [
        str(run_toy),
        "--method-version",
        "wild",
        "--dataset-kind",
        "mnist",
        "--model-kind",
        "image_conv",
        "--diagnostics-kind",
        "image_basic",
        "--image-channels",
        "1",
        "--image-size",
        str(args.image_size),
        "--image-train-size",
        str(args.image_train_size),
        "--image-val-size",
        str(args.image_val_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--n-steps-path",
        str(args.n_steps_path),
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--log-every",
        str(args.log_every),
        "--eval-samples",
        str(args.eval_samples),
        "--debug-eval-batch",
        str(args.debug_eval_batch),
        "--outdir",
        str(args.outdir),
        "--device",
        args.device,
    ]
    if args.run_checks:
        common.append("--run-checks")
    else:
        common.append("--skip-checks")

    for seed in seeds:
        for objective in ("edm", "score"):
            base_exp = f"{args.prefix}_{objective}_baseline_s{seed}"
            wild_exp = f"{args.prefix}_{objective}_wild_s{seed}"

            cmd_base = [
                sys.executable,
                *common,
                "--seed",
                str(seed),
                "--training-objective",
                objective,
                "--exp-name",
                base_exp,
                "--baseline-only",
            ]
            _run(cmd_base, dry_run=args.dry_run)

            cmd_wild = [
                sys.executable,
                *common,
                "--seed",
                str(seed),
                "--training-objective",
                objective,
                "--exp-name",
                wild_exp,
                "--disable-baseline-gate",
                "--wild-update-interval",
                str(args.wild_update_interval),
                "--wild-cache-batches",
                str(args.wild_cache_batches),
                "--wild-inner-steps",
                str(args.wild_inner_steps),
                "--wild-step-size",
                str(args.wild_step_size),
                "--wild-gamma",
                str(args.wild_gamma),
                "--outer-clean-weight",
                str(args.outer_clean_weight),
                "--outer-attack-weight",
                str(args.outer_attack_weight),
                "--warmup-clean-steps",
                str(args.warmup_clean_steps),
                "--warmup-ramp-steps",
                str(args.warmup_ramp_steps),
                "--wild-clamp-samples",
                "--wild-sample-min",
                str(args.wild_sample_min),
                "--wild-sample-max",
                str(args.wild_sample_max),
            ]
            _run(cmd_wild, dry_run=args.dry_run)

    if args.dry_run:
        print("[done] dry-run only")
        return 0

    rows = []
    for seed in seeds:
        for objective in ("edm", "score"):
            for mode in ("baseline", "wild"):
                exp = f"{args.prefix}_{objective}_{mode}_s{seed}"
                m = _load_metrics(args.outdir / exp / "metrics.json")
                od = m["objective_debug"]
                denoise = m["denoise_debug_heldout_pool"]
                sq = m["sample_quality_debug"]["robust_generated_metrics"]
                robust_curve = od.get("robust_outer_loss_curve", [])

                rows.append(
                    {
                        "seed": int(seed),
                        "objective": objective,
                        "mode": mode,
                        "exp_name": exp,
                        "baseline_loss_final": float(od["baseline_loss"]["final"]),
                        "proxy_loss_final": float(od["baseline_proxy_weighted_denoise_loss"]["final"]),
                        "wild_outer_loss_final": float(_safe_last(robust_curve)),
                        "recovery_ref_terminal": float(m["recovery_debug"]["baseline_x0_mse_from_ref_terminal"]),
                        "generated_global_std": float(sq["global_std"]),
                        "robust_batch_equiv_denoiser_evals_total": float(
                            od.get("robust_batch_equiv_denoiser_evals_cumulative", {}).get("final", 0.0) or 0.0
                        ),
                        "robust_minus_baseline_denoise_mean": float(
                            _safe_mean(denoise["robust_on_forward_baseline"])
                            - _safe_mean(denoise["baseline_on_forward_baseline"])
                        ),
                        "wild_inner_attack_loss_final": _summary_final(od.get("wild_inner_attack_loss", {})),
                        "wild_transport_final": _summary_final(od.get("wild_inner_transport_cost", {})),
                        "wild_delta_norm_mean_final": _summary_final(od.get("robust_delta_norm_mean", {})),
                        "attack_training_executed": bool(m["baseline_gate"]["attack_training_executed"]),
                        "baseline_loss_curve": [float(v) for v in od.get("baseline_loss_curve", [])],
                        "wild_outer_loss_curve": [float(v) for v in robust_curve],
                    }
                )

    agg = _aggregate(rows)
    summary = {
        "inputs": {
            "outdir": str(args.outdir),
            "prefix": args.prefix,
            "seeds": seeds,
            "device": args.device,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "wild_update_interval": args.wild_update_interval,
            "wild_cache_batches": args.wild_cache_batches,
            "wild_inner_steps": args.wild_inner_steps,
            "wild_step_size": args.wild_step_size,
            "wild_gamma": args.wild_gamma,
            "outer_clean_weight": args.outer_clean_weight,
            "outer_attack_weight": args.outer_attack_weight,
            "warmup_clean_steps": args.warmup_clean_steps,
            "warmup_ramp_steps": args.warmup_ramp_steps,
        },
        "per_run": rows,
        "aggregate": {f"{o}_{m}": v for (o, m), v in agg.items()},
    }

    summary_path = args.outdir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig_path = args.outdir / f"{args.prefix}_summary.png"
    _plot(rows, agg, fig_path)

    print(f"[saved] summary: {summary_path}")
    print(f"[saved] figure : {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
