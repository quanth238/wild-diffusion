#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


def _parse_seeds(seeds_text: str):
    out = []
    for s in seeds_text.split(","):
        s = s.strip()
        if s:
            out.append(int(s))
    if not out:
        raise ValueError("At least one seed is required.")
    return out


def _run(cmd, dry_run: bool = False):
    print("[run]", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _load_metrics(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in obj:
        raise ValueError(f"Invalid metrics payload: {path}")
    return obj["metrics"]


def _safe_last(xs):
    return xs[-1] if len(xs) > 0 else float("nan")


def _safe_mean(xs):
    return mean(xs) if len(xs) > 0 else float("nan")


def _aggregate_rows(rows):
    groups = {}
    for row in rows:
        key = (row["objective"], row["mode"])
        groups.setdefault(key, []).append(row)
    agg = {}
    for key, vals in groups.items():
        agg[key] = {
            "num_runs": len(vals),
            "proxy_final_mean": _safe_mean([v["proxy_final"] for v in vals]),
            "proxy_final_std": pstdev([v["proxy_final"] for v in vals]) if len(vals) > 1 else 0.0,
            "recovery_ref_terminal_mean": _safe_mean([v["recovery_ref_terminal"] for v in vals]),
            "generated_global_std_mean": _safe_mean([v["generated_global_std"] for v in vals]),
            "attack_win_ratio_mean": _safe_mean([v["attack_win_ratio"] for v in vals]),
            "attack_mean_gap_mean": _safe_mean([v["attack_mean_gap"] for v in vals]),
        }
    return agg


def _plot_summary(rows, agg, out_path: Path):
    labels = ["edm-baseline", "edm-attack", "score-baseline", "score-attack"]
    key_map = [
        ("edm", "baseline"),
        ("edm", "attack"),
        ("score", "baseline"),
        ("score", "attack"),
    ]
    proxy_mean = [agg[k]["proxy_final_mean"] for k in key_map]
    proxy_std = [agg[k]["proxy_final_std"] for k in key_map]
    recovery_mean = [agg[k]["recovery_ref_terminal_mean"] for k in key_map]
    sample_std_mean = [agg[k]["generated_global_std_mean"] for k in key_map]
    attack_win = [agg[k]["attack_win_ratio_mean"] for k in key_map]
    attack_gap = [agg[k]["attack_mean_gap_mean"] for k in key_map]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]
    x = list(range(len(labels)))

    ax00.bar(x, proxy_mean, yerr=proxy_std, color=["#2b8cbe", "#a6bddb", "#ef8a62", "#fddbc7"])
    ax00.set_xticks(x, labels, rotation=20, ha="right")
    ax00.set_title("Proxy Weighted Denoise Loss (final)")
    ax00.set_ylabel("Lower is better")
    ax00.grid(alpha=0.25, axis="y")

    ax01.bar(x, recovery_mean, color=["#2b8cbe", "#a6bddb", "#ef8a62", "#fddbc7"])
    ax01.set_xticks(x, labels, rotation=20, ha="right")
    ax01.set_title("Recovery MSE at Terminal (ref path)")
    ax01.set_ylabel("Lower is better")
    ax01.grid(alpha=0.25, axis="y")

    ax10.bar(x, sample_std_mean, color=["#2b8cbe", "#a6bddb", "#ef8a62", "#fddbc7"])
    ax10.set_xticks(x, labels, rotation=20, ha="right")
    ax10.set_title("Generated Global Std")
    ax10.set_ylabel("Higher usually better (avoid collapse)")
    ax10.grid(alpha=0.25, axis="y")

    ax11.axhline(0.0, color="black", linewidth=1.0)
    ax11.bar(x, attack_gap, color=["#2b8cbe", "#a6bddb", "#ef8a62", "#fddbc7"], label="attack_mean_gap")
    ax11.plot(x, attack_win, marker="o", color="#1b7837", label="attack_win_ratio")
    ax11.set_xticks(x, labels, rotation=20, ha="right")
    ax11.set_title("Attack Metrics (attack runs only)")
    ax11.grid(alpha=0.25, axis="y")
    ax11.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and plot EDM vs Score objective comparison on MNIST.")
    parser.add_argument("--outdir", type=Path, default=Path("toy_outputs_mnist_compare"))
    parser.add_argument("--prefix", type=str, default="mnist_objcmp")
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
    parser.add_argument("--kappa", type=float, default=0.15)
    parser.add_argument("--warmup-clean-steps", type=int, default=80)
    parser.add_argument("--warmup-ramp-steps", type=int, default=80)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--run-checks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    args.outdir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    run_toy = root / "toy" / "run_toy.py"

    common = [
        str(run_toy),
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
        "--method-version",
        "v2",
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
        "--inner-steps",
        str(args.inner_steps),
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
            atk_exp = f"{args.prefix}_{objective}_attack_s{seed}"
            cmd_base = [
                sys.executable,
                *common,
                "--training-objective",
                objective,
                "--seed",
                str(seed),
                "--exp-name",
                base_exp,
                "--baseline-only",
            ]
            _run(cmd_base, dry_run=args.dry_run)

            cmd_atk = [
                sys.executable,
                *common,
                "--training-objective",
                objective,
                "--seed",
                str(seed),
                "--exp-name",
                atk_exp,
                "--disable-baseline-gate",
                "--control-radius-kappa",
                str(args.kappa),
                "--warmup-clean-steps",
                str(args.warmup_clean_steps),
                "--warmup-ramp-steps",
                str(args.warmup_ramp_steps),
            ]
            _run(cmd_atk, dry_run=args.dry_run)

    if args.dry_run:
        print("[done] dry-run only")
        return 0

    rows = []
    for seed in seeds:
        for objective in ("edm", "score"):
            for mode in ("baseline", "attack"):
                exp = f"{args.prefix}_{objective}_{mode}_s{seed}"
                metrics_path = args.outdir / exp / "metrics.json"
                m = _load_metrics(metrics_path)
                od = m["objective_debug"]
                attack_overall = m["constraint_debug"]["attack_gap_windows_heldout"]["overall"]
                rows.append(
                    {
                        "seed": seed,
                        "objective": objective,
                        "mode": mode,
                        "exp_name": exp,
                        "proxy_final": float(od["baseline_proxy_weighted_denoise_loss"]["final"]),
                        "primary_final": float(od["baseline_loss"]["final"]),
                        "recovery_ref_terminal": float(m["recovery_debug"]["baseline_x0_mse_from_ref_terminal"]),
                        "generated_global_std": float(m["sample_quality_debug"]["baseline_generated_metrics"]["global_std"]),
                        "attack_win_ratio": float(attack_overall["win_ratio"]),
                        "attack_mean_gap": float(attack_overall["mean_attack_gap"]),
                        "attack_training_executed": bool(m["baseline_gate"]["attack_training_executed"]),
                    }
                )

    agg = _aggregate_rows(rows)
    summary = {
        "inputs": {
            "outdir": str(args.outdir),
            "prefix": args.prefix,
            "seeds": seeds,
            "device": args.device,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "kappa": args.kappa,
        },
        "per_run": rows,
        "aggregate": {
            f"{obj}_{mode}": val for (obj, mode), val in agg.items()
        },
    }
    summary_path = args.outdir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig_path = args.outdir / f"{args.prefix}_summary.png"
    _plot_summary(rows, agg, fig_path)

    print(f"[saved] summary: {summary_path}")
    print(f"[saved] figure : {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
