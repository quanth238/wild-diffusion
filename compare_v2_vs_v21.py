#!/usr/bin/env python3
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List


def _parse_seeds(seeds_text: str) -> List[int]:
    out: List[int] = []
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


def _load_metrics(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" not in obj or "config" not in obj:
        raise ValueError(f"Invalid metrics payload: {path}")
    return obj


def _safe_float(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _safe_mean(xs: List[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return mean(xs) if xs else float("nan")


def _safe_std(xs: List[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) <= 1:
        return 0.0 if len(xs) == 1 else float("nan")
    return pstdev(xs)


def _mean_list(values: Any) -> float:
    if not isinstance(values, list) or len(values) == 0:
        return float("nan")
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if not math.isnan(v)]
    return mean(vals) if vals else float("nan")


def _extract_row(exp_name: str, seed: int, method_version: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    metrics = payload["metrics"]
    sample = metrics.get("sample_quality_debug", {})
    gate = metrics.get("baseline_gate", {})
    constraint = metrics.get("constraint_debug", {})
    recovery = metrics.get("recovery_debug", {})

    heldout = constraint.get("heldout_rollout_by_step", {})
    windows = constraint.get("attack_gap_windows_heldout", {})
    overall = windows.get("overall", {})
    high_noise = windows.get("high_noise", {})
    paired = recovery.get("paired_reverse_delta_under_plot_setting", {})

    baseline_fid = _safe_float(sample.get("baseline_fid"))
    robust_fid = _safe_float(sample.get("robust_fid"))
    fid_gain = baseline_fid - robust_fid if not math.isnan(baseline_fid) and not math.isnan(robust_fid) else float("nan")

    row = {
        "exp_name": exp_name,
        "seed": seed,
        "method_version": method_version,
        "training_objective": payload["config"].get("training_objective", "unknown"),
        "attack_training_executed": bool(gate.get("attack_training_executed", False)),
        "gate_passed": bool(gate.get("passed", False)),
        "baseline_fid": baseline_fid,
        "robust_fid": robust_fid,
        "fid_gain": fid_gain,
        "attack_gap_overall": _safe_float(overall.get("mean_attack_gap")),
        "attack_win_ratio_overall": _safe_float(overall.get("win_ratio")),
        "attack_gap_high_noise": _safe_float(high_noise.get("mean_attack_gap")),
        "delta_ratio_mean": _mean_list(heldout.get("delta_ratio_mean")),
        "frac_near_boundary_mean": _mean_list(heldout.get("frac_near_boundary")),
        "gap_norm_terminal": _safe_float((heldout.get("gap_norm_mean") or [None])[-1]),
        "recovery_mse_ref_terminal": _safe_float(recovery.get("baseline_x0_mse_from_ref_terminal")),
        "recovery_mse_attack_terminal": _safe_float(recovery.get("baseline_x0_mse_from_attack_terminal")),
        "paired_reverse_x0_l2": _safe_float(paired.get("x0_step_mean_l2")),
        "paired_reverse_terminal_l2": _safe_float(paired.get("terminal_step_mean_l2")),
    }
    return row


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    methods = sorted({r["method_version"] for r in rows})
    for method in methods:
        sub = [r for r in rows if r["method_version"] == method]
        out[method] = {
            "num_runs": len(sub),
            "attack_exec_rate": _safe_mean([1.0 if r["attack_training_executed"] else 0.0 for r in sub]),
            "gate_pass_rate": _safe_mean([1.0 if r["gate_passed"] else 0.0 for r in sub]),
            "baseline_fid_mean": _safe_mean([r["baseline_fid"] for r in sub]),
            "baseline_fid_std": _safe_std([r["baseline_fid"] for r in sub]),
            "robust_fid_mean": _safe_mean([r["robust_fid"] for r in sub]),
            "robust_fid_std": _safe_std([r["robust_fid"] for r in sub]),
            "fid_gain_mean": _safe_mean([r["fid_gain"] for r in sub]),
            "fid_gain_std": _safe_std([r["fid_gain"] for r in sub]),
            "attack_gap_overall_mean": _safe_mean([r["attack_gap_overall"] for r in sub]),
            "attack_gap_high_noise_mean": _safe_mean([r["attack_gap_high_noise"] for r in sub]),
            "attack_win_ratio_overall_mean": _safe_mean([r["attack_win_ratio_overall"] for r in sub]),
            "delta_ratio_mean": _safe_mean([r["delta_ratio_mean"] for r in sub]),
            "frac_near_boundary_mean": _safe_mean([r["frac_near_boundary_mean"] for r in sub]),
            "gap_norm_terminal_mean": _safe_mean([r["gap_norm_terminal"] for r in sub]),
            "recovery_mse_ref_terminal_mean": _safe_mean([r["recovery_mse_ref_terminal"] for r in sub]),
            "recovery_mse_attack_terminal_mean": _safe_mean([r["recovery_mse_attack_terminal"] for r in sub]),
            "paired_reverse_x0_l2_mean": _safe_mean([r["paired_reverse_x0_l2"] for r in sub]),
            "paired_reverse_terminal_l2_mean": _safe_mean([r["paired_reverse_terminal_l2"] for r in sub]),
        }
    return out


def _choose_winner(agg: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    m_v2 = agg.get("v2")
    m_v21 = agg.get("2.1") or agg.get("v2.1")
    if not m_v2 or not m_v21:
        return {
            "metric": "robust_fid_mean (lower is better)",
            "winner": "undetermined",
            "reason": "missing v2 or v2.1 aggregate stats",
        }

    v2_fid = _safe_float(m_v2.get("robust_fid_mean"))
    v21_fid = _safe_float(m_v21.get("robust_fid_mean"))
    if math.isnan(v2_fid) or math.isnan(v21_fid):
        return {
            "metric": "robust_fid_mean (lower is better)",
            "winner": "undetermined",
            "reason": "robust_fid unavailable (enable --compute-fid)",
        }

    tol = 1e-9
    if abs(v2_fid - v21_fid) <= tol:
        winner = "tie"
    else:
        winner = "v2" if v2_fid < v21_fid else "2.1"
    return {
        "metric": "robust_fid_mean (lower is better)",
        "winner": winner,
        "v2_robust_fid_mean": v2_fid,
        "v21_robust_fid_mean": v21_fid,
        "delta_v2_minus_v21": v2_fid - v21_fid,
    }


def _fmt(x: Any, nd: int = 4) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    if x is None:
        return "nan"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "nan"
    return f"{v:.{nd}f}"


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_markdown(path: Path, rows: List[Dict[str, Any]], agg: Dict[str, Dict[str, Any]], winner: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# v2 vs v2.1 Comparison")
    lines.append("")
    lines.append("## Per-run Metrics")
    lines.append("")
    lines.append("| method | seed | attack_exec | gate_pass | baseline_fid | robust_fid | fid_gain | attack_gap_overall | attack_gap_high | attack_win | delta_ratio_mean | near_boundary_mean | rec_mse_ref | rec_mse_attack |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(rows, key=lambda z: (str(z["method_version"]), int(z["seed"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["method_version"]),
                    str(r["seed"]),
                    _fmt(1.0 if r["attack_training_executed"] else 0.0, 0),
                    _fmt(1.0 if r["gate_passed"] else 0.0, 0),
                    _fmt(r["baseline_fid"]),
                    _fmt(r["robust_fid"]),
                    _fmt(r["fid_gain"]),
                    _fmt(r["attack_gap_overall"]),
                    _fmt(r["attack_gap_high_noise"]),
                    _fmt(r["attack_win_ratio_overall"]),
                    _fmt(r["delta_ratio_mean"]),
                    _fmt(r["frac_near_boundary_mean"]),
                    _fmt(r["recovery_mse_ref_terminal"]),
                    _fmt(r["recovery_mse_attack_terminal"]),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Aggregate (mean ± std)")
    lines.append("")
    lines.append("| method | runs | attack_exec_rate | gate_pass_rate | robust_fid | fid_gain | attack_gap_overall | attack_gap_high | attack_win | delta_ratio_mean | near_boundary_mean | rec_mse_ref | rec_mse_attack |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for method in sorted(agg.keys()):
        a = agg[method]
        robust_fid_text = f"{_fmt(a['robust_fid_mean'])} ± {_fmt(a['robust_fid_std'])}"
        fid_gain_text = f"{_fmt(a['fid_gain_mean'])} ± {_fmt(a['fid_gain_std'])}"
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(a["num_runs"]),
                    _fmt(a["attack_exec_rate"]),
                    _fmt(a["gate_pass_rate"]),
                    robust_fid_text,
                    fid_gain_text,
                    _fmt(a["attack_gap_overall_mean"]),
                    _fmt(a["attack_gap_high_noise_mean"]),
                    _fmt(a["attack_win_ratio_overall_mean"]),
                    _fmt(a["delta_ratio_mean"]),
                    _fmt(a["frac_near_boundary_mean"]),
                    _fmt(a["recovery_mse_ref_terminal_mean"]),
                    _fmt(a["recovery_mse_attack_terminal_mean"]),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Auto Conclusion")
    lines.append("")
    lines.append(f"- metric: `{winner.get('metric', 'n/a')}`")
    lines.append(f"- winner: `{winner.get('winner', 'undetermined')}`")
    if "reason" in winner:
        lines.append(f"- reason: {winner['reason']}")
    else:
        lines.append(f"- v2 robust_fid_mean: `{_fmt(winner.get('v2_robust_fid_mean'))}`")
        lines.append(f"- v2.1 robust_fid_mean: `{_fmt(winner.get('v21_robust_fid_mean'))}`")
        lines.append(f"- delta(v2 - v2.1): `{_fmt(winner.get('delta_v2_minus_v21'))}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and compare v2 vs v2.1 across multiple seeds.")
    parser.add_argument("--outdir", type=Path, default=Path("toy_outputs_v2_vs_v21"))
    parser.add_argument("--prefix", type=str, default="v2_vs_v21")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])

    parser.add_argument("--dataset-kind", type=str, default="mnist")
    parser.add_argument("--model-kind", type=str, default="image_conv")
    parser.add_argument("--diagnostics-kind", type=str, default="image_basic")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--image-train-size", type=int, default=2000)
    parser.add_argument("--image-val-size", type=int, default=500)
    parser.add_argument("--image-split-seed", type=int, default=0)
    parser.add_argument("--mnist-train-percent", type=float, default=20.0)
    parser.add_argument("--mnist-val-percent", type=float, default=100.0)
    parser.add_argument("--disable-mnist-percent-split", action="store_true")

    parser.add_argument("--training-objective", type=str, default="edm", choices=["edm", "score"])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-steps-path", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--debug-eval-batch", type=int, default=256)

    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--control-radius-kappa", type=float, default=0.15)
    parser.add_argument("--v21-rho", type=float, default=0.8)
    parser.add_argument("--eval-seed-offset-gate", type=int, default=10000)
    parser.add_argument("--eval-seed-offset-metrics", type=int, default=20000)
    parser.add_argument("--disable-eval-shared-terminal-noise", action="store_true")
    parser.add_argument("--disable-eval-shared-reverse-noise", action="store_true")
    parser.add_argument("--outer-clean-weight", type=float, default=1.0)
    parser.add_argument("--outer-attack-weight", type=float, default=0.5)
    parser.add_argument("--warmup-clean-steps", type=int, default=900)
    parser.add_argument("--warmup-ramp-steps", type=int, default=600)

    parser.add_argument("--compute-fid", action="store_true")
    parser.add_argument("--fid-samples", type=int, default=2000)
    parser.add_argument("--disable-baseline-gate", action="store_true")
    parser.add_argument("--run-checks", action="store_true")
    parser.add_argument(
        "--baseline-ckpt-dir",
        type=Path,
        default=None,
        help="Optional shared baseline checkpoint directory. Defaults to <outdir>/_baseline_cache_shared.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    args.outdir.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[2]
    run_toy = root / "toy" / "run_toy.py"

    common = [
        str(run_toy),
        "--dataset-kind",
        args.dataset_kind,
        "--model-kind",
        args.model_kind,
        "--diagnostics-kind",
        args.diagnostics_kind,
        "--image-channels",
        "1",
        "--image-size",
        str(args.image_size),
        "--image-train-size",
        str(args.image_train_size),
        "--image-val-size",
        str(args.image_val_size),
        "--image-split-seed",
        str(args.image_split_seed),
        "--mnist-train-percent",
        str(args.mnist_train_percent),
        "--mnist-val-percent",
        str(args.mnist_val_percent),
        "--training-objective",
        args.training_objective,
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--n-steps-path",
        str(args.n_steps_path),
        "--log-every",
        str(args.log_every),
        "--eval-samples",
        str(args.eval_samples),
        "--debug-eval-batch",
        str(args.debug_eval_batch),
        "--inner-steps",
        str(args.inner_steps),
        "--control-radius-kappa",
        str(args.control_radius_kappa),
        "--outer-clean-weight",
        str(args.outer_clean_weight),
        "--outer-attack-weight",
        str(args.outer_attack_weight),
        "--eval-seed-offset-gate",
        str(args.eval_seed_offset_gate),
        "--eval-seed-offset-metrics",
        str(args.eval_seed_offset_metrics),
        "--warmup-clean-steps",
        str(args.warmup_clean_steps),
        "--warmup-ramp-steps",
        str(args.warmup_ramp_steps),
        "--outdir",
        str(args.outdir),
        "--device",
        args.device,
    ]
    if args.compute_fid:
        common.extend(["--compute-fid", "--fid-samples", str(args.fid_samples)])
    if args.disable_baseline_gate:
        common.append("--disable-baseline-gate")
    if args.disable_mnist_percent_split:
        common.append("--disable-mnist-percent-split")
    if args.run_checks:
        common.append("--run-checks")
    else:
        common.append("--skip-checks")
    if args.disable_eval_shared_terminal_noise:
        common.append("--disable-eval-shared-terminal-noise")
    if args.disable_eval_shared_reverse_noise:
        common.append("--disable-eval-shared-reverse-noise")

    methods = ["v2", "2.1"]
    baseline_ckpt_dir = args.baseline_ckpt_dir or (args.outdir / "_baseline_cache_shared")
    baseline_ckpt_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        baseline_ckpt_path = baseline_ckpt_dir / f"baseline_seed_{seed}.pt"
        for method in methods:
            exp_name = f"{args.prefix}_{str(method).replace('.', '_')}_s{seed}"
            cmd = [
                sys.executable,
                *common,
                "--method-version",
                str(method),
                "--seed",
                str(seed),
                "--exp-name",
                exp_name,
                "--baseline-ckpt-path",
                str(baseline_ckpt_path),
            ]
            if method in ("2.1", "v2.1"):
                cmd.extend(["--v21-rho", str(args.v21_rho)])
            _run(cmd, dry_run=args.dry_run)

    if args.dry_run:
        print("[done] dry-run only", flush=True)
        return 0

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for method in methods:
            exp_name = f"{args.prefix}_{str(method).replace('.', '_')}_s{seed}"
            metrics_path = args.outdir / exp_name / "metrics.json"
            payload = _load_metrics(metrics_path)
            row = _extract_row(
                exp_name=exp_name,
                seed=seed,
                method_version=str(method),
                payload=payload,
            )
            rows.append(row)

    agg = _aggregate(rows)
    winner = _choose_winner(agg)
    summary = {
        "inputs": {
            "outdir": str(args.outdir),
            "prefix": args.prefix,
            "seeds": seeds,
            "methods": methods,
            "training_objective": args.training_objective,
            "compute_fid": bool(args.compute_fid),
            "fid_samples": int(args.fid_samples),
            "disable_baseline_gate": bool(args.disable_baseline_gate),
            "baseline_ckpt_dir": str(baseline_ckpt_dir),
            "v21_rho": float(args.v21_rho),
            "eval_seed_offset_gate": int(args.eval_seed_offset_gate),
            "eval_seed_offset_metrics": int(args.eval_seed_offset_metrics),
            "eval_shared_terminal_noise": bool(not args.disable_eval_shared_terminal_noise),
            "eval_shared_reverse_noise": bool(not args.disable_eval_shared_reverse_noise),
        },
        "rows": rows,
        "aggregate": agg,
        "winner": winner,
    }

    summary_json = args.outdir / f"{args.prefix}_summary_v2_vs_v21.json"
    summary_csv = args.outdir / f"{args.prefix}_summary_v2_vs_v21.csv"
    summary_md = args.outdir / f"{args.prefix}_summary_v2_vs_v21.md"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(summary_csv, rows)
    _write_markdown(summary_md, rows, agg, winner)

    print("[done] wrote:", summary_json, flush=True)
    print("[done] wrote:", summary_csv, flush=True)
    print("[done] wrote:", summary_md, flush=True)
    print("[result] auto winner:", winner, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
