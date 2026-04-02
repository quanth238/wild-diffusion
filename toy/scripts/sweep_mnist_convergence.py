#!/usr/bin/env python3
"""From-scratch MNIST baseline convergence sweep with threshold extraction.

This script is meant to recreate the paper-style "FID vs images shown" curve
for MNIST subset fractions, using the toy MNIST backend in this repository.

Protocol:
- train baseline-only runs from scratch for each (train_percent, seed, step),
- compute FID for each run,
- aggregate median FID/loss across seeds,
- define the threshold step as the earliest point whose median FID is within
  `threshold_pct` of the best median FID for that subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from toy.export_mnist_fid_ref import build_mnist_fid_reference, default_mnist_fid_policy_name
else:
    from ..export_mnist_fid_ref import build_mnist_fid_reference, default_mnist_fid_policy_name

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_int_list(text: str, *, allow_zero: bool = False) -> List[int]:
    values: List[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        value = int(tok)
        if allow_zero:
            if value < 0:
                raise ValueError(f"Expected non-negative integer, got {value}")
        else:
            if value <= 0:
                raise ValueError(f"Expected positive integer, got {value}")
        values.append(value)
    if not values:
        raise ValueError("Empty list.")
    return sorted(set(values))


def _parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        value = float(tok)
        if value <= 0.0:
            raise ValueError(f"Expected positive float, got {value}")
        values.append(value)
    if not values:
        raise ValueError("Empty list.")
    return sorted(set(values))


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(value: float, ndigits: int = 4) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{ndigits}f}"


def _run(cmd: Sequence[str], *, cwd: Path, dry_run: bool) -> None:
    print("[run]", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


@dataclass
class RunRow:
    train_percent: float
    seed: int
    step: int
    images_shown_m: float
    exp_name: str
    metrics_path: str
    train_subset_size: int
    val_subset_size: int
    baseline_fid: float
    baseline_loss_final: float
    baseline_loss_mean_last: float
    runtime_total_sec: float
    runtime_baseline_train_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_percent": self.train_percent,
            "seed": self.seed,
            "step": self.step,
            "images_shown_m": self.images_shown_m,
            "exp_name": self.exp_name,
            "metrics_path": self.metrics_path,
            "train_subset_size": self.train_subset_size,
            "val_subset_size": self.val_subset_size,
            "baseline_fid": self.baseline_fid,
            "baseline_loss_final": self.baseline_loss_final,
            "baseline_loss_mean_last": self.baseline_loss_mean_last,
            "runtime_total_sec": self.runtime_total_sec,
            "runtime_baseline_train_sec": self.runtime_baseline_train_sec,
        }


@dataclass
class AggregateRow:
    train_percent: float
    step: int
    images_shown_m: float
    train_subset_size_median: float
    n_runs: int
    fid_median: float
    fid_mean: float
    fid_std: float
    loss_final_median: float
    loss_mean_last_median: float
    runtime_total_median_sec: float
    runtime_train_median_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_percent": self.train_percent,
            "step": self.step,
            "images_shown_m": self.images_shown_m,
            "train_subset_size_median": self.train_subset_size_median,
            "n_runs": self.n_runs,
            "fid_median": self.fid_median,
            "fid_mean": self.fid_mean,
            "fid_std": self.fid_std,
            "loss_final_median": self.loss_final_median,
            "loss_mean_last_median": self.loss_mean_last_median,
            "runtime_total_median_sec": self.runtime_total_median_sec,
            "runtime_train_median_sec": self.runtime_train_median_sec,
        }


def _load_row(metrics_path: Path, *, train_percent: float, seed: int, step: int, batch_size: int, exp_name: str) -> RunRow:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    sample = metrics.get("sample_quality_debug", {})
    objective = metrics.get("objective_debug", {})
    dataset_debug = metrics.get("dataset_debug", {})
    runtime = metrics.get("flow_debug", {}).get("runtime", {})

    return RunRow(
        train_percent=float(train_percent),
        seed=int(seed),
        step=int(step),
        images_shown_m=float(step * batch_size) / 1_000_000.0,
        exp_name=exp_name,
        metrics_path=str(metrics_path),
        train_subset_size=int(dataset_debug.get("train_subset_size_resolved") or 0),
        val_subset_size=int(dataset_debug.get("val_subset_size_resolved") or 0),
        baseline_fid=_safe_float(sample.get("baseline_fid")),
        baseline_loss_final=_safe_float(objective.get("baseline_loss", {}).get("final")),
        baseline_loss_mean_last=_safe_float(objective.get("baseline_loss", {}).get("mean_last")),
        runtime_total_sec=_safe_float(runtime.get("total")),
        runtime_baseline_train_sec=_safe_float(runtime.get("baseline_train_only", runtime.get("baseline_train"))),
    )


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows_list[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def _median(values: List[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(statistics.median(finite))


def _mean(values: List[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(statistics.fmean(finite))


def _std(values: List[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return float("nan")
    return float(statistics.stdev(finite))


def _aggregate(rows: List[RunRow]) -> List[AggregateRow]:
    buckets: Dict[tuple[float, int], List[RunRow]] = {}
    for row in rows:
        buckets.setdefault((row.train_percent, row.step), []).append(row)

    out: List[AggregateRow] = []
    for (train_percent, step), sub in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
        fid_values = [r.baseline_fid for r in sub]
        loss_final_values = [r.baseline_loss_final for r in sub]
        loss_mean_last_values = [r.baseline_loss_mean_last for r in sub]
        runtime_total_values = [r.runtime_total_sec for r in sub]
        runtime_train_values = [r.runtime_baseline_train_sec for r in sub]
        train_subset_sizes = [float(r.train_subset_size) for r in sub]
        out.append(
            AggregateRow(
                train_percent=float(train_percent),
                step=int(step),
                images_shown_m=sub[0].images_shown_m,
                train_subset_size_median=_median(train_subset_sizes),
                n_runs=len(sub),
                fid_median=_median(fid_values),
                fid_mean=_mean(fid_values),
                fid_std=_std(fid_values),
                loss_final_median=_median(loss_final_values),
                loss_mean_last_median=_median(loss_mean_last_values),
                runtime_total_median_sec=_median(runtime_total_values),
                runtime_train_median_sec=_median(runtime_train_values),
            )
        )
    return out


def _extract_thresholds(
    rows: List[AggregateRow],
    *,
    threshold_pct: float,
    overfit_pct: float,
    overfit_patience: int,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[float, List[AggregateRow]] = {}
    for row in rows:
        grouped.setdefault(row.train_percent, []).append(row)

    for train_percent, sub in grouped.items():
        sub = sorted(sub, key=lambda row: row.step)
        finite = [row for row in sub if math.isfinite(row.fid_median)]
        if not finite:
            out[str(train_percent)] = {
                "train_percent": float(train_percent),
                "threshold_pct": float(threshold_pct),
                "best_fid_median": None,
                "best_step": None,
                "best_images_shown_m": None,
                "threshold_fid_ceiling": None,
                "threshold_step": None,
                "threshold_images_shown_m": None,
                "threshold_loss_mean_last_median": None,
                "overfit_pct": float(overfit_pct),
                "overfit_step": None,
                "overfit_images_shown_m": None,
            }
            continue

        best_row = min(finite, key=lambda row: (row.fid_median, row.step))
        threshold_fid_ceiling = best_row.fid_median * (1.0 + threshold_pct / 100.0)
        threshold_row = next(
            (row for row in finite if row.fid_median <= threshold_fid_ceiling),
            best_row,
        )

        overfit_fid_floor = best_row.fid_median * (1.0 + overfit_pct / 100.0)
        overfit_row: Optional[AggregateRow] = None
        consecutive = 0
        start_row: Optional[AggregateRow] = None
        for row in finite:
            if row.step <= best_row.step:
                continue
            if row.fid_median > overfit_fid_floor:
                consecutive += 1
                if consecutive == 1:
                    start_row = row
                if consecutive >= overfit_patience:
                    overfit_row = start_row
                    break
            else:
                consecutive = 0
                start_row = None

        out[str(train_percent)] = {
            "train_percent": float(train_percent),
            "threshold_pct": float(threshold_pct),
            "best_fid_median": float(best_row.fid_median),
            "best_step": int(best_row.step),
            "best_images_shown_m": float(best_row.images_shown_m),
            "best_loss_mean_last_median": float(best_row.loss_mean_last_median),
            "threshold_fid_ceiling": float(threshold_fid_ceiling),
            "threshold_step": int(threshold_row.step),
            "threshold_images_shown_m": float(threshold_row.images_shown_m),
            "threshold_loss_mean_last_median": float(threshold_row.loss_mean_last_median),
            "overfit_pct": float(overfit_pct),
            "overfit_fid_floor": float(overfit_fid_floor),
            "overfit_step": None if overfit_row is None else int(overfit_row.step),
            "overfit_images_shown_m": None if overfit_row is None else float(overfit_row.images_shown_m),
        }
    return out


def _plot_curves(
    *,
    rows: List[AggregateRow],
    thresholds: Dict[str, Dict[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to render the convergence figure. "
            "Install it or rerun in an environment that has matplotlib."
        ) from exc

    grouped: Dict[float, List[AggregateRow]] = {}
    for row in rows:
        grouped.setdefault(row.train_percent, []).append(row)

    colors = {
        10.0: "#4d4d4d",
        20.0: "#e84a5f",
        50.0: "#2b6ee8",
        100.0: "#2ca25f",
    }
    markers = {
        10.0: "s",
        20.0: "o",
        50.0: "^",
        100.0: "v",
    }

    fig, (ax_fid, ax_loss) = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)

    for train_percent in sorted(grouped.keys()):
        sub = sorted(grouped[train_percent], key=lambda row: row.step)
        color = colors.get(train_percent, None)
        marker = markers.get(train_percent, "o")
        x = [row.images_shown_m for row in sub]
        y_fid = [row.fid_median for row in sub]
        y_loss = [row.loss_mean_last_median for row in sub]
        label = f"MNIST ({train_percent:g}%)"

        ax_fid.plot(x, y_fid, color=color, marker=marker, linewidth=2.0, markersize=6.5, label=label)
        ax_loss.plot(x, y_loss, color=color, marker=marker, linewidth=2.0, markersize=6.5, label=label)

        summary = thresholds.get(str(train_percent))
        if summary is not None and summary.get("threshold_images_shown_m") is not None:
            x_thr = float(summary["threshold_images_shown_m"])
            ax_fid.axvline(x_thr, color=color, linestyle="--", alpha=0.18, linewidth=1.3)

    ax_fid.set_title("MNIST Convergence (FID)")
    ax_fid.set_xlabel("Number of images shown to model (M)")
    ax_fid.set_ylabel("FID")
    ax_fid.grid(alpha=0.25)
    ax_fid.legend(loc="best", fontsize=9)

    ax_loss.set_title("MNIST Training Loss")
    ax_loss.set_xlabel("Number of images shown to model (M)")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.25)
    ax_loss.legend(loc="best", fontsize=9)

    fig.suptitle(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recreate paper-style MNIST baseline convergence curves and extract FID thresholds."
    )
    parser.add_argument("--outdir", type=Path, default=Path("toy_outputs/mnist_convergence"))
    parser.add_argument("--prefix", type=str, default="mnist_convergence")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--train-percents", type=str, default="10,20,50,100")
    parser.add_argument("--steps-list", type=str, default="1000,2000,4000,8000,12000,16000,20000")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--image-channels", type=int, default=1)
    parser.add_argument("--mnist-val-percent", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--training-objective", type=str, default="edm", choices=["edm", "score"])
    parser.add_argument("--n-steps-path", type=int, default=24)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--auto-log-normal-params", action="store_true")
    parser.add_argument("--use-ema-eval", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--fid-samples", type=int, default=2000)
    parser.add_argument("--fid-ref-split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--fid-ref-subset-percent", type=float, default=100.0)
    parser.add_argument(
        "--fid-ref-subset-sampling",
        type=str,
        default="stratified",
        choices=["first", "global", "stratified"],
    )
    parser.add_argument("--fid-ref-subset-seed", type=int, default=0)
    parser.add_argument("--fid-ref-max-images", type=int, default=5000)
    parser.add_argument("--image-split-seed-offset", type=int, default=0)
    parser.add_argument("--force-ref-refresh", action="store_true")

    parser.add_argument("--threshold-pct", type=float, default=5.0)
    parser.add_argument("--overfit-pct", type=float, default=10.0)
    parser.add_argument("--overfit-patience", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = _parse_int_list(args.seeds, allow_zero=True)
    train_percents = _parse_float_list(args.train_percents)
    steps = _parse_int_list(args.steps_list)
    repo_root = _repo_root()
    run_toy = repo_root / "toy" / "run_toy.py"

    if args.require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("[ERROR] --require-cuda is set but CUDA is unavailable.")

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    fid_ref_policy_name = default_mnist_fid_policy_name(
        split=str(args.fid_ref_split),
        image_size=int(args.image_size),
        subset_percent=float(args.fid_ref_subset_percent),
        subset_sampling=str(args.fid_ref_subset_sampling),
        subset_seed=int(args.fid_ref_subset_seed),
        max_images=int(args.fid_ref_max_images),
    )
    fid_ref_dir = (outdir / "_fid_refs").resolve()
    ref_npz = fid_ref_dir / f"{fid_ref_policy_name}.npz"
    if args.dry_run:
        fid_ref_meta: Dict[str, Any] = {
            "policy_name": fid_ref_policy_name,
            "split": str(args.fid_ref_split),
            "image_size": int(args.image_size),
            "subset_percent_requested": float(args.fid_ref_subset_percent),
            "subset_sampling": str(args.fid_ref_subset_sampling),
            "subset_seed": int(args.fid_ref_subset_seed),
            "max_images_requested": int(args.fid_ref_max_images),
            "dest": str(ref_npz),
        }
        print(f"[dry-run] would ensure MNIST FID reference: {ref_npz}", flush=True)
    else:
        fid_ref_meta = build_mnist_fid_reference(
            split=str(args.fid_ref_split),
            image_size=int(args.image_size),
            subset_percent=float(args.fid_ref_subset_percent),
            subset_sampling=str(args.fid_ref_subset_sampling),
            subset_seed=int(args.fid_ref_subset_seed),
            max_images=int(args.fid_ref_max_images),
            dest=ref_npz,
            images_dir=fid_ref_dir / f"{fid_ref_policy_name}_images",
            policy_name=fid_ref_policy_name,
            force=bool(args.force_ref_refresh),
        )

    rows: List[RunRow] = []
    for train_percent in train_percents:
        for seed in seeds:
            image_split_seed = int(seed + args.image_split_seed_offset)
            for step in steps:
                exp_name = f"{args.prefix}_{train_percent:g}pct_s{seed}_st{step}"
                exp_dir = outdir / exp_name
                metrics_path = exp_dir / "metrics.json"

                if args.skip_existing and metrics_path.is_file():
                    print(f"[skip-existing] {metrics_path}", flush=True)
                else:
                    cmd = [
                        sys.executable,
                        str(run_toy),
                        "--dataset-kind",
                        "mnist",
                        "--model-kind",
                        "image_conv",
                        "--diagnostics-kind",
                        "image_basic",
                        "--image-channels",
                        str(args.image_channels),
                        "--image-size",
                        str(args.image_size),
                        "--mnist-train-percent",
                        str(train_percent),
                        "--mnist-val-percent",
                        str(args.mnist_val_percent),
                        "--image-split-seed",
                        str(image_split_seed),
                        "--device",
                        str(args.device),
                        "--training-objective",
                        str(args.training_objective),
                        "--steps",
                        str(step),
                        "--batch-size",
                        str(args.batch_size),
                        "--hidden-dim",
                        str(args.hidden_dim),
                        "--n-steps-path",
                        str(args.n_steps_path),
                        "--sigma-min",
                        str(args.sigma_min),
                        "--sigma-max",
                        str(args.sigma_max),
                        "--eval-samples",
                        str(args.eval_samples),
                        "--compute-fid",
                        "--fid-samples",
                        str(args.fid_samples),
                        "--fid-ref-path",
                        str(ref_npz),
                        "--fid-ref-policy",
                        fid_ref_policy_name,
                        "--skip-checks",
                        "--disable-baseline-gate",
                        "--baseline-only",
                        "--disable-baseline-ckpt",
                        "--seed",
                        str(seed),
                        "--outdir",
                        str(outdir),
                        "--exp-name",
                        exp_name,
                    ]
                    if args.auto_log_normal_params:
                        cmd.append("--auto-log-normal-params")
                    if args.use_ema_eval:
                        cmd.extend(["--use-ema-eval", "--ema-decay", str(args.ema_decay)])
                    _run(cmd, cwd=repo_root, dry_run=args.dry_run)

                if args.dry_run:
                    continue
                if not metrics_path.is_file():
                    raise RuntimeError(f"Missing metrics file: {metrics_path}")
                row = _load_row(
                    metrics_path,
                    train_percent=train_percent,
                    seed=seed,
                    step=step,
                    batch_size=args.batch_size,
                    exp_name=exp_name,
                )
                rows.append(row)
                print(
                    "[row]"
                    f" pct={train_percent:g} seed={seed} step={step}"
                    f" mimg={_fmt(row.images_shown_m, 3)}"
                    f" fid={_fmt(row.baseline_fid, 3)}"
                    f" loss={_fmt(row.baseline_loss_final, 5)}"
                    f" runtime={_fmt(row.runtime_total_sec, 1)}s",
                    flush=True,
                )

    if args.dry_run:
        print("[done] dry-run completed.", flush=True)
        return

    rows.sort(key=lambda row: (row.train_percent, row.seed, row.step))
    aggregates = _aggregate(rows)
    thresholds = _extract_thresholds(
        aggregates,
        threshold_pct=float(args.threshold_pct),
        overfit_pct=float(args.overfit_pct),
        overfit_patience=int(args.overfit_patience),
    )

    runs_csv = outdir / f"{args.prefix}_runs.csv"
    agg_csv = outdir / f"{args.prefix}_aggregate.csv"
    fig_path = outdir / f"{args.prefix}_curves.png"
    summary_path = outdir / f"{args.prefix}_summary.json"

    _write_csv(runs_csv, (row.to_dict() for row in rows))
    _write_csv(agg_csv, (row.to_dict() for row in aggregates))
    plot_warning = None
    plot_written = False
    if not args.skip_plot:
        try:
            _plot_curves(
                rows=aggregates,
                thresholds=thresholds,
                out_path=fig_path,
                title=(
                    "MNIST Baseline Convergence by Train Fraction"
                    f" | threshold={args.threshold_pct:.1f}% of best median FID"
                ),
            )
            plot_written = True
        except RuntimeError as exc:
            plot_warning = str(exc)
            print(f"[warn] {plot_warning}", flush=True)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "mnist_from_scratch_baseline_convergence",
            "description": (
                "Each point is an independent from-scratch baseline-only MNIST run "
                "evaluated at a fixed total step count."
            ),
            "family": "from_scratch_curve",
            "seeds": seeds,
            "train_percents": train_percents,
            "steps_list": steps,
            "batch_size": int(args.batch_size),
            "image_size": int(args.image_size),
            "training_objective": str(args.training_objective),
            "sigma_min": float(args.sigma_min),
            "sigma_max": float(args.sigma_max),
            "auto_log_normal_params": bool(args.auto_log_normal_params),
            "use_ema_eval": bool(args.use_ema_eval),
            "ema_decay": float(args.ema_decay),
            "force_ref_refresh": bool(args.force_ref_refresh),
            "fid_reference": fid_ref_meta,
            "threshold_pct": float(args.threshold_pct),
            "overfit_pct": float(args.overfit_pct),
            "overfit_patience": int(args.overfit_patience),
        },
        "thresholds": thresholds,
        "artifacts": {
            "runs_csv": str(runs_csv),
            "aggregate_csv": str(agg_csv),
            "curves_png": str(fig_path) if plot_written else None,
        },
        "plot": {
            "requested": bool(not args.skip_plot),
            "written": bool(plot_written),
            "warning": plot_warning,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[thresholds]")
    for train_percent in train_percents:
        info = thresholds[str(train_percent)]
        print(
            f"  {train_percent:g}%:"
            f" best_fid={_fmt(_safe_float(info['best_fid_median']), 3)}"
            f" @ step={info['best_step']} ({_fmt(_safe_float(info['best_images_shown_m']), 3)}M)"
            f" | threshold={_fmt(_safe_float(info['threshold_fid_ceiling']), 3)}"
            f" -> step={info['threshold_step']} ({_fmt(_safe_float(info['threshold_images_shown_m']), 3)}M)",
            flush=True,
        )

    print("[done] saved:")
    print(f"  - {runs_csv}")
    print(f"  - {agg_csv}")
    if plot_written:
        print(f"  - {fig_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
