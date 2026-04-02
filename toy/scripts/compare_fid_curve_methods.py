#!/usr/bin/env python3
"""Sweep continuation steps and plot FID-vs-iteration for multiple methods.

This script compares:
- baseline-only
- clean continuation
- wild
- v1.1
- v1.2

under a fixed baseline checkpoint initialization.

Protocol note:
- For each target robust step count S, the script creates a checkpoint alias whose
  signature uses baseline_steps=S (metadata only), while preserving the same
  baseline weights. This keeps strict checkpoint metadata checks satisfied and
  ensures all methods at the same S start from the same baseline weights.
- Legacy MNIST sweep checkpoints (`state_dict` / `history`) are accepted as
  baseline sources and converted on the fly into alias checkpoints that the
  comparison harness can load.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from toy.export_mnist_fid_ref import build_mnist_fid_reference, default_mnist_fid_policy_name
else:
    from ..export_mnist_fid_ref import build_mnist_fid_reference, default_mnist_fid_policy_name


def _repo_root() -> Path:
    # .../toy/scripts/compare_fid_curve_methods.py -> repo root
    return Path(__file__).resolve().parents[2]


def _parse_steps(text: str) -> List[int]:
    values: List[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        step = int(tok)
        if step <= 0:
            raise ValueError(f"All steps must be > 0, got {step}")
        values.append(step)
    if not values:
        raise ValueError("Empty --steps-list.")
    return sorted(set(values))


def _normalize_method(name: str) -> str:
    key = name.strip().lower().replace("_", "").replace("-", "")
    if key in ("baseline", "baselineonly", "base"):
        return "baseline"
    if key in ("clean", "cleancont", "cleancontinuation", "continuation", "finetune", "cleanfinetune"):
        return "clean"
    if key in ("wild",):
        return "wild"
    if key in ("11", "v11", "v1.1", "1.1"):
        return "1.1"
    if key in ("12", "v12", "v1.2", "1.2"):
        return "1.2"
    raise ValueError(f"Unsupported method alias: {name}")


def _parse_methods(text: str) -> List[str]:
    methods = [_normalize_method(tok) for tok in text.split(",") if tok.strip()]
    if not methods:
        raise ValueError("Empty --methods.")
    order = ["baseline", "clean", "wild", "1.1", "1.2"]
    seen = set()
    out: List[str] = []
    for m in methods:
        if m not in seen:
            out.append(m)
            seen.add(m)
    out.sort(key=lambda x: order.index(x) if x in order else 999)
    return out


def _safe_float(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(v: float, nd: int = 4) -> str:
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{nd}f}"


def _run(cmd: List[str], *, cwd: Path, dry_run: bool) -> None:
    print("[run]", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _empty_baseline_history() -> Dict[str, Any]:
    return {"loss": [], "proxy_weighted_denoise_loss": [], "sigma_counts": []}


def _baseline_ckpt_source_kind(source_ckpt: Path) -> str:
    payload = torch.load(str(source_ckpt), map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload (expect dict): {source_ckpt}")
    if "baseline_state_dict" in payload:
        return "native"
    if "state_dict" in payload:
        return "legacy_state_dict"
    raise RuntimeError(
        "Unsupported checkpoint payload. Expected one of "
        "('baseline_state_dict', 'state_dict') in "
        f"{source_ckpt}"
    )


def _ensure_baseline_alias_ckpt(*, source_ckpt: Path, alias_ckpt: Path, step: int, dry_run: bool) -> None:
    alias_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[dry-run] would create alias ckpt: {alias_ckpt} (baseline_steps={step})", flush=True)
        return

    payload = torch.load(str(source_ckpt), map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint payload (expect dict): {source_ckpt}")
    source_kind = _baseline_ckpt_source_kind(source_ckpt)

    if source_kind == "native":
        state_dict = payload["baseline_state_dict"]
        history = payload.get("baseline_history", _empty_baseline_history())
        sig = payload.get("baseline_signature", {})
        if not isinstance(sig, dict):
            sig = {}
        sig = dict(sig)
    else:
        state_dict = payload["state_dict"]
        history = payload.get("history", _empty_baseline_history())
        sig = {}

    sig["baseline_steps"] = int(step)
    if source_kind != "native":
        sig["legacy_source_format"] = str(payload.get("format", "legacy_state_dict"))

    alias_payload = {
        "format": "toy_baseline_ckpt_alias_v1",
        "baseline_signature": sig,
        "baseline_history": history if isinstance(history, dict) else _empty_baseline_history(),
        "baseline_state_dict": state_dict,
        "alias_from_ckpt": str(source_ckpt),
        "alias_for_baseline_steps": int(step),
        "alias_saved_at": datetime.now(timezone.utc).isoformat(),
    }

    tmp = alias_ckpt.with_suffix(alias_ckpt.suffix + f".tmp.{datetime.now().timestamp()}")
    torch.save(alias_payload, str(tmp))
    tmp.replace(alias_ckpt)


def _method_label(method: str) -> str:
    return {
        "baseline": "Baseline-only",
        "clean": "Clean-cont.",
        "wild": "WILD",
        "1.1": "v1.1",
        "1.2": "v1.2",
    }[method]


def _method_cli_args(method: str, args: argparse.Namespace) -> List[str]:
    if method == "baseline":
        return ["--method-version", "v2", "--baseline-only"]
    if method == "clean":
        return [
            "--method-version",
            "v2",
            "--inner-steps",
            "0",
            "--outer-attack-weight",
            "0.0",
            "--outer-clean-weight",
            "1.0",
            "--control-radius-kappa",
            "0.0",
        ]
    if method == "wild":
        extra = [
            "--method-version",
            "wild",
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
            "--wild-sample-min",
            str(args.wild_sample_min),
            "--wild-sample-max",
            str(args.wild_sample_max),
            "--wild-delta-ratio-denom",
            str(args.wild_delta_ratio_denom),
        ]
        if args.wild_fixed_noise_inner:
            extra.append("--wild-fixed-noise-inner")
        else:
            extra.append("--disable-wild-fixed-noise-inner")
        if args.wild_clamp_samples:
            extra.append("--wild-clamp-samples")
        return extra
    if method == "1.1":
        return [
            "--method-version",
            "1.1",
            "--v11-step-size",
            str(args.v11_step_size),
            "--v11-transport-gamma",
            str(args.v11_transport_gamma),
            "--v11-total-budget-rho",
            str(args.v11_total_budget_rho),
            "--v11-projection-mode",
            str(args.v11_projection_mode),
        ]
    if method == "1.2":
        return [
            "--method-version",
            "1.2",
            "--inner-steps",
            str(args.v12_adv_steps),
            "--v12-step-size",
            str(args.v12_step_size),
            "--v12-lambda-init",
            str(args.v12_lambda_init),
            "--v12-lambda-lr",
            str(args.v12_lambda_lr),
            "--v12-rho-target",
            str(args.v12_rho_target),
            "--v12-robust-mix",
            str(args.v12_robust_mix),
            "--v12-start-step",
            str(args.v12_start_step),
            "--v12-ramp-steps",
            str(args.v12_ramp_steps),
            "--v12-max-delta",
            str(args.v12_max_delta),
            "--v12-sigma-floor",
            str(args.v12_sigma_floor),
            "--v12-sigma-cut",
            str(args.v12_sigma_cut),
            "--v12-gate-power",
            str(args.v12_gate_power),
            "--v12-delta-space",
            str(args.v12_delta_space),
        ]
    raise ValueError(f"Unsupported method: {method}")


@dataclass
class RunRow:
    method: str
    method_label: str
    step: int
    exp_name: str
    metrics_path: str
    baseline_fid: float
    robust_fid: float
    fid_delta: float
    runtime_total_sec: float
    runtime_total_without_fid_sec: float
    runtime_robust_phase_sec: float
    baseline_ckpt_loaded: bool
    baseline_ckpt_signature_hash: str
    baseline_ckpt_path: str
    attack_training_executed: bool
    robust_images_seen_total: float
    effective_train_images_seen_total: float
    robust_batch_equiv_denoiser_evals_total: float
    method_transport_cost_name: str
    method_transport_cost_final: float
    fid_ref_policy_name: str
    fid_ref_resolved_path: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "method_label": self.method_label,
            "step": self.step,
            "exp_name": self.exp_name,
            "metrics_path": self.metrics_path,
            "baseline_fid": self.baseline_fid,
            "robust_fid": self.robust_fid,
            "fid_delta": self.fid_delta,
            "runtime_total_sec": self.runtime_total_sec,
            "runtime_total_without_fid_sec": self.runtime_total_without_fid_sec,
            "runtime_robust_phase_sec": self.runtime_robust_phase_sec,
            "baseline_ckpt_loaded": self.baseline_ckpt_loaded,
            "baseline_ckpt_signature_hash": self.baseline_ckpt_signature_hash,
            "baseline_ckpt_path": self.baseline_ckpt_path,
            "attack_training_executed": self.attack_training_executed,
            "robust_images_seen_total": self.robust_images_seen_total,
            "effective_train_images_seen_total": self.effective_train_images_seen_total,
            "robust_batch_equiv_denoiser_evals_total": self.robust_batch_equiv_denoiser_evals_total,
            "method_transport_cost_name": self.method_transport_cost_name,
            "method_transport_cost_final": self.method_transport_cost_final,
            "fid_ref_policy_name": self.fid_ref_policy_name,
            "fid_ref_resolved_path": self.fid_ref_resolved_path,
        }


def _load_row(metrics_path: Path, method: str, step: int, exp_name: str) -> RunRow:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    sq = metrics.get("sample_quality_debug", {})
    fid_ref = sq.get("fid_reference", {})
    flow = metrics.get("flow_debug", {})
    runtime = flow.get("runtime", {})
    gate = metrics.get("baseline_gate", {})
    objective = metrics.get("objective_debug", {})

    baseline_fid = _safe_float(sq.get("baseline_fid"))
    robust_fid = _safe_float(sq.get("robust_fid"))
    fid_delta = robust_fid - baseline_fid if math.isfinite(robust_fid) and math.isfinite(baseline_fid) else float("nan")
    if method == "wild":
        transport_name = "wild_sample_transport_cost"
        transport_payload = objective.get("wild_sample_transport_cost", objective.get("wild_inner_transport_cost", {}))
        transport_final = _safe_float(transport_payload.get("final") if isinstance(transport_payload, dict) else None)
    elif method == "1.1":
        transport_name = "v11_path_transport_cost"
        transport_payload = objective.get("v11_path_transport_cost", objective.get("robust_energy", {}))
        transport_final = _safe_float(transport_payload.get("final") if isinstance(transport_payload, dict) else None)
    else:
        transport_name = ""
        transport_final = float("nan")

    return RunRow(
        method=method,
        method_label=_method_label(method),
        step=int(step),
        exp_name=exp_name,
        metrics_path=str(metrics_path),
        baseline_fid=baseline_fid,
        robust_fid=robust_fid,
        fid_delta=fid_delta,
        runtime_total_sec=_safe_float(runtime.get("total", flow.get("runtime_total_sec"))),
        runtime_total_without_fid_sec=_safe_float(runtime.get("total_without_fid")),
        runtime_robust_phase_sec=_safe_float(runtime.get("robust_phase")),
        baseline_ckpt_loaded=bool(flow.get("baseline_ckpt_loaded", False)),
        baseline_ckpt_signature_hash=str(flow.get("baseline_ckpt_signature_hash", "")),
        baseline_ckpt_path=str(flow.get("baseline_ckpt_path", "")),
        attack_training_executed=bool(gate.get("attack_training_executed", False)),
        robust_images_seen_total=_safe_float(runtime.get("robust_images_seen_total")),
        effective_train_images_seen_total=_safe_float(runtime.get("effective_train_images_seen_total")),
        robust_batch_equiv_denoiser_evals_total=_safe_float(runtime.get("robust_batch_equiv_denoiser_evals_total")),
        method_transport_cost_name=transport_name,
        method_transport_cost_final=transport_final,
        fid_ref_policy_name=str(fid_ref.get("policy_name", "")),
        fid_ref_resolved_path=str(fid_ref.get("resolved_path", "")),
    )


def _write_csv(path: Path, rows: Iterable[RunRow]) -> None:
    rows_list = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows_list[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


def _group_rows(rows: List[RunRow]) -> Dict[str, List[RunRow]]:
    out: Dict[str, List[RunRow]] = {}
    for row in rows:
        out.setdefault(row.method, []).append(row)
    for method in out:
        out[method] = sorted(out[method], key=lambda r: r.step)
    return out


def _budget_value(row: RunRow, budget_field: str) -> float:
    return _safe_float(getattr(row, budget_field))


def _budget_label(budget_field: str) -> str:
    return {
        "robust_batch_equiv_denoiser_evals_total": "Robust Batch-Equivalent Denoiser Evals",
        "robust_images_seen_total": "Robust Images Seen",
    }[budget_field]


def _budget_summary_rows(rows: List[RunRow], methods: List[str], budget_field: str) -> List[Dict[str, Any]]:
    grouped = _group_rows(rows)
    budgets = sorted(
        {
            float(v)
            for row in rows
            for v in [_budget_value(row, budget_field)]
            if math.isfinite(v)
        }
    )
    summary_rows: List[Dict[str, Any]] = []
    for method in methods:
        sub = sorted(
            grouped.get(method, []),
            key=lambda r: (_budget_value(r, budget_field), r.step),
        )
        for budget_limit in budgets:
            eligible = [
                row
                for row in sub
                if math.isfinite(_budget_value(row, budget_field)) and _budget_value(row, budget_field) <= budget_limit
            ]
            if not eligible:
                continue
            eligible_finite = [row for row in eligible if math.isfinite(row.robust_fid)]
            best_row = min(
                eligible_finite,
                key=lambda r: (r.robust_fid, _budget_value(r, budget_field), r.step),
            ) if eligible_finite else None
            final_row = max(eligible, key=lambda r: (_budget_value(r, budget_field), r.step))
            summary_rows.append(
                {
                    "budget_metric": budget_field,
                    "budget_metric_label": _budget_label(budget_field),
                    "method": method,
                    "method_label": _method_label(method),
                    "budget_limit": float(budget_limit),
                    "best_within_budget_fid": None if best_row is None else float(best_row.robust_fid),
                    "best_within_budget_step": None if best_row is None else int(best_row.step),
                    "best_within_budget_budget": None if best_row is None else float(_budget_value(best_row, budget_field)),
                    "final_at_budget_fid": float(final_row.robust_fid) if math.isfinite(final_row.robust_fid) else None,
                    "final_at_budget_step": int(final_row.step),
                    "final_at_budget_budget": float(_budget_value(final_row, budget_field)),
                }
            )
    return summary_rows


def _plot_panel(rows: List[RunRow], methods: List[str], out_path: Path) -> None:
    grouped = _group_rows(rows)
    ncols = 2
    nrows = max(1, math.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4.5 * nrows, 5.0)), constrained_layout=True)
    axes_flat = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]

    for ax_idx, method in enumerate(methods):
        ax = axes_flat[ax_idx]
        sub = grouped.get(method, [])
        if not sub:
            ax.set_title(f"{_method_label(method)} (no data)")
            ax.axis("off")
            continue
        x = [r.step for r in sub]
        y_rob = [r.robust_fid for r in sub]
        y_base = [r.baseline_fid for r in sub]
        ax.plot(x, y_rob, marker="o", linewidth=2.0, label="robust_fid")
        ax.plot(x, y_base, linestyle="--", linewidth=1.8, label="baseline_fid")
        ax.grid(alpha=0.25)
        ax.set_xlabel("Robust Iterations")
        ax.set_ylabel("FID (lower is better)")
        finite_rob = [v for v in y_rob if math.isfinite(v)]
        if finite_rob:
            best = min(finite_rob)
            best_step = x[y_rob.index(best)]
            ax.set_title(f"{_method_label(method)} | best={best:.3f} @ {best_step}")
        else:
            ax.set_title(_method_label(method))
        ax.legend(loc="best", fontsize=8)

    for idx in range(len(methods), len(axes_flat)):
        axes_flat[idx].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_budget_panel(rows: List[RunRow], methods: List[str], out_path: Path, budget_field: str) -> None:
    grouped = _group_rows(rows)
    ncols = 2
    nrows = max(1, math.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4.5 * nrows, 5.0)), constrained_layout=True)
    axes_flat = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]
    x_label = _budget_label(budget_field)

    for ax_idx, method in enumerate(methods):
        ax = axes_flat[ax_idx]
        sub = [
            row for row in grouped.get(method, []) if math.isfinite(_budget_value(row, budget_field))
        ]
        sub = sorted(sub, key=lambda r: (_budget_value(r, budget_field), r.step))
        if not sub:
            ax.set_title(f"{_method_label(method)} (no data)")
            ax.axis("off")
            continue
        x = [_budget_value(r, budget_field) for r in sub]
        y_rob = [r.robust_fid for r in sub]
        y_base = [r.baseline_fid for r in sub]
        ax.plot(x, y_rob, marker="o", linewidth=2.0, label="robust_fid")
        ax.plot(x, y_base, linestyle="--", linewidth=1.8, label="baseline_fid")
        ax.grid(alpha=0.25)
        ax.set_xlabel(x_label)
        ax.set_ylabel("FID (lower is better)")
        finite_pairs = [(xv, yv, row.step) for xv, yv, row in zip(x, y_rob, sub) if math.isfinite(yv)]
        if finite_pairs:
            best_budget, best_fid, best_step = min(finite_pairs, key=lambda item: (item[1], item[0], item[2]))
            ax.set_title(f"{_method_label(method)} | best={best_fid:.3f} @ {best_budget:.1f} ({best_step} st)")
        else:
            ax.set_title(_method_label(method))
        ax.legend(loc="best", fontsize=8)

    for idx in range(len(methods), len(axes_flat)):
        axes_flat[idx].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _fairness_checks(rows: List[RunRow], steps: List[int]) -> Dict[str, Any]:
    by_step: Dict[int, List[RunRow]] = {}
    for row in rows:
        by_step.setdefault(row.step, []).append(row)

    spread_by_step: Dict[str, float] = {}
    hash_count_by_step: Dict[str, int] = {}
    all_loaded_by_step: Dict[str, bool] = {}
    for step in steps:
        sub = by_step.get(step, [])
        base_fids = [r.baseline_fid for r in sub if math.isfinite(r.baseline_fid)]
        if base_fids:
            spread_by_step[str(step)] = float(max(base_fids) - min(base_fids))
        else:
            spread_by_step[str(step)] = float("nan")
        hashes = {r.baseline_ckpt_signature_hash for r in sub if r.baseline_ckpt_signature_hash}
        hash_count_by_step[str(step)] = int(len(hashes))
        all_loaded_by_step[str(step)] = bool(sub) and all(r.baseline_ckpt_loaded for r in sub)

    return {
        "baseline_fid_spread_by_step": spread_by_step,
        "baseline_ckpt_hash_unique_count_by_step": hash_count_by_step,
        "all_baseline_ckpt_loaded_by_step": all_loaded_by_step,
        "fid_ref_policy_unique_count": int(len({r.fid_ref_policy_name for r in rows if r.fid_ref_policy_name})),
        "fid_ref_path_unique_count": int(len({r.fid_ref_resolved_path for r in rows if r.fid_ref_resolved_path})),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FID-vs-iteration sweep for baseline/clean/wild/v1.1/v1.2.")
    p.add_argument("--outdir", type=Path, default=Path("toy_outputs/fid_curve_methods"))
    p.add_argument("--prefix", type=str, default="mnist_20pct_fid_curve")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps-list", type=str, default="1000,2000,4000,8000,12000,16000,20000")
    p.add_argument("--methods", type=str, default="baseline,wild,v1.1,v1.2")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--require-cuda", action="store_true")

    p.add_argument("--dataset-kind", type=str, default="mnist")
    p.add_argument("--image-channels", type=int, default=1)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--mnist-train-percent", type=float, default=20.0)
    p.add_argument("--mnist-val-percent", type=float, default=100.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--training-objective", type=str, default="edm", choices=["edm", "score"])
    p.add_argument("--n-steps-path", type=int, default=24)
    p.add_argument("--sigma-min", type=float, default=0.01)
    p.add_argument("--sigma-max", type=float, default=80.0)
    p.add_argument("--auto-log-normal-params", action="store_true")
    p.add_argument("--use-ema-eval", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--eval-samples", type=int, default=2000)
    p.add_argument("--fid-samples", type=int, default=2000)
    p.add_argument("--fid-ref-split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--fid-ref-subset-percent", type=float, default=100.0)
    p.add_argument(
        "--fid-ref-subset-sampling",
        type=str,
        default="stratified",
        choices=["first", "global", "stratified"],
    )
    p.add_argument("--fid-ref-subset-seed", type=int, default=0)
    p.add_argument("--fid-ref-max-images", type=int, default=5000)
    p.add_argument("--force-ref-refresh", action="store_true")
    p.add_argument("--warmup-clean-steps", type=int, default=900)
    p.add_argument("--warmup-ramp-steps", type=int, default=600)
    p.add_argument("--device", type=str, default="cuda")

    # Baseline checkpoint protocol.
    p.add_argument(
        "--baseline-ckpt-source",
        type=Path,
        required=True,
        help="Path to the fixed baseline checkpoint (e.g., 20k clean checkpoint).",
    )
    p.add_argument(
        "--alias-ckpt-dir",
        type=Path,
        default=None,
        help="Directory for per-step alias checkpoints. Defaults to <outdir>/_baseline_alias_ckpt.",
    )

    # WILD args.
    p.add_argument("--wild-update-interval", type=int, default=20)
    p.add_argument("--wild-cache-batches", type=int, default=4)
    p.add_argument("--wild-inner-steps", type=int, default=3)
    p.add_argument("--wild-step-size", type=float, default=0.05)
    p.add_argument("--wild-gamma", type=float, default=2.0)
    p.add_argument("--wild-fixed-noise-inner", action="store_true")
    p.add_argument("--wild-clamp-samples", action="store_true")
    p.add_argument("--wild-sample-min", type=float, default=-1.0)
    p.add_argument("--wild-sample-max", type=float, default=1.0)
    p.add_argument("--wild-delta-ratio-denom", type=float, default=1.0)

    # v1.1 args.
    p.add_argument("--v11-step-size", type=float, default=0.0005)
    p.add_argument("--v11-transport-gamma", type=float, default=2.0)
    p.add_argument("--v11-total-budget-rho", type=float, default=0.02)
    p.add_argument(
        "--v11-projection-mode",
        type=str,
        default="global_remaining",
        choices=["global_remaining", "step_clip", "step_exact", "kappa_clip", "none"],
    )

    # v1.2 args.
    p.add_argument("--v12-adv-steps", type=int, default=1)
    p.add_argument("--v12-step-size", type=float, default=0.02)
    p.add_argument("--v12-lambda-init", type=float, default=0.1)
    p.add_argument("--v12-lambda-lr", type=float, default=0.001)
    p.add_argument("--v12-rho-target", type=float, default=0.0001)
    p.add_argument("--v12-robust-mix", type=float, default=0.3)
    p.add_argument("--v12-start-step", type=int, default=0)
    p.add_argument("--v12-ramp-steps", type=int, default=0)
    p.add_argument("--v12-max-delta", type=float, default=0.05)
    p.add_argument("--v12-sigma-floor", type=float, default=0.0)
    p.add_argument("--v12-sigma-cut", type=float, default=0.5)
    p.add_argument("--v12-gate-power", type=float, default=2.0)
    p.add_argument("--v12-delta-space", type=str, default="image", choices=["image", "noise"])
    return p


def main() -> None:
    args = build_parser().parse_args()
    steps = _parse_steps(args.steps_list)
    methods = _parse_methods(args.methods)
    repo_root = _repo_root()
    run_toy = repo_root / "toy" / "run_toy.py"

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("[ERROR] --require-cuda is set but torch.cuda.is_available() is False.")
    if not args.baseline_ckpt_source.is_file():
        raise SystemExit(f"[ERROR] baseline checkpoint not found: {args.baseline_ckpt_source}")
    baseline_ckpt_source_kind = _baseline_ckpt_source_kind(args.baseline_ckpt_source.resolve())
    disable_strict_meta = baseline_ckpt_source_kind != "native"
    print(
        "[info]"
        f" baseline_ckpt_source={args.baseline_ckpt_source.resolve()}"
        f" source_kind={baseline_ckpt_source_kind}"
        f" disable_strict_meta={disable_strict_meta}",
        flush=True,
    )

    outdir: Path = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    alias_dir = (args.alias_ckpt_dir or (outdir / "_baseline_alias_ckpt")).resolve()
    alias_dir.mkdir(parents=True, exist_ok=True)
    fid_ref_policy_name = default_mnist_fid_policy_name(
        split=str(args.fid_ref_split),
        image_size=int(args.image_size),
        subset_percent=float(args.fid_ref_subset_percent),
        subset_sampling=str(args.fid_ref_subset_sampling),
        subset_seed=int(args.fid_ref_subset_seed),
        max_images=int(args.fid_ref_max_images),
    )
    fid_ref_dir = (outdir / "_fid_refs").resolve()
    fid_ref_path = fid_ref_dir / f"{fid_ref_policy_name}.npz"
    fid_ref_meta: Dict[str, Any]
    if args.dry_run:
        fid_ref_meta = {
            "policy_name": fid_ref_policy_name,
            "split": str(args.fid_ref_split),
            "image_size": int(args.image_size),
            "subset_percent_requested": float(args.fid_ref_subset_percent),
            "subset_sampling": str(args.fid_ref_subset_sampling),
            "subset_seed": int(args.fid_ref_subset_seed),
            "max_images_requested": int(args.fid_ref_max_images),
            "dest": str(fid_ref_path),
        }
        print(f"[dry-run] would ensure MNIST FID reference: {fid_ref_path}", flush=True)
    else:
        fid_ref_meta = build_mnist_fid_reference(
            split=str(args.fid_ref_split),
            image_size=int(args.image_size),
            subset_percent=float(args.fid_ref_subset_percent),
            subset_sampling=str(args.fid_ref_subset_sampling),
            subset_seed=int(args.fid_ref_subset_seed),
            max_images=int(args.fid_ref_max_images),
            dest=fid_ref_path,
            images_dir=fid_ref_dir / f"{fid_ref_policy_name}_images",
            policy_name=fid_ref_policy_name,
            force=bool(args.force_ref_refresh),
        )

    rows: List[RunRow] = []
    for step in steps:
        alias_path = alias_dir / f"{args.baseline_ckpt_source.stem}_alias_st{step}.pt"
        _ensure_baseline_alias_ckpt(
            source_ckpt=args.baseline_ckpt_source.resolve(),
            alias_ckpt=alias_path,
            step=step,
            dry_run=args.dry_run,
        )

        for method in methods:
            exp_name = f"{args.prefix}_{method.replace('.', '_')}_st{step}_s{args.seed}"
            exp_dir = outdir / exp_name
            metrics_path = exp_dir / "metrics.json"

            if args.skip_existing and metrics_path.is_file():
                print(f"[skip-existing] {metrics_path}", flush=True)
            else:
                cmd = [
                    sys.executable,
                    str(run_toy),
                    "--dataset-kind",
                    str(args.dataset_kind),
                    "--model-kind",
                    "image_conv",
                    "--diagnostics-kind",
                    "image_basic",
                    "--image-channels",
                    str(args.image_channels),
                    "--image-size",
                    str(args.image_size),
                    "--mnist-train-percent",
                    str(args.mnist_train_percent),
                    "--mnist-val-percent",
                    str(args.mnist_val_percent),
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
                    str(fid_ref_path),
                    "--fid-ref-policy",
                    fid_ref_policy_name,
                    "--skip-checks",
                    "--disable-baseline-gate",
                    "--seed",
                    str(args.seed),
                    "--outdir",
                    str(outdir),
                    "--exp-name",
                    exp_name,
                    "--warmup-clean-steps",
                    str(args.warmup_clean_steps),
                    "--warmup-ramp-steps",
                    str(args.warmup_ramp_steps),
                    "--baseline-ckpt-path",
                    str(alias_path),
                ]
                if args.auto_log_normal_params:
                    cmd.append("--auto-log-normal-params")
                if args.use_ema_eval:
                    cmd.extend(["--use-ema-eval", "--ema-decay", str(args.ema_decay)])
                if disable_strict_meta:
                    cmd.append("--disable-baseline-ckpt-strict-meta")
                cmd.extend(_method_cli_args(method, args))
                _run(cmd, cwd=repo_root, dry_run=args.dry_run)

            if args.dry_run:
                continue
            if not metrics_path.is_file():
                raise RuntimeError(f"Missing metrics file: {metrics_path}")
            row = _load_row(metrics_path, method, step, exp_name)
            rows.append(row)
            transport_msg = ""
            if row.method_transport_cost_name and math.isfinite(row.method_transport_cost_final):
                transport_msg = (
                    f" {row.method_transport_cost_name}="
                    f"{_fmt(row.method_transport_cost_final, 6)}"
                )
            print(
                "[row]"
                f" method={row.method_label} step={row.step}"
                f" baseline_fid={_fmt(row.baseline_fid)} robust_fid={_fmt(row.robust_fid)}"
                f" robust_images={_fmt(row.robust_images_seen_total, 1)}"
                f" robust_be={_fmt(row.robust_batch_equiv_denoiser_evals_total, 1)}"
                f" delta={_fmt(row.fid_delta)} runtime_total={_fmt(row.runtime_total_sec, 2)}s"
                f"{transport_msg}",
                flush=True,
            )

    if args.dry_run:
        print("[done] dry-run completed.", flush=True)
        return

    rows.sort(key=lambda r: (r.method, r.step))
    csv_path = outdir / f"{args.prefix}_fid_curve_rows_s{args.seed}.csv"
    _write_csv(csv_path, rows)

    panel_path = outdir / f"{args.prefix}_fid_curve_panel_s{args.seed}.png"
    _plot_panel(rows, methods, panel_path)
    budget_panel_path = outdir / f"{args.prefix}_fid_curve_budget_panel_s{args.seed}.png"
    _plot_budget_panel(rows, methods, budget_panel_path, "robust_batch_equiv_denoiser_evals_total")

    budget_frontier_compute_rows = _budget_summary_rows(rows, methods, "robust_batch_equiv_denoiser_evals_total")
    budget_frontier_images_rows = _budget_summary_rows(rows, methods, "robust_images_seen_total")
    budget_frontier_compute_csv = outdir / f"{args.prefix}_budget_frontier_compute_s{args.seed}.csv"
    budget_frontier_images_csv = outdir / f"{args.prefix}_budget_frontier_images_s{args.seed}.csv"
    _write_csv(budget_frontier_compute_csv, budget_frontier_compute_rows)
    _write_csv(budget_frontier_images_csv, budget_frontier_images_rows)

    checks = _fairness_checks(rows, steps)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "steps_list": [int(s) for s in steps],
        "methods": methods,
        "protocol": {
            "name": "fixed_baseline_finetune_curve",
            "family": "fixed_baseline_finetune_curve",
            "description": (
                "Each point trains robust method for S steps from the same baseline weights. "
                "Per-step alias checkpoint adjusts baseline_steps metadata only. "
                "Legacy sweep checkpoints are converted to baseline aliases and loaded "
                "with strict signature validation disabled."
            ),
            "comparison_question": (
                "Compare robust fine-tuning behavior from a shared pretrained baseline, "
                "not from-scratch convergence."
            ),
            "baseline_ckpt_source": str(args.baseline_ckpt_source.resolve()),
            "baseline_ckpt_source_kind": baseline_ckpt_source_kind,
            "baseline_ckpt_strict_meta_disabled": bool(disable_strict_meta),
            "alias_ckpt_dir": str(alias_dir),
            "warmup_clean_steps": int(args.warmup_clean_steps),
            "warmup_ramp_steps": int(args.warmup_ramp_steps),
            "fid_reference": fid_ref_meta,
        },
        "rows": [r.to_dict() for r in rows],
        "budget_frontiers": {
            "robust_batch_equiv_denoiser_evals_total": budget_frontier_compute_rows,
            "robust_images_seen_total": budget_frontier_images_rows,
        },
        "fairness_checks": checks,
        "artifacts": {
            "csv": str(csv_path),
            "panel": str(panel_path),
            "budget_panel": str(budget_panel_path),
            "budget_frontier_compute_csv": str(budget_frontier_compute_csv),
            "budget_frontier_images_csv": str(budget_frontier_images_csv),
        },
    }
    summary_path = outdir / f"{args.prefix}_fid_curve_summary_s{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[done] saved:")
    print(f"  - {csv_path}")
    print(f"  - {panel_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
