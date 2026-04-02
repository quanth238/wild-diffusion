"""Cross-experiment comparison plots for MNIST robustness benchmarks.

Usage:
    python3 toy/compare_robustness.py

Reads toy_outputs/<exp_name>/metrics.json and produces:
  - toy_outputs/comparison_robustness.png  (4-panel summary)
  - toy_outputs/comparison_grids.png       (stacked forward/backward grids)
"""

import json
import os
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


METHOD_STYLES = {
    "wild_edm": {
        "color": "#D97706",
        "marker": "o",
        "label": "WILD / EDM (Baseline)",
    },
    "wild_score": {
        "color": "#B45309",
        "marker": "s",
        "label": "WILD / Score (Baseline)",
    },
    "v2_edm": {
        "color": "#2563EB",
        "marker": "^",
        "label": "v2 / EDM (Proposed)",
    },
    "v2_score": {
        "color": "#1D4ED8",
        "marker": "D",
        "label": "v2 / Score (Proposed)",
    },
}

EXPERIMENT_ORDER = ("wild_edm", "wild_score", "v2_edm", "v2_score")
PLOT_EVERY = 4


def load_metrics(exp_dir: str):
    path = os.path.join(exp_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ordered_experiment_items(experiments: Dict) -> Iterable[Tuple[str, Dict]]:
    remaining = set(experiments.keys())
    for key in EXPERIMENT_ORDER:
        if key in experiments:
            yield key, experiments[key]
            remaining.discard(key)
    for key in sorted(remaining):
        yield key, experiments[key]


def _safe_array(values) -> np.ndarray:
    if isinstance(values, (list, tuple)) and len(values) > 0:
        return np.asarray(values, dtype=np.float64)
    return np.asarray([], dtype=np.float64)


def _to_metric_value(value) -> float:
    return float(value) if isinstance(value, (int, float)) else float("nan")


def _subsample_xy(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if xs.size == 0 or ys.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return xs[::PLOT_EVERY], ys[::PLOT_EVERY]


def _compute_summary(metrics: Dict) -> Dict:
    pool = metrics.get("denoise_debug_heldout_pool", {})
    step = _safe_array(pool.get("step", []))
    atk_robust = _safe_array(pool.get("robust_on_forward_attack", []))
    atk_base = _safe_array(pool.get("baseline_on_forward_attack", []))
    clean_robust = _safe_array(pool.get("robust_on_forward_baseline", []))
    clean_base = _safe_array(pool.get("baseline_on_forward_baseline", []))

    atk_sum_robust = float(np.sum(atk_robust)) if atk_robust.size else float("nan")
    atk_sum_base = float(np.sum(atk_base)) if atk_base.size else float("nan")
    clean_sum_robust = float(np.sum(clean_robust)) if clean_robust.size else float("nan")
    clean_sum_base = float(np.sum(clean_base)) if clean_base.size else float("nan")

    attack_gain_pct = float("nan")
    if np.isfinite(atk_sum_base) and abs(atk_sum_base) > 1e-12 and np.isfinite(atk_sum_robust):
        attack_gain_pct = 100.0 * (atk_sum_base - atk_sum_robust) / atk_sum_base

    clean_overhead_pct = float("nan")
    if np.isfinite(clean_sum_base) and abs(clean_sum_base) > 1e-12 and np.isfinite(clean_sum_robust):
        clean_overhead_pct = 100.0 * (clean_sum_robust - clean_sum_base) / clean_sum_base

    attack_gain_curve = np.asarray([], dtype=np.float64)
    if step.size and atk_robust.size == step.size and atk_base.size == step.size:
        denom = np.maximum(np.abs(atk_base), 1e-12)
        attack_gain_curve = 100.0 * (atk_base - atk_robust) / denom

    sq = metrics.get("sample_quality_debug", {})
    baseline_fid = _to_metric_value(sq.get("baseline_fid"))
    robust_fid = _to_metric_value(sq.get("robust_fid"))

    return {
        "step": step,
        "atk_robust": atk_robust,
        "atk_base": atk_base,
        "clean_robust": clean_robust,
        "clean_base": clean_base,
        "atk_sum_robust": atk_sum_robust,
        "atk_sum_base": atk_sum_base,
        "clean_sum_robust": clean_sum_robust,
        "clean_sum_base": clean_sum_base,
        "attack_gain_pct": attack_gain_pct,
        "clean_overhead_pct": clean_overhead_pct,
        "attack_gain_curve": attack_gain_curve,
        "baseline_fid": baseline_fid,
        "robust_fid": robust_fid,
    }


def _style_axes(axes):
    for ax in axes:
        ax.set_facecolor("#F8FAFC")
        ax.grid(color="#E5E7EB", linewidth=0.9, alpha=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)


def _method_style(name: str) -> Dict:
    return METHOD_STYLES.get(
        name,
        {"color": "#111827", "marker": "x", "label": name},
    )


def _short_name(name: str) -> str:
    return name.replace("_edm", "\nEDM").replace("_score", "\nScore").replace("wild", "WILD").replace("v2", "v2")


def _short_name_inline(name: str) -> str:
    return name.replace("_", "-")


def _draw_fid_panel(ax, summaries: Dict[str, Dict]) -> None:
    names = list(summaries.keys())
    base_fids = np.array([summaries[n]["baseline_fid"] for n in names], dtype=np.float64)
    robust_fids = np.array([summaries[n]["robust_fid"] for n in names], dtype=np.float64)
    has_any_fid = np.isfinite(base_fids).any() or np.isfinite(robust_fids).any()

    if not has_any_fid:
        ax.text(
            0.5,
            0.5,
            "FID not available\n(run experiments with --compute-fid)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="#6B7280",
        )
        return

    x = np.arange(len(names))
    width = 0.36
    bar_base = ax.bar(
        x - width / 2,
        base_fids,
        width=width,
        label="Baseline model",
        color="#9CA3AF",
        alpha=0.9,
        zorder=3,
    )
    bar_robust = ax.bar(
        x + width / 2,
        robust_fids,
        width=width,
        label="Robust model",
        color=[_method_style(n)["color"] for n in names],
        alpha=0.9,
        zorder=3,
    )

    for i, (bar_b, bar_r) in enumerate(zip(bar_base, bar_robust)):
        h_b = bar_b.get_height()
        h_r = bar_r.get_height()
        if np.isfinite(h_b):
            ax.text(bar_b.get_x() + bar_b.get_width() / 2, h_b + 0.2, f"{h_b:.1f}", ha="center", va="bottom", fontsize=8)
        if np.isfinite(h_r):
            ax.text(bar_r.get_x() + bar_r.get_width() / 2, h_r + 0.2, f"{h_r:.1f}", ha="center", va="bottom", fontsize=8)
        if np.isfinite(h_b) and np.isfinite(h_r):
            delta = h_r - h_b
            sign = "+" if delta >= 0 else ""
            ax.text(x[i], max(h_b, h_r) + 1.0, f"{sign}{delta:.1f}", ha="center", va="bottom", fontsize=8, color="#374151")

    ymax = np.nanmax(np.concatenate([base_fids, robust_fids]))
    if np.isfinite(ymax):
        ax.set_ylim(0.0, max(2.0, ymax * 1.22))

    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(n) for n in names], fontsize=9)
    ax.set_ylabel("FID (lower is better)", fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=True, facecolor="white", edgecolor="#D1D5DB")


def _draw_tradeoff_panel(ax, summaries: Dict[str, Dict]) -> None:
    xs, ys = [], []
    for name, summary in summaries.items():
        x = summary["clean_overhead_pct"]
        y = summary["attack_gain_pct"]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        style = _method_style(name)
        ax.scatter(
            x,
            y,
            s=92,
            color=style["color"],
            marker=style["marker"],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax.text(x + 0.25, y + 0.25, _short_name_inline(name), fontsize=8, color="#111827")
        xs.append(x)
        ys.append(y)

    ax.axhline(0.0, color="#9CA3AF", linewidth=1.1, linestyle="--", zorder=1)
    ax.axvline(0.0, color="#9CA3AF", linewidth=1.1, linestyle="--", zorder=1)
    ax.set_xlabel("Clean-path overhead % (lower is better)", fontsize=9.5)
    ax.set_ylabel("Attack-path gain % (higher is better)", fontsize=9.5)
    ax.set_title("Robustness vs Utility Tradeoff", fontsize=10.5, fontweight="semibold")
    ax.text(
        0.98,
        0.03,
        "Ideal: upper-left",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#6B7280",
    )

    if xs and ys:
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_pad = max(1.0, (x_max - x_min) * 0.2)
        y_pad = max(1.0, (y_max - y_min) * 0.2)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)


def plot_comparisons(experiments: Dict, out_path: str = "toy_outputs/comparison_robustness.png") -> Dict[str, Dict]:
    fig = plt.figure(figsize=(17, 12))
    fig.patch.set_facecolor("#FFFFFF")
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24, left=0.06, right=0.97, top=0.90, bottom=0.07)
    ax_atk = fig.add_subplot(gs[0, 0])
    ax_gain = fig.add_subplot(gs[0, 1])
    ax_fid = fig.add_subplot(gs[1, 0])
    ax_trade = fig.add_subplot(gs[1, 1])
    _style_axes([ax_atk, ax_gain, ax_fid, ax_trade])

    summaries: Dict[str, Dict] = {}
    for name, data in _ordered_experiment_items(experiments):
        metrics = data.get("metrics", {})
        summary = _compute_summary(metrics)
        summaries[name] = summary
        style = _method_style(name)

        step = summary["step"]
        atk_robust = summary["atk_robust"]
        gain_curve = summary["attack_gain_curve"]

        if step.size and atk_robust.size == step.size:
            xs, ys = _subsample_xy(step, atk_robust)
            ax_atk.plot(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=4.2,
                label=style["label"],
                zorder=3,
            )
        if step.size and gain_curve.size == step.size:
            xs, ys = _subsample_xy(step, gain_curve)
            ax_gain.plot(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=4.0,
                label=style["label"],
                zorder=3,
            )

    ax_gain.axhline(0.0, color="#9CA3AF", linewidth=1.1, linestyle="--", zorder=1)
    ax_atk.set_title("Adversarial-Path Denoise Error", fontsize=10.5, fontweight="semibold")
    ax_atk.set_xlabel("Forward timestep k", fontsize=9.5)
    ax_atk.set_ylabel("Denoise MSE to x0 (lower is better)", fontsize=9.5)
    ax_gain.set_title("Per-Step Robustness Gain", fontsize=10.5, fontweight="semibold")
    ax_gain.set_xlabel("Forward timestep k", fontsize=9.5)
    ax_gain.set_ylabel("Gain vs baseline model (%)", fontsize=9.5)
    _draw_fid_panel(ax_fid, summaries)
    ax_fid.set_title("FID Comparison", fontsize=10.5, fontweight="semibold")
    _draw_tradeoff_panel(ax_trade, summaries)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=_method_style(name)["color"],
            marker=_method_style(name)["marker"],
            linewidth=2.2,
            markersize=6,
            label=_method_style(name)["label"],
        )
        for name in summaries.keys()
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=min(4, len(handles)),
            bbox_to_anchor=(0.5, 0.965),
            fontsize=9.3,
            frameon=True,
            facecolor="white",
            edgecolor="#D1D5DB",
        )

    fig.suptitle(
        "MNIST Robust Diffusion Benchmark: WILD Baseline vs v2 Proposed",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[+] Saved comparison plot -> {out_path}")
    return summaries


def print_markdown_report(summaries: Dict[str, Dict]) -> None:
    print("\n### Robustness Comparison Report")
    print(
        "| Method | Attack Sum (Robust) | Attack Sum (Baseline) | Attack Gain % | "
        "Clean Overhead % | Robust FID | Baseline FID |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name in summaries.keys():
        s = summaries[name]
        atk_r = s["atk_sum_robust"]
        atk_b = s["atk_sum_base"]
        gain = s["attack_gain_pct"]
        clean_ovh = s["clean_overhead_pct"]
        rf = s["robust_fid"]
        bf = s["baseline_fid"]

        atk_r_s = f"{atk_r:.3f}" if np.isfinite(atk_r) else "N/A"
        atk_b_s = f"{atk_b:.3f}" if np.isfinite(atk_b) else "N/A"
        gain_s = f"{gain:+.2f}%" if np.isfinite(gain) else "N/A"
        clean_s = f"{clean_ovh:+.2f}%" if np.isfinite(clean_ovh) else "N/A"
        rf_s = f"{rf:.2f}" if np.isfinite(rf) else "N/A"
        bf_s = f"{bf:.2f}" if np.isfinite(bf) else "N/A"
        print(f"| **{name}** | {atk_r_s} | {atk_b_s} | {gain_s} | {clean_s} | {rf_s} | {bf_s} |")


def merge_image_grids(experiments: Dict, out_path: str = "toy_outputs/comparison_grids.png") -> None:
    images, labels = [], []
    for name, _ in _ordered_experiment_items(experiments):
        img_path = os.path.join("toy_outputs", name, "forward_backward_baseline_attack.png")
        if not os.path.exists(img_path):
            continue
        try:
            images.append(Image.open(img_path).convert("RGB"))
            labels.append(name.upper())
        except Exception as e:
            print(f"[-] Error reading {img_path}: {e}")

    if not images:
        print("[-] No grid images found to merge.")
        return

    label_h = 36
    width = max(img.width for img in images)
    total_h = sum(img.height + label_h for img in images)
    merged = Image.new("RGB", (width, total_h), color=(248, 250, 252))
    draw = ImageDraw.Draw(merged)
    y_offset = 0
    for img, label in zip(images, labels):
        draw.rectangle([0, y_offset, width, y_offset + label_h - 1], fill=(30, 41, 59))
        draw.text((10, y_offset + 8), f"  METHOD: {label}", fill=(248, 250, 252))
        y_offset += label_h
        merged.paste(img.resize((width, img.height), Image.LANCZOS), (0, y_offset))
        y_offset += img.height

    merged.save(out_path)
    print(f"[+] Saved image grid comparison -> {out_path}")


if __name__ == "__main__":
    exp_dirs = {
        "wild_edm": "toy_outputs/wild_edm",
        "wild_score": "toy_outputs/wild_score",
        "v2_edm": "toy_outputs/v2_edm",
        "v2_score": "toy_outputs/v2_score",
    }

    loaded = {name: load_metrics(path) for name, path in exp_dirs.items()}
    valid = {name: value for name, value in loaded.items() if value is not None}
    if not valid:
        print("[-] No metrics.json found in toy_outputs/. Run experiments first.")
    else:
        summaries = plot_comparisons(valid)
        print_markdown_report(summaries)
        merge_image_grids(valid)
