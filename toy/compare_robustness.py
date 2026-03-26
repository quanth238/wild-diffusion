"""Cross-experiment comparison plots for Wild-Diffusion robustness benchmarks.

Usage:
    python3 toy/compare_robustness.py

Reads toy_outputs/<exp_name>/metrics.json for each experiment and produces:
  - toy_outputs/comparison_robustness.png  (4-panel quantitative summary)
  - toy_outputs/comparison_grids.png       (stacked forward/backward image grids)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


# ─── Palette & style ──────────────────────────────────────────────────────────

METHOD_STYLES = {
    "v2_edm":    {"color": "#2563EB", "linestyle": "-",  "marker": "o", "label": "v2 / EDM  [Proposed]"},
    "v2_score":  {"color": "#1D4ED8", "linestyle": "--", "marker": "s", "label": "v2 / Score  [Proposed]"},
    "wild_edm":  {"color": "#D97706", "linestyle": "-",  "marker": "^", "label": "WILD / EDM  [Baseline]"},
    "wild_score": {"color": "#B45309", "linestyle": "--", "marker": "D", "label": "WILD / Score  [Baseline]"},
}

BASELINE_STYLE = {"color": "#6B7280", "linestyle": ":", "linewidth": 1.5, "alpha": 0.75}

PLOT_EVERY = 4   # subsample timestep curves to reduce clutter


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_metrics(exp_dir: str):
    path = os.path.join(exp_dir, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ─── Main 4-panel plot ────────────────────────────────────────────────────────

def plot_comparisons(experiments: dict, out_path: str = "toy_outputs/comparison_robustness.png") -> None:
    """Produce a 4-panel quantitative summary figure."""

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#F8FAFC")

    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30,
                          left=0.07, right=0.97, top=0.91, bottom=0.07)
    ax_atk  = fig.add_subplot(gs[0, 0])   # attack denoise error
    ax_cln  = fig.add_subplot(gs[0, 1])   # clean denoise error
    ax_fid  = fig.add_subplot(gs[1, 0])   # FID bar chart
    ax_loss = fig.add_subplot(gs[1, 1])   # training loss convergence

    _style_axes([ax_atk, ax_cln, ax_fid, ax_loss])

    baseline_plotted_atk = False
    baseline_plotted_cln = False

    for name, data in experiments.items():
        style = METHOD_STYLES.get(name, {"color": "black", "linestyle": "-", "marker": "x", "label": name})
        metrics = data["metrics"]
        pool    = metrics.get("denoise_debug_heldout_pool", {})
        steps   = pool.get("step", [])
        obj     = metrics.get("objective_debug", {})

        # ── Panel 1: Denoising error on adversarial (attack) path ──────────────
        atk_robust  = pool.get("robust_on_forward_attack", [])
        atk_base    = pool.get("baseline_on_forward_attack", [])
        if steps and atk_robust:
            xs, ys_r, ys_b = _subsample(steps, atk_robust, atk_base)
            ax_atk.plot(xs, ys_r,
                        color=style["color"], linestyle=style["linestyle"],
                        marker=style["marker"], markersize=4, linewidth=2.0,
                        label=style["label"])
            if not baseline_plotted_atk and len(ys_b) > 0:
                ax_atk.plot(xs, ys_b, label="No defense (baseline model)",
                            **BASELINE_STYLE)
                baseline_plotted_atk = True

        # ── Panel 2: Denoising error on clean (reference) path ─────────────────
        cln_robust = pool.get("robust_on_forward_baseline", [])
        cln_base   = pool.get("baseline_on_forward_baseline", [])
        if steps and cln_robust:
            xs, ys_r, ys_b = _subsample(steps, cln_robust, cln_base)
            ax_cln.plot(xs, ys_r,
                        color=style["color"], linestyle=style["linestyle"],
                        marker=style["marker"], markersize=4, linewidth=2.0,
                        label=style["label"])
            if not baseline_plotted_cln and len(ys_b) > 0:
                ax_cln.plot(xs, ys_b, label="Baseline model (clean path)",
                            **BASELINE_STYLE)
                baseline_plotted_cln = True

        # ── Panel 4: Training loss curves ─────────────────────────────────────
        loss_curve = obj.get("baseline_loss_curve", [])
        rob_curve  = obj.get("robust_outer_loss_curve", [])
        if loss_curve:
            ax_loss.plot(_smooth(loss_curve, 20),
                         color=style["color"], linestyle="--", linewidth=1.5, alpha=0.55)
        if rob_curve:
            ax_loss.plot(_smooth(rob_curve, 20),
                         color=style["color"], linestyle=style["linestyle"],
                         linewidth=2.0, label=style["label"])

    # ── Panel 3: FID bar chart ─────────────────────────────────────────────────
    _draw_fid_bars(ax_fid, experiments)

    # ── Axis decoration ────────────────────────────────────────────────────────
    _decorate(ax_atk,
              title="Robustness gain ↓  (denoising error on adversarial path)",
              xlabel="Forward timestep k", ylabel="Denoise MSE to x₀",
              note="Robust model on the attack path.\nLower = better recovery from adversarial perturbation.")
    _decorate(ax_cln,
              title="Utility overhead ↓  (denoising error on clean / reference path)",
              xlabel="Forward timestep k", ylabel="Denoise MSE to x₀",
              note="Robust model on the clean path.\nIdeally overlaps baseline – gap = cost of robustness.")
    _decorate(ax_fid,
              title="Sample quality (FID ↓ = better)",
              xlabel="Method", ylabel="FID score (↓ better)",
              note="Lower = generated distribution closer to real MNIST.")
    _decorate(ax_loss,
              title="Training convergence (outer loss curve)",
              xlabel="Training step", ylabel="Weighted denoise loss",
              note="Solid = robust phase (after warm-up)\nDashed = baseline warm-up phase")

    # ── Legend (panels 1 & 2 are crowded; use figure-level legend) ─────────────
    handles = [
        plt.Line2D([0], [0], color=s["color"], linestyle=s["linestyle"],
                   marker=s["marker"], markersize=6, linewidth=2, label=s["label"])
        for s in METHOD_STYLES.values()
    ]
    handles.append(plt.Line2D([0], [0], label="No defense (baseline model)", **BASELINE_STYLE))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.965), fontsize=9.5,
               frameon=True, facecolor="white", edgecolor="#D1D5DB")

    fig.suptitle(
        "Wild-Diffusion vs v2 (Hard-Constraint) — Robustness Benchmark on MNIST",
        fontsize=14, fontweight="bold", y=0.995,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[+] Saved 4-panel comparison plot → {out_path}")


# ─── Helper: FID bar chart ────────────────────────────────────────────────────

def _draw_fid_bars(ax, experiments: dict) -> None:
    """Grouped bar chart: Baseline FID vs Robust FID per method."""
    names, base_fids, rob_fids = [], [], []
    has_any_fid = False
    for name, data in experiments.items():
        sq = data["metrics"].get("sample_quality_debug", {})
        bf = sq.get("baseline_fid")
        rf = sq.get("robust_fid")
        names.append(name)
        base_fids.append(bf if isinstance(bf, (int, float)) else float("nan"))
        rob_fids.append(rf if isinstance(rf, (int, float)) else float("nan"))
        if isinstance(bf, (int, float)) or isinstance(rf, (int, float)):
            has_any_fid = True

    if not has_any_fid:
        ax.text(0.5, 0.5, "FID not available\n(run with --compute-fid)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#9CA3AF", style="italic")
        return

    x = np.arange(len(names))
    w = 0.36
    bar_base = ax.bar(x - w / 2, base_fids, width=w,
                      label="Baseline model", color="#6B7280", alpha=0.7, zorder=3)
    bar_rob  = ax.bar(x + w / 2, rob_fids,  width=w,
                      label="Robust model",
                      color=[METHOD_STYLES.get(n, {}).get("color", "#111827") for n in names],
                      alpha=0.85, zorder=3)

    # Value labels on bars
    for bar in [*bar_base, *bar_rob]:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(n) for n in names], fontsize=9)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8.5, frameon=True, facecolor="white")


# ─── Utilities ────────────────────────────────────────────────────────────────

def _subsample(steps, series_a, series_b):
    """Subsample every PLOT_EVERY points; convert lists to arrays."""
    xs  = np.array(steps)[::PLOT_EVERY]
    ysa = np.array(series_a)[::PLOT_EVERY] if series_a else np.array([])
    ysb = np.array(series_b)[::PLOT_EVERY] if series_b else np.array([])
    return xs, ysa, ysb


def _smooth(values, window: int = 10):
    """Simple box smoothing for noisy loss curves."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _style_axes(axes):
    """Apply common background / grid styling."""
    for ax in axes:
        ax.set_facecolor("#F1F5F9")
        ax.grid(color="white", linewidth=1.1, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)


def _decorate(ax, title: str, xlabel: str, ylabel: str, note: str = ""):
    ax.set_title(title, fontsize=10.5, fontweight="semibold", pad=7)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=True, facecolor="white", edgecolor="#E5E7EB")
    if note:
        ax.text(0.98, 0.97, note, transform=ax.transAxes,
                fontsize=7.5, color="#6B7280", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"))


def _short_name(name: str) -> str:
    return name.replace("_edm", "\nEDM").replace("_score", "\nScore")


# ─── Markdown report ──────────────────────────────────────────────────────────

def print_markdown_report(experiments: dict) -> None:
    labels = list(experiments.keys())
    print("\n### Robustness Comparison Report")
    print("| Method | Atk Denoise Err (sum↓) | vs Baseline | Robust FID↓ | Baseline FID↓ |")
    print("|---|---|---|---|---|")
    for name in labels:
        data = experiments[name]
        metrics = data["metrics"]
        pool = metrics.get("denoise_debug_heldout_pool", {})
        atk_r = pool.get("robust_on_forward_attack", [])
        atk_b = pool.get("baseline_on_forward_attack", [])
        sq = metrics.get("sample_quality_debug", {})
        bf = sq.get("baseline_fid")
        rf = sq.get("robust_fid")
        bf_s = f"{bf:.2f}" if isinstance(bf, (int, float)) else "N/A"
        rf_s = f"{rf:.2f}" if isinstance(rf, (int, float)) else "N/A"
        err_r = f"{sum(atk_r):.2f}" if atk_r else "N/A"
        err_b = f"{sum(atk_b):.2f}" if atk_b else "N/A"
        print(f"| **{name}** | {err_r} | vs {err_b} | {rf_s} | {bf_s} |")


# ─── Image grid ───────────────────────────────────────────────────────────────

def merge_image_grids(experiments: dict, out_path: str = "toy_outputs/comparison_grids.png") -> None:
    images, labels = [], []
    for name in experiments:
        img_path = os.path.join("toy_outputs", name, "forward_backward_baseline_attack.png")
        if os.path.exists(img_path):
            try:
                images.append(Image.open(img_path).convert("RGB"))
                labels.append(name.upper())
            except Exception as e:
                print(f"[-] Error reading {img_path}: {e}")

    if not images:
        print("[-] No grid images found to merge.")
        return

    label_h = 36
    width    = max(img.width for img in images)
    total_h  = sum(img.height + label_h for img in images)
    merged   = Image.new("RGB", (width, total_h), color=(248, 250, 252))
    draw     = ImageDraw.Draw(merged)
    y_offset = 0
    for img, label in zip(images, labels):
        draw.rectangle([0, y_offset, width, y_offset + label_h - 1], fill=(30, 41, 59))
        draw.text((10, y_offset + 8), f"  METHOD: {label}", fill=(248, 250, 252))
        y_offset += label_h
        merged.paste(img.resize((width, img.height), Image.LANCZOS), (0, y_offset))
        y_offset += img.height

    merged.save(out_path)
    print(f"[+] Saved image grid comparison → {out_path}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    exp_dirs = {
        "v2_edm":    "toy_outputs/v2_edm",
        "v2_score":  "toy_outputs/v2_score",
        "wild_edm":  "toy_outputs/wild_edm",
        "wild_score": "toy_outputs/wild_score",
    }

    exps = {k: load_metrics(v) for k, v in exp_dirs.items()}
    valid_exps = {k: v for k, v in exps.items() if v is not None}

    if not valid_exps:
        print("[-] No metrics.json found in toy_outputs/. Run run_all_comparisons.sh first.")
    else:
        plot_comparisons(valid_exps)
        print_markdown_report(valid_exps)
        merge_image_grids(valid_exps)
