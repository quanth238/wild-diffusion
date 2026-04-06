from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_scatter_comparison(
    *,
    path: Path,
    real_points: np.ndarray,
    generated_points: np.ndarray,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    _scatter(axes[0], real_points, "Real data")
    _scatter(axes[1], generated_points, "Generated samples")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_adversarial_debug(
    *,
    path: Path,
    original_points: np.ndarray,
    adversarial_points: np.ndarray,
    title: str,
    mean_l2_shift: float,
    max_l2_shift: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    bounds = _shared_bounds(original_points, adversarial_points)
    viz_scale = _compute_viz_scale(original_points, adversarial_points)
    _scatter(axes[0], original_points, "Original points", color="#1f77b4", bounds=bounds)
    _scatter(axes[1], adversarial_points, "Adversarial points", color="#d62728", bounds=bounds)
    _scatter_overlay(axes[2], original_points, adversarial_points, bounds=bounds)
    _scatter_amplified_overlay(
        axes[3],
        original_points,
        adversarial_points,
        viz_scale=viz_scale,
    )
    fig.suptitle(
        f"{title} | mean L2 shift={mean_l2_shift:.4f}, max L2 shift={max_l2_shift:.4f}, viz x{viz_scale:.1f}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_training_curves(
    *,
    path: Path,
    loss_history: list[float],
    eval_history: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(loss_history, color="#1f77b4", linewidth=1.6)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("EDM loss")
    axes[0].grid(alpha=0.2)

    if eval_history:
        epochs = [item["epoch"] for item in eval_history]
        mmd = [item["mmd_rbf"] for item in eval_history]
        sw = [item["sliced_wasserstein"] for item in eval_history]
        axes[1].plot(epochs, mmd, label="MMD", color="#2ca02c", linewidth=1.6)
        axes[1].plot(epochs, sw, label="SWD", color="#ff7f0e", linewidth=1.6)
        axes[1].legend(frameon=False)
    axes[1].set_title("Eval metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_cdro_sde_forward_curves(
    *,
    path: Path,
    history: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    if history:
        epochs = [item["epoch"] for item in history]
        predictor = [item["predictor_loss"] for item in history]
        control_cost = [item["control_cost"] for item in history]
        lambda_values = [item["lambda_value"] for item in history]

        axes[0].plot(epochs, predictor, color="#1f77b4", linewidth=1.8)
        axes[1].plot(epochs, control_cost, color="#d62728", linewidth=1.8)
        axes[2].plot(epochs, lambda_values, color="#2ca02c", linewidth=1.8)

    axes[0].set_title("Predictor loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[1].set_title("Control cost")
    axes[1].set_xlabel("Epoch")
    axes[2].set_title("Dual lambda")
    axes[2].set_xlabel("Epoch")

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_markov_score_training_curves(
    *,
    path: Path,
    history: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.reshape(2, 4)

    if history:
        epochs = [item["epoch"] for item in history]
        score_loss = [item.get("score_loss", np.nan) for item in history]
        diag_nominal = [item.get("eval_diag_nominal_score_loss", np.nan) for item in history]
        diag_controlled = [item.get("eval_diag_controlled_score_loss", np.nan) for item in history]
        control_cost = [item.get("control_cost", np.nan) for item in history]
        target_budget = [item.get("target_total_budget", np.nan) for item in history]
        lambda_values = [item.get("lambda_value", np.nan) for item in history]
        budget_util = [item.get("budget_utilization", np.nan) for item in history]
        adversary_value = [item.get("adversary_value", np.nan) for item in history]
        eval_swd = [item.get("eval_sliced_wasserstein", np.nan) for item in history]
        eval_swd_nocontrol = [item.get("eval_reverse_no_control_sliced_wasserstein", np.nan) for item in history]
        score_grad = [item.get("mean_score_grad_norm", np.nan) for item in history]
        control_grad = [item.get("mean_control_grad_norm", np.nan) for item in history]
        control_norm = [item.get("mean_control_norm", np.nan) for item in history]
        drift_ratio = [item.get("eval_diag_control_to_drift_ratio", np.nan) for item in history]
        sample_mean_gap = [item.get("eval_sample_mean_gap", np.nan) for item in history]
        sample_std_gap = [item.get("eval_sample_std_gap", np.nan) for item in history]

        axes[0, 0].plot(epochs, score_loss, color="#1f77b4", linewidth=1.8)

        axes[0, 1].plot(epochs, diag_nominal, color="#2ca02c", linewidth=1.6, label="diag nominal")
        axes[0, 1].plot(epochs, diag_controlled, color="#d62728", linewidth=1.6, label="diag controlled")
        if np.isfinite(diag_nominal).any() or np.isfinite(diag_controlled).any():
            axes[0, 1].legend(frameon=False, fontsize=8)

        axes[0, 2].plot(epochs, control_cost, color="#d62728", linewidth=1.8, label="control cost")
        axes[0, 2].plot(epochs, target_budget, color="#7f7f7f", linewidth=1.6, linestyle="--", label="target budget")
        axes[0, 2].legend(frameon=False, fontsize=8)

        axes[0, 3].plot(epochs, lambda_values, color="#2ca02c", linewidth=1.8, label="lambda")
        axes[0, 3].plot(epochs, budget_util, color="#ff7f0e", linewidth=1.6, label="budget util")
        axes[0, 3].plot(epochs, adversary_value, color="#9467bd", linewidth=1.2, alpha=0.8, label="adv obj")
        axes[0, 3].legend(frameon=False, fontsize=8)

        axes[1, 0].plot(epochs, eval_swd, color="#1f77b4", linewidth=1.8, label="eval SWD")
        axes[1, 0].plot(epochs, eval_swd_nocontrol, color="#ff7f0e", linewidth=1.6, linestyle="--", label="no-ctrl SWD")
        if np.isfinite(eval_swd).any() or np.isfinite(eval_swd_nocontrol).any():
            axes[1, 0].legend(frameon=False, fontsize=8)

        axes[1, 1].plot(epochs, score_grad, color="#1f77b4", linewidth=1.8, label="score grad")
        axes[1, 1].plot(epochs, control_grad, color="#d62728", linewidth=1.6, label="control grad")
        axes[1, 1].legend(frameon=False, fontsize=8)

        axes[1, 2].plot(epochs, control_norm, color="#d62728", linewidth=1.8, label="mean control norm")
        axes[1, 2].plot(epochs, drift_ratio, color="#2ca02c", linewidth=1.6, label="control/drift")
        axes[1, 2].legend(frameon=False, fontsize=8)

        axes[1, 3].plot(epochs, sample_mean_gap, color="#1f77b4", linewidth=1.8, label="mean gap")
        axes[1, 3].plot(epochs, sample_std_gap, color="#ff7f0e", linewidth=1.6, label="std gap")
        axes[1, 3].legend(frameon=False, fontsize=8)

    axes[0, 0].set_title("Score loss")
    axes[0, 1].set_title("Diag score losses")
    axes[0, 2].set_title("Control vs budget")
    axes[0, 3].set_title("Lambda / utilization")
    axes[1, 0].set_title("Eval SWD")
    axes[1, 1].set_title("Gradient norms")
    axes[1, 2].set_title("Control magnitude")
    axes[1, 3].set_title("Sample geometry gaps")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_cdro_sde_reverse_curves(
    *,
    path: Path,
    history: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    if history:
        epochs = [item["epoch"] for item in history]
        train_loss = [item["train_loss"] for item in history]
        val_loss = [item["val_loss"] for item in history]
        swd = [item["sliced_wasserstein"] for item in history]
        mmd = [item["mmd_rbf"] for item in history]

        axes[0].plot(epochs, train_loss, color="#1f77b4", linewidth=1.8, label="train")
        axes[0].plot(epochs, val_loss, color="#ff7f0e", linewidth=1.8, label="val")
        axes[0].legend(frameon=False, loc="best")
        axes[1].plot(epochs, swd, color="#d62728", linewidth=1.8)
        axes[2].plot(epochs, mmd, color="#2ca02c", linewidth=1.8)

    axes[0].set_title("Reverse NLL")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[1].set_title("Sample SWD")
    axes[1].set_xlabel("Epoch")
    axes[2].set_title("Sample MMD")
    axes[2].set_xlabel("Epoch")

    for ax in axes:
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_method_training_metric_comparison(
    *,
    path: Path,
    dataset: str,
    fraction_tag: str,
    seed: int,
    method_histories: dict[str, list[dict]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    method_colors = {
        "baseline": "#1f77b4",
        "baseline_score": "#17becf",
        "wdro": "#ff7f0e",
        "wdro_score": "#bcbd22",
        "cdro": "#d62728",
        "cdro_markov": "#2ca02c",
    }
    metric_specs = [
        ("train_loss", "Training loss"),
        ("sliced_wasserstein", "SWD"),
        ("mmd_rbf", "MMD"),
    ]

    for ax, (metric_key, title) in zip(axes, metric_specs):
        for method, history in method_histories.items():
            if not history:
                continue
            epochs = [item["epoch"] for item in history]
            if metric_key == "train_loss":
                values = [item.get("train_loss", item.get("score_loss")) for item in history]
            else:
                values = [item[metric_key] for item in history]
            ax.plot(
                epochs,
                values,
                label=method,
                color=method_colors.get(method, "#444444"),
                linewidth=1.8,
                marker="o",
                markersize=3.5,
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Value")
    axes[1].legend(frameon=False, loc="best")
    pretty_methods = ", ".join(method.replace("_", " ").title() for method in method_histories)
    fig.suptitle(f"{dataset} | {fraction_tag} | seed {seed} | {pretty_methods}")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_labeled_metric_tradeoff_curves(
    *,
    path: Path,
    title: str,
    run_histories: dict[str, list[dict]],
    metric_key: str = "sliced_wasserstein",
    x_key: str = "elapsed_minutes",
    x_label: str | None = None,
    y_label: str | None = None,
    best_so_far: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8))

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for idx, (label, history) in enumerate(run_histories.items()):
        if not history:
            continue
        points = [
            (item.get(x_key), item.get(metric_key))
            for item in history
            if item.get(x_key) is not None and item.get(metric_key) is not None
        ]
        if not points:
            continue
        points.sort(key=lambda pair: float(pair[0]))
        xs = [float(x) for x, _ in points]
        ys = [float(y) for _, y in points]
        if best_so_far:
            ys = list(np.minimum.accumulate(np.asarray(ys, dtype=float)))
        ax.plot(
            xs,
            ys,
            label=label,
            color=palette[idx % len(palette)],
            linewidth=1.8,
            marker="o",
            markersize=4.0,
        )

    ax.set_title(title)
    ax.set_xlabel(x_label or x_key.replace("_", " ").title())
    ax.set_ylabel(y_label or metric_key.replace("_", " ").title())
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_process_snapshots(
    *,
    path: Path,
    rows: list[dict],
    title: str,
    bounds: tuple[float, float, float, float] | None = None,
    bounds_mode: str = "global",
) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    all_snapshots = [snapshot for row in rows for snapshot in row["snapshots"]]
    n_rows = len(rows)
    n_cols = max(len(row["snapshots"]) for row in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows), squeeze=False)

    global_bounds = bounds
    column_bounds: list[tuple[float, float, float, float] | None] | None = None
    if bounds_mode == "global":
        if global_bounds is None:
            global_bounds = _shared_bounds(*all_snapshots)
    elif bounds_mode == "per_column":
        column_bounds = []
        for col_idx in range(n_cols):
            column_snapshots = []
            for row in rows:
                if col_idx < len(row["snapshots"]):
                    column_snapshots.append(row["snapshots"][col_idx])
            column_bounds.append(_shared_bounds(*column_snapshots) if column_snapshots else None)
    elif bounds_mode == "per_group_column":
        grouped_bounds: dict[tuple[str, int], tuple[float, float, float, float] | None] = {}
        groups = {str(row.get("bounds_group", row_idx)) for row_idx, row in enumerate(rows)}
        for group in groups:
            for col_idx in range(n_cols):
                column_snapshots = []
                for row_idx, row in enumerate(rows):
                    row_group = str(row.get("bounds_group", row_idx))
                    if row_group != group or col_idx >= len(row["snapshots"]):
                        continue
                    column_snapshots.append(row["snapshots"][col_idx])
                grouped_bounds[(group, col_idx)] = _shared_bounds(*column_snapshots) if column_snapshots else None
    elif bounds_mode != "per_panel":
        raise ValueError(f"Unsupported bounds_mode: {bounds_mode}")

    for row_idx, row in enumerate(rows):
        color = row.get("color", "#1f77b4")
        row_title = row.get("row_title")
        snapshots = row["snapshots"]
        subtitles = row["titles"]

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(snapshots):
                ax.axis("off")
                continue
            panel_bounds = global_bounds
            if bounds_mode == "per_column" and column_bounds is not None:
                panel_bounds = column_bounds[col_idx]
            elif bounds_mode == "per_group_column":
                row_group = str(row.get("bounds_group", row_idx))
                panel_bounds = grouped_bounds[(row_group, col_idx)]
            elif bounds_mode == "per_panel":
                panel_bounds = _shared_bounds(snapshots[col_idx])
            _scatter(ax, snapshots[col_idx], subtitles[col_idx], color=color, bounds=panel_bounds)
            if row_title is not None and col_idx == 0:
                ax.set_ylabel(row_title)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def compute_plot_bounds(*arrays: np.ndarray) -> tuple[float, float, float, float]:
    return _shared_bounds(*arrays)


def _shared_bounds(*arrays: np.ndarray) -> tuple[float, float, float, float]:
    stacked = np.concatenate(arrays, axis=0)
    x_min, y_min = stacked.min(axis=0)
    x_max, y_max = stacked.max(axis=0)
    pad_x = max(0.05 * (x_max - x_min), 0.1)
    pad_y = max(0.05 * (y_max - y_min), 0.1)
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


def _scatter(
    ax,
    points: np.ndarray,
    title: str,
    color: str = "#1f77b4",
    bounds: tuple[float, float, float, float] | None = None,
) -> None:
    ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.55, color=color, edgecolors="none")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    ax.grid(alpha=0.15)


def _scatter_overlay(
    ax,
    original_points: np.ndarray,
    adversarial_points: np.ndarray,
    *,
    bounds: tuple[float, float, float, float] | None = None,
) -> None:
    ax.scatter(original_points[:, 0], original_points[:, 1], s=8, alpha=0.35, color="#1f77b4", edgecolors="none", label="orig")
    ax.scatter(adversarial_points[:, 0], adversarial_points[:, 1], s=8, alpha=0.35, color="#d62728", edgecolors="none", label="adv")

    deltas = adversarial_points - original_points
    n_show = min(64, original_points.shape[0])
    if n_show > 0:
        ax.quiver(
            original_points[:n_show, 0],
            original_points[:n_show, 1],
            deltas[:n_show, 0],
            deltas[:n_show, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            alpha=0.65,
            color="#2f2f2f",
        )

    ax.set_title("Overlay + displacement")
    ax.set_aspect("equal", adjustable="box")
    if bounds is not None:
        x_min, x_max, y_min, y_max = bounds
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.15)


def _scatter_amplified_overlay(
    ax,
    original_points: np.ndarray,
    adversarial_points: np.ndarray,
    *,
    viz_scale: float,
) -> None:
    deltas = adversarial_points - original_points
    amplified_points = original_points + viz_scale * deltas
    bounds = _shared_bounds(original_points, amplified_points)

    ax.scatter(original_points[:, 0], original_points[:, 1], s=8, alpha=0.35, color="#1f77b4", edgecolors="none", label="orig")
    ax.scatter(amplified_points[:, 0], amplified_points[:, 1], s=8, alpha=0.35, color="#ff7f0e", edgecolors="none", label="adv x scale")

    n_show = min(64, original_points.shape[0])
    if n_show > 0:
        scaled = viz_scale * deltas[:n_show]
        ax.quiver(
            original_points[:n_show, 0],
            original_points[:n_show, 1],
            scaled[:, 0],
            scaled[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            alpha=0.65,
            color="#2f2f2f",
        )

    x_min, x_max, y_min, y_max = bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Amplified displacement x{viz_scale:.1f}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.15)


def _compute_viz_scale(original_points: np.ndarray, adversarial_points: np.ndarray) -> float:
    deltas = adversarial_points - original_points
    shift = np.linalg.norm(deltas, axis=1)
    mean_shift = float(np.mean(shift)) if shift.size > 0 else 0.0
    if mean_shift <= 1e-12:
        return 1.0

    x_min, x_max, y_min, y_max = _shared_bounds(original_points)
    span = max(x_max - x_min, y_max - y_min)
    target_shift = max(0.15 * span, 0.1)
    viz_scale = target_shift / mean_shift
    return float(np.clip(viz_scale, 1.0, 50.0))
