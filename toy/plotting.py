from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history_baseline: Dict[str, list], history_robust: Dict[str, list], out_path: str) -> None:
    """Plot baseline/robust training curves for quick optimization diagnostics."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_baseline["loss"], label="baseline")
    axes[0].plot(history_robust["outer_loss"], label="trajectory_robust_constrained")
    axes[0].set_title("Outer Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Weighted MSE")
    axes[0].legend()

    axes[1].plot(history_robust["inner_obj"], label="inner_obj")
    if "delta_norm_ratio_mean" in history_robust:
        axes[1].plot(history_robust["delta_norm_ratio_mean"], label="delta_norm_ratio_mean")
    elif "energy" in history_robust:
        axes[1].plot(history_robust["energy"], label="energy")
    axes[1].set_title("Inner Terms")
    axes[1].set_xlabel("Step")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_sample_scatter(
    real_samples: np.ndarray,
    baseline_samples: np.ndarray,
    robust_samples: np.ndarray,
    centers: np.ndarray,
    out_path: str,
) -> None:
    """Compare real, baseline-generated, and robust-generated sample clouds."""

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ("Real Data", real_samples),
        ("Baseline EDM", baseline_samples),
        ("Trajectory-Robust (Energy)", robust_samples),
    ]

    for ax, (title, pts) in zip(axes, panels):
        ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.35)
        ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=80)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trajectory_examples(
    ref_paths: np.ndarray,
    ctrl_paths: np.ndarray,
    centers: np.ndarray,
    out_path: str,
    n_show: int = 12,
) -> None:
    """Overlay a few forward trajectories: reference vs controlled rollout."""

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    n = min(n_show, ref_paths.shape[0])
    for i in range(n):
        ax.plot(ref_paths[i, :, 0], ref_paths[i, :, 1], color="tab:blue", alpha=0.35, linewidth=1.0)
        ax.plot(ctrl_paths[i, :, 0], ctrl_paths[i, :, 1], color="tab:orange", alpha=0.7, linewidth=1.2)
        ax.scatter(ref_paths[i, 0, 0], ref_paths[i, 0, 1], color="black", s=8, alpha=0.8)

    ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=80)
    ax.set_title("Trajectory Examples (Blue=Ref, Orange=Controlled)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_forward_timestep_clouds(
    fwd_baseline_paths: np.ndarray,
    bwd_baseline_paths: np.ndarray,
    fwd_attack_paths: np.ndarray,
    bwd_attack_paths: np.ndarray,
    centers: np.ndarray,
    out_path: str,
    max_panels: int = 8,
) -> None:
    """4-row timestep cloud view: fwd/bwd for reference and attack terminals."""

    n_steps = fwd_baseline_paths.shape[1]
    n_show = min(max_panels, n_steps)
    idx_fwd = np.linspace(0, n_steps - 1, n_show, dtype=int)
    idx_rev = np.linspace(n_steps - 1, 0, n_show, dtype=int)
    fig, axes = plt.subplots(4, n_show, figsize=(2.7 * n_show, 11.0), squeeze=False)

    for j in range(n_show):
        k_fwd = idx_fwd[j]
        k_rev = idx_rev[j]
        row_axes = [axes[r, j] for r in range(4)]
        row_axes[0].scatter(
            fwd_baseline_paths[:, k_fwd, 0], fwd_baseline_paths[:, k_fwd, 1], s=4, alpha=0.45, color="tab:blue"
        )
        row_axes[1].scatter(
            bwd_baseline_paths[:, k_rev, 0], bwd_baseline_paths[:, k_rev, 1], s=4, alpha=0.45, color="tab:green"
        )
        row_axes[2].scatter(
            fwd_attack_paths[:, k_fwd, 0], fwd_attack_paths[:, k_fwd, 1], s=4, alpha=0.45, color="tab:orange"
        )
        row_axes[3].scatter(
            bwd_attack_paths[:, k_rev, 0], bwd_attack_paths[:, k_rev, 1], s=4, alpha=0.45, color="tab:red"
        )

        for ax in row_axes:
            ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=70)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)

        row_axes[0].set_title(f"Fwd Baseline k={k_fwd}")
        row_axes[1].set_title(f"Bwd Baseline k={k_rev}")
        row_axes[2].set_title(f"Fwd Attack k={k_fwd}")
        row_axes[3].set_title(f"Bwd Attack k={k_rev}")

    fig.suptitle(
        "Forward/Backward Debug View (Baseline vs Attack)",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_generated_reverse_timestep_clouds(
    baseline_rev_paths: np.ndarray,
    robust_rev_paths: np.ndarray,
    centers: np.ndarray,
    out_path: str,
    max_panels: int = 8,
) -> None:
    """Visualize reverse trajectories from Gaussian prior for baseline vs robust models."""

    n_steps = baseline_rev_paths.shape[1]
    n_show = min(max_panels, n_steps)
    idx_rev = np.linspace(n_steps - 1, 0, n_show, dtype=int)
    fig, axes = plt.subplots(2, n_show, figsize=(2.7 * n_show, 5.6), squeeze=False)

    for j in range(n_show):
        k = idx_rev[j]
        ax_b = axes[0, j]
        ax_r = axes[1, j]
        ax_b.scatter(baseline_rev_paths[:, k, 0], baseline_rev_paths[:, k, 1], s=4, alpha=0.45, color="tab:green")
        ax_r.scatter(robust_rev_paths[:, k, 0], robust_rev_paths[:, k, 1], s=4, alpha=0.45, color="tab:purple")

        for ax in [ax_b, ax_r]:
            ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=70)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)

        ax_b.set_title(f"Gen Bwd Baseline k={k}")
        ax_r.set_title(f"Gen Bwd Robust k={k}")

    fig.suptitle("Generated Reverse Process from Gaussian Prior", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_sample_trajectories(
    ref_paths: np.ndarray,
    ctrl_paths: np.ndarray,
    centers: np.ndarray,
    out_path: str,
    n_show: int = 12,
) -> None:
    """Small multiples of per-sample forward paths for fine-grained debugging."""

    n = min(n_show, ref_paths.shape[0])
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows), squeeze=False)
    axes = axes.reshape(-1)

    for i in range(n):
        ax = axes[i]
        ax.plot(ref_paths[i, :, 0], ref_paths[i, :, 1], "o-", color="tab:blue", alpha=0.7, linewidth=1.0, markersize=2.5)
        ax.plot(ctrl_paths[i, :, 0], ctrl_paths[i, :, 1], "o-", color="tab:orange", alpha=0.8, linewidth=1.0, markersize=2.5)
        ax.scatter(ref_paths[i, 0, 0], ref_paths[i, 0, 1], color="black", s=16, label="x0" if i == 0 else None)
        ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=55)
        ax.set_title(f"Sample #{i}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    for i in range(n, len(axes)):
        axes[i].axis("off")

    handles = [
        plt.Line2D([0], [0], color="tab:blue", marker="o", label="Ref forward"),
        plt.Line2D([0], [0], color="tab:orange", marker="o", label="Attack forward"),
        plt.Line2D([0], [0], color="black", marker="o", linestyle="", label="x0"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Per-Sample Forward Trajectories", y=0.985)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_denoise_error_curves(curves: Dict[str, list], out_path: str) -> None:
    """Plot denoising MSE versus forward timestep for all model/path pairings."""

    steps = curves["step"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(steps, curves["baseline_on_forward_baseline"], label="Baseline on Forward Baseline", linewidth=2.0)
    ax.plot(steps, curves["baseline_on_forward_attack"], label="Baseline on Forward Attack", linewidth=2.0)
    ax.plot(steps, curves["robust_on_forward_baseline"], label="Robust on Forward Baseline", linewidth=2.0)
    ax.plot(steps, curves["robust_on_forward_attack"], label="Robust on Forward Attack", linewidth=2.0)
    ax.set_xlabel("Forward Timestep")
    ax.set_ylabel("Denoise MSE to x0")
    ax.set_title("Denoising Error vs Forward Timestep")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_debug_losses(
    history_baseline: Dict[str, list],
    history_robust: Dict[str, list],
    denoise_curves: Dict[str, list],
    recovery_ref_curve: Dict[str, list],
    recovery_attack_curve: Dict[str, list],
    out_path: str,
) -> None:
    """Combined 2x2 debug panel for losses, inner stats, and recovery curves."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    ax00.plot(history_baseline["loss"], label="baseline_loss", linewidth=2.0)
    ax00.plot(history_robust["outer_loss"], label="robust_outer_loss", linewidth=1.8)
    ax00.set_title("Training Loss")
    ax00.set_xlabel("Step")
    ax00.set_ylabel("Weighted Denoise Loss")
    ax00.grid(alpha=0.25)
    ax00.legend()

    ax01.plot(history_robust["inner_obj"], label="inner_obj", linewidth=1.8)
    if "delta_norm_ratio_mean" in history_robust:
        ax01.plot(history_robust["delta_norm_ratio_mean"], label="delta_norm_ratio_mean", linewidth=1.8)
    elif "energy" in history_robust:
        ax01.plot(history_robust["energy"], label="energy", linewidth=1.8)
    ax01.set_title("Inner Objective Debug")
    ax01.set_xlabel("Step")
    ax01.grid(alpha=0.25)
    ax01.legend()

    x_step = denoise_curves["step"]
    ax10.plot(x_step, denoise_curves["baseline_on_forward_baseline"], label="baseline_on_fwd_baseline", linewidth=2.0)
    ax10.plot(x_step, denoise_curves["baseline_on_forward_attack"], label="baseline_on_fwd_attack", linewidth=2.0)
    ax10.set_title("Baseline Denoise Error vs Timestep")
    ax10.set_xlabel("Forward Timestep")
    ax10.set_ylabel("MSE to x0")
    ax10.grid(alpha=0.25)
    ax10.legend()

    ax11.plot(recovery_ref_curve["step"], recovery_ref_curve["x0_mse"], label="x0_mse_from_fwd_baseline", linewidth=2.0)
    ax11.plot(recovery_attack_curve["step"], recovery_attack_curve["x0_mse"], label="x0_mse_from_fwd_attack", linewidth=2.0)
    ax11.set_title("x0 Recovery MSE vs Terminal Timestep")
    ax11.set_xlabel("Terminal Timestep")
    ax11.set_ylabel("Pairwise MSE to x0")
    ax11.grid(alpha=0.25)
    ax11.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
