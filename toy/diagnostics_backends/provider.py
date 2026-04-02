from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ..metrics import estimate_bayes_posterior_mean_mse, evaluate_baseline_gate
from ..plotting import plot_debug_losses, plot_forward_timestep_clouds, plot_forward_timestep_image_grids


@dataclass
class DiagnosticsBundle:
    """Diagnostics/visualization hooks for a given experiment family."""

    name: str
    estimate_bayes_terminal_mse: Callable[[object, object, float], Optional[float]]
    build_baseline_gate: Callable[[object, Dict[str, object]], Dict]
    plot_forward_backward_debug: Callable[..., None]
    plot_loss_debug: Callable[..., None]
    summarize_warnings: Callable[[object, Dict[str, object]], List[str]]


def build_diagnostics_bundle(cfg, dataset) -> DiagnosticsBundle:
    """Factory for experiment diagnostics backends."""

    diagnostics_kind = cfg.diagnostics_kind
    if diagnostics_kind == "auto":
        diagnostics_kind = dataset.name

    if diagnostics_kind == "toy_gmm":
        return _build_toy_gmm_diagnostics(dataset)
    if diagnostics_kind in ("image_folder", "mnist", "image_basic"):
        return _build_image_diagnostics()
    raise NotImplementedError(
        f"Unsupported diagnostics_kind='{cfg.diagnostics_kind}'. "
        "Add a new branch in toy/diagnostics_backends/provider.py for your experiment family."
    )


def _build_toy_gmm_diagnostics(dataset) -> DiagnosticsBundle:
    """Diagnostics backend for the existing 2D GMM toy."""

    centers_np = dataset.metadata.get("centers_np")

    def estimate_bayes(dataset_bundle, config, sigma: float) -> Optional[float]:
        if centers_np is None:
            return None
        return estimate_bayes_posterior_mean_mse(
            centers=centers_np,
            data_std=config.data_std,
            sigma=sigma,
            n_samples=50000,
            seed=config.seed,
        )

    def build_gate(config, gate_context: Dict[str, object]) -> Dict:
        return evaluate_baseline_gate(
            generated_mode_metrics=gate_context["generated_metrics"],
            endpoint_mode_metrics=gate_context["endpoint_metrics"],
            min_coverage=config.baseline_gate_min_coverage,
            max_generated_avg_min_dist=config.baseline_gate_max_generated_avg_min_dist,
            max_generated_p90_min_dist=config.baseline_gate_max_generated_p90_min_dist,
            max_endpoint_avg_min_dist=config.baseline_gate_max_endpoint_avg_min_dist,
            max_endpoint_p90_min_dist=config.baseline_gate_max_endpoint_p90_min_dist,
        )

    def summarize_warnings(config, metrics: Dict[str, object]) -> List[str]:
        sample_quality = metrics["sample_quality_debug"]
        generated_metrics = sample_quality["baseline_generated_metrics"]
        avg_dist = generated_metrics.get("avg_min_dist_to_mode")
        if avg_dist is None:
            return []
        if avg_dist > config.baseline_gate_max_generated_avg_min_dist:
            return [
                "baseline generated samples are still ring-like "
                f"(avg_min_dist_to_mode={avg_dist:.3f} > {config.baseline_gate_max_generated_avg_min_dist:.3f}).",
                "Increase baseline training steps or use larger batch for sharper mode separation.",
            ]
        return []

    return DiagnosticsBundle(
        name="toy_gmm",
        estimate_bayes_terminal_mse=estimate_bayes,
        build_baseline_gate=build_gate,
        plot_forward_backward_debug=plot_forward_timestep_clouds,
        plot_loss_debug=plot_debug_losses,
        summarize_warnings=summarize_warnings,
    )


def _build_image_diagnostics() -> DiagnosticsBundle:
    """Diagnostics backend for image-shaped toy-protocol experiments."""

    def estimate_bayes(dataset_bundle, config, sigma: float) -> Optional[float]:
        del dataset_bundle, config, sigma
        return None

    def build_gate(config, gate_context: Dict[str, object]) -> Dict:
        generated_metrics = gate_context["generated_metrics"]
        endpoint_metrics = gate_context["endpoint_metrics"]
        endpoint_recovery_mse = float(gate_context["endpoint_recovery_mse_mean"])
        checks = [
            {
                "name": "generated_global_std",
                "value": float(generated_metrics["global_std"]),
                "op": ">=",
                "threshold": float(config.image_gate_min_generated_std),
                "pass": bool(generated_metrics["global_std"] >= config.image_gate_min_generated_std),
            },
            {
                "name": "endpoint_global_std",
                "value": float(endpoint_metrics["global_std"]),
                "op": ">=",
                "threshold": float(config.image_gate_min_endpoint_std),
                "pass": bool(endpoint_metrics["global_std"] >= config.image_gate_min_endpoint_std),
            },
            {
                "name": "endpoint_recovery_mse_mean",
                "value": endpoint_recovery_mse,
                "op": "<=",
                "threshold": float(config.image_gate_max_endpoint_recovery_mse),
                "pass": bool(endpoint_recovery_mse <= config.image_gate_max_endpoint_recovery_mse),
            },
        ]
        failed = [c for c in checks if not c["pass"]]
        return {"passed": bool(len(failed) == 0), "checks": checks, "failed_checks": failed}

    def summarize_warnings(config, metrics: Dict[str, object]) -> List[str]:
        sample_quality = metrics["sample_quality_debug"]
        baseline_generated = sample_quality["baseline_generated_metrics"]
        warnings = []
        if baseline_generated["global_std"] < config.image_gate_min_generated_std:
            warnings.append(
                "baseline generated images have very low global_std "
                f"({baseline_generated['global_std']:.4f} < {config.image_gate_min_generated_std:.4f})."
            )
            warnings.append("This usually indicates collapse or near-constant outputs; increase baseline training first.")
        return warnings

    return DiagnosticsBundle(
        name="image_basic",
        estimate_bayes_terminal_mse=estimate_bayes,
        build_baseline_gate=build_gate,
        plot_forward_backward_debug=plot_forward_timestep_image_grids,
        plot_loss_debug=plot_debug_losses,
        summarize_warnings=summarize_warnings,
    )
