import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def _import_wandb():
    try:
        import wandb  # type: ignore
    except ImportError:
        return None
    return wandb


def _warn(message: str) -> None:
    print(f"[wandb] warning: {message}", file=sys.stderr, flush=True)


def wandb_is_available() -> bool:
    return _import_wandb() is not None


def normalize_tags(tags: Optional[Sequence[str] | str]) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        parts = tags.split(",")
    else:
        parts = list(tags)
    return [str(tag).strip() for tag in parts if str(tag).strip()]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _to_jsonable(vars(value))
        except Exception:
            pass
    return str(value)


def init_wandb_run(
    *,
    enabled: bool,
    project: str,
    config: Optional[Mapping[str, Any]] = None,
    entity: Optional[str] = None,
    name: Optional[str] = None,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
    tags: Optional[Sequence[str] | str] = None,
    mode: str = "online",
    run_dir: Optional[str] = None,
) -> Any:
    if not enabled:
        return None

    wandb = _import_wandb()
    if wandb is None:
        _warn(
            "logging was requested, but the package is not installed; "
            "continuing with wandb disabled."
        )
        return None

    init_kwargs = {
        "project": str(project),
        "entity": entity or None,
        "name": name or None,
        "group": group or None,
        "job_type": job_type or None,
        "tags": normalize_tags(tags),
        "mode": str(mode),
        "config": _to_jsonable(config or {}),
        "dir": run_dir or os.getcwd(),
    }

    def _init_with_mode(selected_mode: str):
        return wandb.init(**{**init_kwargs, "mode": str(selected_mode)})

    try:
        return _init_with_mode(str(mode))
    except Exception as exc:
        requested_mode = str(mode).strip().lower()
        if requested_mode == "online":
            _warn(
                f"init failed in online mode ({type(exc).__name__}: {exc}); "
                "retrying in offline mode."
            )
            try:
                run = _init_with_mode("offline")
            except Exception as offline_exc:
                _warn(
                    "offline fallback also failed "
                    f"({type(offline_exc).__name__}: {offline_exc}); continuing with wandb disabled."
                )
                return None
            _warn("wandb is running in offline fallback mode.")
            return run

        _warn(f"init failed ({type(exc).__name__}: {exc}); continuing with wandb disabled.")
        return None


def log_metrics(run: Any, metrics: Mapping[str, Any], *, step: Optional[int] = None, commit: bool = True) -> None:
    if run is None:
        return
    payload = _to_jsonable(metrics)
    if not isinstance(payload, dict):
        return
    try:
        run.log(payload, step=step, commit=commit)
    except Exception as exc:
        _warn(f"log_metrics failed ({type(exc).__name__}: {exc}); continuing.")


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flat.update(_flatten_scalars(child, child_prefix))
        return flat
    if isinstance(value, (str, int, float, bool)) or value is None:
        flat[prefix] = value
    return flat


def update_summary(run: Any, payload: Mapping[str, Any], *, prefix: Optional[str] = None) -> None:
    if run is None:
        return
    normalized = _to_jsonable(payload)
    summary_payload = _flatten_scalars(normalized, prefix=prefix or "")
    try:
        for key, value in summary_payload.items():
            run.summary[key] = value
    except Exception as exc:
        _warn(f"update_summary failed ({type(exc).__name__}: {exc}); continuing.")


def log_series(run: Any, *, metric_name: str, values: Sequence[Any], start_step: int = 1) -> None:
    if run is None:
        return
    try:
        for idx, value in enumerate(values, start=start_step):
            if not isinstance(value, (int, float)):
                continue
            run.log({metric_name: float(value)}, step=int(idx))
    except Exception as exc:
        _warn(f"log_series failed ({type(exc).__name__}: {exc}); continuing.")


def log_artifact(
    run: Any,
    *,
    name: str,
    artifact_type: str,
    paths: Sequence[str],
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    if run is None:
        return
    wandb = _import_wandb()
    if wandb is None:
        return

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=_to_jsonable(metadata or {}))
    added_any = False
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        if path.is_dir():
            artifact.add_dir(str(path), name=path.name)
        else:
            artifact.add_file(str(path), name=path.name)
        added_any = True
    if added_any:
        try:
            run.log_artifact(artifact)
        except Exception as exc:
            _warn(f"log_artifact failed ({type(exc).__name__}: {exc}); continuing.")


def finish_run(run: Any, *, exit_code: int = 0) -> None:
    if run is None:
        return
    try:
        run.finish(exit_code=exit_code)
    except Exception as exc:
        _warn(f"finish_run failed ({type(exc).__name__}: {exc}); continuing.")
