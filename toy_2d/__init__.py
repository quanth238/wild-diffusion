"""CPU-friendly toy 2D diffusion baselines for continuous-data experiments."""

METHOD_CHOICES = ("baseline", "wdro", "cdro")


def normalize_method_name(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized not in METHOD_CHOICES:
        raise ValueError(f"Unsupported method: {method}")
    return normalized


def normalize_method_names(methods: list[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for method in methods:
        normalized = normalize_method_name(method)
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    return resolved


def normalize_method_config_keys(payload: dict[str, dict]) -> dict[str, dict]:
    normalized: dict[str, dict] = {}
    for method, config in payload.items():
        method_name = normalize_method_name(method)
        if not isinstance(config, dict):
            raise TypeError(f"Method config for {method_name} must be a dict.")
        normalized[method_name] = dict(config)
    return normalized
