"""CPU-friendly toy 2D diffusion baselines for continuous-data experiments."""

LEGACY_METHOD_CHOICES = ("baseline", "wdro", "cdro")
COMPARISON_METHOD_CHOICES = LEGACY_METHOD_CHOICES + ("cdro_markov",)
METHOD_CHOICES = LEGACY_METHOD_CHOICES


def normalize_method_name(method: str, *, comparison: bool = False) -> str:
    normalized = str(method).strip().lower()
    allowed = COMPARISON_METHOD_CHOICES if comparison else LEGACY_METHOD_CHOICES
    if normalized not in allowed:
        raise ValueError(f"Unsupported method: {method}")
    return normalized


def normalize_method_names(methods: list[str], *, comparison: bool = False) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for method in methods:
        normalized = normalize_method_name(method, comparison=comparison)
        if normalized in seen:
            continue
        seen.add(normalized)
        resolved.append(normalized)
    return resolved


def normalize_method_config_keys(payload: dict[str, dict], *, comparison: bool = False) -> dict[str, dict]:
    normalized: dict[str, dict] = {}
    for method, config in payload.items():
        method_name = normalize_method_name(method, comparison=comparison)
        if not isinstance(config, dict):
            raise TypeError(f"Method config for {method_name} must be a dict.")
        normalized[method_name] = dict(config)
    return normalized


def normalize_comparison_method_names(methods: list[str]) -> list[str]:
    return normalize_method_names(methods, comparison=True)


def normalize_comparison_method_config_keys(payload: dict[str, dict]) -> dict[str, dict]:
    return normalize_method_config_keys(payload, comparison=True)
