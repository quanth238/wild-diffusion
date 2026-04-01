from types import ModuleType


SUPPORTED_METHOD_VERSIONS = ("v1", "1.1", "v1.1", "v2", "2.1", "v2.1", "wild")


def resolve_method_module(method_version: str) -> ModuleType:
    """Resolve the method-version module implementing rollout/train APIs."""

    if method_version == "v2":
        from .v2 import method as module

        return module
    if method_version in ("2.1", "v2.1"):
        from .v2_1 import method as module

        return module
    if method_version == "v1":
        from .v1 import method as module

        return module
    if method_version in ("1.1", "v1.1"):
        from .v1_1 import method as module

        return module
    if method_version == "wild":
        from .wild import method as module

        return module
    raise ValueError(
        f"Unsupported method_version='{method_version}'. "
        f"Supported versions: {', '.join(SUPPORTED_METHOD_VERSIONS)}"
    )
