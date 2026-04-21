from __future__ import annotations


def resolve_rf_stage_steps(
    total_steps: int,
    stage1_fraction: float,
    *,
    reflow_start_step: int = 0,
) -> tuple[int, int]:
    """Resolve RF-family training as explicit teacher-pair reflow only.

    The current RF protocol is RF++-style: the shared EDM checkpoint is the
    frozen teacher, and the RF-family student trains directly on teacher
    synthetic pairs. Legacy stage-split knobs are accepted for backward CLI
    compatibility, but they no longer change the RF-family continuation split.
    """

    total_steps_value = max(int(total_steps), 0)
    if total_steps_value <= 0:
        return 0, 0
    del stage1_fraction, reflow_start_step
    return 0, int(total_steps_value)


def resolve_rf_cdro_stage_steps(
    total_steps: int,
    stage1_fraction: float,
    pair_source: str,
    *,
    reflow_start_step: int = 0,
) -> tuple[int, int]:
    """Resolve CDRO-RF as explicit shared-teacher reflow only."""

    total_steps_value = max(int(total_steps), 0)
    mode = str(pair_source).strip().lower()
    if mode not in {"reflow", "auto", "staged", ""}:
        raise ValueError(
            f"Unsupported RF CDRO pair source '{pair_source}'. Expected explicit shared-teacher reflow."
        )
    del stage1_fraction, reflow_start_step
    return 0, int(total_steps_value)
