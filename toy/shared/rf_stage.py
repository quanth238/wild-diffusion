from __future__ import annotations


def resolve_rf_stage_steps(
    total_steps: int,
    stage1_fraction: float,
    *,
    reflow_start_step: int = 0,
) -> tuple[int, int]:
    """Resolve clean/WDRO RF stage-1 vs reflow steps.

    When ``reflow_start_step`` is positive, it is treated as the explicit
    continuation step where reflow begins. Otherwise we fall back to the legacy
    ``rf_stage1_fraction`` split.
    """

    total_steps_value = max(int(total_steps), 0)
    explicit_reflow_start = max(int(reflow_start_step), 0)
    if total_steps_value <= 0:
        return 0, 0
    if explicit_reflow_start > 0:
        stage1_steps = min(explicit_reflow_start, total_steps_value)
        return int(stage1_steps), int(max(total_steps_value - stage1_steps, 0))
    if total_steps_value <= 1:
        return int(total_steps_value), 0
    stage1_steps = int(round(float(total_steps_value) * float(stage1_fraction)))
    stage1_steps = max(1, min(stage1_steps, total_steps_value - 1))
    return int(stage1_steps), int(total_steps_value - stage1_steps)


def resolve_rf_cdro_stage_steps(
    total_steps: int,
    stage1_fraction: float,
    pair_source: str,
    *,
    reflow_start_step: int = 0,
) -> tuple[int, int]:
    """Resolve CDRO-RF stage-1 vs reflow steps under the staged pair source."""

    total_steps_value = max(int(total_steps), 0)
    mode = str(pair_source).strip().lower()
    if mode == "data_noise":
        return int(total_steps_value), 0
    if mode == "reflow":
        return 0, int(total_steps_value)
    return resolve_rf_stage_steps(
        int(total_steps_value),
        float(stage1_fraction),
        reflow_start_step=int(reflow_start_step),
    )
