from __future__ import annotations

import os
from typing import Any, Mapping, Optional

FRONTIER_AGGREGATE_ABS_MODE = "frontier_aggregate_abs"

ABS_AGGREGATION_MODE_ALIASES = {
    "exact_abs": "exact_abs",
    "exact": "exact_abs",
    "sum_abs": "exact_abs",
    FRONTIER_AGGREGATE_ABS_MODE: FRONTIER_AGGREGATE_ABS_MODE,
    "canonical_frontier_abs": FRONTIER_AGGREGATE_ABS_MODE,
    "frontier_abs": FRONTIER_AGGREGATE_ABS_MODE,
    "frontier_abs_of_sum": FRONTIER_AGGREGATE_ABS_MODE,
    "abs_of_sum": FRONTIER_AGGREGATE_ABS_MODE,
}


def normalize_abs_aggregation_mode(value: Any, *, default: str = "exact_abs") -> str:
    if value is None:
        value = default
    text = str(value).strip().lower()
    if not text:
        text = default
    try:
        return ABS_AGGREGATION_MODE_ALIASES[text]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(ABS_AGGREGATION_MODE_ALIASES.values())))
        raise ValueError(
            f"Unsupported abs aggregation mode '{value}'. Expected one of: {allowed}."
        ) from exc


def resolve_abs_aggregation_mode(
    default_mode: Any = "exact_abs",
    *,
    explicit_mode: Any = None,
    env: Optional[Mapping[str, Any]] = None,
) -> str:
    base_mode = normalize_abs_aggregation_mode(default_mode)
    if explicit_mode is not None:
        return normalize_abs_aggregation_mode(explicit_mode, default=base_mode)
    env_map: Mapping[str, Any] = os.environ if env is None else env
    env_mode = env_map.get("SCORING_ABS_AGGREGATION_MODE")
    if env_mode is not None and str(env_mode).strip():
        return normalize_abs_aggregation_mode(env_mode, default=base_mode)
    return base_mode


def is_frontier_aggregate_abs_mode(value: Any) -> bool:
    return normalize_abs_aggregation_mode(value) == FRONTIER_AGGREGATE_ABS_MODE
