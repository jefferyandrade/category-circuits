"""Compatibility facade for measurement-owned step observables."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_OWNER_MODULE = "mechanistic_systems.measurement.step_observables"
__all__ = [
    "CONTEXT_IDENTITY_DRAW",
    "CONTEXT_IDENTITY_SOURCE",
    "PREACTIVATION_PERCENTILES",
    "STEP_CONTEXT_DELTA_FIXED_BYTES_PER_ROW",
    "STEP_CONTEXT_DELTA_MAX_BYTES",
    "STEP_CONTEXT_DELTA_SIDECAR_FILENAME",
    "STEP_CONTEXT_DELTA_SIDECAR_SCHEMA",
    "STEP_CONTEXT_DELTA_SIDECAR_SCHEMA_V1",
    "STEP_OBSERVABLES_ENV",
    "StepContextDeltaAccumulator",
    "decode_step_context_deltas",
    "decode_step_decision_moments",
    "step_observables_enabled",
    "summarize_residual_step_observables",
]


def __getattr__(name: str) -> Any:
    owner = import_module(_OWNER_MODULE)
    try:
        value = getattr(owner, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
