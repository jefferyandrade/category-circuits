"""Identity-preserving facade for neutral hard-mask action contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_OWNER_MODULE = "mechanistic_systems.contracts.hard_mask_actions"
__all__ = [
    "ActionCandidate",
    "HardMask",
    "RemovedRouteIndexSet",
    "candidate_from_removed_indices",
    "candidate_materialization_limit",
    "deduplicate_candidates",
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
