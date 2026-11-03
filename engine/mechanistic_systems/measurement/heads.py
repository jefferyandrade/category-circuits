"""Canonical route-gate score heads and deterministic unit contractions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ROUTE_SCORE_HEAD = "route_scores"
ROUTE_ABS_SCORE_HEAD = "route_abs_scores"
UNIT_SCORE_HEAD = "unit_scores"
UNIT_ABS_SCORE_HEAD = "unit_abs_scores"
CORE_ROUTE_SCORE_HEADS = (
    ROUTE_SCORE_HEAD,
    ROUTE_ABS_SCORE_HEAD,
    UNIT_SCORE_HEAD,
    UNIT_ABS_SCORE_HEAD,
)


def require_canonical_score_head(head: str) -> str:
    token = str(head)
    if token not in CORE_ROUTE_SCORE_HEADS:
        raise ValueError(f"unsupported canonical score head: {token!r}")
    return token


def unit_reduction_from_route_scores(
    route_scores: Mapping[str, Any],
    *,
    absolute: bool,
) -> dict[str, Any]:
    """Reduce incoming routes to their downstream-unit sufficient statistic."""

    import torch

    reduced: dict[str, Any] = {}
    for layer, tensor in route_scores.items():
        value = tensor.detach().to(dtype=torch.float32)
        if absolute:
            value = value.abs()
        if value.ndim > 1:
            value = value.sum(dim=-1)
        reduced[str(layer)] = value.contiguous()
    return reduced


__all__ = [
    "CORE_ROUTE_SCORE_HEADS",
    "ROUTE_ABS_SCORE_HEAD",
    "ROUTE_SCORE_HEAD",
    "UNIT_ABS_SCORE_HEAD",
    "UNIT_SCORE_HEAD",
    "require_canonical_score_head",
    "unit_reduction_from_route_scores",
]
