"""Axis and array invariants for response-measurement capabilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def readonly_vector(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Return a finite, nonempty, C-contiguous read-only float vector."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    if array.flags.writeable or not array.flags.c_contiguous:
        array = np.array(array, dtype=np.float64, copy=True, order="C")
        array.setflags(write=False)
    return array


def readonly_bool_vector(
    values: Sequence[bool] | np.ndarray,
    *,
    size: int,
    name: str,
) -> np.ndarray:
    """Return a size-checked read-only Boolean vector."""

    array = np.asarray(values, dtype=bool).reshape(-1).copy()
    if array.size != int(size):
        raise ValueError(f"{name} must align with the route axis")
    array.setflags(write=False)
    return array


def readonly_matrix(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Return a finite, nonempty, C-contiguous read-only matrix."""

    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] == 0
        or array.shape[1] == 0
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite nonempty matrix")
    if array.flags.writeable or not array.flags.c_contiguous:
        array = np.array(array, dtype=np.float64, copy=True, order="C")
        array.setflags(write=False)
    return array


def validate_route_axis(
    *,
    route_ids: Sequence[str],
    route_costs: Sequence[float] | np.ndarray,
    protected: Sequence[bool] | np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Validate one complete physical route axis without workflow policy."""

    identifiers = tuple(str(item) for item in route_ids)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("route_ids must be finite, nonempty, and unique")
    costs = readonly_vector(route_costs, name="route_costs")
    if costs.size != len(identifiers) or np.any(costs <= 0.0):
        raise ValueError("route_costs must be positive and align with route_ids")
    protected_array = readonly_bool_vector(
        protected,
        size=len(identifiers),
        name="protected",
    )
    return identifiers, costs, protected_array


__all__ = [
    "readonly_bool_vector",
    "readonly_matrix",
    "readonly_vector",
    "validate_route_axis",
]
