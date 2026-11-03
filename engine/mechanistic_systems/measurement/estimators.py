"""Versioned route-gate estimators and compact risk-aware projections."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


ROUTE_GATE_V1 = "route_gate_v1"
RISK_AWARE_ROUTE_GATE_V2 = "risk_aware_route_gate_v2"
ROUTE_GATE_ESTIMATORS = frozenset({ROUTE_GATE_V1, RISK_AWARE_ROUTE_GATE_V2})


@dataclass(frozen=True)
class QueryOutputMetric:
    """Metric and signed direction on a declared query-output basis."""

    axis_ids: tuple[str, ...]
    metric_kind: str = "identity"
    diagonal: tuple[float, ...] = tuple()
    matrix: tuple[tuple[float, ...], ...] = tuple()
    directional_weights: tuple[float, ...] = tuple()

    def validate(self) -> None:
        size = len(self.axis_ids)
        if size <= 0 or len(set(self.axis_ids)) != size:
            raise ValueError("query output metric requires a nonempty unique axis")
        if self.metric_kind not in {"identity", "diagonal", "full_psd"}:
            raise ValueError("unsupported query output metric kind")
        if self.metric_kind == "diagonal" and (
            len(self.diagonal) != size
            or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in self.diagonal)
        ):
            raise ValueError("diagonal query metric must be finite, nonnegative, and axis aligned")
        if self.metric_kind == "full_psd":
            matrix = np.asarray(self.matrix, dtype=np.float64)
            if matrix.shape != (size, size) or not np.isfinite(matrix).all():
                raise ValueError("full query metric must be a finite square matrix")
            if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
                raise ValueError("full query metric must be symmetric")
            if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-9:
                raise ValueError("full query metric must be positive semidefinite")
        if self.directional_weights and (
            len(self.directional_weights) != size
            or any(not math.isfinite(float(value)) for value in self.directional_weights)
        ):
            raise ValueError("query direction must be finite and axis aligned")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "axis_ids": list(self.axis_ids),
            "metric_kind": self.metric_kind,
            "diagonal": list(self.diagonal),
            "matrix": [list(row) for row in self.matrix],
            "directional_weights": list(self.directional_weights),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "QueryOutputMetric":
        metric = cls(
            axis_ids=tuple(str(value) for value in payload.get("axis_ids", ()) or ()),
            metric_kind=str(payload.get("metric_kind", "identity")),
            diagonal=tuple(float(value) for value in payload.get("diagonal", ()) or ()),
            matrix=tuple(tuple(float(value) for value in row) for row in payload.get("matrix", ()) or ()),
            directional_weights=tuple(
                float(value) for value in payload.get("directional_weights", ()) or ()
            ),
        )
        metric.validate()
        return metric

    def sensitivity(self, derivative: np.ndarray) -> np.ndarray:
        """Return ``sqrt(g^T W g)`` over the final output-basis axis."""

        self.validate()
        values = np.asarray(derivative, dtype=np.float64)
        if values.shape[-1] != len(self.axis_ids):
            raise ValueError("gate derivative does not match query-output axis")
        if self.metric_kind == "identity":
            squared = np.sum(values * values, axis=-1)
        elif self.metric_kind == "diagonal":
            squared = np.sum(values * values * np.asarray(self.diagonal), axis=-1)
        else:
            matrix = np.asarray(self.matrix, dtype=np.float64)
            squared = np.einsum("...i,ij,...j->...", values, matrix, values)
        return np.sqrt(np.maximum(squared, 0.0))

    def signed_projection(self, derivative: np.ndarray) -> np.ndarray:
        self.validate()
        if not self.directional_weights:
            raise ValueError("signed route evidence requires a declared query direction")
        values = np.asarray(derivative, dtype=np.float64)
        if values.shape[-1] != len(self.axis_ids):
            raise ValueError("gate derivative does not match query-output axis")
        return values @ np.asarray(self.directional_weights, dtype=np.float64)


@dataclass(frozen=True)
class RouteGateEstimatorSpec:
    estimator_version: str = ROUTE_GATE_V1
    context_block_count: int = 1
    sign_probability_threshold: float = 0.95
    one_sided_z: float = 1.6448536269514722
    finite_gate_doses: tuple[float, ...] = tuple()
    notes: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.estimator_version not in ROUTE_GATE_ESTIMATORS:
            raise ValueError("unsupported route-gate estimator version")
        if int(self.context_block_count) <= 0:
            raise ValueError("context_block_count must be positive")
        if not 0.5 < float(self.sign_probability_threshold) < 1.0:
            raise ValueError("sign probability threshold must lie in (0.5, 1)")
        if not math.isfinite(float(self.one_sided_z)) or float(self.one_sided_z) < 0.0:
            raise ValueError("one-sided risk multiplier must be finite and nonnegative")
        if any(not 0.0 <= float(dose) <= 1.0 for dose in self.finite_gate_doses):
            raise ValueError("finite gate doses must lie in [0, 1]")
        if self.estimator_version == ROUTE_GATE_V1 and self.finite_gate_doses:
            raise ValueError("route_gate_v1 cannot add finite-gate measurements")
        if self.estimator_version == RISK_AWARE_ROUTE_GATE_V2 and self.context_block_count < 2:
            raise ValueError("risk-aware route gates require repeated context blocks")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "estimator_version": self.estimator_version,
            "context_block_count": int(self.context_block_count),
            "sign_probability_threshold": float(self.sign_probability_threshold),
            "one_sided_z": float(self.one_sided_z),
            "finite_gate_doses": list(self.finite_gate_doses),
            "notes": dict(self.notes),
        }


@dataclass(frozen=True)
class RiskAwareRouteEvidence:
    signed_mean: np.ndarray
    signed_standard_error: np.ndarray
    absolute_mean: np.ndarray
    absolute_standard_error: np.ndarray
    positive_probability: np.ndarray
    pruning_priority: np.ndarray
    partition_labels: np.ndarray


def risk_aware_route_evidence(
    *,
    count: np.ndarray,
    score_sum: np.ndarray,
    score_sum_squares: np.ndarray,
    absolute_score_sum: np.ndarray,
    absolute_score_sum_squares: np.ndarray,
    positive_count: np.ndarray,
    negative_count: np.ndarray,
    sign_probability_threshold: float = 0.95,
    one_sided_z: float = 1.6448536269514722,
) -> RiskAwareRouteEvidence:
    """Build a risk-protected route order and three-way sign partition."""

    n = np.asarray(count, dtype=np.float64)
    signed_sum = np.asarray(score_sum, dtype=np.float64)
    second_sum = np.asarray(score_sum_squares, dtype=np.float64)
    absolute_sum = np.asarray(absolute_score_sum, dtype=np.float64)
    absolute_second_sum = np.asarray(absolute_score_sum_squares, dtype=np.float64)
    if not (
        n.shape
        == signed_sum.shape
        == second_sum.shape
        == absolute_sum.shape
        == absolute_second_sum.shape
    ):
        raise ValueError("route evidence moment arrays must share a shape")
    if np.any(n < 2.0) or np.any(second_sum < 0.0) or np.any(absolute_second_sum < 0.0):
        raise ValueError(
            "risk-aware route evidence requires at least two blocks and nonnegative second moments"
        )
    signed_mean = signed_sum / n
    absolute_mean = absolute_sum / n
    signed_variance = np.maximum(
        (second_sum - n * signed_mean * signed_mean) / (n - 1.0),
        0.0,
    )
    absolute_variance = np.maximum(
        (absolute_second_sum - n * absolute_mean * absolute_mean) / (n - 1.0),
        0.0,
    )
    signed_se = np.sqrt(signed_variance / np.maximum(n, 1.0))
    absolute_se = np.sqrt(absolute_variance / np.maximum(n, 1.0))
    positive = np.asarray(positive_count, dtype=np.float64)
    negative = np.asarray(negative_count, dtype=np.float64)
    if positive.shape != n.shape or negative.shape != n.shape:
        raise ValueError("signed count arrays must match score moments")
    if np.any(positive < 0.0) or np.any(negative < 0.0) or np.any(positive + negative > n):
        raise ValueError("signed route counts must be nonnegative and cannot exceed block count")
    probability = posterior_positive_majority_probability(positive, negative)
    threshold = float(sign_probability_threshold)
    labels = np.where(
        probability >= threshold,
        1,
        np.where(probability <= 1.0 - threshold, -1, 0),
    ).astype(np.int8)
    priority = absolute_mean + float(one_sided_z) * absolute_se
    return RiskAwareRouteEvidence(
        signed_mean=signed_mean,
        signed_standard_error=signed_se,
        absolute_mean=absolute_mean,
        absolute_standard_error=absolute_se,
        positive_probability=probability,
        pruning_priority=priority,
        partition_labels=labels,
    )


def posterior_positive_majority_probability(
    positive_count: np.ndarray,
    negative_count: np.ndarray,
) -> np.ndarray:
    """Return ``P(p_positive > .5)`` under a Jeffreys beta working posterior.

    Deterministic context blocks are treated as exchangeable sign replicates
    for this opt-in v2 diagnostic. This is intentionally not the posterior mean
    sign frequency. The role decision is made only when posterior mass supports
    one side of the partition threshold; otherwise the route remains unresolved.
    """

    alpha = np.asarray(positive_count, dtype=np.float64) + 0.5
    beta = np.asarray(negative_count, dtype=np.float64) + 0.5
    if np.any(alpha <= 0.0) or np.any(beta <= 0.0):
        raise ValueError("sign counts must be nonnegative")
    try:
        from scipy.special import betainc
    except ImportError as error:  # pragma: no cover - production dependency gate.
        raise RuntimeError(
            "risk_aware_route_gate_v2 requires scipy.special.betainc; "
            "no approximate posterior fallback is permitted"
        ) from error
    return np.asarray(1.0 - betainc(alpha, beta, 0.5), dtype=np.float64)


def deterministic_context_blocks(
    context_ids: Sequence[str],
    *,
    block_count: int = 4,
    seed: int = 0,
) -> tuple[tuple[str, ...], ...]:
    """Assign contexts to stable balanced blocks without outcome information."""

    identifiers = tuple(str(value) for value in context_ids)
    if int(block_count) <= 0:
        raise ValueError("block_count must be positive")
    if len(identifiers) < int(block_count) or len(set(identifiers)) != len(identifiers):
        raise ValueError("context IDs must be unique and cover every requested block")
    ordered = sorted(
        identifiers,
        key=lambda value: hashlib.sha256(f"{int(seed)}\0{value}".encode("utf-8")).hexdigest(),
    )
    blocks: list[list[str]] = [[] for _ in range(int(block_count))]
    for index, context_id in enumerate(ordered):
        blocks[index % int(block_count)].append(context_id)
    return tuple(tuple(block) for block in blocks)


__all__ = [
    "QueryOutputMetric",
    "RISK_AWARE_ROUTE_GATE_V2",
    "ROUTE_GATE_ESTIMATORS",
    "ROUTE_GATE_V1",
    "RiskAwareRouteEvidence",
    "RouteGateEstimatorSpec",
    "deterministic_context_blocks",
    "risk_aware_route_evidence",
    "posterior_positive_majority_probability",
]
