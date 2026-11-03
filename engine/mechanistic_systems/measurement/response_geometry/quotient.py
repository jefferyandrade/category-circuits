"""Metric-aware nuisance quotient for matrix-valued response factors.

The quotient is fitted in the *declared task metric*, while its residual is
returned in semantic response coordinates.  Native implementations may supply
an explicit per-context metric whitener ``W`` or already-whitened route and
nuisance factors.  The latter is essential for large softmax-Fisher response
spaces where materializing ``W`` itself would defeat the factorized geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Sequence

import numpy as np

from mechanistic_systems.geometry.quotient import (
    IdentificationPolicy,
    pseudoinverse_spectrum,
)
from mechanistic_systems.geometry.metrics import validate_metric

from .tensors import ResponseFactorTensor


@dataclass(frozen=True, slots=True)
class ResponseQuotientDiagnostics:
    """Separate raw tangent identification from regularized solve usability.

    The legacy ``effective_rank`` and ``condition_number`` fields describe the
    regularized Gram that is actually pseudoinverted.  They never imply that
    the unregularized nuisance tangent system is identified.
    """

    nuisance_dimension: int
    effective_rank: int
    condition_number: float
    regularization: float
    refused: bool
    reason: str
    mean_absorbed_energy_fraction: float
    maximum_absorbed_energy_fraction: float
    fit_coordinate_mode: str = "metric_whitener"
    raw_effective_rank: int = 0
    raw_condition_number: float = float("inf")
    raw_singular_cutoff: float = 0.0
    regularized_effective_rank: int = 0
    regularized_condition_number: float = float("inf")
    regularized_singular_cutoff: float = 0.0
    raw_identified: bool = False
    regularized_solve_usable: bool = False
    identification_policy: str = "require_raw_identification"
    regularization_created_rank: bool = False

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        return {
            "nuisance_dimension": int(self.nuisance_dimension),
            "effective_rank": int(self.effective_rank),
            "condition_number": (
                float(self.condition_number)
                if math.isfinite(self.condition_number)
                else None
            ),
            "condition_number_is_finite": math.isfinite(self.condition_number),
            "regularization": float(self.regularization),
            "refused": bool(self.refused),
            "reason": str(self.reason),
            "mean_absorbed_energy_fraction": float(self.mean_absorbed_energy_fraction),
            "maximum_absorbed_energy_fraction": float(
                self.maximum_absorbed_energy_fraction
            ),
            "fit_coordinate_mode": str(self.fit_coordinate_mode),
            "raw_effective_rank": int(self.raw_effective_rank),
            "raw_condition_number": (
                float(self.raw_condition_number)
                if math.isfinite(self.raw_condition_number)
                else None
            ),
            "raw_condition_number_is_finite": math.isfinite(self.raw_condition_number),
            "raw_singular_cutoff": float(self.raw_singular_cutoff),
            "regularized_effective_rank": int(self.regularized_effective_rank),
            "regularized_condition_number": (
                float(self.regularized_condition_number)
                if math.isfinite(self.regularized_condition_number)
                else None
            ),
            "regularized_condition_number_is_finite": math.isfinite(
                self.regularized_condition_number
            ),
            "regularized_singular_cutoff": float(self.regularized_singular_cutoff),
            "raw_identified": bool(self.raw_identified),
            "regularized_solve_usable": bool(self.regularized_solve_usable),
            "identification_policy": str(self.identification_policy),
            "regularization_created_rank": bool(self.regularization_created_rank),
        }


class IntrinsicResponseFactorOperator:
    r"""Profile route-response factors against declared nuisance tangents.

    For each route ``r``, coefficients ``b_r`` minimize

    .. math::

        \sum_x \|W_x(J_r(x)-N_xb_r)\|^2 + \lambda\|b_r\|^2.

    The fit may be supplied in either of two exactly equivalent ways:

    1. semantic route/nuisance rows plus a metric whitener ``W``;
    2. semantic rows plus precomputed ``WJ`` and ``WN`` factors.

    The second form avoids materializing a potentially enormous output-space
    whitener for competitor-complete softmax-Fisher geometry.  In both cases,
    the returned residual remains in semantic response coordinates so target
    writing, competitor integration, and other typed response views survive
    nuisance quotienting.
    """

    def __init__(
        self,
        route_factor: ResponseFactorTensor,
        nuisance_rows: np.ndarray,
        *,
        metric_whitener: np.ndarray | None = None,
        whitened_route_rows: np.ndarray | None = None,
        whitened_nuisance_rows: np.ndarray | None = None,
        regularization: float = 1e-8,
        rcond: float = 1e-10,
        maximum_condition_number: float = 1e12,
        route_chunk_size: int = 4096,
        identification_policy: IdentificationPolicy = "require_raw_identification",
    ) -> None:
        nuisance = np.asarray(nuisance_rows, dtype=np.float64)
        if nuisance.ndim != 3 or nuisance.shape[:2] != route_factor.rows.shape[:2]:
            raise ValueError("nuisance rows must have [context,response,nuisance] axes")
        if nuisance.shape[2] <= 0 or not np.isfinite(nuisance).all():
            raise ValueError(
                "nuisance rows must be finite with positive nuisance dimension"
            )
        direct_requested = (
            whitened_route_rows is not None or whitened_nuisance_rows is not None
        )
        if direct_requested and (
            whitened_route_rows is None or whitened_nuisance_rows is None
        ):
            raise ValueError(
                "prewhitened response quotient requires both route and nuisance factors"
            )
        if direct_requested and metric_whitener is not None:
            raise ValueError(
                "supply either a metric whitener or prewhitened factors, not both"
            )

        if direct_requested:
            whitened_route = np.asarray(whitened_route_rows, dtype=np.float64)
            whitened_nuisance = np.asarray(whitened_nuisance_rows, dtype=np.float64)
            if (
                whitened_route.ndim != 3
                or whitened_nuisance.ndim != 3
                or whitened_route.shape[0] != route_factor.sample_count
                or whitened_nuisance.shape[0] != route_factor.sample_count
                or whitened_route.shape[1] != whitened_nuisance.shape[1]
                or whitened_route.shape[2] != route_factor.route_count
                or whitened_nuisance.shape[2] != nuisance.shape[2]
                or not np.isfinite(whitened_route).all()
                or not np.isfinite(whitened_nuisance).all()
            ):
                raise ValueError(
                    "prewhitened factors require finite [context,metric_factor,route/nuisance] axes"
                )
            whitener = None
            fit_mode = "prewhitened_task_metric_factor"
        else:
            whitened_route = None
            whitened_nuisance = None
            if metric_whitener is None:
                n, d, _ = route_factor.rows.shape
                whitener = np.broadcast_to(
                    np.eye(d, dtype=np.float64), (n, d, d)
                ).copy()
                fit_mode = "explicit_euclidean_whitener"
            else:
                whitener = np.asarray(metric_whitener, dtype=np.float64)
                if (
                    whitener.ndim != 3
                    or whitener.shape[0] != route_factor.sample_count
                    or whitener.shape[2] != route_factor.response_dimension
                    or not np.isfinite(whitener).all()
                ):
                    raise ValueError(
                        "metric whitener must have finite [context,factor,response] axes"
                    )
                fit_mode = "explicit_task_metric_whitener"
        if (
            not math.isfinite(float(regularization))
            or not math.isfinite(float(rcond))
            or not math.isfinite(float(maximum_condition_number))
            or regularization < 0.0
            or rcond < 0.0
            or maximum_condition_number <= 0.0
            or route_chunk_size <= 0
        ):
            raise ValueError("response quotient controls are invalid")
        if identification_policy not in {
            "require_raw_identification",
            "allow_regularized_solve",
        }:
            raise ValueError("response quotient identification_policy is invalid")

        self.route_factor = route_factor
        self.nuisance_rows = np.array(nuisance, copy=True, order="C")
        self.metric_whitener = (
            None if whitener is None else np.array(whitener, copy=True, order="C")
        )
        self._direct_whitened_route = (
            None
            if whitened_route is None
            else np.array(whitened_route, copy=True, order="C")
        )
        self._direct_whitened_nuisance = (
            None
            if whitened_nuisance is None
            else np.array(whitened_nuisance, copy=True, order="C")
        )
        self.fit_coordinate_mode = fit_mode
        self.regularization = float(regularization)
        self.rcond = float(rcond)
        self.maximum_condition_number = float(maximum_condition_number)
        self.route_chunk_size = int(route_chunk_size)
        self.identification_policy = str(identification_policy)

    @cached_property
    def _sqrt_context_weights(self) -> np.ndarray:
        """Square-root design weights for the task-metric least-squares fit.

        ``ResponseFactorTensor.context_weights`` are normalized to sum to one.
        Multiplying every context row by ``sqrt(weight)`` makes the quotient
        objective exactly consistent with the response moments and preserves
        the historical ``1/n`` objective under equal weights.
        """

        values = np.sqrt(
            np.asarray(self.route_factor.context_weights, dtype=np.float64)
        )
        values = np.array(values, copy=True, order="C")
        values.setflags(write=False)
        return values

    @cached_property
    def _whitened_nuisance_tensor(self) -> np.ndarray:
        if self._direct_whitened_nuisance is not None:
            values = self._direct_whitened_nuisance
        else:
            assert self.metric_whitener is not None
            values = np.einsum(
                "nkd,ndp->nkp", self.metric_whitener, self.nuisance_rows, optimize=True
            )
        return self._sqrt_context_weights[:, None, None] * values

    @cached_property
    def _whitened_nuisance(self) -> np.ndarray:
        values = self._whitened_nuisance_tensor
        return values.reshape(-1, values.shape[2])

    def _whitened_route_chunk(self, indices: np.ndarray) -> np.ndarray:
        if self._direct_whitened_route is not None:
            values = self._direct_whitened_route[:, :, indices]
        else:
            assert self.metric_whitener is not None
            raw = self.route_factor.rows[:, :, indices]
            values = np.einsum("nkd,ndr->nkr", self.metric_whitener, raw, optimize=True)
        return self._sqrt_context_weights[:, None, None] * values

    @cached_property
    def _raw_gram(self) -> np.ndarray:
        nuisance = self._whitened_nuisance
        gram = nuisance.T @ nuisance
        return validate_metric(0.5 * (gram + gram.T))

    @cached_property
    def _regularized_gram(self) -> np.ndarray:
        gram = self._raw_gram.copy()
        if self.regularization:
            gram = gram + self.regularization * np.eye(gram.shape[0])
        return validate_metric(0.5 * (gram + gram.T))

    @property
    def _gram(self) -> np.ndarray:
        """Compatibility spelling for the system actually pseudoinverted."""

        return self._regularized_gram

    @cached_property
    def _inverse(self) -> np.ndarray:
        return np.linalg.pinv(
            self._regularized_gram,
            rcond=self.rcond,
            hermitian=True,
        )

    def coefficients(
        self, route_indices: Sequence[int] | np.ndarray | None = None
    ) -> np.ndarray:
        if route_indices is None:
            indices = np.arange(self.route_factor.route_count, dtype=np.int64)
        else:
            indices = np.asarray(
                tuple(int(item) for item in route_indices), dtype=np.int64
            )
        if (
            indices.size == 0
            or np.any(indices < 0)
            or np.any(indices >= self.route_factor.route_count)
            or np.unique(indices).size != indices.size
        ):
            raise ValueError("quotient route indices are invalid")
        nuisance = self._whitened_nuisance
        coefficients = np.zeros(
            (self.nuisance_rows.shape[2], indices.size), dtype=np.float64
        )
        for start in range(0, indices.size, self.route_chunk_size):
            chunk = indices[start : start + self.route_chunk_size]
            whitened = self._whitened_route_chunk(chunk).reshape(-1, chunk.size)
            cross = nuisance.T @ whitened
            coefficients[:, start : start + chunk.size] = self._inverse @ cross
        return coefficients

    def _metric_residual_chunk(
        self, indices: np.ndarray, coefficients: np.ndarray
    ) -> np.ndarray:
        return self._whitened_route_chunk(indices) - np.einsum(
            "nkp,pr->nkr", self._whitened_nuisance_tensor, coefficients, optimize=True
        )

    @cached_property
    def diagnostics(self) -> ResponseQuotientDiagnostics:
        raw = pseudoinverse_spectrum(self._raw_gram, rcond=self.rcond)
        regularized = pseudoinverse_spectrum(
            self._regularized_gram,
            rcond=self.rcond,
        )
        raw_identified = (
            raw.full_rank
            and math.isfinite(raw.condition_number)
            and raw.condition_number <= self.maximum_condition_number
        )
        regularized_usable = (
            regularized.full_rank
            and math.isfinite(regularized.condition_number)
            and regularized.condition_number <= self.maximum_condition_number
        )
        refused = (
            not raw_identified
            if self.identification_policy == "require_raw_identification"
            else not regularized_usable
        )

        routes = self.route_factor.route_count
        original_energy = np.empty(routes, dtype=np.float64)
        residual_energy = np.empty(routes, dtype=np.float64)
        for start in range(0, routes, self.route_chunk_size):
            stop = min(routes, start + self.route_chunk_size)
            indices = np.arange(start, stop, dtype=np.int64)
            coefficients = self.coefficients(indices)
            original = self._whitened_route_chunk(indices)
            residual = self._metric_residual_chunk(indices, coefficients)
            original_energy[start:stop] = np.sum(
                np.sum(original * original, axis=1), axis=0
            )
            residual_energy[start:stop] = np.sum(
                np.sum(residual * residual, axis=1), axis=0
            )
        absorbed = np.divide(
            np.maximum(original_energy - residual_energy, 0.0),
            np.maximum(original_energy, 1e-15),
        )
        return ResponseQuotientDiagnostics(
            nuisance_dimension=int(self.nuisance_rows.shape[2]),
            effective_rank=regularized.effective_rank,
            condition_number=regularized.condition_number,
            regularization=self.regularization,
            refused=refused,
            reason=(
                "nuisance quotient is identified in the declared task metric"
                if raw_identified
                else (
                    "raw nuisance system is rank-deficient at the pseudoinverse cutoff"
                    if self.identification_policy == "require_raw_identification"
                    and not raw.full_rank
                    else (
                        "raw nuisance system exceeds maximum condition number"
                        if self.identification_policy == "require_raw_identification"
                        else (
                            "regularized solve is usable under explicit policy; raw "
                            "nuisance identification is not established"
                            if regularized_usable
                            else (
                                "regularized nuisance system is rank-deficient at the "
                                "pseudoinverse cutoff"
                                if not regularized.full_rank
                                else "regularized nuisance system exceeds maximum condition number"
                            )
                        )
                    )
                )
            ),
            mean_absorbed_energy_fraction=float(np.mean(absorbed)),
            maximum_absorbed_energy_fraction=float(np.max(absorbed)),
            fit_coordinate_mode=self.fit_coordinate_mode,
            raw_effective_rank=raw.effective_rank,
            raw_condition_number=raw.condition_number,
            raw_singular_cutoff=raw.singular_cutoff,
            regularized_effective_rank=regularized.effective_rank,
            regularized_condition_number=regularized.condition_number,
            regularized_singular_cutoff=regularized.singular_cutoff,
            raw_identified=raw_identified,
            regularized_solve_usable=regularized_usable,
            identification_policy=self.identification_policy,
            regularization_created_rank=(
                regularized.effective_rank > raw.effective_rank
            ),
        )

    @cached_property
    def _residualized_factor(self) -> ResponseFactorTensor:
        # Compute once.  v3/v4 repeatedly query diagnostics, action moments, and
        # candidate paths; rebuilding the O(n*d*K) semantic residual at every call
        # would erase the tractability gained by screening before quotienting.
        diagnostics = self.diagnostics
        condition = diagnostics.regularized_condition_number
        routes = self.route_factor.route_count
        residual = np.empty_like(self.route_factor.rows)
        for start in range(0, routes, self.route_chunk_size):
            stop = min(routes, start + self.route_chunk_size)
            indices = np.arange(start, stop, dtype=np.int64)
            coefficients = self.coefficients(indices)
            fitted = np.einsum(
                "ndp,pr->ndr", self.nuisance_rows, coefficients, optimize=True
            )
            residual[:, :, start:stop] = (
                self.route_factor.rows[:, :, start:stop] - fitted
            )
        return ResponseFactorTensor(
            channel_ids=self.route_factor.channel_ids,
            route_ids=self.route_factor.route_ids,
            context_ids=self.route_factor.context_ids,
            rows=residual,
            response_spec_hash=self.route_factor.response_spec_hash,
            metric_id=self.route_factor.metric_id,
            context_weights=self.route_factor.context_weights,
            projection_residual_bound=self.route_factor.projection_residual_bound,
            metadata={
                **dict(self.route_factor.metadata),
                "nuisance_quotiented": True,
                "nuisance_dimension": int(self.nuisance_rows.shape[2]),
                "regularization": self.regularization,
                "condition_number": condition if math.isfinite(condition) else None,
                "condition_number_is_finite": math.isfinite(condition),
                "raw_condition_number": (
                    diagnostics.raw_condition_number
                    if math.isfinite(diagnostics.raw_condition_number)
                    else None
                ),
                "raw_condition_number_is_finite": math.isfinite(
                    diagnostics.raw_condition_number
                ),
                "raw_singular_cutoff": diagnostics.raw_singular_cutoff,
                "regularized_singular_cutoff": diagnostics.regularized_singular_cutoff,
                "raw_nuisance_identified": diagnostics.raw_identified,
                "regularized_solve_usable": diagnostics.regularized_solve_usable,
                "identification_policy": diagnostics.identification_policy,
                "semantic_response_coordinates_preserved": True,
                "quotient_fitted_in_declared_task_metric": True,
                "fit_coordinate_mode": self.fit_coordinate_mode,
                "context_weights_applied_in_quotient_fit": True,
                "residual_factor_cached": True,
            },
        )

    def residualized_factor(
        self, *, refuse_unidentified: bool = True
    ) -> ResponseFactorTensor:
        diagnostics = self.diagnostics
        if refuse_unidentified and diagnostics.refused:
            raise RuntimeError(diagnostics.reason)
        if diagnostics.refused:
            factor = self._residualized_factor
            return replace(
                factor,
                metadata={
                    **dict(factor.metadata),
                    "quotient_refusal_explicitly_bypassed": True,
                    "quotient_refusal_reason": diagnostics.reason,
                    "quotient_result_not_promotable_as_identified": True,
                },
            )
        return self._residualized_factor

    def legacy_residualized_factor(
        self, *, refuse_unidentified: bool = True
    ) -> ResponseFactorTensor:
        """Reproduce the retired v3/v4 factor hash domain exactly.

        Historical state hashes included the residual factor's metadata.  The
        repaired capability records raw and regularized identification
        separately, so its metadata is intentionally different.  This helper
        reconstructs only the old metadata and old regularized-Gram refusal
        rule; it is for immutable replay identities, not scientific use.
        """

        eigenvalues = np.linalg.eigvalsh(self._regularized_gram)
        positive = eigenvalues[eigenvalues > max(self.rcond, 1e-15)]
        effective_rank = int(positive.size)
        condition = (
            float(np.max(positive) / np.min(positive))
            if positive.size
            else float("inf")
        )
        refused = (
            effective_rank == 0
            or not math.isfinite(condition)
            or condition > self.maximum_condition_number
        )
        if refuse_unidentified and refused:
            raise RuntimeError(
                "nuisance quotient is rank-deficient or ill-conditioned"
            )
        return replace(
            self._residualized_factor,
            metadata={
                **dict(self.route_factor.metadata),
                "nuisance_quotiented": True,
                "nuisance_dimension": int(self.nuisance_rows.shape[2]),
                "regularization": self.regularization,
                "condition_number": condition,
                "semantic_response_coordinates_preserved": True,
                "quotient_fitted_in_declared_task_metric": True,
                "fit_coordinate_mode": self.fit_coordinate_mode,
                "context_weights_applied_in_quotient_fit": True,
                "residual_factor_cached": True,
            },
        )


__all__ = ["IntrinsicResponseFactorOperator", "ResponseQuotientDiagnostics"]
