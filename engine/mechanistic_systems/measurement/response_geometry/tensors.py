"""Typed matrix-valued interventional response-measurement geometry.

The primitive scientific object is the differential of a declared vector-valued
response in *physical route-gate coordinates*.  Scalar saliency values are
therefore not stored as competing first-class scores.  They arise only by
contracting the retained response moments with a frozen task-safe response
constraint.

These tensors are capability objects, not workflow versions.  Marginal,
coupled, quotient, and finite-transition workflows may specialize them, but
the tensors themselves carry only the declared response coordinates, route
axis, sampling weights, normalization, and measurement lineage.

The response ellipsoids represented here describe variability over the declared
context distribution.  They are *not* statistical confidence regions unless a
separate finite-sample uncertainty construction is explicitly supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from mechanistic_systems.core.axis_identity import (
    canonical_unique_string_axis,
    ordered_string_axis_hash,
    resolve_packed_string_axis,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.measurement import MeasurementOutcome


class ResponseBoundaryDirection(Protocol):
    direction_id: str
    vector: Sequence[float]
    lower_tolerance: float


class ResponseConstraint(Protocol):
    output_dimension: int
    directions: Sequence[ResponseBoundaryDirection]
    contextual_robustness_multiplier: float
    model_uncertainty_multiplier: float
    estimation_uncertainty_multiplier: float
    energy_metric: Sequence[Sequence[float]]
    energy_tolerance: float | None
    maximum_accepted_risk: float


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _readonly_array(value: np.ndarray, *, ndim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    array = np.array(array, copy=True, order="C")
    array.setflags(write=False)
    return array


def _symmetric_psd(
    value: np.ndarray,
    *,
    name: str,
    tolerance: float = 1e-8,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.isfinite(matrix).all()
    ):
        raise ValueError(f"{name} must be a finite square matrix")
    matrix = 0.5 * (matrix + matrix.T)
    if matrix.size:
        eigenvalues = np.linalg.eigvalsh(matrix)
        minimum = float(np.min(eigenvalues))
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    else:
        minimum = 0.0
        scale = 1.0
    if minimum < -float(tolerance) * scale:
        raise ValueError(f"{name} is materially non-positive-semidefinite")
    if minimum < 0.0:
        matrix = matrix + (-minimum + np.finfo(np.float64).eps * scale) * np.eye(
            matrix.shape[0]
        )
    matrix = np.array(matrix, copy=True, order="C")
    matrix.setflags(write=False)
    return matrix


def _batched_symmetric_psd(
    value: np.ndarray,
    *,
    name: str,
    tolerance: float = 1e-8,
    chunk_size: int = 65_536,
) -> np.ndarray:
    """Vectorized route-local PSD validation with bounded workspace.

    Response moments carry one small matrix per physical route. Calling the
    scalar validator in a Python loop makes a one-channel, million-route v1
    field CPU-bound for minutes. This routine preserves the same per-matrix
    tolerance and diagonal shift while validating route chunks in NumPy.
    """

    matrices = np.asarray(value, dtype=np.float64)
    if (
        matrices.ndim != 3
        or matrices.shape[1] != matrices.shape[2]
        or not np.isfinite(matrices).all()
        or int(chunk_size) <= 0
    ):
        raise ValueError(f"{name} must be finite square matrices")
    symmetric = 0.5 * (matrices + np.swapaxes(matrices, 1, 2))
    result = np.array(symmetric, copy=True, order="C")
    dimension = int(result.shape[1])
    if dimension == 1:
        diagonal = result[:, 0, 0]
        scale = np.maximum(1.0, np.abs(diagonal))
        bad = diagonal < -float(tolerance) * scale
        if np.any(bad):
            first = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"{name} matrix {first} is materially non-positive-semidefinite"
            )
        result[:, 0, 0] = np.where(
            diagonal < 0.0,
            np.finfo(np.float64).eps * scale,
            diagonal,
        )
        result.setflags(write=False)
        return result
    identity = np.eye(dimension, dtype=np.float64)
    for start in range(0, result.shape[0], int(chunk_size)):
        stop = min(result.shape[0], start + int(chunk_size))
        eigenvalues = np.linalg.eigvalsh(result[start:stop])
        minimum = np.min(eigenvalues, axis=1)
        scale = np.maximum(1.0, np.max(np.abs(eigenvalues), axis=1))
        bad = minimum < -float(tolerance) * scale
        if np.any(bad):
            first = start + int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"{name} matrix {first} is materially non-positive-semidefinite"
            )
        shift = np.where(
            minimum < 0.0,
            -minimum + np.finfo(np.float64).eps * scale,
            0.0,
        )
        result[start:stop] += shift[:, None, None] * identity[None, :, :]
    result.setflags(write=False)
    return result


def _normalized_context_weights(
    value: Sequence[float] | np.ndarray | None,
    count: int,
) -> np.ndarray:
    if count <= 0:
        raise ValueError("context count must be positive")
    if value is None:
        weights = np.full(count, 1.0 / float(count), dtype=np.float64)
    else:
        weights = np.asarray(value, dtype=np.float64).reshape(-1)
        if (
            weights.shape != (count,)
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
        ):
            raise ValueError(
                "context_weights must be finite, nonnegative, and align with contexts"
            )
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("context_weights must have positive total mass")
        weights = weights / total
    weights = np.array(weights, copy=True, order="C")
    weights.setflags(write=False)
    return weights


def _weighted_mean_standard_error(
    rows: np.ndarray, weights: np.ndarray, mean: np.ndarray
) -> np.ndarray:
    """Design-based plug-in SE for a weighted mean.

    This is an estimator-uncertainty diagnostic, deliberately distinct from the
    contextual covariance represented by the response geometry.  It is exact
    for equal-weight independent rows only under the usual sampling model; its
    provenance is retained so claim-bearing runs can replace it with a stronger
    bound without changing the geometric object.
    """

    centered = rows - mean
    if rows.shape[0] <= 1:
        return np.zeros_like(mean)
    # Effective-sample correction for normalized unequal weights.
    sum_sq = float(np.sum(weights * weights))
    denominator = max(1e-15, 1.0 - sum_sq)
    weighted_variance = (
        np.einsum("n,n...->...", weights, centered * centered) / denominator
    )
    return np.sqrt(np.maximum(weighted_variance, 0.0) * sum_sq)


def _weighted_mean_estimator_covariance(
    rows: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Plug-in covariance of a weighted vector mean.

    ``rows`` must have shape ``[context, response]``.  The returned matrix is
    the estimator covariance of the weighted mean, not the contextual response
    covariance.  Keeping these objects distinct is load-bearing: contextual
    dispersion belongs to the response geometry, whereas this matrix prices
    uncertainty in the estimated response mean.

    The formula agrees with the diagonal returned by
    :func:`_weighted_mean_standard_error` and reduces to ``sample_covariance/n``
    for equal weights.
    """

    values = np.asarray(rows, dtype=np.float64)
    center = np.asarray(mean, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != center.size:
        raise ValueError(
            "weighted mean estimator covariance requires [context,response] rows"
        )
    if values.shape[0] <= 1:
        return np.zeros((center.size, center.size), dtype=np.float64)
    centered = values - center[None, :]
    sum_sq = float(np.sum(weights * weights))
    denominator = max(1e-15, 1.0 - sum_sq)
    row_covariance = (
        np.einsum("n,nd,ne->de", weights, centered, centered, optimize=True)
        / denominator
    )
    return _symmetric_psd(
        row_covariance * sum_sq,
        name="weighted mean estimator covariance",
        tolerance=1e-6,
    )


def _validate_action(
    indices: Sequence[int] | np.ndarray,
    coefficients: Sequence[float] | np.ndarray | None,
    *,
    route_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(tuple(int(item) for item in indices), dtype=np.int64)
    if selected.ndim != 1:
        selected = selected.reshape(-1)
    if (
        np.any(selected < 0)
        or np.any(selected >= route_count)
        or np.unique(selected).size != selected.size
    ):
        raise ValueError("response action route indices are invalid")
    values = (
        -np.ones(selected.size, dtype=np.float64)
        if coefficients is None
        else np.asarray(coefficients, dtype=np.float64).reshape(-1)
    )
    if values.shape != selected.shape or not np.isfinite(values).all():
        raise ValueError("response action coefficients are invalid")
    return selected, values


def _matrix_square_root(matrix: np.ndarray) -> np.ndarray:
    psd = _symmetric_psd(matrix, name="matrix square-root input")
    eigenvalues, eigenvectors = np.linalg.eigh(psd)
    return (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T


# ---------------------------------------------------------------------------
# Action moments and frozen response constraint
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResponseProjectionReceipt:
    """Frozen approximation contract for a tractable response subspace.

    A projected response geometry is scientific only when the projection is
    fixed before candidate adjudication, its residual is priced, and any event
    that invalidates the projection (for example a new strongest competitor)
    triggers expansion or relinearization rather than silent reuse.
    """

    source_channel_ids: tuple[str, ...]
    projected_channel_ids: tuple[str, ...]
    projection: np.ndarray = field(repr=False, compare=False)
    response_spec_hash: str
    residual_norm_bound: float
    projection_kind: str = "frozen_linear_response_projection"
    comparator_independent: bool = True
    competitor_switch_requires_refresh: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = tuple(map(str, self.source_channel_ids))
        target = tuple(map(str, self.projected_channel_ids))
        matrix = _readonly_array(self.projection, ndim=2, name="response projection")
        if (
            not source
            or not target
            or len(set(source)) != len(source)
            or len(set(target)) != len(target)
            or matrix.shape != (len(target), len(source))
        ):
            raise ValueError("response projection receipt has incompatible axes")
        if np.linalg.matrix_rank(matrix) < matrix.shape[0]:
            raise ValueError("response projection rows must be linearly independent")
        bound = float(self.residual_norm_bound)
        if not np.isfinite(bound) or bound < 0.0:
            raise ValueError(
                "response projection residual bound must be finite and nonnegative"
            )
        if not self.response_spec_hash or not self.projection_kind:
            raise ValueError("response projection receipt requires scientific identity")
        object.__setattr__(self, "source_channel_ids", source)
        object.__setattr__(self, "projected_channel_ids", target)
        object.__setattr__(self, "projection", matrix)
        object.__setattr__(self, "residual_norm_bound", bound)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def exact(self) -> bool:
        return (
            self.residual_norm_bound == 0.0
            and self.projection.shape[0] == self.projection.shape[1]
        )

    @property
    def content_hash(self) -> str:
        return content_hash(
            self.to_dict(include_hash=False), prefix="response-projection-"
        )

    def project(self, rows: np.ndarray) -> np.ndarray:
        values = np.asarray(rows, dtype=np.float64)
        if values.ndim != 3 or values.shape[1] != len(self.source_channel_ids):
            raise ValueError(
                "response projection rows require [context,source_response,route] axes"
            )
        return np.einsum("pd,ndr->npr", self.projection, values, optimize=True)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "source_channel_ids": list(self.source_channel_ids),
            "projected_channel_ids": list(self.projected_channel_ids),
            "projection": self.projection.tolist(),
            "response_spec_hash": self.response_spec_hash,
            "residual_norm_bound": self.residual_norm_bound,
            "projection_kind": self.projection_kind,
            "comparator_independent": self.comparator_independent,
            "competitor_switch_requires_refresh": self.competitor_switch_requires_refresh,
            "exact": self.exact,
            "metadata": dict(self.metadata),
        }
        if include_hash:
            payload["content_hash"] = self.content_hash
        return payload


@dataclass(frozen=True, slots=True)
class ResponseActionMoments:
    """First two response moments induced by one physical gate action."""

    mean: np.ndarray = field(repr=False, compare=False)
    contextual_covariance: np.ndarray = field(repr=False, compare=False)
    second_moment: np.ndarray = field(repr=False, compare=False)
    mean_standard_error: np.ndarray = field(repr=False, compare=False)
    second_moment_standard_error: np.ndarray = field(repr=False, compare=False)
    sample_count: int
    approximation: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    mean_estimator_covariance: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    model_error_covariance: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    projection_residual_bound: float = 0.0

    def __post_init__(self) -> None:
        mean = _readonly_array(self.mean, ndim=1, name="action response mean")
        covariance = _symmetric_psd(
            self.contextual_covariance,
            name="action contextual covariance",
            tolerance=1e-6,
        )
        second = _symmetric_psd(
            self.second_moment, name="action response second moment", tolerance=1e-6
        )
        mean_se = _readonly_array(
            self.mean_standard_error, ndim=1, name="action response mean standard error"
        )
        second_se = _readonly_array(
            self.second_moment_standard_error,
            ndim=2,
            name="action response second-moment standard error",
        )
        dimension = mean.size
        if (
            covariance.shape != (dimension, dimension)
            or second.shape != covariance.shape
        ):
            raise ValueError("action response moments use incompatible response axes")
        if mean_se.shape != (dimension,) or second_se.shape != covariance.shape:
            raise ValueError("action response uncertainties use incompatible axes")
        if np.any(mean_se < 0.0) or np.any(second_se < 0.0):
            raise ValueError("action response uncertainties must be nonnegative")
        estimator_covariance = (
            np.diag(mean_se * mean_se)
            if self.mean_estimator_covariance is None
            else _symmetric_psd(
                self.mean_estimator_covariance,
                name="action response mean-estimator covariance",
                tolerance=1e-6,
            )
        )
        if estimator_covariance.shape != covariance.shape:
            raise ValueError(
                "action response mean-estimator covariance has incompatible axes"
            )
        model_error = (
            np.zeros_like(covariance)
            if self.model_error_covariance is None
            else _symmetric_psd(
                self.model_error_covariance,
                name="action response model-error covariance",
                tolerance=1e-6,
            )
        )
        if model_error.shape != covariance.shape:
            raise ValueError(
                "action response model-error covariance has incompatible axes"
            )
        estimator_diagonal = np.maximum(np.diag(estimator_covariance), 0.0)
        if not np.allclose(mean_se * mean_se, estimator_diagonal, atol=5e-8, rtol=5e-6):
            raise ValueError(
                "action response mean_standard_error must equal the square root of the estimator-covariance diagonal"
            )
        projection_bound = float(self.projection_residual_bound)
        if not np.isfinite(projection_bound) or projection_bound < 0.0:
            raise ValueError(
                "action response projection_residual_bound must be finite and nonnegative"
            )
        expected = covariance + model_error + np.outer(mean, mean)
        if not np.allclose(second, expected, atol=5e-8, rtol=5e-8):
            raise ValueError(
                "second_moment must equal contextual_covariance plus model_error_covariance plus outer(mean, mean)"
            )
        if int(self.sample_count) < 0 or not self.approximation:
            raise ValueError(
                "action response moments require sample_count and approximation"
            )
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "contextual_covariance", covariance)
        object.__setattr__(self, "second_moment", second)
        object.__setattr__(self, "mean_standard_error", mean_se)
        object.__setattr__(self, "second_moment_standard_error", second_se)
        object.__setattr__(self, "mean_estimator_covariance", estimator_covariance)
        object.__setattr__(self, "model_error_covariance", model_error)
        object.__setattr__(self, "projection_residual_bound", projection_bound)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def covariance(self) -> np.ndarray:
        """Compatibility alias for contextual covariance."""

        return self.contextual_covariance

    @property
    def response_dimension(self) -> int:
        return int(self.mean.size)

    def contracted_energy(self, output_metric: np.ndarray) -> float:
        metric = _symmetric_psd(output_metric, name="action contraction metric")
        if metric.shape != self.second_moment.shape:
            raise ValueError("action contraction metric has incompatible dimension")
        return max(0.0, float(np.trace(metric @ self.second_moment)))

    def oriented_mean(self, direction: Sequence[float] | np.ndarray) -> float:
        vector = np.asarray(direction, dtype=np.float64).reshape(-1)
        if vector.shape != self.mean.shape or not np.isfinite(vector).all():
            raise ValueError("oriented response direction has incompatible dimension")
        return float(vector @ self.mean)

    def with_mean_shift(
        self,
        shift: Sequence[float] | np.ndarray,
        *,
        approximation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ResponseActionMoments":
        vector = np.asarray(shift, dtype=np.float64).reshape(-1)
        if vector.shape != self.mean.shape or not np.isfinite(vector).all():
            raise ValueError("response mean shift has incompatible dimension")
        mean = self.mean + vector
        return ResponseActionMoments(
            mean=mean,
            contextual_covariance=self.contextual_covariance,
            second_moment=self.contextual_covariance
            + self.model_error_covariance
            + np.outer(mean, mean),
            mean_standard_error=self.mean_standard_error,
            second_moment_standard_error=self.second_moment_standard_error,
            sample_count=self.sample_count,
            approximation=approximation or self.approximation,
            metadata={**dict(self.metadata), **dict(metadata or {})},
            mean_estimator_covariance=self.mean_estimator_covariance,
            model_error_covariance=self.model_error_covariance,
            projection_residual_bound=self.projection_residual_bound,
        )

    def inflate_contextual_covariance(
        self,
        inflation: np.ndarray | float,
        *,
        approximation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ResponseActionMoments":
        if np.isscalar(inflation):
            value = float(inflation)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "scalar response covariance inflation must be nonnegative"
                )
            matrix = value * np.eye(self.response_dimension)
        else:
            matrix = _symmetric_psd(
                np.asarray(inflation), name="response covariance inflation"
            )
            if matrix.shape != self.contextual_covariance.shape:
                raise ValueError(
                    "response covariance inflation has incompatible dimension"
                )
        covariance = _symmetric_psd(
            self.contextual_covariance + matrix,
            name="inflated response contextual covariance",
        )
        return ResponseActionMoments(
            mean=self.mean,
            contextual_covariance=covariance,
            second_moment=covariance
            + self.model_error_covariance
            + np.outer(self.mean, self.mean),
            mean_standard_error=self.mean_standard_error,
            # Model-residual inflation is part of the conservative response
            # geometry, not sampling uncertainty in the estimated moment.
            second_moment_standard_error=self.second_moment_standard_error,
            sample_count=self.sample_count,
            approximation=approximation or self.approximation,
            metadata={**dict(self.metadata), **dict(metadata or {})},
            mean_estimator_covariance=self.mean_estimator_covariance,
            model_error_covariance=self.model_error_covariance,
            projection_residual_bound=self.projection_residual_bound,
        )

    def inflate_model_error_covariance(
        self,
        inflation: np.ndarray | float,
        *,
        approximation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ResponseActionMoments":
        """Conservatively add local-model misspecification without changing context variability.

        Contextual covariance describes genuine response heterogeneity over the
        declared context population.  Model-error covariance instead records
        uncertainty in the local response approximation itself.  Keeping the
        two separate prevents exact-transition residuals from being
        misinterpreted as biological or stimulus variability.
        """

        if np.isscalar(inflation):
            value = float(inflation)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "scalar response model-error inflation must be nonnegative"
                )
            matrix = value * np.eye(self.response_dimension)
        else:
            matrix = _symmetric_psd(
                np.asarray(inflation), name="response model-error inflation"
            )
            if matrix.shape != self.contextual_covariance.shape:
                raise ValueError(
                    "response model-error inflation has incompatible dimension"
                )
        model_error = _symmetric_psd(
            self.model_error_covariance + matrix,
            name="inflated response model-error covariance",
        )
        return ResponseActionMoments(
            mean=self.mean,
            contextual_covariance=self.contextual_covariance,
            model_error_covariance=model_error,
            second_moment=self.contextual_covariance
            + model_error
            + np.outer(self.mean, self.mean),
            mean_standard_error=self.mean_standard_error,
            second_moment_standard_error=self.second_moment_standard_error,
            sample_count=self.sample_count,
            approximation=approximation or self.approximation,
            metadata={**dict(self.metadata), **dict(metadata or {})},
            mean_estimator_covariance=self.mean_estimator_covariance,
            projection_residual_bound=self.projection_residual_bound,
        )

    def to_summary(self) -> dict[str, Any]:
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "contextual_covariance": self.contextual_covariance.tolist(),
            "model_error_covariance": self.model_error_covariance.tolist(),
            "second_moment": self.second_moment.tolist(),
            "mean_standard_error": self.mean_standard_error.tolist(),
            "mean_estimator_covariance": self.mean_estimator_covariance.tolist(),
            "second_moment_standard_error": self.second_moment_standard_error.tolist(),
            "projection_residual_bound": float(self.projection_residual_bound),
            "sample_count": int(self.sample_count),
            "approximation": self.approximation,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TargetCompetitorMomentDecomposition:
    """Exact two-channel target-writing/competitor-integration moments.

    ``cross_second_moment`` is E[TA], not a centered covariance.  The latter is
    stored separately as ``centered_cross_covariance``.
    """

    target_mean: np.ndarray = field(repr=False, compare=False)
    competitor_mean: np.ndarray = field(repr=False, compare=False)
    target_second_moment: np.ndarray = field(repr=False, compare=False)
    competitor_second_moment: np.ndarray = field(repr=False, compare=False)
    cross_second_moment: np.ndarray = field(repr=False, compare=False)
    centered_cross_covariance: np.ndarray = field(repr=False, compare=False)
    margin_second_moment: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        size: int | None = None
        for name in (
            "target_mean",
            "competitor_mean",
            "target_second_moment",
            "competitor_second_moment",
            "cross_second_moment",
            "centered_cross_covariance",
            "margin_second_moment",
        ):
            value = _readonly_array(np.asarray(getattr(self, name)), ndim=1, name=name)
            size = value.size if size is None else size
            if value.size != size:
                raise ValueError("target/competitor decomposition arrays must align")
            arrays[name] = value
        expected_covariance = (
            arrays["cross_second_moment"]
            - arrays["target_mean"] * arrays["competitor_mean"]
        )
        if not np.allclose(
            arrays["centered_cross_covariance"],
            expected_covariance,
            atol=5e-9,
            rtol=5e-9,
        ):
            raise ValueError("centered cross covariance must equal E[TA]-E[T]E[A]")
        expected_margin = (
            arrays["target_second_moment"]
            + arrays["competitor_second_moment"]
            - 2.0 * arrays["cross_second_moment"]
        )
        if not np.allclose(
            arrays["margin_second_moment"], expected_margin, atol=5e-9, rtol=5e-9
        ):
            raise ValueError("margin second moment must equal E[T^2]+E[A^2]-2E[TA]")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @property
    def route_count(self) -> int:
        return int(self.target_mean.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: np.asarray(getattr(self, name)).tolist()
            for name in (
                "target_mean",
                "competitor_mean",
                "target_second_moment",
                "competitor_second_moment",
                "cross_second_moment",
                "centered_cross_covariance",
                "margin_second_moment",
            )
        }


@dataclass(frozen=True, slots=True)
class ResponseRiskEvaluation:
    """Dimensionless risk induced by a frozen response-safe region."""

    risk: float
    active_constraint_id: str
    directional_risks: Mapping[str, float]
    energy_risk: float
    contextual_component: float
    estimation_component: float
    mean_component: float
    model_component: float = 0.0
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "risk",
            "energy_risk",
            "contextual_component",
            "estimation_component",
            "mean_component",
            "model_component",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < -1e-12:
                raise ValueError(f"response risk {name} must be finite and nonnegative")
        risks = {
            str(key): float(value) for key, value in self.directional_risks.items()
        }
        if any(not np.isfinite(value) or value < -1e-12 for value in risks.values()):
            raise ValueError(
                "directional response risks must be finite and nonnegative"
            )
        object.__setattr__(self, "directional_risks", risks)
        object.__setattr__(self, "details", dict(self.details))

    @property
    def directional_components(self) -> tuple[dict[str, Any], ...]:
        directional = self.details.get("directional", {})
        result = []
        for direction_id, risk in self.directional_risks.items():
            parts = dict(directional.get(direction_id, {}))
            result.append(
                {
                    "constraint_id": direction_id,
                    "risk": float(risk),
                    "mean_allowance": float(parts.get("mean_adverse", 0.0)),
                    "contextual_allowance": float(
                        parts.get("contextual_standard_deviation", 0.0)
                    ),
                    "model_allowance": float(
                        parts.get("model_error_standard_deviation", 0.0)
                    ),
                    "estimation_allowance": float(
                        parts.get("estimator_standard_error", 0.0)
                    ),
                    **parts,
                }
            )
        return tuple(result)

    @property
    def energy_component(self) -> Mapping[str, Any]:
        return dict(self.details.get("energy", {}))


class ResponseConstraintEvaluator:
    """Evaluate matrix-valued response moments against one frozen safe region.

    For a lower-bound direction ``h``, pruning is harmful when ``h @ delta`` is
    negative.  The robust adverse displacement is

        max(0, -h@mean + k_context*sd_context
                         + k_model*sd_model + k_est*se_mean).

    Contextual variability, local-model misspecification, and estimator
    uncertainty remain explicitly separated.  The returned risk is one at the
    declared boundary.
    """

    def __init__(
        self,
        constraint: ResponseConstraint,
        channel_ids: Sequence[str] | None = None,
    ) -> None:
        self.constraint = constraint
        self.channel_ids = tuple(map(str, channel_ids or ()))
        if self.channel_ids and len(self.channel_ids) != constraint.output_dimension:
            raise ValueError(
                "response constraint channel axis differs from output_dimension"
            )

    @property
    def output_dimension(self) -> int:
        return int(self.constraint.output_dimension)

    def evaluate(self, moments: ResponseActionMoments) -> ResponseRiskEvaluation:
        if moments.response_dimension != self.output_dimension:
            raise ValueError("response action and constraint dimensions differ")
        directional: dict[str, float] = {}
        directional_parts: dict[str, dict[str, float]] = {}
        worst_directional = 0.0
        active_id = "none"
        worst_mean = 0.0
        worst_contextual = 0.0
        worst_model = 0.0
        worst_estimation = 0.0
        for direction in self.constraint.directions:
            vector = np.asarray(direction.vector, dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            vector = vector / norm
            tolerance = float(direction.lower_tolerance) / norm
            oriented_mean = float(vector @ moments.mean)
            contextual_sd = float(
                np.sqrt(max(0.0, vector @ moments.contextual_covariance @ vector))
            )
            model_sd = float(
                np.sqrt(max(0.0, vector @ moments.model_error_covariance @ vector))
            )
            estimator_se = float(
                np.sqrt(max(0.0, vector @ moments.mean_estimator_covariance @ vector))
            )
            projection_allowance = float(moments.projection_residual_bound)
            mean_adverse = max(0.0, -oriented_mean)
            contextual = (
                self.constraint.contextual_robustness_multiplier * contextual_sd
            )
            model = self.constraint.model_uncertainty_multiplier * model_sd
            estimation = (
                self.constraint.estimation_uncertainty_multiplier * estimator_se
            )
            risk = max(
                0.0,
                (mean_adverse + contextual + model + estimation + projection_allowance)
                / tolerance,
            )
            directional[direction.direction_id] = risk
            directional_parts[direction.direction_id] = {
                "oriented_mean": oriented_mean,
                "mean_adverse": mean_adverse,
                "contextual_standard_deviation": contextual_sd,
                "model_error_standard_deviation": model_sd,
                "estimator_standard_error": estimator_se,
                "projection_residual_allowance": projection_allowance,
                "normalized_risk": risk,
            }
            if risk > worst_directional:
                worst_directional = risk
                active_id = direction.direction_id
                worst_mean = mean_adverse / tolerance
                worst_contextual = contextual / tolerance
                worst_model = model / tolerance
                worst_estimation = estimation / tolerance

        energy_risk = 0.0
        energy_details: dict[str, Any] = {}
        if self.constraint.energy_metric:
            metric = _symmetric_psd(
                np.asarray(self.constraint.energy_metric, dtype=np.float64),
                name="response constraint energy metric",
            )
            energy = max(0.0, float(np.trace(metric @ moments.second_moment)))
            coherent_energy = max(0.0, float(moments.mean @ metric @ moments.mean))
            contextual_energy = max(
                0.0, float(np.trace(metric @ moments.contextual_covariance))
            )
            model_error_energy = max(
                0.0, float(np.trace(metric @ moments.model_error_covariance))
            )
            # This is an entrywise plug-in upper allowance, not a coverage
            # guarantee.  A stronger simultaneous matrix bound can replace it
            # without changing the retained response geometry.
            energy_estimation_allowance = (
                self.constraint.estimation_uncertainty_multiplier
                * float(np.sum(np.abs(metric) * moments.second_moment_standard_error))
            )
            energy_upper = max(0.0, energy + energy_estimation_allowance)
            tolerance = float(self.constraint.energy_tolerance)
            projection_root_allowance = np.sqrt(
                max(0.0, float(np.max(np.linalg.eigvalsh(metric))))
            ) * float(moments.projection_residual_bound)
            energy_risk = (
                np.sqrt(energy_upper) + projection_root_allowance
            ) / tolerance
            energy_details = {
                "energy": energy,
                "coherent_mean_energy": coherent_energy,
                "contextual_energy": contextual_energy,
                "model_error_energy": model_error_energy,
                "energy_estimation_allowance": energy_estimation_allowance,
                "energy_upper": energy_upper,
                "root_energy_upper": np.sqrt(energy_upper),
                "projection_root_allowance": projection_root_allowance,
                "tolerance": tolerance,
                "uncertainty_method": "entrywise_plugin_second_moment_allowance",
            }
            if energy_risk > worst_directional:
                active_id = "response_energy"

        risk = max(worst_directional, energy_risk)
        return ResponseRiskEvaluation(
            risk=float(risk),
            active_constraint_id=active_id,
            directional_risks=directional,
            energy_risk=float(energy_risk),
            contextual_component=float(worst_contextual),
            estimation_component=float(worst_estimation),
            mean_component=float(worst_mean),
            model_component=float(worst_model),
            details={
                "directional": directional_parts,
                "energy": energy_details,
                "maximum_accepted_risk": self.constraint.maximum_accepted_risk,
                "contextual_variability_is_not_estimator_uncertainty": True,
                "model_error_is_not_contextual_variability": True,
            },
        )

    def route_local_risks(self, tensor: "ResponseMomentTensor") -> np.ndarray:
        """Evaluate all route-local v1 actions without an O(R) Python loop."""

        if tensor.response_dimension != self.output_dimension:
            raise ValueError("response moment tensor and constraint dimensions differ")
        worst = np.zeros(tensor.route_count, dtype=np.float64)
        projection = tensor.projection_residual_bound_per_unit_route_coefficient
        for direction in self.constraint.directions:
            vector = np.asarray(direction.vector, dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            vector = vector / norm
            tolerance = float(direction.lower_tolerance) / norm
            oriented = vector @ tensor.mean
            contextual_variance = np.einsum(
                "a,rab,b->r",
                vector,
                tensor.contextual_covariance_diagonal,
                vector,
                optimize=True,
            )
            contextual = self.constraint.contextual_robustness_multiplier * np.sqrt(
                np.maximum(contextual_variance, 0.0)
            )
            estimator_variance = np.einsum(
                "d,rde,e->r",
                vector,
                tensor.mean_estimator_covariance_diagonal,
                vector,
                optimize=True,
            )
            estimator = self.constraint.estimation_uncertainty_multiplier * np.sqrt(
                np.maximum(estimator_variance, 0.0)
            )
            worst = np.maximum(
                worst,
                (np.maximum(-oriented, 0.0) + contextual + estimator + projection)
                / tolerance,
            )
        if self.constraint.energy_metric:
            metric = _symmetric_psd(
                np.asarray(self.constraint.energy_metric, dtype=np.float64),
                name="response constraint energy metric",
            )
            energy = np.einsum(
                "ab,rba->r", metric, tensor.second_moment_diagonal, optimize=True
            )
            allowance = self.constraint.estimation_uncertainty_multiplier * np.einsum(
                "ab,rab->r",
                np.abs(metric),
                tensor.second_moment_standard_error,
                optimize=True,
            )
            projection_root = (
                np.sqrt(max(0.0, float(np.max(np.linalg.eigvalsh(metric)))))
                * projection
            )
            worst = np.maximum(
                worst,
                (np.sqrt(np.maximum(energy + allowance, 0.0)) + projection_root)
                / float(self.constraint.energy_tolerance),
            )
        return worst

    def accepted(self, moments: ResponseActionMoments) -> bool:
        return self.evaluate(moments).risk <= float(
            self.constraint.maximum_accepted_risk
        )

    def evaluate_route_actions(
        self,
        moments: "ResponseMomentTensor",
        route_indices: Sequence[int] | np.ndarray | None = None,
        *,
        coefficient: float = -1.0,
    ) -> np.ndarray:
        """Vectorized standalone route risk for native-scale v1 screening."""

        indices = (
            np.arange(moments.route_count, dtype=np.int64)
            if route_indices is None
            else np.asarray(tuple(int(item) for item in route_indices), dtype=np.int64)
        )
        if indices.size == 0:
            return np.zeros(0, dtype=np.float64)
        if (
            np.any(indices < 0)
            or np.any(indices >= moments.route_count)
            or np.unique(indices).size != indices.size
        ):
            raise ValueError("route-risk indices are invalid")
        coefficient = float(coefficient)
        projection = (
            abs(coefficient)
            * moments.projection_residual_bound_per_unit_route_coefficient
        )
        means = coefficient * moments.mean[:, indices]  # [d,k]
        covariances = (
            coefficient * coefficient
        ) * moments.contextual_covariance_diagonal[
            indices
        ]  # [k,d,d]
        mean_estimator_covariance = (
            coefficient * coefficient
        ) * moments.mean_estimator_covariance_diagonal[indices]
        risks = np.zeros(indices.size, dtype=np.float64)
        for direction in self.constraint.directions:
            vector = np.asarray(direction.vector, dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            vector = vector / norm
            tolerance = float(direction.lower_tolerance) / norm
            oriented = vector @ means
            contextual_variance = np.einsum("d,kde,e->k", vector, covariances, vector)
            estimator_variance = np.einsum(
                "d,kde,e->k", vector, mean_estimator_covariance, vector
            )
            estimator_se = np.sqrt(np.maximum(estimator_variance, 0.0))
            adverse = (
                np.maximum(0.0, -oriented)
                + self.constraint.contextual_robustness_multiplier
                * np.sqrt(np.maximum(contextual_variance, 0.0))
                + self.constraint.estimation_uncertainty_multiplier * estimator_se
                + projection
            ) / tolerance
            risks = np.maximum(risks, adverse)
        if self.constraint.energy_metric:
            metric = np.asarray(self.constraint.energy_metric, dtype=np.float64)
            second = covariances + np.einsum("dk,ek->kde", means, means)
            energy = np.einsum("de,ked->k", metric, second)
            second_se = (
                abs(coefficient) ** 2 * moments.second_moment_standard_error[indices]
            )
            allowance = self.constraint.estimation_uncertainty_multiplier * np.einsum(
                "de,kde->k", np.abs(metric), second_se
            )
            projection_root = (
                np.sqrt(max(0.0, float(np.max(np.linalg.eigvalsh(metric)))))
                * projection
            )
            risks = np.maximum(
                risks,
                (np.sqrt(np.maximum(energy + allowance, 0.0)) + projection_root)
                / float(self.constraint.energy_tolerance),
            )
        return risks

    def evaluate_v1_additions(
        self,
        current: ResponseActionMoments,
        moments: "ResponseMomentTensor",
        route_indices: Sequence[int] | np.ndarray,
        *,
        coefficient: float = -1.0,
    ) -> np.ndarray:
        """Vectorized risk after adding one route to a current v1 action."""

        if isinstance(route_indices, np.ndarray):
            indices = np.asarray(route_indices, dtype=np.int64).reshape(-1)
        else:
            indices = np.fromiter(
                (int(item) for item in route_indices),
                dtype=np.int64,
            )
        if indices.size == 0:
            return np.zeros(0, dtype=np.float64)
        ordered_unique = bool(indices.size <= 1 or np.all(indices[1:] > indices[:-1]))
        if (
            np.any(indices < 0)
            or np.any(indices >= moments.route_count)
            or (not ordered_unique and np.unique(indices).size != indices.size)
        ):
            raise ValueError("incremental route-risk indices are invalid")
        coefficient = float(coefficient)
        projection = (
            current.projection_residual_bound
            + abs(coefficient)
            * moments.projection_residual_bound_per_unit_route_coefficient
        )
        means = current.mean[:, None] + coefficient * moments.mean[:, indices]
        covariances = (
            current.contextual_covariance[None, :, :]
            + (coefficient * coefficient)
            * moments.contextual_covariance_diagonal[indices]
        )
        model_covariances = np.broadcast_to(
            current.model_error_covariance[None, :, :],
            covariances.shape,
        )
        mean_estimator_covariance = (
            current.mean_estimator_covariance[None, :, :]
            + (coefficient * coefficient)
            * moments.mean_estimator_covariance_diagonal[indices]
        )
        risks = np.zeros(indices.size, dtype=np.float64)
        for direction in self.constraint.directions:
            vector = np.asarray(direction.vector, dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            vector = vector / norm
            tolerance = float(direction.lower_tolerance) / norm
            oriented = vector @ means
            contextual_variance = np.einsum("d,kde,e->k", vector, covariances, vector)
            model_variance = np.einsum("d,kde,e->k", vector, model_covariances, vector)
            estimator_variance = np.einsum(
                "d,kde,e->k", vector, mean_estimator_covariance, vector
            )
            estimator_se = np.sqrt(np.maximum(estimator_variance, 0.0))
            adverse = (
                np.maximum(0.0, -oriented)
                + self.constraint.contextual_robustness_multiplier
                * np.sqrt(np.maximum(contextual_variance, 0.0))
                + self.constraint.model_uncertainty_multiplier
                * np.sqrt(np.maximum(model_variance, 0.0))
                + self.constraint.estimation_uncertainty_multiplier * estimator_se
                + projection
            ) / tolerance
            risks = np.maximum(risks, adverse)
        if self.constraint.energy_metric:
            metric = np.asarray(self.constraint.energy_metric, dtype=np.float64)
            second = (
                covariances + model_covariances + np.einsum("dk,ek->kde", means, means)
            )
            energy = np.einsum("de,ked->k", metric, second)
            second_se = np.sqrt(
                current.second_moment_standard_error[None, :, :] ** 2
                + (
                    (coefficient * coefficient)
                    * moments.second_moment_standard_error[indices]
                )
                ** 2
            )
            allowance = self.constraint.estimation_uncertainty_multiplier * np.einsum(
                "de,kde->k", np.abs(metric), second_se
            )
            projection_root = (
                np.sqrt(max(0.0, float(np.max(np.linalg.eigvalsh(metric)))))
                * projection
            )
            risks = np.maximum(
                risks,
                (np.sqrt(np.maximum(energy + allowance, 0.0)) + projection_root)
                / float(self.constraint.energy_tolerance),
            )
        return risks

    def evaluate_row_candidates(
        self,
        rows: np.ndarray,
        *,
        context_weights: Sequence[float] | np.ndarray | None = None,
        projection_residual_bound: float | Sequence[float] | np.ndarray = 0.0,
    ) -> np.ndarray:
        """Vectorized risk for candidate response rows ``[context,response,candidate]``."""

        values = np.asarray(rows, dtype=np.float64)
        if (
            values.ndim != 3
            or values.shape[0] <= 0
            or values.shape[1] != self.output_dimension
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "candidate response rows require finite [context,response,candidate] axes"
            )
        weights = _normalized_context_weights(context_weights, values.shape[0])
        means = np.einsum("n,ndk->dk", weights, values)
        centered = values - means[None, :, :]
        covariances = np.einsum("n,ndk,nek->kde", weights, centered, centered)
        mean_se = np.empty_like(means)
        mean_estimator_covariance = np.empty(
            (values.shape[2], self.output_dimension, self.output_dimension),
            dtype=np.float64,
        )
        second_rows = np.einsum("ndk,nek->knde", values, values, optimize=True)
        second = np.einsum("n,knde->kde", weights, second_rows, optimize=True)
        second_se = np.empty_like(second)
        for candidate in range(values.shape[2]):
            candidate_rows = values[:, :, candidate]
            candidate_mean = means[:, candidate]
            mean_estimator_covariance[candidate] = _weighted_mean_estimator_covariance(
                candidate_rows, weights, candidate_mean
            )
            mean_se[:, candidate] = np.sqrt(
                np.maximum(np.diag(mean_estimator_covariance[candidate]), 0.0)
            )
            second_se[candidate] = _weighted_mean_standard_error(
                second_rows[candidate], weights, second[candidate]
            )
        residual = np.asarray(projection_residual_bound, dtype=np.float64)
        if residual.ndim == 0:
            residual = np.full(values.shape[2], float(residual), dtype=np.float64)
        else:
            residual = residual.reshape(-1)
        if (
            residual.shape != (values.shape[2],)
            or not np.isfinite(residual).all()
            or np.any(residual < 0.0)
        ):
            raise ValueError(
                "candidate projection residual bounds must be finite, nonnegative, and align with candidates"
            )
        risks = np.zeros(values.shape[2], dtype=np.float64)
        for direction in self.constraint.directions:
            vector = np.asarray(direction.vector, dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            vector = vector / norm
            tolerance = float(direction.lower_tolerance) / norm
            oriented = vector @ means
            contextual_variance = np.einsum("d,kde,e->k", vector, covariances, vector)
            estimator_variance = np.einsum(
                "d,kde,e->k", vector, mean_estimator_covariance, vector
            )
            estimator_se = np.sqrt(np.maximum(estimator_variance, 0.0))
            adverse = (
                np.maximum(0.0, -oriented)
                + self.constraint.contextual_robustness_multiplier
                * np.sqrt(np.maximum(contextual_variance, 0.0))
                + self.constraint.estimation_uncertainty_multiplier * estimator_se
                + residual
            ) / tolerance
            risks = np.maximum(risks, adverse)
        if self.constraint.energy_metric:
            metric = np.asarray(self.constraint.energy_metric, dtype=np.float64)
            energy = np.einsum("de,ked->k", metric, second)
            allowance = self.constraint.estimation_uncertainty_multiplier * np.einsum(
                "de,kde->k", np.abs(metric), second_se
            )
            projection_root = (
                np.sqrt(max(0.0, float(np.max(np.linalg.eigvalsh(metric))))) * residual
            )
            risks = np.maximum(
                risks,
                (np.sqrt(np.maximum(energy + allowance, 0.0)) + projection_root)
                / float(self.constraint.energy_tolerance),
            )
        return risks


# ---------------------------------------------------------------------------
# v1 route-local response moments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResponseMomentTensor:
    """v1 sufficient response moments in semantic response coordinates.

    ``second_moment_diagonal[r]`` is ``E[j_r j_r^T]``.  Centered cross-route
    covariance is deliberately absent.  Coherent route means are summed before
    forming an action's second moment, so v1 does *not* discard mean-level
    target/competitor or cross-route coupling.
    """

    channel_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    mean: np.ndarray = field(repr=False, compare=False)  # [response, route]
    second_moment_diagonal: np.ndarray = field(
        repr=False, compare=False
    )  # [route,response,response]
    mean_standard_error: np.ndarray = field(
        repr=False, compare=False
    )  # [response,route]
    second_moment_standard_error: np.ndarray = field(
        repr=False, compare=False
    )  # [route,response,response]
    sample_count: int
    response_spec_hash: str
    metric_id: str
    context_axis_hash: str
    mean_estimator_covariance_diagonal: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "response_moment_tensor_v1"

    def __post_init__(self) -> None:
        channels = tuple(map(str, self.channel_ids))
        routes = canonical_unique_string_axis(
            self.route_ids,
            name="response moment route axis",
        )
        if not channels or len(set(channels)) != len(channels):
            raise ValueError(
                "response moment tensor requires unique response and route axes"
            )
        if (
            not self.response_spec_hash
            or not self.metric_id
            or not self.context_axis_hash
            or not self.schema
        ):
            raise ValueError(
                "response moment tensor requires complete scientific identity"
            )
        d, r = len(channels), len(routes)
        mean = _readonly_array(self.mean, ndim=2, name="response mean")
        second = _readonly_array(
            self.second_moment_diagonal, ndim=3, name="route response second moment"
        )
        mean_se = _readonly_array(
            self.mean_standard_error, ndim=2, name="response mean standard error"
        )
        second_se = _readonly_array(
            self.second_moment_standard_error,
            ndim=3,
            name="route response second-moment standard error",
        )
        if mean.shape != (d, r) or mean_se.shape != (d, r):
            raise ValueError("response mean arrays have incompatible axes")
        if second.shape != (r, d, d) or second_se.shape != (r, d, d):
            raise ValueError("response second-moment arrays have incompatible axes")
        if np.any(mean_se < 0.0) or np.any(second_se < 0.0):
            raise ValueError("response moment uncertainties must be nonnegative")
        estimator_covariance = (
            np.einsum("rd,de->rde", (mean_se * mean_se).T, np.eye(d, dtype=np.float64))
            if self.mean_estimator_covariance_diagonal is None
            else _readonly_array(
                self.mean_estimator_covariance_diagonal,
                ndim=3,
                name="route response mean-estimator covariance",
            )
        )
        if estimator_covariance.shape != (r, d, d):
            raise ValueError(
                "route response mean-estimator covariance has incompatible axes"
            )
        estimator_copy = _batched_symmetric_psd(
            estimator_covariance,
            name="route mean-estimator covariance",
            tolerance=1e-6,
        )
        estimator_diagonal = np.maximum(
            np.diagonal(estimator_copy, axis1=1, axis2=2).T, 0.0
        )
        if not np.allclose(mean_se * mean_se, estimator_diagonal, atol=5e-8, rtol=5e-6):
            raise ValueError(
                "response mean_standard_error must equal the square root of the route estimator-covariance diagonal"
            )
        second_copy = _batched_symmetric_psd(
            second,
            name="route second moment",
            tolerance=1e-6,
        )
        if int(self.sample_count) <= 0:
            raise ValueError("response moment tensor sample_count must be positive")
        object.__setattr__(self, "channel_ids", channels)
        object.__setattr__(self, "route_ids", routes)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "second_moment_diagonal", second_copy)
        object.__setattr__(self, "mean_standard_error", mean_se)
        object.__setattr__(self, "second_moment_standard_error", second_se)
        object.__setattr__(self, "mean_estimator_covariance_diagonal", estimator_copy)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def response_dimension(self) -> int:
        return len(self.channel_ids)

    @property
    def route_count(self) -> int:
        return len(self.route_ids)

    @property
    def projection_residual_bound_per_unit_route_coefficient(self) -> float:
        value = float(self.metadata.get("projection_residual_bound", 0.0))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "response moment projection residual bound must be finite and nonnegative"
            )
        return value

    @cached_property
    def contextual_covariance_diagonal(self) -> np.ndarray:
        covariance = self.second_moment_diagonal - np.einsum(
            "ir,jr->rij", self.mean, self.mean
        )
        result = np.empty_like(covariance)
        for index in range(self.route_count):
            result[index] = _symmetric_psd(
                covariance[index],
                name=f"route contextual covariance {index}",
                tolerance=1e-5,
            )
        result.setflags(write=False)
        return result

    @property
    def route_second_moment(self) -> np.ndarray:
        return self.second_moment_diagonal

    @property
    def covariance_standard_error(self) -> np.ndarray:
        """Compatibility alias for route-local second-moment uncertainty."""

        return self.second_moment_standard_error

    def target_competitor_decomposition(
        self,
        *,
        target_channel_id: str = "target_writing",
        competitor_channel_id: str = "competitor_integration",
    ) -> TargetCompetitorMomentDecomposition:
        try:
            target_index = self.channel_ids.index(target_channel_id)
            competitor_index = self.channel_ids.index(competitor_channel_id)
        except ValueError as error:
            raise ValueError(
                "response moment tensor lacks target/competitor channels"
            ) from error
        target_mean = self.mean[target_index]
        competitor_mean = self.mean[competitor_index]
        target_second = self.second_moment_diagonal[:, target_index, target_index]
        competitor_second = self.second_moment_diagonal[
            :, competitor_index, competitor_index
        ]
        cross = self.second_moment_diagonal[:, target_index, competitor_index]
        return TargetCompetitorMomentDecomposition(
            target_mean=target_mean,
            competitor_mean=competitor_mean,
            target_second_moment=target_second,
            competitor_second_moment=competitor_second,
            cross_second_moment=cross,
            centered_cross_covariance=cross - target_mean * competitor_mean,
            margin_second_moment=target_second + competitor_second - 2.0 * cross,
        )

    @cached_property
    def content_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "channel_ids": self.channel_ids,
                "route_axis_hash": ordered_string_axis_hash(
                    self.route_ids,
                    name="response moment route axis",
                ),
                "route_count": len(self.route_ids),
                "mean": self.mean,
                "second_moment_diagonal": self.second_moment_diagonal,
                "mean_standard_error": self.mean_standard_error,
                "mean_estimator_covariance_diagonal": self.mean_estimator_covariance_diagonal,
                "second_moment_standard_error": self.second_moment_standard_error,
                "sample_count": self.sample_count,
                "response_spec_hash": self.response_spec_hash,
                "metric_id": self.metric_id,
                "context_axis_hash": self.context_axis_hash,
                "metadata": self.metadata,
            },
            prefix="response-moment-",
        )

    @classmethod
    def from_rows(
        cls,
        *,
        channel_ids: Sequence[str],
        route_ids: Sequence[str],
        rows: np.ndarray,
        response_spec_hash: str,
        metric_id: str,
        context_axis_hash: str,
        context_weights: Sequence[float] | np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
        maximum_dense_response_dimension: int = 32,
        route_chunk_size: int = 8192,
    ) -> "ResponseMomentTensor":
        values = np.asarray(rows, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] <= 0 or not np.isfinite(values).all():
            raise ValueError(
                "response rows must have finite [context,response,route] axes"
            )
        n, d, r = values.shape
        if d != len(tuple(channel_ids)) or r != len(tuple(route_ids)):
            raise ValueError("response rows do not align with declared axes")
        if d > int(maximum_dense_response_dimension):
            raise ValueError(
                "dense route-local response moments scale as O(R*d^2); project, sketch, or select an "
                "active competitor response subspace before requesting response_moments"
            )
        if route_chunk_size <= 0:
            raise ValueError("route_chunk_size must be positive")
        weights = _normalized_context_weights(context_weights, n)
        mean = np.einsum("n,ndr->dr", weights, values, optimize=True)
        weight_sum_sq = float(np.sum(weights * weights))
        estimator_denominator = max(1e-15, 1.0 - weight_sum_sq)
        second = np.empty((r, d, d), dtype=np.float64)
        second_se = np.empty((r, d, d), dtype=np.float64)
        mean_estimator_covariance = np.empty((r, d, d), dtype=np.float64)
        for route_start in range(0, r, int(route_chunk_size)):
            route_stop = min(r, route_start + int(route_chunk_size))
            slab = values[:, :, route_start:route_stop]
            slab_second_rows = np.einsum("ndr,ner->nrde", slab, slab, optimize=True)
            slab_second = np.einsum(
                "n,nrde->rde", weights, slab_second_rows, optimize=True
            )
            second[route_start:route_stop] = slab_second
            second_se[route_start:route_stop] = _weighted_mean_standard_error(
                slab_second_rows, weights, slab_second
            )
            slab_centered = slab - mean[:, route_start:route_stop][None, :, :]
            route_mean_covariance = (
                np.einsum(
                    "n,ndr,ner->rde",
                    weights,
                    slab_centered,
                    slab_centered,
                    optimize=True,
                )
                / estimator_denominator
                * weight_sum_sq
            )
            # Roundoff can leave tiny negative eigenvalues.  The batched result
            # is symmetrized here; per-route PSD validation remains in
            # ``__post_init__`` and therefore cannot be bypassed by streaming.
            mean_estimator_covariance[route_start:route_stop] = 0.5 * (
                route_mean_covariance + route_mean_covariance.transpose(0, 2, 1)
            )
        mean_se = np.sqrt(
            np.maximum(np.diagonal(mean_estimator_covariance, axis1=1, axis2=2).T, 0.0)
        )
        return cls(
            channel_ids=tuple(channel_ids),
            route_ids=tuple(route_ids),
            mean=mean,
            second_moment_diagonal=second,
            mean_standard_error=mean_se,
            second_moment_standard_error=second_se,
            sample_count=n,
            response_spec_hash=str(response_spec_hash),
            metric_id=str(metric_id),
            context_axis_hash=str(context_axis_hash),
            mean_estimator_covariance_diagonal=mean_estimator_covariance,
            metadata={
                **dict(metadata or {}),
                "context_weights_sha256": content_hash(weights, prefix="weights-"),
                "estimator_uncertainty_method": "weighted_plugin_mean_se",
                "maximum_dense_response_dimension": int(
                    maximum_dense_response_dimension
                ),
                "route_chunk_size": int(route_chunk_size),
            },
        )

    def action_moments(
        self,
        indices: Sequence[int] | np.ndarray,
        coefficients: Sequence[float] | np.ndarray | None = None,
    ) -> ResponseActionMoments:
        selected, values = _validate_action(
            indices, coefficients, route_count=self.route_count
        )
        d = self.response_dimension
        if selected.size == 0:
            mean = np.zeros(d)
            covariance = np.zeros((d, d))
            mean_se = np.zeros(d)
            mean_estimator_covariance = np.zeros((d, d))
            second_se = np.zeros((d, d))
        else:
            mean = self.mean[:, selected] @ values
            covariance = np.einsum(
                "r,rij->ij",
                values * values,
                self.contextual_covariance_diagonal[selected],
            )
            mean_estimator_covariance = np.einsum(
                "r,rij->ij",
                values * values,
                self.mean_estimator_covariance_diagonal[selected],
            )
            mean_estimator_covariance = _symmetric_psd(
                mean_estimator_covariance,
                name="v1 action mean-estimator covariance",
                tolerance=1e-6,
            )
            mean_se = np.sqrt(np.maximum(np.diag(mean_estimator_covariance), 0.0))
            second_se = np.sqrt(
                np.sum(
                    (
                        self.second_moment_standard_error[selected]
                        * (values * values)[:, None, None]
                    )
                    ** 2,
                    axis=0,
                )
            )
        covariance = _symmetric_psd(
            covariance, name="v1 action contextual covariance", tolerance=1e-5
        )
        second = covariance + np.outer(mean, mean)
        return ResponseActionMoments(
            mean=mean,
            contextual_covariance=covariance,
            second_moment=second,
            mean_standard_error=mean_se,
            second_moment_standard_error=second_se,
            sample_count=self.sample_count,
            approximation="v1_route_local_context_covariance_exact_mean_coupling",
            metadata={
                "route_count": int(selected.size),
                "cross_route_context_covariance_retained": False,
                "coherent_mean_cross_terms_retained": True,
                "cross_route_estimator_covariance_retained": False,
            },
            mean_estimator_covariance=mean_estimator_covariance,
            projection_residual_bound=(
                self.projection_residual_bound_per_unit_route_coefficient
                * float(np.sum(np.abs(values)))
            ),
        )

    def pack(self) -> tuple[np.ndarray, dict[str, Any]]:
        d, r = self.response_dimension, self.route_count
        values = np.concatenate(
            (
                self.mean,
                self.second_moment_diagonal.transpose(1, 2, 0).reshape(d * d, r),
                self.mean_standard_error,
                self.mean_estimator_covariance_diagonal.transpose(1, 2, 0).reshape(
                    d * d, r
                ),
                self.second_moment_standard_error.transpose(1, 2, 0).reshape(d * d, r),
            ),
            axis=0,
        )
        layout = {
            "schema": self.schema,
            "channel_ids": self.channel_ids,
            "route_ids": self.route_ids,
            "response_dimension": d,
            "sample_count": self.sample_count,
            "response_spec_hash": self.response_spec_hash,
            "metric_id": self.metric_id,
            "context_axis_hash": self.context_axis_hash,
            "sections": {
                "mean": [0, d],
                "second_moment_diagonal": [d, d + d * d],
                "mean_standard_error": [d + d * d, 2 * d + d * d],
                "mean_estimator_covariance_diagonal": [
                    2 * d + d * d,
                    2 * d + 2 * d * d,
                ],
                "second_moment_standard_error": [2 * d + 2 * d * d, 2 * d + 3 * d * d],
            },
            "metadata": dict(self.metadata),
        }
        return np.ascontiguousarray(values), layout

    @classmethod
    def unpack(
        cls,
        values: np.ndarray,
        layout: Mapping[str, Any],
        *,
        route_ids: Sequence[str] | None = None,
    ) -> "ResponseMomentTensor":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or layout.get("schema") != "response_moment_tensor_v1":
            raise ValueError("measurement is not a packed response moment tensor")
        d = int(layout["response_dimension"])
        routes = resolve_packed_string_axis(
            layout,
            expected_axis=route_ids,
            axis_key="route_ids",
            name="response moment route axis",
        )
        if array.shape[1] != len(routes):
            raise ValueError(
                "packed response moment route axis differs from its layout"
            )
        sections = dict(layout["sections"])

        def take(name: str) -> np.ndarray:
            start, stop = map(int, sections[name])
            return array[start:stop]

        second = (
            take("second_moment_diagonal").reshape(d, d, len(routes)).transpose(2, 0, 1)
        )
        second_se = (
            take("second_moment_standard_error")
            .reshape(d, d, len(routes))
            .transpose(2, 0, 1)
        )
        mean_estimator_covariance = None
        if "mean_estimator_covariance_diagonal" in sections:
            mean_estimator_covariance = (
                take("mean_estimator_covariance_diagonal")
                .reshape(d, d, len(routes))
                .transpose(2, 0, 1)
            )
        return cls(
            channel_ids=tuple(layout["channel_ids"]),
            route_ids=routes,
            mean=take("mean"),
            second_moment_diagonal=second,
            mean_standard_error=take("mean_standard_error"),
            second_moment_standard_error=second_se,
            sample_count=int(layout["sample_count"]),
            response_spec_hash=str(layout["response_spec_hash"]),
            metric_id=str(layout["metric_id"]),
            context_axis_hash=str(layout["context_axis_hash"]),
            mean_estimator_covariance_diagonal=mean_estimator_covariance,
            metadata=dict(layout.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# v2 full empirical factor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResponseFactorTensor:
    """Context-by-response-by-route factor retaining cross-route coupling."""

    channel_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    rows: np.ndarray = field(repr=False, compare=False)  # [context,response,route]
    response_spec_hash: str
    metric_id: str
    context_weights: np.ndarray | None = field(default=None, repr=False, compare=False)
    projection_residual_bound: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "response_factor_tensor_v1"

    def __post_init__(self) -> None:
        channels = tuple(map(str, self.channel_ids))
        routes = tuple(map(str, self.route_ids))
        contexts = tuple(map(str, self.context_ids))
        if not channels or not routes or not contexts:
            raise ValueError("response factor tensor requires nonempty axes")
        if (
            len(set(channels)) != len(channels)
            or len(set(routes)) != len(routes)
            or len(set(contexts)) != len(contexts)
        ):
            raise ValueError("response factor tensor axes must be unique")
        rows = _readonly_array(self.rows, ndim=3, name="response factor rows")
        if rows.shape != (len(contexts), len(channels), len(routes)):
            raise ValueError("response factor rows have incompatible axes")
        if not self.response_spec_hash or not self.metric_id or not self.schema:
            raise ValueError("response factor tensor requires scientific identity")
        weights = _normalized_context_weights(self.context_weights, len(contexts))
        bound = float(self.projection_residual_bound)
        if not np.isfinite(bound) or bound < 0.0:
            raise ValueError("projection_residual_bound must be finite and nonnegative")
        object.__setattr__(self, "channel_ids", channels)
        object.__setattr__(self, "route_ids", routes)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "context_weights", weights)
        object.__setattr__(self, "projection_residual_bound", bound)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def sample_count(self) -> int:
        return len(self.context_ids)

    @property
    def response_dimension(self) -> int:
        return len(self.channel_ids)

    @property
    def route_count(self) -> int:
        return len(self.route_ids)

    @cached_property
    def content_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "channel_ids": self.channel_ids,
                "route_ids": self.route_ids,
                "context_ids": self.context_ids,
                "rows": self.rows,
                "response_spec_hash": self.response_spec_hash,
                "metric_id": self.metric_id,
                "context_weights": self.context_weights,
                "projection_residual_bound": self.projection_residual_bound,
                "metadata": self.metadata,
            },
            prefix="response-factor-",
        )

    @cached_property
    def marginal_moments(self) -> ResponseMomentTensor:
        return ResponseMomentTensor.from_rows(
            channel_ids=self.channel_ids,
            route_ids=self.route_ids,
            rows=self.rows,
            response_spec_hash=self.response_spec_hash,
            metric_id=self.metric_id,
            context_axis_hash=content_hash(
                self.context_ids, prefix="response-context-axis-"
            ),
            context_weights=self.context_weights,
            metadata={
                "projected_from": self.schema,
                "projection_residual_bound": self.projection_residual_bound,
                **dict(self.metadata),
            },
        )

    def action_rows(
        self,
        indices: Sequence[int] | np.ndarray,
        coefficients: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        selected, values = _validate_action(
            indices, coefficients, route_count=self.route_count
        )
        if selected.size == 0:
            return np.zeros(
                (self.sample_count, self.response_dimension), dtype=np.float64
            )
        return np.einsum("ndr,r->nd", self.rows[:, :, selected], values)

    def action_moments(
        self,
        indices: Sequence[int] | np.ndarray,
        coefficients: Sequence[float] | np.ndarray | None = None,
    ) -> ResponseActionMoments:
        selected, values = _validate_action(
            indices, coefficients, route_count=self.route_count
        )
        action_rows = self.action_rows(selected, values)
        weights = np.asarray(self.context_weights, dtype=np.float64)
        mean = np.einsum("n,nd->d", weights, action_rows)
        centered = action_rows - mean[None, :]
        covariance = np.einsum("n,nd,ne->de", weights, centered, centered)
        covariance = _symmetric_psd(
            covariance, name="v2 action contextual covariance", tolerance=1e-6
        )
        second_rows = np.einsum("nd,ne->nde", action_rows, action_rows)
        second = np.einsum("n,nde->de", weights, second_rows)
        mean_estimator_covariance = _weighted_mean_estimator_covariance(
            action_rows, weights, mean
        )
        mean_se = np.sqrt(np.maximum(np.diag(mean_estimator_covariance), 0.0))
        second_se = _weighted_mean_standard_error(second_rows, weights, second)
        action_projection_bound = float(self.projection_residual_bound) * float(
            np.sum(np.abs(values))
        )
        return ResponseActionMoments(
            mean=mean,
            contextual_covariance=covariance,
            second_moment=second,
            mean_standard_error=mean_se,
            second_moment_standard_error=second_se,
            sample_count=self.sample_count,
            approximation="v2_full_empirical_cross_route_response_covariance",
            metadata={
                "route_count": int(selected.size),
                "projection_residual_bound_per_unit_route_coefficient": self.projection_residual_bound,
                "action_projection_residual_bound": action_projection_bound,
                "cross_route_context_covariance_retained": True,
                "cross_route_estimator_covariance_retained": True,
            },
            mean_estimator_covariance=mean_estimator_covariance,
            projection_residual_bound=action_projection_bound,
        )

    def screened(
        self, route_indices: Sequence[int] | np.ndarray
    ) -> "ResponseFactorTensor":
        indices = np.asarray(tuple(int(item) for item in route_indices), dtype=np.int64)
        if (
            indices.size == 0
            or np.any(indices < 0)
            or np.any(indices >= self.route_count)
            or np.unique(indices).size != indices.size
        ):
            raise ValueError("screened response-factor indices are invalid")
        return ResponseFactorTensor(
            channel_ids=self.channel_ids,
            route_ids=tuple(self.route_ids[index] for index in indices),
            context_ids=self.context_ids,
            rows=self.rows[:, :, indices],
            response_spec_hash=self.response_spec_hash,
            metric_id=self.metric_id,
            context_weights=self.context_weights,
            projection_residual_bound=self.projection_residual_bound,
            metadata={
                **dict(self.metadata),
                "parent_factor_hash": self.content_hash,
                "screen_indices": indices.tolist(),
            },
        )


# ---------------------------------------------------------------------------
# v4 finite-action model validity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResponseModelValidityReceipt:
    """Matrix-valued residual between local and exact finite-mask responses."""

    predicted: ResponseActionMoments
    exact: ResponseActionMoments
    residual: ResponseActionMoments
    normalized_second_moment_error: float
    normalized_mean_error: float
    active_competitor_changed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.predicted.response_dimension != self.exact.response_dimension
            or self.residual.response_dimension != self.exact.response_dimension
        ):
            raise ValueError(
                "model-validity moments have incompatible response dimensions"
            )
        for name in ("normalized_second_moment_error", "normalized_mean_error"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("model-validity errors must be finite and nonnegative")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def normalized_operator_error(self) -> float:
        """Compatibility alias."""

        return float(self.normalized_second_moment_error)

    @classmethod
    def from_response_rows(
        cls,
        *,
        predicted_rows: np.ndarray,
        exact_rows: np.ndarray,
        normalization_metric: np.ndarray | None = None,
        response_scale: float = 1.0,
        active_competitor_changed: bool = False,
        context_weights: Sequence[float] | np.ndarray | None = None,
        normalization_whitener: np.ndarray | None = None,
        normalization_directions: Sequence[Sequence[float]] | np.ndarray | None = None,
        normalization_tolerances: Sequence[float] | np.ndarray | None = None,
        normalization_energy_metric: (
            Sequence[Sequence[float]] | np.ndarray | None
        ) = None,
        normalization_energy_tolerance: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ResponseModelValidityReceipt":
        predicted = np.asarray(predicted_rows, dtype=np.float64)
        exact = np.asarray(exact_rows, dtype=np.float64)
        if (
            predicted.shape != exact.shape
            or predicted.ndim != 2
            or predicted.shape[0] <= 0
            or not np.isfinite(predicted).all()
            or not np.isfinite(exact).all()
        ):
            raise ValueError(
                "predicted and exact response rows must share finite [context,response] axes"
            )
        if not np.isfinite(response_scale) or response_scale <= 0.0:
            raise ValueError("response_scale must be finite and positive")
        weights = _normalized_context_weights(context_weights, predicted.shape[0])

        def moments(rows: np.ndarray, label: str) -> ResponseActionMoments:
            mean = np.einsum("n,nd->d", weights, rows)
            centered = rows - mean[None, :]
            covariance = np.einsum("n,nd,ne->de", weights, centered, centered)
            second_rows = np.einsum("nd,ne->nde", rows, rows)
            second = np.einsum("n,nde->de", weights, second_rows)
            mean_estimator_covariance = _weighted_mean_estimator_covariance(
                rows, weights, mean
            )
            return ResponseActionMoments(
                mean=mean,
                contextual_covariance=covariance,
                second_moment=second,
                mean_standard_error=np.sqrt(
                    np.maximum(np.diag(mean_estimator_covariance), 0.0)
                ),
                second_moment_standard_error=_weighted_mean_standard_error(
                    second_rows, weights, second
                ),
                sample_count=rows.shape[0],
                approximation=label,
                mean_estimator_covariance=mean_estimator_covariance,
            )

        predicted_moments = moments(predicted, "local_linear_response_prediction")
        exact_moments = moments(exact, "exact_finite_hard_mask_response")
        residual_rows = exact - predicted
        residual_moments = moments(residual_rows, "finite_action_response_residual")
        mean_delta = exact_moments.mean - predicted_moments.mean
        second_delta = exact_moments.second_moment - predicted_moments.second_moment
        mean_components: dict[str, float] = {}
        second_components: dict[str, float] = {}

        directions_supplied = normalization_directions is not None
        if directions_supplied:
            directions = np.asarray(normalization_directions, dtype=np.float64)
            tolerances = np.asarray(normalization_tolerances, dtype=np.float64).reshape(
                -1
            )
            if (
                directions.ndim != 2
                or directions.shape[1] != predicted.shape[1]
                or directions.shape[0] != tolerances.size
                or directions.shape[0] <= 0
                or not np.isfinite(directions).all()
                or not np.isfinite(tolerances).all()
                or np.any(tolerances <= 0.0)
            ):
                raise ValueError(
                    "normalization directions/tolerances must be finite and align with the response axis"
                )
            directional_mean = np.abs(directions @ mean_delta) / tolerances
            directional_second = np.abs(
                np.einsum(
                    "qd,de,qe->q", directions, second_delta, directions, optimize=True
                )
            ) / (tolerances * tolerances)
            mean_components["safe_region_directions"] = float(np.max(directional_mean))
            second_components["safe_region_directions"] = float(
                np.max(directional_second)
            )

        energy_supplied = normalization_energy_metric is not None
        if energy_supplied:
            energy_metric = _symmetric_psd(
                np.asarray(normalization_energy_metric, dtype=np.float64),
                name="model-validity response-constraint energy metric",
            )
            if energy_metric.shape != predicted_moments.second_moment.shape:
                raise ValueError(
                    "normalization energy metric has incompatible response dimension"
                )
            tolerance = float(normalization_energy_tolerance)
            if not np.isfinite(tolerance) or tolerance <= 0.0:
                raise ValueError(
                    "normalization energy tolerance must be finite and positive"
                )
            root = _matrix_square_root(energy_metric)
            mean_components["safe_region_energy"] = float(
                np.linalg.norm(root @ mean_delta) / tolerance
            )
            second_components["safe_region_energy"] = float(
                np.linalg.norm(root @ second_delta @ root, ord=2)
                / (tolerance * tolerance)
            )

        if normalization_whitener is not None:
            whitener = np.asarray(normalization_whitener, dtype=np.float64)
            if (
                whitener.ndim != 3
                or whitener.shape[0] != predicted.shape[0]
                or whitener.shape[2] != predicted.shape[1]
                or whitener.shape[1] <= 0
                or not np.isfinite(whitener).all()
            ):
                raise ValueError(
                    "normalization_whitener must be finite [context,metric_factor,response]"
                )
            predicted_normalized = np.einsum(
                "nkd,nd->nk", whitener, predicted, optimize=True
            )
            exact_normalized = np.einsum("nkd,nd->nk", whitener, exact, optimize=True)
            predicted_mean = np.einsum("n,nk->k", weights, predicted_normalized)
            exact_mean = np.einsum("n,nk->k", weights, exact_normalized)
            predicted_second = np.einsum(
                "n,nk,nl->kl",
                weights,
                predicted_normalized,
                predicted_normalized,
                optimize=True,
            )
            exact_second = np.einsum(
                "n,nk,nl->kl",
                weights,
                exact_normalized,
                exact_normalized,
                optimize=True,
            )
            second_components["context_dependent_metric_whitener"] = float(
                np.linalg.norm(exact_second - predicted_second, ord=2)
                / (response_scale * response_scale)
            )
            mean_components["context_dependent_metric_whitener"] = float(
                np.linalg.norm(exact_mean - predicted_mean) / response_scale
            )

        # A generic fixed metric is a compatibility fallback.  Once the frozen
        # safe-region directions or energy tube are supplied, validity is
        # normalized in those scientific units rather than by an unrelated
        # scalar oracle-loss tolerance.
        if not mean_components and not second_components:
            metric = _symmetric_psd(
                (
                    np.eye(predicted.shape[1])
                    if normalization_metric is None
                    else normalization_metric
                ),
                name="model-validity normalization metric",
            )
            if metric.shape != predicted_moments.second_moment.shape:
                raise ValueError(
                    "normalization metric has incompatible response dimension"
                )
            root = _matrix_square_root(metric)
            mean_components["fixed_response_metric"] = float(
                np.linalg.norm(root @ mean_delta) / response_scale
            )
            second_components["fixed_response_metric"] = float(
                np.linalg.norm(root @ second_delta @ root, ord=2)
                / (response_scale * response_scale)
            )

        mean_error = max(mean_components.values(), default=0.0)
        second_error = max(second_components.values(), default=0.0)
        normalization_mode = "+".join(
            sorted(set(mean_components) | set(second_components))
        )
        normalized_dimension = (
            int(np.asarray(normalization_whitener).shape[1])
            if normalization_whitener is not None
            else predicted.shape[1]
        )
        return cls(
            predicted=predicted_moments,
            exact=exact_moments,
            residual=residual_moments,
            normalized_second_moment_error=second_error,
            normalized_mean_error=mean_error,
            active_competitor_changed=bool(active_competitor_changed),
            metadata={
                **dict(metadata or {}),
                "response_scale": float(response_scale),
                "normalization_metric_is_a_metric_not_a_covariance_scale": True,
                "normalization_mode": normalization_mode,
                "normalized_response_dimension": int(normalized_dimension),
                "context_weights_supplied": context_weights is not None,
                "normalization_mean_components": mean_components,
                "normalization_second_moment_components": second_components,
                "safe_region_normalization_used": bool(
                    directions_supplied or energy_supplied
                ),
            },
        )

    def recommendation(
        self,
        *,
        reuse_threshold: float,
        refresh_threshold: float,
        competitor_switch_requires_refresh: bool = True,
    ) -> str:
        if competitor_switch_requires_refresh and self.active_competitor_changed:
            return "refresh"
        worst = max(self.normalized_second_moment_error, self.normalized_mean_error)
        if worst <= float(reuse_threshold):
            return "reuse"
        if worst > float(refresh_threshold):
            return "refresh"
        return "shrink"


@dataclass(frozen=True, slots=True)
class ResponseTransitionReceipt:
    """Exact finite-action calibration of one local response prediction."""

    action_indices: tuple[int, ...]
    coefficients: tuple[float, ...]
    validity: ResponseModelValidityReceipt
    predicted_risk: float
    exact_risk: float
    state_hash: str
    candidate_mask_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw_indices = tuple(int(value) for value in self.action_indices)
        raw_coefficients = tuple(float(value) for value in self.coefficients)
        if len(raw_indices) != len(raw_coefficients):
            raise ValueError(
                "response transition receipt requires one coefficient per route"
            )
        pairs = tuple(
            sorted(zip(raw_indices, raw_coefficients), key=lambda item: item[0])
        )
        indices = tuple(item[0] for item in pairs)
        coefficients = tuple(item[1] for item in pairs)
        if (
            not indices
            or len(indices) != len(coefficients)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError(
                "response transition receipt requires one coefficient per unique route"
            )
        if any(not np.isfinite(value) for value in coefficients):
            raise ValueError("response transition coefficients must be finite")
        if any(
            not np.isfinite(float(getattr(self, name)))
            or float(getattr(self, name)) < 0.0
            for name in ("predicted_risk", "exact_risk")
        ):
            raise ValueError("response transition risks must be finite and nonnegative")
        if not self.state_hash or not self.candidate_mask_hash:
            raise ValueError(
                "response transition receipt requires state and mask identity"
            )
        object.__setattr__(self, "action_indices", indices)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def action_key(self) -> tuple[tuple[int, ...], tuple[float, ...]]:
        return self.action_indices, self.coefficients

    @property
    def action_l1_magnitude(self) -> float:
        """Return the physical L1 gate displacement represented by the receipt.

        The quantity is the correct homogeneity scale for the conservative v4
        residual envelope.  A second-moment residual scales quadratically with
        an action's first-order magnitude, so envelopes are learned per unit
        squared L1 displacement and re-expanded for a proposed action.
        """

        return float(np.sum(np.abs(np.asarray(self.coefficients, dtype=np.float64))))

    @property
    def normalized_residual_second_moment(self) -> np.ndarray:
        """Residual second moment per unit squared L1 action magnitude."""

        scale = self.action_l1_magnitude
        if scale <= 0.0:
            raise ValueError("response transition receipt has zero action magnitude")
        return np.asarray(self.validity.residual.second_moment, dtype=np.float64) / (
            scale * scale
        )

    @property
    def content_hash(self) -> str:
        return content_hash(
            {
                "action_indices": self.action_indices,
                "coefficients": self.coefficients,
                "predicted": self.validity.predicted.to_dict(),
                "exact": self.validity.exact.to_dict(),
                "residual": self.validity.residual.to_dict(),
                "normalized_second_moment_error": self.validity.normalized_second_moment_error,
                "normalized_mean_error": self.validity.normalized_mean_error,
                "active_competitor_changed": self.validity.active_competitor_changed,
                "predicted_risk": self.predicted_risk,
                "exact_risk": self.exact_risk,
                "state_hash": self.state_hash,
                "candidate_mask_hash": self.candidate_mask_hash,
                "metadata": self.metadata,
            },
            prefix="response-transition-",
        )


RESPONSE_TRANSITION_CACHE_KEY = "irg_latest_response_transition_receipt_v1"


def response_transition_receipt_from_outcome(
    outcome: MeasurementOutcome,
    *,
    expected_response_dimension: int | None = None,
    receipt_state_hash: str | None = None,
) -> ResponseTransitionReceipt:
    """Construct one typed exact-transition receipt from a provider outcome.

    This parser is deliberately shared by ``irg/v1``--``irg/v4``.  The
    response-transition measurement is a property of the common vector-valued
    response law, not of the global planner alone.  Version-specific state may
    retain additional envelopes, but every typed version can use the same
    exact-versus-predicted response rows to calibrate relinearization.
    """

    values = np.asarray(outcome.values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] % 2 != 0:
        raise ValueError(
            "response_transition values must concatenate finite predicted and exact [context,response] rows"
        )
    dimension = values.shape[1] // 2
    declared = int(outcome.metadata.get("response_dimension", dimension))
    if declared != dimension:
        raise ValueError(
            "response_transition metadata disagrees with the measured response dimension"
        )
    if expected_response_dimension is not None and dimension != int(
        expected_response_dimension
    ):
        raise ValueError(
            "response_transition outcome uses an incompatible response dimension"
        )

    predicted_rows, exact_rows = values[:, :dimension], values[:, dimension:]
    normalization = np.asarray(
        outcome.metadata.get("normalization_metric", np.eye(dimension)),
        dtype=np.float64,
    )
    validity = ResponseModelValidityReceipt.from_response_rows(
        predicted_rows=predicted_rows,
        exact_rows=exact_rows,
        normalization_metric=normalization,
        response_scale=float(outcome.metadata.get("response_scale", 1.0)),
        active_competitor_changed=bool(
            outcome.metadata.get("active_competitor_changed", False)
        ),
        context_weights=outcome.metadata.get("context_weights"),
        normalization_whitener=outcome.metadata.get("normalization_whitener"),
        normalization_directions=outcome.metadata.get("normalization_directions"),
        normalization_tolerances=outcome.metadata.get("normalization_tolerances"),
        normalization_energy_metric=outcome.metadata.get("normalization_energy_metric"),
        normalization_energy_tolerance=outcome.metadata.get(
            "normalization_energy_tolerance"
        ),
        metadata={
            "measurement_integrity_hash": outcome.integrity_hash,
            "measurement_request_id": outcome.request_id,
            "calibration_state_id": outcome.calibration_state_id,
            "response_constraint_hash": outcome.metadata.get(
                "response_constraint_hash", ""
            ),
            "normalization_directions_transmitted": outcome.metadata.get(
                "normalization_directions"
            )
            is not None,
            "normalization_energy_transmitted": outcome.metadata.get(
                "normalization_energy_metric"
            )
            is not None,
        },
    )
    action_indices = tuple(
        int(value) for value in outcome.metadata.get("action_indices", ())
    )
    coefficients = tuple(
        float(value)
        for value in outcome.metadata.get("coefficients", (-1.0,) * len(action_indices))
    )
    candidate_mask_hash = str(
        outcome.metadata.get(
            "candidate_mask_hash",
            outcome.candidate_ids[0] if outcome.candidate_ids else "unknown-mask",
        )
    )
    return ResponseTransitionReceipt(
        action_indices=action_indices,
        coefficients=coefficients,
        validity=validity,
        predicted_risk=float(outcome.metadata.get("predicted_risk", 0.0)),
        exact_risk=float(outcome.metadata.get("exact_risk", 0.0)),
        state_hash=str(receipt_state_hash or outcome.state_hash),
        candidate_mask_hash=candidate_mask_hash,
        metadata={
            "outcome_hash": outcome.integrity_hash,
            "outcome_state_hash": outcome.state_hash,
            "response_dimension": dimension,
            "response_constraint_hash": outcome.metadata.get(
                "response_constraint_hash", ""
            ),
            "safe_region_normalization_used": validity.metadata.get(
                "safe_region_normalization_used", False
            ),
            "normalization_mode": validity.metadata.get("normalization_mode", ""),
        },
    )


@dataclass(frozen=True, slots=True)
class ResponseEllipsoidTrustRegionDecision:
    action: str
    reason: str
    recommended_scale: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResponseEllipsoidTrustRegion:
    """Guidance-only matrix-valued local-model validity controller."""

    reuse_threshold: float = 0.10
    refresh_threshold: float = 0.35
    competitor_switch_requires_refresh: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.reuse_threshold < self.refresh_threshold:
            raise ValueError(
                "trust-region thresholds must satisfy 0 <= reuse < refresh"
            )

    def evaluate(
        self, receipt: ResponseTransitionReceipt
    ) -> ResponseEllipsoidTrustRegionDecision:
        recommendation = receipt.validity.recommendation(
            reuse_threshold=self.reuse_threshold,
            refresh_threshold=self.refresh_threshold,
            competitor_switch_requires_refresh=self.competitor_switch_requires_refresh,
        )
        worst = max(
            receipt.validity.normalized_mean_error,
            receipt.validity.normalized_second_moment_error,
        )
        if recommendation == "reuse":
            scale, reason = (
                1.0,
                "exact and predicted response ellipsoids agree within the reuse region",
            )
        elif recommendation == "shrink":
            scale = max(0.1, min(0.9, self.reuse_threshold / max(worst, 1e-12)))
            reason = (
                "local response geometry remains usable only at a smaller action scale"
            )
        else:
            scale = 0.0
            reason = (
                "active competitor changed under the exact action"
                if receipt.validity.active_competitor_changed
                else "exact response residual invalidates the current local geometry"
            )
        return ResponseEllipsoidTrustRegionDecision(
            action=recommendation,
            reason=reason,
            recommended_scale=float(scale),
            diagnostics={
                "normalized_mean_error": receipt.validity.normalized_mean_error,
                "normalized_second_moment_error": receipt.validity.normalized_second_moment_error,
                "predicted_risk": receipt.predicted_risk,
                "exact_risk": receipt.exact_risk,
                "active_competitor_changed": receipt.validity.active_competitor_changed,
                "transition_receipt_hash": receipt.content_hash,
            },
        )


# Compatibility names used by the previous draft and downstream receipts.
ActionResponseMoments = ResponseActionMoments
ResponseMomentSummary = ResponseMomentTensor
ResponseMomentFactor = ResponseFactorTensor


__all__ = [
    "ActionResponseMoments",
    "ResponseActionMoments",
    "ResponseConstraintEvaluator",
    "ResponseFactorTensor",
    "ResponseModelValidityReceipt",
    "ResponseProjectionReceipt",
    "ResponseMomentFactor",
    "ResponseMomentSummary",
    "ResponseMomentTensor",
    "ResponseRiskEvaluation",
    "TargetCompetitorMomentDecomposition",
    "ResponseTransitionReceipt",
    "RESPONSE_TRANSITION_CACHE_KEY",
    "response_transition_receipt_from_outcome",
    "ResponseEllipsoidTrustRegion",
    "ResponseEllipsoidTrustRegionDecision",
]
