"""Streaming providers for matrix-valued IRG response geometry.

The module is the production-facing counterpart of the retained-array category
contrast provider.  It never requires a complete context x response x native
route tensor for v1.  Instead it accumulates the exact first two semantic
response moments, their plug-in estimator covariance, and entrywise
second-moment uncertainty in O(R d^2) memory.  v2--v4 request a factor only on
an already screened physical route subset and can write that factor directly to
an immutable NPY artifact.

The callback is intentionally semantic: it returns physical route derivatives
of the *frozen declared response coordinates*.  Model-specific code may obtain
those derivatives from logits, VJPs, JVPs, or an existing factor store without
changing the IRG scientific contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from itertools import chain
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, cast
import warnings

import numpy as np

from mechanistic_systems.core.axis_identity import (
    canonical_unique_string_axis,
    ordered_string_axis_hash,
    resolve_packed_string_axis,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.functional_atlas import FunctionalAtlasSpec
from mechanistic_systems.contracts.measurement import (
    ArtifactRef,
    MeasurementOutcome,
    MeasurementRequest,
)
from mechanistic_systems.contracts.measurement_costs import RealizedCost
from mechanistic_systems.contracts.trust_geometry import GlobalTrustGeometrySpec
from mechanistic_systems.geometry.trust import (
    FrequentDirectionsTrustAccumulator,
    GlobalTrustGeometryState,
)
from mechanistic_systems.measurement.response_geometry import ResponseMomentTensor
from mechanistic_systems.measurement.gate_geometry import (
    LocalRegimeRouteSummary,
    SmoothTransmissionRouteSummary,
)
from mechanistic_systems.geometry.quotient import (
    IdentificationPolicy,
    pseudoinverse_spectrum,
)
from mechanistic_systems.measurement.local_regime import (
    REGIME_STATISTIC_NAMES,
    SMOOTH_TRANSMISSION_STATISTIC_NAMES,
)


@dataclass(frozen=True, slots=True)
class ResponseJacobianBatch:
    """One ordered semantic response-Jacobian batch.

    ``route_rows`` has ``[context,response,route]`` axes.  Optional
    ``nuisance_rows`` has ``[context,response,nuisance]`` axes.  ``weights`` are
    nonnegative *raw* design weights; normalization is global across batches.
    """

    context_ids: tuple[str, ...]
    route_rows: np.ndarray
    nuisance_rows: np.ndarray | None = None
    metric_whitener: np.ndarray | None = None
    metric_route_rows: np.ndarray | None = None
    metric_nuisance_rows: np.ndarray | None = None
    weights: np.ndarray | None = None
    local_route_rows: np.ndarray | None = None
    local_destination_transport_hash: str | None = None
    loss_route_rows: np.ndarray | None = None
    loss_pullback_hash: str | None = None
    local_regime_summary: (
        LocalRegimeRouteSummary | SmoothTransmissionRouteSummary | None
    ) = None

    def __post_init__(self) -> None:
        ids = tuple(map(str, self.context_ids))
        route = np.asarray(self.route_rows, dtype=np.float64)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(
                "streaming response batch requires unique nonempty context IDs"
            )
        if (
            route.ndim != 3
            or route.shape[0] != len(ids)
            or not np.isfinite(route).all()
        ):
            raise ValueError("route_rows must be finite [context,response,route]")
        nuisance = None
        if self.nuisance_rows is not None:
            nuisance = np.asarray(self.nuisance_rows, dtype=np.float64)
            if (
                nuisance.ndim != 3
                or nuisance.shape[:2] != route.shape[:2]
                or not np.isfinite(nuisance).all()
            ):
                raise ValueError("nuisance_rows must share context and response axes")
        whitener = None
        if self.metric_whitener is not None:
            whitener = np.asarray(self.metric_whitener, dtype=np.float64)
            if (
                whitener.ndim != 3
                or whitener.shape[0] != len(ids)
                or whitener.shape[2] != route.shape[1]
                or whitener.shape[1] <= 0
                or not np.isfinite(whitener).all()
            ):
                raise ValueError(
                    "metric_whitener must be finite [context,metric_factor,response]"
                )
        metric_route = None
        if self.metric_route_rows is not None:
            metric_route = np.asarray(self.metric_route_rows, dtype=np.float64)
            if (
                metric_route.ndim != 3
                or metric_route.shape[0] != len(ids)
                or metric_route.shape[2] != route.shape[2]
                or metric_route.shape[1] <= 0
                or not np.isfinite(metric_route).all()
            ):
                raise ValueError(
                    "metric_route_rows must be finite [context,metric_factor,route]"
                )
        metric_nuisance = None
        if self.metric_nuisance_rows is not None:
            metric_nuisance = np.asarray(self.metric_nuisance_rows, dtype=np.float64)
            if (
                nuisance is None
                or metric_route is None
                or metric_nuisance.ndim != 3
                or metric_nuisance.shape[0] != len(ids)
                or metric_nuisance.shape[1] != metric_route.shape[1]
                or metric_nuisance.shape[2] != nuisance.shape[2]
                or not np.isfinite(metric_nuisance).all()
            ):
                raise ValueError(
                    "metric_nuisance_rows must align with metric route and semantic nuisance axes"
                )
        if whitener is not None and metric_route is not None:
            raise ValueError(
                "supply a metric whitener or direct metric factors, not both"
            )
        if (
            metric_route is not None
            and nuisance is not None
            and metric_nuisance is None
        ):
            raise ValueError(
                "nuisance-bearing direct metric factors require metric_nuisance_rows"
            )
        weights = None
        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
            if (
                weights.shape != (len(ids),)
                or np.any(weights < 0.0)
                or not np.isfinite(weights).all()
            ):
                raise ValueError(
                    "streaming response weights must be finite and nonnegative"
                )
            if float(np.sum(weights)) <= 0.0:
                raise ValueError("streaming response weights must have positive mass")
        local = None
        transport_hash = self.local_destination_transport_hash
        if self.local_route_rows is not None:
            local = np.asarray(self.local_route_rows, dtype=np.float64)
            if (
                local.ndim != 2
                or local.shape != (len(ids), route.shape[2])
                or not np.isfinite(local).all()
                or not str(transport_hash or "")
            ):
                raise ValueError(
                    "local_route_rows require finite [context,route] rows and "
                    "a destination-transport identity"
                )
        elif transport_hash is not None:
            raise ValueError(
                "a local destination transport may not be declared without local rows"
            )
        loss = None
        loss_hash = self.loss_pullback_hash
        if self.loss_route_rows is not None:
            loss = np.asarray(self.loss_route_rows, dtype=np.float64)
            if (
                loss.shape != (len(ids), route.shape[2])
                or not np.isfinite(loss).all()
                or not str(loss_hash or "")
            ):
                raise ValueError(
                    "loss_route_rows require finite [context,route] rows and "
                    "a frozen loss-pullback identity"
                )
            if local is None:
                raise ValueError(
                    "loss pullback rows require the nested local pullback head"
                )
        elif loss_hash is not None:
            raise ValueError("loss pullback identity requires loss rows")
        regime = self.local_regime_summary
        if regime is not None:
            if len(regime.route_ids) != route.shape[2]:
                raise ValueError(
                    "local regime summary differs from the response route axis"
                )
        route = np.ascontiguousarray(route)
        route.setflags(write=False)
        if nuisance is not None:
            nuisance = np.ascontiguousarray(nuisance)
            nuisance.setflags(write=False)
        if whitener is not None:
            whitener = np.ascontiguousarray(whitener)
            whitener.setflags(write=False)
        if metric_route is not None:
            metric_route = np.ascontiguousarray(metric_route)
            metric_route.setflags(write=False)
        if metric_nuisance is not None:
            metric_nuisance = np.ascontiguousarray(metric_nuisance)
            metric_nuisance.setflags(write=False)
        if weights is not None:
            weights = np.ascontiguousarray(weights)
            weights.setflags(write=False)
        if local is not None:
            local = np.ascontiguousarray(local)
            local.setflags(write=False)
        if loss is not None:
            loss = np.ascontiguousarray(loss)
            loss.setflags(write=False)
        object.__setattr__(self, "context_ids", ids)
        object.__setattr__(self, "route_rows", route)
        object.__setattr__(self, "nuisance_rows", nuisance)
        object.__setattr__(self, "metric_whitener", whitener)
        object.__setattr__(self, "metric_route_rows", metric_route)
        object.__setattr__(self, "metric_nuisance_rows", metric_nuisance)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "local_route_rows", local)
        object.__setattr__(
            self,
            "local_destination_transport_hash",
            None if transport_hash is None else str(transport_hash),
        )
        object.__setattr__(self, "loss_route_rows", loss)
        object.__setattr__(
            self,
            "loss_pullback_hash",
            None if loss_hash is None else str(loss_hash),
        )
        object.__setattr__(self, "local_regime_summary", regime)


@dataclass(frozen=True, slots=True)
class LocalAttributionMomentTensor:
    """Whole-route local-attribution moments and covariance-energy sketch.

    The signed mean defines the modern local functional partition.  Exact
    route-diagonal contextual energy and its sampling uncertainty are retained
    for every route.  ``covariance_factor`` is a deterministic Rademacher
    context sketch: for an action ``a``, ``||F @ a||²`` estimates its centered
    local-response covariance energy without materializing an R×R matrix.
    It is guidance/trust geometry, never claim-bearing safety evidence.
    """

    route_ids: tuple[str, ...]
    mean: np.ndarray
    second_moment: np.ndarray
    mean_standard_error: np.ndarray
    second_moment_standard_error: np.ndarray
    covariance_factor: np.ndarray
    sample_count: int
    context_axis_hash: str
    destination_transport_hash: str
    sketch_seed: int
    schema: str = "local_attribution_moment_tensor_v1"
    _content_hash: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        routes = canonical_unique_string_axis(
            self.route_ids,
            name="local attribution route axis",
        )
        size = len(routes)
        if (
            int(self.sample_count) <= 0
            or not self.context_axis_hash
            or not self.destination_transport_hash
        ):
            raise ValueError("local attribution moment identity is incomplete")
        arrays: dict[str, np.ndarray] = {}
        for name in (
            "mean",
            "second_moment",
            "mean_standard_error",
            "second_moment_standard_error",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite on the route axis")
            if name != "mean" and np.any(value < 0.0):
                raise ValueError(f"{name} must be nonnegative")
            value = np.ascontiguousarray(value)
            value.setflags(write=False)
            arrays[name] = value
        factor = np.asarray(self.covariance_factor, dtype=np.float64)
        if (
            factor.ndim != 2
            or factor.shape[1] != size
            or not np.isfinite(factor).all()
        ):
            raise ValueError(
                "local covariance factor must be finite [sketch_rank,route]"
            )
        factor = np.ascontiguousarray(factor)
        factor.setflags(write=False)
        object.__setattr__(self, "route_ids", routes)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "covariance_factor", factor)
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "sketch_seed", int(self.sketch_seed))

    @property
    def contextual_variance(self) -> np.ndarray:
        return np.maximum(self.second_moment - self.mean * self.mean, 0.0)

    @property
    def content_hash(self) -> str:
        cached = self._content_hash
        if cached is None:
            cached = content_hash(
                {
                    "schema": self.schema,
                    "route_axis_hash": ordered_string_axis_hash(
                        self.route_ids,
                        name="local attribution route axis",
                    ),
                    "route_count": len(self.route_ids),
                    "mean": self.mean,
                    "second_moment": self.second_moment,
                    "mean_standard_error": self.mean_standard_error,
                    "second_moment_standard_error": (
                        self.second_moment_standard_error
                    ),
                    "covariance_factor": self.covariance_factor,
                    "sample_count": self.sample_count,
                    "context_axis_hash": self.context_axis_hash,
                    "destination_transport_hash": (
                        self.destination_transport_hash
                    ),
                    "sketch_seed": self.sketch_seed,
                },
                prefix="irg-local-attribution-moments-",
            )
            object.__setattr__(self, "_content_hash", cached)
        return cached

    def pack(self) -> tuple[np.ndarray, dict[str, object]]:
        rank = int(self.covariance_factor.shape[0])
        values = np.concatenate(
            (
                self.mean[None, :],
                self.second_moment[None, :],
                self.mean_standard_error[None, :],
                self.second_moment_standard_error[None, :],
                self.covariance_factor,
            ),
            axis=0,
        )
        return np.ascontiguousarray(values), {
            "schema": self.schema,
            "route_ids": self.route_ids,
            "sample_count": self.sample_count,
            "context_axis_hash": self.context_axis_hash,
            "destination_transport_hash": self.destination_transport_hash,
            "sketch_seed": self.sketch_seed,
            "sketch_rank": rank,
            "sections": {
                "mean": [0, 1],
                "second_moment": [1, 2],
                "mean_standard_error": [2, 3],
                "second_moment_standard_error": [3, 4],
                "covariance_factor": [4, 4 + rank],
            },
        }

    @classmethod
    def unpack(
        cls,
        values: np.ndarray,
        layout: Mapping[str, object],
        *,
        route_ids: Sequence[str] | None = None,
    ) -> "LocalAttributionMomentTensor":
        array = np.asarray(values, dtype=np.float64)
        if (
            array.ndim != 2
            or layout.get("schema") != "local_attribution_moment_tensor_v1"
        ):
            raise ValueError("measurement is not a local attribution moment tensor")
        sections = dict(layout["sections"])  # type: ignore[arg-type]

        def take(name: str) -> np.ndarray:
            start, stop = map(int, sections[name])  # type: ignore[arg-type]
            return array[start:stop]

        return cls(
            route_ids=resolve_packed_string_axis(
                layout,
                expected_axis=route_ids,
                axis_key="route_ids",
                name="local attribution route axis",
            ),
            mean=take("mean").reshape(-1),
            second_moment=take("second_moment").reshape(-1),
            mean_standard_error=take("mean_standard_error").reshape(-1),
            second_moment_standard_error=take(
                "second_moment_standard_error"
            ).reshape(-1),
            covariance_factor=take("covariance_factor"),
            sample_count=int(layout["sample_count"]),
            context_axis_hash=str(layout["context_axis_hash"]),
            destination_transport_hash=str(
                layout["destination_transport_hash"]
            ),
            sketch_seed=int(layout["sketch_seed"]),
        )


class StreamingLocalAttributionAccumulator:
    """Mergeable O(RK) local moments with deterministic context sketching."""

    def __init__(
        self,
        *,
        route_ids: Sequence[str],
        destination_transport_hash: str,
        sketch_rank: int = 8,
        sketch_seed: int = 0,
    ) -> None:
        self.route_ids = canonical_unique_string_axis(
            route_ids,
            name="streaming local-attribution route axis",
        )
        if (
            not self.route_ids
            or not destination_transport_hash
            or int(sketch_rank) < 0
        ):
            raise ValueError("local attribution accumulator identity is invalid")
        self.destination_transport_hash = str(destination_transport_hash)
        self.sketch_rank = int(sketch_rank)
        self.sketch_seed = int(sketch_seed)
        size = len(self.route_ids)
        self.total_weight = 0.0
        self.sum_weight_squared = 0.0
        self.sample_count = 0
        self.sum_rows = np.zeros(size, dtype=np.float64)
        self.sum_squares = np.zeros(size, dtype=np.float64)
        self.sum_fourth = np.zeros(size, dtype=np.float64)
        self.sketch_sum = np.zeros((self.sketch_rank, size), dtype=np.float64)
        self.sketch_constant = np.zeros(self.sketch_rank, dtype=np.float64)
        self._context_ids: list[str] = []

    def _signs(self, context_ids: Sequence[str]) -> np.ndarray:
        signs = np.empty(
            (self.sketch_rank, len(tuple(context_ids))), dtype=np.float64
        )
        for probe in range(self.sketch_rank):
            for column, context_id in enumerate(context_ids):
                digest = hashlib.sha256(
                    f"{self.sketch_seed}:{probe}:{context_id}".encode()
                ).digest()
                signs[probe, column] = 1.0 if digest[0] & 1 else -1.0
        if self.sketch_rank:
            signs /= np.sqrt(float(self.sketch_rank))
        return signs

    def update(
        self,
        rows: np.ndarray,
        *,
        context_ids: Sequence[str],
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        values = np.asarray(rows, dtype=np.float64)
        ids = tuple(map(str, context_ids))
        if (
            values.shape != (len(ids), len(self.route_ids))
            or not ids
            or len(set(ids)) != len(ids)
            or not np.isfinite(values).all()
            or set(ids).intersection(self._context_ids)
        ):
            raise ValueError("local attribution rows/context axis is invalid")
        raw_weights = (
            np.ones(len(ids), dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64).reshape(-1)
        )
        if (
            raw_weights.shape != (len(ids),)
            or np.any(raw_weights < 0.0)
            or not np.isfinite(raw_weights).all()
            or float(raw_weights.sum()) <= 0.0
        ):
            raise ValueError("local attribution weights are invalid")
        self.total_weight += float(raw_weights.sum())
        self.sum_weight_squared += float(np.sum(raw_weights * raw_weights))
        self.sample_count += len(ids)
        self.sum_rows += np.einsum("n,nr->r", raw_weights, values)
        squared = values * values
        self.sum_squares += np.einsum("n,nr->r", raw_weights, squared)
        self.sum_fourth += np.einsum("n,nr->r", raw_weights, squared * squared)
        if self.sketch_rank:
            signs = self._signs(ids)
            root_weights = np.sqrt(raw_weights)
            self.sketch_sum += np.einsum(
                "kn,n,nr->kr", signs, root_weights, values, optimize=True
            )
            self.sketch_constant += np.einsum(
                "kn,n->k", signs, root_weights, optimize=True
            )
        self._context_ids.extend(ids)

    def finalize(self) -> LocalAttributionMomentTensor:
        if self.sample_count <= 0 or self.total_weight <= 0.0:
            raise ValueError("cannot finalize empty local attribution moments")
        weight = self.total_weight
        normalized_w2 = self.sum_weight_squared / (weight * weight)
        denominator = max(1e-15, 1.0 - normalized_w2)
        mean = self.sum_rows / weight
        second = self.sum_squares / weight
        contextual = np.maximum(second - mean * mean, 0.0)
        mean_se = np.sqrt(
            np.maximum(
                contextual / denominator * normalized_w2,
                0.0,
            )
        )
        fourth = self.sum_fourth / weight
        second_se = np.sqrt(
            np.maximum(fourth - second * second, 0.0)
            / denominator
            * normalized_w2
        )
        factor = (
            self.sketch_sum
            - self.sketch_constant[:, None] * mean[None, :]
        ) / np.sqrt(weight)
        return LocalAttributionMomentTensor(
            route_ids=self.route_ids,
            mean=mean,
            second_moment=second,
            mean_standard_error=mean_se,
            second_moment_standard_error=second_se,
            covariance_factor=factor,
            sample_count=self.sample_count,
            context_axis_hash=content_hash(
                tuple(self._context_ids),
                prefix="local-attribution-context-axis-",
            ),
            destination_transport_hash=self.destination_transport_hash,
            sketch_seed=self.sketch_seed,
        )


def _torch_reduction_view(value: np.ndarray):
    """Zero-copy torch view of a float64 array, or ``None`` without torch."""

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a runtime dependency here
        return None
    with warnings.catch_warnings():
        # Frozen tensors are re-validated by ``dataclasses.replace``; their
        # arrays are read-only, and torch only warns about that while we
        # only ever read here.
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(value)


def _all_finite(value: np.ndarray) -> bool:
    """``np.isfinite(value).all()`` as one multi-threaded reduction pass.

    ``max |x|`` is finite iff every element is: an infinity dominates the
    maximum and NaN propagates through torch's max reductions, so the
    infinity norm answers the question without materializing the boolean
    ``isfinite`` temporary (measured 0.13 s vs 0.32 s on a 4.1 GB array).
    """

    if value.size == 0:
        return True
    tensor = _torch_reduction_view(value)
    if tensor is None:
        return bool(np.isfinite(value).all())
    import torch

    return math.isfinite(float(torch.linalg.vector_norm(tensor, float("inf"))))


def _minimum(value: np.ndarray) -> float:
    tensor = _torch_reduction_view(value)
    if tensor is None:
        return float(np.min(value))
    return float(tensor.amin())


def _minimum_route_head_diagonal(value: np.ndarray) -> float:
    """Smallest ``value[r, h, h]`` over a ``[route, head, head]`` array."""

    tensor = _torch_reduction_view(value)
    if tensor is None:
        return float(np.min(np.diagonal(value, axis1=1, axis2=2)))
    return float(tensor.diagonal(dim1=1, dim2=2).amin())


def _frozen_float64_array(
    array_like: object,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    axis_message: str,
    nonnegative_diagonal: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    """Single-pass validation of one tripartite array.

    Accept/reject decisions are exactly those of the historical multi-pass
    form (shape, then finiteness, then the diagonal or sign bound); only the
    execution changed: one contiguous float64 view, one infinity-norm pass
    for finiteness, and one strided ``amin`` for the bound.
    """

    value = np.ascontiguousarray(np.asarray(array_like, dtype=np.float64))
    if value.shape != expected_shape or not _all_finite(value):
        raise ValueError(axis_message)
    if nonnegative_diagonal and value.size and (
        _minimum_route_head_diagonal(value) < -1e-12
    ):
        raise ValueError(f"{name} has a negative diagonal")
    if nonnegative and value.size and _minimum(value) < 0.0:
        raise ValueError(axis_message)
    value.setflags(write=False)
    return value


@dataclass(frozen=True, slots=True)
class TripartitePullbackMomentTensor:
    """Nested local→global→loss pullback moments on one physical route axis."""

    head_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    mean: np.ndarray
    route_head_second_moment: np.ndarray
    mean_estimator_covariance: np.ndarray
    second_moment_standard_error: np.ndarray
    covariance_factor: np.ndarray
    sample_count: int
    context_axis_hash: str
    destination_transport_hash: str
    loss_pullback_hash: str
    sketch_seed: int
    view_activity_mean: np.ndarray | None = None
    view_activity_mean_standard_error: np.ndarray | None = None
    view_activity_second_moment: np.ndarray | None = None
    schema: str = "tripartite_pullback_moment_tensor_v2"
    _content_hash: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        heads = tuple(map(str, self.head_ids))
        routes = canonical_unique_string_axis(
            self.route_ids,
            name="tripartite pullback route axis",
        )
        h, r = len(heads), len(routes)
        if (
            h < 3
            or heads[0] != "local_destination"
            or not heads[-1].startswith("loss:")
            or len(set(heads)) != h
            or int(self.sample_count) <= 0
            or not self.context_axis_hash
            or not self.destination_transport_hash
            or not self.loss_pullback_hash
        ):
            raise ValueError("tripartite pullback identity is incomplete")
        shapes = {
            "mean": (h, r),
            "route_head_second_moment": (r, h, h),
            "mean_estimator_covariance": (r, h, h),
            "second_moment_standard_error": (r, h, h),
        }
        for name, shape in shapes.items():
            value = _frozen_float64_array(
                getattr(self, name),
                name=name,
                expected_shape=shape,
                axis_message=f"{name} has an invalid tripartite axis",
                nonnegative_diagonal=name != "mean",
            )
            object.__setattr__(self, name, value)
        factor_array = np.asarray(self.covariance_factor, dtype=np.float64)
        if factor_array.ndim != 3:
            raise ValueError(
                "tripartite covariance factor must be [rank,head,route]"
            )
        factor = _frozen_float64_array(
            factor_array,
            name="covariance_factor",
            expected_shape=(int(factor_array.shape[0]), h, r),
            axis_message="tripartite covariance factor must be [rank,head,route]",
        )
        activity_fields = (
            self.view_activity_mean,
            self.view_activity_mean_standard_error,
            self.view_activity_second_moment,
        )
        if any(value is not None for value in activity_fields):
            if any(value is None for value in activity_fields):
                raise ValueError(
                    "pullback activity requires mean, uncertainty, and second moment"
                )
            for name in (
                "view_activity_mean",
                "view_activity_mean_standard_error",
                "view_activity_second_moment",
            ):
                value = _frozen_float64_array(
                    getattr(self, name),
                    name=name,
                    expected_shape=(3, r),
                    axis_message=(
                        f"{name} must be finite and nonnegative on [local,global,loss]×route"
                    ),
                    nonnegative=True,
                )
                object.__setattr__(self, name, value)
        object.__setattr__(self, "head_ids", heads)
        object.__setattr__(self, "route_ids", routes)
        object.__setattr__(self, "covariance_factor", factor)
        object.__setattr__(self, "sample_count", int(self.sample_count))

    @property
    def content_hash(self) -> str:
        cached = self._content_hash
        if cached is None:
            cached = content_hash(
                {
                    "schema": self.schema,
                    "head_ids": self.head_ids,
                    "route_axis_hash": ordered_string_axis_hash(
                        self.route_ids,
                        name="tripartite pullback route axis",
                    ),
                    "route_count": len(self.route_ids),
                    "mean": self.mean,
                    "route_head_second_moment": (
                        self.route_head_second_moment
                    ),
                    "mean_estimator_covariance": (
                        self.mean_estimator_covariance
                    ),
                    "second_moment_standard_error": (
                        self.second_moment_standard_error
                    ),
                    "covariance_factor": self.covariance_factor,
                    "sample_count": self.sample_count,
                    "context_axis_hash": self.context_axis_hash,
                    "destination_transport_hash": (
                        self.destination_transport_hash
                    ),
                    "loss_pullback_hash": self.loss_pullback_hash,
                    "sketch_seed": self.sketch_seed,
                    "view_activity_mean": self.view_activity_mean,
                    "view_activity_mean_standard_error": (
                        self.view_activity_mean_standard_error
                    ),
                    "view_activity_second_moment": (
                        self.view_activity_second_moment
                    ),
                },
                prefix="irg-tripartite-pullback-moments-",
            )
            object.__setattr__(self, "_content_hash", cached)
        return cached

    @property
    def local_mean(self) -> np.ndarray:
        return self.mean[0]

    @property
    def loss_mean(self) -> np.ndarray:
        return self.mean[-1]

    @property
    def activity_available(self) -> bool:
        return self.view_activity_mean is not None

    @property
    def local_activity(self) -> np.ndarray:
        if self.view_activity_mean is None:
            raise RuntimeError("legacy tripartite moments have no activity mass")
        return self.view_activity_mean[0]

    @property
    def global_activity(self) -> np.ndarray:
        if self.view_activity_mean is None:
            raise RuntimeError("legacy tripartite moments have no activity mass")
        return self.view_activity_mean[1]

    @property
    def loss_activity(self) -> np.ndarray:
        if self.view_activity_mean is None:
            raise RuntimeError("legacy tripartite moments have no activity mass")
        return self.view_activity_mean[2]

    @property
    def activity_rms(self) -> np.ndarray:
        if self.view_activity_second_moment is None:
            raise RuntimeError("legacy tripartite moments have no activity mass")
        return np.sqrt(np.maximum(self.view_activity_second_moment, 0.0))

    @property
    def effective_context_participation(self) -> np.ndarray:
        if (
            self.view_activity_mean is None
            or self.view_activity_second_moment is None
        ):
            raise RuntimeError("legacy tripartite moments have no activity mass")
        return np.divide(
            self.view_activity_mean * self.view_activity_mean,
            self.view_activity_second_moment,
            out=np.zeros_like(self.view_activity_mean),
            where=self.view_activity_second_moment > 0.0,
        )

    def action_head_covariance(
        self,
        action: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Approximate the full cross-head covariance for one route action.

        The shared context sketch is contracted only after the route action is
        declared.  This preserves local/global/loss cross-covariance and every
        sketched cross-route term without materializing a route-by-route
        matrix.
        """

        vector = np.asarray(action, dtype=np.float64).reshape(-1)
        if (
            vector.shape != (len(self.route_ids),)
            or not np.isfinite(vector).all()
        ):
            raise ValueError("tripartite action must be finite on the route axis")
        projected = np.einsum(
            "khr,r->kh",
            self.covariance_factor,
            vector,
            optimize=True,
        )
        covariance = projected.T @ projected
        return np.ascontiguousarray(0.5 * (covariance + covariance.T))

    def action_energy(
        self,
        action: Sequence[float] | np.ndarray,
        *,
        head_metric: np.ndarray | None = None,
    ) -> float:
        """Return nested pullback covariance energy for one route action."""

        covariance = self.action_head_covariance(action)
        if head_metric is None:
            metric = np.eye(len(self.head_ids), dtype=np.float64)
        else:
            metric = np.asarray(head_metric, dtype=np.float64)
            if (
                metric.shape != covariance.shape
                or not np.isfinite(metric).all()
                or not np.allclose(metric, metric.T, rtol=1e-10, atol=1e-12)
            ):
                raise ValueError(
                    "tripartite head metric must be finite and symmetric"
                )
            eigenvalues = np.linalg.eigvalsh(metric)
            if float(eigenvalues.min(initial=0.0)) < -1e-10:
                raise ValueError("tripartite head metric must be positive semidefinite")
        return max(float(np.sum(metric * covariance)), 0.0)

    def pack(self) -> tuple[np.ndarray, dict[str, object]]:
        h, r = self.mean.shape
        rank = int(self.covariance_factor.shape[0])
        second = self.route_head_second_moment.transpose(1, 2, 0).reshape(h * h, r)
        estimator = self.mean_estimator_covariance.transpose(1, 2, 0).reshape(h * h, r)
        second_se = self.second_moment_standard_error.transpose(1, 2, 0).reshape(h * h, r)
        factor = self.covariance_factor.reshape(rank * h, r)
        sections_and_values: list[tuple[str, np.ndarray]] = [
            ("mean", self.mean),
            ("route_head_second_moment", second),
            ("mean_estimator_covariance", estimator),
            ("second_moment_standard_error", second_se),
            ("covariance_factor", factor),
        ]
        if self.activity_available:
            assert self.view_activity_mean is not None
            assert self.view_activity_mean_standard_error is not None
            assert self.view_activity_second_moment is not None
            sections_and_values.extend(
                (
                    ("view_activity_mean", self.view_activity_mean),
                    (
                        "view_activity_mean_standard_error",
                        self.view_activity_mean_standard_error,
                    ),
                    (
                        "view_activity_second_moment",
                        self.view_activity_second_moment,
                    ),
                )
            )
        values = np.concatenate(
            tuple(value for _name, value in sections_and_values), axis=0
        )
        offsets: dict[str, list[int]] = {}
        cursor = 0
        for name, value in sections_and_values:
            stop = cursor + int(value.shape[0])
            offsets[name] = [cursor, stop]
            cursor = stop
        return np.ascontiguousarray(values), {
            "schema": self.schema,
            "head_ids": self.head_ids,
            "route_ids": self.route_ids,
            "head_count": h,
            "sample_count": self.sample_count,
            "context_axis_hash": self.context_axis_hash,
            "destination_transport_hash": self.destination_transport_hash,
            "loss_pullback_hash": self.loss_pullback_hash,
            "sketch_seed": self.sketch_seed,
            "sketch_rank": rank,
            "activity_view_ids": ("local", "global", "loss"),
            "activity_available": self.activity_available,
            "sections": offsets,
        }

    @classmethod
    def unpack(
        cls,
        values: np.ndarray,
        layout: Mapping[str, object],
        *,
        route_ids: Sequence[str] | None = None,
    ) -> "TripartitePullbackMomentTensor":
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or layout.get("schema") not in {
            "tripartite_pullback_moment_tensor_v1",
            "tripartite_pullback_moment_tensor_v2",
        }:
            raise ValueError("measurement is not a tripartite pullback tensor")
        h = int(layout["head_count"])
        routes = resolve_packed_string_axis(
            layout,
            expected_axis=route_ids,
            axis_key="route_ids",
            name="tripartite pullback route axis",
        )
        rank = int(layout["sketch_rank"])
        sections = dict(layout["sections"])  # type: ignore[arg-type]

        def take(name: str) -> np.ndarray:
            start, stop = map(int, sections[name])  # type: ignore[arg-type]
            return array[start:stop]

        r = len(routes)
        return cls(
            head_ids=tuple(layout["head_ids"]),  # type: ignore[arg-type]
            route_ids=routes,
            mean=take("mean"),
            route_head_second_moment=take("route_head_second_moment")
            .reshape(h, h, r)
            .transpose(2, 0, 1),
            mean_estimator_covariance=take("mean_estimator_covariance")
            .reshape(h, h, r)
            .transpose(2, 0, 1),
            second_moment_standard_error=take("second_moment_standard_error")
            .reshape(h, h, r)
            .transpose(2, 0, 1),
            covariance_factor=take("covariance_factor").reshape(rank, h, r),
            sample_count=int(layout["sample_count"]),
            context_axis_hash=str(layout["context_axis_hash"]),
            destination_transport_hash=str(layout["destination_transport_hash"]),
            loss_pullback_hash=str(layout["loss_pullback_hash"]),
            sketch_seed=int(layout["sketch_seed"]),
            view_activity_mean=(
                take("view_activity_mean")
                if "view_activity_mean" in sections
                else None
            ),
            view_activity_mean_standard_error=(
                take("view_activity_mean_standard_error")
                if "view_activity_mean_standard_error" in sections
                else None
            ),
            view_activity_second_moment=(
                take("view_activity_second_moment")
                if "view_activity_second_moment" in sections
                else None
            ),
            schema=str(layout["schema"]),
        )


class StreamingTripartitePullbackAccumulator:
    """O(R(H²+KH)) streaming moments for nested pullback heads."""

    def __init__(
        self,
        *,
        head_ids: Sequence[str],
        route_ids: Sequence[str],
        destination_transport_hash: str,
        loss_pullback_hash: str,
        sketch_rank: int = 8,
        sketch_seed: int = 0,
    ) -> None:
        self.head_ids = tuple(map(str, head_ids))
        self.route_ids = canonical_unique_string_axis(
            route_ids,
            name="streaming tripartite route axis",
        )
        self.destination_transport_hash = str(destination_transport_hash)
        self.loss_pullback_hash = str(loss_pullback_hash)
        self.sketch_rank = int(sketch_rank)
        self.sketch_seed = int(sketch_seed)
        h, r = len(self.head_ids), len(self.route_ids)
        if h < 3 or r <= 0 or self.sketch_rank < 0:
            raise ValueError("tripartite accumulator axes are invalid")
        self.total_weight = 0.0
        self.sum_weight_squared = 0.0
        self.sample_count = 0
        self.sum_rows = np.zeros((h, r), dtype=np.float64)
        self.sum_outer = np.zeros((r, h, h), dtype=np.float64)
        self.sum_outer_squared = np.zeros((r, h, h), dtype=np.float64)
        self.sum_view_activity = np.zeros((3, r), dtype=np.float64)
        self.sum_view_activity_squared = np.zeros((3, r), dtype=np.float64)
        self.sketch_sum = np.zeros((self.sketch_rank, h, r), dtype=np.float64)
        self.sketch_constant = np.zeros(self.sketch_rank, dtype=np.float64)
        self._context_ids: list[str] = []

    def _signs(self, context_ids: Sequence[str]) -> np.ndarray:
        signs = np.empty((self.sketch_rank, len(context_ids)), dtype=np.float64)
        for probe in range(self.sketch_rank):
            for column, context_id in enumerate(context_ids):
                digest = hashlib.sha256(
                    f"{self.sketch_seed}:tripartite:{probe}:{context_id}".encode()
                ).digest()
                signs[probe, column] = 1.0 if digest[0] & 1 else -1.0
        if self.sketch_rank:
            signs /= np.sqrt(float(self.sketch_rank))
        return signs

    def update(
        self,
        rows: np.ndarray,
        *,
        context_ids: Sequence[str],
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        values = np.asarray(rows, dtype=np.float64)
        ids = tuple(map(str, context_ids))
        if (
            values.shape
            != (len(ids), len(self.head_ids), len(self.route_ids))
            or not np.isfinite(values).all()
            or not ids
            or set(ids).intersection(self._context_ids)
        ):
            raise ValueError("tripartite pullback rows/context axis is invalid")
        raw_weights = (
            np.ones(len(ids), dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64).reshape(-1)
        )
        if (
            raw_weights.shape != (len(ids),)
            or np.any(raw_weights < 0.0)
            or not np.isfinite(raw_weights).all()
            or float(raw_weights.sum()) <= 0.0
        ):
            raise ValueError("tripartite pullback weights are invalid")
        self.total_weight += float(raw_weights.sum())
        self.sum_weight_squared += float(np.sum(raw_weights * raw_weights))
        self.sample_count += len(ids)
        self.sum_rows += np.einsum("n,nhr->hr", raw_weights, values)
        self.sum_outer += np.einsum(
            "n,nhr,njr->rhj", raw_weights, values, values, optimize=True
        )
        squared = values * values
        view_activity = np.stack(
            (
                np.abs(values[:, 0, :]),
                np.sqrt(np.sum(values[:, 1:-1, :] ** 2, axis=1)),
                np.abs(values[:, -1, :]),
            ),
            axis=1,
        )
        self.sum_view_activity += np.einsum(
            "n,nvr->vr", raw_weights, view_activity, optimize=True
        )
        self.sum_view_activity_squared += np.einsum(
            "n,nvr->vr",
            raw_weights,
            view_activity * view_activity,
            optimize=True,
        )
        self.sum_outer_squared += np.einsum(
            "n,nhr,njr->rhj", raw_weights, squared, squared, optimize=True
        )
        if self.sketch_rank:
            signs = self._signs(ids)
            root = np.sqrt(raw_weights)
            self.sketch_sum += np.einsum(
                "kn,n,nhr->khr", signs, root, values, optimize=True
            )
            self.sketch_constant += np.einsum("kn,n->k", signs, root)
        self._context_ids.extend(ids)

    def finalize(self) -> TripartitePullbackMomentTensor:
        if self.sample_count <= 0:
            raise ValueError("cannot finalize an empty tripartite accumulator")
        weight = self.total_weight
        w2 = self.sum_weight_squared / (weight * weight)
        denominator = max(1e-15, 1.0 - w2)
        mean = self.sum_rows / weight
        second = self.sum_outer / weight
        contextual = second - np.einsum("hr,jr->rhj", mean, mean)
        estimator = 0.5 * (
            contextual + contextual.transpose(0, 2, 1)
        ) / denominator * w2
        outer_second = self.sum_outer_squared / weight
        second_se = np.sqrt(
            np.maximum(outer_second - second * second, 0.0)
            / denominator
            * w2
        )
        factor = (
            self.sketch_sum
            - self.sketch_constant[:, None, None] * mean[None, :, :]
        ) / np.sqrt(weight)
        activity_mean = self.sum_view_activity / weight
        activity_second = self.sum_view_activity_squared / weight
        activity_se = np.sqrt(
            np.maximum(activity_second - activity_mean * activity_mean, 0.0)
            / denominator
            * w2
        )
        return TripartitePullbackMomentTensor(
            head_ids=self.head_ids,
            route_ids=self.route_ids,
            mean=mean,
            route_head_second_moment=second,
            mean_estimator_covariance=estimator,
            second_moment_standard_error=second_se,
            covariance_factor=factor,
            sample_count=self.sample_count,
            context_axis_hash=content_hash(
                tuple(self._context_ids),
                prefix="tripartite-pullback-context-axis-",
            ),
            destination_transport_hash=self.destination_transport_hash,
            loss_pullback_hash=self.loss_pullback_hash,
            sketch_seed=self.sketch_seed,
            view_activity_mean=activity_mean,
            view_activity_mean_standard_error=activity_se,
            view_activity_second_moment=activity_second,
        )


class StreamingResponseMomentAccumulator:
    """Mergeable O(R d^2) sufficient-statistic accumulator.

    The accumulator distinguishes three objects that are often conflated:

    * contextual response covariance across the declared context population;
    * covariance of the estimated response mean;
    * entrywise uncertainty of the estimated second moment.

    It also retains the uncentered target/competitor cross moment.  That cross
    moment equals a covariance only after subtracting the product of means.
    """

    def __init__(self, *, channel_ids: Sequence[str], route_ids: Sequence[str]) -> None:
        self.channel_ids = tuple(map(str, channel_ids))
        self.route_ids = canonical_unique_string_axis(
            route_ids,
            name="streaming response-moment route axis",
        )
        if (
            not self.channel_ids
            or not self.route_ids
            or len(set(self.channel_ids)) != len(self.channel_ids)
        ):
            raise ValueError("streaming moment accumulator requires unique axes")
        d, r = len(self.channel_ids), len(self.route_ids)
        self.total_weight = 0.0
        self.sum_weight_squared = 0.0
        self.sample_count = 0
        self.sum_rows = np.zeros((d, r), dtype=np.float64)
        self.sum_outer = np.zeros((r, d, d), dtype=np.float64)
        self.sum_outer_squared = np.zeros((r, d, d), dtype=np.float64)
        self._context_ids: list[str] = []

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(self._context_ids)

    def update(
        self,
        rows: np.ndarray,
        *,
        context_ids: Sequence[str],
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        values = np.asarray(rows, dtype=np.float64)
        ids = tuple(map(str, context_ids))
        if values.ndim != 3 or values.shape[1:] != (
            len(self.channel_ids),
            len(self.route_ids),
        ):
            raise ValueError(
                "streaming moment rows use incompatible response or route axes"
            )
        if values.shape[0] != len(ids) or not ids or len(set(ids)) != len(ids):
            raise ValueError(
                "streaming moment context IDs must uniquely align with rows"
            )
        if any(value in set(self._context_ids) for value in ids):
            raise ValueError(
                "streaming moment accumulator received a duplicate context"
            )
        if not np.isfinite(values).all():
            raise ValueError("streaming moment rows must be finite")
        raw_weights = (
            np.ones(values.shape[0], dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64).reshape(-1)
        )
        if (
            raw_weights.shape != (values.shape[0],)
            or np.any(raw_weights < 0.0)
            or not np.isfinite(raw_weights).all()
            or float(np.sum(raw_weights)) <= 0.0
        ):
            raise ValueError("streaming moment weights are invalid")
        self.total_weight += float(np.sum(raw_weights))
        self.sum_weight_squared += float(np.sum(raw_weights * raw_weights))
        self.sample_count += int(values.shape[0])
        self.sum_rows += np.einsum("n,ndr->dr", raw_weights, values, optimize=True)
        self.sum_outer += np.einsum(
            "n,ndr,ner->rde", raw_weights, values, values, optimize=True
        )
        squared = values * values
        self.sum_outer_squared += np.einsum(
            "n,ndr,ner->rde", raw_weights, squared, squared, optimize=True
        )
        self._context_ids.extend(ids)

    def merge(self, other: "StreamingResponseMomentAccumulator") -> None:
        if self.channel_ids != other.channel_ids or self.route_ids != other.route_ids:
            raise ValueError("cannot merge streaming accumulators with different axes")
        if set(self._context_ids).intersection(other._context_ids):
            raise ValueError(
                "cannot merge streaming accumulators with overlapping contexts"
            )
        self.total_weight += other.total_weight
        self.sum_weight_squared += other.sum_weight_squared
        self.sample_count += other.sample_count
        self.sum_rows += other.sum_rows
        self.sum_outer += other.sum_outer
        self.sum_outer_squared += other.sum_outer_squared
        self._context_ids.extend(other._context_ids)

    def finalize(
        self,
        *,
        response_spec_hash: str,
        metric_id: str,
        context_axis_hash: str | None = None,
        metadata: Mapping[str, object] | None = None,
        maximum_dense_response_dimension: int = 32,
        route_chunk_size: int = 8192,
    ) -> ResponseMomentTensor:
        if self.sample_count <= 0 or self.total_weight <= 0.0:
            raise ValueError("cannot finalize an empty response-moment accumulator")
        weight = self.total_weight
        normalized_weight_square_sum = self.sum_weight_squared / (weight * weight)
        denominator = max(1e-15, 1.0 - normalized_weight_square_sum)
        mean = self.sum_rows / weight
        second = self.sum_outer / weight
        contextual = second - np.einsum("dr,er->rde", mean, mean, optimize=True)
        # Roundoff can make a mathematically PSD diagonal block slightly
        # indefinite.  ResponseMomentTensor performs the final guarded PSD
        # projection; clip only scalar entrywise variances used below.
        estimator_covariance = contextual / denominator * normalized_weight_square_sum
        estimator_covariance = 0.5 * (
            estimator_covariance + estimator_covariance.transpose(0, 2, 1)
        )
        mean_se = np.sqrt(
            np.maximum(np.diagonal(estimator_covariance, axis1=1, axis2=2).T, 0.0)
        )
        outer_second = self.sum_outer_squared / weight
        outer_variance = np.maximum(outer_second - second * second, 0.0) / denominator
        second_se = np.sqrt(outer_variance * normalized_weight_square_sum)
        return ResponseMomentTensor(
            channel_ids=self.channel_ids,
            route_ids=self.route_ids,
            mean=mean,
            second_moment_diagonal=second,
            mean_standard_error=mean_se,
            second_moment_standard_error=second_se,
            sample_count=self.sample_count,
            response_spec_hash=str(response_spec_hash),
            metric_id=str(metric_id),
            context_axis_hash=str(
                context_axis_hash
                or content_hash(self.context_ids, prefix="response-context-axis-")
            ),
            mean_estimator_covariance_diagonal=estimator_covariance,
            metadata={
                "streaming_sufficient_statistics": True,
                "contextual_variability_separate_from_estimation_uncertainty": True,
                "uncentered_cross_moment_is_not_automatically_covariance": True,
                "raw_weight_total": self.total_weight,
                "normalized_weight_square_sum": normalized_weight_square_sum,
                "maximum_dense_response_dimension": int(
                    maximum_dense_response_dimension
                ),
                "route_chunk_size": int(route_chunk_size),
                **dict(metadata or {}),
            },
        )


BatchCallback = Callable[
    [tuple[str, ...], Sequence[int], int],
    Iterable[ResponseJacobianBatch],
]


class StreamingResponseGeometryProvider:
    """Callback provider for native-scale response moments and screened factors.

    The callback must yield the requested contexts exactly once and in request
    order.  It receives ``(context_ids, route_indices, batch_size)`` and returns
    semantic response derivatives.  For v3 it may also return nuisance tangent
    rows.  The provider executes a two-pass streamed quotient for full-route v3
    moments and writes screened v2--v4 factors to a memory-mapped artifact when
    ``artifact_root`` is provided.
    """

    supported_kinds = frozenset(
        {
            "response_moments",
            "response_factor",
            "response_jacobian",
            "covariance_rank_expansion",
        }
    )

    def __init__(
        self,
        *,
        context_ids: Sequence[str],
        channel_ids: Sequence[str],
        route_ids: Sequence[str],
        batch_callback: BatchCallback,
        response_spec_hash: str,
        metric_id: str,
        calibration_state_id: str,
        batch_size: int = 32,
        artifact_root: str | Path | None = None,
        maximum_factor_elements_in_memory: int = 10_000_000,
        projection_residual_bound: float = 0.0,
        sketch_residual_spectral_bound: float = 0.0,
        omitted_route_residual_bound: float = 0.0,
        local_covariance_sketch_rank: int = 8,
        local_covariance_sketch_seed: int = 0,
        realized_cost: RealizedCost | None = None,
    ) -> None:
        self.context_ids = tuple(map(str, context_ids))
        self.channel_ids = tuple(map(str, channel_ids))
        self.route_ids = canonical_unique_string_axis(
            route_ids,
            name="streaming response-provider route axis",
        )
        if (
            not self.context_ids
            or not self.channel_ids
            or not self.route_ids
            or len(set(self.context_ids)) != len(self.context_ids)
            or len(set(self.channel_ids)) != len(self.channel_ids)
        ):
            raise ValueError(
                "streaming response provider requires unique nonempty axes"
            )
        if batch_size <= 0 or maximum_factor_elements_in_memory <= 0:
            raise ValueError(
                "streaming response provider resource controls must be positive"
            )
        bound = float(projection_residual_bound)
        sketch_bound = float(sketch_residual_spectral_bound)
        omitted_bound = float(omitted_route_residual_bound)
        if (
            not np.isfinite(bound)
            or bound < 0.0
            or not np.isfinite(sketch_bound)
            or sketch_bound < 0.0
            or not np.isfinite(omitted_bound)
            or omitted_bound < 0.0
        ):
            raise ValueError(
                "response projection/sketch/omitted-route residual bounds must "
                "be finite and nonnegative"
            )
        self.batch_callback = batch_callback
        self.response_spec_hash = str(response_spec_hash)
        self.metric_id = str(metric_id)
        self.calibration_state_id = str(calibration_state_id)
        self.batch_size = int(batch_size)
        self.artifact_root = None if artifact_root is None else Path(artifact_root)
        if self.artifact_root is not None:
            self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.maximum_factor_elements_in_memory = int(maximum_factor_elements_in_memory)
        self.projection_residual_bound = bound
        self.sketch_residual_spectral_bound = sketch_bound
        self.omitted_route_residual_bound = omitted_bound
        if int(local_covariance_sketch_rank) < 0:
            raise ValueError("local covariance sketch rank must be nonnegative")
        self.local_covariance_sketch_rank = int(local_covariance_sketch_rank)
        self.local_covariance_sketch_seed = int(local_covariance_sketch_seed)
        self.realized_cost = realized_cost or RealizedCost(backward_passes=1)

    def _selected_contexts(self, request: MeasurementRequest) -> tuple[str, ...]:
        selector = request.context_selector
        if selector.context_ids:
            selected = tuple(selector.context_ids)
            universe = set(self.context_ids)
            if any(item not in universe for item in selected):
                raise ValueError(
                    "streaming response request contains an unknown context"
                )
            return selected
        excluded = set(map(str, selector.parameters.get("exclude_context_ids", ())))
        available = np.asarray(
            [
                index
                for index, item in enumerate(self.context_ids)
                if item not in excluded
            ],
            dtype=np.int64,
        )
        count = min(int(selector.requested_count or len(available)), len(available))
        if count <= 0:
            raise ValueError("no streaming response contexts remain")
        generator = np.random.default_rng(int(selector.random_seed))
        chosen = np.sort(generator.choice(available, size=count, replace=False))
        return tuple(self.context_ids[int(index)] for index in chosen)

    def _routes(
        self, request: MeasurementRequest
    ) -> tuple[Sequence[int], Sequence[str]]:
        raw = request.payload.get("route_indices")
        if raw is None:
            # This is the declared whole-axis request.  Its ordinal index axis
            # is constructive and its string axis was validated at provider
            # construction, so neither needs another exhaustive ``set`` or a
            # second full-width tuple of string references.
            indices = (
                tuple(range(len(self.route_ids)))
                if len(self.route_ids) <= 4096
                else range(len(self.route_ids))
            )
            identifiers = self.route_ids
        else:
            indices = tuple(int(value) for value in raw)
            if (
                not indices
                or len(set(indices)) != len(indices)
                or min(indices) < 0
                or max(indices) >= len(self.route_ids)
            ):
                raise ValueError("streaming response route_indices are invalid")
            identifiers = tuple(self.route_ids[index] for index in indices)
        declared = tuple(map(str, request.payload.get("route_ids", ())))
        if declared and (
            len(declared) != len(identifiers)
            or any(left != right for left, right in zip(declared, identifiers, strict=True))
        ):
            raise ValueError(
                "streaming response route IDs disagree with the physical route axis"
            )
        return indices, identifiers

    def _batches(
        self,
        selected: tuple[str, ...],
        route_indices: Sequence[int],
    ) -> Iterable[ResponseJacobianBatch]:
        expected_offset = 0
        for batch in self.batch_callback(selected, route_indices, self.batch_size):
            expected = selected[
                expected_offset : expected_offset + len(batch.context_ids)
            ]
            if batch.context_ids != expected:
                raise ValueError("streaming callback changed context order or coverage")
            if batch.route_rows.shape[1:] != (
                len(self.channel_ids),
                len(route_indices),
            ):
                raise ValueError(
                    "streaming callback returned an incompatible route factor"
                )
            if (
                batch.local_route_rows is not None
                and batch.local_route_rows.shape
                != (len(batch.context_ids), len(route_indices))
            ):
                raise ValueError(
                    "streaming callback returned an incompatible local route factor"
                )
            expected_offset += len(batch.context_ids)
            yield batch
        if expected_offset != len(selected):
            raise ValueError(
                "streaming callback did not return every selected context exactly once"
            )

    def _moments(
        self,
        selected: tuple[str, ...],
        route_indices: Sequence[int],
        route_ids: Sequence[str],
        *,
        intrinsic: bool,
        regularization: float,
        rcond: float,
        maximum_condition_number: float,
        identification_policy: IdentificationPolicy,
        require_task_metric_factor: bool,
        global_trust_spec: GlobalTrustGeometrySpec | None,
        global_trust_requested_rank: int | None,
        atlas_split_fold_count: int,
        measured_mask_hash: str,
        source_measurement_hash: str,
    ) -> tuple[
        ResponseMomentTensor,
        ResponseMomentTensor | None,
        LocalAttributionMomentTensor | None,
        TripartitePullbackMomentTensor | None,
        GlobalTrustGeometryState | None,
        GlobalTrustGeometryState | None,
        LocalRegimeRouteSummary | None,
        dict[str, object],
    ]:
        fixed_accumulator = StreamingResponseMomentAccumulator(
            channel_ids=self.channel_ids,
            route_ids=route_ids,
        )
        nuisance_gram = None
        nuisance_cross = None
        nuisance_dimension = None
        metric_factor_dimension = None
        fit_coordinate_mode = "euclidean_semantic_response"
        local_accumulator: StreamingLocalAttributionAccumulator | None = None
        local_transport_hash: str | None = None
        tripartite_accumulator: StreamingTripartitePullbackAccumulator | None = None
        tripartite_fold_accumulators: tuple[
            StreamingTripartitePullbackAccumulator, ...
        ] = ()
        trust_rank = (
            None
            if global_trust_spec is None
            else (
                global_trust_spec.initial_rank
                if global_trust_requested_rank is None
                else int(global_trust_requested_rank)
            )
        )
        if global_trust_spec is not None and (
            trust_rank is None
            or trust_rank < global_trust_spec.initial_rank
            or trust_rank > global_trust_spec.maximum_rank
        ):
            raise ValueError(
                "requested global-trust rank is outside the frozen rank schedule"
            )
        trust_accumulator = (
            None
            if global_trust_spec is None
            else FrequentDirectionsTrustAccumulator(
                route_ids=route_ids,
                sketch_rank=int(trust_rank),
            )
        )
        split_fold_count = int(atlas_split_fold_count)
        if split_fold_count not in {0, 1} and len(selected) < split_fold_count:
            raise ValueError(
                "functional-atlas split-fold count exceeds measured contexts"
            )
        fold_by_context = {
            context_id: index % split_fold_count
            for index, context_id in enumerate(selected)
        } if split_fold_count >= 2 else {}
        trust_fold_accumulators = (
            tuple(
                FrequentDirectionsTrustAccumulator(
                    route_ids=route_ids,
                    sketch_rank=int(trust_rank),
                )
                for _ in range(split_fold_count)
            )
            if trust_accumulator is not None and split_fold_count >= 2
            else ()
        )
        loss_pullback_hash: str | None = None
        local_regime_summaries: list[
            LocalRegimeRouteSummary | SmoothTransmissionRouteSummary
        ] = []
        streamed_batch_count = 0
        for batch in self._batches(selected, route_indices):
            streamed_batch_count += 1
            fixed_accumulator.update(
                batch.route_rows,
                context_ids=batch.context_ids,
                weights=batch.weights,
            )
            if trust_accumulator is not None:
                if batch.metric_route_rows is not None:
                    trust_rows = batch.metric_route_rows
                elif batch.metric_whitener is not None:
                    trust_rows = np.einsum(
                        "nkd,ndr->nkr",
                        batch.metric_whitener,
                        batch.route_rows,
                        optimize=True,
                    )
                else:
                    trust_rows = batch.route_rows
                trust_accumulator.update(
                    trust_rows,
                    context_ids=batch.context_ids,
                    weights=batch.weights,
                )
                for fold_index, fold_accumulator in enumerate(
                    trust_fold_accumulators
                ):
                    chosen = np.asarray(
                        [
                            fold_by_context[context_id] == fold_index
                            for context_id in batch.context_ids
                        ],
                        dtype=bool,
                    )
                    if not np.any(chosen):
                        continue
                    fold_accumulator.update(
                        trust_rows[chosen],
                        context_ids=tuple(
                            context_id
                            for context_id, retain in zip(
                                batch.context_ids, chosen, strict=True
                            )
                            if retain
                        ),
                        weights=(
                            None
                            if batch.weights is None
                            else np.asarray(batch.weights, dtype=np.float64)[chosen]
                        ),
                    )
            if batch.local_route_rows is not None:
                batch_transport_hash = str(
                    batch.local_destination_transport_hash or ""
                )
                if local_accumulator is None:
                    local_transport_hash = batch_transport_hash
                    local_accumulator = StreamingLocalAttributionAccumulator(
                        route_ids=route_ids,
                        destination_transport_hash=batch_transport_hash,
                        # When the nested loss head is present, the shared
                        # tripartite factor below already contains the local
                        # covariance head.  Avoid retaining it twice.
                        sketch_rank=(
                            0
                            if batch.loss_route_rows is not None
                            else self.local_covariance_sketch_rank
                        ),
                        sketch_seed=self.local_covariance_sketch_seed,
                    )
                elif batch_transport_hash != local_transport_hash:
                    raise ValueError(
                        "local destination transport changed across streamed batches"
                    )
                local_accumulator.update(
                    batch.local_route_rows,
                    context_ids=batch.context_ids,
                    weights=batch.weights,
                )
            elif local_accumulator is not None:
                raise ValueError(
                    "local attribution rows disappeared within a streamed request"
                )
            if batch.loss_route_rows is not None:
                if batch.local_route_rows is None:
                    raise ValueError("loss rows require local pullback rows")
                batch_loss_hash = str(batch.loss_pullback_hash or "")
                if tripartite_accumulator is None:
                    loss_pullback_hash = batch_loss_hash
                    tripartite_accumulator = (
                        StreamingTripartitePullbackAccumulator(
                            head_ids=(
                                "local_destination",
                                *(
                                    f"global:{channel_id}"
                                    for channel_id in self.channel_ids
                                ),
                                "loss:bce",
                            ),
                            route_ids=route_ids,
                            destination_transport_hash=str(
                                batch.local_destination_transport_hash
                            ),
                            loss_pullback_hash=batch_loss_hash,
                            sketch_rank=self.local_covariance_sketch_rank,
                            sketch_seed=self.local_covariance_sketch_seed,
                        )
                    )
                    if split_fold_count >= 2:
                        tripartite_fold_accumulators = tuple(
                            StreamingTripartitePullbackAccumulator(
                                head_ids=tripartite_accumulator.head_ids,
                                route_ids=route_ids,
                                destination_transport_hash=str(
                                    batch.local_destination_transport_hash
                                ),
                                loss_pullback_hash=batch_loss_hash,
                                sketch_rank=self.local_covariance_sketch_rank,
                                sketch_seed=self.local_covariance_sketch_seed,
                            )
                            for _ in range(split_fold_count)
                        )
                elif batch_loss_hash != loss_pullback_hash:
                    raise ValueError(
                        "loss pullback identity changed across streamed batches"
                    )
                tripartite_accumulator.update(
                    tripartite_rows := np.concatenate(
                        (
                            batch.local_route_rows[:, None, :],
                            batch.route_rows,
                            batch.loss_route_rows[:, None, :],
                        ),
                        axis=1,
                    ),
                    context_ids=batch.context_ids,
                    weights=batch.weights,
                )
                for fold_index, fold_accumulator in enumerate(
                    tripartite_fold_accumulators
                ):
                    chosen = np.asarray(
                        [
                            fold_by_context[context_id] == fold_index
                            for context_id in batch.context_ids
                        ],
                        dtype=bool,
                    )
                    if not np.any(chosen):
                        continue
                    fold_accumulator.update(
                        tripartite_rows[chosen],
                        context_ids=tuple(
                            context_id
                            for context_id, retain in zip(
                                batch.context_ids, chosen, strict=True
                            )
                            if retain
                        ),
                        weights=(
                            None
                            if batch.weights is None
                            else np.asarray(batch.weights, dtype=np.float64)[chosen]
                        ),
                    )
            elif tripartite_accumulator is not None:
                raise ValueError(
                    "loss pullback rows disappeared within a streamed request"
                )
            if batch.local_regime_summary is not None:
                if batch.local_regime_summary.route_ids != route_ids:
                    raise ValueError(
                        "local regime summary uses a different physical route axis"
                    )
                local_regime_summaries.append(batch.local_regime_summary)
            elif local_regime_summaries:
                raise ValueError(
                    "local regime diagnostics disappeared within a streamed request"
                )
            if intrinsic:
                if batch.nuisance_rows is None:
                    raise ValueError(
                        "intrinsic streamed moments require nuisance response rows"
                    )
                nuisance_dimension = batch.nuisance_rows.shape[2]
                if nuisance_gram is None:
                    nuisance_gram = np.zeros(
                        (nuisance_dimension, nuisance_dimension), dtype=np.float64
                    )
                    nuisance_cross = np.zeros(
                        (nuisance_dimension, len(route_ids)), dtype=np.float64
                    )
                if batch.nuisance_rows.shape[2] != nuisance_dimension:
                    raise ValueError(
                        "nuisance dimension changed across streamed batches"
                    )
                raw_weights = (
                    np.ones(len(batch.context_ids), dtype=np.float64)
                    if batch.weights is None
                    else np.asarray(batch.weights, dtype=np.float64)
                )
                if batch.metric_route_rows is not None:
                    assert batch.metric_nuisance_rows is not None
                    whitened_route = batch.metric_route_rows
                    whitened_nuisance = batch.metric_nuisance_rows
                    batch_metric_dimension = batch.metric_route_rows.shape[1]
                    fit_coordinate_mode = "streamed_prewhitened_task_metric_factor"
                elif batch.metric_whitener is None:
                    if require_task_metric_factor:
                        raise ValueError(
                            "intrinsic streamed quotient requires the declared task-metric whitener"
                        )
                    whitened_nuisance = batch.nuisance_rows
                    whitened_route = batch.route_rows
                    batch_metric_dimension = batch.route_rows.shape[1]
                else:
                    whitened_nuisance = np.einsum(
                        "nkd,ndp->nkp",
                        batch.metric_whitener,
                        batch.nuisance_rows,
                        optimize=True,
                    )
                    whitened_route = np.einsum(
                        "nkd,ndr->nkr",
                        batch.metric_whitener,
                        batch.route_rows,
                        optimize=True,
                    )
                    batch_metric_dimension = batch.metric_whitener.shape[1]
                    fit_coordinate_mode = "streamed_task_metric_whitener"
                if metric_factor_dimension is None:
                    metric_factor_dimension = batch_metric_dimension
                elif metric_factor_dimension != batch_metric_dimension:
                    raise ValueError(
                        "task-metric factor dimension changed across streamed batches"
                    )
                nuisance_gram += np.einsum(
                    "n,nkp,nkq->pq",
                    raw_weights,
                    whitened_nuisance,
                    whitened_nuisance,
                    optimize=True,
                )
                nuisance_cross += np.einsum(
                    "n,nkp,nkr->pr",
                    raw_weights,
                    whitened_nuisance,
                    whitened_route,
                    optimize=True,
                )
        common_metadata = {
            "streamed_native_scale_provider": True,
            "projection_residual_bound": self.projection_residual_bound,
            "sketch_residual_spectral_bound": self.sketch_residual_spectral_bound,
            "omitted_route_residual_bound": self.omitted_route_residual_bound,
            "nuisance_quotiented": False,
        }
        fixed = fixed_accumulator.finalize(
            response_spec_hash=self.response_spec_hash,
            metric_id=self.metric_id,
            context_axis_hash=content_hash(selected, prefix="response-context-axis-"),
            metadata=common_metadata,
        )
        local_moments = (
            None if local_accumulator is None else local_accumulator.finalize()
        )
        tripartite_moments = (
            None
            if tripartite_accumulator is None
            else tripartite_accumulator.finalize()
        )
        tripartite_fold_moments = tuple(
            accumulator.finalize()
            for accumulator in tripartite_fold_accumulators
        )
        local_regime_summary = None
        if local_regime_summaries:
            if len(local_regime_summaries) != streamed_batch_count:
                raise ValueError(
                    "local regime diagnostics did not cover every streamed batch"
                )
            reference_regime = local_regime_summaries[0]
            if any(
                type(item) is not type(reference_regime)
                or item.activation_kind != reference_regime.activation_kind
                or item.geometry_blocks != reference_regime.geometry_blocks
                or item.measured_mask_hash
                != reference_regime.measured_mask_hash
                or item.scope != reference_regime.scope
                or item.observation_measure_id
                != reference_regime.observation_measure_id
                for item in local_regime_summaries[1:]
            ):
                raise ValueError(
                    "local regime geometry changed across streamed batches"
                )
            combined_fields: dict[str, np.ndarray] = {}
            statistic_names = (
                REGIME_STATISTIC_NAMES
                if isinstance(reference_regime, LocalRegimeRouteSummary)
                else SMOOTH_TRANSMISSION_STATISTIC_NAMES
            )
            maximum_fields = {
                "crossing_risk_max",
                "absolute_gain_change_max",
            }
            for name in statistic_names:
                rows = [
                    np.asarray(getattr(item, name), dtype=np.float64)
                    for item in local_regime_summaries
                ]
                combined_fields[name] = (
                    np.maximum.reduce(rows)
                    if name in maximum_fields
                    else np.sum(rows, axis=0)
                )
            summary_type = type(reference_regime)
            local_regime_summary = summary_type(
                route_ids=route_ids,
                source_context_count=sum(
                    item.source_context_count
                    for item in local_regime_summaries
                ),
                context_axis_hash=content_hash(
                    selected, prefix="local-regime-context-axis-"
                ),
                measured_mask_hash=reference_regime.measured_mask_hash,
                source_measurement_hash=content_hash(
                    {
                        "request_source_identity": source_measurement_hash,
                        "component_summary_hashes": tuple(
                            item.content_hash
                            for item in local_regime_summaries
                        ),
                    },
                    prefix="irg-local-regime-measurement-aggregate-",
                ),
                geometry_blocks=reference_regime.geometry_blocks,
                observation_measure_id=(
                    reference_regime.observation_measure_id
                ),
                scope=reference_regime.scope,
                activation_kind=reference_regime.activation_kind,
                **combined_fields,
            )
        global_head_slice = (
            slice(1, len(tripartite_moments.head_ids) - 1)
            if tripartite_moments is not None
            else slice(0, 0)
        )
        atlas_diagnostics: dict[str, object] = {
            "atlas_split_fold_global_embeddings": tuple(
                np.concatenate(
                    (
                        tensor.mean[global_head_slice],
                        tensor.covariance_factor[
                            :, global_head_slice, :
                        ].reshape(-1, len(route_ids)),
                    ),
                    axis=0,
                )
                for tensor in tripartite_fold_moments
            ),
            "atlas_split_fold_context_axis_hashes": tuple(
                tensor.context_axis_hash for tensor in tripartite_fold_moments
            ),
            "atlas_split_fold_validation_count": len(
                tripartite_fold_moments
            ),
            "atlas_split_fold_shared_projection": (
                "deterministic_context_rademacher_same_probe_map_v1"
                if tripartite_fold_moments
                else "not_available"
            ),
        }
        trust_fold_geometries = (
            tuple(
                accumulator.finalize(
                    spec=global_trust_spec,
                    measured_mask_hash=measured_mask_hash,
                    calibration_state_id=self.calibration_state_id,
                    source_measurement_hashes=(
                        f"{source_measurement_hash}:split-fold:{fold_index}",
                    ),
                    metadata={
                        "split_fold_index": fold_index,
                        "split_fold_count": split_fold_count,
                    },
                )
                for fold_index, accumulator in enumerate(
                    trust_fold_accumulators
                )
            )
            if global_trust_spec is not None and trust_fold_accumulators
            else ()
        )
        trust_geometry = (
            None
            if trust_accumulator is None or global_trust_spec is None
            else trust_accumulator.finalize(
                spec=global_trust_spec,
                measured_mask_hash=measured_mask_hash,
                calibration_state_id=self.calibration_state_id,
                source_measurement_hashes=(source_measurement_hash,),
                metadata={
                    "requested_rank": int(trust_rank),
                    "split_fold_embeddings": tuple(
                        np.concatenate(
                            (
                                geometry.mean_component,
                                geometry.centered_factor,
                            ),
                            axis=0,
                        )
                        for geometry in trust_fold_geometries
                    ),
                    "split_fold_residual_spectral_bounds": tuple(
                        geometry.residual_spectral_bound
                        for geometry in trust_fold_geometries
                    ),
                    "split_fold_context_axis_hashes": tuple(
                        geometry.context_axis_hash
                        for geometry in trust_fold_geometries
                    ),
                    "split_fold_validation_count": len(
                        trust_fold_geometries
                    ),
                    "split_fold_assignment": (
                        "frozen_selected_context_round_robin_v1"
                    ),
                },
            )
        )
        if trust_geometry is not None:
            trust_geometry = replace(
                trust_geometry,
                metadata={
                    **dict(trust_geometry.metadata),
                    **atlas_diagnostics,
                },
            )
        if (
            local_moments is not None
            and local_moments.sample_count != fixed.sample_count
        ):
            raise ValueError(
                "local attribution did not cover every response context"
            )
        if (
            tripartite_moments is not None
            and tripartite_moments.sample_count != fixed.sample_count
        ):
            raise ValueError(
                "tripartite pullback did not cover every response context"
            )
        if not intrinsic:
            return (
                fixed,
                None,
                local_moments,
                tripartite_moments,
                trust_geometry,
                None,
                local_regime_summary,
                atlas_diagnostics,
            )
        assert (
            nuisance_gram is not None
            and nuisance_cross is not None
            and nuisance_dimension is not None
        )
        symmetric = (
            0.5 * (nuisance_gram + nuisance_gram.T) / fixed_accumulator.total_weight
        )
        normalized_cross = nuisance_cross / fixed_accumulator.total_weight
        regularized = symmetric + float(regularization) * np.eye(nuisance_dimension)
        raw_spectrum = pseudoinverse_spectrum(symmetric, rcond=float(rcond))
        regularized_spectrum = pseudoinverse_spectrum(
            regularized,
            rcond=float(rcond),
        )
        raw_identified = (
            raw_spectrum.full_rank
            and np.isfinite(raw_spectrum.condition_number)
            and raw_spectrum.condition_number <= float(maximum_condition_number)
        )
        regularized_solve_usable = (
            regularized_spectrum.full_rank
            and np.isfinite(regularized_spectrum.condition_number)
            and regularized_spectrum.condition_number
            <= float(maximum_condition_number)
        )
        if identification_policy == "require_raw_identification":
            quotient_usable = raw_identified
        elif identification_policy == "allow_regularized_solve":
            quotient_usable = regularized_solve_usable
        else:  # defensive: request decoding should already have rejected this
            raise ValueError("streamed quotient identification_policy is invalid")
        if not quotient_usable:
            raise RuntimeError(
                "streamed intrinsic quotient refused because the declared "
                "nuisance-identification policy is not satisfied"
            )
        inverse = np.linalg.pinv(
            regularized,
            rcond=float(rcond),
            hermitian=True,
        )
        coefficients = inverse @ normalized_cross
        intrinsic_accumulator = StreamingResponseMomentAccumulator(
            channel_ids=self.channel_ids,
            route_ids=route_ids,
        )
        intrinsic_atlas_accumulator = (
            None
            if global_trust_spec is None
            else FrequentDirectionsTrustAccumulator(
                route_ids=route_ids,
                sketch_rank=int(trust_rank),
            )
        )
        intrinsic_fold_accumulators = (
            tuple(
                FrequentDirectionsTrustAccumulator(
                    route_ids=route_ids,
                    sketch_rank=int(trust_rank),
                )
                for _ in range(split_fold_count)
            )
            if intrinsic_atlas_accumulator is not None and split_fold_count >= 2
            else ()
        )
        intrinsic_shared_rank = int(self.local_covariance_sketch_rank)
        intrinsic_fold_weight = np.zeros(split_fold_count, dtype=np.float64)
        intrinsic_fold_sum = (
            np.zeros(
                (
                    split_fold_count,
                    len(self.channel_ids),
                    len(route_ids),
                ),
                dtype=np.float64,
            )
            if split_fold_count >= 2
            else None
        )
        intrinsic_fold_sketch = (
            np.zeros(
                (
                    split_fold_count,
                    intrinsic_shared_rank,
                    len(self.channel_ids),
                    len(route_ids),
                ),
                dtype=np.float64,
            )
            if split_fold_count >= 2 and intrinsic_shared_rank > 0
            else None
        )
        intrinsic_fold_constant = (
            np.zeros(
                (split_fold_count, intrinsic_shared_rank),
                dtype=np.float64,
            )
            if intrinsic_fold_sketch is not None
            else None
        )
        for batch in self._batches(selected, route_indices):
            assert batch.nuisance_rows is not None
            residual = batch.route_rows - np.einsum(
                "ndk,kr->ndr", batch.nuisance_rows, coefficients, optimize=True
            )
            intrinsic_accumulator.update(
                residual,
                context_ids=batch.context_ids,
                weights=batch.weights,
            )
            if intrinsic_atlas_accumulator is not None:
                if batch.metric_route_rows is not None:
                    assert batch.metric_nuisance_rows is not None
                    whitened_route = batch.metric_route_rows
                    whitened_nuisance = batch.metric_nuisance_rows
                elif batch.metric_whitener is not None:
                    whitened_route = np.einsum(
                        "nkd,ndr->nkr",
                        batch.metric_whitener,
                        batch.route_rows,
                        optimize=True,
                    )
                    whitened_nuisance = np.einsum(
                        "nkd,ndp->nkp",
                        batch.metric_whitener,
                        batch.nuisance_rows,
                        optimize=True,
                    )
                else:
                    whitened_route = batch.route_rows
                    whitened_nuisance = batch.nuisance_rows
                intrinsic_atlas_accumulator.update(
                    whitened_residual := (
                        whitened_route
                        - np.einsum(
                        "nkp,pr->nkr",
                        whitened_nuisance,
                        coefficients,
                        optimize=True,
                        )
                    ),
                    context_ids=batch.context_ids,
                    weights=batch.weights,
                )
                raw_weights = (
                    np.ones(len(batch.context_ids), dtype=np.float64)
                    if batch.weights is None
                    else np.asarray(batch.weights, dtype=np.float64)
                )
                for fold_index, fold_accumulator in enumerate(
                    intrinsic_fold_accumulators
                ):
                    chosen = np.asarray(
                        [
                            fold_by_context[context_id] == fold_index
                            for context_id in batch.context_ids
                        ],
                        dtype=bool,
                    )
                    if not np.any(chosen):
                        continue
                    fold_accumulator.update(
                        whitened_residual[chosen],
                        context_ids=tuple(
                            context_id
                            for context_id, retain in zip(
                                batch.context_ids, chosen, strict=True
                            )
                            if retain
                        ),
                        weights=(
                            None
                            if batch.weights is None
                            else np.asarray(batch.weights, dtype=np.float64)[chosen]
                        ),
                    )
                    assert intrinsic_fold_sum is not None
                    chosen_weights = raw_weights[chosen]
                    chosen_rows = whitened_residual[chosen]
                    intrinsic_fold_weight[fold_index] += float(
                        chosen_weights.sum()
                    )
                    intrinsic_fold_sum[fold_index] += np.einsum(
                        "n,nkr->kr",
                        chosen_weights,
                        chosen_rows,
                        optimize=True,
                    )
                    if intrinsic_fold_sketch is not None:
                        chosen_ids = tuple(
                            context_id
                            for context_id, retain in zip(
                                batch.context_ids, chosen, strict=True
                            )
                            if retain
                        )
                        signs = np.empty(
                            (intrinsic_shared_rank, len(chosen_ids)),
                            dtype=np.float64,
                        )
                        for probe in range(intrinsic_shared_rank):
                            for column, context_id in enumerate(chosen_ids):
                                digest = hashlib.sha256(
                                    (
                                        f"{self.local_covariance_sketch_seed}:"
                                        f"intrinsic-atlas:{probe}:{context_id}"
                                    ).encode()
                                ).digest()
                                signs[probe, column] = (
                                    1.0 if digest[0] & 1 else -1.0
                                )
                        signs /= np.sqrt(float(intrinsic_shared_rank))
                        root = np.sqrt(chosen_weights)
                        intrinsic_fold_sketch[fold_index] += np.einsum(
                            "qn,n,nkr->qkr",
                            signs,
                            root,
                            chosen_rows,
                            optimize=True,
                        )
                        assert intrinsic_fold_constant is not None
                        intrinsic_fold_constant[fold_index] += np.einsum(
                            "qn,n->q", signs, root
                        )
        diagnostics = {
            "schema": "streamed_semantic_response_quotient_v1",
            "nuisance_dimension": int(nuisance_dimension),
            "condition_number": regularized_spectrum.condition_number,
            "condition_number_is_finite": np.isfinite(
                regularized_spectrum.condition_number
            ),
            "raw_effective_rank": raw_spectrum.effective_rank,
            "raw_condition_number": (
                raw_spectrum.condition_number
                if np.isfinite(raw_spectrum.condition_number)
                else None
            ),
            "raw_condition_number_is_finite": np.isfinite(
                raw_spectrum.condition_number
            ),
            "raw_singular_cutoff": raw_spectrum.singular_cutoff,
            "regularized_effective_rank": regularized_spectrum.effective_rank,
            "regularized_condition_number": (
                regularized_spectrum.condition_number
                if np.isfinite(regularized_spectrum.condition_number)
                else None
            ),
            "regularized_condition_number_is_finite": np.isfinite(
                regularized_spectrum.condition_number
            ),
            "regularized_singular_cutoff": regularized_spectrum.singular_cutoff,
            "raw_nuisance_identified": raw_identified,
            "regularized_solve_usable": regularized_solve_usable,
            "identification_policy": identification_policy,
            "regularization_created_rank": (
                regularized_spectrum.effective_rank
                > raw_spectrum.effective_rank
            ),
            "quotient_result_not_promotable_as_identified": not raw_identified,
            "regularization": float(regularization),
            "rcond": float(rcond),
            "two_pass_streaming": True,
            "fit_coordinate_mode": fit_coordinate_mode,
            "task_metric_factor_required": bool(require_task_metric_factor),
            "metric_factor_dimension": int(
                metric_factor_dimension or len(self.channel_ids)
            ),
            "context_weights_applied_in_quotient_fit": True,
            **atlas_diagnostics,
        }
        intrinsic_split_fold_embeddings: list[np.ndarray] = []
        if intrinsic_fold_sum is not None:
            for fold_index in range(split_fold_count):
                weight = float(intrinsic_fold_weight[fold_index])
                if weight <= 0.0:
                    raise RuntimeError(
                        "intrinsic atlas split-fold assignment produced an empty fold"
                    )
                fold_mean = intrinsic_fold_sum[fold_index] / weight
                rows = [fold_mean]
                if intrinsic_fold_sketch is not None:
                    assert intrinsic_fold_constant is not None
                    centered = (
                        intrinsic_fold_sketch[fold_index]
                        - intrinsic_fold_constant[fold_index, :, None, None]
                        * fold_mean[None, :, :]
                    ) / np.sqrt(weight)
                    rows.append(centered.reshape(-1, len(route_ids)))
                intrinsic_split_fold_embeddings.append(
                    np.concatenate(rows, axis=0)
                )
        diagnostics.update(
            {
                "intrinsic_atlas_split_fold_embeddings": tuple(
                    intrinsic_split_fold_embeddings
                ),
                "intrinsic_atlas_split_fold_validation_count": len(
                    intrinsic_split_fold_embeddings
                ),
                "intrinsic_atlas_split_fold_shared_projection": (
                    "deterministic_context_rademacher_same_probe_map_v1"
                    if intrinsic_split_fold_embeddings
                    else "not_available"
                ),
            }
        )
        intrinsic_moments = intrinsic_accumulator.finalize(
            response_spec_hash=self.response_spec_hash,
            metric_id=self.metric_id,
            context_axis_hash=content_hash(selected, prefix="response-context-axis-"),
            metadata={
                **common_metadata,
                "nuisance_quotiented": True,
                "quotient_diagnostics": diagnostics,
            },
        )
        intrinsic_fold_geometries = (
            tuple(
                accumulator.finalize(
                    spec=global_trust_spec,
                    measured_mask_hash=measured_mask_hash,
                    calibration_state_id=self.calibration_state_id,
                    source_measurement_hashes=(
                        f"{source_measurement_hash}:intrinsic-split-fold:{fold_index}",
                    ),
                    metadata={
                        "geometry_role": (
                            "nuisance_quotiented_atlas_embedding"
                        ),
                        "not_physical_trust_authority": True,
                        "split_fold_index": fold_index,
                        "split_fold_count": split_fold_count,
                    },
                )
                for fold_index, accumulator in enumerate(
                    intrinsic_fold_accumulators
                )
            )
            if global_trust_spec is not None and intrinsic_fold_accumulators
            else ()
        )
        intrinsic_atlas_geometry = (
            None
            if intrinsic_atlas_accumulator is None or global_trust_spec is None
            else intrinsic_atlas_accumulator.finalize(
                spec=global_trust_spec,
                measured_mask_hash=measured_mask_hash,
                calibration_state_id=self.calibration_state_id,
                source_measurement_hashes=(source_measurement_hash,),
                metadata={
                    "geometry_role": "nuisance_quotiented_atlas_embedding",
                    "not_physical_trust_authority": True,
                    "requested_rank": int(trust_rank),
                    "intrinsic_atlas_split_fold_embeddings": tuple(
                        intrinsic_split_fold_embeddings
                    ),
                    "intrinsic_atlas_split_fold_validation_count": len(
                        intrinsic_split_fold_embeddings
                    ),
                    "intrinsic_atlas_split_fold_shared_projection": (
                        "deterministic_context_rademacher_same_probe_map_v1"
                        if intrinsic_split_fold_embeddings
                        else "not_available"
                    ),
                    "split_fold_embeddings": tuple(
                        np.concatenate(
                            (
                                geometry.mean_component,
                                geometry.centered_factor,
                            ),
                            axis=0,
                        )
                        for geometry in intrinsic_fold_geometries
                    ),
                    "split_fold_residual_spectral_bounds": tuple(
                        geometry.residual_spectral_bound
                        for geometry in intrinsic_fold_geometries
                    ),
                    "split_fold_context_axis_hashes": tuple(
                        geometry.context_axis_hash
                        for geometry in intrinsic_fold_geometries
                    ),
                    "split_fold_validation_count": len(
                        intrinsic_fold_geometries
                    ),
                    "split_fold_assignment": (
                        "frozen_selected_context_round_robin_v1"
                    ),
                },
            )
        )
        return (
            fixed,
            intrinsic_moments,
            local_moments,
            tripartite_moments,
            trust_geometry,
            intrinsic_atlas_geometry,
            local_regime_summary,
            diagnostics,
        )

    def _factor_output(
        self,
        request: MeasurementRequest,
        selected: tuple[str, ...],
        route_indices: Sequence[int],
        route_ids: Sequence[str],
    ) -> tuple[np.ndarray, tuple[ArtifactRef, ...], dict[str, object]]:
        retain_nuisance = "nuisance_factor" in tuple(request.payload.get("retain", ()))
        retain_metric = bool(request.payload.get("retain_task_metric_factor", False))
        require_metric = bool(request.payload.get("require_task_metric_factor", False))
        # Discover nuisance and metric width from the first batch before allocating.
        batches = iter(self._batches(selected, route_indices))
        try:
            first = next(batches)
        except StopIteration as error:  # pragma: no cover - protected by _batches
            raise ValueError(
                "streaming response callback returned no batches"
            ) from error
        nuisance_dimension = 0
        if retain_nuisance:
            if first.nuisance_rows is None:
                raise ValueError(
                    "requested nuisance factor is absent from the streaming callback"
                )
            nuisance_dimension = first.nuisance_rows.shape[2]
        if (
            require_metric
            and first.metric_whitener is None
            and first.metric_route_rows is None
        ):
            raise ValueError(
                "requested task-metric factor is absent from the streaming callback"
            )
        if not retain_metric:
            metric_dimension = 0
        elif first.metric_route_rows is not None:
            metric_dimension = first.metric_route_rows.shape[1]
        elif first.metric_whitener is not None:
            metric_dimension = first.metric_whitener.shape[1]
        else:
            metric_dimension = 0
        width = len(route_ids) + nuisance_dimension
        response_width = len(self.channel_ids) + metric_dimension
        shape = (len(selected), response_width, width)
        elements = int(np.prod(shape, dtype=np.int64))
        artifacts: tuple[ArtifactRef, ...] = ()
        artifact_path = None
        if self.artifact_root is not None:
            artifact_path = self.artifact_root / f"{request.request_id}.npy"
            output = np.lib.format.open_memmap(
                artifact_path, mode="w+", dtype=np.float64, shape=shape
            )
        else:
            if elements > self.maximum_factor_elements_in_memory:
                raise ValueError(
                    "screened response factor exceeds in-memory limit; configure artifact_root or reduce the route screen"
                )
            output = np.empty(shape, dtype=np.float64)

        offset = 0
        for batch in chain((first,), batches):
            count = len(batch.context_ids)
            semantic_slice = slice(0, len(self.channel_ids))
            output[offset : offset + count, semantic_slice, : len(route_ids)] = (
                batch.route_rows
            )
            if retain_nuisance:
                assert batch.nuisance_rows is not None
                if batch.nuisance_rows.shape[2] != nuisance_dimension:
                    raise ValueError(
                        "nuisance dimension changed across streamed batches"
                    )
                output[offset : offset + count, semantic_slice, len(route_ids) :] = (
                    batch.nuisance_rows
                )
            if metric_dimension:
                if batch.metric_route_rows is not None:
                    if batch.metric_route_rows.shape[1] != metric_dimension:
                        raise ValueError(
                            "task-metric factor changed across streamed batches"
                        )
                    metric_route = batch.metric_route_rows
                else:
                    if (
                        batch.metric_whitener is None
                        or batch.metric_whitener.shape[1] != metric_dimension
                    ):
                        raise ValueError(
                            "task-metric factor changed across streamed batches"
                        )
                    metric_route = np.einsum(
                        "nkd,ndr->nkr",
                        batch.metric_whitener,
                        batch.route_rows,
                        optimize=True,
                    )
                metric_slice = slice(len(self.channel_ids), response_width)
                output[offset : offset + count, metric_slice, : len(route_ids)] = (
                    metric_route
                )
                if retain_nuisance:
                    assert batch.nuisance_rows is not None
                    if batch.metric_nuisance_rows is not None:
                        metric_nuisance = batch.metric_nuisance_rows
                    else:
                        assert batch.metric_whitener is not None
                        metric_nuisance = np.einsum(
                            "nkd,ndp->nkp",
                            batch.metric_whitener,
                            batch.nuisance_rows,
                            optimize=True,
                        )
                    output[offset : offset + count, metric_slice, len(route_ids) :] = (
                        metric_nuisance
                    )
            offset += count
        if isinstance(output, np.memmap):
            output.flush()
            del output
            values = np.load(artifact_path, mmap_mode="r", allow_pickle=False)
            array_hash = content_hash(
                np.asarray(values), prefix="streamed-response-factor-"
            )
            artifacts = (
                ArtifactRef(
                    artifact_id=f"response-factor:{request.request_id}",
                    kind="npy_memmap_response_factor",
                    uri=str(artifact_path),
                    content_hash=array_hash,
                ),
            )
        else:
            values = output
        factor_metadata: dict[str, object] = {
            "streamed_native_scale_provider": True,
            "artifact_backed": artifact_path is not None,
            "nuisance_dimension": nuisance_dimension,
            "returned_route_count": len(route_ids),
            "projection_residual_bound": self.projection_residual_bound,
            "sketch_residual_spectral_bound": self.sketch_residual_spectral_bound,
            "omitted_route_residual_bound": self.omitted_route_residual_bound,
            "task_metric_factor_retained": bool(metric_dimension),
        }
        if metric_dimension:
            factor_metadata["response_factor_layout"] = {
                "schema": "semantic_and_metric_response_factor_v1",
                "semantic_rows": (0, len(self.channel_ids)),
                "metric_rows": (len(self.channel_ids), response_width),
            }
        return values, artifacts, factor_metadata

    def execute(self, request: MeasurementRequest) -> MeasurementOutcome:
        if request.kind not in self.supported_kinds:
            raise ValueError(
                "streaming response provider received an unsupported request"
            )
        selected = self._selected_contexts(request)
        route_indices, route_ids = self._routes(request)
        compact_whole_axis = isinstance(route_indices, range)
        metadata_route_indices: object = (
            {
                "schema": "constructive_ordinal_index_axis_ref_v1",
                "start": route_indices.start,
                "stop": route_indices.stop,
                "step": route_indices.step,
            }
            if compact_whole_axis
            else route_indices
        )
        metadata_route_ids: object = (
            {
                "schema": "ordered_string_axis_ref_v1",
                "route_ids_count": len(route_ids),
                "route_ids_hash": ordered_string_axis_hash(
                    route_ids,
                    name="streaming response-provider returned route axis",
                ),
            }
            if compact_whole_axis
            else route_ids
        )
        metadata: dict[str, object] = {
            "kind": request.kind,
            "mask_hash": request.payload.get("mask_hash", "unknown-mask"),
            "response_contract_hash": self.response_spec_hash,
            "response_metric_id": self.metric_id,
            "response_dimension": len(self.channel_ids),
            "response_channel_ids": self.channel_ids,
            "route_indices": metadata_route_indices,
            "route_ids": metadata_route_ids,
            "route_axis_compact_reference": compact_whole_axis,
            "full_route_count": len(self.route_ids),
            "returned_route_count": len(route_ids),
            "projection_residual_bound": self.projection_residual_bound,
            "sketch_residual_spectral_bound": self.sketch_residual_spectral_bound,
            "omitted_route_residual_bound": self.omitted_route_residual_bound,
            "contextual_variability_separate_from_estimation_uncertainty": True,
            "screen_moment_hash": request.payload.get("screen_moment_hash"),
            "screen_receipt": request.payload.get("screen_receipt"),
            "functional_atlas_spec": request.payload.get(
                "functional_atlas_spec"
            ),
            "interaction_schedule_spec": request.payload.get(
                "interaction_schedule_spec"
            ),
        }
        artifacts: tuple[ArtifactRef, ...] = ()
        if request.kind == "response_moments":
            intrinsic = bool(request.payload.get("intrinsic_nuisance_quotient", False))
            raw_trust_spec = request.payload.get("global_trust_geometry_spec")
            if raw_trust_spec is None:
                trust_spec = None
            elif isinstance(raw_trust_spec, GlobalTrustGeometrySpec):
                trust_spec = raw_trust_spec
            elif isinstance(raw_trust_spec, Mapping):
                trust_spec = GlobalTrustGeometrySpec(**dict(raw_trust_spec))
            else:
                raise ValueError(
                    "global trust geometry request payload has an invalid spec"
                )
            raw_atlas_spec = request.payload.get("functional_atlas_spec")
            if raw_atlas_spec is None:
                atlas_spec = None
            elif isinstance(raw_atlas_spec, FunctionalAtlasSpec):
                atlas_spec = raw_atlas_spec
            elif isinstance(raw_atlas_spec, Mapping):
                atlas_spec = FunctionalAtlasSpec(**dict(raw_atlas_spec))
            else:
                raise ValueError(
                    "functional atlas request payload has an invalid spec"
                )
            (
                fixed,
                intrinsic_moments,
                local_moments,
                tripartite_moments,
                trust_geometry,
                intrinsic_atlas_geometry,
                local_regime_summary,
                diagnostics,
            ) = self._moments(
                selected,
                route_indices,
                route_ids,
                intrinsic=intrinsic,
                regularization=float(request.payload.get("regularization", 1e-8)),
                rcond=float(request.payload.get("rcond", 1e-10)),
                maximum_condition_number=float(
                    request.payload.get("maximum_condition_number", 1e12)
                ),
                identification_policy=cast(
                    IdentificationPolicy,
                    request.payload.get(
                        "identification_policy",
                        "require_raw_identification",
                    ),
                ),
                require_task_metric_factor=bool(
                    request.payload.get("require_task_metric_factor", False)
                ),
                global_trust_spec=trust_spec,
                global_trust_requested_rank=(
                    None
                    if request.payload.get("global_trust_requested_rank")
                    is None
                    else int(
                        request.payload["global_trust_requested_rank"]
                    )
                ),
                atlas_split_fold_count=(
                    0 if atlas_spec is None else atlas_spec.split_fold_count
                ),
                measured_mask_hash=str(
                    request.payload.get("mask_hash", "unknown-mask")
                ),
                source_measurement_hash=request.request_id,
            )
            fixed_values, fixed_layout = fixed.pack()
            if intrinsic_moments is None:
                values = fixed_values
                metadata.update(
                    {
                        "moment_layout": fixed_layout,
                        "sample_count": len(selected),
                        "statistic_rows": "matrix_valued_response_moments",
                    }
                )
            else:
                intrinsic_values, intrinsic_layout = intrinsic_moments.pack()
                values = np.concatenate((fixed_values, intrinsic_values), axis=0)
                metadata.update(
                    {
                        "joint_moment_layout": {
                            "schema": "fixed_and_intrinsic_response_moments_v1",
                            "fixed_row_count": int(fixed_values.shape[0]),
                            "fixed": fixed_layout,
                            "intrinsic": intrinsic_layout,
                        },
                        "sample_count": len(selected),
                        "statistic_rows": "fixed_and_intrinsic_matrix_valued_response_moments",
                        "quotient_diagnostics": diagnostics,
                    }
                )
            metadata.update(diagnostics)
            if local_moments is not None:
                local_values, local_layout = local_moments.pack()
                local_offset = int(values.shape[0])
                values = np.concatenate((values, local_values), axis=0)
                metadata.update(
                    {
                        "local_attribution_layout": {
                            **local_layout,
                            "row_offset": local_offset,
                            "row_count": int(local_values.shape[0]),
                        },
                        "local_attribution_functional": (
                            "signed_immediate_destination_contribution_v1"
                        ),
                        "local_covariance_representation": (
                            "deterministic_context_rademacher_energy_factor_v1"
                        ),
                        "local_covariance_guidance_only": True,
                        "local_covariance_residual_bound_available": False,
                        "local_covariance_claim_status": (
                            "randomized_energy_approximation"
                        ),
                    }
                )
            if tripartite_moments is not None:
                pullback_values, pullback_layout = tripartite_moments.pack()
                pullback_offset = int(values.shape[0])
                values = np.concatenate((values, pullback_values), axis=0)
                metadata.update(
                    {
                        "tripartite_pullback_layout": {
                            **pullback_layout,
                            "row_offset": pullback_offset,
                            "row_count": int(pullback_values.shape[0]),
                        },
                        "nested_pullback_order": (
                            "local_destination_to_global_response_to_bce_loss"
                        ),
                        "tripartite_covariance_guidance_only": True,
                        "tripartite_covariance_residual_bound_available": False,
                        "tripartite_covariance_claim_status": (
                            "randomized_energy_approximation"
                        ),
                    }
                )
            if local_regime_summary is not None:
                regime_values, regime_layout = local_regime_summary.pack()
                regime_offset = int(values.shape[0])
                values = np.concatenate((values, regime_values), axis=0)
                metadata.update(
                    {
                        "local_regime_summary_layout": {
                            **regime_layout,
                            "row_offset": regime_offset,
                            "row_count": int(regime_values.shape[0]),
                        },
                        "local_regime_dense_spatial_tensor_retained": False,
                        "local_regime_claim_scope": (
                            "singleton_local_crossing_screen_not_global_gate_proof"
                        ),
                        "population_gate_validation_required": True,
                    }
                )
            if trust_geometry is not None:
                trust_values, trust_layout = trust_geometry.pack()
                trust_offset = int(values.shape[0])
                values = np.concatenate((values, trust_values), axis=0)
                metadata.update(
                    {
                        "global_trust_geometry_layout": {
                            **trust_layout,
                            "row_offset": trust_offset,
                            "row_count": int(trust_values.shape[0]),
                        },
                        "global_trust_cross_partition_terms_retained": True,
                        "global_trust_factorization_is_not_partitioning": True,
                        "global_trust_residual_certified": True,
                    }
                )
            if intrinsic_atlas_geometry is not None:
                intrinsic_values, intrinsic_geometry_layout = (
                    intrinsic_atlas_geometry.pack()
                )
                intrinsic_offset = int(values.shape[0])
                values = np.concatenate((values, intrinsic_values), axis=0)
                metadata.update(
                    {
                        "intrinsic_atlas_geometry_layout": {
                            **intrinsic_geometry_layout,
                            "row_offset": intrinsic_offset,
                            "row_count": int(intrinsic_values.shape[0]),
                        },
                        "intrinsic_atlas_nuisance_quotiented": True,
                        "intrinsic_atlas_geometry_is_not_physical_trust": True,
                    }
                )
        else:
            values, artifacts, factor_metadata = self._factor_output(
                request, selected, route_indices, route_ids
            )
            metadata.update(factor_metadata)
            metadata["returned_tensor"] = "semantic_response_factor"
        return MeasurementOutcome.create(
            request_id=request.request_id,
            state_hash=request.state_hash,
            context_ids=selected,
            candidate_ids=request.candidate_ids,
            values=values,
            calibration_state_id=self.calibration_state_id,
            realized_cost=self.realized_cost,
            cache_artifacts=artifacts,
            metadata=metadata,
        )


__all__ = [
    "LocalAttributionMomentTensor",
    "ResponseJacobianBatch",
    "StreamingLocalAttributionAccumulator",
    "StreamingResponseGeometryProvider",
    "StreamingResponseMomentAccumulator",
    "StreamingTripartitePullbackAccumulator",
    "TripartitePullbackMomentTensor",
]
