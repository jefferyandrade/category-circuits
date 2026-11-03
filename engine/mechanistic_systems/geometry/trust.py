"""Globally coupled, residual-certified response trust geometry."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from mechanistic_systems.core.axis_identity import (
    canonical_unique_string_axis,
    ordered_string_axis_hash,
    resolve_packed_string_axis,
)

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.trust_geometry import (
    GlobalTrustGeometrySpec,
    ModelValidityDecision,
)

GLOBAL_TRUST_RANK_GROWTH_CACHE_KEY = "irg_global_trust_rank_growth_obligation_v1"
GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY = "irg_global_trust_requested_rank_v1"

GLOBAL_TRUST_GEOMETRY_STATE_SCHEMA = "irg_global_trust_geometry_state_v1"
GLOBAL_TRUST_DERIVATION_RECEIPT_SCHEMA = "irg_global_trust_accumulator_derivation_receipt_v1"
GLOBAL_TRUST_CANONICAL_FD_ALGORITHM_ID = "weighted_welford_frequent_directions_v1"
GLOBAL_TRUST_CUDA_FD_ALGORITHM_ID = "weighted_welford_frequent_directions_cuda_v1"
GLOBAL_TRUST_GRAM_FD_ALGORITHM_ID = "weighted_welford_gram_frequent_directions_v2"
GLOBAL_TRUST_CUDA_GRAM_FD_ALGORITHM_ID = "weighted_welford_gram_frequent_directions_cuda_v2"
GLOBAL_TRUST_SOURCE_ROW_HASH_PREFIX = "irg-global-trust-source-row-"
GLOBAL_TRUST_SOURCE_ROW_CHAIN_PREFIX = "irg-global-trust-source-row-chain-"
GLOBAL_TRUST_STREAM_WEIGHT_CHAIN_PREFIX = "irg-global-trust-stream-weight-chain-"

_SOURCE_ROW_CHAIN_SCHEMA = "irg_global_trust_ordered_source_row_chain_v1"
_STREAM_WEIGHT_CHAIN_SCHEMA = "irg_global_trust_stream_weight_chain_v1"
_ACCUMULATOR_DERIVATION_TOKEN = object()
_PROTECTED_ACCUMULATOR_METADATA_KEYS = frozenset(
    {
        "sketch_algorithm",
        "cross_module_terms_retained",
        "spectral_residual_is_pathwise_upper_envelope",
        "residual_diagonal_is_exact_streamed_fd_remainder",
    }
)


def _immutable_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return an immutable, bytes-backed copy of an array.

    Merely clearing NumPy's writable flag is reversible when the exposed
    array owns its allocation.  A view over immutable ``bytes`` cannot be
    made writable again, which is required for metadata that participates in
    a cached content hash.
    """

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"{name} must not contain Python objects")
    contiguous = np.ascontiguousarray(array)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return frozen.reshape(contiguous.shape)


def _deep_freeze_metadata(value: Any, *, name: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise TypeError(f"{name} keys must be nonempty strings")
            frozen[raw_key] = _deep_freeze_metadata(
                raw_value,
                name=f"{name}.{raw_key}",
            )
        return MappingProxyType(frozen)
    if isinstance(value, np.ndarray):
        return _immutable_array(value, name=name)
    if isinstance(value, np.generic):
        return _deep_freeze_metadata(value.item(), name=name)
    if isinstance(value, (tuple, list)):
        return tuple(
            _deep_freeze_metadata(item, name=f"{name}[{index}]") for index, item in enumerate(value)
        )
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze_metadata(item, name=f"{name}[set-item]") for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain nonfinite floats")
        return float(value)
    if isinstance(value, (str, bytes, int, bool, Path, Enum)) or value is None:
        return value
    raise TypeError(f"{name} contains unsupported value {type(value)!r}")


def _frozen_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = _deep_freeze_metadata(value, name="global trust metadata")
    if not isinstance(frozen, Mapping):  # pragma: no cover - defensive
        raise TypeError("global trust metadata must be a mapping")
    return frozen


def _deep_thaw_metadata(value: Any) -> Any:
    """Return a serialization-safe copy of deeply frozen metadata."""

    if isinstance(value, Mapping):
        return {key: _deep_thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_deep_thaw_metadata(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_deep_thaw_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return value


def global_trust_source_row_hash(row: np.ndarray) -> str:
    """Content identity of one exact response-by-route accumulator row."""

    value = np.asarray(row, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("global trust source row must be a finite matrix")
    return content_hash(value, prefix=GLOBAL_TRUST_SOURCE_ROW_HASH_PREFIX)


def _advance_ordered_chain(
    chain_hash: str,
    *,
    index: int,
    value: str | float,
    schema: str,
    prefix: str,
) -> str:
    return content_hash(
        {
            "schema": schema,
            "previous_chain_hash": chain_hash,
            "entry_index": int(index),
            "entry_value": value,
        },
        prefix=prefix,
    )


def _empty_ordered_chain(*, schema: str, prefix: str) -> str:
    return content_hash(
        {"schema": schema, "empty": True},
        prefix=prefix,
    )


def _finalize_ordered_chain(
    chain_hash: str,
    *,
    entry_count: int,
    schema: str,
    prefix: str,
) -> str:
    return content_hash(
        {
            "schema": schema,
            "entry_count": int(entry_count),
            "terminal_chain_hash": chain_hash,
        },
        prefix=prefix,
    )


def global_trust_ordered_source_rows_hash(
    source_row_hashes: Sequence[str],
) -> str:
    """Ordered aggregate independently reproducible from a source ledger."""

    chain = _empty_ordered_chain(
        schema=_SOURCE_ROW_CHAIN_SCHEMA,
        prefix=GLOBAL_TRUST_SOURCE_ROW_CHAIN_PREFIX,
    )
    count = 0
    for index, raw_hash in enumerate(source_row_hashes):
        row_hash = str(raw_hash)
        if not row_hash:
            raise ValueError("source row hashes must be nonempty")
        chain = _advance_ordered_chain(
            chain,
            index=index,
            value=row_hash,
            schema=_SOURCE_ROW_CHAIN_SCHEMA,
            prefix=GLOBAL_TRUST_SOURCE_ROW_CHAIN_PREFIX,
        )
        count += 1
    return _finalize_ordered_chain(
        chain,
        entry_count=count,
        schema=_SOURCE_ROW_CHAIN_SCHEMA,
        prefix=GLOBAL_TRUST_SOURCE_ROW_CHAIN_PREFIX,
    )


def global_trust_stream_weight_hash(
    weights: Sequence[float] | np.ndarray,
) -> str:
    """Ordered float64-normalized weight identity used by accumulator and ledger."""

    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not values.size or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("global trust stream weights must be finite and nonnegative")
    chain = _empty_ordered_chain(
        schema=_STREAM_WEIGHT_CHAIN_SCHEMA,
        prefix=GLOBAL_TRUST_STREAM_WEIGHT_CHAIN_PREFIX,
    )
    for index, weight in enumerate(values):
        chain = _advance_ordered_chain(
            chain,
            index=index,
            value=float(weight),
            schema=_STREAM_WEIGHT_CHAIN_SCHEMA,
            prefix=GLOBAL_TRUST_STREAM_WEIGHT_CHAIN_PREFIX,
        )
    return _finalize_ordered_chain(
        chain,
        entry_count=int(values.size),
        schema=_STREAM_WEIGHT_CHAIN_SCHEMA,
        prefix=GLOBAL_TRUST_STREAM_WEIGHT_CHAIN_PREFIX,
    )


def _readonly_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix")
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


def _readonly_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all() or np.any(array < -1e-10):
        raise ValueError(f"{name} must be a finite nonnegative vector")
    array = np.maximum(array, 0.0)
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True, init=False)
class GlobalTrustGeometryDerivationReceipt:
    """Content-addressed proof emitted by the exact stream accumulator.

    The receipt binds the state arrays to the ordered row and weight stream
    observed by the accumulator.  It is proposal provenance, not a decision
    certificate.  Historical states may omit it and remain usable by legacy
    consumers, but such states are not eligible for the modern-v1 adapter.
    """

    producer_algorithm_id: str
    producer_implementation: str
    trust_spec_hash: str
    ordered_route_axis_hash: str
    context_axis_hash: str
    stream_entry_count: int
    effective_context_count: int
    ordered_source_rows_hash: str
    stream_weight_hash: str
    mean_component_hash: str
    centered_factor_hash: str
    exact_residual_diagonal_hash: str
    total_weight: float
    cumulative_fd_shrink: float
    normalized_stream_squared_frobenius: float
    sketch_rank: int
    measured_mask_hash: str
    calibration_state_id: str
    source_measurement_hashes: tuple[str, ...]
    schema: str = GLOBAL_TRUST_DERIVATION_RECEIPT_SCHEMA

    def __init__(
        self,
        *,
        producer_algorithm_id: str,
        producer_implementation: str,
        trust_spec_hash: str,
        ordered_route_axis_hash: str,
        context_axis_hash: str,
        stream_entry_count: int,
        effective_context_count: int,
        ordered_source_rows_hash: str,
        stream_weight_hash: str,
        mean_component_hash: str,
        centered_factor_hash: str,
        exact_residual_diagonal_hash: str,
        total_weight: float,
        cumulative_fd_shrink: float,
        normalized_stream_squared_frobenius: float,
        sketch_rank: int,
        measured_mask_hash: str,
        calibration_state_id: str,
        source_measurement_hashes: tuple[str, ...],
        schema: str = GLOBAL_TRUST_DERIVATION_RECEIPT_SCHEMA,
        _accumulator_token: object | None = None,
    ) -> None:
        if _accumulator_token is not _ACCUMULATOR_DERIVATION_TOKEN:
            raise TypeError(
                "global trust derivation receipts can only be issued by "
                "FrequentDirectionsTrustAccumulator"
            )
        values = {
            "producer_algorithm_id": producer_algorithm_id,
            "producer_implementation": producer_implementation,
            "trust_spec_hash": trust_spec_hash,
            "ordered_route_axis_hash": ordered_route_axis_hash,
            "context_axis_hash": context_axis_hash,
            "stream_entry_count": stream_entry_count,
            "effective_context_count": effective_context_count,
            "ordered_source_rows_hash": ordered_source_rows_hash,
            "stream_weight_hash": stream_weight_hash,
            "mean_component_hash": mean_component_hash,
            "centered_factor_hash": centered_factor_hash,
            "exact_residual_diagonal_hash": exact_residual_diagonal_hash,
            "total_weight": total_weight,
            "cumulative_fd_shrink": cumulative_fd_shrink,
            "normalized_stream_squared_frobenius": (normalized_stream_squared_frobenius),
            "sketch_rank": sketch_rank,
            "measured_mask_hash": measured_mask_hash,
            "calibration_state_id": calibration_state_id,
            "source_measurement_hashes": source_measurement_hashes,
            "schema": schema,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != GLOBAL_TRUST_DERIVATION_RECEIPT_SCHEMA:
            raise ValueError("unsupported global trust derivation-receipt schema")
        supported_producers = {
            GLOBAL_TRUST_CANONICAL_FD_ALGORITHM_ID: "canonical_numpy",
            GLOBAL_TRUST_CUDA_FD_ALGORITHM_ID: "torch_cuda",
            GLOBAL_TRUST_GRAM_FD_ALGORITHM_ID: "gram_numpy",
            GLOBAL_TRUST_CUDA_GRAM_FD_ALGORITHM_ID: "gram_torch_cuda",
        }
        expected_implementation = supported_producers.get(self.producer_algorithm_id)
        if expected_implementation is None:
            raise ValueError("unsupported global trust derivation algorithm")
        if self.producer_implementation != expected_implementation:
            raise ValueError("global trust derivation implementation/algorithm mismatch")
        for name in (
            "ordered_route_axis_hash",
            "trust_spec_hash",
            "context_axis_hash",
            "ordered_source_rows_hash",
            "stream_weight_hash",
            "mean_component_hash",
            "centered_factor_hash",
            "exact_residual_diagonal_hash",
            "measured_mask_hash",
            "calibration_state_id",
        ):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        stream_count = int(self.stream_entry_count)
        effective_count = int(self.effective_context_count)
        sketch_rank = int(self.sketch_rank)
        if (
            isinstance(self.stream_entry_count, bool)
            or stream_count <= 0
            or stream_count != self.stream_entry_count
            or isinstance(self.effective_context_count, bool)
            or effective_count <= 0
            or effective_count != self.effective_context_count
            or effective_count > stream_count
            or isinstance(self.sketch_rank, bool)
            or sketch_rank <= 0
            or sketch_rank != self.sketch_rank
        ):
            raise ValueError("global trust derivation counts/rank are invalid")
        total_weight = float(self.total_weight)
        shrink = float(self.cumulative_fd_shrink)
        streamed_energy = float(self.normalized_stream_squared_frobenius)
        if (
            not math.isfinite(total_weight)
            or total_weight <= 0.0
            or not math.isfinite(shrink)
            or shrink < 0.0
            or not math.isfinite(streamed_energy)
            or streamed_energy < 0.0
        ):
            raise ValueError("global trust derivation numerics are invalid")
        measurements = tuple(map(str, self.source_measurement_hashes))
        if not measurements or any(not item for item in measurements):
            raise ValueError("source measurement hashes must be nonempty")
        object.__setattr__(self, "stream_entry_count", stream_count)
        object.__setattr__(self, "effective_context_count", effective_count)
        object.__setattr__(self, "sketch_rank", sketch_rank)
        object.__setattr__(self, "total_weight", total_weight)
        object.__setattr__(self, "cumulative_fd_shrink", shrink)
        object.__setattr__(
            self,
            "normalized_stream_squared_frobenius",
            streamed_energy,
        )
        object.__setattr__(self, "source_measurement_hashes", measurements)

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="irg-global-trust-accumulator-derivation-",
        )


@dataclass(frozen=True, slots=True)
class GlobalTrustGeometryState:
    """Full-route factor with a deterministic omitted-energy envelope."""

    spec: GlobalTrustGeometrySpec
    route_ids: tuple[str, ...]
    mean_component: np.ndarray
    centered_factor: np.ndarray
    residual_spectral_bound: float
    total_weight: float
    context_count: int
    context_axis_hash: str
    measured_mask_hash: str
    calibration_state_id: str
    source_measurement_hashes: tuple[str, ...]
    residual_diagonal_bound: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    derivation_receipt: GlobalTrustGeometryDerivationReceipt | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = GLOBAL_TRUST_GEOMETRY_STATE_SCHEMA
    _content_hash: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.schema) is not str or self.schema != GLOBAL_TRUST_GEOMETRY_STATE_SCHEMA:
            raise ValueError("unsupported global trust geometry state schema")
        routes = canonical_unique_string_axis(
            self.route_ids,
            name="global trust route axis",
        )
        mean = _readonly_matrix(self.mean_component, name="trust mean component")
        factor = _readonly_matrix(self.centered_factor, name="trust centered factor")
        residual = float(self.residual_spectral_bound)
        residual_diagonal = (
            None
            if self.residual_diagonal_bound is None
            else _readonly_vector(
                self.residual_diagonal_bound,
                name="trust residual diagonal bound",
            )
        )
        weight = float(self.total_weight)
        if (
            mean.shape[1] != len(routes)
            or factor.shape[1] != len(routes)
            or (residual_diagonal is not None and residual_diagonal.shape != (len(routes),))
            or factor.shape[0] > self.spec.maximum_rank
            or not math.isfinite(residual)
            or residual < 0.0
            or not math.isfinite(weight)
            or weight <= 0.0
            or int(self.context_count) <= 0
            or not self.context_axis_hash
            or not self.measured_mask_hash
            or not self.calibration_state_id
            or not self.source_measurement_hashes
        ):
            raise ValueError("global trust geometry has invalid identity or axes")
        object.__setattr__(self, "route_ids", routes)
        object.__setattr__(self, "mean_component", mean)
        object.__setattr__(self, "centered_factor", factor)
        object.__setattr__(self, "residual_spectral_bound", residual)
        object.__setattr__(
            self,
            "residual_diagonal_bound",
            residual_diagonal,
        )
        object.__setattr__(self, "total_weight", weight)
        object.__setattr__(self, "context_count", int(self.context_count))
        object.__setattr__(self, "source_measurement_hashes", tuple(self.source_measurement_hashes))
        spec = replace(self.spec, metadata=dict(self.spec.metadata))
        object.__setattr__(spec, "metadata", _frozen_metadata(self.spec.metadata))
        object.__setattr__(self, "spec", spec)
        receipt = self.derivation_receipt
        if receipt is not None and type(receipt) is not GlobalTrustGeometryDerivationReceipt:
            raise TypeError("derivation_receipt must be an exact accumulator receipt")
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))

    @property
    def rank(self) -> int:
        return int(self.centered_factor.shape[0])

    @property
    def content_hash(self) -> str:
        cached = self._content_hash
        if cached is None:
            payload = {
                "schema": self.schema,
                "spec": self.spec,
                "route_axis_hash": ordered_string_axis_hash(
                    self.route_ids,
                    name="global trust route axis",
                ),
                "route_count": len(self.route_ids),
                "mean_component": self.mean_component,
                "centered_factor": self.centered_factor,
                "residual_spectral_bound": self.residual_spectral_bound,
                "residual_diagonal_bound": self.residual_diagonal_bound,
                "total_weight": self.total_weight,
                "context_count": self.context_count,
                "context_axis_hash": self.context_axis_hash,
                "measured_mask_hash": self.measured_mask_hash,
                "calibration_state_id": self.calibration_state_id,
                "source_measurement_hashes": self.source_measurement_hashes,
                "metadata": self.metadata,
            }
            # Keep pre-receipt states in their historical hash domain while
            # binding every newly receipted state to its source proof.
            if self.derivation_receipt is not None:
                payload["derivation_receipt"] = self.derivation_receipt
            cached = content_hash(payload, prefix="irg-global-trust-")
            object.__setattr__(self, "_content_hash", cached)
        return cached

    def action_energy_bounds(self, action: Sequence[float] | np.ndarray) -> tuple[float, float]:
        vector = np.asarray(action, dtype=np.float64).reshape(-1)
        if vector.shape != (len(self.route_ids),) or not np.isfinite(vector).all():
            raise ValueError("trust action must be finite on the route axis")
        mean_projection = self.mean_component @ vector
        factor_projection = self.centered_factor @ vector
        lower = float(mean_projection @ mean_projection + factor_projection @ factor_projection)
        spectral_remainder = self.residual_spectral_bound * float(vector @ vector)
        if self.residual_diagonal_bound is None:
            remainder = spectral_remainder
        else:
            # Frequent Directions leaves a PSD residual matrix E.  Its exact
            # diagonal gives |E_ij| <= sqrt(E_ii E_jj), hence
            # a'Ea <= (sum_i |a_i| sqrt(E_ii))^2.  Taking the smaller of this
            # and rho*||a||^2 preserves the deterministic spectral guarantee
            # while avoiding a route-global rho penalty for sparse actions.
            diagonal_remainder = float(
                np.dot(
                    np.abs(vector),
                    np.sqrt(self.residual_diagonal_bound),
                )
                ** 2
            )
            remainder = min(spectral_remainder, diagonal_remainder)
        upper = lower + remainder
        return max(lower, 0.0), max(upper, lower)

    def decision(self, action: Sequence[float] | np.ndarray) -> ModelValidityDecision:
        lower, upper = self.action_energy_bounds(action)
        lower_root, upper_root = math.sqrt(lower), math.sqrt(upper)
        if upper_root <= self.spec.reuse_threshold:
            result = "reuse"
            scale = 1.0
            reason = "global trust upper bound lies inside reuse radius"
        elif upper_root <= self.spec.refresh_threshold:
            result = "shrink"
            scale = min(
                1.0,
                self.spec.reuse_threshold / max(upper_root, 1e-15),
            )
            reason = "global trust action lies in shrink annulus"
        else:
            result = "refresh"
            scale = 0.0
            reason = "global trust upper bound exceeds refresh radius"
        residual_can_change = (
            lower_root <= self.spec.reuse_threshold < upper_root
            or lower_root <= self.spec.refresh_threshold < upper_root
        )
        return ModelValidityDecision(
            action=result,
            reason=reason,
            recommended_scale=scale,
            predicted_rms=upper_root,
            diagnostics={
                "lower_response_energy": lower,
                "upper_response_energy": upper,
                "residual_spectral_bound": self.residual_spectral_bound,
                "residual_diagonal_bound_available": (self.residual_diagonal_bound is not None),
                "sparse_action_residual_envelope": (
                    "min_spectral_and_psd_diagonal_cauchy_v1"
                    if self.residual_diagonal_bound is not None
                    else "spectral_only_legacy_v1"
                ),
                "residual_can_change_trust_action": residual_can_change,
                "cross_module_terms_retained": True,
                "trust_factorization_is_not_functional_partitioning": True,
            },
        )

    def requires_rank_growth(self, action: Sequence[float] | np.ndarray) -> bool:
        lower, upper = self.action_energy_bounds(action)
        lower_root, upper_root = math.sqrt(lower), math.sqrt(upper)
        return bool(
            lower_root <= self.spec.reuse_threshold < upper_root
            or lower_root <= self.spec.refresh_threshold < upper_root
        )

    def next_decision_relevant_rank(self) -> int | None:
        """Next deterministic sketch rank, or ``None`` at the resource cap."""

        capacity = int(self.metadata.get("requested_rank", self.rank))
        if capacity >= self.spec.maximum_rank:
            return None
        return min(
            self.spec.maximum_rank,
            max(capacity + 1, 2 * capacity),
        )

    def pack(self) -> tuple[np.ndarray, dict[str, Any]]:
        residual_rows = (
            np.empty((0, len(self.route_ids)), dtype=np.float64)
            if self.residual_diagonal_bound is None
            else self.residual_diagonal_bound[None, :]
        )
        values = np.concatenate((self.mean_component, self.centered_factor, residual_rows), axis=0)
        return np.ascontiguousarray(values), {
            "schema": self.schema,
            "spec": self.spec,
            "route_ids": self.route_ids,
            "mean_rows": int(self.mean_component.shape[0]),
            "factor_rows": int(self.centered_factor.shape[0]),
            "residual_diagonal_rows": int(residual_rows.shape[0]),
            "residual_spectral_bound": self.residual_spectral_bound,
            "total_weight": self.total_weight,
            "context_count": self.context_count,
            "context_axis_hash": self.context_axis_hash,
            "measured_mask_hash": self.measured_mask_hash,
            "calibration_state_id": self.calibration_state_id,
            "source_measurement_hashes": self.source_measurement_hashes,
            "derivation_receipt": self.derivation_receipt,
            "metadata": _deep_thaw_metadata(self.metadata),
        }

    @classmethod
    def unpack(
        cls,
        values: np.ndarray,
        layout: Mapping[str, Any],
        *,
        route_ids: Sequence[str] | None = None,
    ) -> GlobalTrustGeometryState:
        if layout.get("schema") != GLOBAL_TRUST_GEOMETRY_STATE_SCHEMA:
            raise ValueError("measurement is not a global trust geometry")
        array = np.asarray(values, dtype=np.float64)
        mean_rows = int(layout["mean_rows"])
        factor_rows = int(layout["factor_rows"])
        residual_diagonal_rows = int(layout.get("residual_diagonal_rows", 0))
        if residual_diagonal_rows not in {0, 1}:
            raise ValueError("packed trust residual diagonal has an invalid row count")
        if array.ndim != 2 or array.shape[0] != mean_rows + factor_rows + residual_diagonal_rows:
            raise ValueError("packed global trust rows have incompatible shape")
        spec = layout["spec"]
        if not isinstance(spec, GlobalTrustGeometrySpec):
            raise ValueError("packed global trust geometry lost its spec")
        return cls(
            spec=spec,
            route_ids=resolve_packed_string_axis(
                layout,
                expected_axis=route_ids,
                axis_key="route_ids",
                name="global trust route axis",
            ),
            mean_component=array[:mean_rows],
            centered_factor=array[mean_rows : mean_rows + factor_rows],
            residual_spectral_bound=float(layout["residual_spectral_bound"]),
            total_weight=float(layout["total_weight"]),
            context_count=int(layout["context_count"]),
            context_axis_hash=str(layout["context_axis_hash"]),
            measured_mask_hash=str(layout["measured_mask_hash"]),
            calibration_state_id=str(layout["calibration_state_id"]),
            source_measurement_hashes=tuple(layout["source_measurement_hashes"]),
            residual_diagonal_bound=(
                None if residual_diagonal_rows == 0 else array[mean_rows + factor_rows]
            ),
            derivation_receipt=layout.get("derivation_receipt"),
            metadata=dict(layout.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class GlobalTrustRankGrowthObligation:
    """One candidate-bound request to tighten the FD residual envelope."""

    source_geometry_hash: str
    source_state_hash: str
    action_hash: str
    current_rank: int
    requested_rank: int
    maximum_rank: int
    residual_spectral_bound: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "irg_global_trust_rank_growth_obligation_v1"

    def __post_init__(self) -> None:
        if (
            not self.source_geometry_hash
            or not self.source_state_hash
            or not self.action_hash
            or int(self.current_rank) <= 0
            or int(self.requested_rank) <= int(self.current_rank)
            or int(self.requested_rank) > int(self.maximum_rank)
            or not math.isfinite(float(self.residual_spectral_bound))
            or float(self.residual_spectral_bound) < 0.0
        ):
            raise ValueError("global trust rank-growth obligation is invalid")
        object.__setattr__(self, "current_rank", int(self.current_rank))
        object.__setattr__(self, "requested_rank", int(self.requested_rank))
        object.__setattr__(self, "maximum_rank", int(self.maximum_rank))
        object.__setattr__(
            self,
            "residual_spectral_bound",
            float(self.residual_spectral_bound),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-global-trust-rank-growth-")


def register_rank_growth_obligation(
    *,
    cache: dict[str, Any],
    geometry: GlobalTrustGeometryState,
    source_state_hash: str,
    action: Sequence[float] | np.ndarray,
) -> GlobalTrustRankGrowthObligation | None:
    """Register an ambiguous candidate before it can reach adjudication."""

    vector = np.asarray(action, dtype=np.float64).reshape(-1)
    if not geometry.requires_rank_growth(vector):
        return None
    requested = geometry.next_decision_relevant_rank()
    if requested is None:
        return None
    obligation = GlobalTrustRankGrowthObligation(
        source_geometry_hash=geometry.content_hash,
        source_state_hash=str(source_state_hash),
        action_hash=content_hash(vector, prefix="irg-trust-action-"),
        current_rank=int(geometry.metadata.get("requested_rank", geometry.rank)),
        requested_rank=requested,
        maximum_rank=geometry.spec.maximum_rank,
        residual_spectral_bound=geometry.residual_spectral_bound,
        metadata={
            "decision_relevant_residual_crossing": True,
            "same_mask_remeasurement_required": True,
        },
    )
    raw = cache.setdefault(GLOBAL_TRUST_RANK_GROWTH_CACHE_KEY, {})
    if not isinstance(raw, dict):
        raise TypeError("global trust rank-growth cache is invalid")
    raw[obligation.action_hash] = obligation
    return obligation


class FrequentDirectionsTrustAccumulator:
    """Weighted Welford scatter compressed by deterministic Frequent Directions.

    Each observation is a metric-whitened response matrix ``[response,route]``.
    Welford corrections convert the stream into centered scatter rows without
    retaining contexts. Frequent Directions then supplies a full-route factor
    and a pathwise spectral loss envelope.
    """

    def __init__(self, *, route_ids: Sequence[str], sketch_rank: int) -> None:
        self.route_ids = tuple(map(str, route_ids))
        if not self.route_ids or len(set(self.route_ids)) != len(self.route_ids):
            raise ValueError("trust accumulator requires a unique route axis")
        self.sketch_rank = min(int(sketch_rank), len(self.route_ids))
        if self.sketch_rank <= 0:
            raise ValueError("trust sketch rank must be positive")
        self._buffer = np.zeros((2 * self.sketch_rank, len(self.route_ids)), dtype=np.float64)
        self._filled = 0
        self._total_shrinkage = 0.0
        self._mean: np.ndarray | None = None
        self._weighted_second_moment_diagonal = np.zeros(
            len(self.route_ids),
            dtype=np.float64,
        )
        self.total_weight = 0.0
        self.context_count = 0
        self._context_ids: list[str] = []
        self._stream_entry_count = 0
        self._producer_algorithm_id = GLOBAL_TRUST_CANONICAL_FD_ALGORITHM_ID
        self._producer_implementation = "canonical_numpy"
        self._source_row_chain_hash = _empty_ordered_chain(
            schema=_SOURCE_ROW_CHAIN_SCHEMA,
            prefix=GLOBAL_TRUST_SOURCE_ROW_CHAIN_PREFIX,
        )
        self._stream_weight_chain_hash = _empty_ordered_chain(
            schema=_STREAM_WEIGHT_CHAIN_SCHEMA,
            prefix=GLOBAL_TRUST_STREAM_WEIGHT_CHAIN_PREFIX,
        )

    def _compress(self, *, force: bool = False) -> None:
        if self._filled <= self.sketch_rank and not force:
            return
        if self._filled == 0:
            return
        rows = self._buffer[: self._filled]
        _left, singular, right = np.linalg.svd(rows, full_matrices=False)
        retain = min(self.sketch_rank, singular.size)
        delta = float(singular[self.sketch_rank] ** 2) if singular.size > self.sketch_rank else 0.0
        shrunk = np.sqrt(np.maximum(singular[:retain] ** 2 - delta, 0.0))
        self._buffer.fill(0.0)
        self._buffer[:retain] = shrunk[:, None] * right[:retain]
        self._filled = int(np.sum(shrunk > 0.0))
        self._total_shrinkage += delta

    def _insert(self, rows: np.ndarray) -> None:
        """Insert deterministic blocks without changing FD boundaries.

        A full buffer is compressed only when another logical row is waiting
        to enter, exactly matching the historical row-wise schedule while
        removing one Python loop iteration per response row.
        """

        values = np.asarray(rows, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self._buffer.shape[1]:
            raise ValueError("trust FD insertion requires a [row,route] matrix")
        offset = 0
        while offset < values.shape[0]:
            if self._filled >= self._buffer.shape[0]:
                self._compress()
            room = self._buffer.shape[0] - self._filled
            take = min(room, values.shape[0] - offset)
            self._buffer[self._filled : self._filled + take] = values[offset : offset + take]
            self._filled += take
            offset += take

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
            values.ndim != 3
            or values.shape[0] != len(ids)
            or values.shape[2] != len(self.route_ids)
            or not ids
            or len(set(ids)) != len(ids)
            or set(ids).intersection(self._context_ids)
            or not np.isfinite(values).all()
        ):
            raise ValueError("trust rows require unique contexts and [context,response,route] axes")
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
            raise ValueError("trust weights are invalid")
        for observation, weight in zip(values, raw_weights, strict=True):
            w = float(weight)
            entry_index = self._stream_entry_count
            self._source_row_chain_hash = _advance_ordered_chain(
                self._source_row_chain_hash,
                index=entry_index,
                value=global_trust_source_row_hash(observation),
                schema=_SOURCE_ROW_CHAIN_SCHEMA,
                prefix=GLOBAL_TRUST_SOURCE_ROW_CHAIN_PREFIX,
            )
            self._stream_weight_chain_hash = _advance_ordered_chain(
                self._stream_weight_chain_hash,
                index=entry_index,
                value=w,
                schema=_STREAM_WEIGHT_CHAIN_SCHEMA,
                prefix=GLOBAL_TRUST_STREAM_WEIGHT_CHAIN_PREFIX,
            )
            self._stream_entry_count += 1
            if w <= 0.0:
                continue
            self._weighted_second_moment_diagonal += w * np.sum(
                observation * observation,
                axis=0,
            )
            if self._mean is None:
                self._mean = np.array(observation, copy=True)
                self.total_weight = w
            else:
                delta = observation - self._mean
                updated_weight = self.total_weight + w
                correction = math.sqrt(self.total_weight * w / updated_weight) * delta
                self._insert(correction)
                self._mean += (w / updated_weight) * delta
                self.total_weight = updated_weight
            self.context_count += 1
        self._context_ids.extend(ids)

    def finalize(
        self,
        *,
        spec: GlobalTrustGeometrySpec,
        measured_mask_hash: str,
        calibration_state_id: str,
        source_measurement_hashes: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> GlobalTrustGeometryState:
        if self._mean is None or self.context_count <= 0 or self.total_weight <= 0.0:
            raise ValueError("cannot finalize an empty global trust geometry")
        self._compress(force=True)
        factor = self._buffer[: self._filled] / math.sqrt(self.total_weight)
        centered_diagonal = np.maximum(
            self._weighted_second_moment_diagonal / self.total_weight
            - np.sum(self._mean * self._mean, axis=0),
            0.0,
        )
        residual_diagonal = np.maximum(
            centered_diagonal - np.sum(factor * factor, axis=0),
            0.0,
        )
        caller_metadata = dict(metadata or {})
        protected = _PROTECTED_ACCUMULATOR_METADATA_KEYS.intersection(caller_metadata)
        if protected:
            raise ValueError(
                "global trust accumulator metadata cannot override protected "
                "producer fields: " + ", ".join(sorted(protected))
            )
        measurement_hashes = tuple(map(str, source_measurement_hashes))
        route_axis_hash = ordered_string_axis_hash(
            self.route_ids,
            name="global trust route axis",
        )
        context_axis_hash = content_hash(
            tuple(self._context_ids),
            prefix="irg-trust-context-axis-",
        )
        derivation_receipt = GlobalTrustGeometryDerivationReceipt(
            producer_algorithm_id=self._producer_algorithm_id,
            producer_implementation=self._producer_implementation,
            trust_spec_hash=spec.content_hash,
            ordered_route_axis_hash=route_axis_hash,
            context_axis_hash=context_axis_hash,
            stream_entry_count=self._stream_entry_count,
            effective_context_count=self.context_count,
            ordered_source_rows_hash=_finalize_ordered_chain(
                self._source_row_chain_hash,
                entry_count=self._stream_entry_count,
                schema=_SOURCE_ROW_CHAIN_SCHEMA,
                prefix=GLOBAL_TRUST_SOURCE_ROW_CHAIN_PREFIX,
            ),
            stream_weight_hash=_finalize_ordered_chain(
                self._stream_weight_chain_hash,
                entry_count=self._stream_entry_count,
                schema=_STREAM_WEIGHT_CHAIN_SCHEMA,
                prefix=GLOBAL_TRUST_STREAM_WEIGHT_CHAIN_PREFIX,
            ),
            mean_component_hash=content_hash(
                self._mean,
                prefix="irg-global-trust-derived-mean-",
            ),
            centered_factor_hash=content_hash(
                factor,
                prefix="irg-global-trust-derived-centered-factor-",
            ),
            exact_residual_diagonal_hash=content_hash(
                residual_diagonal,
                prefix="irg-global-trust-derived-residual-diagonal-",
            ),
            total_weight=self.total_weight,
            cumulative_fd_shrink=self._total_shrinkage,
            normalized_stream_squared_frobenius=float(
                np.sum(self._weighted_second_moment_diagonal, dtype=np.float64) / self.total_weight
            ),
            sketch_rank=self.sketch_rank,
            measured_mask_hash=str(measured_mask_hash),
            calibration_state_id=str(calibration_state_id),
            source_measurement_hashes=measurement_hashes,
            _accumulator_token=_ACCUMULATOR_DERIVATION_TOKEN,
        )
        return GlobalTrustGeometryState(
            spec=spec,
            route_ids=self.route_ids,
            mean_component=self._mean,
            centered_factor=factor,
            residual_spectral_bound=(self._total_shrinkage / self.total_weight),
            residual_diagonal_bound=residual_diagonal,
            total_weight=self.total_weight,
            context_count=self.context_count,
            context_axis_hash=context_axis_hash,
            measured_mask_hash=str(measured_mask_hash),
            calibration_state_id=str(calibration_state_id),
            source_measurement_hashes=measurement_hashes,
            derivation_receipt=derivation_receipt,
            metadata={
                **caller_metadata,
                "sketch_algorithm": self._producer_algorithm_id,
                "cross_module_terms_retained": True,
                "spectral_residual_is_pathwise_upper_envelope": True,
                "residual_diagonal_is_exact_streamed_fd_remainder": True,
            },
        )


class GramFrequentDirectionsTrustAccumulator(FrequentDirectionsTrustAccumulator):
    """Block-inserted FD using a small ``2r x 2r`` Gram eigendecomposition.

    The canonical accumulator computes an SVD of the wide ``2r x p`` buffer.
    This variant computes ``rows @ rows.T``, diagonalizes only that small Gram
    matrix, and reconstructs the shrunk right-space rows.  It preserves the
    same weighted-Welford stream, compression schedule, FD shrink rule, source
    chains, residual diagonal, and pathwise spectral envelope.

    Raw factors are not required to be byte-identical because eigenspaces with
    repeated singular values are basis-nonunique; covariance/action invariants
    are the cross-implementation comparison surface.
    """

    def __init__(self, *, route_ids: Sequence[str], sketch_rank: int) -> None:
        super().__init__(route_ids=route_ids, sketch_rank=sketch_rank)
        self._producer_algorithm_id = GLOBAL_TRUST_GRAM_FD_ALGORITHM_ID
        self._producer_implementation = "gram_numpy"

    def _compress(self, *, force: bool = False) -> None:
        if self._filled <= self.sketch_rank and not force:
            return
        if self._filled == 0:
            return
        rows = self._buffer[: self._filled]
        gram = rows @ rows.T
        gram = 0.5 * (gram + gram.T)
        eigenvalues, left = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues, kind="stable")[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        left = left[:, order]
        singular = np.sqrt(eigenvalues)
        retain = min(self.sketch_rank, singular.size)
        delta = (
            float(singular[self.sketch_rank] ** 2)
            if singular.size > self.sketch_rank
            else 0.0
        )
        shrunk = np.sqrt(np.maximum(singular[:retain] ** 2 - delta, 0.0))
        projected = left[:, :retain].T @ rows
        gains = np.divide(
            shrunk,
            singular[:retain],
            out=np.zeros_like(shrunk),
            where=singular[:retain] > 0.0,
        )
        replacement = gains[:, None] * projected
        self._buffer.fill(0.0)
        self._buffer[:retain] = replacement
        self._filled = int(np.sum(shrunk > 0.0))
        self._total_shrinkage += delta

    def _insert(self, rows: np.ndarray) -> None:
        values = np.asarray(rows, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.route_ids):
            raise ValueError("Gram FD insertion rows have the wrong route axis")
        cursor = 0
        while cursor < values.shape[0]:
            if self._filled >= self._buffer.shape[0]:
                self._compress()
            capacity = self._buffer.shape[0] - self._filled
            take = min(capacity, values.shape[0] - cursor)
            self._buffer[self._filled : self._filled + take] = values[cursor : cursor + take]
            self._filled += take
            cursor += take



__all__ = [
    "FrequentDirectionsTrustAccumulator",
    "GramFrequentDirectionsTrustAccumulator",
    "GLOBAL_TRUST_RANK_GROWTH_CACHE_KEY",
    "GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY",
    "GLOBAL_TRUST_CANONICAL_FD_ALGORITHM_ID",
    "GLOBAL_TRUST_CUDA_FD_ALGORITHM_ID",
    "GLOBAL_TRUST_GRAM_FD_ALGORITHM_ID",
    "GLOBAL_TRUST_CUDA_GRAM_FD_ALGORITHM_ID",
    "GLOBAL_TRUST_DERIVATION_RECEIPT_SCHEMA",
    "GLOBAL_TRUST_GEOMETRY_STATE_SCHEMA",
    "GlobalTrustGeometryState",
    "GlobalTrustGeometryDerivationReceipt",
    "GlobalTrustRankGrowthObligation",
    "global_trust_ordered_source_rows_hash",
    "global_trust_source_row_hash",
    "global_trust_stream_weight_hash",
    "register_rank_growth_obligation",
]
