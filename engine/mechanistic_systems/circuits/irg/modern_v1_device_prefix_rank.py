"""Compact prefix-rank custody for adaptive signed-work endpoint queries.

The uint8 device-shell labels are excellent for a bounded batch of endpoints,
but they are not the right representation for the whole canonical prefix path:
there may be far more than 255 possible prefix counts.  This module keeps the
pre-C path as one route-scale rank vector and derives uint8 first-inclusion
labels only for the endpoint counts that a later C-stage query actually needs.

The rank vector is still authority-inert.  It certifies no F/Q/R value, no
calibration, no oracle result, and no mask commit.  Its only job is causal
custody: post-C endpoint-count selection must not change the D-frozen route
order. Logical ranks are uint32. Production CUDA stores the current 57M-route
domain in torch.int32—the smallest broadly supported CUDA ordering dtype—and
fails closed rather than widening silently if that signed range is exceeded.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import InitVar, dataclass
from enum import Enum
import hashlib
import math
from numbers import Integral
from threading import RLock
from typing import Any, Final, Literal, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import canonical_json, content_hash
from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .modern_v1_device_shell_ladder import (
    DEVICE_SHELL_MAXIMUM_ENDPOINTS,
    V1DeviceShellLabelHandle,
)

try:  # pragma: no cover - availability is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


DEVICE_PREFIX_RANK_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_device_prefix_rank_receipt_v2"
)
DEVICE_PREFIX_RANK_QUERY_SCHEMA: Final = (
    "partition_free_v1_device_prefix_rank_endpoint_query_v2"
)
DEVICE_PREFIX_RANK_ALGORITHM_ID: Final = (
    "pre_C_logical_uint32_route_rank_path_to_uint8_endpoint_labels_v2"
)
DEVICE_PREFIX_RANK_AUTHORITY: Final = (
    "prefix_path_custody_and_endpoint_label_derivation_only_no_FQR_CKO_commit_"
    "terminal_or_pareto_v1"
)

_RECEIPT_PREFIX: Final = "partition-free-v1-device-prefix-rank-receipt-"
_QUERY_PREFIX: Final = "partition-free-v1-device-prefix-rank-query-"
_RANK_COMMITMENT_DOMAIN: Final = "partition_free_v1_device_prefix_rank_vector_v1"
_REQUIRED_COMMITMENT_DOMAIN: Final = "partition_free_v1_device_prefix_required_v1"

DEVICE_PREFIX_RANK_LOGICAL_DTYPE: Final = "uint32"
DEVICE_PREFIX_RANK_CUDA_STORAGE_DTYPE: Final = "torch.int32"
DEVICE_PREFIX_RANK_DEFAULT_STREAM_CHUNK_SIZE: Final = 1 << 20
DEVICE_PREFIX_RANK_DEFAULT_HBM_CEILING_FRACTION: Final = 0.85
DEVICE_PREFIX_RANK_CUDA_HBM_SCHEMA: Final = (
    "partition_free_v1_device_prefix_rank_cuda_hbm_forecast_v1"
)
_CUDA_HBM_PREFIX: Final = "partition-free-v1-device-prefix-rank-cuda-hbm-"
_I32_BYTES: Final = np.dtype(np.int32).itemsize
_I64_BYTES: Final = np.dtype(np.int64).itemsize
_U8_BYTES: Final = np.dtype(np.uint8).itemsize
_BOOL_BYTES: Final = np.dtype(np.bool_).itemsize
_CUDA_ALLOCATOR_GUARD_BYTES: Final = 4 << 20

DevicePrefixRankHandleKind = Literal["numpy_reference", "torch_cuda"]


class V1DevicePrefixRankRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_RANK = "invalid_rank"
    INVALID_REQUIRED = "invalid_required"
    ENDPOINT_LIMIT_EXCEEDED = "endpoint_limit_exceeded"
    ENDPOINT_OUT_OF_RANGE = "endpoint_out_of_range"
    HANDLE_EVICTED = "handle_evicted"
    HANDLE_HAS_ACTIVE_LEASES = "handle_has_active_leases"
    RANK_STORAGE_MUTATED = "rank_storage_mutated"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    AUTHORITY_VIOLATION = "authority_violation"


class V1DevicePrefixRankRefusal(RuntimeError):
    """Typed fail-closed outcome for prefix-rank custody."""

    def __init__(self, code: V1DevicePrefixRankRefusalCode, message: str) -> None:
        if type(code) is not V1DevicePrefixRankRefusalCode:
            raise TypeError("prefix-rank refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _concrete_device_identity(device: Any) -> str:
    """``str(device)`` with a bare ``cuda`` bound to its concrete index.

    Label handles report ``str(tensor.device)`` (``cuda:0``); forecasts and
    receipts must name the same string or custody comparisons refuse.
    """

    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
        resolved = torch.device("cuda", int(torch.cuda.current_device()))
    return str(resolved)


def _refuse(code: V1DevicePrefixRankRefusalCode, message: str) -> NoReturn:
    raise V1DevicePrefixRankRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_TYPE,
            f"{name} must be one finite real number",
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result > 0.85:
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_TYPE,
            f"{name} must lie in (0, 0.85]",
        )
    return result


def _stream_digest(*, domain: str, metadata: object) -> "hashlib._Hash":
    encoded_domain = domain.encode("ascii")
    encoded_metadata = canonical_json(metadata).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(encoded_domain).to_bytes(8, "big", signed=False))
    digest.update(encoded_domain)
    digest.update(len(encoded_metadata).to_bytes(8, "big", signed=False))
    digest.update(encoded_metadata)
    return digest


def _normalize_required(required: Sequence[bool] | np.ndarray, *, route_count: int) -> np.ndarray:
    raw = np.asarray(required)
    if raw.dtype != np.dtype(np.bool_) or raw.shape != (route_count,):
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_REQUIRED,
            "required mask must be a boolean vector on the route axis",
        )
    result = np.array(raw, dtype=np.bool_, order="C", copy=True)
    result.setflags(write=False)
    return result


def _validate_rank_array(ranks: np.ndarray, *, required: np.ndarray) -> np.ndarray:
    if ranks.ndim != 1:
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_RANK,
            "prefix ranks must be one-dimensional",
        )
    if ranks.dtype not in (np.dtype(np.uint32), np.dtype(np.uint64)):
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_RANK,
            "prefix ranks must be uint32 or uint64",
        )
    if ranks.dtype == np.dtype(np.uint64) and int(np.max(ranks, initial=0)) > int(
        np.iinfo(np.uint32).max
    ):
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_RANK,
            "prefix ranks exceed the registered uint32 custody range",
        )
    result = np.array(ranks, dtype=np.uint32, order="C", copy=True)
    if result.size != required.size:
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_RANK,
            "prefix ranks must align with the required mask",
        )
    if bool(np.any(result[required] != 0)):
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_RANK,
            "required routes are carried separately and must have rank zero",
        )
    positive = result[result > 0]
    if positive.size:
        ordered = np.sort(positive)
        expected = np.arange(1, positive.size + 1, dtype=np.uint32)
        if not np.array_equal(ordered, expected):
            _refuse(
                V1DevicePrefixRankRefusalCode.INVALID_RANK,
                "positive prefix ranks must be a dense 1..K permutation",
            )
    result.setflags(write=False)
    return result


def _rank_storage_invariant(
    value: object,
    *,
    backend_kind: DevicePrefixRankHandleKind,
) -> tuple[object, ...]:
    if backend_kind == "numpy_reference":
        array = np.asarray(value)
        return (
            "numpy_reference",
            int(array.__array_interface__["data"][0]),
            array.shape,
            array.strides,
            array.dtype.str,
            bool(array.flags.c_contiguous),
            bool(array.flags.writeable),
        )
    tensor = value
    return (
        "torch_cuda",
        int(tensor.data_ptr()),  # type: ignore[union-attr]
        int(tensor.untyped_storage().data_ptr()),  # type: ignore[union-attr]
        int(tensor.storage_offset()),  # type: ignore[union-attr]
        tuple(int(item) for item in tensor.shape),  # type: ignore[union-attr]
        tuple(int(item) for item in tensor.stride()),  # type: ignore[union-attr]
        str(tensor.dtype),  # type: ignore[union-attr]
        str(tensor.device),  # type: ignore[union-attr]
        int(tensor._version),  # type: ignore[union-attr]
    )


@dataclass(frozen=True, slots=True)
class V1DevicePrefixRankCUDAHBMForecast:
    """Pre-outcome CUDA working-set bound for rank validation or a query.

    The reserved-byte forecast retains its raw allocator telemetry.  Admission
    charges live allocation and can reuse cached reservation; incremental terms
    cover only allocations introduced by this operation.  CUDA rank storage is
    deliberately reported as signed int32: the mathematical ranks remain
    uint32, but PyTorch's broadly supported CUDA comparison/indexing dtype is
    int32.  Custody fails closed before the logical rank could exceed
    ``INT32_MAX``.
    """

    operation: Literal["freeze_validation", "endpoint_query"]
    device_identity: str
    route_count: int
    endpoint_count: int
    stream_chunk_size: int
    logical_rank_dtype: str
    cuda_rank_storage_dtype: str
    resident_required_bytes: int
    resident_seen_bytes: int
    resident_label_bytes: int
    endpoint_workspace_bytes: int
    slab_workspace_bytes: int
    allocator_guard_bytes: int
    forecast_incremental_bytes: int
    current_reserved_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_fraction: float
    hbm_ceiling_bytes: int
    admitted: bool
    schema: str = DEVICE_PREFIX_RANK_CUDA_HBM_SCHEMA
    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _admission_memory: tuple[int, int] | None) -> None:
        if self.operation not in {"freeze_validation", "endpoint_query"}:
            raise ValueError("unsupported prefix-rank CUDA forecast operation")
        _identity(self.device_identity, name="device identity")
        _positive_int(self.route_count, name="route count")
        _nonnegative_int(self.endpoint_count, name="endpoint count")
        _positive_int(self.stream_chunk_size, name="stream chunk size")
        if (
            self.logical_rank_dtype != DEVICE_PREFIX_RANK_LOGICAL_DTYPE
            or self.cuda_rank_storage_dtype
            != DEVICE_PREFIX_RANK_CUDA_STORAGE_DTYPE
            or self.schema != DEVICE_PREFIX_RANK_CUDA_HBM_SCHEMA
        ):
            _refuse(
                V1DevicePrefixRankRefusalCode.AUTHORITY_VIOLATION,
                "CUDA rank forecast changed the registered dtype or schema",
            )
        for name in (
            "resident_required_bytes",
            "resident_seen_bytes",
            "resident_label_bytes",
            "endpoint_workspace_bytes",
            "slab_workspace_bytes",
            "allocator_guard_bytes",
            "forecast_incremental_bytes",
            "current_reserved_bytes",
            "forecast_peak_reserved_bytes",
            "device_total_memory_bytes",
            "hbm_ceiling_bytes",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        fraction = _fraction(
            self.hbm_ceiling_fraction,
            name="HBM ceiling fraction",
        )
        if self.device_total_memory_bytes <= 0:
            raise ValueError("device total memory must be positive")
        expected_incremental = (
            self.resident_required_bytes
            + self.resident_seen_bytes
            + self.resident_label_bytes
            + self.endpoint_workspace_bytes
            + self.slab_workspace_bytes
            + self.allocator_guard_bytes
        )
        if self.forecast_incremental_bytes != expected_incremental:
            raise ValueError("CUDA rank forecast incremental bytes do not replay")
        if (
            self.forecast_peak_reserved_bytes
            != self.current_reserved_bytes + expected_incremental
        ):
            raise ValueError("CUDA rank forecast peak bytes do not replay")
        if self.hbm_ceiling_bytes != int(
            math.floor(fraction * self.device_total_memory_bytes)
        ):
            raise ValueError("CUDA rank forecast ceiling bytes do not replay")
        if _admission_memory is None:
            admitted = self.forecast_peak_reserved_bytes <= self.hbm_ceiling_bytes
        else:
            available, allocated = _admission_memory
            admitted = (
                allocated + expected_incremental <= self.hbm_ceiling_bytes
                and expected_incremental <= available
            )
        if self.admitted != admitted:
            raise ValueError("CUDA rank forecast admission does not replay")

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_CUDA_HBM_PREFIX)


def _cuda_memory_limits(
    *,
    device: object,
    device_total_memory_bytes: int | None,
    current_reserved_bytes: int | None,
) -> tuple[int, int, tuple[int, int] | None]:
    if torch is None:  # pragma: no cover - guarded by CUDA construction.
        raise RuntimeError("CUDA prefix-rank forecast requires PyTorch")
    live_memory = device_total_memory_bytes is None or current_reserved_bytes is None
    if device_total_memory_bytes is None:
        device_total_memory_bytes = int(
            torch.cuda.get_device_properties(device).total_memory
        )
    if current_reserved_bytes is None:
        current_reserved_bytes = int(torch.cuda.memory_reserved(device))
    return (
        _positive_int(device_total_memory_bytes, name="device total memory"),
        _nonnegative_int(current_reserved_bytes, name="current reserved bytes"),
        (
            cuda_reusable_memory_bytes(device, framework=torch),
            int(torch.cuda.memory_allocated(device)),
        ) if live_memory else None,
    )


def _cuda_hbm_forecast(
    *,
    operation: Literal["freeze_validation", "endpoint_query"],
    device: object,
    route_count: int,
    endpoint_count: int,
    stream_chunk_size: int,
    hbm_ceiling_fraction: float,
    device_total_memory_bytes: int | None,
    current_reserved_bytes: int | None,
) -> V1DevicePrefixRankCUDAHBMForecast:
    if torch is None:  # pragma: no cover - guarded by CUDA construction.
        raise RuntimeError("CUDA prefix-rank forecast requires PyTorch")
    chunk = min(
        _positive_int(stream_chunk_size, name="stream chunk size"),
        _positive_int(route_count, name="route count"),
    )
    endpoint_total = _nonnegative_int(endpoint_count, name="endpoint count")
    fraction = _fraction(hbm_ceiling_fraction, name="HBM ceiling fraction")
    total, current, admission_memory = _cuda_memory_limits(
        device=device,
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
    )
    if operation == "freeze_validation":
        # One owned required mask remains resident.  The worst-case seen table
        # has one byte for every possible positive int32 rank plus rank zero.
        # Per-slab allowance covers positive compaction, int64 CUDA indices,
        # indexed reads/writes, masks, and allocator/sort-free temporary copies.
        resident_required = route_count * _BOOL_BYTES
        resident_seen = (route_count + 1) * _BOOL_BYTES
        resident_labels = 0
        endpoint_workspace = 0
        slab_workspace = chunk * (
            2 * _I32_BYTES + 2 * _I64_BYTES + 8 * _BOOL_BYTES
        )
    elif operation == "endpoint_query":
        resident_required = 0
        resident_seen = 0
        resident_labels = route_count * _U8_BYTES
        endpoint_workspace = endpoint_total * _I32_BYTES
        # searchsorted(out_int32=True), masks, masked casts, and assignment
        # intermediates all remain bounded by one physical-axis slab.
        slab_workspace = chunk * (
            3 * _I32_BYTES + 8 * _BOOL_BYTES + 4 * _U8_BYTES
        )
    else:  # pragma: no cover - Literal plus exact internal call sites.
        raise ValueError("unsupported CUDA prefix-rank forecast operation")
    incremental = (
        resident_required
        + resident_seen
        + resident_labels
        + endpoint_workspace
        + slab_workspace
        + _CUDA_ALLOCATOR_GUARD_BYTES
    )
    ceiling = int(math.floor(fraction * total))
    peak = current + incremental
    if admission_memory is None:
        admitted = peak <= ceiling
    else:
        available, allocated = admission_memory
        admitted = allocated + incremental <= ceiling and incremental <= available
    return V1DevicePrefixRankCUDAHBMForecast(
        operation=operation,
        device_identity=_concrete_device_identity(device),
        route_count=route_count,
        endpoint_count=endpoint_total,
        stream_chunk_size=chunk,
        logical_rank_dtype=DEVICE_PREFIX_RANK_LOGICAL_DTYPE,
        cuda_rank_storage_dtype=DEVICE_PREFIX_RANK_CUDA_STORAGE_DTYPE,
        resident_required_bytes=resident_required,
        resident_seen_bytes=resident_seen,
        resident_label_bytes=resident_labels,
        endpoint_workspace_bytes=endpoint_workspace,
        slab_workspace_bytes=slab_workspace,
        allocator_guard_bytes=_CUDA_ALLOCATOR_GUARD_BYTES,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        hbm_ceiling_fraction=fraction,
        hbm_ceiling_bytes=ceiling,
        admitted=admitted,
        _admission_memory=admission_memory,
    )


@dataclass(frozen=True, slots=True)
class V1DevicePrefixRankReceipt:
    """Array-free identity for one complete D-frozen prefix order."""

    route_axis_hash: str
    route_count: int
    source_rounding_hash: str
    rank_commitment: str
    required_commitment: str
    required_total_count: int
    maximum_optional_rank: int
    handle_kind: DevicePrefixRankHandleKind
    device_identity: str
    logical_rank_dtype: str
    storage_rank_dtype: str
    cuda_validation_hbm_forecast: V1DevicePrefixRankCUDAHBMForecast | None
    full_prefix_family_frozen_pre_c: bool = True
    endpoint_labels_materialized: bool = False
    c_may_select_counts_without_reordering: bool = True
    algorithm_id: str = DEVICE_PREFIX_RANK_ALGORITHM_ID
    authority_scope: str = DEVICE_PREFIX_RANK_AUTHORITY
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = DEVICE_PREFIX_RANK_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "route_axis_hash",
            "source_rounding_hash",
            "rank_commitment",
            "required_commitment",
            "device_identity",
            "algorithm_id",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        for name in ("required_total_count", "maximum_optional_rank"):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if self.required_total_count + self.maximum_optional_rank > self.route_count:
            _refuse(
                V1DevicePrefixRankRefusalCode.INVALID_RANK,
                "required plus optional prefix ranks exceed the route axis",
            )
        if self.handle_kind not in {"numpy_reference", "torch_cuda"}:
            _refuse(
                V1DevicePrefixRankRefusalCode.INVALID_TYPE,
                "unsupported prefix-rank handle kind",
            )
        if self.logical_rank_dtype != DEVICE_PREFIX_RANK_LOGICAL_DTYPE:
            _refuse(
                V1DevicePrefixRankRefusalCode.AUTHORITY_VIOLATION,
                "prefix-rank receipt changed the logical uint32 domain",
            )
        expected_storage_dtype = (
            "numpy.uint32"
            if self.handle_kind == "numpy_reference"
            else DEVICE_PREFIX_RANK_CUDA_STORAGE_DTYPE
        )
        if self.storage_rank_dtype != expected_storage_dtype:
            _refuse(
                V1DevicePrefixRankRefusalCode.AUTHORITY_VIOLATION,
                "prefix-rank receipt does not disclose its exact storage dtype",
            )
        if self.handle_kind == "numpy_reference":
            if self.cuda_validation_hbm_forecast is not None:
                raise ValueError("CPU rank custody cannot carry a CUDA HBM forecast")
        else:
            if type(self.cuda_validation_hbm_forecast) is not (
                V1DevicePrefixRankCUDAHBMForecast
            ):
                raise TypeError("CUDA rank custody requires its exact HBM forecast")
            forecast = self.cuda_validation_hbm_forecast
            if (
                forecast.operation != "freeze_validation"
                or forecast.device_identity != self.device_identity
                or forecast.route_count != self.route_count
                or not forecast.admitted
            ):
                _refuse(
                    V1DevicePrefixRankRefusalCode.AUTHORITY_VIOLATION,
                    "CUDA validation forecast does not bind this rank receipt",
                )
        flags = (
            self.full_prefix_family_frozen_pre_c,
            self.endpoint_labels_materialized,
            self.c_may_select_counts_without_reordering,
            self.fqr_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(item) is not bool for item in flags):
            raise TypeError("prefix-rank receipt flags must be exact booleans")
        if flags != (True, False, True, False, False, False, False, False, False):
            _refuse(
                V1DevicePrefixRankRefusalCode.AUTHORITY_VIOLATION,
                "prefix-rank receipt crossed its custody-only authority",
            )
        if (
            self.algorithm_id != DEVICE_PREFIX_RANK_ALGORITHM_ID
            or self.authority_scope != DEVICE_PREFIX_RANK_AUTHORITY
            or self.schema != DEVICE_PREFIX_RANK_RECEIPT_SCHEMA
        ):
            _refuse(
                V1DevicePrefixRankRefusalCode.AUTHORITY_VIOLATION,
                "prefix-rank receipt semantics changed",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_RECEIPT_PREFIX)


@dataclass(frozen=True, slots=True)
class V1DevicePrefixRankEndpointQueryReceipt:
    """One bounded endpoint-label query derived from a frozen rank path."""

    prefix_rank_receipt_hash: str
    route_axis_hash: str
    route_count: int
    endpoint_counts: tuple[int, ...]
    endpoint_count: int
    required_total_count: int
    maximum_optional_rank: int
    resulting_label_commitment: str
    endpoint_query_conditioned_on_c_outcomes: bool
    cuda_query_hbm_forecast: V1DevicePrefixRankCUDAHBMForecast | None = None
    composition_recomputed_or_reordered: bool = False
    endpoint_labels_materialized_for_this_query: bool = True
    endpoint_limit: int = DEVICE_SHELL_MAXIMUM_ENDPOINTS
    algorithm_id: str = DEVICE_PREFIX_RANK_ALGORITHM_ID
    authority_scope: str = DEVICE_PREFIX_RANK_AUTHORITY
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = DEVICE_PREFIX_RANK_QUERY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "prefix_rank_receipt_hash",
            "route_axis_hash",
            "resulting_label_commitment",
            "algorithm_id",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        counts = tuple(
            _positive_int(value, name="endpoint count")
            for value in self.endpoint_counts
        )
        if not counts or tuple(sorted(set(counts))) != counts:
            _refuse(
                V1DevicePrefixRankRefusalCode.ENDPOINT_OUT_OF_RANGE,
                "endpoint counts must be nonempty, unique, and increasing",
            )
        object.__setattr__(self, "endpoint_counts", counts)
        object.__setattr__(
            self,
            "endpoint_count",
            _positive_int(self.endpoint_count, name="endpoint count length"),
        )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        object.__setattr__(
            self,
            "required_total_count",
            _nonnegative_int(self.required_total_count, name="required total count"),
        )
        object.__setattr__(
            self,
            "maximum_optional_rank",
            _nonnegative_int(self.maximum_optional_rank, name="maximum optional rank"),
        )
        object.__setattr__(
            self,
            "endpoint_limit",
            _positive_int(self.endpoint_limit, name="endpoint limit"),
        )
        if self.endpoint_count != len(counts):
            raise ValueError("endpoint_count does not replay from endpoint_counts")
        if self.endpoint_count > self.endpoint_limit:
            _refuse(
                V1DevicePrefixRankRefusalCode.ENDPOINT_LIMIT_EXCEEDED,
                "one prefix-rank query cannot exceed the uint8 endpoint-label limit",
            )
        if (
            counts[0] <= self.required_total_count
            or counts[-1] > self.required_total_count + self.maximum_optional_rank
        ):
            _refuse(
                V1DevicePrefixRankRefusalCode.ENDPOINT_OUT_OF_RANGE,
                "endpoint counts fall outside the frozen prefix rank path",
            )
        flags = (
            self.endpoint_query_conditioned_on_c_outcomes,
            self.composition_recomputed_or_reordered,
            self.endpoint_labels_materialized_for_this_query,
            self.fqr_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(item) is not bool for item in flags):
            raise TypeError("prefix-rank query flags must be exact booleans")
        if flags[1:] != (False, True, False, False, False, False, False, False):
            _refuse(
                V1DevicePrefixRankRefusalCode.AUTHORITY_VIOLATION,
                "prefix-rank query changed composition or crossed authority",
            )
        if self.cuda_query_hbm_forecast is not None:
            if type(self.cuda_query_hbm_forecast) is not (
                V1DevicePrefixRankCUDAHBMForecast
            ):
                raise TypeError("CUDA endpoint query requires its exact HBM forecast")
            forecast = self.cuda_query_hbm_forecast
            if (
                forecast.operation != "endpoint_query"
                or forecast.route_count != self.route_count
                or forecast.endpoint_count != self.endpoint_count
                or not forecast.admitted
            ):
                _refuse(
                    V1DevicePrefixRankRefusalCode.AUTHORITY_VIOLATION,
                    "CUDA query forecast does not bind this endpoint query",
                )
        if (
            self.algorithm_id != DEVICE_PREFIX_RANK_ALGORITHM_ID
            or self.authority_scope != DEVICE_PREFIX_RANK_AUTHORITY
            or self.schema != DEVICE_PREFIX_RANK_QUERY_SCHEMA
        ):
            _refuse(
                V1DevicePrefixRankRefusalCode.AUTHORITY_VIOLATION,
                "prefix-rank query semantics changed",
            )

    @property
    def query_hash(self) -> str:
        return content_hash(self, prefix=_QUERY_PREFIX)


class V1DevicePrefixRankHandle:
    """Ephemeral owner for one route-scale prefix-rank vector."""

    __slots__ = (
        "_active_lease_tokens",
        "_backend_kind",
        "_device_identity",
        "_evicted",
        "_rank_storage_invariant",
        "_ranks",
        "_receipt",
        "_required",
        "_required_storage_invariant",
        "_lease_lock",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        ranks: object,
        required: object,
        receipt: V1DevicePrefixRankReceipt,
        backend_kind: DevicePrefixRankHandleKind,
        device_identity: str,
    ) -> None:
        if type(receipt) is not V1DevicePrefixRankReceipt:
            raise TypeError("prefix-rank handle requires its exact receipt")
        receipt.receipt_hash
        self._ranks = ranks
        self._required = required
        self._receipt = receipt
        self._backend_kind = backend_kind
        self._device_identity = device_identity
        self._rank_storage_invariant = _rank_storage_invariant(
            ranks,
            backend_kind=backend_kind,
        )
        self._required_storage_invariant = _rank_storage_invariant(
            required,
            backend_kind=backend_kind,
        )
        self._active_lease_tokens: set[object] = set()
        self._lease_lock = RLock()
        self._evicted = False

    @classmethod
    def from_numpy_reference(
        cls,
        *,
        ranks: Sequence[int] | np.ndarray,
        required: Sequence[bool] | np.ndarray,
        route_axis_hash: str,
        source_rounding_hash: str,
    ) -> "V1DevicePrefixRankHandle":
        raw = np.asarray(ranks)
        route_count = int(raw.size)
        required_array = _normalize_required(required, route_count=route_count)
        rank_array = _validate_rank_array(raw, required=required_array)
        rank_commitment = _rank_commitment_numpy(rank_array)
        required_commitment = _required_commitment_numpy(required_array)
        receipt = V1DevicePrefixRankReceipt(
            route_axis_hash=route_axis_hash,
            route_count=route_count,
            source_rounding_hash=source_rounding_hash,
            rank_commitment=rank_commitment,
            required_commitment=required_commitment,
            required_total_count=int(np.count_nonzero(required_array)),
            maximum_optional_rank=int(np.max(rank_array, initial=0)),
            handle_kind="numpy_reference",
            device_identity="cpu",
            logical_rank_dtype=DEVICE_PREFIX_RANK_LOGICAL_DTYPE,
            storage_rank_dtype="numpy.uint32",
            cuda_validation_hbm_forecast=None,
        )
        return cls(
            ranks=rank_array,
            required=required_array,
            receipt=receipt,
            backend_kind="numpy_reference",
            device_identity="cpu",
        )

    @classmethod
    def from_torch_cuda(
        cls,
        *,
        ranks: object,
        required: Sequence[bool] | np.ndarray,
        route_axis_hash: str,
        source_rounding_hash: str,
        stream_chunk_size: int = DEVICE_PREFIX_RANK_DEFAULT_STREAM_CHUNK_SIZE,
        hbm_ceiling_fraction: float = (
            DEVICE_PREFIX_RANK_DEFAULT_HBM_CEILING_FRACTION
        ),
        device_total_memory_bytes: int | None = None,
        current_reserved_bytes: int | None = None,
    ) -> "V1DevicePrefixRankHandle":
        if torch is None:
            raise RuntimeError("CUDA prefix-rank custody requires PyTorch")
        if not isinstance(ranks, torch.Tensor):
            raise TypeError("CUDA prefix ranks require a torch.Tensor")
        if ranks.device.type != "cuda" or ranks.dtype is not torch.int32:
            _refuse(
                V1DevicePrefixRankRefusalCode.INVALID_RANK,
                "CUDA prefix ranks must use canonical contiguous torch.int32 "
                "storage for the logical uint32 rank domain",
            )
        if ranks.ndim != 1 or int(ranks.numel()) <= 0 or not ranks.is_contiguous():
            _refuse(
                V1DevicePrefixRankRefusalCode.INVALID_RANK,
                "CUDA prefix ranks must be a nonempty contiguous vector",
            )
        initial_storage_invariant = _rank_storage_invariant(
            ranks,
            backend_kind="torch_cuda",
        )
        route_count = int(ranks.numel())
        if route_count > int(np.iinfo(np.int32).max):
            _refuse(
                V1DevicePrefixRankRefusalCode.INVALID_RANK,
                "CUDA int32 custody cannot represent a route-scale dense prefix "
                "above INT32_MAX; no silent int64 widening is permitted",
            )
        chunk = min(
            _positive_int(stream_chunk_size, name="stream chunk size"),
            route_count,
        )
        required_array = _normalize_required(required, route_count=route_count)
        forecast = _cuda_hbm_forecast(
            operation="freeze_validation",
            device=ranks.device,
            route_count=route_count,
            endpoint_count=0,
            stream_chunk_size=chunk,
            hbm_ceiling_fraction=hbm_ceiling_fraction,
            device_total_memory_bytes=device_total_memory_bytes,
            current_reserved_bytes=current_reserved_bytes,
        )
        if not forecast.admitted:
            _refuse(
                V1DevicePrefixRankRefusalCode.HBM_FORECAST_EXCEEDED,
                "CUDA prefix-rank validation exceeds the registered HBM ceiling",
            )
        (
            rank_commitment,
            maximum_optional_rank,
            positive_rank_count,
        ) = _rank_commitment_and_local_validation_torch_bounded(
            ranks,
            required=required_array,
            stream_chunk_size=chunk,
        )
        if positive_rank_count != maximum_optional_rank:
            _refuse(
                V1DevicePrefixRankRefusalCode.INVALID_RANK,
                "positive CUDA prefix ranks cannot form dense 1..K custody",
            )
        required_t = torch.tensor(
            required_array,
            dtype=torch.bool,
            device=ranks.device,
        ).contiguous()
        _validate_cross_slab_rank_uniqueness_cuda(
            ranks,
            maximum_optional_rank=maximum_optional_rank,
            stream_chunk_size=chunk,
        )
        if _rank_storage_invariant(
            ranks,
            backend_kind="torch_cuda",
        ) != initial_storage_invariant:
            _refuse(
                V1DevicePrefixRankRefusalCode.RANK_STORAGE_MUTATED,
                "CUDA prefix-rank storage changed during validation/commitment",
            )
        required_commitment = _required_commitment_numpy(required_array)
        receipt = V1DevicePrefixRankReceipt(
            route_axis_hash=route_axis_hash,
            route_count=route_count,
            source_rounding_hash=source_rounding_hash,
            rank_commitment=rank_commitment,
            required_commitment=required_commitment,
            required_total_count=int(np.count_nonzero(required_array)),
            maximum_optional_rank=maximum_optional_rank,
            handle_kind="torch_cuda",
            device_identity=str(ranks.device),
            logical_rank_dtype=DEVICE_PREFIX_RANK_LOGICAL_DTYPE,
            storage_rank_dtype=DEVICE_PREFIX_RANK_CUDA_STORAGE_DTYPE,
            cuda_validation_hbm_forecast=forecast,
        )
        return cls(
            ranks=ranks,
            required=required_t,
            receipt=receipt,
            backend_kind="torch_cuda",
            device_identity=str(ranks.device),
        )

    @property
    def receipt(self) -> V1DevicePrefixRankReceipt:
        self._require_live_locked_unlocked()
        return self._receipt

    @property
    def is_evicted(self) -> bool:
        return bool(self._evicted)

    @property
    def active_lease_count(self) -> int:
        with self._lease_lock:
            self._require_live_locked()
            return len(self._active_lease_tokens)

    def _require_live_locked_unlocked(self) -> object:
        with self._lease_lock:
            return self._require_live_locked()

    def _require_live_locked(self) -> object:
        if self._evicted or self._ranks is None:
            _refuse(
                V1DevicePrefixRankRefusalCode.HANDLE_EVICTED,
                "prefix-rank handle was evicted",
            )
        if _rank_storage_invariant(self._ranks, backend_kind=self._backend_kind) != (
            self._rank_storage_invariant
        ):
            _refuse(
                V1DevicePrefixRankRefusalCode.RANK_STORAGE_MUTATED,
                "prefix-rank storage changed after custody freeze",
            )
        if _rank_storage_invariant(
            self._required,
            backend_kind=self._backend_kind,
        ) != self._required_storage_invariant:
            _refuse(
                V1DevicePrefixRankRefusalCode.RANK_STORAGE_MUTATED,
                "required-route custody changed after prefix freeze",
            )
        return self._ranks

    def derive_endpoint_labels(
        self,
        endpoint_counts: Sequence[int],
        *,
        conditioned_on_c_outcomes: bool,
        stream_chunk_size: int = DEVICE_PREFIX_RANK_DEFAULT_STREAM_CHUNK_SIZE,
        hbm_ceiling_fraction: float = (
            DEVICE_PREFIX_RANK_DEFAULT_HBM_CEILING_FRACTION
        ),
        device_total_memory_bytes: int | None = None,
        current_reserved_bytes: int | None = None,
    ) -> tuple[V1DeviceShellLabelHandle, V1DevicePrefixRankEndpointQueryReceipt]:
        counts = tuple(
            _positive_int(value, name="endpoint count") for value in endpoint_counts
        )
        if not counts or tuple(sorted(set(counts))) != counts:
            _refuse(
                V1DevicePrefixRankRefusalCode.ENDPOINT_OUT_OF_RANGE,
                "endpoint counts must be nonempty, unique, and increasing",
            )
        if len(counts) > DEVICE_SHELL_MAXIMUM_ENDPOINTS:
            _refuse(
                V1DevicePrefixRankRefusalCode.ENDPOINT_LIMIT_EXCEEDED,
                "one endpoint query exceeds uint8 label capacity",
            )
        receipt = self.receipt
        if (
            counts[0] <= receipt.required_total_count
            or counts[-1] > receipt.required_total_count + receipt.maximum_optional_rank
        ):
            _refuse(
                V1DevicePrefixRankRefusalCode.ENDPOINT_OUT_OF_RANGE,
                "endpoint counts fall outside the frozen prefix path",
            )
        with self._lease_lock:
            ranks = self._require_live_locked()
            token = object()
            self._active_lease_tokens.add(token)
            try:
                cuda_forecast = None
                if self._backend_kind == "torch_cuda":
                    cuda_forecast = _cuda_hbm_forecast(
                        operation="endpoint_query",
                        device=ranks.device,  # type: ignore[union-attr]
                        route_count=receipt.route_count,
                        endpoint_count=len(counts),
                        stream_chunk_size=stream_chunk_size,
                        hbm_ceiling_fraction=hbm_ceiling_fraction,
                        device_total_memory_bytes=device_total_memory_bytes,
                        current_reserved_bytes=current_reserved_bytes,
                    )
                    if not cuda_forecast.admitted:
                        _refuse(
                            V1DevicePrefixRankRefusalCode.HBM_FORECAST_EXCEEDED,
                            "CUDA endpoint-label query exceeds the registered HBM ceiling",
                        )
                labels = self._labels_for_counts_locked(
                    ranks,
                    counts,
                    stream_chunk_size=(
                        stream_chunk_size
                        if cuda_forecast is None
                        else cuda_forecast.stream_chunk_size
                    ),
                )
                if self._backend_kind == "numpy_reference":
                    label_commitment = _label_commitment_numpy(labels)
                else:
                    label_commitment = _label_commitment_torch_bounded(
                        labels,
                        stream_chunk_size=(
                            cuda_forecast.stream_chunk_size  # type: ignore[union-attr]
                        ),
                    )
                # The producer may still retain an alias to the input tensor.
                # Recheck tensor-version/storage custody after every query so
                # a concurrent or callback-side mutation cannot yield labels
                # from a rank state different from the frozen commitment.
                self._require_live_locked()
            finally:
                self._active_lease_tokens.remove(token)
        query = V1DevicePrefixRankEndpointQueryReceipt(
            prefix_rank_receipt_hash=receipt.receipt_hash,
            route_axis_hash=receipt.route_axis_hash,
            route_count=receipt.route_count,
            endpoint_counts=counts,
            endpoint_count=len(counts),
            required_total_count=receipt.required_total_count,
            maximum_optional_rank=receipt.maximum_optional_rank,
            resulting_label_commitment=label_commitment,
            endpoint_query_conditioned_on_c_outcomes=conditioned_on_c_outcomes,
            cuda_query_hbm_forecast=cuda_forecast,
        )
        if self._backend_kind == "numpy_reference":
            label_handle = V1DeviceShellLabelHandle.from_numpy_reference(labels)
        else:
            label_handle = V1DeviceShellLabelHandle.from_torch_cuda(labels)
        return label_handle, query

    def _labels_for_counts_locked(
        self,
        ranks: object,
        endpoint_counts: tuple[int, ...],
        *,
        stream_chunk_size: int,
    ) -> object:
        receipt = self._receipt
        optional_counts = np.asarray(
            [count - receipt.required_total_count for count in endpoint_counts],
            dtype=np.uint32,
        )
        if self._backend_kind == "numpy_reference":
            rank_array = np.asarray(ranks, dtype=np.uint32)
            labels = np.zeros(receipt.route_count, dtype=np.uint8)
            if receipt.required_total_count:
                labels[self._required] = 1
            positive = (rank_array > 0) & (rank_array <= optional_counts[-1])
            if bool(np.any(positive)):
                covered = np.searchsorted(
                    optional_counts,
                    rank_array[positive],
                    side="left",
                )
                labels[positive] = (covered + 1).astype(np.uint8)
            labels.setflags(write=False)
            return labels
        if torch is None:  # pragma: no cover - guarded by CUDA construction.
            raise RuntimeError("CUDA endpoint-label derivation requires PyTorch")
        rank_tensor = ranks
        optional_counts_t = torch.tensor(
            optional_counts.astype(np.int32, copy=False),
            dtype=torch.int32,
            device=rank_tensor.device,  # type: ignore[union-attr]
        )
        labels_t = torch.zeros(
            receipt.route_count,
            dtype=torch.uint8,
            device=rank_tensor.device,  # type: ignore[union-attr]
        )
        if receipt.required_total_count:
            labels_t[self._required] = 1
        chunk = min(
            _positive_int(stream_chunk_size, name="stream chunk size"),
            receipt.route_count,
        )
        maximum_requested = int(optional_counts[-1])
        for start in range(0, receipt.route_count, chunk):
            stop = min(receipt.route_count, start + chunk)
            rank_slab = rank_tensor[start:stop]  # type: ignore[index]
            positive = (rank_slab > 0) & (rank_slab <= maximum_requested)
            covered = torch.searchsorted(
                optional_counts_t,
                rank_slab,
                right=False,
                out_int32=True,
            )
            label_slab = labels_t[start:stop]
            label_slab[positive] = (covered[positive] + 1).to(dtype=torch.uint8)
        return labels_t.contiguous()

    def evict(self) -> None:
        with self._lease_lock:
            self._require_live_locked()
            if self._active_lease_tokens:
                _refuse(
                    V1DevicePrefixRankRefusalCode.HANDLE_HAS_ACTIVE_LEASES,
                    "cannot evict prefix ranks while a query is active",
                )
            self._ranks = None
            self._required = None
            self._evicted = True

    @classmethod
    def evict_all_atomically(
        cls,
        handles: Sequence["V1DevicePrefixRankHandle"],
    ) -> None:
        normalized = tuple(handles)
        if not normalized or any(type(handle) is not cls for handle in normalized):
            raise TypeError("atomic prefix-rank eviction requires exact handles")
        if len({id(handle) for handle in normalized}) != len(normalized):
            raise ValueError("atomic prefix-rank eviction handles must be unique")
        ordered = tuple(sorted(normalized, key=id))
        with ExitStack() as stack:
            for handle in ordered:
                stack.enter_context(handle._lease_lock)
            for handle in normalized:
                handle._require_live_locked()
                if handle._active_lease_tokens:
                    _refuse(
                        V1DevicePrefixRankRefusalCode.HANDLE_HAS_ACTIVE_LEASES,
                        "no prefix-rank handle was evicted because one is active",
                    )
            for handle in normalized:
                handle._ranks = None
                handle._required = None
                handle._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("prefix-rank handles are ephemeral and nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("prefix-rank handles are ephemeral and nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("prefix-rank handles are ephemeral and nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("prefix-rank handles cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("prefix-rank handles cannot be copied")


def _rank_commitment_numpy(ranks: np.ndarray) -> str:
    digest = _stream_digest(
        domain=_RANK_COMMITMENT_DOMAIN,
        metadata={
            "dtype": "uint32",
            "shape": tuple(int(item) for item in ranks.shape),
        },
    )
    digest.update(memoryview(np.ascontiguousarray(ranks, dtype=np.uint32)).cast("B"))
    return "partition-free-v1-device-prefix-ranks-" + digest.hexdigest()


def _copy_cuda_int32_slab_to_host(
    ranks: object,
    *,
    start: int,
    stop: int,
) -> np.ndarray:
    """Transfer exactly one registered rank slab, never the route-scale vector."""

    slab = ranks[start:stop].detach().to(  # type: ignore[index,union-attr]
        device="cpu",
        non_blocking=False,
    )
    result = np.asarray(slab.contiguous().numpy(), dtype=np.int32)
    if result.ndim != 1 or result.size != stop - start:
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_RANK,
            "bounded CUDA rank transfer changed the registered slab shape",
        )
    return result


def _copy_cuda_uint8_slab_to_host(
    labels: object,
    *,
    start: int,
    stop: int,
) -> np.ndarray:
    """Transfer exactly one registered label slab for its SHA commitment."""

    slab = labels[start:stop].detach().to(  # type: ignore[index,union-attr]
        device="cpu",
        non_blocking=False,
    )
    result = np.asarray(slab.contiguous().numpy(), dtype=np.uint8)
    if result.ndim != 1 or result.size != stop - start:
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_RANK,
            "bounded CUDA label commitment changed a registered slab shape",
        )
    return result


def _rank_commitment_and_local_validation_torch_bounded(
    ranks: object,
    *,
    required: np.ndarray,
    stream_chunk_size: int,
) -> tuple[str, int, int]:
    route_count = int(ranks.numel())  # type: ignore[union-attr]
    chunk = min(
        _positive_int(stream_chunk_size, name="stream chunk size"),
        route_count,
    )
    digest = _stream_digest(
        domain=_RANK_COMMITMENT_DOMAIN,
        metadata={
            "dtype": DEVICE_PREFIX_RANK_LOGICAL_DTYPE,
            "shape": (route_count,),
        },
    )
    maximum = 0
    positive_count = 0
    for start in range(0, route_count, chunk):
        stop = min(route_count, start + chunk)
        host = _copy_cuda_int32_slab_to_host(ranks, start=start, stop=stop)
        if bool(np.any(host < 0)):
            _refuse(
                V1DevicePrefixRankRefusalCode.INVALID_RANK,
                "CUDA prefix ranks cannot be negative in the logical uint32 domain",
            )
        if bool(np.any(host[required[start:stop]] != 0)):
            _refuse(
                V1DevicePrefixRankRefusalCode.INVALID_RANK,
                "required routes are carried separately and must have rank zero",
            )
        # Local duplicate detection stays bounded to this registered slab.
        # Global cross-slab uniqueness is checked by a CUDA-resident seen table
        # in a separate pass.
        positive = np.array(host[host > 0], dtype=np.int32, copy=True)
        if positive.size:
            positive.sort(kind="stable")
            if bool(np.any(positive[1:] == positive[:-1])):
                _refuse(
                    V1DevicePrefixRankRefusalCode.INVALID_RANK,
                    "CUDA prefix ranks contain duplicate positive ranks",
                )
            maximum = max(maximum, int(positive[-1]))
            positive_count += int(positive.size)
        # int32 storage is nonnegative, so its bytes are exactly the registered
        # native-endian uint32 logical representation used by the CPU receipt.
        logical = np.ascontiguousarray(host).view(np.uint32)
        digest.update(memoryview(logical).cast("B"))
    return (
        "partition-free-v1-device-prefix-ranks-" + digest.hexdigest(),
        maximum,
        positive_count,
    )


def _validate_cross_slab_rank_uniqueness_cuda(
    ranks: object,
    *,
    maximum_optional_rank: int,
    stream_chunk_size: int,
) -> None:
    """Prove global uniqueness with bounded slabs and CUDA-resident state."""

    if torch is None:  # pragma: no cover - guarded by CUDA construction.
        raise RuntimeError("CUDA rank validation requires PyTorch")
    maximum = _nonnegative_int(
        maximum_optional_rank,
        name="maximum optional rank",
    )
    if maximum == 0:
        return
    route_count = int(ranks.numel())  # type: ignore[union-attr]
    chunk = min(
        _positive_int(stream_chunk_size, name="stream chunk size"),
        route_count,
    )
    seen = torch.zeros(
        maximum + 1,
        dtype=torch.bool,
        device=ranks.device,  # type: ignore[union-attr]
    )
    for start in range(0, route_count, chunk):
        stop = min(route_count, start + chunk)
        slab = ranks[start:stop]  # type: ignore[index]
        positive = slab[slab > 0]
        if int(positive.numel()) == 0:
            continue
        indices = positive.to(dtype=torch.int64)
        if bool(torch.any(seen.index_select(0, indices)).item()):
            _refuse(
                V1DevicePrefixRankRefusalCode.INVALID_RANK,
                "CUDA prefix ranks repeat across validation slabs",
            )
        # Within-slab duplicates were already rejected from the bounded host
        # copy, so this exact index_fill has one writer per logical rank.
        seen.index_fill_(0, indices, True)
    if not bool(torch.all(seen[1:]).item()):
        _refuse(
            V1DevicePrefixRankRefusalCode.INVALID_RANK,
            "positive CUDA prefix ranks are not a dense 1..K permutation",
        )


def _required_commitment_numpy(required: np.ndarray) -> str:
    packed = np.packbits(np.asarray(required, dtype=np.bool_), bitorder="big")
    digest = _stream_digest(
        domain=_REQUIRED_COMMITMENT_DOMAIN,
        metadata={
            "route_count": int(required.size),
            "bitorder": "big",
            "packed_bytes": int(packed.size),
        },
    )
    digest.update(memoryview(np.ascontiguousarray(packed)).cast("B"))
    return "partition-free-v1-device-prefix-required-" + digest.hexdigest()


def _label_commitment_numpy(labels: np.ndarray) -> str:
    digest = _stream_digest(
        domain="partition_free_v1_device_prefix_rank_query_labels_v1",
        metadata={
            "dtype": "uint8",
            "shape": tuple(int(item) for item in labels.shape),
        },
    )
    digest.update(memoryview(np.ascontiguousarray(labels, dtype=np.uint8)).cast("B"))
    return "partition-free-v1-device-prefix-query-labels-" + digest.hexdigest()


def _label_commitment_torch_bounded(
    labels: object,
    *,
    stream_chunk_size: int,
) -> str:
    route_count = int(labels.numel())  # type: ignore[union-attr]
    chunk = min(
        _positive_int(stream_chunk_size, name="stream chunk size"),
        route_count,
    )
    digest = _stream_digest(
        domain="partition_free_v1_device_prefix_rank_query_labels_v1",
        metadata={
            "dtype": "uint8",
            "shape": (route_count,),
        },
    )
    for start in range(0, route_count, chunk):
        stop = min(route_count, start + chunk)
        host = _copy_cuda_uint8_slab_to_host(
            labels,
            start=start,
            stop=stop,
        )
        digest.update(memoryview(np.ascontiguousarray(host)).cast("B"))
    return "partition-free-v1-device-prefix-query-labels-" + digest.hexdigest()


__all__ = [
    "DEVICE_PREFIX_RANK_ALGORITHM_ID",
    "DEVICE_PREFIX_RANK_AUTHORITY",
    "DEVICE_PREFIX_RANK_CUDA_HBM_SCHEMA",
    "DEVICE_PREFIX_RANK_CUDA_STORAGE_DTYPE",
    "DEVICE_PREFIX_RANK_DEFAULT_HBM_CEILING_FRACTION",
    "DEVICE_PREFIX_RANK_DEFAULT_STREAM_CHUNK_SIZE",
    "DEVICE_PREFIX_RANK_LOGICAL_DTYPE",
    "DEVICE_PREFIX_RANK_QUERY_SCHEMA",
    "DEVICE_PREFIX_RANK_RECEIPT_SCHEMA",
    "V1DevicePrefixRankCUDAHBMForecast",
    "V1DevicePrefixRankEndpointQueryReceipt",
    "V1DevicePrefixRankHandle",
    "V1DevicePrefixRankRefusal",
    "V1DevicePrefixRankRefusalCode",
    "V1DevicePrefixRankReceipt",
]
