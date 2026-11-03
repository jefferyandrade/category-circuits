"""GPU-reduced proposal packets for modern-v1 corrected attribution.

This module is the narrow production seam between a route-chunked scorer and
``V1StratifiedCorrectedAttributionMoments``.  The scorer supplies signed
raw-target-logit positive-removal rows as contiguous ``[draw, route-window]``
torch tensors.  The packet reducer keeps only per-stratum routewise Welford
state on the selected device, then transfers the four aggregate
``[stratum, route]`` arrays at packet close.

It deliberately does not emit a promotable FD proof.  A route-row proof for a
Frequent-Directions geometry has a different authority surface from a reduced
mean/M2 packet, and pretending otherwise would make the statistical evidence
look stronger than it is.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import time
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_proposal_statistics import (
    V1StratifiedCorrectedAttributionMoments,
)
from mechanistic_systems.contracts.content_address import content_hash

try:  # pragma: no cover - exercised by import behavior, not local logic.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


GPU_REDUCED_PROPOSAL_PACKET_REQUEST_SCHEMA: Final = (
    "partition_free_v1_gpu_reduced_corrected_attribution_packet_request_v1"
)
GPU_REDUCED_PROPOSAL_PACKET_SCHEMA: Final = (
    "partition_free_v1_gpu_reduced_corrected_attribution_packet_v1"
)
GPU_REDUCED_PROPOSAL_PACKET_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_gpu_reduced_corrected_attribution_packet_receipt_v1"
)
GPU_REDUCED_PROPOSAL_PACKET_BOUNDARY_SNAPSHOT_SCHEMA: Final = (
    "partition_free_v1_gpu_reduced_packet_boundary_snapshot_v1"
)
GPU_REDUCED_PROPOSAL_PACKET_MEMORY_FORECAST_SCHEMA: Final = (
    "partition_free_v1_gpu_reduced_packet_memory_forecast_v1"
)

REGISTERED_GPU_PACKET_ARITHMETIC_ID: Final = (
    "ordered_row_welford_by_contiguous_route_window_float64_packet_v1"
)
REGISTERED_GPU_PACKETIZATION_KIND: Final = (
    "registered_ascending_contiguous_route_windows_one_packet_v1"
)
GPU_PACKET_FD_PROMOTION_STATUS: Final = (
    "not_a_frequent_directions_row_proof_and_not_promotable_to_trust_geometry"
)
REGISTERED_GPU_PACKET_MEMORY_ACCOUNTING_SCOPE: Final = (
    "registered_bulk_tensor_and_host_array_bytes_excluding_allocator_"
    "overhead_and_scorer_owned_source_tensor_v1"
)

_LE_F64: Final = np.dtype("<f8")
_FLOAT64_BYTES: Final = 8
_NORMALIZATION_CHUNK_ELEMENTS: Final = 8 * 1024 * 1024
_NEGATIVE_ZERO_U64: Final = np.uint64(1 << 63)
_REQUEST_PREFIX: Final = "partition-free-v1-gpu-reduced-packet-request-"
_PACKETIZATION_PREFIX: Final = "partition-free-v1-gpu-packetization-"
_RECEIPT_PREFIX: Final = "partition-free-v1-gpu-reduced-packet-receipt-"
_PACKET_PREFIX: Final = "partition-free-v1-gpu-reduced-packet-"
_SNAPSHOT_PREFIX: Final = "partition-free-v1-gpu-reduced-packet-snapshot-"


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be one positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _pair_of_names(value: Sequence[str], *, name: str) -> tuple[str, str]:
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if len(result) != 2 or result[0] == result[1]:
        raise ValueError(f"{name} must contain exactly two distinct names")
    return result  # type: ignore[return-value]


def _pair_of_identities(value: Sequence[str], *, name: str) -> tuple[str, str]:
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two identities")
    return result  # type: ignore[return-value]


def _pair_of_positive_ints(value: Sequence[int], *, name: str) -> tuple[int, int]:
    result = tuple(_positive_int(item, name=f"{name} item") for item in value)
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return result  # type: ignore[return-value]


def _pair_of_nonnegative_ints(value: Sequence[int], *, name: str) -> tuple[int, int]:
    result = tuple(_nonnegative_int(item, name=f"{name} item") for item in value)
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return result  # type: ignore[return-value]


def _readonly_le_f64(value: Any, *, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        candidate: np.ndarray | None = np.asarray(value)
    except (TypeError, ValueError, RuntimeError):
        candidate = None
    base: Any = candidate
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    if candidate is not None and (
        candidate.dtype == _LE_F64
        and candidate.shape == shape
        and candidate.flags.c_contiguous
        and not candidate.flags.writeable
        and isinstance(base, bytes)
    ):
        flat = candidate.reshape(-1)
        canonical = True
        for start in range(0, flat.size, _NORMALIZATION_CHUNK_ELEMENTS):
            chunk = flat[start : start + _NORMALIZATION_CHUNK_ELEMENTS]
            if (
                not np.isfinite(chunk).all()
                or np.any(chunk.view(np.uint64) == _NEGATIVE_ZERO_U64)
            ):
                canonical = False
                break
        if canonical:
            return candidate
    try:
        canonical = np.array(value, dtype=_LE_F64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a canonical float64 array") from error
    if canonical.shape != shape:
        raise ValueError(f"{name} must be one finite array with shape {shape}")
    flat = canonical.reshape(-1)
    for start in range(0, flat.size, _NORMALIZATION_CHUNK_ELEMENTS):
        chunk = flat[start : start + _NORMALIZATION_CHUNK_ELEMENTS]
        if not np.isfinite(chunk).all():
            raise ValueError(f"{name} must be one finite array with shape {shape}")
        bits = chunk.view(np.uint64)
        bits[bits == _NEGATIVE_ZERO_U64] = np.uint64(0)
    immutable_bytes = canonical.tobytes(order="C")
    result = np.frombuffer(immutable_bytes, dtype=_LE_F64).reshape(shape)
    result.setflags(write=False)
    return result


def _nested_identity_pair(
    value: Sequence[Sequence[str]], *, name: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    outer = tuple(tuple(row) for row in value)
    if len(outer) != 2:
        raise ValueError(f"{name} must contain exactly two ledgers")
    result = tuple(
        tuple(_identity(item, name=f"{name} item") for item in row) for row in outer
    )
    return result  # type: ignore[return-value]


def _validate_unique_ledger(values: tuple[str, ...], *, name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} cannot contain duplicates")


def _device_identity(device: Any) -> str:
    if torch is None:
        return "torch-unavailable"
    runtime_device = torch.device(device)
    if runtime_device.type != "cuda":
        return str(runtime_device)
    try:
        name = torch.cuda.get_device_name(runtime_device)
    except (AssertionError, RuntimeError):
        name = "cuda-device-name-unavailable"
    index = runtime_device.index
    if index is None:
        try:
            index = int(torch.cuda.current_device())
        except (AssertionError, RuntimeError):
            index = -1
    return f"cuda:{index}:{name}"


def _canonical_runtime_device(device: Any) -> Any:
    if torch is None:
        raise RuntimeError("GPU reduced proposal packets require PyTorch")
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and runtime_device.index is None:
        try:
            return torch.device("cuda", int(torch.cuda.current_device()))
        except (AssertionError, RuntimeError):
            return runtime_device
    return runtime_device


def _copy_aggregate_to_host_array(
    tensor: Any,
    *,
    role: str,
    route_count: int,
    diagnostics: dict[str, float | int] | None = None,
) -> np.ndarray:
    """Copy one aggregate ``[2, route]`` tensor to immutable NumPy storage."""

    if torch is None or not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{role} transfer requires one torch tensor")
    if tuple(int(axis) for axis in tensor.shape) != (2, int(route_count)):
        raise RuntimeError(
            f"{role} attempted to transfer a non-aggregate tensor of shape "
            f"{tuple(int(axis) for axis in tensor.shape)}"
        )
    transfer_started = time.perf_counter()
    host = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    transfer_seconds = time.perf_counter() - transfer_started
    canonicalization_started = time.perf_counter()
    array = host.numpy()
    flat = array.reshape(-1)
    for start in range(0, flat.size, _NORMALIZATION_CHUNK_ELEMENTS):
        chunk = flat[start : start + _NORMALIZATION_CHUNK_ELEMENTS]
        if not np.isfinite(chunk).all():
            raise ValueError(f"{role} contains non-finite aggregate values")
        bits = chunk.view(np.uint64)
        bits[bits == _NEGATIVE_ZERO_U64] = np.uint64(0)
    result = np.frombuffer(array.tobytes(order="C"), dtype=_LE_F64).reshape(
        (2, int(route_count))
    )
    result.setflags(write=False)
    canonicalization_seconds = time.perf_counter() - canonicalization_started
    if diagnostics is not None:
        # This existing D2H barrier also waits for outstanding reduction
        # kernels; no diagnostic-only synchronization is inserted.
        diagnostics["score_d2h_synchronization_inclusive_seconds"] = float(
            diagnostics.get("score_d2h_synchronization_inclusive_seconds", 0.0)
        ) + float(transfer_seconds)
        diagnostics["score_d2h_array_count"] = int(
            diagnostics.get("score_d2h_array_count", 0)
        ) + 1
        diagnostics["score_d2h_bytes"] = int(
            diagnostics.get("score_d2h_bytes", 0)
        ) + int(result.nbytes)
        diagnostics["score_canonicalization_seconds"] = float(
            diagnostics.get("score_canonicalization_seconds", 0.0)
        ) + float(canonicalization_seconds)
        diagnostics["score_canonicalization_array_count"] = int(
            diagnostics.get("score_canonicalization_array_count", 0)
        ) + 1
        diagnostics["score_canonicalization_bytes"] = int(
            diagnostics.get("score_canonicalization_bytes", 0)
        ) + int(result.nbytes)
    return result


@dataclass(frozen=True, slots=True)
class V1GPUReducedProposalPacketMemoryForecast:
    """Exact registered bulk-array ledger for one reduced packet.

    The scope deliberately excludes allocator bookkeeping, library workspace,
    and the scorer-owned source tensor.  Those belong to the enclosing scorer
    admission forecast rather than this reducer's independently replayable
    allocation ledger.
    """

    route_count: int
    packet_draw_count: int
    registered_route_chunk_size: int
    aggregate_device_bytes: int
    working_chunk_device_bytes: int
    peak_device_bytes: int
    aggregate_host_transfer_bytes: int
    peak_host_bytes: int
    avoided_context_route_host_tensor_bytes: int
    retained_host_context_route_tensor_bytes: int = 0
    device_budget_bytes: int | None = None
    host_budget_bytes: int | None = None
    memory_accounting_scope: str = REGISTERED_GPU_PACKET_MEMORY_ACCOUNTING_SCOPE
    schema: str = GPU_REDUCED_PROPOSAL_PACKET_MEMORY_FORECAST_SCHEMA

    def __post_init__(self) -> None:
        routes = _positive_int(self.route_count, name="route count")
        draws = _positive_int(self.packet_draw_count, name="packet draw count")
        chunk = _positive_int(
            self.registered_route_chunk_size,
            name="registered route chunk size",
        )
        if chunk > routes:
            raise ValueError("registered route chunk size exceeds the route axis")
        expected_aggregate = 4 * 2 * routes * _FLOAT64_BYTES
        expected_working = draws * chunk * _FLOAT64_BYTES
        expected_host_peak = expected_aggregate + 2 * routes * _FLOAT64_BYTES
        expected_avoided = draws * routes * _FLOAT64_BYTES
        expected = {
            "aggregate_device_bytes": expected_aggregate,
            "working_chunk_device_bytes": expected_working,
            "peak_device_bytes": expected_aggregate + expected_working,
            "aggregate_host_transfer_bytes": expected_aggregate,
            "peak_host_bytes": expected_host_peak,
            "avoided_context_route_host_tensor_bytes": expected_avoided,
            "retained_host_context_route_tensor_bytes": 0,
        }
        for name, expected_value in expected.items():
            observed = _nonnegative_int(getattr(self, name), name=name)
            if observed != expected_value:
                raise ValueError(f"{name} does not replay from the forecast inputs")
            object.__setattr__(self, name, observed)
        for name in ("device_budget_bytes", "host_budget_bytes"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonnegative_int(value, name=name))
        if self.memory_accounting_scope != REGISTERED_GPU_PACKET_MEMORY_ACCOUNTING_SCOPE:
            raise ValueError("packet memory accounting scope is invalid")
        if self.schema != GPU_REDUCED_PROPOSAL_PACKET_MEMORY_FORECAST_SCHEMA:
            raise ValueError("GPU reduced packet memory forecast schema is invalid")
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "packet_draw_count", draws)
        object.__setattr__(self, "registered_route_chunk_size", chunk)

    @property
    def admitted(self) -> bool:
        if (
            self.device_budget_bytes is not None
            and self.peak_device_bytes > self.device_budget_bytes
        ):
            return False
        if (
            self.host_budget_bytes is not None
            and self.peak_host_bytes > self.host_budget_bytes
        ):
            return False
        return True

    @property
    def refusal_reason(self) -> str | None:
        if (
            self.device_budget_bytes is not None
            and self.peak_device_bytes > self.device_budget_bytes
        ):
            return "peak device bytes exceed the registered budget"
        if (
            self.host_budget_bytes is not None
            and self.peak_host_bytes > self.host_budget_bytes
        ):
            return "peak host bytes exceed the registered budget"
        return None

    def assert_admissible(self) -> None:
        reason = self.refusal_reason
        if reason is not None:
            raise MemoryError(reason)


def forecast_gpu_reduced_proposal_packet_memory(
    *,
    route_count: int,
    packet_draw_count: int,
    registered_route_chunk_size: int,
    device_budget_bytes: int | None = None,
    host_budget_bytes: int | None = None,
) -> V1GPUReducedProposalPacketMemoryForecast:
    """Return the exact asymptotic byte ledger without allocating tensors."""

    routes = _positive_int(route_count, name="route count")
    draws = _positive_int(packet_draw_count, name="packet draw count")
    chunk = _positive_int(
        registered_route_chunk_size, name="registered route chunk size"
    )
    chunk = min(chunk, routes)
    float_bytes = _FLOAT64_BYTES
    aggregate_device = 4 * 2 * routes * float_bytes
    working_chunk = draws * chunk * float_bytes
    aggregate_host = 4 * 2 * routes * float_bytes
    # During the final aggregate copy, the three previously frozen arrays and
    # the current CPU tensor coexist with the current immutable byte-backed
    # result.  Packet construction reuses those canonical arrays without a
    # second full-size copy.
    peak_host = aggregate_host + 2 * routes * float_bytes
    avoided_host_rows = draws * routes * float_bytes
    return V1GPUReducedProposalPacketMemoryForecast(
        route_count=routes,
        packet_draw_count=draws,
        registered_route_chunk_size=chunk,
        aggregate_device_bytes=aggregate_device,
        working_chunk_device_bytes=working_chunk,
        peak_device_bytes=aggregate_device + working_chunk,
        aggregate_host_transfer_bytes=aggregate_host,
        peak_host_bytes=peak_host,
        avoided_context_route_host_tensor_bytes=avoided_host_rows,
        device_budget_bytes=device_budget_bytes,
        host_budget_bytes=host_budget_bytes,
    )


@dataclass(frozen=True, slots=True)
class V1GPUReducedProposalPacketRequest:
    """Frozen execution identity for one reduced proposal packet."""

    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    stratum_names: tuple[str, str]
    population_stratum_counts: tuple[int, int]
    sealed_population_order_hashes: tuple[str, str]
    sealed_order_start_indices: tuple[int, int]
    expected_context_ids: tuple[str, ...]
    expected_draw_ids: tuple[str, ...]
    expected_stratum_labels: tuple[str, ...]
    packet_index: int
    packetization_plan_hash: str
    registered_route_chunk_size: int
    schema: str = GPU_REDUCED_PROPOSAL_PACKET_REQUEST_SCHEMA
    _request_hash: str | None = field(init=False, default=None, repr=False)
    _packetization_identity_hash: str | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "route_axis_hash",
            "packetization_plan_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        route_count = _positive_int(self.route_count, name="route count")
        names = _pair_of_names(self.stratum_names, name="stratum names")
        populations = _pair_of_positive_ints(
            self.population_stratum_counts, name="population stratum counts"
        )
        order_hashes = _pair_of_identities(
            self.sealed_population_order_hashes,
            name="sealed population order hashes",
        )
        starts = _pair_of_nonnegative_ints(
            self.sealed_order_start_indices, name="sealed order start indices"
        )
        packet_index = _nonnegative_int(self.packet_index, name="packet index")
        route_chunk_size = _positive_int(
            self.registered_route_chunk_size, name="registered route chunk size"
        )
        if route_chunk_size > route_count:
            route_chunk_size = route_count
        contexts = tuple(
            _identity(item, name="expected context id")
            for item in self.expected_context_ids
        )
        draws = tuple(
            _identity(item, name="expected draw id") for item in self.expected_draw_ids
        )
        labels = tuple(
            _identity(item, name="expected stratum label")
            for item in self.expected_stratum_labels
        )
        if not contexts or not (len(contexts) == len(draws) == len(labels)):
            raise ValueError("expected packet ledgers must be aligned and nonempty")
        _validate_unique_ledger(contexts, name="expected context ledger")
        _validate_unique_ledger(draws, name="expected draw ledger")
        name_to_index = {name: index for index, name in enumerate(names)}
        counts = [0, 0]
        for label in labels:
            if label not in name_to_index:
                raise ValueError("expected packet names an unregistered stratum")
            counts[name_to_index[label]] += 1
        if any(
            start + count > population
            for start, count, population in zip(starts, counts, populations)
        ):
            raise ValueError("expected packet exhausts its finite population")
        if self.schema != GPU_REDUCED_PROPOSAL_PACKET_REQUEST_SCHEMA:
            raise ValueError("GPU reduced packet request schema is invalid")

        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(self, "sealed_population_order_hashes", order_hashes)
        object.__setattr__(self, "sealed_order_start_indices", starts)
        object.__setattr__(self, "expected_context_ids", contexts)
        object.__setattr__(self, "expected_draw_ids", draws)
        object.__setattr__(self, "expected_stratum_labels", labels)
        object.__setattr__(self, "packet_index", packet_index)
        object.__setattr__(self, "registered_route_chunk_size", route_chunk_size)

    @property
    def packet_draw_count(self) -> int:
        return len(self.expected_context_ids)

    @property
    def packet_stratum_counts(self) -> tuple[int, int]:
        counts = [0, 0]
        name_to_index = {name: index for index, name in enumerate(self.stratum_names)}
        for label in self.expected_stratum_labels:
            counts[name_to_index[label]] += 1
        return counts[0], counts[1]

    @property
    def expected_context_ids_by_stratum(
        self,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self._group_flat_ledger(self.expected_context_ids)

    @property
    def expected_draw_ids_by_stratum(
        self,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self._group_flat_ledger(self.expected_draw_ids)

    def _group_flat_ledger(
        self, values: tuple[str, ...]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        grouped: list[list[str]] = [[], []]
        name_to_index = {name: index for index, name in enumerate(self.stratum_names)}
        for value, label in zip(values, self.expected_stratum_labels, strict=True):
            grouped[name_to_index[label]].append(value)
        return tuple(tuple(row) for row in grouped)  # type: ignore[return-value]

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "campaign_contract_hash": self.campaign_contract_hash,
            "model_identity_hash": self.model_identity_hash,
            "query_contract_hash": self.query_contract_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "current_mask_hash": self.current_mask_hash,
            "raw_logit_stream_source_hash": self.raw_logit_stream_source_hash,
            "sampling_law_hash": self.sampling_law_hash,
            "proposal_generation": self.proposal_generation,
            "route_count": self.route_count,
            "route_axis_hash": self.route_axis_hash,
            "stratum_names": self.stratum_names,
            "population_stratum_counts": self.population_stratum_counts,
            "sealed_population_order_hashes": self.sealed_population_order_hashes,
            "sealed_order_start_indices": self.sealed_order_start_indices,
            "expected_context_ids": self.expected_context_ids,
            "expected_draw_ids": self.expected_draw_ids,
            "expected_stratum_labels": self.expected_stratum_labels,
            "packet_index": self.packet_index,
            "packetization_plan_hash": self.packetization_plan_hash,
            "registered_route_chunk_size": self.registered_route_chunk_size,
            "packetization_kind": REGISTERED_GPU_PACKETIZATION_KIND,
            "packet_arithmetic": REGISTERED_GPU_PACKET_ARITHMETIC_ID,
        }

    @property
    def request_hash(self) -> str:
        cached = self._request_hash
        if cached is None:
            cached = content_hash(self._hash_payload(), prefix=_REQUEST_PREFIX)
            object.__setattr__(self, "_request_hash", cached)
        return cached

    @property
    def packetization_identity_hash(self) -> str:
        cached = self._packetization_identity_hash
        if cached is None:
            cached = content_hash(
                {
                    "request_hash": self.request_hash,
                    "route_count": self.route_count,
                    "packet_draw_count": self.packet_draw_count,
                    "registered_route_chunk_size": self.registered_route_chunk_size,
                    "packet_index": self.packet_index,
                    "packetization_kind": REGISTERED_GPU_PACKETIZATION_KIND,
                },
                prefix=_PACKETIZATION_PREFIX,
            )
            object.__setattr__(self, "_packetization_identity_hash", cached)
        return cached

    def forecast_memory(
        self,
        *,
        device_budget_bytes: int | None = None,
        host_budget_bytes: int | None = None,
    ) -> V1GPUReducedProposalPacketMemoryForecast:
        return forecast_gpu_reduced_proposal_packet_memory(
            route_count=self.route_count,
            packet_draw_count=self.packet_draw_count,
            registered_route_chunk_size=self.registered_route_chunk_size,
            device_budget_bytes=device_budget_bytes,
            host_budget_bytes=host_budget_bytes,
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_REQUEST_PREFIX)
        if self._request_hash is not None and observed != self._request_hash:
            raise ValueError("GPU reduced packet request was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("GPU reduced packet request hash does not match")
        object.__setattr__(self, "_request_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1GPUReducedProposalPacketDerivationReceipt:
    """Immutable receipt for the reduced packet arithmetic and transfers."""

    request_hash: str
    packetization_identity_hash: str
    device_identity: str
    route_count: int
    packet_draw_count: int
    packet_stratum_counts: tuple[int, int]
    sealed_order_start_indices: tuple[int, int]
    route_chunk_windows: tuple[tuple[int, int], ...]
    observed_context_ids_by_stratum: tuple[tuple[str, ...], tuple[str, ...]]
    observed_draw_ids_by_stratum: tuple[tuple[str, ...], tuple[str, ...]]
    signed_mean_hash: str
    signed_m2_hash: str
    absolute_mean_hash: str
    absolute_m2_hash: str
    peak_device_bytes: int
    peak_host_bytes: int
    aggregate_host_transfer_bytes: int
    avoided_context_route_host_tensor_bytes: int
    retained_host_context_route_tensor_bytes: int = 0
    host_context_route_rows_transferred: bool = False
    arithmetic_identity: str = REGISTERED_GPU_PACKET_ARITHMETIC_ID
    packetization_kind: str = REGISTERED_GPU_PACKETIZATION_KIND
    fd_promotion_status: str = GPU_PACKET_FD_PROMOTION_STATUS
    memory_accounting_scope: str = REGISTERED_GPU_PACKET_MEMORY_ACCOUNTING_SCOPE
    snapshot_policy: str = "packet_boundary_only_replay_partial_packets_from_start_v1"
    schema: str = GPU_REDUCED_PROPOSAL_PACKET_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "request_hash",
            "packetization_identity_hash",
            "device_identity",
            "signed_mean_hash",
            "signed_m2_hash",
            "absolute_mean_hash",
            "absolute_m2_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        route_count = _positive_int(self.route_count, name="route count")
        draw_count = _positive_int(self.packet_draw_count, name="packet draw count")
        counts = _pair_of_nonnegative_ints(
            self.packet_stratum_counts, name="packet stratum counts"
        )
        if sum(counts) != draw_count:
            raise ValueError("receipt stratum counts do not sum to packet draws")
        starts = _pair_of_nonnegative_ints(
            self.sealed_order_start_indices, name="sealed order start indices"
        )
        windows = tuple(
            (
                _nonnegative_int(window[0], name="route window start"),
                _positive_int(window[1], name="route window stop"),
            )
            for window in self.route_chunk_windows
        )
        if not windows:
            raise ValueError("receipt requires at least one route chunk window")
        cursor = 0
        for start, stop in windows:
            if start != cursor or stop <= start or stop > route_count:
                raise ValueError("receipt route windows are not the registered prefix")
            cursor = stop
        if cursor != route_count:
            raise ValueError("receipt route windows do not cover the full route axis")
        contexts = _nested_identity_pair(
            self.observed_context_ids_by_stratum,
            name="receipt context ledgers",
        )
        draws = _nested_identity_pair(
            self.observed_draw_ids_by_stratum,
            name="receipt draw ledgers",
        )
        if any(len(contexts[index]) != counts[index] for index in (0, 1)):
            raise ValueError("receipt context ledgers disagree with counts")
        if any(len(draws[index]) != counts[index] for index in (0, 1)):
            raise ValueError("receipt draw ledgers disagree with counts")
        _validate_unique_ledger(contexts[0] + contexts[1], name="receipt contexts")
        _validate_unique_ledger(draws[0] + draws[1], name="receipt draws")
        byte_fields = (
            "peak_device_bytes",
            "peak_host_bytes",
            "aggregate_host_transfer_bytes",
            "avoided_context_route_host_tensor_bytes",
            "retained_host_context_route_tensor_bytes",
        )
        for name in byte_fields:
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name=name)
            )
        if self.retained_host_context_route_tensor_bytes != 0:
            raise ValueError("GPU reduced packets cannot retain host row tensors")
        if self.host_context_route_rows_transferred is not False:
            raise ValueError("GPU reduced packets cannot transfer context-route rows")
        if self.arithmetic_identity != REGISTERED_GPU_PACKET_ARITHMETIC_ID:
            raise ValueError("receipt arithmetic identity is invalid")
        if self.packetization_kind != REGISTERED_GPU_PACKETIZATION_KIND:
            raise ValueError("receipt packetization kind is invalid")
        if self.fd_promotion_status != GPU_PACKET_FD_PROMOTION_STATUS:
            raise ValueError("receipt FD promotion status is invalid")
        if self.memory_accounting_scope != REGISTERED_GPU_PACKET_MEMORY_ACCOUNTING_SCOPE:
            raise ValueError("receipt memory accounting scope is invalid")
        if self.schema != GPU_REDUCED_PROPOSAL_PACKET_RECEIPT_SCHEMA:
            raise ValueError("GPU reduced packet receipt schema is invalid")
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "packet_draw_count", draw_count)
        object.__setattr__(self, "packet_stratum_counts", counts)
        object.__setattr__(self, "sealed_order_start_indices", starts)
        object.__setattr__(self, "route_chunk_windows", windows)
        object.__setattr__(self, "observed_context_ids_by_stratum", contexts)
        object.__setattr__(self, "observed_draw_ids_by_stratum", draws)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_hash": self.request_hash,
            "packetization_identity_hash": self.packetization_identity_hash,
            "device_identity": self.device_identity,
            "route_count": self.route_count,
            "packet_draw_count": self.packet_draw_count,
            "packet_stratum_counts": self.packet_stratum_counts,
            "sealed_order_start_indices": self.sealed_order_start_indices,
            "route_chunk_windows": self.route_chunk_windows,
            "observed_context_ids_by_stratum": self.observed_context_ids_by_stratum,
            "observed_draw_ids_by_stratum": self.observed_draw_ids_by_stratum,
            "signed_mean_hash": self.signed_mean_hash,
            "signed_m2_hash": self.signed_m2_hash,
            "absolute_mean_hash": self.absolute_mean_hash,
            "absolute_m2_hash": self.absolute_m2_hash,
            "peak_device_bytes": self.peak_device_bytes,
            "peak_host_bytes": self.peak_host_bytes,
            "aggregate_host_transfer_bytes": self.aggregate_host_transfer_bytes,
            "avoided_context_route_host_tensor_bytes": (
                self.avoided_context_route_host_tensor_bytes
            ),
            "retained_host_context_route_tensor_bytes": (
                self.retained_host_context_route_tensor_bytes
            ),
            "host_context_route_rows_transferred": (
                self.host_context_route_rows_transferred
            ),
            "arithmetic_identity": self.arithmetic_identity,
            "packetization_kind": self.packetization_kind,
            "fd_promotion_status": self.fd_promotion_status,
            "memory_accounting_scope": self.memory_accounting_scope,
            "snapshot_policy": self.snapshot_policy,
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        if cached is None:
            cached = content_hash(self._hash_payload(), prefix=_RECEIPT_PREFIX)
            object.__setattr__(self, "_receipt_hash", cached)
        return cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("GPU reduced packet receipt was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("GPU reduced packet receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1GPUReducedProposalPacketBoundarySnapshot:
    """Crash-safe packet boundary marker.

    The snapshot is intentionally tiny.  It records either a pre-packet
    boundary or a completed packet hash, never a partially reduced route
    prefix.
    """

    request_hash: str
    packetization_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_count: int
    sealed_order_start_indices: tuple[int, int]
    completed_packet_hash: str | None
    completed_receipt_hash: str | None
    schema: str = GPU_REDUCED_PROPOSAL_PACKET_BOUNDARY_SNAPSHOT_SCHEMA
    _snapshot_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_hash", _identity(self.request_hash, name="request hash")
        )
        object.__setattr__(
            self,
            "packetization_identity_hash",
            _identity(
                self.packetization_identity_hash, name="packetization identity hash"
            ),
        )
        object.__setattr__(
            self,
            "current_mask_hash",
            _identity(self.current_mask_hash, name="current mask hash"),
        )
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        route_count = _positive_int(self.route_count, name="route count")
        starts = _pair_of_nonnegative_ints(
            self.sealed_order_start_indices, name="sealed order start indices"
        )
        if (self.completed_packet_hash is None) != (
            self.completed_receipt_hash is None
        ):
            raise ValueError("snapshot completion hashes must appear together")
        if self.completed_packet_hash is not None:
            _identity(self.completed_packet_hash, name="completed packet hash")
            _identity(self.completed_receipt_hash, name="completed receipt hash")
        if self.schema != GPU_REDUCED_PROPOSAL_PACKET_BOUNDARY_SNAPSHOT_SCHEMA:
            raise ValueError("GPU reduced packet snapshot schema is invalid")
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "sealed_order_start_indices", starts)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_hash": self.request_hash,
            "packetization_identity_hash": self.packetization_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "route_count": self.route_count,
            "sealed_order_start_indices": self.sealed_order_start_indices,
            "completed_packet_hash": self.completed_packet_hash,
            "completed_receipt_hash": self.completed_receipt_hash,
        }

    @property
    def snapshot_hash(self) -> str:
        cached = self._snapshot_hash
        if cached is None:
            cached = content_hash(self._hash_payload(), prefix=_SNAPSHOT_PREFIX)
            object.__setattr__(self, "_snapshot_hash", cached)
        return cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_SNAPSHOT_PREFIX)
        if self._snapshot_hash is not None and observed != self._snapshot_hash:
            raise ValueError("GPU reduced packet snapshot was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("GPU reduced packet snapshot hash does not match")
        object.__setattr__(self, "_snapshot_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1GPUReducedProposalPacket:
    """Closed reduced packet consumable by the proposal moments API."""

    request: V1GPUReducedProposalPacketRequest
    signed_mean_by_stratum: np.ndarray
    signed_m2_by_stratum: np.ndarray
    absolute_mean_by_stratum: np.ndarray
    absolute_m2_by_stratum: np.ndarray
    derivation_receipt: V1GPUReducedProposalPacketDerivationReceipt
    schema: str = GPU_REDUCED_PROPOSAL_PACKET_SCHEMA
    _packet_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.request) is not V1GPUReducedProposalPacketRequest:
            raise TypeError("GPU reduced packet requires its exact request type")
        self.request.assert_integrity()
        route_count = self.request.route_count
        signed_mean = _readonly_le_f64(
            self.signed_mean_by_stratum,
            name="packet signed mean",
            shape=(2, route_count),
        )
        signed_m2 = _readonly_le_f64(
            self.signed_m2_by_stratum,
            name="packet signed M2",
            shape=(2, route_count),
        )
        absolute_mean = _readonly_le_f64(
            self.absolute_mean_by_stratum,
            name="packet absolute mean",
            shape=(2, route_count),
        )
        absolute_m2 = _readonly_le_f64(
            self.absolute_m2_by_stratum,
            name="packet absolute M2",
            shape=(2, route_count),
        )
        if np.any(signed_m2 < 0.0) or np.any(absolute_mean < 0.0) or np.any(
            absolute_m2 < 0.0
        ):
            raise ValueError("packet moment arrays violate sign constraints")
        for stratum, count in enumerate(self.request.packet_stratum_counts):
            if count == 0 and any(
                np.any(array[stratum] != 0.0)
                for array in (signed_mean, signed_m2, absolute_mean, absolute_m2)
            ):
                raise ValueError("empty packet strata must have zero moments")
            if count < 2 and (
                np.any(signed_m2[stratum] != 0.0)
                or np.any(absolute_m2[stratum] != 0.0)
            ):
                raise ValueError("packet M2 must vanish before two stratum draws")
        tolerance = 128.0 * np.finfo(np.float64).eps * np.maximum(1.0, absolute_mean)
        if np.any(np.abs(signed_mean) > absolute_mean + tolerance):
            raise ValueError("packet signed and absolute means violate Jensen's bound")
        if (
            type(self.derivation_receipt)
            is not V1GPUReducedProposalPacketDerivationReceipt
        ):
            raise TypeError("GPU reduced packet requires its exact receipt type")
        self.derivation_receipt.assert_integrity()
        if self.derivation_receipt.request_hash != self.request.request_hash:
            raise ValueError("packet receipt does not bind the request")
        if (
            self.derivation_receipt.packetization_identity_hash
            != self.request.packetization_identity_hash
        ):
            raise ValueError("packet receipt does not bind packetization")
        if (
            self.derivation_receipt.packet_stratum_counts
            != self.request.packet_stratum_counts
        ):
            raise ValueError("packet receipt has wrong stratum counts")
        if (
            self.derivation_receipt.route_count != self.request.route_count
            or self.derivation_receipt.packet_draw_count
            != self.request.packet_draw_count
            or self.derivation_receipt.sealed_order_start_indices
            != self.request.sealed_order_start_indices
        ):
            raise ValueError("packet receipt has wrong route, draw, or prefix identity")
        if (
            self.derivation_receipt.observed_context_ids_by_stratum
            != self.request.expected_context_ids_by_stratum
            or self.derivation_receipt.observed_draw_ids_by_stratum
            != self.request.expected_draw_ids_by_stratum
        ):
            raise ValueError("packet receipt ledgers differ from the request")
        cursor = 0
        for start, stop in self.derivation_receipt.route_chunk_windows:
            width = stop - start
            if (
                start != cursor
                or width <= 0
                or width > self.request.registered_route_chunk_size
                or stop > self.request.route_count
            ):
                raise ValueError(
                    "packet receipt route windows differ from packetization"
                )
            cursor = stop
        if cursor != self.request.route_count:
            raise ValueError("packet receipt route windows differ from packetization")
        forecast = self.request.forecast_memory()
        memory_checks = {
            "peak_device_bytes": forecast.peak_device_bytes,
            "peak_host_bytes": forecast.peak_host_bytes,
            "aggregate_host_transfer_bytes": forecast.aggregate_host_transfer_bytes,
            "avoided_context_route_host_tensor_bytes": (
                forecast.avoided_context_route_host_tensor_bytes
            ),
            "retained_host_context_route_tensor_bytes": 0,
        }
        for field_name, expected_value in memory_checks.items():
            if getattr(self.derivation_receipt, field_name) != expected_value:
                raise ValueError(f"packet receipt {field_name} does not replay")
        expected_hashes = {
            "signed_mean_hash": content_hash(
                signed_mean, prefix="partition-free-v1-gpu-packet-signed-mean-"
            ),
            "signed_m2_hash": content_hash(
                signed_m2, prefix="partition-free-v1-gpu-packet-signed-m2-"
            ),
            "absolute_mean_hash": content_hash(
                absolute_mean, prefix="partition-free-v1-gpu-packet-absolute-mean-"
            ),
            "absolute_m2_hash": content_hash(
                absolute_m2, prefix="partition-free-v1-gpu-packet-absolute-m2-"
            ),
        }
        for field_name, expected_hash in expected_hashes.items():
            if getattr(self.derivation_receipt, field_name) != expected_hash:
                raise ValueError(f"packet receipt {field_name} does not match")
        if self.schema != GPU_REDUCED_PROPOSAL_PACKET_SCHEMA:
            raise ValueError("GPU reduced packet schema is invalid")
        object.__setattr__(self, "signed_mean_by_stratum", signed_mean)
        object.__setattr__(self, "signed_m2_by_stratum", signed_m2)
        object.__setattr__(self, "absolute_mean_by_stratum", absolute_mean)
        object.__setattr__(self, "absolute_m2_by_stratum", absolute_m2)

    @property
    def packet_hash(self) -> str:
        cached = self._packet_hash
        if cached is None:
            cached = content_hash(self._hash_payload(), prefix=_PACKET_PREFIX)
            object.__setattr__(self, "_packet_hash", cached)
        return cached

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_hash": self.request.request_hash,
            "packetization_identity_hash": self.request.packetization_identity_hash,
            "signed_mean_by_stratum": self.signed_mean_by_stratum,
            "signed_m2_by_stratum": self.signed_m2_by_stratum,
            "absolute_mean_by_stratum": self.absolute_mean_by_stratum,
            "absolute_m2_by_stratum": self.absolute_m2_by_stratum,
            "derivation_receipt_hash": self.derivation_receipt.receipt_hash,
        }

    def _retains_canonical_immutable_arrays(self) -> bool:
        route_count = self.request.route_count
        for name in (
            "signed_mean_by_stratum",
            "signed_m2_by_stratum",
            "absolute_mean_by_stratum",
            "absolute_m2_by_stratum",
        ):
            array = getattr(self, name)
            if (
                type(array) is not np.ndarray
                or array.dtype != _LE_F64
                or array.shape != (2, route_count)
                or not array.flags.c_contiguous
                or array.flags.writeable
            ):
                return False
        return True

    def assert_integrity(
        self,
        expected_hash: str | None = None,
        *,
        trust_cached_immutable: bool = False,
    ) -> str:
        if type(trust_cached_immutable) is not bool:
            raise TypeError("cached packet trust flag must be bool")
        if trust_cached_immutable and self._packet_hash is not None:
            cached = self._packet_hash
            if expected_hash is not None and cached != expected_hash:
                raise ValueError("GPU reduced packet hash does not match")
            if not self._retains_canonical_immutable_arrays():
                raise ValueError("GPU reduced packet lost immutable array custody")
            return cached
        observed = content_hash(self._hash_payload(), prefix=_PACKET_PREFIX)
        if self._packet_hash is not None and observed != self._packet_hash:
            raise ValueError("GPU reduced packet was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("GPU reduced packet hash does not match")
        object.__setattr__(self, "_packet_hash", observed)
        return observed

    def as_reduced_stratum_moments_kwargs(self) -> dict[str, object]:
        return {
            "sealed_order_start_indices": self.request.sealed_order_start_indices,
            "packet_stratum_counts": self.request.packet_stratum_counts,
            "context_ids_by_stratum": self.request.expected_context_ids_by_stratum,
            "draw_ids_by_stratum": self.request.expected_draw_ids_by_stratum,
            "expected_context_ids_by_stratum": (
                self.request.expected_context_ids_by_stratum
            ),
            "expected_draw_ids_by_stratum": self.request.expected_draw_ids_by_stratum,
            "signed_mean_by_stratum": self.signed_mean_by_stratum,
            "signed_m2_by_stratum": self.signed_m2_by_stratum,
            "absolute_mean_by_stratum": self.absolute_mean_by_stratum,
            "absolute_m2_by_stratum": self.absolute_m2_by_stratum,
        }

    def as_initial_moments_kwargs(self) -> dict[str, object]:
        return {
            "campaign_contract_hash": self.request.campaign_contract_hash,
            "model_identity_hash": self.request.model_identity_hash,
            "query_contract_hash": self.request.query_contract_hash,
            "preprocessing_hash": self.request.preprocessing_hash,
            "current_mask_hash": self.request.current_mask_hash,
            "raw_logit_stream_source_hash": self.request.raw_logit_stream_source_hash,
            "sampling_law_hash": self.request.sampling_law_hash,
            "proposal_generation": self.request.proposal_generation,
            "route_axis_hash": self.request.route_axis_hash,
            "stratum_names": self.request.stratum_names,
            "population_stratum_counts": self.request.population_stratum_counts,
            "sealed_population_order_hashes": (
                self.request.sealed_population_order_hashes
            ),
            **self.as_reduced_stratum_moments_kwargs(),
        }

    def to_initial_moments(self) -> V1StratifiedCorrectedAttributionMoments:
        if self.request.sealed_order_start_indices != (0, 0):
            raise ValueError("initial moments require a packet starting at zero")
        return V1StratifiedCorrectedAttributionMoments.from_reduced_stratum_moments(
            **self.as_initial_moments_kwargs()
        )

    def append_to(
        self, state: V1StratifiedCorrectedAttributionMoments
    ) -> V1StratifiedCorrectedAttributionMoments:
        if type(state) is not V1StratifiedCorrectedAttributionMoments:
            raise TypeError("GPU reduced packet append requires modern-v1 moments")
        state.assert_integrity()
        self._assert_state_identity(state)
        return state.append_reduced_stratum_moments(
            **self.as_reduced_stratum_moments_kwargs()
        )

    def _assert_state_identity(
        self, state: V1StratifiedCorrectedAttributionMoments
    ) -> None:
        request = self.request
        checks: Mapping[str, tuple[object, object]] = {
            "campaign contract": (
                state.campaign_contract_hash,
                request.campaign_contract_hash,
            ),
            "model identity": (state.model_identity_hash, request.model_identity_hash),
            "query contract": (state.query_contract_hash, request.query_contract_hash),
            "preprocessing": (state.preprocessing_hash, request.preprocessing_hash),
            "current mask": (state.current_mask_hash, request.current_mask_hash),
            "raw logit stream": (
                state.raw_logit_stream_source_hash,
                request.raw_logit_stream_source_hash,
            ),
            "sampling law": (state.sampling_law_hash, request.sampling_law_hash),
            "proposal generation": (
                state.proposal_generation,
                request.proposal_generation,
            ),
            "route count": (state.route_count, request.route_count),
            "route axis": (state.route_axis_hash, request.route_axis_hash),
            "stratum names": (state.stratum_names, request.stratum_names),
            "population counts": (
                state.population_stratum_counts,
                request.population_stratum_counts,
            ),
            "sealed order hashes": (
                state.sealed_population_order_hashes,
                request.sealed_population_order_hashes,
            ),
            "sealed prefix": (
                state.stratum_counts,
                request.sealed_order_start_indices,
            ),
        }
        for name, (left, right) in checks.items():
            if left != right:
                raise ValueError(f"GPU reduced packet {name} differs from state")

    def boundary_snapshot(self) -> V1GPUReducedProposalPacketBoundarySnapshot:
        return V1GPUReducedProposalPacketBoundarySnapshot(
            request_hash=self.request.request_hash,
            packetization_identity_hash=self.request.packetization_identity_hash,
            current_mask_hash=self.request.current_mask_hash,
            proposal_generation=self.request.proposal_generation,
            route_count=self.request.route_count,
            sealed_order_start_indices=self.request.sealed_order_start_indices,
            completed_packet_hash=self.packet_hash,
            completed_receipt_hash=self.derivation_receipt.receipt_hash,
        )


class V1GPUReducedProposalPacketConsumer:
    """Device consumer for one frozen reduced packet request."""

    def __init__(
        self,
        request: V1GPUReducedProposalPacketRequest,
        *,
        device: Any | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("GPU reduced proposal packets require PyTorch")
        if type(request) is not V1GPUReducedProposalPacketRequest:
            raise TypeError("packet consumer requires its exact request type")
        request.assert_integrity()
        self.request = request
        self.device = _canonical_runtime_device("cpu" if device is None else device)
        forecast = request.forecast_memory()
        self._memory_forecast = forecast
        self._aggregate_device_bytes = int(forecast.aggregate_device_bytes)
        self._avoided_host_row_bytes = int(
            forecast.avoided_context_route_host_tensor_bytes
        )
        shape = (2, request.route_count)
        self._signed_mean = torch.zeros(
            shape, dtype=torch.float64, device=self.device
        )
        self._signed_m2 = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self._absolute_mean = torch.zeros(
            shape, dtype=torch.float64, device=self.device
        )
        self._absolute_m2 = torch.zeros(
            shape, dtype=torch.float64, device=self.device
        )
        self._next_route_start = 0
        self._route_chunk_windows: list[tuple[int, int]] = []
        self._closed_packet: V1GPUReducedProposalPacket | None = None
        self._peak_device_bytes = self._aggregate_device_bytes
        self._row_stratum_indices = tuple(
            request.stratum_names.index(label)
            for label in request.expected_stratum_labels
        )
        # Process-local performance diagnostics are deliberately absent from
        # every request, receipt, packet, and snapshot hash payload.  They
        # measure the physical implementation without becoming scientific
        # state or changing deterministic replay identities.
        self._runtime_diagnostics: dict[str, float | int] = {
            "score_reduction_host_enqueue_seconds": 0.0,
            "score_reduction_chunk_count": 0,
            "score_reduction_input_bytes": 0,
            "score_d2h_synchronization_inclusive_seconds": 0.0,
            "score_d2h_array_count": 0,
            "score_d2h_bytes": 0,
            "score_canonicalization_seconds": 0.0,
            "score_canonicalization_array_count": 0,
            "score_canonicalization_bytes": 0,
            "score_explicit_full_array_digest_seconds": 0.0,
            "score_explicit_full_array_digest_call_count": 0,
            "score_explicit_full_array_digest_bytes": 0,
            "score_receipt_construction_seconds": 0.0,
            "score_packet_construction_seconds": 0.0,
            "score_packet_integrity_seconds": 0.0,
        }

    @property
    def route_chunk_windows(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._route_chunk_windows)

    @property
    def runtime_diagnostics(self) -> Mapping[str, float | int]:
        """Return immutable, authority-inert physical execution metrics."""

        return MappingProxyType(dict(self._runtime_diagnostics))

    @property
    def is_packet_boundary(self) -> bool:
        return self._next_route_start == 0 or self._closed_packet is not None

    def append_route_chunk(
        self,
        *,
        route_start: int,
        signed_rows: Any,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        stratum_labels: Sequence[str],
    ) -> None:
        """Reduce one registered contiguous route window on the packet device."""

        if self._closed_packet is not None:
            raise RuntimeError("cannot append chunks after packet close")
        start = _nonnegative_int(route_start, name="route start")
        if start != self._next_route_start:
            raise ValueError("route chunks must arrive in registered order")
        observed_contexts = tuple(
            _identity(item, name="observed context id") for item in context_ids
        )
        observed_draws = tuple(
            _identity(item, name="observed draw id") for item in draw_ids
        )
        observed_labels = tuple(
            _identity(item, name="observed stratum label") for item in stratum_labels
        )
        if len(observed_contexts) != self.request.packet_draw_count:
            raise ValueError("observed context ledger has the wrong packet length")
        if not (
            len(observed_contexts) == len(observed_draws) == len(observed_labels)
        ):
            raise ValueError("observed packet ledgers are not aligned")
        _validate_unique_ledger(observed_contexts, name="observed context ledger")
        _validate_unique_ledger(observed_draws, name="observed draw ledger")
        if (
            observed_contexts != self.request.expected_context_ids
            or observed_draws != self.request.expected_draw_ids
            or observed_labels != self.request.expected_stratum_labels
        ):
            raise ValueError("observed chunk ledgers differ from the frozen request")
        if not isinstance(signed_rows, torch.Tensor):
            raise TypeError("route chunks must be supplied as torch tensors")
        if _canonical_runtime_device(signed_rows.device) != self.device:
            raise ValueError("route chunks must stay on the registered device")
        if signed_rows.ndim != 2:
            raise ValueError("route chunks must have [draw, route-window] shape")
        if int(signed_rows.shape[0]) != self.request.packet_draw_count:
            raise ValueError("route chunk draw axis differs from the frozen request")
        if not signed_rows.is_contiguous():
            raise ValueError("route chunks must be contiguous")
        width = int(signed_rows.shape[1])
        if (
            width <= 0
            or width > self.request.registered_route_chunk_size
            or start + width > self.request.route_count
        ):
            raise ValueError("route chunk width differs from registered packetization")
        stop = start + width
        values = signed_rows.detach()
        if values.dtype != torch.float64:
            values = values.to(dtype=torch.float64)
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("route chunk contains non-finite corrected attributions")

        self._peak_device_bytes = max(
            self._peak_device_bytes,
            self._aggregate_device_bytes
            + int(values.numel()) * int(values.element_size()),
        )
        # CUDA launches are asynchronous.  This is intentionally host-enqueue
        # wall time; the existing close-time D2H barrier is measured separately.
        reduction_started = time.perf_counter()
        self._reduce_window(route_start=start, route_stop=stop, values=values)
        self._runtime_diagnostics["score_reduction_host_enqueue_seconds"] = float(
            self._runtime_diagnostics["score_reduction_host_enqueue_seconds"]
        ) + (time.perf_counter() - reduction_started)
        self._runtime_diagnostics["score_reduction_chunk_count"] = int(
            self._runtime_diagnostics["score_reduction_chunk_count"]
        ) + 1
        self._runtime_diagnostics["score_reduction_input_bytes"] = int(
            self._runtime_diagnostics["score_reduction_input_bytes"]
        ) + int(values.numel()) * int(values.element_size())
        self._route_chunk_windows.append((start, stop))
        self._next_route_start = stop

    def _timed_array_content_hash(self, array: np.ndarray, *, prefix: str) -> str:
        started = time.perf_counter()
        observed = content_hash(array, prefix=prefix)
        self._runtime_diagnostics["score_explicit_full_array_digest_seconds"] = float(
            self._runtime_diagnostics["score_explicit_full_array_digest_seconds"]
        ) + (time.perf_counter() - started)
        self._runtime_diagnostics["score_explicit_full_array_digest_call_count"] = int(
            self._runtime_diagnostics["score_explicit_full_array_digest_call_count"]
        ) + 1
        self._runtime_diagnostics["score_explicit_full_array_digest_bytes"] = int(
            self._runtime_diagnostics["score_explicit_full_array_digest_bytes"]
        ) + int(array.nbytes)
        return observed

    def _reduce_window(self, *, route_start: int, route_stop: int, values: Any) -> None:
        signed_mean = self._signed_mean[:, route_start:route_stop]
        signed_m2 = self._signed_m2[:, route_start:route_stop]
        absolute_mean = self._absolute_mean[:, route_start:route_stop]
        absolute_m2 = self._absolute_m2[:, route_start:route_stop]
        counts = [0, 0]
        for row_index, stratum in enumerate(self._row_stratum_indices):
            counts[stratum] += 1
            count = counts[stratum]
            signed = values[row_index]
            delta = signed - signed_mean[stratum]
            signed_mean[stratum].add_(delta / count)
            signed_m2[stratum].add_(delta * (signed - signed_mean[stratum]))

            absolute = torch.abs(signed)
            abs_delta = absolute - absolute_mean[stratum]
            absolute_mean[stratum].add_(abs_delta / count)
            absolute_m2[stratum].add_(abs_delta * (absolute - absolute_mean[stratum]))

    def close(self) -> V1GPUReducedProposalPacket:
        """Close the packet, transferring only four routewise aggregate arrays."""

        if self._closed_packet is not None:
            return self._closed_packet
        if self._next_route_start != self.request.route_count:
            raise RuntimeError(
                "cannot close a partial route packet; replay from the prior boundary"
            )
        if self._peak_device_bytes != self._memory_forecast.peak_device_bytes:
            raise RuntimeError("packet device-byte ledger did not replay at close")
        signed_mean = _copy_aggregate_to_host_array(
            self._signed_mean,
            role="packet signed mean",
            route_count=self.request.route_count,
            diagnostics=self._runtime_diagnostics,
        )
        signed_m2 = _copy_aggregate_to_host_array(
            self._signed_m2,
            role="packet signed M2",
            route_count=self.request.route_count,
            diagnostics=self._runtime_diagnostics,
        )
        absolute_mean = _copy_aggregate_to_host_array(
            self._absolute_mean,
            role="packet absolute mean",
            route_count=self.request.route_count,
            diagnostics=self._runtime_diagnostics,
        )
        absolute_m2 = _copy_aggregate_to_host_array(
            self._absolute_m2,
            role="packet absolute M2",
            route_count=self.request.route_count,
            diagnostics=self._runtime_diagnostics,
        )
        aggregate_host_bytes = int(
            signed_mean.nbytes
            + signed_m2.nbytes
            + absolute_mean.nbytes
            + absolute_m2.nbytes
        )
        if aggregate_host_bytes != self._memory_forecast.aggregate_host_transfer_bytes:
            raise RuntimeError("packet host-transfer byte ledger did not replay at close")
        receipt_started = time.perf_counter()
        receipt = V1GPUReducedProposalPacketDerivationReceipt(
            request_hash=self.request.request_hash,
            packetization_identity_hash=self.request.packetization_identity_hash,
            device_identity=_device_identity(self.device),
            route_count=self.request.route_count,
            packet_draw_count=self.request.packet_draw_count,
            packet_stratum_counts=self.request.packet_stratum_counts,
            sealed_order_start_indices=self.request.sealed_order_start_indices,
            route_chunk_windows=tuple(self._route_chunk_windows),
            observed_context_ids_by_stratum=(
                self.request.expected_context_ids_by_stratum
            ),
            observed_draw_ids_by_stratum=self.request.expected_draw_ids_by_stratum,
            signed_mean_hash=self._timed_array_content_hash(
                signed_mean, prefix="partition-free-v1-gpu-packet-signed-mean-"
            ),
            signed_m2_hash=self._timed_array_content_hash(
                signed_m2, prefix="partition-free-v1-gpu-packet-signed-m2-"
            ),
            absolute_mean_hash=self._timed_array_content_hash(
                absolute_mean, prefix="partition-free-v1-gpu-packet-absolute-mean-"
            ),
            absolute_m2_hash=self._timed_array_content_hash(
                absolute_m2, prefix="partition-free-v1-gpu-packet-absolute-m2-"
            ),
            peak_device_bytes=int(self._memory_forecast.peak_device_bytes),
            peak_host_bytes=int(self._memory_forecast.peak_host_bytes),
            aggregate_host_transfer_bytes=aggregate_host_bytes,
            avoided_context_route_host_tensor_bytes=self._avoided_host_row_bytes,
        )
        self._runtime_diagnostics["score_receipt_construction_seconds"] = float(
            self._runtime_diagnostics["score_receipt_construction_seconds"]
        ) + (time.perf_counter() - receipt_started)
        packet_started = time.perf_counter()
        packet = V1GPUReducedProposalPacket(
            request=self.request,
            signed_mean_by_stratum=signed_mean,
            signed_m2_by_stratum=signed_m2,
            absolute_mean_by_stratum=absolute_mean,
            absolute_m2_by_stratum=absolute_m2,
            derivation_receipt=receipt,
        )
        self._runtime_diagnostics["score_packet_construction_seconds"] = float(
            self._runtime_diagnostics["score_packet_construction_seconds"]
        ) + (time.perf_counter() - packet_started)
        integrity_started = time.perf_counter()
        packet.assert_integrity()
        self._runtime_diagnostics["score_packet_integrity_seconds"] = float(
            self._runtime_diagnostics["score_packet_integrity_seconds"]
        ) + (time.perf_counter() - integrity_started)
        self._closed_packet = packet
        return packet

    def snapshot(self) -> V1GPUReducedProposalPacketBoundarySnapshot:
        """Return a crash-safe boundary snapshot, refusing partial packets."""

        if self._closed_packet is not None:
            return self._closed_packet.boundary_snapshot()
        if self._next_route_start != 0:
            raise RuntimeError(
                "partial packet snapshots are refused; replay from packet boundary"
            )
        return V1GPUReducedProposalPacketBoundarySnapshot(
            request_hash=self.request.request_hash,
            packetization_identity_hash=self.request.packetization_identity_hash,
            current_mask_hash=self.request.current_mask_hash,
            proposal_generation=self.request.proposal_generation,
            route_count=self.request.route_count,
            sealed_order_start_indices=self.request.sealed_order_start_indices,
            completed_packet_hash=None,
            completed_receipt_hash=None,
        )

    @classmethod
    def from_boundary_snapshot(
        cls,
        snapshot: V1GPUReducedProposalPacketBoundarySnapshot,
        request: V1GPUReducedProposalPacketRequest,
        *,
        device: Any | None = None,
    ) -> "V1GPUReducedProposalPacketConsumer":
        if type(snapshot) is not V1GPUReducedProposalPacketBoundarySnapshot:
            raise TypeError("packet restoration requires the boundary snapshot type")
        snapshot.assert_integrity()
        request.assert_integrity()
        if snapshot.completed_packet_hash is not None:
            raise ValueError("completed packet snapshots are terminal for this request")
        if snapshot.request_hash != request.request_hash:
            raise ValueError("snapshot request hash differs from the runtime request")
        if snapshot.packetization_identity_hash != request.packetization_identity_hash:
            raise ValueError("snapshot packetization differs from the runtime request")
        if snapshot.current_mask_hash != request.current_mask_hash:
            raise ValueError("snapshot mask differs from the runtime request")
        if snapshot.proposal_generation != request.proposal_generation:
            raise ValueError("snapshot generation differs from the runtime request")
        if snapshot.route_count != request.route_count:
            raise ValueError("snapshot route axis differs from the runtime request")
        if snapshot.sealed_order_start_indices != request.sealed_order_start_indices:
            raise ValueError("snapshot sealed prefix differs from the runtime request")
        return cls(request, device=device)


__all__ = [
    "GPU_PACKET_FD_PROMOTION_STATUS",
    "GPU_REDUCED_PROPOSAL_PACKET_BOUNDARY_SNAPSHOT_SCHEMA",
    "GPU_REDUCED_PROPOSAL_PACKET_MEMORY_FORECAST_SCHEMA",
    "GPU_REDUCED_PROPOSAL_PACKET_RECEIPT_SCHEMA",
    "GPU_REDUCED_PROPOSAL_PACKET_REQUEST_SCHEMA",
    "GPU_REDUCED_PROPOSAL_PACKET_SCHEMA",
    "REGISTERED_GPU_PACKET_ARITHMETIC_ID",
    "REGISTERED_GPU_PACKET_MEMORY_ACCOUNTING_SCOPE",
    "REGISTERED_GPU_PACKETIZATION_KIND",
    "V1GPUReducedProposalPacket",
    "V1GPUReducedProposalPacketBoundarySnapshot",
    "V1GPUReducedProposalPacketConsumer",
    "V1GPUReducedProposalPacketDerivationReceipt",
    "V1GPUReducedProposalPacketMemoryForecast",
    "V1GPUReducedProposalPacketRequest",
    "forecast_gpu_reduced_proposal_packet_memory",
]
