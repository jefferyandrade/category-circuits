"""Exact corrected-evidence row custody for the signed-work CUDA proposal.

The canonical proposal scorer emits corrected Aumann--Shapley evidence in
contiguous route windows.  This module is the nominal consumer that
may retain one *pre-registered* D-prefix as an exact ``[context, route]``
float64 bank.  It performs no scoring and requests no additional VJP.

Production custody is CUDA-only.  A deliberately explicit CPU reference mode
exists for unit and small-domain parity tests, but its receipt carries no
production proposal authority.  The durable receipt binds causal source
identity and exact coverage; the process-local handle additionally freezes
the tensor's storage/version invariant.  Consequently, the bank never needs
to cross the host merely to acquire a route-scale content digest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Final, NoReturn

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_native_fiber_axis import (
    V1NativeDestinationFiberAxisReceipt,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.core.axis_identity import ordered_string_axis_hash
from mechanistic_systems.measurement.optimized_irg_providers import (
    _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE,
    _RouteChunkedLossOnlyConsumer,
)
from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

try:  # pragma: no cover - availability is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SIGNED_WORK_ROW_SPOOL_REQUEST_SCHEMA: Final = (
    "partition_free_v1_signed_work_exact_d_prefix_row_spool_request_v2"
)
SIGNED_WORK_ROW_SPOOL_HBM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_signed_work_exact_d_prefix_row_spool_hbm_forecast_v2"
)
SIGNED_WORK_ROW_SPOOL_CUSTODY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_exact_d_prefix_row_spool_custody_receipt_v2"
)
SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_REQUEST_SCHEMA: Final = (
    "partition_free_v1_signed_work_exact_d_prefix_multi_packet_row_spool_request_v1"
)
SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_HBM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_signed_work_exact_d_prefix_multi_packet_row_spool_hbm_forecast_v1"
)
SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_CUSTODY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_exact_d_prefix_multi_packet_row_spool_custody_receipt_v1"
)
SIGNED_WORK_ROW_SPOOL_AUTHORITY_SCOPE: Final = (
    "signed_work_proposal_input_custody_only_no_fqr_certification_calibration_"
    "oracle_mask_commit_or_terminal_authority_v1"
)
SIGNED_WORK_ROW_SPOOL_MEMORY_SCOPE: Final = (
    "registered_current_reserved_device_bytes_plus_one_exact_float64_context_"
    "route_bank_plus_colive_provider_chunk_plus_two_float64_finiteness_scalars_"
    "excluding_additional_allocator_and_reduction_workspace_v2"
)
SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_MEMORY_SCOPE: Final = (
    "registered_current_reserved_device_bytes_plus_one_exact_float64_context_"
    "route_bank_plus_one_maximum_registered_packet_window_float64_staging_tile_"
    "plus_two_float64_finiteness_scalars_excluding_provider_model_workspace_v1"
)
CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS: Final = (
    "corrected_aumann_shapley_positive_removal_negative_G_phi_c"
)
RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE: Final = (
    _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
)

_REQUEST_PREFIX: Final = "partition-free-v1-signed-work-row-spool-request-"
_FORECAST_PREFIX: Final = "partition-free-v1-signed-work-row-spool-forecast-"
_RECEIPT_PREFIX: Final = "partition-free-v1-signed-work-row-spool-receipt-"
_CONTEXT_AXIS_PREFIX: Final = "partition-free-v1-signed-work-context-axis-"
_DRAW_AXIS_PREFIX: Final = "partition-free-v1-signed-work-draw-axis-"
_STRATUM_LEDGER_PREFIX: Final = "partition-free-v1-signed-work-stratum-ledger-"
_SOURCE_LEDGER_PREFIX: Final = "partition-free-v1-signed-work-source-ledger-"
_WEIGHT_AXIS_PREFIX: Final = "partition-free-v1-signed-work-weight-axis-"
_WINDOW_AXIS_PREFIX: Final = "partition-free-v1-signed-work-window-axis-"
_SOURCE_REPLAY_PREFIX: Final = "partition-free-v1-signed-work-source-replay-"
_MULTI_REQUEST_PREFIX: Final = (
    "partition-free-v1-signed-work-multi-packet-row-spool-request-"
)
_MULTI_FORECAST_PREFIX: Final = (
    "partition-free-v1-signed-work-multi-packet-row-spool-forecast-"
)
_MULTI_RECEIPT_PREFIX: Final = (
    "partition-free-v1-signed-work-multi-packet-row-spool-receipt-"
)
_PACKET_SLICE_PREFIX: Final = (
    "partition-free-v1-signed-work-multi-packet-context-slice-"
)
_PACKET_SLICE_AXIS_PREFIX: Final = (
    "partition-free-v1-signed-work-multi-packet-context-slice-axis-"
)
_FLOAT64_BYTES: Final = 8
_FLOAT64_FINITE_SCALAR_BYTES: Final = 2 * _FLOAT64_BYTES
_RECEIPT_ISSUANCE_TOKEN: Final = object()
_MULTI_RECEIPT_ISSUANCE_TOKEN: Final = object()


class V1SignedWorkRowSpoolExecutionMode(str, Enum):
    """Registered authority boundary for exact-row residency."""

    CUDA_PRODUCTION = "cuda_production"
    CPU_REFERENCE_TEST_ONLY = "cpu_reference_test_only"


class V1SignedWorkRowSpoolRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    AXIS_MISMATCH = "axis_mismatch"
    STALE_IDENTITY = "stale_identity"
    DUPLICATE_OR_OUT_OF_ORDER_CHUNK = "duplicate_or_out_of_order_chunk"
    PARTIAL_PACKET = "partial_packet"
    HASH_MISMATCH = "hash_mismatch"
    MEMORY_INADMISSIBLE = "memory_inadmissible"
    DEVICE_MISMATCH = "device_mismatch"
    NONFINITE_ROW = "nonfinite_row"
    AUTHORITY_VIOLATION = "authority_violation"
    CUSTODY_MUTATED = "custody_mutated"
    ACTIVE_LEASE = "active_lease"
    EVICTED = "evicted"


class V1SignedWorkRowSpoolRefusal(ValueError):
    """Typed fail-closed boundary for row-spool acquisition and custody."""

    def __init__(
        self,
        code: V1SignedWorkRowSpoolRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1SignedWorkRowSpoolRefusalCode, message: str) -> NoReturn:
    raise V1SignedWorkRowSpoolRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        _refuse(
            V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _identity_axis(
    values: Sequence[str],
    *,
    name: str,
    unique: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        _refuse(
            V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
            f"{name} must be a sequence of strings",
        )
    result = tuple(_identity(value, name=f"{name} item") for value in values)
    if not result:
        _refuse(
            V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
            f"{name} cannot be empty",
        )
    if unique and len(set(result)) != len(result):
        _refuse(
            V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
            f"{name} must be globally unique",
        )
    return result


def _context_weights(values: Sequence[float], *, count: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != count:
        _refuse(
            V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
            "context weights must align with the context axis",
        )
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (float, int, np.number)):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "context weights must be finite real scalars",
            )
        scalar = float(value)
        if not math.isfinite(scalar) or scalar < 0.0:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
                "context weights must be finite and nonnegative",
            )
        result.append(0.0 if scalar == 0.0 else scalar)
    total = math.fsum(result)
    tolerance = 128.0 * np.finfo(np.float64).eps
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
        _refuse(
            V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
            "context weights must sum to one in binary64",
        )
    return tuple(result)


def _declared_device(value: object) -> str:
    text = _identity(value, name="declared device")
    if torch is None:
        raise RuntimeError("signed-work row spooling requires PyTorch")
    device = torch.device(text)
    if device.type == "cuda" and device.index is None:
        _refuse(
            V1SignedWorkRowSpoolRefusalCode.DEVICE_MISMATCH,
            "CUDA row-spool requests must name an explicit device index",
        )
    return str(device)


def _canonical_runtime_device(value: object) -> Any:
    if torch is None:
        raise RuntimeError("signed-work row spooling requires PyTorch")
    return torch.device(value)


def _device_identity(device: Any) -> str:
    runtime = _canonical_runtime_device(device)
    if runtime.type != "cuda":
        return str(runtime)
    try:
        name = torch.cuda.get_device_name(runtime)
    except (AssertionError, RuntimeError):  # pragma: no cover - driver dependent.
        name = "cuda-device-name-unavailable"
    return f"cuda:{runtime.index}:{name}"


def _route_axis_hash(route_ids: Sequence[str], *, name: str) -> str:
    if isinstance(route_ids, (str, bytes)) or len(route_ids) <= 0:
        _refuse(
            V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
            f"{name} must be one nonempty route sequence",
        )
    try:
        return ordered_string_axis_hash(route_ids, name=name)
    except (TypeError, ValueError) as error:
        raise V1SignedWorkRowSpoolRefusal(
            V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
            f"{name} is invalid",
        ) from error


def _validate_route_windows(
    windows: Sequence[Sequence[int]],
    *,
    route_count: int,
    maximum_width: int,
) -> tuple[tuple[int, int], ...]:
    cursor = 0
    result: list[tuple[int, int]] = []
    for raw in windows:
        if isinstance(raw, (str, bytes)):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "route windows must contain integer pairs",
            )
        pair = tuple(raw)
        if len(pair) != 2:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "route windows must contain integer pairs",
            )
        start = _nonnegative_int(pair[0], name="route window start")
        stop = _positive_int(pair[1], name="route window stop")
        width = stop - start
        if (
            start != cursor
            or width <= 0
            or width > maximum_width
            or stop > route_count
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.PARTIAL_PACKET,
                "route windows do not cover the registered axis exactly once",
            )
        result.append((start, stop))
        cursor = stop
    if cursor != route_count:
        _refuse(
            V1SignedWorkRowSpoolRefusalCode.PARTIAL_PACKET,
            "route windows do not cover the registered axis exactly once",
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class V1SignedWorkRowSpoolHBMForecast:
    """Pre-outcome dense-payload forecast; construction allocates no tensor."""

    route_count: int
    context_count: int
    registered_route_chunk_size: int
    row_bank_device_bytes: int
    colive_provider_chunk_device_bytes: int
    finiteness_scalar_device_bytes: int
    joint_peak_device_bytes: int
    memory_budget_bytes: int
    admitted: bool
    current_reserved_device_bytes: int = 0
    host_context_route_bytes: int = 0
    row_bank_allocation_count: int = 1
    route_scale_temporary_device_bytes: int = 0
    forecast_constructed_without_tensor_allocation: bool = True
    memory_accounting_scope: str = SIGNED_WORK_ROW_SPOOL_MEMORY_SCOPE
    schema: str = SIGNED_WORK_ROW_SPOOL_HBM_FORECAST_SCHEMA
    _forecast_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        routes = _positive_int(self.route_count, name="forecast route count")
        contexts = _positive_int(self.context_count, name="forecast context count")
        chunk = _positive_int(
            self.registered_route_chunk_size,
            name="forecast route chunk size",
        )
        if chunk > routes:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
                "forecast route chunk exceeds route count",
            )
        current_reserved = _nonnegative_int(
            self.current_reserved_device_bytes,
            name="forecast current reserved device bytes",
        )
        bank = contexts * routes * _FLOAT64_BYTES
        provider = contexts * chunk * _FLOAT64_BYTES
        peak = (
            current_reserved
            + bank
            + provider
            + _FLOAT64_FINITE_SCALAR_BYTES
        )
        expected = {
            "row_bank_device_bytes": bank,
            "colive_provider_chunk_device_bytes": provider,
            "finiteness_scalar_device_bytes": _FLOAT64_FINITE_SCALAR_BYTES,
            "joint_peak_device_bytes": peak,
            "host_context_route_bytes": 0,
            "row_bank_allocation_count": 1,
            "route_scale_temporary_device_bytes": 0,
        }
        for name, value in expected.items():
            if _nonnegative_int(getattr(self, name), name=name) != value:
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                    f"{name} does not replay from forecast inputs",
                )
        budget = _nonnegative_int(
            self.memory_budget_bytes,
            name="row-spool memory budget bytes",
        )
        if self.admitted is not (peak <= budget):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "forecast admission does not replay from the registered budget",
            )
        if self.forecast_constructed_without_tensor_allocation is not True:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "row-spool forecast must remain allocation-free",
            )
        if self.memory_accounting_scope != SIGNED_WORK_ROW_SPOOL_MEMORY_SCOPE:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "row-spool memory-accounting scope is invalid",
            )
        if self.schema != SIGNED_WORK_ROW_SPOOL_HBM_FORECAST_SCHEMA:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "row-spool memory forecast schema is invalid",
            )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "context_count", contexts)
        object.__setattr__(self, "registered_route_chunk_size", chunk)
        object.__setattr__(
            self,
            "current_reserved_device_bytes",
            current_reserved,
        )
        object.__setattr__(self, "memory_budget_bytes", budget)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "route_count": self.route_count,
            "context_count": self.context_count,
            "registered_route_chunk_size": self.registered_route_chunk_size,
            "row_bank_device_bytes": self.row_bank_device_bytes,
            "colive_provider_chunk_device_bytes": (
                self.colive_provider_chunk_device_bytes
            ),
            "finiteness_scalar_device_bytes": self.finiteness_scalar_device_bytes,
            "joint_peak_device_bytes": self.joint_peak_device_bytes,
            "current_reserved_device_bytes": self.current_reserved_device_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "admitted": self.admitted,
            "host_context_route_bytes": self.host_context_route_bytes,
            "row_bank_allocation_count": self.row_bank_allocation_count,
            "route_scale_temporary_device_bytes": (
                self.route_scale_temporary_device_bytes
            ),
            "forecast_constructed_without_tensor_allocation": (
                self.forecast_constructed_without_tensor_allocation
            ),
            "memory_accounting_scope": self.memory_accounting_scope,
        }

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_FORECAST_PREFIX)
        if self._forecast_hash is not None and observed != self._forecast_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "row-spool memory forecast was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "row-spool memory forecast hash differs",
            )
        object.__setattr__(self, "_forecast_hash", observed)
        return observed

    def assert_admissible(self) -> None:
        self.assert_integrity()
        if not self.admitted:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.MEMORY_INADMISSIBLE,
                "exact corrected-evidence row bank exceeds its frozen budget",
            )


def forecast_v1_signed_work_row_spool_hbm(
    *,
    route_count: int,
    context_count: int,
    registered_route_chunk_size: int,
    memory_budget_bytes: int,
    current_reserved_device_bytes: int = 0,
) -> V1SignedWorkRowSpoolHBMForecast:
    """Compute the exact dense-payload upper bound without touching PyTorch."""

    routes = _positive_int(route_count, name="forecast route count")
    contexts = _positive_int(context_count, name="forecast context count")
    chunk = _positive_int(
        registered_route_chunk_size,
        name="forecast route chunk size",
    )
    if chunk > routes:
        _refuse(
            V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
            "forecast route chunk exceeds route count",
        )
    budget = _nonnegative_int(memory_budget_bytes, name="memory budget bytes")
    current_reserved = _nonnegative_int(
        current_reserved_device_bytes,
        name="current reserved device bytes",
    )
    bank = contexts * routes * _FLOAT64_BYTES
    provider = contexts * chunk * _FLOAT64_BYTES
    peak = current_reserved + bank + provider + _FLOAT64_FINITE_SCALAR_BYTES
    return V1SignedWorkRowSpoolHBMForecast(
        route_count=routes,
        context_count=contexts,
        registered_route_chunk_size=chunk,
        row_bank_device_bytes=bank,
        colive_provider_chunk_device_bytes=provider,
        finiteness_scalar_device_bytes=_FLOAT64_FINITE_SCALAR_BYTES,
        joint_peak_device_bytes=peak,
        memory_budget_bytes=budget,
        admitted=peak <= budget,
        current_reserved_device_bytes=current_reserved,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkMultiPacketRowSpoolHBMForecast:
    """Allocation-free forecast for one final bank and one packet/window tile."""

    route_count: int
    context_count: int
    packet_count: int
    maximum_packet_context_count: int
    maximum_route_window_width: int
    row_bank_device_bytes: int
    colive_packet_window_device_bytes: int
    finiteness_scalar_device_bytes: int
    joint_peak_device_bytes: int
    memory_budget_bytes: int
    admitted: bool
    current_reserved_device_bytes: int = 0
    host_context_route_bytes: int = 0
    row_bank_allocation_count: int = 1
    packet_window_staging_tile_count: int = 1
    route_scale_temporary_device_bytes: int = 0
    forecast_constructed_without_tensor_allocation: bool = True
    memory_accounting_scope: str = SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_MEMORY_SCOPE
    schema: str = SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_HBM_FORECAST_SCHEMA
    _forecast_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        routes = _positive_int(self.route_count, name="multi forecast route count")
        contexts = _positive_int(
            self.context_count,
            name="multi forecast context count",
        )
        packets = _positive_int(self.packet_count, name="multi forecast packet count")
        packet_contexts = _positive_int(
            self.maximum_packet_context_count,
            name="maximum packet context count",
        )
        window_width = _positive_int(
            self.maximum_route_window_width,
            name="maximum route-window width",
        )
        if packet_contexts > contexts or window_width > routes:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
                "multi-packet forecast dimensions exceed the final bank",
            )
        current_reserved = _nonnegative_int(
            self.current_reserved_device_bytes,
            name="multi forecast current reserved device bytes",
        )
        bank = contexts * routes * _FLOAT64_BYTES
        packet_window = packet_contexts * window_width * _FLOAT64_BYTES
        peak = (
            current_reserved
            + bank
            + packet_window
            + _FLOAT64_FINITE_SCALAR_BYTES
        )
        expected = {
            "row_bank_device_bytes": bank,
            "colive_packet_window_device_bytes": packet_window,
            "finiteness_scalar_device_bytes": _FLOAT64_FINITE_SCALAR_BYTES,
            "joint_peak_device_bytes": peak,
            "host_context_route_bytes": 0,
            "row_bank_allocation_count": 1,
            "packet_window_staging_tile_count": 1,
            "route_scale_temporary_device_bytes": 0,
        }
        for name, value in expected.items():
            if _nonnegative_int(getattr(self, name), name=name) != value:
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                    f"{name} does not replay from multi-packet forecast inputs",
                )
        budget = _nonnegative_int(
            self.memory_budget_bytes,
            name="multi-packet row-spool memory budget bytes",
        )
        if self.admitted is not (peak <= budget):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "multi-packet forecast admission does not replay",
            )
        if self.forecast_constructed_without_tensor_allocation is not True:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "multi-packet forecast must remain allocation-free",
            )
        if self.memory_accounting_scope != (
            SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_MEMORY_SCOPE
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "multi-packet memory-accounting scope is invalid",
            )
        if self.schema != SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_HBM_FORECAST_SCHEMA:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "multi-packet HBM forecast schema is invalid",
            )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "context_count", contexts)
        object.__setattr__(self, "packet_count", packets)
        object.__setattr__(self, "maximum_packet_context_count", packet_contexts)
        object.__setattr__(self, "maximum_route_window_width", window_width)
        object.__setattr__(self, "current_reserved_device_bytes", current_reserved)
        object.__setattr__(self, "memory_budget_bytes", budget)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "route_count": self.route_count,
            "context_count": self.context_count,
            "packet_count": self.packet_count,
            "maximum_packet_context_count": self.maximum_packet_context_count,
            "maximum_route_window_width": self.maximum_route_window_width,
            "row_bank_device_bytes": self.row_bank_device_bytes,
            "colive_packet_window_device_bytes": (
                self.colive_packet_window_device_bytes
            ),
            "finiteness_scalar_device_bytes": self.finiteness_scalar_device_bytes,
            "joint_peak_device_bytes": self.joint_peak_device_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "admitted": self.admitted,
            "current_reserved_device_bytes": self.current_reserved_device_bytes,
            "host_context_route_bytes": self.host_context_route_bytes,
            "row_bank_allocation_count": self.row_bank_allocation_count,
            "packet_window_staging_tile_count": (
                self.packet_window_staging_tile_count
            ),
            "route_scale_temporary_device_bytes": (
                self.route_scale_temporary_device_bytes
            ),
            "forecast_constructed_without_tensor_allocation": (
                self.forecast_constructed_without_tensor_allocation
            ),
            "memory_accounting_scope": self.memory_accounting_scope,
        }

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_MULTI_FORECAST_PREFIX)
        if self._forecast_hash is not None and observed != self._forecast_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "multi-packet row-spool forecast was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "multi-packet row-spool forecast hash differs",
            )
        object.__setattr__(self, "_forecast_hash", observed)
        return observed

    def assert_admissible(self) -> None:
        self.assert_integrity()
        if not self.admitted:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.MEMORY_INADMISSIBLE,
                "multi-packet corrected-evidence row bank exceeds its frozen budget",
            )


def forecast_v1_signed_work_multi_packet_row_spool_hbm(
    *,
    route_count: int,
    context_count: int,
    packet_count: int,
    maximum_packet_context_count: int,
    maximum_route_window_width: int,
    memory_budget_bytes: int,
    current_reserved_device_bytes: int = 0,
) -> V1SignedWorkMultiPacketRowSpoolHBMForecast:
    """Forecast one final bank plus the largest registered incoming tile."""

    routes = _positive_int(route_count, name="multi forecast route count")
    contexts = _positive_int(context_count, name="multi forecast context count")
    packets = _positive_int(packet_count, name="multi forecast packet count")
    packet_contexts = _positive_int(
        maximum_packet_context_count,
        name="maximum packet context count",
    )
    window_width = _positive_int(
        maximum_route_window_width,
        name="maximum route-window width",
    )
    if packet_contexts > contexts or window_width > routes:
        _refuse(
            V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
            "multi-packet forecast dimensions exceed the final bank",
        )
    budget = _nonnegative_int(memory_budget_bytes, name="memory budget bytes")
    current_reserved = _nonnegative_int(
        current_reserved_device_bytes,
        name="current reserved device bytes",
    )
    bank = contexts * routes * _FLOAT64_BYTES
    packet_window = packet_contexts * window_width * _FLOAT64_BYTES
    peak = current_reserved + bank + packet_window + _FLOAT64_FINITE_SCALAR_BYTES
    return V1SignedWorkMultiPacketRowSpoolHBMForecast(
        route_count=routes,
        context_count=contexts,
        packet_count=packets,
        maximum_packet_context_count=packet_contexts,
        maximum_route_window_width=window_width,
        row_bank_device_bytes=bank,
        colive_packet_window_device_bytes=packet_window,
        finiteness_scalar_device_bytes=_FLOAT64_FINITE_SCALAR_BYTES,
        joint_peak_device_bytes=peak,
        memory_budget_bytes=budget,
        admitted=peak <= budget,
        current_reserved_device_bytes=current_reserved,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkRowSpoolRequest:
    """Frozen D-prefix, chart, stream, route, and native-fiber identity."""

    route_ids: Sequence[str] = field(repr=False, compare=False)
    route_axis_hash: str
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    stratum_labels: tuple[str, ...]
    context_weights: tuple[float, ...]
    source_measurement_hashes: tuple[str, ...]
    registered_d_prefix_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    destination_transport_hash: str
    raw_logit_stream_identity_hash: str
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt = field(
        repr=False,
        compare=False,
    )
    registered_route_chunk_size: int
    memory_budget_bytes: int
    declared_device: str
    execution_mode: V1SignedWorkRowSpoolExecutionMode
    current_reserved_device_bytes: int = 0
    raw_target_logit_row_semantics: str = CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS
    positive_removal_stream_coordinate: str = (
        RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
    )
    authority_scope: str = SIGNED_WORK_ROW_SPOOL_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_ROW_SPOOL_REQUEST_SCHEMA
    _route_id_axis_hash: str = field(init=False, repr=False, compare=False)
    _request_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.native_fiber_axis_receipt) is not (
            V1NativeDestinationFiberAxisReceipt
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "row-spool request requires an exact native-fiber receipt",
            )
        self.native_fiber_axis_receipt.assert_integrity()
        route_axis_hash = _identity(self.route_axis_hash, name="route-axis hash")
        if route_axis_hash != self.native_fiber_axis_receipt.route_axis_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
                "request route axis differs from native-fiber custody",
            )
        if len(self.route_ids) != self.native_fiber_axis_receipt.route_count:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
                "request route count differs from native-fiber custody",
            )
        route_id_hash = _route_axis_hash(
            self.route_ids,
            name="signed-work row-spool route IDs",
        )
        if route_id_hash != self.native_fiber_axis_receipt.route_id_axis_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
                "request route IDs differ from native-fiber custody",
            )
        contexts = _identity_axis(
            self.context_ids,
            name="D-prefix context ledger",
            unique=True,
        )
        draws = _identity_axis(
            self.draw_ids,
            name="D-prefix draw ledger",
            unique=True,
        )
        strata = _identity_axis(
            self.stratum_labels,
            name="D-prefix stratum ledger",
            unique=False,
        )
        sources = _identity_axis(
            self.source_measurement_hashes,
            name="D-prefix source-measurement ledger",
            unique=True,
        )
        if not (len(contexts) == len(draws) == len(strata) == len(sources)):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
                "D-prefix context/draw/stratum/source ledgers must align",
            )
        weights = _context_weights(self.context_weights, count=len(contexts))
        for name in (
            "registered_d_prefix_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "destination_transport_hash",
            "raw_logit_stream_identity_hash",
        ):
            _identity(getattr(self, name), name=name.replace("_", " "))
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        chunk = _positive_int(
            self.registered_route_chunk_size,
            name="registered route chunk size",
        )
        if chunk > len(self.route_ids):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
                "registered route chunk exceeds the route axis",
            )
        budget = _nonnegative_int(
            self.memory_budget_bytes,
            name="row-spool memory budget bytes",
        )
        current_reserved = _nonnegative_int(
            self.current_reserved_device_bytes,
            name="row-spool current reserved device bytes",
        )
        device = _declared_device(self.declared_device)
        if type(self.execution_mode) is not V1SignedWorkRowSpoolExecutionMode:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "row-spool execution mode must be exact",
            )
        runtime_device = _canonical_runtime_device(device)
        if self.execution_mode is V1SignedWorkRowSpoolExecutionMode.CUDA_PRODUCTION:
            if runtime_device.type != "cuda":
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                    "production signed-work row custody is CUDA-only",
                )
        elif runtime_device.type != "cpu" or device != "cpu":
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "CPU reference mode requires the exact CPU device",
            )
        if self.raw_target_logit_row_semantics != (
            CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "row spool accepts only corrected Aumann--Shapley evidence",
            )
        if self.positive_removal_stream_coordinate != (
            RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "row spool accepts only raw-target-logit positive removal rows",
            )
        if self.authority_scope != SIGNED_WORK_ROW_SPOOL_AUTHORITY_SCOPE:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "row-spool authority scope is invalid",
            )
        if self.schema != SIGNED_WORK_ROW_SPOOL_REQUEST_SCHEMA:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "row-spool request schema is invalid",
            )
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "stratum_labels", strata)
        object.__setattr__(self, "source_measurement_hashes", sources)
        object.__setattr__(self, "context_weights", weights)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "registered_route_chunk_size", chunk)
        object.__setattr__(self, "memory_budget_bytes", budget)
        object.__setattr__(
            self,
            "current_reserved_device_bytes",
            current_reserved,
        )
        object.__setattr__(self, "declared_device", device)
        object.__setattr__(self, "_route_id_axis_hash", route_id_hash)
        self.assert_integrity()

    @property
    def route_count(self) -> int:
        return len(self.route_ids)

    @property
    def context_count(self) -> int:
        return len(self.context_ids)

    @property
    def route_id_axis_hash(self) -> str:
        return self._route_id_axis_hash

    @property
    def native_fiber_axis_receipt_hash(self) -> str:
        return self.native_fiber_axis_receipt.receipt_hash

    @property
    def native_fiber_measure_hash(self) -> str:
        return self.native_fiber_axis_receipt.measure_hash

    @property
    def memory_forecast(self) -> V1SignedWorkRowSpoolHBMForecast:
        return forecast_v1_signed_work_row_spool_hbm(
            route_count=self.route_count,
            context_count=self.context_count,
            registered_route_chunk_size=self.registered_route_chunk_size,
            memory_budget_bytes=self.memory_budget_bytes,
            current_reserved_device_bytes=self.current_reserved_device_bytes,
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "route_axis_hash": self.route_axis_hash,
            "route_id_axis_hash": self._route_id_axis_hash,
            "route_count": self.route_count,
            "context_ids": self.context_ids,
            "draw_ids": self.draw_ids,
            "stratum_labels": self.stratum_labels,
            "context_weights": self.context_weights,
            "source_measurement_hashes": self.source_measurement_hashes,
            "registered_d_prefix_hash": self.registered_d_prefix_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "destination_transport_hash": self.destination_transport_hash,
            "raw_logit_stream_identity_hash": self.raw_logit_stream_identity_hash,
            "native_fiber_axis_receipt_hash": (
                self.native_fiber_axis_receipt_hash
            ),
            "native_fiber_measure_hash": self.native_fiber_measure_hash,
            "registered_route_chunk_size": self.registered_route_chunk_size,
            "current_reserved_device_bytes": self.current_reserved_device_bytes,
            "memory_forecast_hash": self.memory_forecast.forecast_hash,
            "declared_device": self.declared_device,
            "execution_mode": self.execution_mode.value,
            "raw_target_logit_row_semantics": self.raw_target_logit_row_semantics,
            "positive_removal_stream_coordinate": (
                self.positive_removal_stream_coordinate
            ),
            "authority_scope": self.authority_scope,
        }

    @property
    def request_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.native_fiber_axis_receipt.assert_integrity()
        observed_route_hash = _route_axis_hash(
            self.route_ids,
            name="signed-work row-spool route IDs",
        )
        if (
            observed_route_hash != self._route_id_axis_hash
            or observed_route_hash
            != self.native_fiber_axis_receipt.route_id_axis_hash
            or self.route_axis_hash
            != self.native_fiber_axis_receipt.route_axis_hash
            or self.route_count != self.native_fiber_axis_receipt.route_count
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
                "row-spool request route custody became stale",
            )
        observed = content_hash(self._hash_payload(), prefix=_REQUEST_PREFIX)
        if self._request_hash is not None and observed != self._request_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "row-spool request was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "row-spool request hash differs",
            )
        object.__setattr__(self, "_request_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SignedWorkRowSpoolContextPacketSlice:
    """One preregistered packet's disjoint slice of the final context axis."""

    packet_ordinal: int
    packet_identity_hash: str
    context_start: int
    context_stop: int
    context_ids: tuple[str, ...]
    _slice_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        ordinal = _nonnegative_int(self.packet_ordinal, name="packet ordinal")
        identity = _identity(
            self.packet_identity_hash,
            name="packet identity hash",
        )
        start = _nonnegative_int(self.context_start, name="packet context start")
        stop = _positive_int(self.context_stop, name="packet context stop")
        contexts = _identity_axis(
            self.context_ids,
            name="packet context IDs",
            unique=True,
        )
        if stop <= start or stop - start != len(contexts):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
                "packet context slice does not match its registered context IDs",
            )
        object.__setattr__(self, "packet_ordinal", ordinal)
        object.__setattr__(self, "packet_identity_hash", identity)
        object.__setattr__(self, "context_start", start)
        object.__setattr__(self, "context_stop", stop)
        object.__setattr__(self, "context_ids", contexts)
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        return {
            "packet_ordinal": self.packet_ordinal,
            "packet_identity_hash": self.packet_identity_hash,
            "context_start": self.context_start,
            "context_stop": self.context_stop,
            "context_ids": self.context_ids,
        }

    @property
    def slice_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_PACKET_SLICE_PREFIX)
        if self._slice_hash is not None and observed != self._slice_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "packet context slice was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "packet context slice hash differs",
            )
        object.__setattr__(self, "_slice_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SignedWorkMultiPacketRowSpoolRequest:
    """Frozen source packet slices and source-native route-window geometry."""

    source_request: V1SignedWorkRowSpoolRequest = field(repr=False, compare=False)
    packet_slices: tuple[V1SignedWorkRowSpoolContextPacketSlice, ...]
    registered_route_windows: tuple[tuple[int, int], ...]
    schema: str = SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_REQUEST_SCHEMA
    _request_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.source_request) is not V1SignedWorkRowSpoolRequest:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "multi-packet row spool requires its exact source request",
            )
        self.source_request.assert_integrity()
        slices = tuple(self.packet_slices)
        if not slices or any(
            type(item) is not V1SignedWorkRowSpoolContextPacketSlice
            for item in slices
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "multi-packet row spool requires exact packet-slice records",
            )
        cursor = 0
        for ordinal, item in enumerate(slices):
            item.assert_integrity()
            if (
                item.packet_ordinal != ordinal
                or item.context_start != cursor
                or item.context_ids
                != self.source_request.context_ids[
                    item.context_start : item.context_stop
                ]
            ):
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
                    "packet slices must preregister one ordered disjoint context cover",
                )
            cursor = item.context_stop
        if cursor != self.source_request.context_count:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.PARTIAL_PACKET,
                "packet slices do not cover the complete registered context axis",
            )
        windows = _validate_route_windows(
            self.registered_route_windows,
            route_count=self.source_request.route_count,
            maximum_width=self.source_request.registered_route_chunk_size,
        )
        if self.schema != SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_REQUEST_SCHEMA:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "multi-packet row-spool request schema is invalid",
            )
        object.__setattr__(self, "packet_slices", slices)
        object.__setattr__(self, "registered_route_windows", windows)
        self.assert_integrity()

    @property
    def route_count(self) -> int:
        return self.source_request.route_count

    @property
    def context_count(self) -> int:
        return self.source_request.context_count

    @property
    def packet_count(self) -> int:
        return len(self.packet_slices)

    @property
    def maximum_packet_context_count(self) -> int:
        return max(item.context_stop - item.context_start for item in self.packet_slices)

    @property
    def maximum_route_window_width(self) -> int:
        return max(stop - start for start, stop in self.registered_route_windows)

    @property
    def packet_slice_axis_hash(self) -> str:
        return content_hash(
            tuple(item.slice_hash for item in self.packet_slices),
            prefix=_PACKET_SLICE_AXIS_PREFIX,
        )

    @property
    def route_window_axis_hash(self) -> str:
        return content_hash(
            self.registered_route_windows,
            prefix=_WINDOW_AXIS_PREFIX,
        )

    @property
    def memory_forecast(self) -> V1SignedWorkMultiPacketRowSpoolHBMForecast:
        return forecast_v1_signed_work_multi_packet_row_spool_hbm(
            route_count=self.route_count,
            context_count=self.context_count,
            packet_count=self.packet_count,
            maximum_packet_context_count=self.maximum_packet_context_count,
            maximum_route_window_width=self.maximum_route_window_width,
            memory_budget_bytes=self.source_request.memory_budget_bytes,
            current_reserved_device_bytes=(
                self.source_request.current_reserved_device_bytes
            ),
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_request_hash": self.source_request.request_hash,
            "packet_slice_axis_hash": self.packet_slice_axis_hash,
            "packet_slice_hashes": tuple(
                item.slice_hash for item in self.packet_slices
            ),
            "route_window_axis_hash": self.route_window_axis_hash,
            "registered_route_windows": self.registered_route_windows,
            "memory_forecast_hash": self.memory_forecast.forecast_hash,
        }

    @property
    def request_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.source_request.assert_integrity()
        for item in self.packet_slices:
            item.assert_integrity()
        observed = content_hash(self._hash_payload(), prefix=_MULTI_REQUEST_PREFIX)
        if self._request_hash is not None and observed != self._request_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "multi-packet row-spool request was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "multi-packet row-spool request hash differs",
            )
        object.__setattr__(self, "_request_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SignedWorkRowSpoolCustodyReceipt:
    """Durable causal receipt; the route-scale tensor remains private."""

    request_hash: str
    memory_forecast_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    context_axis_hash: str
    draw_axis_hash: str
    stratum_ledger_hash: str
    source_measurement_ledger_hash: str
    context_weight_axis_hash: str
    registered_d_prefix_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    destination_transport_hash: str
    raw_logit_stream_identity_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    route_count: int
    context_count: int
    registered_route_chunk_size: int
    route_windows: tuple[tuple[int, int], ...]
    row_bank_shape: tuple[int, int]
    row_bank_dtype: str
    row_bank_layout: str
    row_bank_tensor_payload_bytes: int
    declared_device: str
    device_identity: str
    execution_mode: V1SignedWorkRowSpoolExecutionMode
    source_replay_identity_hash: str
    complete_route_coverage: bool = True
    complete_context_coverage: bool = True
    every_context_route_cell_written_exactly_once: bool = True
    row_bank_allocated_before_packet_open: bool = True
    row_bank_allocation_count: int = 1
    host_context_route_rows_transferred: bool = False
    host_context_route_row_transfer_count: int = 0
    host_context_route_row_transfer_bytes: int = 0
    row_payload_cryptographic_hash_materialized: bool = False
    source_lineage_and_storage_version_custody: bool = True
    production_cuda_proposal_input_authority: bool = True
    cpu_reference_test_only: bool = False
    raw_target_logit_row_semantics: str = CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS
    positive_removal_stream_coordinate: str = (
        RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
    )
    authority_scope: str = SIGNED_WORK_ROW_SPOOL_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_ROW_SPOOL_CUSTODY_RECEIPT_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _RECEIPT_ISSUANCE_TOKEN:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "row-spool custody receipts are issued only at exact close",
            )
        routes = _positive_int(self.route_count, name="receipt route count")
        contexts = _positive_int(self.context_count, name="receipt context count")
        chunk = _positive_int(
            self.registered_route_chunk_size,
            name="receipt route chunk size",
        )
        windows = _validate_route_windows(
            self.route_windows,
            route_count=routes,
            maximum_width=chunk,
        )
        if tuple(self.row_bank_shape) != (contexts, routes):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
                "row-bank shape differs from the registered D and route axes",
            )
        if self.row_bank_dtype != "torch.float64" or self.row_bank_layout != (
            "c_contiguous_context_major_route_minor"
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AXIS_MISMATCH,
                "row-bank tensor representation is not canonical",
            )
        if self.row_bank_tensor_payload_bytes != contexts * routes * _FLOAT64_BYTES:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "row-bank tensor byte count does not replay",
            )
        exact_true_flags = (
            self.complete_route_coverage,
            self.complete_context_coverage,
            self.every_context_route_cell_written_exactly_once,
            self.row_bank_allocated_before_packet_open,
            self.source_lineage_and_storage_version_custody,
        )
        if any(type(value) is not bool or value is not True for value in exact_true_flags):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "row-spool close did not prove complete exact custody",
            )
        if (
            self.row_bank_allocation_count != 1
            or self.host_context_route_rows_transferred is not False
            or self.host_context_route_row_transfer_count != 0
            or self.host_context_route_row_transfer_bytes != 0
            or self.row_payload_cryptographic_hash_materialized is not False
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "row-spool custody crossed its zero-host-row boundary",
            )
        if type(self.execution_mode) is not V1SignedWorkRowSpoolExecutionMode:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "receipt execution mode must be exact",
            )
        production = self.execution_mode is (
            V1SignedWorkRowSpoolExecutionMode.CUDA_PRODUCTION
        )
        if (
            self.production_cuda_proposal_input_authority is not production
            or self.cpu_reference_test_only is production
            or (production and not self.declared_device.startswith("cuda:"))
            or (not production and self.declared_device != "cpu")
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "receipt authority does not match its execution mode",
            )
        if (
            self.raw_target_logit_row_semantics
            != CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS
            or self.positive_removal_stream_coordinate
            != RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
            or self.authority_scope != SIGNED_WORK_ROW_SPOOL_AUTHORITY_SCOPE
            or self.schema != SIGNED_WORK_ROW_SPOOL_CUSTODY_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "row-spool custody receipt changed semantics or authority",
            )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "context_count", contexts)
        object.__setattr__(self, "registered_route_chunk_size", chunk)
        object.__setattr__(self, "route_windows", windows)
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "request_hash": self.request_hash,
            "memory_forecast_hash": self.memory_forecast_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_id_axis_hash": self.route_id_axis_hash,
            "context_axis_hash": self.context_axis_hash,
            "draw_axis_hash": self.draw_axis_hash,
            "stratum_ledger_hash": self.stratum_ledger_hash,
            "source_measurement_ledger_hash": (
                self.source_measurement_ledger_hash
            ),
            "context_weight_axis_hash": self.context_weight_axis_hash,
            "registered_d_prefix_hash": self.registered_d_prefix_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "destination_transport_hash": self.destination_transport_hash,
            "raw_logit_stream_identity_hash": self.raw_logit_stream_identity_hash,
            "native_fiber_axis_receipt_hash": (
                self.native_fiber_axis_receipt_hash
            ),
            "native_fiber_measure_hash": self.native_fiber_measure_hash,
            "route_count": self.route_count,
            "context_count": self.context_count,
            "registered_route_chunk_size": self.registered_route_chunk_size,
            "route_windows": self.route_windows,
            "row_bank_shape": self.row_bank_shape,
            "row_bank_dtype": self.row_bank_dtype,
            "row_bank_layout": self.row_bank_layout,
            "row_bank_tensor_payload_bytes": self.row_bank_tensor_payload_bytes,
            "declared_device": self.declared_device,
            "device_identity": self.device_identity,
            "execution_mode": self.execution_mode.value,
            "source_replay_identity_hash": self.source_replay_identity_hash,
            "complete_route_coverage": self.complete_route_coverage,
            "complete_context_coverage": self.complete_context_coverage,
            "every_context_route_cell_written_exactly_once": (
                self.every_context_route_cell_written_exactly_once
            ),
            "row_bank_allocated_before_packet_open": (
                self.row_bank_allocated_before_packet_open
            ),
            "row_bank_allocation_count": self.row_bank_allocation_count,
            "host_context_route_rows_transferred": (
                self.host_context_route_rows_transferred
            ),
            "host_context_route_row_transfer_count": (
                self.host_context_route_row_transfer_count
            ),
            "host_context_route_row_transfer_bytes": (
                self.host_context_route_row_transfer_bytes
            ),
            "row_payload_cryptographic_hash_materialized": (
                self.row_payload_cryptographic_hash_materialized
            ),
            "source_lineage_and_storage_version_custody": (
                self.source_lineage_and_storage_version_custody
            ),
            "production_cuda_proposal_input_authority": (
                self.production_cuda_proposal_input_authority
            ),
            "cpu_reference_test_only": self.cpu_reference_test_only,
            "raw_target_logit_row_semantics": self.raw_target_logit_row_semantics,
            "positive_removal_stream_coordinate": (
                self.positive_removal_stream_coordinate
            ),
            "authority_scope": self.authority_scope,
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "row-spool custody receipt was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "row-spool custody receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SignedWorkMultiPacketRowSpoolCustodyReceipt:
    """Exact-close proof for packet-major writes into one final row bank."""

    request_hash: str
    source_request_hash: str
    legacy_custody_receipt_hash: str
    memory_forecast_hash: str
    packet_slice_axis_hash: str
    route_window_axis_hash: str
    packet_identity_axis_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    registered_d_prefix_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    raw_logit_stream_identity_hash: str
    native_fiber_axis_receipt_hash: str
    source_replay_identity_hash: str
    packet_count: int
    context_count: int
    route_count: int
    route_window_count: int
    packet_route_window_write_count: int
    context_route_cell_write_count: int
    finiteness_scalar_transfer_count: int
    row_bank_tensor_payload_bytes: int
    declared_device: str
    execution_mode: V1SignedWorkRowSpoolExecutionMode
    complete_packet_slice_coverage: bool = True
    complete_source_native_route_window_coverage: bool = True
    every_context_route_cell_written_exactly_once: bool = True
    one_final_row_bank_allocation: bool = True
    maximum_live_packet_count: int = 1
    maximum_live_staging_tile_count: int = 1
    host_context_route_rows_transferred: bool = False
    host_context_route_row_transfer_count: int = 0
    host_context_route_row_transfer_bytes: int = 0
    row_payload_cryptographic_hash_materialized: bool = False
    route_scale_temporary_device_bytes: int = 0
    queue_count: int = 0
    worker_thread_count: int = 0
    production_cuda_proposal_input_authority: bool = True
    cpu_reference_test_only: bool = False
    fq_certification_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    validation_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_ROW_SPOOL_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_CUSTODY_RECEIPT_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _MULTI_RECEIPT_ISSUANCE_TOKEN:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "multi-packet custody receipts are issued only at exact close",
            )
        for name in (
            "request_hash",
            "source_request_hash",
            "legacy_custody_receipt_hash",
            "memory_forecast_hash",
            "packet_slice_axis_hash",
            "route_window_axis_hash",
            "packet_identity_axis_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "raw_logit_stream_identity_hash",
            "native_fiber_axis_receipt_hash",
            "source_replay_identity_hash",
        ):
            _identity(getattr(self, name), name=name.replace("_", " "))
        packets = _positive_int(self.packet_count, name="receipt packet count")
        contexts = _positive_int(self.context_count, name="receipt context count")
        routes = _positive_int(self.route_count, name="receipt route count")
        windows = _positive_int(
            self.route_window_count,
            name="receipt route-window count",
        )
        expected_updates = packets * windows
        expected_cells = contexts * routes
        if (
            _positive_int(
                self.packet_route_window_write_count,
                name="packet route-window write count",
            )
            != expected_updates
            or _positive_int(
                self.context_route_cell_write_count,
                name="context-route cell write count",
            )
            != expected_cells
            or _nonnegative_int(
                self.finiteness_scalar_transfer_count,
                name="finiteness scalar transfer count",
            )
            != 2 * expected_updates
            or _positive_int(
                self.row_bank_tensor_payload_bytes,
                name="row-bank tensor payload bytes",
            )
            != expected_cells * _FLOAT64_BYTES
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "multi-packet exact-close counts do not replay",
            )
        exact_true = (
            self.complete_packet_slice_coverage,
            self.complete_source_native_route_window_coverage,
            self.every_context_route_cell_written_exactly_once,
            self.one_final_row_bank_allocation,
        )
        exact_false = (
            self.host_context_route_rows_transferred,
            self.row_payload_cryptographic_hash_materialized,
            self.fq_certification_authority,
            self.calibration_authority,
            self.bce_oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.validation_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in exact_true)
            or any(type(value) is not bool or value is not False for value in exact_false)
            or self.maximum_live_packet_count != 1
            or self.maximum_live_staging_tile_count != 1
            or self.host_context_route_row_transfer_count != 0
            or self.host_context_route_row_transfer_bytes != 0
            or self.route_scale_temporary_device_bytes != 0
            or self.queue_count != 0
            or self.worker_thread_count != 0
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "multi-packet custody crossed its bounded-memory authority",
            )
        if type(self.execution_mode) is not V1SignedWorkRowSpoolExecutionMode:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "multi-packet receipt execution mode must be exact",
            )
        production = self.execution_mode is (
            V1SignedWorkRowSpoolExecutionMode.CUDA_PRODUCTION
        )
        if (
            self.production_cuda_proposal_input_authority is not production
            or self.cpu_reference_test_only is production
            or (production and not self.declared_device.startswith("cuda:"))
            or (not production and self.declared_device != "cpu")
            or self.authority_scope != SIGNED_WORK_ROW_SPOOL_AUTHORITY_SCOPE
            or self.schema
            != SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_CUSTODY_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "multi-packet receipt authority does not match execution mode",
            )
        object.__setattr__(self, "packet_count", packets)
        object.__setattr__(self, "context_count", contexts)
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "route_window_count", windows)
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_MULTI_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "multi-packet custody receipt was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.HASH_MISMATCH,
                "multi-packet custody receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


def _tensor_storage_invariant(value: Any) -> tuple[object, ...]:
    storage = value.untyped_storage()
    return (
        str(value.device),
        str(value.dtype),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        int(value.storage_offset()),
        int(value.data_ptr()),
        int(storage.data_ptr()),
        int(value._version),
        bool(value.requires_grad),
        bool(value.is_contiguous()),
    )


class V1SignedWorkRowBankLease:
    """Narrow zero-copy borrow for the future CUDA signed-work operator."""

    __slots__ = ("_handle", "_storage_invariant", "_closed")

    def __init__(self, handle: "V1SignedWorkRowBankDeviceHandle") -> None:
        self._handle = handle
        self._storage_invariant = handle._acquire_lease()
        self._closed = False

    @property
    def rows(self) -> Any:
        if self._closed:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.EVICTED,
                "row-bank lease is already closed",
            )
        return self._handle._leased_rows(self._storage_invariant)

    @property
    def receipt(self) -> V1SignedWorkRowSpoolCustodyReceipt:
        if self._closed:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.EVICTED,
                "row-bank lease is already closed",
            )
        return self._handle.receipt

    @property
    def context_weights(self) -> tuple[float, ...]:
        if self._closed:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.EVICTED,
                "row-bank lease is already closed",
            )
        return self._handle.context_weights

    @property
    def native_fiber_axis_receipt(self) -> V1NativeDestinationFiberAxisReceipt:
        if self._closed:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.EVICTED,
                "row-bank lease is already closed",
            )
        return self._handle.native_fiber_axis_receipt

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._handle._release_lease(self._storage_invariant)

    def __enter__(self) -> "V1SignedWorkRowBankLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("row-bank leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("row-bank leases are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("row-bank leases cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("row-bank leases cannot be copied")


class V1SignedWorkRowBankDeviceHandle:
    """Sole process-local owner of the completed exact corrected row bank."""

    __slots__ = (
        "_tensor",
        "_receipt",
        "_context_weights",
        "_native_fiber_axis_receipt",
        "_storage_invariant",
        "_active_lease_count",
        "_integrity_failed",
    )

    def __init__(
        self,
        *,
        tensor: Any,
        receipt: V1SignedWorkRowSpoolCustodyReceipt,
        context_weights: tuple[float, ...],
        native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    ) -> None:
        if torch is None or not torch.is_tensor(tensor):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "row-bank handle requires one torch tensor",
            )
        if type(receipt) is not V1SignedWorkRowSpoolCustodyReceipt:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "row-bank handle requires its exact custody receipt",
            )
        if type(native_fiber_axis_receipt) is not (
            V1NativeDestinationFiberAxisReceipt
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "row-bank handle requires exact native-fiber custody",
            )
        receipt.assert_integrity()
        native_fiber_axis_receipt.assert_integrity()
        if (
            tensor.dtype != torch.float64
            or tuple(tensor.shape) != receipt.row_bank_shape
            or not tensor.is_contiguous()
            or tensor.requires_grad
            or str(tensor.device) != receipt.declared_device
            or receipt.native_fiber_axis_receipt_hash
            != native_fiber_axis_receipt.receipt_hash
            or len(context_weights) != receipt.context_count
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.CUSTODY_MUTATED,
                "row-bank tensor or metadata differs from its close receipt",
            )
        self._tensor = tensor
        self._receipt = receipt
        self._context_weights = tuple(context_weights)
        self._native_fiber_axis_receipt = native_fiber_axis_receipt
        self._storage_invariant = _tensor_storage_invariant(tensor)
        self._active_lease_count = 0
        self._integrity_failed = False

    def _require_live(self) -> Any:
        if self._tensor is None:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.EVICTED,
                "signed-work row-bank custody was evicted",
            )
        if self._integrity_failed:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.CUSTODY_MUTATED,
                "signed-work row-bank custody previously detected mutation",
            )
        self._receipt.assert_integrity()
        self._native_fiber_axis_receipt.assert_integrity()
        if _tensor_storage_invariant(self._tensor) != self._storage_invariant:
            self._integrity_failed = True
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.CUSTODY_MUTATED,
                "signed-work row-bank storage or tensor version changed",
            )
        return self._tensor

    @property
    def receipt(self) -> V1SignedWorkRowSpoolCustodyReceipt:
        self._require_live()
        return self._receipt

    @property
    def context_weights(self) -> tuple[float, ...]:
        self._require_live()
        return self._context_weights

    @property
    def native_fiber_axis_receipt(self) -> V1NativeDestinationFiberAxisReceipt:
        self._require_live()
        return self._native_fiber_axis_receipt

    @property
    def active_lease_count(self) -> int:
        return self._active_lease_count

    @property
    def retained_tensor_bytes(self) -> int:
        tensor = self._require_live()
        return int(tensor.numel()) * int(tensor.element_size())

    @property
    def is_evicted(self) -> bool:
        return self._tensor is None

    def borrow_for_signed_work_operator(
        self,
        *,
        expected_custody_receipt_hash: str,
        expected_registered_d_prefix_hash: str,
        expected_chart_identity_hash: str,
        expected_raw_logit_stream_identity_hash: str,
        expected_native_fiber_axis_receipt_hash: str,
        expected_route_axis_hash: str,
    ) -> V1SignedWorkRowBankLease:
        receipt = self.receipt
        expected = {
            "custody receipt hash": (
                expected_custody_receipt_hash,
                receipt.receipt_hash,
            ),
            "registered D-prefix hash": (
                expected_registered_d_prefix_hash,
                receipt.registered_d_prefix_hash,
            ),
            "chart identity hash": (
                expected_chart_identity_hash,
                receipt.chart_identity_hash,
            ),
            "raw-logit stream identity hash": (
                expected_raw_logit_stream_identity_hash,
                receipt.raw_logit_stream_identity_hash,
            ),
            "native-fiber receipt hash": (
                expected_native_fiber_axis_receipt_hash,
                receipt.native_fiber_axis_receipt_hash,
            ),
            "route-axis hash": (
                expected_route_axis_hash,
                receipt.route_axis_hash,
            ),
        }
        for name, (observed, registered) in expected.items():
            if _identity(observed, name=f"expected {name}") != registered:
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.STALE_IDENTITY,
                    f"signed-work operator requested stale {name}",
                )
        return V1SignedWorkRowBankLease(self)

    def _acquire_lease(self) -> tuple[object, ...]:
        self._require_live()
        self._active_lease_count += 1
        return self._storage_invariant

    def _leased_rows(self, invariant: tuple[object, ...]) -> Any:
        tensor = self._require_live()
        if invariant != self._storage_invariant:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.CUSTODY_MUTATED,
                "row-bank lease belongs to stale storage",
            )
        return tensor

    def _release_lease(self, invariant: tuple[object, ...]) -> None:
        if self._active_lease_count <= 0:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.CUSTODY_MUTATED,
                "row-bank lease ledger underflowed",
            )
        self._active_lease_count -= 1
        if (
            self._tensor is None
            or invariant != self._storage_invariant
            or _tensor_storage_invariant(self._tensor) != self._storage_invariant
        ):
            self._integrity_failed = True
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.CUSTODY_MUTATED,
                "signed-work operator mutated its borrowed exact row bank",
            )

    def evict(self) -> None:
        if self._active_lease_count:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.ACTIVE_LEASE,
                "cannot evict signed-work row custody while a lease is active",
            )
        self._tensor = None
        self._storage_invariant = ()

    def __reduce__(self) -> NoReturn:
        raise TypeError("row-bank device handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("row-bank device handles are nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("row-bank device handles are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("row-bank device handles cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("row-bank device handles cannot be copied")


def _live_row_spool_hbm_admitted(forecast: Any, device: Any) -> bool:
    allocated = int(torch.cuda.memory_allocated(device))
    incremental = forecast.joint_peak_device_bytes - forecast.current_reserved_device_bytes
    return (
        allocated <= forecast.current_reserved_device_bytes
        and allocated + incremental <= forecast.memory_budget_bytes
        and incremental <= cuda_reusable_memory_bytes(device, framework=torch)
    )


class V1SignedWorkRowSpoolCUDAConsumer(_RouteChunkedLossOnlyConsumer):
    """Nominal one-packet sink for an exact corrected-evidence D-prefix."""

    positive_removal_stream_coordinate = RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE

    def __init__(self, request: V1SignedWorkRowSpoolRequest) -> None:
        if torch is None:
            raise RuntimeError("signed-work row spooling requires PyTorch")
        if type(request) is not V1SignedWorkRowSpoolRequest:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "row-spool consumer requires its exact request",
            )
        request.assert_integrity()
        forecast = request.memory_forecast
        forecast.assert_admissible()
        device = _canonical_runtime_device(request.declared_device)
        if request.execution_mode is V1SignedWorkRowSpoolExecutionMode.CUDA_PRODUCTION:
            if not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.DEVICE_MISMATCH,
                    "CUDA production row spooling requires an available CUDA runtime",
                )
            try:
                torch.cuda.get_device_properties(device)
            except (AssertionError, RuntimeError) as error:
                raise V1SignedWorkRowSpoolRefusal(
                    V1SignedWorkRowSpoolRefusalCode.DEVICE_MISMATCH,
                    "declared CUDA row-spool device is unavailable",
                ) from error
            if not _live_row_spool_hbm_admitted(forecast, device):
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.MEMORY_INADMISSIBLE,
                    "live CUDA allocation exceeds the pre-allocation occupancy "
                    "bound frozen into the row-spool request",
                )

        # The forecast is complete and admitted before this sole bank
        # allocation.  Provider chunks are co-live inputs, never allocations
        # owned by this consumer.
        row_bank = torch.empty(
            (request.context_count, request.route_count),
            dtype=torch.float64,
            device=device,
        )
        if not row_bank.is_contiguous() or row_bank.requires_grad:
            raise RuntimeError("row-spool allocator did not produce canonical storage")
        if (
            int(row_bank.numel()) * int(row_bank.element_size())
            != forecast.row_bank_device_bytes
        ):
            raise RuntimeError("row-spool allocation differs from its frozen forecast")

        self.request = request
        self.route_ids = request.route_ids
        self.device = device
        self.loss_pullback_hash = request.raw_logit_stream_identity_hash
        self.destination_transport_hash = request.destination_transport_hash
        self.head_ids = ("positive_removal:raw_target_logit",)
        self.chunk_size = request.registered_route_chunk_size
        self.memory_forecast = forecast
        self._row_bank: Any | None = row_bank
        self._route_chunk_bound = False
        self._packet_open = False
        self._closed = False
        self._failed = False
        self._failure_reason: str | None = None
        self._next_route = 0
        self._written_cell_count = 0
        self._route_windows: list[tuple[int, int]] = []
        self._input_tensor_object_ids: list[int] = []
        self._finiteness_scalar_transfer_count = 0
        self._receipt: V1SignedWorkRowSpoolCustodyReceipt | None = None
        self._device_handle: V1SignedWorkRowBankDeviceHandle | None = None

    def _assert_live(self) -> None:
        if self._failed:
            raise RuntimeError(
                "signed-work row-spool transaction permanently refused after: "
                f"{self._failure_reason}"
            )
        if self._closed:
            raise RuntimeError("signed-work row-spool transaction is already closed")

    def _abort(self, error: Exception) -> NoReturn:
        self._failed = True
        self._failure_reason = f"{type(error).__name__}: {error}"
        # A refused pre-close transaction cannot retain a route-scale bank.
        # Completed custody has already moved into the private handle and is
        # never reached through this path.
        self._row_bank = None
        raise error

    @property
    def input_tensor_object_ids(self) -> tuple[int, ...]:
        return tuple(self._input_tensor_object_ids)

    @property
    def receipt(self) -> V1SignedWorkRowSpoolCustodyReceipt:
        if self._receipt is None:
            raise RuntimeError("row-spool receipt is unavailable before exact close")
        return self._receipt

    @property
    def private_device_handle(self) -> V1SignedWorkRowBankDeviceHandle:
        if self._device_handle is None:
            raise RuntimeError("row-bank handle is unavailable before exact close")
        self._device_handle.receipt
        return self._device_handle

    def bind_route_chunk_size(self, chunk_size: int) -> None:
        try:
            self._assert_live()
            observed = _positive_int(chunk_size, name="provider route chunk size")
            if observed != self.request.registered_route_chunk_size:
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.STALE_IDENTITY,
                    "provider route chunk size differs from the frozen request",
                )
            if self._packet_open:
                raise RuntimeError("cannot bind route chunks after packet open")
            self.request.assert_integrity()
            self.memory_forecast.assert_admissible()
            self.chunk_size = observed
            self._route_chunk_bound = True
        except Exception as error:
            self._abort(error)

    def begin_packet(
        self,
        context_ids: Sequence[str],
        *,
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
    ) -> None:
        try:
            self._assert_live()
            self.request.assert_integrity()
            if not self._route_chunk_bound:
                raise RuntimeError("provider route chunk size was not bound")
            if self._packet_open:
                raise RuntimeError("signed-work row-spool packet is already open")
            contexts = _identity_axis(
                context_ids,
                name="observed row-spool contexts",
                unique=True,
            )
            if contexts != self.request.context_ids:
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.STALE_IDENTITY,
                    "provider contexts differ from the registered D prefix",
                )
            if (
                len(route_ids) != self.request.route_count
                or _route_axis_hash(
                    route_ids,
                    name="observed signed-work row-spool route IDs",
                )
                != self.request.route_id_axis_hash
            ):
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.STALE_IDENTITY,
                    "provider route axis differs from the registered D prefix",
                )
            if str(destination_transport_hash or "") != (
                self.request.destination_transport_hash
            ):
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.STALE_IDENTITY,
                    "provider destination transport differs from the request",
                )
            if str(loss_pullback_hash or "") != (
                self.request.raw_logit_stream_identity_hash
            ):
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.STALE_IDENTITY,
                    "provider raw-logit stream differs from the request",
                )
            self._packet_open = True
            self._next_route = 0
        except Exception as error:
            self._abort(error)

    def _validate_positive_chunk(
        self,
        start: int,
        stop: int,
        rows: Any,
    ) -> tuple[int, int]:
        begin = _nonnegative_int(start, name="route-window start")
        end = _positive_int(stop, name="route-window stop")
        width = end - begin
        if (
            not self._packet_open
            or begin != self._next_route
            or width <= 0
            or width > self.chunk_size
            or end > self.request.route_count
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.DUPLICATE_OR_OUT_OF_ORDER_CHUNK,
                "row-spool chunks must cover each route cell exactly once in order",
            )
        if (
            not torch.is_tensor(rows)
            or tuple(rows.shape) != (self.request.context_count, width)
            or rows.dtype != torch.float64
            or rows.device != self.device
            or not rows.is_contiguous()
            or rows.requires_grad
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.DEVICE_MISMATCH,
                "row-spool chunk must be contiguous detached float64 "
                "[context,route-window] on the registered device",
            )
        minimum, maximum = torch.aminmax(rows)
        self._finiteness_scalar_transfer_count += 2
        if not (
            bool(torch.isfinite(minimum).item())
            and bool(torch.isfinite(maximum).item())
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.NONFINITE_ROW,
                "corrected-evidence row chunk contains a non-finite value",
            )
        return begin, end

    def consume_positive_removal_chunk(self, start: int, stop: int, rows: Any) -> None:
        try:
            self._assert_live()
            begin, end = self._validate_positive_chunk(start, stop, rows)
            if self._row_bank is None:  # pragma: no cover - state guard.
                raise RuntimeError("row-spool bank lost pre-close custody")
            with torch.no_grad():
                self._row_bank[:, begin:end].copy_(rows)
            self._input_tensor_object_ids.append(id(rows))
            self._route_windows.append((begin, end))
            self._written_cell_count += self.request.context_count * (end - begin)
            self._next_route = end
        except Exception as error:
            self._abort(error)

    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        del start, stop, rows
        self._abort(
            V1SignedWorkRowSpoolRefusal(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "signed-work row spool refuses generic/BCE tripartite rows",
            )
        )

    def consume_chunk_fused(
        self,
        start: int,
        stop: int,
        *,
        global_raw: Any,
        gate: Any,
        pullback: Any,
        affine_scale: float,
        score_epilogue: Any | None = None,
    ) -> None:
        del start, stop, global_raw, gate, pullback, affine_scale, score_epilogue
        self._abort(
            V1SignedWorkRowSpoolRefusal(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "signed-work row spool refuses consumer-side score formation",
            )
        )

    def end_packet(self) -> None:
        try:
            self._assert_live()
            expected_cells = self.request.context_count * self.request.route_count
            if (
                not self._packet_open
                or self._next_route != self.request.route_count
                or self._written_cell_count != expected_cells
            ):
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.PARTIAL_PACKET,
                    "cannot close a partial corrected-evidence row bank",
                )
            windows = _validate_route_windows(
                self._route_windows,
                route_count=self.request.route_count,
                maximum_width=self.chunk_size,
            )
            if self._row_bank is None:  # pragma: no cover - state guard.
                raise RuntimeError("row-spool bank lost close-time custody")
            self.request.assert_integrity()
            self.memory_forecast.assert_integrity()
            source_replay_identity_hash = content_hash(
                {
                    "schema": "signed_work_exact_row_source_replay_v1",
                    "request_hash": self.request.request_hash,
                    "route_window_axis_hash": content_hash(
                        windows,
                        prefix=_WINDOW_AXIS_PREFIX,
                    ),
                    "row_semantics": self.request.raw_target_logit_row_semantics,
                    "every_cell_written_exactly_once": True,
                },
                prefix=_SOURCE_REPLAY_PREFIX,
            )
            production = self.request.execution_mode is (
                V1SignedWorkRowSpoolExecutionMode.CUDA_PRODUCTION
            )
            receipt = V1SignedWorkRowSpoolCustodyReceipt(
                request_hash=self.request.request_hash,
                memory_forecast_hash=self.memory_forecast.forecast_hash,
                route_axis_hash=self.request.route_axis_hash,
                route_id_axis_hash=self.request.route_id_axis_hash,
                context_axis_hash=content_hash(
                    self.request.context_ids,
                    prefix=_CONTEXT_AXIS_PREFIX,
                ),
                draw_axis_hash=content_hash(
                    self.request.draw_ids,
                    prefix=_DRAW_AXIS_PREFIX,
                ),
                stratum_ledger_hash=content_hash(
                    self.request.stratum_labels,
                    prefix=_STRATUM_LEDGER_PREFIX,
                ),
                source_measurement_ledger_hash=content_hash(
                    self.request.source_measurement_hashes,
                    prefix=_SOURCE_LEDGER_PREFIX,
                ),
                context_weight_axis_hash=content_hash(
                    self.request.context_weights,
                    prefix=_WEIGHT_AXIS_PREFIX,
                ),
                registered_d_prefix_hash=self.request.registered_d_prefix_hash,
                chart_identity_hash=self.request.chart_identity_hash,
                current_mask_hash=self.request.current_mask_hash,
                proposal_generation=self.request.proposal_generation,
                destination_transport_hash=(
                    self.request.destination_transport_hash
                ),
                raw_logit_stream_identity_hash=(
                    self.request.raw_logit_stream_identity_hash
                ),
                native_fiber_axis_receipt_hash=(
                    self.request.native_fiber_axis_receipt_hash
                ),
                native_fiber_measure_hash=self.request.native_fiber_measure_hash,
                route_count=self.request.route_count,
                context_count=self.request.context_count,
                registered_route_chunk_size=self.chunk_size,
                route_windows=windows,
                row_bank_shape=(
                    self.request.context_count,
                    self.request.route_count,
                ),
                row_bank_dtype=str(self._row_bank.dtype),
                row_bank_layout="c_contiguous_context_major_route_minor",
                row_bank_tensor_payload_bytes=(
                    int(self._row_bank.numel())
                    * int(self._row_bank.element_size())
                ),
                declared_device=self.request.declared_device,
                device_identity=_device_identity(self.device),
                execution_mode=self.request.execution_mode,
                source_replay_identity_hash=source_replay_identity_hash,
                production_cuda_proposal_input_authority=production,
                cpu_reference_test_only=not production,
                _issuance_token=_RECEIPT_ISSUANCE_TOKEN,
            )
            handle = V1SignedWorkRowBankDeviceHandle(
                tensor=self._row_bank,
                receipt=receipt,
                context_weights=self.request.context_weights,
                native_fiber_axis_receipt=(
                    self.request.native_fiber_axis_receipt
                ),
            )
            self._row_bank = None
            self._receipt = receipt
            self._device_handle = handle
            self._packet_open = False
            self._closed = True
        except Exception as error:
            self._abort(error)

    def route_streaming_receipt(self) -> Mapping[str, object]:
        receipt = self.receipt
        result = {
            "schema": receipt.schema,
            "typed_custody_receipt_hash": receipt.receipt_hash,
            "request_hash": receipt.request_hash,
            "memory_forecast_hash": receipt.memory_forecast_hash,
            "route_chunked_streaming": True,
            "route_chunk_size": receipt.registered_route_chunk_size,
            "route_chunk_update_count": len(receipt.route_windows),
            "loss_only_route_streaming": True,
            "signed_work_exact_d_prefix_row_spooling": True,
            "positive_removal_stream_coordinate": (
                RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
            ),
            "positive_removal_stream_identity_hash": (
                receipt.raw_logit_stream_identity_hash
            ),
            "raw_target_logit_row_semantics": (
                CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS
            ),
            "full_tripartite_rows_materialized": False,
            "exact_context_route_bank_materialized_on_registered_device": True,
            "host_context_route_rows_transferred": False,
            "host_context_route_row_transfer_count": 0,
            "host_context_route_row_transfer_bytes": 0,
            "host_scalar_finiteness_transfer_count": (
                self._finiteness_scalar_transfer_count
            ),
            "private_device_custody_available": True,
            "production_cuda_proposal_input_authority": (
                receipt.production_cuda_proposal_input_authority
            ),
            "cpu_reference_test_only": receipt.cpu_reference_test_only,
            "fq_certification_authority": False,
            "calibration_authority": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
        }
        return MappingProxyType(result)


class V1SignedWorkMultiPacketRowSpoolCUDAConsumer(_RouteChunkedLossOnlyConsumer):
    """Packet-major exact writer into one preregistered final row bank.

    The optimized scorer opens one context packet and streams every native
    route window before opening the next packet.  This sibling preserves that
    producer-native order while writing each packet directly into its frozen,
    disjoint context slice.  It never allocates a packet bank or reconstructs
    route-major rows.
    """

    positive_removal_stream_coordinate = RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE

    def __init__(self, request: V1SignedWorkMultiPacketRowSpoolRequest) -> None:
        if torch is None:
            raise RuntimeError("signed-work row spooling requires PyTorch")
        if type(request) is not V1SignedWorkMultiPacketRowSpoolRequest:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.INVALID_TYPE,
                "multi-packet row-spool consumer requires its exact request",
            )
        request.assert_integrity()
        forecast = request.memory_forecast
        forecast.assert_admissible()
        source = request.source_request
        device = _canonical_runtime_device(source.declared_device)
        if source.execution_mode is V1SignedWorkRowSpoolExecutionMode.CUDA_PRODUCTION:
            if not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.DEVICE_MISMATCH,
                    "CUDA production multi-packet spooling requires CUDA",
                )
            try:
                torch.cuda.get_device_properties(device)
            except (AssertionError, RuntimeError) as error:
                raise V1SignedWorkRowSpoolRefusal(
                    V1SignedWorkRowSpoolRefusalCode.DEVICE_MISMATCH,
                    "declared CUDA multi-packet row-spool device is unavailable",
                ) from error
            if not _live_row_spool_hbm_admitted(forecast, device):
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.MEMORY_INADMISSIBLE,
                    "live CUDA allocation exceeds the frozen occupancy bound",
                )

        row_bank = torch.empty(
            (request.context_count, request.route_count),
            dtype=torch.float64,
            device=device,
        )
        if not row_bank.is_contiguous() or row_bank.requires_grad:
            raise RuntimeError("multi-packet allocator did not produce canonical storage")
        if (
            int(row_bank.numel()) * int(row_bank.element_size())
            != forecast.row_bank_device_bytes
        ):
            raise RuntimeError("multi-packet allocation differs from its forecast")

        self.request = request
        self.source_request = source
        self.route_ids = source.route_ids
        self.device = device
        self.loss_pullback_hash = source.raw_logit_stream_identity_hash
        self.destination_transport_hash = source.destination_transport_hash
        self.head_ids = ("positive_removal:raw_target_logit",)
        self.chunk_size = source.registered_route_chunk_size
        self.memory_forecast = forecast
        self._row_bank: Any | None = row_bank
        self._route_chunk_bound = False
        self._packet_open = False
        self._closed = False
        self._failed = False
        self._failure_reason: str | None = None
        self._next_packet = 0
        self._next_window = 0
        self._active_slice: V1SignedWorkRowSpoolContextPacketSlice | None = None
        self._packet_written_cells = 0
        self._total_written_cells = 0
        self._packet_window_write_count = 0
        self._finiteness_scalar_transfer_count = 0
        self._custody_receipt: V1SignedWorkRowSpoolCustodyReceipt | None = None
        self._receipt: V1SignedWorkMultiPacketRowSpoolCustodyReceipt | None = None
        self._device_handle: V1SignedWorkRowBankDeviceHandle | None = None

    def _assert_live(self) -> None:
        if self._failed:
            raise RuntimeError(
                "multi-packet row-spool transaction permanently refused after: "
                f"{self._failure_reason}"
            )
        if self._closed:
            raise RuntimeError("multi-packet row-spool transaction is already closed")

    def _abort(self, error: Exception) -> NoReturn:
        self._failed = True
        self._failure_reason = f"{type(error).__name__}: {error}"
        self._active_slice = None
        self._packet_open = False
        self._row_bank = None
        raise error

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def unsealed_bank_retained(self) -> bool:
        return self._row_bank is not None

    @property
    def receipt(self) -> V1SignedWorkMultiPacketRowSpoolCustodyReceipt:
        if self._receipt is None:
            raise RuntimeError(
                "multi-packet row-spool receipt is unavailable before finalization"
            )
        return self._receipt

    @property
    def custody_receipt(self) -> V1SignedWorkRowSpoolCustodyReceipt:
        if self._custody_receipt is None:
            raise RuntimeError(
                "row-bank custody receipt is unavailable before finalization"
            )
        return self._custody_receipt

    @property
    def private_device_handle(self) -> V1SignedWorkRowBankDeviceHandle:
        if self._device_handle is None:
            raise RuntimeError("row-bank handle is unavailable before finalization")
        self._device_handle.receipt
        return self._device_handle

    def bind_route_chunk_size(self, chunk_size: int) -> None:
        try:
            self._assert_live()
            observed = _positive_int(chunk_size, name="provider route chunk size")
            if (
                observed != self.source_request.registered_route_chunk_size
                or self._route_chunk_bound
                or self._packet_open
                or self._next_packet != 0
            ):
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.STALE_IDENTITY,
                    "provider route chunk binding differs or was repeated",
                )
            self.request.assert_integrity()
            self.memory_forecast.assert_admissible()
            self.chunk_size = observed
            self._route_chunk_bound = True
        except Exception as error:
            self._abort(error)

    def begin_packet(
        self,
        context_ids: Sequence[str],
        *,
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
        packet_ordinal: int,
        packet_identity_hash: str,
    ) -> None:
        try:
            self._assert_live()
            self.request.assert_integrity()
            if (
                not self._route_chunk_bound
                or self._packet_open
                or self._next_packet >= self.request.packet_count
            ):
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.DUPLICATE_OR_OUT_OF_ORDER_CHUNK,
                    "multi-packet row spool cannot open this packet coordinate",
                )
            expected = self.request.packet_slices[self._next_packet]
            contexts = _identity_axis(
                context_ids,
                name="observed packet contexts",
                unique=True,
            )
            observed_ordinal = _nonnegative_int(
                packet_ordinal,
                name="observed packet ordinal",
            )
            observed_identity = _identity(
                packet_identity_hash,
                name="observed packet identity hash",
            )
            if (
                observed_ordinal != expected.packet_ordinal
                or observed_identity != expected.packet_identity_hash
                or contexts != expected.context_ids
                or route_ids is not self.source_request.route_ids
                or len(route_ids) != self.request.route_count
                or _route_axis_hash(
                    route_ids,
                    name="observed multi-packet row-spool route IDs",
                )
                != self.source_request.route_id_axis_hash
                or str(destination_transport_hash or "")
                != self.source_request.destination_transport_hash
                or str(loss_pullback_hash or "")
                != self.source_request.raw_logit_stream_identity_hash
            ):
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.STALE_IDENTITY,
                    "provider packet/context/route/transport/stream custody drifted",
                )
            self._active_slice = expected
            self._packet_open = True
            self._next_window = 0
            self._packet_written_cells = 0
        except Exception as error:
            self._abort(error)

    def _validate_positive_chunk(
        self,
        start: int,
        stop: int,
        rows: Any,
    ) -> tuple[int, int]:
        if not self._packet_open or self._active_slice is None:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.DUPLICATE_OR_OUT_OF_ORDER_CHUNK,
                "row chunk arrived outside one registered packet",
            )
        if self._next_window >= len(self.request.registered_route_windows):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.DUPLICATE_OR_OUT_OF_ORDER_CHUNK,
                "row spool received an extra native route window",
            )
        begin = _nonnegative_int(start, name="route-window start")
        end = _positive_int(stop, name="route-window stop")
        expected = self.request.registered_route_windows[self._next_window]
        packet_contexts = self._active_slice.context_stop - self._active_slice.context_start
        if (begin, end) != expected:
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.DUPLICATE_OR_OUT_OF_ORDER_CHUNK,
                "route window differs from the preregistered source-native axis",
            )
        width = end - begin
        if (
            not torch.is_tensor(rows)
            or tuple(rows.shape) != (packet_contexts, width)
            or rows.dtype != torch.float64
            or rows.device != self.device
            or not rows.is_contiguous()
            or rows.requires_grad
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.DEVICE_MISMATCH,
                "packet-window rows must be contiguous detached float64 on the registered device",
            )
        minimum, maximum = torch.aminmax(rows)
        self._finiteness_scalar_transfer_count += 2
        if not (
            bool(torch.isfinite(minimum).item())
            and bool(torch.isfinite(maximum).item())
        ):
            _refuse(
                V1SignedWorkRowSpoolRefusalCode.NONFINITE_ROW,
                "corrected-evidence packet window contains a non-finite value",
            )
        return begin, end

    def consume_positive_removal_chunk(self, start: int, stop: int, rows: Any) -> None:
        try:
            self._assert_live()
            begin, end = self._validate_positive_chunk(start, stop, rows)
            if self._row_bank is None or self._active_slice is None:
                raise RuntimeError("multi-packet row bank lost pre-close custody")
            context_start = self._active_slice.context_start
            context_stop = self._active_slice.context_stop
            with torch.no_grad():
                self._row_bank[context_start:context_stop, begin:end].copy_(rows)
            cells = (context_stop - context_start) * (end - begin)
            self._packet_written_cells += cells
            self._total_written_cells += cells
            self._packet_window_write_count += 1
            self._next_window += 1
        except Exception as error:
            self._abort(error)

    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        del start, stop, rows
        self._abort(
            V1SignedWorkRowSpoolRefusal(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "multi-packet row spool refuses generic/BCE rows",
            )
        )

    def consume_chunk_fused(
        self,
        start: int,
        stop: int,
        *,
        global_raw: Any,
        gate: Any,
        pullback: Any,
        affine_scale: float,
        score_epilogue: Any | None = None,
    ) -> None:
        del start, stop, global_raw, gate, pullback, affine_scale, score_epilogue
        self._abort(
            V1SignedWorkRowSpoolRefusal(
                V1SignedWorkRowSpoolRefusalCode.AUTHORITY_VIOLATION,
                "multi-packet row spool refuses consumer-side score formation",
            )
        )

    def end_packet(self) -> None:
        try:
            self._assert_live()
            if not self._packet_open or self._active_slice is None:
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.PARTIAL_PACKET,
                    "no registered packet is open",
                )
            expected_cells = (
                self._active_slice.context_stop - self._active_slice.context_start
            ) * self.request.route_count
            if (
                self._next_window != len(self.request.registered_route_windows)
                or self._packet_written_cells != expected_cells
            ):
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.PARTIAL_PACKET,
                    "cannot close a partial packet route cover",
                )
            self._packet_open = False
            self._active_slice = None
            self._packet_written_cells = 0
            self._next_window = 0
            self._next_packet += 1
        except Exception as error:
            self._abort(error)

    def finalize(self) -> None:
        try:
            self._assert_live()
            expected_cells = self.request.context_count * self.request.route_count
            expected_window_writes = (
                self.request.packet_count * len(self.request.registered_route_windows)
            )
            if (
                self._packet_open
                or self._next_packet != self.request.packet_count
                or self._total_written_cells != expected_cells
                or self._packet_window_write_count != expected_window_writes
                or self._row_bank is None
            ):
                _refuse(
                    V1SignedWorkRowSpoolRefusalCode.PARTIAL_PACKET,
                    "cannot finalize an incomplete packet/context/route cover",
                )
            self.request.assert_integrity()
            self.memory_forecast.assert_integrity()
            source = self.source_request
            windows = self.request.registered_route_windows
            source_replay_identity_hash = content_hash(
                {
                    "schema": "signed_work_exact_multi_packet_row_source_replay_v1",
                    "multi_packet_request_hash": self.request.request_hash,
                    "source_request_hash": source.request_hash,
                    "packet_slice_axis_hash": self.request.packet_slice_axis_hash,
                    "route_window_axis_hash": self.request.route_window_axis_hash,
                    "row_semantics": source.raw_target_logit_row_semantics,
                    "every_context_route_cell_written_exactly_once": True,
                },
                prefix=_SOURCE_REPLAY_PREFIX,
            )
            production = source.execution_mode is (
                V1SignedWorkRowSpoolExecutionMode.CUDA_PRODUCTION
            )
            custody = V1SignedWorkRowSpoolCustodyReceipt(
                request_hash=source.request_hash,
                memory_forecast_hash=self.memory_forecast.forecast_hash,
                route_axis_hash=source.route_axis_hash,
                route_id_axis_hash=source.route_id_axis_hash,
                context_axis_hash=content_hash(
                    source.context_ids,
                    prefix=_CONTEXT_AXIS_PREFIX,
                ),
                draw_axis_hash=content_hash(source.draw_ids, prefix=_DRAW_AXIS_PREFIX),
                stratum_ledger_hash=content_hash(
                    source.stratum_labels,
                    prefix=_STRATUM_LEDGER_PREFIX,
                ),
                source_measurement_ledger_hash=content_hash(
                    source.source_measurement_hashes,
                    prefix=_SOURCE_LEDGER_PREFIX,
                ),
                context_weight_axis_hash=content_hash(
                    source.context_weights,
                    prefix=_WEIGHT_AXIS_PREFIX,
                ),
                registered_d_prefix_hash=source.registered_d_prefix_hash,
                chart_identity_hash=source.chart_identity_hash,
                current_mask_hash=source.current_mask_hash,
                proposal_generation=source.proposal_generation,
                destination_transport_hash=source.destination_transport_hash,
                raw_logit_stream_identity_hash=(
                    source.raw_logit_stream_identity_hash
                ),
                native_fiber_axis_receipt_hash=(
                    source.native_fiber_axis_receipt_hash
                ),
                native_fiber_measure_hash=source.native_fiber_measure_hash,
                route_count=source.route_count,
                context_count=source.context_count,
                registered_route_chunk_size=source.registered_route_chunk_size,
                route_windows=windows,
                row_bank_shape=(source.context_count, source.route_count),
                row_bank_dtype=str(self._row_bank.dtype),
                row_bank_layout="c_contiguous_context_major_route_minor",
                row_bank_tensor_payload_bytes=(
                    int(self._row_bank.numel()) * int(self._row_bank.element_size())
                ),
                declared_device=source.declared_device,
                device_identity=_device_identity(self.device),
                execution_mode=source.execution_mode,
                source_replay_identity_hash=source_replay_identity_hash,
                production_cuda_proposal_input_authority=production,
                cpu_reference_test_only=not production,
                _issuance_token=_RECEIPT_ISSUANCE_TOKEN,
            )
            packet_identity_axis_hash = content_hash(
                tuple(item.packet_identity_hash for item in self.request.packet_slices),
                prefix=_PACKET_SLICE_AXIS_PREFIX + "identities-",
            )
            multi_receipt = V1SignedWorkMultiPacketRowSpoolCustodyReceipt(
                request_hash=self.request.request_hash,
                source_request_hash=source.request_hash,
                legacy_custody_receipt_hash=custody.receipt_hash,
                memory_forecast_hash=self.memory_forecast.forecast_hash,
                packet_slice_axis_hash=self.request.packet_slice_axis_hash,
                route_window_axis_hash=self.request.route_window_axis_hash,
                packet_identity_axis_hash=packet_identity_axis_hash,
                route_axis_hash=source.route_axis_hash,
                route_id_axis_hash=source.route_id_axis_hash,
                registered_d_prefix_hash=source.registered_d_prefix_hash,
                chart_identity_hash=source.chart_identity_hash,
                current_mask_hash=source.current_mask_hash,
                raw_logit_stream_identity_hash=(
                    source.raw_logit_stream_identity_hash
                ),
                native_fiber_axis_receipt_hash=(
                    source.native_fiber_axis_receipt_hash
                ),
                source_replay_identity_hash=source_replay_identity_hash,
                packet_count=self.request.packet_count,
                context_count=self.request.context_count,
                route_count=self.request.route_count,
                route_window_count=len(windows),
                packet_route_window_write_count=self._packet_window_write_count,
                context_route_cell_write_count=self._total_written_cells,
                finiteness_scalar_transfer_count=(
                    self._finiteness_scalar_transfer_count
                ),
                row_bank_tensor_payload_bytes=(
                    int(self._row_bank.numel()) * int(self._row_bank.element_size())
                ),
                declared_device=source.declared_device,
                execution_mode=source.execution_mode,
                production_cuda_proposal_input_authority=production,
                cpu_reference_test_only=not production,
                _issuance_token=_MULTI_RECEIPT_ISSUANCE_TOKEN,
            )
            handle = V1SignedWorkRowBankDeviceHandle(
                tensor=self._row_bank,
                receipt=custody,
                context_weights=source.context_weights,
                native_fiber_axis_receipt=source.native_fiber_axis_receipt,
            )
            self._row_bank = None
            self._custody_receipt = custody
            self._receipt = multi_receipt
            self._device_handle = handle
            self._closed = True
        except Exception as error:
            self._abort(error)

    def abort_unsealed(self) -> None:
        if self._closed:
            raise RuntimeError(
                "finalized row-bank custody must be evicted through its device handle"
            )
        self._failed = True
        self._failure_reason = "explicit unsealed transaction abort"
        self._active_slice = None
        self._packet_open = False
        self._row_bank = None

    def route_streaming_receipt(self) -> Mapping[str, object]:
        receipt = self.receipt
        result = {
            "schema": receipt.schema,
            "typed_multi_packet_receipt_hash": receipt.receipt_hash,
            "typed_custody_receipt_hash": receipt.legacy_custody_receipt_hash,
            "request_hash": receipt.request_hash,
            "memory_forecast_hash": receipt.memory_forecast_hash,
            "route_chunked_streaming": True,
            "source_native_route_windows": True,
            "packet_major_context_slice_writes": True,
            "packet_count": receipt.packet_count,
            "route_chunk_update_count": receipt.packet_route_window_write_count,
            "signed_work_exact_d_prefix_row_spooling": True,
            "exact_context_route_bank_materialized_on_registered_device": True,
            "host_context_route_rows_transferred": False,
            "host_context_route_row_transfer_count": 0,
            "host_context_route_row_transfer_bytes": 0,
            "raw_row_scratch_bytes": 0,
            "row_payload_cryptographic_hash_materialized": False,
            "row_bank_allocation_count": 1,
            "maximum_live_staging_tile_count": 1,
            "host_scalar_finiteness_transfer_count": (
                receipt.finiteness_scalar_transfer_count
            ),
            "private_device_custody_available": True,
            "production_cuda_proposal_input_authority": (
                receipt.production_cuda_proposal_input_authority
            ),
            "cpu_reference_test_only": receipt.cpu_reference_test_only,
            "fq_certification_authority": False,
            "calibration_authority": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
        }
        return MappingProxyType(result)


__all__ = [
    "CORRECTED_AUMANN_SHAPLEY_ROW_SEMANTICS",
    "RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE",
    "SIGNED_WORK_ROW_SPOOL_AUTHORITY_SCOPE",
    "SIGNED_WORK_ROW_SPOOL_CUSTODY_RECEIPT_SCHEMA",
    "SIGNED_WORK_ROW_SPOOL_HBM_FORECAST_SCHEMA",
    "SIGNED_WORK_ROW_SPOOL_MEMORY_SCOPE",
    "SIGNED_WORK_ROW_SPOOL_REQUEST_SCHEMA",
    "SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_CUSTODY_RECEIPT_SCHEMA",
    "SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_HBM_FORECAST_SCHEMA",
    "SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_MEMORY_SCOPE",
    "SIGNED_WORK_MULTI_PACKET_ROW_SPOOL_REQUEST_SCHEMA",
    "V1SignedWorkRowBankDeviceHandle",
    "V1SignedWorkRowBankLease",
    "V1SignedWorkMultiPacketRowSpoolCUDAConsumer",
    "V1SignedWorkMultiPacketRowSpoolCustodyReceipt",
    "V1SignedWorkMultiPacketRowSpoolHBMForecast",
    "V1SignedWorkMultiPacketRowSpoolRequest",
    "V1SignedWorkRowSpoolCUDAConsumer",
    "V1SignedWorkRowSpoolContextPacketSlice",
    "V1SignedWorkRowSpoolCustodyReceipt",
    "V1SignedWorkRowSpoolExecutionMode",
    "V1SignedWorkRowSpoolHBMForecast",
    "V1SignedWorkRowSpoolRefusal",
    "V1SignedWorkRowSpoolRefusalCode",
    "V1SignedWorkRowSpoolRequest",
    "forecast_v1_signed_work_multi_packet_row_spool_hbm",
    "forecast_v1_signed_work_row_spool_hbm",
]
