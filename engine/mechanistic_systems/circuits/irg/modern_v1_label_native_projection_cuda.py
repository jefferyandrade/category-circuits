"""Bounded label-native dual-chart projection for modern v1.

This module projects the two route streams emitted by one conserving scorer
capture onto a pre-C device-shell action axis without materialising endpoint
``HardMask`` objects or route-by-endpoint directions.

For a context ``x`` and registered endpoint label ``e`` it computes

``F_x(e) = mean_cut sum_fiber abs(sum_{label(route)<=e, route in fiber} a_x(route))``

and

``J_x(e) = sum_{label(route)<=e} j_x(route)``.

The first expression permits signed cancellation only inside one native
destination fiber.  The second is the literal global signed response and may
cancel across every route.  Native fibers are contiguous under the registered
target-major route layout, so a streaming route chunk needs at most one
boundary-fiber carry.  Complete fibers are reduced by a fixed padded binary
tree in physical route order.  No floating atomic, scatter-add, endpoint mask,
full label D2H, or full physical-index D2H participates.

The production factory accepts only the exact label-native provider and keeps
an explicit label-handle lease until :meth:`V1LabelNativeProjectionGeometry.close`.
CPU execution is an explicitly private, route-limited reference used by
focused tests.  Production mode fails closed when deterministic CUDA is not
available.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import InitVar, dataclass, field
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Final, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_device_shell_ladder import (
    V1DeviceShellBorrowOperation,
    V1DeviceShellLabelLease,
    V1DeviceShellRefusal,
)
from .modern_v1_native_fiber_axis import (
    V1NativeDestinationFiberAxisReceipt,
    V1NativeDestinationFiberCut,
)
from .modern_v1_signed_work_label_native_direction_provider import (
    V1LabelNativePackedDirectionProvider,
)


LABEL_NATIVE_PROJECTION_CUDA_DESIGN_SCHEMA: Final = (
    "partition_free_v1_label_native_dual_chart_projection_cuda_design_v1"
)
LABEL_NATIVE_PROJECTION_CUDA_FORECAST_SCHEMA: Final = (
    "partition_free_v1_label_native_dual_chart_projection_cuda_forecast_v1"
)
LABEL_NATIVE_PROJECTION_GEOMETRY_SCHEMA: Final = (
    "partition_free_v1_label_native_dual_chart_projection_geometry_v2"
)
LABEL_NATIVE_PROJECTION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_label_native_dual_chart_projection_receipt_v3"
)
LABEL_NATIVE_PROJECTION_RESULT_SCHEMA: Final = (
    "partition_free_v1_label_native_dual_chart_projection_result_v1"
)
LABEL_NATIVE_PROJECTION_ALGORITHM: Final = (
    "route_chunk_shell_threshold_native_fiber_physical_order_fixed_tree_v1"
)
LABEL_NATIVE_PROJECTION_AUTHORITY: Final = (
    "same_capture_corrected_native_fiber_F_and_global_literal_J_projection_"
    "only_no_C_decision_K_O_commit_terminal_or_pareto_authority_v1"
)
LABEL_NATIVE_PROJECTION_NUMERICAL_ENCLOSURE: Final = (
    "binary64_gamma_bound_over_fixed_tree_and_registered_chunk_additions_v1"
)

_DESIGN_PREFIX: Final = "modern-v1-label-native-projection-cuda-design-"
_FORECAST_PREFIX: Final = "modern-v1-label-native-projection-cuda-forecast-"
_GEOMETRY_PREFIX: Final = "modern-v1-label-native-projection-geometry-"
_RECEIPT_PREFIX: Final = "modern-v1-label-native-projection-receipt-"
_MATRIX_PREFIX: Final = "modern-v1-label-native-projection-matrix-"
_WINDOW_PREFIX: Final = "modern-v1-label-native-projection-windows-"
_LEGACY_LANE_PREFIX: Final = "modern-v1-label-native-legacy-full-axis-lanes-"
_F64_BYTES: Final = 8
_I64_BYTES: Final = 8
_BOOL_BYTES: Final = 1
_UNIT_ROUNDOFF: Final = 2.0**-53
_MINIMUM_ALLOCATOR_GUARD_BYTES: Final = 64 * 1024 * 1024


class V1LabelNativeProjectionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    CUDA_UNAVAILABLE = "cuda_unavailable"
    CPU_FALLBACK_FORBIDDEN = "cpu_fallback_forbidden"
    CPU_REFERENCE_LIMIT_EXCEEDED = "cpu_reference_limit_exceeded"
    DETERMINISM_DISABLED = "determinism_disabled"
    STALE_SOURCE = "stale_source"
    ACTION_AXIS_MISMATCH = "action_axis_mismatch"
    LABEL_IDENTITY_MISMATCH = "label_identity_mismatch"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    GEOMETRY_CLOSED = "geometry_closed"
    GEOMETRY_BUSY = "geometry_busy"
    ORDER_MISMATCH = "order_mismatch"
    INVALID_ROW = "invalid_row"
    NONFINITE_ROW = "nonfinite_row"
    PARTIAL_FIBER_MISMATCH = "partial_fiber_mismatch"
    INCOMPLETE_ROUTE_AXIS = "incomplete_route_axis"
    FINALIZED = "finalized"
    AUTHORITY_VIOLATION = "authority_violation"


class V1LabelNativeProjectionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1LabelNativeProjectionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LabelNativeProjectionRefusalCode:
            raise TypeError("label-native projection refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LabelNativeProjectionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LabelNativeProjectionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one exact integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one exact integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one real scalar")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must lie in (0, 1]")
    return result


def _identity_tuple(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be one identity sequence")
    result = tuple(_identity(value, name=f"{name} item") for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent.
        _refuse(
            V1LabelNativeProjectionRefusalCode.CUDA_UNAVAILABLE,
            f"PyTorch is required for label-native projection: {error}",
        )
    return torch


def _require_cuda_determinism(torch: Any) -> None:
    if not bool(torch.cuda.is_available()):
        _refuse(
            V1LabelNativeProjectionRefusalCode.CUDA_UNAVAILABLE,
            "production label-native projection requires available CUDA",
        )
    if not bool(torch.are_deterministic_algorithms_enabled()):
        _refuse(
            V1LabelNativeProjectionRefusalCode.DETERMINISM_DISABLED,
            "torch deterministic algorithms must already be enabled",
        )
    warn_only = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: False,
    )
    if bool(warn_only()):
        _refuse(
            V1LabelNativeProjectionRefusalCode.DETERMINISM_DISABLED,
            "warn-only deterministic mode is not an execution guarantee",
        )


def _fixed_tree_reduce_last(torch: Any, values: Any) -> Any:
    if not isinstance(values, torch.Tensor) or values.ndim < 1:
        raise TypeError("fixed-tree reduction requires one torch tensor")
    original = int(values.shape[-1])
    if original <= 0:
        raise ValueError("fixed-tree reduction axis is empty")
    width = 1 << (original - 1).bit_length()
    if width == original:
        work = values
    else:
        shape = tuple(int(value) for value in values.shape[:-1]) + (width,)
        work = torch.zeros(shape, dtype=torch.float64, device=values.device)
        work[..., :original].copy_(values)
    while width > 1:
        work = work[..., :width:2] + work[..., 1:width:2]
        width //= 2
    return work[..., 0].clone()


def _canonical_matrix(
    value: object,
    *,
    shape: tuple[int, int],
    name: str,
) -> tuple[np.ndarray, str]:
    array = np.array(value, dtype="<f8", order="C", copy=True)
    if array.shape != shape or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must be finite with shape {shape}")
    bits = array.reshape(-1).view(np.uint64)
    bits[bits == np.uint64(1 << 63)] = np.uint64(0)
    encoded = array.tobytes(order="C")
    result = np.frombuffer(encoded, dtype="<f8").reshape(shape)
    result.setflags(write=False)
    digest = content_hash(
        {
            "dtype": "<f8",
            "shape": shape,
            "payload_sha256": hashlib.sha256(encoded).hexdigest(),
            "payload_bytes": len(encoded),
        },
        prefix=_MATRIX_PREFIX,
    )
    return result, digest


def label_native_projection_cuda_source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _legacy_full_axis_lane_registration_hash(
    *,
    provider_receipt_hash: str,
    source_endpoint_labels: tuple[int, ...],
    endpoint_counts: tuple[int, ...],
    endpoint_action_hashes: tuple[str, ...],
) -> str:
    """Bind the private-reference identity map without impersonating qualification."""

    return content_hash(
        {
            "schema": "label_native_legacy_full_axis_identity_lane_registration_v1",
            "provider_receipt_hash": provider_receipt_hash,
            "compact_provider_endpoint_indices": tuple(
                range(1, len(source_endpoint_labels) + 1)
            ),
            "source_endpoint_labels": source_endpoint_labels,
            "endpoint_counts": endpoint_counts,
            "endpoint_action_hashes": endpoint_action_hashes,
            "qualified_lane_registration_used": False,
            "private_reference_only": True,
        },
        prefix=_LEGACY_LANE_PREFIX,
    )


@dataclass(frozen=True, slots=True)
class V1LabelNativeProjectionCUDAExecutionDesign:
    route_chunk_size: int = 262_144
    endpoint_lane_tile_size: int = 8
    maximum_fibers_per_tile: int = 512
    maximum_context_batch_size: int = 8
    hbm_ceiling_fraction: float = 0.85
    private_cpu_route_limit: int = 4096
    private_cpu_workspace_ceiling_bytes: int = 1 << 30
    production_cuda_required: bool = True
    allow_private_cpu_reference: bool = False
    require_deterministic_algorithms: bool = True
    allow_float_atomics: bool = False
    schema: str = LABEL_NATIVE_PROJECTION_CUDA_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "route_chunk_size",
            "endpoint_lane_tile_size",
            "maximum_fibers_per_tile",
            "maximum_context_batch_size",
            "private_cpu_route_limit",
            "private_cpu_workspace_ceiling_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        ceiling = _fraction(self.hbm_ceiling_fraction, name="HBM ceiling fraction")
        if ceiling != 0.85:
            raise ValueError("canonical projection HBM ceiling is exactly 85 percent")
        flags = (
            self.production_cuda_required,
            self.allow_private_cpu_reference,
            self.require_deterministic_algorithms,
            self.allow_float_atomics,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("projection execution flags must be exact booleans")
        if self.production_cuda_required and self.allow_private_cpu_reference:
            raise PermissionError("production CUDA cannot enable a CPU fallback")
        if not self.require_deterministic_algorithms or self.allow_float_atomics:
            raise PermissionError("canonical projection requires deterministic no-atomic reduction")
        if self.schema != LABEL_NATIVE_PROJECTION_CUDA_DESIGN_SCHEMA:
            raise ValueError("unsupported projection design schema")
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_DESIGN_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LabelNativeProjectionCUDAHBMForecast:
    design_hash: str
    provider_receipt_hash: str
    native_fiber_axis_receipt_hash: str
    backend_kind: str
    device_identity: str
    route_count: int
    endpoint_count: int
    selected_endpoint_count: int
    maximum_routes_per_fiber: int
    route_chunk_size: int
    endpoint_lane_tile_size: int
    maximum_fibers_per_tile: int
    maximum_context_batch_size: int
    borrowed_label_bytes: int
    compact_endpoint_bytes: int
    carry_bytes: int
    output_bytes: int
    membership_tile_bytes: int
    value_tile_bytes: int
    tree_intermediate_bytes: int
    allocator_guard_bytes: int
    forecast_incremental_bytes: int
    current_reserved_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_or_workspace_ceiling_bytes: int
    admitted: bool
    route_by_endpoint_mask_materialized: bool = False
    bounded_route_chunk_endpoint_membership_tile_used: bool = True
    endpoint_hard_masks_materialized: bool = False
    schema: str = LABEL_NATIVE_PROJECTION_CUDA_FORECAST_SCHEMA
    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _admission_memory: tuple[int, int] | None) -> None:
        available, allocated = _admission_memory or (self.device_total_or_workspace_ceiling_bytes, self.current_reserved_bytes)
        for name in (
            "design_hash",
            "provider_receipt_hash",
            "native_fiber_axis_receipt_hash",
            "device_identity",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.backend_kind not in {"torch_cuda", "numpy_reference"}:
            raise ValueError("unsupported projection forecast backend")
        for name in (
            "route_count",
            "endpoint_count",
            "selected_endpoint_count",
            "maximum_routes_per_fiber",
            "route_chunk_size",
            "endpoint_lane_tile_size",
            "maximum_fibers_per_tile",
            "maximum_context_batch_size",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        for name in (
            "borrowed_label_bytes",
            "compact_endpoint_bytes",
            "carry_bytes",
            "output_bytes",
            "membership_tile_bytes",
            "value_tile_bytes",
            "tree_intermediate_bytes",
            "allocator_guard_bytes",
            "forecast_incremental_bytes",
            "current_reserved_bytes",
            "forecast_peak_reserved_bytes",
            "device_total_or_workspace_ceiling_bytes",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name=name))
        lanes = min(self.endpoint_lane_tile_size, self.selected_endpoint_count)
        fibers = self.maximum_fibers_per_tile
        routes = self.maximum_routes_per_fiber
        flat_routes = min(self.route_chunk_size, self.route_count)
        batch = self.maximum_context_batch_size
        expected_endpoint = self.selected_endpoint_count * _I64_BYTES
        expected_carry = batch * self.selected_endpoint_count * _F64_BYTES
        expected_output = batch * self.selected_endpoint_count * _F64_BYTES * 4
        expected_membership = max(
            lanes * flat_routes,
            lanes * fibers * routes,
        ) * _BOOL_BYTES
        expected_values = max(
            batch * lanes * flat_routes,
            batch * lanes * fibers * routes,
        ) * _F64_BYTES
        padded_flat_routes = 1 << (flat_routes - 1).bit_length()
        padded_routes = 1 << (routes - 1).bit_length()
        padded_fibers = 1 << (fibers - 1).bit_length()
        flat_tree_elements = padded_flat_routes + padded_flat_routes // 2 + 1
        route_tree_elements = fibers * (
            padded_routes + padded_routes // 2 + 1
        )
        fiber_tree_elements = (
            2 * fibers + padded_fibers + padded_fibers // 2 + 1
        )
        expected_tree = (
            batch
            * lanes
            * max(flat_tree_elements, route_tree_elements, fiber_tree_elements)
            * _F64_BYTES
        )
        working = (
            expected_endpoint
            + expected_carry
            + expected_output
            + expected_membership
            + expected_values
            + expected_tree
        )
        expected_guard = max(_MINIMUM_ALLOCATOR_GUARD_BYTES, math.ceil(0.10 * working))
        expected_incremental = working + expected_guard
        expected_peak = self.current_reserved_bytes + expected_incremental
        expected_admitted = allocated + expected_incremental <= self.device_total_or_workspace_ceiling_bytes and expected_incremental <= available
        observed = (
            self.borrowed_label_bytes,
            self.compact_endpoint_bytes,
            self.carry_bytes,
            self.output_bytes,
            self.membership_tile_bytes,
            self.value_tile_bytes,
            self.tree_intermediate_bytes,
            self.allocator_guard_bytes,
            self.forecast_incremental_bytes,
            self.forecast_peak_reserved_bytes,
            self.admitted,
        )
        expected = (
            0,
            expected_endpoint,
            expected_carry,
            expected_output,
            expected_membership,
            expected_values,
            expected_tree,
            expected_guard,
            expected_incremental,
            expected_peak,
            expected_admitted,
        )
        if observed != expected:
            raise ValueError("projection HBM forecast does not replay")
        flags = (
            self.route_by_endpoint_mask_materialized,
            self.endpoint_hard_masks_materialized,
        )
        if (
            any(type(value) is not bool for value in flags)
            or any(flags)
            or self.bounded_route_chunk_endpoint_membership_tile_used is not True
        ):
            raise PermissionError("projection forecast cannot materialize endpoint masks")
        if type(self.admitted) is not bool:
            raise TypeError("projection forecast admission must be exact")
        if self.schema != LABEL_NATIVE_PROJECTION_CUDA_FORECAST_SCHEMA:
            raise ValueError("unsupported projection forecast schema")

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_FORECAST_PREFIX)


def build_v1_label_native_projection_cuda_hbm_forecast(
    *,
    design: V1LabelNativeProjectionCUDAExecutionDesign,
    provider_receipt_hash: str,
    native_fiber_axis_receipt_hash: str,
    backend_kind: str,
    device_identity: str,
    route_count: int,
    endpoint_count: int,
    selected_endpoint_count: int,
    maximum_routes_per_fiber: int,
    current_reserved_bytes: int,
    device_total_or_workspace_ceiling_bytes: int,
    _admission_memory: tuple[int, int] | None = None,
) -> V1LabelNativeProjectionCUDAHBMForecast:
    if type(design) is not V1LabelNativeProjectionCUDAExecutionDesign:
        raise TypeError("projection forecast requires an exact execution design")
    lanes = min(design.endpoint_lane_tile_size, selected_endpoint_count)
    fibers = min(
        design.maximum_fibers_per_tile,
        max(1, math.ceil(design.route_chunk_size / maximum_routes_per_fiber)),
    )
    flat_routes = min(design.route_chunk_size, route_count)
    membership = max(
        lanes * flat_routes,
        lanes * fibers * maximum_routes_per_fiber,
    ) * _BOOL_BYTES
    values = max(
        design.maximum_context_batch_size * lanes * flat_routes,
        design.maximum_context_batch_size
        * lanes
        * fibers
        * maximum_routes_per_fiber,
    ) * _F64_BYTES
    padded_flat_routes = 1 << (flat_routes - 1).bit_length()
    padded_routes = 1 << (maximum_routes_per_fiber - 1).bit_length()
    padded_fibers = 1 << (fibers - 1).bit_length()
    flat_tree_elements = padded_flat_routes + padded_flat_routes // 2 + 1
    route_tree_elements = fibers * (
        padded_routes + padded_routes // 2 + 1
    )
    fiber_tree_elements = (
        2 * fibers + padded_fibers + padded_fibers // 2 + 1
    )
    tree = (
        design.maximum_context_batch_size
        * lanes
        * max(flat_tree_elements, route_tree_elements, fiber_tree_elements)
        * _F64_BYTES
    )
    endpoint_bytes = selected_endpoint_count * _I64_BYTES
    carry = design.maximum_context_batch_size * selected_endpoint_count * _F64_BYTES
    output = carry * 4
    working = endpoint_bytes + carry + output + membership + values + tree
    guard = max(_MINIMUM_ALLOCATOR_GUARD_BYTES, math.ceil(0.10 * working))
    incremental = working + guard
    current = _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
    ceiling = _positive_int(
        device_total_or_workspace_ceiling_bytes,
        name="device/workspace ceiling bytes",
    )
    available, allocated = _admission_memory or (ceiling, current)
    return V1LabelNativeProjectionCUDAHBMForecast(
        design_hash=design.design_hash,
        provider_receipt_hash=_identity(provider_receipt_hash, name="provider receipt hash"),
        native_fiber_axis_receipt_hash=_identity(
            native_fiber_axis_receipt_hash,
            name="native-fiber receipt hash",
        ),
        backend_kind=backend_kind,
        device_identity=_identity(device_identity, name="device identity"),
        route_count=route_count,
        endpoint_count=endpoint_count,
        selected_endpoint_count=selected_endpoint_count,
        maximum_routes_per_fiber=maximum_routes_per_fiber,
        route_chunk_size=design.route_chunk_size,
        endpoint_lane_tile_size=lanes,
        maximum_fibers_per_tile=fibers,
        maximum_context_batch_size=design.maximum_context_batch_size,
        borrowed_label_bytes=0,
        compact_endpoint_bytes=endpoint_bytes,
        carry_bytes=carry,
        output_bytes=output,
        membership_tile_bytes=membership,
        value_tile_bytes=values,
        tree_intermediate_bytes=tree,
        allocator_guard_bytes=guard,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current,
        forecast_peak_reserved_bytes=current + incremental,
        device_total_or_workspace_ceiling_bytes=ceiling,
        admitted=allocated + incremental <= ceiling and incremental <= available,
        _admission_memory=_admission_memory,
    )


@dataclass(frozen=True, slots=True)
class V1LabelNativeProjectionGeometryReceipt:
    provider_receipt_hash: str
    provider_design_hash: str
    action_axis_registration_hash: str
    source_shell_ladder_receipt_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    route_axis_hash: str
    route_count: int
    parent_mask_hash: str
    source_endpoint_count: int
    qualified_lane_registration_hash: str
    qualified_lane_registration_used: bool
    selected_compact_provider_endpoint_indices: tuple[int, ...]
    selected_source_endpoint_labels: tuple[int, ...]
    selected_endpoint_labels: tuple[int, ...]
    selected_endpoint_counts: tuple[int, ...]
    selected_endpoint_action_hashes: tuple[str, ...]
    qualified_action_crosswalk_receipt_hashes: tuple[str, ...]
    qualified_canonical_physical_identity_hashes: tuple[str, ...]
    execution_design_hash: str
    hbm_forecast_hash: str
    backend_kind: str
    device_identity: str
    compact_label_count_d2h_bytes: int
    full_label_d2h_bytes: int = 0
    full_physical_index_d2h_bytes: int = 0
    endpoint_hard_masks_materialized: bool = False
    route_by_endpoint_mask_materialized: bool = False
    bounded_route_chunk_endpoint_membership_tile_used: bool = True
    stable_grouping_coordinates: tuple[str, ...] = (
        "route_chunk",
        "shell_label",
        "native_fiber",
        "physical_index",
    )
    arithmetic_id: str = LABEL_NATIVE_PROJECTION_ALGORITHM
    authority_scope: str = LABEL_NATIVE_PROJECTION_AUTHORITY
    schema: str = LABEL_NATIVE_PROJECTION_GEOMETRY_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "provider_receipt_hash",
            "provider_design_hash",
            "action_axis_registration_hash",
            "source_shell_ladder_receipt_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "route_axis_hash",
            "parent_mask_hash",
            "qualified_lane_registration_hash",
            "execution_design_hash",
            "hbm_forecast_hash",
            "device_identity",
            "arithmetic_id",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(self, "route_count", _positive_int(self.route_count, name="route count"))
        object.__setattr__(
            self,
            "source_endpoint_count",
            _positive_int(self.source_endpoint_count, name="source endpoint count"),
        )
        compact_indices = tuple(
            _positive_int(value, name="selected compact provider endpoint index")
            for value in self.selected_compact_provider_endpoint_indices
        )
        source_labels = tuple(
            _positive_int(value, name="selected source endpoint label")
            for value in self.selected_source_endpoint_labels
        )
        labels = tuple(_positive_int(value, name="selected endpoint label") for value in self.selected_endpoint_labels)
        counts = tuple(_positive_int(value, name="selected endpoint count") for value in self.selected_endpoint_counts)
        actions = tuple(_identity(value, name="selected endpoint action") for value in self.selected_endpoint_action_hashes)
        crosswalks = tuple(
            _identity(value, name="qualified action crosswalk receipt hash")
            for value in self.qualified_action_crosswalk_receipt_hashes
        )
        physical = tuple(
            _identity(value, name="qualified canonical physical identity hash")
            for value in self.qualified_canonical_physical_identity_hashes
        )
        if (
            not labels
            or tuple(sorted(set(labels))) != labels
            or source_labels != labels
            or compact_indices != tuple(range(1, len(labels) + 1))
            or len(labels) != len(counts)
            or len(labels) != len(actions)
            or tuple(sorted(counts)) != counts
            or labels[-1] > self.source_endpoint_count
        ):
            raise ValueError("selected endpoint geometry axis is invalid")
        if type(self.qualified_lane_registration_used) is not bool:
            raise TypeError("qualified lane-registration flag must be exact")
        if self.qualified_lane_registration_used:
            if len(crosswalks) != len(labels) or len(physical) != len(labels):
                raise ValueError("qualified projection crosswalk axis is incomplete")
        elif crosswalks or physical:
            raise ValueError("legacy identity projection cannot claim qualified crosswalks")
        object.__setattr__(
            self,
            "selected_compact_provider_endpoint_indices",
            compact_indices,
        )
        object.__setattr__(self, "selected_source_endpoint_labels", source_labels)
        object.__setattr__(self, "selected_endpoint_labels", labels)
        object.__setattr__(self, "selected_endpoint_counts", counts)
        object.__setattr__(self, "selected_endpoint_action_hashes", actions)
        object.__setattr__(
            self,
            "qualified_action_crosswalk_receipt_hashes",
            crosswalks,
        )
        object.__setattr__(
            self,
            "qualified_canonical_physical_identity_hashes",
            physical,
        )
        for name in (
            "compact_label_count_d2h_bytes",
            "full_label_d2h_bytes",
            "full_physical_index_d2h_bytes",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name=name))
        if self.compact_label_count_d2h_bytes != (self.source_endpoint_count + 1) * _I64_BYTES:
            raise ValueError("compact shell-count transfer does not replay")
        if self.backend_kind not in {"torch_cuda", "numpy_reference"}:
            raise ValueError("unsupported projection geometry backend")
        flags = (
            self.endpoint_hard_masks_materialized,
            self.route_by_endpoint_mask_materialized,
        )
        if (
            any(type(value) is not bool for value in flags)
            or any(flags)
            or self.bounded_route_chunk_endpoint_membership_tile_used is not True
            or self.full_label_d2h_bytes
            or self.full_physical_index_d2h_bytes
            or self.stable_grouping_coordinates
            != ("route_chunk", "shell_label", "native_fiber", "physical_index")
            or self.arithmetic_id != LABEL_NATIVE_PROJECTION_ALGORITHM
            or self.authority_scope != LABEL_NATIVE_PROJECTION_AUTHORITY
            or self.schema != LABEL_NATIVE_PROJECTION_GEOMETRY_SCHEMA
        ):
            _refuse(
                V1LabelNativeProjectionRefusalCode.AUTHORITY_VIOLATION,
                "projection geometry crossed its bounded identity authority",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "provider_receipt_hash",
                "provider_design_hash",
                "action_axis_registration_hash",
                "source_shell_ladder_receipt_hash",
                "native_fiber_axis_receipt_hash",
                "native_fiber_measure_hash",
                "route_axis_hash",
                "route_count",
                "parent_mask_hash",
                "source_endpoint_count",
                "qualified_lane_registration_hash",
                "qualified_lane_registration_used",
                "selected_compact_provider_endpoint_indices",
                "selected_source_endpoint_labels",
                "selected_endpoint_labels",
                "selected_endpoint_counts",
                "selected_endpoint_action_hashes",
                "qualified_action_crosswalk_receipt_hashes",
                "qualified_canonical_physical_identity_hashes",
                "execution_design_hash",
                "hbm_forecast_hash",
                "backend_kind",
                "device_identity",
                "compact_label_count_d2h_bytes",
                "full_label_d2h_bytes",
                "full_physical_index_d2h_bytes",
                "endpoint_hard_masks_materialized",
                "route_by_endpoint_mask_materialized",
                "bounded_route_chunk_endpoint_membership_tile_used",
                "stable_grouping_coordinates",
                "arithmetic_id",
                "authority_scope",
                "schema",
            )
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_GEOMETRY_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("label-native projection geometry receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("label-native projection geometry hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


def _outward_radius(
    center: np.ndarray,
    triangle: np.ndarray,
    *,
    operation_count: int,
    nonnegative: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if center.shape != triangle.shape or bool(np.any(triangle < 0.0)):
        raise ValueError("projection center/triangle axes are invalid")
    count = _positive_int(operation_count, name="numerical operation count")
    scale = count * _UNIT_ROUNDOFF
    if scale >= 0.5:
        raise FloatingPointError("projection numerical bound is unresolved")
    gamma = math.nextafter(scale / (1.0 - scale), math.inf)
    underflow = math.nextafter(count * np.finfo(np.float64).tiny, math.inf)
    triangle_upper = np.nextafter(
        (triangle + underflow) / (1.0 - gamma),
        math.inf,
    )
    radius = np.nextafter(gamma * triangle_upper + underflow, math.inf)
    lower = np.nextafter(center - radius, -math.inf)
    if nonnegative:
        lower = np.maximum(0.0, lower)
    upper = np.nextafter(center + radius, math.inf)
    if not bool(np.isfinite(radius).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise FloatingPointError("projection numerical enclosure overflowed")
    return radius, lower, upper


@dataclass(frozen=True, slots=True)
class V1LabelNativeProjectionReceipt:
    geometry_receipt_hash: str
    provider_receipt_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    chart_identity_hash: str
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    same_capture_and_vjp_registration_hash: str
    context_ids: tuple[str, ...]
    qualified_lane_registration_hash: str
    qualified_lane_registration_used: bool
    selected_compact_provider_endpoint_indices: tuple[int, ...]
    selected_source_endpoint_labels: tuple[int, ...]
    selected_endpoint_labels: tuple[int, ...]
    selected_endpoint_action_hashes: tuple[str, ...]
    selected_endpoint_counts: tuple[int, ...]
    qualified_action_crosswalk_receipt_hashes: tuple[str, ...]
    qualified_canonical_physical_identity_hashes: tuple[str, ...]
    route_count: int
    route_windows: tuple[tuple[int, int], ...]
    route_window_ledger_hash: str
    fibered_work_center_hash: str
    fibered_work_radius_hash: str
    fibered_work_lower_hash: str
    fibered_work_upper_hash: str
    global_literal_jacobian_center_hash: str
    global_literal_jacobian_radius_hash: str
    projection_source_hash: str
    numerical_backend: str
    production_cuda_execution: bool
    exact_device_to_host_bytes: int
    compact_scalar_device_to_host_bytes: int
    peak_registered_device_bytes_upper: int
    route_chunk_count: int
    floating_atomics_used: bool = False
    endpoint_hard_masks_materialized: bool = False
    route_by_endpoint_mask_materialized: bool = False
    bounded_route_chunk_endpoint_membership_tile_used: bool = True
    within_native_fiber_cancellation_only_for_F: bool = True
    global_signed_cancellation_for_J: bool = True
    candidate_selection_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    arithmetic_id: str = LABEL_NATIVE_PROJECTION_ALGORITHM
    numerical_enclosure_id: str = LABEL_NATIVE_PROJECTION_NUMERICAL_ENCLOSURE
    authority_scope: str = LABEL_NATIVE_PROJECTION_AUTHORITY
    schema: str = LABEL_NATIVE_PROJECTION_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "geometry_receipt_hash",
            "provider_receipt_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "chart_identity_hash",
            "corrected_evidence_stream_identity_hash",
            "literal_jacobian_stream_identity_hash",
            "same_capture_and_vjp_registration_hash",
            "qualified_lane_registration_hash",
            "route_window_ledger_hash",
            "fibered_work_center_hash",
            "fibered_work_radius_hash",
            "fibered_work_lower_hash",
            "fibered_work_upper_hash",
            "global_literal_jacobian_center_hash",
            "global_literal_jacobian_radius_hash",
            "projection_source_hash",
            "numerical_backend",
            "arithmetic_id",
            "numerical_enclosure_id",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        contexts = _identity_tuple(self.context_ids, name="projection context axis")
        compact_indices = tuple(
            _positive_int(value, name="selected compact provider endpoint index")
            for value in self.selected_compact_provider_endpoint_indices
        )
        source_labels = tuple(
            _positive_int(value, name="selected source endpoint label")
            for value in self.selected_source_endpoint_labels
        )
        labels = tuple(_positive_int(value, name="selected endpoint label") for value in self.selected_endpoint_labels)
        actions = tuple(_identity(value, name="selected endpoint action") for value in self.selected_endpoint_action_hashes)
        counts = tuple(
            _positive_int(value, name="selected endpoint count")
            for value in self.selected_endpoint_counts
        )
        crosswalks = tuple(
            _identity(value, name="qualified action crosswalk receipt hash")
            for value in self.qualified_action_crosswalk_receipt_hashes
        )
        physical = tuple(
            _identity(value, name="qualified canonical physical identity hash")
            for value in self.qualified_canonical_physical_identity_hashes
        )
        windows = tuple(
            (
                _nonnegative_int(left, name="route-window start"),
                _positive_int(right, name="route-window stop"),
            )
            for left, right in self.route_windows
        )
        if (
            len(labels) != len(actions)
            or len(labels) != len(counts)
            or source_labels != labels
            or compact_indices != tuple(range(1, len(labels) + 1))
            or tuple(sorted(counts)) != counts
            or tuple(sorted(set(labels))) != labels
            or not windows
            or windows[0][0] != 0
            or any(left >= right for left, right in windows)
            or any(windows[index - 1][1] != windows[index][0] for index in range(1, len(windows)))
        ):
            raise ValueError("projection receipt axes are invalid")
        if type(self.qualified_lane_registration_used) is not bool:
            raise TypeError("qualified lane-registration flag must be exact")
        if self.qualified_lane_registration_used:
            if len(crosswalks) != len(labels) or len(physical) != len(labels):
                raise ValueError("qualified projection crosswalk axis is incomplete")
        elif crosswalks or physical:
            raise ValueError("legacy projection cannot claim qualified crosswalks")
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(
            self,
            "selected_compact_provider_endpoint_indices",
            compact_indices,
        )
        object.__setattr__(self, "selected_source_endpoint_labels", source_labels)
        object.__setattr__(self, "selected_endpoint_labels", labels)
        object.__setattr__(self, "selected_endpoint_action_hashes", actions)
        object.__setattr__(self, "selected_endpoint_counts", counts)
        object.__setattr__(
            self,
            "qualified_action_crosswalk_receipt_hashes",
            crosswalks,
        )
        object.__setattr__(
            self,
            "qualified_canonical_physical_identity_hashes",
            physical,
        )
        object.__setattr__(self, "route_windows", windows)
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        if windows[-1][1] != self.route_count:
            raise ValueError("projection route windows do not cover the route axis")
        for name in (
            "exact_device_to_host_bytes",
            "compact_scalar_device_to_host_bytes",
            "peak_registered_device_bytes_upper",
            "route_chunk_count",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name=name))
        if self.route_chunk_count != len(windows):
            raise ValueError("projection route-chunk count does not replay")
        expected_d2h = len(contexts) * len(labels) * _F64_BYTES * 4
        if self.production_cuda_execution and self.exact_device_to_host_bytes != expected_d2h:
            raise ValueError("projection compact D2H accounting does not replay")
        if not self.production_cuda_execution and self.exact_device_to_host_bytes != 0:
            raise ValueError("CPU reference cannot claim device-to-host output transfer")
        if self.corrected_evidence_stream_identity_hash == self.literal_jacobian_stream_identity_hash:
            raise ValueError("corrected and literal-J stream identities collapsed")
        required_true = (
            self.within_native_fiber_cancellation_only_for_F,
            self.global_signed_cancellation_for_J,
            self.bounded_route_chunk_endpoint_membership_tile_used,
        )
        required_false = (
            self.floating_atomics_used,
            self.endpoint_hard_masks_materialized,
            self.route_by_endpoint_mask_materialized,
            self.candidate_selection_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            type(self.production_cuda_execution) is not bool
            or any(type(value) is not bool for value in required_true + required_false)
            or not all(required_true)
            or any(required_false)
            or self.arithmetic_id != LABEL_NATIVE_PROJECTION_ALGORITHM
            or self.numerical_enclosure_id != LABEL_NATIVE_PROJECTION_NUMERICAL_ENCLOSURE
            or self.authority_scope != LABEL_NATIVE_PROJECTION_AUTHORITY
            or self.schema != LABEL_NATIVE_PROJECTION_RECEIPT_SCHEMA
        ):
            _refuse(
                V1LabelNativeProjectionRefusalCode.AUTHORITY_VIOLATION,
                "projection receipt crossed measurement-only authority",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in (
                "geometry_receipt_hash",
                "provider_receipt_hash",
                "native_fiber_axis_receipt_hash",
                "native_fiber_measure_hash",
                "chart_identity_hash",
                "corrected_evidence_stream_identity_hash",
                "literal_jacobian_stream_identity_hash",
                "same_capture_and_vjp_registration_hash",
                "context_ids",
                "qualified_lane_registration_hash",
                "qualified_lane_registration_used",
                "selected_compact_provider_endpoint_indices",
                "selected_source_endpoint_labels",
                "selected_endpoint_labels",
                "selected_endpoint_action_hashes",
                "selected_endpoint_counts",
                "qualified_action_crosswalk_receipt_hashes",
                "qualified_canonical_physical_identity_hashes",
                "route_count",
                "route_windows",
                "route_window_ledger_hash",
                "fibered_work_center_hash",
                "fibered_work_radius_hash",
                "fibered_work_lower_hash",
                "fibered_work_upper_hash",
                "global_literal_jacobian_center_hash",
                "global_literal_jacobian_radius_hash",
                "projection_source_hash",
                "numerical_backend",
                "production_cuda_execution",
                "exact_device_to_host_bytes",
                "compact_scalar_device_to_host_bytes",
                "peak_registered_device_bytes_upper",
                "route_chunk_count",
                "floating_atomics_used",
                "endpoint_hard_masks_materialized",
                "route_by_endpoint_mask_materialized",
                "bounded_route_chunk_endpoint_membership_tile_used",
                "within_native_fiber_cancellation_only_for_F",
                "global_signed_cancellation_for_J",
                "candidate_selection_authority",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_authority",
                "arithmetic_id",
                "numerical_enclosure_id",
                "authority_scope",
                "schema",
            )
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("label-native projection receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("label-native projection receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1LabelNativeProjectionResult:
    receipt: V1LabelNativeProjectionReceipt
    fibered_work_center_by_context_endpoint: np.ndarray = field(repr=False, compare=False)
    fibered_work_radius_by_context_endpoint: np.ndarray = field(repr=False, compare=False)
    fibered_work_lower_by_context_endpoint: np.ndarray = field(repr=False, compare=False)
    fibered_work_upper_by_context_endpoint: np.ndarray = field(repr=False, compare=False)
    global_literal_jacobian_center_by_context_endpoint: np.ndarray = field(repr=False, compare=False)
    global_literal_jacobian_radius_by_context_endpoint: np.ndarray = field(repr=False, compare=False)
    schema: str = LABEL_NATIVE_PROJECTION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1LabelNativeProjectionReceipt:
            raise TypeError("projection result requires its exact receipt")
        self.receipt.assert_integrity()
        shape = (len(self.receipt.context_ids), len(self.receipt.selected_endpoint_labels))
        specifications = (
            ("fibered_work_center_by_context_endpoint", self.receipt.fibered_work_center_hash),
            ("fibered_work_radius_by_context_endpoint", self.receipt.fibered_work_radius_hash),
            ("fibered_work_lower_by_context_endpoint", self.receipt.fibered_work_lower_hash),
            ("fibered_work_upper_by_context_endpoint", self.receipt.fibered_work_upper_hash),
            ("global_literal_jacobian_center_by_context_endpoint", self.receipt.global_literal_jacobian_center_hash),
            ("global_literal_jacobian_radius_by_context_endpoint", self.receipt.global_literal_jacobian_radius_hash),
        )
        for name, expected_hash in specifications:
            matrix, observed_hash = _canonical_matrix(getattr(self, name), shape=shape, name=name)
            if observed_hash != expected_hash:
                raise ValueError(f"{name} differs from its projection receipt")
            object.__setattr__(self, name, matrix)
        if (
            bool(np.any(self.fibered_work_center_by_context_endpoint < 0.0))
            or bool(np.any(self.fibered_work_radius_by_context_endpoint < 0.0))
            or bool(np.any(self.fibered_work_lower_by_context_endpoint < 0.0))
            or bool(np.any(self.global_literal_jacobian_radius_by_context_endpoint < 0.0))
            or bool(np.any(self.fibered_work_lower_by_context_endpoint > self.fibered_work_center_by_context_endpoint))
            or bool(np.any(self.fibered_work_upper_by_context_endpoint < self.fibered_work_center_by_context_endpoint))
            or self.schema != LABEL_NATIVE_PROJECTION_RESULT_SCHEMA
        ):
            raise ValueError("projection result enclosure is invalid")


class V1LabelNativeProjectionGeometry:
    """Ephemeral label lease plus compact native-fiber projection geometry."""

    __slots__ = (
        "_active_accumulator",
        "_closed",
        "_device",
        "_endpoint_labels",
        "_label_lease",
        "_labels",
        "_provider",
        "_provider_token",
        "design",
        "forecast",
        "native_fiber_axis",
        "receipt",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        provider: V1LabelNativePackedDirectionProvider,
        native_fiber_axis: V1NativeDestinationFiberAxisReceipt,
        selected_endpoint_labels: Sequence[int] | None,
        design: V1LabelNativeProjectionCUDAExecutionDesign,
        production: bool,
        current_reserved_bytes: int | None,
        device_total_or_workspace_ceiling_bytes: int | None,
    ) -> None:
        if type(provider) is not V1LabelNativePackedDirectionProvider:
            raise TypeError("projection geometry requires the exact provider")
        if type(native_fiber_axis) is not V1NativeDestinationFiberAxisReceipt:
            raise TypeError("projection geometry requires the exact native-fiber axis")
        if type(design) is not V1LabelNativeProjectionCUDAExecutionDesign:
            raise TypeError("projection geometry requires the exact execution design")
        if type(production) is not bool:
            raise TypeError("production flag must be exact")
        native_fiber_axis.assert_integrity()
        provider._require_live()
        provider_receipt = provider.receipt
        executed_source_labels = provider_receipt.executed_source_endpoint_labels
        executed_counts = provider_receipt.executed_endpoint_counts
        executed_actions = provider_receipt.executed_endpoint_action_hashes
        labels = (
            executed_source_labels
            if selected_endpoint_labels is None
            else tuple(
                _positive_int(value, name="selected endpoint label")
                for value in selected_endpoint_labels
            )
        )
        if (
            not labels
            or tuple(sorted(set(labels))) != labels
            or labels[-1] > len(provider_receipt.endpoint_counts)
            or labels != executed_source_labels
        ):
            _refuse(
                V1LabelNativeProjectionRefusalCode.ACTION_AXIS_MISMATCH,
                "projection must execute the provider's exact pre-C source-label axis",
            )
        qualified = provider.qualified_lane_registration
        if qualified is not None:
            qualified_hash = qualified.assert_integrity()
            compact_indices = qualified.compact_provider_endpoint_labels
            crosswalk_hashes = qualified.action_crosswalk_receipt_hashes
            physical_identity_hashes = qualified.canonical_physical_identity_hashes
            if (
                qualified.source_endpoint_labels != labels
                or qualified.endpoint_counts != executed_counts
                or qualified.endpoint_action_hashes != executed_actions
                or compact_indices != tuple(range(1, len(labels) + 1))
            ):
                _refuse(
                    V1LabelNativeProjectionRefusalCode.ACTION_AXIS_MISMATCH,
                    "qualified compact-to-source lane registration does not replay",
                )
        else:
            if production:
                _refuse(
                    V1LabelNativeProjectionRefusalCode.ACTION_AXIS_MISMATCH,
                    "production projection requires the exact pre-C qualified lane axis",
                )
            compact_indices = tuple(range(1, len(labels) + 1))
            crosswalk_hashes = ()
            physical_identity_hashes = ()
            qualified_hash = _legacy_full_axis_lane_registration_hash(
                provider_receipt_hash=provider_receipt.receipt_hash,
                source_endpoint_labels=labels,
                endpoint_counts=executed_counts,
                endpoint_action_hashes=executed_actions,
            )
        if (
            native_fiber_axis.route_axis_hash != provider_receipt.route_axis_hash
            or native_fiber_axis.route_count != provider_receipt.route_count
        ):
            _refuse(
                V1LabelNativeProjectionRefusalCode.STALE_SOURCE,
                "native fibers and label-native provider name different route axes",
            )
        torch = _require_torch()
        backend = provider_receipt.backend_kind
        if production:
            if backend != "torch_cuda" or not design.production_cuda_required:
                _refuse(
                    V1LabelNativeProjectionRefusalCode.CPU_FALLBACK_FORBIDDEN,
                    "production projection requires the canonical CUDA design and provider",
                )
            _require_cuda_determinism(torch)
        else:
            if (
                backend != "numpy_reference"
                or design.production_cuda_required
                or not design.allow_private_cpu_reference
            ):
                _refuse(
                    V1LabelNativeProjectionRefusalCode.CPU_FALLBACK_FORBIDDEN,
                    "CPU projection is available only through the private reference gate",
                )
            if provider_receipt.route_count > design.private_cpu_route_limit:
                _refuse(
                    V1LabelNativeProjectionRefusalCode.CPU_REFERENCE_LIMIT_EXCEEDED,
                    "private CPU projection exceeds its registered route limit",
                )
        self._provider = provider
        self._provider_token = provider._reserve_packet_slot()
        self._label_lease = None
        self._labels = None
        self._closed = False
        self._active_accumulator = False
        try:
            lease = provider._label_handle.borrow_for_registered_operation(
                operation=V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
            )
            resident = lease._labels_for_registered_operation(
                operation=V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
            )
            if backend == "torch_cuda":
                if (
                    not isinstance(resident, torch.Tensor)
                    or resident.device.type != "cuda"
                    or resident.dtype is not torch.uint8
                    or resident.ndim != 1
                    or not resident.is_contiguous()
                ):
                    _refuse(
                        V1LabelNativeProjectionRefusalCode.LABEL_IDENTITY_MISMATCH,
                        "projection requires contiguous CUDA uint8 shell labels",
                    )
                labels_tensor = resident
                device = resident.device
                current = (
                    int(torch.cuda.memory_reserved(device))
                    if current_reserved_bytes is None
                    else _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
                )
                if device_total_or_workspace_ceiling_bytes is None:
                    _free, total = torch.cuda.mem_get_info(device)
                    ceiling = math.floor(int(total) * design.hbm_ceiling_fraction)
                else:
                    ceiling = _positive_int(
                        device_total_or_workspace_ceiling_bytes,
                        name="device memory ceiling",
                    )
            else:
                source = np.asarray(resident)
                labels_tensor = torch.tensor(source, dtype=torch.uint8, device="cpu")
                device = torch.device("cpu")
                current = 0 if current_reserved_bytes is None else _nonnegative_int(
                    current_reserved_bytes,
                    name="current reserved bytes",
                )
                ceiling = (
                    design.private_cpu_workspace_ceiling_bytes
                    if device_total_or_workspace_ceiling_bytes is None
                    else _positive_int(
                        device_total_or_workspace_ceiling_bytes,
                        name="CPU workspace ceiling",
                    )
                )
            admission_memory = None
            if backend == "torch_cuda" and current_reserved_bytes is None:
                admission_memory = (cuda_reusable_memory_bytes(device, framework=torch), int(torch.cuda.memory_allocated(device)))
            maximum_rpf = max(cut.routes_per_fiber for cut in native_fiber_axis.cuts)
            forecast = build_v1_label_native_projection_cuda_hbm_forecast(
                design=design,
                provider_receipt_hash=provider_receipt.receipt_hash,
                native_fiber_axis_receipt_hash=native_fiber_axis.receipt_hash,
                backend_kind=backend,
                device_identity=str(device),
                route_count=provider_receipt.route_count,
                endpoint_count=len(provider_receipt.endpoint_counts),
                selected_endpoint_count=len(labels),
                maximum_routes_per_fiber=maximum_rpf,
                current_reserved_bytes=current,
                device_total_or_workspace_ceiling_bytes=ceiling,
                _admission_memory=admission_memory,
            )
            if not forecast.admitted:
                _refuse(
                    V1LabelNativeProjectionRefusalCode.HBM_FORECAST_EXCEEDED,
                    "projection workspace exceeds the registered memory ceiling",
                )
            counts_device = torch.bincount(
                labels_tensor.to(dtype=torch.int64),
                minlength=len(provider_receipt.endpoint_counts) + 1,
            )
            counts_host = counts_device.detach().to("cpu").numpy()
            if (
                counts_host.shape != (len(provider_receipt.endpoint_counts) + 1,)
                or tuple(np.cumsum(counts_host[1:]).tolist())
                != provider_receipt.endpoint_counts
            ):
                _refuse(
                    V1LabelNativeProjectionRefusalCode.LABEL_IDENTITY_MISMATCH,
                    "resident shell counts differ from the frozen action axis",
                )
            source = provider_receipt.action_axis_registration.source_shell_ladder
            selected_counts = executed_counts
            selected_actions = executed_actions
            receipt = V1LabelNativeProjectionGeometryReceipt(
                provider_receipt_hash=provider_receipt.receipt_hash,
                provider_design_hash=provider_receipt.design_hash,
                action_axis_registration_hash=provider_receipt.action_axis_registration.registration_hash,
                source_shell_ladder_receipt_hash=source.receipt_hash,
                native_fiber_axis_receipt_hash=native_fiber_axis.receipt_hash,
                native_fiber_measure_hash=native_fiber_axis.measure_hash,
                route_axis_hash=provider_receipt.route_axis_hash,
                route_count=provider_receipt.route_count,
                parent_mask_hash=provider_receipt.parent_mask_hash,
                source_endpoint_count=len(provider_receipt.endpoint_counts),
                qualified_lane_registration_hash=qualified_hash,
                qualified_lane_registration_used=(qualified is not None),
                selected_compact_provider_endpoint_indices=compact_indices,
                selected_source_endpoint_labels=labels,
                selected_endpoint_labels=labels,
                selected_endpoint_counts=selected_counts,
                selected_endpoint_action_hashes=selected_actions,
                qualified_action_crosswalk_receipt_hashes=crosswalk_hashes,
                qualified_canonical_physical_identity_hashes=(
                    physical_identity_hashes
                ),
                execution_design_hash=design.design_hash,
                hbm_forecast_hash=forecast.forecast_hash,
                backend_kind=backend,
                device_identity=str(device),
                compact_label_count_d2h_bytes=(len(provider_receipt.endpoint_counts) + 1) * _I64_BYTES,
            )
            self.design = design
            self.forecast = forecast
            self.native_fiber_axis = native_fiber_axis
            self.receipt = receipt
            self._device = device
            self._endpoint_labels = torch.tensor(labels, dtype=torch.uint8, device=device)
            self._label_lease = lease
            self._labels = labels_tensor
            del counts_device, counts_host
        except BaseException:
            lease_value = self._label_lease if self._label_lease is not None else locals().get("lease")
            if type(lease_value) is V1DeviceShellLabelLease and not lease_value.is_closed:
                try:
                    lease_value.close()
                except BaseException:
                    pass
            provider._cancel_packet_reservation(self._provider_token)
            self._provider_token = None
            self._closed = True
            raise

    @classmethod
    def from_provider(
        cls,
        *,
        provider: V1LabelNativePackedDirectionProvider,
        native_fiber_axis: V1NativeDestinationFiberAxisReceipt,
        selected_endpoint_labels: Sequence[int] | None = None,
        design: V1LabelNativeProjectionCUDAExecutionDesign = V1LabelNativeProjectionCUDAExecutionDesign(),
        current_reserved_bytes: int | None = None,
        device_total_memory_bytes: int | None = None,
    ) -> "V1LabelNativeProjectionGeometry":
        ceiling = None
        if device_total_memory_bytes is not None:
            ceiling = math.floor(
                _positive_int(device_total_memory_bytes, name="device total memory bytes")
                * design.hbm_ceiling_fraction
            )
        return cls(
            provider=provider,
            native_fiber_axis=native_fiber_axis,
            selected_endpoint_labels=selected_endpoint_labels,
            design=design,
            production=True,
            current_reserved_bytes=current_reserved_bytes,
            device_total_or_workspace_ceiling_bytes=ceiling,
        )

    @classmethod
    def _from_provider_reference_for_tests(
        cls,
        *,
        provider: V1LabelNativePackedDirectionProvider,
        native_fiber_axis: V1NativeDestinationFiberAxisReceipt,
        selected_endpoint_labels: Sequence[int],
        design: V1LabelNativeProjectionCUDAExecutionDesign,
    ) -> "V1LabelNativeProjectionGeometry":
        return cls(
            provider=provider,
            native_fiber_axis=native_fiber_axis,
            selected_endpoint_labels=selected_endpoint_labels,
            design=design,
            production=False,
            current_reserved_bytes=0,
            device_total_or_workspace_ceiling_bytes=design.private_cpu_workspace_ceiling_bytes,
        )

    @property
    def is_closed(self) -> bool:
        return bool(self._closed)

    @property
    def has_active_accumulator(self) -> bool:
        return bool(self._active_accumulator)

    def _require_live(self) -> None:
        if self._closed or self._labels is None or self._label_lease is None:
            _refuse(
                V1LabelNativeProjectionRefusalCode.GEOMETRY_CLOSED,
                "label-native projection geometry is closed",
            )
        self.receipt.assert_integrity()
        self.native_fiber_axis.assert_integrity()
        self._label_lease._labels_for_registered_operation(
            operation=V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
        )

    def open_accumulator(
        self,
        *,
        context_ids: Sequence[str],
        chart_identity_hash: str,
        corrected_evidence_stream_identity_hash: str,
        literal_jacobian_stream_identity_hash: str,
        same_capture_and_vjp_registration_hash: str,
    ) -> "V1LabelNativeProjectionCUDAAccumulator":
        self._require_live()
        if self._active_accumulator:
            _refuse(
                V1LabelNativeProjectionRefusalCode.GEOMETRY_BUSY,
                "projection geometry already owns an active accumulator",
            )
        self._active_accumulator = True
        try:
            return V1LabelNativeProjectionCUDAAccumulator(
                _geometry=self,
                context_ids=context_ids,
                chart_identity_hash=chart_identity_hash,
                corrected_evidence_stream_identity_hash=corrected_evidence_stream_identity_hash,
                literal_jacobian_stream_identity_hash=literal_jacobian_stream_identity_hash,
                same_capture_and_vjp_registration_hash=(
                    same_capture_and_vjp_registration_hash
                ),
            )
        except BaseException:
            self._active_accumulator = False
            raise

    def _release_accumulator(self) -> None:
        self._active_accumulator = False

    def close(self) -> None:
        if self._closed:
            return
        if self._active_accumulator:
            _refuse(
                V1LabelNativeProjectionRefusalCode.GEOMETRY_BUSY,
                "close or finalize the projection accumulator before geometry cleanup",
            )
        lease = self._label_lease
        provider = self._provider
        token = self._provider_token
        self._labels = None
        self._endpoint_labels = None
        self._label_lease = None
        self._provider_token = None
        self._closed = True
        lease_error: BaseException | None = None
        try:
            if lease is not None and not lease.is_closed:
                lease.close()
        except BaseException as error:
            lease_error = error
        finally:
            if provider is not None and token is not None:
                provider._cancel_packet_reservation(token)
        if lease_error is not None:
            raise lease_error

    def __enter__(self) -> "V1LabelNativeProjectionGeometry":
        self._require_live()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("projection geometry leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("projection geometry leases are nonserializable")


class V1LabelNativeProjectionCUDAAccumulator:
    """One same-capture dual-stream projection transaction."""

    __slots__ = (
        "_carry_coordinate",
        "_carry_signed",
        "_closed",
        "_context_ids",
        "_corrected_stream_hash",
        "_f_sum",
        "_f_triangle",
        "_geometry",
        "_j_sum",
        "_j_stream_hash",
        "_j_triangle",
        "_next_route",
        "_same_capture_hash",
        "_scalar_d2h_bytes",
        "_windows",
        "chart_identity_hash",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        _geometry: V1LabelNativeProjectionGeometry,
        context_ids: Sequence[str],
        chart_identity_hash: str,
        corrected_evidence_stream_identity_hash: str,
        literal_jacobian_stream_identity_hash: str,
        same_capture_and_vjp_registration_hash: str,
    ) -> None:
        contexts = _identity_tuple(context_ids, name="projection context axis")
        if len(contexts) > _geometry.design.maximum_context_batch_size:
            raise ValueError("projection context batch exceeds its frozen maximum")
        corrected = _identity(
            corrected_evidence_stream_identity_hash,
            name="corrected-evidence stream identity",
        )
        jacobian = _identity(
            literal_jacobian_stream_identity_hash,
            name="literal-J stream identity",
        )
        if corrected == jacobian:
            raise ValueError("corrected-evidence and literal-J streams must remain distinct")
        self._geometry = _geometry
        self._context_ids = contexts
        self.chart_identity_hash = _identity(chart_identity_hash, name="chart identity")
        self._corrected_stream_hash = corrected
        self._j_stream_hash = jacobian
        self._same_capture_hash = _identity(
            same_capture_and_vjp_registration_hash,
            name="same-capture/VJP registration",
        )
        torch = _require_torch()
        shape = (len(contexts), len(_geometry.receipt.selected_endpoint_labels))
        self._f_sum = torch.zeros(shape, dtype=torch.float64, device=_geometry._device)
        self._f_triangle = torch.zeros_like(self._f_sum)
        self._j_sum = torch.zeros_like(self._f_sum)
        self._j_triangle = torch.zeros_like(self._f_sum)
        self._carry_signed = None
        self._carry_coordinate: tuple[int, int] | None = None
        self._next_route = 0
        self._windows: list[tuple[int, int]] = []
        self._scalar_d2h_bytes = 0
        self._closed = False

    def _require_live(self) -> None:
        if self._closed:
            _refuse(
                V1LabelNativeProjectionRefusalCode.FINALIZED,
                "projection accumulator is already closed",
            )
        self._geometry._require_live()

    def _project_flat(self, rows: Any, labels: Any) -> tuple[Any, Any]:
        torch = _require_torch()
        batch = len(self._context_ids)
        endpoints = self._geometry._endpoint_labels
        lane_width = self._geometry.forecast.endpoint_lane_tile_size
        signed_chunks = []
        triangle_chunks = []
        for left in range(0, int(endpoints.numel()), lane_width):
            right = min(int(endpoints.numel()), left + lane_width)
            endpoint_tile = endpoints[left:right]
            membership = (labels.unsqueeze(0) != 0) & (
                labels.unsqueeze(0) <= endpoint_tile.unsqueeze(1)
            )
            gathered = rows.unsqueeze(1) * membership.unsqueeze(0)
            signed_chunks.append(_fixed_tree_reduce_last(torch, gathered))
            gathered.abs_()
            triangle_chunks.append(_fixed_tree_reduce_last(torch, gathered))
            del endpoint_tile, membership, gathered
        signed = torch.cat(signed_chunks, dim=1).reshape(batch, -1)
        triangle = torch.cat(triangle_chunks, dim=1).reshape(batch, -1)
        return signed, triangle

    def _project_complete_fibers(
        self,
        rows: Any,
        labels: Any,
        *,
        fiber_count: int,
        routes_per_fiber: int,
    ) -> None:
        torch = _require_torch()
        batch = len(self._context_ids)
        endpoints = self._geometry._endpoint_labels
        lane_width = self._geometry.forecast.endpoint_lane_tile_size
        max_fibers = self._geometry.forecast.maximum_fibers_per_tile
        for fiber_left in range(0, fiber_count, max_fibers):
            fiber_right = min(fiber_count, fiber_left + max_fibers)
            fibers = fiber_right - fiber_left
            route_left = fiber_left * routes_per_fiber
            route_right = fiber_right * routes_per_fiber
            row_tile = rows[:, route_left:route_right].reshape(
                batch, fibers, routes_per_fiber
            )
            label_tile = labels[route_left:route_right].reshape(
                fibers, routes_per_fiber
            )
            for lane_left in range(0, int(endpoints.numel()), lane_width):
                lane_right = min(int(endpoints.numel()), lane_left + lane_width)
                endpoint_tile = endpoints[lane_left:lane_right]
                membership = (label_tile.unsqueeze(0) != 0) & (
                    label_tile.unsqueeze(0)
                    <= endpoint_tile.reshape(-1, 1, 1)
                )
                values = row_tile.unsqueeze(1) * membership.unsqueeze(0)
                fiber_signed = _fixed_tree_reduce_last(torch, values)
                fiber_absolute = torch.abs(fiber_signed)
                fiber_sum = _fixed_tree_reduce_last(torch, fiber_absolute)
                self._f_sum[:, lane_left:lane_right].add_(fiber_sum)
                del (
                    endpoint_tile,
                    membership,
                    values,
                    fiber_signed,
                    fiber_absolute,
                    fiber_sum,
                )
            del row_tile, label_tile

    def _consume_partial_fiber(
        self,
        *,
        cut: V1NativeDestinationFiberCut,
        fiber_index_within_cut: int,
        rows: Any,
        labels: Any,
        completes_fiber: bool,
    ) -> None:
        signed, _triangle = self._project_flat(rows, labels)
        coordinate = (cut.cut_index, fiber_index_within_cut)
        if self._carry_signed is None:
            self._carry_signed = signed
            self._carry_coordinate = coordinate
        else:
            if self._carry_coordinate != coordinate:
                _refuse(
                    V1LabelNativeProjectionRefusalCode.PARTIAL_FIBER_MISMATCH,
                    "route chunks changed the native-fiber boundary carry",
                )
            self._carry_signed = self._carry_signed + signed
        if completes_fiber:
            self._f_sum.add_(torch_abs(self._carry_signed))
            self._carry_signed = None
            self._carry_coordinate = None

    def _consume_cut_segment(
        self,
        *,
        cut: V1NativeDestinationFiberCut,
        global_start: int,
        global_stop: int,
        chunk_start: int,
        corrected_rows: Any,
        labels: Any,
    ) -> None:
        rpf = cut.routes_per_fiber
        local_start = global_start - cut.route_offset
        local_stop = global_stop - cut.route_offset
        cursor = local_start

        def slices(begin_local: int, end_local: int) -> tuple[Any, Any]:
            left = cut.route_offset + begin_local - chunk_start
            right = cut.route_offset + end_local - chunk_start
            return corrected_rows[:, left:right], labels[left:right]

        offset = cursor % rpf
        if offset:
            end = min(local_stop, cursor + (rpf - offset))
            row_slice, label_slice = slices(cursor, end)
            self._consume_partial_fiber(
                cut=cut,
                fiber_index_within_cut=cursor // rpf,
                rows=row_slice,
                labels=label_slice,
                completes_fiber=(end % rpf == 0),
            )
            cursor = end
        remaining = local_stop - cursor
        complete = remaining // rpf
        if complete:
            end = cursor + complete * rpf
            row_slice, label_slice = slices(cursor, end)
            self._project_complete_fibers(
                row_slice,
                label_slice,
                fiber_count=complete,
                routes_per_fiber=rpf,
            )
            cursor = end
        if cursor < local_stop:
            row_slice, label_slice = slices(cursor, local_stop)
            self._consume_partial_fiber(
                cut=cut,
                fiber_index_within_cut=cursor // rpf,
                rows=row_slice,
                labels=label_slice,
                completes_fiber=False,
            )

    def consume_dual_chart_chunk(
        self,
        start: int,
        stop: int,
        *,
        corrected_evidence_rows: Any,
        literal_jacobian_rows: Any,
    ) -> None:
        self._require_live()
        torch = _require_torch()
        begin = _nonnegative_int(start, name="route chunk start")
        end = _positive_int(stop, name="route chunk stop")
        span = end - begin
        batch = len(self._context_ids)
        if (
            begin != self._next_route
            or span <= 0
            or end > self._geometry.receipt.route_count
            or span > self._geometry.design.route_chunk_size
        ):
            _refuse(
                V1LabelNativeProjectionRefusalCode.ORDER_MISMATCH,
                "projection chunks must cover one bounded physical-axis prefix",
            )
        rows = (corrected_evidence_rows, literal_jacobian_rows)
        for name, value in zip(("corrected evidence", "literal Jacobian"), rows, strict=True):
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (batch, span)
                or value.dtype is not torch.float64
                or value.device != self._geometry._device
                or not value.is_contiguous()
            ):
                _refuse(
                    V1LabelNativeProjectionRefusalCode.INVALID_ROW,
                    f"{name} rows must be contiguous float64 [context, route] on the geometry device",
                )
        finite = torch.isfinite(corrected_evidence_rows).all()
        finite.logical_and_(torch.isfinite(literal_jacobian_rows).all())
        if not bool(finite.item()):
            _refuse(
                V1LabelNativeProjectionRefusalCode.NONFINITE_ROW,
                "one or both dual-chart route streams contain nonfinite values",
            )
        if corrected_evidence_rows.device.type == "cuda":
            self._scalar_d2h_bytes += 1
        del finite
        self._geometry._label_lease._labels_for_registered_operation(
            operation=V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
        )
        labels = self._geometry._labels[begin:end]
        j_signed, j_triangle = self._project_flat(literal_jacobian_rows, labels)
        _corrected_signed, corrected_triangle = self._project_flat(
            corrected_evidence_rows,
            labels,
        )
        self._j_sum.add_(j_signed)
        self._j_triangle.add_(j_triangle)
        self._f_triangle.add_(corrected_triangle)
        for cut in self._geometry.native_fiber_axis.cuts:
            segment_start = max(begin, cut.route_offset)
            segment_stop = min(end, cut.route_stop)
            if segment_start < segment_stop:
                self._consume_cut_segment(
                    cut=cut,
                    global_start=segment_start,
                    global_stop=segment_stop,
                    chunk_start=begin,
                    corrected_rows=corrected_evidence_rows,
                    labels=labels,
                )
        self._windows.append((begin, end))
        self._next_route = end
        del j_signed, j_triangle, _corrected_signed, corrected_triangle

    def finalize(self) -> V1LabelNativeProjectionResult:
        self._require_live()
        if self._next_route != self._geometry.receipt.route_count:
            _refuse(
                V1LabelNativeProjectionRefusalCode.INCOMPLETE_ROUTE_AXIS,
                "projection cannot finalize before the complete route axis",
            )
        if self._carry_signed is not None or self._carry_coordinate is not None:
            _refuse(
                V1LabelNativeProjectionRefusalCode.PARTIAL_FIBER_MISMATCH,
                "projection ended with an incomplete native fiber",
            )
        cuts = float(self._geometry.native_fiber_axis.cut_count)
        f_center_device = self._f_sum / cuts
        f_triangle_device = self._f_triangle / cuts
        j_center_device = self._j_sum
        j_triangle_device = self._j_triangle
        shape = (len(self._context_ids), len(self._geometry.receipt.selected_endpoint_labels))
        f_center_raw = f_center_device.detach().to("cpu").numpy()
        f_triangle_raw = f_triangle_device.detach().to("cpu").numpy()
        j_center_raw = j_center_device.detach().to("cpu").numpy()
        j_triangle_raw = j_triangle_device.detach().to("cpu").numpy()
        f_center, f_center_hash = _canonical_matrix(f_center_raw, shape=shape, name="fibered-work center")
        f_triangle, _ = _canonical_matrix(f_triangle_raw, shape=shape, name="fibered-work triangle")
        j_center, j_center_hash = _canonical_matrix(j_center_raw, shape=shape, name="literal-J center")
        j_triangle, _ = _canonical_matrix(j_triangle_raw, shape=shape, name="literal-J triangle")
        route_count = self._geometry.receipt.route_count
        fiber_count = self._geometry.native_fiber_axis.fiber_count
        f_radius_raw, f_lower_raw, f_upper_raw = _outward_radius(
            f_center,
            f_triangle,
            operation_count=route_count + fiber_count + len(self._windows) + 16,
            nonnegative=True,
        )
        j_radius_raw, _j_lower, _j_upper = _outward_radius(
            j_center,
            j_triangle,
            operation_count=route_count + len(self._windows) + 8,
            nonnegative=False,
        )
        f_radius, f_radius_hash = _canonical_matrix(f_radius_raw, shape=shape, name="fibered-work radius")
        f_lower, f_lower_hash = _canonical_matrix(f_lower_raw, shape=shape, name="fibered-work lower")
        f_upper, f_upper_hash = _canonical_matrix(f_upper_raw, shape=shape, name="fibered-work upper")
        j_radius, j_radius_hash = _canonical_matrix(j_radius_raw, shape=shape, name="literal-J radius")
        windows = tuple(self._windows)
        window_hash = content_hash(
            {
                "geometry_receipt_hash": self._geometry.receipt.receipt_hash,
                "route_count": route_count,
                "route_windows": windows,
            },
            prefix=_WINDOW_PREFIX,
        )
        cuda = self._geometry.receipt.backend_kind == "torch_cuda"
        receipt = V1LabelNativeProjectionReceipt(
            geometry_receipt_hash=self._geometry.receipt.receipt_hash,
            provider_receipt_hash=self._geometry.receipt.provider_receipt_hash,
            native_fiber_axis_receipt_hash=self._geometry.native_fiber_axis.receipt_hash,
            native_fiber_measure_hash=self._geometry.native_fiber_axis.measure_hash,
            chart_identity_hash=self.chart_identity_hash,
            corrected_evidence_stream_identity_hash=self._corrected_stream_hash,
            literal_jacobian_stream_identity_hash=self._j_stream_hash,
            same_capture_and_vjp_registration_hash=self._same_capture_hash,
            context_ids=self._context_ids,
            qualified_lane_registration_hash=(
                self._geometry.receipt.qualified_lane_registration_hash
            ),
            qualified_lane_registration_used=(
                self._geometry.receipt.qualified_lane_registration_used
            ),
            selected_compact_provider_endpoint_indices=(
                self._geometry.receipt.selected_compact_provider_endpoint_indices
            ),
            selected_source_endpoint_labels=(
                self._geometry.receipt.selected_source_endpoint_labels
            ),
            selected_endpoint_labels=self._geometry.receipt.selected_endpoint_labels,
            selected_endpoint_action_hashes=self._geometry.receipt.selected_endpoint_action_hashes,
            selected_endpoint_counts=self._geometry.receipt.selected_endpoint_counts,
            qualified_action_crosswalk_receipt_hashes=(
                self._geometry.receipt.qualified_action_crosswalk_receipt_hashes
            ),
            qualified_canonical_physical_identity_hashes=(
                self._geometry.receipt.qualified_canonical_physical_identity_hashes
            ),
            route_count=route_count,
            route_windows=windows,
            route_window_ledger_hash=window_hash,
            fibered_work_center_hash=f_center_hash,
            fibered_work_radius_hash=f_radius_hash,
            fibered_work_lower_hash=f_lower_hash,
            fibered_work_upper_hash=f_upper_hash,
            global_literal_jacobian_center_hash=j_center_hash,
            global_literal_jacobian_radius_hash=j_radius_hash,
            projection_source_hash=label_native_projection_cuda_source_hash(),
            numerical_backend=(
                "torch_cuda_fp64_fixed_tree_v1" if cuda else "torch_cpu_fp64_private_reference_fixed_tree_v1"
            ),
            production_cuda_execution=cuda,
            exact_device_to_host_bytes=(shape[0] * shape[1] * _F64_BYTES * 4 if cuda else 0),
            compact_scalar_device_to_host_bytes=self._scalar_d2h_bytes,
            peak_registered_device_bytes_upper=self._geometry.forecast.forecast_peak_reserved_bytes,
            route_chunk_count=len(windows),
        )
        result = V1LabelNativeProjectionResult(
            receipt=receipt,
            fibered_work_center_by_context_endpoint=f_center,
            fibered_work_radius_by_context_endpoint=f_radius,
            fibered_work_lower_by_context_endpoint=f_lower,
            fibered_work_upper_by_context_endpoint=f_upper,
            global_literal_jacobian_center_by_context_endpoint=j_center,
            global_literal_jacobian_radius_by_context_endpoint=j_radius,
        )
        self._closed = True
        self._geometry._release_accumulator()
        self._clear_device_state()
        return result

    def _clear_device_state(self) -> None:
        self._f_sum = None
        self._f_triangle = None
        self._j_sum = None
        self._j_triangle = None
        self._carry_signed = None
        self._carry_coordinate = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._geometry._release_accumulator()
        self._clear_device_state()

    def __enter__(self) -> "V1LabelNativeProjectionCUDAAccumulator":
        self._require_live()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("projection accumulators are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("projection accumulators are nonserializable")


def torch_abs(value: Any) -> Any:
    """Small indirection keeping the boundary-carry path easy to audit."""

    torch = _require_torch()
    return torch.abs(value)


def _project_v1_label_native_reference_for_tests(
    *,
    labels: np.ndarray,
    corrected_evidence_rows: np.ndarray,
    literal_jacobian_rows: np.ndarray,
    native_fiber_axis: V1NativeDestinationFiberAxisReceipt,
    selected_endpoint_labels: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """High-precision small-domain oracle; never a production fallback."""

    from decimal import Decimal, localcontext

    native_fiber_axis.assert_integrity()
    label_values = np.asarray(labels)
    corrected = np.asarray(corrected_evidence_rows, dtype=np.float64)
    jacobian = np.asarray(literal_jacobian_rows, dtype=np.float64)
    endpoints = tuple(int(value) for value in selected_endpoint_labels)
    if (
        label_values.dtype != np.dtype(np.uint8)
        or label_values.shape != (native_fiber_axis.route_count,)
        or corrected.ndim != 2
        or jacobian.shape != corrected.shape
        or corrected.shape[1] != native_fiber_axis.route_count
        or not bool(np.isfinite(corrected).all() and np.isfinite(jacobian).all())
        or not endpoints
        or tuple(sorted(set(endpoints))) != endpoints
    ):
        raise ValueError("private reference axes are invalid")
    f = np.empty((corrected.shape[0], len(endpoints)), dtype=np.float64)
    j = np.empty_like(f)
    with localcontext() as context:
        context.prec = 100
        for row in range(corrected.shape[0]):
            for endpoint_index, endpoint in enumerate(endpoints):
                f_total = Decimal(0)
                for cut in native_fiber_axis.cuts:
                    cut_total = Decimal(0)
                    for fiber in range(cut.fiber_count):
                        left, right = cut.route_span_for_fiber(fiber)
                        signed = Decimal(0)
                        for route in range(left, right):
                            if 0 < int(label_values[route]) <= endpoint:
                                signed += Decimal.from_float(float(corrected[row, route]))
                        cut_total += abs(signed)
                    f_total += cut_total
                f[row, endpoint_index] = float(
                    f_total / Decimal(native_fiber_axis.cut_count)
                )
                signed_j = Decimal(0)
                for route in range(native_fiber_axis.route_count):
                    if 0 < int(label_values[route]) <= endpoint:
                        signed_j += Decimal.from_float(float(jacobian[row, route]))
                j[row, endpoint_index] = float(signed_j)
    return f, j


__all__ = [
    "LABEL_NATIVE_PROJECTION_ALGORITHM",
    "LABEL_NATIVE_PROJECTION_AUTHORITY",
    "V1LabelNativeProjectionCUDAAccumulator",
    "V1LabelNativeProjectionCUDAExecutionDesign",
    "V1LabelNativeProjectionCUDAHBMForecast",
    "V1LabelNativeProjectionGeometry",
    "V1LabelNativeProjectionGeometryReceipt",
    "V1LabelNativeProjectionReceipt",
    "V1LabelNativeProjectionRefusal",
    "V1LabelNativeProjectionRefusalCode",
    "V1LabelNativeProjectionResult",
    "build_v1_label_native_projection_cuda_hbm_forecast",
    "label_native_projection_cuda_source_hash",
]
