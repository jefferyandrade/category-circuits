"""Shard-native weighted two-pass literal-J projection for ``D_fixed``.

This module is intentionally narrow.  It consumes an already-registered
sampled block-support replay plan plus its numeric-source receipt, then
streams literal relaxed-gate Jacobian shards twice:

1. pass 1 computes one resident per-context projection ``p = J z``;
2. pass 2 replays the identical ``D_fixed`` packet/tile order and accumulates
   ``M_J z = E[p J]`` and ``Q(z) = E[p^2]`` under the source-owned rational
   structural ``D_fixed`` measure.

It does not retain a context-by-route bank, does not transfer route-scale rows
to host, and grants no downstream calibration/oracle/commit authority.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import InitVar, dataclass, field
from enum import Enum
from fractions import Fraction
import hashlib
import math
import os
from types import MappingProxyType
from typing import Any, Final, Literal, Mapping, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_affine_shell_direction_cuda import (
    V1LiteralJAffineShellDirectionDeviceHandle,
    V1LiteralJAffineShellDirectionLease,
)

try:  # pragma: no cover - runtime-image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


LITERAL_J_TWO_PASS_D_FIXED_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_literal_j_two_pass_D_fixed_projection_receipt_v1"
)
LITERAL_J_TWO_PASS_D_FIXED_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_literal_j_two_pass_D_fixed_projection_registration_v1"
)
LITERAL_J_TWO_PASS_D_FIXED_PACKET_SCHEMA: Final = (
    "partition_free_v1_literal_j_two_pass_D_fixed_projection_packet_v1"
)
LITERAL_J_TWO_PASS_D_FIXED_DESIGN_SCHEMA: Final = (
    "partition_free_v1_literal_j_two_pass_D_fixed_projection_design_v1"
)
LITERAL_J_TWO_PASS_D_FIXED_TELEMETRY_SCHEMA: Final = (
    "partition_free_v1_literal_j_two_pass_D_fixed_projection_telemetry_v1"
)
LITERAL_J_TWO_PASS_D_FIXED_HBM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_literal_j_two_pass_D_fixed_projection_hbm_forecast_v1"
)
LITERAL_J_TWO_PASS_D_FIXED_ALGORITHM: Final = (
    "weighted_D_fixed_two_pass_literal_J_route_tile_projection_fp64_v1"
)
LITERAL_J_TWO_PASS_D_FIXED_AUTHORITY: Final = (
    "projection_only_no_D_decision_no_C_K_O_commit_terminal_or_pareto_v1"
)

_PREFIX: Final = "partition-free-v1-literal-j-two-pass-D-fixed-projection-"
_F64_BYTES: Final = 8
_ISSUER_TOKEN: Final = object()
_EPS: Final = 2.0**-53
_F64_MIN_SUBNORMAL: Final = float.fromhex("0x0.0000000000001p-1022")


class V1LiteralJTwoPassDFixedProjectionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    STALE_IDENTITY = "stale_identity"
    DEVICE_REQUIRED = "device_required"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    DETERMINISM_DISABLED = "determinism_disabled"
    ORDER_MISMATCH = "order_mismatch"
    PARTIAL_PACKET = "partial_packet"
    INVALID_ROW = "invalid_row"
    REWIND_REQUIRED = "rewind_required"
    FINALIZED = "finalized"
    EVICTED = "evicted"
    ACTIVE_LEASE = "active_lease"
    MUTATED_CUSTODY = "mutated_custody"
    AUTHORITY_VIOLATION = "authority_violation"
    NUMERICAL_REFUSAL = "numerical_refusal"
    VALUE_MISMATCH = "value_mismatch"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"


class V1LiteralJTwoPassDFixedProjectionRefusal(RuntimeError):
    """Typed fail-closed refusal for the D-fixed literal-J projection seam."""

    def __init__(
        self,
        code: V1LiteralJTwoPassDFixedProjectionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LiteralJTwoPassDFixedProjectionRefusalCode:
            raise TypeError("literal-J two-pass refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJTwoPassDFixedProjectionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJTwoPassDFixedProjectionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _identity_axis(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an ordered string axis")
    axis = tuple(_identity(value, name=f"{name} entry") for value in values)
    if len(axis) != len(set(axis)):
        raise ValueError(f"{name} must contain unique identities")
    return axis


def _string_axis(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an ordered string axis")
    return tuple(_identity(value, name=f"{name} entry") for value in values)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _fraction_pair(value: Sequence[int], *, name: str) -> tuple[int, int]:
    pair = tuple(value)
    if (
        len(pair) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in pair)
        or pair[1] <= 0
        or pair[0] < 0
    ):
        raise ValueError(f"{name} must be a nonnegative rational pair")
    fraction = Fraction(pair[0], pair[1])
    return (fraction.numerator, fraction.denominator)


def _fraction_binary64_enclosure(value: Fraction) -> tuple[float, float]:
    center = float(value)
    if not math.isfinite(center):
        _refuse(
            V1LiteralJTwoPassDFixedProjectionRefusalCode.NUMERICAL_REFUSAL,
            "rational weight overflows binary64",
        )
    lower = math.nextafter(center, -math.inf)
    upper = math.nextafter(center, math.inf)
    exact = Fraction(value.numerator, value.denominator)
    if not (Fraction.from_float(lower) <= exact <= Fraction.from_float(upper)):
        _refuse(
            V1LiteralJTwoPassDFixedProjectionRefusalCode.NUMERICAL_REFUSAL,
            "binary64 rational weight enclosure failed",
        )
    radius = max(abs(center - lower), abs(upper - center))
    return center, radius


def _gamma(operation_count: int) -> float:
    n = max(1, int(operation_count))
    value = n * _EPS
    if value >= 0.5:
        _refuse(
            V1LiteralJTwoPassDFixedProjectionRefusalCode.NUMERICAL_REFUSAL,
            "requested dot product is too long for a finite gamma enclosure",
        )
    return value / (1.0 - value)


def _up_tensor(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {"schema": "ordered_axis_v1", "role": role, "values": tuple(values)},
        prefix=_PREFIX + "axis-",
    )


def _tensor_scalar_fingerprint(value: Any, *, role: str) -> str:
    tensor = value.detach()
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        summary = {
            "count": 0,
            "sum": 0.0,
            "l1": 0.0,
            "l2": 0.0,
            "min": 0.0,
            "max": 0.0,
            "max_abs": 0.0,
        }
    else:
        abs_flat = torch.abs(flat)
        summary = {
            "count": int(flat.numel()),
            "sum": float(torch.sum(flat).detach().cpu().item()),
            "l1": float(torch.sum(abs_flat).detach().cpu().item()),
            "l2": float(torch.sqrt(torch.sum(flat * flat)).detach().cpu().item()),
            "min": float(torch.min(flat).detach().cpu().item()),
            "max": float(torch.max(flat).detach().cpu().item()),
            "max_abs": float(torch.max(abs_flat).detach().cpu().item()),
        }
    return content_hash(
        {
            "schema": "route_scale_tensor_scalar_fingerprint_v1",
            "role": role,
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "shape": tuple(int(item) for item in tensor.shape),
            "summary": summary,
        },
        prefix=_PREFIX + "tensor-fingerprint-",
    )


def _tensor_exact_value_digest(value: Any, *, role: str) -> str:
    """Private CPU-reference digest of exact tensor bytes.

    Production CUDA callers must provide source-owned immutable shard/window
    digests instead; this helper is intentionally used only in the small-domain
    CPU reference path where route-scale device-to-host movement is allowed.
    """

    tensor = value.detach()
    array = tensor.cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(b"literal-j-two-pass-private-cpu-tensor-value-v1")
    digest.update(role.encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(int(item) for item in array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return _PREFIX + "tensor-value-" + digest.hexdigest()


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    return (
        str(value.device),
        str(value.dtype),
        tuple(int(item) for item in value.shape),
        bool(value.is_contiguous()),
        int(value.data_ptr()) if hasattr(value, "data_ptr") else None,
        int(getattr(value, "_version", 0)),
    )


def _require_torch() -> Any:
    if torch is None:
        _refuse(
            V1LiteralJTwoPassDFixedProjectionRefusalCode.DEVICE_REQUIRED,
            "PyTorch is required for literal-J two-pass projection",
        )
    return torch


def _require_tensor(
    value: Any,
    *,
    name: str,
    device: Any | None = None,
    shape: tuple[int, ...] | None = None,
) -> Any:
    _require_torch()
    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.float64
        or not value.is_contiguous()
    ):
        _refuse(
            V1LiteralJTwoPassDFixedProjectionRefusalCode.INVALID_TYPE,
            f"{name} must be a contiguous float64 tensor",
        )
    if device is not None and value.device != device:
        _refuse(
            V1LiteralJTwoPassDFixedProjectionRefusalCode.STALE_IDENTITY,
            f"{name} is on a different device",
        )
    if shape is not None and tuple(int(item) for item in value.shape) != shape:
        _refuse(
            V1LiteralJTwoPassDFixedProjectionRefusalCode.INVALID_ROW,
            f"{name} shape does not match the registered axis",
        )
    return value


@dataclass(frozen=True, slots=True)
class V1LiteralJTwoPassDFixedProjectionDesign:
    route_tile_size: int = 131_072
    production_cuda_required: bool = True
    allow_private_cpu_reference: bool = False
    cpu_reference_route_limit: int = 65_536
    hbm_ceiling_fraction: float = 0.85
    schema: str = LITERAL_J_TWO_PASS_D_FIXED_DESIGN_SCHEMA
    algorithm: str = LITERAL_J_TWO_PASS_D_FIXED_ALGORITHM

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_tile_size",
            _positive_int(self.route_tile_size, name="route_tile_size"),
        )
        object.__setattr__(
            self,
            "cpu_reference_route_limit",
            _positive_int(
                self.cpu_reference_route_limit, name="cpu_reference_route_limit"
            ),
        )
        flags = (self.production_cuda_required, self.allow_private_cpu_reference)
        if any(type(item) is not bool for item in flags):
            raise TypeError("two-pass projection design flags must be exact booleans")
        if (
            not math.isfinite(float(self.hbm_ceiling_fraction))
            or float(self.hbm_ceiling_fraction) != 0.85
        ):
            raise ValueError("two-pass projection HBM ceiling is fixed at 85 percent")
        object.__setattr__(self, "hbm_ceiling_fraction", 0.85)
        if (
            self.schema != LITERAL_J_TWO_PASS_D_FIXED_DESIGN_SCHEMA
            or self.algorithm != LITERAL_J_TWO_PASS_D_FIXED_ALGORITHM
        ):
            raise ValueError("unsupported literal-J two-pass design")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True, slots=True)
class V1LiteralJTwoPassDFixedProjectionHBMForecast:
    registration_hash: str
    design_hash: str
    device: str
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    structural_context_count: int
    route_count: int
    direction_count: int
    route_tile_size: int
    max_packet_context_count: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    current_reserved_device_bytes: int
    hbm_ceiling_fraction: float
    hbm_ceiling_bytes: int
    resident_direction_bytes: int
    resident_context_projection_bytes: int
    resident_route_projection_bytes: int
    resident_scalar_bytes: int
    max_input_tile_bytes: int
    max_temporary_tile_bytes: int
    estimated_incremental_device_bytes: int
    projected_occupied_device_bytes: int
    admitted_under_ceiling: bool
    schema: str = LITERAL_J_TWO_PASS_D_FIXED_HBM_FORECAST_SCHEMA
    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _admission_memory: tuple[int, int] | None) -> None:
        available, allocated = _admission_memory or (self.device_free_memory_bytes, self.current_reserved_device_bytes)
        for name in ("registration_hash", "design_hash", "device"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "structural_context_count",
            "route_count",
            "direction_count",
            "route_tile_size",
            "max_packet_context_count",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        for name in (
            "device_total_memory_bytes",
            "device_free_memory_bytes",
            "current_reserved_device_bytes",
            "hbm_ceiling_bytes",
            "resident_direction_bytes",
            "resident_context_projection_bytes",
            "resident_route_projection_bytes",
            "resident_scalar_bytes",
            "max_input_tile_bytes",
            "max_temporary_tile_bytes",
            "estimated_incremental_device_bytes",
            "projected_occupied_device_bytes",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name=name)
            )
        flags = (
            self.production_cuda_execution,
            self.private_cpu_reference_execution,
            self.admitted_under_ceiling,
        )
        if any(type(item) is not bool for item in flags):
            raise TypeError("HBM forecast flags must be exact booleans")
        if self.production_cuda_execution == self.private_cpu_reference_execution:
            raise ValueError("HBM forecast must identify exactly one execution mode")
        if (
            not math.isfinite(float(self.hbm_ceiling_fraction))
            or float(self.hbm_ceiling_fraction) != 0.85
        ):
            raise ValueError("literal-J HBM forecast ceiling is fixed at 85 percent")
        object.__setattr__(self, "hbm_ceiling_fraction", 0.85)
        expected_incremental = sum(
            (
                self.resident_direction_bytes,
                self.resident_context_projection_bytes,
                self.resident_route_projection_bytes,
                self.resident_scalar_bytes,
                self.max_input_tile_bytes,
                self.max_temporary_tile_bytes,
            )
        )
        expected_ceiling = math.floor(self.device_total_memory_bytes * 0.85)
        expected_projected = self.current_reserved_device_bytes + expected_incremental
        expected_admitted = (
            allocated + expected_incremental <= expected_ceiling
            and expected_incremental <= available
        )
        if (
            self.device_free_memory_bytes > self.device_total_memory_bytes
            or self.current_reserved_device_bytes > self.device_total_memory_bytes
            or self.hbm_ceiling_bytes != expected_ceiling
            or self.estimated_incremental_device_bytes != expected_incremental
            or self.projected_occupied_device_bytes != expected_projected
            or self.admitted_under_ceiling != expected_admitted
        ):
            raise ValueError("literal-J HBM forecast arithmetic is inconsistent")
        if self.schema != LITERAL_J_TWO_PASS_D_FIXED_HBM_FORECAST_SCHEMA:
            raise ValueError("literal-J HBM forecast schema changed")

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "hbm-forecast-")


@dataclass(frozen=True, slots=True)
class V1LiteralJTwoPassDFixedProjectionPacketDescriptor:
    packet_index: int
    segment_index: int
    segment_role: str
    context_start: int
    context_count: int
    physical_source_registration_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    literal_j_stream_source_hash: str
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    measurement_ids: tuple[str, ...]
    capture_ids: tuple[str, ...]
    source_packet_hashes: tuple[str, ...]
    physical_context_weight_fraction: tuple[int, int]
    structural_context_weight_fraction: tuple[int, int]
    schema: str = LITERAL_J_TWO_PASS_D_FIXED_PACKET_SCHEMA
    _descriptor_hash: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "segment_role",
            "physical_source_registration_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "witness_stream_source_hash",
            "same_capture_vjp_source_hash",
            "literal_j_stream_source_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "packet_index",
            _nonnegative_int(self.packet_index, name="packet_index"),
        )
        object.__setattr__(
            self,
            "segment_index",
            _nonnegative_int(self.segment_index, name="segment_index"),
        )
        object.__setattr__(
            self,
            "context_start",
            _nonnegative_int(self.context_start, name="context_start"),
        )
        object.__setattr__(
            self,
            "context_count",
            _positive_int(self.context_count, name="context_count"),
        )
        axes = {}
        for name in (
            "context_ids",
            "draw_ids",
            "measurement_ids",
            "capture_ids",
            "source_packet_hashes",
        ):
            axis = _identity_axis(getattr(self, name), name=name)
            object.__setattr__(self, name, axis)
            axes[name] = axis
        if any(len(axis) != self.context_count for axis in axes.values()):
            raise ValueError("packet axes must all match context_count")
        object.__setattr__(
            self,
            "physical_context_weight_fraction",
            _fraction_pair(
                self.physical_context_weight_fraction,
                name="physical_context_weight_fraction",
            ),
        )
        object.__setattr__(
            self,
            "structural_context_weight_fraction",
            _fraction_pair(
                self.structural_context_weight_fraction,
                name="structural_context_weight_fraction",
            ),
        )
        if self.schema != LITERAL_J_TWO_PASS_D_FIXED_PACKET_SCHEMA:
            raise ValueError("literal-J two-pass packet schema changed")
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "packet-")
        if self._descriptor_hash is not None and self._descriptor_hash != observed:
            raise ValueError("literal-J two-pass packet descriptor was mutated")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("literal-J two-pass packet descriptor hash differs")
        object.__setattr__(self, "_descriptor_hash", observed)
        return observed

    @property
    def descriptor_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1LiteralJTwoPassDFixedProjectionRegistration:
    registration_id: str
    replay_registration_receipt_hash: str
    numeric_source_receipt_hash: str
    block_support_source_receipt_hash: str
    full_population_compact_source_receipt_hash: str
    source_replay_axis_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    final_planned_dw2_prefix_hash: str
    structural_source_registration_hash: str
    structural_context_axis_hash: str
    structural_draw_axis_hash: str
    structural_measurement_axis_hash: str
    structural_capture_axis_hash: str
    structural_source_packet_axis_hash: str
    composite_witness_stream_source_hash: str
    composite_same_capture_vjp_source_hash: str
    composite_literal_j_stream_source_hash: str
    same_capture_and_vjp_execution_proof_hash: str
    direction_ids: tuple[str, ...]
    direction_source_hash: str
    direction_axis_hash: str
    segment_roles: tuple[str, ...]
    segment_context_counts: tuple[int, ...]
    segment_uniform_weight_fractions: tuple[tuple[int, int], ...]
    structural_context_count: int
    structural_uniform_weight_fraction: tuple[int, int]
    context_weight_binary64_centers: tuple[float, ...]
    context_weight_binary64_radii: tuple[float, ...]
    packets: tuple[V1LiteralJTwoPassDFixedProjectionPacketDescriptor, ...]
    registered_before_literal_j_outcomes: bool = True
    context_route_bank_retained: bool = False
    route_scale_rows_d2h: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = LITERAL_J_TWO_PASS_D_FIXED_REGISTRATION_SCHEMA
    authority_scope: str = LITERAL_J_TWO_PASS_D_FIXED_AUTHORITY
    _registration_hash: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "registration_id",
            "replay_registration_receipt_hash",
            "numeric_source_receipt_hash",
            "block_support_source_receipt_hash",
            "full_population_compact_source_receipt_hash",
            "source_replay_axis_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "final_planned_dw2_prefix_hash",
            "structural_source_registration_hash",
            "structural_context_axis_hash",
            "structural_draw_axis_hash",
            "structural_measurement_axis_hash",
            "structural_capture_axis_hash",
            "structural_source_packet_axis_hash",
            "composite_witness_stream_source_hash",
            "composite_same_capture_vjp_source_hash",
            "composite_literal_j_stream_source_hash",
            "same_capture_and_vjp_execution_proof_hash",
            "direction_source_hash",
            "direction_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal_generation"),
        )
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route_count")
        )
        object.__setattr__(
            self,
            "structural_context_count",
            _positive_int(
                self.structural_context_count, name="structural_context_count"
            ),
        )
        object.__setattr__(
            self,
            "direction_ids",
            _identity_axis(self.direction_ids, name="direction_ids"),
        )
        object.__setattr__(
            self,
            "segment_roles",
            _identity_axis(self.segment_roles, name="segment_roles"),
        )
        counts = tuple(
            _positive_int(item, name="segment_context_count")
            for item in self.segment_context_counts
        )
        object.__setattr__(self, "segment_context_counts", counts)
        fractions = tuple(
            _fraction_pair(item, name="segment_uniform_weight_fraction")
            for item in self.segment_uniform_weight_fractions
        )
        object.__setattr__(self, "segment_uniform_weight_fractions", fractions)
        structural_fraction = _fraction_pair(
            self.structural_uniform_weight_fraction,
            name="structural_uniform_weight_fraction",
        )
        object.__setattr__(
            self, "structural_uniform_weight_fraction", structural_fraction
        )
        if (
            len(counts) != len(self.segment_roles)
            or len(counts) != len(fractions)
            or sum(counts) != self.structural_context_count
        ):
            raise ValueError(
                "D_fixed segment roles, weights, and structural count do not close"
            )
        centers = tuple(float(item) for item in self.context_weight_binary64_centers)
        radii = tuple(float(item) for item in self.context_weight_binary64_radii)
        if (
            len(centers) != self.structural_context_count
            or len(radii) != self.structural_context_count
        ):
            raise ValueError(
                "context weight enclosures must cover the structural D_fixed axis"
            )
        if any(not math.isfinite(item) for item in centers + radii) or any(
            item < 0.0 for item in radii
        ):
            raise ValueError("context weight enclosures must be finite outward radii")
        object.__setattr__(self, "context_weight_binary64_centers", centers)
        object.__setattr__(self, "context_weight_binary64_radii", radii)
        packets = tuple(self.packets)
        if not packets or any(
            type(item) is not V1LiteralJTwoPassDFixedProjectionPacketDescriptor
            for item in packets
        ):
            raise TypeError(
                "literal-J two-pass registration requires exact packet descriptors"
            )
        for packet in packets:
            packet.assert_integrity()
        if tuple(packet.packet_index for packet in packets) != tuple(
            range(len(packets))
        ):
            raise ValueError("literal-J two-pass packets must form one ordered axis")
        if tuple(packet.segment_role for packet in packets) != self.segment_roles:
            raise ValueError(
                "literal-J two-pass packet roles must replay segment roles"
            )
        if tuple(packet.context_count for packet in packets) != counts:
            raise ValueError(
                "literal-J two-pass packet counts must replay segment counts"
            )
        if any(
            packet.context_start + packet.context_count > self.structural_context_count
            for packet in packets
        ):
            raise ValueError(
                "literal-J two-pass packet exceeds structural context axis"
            )
        object.__setattr__(self, "packets", packets)
        expected_direction_axis = content_hash(
            {
                "schema": "literal_j_two_pass_direction_axis_v1",
                "route_axis_hash": self.route_axis_hash,
                "route_id_axis_hash": self.route_id_axis_hash,
                "route_count": self.route_count,
                "direction_ids": self.direction_ids,
                "direction_source_hash": self.direction_source_hash,
            },
            prefix=_PREFIX + "direction-axis-",
        )
        if self.direction_axis_hash != expected_direction_axis:
            raise ValueError("literal-J two-pass direction axis does not replay")
        flags = (
            self.registered_before_literal_j_outcomes,
            self.context_route_bank_retained,
            self.route_scale_rows_d2h,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(item) is not bool for item in flags) or flags != (
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            raise PermissionError("literal-J two-pass registration crossed authority")
        if (
            self.schema != LITERAL_J_TWO_PASS_D_FIXED_REGISTRATION_SCHEMA
            or self.authority_scope != LITERAL_J_TWO_PASS_D_FIXED_AUTHORITY
        ):
            raise ValueError(
                "literal-J two-pass registration schema or authority changed"
            )
        self.assert_integrity()

    @property
    def packet_descriptor_hashes(self) -> tuple[str, ...]:
        return tuple(packet.descriptor_hash for packet in self.packets)

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(
            context for packet in self.packets for context in packet.context_ids
        )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "registration-")
        if self._registration_hash is not None and self._registration_hash != observed:
            raise ValueError("literal-J two-pass registration was mutated")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("literal-J two-pass registration hash differs")
        object.__setattr__(self, "_registration_hash", observed)
        return observed

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()


def register_v1_literal_j_two_pass_d_fixed_projection(
    *,
    replay_plan: V1SampledBlockSupportReplayRegistrationPlan,
    numeric_source_receipt: V1SampledBlockSupportNumericSourceReceipt,
    route_count: int,
    direction_ids: Sequence[str],
    direction_source_hash: str,
    same_capture_and_vjp_execution_proof_hash: str,
) -> V1LiteralJTwoPassDFixedProjectionRegistration:
    """Create the exact pre-outcome registration for two-pass D-fixed replay."""

    from .modern_v1_sampled_block_support_numeric_source import (
        V1SampledBlockSupportNumericSourceReceipt,
    )
    from .modern_v1_sampled_block_support_replay_registration import (
        V1SampledBlockSupportReplayRegistrationPlan,
    )

    if type(replay_plan) is not V1SampledBlockSupportReplayRegistrationPlan:
        _refuse(
            V1LiteralJTwoPassDFixedProjectionRefusalCode.INVALID_TYPE,
            "replay_plan must have exact sampled block-support replay-plan type",
        )
    if type(numeric_source_receipt) is not V1SampledBlockSupportNumericSourceReceipt:
        _refuse(
            V1LiteralJTwoPassDFixedProjectionRefusalCode.INVALID_TYPE,
            "numeric_source_receipt must have exact sampled block-support numeric-source type",
        )
    plan_receipt = replay_plan.receipt
    plan_receipt.assert_integrity()
    numeric_source_receipt.assert_integrity()
    if (
        numeric_source_receipt.replay_registration_receipt_hash
        != plan_receipt.receipt_hash
        or numeric_source_receipt.structural_source_registration_hash
        != plan_receipt.structural_registration_hash
        or numeric_source_receipt.physical_source_registration_hashes
        != plan_receipt.physical_registration_hashes
        or numeric_source_receipt.model_identity_hash
        != plan_receipt.model_identity_hash
        or numeric_source_receipt.chart_identity_hash
        != plan_receipt.chart_identity_hash
        or numeric_source_receipt.current_mask_hash != plan_receipt.current_mask_hash
        or numeric_source_receipt.proposal_generation
        != plan_receipt.proposal_generation
        or numeric_source_receipt.route_axis_hash != plan_receipt.route_axis_hash
        or numeric_source_receipt.route_id_axis_hash != plan_receipt.route_id_axis_hash
        or numeric_source_receipt.context_count != plan_receipt.structural_context_count
    ):
        _refuse(
            V1LiteralJTwoPassDFixedProjectionRefusalCode.STALE_IDENTITY,
            "numeric-source receipt does not replay the sampled block-support plan",
        )
    route_total = _positive_int(route_count, name="route_count")
    ids = _identity_axis(direction_ids, name="direction_ids")
    direction_source = _identity(direction_source_hash, name="direction_source_hash")
    direction_axis_hash = content_hash(
        {
            "schema": "literal_j_two_pass_direction_axis_v1",
            "route_axis_hash": plan_receipt.route_axis_hash,
            "route_id_axis_hash": plan_receipt.route_id_axis_hash,
            "route_count": route_total,
            "direction_ids": ids,
            "direction_source_hash": direction_source,
        },
        prefix=_PREFIX + "direction-axis-",
    )
    structural_fraction = _fraction_pair(
        plan_receipt.structural_uniform_weight_fraction,
        name="structural_uniform_weight_fraction",
    )
    structural_weight = Fraction(*structural_fraction)
    center, radius = _fraction_binary64_enclosure(structural_weight)
    if (
        replay_plan.structural_registration.registration_hash
        != plan_receipt.structural_registration_hash
    ):
        _refuse(
            V1LiteralJTwoPassDFixedProjectionRefusalCode.STALE_IDENTITY,
            "structural replay registration identity does not match receipt",
        )
    structural_weights = tuple(
        float(item) for item in replay_plan.structural_registration.context_weights
    )
    if len(structural_weights) != plan_receipt.structural_context_count:
        _refuse(
            V1LiteralJTwoPassDFixedProjectionRefusalCode.STALE_IDENTITY,
            "structural replay registration does not expose one D_fixed weight per context",
        )
    # The replay registration stores the binary64 center, not an arbitrary
    # member of the later analytic enclosure.  Accepting a neighbouring float
    # here would silently change the executed structural measure even though
    # the durable receipt continued to claim the canonical rational 1/d.
    if any(not math.isfinite(item) or item != center for item in structural_weights):
        _refuse(
            V1LiteralJTwoPassDFixedProjectionRefusalCode.STALE_IDENTITY,
            "structural replay registration weights are not the canonical uniform 1/d measure",
        )
    centers = (center,) * plan_receipt.structural_context_count
    radii = (radius,) * plan_receipt.structural_context_count
    packets: list[V1LiteralJTwoPassDFixedProjectionPacketDescriptor] = []
    context_start = 0
    for packet_index, source in enumerate(replay_plan.physical_segment_registrations):
        source_role = plan_receipt.segment_roles[packet_index]
        source_count = plan_receipt.segment_context_counts[packet_index]
        physical_fraction = _fraction_pair(
            plan_receipt.segment_uniform_weight_fractions[packet_index],
            name="segment_uniform_weight_fraction",
        )
        if (
            source.registration_hash
            != plan_receipt.physical_registration_hashes[packet_index]
            or source.context_count != source_count
            or source.route_axis_hash != plan_receipt.route_axis_hash
            or source.route_id_axis_hash != plan_receipt.route_id_axis_hash
            or source.registered_d_prefix_hash
            != plan_receipt.template_registered_d_prefix_hashes[packet_index]
            or source.registered_d_w2_prefix_hash
            != plan_receipt.template_registered_dw2_prefix_hashes[packet_index]
            or source.witness_stream_source_hash
            != plan_receipt.witness_stream_source_hashes[packet_index]
            or source.same_capture_vjp_source_hash
            != plan_receipt.same_capture_vjp_source_hashes[packet_index]
            or source.literal_j_stream_source_hash
            != plan_receipt.literal_j_stream_source_hashes[packet_index]
        ):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.STALE_IDENTITY,
                "physical segment registration does not replay the block-support receipt",
            )
        packets.append(
            V1LiteralJTwoPassDFixedProjectionPacketDescriptor(
                packet_index=packet_index,
                segment_index=packet_index,
                segment_role=source_role,
                context_start=context_start,
                context_count=source_count,
                physical_source_registration_hash=source.registration_hash,
                registered_d_prefix_hash=source.registered_d_prefix_hash,
                registered_d_w2_prefix_hash=source.registered_d_w2_prefix_hash,
                witness_stream_source_hash=source.witness_stream_source_hash,
                same_capture_vjp_source_hash=source.same_capture_vjp_source_hash,
                literal_j_stream_source_hash=source.literal_j_stream_source_hash,
                context_ids=source.registered_context_ids,
                draw_ids=source.registered_draw_ids,
                measurement_ids=source.registered_measurement_ids,
                capture_ids=source.registered_capture_ids,
                source_packet_hashes=source.registered_source_packet_hashes,
                physical_context_weight_fraction=physical_fraction,
                structural_context_weight_fraction=structural_fraction,
            )
        )
        context_start += source_count
    registration_id = content_hash(
        {
            "schema": "literal_j_two_pass_D_fixed_registration_id_v1",
            "replay_registration_receipt_hash": plan_receipt.receipt_hash,
            "numeric_source_receipt_hash": numeric_source_receipt.receipt_hash,
            "route_count": route_total,
            "direction_axis_hash": direction_axis_hash,
            "same_capture_and_vjp_execution_proof_hash": same_capture_and_vjp_execution_proof_hash,
        },
        prefix=_PREFIX + "registration-id-",
    )
    source_replay_axis_hash = content_hash(
        {
            "schema": "literal_j_two_pass_D_fixed_source_replay_axis_v1",
            "replay_registration_receipt_hash": plan_receipt.receipt_hash,
            "numeric_source_receipt_hash": numeric_source_receipt.receipt_hash,
            "packet_descriptor_hashes": tuple(
                packet.descriptor_hash for packet in packets
            ),
            "structural_context_axis_hash": (
                numeric_source_receipt.structural_context_axis_hash
            ),
            "structural_source_packet_axis_hash": (
                numeric_source_receipt.structural_source_packet_axis_hash
            ),
        },
        prefix=_PREFIX + "source-replay-axis-",
    )
    return V1LiteralJTwoPassDFixedProjectionRegistration(
        registration_id=registration_id,
        replay_registration_receipt_hash=plan_receipt.receipt_hash,
        numeric_source_receipt_hash=numeric_source_receipt.receipt_hash,
        block_support_source_receipt_hash=numeric_source_receipt.block_support_source_receipt_hash,
        full_population_compact_source_receipt_hash=plan_receipt.full_population_compact_source_receipt_hash,
        source_replay_axis_hash=source_replay_axis_hash,
        model_identity_hash=plan_receipt.model_identity_hash,
        chart_identity_hash=plan_receipt.chart_identity_hash,
        current_mask_hash=plan_receipt.current_mask_hash,
        proposal_generation=plan_receipt.proposal_generation,
        route_axis_hash=plan_receipt.route_axis_hash,
        route_id_axis_hash=plan_receipt.route_id_axis_hash,
        route_count=route_total,
        registered_d_prefix_hash=plan_receipt.conditioned_d1_registered_prefix_hash,
        registered_d_w2_prefix_hash=plan_receipt.residual_registered_dw2_prefix_hash,
        final_planned_dw2_prefix_hash=plan_receipt.final_planned_dw2_prefix_hash,
        structural_source_registration_hash=plan_receipt.structural_registration_hash,
        structural_context_axis_hash=numeric_source_receipt.structural_context_axis_hash,
        structural_draw_axis_hash=numeric_source_receipt.structural_draw_axis_hash,
        structural_measurement_axis_hash=(
            numeric_source_receipt.structural_measurement_axis_hash
        ),
        structural_capture_axis_hash=numeric_source_receipt.structural_capture_axis_hash,
        structural_source_packet_axis_hash=(
            numeric_source_receipt.structural_source_packet_axis_hash
        ),
        composite_witness_stream_source_hash=plan_receipt.composite_witness_stream_source_hash,
        composite_same_capture_vjp_source_hash=plan_receipt.composite_same_capture_vjp_source_hash,
        composite_literal_j_stream_source_hash=plan_receipt.composite_literal_j_stream_source_hash,
        same_capture_and_vjp_execution_proof_hash=_identity(
            same_capture_and_vjp_execution_proof_hash,
            name="same_capture_and_vjp_execution_proof_hash",
        ),
        direction_ids=ids,
        direction_source_hash=direction_source,
        direction_axis_hash=direction_axis_hash,
        segment_roles=plan_receipt.segment_roles,
        segment_context_counts=plan_receipt.segment_context_counts,
        segment_uniform_weight_fractions=plan_receipt.segment_uniform_weight_fractions,
        structural_context_count=plan_receipt.structural_context_count,
        structural_uniform_weight_fraction=structural_fraction,
        context_weight_binary64_centers=centers,
        context_weight_binary64_radii=radii,
        packets=tuple(packets),
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJTwoPassDFixedProjectionReceipt:
    registration_hash: str
    design_hash: str
    telemetry_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    direction_ids: tuple[str, ...]
    direction_axis_hash: str
    source_replay_axis_hash: str
    replay_registration_receipt_hash: str
    numeric_source_receipt_hash: str
    block_support_source_receipt_hash: str
    same_capture_and_vjp_execution_proof_hash: str
    literal_j_stream_source_hash: str
    context_count: int
    packet_count: int
    pass1_tile_count: int
    pass2_tile_count: int
    structural_uniform_weight_fraction: tuple[int, int]
    context_weight_center_fingerprint: str
    context_weight_radius_fingerprint: str
    p_center_fingerprint: str
    p_radius_fingerprint: str
    mjz_center_fingerprint: str
    mjz_radius_fingerprint: str
    q_center_fingerprint: str
    q_radius_fingerprint: str
    literal_j_tile_value_ledger_hash: str
    hbm_forecast_hash: str
    context_route_bank_retained: bool = False
    route_scale_rows_d2h: bool = False
    array_free_receipt: bool = True
    live_device_handle_required_for_values: bool = True
    transient_workspaces_released_after_finalize: bool = True
    projection_authority: bool = True
    d_decision_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_TWO_PASS_D_FIXED_ALGORITHM
    authority_scope: str = LITERAL_J_TWO_PASS_D_FIXED_AUTHORITY
    schema: str = LITERAL_J_TWO_PASS_D_FIXED_PROJECTION_SCHEMA
    _receipt_hash: str | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "design_hash",
            "telemetry_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "direction_axis_hash",
            "source_replay_axis_hash",
            "replay_registration_receipt_hash",
            "numeric_source_receipt_hash",
            "block_support_source_receipt_hash",
            "same_capture_and_vjp_execution_proof_hash",
            "literal_j_stream_source_hash",
            "context_weight_center_fingerprint",
            "context_weight_radius_fingerprint",
            "p_center_fingerprint",
            "p_radius_fingerprint",
            "mjz_center_fingerprint",
            "mjz_radius_fingerprint",
            "q_center_fingerprint",
            "q_radius_fingerprint",
            "literal_j_tile_value_ledger_hash",
            "hbm_forecast_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal_generation"),
        )
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route_count")
        )
        object.__setattr__(
            self,
            "context_count",
            _positive_int(self.context_count, name="context_count"),
        )
        object.__setattr__(
            self, "packet_count", _positive_int(self.packet_count, name="packet_count")
        )
        object.__setattr__(
            self,
            "pass1_tile_count",
            _positive_int(self.pass1_tile_count, name="pass1_tile_count"),
        )
        object.__setattr__(
            self,
            "pass2_tile_count",
            _positive_int(self.pass2_tile_count, name="pass2_tile_count"),
        )
        object.__setattr__(
            self,
            "direction_ids",
            _identity_axis(self.direction_ids, name="direction_ids"),
        )
        object.__setattr__(
            self,
            "structural_uniform_weight_fraction",
            _fraction_pair(
                self.structural_uniform_weight_fraction,
                name="structural_uniform_weight_fraction",
            ),
        )
        required_true = (
            self.array_free_receipt,
            self.live_device_handle_required_for_values,
            self.transient_workspaces_released_after_finalize,
            self.projection_authority,
        )
        required_false = (
            self.context_route_bank_retained,
            self.route_scale_rows_d2h,
            self.d_decision_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(item) is not bool for item in required_true + required_false)
            or any(item is not True for item in required_true)
            or any(item is not False for item in required_false)
            or self.algorithm != LITERAL_J_TWO_PASS_D_FIXED_ALGORITHM
            or self.authority_scope != LITERAL_J_TWO_PASS_D_FIXED_AUTHORITY
            or self.schema != LITERAL_J_TWO_PASS_D_FIXED_PROJECTION_SCHEMA
        ):
            raise PermissionError("literal-J two-pass receipt crossed authority")
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "receipt-")
        if self._receipt_hash is not None and self._receipt_hash != observed:
            raise ValueError("literal-J two-pass receipt was mutated")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("literal-J two-pass receipt hash differs")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1LiteralJTwoPassDFixedProjectionTelemetry:
    registration_hash: str
    device: str
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    context_count: int
    route_count: int
    direction_count: int
    packet_count: int
    pass1_tile_count: int
    pass2_tile_count: int
    hbm_forecast_hash: str
    literal_j_tile_value_ledger_hash: str
    cuda_source_tile_digests_required: bool
    context_route_bank_retained: bool = False
    route_scale_rows_d2h: bool = False
    route_scale_output_handle_retained: bool = True
    deterministic_fp64: bool = True
    schema: str = LITERAL_J_TWO_PASS_D_FIXED_TELEMETRY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "device",
            "hbm_forecast_hash",
            "literal_j_tile_value_ledger_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "context_count",
            "route_count",
            "direction_count",
            "packet_count",
            "pass1_tile_count",
            "pass2_tile_count",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        flags = (
            self.production_cuda_execution,
            self.private_cpu_reference_execution,
            self.cuda_source_tile_digests_required,
            self.context_route_bank_retained,
            self.route_scale_rows_d2h,
            self.route_scale_output_handle_retained,
            self.deterministic_fp64,
        )
        if any(type(item) is not bool for item in flags):
            raise TypeError("telemetry flags must be exact booleans")
        if self.production_cuda_execution == self.private_cpu_reference_execution:
            raise ValueError("telemetry must identify exactly one execution mode")
        if (
            self.context_route_bank_retained
            or self.route_scale_rows_d2h
            or not self.route_scale_output_handle_retained
            or not self.deterministic_fp64
            or self.cuda_source_tile_digests_required != self.production_cuda_execution
        ):
            raise PermissionError("telemetry violates two-pass D-fixed custody")
        if self.schema != LITERAL_J_TWO_PASS_D_FIXED_TELEMETRY_SCHEMA:
            raise ValueError("telemetry schema changed")

    @property
    def telemetry_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "telemetry-")


class V1LiteralJTwoPassDFixedProjectionLease:
    __slots__ = (
        "_owner",
        "_closed",
        "mjz_center",
        "mjz_radius",
        "q_center",
        "q_radius",
    )

    def __init__(
        self,
        owner: "V1LiteralJTwoPassDFixedProjectionDeviceHandle",
        tensors: Mapping[str, Any],
    ) -> None:
        self._owner = owner
        self._closed = False
        self.mjz_center = tensors["mjz_center"]
        self.mjz_radius = tensors["mjz_radius"]
        self.q_center = tensors["q_center"]
        self.q_radius = tensors["q_radius"]

    def __enter__(self) -> "V1LiteralJTwoPassDFixedProjectionLease":
        self._owner.assert_live()
        return self

    def close(self) -> None:
        if not self._closed:
            try:
                self._owner._close_lease()
            finally:
                self._closed = True

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("literal-J two-pass device leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("literal-J two-pass device leases are nonserializable")


class V1LiteralJTwoPassDFixedProjectionDeviceHandle:
    """Mutation-checked process-local custody of route-scale projection output."""

    __slots__ = ("receipt", "_tensors", "_guards", "_invariants", "_leases", "_evicted")

    def __init__(
        self,
        *,
        receipt: V1LiteralJTwoPassDFixedProjectionReceipt,
        tensors: Mapping[str, Any],
    ) -> None:
        if type(receipt) is not V1LiteralJTwoPassDFixedProjectionReceipt:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.INVALID_TYPE,
                "device handle requires exact projection receipt",
            )
        required = ("mjz_center", "mjz_radius", "q_center", "q_radius")
        if set(tensors) != set(required):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.INVALID_TYPE,
                "device handle tensor roles are incomplete",
            )
        self.receipt = receipt
        self._tensors = {role: tensors[role] for role in required}
        self._guards = {role: tensors[role].detach().clone() for role in required}
        self._invariants = {role: _tensor_invariant(tensors[role]) for role in required}
        self._leases = 0
        self._evicted = False
        self.assert_live()

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.EVICTED,
                "literal-J two-pass projection handle was evicted",
            )
        self.receipt.assert_integrity()
        for role, tensor in self._tensors.items():
            if _tensor_invariant(tensor) != self._invariants[role] or not bool(
                torch.equal(tensor, self._guards[role])
            ):
                _refuse(
                    V1LiteralJTwoPassDFixedProjectionRefusalCode.MUTATED_CUSTODY,
                    "literal-J two-pass route-scale tensor mutated",
                )

    def borrow(self) -> V1LiteralJTwoPassDFixedProjectionLease:
        self.assert_live()
        self._leases += 1
        return V1LiteralJTwoPassDFixedProjectionLease(self, self._tensors)

    def _close_lease(self) -> None:
        if self._leases <= 0:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.ACTIVE_LEASE,
                "literal-J two-pass lease accounting underflow",
            )
        try:
            self.assert_live()
        finally:
            self._leases -= 1

    def evict(self) -> None:
        self.assert_live()
        if self._leases:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.ACTIVE_LEASE,
                "cannot evict literal-J two-pass output while leased",
            )
        self._tensors = {}
        self._guards = {}
        self._invariants = {}
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("literal-J two-pass device handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("literal-J two-pass device handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1LiteralJTwoPassDFixedProjectionExecution:
    receipt: V1LiteralJTwoPassDFixedProjectionReceipt
    telemetry: V1LiteralJTwoPassDFixedProjectionTelemetry
    device_handle: V1LiteralJTwoPassDFixedProjectionDeviceHandle = field(
        repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if (
            type(self.receipt) is not V1LiteralJTwoPassDFixedProjectionReceipt
            or type(self.telemetry) is not V1LiteralJTwoPassDFixedProjectionTelemetry
            or type(self.device_handle)
            is not V1LiteralJTwoPassDFixedProjectionDeviceHandle
        ):
            raise TypeError(
                "literal-J two-pass execution requires exact component types"
            )
        if (
            self.receipt.telemetry_hash != self.telemetry.telemetry_hash
            or self.device_handle.receipt.receipt_hash != self.receipt.receipt_hash
        ):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.STALE_IDENTITY,
                "literal-J two-pass execution identities do not close",
            )


class V1LiteralJTwoPassDFixedProjectionCUDAAccumulator:
    """Two-pass route-tile accumulator with no persistent context-route rows."""

    def __init__(
        self,
        *,
        registration: V1LiteralJTwoPassDFixedProjectionRegistration,
        directions: Any,
        direction_source_hash: str,
        design: V1LiteralJTwoPassDFixedProjectionDesign | None = None,
        device_total_memory_bytes: int | None = None,
        device_free_memory_bytes: int | None = None,
        current_reserved_device_bytes: int | None = None,
        structured_direction_custody: (
            V1LiteralJAffineShellDirectionDeviceHandle | None
        ) = None,
    ) -> None:
        _require_torch()
        if type(registration) is not V1LiteralJTwoPassDFixedProjectionRegistration:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.INVALID_TYPE,
                "accumulator requires exact two-pass registration",
            )
        registration.assert_integrity()
        self.registration = registration
        self.design = (
            V1LiteralJTwoPassDFixedProjectionDesign() if design is None else design
        )
        if type(self.design) is not V1LiteralJTwoPassDFixedProjectionDesign:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.INVALID_TYPE,
                "accumulator requires exact two-pass design",
            )
        if _identity(
            direction_source_hash, name="direction_source_hash"
        ) != registration.direction_source_hash:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.STALE_IDENTITY,
                "direction tensor source does not match the registered direction axis",
            )
        self._structured_direction_lease: (
            V1LiteralJAffineShellDirectionLease | None
        ) = None
        self._structured_outside = None
        self._structured_shell_indices: tuple[int, ...] = ()
        if structured_direction_custody is None:
            direction_tensor = _require_tensor(
                directions,
                name="directions",
                shape=(len(registration.direction_ids), registration.route_count),
            )
            self.device = direction_tensor.device
        else:
            if (
                type(structured_direction_custody)
                is not V1LiteralJAffineShellDirectionDeviceHandle
                or directions is not None
            ):
                _refuse(
                    V1LiteralJTwoPassDFixedProjectionRefusalCode.INVALID_TYPE,
                    "structured path requires one exact direction handle and no dense bank",
                )
            structured = structured_direction_custody.receipt
            if (
                structured.direction_source_hash != registration.direction_source_hash
                or structured.direction_ids != registration.direction_ids
                or structured.model_identity_hash != registration.model_identity_hash
                or structured.chart_identity_hash != registration.chart_identity_hash
                or structured.current_mask_hash != registration.current_mask_hash
                or structured.proposal_generation != registration.proposal_generation
                or structured.route_axis_hash != registration.route_axis_hash
                or structured.route_id_axis_hash != registration.route_id_axis_hash
                or structured.route_count != registration.route_count
                or structured.registered_d_prefix_hash
                != registration.registered_d_prefix_hash
                or structured.registered_d_w2_prefix_hash
                != registration.registered_d_w2_prefix_hash
                or structured.literal_j_stream_source_hash
                != registration.composite_literal_j_stream_source_hash
                or structured.same_capture_vjp_source_hash
                != registration.composite_same_capture_vjp_source_hash
                or structured.source_receipt_hash
                != registration.full_population_compact_source_receipt_hash
            ):
                _refuse(
                    V1LiteralJTwoPassDFixedProjectionRefusalCode.STALE_IDENTITY,
                    "structured direction lineage differs from the registered D-fixed chart",
                )
            with structured_direction_custody.borrow() as probe:
                outside, shell_indices = probe.structured_tensors()
                self.device = outside.device
                self._structured_shell_indices = tuple(
                    int(item) for item in shell_indices.detach().cpu().tolist()
                )
            direction_tensor = None
        if self.design.production_cuda_required:
            if self.device.type != "cuda":
                _refuse(
                    V1LiteralJTwoPassDFixedProjectionRefusalCode.CPU_PRODUCTION_FALLBACK,
                    "production two-pass D-fixed projection refuses CPU fallback",
                )
            warn_only = getattr(
                torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False
            )
            if (
                not bool(torch.are_deterministic_algorithms_enabled())
                or bool(warn_only())
                or os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}
            ):
                _refuse(
                    V1LiteralJTwoPassDFixedProjectionRefusalCode.DETERMINISM_DISABLED,
                    "production CUDA requires strict deterministic algorithms and CUBLAS workspace",
                )
        elif self.device.type == "cpu":
            if (
                not self.design.allow_private_cpu_reference
                or registration.route_count > self.design.cpu_reference_route_limit
            ):
                _refuse(
                    V1LiteralJTwoPassDFixedProjectionRefusalCode.CPU_PRODUCTION_FALLBACK,
                    "CPU execution is a small-domain private reference only",
                )
        else:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.INVALID_DESIGN,
                "non-production execution supports only explicit CPU reference",
            )
        if direction_tensor is not None and not bool(
            torch.isfinite(direction_tensor).all().item()
        ):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.INVALID_ROW,
                "directions contain nonfinite values",
            )
        d = len(registration.direction_ids)
        n = registration.structural_context_count
        r = registration.route_count
        if self.device.type == "cuda" and self.design.production_cuda_required:
            if (
                device_total_memory_bytes is not None
                or device_free_memory_bytes is not None
                or current_reserved_device_bytes is not None
            ):
                _refuse(
                    V1LiteralJTwoPassDFixedProjectionRefusalCode.INVALID_DESIGN,
                    "production CUDA memory admission must use live device HBM facts",
                )
            free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            reserved_bytes = int(torch.cuda.memory_reserved(self.device))
            available_bytes = cuda_reusable_memory_bytes(self.device, framework=torch, free_bytes=free_bytes)
            allocated_bytes = int(torch.cuda.memory_allocated(self.device))
        else:
            default_memory = 8 * 1024**3
            total_bytes = (
                default_memory
                if device_total_memory_bytes is None
                else _positive_int(
                    device_total_memory_bytes,
                    name="device_total_memory_bytes",
                )
            )
            free_bytes = (
                total_bytes
                if device_free_memory_bytes is None
                else _nonnegative_int(
                    device_free_memory_bytes,
                    name="device_free_memory_bytes",
                )
            )
            reserved_bytes = (
                0
                if current_reserved_device_bytes is None
                else _nonnegative_int(
                    current_reserved_device_bytes,
                    name="current_reserved_device_bytes",
                )
            )
        if not (self.device.type == "cuda" and self.design.production_cuda_required):
            available_bytes, allocated_bytes = int(free_bytes), int(reserved_bytes)
        max_packet_context_count = max(
            packet.context_count for packet in registration.packets
        )
        route_tile = self.design.route_tile_size
        # The structured path borrows one already-admitted outside vector and
        # B indices.  It allocates no dense (B+1)-by-route direction bank.
        resident_direction_bytes = (
            d * r * _F64_BYTES if direction_tensor is not None else 0
        )
        resident_context_projection_bytes = 3 * n * d * _F64_BYTES
        # Peak finalization holds M_J z center, radius, the proof-only absolute
        # arithmetic mass, and mutation guards for both exported route tensors:
        # five d-by-R owners.  Q analogously holds three working vectors plus
        # two exported guards.  Forecast the peak before proof workspaces are
        # released; omitting the guards would make the 85% ceiling unsound.
        resident_route_projection_bytes = 5 * d * r * _F64_BYTES
        resident_scalar_bytes = (2 * n + 5 * d) * _F64_BYTES
        max_input_tile_bytes = max_packet_context_count * route_tile * _F64_BYTES
        max_temporary_tile_bytes = (
            max_packet_context_count * d * _F64_BYTES * 6
            + d * route_tile * _F64_BYTES * 4
            + max_packet_context_count * route_tile * _F64_BYTES
        )
        estimated_incremental_bytes = (
            resident_direction_bytes
            + resident_context_projection_bytes
            + resident_route_projection_bytes
            + resident_scalar_bytes
            + max_input_tile_bytes
            + max_temporary_tile_bytes
        )
        hbm_ceiling_bytes = math.floor(total_bytes * self.design.hbm_ceiling_fraction)
        projected_occupied_bytes = reserved_bytes + estimated_incremental_bytes
        admitted = (
            allocated_bytes + estimated_incremental_bytes <= hbm_ceiling_bytes
            and estimated_incremental_bytes <= available_bytes
        )
        self._hbm_forecast = V1LiteralJTwoPassDFixedProjectionHBMForecast(
            registration_hash=registration.registration_hash,
            design_hash=self.design.design_hash,
            device=str(self.device),
            production_cuda_execution=self.device.type == "cuda"
            and self.design.production_cuda_required,
            private_cpu_reference_execution=self.device.type == "cpu",
            structural_context_count=n,
            route_count=r,
            direction_count=d,
            route_tile_size=route_tile,
            max_packet_context_count=max_packet_context_count,
            device_total_memory_bytes=total_bytes,
            device_free_memory_bytes=free_bytes,
            current_reserved_device_bytes=reserved_bytes,
            hbm_ceiling_fraction=self.design.hbm_ceiling_fraction,
            hbm_ceiling_bytes=hbm_ceiling_bytes,
            resident_direction_bytes=resident_direction_bytes,
            resident_context_projection_bytes=resident_context_projection_bytes,
            resident_route_projection_bytes=resident_route_projection_bytes,
            resident_scalar_bytes=resident_scalar_bytes,
            max_input_tile_bytes=max_input_tile_bytes,
            max_temporary_tile_bytes=max_temporary_tile_bytes,
            estimated_incremental_device_bytes=estimated_incremental_bytes,
            projected_occupied_device_bytes=projected_occupied_bytes,
            admitted_under_ceiling=admitted,
            _admission_memory=(available_bytes, allocated_bytes),
        )
        if not admitted:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.HBM_FORECAST_EXCEEDED,
                "literal-J two-pass projection exceeds the pre-outcome 85% HBM admission forecast",
            )
        self._directions = (
            direction_tensor.detach().clone(memory_format=torch.contiguous_format)
            if direction_tensor is not None
            else None
        )
        self._weights = torch.tensor(
            registration.context_weight_binary64_centers,
            dtype=torch.float64,
            device=self.device,
        )
        self._weight_radii = torch.tensor(
            registration.context_weight_binary64_radii,
            dtype=torch.float64,
            device=self.device,
        )
        self._p = torch.zeros((n, d), dtype=torch.float64, device=self.device)
        self._p_abs_mass = torch.zeros((n, d), dtype=torch.float64, device=self.device)
        self._p_radius = torch.zeros((n, d), dtype=torch.float64, device=self.device)
        self._mz_center = torch.zeros((d, r), dtype=torch.float64, device=self.device)
        self._mz_radius = torch.zeros((d, r), dtype=torch.float64, device=self.device)
        self._mz_abs_mass = torch.zeros((d, r), dtype=torch.float64, device=self.device)
        self._q_center = torch.zeros(d, dtype=torch.float64, device=self.device)
        self._q_radius = torch.zeros(d, dtype=torch.float64, device=self.device)
        self._q_abs_mass = torch.zeros(d, dtype=torch.float64, device=self.device)
        if structured_direction_custody is not None:
            lease = structured_direction_custody.borrow()
            outside, shell_indices = lease.structured_tensors()
            if tuple(
                int(item) for item in shell_indices.detach().cpu().tolist()
            ) != self._structured_shell_indices:
                lease.close()
                _refuse(
                    V1LiteralJTwoPassDFixedProjectionRefusalCode.MUTATED_CUSTODY,
                    "structured shell indices changed during accumulator construction",
                )
            self._structured_direction_lease = lease
            self._structured_outside = outside
        self._pass = 1
        self._pass1_packet_cursor = 0
        self._pass2_packet_cursor = 0
        self._active_packet: (
            V1LiteralJTwoPassDFixedProjectionPacketDescriptor | None
        ) = None
        self._active_cursor = 0
        self._active_tile_count = 0
        self._active_p_abs = None
        self._active_windows: list[tuple[int, int]] = []
        self._active_pass = 0
        self._pass1_packet_hashes: list[str] = []
        self._pass2_packet_hashes: list[str] = []
        self._pass1_window_hashes: list[str] = []
        self._pass2_window_hashes: list[str] = []
        self._pass1_tile_value_hashes: list[str] = []
        self._pass2_tile_value_hashes: list[str] = []
        self._pass1_tile_count = 0
        self._pass2_tile_count = 0
        self._rewind_hash: str | None = None
        self._finalized = False

    @property
    def hbm_forecast(self) -> V1LiteralJTwoPassDFixedProjectionHBMForecast:
        """Return the immutable pre-outcome memory-admission receipt."""

        return self._hbm_forecast

    def _assert_live(self) -> None:
        if self._finalized:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.FINALIZED,
                "literal-J two-pass accumulator already finalized",
            )

    def _expected_packet(
        self, pass_index: int
    ) -> V1LiteralJTwoPassDFixedProjectionPacketDescriptor:
        cursor = (
            self._pass1_packet_cursor if pass_index == 1 else self._pass2_packet_cursor
        )
        if cursor >= len(self.registration.packets):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.ORDER_MISMATCH,
                "literal-J two-pass stream emitted an extra packet",
            )
        return self.registration.packets[cursor]

    def _validate_packet_begin(
        self,
        *,
        pass_index: int,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        measurement_ids: Sequence[str],
        capture_ids: Sequence[str],
        source_packet_hashes: Sequence[str],
        model_identity_hash: str,
        chart_identity_hash: str,
        current_mask_hash: str,
        proposal_generation: int,
        route_axis_hash: str,
        route_id_axis_hash: str,
        same_capture_and_vjp_execution_proof_hash: str,
        same_capture_vjp_source_hash: str,
        literal_j_stream_source_hash: str,
    ) -> V1LiteralJTwoPassDFixedProjectionPacketDescriptor:
        self._assert_live()
        if self._active_packet is not None:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.PARTIAL_PACKET,
                "a literal-J two-pass packet is already active",
            )
        if pass_index == 2 and self._rewind_hash is None:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.REWIND_REQUIRED,
                "pass 2 requires a verified rewind over the pass-1 D_fixed order",
            )
        packet = self._expected_packet(pass_index)
        supplied_axes = (
            _identity_axis(context_ids, name="context_ids"),
            _identity_axis(draw_ids, name="draw_ids"),
            _identity_axis(measurement_ids, name="measurement_ids"),
            _identity_axis(capture_ids, name="capture_ids"),
            _identity_axis(source_packet_hashes, name="source_packet_hashes"),
        )
        expected_axes = (
            packet.context_ids,
            packet.draw_ids,
            packet.measurement_ids,
            packet.capture_ids,
            packet.source_packet_hashes,
        )
        if supplied_axes != expected_axes:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.ORDER_MISMATCH,
                "packet identities are not the next registered D_fixed slice",
            )
        if (
            _identity(model_identity_hash, name="model_identity_hash")
            != self.registration.model_identity_hash
            or _identity(chart_identity_hash, name="chart_identity_hash")
            != self.registration.chart_identity_hash
            or _identity(current_mask_hash, name="current_mask_hash")
            != self.registration.current_mask_hash
            or _nonnegative_int(proposal_generation, name="proposal_generation")
            != self.registration.proposal_generation
            or _identity(route_axis_hash, name="route_axis_hash")
            != self.registration.route_axis_hash
            or _identity(route_id_axis_hash, name="route_id_axis_hash")
            != self.registration.route_id_axis_hash
            or _identity(
                same_capture_and_vjp_execution_proof_hash,
                name="same_capture_and_vjp_execution_proof_hash",
            )
            != self.registration.same_capture_and_vjp_execution_proof_hash
            or _identity(
                same_capture_vjp_source_hash, name="same_capture_vjp_source_hash"
            )
            != packet.same_capture_vjp_source_hash
            or _identity(
                literal_j_stream_source_hash, name="literal_j_stream_source_hash"
            )
            != packet.literal_j_stream_source_hash
        ):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.STALE_IDENTITY,
                "packet left the registered D_fixed literal-J chart",
            )
        return packet

    def begin_pass1_packet(self, **kwargs: object) -> None:
        packet = self._validate_packet_begin(pass_index=1, **kwargs)  # type: ignore[arg-type]
        self._active_packet = packet
        self._active_cursor = 0
        self._active_tile_count = 0
        self._active_p_abs = torch.zeros(
            (packet.context_count, len(self.registration.direction_ids)),
            dtype=torch.float64,
            device=self.device,
        )
        self._active_windows = []
        self._active_pass = 1

    def begin_pass2_packet(self, **kwargs: object) -> None:
        packet = self._validate_packet_begin(pass_index=2, **kwargs)  # type: ignore[arg-type]
        if (
            self._pass2_packet_cursor >= len(self._pass1_packet_hashes)
            or packet.descriptor_hash
            != self._pass1_packet_hashes[self._pass2_packet_cursor]
        ):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.ORDER_MISMATCH,
                "pass 2 packet does not replay pass 1 packet identity",
            )
        self._active_packet = packet
        self._active_cursor = 0
        self._active_tile_count = 0
        self._active_windows = []
        self._active_pass = 2

    def _tile_value_hash(
        self,
        *,
        packet: V1LiteralJTwoPassDFixedProjectionPacketDescriptor,
        start: int,
        stop: int,
        literal_j_rows: Any,
        literal_j_tile_source_hash: str | None,
    ) -> str:
        source_hash = (
            None
            if literal_j_tile_source_hash is None
            else _identity(
                literal_j_tile_source_hash,
                name="literal_j_tile_source_hash",
            )
        )
        production_cuda = (
            self.device.type == "cuda" and self.design.production_cuda_required
        )
        if production_cuda and source_hash is None:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.STALE_IDENTITY,
                "production CUDA literal-J replay requires source-owned immutable tile digests",
            )
        private_cpu_value_digest = (
            None
            if production_cuda
            else _tensor_exact_value_digest(
                literal_j_rows,
                role=f"literal_j_packet_{packet.packet_index}_routes_{start}_{stop}",
            )
        )
        return content_hash(
            {
                "schema": "literal_j_two_pass_D_fixed_tile_value_identity_v1",
                "registration_hash": self.registration.registration_hash,
                "packet_descriptor_hash": packet.descriptor_hash,
                "route_window": (start, stop),
                "row_dtype": str(literal_j_rows.dtype),
                "row_shape": tuple(int(item) for item in literal_j_rows.shape),
                "same_capture_and_vjp_execution_proof_hash": (
                    self.registration.same_capture_and_vjp_execution_proof_hash
                ),
                "same_capture_vjp_source_hash": packet.same_capture_vjp_source_hash,
                "literal_j_stream_source_hash": packet.literal_j_stream_source_hash,
                "literal_j_tile_source_hash": source_hash,
                "private_cpu_value_digest": private_cpu_value_digest,
                "production_cuda_source_digest_authority": production_cuda,
            },
            prefix=_PREFIX + "tile-value-",
        )

    def _consume_tile(
        self,
        *,
        pass_index: int,
        start: int,
        stop: int,
        literal_j_rows: Any,
        literal_j_tile_source_hash: str | None,
    ) -> None:
        self._assert_live()
        packet = self._active_packet
        if packet is None or self._active_pass != pass_index:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.PARTIAL_PACKET,
                "begin_packet must precede literal-J route tiles",
            )
        begin = _nonnegative_int(start, name="route_start")
        end = _positive_int(stop, name="route_stop")
        if (
            begin != self._active_cursor
            or end <= begin
            or end > self.registration.route_count
            or end - begin > self.design.route_tile_size
        ):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.ORDER_MISMATCH,
                "literal-J route tiles must cover one contiguous registered prefix",
            )
        rows = _require_tensor(
            literal_j_rows,
            name="literal_j_rows",
            device=self.device,
            shape=(packet.context_count, end - begin),
        )
        if not bool(torch.isfinite(rows).all().item()):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.INVALID_ROW,
                "literal-J tile contains nonfinite values",
            )
        tile_value_hash = self._tile_value_hash(
            packet=packet,
            start=begin,
            stop=end,
            literal_j_rows=rows,
            literal_j_tile_source_hash=literal_j_tile_source_hash,
        )
        if pass_index == 1:
            if self._structured_direction_lease is None:
                direction_tile = self._directions[:, begin:end]
                contribution = rows @ direction_tile.T
                absolute_contribution = torch.abs(rows) @ torch.abs(
                    direction_tile
                ).T
            else:
                outside, shell_indices_live = (
                    self._structured_direction_lease.structured_tensors()
                )
                if tuple(
                    int(item)
                    for item in shell_indices_live.detach().cpu().tolist()
                ) != self._structured_shell_indices:
                    _refuse(
                        V1LiteralJTwoPassDFixedProjectionRefusalCode.MUTATED_CUSTODY,
                        "structured shell indices changed during replay",
                    )
                direction_count = 1 + len(self._structured_shell_indices)
                contribution = torch.zeros(
                    (packet.context_count, direction_count),
                    dtype=torch.float64,
                    device=self.device,
                )
                absolute_contribution = torch.zeros_like(contribution)
                outside_tile = outside[begin:end]
                contribution[:, 0] = rows @ outside_tile
                absolute_contribution[:, 0] = torch.abs(rows) @ torch.abs(
                    outside_tile
                )
                for shell_ordinal, physical_index in enumerate(
                    self._structured_shell_indices,
                    start=1,
                ):
                    if begin <= physical_index < end:
                        local_index = physical_index - begin
                        contribution[:, shell_ordinal] = rows[:, local_index]
                        absolute_contribution[:, shell_ordinal] = torch.abs(
                            rows[:, local_index]
                        )
            offset = packet.context_start
            self._p[offset : offset + packet.context_count].add_(contribution)
            self._active_p_abs.add_(absolute_contribution)
            self._pass1_tile_value_hashes.append(tile_value_hash)
            self._pass1_tile_count += 1
        else:
            window = (begin, end)
            expected = self._window_hash(
                packet=packet,
                pass_index=1,
                windows=(window,),
                ordinal=self._pass2_tile_count,
            )
            if (
                self._pass2_tile_count >= len(self._pass1_window_hashes)
                or self._pass1_window_hashes[self._pass2_tile_count] != expected
            ):
                _refuse(
                    V1LiteralJTwoPassDFixedProjectionRefusalCode.ORDER_MISMATCH,
                    "pass 2 tile does not replay pass 1 route window",
                )
            if (
                self._pass2_tile_count >= len(self._pass1_tile_value_hashes)
                or self._pass1_tile_value_hashes[self._pass2_tile_count]
                != tile_value_hash
            ):
                _refuse(
                    V1LiteralJTwoPassDFixedProjectionRefusalCode.VALUE_MISMATCH,
                    "pass 2 literal-J tile values do not replay pass 1",
                )
            offset = packet.context_start
            context_slice = slice(offset, offset + packet.context_count)
            p = self._p[context_slice]
            p_radius = self._p_radius[context_slice]
            weights = self._weights[context_slice]
            weight_radii = self._weight_radii[context_slice]
            weighted_p = p * weights.reshape(-1, 1)
            self._mz_center[:, begin:end].add_(weighted_p.T @ rows)
            abs_rows = torch.abs(rows)
            arithmetic_mass = torch.abs(weighted_p).T @ abs_rows
            self._mz_abs_mass[:, begin:end].add_(arithmetic_mass)
            uncertainty_mass = (
                weights.reshape(-1, 1) * p_radius
                + weight_radii.reshape(-1, 1) * (torch.abs(p) + p_radius)
            ).T @ abs_rows
            tile_gamma = _gamma(2 * packet.context_count + 8)
            self._mz_radius[:, begin:end].add_(
                _up_tensor(
                    uncertainty_mass + arithmetic_mass * tile_gamma + _F64_MIN_SUBNORMAL
                )
            )
            self._pass2_tile_value_hashes.append(tile_value_hash)
            self._pass2_tile_count += 1
        self._active_windows.append((begin, end))
        self._active_cursor = end
        self._active_tile_count += 1

    def consume_pass1_tile(
        self,
        start: int,
        stop: int,
        literal_j_rows: Any,
        *,
        literal_j_tile_source_hash: str | None = None,
    ) -> None:
        self._consume_tile(
            pass_index=1,
            start=start,
            stop=stop,
            literal_j_rows=literal_j_rows,
            literal_j_tile_source_hash=literal_j_tile_source_hash,
        )

    def consume_pass2_tile(
        self,
        start: int,
        stop: int,
        literal_j_rows: Any,
        *,
        literal_j_tile_source_hash: str | None = None,
    ) -> None:
        self._consume_tile(
            pass_index=2,
            start=start,
            stop=stop,
            literal_j_rows=literal_j_rows,
            literal_j_tile_source_hash=literal_j_tile_source_hash,
        )

    def _window_hash(
        self,
        *,
        packet: V1LiteralJTwoPassDFixedProjectionPacketDescriptor,
        pass_index: int,
        windows: Sequence[tuple[int, int]],
        ordinal: int | None = None,
    ) -> str:
        return content_hash(
            {
                "schema": "literal_j_two_pass_D_fixed_window_v1",
                "registration_hash": self.registration.registration_hash,
                "packet_descriptor_hash": packet.descriptor_hash,
                "pass_index": pass_index,
                "ordinal": ordinal,
                "windows": tuple(tuple(item) for item in windows),
            },
            prefix=_PREFIX + "window-",
        )

    def _end_packet(self, *, pass_index: int) -> None:
        packet = self._active_packet
        if (
            packet is None
            or self._active_pass != pass_index
            or self._active_cursor != self.registration.route_count
        ):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.PARTIAL_PACKET,
                "literal-J packet did not cover the complete route axis",
            )
        if pass_index == 1:
            offset = packet.context_start
            active_abs = self._active_p_abs
            if active_abs is None:
                _refuse(
                    V1LiteralJTwoPassDFixedProjectionRefusalCode.PARTIAL_PACKET,
                    "pass-1 projection mass is unavailable",
                )
            radius = _up_tensor(
                active_abs
                * _gamma(
                    2 * self.registration.route_count + self._active_tile_count + 8
                )
                + _F64_MIN_SUBNORMAL
            )
            self._p_abs_mass[offset : offset + packet.context_count] = active_abs
            self._p_radius[offset : offset + packet.context_count] = radius
            self._pass1_packet_hashes.append(packet.descriptor_hash)
            self._pass1_window_hashes.extend(
                self._window_hash(
                    packet=packet,
                    pass_index=1,
                    windows=(window,),
                    ordinal=ordinal,
                )
                for ordinal, window in enumerate(
                    self._active_windows,
                    start=len(self._pass1_window_hashes),
                )
            )
            self._pass1_packet_cursor += 1
        else:
            offset = packet.context_start
            context_slice = slice(offset, offset + packet.context_count)
            p = self._p[context_slice]
            p_radius = self._p_radius[context_slice]
            weights = self._weights[context_slice].reshape(-1, 1)
            weight_radii = self._weight_radii[context_slice].reshape(-1, 1)
            q_abs_mass = torch.sum(weights * torch.abs(p * p), dim=0)
            self._q_center.add_(torch.sum(weights * p * p, dim=0))
            self._q_abs_mass.add_(q_abs_mass)
            q_uncertainty = torch.sum(
                weights * (2.0 * torch.abs(p) * p_radius + p_radius * p_radius)
                + weight_radii * (torch.abs(p) + p_radius) * (torch.abs(p) + p_radius),
                dim=0,
            )
            q_arithmetic = q_abs_mass * _gamma(2 * packet.context_count + 8)
            self._q_radius.add_(
                _up_tensor(q_uncertainty + q_arithmetic + _F64_MIN_SUBNORMAL)
            )
            self._pass2_packet_hashes.append(packet.descriptor_hash)
            self._pass2_window_hashes.extend(
                self._window_hash(
                    packet=packet,
                    pass_index=1,
                    windows=(window,),
                    ordinal=ordinal,
                )
                for ordinal, window in enumerate(
                    self._active_windows,
                    start=len(self._pass2_window_hashes),
                )
            )
            self._pass2_packet_cursor += 1
        self._active_packet = None
        self._active_cursor = 0
        self._active_tile_count = 0
        self._active_p_abs = None
        self._active_windows = []
        self._active_pass = 0

    def end_pass1_packet(self) -> None:
        self._end_packet(pass_index=1)

    def end_pass2_packet(self) -> None:
        self._end_packet(pass_index=2)

    def rewind_for_pass2(self) -> str:
        self._assert_live()
        if self._active_packet is not None or self._pass1_packet_cursor != len(
            self.registration.packets
        ):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.PARTIAL_PACKET,
                "pass 1 must complete before the D_fixed order can rewind",
            )
        if self._rewind_hash is not None:
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.REWIND_REQUIRED,
                "literal-J two-pass replay was already rewound",
            )
        self._rewind_hash = content_hash(
            {
                "schema": "literal_j_two_pass_D_fixed_verified_rewind_v1",
                "registration_hash": self.registration.registration_hash,
                "packet_hashes": tuple(self._pass1_packet_hashes),
                "window_hashes": tuple(self._pass1_window_hashes),
                "tile_value_hashes": tuple(self._pass1_tile_value_hashes),
                "p_center_fingerprint": _tensor_scalar_fingerprint(
                    self._p, role="p_center"
                ),
                "p_radius_fingerprint": _tensor_scalar_fingerprint(
                    self._p_radius, role="p_radius"
                ),
            },
            prefix=_PREFIX + "rewind-",
        )
        return self._rewind_hash

    def finalize(self) -> V1LiteralJTwoPassDFixedProjectionExecution:
        self._assert_live()
        if (
            self._active_packet is not None
            or self._rewind_hash is None
            or self._pass2_packet_cursor != len(self.registration.packets)
        ):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.PARTIAL_PACKET,
                "both literal-J D_fixed passes must complete before finalization",
            )
        if (
            self._pass1_packet_hashes != self._pass2_packet_hashes
            or self._pass1_window_hashes != self._pass2_window_hashes
            or self._pass1_tile_value_hashes != self._pass2_tile_value_hashes
        ):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.VALUE_MISMATCH,
                "pass 2 did not replay the pass-1 D_fixed order and literal-J values",
            )
        self._mz_radius.add_(
            _up_tensor(
                self._mz_abs_mass * _gamma(self._pass2_tile_count + 8)
                + _F64_MIN_SUBNORMAL
            )
        )
        self._q_radius.add_(
            _up_tensor(
                self._q_abs_mass * _gamma(len(self.registration.packets) + 8)
                + _F64_MIN_SUBNORMAL
            )
        )
        tensors = (
            self._p,
            self._p_radius,
            self._mz_center,
            self._mz_radius,
            self._q_center,
            self._q_radius,
        )
        if not all(bool(torch.isfinite(item).all().item()) for item in tensors):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.NUMERICAL_REFUSAL,
                "literal-J two-pass projection produced nonfinite output",
            )
        if (
            bool((self._p_radius < 0.0).any().item())
            or bool((self._mz_radius < 0.0).any().item())
            or bool((self._q_radius < 0.0).any().item())
        ):
            _refuse(
                V1LiteralJTwoPassDFixedProjectionRefusalCode.NUMERICAL_REFUSAL,
                "literal-J two-pass numerical radii must be nonnegative",
            )
        p_center_fingerprint = _tensor_scalar_fingerprint(self._p, role="p_center")
        p_radius_fingerprint = _tensor_scalar_fingerprint(
            self._p_radius, role="p_radius"
        )
        mjz_center_fingerprint = _tensor_scalar_fingerprint(
            self._mz_center, role="mjz_center"
        )
        mjz_radius_fingerprint = _tensor_scalar_fingerprint(
            self._mz_radius, role="mjz_radius"
        )
        q_center_fingerprint = _tensor_scalar_fingerprint(
            self._q_center, role="q_center"
        )
        q_radius_fingerprint = _tensor_scalar_fingerprint(
            self._q_radius, role="q_radius"
        )
        literal_j_tile_value_ledger_hash = content_hash(
            {
                "schema": "literal_j_two_pass_D_fixed_tile_value_ledger_v1",
                "registration_hash": self.registration.registration_hash,
                "pass1_tile_value_hashes": tuple(self._pass1_tile_value_hashes),
                "pass2_tile_value_hashes": tuple(self._pass2_tile_value_hashes),
                "cuda_source_tile_digests_required": self.device.type == "cuda"
                and self.design.production_cuda_required,
            },
            prefix=_PREFIX + "tile-value-ledger-",
        )
        telemetry = V1LiteralJTwoPassDFixedProjectionTelemetry(
            registration_hash=self.registration.registration_hash,
            device=str(self.device),
            production_cuda_execution=self.device.type == "cuda"
            and self.design.production_cuda_required,
            private_cpu_reference_execution=self.device.type == "cpu",
            context_count=self.registration.structural_context_count,
            route_count=self.registration.route_count,
            direction_count=len(self.registration.direction_ids),
            packet_count=len(self.registration.packets),
            pass1_tile_count=self._pass1_tile_count,
            pass2_tile_count=self._pass2_tile_count,
            hbm_forecast_hash=self._hbm_forecast.forecast_hash,
            literal_j_tile_value_ledger_hash=literal_j_tile_value_ledger_hash,
            cuda_source_tile_digests_required=self.device.type == "cuda"
            and self.design.production_cuda_required,
        )
        receipt = V1LiteralJTwoPassDFixedProjectionReceipt(
            registration_hash=self.registration.registration_hash,
            design_hash=self.design.design_hash,
            telemetry_hash=telemetry.telemetry_hash,
            model_identity_hash=self.registration.model_identity_hash,
            chart_identity_hash=self.registration.chart_identity_hash,
            current_mask_hash=self.registration.current_mask_hash,
            proposal_generation=self.registration.proposal_generation,
            route_axis_hash=self.registration.route_axis_hash,
            route_id_axis_hash=self.registration.route_id_axis_hash,
            route_count=self.registration.route_count,
            direction_ids=self.registration.direction_ids,
            direction_axis_hash=self.registration.direction_axis_hash,
            source_replay_axis_hash=self.registration.source_replay_axis_hash,
            replay_registration_receipt_hash=self.registration.replay_registration_receipt_hash,
            numeric_source_receipt_hash=self.registration.numeric_source_receipt_hash,
            block_support_source_receipt_hash=self.registration.block_support_source_receipt_hash,
            same_capture_and_vjp_execution_proof_hash=self.registration.same_capture_and_vjp_execution_proof_hash,
            literal_j_stream_source_hash=self.registration.composite_literal_j_stream_source_hash,
            context_count=self.registration.structural_context_count,
            packet_count=len(self.registration.packets),
            pass1_tile_count=self._pass1_tile_count,
            pass2_tile_count=self._pass2_tile_count,
            structural_uniform_weight_fraction=self.registration.structural_uniform_weight_fraction,
            context_weight_center_fingerprint=_tensor_scalar_fingerprint(
                self._weights, role="context_weight_center"
            ),
            context_weight_radius_fingerprint=_tensor_scalar_fingerprint(
                self._weight_radii, role="context_weight_radius"
            ),
            p_center_fingerprint=p_center_fingerprint,
            p_radius_fingerprint=p_radius_fingerprint,
            mjz_center_fingerprint=mjz_center_fingerprint,
            mjz_radius_fingerprint=mjz_radius_fingerprint,
            q_center_fingerprint=q_center_fingerprint,
            q_radius_fingerprint=q_radius_fingerprint,
            literal_j_tile_value_ledger_hash=literal_j_tile_value_ledger_hash,
            hbm_forecast_hash=self._hbm_forecast.forecast_hash,
        )
        handle = V1LiteralJTwoPassDFixedProjectionDeviceHandle(
            receipt=receipt,
            tensors={
                "mjz_center": self._mz_center,
                "mjz_radius": self._mz_radius,
                "q_center": self._q_center,
                "q_radius": self._q_radius,
            },
        )
        # The handle now owns exported centers/radii and their mutation guards.
        # Release every accumulator-only tensor before returning so proof mass,
        # context projections, weights, and the direction clone cannot survive
        # into downstream geometry or inflate the next phase's HBM footprint.
        for name in (
            "_directions",
            "_structured_outside",
            "_weights",
            "_weight_radii",
            "_p",
            "_p_abs_mass",
            "_p_radius",
            "_mz_center",
            "_mz_radius",
            "_mz_abs_mass",
            "_q_center",
            "_q_radius",
            "_q_abs_mass",
        ):
            setattr(self, name, None)
        if self._structured_direction_lease is not None:
            self._structured_direction_lease.close()
            self._structured_direction_lease = None
        self._finalized = True
        return V1LiteralJTwoPassDFixedProjectionExecution(
            receipt=receipt,
            telemetry=telemetry,
            device_handle=handle,
        )


__all__ = [
    "LITERAL_J_TWO_PASS_D_FIXED_ALGORITHM",
    "LITERAL_J_TWO_PASS_D_FIXED_AUTHORITY",
    "LITERAL_J_TWO_PASS_D_FIXED_DESIGN_SCHEMA",
    "LITERAL_J_TWO_PASS_D_FIXED_HBM_FORECAST_SCHEMA",
    "LITERAL_J_TWO_PASS_D_FIXED_PACKET_SCHEMA",
    "LITERAL_J_TWO_PASS_D_FIXED_PROJECTION_SCHEMA",
    "LITERAL_J_TWO_PASS_D_FIXED_REGISTRATION_SCHEMA",
    "V1LiteralJTwoPassDFixedProjectionCUDAAccumulator",
    "V1LiteralJTwoPassDFixedProjectionDesign",
    "V1LiteralJTwoPassDFixedProjectionDeviceHandle",
    "V1LiteralJTwoPassDFixedProjectionExecution",
    "V1LiteralJTwoPassDFixedProjectionHBMForecast",
    "V1LiteralJTwoPassDFixedProjectionLease",
    "V1LiteralJTwoPassDFixedProjectionPacketDescriptor",
    "V1LiteralJTwoPassDFixedProjectionReceipt",
    "V1LiteralJTwoPassDFixedProjectionRefusal",
    "V1LiteralJTwoPassDFixedProjectionRefusalCode",
    "V1LiteralJTwoPassDFixedProjectionRegistration",
    "V1LiteralJTwoPassDFixedProjectionTelemetry",
    "register_v1_literal_j_two_pass_d_fixed_projection",
]
