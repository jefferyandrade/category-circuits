"""Source-bound CUDA producer for a frozen residual literal-J prefix.

The residual population geometry is estimated only after one predictable,
paired ``D_W2`` packet has been frozen.  That logical packet is represented by
two pre-registered scorer packets (target, then complement).  Their immutable
literal relaxed-gate Jacobian shards are replayed twice in bounded route tiles:

* pass 1 computes ``p = J z`` for the two frozen KKT directions;
* pass 2 computes stratum-local ``E[p J]`` while ``E[p^2]`` is reduced from
  the resident projections.

No context-by-route HBM bank is retained and no route-scale result crosses to
host.  The returned receipt is array-free and has projection-input authority
only.  The promotion module remains the sole issuer of the live residual
population result consumed by the full-population ``D_fixed + D_W2`` join.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import InitVar, dataclass, field
from enum import Enum
from fractions import Fraction
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_d_fixed_residual_population_projection import (
    V1LiteralJDFixedResidualPopulationProjectionRegistration,
    V1LiteralJResidualPopulationProjectionResult,
    _RESIDUAL_CUDA_PRODUCER_CAPABILITY,
    _issue_v1_literal_j_residual_population_projection_from_cuda_producer,
    _observed_prefix_hash as _canonical_residual_observed_prefix_hash,
    _serfling_radius as _canonical_residual_serfling_radius,
)
from .modern_v1_literal_j_projection_cuda import (
    V1LiteralJPrivatePrimalDirectionCUDALease,
    V1LiteralJPrivatePrimalDirectionDescriptorAxis,
)
from .modern_v1_sampled_dual_chart_row_custody import (
    V1SampledDualChartRowGroupCommit,
    V1SampledDualChartRowGroupPacketProofClosure,
    V1SampledDualChartRowGroupPlan,
    V1SampledDualChartRowRole,
    V1SampledDualChartRowShardAttachment,
)
from .modern_v1_sampled_variable_shell_dw2_suffix import (
    V1SampledVariableShellDW2SuffixPacket,
    V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    V1SampledVariableShellDW2SuffixRegistration,
)

try:  # pragma: no cover - runtime image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


LITERAL_J_RESIDUAL_CUDA_DESIGN_SCHEMA: Final = (
    "partition_free_v1_literal_j_residual_population_cuda_design_v1"
)
LITERAL_J_RESIDUAL_CUDA_SUPPORT_SCHEMA: Final = (
    "partition_free_v1_literal_j_residual_population_cuda_support_bounds_v1"
)
LITERAL_J_RESIDUAL_CUDA_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_literal_j_residual_population_cuda_registration_v2"
)
LITERAL_J_RESIDUAL_CUDA_FORECAST_SCHEMA: Final = (
    "partition_free_v1_literal_j_residual_population_cuda_hbm_forecast_v2"
)
LITERAL_J_RESIDUAL_CUDA_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_literal_j_residual_population_cuda_execution_v3"
)
LITERAL_J_RESIDUAL_CUDA_EXECUTION_HBM_SCHEMA: Final = (
    "partition_free_v1_literal_j_residual_population_cuda_execution_hbm_v1"
)
LITERAL_J_RESIDUAL_CUDA_ALGORITHM: Final = (
    "fixed_paired_prefix_two_pass_route_tiled_literal_J_projection_with_"
    "outward_fp64_and_fixed_time_serfling_v1"
)
LITERAL_J_RESIDUAL_CUDA_AUTHORITY: Final = (
    "residual_projection_numerics_only_no_adaptive_stopping_rank_kkt_face_C_K_"
    "O_commit_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-literal-j-residual-population-cuda-"
_F64_BYTES: Final = 8
_EPS: Final = 2.0**-53
_MIN_SUBNORMAL: Final = float.fromhex("0x0.0000000000001p-1022")
_CPU_REFERENCE_ROUTE_LIMIT: Final = 4096
_FINGERPRINT_TILE_CELLS: Final = 1_048_576
_DESIGN_TOKEN: Final = object()
_SUPPORT_TOKEN: Final = object()
_ANALYTIC_SUPPORT_SOURCE_TOKEN: Final = object()
_ANALYTIC_SUPPORT_SOURCE_OWNER_TOKEN: Final = object()
_FORECAST_TOKEN: Final = object()
_REGISTRATION_TOKEN: Final = object()
_EXECUTION_TOKEN: Final = object()
_EXECUTION_HBM_TOKEN: Final = object()

__all__ = (
    "V1LiteralJResidualAnalyticSupportSourceCUDAHandle",
    "V1LiteralJResidualAnalyticSupportSourceCUDAReceipt",
    "V1LiteralJResidualAnalyticSupportSourceCUDAResult",
    "V1LiteralJResidualDirectionSupportBoundsHandle",
    "V1LiteralJResidualDirectionSupportBoundsReceipt",
    "V1LiteralJResidualPopulationProjectionCUDADesign",
    "V1LiteralJResidualPopulationProjectionCUDAHBMForecast",
    "V1LiteralJResidualPopulationProjectionCUDAExecution",
    "V1LiteralJResidualPopulationProjectionCUDAExecutionHBMReceipt",
    "V1LiteralJResidualPopulationProjectionCUDAExecutionReceipt",
    "V1LiteralJResidualPopulationProjectionCUDARefusal",
    "V1LiteralJResidualPopulationProjectionCUDARefusalCode",
    "V1LiteralJResidualPopulationProjectionCUDARegistration",
    "build_v1_literal_j_residual_population_projection_cuda_hbm_forecast",
    "execute_v1_literal_j_residual_population_projection_cuda",
    "fingerprint_v1_literal_j_residual_cuda_tensor",
    "issue_v1_literal_j_residual_direction_support_bounds_cuda",
    "register_v1_literal_j_residual_population_projection_cuda",
    "register_v1_literal_j_residual_population_projection_cuda_design",
)


class V1LiteralJResidualPopulationProjectionCUDARefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    STALE_IDENTITY = "stale_identity"
    ORDER_MISMATCH = "order_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    DEVICE_REQUIRED = "device_required"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    DETERMINISM_DISABLED = "determinism_disabled"
    SUPPORT_VIOLATION = "support_violation"
    NUMERICAL_REFUSAL = "numerical_refusal"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    MUTATED_CUSTODY = "mutated_custody"
    EVICTED = "evicted"
    ACTIVE_LEASE = "active_lease"
    AUTHORITY_VIOLATION = "authority_violation"


class V1LiteralJResidualPopulationProjectionCUDARefusal(RuntimeError):
    """Typed fail-closed refusal for the residual CUDA producer."""

    def __init__(
        self,
        code: V1LiteralJResidualPopulationProjectionCUDARefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LiteralJResidualPopulationProjectionCUDARefusalCode:
            raise TypeError("residual CUDA refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJResidualPopulationProjectionCUDARefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJResidualPopulationProjectionCUDARefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return int(value)


def _identity_pair(value: Sequence[str], *, name: str) -> tuple[str, str]:
    if isinstance(value, (str, bytes)):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one two-entry identity axis",
        )
    axis = tuple(_identity(item, name=f"{name} entry") for item in value)
    if len(axis) != 2 or axis[0] == axis[1]:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must contain two distinct identities",
        )
    return axis[0], axis[1]


def _count_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one immutable two-entry count axis",
        )
    return (
        _nonnegative_int(value[0], name=f"{name}[0]"),
        _nonnegative_int(value[1], name=f"{name}[1]"),
    )


def _finite_probability(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 < value < 1.0:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one finite float in (0, 1)",
        )
    return value


def _divide_probability_down(value: float, count: int) -> float:
    divisor = _positive_int(count, name="probability family divisor")
    exact = Fraction.from_float(value) / divisor
    result = float(exact)
    if Fraction.from_float(result) > exact:
        result = math.nextafter(result, 0.0)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.NUMERICAL_REFUSAL,
            "fixed-time per-statement alpha underflowed binary64",
        )
    return result


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(payload, prefix=_PREFIX + role + "-")


def _gamma(operation_count: int) -> float:
    count = max(1, int(operation_count))
    product = count * _EPS
    if product >= 0.5:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.NUMERICAL_REFUSAL,
            "requested FP64 enclosure exceeds the finite gamma regime",
        )
    return product / (1.0 - product)


def _up(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _float_up(value: float) -> float:
    return math.nextafter(float(value), math.inf)


def _add_up(left: Any, right: Any) -> Any:
    result = left + right
    if torch is not None and type(result) is torch.Tensor:
        return _up(result)
    return _float_up(float(result))


def _mul_up(left: Any, right: Any) -> Any:
    result = left * right
    if torch is not None and type(result) is torch.Tensor:
        return _up(result)
    return _float_up(float(result))


def _div_up(numerator: float, denominator: float) -> float:
    if denominator <= 0.0 or numerator < 0.0:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.NUMERICAL_REFUSAL,
            "outward division requires a nonnegative numerator and positive denominator",
        )
    return _float_up(numerator / denominator)


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    return (
        id(value),
        int(value.data_ptr()),
        str(value.device),
        str(value.dtype),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        int(value.storage_offset()),
        int(value._version),
    )


def _require_fp64_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    device: Any | None = None,
    nonnegative: bool = False,
) -> Any:
    if torch is None or type(value) is not torch.Tensor:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one exact torch.Tensor",
        )
    if (
        value.dtype != torch.float64
        or tuple(int(item) for item in value.shape) != shape
        or not value.is_contiguous()
        or value.requires_grad
        or (device is not None and value.device != device)
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.DEVICE_REQUIRED,
            f"{name} left its registered contiguous FP64 device/shape",
        )
    if not bool(torch.isfinite(value).all().item()):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.NUMERICAL_REFUSAL,
            f"{name} contains nonfinite values",
        )
    if nonnegative and bool((value < 0.0).any().item()):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.SUPPORT_VIOLATION,
            f"{name} contains a negative support bound",
        )
    return value


def fingerprint_v1_literal_j_residual_cuda_tensor(value: Any, *, role: str) -> str:
    """Return a non-authoritative mutation summary with no route-array D2H.

    This eight-scalar summary is intentionally not a content digest.  Live
    custody relies on exact cloned guards and storage invariants; restart and
    provenance authority must come from source-owned receipts.
    """

    if torch is None or type(value) is not torch.Tensor:
        raise TypeError("residual CUDA fingerprint requires one exact tensor")
    tensor = value.detach()
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        scalars = torch.zeros(8, device=tensor.device, dtype=torch.float64)
    else:
        total = torch.zeros((), device=tensor.device, dtype=torch.float64)
        l1 = torch.zeros_like(total)
        l2_square = torch.zeros_like(total)
        minimum = torch.full_like(total, math.inf)
        maximum = torch.full_like(total, -math.inf)
        maximum_absolute = torch.zeros_like(total)
        for start in range(0, int(flat.numel()), _FINGERPRINT_TILE_CELLS):
            tile = flat[start : start + _FINGERPRINT_TILE_CELLS]
            absolute = torch.abs(tile)
            total.add_(torch.sum(tile))
            l1.add_(torch.sum(absolute))
            l2_square.add_(torch.sum(tile * tile))
            minimum.copy_(torch.minimum(minimum, torch.amin(tile)))
            maximum.copy_(torch.maximum(maximum, torch.amax(tile)))
            maximum_absolute.copy_(
                torch.maximum(maximum_absolute, torch.amax(absolute))
            )
        scalars = torch.stack(
            (
                total,
                l1,
                l2_square,
                minimum,
                maximum,
                maximum_absolute,
                flat[0],
                flat[-1],
            )
        )
    payload = tuple(float(item) for item in scalars.detach().cpu().tolist())
    return _payload_hash(
        {
            "schema": "residual_cuda_tensor_compact_fingerprint_v2",
            "role": _identity(role, name="fingerprint role"),
            "dtype": str(tensor.dtype),
            "device_type": tensor.device.type,
            "shape": tuple(int(item) for item in tensor.shape),
            "scalars": payload,
            "scalar_names": (
                "sum",
                "l1",
                "l2_square",
                "min",
                "max",
                "max_abs",
                "first",
                "last",
            ),
            "route_scale_d2h_array": False,
            "mutation_summary_only": True,
            "content_or_provenance_authority": False,
            "bounded_device_tile_cells": _FINGERPRINT_TILE_CELLS,
        },
        role="tensor-fingerprint",
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJResidualPopulationProjectionCUDADesign:
    """Hardware-only execution controls frozen before source outcomes."""

    route_tile_size: int = 131_072
    hbm_ceiling_fraction: float = 0.85
    registered_external_workspace_bytes: int = 0
    deterministic_algorithms_required: bool = True
    production_cuda_only: bool = True
    route_tile_size_is_hardware_only: bool = True
    statistical_sample_count_controlled_by_tile_size: bool = False
    adaptive_stopping_authorized: bool = False
    schema: str = LITERAL_J_RESIDUAL_CUDA_DESIGN_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _design_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _DESIGN_TOKEN:
            raise PermissionError("residual CUDA designs require their typed issuer")
        object.__setattr__(
            self,
            "route_tile_size",
            _positive_int(self.route_tile_size, name="route_tile_size"),
        )
        object.__setattr__(
            self,
            "registered_external_workspace_bytes",
            _nonnegative_int(
                self.registered_external_workspace_bytes,
                name="registered_external_workspace_bytes",
            ),
        )
        ceiling = self.hbm_ceiling_fraction
        if type(ceiling) is not float or not math.isfinite(ceiling) or ceiling != 0.85:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_DESIGN,
                "canonical residual CUDA HBM ceiling is exactly 85%",
            )
        true_flags = (
            self.deterministic_algorithms_required,
            self.production_cuda_only,
            self.route_tile_size_is_hardware_only,
        )
        false_flags = (
            self.statistical_sample_count_controlled_by_tile_size,
            self.adaptive_stopping_authorized,
        )
        if (
            any(type(item) is not bool for item in true_flags + false_flags)
            or any(item is not True for item in true_flags)
            or any(item is not False for item in false_flags)
            or self.schema != LITERAL_J_RESIDUAL_CUDA_DESIGN_SCHEMA
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
                "residual CUDA design changed its hardware/statistical boundary",
            )
        self.assert_integrity()

    @property
    def design_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }
        observed = _payload_hash(payload, role="design")
        if self._design_hash is not None and self._design_hash != observed:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "residual CUDA design mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "residual CUDA design hash differs",
            )
        object.__setattr__(self, "_design_hash", observed)
        return observed


def register_v1_literal_j_residual_population_projection_cuda_design(
    *,
    route_tile_size: int = 131_072,
    registered_external_workspace_bytes: int = 0,
) -> V1LiteralJResidualPopulationProjectionCUDADesign:
    return V1LiteralJResidualPopulationProjectionCUDADesign(
        route_tile_size=route_tile_size,
        registered_external_workspace_bytes=registered_external_workspace_bytes,
        _issuance_token=_DESIGN_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJResidualDirectionSupportBoundsReceipt:
    """Array-free identity of pre-outcome literal-J/directional supports."""

    population_projection_registration_hash: str
    row_group_plan_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    direction_ids: tuple[str, str]
    private_primal_direction_axis_hash: str
    literal_j_abs_bound_source_hash: str
    direct_projection_abs_bound_derivation_receipt_hash: str
    projection_abs_bound_derivation_hash: str
    literal_j_abs_bound_fingerprint: str
    direct_projection_abs_bounds_fingerprint: str
    projection_abs_bounds_fingerprint: str
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    issued_from_typed_kkt_direction_custody: bool = True
    issued_before_residual_outcomes: bool = True
    projection_bounds_derived_as_outward_min_of_route_sum_and_direct_bound: bool = True
    route_scale_d2h_array_hashing: bool = False
    compact_tensor_fingerprints_are_content_authority: bool = False
    exact_live_tensor_guards_are_mutation_authority: bool = True
    adaptive_stopping_authority: bool = False
    proposal_or_commit_authority: bool = False
    schema: str = LITERAL_J_RESIDUAL_CUDA_SUPPORT_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _SUPPORT_TOKEN:
            raise PermissionError(
                "residual support receipts require their source issuer"
            )
        for name in (
            "population_projection_registration_hash",
            "row_group_plan_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "private_primal_direction_axis_hash",
            "literal_j_abs_bound_source_hash",
            "direct_projection_abs_bound_derivation_receipt_hash",
            "projection_abs_bound_derivation_hash",
            "literal_j_abs_bound_fingerprint",
            "direct_projection_abs_bounds_fingerprint",
            "projection_abs_bounds_fingerprint",
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
            "direction_ids",
            _identity_pair(self.direction_ids, name="direction_ids"),
        )
        flags = (
            self.production_cuda_execution,
            self.private_cpu_reference_execution,
            self.issued_from_typed_kkt_direction_custody,
            self.issued_before_residual_outcomes,
            self.projection_bounds_derived_as_outward_min_of_route_sum_and_direct_bound,
            self.route_scale_d2h_array_hashing,
            self.compact_tensor_fingerprints_are_content_authority,
            self.exact_live_tensor_guards_are_mutation_authority,
            self.adaptive_stopping_authority,
            self.proposal_or_commit_authority,
        )
        if any(type(item) is not bool for item in flags):
            raise TypeError("residual support flags must be exact booleans")
        if (
            self.production_cuda_execution == self.private_cpu_reference_execution
            or not self.issued_from_typed_kkt_direction_custody
            or not self.issued_before_residual_outcomes
            or not self.projection_bounds_derived_as_outward_min_of_route_sum_and_direct_bound
            or self.route_scale_d2h_array_hashing
            or self.compact_tensor_fingerprints_are_content_authority
            or not self.exact_live_tensor_guards_are_mutation_authority
            or self.adaptive_stopping_authority
            or self.proposal_or_commit_authority
            or self.schema != LITERAL_J_RESIDUAL_CUDA_SUPPORT_SCHEMA
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
                "residual support receipt crossed its pre-outcome authority boundary",
            )
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(self._payload(), role="support-receipt")
        if self._receipt_hash is not None and self._receipt_hash != observed:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "residual support receipt mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "residual support receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


class _SupportBoundsLease:
    __slots__ = (
        "_owner",
        "_closed",
        "literal_j_abs_bound",
        "projection_abs_bounds",
    )

    def __init__(self, owner: "V1LiteralJResidualDirectionSupportBoundsHandle") -> None:
        self._owner = owner
        self._closed = False
        tensors = owner._borrow()
        self.literal_j_abs_bound = tensors[0]
        self.projection_abs_bounds = tensors[1]

    def __enter__(self) -> "_SupportBoundsLease":
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
        raise TypeError("residual support leases are nonserializable")


class V1LiteralJResidualDirectionSupportBoundsHandle:
    """Mutation-checked live owner of route and relaxed-direction supports."""

    __slots__ = (
        "receipt",
        "_literal_j_abs_bound",
        "_projection_abs_bounds",
        "_guards",
        "_invariants",
        "_leases",
        "_evicted",
    )

    def __init__(
        self,
        *,
        receipt: V1LiteralJResidualDirectionSupportBoundsReceipt,
        literal_j_abs_bound: Any,
        projection_abs_bounds: Any,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _SUPPORT_TOKEN:
            raise PermissionError(
                "residual support handles require their source issuer"
            )
        if type(receipt) is not V1LiteralJResidualDirectionSupportBoundsReceipt:
            raise TypeError("residual support handle requires its exact receipt")
        receipt.assert_integrity()
        route = _require_fp64_tensor(
            literal_j_abs_bound,
            name="literal_j_abs_bound",
            shape=(receipt.route_count,),
            nonnegative=True,
        )
        projection = _require_fp64_tensor(
            projection_abs_bounds,
            name="projection_abs_bounds",
            shape=(len(receipt.direction_ids),),
            device=route.device,
            nonnegative=True,
        )
        self.receipt = receipt
        self._literal_j_abs_bound = route.detach().clone().contiguous()
        self._projection_abs_bounds = projection.detach().clone().contiguous()
        self._guards = (
            self._literal_j_abs_bound.detach().clone(),
            self._projection_abs_bounds.detach().clone(),
        )
        self._invariants = (
            _tensor_invariant(self._literal_j_abs_bound),
            _tensor_invariant(self._projection_abs_bounds),
        )
        self._leases = 0
        self._evicted = False
        self.assert_live()

    @property
    def device(self) -> Any:
        self.assert_live()
        return self._literal_j_abs_bound.device

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.EVICTED,
                "residual support handle was evicted",
            )
        self.receipt.assert_integrity()
        tensors = (self._literal_j_abs_bound, self._projection_abs_bounds)
        if any(
            _tensor_invariant(value) != invariant or not bool(torch.equal(value, guard))
            for value, guard, invariant in zip(
                tensors, self._guards, self._invariants, strict=True
            )
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.MUTATED_CUSTODY,
                "residual support tensor mutated",
            )
        if (
            fingerprint_v1_literal_j_residual_cuda_tensor(
                tensors[0], role="literal_j_abs_bound"
            )
            != self.receipt.literal_j_abs_bound_fingerprint
            or fingerprint_v1_literal_j_residual_cuda_tensor(
                tensors[1], role="projection_abs_bounds"
            )
            != self.receipt.projection_abs_bounds_fingerprint
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.MUTATED_CUSTODY,
                "residual support tensor fingerprint changed",
            )

    def _borrow(self) -> tuple[Any, Any]:
        self.assert_live()
        self._leases += 1
        return self._literal_j_abs_bound, self._projection_abs_bounds

    def borrow(self) -> _SupportBoundsLease:
        return _SupportBoundsLease(self)

    def _close_lease(self) -> None:
        if self._leases <= 0:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.ACTIVE_LEASE,
                "residual support lease accounting underflow",
            )
        try:
            self.assert_live()
        finally:
            self._leases -= 1

    def evict(self) -> None:
        self.assert_live()
        if self._leases:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.ACTIVE_LEASE,
                "cannot evict residual support bounds under an active lease",
            )
        self._literal_j_abs_bound = None
        self._projection_abs_bounds = None
        self._guards = ()
        self._invariants = ()
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("residual support handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1LiteralJResidualAnalyticSupportSourceCUDAReceipt:
    """Reserved exact backend-issued analytic-J support custody.

    The underscored owner adapter is the only constructor.  It requires an
    opaque backend-owner capability, exact chart lineage, and the typed KKT
    descriptor axis; public callers cannot mint support from raw arrays or
    caller-authored axis strings.
    """

    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    direction_ids: tuple[str, str]
    private_primal_direction_axis_hash: str
    source_owner_identity_hash: str
    backend_execution_receipt_hash: str
    analytic_bound_derivation_receipt_hash: str
    direct_projection_abs_bound_derivation_receipt_hash: str
    literal_j_abs_bound_source_hash: str
    literal_j_abs_bound_fingerprint: str
    direct_projection_abs_bounds_fingerprint: str
    production_backend_source_closed: bool = True
    caller_authored_bound_tensor: bool = False
    caller_authored_source_identity: bool = False
    compact_tensor_fingerprints_are_content_authority: bool = False
    exact_live_tensor_guards_are_mutation_authority: bool = True
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _ANALYTIC_SUPPORT_SOURCE_TOKEN:
            raise PermissionError(
                "analytic literal-J support custody requires the model backend issuer"
            )
        for name in (
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "private_primal_direction_axis_hash",
            "source_owner_identity_hash",
            "backend_execution_receipt_hash",
            "analytic_bound_derivation_receipt_hash",
            "direct_projection_abs_bound_derivation_receipt_hash",
            "literal_j_abs_bound_source_hash",
            "literal_j_abs_bound_fingerprint",
            "direct_projection_abs_bounds_fingerprint",
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
            "direction_ids",
            _identity_pair(self.direction_ids, name="direction_ids"),
        )
        if (
            self.production_backend_source_closed is not True
            or self.caller_authored_bound_tensor is not False
            or self.caller_authored_source_identity is not False
            or self.compact_tensor_fingerprints_are_content_authority is not False
            or self.exact_live_tensor_guards_are_mutation_authority is not True
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
                "analytic literal-J support source is not backend-closed CUDA custody",
            )
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(
            self._payload(), role="analytic-support-source-receipt"
        )
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "analytic support source receipt mutated",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "analytic support source receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


class _AnalyticSupportSourceLease:
    __slots__ = (
        "_owner",
        "_closed",
        "literal_j_abs_bound",
        "direct_projection_abs_bounds",
    )

    def __init__(
        self, owner: "V1LiteralJResidualAnalyticSupportSourceCUDAHandle"
    ) -> None:
        self._owner = owner
        self._closed = False
        tensors = owner._borrow()
        self.literal_j_abs_bound = tensors[0]
        self.direct_projection_abs_bounds = tensors[1]

    def __enter__(self) -> "_AnalyticSupportSourceLease":
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
        raise TypeError("analytic support source leases are nonserializable")


class V1LiteralJResidualAnalyticSupportSourceCUDAHandle:
    __slots__ = (
        "receipt",
        "_literal_j_abs_bound",
        "_direct_projection_abs_bounds",
        "_guards",
        "_invariants",
        "_direction_lease",
        "_leases",
        "_evicted",
    )

    def __init__(
        self,
        *,
        receipt: V1LiteralJResidualAnalyticSupportSourceCUDAReceipt,
        literal_j_abs_bound: Any,
        direct_projection_abs_bounds: Any,
        direction_lease: V1LiteralJPrivatePrimalDirectionCUDALease,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _ANALYTIC_SUPPORT_SOURCE_TOKEN:
            raise PermissionError("analytic support handles require the backend issuer")
        if type(receipt) is not V1LiteralJResidualAnalyticSupportSourceCUDAReceipt:
            raise TypeError("analytic support handle requires its exact receipt")
        if type(direction_lease) is not V1LiteralJPrivatePrimalDirectionCUDALease:
            raise TypeError(
                "analytic support handle requires an exact open KKT direction lease"
            )
        receipt.assert_integrity()
        if (
            direction_lease.descriptor_axis_hash
            != receipt.private_primal_direction_axis_hash
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "analytic support source left its live KKT direction custody",
            )
        bound = _require_fp64_tensor(
            literal_j_abs_bound,
            name="backend literal_j_abs_bound",
            shape=(receipt.route_count,),
            nonnegative=True,
        )
        direct = _require_fp64_tensor(
            direct_projection_abs_bounds,
            name="backend direct_projection_abs_bounds",
            shape=(len(receipt.direction_ids),),
            device=bound.device,
            nonnegative=True,
        )
        if bound.device.type != "cuda" or direct.device.type != "cuda":
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.DEVICE_REQUIRED,
                "analytic support source must remain CUDA-resident",
            )
        self.receipt = receipt
        self._literal_j_abs_bound = bound.detach().clone().contiguous()
        self._direct_projection_abs_bounds = direct.detach().clone().contiguous()
        self._guards = (
            self._literal_j_abs_bound.detach().clone(),
            self._direct_projection_abs_bounds.detach().clone(),
        )
        self._invariants = (
            _tensor_invariant(self._literal_j_abs_bound),
            _tensor_invariant(self._direct_projection_abs_bounds),
        )
        self._direction_lease = direction_lease
        self._leases = 0
        self._evicted = False
        self.assert_live()

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.EVICTED,
                "analytic support source was evicted",
            )
        self.receipt.assert_integrity()
        try:
            direction_axis_hash = self._direction_lease.descriptor_axis_hash
            self._direction_lease.directions_for_literal_j_projection()
        except Exception as error:
            raise V1LiteralJResidualPopulationProjectionCUDARefusal(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.MUTATED_CUSTODY,
                "retained KKT direction lease is no longer live",
            ) from error
        if (
            type(self._direction_lease) is not V1LiteralJPrivatePrimalDirectionCUDALease
            or direction_axis_hash != self.receipt.private_primal_direction_axis_hash
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.MUTATED_CUSTODY,
                "analytic support source lost its exact live KKT direction binding",
            )
        tensors = (
            self._literal_j_abs_bound,
            self._direct_projection_abs_bounds,
        )
        if any(
            _tensor_invariant(value) != invariant or not bool(torch.equal(value, guard))
            for value, guard, invariant in zip(
                tensors, self._guards, self._invariants, strict=True
            )
        ) or (
            fingerprint_v1_literal_j_residual_cuda_tensor(
                tensors[0], role="literal_j_abs_bound"
            )
            != self.receipt.literal_j_abs_bound_fingerprint
            or fingerprint_v1_literal_j_residual_cuda_tensor(
                tensors[1], role="direct_projection_abs_bounds"
            )
            != self.receipt.direct_projection_abs_bounds_fingerprint
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.MUTATED_CUSTODY,
                "analytic support source tensor mutated",
            )

    def _borrow(self) -> tuple[Any, Any]:
        self.assert_live()
        self._leases += 1
        return self._literal_j_abs_bound, self._direct_projection_abs_bounds

    def borrow(self) -> _AnalyticSupportSourceLease:
        return _AnalyticSupportSourceLease(self)

    def retained_direction_lease(
        self,
    ) -> V1LiteralJPrivatePrimalDirectionCUDALease:
        self.assert_live()
        return self._direction_lease

    def _close_lease(self) -> None:
        if self._leases <= 0:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.ACTIVE_LEASE,
                "analytic support source lease accounting underflow",
            )
        try:
            self.assert_live()
        finally:
            self._leases -= 1

    def evict(self) -> None:
        self.assert_live()
        if self._leases:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.ACTIVE_LEASE,
                "cannot evict analytic support source under an active lease",
            )
        self._literal_j_abs_bound = None
        self._direct_projection_abs_bounds = None
        self._guards = ()
        self._invariants = ()
        direction_lease = self._direction_lease
        self._direction_lease = None
        self._evicted = True
        direction_lease.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("analytic support source handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1LiteralJResidualAnalyticSupportSourceCUDAResult:
    receipt: V1LiteralJResidualAnalyticSupportSourceCUDAReceipt
    source_handle: V1LiteralJResidualAnalyticSupportSourceCUDAHandle = field(
        repr=False, compare=False
    )
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _ANALYTIC_SUPPORT_SOURCE_TOKEN:
            raise PermissionError("analytic support results require the backend issuer")
        if (
            type(self.receipt) is not V1LiteralJResidualAnalyticSupportSourceCUDAReceipt
            or type(self.source_handle)
            is not V1LiteralJResidualAnalyticSupportSourceCUDAHandle
        ):
            raise TypeError("analytic support result requires exact receipt and handle")
        self.source_handle.assert_live()
        if self.source_handle.receipt.receipt_hash != self.receipt.receipt_hash:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "analytic support receipt and live handle differ",
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError("analytic support source custody is nonserializable")


def _issue_v1_literal_j_residual_analytic_support_source_cuda_from_owner(
    *,
    owner_token: object,
    analytic_bound_owner_receipt: object,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    direction_lease: V1LiteralJPrivatePrimalDirectionCUDALease,
    literal_j_abs_bound: Any,
    direct_projection_abs_bounds: Any,
) -> V1LiteralJResidualAnalyticSupportSourceCUDAResult:
    """Adapt one backend-owned analytic support result into guarded custody.

    This private hook does not accept raw direction ids or a direction-axis
    hash.  The source owner must pass the exact descriptor axis acquired from
    the still-open KKT direction lease used to derive its direct projection
    bounds.  Custody of that lease transfers to the returned source handle.
    """

    if owner_token is not _ANALYTIC_SUPPORT_SOURCE_OWNER_TOKEN:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
            "analytic support adapter rejected an unknown backend-owner capability",
        )
    from .modern_v1_alexnet_literal_j_support_bounds_cuda import (
        V1AlexNetLiteralJSupportDerivationReceipt,
        v1_alexnet_literal_j_support_owner_identity_hash,
    )

    if (
        type(population_registration)
        is not V1LiteralJDFixedResidualPopulationProjectionRegistration
        or type(direction_lease) is not V1LiteralJPrivatePrimalDirectionCUDALease
        or type(analytic_bound_owner_receipt)
        is not V1AlexNetLiteralJSupportDerivationReceipt
    ):
        raise TypeError(
            "analytic support owner adapter requires exact owner, population, and live-lease types"
        )
    population_registration.assert_integrity()
    analytic_bound_owner_receipt.assert_integrity()
    owner_identity = v1_alexnet_literal_j_support_owner_identity_hash()
    descriptor_axis = direction_lease.descriptor_axis
    directions = direction_lease.directions_for_literal_j_projection()
    if (
        analytic_bound_owner_receipt.population_projection_registration_hash
        != population_registration.registration_hash
        or analytic_bound_owner_receipt.model_identity_hash
        != population_registration.model_identity_hash
        or analytic_bound_owner_receipt.current_mask_hash
        != population_registration.current_mask_hash
        or analytic_bound_owner_receipt.proposal_generation
        != population_registration.proposal_generation
        or analytic_bound_owner_receipt.route_axis_hash
        != population_registration.route_axis_hash
        or analytic_bound_owner_receipt.route_id_axis_hash
        != population_registration.route_id_axis_hash
        or analytic_bound_owner_receipt.route_count
        != population_registration.route_count
        or analytic_bound_owner_receipt.private_primal_direction_axis_hash
        != population_registration.private_primal_direction_axis_hash
        or analytic_bound_owner_receipt.direction_ids
        != population_registration.direction_ids
        or analytic_bound_owner_receipt.source_owner_identity_hash != owner_identity
        or descriptor_axis.route_axis_hash != population_registration.route_axis_hash
        or descriptor_axis.route_count != population_registration.route_count
        or descriptor_axis.direction_ids != population_registration.direction_ids
        or descriptor_axis.descriptor_axis_hash
        != population_registration.private_primal_direction_axis_hash
        or population_registration.direction_source_hash
        != population_registration.private_primal_direction_axis_hash
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "backend analytic supports left the population chart or KKT direction axis",
        )
    route_bound = _require_fp64_tensor(
        literal_j_abs_bound,
        name="owner literal_j_abs_bound",
        shape=(population_registration.route_count,),
        nonnegative=True,
    )
    direct_bound = _require_fp64_tensor(
        direct_projection_abs_bounds,
        name="owner direct_projection_abs_bounds",
        shape=(len(descriptor_axis.direction_ids),),
        device=route_bound.device,
        nonnegative=True,
    )
    if route_bound.device.type != "cuda" or route_bound.device != directions.device:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.DEVICE_REQUIRED,
            "backend analytic support owner must hand off CUDA-resident bounds",
        )
    direct_values = tuple(float(item.item()) for item in direct_bound)
    if direct_values != analytic_bound_owner_receipt.direct_projection_abs_bounds:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "direct projection tensor differs from the exact owner receipt",
        )
    receipt = V1LiteralJResidualAnalyticSupportSourceCUDAReceipt(
        model_identity_hash=population_registration.model_identity_hash,
        chart_identity_hash=population_registration.chart_identity_hash,
        current_mask_hash=population_registration.current_mask_hash,
        proposal_generation=population_registration.proposal_generation,
        route_axis_hash=population_registration.route_axis_hash,
        route_id_axis_hash=population_registration.route_id_axis_hash,
        route_count=population_registration.route_count,
        direction_ids=descriptor_axis.direction_ids,
        private_primal_direction_axis_hash=descriptor_axis.descriptor_axis_hash,
        source_owner_identity_hash=owner_identity,
        backend_execution_receipt_hash=(
            analytic_bound_owner_receipt.backend_execution_receipt_hash
        ),
        analytic_bound_derivation_receipt_hash=(
            analytic_bound_owner_receipt.receipt_hash
        ),
        direct_projection_abs_bound_derivation_receipt_hash=(
            analytic_bound_owner_receipt.direct_projection_abs_bound_derivation_receipt_hash
        ),
        literal_j_abs_bound_source_hash=(
            analytic_bound_owner_receipt.literal_j_abs_bound_source_hash
        ),
        literal_j_abs_bound_fingerprint=(
            fingerprint_v1_literal_j_residual_cuda_tensor(
                route_bound, role="literal_j_abs_bound"
            )
        ),
        direct_projection_abs_bounds_fingerprint=(
            fingerprint_v1_literal_j_residual_cuda_tensor(
                direct_bound, role="direct_projection_abs_bounds"
            )
        ),
        _issuance_token=_ANALYTIC_SUPPORT_SOURCE_TOKEN,
    )
    handle = V1LiteralJResidualAnalyticSupportSourceCUDAHandle(
        receipt=receipt,
        literal_j_abs_bound=route_bound,
        direct_projection_abs_bounds=direct_bound,
        direction_lease=direction_lease,
        _issuance_token=_ANALYTIC_SUPPORT_SOURCE_TOKEN,
    )
    return V1LiteralJResidualAnalyticSupportSourceCUDAResult(
        receipt=receipt,
        source_handle=handle,
        _issuance_token=_ANALYTIC_SUPPORT_SOURCE_TOKEN,
    )


def _derive_projection_supports(*, directions: Any, literal_j_abs_bound: Any) -> Any:
    route_count = int(literal_j_abs_bound.numel())
    absolute_mass = torch.sum(
        torch.abs(directions) * literal_j_abs_bound.reshape(1, -1), dim=1
    )
    relative = _gamma(2 * route_count + 1)
    recovery = _div_up(relative, 1.0 - relative)
    rounding = _add_up(
        _mul_up(absolute_mass, recovery),
        _mul_up(float(2 * route_count + 1), _MIN_SUBNORMAL),
    )
    result = _add_up(absolute_mass, rounding).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.NUMERICAL_REFUSAL,
            "derived relaxed-direction support overflowed FP64",
        )
    return result


def _issue_support_bounds(
    *,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
    directions: Any,
    literal_j_abs_bound: Any,
    literal_j_abs_bound_source_hash: str,
    direct_projection_abs_bound: Any | None,
    direct_projection_abs_bound_derivation_receipt_hash: str,
    production_cuda: bool,
) -> V1LiteralJResidualDirectionSupportBoundsHandle:
    population_registration.assert_integrity()
    row_group_plan.assert_integrity()
    if type(descriptor_axis) is not V1LiteralJPrivatePrimalDirectionDescriptorAxis:
        raise TypeError("support-bound issuance requires an exact direction axis")
    route_count = population_registration.route_count
    direction_count = len(population_registration.direction_ids)
    z = _require_fp64_tensor(
        directions,
        name="directions",
        shape=(direction_count, route_count),
    )
    bound = _require_fp64_tensor(
        literal_j_abs_bound,
        name="literal_j_abs_bound",
        shape=(route_count,),
        device=z.device,
        nonnegative=True,
    )
    if production_cuda and z.device.type != "cuda":
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
            "production residual supports must remain CUDA-resident",
        )
    if not production_cuda and (
        z.device.type != "cpu" or route_count > _CPU_REFERENCE_ROUTE_LIMIT
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
            "private CPU support reference is limited to 4096 routes",
        )
    if (
        row_group_plan.chart_identity_hash
        != population_registration.chart_identity_hash
        or row_group_plan.current_mask_hash != population_registration.current_mask_hash
        or row_group_plan.proposal_generation
        != population_registration.proposal_generation
        or row_group_plan.route_axis_hash != population_registration.route_axis_hash
        or row_group_plan.route_id_axis_hash
        != population_registration.route_id_axis_hash
        or row_group_plan.route_count != route_count
        or row_group_plan.literal_j_stream_identity_hash
        != population_registration.residual_literal_j_stream_source_hash
        or descriptor_axis.route_axis_hash != population_registration.route_axis_hash
        or descriptor_axis.route_count != route_count
        or descriptor_axis.direction_ids != population_registration.direction_ids
        or descriptor_axis.descriptor_axis_hash
        != population_registration.private_primal_direction_axis_hash
        or population_registration.direction_source_hash
        != population_registration.private_primal_direction_axis_hash
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "support source, residual chart, and KKT directions do not close",
        )
    route_sum_projection = _derive_projection_supports(
        directions=z, literal_j_abs_bound=bound
    )
    if direct_projection_abs_bound is None:
        if production_cuda:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
                "production support issuance requires source-owned direct projection bounds",
            )
        direct_projection = route_sum_projection.detach().clone().contiguous()
    else:
        direct_projection = _require_fp64_tensor(
            direct_projection_abs_bound,
            name="direct_projection_abs_bound",
            shape=(direction_count,),
            device=z.device,
            nonnegative=True,
        )
    projection = torch.minimum(route_sum_projection, direct_projection).contiguous()
    literal_source = _identity(
        literal_j_abs_bound_source_hash,
        name="literal_j_abs_bound_source_hash",
    )
    direct_derivation = _identity(
        direct_projection_abs_bound_derivation_receipt_hash,
        name="direct_projection_abs_bound_derivation_receipt_hash",
    )
    derivation = _payload_hash(
        {
            "schema": "relaxed_direction_support_derivation_v1",
            "population_registration_hash": population_registration.registration_hash,
            "row_group_plan_hash": row_group_plan.group_plan_hash,
            "private_primal_direction_axis_hash": descriptor_axis.descriptor_axis_hash,
            "literal_j_abs_bound_source_hash": literal_source,
            "direct_projection_abs_bound_derivation_receipt_hash": direct_derivation,
            "route_sum_bound_fingerprint": fingerprint_v1_literal_j_residual_cuda_tensor(
                route_sum_projection, role="route_sum_projection_abs_bounds"
            ),
            "direct_bound_fingerprint": fingerprint_v1_literal_j_residual_cuda_tensor(
                direct_projection, role="direct_projection_abs_bounds"
            ),
            "derivation": "elementwise_min_of_outward_route_sum_and_source_direct_bound",
            "fp64_gamma_operation_count": 2 * route_count + 1,
        },
        role="projection-support-derivation",
    )
    receipt = V1LiteralJResidualDirectionSupportBoundsReceipt(
        population_projection_registration_hash=(
            population_registration.registration_hash
        ),
        row_group_plan_hash=row_group_plan.group_plan_hash,
        model_identity_hash=population_registration.model_identity_hash,
        chart_identity_hash=population_registration.chart_identity_hash,
        current_mask_hash=population_registration.current_mask_hash,
        proposal_generation=population_registration.proposal_generation,
        route_axis_hash=population_registration.route_axis_hash,
        route_id_axis_hash=population_registration.route_id_axis_hash,
        route_count=route_count,
        direction_ids=population_registration.direction_ids,  # type: ignore[arg-type]
        private_primal_direction_axis_hash=descriptor_axis.descriptor_axis_hash,
        literal_j_abs_bound_source_hash=literal_source,
        direct_projection_abs_bound_derivation_receipt_hash=direct_derivation,
        projection_abs_bound_derivation_hash=derivation,
        literal_j_abs_bound_fingerprint=(
            fingerprint_v1_literal_j_residual_cuda_tensor(
                bound, role="literal_j_abs_bound"
            )
        ),
        direct_projection_abs_bounds_fingerprint=(
            fingerprint_v1_literal_j_residual_cuda_tensor(
                direct_projection, role="direct_projection_abs_bounds"
            )
        ),
        projection_abs_bounds_fingerprint=(
            fingerprint_v1_literal_j_residual_cuda_tensor(
                projection, role="projection_abs_bounds"
            )
        ),
        production_cuda_execution=production_cuda,
        private_cpu_reference_execution=not production_cuda,
        _issuance_token=_SUPPORT_TOKEN,
    )
    return V1LiteralJResidualDirectionSupportBoundsHandle(
        receipt=receipt,
        literal_j_abs_bound=bound,
        projection_abs_bounds=projection,
        _issuance_token=_SUPPORT_TOKEN,
    )


def issue_v1_literal_j_residual_direction_support_bounds_cuda(
    *,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    analytic_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
) -> V1LiteralJResidualDirectionSupportBoundsHandle:
    """Issue supports only from exact backend analytic-J custody."""

    if (
        type(analytic_support_source)
        is not V1LiteralJResidualAnalyticSupportSourceCUDAResult
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
            "production support issuance requires the missing typed backend analytic-J source",
        )
    source = analytic_support_source.receipt
    analytic_support_source.source_handle.assert_live()
    if (
        source.model_identity_hash != population_registration.model_identity_hash
        or source.chart_identity_hash != population_registration.chart_identity_hash
        or source.current_mask_hash != population_registration.current_mask_hash
        or source.proposal_generation != population_registration.proposal_generation
        or source.route_axis_hash != population_registration.route_axis_hash
        or source.route_id_axis_hash != population_registration.route_id_axis_hash
        or source.route_count != population_registration.route_count
        or source.direction_ids != population_registration.direction_ids
        or source.private_primal_direction_axis_hash
        != population_registration.private_primal_direction_axis_hash
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "analytic support source left the registered population chart",
        )

    with analytic_support_source.source_handle.borrow() as support_lease:
        direction_lease = (
            analytic_support_source.source_handle.retained_direction_lease()
        )
        axis = direction_lease.descriptor_axis
        return _issue_support_bounds(
            population_registration=population_registration,
            row_group_plan=row_group_plan,
            descriptor_axis=axis,
            directions=(direction_lease.directions_for_literal_j_projection()),
            literal_j_abs_bound=support_lease.literal_j_abs_bound,
            direct_projection_abs_bound=(support_lease.direct_projection_abs_bounds),
            literal_j_abs_bound_source_hash=(source.literal_j_abs_bound_source_hash),
            direct_projection_abs_bound_derivation_receipt_hash=(
                source.direct_projection_abs_bound_derivation_receipt_hash
            ),
            production_cuda=True,
        )


def _issue_v1_literal_j_residual_direction_support_bounds_cpu_reference(
    *,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
    directions: Any,
    literal_j_abs_bound: Any,
    literal_j_abs_bound_source_hash: str,
    direct_projection_abs_bound: Any | None = None,
    direct_projection_abs_bound_derivation_receipt_hash: str = (
        "private-cpu-reference-direct-bound-equals-outward-route-sum"
    ),
) -> V1LiteralJResidualDirectionSupportBoundsHandle:
    return _issue_support_bounds(
        population_registration=population_registration,
        row_group_plan=row_group_plan,
        descriptor_axis=descriptor_axis,
        directions=directions,
        literal_j_abs_bound=literal_j_abs_bound,
        literal_j_abs_bound_source_hash=literal_j_abs_bound_source_hash,
        direct_projection_abs_bound=direct_projection_abs_bound,
        direct_projection_abs_bound_derivation_receipt_hash=(
            direct_projection_abs_bound_derivation_receipt_hash
        ),
        production_cuda=False,
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJResidualPopulationProjectionCUDAHBMForecast:
    """Exact planning ledger; not execution admission until GEMM profiling lands."""

    design_hash: str
    measurement_device_identity: str
    private_primal_direction_axis_hash: str
    analytic_support_source_receipt_hash: str
    derived_support_bounds_receipt_hash: str
    route_count: int
    direction_count: int
    observed_counts: tuple[int, int]
    route_tile_size: int
    component_bytes: tuple[tuple[str, int], ...]
    shared_resident_bytes: int
    producer_owned_peak_bytes: int
    promotion_handoff_peak_bytes: int
    registered_external_workspace_bytes: int
    projected_incremental_peak_bytes: int
    current_reserved_device_bytes: int
    projected_reserved_device_bytes: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    hbm_ceiling_fraction: float
    hbm_ceiling_bytes: int
    admitted_by_ceiling: bool
    admitted_by_free_memory: bool
    admitted: bool
    shared_resident_components_in_current_reserved: bool = True
    includes_direction_custody_under_retained_lease: bool = True
    includes_analytic_source_custody_and_guard: bool = True
    includes_derived_support_custody_and_guard: bool = True
    includes_source_tile_and_gemm_temporaries: bool = True
    includes_output_mutation_guards: bool = True
    includes_promotion_clone_and_guard_peak: bool = True
    external_workspace_registered_explicitly: bool = True
    context_by_route_hbm_bank_retained: bool = False
    route_scale_device_to_host_transfer: bool = False
    production_live_custody_measurement: bool = True
    private_cpu_arithmetic_reference: bool = False
    internally_measured_gemm_workspace_closed: bool = False
    planning_only: bool = True
    production_execution_admission_authority: bool = False
    schema: str = LITERAL_J_RESIDUAL_CUDA_FORECAST_SCHEMA
    _admission_memory: InitVar[tuple[int, int] | None] = None
    _issuance_token: InitVar[object | None] = None
    _forecast_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _admission_memory: tuple[int, int] | None, _issuance_token: object | None) -> None:
        available, allocated = _admission_memory or (self.device_free_memory_bytes, self.current_reserved_device_bytes)
        if _issuance_token is not _FORECAST_TOKEN:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
                "HBM forecasts require the live-custody or private-reference issuer",
            )
        object.__setattr__(
            self, "design_hash", _identity(self.design_hash, name="design_hash")
        )
        for name in (
            "measurement_device_identity",
            "private_primal_direction_axis_hash",
            "analytic_support_source_receipt_hash",
            "derived_support_bounds_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route_count")
        directions = _positive_int(self.direction_count, name="direction_count")
        counts = _count_pair(self.observed_counts, name="observed_counts")
        tile = _positive_int(self.route_tile_size, name="route_tile_size")
        components = tuple(self.component_bytes)
        if (
            not components
            or len({name for name, _ in components}) != len(components)
            or any(
                type(name) is not str
                or not name
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for name, value in components
            )
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_DESIGN,
                "HBM component ledger is invalid",
            )
        values = {
            name: _nonnegative_int(getattr(self, name), name=name)
            for name in (
                "shared_resident_bytes",
                "producer_owned_peak_bytes",
                "promotion_handoff_peak_bytes",
                "registered_external_workspace_bytes",
                "projected_incremental_peak_bytes",
                "current_reserved_device_bytes",
                "projected_reserved_device_bytes",
                "device_total_memory_bytes",
                "device_free_memory_bytes",
                "hbm_ceiling_bytes",
            )
        }
        expected = _forecast_components(
            route_count=routes,
            direction_count=directions,
            observed_counts=counts,
            route_tile_size=tile,
        )
        expected_map = dict(expected)
        shared = (
            expected_map["direction_custody_under_retained_lease"]
            + expected_map["analytic_source_custody_and_guard"]
            + expected_map["derived_support_custody_and_guard"]
        )
        promotion = expected_map["promotion_result_clone_and_guard_peak"]
        producer = sum(expected_map.values()) - shared - promotion
        incremental = (
            producer + promotion + values["registered_external_workspace_bytes"]
        )
        projected = values["current_reserved_device_bytes"] + incremental
        ceiling = math.floor(values["device_total_memory_bytes"] * 0.85)
        by_ceiling = allocated + incremental <= ceiling
        by_free = incremental <= available
        flags = (
            self.admitted_by_ceiling,
            self.admitted_by_free_memory,
            self.admitted,
            self.shared_resident_components_in_current_reserved,
            self.includes_direction_custody_under_retained_lease,
            self.includes_analytic_source_custody_and_guard,
            self.includes_derived_support_custody_and_guard,
            self.includes_source_tile_and_gemm_temporaries,
            self.includes_output_mutation_guards,
            self.includes_promotion_clone_and_guard_peak,
            self.external_workspace_registered_explicitly,
            self.context_by_route_hbm_bank_retained,
            self.route_scale_device_to_host_transfer,
            self.production_live_custody_measurement,
            self.private_cpu_arithmetic_reference,
            self.internally_measured_gemm_workspace_closed,
            self.planning_only,
            self.production_execution_admission_authority,
        )
        if any(type(item) is not bool for item in flags):
            raise TypeError("HBM forecast flags must be exact booleans")
        if (
            components != expected
            or values["shared_resident_bytes"] != shared
            or values["current_reserved_device_bytes"] < shared
            or values["producer_owned_peak_bytes"] != producer
            or values["promotion_handoff_peak_bytes"] != promotion
            or values["projected_incremental_peak_bytes"] != incremental
            or values["projected_reserved_device_bytes"] != projected
            or self.hbm_ceiling_fraction != 0.85
            or values["hbm_ceiling_bytes"] != ceiling
            or self.admitted_by_ceiling != by_ceiling
            or self.admitted_by_free_memory != by_free
            or self.admitted != (by_ceiling and by_free)
            or any(
                item is not True
                for item in (
                    self.includes_direction_custody_under_retained_lease,
                    self.includes_analytic_source_custody_and_guard,
                    self.includes_derived_support_custody_and_guard,
                    self.includes_source_tile_and_gemm_temporaries,
                    self.includes_output_mutation_guards,
                    self.includes_promotion_clone_and_guard_peak,
                    self.external_workspace_registered_explicitly,
                    self.shared_resident_components_in_current_reserved,
                    self.planning_only,
                )
            )
            or self.context_by_route_hbm_bank_retained
            or self.route_scale_device_to_host_transfer
            or self.production_live_custody_measurement
            == self.private_cpu_arithmetic_reference
            or self.internally_measured_gemm_workspace_closed
            or self.production_execution_admission_authority
            or self.schema != LITERAL_J_RESIDUAL_CUDA_FORECAST_SCHEMA
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_DESIGN,
                "HBM forecast arithmetic or closed boundary does not replay",
            )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "direction_count", directions)
        object.__setattr__(self, "observed_counts", counts)
        object.__setattr__(self, "route_tile_size", tile)
        object.__setattr__(self, "component_bytes", components)
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(self._payload(), role="hbm-forecast")
        if self._forecast_hash is not None and observed != self._forecast_hash:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "HBM forecast mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "HBM forecast hash differs",
            )
        object.__setattr__(self, "_forecast_hash", observed)
        return observed


def _forecast_components(
    *,
    route_count: int,
    direction_count: int,
    observed_counts: tuple[int, int],
    route_tile_size: int,
) -> tuple[tuple[str, int], ...]:
    routes = route_count
    directions = direction_count
    contexts = sum(observed_counts)
    maximum_contexts = max(observed_counts)
    tile = min(route_tile_size, routes)
    output_cells = 4 * directions * routes + 4 * directions
    return (
        ("direction_custody_under_retained_lease", directions * routes * _F64_BYTES),
        (
            "analytic_source_custody_and_guard",
            2 * (routes + directions) * _F64_BYTES,
        ),
        (
            "derived_support_custody_and_guard",
            2 * (routes + directions) * _F64_BYTES,
        ),
        ("projection_center_abs_radius", 3 * contexts * directions * _F64_BYTES),
        ("mjz_center_and_radius", 4 * directions * routes * _F64_BYTES),
        ("q_center_and_radius", 4 * directions * _F64_BYTES),
        ("source_tile_and_absolute", 2 * maximum_contexts * tile * _F64_BYTES),
        (
            "gemm_and_interval_temporaries",
            (
                6 * maximum_contexts * directions
                + 6 * directions * tile
                + maximum_contexts * tile
            )
            * _F64_BYTES,
        ),
        (
            "fingerprint_tile_temporaries",
            2 * min(_FINGERPRINT_TILE_CELLS, max(output_cells, routes)) * _F64_BYTES,
        ),
        ("producer_output_mutation_guards", output_cells * _F64_BYTES),
        (
            "promotion_result_clone_and_guard_peak",
            2 * output_cells * _F64_BYTES,
        ),
    )


def _build_v1_literal_j_residual_population_projection_cuda_hbm_forecast_from_measurement(
    *,
    design: V1LiteralJResidualPopulationProjectionCUDADesign,
    measurement_device_identity: str,
    private_primal_direction_axis_hash: str,
    analytic_support_source_receipt_hash: str,
    derived_support_bounds_receipt_hash: str,
    route_count: int,
    direction_count: int,
    observed_counts: tuple[int, int],
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
    current_reserved_device_bytes: int,
    production_live_custody_measurement: bool,
    _admission_memory: tuple[int, int] | None = None,
) -> V1LiteralJResidualPopulationProjectionCUDAHBMForecast:
    if type(design) is not V1LiteralJResidualPopulationProjectionCUDADesign:
        raise TypeError("HBM forecast requires the exact residual CUDA design")
    design.assert_integrity()
    routes = _positive_int(route_count, name="route_count")
    directions = _positive_int(direction_count, name="direction_count")
    counts = _count_pair(observed_counts, name="observed_counts")
    components = _forecast_components(
        route_count=routes,
        direction_count=directions,
        observed_counts=counts,
        route_tile_size=design.route_tile_size,
    )
    ledger = dict(components)
    shared = (
        ledger["direction_custody_under_retained_lease"]
        + ledger["analytic_source_custody_and_guard"]
        + ledger["derived_support_custody_and_guard"]
    )
    promotion = ledger["promotion_result_clone_and_guard_peak"]
    producer = sum(ledger.values()) - shared - promotion
    incremental = producer + promotion + design.registered_external_workspace_bytes
    total = _positive_int(device_total_memory_bytes, name="device_total_memory_bytes")
    free = _nonnegative_int(device_free_memory_bytes, name="device_free_memory_bytes")
    reserved = _nonnegative_int(
        current_reserved_device_bytes, name="current_reserved_device_bytes"
    )
    if reserved < shared:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_DESIGN,
            "HBM forecast must be measured after direction/source/support custody is resident",
        )
    ceiling = math.floor(total * 0.85)
    projected = reserved + incremental
    available, allocated = _admission_memory or (free, reserved)
    return V1LiteralJResidualPopulationProjectionCUDAHBMForecast(
        design_hash=design.design_hash,
        measurement_device_identity=_identity(
            measurement_device_identity, name="measurement_device_identity"
        ),
        private_primal_direction_axis_hash=_identity(
            private_primal_direction_axis_hash,
            name="private_primal_direction_axis_hash",
        ),
        analytic_support_source_receipt_hash=_identity(
            analytic_support_source_receipt_hash,
            name="analytic_support_source_receipt_hash",
        ),
        derived_support_bounds_receipt_hash=_identity(
            derived_support_bounds_receipt_hash,
            name="derived_support_bounds_receipt_hash",
        ),
        route_count=routes,
        direction_count=directions,
        observed_counts=counts,
        route_tile_size=design.route_tile_size,
        component_bytes=components,
        shared_resident_bytes=shared,
        producer_owned_peak_bytes=producer,
        promotion_handoff_peak_bytes=promotion,
        registered_external_workspace_bytes=design.registered_external_workspace_bytes,
        projected_incremental_peak_bytes=incremental,
        current_reserved_device_bytes=reserved,
        projected_reserved_device_bytes=projected,
        device_total_memory_bytes=total,
        device_free_memory_bytes=free,
        hbm_ceiling_fraction=0.85,
        hbm_ceiling_bytes=ceiling,
        admitted_by_ceiling=allocated + incremental <= ceiling,
        admitted_by_free_memory=incremental <= available,
        admitted=allocated + incremental <= ceiling and incremental <= available,
        shared_resident_components_in_current_reserved=True,
        production_live_custody_measurement=production_live_custody_measurement,
        private_cpu_arithmetic_reference=not production_live_custody_measurement,
        internally_measured_gemm_workspace_closed=False,
        planning_only=True,
        production_execution_admission_authority=False,
        _admission_memory=_admission_memory,
        _issuance_token=_FORECAST_TOKEN,
    )


def _build_v1_literal_j_residual_population_projection_cuda_hbm_forecast_cpu_reference(
    *,
    design: V1LiteralJResidualPopulationProjectionCUDADesign,
    route_count: int,
    direction_count: int,
    observed_counts: tuple[int, int],
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
    current_reserved_device_bytes: int,
) -> V1LiteralJResidualPopulationProjectionCUDAHBMForecast:
    """Exercise exact HBM arithmetic without issuing production authority."""

    return _build_v1_literal_j_residual_population_projection_cuda_hbm_forecast_from_measurement(
        design=design,
        measurement_device_identity="private-cpu-arithmetic-reference",
        private_primal_direction_axis_hash="private-cpu-direction-axis",
        analytic_support_source_receipt_hash="private-cpu-analytic-source",
        derived_support_bounds_receipt_hash="private-cpu-derived-support",
        route_count=route_count,
        direction_count=direction_count,
        observed_counts=observed_counts,
        device_total_memory_bytes=device_total_memory_bytes,
        device_free_memory_bytes=device_free_memory_bytes,
        current_reserved_device_bytes=current_reserved_device_bytes,
        production_live_custody_measurement=False,
    )


def build_v1_literal_j_residual_population_projection_cuda_hbm_forecast(
    *,
    design: V1LiteralJResidualPopulationProjectionCUDADesign,
    analytic_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    derived_support_bounds: V1LiteralJResidualDirectionSupportBoundsHandle,
    observed_counts: tuple[int, int],
) -> V1LiteralJResidualPopulationProjectionCUDAHBMForecast:
    """Measure an authoritative HBM peak under all exact live CUDA custody."""

    if (
        type(design) is not V1LiteralJResidualPopulationProjectionCUDADesign
        or type(analytic_support_source)
        is not V1LiteralJResidualAnalyticSupportSourceCUDAResult
        or type(derived_support_bounds)
        is not V1LiteralJResidualDirectionSupportBoundsHandle
    ):
        raise TypeError("production HBM forecast requires exact live-custody types")
    design.assert_integrity()
    analytic_support_source.source_handle.assert_live()
    derived_support_bounds.assert_live()
    source_receipt = analytic_support_source.receipt
    support_receipt = derived_support_bounds.receipt
    if (
        support_receipt.production_cuda_execution is not True
        or support_receipt.private_cpu_reference_execution is not False
        or source_receipt.private_primal_direction_axis_hash
        != support_receipt.private_primal_direction_axis_hash
        or source_receipt.direction_ids != support_receipt.direction_ids
        or source_receipt.route_count != support_receipt.route_count
        or source_receipt.route_axis_hash != support_receipt.route_axis_hash
        or source_receipt.route_id_axis_hash != support_receipt.route_id_axis_hash
        or source_receipt.literal_j_abs_bound_source_hash
        != support_receipt.literal_j_abs_bound_source_hash
        or source_receipt.direct_projection_abs_bound_derivation_receipt_hash
        != support_receipt.direct_projection_abs_bound_derivation_receipt_hash
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "HBM custody does not describe one analytic-source/support handoff",
        )
    if torch is None or not torch.cuda.is_available():
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.DEVICE_REQUIRED,
            "production HBM forecast requires a real CUDA device",
        )
    if (
        design.deterministic_algorithms_required
        and not torch.are_deterministic_algorithms_enabled()
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.DETERMINISM_DISABLED,
            "production HBM forecast requires deterministic CUDA algorithms",
        )
    counts = _count_pair(observed_counts, name="observed_counts")
    if any(value <= 0 for value in counts):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_DESIGN,
            "production HBM forecast requires nonempty paired strata",
        )
    with analytic_support_source.source_handle.borrow() as source_lease:
        with derived_support_bounds.borrow() as support_lease:
            direction_lease = (
                analytic_support_source.source_handle.retained_direction_lease()
            )
            directions = direction_lease.directions_for_literal_j_projection()
            device = directions.device
            tensors = (
                source_lease.literal_j_abs_bound,
                source_lease.direct_projection_abs_bounds,
                support_lease.literal_j_abs_bound,
                support_lease.projection_abs_bounds,
            )
            if device.type != "cuda" or any(item.device != device for item in tensors):
                _refuse(
                    V1LiteralJResidualPopulationProjectionCUDARefusalCode.DEVICE_REQUIRED,
                    "HBM custody left one common CUDA device",
                )
            free, total = torch.cuda.mem_get_info(device)
            reserved = torch.cuda.memory_reserved(device)
            return _build_v1_literal_j_residual_population_projection_cuda_hbm_forecast_from_measurement(
                design=design,
                measurement_device_identity=str(device),
                private_primal_direction_axis_hash=(
                    source_receipt.private_primal_direction_axis_hash
                ),
                analytic_support_source_receipt_hash=source_receipt.receipt_hash,
                derived_support_bounds_receipt_hash=support_receipt.receipt_hash,
                route_count=source_receipt.route_count,
                direction_count=len(source_receipt.direction_ids),
                observed_counts=counts,
                device_total_memory_bytes=int(total),
                device_free_memory_bytes=int(free),
                current_reserved_device_bytes=int(reserved),
                _admission_memory=(cuda_reusable_memory_bytes(device, framework=torch, free_bytes=free), int(torch.cuda.memory_allocated(device))),
                production_live_custody_measurement=True,
            )


@dataclass(frozen=True, slots=True)
class V1LiteralJResidualPopulationProjectionCUDAExecutionHBMReceipt:
    """Array-free live admission after an eager-GEMM workspace probe."""

    planning_forecast_hash: str
    design_hash: str
    measurement_device_identity: str
    private_primal_direction_axis_hash: str
    analytic_support_source_receipt_hash: str
    derived_support_bounds_receipt_hash: str
    observed_counts: tuple[int, int]
    route_count: int
    direction_count: int
    route_tile_size: int
    maximum_probe_context_count: int
    probe_route_count: int
    pre_probe_free_memory_bytes: int
    pre_probe_total_memory_bytes: int
    pre_probe_reserved_memory_bytes: int
    pre_probe_allocated_memory_bytes: int
    probe_peak_reserved_memory_bytes: int
    probe_peak_allocated_memory_bytes: int
    post_probe_free_memory_bytes: int
    post_probe_total_memory_bytes: int
    post_probe_reserved_memory_bytes: int
    post_probe_allocated_memory_bytes: int
    post_probe_device_used_memory_bytes: int
    transient_allocator_gemm_workspace_bytes: int
    compiler_workspace_bytes: int
    registered_external_workspace_bytes: int
    producer_owned_peak_bytes: int
    promotion_handoff_peak_bytes: int
    live_incremental_peak_bytes: int
    live_projected_device_used_bytes: int
    hbm_ceiling_bytes: int
    admitted_by_ceiling: bool
    admitted_by_free_memory: bool
    admitted: bool
    allocator_and_gemm_probe_performed: bool = True
    eager_torch_mm_out_execution: bool = True
    torch_compile_used: bool = False
    compiler_workspace_measured_zero_by_construction: bool = True
    live_remeasurement_immediately_before_execution: bool = True
    planning_forecast_is_execution_authority: bool = False
    production_execution_admission_authority: bool = True
    schema: str = LITERAL_J_RESIDUAL_CUDA_EXECUTION_HBM_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _EXECUTION_HBM_TOKEN:
            raise PermissionError("live residual HBM receipts require their issuer")
        for name in (
            "planning_forecast_hash",
            "design_hash",
            "measurement_device_identity",
            "private_primal_direction_axis_hash",
            "analytic_support_source_receipt_hash",
            "derived_support_bounds_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "observed_counts",
            _count_pair(self.observed_counts, name="observed_counts"),
        )
        for name in (
            "route_count",
            "direction_count",
            "route_tile_size",
            "maximum_probe_context_count",
            "probe_route_count",
            "pre_probe_total_memory_bytes",
            "probe_peak_reserved_memory_bytes",
            "probe_peak_allocated_memory_bytes",
            "post_probe_total_memory_bytes",
            "hbm_ceiling_bytes",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        for name in (
            "pre_probe_free_memory_bytes",
            "pre_probe_reserved_memory_bytes",
            "pre_probe_allocated_memory_bytes",
            "post_probe_free_memory_bytes",
            "post_probe_reserved_memory_bytes",
            "post_probe_allocated_memory_bytes",
            "post_probe_device_used_memory_bytes",
            "transient_allocator_gemm_workspace_bytes",
            "compiler_workspace_bytes",
            "registered_external_workspace_bytes",
            "producer_owned_peak_bytes",
            "promotion_handoff_peak_bytes",
            "live_incremental_peak_bytes",
            "live_projected_device_used_bytes",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name=name)
            )
        expected_incremental = (
            self.producer_owned_peak_bytes
            + self.promotion_handoff_peak_bytes
            + self.registered_external_workspace_bytes
            + self.transient_allocator_gemm_workspace_bytes
            + self.compiler_workspace_bytes
        )
        expected_used = (
            self.post_probe_total_memory_bytes - self.post_probe_free_memory_bytes
        )
        expected_projected = expected_used + expected_incremental
        reusable = max(0, self.post_probe_reserved_memory_bytes - self.post_probe_allocated_memory_bytes)
        by_ceiling = expected_projected - reusable <= self.hbm_ceiling_bytes
        by_free = expected_incremental <= self.post_probe_free_memory_bytes + reusable
        flags = (
            self.admitted_by_ceiling,
            self.admitted_by_free_memory,
            self.admitted,
            self.allocator_and_gemm_probe_performed,
            self.eager_torch_mm_out_execution,
            self.torch_compile_used,
            self.compiler_workspace_measured_zero_by_construction,
            self.live_remeasurement_immediately_before_execution,
            self.planning_forecast_is_execution_authority,
            self.production_execution_admission_authority,
        )
        if any(type(item) is not bool for item in flags):
            raise TypeError("live HBM receipt flags must be exact booleans")
        if (
            any(value <= 0 for value in self.observed_counts)
            or self.probe_route_count > self.route_count
            or self.probe_route_count > self.route_tile_size
            or self.pre_probe_free_memory_bytes
            > self.pre_probe_total_memory_bytes
            or self.post_probe_free_memory_bytes
            > self.post_probe_total_memory_bytes
            or self.pre_probe_total_memory_bytes
            != self.post_probe_total_memory_bytes
            or self.post_probe_reserved_memory_bytes
            < self.post_probe_allocated_memory_bytes
            or self.post_probe_device_used_memory_bytes != expected_used
            or self.probe_peak_reserved_memory_bytes
            < self.probe_peak_allocated_memory_bytes
            or self.probe_peak_reserved_memory_bytes
            < self.pre_probe_reserved_memory_bytes
            or self.compiler_workspace_bytes != 0
            or self.live_incremental_peak_bytes != expected_incremental
            or self.live_projected_device_used_bytes != expected_projected
            or self.hbm_ceiling_bytes
            != math.floor(self.post_probe_total_memory_bytes * 0.85)
            or self.admitted_by_ceiling != by_ceiling
            or self.admitted_by_free_memory != by_free
            or self.admitted != (by_ceiling and by_free)
            or any(item is not True for item in flags[3:5])
            or self.torch_compile_used
            or not self.compiler_workspace_measured_zero_by_construction
            or not self.live_remeasurement_immediately_before_execution
            or self.planning_forecast_is_execution_authority
            or not self.production_execution_admission_authority
            or self.schema != LITERAL_J_RESIDUAL_CUDA_EXECUTION_HBM_SCHEMA
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_DESIGN,
                "live CUDA HBM admission arithmetic or execution boundary does not replay",
            )
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(self._payload(), role="execution-hbm-receipt")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "live CUDA HBM receipt mutated",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "live CUDA HBM receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


def _measure_v1_literal_j_residual_cuda_execution_hbm(
    *,
    design: V1LiteralJResidualPopulationProjectionCUDADesign,
    planning_forecast: V1LiteralJResidualPopulationProjectionCUDAHBMForecast,
    analytic_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    derived_support_bounds: V1LiteralJResidualDirectionSupportBoundsHandle,
) -> V1LiteralJResidualPopulationProjectionCUDAExecutionHBMReceipt:
    """Probe the exact eager GEMM shapes, then remeasure live admission."""

    design.assert_integrity()
    planning_forecast.assert_integrity()
    analytic_support_source.source_handle.assert_live()
    derived_support_bounds.assert_live()
    source_receipt = analytic_support_source.receipt
    support_receipt = derived_support_bounds.receipt
    if (
        planning_forecast.design_hash != design.design_hash
        or planning_forecast.private_primal_direction_axis_hash
        != source_receipt.private_primal_direction_axis_hash
        or planning_forecast.analytic_support_source_receipt_hash
        != source_receipt.receipt_hash
        or planning_forecast.derived_support_bounds_receipt_hash
        != support_receipt.receipt_hash
        or planning_forecast.route_count != source_receipt.route_count
        or planning_forecast.direction_count != len(source_receipt.direction_ids)
        or planning_forecast.production_live_custody_measurement is not True
        or planning_forecast.private_cpu_arithmetic_reference is not False
        or planning_forecast.planning_only is not True
        or planning_forecast.production_execution_admission_authority is not False
        or planning_forecast.admitted is not True
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "planning forecast left the exact live execution custody",
        )
    direction_lease = analytic_support_source.source_handle.retained_direction_lease()
    directions = direction_lease.directions_for_literal_j_projection()
    device = directions.device
    if (
        torch is None
        or not torch.cuda.is_available()
        or device.type != "cuda"
        or str(device) != planning_forecast.measurement_device_identity
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.DEVICE_REQUIRED,
            "live HBM execution admission requires the forecast CUDA device",
        )
    torch.cuda.synchronize(device)
    pre_free, pre_total = torch.cuda.mem_get_info(device)
    pre_reserved = int(torch.cuda.memory_reserved(device))
    pre_allocated = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    maximum_contexts = max(planning_forecast.observed_counts)
    probe_routes = min(design.route_tile_size, source_receipt.route_count)
    rows = rows_abs = projection = None
    pass1 = pass1_abs = pass2 = pass2_abs = z_tile = None
    try:
        rows = torch.zeros(
            (maximum_contexts, probe_routes),
            device=device,
            dtype=torch.float64,
        )
        rows_abs = torch.abs(rows)
        projection = torch.zeros(
            (maximum_contexts, len(source_receipt.direction_ids)),
            device=device,
            dtype=torch.float64,
        )
        pass1 = torch.empty_like(projection)
        pass1_abs = torch.empty_like(projection)
        pass2 = torch.empty(
            (len(source_receipt.direction_ids), probe_routes),
            device=device,
            dtype=torch.float64,
        )
        pass2_abs = torch.empty_like(pass2)
        z_tile = directions[:, :probe_routes]
        torch.mm(rows, z_tile.T, out=pass1)
        torch.mm(rows_abs, torch.abs(z_tile).T, out=pass1_abs)
        torch.mm(projection.T, rows, out=pass2)
        torch.mm(torch.abs(projection).T, rows_abs, out=pass2_abs)
        torch.cuda.synchronize(device)
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
    except Exception as error:
        raise V1LiteralJResidualPopulationProjectionCUDARefusal(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.NUMERICAL_REFUSAL,
            "eager CUDA GEMM workspace probe failed",
        ) from error
    finally:
        rows = rows_abs = projection = None
        pass1 = pass1_abs = pass2 = pass2_abs = z_tile = None
    torch.cuda.synchronize(device)
    post_free, post_total = torch.cuda.mem_get_info(device)
    post_reserved = int(torch.cuda.memory_reserved(device))
    post_allocated = int(torch.cuda.memory_allocated(device))
    transient = max(0, peak_reserved - max(pre_reserved, post_reserved))
    incremental = (
        planning_forecast.producer_owned_peak_bytes
        + planning_forecast.promotion_handoff_peak_bytes
        + design.registered_external_workspace_bytes
        + transient
    )
    available = cuda_reusable_memory_bytes(device, framework=torch, free_bytes=post_free)
    post_used = int(post_total) - int(post_free)
    projected = post_used + incremental
    ceiling = math.floor(int(post_total) * 0.85)
    receipt = V1LiteralJResidualPopulationProjectionCUDAExecutionHBMReceipt(
        planning_forecast_hash=planning_forecast.forecast_hash,
        design_hash=design.design_hash,
        measurement_device_identity=str(device),
        private_primal_direction_axis_hash=(
            source_receipt.private_primal_direction_axis_hash
        ),
        analytic_support_source_receipt_hash=source_receipt.receipt_hash,
        derived_support_bounds_receipt_hash=support_receipt.receipt_hash,
        observed_counts=planning_forecast.observed_counts,
        route_count=source_receipt.route_count,
        direction_count=len(source_receipt.direction_ids),
        route_tile_size=design.route_tile_size,
        maximum_probe_context_count=maximum_contexts,
        probe_route_count=probe_routes,
        pre_probe_free_memory_bytes=int(pre_free),
        pre_probe_total_memory_bytes=int(pre_total),
        pre_probe_reserved_memory_bytes=pre_reserved,
        pre_probe_allocated_memory_bytes=pre_allocated,
        probe_peak_reserved_memory_bytes=peak_reserved,
        probe_peak_allocated_memory_bytes=peak_allocated,
        post_probe_free_memory_bytes=int(post_free),
        post_probe_total_memory_bytes=int(post_total),
        post_probe_reserved_memory_bytes=post_reserved,
        post_probe_allocated_memory_bytes=post_allocated,
        transient_allocator_gemm_workspace_bytes=transient,
        compiler_workspace_bytes=0,
        registered_external_workspace_bytes=(
            design.registered_external_workspace_bytes
        ),
        producer_owned_peak_bytes=planning_forecast.producer_owned_peak_bytes,
        promotion_handoff_peak_bytes=(
            planning_forecast.promotion_handoff_peak_bytes
        ),
        live_incremental_peak_bytes=incremental,
        post_probe_device_used_memory_bytes=post_used,
        live_projected_device_used_bytes=projected,
        hbm_ceiling_bytes=ceiling,
        admitted_by_ceiling=int(post_total) - available + incremental <= ceiling,
        admitted_by_free_memory=incremental <= available,
        admitted=int(post_total) - available + incremental <= ceiling and incremental <= available,
        _issuance_token=_EXECUTION_HBM_TOKEN,
    )
    if not receipt.admitted:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.HBM_FORECAST_EXCEEDED,
            "live post-probe CUDA HBM admission exceeded the registered 85% ceiling or free memory",
        )
    return receipt


@dataclass(frozen=True, slots=True)
class V1LiteralJResidualPopulationProjectionCUDARegistration:
    """Outcome-independent binding of one fixed paired prefix and its scorer plan."""

    population_projection_registration_hash: str
    suffix_registration_hash: str
    packet_receipt_hash: str
    prefix_freeze_receipt_hash: str
    row_group_plan_hash: str
    design_hash: str
    support_bounds_receipt_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    direction_ids: tuple[str, str]
    d_fixed_projection_direction_axis_hash: str
    private_primal_direction_axis_hash: str
    stratum_ids: tuple[str, str]
    residual_population_counts: tuple[int, int]
    registered_observed_counts: tuple[int, int]
    residual_sealed_population_order_hashes: tuple[str, str]
    packet_context_axis_hashes: tuple[str, str]
    packet_draw_axis_hashes: tuple[str, str]
    scorer_source_capture_identity_hashes: tuple[str, str]
    scorer_packet_plan_hashes: tuple[str, str]
    scorer_provider_plan_hashes: tuple[str, str]
    literal_j_stream_source_hash: str
    frozen_dw2_prefix_axis_hashes: tuple[str, str]
    frozen_dw2_pair_prefix_axis_hash: str
    familywise_alpha: float
    simultaneous_statement_count: int
    per_statement_two_sided_alpha: float
    fixed_prefix_registered_before_outcomes: bool = True
    directions_and_supports_frozen_before_outcomes: bool = True
    one_paired_packet_only: bool = True
    target_then_complement_scorer_order: bool = True
    adaptive_stopping_authorized: bool = False
    outcome_payload_present: bool = False
    decision_or_commit_authority: bool = False
    algorithm: str = LITERAL_J_RESIDUAL_CUDA_ALGORITHM
    authority_scope: str = LITERAL_J_RESIDUAL_CUDA_AUTHORITY
    schema: str = LITERAL_J_RESIDUAL_CUDA_REGISTRATION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _registration_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _REGISTRATION_TOKEN:
            raise PermissionError("residual CUDA registrations require their issuer")
        for name in (
            "population_projection_registration_hash",
            "suffix_registration_hash",
            "packet_receipt_hash",
            "prefix_freeze_receipt_hash",
            "row_group_plan_hash",
            "design_hash",
            "support_bounds_receipt_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "d_fixed_projection_direction_axis_hash",
            "private_primal_direction_axis_hash",
            "literal_j_stream_source_hash",
            "frozen_dw2_pair_prefix_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "direction_ids",
            "stratum_ids",
            "residual_sealed_population_order_hashes",
            "packet_context_axis_hashes",
            "packet_draw_axis_hashes",
            "scorer_source_capture_identity_hashes",
            "scorer_packet_plan_hashes",
            "scorer_provider_plan_hashes",
            "frozen_dw2_prefix_axis_hashes",
        ):
            object.__setattr__(
                self, name, _identity_pair(getattr(self, name), name=name)
            )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal_generation"),
        )
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route_count")
        )
        for name in ("residual_population_counts", "registered_observed_counts"):
            object.__setattr__(self, name, _count_pair(getattr(self, name), name=name))
        alpha = _finite_probability(self.familywise_alpha, name="familywise_alpha")
        statements = _positive_int(
            self.simultaneous_statement_count,
            name="simultaneous_statement_count",
        )
        per_statement = _finite_probability(
            self.per_statement_two_sided_alpha,
            name="per_statement_two_sided_alpha",
        )
        if (
            statements != 2 * len(self.direction_ids) * (self.route_count + 1)
            or per_statement != _divide_probability_down(alpha, statements)
            or any(
                observed <= 0 or observed > population
                for observed, population in zip(
                    self.registered_observed_counts,
                    self.residual_population_counts,
                    strict=True,
                )
            )
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_DESIGN,
                "registered residual count or fixed-time statement family is invalid",
            )
        flags = (
            self.fixed_prefix_registered_before_outcomes,
            self.directions_and_supports_frozen_before_outcomes,
            self.one_paired_packet_only,
            self.target_then_complement_scorer_order,
            self.adaptive_stopping_authorized,
            self.outcome_payload_present,
            self.decision_or_commit_authority,
        )
        if any(type(item) is not bool for item in flags):
            raise TypeError("residual CUDA registration flags must be exact booleans")
        if (
            any(item is not True for item in flags[:4])
            or any(item is not False for item in flags[4:])
            or self.algorithm != LITERAL_J_RESIDUAL_CUDA_ALGORITHM
            or self.authority_scope != LITERAL_J_RESIDUAL_CUDA_AUTHORITY
            or self.schema != LITERAL_J_RESIDUAL_CUDA_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
                "residual CUDA registration crossed its causal boundary",
            )
        object.__setattr__(self, "familywise_alpha", alpha)
        object.__setattr__(self, "simultaneous_statement_count", statements)
        object.__setattr__(self, "per_statement_two_sided_alpha", per_statement)
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(self._payload(), role="registration")
        if self._registration_hash is not None and observed != self._registration_hash:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "residual CUDA registration mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "residual CUDA registration hash differs",
            )
        object.__setattr__(self, "_registration_hash", observed)
        return observed


def _axis_hash(role: str, values: Sequence[str]) -> str:
    return _payload_hash(
        {"schema": "registered_ordered_axis_v1", "role": role, "values": tuple(values)},
        role="axis",
    )


def _register_v1_literal_j_residual_population_projection_cuda(
    *,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
    support_bounds: V1LiteralJResidualDirectionSupportBoundsHandle,
    design: V1LiteralJResidualPopulationProjectionCUDADesign,
    production_cuda_required: bool,
) -> V1LiteralJResidualPopulationProjectionCUDARegistration:
    """Freeze one predictable paired residual prefix before scorer outcomes."""

    exact_types = (
        (
            population_registration,
            V1LiteralJDFixedResidualPopulationProjectionRegistration,
        ),
        (suffix_registration, V1SampledVariableShellDW2SuffixRegistration),
        (packet, V1SampledVariableShellDW2SuffixPacket),
        (prefix_freeze, V1SampledVariableShellDW2SuffixPrefixFreezeReceipt),
        (row_group_plan, V1SampledDualChartRowGroupPlan),
        (descriptor_axis, V1LiteralJPrivatePrimalDirectionDescriptorAxis),
        (support_bounds, V1LiteralJResidualDirectionSupportBoundsHandle),
        (design, V1LiteralJResidualPopulationProjectionCUDADesign),
    )
    if any(type(value) is not expected for value, expected in exact_types):
        raise TypeError("residual CUDA registration received an inexact parent type")
    population_registration.assert_integrity()
    suffix_registration.assert_integrity()
    packet.assert_integrity()
    prefix_freeze.assert_integrity()
    row_group_plan.assert_integrity()
    support_bounds.assert_live()
    design.assert_integrity()
    support = support_bounds.receipt
    observed = packet.after_suffix_stratum_counts
    scorer_packets = row_group_plan.packet_plans
    expected_contexts = (packet.target_context_ids, packet.complement_context_ids)
    if (
        population_registration.suffix_registration_hash
        != suffix_registration.receipt_hash
        or packet.suffix_registration_hash != suffix_registration.receipt_hash
        or prefix_freeze.suffix_registration_hash != suffix_registration.receipt_hash
        or prefix_freeze.completed_packet_hashes != (packet.receipt_hash,)
        or prefix_freeze.consumed_dw2_stratum_counts != observed
        or prefix_freeze.generation_frame_hash != packet.generation_frame_hash
        or prefix_freeze.discovery_stream_hash != packet.discovery_stream_hash
        or prefix_freeze.generation_frame_hash
        != suffix_registration.generation_frame_hash
        or prefix_freeze.discovery_stream_hash
        != suffix_registration.discovery_stream_hash
        or prefix_freeze.generation_index != population_registration.proposal_generation
        or prefix_freeze.generation_index != suffix_registration.generation_index
        or prefix_freeze.provisional_shell_hash
        != suffix_registration.provisional_shell_hash
        or prefix_freeze.provisional_action_family_hash
        != suffix_registration.provisional_action_family_hash
        or prefix_freeze.consumed_d1_stratum_counts
        != suffix_registration.consumed_d1_stratum_counts
        or prefix_freeze.population_stratum_counts
        != suffix_registration.population_stratum_counts
        or prefix_freeze.final_d_stratum_counts
        != tuple(
            suffix_registration.consumed_d1_stratum_counts[h] + observed[h]
            for h in (0, 1)
        )
        or packet.packet_ordinal != 0
        or packet.before_suffix_stratum_counts != (0, 0)
        or observed
        != (len(packet.target_context_ids), len(packet.complement_context_ids))
        or any(value <= 0 for value in observed)
        or any(
            observed[h] > population_registration.residual_stratum_counts[h]
            for h in (0, 1)
        )
        or len(scorer_packets) != 2
        or row_group_plan.first_prefix_look_ordinal != 0
        or row_group_plan.expected_prior_manifest_hash is not None
        or tuple(item.packet_ordinal for item in scorer_packets) != (0, 1)
        or tuple(item.context_ids for item in scorer_packets) != expected_contexts
        or tuple(item.stratum_labels for item in scorer_packets)
        != (
            (population_registration.stratum_ids[0],) * observed[0],
            (population_registration.stratum_ids[1],) * observed[1],
        )
        or row_group_plan.chart_identity_hash
        != population_registration.chart_identity_hash
        or row_group_plan.current_mask_hash != population_registration.current_mask_hash
        or row_group_plan.proposal_generation
        != population_registration.proposal_generation
        or row_group_plan.route_axis_hash != population_registration.route_axis_hash
        or row_group_plan.route_id_axis_hash
        != population_registration.route_id_axis_hash
        or row_group_plan.route_count != population_registration.route_count
        or row_group_plan.literal_j_stream_identity_hash
        != population_registration.residual_literal_j_stream_source_hash
        or descriptor_axis.descriptor_axis_hash
        != support.private_primal_direction_axis_hash
        or descriptor_axis.descriptor_axis_hash
        != population_registration.private_primal_direction_axis_hash
        or population_registration.direction_source_hash
        != population_registration.private_primal_direction_axis_hash
        or support.population_projection_registration_hash
        != population_registration.registration_hash
        or support.row_group_plan_hash != row_group_plan.group_plan_hash
        or support.private_primal_direction_axis_hash
        != descriptor_axis.descriptor_axis_hash
        or support.model_identity_hash != population_registration.model_identity_hash
        or support.chart_identity_hash != population_registration.chart_identity_hash
        or support.current_mask_hash != population_registration.current_mask_hash
        or support.proposal_generation != population_registration.proposal_generation
        or support.route_axis_hash != population_registration.route_axis_hash
        or support.route_id_axis_hash != population_registration.route_id_axis_hash
        or support.route_count != population_registration.route_count
        or support.direction_ids != descriptor_axis.direction_ids
        or support.production_cuda_execution is not production_cuda_required
        or support.private_cpu_reference_execution is production_cuda_required
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "paired suffix, scorer plan, support source, and population chart diverged",
        )
    statements = (
        2
        * len(descriptor_axis.direction_ids)
        * (population_registration.route_count + 1)
    )
    return V1LiteralJResidualPopulationProjectionCUDARegistration(
        population_projection_registration_hash=(
            population_registration.registration_hash
        ),
        suffix_registration_hash=suffix_registration.receipt_hash,
        packet_receipt_hash=packet.receipt_hash,
        prefix_freeze_receipt_hash=prefix_freeze.receipt_hash,
        row_group_plan_hash=row_group_plan.group_plan_hash,
        design_hash=design.design_hash,
        support_bounds_receipt_hash=support.receipt_hash,
        model_identity_hash=population_registration.model_identity_hash,
        chart_identity_hash=population_registration.chart_identity_hash,
        current_mask_hash=population_registration.current_mask_hash,
        proposal_generation=population_registration.proposal_generation,
        route_axis_hash=population_registration.route_axis_hash,
        route_id_axis_hash=population_registration.route_id_axis_hash,
        route_count=population_registration.route_count,
        direction_ids=descriptor_axis.direction_ids,
        d_fixed_projection_direction_axis_hash=(
            population_registration.d_fixed_projection_direction_axis_hash
        ),
        private_primal_direction_axis_hash=descriptor_axis.descriptor_axis_hash,
        stratum_ids=population_registration.stratum_ids,
        residual_population_counts=population_registration.residual_stratum_counts,
        registered_observed_counts=observed,
        residual_sealed_population_order_hashes=(
            population_registration.residual_sealed_population_order_hashes
        ),
        packet_context_axis_hashes=(
            _axis_hash("target-contexts", scorer_packets[0].context_ids),
            _axis_hash("complement-contexts", scorer_packets[1].context_ids),
        ),
        packet_draw_axis_hashes=(
            _axis_hash("target-draws", scorer_packets[0].draw_ids),
            _axis_hash("complement-draws", scorer_packets[1].draw_ids),
        ),
        scorer_source_capture_identity_hashes=tuple(
            item.source_capture_identity_hash for item in scorer_packets
        ),  # type: ignore[arg-type]
        scorer_packet_plan_hashes=tuple(
            item.packet_plan_hash for item in scorer_packets
        ),  # type: ignore[arg-type]
        scorer_provider_plan_hashes=tuple(
            item.provider_plan_identity_hash for item in scorer_packets
        ),  # type: ignore[arg-type]
        literal_j_stream_source_hash=(
            population_registration.residual_literal_j_stream_source_hash
        ),
        frozen_dw2_prefix_axis_hashes=(
            prefix_freeze.dw2_target_prefix_axis_hash,
            prefix_freeze.dw2_complement_prefix_axis_hash,
        ),
        frozen_dw2_pair_prefix_axis_hash=prefix_freeze.dw2_pair_prefix_axis_hash,
        familywise_alpha=population_registration.familywise_alpha,
        simultaneous_statement_count=statements,
        per_statement_two_sided_alpha=(
            _divide_probability_down(
                population_registration.familywise_alpha, statements
            )
        ),
        _issuance_token=_REGISTRATION_TOKEN,
    )


def register_v1_literal_j_residual_population_projection_cuda(
    *,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
    support_bounds: V1LiteralJResidualDirectionSupportBoundsHandle,
    design: V1LiteralJResidualPopulationProjectionCUDADesign,
) -> V1LiteralJResidualPopulationProjectionCUDARegistration:
    return _register_v1_literal_j_residual_population_projection_cuda(
        population_registration=population_registration,
        suffix_registration=suffix_registration,
        packet=packet,
        prefix_freeze=prefix_freeze,
        row_group_plan=row_group_plan,
        descriptor_axis=descriptor_axis,
        support_bounds=support_bounds,
        design=design,
        production_cuda_required=True,
    )


def _register_v1_literal_j_residual_population_projection_cpu_reference(
    *,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    descriptor_axis: V1LiteralJPrivatePrimalDirectionDescriptorAxis,
    support_bounds: V1LiteralJResidualDirectionSupportBoundsHandle,
    design: V1LiteralJResidualPopulationProjectionCUDADesign,
) -> V1LiteralJResidualPopulationProjectionCUDARegistration:
    return _register_v1_literal_j_residual_population_projection_cuda(
        population_registration=population_registration,
        suffix_registration=suffix_registration,
        packet=packet,
        prefix_freeze=prefix_freeze,
        row_group_plan=row_group_plan,
        descriptor_axis=descriptor_axis,
        support_bounds=support_bounds,
        design=design,
        production_cuda_required=False,
    )


def _fraction_center_radius(value: Fraction) -> tuple[float, float]:
    center = float(value)
    if not math.isfinite(center):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.NUMERICAL_REFUSAL,
            "exact CPU-reference value overflowed binary64",
        )
    error = abs(value - Fraction.from_float(center))
    radius = float(error)
    if Fraction.from_float(radius) < error:
        radius = math.nextafter(radius, math.inf)
    return center, math.nextafter(radius, math.inf)


def _serfling_factor(*, observed: int, population: int, alpha: float) -> float:
    if observed >= population:
        return 0.0
    if observed <= 0 or population <= 0 or observed > population:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_DESIGN,
            "Serfling counts do not form a nonempty finite-population prefix",
        )
    return _canonical_residual_serfling_radius(
        value_range=1.0,
        observed=observed,
        population=population,
        alpha=alpha,
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJResidualPopulationProjectionCPUReference:
    """Small exact-rational numerical oracle with no production authority."""

    mjz_center: Any = field(repr=False, compare=False)
    mjz_radius: Any = field(repr=False, compare=False)
    q_center: Any = field(repr=False, compare=False)
    q_radius: Any = field(repr=False, compare=False)
    observed_counts: tuple[int, int]
    residual_population_counts: tuple[int, int]
    simultaneous_statement_count: int
    per_statement_two_sided_alpha: float
    alpha_spent: float
    exact_residual_census: bool
    private_cpu_reference_only: bool = True
    route_limit: int = _CPU_REFERENCE_ROUTE_LIMIT
    production_authority: bool = False
    adaptive_stopping_authorized: bool = False

    def __post_init__(self) -> None:
        if torch is None:
            raise RuntimeError("CPU reference requires PyTorch")
        centers = (self.mjz_center, self.q_center)
        radii = (self.mjz_radius, self.q_radius)
        if any(
            type(item) is not torch.Tensor
            or item.device.type != "cpu"
            or item.dtype != torch.float64
            or not item.is_contiguous()
            for item in centers + radii
        ):
            raise TypeError("CPU reference outputs must be contiguous CPU FP64 tensors")
        if any(
            not bool(torch.isfinite(item).all().item()) for item in centers + radii
        ) or any(bool((item < 0.0).any().item()) for item in radii):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.NUMERICAL_REFUSAL,
                "CPU reference emitted a nonfinite center or invalid radius",
            )
        observed = _count_pair(self.observed_counts, name="observed_counts")
        population = _count_pair(
            self.residual_population_counts,
            name="residual_population_counts",
        )
        direction_count = int(self.q_center.numel() // 2)
        route_count = int(self.mjz_center.shape[-1])
        census = observed == population
        if (
            tuple(self.mjz_center.shape) != (2, direction_count, route_count)
            or self.mjz_radius.shape != self.mjz_center.shape
            or tuple(self.q_center.shape) != (2, direction_count)
            or self.q_radius.shape != self.q_center.shape
            or route_count > _CPU_REFERENCE_ROUTE_LIMIT
            or self.simultaneous_statement_count
            != 2 * direction_count * (route_count + 1)
            or self.exact_residual_census != census
            or (census and self.alpha_spent != 0.0)
            or (not census and not 0.0 < self.alpha_spent < 1.0)
            or self.private_cpu_reference_only is not True
            or self.route_limit != _CPU_REFERENCE_ROUTE_LIMIT
            or self.production_authority is not False
            or self.adaptive_stopping_authorized is not False
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
                "CPU reference crossed its numerical-oracle boundary",
            )


def _compute_v1_literal_j_residual_population_projection_cpu_reference(
    *,
    directions: Any,
    literal_j_rows_by_stratum: Sequence[Any],
    literal_j_abs_bound: Any,
    residual_population_counts: tuple[int, int],
    familywise_alpha: float,
) -> V1LiteralJResidualPopulationProjectionCPUReference:
    """Compute exact small-domain centers and outward fixed-time intervals."""

    if torch is None:
        raise RuntimeError("CPU residual reference requires PyTorch")
    if type(directions) is not torch.Tensor or directions.ndim != 2:
        raise TypeError("CPU reference directions must be one exact rank-two tensor")
    direction_count, route_count = tuple(int(item) for item in directions.shape)
    _positive_int(direction_count, name="CPU reference direction_count")
    _positive_int(route_count, name="CPU reference route_count")
    if route_count > _CPU_REFERENCE_ROUTE_LIMIT:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
            "CPU residual reference is limited to 4096 routes",
        )
    z = _require_fp64_tensor(
        directions,
        name="directions",
        shape=(direction_count, route_count),
    )
    if z.device.type != "cpu":
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
            "CPU residual reference requires CPU tensors",
        )
    if len(literal_j_rows_by_stratum) != 2:
        raise TypeError("CPU reference rows must cover exactly two strata")
    rows = tuple(literal_j_rows_by_stratum)
    observed = tuple(int(item.shape[0]) for item in rows)
    rows = tuple(
        _require_fp64_tensor(
            value,
            name=f"literal_j_rows_by_stratum[{index}]",
            shape=(observed[index], route_count),
            device=z.device,
        )
        for index, value in enumerate(rows)
    )
    population = _count_pair(
        residual_population_counts,
        name="residual_population_counts",
    )
    if any(observed[h] <= 0 or observed[h] > population[h] for h in (0, 1)):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_DESIGN,
            "CPU reference rows are not one nonempty bounded residual prefix",
        )
    bound = _require_fp64_tensor(
        literal_j_abs_bound,
        name="literal_j_abs_bound",
        shape=(route_count,),
        device=z.device,
        nonnegative=True,
    )
    if any(
        bool((torch.abs(value) > bound.reshape(1, -1)).any().item()) for value in rows
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.SUPPORT_VIOLATION,
            "CPU reference literal-J row exceeded its pre-outcome support",
        )
    p_bound = _derive_projection_supports(directions=z, literal_j_abs_bound=bound)
    statements = 2 * direction_count * (route_count + 1)
    alpha = _finite_probability(familywise_alpha, name="familywise_alpha")
    per_statement = _divide_probability_down(alpha, statements)
    center_m = torch.zeros((2, direction_count, route_count), dtype=torch.float64)
    radius_m = torch.zeros_like(center_m)
    center_q = torch.zeros((2, direction_count), dtype=torch.float64)
    radius_q = torch.zeros_like(center_q)
    for h in (0, 1):
        exact_m = [
            [Fraction(0) for _ in range(route_count)] for _ in range(direction_count)
        ]
        exact_q = [Fraction(0) for _ in range(direction_count)]
        for row_index in range(observed[h]):
            row = tuple(
                Fraction.from_float(float(rows[h][row_index, route].item()))
                for route in range(route_count)
            )
            for direction in range(direction_count):
                projection = sum(
                    (
                        Fraction.from_float(float(z[direction, route].item()))
                        * row[route]
                        for route in range(route_count)
                    ),
                    Fraction(0),
                )
                if abs(float(projection)) > float(p_bound[direction].item()):
                    _refuse(
                        V1LiteralJResidualPopulationProjectionCUDARefusalCode.SUPPORT_VIOLATION,
                        "CPU reference projection exceeded its derived support",
                    )
                exact_q[direction] += projection * projection
                for route in range(route_count):
                    exact_m[direction][route] += projection * row[route]
        factor = _serfling_factor(
            observed=observed[h],
            population=population[h],
            alpha=per_statement,
        )
        for direction in range(direction_count):
            q_value = exact_q[direction] / observed[h]
            q_center, q_numeric = _fraction_center_radius(q_value)
            projection_support = float(p_bound[direction].item())
            q_sampling = _mul_up(
                factor,
                _mul_up(projection_support, projection_support),
            )
            center_q[h, direction] = q_center
            radius_q[h, direction] = _add_up(q_numeric, q_sampling)
            for route in range(route_count):
                m_value = exact_m[direction][route] / observed[h]
                m_center, m_numeric = _fraction_center_radius(m_value)
                m_sampling = _mul_up(
                    _mul_up(
                        _mul_up(factor, 2.0),
                        projection_support,
                    ),
                    float(bound[route].item()),
                )
                center_m[h, direction, route] = m_center
                radius_m[h, direction, route] = _add_up(m_numeric, m_sampling)
    census = observed == population
    return V1LiteralJResidualPopulationProjectionCPUReference(
        mjz_center=center_m.contiguous(),
        mjz_radius=radius_m.contiguous(),
        q_center=center_q.contiguous(),
        q_radius=radius_q.contiguous(),
        observed_counts=observed,  # type: ignore[arg-type]
        residual_population_counts=population,
        simultaneous_statement_count=statements,
        per_statement_two_sided_alpha=per_statement,
        alpha_spent=0.0 if census else alpha,
        exact_residual_census=census,
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJResidualPopulationProjectionCUDAExecutionReceipt:
    """Array-free closure of source-bound two-pass residual CUDA execution."""

    cuda_registration_hash: str
    population_projection_registration_hash: str
    suffix_registration_hash: str
    packet_receipt_hash: str
    prefix_freeze_receipt_hash: str
    row_group_plan_hash: str
    row_group_commit_receipt_hash: str
    scorer_provider_proof_hashes: tuple[str, str]
    scorer_proof_closure_hashes: tuple[str, str]
    literal_j_shard_receipt_hashes: tuple[str, str]
    literal_j_content_sha256s: tuple[str, str]
    literal_j_shard_payload_bytes: tuple[int, int]
    literal_j_shard_route_window_counts: tuple[int, int]
    literal_j_stream_source_hash: str
    same_capture_vjp_source_hashes: tuple[str, str]
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    direction_ids: tuple[str, str]
    d_fixed_projection_direction_axis_hash: str
    private_primal_direction_axis_hash: str
    support_bounds_receipt_hash: str
    stratum_ids: tuple[str, str]
    residual_population_counts: tuple[int, int]
    observed_counts: tuple[int, int]
    residual_sealed_population_order_hashes: tuple[str, str]
    observed_prefix_axis_hashes: tuple[str, str]
    observed_source_measurement_axis_hashes: tuple[str, str]
    source_pass_axis_hashes: tuple[str, str]
    mjz_center_fingerprint: str
    mjz_radius_fingerprint: str
    q_center_fingerprint: str
    q_radius_fingerprint: str
    simultaneous_statement_count: int
    familywise_alpha: float
    per_statement_two_sided_alpha: float
    alpha_spent: float
    exact_residual_census: bool
    planning_hbm_forecast_hash: str
    execution_hbm_receipt_hash: str
    immutable_shard_read_pass_count: int
    source_route_window_read_count: int
    source_payload_read_bytes: int
    host_to_device_payload_bytes: int
    source_tile_submission_count: int
    output_receipt_scalar_d2h_count: int
    output_receipt_scalar_d2h_bytes: int
    production_cuda_execution: bool = True
    private_cpu_reference_execution: bool = False
    exact_fixed_time_family: bool = True
    context_by_route_hbm_bank_retained: bool = False
    route_scale_device_to_host_transfer: bool = False
    adaptive_stopping_authorized: bool = False
    rank_or_kkt_face_authority: bool = False
    c_k_o_commit_terminal_or_pareto_authority: bool = False
    algorithm: str = LITERAL_J_RESIDUAL_CUDA_ALGORITHM
    authority_scope: str = LITERAL_J_RESIDUAL_CUDA_AUTHORITY
    schema: str = LITERAL_J_RESIDUAL_CUDA_EXECUTION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _EXECUTION_TOKEN:
            raise PermissionError("residual CUDA execution receipts require issuer")
        for name in (
            "cuda_registration_hash",
            "population_projection_registration_hash",
            "suffix_registration_hash",
            "packet_receipt_hash",
            "prefix_freeze_receipt_hash",
            "row_group_plan_hash",
            "row_group_commit_receipt_hash",
            "literal_j_stream_source_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "d_fixed_projection_direction_axis_hash",
            "private_primal_direction_axis_hash",
            "support_bounds_receipt_hash",
            "mjz_center_fingerprint",
            "mjz_radius_fingerprint",
            "q_center_fingerprint",
            "q_radius_fingerprint",
            "planning_hbm_forecast_hash",
            "execution_hbm_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "scorer_provider_proof_hashes",
            "scorer_proof_closure_hashes",
            "literal_j_shard_receipt_hashes",
            "literal_j_content_sha256s",
            "same_capture_vjp_source_hashes",
            "direction_ids",
            "stratum_ids",
            "residual_sealed_population_order_hashes",
            "observed_prefix_axis_hashes",
            "observed_source_measurement_axis_hashes",
            "source_pass_axis_hashes",
        ):
            object.__setattr__(
                self, name, _identity_pair(getattr(self, name), name=name)
            )
        for name in (
            "literal_j_shard_payload_bytes",
            "literal_j_shard_route_window_counts",
        ):
            object.__setattr__(self, name, _count_pair(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal_generation"),
        )
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route_count")
        )
        for name in ("residual_population_counts", "observed_counts"):
            object.__setattr__(self, name, _count_pair(getattr(self, name), name=name))
        for name in (
            "simultaneous_statement_count",
            "immutable_shard_read_pass_count",
            "source_route_window_read_count",
            "source_payload_read_bytes",
            "host_to_device_payload_bytes",
            "source_tile_submission_count",
            "output_receipt_scalar_d2h_count",
            "output_receipt_scalar_d2h_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        alpha = _finite_probability(self.familywise_alpha, name="familywise_alpha")
        per_statement = _finite_probability(
            self.per_statement_two_sided_alpha,
            name="per_statement_two_sided_alpha",
        )
        if type(self.alpha_spent) is not float or not math.isfinite(self.alpha_spent):
            raise TypeError("alpha_spent must be one finite exact float")
        census = self.observed_counts == self.residual_population_counts
        flags = (
            self.exact_residual_census,
            self.production_cuda_execution,
            self.private_cpu_reference_execution,
            self.exact_fixed_time_family,
            self.context_by_route_hbm_bank_retained,
            self.route_scale_device_to_host_transfer,
            self.adaptive_stopping_authorized,
            self.rank_or_kkt_face_authority,
            self.c_k_o_commit_terminal_or_pareto_authority,
        )
        if any(type(item) is not bool for item in flags):
            raise TypeError("residual CUDA execution flags must be exact booleans")
        if (
            self.simultaneous_statement_count
            != 2 * len(self.direction_ids) * (self.route_count + 1)
            or any(
                observed <= 0 or observed > population
                for observed, population in zip(
                    self.observed_counts,
                    self.residual_population_counts,
                    strict=True,
                )
            )
            or per_statement
            != _divide_probability_down(alpha, self.simultaneous_statement_count)
            or self.exact_residual_census != census
            or (census and self.alpha_spent != 0.0)
            or (not census and self.alpha_spent != alpha)
            or self.immutable_shard_read_pass_count != 2
            or any(value <= 0 for value in self.literal_j_shard_payload_bytes)
            or any(
                value <= 0 for value in self.literal_j_shard_route_window_counts
            )
            or self.source_payload_read_bytes
            != 2 * sum(self.literal_j_shard_payload_bytes)
            or self.source_route_window_read_count
            != 2 * sum(self.literal_j_shard_route_window_counts)
            or self.source_payload_read_bytes != self.host_to_device_payload_bytes
            or self.output_receipt_scalar_d2h_count != 32
            or self.output_receipt_scalar_d2h_bytes
            != self.output_receipt_scalar_d2h_count * _F64_BYTES
            or self.production_cuda_execution is not True
            or self.private_cpu_reference_execution is not False
            or self.exact_fixed_time_family is not True
            or any(flags[4:])
            or self.algorithm != LITERAL_J_RESIDUAL_CUDA_ALGORITHM
            or self.authority_scope != LITERAL_J_RESIDUAL_CUDA_AUTHORITY
            or self.schema != LITERAL_J_RESIDUAL_CUDA_EXECUTION_SCHEMA
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
                "residual CUDA execution crossed its statistical or authority boundary",
            )
        object.__setattr__(self, "familywise_alpha", alpha)
        object.__setattr__(self, "per_statement_two_sided_alpha", per_statement)
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(self._payload(), role="execution-receipt")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "residual CUDA execution receipt mutated",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "residual CUDA execution receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1LiteralJResidualPopulationProjectionCUDAExecution:
    receipt: V1LiteralJResidualPopulationProjectionCUDAExecutionReceipt
    hbm_receipt: V1LiteralJResidualPopulationProjectionCUDAExecutionHBMReceipt
    population_result: V1LiteralJResidualPopulationProjectionResult = field(
        repr=False, compare=False
    )
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _EXECUTION_TOKEN:
            raise PermissionError("residual CUDA executions require issuer")
        if (
            type(self.receipt)
            is not V1LiteralJResidualPopulationProjectionCUDAExecutionReceipt
            or type(self.hbm_receipt)
            is not V1LiteralJResidualPopulationProjectionCUDAExecutionHBMReceipt
            or type(self.population_result)
            is not V1LiteralJResidualPopulationProjectionResult
        ):
            raise TypeError("residual CUDA execution requires exact result types")
        self.receipt.assert_integrity()
        self.hbm_receipt.assert_integrity(self.receipt.execution_hbm_receipt_hash)
        self.population_result.device_handle.assert_live()
        population_receipt = self.population_result.receipt
        if (
            population_receipt.registration_hash
            != self.receipt.population_projection_registration_hash
            or population_receipt.prefix_freeze_receipt_hash
            != self.receipt.prefix_freeze_receipt_hash
            or population_receipt.observed_counts != self.receipt.observed_counts
            or population_receipt.mjz_center_fingerprint
            != self.receipt.mjz_center_fingerprint
            or population_receipt.mjz_radius_fingerprint
            != self.receipt.mjz_radius_fingerprint
            or population_receipt.q_center_fingerprint
            != self.receipt.q_center_fingerprint
            or population_receipt.q_radius_fingerprint
            != self.receipt.q_radius_fingerprint
            or population_receipt.production_cuda_execution is not True
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "execution receipt and canonical residual result diverged",
            )


def _scale_interval_by_uniform_mean(
    *, center_sum: Any, radius_sum: Any, count: int
) -> tuple[Any, Any]:
    exact = Fraction(1, count)
    weight = float(exact)
    weight_error_fraction = abs(exact - Fraction.from_float(weight))
    weight_radius = float(weight_error_fraction)
    if Fraction.from_float(weight_radius) < weight_error_fraction:
        weight_radius = math.nextafter(weight_radius, math.inf)
    center = (center_sum * weight).contiguous()
    weight_magnitude = _add_up(abs(weight), weight_radius)
    radius = _add_up(
        _add_up(
            _mul_up(torch.abs(center_sum), weight_radius),
            _mul_up(radius_sum, weight_magnitude),
        ),
        _add_up(_mul_up(torch.abs(center), 2.0**-52), _MIN_SUBNORMAL),
    ).contiguous()
    return center, radius


def _stage_source_tile(rows: np.ndarray, *, device: Any) -> Any:
    owned = np.array(rows, dtype=np.float64, order="C", copy=True)
    if not bool(np.isfinite(owned).all()):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.NUMERICAL_REFUSAL,
            "immutable literal-J shard contains a nonfinite row",
        )
    return torch.from_numpy(owned).to(
        device=device, dtype=torch.float64, non_blocking=False
    )


def _pass1_projections(
    *, attachment: Any, directions: Any, bound: Any, design: Any
) -> tuple[Any, Any, int, int, int]:
    context_count = len(attachment.receipt.context_ids)
    direction_count, route_count = tuple(int(item) for item in directions.shape)
    center = torch.zeros(
        (context_count, direction_count), device=directions.device, dtype=torch.float64
    )
    absolute_mass = torch.zeros_like(center)
    tile_count = 0
    window_count = 0
    payload_bytes = 0
    for parent_start, parent_stop, parent_rows in attachment.iter_route_windows(
        verify=True
    ):
        window_count += 1
        width = parent_stop - parent_start
        for local_start in range(0, width, design.route_tile_size):
            local_stop = min(width, local_start + design.route_tile_size)
            start = parent_start + local_start
            stop = parent_start + local_stop
            rows = _stage_source_tile(
                parent_rows[:, local_start:local_stop], device=directions.device
            )
            route_bound = bound[start:stop]
            if bool((torch.abs(rows) > route_bound.reshape(1, -1)).any().item()):
                _refuse(
                    V1LiteralJResidualPopulationProjectionCUDARefusalCode.SUPPORT_VIOLATION,
                    "literal-J source tile exceeded its analytic support",
                )
            z = directions[:, start:stop]
            center.add_(rows @ z.T)
            absolute_mass.add_(torch.abs(rows) @ torch.abs(z).T)
            payload_bytes += int(rows.numel()) * _F64_BYTES
            tile_count += 1
    operations = 2 * route_count + 2 * tile_count + 1
    relative = _gamma(operations)
    recovery = _div_up(relative, 1.0 - relative)
    radius = _add_up(
        _mul_up(absolute_mass, recovery),
        _mul_up(float(operations), _MIN_SUBNORMAL),
    ).contiguous()
    return center, radius, window_count, tile_count, payload_bytes


def _pass2_mjz(
    *,
    attachment: Any,
    projection: Any,
    projection_radius: Any,
    literal_bound: Any,
    projection_bound: Any,
    population_count: int,
    alpha: float,
    design: Any,
) -> tuple[Any, Any, int, int, int]:
    context_count, direction_count = tuple(int(item) for item in projection.shape)
    route_count = int(literal_bound.numel())
    center = torch.empty(
        (direction_count, route_count), device=projection.device, dtype=torch.float64
    )
    radius = torch.empty_like(center)
    factor = _serfling_factor(
        observed=context_count, population=population_count, alpha=alpha
    )
    tile_count = 0
    window_count = 0
    payload_bytes = 0
    for parent_start, parent_stop, parent_rows in attachment.iter_route_windows(
        verify=True
    ):
        window_count += 1
        width = parent_stop - parent_start
        for local_start in range(0, width, design.route_tile_size):
            local_stop = min(width, local_start + design.route_tile_size)
            start = parent_start + local_start
            stop = parent_start + local_stop
            rows = _stage_source_tile(
                parent_rows[:, local_start:local_stop], device=projection.device
            )
            route_bound = literal_bound[start:stop]
            if bool((torch.abs(rows) > route_bound.reshape(1, -1)).any().item()):
                _refuse(
                    V1LiteralJResidualPopulationProjectionCUDARefusalCode.SUPPORT_VIOLATION,
                    "literal-J replay tile exceeded its analytic support",
                )
            center_sum = projection.T @ rows
            abs_rows = torch.abs(rows)
            round_mass = torch.abs(projection).T @ abs_rows
            input_uncertainty = projection_radius.T @ abs_rows
            operations = 2 * context_count + 1
            relative = _gamma(operations)
            recovery = _div_up(relative, 1.0 - relative)
            input_uncertainty_upper = _add_up(
                input_uncertainty,
                _mul_up(input_uncertainty, recovery),
            )
            sum_radius = _add_up(
                _add_up(
                    _mul_up(round_mass, recovery),
                    input_uncertainty_upper,
                ),
                _mul_up(float(operations), _MIN_SUBNORMAL),
            )
            mean_center, numeric_radius = _scale_interval_by_uniform_mean(
                center_sum=center_sum,
                radius_sum=sum_radius,
                count=context_count,
            )
            sampling = _mul_up(
                _mul_up(
                    _mul_up(2.0, factor),
                    projection_bound.reshape(-1, 1),
                ),
                route_bound.reshape(1, -1),
            )
            center[:, start:stop] = mean_center
            radius[:, start:stop] = _add_up(numeric_radius, sampling)
            payload_bytes += int(rows.numel()) * _F64_BYTES
            tile_count += 1
    return (
        center.contiguous(),
        radius.contiguous(),
        window_count,
        tile_count,
        payload_bytes,
    )


def _quadratic_interval(
    *,
    projection: Any,
    projection_radius: Any,
    projection_bound: Any,
    population_count: int,
    alpha: float,
) -> tuple[Any, Any]:
    context_count = int(projection.shape[0])
    center_sum = torch.sum(projection * projection, dim=0)
    round_mass = torch.sum(torch.abs(projection * projection), dim=0)
    input_terms = _add_up(
        _mul_up(_mul_up(2.0, torch.abs(projection)), projection_radius),
        _mul_up(projection_radius, projection_radius),
    )
    input_uncertainty = torch.sum(input_terms, dim=0)
    operations = 2 * context_count + 1
    relative = _gamma(operations)
    recovery = _div_up(relative, 1.0 - relative)
    input_uncertainty_upper = _add_up(
        input_uncertainty,
        _mul_up(input_uncertainty, recovery),
    )
    sum_radius = _add_up(
        _add_up(
            _mul_up(round_mass, recovery),
            input_uncertainty_upper,
        ),
        _mul_up(float(operations), _MIN_SUBNORMAL),
    )
    center, numeric = _scale_interval_by_uniform_mean(
        center_sum=center_sum,
        radius_sum=sum_radius,
        count=context_count,
    )
    factor = _serfling_factor(
        observed=context_count, population=population_count, alpha=alpha
    )
    sampling = _mul_up(_mul_up(factor, projection_bound), projection_bound)
    return center.contiguous(), _add_up(numeric, sampling).contiguous()


@dataclass(frozen=True, slots=True)
class _V1LiteralJResidualTwoPassDeviceNumerics:
    mjz_center: Any = field(repr=False, compare=False)
    mjz_radius: Any = field(repr=False, compare=False)
    q_center: Any = field(repr=False, compare=False)
    q_radius: Any = field(repr=False, compare=False)
    source_route_window_read_count: int
    source_payload_read_bytes: int
    source_tile_submission_count: int


def _validate_execution_source_closure(
    *,
    cuda_registration: V1LiteralJResidualPopulationProjectionCUDARegistration,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    row_group_commit: V1SampledDualChartRowGroupCommit,
    packet_proof_closures: Sequence[V1SampledDualChartRowGroupPacketProofClosure],
    design: V1LiteralJResidualPopulationProjectionCUDADesign,
    support_bounds: V1LiteralJResidualDirectionSupportBoundsHandle,
) -> tuple[
    tuple[V1SampledDualChartRowShardAttachment, V1SampledDualChartRowShardAttachment],
    tuple[V1SampledDualChartRowGroupPacketProofClosure, V1SampledDualChartRowGroupPacketProofClosure],
    tuple[str, str],
    tuple[str, str],
    tuple[str, str],
]:
    """Close preregistration, scorer proofs, and immutable literal-J shards."""

    exact_types = (
        (cuda_registration, V1LiteralJResidualPopulationProjectionCUDARegistration),
        (
            population_registration,
            V1LiteralJDFixedResidualPopulationProjectionRegistration,
        ),
        (suffix_registration, V1SampledVariableShellDW2SuffixRegistration),
        (packet, V1SampledVariableShellDW2SuffixPacket),
        (prefix_freeze, V1SampledVariableShellDW2SuffixPrefixFreezeReceipt),
        (row_group_plan, V1SampledDualChartRowGroupPlan),
        (row_group_commit, V1SampledDualChartRowGroupCommit),
        (design, V1LiteralJResidualPopulationProjectionCUDADesign),
        (support_bounds, V1LiteralJResidualDirectionSupportBoundsHandle),
    )
    if any(type(value) is not expected for value, expected in exact_types):
        raise TypeError("residual CUDA execution received an inexact owner type")
    closures = tuple(packet_proof_closures)
    if len(closures) != 2 or any(
        type(item) is not V1SampledDualChartRowGroupPacketProofClosure
        for item in closures
    ):
        raise TypeError("residual CUDA execution requires two exact proof closures")
    cuda_registration.assert_integrity()
    population_registration.assert_integrity()
    suffix_registration.assert_integrity()
    packet.assert_integrity()
    prefix_freeze.assert_integrity()
    row_group_plan.assert_integrity()
    row_group_commit.receipt.assert_integrity()
    design.assert_integrity()
    support_bounds.assert_live()
    for closure in closures:
        closure.assert_integrity()
    commit_receipt = row_group_commit.receipt
    manifest = row_group_commit.attachment.manifest
    try:
        row_group_commit.attachment.verify_live_witnesses()
    except Exception as error:
        raise V1LiteralJResidualPopulationProjectionCUDARefusal(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.MUTATED_CUSTODY,
            "published row-group shard custody is stale or mutated",
        ) from error
    packet_plans = row_group_plan.packet_plans
    if len(packet_plans) != 2 or len(manifest.looks) != 2:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.ORDER_MISMATCH,
            "residual execution requires exactly target then complement packets",
        )
    if (
        cuda_registration.population_projection_registration_hash
        != population_registration.registration_hash
        or cuda_registration.suffix_registration_hash
        != suffix_registration.receipt_hash
        or cuda_registration.packet_receipt_hash != packet.receipt_hash
        or cuda_registration.prefix_freeze_receipt_hash != prefix_freeze.receipt_hash
        or cuda_registration.row_group_plan_hash != row_group_plan.group_plan_hash
        or cuda_registration.design_hash != design.design_hash
        or cuda_registration.support_bounds_receipt_hash
        != support_bounds.receipt.receipt_hash
        or commit_receipt.group_plan_hash != row_group_plan.group_plan_hash
        or commit_receipt.packet_plan_hashes
        != tuple(item.packet_plan_hash for item in packet_plans)
        or commit_receipt.packet_count != 2
        or commit_receipt.role_shard_count != 4
        or commit_receipt.proof_closure_hashes
        != tuple(item.closure_hash for item in closures)
        or commit_receipt.provider_proof_hashes
        != tuple(item.provider_proof_hash for item in closures)
        or tuple(item.packet_ordinal for item in closures) != (0, 1)
        or tuple(item.packet_plan_hash for item in closures)
        != cuda_registration.scorer_packet_plan_hashes
        or tuple(item.provider_plan_identity_hash for item in closures)
        != cuda_registration.scorer_provider_plan_hashes
        or tuple(item.closure_hash for item in closures)
        != commit_receipt.proof_closure_hashes
        or manifest.manifest_hash != commit_receipt.manifest_hash
        or manifest.registered_d_prefix_hash
        != row_group_plan.registered_d_prefix_hash
        or manifest.chart_identity_hash != cuda_registration.chart_identity_hash
        or manifest.current_mask_hash != cuda_registration.current_mask_hash
        or manifest.proposal_generation != cuda_registration.proposal_generation
        or manifest.route_axis_hash != cuda_registration.route_axis_hash
        or manifest.route_id_axis_hash != cuda_registration.route_id_axis_hash
        or manifest.route_count != cuda_registration.route_count
        or manifest.literal_j_stream_identity_hash
        != cuda_registration.literal_j_stream_source_hash
        or prefix_freeze.completed_packet_hashes != (packet.receipt_hash,)
        or prefix_freeze.consumed_dw2_stratum_counts
        != cuda_registration.registered_observed_counts
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "residual preregistration, scorer proofs, published group, or frozen prefix diverged",
        )
    attachments = tuple(row_group_commit.attachment.literal_j_attachments)
    if len(attachments) != 2 or any(
        type(item) is not V1SampledDualChartRowShardAttachment for item in attachments
    ):
        raise TypeError("residual execution requires two exact literal-J attachments")
    measurement_hashes: list[str] = []
    pass_hashes: list[str] = []
    same_capture_hashes: list[str] = []
    for index, (plan, closure, look, attachment) in enumerate(
        zip(packet_plans, closures, manifest.looks, attachments, strict=True)
    ):
        shard = attachment.receipt
        expected_stratum = cuda_registration.stratum_ids[index]
        expected_context_axis_hash = cuda_registration.packet_context_axis_hashes[index]
        expected_draw_axis_hash = cuda_registration.packet_draw_axis_hashes[index]
        if (
            closure.group_plan_hash != row_group_plan.group_plan_hash
            or closure.packet_plan_hash != plan.packet_plan_hash
            or closure.packet_ordinal != index
            or closure.provider_plan_identity_hash
            != plan.provider_plan_identity_hash
            or closure.reducer_plan_identity_hashes
            != plan.reducer_plan_identity_hashes
            or closure.reducer_proof_hashes
            != commit_receipt.reducer_proof_hashes[index]
            or look.literal_j_shard.receipt_hash != shard.receipt_hash
            or shard.role is not V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN
            or shard.context_ids != plan.context_ids
            or shard.draw_ids != plan.draw_ids
            or shard.stratum_labels != plan.stratum_labels
            or plan.stratum_labels != (expected_stratum,) * len(plan.context_ids)
            or shard.source_capture_identity_hash
            != plan.source_capture_identity_hash
            or look.corrected_a_shard.source_capture_identity_hash
            != shard.source_capture_identity_hash
            or shard.source_capture_identity_hash
            != cuda_registration.scorer_source_capture_identity_hashes[index]
            or shard.stream_identity_hash
            != cuda_registration.literal_j_stream_source_hash
            or shard.chart_identity_hash != cuda_registration.chart_identity_hash
            or shard.current_mask_hash != cuda_registration.current_mask_hash
            or shard.proposal_generation != cuda_registration.proposal_generation
            or shard.route_axis_hash != cuda_registration.route_axis_hash
            or shard.route_id_axis_hash != cuda_registration.route_id_axis_hash
            or shard.route_count != cuda_registration.route_count
            or _axis_hash("target-contexts" if index == 0 else "complement-contexts", shard.context_ids)
            != expected_context_axis_hash
            or _axis_hash("target-draws" if index == 0 else "complement-draws", shard.draw_ids)
            != expected_draw_axis_hash
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
                "literal-J shard or provider closure left its exact preregistered scorer packet",
            )
        measurement_hashes.append(
            _payload_hash(
                {
                    "schema": "residual_literal_j_source_measurement_axis_v1",
                    "cuda_registration_hash": cuda_registration.registration_hash,
                    "stratum_id": expected_stratum,
                    "literal_j_shard_receipt_hash": shard.receipt_hash,
                    "literal_j_content_sha256": shard.content_sha256,
                    "context_ids": shard.context_ids,
                    "draw_ids": shard.draw_ids,
                    "source_capture_identity_hash": shard.source_capture_identity_hash,
                    "provider_proof_closure_hash": closure.closure_hash,
                },
                role="source-measurement-axis",
            )
        )
        pass_hashes.append(
            _payload_hash(
                {
                    "schema": "immutable_literal_j_two_pass_axis_v1",
                    "literal_j_shard_receipt_hash": shard.receipt_hash,
                    "content_sha256": shard.content_sha256,
                    "registered_route_windows": shard.route_windows,
                    "route_tile_size": design.route_tile_size,
                    "passes": ("p_equals_Jz", "mean_pJ"),
                    "q_reduced_from_resident_p": True,
                },
                role="source-pass-axis",
            )
        )
        same_capture_hashes.append(
            _payload_hash(
                {
                    "schema": "residual_same_capture_vjp_source_closure_v1",
                    "source_capture_identity_hash": shard.source_capture_identity_hash,
                    "provider_plan_identity_hash": plan.provider_plan_identity_hash,
                    "provider_proof_hash": closure.provider_proof_hash,
                    "provider_proof_closure_hash": closure.closure_hash,
                    "corrected_a_shard_receipt_hash": (
                        look.corrected_a_shard.receipt_hash
                    ),
                    "literal_j_shard_receipt_hash": shard.receipt_hash,
                    "both_roles_share_source_capture": (
                        look.corrected_a_shard.source_capture_identity_hash
                        == shard.source_capture_identity_hash
                    ),
                },
                role="same-capture-vjp-source",
            )
        )
    return (
        attachments,  # type: ignore[return-value]
        closures,  # type: ignore[return-value]
        tuple(measurement_hashes),  # type: ignore[return-value]
        tuple(pass_hashes),  # type: ignore[return-value]
        tuple(same_capture_hashes),  # type: ignore[return-value]
    )


def _compute_v1_literal_j_residual_two_pass_device_numerics(
    *,
    attachments: Sequence[V1SampledDualChartRowShardAttachment],
    directions: Any,
    literal_j_abs_bound: Any,
    projection_abs_bounds: Any,
    residual_population_counts: tuple[int, int],
    familywise_alpha: float,
    design: V1LiteralJResidualPopulationProjectionCUDADesign,
    production_cuda_required: bool,
    failure_injection: str | None = None,
) -> _V1LiteralJResidualTwoPassDeviceNumerics:
    """Execute the exact two shard passes; CPU is a tiny private oracle only."""

    if type(design) is not V1LiteralJResidualPopulationProjectionCUDADesign:
        raise TypeError("two-pass device numerics require the exact design")
    if type(production_cuda_required) is not bool:
        raise TypeError("production_cuda_required must be one exact boolean")
    if failure_injection not in (
        None,
        "after_target_pass1",
        "after_target_pass2",
    ):
        raise ValueError("unknown private residual CUDA failure injection")
    shards = tuple(attachments)
    if len(shards) != 2 or any(
        type(item) is not V1SampledDualChartRowShardAttachment for item in shards
    ):
        raise TypeError("two-pass device numerics require two exact shard attachments")
    if torch is None or type(directions) is not torch.Tensor or directions.ndim != 2:
        raise TypeError("two-pass device numerics require one exact direction tensor")
    direction_count, route_count = tuple(int(item) for item in directions.shape)
    _positive_int(direction_count, name="direction_count")
    _positive_int(route_count, name="route_count")
    z = _require_fp64_tensor(
        directions,
        name="directions",
        shape=(direction_count, route_count),
    )
    if production_cuda_required and z.device.type != "cuda":
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
            "production residual numerics require CUDA",
        )
    if not production_cuda_required and (
        z.device.type != "cpu" or route_count > _CPU_REFERENCE_ROUTE_LIMIT
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
            "private two-pass CPU oracle is limited to 4096 routes",
        )
    j_bound = _require_fp64_tensor(
        literal_j_abs_bound,
        name="literal_j_abs_bound",
        shape=(route_count,),
        device=z.device,
        nonnegative=True,
    )
    p_bound = _require_fp64_tensor(
        projection_abs_bounds,
        name="projection_abs_bounds",
        shape=(direction_count,),
        device=z.device,
        nonnegative=True,
    )
    population = _count_pair(
        residual_population_counts, name="residual_population_counts"
    )
    observed = tuple(len(item.receipt.context_ids) for item in shards)
    if any(observed[h] <= 0 or observed[h] > population[h] for h in (0, 1)):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.INVALID_DESIGN,
            "literal-J shard counts are not nonempty residual prefixes",
        )
    statements = 2 * direction_count * (route_count + 1)
    per_statement = _divide_probability_down(
        _finite_probability(familywise_alpha, name="familywise_alpha"), statements
    )
    all_center_m = torch.empty(
        (2, direction_count, route_count), device=z.device, dtype=torch.float64
    )
    all_radius_m = torch.empty_like(all_center_m)
    all_center_q = torch.empty(
        (2, direction_count), device=z.device, dtype=torch.float64
    )
    all_radius_q = torch.empty_like(all_center_q)
    window_reads = tile_submissions = payload_bytes = 0
    for h, attachment in enumerate(shards):
        projection, projection_radius, windows, tiles, payload = _pass1_projections(
            attachment=attachment,
            directions=z,
            bound=j_bound,
            design=design,
        )
        window_reads += windows
        tile_submissions += tiles
        payload_bytes += payload
        projection_upper = _add_up(torch.abs(projection), projection_radius)
        if bool(
            (projection_upper > p_bound.reshape(1, -1)).any().item()
        ):
            _refuse(
                V1LiteralJResidualPopulationProjectionCUDARefusalCode.SUPPORT_VIOLATION,
                "literal-J projection interval exceeded its pre-outcome direct support",
            )
        if h == 0 and failure_injection == "after_target_pass1":
            raise RuntimeError("injected failure after target literal-J pass 1")
        q_c, q_r = _quadratic_interval(
            projection=projection,
            projection_radius=projection_radius,
            projection_bound=p_bound,
            population_count=population[h],
            alpha=per_statement,
        )
        m_c, m_r, windows, tiles, payload = _pass2_mjz(
            attachment=attachment,
            projection=projection,
            projection_radius=projection_radius,
            literal_bound=j_bound,
            projection_bound=p_bound,
            population_count=population[h],
            alpha=per_statement,
            design=design,
        )
        window_reads += windows
        tile_submissions += tiles
        payload_bytes += payload
        if h == 0 and failure_injection == "after_target_pass2":
            raise RuntimeError("injected failure after target literal-J pass 2")
        all_center_m[h].copy_(m_c)
        all_radius_m[h].copy_(m_r)
        all_center_q[h].copy_(q_c)
        all_radius_q[h].copy_(q_r)
    return _V1LiteralJResidualTwoPassDeviceNumerics(
        mjz_center=all_center_m.contiguous(),
        mjz_radius=all_radius_m.contiguous(),
        q_center=all_center_q.contiguous(),
        q_radius=all_radius_q.contiguous(),
        source_route_window_read_count=window_reads,
        source_payload_read_bytes=payload_bytes,
        source_tile_submission_count=tile_submissions,
    )


def execute_v1_literal_j_residual_population_projection_cuda(
    *,
    cuda_registration: V1LiteralJResidualPopulationProjectionCUDARegistration,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    packet: V1SampledVariableShellDW2SuffixPacket,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    row_group_plan: V1SampledDualChartRowGroupPlan,
    row_group_commit: V1SampledDualChartRowGroupCommit,
    packet_proof_closures: Sequence[V1SampledDualChartRowGroupPacketProofClosure],
    analytic_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    support_bounds: V1LiteralJResidualDirectionSupportBoundsHandle,
    design: V1LiteralJResidualPopulationProjectionCUDADesign,
    planning_hbm_forecast: V1LiteralJResidualPopulationProjectionCUDAHBMForecast,
) -> V1LiteralJResidualPopulationProjectionCUDAExecution:
    """Execute one fixed paired residual literal-J prefix on CUDA.

    This is a fixed-time numerical producer only.  It neither selects another
    D prefix nor decides FD rank, a KKT face, C/K/O admission, mask commit, or
    Pareto status.
    """

    (
        attachments,
        closures,
        measurement_hashes,
        pass_hashes,
        same_capture_hashes,
    ) = _validate_execution_source_closure(
        cuda_registration=cuda_registration,
        population_registration=population_registration,
        suffix_registration=suffix_registration,
        packet=packet,
        prefix_freeze=prefix_freeze,
        row_group_plan=row_group_plan,
        row_group_commit=row_group_commit,
        packet_proof_closures=packet_proof_closures,
        design=design,
        support_bounds=support_bounds,
    )
    if (
        type(analytic_support_source)
        is not V1LiteralJResidualAnalyticSupportSourceCUDAResult
        or type(planning_hbm_forecast)
        is not V1LiteralJResidualPopulationProjectionCUDAHBMForecast
    ):
        raise TypeError(
            "residual CUDA execution requires exact analytic support and HBM forecast types"
        )
    if population_registration.d_fixed_shard_replay_receipt_hash is None:
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
            "production residual execution requires a source-bound D_fixed population registration",
        )
    if (
        planning_hbm_forecast.observed_counts
        != cuda_registration.registered_observed_counts
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "planning HBM forecast was measured for a different fixed prefix",
        )
    if torch is None or not torch.cuda.is_available():
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.DEVICE_REQUIRED,
            "residual production execution requires a real CUDA device",
        )
    if (
        design.deterministic_algorithms_required
        and not torch.are_deterministic_algorithms_enabled()
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.DETERMINISM_DISABLED,
            "residual production execution requires deterministic CUDA algorithms",
        )
    analytic_support_source.source_handle.assert_live()
    source_receipt = analytic_support_source.receipt
    support_receipt = support_bounds.receipt
    if (
        source_receipt.model_identity_hash
        != population_registration.model_identity_hash
        or source_receipt.chart_identity_hash
        != population_registration.chart_identity_hash
        or source_receipt.current_mask_hash
        != population_registration.current_mask_hash
        or source_receipt.proposal_generation
        != population_registration.proposal_generation
        or source_receipt.route_axis_hash
        != population_registration.route_axis_hash
        or source_receipt.route_id_axis_hash
        != population_registration.route_id_axis_hash
        or source_receipt.route_count != population_registration.route_count
        or source_receipt.direction_ids != population_registration.direction_ids
        or source_receipt.private_primal_direction_axis_hash
        != population_registration.private_primal_direction_axis_hash
        or support_receipt.production_cuda_execution is not True
        or support_receipt.private_cpu_reference_execution is not False
        or support_receipt.population_projection_registration_hash
        != population_registration.registration_hash
        or support_receipt.row_group_plan_hash != row_group_plan.group_plan_hash
        or support_receipt.private_primal_direction_axis_hash
        != source_receipt.private_primal_direction_axis_hash
        or support_receipt.literal_j_abs_bound_source_hash
        != source_receipt.literal_j_abs_bound_source_hash
        or support_receipt.direct_projection_abs_bound_derivation_receipt_hash
        != source_receipt.direct_projection_abs_bound_derivation_receipt_hash
    ):
        _refuse(
            V1LiteralJResidualPopulationProjectionCUDARefusalCode.STALE_IDENTITY,
            "analytic support source, derived bounds, row group, and population chart diverged",
        )
    with analytic_support_source.source_handle.borrow() as source_lease:
        with support_bounds.borrow() as support_lease:
            direction_lease = (
                analytic_support_source.source_handle.retained_direction_lease()
            )
            directions = direction_lease.directions_for_literal_j_projection()
            device = directions.device
            if (
                device.type != "cuda"
                or source_lease.literal_j_abs_bound.device != device
                or source_lease.direct_projection_abs_bounds.device != device
                or support_lease.literal_j_abs_bound.device != device
                or support_lease.projection_abs_bounds.device != device
                or not bool(
                    torch.equal(
                        source_lease.literal_j_abs_bound,
                        support_lease.literal_j_abs_bound,
                    )
                )
            ):
                _refuse(
                    V1LiteralJResidualPopulationProjectionCUDARefusalCode.MUTATED_CUSTODY,
                    "directions and analytic/derived supports left one live CUDA custody",
                )
            hbm_receipt = _measure_v1_literal_j_residual_cuda_execution_hbm(
                design=design,
                planning_forecast=planning_hbm_forecast,
                analytic_support_source=analytic_support_source,
                derived_support_bounds=support_bounds,
            )
            numerics = _compute_v1_literal_j_residual_two_pass_device_numerics(
                attachments=attachments,
                directions=directions,
                literal_j_abs_bound=support_lease.literal_j_abs_bound,
                projection_abs_bounds=support_lease.projection_abs_bounds,
                residual_population_counts=(
                    population_registration.residual_stratum_counts
                ),
                familywise_alpha=population_registration.familywise_alpha,
                design=design,
                production_cuda_required=True,
            )
    observed = cuda_registration.registered_observed_counts
    residual = population_registration.residual_stratum_counts
    observed_prefix_hashes = (
        _canonical_residual_observed_prefix_hash(
            registration_hash=population_registration.registration_hash,
            stratum_id=population_registration.stratum_ids[0],
            observed=observed[0],
            source_hash=prefix_freeze.dw2_target_prefix_axis_hash,
        ),
        _canonical_residual_observed_prefix_hash(
            registration_hash=population_registration.registration_hash,
            stratum_id=population_registration.stratum_ids[1],
            observed=observed[1],
            source_hash=prefix_freeze.dw2_complement_prefix_axis_hash,
        ),
    )
    output_fingerprints = (
        fingerprint_v1_literal_j_residual_cuda_tensor(
            numerics.mjz_center, role="residual_mjz_center"
        ),
        fingerprint_v1_literal_j_residual_cuda_tensor(
            numerics.mjz_radius, role="residual_mjz_radius"
        ),
        fingerprint_v1_literal_j_residual_cuda_tensor(
            numerics.q_center, role="residual_q_center"
        ),
        fingerprint_v1_literal_j_residual_cuda_tensor(
            numerics.q_radius, role="residual_q_radius"
        ),
    )
    shards = tuple(item.receipt for item in attachments)
    exact_census = observed == residual
    receipt = V1LiteralJResidualPopulationProjectionCUDAExecutionReceipt(
        cuda_registration_hash=cuda_registration.registration_hash,
        population_projection_registration_hash=(
            population_registration.registration_hash
        ),
        suffix_registration_hash=suffix_registration.receipt_hash,
        packet_receipt_hash=packet.receipt_hash,
        prefix_freeze_receipt_hash=prefix_freeze.receipt_hash,
        row_group_plan_hash=row_group_plan.group_plan_hash,
        row_group_commit_receipt_hash=row_group_commit.receipt.receipt_hash,
        scorer_provider_proof_hashes=tuple(
            item.provider_proof_hash for item in closures
        ),  # type: ignore[arg-type]
        scorer_proof_closure_hashes=tuple(
            item.closure_hash for item in closures
        ),  # type: ignore[arg-type]
        literal_j_shard_receipt_hashes=tuple(
            item.receipt_hash for item in shards
        ),  # type: ignore[arg-type]
        literal_j_content_sha256s=tuple(
            item.content_sha256 for item in shards
        ),  # type: ignore[arg-type]
        literal_j_shard_payload_bytes=tuple(
            item.exact_payload_file_bytes for item in shards
        ),  # type: ignore[arg-type]
        literal_j_shard_route_window_counts=tuple(
            len(item.route_windows) for item in shards
        ),  # type: ignore[arg-type]
        literal_j_stream_source_hash=(
            population_registration.residual_literal_j_stream_source_hash
        ),
        same_capture_vjp_source_hashes=same_capture_hashes,
        model_identity_hash=population_registration.model_identity_hash,
        chart_identity_hash=population_registration.chart_identity_hash,
        current_mask_hash=population_registration.current_mask_hash,
        proposal_generation=population_registration.proposal_generation,
        route_axis_hash=population_registration.route_axis_hash,
        route_id_axis_hash=population_registration.route_id_axis_hash,
        route_count=population_registration.route_count,
        direction_ids=population_registration.direction_ids,  # type: ignore[arg-type]
        d_fixed_projection_direction_axis_hash=(
            population_registration.d_fixed_projection_direction_axis_hash
        ),
        private_primal_direction_axis_hash=(
            population_registration.private_primal_direction_axis_hash
        ),
        support_bounds_receipt_hash=support_receipt.receipt_hash,
        stratum_ids=population_registration.stratum_ids,
        residual_population_counts=residual,
        observed_counts=observed,
        residual_sealed_population_order_hashes=(
            population_registration.residual_sealed_population_order_hashes
        ),
        observed_prefix_axis_hashes=observed_prefix_hashes,
        observed_source_measurement_axis_hashes=measurement_hashes,
        source_pass_axis_hashes=pass_hashes,
        mjz_center_fingerprint=output_fingerprints[0],
        mjz_radius_fingerprint=output_fingerprints[1],
        q_center_fingerprint=output_fingerprints[2],
        q_radius_fingerprint=output_fingerprints[3],
        simultaneous_statement_count=(
            cuda_registration.simultaneous_statement_count
        ),
        familywise_alpha=population_registration.familywise_alpha,
        per_statement_two_sided_alpha=(
            cuda_registration.per_statement_two_sided_alpha
        ),
        alpha_spent=(
            0.0 if exact_census else population_registration.familywise_alpha
        ),
        exact_residual_census=exact_census,
        planning_hbm_forecast_hash=planning_hbm_forecast.forecast_hash,
        execution_hbm_receipt_hash=hbm_receipt.receipt_hash,
        immutable_shard_read_pass_count=2,
        source_route_window_read_count=(
            numerics.source_route_window_read_count
        ),
        source_payload_read_bytes=numerics.source_payload_read_bytes,
        host_to_device_payload_bytes=numerics.source_payload_read_bytes,
        source_tile_submission_count=numerics.source_tile_submission_count,
        output_receipt_scalar_d2h_count=32,
        output_receipt_scalar_d2h_bytes=32 * _F64_BYTES,
        _issuance_token=_EXECUTION_TOKEN,
    )
    population_result = (
        _issue_v1_literal_j_residual_population_projection_from_cuda_producer(
            producer_capability=_RESIDUAL_CUDA_PRODUCER_CAPABILITY,
            registration=population_registration,
            prefix_freeze=prefix_freeze,
            cuda_registration=cuda_registration,
            cuda_execution_receipt=receipt,
            mjz_center=numerics.mjz_center,
            mjz_radius=numerics.mjz_radius,
            q_center=numerics.q_center,
            q_radius=numerics.q_radius,
        )
    )
    return V1LiteralJResidualPopulationProjectionCUDAExecution(
        receipt=receipt,
        hbm_receipt=hbm_receipt,
        population_result=population_result,
        _issuance_token=_EXECUTION_TOKEN,
    )
