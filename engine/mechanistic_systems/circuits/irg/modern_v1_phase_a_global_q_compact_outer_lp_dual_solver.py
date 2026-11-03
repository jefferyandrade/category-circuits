"""Typed compact dual solver for the phase-A global-Q outer LP.

This module is the narrow source-owned seam between a frozen support family and
the pre-outcome dual-proposal custody object.  It endogenously chooses the
nonnegative support-plane and block-capacity multipliers for the structural
outer LP

    min_{lambda,beta >= 0}
        lambda · rhs + beta · cap
        + sum_j max(0, resource_j - lambda · c_j - beta_{block(j)})

using the registered lower-endpoint support coefficients.  Equivalently, it
certifies the fractional hard-face primal

    max_x resource · x
        s.t. C x <= rhs, B x <= cap, 0 <= x <= 1.

The executable surface has two deliberately separate authorities: a bounded
small-domain CPU/HiGHS reference and a production CUDA source/solver path.  The
CUDA path consumes the one-pass support-plane stream into an ephemeral,
HBM-forecasted lower/upper coefficient bank, solves the registered fractional
outer LP with deterministic FP64 matrix-free PDHG, and exposes only compact
multiplier/primal-dual receipts.  It must not fall back to a route-scale host
solve, route-scale D2H transfer, or durable coefficient retention.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import hashlib
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_phase_a_global_q_outer_lp_dual_proposal import (
    V1PhaseAGlobalQOuterLPDualProposalResult,
    issue_v1_phase_a_global_q_outer_lp_dual_proposal,
)
from .modern_v1_phase_a_global_q_support_family import (
    V1PhaseAGlobalQD1SupportFamilyReceipt,
    V1PhaseAGlobalQD1SupportFamilyResult,
    V1PhaseAGlobalQSupportFamilyHandle,
    V1PhaseAGlobalQSupportFamilyReceipt,
    V1PhaseAGlobalQSupportFamilyResult,
)
from .modern_v1_phase_a_global_q_support_plane_scan import (
    V1PhaseAGlobalQSupportPlaneScanHandle,
    V1PhaseAGlobalQSupportPlaneScanReceipt,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTProblem
from .modern_v1_sampled_v7_action_pair_alias_source_join import (
    V1SampledV7ActionPairAliasSourceJoinReceipt,
)
from .modern_v1_sampled_v7_phase_a_robust_shell_decision import (
    V1SampledV7PhaseARobustShellDecisionReceipt,
)
from mechanistic_systems.contracts.hard_mask_actions import HardMask

try:  # pragma: no cover - runtime-image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - dependency presence is environment-specific.
    from scipy.optimize import linprog
except ImportError:  # pragma: no cover
    linprog = None  # type: ignore[assignment]


PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_SOURCE_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_q_compact_outer_lp_dual_source_v1"
)
PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_SOLVE_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_q_compact_outer_lp_dual_solve_v2"
)
PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_READINESS_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_q_compact_outer_lp_dual_cuda_readiness_v1"
)
PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_SOURCE_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_q_compact_outer_lp_dual_cuda_source_v2"
)
PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_HBM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_phase_a_global_q_compact_outer_lp_dual_cuda_hbm_forecast_v3"
)
PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_ALGORITHM: Final = (
    "lower_endpoint_support_fractional_hard_face_lp_dual_small_domain_highs_v1"
)
PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_ALGORITHM: Final = (
    "lower_endpoint_support_fractional_hard_face_lp_deterministic_fp64_cuda_"
    "primal_dual_gamma_enclosed_ephemeral_bank_v2"
)
PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_SOURCE_ALGORITHM: Final = (
    "support_plane_scan_to_ephemeral_cuda_lower_upper_endpoint_bank_v2"
)
PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_AUTHORITY: Final = (
    "typed_compact_outer_lp_multiplier_source_only_no_route_scale_hard_face_"
    "D_C_K_O_commit_runtime_terminal_or_pareto_authority_v1"
)
PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_READINESS_ALGORITHM: Final = (
    "production_cuda_compact_outer_lp_source_and_solver_landed_downstream_D_"
    "stability_still_external_v1"
)

_PREFIX: Final = "partition-free-v1-phase-a-global-q-compact-outer-lp-dual-"
_SOURCE_TOKEN: Final = object()
_CUDA_SOURCE_TOKEN: Final = object()
_RECEIPT_TOKEN: Final = object()
_HANDLE_TOKEN: Final = object()
_READINESS_TOKEN: Final = object()
_CUDA_FORECAST_TOKEN: Final = object()
_F64_BYTES: Final = 8
_I64_BYTES: Final = 8
_BOOL_BYTES: Final = 1
_CUDA_DEFAULT_HBM_CEILING_FRACTION: Final = 0.85
_CUDA_PRIMAL_DUAL_ROUTE_VECTOR_COUNT: Final = 8
_CUDA_REDUCTION_TILE_ELEMENTS: Final = 262_144
_MIN_SUBNORMAL: Final = float.fromhex("0x0.0000000000001p-1022")
_LEGACY_PROJECTION_SUPPORT_SOURCE_MODE: Final = "legacy_projection_v1"
_FULL_POPULATION_D1_SUPPORT_SOURCE_MODE: Final = "full_population_d1_v1"


class V1PhaseAGlobalQCompactOuterLPDualExecutionMode(str, Enum):
    SMALL_DOMAIN_CPU_REFERENCE = "small_domain_cpu_reference"
    CUDA_PRODUCTION = "cuda_production"


class V1PhaseAGlobalQCompactOuterLPDualRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    SOURCE_MISMATCH = "source_mismatch"
    DEVICE_REFUSAL = "device_refusal"
    ROUTE_LIMIT_EXCEEDED = "route_limit_exceeded"
    SCIPY_REQUIRED = "scipy_required"
    CUDA_SOLVER_NOT_LANDED = "cuda_solver_not_landed"
    CUDA_UNAVAILABLE = "cuda_unavailable"
    DETERMINISM_DISABLED = "determinism_disabled"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    NONCONVERGENCE = "nonconvergence"
    NUMERICAL_REFUSAL = "numerical_refusal"
    MUTATED_CUSTODY = "mutated_custody"
    ACTIVE_LEASE = "active_lease"
    EVICTED = "evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1PhaseAGlobalQCompactOuterLPDualRefusal(RuntimeError):
    def __init__(
        self,
        code: V1PhaseAGlobalQCompactOuterLPDualRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1PhaseAGlobalQCompactOuterLPDualRefusalCode:
            raise TypeError("compact outer-LP dual refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1PhaseAGlobalQCompactOuterLPDualRefusalCode,
    message: str,
) -> NoReturn:
    raise V1PhaseAGlobalQCompactOuterLPDualRefusal(code, message)


def _support_family_source_mode(
    support_family: object,
) -> str:
    """Classify exact support custody while keeping legacy and D1 graphs disjoint."""

    if (
        type(support_family) is V1PhaseAGlobalQSupportFamilyResult
        and type(support_family.receipt) is V1PhaseAGlobalQSupportFamilyReceipt
        and type(support_family.device_handle) is V1PhaseAGlobalQSupportFamilyHandle
        and support_family.device_handle.receipt is support_family.receipt
    ):
        return _LEGACY_PROJECTION_SUPPORT_SOURCE_MODE
    if (
        type(support_family) is V1PhaseAGlobalQD1SupportFamilyResult
        and type(support_family.receipt) is V1PhaseAGlobalQD1SupportFamilyReceipt
        and type(support_family.device_handle) is V1PhaseAGlobalQSupportFamilyHandle
        and support_family.device_handle.receipt is support_family.receipt
    ):
        return _FULL_POPULATION_D1_SUPPORT_SOURCE_MODE
    _refuse(
        V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
        "compact outer-LP requires one exact legacy-projection or "
        "full-population-D1 support owner",
    )


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(dict(payload), prefix=_PREFIX + role + "-")


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return value


def _finite(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be finite{' and nonnegative' if nonnegative else ''}",
        )
    return result


def _positive_float(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be positive",
        )
    return result


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    return (
        id(value),
        int(value.data_ptr()),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        str(value.dtype),
        str(value.device),
        int(value.storage_offset()),
        int(value._version),
        bool(value.requires_grad),
    )


def _compact_tensor_digest(value: Any, *, role: str) -> str:
    array = value.detach().to(device="cpu").contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(role.encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(int(item) for item in array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return _payload_hash(
        {
            "role": role,
            "shape": tuple(int(item) for item in array.shape),
            "payload_sha256": digest.hexdigest(),
        },
        role="compact-tensor",
    )


def _live_tensor_identity_digest(value: Any, *, role: str) -> str:
    return _payload_hash(
        {
            "role": role,
            "tensor_identity_not_durable_content_digest": True,
            "data_ptr": int(value.data_ptr()),
            "storage_offset": int(value.storage_offset()),
            "shape": tuple(int(item) for item in value.shape),
            "stride": tuple(int(item) for item in value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "version": int(value._version),
        },
        role="live-tensor-identity",
    )


def _array_digest(value: np.ndarray, *, role: str) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(role.encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(int(item) for item in array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return _payload_hash(
        {
            "role": role,
            "shape": tuple(int(item) for item in array.shape),
            "payload_sha256": digest.hexdigest(),
        },
        role="array",
    )


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQCompactOuterLPDualDesign:
    execution_mode: V1PhaseAGlobalQCompactOuterLPDualExecutionMode = (
        V1PhaseAGlobalQCompactOuterLPDualExecutionMode.SMALL_DOMAIN_CPU_REFERENCE
    )
    cpu_reference_route_limit: int = 4096
    feasibility_tolerance: float = 1.0e-8
    optimality_tolerance: float = 1.0e-8
    cuda_max_iterations: int = 8192
    cuda_min_iterations: int = 64
    cuda_check_interval: int = 32
    cuda_operator_norm_safety: float = 0.95
    hbm_ceiling_fraction: float = _CUDA_DEFAULT_HBM_CEILING_FRACTION

    @property
    def design_hash(self) -> str:
        max_iterations = _positive_int(
            self.cuda_max_iterations,
            name="cuda_max_iterations",
        )
        min_iterations = _nonnegative_int(
            self.cuda_min_iterations,
            name="cuda_min_iterations",
        )
        check_interval = _positive_int(
            self.cuda_check_interval,
            name="cuda_check_interval",
        )
        norm_safety = _positive_float(
            self.cuda_operator_norm_safety,
            name="cuda_operator_norm_safety",
        )
        hbm_fraction = _positive_float(
            self.hbm_ceiling_fraction,
            name="hbm_ceiling_fraction",
        )
        if (
            min_iterations > max_iterations
            or check_interval > max_iterations
            or norm_safety >= 1.0
            or hbm_fraction > _CUDA_DEFAULT_HBM_CEILING_FRACTION
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
                "CUDA compact dual design parameters are outside their registered bounds",
            )
        return _payload_hash(
            {
                "execution_mode": self.execution_mode.value,
                "cpu_reference_route_limit": _nonnegative_int(
                    self.cpu_reference_route_limit,
                    name="cpu_reference_route_limit",
                ),
                "feasibility_tolerance": _finite(
                    self.feasibility_tolerance,
                    name="feasibility_tolerance",
                    nonnegative=True,
                ),
                "optimality_tolerance": _finite(
                    self.optimality_tolerance,
                    name="optimality_tolerance",
                    nonnegative=True,
                ),
                "cuda_max_iterations": max_iterations,
                "cuda_min_iterations": min_iterations,
                "cuda_check_interval": check_interval,
                "cuda_operator_norm_safety": norm_safety,
                "hbm_ceiling_fraction": hbm_fraction,
            },
            role="design",
        )


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQCompactOuterLPDualReadinessReceipt:
    design_hash: str
    support_family_receipt_hash: str
    relaxed_kkt_problem_hash: str
    route_count: int
    support_plane_count: int
    block_count: int
    missing_authority: str = "downstream literal-J D-stability, C/K/O, and commit authority"
    small_domain_reference_available: bool = True
    production_cuda_source_landed: bool = True
    production_cuda_solver_landed: bool = True
    dual_multiplier_authority: bool = True
    route_scale_hard_face_authority: bool = False
    d_stability_authority: bool = False
    schema: str = PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_READINESS_SCHEMA
    algorithm: str = PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_READINESS_ALGORITHM
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _READINESS_TOKEN:
            raise PermissionError("compact outer-LP dual readiness receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in ("design_hash", "support_family_receipt_hash", "relaxed_kkt_problem_hash", "missing_authority", "schema", "algorithm"):
            _identity(getattr(self, name), name=name)
        for name in ("route_count", "support_plane_count"):
            _positive_int(getattr(self, name), name=name)
        _nonnegative_int(self.block_count, name="block_count")
        if (
            self.small_domain_reference_available is not True
            or self.production_cuda_source_landed is not True
            or self.production_cuda_solver_landed is not True
            or self.dual_multiplier_authority is not True
            or self.route_scale_hard_face_authority is not False
            or self.d_stability_authority is not False
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.AUTHORITY_VIOLATION,
                "production-readiness receipt overclaimed compact dual authority",
            )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType({item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")})

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="readiness")

    def assert_integrity(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.MUTATED_CUSTODY,
                "compact outer-LP dual readiness receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQCompactOuterLPDenseSourceReceipt:
    support_family_receipt_hash: str
    relaxed_kkt_problem_hash: str
    route_axis_hash: str
    current_mask_hash: str
    route_count: int
    support_plane_count: int
    block_count: int
    support_coefficient_lower_digest: str
    resource_digest: str
    eligible_digest: str
    block_ordinal_digest: str
    support_rhs_upper_digest: str
    capacity_digest: str
    measurement_device_identity: str
    small_domain_reference_only: bool = True
    route_scale_source_payload_present: bool = True
    production_cuda_authority: bool = False
    route_scale_d2h_authority: bool = False
    schema: str = PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_SOURCE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _SOURCE_TOKEN:
            raise PermissionError("compact outer-LP dual dense-source receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "support_family_receipt_hash",
            "relaxed_kkt_problem_hash",
            "route_axis_hash",
            "current_mask_hash",
            "support_coefficient_lower_digest",
            "resource_digest",
            "eligible_digest",
            "block_ordinal_digest",
            "support_rhs_upper_digest",
            "capacity_digest",
            "measurement_device_identity",
            "schema",
        ):
            _identity(getattr(self, name), name=name)
        for name in ("route_count", "support_plane_count"):
            _positive_int(getattr(self, name), name=name)
        _nonnegative_int(self.block_count, name="block_count")
        if (
            self.small_domain_reference_only is not True
            or self.route_scale_source_payload_present is not True
            or self.production_cuda_authority is not False
            or self.route_scale_d2h_authority is not False
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.AUTHORITY_VIOLATION,
                "dense source receipt crossed its small-domain reference authority",
            )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType({item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")})

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="dense-source")

    def assert_integrity(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.MUTATED_CUSTODY,
                "compact outer-LP dense-source receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1PhaseAGlobalQCompactOuterLPDenseSource:
    __slots__ = (
        "receipt",
        "_support_coefficient_lower",
        "_resource",
        "_eligible",
        "_block_ordinals",
        "_support_rhs_upper",
        "_capacities",
        "_guards",
        "_invariants",
    )

    def __init__(
        self,
        *,
        receipt: V1PhaseAGlobalQCompactOuterLPDenseSourceReceipt,
        support_coefficient_lower: Any,
        resource: Any,
        eligible: Any,
        block_ordinals: Any,
        support_rhs_upper: Any,
        capacities: Any,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _SOURCE_TOKEN:
            raise PermissionError("compact outer-LP dense sources require their issuer")
        if type(receipt) is not V1PhaseAGlobalQCompactOuterLPDenseSourceReceipt:
            raise TypeError("compact outer-LP dense source requires its exact receipt")
        values = (
            support_coefficient_lower,
            resource,
            eligible,
            block_ordinals,
            support_rhs_upper,
            capacities,
        )
        if torch is None or any(type(value) is not torch.Tensor for value in values):
            raise TypeError("compact outer-LP dense source numerics require exact torch tensors")
        shapes = (
            (receipt.support_plane_count, receipt.route_count),
            (receipt.route_count,),
            (receipt.route_count,),
            (receipt.route_count,),
            (receipt.support_plane_count,),
            (receipt.block_count,),
        )
        if (
            support_coefficient_lower.dtype is not torch.float64
            or resource.dtype is not torch.float64
            or support_rhs_upper.dtype is not torch.float64
            or capacities.dtype is not torch.float64
            or eligible.dtype is not torch.bool
            or block_ordinals.dtype is not torch.int64
            or any(tuple(value.shape) != shape for value, shape in zip(values, shapes, strict=True))
            or any(not value.is_contiguous() for value in values)
            or any(value.requires_grad for value in values)
            or any(str(value.device) != receipt.measurement_device_identity for value in values)
            or any(not bool(torch.isfinite(value).all().item()) for value in (support_coefficient_lower, resource, support_rhs_upper, capacities))
            or bool((support_rhs_upper < 0.0).any().item())
            or bool((capacities < 0.0).any().item())
            or (receipt.block_count and bool(((block_ordinals < 0) | (block_ordinals >= receipt.block_count)).any().item()))
            or (receipt.block_count == 0 and int(block_ordinals.numel()) != receipt.route_count)
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
                "compact outer-LP dense source tensor layout or values are invalid",
            )
        self.receipt = receipt
        self._support_coefficient_lower = support_coefficient_lower
        self._resource = resource
        self._eligible = eligible
        self._block_ordinals = block_ordinals
        self._support_rhs_upper = support_rhs_upper
        self._capacities = capacities
        self._guards = tuple(value.detach().clone() for value in values)
        self._invariants = tuple(_tensor_invariant(value) for value in values)
        self.assert_integrity()

    def assert_integrity(self) -> str:
        observed = self.receipt.assert_integrity()
        values = (
            self._support_coefficient_lower,
            self._resource,
            self._eligible,
            self._block_ordinals,
            self._support_rhs_upper,
            self._capacities,
        )
        if any(
            _tensor_invariant(value) != invariant or not bool(torch.equal(value, guard))
            for value, invariant, guard in zip(values, self._invariants, self._guards, strict=True)
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.MUTATED_CUSTODY,
                "compact outer-LP dense-source tensor custody mutated",
            )
        return observed

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.assert_integrity()
        return (
            self._support_coefficient_lower.detach().cpu().numpy().copy(),
            self._resource.detach().cpu().numpy().copy(),
            self._eligible.detach().cpu().numpy().copy(),
            self._block_ordinals.detach().cpu().numpy().copy(),
            self._support_rhs_upper.detach().cpu().numpy().copy(),
            self._capacities.detach().cpu().numpy().copy(),
        )


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQCompactOuterLPCUDAHBMForecast:
    support_family_receipt_hash: str
    relaxed_kkt_problem_hash: str
    design_hash: str
    route_count: int
    support_plane_count: int
    block_count: int
    source_route_tile_size: int
    reduction_tile_elements: int
    coefficient_bank_bytes: int
    source_route_vector_bytes: int
    primal_dual_workspace_bytes: int
    support_scan_colive_upper_bytes: int
    compact_workspace_bytes: int
    incremental_peak_bytes: int
    current_reserved_device_bytes: int
    projected_peak_reserved_bytes: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    hbm_ceiling_fraction: float
    hbm_ceiling_bytes: int
    admitted: bool
    schema: str = PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_HBM_FORECAST_SCHEMA
    _admission_memory: tuple[int, int] | None = field(default=None, repr=False, compare=False)
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _CUDA_FORECAST_TOKEN:
            raise PermissionError("compact outer-LP CUDA HBM forecasts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        available, allocated = self._admission_memory or (self.device_free_memory_bytes, self.current_reserved_device_bytes)
        for name in (
            "support_family_receipt_hash",
            "relaxed_kkt_problem_hash",
            "design_hash",
            "schema",
        ):
            _identity(getattr(self, name), name=name)
        for name in ("route_count", "support_plane_count", "source_route_tile_size", "reduction_tile_elements"):
            _positive_int(getattr(self, name), name=name)
        _nonnegative_int(self.block_count, name="block_count")
        for name in (
            "coefficient_bank_bytes",
            "source_route_vector_bytes",
            "primal_dual_workspace_bytes",
            "support_scan_colive_upper_bytes",
            "compact_workspace_bytes",
            "incremental_peak_bytes",
            "current_reserved_device_bytes",
            "projected_peak_reserved_bytes",
            "device_total_memory_bytes",
            "device_free_memory_bytes",
            "hbm_ceiling_bytes",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        if self.device_total_memory_bytes <= 0 or self.hbm_ceiling_bytes <= 0:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
                "CUDA HBM forecast requires positive device memory and ceiling",
            )
        hbm_fraction = _positive_float(
            self.hbm_ceiling_fraction,
            name="hbm_ceiling_fraction",
        )
        object.__setattr__(self, "hbm_ceiling_fraction", hbm_fraction)
        expected_coefficient = (
            2 * self.route_count * self.support_plane_count * _F64_BYTES
        )
        expected_source = self.route_count * (_F64_BYTES + _I64_BYTES + _BOOL_BYTES)
        expected_workspace = (
            self.route_count * _CUDA_PRIMAL_DUAL_ROUTE_VECTOR_COUNT * _F64_BYTES
        )
        route_tile = min(self.route_count, self.source_route_tile_size)
        source_scan_colive = 2 * route_tile * self.support_plane_count * _F64_BYTES
        reduction_scratch = (
            2
            * min(
                self.route_count * self.support_plane_count,
                self.reduction_tile_elements,
            )
            * _F64_BYTES
        )
        expected_scan_colive = max(source_scan_colive, reduction_scratch)
        compact_axis = self.support_plane_count + self.block_count
        expected_compact = _F64_BYTES * (
            compact_axis * compact_axis
            + 8 * compact_axis
            + self.support_plane_count
            + self.block_count
        )
        expected_incremental = (
            expected_coefficient
            + expected_source
            + expected_workspace
            + expected_scan_colive
            + expected_compact
        )
        expected_peak = self.current_reserved_device_bytes + expected_incremental
        expected_ceiling = int(math.floor(self.device_total_memory_bytes * hbm_fraction))
        expected_admitted = (
            allocated + expected_incremental <= expected_ceiling
            and expected_incremental <= available
        )
        observed = (
            self.coefficient_bank_bytes,
            self.source_route_vector_bytes,
            self.primal_dual_workspace_bytes,
            self.support_scan_colive_upper_bytes,
            self.compact_workspace_bytes,
            self.incremental_peak_bytes,
            self.projected_peak_reserved_bytes,
            self.hbm_ceiling_bytes,
            self.admitted,
        )
        expected = (
            expected_coefficient,
            expected_source,
            expected_workspace,
            expected_scan_colive,
            expected_compact,
            expected_incremental,
            expected_peak,
            expected_ceiling,
            expected_admitted,
        )
        if observed != expected or self.schema != PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_HBM_FORECAST_SCHEMA:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.AUTHORITY_VIOLATION,
                "compact outer-LP CUDA HBM forecast does not replay its registered formula",
            )
        if hbm_fraction > _CUDA_DEFAULT_HBM_CEILING_FRACTION:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.HBM_FORECAST_EXCEEDED,
                "compact outer-LP CUDA HBM ceiling cannot exceed 85 percent",
            )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType({item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")})

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="cuda-hbm-forecast")

    def assert_integrity(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.MUTATED_CUSTODY,
                "compact outer-LP CUDA HBM forecast no longer replays",
            )
        return observed

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()


def build_v1_phase_a_global_q_compact_outer_lp_cuda_hbm_forecast(
    *,
    support_family: (
        V1PhaseAGlobalQSupportFamilyResult
        | V1PhaseAGlobalQD1SupportFamilyResult
    ),
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    design: V1PhaseAGlobalQCompactOuterLPDualDesign,
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
    current_reserved_device_bytes: int,
    source_route_tile_size: int | None = None,
    _admission_memory: tuple[int, int] | None = None,
) -> V1PhaseAGlobalQCompactOuterLPCUDAHBMForecast:
    if (
        type(relaxed_kkt_problem) is not V1RelaxedKKTProblem
        or type(design) is not V1PhaseAGlobalQCompactOuterLPDualDesign
    ):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            "CUDA HBM forecast requires exact support-family/problem/design sources",
        )
    _support_family_source_mode(support_family)
    support_family.device_handle.assert_live()
    support_hash = support_family.receipt.assert_integrity()
    route_count = support_family.receipt.route_count
    support_plane_count = support_family.receipt.anchor_count
    block_count = len(relaxed_kkt_problem.block_capacities)
    route_tile_size = (
        route_count
        if source_route_tile_size is None
        else _positive_int(source_route_tile_size, name="source_route_tile_size")
    )
    reduction_tile_elements = _positive_int(
        _CUDA_REDUCTION_TILE_ELEMENTS,
        name="reduction_tile_elements",
    )
    coefficient = 2 * route_count * support_plane_count * _F64_BYTES
    source = route_count * (_F64_BYTES + _I64_BYTES + _BOOL_BYTES)
    workspace = route_count * _CUDA_PRIMAL_DUAL_ROUTE_VECTOR_COUNT * _F64_BYTES
    scan_colive = max(
        2 * min(route_count, route_tile_size) * support_plane_count * _F64_BYTES,
        2 * min(route_count * support_plane_count, reduction_tile_elements) * _F64_BYTES,
    )
    compact_axis = support_plane_count + block_count
    compact = _F64_BYTES * (
        compact_axis * compact_axis
        + 8 * compact_axis
        + support_plane_count
        + block_count
    )
    incremental = coefficient + source + workspace + scan_colive + compact
    reserved = _nonnegative_int(
        current_reserved_device_bytes,
        name="current_reserved_device_bytes",
    )
    total = _positive_int(device_total_memory_bytes, name="device_total_memory_bytes")
    free = _nonnegative_int(device_free_memory_bytes, name="device_free_memory_bytes")
    fraction = _positive_float(design.hbm_ceiling_fraction, name="hbm_ceiling_fraction")
    ceiling = int(math.floor(total * fraction))
    available, allocated = _admission_memory or (free, reserved)
    return V1PhaseAGlobalQCompactOuterLPCUDAHBMForecast(
        support_family_receipt_hash=support_hash,
        relaxed_kkt_problem_hash=relaxed_kkt_problem.problem_hash,
        design_hash=design.design_hash,
        route_count=route_count,
        support_plane_count=support_plane_count,
        block_count=block_count,
        source_route_tile_size=route_tile_size,
        reduction_tile_elements=reduction_tile_elements,
        coefficient_bank_bytes=coefficient,
        source_route_vector_bytes=source,
        primal_dual_workspace_bytes=workspace,
        support_scan_colive_upper_bytes=scan_colive,
        compact_workspace_bytes=compact,
        incremental_peak_bytes=incremental,
        current_reserved_device_bytes=reserved,
        projected_peak_reserved_bytes=reserved + incremental,
        device_total_memory_bytes=total,
        device_free_memory_bytes=free,
        hbm_ceiling_fraction=fraction,
        hbm_ceiling_bytes=ceiling,
        admitted=(allocated + incremental <= ceiling and incremental <= available),
        _admission_memory=_admission_memory,
        _issuance_token=_CUDA_FORECAST_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQCompactOuterLPCUDASourceReceipt:
    support_family_receipt_hash: str
    support_plane_scan_receipt_hash: str
    population_mjv_forecast_hash: str
    population_mjv_execution_receipt_hash: str
    joint_preoutcome_registration_hash: str
    joint_direction_axis_hash: str
    direction_ids: tuple[str, ...]
    relaxed_kkt_problem_hash: str
    hbm_forecast_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    current_mask_hash: str
    route_count: int
    support_plane_count: int
    block_count: int
    tile_receipt_hashes: tuple[str, ...]
    route_ranges: tuple[tuple[int, int], ...]
    coefficient_lower_live_tensor_digest: str
    coefficient_upper_live_tensor_digest: str
    resource_live_tensor_digest: str
    eligible_live_tensor_digest: str
    block_ordinal_live_tensor_digest: str
    support_rhs_upper_digest: str
    capacity_digest: str
    measurement_device_identity: str
    production_cuda_authority: bool = True
    private_cpu_reference: bool = False
    lower_endpoint_support_coefficients_used: bool = True
    lower_and_upper_support_coefficients_retained: bool = True
    upper_endpoint_containment_diagnostic_only: bool = True
    no_route_scale_d2h: bool = True
    ephemeral_coefficient_bank: bool = True
    solver_source_authority: bool = True
    algorithm: str = PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_SOURCE_ALGORITHM
    schema: str = PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_SOURCE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _CUDA_SOURCE_TOKEN:
            raise PermissionError("compact outer-LP CUDA source receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "support_family_receipt_hash",
            "support_plane_scan_receipt_hash",
            "population_mjv_forecast_hash",
            "population_mjv_execution_receipt_hash",
            "joint_preoutcome_registration_hash",
            "joint_direction_axis_hash",
            "relaxed_kkt_problem_hash",
            "hbm_forecast_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "current_mask_hash",
            "coefficient_lower_live_tensor_digest",
            "coefficient_upper_live_tensor_digest",
            "resource_live_tensor_digest",
            "eligible_live_tensor_digest",
            "block_ordinal_live_tensor_digest",
            "support_rhs_upper_digest",
            "capacity_digest",
            "measurement_device_identity",
            "algorithm",
            "schema",
        ):
            _identity(getattr(self, name), name=name)
        for name in ("route_count", "support_plane_count"):
            _positive_int(getattr(self, name), name=name)
        _nonnegative_int(self.block_count, name="block_count")
        if (
            type(self.direction_ids) is not tuple
            or len(self.direction_ids) != self.support_plane_count
            or len(set(self.direction_ids)) != len(self.direction_ids)
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
                "CUDA compact dual source direction_ids must be one immutable support axis",
            )
        direction_ids = tuple(
            _identity(item, name="direction_id") for item in self.direction_ids
        )
        ranges = tuple(self.route_ranges)
        tile_hashes = tuple(_identity(item, name="tile_receipt_hash") for item in self.tile_receipt_hashes)
        if (
            len(tile_hashes) != len(ranges)
            or not ranges
            or ranges[0][0] != 0
            or ranges[-1][1] != self.route_count
            or any(
                type(item) is not tuple
                or len(item) != 2
                or type(item[0]) is not int
                or type(item[1]) is not int
                or not 0 <= item[0] < item[1] <= self.route_count
                for item in ranges
            )
            or any(ranges[index][1] != ranges[index + 1][0] for index in range(len(ranges) - 1))
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.SOURCE_MISMATCH,
                "CUDA compact dual source did not cover one complete route axis",
            )
        positive = (
            self.production_cuda_authority,
            self.lower_endpoint_support_coefficients_used,
            self.lower_and_upper_support_coefficients_retained,
            self.upper_endpoint_containment_diagnostic_only,
            self.no_route_scale_d2h,
            self.ephemeral_coefficient_bank,
            self.solver_source_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or self.private_cpu_reference is not False
            or not self.measurement_device_identity.startswith("cuda:")
            or self.algorithm != PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_SOURCE_ALGORITHM
            or self.schema != PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_SOURCE_SCHEMA
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.AUTHORITY_VIOLATION,
                "CUDA compact dual source receipt crossed its source authority",
            )
        object.__setattr__(self, "direction_ids", direction_ids)
        object.__setattr__(self, "tile_receipt_hashes", tile_hashes)
        object.__setattr__(self, "route_ranges", ranges)

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType({item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")})

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="cuda-source")

    def assert_integrity(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.MUTATED_CUSTODY,
                "compact outer-LP CUDA source receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1PhaseAGlobalQCompactOuterLPCUDASource:
    __slots__ = (
        "receipt",
        "forecast",
        "_coefficient_lower",
        "_coefficient_upper",
        "_resource",
        "_eligible",
        "_block_ordinals",
        "_support_rhs_upper",
        "_capacities",
        "_invariants",
        "_evicted",
    )

    def __init__(
        self,
        *,
        receipt: V1PhaseAGlobalQCompactOuterLPCUDASourceReceipt,
        forecast: V1PhaseAGlobalQCompactOuterLPCUDAHBMForecast,
        coefficient_lower: Any,
        coefficient_upper: Any,
        resource: Any,
        eligible: Any,
        block_ordinals: Any,
        support_rhs_upper: Any,
        capacities: Any,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _CUDA_SOURCE_TOKEN:
            raise PermissionError("compact outer-LP CUDA sources require their issuer")
        if (
            type(receipt) is not V1PhaseAGlobalQCompactOuterLPCUDASourceReceipt
            or type(forecast) is not V1PhaseAGlobalQCompactOuterLPCUDAHBMForecast
        ):
            raise TypeError("compact outer-LP CUDA source requires exact receipt and forecast")
        if torch is None:
            raise RuntimeError("compact outer-LP CUDA source requires PyTorch")
        values = (
            coefficient_lower,
            coefficient_upper,
            resource,
            eligible,
            block_ordinals,
            support_rhs_upper,
            capacities,
        )
        if any(type(value) is not torch.Tensor for value in values):
            raise TypeError("compact outer-LP CUDA source numerics require exact torch tensors")
        device = torch.device(receipt.measurement_device_identity)
        shapes = (
            (receipt.support_plane_count, receipt.route_count),
            (receipt.support_plane_count, receipt.route_count),
            (receipt.route_count,),
            (receipt.route_count,),
            (receipt.route_count,),
            (receipt.support_plane_count,),
            (receipt.block_count,),
        )
        if (
            coefficient_lower.dtype is not torch.float64
            or coefficient_upper.dtype is not torch.float64
            or resource.dtype is not torch.float64
            or support_rhs_upper.dtype is not torch.float64
            or capacities.dtype is not torch.float64
            or eligible.dtype is not torch.bool
            or block_ordinals.dtype is not torch.int64
            or any(tuple(value.shape) != shape for value, shape in zip(values, shapes, strict=True))
            or any(not value.is_contiguous() for value in values)
            or any(value.requires_grad for value in values)
            or any(value.device != device for value in values)
            or any(not bool(torch.isfinite(value).all().item()) for value in (coefficient_lower, coefficient_upper, resource, support_rhs_upper, capacities))
            or bool((coefficient_lower > coefficient_upper).any().item())
            or bool((resource < 0.0).any().item())
            or bool((support_rhs_upper < 0.0).any().item())
            or bool((capacities < 0.0).any().item())
            or (receipt.block_count and bool(((block_ordinals < 0) | (block_ordinals >= receipt.block_count)).any().item()))
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
                "compact outer-LP CUDA source tensor layout or values are invalid",
            )
        if (
            receipt.hbm_forecast_hash != forecast.forecast_hash
            or forecast.route_count != receipt.route_count
            or forecast.support_plane_count != receipt.support_plane_count
            or forecast.block_count != receipt.block_count
            or forecast.admitted is not True
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.SOURCE_MISMATCH,
                "compact outer-LP CUDA source and HBM forecast diverged",
            )
        self.receipt = receipt
        self.forecast = forecast
        self._coefficient_lower = coefficient_lower
        self._coefficient_upper = coefficient_upper
        self._resource = resource
        self._eligible = eligible
        self._block_ordinals = block_ordinals
        self._support_rhs_upper = support_rhs_upper
        self._capacities = capacities
        self._invariants = tuple(_tensor_invariant(value) for value in values)
        self._evicted = False
        self.assert_live()

    def assert_live(self) -> str:
        if self._evicted:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.EVICTED,
                "compact outer-LP CUDA source was evicted",
            )
        observed = self.receipt.assert_integrity()
        self.forecast.assert_integrity()
        values = (
            self._coefficient_lower,
            self._coefficient_upper,
            self._resource,
            self._eligible,
            self._block_ordinals,
            self._support_rhs_upper,
            self._capacities,
        )
        if any(
            _tensor_invariant(value) != invariant
            for value, invariant in zip(values, self._invariants, strict=True)
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.MUTATED_CUSTODY,
                "compact outer-LP CUDA source tensor custody mutated",
            )
        return observed

    def tensors(self) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        self.assert_live()
        return (
            self._coefficient_lower,
            self._coefficient_upper,
            self._resource,
            self._eligible,
            self._block_ordinals,
            self._support_rhs_upper,
            self._capacities,
        )

    def evict(self) -> None:
        self.assert_live()
        device = torch.device(self.receipt.measurement_device_identity)
        self._coefficient_lower = torch.empty((0, 0), dtype=torch.float64, device=device)
        self._coefficient_upper = torch.empty((0, 0), dtype=torch.float64, device=device)
        self._resource = torch.empty(0, dtype=torch.float64, device=device)
        self._eligible = torch.empty(0, dtype=torch.bool, device=device)
        self._block_ordinals = torch.empty(0, dtype=torch.int64, device=device)
        self._support_rhs_upper = torch.empty(0, dtype=torch.float64, device=device)
        self._capacities = torch.empty(0, dtype=torch.float64, device=device)
        self._invariants = ()
        self._evicted = True

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def __reduce__(self) -> NoReturn:
        raise TypeError("compact outer-LP CUDA sources are nonserializable")


def _block_ordinals_for_problem(
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    *,
    device: Any,
) -> Any:
    block_ordinals = torch.zeros(
        (relaxed_kkt_problem.route_count,),
        dtype=torch.int64,
        device=device,
    )
    for ordinal, block in enumerate(relaxed_kkt_problem.block_capacities):
        block_ordinals[block.start : block.stop] = int(ordinal)
    return block_ordinals.contiguous()


def issue_v1_phase_a_global_q_compact_outer_lp_cuda_source_from_scan(
    *,
    support_plane_scan: V1PhaseAGlobalQSupportPlaneScanHandle,
    support_family: (
        V1PhaseAGlobalQSupportFamilyResult
        | V1PhaseAGlobalQD1SupportFamilyResult
    ),
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    design: V1PhaseAGlobalQCompactOuterLPDualDesign = V1PhaseAGlobalQCompactOuterLPDualDesign(
        execution_mode=V1PhaseAGlobalQCompactOuterLPDualExecutionMode.CUDA_PRODUCTION
    ),
) -> V1PhaseAGlobalQCompactOuterLPCUDASource:
    if (
        type(support_plane_scan) is not V1PhaseAGlobalQSupportPlaneScanHandle
        or type(relaxed_kkt_problem) is not V1RelaxedKKTProblem
        or type(design) is not V1PhaseAGlobalQCompactOuterLPDualDesign
    ):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            "CUDA compact dual source requires exact support scan, support family, problem, and design",
        )
    support_source_mode = _support_family_source_mode(support_family)
    if (
        getattr(support_plane_scan, "_source_mode", None) != support_source_mode
        or getattr(support_plane_scan, "_support", None) is not support_family
    ):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.SOURCE_MISMATCH,
            "support-plane scan and compact source crossed the legacy/D1 "
            "source-mode or live support-custody boundary",
        )
    if design.execution_mode is not V1PhaseAGlobalQCompactOuterLPDualExecutionMode.CUDA_PRODUCTION:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            "CUDA compact dual source requires CUDA production design",
        )
    if torch is None or not bool(torch.cuda.is_available()):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.CUDA_UNAVAILABLE,
            "compact outer-LP CUDA source requires an available CUDA runtime",
        )
    if not bool(torch.are_deterministic_algorithms_enabled()):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.DETERMINISM_DISABLED,
            "compact outer-LP CUDA source requires deterministic torch algorithms",
        )
    support_plane_scan.assert_live()
    support_family.device_handle.assert_live()
    support = support_family.receipt
    support_hash = support.assert_integrity()
    if (
        support.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash
        or support.route_axis_hash != relaxed_kkt_problem.route_axis_hash
        or support.route_count != relaxed_kkt_problem.route_count
        or support.production_source_authority is not True
        or support.private_cpu_reference is not False
        or not support.measurement_device_identity.startswith("cuda:")
    ):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.SOURCE_MISMATCH,
            "support family is not the registered production CUDA problem source",
        )
    device = torch.device(support.measurement_device_identity)
    source_route_tile_size = _positive_int(
        getattr(support_plane_scan._stream.forecast, "route_tile_size", None),
        name="source_route_tile_size",
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    forecast = build_v1_phase_a_global_q_compact_outer_lp_cuda_hbm_forecast(
        support_family=support_family,
        relaxed_kkt_problem=relaxed_kkt_problem,
        design=design,
        device_total_memory_bytes=int(total_bytes),
        device_free_memory_bytes=int(free_bytes),
        current_reserved_device_bytes=int(torch.cuda.memory_reserved(device)),
        source_route_tile_size=source_route_tile_size,
        _admission_memory=(cuda_reusable_memory_bytes(device, framework=torch, free_bytes=free_bytes), int(torch.cuda.memory_allocated(device))),
    )
    if not forecast.admitted:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.HBM_FORECAST_EXCEEDED,
            "compact outer-LP CUDA source exceeds the pre-outcome 85% HBM envelope",
        )
    coefficient_lower = torch.empty(
        (support.anchor_count, support.route_count),
        dtype=torch.float64,
        device=device,
    )
    coefficient_upper = torch.empty_like(coefficient_lower)
    resource = torch.as_tensor(
        np.asarray(relaxed_kkt_problem.resource, dtype=np.float64),
        dtype=torch.float64,
        device=device,
    ).clone().contiguous()
    eligible = torch.as_tensor(
        np.asarray(relaxed_kkt_problem.eligible_mask, dtype=np.bool_),
        dtype=torch.bool,
        device=device,
    ).clone().contiguous()
    block_ordinals = _block_ordinals_for_problem(relaxed_kkt_problem, device=device)
    capacities = torch.tensor(
        tuple(float(block.removal_capacity) for block in relaxed_kkt_problem.block_capacities),
        dtype=torch.float64,
        device=device,
    ).contiguous()
    tile_hashes: list[str] = []
    route_ranges: list[tuple[int, int]] = []
    rhs_upper: Any | None = None
    while True:
        tile = support_plane_scan.next_tile()
        if tile is None:
            break
        try:
            receipt = tile.receipt
            start = receipt.route_start
            stop = receipt.route_stop
            if route_ranges and route_ranges[-1][1] != start:
                _refuse(
                    V1PhaseAGlobalQCompactOuterLPDualRefusalCode.SOURCE_MISMATCH,
                    "support-plane tiles are not one contiguous route scan",
                )
            with tile.borrow() as lease:
                lower, upper, tile_rhs = lease.coefficient_bounds()
                coefficient_lower[:, start:stop].copy_(lower)
                coefficient_upper[:, start:stop].copy_(upper)
                if rhs_upper is None:
                    rhs_upper = tile_rhs.clone().contiguous()
                elif not bool(torch.equal(rhs_upper, tile_rhs)):
                    _refuse(
                        V1PhaseAGlobalQCompactOuterLPDualRefusalCode.SOURCE_MISMATCH,
                        "support RHS changed during CUDA source construction",
                    )
            tile_hashes.append(receipt.receipt_hash)
            route_ranges.append((start, stop))
        finally:
            tile.evict()
    scan_receipt = support_plane_scan.finalize()
    if (
        type(scan_receipt) is not V1PhaseAGlobalQSupportPlaneScanReceipt
        or scan_receipt.support_family_receipt_hash != support_hash
        or scan_receipt.joint_direction_axis_hash != support.joint_direction_axis_hash
        or scan_receipt.direction_ids != support.anchor_direction_ids
        or rhs_upper is None
        or not route_ranges
        or route_ranges[0][0] != 0
        or route_ranges[-1][1] != support.route_count
        or scan_receipt.route_ranges != tuple(route_ranges)
        or scan_receipt.tile_receipt_hashes != tuple(tile_hashes)
        or scan_receipt.route_axis_hash != support.route_axis_hash
        or scan_receipt.route_id_axis_hash != support.route_id_axis_hash
        or scan_receipt.current_mask_hash != support.current_mask_hash
        or scan_receipt.production_source_authority is not True
        or scan_receipt.private_cpu_reference is not False
    ):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.SOURCE_MISMATCH,
            "final support-plane scan receipt does not replay the CUDA source construction",
        )
    receipt = V1PhaseAGlobalQCompactOuterLPCUDASourceReceipt(
        support_family_receipt_hash=support_hash,
        support_plane_scan_receipt_hash=scan_receipt.receipt_hash,
        population_mjv_forecast_hash=scan_receipt.population_mjv_forecast_hash,
        population_mjv_execution_receipt_hash=scan_receipt.population_mjv_execution_receipt_hash,
        joint_preoutcome_registration_hash=scan_receipt.joint_preoutcome_registration_hash,
        joint_direction_axis_hash=scan_receipt.joint_direction_axis_hash,
        direction_ids=scan_receipt.direction_ids,
        relaxed_kkt_problem_hash=relaxed_kkt_problem.problem_hash,
        hbm_forecast_hash=forecast.forecast_hash,
        route_axis_hash=support.route_axis_hash,
        route_id_axis_hash=support.route_id_axis_hash,
        current_mask_hash=support.current_mask_hash,
        route_count=support.route_count,
        support_plane_count=support.anchor_count,
        block_count=len(relaxed_kkt_problem.block_capacities),
        tile_receipt_hashes=tuple(tile_hashes),
        route_ranges=tuple(route_ranges),
        coefficient_lower_live_tensor_digest=_live_tensor_identity_digest(coefficient_lower, role="coefficient-lower"),
        coefficient_upper_live_tensor_digest=_live_tensor_identity_digest(coefficient_upper, role="coefficient-upper"),
        resource_live_tensor_digest=_live_tensor_identity_digest(resource, role="resource"),
        eligible_live_tensor_digest=_live_tensor_identity_digest(eligible, role="eligible"),
        block_ordinal_live_tensor_digest=_live_tensor_identity_digest(block_ordinals, role="block-ordinals"),
        support_rhs_upper_digest=_compact_tensor_digest(rhs_upper, role="support-rhs-upper"),
        capacity_digest=_compact_tensor_digest(capacities, role="capacities"),
        measurement_device_identity=str(device),
        _issuance_token=_CUDA_SOURCE_TOKEN,
    )
    return V1PhaseAGlobalQCompactOuterLPCUDASource(
        receipt=receipt,
        forecast=forecast,
        coefficient_lower=coefficient_lower.contiguous(),
        coefficient_upper=coefficient_upper.contiguous(),
        resource=resource,
        eligible=eligible,
        block_ordinals=block_ordinals,
        support_rhs_upper=rhs_upper,
        capacities=capacities,
        _issuance_token=_CUDA_SOURCE_TOKEN,
    )


def issue_v1_phase_a_global_q_compact_outer_lp_dense_source_for_tests(
    *,
    support_family: (
        V1PhaseAGlobalQSupportFamilyResult
        | V1PhaseAGlobalQD1SupportFamilyResult
    ),
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    support_coefficient_lower: Any,
    block_ordinals: Any,
    allow_private_cpu_reference_for_tests: bool = False,
) -> V1PhaseAGlobalQCompactOuterLPDenseSource:
    """Create an exact dense small-domain LP source.

    This constructor is intentionally test/reference scoped.  Production must
    obtain the same coefficients through a tiled CUDA source, not by retaining a
    route-by-support matrix.
    """

    if torch is None:
        raise RuntimeError("compact outer-LP dense-source issuance requires PyTorch")
    if (
        type(relaxed_kkt_problem) is not V1RelaxedKKTProblem
        or type(allow_private_cpu_reference_for_tests) is not bool
    ):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            "dense-source issuance requires exact support-family and KKT problem sources",
        )
    _support_family_source_mode(support_family)
    support_family.device_handle.assert_live()
    support = support_family.receipt
    support_hash = support.assert_integrity()
    if not allow_private_cpu_reference_for_tests:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.AUTHORITY_VIOLATION,
            "dense route-by-support source is private-reference only",
        )
    if (
        support.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash
        or support.route_axis_hash != relaxed_kkt_problem.route_axis_hash
        or support.route_count != relaxed_kkt_problem.route_count
    ):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.SOURCE_MISMATCH,
            "support family and relaxed KKT problem are not one route axis",
        )
    with support_family.device_handle.borrow() as lease:
        rhs = lease.rhs_upper.detach().clone().contiguous()
    coefficient = support_coefficient_lower.detach().clone().contiguous()
    block_axis = block_ordinals.detach().clone().contiguous()
    resource = torch.tensor(
        np.asarray(relaxed_kkt_problem.resource, dtype=np.float64).copy(),
        dtype=torch.float64,
        device=rhs.device,
    )
    eligible = torch.tensor(
        np.asarray(relaxed_kkt_problem.eligible_mask, dtype=np.bool_).copy(),
        dtype=torch.bool,
        device=rhs.device,
    )
    capacities = torch.tensor(
        tuple(float(block.removal_capacity) for block in relaxed_kkt_problem.block_capacities),
        dtype=torch.float64,
        device=rhs.device,
    )
    if (
        str(rhs.device) != "cpu"
        or not support.private_cpu_reference
        or support.production_source_authority
    ):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.DEVICE_REFUSAL,
            "dense small-domain reference source must remain CPU/private",
        )
    receipt = V1PhaseAGlobalQCompactOuterLPDenseSourceReceipt(
        support_family_receipt_hash=support_hash,
        relaxed_kkt_problem_hash=relaxed_kkt_problem.problem_hash,
        route_axis_hash=support.route_axis_hash,
        current_mask_hash=support.current_mask_hash,
        route_count=support.route_count,
        support_plane_count=support.anchor_count,
        block_count=len(relaxed_kkt_problem.block_capacities),
        support_coefficient_lower_digest=_compact_tensor_digest(coefficient, role="support-coefficient-lower"),
        resource_digest=_array_digest(np.asarray(relaxed_kkt_problem.resource, dtype=np.float64), role="resource"),
        eligible_digest=_array_digest(np.asarray(relaxed_kkt_problem.eligible_mask, dtype=np.bool_), role="eligible"),
        block_ordinal_digest=_compact_tensor_digest(block_axis, role="block-ordinals"),
        support_rhs_upper_digest=_compact_tensor_digest(rhs, role="support-rhs-upper"),
        capacity_digest=_compact_tensor_digest(capacities, role="capacities"),
        measurement_device_identity=str(rhs.device),
        _issuance_token=_SOURCE_TOKEN,
    )
    return V1PhaseAGlobalQCompactOuterLPDenseSource(
        receipt=receipt,
        support_coefficient_lower=coefficient,
        resource=resource,
        eligible=eligible,
        block_ordinals=block_axis,
        support_rhs_upper=rhs,
        capacities=capacities,
        _issuance_token=_SOURCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQCompactOuterLPDualSolveReceipt:
    design_hash: str
    dense_source_receipt_hash: str
    support_family_receipt_hash: str
    relaxed_kkt_problem_hash: str
    route_axis_hash: str
    current_mask_hash: str
    route_count: int
    support_plane_count: int
    block_count: int
    support_multiplier_digest: str
    block_multiplier_digest: str
    primal_fractional_witness_digest: str
    structural_dual_upper: float
    fractional_primal_lower: float
    primal_dual_gap_upper: float
    max_fractional_support_violation: float
    max_fractional_block_violation: float
    max_dual_constraint_violation: float
    scipy_status: int
    scipy_message: str
    cuda_coefficient_scan_equivalent_count: int = 0
    cuda_coefficient_bytes_per_pdhg_iteration: int = 0
    cuda_gamma_bounded_reduction_authority: bool = False
    cuda_operator_norm_frobenius_bound: bool = False
    production_cuda_execution: bool = False
    small_domain_cpu_reference: bool = True
    support_and_block_multipliers_endogenous: bool = True
    lower_endpoint_support_coefficients_used: bool = True
    scipy_highs_certificate_used: bool = True
    route_scale_hard_face_authority: bool = False
    d_stability_authority: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_ALGORITHM
    authority_scope: str = PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_AUTHORITY
    schema: str = PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_SOLVE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("compact outer-LP dual solve receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "design_hash",
            "dense_source_receipt_hash",
            "support_family_receipt_hash",
            "relaxed_kkt_problem_hash",
            "route_axis_hash",
            "current_mask_hash",
            "support_multiplier_digest",
            "block_multiplier_digest",
            "primal_fractional_witness_digest",
            "scipy_message",
            "algorithm",
            "authority_scope",
            "schema",
        ):
            _identity(getattr(self, name), name=name)
        for name in ("route_count", "support_plane_count"):
            _positive_int(getattr(self, name), name=name)
        _nonnegative_int(self.block_count, name="block_count")
        _nonnegative_int(
            self.cuda_coefficient_scan_equivalent_count,
            name="cuda_coefficient_scan_equivalent_count",
        )
        _nonnegative_int(
            self.cuda_coefficient_bytes_per_pdhg_iteration,
            name="cuda_coefficient_bytes_per_pdhg_iteration",
        )
        for name in (
            "structural_dual_upper",
            "fractional_primal_lower",
            "primal_dual_gap_upper",
            "max_fractional_support_violation",
            "max_fractional_block_violation",
            "max_dual_constraint_violation",
        ):
            _finite(getattr(self, name), name=name, nonnegative=(name != "fractional_primal_lower"))
        cpu_mode = (
            self.production_cuda_execution is False
            and self.small_domain_cpu_reference is True
            and self.scipy_highs_certificate_used is True
            and self.algorithm == PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_ALGORITHM
        )
        cuda_mode = (
            self.production_cuda_execution is True
            and self.small_domain_cpu_reference is False
            and self.scipy_highs_certificate_used is False
            and self.cuda_gamma_bounded_reduction_authority is True
            and self.cuda_operator_norm_frobenius_bound is True
            and self.cuda_coefficient_scan_equivalent_count > 0
            and self.cuda_coefficient_bytes_per_pdhg_iteration
            == 2 * self.route_count * self.support_plane_count * _F64_BYTES
            and self.algorithm == PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_ALGORITHM
        )
        if (
            (not cpu_mode and not cuda_mode)
            or self.support_and_block_multipliers_endogenous is not True
            or self.lower_endpoint_support_coefficients_used is not True
            or (
                cpu_mode
                and (
                    self.cuda_gamma_bounded_reduction_authority is not False
                    or self.cuda_operator_norm_frobenius_bound is not False
                    or self.cuda_coefficient_scan_equivalent_count != 0
                    or self.cuda_coefficient_bytes_per_pdhg_iteration != 0
                )
            )
            or self.route_scale_hard_face_authority is not False
            or self.d_stability_authority is not False
            or self.c_measurement_authority is not False
            or self.calibration_authority is not False
            or self.oracle_authority is not False
            or self.commit_authority is not False
            or self.runtime_authority is not False
            or self.terminal_authority is not False
            or self.pareto_authority is not False
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.AUTHORITY_VIOLATION,
                "compact outer-LP dual solve receipt overclaimed authority",
            )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType({item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")})

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="solve")

    def assert_integrity(self) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.MUTATED_CUSTODY,
                "compact outer-LP dual solve receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1PhaseAGlobalQCompactOuterLPDualLease:
    __slots__ = ("_owner", "_token", "_closed", "support_multipliers", "block_multipliers")

    def __init__(self, owner: "V1PhaseAGlobalQCompactOuterLPDualHandle", token: object) -> None:
        self._owner = owner
        self._token = token
        self._closed = False
        self.support_multipliers = owner._support_multipliers
        self.block_multipliers = owner._block_multipliers

    def __enter__(self) -> "V1PhaseAGlobalQCompactOuterLPDualLease":
        self._owner.assert_live()
        return self

    def close(self) -> None:
        if not self._closed:
            self._owner._close(self._token)
            self._closed = True

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("compact outer-LP dual leases are nonserializable")


class V1PhaseAGlobalQCompactOuterLPDualHandle:
    __slots__ = (
        "receipt",
        "_support_multipliers",
        "_block_multipliers",
        "_guards",
        "_invariants",
        "_active_token",
        "_evicted",
    )

    def __init__(
        self,
        *,
        receipt: V1PhaseAGlobalQCompactOuterLPDualSolveReceipt,
        support_multipliers: Any,
        block_multipliers: Any,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _HANDLE_TOKEN:
            raise PermissionError("compact outer-LP dual handles require their issuer")
        if type(receipt) is not V1PhaseAGlobalQCompactOuterLPDualSolveReceipt:
            raise TypeError("compact outer-LP dual handle requires its exact receipt")
        if torch is None or type(support_multipliers) is not torch.Tensor or type(block_multipliers) is not torch.Tensor:
            raise TypeError("compact outer-LP dual handle requires exact torch tensors")
        if (
            support_multipliers.dtype is not torch.float64
            or block_multipliers.dtype is not torch.float64
            or tuple(support_multipliers.shape) != (receipt.support_plane_count,)
            or tuple(block_multipliers.shape) != (receipt.block_count,)
            or not support_multipliers.is_contiguous()
            or not block_multipliers.is_contiguous()
            or support_multipliers.requires_grad
            or block_multipliers.requires_grad
            or not bool(torch.isfinite(support_multipliers).all().item())
            or not bool(torch.isfinite(block_multipliers).all().item())
            or bool((support_multipliers < 0.0).any().item())
            or bool((block_multipliers < 0.0).any().item())
            or _compact_tensor_digest(support_multipliers, role="support-multipliers") != receipt.support_multiplier_digest
            or _compact_tensor_digest(block_multipliers, role="block-multipliers") != receipt.block_multiplier_digest
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.MUTATED_CUSTODY,
                "compact outer-LP dual tensor custody does not match receipt",
            )
        self.receipt = receipt
        self._support_multipliers = support_multipliers
        self._block_multipliers = block_multipliers
        self._guards = (support_multipliers.detach().clone(), block_multipliers.detach().clone())
        self._invariants = (_tensor_invariant(support_multipliers), _tensor_invariant(block_multipliers))
        self._active_token: object | None = None
        self._evicted = False

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(V1PhaseAGlobalQCompactOuterLPDualRefusalCode.EVICTED, "compact outer-LP dual handle was evicted")
        self.receipt.assert_integrity()
        values = (self._support_multipliers, self._block_multipliers)
        if any(
            _tensor_invariant(value) != invariant or not bool(torch.equal(value, guard))
            for value, invariant, guard in zip(values, self._invariants, self._guards, strict=True)
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.MUTATED_CUSTODY,
                "compact outer-LP dual tensor custody mutated",
            )

    def borrow(self) -> V1PhaseAGlobalQCompactOuterLPDualLease:
        self.assert_live()
        if self._active_token is not None:
            _refuse(V1PhaseAGlobalQCompactOuterLPDualRefusalCode.ACTIVE_LEASE, "only one compact outer-LP dual lease may be live")
        token = object()
        self._active_token = token
        return V1PhaseAGlobalQCompactOuterLPDualLease(self, token)

    def _close(self, token: object) -> None:
        if token is not self._active_token:
            _refuse(V1PhaseAGlobalQCompactOuterLPDualRefusalCode.ACTIVE_LEASE, "compact outer-LP dual lease token is stale")
        try:
            self.assert_live()
        finally:
            self._active_token = None

    def evict(self) -> None:
        self.assert_live()
        if self._active_token is not None:
            _refuse(V1PhaseAGlobalQCompactOuterLPDualRefusalCode.ACTIVE_LEASE, "cannot evict compact outer-LP dual under an active lease")
        self._support_multipliers = torch.empty(0, dtype=torch.float64)
        self._block_multipliers = torch.empty(0, dtype=torch.float64)
        self._guards = ()
        self._invariants = ()
        self._evicted = True

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def __reduce__(self) -> NoReturn:
        raise TypeError("compact outer-LP dual handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1PhaseAGlobalQCompactOuterLPDualSolveResult:
    receipt: V1PhaseAGlobalQCompactOuterLPDualSolveReceipt
    device_handle: V1PhaseAGlobalQCompactOuterLPDualHandle = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1PhaseAGlobalQCompactOuterLPDualSolveReceipt or type(self.device_handle) is not V1PhaseAGlobalQCompactOuterLPDualHandle:
            raise TypeError("compact outer-LP dual solve result requires exact components")
        self.device_handle.assert_live()
        if self.device_handle.receipt.receipt_hash != self.receipt.receipt_hash:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.SOURCE_MISMATCH,
                "compact outer-LP dual receipt and handle diverged",
            )


def issue_v1_phase_a_global_q_compact_outer_lp_dual_production_readiness(
    *,
    support_family: (
        V1PhaseAGlobalQSupportFamilyResult
        | V1PhaseAGlobalQD1SupportFamilyResult
    ),
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    design: V1PhaseAGlobalQCompactOuterLPDualDesign = V1PhaseAGlobalQCompactOuterLPDualDesign(
        execution_mode=V1PhaseAGlobalQCompactOuterLPDualExecutionMode.CUDA_PRODUCTION
    ),
) -> V1PhaseAGlobalQCompactOuterLPDualReadinessReceipt:
    if type(relaxed_kkt_problem) is not V1RelaxedKKTProblem or type(design) is not V1PhaseAGlobalQCompactOuterLPDualDesign:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            "production readiness requires exact support-family/problem/design sources",
        )
    _support_family_source_mode(support_family)
    support_family.device_handle.assert_live()
    support = support_family.receipt
    support_hash = support.assert_integrity()
    if support.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash or support.route_count != relaxed_kkt_problem.route_count:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.SOURCE_MISMATCH,
            "support family and relaxed KKT problem diverged",
        )
    return V1PhaseAGlobalQCompactOuterLPDualReadinessReceipt(
        design_hash=design.design_hash,
        support_family_receipt_hash=support_hash,
        relaxed_kkt_problem_hash=relaxed_kkt_problem.problem_hash,
        route_count=support.route_count,
        support_plane_count=support.anchor_count,
        block_count=len(relaxed_kkt_problem.block_capacities),
        _issuance_token=_READINESS_TOKEN,
    )


def _solve_small_domain_lp(
    *,
    coefficient: np.ndarray,
    resource: np.ndarray,
    block_ordinals: np.ndarray,
    rhs: np.ndarray,
    capacities: np.ndarray,
    feasibility_tolerance: float,
    optimality_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float, float, int, str]:
    if linprog is None:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.SCIPY_REQUIRED,
            "small-domain compact outer-LP dual reference requires scipy.optimize.linprog",
        )
    direction_count, route_count = coefficient.shape
    block_count = int(capacities.size)
    variable_count = direction_count + block_count + route_count
    c = np.concatenate((rhs, capacities, np.ones(route_count, dtype=np.float64)))
    a_ub = np.zeros((route_count, variable_count), dtype=np.float64)
    a_ub[:, :direction_count] = -coefficient.T
    for route, block in enumerate(block_ordinals.tolist()):
        if block_count:
            a_ub[route, direction_count + int(block)] = -1.0
    a_ub[:, direction_count + block_count :] = -np.eye(route_count, dtype=np.float64)
    b_ub = -resource
    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=[(0.0, None)] * variable_count,
        method="highs",
        options={"primal_feasibility_tolerance": feasibility_tolerance, "dual_feasibility_tolerance": feasibility_tolerance},
    )
    if not result.success:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
            f"small-domain compact outer-LP dual solve failed: {result.message}",
        )
    values = np.asarray(result.x, dtype=np.float64)
    if values.shape != (variable_count,) or not np.all(np.isfinite(values)):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
            "small-domain compact outer-LP dual returned invalid coordinates",
        )
    support_multipliers = np.maximum(values[:direction_count], 0.0)
    block_multipliers = np.maximum(values[direction_count : direction_count + block_count], 0.0)
    advantage = resource - coefficient.T @ support_multipliers
    if block_count:
        advantage = advantage - block_multipliers[block_ordinals]
    slack = np.maximum(advantage, 0.0)
    structural_dual_upper = float(rhs @ support_multipliers + capacities @ block_multipliers + np.sum(slack))
    max_dual_violation = float(max(0.0, np.max(advantage - slack))) if route_count else 0.0
    marginals = getattr(getattr(result, "ineqlin", None), "marginals", None)
    if marginals is None:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
            "HiGHS did not expose fractional primal marginals",
        )
    fractional = np.asarray(-marginals, dtype=np.float64)
    if fractional.shape != (route_count,) or not np.all(np.isfinite(fractional)):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
            "HiGHS fractional primal marginals are invalid",
        )
    fractional = np.clip(fractional, 0.0, 1.0)
    support_violation = coefficient @ fractional - rhs
    if block_count:
        block_sums = np.bincount(block_ordinals, weights=fractional, minlength=block_count)
        block_violation = block_sums - capacities
    else:
        block_violation = np.zeros(0, dtype=np.float64)
    max_support_violation = float(max(0.0, np.max(support_violation))) if direction_count else 0.0
    max_block_violation = float(max(0.0, np.max(block_violation))) if block_count else 0.0
    fractional_lower = float(resource @ fractional)
    gap = max(0.0, structural_dual_upper - fractional_lower)
    if (
        max_support_violation > 32.0 * feasibility_tolerance
        or max_block_violation > 32.0 * feasibility_tolerance
        or max_dual_violation > 32.0 * feasibility_tolerance
        or gap > 32.0 * max(optimality_tolerance, feasibility_tolerance, 1.0e-12) * max(1.0, abs(structural_dual_upper))
    ):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
            "small-domain compact outer-LP dual certificate exceeded tolerance",
        )
    return (
        support_multipliers,
        block_multipliers,
        fractional,
        structural_dual_upper,
        fractional_lower,
        gap,
        max_support_violation,
        max_block_violation,
        max_dual_violation,
        int(result.status),
        str(result.message),
    )


def _down_cuda(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, -math.inf))


def _up_cuda(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _add_down_cuda(left: Any, right: Any) -> Any:
    return _down_cuda(left + right)


def _add_up_cuda(left: Any, right: Any) -> Any:
    return _up_cuda(left + right)


def _sub_down_cuda(left: Any, right: Any) -> Any:
    return _down_cuda(left - right)


def _sub_up_cuda(left: Any, right: Any) -> Any:
    return _up_cuda(left - right)


def _mul_down_cuda(left: Any, right: Any) -> Any:
    raw = left * right
    lower = torch.minimum(_down_cuda(raw), _up_cuda(raw))
    upper = torch.maximum(_down_cuda(raw), _up_cuda(raw))
    return torch.minimum(lower, upper)


def _mul_up_cuda(left: Any, right: Any) -> Any:
    raw = left * right
    lower = torch.minimum(_down_cuda(raw), _up_cuda(raw))
    upper = torch.maximum(_down_cuda(raw), _up_cuda(raw))
    return torch.maximum(lower, upper)


def _div_down_cuda(left: Any, right: Any) -> Any:
    return _down_cuda(left / right)


def _gamma(operation_count: int) -> float:
    if type(operation_count) is not int or operation_count < 0:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            "gamma operation count must be a nonnegative exact integer",
        )
    unit = math.ulp(1.0) / 2.0
    product = math.nextafter(operation_count * unit, math.inf)
    if product >= 1.0:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
            "CUDA reduction gamma is not finite below one",
        )
    return math.nextafter(product / (1.0 - product), math.inf)


def _sum_error_radius_cuda(values: Any, count: int, dim: int | None) -> Any:
    absolute = torch.abs(values)
    mass = (
        torch.sum(absolute, dtype=torch.float64)
        if dim is None
        else torch.sum(absolute, dim=dim, dtype=torch.float64)
    )
    mass = _up_cuda(mass)
    if count <= 1:
        return torch.zeros_like(mass)
    gamma_value = _gamma(count - 1)
    denominator_lower = math.nextafter(1.0 - gamma_value, -math.inf)
    if denominator_lower <= 0.0:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
            "CUDA reduction gamma denominator is not positive",
        )
    gamma_tensor = torch.full_like(mass, gamma_value)
    delta_tensor = torch.full_like(
        mass,
        math.nextafter((count - 1) * _MIN_SUBNORMAL, math.inf),
    )
    numerator_upper = _add_up_cuda(_mul_up_cuda(gamma_tensor, mass), delta_tensor)
    return _up_cuda(numerator_upper / denominator_lower)


def _sum_bounds_direct_cuda(values: Any, *, dim: int | None = None) -> tuple[Any, Any]:
    count = int(values.numel()) if dim is None else int(values.shape[dim])
    if count <= 0:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            "CUDA reduction bounds require at least one term",
        )
    observed = (
        torch.sum(values, dtype=torch.float64)
        if dim is None
        else torch.sum(values, dim=dim, dtype=torch.float64)
    )
    radius = _sum_error_radius_cuda(values, count, dim)
    lower = _sub_down_cuda(observed, radius)
    upper = _add_up_cuda(observed, radius)
    if (
        not bool(torch.isfinite(lower).all().item())
        or not bool(torch.isfinite(upper).all().item())
        or bool((lower > upper).any().item())
    ):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
            "CUDA gamma-bounded reduction produced an invalid interval",
        )
    return lower.contiguous(), upper.contiguous()


def _sum_bounds_cuda(values: Any, *, dim: int | None = None) -> tuple[Any, Any]:
    if dim is None:
        flat = values.reshape(-1)
        count = int(flat.numel())
        if count <= 0:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
                "CUDA reduction bounds require at least one term",
            )
        if count <= _CUDA_REDUCTION_TILE_ELEMENTS:
            return _sum_bounds_direct_cuda(flat)
        lower = torch.zeros((), dtype=flat.dtype, device=flat.device)
        upper = torch.zeros_like(lower)
        for start in range(0, count, _CUDA_REDUCTION_TILE_ELEMENTS):
            stop = min(count, start + _CUDA_REDUCTION_TILE_ELEMENTS)
            tile_lower, tile_upper = _sum_bounds_direct_cuda(flat[start:stop])
            lower = _add_down_cuda(lower, tile_lower)
            upper = _add_up_cuda(upper, tile_upper)
        return lower.contiguous(), upper.contiguous()
    if dim == 1:
        if int(values.ndim) != 2:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
                "route-tiled CUDA row reductions require a rank-2 tensor",
            )
        row_count = int(values.shape[0])
        column_count = int(values.shape[1])
        if row_count <= 0 or column_count <= 0:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
                "CUDA row reduction bounds require a nonempty matrix",
            )
        tile_columns = max(
            1,
            min(
                column_count,
                _CUDA_REDUCTION_TILE_ELEMENTS // max(1, row_count),
            ),
        )
        if column_count <= tile_columns:
            return _sum_bounds_direct_cuda(values, dim=1)
        lower = torch.zeros((row_count,), dtype=values.dtype, device=values.device)
        upper = torch.zeros_like(lower)
        for start in range(0, column_count, tile_columns):
            stop = min(column_count, start + tile_columns)
            tile_lower, tile_upper = _sum_bounds_direct_cuda(
                values[:, start:stop],
                dim=1,
            )
            lower = _add_down_cuda(lower, tile_lower)
            upper = _add_up_cuda(upper, tile_upper)
        return lower.contiguous(), upper.contiguous()
    return _sum_bounds_direct_cuda(values, dim=dim)


def _sum_upper_cuda(values: Any, *, dim: int | None = None) -> Any:
    return _sum_bounds_cuda(values, dim=dim)[1]


def _product_sum_bounds_cuda(left: Any, right: Any, *, dim: int | None = None) -> tuple[Any, Any]:
    left_view, right_view = torch.broadcast_tensors(left, right)
    if dim is None:
        flat_left = left_view.reshape(-1)
        flat_right = right_view.reshape(-1)
        count = int(flat_left.numel())
        if count <= 0:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
                "CUDA product reduction bounds require at least one term",
            )
        lower = torch.zeros((), dtype=flat_left.dtype, device=flat_left.device)
        upper = torch.zeros_like(lower)
        for start in range(0, count, _CUDA_REDUCTION_TILE_ELEMENTS):
            stop = min(count, start + _CUDA_REDUCTION_TILE_ELEMENTS)
            lower_terms = _mul_down_cuda(flat_left[start:stop], flat_right[start:stop])
            tile_lower, _ = _sum_bounds_cuda(lower_terms)
            del lower_terms
            upper_terms = _mul_up_cuda(flat_left[start:stop], flat_right[start:stop])
            _, tile_upper = _sum_bounds_cuda(upper_terms)
            del upper_terms
            lower = _add_down_cuda(lower, tile_lower)
            upper = _add_up_cuda(upper, tile_upper)
        return lower.contiguous(), upper.contiguous()
    if dim == 1:
        if int(left_view.ndim) != 2:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
                "route-tiled CUDA product row reductions require rank-2 operands",
            )
        row_count = int(left_view.shape[0])
        column_count = int(left_view.shape[1])
        if row_count <= 0 or column_count <= 0:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
                "CUDA product row reduction bounds require a nonempty matrix",
            )
        tile_columns = max(
            1,
            min(
                column_count,
                _CUDA_REDUCTION_TILE_ELEMENTS // max(1, row_count),
            ),
        )
        lower = torch.zeros((row_count,), dtype=left_view.dtype, device=left_view.device)
        upper = torch.zeros_like(lower)
        for start in range(0, column_count, tile_columns):
            stop = min(column_count, start + tile_columns)
            lower_terms = _mul_down_cuda(
                left_view[:, start:stop],
                right_view[:, start:stop],
            )
            tile_lower, _ = _sum_bounds_cuda(lower_terms, dim=1)
            del lower_terms
            upper_terms = _mul_up_cuda(
                left_view[:, start:stop],
                right_view[:, start:stop],
            )
            _, tile_upper = _sum_bounds_cuda(upper_terms, dim=1)
            del upper_terms
            lower = _add_down_cuda(lower, tile_lower)
            upper = _add_up_cuda(upper, tile_upper)
        return lower.contiguous(), upper.contiguous()
    lower_terms = _mul_down_cuda(left_view, right_view)
    lower, _ = _sum_bounds_cuda(lower_terms, dim=dim)
    del lower_terms
    upper_terms = _mul_up_cuda(left_view, right_view)
    _, upper = _sum_bounds_cuda(upper_terms, dim=dim)
    del upper_terms
    return lower.contiguous(), upper.contiguous()


def _square_sum_upper_cuda(values: Any) -> Any:
    flat = values.reshape(-1)
    count = int(flat.numel())
    if count <= 0:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            "CUDA square-sum upper bound requires a nonempty tensor",
        )
    upper = torch.zeros((), dtype=flat.dtype, device=flat.device)
    for start in range(0, count, _CUDA_REDUCTION_TILE_ELEMENTS):
        stop = min(count, start + _CUDA_REDUCTION_TILE_ELEMENTS)
        square_upper_terms = _mul_up_cuda(flat[start:stop], flat[start:stop])
        _, tile_upper = _sum_bounds_cuda(square_upper_terms)
        del square_upper_terms
        upper = _add_up_cuda(upper, tile_upper)
    return upper.contiguous()


def _block_lhs_cuda(
    x: Any,
    blocks: tuple[Any, ...],
    *,
    dtype: Any,
    device: Any,
) -> Any:
    if not blocks:
        return torch.empty((0,), dtype=dtype, device=device)
    values = [
        torch.sum(x[block.start : block.stop], dtype=dtype)
        for block in blocks
    ]
    return torch.stack(values).contiguous()


def _add_block_dual_cuda(
    target: Any,
    beta: Any,
    blocks: tuple[Any, ...],
) -> None:
    for ordinal, block in enumerate(blocks):
        target[block.start : block.stop].add_(beta[ordinal])


def _constraint_lhs_cuda(
    coefficient: Any,
    x: Any,
    blocks: tuple[Any, ...],
) -> tuple[Any, Any]:
    support_lhs = torch.mv(coefficient, x)
    block_lhs = _block_lhs_cuda(
        x,
        blocks,
        dtype=coefficient.dtype,
        device=coefficient.device,
    )
    return support_lhs.contiguous(), block_lhs


def _block_lhs_bounds_cuda(
    x: Any,
    blocks: tuple[Any, ...],
    *,
    dtype: Any,
    device: Any,
) -> tuple[Any, Any]:
    if not blocks:
        empty = torch.empty((0,), dtype=dtype, device=device)
        return empty, empty
    lowers: list[Any] = []
    uppers: list[Any] = []
    for block in blocks:
        lower, upper = _sum_bounds_cuda(x[block.start : block.stop])
        lowers.append(lower)
        uppers.append(upper)
    return torch.stack(lowers).contiguous(), torch.stack(uppers).contiguous()


def _constraint_lhs_bounds_cuda(
    coefficient: Any,
    x: Any,
    blocks: tuple[Any, ...],
) -> tuple[Any, Any, Any, Any]:
    support_lower, support_upper = _product_sum_bounds_cuda(
        coefficient,
        x.reshape(1, -1),
        dim=1,
    )
    block_lower, block_upper = _block_lhs_bounds_cuda(
        x,
        blocks,
        dtype=coefficient.dtype,
        device=coefficient.device,
    )
    return (
        support_lower.contiguous(),
        support_upper.contiguous(),
        block_lower.contiguous(),
        block_upper.contiguous(),
    )


def _feasible_scale_cuda(
    support_lhs_upper: Any,
    block_lhs_upper: Any,
    rhs: Any,
    capacities: Any,
) -> Any:
    scale = torch.ones((), dtype=rhs.dtype, device=rhs.device)
    support_positive = support_lhs_upper > rhs
    if bool(torch.any(support_positive).item()):
        ratios = _div_down_cuda(
            rhs[support_positive],
            support_lhs_upper[support_positive],
        )
        scale = torch.minimum(scale, torch.clamp(torch.amin(ratios), min=0.0, max=1.0))
    if int(capacities.numel()):
        block_positive = block_lhs_upper > capacities
        if bool(torch.any(block_positive).item()):
            ratios = _div_down_cuda(
                capacities[block_positive],
                block_lhs_upper[block_positive],
            )
            scale = torch.minimum(scale, torch.clamp(torch.amin(ratios), min=0.0, max=1.0))
    return scale.contiguous()


def _dual_objective_cuda(
    *,
    coefficient: Any,
    resource: Any,
    eligible: Any,
    rhs: Any,
    capacities: Any,
    support_multipliers: Any,
    block_multipliers: Any,
    blocks: tuple[Any, ...],
) -> tuple[Any, Any]:
    weighted = torch.mv(coefficient.t(), support_multipliers)
    if int(block_multipliers.numel()):
        _add_block_dual_cuda(weighted, block_multipliers, blocks)
    advantage = resource - weighted
    slack = torch.where(
        eligible,
        torch.clamp(advantage, min=0.0),
        torch.zeros_like(advantage),
    )
    compact = torch.dot(rhs, support_multipliers)
    if int(capacities.numel()):
        compact = compact + torch.dot(capacities, block_multipliers)
    dual = compact + torch.sum(slack, dtype=torch.float64)
    dual_violation = torch.clamp(advantage - slack, min=0.0)
    dual_violation = torch.where(eligible, dual_violation, torch.zeros_like(dual_violation))
    return dual.contiguous(), torch.amax(dual_violation).contiguous()


def _nonnegative_dot_bounds_cuda(left: Any, right: Any) -> tuple[Any, Any]:
    return _product_sum_bounds_cuda(left, right)


def _dual_objective_upper_cuda(
    *,
    coefficient: Any,
    resource: Any,
    eligible: Any,
    rhs: Any,
    capacities: Any,
    support_multipliers: Any,
    block_multipliers: Any,
    blocks: tuple[Any, ...],
) -> tuple[Any, Any]:
    weighted_lower = torch.zeros_like(resource)
    for direction in range(int(support_multipliers.numel())):
        lam = support_multipliers[direction]
        weighted_lower = _add_down_cuda(
            weighted_lower,
            _mul_down_cuda(lam, coefficient[direction]),
        )
    if int(block_multipliers.numel()):
        for ordinal, block in enumerate(blocks):
            weighted_lower[block.start : block.stop] = _add_down_cuda(
                weighted_lower[block.start : block.stop],
                block_multipliers[ordinal],
            )
    advantage_upper = _sub_up_cuda(resource, weighted_lower)
    slack_upper_terms = torch.where(
        eligible,
        torch.clamp(advantage_upper, min=0.0),
        torch.zeros_like(advantage_upper),
    )
    slack_upper = _sum_upper_cuda(slack_upper_terms)
    _rhs_lower, rhs_upper = _nonnegative_dot_bounds_cuda(rhs, support_multipliers)
    compact_upper = rhs_upper
    if int(capacities.numel()):
        _cap_lower, cap_upper = _nonnegative_dot_bounds_cuda(
            capacities,
            block_multipliers,
        )
        compact_upper = _add_up_cuda(compact_upper, cap_upper)
    dual_upper = _add_up_cuda(compact_upper, slack_upper)
    return dual_upper.contiguous(), torch.zeros((), dtype=torch.float64, device=resource.device)


def _operator_norm_upper_cuda(
    *,
    coefficient: Any,
    blocks: tuple[Any, ...],
) -> Any:
    support_frobenius_square_upper = _square_sum_upper_cuda(coefficient)
    block_square_mass = sum(int(block.stop - block.start) for block in blocks)
    total_square_upper = _add_up_cuda(
        support_frobenius_square_upper,
        torch.as_tensor(
            float(block_square_mass),
            dtype=torch.float64,
            device=coefficient.device,
        ),
    )
    total_square_upper = torch.clamp(total_square_upper, min=1.0)
    operator_upper = _up_cuda(torch.sqrt(total_square_upper))
    if not bool(torch.isfinite(operator_upper).item()):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
            "CUDA operator norm Frobenius envelope overflowed",
        )
    return operator_upper.contiguous()


def _solve_cuda_lp(
    *,
    cuda_source: V1PhaseAGlobalQCompactOuterLPCUDASource,
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    design: V1PhaseAGlobalQCompactOuterLPDualDesign,
) -> tuple[Any, Any, str, float, float, float, float, float, float, int, str, int, int]:
    if torch is None:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.CUDA_UNAVAILABLE,
            "compact outer-LP CUDA solve requires PyTorch",
        )
    if not bool(torch.cuda.is_available()):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.CUDA_UNAVAILABLE,
            "compact outer-LP CUDA solve requires an available CUDA runtime",
        )
    if not bool(torch.are_deterministic_algorithms_enabled()):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.DETERMINISM_DISABLED,
            "compact outer-LP CUDA solve requires deterministic torch algorithms",
        )
    coefficient, _coefficient_upper, resource, eligible, _block_ordinals, rhs, capacities = cuda_source.tensors()
    if not bool(torch.any(eligible).item()):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
            "compact outer-LP CUDA solve requires at least one eligible route",
        )
    blocks = tuple(relaxed_kkt_problem.block_capacities)
    route_count = int(resource.numel())
    direction_count = int(rhs.numel())
    block_count = int(capacities.numel())
    device = resource.device
    max_iterations = _positive_int(design.cuda_max_iterations, name="cuda_max_iterations")
    min_iterations = _nonnegative_int(design.cuda_min_iterations, name="cuda_min_iterations")
    check_interval = _positive_int(design.cuda_check_interval, name="cuda_check_interval")
    feasibility_tolerance = _finite(design.feasibility_tolerance, name="feasibility_tolerance", nonnegative=True)
    optimality_tolerance = _finite(design.optimality_tolerance, name="optimality_tolerance", nonnegative=True)
    norm_safety = _positive_float(design.cuda_operator_norm_safety, name="cuda_operator_norm_safety")
    operator_norm = _operator_norm_upper_cuda(coefficient=coefficient, blocks=blocks)
    step = torch.as_tensor(norm_safety, dtype=torch.float64, device=device) / operator_norm
    x = torch.zeros((route_count,), dtype=torch.float64, device=device)
    x_bar = torch.zeros_like(x)
    y = torch.zeros((direction_count + block_count,), dtype=torch.float64, device=device)
    eligible_upper = eligible.to(dtype=torch.float64)
    best: tuple[float, float, float, float, float, float, int, str, int] | None = None
    checkpoint_count = 0
    coefficient_bytes_per_iteration = (
        2 * route_count * direction_count * _F64_BYTES
    )
    for iteration in range(1, max_iterations + 1):
        support_lhs, block_lhs = _constraint_lhs_cuda(coefficient, x_bar, blocks)
        compact_lhs = torch.cat((support_lhs, block_lhs))
        compact_rhs = torch.cat((rhs, capacities))
        y = torch.clamp(y + step * (compact_lhs - compact_rhs), min=0.0)
        support_multipliers = y[:direction_count]
        block_multipliers = y[direction_count:]
        adjoint = torch.mv(coefficient.t(), support_multipliers)
        if int(block_multipliers.numel()):
            _add_block_dual_cuda(adjoint, block_multipliers, blocks)
        x_next = torch.minimum(
            torch.clamp(x + step * (resource - adjoint), min=0.0),
            eligible_upper,
        )
        x_bar = x_next + (x_next - x)
        x = x_next
        if iteration < min_iterations or iteration % check_interval:
            continue
        checkpoint_count += 1
        (
            _support_lhs_lower,
            support_lhs_upper,
            _block_lhs_lower,
            block_lhs_upper,
        ) = _constraint_lhs_bounds_cuda(coefficient, x, blocks)
        alpha = _feasible_scale_cuda(support_lhs_upper, block_lhs_upper, rhs, capacities)
        primal_unscaled_lower, _primal_unscaled_upper = _nonnegative_dot_bounds_cuda(resource, x)
        primal_lower_tensor = _mul_down_cuda(alpha, primal_unscaled_lower)
        scaled_support_upper = _mul_up_cuda(support_lhs_upper, alpha)
        scaled_block_upper = _mul_up_cuda(block_lhs_upper, alpha)
        max_support_violation = torch.zeros((), dtype=torch.float64, device=device)
        if direction_count:
            max_support_violation = torch.clamp(
                torch.amax(_sub_up_cuda(scaled_support_upper, rhs)),
                min=0.0,
            )
        max_block_violation = torch.zeros((), dtype=torch.float64, device=device)
        if int(capacities.numel()):
            max_block_violation = torch.clamp(
                torch.amax(_sub_up_cuda(scaled_block_upper, capacities)),
                min=0.0,
            )
        dual_upper, dual_violation = _dual_objective_upper_cuda(
            coefficient=coefficient,
            resource=resource,
            eligible=eligible,
            rhs=rhs,
            capacities=capacities,
            support_multipliers=support_multipliers,
            block_multipliers=block_multipliers,
            blocks=blocks,
        )
        primal_lower = _down_cuda(primal_lower_tensor)
        gap = _up_cuda(torch.clamp(_sub_up_cuda(dual_upper, primal_lower), min=0.0))
        scan_equivalent_count = 1 + 2 * iteration + 3 * checkpoint_count
        scale = max(1.0, abs(float(dual_upper.item())))
        gap_value = float(gap.item())
        best = (
            float(dual_upper.item()),
            float(primal_lower.item()),
            gap_value,
            float(max_support_violation.item()),
            float(max_block_violation.item()),
            float(dual_violation.item()),
            iteration,
            f"cuda_pdhg_iteration_{iteration}",
            scan_equivalent_count,
        )
        if (
            best[3] <= 32.0 * feasibility_tolerance
            and best[4] <= 32.0 * feasibility_tolerance
            and best[5] <= 32.0 * feasibility_tolerance
            and gap_value
            <= 32.0 * max(optimality_tolerance, feasibility_tolerance, 1.0e-12) * scale
        ):
            support_tensor = support_multipliers.detach().clone().contiguous()
            block_tensor = block_multipliers.detach().clone().contiguous()
            primal_digest = _payload_hash(
                {
                    "source_receipt_hash": cuda_source.receipt.receipt_hash,
                    "iteration": iteration,
                    "feasible_scaling_alpha": float(alpha.item()),
                    "fractional_primal_lower": best[1],
                    "max_fractional_support_violation": best[3],
                    "max_fractional_block_violation": best[4],
                    "gamma_bounded_reductions": True,
                    "operator_norm_frobenius_upper_bound": True,
                    "route_scale_primal_content_not_d2h": True,
                },
                role="cuda-primal-witness",
            )
            return (
                support_tensor,
                block_tensor,
                primal_digest,
                best[0],
                best[1],
                best[2],
                best[3],
                best[4],
                best[5],
                0,
                best[7],
                best[8],
                coefficient_bytes_per_iteration,
            )
    if best is None:
        message = "compact outer-LP CUDA solve produced no convergence checkpoint"
    else:
        message = (
            "compact outer-LP CUDA solve did not close the registered gap: "
            f"iteration={best[6]} gap={best[2]:.17g} dual_upper={best[0]:.17g} "
            f"primal_lower={best[1]:.17g}"
        )
    _refuse(V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NONCONVERGENCE, message)


def solve_v1_phase_a_global_q_compact_outer_lp_dual(
    *,
    dense_source: V1PhaseAGlobalQCompactOuterLPDenseSource | V1PhaseAGlobalQCompactOuterLPCUDASource,
    support_family: (
        V1PhaseAGlobalQSupportFamilyResult
        | V1PhaseAGlobalQD1SupportFamilyResult
    ),
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    design: V1PhaseAGlobalQCompactOuterLPDualDesign = V1PhaseAGlobalQCompactOuterLPDualDesign(),
) -> V1PhaseAGlobalQCompactOuterLPDualSolveResult:
    if (
        type(dense_source)
        not in {
            V1PhaseAGlobalQCompactOuterLPDenseSource,
            V1PhaseAGlobalQCompactOuterLPCUDASource,
        }
        or type(relaxed_kkt_problem) is not V1RelaxedKKTProblem
        or type(design) is not V1PhaseAGlobalQCompactOuterLPDualDesign
    ):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            "compact outer-LP dual solve requires exact typed source/problem/design",
        )
    _support_family_source_mode(support_family)
    if design.execution_mode is V1PhaseAGlobalQCompactOuterLPDualExecutionMode.CUDA_PRODUCTION:
        if type(dense_source) is not V1PhaseAGlobalQCompactOuterLPCUDASource:
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.DEVICE_REFUSAL,
                "production CUDA compact outer-LP dual solve requires its CUDA source",
            )
        support_family.device_handle.assert_live()
        source_hash = dense_source.assert_live()
        support_hash = support_family.receipt.assert_integrity()
        source_receipt = dense_source.receipt
        if (
            source_receipt.support_family_receipt_hash != support_hash
            or source_receipt.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash
            or support_family.receipt.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash
            or source_receipt.route_axis_hash != relaxed_kkt_problem.route_axis_hash
            or source_receipt.route_count != relaxed_kkt_problem.route_count
            or source_receipt.support_plane_count != support_family.receipt.anchor_count
            or source_receipt.block_count != len(relaxed_kkt_problem.block_capacities)
            or source_receipt.production_cuda_authority is not True
            or source_receipt.private_cpu_reference is not False
        ):
            _refuse(
                V1PhaseAGlobalQCompactOuterLPDualRefusalCode.SOURCE_MISMATCH,
                "compact outer-LP CUDA source, support family, and problem do not replay one source graph",
            )
        (
            support_tensor,
            block_tensor,
            primal_digest,
            structural_upper,
            fractional_lower,
            gap,
            support_violation,
            block_violation,
            dual_violation,
            scipy_status,
            scipy_message,
            scan_equivalent_count,
            coefficient_bytes_per_iteration,
        ) = _solve_cuda_lp(
            cuda_source=dense_source,
            relaxed_kkt_problem=relaxed_kkt_problem,
            design=design,
        )
        receipt = V1PhaseAGlobalQCompactOuterLPDualSolveReceipt(
            design_hash=design.design_hash,
            dense_source_receipt_hash=source_hash,
            support_family_receipt_hash=support_hash,
            relaxed_kkt_problem_hash=relaxed_kkt_problem.problem_hash,
            route_axis_hash=source_receipt.route_axis_hash,
            current_mask_hash=source_receipt.current_mask_hash,
            route_count=source_receipt.route_count,
            support_plane_count=source_receipt.support_plane_count,
            block_count=source_receipt.block_count,
            support_multiplier_digest=_compact_tensor_digest(
                support_tensor,
                role="support-multipliers",
            ),
            block_multiplier_digest=_compact_tensor_digest(
                block_tensor,
                role="block-multipliers",
            ),
            primal_fractional_witness_digest=primal_digest,
            structural_dual_upper=structural_upper,
            fractional_primal_lower=fractional_lower,
            primal_dual_gap_upper=gap,
            max_fractional_support_violation=support_violation,
            max_fractional_block_violation=block_violation,
            max_dual_constraint_violation=dual_violation,
            scipy_status=scipy_status,
            scipy_message=scipy_message,
            cuda_coefficient_scan_equivalent_count=scan_equivalent_count,
            cuda_coefficient_bytes_per_pdhg_iteration=coefficient_bytes_per_iteration,
            cuda_gamma_bounded_reduction_authority=True,
            cuda_operator_norm_frobenius_bound=True,
            production_cuda_execution=True,
            small_domain_cpu_reference=False,
            scipy_highs_certificate_used=False,
            algorithm=PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_ALGORITHM,
            _issuance_token=_RECEIPT_TOKEN,
        )
        return V1PhaseAGlobalQCompactOuterLPDualSolveResult(
            receipt=receipt,
            device_handle=V1PhaseAGlobalQCompactOuterLPDualHandle(
                receipt=receipt,
                support_multipliers=support_tensor,
                block_multipliers=block_tensor,
                _issuance_token=_HANDLE_TOKEN,
            ),
        )
    if design.execution_mode is not V1PhaseAGlobalQCompactOuterLPDualExecutionMode.SMALL_DOMAIN_CPU_REFERENCE:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            "unknown compact outer-LP dual execution mode",
        )
    if type(dense_source) is not V1PhaseAGlobalQCompactOuterLPDenseSource:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.DEVICE_REFUSAL,
            "small-domain CPU reference compact outer-LP dual solve requires its dense CPU source",
        )
    support_family.device_handle.assert_live()
    source_hash = dense_source.assert_integrity()
    support_hash = support_family.receipt.assert_integrity()
    source_receipt = dense_source.receipt
    if (
        source_receipt.support_family_receipt_hash != support_hash
        or source_receipt.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash
        or support_family.receipt.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash
        or source_receipt.route_axis_hash != relaxed_kkt_problem.route_axis_hash
        or source_receipt.route_count != relaxed_kkt_problem.route_count
        or source_receipt.support_plane_count != support_family.receipt.anchor_count
        or source_receipt.block_count != len(relaxed_kkt_problem.block_capacities)
    ):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.SOURCE_MISMATCH,
            "compact outer-LP dense source, support family, and problem do not replay one source graph",
        )
    if source_receipt.route_count > design.cpu_reference_route_limit:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.ROUTE_LIMIT_EXCEEDED,
            "small-domain compact outer-LP dual reference route cap exceeded",
        )
    coefficient, resource, eligible, block_ordinals, rhs, capacities = dense_source.arrays()
    if eligible.shape != (source_receipt.route_count,) or not np.any(eligible):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.NUMERICAL_REFUSAL,
            "compact outer-LP dual solve requires at least one eligible route",
        )
    solved = _solve_small_domain_lp(
        coefficient=coefficient[:, eligible],
        resource=resource[eligible],
        block_ordinals=block_ordinals[eligible],
        rhs=rhs,
        capacities=capacities,
        feasibility_tolerance=design.feasibility_tolerance,
        optimality_tolerance=design.optimality_tolerance,
    )
    (
        support_np,
        block_np,
        fractional_np,
        structural_upper,
        fractional_lower,
        gap,
        support_violation,
        block_violation,
        dual_violation,
        scipy_status,
        scipy_message,
    ) = solved
    support_tensor = torch.tensor(support_np, dtype=torch.float64).contiguous()
    block_tensor = torch.tensor(block_np, dtype=torch.float64).contiguous()
    fractional = torch.tensor(fractional_np, dtype=torch.float64).contiguous()
    receipt = V1PhaseAGlobalQCompactOuterLPDualSolveReceipt(
        design_hash=design.design_hash,
        dense_source_receipt_hash=source_hash,
        support_family_receipt_hash=support_hash,
        relaxed_kkt_problem_hash=relaxed_kkt_problem.problem_hash,
        route_axis_hash=source_receipt.route_axis_hash,
        current_mask_hash=source_receipt.current_mask_hash,
        route_count=source_receipt.route_count,
        support_plane_count=source_receipt.support_plane_count,
        block_count=source_receipt.block_count,
        support_multiplier_digest=_compact_tensor_digest(support_tensor, role="support-multipliers"),
        block_multiplier_digest=_compact_tensor_digest(block_tensor, role="block-multipliers"),
        primal_fractional_witness_digest=_compact_tensor_digest(fractional, role="fractional-primal"),
        structural_dual_upper=structural_upper,
        fractional_primal_lower=fractional_lower,
        primal_dual_gap_upper=gap,
        max_fractional_support_violation=support_violation,
        max_fractional_block_violation=block_violation,
        max_dual_constraint_violation=dual_violation,
        scipy_status=scipy_status,
        scipy_message=scipy_message,
        _issuance_token=_RECEIPT_TOKEN,
    )
    return V1PhaseAGlobalQCompactOuterLPDualSolveResult(
        receipt=receipt,
        device_handle=V1PhaseAGlobalQCompactOuterLPDualHandle(
            receipt=receipt,
            support_multipliers=support_tensor,
            block_multipliers=block_tensor,
            _issuance_token=_HANDLE_TOKEN,
        ),
    )


def issue_v1_phase_a_global_q_outer_lp_dual_proposal_from_compact_solve(
    *,
    compact_dual_solve: V1PhaseAGlobalQCompactOuterLPDualSolveResult,
    support_family: (
        V1PhaseAGlobalQSupportFamilyResult
        | V1PhaseAGlobalQD1SupportFamilyResult
    ),
    relaxed_kkt_problem: V1RelaxedKKTProblem,
    action_pair_alias_source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
    robust_shell_decision: V1SampledV7PhaseARobustShellDecisionReceipt,
    parent_mask: HardMask,
    allow_private_cpu_reference_for_tests: bool = False,
) -> V1PhaseAGlobalQOuterLPDualProposalResult:
    """Issue a frozen dual proposal without crossing its support source mode."""

    if type(compact_dual_solve) is not V1PhaseAGlobalQCompactOuterLPDualSolveResult:
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.INVALID_TYPE,
            "dual proposal bridge requires one typed compact dual solve",
        )
    _support_family_source_mode(support_family)
    compact_dual_solve.device_handle.assert_live()
    receipt = compact_dual_solve.receipt
    support_hash = support_family.receipt.assert_integrity()
    if (
        receipt.support_family_receipt_hash != support_hash
        or receipt.relaxed_kkt_problem_hash != relaxed_kkt_problem.problem_hash
        or receipt.route_axis_hash != relaxed_kkt_problem.route_axis_hash
        or receipt.current_mask_hash != parent_mask.mask_hash
    ):
        _refuse(
            V1PhaseAGlobalQCompactOuterLPDualRefusalCode.SOURCE_MISMATCH,
            "typed compact dual solve cannot be bridged into a different proposal source graph",
        )
    with compact_dual_solve.device_handle.borrow() as lease:
        support_multipliers = lease.support_multipliers.detach().clone().contiguous()
        block_multipliers = lease.block_multipliers.detach().clone().contiguous()
    return issue_v1_phase_a_global_q_outer_lp_dual_proposal(
        support_family=support_family,
        relaxed_kkt_problem=relaxed_kkt_problem,
        action_pair_alias_source_join=action_pair_alias_source_join,
        robust_shell_decision=robust_shell_decision,
        parent_mask=parent_mask,
        support_multipliers=support_multipliers,
        block_multipliers=block_multipliers,
        allow_private_cpu_reference_for_tests=allow_private_cpu_reference_for_tests,
    )


__all__ = (
    "PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_ALGORITHM",
    "PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_AUTHORITY",
    "PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_ALGORITHM",
    "PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_HBM_FORECAST_SCHEMA",
    "PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_READINESS_ALGORITHM",
    "PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_SOURCE_ALGORITHM",
    "PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_CUDA_SOURCE_SCHEMA",
    "PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_READINESS_SCHEMA",
    "PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_SOLVE_SCHEMA",
    "PHASE_A_GLOBAL_Q_COMPACT_OUTER_LP_DUAL_SOURCE_SCHEMA",
    "V1PhaseAGlobalQCompactOuterLPCUDAHBMForecast",
    "V1PhaseAGlobalQCompactOuterLPCUDASource",
    "V1PhaseAGlobalQCompactOuterLPCUDASourceReceipt",
    "V1PhaseAGlobalQCompactOuterLPDenseSource",
    "V1PhaseAGlobalQCompactOuterLPDenseSourceReceipt",
    "V1PhaseAGlobalQCompactOuterLPDualDesign",
    "V1PhaseAGlobalQCompactOuterLPDualExecutionMode",
    "V1PhaseAGlobalQCompactOuterLPDualHandle",
    "V1PhaseAGlobalQCompactOuterLPDualLease",
    "V1PhaseAGlobalQCompactOuterLPDualReadinessReceipt",
    "V1PhaseAGlobalQCompactOuterLPDualRefusal",
    "V1PhaseAGlobalQCompactOuterLPDualRefusalCode",
    "V1PhaseAGlobalQCompactOuterLPDualSolveReceipt",
    "V1PhaseAGlobalQCompactOuterLPDualSolveResult",
    "build_v1_phase_a_global_q_compact_outer_lp_cuda_hbm_forecast",
    "issue_v1_phase_a_global_q_compact_outer_lp_cuda_source_from_scan",
    "issue_v1_phase_a_global_q_compact_outer_lp_dense_source_for_tests",
    "issue_v1_phase_a_global_q_compact_outer_lp_dual_production_readiness",
    "issue_v1_phase_a_global_q_outer_lp_dual_proposal_from_compact_solve",
    "solve_v1_phase_a_global_q_compact_outer_lp_dual",
)
