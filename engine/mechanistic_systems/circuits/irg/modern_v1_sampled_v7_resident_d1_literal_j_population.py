"""Resident exact-D1 literal-J geometry for the V7 residual-population join.

The V7 fixed-primal census already computes, for the two canonical relaxed
KKT directions, the outward routewise interval for ``2 M_J z``.  Replaying
the sealed D1 rows merely to reconstruct ``M_J z`` would be both wasteful and
causally awkward.  This module therefore captures those two live projections
*inside* the fixed-primal consumer, derives the matching quadratic response

``Q(z) = 1/2 z.T (2 M_J z)``,

and transfers the four route-scale interval vectors into one process-local,
nonserializable owner before the source projections are evicted.  The owner
is a geometric handoff only.  It cannot select an action, open C/K/O, commit a
mask, or claim terminal/Pareto authority.

Production capture is bound to the exact spectral--diagonal KKT executions
and the exact private-primal bridge.  A small CPU sibling exists only for
focused numerical tests.  No preferred hard action enters either interface.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Any, Final, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_fixed_primal_projection_cuda import (
    V1SignedWorkFixedPrimalProjectionCUDAExecution,
)
from .modern_v1_signed_work_kkt_cuda import (
    V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
)
from .modern_v1_signed_work_literal_j_private_primal_bridge_cuda import (
    V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecution,
)

try:  # pragma: no cover - runtime image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SAMPLED_V7_RESIDENT_D1_LITERAL_J_POPULATION_DESIGN_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_resident_D1_literal_J_population_design_v1"
)
SAMPLED_V7_RESIDENT_D1_LITERAL_J_POPULATION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_resident_D1_literal_J_population_receipt_v1"
)
SAMPLED_V7_RESIDENT_D1_LITERAL_J_POPULATION_AUTHORITY: Final = (
    "exact_frozen_D1_literal_J_two_MJz_and_Q_live_handoff_for_d_over_N_"
    "residual_population_join_only_no_rank_face_action_stability_C_K_O_commit_"
    "terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-resident-D1-literal-j-population-"
_ISSUER: Final = object()
_F64_BYTES: Final = 8
_F64_EPS: Final = 2.0**-53
_F64_MIN_SUBNORMAL: Final = float.fromhex("0x0.0000000000001p-1022")


class V1SampledV7ResidentD1LiteralJPopulationRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_LINEAGE = "stale_lineage"
    BRANCH_ORDER_MISMATCH = "branch_order_mismatch"
    NONCENSUS_SOURCE = "noncensus_source"
    DEVICE_REQUIRED = "device_required"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    DETERMINISM_DISABLED = "determinism_disabled"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    MUTATED_CUSTODY = "mutated_custody"
    ACTIVE_LEASE = "active_lease"
    REUSED_OR_EVICTED = "reused_or_evicted"
    NUMERICAL_REFUSAL = "numerical_refusal"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7ResidentD1LiteralJPopulationRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7ResidentD1LiteralJPopulationRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7ResidentD1LiteralJPopulationRefusalCode:
            raise TypeError("resident-D1 refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7ResidentD1LiteralJPopulationRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7ResidentD1LiteralJPopulationRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return int(value)


def _finite_fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 0.85:
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.INVALID_TYPE,
            f"{name} must lie in (0, 0.85]",
        )
    return result


def _gamma(operation_count: int) -> float:
    count = _positive_int(operation_count, name="roundoff operation count")
    product = count * _F64_EPS
    if product >= 0.5:
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.NUMERICAL_REFUSAL,
            "the registered reduction is too long for a finite binary64 enclosure",
        )
    return math.nextafter(product / (1.0 - product), math.inf)


def _up(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, math.inf))


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
        bool(value.requires_grad),
        bool(value.is_contiguous()),
    )


def _require_fp64_vector(
    value: Any,
    *,
    name: str,
    route_count: int,
    device: Any | None = None,
) -> Any:
    if (
        torch is None
        or type(value) is not torch.Tensor
        or value.dtype != torch.float64
        or value.ndim != 1
        or int(value.shape[0]) != route_count
        or not value.is_contiguous()
        or value.requires_grad
        or (device is not None and value.device != device)
    ):
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.INVALID_TYPE,
            f"{name} must be one contiguous FP64 route vector on the bound device",
        )
    return value


def _scalar_fingerprint(value: Any, *, role: str) -> str:
    if torch is None or type(value) is not torch.Tensor or value.dtype != torch.float64:
        raise TypeError("resident scalar fingerprints require exact FP64 tensors")
    flat = value.detach().reshape(-1)
    payload = tuple(float(item).hex() for item in flat.cpu().tolist())
    if any(not math.isfinite(float.fromhex(item)) for item in payload):
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.NUMERICAL_REFUSAL,
            f"{role} contains nonfinite output",
        )
    return content_hash(
        {"role": role, "shape": tuple(int(item) for item in value.shape), "hex": payload},
        prefix=_PREFIX + "scalar-fingerprint-",
    )


def _outward_q_from_two_mjz(
    *,
    fixed_primal: Any,
    two_mjz_center: Any,
    two_mjz_radius: Any,
    route_tile_size: int,
) -> tuple[Any, Any]:
    """Enclose ``Q = .5 z.T (2 M_J z)`` without a context-row replay."""

    routes = _positive_int(int(fixed_primal.numel()), name="route count")
    tile = _positive_int(route_tile_size, name="route tile size")
    z = _require_fp64_vector(
        fixed_primal, name="fixed primal", route_count=routes
    )
    center = _require_fp64_vector(
        two_mjz_center,
        name="two-MJz center",
        route_count=routes,
        device=z.device,
    )
    radius = _require_fp64_vector(
        two_mjz_radius,
        name="two-MJz radius",
        route_count=routes,
        device=z.device,
    )
    if bool((radius < 0.0).any().item()):
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.NUMERICAL_REFUSAL,
            "two-MJz interval radius is negative",
        )
    dot = torch.zeros((), dtype=torch.float64, device=z.device)
    absolute_mass = torch.zeros_like(dot)
    source_uncertainty = torch.zeros_like(dot)
    tile_count = math.ceil(routes / tile)
    for start in range(0, routes, tile):
        stop = min(routes, start + tile)
        z_tile = z[start:stop]
        product = z_tile * center[start:stop]
        uncertainty = torch.abs(z_tile) * radius[start:stop]
        dot.add_(torch.sum(product))
        absolute_mass.add_(torch.sum(torch.abs(product)))
        source_uncertainty.add_(torch.sum(uncertainty))
        del product, uncertainty
    # Each of dot, absolute-mass, and source-uncertainty traverses one route
    # multiply, a tile reduction, and a deterministic scalar accumulation.
    # Recover an upper bound on the *exact* nonnegative sums by dividing the
    # rounded observations by (1-gamma), rather than treating their rounded
    # values as upper bounds.  The absolute term separately covers gradual
    # underflow, where a purely relative gamma model is inapplicable.
    path_operations = 2 * routes + 2 * tile_count + 8
    path_gamma = _gamma(path_operations)
    inverse_one_minus_gamma = math.nextafter(
        1.0 / (1.0 - path_gamma),
        math.inf,
    )
    absolute_underflow = math.nextafter(
        path_operations * _F64_MIN_SUBNORMAL,
        math.inf,
    )
    absolute_mass_upper = _up(
        _up(absolute_mass + absolute_underflow) * inverse_one_minus_gamma
    )
    source_uncertainty_upper = _up(
        _up(source_uncertainty + absolute_underflow)
        * inverse_one_minus_gamma
    )
    q_center = (0.5 * dot).reshape(())
    q_source_radius = _up(
        0.5 * source_uncertainty_upper + absolute_underflow
    )
    q_arithmetic_radius = _up(
        0.5
        * _up(absolute_mass_upper * path_gamma + absolute_underflow)
        + absolute_underflow
    )
    q_radius = _up(q_source_radius + q_arithmetic_radius)
    if (
        not bool(torch.isfinite(q_center).item())
        or not bool(torch.isfinite(q_radius).item())
        or bool((q_radius < 0.0).item())
        or bool((q_center + q_radius < 0.0).item())
    ):
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.NUMERICAL_REFUSAL,
            "derived quadratic-response interval is nonfinite or excludes PSD support",
        )
    return q_center, q_radius


@dataclass(frozen=True, slots=True)
class V1SampledV7ResidentD1LiteralJPopulationDesign:
    route_tile_size: int = 131_072
    hbm_ceiling_fraction: float = 0.85
    production_cuda_required: bool = True
    allow_private_cpu_reference: bool = False
    deterministic_fp64_required: bool = True
    schema: str = SAMPLED_V7_RESIDENT_D1_LITERAL_J_POPULATION_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_tile_size",
            _positive_int(self.route_tile_size, name="route tile size"),
        )
        object.__setattr__(
            self,
            "hbm_ceiling_fraction",
            _finite_fraction(self.hbm_ceiling_fraction, name="HBM ceiling fraction"),
        )
        flags = (
            self.production_cuda_required,
            self.allow_private_cpu_reference,
            self.deterministic_fp64_required,
        )
        if any(type(item) is not bool for item in flags):
            raise TypeError("resident-D1 design flags must be exact booleans")
        if (
            self.production_cuda_required == self.allow_private_cpu_reference
            or self.deterministic_fp64_required is not True
            or self.schema
            != SAMPLED_V7_RESIDENT_D1_LITERAL_J_POPULATION_DESIGN_SCHEMA
        ):
            _refuse(
                V1SampledV7ResidentD1LiteralJPopulationRefusalCode.AUTHORITY_VIOLATION,
                "resident-D1 design must select exactly one deterministic execution mode",
            )

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True, slots=True)
class _V1SampledV7ResidentD1SourceBinding:
    branch_axis: tuple[V1RelaxedKKTBranch, V1RelaxedKKTBranch]
    branch_execution_hashes: tuple[str, str]
    branch_receipt_hashes: tuple[str, str]
    private_primal_source_hash: str
    private_primal_descriptor_axis_hash: str
    direction_ids: tuple[str, str]
    direction_descriptor_hashes: tuple[str, str]
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer is not _ISSUER:
            raise PermissionError("resident-D1 source bindings are issued internally")
        if self.branch_axis != (
            V1RelaxedKKTBranch.SPECTRAL,
            V1RelaxedKKTBranch.DIAGONAL_L1,
        ):
            _refuse(
                V1SampledV7ResidentD1LiteralJPopulationRefusalCode.BRANCH_ORDER_MISMATCH,
                "source branch axis must be spectral then diagonal_l1",
            )
        for axis_name in (
            "branch_execution_hashes",
            "branch_receipt_hashes",
            "direction_ids",
            "direction_descriptor_hashes",
        ):
            axis = tuple(_identity(item, name=axis_name) for item in getattr(self, axis_name))
            if len(axis) != 2 or len(set(axis)) != 2:
                _refuse(
                    V1SampledV7ResidentD1LiteralJPopulationRefusalCode.STALE_LINEAGE,
                    f"{axis_name} must be one ordered distinct branch pair",
                )
        _identity(self.private_primal_source_hash, name="private-primal source hash")
        _identity(
            self.private_primal_descriptor_axis_hash,
            name="private-primal descriptor-axis hash",
        )
        if (
            type(self.production_cuda_execution) is not bool
            or type(self.private_cpu_reference_execution) is not bool
            or self.production_cuda_execution == self.private_cpu_reference_execution
        ):
            raise TypeError("source binding must select exactly one execution mode")

    @property
    def binding_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "source-binding-",
        )


@dataclass(frozen=True, slots=True)
class V1SampledV7ResidentD1LiteralJPopulationReceipt:
    design_hash: str
    source_binding_hash: str
    branch_axis: tuple[V1RelaxedKKTBranch, V1RelaxedKKTBranch]
    branch_execution_hashes: tuple[str, str]
    branch_receipt_hashes: tuple[str, str]
    private_primal_source_hash: str
    private_primal_descriptor_axis_hash: str
    direction_ids: tuple[str, str]
    direction_descriptor_hashes: tuple[str, str]
    projection_receipt_hashes: tuple[str, str]
    projection_registration_hashes: tuple[str, str]
    fixed_primal_descriptor_hashes: tuple[str, str]
    fixed_primal_device_fingerprints: tuple[str, str]
    exact_census_source_plan_hash: str
    exact_census_source_binding_hash: str
    census_replay_digest_family_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    native_fiber_axis_receipt_hash: str
    route_count: int
    stratum_ids: tuple[str, str]
    d1_stratum_counts: tuple[int, int]
    exact_d1_stratum_weight_fractions: tuple[tuple[int, int], tuple[int, int]]
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    causal_registration_receipt_hash: str
    two_mjz_center_source_fingerprints: tuple[str, str]
    two_mjz_radius_source_fingerprints: tuple[str, str]
    q_center_fingerprint: str
    q_radius_fingerprint: str
    resident_route_vector_bytes: int
    temporary_tile_bytes_upper: int
    incremental_device_bytes_upper: int
    hbm_ceiling_fraction: float
    hbm_admission_checked: bool
    route_scale_device_to_host_bytes: int
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    exact_frozen_d1_census: bool = True
    source_seals_bound_through_projection_registration_hashes: bool = True
    preferred_hard_action_required: bool = False
    context_route_bank_retained: bool = False
    live_handoff_required: bool = True
    d1_rows_replayable_from_handoff: bool = False
    rank_authority: bool = False
    kkt_face_authority: bool = False
    action_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_V7_RESIDENT_D1_LITERAL_J_POPULATION_AUTHORITY
    schema: str = SAMPLED_V7_RESIDENT_D1_LITERAL_J_POPULATION_RECEIPT_SCHEMA
    _issuer: object = field(default=None, repr=False, compare=False)
    _receipt_hash: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer is not _ISSUER:
            raise PermissionError("resident-D1 receipts are source-issued only")
        for name in (
            "design_hash", "source_binding_hash", "private_primal_source_hash",
            "private_primal_descriptor_axis_hash", "exact_census_source_plan_hash",
            "exact_census_source_binding_hash", "census_replay_digest_family_hash",
            "model_identity_hash", "chart_identity_hash", "current_mask_hash",
            "route_axis_hash", "route_id_axis_hash", "native_fiber_axis_receipt_hash",
            "witness_stream_source_hash", "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash", "literal_j_stream_source_hash",
            "causal_registration_receipt_hash", "q_center_fingerprint",
            "q_radius_fingerprint",
        ):
            _identity(getattr(self, name), name=name)
        for name in (
            "branch_execution_hashes", "branch_receipt_hashes", "direction_ids",
            "direction_descriptor_hashes", "projection_receipt_hashes",
            "projection_registration_hashes", "fixed_primal_descriptor_hashes",
            "fixed_primal_device_fingerprints", "two_mjz_center_source_fingerprints",
            "two_mjz_radius_source_fingerprints", "stratum_ids",
        ):
            axis = tuple(_identity(item, name=name) for item in getattr(self, name))
            if len(axis) != 2:
                raise ValueError(f"{name} must contain exactly two entries")
        if self.branch_axis != (
            V1RelaxedKKTBranch.SPECTRAL,
            V1RelaxedKKTBranch.DIAGONAL_L1,
        ):
            _refuse(
                V1SampledV7ResidentD1LiteralJPopulationRefusalCode.BRANCH_ORDER_MISMATCH,
                "receipt branch axis changed",
            )
        _nonnegative_int(self.proposal_generation, name="proposal generation")
        routes = _positive_int(self.route_count, name="route count")
        counts = tuple(_positive_int(item, name="D1 stratum count") for item in self.d1_stratum_counts)
        total = sum(counts)
        expected_fractions = tuple(
            (Fraction(item, total).numerator, Fraction(item, total).denominator)
            for item in counts
        )
        expected_bytes = 4 * routes * _F64_BYTES
        temporary = _nonnegative_int(
            self.temporary_tile_bytes_upper,
            name="temporary tile bytes upper",
        )
        incremental = _positive_int(
            self.incremental_device_bytes_upper,
            name="incremental device bytes upper",
        )
        ceiling_fraction = _finite_fraction(
            self.hbm_ceiling_fraction,
            name="receipt HBM ceiling fraction",
        )
        required_true = (
            self.exact_frozen_d1_census,
            self.source_seals_bound_through_projection_registration_hashes,
            self.live_handoff_required,
        )
        required_false = (
            self.preferred_hard_action_required, self.context_route_bank_retained,
            self.d1_rows_replayable_from_handoff, self.rank_authority,
            self.kkt_face_authority, self.action_stability_authority,
            self.c_stream_authority, self.calibration_authority,
            self.oracle_authority, self.commit_authority, self.terminal_authority,
            self.pareto_authority,
        )
        if (
            self.exact_d1_stratum_weight_fractions != expected_fractions
            or self.resident_route_vector_bytes != expected_bytes
            or incremental != expected_bytes + 4 * _F64_BYTES + temporary
            or type(self.hbm_admission_checked) is not bool
            or self.hbm_admission_checked != self.production_cuda_execution
            or ceiling_fraction > 0.85
            or self.route_scale_device_to_host_bytes != 0
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or any(type(item) is not bool for item in required_true + required_false)
            or any(item is not True for item in required_true)
            or any(item is not False for item in required_false)
            or self.authority_scope != SAMPLED_V7_RESIDENT_D1_LITERAL_J_POPULATION_AUTHORITY
            or self.schema != SAMPLED_V7_RESIDENT_D1_LITERAL_J_POPULATION_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SampledV7ResidentD1LiteralJPopulationRefusalCode.AUTHORITY_VIOLATION,
                "resident-D1 receipt crossed its numerical or authority boundary",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "receipt-")
        if self._receipt_hash is not None and self._receipt_hash != observed:
            _refuse(
                V1SampledV7ResidentD1LiteralJPopulationRefusalCode.MUTATED_CUSTODY,
                "resident-D1 receipt mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SampledV7ResidentD1LiteralJPopulationRefusalCode.STALE_LINEAGE,
                "resident-D1 receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1SampledV7ResidentD1LiteralJPopulationLease:
    __slots__ = (
        "_owner", "_closed", "two_mjz_center", "two_mjz_radius", "q_center", "q_radius"
    )

    def __init__(
        self,
        owner: "V1SampledV7ResidentD1LiteralJPopulationDeviceHandle",
        tensors: dict[str, Any],
    ) -> None:
        self._owner = owner
        self._closed = False
        self.two_mjz_center = tensors["two_mjz_center"]
        self.two_mjz_radius = tensors["two_mjz_radius"]
        self.q_center = tensors["q_center"]
        self.q_radius = tensors["q_radius"]

    def __enter__(self) -> "V1SampledV7ResidentD1LiteralJPopulationLease":
        self._owner.assert_live_identity()
        return self

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._owner._close_lease()
        finally:
            self._closed = True

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("resident-D1 leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("resident-D1 leases are nonserializable")


class V1SampledV7ResidentD1LiteralJPopulationDeviceHandle:
    __slots__ = ("receipt", "_tensors", "_invariants", "_leases", "_evicted")

    def __init__(
        self,
        *,
        receipt: V1SampledV7ResidentD1LiteralJPopulationReceipt,
        tensors: dict[str, Any],
        _issuance_token: object,
    ) -> None:
        if _issuance_token is not _ISSUER or type(receipt) is not V1SampledV7ResidentD1LiteralJPopulationReceipt:
            raise PermissionError("resident-D1 device handles are source-issued only")
        required = {"two_mjz_center", "two_mjz_radius", "q_center", "q_radius"}
        if set(tensors) != required:
            raise TypeError("resident-D1 tensor roles are incomplete")
        self.receipt = receipt
        self._tensors = dict(tensors)
        self._invariants = {name: _tensor_invariant(value) for name, value in tensors.items()}
        self._leases = 0
        self._evicted = False
        self.assert_live_identity()

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    @property
    def active_lease_count(self) -> int:
        return self._leases

    def assert_live_identity(self) -> str:
        if self._evicted:
            _refuse(
                V1SampledV7ResidentD1LiteralJPopulationRefusalCode.REUSED_OR_EVICTED,
                "resident-D1 geometry was evicted",
            )
        self.receipt.assert_integrity()
        for name, tensor in self._tensors.items():
            if _tensor_invariant(tensor) != self._invariants[name]:
                _refuse(
                    V1SampledV7ResidentD1LiteralJPopulationRefusalCode.MUTATED_CUSTODY,
                    f"resident-D1 {name} tensor identity or version changed",
                )
        return content_hash(
            {
                "receipt_hash": self.receipt.receipt_hash,
                "invariants": tuple((name, self._invariants[name]) for name in sorted(self._invariants)),
                "process_local_nonserializable": True,
            },
            prefix=_PREFIX + "live-identity-",
        )

    def borrow(self) -> V1SampledV7ResidentD1LiteralJPopulationLease:
        self.assert_live_identity()
        self._leases += 1
        return V1SampledV7ResidentD1LiteralJPopulationLease(self, self._tensors)

    def _close_lease(self) -> None:
        if self._leases <= 0:
            _refuse(
                V1SampledV7ResidentD1LiteralJPopulationRefusalCode.ACTIVE_LEASE,
                "resident-D1 lease accounting underflow",
            )
        try:
            self.assert_live_identity()
        finally:
            self._leases -= 1

    def evict(self) -> None:
        if self._evicted:
            return
        if self._leases:
            _refuse(
                V1SampledV7ResidentD1LiteralJPopulationRefusalCode.ACTIVE_LEASE,
                "resident-D1 geometry cannot be evicted while leased",
            )
        try:
            self.assert_live_identity()
        finally:
            self._tensors.clear()
            self._invariants.clear()
            self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("resident-D1 handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("resident-D1 handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1SampledV7ResidentD1LiteralJPopulationResult:
    receipt: V1SampledV7ResidentD1LiteralJPopulationReceipt
    device_handle: V1SampledV7ResidentD1LiteralJPopulationDeviceHandle = field(
        repr=False, compare=False
    )
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self._issuer is not _ISSUER
            or type(self.receipt) is not V1SampledV7ResidentD1LiteralJPopulationReceipt
            or type(self.device_handle)
            is not V1SampledV7ResidentD1LiteralJPopulationDeviceHandle
        ):
            raise PermissionError("resident-D1 results are source-issued only")
        self.device_handle.assert_live_identity()
        if self.device_handle.receipt.receipt_hash != self.receipt.receipt_hash:
            _refuse(
                V1SampledV7ResidentD1LiteralJPopulationRefusalCode.STALE_LINEAGE,
                "resident-D1 receipt and live handle differ",
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError("resident-D1 results are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("resident-D1 results are nonserializable")


class V1SampledV7ResidentD1LiteralJPopulationCapture:
    """One-shot callable accepted directly by ``d1_projection_consumer``."""

    __slots__ = ("_binding", "_design", "_result", "_consumed")

    def __init__(
        self,
        *,
        binding: _V1SampledV7ResidentD1SourceBinding,
        design: V1SampledV7ResidentD1LiteralJPopulationDesign,
        _issuance_token: object,
    ) -> None:
        if (
            _issuance_token is not _ISSUER
            or type(binding) is not _V1SampledV7ResidentD1SourceBinding
            or type(design) is not V1SampledV7ResidentD1LiteralJPopulationDesign
        ):
            raise PermissionError("resident-D1 capture owners are source-issued only")
        self._binding = binding
        self._design = design
        self._result: V1SampledV7ResidentD1LiteralJPopulationResult | None = None
        self._consumed = False

    def __call__(
        self,
        projections: tuple[
            V1SignedWorkFixedPrimalProjectionCUDAExecution,
            V1SignedWorkFixedPrimalProjectionCUDAExecution,
        ],
    ) -> None:
        if self._consumed:
            _refuse(
                V1SampledV7ResidentD1LiteralJPopulationRefusalCode.REUSED_OR_EVICTED,
                "resident-D1 capture is one-shot",
            )
        self._consumed = True
        self._result = _capture_resident_d1(
            projections=projections,
            binding=self._binding,
            design=self._design,
        )

    def take_result(self) -> V1SampledV7ResidentD1LiteralJPopulationResult:
        if self._result is None:
            _refuse(
                V1SampledV7ResidentD1LiteralJPopulationRefusalCode.REUSED_OR_EVICTED,
                "resident-D1 capture has not completed",
            )
        return self._result

    def __reduce__(self) -> NoReturn:
        raise TypeError("resident-D1 capture owners are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("resident-D1 capture owners are nonserializable")


def _source_binding_from_live_owners(
    *,
    branch_executions: Sequence[V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle],
    private_primal_bridge: V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecution,
) -> _V1SampledV7ResidentD1SourceBinding:
    executions = tuple(branch_executions)
    if (
        len(executions) != 2
        or any(
            type(item) is not V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle
            for item in executions
        )
    ):
        raise TypeError("resident-D1 capture requires exactly two branch executions")
    if tuple(item.branch for item in executions) != (
        V1RelaxedKKTBranch.SPECTRAL,
        V1RelaxedKKTBranch.DIAGONAL_L1,
    ):
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.BRANCH_ORDER_MISMATCH,
            "branch executions must be spectral then diagonal_l1",
        )
    if type(private_primal_bridge) is not V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecution:
        raise TypeError("production resident-D1 capture requires the exact private-primal bridge")
    private_primal_bridge.assert_live_identity()
    source = private_primal_bridge.source_receipt
    receipt = private_primal_bridge.receipt
    axis = private_primal_bridge.descriptor_axis
    execution_hashes = tuple(item.execution_hash for item in executions)
    branch_hashes = tuple(item.branch_receipt.receipt_hash for item in executions)
    if (
        source.branch_order != ("spectral", "diagonal_l1")
        or source.source_execution_hashes != execution_hashes
        or source.branch_receipt_hashes != branch_hashes
        or receipt.direction_ids != axis.direction_ids
        or receipt.direction_descriptor_hashes != axis.descriptor_hashes
        or receipt.descriptor_axis_hash != axis.descriptor_axis_hash
    ):
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.STALE_LINEAGE,
            "KKT branches and private-primal source no longer close",
        )
    return _V1SampledV7ResidentD1SourceBinding(
        branch_axis=(V1RelaxedKKTBranch.SPECTRAL, V1RelaxedKKTBranch.DIAGONAL_L1),
        branch_execution_hashes=execution_hashes,  # type: ignore[arg-type]
        branch_receipt_hashes=branch_hashes,  # type: ignore[arg-type]
        private_primal_source_hash=source.source_hash,
        private_primal_descriptor_axis_hash=axis.descriptor_axis_hash,
        direction_ids=axis.direction_ids,
        direction_descriptor_hashes=axis.descriptor_hashes,
        production_cuda_execution=True,
        private_cpu_reference_execution=False,
        _issuer=_ISSUER,
    )


def prepare_v1_sampled_v7_resident_d1_literal_j_population_capture(
    *,
    branch_executions: Sequence[V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle],
    private_primal_bridge: V1SignedWorkLiteralJPrivatePrimalBridgeCUDAExecution,
    design: V1SampledV7ResidentD1LiteralJPopulationDesign | None = None,
) -> V1SampledV7ResidentD1LiteralJPopulationCapture:
    if design is None:
        design = V1SampledV7ResidentD1LiteralJPopulationDesign()
    if type(design) is not V1SampledV7ResidentD1LiteralJPopulationDesign:
        raise TypeError("resident-D1 capture design must be exact")
    if not design.production_cuda_required:
        raise TypeError("public resident-D1 capture requires its CUDA design")
    binding = _source_binding_from_live_owners(
        branch_executions=branch_executions,
        private_primal_bridge=private_primal_bridge,
    )
    return V1SampledV7ResidentD1LiteralJPopulationCapture(
        binding=binding, design=design, _issuance_token=_ISSUER
    )


def _prepare_v1_sampled_v7_resident_d1_literal_j_population_capture_private_reference_for_tests(
    *,
    branch_executions: Sequence[V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle],
    direction_ids: tuple[str, str],
    route_tile_size: int = 131_072,
) -> V1SampledV7ResidentD1LiteralJPopulationCapture:
    executions = tuple(branch_executions)
    if len(executions) != 2 or tuple(item.branch for item in executions) != (
        V1RelaxedKKTBranch.SPECTRAL,
        V1RelaxedKKTBranch.DIAGONAL_L1,
    ):
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.BRANCH_ORDER_MISMATCH,
            "private reference branches must retain canonical order",
        )
    ids = tuple(_identity(item, name="private direction ID") for item in direction_ids)
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("private direction IDs must be one distinct pair")
    execution_hashes = tuple(item.execution_hash for item in executions)
    branch_hashes = tuple(item.branch_receipt.receipt_hash for item in executions)
    binding = _V1SampledV7ResidentD1SourceBinding(
        branch_axis=(V1RelaxedKKTBranch.SPECTRAL, V1RelaxedKKTBranch.DIAGONAL_L1),
        branch_execution_hashes=execution_hashes,  # type: ignore[arg-type]
        branch_receipt_hashes=branch_hashes,  # type: ignore[arg-type]
        private_primal_source_hash=content_hash(execution_hashes, prefix=_PREFIX + "private-source-"),
        private_primal_descriptor_axis_hash=content_hash(ids, prefix=_PREFIX + "private-axis-"),
        direction_ids=ids,  # type: ignore[arg-type]
        direction_descriptor_hashes=tuple(
            content_hash((execution_hashes[index], ids[index]), prefix=_PREFIX + "private-descriptor-")
            for index in range(2)
        ),  # type: ignore[arg-type]
        production_cuda_execution=False,
        private_cpu_reference_execution=True,
        _issuer=_ISSUER,
    )
    design = V1SampledV7ResidentD1LiteralJPopulationDesign(
        route_tile_size=route_tile_size,
        production_cuda_required=False,
        allow_private_cpu_reference=True,
    )
    return V1SampledV7ResidentD1LiteralJPopulationCapture(
        binding=binding, design=design, _issuance_token=_ISSUER
    )


def _common_receipt_value(receipts: tuple[Any, Any], name: str) -> Any:
    values = tuple(getattr(item, name) for item in receipts)
    if values[0] != values[1]:
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.STALE_LINEAGE,
            f"branch projections disagree on {name}",
        )
    return values[0]


def _capture_resident_d1(
    *,
    projections: object,
    binding: _V1SampledV7ResidentD1SourceBinding,
    design: V1SampledV7ResidentD1LiteralJPopulationDesign,
) -> V1SampledV7ResidentD1LiteralJPopulationResult:
    if type(projections) is not tuple or len(projections) != 2 or any(
        type(item) is not V1SignedWorkFixedPrimalProjectionCUDAExecution
        for item in projections
    ):
        raise TypeError("resident-D1 capture accepts exactly two fixed-primal executions")
    pair = projections
    receipts = (pair[0].receipt, pair[1].receipt)
    for item in pair:
        item.device_handle.assert_live_identity()
    if (
        tuple(item.fixed_primal_id for item in receipts) != binding.direction_ids
        or tuple(item.fixed_primal_source_receipt_hash for item in receipts)
        != binding.branch_receipt_hashes
        or tuple(item.production_cuda_execution for item in receipts)
        != (binding.production_cuda_execution,) * 2
        or tuple(item.private_cpu_reference_execution for item in receipts)
        != (binding.private_cpu_reference_execution,) * 2
    ):
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.BRANCH_ORDER_MISMATCH,
            "fixed-primal projections do not retain canonical branch/direction order",
        )
    required_census_flags = all(
        item.exact_census_mode
        and item.exact_finite_population_census
        and not item.sampling_alpha_spent
        and item.familywise_alpha is None
        and item.signed_f_sampling_radius_identically_zero
        and item.literal_j_q_sampling_radius_identically_zero
        and item.census_supports_derived_from_sealed_rows
        and item.windowed_two_pass_source_replay
        and item.pass1_pass2_value_identity_verified
        for item in receipts
    )
    if not required_census_flags:
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.NONCENSUS_SOURCE,
            "resident-D1 capture requires two exact sealed-row fitting censuses",
        )
    common_names = (
        "model_identity_hash", "chart_identity_hash", "current_mask_hash",
        "proposal_generation", "route_axis_hash", "route_id_axis_hash",
        "native_fiber_axis_receipt_hash", "route_count", "stratum_ids",
        "population_counts", "observed_counts", "population_context_axis_hashes",
        "population_draw_axis_hashes", "population_measurement_axis_hashes",
        "population_capture_axis_hashes", "registered_prefix_context_axis_hashes",
        "registered_prefix_draw_axis_hashes", "registered_prefix_measurement_axis_hashes",
        "registered_prefix_capture_axis_hashes", "exact_census_source_plan_hash",
        "exact_census_stratum_weight_fractions", "census_replay_digest_family_hash",
        "witness_stream_source_hash", "same_capture_vjp_source_hash",
        "corrected_evidence_stream_source_hash", "literal_j_stream_source_hash",
        "causal_registration_receipt_hash",
    )
    common = {name: _common_receipt_value(receipts, name) for name in common_names}
    routes = _positive_int(common["route_count"], name="route count")
    counts = tuple(common["observed_counts"])
    if counts != tuple(common["population_counts"]) or sum(counts) <= 0:
        _refuse(
            V1SampledV7ResidentD1LiteralJPopulationRefusalCode.NONCENSUS_SOURCE,
            "D1 counts do not close their exact fitting population",
        )
    device = None
    resident_center = None
    resident_radius = None
    q_center = None
    q_radius = None
    leases: list[Any] = []
    try:
        leases = [pair[0].device_handle.borrow(), pair[1].device_handle.borrow()]
        for lease in leases:
            lease.__enter__()
        device = leases[0].fixed_primal.device
        if leases[1].fixed_primal.device != device:
            _refuse(
                V1SampledV7ResidentD1LiteralJPopulationRefusalCode.DEVICE_REQUIRED,
                "branch projection tensors must share one device",
            )
        if design.production_cuda_required:
            if device.type != "cuda":
                _refuse(
                    V1SampledV7ResidentD1LiteralJPopulationRefusalCode.CPU_PRODUCTION_FALLBACK,
                    "production resident-D1 capture requires CUDA custody",
                )
            # This path uses elementwise FP64 products and tiled reductions,
            # not GEMM/cuBLAS.  Torch deterministic-algorithm custody is the
            # applicable execution gate; a cuBLAS workspace string is not.
            if not torch.are_deterministic_algorithms_enabled():
                _refuse(
                    V1SampledV7ResidentD1LiteralJPopulationRefusalCode.DETERMINISM_DISABLED,
                    "production resident-D1 capture requires deterministic algorithms",
                )
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            temporary_bytes = (
                4 * min(routes, design.route_tile_size) * _F64_BYTES
            )
            incremental = (
                4 * routes * _F64_BYTES
                + 4 * _F64_BYTES
                + temporary_bytes
            )
            available = cuda_reusable_memory_bytes(device, framework=torch, free_bytes=free_bytes)
            occupied = int(total_bytes) - available
            ceiling = math.floor(int(total_bytes) * design.hbm_ceiling_fraction)
            if incremental > available or occupied + incremental > ceiling:
                _refuse(
                    V1SampledV7ResidentD1LiteralJPopulationRefusalCode.HBM_FORECAST_EXCEEDED,
                    "resident-D1 transfer exceeds the registered HBM ceiling",
                )
        elif device.type != "cpu" or not design.allow_private_cpu_reference:
            _refuse(
                V1SampledV7ResidentD1LiteralJPopulationRefusalCode.CPU_PRODUCTION_FALLBACK,
                "CPU resident-D1 capture is a private reference only",
            )
        else:
            temporary_bytes = (
                4 * min(routes, design.route_tile_size) * _F64_BYTES
            )
            incremental = (
                4 * routes * _F64_BYTES
                + 4 * _F64_BYTES
                + temporary_bytes
            )
        resident_center = torch.empty((2, routes), dtype=torch.float64, device=device)
        resident_radius = torch.empty_like(resident_center)
        q_center = torch.empty(2, dtype=torch.float64, device=device)
        q_radius = torch.empty_like(q_center)
        for index, lease in enumerate(leases):
            fixed = _require_fp64_vector(
                lease.fixed_primal,
                name="fixed primal",
                route_count=routes,
                device=device,
            )
            gradient = _require_fp64_vector(
                lease.literal_j_q_gradient_center,
                name="literal-J Q gradient center",
                route_count=routes,
                device=device,
            )
            gradient_radius = _require_fp64_vector(
                lease.literal_j_q_gradient_radius,
                name="literal-J Q gradient radius",
                route_count=routes,
                device=device,
            )
            if bool((gradient_radius < 0.0).any().item()):
                _refuse(
                    V1SampledV7ResidentD1LiteralJPopulationRefusalCode.NUMERICAL_REFUSAL,
                    "literal-J Q-gradient radius is negative",
                )
            branch_q_center, branch_q_radius = _outward_q_from_two_mjz(
                fixed_primal=fixed,
                two_mjz_center=gradient,
                two_mjz_radius=gradient_radius,
                route_tile_size=design.route_tile_size,
            )
            resident_center[index].copy_(gradient)
            resident_radius[index].copy_(gradient_radius)
            q_center[index].copy_(branch_q_center)
            q_radius[index].copy_(branch_q_radius)
        for item in pair:
            item.device_handle.assert_live_identity()
        exact_source_binding = content_hash(
            {
                "projection_registration_hashes": tuple(item.registration_hash for item in receipts),
                "source_plan_hash": common["exact_census_source_plan_hash"],
                "population_axes": tuple(
                    common[name]
                    for name in (
                        "population_context_axis_hashes", "population_draw_axis_hashes",
                        "population_measurement_axis_hashes", "population_capture_axis_hashes",
                    )
                ),
                "registered_prefix_axes": tuple(
                    common[name]
                    for name in (
                        "registered_prefix_context_axis_hashes", "registered_prefix_draw_axis_hashes",
                        "registered_prefix_measurement_axis_hashes", "registered_prefix_capture_axis_hashes",
                    )
                ),
                "exact_weight_fractions": common["exact_census_stratum_weight_fractions"],
                "stream_hashes": (
                    common["witness_stream_source_hash"], common["same_capture_vjp_source_hash"],
                    common["corrected_evidence_stream_source_hash"], common["literal_j_stream_source_hash"],
                ),
                "causal_registration_receipt_hash": common["causal_registration_receipt_hash"],
            },
            prefix=_PREFIX + "exact-census-source-binding-",
        )
        receipt = V1SampledV7ResidentD1LiteralJPopulationReceipt(
            design_hash=design.design_hash,
            source_binding_hash=binding.binding_hash,
            branch_axis=binding.branch_axis,
            branch_execution_hashes=binding.branch_execution_hashes,
            branch_receipt_hashes=binding.branch_receipt_hashes,
            private_primal_source_hash=binding.private_primal_source_hash,
            private_primal_descriptor_axis_hash=binding.private_primal_descriptor_axis_hash,
            direction_ids=binding.direction_ids,
            direction_descriptor_hashes=binding.direction_descriptor_hashes,
            projection_receipt_hashes=tuple(item.receipt_hash for item in receipts),  # type: ignore[arg-type]
            projection_registration_hashes=tuple(item.registration_hash for item in receipts),  # type: ignore[arg-type]
            fixed_primal_descriptor_hashes=tuple(item.fixed_primal_descriptor_hash for item in receipts),  # type: ignore[arg-type]
            fixed_primal_device_fingerprints=tuple(item.fixed_primal_device_fingerprint for item in receipts),  # type: ignore[arg-type]
            exact_census_source_plan_hash=common["exact_census_source_plan_hash"],
            exact_census_source_binding_hash=exact_source_binding,
            census_replay_digest_family_hash=common["census_replay_digest_family_hash"],
            model_identity_hash=common["model_identity_hash"],
            chart_identity_hash=common["chart_identity_hash"],
            current_mask_hash=common["current_mask_hash"],
            proposal_generation=common["proposal_generation"],
            route_axis_hash=common["route_axis_hash"],
            route_id_axis_hash=common["route_id_axis_hash"],
            native_fiber_axis_receipt_hash=common["native_fiber_axis_receipt_hash"],
            route_count=routes,
            stratum_ids=common["stratum_ids"],
            d1_stratum_counts=counts,  # type: ignore[arg-type]
            exact_d1_stratum_weight_fractions=common["exact_census_stratum_weight_fractions"],
            witness_stream_source_hash=common["witness_stream_source_hash"],
            same_capture_vjp_source_hash=common["same_capture_vjp_source_hash"],
            corrected_evidence_stream_source_hash=common["corrected_evidence_stream_source_hash"],
            literal_j_stream_source_hash=common["literal_j_stream_source_hash"],
            causal_registration_receipt_hash=common["causal_registration_receipt_hash"],
            two_mjz_center_source_fingerprints=tuple(item.literal_j_q_gradient_center_fingerprint for item in receipts),  # type: ignore[arg-type]
            two_mjz_radius_source_fingerprints=tuple(item.literal_j_q_gradient_total_radius_fingerprint for item in receipts),  # type: ignore[arg-type]
            q_center_fingerprint=_scalar_fingerprint(q_center, role="D1_Q_center"),
            q_radius_fingerprint=_scalar_fingerprint(q_radius, role="D1_Q_radius"),
            resident_route_vector_bytes=4 * routes * _F64_BYTES,
            temporary_tile_bytes_upper=temporary_bytes,
            incremental_device_bytes_upper=incremental,
            hbm_ceiling_fraction=design.hbm_ceiling_fraction,
            hbm_admission_checked=design.production_cuda_required,
            route_scale_device_to_host_bytes=0,
            production_cuda_execution=design.production_cuda_required,
            private_cpu_reference_execution=design.allow_private_cpu_reference,
            _issuer=_ISSUER,
        )
        handle = V1SampledV7ResidentD1LiteralJPopulationDeviceHandle(
            receipt=receipt,
            tensors={
                "two_mjz_center": resident_center,
                "two_mjz_radius": resident_radius,
                "q_center": q_center,
                "q_radius": q_radius,
            },
            _issuance_token=_ISSUER,
        )
        return V1SampledV7ResidentD1LiteralJPopulationResult(
            receipt=receipt, device_handle=handle, _issuer=_ISSUER
        )
    except BaseException:
        # Partial capture owns no reusable authority.  Release every tensor it
        # allocated; the fixed-primal shell independently burns its sources.
        for name in ("resident_center", "resident_radius", "q_center", "q_radius"):
            value = locals().get(name)
            if value is not None:
                del value
        raise
    finally:
        for lease in reversed(leases):
            lease.close()


def register_v1_sampled_v7_resident_d1_literal_j_population_projection(**kwargs: Any) -> Any:
    """Bind the resident D1 owner to a fresh residual-population design."""

    from .modern_v1_literal_j_d_fixed_residual_population_projection import (
        _register_v1_literal_j_d_fixed_residual_population_projection_from_v7_resident_d1,
    )

    return _register_v1_literal_j_d_fixed_residual_population_projection_from_v7_resident_d1(
        **kwargs
    )


def issue_v1_sampled_v7_resident_d1_literal_j_population_projection(**kwargs: Any) -> Any:
    """Join exact ``d/N`` resident D1 geometry with fresh residual intervals."""

    from .modern_v1_literal_j_d_fixed_residual_population_projection import (
        _issue_v1_literal_j_d_fixed_residual_population_projection_from_v7_resident_d1,
    )

    return _issue_v1_literal_j_d_fixed_residual_population_projection_from_v7_resident_d1(
        **kwargs
    )


__all__ = [
    "SAMPLED_V7_RESIDENT_D1_LITERAL_J_POPULATION_AUTHORITY",
    "SAMPLED_V7_RESIDENT_D1_LITERAL_J_POPULATION_DESIGN_SCHEMA",
    "SAMPLED_V7_RESIDENT_D1_LITERAL_J_POPULATION_RECEIPT_SCHEMA",
    "V1SampledV7ResidentD1LiteralJPopulationCapture",
    "V1SampledV7ResidentD1LiteralJPopulationDesign",
    "V1SampledV7ResidentD1LiteralJPopulationDeviceHandle",
    "V1SampledV7ResidentD1LiteralJPopulationLease",
    "V1SampledV7ResidentD1LiteralJPopulationReceipt",
    "V1SampledV7ResidentD1LiteralJPopulationRefusal",
    "V1SampledV7ResidentD1LiteralJPopulationRefusalCode",
    "V1SampledV7ResidentD1LiteralJPopulationResult",
    "issue_v1_sampled_v7_resident_d1_literal_j_population_projection",
    "prepare_v1_sampled_v7_resident_d1_literal_j_population_capture",
    "register_v1_sampled_v7_resident_d1_literal_j_population_projection",
]
