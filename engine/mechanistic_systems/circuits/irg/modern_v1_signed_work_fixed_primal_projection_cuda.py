"""Source-issued CUDA projections for one frozen signed-work KKT primal.

This module is deliberately narrower than a decision-stability theorem.  For
one primal ``z`` frozen before a predictable two-stratum ``D_W`` witness look,
it streams same-capture corrected-evidence and literal-J rows and estimates

``E[a_j sign(sum_{k in native_fiber(j)} a_k z_k)]``

and

``2 E[j_j (j.T z)]``.

The first expression uses the registered uniform-cut/counting-fiber measure;
the second is the literal relaxed-gate response gradient and has no cut
normalization.  ``sign(0)=0`` is the selected valid subgradient at exact zero.
Cells whose binary64 sign is not separated from zero by the registered
roundoff enclosure are counted explicitly and their worst-case routewise
effect is added to the numerical radius.

Production execution is CUDA-only.  Route-scale centers and radii never leave
the device, native-fiber membership is replayed from the compact cut receipt,
and the durable receipt contains only identities, scalar telemetry, and
bounded device fingerprints.  A private CPU mode exists solely for tiny
high-precision tests.  The result is fixed-z projection evidence only: it does
not establish solution sensitivity, normal-cone stability, an invariant KKT
face, a stable physical action, or authority for C/K/O/commit/Pareto claims.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from decimal import Decimal, ROUND_CEILING, localcontext
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral, Real
import os
from typing import Any, Final, NoReturn, Sequence

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .contracts import content_hash
from .modern_v1_native_fiber_axis import (
    V1NativeDestinationFiberAxisReceipt,
)

try:  # pragma: no cover - import availability is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SIGNED_WORK_FIXED_PRIMAL_PROJECTION_CUDA_DESIGN_SCHEMA: Final = (
    "partition_free_v1_signed_work_fixed_primal_projection_cuda_design_v1"
)
SIGNED_WORK_FIXED_PRIMAL_PROJECTION_STRATUM_SCHEMA: Final = (
    "partition_free_v1_signed_work_fixed_primal_projection_stratum_v1"
)
SIGNED_WORK_FIXED_PRIMAL_PROJECTION_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_signed_work_fixed_primal_projection_registration_v1"
)
SIGNED_WORK_FIXED_PRIMAL_PROJECTION_HBM_SCHEMA: Final = (
    "partition_free_v1_signed_work_fixed_primal_projection_hbm_v1"
)
SIGNED_WORK_FIXED_PRIMAL_ZERO_CELL_SCHEMA: Final = (
    "partition_free_v1_signed_work_fixed_primal_zero_cell_ambiguity_v1"
)
SIGNED_WORK_FIXED_PRIMAL_PROJECTION_TELEMETRY_SCHEMA: Final = (
    "partition_free_v1_signed_work_fixed_primal_projection_telemetry_v2"
)
SIGNED_WORK_FIXED_PRIMAL_PROJECTION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_fixed_primal_projection_receipt_v3"
)
SIGNED_WORK_FIXED_PRIMAL_PROJECTION_ALGORITHM: Final = (
    "predictable_two_stratum_same_capture_native_fiber_signed_F_and_literal_J_"
    "fixed_primal_fp64_tiled_cuda_with_source_support_serfling_enclosures_v1"
)
SIGNED_WORK_FIXED_PRIMAL_PROJECTION_AUTHORITY_SCOPE: Final = (
    "fixed_z_source_projection_only_no_solution_sensitivity_no_normal_cone_"
    "or_action_stability_no_C_K_O_commit_terminal_production_promotion_or_"
    "pareto_authority_v1"
)
SIGNED_WORK_FIXED_PRIMAL_SIGN_POLICY: Final = (
    "binary64_cell_sign_with_sign_zero_equal_zero_and_roundoff_ambiguous_cells_"
    "covered_by_routewise_numerical_radius_no_strict_sign_claim_v1"
)
SIGNED_WORK_FIXED_PRIMAL_ALPHA_FAMILY: Final = (
    "familywise_alpha_split_over_two_gradients_times_all_routes_times_two_tails_"
    "at_one_registered_nonrecyclable_fixed_look_v1"
)
SIGNED_WORK_FIXED_PRIMAL_F_EQUATION: Final = (
    "E_context_a_j_times_sign_native_fiber_sum_a_k_z_k_divided_by_cut_count_v1"
)
SIGNED_WORK_FIXED_PRIMAL_Q_EQUATION: Final = (
    "two_times_E_context_j_j_times_sum_k_j_k_z_k_v1"
)

_PREFIX: Final = "partition-free-v1-signed-work-fixed-primal-projection-"
_EXACT_CENSUS_REGISTRATION_ISSUER: Final = object()
_F64_BYTES: Final = 8
_BOOL_BYTES: Final = 1
_F64_EPS: Final = 2.0**-52
_ISSUER_TOKEN: Final = object()


class V1SignedWorkFixedPrimalProjectionCUDARefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    STALE_IDENTITY = "stale_identity"
    ORDER_MISMATCH = "order_mismatch"
    DUPLICATE_IDENTITY = "duplicate_identity"
    DEVICE_REQUIRED = "device_required"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    SUPPORT_BOUND_VIOLATION = "support_bound_violation"
    NUMERICAL_REFUSAL = "numerical_refusal"
    PREFIX_INCOMPLETE = "prefix_incomplete"
    FIXED_LOOK_REUSE = "fixed_look_reuse"
    DETERMINISM_DISABLED = "determinism_disabled"
    MUTATED_CUSTODY = "mutated_custody"
    EVICTED = "evicted"
    ACTIVE_LEASE = "active_lease"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkFixedPrimalProjectionCUDARefusal(RuntimeError):
    """Typed fail-closed projection refusal."""

    def __init__(
        self,
        code: V1SignedWorkFixedPrimalProjectionCUDARefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkFixedPrimalProjectionCUDARefusalCode:
            raise TypeError("projection refusal code must have exact enum type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkFixedPrimalProjectionCUDARefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkFixedPrimalProjectionCUDARefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _sha256_hex(value: object, *, name: str) -> str:
    result = _identity(value, name=name)
    if (
        len(result) != 64
        or any(character not in "0123456789abcdef" for character in result)
    ):
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one lowercase SHA-256 digest",
        )
    return result


def v1_signed_work_fixed_primal_census_replay_digest_family_hash(
    digests: Sequence[str],
) -> str:
    axis = tuple(
        _sha256_hex(value, name="census replay digest") for value in digests
    )
    if not axis:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
            "census replay digest family cannot be empty",
        )
    return content_hash(
        axis,
        prefix=_PREFIX + "census-replay-digest-family-",
    )


def v1_signed_work_fixed_primal_sampled_replay_digest_family_hash(
    digests: Sequence[str],
) -> str:
    axis = tuple(
        _sha256_hex(value, name="sampled route-window replay digest")
        for value in digests
    )
    if not axis:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
            "sampled replay digest family cannot be empty",
        )
    return content_hash(
        axis,
        prefix=_PREFIX + "sampled-replay-digest-family-",
    )


def _identity_axis(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one ordered identity sequence",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
            f"{name} must be nonempty and unique",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one exact positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
            f"{name} must be nonnegative",
        )
    return result


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one finite positive real",
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
            f"{name} must be finite and positive",
        )
    return result


def _finite_probability(value: object, *, name: str) -> float:
    result = _finite_positive(value, name=name)
    if result >= 1.0:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
            f"{name} must be strictly less than one",
        )
    return result


def _gamma(operation_count: int) -> float:
    count = _positive_int(operation_count, name="roundoff operation count")
    product = count * _F64_EPS
    if product >= 0.25:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.NUMERICAL_REFUSAL,
            "registered binary64 operation depth is outside the stable gamma regime",
        )
    return math.nextafter(product / (1.0 - product), math.inf)


def _decimal_up_float(value: Decimal) -> float:
    result = float(value)
    if not math.isfinite(result):
        return math.inf
    if Decimal.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return math.nextafter(result, math.inf)


def _serfling_weight_coefficient_upper(
    *,
    stratum_weight: float,
    observed: int,
    population: int,
) -> float:
    correction = Fraction(population - observed + 1, population)
    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_CEILING
        weight = Decimal.from_float(stratum_weight)
        correction_d = Decimal(correction.numerator) / Decimal(correction.denominator)
        value = (
            Decimal(2)
            * weight
            * (correction_d / Decimal(observed)).sqrt()
        ).next_plus()
    return _decimal_up_float(value)


def _serfling_log_factor_upper(two_sided_alpha: float) -> float:
    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_CEILING
        alpha = Decimal.from_float(two_sided_alpha)
        value = (
            (Decimal(2) / alpha).ln() / Decimal(2)
        ).next_plus()
    return _decimal_up_float(value)


def _tensor_metadata(value: object, *, name: str) -> tuple[object, ...]:
    if torch is None or type(value) is not torch.Tensor:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one exact torch.Tensor",
        )
    if (
        value.dtype not in {torch.float64, torch.bool}
        or value.ndim != 1
        or not value.is_contiguous()
    ):
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one contiguous one-dimensional float64/bool tensor",
        )
    return (
        name,
        id(value),
        int(value._version),
        int(value.data_ptr()),
        tuple(int(item) for item in value.shape),
        str(value.dtype),
        str(value.device),
    )


def _require_fp64_tensor(
    value: object,
    *,
    name: str,
    ndim: int,
    shape: tuple[int, ...],
    device: object | None = None,
    scan_finite: bool = True,
) -> Any:
    if torch is None:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.DEVICE_REQUIRED,
            "PyTorch is unavailable",
        )
    if type(value) is not torch.Tensor:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must have exact torch.Tensor type",
        )
    if (
        value.dtype != torch.float64
        or value.ndim != ndim
        or tuple(int(item) for item in value.shape) != shape
        or not value.is_contiguous()
        or bool(value.requires_grad)
    ):
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
            f"{name} must be detached contiguous float64 with shape {shape}",
        )
    if device is not None and value.device != device:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.STALE_IDENTITY,
            f"{name} is on another device",
        )
    if scan_finite and not bool(torch.isfinite(value).all().item()):
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.NUMERICAL_REFUSAL,
            f"{name} contains a nonfinite value",
        )
    return value


def _up_tensor(value: Any) -> Any:
    infinity = torch.full((), math.inf, dtype=torch.float64, device=value.device)
    return torch.nextafter(value, infinity)


def _up_add(left: Any, right: Any) -> Any:
    return _up_tensor(left + right)


def _up_mul(left: Any, right: Any) -> Any:
    return _up_tensor(left * right)


def _axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {"schema": "signed_work_fixed_primal_registered_axis_v1", "role": role, "values": tuple(values)},
        prefix=_PREFIX + "axis-",
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkFixedPrimalProjectionCUDAExecutionDesign:
    route_tile_size: int = 131_072
    maximum_batch_size: int = 8
    hbm_ceiling_fraction: float = 0.85
    production_cuda_required: bool = True
    allow_private_cpu_reference: bool = False
    deterministic_fp64_required: bool = True
    forbid_route_scale_host_transfer: bool = True
    forbid_route_scale_fiber_labels: bool = True
    exact_census_windowed_source: bool = False
    sampled_windowed_source: bool = False
    schema: str = SIGNED_WORK_FIXED_PRIMAL_PROJECTION_CUDA_DESIGN_SCHEMA
    algorithm_id: str = SIGNED_WORK_FIXED_PRIMAL_PROJECTION_ALGORITHM

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_tile_size", _positive_int(self.route_tile_size, name="route tile size"))
        object.__setattr__(self, "maximum_batch_size", _positive_int(self.maximum_batch_size, name="maximum batch size"))
        ceiling = _finite_positive(self.hbm_ceiling_fraction, name="HBM ceiling fraction")
        if ceiling > 0.85:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "canonical HBM ceiling cannot exceed 0.85",
            )
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        for name in (
            "production_cuda_required",
            "allow_private_cpu_reference",
            "deterministic_fp64_required",
            "forbid_route_scale_host_transfer",
            "forbid_route_scale_fiber_labels",
            "exact_census_windowed_source",
            "sampled_windowed_source",
        ):
            if type(getattr(self, name)) is not bool:
                _refuse(
                    V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
                    f"{name} must be an exact bool",
                )
        if self.exact_census_windowed_source and self.sampled_windowed_source:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "exact-census and sampled windowed sources are mutually exclusive",
            )
        if self.production_cuda_required and self.allow_private_cpu_reference:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.CPU_PRODUCTION_FALLBACK,
                "production CUDA execution cannot enable a CPU reference fallback",
            )
        if (
            self.deterministic_fp64_required is not True
            or self.forbid_route_scale_host_transfer is not True
            or self.forbid_route_scale_fiber_labels is not True
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "deterministic FP64 and both route-scale prohibitions are mandatory",
            )
        if self.schema != SIGNED_WORK_FIXED_PRIMAL_PROJECTION_CUDA_DESIGN_SCHEMA or self.algorithm_id != SIGNED_WORK_FIXED_PRIMAL_PROJECTION_ALGORITHM:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "unsupported fixed-primal projection design",
            )

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkFixedPrimalProjectionStratumRegistration:
    stratum_id: str
    stratum_weight: float
    population_count: int
    population_context_axis_hash: str
    population_draw_axis_hash: str
    population_measurement_axis_hash: str
    population_capture_axis_hash: str
    registered_context_ids: tuple[str, ...]
    registered_draw_ids: tuple[str, ...]
    registered_measurement_ids: tuple[str, ...]
    registered_capture_ids: tuple[str, ...]
    schema: str = SIGNED_WORK_FIXED_PRIMAL_PROJECTION_STRATUM_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "stratum_id", _identity(self.stratum_id, name="stratum ID"))
        object.__setattr__(self, "stratum_weight", _finite_positive(self.stratum_weight, name="stratum weight"))
        population = _positive_int(self.population_count, name="stratum population count")
        object.__setattr__(self, "population_count", population)
        for name in (
            "population_context_axis_hash",
            "population_draw_axis_hash",
            "population_measurement_axis_hash",
            "population_capture_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        axes = {}
        for name in (
            "registered_context_ids",
            "registered_draw_ids",
            "registered_measurement_ids",
            "registered_capture_ids",
        ):
            axis = _identity_axis(getattr(self, name), name=name)
            object.__setattr__(self, name, axis)
            axes[name] = axis
        lengths = {len(axis) for axis in axes.values()}
        if len(lengths) != 1 or next(iter(lengths)) > population:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "registered stratum axes must have one prefix length no larger than the population",
            )
        if self.schema != SIGNED_WORK_FIXED_PRIMAL_PROJECTION_STRATUM_SCHEMA:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "unsupported stratum registration schema",
            )

    @property
    def registered_count(self) -> int:
        return len(self.registered_context_ids)

    @property
    def registration_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "stratum-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkFixedPrimalExactCensusSourceSeal:
    source_plan_hash: str
    source_axis_hashes: tuple[str, str, str, str]
    stratum_registration_hashes: tuple[str, str]
    exact_stratum_weight_fractions: tuple[tuple[int, int], tuple[int, int]]
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    causal_registration_receipt_hash: str
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuer is not _EXACT_CENSUS_REGISTRATION_ISSUER:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
                "exact-census source seals are issued only from live source custody",
            )
        for name in (
            "source_plan_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "witness_stream_source_hash",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "causal_registration_receipt_hash",
        ):
            _identity(getattr(self, name), name=name)
        if (
            len(self.source_axis_hashes) != 4
            or len(self.stratum_registration_hashes) != 2
            or any(
                type(value) is not str or not value
                for value in self.source_axis_hashes
                + self.stratum_registration_hashes
            )
            or len(self.exact_stratum_weight_fractions) != 2
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "exact-census source seal axes are incomplete",
            )
        _nonnegative_int(self.proposal_generation, name="seal proposal generation")

    @property
    def seal_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "exact-census-source-seal-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkFixedPrimalProjectionRegistration:
    registration_id: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    causal_registration_receipt_hash: str
    fixed_primal_id: str
    fixed_primal_source_receipt_hash: str
    corrected_support_source_receipt_hash: str
    literal_j_support_source_receipt_hash: str
    alpha_ledger_id: str
    alpha_coordinate_id: str
    look_ordinal: int
    familywise_alpha: float | None
    strata: tuple[V1SignedWorkFixedPrimalProjectionStratumRegistration, ...]
    exact_census_mode: bool = False
    sampling_alpha_spent: bool = True
    exact_census_source_plan_hash: str | None = None
    exact_census_source_axis_hashes: tuple[str, str, str, str] | None = None
    exact_census_stratum_weight_fractions: (
        tuple[tuple[int, int], tuple[int, int]] | None
    ) = None
    exact_census_source_seal: V1SignedWorkFixedPrimalExactCensusSourceSeal | None = None
    optional_stopping_reuse_allowed: bool = False
    alpha_coordinate_nonrecyclable: bool = True
    predictable_before_witness_outcomes: bool = True
    sealed_without_replacement_permutations: bool = True
    schema: str = SIGNED_WORK_FIXED_PRIMAL_PROJECTION_REGISTRATION_SCHEMA
    alpha_family: str = SIGNED_WORK_FIXED_PRIMAL_ALPHA_FAMILY

    def __post_init__(self) -> None:
        for name in (
            "registration_id",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "witness_stream_source_hash",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "causal_registration_receipt_hash",
            "fixed_primal_id",
            "fixed_primal_source_receipt_hash",
            "corrected_support_source_receipt_hash",
            "literal_j_support_source_receipt_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(self, "proposal_generation", _nonnegative_int(self.proposal_generation, name="proposal generation"))
        object.__setattr__(self, "look_ordinal", _positive_int(self.look_ordinal, name="look ordinal"))
        strata = tuple(self.strata)
        if len(strata) != 2 or any(type(item) is not V1SignedWorkFixedPrimalProjectionStratumRegistration for item in strata):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "the fixed-primal witness requires exactly two registered strata",
            )
        if len({item.stratum_id for item in strata}) != 2 or math.fsum(item.stratum_weight for item in strata) != 1.0:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "strata must be unique and carry exact total probability mass one",
            )
        for role in (
            "registered_context_ids",
            "registered_draw_ids",
            "registered_measurement_ids",
            "registered_capture_ids",
        ):
            merged = tuple(value for item in strata for value in getattr(item, role))
            if len(merged) != len(set(merged)):
                _refuse(
                    V1SignedWorkFixedPrimalProjectionCUDARefusalCode.DUPLICATE_IDENTITY,
                    f"{role} overlaps across witness strata",
                )
        object.__setattr__(self, "strata", strata)
        if type(self.exact_census_mode) is not bool or type(self.sampling_alpha_spent) is not bool:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
                "census/alpha-spend flags must be exact booleans",
            )
        exact_census = all(
            item.registered_count == item.population_count for item in strata
        )
        if self.exact_census_mode:
            plan_hash = self.exact_census_source_plan_hash
            source_axis_hashes = self.exact_census_source_axis_hashes
            fractions = self.exact_census_stratum_weight_fractions
            seal = self.exact_census_source_seal
            expected_population_hashes = tuple(
                tuple(
                    _axis_hash(
                        f"exact_census:{stratum.stratum_id}:{role}",
                        getattr(stratum, registered_role),
                    )
                    for role, registered_role in (
                        ("context", "registered_context_ids"),
                        ("draw", "registered_draw_ids"),
                        ("measurement", "registered_measurement_ids"),
                        ("capture", "registered_capture_ids"),
                    )
                )
                for stratum in strata
            )
            observed_population_hashes = tuple(
                (
                    stratum.population_context_axis_hash,
                    stratum.population_draw_axis_hash,
                    stratum.population_measurement_axis_hash,
                    stratum.population_capture_axis_hash,
                )
                for stratum in strata
            )
            total = sum(stratum.population_count for stratum in strata)
            expected_fractions = tuple(
                (
                    Fraction(stratum.population_count, total).numerator,
                    Fraction(stratum.population_count, total).denominator,
                )
                for stratum in strata
            )
            merged_axes = tuple(
                tuple(
                    value
                    for stratum in strata
                    for value in getattr(stratum, registered_role)
                )
                for registered_role in (
                    "registered_context_ids",
                    "registered_draw_ids",
                    "registered_measurement_ids",
                    "registered_capture_ids",
                )
            )
            expected_source_axis_hashes = tuple(
                _axis_hash(f"exact_census:source:{role}", values)
                for role, values in zip(
                    ("context", "draw", "measurement", "capture"),
                    merged_axes,
                    strict=True,
                )
            )
            if (
                not exact_census
                or self.familywise_alpha is not None
                or self.sampling_alpha_spent is not False
                or type(seal) is not V1SignedWorkFixedPrimalExactCensusSourceSeal
                or type(plan_hash) is not str
                or not plan_hash
                or source_axis_hashes != expected_source_axis_hashes
                or observed_population_hashes != expected_population_hashes
                or fractions != expected_fractions
                or seal.source_plan_hash != plan_hash
                or seal.source_axis_hashes != source_axis_hashes
                or seal.stratum_registration_hashes
                != tuple(item.registration_hash for item in strata)
                or seal.exact_stratum_weight_fractions != fractions
                or seal.chart_identity_hash != self.chart_identity_hash
                or seal.current_mask_hash != self.current_mask_hash
                or seal.proposal_generation != self.proposal_generation
                or seal.route_axis_hash != self.route_axis_hash
                or seal.route_id_axis_hash != self.route_id_axis_hash
                or seal.witness_stream_source_hash != self.witness_stream_source_hash
                or seal.same_capture_vjp_source_hash != self.same_capture_vjp_source_hash
                or seal.corrected_evidence_stream_source_hash
                != self.corrected_evidence_stream_source_hash
                or seal.literal_j_stream_source_hash
                != self.literal_j_stream_source_hash
                or seal.causal_registration_receipt_hash
                != self.causal_registration_receipt_hash
            ):
                _refuse(
                    V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                    "alpha-free census mode requires source-issued complete axes and exact weights",
                )
        else:
            if (
                self.exact_census_source_plan_hash is not None
                or self.exact_census_source_axis_hashes is not None
                or self.exact_census_stratum_weight_fractions is not None
                or self.exact_census_source_seal is not None
            ):
                _refuse(
                    V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                    "sampled registration cannot carry exact-census source authority",
                )
            object.__setattr__(
                self,
                "familywise_alpha",
                _finite_probability(self.familywise_alpha, name="familywise alpha"),
            )
            if self.sampling_alpha_spent is not True:
                _refuse(
                    V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                    "noncensus projection must retain its registered alpha spend",
                )
        if self.optional_stopping_reuse_allowed is not False or self.alpha_coordinate_nonrecyclable is not True:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.FIXED_LOOK_REUSE,
                "this joint admits one nonrecyclable fixed-look alpha coordinate only",
            )
        if self.predictable_before_witness_outcomes is not True or self.sealed_without_replacement_permutations is not True:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.FIXED_LOOK_REUSE,
                "Serfling inference requires a pre-outcome fixed primal and sealed without-replacement permutations",
            )
        if self.schema != SIGNED_WORK_FIXED_PRIMAL_PROJECTION_REGISTRATION_SCHEMA or self.alpha_family != SIGNED_WORK_FIXED_PRIMAL_ALPHA_FAMILY:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "unsupported fixed-primal registration schema or alpha family",
            )

    @property
    def registration_hash(self) -> str:
        return content_hash(self.to_payload(), prefix=_PREFIX + "registration-")

    def to_payload(self) -> dict[str, object]:
        return {
            name: (
                getattr(self, name).seal_hash
                if name == "exact_census_source_seal"
                and getattr(self, name) is not None
                else getattr(self, name)
            )
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    @property
    def population_counts(self) -> tuple[int, int]:
        return tuple(item.population_count for item in self.strata)  # type: ignore[return-value]

    @property
    def registered_counts(self) -> tuple[int, int]:
        return tuple(item.registered_count for item in self.strata)  # type: ignore[return-value]


def _exact_census_strata_from_axes(
    *,
    stratum_axes: tuple[
        tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]],
        tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]],
    ],
) -> tuple[
    tuple[V1SignedWorkFixedPrimalProjectionStratumRegistration, ...],
    tuple[tuple[int, int], tuple[int, int]],
    tuple[str, str, str, str],
]:
    counts = tuple(len(item[1]) for item in stratum_axes)
    if any(count <= 0 for count in counts):
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
            "an exact fitting census requires both nonempty strata",
        )
    total = sum(counts)
    fractions = tuple(
        (
            Fraction(count, total).numerator,
            Fraction(count, total).denominator,
        )
        for count in counts
    )
    first_weight = float(Fraction(*fractions[0]))
    weights = (first_weight, 1.0 - first_weight)
    strata = tuple(
        V1SignedWorkFixedPrimalProjectionStratumRegistration(
            stratum_id=item[0],
            stratum_weight=weight,
            population_count=count,
            population_context_axis_hash=_axis_hash(
                f"exact_census:{item[0]}:context", item[1]
            ),
            population_draw_axis_hash=_axis_hash(
                f"exact_census:{item[0]}:draw", item[2]
            ),
            population_measurement_axis_hash=_axis_hash(
                f"exact_census:{item[0]}:measurement", item[3]
            ),
            population_capture_axis_hash=_axis_hash(
                f"exact_census:{item[0]}:capture", item[4]
            ),
            registered_context_ids=item[1],
            registered_draw_ids=item[2],
            registered_measurement_ids=item[3],
            registered_capture_ids=item[4],
        )
        for item, count, weight in zip(
            stratum_axes, counts, weights, strict=True
        )
    )
    source_axis_hashes = tuple(
        _axis_hash(
            f"exact_census:source:{role}",
            tuple(value for item in stratum_axes for value in item[index]),
        )
        for index, role in enumerate(
            ("context", "draw", "measurement", "capture"), start=1
        )
    )
    return strata, fractions, source_axis_hashes  # type: ignore[return-value]


def issue_v1_signed_work_fixed_primal_v7_d1_exact_census_registration(
    *,
    d1_axis_plan: object,
    registration_id: str,
    model_identity_hash: str,
    chart_identity_hash: str,
    current_mask_hash: str,
    proposal_generation: int,
    route_axis_hash: str,
    route_id_axis_hash: str,
    witness_stream_source_hash: str,
    same_capture_vjp_source_hash: str,
    corrected_evidence_stream_source_hash: str,
    literal_j_stream_source_hash: str,
    causal_registration_receipt_hash: str,
    fixed_primal_id: str,
    fixed_primal_source_receipt_hash: str,
    corrected_support_source_receipt_hash: str,
    literal_j_support_source_receipt_hash: str,
    alpha_ledger_id: str,
    alpha_coordinate_id: str,
    look_ordinal: int,
) -> V1SignedWorkFixedPrimalProjectionRegistration:
    """Issue alpha-free fitting-census authority only from one live V7 D1 plan."""

    from .modern_v1_sampled_v7_d_fixed_source import (
        V1SampledV7D1CanonicalAxisPlan,
    )

    if type(d1_axis_plan) is not V1SampledV7D1CanonicalAxisPlan:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
            "exact V7 D1 census registration requires its live source plan",
        )
    d1_axis_plan.assert_live()
    source_plan_hash = d1_axis_plan.plan_hash
    counts = d1_axis_plan.d1_stratum_counts
    records = d1_axis_plan.records
    stratum_axes = []
    cursor = 0
    for stratum_id, count in zip(("target", "complement"), counts, strict=True):
        selected = records[cursor : cursor + count]
        cursor += count
        stratum_axes.append(
            (
                stratum_id,
                tuple(item.context_id for item in selected),
                tuple(item.draw_id for item in selected),
                tuple(item.measurement_id for item in selected),
                tuple(item.capture_id for item in selected),
            )
        )
    strata, fractions, source_axis_hashes = _exact_census_strata_from_axes(
        stratum_axes=tuple(stratum_axes),  # type: ignore[arg-type]
    )
    if (
        d1_axis_plan.chart_identity_hash != chart_identity_hash
        or d1_axis_plan.route_axis_hash != route_axis_hash
        or d1_axis_plan.route_id_axis_hash != route_id_axis_hash
        or d1_axis_plan.current_mask_hash != current_mask_hash
        or d1_axis_plan.proposal_generation != proposal_generation
        or witness_stream_source_hash != source_plan_hash
        or d1_axis_plan.corrected_evidence_stream_source_hash
        != corrected_evidence_stream_source_hash
        or d1_axis_plan.literal_j_stream_source_hash
        != literal_j_stream_source_hash
    ):
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.STALE_IDENTITY,
            "V7 D1 source plan differs from the requested projection coordinate",
        )
    seal = V1SignedWorkFixedPrimalExactCensusSourceSeal(
        source_plan_hash=source_plan_hash,
        source_axis_hashes=source_axis_hashes,
        stratum_registration_hashes=tuple(
            item.registration_hash for item in strata
        ),  # type: ignore[arg-type]
        exact_stratum_weight_fractions=fractions,
        chart_identity_hash=chart_identity_hash,
        current_mask_hash=current_mask_hash,
        proposal_generation=proposal_generation,
        route_axis_hash=route_axis_hash,
        route_id_axis_hash=route_id_axis_hash,
        witness_stream_source_hash=witness_stream_source_hash,
        same_capture_vjp_source_hash=same_capture_vjp_source_hash,
        corrected_evidence_stream_source_hash=corrected_evidence_stream_source_hash,
        literal_j_stream_source_hash=literal_j_stream_source_hash,
        causal_registration_receipt_hash=causal_registration_receipt_hash,
        _issuer=_EXACT_CENSUS_REGISTRATION_ISSUER,
    )
    return V1SignedWorkFixedPrimalProjectionRegistration(
        registration_id=registration_id,
        model_identity_hash=model_identity_hash,
        chart_identity_hash=chart_identity_hash,
        current_mask_hash=current_mask_hash,
        proposal_generation=proposal_generation,
        route_axis_hash=route_axis_hash,
        route_id_axis_hash=route_id_axis_hash,
        witness_stream_source_hash=witness_stream_source_hash,
        same_capture_vjp_source_hash=same_capture_vjp_source_hash,
        corrected_evidence_stream_source_hash=corrected_evidence_stream_source_hash,
        literal_j_stream_source_hash=literal_j_stream_source_hash,
        causal_registration_receipt_hash=causal_registration_receipt_hash,
        fixed_primal_id=fixed_primal_id,
        fixed_primal_source_receipt_hash=fixed_primal_source_receipt_hash,
        corrected_support_source_receipt_hash=corrected_support_source_receipt_hash,
        literal_j_support_source_receipt_hash=literal_j_support_source_receipt_hash,
        alpha_ledger_id=alpha_ledger_id,
        alpha_coordinate_id=alpha_coordinate_id,
        look_ordinal=look_ordinal,
        familywise_alpha=None,
        strata=strata,
        exact_census_mode=True,
        sampling_alpha_spent=False,
        exact_census_source_plan_hash=source_plan_hash,
        exact_census_source_axis_hashes=source_axis_hashes,
        exact_census_stratum_weight_fractions=fractions,
        exact_census_source_seal=seal,
    )


def _issue_v1_signed_work_fixed_primal_exact_census_registration_for_tests(
    template: V1SignedWorkFixedPrimalProjectionRegistration,
) -> V1SignedWorkFixedPrimalProjectionRegistration:
    """Private tiny-source sibling; never available to production issuance."""

    if type(template) is not V1SignedWorkFixedPrimalProjectionRegistration:
        raise TypeError("private exact-census issuer requires an exact template")
    if template.exact_census_mode:
        raise ValueError("private exact-census template must start in sampled mode")
    axes = tuple(
        (
            stratum.stratum_id,
            stratum.registered_context_ids,
            stratum.registered_draw_ids,
            stratum.registered_measurement_ids,
            stratum.registered_capture_ids,
        )
        for stratum in template.strata
    )
    strata, fractions, source_axis_hashes = _exact_census_strata_from_axes(
        stratum_axes=axes,  # type: ignore[arg-type]
    )
    payload = template.to_payload()
    for name in (
        "familywise_alpha",
        "strata",
        "exact_census_mode",
        "sampling_alpha_spent",
        "exact_census_source_plan_hash",
        "exact_census_source_axis_hashes",
        "exact_census_stratum_weight_fractions",
        "exact_census_source_seal",
    ):
        payload.pop(name, None)
    seal = V1SignedWorkFixedPrimalExactCensusSourceSeal(
        source_plan_hash="private-test:frozen-D1-fitting-plan",
        source_axis_hashes=source_axis_hashes,
        stratum_registration_hashes=tuple(
            item.registration_hash for item in strata
        ),  # type: ignore[arg-type]
        exact_stratum_weight_fractions=fractions,
        chart_identity_hash=template.chart_identity_hash,
        current_mask_hash=template.current_mask_hash,
        proposal_generation=template.proposal_generation,
        route_axis_hash=template.route_axis_hash,
        route_id_axis_hash=template.route_id_axis_hash,
        witness_stream_source_hash=template.witness_stream_source_hash,
        same_capture_vjp_source_hash=template.same_capture_vjp_source_hash,
        corrected_evidence_stream_source_hash=(
            template.corrected_evidence_stream_source_hash
        ),
        literal_j_stream_source_hash=template.literal_j_stream_source_hash,
        causal_registration_receipt_hash=(
            template.causal_registration_receipt_hash
        ),
        _issuer=_EXACT_CENSUS_REGISTRATION_ISSUER,
    )
    return V1SignedWorkFixedPrimalProjectionRegistration(
        **payload,
        familywise_alpha=None,
        strata=strata,
        exact_census_mode=True,
        sampling_alpha_spent=False,
        exact_census_source_plan_hash="private-test:frozen-D1-fitting-plan",
        exact_census_source_axis_hashes=source_axis_hashes,
        exact_census_stratum_weight_fractions=fractions,
        exact_census_source_seal=seal,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkFixedPrimalProjectionCUDAHBMForecast:
    design_hash: str
    registration_hash: str
    native_fiber_axis_receipt_hash: str
    route_count: int
    cut_count: int
    fiber_count: int
    source_input_bytes: int
    persistent_accumulator_bytes: int
    borrowed_row_batch_bytes_upper: int
    temporary_tile_bytes_upper: int
    compact_axis_bytes_upper: int
    registered_peak_bytes_upper: int
    incremental_allocation_bytes_upper: int
    projected_total_occupied_bytes_upper: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    hbm_ceiling_fraction: float
    admitted: bool
    route_scale_fiber_labels_materialized: bool = False
    schema: str = SIGNED_WORK_FIXED_PRIMAL_PROJECTION_HBM_SCHEMA
    _forecast_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _admission_memory: tuple[int, int] | None) -> None:
        for name in ("design_hash", "registration_hash", "native_fiber_axis_receipt_hash"):
            _identity(getattr(self, name), name=name)
        for name in ("route_count", "cut_count", "fiber_count"):
            _positive_int(getattr(self, name), name=name)
        for name in (
            "source_input_bytes",
            "persistent_accumulator_bytes",
            "borrowed_row_batch_bytes_upper",
            "temporary_tile_bytes_upper",
            "compact_axis_bytes_upper",
            "registered_peak_bytes_upper",
            "incremental_allocation_bytes_upper",
            "projected_total_occupied_bytes_upper",
            "device_total_memory_bytes",
            "device_free_memory_bytes",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        ceiling = _finite_positive(self.hbm_ceiling_fraction, name="HBM ceiling fraction")
        if ceiling > 0.85 or type(self.admitted) is not bool or self.route_scale_fiber_labels_materialized is not False:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "invalid HBM forecast flags",
            )
        if self.schema != SIGNED_WORK_FIXED_PRIMAL_PROJECTION_HBM_SCHEMA:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "unsupported HBM forecast schema",
            )
        if self.device_free_memory_bytes > self.device_total_memory_bytes:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "device free memory cannot exceed total memory",
            )
        incremental = (
            self.persistent_accumulator_bytes
            + self.borrowed_row_batch_bytes_upper
            + self.temporary_tile_bytes_upper
            + self.compact_axis_bytes_upper
        )
        peak = self.source_input_bytes + incremental
        projected = (
            self.device_total_memory_bytes
            - self.device_free_memory_bytes
            + incremental
        )
        available, allocated = _admission_memory or (
            self.device_free_memory_bytes,
            self.device_total_memory_bytes - self.device_free_memory_bytes,
        )
        expected_admitted = bool(
            incremental <= available
            and allocated + incremental
            <= math.floor(
                self.device_total_memory_bytes * self.hbm_ceiling_fraction
            )
        )
        if (
            self.incremental_allocation_bytes_upper != incremental
            or self.registered_peak_bytes_upper != peak
            or self.projected_total_occupied_bytes_upper != projected
            or self.admitted != expected_admitted
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "HBM forecast does not replay its allocation/occupancy formula",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if not name.startswith("_")}

    def to_payload(self) -> dict[str, object]:
        return self._payload()

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "hbm-")
        if self._forecast_hash is not None and self._forecast_hash != observed:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.MUTATED_CUSTODY,
                "HBM forecast was mutated",
            )
        object.__setattr__(self, "_forecast_hash", observed)
        return observed


def build_v1_signed_work_fixed_primal_projection_cuda_hbm_forecast(
    *,
    design: V1SignedWorkFixedPrimalProjectionCUDAExecutionDesign,
    registration: V1SignedWorkFixedPrimalProjectionRegistration,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
    _admission_memory: tuple[int, int] | None = None,
) -> V1SignedWorkFixedPrimalProjectionCUDAHBMForecast:
    if type(design) is not V1SignedWorkFixedPrimalProjectionCUDAExecutionDesign or type(registration) is not V1SignedWorkFixedPrimalProjectionRegistration or type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
            "HBM forecast requires exact design, registration, and native-fiber receipt types",
        )
    native_fiber_axis_receipt.assert_integrity()
    if registration.route_axis_hash != native_fiber_axis_receipt.route_axis_hash or registration.route_id_axis_hash != native_fiber_axis_receipt.route_id_axis_hash:
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.STALE_IDENTITY,
            "registration route axes differ from the native-fiber receipt",
        )
    routes = native_fiber_axis_receipt.route_count
    windowed_source = (
        design.exact_census_windowed_source or design.sampled_windowed_source
    )
    # A census-windowed source derives support maxima in the accumulator, so
    # only the frozen primal is borrowed at construction.  Sampled modes still
    # borrow z plus the four preregistered support vectors.
    source_input = (
        routes * _F64_BYTES
        if design.exact_census_windowed_source
        else 5 * routes * _F64_BYTES
    )
    # The streaming state owns eleven FP64 vectors and one routewise ambiguity
    # bitmap.  Finalization temporarily owns
    # F/Q support envelopes, centers, split radii, and total radii before it
    # evicts everything except z plus four output vectors.  Twenty-seven route
    # vectors is a conservative measured-allocation upper envelope.
    persistent = 27 * routes * _F64_BYTES + routes * _BOOL_BYTES
    tile = min(routes, design.route_tile_size)
    borrowed_rows = (
        2 * design.maximum_batch_size * tile * _F64_BYTES
        if windowed_source
        else 2 * design.maximum_batch_size * routes * _F64_BYTES
    )
    # Full row banks are borrowed, but all finite/support validation is tiled.
    # Twelve batch-by-tile lanes cover fused batch fiber products, absolute
    # envelopes, signs, ambiguity masks, validation, and Q projections.
    temporary = (
        12 * design.maximum_batch_size * tile
        + 32 * design.maximum_batch_size
    ) * _F64_BYTES
    compact = native_fiber_axis_receipt.cut_count * 16 * _F64_BYTES
    if windowed_source:
        # Per-active-batch native-fiber sums, absolute sums, signs, and
        # ambiguity masks.  They are fiber-scale, never route-label arrays.
        compact += (
            3 * design.maximum_batch_size * native_fiber_axis_receipt.fiber_count
            * _F64_BYTES
            + design.maximum_batch_size
            * native_fiber_axis_receipt.fiber_count
            * _BOOL_BYTES
        )
    peak = source_input + persistent + borrowed_rows + temporary + compact
    incremental = persistent + borrowed_rows + temporary + compact
    total = _nonnegative_int(device_total_memory_bytes, name="device total memory bytes")
    free = _nonnegative_int(device_free_memory_bytes, name="device free memory bytes")
    occupied_now = max(0, total - free)
    projected_occupied = occupied_now + incremental
    available, allocated = _admission_memory or (free, occupied_now)
    admitted = bool(
        incremental <= available
        and allocated + incremental
        <= math.floor(total * design.hbm_ceiling_fraction)
    )
    return V1SignedWorkFixedPrimalProjectionCUDAHBMForecast(
        design_hash=design.design_hash,
        registration_hash=registration.registration_hash,
        native_fiber_axis_receipt_hash=native_fiber_axis_receipt.receipt_hash,
        route_count=routes,
        cut_count=native_fiber_axis_receipt.cut_count,
        fiber_count=native_fiber_axis_receipt.fiber_count,
        source_input_bytes=source_input,
        persistent_accumulator_bytes=persistent,
        borrowed_row_batch_bytes_upper=borrowed_rows,
        temporary_tile_bytes_upper=temporary,
        compact_axis_bytes_upper=compact,
        registered_peak_bytes_upper=peak,
        incremental_allocation_bytes_upper=incremental,
        projected_total_occupied_bytes_upper=projected_occupied,
        device_total_memory_bytes=total,
        device_free_memory_bytes=free,
        hbm_ceiling_fraction=design.hbm_ceiling_fraction,
        admitted=admitted,
        _admission_memory=_admission_memory,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkFixedPrimalZeroCellAmbiguityReceipt:
    registration_hash: str
    native_fiber_axis_receipt_hash: str
    evaluated_context_fiber_cells: int
    exact_binary64_zero_cell_count: int
    roundoff_ambiguous_cell_count: int
    exact_zero_counts_by_cut: tuple[int, ...]
    roundoff_ambiguous_counts_by_cut: tuple[int, ...]
    sign_policy: str = SIGNED_WORK_FIXED_PRIMAL_SIGN_POLICY
    strict_sign_claim_for_ambiguous_cells: bool = False
    ambiguity_routewise_effect_in_numerical_radius: bool = True
    schema: str = SIGNED_WORK_FIXED_PRIMAL_ZERO_CELL_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _identity(self.registration_hash, name="zero-cell registration hash")
        _identity(self.native_fiber_axis_receipt_hash, name="zero-cell native-fiber axis hash")
        evaluated = _nonnegative_int(self.evaluated_context_fiber_cells, name="evaluated context-fiber cells")
        exact = _nonnegative_int(self.exact_binary64_zero_cell_count, name="exact zero-cell count")
        ambiguous = _nonnegative_int(self.roundoff_ambiguous_cell_count, name="roundoff-ambiguous cell count")
        exact_by_cut = tuple(_nonnegative_int(item, name="cut exact-zero count") for item in self.exact_zero_counts_by_cut)
        ambiguous_by_cut = tuple(_nonnegative_int(item, name="cut ambiguity count") for item in self.roundoff_ambiguous_counts_by_cut)
        if not exact_by_cut or len(exact_by_cut) != len(ambiguous_by_cut) or sum(exact_by_cut) != exact or sum(ambiguous_by_cut) != ambiguous or exact > evaluated or ambiguous > evaluated:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "zero-cell summary does not close its cut/count axes",
            )
        object.__setattr__(self, "evaluated_context_fiber_cells", evaluated)
        object.__setattr__(self, "exact_binary64_zero_cell_count", exact)
        object.__setattr__(self, "roundoff_ambiguous_cell_count", ambiguous)
        object.__setattr__(self, "exact_zero_counts_by_cut", exact_by_cut)
        object.__setattr__(self, "roundoff_ambiguous_counts_by_cut", ambiguous_by_cut)
        if self.sign_policy != SIGNED_WORK_FIXED_PRIMAL_SIGN_POLICY or self.strict_sign_claim_for_ambiguous_cells is not False or self.ambiguity_routewise_effect_in_numerical_radius is not True or self.schema != SIGNED_WORK_FIXED_PRIMAL_ZERO_CELL_SCHEMA:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
                "zero-cell receipt cannot claim a strict ambiguous-cell sign",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if not name.startswith("_")}

    def to_payload(self) -> dict[str, object]:
        return self._payload()

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "zero-cell-")
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.MUTATED_CUSTODY,
                "zero-cell receipt was mutated",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SignedWorkFixedPrimalProjectionCUDATelemetry:
    registration_hash: str
    device_identity: str
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    consumed_batches: int
    consumed_contexts: int
    route_count: int
    cut_count: int
    fiber_count: int
    route_tiles_per_context: int
    native_fiber_tiles_total: int
    scalar_device_to_host_transfers: int
    route_scale_device_to_host_transfers: int
    route_scale_fiber_labels_materialized: bool
    floating_atomics_used: bool
    deterministic_context_order: bool
    same_capture_rows_processed_in_batched_kernels: bool = True
    per_native_fiber_tile_scalar_device_to_host_transfers: int = 0
    compact_finalize_count_vector_device_to_host_transfers: int = 2
    fixed_primal_only: bool = True
    schema: str = SIGNED_WORK_FIXED_PRIMAL_PROJECTION_TELEMETRY_SCHEMA

    def __post_init__(self) -> None:
        for name in ("registration_hash", "device_identity"):
            _identity(getattr(self, name), name=name)
        for name in ("consumed_batches", "consumed_contexts", "route_tiles_per_context", "native_fiber_tiles_total", "scalar_device_to_host_transfers", "route_scale_device_to_host_transfers"):
            _nonnegative_int(getattr(self, name), name=name)
        for name in (
            "per_native_fiber_tile_scalar_device_to_host_transfers",
            "compact_finalize_count_vector_device_to_host_transfers",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        for name in ("route_count", "cut_count", "fiber_count"):
            _positive_int(getattr(self, name), name=name)
        if self.production_cuda_execution == self.private_cpu_reference_execution:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "telemetry must name exactly one execution mode",
            )
        if (
            self.route_scale_device_to_host_transfers != 0
            or self.route_scale_fiber_labels_materialized is not False
            or self.floating_atomics_used is not False
            or self.deterministic_context_order is not True
            or self.same_capture_rows_processed_in_batched_kernels is not True
            or self.per_native_fiber_tile_scalar_device_to_host_transfers != 0
            or self.compact_finalize_count_vector_device_to_host_transfers != 2
            or self.fixed_primal_only is not True
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.AUTHORITY_VIOLATION,
                "telemetry violates the bounded deterministic projection contract",
            )
        if self.schema != SIGNED_WORK_FIXED_PRIMAL_PROJECTION_TELEMETRY_SCHEMA:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "unsupported telemetry schema",
            )

    @property
    def telemetry_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "telemetry-")

    def to_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class V1SignedWorkFixedPrimalProjectionReceipt:
    registration_hash: str
    design_hash: str
    forecast_hash: str
    telemetry_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    route_count: int
    cut_count: int
    fiber_count: int
    fixed_primal_id: str
    fixed_primal_source_receipt_hash: str
    fixed_primal_descriptor_hash: str
    fixed_primal_device_fingerprint: str
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    causal_registration_receipt_hash: str
    stratum_ids: tuple[str, str]
    stratum_weights: tuple[float, float]
    population_counts: tuple[int, int]
    observed_counts: tuple[int, int]
    population_context_axis_hashes: tuple[str, str]
    population_draw_axis_hashes: tuple[str, str]
    population_measurement_axis_hashes: tuple[str, str]
    population_capture_axis_hashes: tuple[str, str]
    registered_prefix_context_axis_hashes: tuple[str, str]
    registered_prefix_draw_axis_hashes: tuple[str, str]
    registered_prefix_measurement_axis_hashes: tuple[str, str]
    registered_prefix_capture_axis_hashes: tuple[str, str]
    alpha_ledger_id: str
    alpha_coordinate_id: str
    look_ordinal: int
    familywise_alpha: float | None
    simultaneous_two_sided_statement_count: int
    per_statement_two_sided_alpha: float | None
    per_statement_one_sided_alpha: float | None
    corrected_support_source_receipt_hash: str
    literal_j_support_source_receipt_hash: str
    corrected_support_device_fingerprint: str
    literal_j_support_device_fingerprint: str
    signed_f_center_fingerprint: str
    signed_f_sampling_radius_fingerprint: str
    signed_f_numerical_radius_fingerprint: str
    signed_f_total_radius_fingerprint: str
    literal_j_q_gradient_center_fingerprint: str
    literal_j_q_gradient_sampling_radius_fingerprint: str
    literal_j_q_gradient_numerical_radius_fingerprint: str
    literal_j_q_gradient_total_radius_fingerprint: str
    zero_cell_ambiguity_receipt_hash: str
    zero_cell_route_bitmap_fingerprint: str
    exact_finite_population_census: bool
    signed_f_sampling_radius_identically_zero: bool
    literal_j_q_sampling_radius_identically_zero: bool
    device_identity: str
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    exact_census_mode: bool = False
    sampling_alpha_spent: bool = True
    census_supports_derived_from_sealed_rows: bool = False
    windowed_two_pass_source_replay: bool = False
    sampled_windowed_two_pass_source_replay: bool = False
    exact_census_source_plan_hash: str | None = None
    exact_census_stratum_weight_fractions: (
        tuple[tuple[int, int], tuple[int, int]] | None
    ) = None
    census_replay_digest_family_hash: str | None = None
    sampled_replay_digest_family_hash: str | None = None
    pass1_pass2_value_identity_verified: bool = False
    predictable_before_witness_outcomes: bool = True
    sealed_without_replacement_permutations: bool = True
    source_projection_authority: bool = True
    fixed_z_only_no_solution_sensitivity: bool = True
    normal_cone_authority: bool = False
    action_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    production_promotion_authority: bool = False
    pareto_authority: bool = False
    f_equation: str = SIGNED_WORK_FIXED_PRIMAL_F_EQUATION
    q_equation: str = SIGNED_WORK_FIXED_PRIMAL_Q_EQUATION
    sign_policy: str = SIGNED_WORK_FIXED_PRIMAL_SIGN_POLICY
    alpha_family: str = SIGNED_WORK_FIXED_PRIMAL_ALPHA_FAMILY
    authority_scope: str = SIGNED_WORK_FIXED_PRIMAL_PROJECTION_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_FIXED_PRIMAL_PROJECTION_RECEIPT_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
                "projection receipts are source-issued only",
            )
        for name in (
            "registration_hash", "design_hash", "forecast_hash", "telemetry_hash",
            "model_identity_hash", "chart_identity_hash", "current_mask_hash",
            "route_axis_hash", "route_id_axis_hash", "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash", "fixed_primal_id", "fixed_primal_source_receipt_hash",
            "fixed_primal_descriptor_hash", "fixed_primal_device_fingerprint",
            "witness_stream_source_hash", "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash", "literal_j_stream_source_hash",
            "causal_registration_receipt_hash",
            "alpha_ledger_id", "alpha_coordinate_id", "corrected_support_source_receipt_hash",
            "literal_j_support_source_receipt_hash", "corrected_support_device_fingerprint",
            "literal_j_support_device_fingerprint", "signed_f_center_fingerprint",
            "signed_f_sampling_radius_fingerprint", "signed_f_numerical_radius_fingerprint",
            "signed_f_total_radius_fingerprint", "literal_j_q_gradient_center_fingerprint",
            "literal_j_q_gradient_sampling_radius_fingerprint",
            "literal_j_q_gradient_numerical_radius_fingerprint",
            "literal_j_q_gradient_total_radius_fingerprint", "zero_cell_ambiguity_receipt_hash",
            "zero_cell_route_bitmap_fingerprint",
            "device_identity",
        ):
            _identity(getattr(self, name), name=name)
        for name in ("proposal_generation",):
            _nonnegative_int(getattr(self, name), name=name)
        for name in ("route_count", "cut_count", "fiber_count", "look_ordinal", "simultaneous_two_sided_statement_count"):
            _positive_int(getattr(self, name), name=name)
        if (
            len(self.stratum_ids) != 2
            or len(set(self.stratum_ids)) != 2
            or any(type(item) is not str or not item for item in self.stratum_ids)
            or len(self.stratum_weights) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, Real)
                or not math.isfinite(float(item))
                or float(item) <= 0.0
                for item in self.stratum_weights
            )
            or math.fsum(self.stratum_weights) != 1.0
        ):
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN, "receipt stratum axis is invalid")
        if len(self.population_counts) != 2 or len(self.observed_counts) != 2:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN, "receipt count axes must have two entries")
        populations = tuple(
            _positive_int(item, name="receipt population count")
            for item in self.population_counts
        )
        observed = tuple(
            _positive_int(item, name="receipt observed count")
            for item in self.observed_counts
        )
        if any(value > population for value, population in zip(observed, populations, strict=True)):
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN, "receipt observed count exceeds its population")
        for pair_name in (
            "population_counts", "observed_counts", "population_context_axis_hashes",
            "population_draw_axis_hashes", "population_measurement_axis_hashes",
            "population_capture_axis_hashes", "registered_prefix_context_axis_hashes",
            "registered_prefix_draw_axis_hashes", "registered_prefix_measurement_axis_hashes",
            "registered_prefix_capture_axis_hashes",
        ):
            if len(getattr(self, pair_name)) != 2:
                _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN, f"{pair_name} must have two entries")
        for pair_name in (
            "population_context_axis_hashes", "population_draw_axis_hashes",
            "population_measurement_axis_hashes", "population_capture_axis_hashes",
            "registered_prefix_context_axis_hashes", "registered_prefix_draw_axis_hashes",
            "registered_prefix_measurement_axis_hashes", "registered_prefix_capture_axis_hashes",
        ):
            for item in getattr(self, pair_name):
                _identity(item, name=pair_name)
        if self.simultaneous_two_sided_statement_count != 2 * self.route_count:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN, "alpha statement count must be two gradients times all routes")
        census_flags = (
            self.exact_census_mode,
            self.sampling_alpha_spent,
            self.census_supports_derived_from_sealed_rows,
            self.windowed_two_pass_source_replay,
            self.sampled_windowed_two_pass_source_replay,
            self.pass1_pass2_value_identity_verified,
        )
        if any(type(value) is not bool for value in census_flags):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
                "census source flags must be exact booleans",
            )
        if self.exact_census_mode:
            if (
                self.familywise_alpha is not None
                or self.per_statement_two_sided_alpha is not None
                or self.per_statement_one_sided_alpha is not None
                or self.sampling_alpha_spent is not False
                or self.census_supports_derived_from_sealed_rows is not True
                or self.windowed_two_pass_source_replay is not True
                or self.sampled_windowed_two_pass_source_replay is not False
                or type(self.exact_census_source_plan_hash) is not str
                or not self.exact_census_source_plan_hash
                or self.exact_census_stratum_weight_fractions
                != tuple(
                    (
                        Fraction(count, sum(populations)).numerator,
                        Fraction(count, sum(populations)).denominator,
                    )
                    for count in populations
                )
                or type(self.census_replay_digest_family_hash) is not str
                or not self.census_replay_digest_family_hash
                or self.sampled_replay_digest_family_hash is not None
                or self.pass1_pass2_value_identity_verified is not True
            ):
                _refuse(
                    V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                    "alpha-free census receipt requires sealed-row windowed support derivation",
                )
        else:
            if (
                self.exact_census_source_plan_hash is not None
                or self.exact_census_stratum_weight_fractions is not None
                or self.census_replay_digest_family_hash is not None
            ):
                _refuse(
                    V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                    "sampled receipt cannot carry exact-census replay authority",
                )
            if self.windowed_two_pass_source_replay is not False:
                _refuse(
                    V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                    "sampled receipt cannot set the exact-census window replay flag",
                )
            familywise_alpha = _finite_probability(
                self.familywise_alpha,
                name="receipt familywise alpha",
            )
            per_two = _finite_probability(
                self.per_statement_two_sided_alpha,
                name="per-statement two-sided alpha",
            )
            per_one = _finite_probability(
                self.per_statement_one_sided_alpha,
                name="per-statement one-sided alpha",
            )
            expected_two = familywise_alpha / self.simultaneous_two_sided_statement_count
            expected_one = expected_two / 2.0
            if (
                per_two != expected_two
                or per_one != expected_one
                or self.sampling_alpha_spent is not True
                or self.census_supports_derived_from_sealed_rows is not False
            ):
                _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN, "alpha split does not replay")
            if self.sampled_windowed_two_pass_source_replay:
                if (
                    type(self.sampled_replay_digest_family_hash) is not str
                    or not self.sampled_replay_digest_family_hash
                    or self.pass1_pass2_value_identity_verified is not True
                ):
                    _refuse(
                        V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                        "sampled windowed receipt requires pass identity replay",
                    )
            elif (
                self.sampled_replay_digest_family_hash is not None
                or self.pass1_pass2_value_identity_verified is not False
            ):
                _refuse(
                    V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                    "non-windowed sampled receipt cannot carry sampled replay identity",
                )
        if (
            type(self.production_cuda_execution) is not bool
            or type(self.private_cpu_reference_execution) is not bool
            or self.production_cuda_execution == self.private_cpu_reference_execution
        ):
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN, "receipt must name exactly one execution mode")
        if type(self.exact_finite_population_census) is not bool or type(self.signed_f_sampling_radius_identically_zero) is not bool or type(self.literal_j_q_sampling_radius_identically_zero) is not bool:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE, "census/radius flags must be exact bools")
        if self.exact_finite_population_census and (not self.signed_f_sampling_radius_identically_zero or not self.literal_j_q_sampling_radius_identically_zero):
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.NUMERICAL_REFUSAL, "finite-population census must have zero sampling radius")
        if self.exact_finite_population_census != (observed == populations):
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN, "census flag does not replay population counts")
        closed_flags = (
            self.normal_cone_authority, self.action_stability_authority,
            self.c_stream_authority, self.calibration_authority,
            self.oracle_authority, self.commit_authority,
            self.production_promotion_authority, self.pareto_authority,
        )
        if self.source_projection_authority is not True or self.fixed_z_only_no_solution_sensitivity is not True or any(value is not False for value in closed_flags):
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.AUTHORITY_VIOLATION, "projection receipt claims downstream authority")
        if self.predictable_before_witness_outcomes is not True or self.sealed_without_replacement_permutations is not True:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.AUTHORITY_VIOLATION, "receipt lost its predictable without-replacement source scope")
        if self.f_equation != SIGNED_WORK_FIXED_PRIMAL_F_EQUATION or self.q_equation != SIGNED_WORK_FIXED_PRIMAL_Q_EQUATION or self.sign_policy != SIGNED_WORK_FIXED_PRIMAL_SIGN_POLICY or self.alpha_family != SIGNED_WORK_FIXED_PRIMAL_ALPHA_FAMILY or self.authority_scope != SIGNED_WORK_FIXED_PRIMAL_PROJECTION_AUTHORITY_SCOPE or self.schema != SIGNED_WORK_FIXED_PRIMAL_PROJECTION_RECEIPT_SCHEMA:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.AUTHORITY_VIOLATION, "projection receipt semantics were broadened")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__ if not name.startswith("_")}

    def to_payload(self) -> dict[str, object]:
        return self._payload()

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self) -> str:
        observed = content_hash(self._payload(), prefix=_PREFIX + "receipt-")
        if self._receipt_hash is not None and self._receipt_hash != observed:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.MUTATED_CUSTODY, "projection receipt was mutated")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


def _device_vector_fingerprint(value: Any, *, role: str, tile_size: int) -> str:
    """Return a bounded scalar sketch; never copy route-scale values to host."""

    _tensor_metadata(value, name=role)
    count = int(value.numel())
    device = value.device
    total = torch.zeros((), dtype=torch.float64, device=device)
    squares = torch.zeros((), dtype=torch.float64, device=device)
    checksum_a = torch.zeros((), dtype=torch.float64, device=device)
    checksum_b = torch.zeros((), dtype=torch.float64, device=device)
    minimum = torch.full((), math.inf, dtype=torch.float64, device=device)
    maximum = torch.full((), -math.inf, dtype=torch.float64, device=device)
    for start in range(0, count, tile_size):
        stop = min(count, start + tile_size)
        source_chunk = value[start:stop]
        chunk = source_chunk.to(dtype=torch.float64)
        ordinals = torch.arange(start, stop, dtype=torch.int64, device=device)
        weights_a = ((ordinals.remainder(104_729) + 1).to(torch.float64) / 104_729.0)
        weights_b = ((ordinals.remainder(130_363) + 1).to(torch.float64) / 130_363.0)
        total.add_(torch.sum(chunk))
        squares.add_(torch.sum(chunk * chunk))
        checksum_a.add_(torch.sum(chunk * weights_a))
        checksum_b.add_(torch.sum(chunk * weights_b))
        minimum.copy_(torch.minimum(minimum, torch.min(chunk)))
        maximum.copy_(torch.maximum(maximum, torch.max(chunk)))
    scalars = tuple(float(item.item()) for item in (total, squares, checksum_a, checksum_b, minimum, maximum))
    if not all(math.isfinite(item) for item in scalars):
        _refuse(
            V1SignedWorkFixedPrimalProjectionCUDARefusalCode.NUMERICAL_REFUSAL,
            f"{role} fingerprint encountered nonfinite output",
        )
    normalized = tuple(0.0 if item == 0.0 else item for item in scalars)
    return content_hash(
        {
            "schema": "signed_work_fixed_primal_device_vector_fingerprint_v1",
            "role": role,
            "dtype": str(value.dtype),
            "shape": (count,),
            "device_type": value.device.type,
            "scalar_sketch_hex": tuple(item.hex() for item in normalized),
            "route_scale_host_values_transferred": 0,
        },
        prefix=_PREFIX + "device-fingerprint-",
    )


class V1SignedWorkFixedPrimalProjectionDeviceLease:
    """Borrow the six retained vectors without transferring them to host."""

    __slots__ = ("_handle", "_closed")

    def __init__(self, handle: "V1SignedWorkFixedPrimalProjectionDeviceHandle") -> None:
        self._handle = handle
        self._closed = False

    def _tensor(self, role: str) -> Any:
        if self._closed:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.EVICTED, "device lease is closed")
        return self._handle._borrow_tensor(role)

    @property
    def fixed_primal(self) -> Any:
        return self._tensor("fixed_primal")

    @property
    def signed_f_subgradient_center(self) -> Any:
        return self._tensor("signed_f_subgradient_center")

    @property
    def signed_f_subgradient_radius(self) -> Any:
        return self._tensor("signed_f_subgradient_radius")

    @property
    def literal_j_q_gradient_center(self) -> Any:
        return self._tensor("literal_j_q_gradient_center")

    @property
    def literal_j_q_gradient_radius(self) -> Any:
        return self._tensor("literal_j_q_gradient_radius")

    @property
    def zero_cell_ambiguity_bitmap(self) -> Any:
        """Routewise union of every observed zero/roundoff-ambiguous fiber cell."""

        return self._tensor("zero_cell_ambiguity_bitmap")

    @property
    def zero_cell_ambiguity_receipt(self) -> V1SignedWorkFixedPrimalZeroCellAmbiguityReceipt:
        if self._closed:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.EVICTED, "device lease is closed")
        return self._handle.zero_cell_ambiguity_receipt

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._handle.assert_live_identity()
        finally:
            self._closed = True
            self._handle._release_lease()

    def __enter__(self) -> "V1SignedWorkFixedPrimalProjectionDeviceLease":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __copy__(self) -> NoReturn:
        raise TypeError("fixed-primal projection leases are noncopyable")

    def __deepcopy__(self, memo: object) -> NoReturn:
        raise TypeError("fixed-primal projection leases are noncopyable")

    def __reduce__(self) -> NoReturn:
        raise TypeError("fixed-primal projection leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        raise TypeError("fixed-primal projection leases are nonserializable")


class V1SignedWorkFixedPrimalProjectionDeviceHandle:
    """Mutation-checked exclusive custody of the fixed-z projection vectors."""

    _ROLES: Final = (
        "fixed_primal",
        "signed_f_subgradient_center",
        "signed_f_subgradient_radius",
        "literal_j_q_gradient_center",
        "literal_j_q_gradient_radius",
        "zero_cell_ambiguity_bitmap",
    )

    def __init__(
        self,
        *,
        receipt: V1SignedWorkFixedPrimalProjectionReceipt,
        zero_cell_ambiguity_receipt: V1SignedWorkFixedPrimalZeroCellAmbiguityReceipt,
        tensors: dict[str, Any],
        fingerprints: dict[str, str],
        route_tile_size: int,
        issuance_token: object,
    ) -> None:
        if issuance_token is not _ISSUER_TOKEN or type(receipt) is not V1SignedWorkFixedPrimalProjectionReceipt or type(zero_cell_ambiguity_receipt) is not V1SignedWorkFixedPrimalZeroCellAmbiguityReceipt:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE, "device handles are source-issued only")
        if set(tensors) != set(self._ROLES) or set(fingerprints) != set(self._ROLES):
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE, "device handle custody is incomplete")
        self._receipt = receipt
        self._zero = zero_cell_ambiguity_receipt
        self._tensors = tensors
        self._fingerprints = dict(fingerprints)
        self._metadata = {role: _tensor_metadata(tensors[role], name=role) for role in self._ROLES}
        self._route_tile_size = int(route_tile_size)
        self._active_leases = 0
        self._evicted = False
        self.assert_live_identity()

    @property
    def receipt(self) -> V1SignedWorkFixedPrimalProjectionReceipt:
        return self._receipt

    @property
    def zero_cell_ambiguity_receipt(self) -> V1SignedWorkFixedPrimalZeroCellAmbiguityReceipt:
        self.assert_live_identity()
        return self._zero

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    @property
    def active_lease_count(self) -> int:
        return self._active_leases

    def assert_live_identity(self) -> str:
        if self._evicted:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.EVICTED, "fixed-primal projection custody was evicted")
        self._receipt.assert_integrity()
        self._zero.assert_integrity()
        for role in self._ROLES:
            tensor = self._tensors.get(role)
            if tensor is None or _tensor_metadata(tensor, name=role) != self._metadata[role]:
                _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.MUTATED_CUSTODY, f"{role} tensor identity or version changed")
            if _device_vector_fingerprint(tensor, role=role, tile_size=self._route_tile_size) != self._fingerprints[role]:
                _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.MUTATED_CUSTODY, f"{role} device fingerprint changed")
        return content_hash(
            {
                "schema": "signed_work_fixed_primal_live_device_identity_v1",
                "receipt_hash": self._receipt.receipt_hash,
                "zero_cell_receipt_hash": self._zero.receipt_hash,
                "fingerprints": tuple((role, self._fingerprints[role]) for role in self._ROLES),
                "process_identity_bound": True,
                "nonserializable": True,
            },
            prefix=_PREFIX + "live-identity-",
        )

    def borrow(self) -> V1SignedWorkFixedPrimalProjectionDeviceLease:
        self.assert_live_identity()
        self._active_leases += 1
        return V1SignedWorkFixedPrimalProjectionDeviceLease(self)

    def _borrow_tensor(self, role: str) -> Any:
        if role not in self._ROLES or self._evicted:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.EVICTED, "requested device tensor is unavailable")
        self.assert_live_identity()
        return self._tensors[role]

    def _release_lease(self) -> None:
        if self._active_leases <= 0:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.ACTIVE_LEASE, "device lease accounting underflow")
        self._active_leases -= 1

    def evict(self) -> None:
        if self._evicted:
            return
        if self._active_leases:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.ACTIVE_LEASE, "cannot evict while a device lease is active")
        # Eviction is a fail-closed lifetime transition, not an integrity
        # certification operation.  Preserve a mutation refusal, but burn the
        # tensors even when that final verification fails so an invalidated
        # projection can never remain borrowable after cleanup.
        try:
            self.assert_live_identity()
        finally:
            self._tensors.clear()
            self._evicted = True

    def __copy__(self) -> NoReturn:
        raise TypeError("fixed-primal projection handles are noncopyable")

    def __deepcopy__(self, memo: object) -> NoReturn:
        raise TypeError("fixed-primal projection handles are noncopyable")

    def __reduce__(self) -> NoReturn:
        raise TypeError("fixed-primal projection handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        raise TypeError("fixed-primal projection handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1SignedWorkFixedPrimalProjectionCUDAExecution:
    receipt: V1SignedWorkFixedPrimalProjectionReceipt
    zero_cell_ambiguity_receipt: V1SignedWorkFixedPrimalZeroCellAmbiguityReceipt
    telemetry: V1SignedWorkFixedPrimalProjectionCUDATelemetry
    forecast: V1SignedWorkFixedPrimalProjectionCUDAHBMForecast
    device_handle: V1SignedWorkFixedPrimalProjectionDeviceHandle = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1SignedWorkFixedPrimalProjectionReceipt or type(self.zero_cell_ambiguity_receipt) is not V1SignedWorkFixedPrimalZeroCellAmbiguityReceipt or type(self.telemetry) is not V1SignedWorkFixedPrimalProjectionCUDATelemetry or type(self.forecast) is not V1SignedWorkFixedPrimalProjectionCUDAHBMForecast or type(self.device_handle) is not V1SignedWorkFixedPrimalProjectionDeviceHandle:
            raise TypeError("fixed-primal execution requires exact component types")
        if self.receipt.zero_cell_ambiguity_receipt_hash != self.zero_cell_ambiguity_receipt.receipt_hash or self.receipt.telemetry_hash != self.telemetry.telemetry_hash or self.receipt.forecast_hash != self.forecast.forecast_hash or self.device_handle.receipt.receipt_hash != self.receipt.receipt_hash:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.STALE_IDENTITY, "execution bundle identities do not close")
        self.device_handle.assert_live_identity()


class V1SignedWorkFixedPrimalProjectionCUDAAccumulator:
    """Stream one registered fixed-look ``D_W`` prefix into device statistics."""

    def __init__(
        self,
        *,
        registration: V1SignedWorkFixedPrimalProjectionRegistration,
        native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
        fixed_primal: Any,
        fixed_primal_id: str,
        fixed_primal_source_receipt_hash: str,
        corrected_evidence_abs_support_by_stratum: Any | None,
        literal_j_abs_support_by_stratum: Any | None,
        corrected_support_source_receipt_hash: str,
        literal_j_support_source_receipt_hash: str,
        design: V1SignedWorkFixedPrimalProjectionCUDAExecutionDesign | None = None,
        device_total_memory_bytes: int | None = None,
        device_free_memory_bytes: int | None = None,
    ) -> None:
        if type(registration) is not V1SignedWorkFixedPrimalProjectionRegistration or type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE, "accumulator requires exact registration and native-fiber receipt types")
        registration_hash = registration.registration_hash
        native_fiber_axis_receipt.assert_integrity()
        if registration.route_axis_hash != native_fiber_axis_receipt.route_axis_hash or registration.route_id_axis_hash != native_fiber_axis_receipt.route_id_axis_hash:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.STALE_IDENTITY, "registration and native-fiber route axes differ")
        if design is None:
            design = V1SignedWorkFixedPrimalProjectionCUDAExecutionDesign()
        if type(design) is not V1SignedWorkFixedPrimalProjectionCUDAExecutionDesign:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE, "execution design must have exact type")
        route_count = native_fiber_axis_receipt.route_count
        fixed_primal = _require_fp64_tensor(
            fixed_primal,
            name="fixed primal",
            ndim=1,
            shape=(route_count,),
            scan_finite=False,
        )
        device = fixed_primal.device
        supports_shape = (2, route_count)
        if design.exact_census_windowed_source:
            if (
                registration.exact_census_mode is not True
                or registration.sampling_alpha_spent is not False
                or corrected_evidence_abs_support_by_stratum is not None
                or literal_j_abs_support_by_stratum is not None
            ):
                _refuse(
                    V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                    "windowed census mode derives supports from a complete alpha-free census",
                )
            corrected_support = torch.zeros(
                supports_shape,
                dtype=torch.float64,
                device=device,
            )
            literal_j_support = torch.zeros_like(corrected_support)
        else:
            if registration.exact_census_mode:
                _refuse(
                    V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                    "exact census registration requires the windowed census source",
                )
            corrected_support = _require_fp64_tensor(
                corrected_evidence_abs_support_by_stratum,
                name="corrected-evidence absolute support",
                ndim=2,
                shape=supports_shape,
                device=device,
                scan_finite=False,
            )
            literal_j_support = _require_fp64_tensor(
                literal_j_abs_support_by_stratum,
                name="literal-J absolute support",
                ndim=2,
                shape=supports_shape,
                device=device,
                scan_finite=False,
            )
        if design.production_cuda_required:
            if device.type != "cuda":
                _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.CPU_PRODUCTION_FALLBACK, "production fixed-primal projection refuses CPU fallback")
            if torch is None or not torch.cuda.is_available():
                _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.DEVICE_REQUIRED, "CUDA runtime is unavailable")
            warn_only = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)
            if (
                not bool(torch.are_deterministic_algorithms_enabled())
                or bool(warn_only())
                or os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}
            ):
                _refuse(
                    V1SignedWorkFixedPrimalProjectionCUDARefusalCode.DETERMINISM_DISABLED,
                    "production CUDA requires strict deterministic algorithms and a qualified CUBLAS workspace",
                )
        elif device.type == "cpu":
            if not design.allow_private_cpu_reference:
                _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.CPU_PRODUCTION_FALLBACK, "CPU execution is private-reference only")
        else:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.DEVICE_REQUIRED, "non-production execution supports only explicit CPU reference custody")
        if device.type == "cuda":
            if device_total_memory_bytes is not None or device_free_memory_bytes is not None:
                _refuse(
                    V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                    "production CUDA memory custody must come from live mem_get_info, not caller overrides",
                )
            free_observed, total_observed = torch.cuda.mem_get_info(device)
            admission_memory = (
                cuda_reusable_memory_bytes(device, framework=torch, free_bytes=int(free_observed)),
                int(torch.cuda.memory_allocated(device)),
            )
            total = int(total_observed)
            free = int(free_observed)
        else:
            if device_total_memory_bytes is None or device_free_memory_bytes is None:
                _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN, "private CPU reference requires explicit synthetic memory bounds")
            admission_memory = None
            total = int(device_total_memory_bytes)
            free = int(device_free_memory_bytes)
        forecast = build_v1_signed_work_fixed_primal_projection_cuda_hbm_forecast(
            design=design,
            registration=registration,
            native_fiber_axis_receipt=native_fiber_axis_receipt,
            device_total_memory_bytes=total,
            device_free_memory_bytes=free,
            _admission_memory=admission_memory,
        )
        if not forecast.admitted:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.HBM_FORECAST_EXCEEDED, "fixed-primal projection exceeds the registered 85-percent HBM envelope")

        source_invalid = torch.zeros((), dtype=torch.bool, device=device)
        for start in range(0, route_count, design.route_tile_size):
            stop = min(route_count, start + design.route_tile_size)
            z_tile = fixed_primal[start:stop]
            a_tile = corrected_support[:, start:stop]
            j_tile = literal_j_support[:, start:stop]
            source_invalid.logical_or_(
                (~torch.isfinite(z_tile)).any()
                | (z_tile < 0.0).any()
                | (z_tile > 1.0).any()
                | (~torch.isfinite(a_tile)).any()
                | (a_tile < 0.0).any()
                | (~torch.isfinite(j_tile)).any()
                | (j_tile < 0.0).any()
            )
        if bool(source_invalid.item()):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "fixed primal/support source contains a nonfinite, out-of-box, or negative value",
            )

        self._registration = registration
        self._registration_hash = registration_hash
        self._axis = native_fiber_axis_receipt
        self._design = design
        self._forecast = forecast
        self._device = device
        self._fixed_primal_id = _identity(fixed_primal_id, name="fixed primal ID")
        self._fixed_primal_source_receipt_hash = _identity(fixed_primal_source_receipt_hash, name="fixed primal source receipt hash")
        self._corrected_support_source_receipt_hash = _identity(corrected_support_source_receipt_hash, name="corrected support source receipt hash")
        self._literal_j_support_source_receipt_hash = _identity(literal_j_support_source_receipt_hash, name="literal-J support source receipt hash")
        if (
            self._fixed_primal_id != registration.fixed_primal_id
            or self._fixed_primal_source_receipt_hash != registration.fixed_primal_source_receipt_hash
            or self._corrected_support_source_receipt_hash != registration.corrected_support_source_receipt_hash
            or self._literal_j_support_source_receipt_hash != registration.literal_j_support_source_receipt_hash
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.STALE_IDENTITY,
                "fixed-primal or support-source custody differs from the pre-outcome registration",
            )

        # Copies are owned exclusively by this fixed-look accumulator.
        self._z = fixed_primal.detach().clone(memory_format=torch.contiguous_format)
        self._a_support = corrected_support.detach().clone(memory_format=torch.contiguous_format)
        self._j_support = literal_j_support.detach().clone(memory_format=torch.contiguous_format)
        self._f_sums = [torch.zeros(route_count, dtype=torch.float64, device=device) for _ in range(2)]
        self._q_sums = [torch.zeros(route_count, dtype=torch.float64, device=device) for _ in range(2)]
        self._f_ambiguity_sums = [torch.zeros(route_count, dtype=torch.float64, device=device) for _ in range(2)]
        self._zero_cell_ambiguity_bitmap = torch.zeros(
            route_count, dtype=torch.bool, device=device
        )
        self._observed = [0, 0]
        self._seen_context: set[str] = set()
        self._seen_draw: set[str] = set()
        self._seen_measurement: set[str] = set()
        self._seen_capture: set[str] = set()
        self._consumed_batches = 0
        self._scalar_d2h = 1
        self._exact_zero_by_cut = torch.zeros(
            self._axis.cut_count, dtype=torch.int64, device=device
        )
        self._ambiguous_by_cut = torch.zeros(
            self._axis.cut_count, dtype=torch.int64, device=device
        )
        self._finalized = False
        self._census_active: dict[str, Any] | None = None
        self._census_completed_contexts = 0
        self._census_replay_digests: list[str] = []
        self._sampled_replay_digests: list[str] = []

    @property
    def forecast(self) -> V1SignedWorkFixedPrimalProjectionCUDAHBMForecast:
        return self._forecast

    @property
    def observed_counts(self) -> tuple[int, int]:
        return tuple(self._observed)  # type: ignore[return-value]

    def _validate_call_identities(self, supplied: dict[str, object]) -> None:
        expected = {
            "model_identity_hash": self._registration.model_identity_hash,
            "chart_identity_hash": self._registration.chart_identity_hash,
            "current_mask_hash": self._registration.current_mask_hash,
            "proposal_generation": self._registration.proposal_generation,
            "witness_stream_source_hash": self._registration.witness_stream_source_hash,
            "same_capture_vjp_source_hash": self._registration.same_capture_vjp_source_hash,
            "corrected_evidence_stream_source_hash": self._registration.corrected_evidence_stream_source_hash,
            "literal_j_stream_source_hash": self._registration.literal_j_stream_source_hash,
            "alpha_ledger_id": self._registration.alpha_ledger_id,
            "alpha_coordinate_id": self._registration.alpha_coordinate_id,
            "look_ordinal": self._registration.look_ordinal,
        }
        if supplied != expected:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.STALE_IDENTITY, "stream call differs from the registered chart/mask/source/alpha coordinate")

    def consume_same_capture_batch(
        self,
        *,
        stratum_id: str,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        measurement_ids: Sequence[str],
        capture_ids: Sequence[str],
        corrected_evidence_rows: Any,
        literal_j_rows: Any,
        model_identity_hash: str,
        chart_identity_hash: str,
        current_mask_hash: str,
        proposal_generation: int,
        witness_stream_source_hash: str,
        same_capture_vjp_source_hash: str,
        corrected_evidence_stream_source_hash: str,
        literal_j_stream_source_hash: str,
        alpha_ledger_id: str,
        alpha_coordinate_id: str,
        look_ordinal: int,
    ) -> None:
        if self._design.exact_census_windowed_source:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "windowed census custody cannot consume a context-by-route row bank",
            )
        if self._design.sampled_windowed_source:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "windowed sampled custody cannot consume a context-by-route row bank",
            )
        if self._finalized:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.FIXED_LOOK_REUSE, "fixed look was already finalized")
        self._validate_call_identities({
            "model_identity_hash": model_identity_hash,
            "chart_identity_hash": chart_identity_hash,
            "current_mask_hash": current_mask_hash,
            "proposal_generation": proposal_generation,
            "witness_stream_source_hash": witness_stream_source_hash,
            "same_capture_vjp_source_hash": same_capture_vjp_source_hash,
            "corrected_evidence_stream_source_hash": corrected_evidence_stream_source_hash,
            "literal_j_stream_source_hash": literal_j_stream_source_hash,
            "alpha_ledger_id": alpha_ledger_id,
            "alpha_coordinate_id": alpha_coordinate_id,
            "look_ordinal": look_ordinal,
        })
        stratum_id = _identity(stratum_id, name="stratum ID")
        candidates = [index for index, item in enumerate(self._registration.strata) if item.stratum_id == stratum_id]
        if len(candidates) != 1:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.STALE_IDENTITY, "batch names an unregistered stratum")
        stratum_index = candidates[0]
        stratum = self._registration.strata[stratum_index]
        axes = tuple(_identity_axis(value, name=name) for value, name in (
            (context_ids, "batch context IDs"), (draw_ids, "batch draw IDs"),
            (measurement_ids, "batch measurement IDs"), (capture_ids, "batch capture IDs"),
        ))
        batch_sizes = {len(axis) for axis in axes}
        if len(batch_sizes) != 1:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.ORDER_MISMATCH, "same-capture batch axes have different lengths")
        batch = next(iter(batch_sizes))
        if batch > self._design.maximum_batch_size:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.HBM_FORECAST_EXCEEDED, "batch exceeds the forecasted row-batch envelope")
        start = self._observed[stratum_index]
        stop = start + batch
        expected_axes = (
            stratum.registered_context_ids[start:stop], stratum.registered_draw_ids[start:stop],
            stratum.registered_measurement_ids[start:stop], stratum.registered_capture_ids[start:stop],
        )
        if axes != expected_axes or stop > stratum.registered_count:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.ORDER_MISMATCH, "batch is not the next exact registered predictable prefix slice")
        seen_sets = (self._seen_context, self._seen_draw, self._seen_measurement, self._seen_capture)
        if any(any(item in seen for item in axis) for axis, seen in zip(axes, seen_sets, strict=True)):
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.DUPLICATE_IDENTITY, "batch reuses a context/draw/measurement/capture identity")
        shape = (batch, self._axis.route_count)
        a_rows = _require_fp64_tensor(
            corrected_evidence_rows,
            name="corrected-evidence rows",
            ndim=2,
            shape=shape,
            device=self._device,
            scan_finite=False,
        )
        j_rows = _require_fp64_tensor(
            literal_j_rows,
            name="literal-J rows",
            ndim=2,
            shape=shape,
            device=self._device,
            scan_finite=False,
        )
        invalid_batch = torch.zeros((), dtype=torch.bool, device=self._device)
        for route_start in range(0, self._axis.route_count, self._design.route_tile_size):
            route_stop = min(
                self._axis.route_count,
                route_start + self._design.route_tile_size,
            )
            a_tile = a_rows[:, route_start:route_stop]
            j_tile = j_rows[:, route_start:route_stop]
            invalid_batch.logical_or_(
                (~torch.isfinite(a_tile)).any()
                | (~torch.isfinite(j_tile)).any()
                | (
                    torch.abs(a_tile)
                    > self._a_support[stratum_index, route_start:route_stop]
                ).any()
                | (
                    torch.abs(j_tile)
                    > self._j_support[stratum_index, route_start:route_stop]
                ).any()
            )
        self._scalar_d2h += 1
        if bool(invalid_batch.item()):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.SUPPORT_BOUND_VIOLATION,
                "same-capture row is nonfinite or exceeds a pre-outcome device support bound",
            )
        self._consume_batch(stratum_index, a_rows, j_rows)
        for axis, seen in zip(axes, seen_sets, strict=True):
            seen.update(axis)
        self._observed[stratum_index] = stop
        self._consumed_batches += 1

    def _validate_next_census_batch(
        self,
        *,
        stratum_id: str,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        measurement_ids: Sequence[str],
        capture_ids: Sequence[str],
    ) -> tuple[int, tuple[tuple[str, ...], ...], int, int]:
        stratum_id = _identity(stratum_id, name="stratum ID")
        candidates = [
            index
            for index, item in enumerate(self._registration.strata)
            if item.stratum_id == stratum_id
        ]
        if len(candidates) != 1:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.STALE_IDENTITY,
                "batch names an unregistered stratum",
            )
        stratum_index = candidates[0]
        stratum = self._registration.strata[stratum_index]
        axes = tuple(
            _identity_axis(value, name=name)
            for value, name in (
                (context_ids, "batch context IDs"),
                (draw_ids, "batch draw IDs"),
                (measurement_ids, "batch measurement IDs"),
                (capture_ids, "batch capture IDs"),
            )
        )
        batch_sizes = {len(axis) for axis in axes}
        if len(batch_sizes) != 1:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.ORDER_MISMATCH,
                "same-capture batch axes have different lengths",
            )
        batch = next(iter(batch_sizes))
        if batch > self._design.maximum_batch_size:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.HBM_FORECAST_EXCEEDED,
                "batch exceeds the forecasted row-batch envelope",
            )
        start = self._observed[stratum_index]
        stop = start + batch
        expected_axes = (
            stratum.registered_context_ids[start:stop],
            stratum.registered_draw_ids[start:stop],
            stratum.registered_measurement_ids[start:stop],
            stratum.registered_capture_ids[start:stop],
        )
        if axes != expected_axes or stop > stratum.registered_count:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.ORDER_MISMATCH,
                "batch is not the next exact registered predictable prefix slice",
            )
        seen_sets = (
            self._seen_context,
            self._seen_draw,
            self._seen_measurement,
            self._seen_capture,
        )
        if any(
            any(item in seen for item in axis)
            for axis, seen in zip(axes, seen_sets, strict=True)
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.DUPLICATE_IDENTITY,
                "batch reuses a context/draw/measurement/capture identity",
            )
        return stratum_index, axes, batch, stop

    def begin_census_same_capture_batch(
        self,
        *,
        stratum_id: str,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        measurement_ids: Sequence[str],
        capture_ids: Sequence[str],
        model_identity_hash: str,
        chart_identity_hash: str,
        current_mask_hash: str,
        proposal_generation: int,
        witness_stream_source_hash: str,
        same_capture_vjp_source_hash: str,
        corrected_evidence_stream_source_hash: str,
        literal_j_stream_source_hash: str,
        alpha_ledger_id: str,
        alpha_coordinate_id: str,
        look_ordinal: int,
    ) -> None:
        """Open one bounded two-pass census batch without retaining route rows."""

        if not self._design.exact_census_windowed_source:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "windowed census entry point requires exact census source custody",
            )
        if self._finalized or self._census_active is not None:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.FIXED_LOOK_REUSE,
                "census batch custody is finalized or already active",
            )
        self._validate_call_identities(
            {
                "model_identity_hash": model_identity_hash,
                "chart_identity_hash": chart_identity_hash,
                "current_mask_hash": current_mask_hash,
                "proposal_generation": proposal_generation,
                "witness_stream_source_hash": witness_stream_source_hash,
                "same_capture_vjp_source_hash": same_capture_vjp_source_hash,
                "corrected_evidence_stream_source_hash": corrected_evidence_stream_source_hash,
                "literal_j_stream_source_hash": literal_j_stream_source_hash,
                "alpha_ledger_id": alpha_ledger_id,
                "alpha_coordinate_id": alpha_coordinate_id,
                "look_ordinal": look_ordinal,
            }
        )
        stratum_index, axes, batch, stop = self._validate_next_census_batch(
            stratum_id=stratum_id,
            context_ids=context_ids,
            draw_ids=draw_ids,
            measurement_ids=measurement_ids,
            capture_ids=capture_ids,
        )
        self._census_active = {
            "source_mode": "exact_census",
            "stratum_index": stratum_index,
            "axes": axes,
            "batch": batch,
            "stop": stop,
            "phase": "pass1",
            "pass1_cursor": 0,
            "pass2_cursor": 0,
            "projection": torch.zeros(batch, dtype=torch.float64, device=self._device),
            "cell_sums": [
                torch.zeros(
                    (batch, cut.fiber_count),
                    dtype=torch.float64,
                    device=self._device,
                )
                for cut in self._axis.cuts
            ],
            "cell_abs_sums": [
                torch.zeros(
                    (batch, cut.fiber_count),
                    dtype=torch.float64,
                    device=self._device,
                )
                for cut in self._axis.cuts
            ],
            "signs": None,
            "ambiguous": None,
        }

    def _require_census_window(
        self,
        *,
        route_start: int,
        corrected_evidence_rows: Any,
        literal_j_rows: Any,
        phase: str,
        source_mode: str = "exact_census",
    ) -> tuple[dict[str, Any], Any, Any, int]:
        active = self._census_active
        if (
            active is None
            or active["phase"] != phase
            or active.get("source_mode") != source_mode
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.ORDER_MISMATCH,
                f"windowed source is not accepting {phase} route windows",
            )
        route_start = _nonnegative_int(route_start, name="route-window start")
        expected = int(active[f"{phase}_cursor"])
        if route_start != expected:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.ORDER_MISMATCH,
                "route window is not the next exact canonical-axis slice",
            )
        batch = int(active["batch"])
        if not isinstance(corrected_evidence_rows, torch.Tensor) or corrected_evidence_rows.ndim != 2:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_TYPE,
                "corrected-evidence route window must be one FP64 tensor",
            )
        width = int(corrected_evidence_rows.shape[1])
        if (
            width <= 0
            or width > self._design.route_tile_size
            or route_start + width > self._axis.route_count
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.HBM_FORECAST_EXCEEDED,
                "route window exceeds the registered bounded tile envelope",
            )
        shape = (batch, width)
        a_rows = _require_fp64_tensor(
            corrected_evidence_rows,
            name="corrected-evidence route window",
            ndim=2,
            shape=shape,
            device=self._device,
            scan_finite=False,
        )
        j_rows = _require_fp64_tensor(
            literal_j_rows,
            name="literal-J route window",
            ndim=2,
            shape=shape,
            device=self._device,
            scan_finite=False,
        )
        if not a_rows.is_contiguous() or not j_rows.is_contiguous():
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "census route windows must be contiguous source-owned tensors",
            )
        invalid = (~torch.isfinite(a_rows)).any() | (~torch.isfinite(j_rows)).any()
        self._scalar_d2h += 1
        if bool(invalid.item()):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.SUPPORT_BOUND_VIOLATION,
                "census source window contains a nonfinite value",
            )
        return active, a_rows, j_rows, width

    def _accumulate_census_fiber_cells(
        self,
        *,
        active: dict[str, Any],
        corrected_evidence_rows: Any,
        route_start: int,
        route_stop: int,
    ) -> None:
        """Accumulate arbitrary bounded windows without CUDA atomics or labels."""

        z = self._z
        for cut in self._axis.cuts:
            overlap_start = max(route_start, cut.route_offset)
            overlap_stop = min(route_stop, cut.route_stop)
            if overlap_start >= overlap_stop:
                continue
            local_start = overlap_start - cut.route_offset
            local_stop = overlap_stop - cut.route_offset
            source_offset = overlap_start - route_start
            rpf = cut.routes_per_fiber
            cursor = local_start

            def add_segment(segment_stop: int, fiber_start: int, fiber_count: int) -> None:
                nonlocal cursor, source_offset
                width = segment_stop - cursor
                values = corrected_evidence_rows[:, source_offset : source_offset + width]
                local_z = z[
                    cut.route_offset + cursor : cut.route_offset + segment_stop
                ].reshape(1, -1)
                products = values * local_z
                if fiber_count == 1 and width != rpf:
                    sums = torch.sum(products, dim=1, keepdim=True)
                    abs_sums = torch.sum(torch.abs(products), dim=1, keepdim=True)
                else:
                    sums = torch.sum(
                        products.reshape(int(active["batch"]), fiber_count, rpf),
                        dim=2,
                    )
                    abs_sums = torch.sum(
                        torch.abs(products).reshape(
                            int(active["batch"]), fiber_count, rpf
                        ),
                        dim=2,
                    )
                active["cell_sums"][cut.cut_index][
                    :, fiber_start : fiber_start + fiber_count
                ].add_(sums)
                active["cell_abs_sums"][cut.cut_index][
                    :, fiber_start : fiber_start + fiber_count
                ].add_(abs_sums)
                source_offset += width
                cursor = segment_stop

            offset = cursor % rpf
            if offset:
                segment_stop = min(local_stop, cursor + (rpf - offset))
                add_segment(segment_stop, cursor // rpf, 1)
            complete = (local_stop - cursor) // rpf
            if complete:
                segment_stop = cursor + complete * rpf
                add_segment(segment_stop, cursor // rpf, complete)
            if cursor < local_stop:
                add_segment(local_stop, cursor // rpf, 1)

    def consume_census_route_window_pass1(
        self,
        *,
        route_start: int,
        corrected_evidence_rows: Any,
        literal_j_rows: Any,
    ) -> None:
        active, a_rows, j_rows, width = self._require_census_window(
            route_start=route_start,
            corrected_evidence_rows=corrected_evidence_rows,
            literal_j_rows=literal_j_rows,
            phase="pass1",
        )
        route_stop = route_start + width
        stratum_index = int(active["stratum_index"])
        self._a_support[stratum_index, route_start:route_stop] = torch.maximum(
            self._a_support[stratum_index, route_start:route_stop],
            torch.amax(torch.abs(a_rows), dim=0),
        )
        self._j_support[stratum_index, route_start:route_stop] = torch.maximum(
            self._j_support[stratum_index, route_start:route_stop],
            torch.amax(torch.abs(j_rows), dim=0),
        )
        active["projection"].add_(
            torch.sum(
                j_rows * self._z[route_start:route_stop].reshape(1, -1),
                dim=1,
            )
        )
        self._accumulate_census_fiber_cells(
            active=active,
            corrected_evidence_rows=a_rows,
            route_start=route_start,
            route_stop=route_stop,
        )
        active["pass1_cursor"] = route_stop

    def finish_census_route_window_pass1(
        self,
        *,
        streaming_replay_digest: str,
    ) -> None:
        active = self._census_active
        if (
            active is None
            or active["phase"] != "pass1"
            or active["pass1_cursor"] != self._axis.route_count
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.PREFIX_INCOMPLETE,
                "census pass one did not cover the complete canonical route axis",
            )
        signs: list[Any] = []
        ambiguous_family: list[Any] = []
        for cut in self._axis.cuts:
            cells = active["cell_sums"][cut.cut_index]
            abs_sums = active["cell_abs_sums"][cut.cut_index]
            cell_gamma = _gamma(2 * cut.routes_per_fiber + 4)
            cell_error = _up_tensor(
                abs_sums * math.nextafter(2.0 * cell_gamma, math.inf)
            )
            ambiguous = torch.abs(cells) <= cell_error
            signs.append(torch.sign(cells))
            ambiguous_family.append(ambiguous)
            self._exact_zero_by_cut[cut.cut_index].add_(
                torch.sum(cells == 0.0, dtype=torch.int64)
            )
            self._ambiguous_by_cut[cut.cut_index].add_(
                torch.sum(ambiguous, dtype=torch.int64)
            )
        active["signs"] = signs
        active["ambiguous"] = ambiguous_family
        active["pass1_replay_digest"] = _sha256_hex(
            streaming_replay_digest,
            name="census pass-one replay digest",
        )
        active["phase"] = "pass2"

    def consume_census_route_window_pass2(
        self,
        *,
        route_start: int,
        corrected_evidence_rows: Any,
        literal_j_rows: Any,
    ) -> None:
        active, a_rows, j_rows, width = self._require_census_window(
            route_start=route_start,
            corrected_evidence_rows=corrected_evidence_rows,
            literal_j_rows=literal_j_rows,
            phase="pass2",
        )
        route_stop = route_start + width
        stratum_index = int(active["stratum_index"])
        invalid_support = (
            torch.abs(a_rows)
            > self._a_support[stratum_index, route_start:route_stop]
        ).any() | (
            torch.abs(j_rows)
            > self._j_support[stratum_index, route_start:route_stop]
        ).any()
        self._scalar_d2h += 1
        if bool(invalid_support.item()):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.MUTATED_CUSTODY,
                "census pass-two rows differ from the sealed pass-one support envelope",
            )
        cut_count = float(self._axis.cut_count)
        for cut in self._axis.cuts:
            overlap_start = max(route_start, cut.route_offset)
            overlap_stop = min(route_stop, cut.route_stop)
            if overlap_start >= overlap_stop:
                continue
            local = torch.arange(
                overlap_start - cut.route_offset,
                overlap_stop - cut.route_offset,
                dtype=torch.int64,
                device=self._device,
            )
            fiber_indices = torch.div(
                local, cut.routes_per_fiber, rounding_mode="floor"
            )
            source_start = overlap_start - route_start
            source_stop = overlap_stop - route_start
            values = a_rows[:, source_start:source_stop]
            signs = active["signs"][cut.cut_index][:, fiber_indices]
            ambiguous = active["ambiguous"][cut.cut_index][:, fiber_indices]
            self._f_sums[stratum_index][overlap_start:overlap_stop].add_(
                torch.sum(values * signs, dim=0).div_(cut_count)
            )
            self._f_ambiguity_sums[stratum_index][
                overlap_start:overlap_stop
            ].add_(
                torch.where(
                    ambiguous,
                    _up_tensor(2.0 * torch.abs(values) / cut_count),
                    torch.zeros((), dtype=torch.float64, device=self._device),
                ).sum(dim=0)
            )
            self._zero_cell_ambiguity_bitmap[
                overlap_start:overlap_stop
            ].logical_or_(torch.any(ambiguous, dim=0))
        self._q_sums[stratum_index][route_start:route_stop].add_(
            torch.sum(
                2.0 * j_rows * active["projection"].reshape(-1, 1),
                dim=0,
            )
        )
        active["pass2_cursor"] = route_stop

    def finish_census_same_capture_batch(
        self,
        *,
        streaming_replay_digest: str,
    ) -> None:
        active = self._census_active
        if (
            active is None
            or active["phase"] != "pass2"
            or active["pass2_cursor"] != self._axis.route_count
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.PREFIX_INCOMPLETE,
                "census pass two did not cover the complete canonical route axis",
            )
        pass2_digest = _sha256_hex(
            streaming_replay_digest,
            name="census pass-two replay digest",
        )
        if pass2_digest != active.get("pass1_replay_digest"):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.MUTATED_CUSTODY,
                "census pass-one/pass-two streaming value digests differ",
            )
        axes = active["axes"]
        for axis, seen in zip(
            axes,
            (
                self._seen_context,
                self._seen_draw,
                self._seen_measurement,
                self._seen_capture,
            ),
            strict=True,
        ):
            seen.update(axis)
        self._observed[int(active["stratum_index"])] = int(active["stop"])
        self._consumed_batches += 1
        self._census_completed_contexts += int(active["batch"])
        self._census_replay_digests.append(pass2_digest)
        self._census_active = None

    def begin_sampled_same_capture_batch(
        self,
        *,
        stratum_id: str,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        measurement_ids: Sequence[str],
        capture_ids: Sequence[str],
        model_identity_hash: str,
        chart_identity_hash: str,
        current_mask_hash: str,
        proposal_generation: int,
        witness_stream_source_hash: str,
        same_capture_vjp_source_hash: str,
        corrected_evidence_stream_source_hash: str,
        literal_j_stream_source_hash: str,
        alpha_ledger_id: str,
        alpha_coordinate_id: str,
        look_ordinal: int,
    ) -> None:
        """Open one bounded sampled batch without retaining route-scale rows."""

        if not self._design.sampled_windowed_source:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "windowed sampled entry point requires sampled source custody",
            )
        if self._design.exact_census_windowed_source or self._registration.exact_census_mode:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "sampled windowed custody cannot impersonate exact-census mode",
            )
        if self._finalized or self._census_active is not None:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.FIXED_LOOK_REUSE,
                "sampled batch custody is finalized or already active",
            )
        self._validate_call_identities(
            {
                "model_identity_hash": model_identity_hash,
                "chart_identity_hash": chart_identity_hash,
                "current_mask_hash": current_mask_hash,
                "proposal_generation": proposal_generation,
                "witness_stream_source_hash": witness_stream_source_hash,
                "same_capture_vjp_source_hash": same_capture_vjp_source_hash,
                "corrected_evidence_stream_source_hash": corrected_evidence_stream_source_hash,
                "literal_j_stream_source_hash": literal_j_stream_source_hash,
                "alpha_ledger_id": alpha_ledger_id,
                "alpha_coordinate_id": alpha_coordinate_id,
                "look_ordinal": look_ordinal,
            }
        )
        stratum_index, axes, batch, stop = self._validate_next_census_batch(
            stratum_id=stratum_id,
            context_ids=context_ids,
            draw_ids=draw_ids,
            measurement_ids=measurement_ids,
            capture_ids=capture_ids,
        )
        self._census_active = {
            "source_mode": "sampled",
            "stratum_index": stratum_index,
            "axes": axes,
            "batch": batch,
            "stop": stop,
            "phase": "pass1",
            "pass1_cursor": 0,
            "pass2_cursor": 0,
            "projection": torch.zeros(batch, dtype=torch.float64, device=self._device),
            "cell_sums": [
                torch.zeros(
                    (batch, cut.fiber_count),
                    dtype=torch.float64,
                    device=self._device,
                )
                for cut in self._axis.cuts
            ],
            "cell_abs_sums": [
                torch.zeros(
                    (batch, cut.fiber_count),
                    dtype=torch.float64,
                    device=self._device,
                )
                for cut in self._axis.cuts
            ],
            "signs": None,
            "ambiguous": None,
        }

    def consume_sampled_route_window_pass1(
        self,
        *,
        route_start: int,
        corrected_evidence_rows: Any,
        literal_j_rows: Any,
    ) -> None:
        active, a_rows, j_rows, width = self._require_census_window(
            route_start=route_start,
            corrected_evidence_rows=corrected_evidence_rows,
            literal_j_rows=literal_j_rows,
            phase="pass1",
            source_mode="sampled",
        )
        route_stop = route_start + width
        stratum_index = int(active["stratum_index"])
        invalid_support = (
            torch.abs(a_rows)
            > self._a_support[stratum_index, route_start:route_stop]
        ).any() | (
            torch.abs(j_rows)
            > self._j_support[stratum_index, route_start:route_stop]
        ).any()
        self._scalar_d2h += 1
        if bool(invalid_support.item()):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.SUPPORT_BOUND_VIOLATION,
                "sampled route window exceeds a pre-outcome support bound",
            )
        active["projection"].add_(
            torch.sum(
                j_rows * self._z[route_start:route_stop].reshape(1, -1),
                dim=1,
            )
        )
        self._accumulate_census_fiber_cells(
            active=active,
            corrected_evidence_rows=a_rows,
            route_start=route_start,
            route_stop=route_stop,
        )
        active["pass1_cursor"] = route_stop

    def finish_sampled_route_window_pass1(
        self,
        *,
        streaming_replay_digest: str,
    ) -> None:
        active = self._census_active
        if (
            active is None
            or active.get("source_mode") != "sampled"
            or active["phase"] != "pass1"
            or active["pass1_cursor"] != self._axis.route_count
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.PREFIX_INCOMPLETE,
                "sampled pass one did not cover the complete canonical route axis",
            )
        signs: list[Any] = []
        ambiguous_family: list[Any] = []
        for cut in self._axis.cuts:
            cells = active["cell_sums"][cut.cut_index]
            abs_sums = active["cell_abs_sums"][cut.cut_index]
            cell_gamma = _gamma(2 * cut.routes_per_fiber + 4)
            cell_error = _up_tensor(
                abs_sums * math.nextafter(2.0 * cell_gamma, math.inf)
            )
            ambiguous = torch.abs(cells) <= cell_error
            signs.append(torch.sign(cells))
            ambiguous_family.append(ambiguous)
            self._exact_zero_by_cut[cut.cut_index].add_(
                torch.sum(cells == 0.0, dtype=torch.int64)
            )
            self._ambiguous_by_cut[cut.cut_index].add_(
                torch.sum(ambiguous, dtype=torch.int64)
            )
        active["signs"] = signs
        active["ambiguous"] = ambiguous_family
        active["pass1_replay_digest"] = _sha256_hex(
            streaming_replay_digest,
            name="sampled pass-one replay digest",
        )
        active["phase"] = "pass2"

    def consume_sampled_route_window_pass2(
        self,
        *,
        route_start: int,
        corrected_evidence_rows: Any,
        literal_j_rows: Any,
    ) -> None:
        active, a_rows, j_rows, width = self._require_census_window(
            route_start=route_start,
            corrected_evidence_rows=corrected_evidence_rows,
            literal_j_rows=literal_j_rows,
            phase="pass2",
            source_mode="sampled",
        )
        route_stop = route_start + width
        stratum_index = int(active["stratum_index"])
        invalid_support = (
            torch.abs(a_rows)
            > self._a_support[stratum_index, route_start:route_stop]
        ).any() | (
            torch.abs(j_rows)
            > self._j_support[stratum_index, route_start:route_stop]
        ).any()
        self._scalar_d2h += 1
        if bool(invalid_support.item()):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.SUPPORT_BOUND_VIOLATION,
                "sampled pass-two route window exceeds a pre-outcome support bound",
            )
        cut_count = float(self._axis.cut_count)
        for cut in self._axis.cuts:
            overlap_start = max(route_start, cut.route_offset)
            overlap_stop = min(route_stop, cut.route_stop)
            if overlap_start >= overlap_stop:
                continue
            local = torch.arange(
                overlap_start - cut.route_offset,
                overlap_stop - cut.route_offset,
                dtype=torch.int64,
                device=self._device,
            )
            fiber_indices = torch.div(
                local, cut.routes_per_fiber, rounding_mode="floor"
            )
            source_start = overlap_start - route_start
            source_stop = overlap_stop - route_start
            values = a_rows[:, source_start:source_stop]
            signs = active["signs"][cut.cut_index][:, fiber_indices]
            ambiguous = active["ambiguous"][cut.cut_index][:, fiber_indices]
            self._f_sums[stratum_index][overlap_start:overlap_stop].add_(
                torch.sum(values * signs, dim=0).div_(cut_count)
            )
            self._f_ambiguity_sums[stratum_index][
                overlap_start:overlap_stop
            ].add_(
                torch.where(
                    ambiguous,
                    _up_tensor(2.0 * torch.abs(values) / cut_count),
                    torch.zeros((), dtype=torch.float64, device=self._device),
                ).sum(dim=0)
            )
            self._zero_cell_ambiguity_bitmap[
                overlap_start:overlap_stop
            ].logical_or_(torch.any(ambiguous, dim=0))
        self._q_sums[stratum_index][route_start:route_stop].add_(
            torch.sum(
                2.0 * j_rows * active["projection"].reshape(-1, 1),
                dim=0,
            )
        )
        active["pass2_cursor"] = route_stop

    def finish_sampled_same_capture_batch(
        self,
        *,
        streaming_replay_digest: str,
    ) -> None:
        active = self._census_active
        if (
            active is None
            or active.get("source_mode") != "sampled"
            or active["phase"] != "pass2"
            or active["pass2_cursor"] != self._axis.route_count
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.PREFIX_INCOMPLETE,
                "sampled pass two did not cover the complete canonical route axis",
            )
        pass2_digest = _sha256_hex(
            streaming_replay_digest,
            name="sampled pass-two replay digest",
        )
        if pass2_digest != active.get("pass1_replay_digest"):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.MUTATED_CUSTODY,
                "sampled pass-one/pass-two streaming value digests differ",
            )
        axes = active["axes"]
        for axis, seen in zip(
            axes,
            (
                self._seen_context,
                self._seen_draw,
                self._seen_measurement,
                self._seen_capture,
            ),
            strict=True,
        ):
            seen.update(axis)
        self._observed[int(active["stratum_index"])] = int(active["stop"])
        self._consumed_batches += 1
        self._sampled_replay_digests.append(pass2_digest)
        self._census_active = None

    def _consume_batch(self, stratum_index: int, a_rows: Any, j_rows: Any) -> None:
        """Fuse every same-capture row across each native-fiber tile.

        Exact/ambiguous cell counts remain as compact device vectors until
        finalization.  There is no scalar host synchronization in this loop.
        """

        z = self._z
        route_tile = self._design.route_tile_size
        cut_count = self._axis.cut_count
        batch = int(a_rows.shape[0])
        for cut in self._axis.cuts:
            fibers_per_tile = max(1, route_tile // cut.routes_per_fiber)
            for local_fiber_start in range(0, cut.fiber_count, fibers_per_tile):
                local_fiber_stop = min(cut.fiber_count, local_fiber_start + fibers_per_tile)
                start = cut.route_offset + local_fiber_start * cut.routes_per_fiber
                stop = cut.route_offset + local_fiber_stop * cut.routes_per_fiber
                fiber_count = local_fiber_stop - local_fiber_start
                a = a_rows[:, start:stop].reshape(
                    batch, fiber_count, cut.routes_per_fiber
                )
                local_z = z[start:stop].reshape(
                    1, fiber_count, cut.routes_per_fiber
                )
                products = a * local_z
                cells = torch.sum(products, dim=2)
                abs_sum = torch.sum(torch.abs(products), dim=2)
                cell_gamma = _gamma(2 * cut.routes_per_fiber + 4)
                cell_error = _up_tensor(
                    abs_sum * math.nextafter(2.0 * cell_gamma, math.inf)
                )
                ambiguous = torch.abs(cells) <= cell_error
                exact_zero = cells == 0.0
                signs = torch.sign(cells)
                contribution = torch.sum(
                    a * signs.unsqueeze(2), dim=0
                ).div_(float(cut_count)).reshape(-1)
                self._f_sums[stratum_index][start:stop].add_(contribution)
                ambiguity_bound = _up_tensor(
                    2.0 * torch.abs(a) / float(cut_count)
                )
                ambiguity = torch.where(
                    ambiguous.unsqueeze(2),
                    ambiguity_bound,
                    torch.zeros((), dtype=torch.float64, device=self._device),
                ).sum(dim=0).reshape(-1)
                self._f_ambiguity_sums[stratum_index][start:stop].add_(ambiguity)
                routewise_ambiguous = (
                    torch.any(ambiguous, dim=0)
                    .unsqueeze(1)
                    .expand(-1, cut.routes_per_fiber)
                    .reshape(-1)
                )
                self._zero_cell_ambiguity_bitmap[start:stop].logical_or_(
                    routewise_ambiguous
                )
                self._exact_zero_by_cut[cut.cut_index].add_(
                    torch.sum(exact_zero, dtype=torch.int64)
                )
                self._ambiguous_by_cut[cut.cut_index].add_(
                    torch.sum(ambiguous, dtype=torch.int64)
                )

        projection = torch.zeros(
            batch, dtype=torch.float64, device=self._device
        )
        for start in range(0, self._axis.route_count, route_tile):
            stop = min(self._axis.route_count, start + route_tile)
            partial = torch.sum(
                j_rows[:, start:stop] * z[start:stop].reshape(1, -1),
                dim=1,
            )
            projection.add_(partial)
        for start in range(0, self._axis.route_count, route_tile):
            stop = min(self._axis.route_count, start + route_tile)
            self._q_sums[stratum_index][start:stop].add_(
                torch.sum(
                    2.0
                    * j_rows[:, start:stop]
                    * projection.reshape(-1, 1),
                    dim=0,
                )
            )

    def _positive_dot_up(self, left: Any, right: Any) -> Any:
        result = torch.zeros((), dtype=torch.float64, device=self._device)
        for start in range(0, self._axis.route_count, self._design.route_tile_size):
            stop = min(self._axis.route_count, start + self._design.route_tile_size)
            width = stop - start
            partial_raw = torch.sum(left[start:stop] * right[start:stop])
            partial = _up_tensor(
                partial_raw
                * math.nextafter(
                    1.0 + 2.0 * _gamma(2 * width + 2),
                    math.inf,
                )
            )
            result = _up_add(result, partial)
        return result

    def _sampling_radius(
        self,
        bounds: tuple[Any, Any],
        per_statement_two_sided_alpha: float | None,
    ) -> Any:
        if all(
            self._observed[index] >= stratum.population_count
            for index, stratum in enumerate(self._registration.strata)
        ):
            return torch.zeros(
                self._axis.route_count,
                dtype=torch.float64,
                device=self._device,
            )
        if per_statement_two_sided_alpha is None:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.INVALID_DESIGN,
                "a partial finite population cannot use an alpha-free sampling radius",
            )
        variance_proxy = torch.zeros(self._axis.route_count, dtype=torch.float64, device=self._device)
        for index, (bound, stratum) in enumerate(zip(bounds, self._registration.strata, strict=True)):
            observed = self._observed[index]
            population = stratum.population_count
            if observed >= population:
                continue
            # Classical Hoeffding--Serfling correction.  Census is handled
            # exactly above; a partial prefix uses 1-(n-1)/N.
            coefficient = _serfling_weight_coefficient_upper(
                stratum_weight=stratum.stratum_weight,
                observed=observed,
                population=population,
            )
            term = _up_tensor(bound * coefficient)
            variance_proxy = _up_add(variance_proxy, _up_tensor(term * term))
        log_factor = _serfling_log_factor_upper(per_statement_two_sided_alpha)
        return _up_tensor(torch.sqrt(_up_tensor(variance_proxy * log_factor)))

    def _weighted_mean_into(self, sums: list[Any]) -> Any:
        if self._registration.exact_census_mode:
            total = sum(self._observed)
            # D1 is one uniform empirical fitting measure.  Aggregate its
            # integer row sums first and perform one final division; do not
            # round two rational stratum weights independently.
            result = sums[0]
            result.add_(sums[1])
            result.div_(float(total))
            return result
        result = sums[0].mul(self._registration.strata[0].stratum_weight / self._observed[0])
        result.add_(sums[1], alpha=self._registration.strata[1].stratum_weight / self._observed[1])
        return result

    def _stratum_weight_upper(self, index: int) -> float:
        if not self._registration.exact_census_mode:
            return self._registration.strata[index].stratum_weight
        fractions = self._registration.exact_census_stratum_weight_fractions
        assert fractions is not None
        numerator, denominator = fractions[index]
        with localcontext() as context:
            context.prec = 100
            context.rounding = ROUND_CEILING
            value = (
                Decimal(numerator) / Decimal(denominator)
            ).next_plus()
        return _decimal_up_float(value)

    def finalize_fixed_look(self) -> V1SignedWorkFixedPrimalProjectionCUDAExecution:
        if self._finalized:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.FIXED_LOOK_REUSE, "fixed look cannot be finalized or recycled twice")
        if self._census_active is not None:
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.PREFIX_INCOMPLETE,
                "an open census replay batch cannot be finalized",
            )
        if tuple(self._observed) != self._registration.registered_counts:
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.PREFIX_INCOMPLETE, "registered predictable witness prefix is incomplete")
        if (
            self._design.exact_census_windowed_source
            and self._census_completed_contexts != sum(self._registration.registered_counts)
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.PREFIX_INCOMPLETE,
                "the exact census was not completed through the bounded two-pass source",
            )
        if (
            self._design.exact_census_windowed_source
            and len(self._census_replay_digests) != self._consumed_batches
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.PREFIX_INCOMPLETE,
                "the exact census lacks one value-identity digest per replay batch",
            )
        if (
            self._design.sampled_windowed_source
            and len(self._sampled_replay_digests) != self._consumed_batches
        ):
            _refuse(
                V1SignedWorkFixedPrimalProjectionCUDARefusalCode.PREFIX_INCOMPLETE,
                "the sampled source lacks one value-identity digest per replay batch",
            )
        self._finalized = True
        route_count = self._axis.route_count
        statements = 2 * route_count
        if self._registration.exact_census_mode:
            per_two = None
            per_one = None
        else:
            assert self._registration.familywise_alpha is not None
            per_two = self._registration.familywise_alpha / statements
            per_one = per_two / 2.0

        fixed_primal_fingerprint = _device_vector_fingerprint(self._z, role="fixed_primal", tile_size=self._design.route_tile_size)
        corrected_support_fingerprint = content_hash(
            tuple(_device_vector_fingerprint(self._a_support[index], role=f"corrected_support_stratum_{index}", tile_size=self._design.route_tile_size) for index in range(2)),
            prefix=_PREFIX + "corrected-support-family-",
        )
        literal_j_support_fingerprint = content_hash(
            tuple(_device_vector_fingerprint(self._j_support[index], role=f"literal_j_support_stratum_{index}", tile_size=self._design.route_tile_size) for index in range(2)),
            prefix=_PREFIX + "literal-j-support-family-",
        )
        fixed_primal_descriptor_hash = content_hash(
            {
                "schema": "signed_work_fixed_primal_descriptor_v1",
                "fixed_primal_id": self._fixed_primal_id,
                "fixed_primal_source_receipt_hash": self._fixed_primal_source_receipt_hash,
                "device_fingerprint": fixed_primal_fingerprint,
                "chart_identity_hash": self._registration.chart_identity_hash,
                "current_mask_hash": self._registration.current_mask_hash,
                "route_axis_hash": self._registration.route_axis_hash,
                "proposal_generation": self._registration.proposal_generation,
            },
            prefix=_PREFIX + "fixed-primal-descriptor-",
        )

        f_bounds = tuple(_up_tensor(self._a_support[index] / float(self._axis.cut_count)) for index in range(2))
        projection_bounds = tuple(self._positive_dot_up(self._j_support[index], torch.abs(self._z)) for index in range(2))
        q_bounds = tuple(
            _up_mul(
                _up_mul(
                    self._j_support[index],
                    torch.full(
                        (),
                        2.0,
                        dtype=torch.float64,
                        device=self._device,
                    ),
                ),
                projection_bounds[index],
            )
            for index in range(2)
        )

        f_sampling = self._sampling_radius(f_bounds, per_two)
        q_sampling = self._sampling_radius(q_bounds, per_two)
        f_sampling_zero = bool((f_sampling == 0.0).all().item())
        q_sampling_zero = bool((q_sampling == 0.0).all().item())
        self._scalar_d2h += 2

        f_center = self._weighted_mean_into(self._f_sums)
        q_center = self._weighted_mean_into(self._q_sums)
        f_numerical = torch.zeros(route_count, dtype=torch.float64, device=self._device)
        if self._registration.exact_census_mode:
            ambiguity_mean = self._f_ambiguity_sums[0]
            ambiguity_mean.add_(self._f_ambiguity_sums[1])
            ambiguity_mean.div_(float(sum(self._observed)))
            ambiguity_mean = _up_tensor(
                ambiguity_mean
                * math.nextafter(
                    1.0 + 2.0 * _gamma(sum(self._observed) + 4),
                    math.inf,
                )
            )
            f_numerical = _up_add(f_numerical, ambiguity_mean)
        else:
            for index, stratum in enumerate(self._registration.strata):
                ambiguity_mean = self._f_ambiguity_sums[index] * (
                    stratum.stratum_weight / self._observed[index]
                )
                ambiguity_mean = _up_tensor(
                    ambiguity_mean
                    * math.nextafter(
                        1.0 + 2.0 * _gamma(self._observed[index] + 4),
                        math.inf,
                    )
                )
                f_numerical = _up_add(f_numerical, ambiguity_mean)
        q_numerical = torch.zeros(route_count, dtype=torch.float64, device=self._device)
        maximum_routes_per_fiber = max(cut.routes_per_fiber for cut in self._axis.cuts)
        tile_count = math.ceil(route_count / self._design.route_tile_size)
        for index, stratum in enumerate(self._registration.strata):
            observed = self._observed[index]
            weight_upper = self._stratum_weight_upper(index)
            f_round = _gamma(observed + maximum_routes_per_fiber + self._axis.cut_count + 8)
            f_numerical = _up_add(f_numerical, _up_tensor(f_bounds[index] * (weight_upper * f_round)))
            projection_error = _up_tensor(projection_bounds[index] * _gamma(2 * route_count + tile_count + 8))
            q_projection_error = _up_tensor(2.0 * self._j_support[index] * projection_error)
            q_round = math.nextafter(_gamma(8) + _gamma(observed + 4), math.inf)
            q_arithmetic_error = _up_tensor(q_bounds[index] * q_round)
            q_numerical = _up_add(q_numerical, _up_tensor((q_projection_error + q_arithmetic_error) * weight_upper))
        f_total = _up_add(f_sampling, f_numerical)
        q_total = _up_add(q_sampling, q_numerical)
        for value, name in ((f_center, "signed-F center"), (q_center, "literal-J Q-gradient center"), (f_total, "signed-F radius"), (q_total, "literal-J Q-gradient radius")):
            if not bool(torch.isfinite(value).all().item()):
                _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.NUMERICAL_REFUSAL, f"{name} is nonfinite")
        if bool((f_total < 0.0).any().item()) or bool((q_total < 0.0).any().item()):
            _refuse(V1SignedWorkFixedPrimalProjectionCUDARefusalCode.NUMERICAL_REFUSAL, "outward radii must be nonnegative")
        self._scalar_d2h += 6

        roles_and_tensors = {
            "signed_f_subgradient_center": f_center,
            "signed_f_sampling_radius": f_sampling,
            "signed_f_numerical_radius": f_numerical,
            "signed_f_subgradient_radius": f_total,
            "literal_j_q_gradient_center": q_center,
            "literal_j_q_gradient_sampling_radius": q_sampling,
            "literal_j_q_gradient_numerical_radius": q_numerical,
            "literal_j_q_gradient_radius": q_total,
            "zero_cell_ambiguity_bitmap": self._zero_cell_ambiguity_bitmap,
        }
        fingerprints = {role: _device_vector_fingerprint(tensor, role=role, tile_size=self._design.route_tile_size) for role, tensor in roles_and_tensors.items()}
        # z + two A-support + two J-support + nine output fingerprints.
        self._scalar_d2h += 6 * (5 + len(roles_and_tensors))

        # These are compact cut-axis vectors (seven entries for the current
        # AlexNet axis), not route-scale results.  Copy each vector once at
        # finalization instead of synchronizing once per cut and context.
        exact_zero_counts_by_cut = tuple(
            int(value)
            for value in self._exact_zero_by_cut.detach().cpu().tolist()
        )
        ambiguous_counts_by_cut = tuple(
            int(value)
            for value in self._ambiguous_by_cut.detach().cpu().tolist()
        )
        self._scalar_d2h += 2
        evaluated_cells = int(sum(self._observed) * self._axis.fiber_count)
        zero_receipt = V1SignedWorkFixedPrimalZeroCellAmbiguityReceipt(
            registration_hash=self._registration_hash,
            native_fiber_axis_receipt_hash=self._axis.receipt_hash,
            evaluated_context_fiber_cells=evaluated_cells,
            exact_binary64_zero_cell_count=sum(exact_zero_counts_by_cut),
            roundoff_ambiguous_cell_count=sum(ambiguous_counts_by_cut),
            exact_zero_counts_by_cut=exact_zero_counts_by_cut,
            roundoff_ambiguous_counts_by_cut=ambiguous_counts_by_cut,
        )
        production = self._device.type == "cuda" and self._design.production_cuda_required
        telemetry = V1SignedWorkFixedPrimalProjectionCUDATelemetry(
            registration_hash=self._registration_hash,
            device_identity=str(self._device),
            production_cuda_execution=production,
            private_cpu_reference_execution=not production,
            consumed_batches=self._consumed_batches,
            consumed_contexts=sum(self._observed),
            route_count=route_count,
            cut_count=self._axis.cut_count,
            fiber_count=self._axis.fiber_count,
            route_tiles_per_context=math.ceil(route_count / self._design.route_tile_size),
            native_fiber_tiles_total=sum(
                math.ceil(cut.fiber_count / max(1, self._design.route_tile_size // cut.routes_per_fiber))
                for cut in self._axis.cuts
            ) * self._consumed_batches,
            # Handle issuance and execution-bundle closure each replay the
            # six retained fingerprints (six scalar sketches per vector).
            scalar_device_to_host_transfers=self._scalar_d2h + 2 * 6 * 6,
            route_scale_device_to_host_transfers=0,
            route_scale_fiber_labels_materialized=False,
            floating_atomics_used=False,
            deterministic_context_order=True,
            same_capture_rows_processed_in_batched_kernels=True,
            per_native_fiber_tile_scalar_device_to_host_transfers=0,
            compact_finalize_count_vector_device_to_host_transfers=2,
        )
        prefix_hashes = lambda role: tuple(_axis_hash(role, getattr(item, role)) for item in self._registration.strata)
        receipt = V1SignedWorkFixedPrimalProjectionReceipt(
            registration_hash=self._registration_hash,
            design_hash=self._design.design_hash,
            forecast_hash=self._forecast.forecast_hash,
            telemetry_hash=telemetry.telemetry_hash,
            model_identity_hash=self._registration.model_identity_hash,
            chart_identity_hash=self._registration.chart_identity_hash,
            current_mask_hash=self._registration.current_mask_hash,
            proposal_generation=self._registration.proposal_generation,
            route_axis_hash=self._registration.route_axis_hash,
            route_id_axis_hash=self._registration.route_id_axis_hash,
            native_fiber_axis_receipt_hash=self._axis.receipt_hash,
            native_fiber_measure_hash=self._axis.measure_hash,
            route_count=route_count,
            cut_count=self._axis.cut_count,
            fiber_count=self._axis.fiber_count,
            fixed_primal_id=self._fixed_primal_id,
            fixed_primal_source_receipt_hash=self._fixed_primal_source_receipt_hash,
            fixed_primal_descriptor_hash=fixed_primal_descriptor_hash,
            fixed_primal_device_fingerprint=fixed_primal_fingerprint,
            witness_stream_source_hash=self._registration.witness_stream_source_hash,
            same_capture_vjp_source_hash=self._registration.same_capture_vjp_source_hash,
            corrected_evidence_stream_source_hash=self._registration.corrected_evidence_stream_source_hash,
            literal_j_stream_source_hash=self._registration.literal_j_stream_source_hash,
            causal_registration_receipt_hash=self._registration.causal_registration_receipt_hash,
            stratum_ids=tuple(item.stratum_id for item in self._registration.strata),  # type: ignore[arg-type]
            stratum_weights=tuple(item.stratum_weight for item in self._registration.strata),  # type: ignore[arg-type]
            population_counts=self._registration.population_counts,
            observed_counts=tuple(self._observed),  # type: ignore[arg-type]
            population_context_axis_hashes=tuple(item.population_context_axis_hash for item in self._registration.strata),  # type: ignore[arg-type]
            population_draw_axis_hashes=tuple(item.population_draw_axis_hash for item in self._registration.strata),  # type: ignore[arg-type]
            population_measurement_axis_hashes=tuple(item.population_measurement_axis_hash for item in self._registration.strata),  # type: ignore[arg-type]
            population_capture_axis_hashes=tuple(item.population_capture_axis_hash for item in self._registration.strata),  # type: ignore[arg-type]
            registered_prefix_context_axis_hashes=prefix_hashes("registered_context_ids"),  # type: ignore[arg-type]
            registered_prefix_draw_axis_hashes=prefix_hashes("registered_draw_ids"),  # type: ignore[arg-type]
            registered_prefix_measurement_axis_hashes=prefix_hashes("registered_measurement_ids"),  # type: ignore[arg-type]
            registered_prefix_capture_axis_hashes=prefix_hashes("registered_capture_ids"),  # type: ignore[arg-type]
            alpha_ledger_id=self._registration.alpha_ledger_id,
            alpha_coordinate_id=self._registration.alpha_coordinate_id,
            look_ordinal=self._registration.look_ordinal,
            familywise_alpha=self._registration.familywise_alpha,
            simultaneous_two_sided_statement_count=statements,
            per_statement_two_sided_alpha=per_two,
            per_statement_one_sided_alpha=per_one,
            corrected_support_source_receipt_hash=self._corrected_support_source_receipt_hash,
            literal_j_support_source_receipt_hash=self._literal_j_support_source_receipt_hash,
            corrected_support_device_fingerprint=corrected_support_fingerprint,
            literal_j_support_device_fingerprint=literal_j_support_fingerprint,
            signed_f_center_fingerprint=fingerprints["signed_f_subgradient_center"],
            signed_f_sampling_radius_fingerprint=fingerprints["signed_f_sampling_radius"],
            signed_f_numerical_radius_fingerprint=fingerprints["signed_f_numerical_radius"],
            signed_f_total_radius_fingerprint=fingerprints["signed_f_subgradient_radius"],
            literal_j_q_gradient_center_fingerprint=fingerprints["literal_j_q_gradient_center"],
            literal_j_q_gradient_sampling_radius_fingerprint=fingerprints["literal_j_q_gradient_sampling_radius"],
            literal_j_q_gradient_numerical_radius_fingerprint=fingerprints["literal_j_q_gradient_numerical_radius"],
            literal_j_q_gradient_total_radius_fingerprint=fingerprints["literal_j_q_gradient_radius"],
            zero_cell_ambiguity_receipt_hash=zero_receipt.receipt_hash,
            zero_cell_route_bitmap_fingerprint=fingerprints[
                "zero_cell_ambiguity_bitmap"
            ],
            exact_finite_population_census=tuple(self._observed) == self._registration.population_counts,
            signed_f_sampling_radius_identically_zero=f_sampling_zero,
            literal_j_q_sampling_radius_identically_zero=q_sampling_zero,
            device_identity=str(self._device),
            production_cuda_execution=production,
            private_cpu_reference_execution=not production,
            exact_census_mode=self._registration.exact_census_mode,
            sampling_alpha_spent=self._registration.sampling_alpha_spent,
            census_supports_derived_from_sealed_rows=self._design.exact_census_windowed_source,
            windowed_two_pass_source_replay=self._design.exact_census_windowed_source,
            sampled_windowed_two_pass_source_replay=self._design.sampled_windowed_source,
            exact_census_source_plan_hash=(
                self._registration.exact_census_source_plan_hash
                if self._registration.exact_census_mode
                else None
            ),
            exact_census_stratum_weight_fractions=(
                self._registration.exact_census_stratum_weight_fractions
                if self._registration.exact_census_mode
                else None
            ),
            census_replay_digest_family_hash=(
                v1_signed_work_fixed_primal_census_replay_digest_family_hash(
                    self._census_replay_digests
                )
                if self._registration.exact_census_mode
                else None
            ),
            sampled_replay_digest_family_hash=(
                v1_signed_work_fixed_primal_sampled_replay_digest_family_hash(
                    self._sampled_replay_digests
                )
                if self._design.sampled_windowed_source
                else None
            ),
            pass1_pass2_value_identity_verified=(
                self._registration.exact_census_mode
                or self._design.sampled_windowed_source
            ),
            _issuance_token=_ISSUER_TOKEN,
        )
        retained = {
            "fixed_primal": self._z,
            "signed_f_subgradient_center": f_center,
            "signed_f_subgradient_radius": f_total,
            "literal_j_q_gradient_center": q_center,
            "literal_j_q_gradient_radius": q_total,
            "zero_cell_ambiguity_bitmap": self._zero_cell_ambiguity_bitmap,
        }
        retained_fingerprints = {
            "fixed_primal": fixed_primal_fingerprint,
            "signed_f_subgradient_center": fingerprints["signed_f_subgradient_center"],
            "signed_f_subgradient_radius": fingerprints["signed_f_subgradient_radius"],
            "literal_j_q_gradient_center": fingerprints["literal_j_q_gradient_center"],
            "literal_j_q_gradient_radius": fingerprints["literal_j_q_gradient_radius"],
            "zero_cell_ambiguity_bitmap": fingerprints[
                "zero_cell_ambiguity_bitmap"
            ],
        }
        handle = V1SignedWorkFixedPrimalProjectionDeviceHandle(
            receipt=receipt,
            zero_cell_ambiguity_receipt=zero_receipt,
            tensors=retained,
            fingerprints=retained_fingerprints,
            route_tile_size=self._design.route_tile_size,
            issuance_token=_ISSUER_TOKEN,
        )
        # Drop every route-scale vector not retained by the live handle.
        self._a_support = None
        self._j_support = None
        self._f_sums = []
        self._q_sums = []
        self._f_ambiguity_sums = []
        self._zero_cell_ambiguity_bitmap = None
        self._z = None
        return V1SignedWorkFixedPrimalProjectionCUDAExecution(
            receipt=receipt,
            zero_cell_ambiguity_receipt=zero_receipt,
            telemetry=telemetry,
            forecast=self._forecast,
            device_handle=handle,
        )

    def __copy__(self) -> NoReturn:
        raise TypeError("fixed-look accumulators are noncopyable")

    def __deepcopy__(self, memo: object) -> NoReturn:
        raise TypeError("fixed-look accumulators are noncopyable")

    def __reduce__(self) -> NoReturn:
        raise TypeError("fixed-look accumulators are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        raise TypeError("fixed-look accumulators are nonserializable")


__all__ = [
    "SIGNED_WORK_FIXED_PRIMAL_ALPHA_FAMILY",
    "SIGNED_WORK_FIXED_PRIMAL_F_EQUATION",
    "SIGNED_WORK_FIXED_PRIMAL_PROJECTION_ALGORITHM",
    "SIGNED_WORK_FIXED_PRIMAL_PROJECTION_AUTHORITY_SCOPE",
    "SIGNED_WORK_FIXED_PRIMAL_Q_EQUATION",
    "SIGNED_WORK_FIXED_PRIMAL_SIGN_POLICY",
    "V1SignedWorkFixedPrimalProjectionCUDAAccumulator",
    "V1SignedWorkFixedPrimalProjectionCUDAExecution",
    "V1SignedWorkFixedPrimalProjectionCUDAExecutionDesign",
    "V1SignedWorkFixedPrimalProjectionCUDAHBMForecast",
    "V1SignedWorkFixedPrimalProjectionCUDARefusal",
    "V1SignedWorkFixedPrimalProjectionCUDARefusalCode",
    "V1SignedWorkFixedPrimalProjectionCUDATelemetry",
    "V1SignedWorkFixedPrimalProjectionDeviceHandle",
    "V1SignedWorkFixedPrimalProjectionDeviceLease",
    "V1SignedWorkFixedPrimalProjectionReceipt",
    "V1SignedWorkFixedPrimalProjectionRegistration",
    "V1SignedWorkFixedPrimalProjectionStratumRegistration",
    "V1SignedWorkFixedPrimalExactCensusSourceSeal",
    "V1SignedWorkFixedPrimalZeroCellAmbiguityReceipt",
    "build_v1_signed_work_fixed_primal_projection_cuda_hbm_forecast",
    "issue_v1_signed_work_fixed_primal_v7_d1_exact_census_registration",
    "v1_signed_work_fixed_primal_census_replay_digest_family_hash",
    "v1_signed_work_fixed_primal_sampled_replay_digest_family_hash",
]
