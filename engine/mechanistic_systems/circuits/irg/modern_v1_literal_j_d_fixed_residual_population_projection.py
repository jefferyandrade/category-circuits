"""Full-population literal-J projection from exact ``D_fixed`` plus residual D.

The route-tiled two-pass owner computes the exact empirical ``D_fixed`` mean
under weight ``1/d``.  That object is not, by itself, a full-train population
geometry.  This seam promotes it only after joining a fresh, pre-registered
finite-population residual projection under the exact decomposition

``(d/N) mean(D_fixed) + sum_h ((N_h-d_h)/N) mean(D_residual,h)``.

Receipts are array-free.  Route-scale values remain in mutation-checked device
handles.  An explicitly small CPU reference remains a numerical oracle only;
the source-bound CUDA residual owner enters through one module-private
capability bridge and must present its exact execution receipt.  No
caller-authored aggregate arrays, scalars, or hashes are accepted by the
promotion issuer.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from decimal import Decimal, ROUND_CEILING, localcontext
from enum import Enum
from fractions import Fraction
import math
from numbers import Real
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Mapping, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_two_pass_d_fixed_projection_cuda import (
    V1LiteralJTwoPassDFixedProjectionExecution,
    V1LiteralJTwoPassDFixedProjectionRegistration,
)
from .modern_v1_sampled_residual_population_measure import (
    V1SampledResidualPopulationMeasureDesign,
    V1SampledResidualPopulationWeightReceipt,
)
from .modern_v1_sampled_variable_shell_dw2_suffix import (
    V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    V1SampledVariableShellDW2SuffixRegistration,
)

try:  # pragma: no cover - runtime image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .modern_v1_literal_j_joint_affine_shell_family_cuda import (
        V1LiteralJJointAffineShellFamilyDeviceHandle,
        V1LiteralJJointAffineShellPreoutcomeRegistration,
    )
    from .modern_v1_literal_j_two_pass_d_fixed_shard_replay import (
        V1LiteralJAffineShellShardReplayResult,
        V1LiteralJTwoPassDFixedShardReplayResult,
    )


LITERAL_J_D_FIXED_RESIDUAL_POPULATION_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_literal_j_D_fixed_residual_population_registration_v3"
)
LITERAL_J_RESIDUAL_POPULATION_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_literal_j_residual_population_projection_receipt_v1"
)
LITERAL_J_D_FIXED_RESIDUAL_POPULATION_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_literal_j_D_fixed_residual_population_projection_receipt_v3"
)
LITERAL_J_D_FIXED_RESIDUAL_POPULATION_ALGORITHM: Final = (
    "exact_rational_D_fixed_plus_stratified_residual_literal_J_projection_v1"
)
LITERAL_J_D_FIXED_RESIDUAL_POPULATION_AUTHORITY: Final = (
    "full_population_fixed_direction_literal_J_projection_geometry_only_no_rank_"
    "kkt_face_C_K_O_commit_terminal_or_pareto_authority_v1"
)
LITERAL_J_JOINT_AXIS_RESIDUAL_POPULATION_MEASURE_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_literal_j_joint_axis_residual_population_measure_"
    "registration_v1"
)
LITERAL_J_JOINT_AXIS_RESIDUAL_POPULATION_MEASURE_AUTHORITY: Final = (
    "preoutcome_action_complete_population_measure_and_lineage_registration_"
    "only_no_D1_numerical_projection_provider_outcome_confidence_FD_solver_D_"
    "C_K_O_commit_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-literal-j-D-fixed-residual-population-"
_REGISTRATION_TOKEN: Final = object()
_RESIDUAL_RECEIPT_TOKEN: Final = object()
_RESIDUAL_LIVE_TOKEN: Final = object()
_POPULATION_RECEIPT_TOKEN: Final = object()
_REPLAY_TOKEN: Final = object()
_RESIDUAL_CUDA_PRODUCER_CAPABILITY: Final = object()
_AFFINE_SHELL_RESIDUAL_CUDA_PRODUCER_CAPABILITY: Final = object()
_JOINT_AXIS_MEASURE_RECEIPT_TOKEN: Final = object()
_JOINT_AXIS_MEASURE_OWNER_TOKEN: Final = object()
_CPU_REFERENCE_ROUTE_LIMIT: Final = 4096
_F64_MIN_SUBNORMAL: Final = float.fromhex("0x0.0000000000001p-1022")


class V1LiteralJDFixedResidualPopulationProjectionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_IDENTITY = "stale_identity"
    COUNT_MISMATCH = "count_mismatch"
    MASS_MISMATCH = "mass_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    POST_SELECTION = "post_selection"
    ALPHA_MISMATCH = "alpha_mismatch"
    D_FIXED_ONLY = "d_fixed_only"
    DEVICE_REQUIRED = "device_required"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    MUTATED_CUSTODY = "mutated_custody"
    EVICTED = "evicted"
    ACTIVE_LEASE = "active_lease"
    NUMERICAL_REFUSAL = "numerical_refusal"
    PREOUTCOME_REQUIRED = "preoutcome_required"
    JOINT_AXIS_MISMATCH = "joint_axis_mismatch"
    MEASURE_LINEAGE_MISMATCH = "measure_lineage_mismatch"
    RESIDENT_RELABEL_FORBIDDEN = "resident_relabel_forbidden"
    AUTHORITY_VIOLATION = "authority_violation"


class V1LiteralJDFixedResidualPopulationProjectionRefusal(RuntimeError):
    """Typed fail-closed refusal for full-population literal-J promotion."""

    def __init__(
        self,
        code: V1LiteralJDFixedResidualPopulationProjectionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LiteralJDFixedResidualPopulationProjectionRefusalCode:
            raise TypeError("population-projection refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJDFixedResidualPopulationProjectionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJDFixedResidualPopulationProjectionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _identity_pair(value: Sequence[str], *, name: str) -> tuple[str, str]:
    if isinstance(value, (str, bytes)):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.INVALID_TYPE,
            f"{name} must be one two-entry identity axis",
        )
    result = tuple(value)
    if len(result) != 2:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.INVALID_TYPE,
            f"{name} must contain exactly two entries",
        )
    first = _identity(result[0], name=f"{name}[0]")
    second = _identity(result[1], name=f"{name}[1]")
    if first == second:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.INVALID_TYPE,
            f"{name} entries must be distinct",
        )
    return first, second


def _identity_axis(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered identity axis",
        )
    result = tuple(_identity(item, name=f"{name} entry") for item in value)
    if not result or len(result) != len(set(result)):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.INVALID_TYPE,
            f"{name} must be nonempty and unique",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _count_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable two-stratum count axis",
        )
    return (
        _nonnegative_int(value[0], name=f"{name}[0]"),
        _nonnegative_int(value[1], name=f"{name}[1]"),
    )


def _fraction_pair(value: Fraction) -> tuple[int, int]:
    return value.numerator, value.denominator


def _validate_fraction_pair(value: object, *, name: str) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] < 0
        or value[1] <= 0
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.MASS_MISMATCH,
            f"{name} must be one exact nonnegative rational pair",
        )
    fraction = Fraction(value[0], value[1])
    return fraction.numerator, fraction.denominator


def _finite_alpha(value: object, *, name: str, allow_zero: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.ALPHA_MISMATCH,
            f"{name} must be one finite float",
        )
    if (allow_zero and not 0.0 <= value < 1.0) or (
        not allow_zero and not 0.0 < value < 1.0
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.ALPHA_MISMATCH,
            f"{name} lies outside its registered probability range",
        )
    return value


def _divide_probability_down(value: float, count: int) -> float:
    """Divide one binary64 probability without exceeding the exact quotient."""

    divisor = _positive_int(count, name="probability family divisor")
    exact = Fraction.from_float(value) / divisor
    result = float(exact)
    if Fraction.from_float(result) > exact:
        result = math.nextafter(result, 0.0)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
            "per-statement probability underflowed binary64",
        )
    return result


def _nonnegative_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be finite and nonnegative",
        )
    return result


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(payload, prefix=_PREFIX + role + "-")


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


def _require_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    device: Any | None = None,
) -> Any:
    if torch is None or type(value) is not torch.Tensor:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.INVALID_TYPE,
            f"{name} must be an exact torch.Tensor",
        )
    if (
        value.dtype != torch.float64
        or tuple(int(item) for item in value.shape) != shape
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.INVALID_TYPE,
            f"{name} must be contiguous FP64 with shape {shape}",
        )
    if not value.is_contiguous() or (device is not None and value.device != device):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.DEVICE_REQUIRED,
            f"{name} left the registered contiguous device axis",
        )
    if not bool(torch.isfinite(value).all().item()):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
            f"{name} contains nonfinite values",
        )
    return value


def _tensor_private_hash(value: Any, *, role: str) -> str:
    """Exact bytes hash for the explicitly small CPU reference only."""

    if value.device.type != "cpu":
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.DEVICE_REQUIRED,
            "production device values require a CUDA-source-owned digest",
        )
    array = value.detach().contiguous().numpy()
    return _payload_hash(
        {
            "schema": "private_cpu_exact_tensor_bytes_v1",
            "role": role,
            "dtype": str(array.dtype),
            "shape": tuple(int(item) for item in array.shape),
            "values": tuple(float(item) for item in array.reshape(-1).tolist()),
        },
        role="private-tensor",
    )


def _up(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _fraction_binary64_center_radius(value: Fraction) -> tuple[float, float]:
    center = float(value)
    if not math.isfinite(center):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
            "exact rational mass overflowed binary64",
        )
    exact = Fraction(value.numerator, value.denominator)
    lower = math.nextafter(center, -math.inf)
    upper = math.nextafter(center, math.inf)
    if not Fraction.from_float(lower) <= exact <= Fraction.from_float(upper):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
            "binary64 rational mass enclosure is not outward",
        )
    return center, max(center - lower, upper - center)


def _fraction_center_radius(value: Fraction) -> tuple[float, float]:
    center = float(value)
    center_fraction = Fraction.from_float(center)
    error = abs(value - center_fraction)
    radius = float(error)
    if Fraction.from_float(radius) < error:
        radius = math.nextafter(radius, math.inf)
    return center, math.nextafter(radius, math.inf)


def _tensor_linf_radius_upper_bound(value: Any, *, name: str) -> float:
    if torch is None:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.DEVICE_REQUIRED,
            "PyTorch is required for live population radius reduction",
        )
    if value.numel() == 0:
        return 0.0
    observed = float(torch.amax(value).detach().cpu().item())
    if not math.isfinite(observed) or observed < 0.0:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
            f"{name} produced a nonfinite or negative radius bound",
        )
    return math.nextafter(observed, math.inf)


def _serfling_radius(
    *, value_range: float, observed: int, population: int, alpha: float
) -> float:
    if population == 0:
        return 0.0
    if population < 0 or observed <= 0 or observed > population:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.D_FIXED_ONLY,
            "positive residual strata require 0 < observed <= population",
        )
    if observed >= population:
        return 0.0
    if not math.isfinite(value_range) or value_range < 0.0:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
            "Serfling value range must be finite and nonnegative",
        )
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.ALPHA_MISMATCH,
            "Serfling alpha must lie strictly between zero and one",
        )
    if value_range == 0.0:
        return 0.0

    # Classical fixed-time Hoeffding--Serfling correction.  Use the same
    # claim-bearing construction as the canonical F/Q/R sequential owners:
    #     ((N - n + 1) / N) * log(2 / alpha) / (2 n).
    # The former variance-style sqrt((N-n)/(N-1)) factor is not the registered
    # Hoeffding--Serfling theorem and can be smaller near census.  Decimal
    # ceiling arithmetic keeps every positive operation outward before the
    # final binary64 conversion.
    correction = Fraction(population - observed + 1, population)
    with localcontext() as context:
        context.prec = 100
        context.rounding = ROUND_CEILING
        span = Decimal.from_float(value_range)
        correction_d = Decimal(correction.numerator) / Decimal(
            correction.denominator
        )
        alpha_d = Decimal.from_float(alpha)
        log_factor = (Decimal(2) / alpha_d).ln().next_plus()
        radius = (
            span
            * (correction_d * log_factor / Decimal(2 * observed)).sqrt()
        ).next_plus()
    result = float(radius)
    if not math.isfinite(result):
        return math.inf
    if Decimal.from_float(result) < radius:
        result = math.nextafter(result, math.inf)
    return math.nextafter(result, math.inf)


@dataclass(frozen=True, slots=True)
class V1LiteralJDFixedResidualPopulationProjectionRegistration:
    """Outcome-independent full-population mixture and residual-inference design."""

    registration_id: str
    d_fixed_registration_hash: str
    population_measure_design_hash: str
    suffix_registration_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    direction_ids: tuple[str, ...]
    direction_source_hash: str
    d_fixed_projection_direction_axis_hash: str
    private_primal_direction_axis_hash: str
    stratum_ids: tuple[str, str]
    population_stratum_counts: tuple[int, int]
    d_fixed_stratum_counts: tuple[int, int]
    residual_stratum_counts: tuple[int, int]
    population_count: int
    d_fixed_count: int
    d_fixed_mass_fraction: tuple[int, int]
    residual_stratum_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    d_fixed_source_replay_axis_hash: str
    d_fixed_registered_prefix_hash: str
    d_fixed_pair_prefix_axis_hash: str
    residual_pair_suffix_axis_hash: str
    residual_sealed_population_order_hashes: tuple[str, str]
    residual_witness_stream_source_hash: str
    residual_literal_j_stream_source_hash: str
    proposal_influencing_history_hash: str
    residual_design_lineage_hash: str
    alpha_ledger_hash: str
    alpha_coordinate_hash: str
    alpha_lineage_hash: str
    familywise_alpha: float
    d_fixed_shard_replay_receipt_hash: str | None = None
    affine_shell_shard_replay_receipt_hash: str | None = None
    # Additive joint-action custody.  Empty defaults preserve legacy v3
    # payloads byte-for-byte; the public joint issuer fills the whole family
    # atomically.  A singleton affine replay is never relabelled as a joint
    # action-complete replay.
    joint_preoutcome_registration_hash: str | None = None
    joint_family_receipt_hash: str | None = None
    joint_postsource_custody_axis_hash: str | None = None
    joint_physical_action_axis_hash: str | None = None
    joint_action_count: int | None = None
    joint_direction_receipt_hashes: tuple[str, ...] = ()
    joint_affine_shell_shard_replay_receipt_hashes: tuple[str, ...] = ()
    joint_d_fixed_registration_hashes: tuple[str, ...] = ()
    joint_d_fixed_source_replay_receipt_hashes: tuple[str, ...] = ()
    registered_before_residual_outcomes: bool = True
    directions_frozen_before_residual_outcomes: bool = True
    residual_order_independent_or_predictable: bool = True
    residual_population_post_selected: bool = False
    proposal_rows_reused_as_residual: bool = False
    array_payload_present: bool = False
    rank_authority: bool = False
    kkt_face_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_D_FIXED_RESIDUAL_POPULATION_ALGORITHM
    authority_scope: str = LITERAL_J_D_FIXED_RESIDUAL_POPULATION_AUTHORITY
    schema: str = LITERAL_J_D_FIXED_RESIDUAL_POPULATION_REGISTRATION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _registration_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _REGISTRATION_TOKEN:
            raise PermissionError("population-projection registration requires issuer")
        for name in (
            "registration_id",
            "d_fixed_registration_hash",
            "population_measure_design_hash",
            "suffix_registration_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "direction_source_hash",
            "d_fixed_projection_direction_axis_hash",
            "private_primal_direction_axis_hash",
            "d_fixed_source_replay_axis_hash",
            "d_fixed_registered_prefix_hash",
            "d_fixed_pair_prefix_axis_hash",
            "residual_pair_suffix_axis_hash",
            "residual_witness_stream_source_hash",
            "residual_literal_j_stream_source_hash",
            "proposal_influencing_history_hash",
            "residual_design_lineage_hash",
            "alpha_ledger_hash",
            "alpha_coordinate_hash",
            "alpha_lineage_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.d_fixed_shard_replay_receipt_hash is not None:
            object.__setattr__(
                self,
                "d_fixed_shard_replay_receipt_hash",
                _identity(
                    self.d_fixed_shard_replay_receipt_hash,
                    name="d_fixed_shard_replay_receipt_hash",
                ),
            )
        if self.affine_shell_shard_replay_receipt_hash is not None:
            object.__setattr__(
                self,
                "affine_shell_shard_replay_receipt_hash",
                _identity(
                    self.affine_shell_shard_replay_receipt_hash,
                    name="affine_shell_shard_replay_receipt_hash",
                ),
            )
        joint_optional_identities = (
            "joint_preoutcome_registration_hash",
            "joint_family_receipt_hash",
            "joint_postsource_custody_axis_hash",
            "joint_physical_action_axis_hash",
        )
        joint_axes = (
            "joint_direction_receipt_hashes",
            "joint_affine_shell_shard_replay_receipt_hashes",
            "joint_d_fixed_registration_hashes",
            "joint_d_fixed_source_replay_receipt_hashes",
        )
        joint_presence = tuple(
            getattr(self, name) is not None for name in joint_optional_identities
        ) + (self.joint_action_count is not None,) + tuple(
            bool(getattr(self, name)) for name in joint_axes
        )
        if any(joint_presence):
            if not all(joint_presence):
                _refuse(
                    V1LiteralJDFixedResidualPopulationProjectionRefusalCode.AUTHORITY_VIOLATION,
                    "joint population custody must be present as one complete family",
                )
            action_count = _positive_int(
                self.joint_action_count,
                name="joint_action_count",
            )
            for name in joint_optional_identities:
                object.__setattr__(
                    self,
                    name,
                    _identity(getattr(self, name), name=name),
                )
            for name in joint_axes:
                value = getattr(self, name)
                if type(value) is not tuple:
                    _refuse(
                        V1LiteralJDFixedResidualPopulationProjectionRefusalCode.INVALID_TYPE,
                        f"{name} must be one immutable ordered action axis",
                    )
                axis = tuple(
                    _identity(item, name=f"{name} entry") for item in value
                )
                if len(axis) != action_count or len(set(axis)) != action_count:
                    _refuse(
                        V1LiteralJDFixedResidualPopulationProjectionRefusalCode.COUNT_MISMATCH,
                        f"{name} must cover each joint physical action exactly once",
                    )
                object.__setattr__(self, name, axis)
            if (
                self.d_fixed_shard_replay_receipt_hash is not None
                or self.affine_shell_shard_replay_receipt_hash is not None
            ):
                _refuse(
                    V1LiteralJDFixedResidualPopulationProjectionRefusalCode.AUTHORITY_VIOLATION,
                    "joint action-complete custody cannot impersonate a singleton replay",
                )
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
            _identity_axis(self.direction_ids, name="direction_ids"),
        )
        object.__setattr__(
            self, "stratum_ids", _identity_pair(self.stratum_ids, name="stratum_ids")
        )
        for name in (
            "population_stratum_counts",
            "d_fixed_stratum_counts",
            "residual_stratum_counts",
        ):
            object.__setattr__(self, name, _count_pair(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "residual_sealed_population_order_hashes",
            _identity_pair(
                self.residual_sealed_population_order_hashes,
                name="residual_sealed_population_order_hashes",
            ),
        )
        object.__setattr__(
            self,
            "population_count",
            _positive_int(self.population_count, name="population_count"),
        )
        object.__setattr__(
            self,
            "d_fixed_count",
            _positive_int(self.d_fixed_count, name="d_fixed_count"),
        )
        d_mass = _validate_fraction_pair(
            self.d_fixed_mass_fraction, name="d_fixed_mass_fraction"
        )
        residual_masses = tuple(
            _validate_fraction_pair(value, name="residual_stratum_mass_fraction")
            for value in self.residual_stratum_mass_fractions
        )
        if len(residual_masses) != 2:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.MASS_MISMATCH,
                "residual masses must cover both strata",
            )
        object.__setattr__(self, "d_fixed_mass_fraction", d_mass)
        object.__setattr__(self, "residual_stratum_mass_fractions", residual_masses)
        population = self.population_stratum_counts
        d_fixed = self.d_fixed_stratum_counts
        residual = self.residual_stratum_counts
        if (
            self.population_count != sum(population)
            or self.d_fixed_count != sum(d_fixed)
            or self.d_fixed_count <= 0
            or any(d_fixed[h] + residual[h] != population[h] for h in (0, 1))
            or Fraction(*d_mass) != Fraction(self.d_fixed_count, self.population_count)
            or any(
                Fraction(*residual_masses[h])
                != Fraction(residual[h], self.population_count)
                for h in (0, 1)
            )
            or Fraction(*d_mass)
            + sum((Fraction(*item) for item in residual_masses), Fraction(0))
            != 1
        ):
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.MASS_MISMATCH,
                "D_fixed and residual rational masses do not partition the population",
            )
        object.__setattr__(
            self,
            "familywise_alpha",
            _finite_alpha(self.familywise_alpha, name="familywise_alpha"),
        )
        positive = (
            self.registered_before_residual_outcomes,
            self.directions_frozen_before_residual_outcomes,
            self.residual_order_independent_or_predictable,
        )
        closed = (
            self.residual_population_post_selected,
            self.proposal_rows_reused_as_residual,
            self.array_payload_present,
            self.rank_authority,
            self.kkt_face_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool for value in positive + closed)
            or any(value is not True for value in positive)
            or any(value is not False for value in closed)
            or self.proposal_influencing_history_hash
            == self.residual_design_lineage_hash
            or self.schema != LITERAL_J_D_FIXED_RESIDUAL_POPULATION_REGISTRATION_SCHEMA
            or self.algorithm != LITERAL_J_D_FIXED_RESIDUAL_POPULATION_ALGORITHM
            or self.authority_scope != LITERAL_J_D_FIXED_RESIDUAL_POPULATION_AUTHORITY
        ):
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.AUTHORITY_VIOLATION,
                "population registration crossed its causal or authority boundary",
            )
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        payload = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }
        # Backward-readable v3: legacy singleton/private registrations did not
        # contain these additive coordinates, so omit their all-empty defaults
        # from the hashed payload.  Joint registrations include the complete
        # family and therefore receive a distinct content identity.
        if self.joint_preoutcome_registration_hash is None:
            for name in (
                "joint_preoutcome_registration_hash",
                "joint_family_receipt_hash",
                "joint_postsource_custody_axis_hash",
                "joint_physical_action_axis_hash",
                "joint_action_count",
                "joint_direction_receipt_hashes",
                "joint_affine_shell_shard_replay_receipt_hashes",
                "joint_d_fixed_registration_hashes",
                "joint_d_fixed_source_replay_receipt_hashes",
            ):
                payload.pop(name)
        return MappingProxyType(payload)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(self._payload(), role="registration")
        if self._registration_hash is not None and self._registration_hash != observed:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
                "population registration mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
                "population registration hash differs",
            )
        object.__setattr__(self, "_registration_hash", observed)
        return observed

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAxisResidualPopulationMeasureRegistrationReceipt:
    """Outcome-independent action-axis registration for one population measure.

    The resident-D1 parent contributes only its already registered population
    decomposition and causal lineage.  Its two private-primal numerical
    projections are preserved as distinct provenance and are never relabelled
    as projections of the action-plus-shell direction axis.
    """

    resident_population_registration_hash: str
    resident_d1_result_receipt_hash: str
    resident_two_direction_ids: tuple[str, str]
    resident_two_direction_axis_hash: str
    resident_private_primal_direction_axis_hash: str
    resident_d1_source_replay_axis_hash: str
    action_axis_preoutcome_receipt_hash: str
    singular_preoutcome_receipt_hashes: tuple[str, ...]
    joint_preoutcome_registration_hash: str
    physical_action_axis_hash: str
    physical_action_identity_hashes: tuple[str, ...]
    physical_action_hashes: tuple[str, ...]
    packed_action_hashes: tuple[str, ...]
    joint_direction_axis_hash: str
    joint_direction_ids: tuple[str, ...]
    population_measure_design_hash: str
    population_measure_weight_receipt_hash: str
    suffix_registration_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    d1_registered_prefix_hash: str
    d1_pair_prefix_axis_hash: str
    residual_pair_suffix_axis_hash: str
    residual_sealed_population_order_hashes: tuple[str, str]
    d1_source_axis_hash: str
    d1_source_registration_hash: str
    d1_literal_j_stream_source_hash: str
    d1_same_capture_vjp_source_hash: str
    residual_witness_stream_source_hash: str
    residual_literal_j_stream_source_hash: str
    proposal_influencing_history_hash: str
    residual_design_lineage_hash: str
    alpha_ledger_hash: str
    alpha_coordinate_hash: str
    alpha_lineage_hash: str
    familywise_alpha: float
    stratum_ids: tuple[str, str]
    population_stratum_counts: tuple[int, int]
    d1_stratum_counts: tuple[int, int]
    residual_population_stratum_counts: tuple[int, int]
    registered_residual_sample_stratum_counts: tuple[int, int]
    population_count: int
    d1_context_count: int
    d1_mass_fraction: tuple[int, int]
    residual_stratum_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    d1_context_weight_fraction: tuple[int, int]
    residual_context_weight_fractions: tuple[tuple[int, int], tuple[int, int]]
    residual_population_mass_fraction: tuple[int, int]
    common_measure_denominator: int
    d1_context_quantum: int
    residual_context_quanta: tuple[int, int]
    total_measure_quanta: int
    action_count: int
    shell_size: int
    direction_count: int
    provider_outcomes_seen: bool = False
    logical_sources_bound_at_registration: bool = False
    exact_complete_action_shell_axis: bool = True
    exact_population_measure_bound: bool = True
    resident_parent_used_for_measure_lineage_only: bool = True
    resident_two_direction_registration_relabelled: bool = False
    numerical_d1_projection_claim: bool = False
    numerical_d1_projection_payload_present: bool = False
    provider_vjp_or_row_read_performed: bool = False
    array_payload_present: bool = False
    residual_population_measure_registration_authority: bool = True
    joint_axis_registration_authority: bool = True
    population_projection_authority: bool = False
    confidence_authority: bool = False
    fd_replacement_authority: bool = False
    solver_authority: bool = False
    d_stability_authority: bool = False
    c_k_o_authority: bool = False
    mask_commit_authority: bool = False
    terminal_or_pareto_authority: bool = False
    authority_scope: str = LITERAL_J_JOINT_AXIS_RESIDUAL_POPULATION_MEASURE_AUTHORITY
    schema: str = LITERAL_J_JOINT_AXIS_RESIDUAL_POPULATION_MEASURE_REGISTRATION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _JOINT_AXIS_MEASURE_RECEIPT_TOKEN:
            raise PermissionError("joint-axis population registrations are issuer-only")
        for name in (
            "resident_population_registration_hash",
            "resident_d1_result_receipt_hash",
            "resident_two_direction_axis_hash",
            "resident_private_primal_direction_axis_hash",
            "resident_d1_source_replay_axis_hash",
            "action_axis_preoutcome_receipt_hash",
            "joint_preoutcome_registration_hash",
            "physical_action_axis_hash",
            "joint_direction_axis_hash",
            "population_measure_design_hash",
            "population_measure_weight_receipt_hash",
            "suffix_registration_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "d1_registered_prefix_hash",
            "d1_pair_prefix_axis_hash",
            "residual_pair_suffix_axis_hash",
            "d1_source_axis_hash",
            "d1_source_registration_hash",
            "d1_literal_j_stream_source_hash",
            "d1_same_capture_vjp_source_hash",
            "residual_witness_stream_source_hash",
            "residual_literal_j_stream_source_hash",
            "proposal_influencing_history_hash",
            "residual_design_lineage_hash",
            "alpha_ledger_hash",
            "alpha_coordinate_hash",
            "alpha_lineage_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "resident_two_direction_ids",
            _identity_pair(
                self.resident_two_direction_ids,
                name="resident_two_direction_ids",
            ),
        )
        for name in (
            "singular_preoutcome_receipt_hashes",
            "physical_action_identity_hashes",
            "physical_action_hashes",
            "packed_action_hashes",
            "joint_direction_ids",
        ):
            object.__setattr__(
                self,
                name,
                _identity_axis(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "residual_sealed_population_order_hashes",
            _identity_pair(
                self.residual_sealed_population_order_hashes,
                name="residual_sealed_population_order_hashes",
            ),
        )
        object.__setattr__(
            self,
            "stratum_ids",
            _identity_pair(self.stratum_ids, name="stratum_ids"),
        )
        for name in (
            "population_stratum_counts",
            "d1_stratum_counts",
            "residual_population_stratum_counts",
            "registered_residual_sample_stratum_counts",
            "residual_context_quanta",
        ):
            object.__setattr__(self, name, _count_pair(getattr(self, name), name=name))
        for name in (
            "proposal_generation",
            "route_count",
            "population_count",
            "d1_context_count",
            "common_measure_denominator",
            "d1_context_quantum",
            "total_measure_quanta",
            "action_count",
            "shell_size",
            "direction_count",
        ):
            validator = _nonnegative_int if name == "proposal_generation" else _positive_int
            object.__setattr__(self, name, validator(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "familywise_alpha",
            _finite_alpha(self.familywise_alpha, name="familywise_alpha"),
        )
        object.__setattr__(
            self,
            "d1_mass_fraction",
            _validate_fraction_pair(self.d1_mass_fraction, name="d1_mass_fraction"),
        )
        residual_masses = tuple(
            _validate_fraction_pair(item, name="residual_stratum_mass_fraction")
            for item in self.residual_stratum_mass_fractions
        )
        residual_weights = tuple(
            _validate_fraction_pair(item, name="residual_context_weight_fraction")
            for item in self.residual_context_weight_fractions
        )
        if len(residual_masses) != 2 or len(residual_weights) != 2:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.MASS_MISMATCH,
                "joint-axis population weights must cover both strata",
            )
        object.__setattr__(self, "residual_stratum_mass_fractions", residual_masses)
        object.__setattr__(self, "residual_context_weight_fractions", residual_weights)
        object.__setattr__(
            self,
            "d1_context_weight_fraction",
            _validate_fraction_pair(
                self.d1_context_weight_fraction,
                name="d1_context_weight_fraction",
            ),
        )
        object.__setattr__(
            self,
            "residual_population_mass_fraction",
            _validate_fraction_pair(
                self.residual_population_mass_fraction,
                name="residual_population_mass_fraction",
            ),
        )

        populations = self.population_stratum_counts
        d1 = self.d1_stratum_counts
        residual = self.residual_population_stratum_counts
        samples = self.registered_residual_sample_stratum_counts
        n = self.population_count
        d = self.d1_context_count
        action_axes = (
            self.singular_preoutcome_receipt_hashes,
            self.physical_action_identity_hashes,
            self.physical_action_hashes,
            self.packed_action_hashes,
        )
        sample_counts_legal = all(
            (residual[h] == 0 and samples[h] == 0)
            or (residual[h] > 0 and 0 < samples[h] <= residual[h])
            for h in (0, 1)
        )
        residual_weights_exact = all(
            Fraction(*residual_weights[h])
            == (
                Fraction(0)
                if residual[h] == 0
                else Fraction(residual[h], n * samples[h])
            )
            for h in (0, 1)
        )
        exact_quanta = (
            self.total_measure_quanta == self.common_measure_denominator
            and Fraction(self.d1_context_quantum, self.common_measure_denominator)
            == Fraction(*self.d1_context_weight_fraction)
            and all(
                Fraction(
                    self.residual_context_quanta[h],
                    self.common_measure_denominator,
                )
                == Fraction(*residual_weights[h])
                for h in (0, 1)
            )
            and d * self.d1_context_quantum
            + sum(
                samples[h] * self.residual_context_quanta[h]
                for h in (0, 1)
            )
            == self.total_measure_quanta
        )
        exact_partition = (
            n == sum(populations)
            and d == sum(d1)
            and all(d1[h] + residual[h] == populations[h] for h in (0, 1))
            and sample_counts_legal
            and Fraction(*self.d1_mass_fraction) == Fraction(d, n)
            and all(
                Fraction(*residual_masses[h]) == Fraction(residual[h], n)
                for h in (0, 1)
            )
            and Fraction(*self.d1_context_weight_fraction) == Fraction(1, n)
            and residual_weights_exact
            and Fraction(*self.residual_population_mass_fraction)
            == Fraction(sum(residual), n)
            and Fraction(*self.d1_mass_fraction)
            + sum((Fraction(*item) for item in residual_masses), Fraction(0))
            == 1
            and exact_quanta
        )
        required_true = (
            self.exact_complete_action_shell_axis,
            self.exact_population_measure_bound,
            self.resident_parent_used_for_measure_lineage_only,
            self.residual_population_measure_registration_authority,
            self.joint_axis_registration_authority,
        )
        required_false = (
            self.provider_outcomes_seen,
            self.logical_sources_bound_at_registration,
            self.resident_two_direction_registration_relabelled,
            self.numerical_d1_projection_claim,
            self.numerical_d1_projection_payload_present,
            self.provider_vjp_or_row_read_performed,
            self.array_payload_present,
            self.population_projection_authority,
            self.confidence_authority,
            self.fd_replacement_authority,
            self.solver_authority,
            self.d_stability_authority,
            self.c_k_o_authority,
            self.mask_commit_authority,
            self.terminal_or_pareto_authority,
        )
        if (
            not exact_partition
            or self.action_count <= 0
            or any(len(axis) != self.action_count for axis in action_axes)
            or self.direction_count != self.action_count + self.shell_size
            or len(self.joint_direction_ids) != self.direction_count
            or self.resident_two_direction_axis_hash == self.joint_direction_axis_hash
            or self.resident_two_direction_ids == self.joint_direction_ids
            or self.proposal_influencing_history_hash
            == self.residual_design_lineage_hash
            or any(type(value) is not bool for value in required_true + required_false)
            or any(value is not True for value in required_true)
            or any(value is not False for value in required_false)
            or self.authority_scope
            != LITERAL_J_JOINT_AXIS_RESIDUAL_POPULATION_MEASURE_AUTHORITY
            or self.schema
            != LITERAL_J_JOINT_AXIS_RESIDUAL_POPULATION_MEASURE_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.AUTHORITY_VIOLATION,
                "joint-axis population registration crossed its measure, axis, or authority boundary",
            )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    def _compute_hash(self) -> str:
        return _payload_hash(self._payload(), role="joint-axis-population-measure")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and expected_hash != observed
        ):
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
                "joint-axis population registration receipt changed after issuance",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def exact_d1_context_weight(self) -> Fraction:
        return Fraction(*self.d1_context_weight_fraction)

    def exact_residual_context_weight(self, stratum_id: str) -> Fraction:
        try:
            index = self.stratum_ids.index(stratum_id)
        except ValueError:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.INVALID_TYPE,
                "unknown residual population stratum",
            )
        return Fraction(*self.residual_context_weight_fractions[index])


class V1LiteralJJointAxisResidualPopulationMeasureRegistration:
    """Process-local custody for an outcome-independent joint-axis registration."""

    __slots__ = (
        "receipt",
        "_action_axis",
        "_resident_lineage",
        "_measure_design",
        "_measure_weights",
        "_state",
        "_lock",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        receipt: V1LiteralJJointAxisResidualPopulationMeasureRegistrationReceipt,
        action_axis: object,
        resident_lineage: V1LiteralJDFixedResidualPopulationProjectionRegistration,
        measure_design: V1SampledResidualPopulationMeasureDesign,
        measure_weights: V1SampledResidualPopulationWeightReceipt,
        _issuance_token: object | None = None,
    ) -> None:
        if _issuance_token is not _JOINT_AXIS_MEASURE_OWNER_TOKEN:
            raise PermissionError("joint-axis population registration owners are issuer-only")
        self.receipt = receipt
        self._action_axis = action_axis
        self._resident_lineage = resident_lineage
        self._measure_design = measure_design
        self._measure_weights = measure_weights
        self._state = "live"
        self._lock = RLock()
        self.assert_preoutcome_live()

    @property
    def is_evicted(self) -> bool:
        return self._state == "evicted"

    def _assert_parent_replay(self, *, require_preoutcome: bool) -> str:
        from .modern_v1_literal_j_joint_population_action_axis import (
            V1LiteralJJointPopulationActionAxisPreoutcomeSource,
        )

        if self._state == "evicted":
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.EVICTED,
                "joint-axis population registration owner was evicted",
            )
        if type(self._action_axis) is not V1LiteralJJointPopulationActionAxisPreoutcomeSource:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
                "joint-axis population registration lost exact PRE custody",
            )
        try:
            self.receipt.assert_integrity()
            if require_preoutcome:
                action_hash = self._action_axis.assert_preoutcome_live()
            elif self._action_axis.is_consumed:
                self._action_axis.receipt.assert_integrity()
                hashes = self._action_axis._replay_sources(require_preoutcome=False)
                if hashes != self._action_axis.receipt.singular_preoutcome_receipt_hashes:
                    raise RuntimeError("singular PRE source axis changed")
                action_hash = self._action_axis.receipt.receipt_hash
            else:
                action_hash = self._action_axis.assert_preoutcome_live()
            resident_hash = self._resident_lineage.assert_integrity()
            design_hash = self._measure_design.assert_integrity()
            weight_hash = self._measure_weights.assert_integrity()
        except V1LiteralJDFixedResidualPopulationProjectionRefusal:
            raise
        except Exception as error:
            code = (
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.PREOUTCOME_REQUIRED
                if require_preoutcome
                else V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY
            )
            _refuse(
                code,
                "joint-axis population source failed live replay: "
                f"{type(error).__name__}: {error}",
            )
        if (
            action_hash != self.receipt.action_axis_preoutcome_receipt_hash
            or resident_hash != self.receipt.resident_population_registration_hash
            or design_hash != self.receipt.population_measure_design_hash
            or weight_hash != self.receipt.population_measure_weight_receipt_hash
        ):
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
                "joint-axis population parent identity changed",
            )
        return self.receipt.receipt_hash

    def assert_preoutcome_live(self) -> str:
        with self._lock:
            return self._assert_parent_replay(require_preoutcome=True)

    def assert_live(self) -> str:
        """Replay lineage both before and after exact action-axis binding."""

        with self._lock:
            return self._assert_parent_replay(require_preoutcome=False)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self.assert_live()
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
                "joint-axis population registration hash differs",
            )
        return observed

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()

    def evict(self) -> None:
        with self._lock:
            self._state = "evicted"

    def __reduce__(self) -> NoReturn:
        raise TypeError("live joint-axis population registrations are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live joint-axis population registrations are nonserializable")


def register_v1_literal_j_joint_axis_residual_population_measure(
    *,
    action_axis_preoutcome_source: object,
    resident_d1_measure_lineage_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
) -> V1LiteralJJointAxisResidualPopulationMeasureRegistration:
    """Join a complete PRE action axis to the existing population measure.

    This issuer deliberately does not borrow, inspect, or relabel the
    resident parent's numerical two-direction projection.  It registers only
    which action-plus-shell directions the already frozen population measure
    will govern when their own D1 and residual projections are later produced.
    """

    from .modern_v1_literal_j_joint_population_action_axis import (
        V1LiteralJJointPopulationActionAxisPreoutcomeSource,
    )

    if type(action_axis_preoutcome_source) is not V1LiteralJJointPopulationActionAxisPreoutcomeSource:
        raise TypeError("action axis must be one exact PRE source")
    if (
        type(resident_d1_measure_lineage_registration)
        is not V1LiteralJDFixedResidualPopulationProjectionRegistration
    ):
        raise TypeError("resident D1 measure-lineage registration must be exact")
    try:
        action_hash = action_axis_preoutcome_source.assert_preoutcome_live()
        resident_hash = resident_d1_measure_lineage_registration.assert_integrity()
        first_source = action_axis_preoutcome_source._sources[0]
        design = first_source._measure_design
        weights = first_source._measure_weight_receipt
        if type(design) is not V1SampledResidualPopulationMeasureDesign or type(
            weights
        ) is not V1SampledResidualPopulationWeightReceipt:
            raise TypeError("action-axis measure parents are not exact")
        design_hash = design.assert_integrity()
        weight_hash = weights.assert_integrity()
        joint = first_source._joint
        preregistration = first_source._registration
        joint_hash = joint.assert_integrity()
        preregistration_hash = preregistration.assert_integrity()
    except V1LiteralJDFixedResidualPopulationProjectionRefusal:
        raise
    except Exception as error:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.PREOUTCOME_REQUIRED,
            "joint-axis population registration requires live outcome-independent parents: "
            f"{type(error).__name__}: {error}",
        )

    pre = action_axis_preoutcome_source.receipt
    singular_pre = first_source.receipt
    resident = resident_d1_measure_lineage_registration
    if (
        resident.joint_preoutcome_registration_hash is not None
        or resident.joint_family_receipt_hash is not None
        or resident.joint_postsource_custody_axis_hash is not None
        or resident.joint_physical_action_axis_hash is not None
        or resident.joint_action_count is not None
        or resident.joint_direction_receipt_hashes
        or resident.joint_affine_shell_shard_replay_receipt_hashes
        or resident.joint_d_fixed_registration_hashes
        or resident.joint_d_fixed_source_replay_receipt_hashes
        or len(resident.direction_ids) != 2
        or resident.d_fixed_shard_replay_receipt_hash
        != resident.d_fixed_registration_hash
        or resident.affine_shell_shard_replay_receipt_hash is not None
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.RESIDENT_RELABEL_FORBIDDEN,
            "the measure-lineage parent must remain the unmodified resident two-private-primal registration",
        )
    if (
        resident.d_fixed_projection_direction_axis_hash
        == pre.joint_direction_axis_hash
        or resident.direction_ids == pre.joint_direction_ids
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.RESIDENT_RELABEL_FORBIDDEN,
            "a resident two-direction numerical axis cannot impersonate the action-plus-shell axis",
        )

    common = (
        (pre.measure_design_hash, design_hash),
        (pre.measure_weight_receipt_hash, weight_hash),
        (resident.population_measure_design_hash, design_hash),
        (resident.suffix_registration_hash, design.suffix_registration_hash),
        (resident.model_identity_hash, pre.model_identity_hash),
        (resident.chart_identity_hash, pre.chart_identity_hash),
        (resident.current_mask_hash, pre.current_mask_hash),
        (resident.proposal_generation, pre.proposal_generation),
        (resident.route_axis_hash, pre.route_axis_hash),
        (resident.route_id_axis_hash, pre.route_id_axis_hash),
        (resident.route_count, pre.route_count),
        (resident.population_stratum_counts, design.population_stratum_counts),
        (resident.population_stratum_counts, pre.population_stratum_counts),
        (resident.d_fixed_stratum_counts, design.d1_stratum_counts),
        (resident.d_fixed_stratum_counts, pre.d1_stratum_counts),
        (resident.residual_stratum_counts, design.residual_stratum_counts),
        (resident.d_fixed_pair_prefix_axis_hash, design.d1_pair_prefix_axis_hash),
        (resident.d_fixed_pair_prefix_axis_hash, singular_pre.d1_pair_prefix_axis_hash),
        (resident.residual_pair_suffix_axis_hash, design.residual_pair_suffix_axis_hash),
        (resident.residual_literal_j_stream_source_hash, pre.literal_j_stream_source_hash),
        (resident.alpha_lineage_hash, pre.alpha_lineage_hash),
        (joint_hash, pre.joint_preoutcome_registration_hash),
        (preregistration_hash, pre.residual_preoutcome_registration_hashes[0]),
        (joint.historical_alpha_ledger_hash, resident.alpha_ledger_hash),
        (joint.historical_alpha_coordinate_hash, resident.alpha_coordinate_hash),
        (joint.historical_alpha_lineage_hash, resident.alpha_lineage_hash),
        (joint.familywise_alpha, resident.familywise_alpha),
        (preregistration.witness_stream_source_hash, resident.residual_witness_stream_source_hash),
    )
    if any(left != right for left, right in common):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.MEASURE_LINEAGE_MISMATCH,
            "the action axis and resident parent do not share one exact population measure and causal lineage",
        )
    if (
        weights.measure_design_hash != design_hash
        or weights.suffix_registration_hash != design.suffix_registration_hash
        or weights.population_stratum_counts != design.population_stratum_counts
        or weights.d1_stratum_counts != design.d1_stratum_counts
        or weights.residual_stratum_counts != design.residual_stratum_counts
        or weights.cumulative_dw2_stratum_counts
        != pre.residual_sample_stratum_counts
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.MEASURE_LINEAGE_MISMATCH,
            "the registered residual weight receipt left its source-issued measure",
        )

    receipt = V1LiteralJJointAxisResidualPopulationMeasureRegistrationReceipt(
        resident_population_registration_hash=resident_hash,
        resident_d1_result_receipt_hash=resident.d_fixed_registration_hash,
        resident_two_direction_ids=resident.direction_ids,  # type: ignore[arg-type]
        resident_two_direction_axis_hash=resident.d_fixed_projection_direction_axis_hash,
        resident_private_primal_direction_axis_hash=resident.private_primal_direction_axis_hash,
        resident_d1_source_replay_axis_hash=resident.d_fixed_source_replay_axis_hash,
        action_axis_preoutcome_receipt_hash=action_hash,
        singular_preoutcome_receipt_hashes=pre.singular_preoutcome_receipt_hashes,
        joint_preoutcome_registration_hash=pre.joint_preoutcome_registration_hash,
        physical_action_axis_hash=pre.physical_action_axis_hash,
        physical_action_identity_hashes=pre.physical_action_identity_hashes,
        physical_action_hashes=pre.physical_action_hashes,
        packed_action_hashes=pre.packed_action_hashes,
        joint_direction_axis_hash=pre.joint_direction_axis_hash,
        joint_direction_ids=pre.joint_direction_ids,
        population_measure_design_hash=design_hash,
        population_measure_weight_receipt_hash=weight_hash,
        suffix_registration_hash=design.suffix_registration_hash,
        model_identity_hash=pre.model_identity_hash,
        chart_identity_hash=pre.chart_identity_hash,
        current_mask_hash=pre.current_mask_hash,
        proposal_generation=pre.proposal_generation,
        route_axis_hash=pre.route_axis_hash,
        route_id_axis_hash=pre.route_id_axis_hash,
        route_count=pre.route_count,
        registered_d_prefix_hash=pre.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=pre.registered_d_w2_prefix_hash,
        d1_registered_prefix_hash=singular_pre.d1_registered_prefix_hash,
        d1_pair_prefix_axis_hash=singular_pre.d1_pair_prefix_axis_hash,
        residual_pair_suffix_axis_hash=design.residual_pair_suffix_axis_hash,
        residual_sealed_population_order_hashes=resident.residual_sealed_population_order_hashes,
        d1_source_axis_hash=pre.d1_source_axis_hash,
        d1_source_registration_hash=pre.d1_source_registration_hash,
        d1_literal_j_stream_source_hash=pre.d1_literal_j_stream_source_hash,
        d1_same_capture_vjp_source_hash=pre.d1_same_capture_vjp_source_hash,
        residual_witness_stream_source_hash=resident.residual_witness_stream_source_hash,
        residual_literal_j_stream_source_hash=pre.literal_j_stream_source_hash,
        proposal_influencing_history_hash=resident.proposal_influencing_history_hash,
        residual_design_lineage_hash=resident.residual_design_lineage_hash,
        alpha_ledger_hash=resident.alpha_ledger_hash,
        alpha_coordinate_hash=resident.alpha_coordinate_hash,
        alpha_lineage_hash=pre.alpha_lineage_hash,
        familywise_alpha=resident.familywise_alpha,
        stratum_ids=resident.stratum_ids,
        population_stratum_counts=design.population_stratum_counts,
        d1_stratum_counts=design.d1_stratum_counts,
        residual_population_stratum_counts=design.residual_stratum_counts,
        registered_residual_sample_stratum_counts=weights.cumulative_dw2_stratum_counts,
        population_count=design.population_count,
        d1_context_count=sum(design.d1_stratum_counts),
        d1_mass_fraction=resident.d_fixed_mass_fraction,
        residual_stratum_mass_fractions=weights.residual_stratum_mass_fractions,
        d1_context_weight_fraction=weights.d1_context_weight_fraction,
        residual_context_weight_fractions=weights.dw2_context_weight_fractions,
        residual_population_mass_fraction=weights.residual_population_scale_fraction,
        common_measure_denominator=weights.common_denominator,
        d1_context_quantum=weights.d1_context_quantum,
        residual_context_quanta=weights.dw2_context_quanta,
        total_measure_quanta=weights.total_weight_quanta,
        action_count=pre.action_count,
        shell_size=pre.shell_size,
        direction_count=pre.direction_count,
        _issuance_token=_JOINT_AXIS_MEASURE_RECEIPT_TOKEN,
    )
    return V1LiteralJJointAxisResidualPopulationMeasureRegistration(
        receipt=receipt,
        action_axis=action_axis_preoutcome_source,
        resident_lineage=resident,
        measure_design=design,
        measure_weights=weights,
        _issuance_token=_JOINT_AXIS_MEASURE_OWNER_TOKEN,
    )


def _register_v1_literal_j_d_fixed_residual_population_projection(
    *,
    registration_id: str,
    d_fixed_registration: V1LiteralJTwoPassDFixedProjectionRegistration,
    population_measure_design: V1SampledResidualPopulationMeasureDesign,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    stratum_ids: Sequence[str],
    residual_sealed_population_order_hashes: Sequence[str],
    residual_witness_stream_source_hash: str,
    residual_literal_j_stream_source_hash: str,
    proposal_influencing_history_hash: str,
    residual_design_lineage_hash: str,
    alpha_ledger_hash: str,
    alpha_coordinate_hash: str,
    alpha_lineage_hash: str,
    familywise_alpha: float,
    d_fixed_shard_replay_receipt_hash: str | None = None,
    affine_shell_shard_replay_receipt_hash: str | None = None,
) -> V1LiteralJDFixedResidualPopulationProjectionRegistration:
    """Bind one full-population mixture before residual literal-J outcomes."""

    if type(d_fixed_registration) is not V1LiteralJTwoPassDFixedProjectionRegistration:
        raise TypeError("D_fixed registration must be exact")
    if type(population_measure_design) is not V1SampledResidualPopulationMeasureDesign:
        raise TypeError("population measure design must be exact")
    if type(suffix_registration) is not V1SampledVariableShellDW2SuffixRegistration:
        raise TypeError("suffix registration must be exact")
    d_fixed_registration.assert_integrity()
    population_measure_design.assert_integrity()
    suffix_registration.assert_integrity()
    residual_orders = _identity_pair(
        residual_sealed_population_order_hashes,
        name="residual_sealed_population_order_hashes",
    )
    if (
        population_measure_design.suffix_registration_hash
        != suffix_registration.receipt_hash
        or population_measure_design.population_stratum_counts
        != suffix_registration.population_stratum_counts
        or population_measure_design.d1_stratum_counts
        != suffix_registration.consumed_d1_stratum_counts
        or population_measure_design.residual_stratum_counts
        != suffix_registration.suffix_capacity_stratum_counts
        or population_measure_design.current_mask_hash
        != d_fixed_registration.current_mask_hash
        or population_measure_design.generation_index
        != d_fixed_registration.proposal_generation
        or sum(population_measure_design.d1_stratum_counts)
        != d_fixed_registration.structural_context_count
        or d_fixed_registration.structural_uniform_weight_fraction
        != (1, d_fixed_registration.structural_context_count)
        or residual_orders
        != (
            suffix_registration.target_suffix_axis_hash,
            suffix_registration.complement_suffix_axis_hash,
        )
        or residual_witness_stream_source_hash
        == d_fixed_registration.composite_witness_stream_source_hash
        or residual_literal_j_stream_source_hash
        == d_fixed_registration.composite_literal_j_stream_source_hash
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
            "D_fixed, suffix, and exact population measure do not share one chart",
        )
    n = population_measure_design.population_count
    d = d_fixed_registration.structural_context_count
    residual = population_measure_design.residual_stratum_counts
    return V1LiteralJDFixedResidualPopulationProjectionRegistration(
        registration_id=registration_id,
        d_fixed_registration_hash=d_fixed_registration.registration_hash,
        population_measure_design_hash=population_measure_design.design_hash,
        suffix_registration_hash=suffix_registration.receipt_hash,
        model_identity_hash=d_fixed_registration.model_identity_hash,
        chart_identity_hash=d_fixed_registration.chart_identity_hash,
        current_mask_hash=d_fixed_registration.current_mask_hash,
        proposal_generation=d_fixed_registration.proposal_generation,
        route_axis_hash=d_fixed_registration.route_axis_hash,
        route_id_axis_hash=d_fixed_registration.route_id_axis_hash,
        route_count=d_fixed_registration.route_count,
        direction_ids=d_fixed_registration.direction_ids,
        direction_source_hash=d_fixed_registration.direction_source_hash,
        d_fixed_projection_direction_axis_hash=(
            d_fixed_registration.direction_axis_hash
        ),
        private_primal_direction_axis_hash=d_fixed_registration.direction_source_hash,
        stratum_ids=tuple(stratum_ids),  # type: ignore[arg-type]
        population_stratum_counts=population_measure_design.population_stratum_counts,
        d_fixed_stratum_counts=population_measure_design.d1_stratum_counts,
        residual_stratum_counts=residual,
        population_count=n,
        d_fixed_count=d,
        d_fixed_mass_fraction=_fraction_pair(Fraction(d, n)),
        residual_stratum_mass_fractions=tuple(
            _fraction_pair(Fraction(value, n)) for value in residual
        ),  # type: ignore[arg-type]
        d_fixed_source_replay_axis_hash=d_fixed_registration.source_replay_axis_hash,
        d_fixed_registered_prefix_hash=d_fixed_registration.registered_d_prefix_hash,
        d_fixed_pair_prefix_axis_hash=population_measure_design.d1_pair_prefix_axis_hash,
        residual_pair_suffix_axis_hash=population_measure_design.residual_pair_suffix_axis_hash,
        residual_sealed_population_order_hashes=residual_orders,
        residual_witness_stream_source_hash=residual_witness_stream_source_hash,
        residual_literal_j_stream_source_hash=residual_literal_j_stream_source_hash,
        proposal_influencing_history_hash=proposal_influencing_history_hash,
        residual_design_lineage_hash=residual_design_lineage_hash,
        alpha_ledger_hash=alpha_ledger_hash,
        alpha_coordinate_hash=alpha_coordinate_hash,
        alpha_lineage_hash=alpha_lineage_hash,
        familywise_alpha=familywise_alpha,
        d_fixed_shard_replay_receipt_hash=d_fixed_shard_replay_receipt_hash,
        affine_shell_shard_replay_receipt_hash=(
            affine_shell_shard_replay_receipt_hash
        ),
        _issuance_token=_REGISTRATION_TOKEN,
    )


def register_v1_literal_j_d_fixed_residual_population_projection(
    *,
    registration_id: str,
    d_fixed_source_result: V1LiteralJTwoPassDFixedShardReplayResult,
    population_measure_design: V1SampledResidualPopulationMeasureDesign,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    stratum_ids: Sequence[str],
    residual_sealed_population_order_hashes: Sequence[str],
    residual_witness_stream_source_hash: str,
    residual_literal_j_stream_source_hash: str,
    proposal_influencing_history_hash: str,
    residual_design_lineage_hash: str,
    alpha_ledger_hash: str,
    alpha_coordinate_hash: str,
    alpha_lineage_hash: str,
    familywise_alpha: float,
) -> V1LiteralJDFixedResidualPopulationProjectionRegistration:
    """Freeze a source-bound population design before residual outcomes."""

    from .modern_v1_literal_j_two_pass_d_fixed_shard_replay import (
        V1LiteralJTwoPassDFixedShardReplayResult,
    )

    if type(d_fixed_source_result) is not V1LiteralJTwoPassDFixedShardReplayResult:
        raise TypeError("D_fixed source result must be exact shard-replay custody")
    d_fixed_source_result.receipt.assert_integrity()
    d_fixed_source_result.registration.assert_integrity()
    if (
        d_fixed_source_result.receipt.registration_hash
        != d_fixed_source_result.registration.registration_hash
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
            "D_fixed source result no longer binds its projection registration",
        )
    return _register_v1_literal_j_d_fixed_residual_population_projection(
        registration_id=registration_id,
        d_fixed_registration=d_fixed_source_result.registration,
        population_measure_design=population_measure_design,
        suffix_registration=suffix_registration,
        stratum_ids=stratum_ids,
        residual_sealed_population_order_hashes=residual_sealed_population_order_hashes,
        residual_witness_stream_source_hash=residual_witness_stream_source_hash,
        residual_literal_j_stream_source_hash=residual_literal_j_stream_source_hash,
        proposal_influencing_history_hash=proposal_influencing_history_hash,
        residual_design_lineage_hash=residual_design_lineage_hash,
        alpha_ledger_hash=alpha_ledger_hash,
        alpha_coordinate_hash=alpha_coordinate_hash,
        alpha_lineage_hash=alpha_lineage_hash,
        familywise_alpha=familywise_alpha,
        d_fixed_shard_replay_receipt_hash=(d_fixed_source_result.receipt.receipt_hash),
    )


def register_v1_literal_j_affine_shell_d_fixed_residual_population_projection(
    *,
    affine_shell_source_result: "V1LiteralJAffineShellShardReplayResult",
    registration_id: str,
    population_measure_design: V1SampledResidualPopulationMeasureDesign,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    stratum_ids: Sequence[str],
    residual_sealed_population_order_hashes: Sequence[str],
    residual_witness_stream_source_hash: str,
    residual_literal_j_stream_source_hash: str,
    proposal_influencing_history_hash: str,
    residual_design_lineage_hash: str,
    alpha_ledger_hash: str,
    alpha_coordinate_hash: str,
    alpha_lineage_hash: str,
    familywise_alpha: float,
) -> V1LiteralJDFixedResidualPopulationProjectionRegistration:
    """Freeze a source-bound affine-shell population design before residual outcomes."""

    from .modern_v1_literal_j_two_pass_d_fixed_shard_replay import (
        V1LiteralJAffineShellShardReplayResult,
    )

    if type(affine_shell_source_result) is not V1LiteralJAffineShellShardReplayResult:
        raise TypeError("affine-shell D_fixed source result must be exact")
    affine_shell_source_result.receipt.assert_integrity()
    source = affine_shell_source_result.source_bound_replay
    source.receipt.assert_integrity()
    source.registration.assert_integrity()
    if (
        affine_shell_source_result.receipt.source_bound_replay_receipt_hash
        != source.receipt.receipt_hash
        or affine_shell_source_result.receipt.registration_hash
        != source.registration.registration_hash
        or affine_shell_source_result.receipt.projection_receipt_hash
        != source.projection.receipt.receipt_hash
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
            "affine-shell D_fixed wrapper no longer binds its generic replay",
        )
    return _register_v1_literal_j_d_fixed_residual_population_projection(
        registration_id=registration_id,
        d_fixed_registration=source.registration,
        population_measure_design=population_measure_design,
        suffix_registration=suffix_registration,
        stratum_ids=stratum_ids,
        residual_sealed_population_order_hashes=residual_sealed_population_order_hashes,
        residual_witness_stream_source_hash=residual_witness_stream_source_hash,
        residual_literal_j_stream_source_hash=residual_literal_j_stream_source_hash,
        proposal_influencing_history_hash=proposal_influencing_history_hash,
        residual_design_lineage_hash=residual_design_lineage_hash,
        alpha_ledger_hash=alpha_ledger_hash,
        alpha_coordinate_hash=alpha_coordinate_hash,
        alpha_lineage_hash=alpha_lineage_hash,
        familywise_alpha=familywise_alpha,
        d_fixed_shard_replay_receipt_hash=source.receipt.receipt_hash,
        affine_shell_shard_replay_receipt_hash=(
            affine_shell_source_result.receipt.receipt_hash
        ),
    )


def register_v1_literal_j_joint_affine_shell_d_fixed_residual_population_projection(
    *,
    joint_preoutcome_registration: "V1LiteralJJointAffineShellPreoutcomeRegistration",
    joint_family_handle: "V1LiteralJJointAffineShellFamilyDeviceHandle",
    affine_shell_source_results: tuple["V1LiteralJAffineShellShardReplayResult", ...],
    registration_id: str,
    population_measure_design: V1SampledResidualPopulationMeasureDesign,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    stratum_ids: Sequence[str],
    residual_sealed_population_order_hashes: Sequence[str],
    residual_witness_stream_source_hash: str,
    residual_literal_j_stream_source_hash: str,
    proposal_influencing_history_hash: str,
    residual_design_lineage_hash: str,
    alpha_ledger_hash: str,
    alpha_coordinate_hash: str,
    alpha_lineage_hash: str,
    familywise_alpha: float,
) -> V1LiteralJDFixedResidualPopulationProjectionRegistration:
    """Issue one action-complete population family from exact source custody.

    Every joint coordinate is derived from the pre-outcome registration, live
    postsource family, and ordered per-action D-fixed replay results.  The
    caller supplies no joint direction or action axis.  This registration is
    population-geometry provenance only; it grants no D/C/downstream decision
    authority.
    """

    from .modern_v1_literal_j_joint_affine_shell_family_cuda import (
        V1LiteralJJointAffineShellFamilyDeviceHandle,
        V1LiteralJJointAffineShellPreoutcomeRegistration,
    )
    from .modern_v1_literal_j_two_pass_d_fixed_shard_replay import (
        V1LiteralJAffineShellShardReplayResult,
    )

    exact_parents = (
        (
            joint_preoutcome_registration,
            V1LiteralJJointAffineShellPreoutcomeRegistration,
        ),
        (joint_family_handle, V1LiteralJJointAffineShellFamilyDeviceHandle),
        (population_measure_design, V1SampledResidualPopulationMeasureDesign),
        (suffix_registration, V1SampledVariableShellDW2SuffixRegistration),
    )
    if any(type(value) is not owner for value, owner in exact_parents):
        raise TypeError("joint population issuance requires exact source owners")
    if type(affine_shell_source_results) is not tuple or any(
        type(item) is not V1LiteralJAffineShellShardReplayResult
        for item in affine_shell_source_results
    ):
        raise TypeError(
            "joint population issuance requires one exact immutable replay tuple"
        )

    try:
        preregistration_hash = joint_preoutcome_registration.assert_integrity()
        joint = joint_family_handle.receipt
        joint_family_hash = joint.assert_integrity()
        population_measure_design.assert_integrity()
        suffix_registration.assert_integrity()
        for result in affine_shell_source_results:
            result.receipt.assert_integrity()
            result.source_bound_replay.receipt.assert_integrity()
            result.registration.assert_integrity()
            result.projection.device_handle.assert_live()
    except V1LiteralJDFixedResidualPopulationProjectionRefusal:
        raise
    except Exception as error:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.MUTATED_CUSTODY,
            "one joint population parent failed live replay: "
            f"{type(error).__name__}: {error}",
        )

    prereg = joint_preoutcome_registration
    action_count = prereg.action_count
    if len(affine_shell_source_results) != action_count:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.COUNT_MISMATCH,
            "joint D-fixed replay tuple must contain exactly one result per physical action",
        )
    if (
        joint.joint_preoutcome_registration_hash != preregistration_hash
        or joint.preoutcome_joint_direction_axis_hash
        != prereg.joint_direction_axis_hash
        or joint.joint_direction_ids != prereg.joint_direction_ids
        or joint.physical_action_ordinals != prereg.physical_action_ordinals
        or joint.physical_action_identity_hashes
        != prereg.physical_action_identity_hashes
        or joint.physical_action_hashes != prereg.physical_action_hashes
        or joint.packed_action_hashes != prereg.packed_action_hashes
        or joint.source_registration_hashes != prereg.source_registration_hashes
        or joint.preoutcome_registration_hashes
        != prereg.preoutcome_registration_hashes
        or joint.destination_binding_hashes != prereg.destination_binding_hashes
        or joint.action_count != action_count
        or joint.shell_indices != prereg.shell_indices
        or joint.shell_route_ids != prereg.shell_route_ids
        or joint.literal_j_stream_source_hash != prereg.literal_j_stream_source_hash
        or joint.historical_alpha_ledger_hash
        != prereg.historical_alpha_ledger_hash
        or joint.historical_alpha_coordinate_hash
        != prereg.historical_alpha_coordinate_hash
        or joint.historical_alpha_lineage_hash
        != prereg.historical_alpha_lineage_hash
        or joint.familywise_alpha != prereg.familywise_alpha
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
            "the live joint family is not the postsource realization of its preregistration",
        )

    replay_receipts = tuple(item.receipt for item in affine_shell_source_results)
    registrations = tuple(item.registration for item in affine_shell_source_results)
    source_receipts = tuple(
        item.source_bound_replay.receipt for item in affine_shell_source_results
    )
    if (
        tuple(item.direction_receipt_hash for item in replay_receipts)
        != joint.direction_receipt_hashes
        or tuple(item.preoutcome_registration_hash for item in replay_receipts)
        != prereg.preoutcome_registration_hashes
        or tuple(item.source_registration_hash for item in replay_receipts)
        != prereg.source_registration_hashes
        or tuple(item.postsource_receipt_hash for item in replay_receipts)
        != joint.postsource_receipt_hashes
        or tuple(item.exact_postsource_custody_hash for item in replay_receipts)
        != joint.exact_postsource_custody_hashes
        or tuple(
            item.projection_replay_registration_receipt_hash
            for item in replay_receipts
        )
        != joint.projection_replay_registration_receipt_hashes
        or tuple(item.preoutcome_destination_binding_hash for item in replay_receipts)
        != prereg.destination_binding_hashes
        or any(
            item.preoutcome_fanout_receipt_hash != prereg.fanout_receipt_hash
            or item.preoutcome_binding_axis_hash != prereg.fanout_binding_axis_hash
            or item.preoutcome_physical_action_axis_hash
            != prereg.physical_action_axis_hash
            or item.literal_j_cell_hash != prereg.literal_j_cell_hash
            or item.literal_j_cell_scope_hash != prereg.literal_j_cell_scope_hash
            or item.literal_j_stream_binding_hash
            != prereg.literal_j_stream_binding_hash
            or item.shell_freeze_receipt_hash != prereg.shell_freeze_receipt_hash
            or item.shell_indices != prereg.shell_indices
            or item.shell_route_ids != prereg.shell_route_ids
            or item.alpha_lineage_hash != prereg.historical_alpha_lineage_hash
            or item.route_count != prereg.route_count
            for item in replay_receipts
        )
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
            "the ordered D-fixed replay tuple is not the complete physical-action family",
        )

    first = registrations[0]
    common_registration_fields = (
        "source_replay_axis_hash",
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "route_count",
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
        "segment_roles",
        "segment_context_counts",
        "segment_uniform_weight_fractions",
        "structural_context_count",
        "structural_uniform_weight_fraction",
        "context_weight_binary64_centers",
        "context_weight_binary64_radii",
        "packet_descriptor_hashes",
    )
    mismatched_fields = tuple(
        name
        for registration in registrations[1:]
        for name in common_registration_fields
        if getattr(registration, name) != getattr(first, name)
    )
    if mismatched_fields:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
            "per-action D-fixed replays do not share one chart/prefix/source axis: "
            + ", ".join(sorted(set(mismatched_fields))),
        )
    if (
        first.model_identity_hash != prereg.model_identity_hash
        or first.chart_identity_hash != prereg.chart_identity_hash
        or first.current_mask_hash != prereg.current_mask_hash
        or first.proposal_generation != prereg.proposal_generation
        or first.route_axis_hash != prereg.route_axis_hash
        or first.route_id_axis_hash != prereg.route_id_axis_hash
        or first.route_count != prereg.route_count
        or first.registered_d_prefix_hash != prereg.registered_d_prefix_hash
        or first.registered_d_w2_prefix_hash != prereg.registered_d_w2_prefix_hash
        or first.composite_literal_j_stream_source_hash
        != prereg.literal_j_stream_source_hash
        or first.composite_same_capture_vjp_source_hash
        != prereg.same_capture_vjp_source_hash
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
            "D-fixed action replays left the preregistered model/chart/prefix/J axis",
        )

    residual_orders = _identity_pair(
        residual_sealed_population_order_hashes,
        name="residual_sealed_population_order_hashes",
    )
    stratum_axis = _identity_pair(stratum_ids, name="stratum_ids")
    if stratum_axis != ("target", "complement"):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.PREFIX_MISMATCH,
            "joint population strata must preserve target then complement order",
        )
    if (
        population_measure_design.suffix_registration_hash
        != suffix_registration.receipt_hash
        or population_measure_design.population_stratum_counts
        != suffix_registration.population_stratum_counts
        or population_measure_design.d1_stratum_counts
        != suffix_registration.consumed_d1_stratum_counts
        or population_measure_design.residual_stratum_counts
        != suffix_registration.suffix_capacity_stratum_counts
        or population_measure_design.current_mask_hash != prereg.current_mask_hash
        or population_measure_design.generation_index != prereg.proposal_generation
        or population_measure_design.d1_pair_prefix_axis_hash
        != suffix_registration.d1_pair_prefix_axis_hash
        or population_measure_design.residual_pair_suffix_axis_hash
        != suffix_registration.suffix_pair_axis_hash
        or sum(population_measure_design.d1_stratum_counts)
        != first.structural_context_count
        or residual_orders
        != (
            suffix_registration.target_suffix_axis_hash,
            suffix_registration.complement_suffix_axis_hash,
        )
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.PREFIX_MISMATCH,
            "joint D-fixed sources and the exact population/suffix measure do not share one prefix",
        )
    residual_witness = _identity(
        residual_witness_stream_source_hash,
        name="residual_witness_stream_source_hash",
    )
    residual_literal = _identity(
        residual_literal_j_stream_source_hash,
        name="residual_literal_j_stream_source_hash",
    )
    if (
        residual_witness == first.composite_witness_stream_source_hash
        or residual_literal != prereg.literal_j_stream_source_hash
        or _identity(alpha_ledger_hash, name="alpha_ledger_hash")
        != prereg.historical_alpha_ledger_hash
        or _identity(alpha_coordinate_hash, name="alpha_coordinate_hash")
        != prereg.historical_alpha_coordinate_hash
        or _identity(alpha_lineage_hash, name="alpha_lineage_hash")
        != prereg.historical_alpha_lineage_hash
        or _finite_alpha(familywise_alpha, name="familywise_alpha")
        != prereg.familywise_alpha
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.ALPHA_MISMATCH,
            "joint residual source or alpha identity left the preregistered J cell",
        )

    replay_hashes = tuple(item.receipt_hash for item in replay_receipts)
    registration_hashes = tuple(item.registration_hash for item in registrations)
    source_replay_receipt_hashes = tuple(
        item.receipt_hash for item in source_receipts
    )
    joint_registration_hash = _payload_hash(
        {
            "schema": "literal_J_joint_D_fixed_registration_axis_v1",
            "joint_preoutcome_registration_hash": preregistration_hash,
            "joint_family_receipt_hash": joint_family_hash,
            "physical_action_axis_hash": prereg.physical_action_axis_hash,
            "ordered_registration_hashes": registration_hashes,
        },
        role="joint-d-fixed-registration-axis",
    )
    joint_source_replay_axis_hash = _payload_hash(
        {
            "schema": "literal_J_joint_D_fixed_source_replay_axis_v1",
            "joint_preoutcome_registration_hash": preregistration_hash,
            "joint_postsource_custody_axis_hash": joint.postsource_custody_axis_hash,
            "ordered_source_replay_axis_hashes": tuple(
                item.source_replay_axis_hash for item in registrations
            ),
            "ordered_source_replay_receipt_hashes": source_replay_receipt_hashes,
            "ordered_affine_replay_receipt_hashes": replay_hashes,
        },
        role="joint-d-fixed-source-replay-axis",
    )

    n = population_measure_design.population_count
    d_counts = population_measure_design.d1_stratum_counts
    d = sum(d_counts)
    residual = population_measure_design.residual_stratum_counts
    return V1LiteralJDFixedResidualPopulationProjectionRegistration(
        registration_id=registration_id,
        d_fixed_registration_hash=joint_registration_hash,
        population_measure_design_hash=population_measure_design.design_hash,
        suffix_registration_hash=suffix_registration.receipt_hash,
        model_identity_hash=prereg.model_identity_hash,
        chart_identity_hash=prereg.chart_identity_hash,
        current_mask_hash=prereg.current_mask_hash,
        proposal_generation=prereg.proposal_generation,
        route_axis_hash=prereg.route_axis_hash,
        route_id_axis_hash=prereg.route_id_axis_hash,
        route_count=prereg.route_count,
        direction_ids=prereg.joint_direction_ids,
        direction_source_hash=joint.postsource_custody_axis_hash,
        d_fixed_projection_direction_axis_hash=prereg.joint_direction_axis_hash,
        private_primal_direction_axis_hash=joint.postsource_custody_axis_hash,
        stratum_ids=stratum_axis,
        population_stratum_counts=population_measure_design.population_stratum_counts,
        d_fixed_stratum_counts=d_counts,
        residual_stratum_counts=residual,
        population_count=n,
        d_fixed_count=d,
        d_fixed_mass_fraction=_fraction_pair(Fraction(d, n)),
        residual_stratum_mass_fractions=tuple(
            _fraction_pair(Fraction(value, n)) for value in residual
        ),  # type: ignore[arg-type]
        d_fixed_source_replay_axis_hash=joint_source_replay_axis_hash,
        d_fixed_registered_prefix_hash=prereg.registered_d_prefix_hash,
        d_fixed_pair_prefix_axis_hash=population_measure_design.d1_pair_prefix_axis_hash,
        residual_pair_suffix_axis_hash=(
            population_measure_design.residual_pair_suffix_axis_hash
        ),
        residual_sealed_population_order_hashes=residual_orders,
        residual_witness_stream_source_hash=residual_witness,
        residual_literal_j_stream_source_hash=residual_literal,
        proposal_influencing_history_hash=proposal_influencing_history_hash,
        residual_design_lineage_hash=residual_design_lineage_hash,
        # The public joint issuer derives each distinct alpha axis from the
        # source-owned historical measurement owner.  Caller strings are
        # consistency guards only and never become authority-bearing values.
        alpha_ledger_hash=prereg.historical_alpha_ledger_hash,
        alpha_coordinate_hash=prereg.historical_alpha_coordinate_hash,
        alpha_lineage_hash=prereg.historical_alpha_lineage_hash,
        familywise_alpha=prereg.familywise_alpha,
        joint_preoutcome_registration_hash=preregistration_hash,
        joint_family_receipt_hash=joint_family_hash,
        joint_postsource_custody_axis_hash=joint.postsource_custody_axis_hash,
        joint_physical_action_axis_hash=prereg.physical_action_axis_hash,
        joint_action_count=action_count,
        joint_direction_receipt_hashes=joint.direction_receipt_hashes,
        joint_affine_shell_shard_replay_receipt_hashes=replay_hashes,
        joint_d_fixed_registration_hashes=registration_hashes,
        joint_d_fixed_source_replay_receipt_hashes=source_replay_receipt_hashes,
        _issuance_token=_REGISTRATION_TOKEN,
    )


def _resident_v7_d1_direction_axis_hash(receipt: object) -> str:
    """Bind the ordered relaxed directions without accepting a hard action."""

    return _payload_hash(
        {
            "schema": "resident_v7_D1_literal_J_direction_axis_v1",
            "branch_axis": tuple(item.value for item in receipt.branch_axis),
            "direction_ids": receipt.direction_ids,
            "direction_descriptor_hashes": receipt.direction_descriptor_hashes,
            "private_primal_source_hash": receipt.private_primal_source_hash,
            "private_primal_descriptor_axis_hash": (
                receipt.private_primal_descriptor_axis_hash
            ),
            "projection_receipt_hashes": receipt.projection_receipt_hashes,
            "preferred_hard_action_present": False,
        },
        role="resident-v7-d1-direction-axis",
    )


def _register_v1_literal_j_d_fixed_residual_population_projection_from_v7_resident_d1(
    *,
    registration_id: str,
    resident_d1_result: object,
    population_measure_design: V1SampledResidualPopulationMeasureDesign,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    stratum_ids: Sequence[str],
    residual_sealed_population_order_hashes: Sequence[str],
    residual_witness_stream_source_hash: str,
    residual_literal_j_stream_source_hash: str,
    proposal_influencing_history_hash: str,
    residual_design_lineage_hash: str,
    alpha_ledger_hash: str,
    alpha_coordinate_hash: str,
    alpha_lineage_hash: str,
    familywise_alpha: float,
) -> V1LiteralJDFixedResidualPopulationProjectionRegistration:
    """Freeze fresh residual inference against resident exact-D1 geometry.

    The resident result replaces the legacy shard-replay D-fixed owner.  Its
    receipt commits both source-issued exact-census registration hashes, and
    its live handle retains the already computed branchwise ``2 M_J z`` and
    ``Q`` intervals.  No preferred hard action participates in this binding.
    """

    from .modern_v1_sampled_v7_resident_d1_literal_j_population import (
        V1SampledV7ResidentD1LiteralJPopulationResult,
    )

    if type(resident_d1_result) is not V1SampledV7ResidentD1LiteralJPopulationResult:
        raise TypeError("resident D1 result must be exact")
    if type(population_measure_design) is not V1SampledResidualPopulationMeasureDesign:
        raise TypeError("population measure design must be exact")
    if type(suffix_registration) is not V1SampledVariableShellDW2SuffixRegistration:
        raise TypeError("suffix registration must be exact")
    resident_d1_result.device_handle.assert_live_identity()
    receipt = resident_d1_result.receipt
    receipt.assert_integrity()
    population_measure_design.assert_integrity()
    suffix_registration.assert_integrity()
    residual_orders = _identity_pair(
        residual_sealed_population_order_hashes,
        name="residual_sealed_population_order_hashes",
    )
    if (
        population_measure_design.suffix_registration_hash
        != suffix_registration.receipt_hash
        or population_measure_design.population_stratum_counts
        != suffix_registration.population_stratum_counts
        or population_measure_design.d1_stratum_counts
        != suffix_registration.consumed_d1_stratum_counts
        or population_measure_design.residual_stratum_counts
        != suffix_registration.suffix_capacity_stratum_counts
        or population_measure_design.d1_stratum_counts
        != receipt.d1_stratum_counts
        or population_measure_design.current_mask_hash != receipt.current_mask_hash
        or population_measure_design.generation_index != receipt.proposal_generation
        or population_measure_design.d1_pair_prefix_axis_hash
        != suffix_registration.d1_pair_prefix_axis_hash
        or residual_orders
        != (
            suffix_registration.target_suffix_axis_hash,
            suffix_registration.complement_suffix_axis_hash,
        )
        or residual_witness_stream_source_hash == receipt.witness_stream_source_hash
        or residual_literal_j_stream_source_hash == receipt.literal_j_stream_source_hash
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
            "resident D1, suffix, and exact population measure do not share one chart",
        )
    n = population_measure_design.population_count
    d_counts = population_measure_design.d1_stratum_counts
    d = sum(d_counts)
    residual = population_measure_design.residual_stratum_counts
    return V1LiteralJDFixedResidualPopulationProjectionRegistration(
        registration_id=registration_id,
        d_fixed_registration_hash=receipt.receipt_hash,
        population_measure_design_hash=population_measure_design.design_hash,
        suffix_registration_hash=suffix_registration.receipt_hash,
        model_identity_hash=receipt.model_identity_hash,
        chart_identity_hash=receipt.chart_identity_hash,
        current_mask_hash=receipt.current_mask_hash,
        proposal_generation=receipt.proposal_generation,
        route_axis_hash=receipt.route_axis_hash,
        route_id_axis_hash=receipt.route_id_axis_hash,
        route_count=receipt.route_count,
        direction_ids=receipt.direction_ids,
        direction_source_hash=receipt.private_primal_source_hash,
        d_fixed_projection_direction_axis_hash=(
            _resident_v7_d1_direction_axis_hash(receipt)
        ),
        private_primal_direction_axis_hash=(
            receipt.private_primal_descriptor_axis_hash
        ),
        stratum_ids=tuple(stratum_ids),  # type: ignore[arg-type]
        population_stratum_counts=population_measure_design.population_stratum_counts,
        d_fixed_stratum_counts=d_counts,
        residual_stratum_counts=residual,
        population_count=n,
        d_fixed_count=d,
        d_fixed_mass_fraction=_fraction_pair(Fraction(d, n)),
        residual_stratum_mass_fractions=tuple(
            _fraction_pair(Fraction(value, n)) for value in residual
        ),  # type: ignore[arg-type]
        d_fixed_source_replay_axis_hash=receipt.exact_census_source_binding_hash,
        d_fixed_registered_prefix_hash=receipt.exact_census_source_plan_hash,
        d_fixed_pair_prefix_axis_hash=population_measure_design.d1_pair_prefix_axis_hash,
        residual_pair_suffix_axis_hash=(
            population_measure_design.residual_pair_suffix_axis_hash
        ),
        residual_sealed_population_order_hashes=residual_orders,
        residual_witness_stream_source_hash=residual_witness_stream_source_hash,
        residual_literal_j_stream_source_hash=residual_literal_j_stream_source_hash,
        proposal_influencing_history_hash=proposal_influencing_history_hash,
        residual_design_lineage_hash=residual_design_lineage_hash,
        alpha_ledger_hash=alpha_ledger_hash,
        alpha_coordinate_hash=alpha_coordinate_hash,
        alpha_lineage_hash=alpha_lineage_hash,
        familywise_alpha=familywise_alpha,
        d_fixed_shard_replay_receipt_hash=receipt.receipt_hash,
        _issuance_token=_REGISTRATION_TOKEN,
    )


def _register_private_cpu_reference_d_fixed_residual_population_projection(
    *,
    registration_id: str,
    d_fixed_registration: V1LiteralJTwoPassDFixedProjectionRegistration,
    population_measure_design: V1SampledResidualPopulationMeasureDesign,
    suffix_registration: V1SampledVariableShellDW2SuffixRegistration,
    stratum_ids: Sequence[str],
    residual_sealed_population_order_hashes: Sequence[str],
    residual_witness_stream_source_hash: str,
    residual_literal_j_stream_source_hash: str,
    proposal_influencing_history_hash: str,
    residual_design_lineage_hash: str,
    alpha_ledger_hash: str,
    alpha_coordinate_hash: str,
    alpha_lineage_hash: str,
    familywise_alpha: float,
) -> V1LiteralJDFixedResidualPopulationProjectionRegistration:
    """Register the small unbound CPU numerical oracle only."""

    return _register_v1_literal_j_d_fixed_residual_population_projection(
        registration_id=registration_id,
        d_fixed_registration=d_fixed_registration,
        population_measure_design=population_measure_design,
        suffix_registration=suffix_registration,
        stratum_ids=stratum_ids,
        residual_sealed_population_order_hashes=residual_sealed_population_order_hashes,
        residual_witness_stream_source_hash=residual_witness_stream_source_hash,
        residual_literal_j_stream_source_hash=residual_literal_j_stream_source_hash,
        proposal_influencing_history_hash=proposal_influencing_history_hash,
        residual_design_lineage_hash=residual_design_lineage_hash,
        alpha_ledger_hash=alpha_ledger_hash,
        alpha_coordinate_hash=alpha_coordinate_hash,
        alpha_lineage_hash=alpha_lineage_hash,
        familywise_alpha=familywise_alpha,
        d_fixed_shard_replay_receipt_hash=None,
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJResidualPopulationProjectionReceipt:
    """Array-free receipt for fresh residual-stratum projection numerics."""

    registration_hash: str
    prefix_freeze_receipt_hash: str
    suffix_registration_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_count: int
    direction_ids: tuple[str, ...]
    d_fixed_projection_direction_axis_hash: str
    private_primal_direction_axis_hash: str
    stratum_ids: tuple[str, str]
    residual_population_counts: tuple[int, int]
    observed_counts: tuple[int, int]
    residual_sealed_population_order_hashes: tuple[str, str]
    observed_prefix_axis_hashes: tuple[str, str]
    observed_source_measurement_axis_hashes: tuple[str, str]
    mjz_center_fingerprint: str
    mjz_radius_fingerprint: str
    q_center_fingerprint: str
    q_radius_fingerprint: str
    simultaneous_statement_count: int
    familywise_alpha: float
    per_statement_two_sided_alpha: float
    alpha_spent: float
    alpha_ledger_hash: str
    alpha_coordinate_hash: str
    alpha_lineage_hash: str
    exact_residual_census: bool
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    residual_registered_before_outcomes: bool = True
    directions_frozen_before_outcomes: bool = True
    independent_or_predictable_residual_order: bool = True
    residual_population_post_selected: bool = False
    proposal_rows_reused: bool = False
    array_free_receipt: bool = True
    live_numeric_handle_required: bool = True
    full_population_authority: bool = False
    rank_authority: bool = False
    kkt_face_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_D_FIXED_RESIDUAL_POPULATION_ALGORITHM
    authority_scope: str = LITERAL_J_D_FIXED_RESIDUAL_POPULATION_AUTHORITY
    schema: str = LITERAL_J_RESIDUAL_POPULATION_PROJECTION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token not in (_RESIDUAL_RECEIPT_TOKEN, _REPLAY_TOKEN):
            raise PermissionError("residual projection receipts require typed issuer")
        for name in (
            "registration_hash",
            "prefix_freeze_receipt_hash",
            "suffix_registration_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "d_fixed_projection_direction_axis_hash",
            "private_primal_direction_axis_hash",
            "mjz_center_fingerprint",
            "mjz_radius_fingerprint",
            "q_center_fingerprint",
            "q_radius_fingerprint",
            "alpha_ledger_hash",
            "alpha_coordinate_hash",
            "alpha_lineage_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route_count")
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal_generation"),
        )
        object.__setattr__(
            self,
            "direction_ids",
            _identity_axis(self.direction_ids, name="direction_ids"),
        )
        object.__setattr__(
            self, "stratum_ids", _identity_pair(self.stratum_ids, name="stratum_ids")
        )
        for name in (
            "residual_population_counts",
            "observed_counts",
        ):
            object.__setattr__(self, name, _count_pair(getattr(self, name), name=name))
        for name in (
            "residual_sealed_population_order_hashes",
            "observed_prefix_axis_hashes",
            "observed_source_measurement_axis_hashes",
        ):
            object.__setattr__(
                self, name, _identity_pair(getattr(self, name), name=name)
            )
        if any(
            self.observed_counts[h] > self.residual_population_counts[h]
            or (self.residual_population_counts[h] > 0 and self.observed_counts[h] <= 0)
            for h in (0, 1)
        ):
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.COUNT_MISMATCH,
                "residual observed counts are not nonempty bounded prefixes",
            )
        object.__setattr__(
            self,
            "simultaneous_statement_count",
            _positive_int(
                self.simultaneous_statement_count, name="simultaneous_statement_count"
            ),
        )
        object.__setattr__(
            self,
            "familywise_alpha",
            _finite_alpha(self.familywise_alpha, name="familywise_alpha"),
        )
        object.__setattr__(
            self,
            "per_statement_two_sided_alpha",
            _finite_alpha(
                self.per_statement_two_sided_alpha, name="per_statement_two_sided_alpha"
            ),
        )
        object.__setattr__(
            self,
            "alpha_spent",
            _finite_alpha(self.alpha_spent, name="alpha_spent", allow_zero=True),
        )
        expected_census = self.observed_counts == self.residual_population_counts
        flags = (
            self.exact_residual_census,
            self.production_cuda_execution,
            self.private_cpu_reference_execution,
            self.residual_registered_before_outcomes,
            self.directions_frozen_before_outcomes,
            self.independent_or_predictable_residual_order,
            self.residual_population_post_selected,
            self.proposal_rows_reused,
            self.array_free_receipt,
            self.live_numeric_handle_required,
            self.full_population_authority,
            self.rank_authority,
            self.kkt_face_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("residual projection flags must be exact booleans")
        if (
            self.exact_residual_census != expected_census
            or self.simultaneous_statement_count
            != 2 * len(self.direction_ids) * (self.route_count + 1)
            or self.per_statement_two_sided_alpha
            != _divide_probability_down(
                self.familywise_alpha, self.simultaneous_statement_count
            )
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or not self.residual_registered_before_outcomes
            or not self.directions_frozen_before_outcomes
            or not self.independent_or_predictable_residual_order
            or self.residual_population_post_selected
            or self.proposal_rows_reused
            or not self.array_free_receipt
            or not self.live_numeric_handle_required
            or any(
                (
                    self.full_population_authority,
                    self.rank_authority,
                    self.kkt_face_authority,
                    self.c_stream_authority,
                    self.calibration_authority,
                    self.oracle_authority,
                    self.commit_authority,
                    self.terminal_authority,
                    self.pareto_authority,
                )
            )
            or (expected_census and self.alpha_spent != 0.0)
            or (not expected_census and self.alpha_spent != self.familywise_alpha)
            or self.schema != LITERAL_J_RESIDUAL_POPULATION_PROJECTION_SCHEMA
            or self.algorithm != LITERAL_J_D_FIXED_RESIDUAL_POPULATION_ALGORITHM
            or self.authority_scope != LITERAL_J_D_FIXED_RESIDUAL_POPULATION_AUTHORITY
        ):
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.AUTHORITY_VIOLATION,
                "residual projection crossed its statistical or authority boundary",
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

    def to_payload(self) -> dict[str, object]:
        return dict(self._payload())

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "V1LiteralJResidualPopulationProjectionReceipt":
        if not isinstance(payload, Mapping):
            raise TypeError("residual projection payload must be a mapping")
        values = dict(payload)
        for name in (
            "direction_ids",
            "stratum_ids",
            "residual_population_counts",
            "observed_counts",
            "residual_sealed_population_order_hashes",
            "observed_prefix_axis_hashes",
            "observed_source_measurement_axis_hashes",
        ):
            values[name] = tuple(values[name])  # type: ignore[arg-type]
        return cls(**values, _issuance_token=_REPLAY_TOKEN)  # type: ignore[arg-type]

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(self._payload(), role="residual-receipt")
        if self._receipt_hash is not None and self._receipt_hash != observed:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
                "residual projection receipt mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
                "residual projection receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class _LiveProjectionLease:
    __slots__ = (
        "_owner",
        "_closed",
        "mjz_center",
        "mjz_radius",
        "q_center",
        "q_radius",
    )

    def __init__(
        self, owner: "_LiveProjectionHandle", tensors: Mapping[str, Any]
    ) -> None:
        self._owner = owner
        self._closed = False
        self.mjz_center = tensors["mjz_center"]
        self.mjz_radius = tensors["mjz_radius"]
        self.q_center = tensors["q_center"]
        self.q_radius = tensors["q_radius"]

    def __enter__(self) -> "_LiveProjectionLease":
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
        raise TypeError("live population projection leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live population projection leases are nonserializable")


class _LiveProjectionHandle:
    __slots__ = ("receipt", "_tensors", "_guards", "_invariants", "_leases", "_evicted")

    def __init__(
        self, *, receipt: object, tensors: Mapping[str, Any], issuance_token: object
    ) -> None:
        if issuance_token is not _RESIDUAL_LIVE_TOKEN:
            raise PermissionError("live population numerics require typed issuer")
        required = ("mjz_center", "mjz_radius", "q_center", "q_radius")
        if set(tensors) != set(required):
            raise TypeError("live projection tensor roles are incomplete")
        self.receipt = receipt
        self._tensors = {name: tensors[name] for name in required}
        self._guards = {name: tensors[name].detach().clone() for name in required}
        self._invariants = {name: _tensor_invariant(tensors[name]) for name in required}
        self._leases = 0
        self._evicted = False
        self.assert_live()

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.EVICTED,
                "population projection handle was evicted",
            )
        getattr(self.receipt, "assert_integrity")()
        for role, tensor in self._tensors.items():
            if _tensor_invariant(tensor) != self._invariants[role] or not bool(
                torch.equal(tensor, self._guards[role])
            ):
                _refuse(
                    V1LiteralJDFixedResidualPopulationProjectionRefusalCode.MUTATED_CUSTODY,
                    "route-scale population projection tensor mutated",
                )

    def borrow(self) -> _LiveProjectionLease:
        self.assert_live()
        self._leases += 1
        return _LiveProjectionLease(self, self._tensors)

    def _close_lease(self) -> None:
        if self._leases <= 0:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.ACTIVE_LEASE,
                "population projection lease accounting underflow",
            )
        try:
            self.assert_live()
        finally:
            self._leases -= 1

    def evict(self) -> None:
        self.assert_live()
        if self._leases:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.ACTIVE_LEASE,
                "cannot evict population projection while leased",
            )
        self._tensors = {}
        self._guards = {}
        self._invariants = {}
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("live population projection handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live population projection handles are nonserializable")


V1LiteralJResidualPopulationProjectionDeviceHandle = _LiveProjectionHandle


@dataclass(frozen=True, slots=True)
class V1LiteralJResidualPopulationProjectionResult:
    receipt: V1LiteralJResidualPopulationProjectionReceipt
    device_handle: _LiveProjectionHandle = field(repr=False, compare=False)
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RESIDUAL_LIVE_TOKEN:
            raise PermissionError("residual projection results require typed issuer")
        if type(self.receipt) is not V1LiteralJResidualPopulationProjectionReceipt:
            raise TypeError("residual result receipt must be exact")
        if type(self.device_handle) is not _LiveProjectionHandle:
            raise TypeError("residual result handle must be exact")
        self.device_handle.assert_live()
        if self.device_handle.receipt.receipt_hash != self.receipt.receipt_hash:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
                "residual receipt and live numerics do not close",
            )


def _issue_v1_literal_j_residual_population_projection_from_cuda_producer(
    *,
    producer_capability: object,
    registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    cuda_registration: object,
    cuda_execution_receipt: object,
    mjz_center: Any,
    mjz_radius: Any,
    q_center: Any,
    q_radius: Any,
) -> V1LiteralJResidualPopulationProjectionResult:
    """Issue canonical live residual numerics from the one CUDA owner.

    The CUDA module cannot import or impersonate the receipt/live-handle
    issuance tokens.  It instead presents this narrower module-private
    capability together with exact source-bound registration and execution
    receipts.  CUDA types are imported only inside the call, avoiding a
    circular module-initialization dependency.
    """

    if producer_capability is not _RESIDUAL_CUDA_PRODUCER_CAPABILITY:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.AUTHORITY_VIOLATION,
            "residual CUDA result issuer rejected an unknown producer capability",
        )
    from .modern_v1_literal_j_residual_population_projection_cuda import (
        V1LiteralJResidualPopulationProjectionCUDAExecutionReceipt,
        V1LiteralJResidualPopulationProjectionCUDARegistration,
        fingerprint_v1_literal_j_residual_cuda_tensor,
    )

    if (
        type(registration)
        is not V1LiteralJDFixedResidualPopulationProjectionRegistration
        or type(prefix_freeze)
        is not V1SampledVariableShellDW2SuffixPrefixFreezeReceipt
        or type(cuda_registration)
        is not V1LiteralJResidualPopulationProjectionCUDARegistration
        or type(cuda_execution_receipt)
        is not V1LiteralJResidualPopulationProjectionCUDAExecutionReceipt
    ):
        raise TypeError(
            "residual CUDA result issuer requires exact population, prefix, registration, and execution types"
        )
    registration.assert_integrity()
    prefix_freeze.assert_integrity()
    cuda_registration.assert_integrity()
    cuda_execution_receipt.assert_integrity()
    execution = cuda_execution_receipt
    observed = execution.observed_counts
    residual = registration.residual_stratum_counts
    statement_count = 2 * len(registration.direction_ids) * (
        registration.route_count + 1
    )
    if (
        registration.d_fixed_shard_replay_receipt_hash is None
        or cuda_registration.population_projection_registration_hash
        != registration.registration_hash
        or execution.population_projection_registration_hash
        != registration.registration_hash
        or execution.cuda_registration_hash != cuda_registration.registration_hash
        or execution.prefix_freeze_receipt_hash != prefix_freeze.receipt_hash
        or prefix_freeze.suffix_registration_hash
        != registration.suffix_registration_hash
        or prefix_freeze.consumed_dw2_stratum_counts != observed
        or execution.suffix_registration_hash != registration.suffix_registration_hash
        or execution.model_identity_hash != registration.model_identity_hash
        or execution.chart_identity_hash != registration.chart_identity_hash
        or execution.current_mask_hash != registration.current_mask_hash
        or execution.proposal_generation != registration.proposal_generation
        or execution.route_axis_hash != registration.route_axis_hash
        or execution.route_count != registration.route_count
        or execution.direction_ids != registration.direction_ids
        or execution.d_fixed_projection_direction_axis_hash
        != registration.d_fixed_projection_direction_axis_hash
        or execution.private_primal_direction_axis_hash
        != registration.private_primal_direction_axis_hash
        or execution.stratum_ids != registration.stratum_ids
        or execution.residual_population_counts != residual
        or execution.residual_sealed_population_order_hashes
        != registration.residual_sealed_population_order_hashes
        or cuda_registration.support_bounds_receipt_hash
        != execution.support_bounds_receipt_hash
        or execution.simultaneous_statement_count != statement_count
        or execution.familywise_alpha != registration.familywise_alpha
        or execution.per_statement_two_sided_alpha
        != _divide_probability_down(registration.familywise_alpha, statement_count)
        or execution.production_cuda_execution is not True
        or execution.private_cpu_reference_execution is not False
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
            "residual CUDA execution left its registered chart, prefix, direction, support, or alpha lineage",
        )
    direction_count = len(registration.direction_ids)
    route_count = registration.route_count
    centers_m = _require_tensor(
        mjz_center,
        name="CUDA residual M_Jz center",
        shape=(2, direction_count, route_count),
    )
    radii_m = _require_tensor(
        mjz_radius,
        name="CUDA residual M_Jz radius",
        shape=(2, direction_count, route_count),
        device=centers_m.device,
    )
    centers_q = _require_tensor(
        q_center,
        name="CUDA residual Q center",
        shape=(2, direction_count),
        device=centers_m.device,
    )
    radii_q = _require_tensor(
        q_radius,
        name="CUDA residual Q radius",
        shape=(2, direction_count),
        device=centers_m.device,
    )
    if centers_m.device.type != "cuda":
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.CPU_PRODUCTION_FALLBACK,
            "source-bound residual CUDA numerics must remain CUDA-resident",
        )
    if bool((radii_m < 0.0).any().item()) or bool((radii_q < 0.0).any().item()):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
            "residual CUDA interval radius is negative",
        )
    observed_fingerprints = (
        fingerprint_v1_literal_j_residual_cuda_tensor(
            centers_m, role="residual_mjz_center"
        ),
        fingerprint_v1_literal_j_residual_cuda_tensor(
            radii_m, role="residual_mjz_radius"
        ),
        fingerprint_v1_literal_j_residual_cuda_tensor(
            centers_q, role="residual_q_center"
        ),
        fingerprint_v1_literal_j_residual_cuda_tensor(
            radii_q, role="residual_q_radius"
        ),
    )
    if observed_fingerprints != (
        execution.mjz_center_fingerprint,
        execution.mjz_radius_fingerprint,
        execution.q_center_fingerprint,
        execution.q_radius_fingerprint,
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.MUTATED_CUSTODY,
            "residual CUDA output tensors differ from their execution receipt",
        )
    exact_census = observed == residual
    receipt = V1LiteralJResidualPopulationProjectionReceipt(
        registration_hash=registration.registration_hash,
        prefix_freeze_receipt_hash=prefix_freeze.receipt_hash,
        suffix_registration_hash=registration.suffix_registration_hash,
        model_identity_hash=registration.model_identity_hash,
        chart_identity_hash=registration.chart_identity_hash,
        current_mask_hash=registration.current_mask_hash,
        proposal_generation=registration.proposal_generation,
        route_axis_hash=registration.route_axis_hash,
        route_count=registration.route_count,
        direction_ids=registration.direction_ids,
        d_fixed_projection_direction_axis_hash=(
            registration.d_fixed_projection_direction_axis_hash
        ),
        private_primal_direction_axis_hash=(
            registration.private_primal_direction_axis_hash
        ),
        stratum_ids=registration.stratum_ids,
        residual_population_counts=residual,
        observed_counts=observed,
        residual_sealed_population_order_hashes=(
            registration.residual_sealed_population_order_hashes
        ),
        observed_prefix_axis_hashes=execution.observed_prefix_axis_hashes,
        observed_source_measurement_axis_hashes=(
            execution.observed_source_measurement_axis_hashes
        ),
        mjz_center_fingerprint=observed_fingerprints[0],
        mjz_radius_fingerprint=observed_fingerprints[1],
        q_center_fingerprint=observed_fingerprints[2],
        q_radius_fingerprint=observed_fingerprints[3],
        simultaneous_statement_count=statement_count,
        familywise_alpha=registration.familywise_alpha,
        per_statement_two_sided_alpha=(
            _divide_probability_down(registration.familywise_alpha, statement_count)
        ),
        alpha_spent=0.0 if exact_census else registration.familywise_alpha,
        alpha_ledger_hash=registration.alpha_ledger_hash,
        alpha_coordinate_hash=registration.alpha_coordinate_hash,
        alpha_lineage_hash=registration.alpha_lineage_hash,
        exact_residual_census=exact_census,
        production_cuda_execution=True,
        private_cpu_reference_execution=False,
        _issuance_token=_RESIDUAL_RECEIPT_TOKEN,
    )
    handle = _LiveProjectionHandle(
        receipt=receipt,
        tensors={
            "mjz_center": centers_m,
            "mjz_radius": radii_m,
            "q_center": centers_q,
            "q_radius": radii_q,
        },
        issuance_token=_RESIDUAL_LIVE_TOKEN,
    )
    return V1LiteralJResidualPopulationProjectionResult(
        receipt=receipt,
        device_handle=handle,
        _issuance_token=_RESIDUAL_LIVE_TOKEN,
    )


def _issue_v1_literal_j_affine_shell_residual_population_projection_from_cuda_producer(
    *,
    producer_capability: object,
    registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    cuda_registration: object,
    cuda_execution_receipt: object,
    mjz_center: Any,
    mjz_radius: Any,
    q_center: Any,
    q_radius: Any,
) -> V1LiteralJResidualPopulationProjectionResult:
    """Issue canonical residual numerics from the affine-shell CUDA sibling."""

    if producer_capability is not _AFFINE_SHELL_RESIDUAL_CUDA_PRODUCER_CAPABILITY:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.AUTHORITY_VIOLATION,
            "affine-shell residual CUDA issuer rejected an unknown producer capability",
        )
    from .modern_v1_literal_j_affine_shell_residual_population_projection_cuda import (
        V1LiteralJAffineShellResidualPopulationProjectionCUDAExecutionReceipt,
        V1LiteralJAffineShellResidualPopulationProjectionCUDARegistration,
        fingerprint_v1_literal_j_affine_shell_residual_cuda_tensor,
    )

    if (
        type(registration)
        is not V1LiteralJDFixedResidualPopulationProjectionRegistration
        or type(prefix_freeze)
        is not V1SampledVariableShellDW2SuffixPrefixFreezeReceipt
        or type(cuda_registration)
        is not V1LiteralJAffineShellResidualPopulationProjectionCUDARegistration
        or type(cuda_execution_receipt)
        is not V1LiteralJAffineShellResidualPopulationProjectionCUDAExecutionReceipt
    ):
        raise TypeError(
            "affine-shell residual issuer requires exact population, prefix, registration, and execution types"
        )
    registration.assert_integrity()
    prefix_freeze.assert_integrity()
    cuda_registration.assert_integrity()
    cuda_execution_receipt.assert_integrity()
    execution = cuda_execution_receipt
    observed = execution.observed_counts
    residual = registration.residual_stratum_counts
    statement_count = 2 * len(registration.direction_ids) * (
        registration.route_count + 1
    )
    if (
        registration.affine_shell_shard_replay_receipt_hash is None
        or registration.d_fixed_shard_replay_receipt_hash is None
        or cuda_registration.population_projection_registration_hash
        != registration.registration_hash
        or execution.population_projection_registration_hash
        != registration.registration_hash
        or execution.cuda_registration_hash != cuda_registration.registration_hash
        or execution.prefix_freeze_receipt_hash != prefix_freeze.receipt_hash
        or prefix_freeze.suffix_registration_hash
        != registration.suffix_registration_hash
        or prefix_freeze.consumed_dw2_stratum_counts != observed
        or execution.suffix_registration_hash != registration.suffix_registration_hash
        or execution.model_identity_hash != registration.model_identity_hash
        or execution.chart_identity_hash != registration.chart_identity_hash
        or execution.current_mask_hash != registration.current_mask_hash
        or execution.proposal_generation != registration.proposal_generation
        or execution.route_axis_hash != registration.route_axis_hash
        or execution.route_count != registration.route_count
        or execution.direction_ids != registration.direction_ids
        or execution.d_fixed_projection_direction_axis_hash
        != registration.d_fixed_projection_direction_axis_hash
        or execution.affine_direction_source_hash != registration.direction_source_hash
        or execution.affine_shell_shard_replay_receipt_hash
        != registration.affine_shell_shard_replay_receipt_hash
        or execution.stratum_ids != registration.stratum_ids
        or execution.residual_population_counts != residual
        or execution.residual_sealed_population_order_hashes
        != registration.residual_sealed_population_order_hashes
        or cuda_registration.affine_support_receipt_hash
        != execution.affine_support_receipt_hash
        or cuda_registration.route_support_source_receipt_hash
        != execution.route_support_source_receipt_hash
        or execution.simultaneous_statement_count != statement_count
        or execution.familywise_alpha != registration.familywise_alpha
        or execution.per_statement_two_sided_alpha
        != _divide_probability_down(registration.familywise_alpha, statement_count)
        or execution.production_cuda_execution
        == execution.private_cpu_reference_execution
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
            "affine-shell residual CUDA execution left its registered chart, direction, support, or alpha lineage",
        )
    direction_count = len(registration.direction_ids)
    route_count = registration.route_count
    centers_m = _require_tensor(
        mjz_center,
        name="affine-shell residual M_Jz center",
        shape=(2, direction_count, route_count),
    )
    radii_m = _require_tensor(
        mjz_radius,
        name="affine-shell residual M_Jz radius",
        shape=(2, direction_count, route_count),
        device=centers_m.device,
    )
    centers_q = _require_tensor(
        q_center,
        name="affine-shell residual Q center",
        shape=(2, direction_count),
        device=centers_m.device,
    )
    radii_q = _require_tensor(
        q_radius,
        name="affine-shell residual Q radius",
        shape=(2, direction_count),
        device=centers_m.device,
    )
    if execution.production_cuda_execution and centers_m.device.type != "cuda":
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.CPU_PRODUCTION_FALLBACK,
            "affine-shell residual production numerics must remain CUDA-resident",
        )
    if execution.private_cpu_reference_execution and centers_m.device.type != "cpu":
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.CPU_PRODUCTION_FALLBACK,
            "affine-shell residual CPU reference must remain CPU-resident",
        )
    if bool((radii_m < 0.0).any().item()) or bool((radii_q < 0.0).any().item()):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
            "affine-shell residual interval radius is negative",
        )
    observed_fingerprints = (
        fingerprint_v1_literal_j_affine_shell_residual_cuda_tensor(
            centers_m, role="affine_shell_residual_mjz_center"
        ),
        fingerprint_v1_literal_j_affine_shell_residual_cuda_tensor(
            radii_m, role="affine_shell_residual_mjz_radius"
        ),
        fingerprint_v1_literal_j_affine_shell_residual_cuda_tensor(
            centers_q, role="affine_shell_residual_q_center"
        ),
        fingerprint_v1_literal_j_affine_shell_residual_cuda_tensor(
            radii_q, role="affine_shell_residual_q_radius"
        ),
    )
    if observed_fingerprints != (
        execution.mjz_center_fingerprint,
        execution.mjz_radius_fingerprint,
        execution.q_center_fingerprint,
        execution.q_radius_fingerprint,
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.MUTATED_CUSTODY,
            "affine-shell residual CUDA tensors differ from their execution receipt",
        )
    exact_census = observed == residual
    receipt = V1LiteralJResidualPopulationProjectionReceipt(
        registration_hash=registration.registration_hash,
        prefix_freeze_receipt_hash=prefix_freeze.receipt_hash,
        suffix_registration_hash=registration.suffix_registration_hash,
        model_identity_hash=registration.model_identity_hash,
        chart_identity_hash=registration.chart_identity_hash,
        current_mask_hash=registration.current_mask_hash,
        proposal_generation=registration.proposal_generation,
        route_axis_hash=registration.route_axis_hash,
        route_count=registration.route_count,
        direction_ids=registration.direction_ids,
        d_fixed_projection_direction_axis_hash=(
            registration.d_fixed_projection_direction_axis_hash
        ),
        private_primal_direction_axis_hash=(
            registration.private_primal_direction_axis_hash
        ),
        stratum_ids=registration.stratum_ids,
        residual_population_counts=residual,
        observed_counts=observed,
        residual_sealed_population_order_hashes=(
            registration.residual_sealed_population_order_hashes
        ),
        observed_prefix_axis_hashes=execution.observed_prefix_axis_hashes,
        observed_source_measurement_axis_hashes=(
            execution.observed_source_measurement_axis_hashes
        ),
        mjz_center_fingerprint=observed_fingerprints[0],
        mjz_radius_fingerprint=observed_fingerprints[1],
        q_center_fingerprint=observed_fingerprints[2],
        q_radius_fingerprint=observed_fingerprints[3],
        simultaneous_statement_count=statement_count,
        familywise_alpha=registration.familywise_alpha,
        per_statement_two_sided_alpha=(
            _divide_probability_down(registration.familywise_alpha, statement_count)
        ),
        alpha_spent=0.0 if exact_census else registration.familywise_alpha,
        alpha_ledger_hash=registration.alpha_ledger_hash,
        alpha_coordinate_hash=registration.alpha_coordinate_hash,
        alpha_lineage_hash=registration.alpha_lineage_hash,
        exact_residual_census=exact_census,
        production_cuda_execution=execution.production_cuda_execution,
        private_cpu_reference_execution=execution.private_cpu_reference_execution,
        _issuance_token=_RESIDUAL_RECEIPT_TOKEN,
    )
    handle = _LiveProjectionHandle(
        receipt=receipt,
        tensors={
            "mjz_center": centers_m,
            "mjz_radius": radii_m,
            "q_center": centers_q,
            "q_radius": radii_q,
        },
        issuance_token=_RESIDUAL_LIVE_TOKEN,
    )
    return V1LiteralJResidualPopulationProjectionResult(
        receipt=receipt,
        device_handle=handle,
        _issuance_token=_RESIDUAL_LIVE_TOKEN,
    )


def _observed_prefix_hash(
    *, registration_hash: str, stratum_id: str, observed: int, source_hash: str
) -> str:
    return _payload_hash(
        {
            "schema": "registered_residual_prefix_identity_v1",
            "registration_hash": registration_hash,
            "stratum_id": stratum_id,
            "observed_count": observed,
            "source_hash": source_hash,
        },
        role="residual-prefix",
    )


def _issue_private_cpu_reference_residual_population_projection(
    *,
    registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    prefix_freeze: V1SampledVariableShellDW2SuffixPrefixFreezeReceipt,
    directions: Any,
    direction_source_hash: str,
    literal_j_rows_by_stratum: Sequence[Any],
    literal_j_abs_bound: Any,
    projection_abs_bounds: Any,
    observed_source_measurement_axis_hashes: Sequence[str],
) -> V1LiteralJResidualPopulationProjectionResult:
    """Small exact-arithmetic CPU oracle; never a production fallback."""

    if (
        type(registration)
        is not V1LiteralJDFixedResidualPopulationProjectionRegistration
    ):
        raise TypeError("population registration must be exact")
    if type(prefix_freeze) is not V1SampledVariableShellDW2SuffixPrefixFreezeReceipt:
        raise TypeError("residual prefix freeze must be exact")
    registration.assert_integrity()
    prefix_freeze.assert_integrity()
    if (
        prefix_freeze.suffix_registration_hash != registration.suffix_registration_hash
        or prefix_freeze.consumed_d1_stratum_counts
        != registration.d_fixed_stratum_counts
        or prefix_freeze.population_stratum_counts
        != registration.population_stratum_counts
        or _identity(direction_source_hash, name="direction_source_hash")
        != registration.direction_source_hash
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.PREFIX_MISMATCH,
            "residual prefix left its registered population or direction lineage",
        )
    observed = prefix_freeze.consumed_dw2_stratum_counts
    residual = registration.residual_stratum_counts
    if any(
        observed[h] > residual[h] or (residual[h] > 0 and observed[h] <= 0)
        for h in (0, 1)
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.D_FIXED_ONLY,
            "positive residual population requires a nonempty fresh prefix",
        )
    d_count = len(registration.direction_ids)
    r_count = registration.route_count
    if r_count > _CPU_REFERENCE_ROUTE_LIMIT:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.CPU_PRODUCTION_FALLBACK,
            "private CPU residual reference is limited to tiny route domains",
        )
    z = _require_tensor(directions, name="directions", shape=(d_count, r_count))
    if z.device.type != "cpu":
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.CPU_PRODUCTION_FALLBACK,
            "private residual issuer accepts CPU tensors only",
        )
    if len(literal_j_rows_by_stratum) != 2:
        raise TypeError("literal-J rows must cover both residual strata")
    rows = tuple(
        _require_tensor(
            literal_j_rows_by_stratum[h],
            name=f"literal_j_rows_by_stratum[{h}]",
            shape=(observed[h], r_count),
            device=z.device,
        )
        for h in (0, 1)
    )
    j_bound = _require_tensor(
        literal_j_abs_bound,
        name="literal_j_abs_bound",
        shape=(r_count,),
        device=z.device,
    )
    p_bound = _require_tensor(
        projection_abs_bounds,
        name="projection_abs_bounds",
        shape=(d_count,),
        device=z.device,
    )
    if bool((j_bound < 0.0).any().item()) or bool((p_bound < 0.0).any().item()):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
            "pre-outcome support bounds must be nonnegative",
        )
    for row in rows:
        if row.numel() and bool((torch.abs(row) > j_bound.reshape(1, -1)).any().item()):
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
                "literal-J row exceeded its registered absolute support",
            )
    centers_m = torch.zeros((2, d_count, r_count), dtype=torch.float64)
    radii_m = torch.zeros_like(centers_m)
    centers_q = torch.zeros((2, d_count), dtype=torch.float64)
    radii_q = torch.zeros_like(centers_q)
    exact_census = observed == residual
    statement_count = 2 * d_count * (r_count + 1)
    per_statement_alpha = _divide_probability_down(
        registration.familywise_alpha, statement_count
    )
    for h in (0, 1):
        if residual[h] == 0:
            continue
        exact_m = [[Fraction(0) for _ in range(r_count)] for _ in range(d_count)]
        exact_q = [Fraction(0) for _ in range(d_count)]
        for row_index in range(observed[h]):
            row_fraction = [
                Fraction.from_float(float(rows[h][row_index, route].item()))
                for route in range(r_count)
            ]
            for direction in range(d_count):
                projection = sum(
                    (
                        Fraction.from_float(float(z[direction, route].item()))
                        * row_fraction[route]
                        for route in range(r_count)
                    ),
                    Fraction(0),
                )
                if abs(float(projection)) > float(p_bound[direction].item()):
                    _refuse(
                        V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
                        "literal-J projection exceeded its registered absolute support",
                    )
                exact_q[direction] += projection * projection
                for route in range(r_count):
                    exact_m[direction][route] += projection * row_fraction[route]
        for direction in range(d_count):
            q_fraction = exact_q[direction] / observed[h]
            q_center, q_numeric_radius = _fraction_center_radius(q_fraction)
            q_sampling = _serfling_radius(
                value_range=float(p_bound[direction].item()) ** 2,
                observed=observed[h],
                population=residual[h],
                alpha=per_statement_alpha,
            )
            centers_q[h, direction] = q_center
            radii_q[h, direction] = math.nextafter(
                q_numeric_radius + q_sampling, math.inf
            )
            for route in range(r_count):
                value = exact_m[direction][route] / observed[h]
                center, numeric_radius = _fraction_center_radius(value)
                sampling = _serfling_radius(
                    value_range=2.0
                    * float(p_bound[direction].item())
                    * float(j_bound[route].item()),
                    observed=observed[h],
                    population=residual[h],
                    alpha=per_statement_alpha,
                )
                centers_m[h, direction, route] = center
                radii_m[h, direction, route] = math.nextafter(
                    numeric_radius + sampling, math.inf
                )
    source_hashes = _identity_pair(
        observed_source_measurement_axis_hashes,
        name="observed_source_measurement_axis_hashes",
    )
    observed_prefix_hashes = (
        _observed_prefix_hash(
            registration_hash=registration.registration_hash,
            stratum_id=registration.stratum_ids[0],
            observed=observed[0],
            source_hash=prefix_freeze.dw2_target_prefix_axis_hash,
        ),
        _observed_prefix_hash(
            registration_hash=registration.registration_hash,
            stratum_id=registration.stratum_ids[1],
            observed=observed[1],
            source_hash=prefix_freeze.dw2_complement_prefix_axis_hash,
        ),
    )
    receipt = V1LiteralJResidualPopulationProjectionReceipt(
        registration_hash=registration.registration_hash,
        prefix_freeze_receipt_hash=prefix_freeze.receipt_hash,
        suffix_registration_hash=registration.suffix_registration_hash,
        model_identity_hash=registration.model_identity_hash,
        chart_identity_hash=registration.chart_identity_hash,
        current_mask_hash=registration.current_mask_hash,
        proposal_generation=registration.proposal_generation,
        route_axis_hash=registration.route_axis_hash,
        route_count=registration.route_count,
        direction_ids=registration.direction_ids,
        d_fixed_projection_direction_axis_hash=(
            registration.d_fixed_projection_direction_axis_hash
        ),
        private_primal_direction_axis_hash=registration.private_primal_direction_axis_hash,
        stratum_ids=registration.stratum_ids,
        residual_population_counts=residual,
        observed_counts=observed,
        residual_sealed_population_order_hashes=registration.residual_sealed_population_order_hashes,
        observed_prefix_axis_hashes=observed_prefix_hashes,
        observed_source_measurement_axis_hashes=source_hashes,
        mjz_center_fingerprint=_tensor_private_hash(
            centers_m, role="residual_mjz_center"
        ),
        mjz_radius_fingerprint=_tensor_private_hash(
            radii_m, role="residual_mjz_radius"
        ),
        q_center_fingerprint=_tensor_private_hash(centers_q, role="residual_q_center"),
        q_radius_fingerprint=_tensor_private_hash(radii_q, role="residual_q_radius"),
        simultaneous_statement_count=statement_count,
        familywise_alpha=registration.familywise_alpha,
        per_statement_two_sided_alpha=float(per_statement_alpha),
        alpha_spent=0.0 if exact_census else registration.familywise_alpha,
        alpha_ledger_hash=registration.alpha_ledger_hash,
        alpha_coordinate_hash=registration.alpha_coordinate_hash,
        alpha_lineage_hash=registration.alpha_lineage_hash,
        exact_residual_census=exact_census,
        production_cuda_execution=False,
        private_cpu_reference_execution=True,
        _issuance_token=_RESIDUAL_RECEIPT_TOKEN,
    )
    handle = _LiveProjectionHandle(
        receipt=receipt,
        tensors={
            "mjz_center": centers_m,
            "mjz_radius": radii_m,
            "q_center": centers_q,
            "q_radius": radii_q,
        },
        issuance_token=_RESIDUAL_LIVE_TOKEN,
    )
    return V1LiteralJResidualPopulationProjectionResult(
        receipt=receipt,
        device_handle=handle,
        _issuance_token=_RESIDUAL_LIVE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJDFixedResidualPopulationProjectionReceipt:
    """Array-free full-population fixed-direction literal-J authority."""

    registration_hash: str
    d_fixed_projection_receipt_hash: str
    residual_projection_receipt_hash: str | None
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_count: int
    direction_ids: tuple[str, ...]
    d_fixed_projection_direction_axis_hash: str
    private_primal_direction_axis_hash: str
    stratum_ids: tuple[str, str]
    population_stratum_counts: tuple[int, int]
    d_fixed_stratum_counts: tuple[int, int]
    residual_population_counts: tuple[int, int]
    residual_observed_counts: tuple[int, int]
    population_count: int
    d_fixed_mass_fraction: tuple[int, int]
    residual_stratum_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    d_fixed_source_replay_axis_hash: str
    d_fixed_shard_replay_receipt_hash: str | None
    affine_shell_shard_replay_receipt_hash: str | None
    d_fixed_registered_prefix_hash: str
    d_fixed_pair_prefix_axis_hash: str
    residual_pair_suffix_axis_hash: str
    residual_sealed_population_order_hashes: tuple[str, str]
    residual_observed_prefix_axis_hashes: tuple[str, str] | None
    alpha_ledger_hash: str
    alpha_coordinate_hash: str
    alpha_lineage_hash: str
    registered_residual_familywise_alpha: float
    alpha_spent: float
    d_fixed_numerical_alpha_spent: float
    exact_full_population_census: bool
    combined_numerical_binding_hash: str
    mjz_sampling_linf_radius_upper_bound: float
    q_sampling_linf_radius_upper_bound: float
    d_fixed_source_bound: bool
    private_unbound_cpu_reference_only: bool
    array_free_receipt: bool = True
    live_numeric_handle_required: bool = True
    full_population_literal_j_projection_authority: bool = True
    kkt_geometry_uncertainty_input_authority: bool = True
    fd_rank_authority: bool = False
    kkt_face_or_solution_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_D_FIXED_RESIDUAL_POPULATION_ALGORITHM
    authority_scope: str = LITERAL_J_D_FIXED_RESIDUAL_POPULATION_AUTHORITY
    schema: str = LITERAL_J_D_FIXED_RESIDUAL_POPULATION_PROJECTION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token not in (_POPULATION_RECEIPT_TOKEN, _REPLAY_TOKEN):
            raise PermissionError("population projection receipt requires typed issuer")
        for name in (
            "registration_hash",
            "d_fixed_projection_receipt_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "d_fixed_projection_direction_axis_hash",
            "private_primal_direction_axis_hash",
            "d_fixed_source_replay_axis_hash",
            "d_fixed_registered_prefix_hash",
            "d_fixed_pair_prefix_axis_hash",
            "residual_pair_suffix_axis_hash",
            "alpha_ledger_hash",
            "alpha_coordinate_hash",
            "alpha_lineage_hash",
            "combined_numerical_binding_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.residual_projection_receipt_hash is not None:
            object.__setattr__(
                self,
                "residual_projection_receipt_hash",
                _identity(
                    self.residual_projection_receipt_hash,
                    name="residual_projection_receipt_hash",
                ),
            )
        if self.d_fixed_shard_replay_receipt_hash is not None:
            object.__setattr__(
                self,
                "d_fixed_shard_replay_receipt_hash",
                _identity(
                    self.d_fixed_shard_replay_receipt_hash,
                    name="d_fixed_shard_replay_receipt_hash",
                ),
            )
        if self.affine_shell_shard_replay_receipt_hash is not None:
            object.__setattr__(
                self,
                "affine_shell_shard_replay_receipt_hash",
                _identity(
                    self.affine_shell_shard_replay_receipt_hash,
                    name="affine_shell_shard_replay_receipt_hash",
                ),
            )
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
            "population_count",
            _positive_int(self.population_count, name="population_count"),
        )
        object.__setattr__(
            self,
            "direction_ids",
            _identity_axis(self.direction_ids, name="direction_ids"),
        )
        object.__setattr__(
            self, "stratum_ids", _identity_pair(self.stratum_ids, name="stratum_ids")
        )
        for name in (
            "population_stratum_counts",
            "d_fixed_stratum_counts",
            "residual_population_counts",
            "residual_observed_counts",
        ):
            object.__setattr__(self, name, _count_pair(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "residual_sealed_population_order_hashes",
            _identity_pair(
                self.residual_sealed_population_order_hashes,
                name="residual_sealed_population_order_hashes",
            ),
        )
        if self.residual_observed_prefix_axis_hashes is not None:
            object.__setattr__(
                self,
                "residual_observed_prefix_axis_hashes",
                _identity_pair(
                    self.residual_observed_prefix_axis_hashes,
                    name="residual_observed_prefix_axis_hashes",
                ),
            )
        object.__setattr__(
            self,
            "d_fixed_mass_fraction",
            _validate_fraction_pair(
                self.d_fixed_mass_fraction, name="d_fixed_mass_fraction"
            ),
        )
        masses = tuple(
            _validate_fraction_pair(item, name="residual_stratum_mass_fraction")
            for item in self.residual_stratum_mass_fractions
        )
        if len(masses) != 2:
            raise ValueError("residual mass axis must have two entries")
        object.__setattr__(self, "residual_stratum_mass_fractions", masses)
        object.__setattr__(
            self,
            "registered_residual_familywise_alpha",
            _finite_alpha(
                self.registered_residual_familywise_alpha,
                name="registered_residual_familywise_alpha",
            ),
        )
        object.__setattr__(
            self,
            "alpha_spent",
            _finite_alpha(self.alpha_spent, name="alpha_spent", allow_zero=True),
        )
        object.__setattr__(
            self,
            "d_fixed_numerical_alpha_spent",
            _finite_alpha(
                self.d_fixed_numerical_alpha_spent,
                name="d_fixed_numerical_alpha_spent",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "mjz_sampling_linf_radius_upper_bound",
            _nonnegative_float(
                self.mjz_sampling_linf_radius_upper_bound,
                name="mjz_sampling_linf_radius_upper_bound",
            ),
        )
        object.__setattr__(
            self,
            "q_sampling_linf_radius_upper_bound",
            _nonnegative_float(
                self.q_sampling_linf_radius_upper_bound,
                name="q_sampling_linf_radius_upper_bound",
            ),
        )
        flags = (
            self.exact_full_population_census,
            self.d_fixed_source_bound,
            self.private_unbound_cpu_reference_only,
            self.array_free_receipt,
            self.live_numeric_handle_required,
            self.full_population_literal_j_projection_authority,
            self.kkt_geometry_uncertainty_input_authority,
            self.fd_rank_authority,
            self.kkt_face_or_solution_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("population projection flags must be exact booleans")
        residual_nonempty = sum(self.residual_population_counts) > 0
        if (
            self.population_count != sum(self.population_stratum_counts)
            or any(
                self.d_fixed_stratum_counts[h] + self.residual_population_counts[h]
                != self.population_stratum_counts[h]
                for h in (0, 1)
            )
            or self.exact_full_population_census
            != (self.residual_observed_counts == self.residual_population_counts)
            or residual_nonempty != (self.residual_projection_receipt_hash is not None)
            or residual_nonempty
            != (self.residual_observed_prefix_axis_hashes is not None)
            or self.d_fixed_source_bound
            != (self.d_fixed_shard_replay_receipt_hash is not None)
            or (
                self.affine_shell_shard_replay_receipt_hash is not None
                and not self.d_fixed_source_bound
            )
            or self.private_unbound_cpu_reference_only == self.d_fixed_source_bound
            or Fraction(*self.d_fixed_mass_fraction)
            + sum((Fraction(*item) for item in masses), Fraction(0))
            != 1
            or (not self.exact_full_population_census)
            and self.alpha_spent != self.registered_residual_familywise_alpha
            or self.exact_full_population_census
            and self.alpha_spent != 0.0
            or self.d_fixed_numerical_alpha_spent != 0.0
            or not self.array_free_receipt
            or not self.live_numeric_handle_required
            or self.full_population_literal_j_projection_authority
            != self.d_fixed_source_bound
            or self.kkt_geometry_uncertainty_input_authority
            != self.d_fixed_source_bound
            or any(
                (
                    self.fd_rank_authority,
                    self.kkt_face_or_solution_authority,
                    self.c_stream_authority,
                    self.calibration_authority,
                    self.oracle_authority,
                    self.commit_authority,
                    self.terminal_authority,
                    self.pareto_authority,
                )
            )
            or self.schema != LITERAL_J_D_FIXED_RESIDUAL_POPULATION_PROJECTION_SCHEMA
            or self.algorithm != LITERAL_J_D_FIXED_RESIDUAL_POPULATION_ALGORITHM
            or self.authority_scope != LITERAL_J_D_FIXED_RESIDUAL_POPULATION_AUTHORITY
        ):
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.AUTHORITY_VIOLATION,
                "full-population projection crossed its measure or authority boundary",
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

    def to_payload(self) -> dict[str, object]:
        return dict(self._payload())

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "V1LiteralJDFixedResidualPopulationProjectionReceipt":
        if not isinstance(payload, Mapping):
            raise TypeError("population projection payload must be a mapping")
        values = dict(payload)
        for name in (
            "direction_ids",
            "stratum_ids",
            "population_stratum_counts",
            "d_fixed_stratum_counts",
            "residual_population_counts",
            "residual_observed_counts",
            "d_fixed_mass_fraction",
            "residual_stratum_mass_fractions",
            "residual_sealed_population_order_hashes",
        ):
            values[name] = tuple(values[name])  # type: ignore[arg-type]
        values["residual_stratum_mass_fractions"] = tuple(
            tuple(item) for item in values["residual_stratum_mass_fractions"]
        )
        if values.get("residual_observed_prefix_axis_hashes") is not None:
            values["residual_observed_prefix_axis_hashes"] = tuple(
                values["residual_observed_prefix_axis_hashes"]
            )
        values.setdefault("d_fixed_shard_replay_receipt_hash", None)
        values.setdefault("affine_shell_shard_replay_receipt_hash", None)
        if "d_fixed_source_bound" not in values:
            values["d_fixed_source_bound"] = (
                values["d_fixed_shard_replay_receipt_hash"] is not None
            )
        source_bound = bool(values["d_fixed_source_bound"])
        values.setdefault("private_unbound_cpu_reference_only", not source_bound)
        values.setdefault(
            "full_population_literal_j_projection_authority", source_bound
        )
        values.setdefault("kkt_geometry_uncertainty_input_authority", source_bound)
        return cls(**values, _issuance_token=_REPLAY_TOKEN)  # type: ignore[arg-type]

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(self._payload(), role="population-receipt")
        if self._receipt_hash is not None and self._receipt_hash != observed:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
                "population projection receipt mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
                "population projection receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


V1LiteralJDFixedResidualPopulationProjectionDeviceHandle = _LiveProjectionHandle


@dataclass(frozen=True, slots=True)
class V1LiteralJDFixedResidualPopulationProjectionResult:
    receipt: V1LiteralJDFixedResidualPopulationProjectionReceipt
    device_handle: _LiveProjectionHandle = field(repr=False, compare=False)
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RESIDUAL_LIVE_TOKEN:
            raise PermissionError("population projection result requires typed issuer")
        if (
            type(self.receipt)
            is not V1LiteralJDFixedResidualPopulationProjectionReceipt
            or type(self.device_handle) is not _LiveProjectionHandle
        ):
            raise TypeError("population projection result components must be exact")
        self.device_handle.assert_live()
        if self.device_handle.receipt.receipt_hash != self.receipt.receipt_hash:
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
                "population receipt and live numerics do not close",
            )


def _weighted_interval_sum(
    *, centers: Sequence[Any], radii: Sequence[Any], masses: Sequence[Fraction]
) -> tuple[Any, Any]:
    if not centers or len(centers) != len(radii) or len(centers) != len(masses):
        raise ValueError("weighted interval family is incomplete")
    shape = tuple(int(item) for item in centers[0].shape)
    device = centers[0].device
    output_center = torch.zeros(shape, dtype=torch.float64, device=device)
    output_radius = torch.zeros_like(output_center)
    arithmetic_mass = torch.zeros_like(output_center)
    for center, radius, exact_mass in zip(centers, radii, masses, strict=True):
        _require_tensor(center, name="component_center", shape=shape, device=device)
        _require_tensor(radius, name="component_radius", shape=shape, device=device)
        if bool((radius < 0.0).any().item()):
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
                "component interval radius is negative",
            )
        mass_center, mass_radius = _fraction_binary64_center_radius(exact_mass)
        contribution_center = center * mass_center
        center_mass_uncertainty = _up(torch.abs(center) * mass_radius)
        source_interval_uncertainty = _up(radius * abs(mass_center))
        mixed_uncertainty = _up(radius * mass_radius)
        multiplication_rounding = _up(
            torch.abs(contribution_center) * (2.0**-52) + _F64_MIN_SUBNORMAL
        )
        contribution_radius = _up(
            _up(
                _up(center_mass_uncertainty + source_interval_uncertainty)
                + mixed_uncertainty
            )
            + multiplication_rounding
        )
        output_center.add_(contribution_center)
        output_radius = _up(output_radius + contribution_radius)
        arithmetic_mass = _up(arithmetic_mass + torch.abs(contribution_center))
    output_radius = _up(
        output_radius
        + _up(arithmetic_mass * (len(centers) + 2) * (2.0**-52) + _F64_MIN_SUBNORMAL)
    )
    output_radius = _up(output_radius)
    if not bool(torch.isfinite(output_center).all().item()) or not bool(
        torch.isfinite(output_radius).all().item()
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
            "population mixture produced nonfinite intervals",
        )
    return output_center.contiguous(), output_radius.contiguous()


def _issue_v1_literal_j_d_fixed_residual_population_projection(
    *,
    registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    d_fixed_execution: V1LiteralJTwoPassDFixedProjectionExecution,
    d_fixed_shard_replay_receipt_hash: str | None,
    d_fixed_source_bound: bool,
    residual_result: V1LiteralJResidualPopulationProjectionResult | None,
    production_cuda_required: bool = True,
    allow_private_cpu_reference: bool = False,
) -> V1LiteralJDFixedResidualPopulationProjectionResult:
    """Promote exact D_fixed plus fresh residual intervals to population geometry."""

    if (
        type(registration)
        is not V1LiteralJDFixedResidualPopulationProjectionRegistration
    ):
        raise TypeError("population registration must be exact")
    if type(d_fixed_execution) is not V1LiteralJTwoPassDFixedProjectionExecution:
        raise TypeError("D_fixed execution must be exact")
    if (
        type(d_fixed_source_bound) is not bool
        or type(production_cuda_required) is not bool
        or type(allow_private_cpu_reference) is not bool
    ):
        raise TypeError("execution policy flags must be exact booleans")
    if d_fixed_source_bound:
        d_fixed_shard_replay_receipt_hash = _identity(
            d_fixed_shard_replay_receipt_hash,
            name="d_fixed_shard_replay_receipt_hash",
        )
        if (
            registration.d_fixed_shard_replay_receipt_hash
            != d_fixed_shard_replay_receipt_hash
        ):
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.POST_SELECTION,
                "the residual design did not pre-bind this D_fixed source replay",
            )
    elif d_fixed_shard_replay_receipt_hash is not None:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.AUTHORITY_VIOLATION,
            "an unbound private reference cannot claim a source-replay receipt",
        )
    elif registration.d_fixed_shard_replay_receipt_hash is not None:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.AUTHORITY_VIOLATION,
            "a source-bound population design cannot enter the unbound CPU oracle",
        )
    registration.assert_integrity()
    d_receipt = d_fixed_execution.receipt
    d_fixed_execution.device_handle.assert_live()
    if (
        d_receipt.registration_hash != registration.d_fixed_registration_hash
        or d_receipt.model_identity_hash != registration.model_identity_hash
        or d_receipt.chart_identity_hash != registration.chart_identity_hash
        or d_receipt.current_mask_hash != registration.current_mask_hash
        or d_receipt.proposal_generation != registration.proposal_generation
        or d_receipt.route_axis_hash != registration.route_axis_hash
        or d_receipt.route_count != registration.route_count
        or d_receipt.direction_ids != registration.direction_ids
        or d_receipt.direction_axis_hash
        != registration.d_fixed_projection_direction_axis_hash
        or d_receipt.source_replay_axis_hash
        != registration.d_fixed_source_replay_axis_hash
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
            "D_fixed live projection left the registered chart",
        )
    residual_nonempty = sum(registration.residual_stratum_counts) > 0
    if residual_nonempty and residual_result is None:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.D_FIXED_ONLY,
            "D_fixed empirical geometry cannot impersonate the full population",
        )
    if (
        residual_result is not None
        and type(residual_result) is not V1LiteralJResidualPopulationProjectionResult
    ):
        raise TypeError("residual result must be exact")
    residual_receipt = None if residual_result is None else residual_result.receipt
    if residual_result is not None:
        residual_result.device_handle.assert_live()
        if (
            residual_receipt.registration_hash != registration.registration_hash
            or residual_receipt.model_identity_hash != registration.model_identity_hash
            or residual_receipt.chart_identity_hash != registration.chart_identity_hash
            or residual_receipt.current_mask_hash != registration.current_mask_hash
            or residual_receipt.proposal_generation != registration.proposal_generation
            or residual_receipt.route_axis_hash != registration.route_axis_hash
            or residual_receipt.route_count != registration.route_count
            or residual_receipt.direction_ids != registration.direction_ids
            or residual_receipt.d_fixed_projection_direction_axis_hash
            != registration.d_fixed_projection_direction_axis_hash
            or residual_receipt.private_primal_direction_axis_hash
            != registration.private_primal_direction_axis_hash
            or residual_receipt.residual_population_counts
            != registration.residual_stratum_counts
            or residual_receipt.residual_sealed_population_order_hashes
            != registration.residual_sealed_population_order_hashes
            or residual_receipt.alpha_ledger_hash != registration.alpha_ledger_hash
            or residual_receipt.alpha_coordinate_hash
            != registration.alpha_coordinate_hash
            or residual_receipt.alpha_lineage_hash != registration.alpha_lineage_hash
            or residual_receipt.familywise_alpha != registration.familywise_alpha
            or (
                not residual_receipt.exact_residual_census
                and residual_receipt.alpha_spent != registration.familywise_alpha
            )
        ):
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
                "residual live projection left its registered population, direction, or alpha lineage",
            )
    with d_fixed_execution.device_handle.borrow() as d_lease:
        device = d_lease.mjz_center.device
        if production_cuda_required and device.type != "cuda":
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.CPU_PRODUCTION_FALLBACK,
                "production population promotion requires CUDA-resident D_fixed values",
            )
        if not production_cuda_required and (
            device.type != "cpu" or not allow_private_cpu_reference
        ):
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.CPU_PRODUCTION_FALLBACK,
                "CPU population promotion is an explicit private reference only",
            )
        mz_centers = [d_lease.mjz_center]
        mz_radii = [d_lease.mjz_radius]
        q_centers = [d_lease.q_center]
        q_radii = [d_lease.q_radius]
        masses = [Fraction(*registration.d_fixed_mass_fraction)]
        if residual_result is not None:
            with residual_result.device_handle.borrow() as r_lease:
                if r_lease.mjz_center.device != device:
                    _refuse(
                        V1LiteralJDFixedResidualPopulationProjectionRefusalCode.DEVICE_REQUIRED,
                        "D_fixed and residual values must share one device",
                    )
                if (
                    production_cuda_required
                    and not residual_receipt.production_cuda_execution
                ):
                    _refuse(
                        V1LiteralJDFixedResidualPopulationProjectionRefusalCode.CPU_PRODUCTION_FALLBACK,
                        "private CPU residual numerics cannot authorize production",
                    )
                for h in (0, 1):
                    if registration.residual_stratum_counts[h] > 0:
                        mz_centers.append(r_lease.mjz_center[h])
                        mz_radii.append(r_lease.mjz_radius[h])
                        q_centers.append(r_lease.q_center[h])
                        q_radii.append(r_lease.q_radius[h])
                        masses.append(
                            Fraction(*registration.residual_stratum_mass_fractions[h])
                        )
                mjz_center, mjz_radius = _weighted_interval_sum(
                    centers=mz_centers, radii=mz_radii, masses=masses
                )
                q_center, q_radius = _weighted_interval_sum(
                    centers=q_centers, radii=q_radii, masses=masses
                )
        else:
            mjz_center, mjz_radius = _weighted_interval_sum(
                centers=mz_centers, radii=mz_radii, masses=masses
            )
            q_center, q_radius = _weighted_interval_sum(
                centers=q_centers, radii=q_radii, masses=masses
            )
    residual_hash = None if residual_receipt is None else residual_receipt.receipt_hash
    combined_binding = _payload_hash(
        {
            "schema": "deterministic_population_interval_binding_v2",
            "registration_hash": registration.registration_hash,
            "d_fixed_projection_receipt_hash": d_receipt.receipt_hash,
            "d_fixed_shard_replay_receipt_hash": d_fixed_shard_replay_receipt_hash,
            "affine_shell_shard_replay_receipt_hash": (
                registration.affine_shell_shard_replay_receipt_hash
            ),
            "d_fixed_source_bound": d_fixed_source_bound,
            "residual_projection_receipt_hash": residual_hash,
            "d_fixed_mass_fraction": registration.d_fixed_mass_fraction,
            "residual_stratum_mass_fractions": registration.residual_stratum_mass_fractions,
            "algorithm": LITERAL_J_D_FIXED_RESIDUAL_POPULATION_ALGORITHM,
        },
        role="combined-numerical-binding",
    )
    observed = (0, 0) if residual_receipt is None else residual_receipt.observed_counts
    exact_census = residual_receipt is None or residual_receipt.exact_residual_census
    alpha_spent = 0.0 if residual_receipt is None else residual_receipt.alpha_spent
    mjz_radius_bound = _tensor_linf_radius_upper_bound(
        mjz_radius, name="full-population M_Jz radius"
    )
    q_radius_bound = _tensor_linf_radius_upper_bound(
        q_radius, name="full-population Q radius"
    )
    receipt = V1LiteralJDFixedResidualPopulationProjectionReceipt(
        registration_hash=registration.registration_hash,
        d_fixed_projection_receipt_hash=d_receipt.receipt_hash,
        residual_projection_receipt_hash=residual_hash,
        model_identity_hash=registration.model_identity_hash,
        chart_identity_hash=registration.chart_identity_hash,
        current_mask_hash=registration.current_mask_hash,
        proposal_generation=registration.proposal_generation,
        route_axis_hash=registration.route_axis_hash,
        route_count=registration.route_count,
        direction_ids=registration.direction_ids,
        d_fixed_projection_direction_axis_hash=(
            registration.d_fixed_projection_direction_axis_hash
        ),
        private_primal_direction_axis_hash=registration.private_primal_direction_axis_hash,
        stratum_ids=registration.stratum_ids,
        population_stratum_counts=registration.population_stratum_counts,
        d_fixed_stratum_counts=registration.d_fixed_stratum_counts,
        residual_population_counts=registration.residual_stratum_counts,
        residual_observed_counts=observed,
        population_count=registration.population_count,
        d_fixed_mass_fraction=registration.d_fixed_mass_fraction,
        residual_stratum_mass_fractions=registration.residual_stratum_mass_fractions,
        d_fixed_source_replay_axis_hash=registration.d_fixed_source_replay_axis_hash,
        d_fixed_shard_replay_receipt_hash=d_fixed_shard_replay_receipt_hash,
        affine_shell_shard_replay_receipt_hash=(
            registration.affine_shell_shard_replay_receipt_hash
        ),
        d_fixed_registered_prefix_hash=registration.d_fixed_registered_prefix_hash,
        d_fixed_pair_prefix_axis_hash=registration.d_fixed_pair_prefix_axis_hash,
        residual_pair_suffix_axis_hash=registration.residual_pair_suffix_axis_hash,
        residual_sealed_population_order_hashes=registration.residual_sealed_population_order_hashes,
        residual_observed_prefix_axis_hashes=(
            None
            if residual_receipt is None
            else residual_receipt.observed_prefix_axis_hashes
        ),
        alpha_ledger_hash=registration.alpha_ledger_hash,
        alpha_coordinate_hash=registration.alpha_coordinate_hash,
        alpha_lineage_hash=registration.alpha_lineage_hash,
        registered_residual_familywise_alpha=registration.familywise_alpha,
        alpha_spent=alpha_spent,
        d_fixed_numerical_alpha_spent=0.0,
        exact_full_population_census=exact_census,
        combined_numerical_binding_hash=combined_binding,
        mjz_sampling_linf_radius_upper_bound=mjz_radius_bound,
        q_sampling_linf_radius_upper_bound=q_radius_bound,
        d_fixed_source_bound=d_fixed_source_bound,
        private_unbound_cpu_reference_only=not d_fixed_source_bound,
        full_population_literal_j_projection_authority=d_fixed_source_bound,
        kkt_geometry_uncertainty_input_authority=d_fixed_source_bound,
        _issuance_token=_POPULATION_RECEIPT_TOKEN,
    )
    handle = _LiveProjectionHandle(
        receipt=receipt,
        tensors={
            "mjz_center": mjz_center,
            "mjz_radius": mjz_radius,
            "q_center": q_center,
            "q_radius": q_radius,
        },
        issuance_token=_RESIDUAL_LIVE_TOKEN,
    )
    return V1LiteralJDFixedResidualPopulationProjectionResult(
        receipt=receipt, device_handle=handle, _issuance_token=_RESIDUAL_LIVE_TOKEN
    )


def issue_v1_literal_j_d_fixed_residual_population_projection(
    *,
    registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    d_fixed_source_result: V1LiteralJTwoPassDFixedShardReplayResult,
    residual_result: V1LiteralJResidualPopulationProjectionResult | None,
    production_cuda_required: bool = True,
    allow_private_cpu_reference: bool = False,
) -> V1LiteralJDFixedResidualPopulationProjectionResult:
    """Issue source-bound population geometry from exact shard replay.

    A bare two-pass projection is intentionally insufficient here: only the
    shard-replay adapter proves that every literal-J tile came from the typed,
    immutable D source joined to the live block-support coefficients.
    """

    from .modern_v1_literal_j_two_pass_d_fixed_shard_replay import (
        V1LiteralJTwoPassDFixedShardReplayResult,
    )

    if type(d_fixed_source_result) is not V1LiteralJTwoPassDFixedShardReplayResult:
        raise TypeError("D_fixed source result must be exact shard-replay custody")
    d_fixed_source_result.receipt.assert_integrity()
    if (
        d_fixed_source_result.receipt.registration_hash
        != registration.d_fixed_registration_hash
        or d_fixed_source_result.receipt.projection_receipt_hash
        != d_fixed_source_result.projection.receipt.receipt_hash
        or registration.affine_shell_shard_replay_receipt_hash is not None
        or (
            registration.d_fixed_shard_replay_receipt_hash is not None
            and registration.d_fixed_shard_replay_receipt_hash
            != d_fixed_source_result.receipt.receipt_hash
        )
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
            "source-bound D_fixed replay left the registered projection",
        )
    return _issue_v1_literal_j_d_fixed_residual_population_projection(
        registration=registration,
        d_fixed_execution=d_fixed_source_result.projection,
        d_fixed_shard_replay_receipt_hash=(d_fixed_source_result.receipt.receipt_hash),
        d_fixed_source_bound=True,
        residual_result=residual_result,
        production_cuda_required=production_cuda_required,
        allow_private_cpu_reference=allow_private_cpu_reference,
    )


def issue_v1_literal_j_affine_shell_d_fixed_residual_population_projection(
    *,
    registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    affine_shell_source_result: "V1LiteralJAffineShellShardReplayResult",
    residual_result: V1LiteralJResidualPopulationProjectionResult | None,
    production_cuda_required: bool = True,
    allow_private_cpu_reference: bool = False,
) -> V1LiteralJDFixedResidualPopulationProjectionResult:
    """Issue population geometry while preserving affine-shell replay identity."""

    from .modern_v1_literal_j_two_pass_d_fixed_shard_replay import (
        V1LiteralJAffineShellShardReplayResult,
    )

    if type(affine_shell_source_result) is not V1LiteralJAffineShellShardReplayResult:
        raise TypeError("affine-shell D_fixed source result must be exact")
    affine_shell_source_result.receipt.assert_integrity()
    source = affine_shell_source_result.source_bound_replay
    if (
        registration.affine_shell_shard_replay_receipt_hash
        != affine_shell_source_result.receipt.receipt_hash
        or registration.d_fixed_shard_replay_receipt_hash
        != source.receipt.receipt_hash
        or affine_shell_source_result.receipt.source_bound_replay_receipt_hash
        != source.receipt.receipt_hash
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
            "affine-shell population registration does not match the supplied replay wrapper",
        )
    return _issue_v1_literal_j_d_fixed_residual_population_projection(
        registration=registration,
        d_fixed_execution=source.projection,
        d_fixed_shard_replay_receipt_hash=source.receipt.receipt_hash,
        d_fixed_source_bound=True,
        residual_result=residual_result,
        production_cuda_required=production_cuda_required,
        allow_private_cpu_reference=allow_private_cpu_reference,
    )


def _issue_v1_literal_j_d_fixed_residual_population_projection_from_v7_resident_d1_impl(
    *,
    registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    resident_d1_result: object,
    residual_result: V1LiteralJResidualPopulationProjectionResult | None,
    production_cuda_required: bool = True,
    allow_private_cpu_reference: bool = False,
) -> V1LiteralJDFixedResidualPopulationProjectionResult:
    """Consume resident exact-D1 geometry and join fresh residual intervals.

    The resident owner stores ``2 M_J z`` because that is the exact gradient
    emitted by the fixed-primal source.  Multiplication by the exact rational
    ``1/2`` recovers ``M_J z`` for the population mixture.  The owner is
    burned after this call on both success and failure; no D1 row replay or
    preferred hard action is involved.
    """

    from .modern_v1_sampled_v7_resident_d1_literal_j_population import (
        V1SampledV7ResidentD1LiteralJPopulationResult,
    )

    if (
        type(registration)
        is not V1LiteralJDFixedResidualPopulationProjectionRegistration
    ):
        raise TypeError("population registration must be exact")
    if type(resident_d1_result) is not V1SampledV7ResidentD1LiteralJPopulationResult:
        raise TypeError("resident D1 result must be exact")
    if (
        type(production_cuda_required) is not bool
        or type(allow_private_cpu_reference) is not bool
    ):
        raise TypeError("execution policy flags must be exact booleans")
    if resident_d1_result.device_handle.active_lease_count:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.ACTIVE_LEASE,
            "resident D1 geometry has an external active lease",
        )
    registration.assert_integrity()
    resident_d1_result.device_handle.assert_live_identity()
    d_receipt = resident_d1_result.receipt
    d_receipt.assert_integrity()
    expected_direction_axis_hash = _resident_v7_d1_direction_axis_hash(d_receipt)
    if (
        registration.d_fixed_registration_hash != d_receipt.receipt_hash
        or registration.d_fixed_shard_replay_receipt_hash != d_receipt.receipt_hash
        or registration.model_identity_hash != d_receipt.model_identity_hash
        or registration.chart_identity_hash != d_receipt.chart_identity_hash
        or registration.current_mask_hash != d_receipt.current_mask_hash
        or registration.proposal_generation != d_receipt.proposal_generation
        or registration.route_axis_hash != d_receipt.route_axis_hash
        or registration.route_id_axis_hash != d_receipt.route_id_axis_hash
        or registration.route_count != d_receipt.route_count
        or registration.direction_ids != d_receipt.direction_ids
        or registration.direction_source_hash != d_receipt.private_primal_source_hash
        or registration.d_fixed_projection_direction_axis_hash
        != expected_direction_axis_hash
        or registration.private_primal_direction_axis_hash
        != d_receipt.private_primal_descriptor_axis_hash
        or registration.d_fixed_source_replay_axis_hash
        != d_receipt.exact_census_source_binding_hash
        or registration.d_fixed_registered_prefix_hash
        != d_receipt.exact_census_source_plan_hash
        or registration.d_fixed_stratum_counts != d_receipt.d1_stratum_counts
    ):
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
            "resident D1 geometry left the registered chart or direction axis",
        )
    residual_nonempty = sum(registration.residual_stratum_counts) > 0
    if residual_nonempty and residual_result is None:
        _refuse(
            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.D_FIXED_ONLY,
            "resident D1 geometry cannot impersonate the full train population",
        )
    if residual_result is not None and type(residual_result) is not V1LiteralJResidualPopulationProjectionResult:
        raise TypeError("residual result must be exact")
    residual_receipt = None if residual_result is None else residual_result.receipt
    if residual_result is not None:
        residual_result.device_handle.assert_live()
        if (
            residual_receipt.registration_hash != registration.registration_hash
            or residual_receipt.model_identity_hash != registration.model_identity_hash
            or residual_receipt.chart_identity_hash != registration.chart_identity_hash
            or residual_receipt.current_mask_hash != registration.current_mask_hash
            or residual_receipt.proposal_generation != registration.proposal_generation
            or residual_receipt.route_axis_hash != registration.route_axis_hash
            or residual_receipt.route_count != registration.route_count
            or residual_receipt.direction_ids != registration.direction_ids
            or residual_receipt.d_fixed_projection_direction_axis_hash
            != registration.d_fixed_projection_direction_axis_hash
            or residual_receipt.private_primal_direction_axis_hash
            != registration.private_primal_direction_axis_hash
            or residual_receipt.residual_population_counts
            != registration.residual_stratum_counts
            or residual_receipt.residual_sealed_population_order_hashes
            != registration.residual_sealed_population_order_hashes
            or residual_receipt.alpha_ledger_hash != registration.alpha_ledger_hash
            or residual_receipt.alpha_coordinate_hash
            != registration.alpha_coordinate_hash
            or residual_receipt.alpha_lineage_hash != registration.alpha_lineage_hash
            or residual_receipt.familywise_alpha != registration.familywise_alpha
            or (
                not residual_receipt.exact_residual_census
                and residual_receipt.alpha_spent != registration.familywise_alpha
            )
        ):
            _refuse(
                V1LiteralJDFixedResidualPopulationProjectionRefusalCode.STALE_IDENTITY,
                "residual projection left its registered resident-D1 population design",
            )
    output_result: V1LiteralJDFixedResidualPopulationProjectionResult | None = None
    try:
        with resident_d1_result.device_handle.borrow() as d_lease:
            device = d_lease.two_mjz_center.device
            if production_cuda_required and device.type != "cuda":
                _refuse(
                    V1LiteralJDFixedResidualPopulationProjectionRefusalCode.CPU_PRODUCTION_FALLBACK,
                    "production population promotion requires CUDA-resident D1 values",
                )
            if not production_cuda_required and (
                device.type != "cpu" or not allow_private_cpu_reference
            ):
                _refuse(
                    V1LiteralJDFixedResidualPopulationProjectionRefusalCode.CPU_PRODUCTION_FALLBACK,
                    "CPU population promotion is an explicit private reference only",
                )
            d_mjz_center = (0.5 * d_lease.two_mjz_center).contiguous()
            d_mjz_radius = _up(
                0.5 * d_lease.two_mjz_radius + _F64_MIN_SUBNORMAL
            ).contiguous()
            d_q_center = d_lease.q_center
            d_q_radius = d_lease.q_radius
            if (
                bool((d_mjz_radius < 0.0).any().item())
                or bool((d_q_radius < 0.0).any().item())
                or not bool(torch.isfinite(d_mjz_center).all().item())
                or not bool(torch.isfinite(d_mjz_radius).all().item())
                or not bool(torch.isfinite(d_q_center).all().item())
                or not bool(torch.isfinite(d_q_radius).all().item())
            ):
                _refuse(
                    V1LiteralJDFixedResidualPopulationProjectionRefusalCode.NUMERICAL_REFUSAL,
                    "resident D1 intervals are nonfinite or have negative radius",
                )
            mz_centers = [d_mjz_center]
            mz_radii = [d_mjz_radius]
            q_centers = [d_q_center]
            q_radii = [d_q_radius]
            masses = [Fraction(*registration.d_fixed_mass_fraction)]
            if residual_result is not None:
                with residual_result.device_handle.borrow() as r_lease:
                    if r_lease.mjz_center.device != device:
                        _refuse(
                            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.DEVICE_REQUIRED,
                            "resident D1 and residual values must share one device",
                        )
                    if production_cuda_required and not residual_receipt.production_cuda_execution:
                        _refuse(
                            V1LiteralJDFixedResidualPopulationProjectionRefusalCode.CPU_PRODUCTION_FALLBACK,
                            "private CPU residual numerics cannot authorize production",
                        )
                    for h in (0, 1):
                        if registration.residual_stratum_counts[h] > 0:
                            mz_centers.append(r_lease.mjz_center[h])
                            mz_radii.append(r_lease.mjz_radius[h])
                            q_centers.append(r_lease.q_center[h])
                            q_radii.append(r_lease.q_radius[h])
                            masses.append(
                                Fraction(
                                    *registration.residual_stratum_mass_fractions[h]
                                )
                            )
                    mjz_center, mjz_radius = _weighted_interval_sum(
                        centers=mz_centers, radii=mz_radii, masses=masses
                    )
                    q_center, q_radius = _weighted_interval_sum(
                        centers=q_centers, radii=q_radii, masses=masses
                    )
            else:
                mjz_center, mjz_radius = _weighted_interval_sum(
                    centers=mz_centers, radii=mz_radii, masses=masses
                )
                q_center, q_radius = _weighted_interval_sum(
                    centers=q_centers, radii=q_radii, masses=masses
                )
        residual_hash = (
            None if residual_receipt is None else residual_receipt.receipt_hash
        )
        combined_binding = _payload_hash(
            {
                "schema": "resident_v7_D1_population_interval_binding_v1",
                    "registration_hash": registration.registration_hash,
                    "resident_d1_receipt_hash": d_receipt.receipt_hash,
                    "affine_shell_shard_replay_receipt_hash": (
                        registration.affine_shell_shard_replay_receipt_hash
                    ),
                    "residual_projection_receipt_hash": residual_hash,
                "d_fixed_mass_fraction": registration.d_fixed_mass_fraction,
                "residual_stratum_mass_fractions": (
                    registration.residual_stratum_mass_fractions
                ),
                "resident_source_consumed_without_D1_replay": True,
                "preferred_hard_action_present": False,
                "algorithm": LITERAL_J_D_FIXED_RESIDUAL_POPULATION_ALGORITHM,
            },
            role="resident-v7-combined-numerical-binding",
        )
        observed = (
            (0, 0) if residual_receipt is None else residual_receipt.observed_counts
        )
        exact_census = (
            residual_receipt is None or residual_receipt.exact_residual_census
        )
        alpha_spent = (
            0.0 if residual_receipt is None else residual_receipt.alpha_spent
        )
        receipt = V1LiteralJDFixedResidualPopulationProjectionReceipt(
            registration_hash=registration.registration_hash,
            d_fixed_projection_receipt_hash=d_receipt.receipt_hash,
            residual_projection_receipt_hash=residual_hash,
            model_identity_hash=registration.model_identity_hash,
            chart_identity_hash=registration.chart_identity_hash,
            current_mask_hash=registration.current_mask_hash,
            proposal_generation=registration.proposal_generation,
            route_axis_hash=registration.route_axis_hash,
            route_count=registration.route_count,
            direction_ids=registration.direction_ids,
            d_fixed_projection_direction_axis_hash=(
                registration.d_fixed_projection_direction_axis_hash
            ),
            private_primal_direction_axis_hash=(
                registration.private_primal_direction_axis_hash
            ),
            stratum_ids=registration.stratum_ids,
            population_stratum_counts=registration.population_stratum_counts,
            d_fixed_stratum_counts=registration.d_fixed_stratum_counts,
            residual_population_counts=registration.residual_stratum_counts,
            residual_observed_counts=observed,
            population_count=registration.population_count,
            d_fixed_mass_fraction=registration.d_fixed_mass_fraction,
            residual_stratum_mass_fractions=(
                registration.residual_stratum_mass_fractions
            ),
                d_fixed_source_replay_axis_hash=(
                    registration.d_fixed_source_replay_axis_hash
                ),
                d_fixed_shard_replay_receipt_hash=d_receipt.receipt_hash,
                affine_shell_shard_replay_receipt_hash=(
                    registration.affine_shell_shard_replay_receipt_hash
                ),
                d_fixed_registered_prefix_hash=registration.d_fixed_registered_prefix_hash,
            d_fixed_pair_prefix_axis_hash=registration.d_fixed_pair_prefix_axis_hash,
            residual_pair_suffix_axis_hash=registration.residual_pair_suffix_axis_hash,
            residual_sealed_population_order_hashes=(
                registration.residual_sealed_population_order_hashes
            ),
            residual_observed_prefix_axis_hashes=(
                None
                if residual_receipt is None
                else residual_receipt.observed_prefix_axis_hashes
            ),
            alpha_ledger_hash=registration.alpha_ledger_hash,
            alpha_coordinate_hash=registration.alpha_coordinate_hash,
            alpha_lineage_hash=registration.alpha_lineage_hash,
            registered_residual_familywise_alpha=registration.familywise_alpha,
            alpha_spent=alpha_spent,
            d_fixed_numerical_alpha_spent=0.0,
            exact_full_population_census=exact_census,
            combined_numerical_binding_hash=combined_binding,
            mjz_sampling_linf_radius_upper_bound=_tensor_linf_radius_upper_bound(
                mjz_radius, name="full-population M_Jz radius"
            ),
            q_sampling_linf_radius_upper_bound=_tensor_linf_radius_upper_bound(
                q_radius, name="full-population Q radius"
            ),
            d_fixed_source_bound=True,
            private_unbound_cpu_reference_only=False,
            full_population_literal_j_projection_authority=True,
            kkt_geometry_uncertainty_input_authority=True,
            _issuance_token=_POPULATION_RECEIPT_TOKEN,
        )
        handle = _LiveProjectionHandle(
            receipt=receipt,
            tensors={
                "mjz_center": mjz_center,
                "mjz_radius": mjz_radius,
                "q_center": q_center,
                "q_radius": q_radius,
            },
            issuance_token=_RESIDUAL_LIVE_TOKEN,
        )
        output_result = V1LiteralJDFixedResidualPopulationProjectionResult(
            receipt=receipt,
            device_handle=handle,
            _issuance_token=_RESIDUAL_LIVE_TOKEN,
        )
        return output_result
    except BaseException:
        if output_result is not None and not output_result.device_handle.is_evicted:
            output_result.device_handle.evict()
        raise


def _issue_v1_literal_j_d_fixed_residual_population_projection_from_v7_resident_d1(
    *,
    registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    resident_d1_result: object,
    residual_result: V1LiteralJResidualPopulationProjectionResult | None,
    production_cuda_required: bool = True,
    allow_private_cpu_reference: bool = False,
) -> V1LiteralJDFixedResidualPopulationProjectionResult:
    """Fail-closed consuming wrapper for the resident-D1 transition."""

    from .modern_v1_sampled_v7_resident_d1_literal_j_population import (
        V1SampledV7ResidentD1LiteralJPopulationResult,
    )

    if type(resident_d1_result) is not V1SampledV7ResidentD1LiteralJPopulationResult:
        raise TypeError("resident D1 result must be exact")
    try:
        result = _issue_v1_literal_j_d_fixed_residual_population_projection_from_v7_resident_d1_impl(
            registration=registration,
            resident_d1_result=resident_d1_result,
            residual_result=residual_result,
            production_cuda_required=production_cuda_required,
            allow_private_cpu_reference=allow_private_cpu_reference,
        )
    except BaseException as primary_error:
        cleanup_error: BaseException | None = None
        if not resident_d1_result.device_handle.is_evicted:
            try:
                resident_d1_result.device_handle.evict()
            except BaseException as error:
                cleanup_error = error
        if cleanup_error is not None:
            raise BaseExceptionGroup(
                "resident-D1 population transition and cleanup both failed",
                [primary_error, cleanup_error],
            )
        raise
    if not resident_d1_result.device_handle.is_evicted:
        resident_d1_result.device_handle.evict()
    return result


def _issue_private_cpu_reference_d_fixed_residual_population_projection(
    *,
    registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    d_fixed_execution: V1LiteralJTwoPassDFixedProjectionExecution,
    residual_result: V1LiteralJResidualPopulationProjectionResult | None,
) -> V1LiteralJDFixedResidualPopulationProjectionResult:
    """Small unbound numerical oracle with no KKT-geometry authority."""

    return _issue_v1_literal_j_d_fixed_residual_population_projection(
        registration=registration,
        d_fixed_execution=d_fixed_execution,
        d_fixed_shard_replay_receipt_hash=None,
        d_fixed_source_bound=False,
        residual_result=residual_result,
        production_cuda_required=False,
        allow_private_cpu_reference=True,
    )


__all__ = [
    "LITERAL_J_D_FIXED_RESIDUAL_POPULATION_ALGORITHM",
    "LITERAL_J_D_FIXED_RESIDUAL_POPULATION_AUTHORITY",
    "LITERAL_J_D_FIXED_RESIDUAL_POPULATION_PROJECTION_SCHEMA",
    "LITERAL_J_D_FIXED_RESIDUAL_POPULATION_REGISTRATION_SCHEMA",
    "LITERAL_J_JOINT_AXIS_RESIDUAL_POPULATION_MEASURE_AUTHORITY",
    "LITERAL_J_JOINT_AXIS_RESIDUAL_POPULATION_MEASURE_REGISTRATION_SCHEMA",
    "LITERAL_J_RESIDUAL_POPULATION_PROJECTION_SCHEMA",
    "V1LiteralJDFixedResidualPopulationProjectionDeviceHandle",
    "V1LiteralJDFixedResidualPopulationProjectionReceipt",
    "V1LiteralJDFixedResidualPopulationProjectionRefusal",
    "V1LiteralJDFixedResidualPopulationProjectionRefusalCode",
    "V1LiteralJDFixedResidualPopulationProjectionRegistration",
    "V1LiteralJDFixedResidualPopulationProjectionResult",
    "V1LiteralJJointAxisResidualPopulationMeasureRegistration",
    "V1LiteralJJointAxisResidualPopulationMeasureRegistrationReceipt",
    "V1LiteralJResidualPopulationProjectionDeviceHandle",
    "V1LiteralJResidualPopulationProjectionReceipt",
    "V1LiteralJResidualPopulationProjectionResult",
    "issue_v1_literal_j_affine_shell_d_fixed_residual_population_projection",
    "issue_v1_literal_j_d_fixed_residual_population_projection",
    "register_v1_literal_j_affine_shell_d_fixed_residual_population_projection",
    "register_v1_literal_j_d_fixed_residual_population_projection",
    "register_v1_literal_j_joint_affine_shell_d_fixed_residual_population_projection",
    "register_v1_literal_j_joint_axis_residual_population_measure",
]
