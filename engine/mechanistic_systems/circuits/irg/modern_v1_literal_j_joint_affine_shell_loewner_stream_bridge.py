"""Joint no-double-spend Loewner bridge for streamed literal-J geometry.

The canonical residual family is registered once on the action-complete axis

``(z_outside^1, ..., z_outside^A, e_b1, ..., e_bB)``.

The exact ``D_fixed`` census is currently source-replayed once per physical
action on ``(z_outside^a, e_b1, ..., e_bB)``.  This bridge verifies that those
replays are the ordered post-source members of one joint family, intersects
their redundant shared-basis intervals, consumes the joint residual Mv stream
exactly once, and mixes the three population pieces with the exact rational
masses frozen by the population registration.

The issuer-derived joint population source axis is retained separately from
the common member replay axis; neither is accepted as an alias for the other.

Only shell-coordinate Mv columns and compact Q values survive the stream.  The
durable receipt is array-free; the live handle owns only ``A x (B+1) x (B+1)``
small matrices.  It allocates no alpha and grants no solver, D-face, C, K, O,
commit, performance-promotion, terminal, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from fractions import Fraction
import hashlib
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_literal_j_d_fixed_residual_population_projection import (
    V1LiteralJDFixedResidualPopulationProjectionRegistration,
    V1LiteralJJointAxisResidualPopulationMeasureRegistration,
    V1LiteralJJointAxisResidualPopulationMeasureRegistrationReceipt,
)
from .modern_v1_literal_j_joint_affine_shell_family_cuda import (
    V1LiteralJJointAffineShellFamilyDeviceHandle,
    V1LiteralJJointAffineShellPreoutcomeRegistration,
)
from .modern_v1_literal_j_joint_affine_shell_d1_population_stream_cuda import (
    V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle,
)
from .modern_v1_literal_j_joint_affine_shell_residual_population_stream_cuda import (
    V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
    V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMReceipt,
)
from .modern_v1_literal_j_joint_population_source import (
    V1LiteralJJointPopulationSource,
)
from .modern_v1_literal_j_joint_population_action_axis import (
    V1LiteralJJointPopulationActionAxisSource,
)
from .modern_v1_literal_j_two_pass_d_fixed_shard_replay import (
    V1LiteralJAffineShellShardReplayResult,
)

try:  # pragma: no cover - runtime-image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


LITERAL_J_JOINT_AFFINE_SHELL_LOEWNER_STREAM_BRIDGE_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_loewner_stream_bridge_v1"
)
LITERAL_J_JOINT_AFFINE_SHELL_LOEWNER_STREAM_BRIDGE_ALGORITHM: Final = (
    "ordered_per_action_exact_D_fixed_plus_one_joint_residual_stream_exact_rational_"
    "population_mix_symmetric_diagonal_intersection_row_sum_loewner_v1"
)
LITERAL_J_JOINT_AFFINE_SHELL_LOEWNER_STREAM_BRIDGE_AUTHORITY: Final = (
    "source_bound_population_loewner_and_kkt_geometry_uncertainty_input_only_no_"
    "new_alpha_no_solver_D_face_C_K_O_commit_performance_terminal_or_pareto_"
    "authority_v1"
)
LITERAL_J_JOINT_AFFINE_SHELL_PASS1_LOEWNER_BRIDGE_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_pass1_loewner_bridge_v1"
)
LITERAL_J_JOINT_AFFINE_SHELL_PASS1_LOEWNER_BRIDGE_ALGORITHM: Final = (
    "ordered_exact_D_fixed_plus_source_owned_pass1_shell_M_exact_rational_"
    "population_mix_symmetric_diagonal_intersection_row_sum_loewner_v1"
)
LITERAL_J_JOINT_AFFINE_SHELL_PASS1_LOEWNER_BRIDGE_AUTHORITY: Final = (
    "source_bound_pass1_population_loewner_and_kkt_geometry_input_only_pass2_"
    "unopened_no_new_alpha_no_solver_D_face_C_K_O_commit_performance_terminal_"
    "or_pareto_authority_v1"
)
LITERAL_J_JOINT_AFFINE_SHELL_D1_PASS1_LOEWNER_BRIDGE_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_D1_pass1_loewner_bridge_v2"
)
LITERAL_J_JOINT_AFFINE_SHELL_D1_PASS1_LOEWNER_BRIDGE_ALGORITHM: Final = (
    "one_joint_D1_compact_plus_two_residual_compacts_exact_rational_population_"
    "mix_exact_population_measure_ancestry_symmetric_diagonal_intersection_"
    "row_sum_loewner_no_PSD_projection_v2"
)
LITERAL_J_JOINT_AFFINE_SHELL_D1_PASS1_LOEWNER_BRIDGE_AUTHORITY: Final = (
    "action_complete_source_bound_population_loewner_and_KKT_geometry_"
    "uncertainty_input_only_private_atomic_tuple_reference_nonauthoritative_"
    "exact_measure_ancestry_required_in_production_no_new_alpha_solver_D_face_"
    "C_K_O_commit_terminal_or_pareto_authority_v2"
)

LITERAL_J_JOINT_AFFINE_SHELL_FACTORED_LOEWNER_BRIDGE_SCHEMA: Final = (
    "partition_free_v1_literal_J_joint_affine_shell_factored_population_loewner_bridge_v2"
)
LITERAL_J_JOINT_AFFINE_SHELL_FACTORED_LOEWNER_BRIDGE_ALGORITHM: Final = (
    "factored_row_bilinear_forms_D1_plus_two_residual_strata_exact_rational_population_"
    "mix_symmetric_diagonal_intersection_row_sum_loewner_no_PSD_projection_no_rows_v2"
)
LITERAL_J_JOINT_AFFINE_SHELL_FACTORED_LOEWNER_BRIDGE_AUTHORITY: Final = (
    "row_free_population_loewner_and_KKT_geometry_uncertainty_input_only_when_alpha_lineage_"
    "bound_and_production_cuda_private_reference_nonauthoritative_no_new_alpha_solver_D_face_"
    "C_K_O_commit_terminal_or_pareto_authority_v2"
)
_PREFIX: Final = "partition-free-v1-literal-j-joint-affine-shell-loewner-stream-"
_RECEIPT_TOKEN: Final = object()
_PASS1_RECEIPT_TOKEN: Final = object()
_D1_PASS1_RECEIPT_TOKEN: Final = object()
_FACTORED_RECEIPT_TOKEN: Final = object()
_HANDLE_TOKEN: Final = object()
_F64_BYTES: Final = 8
_F64_MIN_SUBNORMAL: Final = float.fromhex("0x0.0000000000001p-1022")
_CPU_REFERENCE_ROUTE_LIMIT: Final = 4096


class V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_IDENTITY = "stale_identity"
    JOINT_AXIS_MISMATCH = "joint_axis_mismatch"
    D_FIXED_AXIS_MISMATCH = "d_fixed_axis_mismatch"
    MASS_COUNT_MISMATCH = "mass_count_mismatch"
    SOURCE_MISMATCH = "source_mismatch"
    ALPHA_MISMATCH = "alpha_mismatch"
    ORDER_COVERAGE_MISMATCH = "order_coverage_mismatch"
    DEVICE_MISMATCH = "device_mismatch"
    CPU_PRIVATE_REFERENCE_FORBIDDEN = "cpu_private_reference_forbidden"
    NUMERICAL_REFUSAL = "numerical_refusal"
    EMPTY_INTERSECTION = "empty_intersection"
    MUTATED_CUSTODY = "mutated_custody"
    ACTIVE_LEASE = "active_lease"
    EVICTED = "evicted"
    ACTION_COMPLETE_SOURCE_BINDING_REQUIRED = (
        "action_complete_source_binding_required"
    )
    AUTHORITY_VIOLATION = "authority_violation"
    ALPHA_LINEAGE_ABSENT = "alpha_lineage_absent"  # never defaulted
    POPULATION_ENCLOSURE_ABSENT = "population_enclosure_absent"


class V1LiteralJJointAffineShellLoewnerStreamBridgeRefusal(RuntimeError):
    """Typed fail-closed refusal for the joint Loewner stream bridge."""

    def __init__(
        self,
        code: V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode:
            raise TypeError("joint Loewner refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LiteralJJointAffineShellLoewnerStreamBridgeRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _identity_axis(value: object, *, name: str, unique: bool = True) -> tuple[str, ...]:
    if type(value) is not tuple:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable tuple",
        )
    result = tuple(_identity(item, name=f"{name} entry") for item in value)
    if not result or (unique and len(result) != len(set(result))):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be nonempty" + (" and unique" if unique else ""),
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _fraction_pair(value: object, *, name: str) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or value[0] < 0
        or value[1] <= 0
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MASS_COUNT_MISMATCH,
            f"{name} must be one exact nonnegative rational pair",
        )
    reduced = Fraction(value[0], value[1])
    return reduced.numerator, reduced.denominator


def _finite_nonnegative(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be one finite nonnegative binary64 scalar",
        )
    return value


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    return (
        id(value),
        int(value.data_ptr()),
        int(value._version),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        str(value.dtype),
        str(value.device),
        bool(value.requires_grad),
    )


def _require_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    device: object | None = None,
    nonnegative: bool = False,
) -> Any:
    if (
        torch is None
        or type(value) is not torch.Tensor
        or value.dtype != torch.float64
        or tuple(int(item) for item in value.shape) != shape
        or not value.is_contiguous()
        or bool(value.requires_grad)
        or not bool(torch.isfinite(value).all().item())
        or (device is not None and value.device != device)
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.INVALID_TYPE,
            f"{name} must be one finite contiguous detached FP64 tensor with shape {shape}",
        )
    if nonnegative and bool((value < 0.0).any().item()):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.NUMERICAL_REFUSAL,
            f"{name} contains a negative radius",
        )
    return value


def _down(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, -math.inf))


def _up(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _bounds(center: Any, radius: Any) -> tuple[Any, Any]:
    radius = torch.clamp(radius, min=0.0)
    return _down(center - radius), _up(center + radius)


def _center_radius(lower: Any, upper: Any) -> tuple[Any, Any]:
    if bool((lower > upper).any().item()):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.EMPTY_INTERSECTION,
            "an interval intersection is empty",
        )
    # Half-before-add avoids overflow for finite same-sign endpoints.
    center = lower * 0.5 + upper * 0.5
    radius = torch.maximum(_up(center - lower), _up(upper - center))
    radius = _up(torch.clamp(radius, min=0.0) + _F64_MIN_SUBNORMAL)
    if not bool(torch.isfinite(center).all().item()) or not bool(
        torch.isfinite(radius).all().item()
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.NUMERICAL_REFUSAL,
            "finite interval conversion overflowed binary64",
        )
    return center.contiguous(), radius.contiguous()


def _intersect(
    first_lower: Any,
    first_upper: Any,
    second_lower: Any,
    second_upper: Any,
) -> tuple[Any, Any]:
    lower = torch.maximum(first_lower, second_lower)
    upper = torch.minimum(first_upper, second_upper)
    if bool((lower > upper).any().item()):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.EMPTY_INTERSECTION,
            "simultaneously valid literal-J orientations have empty intersection",
        )
    return lower, upper


def _weight_bounds(weight: Fraction) -> tuple[float, float]:
    if weight < 0:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MASS_COUNT_MISMATCH,
            "population mixture weights must be nonnegative",
        )
    if weight == 0:
        return 0.0, 0.0
    center = float(weight)
    if not math.isfinite(center):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.NUMERICAL_REFUSAL,
            "a rational population mass overflows binary64",
        )
    lower = math.nextafter(center, -math.inf)
    upper = math.nextafter(center, math.inf)
    exact = Fraction(weight.numerator, weight.denominator)
    if not (Fraction.from_float(lower) <= exact <= Fraction.from_float(upper)):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.NUMERICAL_REFUSAL,
            "binary64 rational-mass enclosure failed",
        )
    return lower, upper


def _scale_bounds(lower: Any, upper: Any, weight: Fraction) -> tuple[Any, Any]:
    if weight == 0:
        zero = torch.zeros_like(lower)
        return zero, zero.clone()
    weight_lower, weight_upper = _weight_bounds(weight)
    candidates = torch.stack(
        (
            lower * weight_lower,
            lower * weight_upper,
            upper * weight_lower,
            upper * weight_upper,
        ),
        dim=0,
    )
    result_lower = _down(torch.amin(candidates, dim=0))
    result_upper = _up(torch.amax(candidates, dim=0))
    if not bool(torch.isfinite(result_lower).all().item()) or not bool(
        torch.isfinite(result_upper).all().item()
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.NUMERICAL_REFUSAL,
            "population mass scaling overflowed binary64",
        )
    return result_lower, result_upper


def _mix_population_bounds(
    components: tuple[tuple[Any, Any, Fraction], ...],
) -> tuple[Any, Any]:
    if not components:
        raise ValueError("population interval mixture requires components")
    lower, upper = _scale_bounds(*components[0])
    for next_lower, next_upper, weight in components[1:]:
        scaled_lower, scaled_upper = _scale_bounds(next_lower, next_upper, weight)
        lower = _down(lower + scaled_lower)
        upper = _up(upper + scaled_upper)
    if not bool(torch.isfinite(lower).all().item()) or not bool(
        torch.isfinite(upper).all().item()
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.NUMERICAL_REFUSAL,
            "population interval addition overflowed binary64",
        )
    return lower.contiguous(), upper.contiguous()


def _extract_component_bounds(
    *,
    shell_m_center: Any,
    shell_m_radius: Any,
    q_center: Any,
    q_radius: Any,
    action_count: int,
    shell_size: int,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Extract shared G and per-action h/c from one population component."""

    direction_count = action_count + shell_size
    shell_m_center = _require_tensor(
        shell_m_center,
        name="shell Mv center",
        shape=(direction_count, shell_size),
    )
    device = shell_m_center.device
    shell_m_radius = _require_tensor(
        shell_m_radius,
        name="shell Mv radius",
        shape=(direction_count, shell_size),
        device=device,
        nonnegative=True,
    )
    q_center = _require_tensor(
        q_center,
        name="compact Q center",
        shape=(direction_count,),
        device=device,
    )
    q_radius = _require_tensor(
        q_radius,
        name="compact Q radius",
        shape=(direction_count,),
        device=device,
        nonnegative=True,
    )
    m_lower, m_upper = _bounds(shell_m_center, shell_m_radius)
    q_lower, q_upper = _bounds(q_center, q_radius)
    g_lower = torch.empty((shell_size, shell_size), dtype=torch.float64, device=device)
    g_upper = torch.empty_like(g_lower)
    basis = action_count
    for i in range(shell_size):
        for j in range(i, shell_size):
            lower = m_lower[basis + j, i]
            upper = m_upper[basis + j, i]
            if i != j:
                lower, upper = _intersect(
                    lower,
                    upper,
                    m_lower[basis + i, j],
                    m_upper[basis + i, j],
                )
            else:
                lower, upper = _intersect(
                    lower,
                    upper,
                    q_lower[basis + i],
                    q_upper[basis + i],
                )
            g_lower[i, j] = lower
            g_lower[j, i] = lower
            g_upper[i, j] = upper
            g_upper[j, i] = upper
    h_lower = m_lower[:action_count].contiguous()
    h_upper = m_upper[:action_count].contiguous()
    c_lower = q_lower[:action_count].contiguous()
    c_upper = q_upper[:action_count].contiguous()
    return (
        g_lower.contiguous(),
        g_upper.contiguous(),
        h_lower,
        h_upper,
        c_lower,
        c_upper,
    )


def _loewner_from_bounds(
    *,
    g_lower: Any,
    g_upper: Any,
    h_lower: Any,
    h_upper: Any,
    c_lower: Any,
    c_upper: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    action_count, shell_size = tuple(int(item) for item in h_lower.shape)
    dimension = shell_size + 1
    device = h_lower.device
    lower = torch.empty(
        (action_count, dimension, dimension), dtype=torch.float64, device=device
    )
    upper = torch.empty_like(lower)
    lower[:, :shell_size, :shell_size] = g_lower
    upper[:, :shell_size, :shell_size] = g_upper
    lower[:, :shell_size, -1] = h_lower
    lower[:, -1, :shell_size] = h_lower
    upper[:, :shell_size, -1] = h_upper
    upper[:, -1, :shell_size] = h_upper
    lower[:, -1, -1] = c_lower
    upper[:, -1, -1] = c_upper
    center, radius = _center_radius(lower, upper)
    row_sum = torch.zeros(
        (action_count, dimension), dtype=torch.float64, device=device
    )
    for column in range(dimension):
        row_sum = _up(row_sum + radius[:, :, column])
    row_sum = _up(row_sum + _F64_MIN_SUBNORMAL).contiguous()
    loewner_lower = center.clone()
    loewner_upper = center.clone()
    diagonal = torch.arange(dimension, dtype=torch.int64, device=device)
    loewner_lower[:, diagonal, diagonal] = _down(
        center[:, diagonal, diagonal] - row_sum
    )
    loewner_upper[:, diagonal, diagonal] = _up(
        center[:, diagonal, diagonal] + row_sum
    )
    if not bool(torch.isfinite(loewner_lower).all().item()) or not bool(
        torch.isfinite(loewner_upper).all().item()
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.NUMERICAL_REFUSAL,
            "row-sum Loewner construction overflowed binary64",
        )
    return (
        center.contiguous(),
        radius.contiguous(),
        loewner_lower.contiguous(),
        loewner_upper.contiguous(),
        row_sum,
    )


def _tensor_digest(value: Any, *, role: str) -> str:
    # Every tensor accepted here is bounded by A*(B+1)^2, never route scale.
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(role.encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(repr(tuple(int(item) for item in array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return content_hash(
        {
            "schema": "joint_small_loewner_tensor_digest_v1",
            "role": role,
            "shape": tuple(int(item) for item in array.shape),
            "payload_sha256": digest.hexdigest(),
        },
        prefix=_PREFIX + "tensor-",
    )


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellLoewnerStreamReceipt:
    joint_preoutcome_registration_hash: str
    joint_family_receipt_hash: str
    joint_postsource_custody_axis_hash: str
    population_projection_registration_hash: str
    residual_stream_registration_hash: str
    residual_q_receipt_hash: str
    residual_stream_execution_receipt_hash: str
    d_fixed_affine_replay_receipt_hashes: tuple[str, ...]
    d_fixed_direction_receipt_hashes: tuple[str, ...]
    d_fixed_registration_hashes: tuple[str, ...]
    d_fixed_source_replay_receipt_hashes: tuple[str, ...]
    d_fixed_projection_receipt_hashes: tuple[str, ...]
    d_fixed_common_source_replay_axis_hash: str
    d_fixed_population_source_replay_axis_hash: str
    d_fixed_literal_j_tile_source_ledger_hashes: tuple[str, ...]
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    action_count: int
    shell_size: int
    physical_action_identity_hashes: tuple[str, ...]
    physical_action_hashes: tuple[str, ...]
    packed_action_hashes: tuple[str, ...]
    joint_direction_ids: tuple[str, ...]
    joint_direction_axis_hash: str
    small_axis_ids: tuple[tuple[str, ...], ...]
    d_fixed_stratum_counts: tuple[int, int]
    residual_population_counts: tuple[int, int]
    residual_observed_counts: tuple[int, int]
    population_count: int
    d_fixed_count: int
    d_fixed_mass_fraction: tuple[int, int]
    residual_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    alpha_ledger_hash: str
    alpha_coordinate_hash: str
    alpha_lineage_hash: str
    inherited_familywise_alpha: float
    inherited_alpha_spent: float
    additional_alpha_spent: float
    residual_exact_census: bool
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    measurement_device_identity: str
    residual_tile_receipt_hashes: tuple[str, ...]
    residual_pass2_tile_count: int
    d_fixed_action_replay_count: int
    d_fixed_total_source_pass_count: int
    d_fixed_total_source_payload_read_bytes: int
    residual_source_pass_count: int
    residual_source_payload_read_bytes: int
    compact_tensor_d2h_scalar_count: int
    compact_tensor_d2h_bytes: int
    small_matrix_center_digest: str
    small_matrix_radius_digest: str
    loewner_lower_digest: str
    loewner_upper_digest: str
    row_sum_radius_digest: str
    max_row_sum_radius_upper_bound: float
    no_additional_alpha_spend: bool = True
    exact_rational_population_mixing: bool = True
    target_complement_masses_kept_distinct: bool = True
    symmetric_shared_g_orientation_intersection: bool = True
    diagonal_mjv_q_intersection: bool = True
    redundant_d_fixed_shared_g_intersection: bool = True
    indefinite_loewner_lower_retained_without_psd_projection: bool = True
    only_shell_columns_retained_from_residual_mjv_stream: bool = True
    every_residual_mjv_tile_consumed_once: bool = True
    repeated_d_fixed_census_work_disclosed: bool = True
    single_pass_d_fixed_performance_claim: bool = False
    joint_action_complete_population_custody: bool = True
    legacy_private_population_registration: bool = False
    route_scale_d2h: bool = False
    route_scale_mjv_bank_retained: bool = False
    array_free_receipt: bool = True
    live_small_matrix_handle_required: bool = True
    population_loewner_authority: bool = False
    kkt_geometry_uncertainty_input_authority: bool = False
    solver_authority: bool = False
    d_face_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    performance_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_JOINT_AFFINE_SHELL_LOEWNER_STREAM_BRIDGE_ALGORITHM
    authority_scope: str = LITERAL_J_JOINT_AFFINE_SHELL_LOEWNER_STREAM_BRIDGE_AUTHORITY
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_LOEWNER_STREAM_BRIDGE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("joint Loewner stream receipts require their issuer")
        for name in (
            "joint_preoutcome_registration_hash",
            "joint_family_receipt_hash",
            "joint_postsource_custody_axis_hash",
            "population_projection_registration_hash",
            "residual_stream_registration_hash",
            "residual_q_receipt_hash",
            "residual_stream_execution_receipt_hash",
            "d_fixed_common_source_replay_axis_hash",
            "d_fixed_population_source_replay_axis_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "shell_freeze_receipt_hash",
            "joint_direction_axis_hash",
            "alpha_ledger_hash",
            "alpha_coordinate_hash",
            "alpha_lineage_hash",
            "measurement_device_identity",
            "small_matrix_center_digest",
            "small_matrix_radius_digest",
            "loewner_lower_digest",
            "loewner_upper_digest",
            "row_sum_radius_digest",
        ):
            _identity(getattr(self, name), name=name)
        actions = _positive_int(self.action_count, name="action_count")
        shell = _positive_int(self.shell_size, name="shell_size")
        routes = _positive_int(self.route_count, name="route_count")
        _nonnegative_int(self.proposal_generation, name="proposal_generation")
        for name in (
            "population_count",
            "d_fixed_count",
            "residual_pass2_tile_count",
            "d_fixed_action_replay_count",
            "d_fixed_total_source_pass_count",
            "d_fixed_total_source_payload_read_bytes",
            "residual_source_pass_count",
            "residual_source_payload_read_bytes",
            "compact_tensor_d2h_scalar_count",
            "compact_tensor_d2h_bytes",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        axes = (
            self.d_fixed_affine_replay_receipt_hashes,
            self.d_fixed_direction_receipt_hashes,
            self.d_fixed_registration_hashes,
            self.d_fixed_source_replay_receipt_hashes,
            self.d_fixed_projection_receipt_hashes,
            self.d_fixed_literal_j_tile_source_ledger_hashes,
            self.physical_action_identity_hashes,
            self.physical_action_hashes,
            self.packed_action_hashes,
        )
        if (
            any(
                len(_identity_axis(axis, name="action receipt axis")) != actions
                for axis in axes
            )
            or self.packed_action_hashes == self.physical_action_hashes
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.JOINT_AXIS_MISMATCH,
                "one receipt axis is incomplete or packed actions were conflated "
                "with wrapper actions",
            )
        shell_ids = _identity_axis(self.shell_route_ids, name="shell_route_ids")
        joint_ids = _identity_axis(self.joint_direction_ids, name="joint_direction_ids")
        if (
            type(self.shell_indices) is not tuple
            or len(self.shell_indices) != shell
            or len(shell_ids) != shell
            or any(type(item) is not int or item < 0 or item >= routes for item in self.shell_indices)
            or len(set(self.shell_indices)) != shell
            or len(joint_ids) != actions + shell
            or type(self.small_axis_ids) is not tuple
            or len(self.small_axis_ids) != actions
            or any(
                type(axis) is not tuple
                or len(axis) != shell + 1
                or tuple(axis[:-1]) != joint_ids[actions:]
                or axis[-1] != joint_ids[index]
                for index, axis in enumerate(self.small_axis_ids)
            )
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.JOINT_AXIS_MISMATCH,
                "shell, joint, and per-action small axes do not replay",
            )
        if (
            type(self.d_fixed_stratum_counts) is not tuple
            or type(self.residual_population_counts) is not tuple
            or type(self.residual_observed_counts) is not tuple
            or any(len(axis) != 2 for axis in (self.d_fixed_stratum_counts, self.residual_population_counts, self.residual_observed_counts))
            or any(type(item) is not int or item < 0 for axis in (self.d_fixed_stratum_counts, self.residual_population_counts, self.residual_observed_counts) for item in axis)
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MASS_COUNT_MISMATCH,
                "population count axes must be exact nonnegative pairs",
            )
        d_mass = _fraction_pair(self.d_fixed_mass_fraction, name="d_fixed_mass_fraction")
        residual_masses = tuple(
            _fraction_pair(item, name="residual_mass_fraction")
            for item in self.residual_mass_fractions
        )
        if len(residual_masses) != 2:
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MASS_COUNT_MISMATCH,
                "two residual mass fractions are required",
            )
        alpha = _finite_nonnegative(self.inherited_familywise_alpha, name="inherited_familywise_alpha")
        spent = _finite_nonnegative(self.inherited_alpha_spent, name="inherited_alpha_spent")
        additional = _finite_nonnegative(self.additional_alpha_spent, name="additional_alpha_spent")
        _finite_nonnegative(self.max_row_sum_radius_upper_bound, name="max_row_sum_radius_upper_bound")
        if (
            self.population_count <= 0
            or self.d_fixed_count <= 0
            or self.d_fixed_count != sum(self.d_fixed_stratum_counts)
            or Fraction(*d_mass) != Fraction(self.d_fixed_count, self.population_count)
            or any(
                Fraction(*residual_masses[h])
                != Fraction(self.residual_population_counts[h], self.population_count)
                for h in (0, 1)
            )
            or Fraction(*d_mass)
            + Fraction(*residual_masses[0])
            + Fraction(*residual_masses[1])
            != 1
            or any(self.residual_observed_counts[h] <= 0 or self.residual_observed_counts[h] > self.residual_population_counts[h] for h in (0, 1))
            or self.residual_exact_census
            != (self.residual_observed_counts == self.residual_population_counts)
            or spent != (0.0 if self.residual_exact_census else alpha)
            or additional != 0.0
            or self.d_fixed_action_replay_count != actions
            or self.d_fixed_total_source_pass_count != 2 * actions
            or self.residual_source_pass_count != 2
            or self.compact_tensor_d2h_bytes
            != self.compact_tensor_d2h_scalar_count * _F64_BYTES
            or (self.production_cuda_execution and self.compact_tensor_d2h_scalar_count <= 0)
            or (self.private_cpu_reference_execution and self.compact_tensor_d2h_scalar_count != 0)
            or self.production_cuda_execution == self.private_cpu_reference_execution
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MASS_COUNT_MISMATCH,
                "population masses, counts, alpha spend, or execution work do not close",
            )
        positive = (
            self.no_additional_alpha_spend,
            self.exact_rational_population_mixing,
            self.target_complement_masses_kept_distinct,
            self.symmetric_shared_g_orientation_intersection,
            self.diagonal_mjv_q_intersection,
            self.redundant_d_fixed_shared_g_intersection,
            self.indefinite_loewner_lower_retained_without_psd_projection,
            self.only_shell_columns_retained_from_residual_mjv_stream,
            self.every_residual_mjv_tile_consumed_once,
            self.repeated_d_fixed_census_work_disclosed,
            self.array_free_receipt,
            self.live_small_matrix_handle_required,
        )
        closed = (
            self.single_pass_d_fixed_performance_claim,
            self.route_scale_d2h,
            self.route_scale_mjv_bank_retained,
            self.solver_authority,
            self.d_face_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.performance_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            not 0.0 < alpha < 1.0
            or any(
                type(item) is not bool
                for item in (
                    self.production_cuda_execution,
                    self.private_cpu_reference_execution,
                    self.population_loewner_authority,
                    self.kkt_geometry_uncertainty_input_authority,
                    self.joint_action_complete_population_custody,
                    self.legacy_private_population_registration,
                )
            )
            or self.joint_action_complete_population_custody
            == self.legacy_private_population_registration
            or (
                self.production_cuda_execution
                and not self.joint_action_complete_population_custody
            )
            or (
                self.legacy_private_population_registration
                and not self.private_cpu_reference_execution
            )
            or any(type(item) is not bool or item is not True for item in positive)
            or any(type(item) is not bool or item is not False for item in closed)
            or self.population_loewner_authority != self.production_cuda_execution
            or self.kkt_geometry_uncertainty_input_authority
            != self.production_cuda_execution
            or (
                self.production_cuda_execution
                and not self.measurement_device_identity.startswith("cuda:")
            )
            or (
                self.private_cpu_reference_execution
                and self.measurement_device_identity != "cpu"
            )
            or self.algorithm != LITERAL_J_JOINT_AFFINE_SHELL_LOEWNER_STREAM_BRIDGE_ALGORITHM
            or self.authority_scope != LITERAL_J_JOINT_AFFINE_SHELL_LOEWNER_STREAM_BRIDGE_AUTHORITY
            or self.schema != LITERAL_J_JOINT_AFFINE_SHELL_LOEWNER_STREAM_BRIDGE_SCHEMA
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.AUTHORITY_VIOLATION,
                "joint Loewner receipt crossed its alpha, memory, or downstream authority boundary",
            )
        object.__setattr__(self, "shell_route_ids", shell_ids)
        object.__setattr__(self, "joint_direction_ids", joint_ids)
        object.__setattr__(self, "d_fixed_mass_fraction", d_mass)
        object.__setattr__(self, "residual_mass_fractions", residual_masses)
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}
        )

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "receipt-")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_hash()
        if observed != self._sealed_hash or (expected_hash is not None and observed != expected_hash):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MUTATED_CUSTODY,
                "joint Loewner receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellPass1LoewnerReceipt:
    """Compact population Loewner custody issued before route-scale pass 2."""

    joint_preoutcome_registration_hash: str
    joint_family_receipt_hash: str
    joint_postsource_custody_axis_hash: str
    population_projection_registration_hash: str
    residual_stream_registration_hash: str
    residual_q_receipt_hash: str
    residual_pass1_compact_shell_m_receipt_hash: str
    residual_mv_coordinate_subset_hash: str
    d_fixed_affine_replay_receipt_hashes: tuple[str, ...]
    d_fixed_registration_hashes: tuple[str, ...]
    d_fixed_source_replay_receipt_hashes: tuple[str, ...]
    d_fixed_projection_receipt_hashes: tuple[str, ...]
    d_fixed_common_source_replay_axis_hash: str
    d_fixed_population_source_replay_axis_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    action_count: int
    shell_size: int
    physical_action_identity_hashes: tuple[str, ...]
    physical_action_hashes: tuple[str, ...]
    packed_action_hashes: tuple[str, ...]
    joint_direction_ids: tuple[str, ...]
    joint_direction_axis_hash: str
    small_axis_ids: tuple[tuple[str, ...], ...]
    d_fixed_stratum_counts: tuple[int, int]
    residual_population_counts: tuple[int, int]
    residual_observed_counts: tuple[int, int]
    population_count: int
    d_fixed_count: int
    d_fixed_mass_fraction: tuple[int, int]
    residual_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    alpha_ledger_hash: str
    alpha_coordinate_hash: str
    alpha_lineage_hash: str
    inherited_familywise_alpha: float
    inherited_alpha_spent: float
    additional_alpha_spent: float
    residual_exact_census: bool
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    measurement_device_identity: str
    compact_tensor_d2h_scalar_count: int
    compact_tensor_d2h_bytes: int
    small_matrix_center_digest: str
    small_matrix_radius_digest: str
    loewner_lower_digest: str
    loewner_upper_digest: str
    row_sum_radius_digest: str
    max_row_sum_radius_upper_bound: float
    pass2_unopened_at_issuance: bool = True
    pass2_unopened_after_issuance: bool = True
    source_owned_pass1_shell_m_used: bool = True
    no_additional_alpha_spend: bool = True
    exact_rational_population_mixing: bool = True
    target_complement_masses_kept_distinct: bool = True
    symmetric_shared_g_orientation_intersection: bool = True
    diagonal_mjv_q_intersection: bool = True
    redundant_d_fixed_shared_g_intersection: bool = True
    indefinite_loewner_lower_retained_without_psd_projection: bool = True
    route_scale_d2h: bool = False
    route_scale_mjv_bank_retained: bool = False
    array_free_receipt: bool = True
    live_small_matrix_handle_required: bool = True
    joint_action_complete_population_custody: bool = True
    legacy_private_population_registration: bool = False
    population_loewner_authority: bool = False
    kkt_geometry_uncertainty_input_authority: bool = False
    solver_authority: bool = False
    d_face_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    performance_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_JOINT_AFFINE_SHELL_PASS1_LOEWNER_BRIDGE_ALGORITHM
    authority_scope: str = LITERAL_J_JOINT_AFFINE_SHELL_PASS1_LOEWNER_BRIDGE_AUTHORITY
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_PASS1_LOEWNER_BRIDGE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _PASS1_RECEIPT_TOKEN:
            raise PermissionError("pass-1 Loewner receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "joint_preoutcome_registration_hash",
            "joint_family_receipt_hash",
            "joint_postsource_custody_axis_hash",
            "population_projection_registration_hash",
            "residual_stream_registration_hash",
            "residual_q_receipt_hash",
            "residual_pass1_compact_shell_m_receipt_hash",
            "residual_mv_coordinate_subset_hash",
            "d_fixed_common_source_replay_axis_hash",
            "d_fixed_population_source_replay_axis_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "shell_freeze_receipt_hash",
            "joint_direction_axis_hash",
            "alpha_ledger_hash",
            "alpha_coordinate_hash",
            "alpha_lineage_hash",
            "measurement_device_identity",
            "small_matrix_center_digest",
            "small_matrix_radius_digest",
            "loewner_lower_digest",
            "loewner_upper_digest",
            "row_sum_radius_digest",
        ):
            _identity(getattr(self, name), name=name)
        actions = _positive_int(self.action_count, name="action_count")
        shell = _positive_int(self.shell_size, name="shell_size")
        routes = _positive_int(self.route_count, name="route_count")
        _nonnegative_int(self.proposal_generation, name="proposal_generation")
        for name in (
            "population_count",
            "d_fixed_count",
            "compact_tensor_d2h_scalar_count",
            "compact_tensor_d2h_bytes",
        ):
            _nonnegative_int(getattr(self, name), name=name)
        action_axes = (
            self.d_fixed_affine_replay_receipt_hashes,
            self.d_fixed_registration_hashes,
            self.d_fixed_source_replay_receipt_hashes,
            self.d_fixed_projection_receipt_hashes,
            self.physical_action_identity_hashes,
            self.physical_action_hashes,
            self.packed_action_hashes,
        )
        if any(len(_identity_axis(axis, name="pass1 action axis")) != actions for axis in action_axes):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.JOINT_AXIS_MISMATCH,
                "pass-1 Loewner action custody is incomplete",
            )
        shell_ids = _identity_axis(self.shell_route_ids, name="shell_route_ids")
        joint_ids = _identity_axis(self.joint_direction_ids, name="joint_direction_ids")
        if (
            self.packed_action_hashes == self.physical_action_hashes
            or type(self.shell_indices) is not tuple
            or len(self.shell_indices) != shell
            or len(shell_ids) != shell
            or len(set(self.shell_indices)) != shell
            or any(type(item) is not int or item < 0 or item >= routes for item in self.shell_indices)
            or len(joint_ids) != actions + shell
            or type(self.small_axis_ids) is not tuple
            or len(self.small_axis_ids) != actions
            or any(
                type(axis) is not tuple
                or len(axis) != shell + 1
                or tuple(axis[:-1]) != joint_ids[actions:]
                or axis[-1] != joint_ids[index]
                for index, axis in enumerate(self.small_axis_ids)
            )
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.JOINT_AXIS_MISMATCH,
                "pass-1 Loewner shell, action, or compact axes do not replay",
            )
        count_axes = (
            self.d_fixed_stratum_counts,
            self.residual_population_counts,
            self.residual_observed_counts,
        )
        if any(type(axis) is not tuple or len(axis) != 2 or any(type(item) is not int or item < 0 for item in axis) for axis in count_axes):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MASS_COUNT_MISMATCH,
                "pass-1 population count axes are malformed",
            )
        d_mass = _fraction_pair(self.d_fixed_mass_fraction, name="d_fixed_mass_fraction")
        residual_masses = tuple(
            _fraction_pair(item, name="residual_mass_fraction")
            for item in self.residual_mass_fractions
        )
        alpha = _finite_nonnegative(self.inherited_familywise_alpha, name="inherited_familywise_alpha")
        spent = _finite_nonnegative(self.inherited_alpha_spent, name="inherited_alpha_spent")
        additional = _finite_nonnegative(self.additional_alpha_spent, name="additional_alpha_spent")
        _finite_nonnegative(self.max_row_sum_radius_upper_bound, name="max_row_sum_radius_upper_bound")
        if (
            len(residual_masses) != 2
            or self.population_count <= 0
            or self.d_fixed_count <= 0
            or self.d_fixed_count != sum(self.d_fixed_stratum_counts)
            or Fraction(*d_mass) != Fraction(self.d_fixed_count, self.population_count)
            or any(Fraction(*residual_masses[h]) != Fraction(self.residual_population_counts[h], self.population_count) for h in (0, 1))
            or Fraction(*d_mass) + Fraction(*residual_masses[0]) + Fraction(*residual_masses[1]) != 1
            or any(self.residual_observed_counts[h] <= 0 or self.residual_observed_counts[h] > self.residual_population_counts[h] for h in (0, 1))
            or self.residual_exact_census != (self.residual_observed_counts == self.residual_population_counts)
            or spent != (0.0 if self.residual_exact_census else alpha)
            or additional != 0.0
            or self.compact_tensor_d2h_bytes != self.compact_tensor_d2h_scalar_count * _F64_BYTES
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or (self.production_cuda_execution and self.compact_tensor_d2h_scalar_count <= 0)
            or (self.private_cpu_reference_execution and self.compact_tensor_d2h_scalar_count != 0)
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MASS_COUNT_MISMATCH,
                "pass-1 population masses, alpha, or compact transfer accounting do not close",
            )
        positive = (
            self.pass2_unopened_at_issuance,
            self.pass2_unopened_after_issuance,
            self.source_owned_pass1_shell_m_used,
            self.no_additional_alpha_spend,
            self.exact_rational_population_mixing,
            self.target_complement_masses_kept_distinct,
            self.symmetric_shared_g_orientation_intersection,
            self.diagonal_mjv_q_intersection,
            self.redundant_d_fixed_shared_g_intersection,
            self.indefinite_loewner_lower_retained_without_psd_projection,
            self.array_free_receipt,
            self.live_small_matrix_handle_required,
        )
        closed = (
            self.route_scale_d2h,
            self.route_scale_mjv_bank_retained,
            self.solver_authority,
            self.d_face_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.performance_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            not 0.0 < alpha < 1.0
            or any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or type(self.residual_exact_census) is not bool
            or type(self.production_cuda_execution) is not bool
            or type(self.private_cpu_reference_execution) is not bool
            or type(self.joint_action_complete_population_custody) is not bool
            or type(self.legacy_private_population_registration) is not bool
            or type(self.population_loewner_authority) is not bool
            or type(self.kkt_geometry_uncertainty_input_authority) is not bool
            or self.joint_action_complete_population_custody == self.legacy_private_population_registration
            or (self.production_cuda_execution and not self.joint_action_complete_population_custody)
            or (self.legacy_private_population_registration and not self.private_cpu_reference_execution)
            or self.population_loewner_authority != self.production_cuda_execution
            or self.kkt_geometry_uncertainty_input_authority != self.production_cuda_execution
            or (self.production_cuda_execution and not self.measurement_device_identity.startswith("cuda:"))
            or (self.private_cpu_reference_execution and self.measurement_device_identity != "cpu")
            or self.algorithm != LITERAL_J_JOINT_AFFINE_SHELL_PASS1_LOEWNER_BRIDGE_ALGORITHM
            or self.authority_scope != LITERAL_J_JOINT_AFFINE_SHELL_PASS1_LOEWNER_BRIDGE_AUTHORITY
            or self.schema != LITERAL_J_JOINT_AFFINE_SHELL_PASS1_LOEWNER_BRIDGE_SCHEMA
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.AUTHORITY_VIOLATION,
                "pass-1 Loewner receipt crossed its source or downstream authority boundary",
            )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}
        )

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "pass1-receipt-")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (expected_hash is not None and observed != expected_hash):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MUTATED_CUSTODY,
                "pass-1 Loewner receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellD1Pass1LoewnerReceipt:
    """D1/residual population interval custody with explicit measure ancestry.

    Production receipts bind the exact outcome-independent population-measure owner
    and the producer-owned PRE/POST action-axis aggregate.  The bounded CPU
    reference deliberately records sentinel measure ancestry and an ordered
    tuple of *atomic* action sources; it never promotes that tuple into
    action-complete causal custody.
    """

    joint_preoutcome_registration_hash: str
    joint_family_receipt_hash: str
    joint_postsource_custody_axis_hash: str
    population_measure_registration_hash: str
    population_measure_receipt_hash: str
    resident_population_registration_hash: str
    population_measure_design_hash: str
    population_measure_weight_receipt_hash: str
    action_axis_preoutcome_receipt_hash: str
    action_axis_source_receipt_hash: str
    atomic_source_receipt_hashes: tuple[str, ...]
    atomic_preoutcome_receipt_hashes: tuple[str, ...]
    d1_stream_receipt_hash: str
    d1_q_receipt_hash: str
    d1_compact_shell_m_receipt_hash: str
    d1_source_axis_hash: str
    d1_literal_j_shard_receipt_hashes: tuple[str, ...]
    d1_literal_j_content_sha256s: tuple[str, ...]
    residual_stream_registration_hash: str
    residual_q_receipt_hash: str
    residual_compact_shell_m_receipt_hash: str
    residual_source_proof_closure_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    residual_source_shard_receipt_hash_axes: tuple[tuple[str, ...], tuple[str, ...]]
    residual_source_content_sha256_axes: tuple[tuple[str, ...], tuple[str, ...]]
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    action_count: int
    shell_size: int
    physical_action_axis_hash: str
    physical_action_identity_hashes: tuple[str, ...]
    physical_action_hashes: tuple[str, ...]
    packed_action_hashes: tuple[str, ...]
    joint_direction_ids: tuple[str, ...]
    joint_direction_axis_hash: str
    small_axis_ids: tuple[tuple[str, ...], ...]
    d1_stratum_counts: tuple[int, int]
    residual_population_counts: tuple[int, int]
    residual_observed_counts: tuple[int, int]
    population_count: int
    d1_count: int
    d1_mass_fraction: tuple[int, int]
    residual_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    alpha_ledger_hash: str
    alpha_coordinate_hash: str
    alpha_lineage_hash: str
    inherited_familywise_alpha: float
    inherited_alpha_spent: float
    additional_alpha_spent: float
    measurement_device_identity: str
    small_matrix_center_digest: str
    small_matrix_radius_digest: str
    loewner_lower_digest: str
    loewner_upper_digest: str
    row_sum_radius_digest: str
    max_row_sum_radius_upper_bound: float
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    d1_sampling_radius_included: bool = False
    d1_numeric_radius_included: bool = True
    residual_sampling_and_numeric_radii_included: bool = True
    exact_rational_population_mixing: bool = True
    target_complement_masses_kept_distinct: bool = True
    centers_may_cancel_radii_may_not: bool = True
    indefinite_loewner_lower_retained_without_psd_projection: bool = True
    pass2_streams_unopened: bool = True
    route_scale_d2h: bool = False
    route_scale_mjv_bank_retained: bool = False
    array_free_receipt: bool = True
    live_small_matrix_handle_required: bool = True
    ordered_atomic_source_tuple_recorded: bool = True
    population_measure_binding_present: bool = False
    action_complete_source_binding_present: bool = False
    atomic_source_tuple_nonauthoritative: bool = True
    population_loewner_authority: bool = False
    kkt_geometry_uncertainty_input_authority: bool = False
    solver_authority: bool = False
    d_face_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    performance_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_JOINT_AFFINE_SHELL_D1_PASS1_LOEWNER_BRIDGE_ALGORITHM
    authority_scope: str = LITERAL_J_JOINT_AFFINE_SHELL_D1_PASS1_LOEWNER_BRIDGE_AUTHORITY
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_D1_PASS1_LOEWNER_BRIDGE_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _D1_PASS1_RECEIPT_TOKEN:
            raise PermissionError("D1 pass-1 Loewner receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "joint_preoutcome_registration_hash",
            "joint_family_receipt_hash",
            "joint_postsource_custody_axis_hash",
            "population_measure_registration_hash",
            "population_measure_receipt_hash",
            "resident_population_registration_hash",
            "population_measure_design_hash",
            "population_measure_weight_receipt_hash",
            "action_axis_preoutcome_receipt_hash",
            "action_axis_source_receipt_hash",
            "d1_stream_receipt_hash",
            "d1_q_receipt_hash",
            "d1_compact_shell_m_receipt_hash",
            "d1_source_axis_hash",
            "residual_stream_registration_hash",
            "residual_q_receipt_hash",
            "residual_compact_shell_m_receipt_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "shell_freeze_receipt_hash",
            "physical_action_axis_hash",
            "joint_direction_axis_hash",
            "alpha_ledger_hash",
            "alpha_coordinate_hash",
            "alpha_lineage_hash",
            "measurement_device_identity",
            "small_matrix_center_digest",
            "small_matrix_radius_digest",
            "loewner_lower_digest",
            "loewner_upper_digest",
            "row_sum_radius_digest",
        ):
            _identity(getattr(self, name), name=name)
        actions = _positive_int(self.action_count, name="action_count")
        shell = _positive_int(self.shell_size, name="shell_size")
        routes = _positive_int(self.route_count, name="route_count")
        _nonnegative_int(self.proposal_generation, name="proposal_generation")
        atomic = _identity_axis(
            self.atomic_source_receipt_hashes,
            name="atomic_source_receipt_hashes",
        )
        atomic_pre = _identity_axis(
            self.atomic_preoutcome_receipt_hashes,
            name="atomic_preoutcome_receipt_hashes",
        )
        physical_identity = _identity_axis(
            self.physical_action_identity_hashes,
            name="physical_action_identity_hashes",
        )
        physical = _identity_axis(
            self.physical_action_hashes,
            name="physical_action_hashes",
        )
        packed = _identity_axis(
            self.packed_action_hashes,
            name="packed_action_hashes",
        )
        d1_shards = _identity_axis(
            self.d1_literal_j_shard_receipt_hashes,
            name="d1_literal_j_shard_receipt_hashes",
        )
        d1_contents = _identity_axis(
            self.d1_literal_j_content_sha256s,
            name="d1_literal_j_content_sha256s",
        )
        joint_ids = _identity_axis(self.joint_direction_ids, name="joint_direction_ids")
        shell_ids = _identity_axis(self.shell_route_ids, name="shell_route_ids")
        if (
            len(atomic) != actions
            or len(atomic_pre) != actions
            or len(physical_identity) != actions
            or len(physical) != actions
            or len(packed) != actions
            or physical == packed
            or len(d1_shards) != len(d1_contents)
            or len(joint_ids) != actions + shell
            or len(shell_ids) != shell
            or type(self.shell_indices) is not tuple
            or len(self.shell_indices) != shell
            or len(set(self.shell_indices)) != shell
            or any(type(item) is not int or item < 0 or item >= routes for item in self.shell_indices)
            or type(self.small_axis_ids) is not tuple
            or len(self.small_axis_ids) != actions
            or any(
                type(axis) is not tuple
                or len(axis) != shell + 1
                or tuple(axis[:-1]) != joint_ids[actions:]
                or axis[-1] != joint_ids[action]
                for action, axis in enumerate(self.small_axis_ids)
            )
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.JOINT_AXIS_MISMATCH,
                "D1 pass-1 source, action, shell, or small-matrix axes are incomplete",
            )
        for name, axis in (
            ("d1_stratum_counts", self.d1_stratum_counts),
            ("residual_population_counts", self.residual_population_counts),
            ("residual_observed_counts", self.residual_observed_counts),
        ):
            if type(axis) is not tuple or len(axis) != 2 or any(
                type(item) is not int or item < 0 for item in axis
            ):
                _refuse(
                    V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MASS_COUNT_MISMATCH,
                    f"{name} is not one target/complement count pair",
                )
        proof_axes = self.residual_source_proof_closure_hash_axes
        shard_axes = self.residual_source_shard_receipt_hash_axes
        content_axes = self.residual_source_content_sha256_axes
        if any(
            type(axes) is not tuple
            or len(axes) != 2
            or any(type(axis) is not tuple or not axis for axis in axes)
            for axes in (proof_axes, shard_axes, content_axes)
        ) or any(
            not (
                len(proof_axes[h]) == len(shard_axes[h]) == len(content_axes[h])
            )
            for h in (0, 1)
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.SOURCE_MISMATCH,
                "residual proof/shard/content segment axes do not nest identically",
            )
        d1_mass = _fraction_pair(self.d1_mass_fraction, name="d1_mass_fraction")
        residual_masses = tuple(
            _fraction_pair(value, name="residual_mass_fraction")
            for value in self.residual_mass_fractions
        )
        alpha = _finite_nonnegative(
            self.inherited_familywise_alpha,
            name="inherited_familywise_alpha",
        )
        spent = _finite_nonnegative(
            self.inherited_alpha_spent,
            name="inherited_alpha_spent",
        )
        additional = _finite_nonnegative(
            self.additional_alpha_spent,
            name="additional_alpha_spent",
        )
        _finite_nonnegative(
            self.max_row_sum_radius_upper_bound,
            name="max_row_sum_radius_upper_bound",
        )
        if (
            self.population_count <= 0
            or self.d1_count != sum(self.d1_stratum_counts)
            or Fraction(*d1_mass) != Fraction(self.d1_count, self.population_count)
            or any(
                Fraction(*residual_masses[h])
                != Fraction(self.residual_population_counts[h], self.population_count)
                for h in (0, 1)
            )
            or Fraction(*d1_mass)
            + Fraction(*residual_masses[0])
            + Fraction(*residual_masses[1])
            != 1
            or any(
                self.residual_observed_counts[h] <= 0
                or self.residual_observed_counts[h]
                > self.residual_population_counts[h]
                for h in (0, 1)
            )
            or not 0.0 < alpha < 1.0
            or spent
            != (
                0.0
                if self.residual_observed_counts == self.residual_population_counts
                else alpha
            )
            or additional != 0.0
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MASS_COUNT_MISMATCH,
                "D1/residual exact population masses or inherited alpha do not close",
            )
        common_positive = (
            self.d1_numeric_radius_included,
            self.residual_sampling_and_numeric_radii_included,
            self.exact_rational_population_mixing,
            self.target_complement_masses_kept_distinct,
            self.centers_may_cancel_radii_may_not,
            self.indefinite_loewner_lower_retained_without_psd_projection,
            self.pass2_streams_unopened,
            self.array_free_receipt,
            self.live_small_matrix_handle_required,
            self.ordered_atomic_source_tuple_recorded,
        )
        common_closed = (
            self.d1_sampling_radius_included,
            self.route_scale_d2h,
            self.route_scale_mjv_bank_retained,
            self.solver_authority,
            self.d_face_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.performance_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in common_positive)
            or any(type(value) is not bool or value is not False for value in common_closed)
            or type(self.production_cuda_execution) is not bool
            or type(self.private_cpu_reference_execution) is not bool
            or self.production_cuda_execution == self.private_cpu_reference_execution
            or type(self.action_complete_source_binding_present) is not bool
            or type(self.population_measure_binding_present) is not bool
            or type(self.atomic_source_tuple_nonauthoritative) is not bool
            or type(self.population_loewner_authority) is not bool
            or type(self.kkt_geometry_uncertainty_input_authority) is not bool
            or self.action_complete_source_binding_present != self.production_cuda_execution
            or self.population_measure_binding_present
            != self.production_cuda_execution
            or self.atomic_source_tuple_nonauthoritative != self.private_cpu_reference_execution
            or self.population_loewner_authority != self.production_cuda_execution
            or self.kkt_geometry_uncertainty_input_authority
            != self.production_cuda_execution
            or (
                self.production_cuda_execution
                and not self.measurement_device_identity.startswith("cuda:")
            )
            or (
                self.private_cpu_reference_execution
                and self.measurement_device_identity != "cpu"
            )
            or (
                self.production_cuda_execution
                and any(
                    value.startswith("private-reference:")
                    for value in (
                        self.population_measure_registration_hash,
                        self.population_measure_receipt_hash,
                        self.resident_population_registration_hash,
                        self.population_measure_design_hash,
                        self.population_measure_weight_receipt_hash,
                        self.action_axis_preoutcome_receipt_hash,
                        self.action_axis_source_receipt_hash,
                    )
                )
            )
            or (
                self.private_cpu_reference_execution
                and any(
                    not value.startswith("private-reference:")
                    for value in (
                        self.population_measure_registration_hash,
                        self.population_measure_receipt_hash,
                        self.resident_population_registration_hash,
                        self.population_measure_design_hash,
                        self.population_measure_weight_receipt_hash,
                        self.action_axis_preoutcome_receipt_hash,
                        self.action_axis_source_receipt_hash,
                    )
                )
            )
            or self.algorithm
            != LITERAL_J_JOINT_AFFINE_SHELL_D1_PASS1_LOEWNER_BRIDGE_ALGORITHM
            or self.authority_scope
            != LITERAL_J_JOINT_AFFINE_SHELL_D1_PASS1_LOEWNER_BRIDGE_AUTHORITY
            or self.schema
            != LITERAL_J_JOINT_AFFINE_SHELL_D1_PASS1_LOEWNER_BRIDGE_SCHEMA
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.AUTHORITY_VIOLATION,
                "D1 pass-1 reference receipt widened beyond non-authoritative CPU numerics",
            )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                item.name: getattr(self, item.name)
                for item in fields(self)
                if not item.name.startswith("_")
            }
        )

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "d1-pass1-receipt-")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MUTATED_CUSTODY,
                "D1 pass-1 Loewner receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()



@dataclass(frozen=True)
class V1FactoredJointLoewnerLineage:
    """Identity, axis, action, and custody coordinates of a factored joint Loewner issuance.

    The factored plane has no D2 fanout registration; these coordinates are
    supplied by the factored endpoint consumer from its live handoff (session
    design, replay boundary, packet receipts) and from the frozen direction
    family (physical actions and shared shell).  No alpha coordinate lives
    here: alpha lineage enters only through an owner binding receipt.
    """

    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    shell_route_ids: tuple[str, ...]
    physical_action_identity_hashes: tuple[str, ...]
    physical_action_hashes: tuple[str, ...]
    packed_action_hashes: tuple[str, ...]
    outside_direction_ids: tuple[str, ...]
    shared_shell_direction_ids: tuple[str, ...]
    factored_packet_receipt_hashes: tuple[str, ...]
    session_design_hash: str
    replay_boundary_hash: str


@dataclass(frozen=True, slots=True)
class V1FactoredJointAffineShellLoewnerReceipt:
    """Array-free private-reference receipt of factored mixed-sample geometry.

    Sibling of the D1 pass-1 receipt: same geometry, axis, mass, and digest
    fields; factored custody hashes in place of the row-backed stream lineage.
    ``bridge_result_bound`` is ``True``: this receipt *is* the binding of the
    factored Loewner stream into the joint bridge result.
    This family has no source-bound population sampling enclosure or census
    witness. It cannot carry production or population/KKT authority.
    """

    factored_population_bounds_receipt_hash: str
    component_stream_receipt_hashes: tuple[str, str, str]
    factored_packet_receipt_hashes: tuple[str, ...]
    session_design_hash: str
    replay_boundary_hash: str
    mass_law_id: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    action_count: int
    shell_size: int
    physical_action_identity_hashes: tuple[str, ...]
    physical_action_hashes: tuple[str, ...]
    packed_action_hashes: tuple[str, ...]
    joint_direction_ids: tuple[str, ...]
    joint_direction_axis_hash: str
    small_axis_ids: tuple[tuple[str, ...], ...]
    d1_stratum_counts: tuple[int, int]
    residual_population_counts: tuple[int, int]
    residual_observed_counts: tuple[int, int]
    population_count: int
    d1_count: int
    d1_mass_fraction: tuple[int, int]
    residual_mass_fractions: tuple[tuple[int, int], tuple[int, int]]
    alpha_owner_binding_receipt_hash: str
    alpha_ledger_hash: str
    alpha_coordinate_hash: str
    alpha_lineage_hash: str
    inherited_familywise_alpha: float
    inherited_alpha_spent: float
    additional_alpha_spent: float
    alpha_lineage_bound: bool
    measurement_device_identity: str
    small_matrix_center_digest: str
    small_matrix_radius_digest: str
    loewner_lower_digest: str
    loewner_upper_digest: str
    row_sum_radius_digest: str
    max_row_sum_radius_upper_bound: float
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    bridge_result_bound: bool = True
    row_payload_present: bool = False
    exact_rational_population_mixing: bool = True
    target_complement_masses_kept_distinct: bool = True
    centers_may_cancel_radii_may_not: bool = True
    indefinite_loewner_lower_retained_without_psd_projection: bool = True
    route_scale_d2h: bool = False
    array_free_receipt: bool = True
    live_small_matrix_handle_required: bool = True
    population_loewner_authority: bool = False
    kkt_geometry_uncertainty_input_authority: bool = False
    solver_authority: bool = False
    d_face_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    performance_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = LITERAL_J_JOINT_AFFINE_SHELL_FACTORED_LOEWNER_BRIDGE_ALGORITHM
    authority_scope: str = LITERAL_J_JOINT_AFFINE_SHELL_FACTORED_LOEWNER_BRIDGE_AUTHORITY
    schema: str = LITERAL_J_JOINT_AFFINE_SHELL_FACTORED_LOEWNER_BRIDGE_SCHEMA
    _sealed_hash: str = field(init=False, repr=False, compare=False, default="")
    _issuance_token: InitVar[object | None] = None

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _FACTORED_RECEIPT_TOKEN:
            raise PermissionError("factored joint Loewner receipts require their issuer")
        self._validate()
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _validate(self) -> None:
        for name in (
            "factored_population_bounds_receipt_hash",
            "session_design_hash",
            "replay_boundary_hash",
            "mass_law_id",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "joint_direction_axis_hash",
            "alpha_owner_binding_receipt_hash",
            "alpha_ledger_hash",
            "alpha_coordinate_hash",
            "alpha_lineage_hash",
            "measurement_device_identity",
            "small_matrix_center_digest",
            "small_matrix_radius_digest",
            "loewner_lower_digest",
            "loewner_upper_digest",
            "row_sum_radius_digest",
        ):
            _identity(getattr(self, name), name=name)
        actions = _positive_int(self.action_count, name="action_count")
        shell = _positive_int(self.shell_size, name="shell_size")
        routes = _positive_int(self.route_count, name="route_count")
        _nonnegative_int(self.proposal_generation, name="proposal_generation")
        components = _identity_axis(self.component_stream_receipt_hashes, name="component_stream_receipt_hashes")
        packets = _identity_axis(self.factored_packet_receipt_hashes, name="factored_packet_receipt_hashes")
        physical_identity = _identity_axis(self.physical_action_identity_hashes, name="physical_action_identity_hashes")
        physical = _identity_axis(self.physical_action_hashes, name="physical_action_hashes")
        packed = _identity_axis(self.packed_action_hashes, name="packed_action_hashes")
        joint_ids = _identity_axis(self.joint_direction_ids, name="joint_direction_ids")
        shell_ids = _identity_axis(self.shell_route_ids, name="shell_route_ids")
        if (
            len(components) != 3
            or not packets
            or len(physical_identity) != actions
            or len(physical) != actions
            or len(packed) != actions
            or physical == packed
            or len(joint_ids) != actions + shell
            or len(shell_ids) != shell
            or type(self.shell_indices) is not tuple
            or len(self.shell_indices) != shell
            or len(set(self.shell_indices)) != shell
            or any(type(item) is not int or item < 0 or item >= routes for item in self.shell_indices)
            or type(self.small_axis_ids) is not tuple
            or len(self.small_axis_ids) != actions
            or any(
                type(axis) is not tuple
                or len(axis) != shell + 1
                or tuple(axis[:-1]) != joint_ids[actions:]
                or axis[-1] != joint_ids[action]
                for action, axis in enumerate(self.small_axis_ids)
            )
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.JOINT_AXIS_MISMATCH,
                "factored joint Loewner source, action, shell, or small-matrix axes are incomplete",
            )
        for name, axis in (
            ("d1_stratum_counts", self.d1_stratum_counts),
            ("residual_population_counts", self.residual_population_counts),
            ("residual_observed_counts", self.residual_observed_counts),
        ):
            if type(axis) is not tuple or len(axis) != 2 or any(type(item) is not int or item < 0 for item in axis):
                _refuse(
                    V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MASS_COUNT_MISMATCH,
                    f"{name} is not one target/complement count pair",
                )
        d1_mass = _fraction_pair(self.d1_mass_fraction, name="d1_mass_fraction")
        residual_masses = tuple(_fraction_pair(value, name="residual_mass_fraction") for value in self.residual_mass_fractions)
        alpha = _finite_nonnegative(self.inherited_familywise_alpha, name="inherited_familywise_alpha")
        spent = _finite_nonnegative(self.inherited_alpha_spent, name="inherited_alpha_spent")
        additional = _finite_nonnegative(self.additional_alpha_spent, name="additional_alpha_spent")
        _finite_nonnegative(self.max_row_sum_radius_upper_bound, name="max_row_sum_radius_upper_bound")
        if (
            self.population_count <= 0
            or self.d1_count != sum(self.d1_stratum_counts)
            or Fraction(*d1_mass) != Fraction(self.d1_count, self.population_count)
            or any(
                Fraction(*residual_masses[h])
                != Fraction(self.residual_population_counts[h] - self.d1_stratum_counts[h], self.population_count)
                for h in (0, 1)
            )
            or Fraction(*d1_mass) + Fraction(*residual_masses[0]) + Fraction(*residual_masses[1]) != 1
            or any(
                (self.residual_population_counts[h] - self.d1_stratum_counts[h] > 0 and self.residual_observed_counts[h] <= 0)
                or self.residual_observed_counts[h] > max(0, self.residual_population_counts[h] - self.d1_stratum_counts[h])
                for h in (0, 1)
            )
            or not 0.0 < alpha < 1.0
            or additional != 0.0
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MASS_COUNT_MISMATCH,
                "factored D1/residual exact population masses or inherited alpha do not close",
            )
        del spent
        private_alpha = any(
            value.startswith("private-reference:")
            for value in (self.alpha_owner_binding_receipt_hash, self.alpha_ledger_hash, self.alpha_coordinate_hash, self.alpha_lineage_hash)
        )
        common_positive = (
            self.bridge_result_bound,
            self.exact_rational_population_mixing,
            self.target_complement_masses_kept_distinct,
            self.centers_may_cancel_radii_may_not,
            self.indefinite_loewner_lower_retained_without_psd_projection,
            self.array_free_receipt,
            self.live_small_matrix_handle_required,
        )
        common_closed = (
            self.row_payload_present,
            self.route_scale_d2h,
            self.solver_authority,
            self.d_face_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.performance_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        production = self.production_cuda_execution
        if (
            any(type(value) is not bool or value is not True for value in common_positive)
            or any(type(value) is not bool or value is not False for value in common_closed)
            or type(production) is not bool
            or type(self.private_cpu_reference_execution) is not bool
            or production == self.private_cpu_reference_execution
            or type(self.alpha_lineage_bound) is not bool
            or type(self.population_loewner_authority) is not bool
            or type(self.kkt_geometry_uncertainty_input_authority) is not bool
            or self.alpha_lineage_bound == private_alpha
            or production is not False
            or self.alpha_lineage_bound is not False
            or self.population_loewner_authority is not False
            or self.kkt_geometry_uncertainty_input_authority is not False
            or (production and not self.measurement_device_identity.startswith("cuda:"))
            or (self.private_cpu_reference_execution and self.measurement_device_identity != "cpu")
            or self.algorithm != LITERAL_J_JOINT_AFFINE_SHELL_FACTORED_LOEWNER_BRIDGE_ALGORITHM
            or self.authority_scope != LITERAL_J_JOINT_AFFINE_SHELL_FACTORED_LOEWNER_BRIDGE_AUTHORITY
            or self.schema != LITERAL_J_JOINT_AFFINE_SHELL_FACTORED_LOEWNER_BRIDGE_SCHEMA
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.AUTHORITY_VIOLATION,
                "factored joint Loewner receipt widened beyond its authority law",
            )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {item.name: getattr(self, item.name) for item in fields(self) if not item.name.startswith("_")}
        )

    def _compute_hash(self) -> str:
        return content_hash(self._payload(), prefix=_PREFIX + "factored-receipt-")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._validate()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (expected_hash is not None and observed != expected_hash):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MUTATED_CUSTODY,
                "factored joint Loewner receipt no longer replays",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self._sealed_hash

    def to_dict(self) -> dict[str, object]:
        return dict(self._payload())


class V1LiteralJJointAffineShellLoewnerStreamLease:
    __slots__ = (
        "_owner",
        "_token",
        "_closed",
        "small_center",
        "small_radius",
        "loewner_lower",
        "loewner_upper",
        "row_sum_radius",
    )

    def __init__(
        self,
        owner: "V1LiteralJJointAffineShellLoewnerStreamHandle",
        token: object,
        tensors: Mapping[str, Any],
    ) -> None:
        self._owner = owner
        self._token = token
        self._closed = False
        self.small_center = tensors["small_center"]
        self.small_radius = tensors["small_radius"]
        self.loewner_lower = tensors["loewner_lower"]
        self.loewner_upper = tensors["loewner_upper"]
        self.row_sum_radius = tensors["row_sum_radius"]

    def __enter__(self) -> "V1LiteralJJointAffineShellLoewnerStreamLease":
        self._owner.assert_live()
        return self

    def close(self) -> None:
        if not self._closed:
            self._owner._close_lease(self._token)
            self._closed = True

    def __exit__(self, *_: object) -> bool:
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("joint Loewner leases are nonserializable")


class V1LiteralJJointAffineShellLoewnerStreamHandle:
    """Mutation-guarded custody of only the bounded per-action matrices."""

    __slots__ = ("receipt", "_tensors", "_guards", "_invariants", "_tokens", "_evicted")

    def __init__(
        self,
        *,
        receipt: (
            V1LiteralJJointAffineShellLoewnerStreamReceipt
            | V1LiteralJJointAffineShellPass1LoewnerReceipt
            | V1LiteralJJointAffineShellD1Pass1LoewnerReceipt
            | V1FactoredJointAffineShellLoewnerReceipt
        ),
        tensors: Mapping[str, Any],
        issuance_token: object | None = None,
    ) -> None:
        if issuance_token is not _HANDLE_TOKEN:
            raise PermissionError("joint Loewner handles require their issuer")
        if type(receipt) not in (
            V1LiteralJJointAffineShellLoewnerStreamReceipt,
            V1LiteralJJointAffineShellPass1LoewnerReceipt,
            V1LiteralJJointAffineShellD1Pass1LoewnerReceipt,
            V1FactoredJointAffineShellLoewnerReceipt,  # allow-listed sibling
        ):
            raise TypeError("joint Loewner handle requires its exact receipt")
        required = (
            "small_center",
            "small_radius",
            "loewner_lower",
            "loewner_upper",
            "row_sum_radius",
        )
        if set(tensors) != set(required):
            raise TypeError("joint Loewner handle tensor roles are incomplete")
        receipt.assert_integrity()
        dimension = receipt.shell_size + 1
        expected_shapes = {
            "small_center": (receipt.action_count, dimension, dimension),
            "small_radius": (receipt.action_count, dimension, dimension),
            "loewner_lower": (receipt.action_count, dimension, dimension),
            "loewner_upper": (receipt.action_count, dimension, dimension),
            "row_sum_radius": (receipt.action_count, dimension),
        }
        expected_digests = {
            "small_center": receipt.small_matrix_center_digest,
            "small_radius": receipt.small_matrix_radius_digest,
            "loewner_lower": receipt.loewner_lower_digest,
            "loewner_upper": receipt.loewner_upper_digest,
            "row_sum_radius": receipt.row_sum_radius_digest,
        }
        device = None
        checked: dict[str, Any] = {}
        for role in required:
            tensor = _require_tensor(
                tensors[role],
                name=role,
                shape=expected_shapes[role],
                device=device,
                nonnegative=role in {"small_radius", "row_sum_radius"},
            )
            if device is None:
                device = tensor.device
            if _tensor_digest(tensor, role=role) != expected_digests[role]:
                _refuse(
                    V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.SOURCE_MISMATCH,
                    f"{role} differs from its issuer-sealed compact digest",
                )
            checked[role] = tensor
        if str(device) != receipt.measurement_device_identity:
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.DEVICE_MISMATCH,
                "small geometry left the measured source device",
            )
        self.receipt = receipt
        self._tensors = checked
        self._guards = {name: tensor.detach().clone() for name, tensor in checked.items()}
        self._invariants = {name: _tensor_invariant(tensor) for name, tensor in checked.items()}
        self._tokens: set[object] = set()
        self._evicted = False
        self.assert_live()

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.EVICTED,
                "joint Loewner handle was evicted",
            )
        self.receipt.assert_integrity()
        for role, tensor in self._tensors.items():
            if _tensor_invariant(tensor) != self._invariants[role] or not bool(
                torch.equal(tensor, self._guards[role])
            ):
                _refuse(
                    V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MUTATED_CUSTODY,
                    f"compact {role} tensor mutated after issuance",
                )

    def borrow(self) -> V1LiteralJJointAffineShellLoewnerStreamLease:
        self.assert_live()
        token = object()
        self._tokens.add(token)
        return V1LiteralJJointAffineShellLoewnerStreamLease(self, token, self._tensors)

    def _close_lease(self, token: object) -> None:
        if token not in self._tokens:
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.STALE_IDENTITY,
                "joint Loewner lease token is stale",
            )
        try:
            self.assert_live()
        finally:
            self._tokens.remove(token)

    def evict(self) -> None:
        if self._evicted:
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.EVICTED,
                "joint Loewner handle was already evicted",
            )
        if self._tokens:
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ACTIVE_LEASE,
                "cannot evict joint Loewner geometry under an active lease",
            )
        integrity_error: BaseException | None = None
        try:
            self.assert_live()
        except BaseException as error:
            integrity_error = error
        self._tensors = {}
        self._guards = {}
        self._invariants = {}
        self._evicted = True
        if integrity_error is not None:
            raise integrity_error

    def __reduce__(self) -> NoReturn:
        raise TypeError("joint Loewner handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1LiteralJJointAffineShellLoewnerStreamResult:
    receipt: (
        V1LiteralJJointAffineShellLoewnerStreamReceipt
        | V1LiteralJJointAffineShellPass1LoewnerReceipt
        | V1LiteralJJointAffineShellD1Pass1LoewnerReceipt
        | V1FactoredJointAffineShellLoewnerReceipt
    )
    device_handle: V1LiteralJJointAffineShellLoewnerStreamHandle = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            type(self.receipt)
            not in (
                V1LiteralJJointAffineShellLoewnerStreamReceipt,
                V1LiteralJJointAffineShellPass1LoewnerReceipt,
                V1LiteralJJointAffineShellD1Pass1LoewnerReceipt,
                V1FactoredJointAffineShellLoewnerReceipt,  # allow-listed sibling
            )
            or type(self.device_handle) is not V1LiteralJJointAffineShellLoewnerStreamHandle
        ):
            raise TypeError("joint Loewner result requires exact components")
        self.device_handle.assert_live()
        if self.device_handle.receipt.receipt_hash != self.receipt.receipt_hash:
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.STALE_IDENTITY,
                "joint Loewner receipt and live handle do not close",
            )


def _validate_lineage(
    *,
    preregistration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    d_fixed_results: tuple[V1LiteralJAffineShellShardReplayResult, ...],
    population: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
    allow_private_reference_for_tests: bool,
) -> tuple[object, object, tuple[object, ...], bool, bool]:
    exact = (
        (preregistration, V1LiteralJJointAffineShellPreoutcomeRegistration),
        (family_handle, V1LiteralJJointAffineShellFamilyDeviceHandle),
        (population, V1LiteralJDFixedResidualPopulationProjectionRegistration),
        (residual_stream, V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle),
    )
    if any(type(value) is not owner for value, owner in exact):
        raise TypeError("joint Loewner bridge received an inexact parent")
    if type(d_fixed_results) is not tuple or any(
        type(item) is not V1LiteralJAffineShellShardReplayResult
        for item in d_fixed_results
    ):
        raise TypeError("D_fixed custody must be one exact ordered affine-replay tuple")
    if type(allow_private_reference_for_tests) is not bool:
        raise TypeError("allow_private_reference_for_tests must be an exact bool")
    preregistration.assert_integrity()
    population.assert_integrity()
    residual_stream.assert_live()
    if residual_stream.active_tile:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ORDER_COVERAGE_MISMATCH,
            "a residual Mv tile was already active before bridge entry",
        )
    family = family_handle.receipt
    family.assert_integrity()
    stream_registration = residual_stream.registration
    stream_registration.assert_integrity()
    q_receipt = residual_stream.q_handle.receipt
    q_receipt.assert_integrity()
    production = q_receipt.production_cuda_execution
    common = (
        preregistration.model_identity_hash,
        preregistration.chart_identity_hash,
        preregistration.current_mask_hash,
        preregistration.proposal_generation,
        preregistration.route_axis_hash,
        preregistration.route_id_axis_hash,
        preregistration.route_count,
    )
    if (
        len(d_fixed_results) != preregistration.action_count
        or family.joint_preoutcome_registration_hash != preregistration.registration_hash
        or family.preoutcome_joint_direction_axis_hash != preregistration.joint_direction_axis_hash
        or family.joint_direction_ids != preregistration.joint_direction_ids
        or family.physical_action_axis_hash
        != preregistration.physical_action_axis_hash
        or family.physical_action_ordinals
        != preregistration.physical_action_ordinals
        or family.action_count != preregistration.action_count
        or family.shell_size != preregistration.shell_size
        or family.shell_indices != preregistration.shell_indices
        or family.shell_route_ids != preregistration.shell_route_ids
        or family.physical_action_identity_hashes != preregistration.physical_action_identity_hashes
        or family.physical_action_hashes != preregistration.physical_action_hashes
        or family.packed_action_hashes != preregistration.packed_action_hashes
        or family.historical_alpha_ledger_hash
        != preregistration.historical_alpha_ledger_hash
        or family.historical_alpha_coordinate_hash
        != preregistration.historical_alpha_coordinate_hash
        or family.historical_alpha_lineage_hash
        != preregistration.historical_alpha_lineage_hash
        or family.source_registration_hashes
        != preregistration.source_registration_hashes
        or family.preoutcome_registration_hashes
        != preregistration.preoutcome_registration_hashes
        or family.destination_binding_hashes
        != preregistration.destination_binding_hashes
        or population.direction_ids != preregistration.joint_direction_ids
        or population.d_fixed_projection_direction_axis_hash
        != preregistration.joint_direction_axis_hash
        or population.alpha_ledger_hash
        != preregistration.historical_alpha_ledger_hash
        or population.alpha_coordinate_hash
        != preregistration.historical_alpha_coordinate_hash
        or population.alpha_lineage_hash != preregistration.historical_alpha_lineage_hash
        or population.familywise_alpha != preregistration.familywise_alpha
        or (
            population.model_identity_hash,
            population.chart_identity_hash,
            population.current_mask_hash,
            population.proposal_generation,
            population.route_axis_hash,
            population.route_id_axis_hash,
            population.route_count,
        )
        != common
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.JOINT_AXIS_MISMATCH,
            "joint family, population registration, and preregistered action axis diverged",
        )
    joint_population_custody = (
        population.joint_preoutcome_registration_hash is not None
    )
    if joint_population_custody:
        replay_hashes = tuple(item.receipt.receipt_hash for item in d_fixed_results)
        direction_hashes = tuple(item.receipt.direction_receipt_hash for item in d_fixed_results)
        registration_hashes = tuple(
            item.registration.registration_hash for item in d_fixed_results
        )
        source_replay_hashes = tuple(
            item.source_bound_replay.receipt.receipt_hash
            for item in d_fixed_results
        )
        member_source_axes = tuple(
            item.registration.source_replay_axis_hash for item in d_fixed_results
        )
        if (
            population.joint_preoutcome_registration_hash
            != preregistration.registration_hash
            or population.joint_family_receipt_hash != family.receipt_hash
            or population.joint_postsource_custody_axis_hash
            != family.postsource_custody_axis_hash
            or population.joint_physical_action_axis_hash
            != preregistration.physical_action_axis_hash
            or population.joint_action_count != preregistration.action_count
            or population.direction_source_hash
            != family.postsource_custody_axis_hash
            or population.private_primal_direction_axis_hash
            != family.postsource_custody_axis_hash
            or population.joint_direction_receipt_hashes
            != family.direction_receipt_hashes
            or population.joint_direction_receipt_hashes != direction_hashes
            or population.joint_affine_shell_shard_replay_receipt_hashes
            != replay_hashes
            or population.joint_d_fixed_registration_hashes
            != registration_hashes
            or population.joint_d_fixed_source_replay_receipt_hashes
            != source_replay_hashes
            or population.d_fixed_shard_replay_receipt_hash is not None
            or population.affine_shell_shard_replay_receipt_hash is not None
            or population.d_fixed_registration_hash in registration_hashes
            or population.d_fixed_source_replay_axis_hash in member_source_axes
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.D_FIXED_AXIS_MISMATCH,
                "joint population custody does not replay the complete ordered D_fixed family",
            )
    elif production:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.AUTHORITY_VIOLATION,
            "production population geometry requires additive joint D_fixed custody; "
            "a legacy singleton registration cannot be promoted",
        )
    if (
        stream_registration.joint_preoutcome_registration_hash != preregistration.registration_hash
        or stream_registration.joint_family_receipt_hash != family.receipt_hash
        or stream_registration.joint_postsource_custody_axis_hash
        != family.postsource_custody_axis_hash
        or stream_registration.population_projection_registration_hash
        != population.registration_hash
        or stream_registration.joint_direction_ids != preregistration.joint_direction_ids
        or stream_registration.joint_direction_axis_hash != preregistration.joint_direction_axis_hash
        or stream_registration.residual_population_counts != population.residual_stratum_counts
        or stream_registration.familywise_alpha != preregistration.familywise_alpha
        or q_receipt.registration_hash != stream_registration.registration_hash
        or q_receipt.joint_direction_ids != preregistration.joint_direction_ids
        or q_receipt.joint_direction_axis_hash != preregistration.joint_direction_axis_hash
        or q_receipt.residual_population_counts != population.residual_stratum_counts
        or q_receipt.observed_counts != stream_registration.registered_observed_counts
        or q_receipt.familywise_alpha != preregistration.familywise_alpha
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.STALE_IDENTITY,
            "live residual Q/tile stream left its preregistered joint population family",
        )
    expected_d_ids = ("literal-j:outside",) + tuple(
        f"literal-j:shell-basis:{route_id}" for route_id in preregistration.shell_route_ids
    )
    common_source: tuple[object, ...] | None = None
    registrations: list[object] = []
    for index, result in enumerate(d_fixed_results):
        wrapper = result.receipt
        source = result.source_bound_replay
        registration = source.registration
        execution = source.projection
        wrapper.assert_integrity()
        source.receipt.assert_integrity()
        registration.assert_integrity()
        execution.receipt.assert_integrity()
        execution.device_handle.assert_live()
        if (
            wrapper.direction_receipt_hash != family.direction_receipt_hashes[index]
            or wrapper.preoutcome_registration_hash != family.preoutcome_registration_hashes[index]
            or wrapper.source_registration_hash != family.source_registration_hashes[index]
            or wrapper.postsource_receipt_hash != family.postsource_receipt_hashes[index]
            or wrapper.exact_postsource_custody_hash != family.exact_postsource_custody_hashes[index]
            or wrapper.projection_replay_registration_receipt_hash
            != family.projection_replay_registration_receipt_hashes[index]
            or wrapper.literal_j_cell_hash != family.literal_j_cell_hash
            or wrapper.literal_j_cell_scope_hash != family.literal_j_cell_scope_hash
            or wrapper.literal_j_stream_binding_hash != family.literal_j_stream_binding_hash
            or wrapper.shell_freeze_receipt_hash != preregistration.shell_freeze_receipt_hash
            or wrapper.shell_indices != preregistration.shell_indices
            or wrapper.shell_route_ids != preregistration.shell_route_ids
            or wrapper.alpha_lineage_hash != preregistration.historical_alpha_lineage_hash
            or wrapper.route_count != preregistration.route_count
            or registration.direction_ids != expected_d_ids
            or registration.direction_axis_hash != execution.receipt.direction_axis_hash
            or registration.direction_source_hash != wrapper.direction_source_hash
            or registration.structural_context_count != population.d_fixed_count
            or registration.structural_uniform_weight_fraction
            != (1, population.d_fixed_count)
            or registration.registered_d_prefix_hash != population.d_fixed_registered_prefix_hash
            or registration.registered_d_w2_prefix_hash != preregistration.registered_d_w2_prefix_hash
            or registration.composite_literal_j_stream_source_hash
            != preregistration.literal_j_stream_source_hash
            or registration.composite_same_capture_vjp_source_hash
            != preregistration.same_capture_vjp_source_hash
            or (
                registration.model_identity_hash,
                registration.chart_identity_hash,
                registration.current_mask_hash,
                registration.proposal_generation,
                registration.route_axis_hash,
                registration.route_id_axis_hash,
                registration.route_count,
            )
            != common
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.D_FIXED_AXIS_MISMATCH,
                f"D_fixed action replay {index + 1} is not its ordered joint-family member",
            )
        source_lineage = (
            registration.source_replay_axis_hash,
            registration.replay_registration_receipt_hash,
            registration.numeric_source_receipt_hash,
            registration.block_support_source_receipt_hash,
            registration.full_population_compact_source_receipt_hash,
            registration.structural_context_axis_hash,
            registration.structural_draw_axis_hash,
            registration.structural_measurement_axis_hash,
            registration.structural_capture_axis_hash,
            registration.structural_source_packet_axis_hash,
            source.receipt.conditioned_d1_replay_receipt_hash,
            source.receipt.residual_replay_receipt_hashes,
            source.receipt.packet_source_axis_hashes,
            source.receipt.same_capture_and_vjp_execution_proof_hash,
            source.receipt.context_count,
            source.receipt.route_count,
            source.receipt.packet_count,
            source.receipt.raw_shard_count,
        )
        if common_source is None:
            common_source = source_lineage
        elif source_lineage != common_source:
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.SOURCE_MISMATCH,
                "per-action D_fixed replays do not read one identical source census",
            )
        registrations.append(registration)
    if common_source is None:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.SOURCE_MISMATCH,
            "the ordered D_fixed replay family has no common source census",
        )
    if (
        not joint_population_custody
        and common_source[0] != population.d_fixed_source_replay_axis_hash
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.SOURCE_MISMATCH,
            "legacy private population registration left its singular D_fixed source axis",
        )
    d_modes = tuple(item.projection.telemetry.production_cuda_execution for item in d_fixed_results)
    if any(mode != production for mode in d_modes):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.DEVICE_MISMATCH,
            "D_fixed and residual executions mix CPU-reference and CUDA modes",
        )
    if not production and not allow_private_reference_for_tests:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.CPU_PRIVATE_REFERENCE_FORBIDDEN,
            "CPU execution is a private small-domain reference, not a production fallback",
        )
    if not production and preregistration.route_count > _CPU_REFERENCE_ROUTE_LIMIT:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.CPU_PRIVATE_REFERENCE_FORBIDDEN,
            "CPU reference route count exceeds the private small-domain cap",
        )
    return (
        family,
        q_receipt,
        tuple(registrations),
        production,
        joint_population_custody,
    )


def issue_v1_literal_j_joint_affine_shell_loewner_pass1_bridge(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    d_fixed_source_results: tuple[V1LiteralJAffineShellShardReplayResult, ...],
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
    allow_private_reference_for_tests: bool = False,
) -> V1LiteralJJointAffineShellLoewnerStreamResult:
    """Issue local population Loewner geometry without opening residual pass 2."""

    if torch is None:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.INVALID_TYPE,
            "pass-1 Loewner bridge requires PyTorch",
        )
    (
        family,
        q_receipt,
        d_registrations,
        production,
        joint_population_custody,
    ) = _validate_lineage(
        preregistration=joint_preoutcome_registration,
        family_handle=joint_family_handle,
        d_fixed_results=d_fixed_source_results,
        population=population_registration,
        residual_stream=residual_stream,
        allow_private_reference_for_tests=allow_private_reference_for_tests,
    )
    if not residual_stream.pass2_unopened:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ORDER_COVERAGE_MISMATCH,
            "the residual route-scale pass was already opened before pass-1 issuance",
        )
    compact_receipt = residual_stream.compact_shell_m_handle.receipt
    if type(compact_receipt) is not V1LiteralJJointAffineShellResidualPopulationStreamCompactShellMReceipt:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.INVALID_TYPE,
            "residual stream lost its exact pass-1 compact shell-M receipt",
        )
    compact_receipt.assert_integrity()
    prereg = joint_preoutcome_registration
    actions = prereg.action_count
    shell = prereg.shell_size
    directions = prereg.direction_count
    routes = prereg.route_count
    if (
        compact_receipt.registration_hash
        != residual_stream.registration.registration_hash
        or compact_receipt.q_receipt_hash != q_receipt.receipt_hash
        or compact_receipt.joint_preoutcome_registration_hash
        != prereg.registration_hash
        or compact_receipt.joint_family_receipt_hash != family.receipt_hash
        or compact_receipt.population_projection_registration_hash
        != population_registration.registration_hash
        or compact_receipt.joint_direction_axis_hash
        != prereg.joint_direction_axis_hash
        or compact_receipt.joint_direction_ids != prereg.joint_direction_ids
        or compact_receipt.shell_indices != prereg.shell_indices
        or compact_receipt.shell_route_ids != prereg.shell_route_ids
        or compact_receipt.residual_population_counts
        != population_registration.residual_stratum_counts
        or compact_receipt.observed_counts != q_receipt.observed_counts
        or compact_receipt.familywise_alpha != population_registration.familywise_alpha
        or compact_receipt.production_cuda_execution != production
        or compact_receipt.pass2_unopened_at_issuance is not True
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.STALE_IDENTITY,
            "pass-1 compact shell-M source left the registered joint population family",
        )
    with joint_family_handle.borrow() as family_lease:
        outside_vectors, live_shell = family_lease.projection_custody()
        device = outside_vectors[0].device
        shell_index = torch.tensor(
            prereg.shell_indices,
            dtype=torch.int64,
            device=device,
        )
        if (
            len(outside_vectors) != actions
            or any(value.device != device for value in outside_vectors)
            or live_shell.device != device
            or not bool(torch.equal(live_shell, shell_index))
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.DEVICE_MISMATCH,
                "live joint action/shell family left its registered device axis",
            )

    d_g_lower = d_g_upper = None
    d_h_lower = torch.empty((actions, shell), dtype=torch.float64, device=device)
    d_h_upper = torch.empty_like(d_h_lower)
    d_c_lower = torch.empty((actions,), dtype=torch.float64, device=device)
    d_c_upper = torch.empty_like(d_c_lower)
    for action, result in enumerate(d_fixed_source_results):
        with result.projection.device_handle.borrow() as lease:
            mjz_center = _require_tensor(
                lease.mjz_center,
                name="D_fixed Mv center",
                shape=(shell + 1, routes),
                device=device,
            )
            mjz_radius = _require_tensor(
                lease.mjz_radius,
                name="D_fixed Mv radius",
                shape=(shell + 1, routes),
                device=device,
                nonnegative=True,
            )
            q_center = _require_tensor(
                lease.q_center,
                name="D_fixed Q center",
                shape=(shell + 1,),
                device=device,
            )
            q_radius = _require_tensor(
                lease.q_radius,
                name="D_fixed Q radius",
                shape=(shell + 1,),
                device=device,
                nonnegative=True,
            )
            component = _extract_component_bounds(
                shell_m_center=mjz_center.index_select(1, shell_index).contiguous(),
                shell_m_radius=mjz_radius.index_select(1, shell_index).contiguous(),
                q_center=q_center,
                q_radius=q_radius,
                action_count=1,
                shell_size=shell,
            )
        g_lower, g_upper, h_lower, h_upper, c_lower, c_upper = component
        if d_g_lower is None:
            d_g_lower, d_g_upper = g_lower, g_upper
        else:
            d_g_lower, d_g_upper = _intersect(
                d_g_lower,
                d_g_upper,
                g_lower,
                g_upper,
            )
        d_h_lower[action] = h_lower[0]
        d_h_upper[action] = h_upper[0]
        d_c_lower[action] = c_lower[0]
        d_c_upper[action] = c_upper[0]
    if d_g_lower is None or d_g_upper is None:
        raise AssertionError("validated pass-1 D_fixed action axis disappeared")

    with residual_stream.q_handle.borrow() as q_lease:
        residual_q_center, residual_q_radius = q_lease.intervals()
        residual_q_center = _require_tensor(
            residual_q_center,
            name="residual Q center",
            shape=(2, directions),
            device=device,
        )
        residual_q_radius = _require_tensor(
            residual_q_radius,
            name="residual Q radius",
            shape=(2, directions),
            device=device,
            nonnegative=True,
        )
        with residual_stream.compact_shell_m_handle.borrow() as compact_lease:
            residual_m_center, residual_m_radius = compact_lease.intervals()
            residual_m_center = _require_tensor(
                residual_m_center,
                name="pass-1 residual shell Mv center",
                shape=(2, directions, shell),
                device=device,
            )
            residual_m_radius = _require_tensor(
                residual_m_radius,
                name="pass-1 residual shell Mv radius",
                shape=(2, directions, shell),
                device=device,
                nonnegative=True,
            )
            residual_components = tuple(
                _extract_component_bounds(
                    shell_m_center=residual_m_center[h],
                    shell_m_radius=residual_m_radius[h],
                    q_center=residual_q_center[h],
                    q_radius=residual_q_radius[h],
                    action_count=actions,
                    shell_size=shell,
                )
                for h in (0, 1)
            )

    d_weight = Fraction(*population_registration.d_fixed_mass_fraction)
    residual_weights = tuple(
        Fraction(*value)
        for value in population_registration.residual_stratum_mass_fractions
    )
    target, complement = residual_components
    g_lower, g_upper = _mix_population_bounds(
        (
            (d_g_lower, d_g_upper, d_weight),
            (target[0], target[1], residual_weights[0]),
            (complement[0], complement[1], residual_weights[1]),
        )
    )
    h_lower, h_upper = _mix_population_bounds(
        (
            (d_h_lower, d_h_upper, d_weight),
            (target[2], target[3], residual_weights[0]),
            (complement[2], complement[3], residual_weights[1]),
        )
    )
    c_lower, c_upper = _mix_population_bounds(
        (
            (d_c_lower, d_c_upper, d_weight),
            (target[4], target[5], residual_weights[0]),
            (complement[4], complement[5], residual_weights[1]),
        )
    )
    center, radius, lower, upper, row_sum = _loewner_from_bounds(
        g_lower=g_lower,
        g_upper=g_upper,
        h_lower=h_lower,
        h_upper=h_upper,
        c_lower=c_lower,
        c_upper=c_upper,
    )
    if not residual_stream.pass2_unopened:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ORDER_COVERAGE_MISMATCH,
            "pass-1 Loewner issuance opened the residual route-scale pass",
        )
    max_row_sum = float(torch.max(row_sum).detach().cpu().item())
    if not math.isfinite(max_row_sum) or max_row_sum < 0.0:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.NUMERICAL_REFUSAL,
            "pass-1 maximum row-sum radius is not finite and nonnegative",
        )
    compact_tensor_scalars = int(
        center.numel()
        + radius.numel()
        + lower.numel()
        + upper.numel()
        + row_sum.numel()
    )
    compact_scalars = 0 if not production else 2 * compact_tensor_scalars + 1
    receipt = V1LiteralJJointAffineShellPass1LoewnerReceipt(
        joint_preoutcome_registration_hash=prereg.registration_hash,
        joint_family_receipt_hash=family.receipt_hash,
        joint_postsource_custody_axis_hash=family.postsource_custody_axis_hash,
        population_projection_registration_hash=population_registration.registration_hash,
        residual_stream_registration_hash=residual_stream.registration.registration_hash,
        residual_q_receipt_hash=q_receipt.receipt_hash,
        residual_pass1_compact_shell_m_receipt_hash=compact_receipt.receipt_hash,
        residual_mv_coordinate_subset_hash=compact_receipt.mv_coordinate_subset_hash,
        d_fixed_affine_replay_receipt_hashes=tuple(
            value.receipt.receipt_hash for value in d_fixed_source_results
        ),
        d_fixed_registration_hashes=tuple(
            value.registration.registration_hash for value in d_fixed_source_results
        ),
        d_fixed_source_replay_receipt_hashes=tuple(
            value.source_bound_replay.receipt.receipt_hash
            for value in d_fixed_source_results
        ),
        d_fixed_projection_receipt_hashes=tuple(
            value.projection.receipt.receipt_hash for value in d_fixed_source_results
        ),
        d_fixed_common_source_replay_axis_hash=d_registrations[0].source_replay_axis_hash,
        d_fixed_population_source_replay_axis_hash=population_registration.d_fixed_source_replay_axis_hash,
        model_identity_hash=prereg.model_identity_hash,
        chart_identity_hash=prereg.chart_identity_hash,
        current_mask_hash=prereg.current_mask_hash,
        proposal_generation=prereg.proposal_generation,
        route_axis_hash=prereg.route_axis_hash,
        route_id_axis_hash=prereg.route_id_axis_hash,
        route_count=routes,
        registered_d_prefix_hash=prereg.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=prereg.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=prereg.shell_freeze_receipt_hash,
        shell_indices=prereg.shell_indices,
        shell_route_ids=prereg.shell_route_ids,
        action_count=actions,
        shell_size=shell,
        physical_action_identity_hashes=prereg.physical_action_identity_hashes,
        physical_action_hashes=prereg.physical_action_hashes,
        packed_action_hashes=prereg.packed_action_hashes,
        joint_direction_ids=prereg.joint_direction_ids,
        joint_direction_axis_hash=prereg.joint_direction_axis_hash,
        small_axis_ids=tuple(
            prereg.shared_shell_direction_ids
            + (prereg.outside_direction_ids[action],)
            for action in range(actions)
        ),
        d_fixed_stratum_counts=population_registration.d_fixed_stratum_counts,
        residual_population_counts=population_registration.residual_stratum_counts,
        residual_observed_counts=compact_receipt.observed_counts,
        population_count=population_registration.population_count,
        d_fixed_count=population_registration.d_fixed_count,
        d_fixed_mass_fraction=population_registration.d_fixed_mass_fraction,
        residual_mass_fractions=population_registration.residual_stratum_mass_fractions,
        alpha_ledger_hash=population_registration.alpha_ledger_hash,
        alpha_coordinate_hash=population_registration.alpha_coordinate_hash,
        alpha_lineage_hash=population_registration.alpha_lineage_hash,
        inherited_familywise_alpha=population_registration.familywise_alpha,
        inherited_alpha_spent=(
            0.0
            if compact_receipt.observed_counts
            == compact_receipt.residual_population_counts
            else population_registration.familywise_alpha
        ),
        additional_alpha_spent=0.0,
        residual_exact_census=(
            compact_receipt.observed_counts
            == compact_receipt.residual_population_counts
        ),
        production_cuda_execution=production,
        private_cpu_reference_execution=not production,
        measurement_device_identity=str(device),
        compact_tensor_d2h_scalar_count=compact_scalars,
        compact_tensor_d2h_bytes=compact_scalars * _F64_BYTES,
        small_matrix_center_digest=_tensor_digest(center, role="small_center"),
        small_matrix_radius_digest=_tensor_digest(radius, role="small_radius"),
        loewner_lower_digest=_tensor_digest(lower, role="loewner_lower"),
        loewner_upper_digest=_tensor_digest(upper, role="loewner_upper"),
        row_sum_radius_digest=_tensor_digest(row_sum, role="row_sum_radius"),
        max_row_sum_radius_upper_bound=max_row_sum,
        joint_action_complete_population_custody=joint_population_custody,
        legacy_private_population_registration=not joint_population_custody,
        population_loewner_authority=production,
        kkt_geometry_uncertainty_input_authority=production,
        _issuance_token=_PASS1_RECEIPT_TOKEN,
    )
    handle = V1LiteralJJointAffineShellLoewnerStreamHandle(
        receipt=receipt,
        tensors={
            "small_center": center,
            "small_radius": radius,
            "loewner_lower": lower,
            "loewner_upper": upper,
            "row_sum_radius": row_sum,
        },
        issuance_token=_HANDLE_TOKEN,
    )
    return V1LiteralJJointAffineShellLoewnerStreamResult(
        receipt=receipt,
        device_handle=handle,
    )


def _validate_d1_atomic_population_sources_nonauthoritative(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    atomic_population_sources: tuple[V1LiteralJJointPopulationSource, ...],
    d1_stream: V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, int],
    tuple[int, int],
    tuple[tuple[int, int], tuple[int, int]],
]:
    """Validate an ordered atomic tuple without granting aggregate authority."""

    exact = (
        (joint_preoutcome_registration, V1LiteralJJointAffineShellPreoutcomeRegistration),
        (joint_family_handle, V1LiteralJJointAffineShellFamilyDeviceHandle),
        (d1_stream, V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle),
        (residual_stream, V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle),
    )
    if any(type(value) is not owner for value, owner in exact):
        raise TypeError("D1/residual Loewner reference received an inexact stream parent")
    if type(atomic_population_sources) is not tuple or any(
        type(item) is not V1LiteralJJointPopulationSource
        for item in atomic_population_sources
    ):
        raise TypeError("atomic population sources must be one exact immutable tuple")
    prereg = joint_preoutcome_registration
    prereg.assert_integrity()
    family = joint_family_handle.receipt
    family.assert_integrity()
    d1_stream.assert_live()
    residual_stream.assert_live()
    if len(atomic_population_sources) != prereg.action_count:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ORDER_COVERAGE_MISMATCH,
            "the atomic source tuple does not cover every preregistered action",
        )
    if (
        family.joint_preoutcome_registration_hash != prereg.registration_hash
        or family.preoutcome_joint_direction_axis_hash
        != prereg.joint_direction_axis_hash
        or family.joint_direction_ids != prereg.joint_direction_ids
        or family.action_count != prereg.action_count
        or family.shell_size != prereg.shell_size
        or family.shell_indices != prereg.shell_indices
        or family.shell_route_ids != prereg.shell_route_ids
        or family.physical_action_identity_hashes
        != prereg.physical_action_identity_hashes
        or family.physical_action_hashes != prereg.physical_action_hashes
        or family.packed_action_hashes != prereg.packed_action_hashes
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.JOINT_AXIS_MISMATCH,
            "joint family left the preregistered action/direction axis",
        )
    d1_receipt = d1_stream.receipt
    d1_q = d1_stream.q_handle.receipt
    d1_compact = d1_stream.compact_shell_m_handle.receipt
    residual_registration = residual_stream.registration
    residual_q = residual_stream.q_handle.receipt
    residual_compact = residual_stream.compact_shell_m_handle.receipt
    for value in (
        d1_receipt,
        d1_q,
        d1_compact,
        residual_registration,
        residual_q,
        residual_compact,
    ):
        value.assert_integrity()
    common_chart = (
        prereg.model_identity_hash,
        prereg.chart_identity_hash,
        prereg.current_mask_hash,
        prereg.proposal_generation,
        prereg.route_axis_hash,
        prereg.route_id_axis_hash,
        prereg.route_count,
    )
    if (
        d1_receipt.joint_preoutcome_registration_hash != prereg.registration_hash
        or d1_receipt.joint_family_receipt_hash != family.receipt_hash
        or d1_receipt.joint_postsource_custody_axis_hash
        != family.postsource_custody_axis_hash
        or d1_receipt.joint_direction_axis_hash != prereg.joint_direction_axis_hash
        or d1_receipt.direction_ids != prereg.joint_direction_ids
        or d1_receipt.action_count != prereg.action_count
        or d1_receipt.shell_size != prereg.shell_size
        or d1_receipt.shell_indices != prereg.shell_indices
        or d1_receipt.shell_route_ids != prereg.shell_route_ids
        or (
            d1_receipt.model_identity_hash,
            d1_receipt.chart_identity_hash,
            d1_receipt.current_mask_hash,
            d1_receipt.proposal_generation,
            d1_receipt.route_axis_hash,
            d1_receipt.route_id_axis_hash,
            d1_receipt.route_count,
        )
        != common_chart
        or d1_q.stream_receipt_hash != d1_receipt.receipt_hash
        or d1_q.joint_direction_axis_hash != prereg.joint_direction_axis_hash
        or d1_q.direction_ids != prereg.joint_direction_ids
        or d1_compact.stream_receipt_hash != d1_receipt.receipt_hash
        or d1_compact.q_receipt_hash != d1_q.receipt_hash
        or d1_compact.joint_direction_axis_hash != prereg.joint_direction_axis_hash
        or d1_compact.direction_ids != prereg.joint_direction_ids
        or d1_compact.shell_indices != prereg.shell_indices
        or d1_compact.shell_route_ids != prereg.shell_route_ids
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.D_FIXED_AXIS_MISMATCH,
            "D1 stream left the exact joint action, shell, chart, or source axis",
        )
    if (
        residual_registration.joint_preoutcome_registration_hash
        != prereg.registration_hash
        or residual_registration.joint_family_receipt_hash != family.receipt_hash
        or residual_registration.joint_postsource_custody_axis_hash
        != family.postsource_custody_axis_hash
        or residual_registration.joint_direction_axis_hash
        != prereg.joint_direction_axis_hash
        or residual_registration.joint_direction_ids != prereg.joint_direction_ids
        or residual_registration.action_count != prereg.action_count
        or residual_registration.shell_size != prereg.shell_size
        or residual_registration.shell_indices != prereg.shell_indices
        or (
            residual_registration.model_identity_hash,
            residual_registration.chart_identity_hash,
            residual_registration.current_mask_hash,
            residual_registration.proposal_generation,
            residual_registration.route_axis_hash,
            residual_registration.route_id_axis_hash,
            residual_registration.route_count,
        )
        != common_chart
        or residual_q.registration_hash != residual_registration.registration_hash
        or residual_q.joint_direction_axis_hash != prereg.joint_direction_axis_hash
        or residual_q.joint_direction_ids != prereg.joint_direction_ids
        or residual_compact.registration_hash
        != residual_registration.registration_hash
        or residual_compact.q_receipt_hash != residual_q.receipt_hash
        or residual_compact.joint_direction_axis_hash
        != prereg.joint_direction_axis_hash
        or residual_compact.joint_direction_ids != prereg.joint_direction_ids
        or residual_compact.shell_indices != prereg.shell_indices
        or residual_compact.shell_route_ids != prereg.shell_route_ids
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.SOURCE_MISMATCH,
            "residual stream left the exact joint action, shell, chart, or source axis",
        )
    if (
        not residual_stream.pass2_unopened
        or d1_stream._cursor.route_tile_read_count != 0
        or d1_stream._exhausted
        or d1_stream._active_tile_hash is not None
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ORDER_COVERAGE_MISMATCH,
            "D1 or residual route-scale pass was opened before compact Loewner issuance",
        )

    source_hashes: list[str] = []
    pre_hashes: list[str] = []
    first_logical_result: object | None = None
    common_pre: object | None = None
    for action, source in enumerate(atomic_population_sources):
        try:
            source_hash = source.assert_live()
            pre = source.preoutcome_receipt
            pre_hash = pre.assert_integrity()
        except Exception as error:
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MUTATED_CUSTODY,
                "one atomic joint-population source no longer replays: "
                f"{type(error).__name__}: {error}",
            )
        if first_logical_result is None:
            first_logical_result = source.logical_result
            common_pre = pre
        if (
            source.logical_result is not first_logical_result
            or pre.physical_action_ordinal != action + 1
            or pre.physical_action_identity_hash
            != prereg.physical_action_identity_hashes[action]
            or pre.physical_action_hash != prereg.physical_action_hashes[action]
            or pre.packed_action_hash != prereg.packed_action_hashes[action]
            or pre.joint_preoutcome_registration_hash != prereg.registration_hash
            or pre.joint_direction_axis_hash != prereg.joint_direction_axis_hash
            or pre.joint_direction_ids != prereg.joint_direction_ids
            or pre.d1_replay_receipt_hash
            != d1_receipt.conditioned_d1_replay_receipt_hash
            or pre.d1_compact_source_receipt_hash
            != d1_receipt.compact_source_receipt_hash
            or pre.d1_literal_j_shard_receipt_hashes
            != d1_receipt.literal_j_shard_receipt_hashes
            or pre.d1_literal_j_content_sha256s
            != d1_receipt.literal_j_content_sha256s
            or pre.d1_context_count != d1_receipt.context_count
            or pre.d1_stratum_counts != d1_receipt.d1_stratum_counts
            or pre.residual_sample_stratum_counts
            != residual_registration.registered_observed_counts
            or tuple(
                pre.population_stratum_counts[h] - pre.d1_stratum_counts[h]
                for h in (0, 1)
            )
            != residual_registration.residual_population_counts
            or (
                pre.model_identity_hash,
                pre.chart_identity_hash,
                pre.current_mask_hash,
                pre.proposal_generation,
                pre.route_axis_hash,
                pre.route_id_axis_hash,
                pre.route_count,
            )
            != common_chart
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ORDER_COVERAGE_MISMATCH,
                f"atomic source {action + 1} is not its ordered joint action member",
            )
        if common_pre is not None and any(
            getattr(pre, name) != getattr(common_pre, name)
            for name in (
                "logical_source_receipt_hash",
                "d1_source_axis_hash",
                "measure_design_hash",
                "measure_weight_receipt_hash",
                "joint_preoutcome_registration_hash",
                "joint_direction_axis_hash",
                "d1_context_count",
                "d1_stratum_counts",
                "residual_sample_stratum_counts",
                "population_stratum_counts",
                "d1_context_weight_fraction",
                "residual_context_weight_fractions",
                "residual_stratum_mass_fractions",
                "residual_population_mass_fraction",
            )
        ):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.SOURCE_MISMATCH,
                "atomic actions do not share one exact D1/residual population measure",
            )
        source_hashes.append(source_hash)
        pre_hashes.append(pre_hash)
    assert common_pre is not None
    if (
        residual_stream._segmented_source_result is not None
        and residual_stream._segmented_source_result is not first_logical_result
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.SOURCE_MISMATCH,
            "atomic sources do not bind the exact residual stream logical result",
        )
    population_count = sum(common_pre.population_stratum_counts)
    d1_mass = Fraction(d1_receipt.context_count, population_count)
    residual_masses = tuple(
        Fraction(value, population_count)
        for value in residual_registration.residual_population_counts
    )
    if (
        Fraction(*d1_receipt.unapplied_d1_population_mass_fraction) != d1_mass
        or tuple(Fraction(*value) for value in common_pre.residual_stratum_mass_fractions)
        != residual_masses
        or d1_mass + residual_masses[0] + residual_masses[1] != 1
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MASS_COUNT_MISMATCH,
            "atomic source, D1 stream, and residual stream masses do not close",
        )
    return (
        tuple(source_hashes),
        tuple(pre_hashes),
        common_pre.population_stratum_counts,
        d1_receipt.d1_stratum_counts,
        tuple(
            (value.numerator, value.denominator) for value in residual_masses
        ),
    )


def _validate_d1_population_measure_ancestry(
    *,
    population_measure_registration: V1LiteralJJointAxisResidualPopulationMeasureRegistration,
    action_axis_source: V1LiteralJJointPopulationActionAxisSource,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    d1_stream: V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
) -> V1LiteralJJointAxisResidualPopulationMeasureRegistrationReceipt:
    """Close the production D1/residual graph over one exact measure owner."""

    if (
        type(population_measure_registration)
        is not V1LiteralJJointAxisResidualPopulationMeasureRegistration
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.INVALID_TYPE,
            "production D1 Loewner issuance requires the exact joint-axis "
            "population-measure owner",
        )
    try:
        measure_hash = population_measure_registration.assert_live()
        measure = population_measure_registration.receipt
        measure.assert_integrity(measure_hash)
    except Exception as error:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MUTATED_CUSTODY,
            "the joint-axis population-measure owner no longer replays: "
            f"{type(error).__name__}: {error}",
        )
    if (
        type(measure)
        is not V1LiteralJJointAxisResidualPopulationMeasureRegistrationReceipt
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.INVALID_TYPE,
            "joint-axis population-measure custody has an inexact receipt",
        )

    aggregate = action_axis_source.receipt
    pre = action_axis_source.preoutcome_source.receipt
    prereg = joint_preoutcome_registration
    family = joint_family_handle.receipt
    d1 = d1_stream.receipt
    residual = residual_stream.registration

    action_axis_pairs = (
        (measure.action_axis_preoutcome_receipt_hash, pre.receipt_hash),
        (aggregate.preoutcome_action_axis_receipt_hash, pre.receipt_hash),
        (measure.joint_preoutcome_registration_hash, prereg.registration_hash),
        (aggregate.joint_preoutcome_registration_hash, prereg.registration_hash),
        (pre.joint_preoutcome_registration_hash, prereg.registration_hash),
        (family.joint_preoutcome_registration_hash, prereg.registration_hash),
        (d1.joint_preoutcome_registration_hash, prereg.registration_hash),
        (residual.joint_preoutcome_registration_hash, prereg.registration_hash),
        (measure.physical_action_axis_hash, prereg.physical_action_axis_hash),
        (aggregate.physical_action_axis_hash, prereg.physical_action_axis_hash),
        (pre.physical_action_axis_hash, prereg.physical_action_axis_hash),
        (family.physical_action_axis_hash, prereg.physical_action_axis_hash),
        (
            measure.physical_action_identity_hashes,
            prereg.physical_action_identity_hashes,
        ),
        (
            aggregate.physical_action_identity_hashes,
            prereg.physical_action_identity_hashes,
        ),
        (pre.physical_action_identity_hashes, prereg.physical_action_identity_hashes),
        (family.physical_action_identity_hashes, prereg.physical_action_identity_hashes),
        (d1.physical_action_identity_hashes, prereg.physical_action_identity_hashes),
        (measure.physical_action_hashes, prereg.physical_action_hashes),
        (aggregate.physical_action_hashes, prereg.physical_action_hashes),
        (pre.physical_action_hashes, prereg.physical_action_hashes),
        (family.physical_action_hashes, prereg.physical_action_hashes),
        (d1.physical_action_hashes, prereg.physical_action_hashes),
        (measure.packed_action_hashes, prereg.packed_action_hashes),
        (aggregate.packed_action_hashes, prereg.packed_action_hashes),
        (pre.packed_action_hashes, prereg.packed_action_hashes),
        (family.packed_action_hashes, prereg.packed_action_hashes),
        (d1.packed_action_hashes, prereg.packed_action_hashes),
        (measure.joint_direction_axis_hash, prereg.joint_direction_axis_hash),
        (aggregate.joint_direction_axis_hash, prereg.joint_direction_axis_hash),
        (pre.joint_direction_axis_hash, prereg.joint_direction_axis_hash),
        (family.preoutcome_joint_direction_axis_hash, prereg.joint_direction_axis_hash),
        (d1.joint_direction_axis_hash, prereg.joint_direction_axis_hash),
        (residual.joint_direction_axis_hash, prereg.joint_direction_axis_hash),
        (measure.joint_direction_ids, prereg.joint_direction_ids),
        (pre.joint_direction_ids, prereg.joint_direction_ids),
        (family.joint_direction_ids, prereg.joint_direction_ids),
        (d1.direction_ids, prereg.joint_direction_ids),
        (residual.joint_direction_ids, prereg.joint_direction_ids),
    )
    if any(left != right for left, right in action_axis_pairs):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.JOINT_AXIS_MISMATCH,
            "population measure, PRE/POST aggregate, family, D1, and residual "
            "parents do not share one physical action-plus-shell axis",
        )

    source_pairs = (
        (measure.population_measure_design_hash, pre.measure_design_hash),
        (measure.population_measure_weight_receipt_hash, pre.measure_weight_receipt_hash),
        (measure.d1_source_axis_hash, pre.d1_source_axis_hash),
        (measure.d1_source_axis_hash, d1.source_axis_hash),
        (measure.d1_source_registration_hash, pre.d1_source_registration_hash),
        (measure.d1_registered_prefix_hash, d1.registered_d1_capture_prefix_hash),
        (measure.d1_pair_prefix_axis_hash, d1.d1_pair_prefix_axis_hash),
        (measure.registered_d_prefix_hash, prereg.registered_d_prefix_hash),
        (measure.registered_d_prefix_hash, d1.joint_registered_d_prefix_hash),
        (measure.registered_d_w2_prefix_hash, prereg.registered_d_w2_prefix_hash),
        (measure.registered_d_w2_prefix_hash, residual.registered_d_w2_prefix_hash),
        (measure.d1_literal_j_stream_source_hash, pre.d1_literal_j_stream_source_hash),
        (measure.d1_literal_j_stream_source_hash, d1.d1_literal_j_stream_source_hash),
        (measure.d1_same_capture_vjp_source_hash, pre.d1_same_capture_vjp_source_hash),
        (measure.d1_same_capture_vjp_source_hash, prereg.same_capture_vjp_source_hash),
        (measure.residual_literal_j_stream_source_hash, residual.literal_j_stream_source_hash),
        (measure.suffix_registration_hash, residual.suffix_registration_hash),
        (
            measure.residual_sealed_population_order_hashes,
            residual.residual_sealed_population_order_hashes,
        ),
        (family.receipt_hash, d1.joint_family_receipt_hash),
        (family.receipt_hash, residual.joint_family_receipt_hash),
        (family.postsource_custody_axis_hash, d1.joint_postsource_custody_axis_hash),
        (
            family.postsource_custody_axis_hash,
            residual.joint_postsource_custody_axis_hash,
        ),
        (
            residual.population_projection_registration_hash,
            measure_hash,
        ),
        (
            residual.joint_population_action_axis_preoutcome_receipt_hash,
            pre.receipt_hash,
        ),
        (
            residual.joint_population_action_axis_postsource_receipt_hash,
            aggregate.receipt_hash,
        ),
    )
    if any(left != right for left, right in source_pairs):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.SOURCE_MISMATCH,
            "population measure, action aggregate, family, D1, and residual "
            "parents do not descend from one preregistered source graph",
        )

    count_and_mass_pairs = (
        (measure.action_count, prereg.action_count),
        (measure.shell_size, prereg.shell_size),
        (measure.direction_count, prereg.direction_count),
        (measure.d1_context_count, d1.context_count),
        (measure.d1_stratum_counts, d1.d1_stratum_counts),
        (measure.population_count, d1.population_count),
        (measure.d1_mass_fraction, d1.unapplied_d1_population_mass_fraction),
        (
            measure.residual_population_stratum_counts,
            residual.residual_population_counts,
        ),
        (
            measure.registered_residual_sample_stratum_counts,
            residual.registered_observed_counts,
        ),
        (measure.population_stratum_counts, pre.population_stratum_counts),
        (measure.d1_stratum_counts, pre.d1_stratum_counts),
        (measure.d1_context_count, pre.d1_context_count),
        (measure.d1_context_weight_fraction, pre.d1_context_weight_fraction),
        (
            measure.residual_context_weight_fractions,
            pre.residual_context_weight_fractions,
        ),
        (
            measure.residual_stratum_mass_fractions,
            pre.residual_stratum_mass_fractions,
        ),
        (
            measure.residual_population_mass_fraction,
            pre.residual_population_mass_fraction,
        ),
        (measure.common_measure_denominator, pre.common_measure_denominator),
        (measure.d1_context_quantum, pre.d1_context_quantum),
        (measure.residual_context_quanta, pre.residual_context_quanta),
        (measure.total_measure_quanta, pre.total_measure_quanta),
    )
    if any(left != right for left, right in count_and_mass_pairs):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MASS_COUNT_MISMATCH,
            "population measure, PRE aggregate, D1, and residual exact counts "
            "or rational masses do not close",
        )

    alpha_pairs = (
        (measure.alpha_ledger_hash, prereg.historical_alpha_ledger_hash),
        (measure.alpha_ledger_hash, family.historical_alpha_ledger_hash),
        (measure.alpha_coordinate_hash, prereg.historical_alpha_coordinate_hash),
        (measure.alpha_coordinate_hash, family.historical_alpha_coordinate_hash),
        (measure.alpha_lineage_hash, prereg.historical_alpha_lineage_hash),
        (measure.alpha_lineage_hash, family.historical_alpha_lineage_hash),
        (measure.alpha_lineage_hash, pre.alpha_lineage_hash),
        (measure.familywise_alpha, prereg.familywise_alpha),
        (measure.familywise_alpha, family.familywise_alpha),
        (measure.familywise_alpha, residual.familywise_alpha),
    )
    if any(left != right for left, right in alpha_pairs):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ALPHA_MISMATCH,
            "population measure, preregistration, family, PRE aggregate, and "
            "residual estimator do not share one alpha cell",
        )
    return measure


def _issue_v1_literal_j_joint_affine_shell_d1_loewner_pass1_bridge_cpu_reference(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    atomic_population_sources: tuple[V1LiteralJJointPopulationSource, ...],
    d1_stream: V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
    _action_axis_source: V1LiteralJJointPopulationActionAxisSource | None = None,
    _population_measure_receipt: V1LiteralJJointAxisResidualPopulationMeasureRegistrationReceipt
    | None = None,
    _production: bool = False,
) -> V1LiteralJJointAffineShellLoewnerStreamResult:
    """Shared issuer; the public wrapper supplies exact aggregate custody."""

    if torch is None:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.INVALID_TYPE,
            "D1 pass-1 Loewner reference requires PyTorch",
        )
    (
        atomic_hashes,
        atomic_pre_hashes,
        population_strata,
        d1_strata,
        residual_mass_pairs,
    ) = _validate_d1_atomic_population_sources_nonauthoritative(
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        atomic_population_sources=atomic_population_sources,
        d1_stream=d1_stream,
        residual_stream=residual_stream,
    )
    prereg = joint_preoutcome_registration
    family = joint_family_handle.receipt
    d1_receipt = d1_stream.receipt
    d1_q_receipt = d1_stream.q_handle.receipt
    d1_compact_receipt = d1_stream.compact_shell_m_handle.receipt
    residual_q_receipt = residual_stream.q_handle.receipt
    residual_compact_receipt = residual_stream.compact_shell_m_handle.receipt
    actions = prereg.action_count
    shell = prereg.shell_size
    directions = prereg.direction_count
    device = d1_stream._projection.device
    if type(_production) is not bool:
        raise TypeError("_production must be exact")
    if _production != (_population_measure_receipt is not None):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.AUTHORITY_VIOLATION,
            "production D1 Loewner numerics require exact population-measure ancestry",
        )
    if _production and _action_axis_source is None:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.AUTHORITY_VIOLATION,
            "production D1 Loewner numerics require exact action-complete custody",
        )
    if (
        residual_stream._projection_centers[0].device != device
        or residual_stream.q_handle.receipt.production_cuda_execution != _production
        or d1_receipt.production_cuda_execution != _production
        or (_production and device.type != "cuda")
        or (not _production and device.type != "cpu")
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.CPU_PRIVATE_REFERENCE_FORBIDDEN,
            "D1 and residual compact streams do not share the requested execution device",
        )
    if not _production and prereg.route_count > _CPU_REFERENCE_ROUTE_LIMIT:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.CPU_PRIVATE_REFERENCE_FORBIDDEN,
            "D1 pass-1 CPU reference exceeds its bounded route domain",
        )
    with d1_stream.q_handle.borrow() as d1_q_lease:
        d1_q_center, d1_q_radius = d1_q_lease.intervals()
        d1_q_center = _require_tensor(
            d1_q_center,
            name="D1 Q center",
            shape=(directions,),
            device=device,
        )
        d1_q_radius = _require_tensor(
            d1_q_radius,
            name="D1 Q numeric radius",
            shape=(directions,),
            device=device,
            nonnegative=True,
        )
        with d1_stream.compact_shell_m_handle.borrow() as d1_m_lease:
            d1_m_center, d1_m_radius = d1_m_lease.intervals()
            d1_m_center = _require_tensor(
                d1_m_center,
                name="D1 compact shell-M center",
                shape=(directions, shell),
                device=device,
            )
            d1_m_radius = _require_tensor(
                d1_m_radius,
                name="D1 compact shell-M numeric radius",
                shape=(directions, shell),
                device=device,
                nonnegative=True,
            )
            d1_component = _extract_component_bounds(
                shell_m_center=d1_m_center,
                shell_m_radius=d1_m_radius,
                q_center=d1_q_center,
                q_radius=d1_q_radius,
                action_count=actions,
                shell_size=shell,
            )
    with residual_stream.q_handle.borrow() as residual_q_lease:
        residual_q_center, residual_q_radius = residual_q_lease.intervals()
        residual_q_center = _require_tensor(
            residual_q_center,
            name="residual Q center",
            shape=(2, directions),
            device=device,
        )
        residual_q_radius = _require_tensor(
            residual_q_radius,
            name="residual Q radius",
            shape=(2, directions),
            device=device,
            nonnegative=True,
        )
        with residual_stream.compact_shell_m_handle.borrow() as residual_m_lease:
            residual_m_center, residual_m_radius = residual_m_lease.intervals()
            residual_m_center = _require_tensor(
                residual_m_center,
                name="residual compact shell-M center",
                shape=(2, directions, shell),
                device=device,
            )
            residual_m_radius = _require_tensor(
                residual_m_radius,
                name="residual compact shell-M radius",
                shape=(2, directions, shell),
                device=device,
                nonnegative=True,
            )
            residual_components = tuple(
                _extract_component_bounds(
                    shell_m_center=residual_m_center[h],
                    shell_m_radius=residual_m_radius[h],
                    q_center=residual_q_center[h],
                    q_radius=residual_q_radius[h],
                    action_count=actions,
                    shell_size=shell,
                )
                for h in (0, 1)
            )
    population_count = sum(population_strata)
    d1_weight = Fraction(sum(d1_strata), population_count)
    residual_weights = tuple(Fraction(*value) for value in residual_mass_pairs)
    target, complement = residual_components
    mixed = tuple(
        _mix_population_bounds(
            (
                (d1_component[index], d1_component[index + 1], d1_weight),
                (target[index], target[index + 1], residual_weights[0]),
                (
                    complement[index],
                    complement[index + 1],
                    residual_weights[1],
                ),
            )
        )
        for index in (0, 2, 4)
    )
    center, radius, lower, upper, row_sum = _loewner_from_bounds(
        g_lower=mixed[0][0],
        g_upper=mixed[0][1],
        h_lower=mixed[1][0],
        h_upper=mixed[1][1],
        c_lower=mixed[2][0],
        c_upper=mixed[2][1],
    )
    if (
        not residual_stream.pass2_unopened
        or d1_stream._cursor.route_tile_read_count != 0
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ORDER_COVERAGE_MISMATCH,
            "compact D1/residual issuance consumed route-scale source rows",
        )
    max_row_sum = float(torch.max(row_sum).item())
    receipt = V1LiteralJJointAffineShellD1Pass1LoewnerReceipt(
        joint_preoutcome_registration_hash=prereg.registration_hash,
        joint_family_receipt_hash=family.receipt_hash,
        joint_postsource_custody_axis_hash=family.postsource_custody_axis_hash,
        population_measure_registration_hash=(
            "private-reference:no-population-measure-registration"
            if _population_measure_receipt is None
            else _population_measure_receipt.receipt_hash
        ),
        population_measure_receipt_hash=(
            "private-reference:no-population-measure-receipt"
            if _population_measure_receipt is None
            else _population_measure_receipt.receipt_hash
        ),
        resident_population_registration_hash=(
            "private-reference:no-resident-population-registration"
            if _population_measure_receipt is None
            else _population_measure_receipt.resident_population_registration_hash
        ),
        population_measure_design_hash=(
            "private-reference:no-population-measure-design"
            if _population_measure_receipt is None
            else _population_measure_receipt.population_measure_design_hash
        ),
        population_measure_weight_receipt_hash=(
            "private-reference:no-population-measure-weight-receipt"
            if _population_measure_receipt is None
            else _population_measure_receipt.population_measure_weight_receipt_hash
        ),
        action_axis_preoutcome_receipt_hash=(
            "private-reference:no-action-axis-preoutcome"
            if _action_axis_source is None
            else _action_axis_source.preoutcome_source.receipt.receipt_hash
        ),
        action_axis_source_receipt_hash=(
            "private-reference:no-action-axis-postsource"
            if _action_axis_source is None
            else _action_axis_source.receipt.receipt_hash
        ),
        atomic_source_receipt_hashes=atomic_hashes,
        atomic_preoutcome_receipt_hashes=atomic_pre_hashes,
        d1_stream_receipt_hash=d1_receipt.receipt_hash,
        d1_q_receipt_hash=d1_q_receipt.receipt_hash,
        d1_compact_shell_m_receipt_hash=d1_compact_receipt.receipt_hash,
        d1_source_axis_hash=d1_receipt.source_axis_hash,
        d1_literal_j_shard_receipt_hashes=d1_receipt.literal_j_shard_receipt_hashes,
        d1_literal_j_content_sha256s=d1_receipt.literal_j_content_sha256s,
        residual_stream_registration_hash=residual_stream.registration.registration_hash,
        residual_q_receipt_hash=residual_q_receipt.receipt_hash,
        residual_compact_shell_m_receipt_hash=residual_compact_receipt.receipt_hash,
        residual_source_proof_closure_hash_axes=(
            residual_compact_receipt.source_proof_closure_hash_axes[0],
            residual_compact_receipt.source_proof_closure_hash_axes[1],
        ),
        residual_source_shard_receipt_hash_axes=(
            residual_compact_receipt.source_shard_receipt_hash_axes[0],
            residual_compact_receipt.source_shard_receipt_hash_axes[1],
        ),
        residual_source_content_sha256_axes=(
            residual_compact_receipt.source_content_sha256_axes[0],
            residual_compact_receipt.source_content_sha256_axes[1],
        ),
        model_identity_hash=prereg.model_identity_hash,
        chart_identity_hash=prereg.chart_identity_hash,
        current_mask_hash=prereg.current_mask_hash,
        proposal_generation=prereg.proposal_generation,
        route_axis_hash=prereg.route_axis_hash,
        route_id_axis_hash=prereg.route_id_axis_hash,
        route_count=prereg.route_count,
        registered_d_prefix_hash=prereg.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=prereg.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=prereg.shell_freeze_receipt_hash,
        shell_indices=prereg.shell_indices,
        shell_route_ids=prereg.shell_route_ids,
        action_count=actions,
        shell_size=shell,
        physical_action_axis_hash=prereg.physical_action_axis_hash,
        physical_action_identity_hashes=prereg.physical_action_identity_hashes,
        physical_action_hashes=prereg.physical_action_hashes,
        packed_action_hashes=prereg.packed_action_hashes,
        joint_direction_ids=prereg.joint_direction_ids,
        joint_direction_axis_hash=prereg.joint_direction_axis_hash,
        small_axis_ids=tuple(
            prereg.shared_shell_direction_ids
            + (prereg.outside_direction_ids[action],)
            for action in range(actions)
        ),
        d1_stratum_counts=d1_strata,
        residual_population_counts=residual_stream.registration.residual_population_counts,
        residual_observed_counts=residual_stream.registration.registered_observed_counts,
        population_count=population_count,
        d1_count=sum(d1_strata),
        d1_mass_fraction=(d1_weight.numerator, d1_weight.denominator),
        residual_mass_fractions=residual_mass_pairs,
        alpha_ledger_hash=prereg.historical_alpha_ledger_hash,
        alpha_coordinate_hash=prereg.historical_alpha_coordinate_hash,
        alpha_lineage_hash=prereg.historical_alpha_lineage_hash,
        inherited_familywise_alpha=residual_stream.registration.familywise_alpha,
        inherited_alpha_spent=(
            0.0
            if residual_stream.registration.registered_observed_counts
            == residual_stream.registration.residual_population_counts
            else residual_stream.registration.familywise_alpha
        ),
        additional_alpha_spent=0.0,
        measurement_device_identity=str(device),
        small_matrix_center_digest=_tensor_digest(center, role="small_center"),
        small_matrix_radius_digest=_tensor_digest(radius, role="small_radius"),
        loewner_lower_digest=_tensor_digest(lower, role="loewner_lower"),
        loewner_upper_digest=_tensor_digest(upper, role="loewner_upper"),
        row_sum_radius_digest=_tensor_digest(row_sum, role="row_sum_radius"),
        max_row_sum_radius_upper_bound=max_row_sum,
        production_cuda_execution=_production,
        private_cpu_reference_execution=not _production,
        population_measure_binding_present=_population_measure_receipt is not None,
        action_complete_source_binding_present=_action_axis_source is not None,
        atomic_source_tuple_nonauthoritative=_action_axis_source is None,
        population_loewner_authority=_production,
        kkt_geometry_uncertainty_input_authority=_production,
        _issuance_token=_D1_PASS1_RECEIPT_TOKEN,
    )
    handle = V1LiteralJJointAffineShellLoewnerStreamHandle(
        receipt=receipt,
        tensors={
            "small_center": center,
            "small_radius": radius,
            "loewner_lower": lower,
            "loewner_upper": upper,
            "row_sum_radius": row_sum,
        },
        issuance_token=_HANDLE_TOKEN,
    )
    return V1LiteralJJointAffineShellLoewnerStreamResult(
        receipt=receipt,
        device_handle=handle,
    )


def issue_v1_literal_j_joint_affine_shell_d1_loewner_pass1_bridge(
    *,
    action_complete_population_source_binding: V1LiteralJJointPopulationActionAxisSource,
    population_measure_registration: V1LiteralJJointAxisResidualPopulationMeasureRegistration,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    d1_stream: V1LiteralJJointAffineShellD1PopulationStreamDeviceHandle,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
) -> V1LiteralJJointAffineShellLoewnerStreamResult:
    """Issue authoritative compact geometry from one exact POST aggregate."""

    if (
        type(action_complete_population_source_binding)
        is not V1LiteralJJointPopulationActionAxisSource
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ACTION_COMPLETE_SOURCE_BINDING_REQUIRED,
            "authoritative D1/residual Loewner issuance requires the exact "
            "producer-owned POST action-axis aggregate",
        )
    try:
        action_complete_population_source_binding.assert_live()
    except Exception as error:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.MUTATED_CUSTODY,
            "the action-complete population source no longer replays: "
            f"{type(error).__name__}: {error}",
        )
    aggregate = action_complete_population_source_binding.receipt
    pre = action_complete_population_source_binding.preoutcome_source.receipt
    prereg = joint_preoutcome_registration
    if (
        aggregate.preoutcome_action_axis_receipt_hash != pre.receipt_hash
        or aggregate.joint_preoutcome_registration_hash != prereg.registration_hash
        or pre.joint_preoutcome_registration_hash != prereg.registration_hash
        or aggregate.joint_direction_axis_hash != prereg.joint_direction_axis_hash
        or pre.joint_direction_axis_hash != prereg.joint_direction_axis_hash
        or pre.joint_direction_ids != prereg.joint_direction_ids
        or aggregate.physical_action_axis_hash != prereg.physical_action_axis_hash
        or pre.physical_action_axis_hash != prereg.physical_action_axis_hash
        or aggregate.physical_action_identity_hashes
        != prereg.physical_action_identity_hashes
        or aggregate.physical_action_hashes != prereg.physical_action_hashes
        or aggregate.packed_action_hashes != prereg.packed_action_hashes
        or pre.action_count != prereg.action_count
        or pre.shell_size != prereg.shell_size
        or pre.direction_count != prereg.direction_count
        or pre.route_count != prereg.route_count
        or pre.d1_context_count != d1_stream.receipt.context_count
        or pre.d1_stratum_counts != d1_stream.receipt.d1_stratum_counts
        or pre.residual_sample_stratum_counts
        != residual_stream.registration.registered_observed_counts
        or tuple(
            pre.population_stratum_counts[h] - pre.d1_stratum_counts[h]
            for h in (0, 1)
        )
        != residual_stream.registration.residual_population_counts
        or pre.alpha_lineage_hash
        != joint_preoutcome_registration.historical_alpha_lineage_hash
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.SOURCE_MISMATCH,
            "POST action-axis aggregate, D1 stream, residual stream, and joint "
            "registration do not name one complete population geometry source",
        )
    measure = _validate_d1_population_measure_ancestry(
        population_measure_registration=population_measure_registration,
        action_axis_source=action_complete_population_source_binding,
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        d1_stream=d1_stream,
        residual_stream=residual_stream,
    )
    return _issue_v1_literal_j_joint_affine_shell_d1_loewner_pass1_bridge_cpu_reference(
        joint_preoutcome_registration=joint_preoutcome_registration,
        joint_family_handle=joint_family_handle,
        atomic_population_sources=action_complete_population_source_binding._sources,
        d1_stream=d1_stream,
        residual_stream=residual_stream,
        _action_axis_source=action_complete_population_source_binding,
        _population_measure_receipt=measure,
        _production=True,
    )


def issue_v1_literal_j_joint_affine_shell_loewner_stream_bridge(
    *,
    joint_preoutcome_registration: V1LiteralJJointAffineShellPreoutcomeRegistration,
    joint_family_handle: V1LiteralJJointAffineShellFamilyDeviceHandle,
    d_fixed_source_results: tuple[V1LiteralJAffineShellShardReplayResult, ...],
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    residual_stream: V1LiteralJJointAffineShellResidualPopulationStreamCUDAHandle,
    allow_private_reference_for_tests: bool = False,
) -> V1LiteralJJointAffineShellLoewnerStreamResult:
    """Consume one joint residual stream and issue bounded per-action geometry."""

    if torch is None:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.INVALID_TYPE,
            "joint Loewner bridge requires PyTorch",
        )
    (
        family,
        q_receipt,
        d_registrations,
        production,
        joint_population_custody,
    ) = _validate_lineage(
        preregistration=joint_preoutcome_registration,
        family_handle=joint_family_handle,
        d_fixed_results=d_fixed_source_results,
        population=population_registration,
        residual_stream=residual_stream,
        allow_private_reference_for_tests=allow_private_reference_for_tests,
    )
    prereg = joint_preoutcome_registration
    actions = prereg.action_count
    shell = prereg.shell_size
    directions = prereg.direction_count
    routes = prereg.route_count
    shell_index = None
    device = None
    with joint_family_handle.borrow() as family_lease:
        outside_vectors, live_shell = family_lease.projection_custody()
        if len(outside_vectors) != actions:
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.JOINT_AXIS_MISMATCH,
                "live joint family lost one physical action",
            )
        device = outside_vectors[0].device
        if any(item.device != device for item in outside_vectors) or live_shell.device != device:
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.DEVICE_MISMATCH,
                "joint family tensors occupy different device ordinals",
            )
        shell_index = torch.tensor(prereg.shell_indices, dtype=torch.int64, device=device)
        if not bool(torch.equal(live_shell, shell_index)):
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.JOINT_AXIS_MISMATCH,
                "live shell tensor does not replay the frozen shell indices",
            )
    d_g_lower = d_g_upper = None
    d_h_lower = torch.empty((actions, shell), dtype=torch.float64, device=device)
    d_h_upper = torch.empty_like(d_h_lower)
    d_c_lower = torch.empty((actions,), dtype=torch.float64, device=device)
    d_c_upper = torch.empty_like(d_c_lower)
    d_source_payload_bytes = 0
    for action, result in enumerate(d_fixed_source_results):
        with result.projection.device_handle.borrow() as lease:
            mjz_center = _require_tensor(
                lease.mjz_center,
                name="D_fixed Mv center",
                shape=(shell + 1, routes),
                device=device,
            )
            mjz_radius = _require_tensor(
                lease.mjz_radius,
                name="D_fixed Mv radius",
                shape=(shell + 1, routes),
                device=device,
                nonnegative=True,
            )
            q_center = _require_tensor(
                lease.q_center,
                name="D_fixed Q center",
                shape=(shell + 1,),
                device=device,
            )
            q_radius = _require_tensor(
                lease.q_radius,
                name="D_fixed Q radius",
                shape=(shell + 1,),
                device=device,
                nonnegative=True,
            )
            shell_center = mjz_center.index_select(1, shell_index).contiguous()
            shell_radius = mjz_radius.index_select(1, shell_index).contiguous()
            component = _extract_component_bounds(
                shell_m_center=shell_center,
                shell_m_radius=shell_radius,
                q_center=q_center,
                q_radius=q_radius,
                action_count=1,
                shell_size=shell,
            )
        g_lower, g_upper, h_lower, h_upper, c_lower, c_upper = component
        if d_g_lower is None:
            d_g_lower, d_g_upper = g_lower, g_upper
        else:
            d_g_lower, d_g_upper = _intersect(
                d_g_lower, d_g_upper, g_lower, g_upper
            )
        d_h_lower[action] = h_lower[0]
        d_h_upper[action] = h_upper[0]
        d_c_lower[action] = c_lower[0]
        d_c_upper[action] = c_upper[0]
        d_source_payload_bytes += result.source_bound_replay.receipt.source_payload_read_bytes
    if d_g_lower is None or d_g_upper is None:
        raise AssertionError("validated nonempty D_fixed action axis disappeared")
    with residual_stream.q_handle.borrow() as q_lease:
        residual_q_center, residual_q_radius = q_lease.intervals()
        residual_q_center = _require_tensor(
            residual_q_center,
            name="residual Q center",
            shape=(2, directions),
            device=device,
        ).clone().contiguous()
        residual_q_radius = _require_tensor(
            residual_q_radius,
            name="residual Q radius",
            shape=(2, directions),
            device=device,
            nonnegative=True,
        ).clone().contiguous()
    residual_m_center = torch.empty(
        (2, directions, shell), dtype=torch.float64, device=device
    )
    residual_m_radius = torch.empty_like(residual_m_center)
    shell_seen = torch.zeros((2, shell), dtype=torch.bool, device=device)
    next_route = [0, 0]
    pending_target_interval: tuple[int, int] | None = None
    tile_hashes: list[str] = []
    while True:
        tile = residual_stream.next_tile()
        if tile is None:
            break
        eviction_error: BaseException | None = None
        try:
            receipt = tile.receipt
            receipt.assert_integrity()
            if (
                receipt.tile_ordinal != len(tile_hashes)
                or receipt.registration_hash != residual_stream.registration.registration_hash
                or receipt.q_receipt_hash != q_receipt.receipt_hash
                or receipt.joint_direction_axis_hash != prereg.joint_direction_axis_hash
                or receipt.direction_ids != prereg.joint_direction_ids
                or receipt.route_count != routes
                or receipt.direction_count != directions
                or receipt.route_start != next_route[receipt.stratum_ordinal]
            ):
                _refuse(
                    V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ORDER_COVERAGE_MISMATCH,
                    "residual Mv tiles do not form one route-major target/complement census",
                )
            if receipt.stratum_ordinal == 0:
                if pending_target_interval is not None:
                    _refuse(
                        V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ORDER_COVERAGE_MISMATCH,
                        "target residual tile arrived before its prior complement",
                    )
                pending_target_interval = (receipt.route_start, receipt.route_stop)
            elif pending_target_interval != (receipt.route_start, receipt.route_stop):
                _refuse(
                    V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ORDER_COVERAGE_MISMATCH,
                    "complement residual tile did not close the matching target route interval",
                )
            with tile.borrow() as tile_lease:
                center, radius = tile_lease.intervals()
                width = receipt.route_stop - receipt.route_start
                center = _require_tensor(
                    center,
                    name="residual Mv tile center",
                    shape=(directions, width),
                    device=device,
                )
                radius = _require_tensor(
                    radius,
                    name="residual Mv tile radius",
                    shape=(directions, width),
                    device=device,
                    nonnegative=True,
                )
                for shell_ordinal, physical in enumerate(prereg.shell_indices):
                    if receipt.route_start <= physical < receipt.route_stop:
                        local = physical - receipt.route_start
                        h = receipt.stratum_ordinal
                        if bool(shell_seen[h, shell_ordinal].item()):
                            _refuse(
                                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ORDER_COVERAGE_MISMATCH,
                                "a shell coordinate appeared in more than one residual tile",
                            )
                        residual_m_center[h, :, shell_ordinal] = center[:, local]
                        residual_m_radius[h, :, shell_ordinal] = radius[:, local]
                        shell_seen[h, shell_ordinal] = True
            next_route[receipt.stratum_ordinal] = receipt.route_stop
            if receipt.stratum_ordinal == 1:
                pending_target_interval = None
            tile_hashes.append(receipt.receipt_hash)
        finally:
            try:
                tile.evict()
            except BaseException as error:
                eviction_error = error
            if eviction_error is not None:
                raise eviction_error
    if (
        next_route != [routes, routes]
        or pending_target_interval is not None
        or not bool(shell_seen.all().item())
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ORDER_COVERAGE_MISMATCH,
            "residual Mv stream omitted or duplicated route/shell coverage",
        )
    stream_execution = residual_stream.finalize()
    stream_execution.assert_integrity()
    if (
        stream_execution.tile_receipt_hashes != tuple(tile_hashes)
        or stream_execution.q_receipt_hash != q_receipt.receipt_hash
        or stream_execution.joint_preoutcome_registration_hash != prereg.registration_hash
        or stream_execution.joint_direction_axis_hash != prereg.joint_direction_axis_hash
        or stream_execution.joint_direction_ids != prereg.joint_direction_ids
        or stream_execution.residual_population_counts
        != population_registration.residual_stratum_counts
        or stream_execution.observed_counts != q_receipt.observed_counts
        or stream_execution.production_cuda_execution != production
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.STALE_IDENTITY,
            "final residual execution receipt does not close the consumed stream",
        )
    residual_components = tuple(
        _extract_component_bounds(
            shell_m_center=residual_m_center[h],
            shell_m_radius=residual_m_radius[h],
            q_center=residual_q_center[h],
            q_radius=residual_q_radius[h],
            action_count=actions,
            shell_size=shell,
        )
        for h in (0, 1)
    )
    d_weight = Fraction(*population_registration.d_fixed_mass_fraction)
    residual_weights = tuple(
        Fraction(*item)
        for item in population_registration.residual_stratum_mass_fractions
    )
    target, complement = residual_components
    g_lower, g_upper = _mix_population_bounds(
        (
            (d_g_lower, d_g_upper, d_weight),
            (target[0], target[1], residual_weights[0]),
            (complement[0], complement[1], residual_weights[1]),
        )
    )
    h_lower, h_upper = _mix_population_bounds(
        (
            (d_h_lower, d_h_upper, d_weight),
            (target[2], target[3], residual_weights[0]),
            (complement[2], complement[3], residual_weights[1]),
        )
    )
    c_lower, c_upper = _mix_population_bounds(
        (
            (d_c_lower, d_c_upper, d_weight),
            (target[4], target[5], residual_weights[0]),
            (complement[4], complement[5], residual_weights[1]),
        )
    )
    tensors = _loewner_from_bounds(
        g_lower=g_lower,
        g_upper=g_upper,
        h_lower=h_lower,
        h_upper=h_upper,
        c_lower=c_lower,
        c_upper=c_upper,
    )
    center, radius, lower, upper, row_sum = tensors
    max_row_sum = float(torch.max(row_sum).detach().cpu().item())
    if not math.isfinite(max_row_sum) or max_row_sum < 0.0:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.NUMERICAL_REFUSAL,
            "maximum row-sum radius is not finite and nonnegative",
        )
    compact_tensor_scalars = int(
        center.numel()
        + radius.numel()
        + lower.numel()
        + upper.numel()
        + row_sum.numel()
    )
    # Each compact tensor is copied once to seal the durable digest and once
    # more when the live handle verifies that digest.  The maximum row-sum
    # scalar is the only other FP64 payload copied to host.  Boolean control
    # synchronizations are not FP64 payload and are deliberately not counted.
    compact_scalars = 2 * compact_tensor_scalars + 1
    if not production:
        compact_scalars = 0
    receipt = V1LiteralJJointAffineShellLoewnerStreamReceipt(
        joint_preoutcome_registration_hash=prereg.registration_hash,
        joint_family_receipt_hash=family.receipt_hash,
        joint_postsource_custody_axis_hash=family.postsource_custody_axis_hash,
        population_projection_registration_hash=population_registration.registration_hash,
        residual_stream_registration_hash=residual_stream.registration.registration_hash,
        residual_q_receipt_hash=q_receipt.receipt_hash,
        residual_stream_execution_receipt_hash=stream_execution.receipt_hash,
        d_fixed_affine_replay_receipt_hashes=tuple(item.receipt.receipt_hash for item in d_fixed_source_results),
        d_fixed_direction_receipt_hashes=tuple(
            item.receipt.direction_receipt_hash for item in d_fixed_source_results
        ),
        d_fixed_registration_hashes=tuple(
            item.registration.registration_hash for item in d_fixed_source_results
        ),
        d_fixed_source_replay_receipt_hashes=tuple(item.source_bound_replay.receipt.receipt_hash for item in d_fixed_source_results),
        d_fixed_projection_receipt_hashes=tuple(item.projection.receipt.receipt_hash for item in d_fixed_source_results),
        d_fixed_common_source_replay_axis_hash=d_registrations[0].source_replay_axis_hash,
        d_fixed_population_source_replay_axis_hash=(
            population_registration.d_fixed_source_replay_axis_hash
        ),
        d_fixed_literal_j_tile_source_ledger_hashes=tuple(
            item.source_bound_replay.receipt.literal_j_tile_source_ledger_hash
            for item in d_fixed_source_results
        ),
        model_identity_hash=prereg.model_identity_hash,
        chart_identity_hash=prereg.chart_identity_hash,
        current_mask_hash=prereg.current_mask_hash,
        proposal_generation=prereg.proposal_generation,
        route_axis_hash=prereg.route_axis_hash,
        route_id_axis_hash=prereg.route_id_axis_hash,
        route_count=routes,
        registered_d_prefix_hash=prereg.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=prereg.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=prereg.shell_freeze_receipt_hash,
        shell_indices=prereg.shell_indices,
        shell_route_ids=prereg.shell_route_ids,
        action_count=actions,
        shell_size=shell,
        physical_action_identity_hashes=prereg.physical_action_identity_hashes,
        physical_action_hashes=prereg.physical_action_hashes,
        packed_action_hashes=prereg.packed_action_hashes,
        joint_direction_ids=prereg.joint_direction_ids,
        joint_direction_axis_hash=prereg.joint_direction_axis_hash,
        small_axis_ids=tuple(
            prereg.shared_shell_direction_ids + (prereg.outside_direction_ids[action],)
            for action in range(actions)
        ),
        d_fixed_stratum_counts=population_registration.d_fixed_stratum_counts,
        residual_population_counts=population_registration.residual_stratum_counts,
        residual_observed_counts=stream_execution.observed_counts,
        population_count=population_registration.population_count,
        d_fixed_count=population_registration.d_fixed_count,
        d_fixed_mass_fraction=population_registration.d_fixed_mass_fraction,
        residual_mass_fractions=population_registration.residual_stratum_mass_fractions,
        alpha_ledger_hash=population_registration.alpha_ledger_hash,
        alpha_coordinate_hash=population_registration.alpha_coordinate_hash,
        alpha_lineage_hash=population_registration.alpha_lineage_hash,
        inherited_familywise_alpha=population_registration.familywise_alpha,
        inherited_alpha_spent=stream_execution.alpha_spent,
        additional_alpha_spent=0.0,
        residual_exact_census=stream_execution.exact_residual_census,
        production_cuda_execution=production,
        private_cpu_reference_execution=not production,
        measurement_device_identity=str(device),
        residual_tile_receipt_hashes=stream_execution.tile_receipt_hashes,
        residual_pass2_tile_count=stream_execution.pass2_route_tile_read_count,
        d_fixed_action_replay_count=actions,
        d_fixed_total_source_pass_count=sum(item.receipt.immutable_shard_read_pass_count for item in d_fixed_source_results),
        d_fixed_total_source_payload_read_bytes=d_source_payload_bytes,
        residual_source_pass_count=stream_execution.source_pass_count,
        residual_source_payload_read_bytes=stream_execution.source_payload_read_bytes,
        compact_tensor_d2h_scalar_count=compact_scalars,
        compact_tensor_d2h_bytes=compact_scalars * _F64_BYTES,
        small_matrix_center_digest=_tensor_digest(center, role="small_center"),
        small_matrix_radius_digest=_tensor_digest(radius, role="small_radius"),
        loewner_lower_digest=_tensor_digest(lower, role="loewner_lower"),
        loewner_upper_digest=_tensor_digest(upper, role="loewner_upper"),
        row_sum_radius_digest=_tensor_digest(row_sum, role="row_sum_radius"),
        max_row_sum_radius_upper_bound=max_row_sum,
        joint_action_complete_population_custody=joint_population_custody,
        legacy_private_population_registration=not joint_population_custody,
        population_loewner_authority=production,
        kkt_geometry_uncertainty_input_authority=production,
        _issuance_token=_RECEIPT_TOKEN,
    )
    handle = V1LiteralJJointAffineShellLoewnerStreamHandle(
        receipt=receipt,
        tensors={
            "small_center": center,
            "small_radius": radius,
            "loewner_lower": lower,
            "loewner_upper": upper,
            "row_sum_radius": row_sum,
        },
        issuance_token=_HANDLE_TOKEN,
    )
    return V1LiteralJJointAffineShellLoewnerStreamResult(
        receipt=receipt,
        device_handle=handle,
    )



def issue_v1_factored_joint_affine_shell_loewner_result(
    *,
    bounds: Any,
    lineage: V1FactoredJointLoewnerLineage,
    alpha_owner_binding: Any | None = None,
    private_reference_alpha: tuple[str, str, str, float] | None = None,
    production_cuda_execution: bool = False,
    allow_private_reference_for_tests: bool = False,
) -> V1LiteralJJointAffineShellLoewnerStreamResult:
    """Bind factored mixed-sample geometry for private CPU reference use.

    ``bounds`` is a ``V1FactoredRowJointPopulationLoewnerBounds``.
    Its radii account for observed arithmetic, not missing population
    contexts. Production refuses until a source-bound statistical enclosure
    or census witness and a factored-native alpha owner exist. A private
    reference alpha tuple ``(ledger, coordinate, lineage, familywise_alpha)``
    is admitted only with ``allow_private_reference_for_tests`` and yields a
    non-authoritative receipt. A row-backed alpha owner cannot bind factored
    packet ancestry merely by supplying ledger hashes.
    """

    from .modern_v1_factored_row_consumers import V1FactoredRowJointPopulationLoewnerBounds

    if torch is None:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.INVALID_TYPE,
            "factored joint Loewner issuance requires PyTorch",
        )
    if type(bounds) is not V1FactoredRowJointPopulationLoewnerBounds:
        raise TypeError("factored joint Loewner issuance requires exact factored population bounds")
    if type(lineage) is not V1FactoredJointLoewnerLineage:
        raise TypeError("factored joint Loewner issuance requires its exact lineage")
    if type(production_cuda_execution) is not bool or type(allow_private_reference_for_tests) is not bool:
        raise TypeError("execution and reference flags must be exact bools")
    if production_cuda_execution:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.POPULATION_ENCLOSURE_ABSENT,
            "factored mixed-sample radii have no source-bound population sampling "
            "enclosure or census witness; CUDA execution and alpha metadata cannot supply one",
        )
    if bounds.bridge_result_bound is not False:
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.AUTHORITY_VIOLATION,
            "factored bounds already claim a bridge binding",
        )
    actions = int(bounds.action_count)
    shell = int(bounds.shell_size)
    if (
        len(lineage.physical_action_identity_hashes) != actions
        or len(lineage.physical_action_hashes) != actions
        or len(lineage.packed_action_hashes) != actions
        or len(lineage.outside_direction_ids) != actions
        or len(lineage.shared_shell_direction_ids) != shell
        or len(lineage.shell_route_ids) != shell
        or int(lineage.route_count) <= max(int(v) for v in bounds.shell_route_indices)
    ):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.JOINT_AXIS_MISMATCH,
            "factored lineage axes do not match the population bounds",
        )
    if alpha_owner_binding is not None:
        from .modern_v1_sampled_v7_d_alpha_measurement_owner_binding import (
            V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
        )

        if type(alpha_owner_binding) is not V1SampledV7DAlphaMeasurementOwnerBindingReceipt:
            raise TypeError("alpha lineage enters only through the exact owner binding receipt")
        if private_reference_alpha is not None:
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.AUTHORITY_VIOLATION,
                "a bound alpha lineage cannot be mixed with a private reference alpha",
            )
        alpha_owner_binding.assert_live()
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ALPHA_LINEAGE_ABSENT,
            "the row-backed alpha owner does not bind this factored session, "
            "replay boundary, direction family, and component packet ancestry (F5.3)",
        )
    else:
        if private_reference_alpha is None:
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.ALPHA_LINEAGE_ABSENT,
                "factored joint Loewner issuance requires an alpha owner binding (doc 51 §9.4 F5.3) "
                "or an explicit private reference alpha",
            )
        if not allow_private_reference_for_tests or production_cuda_execution:
            _refuse(
                V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.AUTHORITY_VIOLATION,
                "a private reference alpha is admitted only for non-production small-domain references",
            )
        ledger, coordinate, alpha_lineage, familywise = private_reference_alpha
        alpha_fields = (
            "private-reference:no-alpha-owner-binding",
            "private-reference:" + str(ledger),
            "private-reference:" + str(coordinate),
            "private-reference:" + str(alpha_lineage),
            float(familywise),
        )
        alpha_bound = False
    center = bounds.center
    radius = bounds.radius
    lower = bounds.loewner_lower
    upper = bounds.loewner_upper
    row_sum = bounds.row_sum
    device = center.device
    joint_ids = tuple(lineage.outside_direction_ids) + tuple(lineage.shared_shell_direction_ids)
    if len(set(joint_ids)) != len(joint_ids):
        _refuse(
            V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode.JOINT_AXIS_MISMATCH,
            "joint direction ids must be unique",
        )
    observed = tuple(int(v) for v in bounds.residual_observed_counts)
    remaining = tuple(
        int(bounds.residual_population_counts[h]) - int(bounds.d1_stratum_counts[h]) for h in (0, 1)
    )
    receipt = V1FactoredJointAffineShellLoewnerReceipt(
        factored_population_bounds_receipt_hash=bounds.receipt_hash,
        component_stream_receipt_hashes=tuple(bounds.component_receipt_hashes),  # type: ignore[arg-type]
        factored_packet_receipt_hashes=tuple(lineage.factored_packet_receipt_hashes),
        session_design_hash=lineage.session_design_hash,
        replay_boundary_hash=lineage.replay_boundary_hash,
        mass_law_id=str(bounds.mass_law_id),
        model_identity_hash=lineage.model_identity_hash,
        chart_identity_hash=lineage.chart_identity_hash,
        current_mask_hash=lineage.current_mask_hash,
        proposal_generation=int(lineage.proposal_generation),
        route_axis_hash=lineage.route_axis_hash,
        route_id_axis_hash=lineage.route_id_axis_hash,
        route_count=int(lineage.route_count),
        shell_indices=tuple(int(v) for v in bounds.shell_route_indices),
        shell_route_ids=tuple(lineage.shell_route_ids),
        action_count=actions,
        shell_size=shell,
        physical_action_identity_hashes=tuple(lineage.physical_action_identity_hashes),
        physical_action_hashes=tuple(lineage.physical_action_hashes),
        packed_action_hashes=tuple(lineage.packed_action_hashes),
        joint_direction_ids=joint_ids,
        joint_direction_axis_hash=content_hash(
            {"schema": "factored_joint_direction_axis_v1", "ids": joint_ids},
            prefix=_PREFIX + "factored-direction-axis-",
        ),
        small_axis_ids=tuple(
            tuple(lineage.shared_shell_direction_ids) + (lineage.outside_direction_ids[action],)
            for action in range(actions)
        ),
        d1_stratum_counts=tuple(int(v) for v in bounds.d1_stratum_counts),  # type: ignore[arg-type]
        residual_population_counts=tuple(int(v) for v in bounds.residual_population_counts),  # type: ignore[arg-type]
        residual_observed_counts=observed,  # type: ignore[arg-type]
        population_count=int(bounds.population_count),
        d1_count=int(bounds.d1_count),
        d1_mass_fraction=tuple(int(v) for v in bounds.d1_mass_fraction),  # type: ignore[arg-type]
        residual_mass_fractions=tuple(tuple(int(v) for v in pair) for pair in bounds.residual_mass_fractions),  # type: ignore[arg-type]
        alpha_owner_binding_receipt_hash=alpha_fields[0],
        alpha_ledger_hash=alpha_fields[1],
        alpha_coordinate_hash=alpha_fields[2],
        alpha_lineage_hash=alpha_fields[3],
        inherited_familywise_alpha=alpha_fields[4],
        inherited_alpha_spent=0.0 if observed == remaining else alpha_fields[4],
        additional_alpha_spent=0.0,
        alpha_lineage_bound=alpha_bound,
        measurement_device_identity=str(device),
        small_matrix_center_digest=_tensor_digest(center, role="small_center"),
        small_matrix_radius_digest=_tensor_digest(radius, role="small_radius"),
        loewner_lower_digest=_tensor_digest(lower, role="loewner_lower"),
        loewner_upper_digest=_tensor_digest(upper, role="loewner_upper"),
        row_sum_radius_digest=_tensor_digest(row_sum, role="row_sum_radius"),
        max_row_sum_radius_upper_bound=float(row_sum.max().item()),
        production_cuda_execution=production_cuda_execution,
        private_cpu_reference_execution=not production_cuda_execution,
        population_loewner_authority=False,
        kkt_geometry_uncertainty_input_authority=False,
        _issuance_token=_FACTORED_RECEIPT_TOKEN,
    )
    handle = V1LiteralJJointAffineShellLoewnerStreamHandle(
        receipt=receipt,
        tensors={
            "small_center": center,
            "small_radius": radius,
            "loewner_lower": lower,
            "loewner_upper": upper,
            "row_sum_radius": row_sum,
        },
        issuance_token=_HANDLE_TOKEN,
    )
    return V1LiteralJJointAffineShellLoewnerStreamResult(receipt=receipt, device_handle=handle)


__all__ = (
    "LITERAL_J_JOINT_AFFINE_SHELL_FACTORED_LOEWNER_BRIDGE_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_FACTORED_LOEWNER_BRIDGE_ALGORITHM",
    "LITERAL_J_JOINT_AFFINE_SHELL_FACTORED_LOEWNER_BRIDGE_AUTHORITY",
    "V1FactoredJointLoewnerLineage",
    "V1FactoredJointAffineShellLoewnerReceipt",
    "issue_v1_factored_joint_affine_shell_loewner_result",
    "LITERAL_J_JOINT_AFFINE_SHELL_LOEWNER_STREAM_BRIDGE_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_LOEWNER_STREAM_BRIDGE_ALGORITHM",
    "LITERAL_J_JOINT_AFFINE_SHELL_LOEWNER_STREAM_BRIDGE_AUTHORITY",
    "LITERAL_J_JOINT_AFFINE_SHELL_PASS1_LOEWNER_BRIDGE_SCHEMA",
    "LITERAL_J_JOINT_AFFINE_SHELL_PASS1_LOEWNER_BRIDGE_ALGORITHM",
    "LITERAL_J_JOINT_AFFINE_SHELL_PASS1_LOEWNER_BRIDGE_AUTHORITY",
    "V1LiteralJJointAffineShellLoewnerStreamBridgeRefusalCode",
    "V1LiteralJJointAffineShellLoewnerStreamBridgeRefusal",
    "V1LiteralJJointAffineShellLoewnerStreamReceipt",
    "V1LiteralJJointAffineShellPass1LoewnerReceipt",
    "V1LiteralJJointAffineShellLoewnerStreamLease",
    "V1LiteralJJointAffineShellLoewnerStreamHandle",
    "V1LiteralJJointAffineShellLoewnerStreamResult",
    "issue_v1_literal_j_joint_affine_shell_loewner_pass1_bridge",
    "issue_v1_literal_j_joint_affine_shell_loewner_stream_bridge",
)
