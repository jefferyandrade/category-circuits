"""Bounded-shell physical-action cells for the missing D-side theorem.

Canonical signed-work v1 is lexicographic.  Phase A fixes the maximum
registered Q/resource coordinate.  Conditional on that frozen coordinate,
Phase B minimizes the live source native-fiber work

``F(z) = sum_i alpha_i |(A z + b)_i|``

under the registered resource floor, box, compact block capacities, and a
closed deterministic hard-rounding cover.  Canonical production lower bounds
drop Q from alternate domains, because this is a conservative superset; the
preferred hard action restores Q through its original registered branch replay.
The older literal-J response ball remains only for tiny synthetic reference
tests where the branch domain is explicitly declared identical to the sampled
literal-J diagnostic.  Consequently F is an objective here, never a feasibility
budget.

For endpoint cardinality ``k``, one exact physical action ``S`` has
the closed convex conservative cover cell

``C_S = {z in [0,1]^B : z_i >= z_j for i in S, j not in S}``.

The deterministic rounding preimage itself is half-open because its secondary
tie-break selects one action.  The ``binom(B,k)`` closed cells instead cover
the box, and an exact kth-boundary tie belongs to every compatible cell.
Solving all overcovers is conservative: strict separation excludes even the
extra boundary points.  Across distinct registered cardinalities there
are at most ``2**B <= 256`` cells.  Internal permutations within ``S`` or its
complement are irrelevant and are not a production solve axis.  The older
``B!`` permutation construction remains only as an explicit tiny reference
proof of this quotient.

This module lands the live-source registry, an exact touched-fiber quotient,
the deterministic matrix-free Phase-B PDHG joint, and a conditional
within-frozen-shell action-separation issuer.  The closed subset cells are
conservative overcovers, so every preferred primal witness is replayed through
the deterministic physical-action map.  Durable receipts are array-free, live
source handles are nonserializable, and every global action-invariance, C/K/O,
commit, terminal, and Pareto authority remains closed until an independently
source-bound outside-shell exclusion theorem is supplied.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from enum import Enum
from fractions import Fraction
import hashlib
from itertools import combinations, permutations
import math
from numbers import Integral, Real
from typing import Any, Final, Iterator, Mapping, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import RemovedRouteIndexSet

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .contracts import content_hash
from .modern_v1_decision_level_cuda_hash_custody import (
    MAXIMUM_HOST_LABEL_SLAB_BYTES,
    PREFIX_LABEL,
    SHELL_LABEL_BASE,
    SUFFIX_LABEL,
    V1DecisionLevelCudaHashCustodyIssuance,
    V1DecisionLevelCudaStreamedHashFamily,
    stream_v1_decision_level_hash_family_from_membership_slabs,
)
from .modern_v1_decision_level_cuda_issuer import (
    V1DecisionLevelCudaPrivatePrimalBinding,
)
from .modern_v1_device_shell_ladder import (
    V1DeviceShellFrozenLadderReceipt,
    V1DeviceShellLabelHandle,
)
from .modern_v1_sampled_full_population_compact_source import (
    V1SampledFullPopulationCompactSourceHandle,
    V1SampledFullPopulationCompactSourceLease,
    V1SampledFullPopulationCompactSourceReceipt,
)
from .modern_v1_sampled_variable_shell_packet_registration import (
    SAMPLED_VARIABLE_SHELL_TWO_BRANCH_KKT_ID,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceDeviceHandle,
    V1SignedWorkVariableShellSourceReceipt,
    V1SignedWorkVariableShellSourceTensorLease,
)
from .modern_v1_signed_work_variable_shell_source_append import (
    V1SignedWorkVariableShellSourceAppendSegmentLease,
    V1SignedWorkVariableShellSourceAppendSegmentedHandle,
    V1SignedWorkVariableShellSourceAppendSegmentedReceipt,
)

try:  # pragma: no cover - availability depends on the runtime image.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


V1SignedWorkVariableShellCompactSourceHandle = (
    V1SignedWorkVariableShellSourceDeviceHandle
    | V1SignedWorkVariableShellSourceAppendSegmentedHandle
    | V1SampledFullPopulationCompactSourceHandle
)
V1SignedWorkVariableShellCompactSourceReceipt = (
    V1SignedWorkVariableShellSourceReceipt
    | V1SignedWorkVariableShellSourceAppendSegmentedReceipt
    | V1SampledFullPopulationCompactSourceReceipt
)
V1SignedWorkVariableShellCompactSourceLease = (
    V1SignedWorkVariableShellSourceTensorLease
    | V1SignedWorkVariableShellSourceAppendSegmentLease
    | V1SampledFullPopulationCompactSourceLease
)


def _is_compact_source_handle(value: object) -> bool:
    return type(value) in {
        V1SignedWorkVariableShellSourceDeviceHandle,
        V1SignedWorkVariableShellSourceAppendSegmentedHandle,
        V1SampledFullPopulationCompactSourceHandle,
    }


def _is_compact_source_receipt(value: object) -> bool:
    return type(value) in {
        V1SignedWorkVariableShellSourceReceipt,
        V1SignedWorkVariableShellSourceAppendSegmentedReceipt,
        V1SampledFullPopulationCompactSourceReceipt,
    }


def _compact_source_device(
    source: V1SignedWorkVariableShellCompactSourceHandle,
    *,
    purpose_hash: str,
) -> Any:
    """Return one live device identity through the source-owned borrow API."""

    if type(source) is V1SignedWorkVariableShellSourceDeviceHandle:
        with source.borrow_compact_tensors(borrow_purpose_hash=purpose_hash) as lease:
            return lease.compact_tensors()[0].device
    if type(source) is V1SignedWorkVariableShellSourceAppendSegmentedHandle:
        with source.borrow_compact_segments(borrow_purpose_hash=purpose_hash) as lease:
            segments = lease.compact_segments()
            devices = tuple(segment.compact_tensors()[0].device for segment in segments)
            if not devices or any(device != devices[0] for device in devices[1:]):
                _refuse(
                    V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                    "segmented compact source spans multiple execution devices",
                )
            return devices[0]
    if type(source) is V1SampledFullPopulationCompactSourceHandle:
        with source.borrow_components(borrow_purpose_hash=purpose_hash) as lease:
            d1_device = lease.d1_compact_tensors()[0].device
            residual_devices = tuple(
                segment.compact_tensors()[0].device
                for segment in lease.residual_compact_segments()
            )
            if not residual_devices or any(
                device != d1_device for device in residual_devices
            ):
                _refuse(
                    V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                    "full-population compact source spans multiple execution devices",
                )
            return d1_device
    raise TypeError("compact source must have one exact registered handle type")


def _source_kkt_coordinate_supports_branch(
    source_kkt_branch_id: str,
    branch: "V1SignedWorkVariableShellQBranch",
) -> bool:
    """Whether one source coordinate lawfully supplies a Q branch.

    Legacy fixtures retain branch-local compact sources.  Canonical sampled
    execution freezes one common signed-work/Jacobian shell before solving the
    spectral and diagonal residual envelopes, so its source identity names the
    registered two-branch union.  Treating that union as either branch's
    *value* would be false; this predicate instead records containment while
    the explicit ``branch`` field continues to own the branch-local problem.
    """

    return source_kkt_branch_id in {
        branch.value,
        SAMPLED_VARIABLE_SHELL_TWO_BRANCH_KKT_ID,
    }


ACTION_SEPARATION_DESIGN_SCHEMA: Final = (
    "partition_free_v1_variable_shell_action_separation_cuda_design_v2"
)
ACTION_SUBSET_IDENTITY_SCHEMA: Final = (
    "partition_free_v1_variable_shell_action_subset_identity_v2"
)
ACTION_RESOURCE_ALIAS_SCHEMA: Final = (
    "partition_free_v1_variable_shell_phase_a_resource_alias_v2"
)
ACTION_CLOSED_SUBSET_CELL_SCHEMA: Final = (
    "partition_free_v1_variable_shell_closed_subset_action_cell_v2"
)
ACTION_REFERENCE_ORDER_CELL_SCHEMA: Final = (
    "partition_free_v1_variable_shell_reference_order_cell_v2"
)
ACTION_BRANCH_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_variable_shell_branch_action_cell_registration_v2"
)
ACTION_CELL_REGISTRY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_variable_shell_action_cell_registry_receipt_v2"
)
ACTION_CELL_IDENTITY_SCHEMA: Final = (
    "partition_free_v1_variable_shell_action_cell_identity_v2"
)
ACTION_BLOCK_CAPACITY_SCHEMA: Final = (
    "partition_free_v1_variable_shell_action_cell_block_capacity_v2"
)
ACTION_RESTRICTED_REQUEST_SCHEMA: Final = (
    "partition_free_v1_variable_shell_phase_b_restricted_cell_request_v2"
)
ACTION_DUAL_CONTRACT_SCHEMA: Final = (
    "partition_free_v1_variable_shell_phase_b_dual_lower_bound_contract_v2"
)
ACTION_DUAL_REFERENCE_SCHEMA: Final = (
    "partition_free_v1_variable_shell_phase_b_dual_lower_bound_reference_v2"
)
ACTION_PRIMAL_REFERENCE_SCHEMA: Final = (
    "partition_free_v1_variable_shell_phase_b_primal_reference_v2"
)
ACTION_QUANTIFIER_OBSERVATION_SCHEMA: Final = (
    "partition_free_v1_variable_shell_action_separation_bound_observation_v2"
)
ACTION_QUANTIFIER_REFERENCE_SCHEMA: Final = (
    "partition_free_v1_variable_shell_action_separation_quantifier_reference_v2"
)
ACTION_CUDA_HBM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_variable_shell_action_separation_cuda_hbm_v2"
)
ACTION_SOURCE_UNCERTAINTY_GROUP_SCHEMA: Final = (
    "partition_free_v1_variable_shell_source_uncertainty_group_receipt_v1"
)
ACTION_SOURCE_UNCERTAINTY_FAMILY_SCHEMA: Final = (
    "partition_free_v1_variable_shell_source_uncertainty_family_receipt_v1"
)
ACTION_SAMPLED_SUPPORT_UNCERTAINTY_GROUP_SCHEMA: Final = (
    "partition_free_v1_variable_shell_sampled_support_uncertainty_group_receipt_v1"
)
ACTION_SAMPLED_SUPPORT_UNCERTAINTY_FAMILY_SCHEMA: Final = (
    "partition_free_v1_variable_shell_sampled_support_uncertainty_family_receipt_v1"
)
ACTION_ALIAS_PROBLEM_SPEC_SCHEMA: Final = (
    "partition_free_v1_variable_shell_phase_b_alias_problem_spec_v1"
)
ACTION_PDHG_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_variable_shell_phase_b_pdhg_execution_receipt_v1"
)
ACTION_SOURCE_COMPACTION_SCHEMA: Final = (
    "partition_free_v1_variable_shell_phase_b_exact_source_compaction_v1"
)
ACTION_COMPACT_EVALUATION_SCHEMA: Final = (
    "partition_free_v1_variable_shell_phase_b_compact_evaluation_v1"
)
ACTION_SOLVER_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_variable_shell_phase_b_cell_optimality_receipt_v2"
)
ACTION_SEPARATION_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_variable_shell_action_separation_certificate_v2"
)
ACTION_OUTSIDE_SHELL_DEPENDENCY_SCHEMA: Final = (
    "partition_free_v1_outside_shell_exclusion_dependency_v1"
)
ACTION_Q_BRANCH_DOMAIN_BRIDGE_SCHEMA: Final = (
    "partition_free_v1_registered_q_branch_domain_bridge_v1"
)
ACTION_STREAMED_HASH_CUSTODY_FAMILY_SCHEMA: Final = (
    "partition_free_v1_variable_shell_streamed_action_coordinate_custody_family_v2"
)
ACTION_STREAMED_HASH_CUSTODY_ISSUANCE_SCHEMA: Final = (
    "partition_free_v1_variable_shell_streamed_action_coordinate_custody_issuance_v1"
)
ACTION_SOURCE_NATIVE_HASH_CUSTODY_SCHEMA: Final = (
    "partition_free_v1_variable_shell_source_native_action_coordinate_custody_v1"
)

ACTION_CELL_COVER_RULE: Final = (
    "for_fixed_k_all_closed_top_k_subset_cells_cover_box_and_exact_ties_"
    "belong_to_every_compatible_subset_cell_v2"
)
ACTION_CELL_IMAGE_RULE: Final = (
    "closed_subset_cell_is_conservative_overcover_of_half_open_deterministic_"
    "byte_action_preimage_and_preferred_primal_requires_exact_replay_v3"
)
ACTION_REFERENCE_PERMUTATION_RULE: Final = (
    "B_factorial_permutation_chambers_are_non_authoritative_reference_only_"
    "and_quotient_to_top_k_subset_cells_v2"
)
ACTION_LEXICOGRAPHIC_RULE: Final = (
    "phase_A_freezes_maximal_registered_Q_resource_coordinate_then_phase_B_"
    "minimizes_live_source_native_fiber_F_without_reopening_phase_A_v2"
)
ACTION_RESTRICTED_PROBLEM_RULE: Final = (
    "phase_B_minimize_source_native_fiber_F_subject_to_registered_phase_A_"
    "resource_floor_box_block_closed_subset_order_and_only_synthetic_"
    "reference_literal_J_Q_ball_when_declared_v3"
)
ACTION_PHASE_B_DUAL_LOWER_BOUND_RULE: Final = (
    "canonical_alternate_dual_lower_drops_Q_with_s_equals_A_T_yF_minus_"
    "lambda_c_plus_B_T_mu_plus_O_T_nu_while_synthetic_reference_may_add_"
    "weighted_J_T_yQ_and_minus_R_norm_yQ_v3"
)
ACTION_PHASE_B_DUAL_PROJECTION_RULE: Final = (
    "clip_yF_to_abs_alpha_zero_yQ_when_Q_dropped_otherwise_leave_yQ_free_"
    "and_project_lambda_mu_nu_nonnegative_before_outward_binary64_lower_"
    "enclosure_v3"
)
ACTION_SOURCE_COMPACTION_RULE: Final = (
    "untouched_native_fibers_are_z_independent_outward_constant_and_touched_"
    "fibers_are_exactly_coalesced_by_destination_before_absolute_value_v1"
)
ACTION_Q_DOMAIN_HIERARCHY_RULE: Final = (
    "alternate_cell_F_lower_bounds_first_drop_Q_as_a_sound_convex_superset_"
    "preferred_F_upper_requires_original_registered_branch_q_U_hard_replay_"
    "and_only_unresolved_cases_add_a_source_issued_convex_q_L_outer_set_v1"
)
ACTION_SEPARATION_RULE: Final = (
    "minimum_preferred_cell_feasible_primal_F_upper_is_strictly_below_every_"
    "alternate_physical_action_cell_global_dual_F_lower_v2"
)
ACTION_SEPARATION_AUTHORITY_SCOPE: Final = (
    "closed_physical_action_cell_registry_and_phase_B_solver_contract_only_"
    "no_statistical_uniformity_action_invariance_C_K_O_commit_terminal_or_"
    "pareto_authority_v2"
)
ACTION_LEGACY_HASH_CUSTODY_AUTHORITY_KIND: Final = (
    "legacy_decision_level_cuda_hash_custody_and_private_primal_binding"
)
ACTION_SOURCE_NATIVE_HASH_CUSTODY_AUTHORITY_KIND: Final = (
    "source_native_signed_work_d_w2_streamed_hash_custody"
)
ACTION_CELL_NUMERICAL_DTYPE: Final = "IEEE754_binary64"
ACTION_CELL_SOLVER_BACKEND: Final = "deterministic_torch_cuda_pdhg_phase_b_v2"

_PREFIX: Final = "partition-free-v1-variable-shell-action-separation-v2-"
_MAXIMUM_SHELL_SIZE: Final = 8
_MAXIMUM_ACTION_CELLS: Final = 1 << _MAXIMUM_SHELL_SIZE
_MAXIMUM_DUAL_WITNESS_SCALARS: Final = 1 << 20
_F64_EPS: Final = 2.0**-52
_F64_MIN_SUBNORMAL: Final = math.nextafter(0.0, math.inf)
_REGISTRATION_TOKEN: Final = object()
_RECEIPT_TOKEN: Final = object()
_SOLVER_RECEIPT_TOKEN: Final = object()
_CERTIFICATE_TOKEN: Final = object()
_UNCERTAINTY_GROUP_TOKEN: Final = object()
_UNCERTAINTY_FAMILY_TOKEN: Final = object()
_Q_DOMAIN_BRIDGE_TOKEN: Final = object()
_SAMPLED_UNCERTAINTY_GROUP_TOKEN: Final = object()
_SAMPLED_UNCERTAINTY_FAMILY_TOKEN: Final = object()
_ACTION_COORDINATE_CUSTODY_TOKEN: Final = object()


def _gamma(operation_count: int) -> float:
    count = max(1, _positive_int(operation_count, name="binary64 operation count"))
    product = count * _F64_EPS
    if product >= 0.25:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.NONFINITE_SOLVER,
            "binary64 reduction depth exceeds the registered error regime",
        )
    return math.nextafter(product / (1.0 - product), math.inf)


def _down(value: float) -> float:
    return value if not math.isfinite(value) else math.nextafter(value, -math.inf)


def _up(value: float) -> float:
    return value if not math.isfinite(value) else math.nextafter(value, math.inf)


def _outward_nonnegative_reduction_upper(
    center: float,
    *,
    operation_count: int,
) -> float:
    """Enclose a nonnegative binary64 reduction, including underflow.

    ``center`` may have been produced by a deterministic CUDA reduction.  A
    final ``nextafter`` is not an enclosure for a multi-operation reduction:
    the accumulated error can exceed one ulp.  The standard-model gamma bound
    covers normal rounding and the explicit minimum-subnormal mass covers
    gradual-underflow loss.
    """

    value = _finite_nonnegative(center, name="nonnegative reduction center")
    operations = max(
        1,
        _positive_int(operation_count, name="nonnegative reduction operation count"),
    )
    gamma = _gamma(operations)
    return _up(value / (1.0 - gamma) + _F64_MIN_SUBNORMAL * float(operations))


def _outward_nonnegative_tensor_sum(
    values: Any,
    *,
    operations_per_term: int,
) -> tuple[float, float]:
    """Return the ordinary center and a deterministic outward upper bound."""

    if torch is None or type(values) is not torch.Tensor:
        raise TypeError("outward tensor reduction requires one exact tensor")
    if values.dtype != torch.float64 or values.requires_grad:
        raise TypeError("outward tensor reduction requires detached float64 values")
    if not bool(torch.isfinite(values).all().item()) or bool(
        (values < 0.0).any().item()
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.NONFINITE_SOLVER,
            "outward nonnegative reduction received invalid terms",
        )
    center = float(torch.sum(values).item())
    operations = max(1, int(values.numel()) * max(1, int(operations_per_term)))
    return center, _outward_nonnegative_reduction_upper(
        center,
        operation_count=operations,
    )


def _stable_l2_norm_center_upper(*values: Any) -> tuple[float, float]:
    """Scale before squaring so tiny or huge dual coordinates remain visible."""

    tensors = tuple(value for value in values if int(value.numel()) > 0)
    if not tensors:
        return 0.0, 0.0
    if any(
        torch is None
        or type(value) is not torch.Tensor
        or value.dtype != torch.float64
        or value.requires_grad
        or not bool(torch.isfinite(value).all().item())
        for value in tensors
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.NONFINITE_SOLVER,
            "scaled L2 norm received invalid tensors",
        )
    scale = max(float(torch.max(torch.abs(value)).item()) for value in tensors)
    if scale == 0.0:
        return 0.0, 0.0
    normalized_square = sum(
        (torch.sum((value / scale) * (value / scale)) for value in tensors),
        torch.zeros((), dtype=torch.float64, device=tensors[0].device),
    )
    normalized_center = float(normalized_square.item())
    term_count = sum(int(value.numel()) for value in tensors)
    normalized_upper = _outward_nonnegative_reduction_upper(
        normalized_center,
        operation_count=max(8, 6 * term_count + 2 * len(tensors)),
    )
    center = scale * math.sqrt(max(0.0, normalized_center))
    upper = _up(scale * math.sqrt(normalized_upper))
    if not math.isfinite(center) or not math.isfinite(upper):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.NONFINITE_SOLVER,
            "scaled L2 norm overflowed",
        )
    return center, upper


def _bounded_fp64_tensor_digest(value: Any, *, role: str) -> str:
    """Hash the exact bounded dual vector, never only a norm summary."""

    if (
        torch is None
        or type(value) is not torch.Tensor
        or value.dtype != torch.float64
        or value.requires_grad
        or int(value.numel()) > _MAXIMUM_DUAL_WITNESS_SCALARS
        or not bool(torch.isfinite(value).all().item())
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.NONFINITE_SOLVER,
            "dual witness tensor is invalid or exceeds its bounded replay domain",
        )
    host = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256(memoryview(host).cast("B")).hexdigest()
    return content_hash(
        {
            "schema": "partition_free_v1_bounded_fp64_tensor_digest_v1",
            "role": _identity(role, name="dual witness tensor role"),
            "shape": tuple(int(item) for item in host.shape),
            "dtype": "float64",
            "little_endian_payload_sha256": digest,
        },
        prefix=_PREFIX + "dual-vector-",
    )


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    """Capture the process-local mutation identity of one live tensor."""

    if torch is None or type(value) is not torch.Tensor:
        raise TypeError("compact Phase-B custody requires exact tensors")
    return (
        id(value),
        int(value._version),
        int(value.data_ptr()),
        tuple(int(item) for item in value.shape),
        str(value.dtype),
        str(value.device),
    )


class V1SignedWorkVariableShellActionSeparationMode(str, Enum):
    CUDA_PRODUCTION = "cuda_production"
    EXACT_REFERENCE_TEST_ONLY = "exact_reference_test_only"


class V1SignedWorkVariableShellQBranch(str, Enum):
    SPECTRAL = "spectral"
    DIAGONAL_L1 = "diagonal_l1"


class V1SignedWorkVariableShellActionSeparationRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    SHELL_TOO_LARGE = "shell_too_large"
    INCOMPLETE_BRANCH_AXIS = "incomplete_branch_axis"
    INCOMPLETE_SUBSET_AXIS = "incomplete_subset_axis"
    INCOMPLETE_ACTION_CELL_AXIS = "incomplete_action_cell_axis"
    DUPLICATE_ALIAS = "duplicate_alias"
    AXIS_MISMATCH = "axis_mismatch"
    STALE_SOURCE = "stale_source"
    EVICTED_SOURCE = "evicted_source"
    CUDA_REQUIRED = "cuda_required"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    PRIMAL_INFEASIBLE = "primal_infeasible"
    INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY = "incomplete_source_uncertainty_family"
    INCOMPLETE_PROBLEM_AXIS = "incomplete_problem_axis"
    PRIMAL_FEASIBILITY_UNRESOLVED = "primal_feasibility_unresolved"
    HARD_REPLAY_INFEASIBLE = "hard_replay_infeasible"
    DETERMINISTIC_ROUNDING_MISMATCH = "deterministic_rounding_mismatch"
    DUAL_GAP_UNRESOLVED = "dual_gap_unresolved"
    NONFINITE_SOLVER = "nonfinite_solver"
    OUTSIDE_SHELL_EXCLUSION_UNAVAILABLE = "outside_shell_exclusion_unavailable"
    REGISTERED_Q_BRANCH_DOMAIN_BRIDGE_UNAVAILABLE = (
        "registered_q_branch_domain_bridge_unavailable"
    )
    GLOBAL_RESTRICTED_SOLVER_UNAVAILABLE = "global_restricted_solver_unavailable"
    INCOMPLETE_CELL_PROOF = "incomplete_cell_proof"
    INCOMPLETE_SOURCE_UNCERTAINTY_COVERAGE = "incomplete_source_uncertainty_coverage"
    NONSTRICT_ACTION_SEPARATION = "nonstrict_action_separation"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkVariableShellActionSeparationRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkVariableShellActionSeparationRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkVariableShellActionSeparationRefusalCode:
            raise TypeError("action-separation refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkVariableShellActionSeparationRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkVariableShellActionSeparationRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
            f"{name} must be positive",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_TYPE,
            f"{name} must be one finite nonnegative real",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
            f"{name} must be finite and nonnegative",
        )
    return 0.0 if result == 0.0 else result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_TYPE,
            f"{name} must be one finite real",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
            f"{name} must be finite",
        )
    return 0.0 if result == 0.0 else result


def _finite_positive(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result <= 0.0:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
            f"{name} must be positive",
        )
    return result


def _fraction(value: Real, *, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_TYPE,
            f"{name} must be real",
        )
    scalar = float(value)
    if not math.isfinite(scalar):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
            f"{name} must be finite",
        )
    return Fraction.from_float(scalar)


def _fraction_binary64_center_radius(value: Fraction) -> tuple[float, float]:
    """Return nearest binary64 plus a strict radius enclosing ``value``."""

    if type(value) is not Fraction or value < 0:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_TYPE,
            "exact nonnegative weight must be one Fraction",
        )
    center = float(value)
    if not math.isfinite(center):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.NONFINITE_SOLVER,
            "exact population weight overflowed binary64",
        )
    represented = Fraction.from_float(center)
    lower = center if represented <= value else _down(center)
    upper = center if represented >= value else _up(center)
    radius = max(center - lower, upper - center)
    return center, 0.0 if radius == 0.0 else _up(radius)


def _offset_axis(
    value: object,
    *,
    shell_size: int,
    name: str,
    allow_empty: bool = True,
) -> tuple[int, ...]:
    if type(value) is not tuple:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_TYPE,
            f"{name} must be one immutable tuple",
        )
    result = tuple(_nonnegative_int(item, name=f"{name} item") for item in value)
    if result != tuple(sorted(set(result))):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            f"{name} must be sorted and unique",
        )
    if not allow_empty and not result:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            f"{name} cannot be empty",
        )
    if result and result[-1] >= shell_size:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            f"{name} is outside the shell axis",
        )
    return result


def _closed_authority(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool for value in values) or any(
        value is not False for value in values
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its authority boundary",
        )


def _action_coordinate_alias(
    canonical: object,
    legacy: object = None,
    *,
    name: str,
) -> str:
    """Resolve legacy byte-action names onto the canonical coordinate hash."""

    if canonical is None and legacy is None:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_TYPE,
            f"{name} requires a physical action coordinate hash",
        )
    if canonical is not None and legacy is not None and canonical != legacy:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            f"{name} legacy byte-action alias differs from coordinate hash",
        )
    return _identity(canonical if canonical is not None else legacy, name=name)


def _complete_subset_axis(shell_size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        subset
        for cardinality in range(shell_size + 1)
        for subset in combinations(range(shell_size), cardinality)
    )


def _validated_gates(shell_gates: Sequence[Real], shell_size: int) -> tuple[float, ...]:
    if isinstance(shell_gates, (str, bytes)) or len(shell_gates) != shell_size:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            "shell gate vector does not align with the frozen shell",
        )
    result: list[float] = []
    for value in shell_gates:
        scalar = float(_fraction(value, name="shell gate"))
        if not 0.0 <= scalar <= 1.0:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
                "shell gates must lie in the closed unit box",
            )
        result.append(scalar)
    return tuple(result)


def deterministic_v1_signed_work_variable_shell_rounding_subset(
    *,
    shell_gates: Sequence[Real],
    shell_route_indices: Sequence[int],
    shell_prefix_count: int,
) -> tuple[int, ...]:
    """Unique physical-action subset: descending gate, then route index."""

    if isinstance(shell_route_indices, (str, bytes)):
        raise TypeError("shell route indices must be one bounded sequence")
    shell = len(shell_route_indices)
    gates = _validated_gates(shell_gates, shell)
    physical = tuple(
        _nonnegative_int(value, name="shell physical route index")
        for value in shell_route_indices
    )
    if len(set(physical)) != shell:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            "shell physical route indices must be unique",
        )
    cardinality = _nonnegative_int(shell_prefix_count, name="shell prefix count")
    if cardinality > shell:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            "rounding cardinality exceeds the shell",
        )
    order = tuple(
        sorted(range(shell), key=lambda index: (-gates[index], physical[index]))
    )
    return tuple(sorted(order[:cardinality]))


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellActionSeparationCUDAExecutionDesign:
    maximum_shell_size: int = _MAXIMUM_SHELL_SIZE
    maximum_action_cell_count: int = _MAXIMUM_ACTION_CELLS
    maximum_resource_alias_count: int = _MAXIMUM_SHELL_SIZE + 1
    maximum_concurrent_cells: int = 16
    maximum_pdhg_iterations: int = 20_000
    minimum_pdhg_iterations: int = 64
    pdhg_gap_check_interval: int = 16
    absolute_duality_gap_tolerance: float = 1.0e-8
    relative_duality_gap_tolerance: float = 1.0e-8
    operator_norm_safety_factor: float = 0.99
    pdhg_extrapolation_theta: float = 1.0
    hbm_ceiling_fraction: float = 0.85
    execution_mode: V1SignedWorkVariableShellActionSeparationMode = (
        V1SignedWorkVariableShellActionSeparationMode.CUDA_PRODUCTION
    )
    deterministic_fp64_required: bool = True
    closed_subset_cells_required: bool = True
    ties_in_every_compatible_cell_required: bool = True
    complete_cell_coverage_required: bool = True
    permutation_cells_production_authority: bool = False
    route_scale_d2h_forbidden: bool = True
    schema: str = ACTION_SEPARATION_DESIGN_SCHEMA
    solver_backend: str = ACTION_CELL_SOLVER_BACKEND

    def __post_init__(self) -> None:
        shell = _positive_int(self.maximum_shell_size, name="maximum shell size")
        if shell > _MAXIMUM_SHELL_SIZE:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.SHELL_TOO_LARGE,
                "canonical action separation is bounded by B=8",
            )
        object.__setattr__(self, "maximum_shell_size", shell)
        cells = _positive_int(
            self.maximum_action_cell_count,
            name="maximum action-cell count",
        )
        if cells > _MAXIMUM_ACTION_CELLS or cells < 1 << shell:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
                "action-cell ceiling must cover the shell powerset quotient",
            )
        object.__setattr__(self, "maximum_action_cell_count", cells)
        aliases = _positive_int(
            self.maximum_resource_alias_count,
            name="maximum resource alias count",
        )
        if aliases > shell + 1:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
                "one branch may register at most one Phase-A alias per cardinality",
            )
        object.__setattr__(self, "maximum_resource_alias_count", aliases)
        object.__setattr__(
            self,
            "maximum_concurrent_cells",
            _positive_int(
                self.maximum_concurrent_cells,
                name="maximum concurrent cells",
            ),
        )
        maximum_iterations = _positive_int(
            self.maximum_pdhg_iterations,
            name="maximum PDHG iterations",
        )
        minimum_iterations = _positive_int(
            self.minimum_pdhg_iterations,
            name="minimum PDHG iterations",
        )
        interval = _positive_int(
            self.pdhg_gap_check_interval,
            name="PDHG gap-check interval",
        )
        if minimum_iterations > maximum_iterations or interval > maximum_iterations:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
                "PDHG iteration/check axes are inconsistent",
            )
        object.__setattr__(self, "maximum_pdhg_iterations", maximum_iterations)
        object.__setattr__(self, "minimum_pdhg_iterations", minimum_iterations)
        object.__setattr__(self, "pdhg_gap_check_interval", interval)
        for name in (
            "absolute_duality_gap_tolerance",
            "relative_duality_gap_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name.replace("_", " ")),
            )
        safety = _finite_positive(
            self.operator_norm_safety_factor,
            name="operator-norm safety factor",
        )
        if safety >= 1.0:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
                "PDHG operator-norm safety factor must be strictly below one",
            )
        object.__setattr__(self, "operator_norm_safety_factor", safety)
        theta = _finite_nonnegative(
            self.pdhg_extrapolation_theta,
            name="PDHG extrapolation theta",
        )
        if theta > 1.0:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
                "canonical PDHG extrapolation theta must lie in [0,1]",
            )
        object.__setattr__(self, "pdhg_extrapolation_theta", theta)
        ceiling = _finite_positive(
            self.hbm_ceiling_fraction,
            name="HBM ceiling fraction",
        )
        if ceiling > 0.85:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
                "canonical HBM ceiling cannot exceed 85 percent",
            )
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        if (
            type(self.execution_mode)
            is not V1SignedWorkVariableShellActionSeparationMode
        ):
            raise TypeError("execution mode must be exact")
        required_true = (
            self.deterministic_fp64_required,
            self.closed_subset_cells_required,
            self.ties_in_every_compatible_cell_required,
            self.complete_cell_coverage_required,
            self.route_scale_d2h_forbidden,
        )
        if any(type(value) is not bool for value in required_true) or any(
            value is not True for value in required_true
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
                "determinism, closed coverage, and no route D2H are mandatory",
            )
        if self.permutation_cells_production_authority is not False:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "factorial permutation chambers cannot become a production solve axis",
            )
        if (
            self.schema != ACTION_SEPARATION_DESIGN_SCHEMA
            or self.solver_backend != ACTION_CELL_SOLVER_BACKEND
        ):
            raise ValueError("unsupported action-separation design")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellStreamedHashCustodyFamilyReceipt:
    """Source-bound custody for the complete hard-action coordinate family.

    The upstream CUDA scan proves the exact mapping from every bounded-shell
    subset to its removed-route set.  Phase B joins that mapping to the exact
    parent-mask coordinate, but deliberately does *not* claim that a packed
    child mask was materialized for every subset.  A selected action must still
    pass the dedicated packed-mask materialization crosswalk before C.
    """

    source_receipt_hash: str
    source_replay_hash: str
    upstream_hash_custody_receipt_hash: str
    upstream_hash_custody_hash: str
    upstream_private_primal_binding_hash: str
    source_evidence_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    fixed_prefix_count: int
    complete_subset_mapping_commitment: str
    upstream_authority_kind: str = ACTION_LEGACY_HASH_CUSTODY_AUTHORITY_KIND
    parent_mask_required: bool = True
    streamed_subset_hashes_required: bool = True
    exact_action_coordinates_required: bool = True
    route_scale_payload_materialized_here: bool = False
    packed_child_masks_materialized_here: bool = False
    packed_byte_action_hash_authority: bool = False
    downstream_authority: bool = False
    schema: str = ACTION_STREAMED_HASH_CUSTODY_FAMILY_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _ACTION_COORDINATE_CUSTODY_TOKEN:
            raise TypeError("action-coordinate custody receipts are issuer-only")
        for name in (
            "source_receipt_hash",
            "source_replay_hash",
            "upstream_hash_custody_receipt_hash",
            "upstream_hash_custody_hash",
            "upstream_private_primal_binding_hash",
            "upstream_authority_kind",
            "source_evidence_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "shell_freeze_receipt_hash",
            "complete_subset_mapping_commitment",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="physical route count"),
        )
        object.__setattr__(
            self,
            "fixed_prefix_count",
            _nonnegative_int(self.fixed_prefix_count, name="fixed prefix count"),
        )
        if (
            type(self.shell_indices) is not tuple
            or type(self.shell_route_ids) is not tuple
        ):
            raise TypeError("streamed hash custody shell axes must be immutable tuples")
        if (
            not self.shell_indices
            or len(self.shell_indices) != len(self.shell_route_ids)
            or len(set(self.shell_indices)) != len(self.shell_indices)
            or len(set(self.shell_route_ids)) != len(self.shell_route_ids)
            or any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in self.shell_indices
            )
            or any(type(value) is not str for value in self.shell_route_ids)
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "streamed hash custody shell axes do not replay",
            )
        object.__setattr__(
            self,
            "shell_indices",
            tuple(
                _nonnegative_int(value, name="shell route index")
                for value in self.shell_indices
            ),
        )
        tuple(_identity(value, name="shell route id") for value in self.shell_route_ids)
        required_true = (
            self.parent_mask_required,
            self.streamed_subset_hashes_required,
            self.exact_action_coordinates_required,
        )
        if self.upstream_authority_kind not in (
            ACTION_LEGACY_HASH_CUSTODY_AUTHORITY_KIND,
            ACTION_SOURCE_NATIVE_HASH_CUSTODY_AUTHORITY_KIND,
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "streamed hash custody family has an unknown upstream authority kind",
            )
        if any(type(value) is not bool for value in required_true) or any(
            value is not True for value in required_true
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "streamed hash custody lost a mandatory binding",
            )
        _closed_authority(
            (
                self.route_scale_payload_materialized_here,
                self.packed_child_masks_materialized_here,
                self.packed_byte_action_hash_authority,
                self.downstream_authority,
            ),
            owner="streamed action hash custody family",
        )
        if self.schema != ACTION_STREAMED_HASH_CUSTODY_FAMILY_SCHEMA:
            raise ValueError("unsupported streamed hash custody schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "source_receipt_hash": self.source_receipt_hash,
                "source_replay_hash": self.source_replay_hash,
                "upstream_hash_custody_receipt_hash": (
                    self.upstream_hash_custody_receipt_hash
                ),
                "upstream_hash_custody_hash": self.upstream_hash_custody_hash,
                "upstream_private_primal_binding_hash": (
                    self.upstream_private_primal_binding_hash
                ),
                "upstream_authority_kind": self.upstream_authority_kind,
                "source_evidence_hash": self.source_evidence_hash,
                "parent_mask_hash": self.parent_mask_hash,
                "route_axis_hash": self.route_axis_hash,
                "route_id_axis_hash": self.route_id_axis_hash,
                "route_count": self.route_count,
                "shell_freeze_receipt_hash": self.shell_freeze_receipt_hash,
                "shell_indices": self.shell_indices,
                "shell_route_ids": self.shell_route_ids,
                "fixed_prefix_count": self.fixed_prefix_count,
                "complete_subset_mapping_commitment": (
                    self.complete_subset_mapping_commitment
                ),
                "parent_mask_required": True,
                "streamed_subset_hashes_required": True,
                "exact_action_coordinates_required": True,
                "route_scale_payload_materialized_here": False,
                "packed_child_masks_materialized_here": False,
                "packed_byte_action_hash_authority": False,
                "downstream_authority": False,
            },
            prefix=_PREFIX + "streamed-hash-custody-family-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSourceNativeHashCustodyReceipt:
    """Source-native D_W2 action-coordinate custody for signed-work v1.

    This receipt binds the neutral streamed subset-hash family directly to the
    same-capture variable-shell source and current signed-work KKT lineage.  It
    deliberately avoids the legacy relaxed-KKT private-primal receipt surface:
    the only imported primitive is the array-free complete powerset hash
    family, which is a digest of bounded membership slabs rather than an
    authority over F, Q, C, K, O, commits, or Pareto claims.
    """

    source_receipt_hash: str
    source_replay_hash: str
    source_evidence_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    kkt_branch_id: str
    kkt_branch_source_receipt_hash: str
    signed_work_problem_hash: str
    signed_work_policy_hash: str
    signed_work_branch_execution_hash: str
    device_shell_execution_receipt_hash: str
    device_shell_ladder_hash: str
    streamed_hash_family_hash: str
    shell_route_index_hash: str
    fixed_prefix_count: int
    fixed_suffix_count: int
    eligible_route_count: int
    complete_subset_mapping_commitment: str
    registered_endpoint_shell_offsets: tuple[tuple[int, ...], ...]
    label_d2h_bytes_total: int
    maximum_label_slab_bytes_observed: int
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    source_native_signed_work_kkt_lineage: bool = True
    complete_physical_scan: bool = True
    legacy_relaxed_kkt_authority_used: bool = False
    route_scale_host_row_retention_used: bool = False
    full_key_or_index_d2h_used: bool = False
    packed_child_masks_materialized_here: bool = False
    downstream_authority: bool = False
    schema: str = ACTION_SOURCE_NATIVE_HASH_CUSTODY_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _ACTION_COORDINATE_CUSTODY_TOKEN:
            raise TypeError("source-native hash custody receipts are issuer-only")
        for name in (
            "source_receipt_hash",
            "source_replay_hash",
            "source_evidence_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "shell_freeze_receipt_hash",
            "kkt_branch_id",
            "kkt_branch_source_receipt_hash",
            "signed_work_problem_hash",
            "signed_work_policy_hash",
            "signed_work_branch_execution_hash",
            "device_shell_execution_receipt_hash",
            "device_shell_ladder_hash",
            "streamed_hash_family_hash",
            "shell_route_index_hash",
            "complete_subset_mapping_commitment",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="physical route count"),
        )
        for name in (
            "fixed_prefix_count",
            "fixed_suffix_count",
            "eligible_route_count",
            "label_d2h_bytes_total",
            "maximum_label_slab_bytes_observed",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if (
            type(self.shell_indices) is not tuple
            or type(self.shell_route_ids) is not tuple
        ):
            raise TypeError("source-native hash custody shell axes must be tuples")
        shell_indices = tuple(
            _nonnegative_int(value, name="shell route index")
            for value in self.shell_indices
        )
        if (
            not shell_indices
            or shell_indices != tuple(sorted(set(shell_indices)))
            or shell_indices[-1] >= self.route_count
            or len(shell_indices) > _MAXIMUM_SHELL_SIZE
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "source-native hash custody has an invalid bounded shell",
            )
        object.__setattr__(self, "shell_indices", shell_indices)
        shell_route_ids = tuple(
            _identity(value, name="shell route id") for value in self.shell_route_ids
        )
        if len(shell_route_ids) != len(shell_indices) or len(
            set(shell_route_ids)
        ) != len(shell_route_ids):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "source-native hash custody route-id shell does not close",
            )
        object.__setattr__(self, "shell_route_ids", shell_route_ids)
        endpoints = tuple(
            tuple(value) for value in self.registered_endpoint_shell_offsets
        )
        if any(
            offsets != tuple(sorted(set(offsets)))
            or any(offset < 0 or offset >= len(shell_indices) for offset in offsets)
            for offsets in endpoints
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "source-native hash custody endpoints are not shell offsets",
            )
        object.__setattr__(self, "registered_endpoint_shell_offsets", endpoints)
        flags = (
            self.source_native_signed_work_kkt_lineage,
            self.complete_physical_scan,
            self.legacy_relaxed_kkt_authority_used,
            self.route_scale_host_row_retention_used,
            self.full_key_or_index_d2h_used,
            self.packed_child_masks_materialized_here,
            self.downstream_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("source-native hash custody flags must be exact booleans")
        if flags != (True, True, False, False, False, False, False):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "source-native hash custody crossed its authority boundary",
            )
        if self.label_d2h_bytes_total != self.route_count:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "source-native hash custody must account for one label byte per route",
            )
        if self.eligible_route_count != (
            self.fixed_prefix_count + self.fixed_suffix_count + len(shell_indices)
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "source-native hash custody eligible census does not close",
            )
        if (
            self.source_evidence_hash != self.corrected_evidence_stream_source_hash
            or self.schema != ACTION_SOURCE_NATIVE_HASH_CUSTODY_SCHEMA
        ):
            raise ValueError("unsupported or mismatched source-native hash custody")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix=_PREFIX + "source-native-hash-custody-receipt-",
        )

    @property
    def custody_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "receipt_hash": self.receipt_hash,
                "streamed_hash_family_hash": self.streamed_hash_family_hash,
                "complete_subset_mapping_commitment": (
                    self.complete_subset_mapping_commitment
                ),
                "source_native_signed_work_kkt_lineage": True,
            },
            prefix=_PREFIX + "source-native-hash-custody-",
        )

    @property
    def lineage_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "source_receipt_hash": self.source_receipt_hash,
                "kkt_branch_id": self.kkt_branch_id,
                "kkt_branch_source_receipt_hash": self.kkt_branch_source_receipt_hash,
                "signed_work_problem_hash": self.signed_work_problem_hash,
                "signed_work_policy_hash": self.signed_work_policy_hash,
                "signed_work_branch_execution_hash": (
                    self.signed_work_branch_execution_hash
                ),
                "device_shell_execution_receipt_hash": (
                    self.device_shell_execution_receipt_hash
                ),
                "device_shell_ladder_hash": self.device_shell_ladder_hash,
            },
            prefix=_PREFIX + "source-native-action-coordinate-lineage-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellActionSubsetIdentity:
    shell_size: int
    shell_offsets: tuple[int, ...]
    removed_count: int
    removed_route_index_hash: str
    physical_action_coordinate_hash: str
    streamed_hash_custody_family_hash: str
    parent_mask_hash: str
    schema: str = ACTION_SUBSET_IDENTITY_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _ACTION_COORDINATE_CUSTODY_TOKEN:
            raise TypeError("action-subset identities are issuer-only")
        shell = _positive_int(self.shell_size, name="shell size")
        if shell > _MAXIMUM_SHELL_SIZE:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.SHELL_TOO_LARGE,
                "subset identity exceeds B=8",
            )
        object.__setattr__(self, "shell_size", shell)
        object.__setattr__(
            self,
            "shell_offsets",
            _offset_axis(
                self.shell_offsets,
                shell_size=shell,
                name="shell offsets",
            ),
        )
        object.__setattr__(
            self,
            "removed_count",
            _nonnegative_int(self.removed_count, name="removed route count"),
        )
        for name in (
            "removed_route_index_hash",
            "physical_action_coordinate_hash",
            "streamed_hash_custody_family_hash",
            "parent_mask_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.schema != ACTION_SUBSET_IDENTITY_SCHEMA:
            raise ValueError("unsupported subset identity schema")

    @property
    def subset_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "subset-")

    @property
    def byte_physical_action_hash(self) -> str:
        """Legacy wire alias; this is an action coordinate, not packed bytes."""

        return self.physical_action_coordinate_hash


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellActionCoordinateCustodyIssuance:
    """Issuer-only complete powerset of exact hard-action coordinates."""

    family_receipt: V1SignedWorkVariableShellStreamedHashCustodyFamilyReceipt
    subset_identities: tuple[V1SignedWorkVariableShellActionSubsetIdentity, ...]
    complete_subset_axis: bool = True
    packed_action_materialization_authority: bool = False
    downstream_authority: bool = False
    schema: str = ACTION_STREAMED_HASH_CUSTODY_ISSUANCE_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _ACTION_COORDINATE_CUSTODY_TOKEN:
            raise TypeError("action-coordinate custody issuances are issuer-only")
        if type(self.family_receipt) is not (
            V1SignedWorkVariableShellStreamedHashCustodyFamilyReceipt
        ):
            raise TypeError(
                "action-coordinate custody requires its exact family receipt"
            )
        subsets = tuple(self.subset_identities)
        shell = len(self.family_receipt.shell_indices)
        expected_offsets = _complete_subset_axis(shell)
        observed_offsets = (
            tuple(
                sorted(
                    (value.shell_offsets for value in subsets),
                    key=lambda value: (len(value), value),
                )
            )
            if all(
                type(value) is V1SignedWorkVariableShellActionSubsetIdentity
                for value in subsets
            )
            else ()
        )
        if (
            len(subsets) != 1 << shell
            or observed_offsets != expected_offsets
            or any(value.shell_size != shell for value in subsets)
            or any(
                value.streamed_hash_custody_family_hash
                != self.family_receipt.receipt_hash
                for value in subsets
            )
            or any(
                value.parent_mask_hash != self.family_receipt.parent_mask_hash
                for value in subsets
            )
            or len({value.removed_route_index_hash for value in subsets})
            != len(subsets)
            or len({value.physical_action_coordinate_hash for value in subsets})
            != len(subsets)
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SUBSET_AXIS,
                "action-coordinate custody does not contain the exact shell powerset",
            )
        object.__setattr__(self, "subset_identities", subsets)
        if (
            type(self.complete_subset_axis) is not bool
            or self.complete_subset_axis is not True
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "action-coordinate custody lost complete powerset coverage",
            )
        _closed_authority(
            (
                self.packed_action_materialization_authority,
                self.downstream_authority,
            ),
            owner="action-coordinate custody issuance",
        )
        if self.schema != ACTION_STREAMED_HASH_CUSTODY_ISSUANCE_SCHEMA:
            raise ValueError("unsupported action-coordinate custody issuance schema")

    @property
    def issuance_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "family_receipt_hash": self.family_receipt.receipt_hash,
                "subset_hashes": tuple(
                    value.subset_hash for value in self.subset_identities
                ),
                "complete_subset_axis": True,
                "packed_action_materialization_authority": False,
                "downstream_authority": False,
            },
            prefix=_PREFIX + "action-coordinate-custody-issuance-",
        )


def issue_v1_signed_work_variable_shell_action_coordinate_custody(
    *,
    source_receipt: V1SignedWorkVariableShellCompactSourceReceipt,
    hash_custody_issuance: V1DecisionLevelCudaHashCustodyIssuance,
    private_primal_binding: V1DecisionLevelCudaPrivatePrimalBinding,
) -> V1SignedWorkVariableShellActionCoordinateCustodyIssuance:
    """Join exact streamed removed-set custody to one parent-mask coordinate.

    This joint is bounded by ``2**B <= 256``.  It neither receives route-scale
    indices nor manufactures packed child masks.  Its action coordinate is the
    exact parent mask plus exact removed-set identity; the eventual winner must
    still be byte-materialized and cross-checked before C.
    """

    if not _is_compact_source_receipt(source_receipt):
        raise TypeError(
            "source_receipt must be an exact one-shot or segmented variable-shell receipt"
        )
    if type(hash_custody_issuance) is not V1DecisionLevelCudaHashCustodyIssuance:
        raise TypeError("hash_custody_issuance must be exact CUDA hash custody")
    if type(private_primal_binding) is not V1DecisionLevelCudaPrivatePrimalBinding:
        raise TypeError("private_primal_binding must be exact CUDA primal binding")

    custody = hash_custody_issuance.custody
    upstream = hash_custody_issuance.receipt
    binding = private_primal_binding
    shell = len(source_receipt.shell_indices)
    expected_shell_hash = RemovedRouteIndexSet.from_indices(
        source_receipt.shell_indices,
        route_count=source_receipt.route_count,
    ).index_hash
    mapping = dict(custody.endpoint_hash_by_shell_offsets)
    expected_offsets = set(_complete_subset_axis(shell))
    if (
        upstream.route_axis_hash != source_receipt.route_axis_hash
        or upstream.route_count != source_receipt.route_count
        or upstream.shell_count != shell
        or upstream.shell_route_index_hash != expected_shell_hash
        or upstream.shell_route_index_hash != custody.shell_route_index_hash
        or upstream.custody_hash != custody.custody_hash
        or upstream.complete_subset_mapping_commitment
        != custody.outcome_ladder_axis_hash
        or set(mapping) != expected_offsets
        or binding.route_axis_hash != source_receipt.route_axis_hash
        or binding.route_count != source_receipt.route_count
        or binding.parent_mask_hash != source_receipt.current_mask_hash
        or binding.hash_custody_hash != custody.custody_hash
        or binding.shell_route_index_hash != expected_shell_hash
        or binding.source_evidence_hash != custody.source_evidence_hash
        or binding.source_evidence_hash
        != source_receipt.corrected_evidence_stream_source_hash
        or binding.branch != source_receipt.kkt_branch_id
        or binding.branch_receipt_hash != source_receipt.kkt_branch_source_receipt_hash
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
            "CUDA removed-set custody, private-primal parent, and variable-shell source do not close",
        )
    if len(set(mapping.values())) != len(mapping):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SUBSET_AXIS,
            "streamed custody aliases two distinct shell subsets to one removed-set identity",
        )

    return _issue_v1_signed_work_variable_shell_action_coordinate_custody_from_mapping(
        source_receipt=source_receipt,
        upstream_hash_custody_receipt_hash=upstream.receipt_hash,
        upstream_hash_custody_hash=custody.custody_hash,
        upstream_private_primal_binding_hash=binding.binding_hash,
        upstream_authority_kind=ACTION_LEGACY_HASH_CUSTODY_AUTHORITY_KIND,
        source_evidence_hash=binding.source_evidence_hash,
        fixed_prefix_count=custody.fixed_prefix_count,
        complete_subset_mapping_commitment=upstream.complete_subset_mapping_commitment,
        mapping=mapping,
    )


def stream_v1_signed_work_variable_shell_source_native_hash_family_from_device_shell_labels(
    *,
    source_receipt: V1SignedWorkVariableShellCompactSourceReceipt,
    label_handle: V1DeviceShellLabelHandle,
    frozen_ladder_receipt: V1DeviceShellFrozenLadderReceipt,
    registered_endpoint_counts: Sequence[int] | None = None,
    registered_endpoint_hashes: Sequence[str] | None = None,
    maximum_label_slab_bytes: int = MAXIMUM_HOST_LABEL_SLAB_BYTES,
) -> V1DecisionLevelCudaStreamedHashFamily:
    """Stream complete shell-subset hashes from live device-shell labels.

    The device-shell ladder uses endpoint-inclusion labels.  The neutral
    decision-level hash primitive instead expects three classes: fixed prefix,
    fixed suffix, and bounded shell offsets.  This bridge performs only that
    reversible relabeling in bounded host slabs:

    * endpoint label ``1`` outside the variable shell is fixed prefix;
    * endpoint labels ``>1`` outside the variable shell are fixed suffix;
    * every source-shell route is overwritten with ``SHELL_LABEL_BASE+offset``;
    * label ``0`` stays ignored/ineligible for this local coordinate hash.

    The function imports no legacy relaxed-KKT authority and produces no action
    custody by itself.  Its result must still be joined by the source-native
    action-coordinate issuer.
    """

    if not _is_compact_source_receipt(source_receipt):
        raise TypeError(
            "source_receipt must be an exact one-shot or segmented variable-shell receipt"
        )
    if type(label_handle) is not V1DeviceShellLabelHandle:
        raise TypeError("label_handle must be an exact device-shell label handle")
    if type(frozen_ladder_receipt) is not V1DeviceShellFrozenLadderReceipt:
        raise TypeError("frozen_ladder_receipt must be an exact frozen ladder receipt")
    route_count = source_receipt.route_count
    if (
        label_handle.route_count != route_count
        or frozen_ladder_receipt.route_count != route_count
        or frozen_ladder_receipt.route_axis_hash != source_receipt.route_axis_hash
        or frozen_ladder_receipt.parent_mask_hash != source_receipt.current_mask_hash
        or frozen_ladder_receipt.endpoint_count != len(frozen_ladder_receipt.actions)
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
            "device-shell labels, frozen ladder, and variable-shell source differ",
        )
    shell_indices = source_receipt.shell_indices
    shell_index_hash = RemovedRouteIndexSet.from_indices(
        shell_indices,
        route_count=route_count,
    ).index_hash
    shell_offsets_by_index = {
        route: offset for offset, route in enumerate(shell_indices)
    }
    shell_endpoint_labels = tuple(
        int(label_handle.copy_host_chunk(route, route + 1)[0])
        for route in shell_indices
    )
    shell_active_count = sum(1 for label in shell_endpoint_labels if label > 0)
    expected_eligible_count = (
        frozen_ladder_receipt.cumulative_counts[-1]
        - shell_active_count
        + len(shell_indices)
    )
    if expected_eligible_count < len(shell_indices):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            "device-shell cumulative count is smaller than the source shell",
        )
    if registered_endpoint_counts is None and registered_endpoint_hashes is None:
        endpoint_counts = tuple(
            action.cumulative_removed_count for action in frozen_ladder_receipt.actions
        )
        endpoint_hashes = tuple(
            action.removed_route_index_hash for action in frozen_ladder_receipt.actions
        )
    elif (
        registered_endpoint_counts is not None
        and registered_endpoint_hashes is not None
    ):
        endpoint_counts = tuple(
            _positive_int(value, name="registered endpoint count")
            for value in registered_endpoint_counts
        )
        endpoint_hashes = tuple(
            _identity(value, name="registered endpoint hash")
            for value in registered_endpoint_hashes
        )
    else:
        raise TypeError(
            "registered endpoint counts and hashes must be supplied together"
        )
    if not endpoint_counts or len(endpoint_counts) != len(endpoint_hashes):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            "registered endpoint count/hash axes do not close",
        )

    slab_bytes = _positive_int(
        maximum_label_slab_bytes,
        name="maximum label slab bytes",
    )
    chunk_routes = max(1, slab_bytes)

    def _membership_slabs() -> Iterator[tuple[int, np.ndarray]]:
        for start in range(0, route_count, chunk_routes):
            stop = min(route_count, start + chunk_routes)
            original = label_handle.copy_host_chunk(start, stop)
            if original.dtype != np.dtype("uint8") or original.shape != (stop - start,):
                _refuse(
                    V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                    "device-shell label slab is not one uint8 route interval",
                )
            recoded = np.zeros_like(original, dtype=np.uint8)
            recoded[original == 1] = PREFIX_LABEL
            recoded[original > 1] = SUFFIX_LABEL
            for route, offset in shell_offsets_by_index.items():
                if start <= route < stop:
                    recoded[route - start] = SHELL_LABEL_BASE + offset
            recoded.setflags(write=False)
            yield start, recoded

    return stream_v1_decision_level_hash_family_from_membership_slabs(
        route_count=route_count,
        shell_route_indices=shell_indices,
        expected_eligible_route_count=expected_eligible_count,
        expected_shell_route_index_hash=shell_index_hash,
        membership_slabs=_membership_slabs(),
        registered_endpoint_counts=endpoint_counts,
        registered_endpoint_hashes=endpoint_hashes,
        maximum_label_slab_bytes=slab_bytes,
    )


def issue_v1_signed_work_variable_shell_source_native_action_coordinate_custody(
    *,
    source_receipt: V1SignedWorkVariableShellCompactSourceReceipt,
    streamed_hash_family: V1DecisionLevelCudaStreamedHashFamily,
    signed_work_problem_hash: str,
    signed_work_policy_hash: str,
    signed_work_branch_execution_hash: str,
    device_shell_execution_receipt_hash: str,
    device_shell_ladder_hash: str,
) -> V1SignedWorkVariableShellActionCoordinateCustodyIssuance:
    """Issue exact shell action coordinates from source-native signed-work D_W2.

    This is the canonical V7 seam for the live common-coordinate path: the
    neutral streamed powerset hashes are joined directly to the same-capture
    corrected-evidence/literal-J source receipt and current signed-work KKT
    lineage.  No legacy relaxed-KKT private-primal binding participates.
    """

    if not _is_compact_source_receipt(source_receipt):
        raise TypeError(
            "source_receipt must be an exact one-shot or segmented variable-shell receipt"
        )
    if type(streamed_hash_family) is not V1DecisionLevelCudaStreamedHashFamily:
        raise TypeError("streamed_hash_family must be exact streamed hash family")
    shell = len(source_receipt.shell_indices)
    expected_shell_hash = RemovedRouteIndexSet.from_indices(
        source_receipt.shell_indices,
        route_count=source_receipt.route_count,
    ).index_hash
    mapping = dict(streamed_hash_family.endpoint_hash_by_shell_offsets)
    expected_offsets = set(_complete_subset_axis(shell))
    if (
        streamed_hash_family.route_count != source_receipt.route_count
        or streamed_hash_family.shell_route_indices != source_receipt.shell_indices
        or streamed_hash_family.shell_route_index_hash != expected_shell_hash
        or set(mapping) != expected_offsets
        or streamed_hash_family.complete_physical_scan is not True
        or streamed_hash_family.route_scale_host_row_retention_used is not False
        or streamed_hash_family.full_key_or_index_d2h_used is not False
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
            "source-native streamed family is not bound to the variable-shell source",
        )
    if len(set(mapping.values())) != len(mapping):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SUBSET_AXIS,
            "source-native streamed custody aliases two shell subsets",
        )

    native_receipt = V1SignedWorkVariableShellSourceNativeHashCustodyReceipt(
        source_receipt_hash=source_receipt.receipt_hash,
        source_replay_hash=source_receipt.source_replay_hash,
        source_evidence_hash=source_receipt.corrected_evidence_stream_source_hash,
        parent_mask_hash=source_receipt.current_mask_hash,
        route_axis_hash=source_receipt.route_axis_hash,
        route_id_axis_hash=source_receipt.route_id_axis_hash,
        route_count=source_receipt.route_count,
        shell_freeze_receipt_hash=source_receipt.shell_freeze_receipt_hash,
        shell_indices=source_receipt.shell_indices,
        shell_route_ids=source_receipt.shell_route_ids,
        kkt_branch_id=source_receipt.kkt_branch_id,
        kkt_branch_source_receipt_hash=source_receipt.kkt_branch_source_receipt_hash,
        signed_work_problem_hash=_identity(
            signed_work_problem_hash,
            name="signed-work problem hash",
        ),
        signed_work_policy_hash=_identity(
            signed_work_policy_hash,
            name="signed-work policy hash",
        ),
        signed_work_branch_execution_hash=_identity(
            signed_work_branch_execution_hash,
            name="signed-work branch execution hash",
        ),
        device_shell_execution_receipt_hash=_identity(
            device_shell_execution_receipt_hash,
            name="device-shell execution receipt hash",
        ),
        device_shell_ladder_hash=_identity(
            device_shell_ladder_hash,
            name="device-shell ladder hash",
        ),
        streamed_hash_family_hash=streamed_hash_family.family_hash,
        shell_route_index_hash=streamed_hash_family.shell_route_index_hash,
        fixed_prefix_count=streamed_hash_family.fixed_prefix_count,
        fixed_suffix_count=streamed_hash_family.fixed_suffix_count,
        eligible_route_count=streamed_hash_family.eligible_route_count,
        complete_subset_mapping_commitment=(
            streamed_hash_family.complete_subset_mapping_commitment
        ),
        registered_endpoint_shell_offsets=(
            streamed_hash_family.registered_endpoint_shell_offsets
        ),
        label_d2h_bytes_total=streamed_hash_family.label_d2h_bytes_total,
        maximum_label_slab_bytes_observed=(
            streamed_hash_family.maximum_label_slab_bytes_observed
        ),
        same_capture_vjp_source_hash=source_receipt.same_capture_vjp_source_hash,
        corrected_evidence_stream_source_hash=(
            source_receipt.corrected_evidence_stream_source_hash
        ),
        literal_j_stream_source_hash=source_receipt.literal_j_stream_source_hash,
        _issuer_token=_ACTION_COORDINATE_CUSTODY_TOKEN,
    )

    return _issue_v1_signed_work_variable_shell_action_coordinate_custody_from_mapping(
        source_receipt=source_receipt,
        upstream_hash_custody_receipt_hash=native_receipt.receipt_hash,
        upstream_hash_custody_hash=native_receipt.custody_hash,
        upstream_private_primal_binding_hash=native_receipt.lineage_hash,
        upstream_authority_kind=ACTION_SOURCE_NATIVE_HASH_CUSTODY_AUTHORITY_KIND,
        source_evidence_hash=source_receipt.corrected_evidence_stream_source_hash,
        fixed_prefix_count=streamed_hash_family.fixed_prefix_count,
        complete_subset_mapping_commitment=(
            streamed_hash_family.complete_subset_mapping_commitment
        ),
        mapping=mapping,
    )


def _issue_v1_signed_work_variable_shell_action_coordinate_custody_from_mapping(
    *,
    source_receipt: V1SignedWorkVariableShellCompactSourceReceipt,
    upstream_hash_custody_receipt_hash: str,
    upstream_hash_custody_hash: str,
    upstream_private_primal_binding_hash: str,
    upstream_authority_kind: str,
    source_evidence_hash: str,
    fixed_prefix_count: int,
    complete_subset_mapping_commitment: str,
    mapping: Mapping[tuple[int, ...], str],
) -> V1SignedWorkVariableShellActionCoordinateCustodyIssuance:
    shell = len(source_receipt.shell_indices)
    expected_offsets = set(_complete_subset_axis(shell))
    if set(mapping) != expected_offsets or len(set(mapping.values())) != len(mapping):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SUBSET_AXIS,
            "action-coordinate custody mapping is not the exact shell powerset",
        )
    family = V1SignedWorkVariableShellStreamedHashCustodyFamilyReceipt(
        source_receipt_hash=source_receipt.receipt_hash,
        source_replay_hash=source_receipt.source_replay_hash,
        upstream_hash_custody_receipt_hash=_identity(
            upstream_hash_custody_receipt_hash,
            name="upstream hash-custody receipt hash",
        ),
        upstream_hash_custody_hash=_identity(
            upstream_hash_custody_hash,
            name="upstream hash-custody hash",
        ),
        upstream_private_primal_binding_hash=_identity(
            upstream_private_primal_binding_hash,
            name="upstream private-primal/lineage binding hash",
        ),
        upstream_authority_kind=_identity(
            upstream_authority_kind,
            name="upstream authority kind",
        ),
        source_evidence_hash=_identity(
            source_evidence_hash, name="source evidence hash"
        ),
        parent_mask_hash=source_receipt.current_mask_hash,
        route_axis_hash=source_receipt.route_axis_hash,
        route_id_axis_hash=source_receipt.route_id_axis_hash,
        route_count=source_receipt.route_count,
        shell_freeze_receipt_hash=source_receipt.shell_freeze_receipt_hash,
        shell_indices=source_receipt.shell_indices,
        shell_route_ids=source_receipt.shell_route_ids,
        fixed_prefix_count=_nonnegative_int(
            fixed_prefix_count,
            name="fixed prefix count",
        ),
        complete_subset_mapping_commitment=(
            _identity(
                complete_subset_mapping_commitment,
                name="complete subset mapping commitment",
            )
        ),
        _issuer_token=_ACTION_COORDINATE_CUSTODY_TOKEN,
    )
    subsets: list[V1SignedWorkVariableShellActionSubsetIdentity] = []
    for offsets in _complete_subset_axis(shell):
        removed_count = family.fixed_prefix_count + len(offsets)
        removed_hash = _identity(
            mapping[offsets],
            name="streamed removed-route index hash",
        )
        coordinate_hash = content_hash(
            {
                "schema": "partition_free_v1_hard_action_coordinate_v1",
                "parent_mask_hash": source_receipt.current_mask_hash,
                "route_axis_hash": source_receipt.route_axis_hash,
                "route_id_axis_hash": source_receipt.route_id_axis_hash,
                "route_count": source_receipt.route_count,
                "removed_count": removed_count,
                "removed_route_index_hash": removed_hash,
            },
            prefix=_PREFIX + "physical-action-coordinate-",
        )
        subsets.append(
            V1SignedWorkVariableShellActionSubsetIdentity(
                shell_size=shell,
                shell_offsets=offsets,
                removed_count=removed_count,
                removed_route_index_hash=removed_hash,
                physical_action_coordinate_hash=coordinate_hash,
                streamed_hash_custody_family_hash=family.receipt_hash,
                parent_mask_hash=source_receipt.current_mask_hash,
                _issuer_token=_ACTION_COORDINATE_CUSTODY_TOKEN,
            )
        )
    return V1SignedWorkVariableShellActionCoordinateCustodyIssuance(
        family_receipt=family,
        subset_identities=tuple(subsets),
        _issuer_token=_ACTION_COORDINATE_CUSTODY_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellResourceAlias:
    """One frozen Phase-A coordinate; Phase B may not optimize it again."""

    branch: V1SignedWorkVariableShellQBranch
    resource_alias_id: str
    phase_a_resource_cell_id: str
    phase_a_invariance_receipt_hash: str
    source_resource_receipt_hash: str
    resource_coordinate_hash: str
    source_kkt_branch_id: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    registered_shell_resource_floor: float
    fixed_outside_removed_count: int
    shell_prefix_count: int
    total_removed_count: int
    phase_a_coordinate_frozen: bool = True
    phase_a_invariance_authority_claimed_here: bool = False
    schema: str = ACTION_RESOURCE_ALIAS_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("resource alias branch must be exact")
        for name in (
            "resource_alias_id",
            "phase_a_resource_cell_id",
            "phase_a_invariance_receipt_hash",
            "source_resource_receipt_hash",
            "resource_coordinate_hash",
            "source_kkt_branch_id",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if not _source_kkt_coordinate_supports_branch(
            self.source_kkt_branch_id,
            self.branch,
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
                "resource alias branch is not contained in its source KKT coordinate",
            )
        object.__setattr__(
            self,
            "registered_shell_resource_floor",
            _finite_nonnegative(
                self.registered_shell_resource_floor,
                name="registered shell resource floor",
            ),
        )
        fixed = _nonnegative_int(
            self.fixed_outside_removed_count,
            name="fixed outside removed count",
        )
        prefix = _nonnegative_int(self.shell_prefix_count, name="shell prefix count")
        total = _nonnegative_int(self.total_removed_count, name="total removed count")
        if total != fixed + prefix:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "resource alias total is not fixed-outside plus shell endpoint",
            )
        object.__setattr__(self, "fixed_outside_removed_count", fixed)
        object.__setattr__(self, "shell_prefix_count", prefix)
        object.__setattr__(self, "total_removed_count", total)
        if (
            self.phase_a_coordinate_frozen is not True
            or self.phase_a_invariance_authority_claimed_here is not False
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "resource alias must bind, but cannot issue, Phase-A invariance",
            )
        if self.schema != ACTION_RESOURCE_ALIAS_SCHEMA:
            raise ValueError("unsupported Phase-A resource alias schema")

    @property
    def alias_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "phase-a-resource-alias-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellClosedSubsetActionCell:
    shell_size: int
    selected_shell_offsets: tuple[int, ...]
    cell_ordinal: int
    cover_rule: str = ACTION_CELL_COVER_RULE
    schema: str = ACTION_CLOSED_SUBSET_CELL_SCHEMA

    def __post_init__(self) -> None:
        shell = _positive_int(self.shell_size, name="shell size")
        if shell > _MAXIMUM_SHELL_SIZE:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.SHELL_TOO_LARGE,
                "closed subset cell exceeds B=8",
            )
        object.__setattr__(self, "shell_size", shell)
        object.__setattr__(
            self,
            "selected_shell_offsets",
            _offset_axis(
                self.selected_shell_offsets,
                shell_size=shell,
                name="selected shell offsets",
            ),
        )
        object.__setattr__(
            self,
            "cell_ordinal",
            _nonnegative_int(self.cell_ordinal, name="cell ordinal"),
        )
        if (
            self.cover_rule != ACTION_CELL_COVER_RULE
            or self.schema != ACTION_CLOSED_SUBSET_CELL_SCHEMA
        ):
            raise ValueError("unsupported closed subset-cell semantics")

    @property
    def shell_prefix_count(self) -> int:
        return len(self.selected_shell_offsets)

    @property
    def weak_cross_inequalities(self) -> tuple[tuple[int, int], ...]:
        """Pairs ``(i,j)`` encoding ``z_i >= z_j``."""

        selected = set(self.selected_shell_offsets)
        return tuple(
            (left, right)
            for left in self.selected_shell_offsets
            for right in range(self.shell_size)
            if right not in selected
        )

    def contains(self, shell_gates: Sequence[Real]) -> bool:
        gates = _validated_gates(shell_gates, self.shell_size)
        return all(
            gates[left] >= gates[right] for left, right in self.weak_cross_inequalities
        )

    @property
    def cell_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "closed-subset-cell-")


def enumerate_v1_signed_work_variable_shell_closed_subset_action_cells(
    shell_size: int,
    shell_prefix_count: int,
) -> tuple[V1SignedWorkVariableShellClosedSubsetActionCell, ...]:
    """Enumerate the canonical ``binom(B,k)`` physical-action solve axis."""

    shell = _positive_int(shell_size, name="shell size")
    if shell > _MAXIMUM_SHELL_SIZE:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.SHELL_TOO_LARGE,
            "closed subset-cell enumeration is bounded by B=8",
        )
    cardinality = _nonnegative_int(shell_prefix_count, name="shell prefix count")
    if cardinality > shell:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            "shell endpoint cardinality exceeds the shell",
        )
    cells = tuple(
        V1SignedWorkVariableShellClosedSubsetActionCell(
            shell_size=shell,
            selected_shell_offsets=tuple(subset),
            cell_ordinal=ordinal,
        )
        for ordinal, subset in enumerate(combinations(range(shell), cardinality))
    )
    if len(cells) != math.comb(shell, cardinality):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_ACTION_CELL_AXIS,
            "closed subset-cell enumeration is incomplete",
        )
    return cells


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellClosedOrderCell:
    """Non-authoritative ``B!`` reference chamber, never a production cell."""

    shell_size: int
    cell_ordinal: int
    permutation: tuple[int, ...]
    production_authority: bool = False
    reference_rule: str = ACTION_REFERENCE_PERMUTATION_RULE
    schema: str = ACTION_REFERENCE_ORDER_CELL_SCHEMA

    def __post_init__(self) -> None:
        shell = _positive_int(self.shell_size, name="shell size")
        if shell > _MAXIMUM_SHELL_SIZE:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.SHELL_TOO_LARGE,
                "reference order cell exceeds B=8",
            )
        object.__setattr__(self, "shell_size", shell)
        object.__setattr__(
            self,
            "cell_ordinal",
            _nonnegative_int(self.cell_ordinal, name="cell ordinal"),
        )
        if type(self.permutation) is not tuple:
            raise TypeError("reference cell permutation must be immutable")
        axis = tuple(
            _nonnegative_int(value, name="permutation item")
            for value in self.permutation
        )
        if len(axis) != shell or set(axis) != set(range(shell)):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "reference order cell must contain one shell permutation",
            )
        object.__setattr__(self, "permutation", axis)
        if (
            self.production_authority is not False
            or self.reference_rule != ACTION_REFERENCE_PERMUTATION_RULE
            or self.schema != ACTION_REFERENCE_ORDER_CELL_SCHEMA
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "permutation chamber reference acquired production authority",
            )

    @property
    def adjacent_weak_inequalities(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (self.permutation[i], self.permutation[i + 1])
            for i in range(self.shell_size - 1)
        )

    def contains(self, shell_gates: Sequence[Real]) -> bool:
        gates = _validated_gates(shell_gates, self.shell_size)
        return all(
            gates[left] >= gates[right]
            for left, right in self.adjacent_weak_inequalities
        )


def enumerate_v1_signed_work_variable_shell_closed_order_cells_reference_only(
    shell_size: int,
) -> tuple[V1SignedWorkVariableShellClosedOrderCell, ...]:
    shell = _positive_int(shell_size, name="shell size")
    if shell > _MAXIMUM_SHELL_SIZE:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.SHELL_TOO_LARGE,
            "reference order-cell enumeration is bounded by B=8",
        )
    return tuple(
        V1SignedWorkVariableShellClosedOrderCell(
            shell_size=shell,
            cell_ordinal=ordinal,
            permutation=tuple(order),
        )
        for ordinal, order in enumerate(permutations(range(shell)))
    )


# Compatibility spelling is intentionally explicit about reference-only use.
enumerate_v1_signed_work_variable_shell_closed_order_cells = (
    enumerate_v1_signed_work_variable_shell_closed_order_cells_reference_only
)


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellBranchActionCellRegistration:
    branch: V1SignedWorkVariableShellQBranch
    source_receipt_hash: str
    source_replay_hash: str
    source_kkt_branch_id: str
    kkt_branch_source_receipt_hash: str
    source_resource_receipt_hash: str
    shell_freeze_receipt_hash: str
    streamed_hash_custody_family_hash: str
    parent_mask_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    shell_size: int
    subset_identity_count: int
    subset_identity_family_hash: str
    resource_alias_count: int
    resource_alias_axis_hash: str
    distinct_registered_cardinality_count: int
    action_cell_count: int
    action_cell_axis_hash: str
    cross_order_inequality_count: int
    complete_closed_subset_coverage: bool = True
    exact_ties_multicovered: bool = True
    complete_subset_identity_axis: bool = True
    one_cell_per_physical_action_per_alias: bool = True
    factorial_permutation_axis_used_in_production: bool = False
    source_handle_required_for_solver: bool = True
    phase_a_coordinate_reopened_by_phase_b: bool = False
    global_solver_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    cover_rule: str = ACTION_CELL_COVER_RULE
    action_image_rule: str = ACTION_CELL_IMAGE_RULE
    lexicographic_rule: str = ACTION_LEXICOGRAPHIC_RULE
    authority_scope: str = ACTION_SEPARATION_AUTHORITY_SCOPE
    schema: str = ACTION_BRANCH_REGISTRATION_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _REGISTRATION_TOKEN:
            raise TypeError("branch action-cell registrations are issuer-only")
        if type(self.branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("branch registration requires an exact Q branch")
        for name in (
            "source_receipt_hash",
            "source_replay_hash",
            "source_kkt_branch_id",
            "kkt_branch_source_receipt_hash",
            "source_resource_receipt_hash",
            "shell_freeze_receipt_hash",
            "streamed_hash_custody_family_hash",
            "parent_mask_hash",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "subset_identity_family_hash",
            "resource_alias_axis_hash",
            "action_cell_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if not _source_kkt_coordinate_supports_branch(
            self.source_kkt_branch_id,
            self.branch,
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
                "branch registration is not contained in its source KKT coordinate",
            )
        shell = _positive_int(self.shell_size, name="shell size")
        if shell > _MAXIMUM_SHELL_SIZE:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.SHELL_TOO_LARGE,
                "branch action-cell registration exceeds B=8",
            )
        if (
            type(self.shell_indices) is not tuple
            or type(self.shell_route_ids) is not tuple
        ):
            raise TypeError("branch shell axes must be immutable tuples")
        if (
            len(self.shell_indices) != shell
            or len(self.shell_route_ids) != shell
            or len(set(self.shell_indices)) != shell
            or len(set(self.shell_route_ids)) != shell
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "branch shell axes are incomplete or duplicated",
            )
        object.__setattr__(self, "shell_size", shell)
        expected_subsets = 1 << shell
        subsets = _positive_int(
            self.subset_identity_count, name="subset identity count"
        )
        aliases = _positive_int(self.resource_alias_count, name="resource alias count")
        cardinalities = _positive_int(
            self.distinct_registered_cardinality_count,
            name="distinct registered cardinality count",
        )
        cells = _positive_int(self.action_cell_count, name="action cell count")
        inequalities = _nonnegative_int(
            self.cross_order_inequality_count,
            name="cross-order inequality count",
        )
        if (
            subsets != expected_subsets
            or aliases != cardinalities
            or cells > expected_subsets
            or cells < aliases
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_ACTION_CELL_AXIS,
                "branch subset/action counts do not close on the powerset quotient",
            )
        object.__setattr__(self, "subset_identity_count", subsets)
        object.__setattr__(self, "resource_alias_count", aliases)
        object.__setattr__(self, "distinct_registered_cardinality_count", cardinalities)
        object.__setattr__(self, "action_cell_count", cells)
        object.__setattr__(self, "cross_order_inequality_count", inequalities)
        required_true = (
            self.complete_closed_subset_coverage,
            self.exact_ties_multicovered,
            self.complete_subset_identity_axis,
            self.one_cell_per_physical_action_per_alias,
            self.source_handle_required_for_solver,
        )
        if any(type(value) is not bool for value in required_true) or any(
            value is not True for value in required_true
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "branch registration lost a mandatory coverage/source condition",
            )
        _closed_authority(
            (
                self.factorial_permutation_axis_used_in_production,
                self.phase_a_coordinate_reopened_by_phase_b,
                self.global_solver_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="branch action-cell registration",
        )
        if (
            self.cover_rule != ACTION_CELL_COVER_RULE
            or self.action_image_rule != ACTION_CELL_IMAGE_RULE
            or self.lexicographic_rule != ACTION_LEXICOGRAPHIC_RULE
            or self.authority_scope != ACTION_SEPARATION_AUTHORITY_SCOPE
            or self.schema != ACTION_BRANCH_REGISTRATION_SCHEMA
        ):
            raise ValueError("unsupported branch action-cell semantics")

    @property
    def registration_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "branch-registration-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellActionCellIdentity:
    registry_receipt_hash: str
    branch_registration_hash: str
    source_receipt_hash: str
    branch: V1SignedWorkVariableShellQBranch
    resource_alias_id: str
    phase_a_resource_cell_id: str
    phase_a_invariance_receipt_hash: str
    action_cell_ordinal: int
    closed_subset_cell_hash: str
    shell_prefix_count: int
    selected_shell_offsets: tuple[int, ...]
    removed_route_index_hash: str
    physical_action_coordinate_hash: str
    streamed_hash_custody_family_hash: str
    parent_mask_hash: str
    phase_b_objective: str = "minimize_native_fiber_F"
    schema: str = ACTION_CELL_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "registry_receipt_hash",
            "branch_registration_hash",
            "source_receipt_hash",
            "resource_alias_id",
            "phase_a_resource_cell_id",
            "phase_a_invariance_receipt_hash",
            "closed_subset_cell_hash",
            "removed_route_index_hash",
            "physical_action_coordinate_hash",
            "streamed_hash_custody_family_hash",
            "parent_mask_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("action-cell identity branch must be exact")
        ordinal = _nonnegative_int(self.action_cell_ordinal, name="action cell ordinal")
        prefix = _nonnegative_int(self.shell_prefix_count, name="shell prefix count")
        offsets = tuple(self.selected_shell_offsets)
        if (
            offsets != tuple(sorted(set(offsets)))
            or len(offsets) != prefix
            or any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in offsets
            )
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "action-cell selected subset does not match its endpoint cardinality",
            )
        object.__setattr__(self, "action_cell_ordinal", ordinal)
        object.__setattr__(self, "shell_prefix_count", prefix)
        object.__setattr__(
            self, "selected_shell_offsets", tuple(int(value) for value in offsets)
        )
        if (
            self.phase_b_objective != "minimize_native_fiber_F"
            or self.schema != ACTION_CELL_IDENTITY_SCHEMA
        ):
            raise ValueError("unsupported action-cell identity semantics")

    @property
    def action_cell_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "action-cell-")

    @property
    def byte_physical_action_hash(self) -> str:
        """Legacy wire alias; no packed child bytes are claimed here."""

        return self.physical_action_coordinate_hash


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellActionCellRegistryReceipt:
    design_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    shell_freeze_receipt_hash: str
    shell_index_axis_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    resource_axis_hash: str
    alpha_lineage_hash: str
    branch_registration_hashes: tuple[str, ...]
    branch_axis: tuple[V1SignedWorkVariableShellQBranch, ...]
    resource_alias_axis: tuple[tuple[str, str, int, str, str, str, str, str, str], ...]
    source_receipt_hashes: tuple[str, ...]
    source_replay_hashes: tuple[str, ...]
    total_action_cell_count: int
    maximum_distinct_action_cells_per_branch: int
    action_cell_axis_hash: str
    complete_branch_axis: bool
    complete_action_cell_coverage: bool = True
    exact_ties_registered_in_every_compatible_cell: bool = True
    factorial_permutation_axis_used_in_production: bool = False
    phase_a_resource_axis_separate_from_phase_b_objective: bool = True
    live_source_required_for_solver_or_certificate: bool = True
    array_free_receipt: bool = True
    global_restricted_solver_landed: bool = False
    statistical_uniform_confidence_authority: bool = False
    global_cell_optimality_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    cover_rule: str = ACTION_CELL_COVER_RULE
    lexicographic_rule: str = ACTION_LEXICOGRAPHIC_RULE
    restricted_problem_rule: str = ACTION_RESTRICTED_PROBLEM_RULE
    separation_rule: str = ACTION_SEPARATION_RULE
    authority_scope: str = ACTION_SEPARATION_AUTHORITY_SCOPE
    schema: str = ACTION_CELL_REGISTRY_RECEIPT_SCHEMA
    _issuer_token: InitVar[object] = None

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _RECEIPT_TOKEN:
            raise TypeError("action-cell registry receipts are issuer-only")
        for name in (
            "design_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "shell_freeze_receipt_hash",
            "shell_index_axis_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "resource_axis_hash",
            "alpha_lineage_hash",
            "action_cell_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        branches = tuple(self.branch_axis)
        registrations = tuple(self.branch_registration_hashes)
        source_hashes = tuple(self.source_receipt_hashes)
        replay_hashes = tuple(self.source_replay_hashes)
        aliases = tuple(tuple(value) for value in self.resource_alias_axis)
        if (
            not branches
            or any(
                type(value) is not V1SignedWorkVariableShellQBranch
                for value in branches
            )
            or len(set(branches)) != len(branches)
            or not (
                len(branches)
                == len(registrations)
                == len(source_hashes)
                == len(replay_hashes)
            )
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_BRANCH_AXIS,
                "registry branch/source axes do not close",
            )
        if any(
            type(value) is not tuple
            or len(value) != 9
            or type(value[0]) is not str
            or type(value[1]) is not str
            or isinstance(value[2], bool)
            or not isinstance(value[2], Integral)
            or type(value[3]) is not str
            or any(type(item) is not str for item in value[4:])
            for value in aliases
        ):
            raise TypeError("resource alias receipt axis is malformed")
        for axis_name, axis in (
            ("branch registration hash", registrations),
            ("source receipt hash", source_hashes),
            ("source replay hash", replay_hashes),
        ):
            tuple(_identity(value, name=axis_name) for value in axis)
        total = _positive_int(
            self.total_action_cell_count, name="total action-cell count"
        )
        maximum = _positive_int(
            self.maximum_distinct_action_cells_per_branch,
            name="maximum distinct action cells per branch",
        )
        if maximum > _MAXIMUM_ACTION_CELLS or total < maximum:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_ACTION_CELL_AXIS,
                "registry cell counts exceed the B=8 powerset quotient",
            )
        object.__setattr__(self, "branch_axis", branches)
        object.__setattr__(self, "branch_registration_hashes", registrations)
        object.__setattr__(self, "source_receipt_hashes", source_hashes)
        object.__setattr__(self, "source_replay_hashes", replay_hashes)
        object.__setattr__(self, "resource_alias_axis", aliases)
        object.__setattr__(self, "total_action_cell_count", total)
        object.__setattr__(self, "maximum_distinct_action_cells_per_branch", maximum)
        required_true = (
            self.complete_action_cell_coverage,
            self.exact_ties_registered_in_every_compatible_cell,
            self.phase_a_resource_axis_separate_from_phase_b_objective,
            self.live_source_required_for_solver_or_certificate,
            self.array_free_receipt,
        )
        if any(type(value) is not bool for value in required_true) or any(
            value is not True for value in required_true
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "registry lost a required coverage/custody property",
            )
        if type(self.complete_branch_axis) is not bool:
            raise TypeError("complete branch marker must be exact bool")
        _closed_authority(
            (
                self.factorial_permutation_axis_used_in_production,
                self.global_restricted_solver_landed,
                self.statistical_uniform_confidence_authority,
                self.global_cell_optimality_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="action-cell registry receipt",
        )
        if (
            self.cover_rule != ACTION_CELL_COVER_RULE
            or self.lexicographic_rule != ACTION_LEXICOGRAPHIC_RULE
            or self.restricted_problem_rule != ACTION_RESTRICTED_PROBLEM_RULE
            or self.separation_rule != ACTION_SEPARATION_RULE
            or self.authority_scope != ACTION_SEPARATION_AUTHORITY_SCOPE
            or self.schema != ACTION_CELL_REGISTRY_RECEIPT_SCHEMA
        ):
            raise ValueError("unsupported action-cell registry receipt semantics")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "registry-receipt-",
        )


class V1SignedWorkVariableShellBranchActionCellSource:
    """One branch's live source, frozen Phase-A aliases, and subset cells."""

    __slots__ = (
        "branch",
        "source_handle",
        "resource_aliases",
        "subset_identities",
        "registration",
        "_alias_by_id",
        "_subset_by_offsets",
        "_cells_by_alias",
        "_source_receipt_hash",
        "_live_identity_hash",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        branch: V1SignedWorkVariableShellQBranch,
        source_handle: V1SignedWorkVariableShellCompactSourceHandle,
        resource_aliases: Sequence[V1SignedWorkVariableShellResourceAlias],
        action_coordinate_custody: V1SignedWorkVariableShellActionCoordinateCustodyIssuance,
        design: V1SignedWorkVariableShellActionSeparationCUDAExecutionDesign,
    ) -> None:
        if type(branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("branch source requires an exact Q branch")
        if not _is_compact_source_handle(source_handle):
            raise TypeError(
                "branch source requires an exact one-shot or segmented source handle"
            )
        if (
            type(design)
            is not V1SignedWorkVariableShellActionSeparationCUDAExecutionDesign
        ):
            raise TypeError("branch source requires the exact execution design")
        receipt = source_handle.receipt
        if not _is_compact_source_receipt(receipt):
            raise TypeError("branch source returned another receipt type")
        shell = len(receipt.shell_indices)
        if shell > design.maximum_shell_size:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.SHELL_TOO_LARGE,
                "live branch shell exceeds its frozen design",
            )
        if not _source_kkt_coordinate_supports_branch(
            receipt.kkt_branch_id,
            branch,
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
                "live source KKT coordinate does not contain the action-separation branch",
            )
        if type(action_coordinate_custody) is not (
            V1SignedWorkVariableShellActionCoordinateCustodyIssuance
        ):
            raise TypeError("branch source requires exact action-coordinate custody")
        custody_family = action_coordinate_custody.family_receipt
        if (
            custody_family.source_receipt_hash != receipt.receipt_hash
            or custody_family.source_replay_hash != receipt.source_replay_hash
            or custody_family.parent_mask_hash != receipt.current_mask_hash
            or custody_family.route_axis_hash != receipt.route_axis_hash
            or custody_family.route_id_axis_hash != receipt.route_id_axis_hash
            or custody_family.route_count != receipt.route_count
            or custody_family.shell_freeze_receipt_hash
            != receipt.shell_freeze_receipt_hash
            or custody_family.shell_indices != receipt.shell_indices
            or custody_family.shell_route_ids != receipt.shell_route_ids
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
                "action-coordinate custody is not bound to the live variable-shell source",
            )
        probe_purpose = content_hash(
            {
                "source_receipt_hash": receipt.receipt_hash,
                "branch": branch.value,
                "operation": "action_cell_registry_device_probe",
            },
            prefix=_PREFIX + "device-probe-",
        )
        device_type = str(
            _compact_source_device(
                source_handle,
                purpose_hash=probe_purpose,
            ).type
        )
        if (
            design.execution_mode
            is V1SignedWorkVariableShellActionSeparationMode.CUDA_PRODUCTION
            and device_type != "cuda"
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.CUDA_REQUIRED,
                "production action separation requires a CUDA-resident source",
            )
        aliases = tuple(resource_aliases)
        prefix_axis = tuple(value.shell_prefix_count for value in aliases)
        if (
            not aliases
            or len(aliases) > design.maximum_resource_alias_count
            or any(
                type(value) is not V1SignedWorkVariableShellResourceAlias
                for value in aliases
            )
            or any(value.branch is not branch for value in aliases)
            or any(value.shell_prefix_count > shell for value in aliases)
            or any(
                value.source_kkt_branch_id != receipt.kkt_branch_id for value in aliases
            )
            or any(
                value.source_resource_receipt_hash
                != receipt.resource_source_receipt_hash
                for value in aliases
            )
            or any(
                value.same_capture_vjp_source_hash
                != receipt.same_capture_vjp_source_hash
                for value in aliases
            )
            or any(
                value.corrected_evidence_stream_source_hash
                != receipt.corrected_evidence_stream_source_hash
                for value in aliases
            )
            or any(
                value.literal_j_stream_source_hash
                != receipt.literal_j_stream_source_hash
                for value in aliases
            )
            or len({value.resource_alias_id for value in aliases}) != len(aliases)
            or len(set(prefix_axis)) != len(prefix_axis)
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.DUPLICATE_ALIAS,
                "aliases are empty, duplicated, cross-branch, or repeat a cardinality",
            )
        subsets = tuple(action_coordinate_custody.subset_identities)
        expected_offsets = _complete_subset_axis(shell)
        observed_offsets = (
            tuple(
                sorted(
                    (value.shell_offsets for value in subsets),
                    key=lambda value: (len(value), value),
                )
            )
            if all(
                type(value) is V1SignedWorkVariableShellActionSubsetIdentity
                for value in subsets
            )
            else ()
        )
        if (
            len(subsets) != 1 << shell
            or observed_offsets != expected_offsets
            or any(value.shell_size != shell for value in subsets)
            or any(
                value.parent_mask_hash != receipt.current_mask_hash for value in subsets
            )
            or any(
                value.streamed_hash_custody_family_hash != custody_family.receipt_hash
                for value in subsets
            )
            or len({value.shell_offsets for value in subsets}) != len(subsets)
            or len({value.removed_route_index_hash for value in subsets})
            != len(subsets)
            or len({value.physical_action_coordinate_hash for value in subsets})
            != len(subsets)
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SUBSET_AXIS,
                "subset identities must cover the exact frozen-shell powerset and hash-custody family",
            )
        cells_by_alias = {
            alias.resource_alias_id: (
                enumerate_v1_signed_work_variable_shell_closed_subset_action_cells(
                    shell,
                    alias.shell_prefix_count,
                )
            )
            for alias in aliases
        }
        cell_count = sum(len(value) for value in cells_by_alias.values())
        if cell_count > design.maximum_action_cell_count or cell_count > 1 << shell:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_ACTION_CELL_AXIS,
                "registered cardinalities exceed the shell powerset cell ceiling",
            )
        subset_hash = content_hash(
            tuple(value.subset_hash for value in subsets),
            prefix=_PREFIX + "subset-family-",
        )
        alias_hash = content_hash(
            tuple(value.alias_hash for value in aliases),
            prefix=_PREFIX + "phase-a-alias-axis-",
        )
        cell_axis_hash = content_hash(
            tuple(
                (
                    alias.resource_alias_id,
                    tuple(
                        cell.cell_hash
                        for cell in cells_by_alias[alias.resource_alias_id]
                    ),
                )
                for alias in aliases
            ),
            prefix=_PREFIX + "branch-action-cell-axis-",
        )
        inequality_count = sum(
            len(cell.weak_cross_inequalities)
            for cells in cells_by_alias.values()
            for cell in cells
        )
        registration = V1SignedWorkVariableShellBranchActionCellRegistration(
            branch=branch,
            source_receipt_hash=receipt.receipt_hash,
            source_replay_hash=receipt.source_replay_hash,
            source_kkt_branch_id=receipt.kkt_branch_id,
            kkt_branch_source_receipt_hash=receipt.kkt_branch_source_receipt_hash,
            source_resource_receipt_hash=receipt.resource_source_receipt_hash,
            shell_freeze_receipt_hash=receipt.shell_freeze_receipt_hash,
            streamed_hash_custody_family_hash=custody_family.receipt_hash,
            parent_mask_hash=receipt.current_mask_hash,
            same_capture_vjp_source_hash=receipt.same_capture_vjp_source_hash,
            corrected_evidence_stream_source_hash=(
                receipt.corrected_evidence_stream_source_hash
            ),
            literal_j_stream_source_hash=receipt.literal_j_stream_source_hash,
            shell_indices=receipt.shell_indices,
            shell_route_ids=receipt.shell_route_ids,
            shell_size=shell,
            subset_identity_count=len(subsets),
            subset_identity_family_hash=subset_hash,
            resource_alias_count=len(aliases),
            resource_alias_axis_hash=alias_hash,
            distinct_registered_cardinality_count=len(set(prefix_axis)),
            action_cell_count=cell_count,
            action_cell_axis_hash=cell_axis_hash,
            cross_order_inequality_count=inequality_count,
            _issuer_token=_REGISTRATION_TOKEN,
        )
        self.branch = branch
        self.source_handle = source_handle
        self.resource_aliases = aliases
        self.subset_identities = subsets
        self.registration = registration
        self._alias_by_id = {value.resource_alias_id: value for value in aliases}
        self._subset_by_offsets = {value.shell_offsets: value for value in subsets}
        self._cells_by_alias = cells_by_alias
        self._source_receipt_hash = receipt.receipt_hash
        self._live_identity_hash = self._current_live_identity_hash()

    def _current_live_identity_hash(self) -> str:
        if (
            type(self.branch) is not V1SignedWorkVariableShellQBranch
            or type(self.resource_aliases) is not tuple
            or type(self.subset_identities) is not tuple
            or type(self.registration)
            is not V1SignedWorkVariableShellBranchActionCellRegistration
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
                "branch source live identity changed type",
            )
        return content_hash(
            {
                "branch": self.branch.value,
                "registration_hash": self.registration.registration_hash,
                "resource_alias_hashes": tuple(
                    value.alias_hash for value in self.resource_aliases
                ),
                "subset_hashes": tuple(
                    value.subset_hash for value in self.subset_identities
                ),
                "cell_axis": tuple(
                    (
                        alias_id,
                        tuple(value.cell_hash for value in cells),
                    )
                    for alias_id, cells in sorted(self._cells_by_alias.items())
                ),
                "source_receipt_hash": self._source_receipt_hash,
            },
            prefix=_PREFIX + "live-branch-source-",
        )

    def assert_live_source(self) -> V1SignedWorkVariableShellCompactSourceHandle:
        try:
            receipt = self.source_handle.receipt
        except Exception as error:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.EVICTED_SOURCE,
                f"branch source is unavailable: {type(error).__name__}",
            )
        if (
            receipt.receipt_hash != self._source_receipt_hash
            or self._current_live_identity_hash() != self._live_identity_hash
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
                "branch source receipt or registered live identity changed",
            )
        return self.source_handle

    def resource_alias(
        self,
        resource_alias_id: str,
    ) -> V1SignedWorkVariableShellResourceAlias:
        alias_id = _identity(resource_alias_id, name="resource alias id")
        alias = self._alias_by_id.get(alias_id)
        if alias is None:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "resource alias is not registered on this branch",
            )
        return alias

    def closed_cells(
        self,
        resource_alias_id: str,
    ) -> tuple[V1SignedWorkVariableShellClosedSubsetActionCell, ...]:
        alias = self.resource_alias(resource_alias_id)
        return self._cells_by_alias[alias.resource_alias_id]

    def closed_cells_containing(
        self,
        *,
        resource_alias_id: str,
        shell_gates: Sequence[Real],
    ) -> tuple[V1SignedWorkVariableShellClosedSubsetActionCell, ...]:
        self.assert_live_source()
        result = tuple(
            cell
            for cell in self.closed_cells(resource_alias_id)
            if cell.contains(shell_gates)
        )
        if not result:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_ACTION_CELL_AXIS,
                "closed subset cells failed to cover a valid shell point",
            )
        return result

    def action_cell(
        self,
        *,
        registry_receipt_hash: str,
        resource_alias_id: str,
        action_cell_ordinal: int,
    ) -> V1SignedWorkVariableShellActionCellIdentity:
        self.assert_live_source()
        alias = self.resource_alias(resource_alias_id)
        ordinal = _nonnegative_int(action_cell_ordinal, name="action cell ordinal")
        cells = self._cells_by_alias[alias.resource_alias_id]
        if ordinal >= len(cells):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "action-cell ordinal is outside the alias axis",
            )
        cell = cells[ordinal]
        subset = self._subset_by_offsets[cell.selected_shell_offsets]
        return V1SignedWorkVariableShellActionCellIdentity(
            registry_receipt_hash=registry_receipt_hash,
            branch_registration_hash=self.registration.registration_hash,
            source_receipt_hash=self.registration.source_receipt_hash,
            branch=self.branch,
            resource_alias_id=alias.resource_alias_id,
            phase_a_resource_cell_id=alias.phase_a_resource_cell_id,
            phase_a_invariance_receipt_hash=alias.phase_a_invariance_receipt_hash,
            action_cell_ordinal=ordinal,
            closed_subset_cell_hash=cell.cell_hash,
            shell_prefix_count=alias.shell_prefix_count,
            selected_shell_offsets=cell.selected_shell_offsets,
            removed_route_index_hash=subset.removed_route_index_hash,
            physical_action_coordinate_hash=(subset.physical_action_coordinate_hash),
            streamed_hash_custody_family_hash=(
                subset.streamed_hash_custody_family_hash
            ),
            parent_mask_hash=subset.parent_mask_hash,
        )

    def borrow_solver_tensors(
        self,
        *,
        request_hash: str,
    ) -> V1SignedWorkVariableShellCompactSourceLease:
        self.assert_live_source()
        purpose = content_hash(
            {
                "branch_registration_hash": self.registration.registration_hash,
                "request_hash": _identity(request_hash, name="restricted request hash"),
                "operation": "phase_b_pdhg_compact_source_borrow",
            },
            prefix=_PREFIX + "solver-tensor-borrow-",
        )
        if type(self.source_handle) is V1SignedWorkVariableShellSourceDeviceHandle:
            return self.source_handle.borrow_compact_tensors(
                borrow_purpose_hash=purpose
            )
        if (
            type(self.source_handle)
            is V1SignedWorkVariableShellSourceAppendSegmentedHandle
        ):
            return self.source_handle.borrow_compact_segments(
                borrow_purpose_hash=purpose
            )
        if type(self.source_handle) is V1SampledFullPopulationCompactSourceHandle:
            return self.source_handle.borrow_components(borrow_purpose_hash=purpose)
        raise TypeError("branch source handle type changed after registration")

    def __reduce__(self) -> NoReturn:
        raise TypeError("live branch action-cell sources are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live branch action-cell sources are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("live branch action-cell sources cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("live branch action-cell sources cannot be copied")


class V1SignedWorkVariableShellActionCellRegistry:
    """Live two-branch registry retained through the future theorem issuer."""

    __slots__ = ("design", "branch_sources", "receipt", "_branch_by_id")
    __hash__ = None

    def __init__(
        self,
        *,
        design: V1SignedWorkVariableShellActionSeparationCUDAExecutionDesign,
        branch_sources: tuple[V1SignedWorkVariableShellBranchActionCellSource, ...],
        receipt: V1SignedWorkVariableShellActionCellRegistryReceipt,
    ) -> None:
        if (
            type(design)
            is not V1SignedWorkVariableShellActionSeparationCUDAExecutionDesign
        ):
            raise TypeError("registry requires the exact execution design")
        if type(receipt) is not V1SignedWorkVariableShellActionCellRegistryReceipt:
            raise TypeError("registry requires its exact durable receipt")
        sources = tuple(branch_sources)
        if not sources or any(
            type(value) is not V1SignedWorkVariableShellBranchActionCellSource
            for value in sources
        ):
            raise TypeError("registry requires exact live branch sources")
        if tuple(value.branch for value in sources) != receipt.branch_axis:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "live branch order differs from the registry receipt",
            )
        self.design = design
        self.branch_sources = sources
        self.receipt = receipt
        self._branch_by_id = {value.branch: value for value in sources}
        self.assert_live_sources()

    def assert_live_sources(
        self,
    ) -> tuple[V1SignedWorkVariableShellCompactSourceHandle, ...]:
        handles = tuple(value.assert_live_source() for value in self.branch_sources)
        if (
            tuple(value.branch for value in self.branch_sources)
            != self.receipt.branch_axis
            or tuple(handle.receipt.receipt_hash for handle in handles)
            != self.receipt.source_receipt_hashes
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
                "live source family differs from its registry receipt",
            )
        return handles

    def branch_source(
        self,
        branch: V1SignedWorkVariableShellQBranch,
    ) -> V1SignedWorkVariableShellBranchActionCellSource:
        if type(branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("branch source lookup requires an exact branch")
        source = self._branch_by_id.get(branch)
        if source is None:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_BRANCH_AXIS,
                "requested branch is absent from the live registry",
            )
        source.assert_live_source()
        return source

    def live_source_handle(
        self,
        branch: V1SignedWorkVariableShellQBranch,
    ) -> V1SignedWorkVariableShellCompactSourceHandle:
        return self.branch_source(branch).assert_live_source()

    def action_cell(
        self,
        *,
        branch: V1SignedWorkVariableShellQBranch,
        resource_alias_id: str,
        action_cell_ordinal: int,
    ) -> V1SignedWorkVariableShellActionCellIdentity:
        return self.branch_source(branch).action_cell(
            registry_receipt_hash=self.receipt.receipt_hash,
            resource_alias_id=resource_alias_id,
            action_cell_ordinal=action_cell_ordinal,
        )

    def iter_action_cells(
        self,
    ) -> Iterator[V1SignedWorkVariableShellActionCellIdentity]:
        self.assert_live_sources()
        for source in self.branch_sources:
            for alias in source.resource_aliases:
                for cell in source.closed_cells(alias.resource_alias_id):
                    yield source.action_cell(
                        registry_receipt_hash=self.receipt.receipt_hash,
                        resource_alias_id=alias.resource_alias_id,
                        action_cell_ordinal=cell.cell_ordinal,
                    )

    def reachable_physical_action_coordinate_hashes(
        self,
        *,
        branch: V1SignedWorkVariableShellQBranch,
        resource_alias_id: str,
        shell_gates: Sequence[Real],
    ) -> tuple[str, ...]:
        source = self.branch_source(branch)
        cells = source.closed_cells_containing(
            resource_alias_id=resource_alias_id,
            shell_gates=shell_gates,
        )
        return tuple(
            sorted(
                {
                    source.action_cell(
                        registry_receipt_hash=self.receipt.receipt_hash,
                        resource_alias_id=resource_alias_id,
                        action_cell_ordinal=cell.cell_ordinal,
                    ).physical_action_coordinate_hash
                    for cell in cells
                }
            )
        )

    def reachable_physical_action_hashes(
        self,
        *,
        branch: V1SignedWorkVariableShellQBranch,
        resource_alias_id: str,
        shell_gates: Sequence[Real],
    ) -> tuple[str, ...]:
        """Legacy name for the exact hard-action coordinate query."""

        return self.reachable_physical_action_coordinate_hashes(
            branch=branch,
            resource_alias_id=resource_alias_id,
            shell_gates=shell_gates,
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("live action-cell registries are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live action-cell registries are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("live action-cell registries cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("live action-cell registries cannot be copied")


def build_v1_signed_work_variable_shell_action_cell_registry(
    *,
    branch_sources: Sequence[V1SignedWorkVariableShellBranchActionCellSource],
    design: V1SignedWorkVariableShellActionSeparationCUDAExecutionDesign,
) -> V1SignedWorkVariableShellActionCellRegistry:
    if type(design) is not V1SignedWorkVariableShellActionSeparationCUDAExecutionDesign:
        raise TypeError("registry factory requires the exact execution design")
    sources = tuple(branch_sources)
    if not sources or any(
        type(value) is not V1SignedWorkVariableShellBranchActionCellSource
        for value in sources
    ):
        raise TypeError("registry factory requires exact live branch sources")
    expected_branches = (
        V1SignedWorkVariableShellQBranch.SPECTRAL,
        V1SignedWorkVariableShellQBranch.DIAGONAL_L1,
    )
    branches = tuple(value.branch for value in sources)
    if len(set(branches)) != len(branches):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_BRANCH_AXIS,
            "branch source axis contains duplicates",
        )
    if (
        design.execution_mode
        is V1SignedWorkVariableShellActionSeparationMode.CUDA_PRODUCTION
        and branches != expected_branches
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_BRANCH_AXIS,
            "production registry requires spectral then diagonal-l1 branches",
        )
    receipts = tuple(source.assert_live_source().receipt for source in sources)
    first = receipts[0]
    common_names = (
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "registered_d_prefix_hash",
        "registered_d_w2_prefix_hash",
        "shell_freeze_receipt_hash",
        "shell_indices",
        "shell_route_ids",
        "native_fiber_axis_receipt_hash",
        "native_fiber_measure_hash",
        "resource_axis_hash",
        "same_capture_vjp_source_hash",
        "corrected_evidence_stream_source_hash",
        "literal_j_stream_source_hash",
        "alpha_lineage_hash",
        "context_axis_hash",
        "draw_axis_hash",
        "measurement_axis_hash",
        "capture_axis_hash",
        "source_packet_axis_hash",
    )
    if any(
        any(getattr(receipt, name) != getattr(first, name) for name in common_names)
        for receipt in receipts[1:]
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
            "branch sources do not share one chart/shell/D_W2/resource lineage",
        )
    registrations = tuple(source.registration for source in sources)
    aliases = tuple(
        (
            source.branch.value,
            alias.resource_alias_id,
            alias.shell_prefix_count,
            alias.phase_a_invariance_receipt_hash,
            alias.source_kkt_branch_id,
            alias.source_resource_receipt_hash,
            alias.same_capture_vjp_source_hash,
            alias.corrected_evidence_stream_source_hash,
            alias.literal_j_stream_source_hash,
        )
        for source in sources
        for alias in source.resource_aliases
    )
    total = sum(value.action_cell_count for value in registrations)
    maximum = max(value.action_cell_count for value in registrations)
    cell_axis_hash = content_hash(
        {
            "branch_registration_hashes": tuple(
                value.registration_hash for value in registrations
            ),
            "resource_alias_axis": aliases,
            "total_action_cell_count": total,
            "cover_rule": ACTION_CELL_COVER_RULE,
            "phase_rule": ACTION_LEXICOGRAPHIC_RULE,
        },
        prefix=_PREFIX + "complete-action-cell-axis-",
    )
    receipt = V1SignedWorkVariableShellActionCellRegistryReceipt(
        design_hash=design.design_hash,
        model_identity_hash=first.model_identity_hash,
        chart_identity_hash=first.chart_identity_hash,
        current_mask_hash=first.current_mask_hash,
        proposal_generation=first.proposal_generation,
        route_axis_hash=first.route_axis_hash,
        route_id_axis_hash=first.route_id_axis_hash,
        registered_d_prefix_hash=first.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=first.registered_d_w2_prefix_hash,
        shell_freeze_receipt_hash=first.shell_freeze_receipt_hash,
        shell_index_axis_hash=content_hash(
            {
                "shell_indices": first.shell_indices,
                "shell_route_ids": first.shell_route_ids,
            },
            prefix=_PREFIX + "shell-axis-",
        ),
        native_fiber_axis_receipt_hash=first.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=first.native_fiber_measure_hash,
        resource_axis_hash=first.resource_axis_hash,
        alpha_lineage_hash=first.alpha_lineage_hash,
        branch_registration_hashes=tuple(
            value.registration_hash for value in registrations
        ),
        branch_axis=branches,
        resource_alias_axis=aliases,
        source_receipt_hashes=tuple(value.receipt_hash for value in receipts),
        source_replay_hashes=tuple(value.source_replay_hash for value in receipts),
        total_action_cell_count=total,
        maximum_distinct_action_cells_per_branch=maximum,
        action_cell_axis_hash=cell_axis_hash,
        complete_branch_axis=branches == expected_branches,
        _issuer_token=_RECEIPT_TOKEN,
    )
    return V1SignedWorkVariableShellActionCellRegistry(
        design=design,
        branch_sources=sources,
        receipt=receipt,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSourceUncertaintyGroupReceipt:
    """Explicit census coverage for one source/branch/resource group."""

    registry_receipt_hash: str
    source_uncertainty_group_id: str
    source_receipt_hash: str
    source_replay_hash: str
    branch_registration_hash: str
    branch: V1SignedWorkVariableShellQBranch
    resource_alias_id: str
    phase_a_invariance_receipt_hash: str
    registered_d_w2_prefix_hash: str
    alpha_lineage_hash: str
    population_context_axis_hash: str
    context_count: int
    population_count: int
    coverage_basis_hash: str
    _issuer_token: InitVar[object] = None
    exact_registered_population_census: bool = True
    sampling_uncertainty_radius_zero_by_census: bool = True
    deterministic_branch_and_resource_source_bound: bool = True
    array_free_receipt: bool = True
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = ACTION_SOURCE_UNCERTAINTY_GROUP_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _UNCERTAINTY_GROUP_TOKEN:
            raise TypeError("source-uncertainty group receipts are issuer-only")
        for name in (
            "registry_receipt_hash",
            "source_uncertainty_group_id",
            "source_receipt_hash",
            "source_replay_hash",
            "branch_registration_hash",
            "resource_alias_id",
            "phase_a_invariance_receipt_hash",
            "registered_d_w2_prefix_hash",
            "alpha_lineage_hash",
            "population_context_axis_hash",
            "coverage_basis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("uncertainty group branch must be exact")
        context_count = _positive_int(self.context_count, name="context count")
        population_count = _positive_int(self.population_count, name="population count")
        if context_count != population_count:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "exact source-uncertainty coverage requires the full registered census",
            )
        object.__setattr__(self, "context_count", context_count)
        object.__setattr__(self, "population_count", population_count)
        required_true = (
            self.exact_registered_population_census,
            self.sampling_uncertainty_radius_zero_by_census,
            self.deterministic_branch_and_resource_source_bound,
            self.array_free_receipt,
        )
        if any(value is not True for value in required_true):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "source-uncertainty group coverage is incomplete",
            )
        _closed_authority(
            (
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="source-uncertainty group receipt",
        )
        if self.schema != ACTION_SOURCE_UNCERTAINTY_GROUP_SCHEMA:
            raise ValueError("unsupported source-uncertainty group schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "source-uncertainty-group-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSourceUncertaintyFamilyReceipt:
    """Complete explicit group family; a boolean alone can never replace it."""

    registry_receipt_hash: str
    registered_d_w2_prefix_hash: str
    alpha_lineage_hash: str
    group_axis: tuple[tuple[str, str, str, str, str], ...]
    group_receipt_hashes: tuple[str, ...]
    group_family_hash: str
    source_receipt_hashes: tuple[str, ...]
    branch_registration_hashes: tuple[str, ...]
    source_groups: tuple[
        V1SignedWorkVariableShellSourceUncertaintyGroupReceipt, ...
    ] = field(
        repr=False,
        compare=False,
    )
    _issuer_token: InitVar[object] = None
    complete_registered_branch_resource_source_product: bool = True
    every_group_has_explicit_coverage_receipt: bool = True
    exact_registered_population_census_family: bool = True
    array_free_receipt: bool = True
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = ACTION_SOURCE_UNCERTAINTY_FAMILY_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _UNCERTAINTY_FAMILY_TOKEN:
            raise TypeError("source-uncertainty family receipts are issuer-only")
        for name in (
            "registry_receipt_hash",
            "registered_d_w2_prefix_hash",
            "alpha_lineage_hash",
            "group_family_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        groups = tuple(self.source_groups)
        axis = tuple(tuple(value) for value in self.group_axis)
        hashes = tuple(self.group_receipt_hashes)
        source_hashes = tuple(self.source_receipt_hashes)
        registration_hashes = tuple(self.branch_registration_hashes)
        if (
            not groups
            or any(
                type(value)
                is not V1SignedWorkVariableShellSourceUncertaintyGroupReceipt
                for value in groups
            )
            or len(groups) != len(axis)
            or len(groups) != len(hashes)
            or tuple(value.receipt_hash for value in groups) != hashes
            or len(set(axis)) != len(axis)
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "source-uncertainty group receipt axis is incomplete",
            )
        observed_axis = tuple(
            (
                value.source_uncertainty_group_id,
                value.source_receipt_hash,
                value.branch.value,
                value.resource_alias_id,
                value.phase_a_invariance_receipt_hash,
            )
            for value in groups
        )
        if observed_axis != axis:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "source-uncertainty group identity differs from its family axis",
            )
        expected_family_hash = content_hash(
            {"group_axis": axis, "group_receipt_hashes": hashes},
            prefix=_PREFIX + "source-uncertainty-family-axis-",
        )
        if self.group_family_hash != expected_family_hash:
            raise ValueError("source-uncertainty family hash does not replay")
        for name, values in (
            ("source receipt hash", source_hashes),
            ("branch registration hash", registration_hashes),
        ):
            if not values:
                raise ValueError(f"{name} axis cannot be empty")
            tuple(_identity(value, name=name) for value in values)
        object.__setattr__(self, "source_groups", groups)
        object.__setattr__(self, "group_axis", axis)
        object.__setattr__(self, "group_receipt_hashes", hashes)
        object.__setattr__(self, "source_receipt_hashes", source_hashes)
        object.__setattr__(self, "branch_registration_hashes", registration_hashes)
        required_true = (
            self.complete_registered_branch_resource_source_product,
            self.every_group_has_explicit_coverage_receipt,
            self.exact_registered_population_census_family,
            self.array_free_receipt,
        )
        if any(value is not True for value in required_true):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "source-uncertainty family coverage is incomplete",
            )
        _closed_authority(
            (
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="source-uncertainty family receipt",
        )
        if self.schema != ACTION_SOURCE_UNCERTAINTY_FAMILY_SCHEMA:
            raise ValueError("unsupported source-uncertainty family schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "registry_receipt_hash": self.registry_receipt_hash,
                "registered_d_w2_prefix_hash": self.registered_d_w2_prefix_hash,
                "alpha_lineage_hash": self.alpha_lineage_hash,
                "group_axis": self.group_axis,
                "group_receipt_hashes": self.group_receipt_hashes,
                "group_family_hash": self.group_family_hash,
                "source_receipt_hashes": self.source_receipt_hashes,
                "branch_registration_hashes": self.branch_registration_hashes,
                "complete_registered_branch_resource_source_product": True,
                "every_group_has_explicit_coverage_receipt": True,
                "exact_registered_population_census_family": True,
                "array_free_receipt": True,
                "all_downstream_authority": False,
            },
            prefix=_PREFIX + "source-uncertainty-family-",
        )

    def group_for(
        self,
        *,
        branch: V1SignedWorkVariableShellQBranch,
        resource_alias_id: str,
        source_receipt_hash: str,
    ) -> V1SignedWorkVariableShellSourceUncertaintyGroupReceipt:
        matches = tuple(
            value
            for value in self.source_groups
            if value.branch is branch
            and value.resource_alias_id == resource_alias_id
            and value.source_receipt_hash == source_receipt_hash
        )
        if len(matches) != 1:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "source/branch/resource group has no unique uncertainty receipt",
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSampledSupportUncertaintyGroupReceipt:
    """Uniform sampled support-envelope coverage for one source/branch/resource group.

    This is the production-tractable successor to the exact-census-only source
    receipt.  The fitted variable-shell source may solve and freeze primals and
    support anchors, but population lower bounds must come from a fresh,
    independently registered D_W2 certification prefix.  For every frozen
    anchor ``z_t``, convexity of native-fiber signed work gives the pointwise
    support plane

    ``|b + A z| >= sign(b + A z_t) * (b + A z)``.

    A confidence-valid lower envelope over the pre-registered action cells and
    a conservative Q outer set may then replace a full train census for
    alternate-action lower bounds.  The receipt is still only a source
    uncertainty object: it grants no action-invariance or C authority.
    """

    registry_receipt_hash: str
    source_uncertainty_group_id: str
    fit_source_receipt_hash: str
    fit_source_replay_hash: str
    certification_source_hash: str
    branch_registration_hash: str
    branch: V1SignedWorkVariableShellQBranch
    resource_alias_id: str
    phase_a_invariance_receipt_hash: str
    registered_d_w2_fit_prefix_hash: str
    registered_d_w2_cert_prefix_hash: str
    alpha_lineage_hash: str
    population_context_axis_hash: str
    fit_context_count: int
    cert_context_count: int
    population_count: int
    shell_size: int
    support_anchor_axis_hash: str
    support_anchor_count: int
    q_outer_set_hash: str
    uniform_affine_lower_envelope_hash: str
    optional_stopping_eprocess_receipt_hash: str
    alpha_ledger_family_hash: str
    familywise_alpha_spent: float
    _issuer_token: InitVar[object] = None
    sampled_fit_prefix_allowed: bool = True
    fit_stream_frozen_before_cert_stream: bool = True
    cert_stream_independent_after_anchor_freeze: bool = True
    support_planes_from_convex_abs: bool = True
    uniform_confidence_over_registered_action_cells: bool = True
    q_outer_set_contains_all_admissible_cells: bool = True
    q_outer_set_is_conservative_registered_branch_fd_envelope: bool = True
    d_w2_exact_sample_q_not_used_as_registered_branch_feasible_set: bool = True
    lower_envelope_uses_only_cert_stream: bool = True
    exact_registered_population_census: bool = False
    statistical_uncertainty_spends_alpha: bool = True
    array_free_receipt: bool = True
    no_post_selection_leakage: bool = True
    actual_population_bounds_consumed: bool = False
    sampled_support_metadata_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = ACTION_SAMPLED_SUPPORT_UNCERTAINTY_GROUP_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _SAMPLED_UNCERTAINTY_GROUP_TOKEN:
            raise TypeError("sampled support group receipts are issuer-only")
        for name in (
            "registry_receipt_hash",
            "source_uncertainty_group_id",
            "fit_source_receipt_hash",
            "fit_source_replay_hash",
            "certification_source_hash",
            "branch_registration_hash",
            "resource_alias_id",
            "phase_a_invariance_receipt_hash",
            "registered_d_w2_fit_prefix_hash",
            "registered_d_w2_cert_prefix_hash",
            "alpha_lineage_hash",
            "population_context_axis_hash",
            "support_anchor_axis_hash",
            "q_outer_set_hash",
            "uniform_affine_lower_envelope_hash",
            "optional_stopping_eprocess_receipt_hash",
            "alpha_ledger_family_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("sampled support group branch must be exact")
        fit_count = _positive_int(self.fit_context_count, name="fit context count")
        cert_count = _positive_int(self.cert_context_count, name="cert context count")
        population = _positive_int(self.population_count, name="population count")
        shell = _positive_int(self.shell_size, name="shell size")
        anchors = _positive_int(self.support_anchor_count, name="support anchor count")
        if fit_count >= population or cert_count >= population:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled support issuer is for proper prefixes; exact census uses the census issuer",
            )
        if shell > _MAXIMUM_SHELL_SIZE or anchors > _MAXIMUM_ACTION_CELLS:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled support envelope exceeded the registered B<=8 shell",
            )
        if (
            self.registered_d_w2_fit_prefix_hash
            == self.registered_d_w2_cert_prefix_hash
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled support certification must use an independent D_W2 prefix",
            )
        alpha = _finite_positive(
            self.familywise_alpha_spent,
            name="sampled support familywise alpha spent",
        )
        if alpha > 1.0:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled support familywise alpha must be a probability",
            )
        object.__setattr__(self, "fit_context_count", fit_count)
        object.__setattr__(self, "cert_context_count", cert_count)
        object.__setattr__(self, "population_count", population)
        object.__setattr__(self, "shell_size", shell)
        object.__setattr__(self, "support_anchor_count", anchors)
        object.__setattr__(self, "familywise_alpha_spent", alpha)
        required_true = (
            self.sampled_fit_prefix_allowed,
            self.fit_stream_frozen_before_cert_stream,
            self.cert_stream_independent_after_anchor_freeze,
            self.support_planes_from_convex_abs,
            self.uniform_confidence_over_registered_action_cells,
            self.q_outer_set_contains_all_admissible_cells,
            self.q_outer_set_is_conservative_registered_branch_fd_envelope,
            self.d_w2_exact_sample_q_not_used_as_registered_branch_feasible_set,
            self.lower_envelope_uses_only_cert_stream,
            self.statistical_uncertainty_spends_alpha,
            self.array_free_receipt,
            self.no_post_selection_leakage,
        )
        if (
            any(value is not True for value in required_true)
            or self.exact_registered_population_census is not False
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled support uncertainty coverage is incomplete",
            )
        _closed_authority(
            (
                self.actual_population_bounds_consumed,
                self.sampled_support_metadata_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="sampled support uncertainty group receipt",
        )
        if self.schema != ACTION_SAMPLED_SUPPORT_UNCERTAINTY_GROUP_SCHEMA:
            raise ValueError("unsupported sampled support uncertainty group schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "sampled-support-group-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSampledSupportUncertaintyFamilyReceipt:
    """Complete sampled support-envelope family over the frozen action registry."""

    registry_receipt_hash: str
    registered_d_w2_fit_prefix_hash: str
    registered_d_w2_cert_prefix_hash: str
    alpha_lineage_hash: str
    support_anchor_axis_hash: str
    q_outer_set_hash: str
    uniform_affine_lower_envelope_hash: str
    optional_stopping_eprocess_receipt_hash: str
    alpha_ledger_family_hash: str
    familywise_alpha_spent: float
    group_axis: tuple[tuple[str, str, str, str, str, str, str, str], ...]
    group_receipt_hashes: tuple[str, ...]
    group_family_hash: str
    fit_source_receipt_hashes: tuple[str, ...]
    branch_registration_hashes: tuple[str, ...]
    source_groups: tuple[
        V1SignedWorkVariableShellSampledSupportUncertaintyGroupReceipt, ...
    ] = field(repr=False, compare=False)
    _issuer_token: InitVar[object] = None
    complete_registered_branch_resource_source_product: bool = True
    every_group_has_uniform_support_envelope: bool = True
    sampled_support_family: bool = True
    exact_registered_population_census_family: bool = False
    statistical_uncertainty_spends_alpha: bool = True
    q_outer_set_uniform_over_all_action_cells: bool = True
    q_outer_set_is_conservative_registered_branch_fd_envelope: bool = True
    d_w2_exact_sample_q_not_used_as_registered_branch_feasible_set: bool = True
    array_free_receipt: bool = True
    actual_population_bounds_consumed: bool = False
    sampled_support_metadata_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = ACTION_SAMPLED_SUPPORT_UNCERTAINTY_FAMILY_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _SAMPLED_UNCERTAINTY_FAMILY_TOKEN:
            raise TypeError("sampled support family receipts are issuer-only")
        for name in (
            "registry_receipt_hash",
            "registered_d_w2_fit_prefix_hash",
            "registered_d_w2_cert_prefix_hash",
            "alpha_lineage_hash",
            "support_anchor_axis_hash",
            "q_outer_set_hash",
            "uniform_affine_lower_envelope_hash",
            "optional_stopping_eprocess_receipt_hash",
            "alpha_ledger_family_hash",
            "group_family_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        groups = tuple(self.source_groups)
        axis = tuple(tuple(value) for value in self.group_axis)
        hashes = tuple(self.group_receipt_hashes)
        if (
            not groups
            or any(
                type(value)
                is not V1SignedWorkVariableShellSampledSupportUncertaintyGroupReceipt
                for value in groups
            )
            or len(groups) != len(axis)
            or len(groups) != len(hashes)
            or tuple(value.receipt_hash for value in groups) != hashes
            or len(set(axis)) != len(axis)
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled support group receipt axis is incomplete",
            )
        observed_axis = tuple(
            (
                value.source_uncertainty_group_id,
                value.fit_source_receipt_hash,
                value.certification_source_hash,
                value.branch.value,
                value.resource_alias_id,
                value.phase_a_invariance_receipt_hash,
                value.support_anchor_axis_hash,
                value.q_outer_set_hash,
            )
            for value in groups
        )
        if observed_axis != axis:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled support group identity differs from its family axis",
            )
        if (
            self.registered_d_w2_fit_prefix_hash
            == self.registered_d_w2_cert_prefix_hash
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled support family collapsed fit and certification streams",
            )
        expected_family_hash = content_hash(
            {
                "group_axis": axis,
                "group_receipt_hashes": hashes,
                "support_anchor_axis_hash": self.support_anchor_axis_hash,
                "q_outer_set_hash": self.q_outer_set_hash,
                "uniform_affine_lower_envelope_hash": (
                    self.uniform_affine_lower_envelope_hash
                ),
            },
            prefix=_PREFIX + "sampled-support-family-axis-",
        )
        if self.group_family_hash != expected_family_hash:
            raise ValueError("sampled support family hash does not replay")
        tuple(
            _identity(value, name="fit source hash")
            for value in self.fit_source_receipt_hashes
        )
        tuple(
            _identity(value, name="branch registration hash")
            for value in self.branch_registration_hashes
        )
        alpha = _finite_positive(
            self.familywise_alpha_spent,
            name="sampled support familywise alpha spent",
        )
        if alpha > 1.0:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled support familywise alpha must be a probability",
            )
        object.__setattr__(self, "familywise_alpha_spent", alpha)
        required_true = (
            self.complete_registered_branch_resource_source_product,
            self.every_group_has_uniform_support_envelope,
            self.sampled_support_family,
            self.statistical_uncertainty_spends_alpha,
            self.q_outer_set_uniform_over_all_action_cells,
            self.q_outer_set_is_conservative_registered_branch_fd_envelope,
            self.d_w2_exact_sample_q_not_used_as_registered_branch_feasible_set,
            self.array_free_receipt,
        )
        if (
            any(value is not True for value in required_true)
            or self.exact_registered_population_census_family is not False
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled support family coverage is incomplete",
            )
        _closed_authority(
            (
                self.actual_population_bounds_consumed,
                self.sampled_support_metadata_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="sampled support uncertainty family receipt",
        )
        if self.schema != ACTION_SAMPLED_SUPPORT_UNCERTAINTY_FAMILY_SCHEMA:
            raise ValueError("unsupported sampled support uncertainty family schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "registry_receipt_hash": self.registry_receipt_hash,
                "registered_d_w2_fit_prefix_hash": (
                    self.registered_d_w2_fit_prefix_hash
                ),
                "registered_d_w2_cert_prefix_hash": (
                    self.registered_d_w2_cert_prefix_hash
                ),
                "alpha_lineage_hash": self.alpha_lineage_hash,
                "support_anchor_axis_hash": self.support_anchor_axis_hash,
                "q_outer_set_hash": self.q_outer_set_hash,
                "uniform_affine_lower_envelope_hash": (
                    self.uniform_affine_lower_envelope_hash
                ),
                "optional_stopping_eprocess_receipt_hash": (
                    self.optional_stopping_eprocess_receipt_hash
                ),
                "alpha_ledger_family_hash": self.alpha_ledger_family_hash,
                "familywise_alpha_spent": self.familywise_alpha_spent,
                "group_axis": self.group_axis,
                "group_receipt_hashes": self.group_receipt_hashes,
                "group_family_hash": self.group_family_hash,
                "fit_source_receipt_hashes": self.fit_source_receipt_hashes,
                "branch_registration_hashes": self.branch_registration_hashes,
                "sampled_support_family": True,
                "exact_registered_population_census_family": False,
                "q_outer_set_is_conservative_registered_branch_fd_envelope": True,
                "d_w2_exact_sample_q_not_used_as_registered_branch_feasible_set": True,
                "actual_population_bounds_consumed": False,
                "sampled_support_metadata_authority": False,
                "all_downstream_authority": False,
            },
            prefix=_PREFIX + "sampled-support-family-",
        )

    def group_for(
        self,
        *,
        branch: V1SignedWorkVariableShellQBranch,
        resource_alias_id: str,
        fit_source_receipt_hash: str,
        certification_source_hash: str,
    ) -> V1SignedWorkVariableShellSampledSupportUncertaintyGroupReceipt:
        matches = tuple(
            value
            for value in self.source_groups
            if value.branch is branch
            and value.resource_alias_id == resource_alias_id
            and value.fit_source_receipt_hash == fit_source_receipt_hash
            and value.certification_source_hash == certification_source_hash
        )
        if len(matches) != 1:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled support source/branch/resource group has no unique receipt",
            )
        return matches[0]

    def group_for_fit_source(
        self,
        *,
        branch: V1SignedWorkVariableShellQBranch,
        resource_alias_id: str,
        fit_source_receipt_hash: str,
    ) -> V1SignedWorkVariableShellSampledSupportUncertaintyGroupReceipt:
        matches = tuple(
            value
            for value in self.source_groups
            if value.branch is branch
            and value.resource_alias_id == resource_alias_id
            and value.fit_source_receipt_hash == fit_source_receipt_hash
        )
        if len(matches) != 1:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled support fit source/branch/resource group has no unique receipt",
            )
        return matches[0]


def issue_v1_signed_work_variable_shell_sampled_support_uncertainty_family(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    certification_source_hash: str,
    registered_d_w2_cert_prefix_hash: str,
    cert_context_count: int,
    support_anchor_axis_hash: str,
    support_anchor_count: int,
    q_outer_set_hash: str,
    uniform_affine_lower_envelope_hash: str,
    optional_stopping_eprocess_receipt_hash: str,
    alpha_ledger_family_hash: str,
    familywise_alpha_spent: float,
) -> V1SignedWorkVariableShellSampledSupportUncertaintyFamilyReceipt:
    """Issue sampled D_W2 support-envelope coverage for the frozen registry.

    The returned family is sufficient to identify which source/branch/resource
    groups are uniformly enclosed by an independent certification stream.  It
    does not solve any restricted cell and therefore cannot by itself prove
    action separation or open C.
    """

    if type(registry) is not V1SignedWorkVariableShellActionCellRegistry:
        raise TypeError("sampled support issuance requires the exact live registry")
    registry.assert_live_sources()
    if not registry.receipt.complete_branch_axis:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
            "sampled support issuance requires the complete Q-branch axis",
        )
    cert_hash = _identity(certification_source_hash, name="certification source hash")
    cert_prefix = _identity(
        registered_d_w2_cert_prefix_hash,
        name="registered D_W2 cert prefix hash",
    )
    anchor_hash = _identity(support_anchor_axis_hash, name="support anchor axis hash")
    q_outer = _identity(q_outer_set_hash, name="Q outer-set hash")
    lower_hash = _identity(
        uniform_affine_lower_envelope_hash,
        name="uniform affine lower-envelope hash",
    )
    eprocess_hash = _identity(
        optional_stopping_eprocess_receipt_hash,
        name="optional-stopping e-process receipt hash",
    )
    ledger_hash = _identity(alpha_ledger_family_hash, name="alpha ledger family hash")
    cert_count = _positive_int(cert_context_count, name="cert context count")
    anchor_count = _positive_int(support_anchor_count, name="support anchor count")
    alpha = _finite_positive(familywise_alpha_spent, name="familywise alpha spent")
    if alpha > 1.0:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
            "sampled support familywise alpha must be a probability",
        )
    groups: list[V1SignedWorkVariableShellSampledSupportUncertaintyGroupReceipt] = []
    for source in registry.branch_sources:
        receipt = source.assert_live_source().receipt
        if receipt.finite_census_complete:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "exact D_W2 census must use the exact-census uncertainty issuer",
            )
        if cert_count >= receipt.population_count:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled support certification prefix must be a proper prefix",
            )
        for alias in source.resource_aliases:
            group_id = content_hash(
                {
                    "registry_receipt_hash": registry.receipt.receipt_hash,
                    "fit_source_receipt_hash": receipt.receipt_hash,
                    "certification_source_hash": cert_hash,
                    "branch": source.branch.value,
                    "resource_alias_id": alias.resource_alias_id,
                    "phase_a_invariance_receipt_hash": (
                        alias.phase_a_invariance_receipt_hash
                    ),
                    "support_anchor_axis_hash": anchor_hash,
                    "q_outer_set_hash": q_outer,
                    "uniform_affine_lower_envelope_hash": lower_hash,
                },
                prefix=_PREFIX + "sampled-support-group-id-",
            )
            groups.append(
                V1SignedWorkVariableShellSampledSupportUncertaintyGroupReceipt(
                    registry_receipt_hash=registry.receipt.receipt_hash,
                    source_uncertainty_group_id=group_id,
                    fit_source_receipt_hash=receipt.receipt_hash,
                    fit_source_replay_hash=receipt.source_replay_hash,
                    certification_source_hash=cert_hash,
                    branch_registration_hash=source.registration.registration_hash,
                    branch=source.branch,
                    resource_alias_id=alias.resource_alias_id,
                    phase_a_invariance_receipt_hash=(
                        alias.phase_a_invariance_receipt_hash
                    ),
                    registered_d_w2_fit_prefix_hash=(
                        receipt.registered_d_w2_prefix_hash
                    ),
                    registered_d_w2_cert_prefix_hash=cert_prefix,
                    alpha_lineage_hash=receipt.alpha_lineage_hash,
                    population_context_axis_hash=(receipt.population_context_axis_hash),
                    fit_context_count=receipt.context_count,
                    cert_context_count=cert_count,
                    population_count=receipt.population_count,
                    shell_size=len(receipt.shell_indices),
                    support_anchor_axis_hash=anchor_hash,
                    support_anchor_count=anchor_count,
                    q_outer_set_hash=q_outer,
                    uniform_affine_lower_envelope_hash=lower_hash,
                    optional_stopping_eprocess_receipt_hash=eprocess_hash,
                    alpha_ledger_family_hash=ledger_hash,
                    familywise_alpha_spent=alpha,
                    _issuer_token=_SAMPLED_UNCERTAINTY_GROUP_TOKEN,
                )
            )
    axis = tuple(
        (
            value.source_uncertainty_group_id,
            value.fit_source_receipt_hash,
            value.certification_source_hash,
            value.branch.value,
            value.resource_alias_id,
            value.phase_a_invariance_receipt_hash,
            value.support_anchor_axis_hash,
            value.q_outer_set_hash,
        )
        for value in groups
    )
    hashes = tuple(value.receipt_hash for value in groups)
    family_hash = content_hash(
        {
            "group_axis": axis,
            "group_receipt_hashes": hashes,
            "support_anchor_axis_hash": anchor_hash,
            "q_outer_set_hash": q_outer,
            "uniform_affine_lower_envelope_hash": lower_hash,
        },
        prefix=_PREFIX + "sampled-support-family-axis-",
    )
    return V1SignedWorkVariableShellSampledSupportUncertaintyFamilyReceipt(
        registry_receipt_hash=registry.receipt.receipt_hash,
        registered_d_w2_fit_prefix_hash=groups[0].registered_d_w2_fit_prefix_hash,
        registered_d_w2_cert_prefix_hash=cert_prefix,
        alpha_lineage_hash=registry.receipt.alpha_lineage_hash,
        support_anchor_axis_hash=anchor_hash,
        q_outer_set_hash=q_outer,
        uniform_affine_lower_envelope_hash=lower_hash,
        optional_stopping_eprocess_receipt_hash=eprocess_hash,
        alpha_ledger_family_hash=ledger_hash,
        group_axis=axis,
        group_receipt_hashes=hashes,
        group_family_hash=family_hash,
        fit_source_receipt_hashes=tuple(value.receipt_hash for value in groups),
        branch_registration_hashes=tuple(
            source.registration.registration_hash for source in registry.branch_sources
        ),
        source_groups=tuple(groups),
        familywise_alpha_spent=alpha,
        _issuer_token=_SAMPLED_UNCERTAINTY_FAMILY_TOKEN,
    )


def issue_v1_signed_work_variable_shell_exact_census_uncertainty_family(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
) -> V1SignedWorkVariableShellSourceUncertaintyFamilyReceipt:
    """Issue the only currently sound family: every registered source is census-exact."""

    if type(registry) is not V1SignedWorkVariableShellActionCellRegistry:
        raise TypeError("uncertainty-family issuance requires the exact live registry")
    registry.assert_live_sources()
    if not registry.receipt.complete_branch_axis:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
            "uncertainty-family issuance requires the complete Q-branch axis",
        )
    groups: list[V1SignedWorkVariableShellSourceUncertaintyGroupReceipt] = []
    for source in registry.branch_sources:
        receipt = source.assert_live_source().receipt
        if type(receipt) is V1SignedWorkVariableShellSourceAppendSegmentedReceipt:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "segmented D_W2 is proposal/fit-only until residual population measure and exact D1 F/Q are joined",
            )
        if not receipt.finite_census_complete:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled D_W2 prefixes need a future uniform-confidence family issuer",
            )
        for alias in source.resource_aliases:
            group_id = content_hash(
                {
                    "registry_receipt_hash": registry.receipt.receipt_hash,
                    "source_receipt_hash": receipt.receipt_hash,
                    "branch": source.branch.value,
                    "resource_alias_id": alias.resource_alias_id,
                    "phase_a_invariance_receipt_hash": (
                        alias.phase_a_invariance_receipt_hash
                    ),
                },
                prefix=_PREFIX + "source-uncertainty-group-id-",
            )
            coverage_basis_hash = content_hash(
                {
                    "source_replay_hash": receipt.source_replay_hash,
                    "registered_d_w2_prefix_hash": receipt.registered_d_w2_prefix_hash,
                    "context_axis_hash": receipt.context_axis_hash,
                    "context_count": receipt.context_count,
                    "population_count": receipt.population_count,
                    "finite_census_complete": receipt.finite_census_complete,
                    "kkt_branch_source_receipt_hash": (
                        receipt.kkt_branch_source_receipt_hash
                    ),
                    "source_resource_receipt_hash": alias.source_resource_receipt_hash,
                },
                prefix=_PREFIX + "exact-census-coverage-",
            )
            groups.append(
                V1SignedWorkVariableShellSourceUncertaintyGroupReceipt(
                    registry_receipt_hash=registry.receipt.receipt_hash,
                    source_uncertainty_group_id=group_id,
                    source_receipt_hash=receipt.receipt_hash,
                    source_replay_hash=receipt.source_replay_hash,
                    branch_registration_hash=source.registration.registration_hash,
                    branch=source.branch,
                    resource_alias_id=alias.resource_alias_id,
                    phase_a_invariance_receipt_hash=(
                        alias.phase_a_invariance_receipt_hash
                    ),
                    registered_d_w2_prefix_hash=receipt.registered_d_w2_prefix_hash,
                    alpha_lineage_hash=receipt.alpha_lineage_hash,
                    population_context_axis_hash=(receipt.population_context_axis_hash),
                    context_count=receipt.context_count,
                    population_count=receipt.population_count,
                    coverage_basis_hash=coverage_basis_hash,
                    _issuer_token=_UNCERTAINTY_GROUP_TOKEN,
                )
            )
    group_axis = tuple(
        (
            value.source_uncertainty_group_id,
            value.source_receipt_hash,
            value.branch.value,
            value.resource_alias_id,
            value.phase_a_invariance_receipt_hash,
        )
        for value in groups
    )
    hashes = tuple(value.receipt_hash for value in groups)
    return V1SignedWorkVariableShellSourceUncertaintyFamilyReceipt(
        registry_receipt_hash=registry.receipt.receipt_hash,
        registered_d_w2_prefix_hash=registry.receipt.registered_d_w2_prefix_hash,
        alpha_lineage_hash=registry.receipt.alpha_lineage_hash,
        group_axis=group_axis,
        group_receipt_hashes=hashes,
        group_family_hash=content_hash(
            {"group_axis": group_axis, "group_receipt_hashes": hashes},
            prefix=_PREFIX + "source-uncertainty-family-axis-",
        ),
        source_receipt_hashes=registry.receipt.source_receipt_hashes,
        branch_registration_hashes=registry.receipt.branch_registration_hashes,
        source_groups=tuple(groups),
        _issuer_token=_UNCERTAINTY_FAMILY_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellRegisteredQBranchDomainBridgeReceipt:
    """Typed relation between registered D branch Q and D_W2 literal-J Q.

    Current production acquisition does not yet issue this relation.  The
    exact-reference issuer below exists only for synthetic fixtures whose
    problem registration explicitly defines the two domains to be identical.
    """

    registry_receipt_hash: str
    request_hash: str
    source_receipt_hash: str
    branch_registration_hash: str
    branch: V1SignedWorkVariableShellQBranch
    resource_alias_id: str
    phase_a_invariance_receipt_hash: str
    registered_q_branch_source_receipt_hash: str
    d_w2_literal_j_stream_source_hash: str
    uncertainty_relation_receipt_hash: str
    preferred_original_registered_q_upper_hard_replay_receipt_hash: str
    literal_j_response_radius: float
    _issuer_token: InitVar[object] = None
    exact_synthetic_reference_identity: bool = False
    production_registered_branch_enclosure_authority: bool = False
    alternate_cell_lower_bound_Q_dropped: bool = False
    d_w2_empirical_q_silently_substituted_for_registered_branch: bool = False
    q_domain_hierarchy_rule: str = ACTION_Q_DOMAIN_HIERARCHY_RULE
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    schema: str = ACTION_Q_BRANCH_DOMAIN_BRIDGE_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _Q_DOMAIN_BRIDGE_TOKEN:
            raise TypeError("registered-Q branch bridges require their private issuer")
        for name in (
            "registry_receipt_hash",
            "request_hash",
            "source_receipt_hash",
            "branch_registration_hash",
            "resource_alias_id",
            "phase_a_invariance_receipt_hash",
            "registered_q_branch_source_receipt_hash",
            "d_w2_literal_j_stream_source_hash",
            "uncertainty_relation_receipt_hash",
            "preferred_original_registered_q_upper_hard_replay_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("registered-Q branch bridge branch must be exact")
        object.__setattr__(
            self,
            "literal_j_response_radius",
            _finite_nonnegative(
                self.literal_j_response_radius,
                name="literal-J response radius",
            ),
        )
        if type(self.exact_synthetic_reference_identity) is not bool:
            raise TypeError("synthetic Q-domain identity marker must be boolean")
        _closed_authority(
            (self.action_invariance_authority, self.C_stream_open_authority),
            owner="registered-Q branch bridge",
        )
        if (
            self.exact_synthetic_reference_identity
            and self.production_registered_branch_enclosure_authority
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "a synthetic Q-domain identity cannot acquire production enclosure authority",
            )
        if (
            not self.exact_synthetic_reference_identity
            and not self.production_registered_branch_enclosure_authority
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.REGISTERED_Q_BRANCH_DOMAIN_BRIDGE_UNAVAILABLE,
                "Q-domain bridge proves neither a synthetic identity nor a production enclosure",
            )
        if (
            self.production_registered_branch_enclosure_authority
            and not self.alternate_cell_lower_bound_Q_dropped
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.REGISTERED_Q_BRANCH_DOMAIN_BRIDGE_UNAVAILABLE,
                "the first production tier must drop Q for alternate lower-bound domains",
            )
        if (
            self.d_w2_empirical_q_silently_substituted_for_registered_branch
            is not False
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "fresh-prefix empirical literal-J Q cannot impersonate registered q_rho/q_d",
            )
        if self.q_domain_hierarchy_rule != ACTION_Q_DOMAIN_HIERARCHY_RULE:
            raise ValueError("registered-Q domain hierarchy changed")
        if self.schema != ACTION_Q_BRANCH_DOMAIN_BRIDGE_SCHEMA:
            raise ValueError("unsupported registered-Q branch bridge schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "registered-q-branch-domain-bridge-",
        )


def issue_v1_signed_work_variable_shell_exact_reference_q_domain_identity(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    request: V1SignedWorkVariableShellRestrictedConvexCellRequest,
    alternate_cell_lower_bound_Q_dropped: bool = False,
) -> V1SignedWorkVariableShellRegisteredQBranchDomainBridgeReceipt:
    """Issue a fixture-only identity; never upgrades production Q semantics."""

    if type(registry) is not V1SignedWorkVariableShellActionCellRegistry:
        raise TypeError("reference Q identity requires the exact live registry")
    if type(request) is not V1SignedWorkVariableShellRestrictedConvexCellRequest:
        raise TypeError("reference Q identity requires the exact cell request")
    if (
        registry.design.execution_mode
        is not V1SignedWorkVariableShellActionSeparationMode.EXACT_REFERENCE_TEST_ONLY
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.REGISTERED_Q_BRANCH_DOMAIN_BRIDGE_UNAVAILABLE,
            "production registered q_rho/q_d domains lack a source-bound D-to-D_W2 enclosure",
        )
    if type(alternate_cell_lower_bound_Q_dropped) is not bool:
        raise TypeError("Q-dropped reference marker must be boolean")
    source = registry.branch_source(request.action_cell.branch)
    receipt = source.assert_live_source().receipt
    if (
        request.registry_receipt_hash != registry.receipt.receipt_hash
        or request.source_receipt_hash != receipt.receipt_hash
        or not receipt.finite_census_complete
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
            "synthetic Q identity requires one exact-census request/source registry",
        )
    relation = content_hash(
        {
            "request_hash": request.request_hash,
            "registered_q_branch_source_receipt_hash": (
                receipt.kkt_branch_source_receipt_hash
            ),
            "d_w2_literal_j_stream_source_hash": receipt.literal_j_stream_source_hash,
            "fixture_axiom": "registered_synthetic_branch_domain_equals_exact_census_literal_j_ball",
        },
        prefix=_PREFIX + "synthetic-q-domain-identity-",
    )
    return V1SignedWorkVariableShellRegisteredQBranchDomainBridgeReceipt(
        registry_receipt_hash=registry.receipt.receipt_hash,
        request_hash=request.request_hash,
        source_receipt_hash=receipt.receipt_hash,
        branch_registration_hash=source.registration.registration_hash,
        branch=request.action_cell.branch,
        resource_alias_id=request.action_cell.resource_alias_id,
        phase_a_invariance_receipt_hash=(
            request.action_cell.phase_a_invariance_receipt_hash
        ),
        registered_q_branch_source_receipt_hash=(
            receipt.kkt_branch_source_receipt_hash
        ),
        d_w2_literal_j_stream_source_hash=receipt.literal_j_stream_source_hash,
        uncertainty_relation_receipt_hash=relation,
        preferred_original_registered_q_upper_hard_replay_receipt_hash=relation,
        literal_j_response_radius=request.literal_j_response_radius,
        exact_synthetic_reference_identity=True,
        alternate_cell_lower_bound_Q_dropped=alternate_cell_lower_bound_Q_dropped,
        _issuer_token=_Q_DOMAIN_BRIDGE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellPhaseBSourceCompactionReceipt:
    source_receipt_hash: str
    source_replay_hash: str
    compact_source_tensor_lease_hash: str
    context_count: int
    full_native_fiber_count: int
    cut_count: int
    shell_size: int
    unique_shell_native_fiber_indices: tuple[int, ...]
    shell_to_unique_fiber_offsets: tuple[int, ...]
    untouched_native_fiber_count: int
    variable_affine_row_count: int
    untouched_F_constant_lower: float
    untouched_F_constant_center: float
    untouched_F_constant_upper: float
    _issuer_token: InitVar[object] = None
    exact_destination_fiber_coalescing: bool = True
    untouched_fibers_z_independent: bool = True
    multiple_shell_routes_same_fiber_sum_before_abs: bool = True
    structural_equality_to_full_source_F: bool = True
    segmented_proposal_fit_source_only: bool = False
    exact_d1_join_required: bool = False
    full_population_action_witness_authority: bool = False
    route_scale_rows_retained: bool = False
    route_scale_D2H_bytes: int = 0
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    compaction_rule: str = ACTION_SOURCE_COMPACTION_RULE
    schema: str = ACTION_SOURCE_COMPACTION_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _SOLVER_RECEIPT_TOKEN:
            raise TypeError("source compaction receipts require the private issuer")
        for name in (
            "source_receipt_hash",
            "source_replay_hash",
            "compact_source_tensor_lease_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "context_count",
            "full_native_fiber_count",
            "cut_count",
            "shell_size",
            "variable_affine_row_count",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name.replace("_", " ")),
            )
        object.__setattr__(
            self,
            "untouched_native_fiber_count",
            _nonnegative_int(
                self.untouched_native_fiber_count,
                name="untouched native fiber count",
            ),
        )
        unique = tuple(
            _nonnegative_int(value, name="unique shell native-fiber index")
            for value in self.unique_shell_native_fiber_indices
        )
        inverse = tuple(
            _nonnegative_int(value, name="shell-to-unique fiber offset")
            for value in self.shell_to_unique_fiber_offsets
        )
        if (
            unique != tuple(sorted(set(unique)))
            or len(unique) > self.shell_size
            or len(inverse) != self.shell_size
            or any(value >= len(unique) for value in inverse)
            or self.untouched_native_fiber_count
            != self.full_native_fiber_count - len(unique)
            or self.variable_affine_row_count != self.context_count * len(unique)
        ):
            raise ValueError("source compaction axes do not close")
        object.__setattr__(self, "unique_shell_native_fiber_indices", unique)
        object.__setattr__(self, "shell_to_unique_fiber_offsets", inverse)
        lower = _finite_nonnegative(
            self.untouched_F_constant_lower,
            name="untouched F constant lower",
        )
        center = _finite_nonnegative(
            self.untouched_F_constant_center,
            name="untouched F constant center",
        )
        upper = _finite_nonnegative(
            self.untouched_F_constant_upper,
            name="untouched F constant upper",
        )
        if not lower <= center <= upper:
            raise ValueError("untouched F constant enclosure is reversed")
        object.__setattr__(self, "untouched_F_constant_lower", lower)
        object.__setattr__(self, "untouched_F_constant_center", center)
        object.__setattr__(self, "untouched_F_constant_upper", upper)
        required_true = (
            self.exact_destination_fiber_coalescing,
            self.untouched_fibers_z_independent,
            self.multiple_shell_routes_same_fiber_sum_before_abs,
            self.structural_equality_to_full_source_F,
        )
        if any(value is not True for value in required_true):
            raise ValueError("source compaction lost its exact structural identity")
        if any(
            type(value) is not bool
            for value in (
                self.segmented_proposal_fit_source_only,
                self.exact_d1_join_required,
                self.full_population_action_witness_authority,
            )
        ):
            raise TypeError("source-compaction scope markers must be exact booleans")
        if self.segmented_proposal_fit_source_only is not self.exact_d1_join_required:
            raise ValueError("segmented fit scope and missing exact-D1 join disagree")
        _closed_authority(
            (
                self.full_population_action_witness_authority,
                self.route_scale_rows_retained,
                self.action_invariance_authority,
                self.C_stream_open_authority,
            ),
            owner="source compaction receipt",
        )
        if (
            _nonnegative_int(self.route_scale_D2H_bytes, name="route-scale D2H bytes")
            != 0
        ):
            raise ValueError("source compaction performed a route-scale D2H transfer")
        if (
            self.compaction_rule != ACTION_SOURCE_COMPACTION_RULE
            or self.schema != ACTION_SOURCE_COMPACTION_SCHEMA
        ):
            raise ValueError("unsupported source-compaction semantics")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "source-compaction-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellPhaseBCompactEvaluationReceipt:
    source_compaction_receipt_hash: str
    source_receipt_hash: str
    shell_gate_hash: str
    F_lower: float
    F_center: float
    F_upper: float
    Q_lower: float
    Q_center: float
    Q_upper: float
    structural_equality_to_full_source: bool = True
    sample_function_evaluation_only: bool = True
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    schema: str = ACTION_COMPACT_EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "source_compaction_receipt_hash",
            "source_receipt_hash",
            "shell_gate_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for lower_name, center_name, upper_name in (
            ("F_lower", "F_center", "F_upper"),
            ("Q_lower", "Q_center", "Q_upper"),
        ):
            lower = _finite_nonnegative(getattr(self, lower_name), name=lower_name)
            center = _finite_nonnegative(getattr(self, center_name), name=center_name)
            upper = _finite_nonnegative(getattr(self, upper_name), name=upper_name)
            if not lower <= center <= upper:
                raise ValueError("compact evaluation interval is reversed")
            object.__setattr__(self, lower_name, lower)
            object.__setattr__(self, center_name, center)
            object.__setattr__(self, upper_name, upper)
        if (
            self.structural_equality_to_full_source is not True
            or self.sample_function_evaluation_only is not True
        ):
            raise ValueError("compact evaluation changed its evidence scope")
        _closed_authority(
            (self.action_invariance_authority, self.C_stream_open_authority),
            owner="compact source evaluation",
        )
        if self.schema != ACTION_COMPACT_EVALUATION_SCHEMA:
            raise ValueError("unsupported compact-evaluation schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "compact-evaluation-")


class V1SignedWorkVariableShellPhaseBCompactSourceHandle:
    """Nonserializable exact touched-fiber quotient retaining one source lease."""

    __slots__ = (
        "source_handle",
        "source_lease",
        "receipt",
        "b_touched",
        "b_radius_touched",
        "a_compact",
        "d_response",
        "d_radius",
        "j_shell",
        "weights",
        "weight_radii",
        "sqrt_weights",
        "alpha",
        "_tensor_invariants",
        "_closed",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        source_handle: V1SignedWorkVariableShellCompactSourceHandle,
        source_lease: V1SignedWorkVariableShellCompactSourceLease,
        receipt: V1SignedWorkVariableShellPhaseBSourceCompactionReceipt,
        b_touched: Any,
        b_radius_touched: Any,
        a_compact: Any,
        d_response: Any,
        d_radius: Any,
        j_shell: Any,
        weights: Any,
        weight_radii: Any,
    ) -> None:
        self.source_handle = source_handle
        self.source_lease = source_lease
        self.receipt = receipt
        self.b_touched = b_touched
        self.b_radius_touched = b_radius_touched
        self.a_compact = a_compact
        self.d_response = d_response
        self.d_radius = d_radius
        self.j_shell = j_shell
        self.weights = weights
        self.weight_radii = weight_radii
        self.sqrt_weights = torch.sqrt(weights)
        self.alpha = weights.unsqueeze(1) / float(receipt.cut_count)
        self._tensor_invariants = tuple(
            _tensor_invariant(value)
            for value in (
                self.b_touched,
                self.b_radius_touched,
                self.a_compact,
                self.d_response,
                self.d_radius,
                self.j_shell,
                self.weights,
                self.weight_radii,
                self.sqrt_weights,
                self.alpha,
            )
        )
        self._closed = False

    def assert_live(self) -> None:
        if self._closed:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.EVICTED_SOURCE,
                "compact Phase-B source was closed",
            )
        if type(self.source_lease) is V1SignedWorkVariableShellSourceTensorLease:
            self.source_lease.compact_tensors()
        elif (
            type(self.source_lease) is V1SignedWorkVariableShellSourceAppendSegmentLease
        ):
            self.source_lease.compact_segments()
        elif type(self.source_lease) is V1SampledFullPopulationCompactSourceLease:
            self.source_lease.d1_compact_tensors()
            self.source_lease.residual_compact_segments()
        else:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
                "compact Phase-B source lease changed type",
            )
        live_tensors = (
            self.b_touched,
            self.b_radius_touched,
            self.a_compact,
            self.d_response,
            self.d_radius,
            self.j_shell,
            self.weights,
            self.weight_radii,
            self.sqrt_weights,
            self.alpha,
        )
        if tuple(_tensor_invariant(value) for value in live_tensors) != (
            self._tensor_invariants
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
                "compact Phase-B derived tensors changed after sealing",
            )
        if self.source_handle.receipt.receipt_hash != self.receipt.source_receipt_hash:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
                "compact Phase-B source identity changed",
            )

    def evaluate(
        self,
        shell_gates: Sequence[Real] | Any,
    ) -> V1SignedWorkVariableShellPhaseBCompactEvaluationReceipt:
        self.assert_live()
        shell = self.receipt.shell_size
        device = self.b_touched.device
        if torch is not None and type(shell_gates) is torch.Tensor:
            if (
                shell_gates.dtype != torch.float64
                or tuple(shell_gates.shape) != (shell,)
                or shell_gates.device != device
                or not shell_gates.is_contiguous()
                or shell_gates.requires_grad
            ):
                raise TypeError(
                    "compact evaluation gates must be contiguous, grad-free FP64 "
                    "on the compact source device with shell shape"
                )
            if not bool(torch.isfinite(shell_gates).all().item()) or bool(
                ((shell_gates < 0.0) | (shell_gates > 1.0)).any().item()
            ):
                raise ValueError("compact evaluation gates must be finite in [0,1]")
            z = shell_gates
            gates = tuple(float(value) for value in z.detach().cpu().tolist())
        else:
            gates = _validated_gates(shell_gates, shell)
            z = torch.tensor(gates, dtype=torch.float64, device=device)
        response = self.b_touched + torch.einsum("cub,b->cu", self.a_compact, z)
        literal = self.d_response + torch.sum(self.j_shell * z.unsqueeze(0), dim=1)
        variable_center_terms = self.weights.unsqueeze(1) * torch.abs(response)
        variable_center_sum, variable_center_sum_upper = (
            _outward_nonnegative_tensor_sum(
                variable_center_terms,
                operations_per_term=4,
            )
        )
        variable_center = variable_center_sum / float(self.receipt.cut_count)
        variable_center_upper = _up(
            variable_center_sum_upper / float(self.receipt.cut_count)
        )
        q_center_terms = self.weights * literal * literal
        q_center, q_center_upper = _outward_nonnegative_tensor_sum(
            q_center_terms,
            operations_per_term=5,
        )
        affine_gamma = _gamma(2 * shell + 1)
        operand_mass = torch.abs(self.b_touched) + torch.einsum(
            "cub,b->cu",
            torch.abs(self.a_compact),
            torch.abs(z),
        )
        response_error = self.b_radius_touched + affine_gamma * operand_mass
        variable_error_terms = self.weights.unsqueeze(
            1
        ) * response_error + self.weight_radii.unsqueeze(1) * (
            torch.abs(response) + response_error
        )
        _variable_error_center, variable_error_sum_upper = (
            _outward_nonnegative_tensor_sum(
                variable_error_terms,
                operations_per_term=max(8, 2 * shell + 8),
            )
        )
        variable_error_upper = _up(
            variable_error_sum_upper / float(self.receipt.cut_count)
        )
        literal_mass = torch.abs(self.d_response) + torch.sum(
            torch.abs(self.j_shell * z.unsqueeze(0)), dim=1
        )
        literal_error = self.d_radius + affine_gamma * literal_mass
        q_error_terms = (
            self.weights
            * (2.0 * torch.abs(literal) * literal_error + literal_error * literal_error)
            + self.weight_radii * (torch.abs(literal) + literal_error) ** 2
        )
        _q_error_center, q_error_upper = _outward_nonnegative_tensor_sum(
            q_error_terms,
            operations_per_term=max(10, 2 * shell + 10),
        )
        f_center = self.receipt.untouched_F_constant_center + variable_center
        f_error = (
            self.receipt.untouched_F_constant_upper
            - self.receipt.untouched_F_constant_center
            + variable_error_upper
            + max(0.0, variable_center_upper - variable_center)
        )
        q_error = q_error_upper + max(0.0, q_center_upper - q_center)
        return V1SignedWorkVariableShellPhaseBCompactEvaluationReceipt(
            source_compaction_receipt_hash=self.receipt.receipt_hash,
            source_receipt_hash=self.receipt.source_receipt_hash,
            shell_gate_hash=content_hash(
                {
                    "source_receipt_hash": self.receipt.source_receipt_hash,
                    "gates": gates,
                },
                prefix=_PREFIX + "compact-z-",
            ),
            F_lower=max(0.0, _down(f_center - f_error)),
            F_center=f_center,
            F_upper=_up(f_center + f_error),
            Q_lower=max(0.0, _down(q_center - q_error)),
            Q_center=q_center,
            Q_upper=_up(q_center + q_error),
        )

    def close(self) -> None:
        if self._closed:
            return
        self.source_lease.close()
        self._closed = True
        self.b_touched = None
        self.b_radius_touched = None
        self.a_compact = None
        self.d_response = None
        self.d_radius = None
        self.j_shell = None
        self.weights = None
        self.weight_radii = None
        self.sqrt_weights = None
        self.alpha = None
        self._tensor_invariants = ()

    def __reduce__(self) -> NoReturn:
        raise TypeError("live compact Phase-B sources are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live compact Phase-B sources are nonserializable")


def _build_v1_signed_work_variable_shell_phase_b_compact_source_one_shot(
    *,
    branch_source: V1SignedWorkVariableShellBranchActionCellSource,
) -> V1SignedWorkVariableShellPhaseBCompactSourceHandle:
    if type(branch_source) is not V1SignedWorkVariableShellBranchActionCellSource:
        raise TypeError("source compaction requires an exact live branch source")
    source = branch_source.assert_live_source()
    if type(source) is not V1SignedWorkVariableShellSourceDeviceHandle:
        raise TypeError("one-shot compaction requires the exact one-shot source")
    purpose = content_hash(
        {
            "branch_registration_hash": branch_source.registration.registration_hash,
            "operation": "exact_touched_native_fiber_compaction",
        },
        prefix=_PREFIX + "source-compaction-purpose-",
    )
    lease = source.borrow_compact_tensors(borrow_purpose_hash=purpose)
    try:
        (
            b_fiber,
            b_radius,
            a_shell,
            d_response,
            d_radius,
            j_shell,
            weights,
            _shell_fibers_tensor,
        ) = lease.compact_tensors()
        receipt = source.receipt
        unique = tuple(sorted(set(receipt.shell_native_fiber_indices)))
        unique_offset = {value: index for index, value in enumerate(unique)}
        inverse = tuple(
            unique_offset[value] for value in receipt.shell_native_fiber_indices
        )
        device = b_fiber.device
        unique_tensor = torch.tensor(unique, dtype=torch.int64, device=device)
        b_touched = torch.index_select(b_fiber, 1, unique_tensor).contiguous()
        b_radius_touched = torch.index_select(b_radius, 1, unique_tensor).contiguous()
        contexts = receipt.context_count
        shell = len(receipt.shell_indices)
        a_compact = torch.zeros(
            (contexts, len(unique), shell),
            dtype=torch.float64,
            device=device,
        )
        for shell_offset, fiber_offset in enumerate(inverse):
            a_compact[:, fiber_offset, shell_offset] = a_shell[:, shell_offset]
        untouched_mask = torch.ones(
            (receipt.fiber_count,), dtype=torch.bool, device=device
        )
        untouched_mask[unique_tensor] = False
        b_untouched = b_fiber[:, untouched_mask]
        if b_untouched.numel():
            constant_center_terms = weights.unsqueeze(1) * torch.abs(b_untouched)
            constant_center_sum, constant_center_sum_upper = (
                _outward_nonnegative_tensor_sum(
                    constant_center_terms,
                    operations_per_term=4,
                )
            )
            del constant_center_terms, b_untouched
            radius_untouched = b_radius[:, untouched_mask]
            constant_radius_terms = weights.unsqueeze(1) * radius_untouched
            _constant_radius_sum, constant_radius_sum_upper = (
                _outward_nonnegative_tensor_sum(
                    constant_radius_terms,
                    operations_per_term=4,
                )
            )
            del constant_radius_terms, radius_untouched
            constant_center = constant_center_sum / float(receipt.cut_count)
            constant_center_upper = _up(
                constant_center_sum_upper / float(receipt.cut_count)
            )
            constant_radius = _up(
                constant_radius_sum_upper / float(receipt.cut_count)
                + max(0.0, constant_center_upper - constant_center)
            )
        else:
            constant_center = 0.0
            constant_radius = 0.0
        compaction_receipt = V1SignedWorkVariableShellPhaseBSourceCompactionReceipt(
            source_receipt_hash=receipt.receipt_hash,
            source_replay_hash=receipt.source_replay_hash,
            compact_source_tensor_lease_hash=lease.lease_hash,
            context_count=contexts,
            full_native_fiber_count=receipt.fiber_count,
            cut_count=receipt.cut_count,
            shell_size=shell,
            unique_shell_native_fiber_indices=unique,
            shell_to_unique_fiber_offsets=inverse,
            untouched_native_fiber_count=receipt.fiber_count - len(unique),
            variable_affine_row_count=contexts * len(unique),
            untouched_F_constant_lower=max(
                0.0, _down(constant_center - constant_radius)
            ),
            untouched_F_constant_center=constant_center,
            untouched_F_constant_upper=_up(constant_center + constant_radius),
            _issuer_token=_SOLVER_RECEIPT_TOKEN,
        )
        return V1SignedWorkVariableShellPhaseBCompactSourceHandle(
            source_handle=source,
            source_lease=lease,
            receipt=compaction_receipt,
            b_touched=b_touched,
            b_radius_touched=b_radius_touched,
            a_compact=a_compact,
            d_response=d_response,
            d_radius=d_radius,
            j_shell=j_shell,
            weights=weights,
            weight_radii=torch.zeros_like(weights),
        )
    except Exception:
        lease.close()
        raise


def _build_v1_signed_work_variable_shell_phase_b_compact_source_segmented(
    *,
    branch_source: V1SignedWorkVariableShellBranchActionCellSource,
) -> V1SignedWorkVariableShellPhaseBCompactSourceHandle:
    source = branch_source.assert_live_source()
    if type(source) is not V1SignedWorkVariableShellSourceAppendSegmentedHandle:
        raise TypeError("segmented compaction requires the exact segmented source")
    aggregate = source.receipt
    purpose = content_hash(
        {
            "branch_registration_hash": branch_source.registration.registration_hash,
            "append_receipt_hash": aggregate.append_receipt_hash,
            "source_chain_hash": aggregate.source_chain_hash,
            "operation": "per_segment_exact_touched_native_fiber_compaction",
        },
        prefix=_PREFIX + "segmented-source-compaction-purpose-",
    )
    lease = source.borrow_compact_segments(borrow_purpose_hash=purpose)
    try:
        segments = lease.compact_segments()
        common_fields = (
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "proposal_generation",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "shell_freeze_receipt_hash",
            "shell_indices",
            "shell_route_ids",
            "shell_native_fiber_indices",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "kkt_branch_id",
            "kkt_branch_source_receipt_hash",
            "resource_axis_hash",
            "resource_source_receipt_hash",
            "witness_stream_source_hash",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
            "alpha_lineage_hash",
            "design_hash",
            "population_count",
            "population_context_axis_hash",
            "population_draw_axis_hash",
            "route_count",
            "fiber_count",
            "cut_count",
        )
        if any(
            any(
                getattr(segment.source_receipt, name) != getattr(aggregate, name)
                for name in common_fields
            )
            for segment in segments
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
                "segmented source packet lineage or geometry differs from its aggregate receipt",
            )
        if (
            len(segments) != aggregate.segment_count
            or sum(segment.source_receipt.context_count for segment in segments)
            != aggregate.context_count
            or sum(segment.segment_measure_quanta for segment in segments)
            != aggregate.cumulative_measure_quanta
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "segmented source counts or integer measure do not close",
            )

        unique = tuple(sorted(set(aggregate.shell_native_fiber_indices)))
        unique_offset = {value: index for index, value in enumerate(unique)}
        inverse = tuple(
            unique_offset[value] for value in aggregate.shell_native_fiber_indices
        )
        shell = len(aggregate.shell_indices)
        # Packet-local normalizations are transport coordinates only.  Their
        # stratum totals generally change for an unequal terminal packet and
        # therefore cannot define the cumulative residual-population measure.
        # The append lineage instead binds one common integer unit: within a
        # stratum every context carries the same population quantum in every
        # segment.  Recover the invariant stratum capacities from that unit,
        # then divide each capacity by its *cumulative* sample count.
        stratum_context_quanta: dict[str, int] = {}
        cumulative_stratum_counts: dict[str, int] = {}
        for segment in segments:
            packet_quanta_by_stratum: dict[str, list[int]] = {}
            for stratum, quantum in zip(
                segment.logical_stratum_ids,
                segment.context_measure_quanta,
                strict=True,
            ):
                packet_quanta_by_stratum.setdefault(stratum, []).append(quantum)
                cumulative_stratum_counts[stratum] = (
                    cumulative_stratum_counts.get(stratum, 0) + 1
                )
            if any(
                not values
                or any(value <= 0 for value in values)
                or len(set(values)) != 1
                for values in packet_quanta_by_stratum.values()
            ):
                _refuse(
                    V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                    "one adaptive packet violates equal positive measure within a registered stratum",
                )
            if sum(segment.context_measure_quanta) != segment.segment_measure_quanta:
                _refuse(
                    V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                    "one segmented packet does not carry a complete normalized stratum measure",
                )
            for stratum, values in packet_quanta_by_stratum.items():
                quantum = values[0]
                previous = stratum_context_quanta.setdefault(stratum, quantum)
                if quantum != previous:
                    _refuse(
                        V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                        "adaptive packets changed the common population quantum of one stratum",
                    )
        total_stratum_quanta = sum(stratum_context_quanta.values())
        stratum_masses = (
            {
                stratum: Fraction(quantum, total_stratum_quanta)
                for stratum, quantum in stratum_context_quanta.items()
            }
            if total_stratum_quanta > 0
            else {}
        )
        if (
            not stratum_masses
            or set(cumulative_stratum_counts) != set(stratum_masses)
            or any(value <= 0 for value in stratum_masses.values())
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "segmented stratum population measure is empty or incomplete",
            )
        b_touched_segments: list[Any] = []
        b_radius_segments: list[Any] = []
        a_compact_segments: list[Any] = []
        d_response_segments: list[Any] = []
        d_radius_segments: list[Any] = []
        j_shell_segments: list[Any] = []
        global_weight_segments: list[Any] = []
        global_weight_radius_segments: list[Any] = []
        constant_centers: list[float] = []
        constant_radii: list[float] = []
        device = None

        for segment in segments:
            (
                b_fiber,
                b_radius,
                a_shell,
                d_response,
                d_radius,
                j_shell,
                _packet_local_weights,
                _shell_fibers_tensor,
            ) = segment.compact_tensors()
            if device is None:
                device = b_fiber.device
            elif b_fiber.device != device:
                _refuse(
                    V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                    "segmented source packets span multiple devices",
                )
            segment_receipt = segment.source_receipt
            unique_tensor = torch.tensor(unique, dtype=torch.int64, device=device)
            b_touched_segments.append(
                torch.index_select(b_fiber, 1, unique_tensor).contiguous()
            )
            b_radius_segments.append(
                torch.index_select(b_radius, 1, unique_tensor).contiguous()
            )
            a_compact = torch.zeros(
                (segment_receipt.context_count, len(unique), shell),
                dtype=torch.float64,
                device=device,
            )
            for shell_offset, fiber_offset in enumerate(inverse):
                a_compact[:, fiber_offset, shell_offset] = a_shell[:, shell_offset]
            a_compact_segments.append(a_compact)
            d_response_segments.append(d_response.contiguous())
            d_radius_segments.append(d_radius.contiguous())
            j_shell_segments.append(j_shell.contiguous())
            exact_weights = tuple(
                stratum_masses[stratum] / cumulative_stratum_counts[stratum]
                for stratum in segment.logical_stratum_ids
            )
            centers_and_radii = tuple(
                _fraction_binary64_center_radius(value) for value in exact_weights
            )
            weights = torch.tensor(
                tuple(value[0] for value in centers_and_radii),
                dtype=torch.float64,
                device=device,
            )
            weight_radii = torch.tensor(
                tuple(value[1] for value in centers_and_radii),
                dtype=torch.float64,
                device=device,
            )
            global_weight_segments.append(weights)
            global_weight_radius_segments.append(weight_radii)

            untouched_mask = torch.ones(
                (aggregate.fiber_count,), dtype=torch.bool, device=device
            )
            untouched_mask[unique_tensor] = False
            b_untouched = b_fiber[:, untouched_mask]
            if b_untouched.numel():
                center_terms = weights.unsqueeze(1) * torch.abs(b_untouched)
                center_sum, center_sum_upper = _outward_nonnegative_tensor_sum(
                    center_terms,
                    operations_per_term=4,
                )
                del center_terms, b_untouched
                radius_untouched = b_radius[:, untouched_mask]
                radius_terms = weights.unsqueeze(
                    1
                ) * radius_untouched + weight_radii.unsqueeze(1) * (
                    torch.abs(b_fiber[:, untouched_mask]) + radius_untouched
                )
                _radius_sum, radius_sum_upper = _outward_nonnegative_tensor_sum(
                    radius_terms,
                    operations_per_term=4,
                )
                del radius_terms, radius_untouched
                center = center_sum / float(aggregate.cut_count)
                center_upper = _up(center_sum_upper / float(aggregate.cut_count))
                radius = _up(
                    radius_sum_upper / float(aggregate.cut_count)
                    + max(0.0, center_upper - center)
                )
            else:
                center = 0.0
                radius = 0.0
            constant_centers.append(center)
            constant_radii.append(radius)

        if device is None:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
                "segmented source unexpectedly contained no packet",
            )
        center_tensor = torch.tensor(
            constant_centers, dtype=torch.float64, device=device
        )
        constant_center, constant_center_upper = _outward_nonnegative_tensor_sum(
            center_tensor,
            operations_per_term=2,
        )
        radius_tensor = torch.tensor(constant_radii, dtype=torch.float64, device=device)
        _radius_center, constant_radius_upper = _outward_nonnegative_tensor_sum(
            radius_tensor,
            operations_per_term=2,
        )
        constant_radius = _up(
            constant_radius_upper + max(0.0, constant_center_upper - constant_center)
        )
        b_touched = torch.cat(b_touched_segments, dim=0).contiguous()
        b_radius_touched = torch.cat(b_radius_segments, dim=0).contiguous()
        a_compact = torch.cat(a_compact_segments, dim=0).contiguous()
        d_response = torch.cat(d_response_segments, dim=0).contiguous()
        d_radius = torch.cat(d_radius_segments, dim=0).contiguous()
        j_shell = torch.cat(j_shell_segments, dim=0).contiguous()
        weights = torch.cat(global_weight_segments, dim=0).contiguous()
        weight_radii = torch.cat(global_weight_radius_segments, dim=0).contiguous()
        contexts = aggregate.context_count
        if any(
            value.shape[0] != contexts
            for value in (
                b_touched,
                b_radius_touched,
                a_compact,
                d_response,
                d_radius,
                j_shell,
                weights,
                weight_radii,
            )
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "segmented compact tensors do not share the aggregate context axis",
            )

        compaction_receipt = V1SignedWorkVariableShellPhaseBSourceCompactionReceipt(
            source_receipt_hash=aggregate.receipt_hash,
            source_replay_hash=aggregate.source_replay_hash,
            compact_source_tensor_lease_hash=lease.lease_hash,
            context_count=contexts,
            full_native_fiber_count=aggregate.fiber_count,
            cut_count=aggregate.cut_count,
            shell_size=shell,
            unique_shell_native_fiber_indices=unique,
            shell_to_unique_fiber_offsets=inverse,
            untouched_native_fiber_count=aggregate.fiber_count - len(unique),
            variable_affine_row_count=contexts * len(unique),
            untouched_F_constant_lower=max(
                0.0, _down(constant_center - constant_radius)
            ),
            untouched_F_constant_center=constant_center,
            untouched_F_constant_upper=_up(constant_center + constant_radius),
            segmented_proposal_fit_source_only=True,
            exact_d1_join_required=True,
            _issuer_token=_SOLVER_RECEIPT_TOKEN,
        )
        return V1SignedWorkVariableShellPhaseBCompactSourceHandle(
            source_handle=source,
            source_lease=lease,
            receipt=compaction_receipt,
            b_touched=b_touched,
            b_radius_touched=b_radius_touched,
            a_compact=a_compact,
            d_response=d_response,
            d_radius=d_radius,
            j_shell=j_shell,
            weights=weights,
            weight_radii=weight_radii,
        )
    except Exception:
        lease.close()
        raise


def _compact_v1_signed_work_full_population_component(
    *,
    compact_tensors: tuple[object, ...],
    unique_shell_fibers: tuple[int, ...],
    shell_to_unique_offsets: tuple[int, ...],
    context_weights: Any,
    context_weight_radii: Any,
    full_native_fiber_count: int,
    cut_count: int,
    shell_size: int,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, float, float]:
    """Compact one retained component under externally registered weights.

    The source tensor ABI keeps the per-context measure in a separate vector,
    so the D1 and sampled-residual coordinates can be reweighted without
    replaying or rescaling any route row.  This helper deliberately ignores
    the component-local transport weights and applies only the exact
    full-population centers and radii supplied by the compositor receipt.
    """

    if torch is None:  # pragma: no cover - import availability is environment-bound.
        raise RuntimeError("torch is required for compact source evaluation")
    if len(compact_tensors) != 8:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            "one full-population component changed the compact tensor ABI",
        )
    (
        b_fiber,
        b_radius,
        a_shell,
        d_response,
        d_radius,
        j_shell,
        _component_local_weights,
        _shell_fibers_tensor,
    ) = compact_tensors
    contexts = int(b_fiber.shape[0])
    device = b_fiber.device
    if (
        tuple(b_fiber.shape) != (contexts, full_native_fiber_count)
        or tuple(b_radius.shape) != tuple(b_fiber.shape)
        or tuple(a_shell.shape) != (contexts, shell_size)
        or tuple(d_response.shape) != (contexts,)
        or tuple(d_radius.shape) != (contexts,)
        or tuple(j_shell.shape) != (contexts, shell_size)
        or tuple(context_weights.shape) != (contexts,)
        or tuple(context_weight_radii.shape) != (contexts,)
        or any(
            tensor.dtype != torch.float64 or tensor.device != device
            for tensor in (
                b_fiber,
                b_radius,
                a_shell,
                d_response,
                d_radius,
                j_shell,
                context_weights,
                context_weight_radii,
            )
        )
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            "one full-population component does not share the compact FP64 axes",
        )
    unique_tensor = torch.tensor(
        unique_shell_fibers,
        dtype=torch.int64,
        device=device,
    )
    b_touched = torch.index_select(b_fiber, 1, unique_tensor).contiguous()
    b_radius_touched = torch.index_select(b_radius, 1, unique_tensor).contiguous()
    a_compact = torch.zeros(
        (contexts, len(unique_shell_fibers), shell_size),
        dtype=torch.float64,
        device=device,
    )
    for shell_offset, fiber_offset in enumerate(shell_to_unique_offsets):
        a_compact[:, fiber_offset, shell_offset] = a_shell[:, shell_offset]

    untouched_mask = torch.ones(
        (full_native_fiber_count,),
        dtype=torch.bool,
        device=device,
    )
    untouched_mask[unique_tensor] = False
    b_untouched = b_fiber[:, untouched_mask]
    if b_untouched.numel():
        radius_untouched = b_radius[:, untouched_mask]
        absolute_b_untouched = torch.abs(b_untouched)
        center_terms = context_weights.unsqueeze(1) * absolute_b_untouched
        center_sum, center_sum_upper = _outward_nonnegative_tensor_sum(
            center_terms,
            operations_per_term=4,
        )
        radius_terms = context_weights.unsqueeze(1) * radius_untouched + (
            context_weight_radii.unsqueeze(1)
            * (absolute_b_untouched + radius_untouched)
        )
        _radius_sum, radius_sum_upper = _outward_nonnegative_tensor_sum(
            radius_terms,
            operations_per_term=7,
        )
        constant_center = center_sum / float(cut_count)
        constant_center_upper = _up(center_sum_upper / float(cut_count))
        constant_radius = _up(
            radius_sum_upper / float(cut_count)
            + max(0.0, constant_center_upper - constant_center)
        )
    else:
        constant_center = 0.0
        constant_radius = 0.0
    return (
        b_touched,
        b_radius_touched,
        a_compact,
        d_response.contiguous(),
        d_radius.contiguous(),
        j_shell.contiguous(),
        context_weights.contiguous(),
        context_weight_radii.contiguous(),
        constant_center,
        constant_radius,
    )


def _build_v1_signed_work_variable_shell_phase_b_compact_source_full_population(
    *,
    branch_source: V1SignedWorkVariableShellBranchActionCellSource,
) -> V1SignedWorkVariableShellPhaseBCompactSourceHandle:
    """Join exact D1 and sampled-residual measures in bounded compact form."""

    source = branch_source.assert_live_source()
    if type(source) is not V1SampledFullPopulationCompactSourceHandle:
        raise TypeError(
            "full-population compaction requires the exact composite source"
        )
    aggregate = source.receipt
    purpose = content_hash(
        {
            "branch_registration_hash": branch_source.registration.registration_hash,
            "full_population_source_receipt_hash": aggregate.receipt_hash,
            "operation": "exact_D1_plus_sampled_residual_touched_fiber_compaction",
        },
        prefix=_PREFIX + "full-population-source-compaction-purpose-",
    )
    lease = source.borrow_components(borrow_purpose_hash=purpose)
    try:
        d1_tensors = lease.d1_compact_tensors()
        residual_segments = lease.residual_compact_segments()
        if (
            len(residual_segments) != aggregate.segment_count
            or sum(
                segment.source_receipt.context_count for segment in residual_segments
            )
            != aggregate.residual_context_count
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "full-population residual segments do not close to the frozen prefix",
            )

        unique = tuple(sorted(set(aggregate.shell_native_fiber_indices)))
        unique_offset = {value: index for index, value in enumerate(unique)}
        inverse = tuple(
            unique_offset[value] for value in aggregate.shell_native_fiber_indices
        )
        shell = len(aggregate.shell_indices)
        d1_device = d1_tensors[0].device
        d1_weights = torch.full(
            (aggregate.d1_context_count,),
            aggregate.d1_context_weight_binary64_center,
            dtype=torch.float64,
            device=d1_device,
        )
        d1_weight_radii = torch.full(
            (aggregate.d1_context_count,),
            aggregate.d1_context_weight_binary64_radius,
            dtype=torch.float64,
            device=d1_device,
        )
        components = [
            _compact_v1_signed_work_full_population_component(
                compact_tensors=d1_tensors,
                unique_shell_fibers=unique,
                shell_to_unique_offsets=inverse,
                context_weights=d1_weights,
                context_weight_radii=d1_weight_radii,
                full_native_fiber_count=aggregate.fiber_count,
                cut_count=aggregate.cut_count,
                shell_size=shell,
            )
        ]
        stratum_weight = {
            stratum: (
                aggregate.residual_context_weight_binary64_centers[index],
                aggregate.residual_context_weight_binary64_radii[index],
            )
            for index, stratum in enumerate(aggregate.canonical_stratum_order)
        }
        observed_stratum_counts = {
            stratum: 0 for stratum in aggregate.canonical_stratum_order
        }
        for segment in residual_segments:
            tensors = segment.compact_tensors()
            if tensors[0].device != d1_device:
                _refuse(
                    V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                    "D1 and sampled-residual compact tensors span multiple devices",
                )
            try:
                centers_and_radii = tuple(
                    stratum_weight[stratum] for stratum in segment.logical_stratum_ids
                )
            except KeyError:
                _refuse(
                    V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                    "sampled-residual segment contains an unknown logical stratum",
                )
            for stratum in segment.logical_stratum_ids:
                observed_stratum_counts[stratum] += 1
            weights = torch.tensor(
                tuple(value[0] for value in centers_and_radii),
                dtype=torch.float64,
                device=d1_device,
            )
            weight_radii = torch.tensor(
                tuple(value[1] for value in centers_and_radii),
                dtype=torch.float64,
                device=d1_device,
            )
            components.append(
                _compact_v1_signed_work_full_population_component(
                    compact_tensors=tensors,
                    unique_shell_fibers=unique,
                    shell_to_unique_offsets=inverse,
                    context_weights=weights,
                    context_weight_radii=weight_radii,
                    full_native_fiber_count=aggregate.fiber_count,
                    cut_count=aggregate.cut_count,
                    shell_size=shell,
                )
            )
        if (
            tuple(
                observed_stratum_counts[stratum]
                for stratum in aggregate.canonical_stratum_order
            )
            != aggregate.final_dw2_stratum_counts
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "sampled-residual logical-stratum counts changed after composition",
            )

        tensor_axes = tuple(
            torch.cat(
                tuple(component[index] for component in components), dim=0
            ).contiguous()
            for index in range(8)
        )
        contexts = aggregate.d1_context_count + aggregate.residual_context_count
        if any(value.shape[0] != contexts for value in tensor_axes):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "full-population compact tensors do not share the joined context axis",
            )
        constant_center_tensor = torch.tensor(
            tuple(component[8] for component in components),
            dtype=torch.float64,
            device=d1_device,
        )
        constant_center, constant_center_upper = _outward_nonnegative_tensor_sum(
            constant_center_tensor,
            operations_per_term=2,
        )
        constant_radius_tensor = torch.tensor(
            tuple(component[9] for component in components),
            dtype=torch.float64,
            device=d1_device,
        )
        _constant_radius, constant_radius_upper = _outward_nonnegative_tensor_sum(
            constant_radius_tensor,
            operations_per_term=2,
        )
        constant_radius = _up(
            constant_radius_upper + max(0.0, constant_center_upper - constant_center)
        )
        compaction_receipt = V1SignedWorkVariableShellPhaseBSourceCompactionReceipt(
            source_receipt_hash=aggregate.receipt_hash,
            source_replay_hash=aggregate.source_replay_hash,
            compact_source_tensor_lease_hash=lease.lease_hash,
            context_count=contexts,
            full_native_fiber_count=aggregate.fiber_count,
            cut_count=aggregate.cut_count,
            shell_size=shell,
            unique_shell_native_fiber_indices=unique,
            shell_to_unique_fiber_offsets=inverse,
            untouched_native_fiber_count=aggregate.fiber_count - len(unique),
            variable_affine_row_count=contexts * len(unique),
            untouched_F_constant_lower=max(
                0.0, _down(constant_center - constant_radius)
            ),
            untouched_F_constant_center=constant_center,
            untouched_F_constant_upper=_up(constant_center + constant_radius),
            _issuer_token=_SOLVER_RECEIPT_TOKEN,
        )
        return V1SignedWorkVariableShellPhaseBCompactSourceHandle(
            source_handle=source,
            source_lease=lease,
            receipt=compaction_receipt,
            b_touched=tensor_axes[0],
            b_radius_touched=tensor_axes[1],
            a_compact=tensor_axes[2],
            d_response=tensor_axes[3],
            d_radius=tensor_axes[4],
            j_shell=tensor_axes[5],
            weights=tensor_axes[6],
            weight_radii=tensor_axes[7],
        )
    except Exception:
        lease.close()
        raise


def build_v1_signed_work_variable_shell_phase_b_compact_source(
    *,
    branch_source: V1SignedWorkVariableShellBranchActionCellSource,
) -> V1SignedWorkVariableShellPhaseBCompactSourceHandle:
    """Build the exact touched-fiber quotient for one-shot or segmented D_W2."""

    if type(branch_source) is not V1SignedWorkVariableShellBranchActionCellSource:
        raise TypeError("source compaction requires an exact live branch source")
    source = branch_source.assert_live_source()
    if type(source) is V1SignedWorkVariableShellSourceDeviceHandle:
        return _build_v1_signed_work_variable_shell_phase_b_compact_source_one_shot(
            branch_source=branch_source
        )
    if type(source) is V1SignedWorkVariableShellSourceAppendSegmentedHandle:
        return _build_v1_signed_work_variable_shell_phase_b_compact_source_segmented(
            branch_source=branch_source
        )
    if type(source) is V1SampledFullPopulationCompactSourceHandle:
        return (
            _build_v1_signed_work_variable_shell_phase_b_compact_source_full_population(
                branch_source=branch_source
            )
        )
    raise TypeError("branch source returned an unsupported compact-source handle")


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellActionCellBlockCapacity:
    shell_size: int
    shell_offsets: tuple[int, ...]
    remaining_capacity: int
    block_id: str
    schema: str = ACTION_BLOCK_CAPACITY_SCHEMA

    def __post_init__(self) -> None:
        shell = _positive_int(self.shell_size, name="shell size")
        object.__setattr__(self, "shell_size", shell)
        offsets = _offset_axis(
            self.shell_offsets,
            shell_size=shell,
            name="block shell offsets",
            allow_empty=False,
        )
        capacity = _nonnegative_int(
            self.remaining_capacity,
            name="remaining block capacity",
        )
        if capacity > len(offsets):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
                "remaining block capacity exceeds its shell membership",
            )
        object.__setattr__(self, "shell_offsets", offsets)
        object.__setattr__(self, "remaining_capacity", capacity)
        object.__setattr__(self, "block_id", _identity(self.block_id, name="block id"))
        if self.schema != ACTION_BLOCK_CAPACITY_SCHEMA:
            raise ValueError("unsupported block-capacity schema")

    @property
    def capacity_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "block-capacity-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellPhaseBAliasProblemSpecification:
    """Complete registered Phase-B constraints for one branch/resource alias."""

    registry_receipt_hash: str
    branch: V1SignedWorkVariableShellQBranch
    resource_alias_id: str
    phase_a_invariance_receipt_hash: str
    shell_resource_coefficients: tuple[float, ...]
    literal_j_response_radius: float
    block_capacities: tuple[V1SignedWorkVariableShellActionCellBlockCapacity, ...]
    native_fiber_F_is_minimization_objective: bool = True
    F_feasibility_budget_present: bool = False
    schema: str = ACTION_ALIAS_PROBLEM_SPEC_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "registry_receipt_hash",
            _identity(self.registry_receipt_hash, name="registry receipt hash"),
        )
        if type(self.branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("alias problem branch must be exact")
        for name in ("resource_alias_id", "phase_a_invariance_receipt_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        resources = tuple(
            _finite_nonnegative(value, name="shell resource coefficient")
            for value in self.shell_resource_coefficients
        )
        if not 1 <= len(resources) <= _MAXIMUM_SHELL_SIZE:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "alias problem resource vector is outside B<=8",
            )
        object.__setattr__(self, "shell_resource_coefficients", resources)
        object.__setattr__(
            self,
            "literal_j_response_radius",
            _finite_nonnegative(
                self.literal_j_response_radius,
                name="literal-J response radius",
            ),
        )
        capacities = tuple(self.block_capacities)
        flattened = tuple(
            index for value in capacities for index in value.shell_offsets
        )
        if (
            not capacities
            or any(
                type(value) is not V1SignedWorkVariableShellActionCellBlockCapacity
                for value in capacities
            )
            or any(value.shell_size != len(resources) for value in capacities)
            or tuple(sorted(flattened)) != tuple(range(len(resources)))
            or len(set(flattened)) != len(resources)
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "alias problem block capacities must partition the shell",
            )
        object.__setattr__(self, "block_capacities", capacities)
        if (
            self.native_fiber_F_is_minimization_objective is not True
            or self.F_feasibility_budget_present is not False
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "alias problem changed the lexicographic Phase-B objective",
            )
        if self.schema != ACTION_ALIAS_PROBLEM_SPEC_SCHEMA:
            raise ValueError("unsupported Phase-B alias problem schema")

    @property
    def problem_specification_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "alias-problem-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellRestrictedConvexCellRequest:
    """One Phase-B cell problem; F is the minimization objective."""

    registry_receipt_hash: str
    source_receipt_hash: str
    source_replay_hash: str
    action_cell: V1SignedWorkVariableShellActionCellIdentity
    registered_phase_a_resource_floor: float
    shell_resource_coefficients: tuple[float, ...]
    literal_j_response_radius: float
    block_capacities: tuple[V1SignedWorkVariableShellActionCellBlockCapacity, ...]
    native_fiber_F_is_objective: bool = True
    F_feasibility_budget_present: bool = False
    phase_a_coordinate_frozen: bool = True
    numerical_dtype: str = ACTION_CELL_NUMERICAL_DTYPE
    solver_backend: str = ACTION_CELL_SOLVER_BACKEND
    problem_rule: str = ACTION_RESTRICTED_PROBLEM_RULE
    source_handle_required: bool = True
    global_solver_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = ACTION_SEPARATION_AUTHORITY_SCOPE
    schema: str = ACTION_RESTRICTED_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "registry_receipt_hash",
            "source_receipt_hash",
            "source_replay_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.action_cell) is not V1SignedWorkVariableShellActionCellIdentity:
            raise TypeError("restricted request requires an exact action-cell identity")
        object.__setattr__(
            self,
            "registered_phase_a_resource_floor",
            _finite_nonnegative(
                self.registered_phase_a_resource_floor,
                name="registered Phase-A resource floor",
            ),
        )
        resources = tuple(
            _finite_nonnegative(value, name="shell resource coefficient")
            for value in self.shell_resource_coefficients
        )
        shell = max(self.action_cell.selected_shell_offsets, default=-1) + 1
        capacities = tuple(self.block_capacities)
        if not capacities or any(
            type(value) is not V1SignedWorkVariableShellActionCellBlockCapacity
            for value in capacities
        ):
            raise TypeError("restricted request requires exact block capacities")
        if capacities:
            shell = capacities[0].shell_size
        if len(resources) != shell:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "shell resources do not align with the complete shell",
            )
        object.__setattr__(self, "shell_resource_coefficients", resources)
        object.__setattr__(
            self,
            "literal_j_response_radius",
            _finite_nonnegative(
                self.literal_j_response_radius,
                name="literal-J response radius",
            ),
        )
        flattened = tuple(
            offset for value in capacities for offset in value.shell_offsets
        )
        if (
            any(value.shell_size != shell for value in capacities)
            or tuple(sorted(flattened)) != tuple(range(shell))
            or len(set(flattened)) != shell
            or len({value.block_id for value in capacities}) != len(capacities)
            or any(
                offset >= shell for offset in self.action_cell.selected_shell_offsets
            )
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "block capacities must partition the complete shell exactly once",
            )
        object.__setattr__(self, "block_capacities", capacities)
        required_true = (
            self.native_fiber_F_is_objective,
            self.phase_a_coordinate_frozen,
            self.source_handle_required,
        )
        if (
            any(value is not True for value in required_true)
            or self.F_feasibility_budget_present is not False
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "Phase B must minimize F at a frozen Phase-A resource coordinate",
            )
        _closed_authority(
            (
                self.global_solver_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="restricted Phase-B request",
        )
        if (
            self.numerical_dtype != ACTION_CELL_NUMERICAL_DTYPE
            or self.solver_backend != ACTION_CELL_SOLVER_BACKEND
            or self.problem_rule != ACTION_RESTRICTED_PROBLEM_RULE
            or self.authority_scope != ACTION_SEPARATION_AUTHORITY_SCOPE
            or self.schema != ACTION_RESTRICTED_REQUEST_SCHEMA
        ):
            raise ValueError("unsupported restricted Phase-B request semantics")

    @property
    def shell_size(self) -> int:
        return len(self.shell_resource_coefficients)

    @property
    def request_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "restricted-request-")


def build_v1_signed_work_variable_shell_restricted_cell_request(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    branch: V1SignedWorkVariableShellQBranch,
    resource_alias_id: str,
    action_cell_ordinal: int,
    shell_resource_coefficients: Sequence[float],
    literal_j_response_radius: float,
    block_capacities: Sequence[V1SignedWorkVariableShellActionCellBlockCapacity],
) -> V1SignedWorkVariableShellRestrictedConvexCellRequest:
    if type(registry) is not V1SignedWorkVariableShellActionCellRegistry:
        raise TypeError("restricted request requires the exact live registry")
    source = registry.branch_source(branch)
    alias = source.resource_alias(resource_alias_id)
    action_cell = source.action_cell(
        registry_receipt_hash=registry.receipt.receipt_hash,
        resource_alias_id=resource_alias_id,
        action_cell_ordinal=action_cell_ordinal,
    )
    receipt = source.assert_live_source().receipt
    return V1SignedWorkVariableShellRestrictedConvexCellRequest(
        registry_receipt_hash=registry.receipt.receipt_hash,
        source_receipt_hash=receipt.receipt_hash,
        source_replay_hash=receipt.source_replay_hash,
        action_cell=action_cell,
        registered_phase_a_resource_floor=alias.registered_shell_resource_floor,
        shell_resource_coefficients=tuple(shell_resource_coefficients),
        literal_j_response_radius=literal_j_response_radius,
        block_capacities=tuple(block_capacities),
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellPhaseBDualLowerBoundContract:
    request_hash: str
    source_receipt_hash: str
    source_replay_hash: str
    shell_size: int
    native_fiber_affine_row_count: int
    literal_j_response_row_count: int
    block_constraint_count: int
    subset_order_constraint_count: int
    native_fiber_affine_row_axis_hash: str
    literal_j_response_row_axis_hash: str
    compact_live_tensor_lease_required: bool = True
    polytope_vertex_enumeration_required: bool = False
    projected_dual_feasibility_required: bool = True
    outward_binary64_lower_enclosure_required: bool = True
    dual_lower_bound_rule: str = ACTION_PHASE_B_DUAL_LOWER_BOUND_RULE
    dual_projection_rule: str = ACTION_PHASE_B_DUAL_PROJECTION_RULE
    global_solver_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    schema: str = ACTION_DUAL_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "request_hash",
            "source_receipt_hash",
            "source_replay_hash",
            "native_fiber_affine_row_axis_hash",
            "literal_j_response_row_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "shell_size",
            "native_fiber_affine_row_count",
            "literal_j_response_row_count",
            "block_constraint_count",
            "subset_order_constraint_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name.replace("_", " ")),
            )
        if not 1 <= self.shell_size <= _MAXIMUM_SHELL_SIZE:
            raise ValueError("dual contract shell size is outside B<=8")
        if (
            min(self.native_fiber_affine_row_count, self.literal_j_response_row_count)
            <= 0
        ):
            raise ValueError("dual contract source row axes must be nonempty")
        required_true = (
            self.compact_live_tensor_lease_required,
            self.projected_dual_feasibility_required,
            self.outward_binary64_lower_enclosure_required,
        )
        if (
            any(value is not True for value in required_true)
            or self.polytope_vertex_enumeration_required is not False
        ):
            raise ValueError("dual contract projection/enclosure semantics changed")
        _closed_authority(
            (
                self.global_solver_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
            ),
            owner="Phase-B dual contract",
        )
        if (
            self.dual_lower_bound_rule != ACTION_PHASE_B_DUAL_LOWER_BOUND_RULE
            or self.dual_projection_rule != ACTION_PHASE_B_DUAL_PROJECTION_RULE
            or self.schema != ACTION_DUAL_CONTRACT_SCHEMA
        ):
            raise ValueError("unsupported Phase-B dual contract")

    @property
    def contract_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "dual-contract-")


def build_v1_signed_work_variable_shell_phase_b_dual_lower_bound_contract(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    request: V1SignedWorkVariableShellRestrictedConvexCellRequest,
) -> V1SignedWorkVariableShellPhaseBDualLowerBoundContract:
    if type(registry) is not V1SignedWorkVariableShellActionCellRegistry:
        raise TypeError("dual contract requires the exact live registry")
    if type(request) is not V1SignedWorkVariableShellRestrictedConvexCellRequest:
        raise TypeError("dual contract requires the exact restricted request")
    if request.registry_receipt_hash != registry.receipt.receipt_hash:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
            "dual contract request belongs to another registry",
        )
    source = registry.branch_source(request.action_cell.branch)
    receipt = source.assert_live_source().receipt
    if receipt.receipt_hash != request.source_receipt_hash:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
            "dual contract request belongs to another source",
        )
    order_count = request.action_cell.shell_prefix_count * (
        request.shell_size - request.action_cell.shell_prefix_count
    )
    return V1SignedWorkVariableShellPhaseBDualLowerBoundContract(
        request_hash=request.request_hash,
        source_receipt_hash=receipt.receipt_hash,
        source_replay_hash=receipt.source_replay_hash,
        shell_size=request.shell_size,
        native_fiber_affine_row_count=(
            receipt.context_count * len(set(receipt.shell_native_fiber_indices))
        ),
        literal_j_response_row_count=receipt.context_count,
        block_constraint_count=len(request.block_capacities),
        subset_order_constraint_count=order_count,
        native_fiber_affine_row_axis_hash=content_hash(
            {
                "context_axis_hash": receipt.context_axis_hash,
                "native_fiber_axis_receipt_hash": receipt.native_fiber_axis_receipt_hash,
                "cut_count": receipt.cut_count,
                "unique_shell_native_fibers": tuple(
                    sorted(set(receipt.shell_native_fiber_indices))
                ),
            },
            prefix=_PREFIX + "native-fiber-affine-row-axis-",
        ),
        literal_j_response_row_axis_hash=content_hash(
            {
                "context_axis_hash": receipt.context_axis_hash,
                "literal_j_stream_source_hash": receipt.literal_j_stream_source_hash,
            },
            prefix=_PREFIX + "literal-j-response-row-axis-",
        ),
    )


def _fraction_vector(
    values: Sequence[Real],
    *,
    count: int,
    name: str,
) -> tuple[Fraction, ...]:
    if isinstance(values, (str, bytes)) or len(values) != count:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            f"{name} does not align with its registered axis",
        )
    return tuple(_fraction(value, name=f"{name} item") for value in values)


def _fraction_matrix(
    values: Sequence[Sequence[Real]],
    *,
    row_count: int,
    column_count: int,
    name: str,
) -> tuple[tuple[Fraction, ...], ...]:
    if isinstance(values, (str, bytes)) or len(values) != row_count:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            f"{name} row axis does not align",
        )
    return tuple(
        _fraction_vector(row, count=column_count, name=f"{name} row") for row in values
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellPhaseBDualLowerBoundReference:
    contract_hash: str
    request_hash: str
    projected_dual_witness_hash: str
    projected_y_f_count: int
    free_y_q_count: int
    nonnegative_inequality_multiplier_count: int
    exact_linear_term_numerator: int
    exact_linear_term_denominator: int
    weighted_q_norm_upper: float
    operator_frobenius_upper_bound: float
    projected_dual_mass_l1_upper_bound: float
    dual_value_center: float
    dual_lower_bound_outward: float
    numerical_radius_upper: float
    exact_when_y_q_zero: bool
    dual_witness_replay_complete: bool = True
    projected_dual_feasible: bool = True
    reference_test_only: bool = True
    global_solver_authority: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    dual_lower_bound_rule: str = ACTION_PHASE_B_DUAL_LOWER_BOUND_RULE
    dual_projection_rule: str = ACTION_PHASE_B_DUAL_PROJECTION_RULE
    schema: str = ACTION_DUAL_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        for name in ("contract_hash", "request_hash", "projected_dual_witness_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "projected_y_f_count",
            "free_y_q_count",
            "nonnegative_inequality_multiplier_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name.replace("_", " ")),
            )
        if isinstance(self.exact_linear_term_numerator, bool) or not isinstance(
            self.exact_linear_term_numerator, Integral
        ):
            raise TypeError("exact linear numerator must be integer")
        denominator = _positive_int(
            self.exact_linear_term_denominator,
            name="exact linear denominator",
        )
        object.__setattr__(
            self, "exact_linear_term_numerator", int(self.exact_linear_term_numerator)
        )
        object.__setattr__(self, "exact_linear_term_denominator", denominator)
        norm = _finite_nonnegative(
            self.weighted_q_norm_upper, name="weighted Q norm upper"
        )
        frobenius = _finite_nonnegative(
            self.operator_frobenius_upper_bound,
            name="operator Frobenius upper bound",
        )
        dual_mass = _finite_nonnegative(
            self.projected_dual_mass_l1_upper_bound,
            name="projected dual mass L1 upper bound",
        )
        center = _finite(self.dual_value_center, name="dual value center")
        lower = _finite(self.dual_lower_bound_outward, name="dual lower bound")
        radius = _finite_nonnegative(
            self.numerical_radius_upper, name="dual numerical radius"
        )
        if lower > center or radius < center - lower:
            raise ValueError("dual reference enclosure is not outward")
        object.__setattr__(self, "weighted_q_norm_upper", norm)
        object.__setattr__(self, "operator_frobenius_upper_bound", frobenius)
        object.__setattr__(self, "projected_dual_mass_l1_upper_bound", dual_mass)
        object.__setattr__(self, "dual_value_center", center)
        object.__setattr__(self, "dual_lower_bound_outward", lower)
        object.__setattr__(self, "numerical_radius_upper", radius)
        if (
            type(self.exact_when_y_q_zero) is not bool
            or self.dual_witness_replay_complete is not True
            or self.projected_dual_feasible is not True
            or self.reference_test_only is not True
        ):
            raise TypeError("dual reference mode/feasibility markers changed")
        _closed_authority(
            (
                self.global_solver_authority,
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="Phase-B dual reference",
        )
        if (
            self.dual_lower_bound_rule != ACTION_PHASE_B_DUAL_LOWER_BOUND_RULE
            or self.dual_projection_rule != ACTION_PHASE_B_DUAL_PROJECTION_RULE
            or self.schema != ACTION_DUAL_REFERENCE_SCHEMA
        ):
            raise ValueError("unsupported Phase-B dual reference semantics")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "dual-reference-")


def evaluate_v1_signed_work_variable_shell_phase_b_dual_lower_bound_reference(
    *,
    contract: V1SignedWorkVariableShellPhaseBDualLowerBoundContract,
    request: V1SignedWorkVariableShellRestrictedConvexCellRequest,
    source_abs_affine_rows: Sequence[Sequence[Real]],
    source_abs_offsets: Sequence[Real],
    source_abs_weights: Sequence[Real],
    weighted_literal_j_rows: Sequence[Sequence[Real]],
    weighted_literal_j_offsets: Sequence[Real],
    y_f: Sequence[Real],
    y_q: Sequence[Real],
    resource_multiplier: Real,
    block_multipliers: Sequence[Real],
    subset_order_multipliers: Sequence[Real],
) -> V1SignedWorkVariableShellPhaseBDualLowerBoundReference:
    """Evaluate the projected compact dual formula on tiny CPU inputs.

    The evaluator is a mathematical/reference seam, not the production PDHG
    solver.  All binary64 inputs are interpreted exactly as rationals.  Only
    ``sqrt(sum(y_q**2))`` is irrational; its penalty is rounded upward at 160
    decimal digits and the final lower bound is rounded toward ``-inf``.
    """

    if type(contract) is not V1SignedWorkVariableShellPhaseBDualLowerBoundContract:
        raise TypeError("dual reference requires the exact dual contract")
    if type(request) is not V1SignedWorkVariableShellRestrictedConvexCellRequest:
        raise TypeError("dual reference requires the exact restricted request")
    if contract.request_hash != request.request_hash:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
            "dual contract and request identities differ",
        )
    shell = contract.shell_size
    f_rows = contract.native_fiber_affine_row_count
    q_rows = contract.literal_j_response_row_count
    block_count = contract.block_constraint_count
    order_count = contract.subset_order_constraint_count
    a_matrix = _fraction_matrix(
        source_abs_affine_rows,
        row_count=f_rows,
        column_count=shell,
        name="source absolute-affine matrix",
    )
    b_vector = _fraction_vector(
        source_abs_offsets,
        count=f_rows,
        name="source absolute-affine offset",
    )
    alpha = _fraction_vector(
        source_abs_weights,
        count=f_rows,
        name="source absolute weights",
    )
    if any(value < 0 for value in alpha):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
            "source absolute weights must be nonnegative",
        )
    j_matrix = _fraction_matrix(
        weighted_literal_j_rows,
        row_count=q_rows,
        column_count=shell,
        name="weighted literal-J matrix",
    )
    d_vector = _fraction_vector(
        weighted_literal_j_offsets,
        count=q_rows,
        name="weighted literal-J offset",
    )
    yf_input = _fraction_vector(y_f, count=f_rows, name="y_F")
    yq = _fraction_vector(y_q, count=q_rows, name="y_Q")
    yf = tuple(
        max(-bound, min(bound, value))
        for value, bound in zip(yf_input, alpha, strict=True)
    )
    lam = max(Fraction(0), _fraction(resource_multiplier, name="resource multiplier"))
    mu_input = _fraction_vector(
        block_multipliers,
        count=block_count,
        name="block multiplier",
    )
    nu_input = _fraction_vector(
        subset_order_multipliers,
        count=order_count,
        name="subset-order multiplier",
    )
    mu = tuple(max(Fraction(0), value) for value in mu_input)
    nu = tuple(max(Fraction(0), value) for value in nu_input)
    resource = tuple(
        Fraction.from_float(value) for value in request.shell_resource_coefficients
    )
    floor = Fraction.from_float(request.registered_phase_a_resource_floor)
    s = [Fraction(0) for _ in range(shell)]
    for row, multiplier in zip(a_matrix, yf, strict=True):
        for index, coefficient in enumerate(row):
            s[index] += coefficient * multiplier
    for row, multiplier in zip(j_matrix, yq, strict=True):
        for index, coefficient in enumerate(row):
            s[index] += coefficient * multiplier
    for index, coefficient in enumerate(resource):
        s[index] -= lam * coefficient
    for capacity, multiplier in zip(request.block_capacities, mu, strict=True):
        for index in capacity.shell_offsets:
            s[index] += multiplier
    selected = set(request.action_cell.selected_shell_offsets)
    order_pairs = tuple(
        (left, right)
        for left in request.action_cell.selected_shell_offsets
        for right in range(shell)
        if right not in selected
    )
    if len(order_pairs) != order_count:
        raise ValueError("dual order axis changed after contract registration")
    for (left, right), multiplier in zip(order_pairs, nu, strict=True):
        # O z <= 0 is z_right - z_left <= 0.
        s[left] -= multiplier
        s[right] += multiplier
    linear = sum(value * offset for value, offset in zip(yf, b_vector, strict=True))
    linear += sum(value * offset for value, offset in zip(yq, d_vector, strict=True))
    linear += lam * floor
    linear -= sum(
        multiplier * capacity.remaining_capacity
        for multiplier, capacity in zip(mu, request.block_capacities, strict=True)
    )
    linear += sum(min(Fraction(0), value) for value in s)
    norm_square = sum(value * value for value in yq)
    operator_square = Fraction(0)
    for rows in (a_matrix, j_matrix):
        for row in rows:
            operator_square += sum(value * value for value in row)
    operator_square += sum(value * value for value in resource)
    for capacity in request.block_capacities:
        operator_square += Fraction(len(capacity.shell_offsets), 1)
    operator_square += Fraction(2 * len(order_pairs), 1)
    dual_mass = (
        sum(abs(value) for value in yf)
        + sum(abs(value) for value in yq)
        + abs(lam)
        + sum(abs(value) for value in mu)
        + sum(abs(value) for value in nu)
    )
    radius = Fraction.from_float(request.literal_j_response_radius)
    with localcontext() as context:
        context.prec = 160
        context.rounding = ROUND_CEILING
        norm_square_upper = Decimal(norm_square.numerator) / Decimal(
            norm_square.denominator
        )
        norm_upper_decimal = context.sqrt(norm_square_upper)
        operator_square_upper = Decimal(operator_square.numerator) / Decimal(
            operator_square.denominator
        )
        operator_frobenius_upper_decimal = context.sqrt(operator_square_upper)
        penalty_upper = Decimal(radius.numerator) / Decimal(radius.denominator)
        penalty_upper *= norm_upper_decimal
        context.rounding = ROUND_FLOOR
        linear_lower = Decimal(linear.numerator) / Decimal(linear.denominator)
        lower_decimal = linear_lower - penalty_upper
    lower = math.nextafter(float(lower_decimal), -math.inf)
    norm_center = math.sqrt(float(norm_square))
    center = float(linear) - float(radius) * norm_center
    norm_upper = math.nextafter(float(norm_upper_decimal), math.inf)
    operator_frobenius_upper = math.nextafter(
        float(operator_frobenius_upper_decimal),
        math.inf,
    )
    dual_mass_upper = math.nextafter(float(dual_mass), math.inf)
    witness_hash = content_hash(
        {
            "request_hash": request.request_hash,
            "yf": tuple((value.numerator, value.denominator) for value in yf),
            "yq": tuple((value.numerator, value.denominator) for value in yq),
            "lambda": (lam.numerator, lam.denominator),
            "mu": tuple((value.numerator, value.denominator) for value in mu),
            "nu": tuple((value.numerator, value.denominator) for value in nu),
            "source_a": tuple(
                tuple((value.numerator, value.denominator) for value in row)
                for row in a_matrix
            ),
            "source_b": tuple(
                (value.numerator, value.denominator) for value in b_vector
            ),
            "source_alpha": tuple(
                (value.numerator, value.denominator) for value in alpha
            ),
            "source_j": tuple(
                tuple((value.numerator, value.denominator) for value in row)
                for row in j_matrix
            ),
            "source_d": tuple(
                (value.numerator, value.denominator) for value in d_vector
            ),
            "dual_lower_bound_outward": lower,
            "operator_frobenius_upper_bound": operator_frobenius_upper,
            "projected_dual_mass_l1_upper_bound": dual_mass_upper,
            "dual_witness_replay_complete": True,
        },
        prefix=_PREFIX + "projected-dual-witness-",
    )
    return V1SignedWorkVariableShellPhaseBDualLowerBoundReference(
        contract_hash=contract.contract_hash,
        request_hash=request.request_hash,
        projected_dual_witness_hash=witness_hash,
        projected_y_f_count=len(yf),
        free_y_q_count=len(yq),
        nonnegative_inequality_multiplier_count=1 + len(mu) + len(nu),
        exact_linear_term_numerator=linear.numerator,
        exact_linear_term_denominator=linear.denominator,
        weighted_q_norm_upper=norm_upper,
        operator_frobenius_upper_bound=operator_frobenius_upper,
        projected_dual_mass_l1_upper_bound=dual_mass_upper,
        dual_value_center=center,
        dual_lower_bound_outward=lower,
        numerical_radius_upper=math.nextafter(max(0.0, center - lower), math.inf),
        exact_when_y_q_zero=norm_square == 0,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellPhaseBPrimalReference:
    request_hash: str
    source_receipt_hash: str
    shell_gate_hash: str
    f_value_numerator: int
    f_value_denominator: int
    q_value_numerator: int
    q_value_denominator: int
    resource_value_numerator: int
    resource_value_denominator: int
    exact_binary64_sample_inputs: bool = True
    reference_test_only: bool = True
    globally_optimal_claim: bool = False
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    schema: str = ACTION_PRIMAL_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        for name in ("request_hash", "source_receipt_hash", "shell_gate_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "f_value_numerator",
            "q_value_numerator",
            "resource_value_numerator",
        ):
            if isinstance(getattr(self, name), bool) or not isinstance(
                getattr(self, name), Integral
            ):
                raise TypeError(f"{name} must be integer")
            object.__setattr__(self, name, int(getattr(self, name)))
        for name in (
            "f_value_denominator",
            "q_value_denominator",
            "resource_value_denominator",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name.replace("_", " ")),
            )
        if (
            self.exact_binary64_sample_inputs is not True
            or self.reference_test_only is not True
        ):
            raise ValueError("primal reference mode changed")
        _closed_authority(
            (
                self.globally_optimal_claim,
                self.action_invariance_authority,
                self.C_stream_open_authority,
            ),
            owner="Phase-B primal reference",
        )
        if self.schema != ACTION_PRIMAL_REFERENCE_SCHEMA:
            raise ValueError("unsupported Phase-B primal reference schema")

    @property
    def f_value(self) -> Fraction:
        return Fraction(self.f_value_numerator, self.f_value_denominator)

    @property
    def q_value(self) -> Fraction:
        return Fraction(self.q_value_numerator, self.q_value_denominator)


def evaluate_v1_signed_work_variable_shell_phase_b_primal_reference(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    request: V1SignedWorkVariableShellRestrictedConvexCellRequest,
    shell_gates: Sequence[Real],
) -> V1SignedWorkVariableShellPhaseBPrimalReference:
    """Exact feasibility/objective check for tiny CPU-reference sources."""

    if type(registry) is not V1SignedWorkVariableShellActionCellRegistry:
        raise TypeError("primal reference requires the exact live registry")
    if type(request) is not V1SignedWorkVariableShellRestrictedConvexCellRequest:
        raise TypeError("primal reference requires the exact restricted request")
    if request.registry_receipt_hash != registry.receipt.receipt_hash:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
            "primal request belongs to another registry",
        )
    source = registry.branch_source(request.action_cell.branch)
    handle = source.assert_live_source()
    if handle.receipt.receipt_hash != request.source_receipt_hash:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
            "primal request belongs to another source",
        )
    gates = _validated_gates(shell_gates, request.shell_size)
    cells = source.closed_cells(request.action_cell.resource_alias_id)
    cell = cells[request.action_cell.action_cell_ordinal]
    z = tuple(Fraction.from_float(value) for value in gates)
    resource = sum(
        Fraction.from_float(coefficient) * value
        for coefficient, value in zip(
            request.shell_resource_coefficients, z, strict=True
        )
    )
    floor = Fraction.from_float(request.registered_phase_a_resource_floor)
    blocks_feasible = all(
        sum(z[index] for index in capacity.shell_offsets) <= capacity.remaining_capacity
        for capacity in request.block_capacities
    )
    if not cell.contains(gates) or resource < floor or not blocks_feasible:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.PRIMAL_INFEASIBLE,
            "reference gates violate cell/resource/box/block feasibility",
        )
    try:
        exact = handle.evaluate_exact_cpu_reference(gates)
    except Exception as error:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
            f"exact primal reference requires a CPU-reference source: {type(error).__name__}",
        )
    radius = Fraction.from_float(request.literal_j_response_radius)
    if exact.q_value > radius * radius:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.PRIMAL_INFEASIBLE,
            "reference gates violate the literal-J response ball",
        )
    return V1SignedWorkVariableShellPhaseBPrimalReference(
        request_hash=request.request_hash,
        source_receipt_hash=handle.receipt.receipt_hash,
        shell_gate_hash=content_hash(
            {"shell_indices": handle.receipt.shell_indices, "gates": gates},
            prefix=_PREFIX + "primal-z-",
        ),
        f_value_numerator=exact.f_value.numerator,
        f_value_denominator=exact.f_value.denominator,
        q_value_numerator=exact.q_value.numerator,
        q_value_denominator=exact.q_value.denominator,
        resource_value_numerator=resource.numerator,
        resource_value_denominator=resource.denominator,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellActionSeparationCUDAHBMForecast:
    registry_receipt_hash: str
    design_hash: str
    maximum_shell_size_observed: int
    maximum_action_cells_per_branch: int
    maximum_concurrent_cells: int
    maximum_native_fiber_affine_rows: int
    maximum_literal_j_response_rows: int
    simultaneous_compact_source_count: int
    source_retained_tensor_bytes_already_reserved: int
    simultaneous_compact_source_workspace_bytes: int
    maximum_source_compaction_workspace_bytes: int
    bounded_solver_workspace_bytes: int
    current_reserved_device_bytes: int
    projected_peak_reserved_bytes: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    hbm_ceiling_fraction: float
    hbm_ceiling_bytes: int
    admitted: bool
    _issuer_token: InitVar[object] = None
    schema: str = ACTION_CUDA_HBM_FORECAST_SCHEMA

    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _issuer_token: object, _admission_memory: tuple[int, int] | None) -> None:
        if _issuer_token is not _SOLVER_RECEIPT_TOKEN:
            raise TypeError("action-separation HBM forecasts are issuer-only")
        for name in ("registry_receipt_hash", "design_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "maximum_shell_size_observed",
            "maximum_action_cells_per_branch",
            "maximum_concurrent_cells",
            "maximum_native_fiber_affine_rows",
            "maximum_literal_j_response_rows",
            "simultaneous_compact_source_count",
            "source_retained_tensor_bytes_already_reserved",
            "simultaneous_compact_source_workspace_bytes",
            "maximum_source_compaction_workspace_bytes",
            "bounded_solver_workspace_bytes",
            "current_reserved_device_bytes",
            "projected_peak_reserved_bytes",
            "device_total_memory_bytes",
            "device_free_memory_bytes",
            "hbm_ceiling_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name.replace("_", " ")),
            )
        shell = self.maximum_shell_size_observed
        concurrent = self.maximum_concurrent_cells
        if not 1 <= shell <= _MAXIMUM_SHELL_SIZE or concurrent <= 0:
            raise ValueError("HBM forecast shell/concurrency axes are invalid")
        if not 1 <= self.maximum_action_cells_per_branch <= 1 << shell:
            raise ValueError("HBM forecast action-cell quotient count is invalid")
        # Every live branch source may be simultaneously compacted while the
        # certificate compares branches.  This forecast therefore separates the
        # source tensors that are already reserved from the additional compact
        # Phase-B branch-source workspace and refuses any one-source max fiction.
        compact_source_upper = 8 * (
            self.maximum_native_fiber_affine_rows * shell
            + 3 * self.maximum_native_fiber_affine_rows
            + self.maximum_literal_j_response_rows * shell
            + 4 * self.maximum_literal_j_response_rows
        )
        if self.simultaneous_compact_source_count <= 0:
            raise ValueError("HBM forecast must count at least one compact source")
        expected_compact_sources = (
            self.simultaneous_compact_source_count * compact_source_upper
        )
        per_cell = (
            self.maximum_native_fiber_affine_rows
            + self.maximum_literal_j_response_rows
            + 24 * shell * shell
            + 96 * shell
            + 256
        )
        expected_workspace = expected_compact_sources + max(
            self.maximum_source_compaction_workspace_bytes,
            8 * concurrent * per_cell,
        )
        if self.simultaneous_compact_source_workspace_bytes != expected_compact_sources:
            raise ValueError("action-separation compact-source HBM formula changed")
        if self.bounded_solver_workspace_bytes != expected_workspace:
            raise ValueError("action-separation HBM workspace formula changed")
        if self.projected_peak_reserved_bytes != (
            self.current_reserved_device_bytes + expected_workspace
        ):
            raise ValueError("action-separation projected HBM peak does not close")
        ceiling = _finite_positive(
            self.hbm_ceiling_fraction, name="HBM ceiling fraction"
        )
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        expected_ceiling = math.floor(self.device_total_memory_bytes * ceiling)
        available, allocated = _admission_memory or (
            self.device_free_memory_bytes, self.current_reserved_device_bytes
        )
        expected_admitted = (
            allocated + expected_workspace <= expected_ceiling
            and expected_workspace <= available
        )
        if (
            self.hbm_ceiling_bytes != expected_ceiling
            or type(self.admitted) is not bool
            or self.admitted is not expected_admitted
        ):
            raise ValueError("action-separation HBM admission does not replay")
        if self.schema != ACTION_CUDA_HBM_FORECAST_SCHEMA:
            raise ValueError("unsupported action-separation HBM schema")

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "cuda-hbm-")


def build_v1_signed_work_variable_shell_action_separation_cuda_hbm_forecast(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
    current_reserved_device_bytes: int,
) -> V1SignedWorkVariableShellActionSeparationCUDAHBMForecast:
    admission_memory = None
    if type(registry) is not V1SignedWorkVariableShellActionCellRegistry:
        raise TypeError("HBM forecast requires the exact live registry")
    handles = registry.assert_live_sources()
    shell = max(len(handle.receipt.shell_indices) for handle in handles)
    f_rows = max(
        handle.receipt.context_count
        * len(set(handle.receipt.shell_native_fiber_indices))
        for handle in handles
    )
    q_rows = max(handle.receipt.context_count for handle in handles)
    concurrent = min(
        registry.design.maximum_concurrent_cells,
        registry.receipt.maximum_distinct_action_cells_per_branch,
    )
    compact_source = 8 * (f_rows * shell + 3 * f_rows + q_rows * shell + 4 * q_rows)
    per_cell = f_rows + q_rows + 24 * shell * shell + 96 * shell + 256
    simultaneous_compact_source_count = len(handles)
    compact_sources = simultaneous_compact_source_count * compact_source
    maximum_source_compaction_workspace = max(
        3
        * handle.receipt.context_count
        * max(
            0,
            handle.receipt.fiber_count
            - len(set(handle.receipt.shell_native_fiber_indices)),
        )
        * 8
        + handle.receipt.fiber_count
        for handle in handles
    )
    workspace = compact_sources + max(
        maximum_source_compaction_workspace,
        8 * concurrent * per_cell,
    )
    total = _positive_int(device_total_memory_bytes, name="device total memory bytes")
    free = _nonnegative_int(device_free_memory_bytes, name="device free memory bytes")
    reserved = _nonnegative_int(
        current_reserved_device_bytes,
        name="current reserved device bytes",
    )
    if (
        registry.design.execution_mode
        is V1SignedWorkVariableShellActionSeparationMode.CUDA_PRODUCTION
    ):
        if torch is None or not torch.cuda.is_available():
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.CUDA_REQUIRED,
                "production HBM forecast requires live CUDA memory facts",
            )
        devices = tuple(
            _compact_source_device(
                handle,
                purpose_hash=content_hash(
                    {
                        "registry_receipt_hash": registry.receipt.receipt_hash,
                        "source_receipt_hash": handle.receipt.receipt_hash,
                        "operation": "phase_b_hbm_forecast_device_probe",
                    },
                    prefix=_PREFIX + "hbm-device-probe-",
                ),
            )
            for handle in handles
        )
        first_device = devices[0]
        if any(device != first_device for device in devices[1:]):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                "production action separation requires one shared CUDA device",
            )
        live_free, live_total = torch.cuda.mem_get_info(first_device)
        admission_memory = (
            cuda_reusable_memory_bytes(first_device, framework=torch, free_bytes=int(live_free)),
            int(torch.cuda.memory_allocated(first_device)),
        )
        total = _positive_int(int(live_total), name="live device total memory bytes")
        free = _nonnegative_int(int(live_free), name="live device free memory bytes")
        reserved = _nonnegative_int(
            int(torch.cuda.memory_reserved(first_device)),
            name="live current reserved device bytes",
        )
    available, allocated = admission_memory or (free, reserved)
    peak = reserved + workspace
    ceiling = math.floor(total * registry.design.hbm_ceiling_fraction)
    return V1SignedWorkVariableShellActionSeparationCUDAHBMForecast(
        registry_receipt_hash=registry.receipt.receipt_hash,
        design_hash=registry.design.design_hash,
        maximum_shell_size_observed=shell,
        maximum_action_cells_per_branch=(
            registry.receipt.maximum_distinct_action_cells_per_branch
        ),
        maximum_concurrent_cells=concurrent,
        maximum_native_fiber_affine_rows=f_rows,
        maximum_literal_j_response_rows=q_rows,
        simultaneous_compact_source_count=simultaneous_compact_source_count,
        source_retained_tensor_bytes_already_reserved=sum(
            handle.retained_tensor_bytes for handle in handles
        ),
        simultaneous_compact_source_workspace_bytes=compact_sources,
        maximum_source_compaction_workspace_bytes=(maximum_source_compaction_workspace),
        bounded_solver_workspace_bytes=workspace,
        current_reserved_device_bytes=reserved,
        projected_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        device_free_memory_bytes=free,
        hbm_ceiling_fraction=registry.design.hbm_ceiling_fraction,
        hbm_ceiling_bytes=ceiling,
        admitted=(allocated + workspace <= ceiling and workspace <= available),
        _issuer_token=_SOLVER_RECEIPT_TOKEN,
        _admission_memory=admission_memory,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellPhaseBPDHGExecutionReceipt:
    request_hash: str
    source_receipt_hash: str
    source_replay_hash: str
    source_uncertainty_family_receipt_hash: str
    source_uncertainty_group_receipt_hash: str
    registered_q_branch_domain_bridge_receipt_hash: str
    design_hash: str
    device_type: str
    execution_mode: V1SignedWorkVariableShellActionSeparationMode
    iteration_count: int
    gap_check_count: int
    operator_frobenius_upper_bound: float
    primal_step_size: float
    dual_step_size: float
    relaxed_primal_shell_gates: tuple[float, ...]
    relaxed_primal_gate_hash: str
    deterministic_rounded_shell_offsets: tuple[int, ...]
    deterministic_rounding_action_hash: str
    relaxed_primal_F_upper_bound: float
    projected_dual_F_lower_bound: float
    duality_gap_upper_bound: float
    registered_gap_tolerance: float
    hard_replay_gate_hash: str
    hard_replay_F_lower: float
    hard_replay_F_upper: float
    hard_replay_Q_lower: float
    hard_replay_Q_upper: float
    hard_replay_receipt_hash: str
    source_compaction_receipt_hash: str
    compact_source_tensor_lease_hash: str
    _issuer_token: InitVar[object] = None
    deterministic_fp64_matrix_free_pdhg: bool = True
    primal_feasibility_outward_certified: bool = True
    projected_dual_lower_bound_outward_certified: bool = True
    hard_F_Q_replay_outward_certified: bool = True
    dual_gap_closed_within_registered_tolerance: bool = True
    production_cuda_execution: bool = False
    CPU_reference_test_only: bool = False
    alternate_cell_lower_bound_Q_dropped: bool = False
    d_w2_empirical_q_used_as_feasible_set: bool = True
    route_scale_D2H_bytes: int = 0
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    commit_authority: bool = False
    schema: str = ACTION_PDHG_EXECUTION_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _SOLVER_RECEIPT_TOKEN:
            raise TypeError("PDHG execution receipts require the private solver issuer")
        for name in (
            "request_hash",
            "source_receipt_hash",
            "source_replay_hash",
            "source_uncertainty_family_receipt_hash",
            "source_uncertainty_group_receipt_hash",
            "registered_q_branch_domain_bridge_receipt_hash",
            "design_hash",
            "device_type",
            "relaxed_primal_gate_hash",
            "deterministic_rounding_action_hash",
            "hard_replay_gate_hash",
            "hard_replay_receipt_hash",
            "source_compaction_receipt_hash",
            "compact_source_tensor_lease_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            type(self.execution_mode)
            is not V1SignedWorkVariableShellActionSeparationMode
        ):
            raise TypeError("PDHG execution mode must be exact")
        iterations = _positive_int(self.iteration_count, name="PDHG iteration count")
        checks = _positive_int(self.gap_check_count, name="PDHG gap-check count")
        object.__setattr__(self, "iteration_count", iterations)
        object.__setattr__(self, "gap_check_count", checks)
        for name in (
            "operator_frobenius_upper_bound",
            "primal_step_size",
            "dual_step_size",
            "relaxed_primal_F_upper_bound",
            "duality_gap_upper_bound",
            "registered_gap_tolerance",
            "hard_replay_F_lower",
            "hard_replay_F_upper",
            "hard_replay_Q_lower",
            "hard_replay_Q_upper",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name.replace("_", " ")),
            )
        object.__setattr__(
            self,
            "projected_dual_F_lower_bound",
            _finite(self.projected_dual_F_lower_bound, name="projected dual F lower"),
        )
        gates = tuple(
            _finite_nonnegative(value, name="relaxed primal shell gate")
            for value in self.relaxed_primal_shell_gates
        )
        if (
            not gates
            or len(gates) > _MAXIMUM_SHELL_SIZE
            or any(value > 1.0 for value in gates)
        ):
            raise ValueError("PDHG relaxed primal gates are outside the bounded box")
        rounded = _offset_axis(
            tuple(self.deterministic_rounded_shell_offsets),
            shell_size=len(gates),
            name="deterministically rounded shell offsets",
        )
        object.__setattr__(self, "relaxed_primal_shell_gates", gates)
        object.__setattr__(self, "deterministic_rounded_shell_offsets", rounded)
        if (
            self.hard_replay_F_lower > self.hard_replay_F_upper
            or self.hard_replay_Q_lower > self.hard_replay_Q_upper
            or self.relaxed_primal_F_upper_bound < self.projected_dual_F_lower_bound
            or self.duality_gap_upper_bound
            < self.relaxed_primal_F_upper_bound - self.projected_dual_F_lower_bound
            or self.duality_gap_upper_bound > self.registered_gap_tolerance
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.DUAL_GAP_UNRESOLVED,
                "PDHG primal/dual/outward gap bounds do not close",
            )
        expected_modes = (
            self.execution_mode
            is V1SignedWorkVariableShellActionSeparationMode.CUDA_PRODUCTION,
            self.execution_mode
            is V1SignedWorkVariableShellActionSeparationMode.EXACT_REFERENCE_TEST_ONLY,
        )
        if (
            self.production_cuda_execution,
            self.CPU_reference_test_only,
        ) != expected_modes:
            raise ValueError("PDHG execution evidence does not match its mode")
        if (
            type(self.alternate_cell_lower_bound_Q_dropped) is not bool
            or type(self.d_w2_empirical_q_used_as_feasible_set) is not bool
        ):
            raise TypeError("PDHG Q-domain markers must be boolean")
        if (
            self.alternate_cell_lower_bound_Q_dropped
            is self.d_w2_empirical_q_used_as_feasible_set
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "PDHG receipt must distinguish Q-dropped alternate domains from synthetic Q reference domains",
            )
        if (
            self.production_cuda_execution
            and self.alternate_cell_lower_bound_Q_dropped is not True
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "production Phase-B lower bounds must drop Q from alternate cell domains",
            )
        required_true = (
            self.deterministic_fp64_matrix_free_pdhg,
            self.primal_feasibility_outward_certified,
            self.projected_dual_lower_bound_outward_certified,
            self.hard_F_Q_replay_outward_certified,
            self.dual_gap_closed_within_registered_tolerance,
        )
        if any(value is not True for value in required_true):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_CELL_PROOF,
                "PDHG execution receipt is incomplete",
            )
        if (
            _nonnegative_int(self.route_scale_D2H_bytes, name="route-scale D2H bytes")
            != 0
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "PDHG execution crossed the route-scale host seam",
            )
        _closed_authority(
            (
                self.action_invariance_authority,
                self.C_stream_open_authority,
                self.commit_authority,
            ),
            owner="PDHG execution receipt",
        )
        if self.schema != ACTION_PDHG_EXECUTION_SCHEMA:
            raise ValueError("unsupported PDHG execution receipt schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "pdhg-execution-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellRestrictedCellOptimalityReceipt:
    """Array-free global-optimality receipt for one closed cover cell."""

    registry_receipt_hash: str
    request_hash: str
    action_cell_hash: str
    source_receipt_hash: str
    branch: V1SignedWorkVariableShellQBranch
    resource_alias_id: str
    phase_a_invariance_receipt_hash: str
    source_uncertainty_group_id: str
    source_uncertainty_family_receipt_hash: str
    source_uncertainty_group_receipt_hash: str
    registered_q_branch_domain_bridge_receipt_hash: str
    source_compaction_receipt_hash: str
    physical_action_coordinate_hash: str
    primal_witness_deterministic_rounding_action_hash: str
    robust_feasible_primal_F_upper_bound: float
    robust_dual_F_lower_bound: float
    global_optimality_gap_upper_bound: float
    primal_witness_hash: str
    dual_witness_hash: str
    live_source_evaluation_family_hash: str
    pdhg_execution_receipt_hash: str
    hard_replay_receipt_hash: str
    _issuer_token: InitVar[object] = None
    complete_closed_cell_solved: bool = True
    primal_feasibility_outward_certified: bool = True
    projected_dual_feasibility_outward_certified: bool = True
    global_convex_optimality_for_this_closed_cell: bool = True
    hard_F_Q_replay_outward_certified: bool = True
    primal_witness_inside_closed_cell: bool = True
    phase_a_coordinate_reopened: bool = False
    F_treated_as_feasibility_budget: bool = False
    hard_rounding_path_global_optimality_claim: bool = False
    alternate_cell_lower_bound_Q_dropped: bool = False
    d_w2_empirical_q_used_as_feasible_set: bool = True
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = ACTION_SOLVER_RECEIPT_SCHEMA

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _SOLVER_RECEIPT_TOKEN:
            raise TypeError(
                "restricted-cell optimality receipts require the private solver issuer"
            )
        for name in (
            "registry_receipt_hash",
            "request_hash",
            "action_cell_hash",
            "source_receipt_hash",
            "resource_alias_id",
            "phase_a_invariance_receipt_hash",
            "source_uncertainty_group_id",
            "source_uncertainty_family_receipt_hash",
            "source_uncertainty_group_receipt_hash",
            "registered_q_branch_domain_bridge_receipt_hash",
            "source_compaction_receipt_hash",
            "physical_action_coordinate_hash",
            "primal_witness_deterministic_rounding_action_hash",
            "primal_witness_hash",
            "dual_witness_hash",
            "live_source_evaluation_family_hash",
            "pdhg_execution_receipt_hash",
            "hard_replay_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("solver receipt branch must be exact")
        if (
            self.primal_witness_deterministic_rounding_action_hash
            != self.physical_action_coordinate_hash
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_CELL_PROOF,
                "feasible primal witness does not round to its physical action coordinate",
            )
        upper = _finite_nonnegative(
            self.robust_feasible_primal_F_upper_bound,
            name="feasible primal F upper bound",
        )
        lower = _finite(self.robust_dual_F_lower_bound, name="dual F lower bound")
        gap = _finite_nonnegative(
            self.global_optimality_gap_upper_bound,
            name="global optimality gap upper bound",
        )
        if upper < lower or gap < upper - lower:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_CELL_PROOF,
                "Phase-B primal/dual F bounds do not enclose one cell optimum",
            )
        object.__setattr__(self, "robust_feasible_primal_F_upper_bound", upper)
        object.__setattr__(self, "robust_dual_F_lower_bound", lower)
        object.__setattr__(self, "global_optimality_gap_upper_bound", gap)
        required_true = (
            self.complete_closed_cell_solved,
            self.primal_feasibility_outward_certified,
            self.projected_dual_feasibility_outward_certified,
            self.global_convex_optimality_for_this_closed_cell,
            self.hard_F_Q_replay_outward_certified,
            self.primal_witness_inside_closed_cell,
        )
        if any(value is not True for value in required_true):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_CELL_PROOF,
                "restricted-cell solver proof is incomplete",
            )
        if (
            type(self.alternate_cell_lower_bound_Q_dropped) is not bool
            or type(self.d_w2_empirical_q_used_as_feasible_set) is not bool
        ):
            raise TypeError("restricted-cell Q-domain markers must be boolean")
        if (
            self.alternate_cell_lower_bound_Q_dropped
            is self.d_w2_empirical_q_used_as_feasible_set
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "restricted-cell receipt must distinguish Q-dropped alternate domains from synthetic Q reference domains",
            )
        _closed_authority(
            (
                self.phase_a_coordinate_reopened,
                self.F_treated_as_feasibility_budget,
                self.hard_rounding_path_global_optimality_claim,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="restricted-cell optimality receipt",
        )
        if self.schema != ACTION_SOLVER_RECEIPT_SCHEMA:
            raise ValueError("unsupported restricted-cell solver receipt schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "cell-optimality-",
        )

    @property
    def byte_physical_action_hash(self) -> str:
        """Legacy wire alias; this receipt stores an action coordinate."""

        return self.physical_action_coordinate_hash


def _phase_b_order_matrix(
    request: V1SignedWorkVariableShellRestrictedConvexCellRequest,
) -> Any:
    shell = request.shell_size
    selected = set(request.action_cell.selected_shell_offsets)
    pairs = tuple(
        (left, right)
        for left in request.action_cell.selected_shell_offsets
        for right in range(shell)
        if right not in selected
    )
    source = torch.zeros(
        (len(pairs), shell),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    for row, (left, right) in enumerate(pairs):
        source[row, left] = -1.0
        source[row, right] = 1.0
    return source


def _phase_b_block_matrix(
    request: V1SignedWorkVariableShellRestrictedConvexCellRequest,
) -> Any:
    source = torch.zeros(
        (len(request.block_capacities), request.shell_size),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    for row, capacity in enumerate(request.block_capacities):
        for offset in capacity.shell_offsets:
            source[row, offset] = 1.0
    return source


def _phase_b_candidate_feasibility(
    *,
    request: V1SignedWorkVariableShellRestrictedConvexCellRequest,
    branch_source: V1SignedWorkVariableShellBranchActionCellSource,
    compact_source: V1SignedWorkVariableShellPhaseBCompactSourceHandle,
    gates: Sequence[Real],
    enforce_literal_j_q: bool,
) -> V1SignedWorkVariableShellPhaseBCompactEvaluationReceipt | None:
    """Outward-check one bounded primal candidate and deterministic replay."""

    values = _validated_gates(gates, request.shell_size)
    cell = branch_source.closed_cells(request.action_cell.resource_alias_id)[
        request.action_cell.action_cell_ordinal
    ]
    if not cell.contains(values):
        return None
    rounded = deterministic_v1_signed_work_variable_shell_rounding_subset(
        shell_gates=values,
        shell_route_indices=compact_source.source_handle.receipt.shell_indices,
        shell_prefix_count=request.action_cell.shell_prefix_count,
    )
    if rounded != request.action_cell.selected_shell_offsets:
        return None
    z = tuple(Fraction.from_float(value) for value in values)
    resource = sum(
        Fraction.from_float(coefficient) * value
        for coefficient, value in zip(
            request.shell_resource_coefficients,
            z,
            strict=True,
        )
    )
    if resource < Fraction.from_float(request.registered_phase_a_resource_floor):
        return None
    if any(
        sum(z[index] for index in capacity.shell_offsets) > capacity.remaining_capacity
        for capacity in request.block_capacities
    ):
        return None
    evaluation = compact_source.evaluate(values)
    if enforce_literal_j_q and Fraction.from_float(evaluation.Q_upper) > (
        Fraction.from_float(request.literal_j_response_radius) ** 2
    ):
        return None
    return evaluation


def _phase_b_best_feasible_candidate(
    *,
    request: V1SignedWorkVariableShellRestrictedConvexCellRequest,
    branch_source: V1SignedWorkVariableShellBranchActionCellSource,
    compact_source: V1SignedWorkVariableShellPhaseBCompactSourceHandle,
    relaxed_gates: Sequence[Real],
    enforce_literal_j_q: bool,
) -> (
    tuple[tuple[float, ...], V1SignedWorkVariableShellPhaseBCompactEvaluationReceipt]
    | None
):
    """Repair a converged cover-cell point toward its exact hard action.

    The hard endpoint lies in the deterministic physical-action preimage.  Convex
    mixtures approach a cover-cell boundary solution without ever treating a
    tied overcover point as an action witness.
    """

    base = _validated_gates(relaxed_gates, request.shell_size)
    selected = set(request.action_cell.selected_shell_offsets)
    hard = tuple(
        1.0 if index in selected else 0.0 for index in range(request.shell_size)
    )
    # Zero first preserves an already valid iterate.  Powers of two make the
    # repair axis deterministic and include the exact hard endpoint.
    mixing = (0.0,) + tuple(2.0**exponent for exponent in range(-52, 0, 4)) + (1.0,)
    best: (
        tuple[
            tuple[float, ...],
            V1SignedWorkVariableShellPhaseBCompactEvaluationReceipt,
        ]
        | None
    ) = None
    for amount in mixing:
        gates = tuple(
            (1.0 - amount) * value + amount * endpoint
            for value, endpoint in zip(base, hard, strict=True)
        )
        # Ergodic PDHG iterates approach an active resource floor from either
        # side.  Mixing with a hard endpoint that itself lies on the floor
        # cannot repair a one-ulp shortfall until amount==1, so perform the
        # exact bounded B-dimensional resource projection explicitly.
        mutable = list(gates)
        floor = Fraction.from_float(request.registered_phase_a_resource_floor)
        resource = sum(
            Fraction.from_float(coefficient) * Fraction.from_float(value)
            for coefficient, value in zip(
                request.shell_resource_coefficients,
                mutable,
                strict=True,
            )
        )
        if resource < floor:
            selected_order = tuple(request.action_cell.selected_shell_offsets) + tuple(
                index for index in range(request.shell_size) if index not in selected
            )
            for index in selected_order:
                coefficient = Fraction.from_float(
                    request.shell_resource_coefficients[index]
                )
                if coefficient <= 0 or mutable[index] >= 1.0:
                    continue
                capacity = next(
                    value
                    for value in request.block_capacities
                    if index in value.shell_offsets
                )
                block_sum = sum(
                    Fraction.from_float(mutable[offset])
                    for offset in capacity.shell_offsets
                )
                headroom = min(
                    Fraction.from_float(1.0 - mutable[index]),
                    Fraction(capacity.remaining_capacity) - block_sum,
                )
                if headroom <= 0:
                    continue
                needed = (floor - resource) / coefficient
                increment = min(headroom, needed)
                proposed = mutable[index] + float(increment)
                mutable[index] = min(1.0, math.nextafter(proposed, math.inf))
                resource = sum(
                    Fraction.from_float(value) * Fraction.from_float(mutable[offset])
                    for offset, value in enumerate(request.shell_resource_coefficients)
                )
                if resource >= floor:
                    break
            gates = tuple(mutable)
        evaluation = _phase_b_candidate_feasibility(
            request=request,
            branch_source=branch_source,
            compact_source=compact_source,
            gates=gates,
            enforce_literal_j_q=enforce_literal_j_q,
        )
        if evaluation is not None and (
            best is None or evaluation.F_upper < best[1].F_upper
        ):
            best = (gates, evaluation)
    return best


def _phase_b_projected_dual_lower_bound(
    *,
    request: V1SignedWorkVariableShellRestrictedConvexCellRequest,
    compact_source: V1SignedWorkVariableShellPhaseBCompactSourceHandle,
    a_flat: Any,
    b_flat: Any,
    alpha_flat: Any,
    jw: Any,
    dw: Any,
    block_matrix: Any,
    block_capacities: Any,
    order_matrix: Any,
    resource: Any,
    y_f: Any,
    y_q: Any,
    resource_multiplier: Any,
    block_multipliers: Any,
    order_multipliers: Any,
    drop_q_domain: bool,
) -> tuple[float, str]:
    """Project dual feasibility and return a conservative binary64 lower bound."""

    yf = torch.maximum(torch.minimum(y_f, alpha_flat), -alpha_flat)
    yq = torch.zeros_like(y_q) if drop_q_domain else y_q
    lam = torch.clamp(resource_multiplier, min=0.0)
    mu = torch.clamp(block_multipliers, min=0.0)
    nu = torch.clamp(order_multipliers, min=0.0)
    s = torch.mv(a_flat.t(), yf) - lam * resource
    if not drop_q_domain:
        s = s + torch.mv(jw.t(), yq)
    if block_matrix.shape[0]:
        s = s + torch.mv(block_matrix.t(), mu)
    if order_matrix.shape[0]:
        s = s + torch.mv(order_matrix.t(), nu)
    f_linear = torch.dot(yf, b_flat)
    if drop_q_domain:
        q_linear = torch.zeros((), dtype=torch.float64, device=a_flat.device)
        q_norm_center = 0.0
        q_norm_upper = 0.0
        q_penalty_center = 0.0
        q_penalty_upper = 0.0
    else:
        q_linear = torch.dot(yq, dw)
        q_norm_center, q_norm_upper = _stable_l2_norm_center_upper(yq)
        q_penalty_center = request.literal_j_response_radius * q_norm_center
        q_penalty_upper = _up(
            request.literal_j_response_radius * q_norm_upper + _F64_MIN_SUBNORMAL
        )
    resource_linear = lam * request.registered_phase_a_resource_floor
    block_linear = (
        torch.dot(mu, block_capacities)
        if block_capacities.numel()
        else torch.zeros((), dtype=torch.float64, device=a_flat.device)
    )
    box_linear = torch.sum(torch.minimum(s, torch.zeros_like(s)))
    linear_center_tensor = (
        f_linear + q_linear + resource_linear - block_linear + box_linear
    )
    b_radius_flat = compact_source.b_radius_touched.reshape(-1)
    d_radius_weighted = compact_source.sqrt_weights * compact_source.d_radius
    source_radius_inputs = [torch.abs(yf) * b_radius_flat]
    if not drop_q_domain:
        source_radius_inputs.append(torch.abs(yq) * d_radius_weighted)
    source_radius_terms = torch.cat(tuple(source_radius_inputs))
    _source_radius_center, source_radius_upper = _outward_nonnegative_tensor_sum(
        source_radius_terms,
        operations_per_term=5,
    )
    # A standard-model enclosure for all reductions and B-dimensional
    # stationarity terms.  It is deliberately conservative at a numerical
    # boundary, where the caller must refuse rather than infer separation.
    mass_term_tensors = (
        torch.abs(yf) * torch.abs(b_flat),
        torch.abs(yf).unsqueeze(1) * torch.abs(a_flat),
        torch.abs(lam) * torch.abs(resource),
        torch.abs(mu).unsqueeze(1) * torch.abs(block_matrix),
        torch.abs(nu).unsqueeze(1) * torch.abs(order_matrix),
        torch.abs(mu) * torch.abs(block_capacities),
        torch.abs(s),
    )
    if not drop_q_domain:
        mass_term_tensors = (
            mass_term_tensors[0],
            torch.abs(yq) * torch.abs(dw),
            mass_term_tensors[1],
            torch.abs(yq).unsqueeze(1) * torch.abs(jw),
        ) + mass_term_tensors[2:]
    mass_uppers = tuple(
        _outward_nonnegative_tensor_sum(
            terms,
            operations_per_term=6,
        )[1]
        for terms in mass_term_tensors
    )
    absolute_mass_upper = _up(
        math.fsum(
            mass_uppers
            + (
                abs(q_penalty_upper),
                abs(float(resource_linear.item())),
                abs(float(block_linear.item())),
            )
        )
    )
    linear_center = float(linear_center_tensor.item())
    center = linear_center - q_penalty_center
    if not all(
        math.isfinite(value)
        for value in (center, source_radius_upper, absolute_mass_upper)
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.NONFINITE_SOLVER,
            "projected dual lower-bound reduction became nonfinite",
        )
    operation_count = max(
        16,
        8 * int(a_flat.numel() + (0 if drop_q_domain else jw.numel()))
        + 32 * request.shell_size,
    )
    numerical_radius = _up(
        _gamma(operation_count) * absolute_mass_upper
        + _F64_MIN_SUBNORMAL * float(operation_count)
    )
    lower = _down(
        compact_source.receipt.untouched_F_constant_lower
        + linear_center
        - q_penalty_upper
        - source_radius_upper
        - numerical_radius
        - _F64_MIN_SUBNORMAL * float(operation_count)
    )
    witness_hash = content_hash(
        {
            "request_hash": request.request_hash,
            "source_compaction_receipt_hash": compact_source.receipt.receipt_hash,
            "projected_dual_lower": lower,
            "center": center,
            "source_radius_upper": source_radius_upper,
            "numerical_radius": numerical_radius,
            "absolute_mass_upper": absolute_mass_upper,
            "q_norm_upper": q_norm_upper,
            "alternate_cell_lower_bound_Q_dropped": drop_q_domain,
            "dual_vector_hashes": {
                "y_f": _bounded_fp64_tensor_digest(yf, role="dual_y_f"),
                "y_q": _bounded_fp64_tensor_digest(yq, role="dual_y_q"),
                "lambda": _bounded_fp64_tensor_digest(lam, role="dual_lambda"),
                "mu": _bounded_fp64_tensor_digest(mu, role="dual_mu"),
                "nu": _bounded_fp64_tensor_digest(nu, role="dual_nu"),
            },
            "bounded_problem_tensor_hashes": {
                "a": _bounded_fp64_tensor_digest(a_flat, role="problem_a"),
                "b": _bounded_fp64_tensor_digest(b_flat, role="problem_b"),
                "alpha": _bounded_fp64_tensor_digest(alpha_flat, role="problem_alpha"),
                "j": _bounded_fp64_tensor_digest(jw, role="problem_j"),
                "d": _bounded_fp64_tensor_digest(dw, role="problem_d"),
                "block": _bounded_fp64_tensor_digest(
                    block_matrix, role="problem_block"
                ),
                "order": _bounded_fp64_tensor_digest(
                    order_matrix, role="problem_order"
                ),
                "resource": _bounded_fp64_tensor_digest(
                    resource, role="problem_resource"
                ),
            },
            "replay_complete_bounded_dual_and_problem": True,
        },
        prefix=_PREFIX + "projected-dual-device-witness-",
    )
    return lower, witness_hash


def _solve_v1_signed_work_variable_shell_phase_b_cell_pdhg(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    request: V1SignedWorkVariableShellRestrictedConvexCellRequest,
    compact_source: V1SignedWorkVariableShellPhaseBCompactSourceHandle,
    source_uncertainty_family: (
        V1SignedWorkVariableShellSourceUncertaintyFamilyReceipt
        | V1SignedWorkVariableShellSampledSupportUncertaintyFamilyReceipt
    ),
    source_uncertainty_group: (
        V1SignedWorkVariableShellSourceUncertaintyGroupReceipt
        | V1SignedWorkVariableShellSampledSupportUncertaintyGroupReceipt
    ),
    q_domain_bridge: V1SignedWorkVariableShellRegisteredQBranchDomainBridgeReceipt,
) -> tuple[
    V1SignedWorkVariableShellRestrictedCellOptimalityReceipt,
    V1SignedWorkVariableShellPhaseBPDHGExecutionReceipt,
]:
    """Solve and certify one registered Phase-B closed overcover cell."""

    if request.registry_receipt_hash != registry.receipt.receipt_hash:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
            "restricted cell request belongs to another registry",
        )
    compact_source.assert_live()
    if (
        type(q_domain_bridge)
        is not V1SignedWorkVariableShellRegisteredQBranchDomainBridgeReceipt
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.REGISTERED_Q_BRANCH_DOMAIN_BRIDGE_UNAVAILABLE,
            "Phase-B PDHG requires a typed registered-branch/D_W2 Q-domain relation",
        )
    branch_source = registry.branch_source(request.action_cell.branch)
    if (
        type(source_uncertainty_family)
        is V1SignedWorkVariableShellSourceUncertaintyFamilyReceipt
    ):
        if (
            type(source_uncertainty_group)
            is not V1SignedWorkVariableShellSourceUncertaintyGroupReceipt
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "exact-census source family was paired with a non-census source group",
            )
        group_source_receipt_hash = source_uncertainty_group.source_receipt_hash
    elif (
        type(source_uncertainty_family)
        is V1SignedWorkVariableShellSampledSupportUncertaintyFamilyReceipt
    ):
        if (
            type(source_uncertainty_group)
            is not V1SignedWorkVariableShellSampledSupportUncertaintyGroupReceipt
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
                "sampled-support source family was paired with a non-sampled source group",
            )
        group_source_receipt_hash = source_uncertainty_group.fit_source_receipt_hash
    else:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
            "Phase-B solver requires exact-census or sampled-support source uncertainty",
        )
    if (
        request.source_receipt_hash != compact_source.receipt.source_receipt_hash
        or group_source_receipt_hash != request.source_receipt_hash
        or source_uncertainty_family.registry_receipt_hash
        != registry.receipt.receipt_hash
        or q_domain_bridge.registry_receipt_hash != registry.receipt.receipt_hash
        or q_domain_bridge.request_hash != request.request_hash
        or q_domain_bridge.source_receipt_hash != request.source_receipt_hash
        or q_domain_bridge.branch is not request.action_cell.branch
        or q_domain_bridge.resource_alias_id != request.action_cell.resource_alias_id
        or q_domain_bridge.literal_j_response_radius
        != request.literal_j_response_radius
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
            "solver source, uncertainty family, and cell request do not share custody",
        )
    device = compact_source.a_compact.device
    mode = registry.design.execution_mode
    drop_q_domain = q_domain_bridge.alternate_cell_lower_bound_Q_dropped
    if mode is V1SignedWorkVariableShellActionSeparationMode.CUDA_PRODUCTION:
        if (
            not q_domain_bridge.production_registered_branch_enclosure_authority
            or not drop_q_domain
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.REGISTERED_Q_BRANCH_DOMAIN_BRIDGE_UNAVAILABLE,
                "production requires a source-bound bridge that drops Q for alternate lower-bound domains and restores original registered Q for the preferred hard action",
            )
        if device.type != "cuda":
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.CUDA_REQUIRED,
                "production Phase-B PDHG has no CPU fallback",
            )
        warn_only = getattr(
            torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False
        )
        if not torch.are_deterministic_algorithms_enabled() or bool(warn_only()):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INVALID_DESIGN,
                "production Phase-B PDHG requires strict deterministic algorithms",
            )
    elif device.type != "cpu":
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.CUDA_REQUIRED,
            "exact reference solver mode is private to tiny CPU fixtures",
        )
    elif not q_domain_bridge.exact_synthetic_reference_identity:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.REGISTERED_Q_BRANCH_DOMAIN_BRIDGE_UNAVAILABLE,
            "reference PDHG requires an explicit synthetic Q-domain identity",
        )

    shell = request.shell_size
    a_flat = compact_source.a_compact.reshape(-1, shell)
    b_flat = compact_source.b_touched.reshape(-1)
    alpha_flat = compact_source.alpha.expand_as(compact_source.b_touched).reshape(-1)
    jw = compact_source.sqrt_weights.unsqueeze(1) * compact_source.j_shell
    dw = compact_source.sqrt_weights * compact_source.d_response
    resource = torch.tensor(
        request.shell_resource_coefficients,
        dtype=torch.float64,
        device=device,
    )
    block_matrix = _phase_b_block_matrix(request).to(device=device)
    block_capacities = torch.tensor(
        tuple(float(value.remaining_capacity) for value in request.block_capacities),
        dtype=torch.float64,
        device=device,
    )
    order_matrix = _phase_b_order_matrix(request).to(device=device)

    operator_terms = [a_flat, resource, block_matrix, order_matrix]
    if not drop_q_domain:
        operator_terms.insert(1, jw)
    _operator_center, operator_bound = _stable_l2_norm_center_upper(*operator_terms)
    if not math.isfinite(operator_bound) or operator_bound <= 0.0:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.NONFINITE_SOLVER,
            "PDHG operator norm bound is nonpositive or nonfinite",
        )
    step = math.nextafter(
        registry.design.operator_norm_safety_factor / operator_bound,
        0.0,
    )
    if not math.isfinite(step) or step <= 0.0:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.NONFINITE_SOLVER,
            "PDHG step size cannot be represented safely",
        )

    selected = set(request.action_cell.selected_shell_offsets)
    hard_gates = tuple(1.0 if index in selected else 0.0 for index in range(shell))
    hard_evaluation = _phase_b_candidate_feasibility(
        request=request,
        branch_source=branch_source,
        compact_source=compact_source,
        gates=hard_gates,
        enforce_literal_j_q=not drop_q_domain,
    )
    if hard_evaluation is None:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.HARD_REPLAY_INFEASIBLE,
            "the registered physical action failed outward F/Q/resource replay",
        )
    z = torch.tensor(hard_gates, dtype=torch.float64, device=device)
    z_bar = z.clone()
    y_f = torch.zeros_like(b_flat)
    y_q = torch.zeros_like(dw)
    lam = torch.zeros((), dtype=torch.float64, device=device)
    mu = torch.zeros((block_matrix.shape[0],), dtype=torch.float64, device=device)
    nu = torch.zeros((order_matrix.shape[0],), dtype=torch.float64, device=device)
    z_sum = torch.zeros_like(z)
    yf_sum = torch.zeros_like(y_f)
    yq_sum = torch.zeros_like(y_q)
    lam_sum = torch.zeros_like(lam)
    mu_sum = torch.zeros_like(mu)
    nu_sum = torch.zeros_like(nu)
    best_gates = hard_gates
    best_evaluation = hard_evaluation
    best_dual_lower = -math.inf
    best_dual_hash = ""
    closed_iteration = 0
    gap_checks = 0
    final_gap = math.inf
    final_tolerance = 0.0

    for iteration in range(1, registry.design.maximum_pdhg_iterations + 1):
        y_f = torch.maximum(
            torch.minimum(
                y_f + step * (torch.mv(a_flat, z_bar) + b_flat),
                alpha_flat,
            ),
            -alpha_flat,
        )
        if drop_q_domain:
            y_q = torch.zeros_like(y_q)
        else:
            q_candidate = y_q + step * (torch.mv(jw, z_bar) + dw)
            q_norm = torch.linalg.vector_norm(q_candidate)
            shrink = torch.clamp(
                1.0
                - step
                * request.literal_j_response_radius
                / torch.clamp(q_norm, min=torch.finfo(torch.float64).tiny),
                min=0.0,
            )
            y_q = shrink * q_candidate
        lam = torch.clamp(
            lam
            + step
            * (request.registered_phase_a_resource_floor - torch.dot(resource, z_bar)),
            min=0.0,
        )
        if block_matrix.shape[0]:
            mu = torch.clamp(
                mu + step * (torch.mv(block_matrix, z_bar) - block_capacities),
                min=0.0,
            )
        if order_matrix.shape[0]:
            nu = torch.clamp(
                nu + step * torch.mv(order_matrix, z_bar),
                min=0.0,
            )
        gradient = torch.mv(a_flat.t(), y_f) - lam * resource
        if not drop_q_domain:
            gradient = gradient + torch.mv(jw.t(), y_q)
        if block_matrix.shape[0]:
            gradient = gradient + torch.mv(block_matrix.t(), mu)
        if order_matrix.shape[0]:
            gradient = gradient + torch.mv(order_matrix.t(), nu)
        z_next = torch.clamp(z - step * gradient, min=0.0, max=1.0)
        z_bar = z_next + registry.design.pdhg_extrapolation_theta * (z_next - z)
        z = z_next
        z_sum += z
        yf_sum += y_f
        yq_sum += y_q
        lam_sum += lam
        mu_sum += mu
        nu_sum += nu

        should_check = (
            iteration % registry.design.pdhg_gap_check_interval == 0
            or iteration == registry.design.maximum_pdhg_iterations
        )
        if not should_check:
            continue
        gap_checks += 1
        divisor = float(iteration)
        averaged_z = z_sum / divisor
        averaged_gates = tuple(
            float(value) for value in averaged_z.detach().cpu().tolist()
        )
        candidate = _phase_b_best_feasible_candidate(
            request=request,
            branch_source=branch_source,
            compact_source=compact_source,
            relaxed_gates=averaged_gates,
            enforce_literal_j_q=not drop_q_domain,
        )
        if candidate is not None and candidate[1].F_upper < best_evaluation.F_upper:
            best_gates, best_evaluation = candidate
        current_gates = tuple(float(value) for value in z.detach().cpu().tolist())
        current_candidate = _phase_b_best_feasible_candidate(
            request=request,
            branch_source=branch_source,
            compact_source=compact_source,
            relaxed_gates=current_gates,
            enforce_literal_j_q=not drop_q_domain,
        )
        if (
            current_candidate is not None
            and current_candidate[1].F_upper < best_evaluation.F_upper
        ):
            best_gates, best_evaluation = current_candidate
        dual_candidates = tuple(
            _phase_b_projected_dual_lower_bound(
                request=request,
                compact_source=compact_source,
                a_flat=a_flat,
                b_flat=b_flat,
                alpha_flat=alpha_flat,
                jw=jw,
                dw=dw,
                block_matrix=block_matrix,
                block_capacities=block_capacities,
                order_matrix=order_matrix,
                resource=resource,
                y_f=yf_value,
                y_q=yq_value,
                resource_multiplier=lam_value,
                block_multipliers=mu_value,
                order_multipliers=nu_value,
                drop_q_domain=drop_q_domain,
            )
            for yf_value, yq_value, lam_value, mu_value, nu_value in (
                (
                    yf_sum / divisor,
                    yq_sum / divisor,
                    lam_sum / divisor,
                    mu_sum / divisor,
                    nu_sum / divisor,
                ),
                (y_f, y_q, lam, mu, nu),
            )
        )
        dual_lower, dual_hash = max(dual_candidates, key=lambda value: value[0])
        # Any reversal invalidates at least one enclosure.  Clipping a dual
        # value below a feasible primal value cannot turn it into a valid
        # lower bound and can manufacture a false small gap.
        if dual_lower > best_evaluation.F_upper:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.NONFINITE_SOLVER,
                "projected dual lower bound exceeded a feasible primal value",
            )
        if dual_lower > best_dual_lower:
            best_dual_lower = dual_lower
            best_dual_hash = dual_hash
        final_gap = _up(best_evaluation.F_upper - best_dual_lower)
        final_tolerance = _up(
            registry.design.absolute_duality_gap_tolerance
            + registry.design.relative_duality_gap_tolerance
            * max(
                1.0,
                abs(best_evaluation.F_upper),
                abs(best_dual_lower),
            )
        )
        if (
            iteration >= registry.design.minimum_pdhg_iterations
            and final_gap <= final_tolerance
        ):
            closed_iteration = iteration
            break

    if closed_iteration == 0 or not best_dual_hash:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.DUAL_GAP_UNRESOLVED,
            "Phase-B cell PDHG did not close its outward primal/dual gap "
            f"(upper={best_evaluation.F_upper:.17g}, "
            f"lower={best_dual_lower:.17g}, gap={final_gap:.17g}, "
            f"tolerance={final_tolerance:.17g})",
        )
    rounded_offsets = deterministic_v1_signed_work_variable_shell_rounding_subset(
        shell_gates=best_gates,
        shell_route_indices=compact_source.source_handle.receipt.shell_indices,
        shell_prefix_count=request.action_cell.shell_prefix_count,
    )
    if rounded_offsets != request.action_cell.selected_shell_offsets:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.DETERMINISTIC_ROUNDING_MISMATCH,
            "certified relaxed primal does not replay to the registered physical action",
        )
    relaxed_gate_hash = content_hash(
        {"request_hash": request.request_hash, "gates": best_gates},
        prefix=_PREFIX + "relaxed-primal-gates-",
    )
    hard_replay_hash = content_hash(
        {
            "request_hash": request.request_hash,
            "hard_gates": hard_gates,
            "compact_evaluation_receipt_hash": hard_evaluation.receipt_hash,
            "physical_action_coordinate_hash": (
                request.action_cell.physical_action_coordinate_hash
            ),
        },
        prefix=_PREFIX + "hard-fq-replay-",
    )
    execution = V1SignedWorkVariableShellPhaseBPDHGExecutionReceipt(
        request_hash=request.request_hash,
        source_receipt_hash=request.source_receipt_hash,
        source_replay_hash=request.source_replay_hash,
        source_uncertainty_family_receipt_hash=source_uncertainty_family.receipt_hash,
        source_uncertainty_group_receipt_hash=source_uncertainty_group.receipt_hash,
        registered_q_branch_domain_bridge_receipt_hash=q_domain_bridge.receipt_hash,
        design_hash=registry.design.design_hash,
        device_type=device.type,
        execution_mode=mode,
        iteration_count=closed_iteration,
        gap_check_count=gap_checks,
        operator_frobenius_upper_bound=operator_bound,
        primal_step_size=step,
        dual_step_size=step,
        relaxed_primal_shell_gates=best_gates,
        relaxed_primal_gate_hash=relaxed_gate_hash,
        deterministic_rounded_shell_offsets=rounded_offsets,
        deterministic_rounding_action_hash=(
            request.action_cell.physical_action_coordinate_hash
        ),
        relaxed_primal_F_upper_bound=best_evaluation.F_upper,
        projected_dual_F_lower_bound=best_dual_lower,
        duality_gap_upper_bound=final_gap,
        registered_gap_tolerance=final_tolerance,
        hard_replay_gate_hash=hard_evaluation.shell_gate_hash,
        hard_replay_F_lower=hard_evaluation.F_lower,
        hard_replay_F_upper=hard_evaluation.F_upper,
        hard_replay_Q_lower=hard_evaluation.Q_lower,
        hard_replay_Q_upper=hard_evaluation.Q_upper,
        hard_replay_receipt_hash=hard_replay_hash,
        source_compaction_receipt_hash=compact_source.receipt.receipt_hash,
        compact_source_tensor_lease_hash=(
            compact_source.receipt.compact_source_tensor_lease_hash
        ),
        _issuer_token=_SOLVER_RECEIPT_TOKEN,
        production_cuda_execution=(
            mode is V1SignedWorkVariableShellActionSeparationMode.CUDA_PRODUCTION
        ),
        CPU_reference_test_only=(
            mode
            is V1SignedWorkVariableShellActionSeparationMode.EXACT_REFERENCE_TEST_ONLY
        ),
        alternate_cell_lower_bound_Q_dropped=drop_q_domain,
        d_w2_empirical_q_used_as_feasible_set=not drop_q_domain,
    )
    optimality = V1SignedWorkVariableShellRestrictedCellOptimalityReceipt(
        registry_receipt_hash=registry.receipt.receipt_hash,
        request_hash=request.request_hash,
        action_cell_hash=request.action_cell.action_cell_hash,
        source_receipt_hash=request.source_receipt_hash,
        branch=request.action_cell.branch,
        resource_alias_id=request.action_cell.resource_alias_id,
        phase_a_invariance_receipt_hash=(
            request.action_cell.phase_a_invariance_receipt_hash
        ),
        source_uncertainty_group_id=(
            source_uncertainty_group.source_uncertainty_group_id
        ),
        source_uncertainty_family_receipt_hash=(source_uncertainty_family.receipt_hash),
        source_uncertainty_group_receipt_hash=(source_uncertainty_group.receipt_hash),
        registered_q_branch_domain_bridge_receipt_hash=q_domain_bridge.receipt_hash,
        source_compaction_receipt_hash=compact_source.receipt.receipt_hash,
        physical_action_coordinate_hash=(
            request.action_cell.physical_action_coordinate_hash
        ),
        primal_witness_deterministic_rounding_action_hash=(
            request.action_cell.physical_action_coordinate_hash
        ),
        robust_feasible_primal_F_upper_bound=best_evaluation.F_upper,
        robust_dual_F_lower_bound=best_dual_lower,
        global_optimality_gap_upper_bound=final_gap,
        primal_witness_hash=relaxed_gate_hash,
        dual_witness_hash=best_dual_hash,
        live_source_evaluation_family_hash=content_hash(
            {
                "relaxed": best_evaluation.receipt_hash,
                "hard": hard_evaluation.receipt_hash,
                "compaction": compact_source.receipt.receipt_hash,
            },
            prefix=_PREFIX + "live-source-evaluation-family-",
        ),
        pdhg_execution_receipt_hash=execution.receipt_hash,
        hard_replay_receipt_hash=hard_replay_hash,
        _issuer_token=_SOLVER_RECEIPT_TOKEN,
        alternate_cell_lower_bound_Q_dropped=drop_q_domain,
        d_w2_empirical_q_used_as_feasible_set=not drop_q_domain,
    )
    return optimality, execution


def _action_separation_source_group_key(
    value: object,
) -> tuple[str, str, str, str, str]:
    return (
        _identity(
            getattr(value, "source_uncertainty_group_id"),
            name="source uncertainty group id",
        ),
        _identity(getattr(value, "source_receipt_hash"), name="source receipt hash"),
        getattr(value, "branch").value,
        _identity(getattr(value, "resource_alias_id"), name="resource alias id"),
        _identity(
            getattr(value, "phase_a_invariance_receipt_hash"),
            name="Phase-A invariance receipt hash",
        ),
    )


def _aggregate_action_separation_bounds(
    *,
    observations: Sequence[object],
    preferred_physical_action_coordinate_hash: str,
) -> tuple[
    tuple[tuple[tuple[str, str, str, str, str], float], ...],
    float,
    float,
]:
    preferred = _identity(
        preferred_physical_action_coordinate_hash,
        name="preferred physical action coordinate hash",
    )
    values = tuple(observations)
    group_keys = tuple(
        sorted({_action_separation_source_group_key(value) for value in values})
    )
    preferred_group_uppers: list[tuple[tuple[str, str, str, str, str], float]] = []
    for key in group_keys:
        candidates = tuple(
            float(getattr(value, "robust_feasible_primal_F_upper_bound"))
            for value in values
            if _action_separation_source_group_key(value) == key
            and getattr(value, "physical_action_coordinate_hash") == preferred
            and getattr(value, "primal_witness_deterministic_rounding_action_hash")
            == preferred
        )
        if not candidates:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_COVERAGE,
                "one required source/branch/resource group has no preferred-action primal witness",
            )
        # Existential only within duplicate compatible cells of one common
        # source/branch/resource group.
        preferred_group_uppers.append((key, min(candidates)))
    alternate_lowers_by_group: list[tuple[tuple[str, str, str, str, str], float]] = []
    for key in group_keys:
        candidates = tuple(
            float(getattr(value, "robust_dual_F_lower_bound"))
            for value in values
            if _action_separation_source_group_key(value) == key
            and getattr(value, "physical_action_coordinate_hash") != preferred
        )
        if not candidates:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_COVERAGE,
                "one required source/branch/resource group has no alternate-action dual lower bound",
            )
        alternate_lowers_by_group.append((key, min(candidates)))
    if not preferred_group_uppers or not alternate_lowers_by_group:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_CELL_PROOF,
            "nontrivial action separation needs preferred and alternate cells",
        )
    # Preferred feasibility must hold for every required uncertainty group;
    # alternate exclusion must hold against every admissible alternate cell.
    return (
        tuple(preferred_group_uppers),
        max(value for _key, value in preferred_group_uppers),
        min(value for _key, value in alternate_lowers_by_group),
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellActionSeparationBoundObservation:
    """Array-free exact-reference input for the robust quantifier order."""

    source_uncertainty_group_id: str
    source_receipt_hash: str
    branch: V1SignedWorkVariableShellQBranch
    resource_alias_id: str
    phase_a_invariance_receipt_hash: str
    action_cell_hash: str
    physical_action_coordinate_hash: str
    primal_witness_deterministic_rounding_action_hash: str
    robust_feasible_primal_F_upper_bound: float
    robust_dual_F_lower_bound: float
    source_uncertainty_coverage_complete: bool = True
    hard_F_Q_replay_outward_certified: bool = True
    reference_test_only: bool = True
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    schema: str = ACTION_QUANTIFIER_OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "source_uncertainty_group_id",
            "source_receipt_hash",
            "resource_alias_id",
            "phase_a_invariance_receipt_hash",
            "action_cell_hash",
            "physical_action_coordinate_hash",
            "primal_witness_deterministic_rounding_action_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("bound observation branch must be exact")
        object.__setattr__(
            self,
            "robust_feasible_primal_F_upper_bound",
            _finite_nonnegative(
                self.robust_feasible_primal_F_upper_bound,
                name="reference primal F upper",
            ),
        )
        object.__setattr__(
            self,
            "robust_dual_F_lower_bound",
            _finite(self.robust_dual_F_lower_bound, name="reference dual F lower"),
        )
        if (
            self.source_uncertainty_coverage_complete is not True
            or self.hard_F_Q_replay_outward_certified is not True
            or self.reference_test_only is not True
        ):
            raise ValueError("bound observation lacks source/replay coverage")
        _closed_authority(
            (self.action_invariance_authority, self.C_stream_open_authority),
            owner="action-separation bound observation",
        )
        if self.schema != ACTION_QUANTIFIER_OBSERVATION_SCHEMA:
            raise ValueError("unsupported bound-observation schema")

    @property
    def byte_physical_action_hash(self) -> str:
        """Legacy wire alias; this observation stores an action coordinate."""

        return self.physical_action_coordinate_hash


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellActionSeparationQuantifierReference:
    preferred_physical_action_coordinate_hash: str
    preferred_source_group_F_upper_bounds: tuple[
        tuple[tuple[str, str, str, str, str], float], ...
    ]
    robust_preferred_feasible_primal_F_upper_bound: float
    minimum_alternate_dual_F_lower_bound: float
    strict_separation_margin: float
    strict_separation_holds: bool
    existential_duplicate_cell_then_universal_source_group_order: bool = True
    reference_test_only: bool = True
    action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    schema: str = ACTION_QUANTIFIER_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preferred_physical_action_coordinate_hash",
            _identity(
                self.preferred_physical_action_coordinate_hash,
                name="preferred physical action coordinate hash",
            ),
        )
        upper = _finite_nonnegative(
            self.robust_preferred_feasible_primal_F_upper_bound,
            name="robust preferred F upper",
        )
        lower = _finite(
            self.minimum_alternate_dual_F_lower_bound,
            name="minimum alternate F lower",
        )
        margin = _finite(self.strict_separation_margin, name="strict separation margin")
        if margin != lower - upper:
            raise ValueError("quantifier reference margin does not replay")
        object.__setattr__(
            self, "robust_preferred_feasible_primal_F_upper_bound", upper
        )
        object.__setattr__(self, "minimum_alternate_dual_F_lower_bound", lower)
        object.__setattr__(self, "strict_separation_margin", margin)
        if self.strict_separation_holds is not (margin > 0.0):
            raise ValueError("quantifier strict-separation marker does not replay")
        if (
            self.existential_duplicate_cell_then_universal_source_group_order
            is not True
            or self.reference_test_only is not True
        ):
            raise ValueError("quantifier reference order/mode changed")
        _closed_authority(
            (self.action_invariance_authority, self.C_stream_open_authority),
            owner="action-separation quantifier reference",
        )
        if self.schema != ACTION_QUANTIFIER_REFERENCE_SCHEMA:
            raise ValueError("unsupported quantifier-reference schema")

    @property
    def preferred_byte_physical_action_hash(self) -> str:
        """Legacy wire alias; this reference stores an action coordinate."""

        return self.preferred_physical_action_coordinate_hash


def evaluate_v1_signed_work_variable_shell_action_separation_quantifiers_reference_only(
    *,
    observations: Sequence[V1SignedWorkVariableShellActionSeparationBoundObservation],
    preferred_physical_action_coordinate_hash: str,
) -> V1SignedWorkVariableShellActionSeparationQuantifierReference:
    values = tuple(observations)
    if not values or any(
        type(value) is not V1SignedWorkVariableShellActionSeparationBoundObservation
        for value in values
    ):
        raise TypeError("quantifier reference requires exact bound observations")
    group_bounds, robust_upper, alternate_lower = _aggregate_action_separation_bounds(
        observations=values,
        preferred_physical_action_coordinate_hash=(
            preferred_physical_action_coordinate_hash
        ),
    )
    margin = alternate_lower - robust_upper
    return V1SignedWorkVariableShellActionSeparationQuantifierReference(
        preferred_physical_action_coordinate_hash=(
            preferred_physical_action_coordinate_hash
        ),
        preferred_source_group_F_upper_bounds=group_bounds,
        robust_preferred_feasible_primal_F_upper_bound=robust_upper,
        minimum_alternate_dual_F_lower_bound=alternate_lower,
        strict_separation_margin=margin,
        strict_separation_holds=margin > 0.0,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellOutsideShellExclusionDependency:
    """Typed missing premise separating shell-conditional from global claims."""

    registry_receipt_hash: str
    shell_freeze_receipt_hash: str
    within_shell_proof_hash: str
    refusal_code: V1SignedWorkVariableShellActionSeparationRefusalCode = (
        V1SignedWorkVariableShellActionSeparationRefusalCode.OUTSIDE_SHELL_EXCLUSION_UNAVAILABLE
    )
    complete_outside_coordinate_screen: bool = False
    global_next_action_invariance_authority: bool = False
    C_stream_open_authority: bool = False
    commit_authority: bool = False
    schema: str = ACTION_OUTSIDE_SHELL_DEPENDENCY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "registry_receipt_hash",
            "shell_freeze_receipt_hash",
            "within_shell_proof_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            self.refusal_code
            is not V1SignedWorkVariableShellActionSeparationRefusalCode.OUTSIDE_SHELL_EXCLUSION_UNAVAILABLE
        ):
            raise ValueError("outside-shell dependency has another refusal code")
        _closed_authority(
            (
                self.complete_outside_coordinate_screen,
                self.global_next_action_invariance_authority,
                self.C_stream_open_authority,
                self.commit_authority,
            ),
            owner="outside-shell exclusion dependency",
        )
        if self.schema != ACTION_OUTSIDE_SHELL_DEPENDENCY_SCHEMA:
            raise ValueError("unsupported outside-shell dependency schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "outside-shell-dependency-")


class V1SignedWorkVariableShellActionSeparationCertificate:
    """Live Phase-B separation theorem, conditional on the frozen shell only."""

    __slots__ = (
        "registry",
        "source_uncertainty_family",
        "optimality_receipts",
        "pdhg_execution_receipts",
        "preferred_physical_action_coordinate_hash",
        "preferred_source_group_F_upper_bounds",
        "robust_preferred_feasible_primal_F_upper_bound",
        "minimum_alternate_dual_F_lower_bound",
        "strict_separation_margin_lower_bound",
        "within_shell_proof_hash",
        "outside_shell_exclusion_dependency",
        "certificate_hash",
        "within_frozen_shell_conditional_separation_certified",
        "action_invariance_authority",
        "global_next_action_invariance_authority",
        "outside_shell_exclusion_authority",
        "production_C_stream_open_authority",
        "calibration_authority",
        "oracle_authority",
        "commit_authority",
        "terminal_authority",
        "pareto_authority",
        "schema",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        registry: V1SignedWorkVariableShellActionCellRegistry,
        source_uncertainty_family: (
            V1SignedWorkVariableShellSourceUncertaintyFamilyReceipt
            | V1SignedWorkVariableShellSampledSupportUncertaintyFamilyReceipt
        ),
        optimality_receipts: Sequence[
            V1SignedWorkVariableShellRestrictedCellOptimalityReceipt
        ],
        pdhg_execution_receipts: Sequence[
            V1SignedWorkVariableShellPhaseBPDHGExecutionReceipt
        ],
        preferred_physical_action_coordinate_hash: str | None = None,
        strict_separation_margin_lower_bound: float = math.nan,
        _issuer_token: object = None,
        preferred_byte_physical_action_hash: str | None = None,
    ) -> None:
        if _issuer_token is not _CERTIFICATE_TOKEN:
            raise TypeError(
                "action-separation certificates require the private registry issuer"
            )
        if type(registry) is not V1SignedWorkVariableShellActionCellRegistry:
            raise TypeError("action-separation certificate requires the live registry")
        if type(source_uncertainty_family) not in {
            V1SignedWorkVariableShellSourceUncertaintyFamilyReceipt,
            V1SignedWorkVariableShellSampledSupportUncertaintyFamilyReceipt,
        }:
            raise TypeError("action separation requires an explicit uncertainty family")
        registry.assert_live_sources()
        if (
            source_uncertainty_family.registry_receipt_hash
            != registry.receipt.receipt_hash
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
                "action-separation uncertainty family belongs to another registry",
            )
        receipts = tuple(optimality_receipts)
        executions = tuple(pdhg_execution_receipts)
        expected_cells = tuple(registry.iter_action_cells())
        expected_axis = set()
        for cell in expected_cells:
            if (
                type(source_uncertainty_family)
                is V1SignedWorkVariableShellSourceUncertaintyFamilyReceipt
            ):
                group = source_uncertainty_family.group_for(
                    branch=cell.branch,
                    resource_alias_id=cell.resource_alias_id,
                    source_receipt_hash=cell.source_receipt_hash,
                )
            else:
                group = source_uncertainty_family.group_for_fit_source(
                    branch=cell.branch,
                    resource_alias_id=cell.resource_alias_id,
                    fit_source_receipt_hash=cell.source_receipt_hash,
                )
            expected_axis.add((cell.action_cell_hash, group.receipt_hash))
        if (
            len(receipts) != len(expected_cells)
            or any(
                type(value)
                is not V1SignedWorkVariableShellRestrictedCellOptimalityReceipt
                for value in receipts
            )
            or {
                (value.action_cell_hash, value.source_uncertainty_group_receipt_hash)
                for value in receipts
            }
            != expected_axis
            or any(
                value.registry_receipt_hash != registry.receipt.receipt_hash
                for value in receipts
            )
            or any(
                value.source_uncertainty_family_receipt_hash
                != source_uncertainty_family.receipt_hash
                for value in receipts
            )
            or len(executions) != len(receipts)
            or any(
                type(value) is not V1SignedWorkVariableShellPhaseBPDHGExecutionReceipt
                for value in executions
            )
            or {value.receipt_hash for value in executions}
            != {value.pdhg_execution_receipt_hash for value in receipts}
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_CELL_PROOF,
                "action-separation certificate omitted or mixed a closed physical-action cell",
            )
        preferred = _action_coordinate_alias(
            preferred_physical_action_coordinate_hash,
            preferred_byte_physical_action_hash,
            name="preferred physical action coordinate hash",
        )
        declared_margin = _finite_positive(
            strict_separation_margin_lower_bound,
            name="strict action-separation margin",
        )
        (
            preferred_group_uppers,
            preferred_upper,
            alternate_lower,
        ) = _aggregate_action_separation_bounds(
            observations=receipts,
            preferred_physical_action_coordinate_hash=preferred,
        )
        certified_margin = alternate_lower - preferred_upper
        if certified_margin < declared_margin:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.NONSTRICT_ACTION_SEPARATION,
                "preferred feasible F upper is not below every alternate dual F lower",
            )
        self.registry = registry
        self.source_uncertainty_family = source_uncertainty_family
        self.optimality_receipts = receipts
        self.pdhg_execution_receipts = executions
        self.preferred_physical_action_coordinate_hash = preferred
        self.preferred_source_group_F_upper_bounds = preferred_group_uppers
        self.robust_preferred_feasible_primal_F_upper_bound = preferred_upper
        self.minimum_alternate_dual_F_lower_bound = alternate_lower
        self.strict_separation_margin_lower_bound = declared_margin
        self.within_shell_proof_hash = content_hash(
            {
                "registry_receipt_hash": registry.receipt.receipt_hash,
                "source_uncertainty_family_receipt_hash": source_uncertainty_family.receipt_hash,
                "optimality_receipt_hashes": tuple(
                    value.receipt_hash for value in receipts
                ),
                "pdhg_execution_receipt_hashes": tuple(
                    value.receipt_hash for value in executions
                ),
                "preferred_physical_action_coordinate_hash": preferred,
                "strict_separation_margin_lower_bound": declared_margin,
                "scope": "conditional_on_frozen_shell_only",
            },
            prefix=_PREFIX + "within-shell-proof-",
        )
        self.outside_shell_exclusion_dependency = (
            V1SignedWorkVariableShellOutsideShellExclusionDependency(
                registry_receipt_hash=registry.receipt.receipt_hash,
                shell_freeze_receipt_hash=registry.receipt.shell_freeze_receipt_hash,
                within_shell_proof_hash=self.within_shell_proof_hash,
            )
        )
        self.within_frozen_shell_conditional_separation_certified = True
        self.action_invariance_authority = False
        self.global_next_action_invariance_authority = False
        self.outside_shell_exclusion_authority = False
        self.production_C_stream_open_authority = False
        self.calibration_authority = False
        self.oracle_authority = False
        self.commit_authority = False
        self.terminal_authority = False
        self.pareto_authority = False
        self.schema = ACTION_SEPARATION_CERTIFICATE_SCHEMA
        self.certificate_hash = content_hash(
            {
                "schema": self.schema,
                "registry_receipt_hash": registry.receipt.receipt_hash,
                "source_uncertainty_family_receipt_hash": source_uncertainty_family.receipt_hash,
                "optimality_receipt_hashes": tuple(
                    value.receipt_hash for value in receipts
                ),
                "pdhg_execution_receipt_hashes": tuple(
                    value.receipt_hash for value in executions
                ),
                "preferred_physical_action_coordinate_hash": preferred,
                "preferred_source_group_F_upper_bounds": self.preferred_source_group_F_upper_bounds,
                "robust_preferred_feasible_primal_F_upper_bound": preferred_upper,
                "minimum_alternate_dual_F_lower_bound": alternate_lower,
                "strict_separation_margin_lower_bound": declared_margin,
                "within_shell_proof_hash": self.within_shell_proof_hash,
                "outside_shell_exclusion_dependency_hash": (
                    self.outside_shell_exclusion_dependency.receipt_hash
                ),
                "within_frozen_shell_conditional_separation_certified": (
                    self.within_frozen_shell_conditional_separation_certified
                ),
                "action_invariance_authority": self.action_invariance_authority,
                "global_next_action_invariance_authority": (
                    self.global_next_action_invariance_authority
                ),
                "outside_shell_exclusion_authority": (
                    self.outside_shell_exclusion_authority
                ),
                "production_C_stream_open_authority": (
                    self.production_C_stream_open_authority
                ),
                "calibration_authority": self.calibration_authority,
                "oracle_authority": self.oracle_authority,
                "commit_authority": self.commit_authority,
                "terminal_authority": self.terminal_authority,
                "pareto_authority": self.pareto_authority,
                "separation_rule": ACTION_SEPARATION_RULE,
            },
            prefix=_PREFIX + "certificate-",
        )

    def assert_live_identity(self) -> None:
        self.registry.assert_live_sources()
        if (
            self.within_frozen_shell_conditional_separation_certified,
            self.action_invariance_authority,
            self.global_next_action_invariance_authority,
            self.outside_shell_exclusion_authority,
            self.production_C_stream_open_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        ) != (
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.AUTHORITY_VIOLATION,
                "conditional shell certificate authority fields changed after issuance",
            )
        observed = content_hash(
            {
                "schema": self.schema,
                "registry_receipt_hash": self.registry.receipt.receipt_hash,
                "source_uncertainty_family_receipt_hash": (
                    self.source_uncertainty_family.receipt_hash
                ),
                "optimality_receipt_hashes": tuple(
                    value.receipt_hash for value in self.optimality_receipts
                ),
                "pdhg_execution_receipt_hashes": tuple(
                    value.receipt_hash for value in self.pdhg_execution_receipts
                ),
                "preferred_physical_action_coordinate_hash": (
                    self.preferred_physical_action_coordinate_hash
                ),
                "preferred_source_group_F_upper_bounds": self.preferred_source_group_F_upper_bounds,
                "robust_preferred_feasible_primal_F_upper_bound": self.robust_preferred_feasible_primal_F_upper_bound,
                "minimum_alternate_dual_F_lower_bound": self.minimum_alternate_dual_F_lower_bound,
                "strict_separation_margin_lower_bound": self.strict_separation_margin_lower_bound,
                "within_shell_proof_hash": self.within_shell_proof_hash,
                "outside_shell_exclusion_dependency_hash": (
                    self.outside_shell_exclusion_dependency.receipt_hash
                ),
                "within_frozen_shell_conditional_separation_certified": (
                    self.within_frozen_shell_conditional_separation_certified
                ),
                "action_invariance_authority": self.action_invariance_authority,
                "global_next_action_invariance_authority": (
                    self.global_next_action_invariance_authority
                ),
                "outside_shell_exclusion_authority": (
                    self.outside_shell_exclusion_authority
                ),
                "production_C_stream_open_authority": (
                    self.production_C_stream_open_authority
                ),
                "calibration_authority": self.calibration_authority,
                "oracle_authority": self.oracle_authority,
                "commit_authority": self.commit_authority,
                "terminal_authority": self.terminal_authority,
                "pareto_authority": self.pareto_authority,
                "separation_rule": ACTION_SEPARATION_RULE,
            },
            prefix=_PREFIX + "certificate-",
        )
        if observed != self.certificate_hash:
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
                "action-separation certificate identity changed",
            )

    def __reduce__(self) -> NoReturn:
        raise TypeError("live action-separation certificates are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live action-separation certificates are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("live action-separation certificates cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("live action-separation certificates cannot be copied")


def issue_v1_signed_work_variable_shell_within_shell_action_separation(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
    source_uncertainty_family: (
        V1SignedWorkVariableShellSourceUncertaintyFamilyReceipt
        | V1SignedWorkVariableShellSampledSupportUncertaintyFamilyReceipt
    ),
    problem_specifications: Sequence[
        V1SignedWorkVariableShellPhaseBAliasProblemSpecification
    ],
    hbm_forecast: V1SignedWorkVariableShellActionSeparationCUDAHBMForecast,
    q_domain_bridges: Sequence[
        V1SignedWorkVariableShellRegisteredQBranchDomainBridgeReceipt
    ] = (),
) -> V1SignedWorkVariableShellActionSeparationCertificate:
    """Solve the complete registered cell product and prove shell separation.

    At present the production call refuses because acquisition has not yet
    supplied the typed relation between registered q_rho/q_d FD domains and
    fresh D_W2 literal-J rows.  Tiny exact-reference fixtures can explicitly
    define that relation as identity and exercise the complete numerical seam.
    Even a returned certificate is conditional on the frozen shell and cannot
    open C until an outside-shell exclusion certificate is independently bound.
    """

    if type(registry) is not V1SignedWorkVariableShellActionCellRegistry:
        raise TypeError("action-separation issuance requires the exact live registry")
    if type(source_uncertainty_family) not in {
        V1SignedWorkVariableShellSourceUncertaintyFamilyReceipt,
        V1SignedWorkVariableShellSampledSupportUncertaintyFamilyReceipt,
    }:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_SOURCE_UNCERTAINTY_FAMILY,
            "the current solver issuer accepts only exact-census or sampled-support source uncertainty families",
        )
    if (
        type(hbm_forecast)
        is not V1SignedWorkVariableShellActionSeparationCUDAHBMForecast
    ):
        raise TypeError("action-separation issuance requires its exact HBM forecast")
    registry.assert_live_sources()
    if (
        source_uncertainty_family.registry_receipt_hash != registry.receipt.receipt_hash
        or hbm_forecast.registry_receipt_hash != registry.receipt.receipt_hash
        or hbm_forecast.design_hash != registry.design.design_hash
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.STALE_SOURCE,
            "registry, uncertainty family, and HBM forecast do not share identity",
        )
    if not hbm_forecast.admitted:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.HBM_FORECAST_EXCEEDED,
            "complete Phase-B cell solve exceeds its pre-outcome HBM envelope",
        )
    if (
        registry.design.execution_mode
        is V1SignedWorkVariableShellActionSeparationMode.CUDA_PRODUCTION
    ):
        live_hbm = (
            build_v1_signed_work_variable_shell_action_separation_cuda_hbm_forecast(
                registry=registry,
                device_total_memory_bytes=hbm_forecast.device_total_memory_bytes,
                device_free_memory_bytes=hbm_forecast.device_free_memory_bytes,
                current_reserved_device_bytes=(
                    hbm_forecast.current_reserved_device_bytes
                ),
            )
        )
        if (
            live_hbm.bounded_solver_workspace_bytes
            != hbm_forecast.bounded_solver_workspace_bytes
            or not live_hbm.admitted
        ):
            _refuse(
                V1SignedWorkVariableShellActionSeparationRefusalCode.HBM_FORECAST_EXCEEDED,
                "live CUDA HBM facts no longer admit the sealed Phase-B solve",
            )
    specifications = tuple(problem_specifications)
    if not specifications or any(
        type(value) is not V1SignedWorkVariableShellPhaseBAliasProblemSpecification
        for value in specifications
    ):
        raise TypeError("action-separation issuance requires exact alias problems")
    expected_axis = {
        (source.branch, alias.resource_alias_id)
        for source in registry.branch_sources
        for alias in source.resource_aliases
    }
    observed_axis = {
        (value.branch, value.resource_alias_id) for value in specifications
    }
    if (
        len(specifications) != len(expected_axis)
        or observed_axis != expected_axis
        or any(
            value.registry_receipt_hash != registry.receipt.receipt_hash
            for value in specifications
        )
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.INCOMPLETE_PROBLEM_AXIS,
            "Phase-B problem specifications do not cover every branch/resource alias once",
        )
    specification_by_axis = {
        (value.branch, value.resource_alias_id): value for value in specifications
    }
    supplied_bridges = tuple(q_domain_bridges)
    if any(
        type(value) is not V1SignedWorkVariableShellRegisteredQBranchDomainBridgeReceipt
        for value in supplied_bridges
    ):
        raise TypeError("Q-domain bridge axis contains another receipt type")
    bridge_by_request = {value.request_hash: value for value in supplied_bridges}
    if len(bridge_by_request) != len(supplied_bridges):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
            "Q-domain bridge axis contains duplicate cell requests",
        )
    if (
        registry.design.execution_mode
        is V1SignedWorkVariableShellActionSeparationMode.CUDA_PRODUCTION
        and not supplied_bridges
    ):
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.REGISTERED_Q_BRANCH_DOMAIN_BRIDGE_UNAVAILABLE,
            "registered q_rho/q_d FD feasibility has no live source-bound enclosure in D_W2 literal-J coordinates",
        )

    optimality_receipts: list[
        V1SignedWorkVariableShellRestrictedCellOptimalityReceipt
    ] = []
    execution_receipts: list[V1SignedWorkVariableShellPhaseBPDHGExecutionReceipt] = []
    compact_sources: dict[
        V1SignedWorkVariableShellQBranch,
        V1SignedWorkVariableShellPhaseBCompactSourceHandle,
    ] = {}
    try:
        for branch_source in registry.branch_sources:
            compact_sources[branch_source.branch] = (
                build_v1_signed_work_variable_shell_phase_b_compact_source(
                    branch_source=branch_source
                )
            )
        for action_cell in registry.iter_action_cells():
            specification = specification_by_axis[
                (action_cell.branch, action_cell.resource_alias_id)
            ]
            branch_source = registry.branch_source(action_cell.branch)
            alias = branch_source.resource_alias(action_cell.resource_alias_id)
            if (
                specification.phase_a_invariance_receipt_hash
                != alias.phase_a_invariance_receipt_hash
                or len(specification.shell_resource_coefficients)
                != branch_source.registration.shell_size
            ):
                _refuse(
                    V1SignedWorkVariableShellActionSeparationRefusalCode.AXIS_MISMATCH,
                    "alias problem changed its Phase-A or shell coordinate",
                )
            request = build_v1_signed_work_variable_shell_restricted_cell_request(
                registry=registry,
                branch=action_cell.branch,
                resource_alias_id=action_cell.resource_alias_id,
                action_cell_ordinal=action_cell.action_cell_ordinal,
                shell_resource_coefficients=(specification.shell_resource_coefficients),
                literal_j_response_radius=(specification.literal_j_response_radius),
                block_capacities=specification.block_capacities,
            )
            bridge = bridge_by_request.get(request.request_hash)
            if bridge is None:
                if (
                    type(source_uncertainty_family)
                    is V1SignedWorkVariableShellSampledSupportUncertaintyFamilyReceipt
                ):
                    _refuse(
                        V1SignedWorkVariableShellActionSeparationRefusalCode.REGISTERED_Q_BRANCH_DOMAIN_BRIDGE_UNAVAILABLE,
                        "sampled-support source uncertainty requires an explicit registered-Q branch domain bridge",
                    )
                if (
                    registry.design.execution_mode
                    is V1SignedWorkVariableShellActionSeparationMode.EXACT_REFERENCE_TEST_ONLY
                ):
                    bridge = issue_v1_signed_work_variable_shell_exact_reference_q_domain_identity(
                        registry=registry,
                        request=request,
                    )
                else:
                    _refuse(
                        V1SignedWorkVariableShellActionSeparationRefusalCode.REGISTERED_Q_BRANCH_DOMAIN_BRIDGE_UNAVAILABLE,
                        "one production cell lacks a registered-Q branch domain bridge",
                    )
            if (
                type(source_uncertainty_family)
                is V1SignedWorkVariableShellSourceUncertaintyFamilyReceipt
            ):
                group = source_uncertainty_family.group_for(
                    branch=action_cell.branch,
                    resource_alias_id=action_cell.resource_alias_id,
                    source_receipt_hash=action_cell.source_receipt_hash,
                )
            else:
                group = source_uncertainty_family.group_for_fit_source(
                    branch=action_cell.branch,
                    resource_alias_id=action_cell.resource_alias_id,
                    fit_source_receipt_hash=action_cell.source_receipt_hash,
                )
            optimality, execution = (
                _solve_v1_signed_work_variable_shell_phase_b_cell_pdhg(
                    registry=registry,
                    request=request,
                    compact_source=compact_sources[action_cell.branch],
                    source_uncertainty_family=source_uncertainty_family,
                    source_uncertainty_group=group,
                    q_domain_bridge=bridge,
                )
            )
            optimality_receipts.append(optimality)
            execution_receipts.append(execution)
    finally:
        for value in compact_sources.values():
            value.close()

    candidates: list[tuple[str, float]] = []
    for action_hash in sorted(
        {value.physical_action_coordinate_hash for value in optimality_receipts}
    ):
        try:
            _groups, preferred_upper, alternate_lower = (
                _aggregate_action_separation_bounds(
                    observations=optimality_receipts,
                    preferred_physical_action_coordinate_hash=action_hash,
                )
            )
        except V1SignedWorkVariableShellActionSeparationRefusal:
            continue
        margin = alternate_lower - preferred_upper
        if margin > 0.0:
            candidates.append((action_hash, margin))
    if len(candidates) != 1:
        _refuse(
            V1SignedWorkVariableShellActionSeparationRefusalCode.NONSTRICT_ACTION_SEPARATION,
            "complete robust source groups do not select exactly one physical shell action",
        )
    preferred, margin = candidates[0]
    return V1SignedWorkVariableShellActionSeparationCertificate(
        registry=registry,
        source_uncertainty_family=source_uncertainty_family,
        optimality_receipts=tuple(optimality_receipts),
        pdhg_execution_receipts=tuple(execution_receipts),
        preferred_physical_action_coordinate_hash=preferred,
        strict_separation_margin_lower_bound=math.nextafter(margin, 0.0),
        _issuer_token=_CERTIFICATE_TOKEN,
    )


def refuse_v1_signed_work_variable_shell_outside_shell_exclusion_unavailable(
    *,
    certificate: V1SignedWorkVariableShellActionSeparationCertificate,
) -> NoReturn:
    if type(certificate) is not V1SignedWorkVariableShellActionSeparationCertificate:
        raise TypeError(
            "outside-shell refusal requires the live conditional certificate"
        )
    certificate.assert_live_identity()
    _refuse(
        V1SignedWorkVariableShellActionSeparationRefusalCode.OUTSIDE_SHELL_EXCLUSION_UNAVAILABLE,
        "within-shell cell separation is proved, but no live source-bound screen excludes every outside coordinate from every admissible next rounded action; global invariance and C remain closed",
    )


def refuse_v1_signed_work_variable_shell_global_restricted_solver_unavailable(
    *,
    registry: V1SignedWorkVariableShellActionCellRegistry,
) -> NoReturn:
    if type(registry) is not V1SignedWorkVariableShellActionCellRegistry:
        raise TypeError("solver refusal requires the exact live registry")
    registry.assert_live_sources()
    _refuse(
        V1SignedWorkVariableShellActionSeparationRefusalCode.GLOBAL_RESTRICTED_SOLVER_UNAVAILABLE,
        "the <=2^B cell registry, exact fiber quotient, and deterministic PDHG "
        "joint are landed, but production still requires Q-dropped alternate "
        "domains plus original registered-q_U preferred replay and an outside-"
        "shell exclusion theorem; global action invariance and C remain closed",
    )


__all__ = [
    "ACTION_CELL_COVER_RULE",
    "ACTION_CELL_IMAGE_RULE",
    "ACTION_LEXICOGRAPHIC_RULE",
    "ACTION_PHASE_B_DUAL_LOWER_BOUND_RULE",
    "ACTION_PHASE_B_DUAL_PROJECTION_RULE",
    "ACTION_Q_DOMAIN_HIERARCHY_RULE",
    "ACTION_REFERENCE_PERMUTATION_RULE",
    "ACTION_RESTRICTED_PROBLEM_RULE",
    "ACTION_SEPARATION_AUTHORITY_SCOPE",
    "ACTION_SEPARATION_RULE",
    "V1SignedWorkVariableShellActionCellBlockCapacity",
    "V1SignedWorkVariableShellActionCellIdentity",
    "V1SignedWorkVariableShellActionCellRegistry",
    "V1SignedWorkVariableShellActionCellRegistryReceipt",
    "V1SignedWorkVariableShellActionCoordinateCustodyIssuance",
    "V1SignedWorkVariableShellActionSeparationBoundObservation",
    "V1SignedWorkVariableShellActionSeparationCUDAExecutionDesign",
    "V1SignedWorkVariableShellActionSeparationCUDAHBMForecast",
    "V1SignedWorkVariableShellActionSeparationCertificate",
    "V1SignedWorkVariableShellActionSeparationMode",
    "V1SignedWorkVariableShellActionSeparationQuantifierReference",
    "V1SignedWorkVariableShellActionSeparationRefusal",
    "V1SignedWorkVariableShellActionSeparationRefusalCode",
    "V1SignedWorkVariableShellActionSubsetIdentity",
    "V1SignedWorkVariableShellBranchActionCellRegistration",
    "V1SignedWorkVariableShellBranchActionCellSource",
    "V1SignedWorkVariableShellClosedOrderCell",
    "V1SignedWorkVariableShellClosedSubsetActionCell",
    "V1SignedWorkVariableShellPhaseBDualLowerBoundContract",
    "V1SignedWorkVariableShellPhaseBDualLowerBoundReference",
    "V1SignedWorkVariableShellPhaseBAliasProblemSpecification",
    "V1SignedWorkVariableShellPhaseBCompactEvaluationReceipt",
    "V1SignedWorkVariableShellPhaseBCompactSourceHandle",
    "V1SignedWorkVariableShellPhaseBPDHGExecutionReceipt",
    "V1SignedWorkVariableShellPhaseBPrimalReference",
    "V1SignedWorkVariableShellPhaseBSourceCompactionReceipt",
    "V1SignedWorkVariableShellQBranch",
    "V1SignedWorkVariableShellRegisteredQBranchDomainBridgeReceipt",
    "V1SignedWorkVariableShellResourceAlias",
    "V1SignedWorkVariableShellRestrictedCellOptimalityReceipt",
    "V1SignedWorkVariableShellRestrictedConvexCellRequest",
    "V1SignedWorkVariableShellSampledSupportUncertaintyFamilyReceipt",
    "V1SignedWorkVariableShellSampledSupportUncertaintyGroupReceipt",
    "V1SignedWorkVariableShellSourceUncertaintyFamilyReceipt",
    "V1SignedWorkVariableShellSourceUncertaintyGroupReceipt",
    "V1SignedWorkVariableShellOutsideShellExclusionDependency",
    "V1SignedWorkVariableShellStreamedHashCustodyFamilyReceipt",
    "build_v1_signed_work_variable_shell_action_cell_registry",
    "build_v1_signed_work_variable_shell_action_separation_cuda_hbm_forecast",
    "build_v1_signed_work_variable_shell_phase_b_dual_lower_bound_contract",
    "build_v1_signed_work_variable_shell_phase_b_compact_source",
    "build_v1_signed_work_variable_shell_restricted_cell_request",
    "enumerate_v1_signed_work_variable_shell_closed_order_cells",
    "enumerate_v1_signed_work_variable_shell_closed_order_cells_reference_only",
    "enumerate_v1_signed_work_variable_shell_closed_subset_action_cells",
    "evaluate_v1_signed_work_variable_shell_action_separation_quantifiers_reference_only",
    "evaluate_v1_signed_work_variable_shell_phase_b_dual_lower_bound_reference",
    "evaluate_v1_signed_work_variable_shell_phase_b_primal_reference",
    "issue_v1_signed_work_variable_shell_sampled_support_uncertainty_family",
    "issue_v1_signed_work_variable_shell_action_coordinate_custody",
    "issue_v1_signed_work_variable_shell_exact_census_uncertainty_family",
    "issue_v1_signed_work_variable_shell_exact_reference_q_domain_identity",
    "issue_v1_signed_work_variable_shell_within_shell_action_separation",
    "refuse_v1_signed_work_variable_shell_global_restricted_solver_unavailable",
    "refuse_v1_signed_work_variable_shell_outside_shell_exclusion_unavailable",
]
