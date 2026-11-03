"""Live common-D-source to origin-branch-Q projection binding.

The sampled-V7 D source is a common, two-branch source while registered hard-Q
is evaluated on a branch-local source.  Their receipt hashes are intentionally
different.  This module proves the narrower live-source projection: the exact
V7 conditioned-D1 rows retained by the full-population composite are the rows
resident in the Phase-A origin-Q source, on the same D/capture/literal-J axes.

Only projection-binding authority is granted.  F/J alpha-family closure,
continuum stability, D decision stability, C/K/O, commit, terminal,
performance, and Pareto authority remain closed.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_sampled_full_population_compact_source import (
    V1SampledFullPopulationCompactSourceHandle,
)
from .modern_v1_sampled_v7_d_fixed_source import V1SampledV7D1FixedSourceResult
from .modern_v1_signed_work_variable_shell_action_separation_cuda import (
    V1SignedWorkVariableShellActionCellRegistry,
    V1SignedWorkVariableShellQBranch,
)
from .modern_v1_signed_work_variable_shell_registered_q_bridge import (
    V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt,
)
from .modern_v1_signed_work_variable_shell_source_cuda import (
    V1SignedWorkVariableShellSourceDeviceHandle,
)


SAMPLED_V7_COMMON_D_ORIGIN_Q_LIVE_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_common_D_to_origin_branch_Q_live_projection_v1"
)
SAMPLED_V7_COMMON_D_ORIGIN_Q_LIVE_PROJECTION_AUTHORITY: Final = (
    "exact_live_common_D1_to_phase_A_origin_branch_registered_Q_source_"
    "projection_binding_only_no_F_or_J_alpha_continuum_D_stability_C_K_O_"
    "commit_terminal_performance_or_pareto_authority_v1"
)
SAMPLED_V7_COMMON_D_ORIGIN_Q_LIVE_PROJECTION_RULE: Final = (
    "retain_distinct_common_and_branch_receipts_while_replaying_exact_live_"
    "compact_rows_D_prefix_capture_VJP_chart_route_shell_population_literal_J_"
    "phase_A_action_and_registered_origin_Q_lineage_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-common-D-origin-Q-live-projection-"
_ISSUER: Final = object()
_COMPACT_TENSOR_ROLES: Final = (
    "fixed_outside_fiber_center",
    "fixed_outside_fiber_radius",
    "shell_corrected_center",
    "fixed_outside_literal_j_response",
    "fixed_outside_literal_j_radius",
    "shell_literal_j_response",
    "context_weight",
    "shell_native_fiber",
)


def _compact_tensor_exactly_equal(left: object, right: object) -> bool:
    """Replay exact compact equality without an avoidable device-wide scan.

    Live source composition normally preserves the same tensor objects (or
    exact views of the same storage).  Those cases are identity proofs and do
    not require launching eight equality kernels per source pair.  A genuine
    distinct allocation uses the framework's exact equality primitive as the
    conservative fallback.
    """

    if (
        type(left) is not type(right)
        or getattr(left, "dtype", None) != getattr(right, "dtype", None)
        or getattr(left, "device", None) != getattr(right, "device", None)
        or tuple(getattr(left, "shape", ())) != tuple(getattr(right, "shape", ()))
    ):
        return False
    if left is right:
        return True
    left_ptr = getattr(left, "data_ptr", None)
    right_ptr = getattr(right, "data_ptr", None)
    left_stride = getattr(left, "stride", None)
    right_stride = getattr(right, "stride", None)
    if (
        callable(left_ptr)
        and callable(right_ptr)
        and callable(left_stride)
        and callable(right_stride)
        and int(left_ptr()) == int(right_ptr())
        and tuple(left_stride()) == tuple(right_stride())
        and int(getattr(left, "storage_offset", lambda: 0)())
        == int(getattr(right, "storage_offset", lambda: 0)())
    ):
        return True
    exact_equal = getattr(left, "equal", None)
    if callable(exact_equal):
        return bool(exact_equal(right))
    return bool((left == right).all().item())  # type: ignore[operator]


class V1SampledV7CommonDOriginQLiveProjectionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    D1_COMPOSITE_MISMATCH = "D1_composite_mismatch"
    COMPACT_ROW_MISMATCH = "compact_row_mismatch"
    PREFIX_MISMATCH = "prefix_mismatch"
    CHART_OR_ROUTE_MISMATCH = "chart_or_route_mismatch"
    LITERAL_J_LINEAGE_MISMATCH = "literal_J_lineage_mismatch"
    PHASE_A_SOURCE_MISMATCH = "phase_A_source_mismatch"
    BRANCH_MISMATCH = "branch_mismatch"
    ACTION_MISMATCH = "action_mismatch"
    REGISTERED_Q_MISMATCH = "registered_Q_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SampledV7CommonDOriginQLiveProjectionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7CommonDOriginQLiveProjectionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7CommonDOriginQLiveProjectionRefusalCode:
            raise TypeError("common-D/origin-Q live refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7CommonDOriginQLiveProjectionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7CommonDOriginQLiveProjectionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _phase_a_branch(branch: V1RelaxedKKTBranch) -> V1SignedWorkVariableShellQBranch:
    if type(branch) is not V1RelaxedKKTBranch:
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.INVALID_TYPE,
            "registered-Q branch must be exact",
        )
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        return V1SignedWorkVariableShellQBranch.SPECTRAL
    if branch is V1RelaxedKKTBranch.DIAGONAL_L1:
        return V1SignedWorkVariableShellQBranch.DIAGONAL_L1
    _refuse(
        V1SampledV7CommonDOriginQLiveProjectionRefusalCode.BRANCH_MISMATCH,
        "registered-Q branch has no Phase-A descriptor",
    )


def _fresh_live_parents(
    *,
    d1_fixed_source: V1SampledV7D1FixedSourceResult,
    common_source: V1SampledFullPopulationCompactSourceHandle,
    phase_a_registry: V1SignedWorkVariableShellActionCellRegistry,
    origin_q_source: V1SignedWorkVariableShellSourceDeviceHandle,
    registered_q_bridge: (
        V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt
    ),
):
    exact = (
        (d1_fixed_source, V1SampledV7D1FixedSourceResult),
        (common_source, V1SampledFullPopulationCompactSourceHandle),
        (phase_a_registry, V1SignedWorkVariableShellActionCellRegistry),
        (origin_q_source, V1SignedWorkVariableShellSourceDeviceHandle),
        (
            registered_q_bridge,
            V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt,
        ),
    )
    if any(type(value) is not expected for value, expected in exact):
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.INVALID_TYPE,
            "live projection issuance requires exact live parent types",
        )
    try:
        d1_receipt = d1_fixed_source.receipt
        d1_conditioned = d1_fixed_source.conditioned_d1_source
        d1_conditioned.receipt.assert_integrity()
        d1_device = d1_conditioned.source_handle
        d1_device_receipt = d1_device.receipt
        projection_registration = d1_fixed_source.literal_j_projection.registration
        projection_registration.assert_integrity()
        projection_receipt = d1_fixed_source.literal_j_projection.receipt
        projection_receipt.assert_integrity()

        common_receipt = common_source.receipt
        common_conditioned = common_source.conditioned_d1_source
        common_conditioned.receipt.assert_integrity()
        common_device = common_conditioned.source_handle
        common_device_receipt = common_device.receipt

        origin_receipt = origin_q_source.receipt
        registry_handles = phase_a_registry.assert_live_sources()
        branch = _phase_a_branch(registered_q_bridge.branch)
        branch_source = phase_a_registry.branch_source(branch)
        branch_handle = branch_source.assert_live_source()
        bridge_hash = registered_q_bridge.receipt_hash
    except V1SampledV7CommonDOriginQLiveProjectionRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.STALE_SOURCE,
            f"one live parent failed replay: {type(error).__name__}: {error}",
        )

    if (
        d1_receipt.conditioned_d1_source_receipt_hash
        != d1_conditioned.receipt.receipt_hash
        or d1_receipt.conditioned_d1_compact_source_receipt_hash
        != d1_device_receipt.receipt_hash
        or projection_registration.full_population_compact_source_receipt_hash
        != d1_device_receipt.receipt_hash
        or projection_receipt.conditioned_d1_replay_receipt_hash
        != d1_conditioned.receipt.receipt_hash
        or common_conditioned.receipt.row_prefix_manifest_hash
        != d1_conditioned.receipt.row_prefix_manifest_hash
        or common_conditioned.receipt.original_capture_binding_hash
        != d1_conditioned.receipt.original_capture_binding_hash
    ):
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.D1_COMPOSITE_MISMATCH,
            "full-population composite does not retain the same D1 row source",
        )

    common_chart = (
        common_device_receipt.model_identity_hash,
        common_device_receipt.chart_identity_hash,
        common_device_receipt.current_mask_hash,
        common_device_receipt.proposal_generation,
        common_device_receipt.route_axis_hash,
        common_device_receipt.route_id_axis_hash,
        common_device_receipt.route_count,
        common_device_receipt.shell_freeze_receipt_hash,
        common_device_receipt.shell_indices,
        common_device_receipt.shell_route_ids,
        common_device_receipt.shell_native_fiber_indices,
        common_device_receipt.native_fiber_axis_receipt_hash,
        common_device_receipt.native_fiber_measure_hash,
    )
    d1_chart = (
        d1_device_receipt.model_identity_hash,
        d1_device_receipt.chart_identity_hash,
        d1_device_receipt.current_mask_hash,
        d1_device_receipt.proposal_generation,
        d1_device_receipt.route_axis_hash,
        d1_device_receipt.route_id_axis_hash,
        d1_device_receipt.route_count,
        d1_device_receipt.shell_freeze_receipt_hash,
        d1_device_receipt.shell_indices,
        d1_device_receipt.shell_route_ids,
        d1_device_receipt.shell_native_fiber_indices,
        d1_device_receipt.native_fiber_axis_receipt_hash,
        d1_device_receipt.native_fiber_measure_hash,
    )
    origin_chart = (
        origin_receipt.model_identity_hash,
        origin_receipt.chart_identity_hash,
        origin_receipt.current_mask_hash,
        origin_receipt.proposal_generation,
        origin_receipt.route_axis_hash,
        origin_receipt.route_id_axis_hash,
        origin_receipt.route_count,
        origin_receipt.shell_freeze_receipt_hash,
        origin_receipt.shell_indices,
        origin_receipt.shell_route_ids,
        origin_receipt.shell_native_fiber_indices,
        origin_receipt.native_fiber_axis_receipt_hash,
        origin_receipt.native_fiber_measure_hash,
    )
    if common_chart != d1_chart or common_chart != origin_chart:
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.CHART_OR_ROUTE_MISMATCH,
            "common D1, V7 D1, and origin Q do not share a chart/route/shell/native axis",
        )

    if (
        common_device_receipt.registered_d_prefix_hash
        != d1_device_receipt.registered_d_prefix_hash
        or common_device_receipt.registered_d_prefix_hash
        != origin_receipt.registered_d_prefix_hash
        or common_device_receipt.registered_d_w2_prefix_hash
        != d1_device_receipt.registered_d_w2_prefix_hash
        or common_device_receipt.registered_d_w2_prefix_hash
        != origin_receipt.registered_d_w2_prefix_hash
        or common_conditioned.receipt.registered_d_prefix_hash
        != d1_conditioned.receipt.registered_d_prefix_hash
        or common_conditioned.receipt.registered_d_prefix_hash
        != origin_receipt.registered_d_prefix_hash
        or common_conditioned.receipt.d1_pair_prefix_axis_hash
        != d1_conditioned.receipt.d1_pair_prefix_axis_hash
        or common_conditioned.receipt.d1_pair_prefix_axis_hash
        != origin_receipt.registered_d_w2_prefix_hash
        or common_receipt.d1_pair_prefix_axis_hash
        != origin_receipt.registered_d_w2_prefix_hash
        or projection_registration.registered_d_prefix_hash
        != origin_receipt.registered_d_prefix_hash
        or projection_registration.registered_d_w2_prefix_hash
        != origin_receipt.registered_d_w2_prefix_hash
    ):
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.PREFIX_MISMATCH,
            "V7 D1, common composite, and origin Q do not share one D prefix",
        )

    common_lineage = (
        common_device_receipt.same_capture_vjp_source_hash,
        common_device_receipt.literal_j_stream_source_hash,
        common_device_receipt.population_context_axis_hash,
        common_device_receipt.population_draw_axis_hash,
        common_device_receipt.context_axis_hash,
        common_device_receipt.draw_axis_hash,
        common_device_receipt.measurement_axis_hash,
        common_device_receipt.capture_axis_hash,
        common_device_receipt.source_packet_axis_hash,
        common_device_receipt.context_count,
        common_device_receipt.population_count,
    )
    d1_lineage = (
        d1_device_receipt.same_capture_vjp_source_hash,
        d1_device_receipt.literal_j_stream_source_hash,
        d1_device_receipt.population_context_axis_hash,
        d1_device_receipt.population_draw_axis_hash,
        d1_device_receipt.context_axis_hash,
        d1_device_receipt.draw_axis_hash,
        d1_device_receipt.measurement_axis_hash,
        d1_device_receipt.capture_axis_hash,
        d1_device_receipt.source_packet_axis_hash,
        d1_device_receipt.context_count,
        d1_device_receipt.population_count,
    )
    origin_lineage = (
        origin_receipt.same_capture_vjp_source_hash,
        origin_receipt.literal_j_stream_source_hash,
        origin_receipt.population_context_axis_hash,
        origin_receipt.population_draw_axis_hash,
        origin_receipt.context_axis_hash,
        origin_receipt.draw_axis_hash,
        origin_receipt.measurement_axis_hash,
        origin_receipt.capture_axis_hash,
        origin_receipt.source_packet_axis_hash,
        origin_receipt.context_count,
        origin_receipt.population_count,
    )
    if common_lineage != d1_lineage or common_lineage != origin_lineage:
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.LITERAL_J_LINEAGE_MISMATCH,
            "common D1, V7 D1, and origin Q do not share capture/literal-J population axes",
        )
    if (
        projection_registration.model_identity_hash != origin_receipt.model_identity_hash
        or projection_registration.chart_identity_hash
        != origin_receipt.chart_identity_hash
        or projection_registration.current_mask_hash != origin_receipt.current_mask_hash
        or projection_registration.proposal_generation
        != origin_receipt.proposal_generation
        or projection_registration.route_axis_hash != origin_receipt.route_axis_hash
        or projection_registration.route_id_axis_hash
        != origin_receipt.route_id_axis_hash
        or projection_registration.route_count != origin_receipt.route_count
        or projection_registration.composite_same_capture_vjp_source_hash
        != origin_receipt.same_capture_vjp_source_hash
        or projection_registration.composite_literal_j_stream_source_hash
        != origin_receipt.literal_j_stream_source_hash
    ):
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.LITERAL_J_LINEAGE_MISMATCH,
            "V7 literal-J projection registration names another D1 source",
        )

    purpose = content_hash(
        {
            "schema": SAMPLED_V7_COMMON_D_ORIGIN_Q_LIVE_PROJECTION_SCHEMA,
            "common_source_receipt_hash": common_receipt.receipt_hash,
            "origin_source_receipt_hash": origin_receipt.receipt_hash,
        },
        prefix=_PREFIX + "borrow-purpose-",
    )
    try:
        with common_device.borrow_compact_tensors(
            borrow_purpose_hash=purpose + ":common"
        ) as common_lease, d1_device.borrow_compact_tensors(
            borrow_purpose_hash=purpose + ":v7-d1"
        ) as d1_lease, origin_q_source.borrow_compact_tensors(
            borrow_purpose_hash=purpose + ":origin"
        ) as origin_lease:
            common_tensors = common_lease.compact_tensors()
            d1_tensors = d1_lease.compact_tensors()
            origin_tensors = origin_lease.compact_tensors()
            if (
                len(common_tensors) != len(_COMPACT_TENSOR_ROLES)
                or len(d1_tensors) != len(_COMPACT_TENSOR_ROLES)
                or len(origin_tensors) != len(_COMPACT_TENSOR_ROLES)
            ):
                _refuse(
                    V1SampledV7CommonDOriginQLiveProjectionRefusalCode.COMPACT_ROW_MISMATCH,
                    "compact-source ABI changed",
                )
            d1_equal = tuple(
                _compact_tensor_exactly_equal(left, right)
                for left, right in zip(common_tensors, d1_tensors, strict=True)
            )
            origin_equal = tuple(
                _compact_tensor_exactly_equal(left, right)
                for left, right in zip(common_tensors, origin_tensors, strict=True)
            )
    except V1SampledV7CommonDOriginQLiveProjectionRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.STALE_SOURCE,
            f"compact-row replay failed: {type(error).__name__}: {error}",
        )
    if not all(d1_equal) or not all(origin_equal):
        roles = tuple(
            role
            for role, same in zip(
                _COMPACT_TENSOR_ROLES,
                tuple(a and b for a, b in zip(d1_equal, origin_equal, strict=True)),
                strict=True,
            )
            if not same
        )
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.COMPACT_ROW_MISMATCH,
            f"common D1, V7 D1, and origin Q differ on compact roles {roles}",
        )

    branch_index = phase_a_registry.receipt.branch_axis.index(branch)
    if (
        branch_handle is not origin_q_source
        or origin_q_source not in registry_handles
        or branch_source.registration.source_receipt_hash
        != origin_receipt.receipt_hash
        or branch_source.registration.source_replay_hash
        != origin_receipt.source_replay_hash
        or phase_a_registry.receipt.source_receipt_hashes[branch_index]
        != origin_receipt.receipt_hash
        or phase_a_registry.receipt.source_replay_hashes[branch_index]
        != origin_receipt.source_replay_hash
    ):
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.PHASE_A_SOURCE_MISMATCH,
            "origin Q is not the exact live source on its Phase-A branch axis",
        )

    aliases = branch_source.resource_aliases
    if (
        not aliases
        or any(alias.branch is not branch for alias in aliases)
        or any(
            alias.source_kkt_branch_id != origin_receipt.kkt_branch_id
            or alias.same_capture_vjp_source_hash
            != origin_receipt.same_capture_vjp_source_hash
            or alias.literal_j_stream_source_hash
            != origin_receipt.literal_j_stream_source_hash
            for alias in aliases
        )
    ):
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.PHASE_A_SOURCE_MISMATCH,
            "Phase-A aliases no longer bind the origin-Q source",
        )

    subset_matches = tuple(
        subset
        for subset in branch_source.subset_identities
        if subset.subset_hash == registered_q_bridge.action_subset_identity_hash
        and subset.physical_action_coordinate_hash
        == registered_q_bridge.physical_action_coordinate_hash
    )
    if (
        len(subset_matches) != 1
        or branch_source.registration.streamed_hash_custody_family_hash
        != registered_q_bridge.action_coordinate_family_receipt_hash
    ):
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.ACTION_MISMATCH,
            "registered-Q action is absent from the Phase-A subset axis",
        )

    if (
        registered_q_bridge.source_receipt_hash != origin_receipt.receipt_hash
        or registered_q_bridge.source_replay_hash != origin_receipt.source_replay_hash
        or registered_q_bridge.model_identity_hash != origin_receipt.model_identity_hash
        or registered_q_bridge.chart_identity_hash != origin_receipt.chart_identity_hash
        or registered_q_bridge.parent_mask_hash != origin_receipt.current_mask_hash
        or registered_q_bridge.proposal_generation
        != origin_receipt.proposal_generation
        or registered_q_bridge.route_axis_hash != origin_receipt.route_axis_hash
        or registered_q_bridge.route_id_axis_hash != origin_receipt.route_id_axis_hash
        or registered_q_bridge.route_count != origin_receipt.route_count
        or registered_q_bridge.registered_d_prefix_hash
        != origin_receipt.registered_d_prefix_hash
        or registered_q_bridge.registered_d_w2_prefix_hash
        != origin_receipt.registered_d_w2_prefix_hash
        or registered_q_bridge.shell_freeze_receipt_hash
        != origin_receipt.shell_freeze_receipt_hash
        or registered_q_bridge.source_kkt_branch_id != origin_receipt.kkt_branch_id
        or registered_q_bridge.source_kkt_branch_receipt_hash
        != origin_receipt.kkt_branch_source_receipt_hash
        or registered_q_bridge.branch.value != branch.value
        or registered_q_bridge.origin_branch_specific_q_used is not True
        or registered_q_bridge.strict_q_budget_inequality_used is not True
        or not registered_q_bridge.origin_branch_q_upper
        < registered_q_bridge.quadratic_budget
    ):
        _refuse(
            V1SampledV7CommonDOriginQLiveProjectionRefusalCode.REGISTERED_Q_MISMATCH,
            "registered-Q bridge does not replay the origin branch/action/Q boundary",
        )

    return (
        d1_receipt,
        common_receipt,
        common_conditioned,
        origin_receipt,
        branch_source,
        bridge_hash,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7CommonDOriginQLiveProjectionBindingReceipt:
    v7_d1_fixed_source_receipt_hash: str
    v7_d1_conditioned_source_receipt_hash: str
    v7_d1_conditioned_device_source_receipt_hash: str
    v7_d1_literal_j_projection_registration_hash: str
    v7_d1_literal_j_projection_receipt_hash: str
    common_source_receipt_hash: str
    common_source_replay_hash: str
    conditioned_d1_row_prefix_manifest_hash: str
    conditioned_d1_original_capture_binding_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    shell_freeze_receipt_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    same_capture_vjp_source_hash: str
    literal_j_stream_source_hash: str
    phase_a_registry_receipt_hash: str
    phase_a_branch_registration_hash: str
    phase_a_branch: V1SignedWorkVariableShellQBranch
    phase_a_resource_alias_hashes: tuple[str, ...]
    origin_q_source_receipt_hash: str
    origin_q_source_replay_hash: str
    origin_q_source_kkt_receipt_hash: str
    registered_q_bridge_receipt_hash: str
    physical_action_coordinate_hash: str
    action_subset_identity_hash: str
    compact_row_projection_hash: str
    projection_binding_hash: str
    receipt_hash: str = field(init=False)
    common_and_origin_receipt_hashes_distinct_allowed: bool = True
    exact_live_compact_rows_replayed: bool = True
    exact_d_prefix_bound: bool = True
    exact_capture_vjp_literal_j_lineage_bound: bool = True
    exact_phase_a_branch_source_axis_bound: bool = True
    exact_registered_q_action_ancestry_bound: bool = True
    common_d_origin_q_projection_binding_authority: bool = True
    f_alpha_family_authority: bool = False
    j_alpha_family_authority: bool = False
    same_source_f_q_witness_authority: bool = False
    continuum_stability_authority: bool = False
    d_decision_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    rule: str = SAMPLED_V7_COMMON_D_ORIGIN_Q_LIVE_PROJECTION_RULE
    authority_scope: str = SAMPLED_V7_COMMON_D_ORIGIN_Q_LIVE_PROJECTION_AUTHORITY
    schema: str = SAMPLED_V7_COMMON_D_ORIGIN_Q_LIVE_PROJECTION_SCHEMA
    _d1_fixed_source: V1SampledV7D1FixedSourceResult = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _common_source: V1SampledFullPopulationCompactSourceHandle = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _phase_a_registry: V1SignedWorkVariableShellActionCellRegistry = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _origin_q_source: V1SignedWorkVariableShellSourceDeviceHandle = field(
        default=None, repr=False, compare=False  # type: ignore[assignment]
    )
    _registered_q_bridge: (
        V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt
    ) = field(default=None, repr=False, compare=False)  # type: ignore[assignment]
    _issuer_token: InitVar[object | None] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object | None) -> None:
        if _issuer_token is not _ISSUER:
            raise TypeError("live common-D/origin-Q receipts are issuer-only")
        if type(self.phase_a_branch) is not V1SignedWorkVariableShellQBranch:
            raise TypeError("live projection Phase-A branch must be exact")
        for name in (
            "v7_d1_fixed_source_receipt_hash",
            "v7_d1_conditioned_source_receipt_hash",
            "v7_d1_conditioned_device_source_receipt_hash",
            "v7_d1_literal_j_projection_registration_hash",
            "v7_d1_literal_j_projection_receipt_hash",
            "common_source_receipt_hash",
            "common_source_replay_hash",
            "conditioned_d1_row_prefix_manifest_hash",
            "conditioned_d1_original_capture_binding_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "shell_freeze_receipt_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "same_capture_vjp_source_hash",
            "literal_j_stream_source_hash",
            "phase_a_registry_receipt_hash",
            "phase_a_branch_registration_hash",
            "origin_q_source_receipt_hash",
            "origin_q_source_replay_hash",
            "origin_q_source_kkt_receipt_hash",
            "registered_q_bridge_receipt_hash",
            "physical_action_coordinate_hash",
            "action_subset_identity_hash",
            "compact_row_projection_hash",
            "projection_binding_hash",
            "rule",
            "authority_scope",
            "schema",
        ):
            _identity(getattr(self, name), name=name)
        if (
            isinstance(self.proposal_generation, bool)
            or not isinstance(self.proposal_generation, int)
            or self.proposal_generation < 0
            or isinstance(self.route_count, bool)
            or not isinstance(self.route_count, int)
            or self.route_count <= 0
            or type(self.phase_a_resource_alias_hashes) is not tuple
            or not self.phase_a_resource_alias_hashes
            or any(type(value) is not str or not value for value in self.phase_a_resource_alias_hashes)
        ):
            raise TypeError("live projection receipt has an invalid finite axis")
        required_true = (
            self.common_and_origin_receipt_hashes_distinct_allowed,
            self.exact_live_compact_rows_replayed,
            self.exact_d_prefix_bound,
            self.exact_capture_vjp_literal_j_lineage_bound,
            self.exact_phase_a_branch_source_axis_bound,
            self.exact_registered_q_action_ancestry_bound,
            self.common_d_origin_q_projection_binding_authority,
        )
        required_false = (
            self.f_alpha_family_authority,
            self.j_alpha_family_authority,
            self.same_source_f_q_witness_authority,
            self.continuum_stability_authority,
            self.d_decision_stability_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.performance_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool for value in required_true + required_false)
            or any(value is not True for value in required_true)
            or any(value is not False for value in required_false)
            or self.rule != SAMPLED_V7_COMMON_D_ORIGIN_Q_LIVE_PROJECTION_RULE
            or self.authority_scope
            != SAMPLED_V7_COMMON_D_ORIGIN_Q_LIVE_PROJECTION_AUTHORITY
            or self.schema != SAMPLED_V7_COMMON_D_ORIGIN_Q_LIVE_PROJECTION_SCHEMA
        ):
            _refuse(
                V1SampledV7CommonDOriginQLiveProjectionRefusalCode.AUTHORITY_VIOLATION,
                "live projection crossed its binding-only authority",
            )
        observed = self._compute_hash()
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1SampledV7CommonDOriginQLiveProjectionRefusalCode.STALE_SOURCE,
                "live projection receipt was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(self, "receipt_hash", observed)

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_") and name != "receipt_hash"
        }

    def _compute_hash(self) -> str:
        (
            d1_receipt,
            common_receipt,
            common_conditioned,
            origin_receipt,
            branch_source,
            bridge_hash,
        ) = _fresh_live_parents(
            d1_fixed_source=self._d1_fixed_source,
            common_source=self._common_source,
            phase_a_registry=self._phase_a_registry,
            origin_q_source=self._origin_q_source,
            registered_q_bridge=self._registered_q_bridge,
        )
        alias_hashes = tuple(value.alias_hash for value in branch_source.resource_aliases)
        compact_projection = _compact_projection_hash(
            common_receipt=common_receipt,
            origin_receipt=origin_receipt,
        )
        projection = _projection_hash(
            common_receipt=common_receipt,
            origin_receipt=origin_receipt,
            phase_a_registry=self._phase_a_registry,
            branch_source=branch_source,
            bridge_hash=bridge_hash,
            compact_projection_hash=compact_projection,
        )
        observed = {
            "v7_d1_fixed_source_receipt_hash": d1_receipt.receipt_hash,
            "v7_d1_conditioned_source_receipt_hash": common_conditioned.receipt.receipt_hash,
            "v7_d1_conditioned_device_source_receipt_hash": common_conditioned.source_handle.receipt.receipt_hash,
            "v7_d1_literal_j_projection_registration_hash": self._d1_fixed_source.literal_j_projection.registration.registration_hash,
            "v7_d1_literal_j_projection_receipt_hash": self._d1_fixed_source.literal_j_projection.receipt.receipt_hash,
            "common_source_receipt_hash": common_receipt.receipt_hash,
            "common_source_replay_hash": common_receipt.source_replay_hash,
            "conditioned_d1_row_prefix_manifest_hash": common_conditioned.receipt.row_prefix_manifest_hash,
            "conditioned_d1_original_capture_binding_hash": common_conditioned.receipt.original_capture_binding_hash,
            "model_identity_hash": origin_receipt.model_identity_hash,
            "chart_identity_hash": origin_receipt.chart_identity_hash,
            "current_mask_hash": origin_receipt.current_mask_hash,
            "proposal_generation": origin_receipt.proposal_generation,
            "route_axis_hash": origin_receipt.route_axis_hash,
            "route_id_axis_hash": origin_receipt.route_id_axis_hash,
            "route_count": origin_receipt.route_count,
            "shell_freeze_receipt_hash": origin_receipt.shell_freeze_receipt_hash,
            "registered_d_prefix_hash": origin_receipt.registered_d_prefix_hash,
            "registered_d_w2_prefix_hash": origin_receipt.registered_d_w2_prefix_hash,
            "same_capture_vjp_source_hash": origin_receipt.same_capture_vjp_source_hash,
            "literal_j_stream_source_hash": origin_receipt.literal_j_stream_source_hash,
            "phase_a_registry_receipt_hash": self._phase_a_registry.receipt.receipt_hash,
            "phase_a_branch_registration_hash": branch_source.registration.registration_hash,
            "phase_a_branch": branch_source.branch,
            "phase_a_resource_alias_hashes": alias_hashes,
            "origin_q_source_receipt_hash": origin_receipt.receipt_hash,
            "origin_q_source_replay_hash": origin_receipt.source_replay_hash,
            "origin_q_source_kkt_receipt_hash": origin_receipt.kkt_branch_source_receipt_hash,
            "registered_q_bridge_receipt_hash": bridge_hash,
            "physical_action_coordinate_hash": self._registered_q_bridge.physical_action_coordinate_hash,
            "action_subset_identity_hash": self._registered_q_bridge.action_subset_identity_hash,
            "compact_row_projection_hash": compact_projection,
            "projection_binding_hash": projection,
        }
        if any(getattr(self, name) != value for name, value in observed.items()):
            _refuse(
                V1SampledV7CommonDOriginQLiveProjectionRefusalCode.STALE_SOURCE,
                "live projection receipt no longer matches its parent graph",
            )
        return content_hash(self._payload(), prefix=_PREFIX + "receipt-")

    def assert_integrity(self) -> str:
        observed = self._compute_hash()
        if observed != self._integrity_seal or observed != self.receipt_hash:
            _refuse(
                V1SampledV7CommonDOriginQLiveProjectionRefusalCode.STALE_SOURCE,
                "live projection receipt no longer replays",
            )
        return observed

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live common-D/origin-Q receipts are nonserializable")


def _compact_projection_hash(*, common_receipt, origin_receipt) -> str:
    return content_hash(
        {
            "common_conditioned_device_source_receipt_hash": (
                common_receipt.conditioned_d1_device_source_receipt_hash
            ),
            "origin_q_source_receipt_hash": origin_receipt.receipt_hash,
            "tensor_roles": _COMPACT_TENSOR_ROLES,
            "exact_live_tensor_equality_replayed": True,
        },
        prefix=_PREFIX + "compact-row-projection-",
    )


def _projection_hash(
    *,
    common_receipt,
    origin_receipt,
    phase_a_registry,
    branch_source,
    bridge_hash,
    compact_projection_hash,
) -> str:
    return content_hash(
        {
            "common_source_receipt_hash": common_receipt.receipt_hash,
            "origin_q_source_receipt_hash": origin_receipt.receipt_hash,
            "phase_a_registry_receipt_hash": phase_a_registry.receipt.receipt_hash,
            "phase_a_branch_registration_hash": branch_source.registration.registration_hash,
            "registered_q_bridge_receipt_hash": bridge_hash,
            "compact_row_projection_hash": compact_projection_hash,
            "receipt_hash_equality_not_required": True,
        },
        prefix=_PREFIX + "binding-",
    )


def issue_v1_sampled_v7_common_d_origin_q_live_projection_binding(
    *,
    d1_fixed_source: V1SampledV7D1FixedSourceResult,
    common_source: V1SampledFullPopulationCompactSourceHandle,
    phase_a_registry: V1SignedWorkVariableShellActionCellRegistry,
    origin_q_source: V1SignedWorkVariableShellSourceDeviceHandle,
    registered_q_bridge: (
        V1SignedWorkVariableShellRegisteredQPreferredHardActionBridgeReceipt
    ),
) -> V1SampledV7CommonDOriginQLiveProjectionBindingReceipt:
    """Issue an exact live common-D1 to origin-branch-Q projection binding."""

    (
        d1_receipt,
        common_receipt,
        common_conditioned,
        origin_receipt,
        branch_source,
        bridge_hash,
    ) = _fresh_live_parents(
        d1_fixed_source=d1_fixed_source,
        common_source=common_source,
        phase_a_registry=phase_a_registry,
        origin_q_source=origin_q_source,
        registered_q_bridge=registered_q_bridge,
    )
    compact_projection = _compact_projection_hash(
        common_receipt=common_receipt,
        origin_receipt=origin_receipt,
    )
    projection = _projection_hash(
        common_receipt=common_receipt,
        origin_receipt=origin_receipt,
        phase_a_registry=phase_a_registry,
        branch_source=branch_source,
        bridge_hash=bridge_hash,
        compact_projection_hash=compact_projection,
    )
    return V1SampledV7CommonDOriginQLiveProjectionBindingReceipt(
        v7_d1_fixed_source_receipt_hash=d1_receipt.receipt_hash,
        v7_d1_conditioned_source_receipt_hash=common_conditioned.receipt.receipt_hash,
        v7_d1_conditioned_device_source_receipt_hash=common_conditioned.source_handle.receipt.receipt_hash,
        v7_d1_literal_j_projection_registration_hash=d1_fixed_source.literal_j_projection.registration.registration_hash,
        v7_d1_literal_j_projection_receipt_hash=d1_fixed_source.literal_j_projection.receipt.receipt_hash,
        common_source_receipt_hash=common_receipt.receipt_hash,
        common_source_replay_hash=common_receipt.source_replay_hash,
        conditioned_d1_row_prefix_manifest_hash=common_conditioned.receipt.row_prefix_manifest_hash,
        conditioned_d1_original_capture_binding_hash=common_conditioned.receipt.original_capture_binding_hash,
        model_identity_hash=origin_receipt.model_identity_hash,
        chart_identity_hash=origin_receipt.chart_identity_hash,
        current_mask_hash=origin_receipt.current_mask_hash,
        proposal_generation=origin_receipt.proposal_generation,
        route_axis_hash=origin_receipt.route_axis_hash,
        route_id_axis_hash=origin_receipt.route_id_axis_hash,
        route_count=origin_receipt.route_count,
        shell_freeze_receipt_hash=origin_receipt.shell_freeze_receipt_hash,
        registered_d_prefix_hash=origin_receipt.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=origin_receipt.registered_d_w2_prefix_hash,
        same_capture_vjp_source_hash=origin_receipt.same_capture_vjp_source_hash,
        literal_j_stream_source_hash=origin_receipt.literal_j_stream_source_hash,
        phase_a_registry_receipt_hash=phase_a_registry.receipt.receipt_hash,
        phase_a_branch_registration_hash=branch_source.registration.registration_hash,
        phase_a_branch=branch_source.branch,
        phase_a_resource_alias_hashes=tuple(
            value.alias_hash for value in branch_source.resource_aliases
        ),
        origin_q_source_receipt_hash=origin_receipt.receipt_hash,
        origin_q_source_replay_hash=origin_receipt.source_replay_hash,
        origin_q_source_kkt_receipt_hash=origin_receipt.kkt_branch_source_receipt_hash,
        registered_q_bridge_receipt_hash=bridge_hash,
        physical_action_coordinate_hash=registered_q_bridge.physical_action_coordinate_hash,
        action_subset_identity_hash=registered_q_bridge.action_subset_identity_hash,
        compact_row_projection_hash=compact_projection,
        projection_binding_hash=projection,
        _d1_fixed_source=d1_fixed_source,
        _common_source=common_source,
        _phase_a_registry=phase_a_registry,
        _origin_q_source=origin_q_source,
        _registered_q_bridge=registered_q_bridge,
        _issuer_token=_ISSUER,
    )


__all__ = (
    "SAMPLED_V7_COMMON_D_ORIGIN_Q_LIVE_PROJECTION_AUTHORITY",
    "SAMPLED_V7_COMMON_D_ORIGIN_Q_LIVE_PROJECTION_RULE",
    "SAMPLED_V7_COMMON_D_ORIGIN_Q_LIVE_PROJECTION_SCHEMA",
    "V1SampledV7CommonDOriginQLiveProjectionBindingReceipt",
    "V1SampledV7CommonDOriginQLiveProjectionRefusal",
    "V1SampledV7CommonDOriginQLiveProjectionRefusalCode",
    "issue_v1_sampled_v7_common_d_origin_q_live_projection_binding",
)
