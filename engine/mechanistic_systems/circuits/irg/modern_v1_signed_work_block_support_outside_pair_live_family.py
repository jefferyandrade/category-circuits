"""Numerical outside-pair bounds owned by the source-native live cover.

This is the versioned numerical sibling of the legacy outside-pair family.
The pair domain is supplied exclusively by
``V1BlockSupportOutsidePairLiveCoverProvider``.  In particular, no legacy
host tuple, legacy pair-cover receipt, or legacy tuple hash is an input to the
production issuer.

The live cover fixes the two rectangles using the alias-specific rounding
anchor ``P_alpha``.  The preferred action ``P`` remains a distinct source and
is used only by the dual constant ``c.T (z - P)``.  The numerical kernel drops
alternate Q and relaxed-cardinality equality exactly as in the registered
linear outside-pair theorem.  Receipts are compact, array-free, and carry no
canonical block-observation, FQR, C/K/O, commit, terminal, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import math
from numbers import Real
from typing import Any, Final, NoReturn, Sequence

try:  # pragma: no cover - runtime image dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .contracts import content_hash
from .modern_v1_signed_work_block_support_outside_pair_cuda import (
    V1BlockSupportOutsidePairCUDAExecutionDesign,
    V1BlockSupportOutsidePairDualSourceHandle,
    V1BlockSupportOutsidePairExecutionMode,
    V1BlockSupportOutsidePairHBMForecast,
    _down,
    _scan_live_outside_pairs,
    _up,
)
from .modern_v1_signed_work_block_support_outside_pair_live_cover import (
    V1BlockSupportOutsidePairLiveCoverProvider,
    V1BlockSupportOutsidePairLiveCoverReference,
)
from .modern_v1_signed_work_block_support_outside_pair_production import (
    V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2,
    V1BlockSupportOutsidePairProductionDualSourceHandleV2,
    _live_cover_source_native_payload_hash,
)
from .modern_v1_signed_work_variable_shell_outside_exclusion_cuda import (
    V1SignedWorkVariableShellOutsidePairRectangle,
)


OUTSIDE_PAIR_LIVE_FAMILY_SCHEMA: Final = (
    "partition_free_v1_block_support_outside_pair_live_family_v2"
)
OUTSIDE_PAIR_LIVE_FAMILY_REFERENCE_SCHEMA: Final = (
    "partition_free_v1_block_support_outside_pair_live_family_reference_v1"
)
OUTSIDE_PAIR_LIVE_FAMILY_EQUATION: Final = (
    "live_P_alpha_rectangles_and_global_P_dual_constant_with_Q_and_relaxed_"
    "cardinality_equality_dropped_v2"
)
OUTSIDE_PAIR_LIVE_FAMILY_AUTHORITY: Final = (
    "one_block_one_alias_source_native_Q_dropped_outside_pair_lower_bound_"
    "only_no_canonical_block_full_U_FQR_C_K_O_commit_terminal_or_pareto_v2"
)
OUTSIDE_PAIR_LIVE_FAMILY_V3_SCHEMA: Final = (
    "partition_free_v1_block_support_outside_pair_live_family_v3"
)
OUTSIDE_PAIR_LIVE_FAMILY_V3_EQUATION: Final = (
    "source_native_live_P_alpha_rectangles_global_P_dual_constant_exact_"
    "problem_final_KKT_resource_block_duals_Q_and_cardinality_dropped_v3"
)
OUTSIDE_PAIR_LIVE_FAMILY_V3_AUTHORITY: Final = (
    "one_block_one_alias_source_native_raw_outside_family_only_no_complete_"
    "union_statistical_block_observation_full_U_FQR_C_K_O_commit_terminal_"
    "or_pareto_authority_v3"
)

_PREFIX: Final = "partition-free-v1-block-support-outside-pair-live-family-"
_RECEIPT_TOKEN: Final = object()
_RECEIPT_V3_TOKEN: Final = object()


class V1BlockSupportOutsidePairLiveFamilyRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    CUDA_REQUIRED = "cuda_required"
    DETERMINISM_REQUIRED = "determinism_required"
    SOURCE_MISMATCH = "source_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    NUMERICAL_REFUSAL = "numerical_refusal"
    MUTATED_CUSTODY = "mutated_custody"
    AUTHORITY_VIOLATION = "authority_violation"


class V1BlockSupportOutsidePairLiveFamilyRefusal(RuntimeError):
    def __init__(
        self,
        code: V1BlockSupportOutsidePairLiveFamilyRefusalCode,
        detail: str,
    ) -> None:
        if type(code) is not V1BlockSupportOutsidePairLiveFamilyRefusalCode:
            raise TypeError("live-family refusal code has another type")
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


def _refuse(
    code: V1BlockSupportOutsidePairLiveFamilyRefusalCode, detail: str
) -> NoReturn:
    raise V1BlockSupportOutsidePairLiveFamilyRefusal(code, detail)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.INVALID_TYPE,
            f"{name} must be a nonempty string",
        )
    return value


def _integer(value: object, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or type(value) is not int:
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.INVALID_TYPE,
            f"{name} must be an exact integer",
        )
    lower = 1 if positive else 0
    if value < lower:
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.INVALID_TYPE,
            f"{name} is outside its registered domain",
        )
    return value


def _finite(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.INVALID_TYPE,
            f"{name} must be a real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.NUMERICAL_REFUSAL,
            f"{name} is not a finite{' nonnegative' if nonnegative else ''} scalar",
        )
    return result


def _payload(value: object) -> dict[str, object]:
    return {
        item.name: (
            getattr(value, item.name).value
            if isinstance(getattr(value, item.name), Enum)
            else getattr(value, item.name)
        )
        for item in fields(value)  # type: ignore[arg-type]
        if not item.name.startswith("_") and item.name != "receipt_hash"
    }


def _payload_v3(value: object) -> dict[str, object]:
    """Canonicalize the theorem's exact vacuous ``+inf`` lower sentinel."""

    payload = _payload(value)
    for name in (
        "add_outside_pair_lower_outward",
        "drop_outside_add_inside_pair_lower_outward",
        "worst_pair_lower_center",
        "worst_pair_lower_outward",
    ):
        observed = payload.get(name)
        if isinstance(observed, float) and math.isinf(observed):
            payload[name] = "+inf" if observed > 0.0 else "-inf"
    return payload


def _closed(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool or value for value in values):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its authority boundary",
        )


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairLiveFamilyReceipt:
    """Compact production result for one source-native live rectangle family."""

    live_cover_receipt_hash: str
    source_native_pair_axis_commitment_hash: str
    physical_block_support_binding_receipt_hash: str
    structural_domain_receipt_hash: str
    d_fixed_support_receipt_hash: str
    residual_support_receipt_hash: str
    dual_source_receipt_hash: str
    block_support_registration_hash: str
    block_index: int
    block_id: str
    block_context_axis_hash: str
    residual_source_context_axis_hash: str
    source_registration_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    preferred_action_hash: str
    preferred_action_binding_hash: str
    preferred_removed_route_index_hash: str
    preferred_selected_count: int
    rounding_anchor_action_coordinate_hash: str
    rounding_anchor_removed_route_index_hash: str
    rounding_anchor_subset_identity_hash: str
    rounding_anchor_selected_count: int
    alias_hash: str
    branch_id: str
    fd_rank: int
    resource_alias_id: str
    source_receipt_hash: str
    source_replay_hash: str
    cardinality_alias_id: str
    alias_total_removed_count: int
    kkt_state_identity_hash: str
    kkt_problem_hash: str
    kkt_policy_hash: str
    kkt_branch_receipt_hash: str
    kkt_eligible_mask_identity_hash: str
    kkt_required_mask_identity_hash: str
    fixed_membership_binding_hash: str
    parent_live_membership_identity_hash: str
    selectable_membership_identity_hash: str
    add_outside_drop_axis_identity_hash: str
    add_outside_add_axis_identity_hash: str
    drop_outside_add_inside_drop_axis_identity_hash: str
    drop_outside_add_inside_add_axis_identity_hash: str
    resource_axis_hash: str
    resource_coefficient_source_hash: str
    capacity_operator_source_hash: str
    resource_multiplier_source_hash: str
    capacity_multiplier_source_hash: str
    route_count: int
    capacity_count: int
    route_tile_size: int
    route_tile_scan_count: int
    complete_route_coefficient_scan_count: int
    live_cover_complete_route_scan_count: int
    add_outside_rectangle_pair_count: int
    drop_outside_add_inside_rectangle_pair_count: int
    complete_pair_cover_count: int
    add_outside_rectangle_empty: bool
    drop_outside_add_inside_rectangle_empty: bool
    vacuous_no_outside_pair: bool
    base_dual_nominal: float
    base_dual_downward_radius: float
    base_dual_lower_outward: float
    comparison_preferred_dot_nominal: float
    comparison_preferred_dot_absolute_radius: float
    comparison_preferred_dot_upper_outward: float
    support_coefficient_source_l1_radius: float
    weight_conversion_l1_radius: float
    weighted_combination_roundoff_l1_radius: float
    resource_dual_roundoff_l1_radius: float
    capacity_dual_roundoff_l1_radius: float
    stationarity_addition_roundoff_l1_radius: float
    box_reduction_roundoff_radius: float
    dual_constant_roundoff_radius: float
    stationarity_sign_linf_radius: float
    ambiguous_p_endpoint_count: int
    ambiguous_q_endpoint_count: int
    add_outside_correction_lower: float
    drop_outside_add_inside_correction_lower: float
    add_outside_pair_lower_outward: float
    drop_outside_add_inside_pair_lower_outward: float
    worst_pair_rectangle: V1SignedWorkVariableShellOutsidePairRectangle | None
    worst_pair_preferred_selected_index: int | None
    worst_pair_preferred_unselected_index: int | None
    worst_pair_ordinal: int | None
    worst_pair_lower_center: float
    worst_pair_total_downward_radius: float
    worst_pair_lower_outward: float
    design_hash: str
    hbm_forecast_hash: str
    live_cover_design_hash: str
    live_cover_hbm_forecast_hash: str
    resident_device_identity: str
    scalar_device_to_host_transfer_bytes: int
    scan_device_to_host_synchronization_count: int
    receipt_hash: str = field(init=False)
    source_native_live_cover_joined: bool = True
    complete_two_rectangle_pair_cover_scanned: bool = True
    complete_route_scan: bool = True
    global_P_used_only_for_dual_constant: bool = True
    P_alpha_used_for_rectangle_geometry: bool = True
    alternate_Q_dropped: bool = True
    relaxed_cardinality_equality_enforced: bool = False
    cardinality_multiplier_present: bool = False
    legacy_pair_cover_authority_used: bool = False
    legacy_pair_axis_hash_authority_used: bool = False
    dual_legacy_pair_fields_ignored: bool = True
    route_scale_device_to_host_transfer_bytes: int = 0
    route_scale_content_digest_bytes: int = 0
    owned_route_scale_device_bytes: int = 0
    pair_matrix_bytes: int = 0
    bounded_workspace: bool = True
    array_free_receipt: bool = True
    production_cuda_execution: bool = True
    deterministic_for_frozen_design_only: bool = True
    canonical_block_observation_authority: bool = False
    complete_physical_alternate_union_authority: bool = False
    full_u_authority: bool = False
    fqr_authority: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    equation: str = OUTSIDE_PAIR_LIVE_FAMILY_EQUATION
    authority_scope: str = OUTSIDE_PAIR_LIVE_FAMILY_AUTHORITY
    schema: str = OUTSIDE_PAIR_LIVE_FAMILY_SCHEMA
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _RECEIPT_TOKEN:
            raise TypeError("live-family receipts are issuer-only")
        for item in fields(self):
            if (
                not item.name.startswith("_")
                and item.name != "receipt_hash"
                and item.name.endswith(("_hash", "_identity", "_commitment"))
            ):
                object.__setattr__(
                    self,
                    item.name,
                    _identity(
                        getattr(self, item.name), name=item.name.replace("_", " ")
                    ),
                )
        for name in (
            "block_id",
            "branch_id",
            "resource_alias_id",
            "cardinality_alias_id",
            "resident_device_identity",
            "equation",
            "authority_scope",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name.replace("_", " ")),
            )
        positive_integer_names = {"fd_rank", "route_count", "route_tile_size"}
        integer_names = (
            "block_index",
            "proposal_generation",
            "preferred_selected_count",
            "rounding_anchor_selected_count",
            "fd_rank",
            "alias_total_removed_count",
            "route_count",
            "capacity_count",
            "route_tile_size",
            "route_tile_scan_count",
            "complete_route_coefficient_scan_count",
            "live_cover_complete_route_scan_count",
            "add_outside_rectangle_pair_count",
            "drop_outside_add_inside_rectangle_pair_count",
            "complete_pair_cover_count",
            "ambiguous_p_endpoint_count",
            "ambiguous_q_endpoint_count",
            "scalar_device_to_host_transfer_bytes",
            "scan_device_to_host_synchronization_count",
            "route_scale_device_to_host_transfer_bytes",
            "route_scale_content_digest_bytes",
            "owned_route_scale_device_bytes",
            "pair_matrix_bytes",
        )
        for name in integer_names:
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=name.replace("_", " "),
                    positive=name in positive_integer_names,
                ),
            )
        for name in (
            "base_dual_nominal",
            "base_dual_downward_radius",
            "base_dual_lower_outward",
            "comparison_preferred_dot_nominal",
            "comparison_preferred_dot_absolute_radius",
            "comparison_preferred_dot_upper_outward",
            "support_coefficient_source_l1_radius",
            "weight_conversion_l1_radius",
            "weighted_combination_roundoff_l1_radius",
            "resource_dual_roundoff_l1_radius",
            "capacity_dual_roundoff_l1_radius",
            "stationarity_addition_roundoff_l1_radius",
            "box_reduction_roundoff_radius",
            "dual_constant_roundoff_radius",
            "stationarity_sign_linf_radius",
            "add_outside_correction_lower",
            "drop_outside_add_inside_correction_lower",
            "worst_pair_lower_center",
            "worst_pair_total_downward_radius",
        ):
            object.__setattr__(
                self,
                name,
                _finite(
                    getattr(self, name),
                    name=name.replace("_", " "),
                    nonnegative=name.endswith("radius")
                    or "_radius" in name
                    or name.endswith("correction_lower"),
                ),
            )
        for name in (
            "add_outside_pair_lower_outward",
            "drop_outside_add_inside_pair_lower_outward",
            "worst_pair_lower_outward",
        ):
            value = float(getattr(self, name))
            if math.isnan(value):
                _refuse(
                    V1BlockSupportOutsidePairLiveFamilyRefusalCode.NUMERICAL_REFUSAL,
                    f"{name.replace('_', ' ')} may not be NaN",
                )
            object.__setattr__(self, name, value)
        optional_integers = (
            self.worst_pair_preferred_selected_index,
            self.worst_pair_preferred_unselected_index,
            self.worst_pair_ordinal,
        )
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in optional_integers
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.INVALID_TYPE,
                "optional worst-pair coordinates must be nonnegative exact integers",
            )
        required_true = (
            self.source_native_live_cover_joined,
            self.complete_two_rectangle_pair_cover_scanned,
            self.complete_route_scan,
            self.global_P_used_only_for_dual_constant,
            self.P_alpha_used_for_rectangle_geometry,
            self.alternate_Q_dropped,
            self.dual_legacy_pair_fields_ignored,
            self.bounded_workspace,
            self.array_free_receipt,
            self.production_cuda_execution,
            self.deterministic_for_frozen_design_only,
        )
        if any(type(value) is not bool or value is not True for value in required_true):
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.AUTHORITY_VIOLATION,
                "live-family receipt weakened a mandatory production invariant",
            )
        required_false = (
            self.relaxed_cardinality_equality_enforced,
            self.cardinality_multiplier_present,
            self.legacy_pair_cover_authority_used,
            self.legacy_pair_axis_hash_authority_used,
            self.canonical_block_observation_authority,
            self.complete_physical_alternate_union_authority,
            self.full_u_authority,
            self.fqr_authority,
            self.C_stream_open_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        _closed(required_false, owner="outside-pair live-family receipt")
        if (
            self.schema != OUTSIDE_PAIR_LIVE_FAMILY_SCHEMA
            or self.equation != OUTSIDE_PAIR_LIVE_FAMILY_EQUATION
            or self.authority_scope != OUTSIDE_PAIR_LIVE_FAMILY_AUTHORITY
            or self.resident_device_identity.split(":", 1)[0] != "cuda"
            or self.preferred_selected_count <= 0
            or self.rounding_anchor_selected_count != self.alias_total_removed_count
            or self.complete_route_coefficient_scan_count != self.route_count
            or self.live_cover_complete_route_scan_count != self.route_count
            or self.complete_pair_cover_count
            != self.add_outside_rectangle_pair_count
            + self.drop_outside_add_inside_rectangle_pair_count
            or self.add_outside_rectangle_empty
            != (self.add_outside_rectangle_pair_count == 0)
            or self.drop_outside_add_inside_rectangle_empty
            != (self.drop_outside_add_inside_rectangle_pair_count == 0)
            or self.vacuous_no_outside_pair != (self.complete_pair_cover_count == 0)
            or self.route_scale_device_to_host_transfer_bytes != 0
            or self.route_scale_content_digest_bytes != 0
            or self.owned_route_scale_device_bytes != 0
            or self.pair_matrix_bytes != 0
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.AUTHORITY_VIOLATION,
                "live-family source, cover, memory, or authority accounting does not close",
            )
        if self.complete_pair_cover_count == 0:
            if (
                not math.isinf(self.worst_pair_lower_outward)
                or self.worst_pair_rectangle is not None
                or any(value is not None for value in optional_integers)
            ):
                _refuse(
                    V1BlockSupportOutsidePairLiveFamilyRefusalCode.NUMERICAL_REFUSAL,
                    "vacuous family carries a fake worst pair",
                )
        elif (
            self.worst_pair_rectangle is None
            or any(value is None for value in optional_integers)
            or not math.isfinite(self.worst_pair_lower_outward)
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.NUMERICAL_REFUSAL,
                "nonempty live family lost its deterministic worst-pair witness",
            )
        seal = content_hash(_payload(self), prefix=_PREFIX + "receipt-")
        if self._integrity_seal and self._integrity_seal != seal:
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.MUTATED_CUSTODY,
                "live-family receipt changed after issuance",
            )
        object.__setattr__(self, "_integrity_seal", seal)
        object.__setattr__(self, "receipt_hash", seal)

    def assert_integrity(self) -> str:
        seal = content_hash(_payload(self), prefix=_PREFIX + "receipt-")
        if seal != self._integrity_seal or seal != self.receipt_hash:
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.MUTATED_CUSTODY,
                "live-family receipt integrity no longer replays",
            )
        return seal

    def to_dict(self) -> dict[str, object]:
        return {**_payload(self), "receipt_hash": self.assert_integrity()}


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairLiveFamilyReceiptV3:
    """Production raw outside-family result with no legacy authority parent."""

    live_cover_authority_receipt_hash: str
    production_dual_source_receipt_hash: str
    live_cover_source_native_payload_hash: str
    source_native_alias_commitment_hash: str
    bound_witness_registration_receipt_hash: str
    preoutcome_domain_receipt_hash: str
    canonical_block_context_receipt_hash: str
    canonical_context_tuple_hash: str
    block_support_context_axis_hash: str
    physical_block_support_binding_receipt_hash: str
    structural_domain_receipt_hash: str
    d_fixed_support_receipt_hash: str
    residual_support_receipt_hash: str
    block_index: int
    block_id: str
    source_registration_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    preferred_action_hash: str
    preferred_action_binding_hash: str
    preferred_removed_route_index_hash: str
    preferred_selected_count: int
    resource_alias_hash: str
    branch_id: str
    resource_alias_id: str
    phase_a_source_receipt_hash: str
    phase_a_source_replay_hash: str
    source_resource_receipt_hash: str
    resource_coordinate_hash: str
    phase_a_resource_cell_id: str
    phase_a_invariance_receipt_hash: str
    registered_fd_rank: int
    cardinality_alias_id: str
    fixed_outside_removed_count: int
    shell_top_k_count: int
    hard_total_removed_count: int
    rounding_anchor_action_coordinate_hash: str
    rounding_anchor_removed_route_index_hash: str
    rounding_anchor_subset_identity_hash: str
    kkt_state_identity_hash: str
    kkt_problem_hash: str
    kkt_policy_hash: str
    kkt_branch_receipt_hash: str
    kkt_eligible_mask_identity_hash: str
    kkt_required_mask_identity_hash: str
    least_work_problem_hash: str
    source_problem_hash: str
    signed_work_operator_receipt_hash: str
    fixed_membership_binding_hash: str
    parent_live_membership_identity_hash: str
    selectable_membership_identity_hash: str
    add_outside_drop_axis_identity_hash: str
    add_outside_add_axis_identity_hash: str
    drop_outside_add_inside_drop_axis_identity_hash: str
    drop_outside_add_inside_add_axis_identity_hash: str
    source_native_pair_axis_commitment_hash: str
    resource_axis_hash: str
    resource_coefficient_source_hash: str
    capacity_operator_source_hash: str
    resource_multiplier_source_hash: str
    capacity_multiplier_source_hash: str
    operator_scales_hash: str
    route_count: int
    capacity_count: int
    route_tile_size: int
    route_tile_scan_count: int
    complete_route_coefficient_scan_count: int
    live_cover_complete_route_scan_count: int
    add_outside_droppable_count: int
    add_outside_addable_count: int
    drop_outside_add_inside_droppable_count: int
    drop_outside_add_inside_addable_count: int
    add_outside_rectangle_pair_count: int
    drop_outside_add_inside_rectangle_pair_count: int
    complete_pair_cover_count: int
    add_outside_rectangle_empty: bool
    drop_outside_add_inside_rectangle_empty: bool
    vacuous_no_outside_pair: bool
    base_dual_nominal: float
    base_dual_downward_radius: float
    base_dual_lower_outward: float
    comparison_preferred_dot_nominal: float
    comparison_preferred_dot_absolute_radius: float
    comparison_preferred_dot_upper_outward: float
    support_coefficient_source_l1_radius: float
    weight_conversion_l1_radius: float
    weighted_combination_roundoff_l1_radius: float
    resource_dual_roundoff_l1_radius: float
    capacity_dual_roundoff_l1_radius: float
    stationarity_addition_roundoff_l1_radius: float
    box_reduction_roundoff_radius: float
    dual_constant_roundoff_radius: float
    stationarity_sign_linf_radius: float
    ambiguous_p_endpoint_count: int
    ambiguous_q_endpoint_count: int
    add_outside_correction_lower: float
    drop_outside_add_inside_correction_lower: float
    add_outside_pair_lower_outward: float
    drop_outside_add_inside_pair_lower_outward: float
    worst_pair_rectangle: V1SignedWorkVariableShellOutsidePairRectangle | None
    worst_pair_preferred_selected_index: int | None
    worst_pair_preferred_unselected_index: int | None
    worst_pair_ordinal: int | None
    worst_pair_lower_center: float
    worst_pair_total_downward_radius: float
    worst_pair_lower_outward: float
    design_hash: str
    hbm_forecast_hash: str
    resident_device_identity: str
    scalar_device_to_host_transfer_bytes: int
    scan_device_to_host_synchronization_count: int
    receipt_hash: str = field(init=False)
    exact_live_cover_authority_v2_join: bool = True
    exact_problem_final_KKT_dual_v2_join: bool = True
    complete_two_rectangle_pair_cover_scanned: bool = True
    complete_route_scan: bool = True
    global_P_used_only_for_dual_constant: bool = True
    P_alpha_used_for_rectangle_geometry: bool = True
    alternate_Q_dropped: bool = True
    relaxed_cardinality_equality_enforced: bool = False
    cardinality_multiplier_present: bool = False
    legacy_live_cover_receipt_hash_authority_used: bool = False
    legacy_pair_cover_authority_used: bool = False
    caller_source_hash_authority_used: bool = False
    caller_raw_operator_or_multiplier_used: bool = False
    route_scale_device_to_host_transfer_bytes: int = 0
    route_scale_content_digest_bytes: int = 0
    owned_route_scale_device_bytes: int = 0
    pair_matrix_bytes: int = 0
    bounded_workspace: bool = True
    array_free_receipt: bool = True
    production_cuda_execution: bool = True
    deterministic_for_frozen_design_only: bool = True
    one_alias_raw_outside_family_authority: bool = True
    complete_union_authority: bool = False
    statistical_authority: bool = False
    canonical_block_observation_authority: bool = False
    complete_physical_alternate_union_authority: bool = False
    full_u_authority: bool = False
    fqr_authority: bool = False
    c_authority: bool = False
    k_authority: bool = False
    o_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    equation: str = OUTSIDE_PAIR_LIVE_FAMILY_V3_EQUATION
    authority_scope: str = OUTSIDE_PAIR_LIVE_FAMILY_V3_AUTHORITY
    schema: str = OUTSIDE_PAIR_LIVE_FAMILY_V3_SCHEMA
    _issuer_token: InitVar[object] = None
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _RECEIPT_V3_TOKEN:
            raise TypeError("live-family V3 receipts are issuer-only")
        for item in fields(self):
            if (
                not item.name.startswith("_")
                and item.name != "receipt_hash"
                and item.name.endswith(("_hash", "_identity", "_commitment"))
            ):
                object.__setattr__(
                    self,
                    item.name,
                    _identity(getattr(self, item.name), name=item.name),
                )
        for name in (
            "block_id",
            "branch_id",
            "resource_alias_id",
            "phase_a_resource_cell_id",
            "cardinality_alias_id",
            "resident_device_identity",
            "equation",
            "authority_scope",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name),
            )
        positive = {
            "registered_fd_rank",
            "route_count",
            "route_tile_size",
        }
        for name in (
            "block_index",
            "proposal_generation",
            "preferred_selected_count",
            "registered_fd_rank",
            "fixed_outside_removed_count",
            "shell_top_k_count",
            "hard_total_removed_count",
            "route_count",
            "capacity_count",
            "route_tile_size",
            "route_tile_scan_count",
            "complete_route_coefficient_scan_count",
            "live_cover_complete_route_scan_count",
            "add_outside_droppable_count",
            "add_outside_addable_count",
            "drop_outside_add_inside_droppable_count",
            "drop_outside_add_inside_addable_count",
            "add_outside_rectangle_pair_count",
            "drop_outside_add_inside_rectangle_pair_count",
            "complete_pair_cover_count",
            "ambiguous_p_endpoint_count",
            "ambiguous_q_endpoint_count",
            "scalar_device_to_host_transfer_bytes",
            "scan_device_to_host_synchronization_count",
            "route_scale_device_to_host_transfer_bytes",
            "route_scale_content_digest_bytes",
            "owned_route_scale_device_bytes",
            "pair_matrix_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=name,
                    positive=name in positive,
                ),
            )
        nonnegative = {
            "base_dual_downward_radius",
            "comparison_preferred_dot_absolute_radius",
            "support_coefficient_source_l1_radius",
            "weight_conversion_l1_radius",
            "weighted_combination_roundoff_l1_radius",
            "resource_dual_roundoff_l1_radius",
            "capacity_dual_roundoff_l1_radius",
            "stationarity_addition_roundoff_l1_radius",
            "box_reduction_roundoff_radius",
            "dual_constant_roundoff_radius",
            "stationarity_sign_linf_radius",
            "add_outside_correction_lower",
            "drop_outside_add_inside_correction_lower",
            "worst_pair_total_downward_radius",
        }
        for name in (
            "base_dual_nominal",
            "base_dual_downward_radius",
            "base_dual_lower_outward",
            "comparison_preferred_dot_nominal",
            "comparison_preferred_dot_absolute_radius",
            "comparison_preferred_dot_upper_outward",
            "support_coefficient_source_l1_radius",
            "weight_conversion_l1_radius",
            "weighted_combination_roundoff_l1_radius",
            "resource_dual_roundoff_l1_radius",
            "capacity_dual_roundoff_l1_radius",
            "stationarity_addition_roundoff_l1_radius",
            "box_reduction_roundoff_radius",
            "dual_constant_roundoff_radius",
            "stationarity_sign_linf_radius",
            "add_outside_correction_lower",
            "drop_outside_add_inside_correction_lower",
            "worst_pair_total_downward_radius",
        ):
            object.__setattr__(
                self,
                name,
                _finite(
                    getattr(self, name),
                    name=name,
                    nonnegative=name in nonnegative,
                ),
            )
        for name in (
            "add_outside_pair_lower_outward",
            "drop_outside_add_inside_pair_lower_outward",
            "worst_pair_lower_center",
            "worst_pair_lower_outward",
        ):
            value = float(getattr(self, name))
            if math.isnan(value):
                _refuse(
                    V1BlockSupportOutsidePairLiveFamilyRefusalCode.NUMERICAL_REFUSAL,
                    f"{name} may not be NaN",
                )
            object.__setattr__(self, name, value)
        optional_integers = (
            self.worst_pair_preferred_selected_index,
            self.worst_pair_preferred_unselected_index,
            self.worst_pair_ordinal,
        )
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in optional_integers
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.INVALID_TYPE,
                "V3 worst-pair coordinates must be nonnegative exact integers",
            )
        required = (
            self.exact_live_cover_authority_v2_join,
            self.exact_problem_final_KKT_dual_v2_join,
            self.complete_two_rectangle_pair_cover_scanned,
            self.complete_route_scan,
            self.global_P_used_only_for_dual_constant,
            self.P_alpha_used_for_rectangle_geometry,
            self.alternate_Q_dropped,
            self.bounded_workspace,
            self.array_free_receipt,
            self.production_cuda_execution,
            self.deterministic_for_frozen_design_only,
            self.one_alias_raw_outside_family_authority,
        )
        if any(type(value) is not bool or value is not True for value in required):
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.AUTHORITY_VIOLATION,
                "live-family V3 lost a mandatory production invariant",
            )
        _closed(
            (
                self.relaxed_cardinality_equality_enforced,
                self.cardinality_multiplier_present,
                self.legacy_live_cover_receipt_hash_authority_used,
                self.legacy_pair_cover_authority_used,
                self.caller_source_hash_authority_used,
                self.caller_raw_operator_or_multiplier_used,
                self.complete_union_authority,
                self.statistical_authority,
                self.canonical_block_observation_authority,
                self.complete_physical_alternate_union_authority,
                self.full_u_authority,
                self.fqr_authority,
                self.c_authority,
                self.k_authority,
                self.o_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="outside-pair live-family V3",
        )
        if (
            self.hard_total_removed_count
            != self.fixed_outside_removed_count + self.shell_top_k_count
            or self.complete_route_coefficient_scan_count != self.route_count
            or self.live_cover_complete_route_scan_count != self.route_count
            or self.add_outside_rectangle_pair_count
            != self.add_outside_droppable_count * self.add_outside_addable_count
            or self.drop_outside_add_inside_rectangle_pair_count
            != self.drop_outside_add_inside_droppable_count
            * self.drop_outside_add_inside_addable_count
            or self.complete_pair_cover_count
            != self.add_outside_rectangle_pair_count
            + self.drop_outside_add_inside_rectangle_pair_count
            or self.add_outside_rectangle_empty
            != (self.add_outside_rectangle_pair_count == 0)
            or self.drop_outside_add_inside_rectangle_empty
            != (self.drop_outside_add_inside_rectangle_pair_count == 0)
            or self.vacuous_no_outside_pair != (self.complete_pair_cover_count == 0)
            or self.route_scale_device_to_host_transfer_bytes != 0
            or self.route_scale_content_digest_bytes != 0
            or self.owned_route_scale_device_bytes != 0
            or self.pair_matrix_bytes != 0
            or not self.resident_device_identity.startswith("cuda")
            or self.equation != OUTSIDE_PAIR_LIVE_FAMILY_V3_EQUATION
            or self.authority_scope != OUTSIDE_PAIR_LIVE_FAMILY_V3_AUTHORITY
            or self.schema != OUTSIDE_PAIR_LIVE_FAMILY_V3_SCHEMA
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.AUTHORITY_VIOLATION,
                "live-family V3 source, rectangle, memory, or authority accounting failed",
            )
        if (
            self.add_outside_rectangle_empty
            and self.add_outside_pair_lower_outward != math.inf
        ) or (
            not self.add_outside_rectangle_empty
            and not math.isfinite(self.add_outside_pair_lower_outward)
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.NUMERICAL_REFUSAL,
                "live-family V3 add-outside lower has the wrong empty semantics",
            )
        if (
            self.drop_outside_add_inside_rectangle_empty
            and self.drop_outside_add_inside_pair_lower_outward != math.inf
        ) or (
            not self.drop_outside_add_inside_rectangle_empty
            and not math.isfinite(self.drop_outside_add_inside_pair_lower_outward)
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.NUMERICAL_REFUSAL,
                "live-family V3 drop/add lower has the wrong empty semantics",
            )
        if self.complete_pair_cover_count == 0:
            if (
                self.worst_pair_lower_center != math.inf
                or self.worst_pair_lower_outward != math.inf
                or self.worst_pair_total_downward_radius != 0.0
                or self.worst_pair_rectangle is not None
                or any(value is not None for value in optional_integers)
            ):
                _refuse(
                    V1BlockSupportOutsidePairLiveFamilyRefusalCode.NUMERICAL_REFUSAL,
                    "vacuous live-family V3 carries a fake worst pair",
                )
        elif (
            self.worst_pair_rectangle is None
            or any(value is None for value in optional_integers)
            or not math.isfinite(self.worst_pair_lower_center)
            or not math.isfinite(self.worst_pair_lower_outward)
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.NUMERICAL_REFUSAL,
                "nonempty live-family V3 lost its deterministic worst pair",
            )
        seal = content_hash(_payload_v3(self), prefix=_PREFIX + "receipt-v3-")
        if self._integrity_seal and self._integrity_seal != seal:
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.MUTATED_CUSTODY,
                "live-family V3 receipt changed after issuance",
            )
        object.__setattr__(self, "_integrity_seal", seal)
        object.__setattr__(self, "receipt_hash", seal)

    def assert_integrity(self) -> str:
        seal = content_hash(_payload_v3(self), prefix=_PREFIX + "receipt-v3-")
        if seal != self._integrity_seal or seal != self.receipt_hash:
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.MUTATED_CUSTODY,
                "live-family V3 receipt integrity no longer replays",
            )
        return seal

    def to_dict(self) -> dict[str, object]:
        return {**_payload_v3(self), "receipt_hash": self.assert_integrity()}


@dataclass(frozen=True, slots=True)
class V1BlockSupportOutsidePairLiveFamilyReference:
    """Small CPU numerical adapter; never source or production authority."""

    route_count: int
    add_outside_rectangle_pair_count: int
    drop_outside_add_inside_rectangle_pair_count: int
    add_outside_pair_lower_outward: float
    drop_outside_add_inside_pair_lower_outward: float
    worst_pair_lower_outward: float
    worst_pair_rectangle: V1SignedWorkVariableShellOutsidePairRectangle | None
    worst_pair_preferred_selected_index: int | None
    worst_pair_preferred_unselected_index: int | None
    worst_pair_ordinal: int | None
    base_dual_lower_outward: float
    route_tile_scan_count: int
    scalar_device_to_host_transfer_bytes: int
    cpu_reference_test_only: bool = True
    source_authority: bool = False
    production_authority: bool = False
    downstream_authority: bool = False
    schema: str = OUTSIDE_PAIR_LIVE_FAMILY_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.cpu_reference_test_only is not True:
            raise ValueError("live-family reference must remain test-only")
        _closed(
            (
                self.source_authority,
                self.production_authority,
                self.downstream_authority,
            ),
            owner="outside-pair live-family reference",
        )


def _preferred_shell_routes(
    provider: V1BlockSupportOutsidePairLiveCoverProvider,
) -> tuple[int, ...]:
    structural_provider = provider.physical_block_support_binding.structural_provider
    receipt = structural_provider.receipt
    excluded = set(receipt.excluded_preferred_action_cell_hashes)
    matched = tuple(
        cell
        for cell in structural_provider.registry.iter_action_cells()
        if cell.action_cell_hash in excluded
    )
    offset_axes = {cell.selected_shell_offsets for cell in matched}
    if len(matched) != len(excluded) or len(offset_axes) != 1:
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.SOURCE_MISMATCH,
            "structural preferred cells do not determine one global-P shell subset",
        )
    offsets = next(iter(offset_axes))
    routes = tuple(receipt.shell_indices[offset] for offset in offsets)
    if (
        receipt.fixed_outside_removed_count + len(routes)
        != receipt.preferred_selected_count
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.SOURCE_MISMATCH,
            "global-P fixed plus shell membership has the wrong cardinality",
        )
    return routes


def _validate_source_join(
    *,
    provider: V1BlockSupportOutsidePairLiveCoverProvider,
    dual_source: V1BlockSupportOutsidePairDualSourceHandle,
    design: V1BlockSupportOutsidePairCUDAExecutionDesign,
    forecast: V1BlockSupportOutsidePairHBMForecast,
) -> tuple[Any, Any, Any, Any, Any]:
    provider.assert_live()
    dual_source.assert_live()
    cover = provider.receipt
    physical = provider.physical_block_support_binding
    physical.assert_live()
    binding = physical.receipt
    d_receipt = physical.d_fixed_support.receipt
    residual_receipt = physical.residual_support.receipt
    dual = dual_source.receipt
    dual.assert_integrity()
    if (
        design.execution_mode
        is not V1BlockSupportOutsidePairExecutionMode.CUDA_PRODUCTION
        or dual_source.design is not design
        or dual_source.forecast is not forecast
        or forecast.design_hash != design.design_hash
        or not forecast.admitted
        or cover.physical_block_support_binding_receipt_hash != binding.receipt_hash
        or cover.structural_domain_receipt_hash
        != binding.structural_domain_receipt_hash
        or binding.d_fixed_support_receipt_hash != d_receipt.receipt_hash
        or binding.residual_support_receipt_hash != residual_receipt.receipt_hash
        or cover.block_support_registration_hash
        != provider.block_witness_source.registration.registration_hash
        or cover.alias_hash != provider.alias_binding.alias_hash
        or cover.route_count != binding.route_count
        or cover.route_count != d_receipt.route_count
        or cover.route_count != residual_receipt.route_count
        or cover.route_count != dual.route_count
        or forecast.route_count != cover.route_count
        or cover.resident_device_identity != binding.resident_device_identity
        or dual.device_type != "cuda"
        or dual.resource_custody_kind != "production_kkt_live_borrow"
        or dual.alias_hash != cover.alias_hash
        or dual.branch_id != cover.branch_id
        or dual.fd_rank != cover.fd_rank
        or dual.resource_alias_id != cover.resource_alias_id
        or dual.source_receipt_hash != cover.source_receipt_hash
        or dual.source_replay_hash != cover.source_replay_hash
        or dual.cardinality_alias_id != cover.cardinality_alias_id
        or dual.cardinality != cover.alias_total_removed_count
        or dual.model_identity_hash != cover.model_identity_hash
        or dual.chart_identity_hash != cover.chart_identity_hash
        or dual.current_mask_hash != cover.current_mask_hash
        or dual.proposal_generation != cover.proposal_generation
        or dual.route_axis_hash != cover.route_axis_hash
        or dual.route_id_axis_hash != cover.route_id_axis_hash
        or dual.registered_d_prefix_hash != cover.registered_d_prefix_hash
        or dual.registered_d_w2_prefix_hash != cover.registered_d_w2_prefix_hash
        or dual.preferred_action_hash != cover.comparison_preferred_action_hash
        or dual.preferred_action_binding_hash != cover.preferred_action_binding_hash
        or dual.design_hash != design.design_hash
        or dual.hbm_forecast_hash != forecast.forecast_hash
        or dual.alternate_Q_dropped is not True
        or dual.relaxed_cardinality_equality_enforced is not False
        or dual.route_scale_d2h_bytes != 0
        or dual.route_scale_tensor_digest_bytes != 0
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.SOURCE_MISMATCH,
            "live cover, physical support, KKT, or dual source identities differ",
        )
    source_names = (
        "source_registration_hash",
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "registered_d_prefix_hash",
        "registered_d_w2_prefix_hash",
        "native_fiber_axis_receipt_hash",
        "native_fiber_measure_hash",
        "preferred_action_binding_hash",
        "preferred_action_hash",
        "preferred_removed_route_index_hash",
    )
    if any(
        getattr(d_receipt, name) != getattr(binding, name)
        or getattr(residual_receipt, name) != getattr(binding, name)
        for name in source_names
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.SOURCE_MISMATCH,
            "D/residual source does not replay the physical binding",
        )
    return cover, binding, d_receipt, residual_receipt, dual


def _validate_source_join_v3(
    *,
    live_cover_authority: V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2,
    provider: V1BlockSupportOutsidePairLiveCoverProvider,
    dual_source: V1BlockSupportOutsidePairProductionDualSourceHandleV2,
    design: V1BlockSupportOutsidePairCUDAExecutionDesign,
    forecast: V1BlockSupportOutsidePairHBMForecast,
) -> tuple[Any, Any, Any, Any, Any]:
    live_cover_authority.assert_integrity()
    provider.assert_live()
    dual_source.assert_live()
    cover = provider.receipt
    physical = provider.physical_block_support_binding
    physical.assert_live()
    binding = physical.receipt
    d_receipt = physical.d_fixed_support.receipt
    residual_receipt = physical.residual_support.receipt
    dual = dual_source.receipt
    dual.assert_integrity()
    if (
        design.execution_mode
        is not V1BlockSupportOutsidePairExecutionMode.CUDA_PRODUCTION
        or dual_source.cover_authority is not live_cover_authority
        or dual_source.live_cover_provider is not provider
        or dual_source.design is not design
        or dual_source.forecast is not forecast
        or forecast.design_hash != design.design_hash
        or forecast.forecast_hash != dual.hbm_forecast_hash
        or not forecast.admitted
        or live_cover_authority.live_cover_source_native_payload_hash
        != _live_cover_source_native_payload_hash(cover)
        or dual.live_cover_authority_receipt_hash != live_cover_authority.receipt_hash
        or dual.live_cover_source_native_payload_hash
        != live_cover_authority.live_cover_source_native_payload_hash
        or dual.source_native_alias_commitment_hash
        != live_cover_authority.source_native_alias_commitment_hash
        or dual.physical_block_support_binding_receipt_hash != binding.receipt_hash
        or dual.kkt_state_identity_hash != live_cover_authority.kkt_state_identity_hash
        or dual.least_work_problem_hash != live_cover_authority.kkt_problem_hash
        or dual.kkt_policy_hash != live_cover_authority.kkt_policy_hash
        or dual.kkt_branch_receipt_hash != live_cover_authority.kkt_branch_receipt_hash
        or dual.kkt_eligible_mask_identity_hash
        != live_cover_authority.kkt_eligible_mask_identity_hash
        or dual.kkt_required_mask_identity_hash
        != live_cover_authority.kkt_required_mask_identity_hash
        or dual.resource_alias_hash != live_cover_authority.resource_alias_hash
        or dual.branch_id != live_cover_authority.branch_id
        or dual.resource_alias_id != live_cover_authority.resource_alias_id
        or dual.cardinality_alias_id != live_cover_authority.cardinality_alias_id
        or dual.registered_fd_rank != live_cover_authority.registered_fd_rank
        or dual.route_count != live_cover_authority.route_count
        or dual.route_axis_hash != live_cover_authority.route_axis_hash
        or dual.source_registration_hash
        != live_cover_authority.source_registration_hash
        or dual.block_index != live_cover_authority.block_index
        or dual.block_id != live_cover_authority.block_id
        or dual.canonical_block_context_receipt_hash
        != live_cover_authority.canonical_block_context_receipt_hash
        or dual.source_native_pair_axis_commitment_hash
        != live_cover_authority.source_native_pair_axis_commitment_hash
        or dual.legacy_pair_cover_consumed is not False
        or dual.caller_source_hash_consumed is not False
        or dual.caller_raw_operator_consumed is not False
        or dual.caller_raw_multiplier_consumed is not False
        or dual.one_alias_raw_outside_family_authority is not True
        or dual.complete_union_authority is not False
        or dual.statistical_authority is not False
        or cover.physical_block_support_binding_receipt_hash != binding.receipt_hash
        or cover.structural_domain_receipt_hash
        != binding.structural_domain_receipt_hash
        or binding.d_fixed_support_receipt_hash != d_receipt.receipt_hash
        or binding.residual_support_receipt_hash != residual_receipt.receipt_hash
        or cover.route_count != binding.route_count
        or cover.route_count != d_receipt.route_count
        or cover.route_count != residual_receipt.route_count
        or forecast.route_count != cover.route_count
        or cover.resident_device_identity != binding.resident_device_identity
        or dual.resident_device_identity != cover.resident_device_identity
        or not dual.resident_device_identity.startswith("cuda")
        or dual.resource_floor_nominal
        != live_cover_authority.registered_shell_resource_floor
        or dual_source.kkt_state is not provider.kkt_state
        or dual_source.least_work_problem.problem_hash
        != live_cover_authority.kkt_problem_hash
        or dual_source.kkt_branch_receipt.receipt_hash
        != live_cover_authority.kkt_branch_receipt_hash
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.SOURCE_MISMATCH,
            "production cover authority, physical support, KKT, or dual source identities differ",
        )
    source_names = (
        "source_registration_hash",
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "registered_d_prefix_hash",
        "registered_d_w2_prefix_hash",
        "native_fiber_axis_receipt_hash",
        "native_fiber_measure_hash",
        "preferred_action_binding_hash",
        "preferred_action_hash",
        "preferred_removed_route_index_hash",
    )
    if any(
        getattr(d_receipt, name) != getattr(binding, name)
        or getattr(residual_receipt, name) != getattr(binding, name)
        or getattr(live_cover_authority, name) != getattr(binding, name)
        for name in source_names
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.SOURCE_MISMATCH,
            "D/residual/source-authority receipts do not replay the physical binding",
        )
    return cover, binding, d_receipt, residual_receipt, dual


def _pair_lowers(scan: Any) -> tuple[float, float]:
    first = scan.first_rectangle
    second = scan.second_rectangle
    return (
        (
            math.inf
            if not first.pair_count
            else _down(scan.base_lower + first.correction_lower)
        ),
        (
            math.inf
            if not second.pair_count
            else _down(scan.base_lower + second.correction_lower)
        ),
    )


def evaluate_v1_block_support_outside_pair_live_family_reference(
    *,
    live_cover: V1BlockSupportOutsidePairLiveCoverReference,
    comparison_preferred_indices: Sequence[int],
    d_coefficients: Sequence[Real],
    residual_coefficients: Sequence[Real],
    resource_coefficients: Sequence[Real],
    capacity_adjoint: Sequence[Real],
    d_weight_numerator: int,
    residual_weight_numerator: int,
    population_weight_denominator: int,
    d_source_l1_radius: float,
    residual_source_l1_radius: float,
    resource_multiplier: float,
    resource_floor: float,
    capacity_multiplier_dot_limit: float,
    resource_multiplier_conversion_radius: float = 0.0,
    capacity_adjoint_l1_radius: float = 0.0,
    capacity_adjoint_linf_radius: float = 0.0,
    dual_constant_radius: float = 0.0,
    route_tile_size: int = 8,
) -> V1BlockSupportOutsidePairLiveFamilyReference:
    """Run the same tiled arithmetic on a bounded host reference domain."""

    if type(live_cover) is not V1BlockSupportOutsidePairLiveCoverReference:
        raise TypeError("live-family reference requires the exact live-cover reference")
    if torch is None:
        raise RuntimeError("torch is required for the live-family reference")
    count = live_cover.route_count
    axes = tuple(
        tuple(float(value) for value in source)
        for source in (
            d_coefficients,
            residual_coefficients,
            resource_coefficients,
            capacity_adjoint,
        )
    )
    if any(len(source) != count for source in axes):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.SOURCE_MISMATCH,
            "reference numerical axes do not share the live-cover route count",
        )
    comparison = tuple(
        _integer(value, name="comparison preferred index")
        for value in comparison_preferred_indices
    )
    if comparison != tuple(sorted(set(comparison))) or any(
        value >= count for value in comparison
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.INVALID_TYPE,
            "comparison preferred indices are not a canonical physical-index set",
        )
    # Designs are issuer-only; use the public builder rather than weakening it.
    from .modern_v1_signed_work_block_support_outside_pair_cuda import (
        build_v1_block_support_outside_pair_cuda_design,
    )

    design = build_v1_block_support_outside_pair_cuda_design(
        execution_mode=V1BlockSupportOutsidePairExecutionMode.CPU_EXACT_REFERENCE_TEST_ONLY,
        route_tile_size=route_tile_size,
        cpu_reference_maximum_routes=max(1, count),
    )

    def membership(indices: Sequence[int]) -> Any:
        selected = set(indices)
        return torch.tensor(
            tuple(index in selected for index in range(count)), dtype=torch.bool
        )

    scan = _scan_live_outside_pairs(
        design=design,
        d_coefficients=torch.tensor(axes[0], dtype=torch.float64),
        residual_coefficients=torch.tensor(axes[1], dtype=torch.float64),
        resource_coefficients=torch.tensor(axes[2], dtype=torch.float64),
        capacity_adjoint=torch.tensor(axes[3], dtype=torch.float64),
        rounding_anchor=membership(live_cover.rounding_anchor_indices),
        comparison_preferred=membership(comparison),
        required=membership(live_cover.required_indices),
        selectable=membership(live_cover.selectable_indices),
        shell=membership(live_cover.shell_indices),
        d_weight_numerator=d_weight_numerator,
        residual_weight_numerator=residual_weight_numerator,
        population_weight_denominator=population_weight_denominator,
        d_source_l1_radius=d_source_l1_radius,
        residual_source_l1_radius=residual_source_l1_radius,
        resource_multiplier=resource_multiplier,
        resource_floor=resource_floor,
        capacity_multiplier_dot_limit=capacity_multiplier_dot_limit,
        resource_multiplier_conversion_radius=resource_multiplier_conversion_radius,
        capacity_adjoint_l1_radius=capacity_adjoint_l1_radius,
        capacity_adjoint_linf_radius=capacity_adjoint_linf_radius,
        dual_constant_radius=dual_constant_radius,
    )
    first_lower, second_lower = _pair_lowers(scan)
    worst = scan.worst_rectangle
    if (
        scan.first_rectangle.pair_count != live_cover.add_outside_rectangle_pair_count
        or scan.second_rectangle.pair_count
        != live_cover.drop_outside_add_inside_rectangle_pair_count
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.SOURCE_MISMATCH,
            "reference numerical scan differs from the live-cover rectangles",
        )
    return V1BlockSupportOutsidePairLiveFamilyReference(
        route_count=count,
        add_outside_rectangle_pair_count=scan.first_rectangle.pair_count,
        drop_outside_add_inside_rectangle_pair_count=scan.second_rectangle.pair_count,
        add_outside_pair_lower_outward=first_lower,
        drop_outside_add_inside_pair_lower_outward=second_lower,
        worst_pair_lower_outward=scan.worst_lower,
        worst_pair_rectangle=None if worst is None else worst.rectangle,
        worst_pair_preferred_selected_index=None if worst is None else worst.p_index,
        worst_pair_preferred_unselected_index=None if worst is None else worst.q_index,
        worst_pair_ordinal=None if worst is None else worst.pair_ordinal,
        base_dual_lower_outward=scan.base_lower,
        route_tile_scan_count=scan.route_tile_scan_count,
        scalar_device_to_host_transfer_bytes=scan.scalar_d2h_bytes,
    )


def issue_v1_block_support_outside_pair_live_family_cuda(
    *,
    live_cover_provider: V1BlockSupportOutsidePairLiveCoverProvider,
    dual_source: V1BlockSupportOutsidePairDualSourceHandle,
    design: V1BlockSupportOutsidePairCUDAExecutionDesign,
    hbm_forecast: V1BlockSupportOutsidePairHBMForecast,
) -> V1BlockSupportOutsidePairLiveFamilyReceipt:
    """Issue a production family directly from source-native live custody."""

    if torch is None or not torch.cuda.is_available():
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.CUDA_REQUIRED,
            "production live-family execution fails closed without CUDA",
        )
    if not torch.are_deterministic_algorithms_enabled():
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.DETERMINISM_REQUIRED,
            "production live-family execution requires deterministic algorithms",
        )
    if (
        type(live_cover_provider) is not V1BlockSupportOutsidePairLiveCoverProvider
        or type(dual_source) is not V1BlockSupportOutsidePairDualSourceHandle
        or type(design) is not V1BlockSupportOutsidePairCUDAExecutionDesign
        or type(hbm_forecast) is not V1BlockSupportOutsidePairHBMForecast
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.INVALID_TYPE,
            "production live-family execution requires exact typed sources",
        )
    cover, binding, d_receipt, residual_receipt, dual = _validate_source_join(
        provider=live_cover_provider,
        dual_source=dual_source,
        design=design,
        forecast=hbm_forecast,
    )
    physical = live_cover_provider.physical_block_support_binding
    preferred_shell = _preferred_shell_routes(live_cover_provider)
    kkt = live_cover_provider.kkt_state
    with (
        physical.d_fixed_support.open_coefficient_lease() as d_lease,
        physical.residual_support.open_coefficient_lease() as residual_lease,
        live_cover_provider.open_live_cover_lease() as cover_lease,
        dual_source.open_dual_lease() as dual_lease,
        kkt.borrow_for_block_support_outside_pair(
            expected_problem_hash=cover.kkt_problem_hash,
            expected_policy_hash=cover.kkt_policy_hash,
            expected_branch_receipt_hashes=kkt.branch_receipt_hashes,
        ) as kkt_lease,
    ):
        fixed, parent_packed, eligible, required = cover_lease.live_membership_sources()
        resource, kkt_eligible, kkt_required = kkt_lease.live_source_tensors()
        capacity_adjoint = dual_lease.live_capacity_adjoint()
        lam, rho, _mu, _limits = dual_lease.compact_duals
        device = d_lease.tensor.device
        tensors = (
            residual_lease.tensor,
            fixed,
            parent_packed,
            eligible,
            required,
            resource,
            kkt_eligible,
            kkt_required,
            capacity_adjoint,
        )
        if (
            device.type != "cuda"
            or str(device) != cover.resident_device_identity
            or any(value.device != device for value in tensors)
            or eligible.data_ptr() != kkt_eligible.data_ptr()
            or required.data_ptr() != kkt_required.data_ptr()
            or cover_lease.live_cover_receipt_hash != cover.receipt_hash
            or cover_lease.source_native_pair_axis_commitment_hash
            != cover.source_native_pair_axis_commitment_hash
            or kkt_lease.state_identity_hash != cover.kkt_state_identity_hash
            or kkt_lease.eligible_mask_identity_hash
            != cover.kkt_eligible_mask_identity_hash
            or kkt_lease.required_mask_identity_hash
            != cover.kkt_required_mask_identity_hash
            or cover_lease.route_scale_d2h_bytes != 0
            or cover_lease.route_scale_content_digest_bytes != 0
            or kkt_lease.route_scale_d2h_bytes != 0
            or kkt_lease.route_scale_content_digest_bytes != 0
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.MUTATED_CUSTODY,
                "live source tensors differ from their sealed provider identities",
            )
        registration = live_cover_provider.block_witness_source.registration
        scan = _scan_live_outside_pairs(
            design=design,
            d_coefficients=d_lease.tensor,
            residual_coefficients=residual_lease.tensor,
            resource_coefficients=resource,
            capacity_adjoint=capacity_adjoint,
            required=required,
            d_weight_numerator=registration.d_weight_numerator,
            residual_weight_numerator=registration.residual_weight_numerator,
            population_weight_denominator=registration.population_weight_denominator,
            d_source_l1_radius=d_receipt.coefficient_construction_l1_radius_upper,
            residual_source_l1_radius=residual_receipt.coefficient_construction_l1_radius_upper,
            resource_multiplier=lam,
            resource_floor=rho,
            capacity_multiplier_dot_limit=dual.capacity_multiplier_dot_limit_nominal,
            resource_multiplier_conversion_radius=dual.resource_multiplier_conversion_radius,
            capacity_adjoint_l1_radius=dual.capacity_adjoint_roundoff_l1_radius,
            capacity_adjoint_linf_radius=dual.capacity_adjoint_roundoff_linf_radius,
            dual_constant_radius=dual.dual_constant_roundoff_radius,
            eligible=eligible,
            fixed_membership=fixed,
            resident_parent_packed=parent_packed,
            shell_indices=cover.shell_indices,
            rounding_anchor_shell_indices=cover.rounding_anchor_shell_route_indices,
            comparison_preferred_shell_indices=preferred_shell,
        )
    first = scan.first_rectangle
    second = scan.second_rectangle
    if (
        scan.rounding_anchor_count != cover.rounding_anchor_count
        or scan.comparison_preferred_count != cover.comparison_preferred_selected_count
        or scan.required_count != cover.required_count
        or scan.selectable_count != cover.selectable_count
        or scan.shell_count != cover.shell_count
        or first.p_count != cover.add_outside_droppable_count
        or first.q_count != cover.add_outside_addable_count
        or second.p_count != cover.drop_outside_add_inside_droppable_count
        or second.q_count != cover.drop_outside_add_inside_addable_count
        or first.pair_count != cover.add_outside_rectangle_pair_count
        or second.pair_count != cover.drop_outside_add_inside_rectangle_pair_count
        or first.pair_count + second.pair_count != cover.complete_pair_count
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.SOURCE_MISMATCH,
            "numerical route scan does not replay the source-native live cover",
        )
    first_lower, second_lower = _pair_lowers(scan)
    worst = scan.worst_rectangle
    preferred_dot_radius = _up(
        scan.support_source_l1_radius
        + scan.weight_conversion_l1_radius
        + scan.weighted_combination_l1_radius
        + scan.preferred_dot_reduction_radius
    )
    return V1BlockSupportOutsidePairLiveFamilyReceipt(
        live_cover_receipt_hash=cover.receipt_hash,
        source_native_pair_axis_commitment_hash=cover.source_native_pair_axis_commitment_hash,
        physical_block_support_binding_receipt_hash=binding.receipt_hash,
        structural_domain_receipt_hash=cover.structural_domain_receipt_hash,
        d_fixed_support_receipt_hash=d_receipt.receipt_hash,
        residual_support_receipt_hash=residual_receipt.receipt_hash,
        dual_source_receipt_hash=dual.receipt_hash,
        block_support_registration_hash=cover.block_support_registration_hash,
        block_index=cover.block_index,
        block_id=cover.block_id,
        block_context_axis_hash=cover.block_context_axis_hash,
        residual_source_context_axis_hash=cover.residual_source_context_axis_hash,
        source_registration_hash=cover.source_registration_hash,
        model_identity_hash=cover.model_identity_hash,
        chart_identity_hash=cover.chart_identity_hash,
        current_mask_hash=cover.current_mask_hash,
        proposal_generation=cover.proposal_generation,
        route_axis_hash=cover.route_axis_hash,
        route_id_axis_hash=cover.route_id_axis_hash,
        registered_d_prefix_hash=cover.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=cover.registered_d_w2_prefix_hash,
        native_fiber_axis_receipt_hash=cover.native_fiber_axis_receipt_hash,
        native_fiber_measure_hash=cover.native_fiber_measure_hash,
        preferred_action_hash=cover.comparison_preferred_action_hash,
        preferred_action_binding_hash=cover.preferred_action_binding_hash,
        preferred_removed_route_index_hash=cover.comparison_preferred_removed_route_index_hash,
        preferred_selected_count=cover.comparison_preferred_selected_count,
        rounding_anchor_action_coordinate_hash=cover.rounding_anchor_action_coordinate_hash,
        rounding_anchor_removed_route_index_hash=cover.rounding_anchor_removed_route_index_hash,
        rounding_anchor_subset_identity_hash=cover.rounding_anchor_subset_identity_hash,
        rounding_anchor_selected_count=scan.rounding_anchor_count,
        alias_hash=cover.alias_hash,
        branch_id=cover.branch_id,
        fd_rank=cover.fd_rank,
        resource_alias_id=cover.resource_alias_id,
        source_receipt_hash=cover.source_receipt_hash,
        source_replay_hash=cover.source_replay_hash,
        cardinality_alias_id=cover.cardinality_alias_id,
        alias_total_removed_count=cover.alias_total_removed_count,
        kkt_state_identity_hash=cover.kkt_state_identity_hash,
        kkt_problem_hash=cover.kkt_problem_hash,
        kkt_policy_hash=cover.kkt_policy_hash,
        kkt_branch_receipt_hash=cover.kkt_branch_receipt_hash,
        kkt_eligible_mask_identity_hash=cover.kkt_eligible_mask_identity_hash,
        kkt_required_mask_identity_hash=cover.kkt_required_mask_identity_hash,
        fixed_membership_binding_hash=cover.fixed_membership_binding_hash,
        parent_live_membership_identity_hash=cover.parent_live_membership_identity_hash,
        selectable_membership_identity_hash=cover.selectable_membership_identity_hash,
        add_outside_drop_axis_identity_hash=cover.add_outside_drop_axis_identity_hash,
        add_outside_add_axis_identity_hash=cover.add_outside_add_axis_identity_hash,
        drop_outside_add_inside_drop_axis_identity_hash=cover.drop_outside_add_inside_drop_axis_identity_hash,
        drop_outside_add_inside_add_axis_identity_hash=cover.drop_outside_add_inside_add_axis_identity_hash,
        resource_axis_hash=dual.resource_axis_hash,
        resource_coefficient_source_hash=dual.resource_coefficient_source_hash,
        capacity_operator_source_hash=dual.capacity_operator_source_hash,
        resource_multiplier_source_hash=dual.resource_multiplier_source_hash,
        capacity_multiplier_source_hash=dual.capacity_multiplier_source_hash,
        route_count=cover.route_count,
        capacity_count=dual.capacity_count,
        route_tile_size=design.route_tile_size,
        route_tile_scan_count=scan.route_tile_scan_count,
        complete_route_coefficient_scan_count=cover.route_count,
        live_cover_complete_route_scan_count=cover.complete_route_scan_count,
        add_outside_rectangle_pair_count=first.pair_count,
        drop_outside_add_inside_rectangle_pair_count=second.pair_count,
        complete_pair_cover_count=first.pair_count + second.pair_count,
        add_outside_rectangle_empty=not bool(first.pair_count),
        drop_outside_add_inside_rectangle_empty=not bool(second.pair_count),
        vacuous_no_outside_pair=worst is None,
        base_dual_nominal=scan.base_nominal,
        base_dual_downward_radius=scan.base_radius,
        base_dual_lower_outward=scan.base_lower,
        comparison_preferred_dot_nominal=scan.preferred_dot_nominal,
        comparison_preferred_dot_absolute_radius=preferred_dot_radius,
        comparison_preferred_dot_upper_outward=_up(
            scan.preferred_dot_nominal + preferred_dot_radius
        ),
        support_coefficient_source_l1_radius=scan.support_source_l1_radius,
        weight_conversion_l1_radius=scan.weight_conversion_l1_radius,
        weighted_combination_roundoff_l1_radius=scan.weighted_combination_l1_radius,
        resource_dual_roundoff_l1_radius=scan.resource_dual_l1_radius,
        capacity_dual_roundoff_l1_radius=scan.capacity_dual_l1_radius,
        stationarity_addition_roundoff_l1_radius=scan.stationarity_addition_l1_radius,
        box_reduction_roundoff_radius=scan.box_reduction_radius,
        dual_constant_roundoff_radius=scan.constant_radius,
        stationarity_sign_linf_radius=scan.stationarity_sign_linf_radius,
        ambiguous_p_endpoint_count=first.ambiguous_p_count + second.ambiguous_p_count,
        ambiguous_q_endpoint_count=first.ambiguous_q_count + second.ambiguous_q_count,
        add_outside_correction_lower=(
            0.0 if not first.pair_count else first.correction_lower
        ),
        drop_outside_add_inside_correction_lower=(
            0.0 if not second.pair_count else second.correction_lower
        ),
        add_outside_pair_lower_outward=first_lower,
        drop_outside_add_inside_pair_lower_outward=second_lower,
        worst_pair_rectangle=None if worst is None else worst.rectangle,
        worst_pair_preferred_selected_index=None if worst is None else worst.p_index,
        worst_pair_preferred_unselected_index=None if worst is None else worst.q_index,
        worst_pair_ordinal=None if worst is None else worst.pair_ordinal,
        worst_pair_lower_center=scan.worst_center,
        worst_pair_total_downward_radius=scan.worst_radius,
        worst_pair_lower_outward=scan.worst_lower,
        design_hash=design.design_hash,
        hbm_forecast_hash=hbm_forecast.forecast_hash,
        live_cover_design_hash=cover.design_hash,
        live_cover_hbm_forecast_hash=cover.hbm_forecast_hash,
        resident_device_identity=cover.resident_device_identity,
        scalar_device_to_host_transfer_bytes=scan.scalar_d2h_bytes,
        scan_device_to_host_synchronization_count=scan.d2h_synchronization_count,
        _issuer_token=_RECEIPT_TOKEN,
    )


def issue_v1_block_support_outside_pair_live_family_cuda_v3(
    *,
    live_cover_authority: V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2,
    live_cover_provider: V1BlockSupportOutsidePairLiveCoverProvider,
    dual_source: V1BlockSupportOutsidePairProductionDualSourceHandleV2,
    design: V1BlockSupportOutsidePairCUDAExecutionDesign,
    hbm_forecast: V1BlockSupportOutsidePairHBMForecast,
) -> V1BlockSupportOutsidePairLiveFamilyReceiptV3:
    """Issue the production-authority outside-family receipt.

    This sibling is intentionally separate from the legacy V2 numerical
    surface.  It accepts only the source-native cover authority and the exact
    problem/state-derived production dual handle.
    """

    if torch is None or not torch.cuda.is_available():
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.CUDA_REQUIRED,
            "production live-family V3 execution fails closed without CUDA",
        )
    if not torch.are_deterministic_algorithms_enabled():
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.DETERMINISM_REQUIRED,
            "production live-family V3 execution requires deterministic algorithms",
        )
    if (
        type(live_cover_authority)
        is not V1BlockSupportOutsidePairLiveCoverAuthorityReceiptV2
        or type(live_cover_provider) is not V1BlockSupportOutsidePairLiveCoverProvider
        or type(dual_source)
        is not V1BlockSupportOutsidePairProductionDualSourceHandleV2
        or type(design) is not V1BlockSupportOutsidePairCUDAExecutionDesign
        or type(hbm_forecast) is not V1BlockSupportOutsidePairHBMForecast
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.INVALID_TYPE,
            "production live-family V3 requires exact typed production parents",
        )
    cover, binding, d_receipt, residual_receipt, dual = _validate_source_join_v3(
        live_cover_authority=live_cover_authority,
        provider=live_cover_provider,
        dual_source=dual_source,
        design=design,
        forecast=hbm_forecast,
    )
    physical = live_cover_provider.physical_block_support_binding
    preferred_shell = _preferred_shell_routes(live_cover_provider)
    kkt = live_cover_provider.kkt_state
    with (
        physical.d_fixed_support.open_coefficient_lease() as d_lease,
        physical.residual_support.open_coefficient_lease() as residual_lease,
        live_cover_provider.open_live_cover_lease() as cover_lease,
        dual_source.open_dual_lease() as dual_lease,
        kkt.borrow_for_block_support_outside_pair(
            expected_problem_hash=cover.kkt_problem_hash,
            expected_policy_hash=cover.kkt_policy_hash,
            expected_branch_receipt_hashes=kkt.branch_receipt_hashes,
        ) as kkt_lease,
    ):
        fixed, parent_packed, eligible, required = cover_lease.live_membership_sources()
        resource, kkt_eligible, kkt_required = kkt_lease.live_source_tensors()
        capacity_adjoint = dual_lease.live_capacity_adjoint()
        lam, rho, _mu, _limits = dual_lease.compact_duals
        device = d_lease.tensor.device
        tensors = (
            residual_lease.tensor,
            fixed,
            parent_packed,
            eligible,
            required,
            resource,
            kkt_eligible,
            kkt_required,
            capacity_adjoint,
        )
        if (
            device.type != "cuda"
            or str(device) != cover.resident_device_identity
            or any(value.device != device for value in tensors)
            or eligible.data_ptr() != kkt_eligible.data_ptr()
            or required.data_ptr() != kkt_required.data_ptr()
            or cover_lease.live_cover_receipt_hash != cover.receipt_hash
            or cover_lease.source_native_pair_axis_commitment_hash
            != cover.source_native_pair_axis_commitment_hash
            or kkt_lease.state_identity_hash != cover.kkt_state_identity_hash
            or kkt_lease.eligible_mask_identity_hash
            != cover.kkt_eligible_mask_identity_hash
            or kkt_lease.required_mask_identity_hash
            != cover.kkt_required_mask_identity_hash
            or cover_lease.route_scale_d2h_bytes != 0
            or cover_lease.route_scale_content_digest_bytes != 0
            or kkt_lease.route_scale_d2h_bytes != 0
            or kkt_lease.route_scale_content_digest_bytes != 0
            or dual_lease.dual_source_receipt_hash != dual.receipt_hash
            or dual_lease.route_scale_device_to_host_transfer_bytes != 0
            or dual_lease.route_scale_content_digest_bytes != 0
            or rho != dual.resource_floor_nominal
            or rho != live_cover_authority.registered_shell_resource_floor
            or len(_mu) != dual.capacity_count
            or len(_limits) != dual.capacity_count
        ):
            _refuse(
                V1BlockSupportOutsidePairLiveFamilyRefusalCode.MUTATED_CUSTODY,
                "live V3 source tensors differ from their sealed identities",
            )
        scan = _scan_live_outside_pairs(
            design=design,
            d_coefficients=d_lease.tensor,
            residual_coefficients=residual_lease.tensor,
            resource_coefficients=resource,
            capacity_adjoint=capacity_adjoint,
            required=required,
            d_weight_numerator=live_cover_authority.d_weight_numerator,
            residual_weight_numerator=live_cover_authority.residual_weight_numerator,
            population_weight_denominator=(
                live_cover_authority.population_weight_denominator
            ),
            d_source_l1_radius=d_receipt.coefficient_construction_l1_radius_upper,
            residual_source_l1_radius=residual_receipt.coefficient_construction_l1_radius_upper,
            resource_multiplier=lam,
            resource_floor=rho,
            capacity_multiplier_dot_limit=dual.capacity_multiplier_dot_limit_nominal,
            resource_multiplier_conversion_radius=dual.resource_multiplier_conversion_radius,
            capacity_adjoint_l1_radius=dual.capacity_adjoint_roundoff_l1_radius,
            capacity_adjoint_linf_radius=dual.capacity_adjoint_roundoff_linf_radius,
            dual_constant_radius=dual.dual_constant_roundoff_radius,
            eligible=eligible,
            fixed_membership=fixed,
            resident_parent_packed=parent_packed,
            shell_indices=cover.shell_indices,
            rounding_anchor_shell_indices=cover.rounding_anchor_shell_route_indices,
            comparison_preferred_shell_indices=preferred_shell,
        )
    first = scan.first_rectangle
    second = scan.second_rectangle
    if (
        scan.rounding_anchor_count != cover.rounding_anchor_count
        or scan.comparison_preferred_count != cover.comparison_preferred_selected_count
        or scan.required_count != cover.required_count
        or scan.selectable_count != cover.selectable_count
        or scan.shell_count != cover.shell_count
        or first.p_count != cover.add_outside_droppable_count
        or first.q_count != cover.add_outside_addable_count
        or second.p_count != cover.drop_outside_add_inside_droppable_count
        or second.q_count != cover.drop_outside_add_inside_addable_count
        or first.pair_count != cover.add_outside_rectangle_pair_count
        or second.pair_count != cover.drop_outside_add_inside_rectangle_pair_count
        or first.pair_count + second.pair_count != cover.complete_pair_count
        or first.p_count != live_cover_authority.add_outside_droppable_count
        or first.q_count != live_cover_authority.add_outside_addable_count
        or second.p_count
        != live_cover_authority.drop_outside_add_inside_droppable_count
        or second.q_count != live_cover_authority.drop_outside_add_inside_addable_count
        or first.pair_count != live_cover_authority.add_outside_rectangle_pair_count
        or second.pair_count
        != live_cover_authority.drop_outside_add_inside_rectangle_pair_count
        or first.pair_count + second.pair_count
        != live_cover_authority.complete_pair_count
    ):
        _refuse(
            V1BlockSupportOutsidePairLiveFamilyRefusalCode.SOURCE_MISMATCH,
            "V3 numerical scan does not replay the source-native live cover",
        )
    first_lower, second_lower = _pair_lowers(scan)
    worst = scan.worst_rectangle
    preferred_dot_radius = _up(
        scan.support_source_l1_radius
        + scan.weight_conversion_l1_radius
        + scan.weighted_combination_l1_radius
        + scan.preferred_dot_reduction_radius
    )
    return V1BlockSupportOutsidePairLiveFamilyReceiptV3(
        live_cover_authority_receipt_hash=live_cover_authority.receipt_hash,
        production_dual_source_receipt_hash=dual.receipt_hash,
        live_cover_source_native_payload_hash=(
            live_cover_authority.live_cover_source_native_payload_hash
        ),
        source_native_alias_commitment_hash=(
            live_cover_authority.source_native_alias_commitment_hash
        ),
        bound_witness_registration_receipt_hash=(
            live_cover_authority.bound_witness_registration_receipt_hash
        ),
        preoutcome_domain_receipt_hash=(
            live_cover_authority.preoutcome_domain_receipt_hash
        ),
        canonical_block_context_receipt_hash=(
            live_cover_authority.canonical_block_context_receipt_hash
        ),
        canonical_context_tuple_hash=live_cover_authority.canonical_context_tuple_hash,
        block_support_context_axis_hash=(
            live_cover_authority.block_support_context_axis_hash
        ),
        physical_block_support_binding_receipt_hash=binding.receipt_hash,
        structural_domain_receipt_hash=live_cover_authority.structural_domain_receipt_hash,
        d_fixed_support_receipt_hash=d_receipt.receipt_hash,
        residual_support_receipt_hash=residual_receipt.receipt_hash,
        block_index=live_cover_authority.block_index,
        block_id=live_cover_authority.block_id,
        source_registration_hash=live_cover_authority.source_registration_hash,
        model_identity_hash=live_cover_authority.model_identity_hash,
        chart_identity_hash=live_cover_authority.chart_identity_hash,
        current_mask_hash=live_cover_authority.current_mask_hash,
        proposal_generation=live_cover_authority.proposal_generation,
        route_axis_hash=live_cover_authority.route_axis_hash,
        route_id_axis_hash=live_cover_authority.route_id_axis_hash,
        native_fiber_axis_receipt_hash=(
            live_cover_authority.native_fiber_axis_receipt_hash
        ),
        native_fiber_measure_hash=live_cover_authority.native_fiber_measure_hash,
        registered_d_prefix_hash=live_cover_authority.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=live_cover_authority.registered_d_w2_prefix_hash,
        preferred_action_hash=live_cover_authority.preferred_action_hash,
        preferred_action_binding_hash=(
            live_cover_authority.preferred_action_binding_hash
        ),
        preferred_removed_route_index_hash=(
            live_cover_authority.preferred_removed_route_index_hash
        ),
        preferred_selected_count=live_cover_authority.preferred_selected_count,
        resource_alias_hash=live_cover_authority.resource_alias_hash,
        branch_id=live_cover_authority.branch_id,
        resource_alias_id=live_cover_authority.resource_alias_id,
        phase_a_source_receipt_hash=(live_cover_authority.phase_a_source_receipt_hash),
        phase_a_source_replay_hash=live_cover_authority.phase_a_source_replay_hash,
        source_resource_receipt_hash=live_cover_authority.source_resource_receipt_hash,
        resource_coordinate_hash=live_cover_authority.resource_coordinate_hash,
        phase_a_resource_cell_id=live_cover_authority.phase_a_resource_cell_id,
        phase_a_invariance_receipt_hash=(
            live_cover_authority.phase_a_invariance_receipt_hash
        ),
        registered_fd_rank=live_cover_authority.registered_fd_rank,
        cardinality_alias_id=live_cover_authority.cardinality_alias_id,
        fixed_outside_removed_count=(live_cover_authority.fixed_outside_removed_count),
        shell_top_k_count=live_cover_authority.shell_top_k_count,
        hard_total_removed_count=live_cover_authority.hard_total_removed_count,
        rounding_anchor_action_coordinate_hash=(
            live_cover_authority.rounding_anchor_action_coordinate_hash
        ),
        rounding_anchor_removed_route_index_hash=(
            live_cover_authority.rounding_anchor_removed_route_index_hash
        ),
        rounding_anchor_subset_identity_hash=(
            live_cover_authority.rounding_anchor_subset_identity_hash
        ),
        kkt_state_identity_hash=live_cover_authority.kkt_state_identity_hash,
        kkt_problem_hash=live_cover_authority.kkt_problem_hash,
        kkt_policy_hash=live_cover_authority.kkt_policy_hash,
        kkt_branch_receipt_hash=live_cover_authority.kkt_branch_receipt_hash,
        kkt_eligible_mask_identity_hash=(
            live_cover_authority.kkt_eligible_mask_identity_hash
        ),
        kkt_required_mask_identity_hash=(
            live_cover_authority.kkt_required_mask_identity_hash
        ),
        least_work_problem_hash=dual.least_work_problem_hash,
        source_problem_hash=dual.source_problem_hash,
        signed_work_operator_receipt_hash=dual.signed_work_operator_receipt_hash,
        fixed_membership_binding_hash=(
            live_cover_authority.fixed_membership_binding_hash
        ),
        parent_live_membership_identity_hash=(
            live_cover_authority.parent_live_membership_identity_hash
        ),
        selectable_membership_identity_hash=(
            live_cover_authority.selectable_membership_identity_hash
        ),
        add_outside_drop_axis_identity_hash=(
            live_cover_authority.add_outside_drop_axis_identity_hash
        ),
        add_outside_add_axis_identity_hash=(
            live_cover_authority.add_outside_add_axis_identity_hash
        ),
        drop_outside_add_inside_drop_axis_identity_hash=(
            live_cover_authority.drop_outside_add_inside_drop_axis_identity_hash
        ),
        drop_outside_add_inside_add_axis_identity_hash=(
            live_cover_authority.drop_outside_add_inside_add_axis_identity_hash
        ),
        source_native_pair_axis_commitment_hash=(
            live_cover_authority.source_native_pair_axis_commitment_hash
        ),
        resource_axis_hash=dual.resource_axis_hash,
        resource_coefficient_source_hash=dual.resource_coefficient_source_hash,
        capacity_operator_source_hash=dual.capacity_operator_source_hash,
        resource_multiplier_source_hash=dual.resource_multiplier_source_hash,
        capacity_multiplier_source_hash=dual.capacity_multiplier_source_hash,
        operator_scales_hash=dual.operator_scales_hash,
        route_count=live_cover_authority.route_count,
        capacity_count=dual.capacity_count,
        route_tile_size=design.route_tile_size,
        route_tile_scan_count=scan.route_tile_scan_count,
        complete_route_coefficient_scan_count=cover.route_count,
        live_cover_complete_route_scan_count=(
            live_cover_authority.complete_route_scan_count
        ),
        add_outside_droppable_count=(live_cover_authority.add_outside_droppable_count),
        add_outside_addable_count=live_cover_authority.add_outside_addable_count,
        drop_outside_add_inside_droppable_count=(
            live_cover_authority.drop_outside_add_inside_droppable_count
        ),
        drop_outside_add_inside_addable_count=(
            live_cover_authority.drop_outside_add_inside_addable_count
        ),
        add_outside_rectangle_pair_count=first.pair_count,
        drop_outside_add_inside_rectangle_pair_count=second.pair_count,
        complete_pair_cover_count=first.pair_count + second.pair_count,
        add_outside_rectangle_empty=not bool(first.pair_count),
        drop_outside_add_inside_rectangle_empty=not bool(second.pair_count),
        vacuous_no_outside_pair=worst is None,
        base_dual_nominal=scan.base_nominal,
        base_dual_downward_radius=scan.base_radius,
        base_dual_lower_outward=scan.base_lower,
        comparison_preferred_dot_nominal=scan.preferred_dot_nominal,
        comparison_preferred_dot_absolute_radius=preferred_dot_radius,
        comparison_preferred_dot_upper_outward=_up(
            scan.preferred_dot_nominal + preferred_dot_radius
        ),
        support_coefficient_source_l1_radius=scan.support_source_l1_radius,
        weight_conversion_l1_radius=scan.weight_conversion_l1_radius,
        weighted_combination_roundoff_l1_radius=scan.weighted_combination_l1_radius,
        resource_dual_roundoff_l1_radius=scan.resource_dual_l1_radius,
        capacity_dual_roundoff_l1_radius=scan.capacity_dual_l1_radius,
        stationarity_addition_roundoff_l1_radius=scan.stationarity_addition_l1_radius,
        box_reduction_roundoff_radius=scan.box_reduction_radius,
        dual_constant_roundoff_radius=scan.constant_radius,
        stationarity_sign_linf_radius=scan.stationarity_sign_linf_radius,
        ambiguous_p_endpoint_count=first.ambiguous_p_count + second.ambiguous_p_count,
        ambiguous_q_endpoint_count=first.ambiguous_q_count + second.ambiguous_q_count,
        add_outside_correction_lower=(
            0.0 if not first.pair_count else first.correction_lower
        ),
        drop_outside_add_inside_correction_lower=(
            0.0 if not second.pair_count else second.correction_lower
        ),
        add_outside_pair_lower_outward=first_lower,
        drop_outside_add_inside_pair_lower_outward=second_lower,
        worst_pair_rectangle=None if worst is None else worst.rectangle,
        worst_pair_preferred_selected_index=None if worst is None else worst.p_index,
        worst_pair_preferred_unselected_index=None if worst is None else worst.q_index,
        worst_pair_ordinal=None if worst is None else worst.pair_ordinal,
        worst_pair_lower_center=scan.worst_center,
        worst_pair_total_downward_radius=scan.worst_radius,
        worst_pair_lower_outward=scan.worst_lower,
        design_hash=design.design_hash,
        hbm_forecast_hash=hbm_forecast.forecast_hash,
        resident_device_identity=cover.resident_device_identity,
        scalar_device_to_host_transfer_bytes=scan.scalar_d2h_bytes,
        scan_device_to_host_synchronization_count=scan.d2h_synchronization_count,
        _issuer_token=_RECEIPT_V3_TOKEN,
    )


__all__ = (
    "OUTSIDE_PAIR_LIVE_FAMILY_AUTHORITY",
    "OUTSIDE_PAIR_LIVE_FAMILY_EQUATION",
    "OUTSIDE_PAIR_LIVE_FAMILY_REFERENCE_SCHEMA",
    "OUTSIDE_PAIR_LIVE_FAMILY_SCHEMA",
    "OUTSIDE_PAIR_LIVE_FAMILY_V3_AUTHORITY",
    "OUTSIDE_PAIR_LIVE_FAMILY_V3_EQUATION",
    "OUTSIDE_PAIR_LIVE_FAMILY_V3_SCHEMA",
    "V1BlockSupportOutsidePairLiveFamilyReceipt",
    "V1BlockSupportOutsidePairLiveFamilyReceiptV3",
    "V1BlockSupportOutsidePairLiveFamilyReference",
    "V1BlockSupportOutsidePairLiveFamilyRefusal",
    "V1BlockSupportOutsidePairLiveFamilyRefusalCode",
    "evaluate_v1_block_support_outside_pair_live_family_reference",
    "issue_v1_block_support_outside_pair_live_family_cuda",
    "issue_v1_block_support_outside_pair_live_family_cuda_v3",
)
