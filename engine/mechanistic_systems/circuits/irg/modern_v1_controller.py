"""Pure state machine for the genuine partition-free modern-v1 controller.

This module is the only owner of live parent-to-exact-child mutation inside the
modern-v1 development path.  Proposal geometry, candidate composition, and the
staged-alpha decision objects remain authority inert.  A resolved accepted
look is checked again inside this state machine, a private single-use commit
permit is born and consumed immediately, and only an inert transition receipt
leaves the boundary.

The public state is immutable.  Every function returns a new state, making the
logical controller replayable and independent of execution batching.  Selector
promotion and scientific-claim authority are deliberately absent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field, fields, replace
from numbers import Integral
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .contracts import content_hash
from .modern_v1_acquisition import V1FQAuthorizedAcquisition, V1RelinearizedAcquisition
from .modern_v1_campaign_contract import (
    V1OptionACampaignBinding,
    v1_dense_referent_mask_hash,
)
from .modern_v1_candidate_composition import (
    FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID,
    V1CandidateProposal,
    V1FQCandidateProposal,
    V1RemovedRouteSetIdentity,
)
from .modern_v1_core import V1CertificateFreeze
from .modern_v1_fqr_runtime_acquisition import (
    V1FQRAuthorizedAcquisition,
    V1FQRCandidateProposal,
)
from .modern_v1_signed_work_runtime_acquisition import (
    V1SignedWorkAuthorizedAcquisition,
    V1SignedWorkAuthorizedAcquisitionV2,
    V1SignedWorkCandidateProposal,
)
from .modern_v1_signed_work_label_native_controller_acquisition import (
    V1LabelNativeControllerCandidateCarrier,
    V1LabelNativeControllerSingleEndpointAcquisition,
)
from .modern_v1_signed_work_label_native_runtime_acquisition import (
    V1LabelNativeRuntimeFallbackPlan,
)
from .modern_v1_fqr_horizon import (
    FQR_NO_CANDIDATE_HORIZON_STOP_REASON,
    V1ControllerProposalHorizonWitness,
)
from .modern_v1_receipts import (
    ControllerPhase,
    V1CandidateAttemptReceipt,
    V1ControllerSnapshotReceipt,
    V1MaskTransitionReceipt,
)
from .modern_v1_pareto_milestones import (
    ALL_REGISTERED_ACTIONS_INFEASIBLE_AT_FULL_LOOK,
    ALL_REGISTERED_FQR_ACTIONS_REJECTED_BY_PAIRED_BCE_BARRIER,
    REGISTERED_MINIMUM_RETAINED_ROUTE_COUNT_REACHED,
    V1ExactPrefixRungRegistration,
    V1ParetoMilestoneCheckpointBinding,
    V1ParetoMilestonePending,
    V1ParetoMilestoneReceipt,
    V1ParetoMilestoneRegistration,
    V1ScientificHorizonDisposition,
    classify_v1_scientific_horizon,
)
from .modern_v1_paired_bce_acquisition import (
    V1PairedBCEAcquisitionRegistration,
    V1PairedBCEAcquisitionState,
    V1PairedBCECalibrationHorizonSkipDecision,
    V1PairedBCEDecision,
    initial_paired_bce_acquisition_state,
)
from .modern_v1_staged_alpha import (
    STAGED_ALPHA_SAMPLING_MASS_TOTAL,
    V1StagedAlphaDecision,
    V1StagedAlphaLedger,
    V1StagedAlphaPacketEnvelope,
    V1StagedAlphaStepFreeze,
    V1StagedAlphaTrajectoryLedger,
    advance_v1_staged_alpha_ledger,
    advance_v1_staged_alpha_trajectory,
    bind_v1_staged_alpha_step,
    evaluate_v1_staged_alpha_look,
    quantize_v1_staged_alpha_sampling_masses,
)

_SAMPLING_LAW_SCHEMA: Final = "partition_free_v1_step_sampling_law_v2"
_THRESHOLD_SCHEMA: Final = "partition_free_v1_candidate_damage_threshold_v1"
_FROZEN_AFFINE_CALIBRATION_SCHEMA: Final = (
    "partition_free_v1_frozen_affine_calibration_v1"
)
_CANDIDATE_CALIBRATION_FREEZE_SCHEMA: Final = (
    "partition_free_v1_candidate_calibration_freeze_v1"
)
_CALIBRATION_FREEZE_METADATA_KEY: Final = "candidate_calibration_freeze"
_CALIBRATION_FREEZE_HASH_METADATA_KEY: Final = "candidate_calibration_freeze_hash"
_ACTIVE_SCHEMA: Final = "partition_free_v1_active_candidate_v1"
_CONTROLLER_SCHEMA: Final = "partition_free_v1_controller_state_v3"
_NULL_ID: Final = "conditional_mean_direct_increment_at_most_frozen_threshold_v1"
_COMMIT_PERMIT_SCHEMA: Final = "partition_free_v1_private_mask_commit_permit_v1"
_LEGACY_ACCEPTED_ORACLE_TRAJECTORY_ENTRY_SCHEMA: Final = (
    "partition_free_v1_accepted_oracle_trajectory_entry_v1"
)
_ACCEPTED_ORACLE_TRAJECTORY_ENTRY_SCHEMA: Final = (
    "partition_free_v1_accepted_oracle_trajectory_entry_v2"
)
_ACCEPTED_ORACLE_TRAJECTORY_SCHEMA: Final = (
    "partition_free_v1_accepted_oracle_trajectory_v1"
)
_COMMIT_TOKEN: Final = object()
FQR_D_DECISION_STABILITY_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_fqr_D_decision_stability_certificate_v1"
)
FQR_D_DECISION_STABILITY_CERTIFICATE_PREFIX: Final = (
    "partition-free-v1-fqr-D-decision-stability-certificate-"
)
FQR_D_DECISION_STABILITY_REFUSAL_SCHEMA: Final = (
    "partition_free_v1_fqr_D_decision_stability_refusal_v1"
)
FQR_D_DECISION_STABILITY_REFUSAL_PREFIX: Final = (
    "partition-free-v1-fqr-D-decision-stability-refusal-"
)
_FQR_D_DECISION_STABILITY_ISSUANCE_TOKEN: Final = object()
_FQR_D_DECISION_STABILITY_REFUSAL_ISSUANCE_TOKEN: Final = object()

OracleAcceptanceProtocol = Literal["legacy_staged_alpha", "paired_bce"]

StopReason = Literal[
    "registered_minimum_retained_route_count_reached",
    "all_registered_current_generation_actions_certified_infeasible_at_full_look",
    "all_registered_current_generation_fqr_actions_rejected_by_paired_bce_barrier",
    "all_registered_frozen_current_generation_fqr_endpoints_certified_infeasible_at_stable_C_horizon",
    "maximum_step_count_reached",
    "maximum_candidate_count_reached",
    "maximum_removed_count_reached",
    "signed_work_resolved_prefix_exhausted_needs_more_C",
    "global_queue_exhausted",
    "operator_no_admissible_candidate",
    "explicit_registered_stop",
]

SIGNED_WORK_RESOLVED_PREFIX_EXHAUSTED_NEEDS_MORE_C: Final = (
    "signed_work_resolved_prefix_exhausted_needs_more_C"
)

from .modern_v1_operator_acquisition import V1OperatorAcquisition, V1OperatorCandidate

ModernV1Acquisition = (
    V1RelinearizedAcquisition
    | V1FQAuthorizedAcquisition
    | V1FQRAuthorizedAcquisition
    | V1SignedWorkAuthorizedAcquisition
    | V1SignedWorkAuthorizedAcquisitionV2
    | V1LabelNativeControllerSingleEndpointAcquisition
    | V1OperatorAcquisition
)
ModernV1Proposal = (
    V1CandidateProposal
    | V1FQCandidateProposal
    | V1FQRCandidateProposal
    | V1SignedWorkCandidateProposal
    | V1LabelNativeControllerCandidateCarrier
    | V1OperatorCandidate
)


def _is_modern_v1_acquisition(value: object) -> bool:
    return type(value) in {
        V1OperatorAcquisition,
        V1RelinearizedAcquisition,
        V1FQAuthorizedAcquisition,
        V1FQRAuthorizedAcquisition,
        V1SignedWorkAuthorizedAcquisition,
        V1SignedWorkAuthorizedAcquisitionV2,
        V1LabelNativeControllerSingleEndpointAcquisition,
    }


def _is_modern_v1_proposal(value: object) -> bool:
    return type(value) in {
        V1OperatorCandidate,
        V1CandidateProposal,
        V1FQCandidateProposal,
        V1FQRCandidateProposal,
        V1SignedWorkCandidateProposal,
        V1LabelNativeControllerCandidateCarrier,
    }


def _v1_controller_proposal_mask_lineage_hash(
    mask_lineage_hashes: tuple[str, ...],
    transition_receipt_hashes: tuple[str, ...],
) -> str:
    """Replay the proposal runtime's exact committed-lineage identity."""

    return content_hash(
        {
            "schema": "partition_free_v1_runtime_proposal_mask_lineage_v1",
            "mask_lineage_hashes": mask_lineage_hashes,
            "transition_receipt_hashes": transition_receipt_hashes,
        },
        prefix="partition-free-v1-runtime-proposal-mask-lineage-",
    )


def _is_label_native_acquisition(value: object) -> bool:
    return type(value) is V1LabelNativeControllerSingleEndpointAcquisition


def _selected_proposal(
    acquisition: ModernV1Acquisition,
    *,
    queue_cursor: int,
) -> ModernV1Proposal:
    """Return the exact active proposal without inventing an eager queue.

    A label-native controller acquisition deliberately carries one selected
    endpoint even when its immutable plan coordinate is greater than zero.
    Its global cursor therefore indexes the receipt-only plan, not the
    singleton ``acquisition.queue`` compatibility view.
    """

    if _is_label_native_acquisition(acquisition):
        if queue_cursor != acquisition.receipt.plan_queue_index:
            raise ValueError(
                "label-native controller cursor differs from its selected plan coordinate"
            )
        return acquisition.candidate
    if queue_cursor >= len(acquisition.queue):
        raise ValueError("global queue cursor is exhausted")
    return acquisition.queue[queue_cursor]


def _attempt_queue_index(
    acquisition: ModernV1Acquisition,
    *,
    queue_cursor: int,
) -> int:
    if _is_label_native_acquisition(acquisition):
        if queue_cursor != acquisition.receipt.plan_queue_index:
            raise ValueError(
                "label-native attempt cursor differs from its frozen plan coordinate"
            )
        return acquisition.receipt.plan_queue_index
    return queue_cursor


def _attempt_history_key(attempt: V1CandidateAttemptReceipt) -> str:
    plan_hash = attempt.metadata.get("label_native_runtime_plan_hash")
    if plan_hash is None:
        return attempt.acquisition_receipt_hash
    return _nonempty(plan_hash, "label_native_runtime_plan_hash")


def _candidate_queue_order_metadata(
    acquisition: ModernV1Acquisition,
) -> Mapping[str, object]:
    if type(acquisition) is V1OperatorAcquisition:
        acquisition.assert_integrity()
        return dict(candidate_queue_order_primary='single_frozen_operator_binary_action',
                    operator_physical_progress_hash=acquisition.receipt.physical_progress_hash,
                    hard_global_optimality_claim=False,oracle_conditioned_composition=False)
    if _is_label_native_acquisition(acquisition):
        receipt = acquisition.receipt
        descriptor = acquisition.candidate.descriptor
        return {
            "candidate_queue_order_primary": (
                "receipt_only_prefrozen_signed_work_direct_D_resource_descending"
            ),
            "candidate_queue_order_secondary": (
                "same_plan_paired_O_fallback_without_recomposition_or_eager_masks"
            ),
            "candidate_queue_order_tertiary": "pre_C_resource_priority_coordinate",
            "label_native_runtime_plan_hash": receipt.runtime_plan_hash,
            "label_native_plan_queue_index": receipt.plan_queue_index,
            "label_native_queue_descriptor_count": receipt.queue_descriptor_count,
            "label_native_ordered_descriptor_axis_hash": (
                receipt.ordered_descriptor_axis_hash
            ),
            "label_native_runtime_descriptor_hash": receipt.runtime_descriptor_hash,
            "label_native_selected_acquisition_receipt_hash": receipt.receipt_hash,
            "label_native_selected_materialization_receipt_hash": (
                receipt.selected_materialization_receipt_hash
            ),
            "label_native_selected_coordinate_hash": descriptor.coordinate_hash,
            "label_native_blocked_unresolved_coordinate_hash": None,
            "complete_resolved_prefix_frozen_before_calibration_K_and_paired_oracle_O": True,
            "exactly_one_endpoint_materialized_at_current_cursor": True,
            "unselected_fallback_masks_materialized": False,
            "provider_handle_retained": False,
            "same_plan_fallback_without_recomposition": True,
            "oracle_conditioned_composition": False,
            "hard_global_optimality_claim": False,
            "omitted_mask_optimality_authority": False,
            "additive_surrogate_used": False,
        }
    if type(acquisition) is V1SignedWorkAuthorizedAcquisitionV2:
        return {
            "candidate_queue_order_primary": (
                "prefrozen_signed_work_direct_D_certified_resource_descending_endpoint"
            ),
            "candidate_queue_order_secondary": (
                "same_generation_paired_O_fallback_without_recomposition"
            ),
            "candidate_queue_order_tertiary": "pre_C_resource_priority_coordinate",
            "canonical_signed_work_six_role_queue_preserved_without_reorder": True,
            "complete_resolved_prefix_queue_frozen_before_calibration_K_and_paired_oracle_O": True,
            "resolved_prefix_queue_frozen_before_K_O": True,
            "canonical_signed_work_queue_length": len(acquisition.queue),
            "canonical_signed_work_ordered_candidate_ids_hash": content_hash(
                acquisition.receipt.ordered_candidate_ids,
                prefix=(
                    "partition-free-v1-signed-work-controller-ordered-candidate-axis-"
                ),
            ),
            "canonical_signed_work_composition_hash": acquisition.composition.result_hash,
            "canonical_signed_work_blocked_unresolved_coordinate_hash": (
                acquisition.composition.blocked_unresolved_coordinate_hash
            ),
            "canonical_signed_work_fallback_exhaustion_status": (
                acquisition.composition.fallback_exhaustion_status
            ),
            "direct_D_owns_finite_response_admission": True,
            "QR_transport_is_label_not_hidden_C_veto": True,
            "oracle_conditioned_composition": False,
            "hard_global_optimality_claim": False,
            "omitted_mask_optimality_authority": False,
            "additive_surrogate_used": False,
        }
    if type(acquisition) is V1SignedWorkAuthorizedAcquisition:
        return {
            "candidate_queue_order_primary": (
                "single_source_replayed_signed_work_direct_D_certified_endpoint"
            ),
            "candidate_queue_order_secondary": (
                "no_same_generation_fallback_queue_authorized_by_this_carrier"
            ),
            "candidate_queue_order_tertiary": "pre_C_resource_priority_coordinate",
            "canonical_signed_work_six_role_queue_preserved_without_filter_or_reorder": True,
            "complete_queue_frozen_before_calibration_K_and_paired_oracle_O": True,
            "canonical_signed_work_queue_length": len(acquisition.queue),
            "canonical_signed_work_ordered_candidate_ids_hash": content_hash(
                acquisition.receipt.ordered_candidate_ids,
                prefix=(
                    "partition-free-v1-signed-work-controller-ordered-candidate-axis-"
                ),
            ),
            "canonical_signed_work_composition_hash": acquisition.composition.result_hash,
            "direct_D_owns_finite_response_admission": True,
            "QR_transport_is_label_not_hidden_C_veto": True,
            "oracle_conditioned_composition": False,
            "hard_global_optimality_claim": False,
            "omitted_mask_optimality_authority": False,
            "additive_surrogate_used": False,
        }
    if type(acquisition) is V1FQRAuthorizedAcquisition:
        return {
            "candidate_queue_order_primary": (
                "largest_prefrozen_fqr_certified_physical_resource_release_first"
            ),
            "candidate_queue_order_secondary": (
                "smaller_prefrozen_fqr_candidate_after_paired_oracle_rejection"
            ),
            "candidate_queue_order_tertiary": (
                "branch_value_then_packed_action_hash_then_endpoint_index"
            ),
            "canonical_fqr_cross_branch_queue_preserved_without_filter_or_reorder": True,
            "complete_queue_frozen_before_calibration_K_and_paired_oracle_O": True,
            "canonical_fqr_complete_queue_length": len(acquisition.queue),
            "canonical_fqr_ordered_candidate_ids_hash": content_hash(
                acquisition.receipt.ordered_candidate_ids,
                prefix="partition-free-v1-fqr-controller-ordered-candidate-axis-",
            ),
            "canonical_fqr_composition_hash": acquisition.composition.result_hash,
            "oracle_conditioned_composition": False,
            "fqr_monotonicity_assumed": False,
            "hard_global_optimality_claim": False,
            "additive_surrogate_used": False,
        }
    if type(acquisition) is V1FQAuthorizedAcquisition:
        return {
            "candidate_queue_order_primary": (
                "largest_fq_authorized_physical_resource_release_first"
            ),
            "candidate_queue_order_secondary": (
                "smaller_fq_certified_candidate_after_direct_oracle_rejection"
            ),
            "candidate_queue_order_tertiary": "selected_shell_fq_authorization_order",
            "fq_authorized_queue_preserved_without_filter_or_reorder": True,
            "fq_monotonicity_assumed": False,
            "exact_registered_surrogate_evaluation": False,
            "additive_surrogate_used": False,
        }
    if (
        acquisition.composition.composer_id
        == FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID
    ):
        return {
            "candidate_queue_order_primary": (
                "largest_frozen_physical_resource_release_first"
            ),
            "candidate_queue_order_secondary": (
                "smaller_frozen_candidate_after_direct_oracle_rejection"
            ),
            "candidate_queue_order_tertiary": "declared_selector_frontier_order",
            "legacy_signed_quadratic_surrogate_used_for_queue_order": False,
        }
    return {
        "candidate_queue_order_primary": (
            "minimize_registered_damage_screen_per_resource"
        ),
        "candidate_queue_order_secondary": (
            "minimize_predicted_surrogate_damage_per_resource"
        ),
        "candidate_queue_order_tertiary": (
            "minimize_registered_damage_screen_then_maximize_resource_gain"
        ),
    }


def _queue_exhaustion_stop_reason(
    acquisition: ModernV1Acquisition,
) -> StopReason:
    """Name unresolved signed-work C rather than generic queue exhaustion.

    A V2 signed-work queue may be only the maximal decision-resolved prefix of
    its pre-C action order.  Exhausting that prefix cannot mean scientific
    saturation when a lower candidate is still unresolved on C.  This helper
    is intentionally narrow: a fully resolved signed-work prefix still has no
    terminal authority and therefore remains generic queue exhaustion until a
    separate completeness/saturation certificate exists.
    """

    if _is_label_native_acquisition(acquisition):
        raise PermissionError(
            "label-native exhaustion must be resolved from the retained runtime plan"
        )
    if type(acquisition) is V1SignedWorkAuthorizedAcquisitionV2:
        acquisition.composition.assert_integrity()
        status = acquisition.composition.fallback_exhaustion_status
        blocker = acquisition.composition.blocked_unresolved_coordinate_hash
        if status == "blocked_on_more_C":
            if type(blocker) is not str or not blocker:
                raise PermissionError(
                    "blocked signed-work fallback exhaustion lacks its C coordinate"
                )
            return SIGNED_WORK_RESOLVED_PREFIX_EXHAUSTED_NEEDS_MORE_C
        if status != "fully_resolved_no_more_candidate" or blocker is not None:
            raise PermissionError(
                "signed-work fallback exhaustion status is internally inconsistent"
            )
    return "global_queue_exhausted"


def _label_native_plan_exhaustion_stop_reason(
    plan: V1LabelNativeRuntimeFallbackPlan,
) -> StopReason:
    plan.assert_integrity()
    if plan.fallback_exhaustion_status == "blocked_on_more_C":
        if not plan.blocked_unresolved_coordinate_hash:
            raise PermissionError(
                "blocked label-native plan exhaustion lacks its C coordinate"
            )
        return SIGNED_WORK_RESOLVED_PREFIX_EXHAUSTED_NEEDS_MORE_C
    if (
        plan.fallback_exhaustion_status != "fully_resolved_no_more_candidate"
        or plan.blocked_unresolved_coordinate_hash is not None
    ):
        raise PermissionError(
            "label-native plan exhaustion status is internally inconsistent"
        )
    return "global_queue_exhausted"


def _label_native_plan_attempt_metadata(
    state: V1ControllerState,
) -> dict[str, object]:
    plan = state.label_native_runtime_plan
    if plan is None or not _is_label_native_acquisition(state.acquisition):
        return {}
    return {
        "label_native_runtime_plan_hash": plan.plan_hash,
        "label_native_plan_queue_index": state.queue_cursor,
        "label_native_rejected_plan_indices_before_attempt": (
            state.label_native_rejected_plan_indices
        ),
        "label_native_blocked_unresolved_coordinate_hash": (
            plan.blocked_unresolved_coordinate_hash
        ),
        "label_native_fallback_exhaustion_status": (
            plan.fallback_exhaustion_status
        ),
        "label_native_plan_retained_without_provider_or_masks": True,
        "post_O_composition_or_reorder": False,
    }


def _acquisition_invalidation_chart_hash(acquisition: ModernV1Acquisition) -> str:
    if type(acquisition) in {
        V1OperatorAcquisition,
        V1FQAuthorizedAcquisition,
        V1FQRAuthorizedAcquisition,
        V1SignedWorkAuthorizedAcquisition,
        V1SignedWorkAuthorizedAcquisitionV2,
        V1LabelNativeControllerSingleEndpointAcquisition,
    }:
        return acquisition.invalidation_chart_hash
    return acquisition.chart.chart_hash


def _acquisition_invalidation_composition_hash(
    acquisition: ModernV1Acquisition,
) -> str:
    if type(acquisition) is V1OperatorAcquisition:
        return acquisition.acquisition_hash
    if _is_label_native_acquisition(acquisition):
        return acquisition.receipt.runtime_plan_hash
    return acquisition.composition.result_hash


def _acquisition_proposal_diagnostics(
    acquisition: ModernV1Acquisition,
    *,
    queue_cursor: int,
) -> object:
    if type(acquisition) in {
        V1OperatorAcquisition,
        V1FQAuthorizedAcquisition,
        V1FQRAuthorizedAcquisition,
        V1SignedWorkAuthorizedAcquisition,
        V1SignedWorkAuthorizedAcquisitionV2,
        V1LabelNativeControllerSingleEndpointAcquisition,
    }:
        return None
    return (
        acquisition.chart.build_contract.metadata.get("proposal_diagnostics")
        if queue_cursor == 0
        else None
    )


def _proposal_attempt_metadata(proposal: ModernV1Proposal) -> dict[str, object]:
    if type(proposal) is V1OperatorCandidate:
        return dict(candidate_resource_gain=proposal.resource_gain,
                    candidate_resource_units='removed_route_count',
                    operator_candidate_definition=proposal.definition_payload(),
                    candidate_paired_bce_oracle_required=True,
                    candidate_direct_D_owns_finite_response_admission=True)
    if type(proposal) is V1LabelNativeControllerCandidateCarrier:
        descriptor = proposal.descriptor
        return {
            "candidate_resource_gain": proposal.resource_gain,
            "candidate_resource_units": "removed_route_count",
            "candidate_resource_source_hash": descriptor.source_action_receipt_hash,
            "candidate_resource_receipt_hash": descriptor.source_action_receipt_hash,
            "candidate_f_native_interval": descriptor.f_native_interval,
            "candidate_q_j_interval": descriptor.q_j_interval,
            "candidate_r_half_interval": descriptor.r_half_interval,
            "candidate_r_one_interval": descriptor.r_one_interval,
            "candidate_d_half_interval": descriptor.d_half_interval,
            "candidate_d_one_interval": descriptor.d_one_interval,
            "candidate_chart_half_interval": descriptor.chart_half_interval,
            "candidate_chart_one_interval": descriptor.chart_one_interval,
            "candidate_tau_l": descriptor.certificate.tau_l,
            "candidate_direct_D_status": descriptor.certificate.direct_D_status.value,
            "candidate_chart_transport_status": (
                descriptor.certificate.chart_transport_status.value
            ),
            "candidate_admission_basis": descriptor.certificate.admission_basis,
            "candidate_six_role_certification_receipt_hash": (
                descriptor.six_role_certification_receipt_hash
            ),
            "candidate_six_role_decision_hash": descriptor.six_role_decision_hash,
            "candidate_six_role_decision_design_hash": (
                descriptor.six_role_decision_design_hash
            ),
            "candidate_six_role_C_design_hash": descriptor.six_role_C_design_hash,
            "candidate_six_role_C_progress_receipt_hash": (
                descriptor.six_role_C_progress_receipt_hash
            ),
            "candidate_selected_coordinate_hash": descriptor.coordinate_hash,
            "candidate_representative_branch": descriptor.branch,
            "candidate_representative_endpoint_index": (
                descriptor.coordinate.endpoint_index
            ),
            "candidate_old_chart_validity_claim": (
                descriptor.certificate.admission_basis == "chart_certified"
            ),
            "candidate_direct_D_owns_finite_response_admission": True,
            "candidate_QR_transport_is_label_not_hidden_C_veto": True,
            "candidate_exact_intervention_damage_evaluated": False,
            "candidate_paired_bce_oracle_required": True,
            "candidate_controller_freeze_authority_from_carrier": False,
            "candidate_controller_freeze_authority_from_controller_checks": True,
            "additive_surrogate_used": False,
        }
    if type(proposal) is V1SignedWorkCandidateProposal:
        state = proposal.numeric_state
        return {
            "candidate_resource_gain": proposal.resource_gain,
            "candidate_resource_units": state.resource_units,
            "candidate_resource_source_hash": state.resource_source_hash,
            "candidate_resource_receipt_hash": state.resource_receipt_hash,
            "candidate_f_native_lower": state.f_native_interval[0],
            "candidate_f_native_upper": state.f_native_interval[1],
            "candidate_f_native_interval": state.f_native_interval,
            "candidate_q_j_lower": state.q_j_interval[0],
            "candidate_q_j_upper": state.q_j_interval[1],
            "candidate_q_j_interval": state.q_j_interval,
            "candidate_r_half_interval": state.r_half_interval,
            "candidate_r_one_interval": state.r_one_interval,
            "candidate_d_half_interval": state.d_half_interval,
            "candidate_d_one_interval": state.d_one_interval,
            "candidate_chart_half_interval": state.chart_half_interval,
            "candidate_chart_one_interval": state.chart_one_interval,
            "candidate_tau_l": state.tau_l,
            "candidate_direct_D_status": state.direct_D_status,
            "candidate_chart_transport_status": state.chart_transport_status,
            "candidate_admission_basis": state.admission_basis,
            "candidate_six_role_interval_hash": state.six_role_interval_hash,
            "candidate_six_role_authorization_hash": (
                state.six_role_authorization_hash
            ),
            "candidate_six_role_certification_receipt_hash": (
                state.six_role_certification_receipt_hash
            ),
            "candidate_six_role_decision_hash": state.six_role_decision_hash,
            "candidate_six_role_decision_design_hash": (
                state.six_role_decision_design_hash
            ),
            "candidate_six_role_C_design_hash": state.six_role_C_design_hash,
            "candidate_six_role_C_progress_receipt_hash": (
                state.six_role_C_progress_receipt_hash
            ),
            "candidate_selected_coordinate_hash": state.selected_coordinate_hash,
            "candidate_representative_branch": state.representative_branch,
            "candidate_representative_endpoint_index": (
                state.representative_endpoint_index
            ),
            "candidate_old_chart_validity_claim": state.old_chart_validity_claim,
            "candidate_direct_D_owns_finite_response_admission": True,
            "candidate_QR_transport_is_label_not_hidden_C_veto": True,
            "candidate_exact_intervention_damage_evaluated": False,
            "candidate_paired_bce_oracle_required": True,
            "additive_surrogate_used": False,
        }
    if type(proposal) is V1FQRCandidateProposal:
        state = proposal.numeric_state
        return {
            "candidate_resource_gain": proposal.resource_gain,
            "candidate_resource_units": state.resource_units,
            "candidate_resource_source_hash": state.resource_source_hash,
            "candidate_resource_receipt_hash": state.resource_receipt_hash,
            "candidate_f_lower": state.f_lower,
            "candidate_f_upper": state.f_upper,
            "candidate_f_interval": (state.f_lower, state.f_upper),
            "candidate_q_lower": state.q_lower,
            "candidate_q_upper": state.q_upper,
            "candidate_q_interval": (state.q_lower, state.q_upper),
            "candidate_r_half_lower": state.r_half_lower,
            "candidate_r_half_upper": state.r_half_upper,
            "candidate_r_half_interval": (state.r_half_lower, state.r_half_upper),
            "candidate_r_one_lower": state.r_one_lower,
            "candidate_r_one_upper": state.r_one_upper,
            "candidate_r_one_interval": (state.r_one_lower, state.r_one_upper),
            "candidate_half_response_lower": state.half_response_lower,
            "candidate_half_response_upper": state.half_response_upper,
            "candidate_half_response_interval": (
                state.half_response_lower,
                state.half_response_upper,
            ),
            "candidate_one_response_lower": state.one_response_lower,
            "candidate_one_response_upper": state.one_response_upper,
            "candidate_one_response_interval": (
                state.one_response_lower,
                state.one_response_upper,
            ),
            "candidate_fqr_interval_hash": state.fqr_interval_hash,
            "candidate_fqr_branch_family_authorization_hash": (
                state.branch_family_authorization_hash
            ),
            "candidate_fqr_branch_family_entry_hash": (
                state.branch_family_entry_hash
            ),
            "candidate_fqr_prefrozen_family_hash": state.prefrozen_family_hash,
            "candidate_fqr_joint_family_registration_hash": (
                state.joint_fqr_family_registration_hash
            ),
            "candidate_fqr_source_branch_evidence_hashes": (
                state.source_branch_evidence_hashes
            ),
            "candidate_fqr_representative_branch": state.representative_branch,
            "candidate_fqr_representative_endpoint_index": (
                state.representative_endpoint_index
            ),
            "candidate_fqr_source_alias_hashes": tuple(
                value.alias_hash for value in state.source_aliases
            ),
            "candidate_exact_intervention_damage_evaluated": False,
            "candidate_paired_bce_oracle_required": True,
            "additive_surrogate_used": False,
        }
    if type(proposal) is V1FQCandidateProposal:
        return {
            "candidate_resource_gain": proposal.resource_gain,
            "candidate_f_lower": proposal.f_lower,
            "candidate_f_upper": proposal.f_upper,
            "candidate_f_interval": (proposal.f_lower, proposal.f_upper),
            "candidate_q_lower": proposal.q_lower,
            "candidate_q_upper": proposal.q_upper,
            "candidate_q_interval": (proposal.q_lower, proposal.q_upper),
            "candidate_sqrt_q_lower": proposal.sqrt_q_lower,
            "candidate_sqrt_q_upper": proposal.sqrt_q_upper,
            "candidate_sqrt_q_interval": (
                proposal.sqrt_q_lower,
                proposal.sqrt_q_upper,
            ),
            "candidate_f_units": proposal.f_units,
            "candidate_q_units": proposal.q_units,
            "candidate_sqrt_q_units": proposal.sqrt_q_units,
            "candidate_f_upper_budget": proposal.f_upper_budget,
            "candidate_q_upper_budget": proposal.q_upper_budget,
            "candidate_sqrt_q_upper_budget": math.sqrt(proposal.q_upper_budget),
            "candidate_f_upper_budget_ratio": proposal.f_upper_budget_ratio,
            "candidate_q_upper_budget_ratio": proposal.q_upper_budget_ratio,
            "candidate_sqrt_q_upper_budget_ratio": (
                proposal.sqrt_q_upper_budget_ratio
            ),
            "candidate_fq_interval_hash": proposal.fq_interval_hash,
            "candidate_fq_budget_hash": proposal.fq_budget_hash,
            "candidate_decisive_fq_commit_hash": proposal.decisive_fq_commit_hash,
            "candidate_fq_queue_authorization_hash": (
                proposal.fq_queue_authorization_hash
            ),
            "candidate_sampled_diagnostic_lineage_hash": (
                proposal.sampled_diagnostic_lineage_hash
            ),
            "candidate_exact_intervention_damage_evaluated": False,
            "candidate_exact_registered_surrogate_evaluation": False,
            "additive_surrogate_used": False,
        }
    return {
        "candidate_resource_gain": proposal.resource_gain,
        "candidate_corrected_linear_damage": proposal.corrected_linear_damage,
        "candidate_fd_quadratic": proposal.fd_quadratic,
        "candidate_residual_diagonal_damage": proposal.residual_diagonal_damage,
        "candidate_predicted_surrogate_damage": proposal.predicted_surrogate_damage,
        "candidate_proposal_damage_constraint_value": (
            proposal.proposal_damage_constraint_value
        ),
        "candidate_fd_spectral_upper": proposal.fd_spectral_upper,
        "candidate_diagonal_hard_bound_used": proposal.diagonal_hard_bound_used,
    }


def _calibration_horizon_skip_from_attempt(
    attempt: V1CandidateAttemptReceipt,
) -> V1PairedBCECalibrationHorizonSkipDecision:
    """Rebuild and bind the full typed skip payload stored in one v2 attempt."""

    if attempt.action != "skip_candidate_calibration_refusal":
        raise TypeError("attempt is not a calibration-horizon skip")
    raw = attempt.metadata.get("calibration_horizon_skip_decision")
    if not isinstance(raw, Mapping):
        raise ValueError("calibration-horizon attempt lacks its decision payload")
    decision = V1PairedBCECalibrationHorizonSkipDecision.from_payload(raw)
    decision.assert_integrity(attempt.decision_receipt_hash)
    refusal = attempt.metadata.get("causal_horizon_refusal_receipt")
    if not isinstance(refusal, Mapping) or dict(refusal) != dict(
        decision.causal_horizon_refusal_receipt
    ):
        raise ValueError("calibration-horizon attempt changed its refusal payload")
    exact = {
        "campaign_contract_hash": attempt.campaign_contract_hash,
        "controller_run_id": attempt.controller_run_id,
        "acquisition_receipt_hash": attempt.acquisition_receipt_hash,
        "acquisition_generation_index": attempt.acquisition_generation_index,
        "accepted_step_index": attempt.accepted_step_index,
        "queue_index": attempt.queue_index,
        "candidate_id": attempt.candidate_id,
        "parent_mask_hash": attempt.parent_mask_hash,
        "candidate_mask_hash": attempt.candidate_mask_hash,
        "reason": attempt.reason,
    }
    if (
        any(getattr(decision, name) != value for name, value in exact.items())
        or decision.candidate_attempt_index != attempt.candidate_attempt_index
        or decision.refusal_ledger_hash != attempt.ledger_hash
        or decision.decision_hash != attempt.decision_receipt_hash
        or attempt.step_freeze_hash is not None
        or attempt.draw_ids
        or attempt.source_context_ids
        or attempt.mask_mutated
        or attempt.resulting_current_mask_hash != attempt.parent_mask_hash
        or attempt.metadata.get("causal_horizon_refusal_receipt_hash")
        != decision.refusal_receipt_hash
        or attempt.metadata.get("calibration_horizon_skip_decision_hash")
        != decision.decision_hash
        or attempt.metadata.get("candidate_threshold_hash")
        != decision.candidate_threshold_hash
        or attempt.metadata.get("maximum_mean_damage")
        != decision.maximum_mean_damage
        or attempt.metadata.get("candidate_attempt_alpha_axis_hash")
        != decision.candidate_attempt_alpha_axis_hash
        or attempt.metadata.get("candidate_attempt_alpha_coordinate_hash")
        != decision.candidate_attempt_alpha_coordinate_hash
        or attempt.metadata.get("candidate_attempt_alpha_index")
        != decision.candidate_attempt_index
        or attempt.metadata.get("maximum_candidate_attempts")
        != decision.maximum_candidate_attempts
        or attempt.metadata.get("oracle_familywise_alpha")
        != decision.oracle_familywise_alpha
        or attempt.metadata.get("alpha_committed_before_attempt")
        != decision.alpha_committed_before_attempt
        or attempt.metadata.get("attempt_alpha_allocation")
        != decision.attempt_alpha_allocation
        or attempt.metadata.get("master_decision_alpha_registration_hash")
        != decision.master_decision_alpha_registration_hash
        or attempt.metadata.get("oracle_protocol")
        != "paired_bce_calibration_horizon_skip"
        or attempt.metadata.get("attempt_alpha_retired") is not True
        or attempt.metadata.get("unused_alpha_recycled") is not False
        or attempt.metadata.get("sampling_law_opened") is not False
        or attempt.metadata.get("oracle_registration_opened") is not False
        or attempt.metadata.get("oracle_stream_opened") is not False
        or attempt.metadata.get("oracle_rows_observed") is not False
    ):
        raise ValueError("calibration-horizon attempt does not replay its decision")
    return decision


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class V1FQRDDecisionStabilityCertificate:
    """Hash-replayable authority that closes the proposal-only D firewall.

    The certificate is deliberately narrower than mask-commit or Pareto
    authority.  It proves that the *registered next proposal family* is
    unchanged over the declared D-sampling, FD-truncation, and KKT numerical
    uncertainty sets.  Independent C and, for a nonempty queue, the paired BCE
    oracle must still evaluate the frozen family before trajectory saturation
    can be claimed.
    """

    proposal_handoff_receipt_hash: str
    policy_hash: str
    registered_d_look_schedule_hash: str
    registered_d_prefix_identity_hash: str
    registered_d_transaction_receipt_hash: str
    complete_train_population_hash: str
    population_context_ids_hash: str
    sealed_population_order_hashes: tuple[str, str]
    current_mask_hash: str
    route_axis_hash: str
    proposal_generation: int
    registered_look_index: int
    observed_stratum_counts: tuple[int, int]
    registered_d_census_stratum_counts: tuple[int, int]
    d_sampling_uncertainty_mode: str
    d_sampling_uncertainty_receipt_hash: str
    fd_rank_policy_hash: str
    fd_uncertainty_set_hash: str
    admitted_rank_ladder: tuple[int, ...]
    executed_rank_prefix: tuple[int, ...]
    rank_replay_record_hashes: tuple[str, ...]
    identical_d_prefix_replay_ledger_hash: str
    deterministic_fd_residual_bound_receipt_hashes: tuple[str, ...]
    fd_rank_uncertainty_mode: str
    kkt_numerical_uncertainty_set_hash: str
    kkt_decision_invariance_receipt_hash: str
    candidate_ladder_identity_hash: str
    proposal_decision_identity_hash: str
    _issuance_token: InitVar[object]
    registered_D_population_and_order_bound: bool = True
    same_registered_D_prefix_replayed_across_ranks: bool = True
    D_sampling_decision_invariant: bool = True
    fd_rank_truncation_decision_invariant: bool = True
    kkt_numerical_decision_invariant: bool = True
    all_admissible_registered_uncertainty_same_next_decision: bool = True
    downstream_FQR_or_paired_BCE_still_required: bool = True
    terminal_saturation_after_complete_downstream_family_authority: bool = True
    hard_mask_global_optimality_claim: bool = False
    irreversible_mask_commit_authority: bool = False
    terminal_validation_access_authority: bool = False
    pareto_claim_authority: bool = False
    validation_outcomes_observed: bool = False
    refusal_reason: None = None
    schema: str = FQR_D_DECISION_STABILITY_CERTIFICATE_SCHEMA
    _certificate_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _FQR_D_DECISION_STABILITY_ISSUANCE_TOKEN:
            raise PermissionError(
                "D decision stability requires the registered source verifier"
            )
        for name in (
            "proposal_handoff_receipt_hash",
            "policy_hash",
            "registered_d_look_schedule_hash",
            "registered_d_prefix_identity_hash",
            "registered_d_transaction_receipt_hash",
            "complete_train_population_hash",
            "population_context_ids_hash",
            "current_mask_hash",
            "route_axis_hash",
            "d_sampling_uncertainty_receipt_hash",
            "fd_rank_policy_hash",
            "fd_uncertainty_set_hash",
            "identical_d_prefix_replay_ledger_hash",
            "kkt_numerical_uncertainty_set_hash",
            "kkt_decision_invariance_receipt_hash",
            "candidate_ladder_identity_hash",
            "proposal_decision_identity_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        orders = tuple(
            _nonempty(value, "sealed D population-order hash")
            for value in self.sealed_population_order_hashes
        )
        if len(orders) != 2 or len(set(orders)) != 2:
            raise ValueError(
                "D decision stability requires two distinct sealed population orders"
            )
        object.__setattr__(self, "sealed_population_order_hashes", orders)
        generation = _nonnegative_int(
            self.proposal_generation, "D decision-stability generation"
        )
        look_index = _nonnegative_int(
            self.registered_look_index, "D decision-stability look index"
        )
        if look_index == 0:
            raise ValueError("D decision stability requires a positive look index")
        observed = tuple(
            _nonnegative_int(value, "observed D stratum count")
            for value in self.observed_stratum_counts
        )
        census = tuple(
            _nonnegative_int(value, "registered D census stratum count")
            for value in self.registered_d_census_stratum_counts
        )
        if (
            len(observed) != 2
            or len(census) != 2
            or any(value == 0 for value in census)
            or any(value > limit for value, limit in zip(observed, census, strict=True))
        ):
            raise ValueError("D decision-stability stratum counts are invalid")
        sampling_mode = _nonempty(
            self.d_sampling_uncertainty_mode, "D sampling uncertainty mode"
        )
        if sampling_mode not in {
            "exact_full_registered_D_census",
            "rigorous_registered_interval_invariance",
        }:
            raise ValueError("D sampling uncertainty mode is unsupported")
        if sampling_mode == "exact_full_registered_D_census" and observed != census:
            raise ValueError("exact D-census authority requires the complete registered axis")
        ranks = tuple(
            _nonnegative_int(value, "admitted FD rank")
            for value in self.admitted_rank_ladder
        )
        executed = tuple(
            _nonnegative_int(value, "executed FD rank")
            for value in self.executed_rank_prefix
        )
        replay_hashes = tuple(
            _nonempty(value, "FD rank replay-record hash")
            for value in self.rank_replay_record_hashes
        )
        residual_hashes = tuple(
            _nonempty(value, "deterministic FD residual-bound receipt hash")
            for value in self.deterministic_fd_residual_bound_receipt_hashes
        )
        if (
            not ranks
            or any(value == 0 for value in ranks)
            or ranks != tuple(sorted(set(ranks)))
            or not executed
            or executed != ranks[: len(executed)]
            or len(replay_hashes) != len(executed)
            or len(residual_hashes) != 2
            or len(set(replay_hashes)) != len(replay_hashes)
            or len(set(residual_hashes)) != len(residual_hashes)
        ):
            raise ValueError(
                "D decision stability requires one contiguous, receipt-bound FD rank prefix"
            )
        fd_mode = _nonempty(
            self.fd_rank_uncertainty_mode, "FD-rank uncertainty mode"
        )
        if fd_mode not in {
            "typed_adaptive_rank_interval_invariance",
            "exact_full_rank_FD_geometry",
        }:
            raise ValueError("FD-rank uncertainty mode is unsupported")
        required_true = (
            self.registered_D_population_and_order_bound,
            self.same_registered_D_prefix_replayed_across_ranks,
            self.D_sampling_decision_invariant,
            self.fd_rank_truncation_decision_invariant,
            self.kkt_numerical_decision_invariant,
            self.all_admissible_registered_uncertainty_same_next_decision,
            self.downstream_FQR_or_paired_BCE_still_required,
            self.terminal_saturation_after_complete_downstream_family_authority,
        )
        required_false = (
            self.hard_mask_global_optimality_claim,
            self.irreversible_mask_commit_authority,
            self.terminal_validation_access_authority,
            self.pareto_claim_authority,
            self.validation_outcomes_observed,
        )
        if (
            any(type(value) is not bool for value in (*required_true, *required_false))
            or not all(required_true)
            or any(required_false)
            or self.refusal_reason is not None
            or self.schema != FQR_D_DECISION_STABILITY_CERTIFICATE_SCHEMA
        ):
            raise PermissionError(
                "D decision-stability certificate crossed its terminal-saturation boundary"
            )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "registered_look_index", look_index)
        object.__setattr__(self, "observed_stratum_counts", observed)
        object.__setattr__(self, "registered_d_census_stratum_counts", census)
        object.__setattr__(self, "d_sampling_uncertainty_mode", sampling_mode)
        object.__setattr__(self, "admitted_rank_ladder", ranks)
        object.__setattr__(self, "executed_rank_prefix", executed)
        object.__setattr__(self, "rank_replay_record_hashes", replay_hashes)
        object.__setattr__(
            self,
            "deterministic_fd_residual_bound_receipt_hashes",
            residual_hashes,
        )
        object.__setattr__(self, "fd_rank_uncertainty_mode", fd_mode)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "proposal_handoff_receipt_hash",
                "policy_hash",
                "registered_d_look_schedule_hash",
                "registered_d_prefix_identity_hash",
                "registered_d_transaction_receipt_hash",
                "complete_train_population_hash",
                "population_context_ids_hash",
                "sealed_population_order_hashes",
                "current_mask_hash",
                "route_axis_hash",
                "proposal_generation",
                "registered_look_index",
                "observed_stratum_counts",
                "registered_d_census_stratum_counts",
                "d_sampling_uncertainty_mode",
                "d_sampling_uncertainty_receipt_hash",
                "fd_rank_policy_hash",
                "fd_uncertainty_set_hash",
                "admitted_rank_ladder",
                "executed_rank_prefix",
                "rank_replay_record_hashes",
                "identical_d_prefix_replay_ledger_hash",
                "deterministic_fd_residual_bound_receipt_hashes",
                "fd_rank_uncertainty_mode",
                "kkt_numerical_uncertainty_set_hash",
                "kkt_decision_invariance_receipt_hash",
                "candidate_ladder_identity_hash",
                "proposal_decision_identity_hash",
                "registered_D_population_and_order_bound",
                "same_registered_D_prefix_replayed_across_ranks",
                "D_sampling_decision_invariant",
                "fd_rank_truncation_decision_invariant",
                "kkt_numerical_decision_invariant",
                "all_admissible_registered_uncertainty_same_next_decision",
                "downstream_FQR_or_paired_BCE_still_required",
                "terminal_saturation_after_complete_downstream_family_authority",
                "hard_mask_global_optimality_claim",
                "irreversible_mask_commit_authority",
                "terminal_validation_access_authority",
                "pareto_claim_authority",
                "validation_outcomes_observed",
                "refusal_reason",
            )
        }

    @property
    def certificate_hash(self) -> str:
        cached = self._certificate_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(), prefix=FQR_D_DECISION_STABILITY_CERTIFICATE_PREFIX
        )
        if self._certificate_hash is not None and self._certificate_hash != observed:
            raise ValueError("D decision-stability certificate was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("D decision-stability certificate hash does not match")
        object.__setattr__(self, "_certificate_hash", observed)
        return observed

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "certificate_hash": self.certificate_hash}

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "V1FQRDDecisionStabilityCertificate":
        if not isinstance(payload, Mapping):
            raise TypeError("D decision-stability payload must be a mapping")
        plain = dict(payload)
        expected_hash = _nonempty(
            plain.pop("certificate_hash", None),
            "D decision-stability certificate hash",
        )
        expected_fields = set(cls.__dataclass_fields__) - {
            "_issuance_token",
            "_certificate_hash",
        }
        if set(plain) != expected_fields:
            raise ValueError("D decision-stability certificate fields are not exact")
        for name in (
            "sealed_population_order_hashes",
            "observed_stratum_counts",
            "registered_d_census_stratum_counts",
            "admitted_rank_ladder",
            "executed_rank_prefix",
            "rank_replay_record_hashes",
            "deterministic_fd_residual_bound_receipt_hashes",
        ):
            plain[name] = tuple(plain[name])
        certificate = cls(
            **plain,
            _issuance_token=_FQR_D_DECISION_STABILITY_ISSUANCE_TOKEN,
        )
        certificate.assert_integrity(expected_hash)
        return certificate


def issue_v1_fqr_d_decision_stability_certificate(
    **coordinates: object,
) -> V1FQRDDecisionStabilityCertificate:
    """Issue a certificate after a source verifier has closed all coordinates."""

    return V1FQRDDecisionStabilityCertificate(
        **coordinates,
        _issuance_token=_FQR_D_DECISION_STABILITY_ISSUANCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1FQRDDecisionStabilityRefusalReceipt:
    """Typed no-authority result naming the unresolved D decision source."""

    proposal_handoff_receipt_hash: str
    policy_hash: str
    registered_d_look_schedule_hash: str
    registered_d_prefix_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    registered_look_index: int
    observed_stratum_counts: tuple[int, int]
    registered_d_census_stratum_counts: tuple[int, int]
    uncertainty_status_by_source: tuple[tuple[str, str], ...]
    refusal_reason: str
    _issuance_token: InitVar[object]
    certification_status: str = "refused_unresolved_decision_uncertainty"
    scientific_terminal_endpoint_authority: bool = False
    pareto_saturation_authority: bool = False
    terminal_validation_access_authority: bool = False
    pareto_claim_authority: bool = False
    validation_outcomes_observed: bool = False
    schema: str = FQR_D_DECISION_STABILITY_REFUSAL_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _FQR_D_DECISION_STABILITY_REFUSAL_ISSUANCE_TOKEN:
            raise PermissionError(
                "D decision-stability refusal requires the registered source verifier"
            )
        for name in (
            "proposal_handoff_receipt_hash",
            "policy_hash",
            "registered_d_look_schedule_hash",
            "registered_d_prefix_identity_hash",
            "current_mask_hash",
            "refusal_reason",
            "certification_status",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        generation = _nonnegative_int(
            self.proposal_generation, "D decision-stability refusal generation"
        )
        look_index = _nonnegative_int(
            self.registered_look_index, "D decision-stability refusal look index"
        )
        if look_index == 0:
            raise ValueError("D decision-stability refusal requires a positive look")
        observed = tuple(
            _nonnegative_int(value, "observed D stratum count")
            for value in self.observed_stratum_counts
        )
        census = tuple(
            _nonnegative_int(value, "registered D census stratum count")
            for value in self.registered_d_census_stratum_counts
        )
        status_rows = tuple(
            (
                _nonempty(source, "D decision uncertainty source"),
                _nonempty(status, "D decision uncertainty status"),
            )
            for source, status in self.uncertainty_status_by_source
        )
        expected_sources = ("D_sampling", "FD_rank_truncation", "KKT_numerical")
        flags = (
            self.scientific_terminal_endpoint_authority,
            self.pareto_saturation_authority,
            self.terminal_validation_access_authority,
            self.pareto_claim_authority,
            self.validation_outcomes_observed,
        )
        if (
            len(observed) != 2
            or len(census) != 2
            or tuple(source for source, _status in status_rows) != expected_sources
            or any(type(value) is not bool for value in flags)
            or any(flags)
            or self.certification_status
            != "refused_unresolved_decision_uncertainty"
            or self.schema != FQR_D_DECISION_STABILITY_REFUSAL_SCHEMA
        ):
            raise PermissionError("D decision-stability refusal gained authority")
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "registered_look_index", look_index)
        object.__setattr__(self, "observed_stratum_counts", observed)
        object.__setattr__(self, "registered_d_census_stratum_counts", census)
        object.__setattr__(self, "uncertainty_status_by_source", status_rows)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "proposal_handoff_receipt_hash",
                "policy_hash",
                "registered_d_look_schedule_hash",
                "registered_d_prefix_identity_hash",
                "current_mask_hash",
                "proposal_generation",
                "registered_look_index",
                "observed_stratum_counts",
                "registered_d_census_stratum_counts",
                "uncertainty_status_by_source",
                "refusal_reason",
                "certification_status",
                "scientific_terminal_endpoint_authority",
                "pareto_saturation_authority",
                "terminal_validation_access_authority",
                "pareto_claim_authority",
                "validation_outcomes_observed",
            )
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(), prefix=FQR_D_DECISION_STABILITY_REFUSAL_PREFIX
        )
        if self._receipt_hash is not None and self._receipt_hash != observed:
            raise ValueError("D decision-stability refusal was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("D decision-stability refusal hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "V1FQRDDecisionStabilityRefusalReceipt":
        if not isinstance(payload, Mapping):
            raise TypeError("D decision-stability refusal payload must be a mapping")
        plain = dict(payload)
        expected_hash = _nonempty(
            plain.pop("receipt_hash", None), "D decision-stability refusal hash"
        )
        expected_fields = set(cls.__dataclass_fields__) - {
            "_issuance_token",
            "_receipt_hash",
        }
        if set(plain) != expected_fields:
            raise ValueError("D decision-stability refusal fields are not exact")
        for name in (
            "observed_stratum_counts",
            "registered_d_census_stratum_counts",
            "uncertainty_status_by_source",
        ):
            plain[name] = tuple(plain[name])
        plain["uncertainty_status_by_source"] = tuple(
            tuple(value) for value in plain["uncertainty_status_by_source"]
        )
        receipt = cls(
            **plain,
            _issuance_token=_FQR_D_DECISION_STABILITY_REFUSAL_ISSUANCE_TOKEN,
        )
        receipt.assert_integrity(expected_hash)
        return receipt


def issue_v1_fqr_d_decision_stability_refusal(
    **coordinates: object,
) -> V1FQRDDecisionStabilityRefusalReceipt:
    return V1FQRDDecisionStabilityRefusalReceipt(
        **coordinates,
        _issuance_token=_FQR_D_DECISION_STABILITY_REFUSAL_ISSUANCE_TOKEN,
    )


def _freeze(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{label} keys must be nonempty strings")
            frozen[key] = _freeze(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, f"{label}[{index}]") for index, item in enumerate(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a nonfinite float")
        return float(value)
    if isinstance(value, (str, bytes, int, bool)) or value is None:
        return value
    raise TypeError(f"{label} contains unsupported value {type(value)!r}")


def _probabilities(values: Sequence[float], label: str, size: int) -> tuple[float, ...]:
    array = np.asarray(tuple(values), dtype=np.float64).reshape(-1)
    if array.size != size or not np.isfinite(array).all() or bool(np.any(array < 0.0)):
        raise ValueError(f"{label} must be a finite nonnegative population law")
    tolerance = 64.0 * np.finfo(np.float64).eps
    if not math.isclose(
        float(np.sum(array, dtype=np.float64)),
        1.0,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(f"{label} must sum to one")
    return tuple(float(value) for value in array)


@dataclass(frozen=True, slots=True)
class V1StepSamplingLaw:
    """Complete predictable target/sampling law for one frozen candidate."""

    campaign_contract_hash: str
    population_context_ids: tuple[str, ...]
    target_population_probabilities: tuple[float, ...]
    sampling_population_probabilities: tuple[float, ...]
    sampling_masses_u64: tuple[int, ...]
    declared_target_population_hash: str
    sampling_law_hash: str
    support_construction_hash: str
    source_receipt_hash: str
    maximum_importance_weight: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _SAMPLING_LAW_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "declared_target_population_hash",
            "sampling_law_hash",
            "support_construction_hash",
            "source_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        population = tuple(_nonempty(value, "population_context_id") for value in self.population_context_ids)
        if not population or len(set(population)) != len(population):
            raise ValueError("step sampling population must be nonempty and unique")
        target = _probabilities(
            self.target_population_probabilities,
            "target_population_probabilities",
            len(population),
        )
        sampling = _probabilities(
            self.sampling_population_probabilities,
            "sampling_population_probabilities",
            len(population),
        )
        masses = tuple(
            _nonnegative_int(value, "sampling_masses_u64")
            for value in self.sampling_masses_u64
        )
        exact_sampling = tuple(
            float(value / STAGED_ALPHA_SAMPLING_MASS_TOTAL) for value in masses
        )
        if (
            len(masses) != len(population)
            or any(value <= 0 for value in masses)
            or sum(masses) != STAGED_ALPHA_SAMPLING_MASS_TOTAL
            or sampling != exact_sampling
        ):
            raise ValueError(
                "step sampling support probabilities must be induced by their exact uint64 masses"
            )
        target_array = np.asarray(target, dtype=np.float64)
        sampling_array = np.asarray(sampling, dtype=np.float64)
        if bool(np.any((target_array > 0.0) & (sampling_array <= 0.0))):
            raise ValueError("step sampling law lacks support wherever target mass is positive")
        maximum = float(self.maximum_importance_weight)
        if not math.isfinite(maximum) or maximum < 1.0:
            raise ValueError("maximum_importance_weight must be finite and at least one")
        population_weights = np.divide(
            target_array,
            sampling_array,
            out=np.zeros_like(target_array),
            where=sampling_array > 0.0,
        )
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, maximum)
        if bool(np.any(population_weights > maximum + tolerance)):
            raise ValueError("complete-population importance ratio exceeds W_max")
        object.__setattr__(self, "population_context_ids", population)
        object.__setattr__(self, "target_population_probabilities", target)
        object.__setattr__(self, "sampling_population_probabilities", sampling)
        object.__setattr__(self, "sampling_masses_u64", masses)
        object.__setattr__(self, "maximum_importance_weight", maximum)
        object.__setattr__(self, "metadata", _freeze(self.metadata, "step sampling metadata"))
        if self.schema != _SAMPLING_LAW_SCHEMA:
            raise ValueError("unsupported modern-v1 step sampling-law schema")

    @property
    def importance_weight_hash(self) -> str:
        return content_hash(
            {
                "population_context_ids": self.population_context_ids,
                "target_population_probabilities": np.asarray(
                    self.target_population_probabilities,
                    dtype=np.float64,
                ),
                "sampling_population_probabilities": np.asarray(
                    self.sampling_population_probabilities,
                    dtype=np.float64,
                ),
            },
            prefix="partition-free-v1-importance-probabilities-",
        )

    @property
    def sampling_state_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-step-sampling-law-")


@dataclass(frozen=True, slots=True)
class V1CandidateDamageThreshold:
    """Candidate-specific total dense-baseline damage target frozen before any draw.

    ``parent_mask_hash`` is the proposal/commit parent owned by the current
    relinearized queue.  The statistical loss referent is recorded separately
    in metadata as ``loss_referent_mask_hash`` so the exact oracle can measure
    the complete candidate mask against the dense baseline while commit
    authority still consumes only the current-parent -> candidate transition.
    """

    campaign_contract_hash: str
    acquisition_receipt_hash: str
    candidate_id: str
    parent_mask_hash: str
    candidate_mask_hash: str
    maximum_mean_damage: float
    damage_target_derivation_hash: str
    threshold_source_receipt_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _THRESHOLD_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "acquisition_receipt_hash",
            "candidate_id",
            "parent_mask_hash",
            "candidate_mask_hash",
            "damage_target_derivation_hash",
            "threshold_source_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "maximum_mean_damage",
            _finite_nonnegative(self.maximum_mean_damage, "maximum_mean_damage"),
        )
        if self.parent_mask_hash == self.candidate_mask_hash:
            raise ValueError("candidate threshold must bind a changed child mask")
        object.__setattr__(self, "metadata", _freeze(self.metadata, "candidate threshold metadata"))
        if self.schema != _THRESHOLD_SCHEMA:
            raise ValueError("unsupported candidate damage-threshold schema")

    @property
    def threshold_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-candidate-damage-threshold-")


@dataclass(frozen=True, slots=True)
class V1FrozenAffineCalibration:
    """Exact validation-independent affine state and its successful fit receipt.

    This type is backend-neutral on purpose.  It records the complete numeric
    state needed to reconstruct a ``FixedAffineBCECalibration`` without
    importing or executing a fitter inside the controller.
    """

    mask_hash: str
    query_id: str
    query_hash: str
    target_class_id: int
    scale: float
    offset: float
    calibration_hash: str
    calibration_state_id: str
    calibration_source_hash: str
    calibration_context_axis_hash: str
    source_label_map_hash: str
    model_state_hash: str
    calibration_schema: str
    fit_receipt: Mapping[str, Any]
    fit_receipt_hash: str
    validation_independent: bool = True
    terminal_rows_used_for_fit: bool = False
    terminal_outcomes_opened: bool = False
    schema: str = _FROZEN_AFFINE_CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "mask_hash",
            "query_id",
            "query_hash",
            "calibration_hash",
            "calibration_state_id",
            "calibration_source_hash",
            "calibration_context_axis_hash",
            "source_label_map_hash",
            "model_state_hash",
            "calibration_schema",
            "fit_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        target = _nonnegative_int(self.target_class_id, "target_class_id")
        scale = float(self.scale)
        offset = float(self.offset)
        if not math.isfinite(scale) or scale <= 0.0 or not math.isfinite(offset):
            raise ValueError("frozen affine calibration requires finite positive scale/offset")
        receipt = _freeze(self.fit_receipt, "calibration fit receipt")
        if receipt.get("fit_succeeded") is not True:
            raise ValueError("frozen affine calibration requires a successful fit receipt")
        if (
            receipt.get("mask_hash") != self.mask_hash
            or receipt.get("validation_independent") is not True
            or receipt.get("terminal_rows_used_for_fit") is not False
            or receipt.get("terminal_outcomes_opened") is not False
        ):
            raise ValueError("frozen affine calibration fit receipt changed its causal scope")
        expected_receipt_hash = content_hash(
            receipt,
            prefix="partition-free-v1-frozen-calibration-fit-receipt-",
        )
        if self.fit_receipt_hash != expected_receipt_hash:
            raise ValueError("frozen affine calibration fit receipt is not reproducible")
        if (
            self.validation_independent is not True
            or self.terminal_rows_used_for_fit is not False
            or self.terminal_outcomes_opened is not False
            or self.schema != _FROZEN_AFFINE_CALIBRATION_SCHEMA
        ):
            raise ValueError("frozen affine calibration crossed the validation firewall")
        object.__setattr__(self, "target_class_id", target)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "fit_receipt", receipt)

    @property
    def frozen_calibration_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-frozen-affine-calibration-")

    def to_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> V1FrozenAffineCalibration:
        if not isinstance(payload, Mapping):
            raise TypeError("frozen affine calibration payload must be a mapping")
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("frozen affine calibration payload fields differ")
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class V1CandidateCalibrationFreeze:
    """Candidate-specific affine calibration state frozen before oracle draws."""

    campaign_contract_hash: str
    acquisition_receipt_hash: str
    queue_index: int
    candidate_id: str
    proposal_parent_mask_hash: str
    dense_referent_mask_hash: str
    candidate_mask_hash: str
    calibration_policy_id: str
    dense_calibration: V1FrozenAffineCalibration
    candidate_calibration: V1FrozenAffineCalibration
    calibration_recipe: Mapping[str, Any]
    calibration_recipe_hash: str
    calibration_context_ids: tuple[str, ...]
    calibration_context_ids_hash: str
    calibration_context_selection_hash: str
    calibration_context_selection_policy: Mapping[str, Any]
    calibration_context_selection_policy_hash: str
    source_receipt_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    validation_independent: bool = True
    terminal_rows_used_for_fit: bool = False
    terminal_outcomes_opened: bool = False
    schema: str = _CANDIDATE_CALIBRATION_FREEZE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "acquisition_receipt_hash",
            "candidate_id",
            "proposal_parent_mask_hash",
            "dense_referent_mask_hash",
            "candidate_mask_hash",
            "calibration_policy_id",
            "calibration_recipe_hash",
            "calibration_context_ids_hash",
            "calibration_context_selection_hash",
            "calibration_context_selection_policy_hash",
            "source_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(self, "queue_index", _nonnegative_int(self.queue_index, "queue_index"))
        if type(self.dense_calibration) is not V1FrozenAffineCalibration or type(
            self.candidate_calibration
        ) is not V1FrozenAffineCalibration:
            raise TypeError("candidate calibration freeze requires exact affine states")
        if (
            self.dense_referent_mask_hash == self.candidate_mask_hash
            or self.dense_calibration.mask_hash != self.dense_referent_mask_hash
            or self.candidate_calibration.mask_hash != self.candidate_mask_hash
        ):
            raise ValueError("candidate calibration freeze names mismatched masks")
        dense = self.dense_calibration
        candidate = self.candidate_calibration
        for name in (
            "query_id",
            "query_hash",
            "target_class_id",
            "calibration_context_axis_hash",
            "source_label_map_hash",
            "model_state_hash",
            "calibration_schema",
        ):
            if getattr(dense, name) != getattr(candidate, name):
                raise ValueError(f"dense/candidate calibration differs on {name}")
        recipe = _freeze(self.calibration_recipe, "calibration recipe")
        selection_policy = _freeze(
            self.calibration_context_selection_policy,
            "calibration context selection policy",
        )
        contexts = tuple(
            _nonempty(value, "calibration_context_id")
            for value in self.calibration_context_ids
        )
        if not contexts or len(set(contexts)) != len(contexts):
            raise ValueError("calibration context IDs must be nonempty and unique")
        if self.calibration_recipe_hash != content_hash(
            recipe,
            prefix="partition-free-v1-candidate-calibration-recipe-",
        ):
            raise ValueError("candidate calibration recipe hash is not reproducible")
        if self.calibration_context_ids_hash != content_hash(
            contexts,
            prefix="partition-free-v1-candidate-calibration-contexts-",
        ):
            raise ValueError("candidate calibration context hash is not reproducible")
        if self.calibration_context_selection_policy_hash != content_hash(
            selection_policy,
            prefix="partition-free-v1-candidate-calibration-selection-policy-",
        ):
            raise ValueError("candidate calibration selection policy hash is not reproducible")
        if (
            self.validation_independent is not True
            or self.terminal_rows_used_for_fit is not False
            or self.terminal_outcomes_opened is not False
            or self.schema != _CANDIDATE_CALIBRATION_FREEZE_SCHEMA
        ):
            raise ValueError("candidate calibration freeze crossed the validation firewall")
        object.__setattr__(self, "calibration_recipe", recipe)
        object.__setattr__(self, "calibration_context_ids", contexts)
        object.__setattr__(self, "calibration_context_selection_policy", selection_policy)
        object.__setattr__(self, "metadata", _freeze(self.metadata, "candidate calibration metadata"))

    @property
    def freeze_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-candidate-calibration-freeze-")

    def to_payload(self) -> Mapping[str, Any]:
        payload = {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }
        payload["dense_calibration"] = self.dense_calibration.to_payload()
        payload["candidate_calibration"] = self.candidate_calibration.to_payload()
        return MappingProxyType(payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> V1CandidateCalibrationFreeze:
        if not isinstance(payload, Mapping):
            raise TypeError("candidate calibration freeze payload must be a mapping")
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("candidate calibration freeze payload fields differ")
        values = dict(payload)
        values["dense_calibration"] = V1FrozenAffineCalibration.from_payload(
            values["dense_calibration"]
        )
        values["candidate_calibration"] = V1FrozenAffineCalibration.from_payload(
            values["candidate_calibration"]
        )
        return cls(**values)


def calibration_freeze_from_certificate(
    certificate_freeze: V1CertificateFreeze,
    *,
    required: bool = False,
) -> V1CandidateCalibrationFreeze | None:
    """Recover and verify the complete candidate calibration payload."""

    if type(certificate_freeze) is not V1CertificateFreeze:
        raise TypeError("calibration freeze recovery requires an exact certificate freeze")
    payload = certificate_freeze.metadata.get(_CALIBRATION_FREEZE_METADATA_KEY)
    declared_hash = certificate_freeze.metadata.get(
        _CALIBRATION_FREEZE_HASH_METADATA_KEY
    )
    if payload is None and declared_hash is None:
        if required:
            raise ValueError("candidate certificate lacks its required calibration freeze")
        return None
    if payload is None or declared_hash is None:
        raise ValueError("candidate calibration payload/hash presence disagrees")
    freeze = V1CandidateCalibrationFreeze.from_payload(payload)
    if declared_hash != freeze.freeze_hash:
        raise ValueError("candidate calibration freeze hash is not reproducible")
    if (
        freeze.candidate_mask_hash != certificate_freeze.candidate_mask_hash
        or freeze.dense_referent_mask_hash != certificate_freeze.parent_mask_hash
    ):
            raise ValueError("candidate calibration freeze differs from its certificate masks")
    return freeze


@dataclass(frozen=True, slots=True)
class V1AcceptedOracleTrajectoryEntry:
    """One accepted irreversible step bound to its real oracle evidence kind."""

    protocol: OracleAcceptanceProtocol
    accepted_step_index: int
    candidate_attempt_index: int
    acquisition_receipt_hash: str
    candidate_id: str
    parent_mask_hash: str
    candidate_mask_hash: str
    step_freeze_hash: str
    evidence_ledger_hash: str
    decision_receipt_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _ACCEPTED_ORACLE_TRAJECTORY_ENTRY_SCHEMA
    # Appended after ``schema`` so trusted live readers can still hydrate the
    # exact positional slot layout of pre-v2 checkpoints.  ``__post_init__``
    # supplies the absent slot as ``None`` before admitting that legacy state.
    candidate_calibration_freeze: V1CandidateCalibrationFreeze | None = None

    def __post_init__(self) -> None:
        calibration_freeze = getattr(
            self,
            "candidate_calibration_freeze",
            None,
        )
        if not hasattr(self, "candidate_calibration_freeze"):
            object.__setattr__(self, "candidate_calibration_freeze", None)
        if self.protocol not in {"legacy_staged_alpha", "paired_bce"}:
            raise ValueError("accepted oracle trajectory protocol is unsupported")
        for name in (
            "acquisition_receipt_hash",
            "candidate_id",
            "parent_mask_hash",
            "candidate_mask_hash",
            "step_freeze_hash",
            "evidence_ledger_hash",
            "decision_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "accepted_step_index",
            _nonnegative_int(self.accepted_step_index, "accepted_step_index"),
        )
        object.__setattr__(
            self,
            "candidate_attempt_index",
            _nonnegative_int(
                self.candidate_attempt_index,
                "candidate_attempt_index",
            ),
        )
        if self.parent_mask_hash == self.candidate_mask_hash:
            raise ValueError("accepted oracle trajectory must change the mask")
        object.__setattr__(
            self,
            "metadata",
            _freeze(self.metadata, "accepted oracle trajectory metadata"),
        )
        if self.schema not in {
            _LEGACY_ACCEPTED_ORACLE_TRAJECTORY_ENTRY_SCHEMA,
            _ACCEPTED_ORACLE_TRAJECTORY_ENTRY_SCHEMA,
        }:
            raise ValueError("unsupported accepted-oracle trajectory entry schema")
        if self.schema == _LEGACY_ACCEPTED_ORACLE_TRAJECTORY_ENTRY_SCHEMA:
            if calibration_freeze is not None:
                raise ValueError(
                    "legacy accepted-oracle entry cannot acquire a calibration freeze"
                )
            return
        if self.protocol != "paired_bce":
            if calibration_freeze is not None:
                raise ValueError(
                    "legacy staged-alpha acceptance cannot acquire paired calibration authority"
                )
            return
        if type(calibration_freeze) is not V1CandidateCalibrationFreeze:
            raise TypeError(
                "paired-BCE accepted entry requires its exact candidate calibration freeze"
            )
        calibration_freeze.__post_init__()
        if (
            calibration_freeze.acquisition_receipt_hash
            != self.acquisition_receipt_hash
            or calibration_freeze.candidate_id != self.candidate_id
            or calibration_freeze.proposal_parent_mask_hash != self.parent_mask_hash
            or calibration_freeze.candidate_mask_hash != self.candidate_mask_hash
        ):
            raise ValueError(
                "accepted paired-BCE calibration freeze names another transition"
            )
        metadata = self.metadata
        if (
            metadata.get("candidate_calibration_freeze_hash")
            != calibration_freeze.freeze_hash
            or metadata.get("dense_affine_calibration_hash")
            != calibration_freeze.dense_calibration.frozen_calibration_hash
            or metadata.get("candidate_affine_calibration_hash")
            != calibration_freeze.candidate_calibration.frozen_calibration_hash
        ):
            raise ValueError(
                "accepted paired-BCE calibration freeze differs from its durable hashes"
            )

    def _identity_payload(self) -> Mapping[str, Any]:
        """Return the exact schema-owned coordinates used by durable hashing."""

        names = (
            "protocol",
            "accepted_step_index",
            "candidate_attempt_index",
            "acquisition_receipt_hash",
            "candidate_id",
            "parent_mask_hash",
            "candidate_mask_hash",
            "step_freeze_hash",
            "evidence_ledger_hash",
            "decision_receipt_hash",
            "metadata",
            "schema",
        )
        payload = {name: getattr(self, name) for name in names}
        if self.schema == _ACCEPTED_ORACLE_TRAJECTORY_ENTRY_SCHEMA:
            payload["candidate_calibration_freeze"] = (
                self.candidate_calibration_freeze
            )
        return MappingProxyType(payload)

    def require_terminal_milestone_calibration_freeze(
        self,
    ) -> V1CandidateCalibrationFreeze:
        """Return exact validation-independent calibration or refuse legacy evidence."""

        self.__post_init__()
        calibration_freeze = self.candidate_calibration_freeze
        if (
            self.schema != _ACCEPTED_ORACLE_TRAJECTORY_ENTRY_SCHEMA
            or self.protocol != "paired_bce"
            or type(calibration_freeze) is not V1CandidateCalibrationFreeze
        ):
            raise PermissionError(
                "legacy accepted-oracle entry has no terminal-milestone calibration authority"
            )
        return calibration_freeze

    @property
    def entry_hash(self) -> str:
        return content_hash(
            self._identity_payload(),
            prefix="partition-free-v1-accepted-oracle-trajectory-entry-",
        )


@dataclass(frozen=True, slots=True)
class V1AcceptedOracleTrajectoryLedger:
    """Accepted-step history across oracle protocols, with no commit authority."""

    entries: tuple[V1AcceptedOracleTrajectoryEntry, ...] = ()
    schema: str = _ACCEPTED_ORACLE_TRAJECTORY_SCHEMA

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        if any(type(item) is not V1AcceptedOracleTrajectoryEntry for item in entries):
            raise TypeError("accepted oracle trajectory contains a nonentry value")
        previous_candidate: str | None = None
        for index, entry in enumerate(entries):
            entry.__post_init__()
            if entry.accepted_step_index != index:
                raise ValueError("accepted oracle trajectory is not consecutive")
            if previous_candidate is not None and entry.parent_mask_hash != previous_candidate:
                raise ValueError("accepted oracle trajectory ancestry is broken")
            previous_candidate = entry.candidate_mask_hash
        object.__setattr__(self, "entries", entries)
        if self.schema != _ACCEPTED_ORACLE_TRAJECTORY_SCHEMA:
            raise ValueError("unsupported accepted-oracle trajectory schema")

    @property
    def trajectory_ledger_hash(self) -> str:
        return content_hash(
            {
                "entries": tuple(entry._identity_payload() for entry in self.entries),
                "schema": self.schema,
            },
            prefix="partition-free-v1-accepted-oracle-trajectory-",
        )

    @property
    def step_decisions(self) -> tuple[V1AcceptedOracleTrajectoryEntry, ...]:
        """Compatibility count view; entries are not staged-alpha decisions."""

        return self.entries

    @property
    def step_ledgers(self) -> tuple[Any, ...]:
        """Compatibility empty view; paired evidence lives in entry metadata."""

        return ()

    @property
    def can_certify_trajectory(self) -> bool:
        return False

    @property
    def can_commit_mask(self) -> bool:
        return False


def _append_accepted_oracle_entry(
    trajectory: V1AcceptedOracleTrajectoryLedger | None,
    *,
    protocol: OracleAcceptanceProtocol,
    accepted_step_index: int,
    candidate_attempt_index: int,
    acquisition_receipt_hash: str,
    candidate_id: str,
    parent_mask_hash: str,
    candidate_mask_hash: str,
    step_freeze_hash: str,
    evidence_ledger_hash: str,
    decision_receipt_hash: str,
    metadata: Mapping[str, Any],
    candidate_calibration_freeze: V1CandidateCalibrationFreeze | None,
) -> V1AcceptedOracleTrajectoryLedger:
    ledger = trajectory or V1AcceptedOracleTrajectoryLedger()
    return V1AcceptedOracleTrajectoryLedger(
        entries=ledger.entries
        + (
            V1AcceptedOracleTrajectoryEntry(
                protocol=protocol,
                accepted_step_index=accepted_step_index,
                candidate_attempt_index=candidate_attempt_index,
                acquisition_receipt_hash=acquisition_receipt_hash,
                candidate_id=candidate_id,
                parent_mask_hash=parent_mask_hash,
                candidate_mask_hash=candidate_mask_hash,
                step_freeze_hash=step_freeze_hash,
                evidence_ledger_hash=evidence_ledger_hash,
                decision_receipt_hash=decision_receipt_hash,
                metadata=metadata,
                candidate_calibration_freeze=candidate_calibration_freeze,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class V1ActiveCandidate:
    """One frozen candidate and its live oracle evidence state."""

    acquisition_receipt_hash: str
    queue_index: int
    proposal: ModernV1Proposal
    threshold: V1CandidateDamageThreshold
    sampling_law: V1StepSamplingLaw
    step_freeze: V1StagedAlphaStepFreeze
    ledger: V1StagedAlphaLedger
    paired_bce_registration: V1PairedBCEAcquisitionRegistration | None = None
    paired_bce_state: V1PairedBCEAcquisitionState | None = None
    schema: str = _ACTIVE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "acquisition_receipt_hash",
            _nonempty(self.acquisition_receipt_hash, "acquisition_receipt_hash"),
        )
        object.__setattr__(self, "queue_index", _nonnegative_int(self.queue_index, "queue_index"))
        if not _is_modern_v1_proposal(self.proposal):
            raise TypeError("active candidate requires a closed nominal proposal")
        if type(self.threshold) is not V1CandidateDamageThreshold:
            raise TypeError("active candidate requires an exact threshold")
        if type(self.sampling_law) is not V1StepSamplingLaw:
            raise TypeError("active candidate requires an exact sampling law")
        if type(self.step_freeze) is not V1StagedAlphaStepFreeze:
            raise TypeError("active candidate requires an exact step freeze")
        if type(self.ledger) is not V1StagedAlphaLedger:
            raise TypeError("active candidate requires an exact staged-alpha ledger")
        paired_registration = self.paired_bce_registration
        paired_state = self.paired_bce_state
        if (paired_registration is None) != (paired_state is None):
            raise ValueError("active paired-BCE oracle state must be registered")
        if paired_registration is not None:
            if type(paired_registration) is not V1PairedBCEAcquisitionRegistration:
                raise TypeError("active paired-BCE registration has the wrong type")
            if type(paired_state) is not V1PairedBCEAcquisitionState:
                raise TypeError("active paired-BCE state has the wrong type")
            paired_registration.assert_integrity()
            paired_state.assert_integrity()
        freeze = self.step_freeze.certificate_freeze
        calibration_freeze = calibration_freeze_from_certificate(
            freeze,
            required=freeze.metadata.get(
                "candidate_calibration_freeze_required",
                False,
            )
            is True,
        )
        loss_referent = self.threshold.metadata.get(
            "loss_referent_mask_hash",
            self.threshold.parent_mask_hash,
        )
        if (
            self.threshold.acquisition_receipt_hash != self.acquisition_receipt_hash
            or self.threshold.candidate_id != self.proposal.candidate_id
            or self.threshold.parent_mask_hash != self.proposal.parent_mask_hash
            or self.threshold.candidate_mask_hash != self.proposal.resulting_mask.mask_hash
            or freeze.parent_mask_hash != loss_referent
            or freeze.candidate_mask_hash != self.proposal.resulting_mask.mask_hash
            or freeze.importance_weight_hash != self.sampling_law.importance_weight_hash
            or freeze.sampling_law_hash != self.sampling_law.sampling_law_hash
            or self.ledger.step_freeze.step_freeze_hash != self.step_freeze.step_freeze_hash
        ):
            raise ValueError("active candidate contains mismatched frozen identities")
        if calibration_freeze is not None and (
            calibration_freeze.campaign_contract_hash
            != self.threshold.campaign_contract_hash
            or calibration_freeze.acquisition_receipt_hash
            != self.acquisition_receipt_hash
            or calibration_freeze.queue_index != self.queue_index
            or calibration_freeze.candidate_id != self.proposal.candidate_id
            or calibration_freeze.proposal_parent_mask_hash
            != self.proposal.parent_mask_hash
            or calibration_freeze.calibration_policy_id
            != self.step_freeze.certificate_freeze.metadata.get(
                "calibration_policy_id"
                )
        ):
            raise ValueError("active candidate calibration freeze has stale lineage")
        if paired_registration is not None:
            assert paired_state is not None
            if calibration_freeze is None:
                raise ValueError("paired-BCE oracle requires a frozen candidate calibration")
            if (
                paired_registration.registration_hash != paired_state.registration_hash
                or paired_registration.campaign_contract_hash
                != self.threshold.campaign_contract_hash
                or paired_registration.candidate_mask_hash
                != self.proposal.resulting_mask.mask_hash
                or paired_registration.candidate_calibration_freeze_hash
                != calibration_freeze.freeze_hash
                or paired_state.dynamic_oracle_receipt_hash
                != paired_registration.dynamic_oracle_receipt_hash
                or paired_registration.candidate_attempt_index
                != self.step_freeze.candidate_attempt_index
                or not math.isclose(
                    paired_registration.loss_ceiling,
                    self.threshold.maximum_mean_damage,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise ValueError("active paired-BCE registration has stale candidate lineage")
            dynamic = paired_registration.dynamic_oracle_receipt
            if dynamic is not None and (
                dynamic.dense_affine_calibration_hash
                != calibration_freeze.dense_calibration.frozen_calibration_hash
                or dynamic.candidate_affine_calibration_hash
                != calibration_freeze.candidate_calibration.frozen_calibration_hash
                or dynamic.dense_reference_hash
                != calibration_freeze.dense_referent_mask_hash
                or dynamic.candidate_mask_hash
                != calibration_freeze.candidate_mask_hash
            ):
                raise ValueError("active paired-BCE dynamic O frame has stale affine state")
        if self.schema != _ACTIVE_SCHEMA:
            raise ValueError("unsupported active-candidate schema")

    @property
    def active_hash(self) -> str:
        if type(self.proposal) is V1OperatorCandidate:
            # Bind the complete frozen evidence definition, not the live
            # process-local producer capability. Historical hashes are intact.
            payload = {item.name: getattr(self, item.name) for item in fields(self)}
            payload["proposal"] = self.proposal.definition_payload()
            return content_hash(payload, prefix="partition-free-v1-active-candidate-")
        return content_hash(self, prefix="partition-free-v1-active-candidate-")


@dataclass(slots=True)
class _V1MaskCommitPermit:
    """Private, token-gated, single-use exact-child capability."""

    _issuer_token: object
    campaign_contract_hash: str
    controller_run_id: str
    acquisition_receipt_hash: str
    candidate_id: str
    parent_mask_hash: str
    child_mask_hash: str
    decision_receipt_hash: str
    ledger_hash: str
    consumed: bool = False

    def __post_init__(self) -> None:
        if self._issuer_token is not _COMMIT_TOKEN:
            raise TypeError("modern-v1 mask commit permits are controller-private")
        for name in (
            "campaign_contract_hash",
            "controller_run_id",
            "acquisition_receipt_hash",
            "candidate_id",
            "parent_mask_hash",
            "child_mask_hash",
            "decision_receipt_hash",
            "ledger_hash",
        ):
            _nonempty(getattr(self, name), name)
        if self.parent_mask_hash == self.child_mask_hash:
            raise ValueError("commit permit requires a changed exact child")

    @property
    def permit_hash(self) -> str:
        return content_hash(
            {
                "schema": _COMMIT_PERMIT_SCHEMA,
                "campaign_contract_hash": self.campaign_contract_hash,
                "controller_run_id": self.controller_run_id,
                "acquisition_receipt_hash": self.acquisition_receipt_hash,
                "candidate_id": self.candidate_id,
                "parent_mask_hash": self.parent_mask_hash,
                "child_mask_hash": self.child_mask_hash,
                "decision_receipt_hash": self.decision_receipt_hash,
                "ledger_hash": self.ledger_hash,
            },
            prefix="partition-free-v1-private-commit-permit-",
        )

    def consume(self, *, parent: HardMask, child: HardMask) -> str:
        if self.consumed:
            raise PermissionError("modern-v1 mask commit permit was already consumed")
        if parent.mask_hash != self.parent_mask_hash or child.mask_hash != self.child_mask_hash:
            raise PermissionError("modern-v1 commit permit names another parent or child")
        removed_count, _ = child.removed_index_count_and_hash_from(parent)
        if removed_count <= 0:
            raise PermissionError("modern-v1 commit permit requires a nonempty mask delta")
        self.consumed = True
        return content_hash(
            {
                "permit_hash": self.permit_hash,
                "parent_mask_hash": parent.mask_hash,
                "child_mask_hash": child.mask_hash,
                "single_use_consumed": True,
            },
            prefix="partition-free-v1-commit-consumption-",
        )


@dataclass(frozen=True, slots=True)
class V1ControllerState:
    """Immutable complete logical state of one modern-v1 development run."""

    binding: V1OptionACampaignBinding
    controller_run_id: str
    initial_mask: HardMask
    current_mask: HardMask
    phase: ControllerPhase
    acquisition: ModernV1Acquisition | None = None
    queue_cursor: int = 0
    label_native_runtime_plan: V1LabelNativeRuntimeFallbackPlan | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    label_native_runtime_plan_hashes: tuple[str, ...] = ()
    label_native_rejected_plan_indices: tuple[int, ...] = ()
    active_candidate: V1ActiveCandidate | None = None
    accepted_trajectory: V1StagedAlphaTrajectoryLedger = field(
        default_factory=V1StagedAlphaTrajectoryLedger
    )
    accepted_oracle_trajectory: V1AcceptedOracleTrajectoryLedger | None = None
    attempt_receipts: tuple[V1CandidateAttemptReceipt, ...] = ()
    transition_receipts: tuple[V1MaskTransitionReceipt, ...] = ()
    acquisition_receipt_hashes: tuple[str, ...] = ()
    globally_used_draw_ids: tuple[str, ...] = ()
    pareto_milestone_registration: V1ParetoMilestoneRegistration | None = None
    live_exact_prefix_rung_registration: V1ExactPrefixRungRegistration | None = None
    pending_pareto_milestone: V1ParetoMilestonePending | None = None
    pareto_milestone_receipt: V1ParetoMilestoneReceipt | None = None
    fqr_no_candidate_horizon_witness: V1ControllerProposalHorizonWitness | None = None
    stop_reason: StopReason | None = None
    logical_sequence: int = 0
    schema: str = _CONTROLLER_SCHEMA

    def __post_init__(self) -> None:
        if type(self.binding) is not V1OptionACampaignBinding:
            raise TypeError("controller requires an exact campaign binding")
        object.__setattr__(self, "controller_run_id", _nonempty(self.controller_run_id, "controller_run_id"))
        if type(self.initial_mask) is not HardMask or type(self.current_mask) is not HardMask:
            raise TypeError("controller masks must be exact HardMask objects")
        self.initial_mask.assert_same_axis(self.current_mask)
        if self.initial_mask.route_axis_hash != self.binding.contract.route_axis_hash:
            raise ValueError("controller mask axis differs from the campaign contract")
        object.__setattr__(self, "queue_cursor", _nonnegative_int(self.queue_cursor, "queue_cursor"))
        object.__setattr__(
            self,
            "logical_sequence",
            _nonnegative_int(self.logical_sequence, "logical_sequence"),
        )
        if self.phase not in {"needs_acquisition", "queue_ready", "candidate_active", "stopped"}:
            raise ValueError("unsupported modern-v1 controller phase")
        if type(self.accepted_trajectory) is not V1StagedAlphaTrajectoryLedger:
            raise TypeError("controller requires an exact accepted trajectory")
        oracle_trajectory = self.accepted_oracle_trajectory
        if oracle_trajectory is not None and type(
            oracle_trajectory
        ) is not V1AcceptedOracleTrajectoryLedger:
            raise TypeError("controller accepted-oracle trajectory has the wrong type")
        if oracle_trajectory is not None:
            oracle_trajectory.__post_init__()
        attempts = tuple(self.attempt_receipts)
        transitions = tuple(self.transition_receipts)
        if any(type(item) is not V1CandidateAttemptReceipt for item in attempts):
            raise TypeError("controller attempt history contains a nonreceipt value")
        if any(type(item) is not V1MaskTransitionReceipt for item in transitions):
            raise TypeError("controller transition history contains a nonreceipt value")
        legacy_accepted_count = len(self.accepted_trajectory.step_decisions)
        oracle_accepted_count = (
            None if oracle_trajectory is None else len(oracle_trajectory.entries)
        )
        if oracle_accepted_count is None:
            if len(transitions) != legacy_accepted_count:
                raise ValueError(
                    "controller transition and accepted-trajectory counts differ"
                )
        elif len(transitions) != oracle_accepted_count:
            raise ValueError(
                "controller transition and accepted-oracle trajectory counts differ"
            )
        elif legacy_accepted_count != 0:
            raise ValueError(
                "controller cannot mix legacy staged trajectory entries with "
                "accepted-oracle trajectory entries"
            )
        if sum(item.action == "accept_candidate" for item in attempts) != len(transitions):
            raise ValueError("controller attempt history and transition count differ")
        if self.active_candidate is not None and type(
            self.active_candidate
        ) is not V1ActiveCandidate:
            raise TypeError("controller active candidate must be exact")

        runtime_plan = self.label_native_runtime_plan
        plan_hashes = tuple(
            _nonempty(value, "label_native_runtime_plan_hash")
            for value in self.label_native_runtime_plan_hashes
        )
        if len(set(plan_hashes)) != len(plan_hashes):
            raise ValueError("label-native runtime plan history must be unique")
        rejected_plan_indices = tuple(
            _nonnegative_int(value, "label_native_rejected_plan_index")
            for value in self.label_native_rejected_plan_indices
        )
        if runtime_plan is None:
            if rejected_plan_indices:
                raise ValueError(
                    "controller retained label-native rejections without their live plan"
                )
            if _is_label_native_acquisition(self.acquisition):
                raise ValueError(
                    "label-native selected acquisition lacks its immutable runtime plan"
                )
        else:
            if type(runtime_plan) is not V1LabelNativeRuntimeFallbackPlan:
                raise TypeError("controller label-native runtime plan has the wrong type")
            plan_hash = runtime_plan.assert_integrity()
            if not plan_hashes or plan_hashes[-1] != plan_hash:
                raise ValueError(
                    "live label-native plan is absent from immutable plan history"
                )
            if (
                runtime_plan.campaign_identity_hash != self.binding.binding_hash
                or runtime_plan.campaign_contract_hash
                != self.binding.contract.campaign_contract_hash
                or runtime_plan.campaign_registration_hash
                != self.binding.registration.registration_hash
                or runtime_plan.live_current_mask_hash != self.current_mask.mask_hash
                or runtime_plan.live_route_axis_hash != self.current_mask.route_axis_hash
                or runtime_plan.live_route_count != self.current_mask.route_count
                or runtime_plan.live_proposal_generation
                != self.acquisition_generation_index
            ):
                raise ValueError(
                    "controller retains a stale label-native runtime plan frame"
                )
            if self.queue_cursor > len(runtime_plan.descriptors):
                raise ValueError("label-native plan cursor lies outside its frozen prefix")
            if rejected_plan_indices != tuple(range(self.queue_cursor)):
                raise ValueError(
                    "label-native rejected-index ledger is not the exact consumed prefix"
                )
            if self.phase in {"needs_acquisition", "queue_ready", "candidate_active"} and (
                self.queue_cursor >= len(runtime_plan.descriptors)
            ):
                raise ValueError(
                    "live label-native plan phase cannot retain an exhausted cursor"
                )
        object.__setattr__(self, "label_native_runtime_plan_hashes", plan_hashes)
        object.__setattr__(
            self,
            "label_native_rejected_plan_indices",
            rejected_plan_indices,
        )

        milestone_registration = self.pareto_milestone_registration
        live_rung = self.live_exact_prefix_rung_registration
        pending_milestone = self.pending_pareto_milestone
        milestone_receipt = self.pareto_milestone_receipt
        horizon_witness = self.fqr_no_candidate_horizon_witness
        if horizon_witness is not None and type(
            horizon_witness
        ) is not V1ControllerProposalHorizonWitness:
            raise TypeError(
                "controller F/Q/R no-candidate horizon witness has the wrong type"
            )
        if milestone_registration is None:
            if any(
                value is not None
                for value in (live_rung, pending_milestone, milestone_receipt)
            ):
                raise ValueError(
                    "controller carries Pareto milestone state without its registration"
                )
        elif type(milestone_registration) is not V1ParetoMilestoneRegistration:
            raise TypeError("controller Pareto milestone registration has the wrong type")
        else:
            contract = self.binding.contract
            if (
                milestone_registration.campaign_contract_hash
                != contract.campaign_contract_hash
                or milestone_registration.route_axis_hash != contract.route_axis_hash
                or milestone_registration.route_count != self.initial_mask.route_count
            ):
                raise ValueError(
                    "controller Pareto milestone registration names another campaign axis"
                )
            comparator_count = milestone_registration.comparator_retained_route_count
            if self.initial_mask.retained_count <= comparator_count:
                raise ValueError(
                    "controller must begin strictly denser than its Pareto milestone"
                )
            if (
                self.initial_mask.retained_count - comparator_count
                > contract.maximum_removed_count
            ):
                raise ValueError(
                    "Pareto milestone lies beyond the registered removal horizon"
                )
            if live_rung is not None and type(live_rung) is not V1ExactPrefixRungRegistration:
                raise TypeError("controller live exact-prefix registration has the wrong type")
            if pending_milestone is not None and type(
                pending_milestone
            ) is not V1ParetoMilestonePending:
                raise TypeError("controller pending Pareto milestone has the wrong type")
            if milestone_receipt is not None and type(
                milestone_receipt
            ) is not V1ParetoMilestoneReceipt:
                raise TypeError("controller Pareto milestone receipt has the wrong type")

        expected_parent_mask_hash = self.initial_mask.mask_hash
        expected_accepted_step = 0
        accepted_attempts: list[V1CandidateAttemptReceipt] = []
        previous_attempt_sequence = -1
        previous_acquisition_history_key: str | None = None
        previous_queue_index = -1
        for expected_attempt_index, attempt in enumerate(attempts):
            if attempt.action == "skip_candidate_calibration_refusal":
                skip = _calibration_horizon_skip_from_attempt(attempt)
                if (
                    skip.maximum_candidate_attempts
                    != self.binding.contract.maximum_candidate_count
                    or skip.accepted_step_index != expected_accepted_step
                ):
                    raise ValueError(
                        "calibration-horizon skip differs from the campaign frontier"
                    )
            if (
                attempt.campaign_contract_hash
                != self.binding.contract.campaign_contract_hash
                or attempt.controller_run_id != self.controller_run_id
                or attempt.candidate_attempt_index != expected_attempt_index
                or attempt.accepted_step_index != expected_accepted_step
                or attempt.acquisition_generation_index != expected_accepted_step
                or attempt.parent_mask_hash != expected_parent_mask_hash
                or attempt.logical_sequence <= previous_attempt_sequence
            ):
                raise ValueError(
                    "controller attempt history is nonconsecutive or has alternate lineage"
                )
            acquisition_history_key = _attempt_history_key(attempt)
            if acquisition_history_key == previous_acquisition_history_key:
                if attempt.queue_index != previous_queue_index + 1:
                    raise ValueError(
                        "controller queue history is nonconsecutive within an acquisition"
                    )
            elif attempt.queue_index != 0:
                raise ValueError("a fresh acquisition must begin at global queue index zero")
            previous_acquisition_history_key = acquisition_history_key
            previous_queue_index = attempt.queue_index
            previous_attempt_sequence = attempt.logical_sequence
            if attempt.action == "accept_candidate":
                accepted_attempts.append(attempt)
                expected_parent_mask_hash = attempt.candidate_mask_hash
                expected_accepted_step += 1
                previous_acquisition_history_key = None
                previous_queue_index = -1
        label_native_attempts = tuple(
            attempt
            for attempt in attempts
            if attempt.metadata.get("label_native_runtime_plan_hash") is not None
        )
        for attempt in label_native_attempts:
            attempt_plan_hash = _nonempty(
                attempt.metadata.get("label_native_runtime_plan_hash"),
                "label_native_runtime_plan_hash",
            )
            if (
                attempt_plan_hash not in plan_hashes
                or attempt.metadata.get("label_native_plan_queue_index")
                != attempt.queue_index
                or attempt.metadata.get(
                    "label_native_selected_acquisition_receipt_hash"
                )
                != attempt.acquisition_receipt_hash
                or attempt.metadata.get(
                    "exactly_one_endpoint_materialized_at_current_cursor"
                )
                is not True
                or attempt.metadata.get("unselected_fallback_masks_materialized")
                is not False
                or attempt.metadata.get("provider_handle_retained") is not False
                or attempt.metadata.get("oracle_conditioned_composition") is not False
            ):
                raise ValueError(
                    "label-native attempt left its receipt-only plan transaction"
                )
        if runtime_plan is not None:
            current_plan_attempts = tuple(
                attempt
                for attempt in label_native_attempts
                if attempt.metadata.get("label_native_runtime_plan_hash")
                == runtime_plan.plan_hash
            )
            if (
                tuple(attempt.queue_index for attempt in current_plan_attempts)
                != rejected_plan_indices
                or any(
                    attempt.action
                    not in {"reject_candidate", "skip_candidate_calibration_refusal"}
                    for attempt in current_plan_attempts
                )
            ):
                raise ValueError(
                    "live label-native rejection ledger differs from durable attempts"
                )
        if expected_parent_mask_hash != self.current_mask.mask_hash:
            raise ValueError("controller current mask does not close its attempt ancestry")
        if attempts and attempts[-1].logical_sequence > self.logical_sequence:
            raise ValueError("controller attempt history lies after the state sequence")

        if oracle_trajectory is None:
            for expected_transition_index, (
                transition,
                attempt,
                ledger,
                decision,
            ) in enumerate(
                zip(
                    transitions,
                    accepted_attempts,
                    self.accepted_trajectory.step_ledgers,
                    self.accepted_trajectory.step_decisions,
                    strict=True,
                )
            ):
                freeze = ledger.step_freeze.certificate_freeze
                if (
                    transition.campaign_contract_hash
                    != self.binding.contract.campaign_contract_hash
                    or transition.controller_run_id != self.controller_run_id
                    or transition.transition_index != expected_transition_index
                    or transition.accepted_step_index != expected_transition_index
                    or transition.candidate_attempt_index
                    != attempt.candidate_attempt_index
                    or transition.candidate_id != attempt.candidate_id
                    or transition.parent_mask_hash != attempt.parent_mask_hash
                    or transition.child_mask_hash != attempt.candidate_mask_hash
                    or transition.accepted_decision_receipt_hash
                    != attempt.decision_receipt_hash
                    or transition.accepted_ledger_hash != attempt.ledger_hash
                    or transition.acquisition_receipt_hash
                    != attempt.acquisition_receipt_hash
                    or transition.logical_sequence != attempt.logical_sequence
                    or decision.decision_receipt_hash != attempt.decision_receipt_hash
                    or ledger.ledger_hash != attempt.ledger_hash
                    or freeze.parent_mask_hash != self.initial_mask.mask_hash
                    or freeze.candidate_mask_hash != attempt.candidate_mask_hash
                ):
                    raise ValueError(
                        "controller transition, attempt, and accepted trajectory do not align"
                    )
        else:
            for expected_transition_index, (transition, attempt, entry) in enumerate(
                zip(
                    transitions,
                    accepted_attempts,
                    oracle_trajectory.entries,
                    strict=True,
                )
            ):
                entry_calibration = entry.candidate_calibration_freeze
                entry_calibration_mismatch = False
                if entry_calibration is not None:
                    entry_calibration_mismatch = (
                        type(entry_calibration) is not V1CandidateCalibrationFreeze
                        or entry_calibration.campaign_contract_hash
                        != self.binding.contract.campaign_contract_hash
                        or entry_calibration.queue_index != attempt.queue_index
                        or entry_calibration.dense_referent_mask_hash
                        != self.initial_mask.mask_hash
                        or entry_calibration.calibration_policy_id
                        != self.binding.contract.calibration_identity
                        or attempt.metadata.get(
                            "candidate_calibration_freeze_hash"
                        )
                        != entry_calibration.freeze_hash
                    )
                if (
                    transition.campaign_contract_hash
                    != self.binding.contract.campaign_contract_hash
                    or transition.controller_run_id != self.controller_run_id
                    or transition.transition_index != expected_transition_index
                    or transition.accepted_step_index != expected_transition_index
                    or transition.candidate_attempt_index
                    != attempt.candidate_attempt_index
                    or transition.candidate_id != attempt.candidate_id
                    or transition.parent_mask_hash != attempt.parent_mask_hash
                    or transition.child_mask_hash != attempt.candidate_mask_hash
                    or transition.accepted_decision_receipt_hash
                    != attempt.decision_receipt_hash
                    or transition.accepted_ledger_hash != attempt.ledger_hash
                    or transition.acquisition_receipt_hash
                    != attempt.acquisition_receipt_hash
                    or transition.logical_sequence != attempt.logical_sequence
                    or entry.accepted_step_index != expected_transition_index
                    or entry.candidate_attempt_index != attempt.candidate_attempt_index
                    or entry.acquisition_receipt_hash != attempt.acquisition_receipt_hash
                    or entry.candidate_id != attempt.candidate_id
                    or entry.parent_mask_hash != attempt.parent_mask_hash
                    or entry.candidate_mask_hash != attempt.candidate_mask_hash
                    or entry.step_freeze_hash != attempt.step_freeze_hash
                    or entry.evidence_ledger_hash != attempt.ledger_hash
                    or entry.decision_receipt_hash != attempt.decision_receipt_hash
                    or entry_calibration_mismatch
                ):
                    raise ValueError(
                        "controller transition, attempt, and accepted oracle trajectory do not align"
                    )

        expected_draw_history = tuple(
            draw_id for attempt in attempts for draw_id in attempt.draw_ids
        )
        if self.active_candidate is not None:
            expected_draw_history += tuple(
                draw_id
                for packet in self.active_candidate.ledger.packets
                for draw_id in packet.draw_ids
            )
        acquisition_hashes = tuple(
            _nonempty(value, "acquisition_receipt_hash")
            for value in self.acquisition_receipt_hashes
        )
        if len(set(acquisition_hashes)) != len(acquisition_hashes):
            raise ValueError("controller acquisition receipt history must be unique")
        accepted_count = len(transitions)
        if not plan_hashes:
            if self.phase == "needs_acquisition" and len(acquisition_hashes) != accepted_count:
                raise ValueError("needs-acquisition controller has the wrong acquisition history length")
            if self.phase in {"queue_ready", "candidate_active"} and len(acquisition_hashes) != accepted_count + 1:
                raise ValueError("queue-bearing controller has the wrong acquisition history length")
            if self.phase == "stopped" and len(acquisition_hashes) not in {
                accepted_count,
                accepted_count + 1,
            }:
                raise ValueError("stopped controller has an impossible acquisition history length")
        acquisition_set = set(acquisition_hashes)
        if any(item.acquisition_receipt_hash not in acquisition_set for item in attempts):
            raise ValueError("controller attempt history names an unrecorded acquisition")
        if any(item.acquisition_receipt_hash not in acquisition_set for item in transitions):
            raise ValueError("controller transition history names an unrecorded acquisition")
        label_native_receipt_hashes = tuple(
            attempt.acquisition_receipt_hash for attempt in label_native_attempts
        )
        if _is_label_native_acquisition(self.acquisition):
            label_native_receipt_hashes += (self.acquisition.receipt.receipt_hash,)
        if (
            len(set(label_native_receipt_hashes)) != len(label_native_receipt_hashes)
            or any(value not in acquisition_set for value in label_native_receipt_hashes)
        ):
            raise ValueError(
                "label-native selected-receipt history is duplicated or incomplete"
            )
        draws = tuple(_nonempty(value, "globally_used_draw_id") for value in self.globally_used_draw_ids)
        if len(set(draws)) != len(draws):
            raise ValueError("controller globally reused a certificate draw ID")
        if draws != expected_draw_history:
            raise ValueError("controller global draw ledger differs from exact attempt history")
        object.__setattr__(self, "attempt_receipts", attempts)
        object.__setattr__(self, "transition_receipts", transitions)
        object.__setattr__(self, "acquisition_receipt_hashes", acquisition_hashes)
        object.__setattr__(self, "globally_used_draw_ids", draws)
        if milestone_registration is not None:
            comparator_count = milestone_registration.comparator_retained_route_count
            if pending_milestone is not None and milestone_receipt is not None:
                raise ValueError(
                    "controller cannot carry both pending and sealed Pareto milestones"
                )
            if pending_milestone is not None:
                if (
                    self.current_mask.retained_count != comparator_count
                    or pending_milestone.campaign_contract_hash
                    != self.binding.contract.campaign_contract_hash
                    or pending_milestone.campaign_binding_hash
                    != self.binding.binding_hash
                    or pending_milestone.controller_run_id != self.controller_run_id
                    or pending_milestone.milestone_registration_hash
                    != milestone_registration.registration_hash
                    or pending_milestone.comparator_retained_route_count
                    != comparator_count
                    or pending_milestone.milestone_mask_hash
                    != self.current_mask.mask_hash
                    or pending_milestone.mask_lineage_hashes
                    != self.mask_lineage_hashes
                    or pending_milestone.transition_receipt_hashes
                    != self.transition_receipt_hashes
                    or pending_milestone.accepted_trajectory_hash
                    != self.accepted_trajectory_hash
                    or pending_milestone.accepted_step_count
                    != self.accepted_step_count
                    or pending_milestone.candidate_attempt_count
                    != self.candidate_attempt_count
                    or not self.transition_receipt_hashes
                    or pending_milestone.milestone_transition_receipt_hash
                    != self.transition_receipt_hashes[-1]
                    or pending_milestone.logical_sequence != self.logical_sequence
                ):
                    raise ValueError(
                        "pending Pareto milestone differs from controller state or lineage"
                    )
            if milestone_receipt is not None:
                receipt_pending = milestone_receipt.pending
                prefix_steps = receipt_pending.accepted_step_count
                if (
                    receipt_pending.milestone_registration_hash
                    != milestone_registration.registration_hash
                    or receipt_pending.comparator_retained_route_count
                    != comparator_count
                    or self.current_mask.retained_count > comparator_count
                    or self.mask_lineage_hashes[: prefix_steps + 1]
                    != receipt_pending.mask_lineage_hashes
                    or self.transition_receipt_hashes[:prefix_steps]
                    != receipt_pending.transition_receipt_hashes
                ):
                    raise ValueError(
                        "sealed Pareto milestone is not an immutable prefix of this controller"
                    )
            if self.current_mask.retained_count > comparator_count:
                if pending_milestone is not None or milestone_receipt is not None:
                    raise ValueError("Pareto milestone appears before its registered count")
            elif self.current_mask.retained_count == comparator_count:
                if (pending_milestone is None) == (milestone_receipt is None):
                    raise ValueError(
                        "controller at the Pareto count must carry exactly one milestone state"
                    )
            elif milestone_receipt is None:
                raise ValueError(
                    "controller crossed the Pareto count without a sealed milestone"
                )
        expected_phase_fields = {
            "needs_acquisition": (False, False, False),
            "queue_ready": (True, False, False),
            "candidate_active": (True, True, False),
            "stopped": (False, False, True),
        }
        expected_acquisition, expected_active, expected_stop = expected_phase_fields[self.phase]
        if (self.acquisition is not None) is not expected_acquisition:
            raise ValueError("controller phase and acquisition ownership disagree")
        if (self.active_candidate is not None) is not expected_active:
            raise ValueError("controller phase and active-candidate ownership disagree")
        if (self.stop_reason is not None) is not expected_stop:
            raise ValueError("controller phase and stop reason disagree")
        if (horizon_witness is not None) != (
            self.stop_reason == FQR_NO_CANDIDATE_HORIZON_STOP_REASON
        ):
            raise ValueError(
                "controller F/Q/R horizon reason and witness disagree"
            )
        if horizon_witness is not None:
            proof = horizon_witness.proof
            expected_lineage_hash = _v1_controller_proposal_mask_lineage_hash(
                self.mask_lineage_hashes,
                self.transition_receipt_hashes,
            )
            if (
                horizon_witness.campaign_contract_hash
                != self.binding.contract.campaign_contract_hash
                or horizon_witness.campaign_binding_hash != self.binding.binding_hash
                or horizon_witness.controller_run_id != self.controller_run_id
                or horizon_witness.generation_index
                != self.acquisition_generation_index
                or horizon_witness.current_mask_hash != self.current_mask.mask_hash
                or horizon_witness.route_axis_hash
                != self.current_mask.route_axis_hash
                or horizon_witness.mask_lineage_hash != expected_lineage_hash
                or horizon_witness.post_stop_logical_sequence
                != self.logical_sequence
                or proof.campaign_contract_hash
                != self.binding.contract.campaign_contract_hash
            ):
                raise ValueError(
                    "controller F/Q/R horizon witness names another live lineage"
                )
            pre_stop_state = replace(
                self,
                phase="needs_acquisition",
                fqr_no_candidate_horizon_witness=None,
                stop_reason=None,
                logical_sequence=self.logical_sequence - 1,
            )
            if (
                horizon_witness.pre_stop_controller_state_hash
                != pre_stop_state.state_hash
                or horizon_witness.pre_stop_controller_snapshot_hash
                != pre_stop_state.snapshot_receipt().receipt_hash
                or horizon_witness.pre_stop_logical_sequence
                != pre_stop_state.logical_sequence
            ):
                raise ValueError(
                    "controller F/Q/R horizon witness does not replay its pre-stop state"
                )
        if (
            self.stop_reason
            == ALL_REGISTERED_FQR_ACTIONS_REJECTED_BY_PAIRED_BCE_BARRIER
            and not _paired_fqr_barrier_attempt_history_is_saturated(attempts)
        ):
            raise ValueError(
                "paired-BCE F/Q/R saturation endpoint lacks a complete decisive "
                "barrier-rejection history"
            )
        if self.acquisition is not None:
            if not _is_modern_v1_acquisition(self.acquisition):
                raise TypeError("controller requires an exact modern-v1 acquisition")
            if type(self.acquisition) is V1OperatorAcquisition:
                self.acquisition.assert_integrity()
            if _is_label_native_acquisition(self.acquisition):
                if runtime_plan is None:
                    raise ValueError(
                        "label-native selected acquisition lost its runtime plan"
                    )
                if (
                    self.acquisition.binding.binding_hash != self.binding.binding_hash
                    or self.acquisition.current_mask.mask_hash
                    != self.current_mask.mask_hash
                    or self.acquisition.receipt.runtime_plan_hash
                    != runtime_plan.plan_hash
                    or self.acquisition.receipt.plan_queue_index != self.queue_cursor
                    or self.acquisition.receipt.runtime_descriptor_hash
                    != runtime_plan.descriptors[self.queue_cursor].descriptor_hash
                    or self.acquisition.committed_mask_lineage_hashes
                    != self.mask_lineage_hashes
                    or self.acquisition.committed_mask_transition_receipt_hashes
                    != self.transition_receipt_hashes
                ):
                    raise ValueError(
                        "controller retains a stale label-native selected endpoint"
                    )
            elif (
                self.acquisition.binding.binding_hash != self.binding.binding_hash
                or self.acquisition.current_mask.mask_hash != self.current_mask.mask_hash
                or self.queue_cursor > len(self.acquisition.queue)
            ):
                raise ValueError("controller retains a stale or invalid acquisition")
            if (
                not acquisition_hashes
                or acquisition_hashes[-1] != self.acquisition.receipt.receipt_hash
            ):
                raise ValueError("controller live acquisition is absent from durable history")
        if live_rung is not None:
            if self.acquisition is None or milestone_registration is None:
                raise ValueError("live exact-prefix registration lacks its acquisition")
            if _is_label_native_acquisition(self.acquisition):
                if live_rung.queue_index != self.queue_cursor:
                    raise ValueError(
                        "live exact-prefix registration differs from selected plan cursor"
                    )
                rung_proposal = self.acquisition.candidate
            else:
                if live_rung.queue_index >= len(self.acquisition.queue):
                    raise ValueError("live exact-prefix registration lies outside the queue")
                rung_proposal = self.acquisition.queue[live_rung.queue_index]
            if (
                live_rung.campaign_contract_hash
                != self.binding.contract.campaign_contract_hash
                or live_rung.milestone_registration_hash
                != milestone_registration.registration_hash
                or live_rung.acquisition_receipt_hash
                != self.acquisition.receipt.receipt_hash
                or live_rung.generation_index != self.acquisition.generation_index
                or live_rung.parent_mask_hash != self.current_mask.mask_hash
                or live_rung.parent_retained_route_count
                != self.current_mask.retained_count
                or live_rung.comparator_retained_route_count
                != milestone_registration.comparator_retained_route_count
                or live_rung.candidate_id != rung_proposal.candidate_id
                or live_rung.candidate_mask_hash
                != rung_proposal.resulting_mask.mask_hash
                or live_rung.required_removed_route_count
                != rung_proposal.removed_count
                or live_rung.removed_route_index_hash
                != rung_proposal.removed_route_index_hash
            ):
                raise ValueError(
                    "live exact-prefix registration differs from its acquisition proposal"
                )
        elif self.acquisition is not None and milestone_registration is not None:
            comparator_count = milestone_registration.comparator_retained_route_count
            if (
                self.current_mask.retained_count > comparator_count
                and any(
                    proposal.resulting_mask.retained_count <= comparator_count
                    for proposal in (
                        (self.acquisition.candidate,)
                        if _is_label_native_acquisition(self.acquisition)
                        else self.acquisition.queue
                    )
                )
            ):
                raise ValueError(
                    "live ladder reaches or crosses the Pareto count without a registered rung"
                )
        if self.active_candidate is not None:
            if self.acquisition is None:
                raise ValueError("active candidate lacks its acquisition")
            selected = _selected_proposal(
                self.acquisition,
                queue_cursor=self.queue_cursor,
            )
            if (
                self.active_candidate.queue_index != self.queue_cursor
                or selected.candidate_id != self.active_candidate.proposal.candidate_id
            ):
                raise ValueError("active candidate is not the exact queue cursor proposal")
            if (
                self.active_candidate.step_freeze.candidate_attempt_index
                != self.candidate_attempt_count
            ):
                raise ValueError(
                    "active candidate alpha stake differs from the global attempt frontier"
                )
            paired_registration = self.active_candidate.paired_bce_registration
            dynamic = (
                None
                if paired_registration is None
                else paired_registration.dynamic_oracle_receipt
            )
            if (
                dynamic is not None
                and dynamic.generation_index != self.acquisition.generation_index
            ):
                raise ValueError(
                    "active paired-BCE dynamic O frame belongs to a stale generation"
                )
        if pending_milestone is not None and self.phase not in {
            "needs_acquisition",
            "stopped",
        }:
            raise ValueError(
                "pending Pareto milestone must cross the checkpoint seam before acquisition"
            )
        if self.cumulative_removed_count > self.binding.contract.maximum_removed_count:
            raise ValueError("controller state exceeds the campaign removal cap")
        if self.accepted_step_count > self.binding.contract.maximum_step_count:
            raise ValueError("controller state exceeds the campaign step cap")
        if self.candidate_attempt_count > self.binding.contract.maximum_candidate_count:
            raise ValueError("controller state exceeds the campaign candidate cap")
        if self.schema != _CONTROLLER_SCHEMA:
            raise ValueError("unsupported modern-v1 controller-state schema")

    @property
    def accepted_step_count(self) -> int:
        return len(self.transition_receipts)

    @property
    def candidate_attempt_count(self) -> int:
        return len(self.attempt_receipts)

    @property
    def cumulative_removed_count(self) -> int:
        return self.initial_mask.retained_count - self.current_mask.retained_count

    @property
    def acquisition_generation_index(self) -> int:
        return self.accepted_step_count

    @property
    def mask_lineage_hashes(self) -> tuple[str, ...]:
        return (self.initial_mask.mask_hash,) + tuple(
            receipt.child_mask_hash for receipt in self.transition_receipts
        )

    @property
    def transition_receipt_hashes(self) -> tuple[str, ...]:
        return tuple(receipt.receipt_hash for receipt in self.transition_receipts)

    @property
    def attempt_receipt_hashes(self) -> tuple[str, ...]:
        return tuple(receipt.receipt_hash for receipt in self.attempt_receipts)

    @property
    def accepted_trajectory_hash(self) -> str:
        if self.accepted_oracle_trajectory is not None:
            return self.accepted_oracle_trajectory.trajectory_ledger_hash
        return self.accepted_trajectory.trajectory_ledger_hash

    @property
    def state_hash(self) -> str:
        payload: dict[str, object] = {
            "schema": self.schema,
            "campaign_binding_hash": self.binding.binding_hash,
            "controller_run_id": self.controller_run_id,
            "initial_mask_hash": self.initial_mask.mask_hash,
            "current_mask_hash": self.current_mask.mask_hash,
            "phase": self.phase,
            "acquisition_hash": (
                None if self.acquisition is None else self.acquisition.acquisition_hash
            ),
            "queue_cursor": self.queue_cursor,
            "active_candidate_hash": (
                None if self.active_candidate is None else self.active_candidate.active_hash
            ),
            "accepted_trajectory_hash": self.accepted_trajectory_hash,
            "attempt_receipt_hashes": self.attempt_receipt_hashes,
            "transition_receipt_hashes": self.transition_receipt_hashes,
            "acquisition_receipt_hashes": self.acquisition_receipt_hashes,
            "globally_used_draw_ids": self.globally_used_draw_ids,
            "stop_reason": self.stop_reason,
            "logical_sequence": self.logical_sequence,
        }
        if self.label_native_runtime_plan_hashes:
            payload.update(
                {
                    "label_native_runtime_plan_hash": (
                        None
                        if self.label_native_runtime_plan is None
                        else self.label_native_runtime_plan.plan_hash
                    ),
                    "label_native_runtime_plan_hashes": (
                        self.label_native_runtime_plan_hashes
                    ),
                    "label_native_rejected_plan_indices": (
                        self.label_native_rejected_plan_indices
                    ),
                    "label_native_selected_acquisition_receipt_hash": (
                        self.acquisition.receipt.receipt_hash
                        if _is_label_native_acquisition(self.acquisition)
                        else None
                    ),
                }
            )
        if self.pareto_milestone_registration is not None:
            payload.update(
                {
                    "pareto_milestone_registration_hash": (
                        self.pareto_milestone_registration.registration_hash
                    ),
                    "live_exact_prefix_rung_registration_hash": (
                        None
                        if self.live_exact_prefix_rung_registration is None
                        else self.live_exact_prefix_rung_registration.registration_hash
                    ),
                    "pending_pareto_milestone_hash": (
                        None
                        if self.pending_pareto_milestone is None
                        else self.pending_pareto_milestone.pending_hash
                    ),
                    "pareto_milestone_receipt_hash": (
                        None
                        if self.pareto_milestone_receipt is None
                        else self.pareto_milestone_receipt.receipt_hash
                    ),
                }
            )
        if self.fqr_no_candidate_horizon_witness is not None:
            payload["fqr_no_candidate_horizon_witness_hash"] = (
                self.fqr_no_candidate_horizon_witness.witness_hash
            )
        return content_hash(
            payload,
            prefix="partition-free-v1-controller-state-",
        )

    @property
    def scientific_horizon(self) -> V1ScientificHorizonDisposition:
        return classify_v1_scientific_horizon(self.stop_reason)

    @property
    def pareto_milestone_checkpoint_pending(self) -> bool:
        return self.pending_pareto_milestone is not None

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def mask_commit_authority(self) -> bool:
        return False

    def snapshot_receipt(self) -> V1ControllerSnapshotReceipt:
        active = self.active_candidate
        return V1ControllerSnapshotReceipt(
            campaign_contract_hash=self.binding.contract.campaign_contract_hash,
            campaign_binding_hash=self.binding.binding_hash,
            controller_run_id=self.controller_run_id,
            phase=self.phase,
            current_mask_hash=self.current_mask.mask_hash,
            acquisition_generation_index=self.acquisition_generation_index,
            queue_cursor=self.queue_cursor,
            accepted_step_count=self.accepted_step_count,
            candidate_attempt_count=self.candidate_attempt_count,
            cumulative_removed_count=self.cumulative_removed_count,
            active_candidate_id=(None if active is None else active.proposal.candidate_id),
            active_step_freeze_hash=(None if active is None else active.step_freeze.step_freeze_hash),
            active_ledger_hash=(
                None
                if active is None
                else (
                    active.paired_bce_state.state_hash
                    if active.paired_bce_state is not None
                    else active.ledger.ledger_hash
                )
            ),
            accepted_trajectory_hash=self.accepted_trajectory_hash,
            acquisition_receipt_hash=(
                None if self.acquisition is None else self.acquisition.receipt.receipt_hash
            ),
            acquisition_receipt_hashes=self.acquisition_receipt_hashes,
            attempt_receipt_hashes=self.attempt_receipt_hashes,
            transition_receipt_hashes=self.transition_receipt_hashes,
            globally_used_draw_ids_hash=content_hash(
                self.globally_used_draw_ids,
                prefix="partition-free-v1-globally-used-draw-ids-",
            ),
            stop_reason=self.stop_reason,
            logical_sequence=self.logical_sequence,
            fqr_no_candidate_horizon_witness_hash=(
                None
                if self.fqr_no_candidate_horizon_witness is None
                else self.fqr_no_candidate_horizon_witness.witness_hash
            ),
            label_native_runtime_plan_hash=(
                None
                if self.label_native_runtime_plan is None
                else self.label_native_runtime_plan.plan_hash
            ),
            label_native_runtime_plan_hashes=(
                self.label_native_runtime_plan_hashes
            ),
            label_native_rejected_plan_indices=(
                self.label_native_rejected_plan_indices
            ),
            label_native_selected_acquisition_receipt_hash=(
                self.acquisition.receipt.receipt_hash
                if _is_label_native_acquisition(self.acquisition)
                else None
            ),
            schema=(
                (
                    "partition_free_v1_controller_snapshot_receipt_v4"
                    if self.fqr_no_candidate_horizon_witness is None
                    else "partition_free_v1_controller_snapshot_receipt_v5"
                )
                if self.label_native_runtime_plan_hashes
                else (
                    "partition_free_v1_controller_snapshot_receipt_v2"
                    if self.fqr_no_candidate_horizon_witness is None
                    else "partition_free_v1_controller_snapshot_receipt_v3"
                )
            ),
        )


def initialize_v1_controller(
    *,
    binding: V1OptionACampaignBinding,
    controller_run_id: str,
    initial_mask: HardMask,
    pareto_milestone_registration: V1ParetoMilestoneRegistration | None = None,
) -> V1ControllerState:
    """Create a fail-closed controller that owns no queue yet."""

    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("controller initialization requires an exact campaign binding")
    if type(initial_mask) is not HardMask:
        raise TypeError("controller initialization requires an exact hard mask")
    removed = initial_mask.route_count - initial_mask.retained_count
    if removed > binding.contract.maximum_removed_count:
        raise ValueError("initial mask already exceeds the campaign removal cap")
    return V1ControllerState(
        binding=binding,
        controller_run_id=controller_run_id,
        initial_mask=initial_mask,
        current_mask=initial_mask,
        phase="needs_acquisition",
        pareto_milestone_registration=pareto_milestone_registration,
    )


def attach_v1_label_native_runtime_plan(
    state: V1ControllerState,
    plan: V1LabelNativeRuntimeFallbackPlan,
) -> V1ControllerState:
    """Retain one immutable receipt-only fallback plan before materialization.

    The plan carries descriptors and source receipts only.  This transaction
    stores no provider, hard mask, route row, calibration, or oracle object and
    opens no candidate decision authority.
    """

    if type(state) is not V1ControllerState:
        raise TypeError("label-native plan attachment requires an exact controller")
    if type(plan) is not V1LabelNativeRuntimeFallbackPlan:
        raise TypeError("label-native plan attachment requires the exact runtime plan")
    if (
        state.phase != "needs_acquisition"
        or state.acquisition is not None
        or state.active_candidate is not None
        or state.label_native_runtime_plan is not None
    ):
        raise PermissionError(
            "label-native plan can attach only at an acquisition-free generation seam"
        )
    if state.pending_pareto_milestone is not None:
        raise PermissionError(
            "Pareto milestone transition must cross the checkpoint seam before continuing"
        )
    plan_hash = plan.assert_integrity()
    if (
        plan.campaign_identity_hash != state.binding.binding_hash
        or plan.campaign_contract_hash
        != state.binding.contract.campaign_contract_hash
        or plan.campaign_registration_hash
        != state.binding.registration.registration_hash
        or plan.live_current_mask_hash != state.current_mask.mask_hash
        or plan.live_route_axis_hash != state.current_mask.route_axis_hash
        or plan.live_route_count != state.current_mask.route_count
        or plan.live_proposal_generation != state.acquisition_generation_index
    ):
        raise ValueError("label-native runtime plan is stale or names another lineage")
    if plan_hash in set(state.label_native_runtime_plan_hashes):
        raise ValueError("label-native runtime plan was already attached")
    return replace(
        state,
        label_native_runtime_plan=plan,
        label_native_runtime_plan_hashes=(
            state.label_native_runtime_plan_hashes + (plan_hash,)
        ),
        label_native_rejected_plan_indices=(),
        queue_cursor=0,
        logical_sequence=state.logical_sequence + 1,
    )


def attach_v1_label_native_selected_acquisition(
    state: V1ControllerState,
    acquisition: V1LabelNativeControllerSingleEndpointAcquisition,
    *,
    exact_prefix_rung_registration: V1ExactPrefixRungRegistration | None = None,
) -> V1ControllerState:
    """Attach exactly one materialized endpoint at the frozen plan cursor."""

    if type(state) is not V1ControllerState:
        raise TypeError(
            "label-native selected attachment requires an exact controller state"
        )
    if type(acquisition) is not V1LabelNativeControllerSingleEndpointAcquisition:
        raise TypeError(
            "label-native selected attachment requires the exact acquisition carrier"
        )
    plan = state.label_native_runtime_plan
    if (
        state.phase != "needs_acquisition"
        or state.acquisition is not None
        or state.active_candidate is not None
        or plan is None
    ):
        raise PermissionError(
            "controller is not awaiting one endpoint from a retained label-native plan"
        )
    if state.pending_pareto_milestone is not None:
        raise PermissionError(
            "Pareto milestone transition must cross the checkpoint seam before continuing"
        )
    acquisition.assert_integrity()
    receipt = acquisition.receipt
    descriptor = plan.descriptors[state.queue_cursor]
    if (
        acquisition.binding.binding_hash != state.binding.binding_hash
        or acquisition.generation_index != state.acquisition_generation_index
        or acquisition.current_mask.mask_hash != state.current_mask.mask_hash
        or acquisition.committed_mask_lineage_hashes != state.mask_lineage_hashes
        or acquisition.committed_mask_transition_receipt_hashes
        != state.transition_receipt_hashes
        or receipt.runtime_plan_hash != plan.plan_hash
        or receipt.plan_queue_index != state.queue_cursor
        or receipt.runtime_descriptor_hash != descriptor.descriptor_hash
        or acquisition.candidate.descriptor.descriptor_hash
        != descriptor.descriptor_hash
        or receipt.provider_handle_retained
        or receipt.unselected_fallback_masks_materialized
        or receipt.candidate_freeze_authority
        or receipt.calibration_K_open_authority
        or receipt.paired_oracle_O_open_authority
        or receipt.irreversible_decision_authority
        or receipt.mask_commit_authority
    ):
        raise ValueError(
            "selected label-native endpoint differs from the exact live plan cursor"
        )
    receipt_hash = receipt.receipt_hash
    if receipt_hash in set(state.acquisition_receipt_hashes):
        raise ValueError("selected label-native endpoint receipt was already consumed")

    milestone_registration = state.pareto_milestone_registration
    if exact_prefix_rung_registration is not None and type(
        exact_prefix_rung_registration
    ) is not V1ExactPrefixRungRegistration:
        raise TypeError("exact prefix-rung registration has the wrong type")
    if milestone_registration is None:
        if exact_prefix_rung_registration is not None:
            raise ValueError("exact prefix rung supplied without a Pareto registration")
    elif state.pareto_milestone_receipt is None:
        comparator_count = milestone_registration.comparator_retained_route_count
        candidate_count = acquisition.candidate.resulting_mask.retained_count
        if candidate_count <= comparator_count:
            if exact_prefix_rung_registration is None:
                raise PermissionError(
                    "selected endpoint reaches or crosses k_cmp without a registered exact prefix"
                )
            expected = register_v1_exact_pareto_prefix_rung(
                state,
                acquisition,
                queue_index=state.queue_cursor,
                v1_order_receipt_hash=(
                    exact_prefix_rung_registration.v1_order_receipt_hash
                ),
            )
            if expected.registration_hash != exact_prefix_rung_registration.registration_hash:
                raise ValueError("exact prefix-rung registration is not reproducible")
        elif exact_prefix_rung_registration is not None:
            raise ValueError("exact prefix rung supplied above k_cmp")
    elif exact_prefix_rung_registration is not None:
        raise ValueError("exact prefix rung supplied after the milestone was sealed")

    return replace(
        state,
        phase="queue_ready",
        acquisition=acquisition,
        acquisition_receipt_hashes=(
            state.acquisition_receipt_hashes + (receipt_hash,)
        ),
        live_exact_prefix_rung_registration=exact_prefix_rung_registration,
        logical_sequence=state.logical_sequence + 1,
    )


def register_v1_exact_pareto_prefix_rung(
    state: V1ControllerState,
    acquisition: ModernV1Acquisition,
    *,
    queue_index: int,
    v1_order_receipt_hash: str,
) -> V1ExactPrefixRungRegistration:
    """Register the exact v1-order prefix that lands on the comparator count.

    This is a pre-oracle registration seam.  The supplied order receipt must
    come from the v1 score-order producer; no comparator ordering coordinate is
    accepted.  Any proposal that reaches below the milestone before this rung
    makes the acquisition inadmissible rather than being silently truncated.
    """

    if type(state) is not V1ControllerState or not _is_modern_v1_acquisition(
        acquisition
    ):
        raise TypeError("exact prefix-rung registration requires exact controller/acquisition")
    if state.phase != "needs_acquisition":
        raise PermissionError("exact prefix rung can be registered only before acquisition attach")
    registration = state.pareto_milestone_registration
    if registration is None:
        raise PermissionError("controller has no Pareto milestone registration")
    if state.pending_pareto_milestone is not None:
        raise PermissionError("pending Pareto milestone must be checkpointed first")
    if state.pareto_milestone_receipt is not None:
        raise PermissionError("Pareto milestone is already sealed")
    if (
        acquisition.binding.binding_hash != state.binding.binding_hash
        or acquisition.generation_index != state.acquisition_generation_index
        or acquisition.current_mask.mask_hash != state.current_mask.mask_hash
        or acquisition.committed_mask_lineage_hashes != state.mask_lineage_hashes
        or acquisition.committed_mask_transition_receipt_hashes
        != state.transition_receipt_hashes
    ):
        raise ValueError("exact prefix-rung acquisition is stale or has alternate ancestry")
    index = _nonnegative_int(queue_index, "queue_index")
    if _is_label_native_acquisition(acquisition):
        if index != acquisition.receipt.plan_queue_index:
            raise ValueError(
                "exact prefix-rung index differs from the selected plan coordinate"
            )
    elif index >= len(acquisition.queue):
        raise ValueError("exact prefix-rung queue index lies outside the acquisition")
    comparator_count = registration.comparator_retained_route_count
    parent_count = state.current_mask.retained_count
    if parent_count <= comparator_count:
        raise ValueError("exact prefix rung is unnecessary after reaching the milestone")
    proposal = (
        acquisition.candidate
        if _is_label_native_acquisition(acquisition)
        else acquisition.queue[index]
    )
    if (
        proposal.resulting_mask.retained_count != comparator_count
        or proposal.removed_count != parent_count - comparator_count
    ):
        raise ValueError("registered proposal does not land exactly on the comparator count")
    if _is_label_native_acquisition(acquisition):
        plan = acquisition.candidate.runtime_plan_hash
        if state.label_native_runtime_plan is None or (
            state.label_native_runtime_plan.plan_hash != plan
        ):
            raise ValueError("exact prefix rung lacks the retained runtime plan")
        earlier_crosses = any(
            parent_count - descriptor.cumulative_removed_count <= comparator_count
            for descriptor in state.label_native_runtime_plan.descriptors[:index]
        )
    else:
        earlier_crosses = any(
            earlier.resulting_mask.retained_count <= comparator_count
            for earlier in acquisition.queue[:index]
        )
    if earlier_crosses:
        raise ValueError(
            "a crossing proposal appears before the registered exact prefix rung"
        )
    return V1ExactPrefixRungRegistration(
        campaign_contract_hash=state.binding.contract.campaign_contract_hash,
        milestone_registration_hash=registration.registration_hash,
        acquisition_receipt_hash=acquisition.receipt.receipt_hash,
        generation_index=acquisition.generation_index,
        parent_mask_hash=state.current_mask.mask_hash,
        parent_retained_route_count=parent_count,
        comparator_retained_route_count=comparator_count,
        required_removed_route_count=proposal.removed_count,
        queue_index=index,
        candidate_id=proposal.candidate_id,
        candidate_mask_hash=proposal.resulting_mask.mask_hash,
        removed_route_index_hash=proposal.removed_route_index_hash,
        v1_order_receipt_hash=_nonempty(v1_order_receipt_hash, "v1_order_receipt_hash"),
    )


def attach_v1_acquisition(
    state: V1ControllerState,
    acquisition: ModernV1Acquisition,
    *,
    exact_prefix_rung_registration: V1ExactPrefixRungRegistration | None = None,
) -> V1ControllerState:
    """Attach one exact fresh generation and expose its global queue."""

    if type(state) is not V1ControllerState or not _is_modern_v1_acquisition(
        acquisition
    ):
        raise TypeError("acquisition attachment requires exact controller/acquisition objects")
    if _is_label_native_acquisition(acquisition):
        raise PermissionError(
            "label-native single-endpoint acquisitions require the plan-bound attachment transaction"
        )
    if type(acquisition) is V1OperatorAcquisition:
        acquisition.assert_integrity()
        from .modern_v1_operator_acquisition import validate_operator_resource_target
        proposal=(acquisition._no_candidate_proposal if acquisition.candidate is None
                  else acquisition.candidate._producer.proposal)
        validate_operator_resource_target(controller=state,discovery=proposal.discovery)
        if (acquisition.receipt.controller_run_id!=state.controller_run_id
                or acquisition.receipt.source_controller_state_hash!=state.state_hash):
            raise ValueError('operator acquisition belongs to another controller state')
    if type(acquisition) is V1FQAuthorizedAcquisition and (
        acquisition.metadata.get(
            "canonical_fqr_required_before_calibration_or_oracle"
        )
        is True
        or acquisition.metadata.get("fq_only_candidate_queue_authority") is True
    ):
        raise PermissionError(
            "legacy F/Q-only acquisition cannot enter a canonical F/Q/R "
            "calibration-oracle trajectory"
        )
    if state.phase != "needs_acquisition":
        raise PermissionError("controller is not awaiting a fresh acquisition")
    if state.label_native_runtime_plan is not None:
        raise PermissionError(
            "a retained label-native plan cannot be replaced by a legacy eager acquisition"
        )
    if state.pending_pareto_milestone is not None:
        raise PermissionError(
            "Pareto milestone transition must cross the checkpoint seam before continuing"
        )
    if (
        acquisition.binding.binding_hash != state.binding.binding_hash
        or acquisition.generation_index != state.acquisition_generation_index
        or acquisition.current_mask.mask_hash != state.current_mask.mask_hash
        or acquisition.committed_mask_lineage_hashes != state.mask_lineage_hashes
        or acquisition.committed_mask_transition_receipt_hashes
        != state.transition_receipt_hashes
    ):
        raise ValueError("acquisition generation is stale or has alternate same-shaped ancestry")
    receipt_hash = acquisition.receipt.receipt_hash
    if (
        len(state.acquisition_receipt_hashes) != acquisition.generation_index
        or receipt_hash in set(state.acquisition_receipt_hashes)
    ):
        raise ValueError("acquisition generation history is duplicated or incomplete")
    acquisition_history = state.acquisition_receipt_hashes + (receipt_hash,)
    milestone_registration = state.pareto_milestone_registration
    if exact_prefix_rung_registration is not None and type(
        exact_prefix_rung_registration
    ) is not V1ExactPrefixRungRegistration:
        raise TypeError("exact prefix-rung registration has the wrong type")
    if milestone_registration is None:
        if exact_prefix_rung_registration is not None:
            raise ValueError("exact prefix rung supplied without a Pareto registration")
    elif state.pareto_milestone_receipt is None:
        comparator_count = milestone_registration.comparator_retained_route_count
        at_or_below = tuple(
            index
            for index, proposal in enumerate(acquisition.queue)
            if proposal.resulting_mask.retained_count <= comparator_count
        )
        if at_or_below:
            if exact_prefix_rung_registration is None:
                raise PermissionError(
                    "live ladder reaches or crosses k_cmp without a registered exact v1 prefix"
                )
            if exact_prefix_rung_registration.queue_index != at_or_below[0]:
                raise PermissionError(
                    "exact v1 prefix must precede every proposal that crosses k_cmp"
                )
            expected = register_v1_exact_pareto_prefix_rung(
                state,
                acquisition,
                queue_index=exact_prefix_rung_registration.queue_index,
                v1_order_receipt_hash=(
                    exact_prefix_rung_registration.v1_order_receipt_hash
                ),
            )
            if expected.registration_hash != exact_prefix_rung_registration.registration_hash:
                raise ValueError("exact prefix-rung registration is not reproducible")
        elif exact_prefix_rung_registration is not None:
            raise ValueError("exact prefix rung supplied for a ladder that stays above k_cmp")
    elif exact_prefix_rung_registration is not None:
        raise ValueError("exact prefix rung supplied after the milestone was sealed")
    if not acquisition.queue:
        return V1ControllerState(
            binding=state.binding,
            controller_run_id=state.controller_run_id,
            initial_mask=state.initial_mask,
            current_mask=state.current_mask,
            phase="stopped",
            accepted_trajectory=state.accepted_trajectory,
            accepted_oracle_trajectory=state.accepted_oracle_trajectory,
            attempt_receipts=state.attempt_receipts,
            transition_receipts=state.transition_receipts,
            acquisition_receipt_hashes=acquisition_history,
            globally_used_draw_ids=state.globally_used_draw_ids,
            pareto_milestone_registration=state.pareto_milestone_registration,
            pareto_milestone_receipt=state.pareto_milestone_receipt,
            stop_reason=("operator_no_admissible_candidate" if type(acquisition) is V1OperatorAcquisition
                         else "global_queue_exhausted"),
            logical_sequence=state.logical_sequence + 1,
        )
    return V1ControllerState(
        binding=state.binding,
        controller_run_id=state.controller_run_id,
        initial_mask=state.initial_mask,
        current_mask=state.current_mask,
        phase="queue_ready",
        acquisition=acquisition,
        queue_cursor=0,
        accepted_trajectory=state.accepted_trajectory,
        accepted_oracle_trajectory=state.accepted_oracle_trajectory,
        attempt_receipts=state.attempt_receipts,
        transition_receipts=state.transition_receipts,
        acquisition_receipt_hashes=acquisition_history,
        globally_used_draw_ids=state.globally_used_draw_ids,
        pareto_milestone_registration=state.pareto_milestone_registration,
        live_exact_prefix_rung_registration=exact_prefix_rung_registration,
        pareto_milestone_receipt=state.pareto_milestone_receipt,
        logical_sequence=state.logical_sequence + 1,
    )


def create_v1_step_sampling_law(
    *,
    binding: V1OptionACampaignBinding,
    population_context_ids: Sequence[str],
    target_population_probabilities: Sequence[float],
    sampling_population_probabilities: Sequence[float],
    source_receipt_hash: str,
    metadata: Mapping[str, Any] | None = None,
) -> V1StepSamplingLaw:
    """Bind a complete predictable p/q law to the immutable campaign."""

    contract = binding.contract
    sampling_masses = quantize_v1_staged_alpha_sampling_masses(
        sampling_population_probabilities
    )
    canonical_sampling = tuple(
        float(value / STAGED_ALPHA_SAMPLING_MASS_TOTAL)
        for value in sampling_masses
    )
    return V1StepSamplingLaw(
        campaign_contract_hash=contract.campaign_contract_hash,
        population_context_ids=tuple(population_context_ids),
        target_population_probabilities=tuple(target_population_probabilities),
        sampling_population_probabilities=canonical_sampling,
        sampling_masses_u64=sampling_masses,
        declared_target_population_hash=contract.target_population_hash,
        sampling_law_hash=contract.sampling_law_hash,
        support_construction_hash=contract.support_construction_hash,
        source_receipt_hash=source_receipt_hash,
        maximum_importance_weight=contract.maximum_importance_weight,
        metadata={} if metadata is None else metadata,
    )


def create_v1_candidate_threshold(
    state: V1ControllerState,
    *,
    maximum_mean_damage: float,
    threshold_source_receipt_hash: str,
    metadata: Mapping[str, Any] | None = None,
) -> V1CandidateDamageThreshold:
    """Freeze tau_t for the exact next global-queue candidate."""

    if type(state) is not V1ControllerState or state.phase != "queue_ready":
        raise PermissionError("candidate threshold can be created only at a ready queue cursor")
    acquisition = state.acquisition
    if acquisition is None:  # defensive: queue_ready state validation must imply acquisition
        raise ValueError("ready controller state is missing its acquisition")
    proposal = _selected_proposal(acquisition, queue_cursor=state.queue_cursor)
    _assert_v1_pareto_candidate_admissible(state, proposal)
    threshold_metadata: dict[str, Any] = {
        "proposal_parent_mask_hash": proposal.parent_mask_hash,
        "loss_referent_mask_hash": state.initial_mask.mask_hash,
        "loss_referent_role": "dense_initial_baseline_mask",
        "damage_currency": "direct_raw_bce_candidate_minus_dense_baseline",
        "budget_scope": "total_candidate_mask_increment_from_dense",
        "current_mask_is_relinearization_and_commit_parent_only": True,
    }
    if metadata is not None:
        threshold_metadata.update(dict(metadata))
    return V1CandidateDamageThreshold(
        campaign_contract_hash=state.binding.contract.campaign_contract_hash,
        acquisition_receipt_hash=acquisition.receipt.receipt_hash,
        candidate_id=proposal.candidate_id,
        parent_mask_hash=proposal.parent_mask_hash,
        candidate_mask_hash=proposal.resulting_mask.mask_hash,
        maximum_mean_damage=maximum_mean_damage,
        damage_target_derivation_hash=state.binding.contract.damage_target_derivation_hash,
        threshold_source_receipt_hash=threshold_source_receipt_hash,
        metadata=threshold_metadata,
    )


def _assert_v1_pareto_candidate_admissible(
    state: V1ControllerState,
    proposal: ModernV1Proposal,
) -> None:
    """Refuse an irreversible milestone crossing before calibration/oracle work."""

    if state.pending_pareto_milestone is not None:
        raise PermissionError("pending Pareto milestone must be checkpointed first")
    milestone_registration = state.pareto_milestone_registration
    if milestone_registration is None or state.pareto_milestone_receipt is not None:
        return
    comparator_count = milestone_registration.comparator_retained_route_count
    candidate_count = proposal.resulting_mask.retained_count
    if candidate_count < comparator_count:
        raise PermissionError(
            "irreversible proposal would cross k_cmp before its milestone checkpoint"
        )
    if candidate_count == comparator_count:
        rung = state.live_exact_prefix_rung_registration
        if (
            rung is None
            or rung.queue_index != state.queue_cursor
            or rung.candidate_id != proposal.candidate_id
        ):
            raise PermissionError(
                "proposal landing on k_cmp lacks its pre-oracle exact-prefix registration"
            )


def freeze_v1_candidate(
    state: V1ControllerState,
    *,
    threshold: V1CandidateDamageThreshold,
    sampling_law: V1StepSamplingLaw,
    candidate_calibration_freeze: V1CandidateCalibrationFreeze | None = None,
    paired_bce_registration: V1PairedBCEAcquisitionRegistration | None = None,
) -> V1ControllerState:
    """Freeze candidate, calibration, threshold, p/q, and filtration pre-outcome."""

    if type(state) is not V1ControllerState:
        raise TypeError("candidate freeze requires an exact controller state")
    if state.phase != "queue_ready" or state.acquisition is None:
        raise PermissionError("controller has no ready global candidate")
    contract = state.binding.contract
    if state.candidate_attempt_count >= contract.maximum_candidate_count:
        return stop_v1_controller(state, reason="maximum_candidate_count_reached")
    proposal = _selected_proposal(
        state.acquisition,
        queue_cursor=state.queue_cursor,
    )
    _assert_v1_pareto_candidate_admissible(state, proposal)
    if type(threshold) is not V1CandidateDamageThreshold or type(sampling_law) is not V1StepSamplingLaw:
        raise TypeError("candidate freeze requires exact threshold and sampling-law objects")
    if (
        threshold.campaign_contract_hash != contract.campaign_contract_hash
        or threshold.acquisition_receipt_hash != state.acquisition.receipt.receipt_hash
        or threshold.candidate_id != proposal.candidate_id
        or threshold.parent_mask_hash != state.current_mask.mask_hash
        or threshold.candidate_mask_hash != proposal.resulting_mask.mask_hash
        or threshold.damage_target_derivation_hash != contract.damage_target_derivation_hash
    ):
        raise ValueError("candidate threshold is stale or names another proposal")
    if (
        sampling_law.campaign_contract_hash != contract.campaign_contract_hash
        or sampling_law.declared_target_population_hash != contract.target_population_hash
        or sampling_law.sampling_law_hash != contract.sampling_law_hash
        or sampling_law.support_construction_hash != contract.support_construction_hash
        or not math.isclose(
            sampling_law.maximum_importance_weight,
            contract.maximum_importance_weight,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("step sampling law differs from the campaign contract")
    projected_removed = state.cumulative_removed_count + proposal.removed_count
    if projected_removed > contract.maximum_removed_count:
        raise ValueError("next candidate would exceed the campaign removal cap")
    raw_loss_referent = threshold.metadata.get("loss_referent_mask_hash")
    loss_referent_mask_hash = _nonempty(
        raw_loss_referent
        if raw_loss_referent is not None
        else state.initial_mask.mask_hash,
        "loss_referent_mask_hash",
    )
    if loss_referent_mask_hash != state.initial_mask.mask_hash:
        raise ValueError("modern-v1 loss referent must be the dense initial mask")
    if type(proposal) in {
        V1OperatorCandidate,
        V1FQRCandidateProposal,
        V1SignedWorkCandidateProposal,
        V1LabelNativeControllerCandidateCarrier,
    } and (
        candidate_calibration_freeze is None or paired_bce_registration is None
    ):
        raise PermissionError(
            "canonical F/Q/R and signed-work proposals require candidate calibration and the "
            "paired-BCE oracle; the legacy staged-alpha commit path is forbidden"
        )
    if candidate_calibration_freeze is not None:
        if type(candidate_calibration_freeze) is not V1CandidateCalibrationFreeze:
            raise TypeError("candidate calibration freeze has the wrong type")
        if (
            candidate_calibration_freeze.campaign_contract_hash
            != contract.campaign_contract_hash
            or candidate_calibration_freeze.acquisition_receipt_hash
            != state.acquisition.receipt.receipt_hash
            or candidate_calibration_freeze.queue_index != state.queue_cursor
            or candidate_calibration_freeze.candidate_id != proposal.candidate_id
            or candidate_calibration_freeze.proposal_parent_mask_hash
            != proposal.parent_mask_hash
            or candidate_calibration_freeze.dense_referent_mask_hash
            != loss_referent_mask_hash
            or candidate_calibration_freeze.candidate_mask_hash
            != proposal.resulting_mask.mask_hash
            or candidate_calibration_freeze.calibration_policy_id
            != contract.calibration_identity
        ):
            raise ValueError("candidate calibration freeze is stale or names another proposal")
    if paired_bce_registration is not None:
        if type(paired_bce_registration) is not V1PairedBCEAcquisitionRegistration:
            raise TypeError("paired-BCE registration has the wrong type")
        if candidate_calibration_freeze is None:
            raise ValueError("paired-BCE oracle requires a candidate calibration freeze")
        paired_bce_registration.assert_integrity()
        if (
            paired_bce_registration.campaign_contract_hash
            != contract.campaign_contract_hash
            or paired_bce_registration.candidate_mask_hash
            != proposal.resulting_mask.mask_hash
            or paired_bce_registration.dense_reference_hash
            != v1_dense_referent_mask_hash(contract)
            or paired_bce_registration.candidate_calibration_freeze_hash
            != candidate_calibration_freeze.freeze_hash
            or paired_bce_registration.candidate_attempt_index
            != state.candidate_attempt_count
            or paired_bce_registration.maximum_candidate_attempts
            != contract.maximum_candidate_count
            or not math.isclose(
                paired_bce_registration.loss_ceiling,
                threshold.maximum_mean_damage,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError("paired-BCE registration is stale or names another candidate")
        dynamic = paired_bce_registration.dynamic_oracle_receipt
        if dynamic is not None and (
            dynamic.generation_index != state.acquisition.generation_index
            or dynamic.dense_affine_calibration_hash
            != candidate_calibration_freeze.dense_calibration.frozen_calibration_hash
            or dynamic.candidate_affine_calibration_hash
            != candidate_calibration_freeze.candidate_calibration.frozen_calibration_hash
            or dynamic.dense_reference_hash
            != candidate_calibration_freeze.dense_referent_mask_hash
            or dynamic.candidate_mask_hash
            != candidate_calibration_freeze.candidate_mask_hash
        ):
            raise ValueError(
                "paired-BCE dynamic O frame differs from the frozen affine pair"
            )
    calibration_freeze_hash = (
        None
        if candidate_calibration_freeze is None
        else candidate_calibration_freeze.freeze_hash
    )
    snapshot_hash = state.snapshot_receipt().receipt_hash
    candidate_definition_payload: dict[str, object]
    if type(proposal) is V1OperatorCandidate:
        candidate_definition_payload=proposal.definition_payload()
        candidate_definition_hash=content_hash(candidate_definition_payload,
            prefix='partition-free-v1-operator-candidate-definition-')
    elif type(proposal) is V1LabelNativeControllerCandidateCarrier:
        descriptor = proposal.descriptor
        candidate_definition_payload = {
            "schema": "partition_free_v1_label_native_controller_candidate_definition_v1",
            "candidate_id": proposal.candidate_id,
            "runtime_plan_hash": proposal.runtime_plan_hash,
            "runtime_descriptor_hash": descriptor.descriptor_hash,
            "selected_materialization_receipt_hash": (
                proposal.selected_endpoint.receipt.receipt_hash
            ),
            "selected_coordinate_hash": descriptor.coordinate_hash,
            "certificate_hash": descriptor.certificate_hash,
            "crosswalk_receipt_hash": descriptor.crosswalk.receipt_hash,
            "source_action_receipt_hash": descriptor.source_action_receipt_hash,
            "qualified_lane_registration_hash": (
                descriptor.qualified_lane_registration_hash
            ),
            "parent_mask_hash": proposal.parent_mask_hash,
            "candidate_mask_hash": proposal.resulting_mask.mask_hash,
            "removed_route_count": proposal.removed_count,
            "removed_route_index_hash": proposal.removed_route_index_hash,
            "plan_queue_index": proposal.plan_queue_index,
            "queue_descriptor_count": proposal.queue_descriptor_count,
            "packed_action_hash": proposal.packed_action_hash,
            "six_role_C_design_hash": descriptor.six_role_C_design_hash,
            "six_role_C_progress_receipt_hash": (
                descriptor.six_role_C_progress_receipt_hash
            ),
            "six_role_certification_receipt_hash": (
                descriptor.six_role_certification_receipt_hash
            ),
            "six_role_decision_hash": descriptor.six_role_decision_hash,
            "complete_resolved_prefix_frozen_before_K_O": True,
            "exactly_one_endpoint_materialized": True,
            "unselected_fallback_masks_materialized": False,
            "controller_freeze_authority_from_carrier": False,
            "controller_freeze_authority_from_controller_checks": True,
            "paired_bce_oracle_required": True,
            "oracle_conditioned_composition": False,
            "additive_surrogate_used": False,
        }
        candidate_definition_hash = content_hash(
            candidate_definition_payload,
            prefix="partition-free-v1-label-native-controller-candidate-definition-",
        )
    elif type(proposal) is V1SignedWorkCandidateProposal:
        numeric = proposal.numeric_state
        candidate_definition_payload = {
            "schema": "partition_free_v1_signed_work_candidate_definition_v1",
            "candidate_id": proposal.candidate_id,
            "evaluation_receipt_hash": proposal.evaluation_receipt_hash,
            "numeric_state_hash": proposal.numeric_state_hash,
            "domain_hash": proposal.domain_hash,
            "certification_scope_hash": proposal.certification_scope_hash,
            "parent_mask_hash": proposal.parent_mask_hash,
            "candidate_mask_hash": proposal.resulting_mask.mask_hash,
            "removed_route_count": proposal.removed_count,
            "removed_route_index_hash": proposal.removed_route_index_hash,
            "queue_index": numeric.queue_index,
            "representative_branch": numeric.representative_branch,
            "representative_endpoint_index": (
                numeric.representative_endpoint_index
            ),
            "representative_frozen_action_hash": (
                numeric.representative_frozen_action_hash
            ),
            "packed_action_hash": numeric.packed_action_hash,
            "resource_receipt_hash": numeric.resource_receipt_hash,
            "six_role_interval_hash": numeric.six_role_interval_hash,
            "six_role_authorization_hash": numeric.six_role_authorization_hash,
            "six_role_certification_receipt_hash": (
                numeric.six_role_certification_receipt_hash
            ),
            "six_role_decision_hash": numeric.six_role_decision_hash,
            "six_role_decision_design_hash": numeric.six_role_decision_design_hash,
            "six_role_C_design_hash": numeric.six_role_C_design_hash,
            "six_role_C_progress_receipt_hash": (
                numeric.six_role_C_progress_receipt_hash
            ),
            "selected_coordinate_hash": numeric.selected_coordinate_hash,
            "admission_basis": numeric.admission_basis,
            "chart_transport_status": numeric.chart_transport_status,
            "direct_D_owns_finite_response_admission": True,
            "QR_transport_is_label_not_hidden_C_veto": True,
            "complete_queue_frozen_before_K_O": True,
            "paired_bce_oracle_required": True,
            "oracle_conditioned_composition": False,
            "additive_surrogate_used": False,
        }
        candidate_definition_hash = content_hash(
            candidate_definition_payload,
            prefix="partition-free-v1-signed-work-candidate-definition-",
        )
    elif type(proposal) is V1FQRCandidateProposal:
        numeric = proposal.numeric_state
        candidate_definition_payload = {
            "schema": "partition_free_v1_fqr_candidate_definition_v1",
            "candidate_id": proposal.candidate_id,
            "evaluation_receipt_hash": proposal.evaluation_receipt_hash,
            "numeric_state_hash": proposal.numeric_state_hash,
            "domain_hash": proposal.domain_hash,
            "certification_scope_hash": proposal.certification_scope_hash,
            "parent_mask_hash": proposal.parent_mask_hash,
            "candidate_mask_hash": proposal.resulting_mask.mask_hash,
            "removed_route_count": proposal.removed_count,
            "removed_route_index_hash": proposal.removed_route_index_hash,
            "queue_index": numeric.queue_index,
            "representative_branch": numeric.representative_branch,
            "representative_endpoint_index": (
                numeric.representative_endpoint_index
            ),
            "representative_frozen_action_hash": (
                numeric.representative_frozen_action_hash
            ),
            "packed_action_hash": numeric.packed_action_hash,
            "resource_receipt_hash": numeric.resource_receipt_hash,
            "fqr_interval_hash": numeric.fqr_interval_hash,
            "branch_family_authorization_hash": (
                numeric.branch_family_authorization_hash
            ),
            "branch_family_entry_hash": numeric.branch_family_entry_hash,
            "prefrozen_family_hash": numeric.prefrozen_family_hash,
            "joint_fqr_family_registration_hash": (
                numeric.joint_fqr_family_registration_hash
            ),
            "source_branch_evidence_hashes": (
                numeric.source_branch_evidence_hashes
            ),
            "source_alias_hashes": tuple(
                value.alias_hash for value in numeric.source_aliases
            ),
            "complete_queue_frozen_before_K_O": True,
            "paired_bce_oracle_required": True,
            "oracle_conditioned_composition": False,
            "additive_surrogate_used": False,
        }
        candidate_definition_hash = content_hash(
            candidate_definition_payload,
            prefix="partition-free-v1-fqr-candidate-definition-",
        )
    elif type(proposal) is V1FQCandidateProposal:
        candidate_definition_payload = {
            "schema": "partition_free_v1_fq_candidate_definition_v1",
            "candidate_id": proposal.candidate_id,
            "evaluation_receipt_hash": proposal.evaluation_receipt_hash,
            "numeric_state_hash": proposal.numeric_state_hash,
            "domain_hash": proposal.domain_hash,
            "constraints_hash": proposal.constraints_hash,
            "parent_mask_hash": proposal.parent_mask_hash,
            "candidate_mask_hash": proposal.resulting_mask.mask_hash,
            "removed_route_count": proposal.removed_count,
            "removed_route_index_hash": proposal.removed_route_index_hash,
            "endpoint_index": proposal.endpoint_index,
            "action_hash": proposal.action_hash,
            "resource_gain_receipt_hash": proposal.resource_gain_receipt_hash,
            "atom_cost_source_hash": proposal.atom_cost_source_hash,
            "fq_interval_hash": proposal.fq_interval_hash,
            "fq_budget_hash": proposal.fq_budget_hash,
            "fq_certification_receipt_hash": (
                proposal.numeric_state.fq_certification_receipt_hash
            ),
            "decisive_fq_commit_hash": proposal.decisive_fq_commit_hash,
            "fq_queue_authorization_hash": proposal.fq_queue_authorization_hash,
            "sampled_diagnostic_lineage_hash": (
                proposal.sampled_diagnostic_lineage_hash
            ),
            "exact_registered_surrogate_evaluation": False,
            "additive_surrogate_used": False,
        }
        candidate_definition_hash = content_hash(
            candidate_definition_payload,
            prefix="partition-free-v1-fq-candidate-definition-",
        )
    else:
        candidate_definition_payload = {
            "candidate_id": proposal.candidate_id,
            "evaluation_receipt_hash": proposal.evaluation_receipt_hash,
            "numeric_state_hash": proposal.numeric_state_hash,
            "domain_hash": proposal.domain_hash,
            "constraints_hash": proposal.constraints_hash,
            "parent_mask_hash": proposal.parent_mask_hash,
            "candidate_mask_hash": proposal.resulting_mask.mask_hash,
        }
        if proposal.compact_removed_indices_available:
            candidate_definition_payload["removed_route_indices"] = (
                proposal.removed_routes.to_tuple()  # type: ignore[union-attr]
            )
        else:
            candidate_definition_payload.update(
                {
                    "schema": "partition_free_v1_candidate_definition_v2",
                    "removed_route_count": proposal.removed_count,
                    "removed_route_index_hash": proposal.removed_route_index_hash,
                }
            )
        candidate_definition_hash = content_hash(
            candidate_definition_payload,
            prefix="partition-free-v1-candidate-definition-",
        )
    filtration_hash = content_hash(
        {
            "controller_snapshot_hash": snapshot_hash,
            "acquisition_receipt_hash": state.acquisition.receipt.receipt_hash,
            "queue_index": state.queue_cursor,
            "candidate_id": proposal.candidate_id,
            "threshold_hash": threshold.threshold_hash,
            "sampling_state_hash": sampling_law.sampling_state_hash,
            "candidate_calibration_freeze_hash": calibration_freeze_hash,
            "globally_used_draw_ids_hash": content_hash(
                state.globally_used_draw_ids,
                prefix="partition-free-v1-globally-used-draw-ids-",
            ),
        },
        prefix="partition-free-v1-pre-draw-filtration-",
    )
    calibration_metadata: dict[str, Any]
    if candidate_calibration_freeze is None:
        calibration_metadata = {
            "candidate_calibration_freeze_required": False,
            "candidate_calibration_freeze_evidence_ceiling": (
                "legacy_policy_identity_only_no_numeric_candidate_freeze"
            ),
        }
    else:
        calibration_metadata = {
            "candidate_calibration_freeze_required": True,
            _CALIBRATION_FREEZE_HASH_METADATA_KEY: calibration_freeze_hash,
            _CALIBRATION_FREEZE_METADATA_KEY: (
                candidate_calibration_freeze.to_payload()
            ),
            "candidate_calibration_freeze_evidence_ceiling": (
                "exact_validation_independent_affine_state_and_fit_receipt_bound_pre_oracle"
            ),
            "calibration_policy_id": (
                candidate_calibration_freeze.calibration_policy_id
            ),
        }
    certificate_freeze = V1CertificateFreeze(
        registration_hash=state.binding.registration.registration_hash,
        parent_mask_hash=loss_referent_mask_hash,
        candidate_mask_hash=proposal.resulting_mask.mask_hash,
        candidate_definition_hash=candidate_definition_hash,
        null_id=_NULL_ID,
        loss_id=contract.direct_increment_target_id,
        sampling_law_hash=contract.sampling_law_hash,
        importance_weight_hash=sampling_law.importance_weight_hash,
        certificate_factor_hash=content_hash(
            {
                "coordinates_hash": state.binding.coordinates.coordinates_hash,
                "proof_artifact_hash": contract.proof_artifact_hash,
                "student_t_evidence_disclosure_hash": (
                    contract.student_t_evidence_disclosure_hash
                ),
            },
            prefix="partition-free-v1-certificate-factor-",
        ),
        pre_draw_filtration_hash=filtration_hash,
        metadata={
            "campaign_contract_hash": contract.campaign_contract_hash,
            "acquisition_receipt_hash": state.acquisition.receipt.receipt_hash,
            "candidate_threshold_hash": threshold.threshold_hash,
            "sampling_state_hash": sampling_law.sampling_state_hash,
            "maximum_mean_damage": threshold.maximum_mean_damage,
            "direct_increment_orientation_id": contract.increment_orientation_id,
            "proposal_parent_mask_hash": proposal.parent_mask_hash,
            "loss_referent_mask_hash": loss_referent_mask_hash,
            "loss_referent_role": "dense_initial_baseline_mask",
            "budget_scope": "total_candidate_mask_increment_from_dense",
            "proposal_only_source": True,
            **calibration_metadata,
        },
    )
    step_freeze = bind_v1_staged_alpha_step(
        registration=state.binding.registration,
        coordinates=state.binding.coordinates,
        certificate_freeze=certificate_freeze,
        step_index=state.accepted_step_count,
        candidate_attempt_index=state.candidate_attempt_count,
        maximum_mean_damage=threshold.maximum_mean_damage,
        cumulative_draw_counts=contract.cumulative_draw_counts,
        schedule_construction_hash=contract.look_schedule_policy_hash,
    )
    active = V1ActiveCandidate(
        acquisition_receipt_hash=state.acquisition.receipt.receipt_hash,
        queue_index=state.queue_cursor,
        proposal=proposal,
        threshold=threshold,
        sampling_law=sampling_law,
        step_freeze=step_freeze,
        ledger=V1StagedAlphaLedger(step_freeze),
        paired_bce_registration=paired_bce_registration,
        paired_bce_state=(
            None
            if paired_bce_registration is None
            else initial_paired_bce_acquisition_state(paired_bce_registration)
        ),
    )
    return replace(
        state,
        phase="candidate_active",
        active_candidate=active,
        logical_sequence=state.logical_sequence + 1,
    )


def _current_generation_full_look_infeasible(
    state: V1ControllerState,
    attempts: tuple[V1CandidateAttemptReceipt, ...],
) -> bool:
    """Whether every current-generation queued action has a full-look reject."""

    acquisition = state.acquisition
    if acquisition is None or not acquisition.queue:
        return False
    current = tuple(
        attempt
        for attempt in attempts
        if attempt.acquisition_receipt_hash == acquisition.receipt.receipt_hash
    )
    final_look_index = len(state.binding.contract.cumulative_draw_counts) - 1
    return (
        len(current) == len(acquisition.queue)
        and tuple(attempt.queue_index for attempt in current)
        == tuple(range(len(acquisition.queue)))
        and all(
            attempt.action == "reject_candidate"
            and attempt.metadata.get("look_index") == final_look_index
            for attempt in current
        )
    )


def _paired_fqr_barrier_attempt_history_is_saturated(
    attempts: tuple[V1CandidateAttemptReceipt, ...],
) -> bool:
    """Whether one complete frozen F/Q/R queue has decisive paired-O rejects.

    This predicate deliberately excludes calibration-horizon skips and generic
    queue exhaustion.  A partial paired-BCE look is sufficient only when its
    registered lower confidence bound is strictly above the frozen ceiling;
    a census decision must carry the exact-finite-set flag and a mean strictly
    above that same ceiling.
    """

    if not attempts:
        return False
    terminal_receipt_hash = attempts[-1].acquisition_receipt_hash
    current = tuple(
        attempt
        for attempt in attempts
        if attempt.acquisition_receipt_hash == terminal_receipt_hash
    )
    raw_queue_length = attempts[-1].metadata.get(
        "canonical_fqr_complete_queue_length"
    )
    if (
        isinstance(raw_queue_length, bool)
        or not isinstance(raw_queue_length, Integral)
        or int(raw_queue_length) <= 0
    ):
        return False
    queue_length = int(raw_queue_length)
    if (
        len(current) != queue_length
        or current != attempts[-queue_length:]
        or tuple(attempt.queue_index for attempt in current)
        != tuple(range(queue_length))
    ):
        return False

    axis_hash = current[0].metadata.get(
        "canonical_fqr_ordered_candidate_ids_hash"
    )
    composition_hash = current[0].metadata.get(
        "canonical_fqr_composition_hash"
    )
    branch_family_hash = current[0].metadata.get(
        "candidate_fqr_branch_family_authorization_hash"
    )
    joint_family_hash = current[0].metadata.get(
        "candidate_fqr_joint_family_registration_hash"
    )
    if any(
        type(value) is not str or not value
        for value in (
            axis_hash,
            composition_hash,
            branch_family_hash,
            joint_family_hash,
        )
    ):
        return False

    rejection_reasons = {
        "anytime_betting_lower_above_ceiling",
        "registered_serfling_lower_above_ceiling",
        "exact_joint_census_mean_above_ceiling",
    }
    for attempt in current:
        metadata = attempt.metadata
        lower_raw = metadata.get("paired_bce_weighted_confidence_lower")
        mean_raw = metadata.get("paired_bce_weighted_point_mean")
        ceiling_raw = metadata.get("paired_bce_loss_ceiling")
        if any(isinstance(value, bool) for value in (lower_raw, mean_raw, ceiling_raw)):
            return False
        try:
            lower = float(lower_raw)
            point_mean = float(mean_raw)
            ceiling = float(ceiling_raw)
        except (TypeError, ValueError, OverflowError):
            return False
        if (
            not all(math.isfinite(value) for value in (lower, point_mean, ceiling))
            or lower <= ceiling
            or attempt.action != "reject_candidate"
            or attempt.reason not in rejection_reasons
            or metadata.get("oracle_protocol") != "paired_bce"
            or metadata.get(
                "canonical_fqr_cross_branch_queue_preserved_without_filter_or_reorder"
            )
            is not True
            or metadata.get(
                "complete_queue_frozen_before_calibration_K_and_paired_oracle_O"
            )
            is not True
            or metadata.get("canonical_fqr_complete_queue_length") != queue_length
            or metadata.get("canonical_fqr_ordered_candidate_ids_hash") != axis_hash
            or metadata.get("canonical_fqr_composition_hash") != composition_hash
            or metadata.get("candidate_fqr_branch_family_authorization_hash")
            != branch_family_hash
            or metadata.get("candidate_fqr_joint_family_registration_hash")
            != joint_family_hash
            or metadata.get("candidate_paired_bce_oracle_required") is not True
            or metadata.get("oracle_conditioned_composition") is not False
            or metadata.get("paired_bce_core_mask_commit_authority") is not False
        ):
            return False
        if metadata.get("paired_bce_full_census") is True:
            if (
                metadata.get("paired_bce_exact_finite_set_decision") is not True
                or point_mean <= ceiling
            ):
                return False
        elif metadata.get("paired_bce_partial_confidence_claim_bearing") is not True:
            return False
    return True


def _current_generation_paired_fqr_barrier_is_saturated(
    state: V1ControllerState,
    attempts: tuple[V1CandidateAttemptReceipt, ...],
) -> bool:
    """Bind the historical saturation proof to the live exact F/Q/R queue."""

    acquisition = state.acquisition
    if (
        type(acquisition) is not V1FQRAuthorizedAcquisition
        or not acquisition.queue
        or _fqr_queue_has_unresolved_proposal_only_d_origin(acquisition)
        or not _paired_fqr_barrier_attempt_history_is_saturated(attempts)
    ):
        return False
    current = attempts[-len(acquisition.queue) :]
    expected_axis_hash = content_hash(
        acquisition.receipt.ordered_candidate_ids,
        prefix="partition-free-v1-fqr-controller-ordered-candidate-axis-",
    )
    if (
        current[0].metadata.get("canonical_fqr_complete_queue_length")
        != len(acquisition.queue)
        or current[0].metadata.get("canonical_fqr_ordered_candidate_ids_hash")
        != expected_axis_hash
        or current[0].metadata.get("canonical_fqr_composition_hash")
        != acquisition.composition.result_hash
    ):
        return False
    return all(
        attempt.acquisition_receipt_hash == acquisition.receipt.receipt_hash
        and attempt.queue_index == queue_index
        and attempt.candidate_id == proposal.candidate_id
        and attempt.parent_mask_hash == acquisition.current_mask.mask_hash
        and attempt.candidate_mask_hash == proposal.resulting_mask.mask_hash
        and attempt.metadata.get("candidate_fqr_branch_family_entry_hash")
        == proposal.numeric_state.branch_family_entry_hash
        and attempt.metadata.get("candidate_fqr_interval_hash")
        == proposal.numeric_state.fqr_interval_hash
        for queue_index, (attempt, proposal) in enumerate(
            zip(current, acquisition.queue, strict=True)
        )
    )


def _fqr_queue_has_unresolved_proposal_only_d_origin(
    acquisition: V1FQRAuthorizedAcquisition,
) -> bool:
    """Replay the v4 D-origin firewall carried by acquisition identity.

    A rank-identifiable D prefix is sufficient to *propose* a KKT/C ladder,
    but its remaining D sampling uncertainty may change that ladder's
    identity.  Rejecting every member therefore cannot prove that every
    admissible next action was rejected.  The acquisition metadata is part of
    ``acquisition_hash`` and its generation receipt, so this check survives
    restart without trusting a mutable side channel.

    There is intentionally no string-only escape hatch.  The predicate returns
    ``False`` for a v4-origin queue only after replaying the exact typed D
    decision-stability certificate embedded in the acquisition identity.
    """

    if type(acquisition) is not V1FQRAuthorizedAcquisition:
        raise TypeError("F/Q/R D-origin firewall requires an exact acquisition")
    metadata = acquisition.metadata
    marker_names = (
        "D_proposal_handoff_receipt_hash",
        "D_sampling_uncertainty_may_change_proposal_identity",
        "D_score_order_unresolved",
        "D_relaxed_or_hard_optimality_claim",
        "scientific_terminal_requires_separate_decision_stability_or_exhaustion_certificate",
        "decision_stability_or_exhaustion_certificate_hash",
        "scientific_terminal_endpoint_authority",
        "pareto_saturation_authority",
    )
    present = tuple(name for name in marker_names if name in metadata)
    if not present:
        return False
    if len(present) != len(marker_names):
        raise PermissionError(
            "F/Q/R acquisition has a partial proposal-only D-origin firewall"
        )
    receipt_hash = metadata.get("D_proposal_handoff_receipt_hash")
    if type(receipt_hash) is not str or not receipt_hash:
        raise PermissionError(
            "F/Q/R acquisition changed its proposal-only D-origin authority"
        )
    certificate_hash = metadata.get(
        "decision_stability_or_exhaustion_certificate_hash"
    )
    certificate_payload = metadata.get(
        "decision_stability_or_exhaustion_certificate"
    )
    refusal_payload = metadata.get(
        "decision_stability_or_exhaustion_refusal_receipt"
    )
    refusal_hash = metadata.get(
        "decision_stability_or_exhaustion_refusal_receipt_hash"
    )
    refusal_reason = metadata.get(
        "decision_stability_or_exhaustion_refusal_reason"
    )
    conditional_terminal_authority = metadata.get(
        "terminal_saturation_after_complete_downstream_family_authority"
    )
    if certificate_hash is None:
        if (
            certificate_payload is not None
            or metadata.get("D_sampling_uncertainty_may_change_proposal_identity")
            is not True
            or metadata.get("D_score_order_unresolved") is not True
            or metadata.get("D_relaxed_or_hard_optimality_claim") is not False
            or metadata.get(
                "scientific_terminal_requires_separate_decision_stability_or_exhaustion_certificate"
            )
            is not True
            or metadata.get("scientific_terminal_endpoint_authority") is not False
            or metadata.get("pareto_saturation_authority") is not False
            or (
                conditional_terminal_authority is not None
                and conditional_terminal_authority is not False
            )
        ):
            raise PermissionError(
                "F/Q/R acquisition changed its proposal-only D-origin authority"
            )
        refusal_fields = (refusal_payload, refusal_hash, refusal_reason)
        if any(value is not None for value in refusal_fields):
            if (
                not all(value is not None for value in refusal_fields)
                or not isinstance(refusal_payload, Mapping)
            ):
                raise PermissionError(
                    "F/Q/R acquisition has a partial D-stability refusal receipt"
                )
            refusal = V1FQRDDecisionStabilityRefusalReceipt.from_payload(
                refusal_payload
            )
            if (
                refusal.receipt_hash != refusal_hash
                or refusal.refusal_reason != refusal_reason
                or refusal.proposal_handoff_receipt_hash != receipt_hash
                or refusal.current_mask_hash != acquisition.current_mask.mask_hash
                or refusal.proposal_generation != acquisition.generation_index
            ):
                raise PermissionError(
                    "F/Q/R acquisition changed its D-stability refusal identity"
                )
        return True
    if (
        type(certificate_hash) is not str
        or not certificate_hash
        or not isinstance(certificate_payload, Mapping)
        or any(value is not None for value in (refusal_payload, refusal_hash, refusal_reason))
    ):
        raise PermissionError(
            "F/Q/R acquisition has a partial D decision-stability certificate"
        )
    certificate = V1FQRDDecisionStabilityCertificate.from_payload(
        certificate_payload
    )
    if (
        certificate.certificate_hash != certificate_hash
        or certificate.proposal_handoff_receipt_hash != receipt_hash
        or certificate.current_mask_hash != acquisition.current_mask.mask_hash
        or certificate.route_axis_hash != acquisition.current_mask.route_axis_hash
        or certificate.proposal_generation != acquisition.generation_index
        or certificate.candidate_ladder_identity_hash
        != acquisition.composition.prefrozen_family_hash
        or metadata.get("D_sampling_uncertainty_may_change_proposal_identity")
        is not False
        or metadata.get("D_score_order_unresolved") is not False
        or metadata.get("D_relaxed_or_hard_optimality_claim") is not False
        or metadata.get(
            "scientific_terminal_requires_separate_decision_stability_or_exhaustion_certificate"
        )
        is not False
        or conditional_terminal_authority is not True
        or metadata.get("scientific_terminal_endpoint_authority") is not False
        or metadata.get("pareto_saturation_authority") is not False
    ):
        raise PermissionError(
            "F/Q/R acquisition changed its typed D decision-stability authority"
        )
    return False
def update_v1_controller_paired_bce_state(
    state: V1ControllerState,
    paired_bce_state: V1PairedBCEAcquisitionState,
) -> V1ControllerState:
    """Persist compact paired-BCE acquisition progress inside the active candidate."""

    if type(state) is not V1ControllerState:
        raise TypeError("paired-BCE progress update requires an exact controller state")
    if type(paired_bce_state) is not V1PairedBCEAcquisitionState:
        raise TypeError("paired-BCE progress update requires an exact compact state")
    active = state.active_candidate
    if state.phase != "candidate_active" or active is None:
        raise PermissionError("controller has no active candidate paired-BCE state")
    registration = active.paired_bce_registration
    previous = active.paired_bce_state
    if registration is None or previous is None:
        raise PermissionError("active candidate is not registered for paired-BCE oracle")
    paired_bce_state.assert_integrity()
    if paired_bce_state.registration_hash != registration.registration_hash:
        raise ValueError("paired-BCE progress names another registration")
    if (
        paired_bce_state.dynamic_oracle_receipt_hash
        != registration.dynamic_oracle_receipt_hash
        or paired_bce_state.dynamic_oracle_receipt_hash
        != previous.dynamic_oracle_receipt_hash
    ):
        raise ValueError("paired-BCE progress attempted to replace its opened O frame")
    if paired_bce_state.chunk_count < previous.chunk_count or any(
        new < old
        for new, old in zip(
            paired_bce_state.stratum_counts,
            previous.stratum_counts,
            strict=True,
        )
    ):
        raise ValueError("paired-BCE progress cannot move backwards")
    return replace(
        state,
        active_candidate=replace(active, paired_bce_state=paired_bce_state),
        logical_sequence=state.logical_sequence + 1,
    )


def _paired_bce_terminal_lineage_ids(
    state: V1PairedBCEAcquisitionState,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    draw_id = content_hash(
        {
            "schema": "partition_free_v1_paired_bce_compact_attempt_draw_id_v1",
            "state_hash": state.state_hash,
            "receipt_chain_hash": state.receipt_chain_hash,
            "last_chunk_receipt_hash": state.last_chunk_receipt_hash,
            "chunk_count": state.chunk_count,
        },
        prefix="partition-free-v1-paired-bce-attempt-draw-",
    )
    source_id = content_hash(
        {
            "schema": "partition_free_v1_paired_bce_external_source_id_v1",
            "state_hash": state.state_hash,
            "receipt_chain_hash": state.receipt_chain_hash,
        },
        prefix="partition-free-v1-paired-bce-source-",
    )
    return (draw_id,), (source_id,)


def _paired_bce_evidence_ledger_hash(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
    decision: V1PairedBCEDecision,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_paired_bce_evidence_ledger_v1",
            "registration_hash": registration.registration_hash,
            "state_hash": state.state_hash,
            "decision_hash": decision.decision_hash,
            "receipt_chain_hash": state.receipt_chain_hash,
            "last_chunk_receipt_hash": state.last_chunk_receipt_hash,
            "chunk_count": state.chunk_count,
        },
        prefix="partition-free-v1-paired-bce-evidence-ledger-",
    )


def _paired_bce_attempt_metadata(
    *,
    registration: V1PairedBCEAcquisitionRegistration,
    state: V1PairedBCEAcquisitionState,
    decision: V1PairedBCEDecision,
) -> dict[str, object]:
    role_separation = registration.oracle_role_separation_receipt
    dynamic = registration.dynamic_oracle_receipt
    return {
        "oracle_protocol": "paired_bce",
        "paired_bce_registration_hash": registration.registration_hash,
        "paired_bce_state_hash": state.state_hash,
        "paired_bce_decision_hash": decision.decision_hash,
        "paired_bce_receipt_chain_hash": state.receipt_chain_hash,
        "paired_bce_last_chunk_receipt_hash": state.last_chunk_receipt_hash,
        "paired_bce_completed_chunk_count": state.chunk_count,
        "paired_bce_completed_look_count": state.completed_look_count,
        "paired_bce_stratum_counts": state.stratum_counts,
        "paired_bce_full_population_accounted_stratum_counts": (
            decision.stratum_counts
        ),
        "paired_bce_calibration_prefix_stratum_counts": (
            decision.calibration_prefix_stratum_counts
        ),
        "paired_bce_residual_oracle_stratum_counts": (
            decision.residual_oracle_stratum_counts
        ),
        "paired_bce_residual_oracle_population_stratum_counts": (
            decision.residual_oracle_population_stratum_counts
        ),
        "paired_bce_population_stratum_counts": (
            decision.population_stratum_counts
        ),
        "paired_bce_stratum_point_means": decision.stratum_point_means,
        "paired_bce_stratum_confidence_intervals": (
            decision.stratum_confidence_intervals
        ),
        "paired_bce_weighted_point_mean": decision.weighted_point_mean,
        "paired_bce_weighted_confidence_lower": (
            decision.weighted_confidence_lower
        ),
        "paired_bce_weighted_confidence_upper": (
            decision.weighted_confidence_upper
        ),
        "paired_bce_loss_ceiling": decision.loss_ceiling,
        "paired_bce_loss_ceiling_identity_hash": (
            registration.loss_ceiling_identity_hash
        ),
        "paired_bce_evaluated_registered_look_index": (
            decision.evaluated_registered_look_index
        ),
        "paired_bce_look_alpha": decision.look_alpha,
        "paired_bce_cumulative_registered_alpha_through_look": (
            decision.cumulative_registered_alpha_through_look
        ),
        "paired_bce_full_census": decision.full_census,
        "paired_bce_exact_finite_set_decision": (
            decision.exact_finite_set_decision
        ),
        "paired_bce_partial_confidence_claim_bearing": (
            decision.partial_confidence_claim_bearing
        ),
        "paired_bce_partial_evidence_method": decision.partial_evidence_method,
        "paired_bce_partial_evidence_ceiling": decision.partial_evidence_ceiling,
        "paired_bce_exact_census_method": decision.exact_census_method,
        "paired_bce_alpha_nonrecycling_policy": (
            decision.alpha_nonrecycling_policy
        ),
        "paired_bce_candidate_attempt_alpha_axis_hash": (
            decision.candidate_attempt_alpha_axis_hash
        ),
        "paired_bce_candidate_attempt_alpha_coordinate_hash": (
            decision.candidate_attempt_alpha_coordinate_hash
        ),
        "paired_bce_oracle_familywise_alpha": registration.oracle_familywise_alpha,
        "paired_bce_alpha_committed_before_attempt": (
            registration.alpha_committed_before_attempt
        ),
        "paired_bce_attempt_alpha_allocation": (
            registration.attempt_alpha_allocation
        ),
        "paired_bce_registered_look_stratum_counts": (
            registration.registered_look_stratum_counts
        ),
        "paired_bce_alpha_by_look": registration.alpha_by_look,
        "paired_bce_oracle_fold_identity_hash": (
            registration.oracle_fold_identity_hash
        ),
        "paired_bce_oracle_role_separation_receipt_hash": (
            None if role_separation is None else role_separation.receipt_hash
        ),
        "paired_bce_oracle_target_population_identity_hash": (
            registration.strata[0].population_identity_hash
        ),
        "paired_bce_oracle_complement_population_identity_hash": (
            registration.strata[1].population_identity_hash
        ),
        "paired_bce_dynamic_oracle_receipt_hash": (
            None if dynamic is None else dynamic.receipt_hash
        ),
        "paired_bce_generation_frame_hash": (
            None if dynamic is None else dynamic.generation_frame_hash
        ),
        "paired_bce_calibration_prefix_freeze_hash": (
            None if dynamic is None else dynamic.calibration_prefix_freeze_hash
        ),
        "paired_bce_dense_affine_calibration_hash": (
            None if dynamic is None else dynamic.dense_affine_calibration_hash
        ),
        "paired_bce_candidate_affine_calibration_hash": (
            None if dynamic is None else dynamic.candidate_affine_calibration_hash
        ),
        "paired_bce_authority_scope": decision.authority_scope,
        "paired_bce_core_mask_commit_authority": decision.mask_commit_authority,
        "paired_bce_decision_reason": decision.reason,
    }


def _assert_paired_bce_accepted_calibration_freeze(
    *,
    state: V1ControllerState,
    registration: V1PairedBCEAcquisitionRegistration,
    calibration_freeze: V1CandidateCalibrationFreeze,
) -> None:
    """Revalidate the exact pre-O affine pair at the commit boundary."""

    if type(calibration_freeze) is not V1CandidateCalibrationFreeze:
        raise TypeError("paired-BCE acceptance requires an exact calibration freeze")
    active = state.active_candidate
    acquisition = state.acquisition
    if active is None or acquisition is None:
        raise PermissionError("paired-BCE calibration acceptance lacks live state")
    proposal = active.proposal
    calibration_freeze.__post_init__()
    if (
        calibration_freeze.campaign_contract_hash
        != state.binding.contract.campaign_contract_hash
        or calibration_freeze.acquisition_receipt_hash
        != acquisition.receipt.receipt_hash
        or calibration_freeze.queue_index != state.queue_cursor
        or calibration_freeze.candidate_id != proposal.candidate_id
        or calibration_freeze.proposal_parent_mask_hash
        != state.current_mask.mask_hash
        or calibration_freeze.dense_referent_mask_hash
        != state.initial_mask.mask_hash
        or calibration_freeze.candidate_mask_hash
        != proposal.resulting_mask.mask_hash
        or calibration_freeze.calibration_policy_id
        != state.binding.contract.calibration_identity
        or registration.candidate_calibration_freeze_hash
        != calibration_freeze.freeze_hash
        or registration.candidate_mask_hash
        != calibration_freeze.candidate_mask_hash
    ):
        raise ValueError(
            "paired-BCE accepted calibration freeze is stale or names another action"
        )
    dynamic = registration.dynamic_oracle_receipt
    if dynamic is not None and (
        dynamic.candidate_calibration_freeze_hash
        != calibration_freeze.freeze_hash
        or dynamic.dense_affine_calibration_hash
        != calibration_freeze.dense_calibration.frozen_calibration_hash
        or dynamic.candidate_affine_calibration_hash
        != calibration_freeze.candidate_calibration.frozen_calibration_hash
        or dynamic.dense_reference_hash
        != calibration_freeze.dense_referent_mask_hash
        or dynamic.candidate_mask_hash != calibration_freeze.candidate_mask_hash
    ):
        raise ValueError(
            "paired-BCE dynamic oracle differs from its accepted affine freeze"
        )


def advance_v1_controller_look(
    state: V1ControllerState,
    packet: V1StagedAlphaPacketEnvelope,
) -> tuple[V1ControllerState, V1StagedAlphaDecision]:
    """Admit one exact look and deterministically apply its typed action."""

    if type(state) is not V1ControllerState or type(packet) is not V1StagedAlphaPacketEnvelope:
        raise TypeError("controller look advance requires exact state and packet")
    if state.phase != "candidate_active" or state.active_candidate is None or state.acquisition is None:
        raise PermissionError("controller has no active candidate ledger")
    if _is_label_native_acquisition(state.acquisition):
        raise PermissionError(
            "label-native candidates require the paired-BCE decision path"
        )
    active = state.active_candidate
    overlap = set(state.globally_used_draw_ids).intersection(packet.draw_ids)
    if overlap:
        raise ValueError("certificate draw IDs were reused anywhere in the controller run")
    ledger = advance_v1_staged_alpha_ledger(active.ledger, packet)
    decision = evaluate_v1_staged_alpha_look(
        coordinates=state.binding.coordinates,
        ledger=ledger,
    )
    used_draws = state.globally_used_draw_ids + tuple(packet.draw_ids)
    if decision.action == "continue":
        return (
            replace(
                state,
                active_candidate=replace(active, ledger=ledger),
                globally_used_draw_ids=used_draws,
                logical_sequence=state.logical_sequence + 1,
            ),
            decision,
        )

    proposal = active.proposal
    draw_ids = tuple(draw for item in ledger.packets for draw in item.draw_ids)
    source_ids = tuple(source for item in ledger.packets for source in item.source_context_ids)
    attempt_sequence = state.logical_sequence + 1
    mutates = decision.action == "accept_candidate"
    resulting_mask_hash = (
        proposal.resulting_mask.mask_hash if mutates else state.current_mask.mask_hash
    )
    parent_retained = state.current_mask.retained_count
    proposal_removed_fraction = float(proposal.removed_count) / float(parent_retained)
    acquisition_proposal_diagnostics = _acquisition_proposal_diagnostics(
        state.acquisition,
        queue_cursor=state.queue_cursor,
    )
    calibration_freeze = calibration_freeze_from_certificate(
        active.step_freeze.certificate_freeze,
        required=active.step_freeze.certificate_freeze.metadata.get(
            "candidate_calibration_freeze_required",
            False,
        )
        is True,
    )
    attempt = V1CandidateAttemptReceipt(
        campaign_contract_hash=state.binding.contract.campaign_contract_hash,
        controller_run_id=state.controller_run_id,
        candidate_attempt_index=state.candidate_attempt_count,
        accepted_step_index=state.accepted_step_count,
        acquisition_generation_index=state.acquisition.generation_index,
        acquisition_receipt_hash=state.acquisition.receipt.receipt_hash,
        queue_index=_attempt_queue_index(
            state.acquisition,
            queue_cursor=state.queue_cursor,
        ),
        candidate_id=proposal.candidate_id,
        parent_mask_hash=state.current_mask.mask_hash,
        candidate_mask_hash=proposal.resulting_mask.mask_hash,
        step_freeze_hash=active.step_freeze.step_freeze_hash,
        ledger_hash=ledger.ledger_hash,
        decision_receipt_hash=decision.decision_receipt_hash,
        action=decision.action,
        mask_mutated=mutates,
        resulting_current_mask_hash=resulting_mask_hash,
        draw_ids=draw_ids,
        source_context_ids=source_ids,
        reason=decision.reason,
        logical_sequence=attempt_sequence,
        metadata={
            "look_index": decision.look_index,
            "sample_mean_damage": decision.sample_mean_damage,
            "lower_mean_damage": decision.lower_mean_damage,
            "upper_mean_damage": decision.upper_mean_damage,
            "maximum_mean_damage": decision.maximum_mean_damage,
            "candidate_threshold_hash": active.threshold.threshold_hash,
            "sampling_state_hash": active.sampling_law.sampling_state_hash,
            "proposal_parent_mask_hash": proposal.parent_mask_hash,
            "loss_referent_mask_hash": (
                active.step_freeze.certificate_freeze.parent_mask_hash
            ),
            "loss_referent_role": "dense_initial_baseline_mask",
            "damage_currency": "direct_raw_bce_candidate_minus_dense_baseline",
            "budget_scope": "total_candidate_mask_increment_from_dense",
            "candidate_calibration_freeze_hash": (
                None if calibration_freeze is None else calibration_freeze.freeze_hash
            ),
            "candidate_calibration_freeze_evidence_ceiling": (
                active.step_freeze.certificate_freeze.metadata.get(
                    "candidate_calibration_freeze_evidence_ceiling",
                    "legacy_metadata_absent_prior_evidence_ceiling",
                )
            ),
            "candidate_removed_count": proposal.removed_count,
            "candidate_retained_count": proposal.resulting_mask.retained_count,
            "candidate_removed_fraction_of_current": proposal_removed_fraction,
            "candidate_retained_density": (
                float(proposal.resulting_mask.retained_count)
                / float(proposal.resulting_mask.route_count)
            ),
            **_proposal_attempt_metadata(proposal),
            **_candidate_queue_order_metadata(state.acquisition),
            **(
                {
                    "candidate_exact_intervention_damage_evaluated": (
                        proposal.exact_intervention_damage_evaluated
                    )
                }
                if type(proposal) is V1CandidateProposal
                else {}
            ),
            "acquisition_proposal_diagnostics": (
                acquisition_proposal_diagnostics
            ),
        },
    )
    attempts = state.attempt_receipts + (attempt,)

    if decision.action != "accept_candidate":
        next_cursor = state.queue_cursor + 1
        should_stop = len(attempts) >= state.binding.contract.maximum_candidate_count
        queue_exhausted = next_cursor >= len(state.acquisition.queue)
        if should_stop or queue_exhausted:
            if queue_exhausted and _current_generation_full_look_infeasible(
                state,
                attempts,
            ):
                reason: StopReason = (
                    ALL_REGISTERED_ACTIONS_INFEASIBLE_AT_FULL_LOOK
                )
            elif should_stop:
                reason = "maximum_candidate_count_reached"
            else:
                reason = _queue_exhaustion_stop_reason(state.acquisition)
            return (
                V1ControllerState(
                    binding=state.binding,
                    controller_run_id=state.controller_run_id,
                    initial_mask=state.initial_mask,
                    current_mask=state.current_mask,
                    phase="stopped",
                    accepted_trajectory=state.accepted_trajectory,
                    accepted_oracle_trajectory=state.accepted_oracle_trajectory,
                    attempt_receipts=attempts,
                    transition_receipts=state.transition_receipts,
                    acquisition_receipt_hashes=state.acquisition_receipt_hashes,
                    globally_used_draw_ids=used_draws,
                    pareto_milestone_registration=(
                        state.pareto_milestone_registration
                    ),
                    pareto_milestone_receipt=state.pareto_milestone_receipt,
                    stop_reason=reason,
                    logical_sequence=attempt_sequence,
                ),
                decision,
            )
        return (
            V1ControllerState(
                binding=state.binding,
                controller_run_id=state.controller_run_id,
                initial_mask=state.initial_mask,
                current_mask=state.current_mask,
                phase="queue_ready",
                acquisition=state.acquisition,
                queue_cursor=next_cursor,
                accepted_trajectory=state.accepted_trajectory,
                accepted_oracle_trajectory=state.accepted_oracle_trajectory,
                attempt_receipts=attempts,
                transition_receipts=state.transition_receipts,
                acquisition_receipt_hashes=state.acquisition_receipt_hashes,
                globally_used_draw_ids=used_draws,
                pareto_milestone_registration=state.pareto_milestone_registration,
                live_exact_prefix_rung_registration=(
                    state.live_exact_prefix_rung_registration
                ),
                pareto_milestone_receipt=state.pareto_milestone_receipt,
                logical_sequence=attempt_sequence,
            ),
            decision,
        )

    permit = _V1MaskCommitPermit(
        _issuer_token=_COMMIT_TOKEN,
        campaign_contract_hash=state.binding.contract.campaign_contract_hash,
        controller_run_id=state.controller_run_id,
        acquisition_receipt_hash=state.acquisition.receipt.receipt_hash,
        candidate_id=proposal.candidate_id,
        parent_mask_hash=state.current_mask.mask_hash,
        child_mask_hash=proposal.resulting_mask.mask_hash,
        decision_receipt_hash=decision.decision_receipt_hash,
        ledger_hash=ledger.ledger_hash,
    )
    consumption_hash = permit.consume(parent=state.current_mask, child=proposal.resulting_mask)
    removed_indices: tuple[int, ...] | V1RemovedRouteSetIdentity
    if proposal.compact_removed_indices_available:
        removed_indices = proposal.removed_routes.to_tuple()  # type: ignore[union-attr]
    else:
        removed_indices = V1RemovedRouteSetIdentity(
            route_count=proposal.resulting_mask.route_count,
            count=proposal.removed_count,
            index_hash=proposal.removed_route_index_hash,
        )
    transition = V1MaskTransitionReceipt(
        campaign_contract_hash=state.binding.contract.campaign_contract_hash,
        controller_run_id=state.controller_run_id,
        transition_index=state.accepted_step_count,
        accepted_step_index=state.accepted_step_count,
        candidate_attempt_index=state.candidate_attempt_count,
        candidate_id=proposal.candidate_id,
        parent_mask_hash=state.current_mask.mask_hash,
        child_mask_hash=proposal.resulting_mask.mask_hash,
        removed_route_indices=removed_indices,
        accepted_decision_receipt_hash=decision.decision_receipt_hash,
        accepted_ledger_hash=ledger.ledger_hash,
        acquisition_receipt_hash=state.acquisition.receipt.receipt_hash,
        invalidated_chart_hash=_acquisition_invalidation_chart_hash(
            state.acquisition
        ),
        invalidated_composition_hash=state.acquisition.composition.result_hash,
        commit_capability_consumption_hash=consumption_hash,
        logical_sequence=attempt_sequence,
    )
    trajectory = advance_v1_staged_alpha_trajectory(
        state.accepted_trajectory,
        ledger=ledger,
        decision=decision,
    )
    transitions = state.transition_receipts + (transition,)
    accepted_steps = len(transitions)
    cumulative_removed = state.initial_mask.retained_count - proposal.resulting_mask.retained_count
    pending_milestone: V1ParetoMilestonePending | None = None
    milestone_registration = state.pareto_milestone_registration
    if (
        milestone_registration is not None
        and state.pareto_milestone_receipt is None
        and proposal.resulting_mask.retained_count
        == milestone_registration.comparator_retained_route_count
    ):
        rung = state.live_exact_prefix_rung_registration
        if (
            rung is None
            or rung.queue_index != state.queue_cursor
            or rung.candidate_id != proposal.candidate_id
        ):
            raise PermissionError(
                "accepted k_cmp transition lacks its registered exact v1 prefix"
            )
        pending_milestone = V1ParetoMilestonePending(
            campaign_contract_hash=state.binding.contract.campaign_contract_hash,
            campaign_binding_hash=state.binding.binding_hash,
            controller_run_id=state.controller_run_id,
            milestone_registration_hash=milestone_registration.registration_hash,
            comparator_retained_route_count=(
                milestone_registration.comparator_retained_route_count
            ),
            milestone_mask_hash=proposal.resulting_mask.mask_hash,
            mask_lineage_hashes=state.mask_lineage_hashes
            + (proposal.resulting_mask.mask_hash,),
            transition_receipt_hashes=state.transition_receipt_hashes
            + (transition.receipt_hash,),
            accepted_trajectory_hash=trajectory.trajectory_ledger_hash,
            accepted_step_count=accepted_steps,
            candidate_attempt_count=len(attempts),
            milestone_transition_receipt_hash=transition.receipt_hash,
            logical_sequence=attempt_sequence,
        )
    stop_reason: StopReason | None = None
    # Scientific endpoints are resolved before finite compute/statistical caps.
    if cumulative_removed >= state.binding.contract.maximum_removed_count:
        stop_reason = REGISTERED_MINIMUM_RETAINED_ROUTE_COUNT_REACHED
    elif accepted_steps >= state.binding.contract.maximum_step_count:
        stop_reason = "maximum_step_count_reached"
    elif len(attempts) >= state.binding.contract.maximum_candidate_count:
        stop_reason = "maximum_candidate_count_reached"
    phase: ControllerPhase = "stopped" if stop_reason is not None else "needs_acquisition"
    return (
        V1ControllerState(
            binding=state.binding,
            controller_run_id=state.controller_run_id,
            initial_mask=state.initial_mask,
            current_mask=proposal.resulting_mask,
            phase=phase,
            accepted_trajectory=trajectory,
            accepted_oracle_trajectory=state.accepted_oracle_trajectory,
            attempt_receipts=attempts,
            transition_receipts=transitions,
            acquisition_receipt_hashes=state.acquisition_receipt_hashes,
            globally_used_draw_ids=used_draws,
            pareto_milestone_registration=state.pareto_milestone_registration,
            pending_pareto_milestone=pending_milestone,
            pareto_milestone_receipt=state.pareto_milestone_receipt,
            stop_reason=stop_reason,
            logical_sequence=attempt_sequence,
        ),
        decision,
    )


def advance_v1_controller_paired_bce_decision(
    state: V1ControllerState,
    decision: V1PairedBCEDecision,
) -> tuple[V1ControllerState, V1PairedBCEDecision]:
    """Admit one terminal paired-BCE decision and apply its typed action."""

    if type(state) is not V1ControllerState or type(decision) is not V1PairedBCEDecision:
        raise TypeError("paired-BCE controller advance requires exact state and decision")
    if state.phase != "candidate_active" or state.active_candidate is None or state.acquisition is None:
        raise PermissionError("controller has no active paired-BCE candidate")
    active = state.active_candidate
    registration = active.paired_bce_registration
    paired_state = active.paired_bce_state
    if registration is None or paired_state is None:
        raise PermissionError("active candidate is not registered for paired-BCE oracle")
    registration.assert_integrity()
    paired_state.assert_integrity()
    decision.assert_integrity()
    if decision.status == "acquire_more":
        raise PermissionError("paired-BCE acquisition is not terminal")
    if decision.mask_commit_authority:
        raise PermissionError("paired-BCE decision cannot carry mask authority")
    if (
        decision.registration_hash != registration.registration_hash
        or decision.state_hash != paired_state.state_hash
        or decision.candidate_attempt_index != state.candidate_attempt_count
        or decision.candidate_attempt_alpha_axis_hash
        != registration.candidate_attempt_alpha_axis_hash
        or decision.candidate_attempt_alpha_coordinate_hash
        != registration.candidate_attempt_alpha_coordinate_hash
        or decision.dynamic_oracle_receipt_hash
        != registration.dynamic_oracle_receipt_hash
    ):
        raise ValueError("paired-BCE decision has stale registration/state lineage")

    proposal = active.proposal
    draw_ids, source_ids = _paired_bce_terminal_lineage_ids(paired_state)
    overlap = set(state.globally_used_draw_ids).intersection(draw_ids)
    if overlap:
        raise ValueError("paired-BCE terminal lineage was reused anywhere in the controller run")
    used_draws = state.globally_used_draw_ids + draw_ids
    attempt_sequence = state.logical_sequence + 1
    mutates = decision.status == "accept_candidate"
    resulting_mask_hash = (
        proposal.resulting_mask.mask_hash if mutates else state.current_mask.mask_hash
    )
    evidence_ledger_hash = _paired_bce_evidence_ledger_hash(
        registration=registration,
        state=paired_state,
        decision=decision,
    )
    parent_retained = state.current_mask.retained_count
    proposal_removed_fraction = float(proposal.removed_count) / float(parent_retained)
    acquisition_proposal_diagnostics = _acquisition_proposal_diagnostics(
        state.acquisition,
        queue_cursor=state.queue_cursor,
    )
    calibration_freeze = calibration_freeze_from_certificate(
        active.step_freeze.certificate_freeze,
        required=True,
    )
    if calibration_freeze is None:  # required=True is a fail-closed contract.
        raise AssertionError("required paired-BCE calibration freeze is absent")
    _assert_paired_bce_accepted_calibration_freeze(
        state=state,
        registration=registration,
        calibration_freeze=calibration_freeze,
    )
    attempt = V1CandidateAttemptReceipt(
        campaign_contract_hash=state.binding.contract.campaign_contract_hash,
        controller_run_id=state.controller_run_id,
        candidate_attempt_index=state.candidate_attempt_count,
        accepted_step_index=state.accepted_step_count,
        acquisition_generation_index=state.acquisition.generation_index,
        acquisition_receipt_hash=state.acquisition.receipt.receipt_hash,
        queue_index=state.queue_cursor,
        candidate_id=proposal.candidate_id,
        parent_mask_hash=state.current_mask.mask_hash,
        candidate_mask_hash=proposal.resulting_mask.mask_hash,
        step_freeze_hash=active.step_freeze.step_freeze_hash,
        ledger_hash=evidence_ledger_hash,
        decision_receipt_hash=decision.decision_hash,
        action=decision.status,
        mask_mutated=mutates,
        resulting_current_mask_hash=resulting_mask_hash,
        draw_ids=draw_ids,
        source_context_ids=source_ids,
        reason=decision.reason,
        logical_sequence=attempt_sequence,
        metadata={
            "look_index": decision.evaluated_registered_look_index,
            "sample_mean_damage": decision.weighted_point_mean,
            "lower_mean_damage": decision.weighted_confidence_lower,
            "upper_mean_damage": decision.weighted_confidence_upper,
            "maximum_mean_damage": decision.loss_ceiling,
            "candidate_threshold_hash": active.threshold.threshold_hash,
            "sampling_state_hash": active.sampling_law.sampling_state_hash,
            "proposal_parent_mask_hash": proposal.parent_mask_hash,
            "loss_referent_mask_hash": (
                active.step_freeze.certificate_freeze.parent_mask_hash
            ),
            "loss_referent_role": "dense_initial_baseline_mask",
            "damage_currency": "paired_raw_bce_candidate_minus_dense_baseline",
            "budget_scope": "total_candidate_mask_increment_from_dense",
            "candidate_calibration_freeze_hash": calibration_freeze.freeze_hash,
            "candidate_calibration_freeze_evidence_ceiling": (
                active.step_freeze.certificate_freeze.metadata.get(
                    "candidate_calibration_freeze_evidence_ceiling"
                )
            ),
            "candidate_removed_count": proposal.removed_count,
            "candidate_retained_count": proposal.resulting_mask.retained_count,
            "candidate_removed_fraction_of_current": proposal_removed_fraction,
            "candidate_retained_density": (
                float(proposal.resulting_mask.retained_count)
                / float(proposal.resulting_mask.route_count)
            ),
            **_proposal_attempt_metadata(proposal),
            **_candidate_queue_order_metadata(state.acquisition),
            **_label_native_plan_attempt_metadata(state),
            **_paired_bce_attempt_metadata(
                registration=registration,
                state=paired_state,
                decision=decision,
            ),
            "acquisition_proposal_diagnostics": (
                acquisition_proposal_diagnostics
            ),
        },
    )
    attempts = state.attempt_receipts + (attempt,)

    if decision.status != "accept_candidate":
        if _is_label_native_acquisition(state.acquisition):
            plan = state.label_native_runtime_plan
            if plan is None:
                raise ValueError(
                    "label-native rejection lost its immutable runtime plan"
                )
            next_cursor = state.queue_cursor + 1
            rejected_indices = (
                state.label_native_rejected_plan_indices + (state.queue_cursor,)
            )
            should_stop = (
                len(attempts) >= state.binding.contract.maximum_candidate_count
            )
            plan_exhausted = next_cursor >= len(plan.descriptors)
            reason: StopReason | None = None
            if plan_exhausted:
                reason = _label_native_plan_exhaustion_stop_reason(plan)
            elif should_stop:
                reason = "maximum_candidate_count_reached"
            return (
                V1ControllerState(
                    binding=state.binding,
                    controller_run_id=state.controller_run_id,
                    initial_mask=state.initial_mask,
                    current_mask=state.current_mask,
                    phase="stopped" if reason is not None else "needs_acquisition",
                    queue_cursor=next_cursor,
                    label_native_runtime_plan=plan,
                    label_native_runtime_plan_hashes=(
                        state.label_native_runtime_plan_hashes
                    ),
                    label_native_rejected_plan_indices=rejected_indices,
                    accepted_trajectory=state.accepted_trajectory,
                    accepted_oracle_trajectory=state.accepted_oracle_trajectory,
                    attempt_receipts=attempts,
                    transition_receipts=state.transition_receipts,
                    acquisition_receipt_hashes=state.acquisition_receipt_hashes,
                    globally_used_draw_ids=used_draws,
                    pareto_milestone_registration=(
                        state.pareto_milestone_registration
                    ),
                    pareto_milestone_receipt=state.pareto_milestone_receipt,
                    stop_reason=reason,
                    logical_sequence=attempt_sequence,
                ),
                decision,
            )
        next_cursor = state.queue_cursor + 1
        should_stop = len(attempts) >= state.binding.contract.maximum_candidate_count
        queue_exhausted = next_cursor >= len(state.acquisition.queue)
        if should_stop or queue_exhausted:
            if queue_exhausted and _current_generation_paired_fqr_barrier_is_saturated(
                state,
                attempts,
            ):
                reason: StopReason = (
                    ALL_REGISTERED_FQR_ACTIONS_REJECTED_BY_PAIRED_BCE_BARRIER
                )
            elif should_stop:
                reason = "maximum_candidate_count_reached"
            else:
                reason = _queue_exhaustion_stop_reason(state.acquisition)
            return (
                V1ControllerState(
                    binding=state.binding,
                    controller_run_id=state.controller_run_id,
                    initial_mask=state.initial_mask,
                    current_mask=state.current_mask,
                    phase="stopped",
                    accepted_trajectory=state.accepted_trajectory,
                    accepted_oracle_trajectory=state.accepted_oracle_trajectory,
                    attempt_receipts=attempts,
                    transition_receipts=state.transition_receipts,
                    acquisition_receipt_hashes=state.acquisition_receipt_hashes,
                    globally_used_draw_ids=used_draws,
                    pareto_milestone_registration=(
                        state.pareto_milestone_registration
                    ),
                    pareto_milestone_receipt=state.pareto_milestone_receipt,
                    stop_reason=reason,
                    logical_sequence=attempt_sequence,
                ),
                decision,
            )
        return (
            V1ControllerState(
                binding=state.binding,
                controller_run_id=state.controller_run_id,
                initial_mask=state.initial_mask,
                current_mask=state.current_mask,
                phase="queue_ready",
                acquisition=state.acquisition,
                queue_cursor=next_cursor,
                accepted_trajectory=state.accepted_trajectory,
                accepted_oracle_trajectory=state.accepted_oracle_trajectory,
                attempt_receipts=attempts,
                transition_receipts=state.transition_receipts,
                acquisition_receipt_hashes=state.acquisition_receipt_hashes,
                globally_used_draw_ids=used_draws,
                pareto_milestone_registration=state.pareto_milestone_registration,
                live_exact_prefix_rung_registration=(
                    state.live_exact_prefix_rung_registration
                ),
                pareto_milestone_receipt=state.pareto_milestone_receipt,
                logical_sequence=attempt_sequence,
            ),
            decision,
        )

    permit = _V1MaskCommitPermit(
        _issuer_token=_COMMIT_TOKEN,
        campaign_contract_hash=state.binding.contract.campaign_contract_hash,
        controller_run_id=state.controller_run_id,
        acquisition_receipt_hash=state.acquisition.receipt.receipt_hash,
        candidate_id=proposal.candidate_id,
        parent_mask_hash=state.current_mask.mask_hash,
        child_mask_hash=proposal.resulting_mask.mask_hash,
        decision_receipt_hash=decision.decision_hash,
        ledger_hash=evidence_ledger_hash,
    )
    consumption_hash = permit.consume(parent=state.current_mask, child=proposal.resulting_mask)
    removed_indices: tuple[int, ...] | V1RemovedRouteSetIdentity
    if proposal.compact_removed_indices_available:
        removed_indices = proposal.removed_routes.to_tuple()  # type: ignore[union-attr]
    else:
        removed_indices = V1RemovedRouteSetIdentity(
            route_count=proposal.resulting_mask.route_count,
            count=proposal.removed_count,
            index_hash=proposal.removed_route_index_hash,
        )
    transition = V1MaskTransitionReceipt(
        campaign_contract_hash=state.binding.contract.campaign_contract_hash,
        controller_run_id=state.controller_run_id,
        transition_index=state.accepted_step_count,
        accepted_step_index=state.accepted_step_count,
        candidate_attempt_index=state.candidate_attempt_count,
        candidate_id=proposal.candidate_id,
        parent_mask_hash=state.current_mask.mask_hash,
        child_mask_hash=proposal.resulting_mask.mask_hash,
        removed_route_indices=removed_indices,
        accepted_decision_receipt_hash=decision.decision_hash,
        accepted_ledger_hash=evidence_ledger_hash,
        acquisition_receipt_hash=state.acquisition.receipt.receipt_hash,
        invalidated_chart_hash=_acquisition_invalidation_chart_hash(
            state.acquisition
        ),
        invalidated_composition_hash=_acquisition_invalidation_composition_hash(
            state.acquisition
        ),
        commit_capability_consumption_hash=consumption_hash,
        logical_sequence=attempt_sequence,
    )
    trajectory = _append_accepted_oracle_entry(
        state.accepted_oracle_trajectory,
        protocol="paired_bce",
        accepted_step_index=state.accepted_step_count,
        candidate_attempt_index=state.candidate_attempt_count,
        acquisition_receipt_hash=state.acquisition.receipt.receipt_hash,
        candidate_id=proposal.candidate_id,
        parent_mask_hash=state.current_mask.mask_hash,
        candidate_mask_hash=proposal.resulting_mask.mask_hash,
        step_freeze_hash=active.step_freeze.step_freeze_hash,
        evidence_ledger_hash=evidence_ledger_hash,
        decision_receipt_hash=decision.decision_hash,
        metadata={
            "paired_bce_registration_hash": registration.registration_hash,
            "paired_bce_state_hash": paired_state.state_hash,
            "paired_bce_receipt_chain_hash": paired_state.receipt_chain_hash,
            "paired_bce_decision_hash": decision.decision_hash,
            "candidate_calibration_freeze_hash": calibration_freeze.freeze_hash,
            "dense_affine_calibration_hash": (
                calibration_freeze.dense_calibration.frozen_calibration_hash
            ),
            "candidate_affine_calibration_hash": (
                calibration_freeze.candidate_calibration.frozen_calibration_hash
            ),
            "paired_bce_dynamic_oracle_receipt_hash": (
                registration.dynamic_oracle_receipt_hash
            ),
            "paired_bce_full_census": decision.full_census,
            "paired_bce_exact_finite_set_decision": (
                decision.exact_finite_set_decision
            ),
        },
        candidate_calibration_freeze=calibration_freeze,
    )
    transitions = state.transition_receipts + (transition,)
    accepted_steps = len(transitions)
    cumulative_removed = (
        state.initial_mask.retained_count - proposal.resulting_mask.retained_count
    )
    pending_milestone: V1ParetoMilestonePending | None = None
    milestone_registration = state.pareto_milestone_registration
    if (
        milestone_registration is not None
        and state.pareto_milestone_receipt is None
        and proposal.resulting_mask.retained_count
        == milestone_registration.comparator_retained_route_count
    ):
        rung = state.live_exact_prefix_rung_registration
        if (
            rung is None
            or rung.queue_index != state.queue_cursor
            or rung.candidate_id != proposal.candidate_id
        ):
            raise PermissionError(
                "accepted k_cmp transition lacks its registered exact v1 prefix"
            )
        pending_milestone = V1ParetoMilestonePending(
            campaign_contract_hash=state.binding.contract.campaign_contract_hash,
            campaign_binding_hash=state.binding.binding_hash,
            controller_run_id=state.controller_run_id,
            milestone_registration_hash=milestone_registration.registration_hash,
            comparator_retained_route_count=(
                milestone_registration.comparator_retained_route_count
            ),
            milestone_mask_hash=proposal.resulting_mask.mask_hash,
            mask_lineage_hashes=state.mask_lineage_hashes
            + (proposal.resulting_mask.mask_hash,),
            transition_receipt_hashes=state.transition_receipt_hashes
            + (transition.receipt_hash,),
            accepted_trajectory_hash=trajectory.trajectory_ledger_hash,
            accepted_step_count=accepted_steps,
            candidate_attempt_count=len(attempts),
            milestone_transition_receipt_hash=transition.receipt_hash,
            logical_sequence=attempt_sequence,
        )
    stop_reason: StopReason | None = None
    if cumulative_removed >= state.binding.contract.maximum_removed_count:
        stop_reason = REGISTERED_MINIMUM_RETAINED_ROUTE_COUNT_REACHED
    elif accepted_steps >= state.binding.contract.maximum_step_count:
        stop_reason = "maximum_step_count_reached"
    elif len(attempts) >= state.binding.contract.maximum_candidate_count:
        stop_reason = "maximum_candidate_count_reached"
    phase: ControllerPhase = "stopped" if stop_reason is not None else "needs_acquisition"
    return (
        V1ControllerState(
            binding=state.binding,
            controller_run_id=state.controller_run_id,
            initial_mask=state.initial_mask,
            current_mask=proposal.resulting_mask,
            phase=phase,
            accepted_trajectory=state.accepted_trajectory,
            accepted_oracle_trajectory=trajectory,
            attempt_receipts=attempts,
            transition_receipts=transitions,
            acquisition_receipt_hashes=state.acquisition_receipt_hashes,
            label_native_runtime_plan_hashes=(
                state.label_native_runtime_plan_hashes
            ),
            globally_used_draw_ids=used_draws,
            pareto_milestone_registration=state.pareto_milestone_registration,
            pending_pareto_milestone=pending_milestone,
            pareto_milestone_receipt=state.pareto_milestone_receipt,
            stop_reason=stop_reason,
            logical_sequence=attempt_sequence,
        ),
        decision,
    )


def advance_v1_controller_calibration_horizon_skip(
    state: V1ControllerState,
    decision: V1PairedBCECalibrationHorizonSkipDecision,
) -> tuple[V1ControllerState, V1PairedBCECalibrationHorizonSkipDecision]:
    """Retire one exact causal K-horizon refusal without opening O or a mask.

    The transition is deliberately pure and queue-local.  It consumes exactly
    one global candidate-attempt/alpha coordinate, advances exactly one frozen
    queue candidate, and carries no staged sampling law, oracle registration,
    draw lineage, or mask-commit path.
    """

    if type(state) is not V1ControllerState or type(
        decision
    ) is not V1PairedBCECalibrationHorizonSkipDecision:
        raise TypeError("calibration-horizon skip requires exact state/decision")
    if (
        state.phase != "queue_ready"
        or state.acquisition is None
        or state.active_candidate is not None
    ):
        raise PermissionError("calibration-horizon skip requires a ready queue")
    decision.assert_integrity()
    proposal = _selected_proposal(
        state.acquisition,
        queue_cursor=state.queue_cursor,
    )
    attempt_queue_index = _attempt_queue_index(
        state.acquisition,
        queue_cursor=state.queue_cursor,
    )
    expected = {
        "campaign_contract_hash": state.binding.contract.campaign_contract_hash,
        "controller_run_id": state.controller_run_id,
        "acquisition_receipt_hash": state.acquisition.receipt.receipt_hash,
        "acquisition_generation_index": state.acquisition.generation_index,
        "accepted_step_index": state.accepted_step_count,
        "queue_index": attempt_queue_index,
        "candidate_id": proposal.candidate_id,
        "parent_mask_hash": state.current_mask.mask_hash,
        "candidate_mask_hash": proposal.resulting_mask.mask_hash,
        "candidate_attempt_index": state.candidate_attempt_count,
        "maximum_candidate_attempts": (
            state.binding.contract.maximum_candidate_count
        ),
    }
    if any(getattr(decision, name) != value for name, value in expected.items()):
        raise ValueError("calibration-horizon skip has stale queue/campaign lineage")
    if (
        decision.causal_horizon_refusal_receipt["candidate_mask_hash"]
        != proposal.resulting_mask.mask_hash
        or decision.causal_horizon_refusal_receipt["candidate_attempt_index"]
        != state.candidate_attempt_count
        or decision.mask_mutated
        or decision.mask_commit_authority
        or decision.sampling_law_opened
        or decision.oracle_registration_opened
        or decision.oracle_stream_opened
        or decision.oracle_rows_observed
    ):
        raise PermissionError("calibration-horizon skip crossed its authority firewall")

    attempt_sequence = state.logical_sequence + 1
    metadata = {
        "oracle_protocol": "paired_bce_calibration_horizon_skip",
        "calibration_horizon_skip_decision": dict(decision.to_payload()),
        "calibration_horizon_skip_decision_hash": decision.decision_hash,
        "causal_horizon_refusal_receipt": dict(
            decision.causal_horizon_refusal_receipt
        ),
        "causal_horizon_refusal_receipt_hash": decision.refusal_receipt_hash,
        "candidate_threshold_hash": decision.candidate_threshold_hash,
        "maximum_mean_damage": decision.maximum_mean_damage,
        "candidate_attempt_alpha_axis_hash": (
            decision.candidate_attempt_alpha_axis_hash
        ),
        "candidate_attempt_alpha_coordinate_hash": (
            decision.candidate_attempt_alpha_coordinate_hash
        ),
        "candidate_attempt_alpha_index": decision.candidate_attempt_index,
        "maximum_candidate_attempts": decision.maximum_candidate_attempts,
        "oracle_familywise_alpha": decision.oracle_familywise_alpha,
        "alpha_committed_before_attempt": decision.alpha_committed_before_attempt,
        "attempt_alpha_allocation": decision.attempt_alpha_allocation,
        "master_decision_alpha_registration_hash": (
            decision.master_decision_alpha_registration_hash
        ),
        "attempt_alpha_retired": True,
        "unused_alpha_recycled": False,
        "sampling_law_opened": False,
        "oracle_registration_opened": False,
        "oracle_stream_opened": False,
        "oracle_rows_observed": False,
        "candidate_removed_count": proposal.removed_count,
        "candidate_retained_count": proposal.resulting_mask.retained_count,
        **_proposal_attempt_metadata(proposal),
        **_candidate_queue_order_metadata(state.acquisition),
        **_label_native_plan_attempt_metadata(state),
        "acquisition_proposal_diagnostics": _acquisition_proposal_diagnostics(
            state.acquisition,
            queue_cursor=state.queue_cursor,
        ),
    }
    attempt = V1CandidateAttemptReceipt(
        campaign_contract_hash=state.binding.contract.campaign_contract_hash,
        controller_run_id=state.controller_run_id,
        candidate_attempt_index=state.candidate_attempt_count,
        accepted_step_index=state.accepted_step_count,
        acquisition_generation_index=state.acquisition.generation_index,
        acquisition_receipt_hash=state.acquisition.receipt.receipt_hash,
        queue_index=attempt_queue_index,
        candidate_id=proposal.candidate_id,
        parent_mask_hash=state.current_mask.mask_hash,
        candidate_mask_hash=proposal.resulting_mask.mask_hash,
        step_freeze_hash=None,
        ledger_hash=decision.refusal_ledger_hash,
        decision_receipt_hash=decision.decision_hash,
        action="skip_candidate_calibration_refusal",
        mask_mutated=False,
        resulting_current_mask_hash=state.current_mask.mask_hash,
        draw_ids=(),
        source_context_ids=(),
        reason=decision.reason,
        logical_sequence=attempt_sequence,
        metadata=metadata,
        schema="partition_free_v1_candidate_attempt_receipt_v2",
    )
    attempts = state.attempt_receipts + (attempt,)
    if _is_label_native_acquisition(state.acquisition):
        plan = state.label_native_runtime_plan
        if plan is None:
            raise ValueError(
                "label-native calibration refusal lost its immutable runtime plan"
            )
        next_cursor = state.queue_cursor + 1
        rejected_indices = (
            state.label_native_rejected_plan_indices + (state.queue_cursor,)
        )
        candidate_cap = (
            len(attempts) >= state.binding.contract.maximum_candidate_count
        )
        plan_exhausted = next_cursor >= len(plan.descriptors)
        stop_reason: StopReason | None = None
        if plan_exhausted:
            stop_reason = _label_native_plan_exhaustion_stop_reason(plan)
        elif candidate_cap:
            stop_reason = "maximum_candidate_count_reached"
        return (
            V1ControllerState(
                binding=state.binding,
                controller_run_id=state.controller_run_id,
                initial_mask=state.initial_mask,
                current_mask=state.current_mask,
                phase="stopped" if stop_reason is not None else "needs_acquisition",
                queue_cursor=next_cursor,
                label_native_runtime_plan=plan,
                label_native_runtime_plan_hashes=(
                    state.label_native_runtime_plan_hashes
                ),
                label_native_rejected_plan_indices=rejected_indices,
                accepted_trajectory=state.accepted_trajectory,
                accepted_oracle_trajectory=state.accepted_oracle_trajectory,
                attempt_receipts=attempts,
                transition_receipts=state.transition_receipts,
                acquisition_receipt_hashes=state.acquisition_receipt_hashes,
                globally_used_draw_ids=state.globally_used_draw_ids,
                pareto_milestone_registration=state.pareto_milestone_registration,
                pareto_milestone_receipt=state.pareto_milestone_receipt,
                stop_reason=stop_reason,
                logical_sequence=attempt_sequence,
            ),
            decision,
        )
    next_cursor = state.queue_cursor + 1
    candidate_cap = len(attempts) >= state.binding.contract.maximum_candidate_count
    queue_exhausted = next_cursor >= len(state.acquisition.queue)
    if candidate_cap or queue_exhausted:
        stop_reason: StopReason = (
            "maximum_candidate_count_reached"
            if candidate_cap
            else _queue_exhaustion_stop_reason(state.acquisition)
        )
        return (
            V1ControllerState(
                binding=state.binding,
                controller_run_id=state.controller_run_id,
                initial_mask=state.initial_mask,
                current_mask=state.current_mask,
                phase="stopped",
                accepted_trajectory=state.accepted_trajectory,
                accepted_oracle_trajectory=state.accepted_oracle_trajectory,
                attempt_receipts=attempts,
                transition_receipts=state.transition_receipts,
                acquisition_receipt_hashes=state.acquisition_receipt_hashes,
                globally_used_draw_ids=state.globally_used_draw_ids,
                pareto_milestone_registration=state.pareto_milestone_registration,
                pareto_milestone_receipt=state.pareto_milestone_receipt,
                stop_reason=stop_reason,
                logical_sequence=attempt_sequence,
            ),
            decision,
        )
    return (
        V1ControllerState(
            binding=state.binding,
            controller_run_id=state.controller_run_id,
            initial_mask=state.initial_mask,
            current_mask=state.current_mask,
            phase="queue_ready",
            acquisition=state.acquisition,
            queue_cursor=next_cursor,
            accepted_trajectory=state.accepted_trajectory,
            accepted_oracle_trajectory=state.accepted_oracle_trajectory,
            attempt_receipts=attempts,
            transition_receipts=state.transition_receipts,
            acquisition_receipt_hashes=state.acquisition_receipt_hashes,
            globally_used_draw_ids=state.globally_used_draw_ids,
            pareto_milestone_registration=state.pareto_milestone_registration,
            live_exact_prefix_rung_registration=(
                state.live_exact_prefix_rung_registration
            ),
            pareto_milestone_receipt=state.pareto_milestone_receipt,
            logical_sequence=attempt_sequence,
        ),
        decision,
    )


def seal_v1_pareto_milestone_after_checkpoint(
    state: V1ControllerState,
    *,
    checkpoint_binding: V1ParetoMilestoneCheckpointBinding,
) -> V1ControllerState:
    """Seal the matched-sparsity receipt after the normal checkpoint seam.

    The binding must name the exact pending controller state and snapshot.  No
    terminal-validation result is accepted or opened here, and the milestone
    itself does not stop the pruning trajectory.
    """

    if type(state) is not V1ControllerState:
        raise TypeError("Pareto milestone sealing requires an exact controller state")
    pending = state.pending_pareto_milestone
    registration = state.pareto_milestone_registration
    if pending is None or registration is None:
        raise PermissionError("controller has no pending Pareto milestone")
    if state.pareto_milestone_receipt is not None:
        raise ValueError("controller already carries a sealed Pareto milestone")
    if type(checkpoint_binding) is not V1ParetoMilestoneCheckpointBinding:
        raise TypeError("Pareto milestone checkpoint binding has the wrong type")
    if (
        checkpoint_binding.pending_milestone_hash != pending.pending_hash
        or checkpoint_binding.controller_state_hash != state.state_hash
        or checkpoint_binding.controller_snapshot_receipt_hash
        != state.snapshot_receipt().receipt_hash
    ):
        raise ValueError("checkpoint binding does not name the pending controller state")
    receipt = V1ParetoMilestoneReceipt(
        pending=pending,
        checkpoint_binding=checkpoint_binding,
    )
    return replace(
        state,
        pending_pareto_milestone=None,
        pareto_milestone_receipt=receipt,
        logical_sequence=state.logical_sequence + 1,
    )


def stop_v1_controller(
    state: V1ControllerState,
    *,
    reason: StopReason = "explicit_registered_stop",
) -> V1ControllerState:
    """Enter one explicit terminal state; empty arrays never stop implicitly."""

    if type(state) is not V1ControllerState:
        raise TypeError("controller stop requires an exact state")
    if state.phase == "candidate_active":
        raise PermissionError("cannot stop while a candidate has unresolved observed state")
    if reason == REGISTERED_MINIMUM_RETAINED_ROUTE_COUNT_REACHED and (
        state.cumulative_removed_count
        != state.binding.contract.maximum_removed_count
    ):
        raise PermissionError(
            "minimum-retained stop requires the exact registered count"
        )
    if reason == ALL_REGISTERED_ACTIONS_INFEASIBLE_AT_FULL_LOOK:
        raise PermissionError(
            "all-actions scientific endpoint can be issued only by queue exhaustion"
        )
    if reason == ALL_REGISTERED_FQR_ACTIONS_REJECTED_BY_PAIRED_BCE_BARRIER:
        raise PermissionError(
            "paired-BCE F/Q/R saturation endpoint can be issued only by exact "
            "canonical queue exhaustion"
        )
    if reason == SIGNED_WORK_RESOLVED_PREFIX_EXHAUSTED_NEEDS_MORE_C:
        raise PermissionError(
            "signed-work C-blocked exhaustion can be issued only by exact "
            "resolved-prefix queue exhaustion"
        )
    if reason == FQR_NO_CANDIDATE_HORIZON_STOP_REASON:
        raise PermissionError(
            "stable-negative F/Q/R horizon can be issued only by its typed "
            "proof-bearing stop transition"
        )
    if reason == "maximum_step_count_reached" and (
        state.accepted_step_count < state.binding.contract.maximum_step_count
    ):
        raise PermissionError("step-cap exhaustion was requested before its horizon")
    if reason == "maximum_candidate_count_reached" and (
        state.candidate_attempt_count
        < state.binding.contract.maximum_candidate_count
    ):
        raise PermissionError("candidate-cap exhaustion was requested before its horizon")
    if state.phase == "stopped":
        if state.stop_reason != reason:
            raise ValueError("controller is already stopped under another reason")
        return state
    return V1ControllerState(
        binding=state.binding,
        controller_run_id=state.controller_run_id,
        initial_mask=state.initial_mask,
        current_mask=state.current_mask,
        phase="stopped",
        queue_cursor=state.queue_cursor,
        label_native_runtime_plan=state.label_native_runtime_plan,
        label_native_runtime_plan_hashes=state.label_native_runtime_plan_hashes,
        label_native_rejected_plan_indices=(
            state.label_native_rejected_plan_indices
        ),
        accepted_trajectory=state.accepted_trajectory,
        accepted_oracle_trajectory=state.accepted_oracle_trajectory,
        attempt_receipts=state.attempt_receipts,
        transition_receipts=state.transition_receipts,
        acquisition_receipt_hashes=state.acquisition_receipt_hashes,
        globally_used_draw_ids=state.globally_used_draw_ids,
        pareto_milestone_registration=state.pareto_milestone_registration,
        pending_pareto_milestone=state.pending_pareto_milestone,
        pareto_milestone_receipt=state.pareto_milestone_receipt,
        stop_reason=reason,
        logical_sequence=state.logical_sequence + 1,
    )


def stop_v1_controller_at_fqr_no_candidate_horizon(
    state: V1ControllerState,
    witness: V1ControllerProposalHorizonWitness,
) -> V1ControllerState:
    """Stop at a replayable stable-negative C horizon without losing evidence."""

    if type(state) is not V1ControllerState:
        raise TypeError("F/Q/R horizon stop requires an exact controller state")
    if type(witness) is not V1ControllerProposalHorizonWitness:
        raise TypeError("F/Q/R horizon stop requires the exact typed witness")
    if state.phase == "stopped":
        if (
            state.stop_reason != FQR_NO_CANDIDATE_HORIZON_STOP_REASON
            or state.fqr_no_candidate_horizon_witness != witness
        ):
            raise ValueError("controller is already stopped under another horizon")
        return state
    if state.phase != "needs_acquisition":
        raise PermissionError(
            "stable-negative F/Q/R horizon requires an acquisition-free proposal seam"
        )
    if (
        state.acquisition is not None
        or state.active_candidate is not None
        or state.label_native_runtime_plan is not None
    ):
        raise PermissionError(
            "stable-negative F/Q/R horizon cannot discard live scientific state"
        )
    # The returned state's invariant reconstructs this exact pre-stop state and
    # snapshot and verifies every campaign, generation, mask, and lineage
    # coordinate carried by the witness.
    return replace(
        state,
        phase="stopped",
        fqr_no_candidate_horizon_witness=witness,
        stop_reason=FQR_NO_CANDIDATE_HORIZON_STOP_REASON,
        logical_sequence=state.logical_sequence + 1,
    )


__all__ = [
    "SIGNED_WORK_RESOLVED_PREFIX_EXHAUSTED_NEEDS_MORE_C",
    "StopReason",
    "V1AcceptedOracleTrajectoryEntry",
    "V1AcceptedOracleTrajectoryLedger",
    "V1ActiveCandidate",
    "V1CandidateCalibrationFreeze",
    "V1CandidateDamageThreshold",
    "V1ControllerState",
    "V1FrozenAffineCalibration",
    "V1StepSamplingLaw",
    "advance_v1_controller_calibration_horizon_skip",
    "advance_v1_controller_paired_bce_decision",
    "advance_v1_controller_look",
    "attach_v1_acquisition",
    "attach_v1_label_native_runtime_plan",
    "attach_v1_label_native_selected_acquisition",
    "calibration_freeze_from_certificate",
    "create_v1_candidate_threshold",
    "create_v1_step_sampling_law",
    "freeze_v1_candidate",
    "initialize_v1_controller",
    "register_v1_exact_pareto_prefix_rung",
    "seal_v1_pareto_milestone_after_checkpoint",
    "stop_v1_controller",
    "stop_v1_controller_at_fqr_no_candidate_horizon",
    "update_v1_controller_paired_bce_state",
]
