"""Final sampled-v7 D-decision stability composition.

The global true-Q theorem and the corrected signed-F witness answer different
parts of the D decision.  This module is the first seam allowed to join them.
It derives the selected packed action from the exact action-pair source graph,
requires strict corrected-F separation for every branch alias of that physical
action, and binds the consumed A/F/J observation family to the same D source.

Only D-decision stability is opened.  Independent C measurement remains a
mandatory next stage; K, O, commit, relinearization, terminal, runtime, and
Pareto authority remain closed.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
import math
from types import MappingProxyType
from typing import Final, Mapping, NoReturn

from mechanistic_systems.contracts.content_address import content_hash
from .modern_v1_phase_a_global_q_hard_face_margin_certificate import (
    V1PhaseAGlobalQHardFaceMarginCertificateResult,
    V1PhaseAGlobalQHardFaceMarginCertificateStatus,
)
from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_phase_a_true_q_d_decision_witness import (
    V1PhaseATrueQDDecisionWitnessReceipt,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_sampled_full_population_compact_source import (
    V1SampledFullPopulationCompactSourceHandle,
    V1SampledFullPopulationCompactSourceReceipt,
)
from .modern_v1_sampled_v7_action_pair_alias_source_join import (
    V1SampledV7ActionPairAliasSourceJoinReceipt,
)
from .modern_v1_sampled_v7_branch_action_pair import (
    V1SampledV7BranchActionAlias,
    V1SampledV7PhysicalActionCustody,
)
from .modern_v1_sampled_v7_branch_decision_image_production import (
    V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt,
)
from .modern_v1_sampled_v7_complete_action_coordinate_packed_crosswalk import (
    V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt,
    V1SampledV7StructuralActionCrosswalkRow,
)
from .modern_v1_sampled_v7_corrected_a_observation_binding import (
    V1SampledV7CorrectedAObservationBindingReceipt,
)
from .modern_v1_sampled_v7_d_alpha_measurement_owner_binding import (
    V1SampledV7DAlphaMeasurementOwnerBindingReceipt,
)
from .modern_v1_sampled_v7_d_alpha_observation_family_closure import (
    V1SampledV7DAlphaObservationFamilyClosureReceipt,
)
from .modern_v1_sampled_v7_d_source_image_binding import (
    V1SampledV7DSourceImageBindingReceipt,
)
from .modern_v1_sampled_v7_literal_j_observation_binding import (
    V1SampledV7LiteralJObservationBindingReceipt,
)
from .modern_v1_sampled_v7_phase_a_robust_shell_decision import (
    V1SampledV7PhaseARobustDecisionStatus,
    V1SampledV7PhaseARobustShellDecisionReceipt,
)
from .modern_v1_sampled_v7_signed_f_observation_binding import (
    V1SampledV7SignedFObservationBindingReceipt,
)
from .modern_v1_signed_work_block_support_production_witness import (
    require_v1_block_support_production_witness_decision_v2,
)
from .modern_v1_signed_work_pre_c_packed_actions import (
    V1SignedWorkPackedActionDedupIdentity,
    v1_signed_work_packed_action_dedup_identity,
)


SAMPLED_V7_D_DECISION_STABILITY_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_D_decision_stability_"
    "true_Q_signed_F_nonrecycling_A_F_J_v1"
)
SAMPLED_V7_D_DECISION_STABILITY_ALGORITHM: Final = (
    "derive_Q_selected_exact_packed_action_join_all_matching_corrected_F_"
    "branch_alias_witnesses_and_consumed_same_D_A_F_J_family_v1"
)
SAMPLED_V7_D_DECISION_STABILITY_AUTHORITY: Final = (
    "D_decision_stability_only_independent_C_still_required_no_C_outcome_K_O_"
    "commit_relinearization_runtime_terminal_or_pareto_authority_v1"
)

_PREFIX: Final = "partition-free-v1-sampled-v7-D-decision-stability-"
_RECEIPT_TOKEN: Final = object()
_CANONICAL_BRANCHES: Final = (
    V1RelaxedKKTBranch.SPECTRAL,
    V1RelaxedKKTBranch.DIAGONAL_L1,
)


class V1SampledV7DDecisionStabilityRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    PARENT_INTEGRITY = "parent_integrity"
    SOURCE_MISMATCH = "source_mismatch"
    PRODUCTION_CUDA_REQUIRED = "production_cuda_required"
    Q_COMPONENT_UNCERTIFIED = "q_component_uncertified"
    SELECTED_ACTION_MISSING_OR_AMBIGUOUS = "selected_action_missing_or_ambiguous"
    ACTION_IDENTITY_MISMATCH = "action_identity_mismatch"
    ALIAS_COVERAGE_INCOMPLETE = "alias_coverage_incomplete"
    SIGNED_F_EVIDENCE_MISSING = "signed_F_evidence_missing"
    SIGNED_F_SEPARATION_UNSTABLE = "signed_F_separation_unstable"
    ALPHA_FAMILY_MISMATCH = "alpha_family_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"
    MUTATED_CUSTODY = "mutated_custody"


class V1SampledV7DDecisionStabilityRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SampledV7DDecisionStabilityRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SampledV7DDecisionStabilityRefusalCode:
            raise TypeError("sampled-v7 D-stability refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledV7DDecisionStabilityRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledV7DDecisionStabilityRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return value


def _count_pair(value: object, *, name: str) -> tuple[int, int]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(
            type(item) is not int or isinstance(item, bool) or item < 0
            for item in value
        )
        or sum(value) <= 0
    ):
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty nonnegative count pair",
        )
    return value


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(dict(payload), prefix=_PREFIX + role + "-")


def _assert_exact_parents(
    *,
    true_q_decision_witness: V1PhaseATrueQDDecisionWitnessReceipt,
    robust_shell_decision: V1SampledV7PhaseARobustShellDecisionReceipt,
    action_pair_alias_source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
    hard_face_margin_certificate: V1PhaseAGlobalQHardFaceMarginCertificateResult,
    alpha_observation_family_closure: V1SampledV7DAlphaObservationFamilyClosureReceipt,
) -> tuple[str, str, str, str, str]:
    expected = (
        (
            true_q_decision_witness,
            V1PhaseATrueQDDecisionWitnessReceipt,
            "true-Q D component",
        ),
        (
            robust_shell_decision,
            V1SampledV7PhaseARobustShellDecisionReceipt,
            "robust shell decision",
        ),
        (
            action_pair_alias_source_join,
            V1SampledV7ActionPairAliasSourceJoinReceipt,
            "action-pair source join",
        ),
        (
            hard_face_margin_certificate,
            V1PhaseAGlobalQHardFaceMarginCertificateResult,
            "hard-face margin certificate",
        ),
        (
            alpha_observation_family_closure,
            V1SampledV7DAlphaObservationFamilyClosureReceipt,
            "A/F/J observation-family closure",
        ),
    )
    if any(type(value) is not exact for value, exact, _name in expected):
        names = ", ".join(name for _value, _exact, name in expected)
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.INVALID_TYPE,
            f"D stability requires exact live parents: {names}",
        )
    try:
        q_hash = true_q_decision_witness.assert_integrity()
        robust_hash = robust_shell_decision.assert_integrity()
        source_join_hash = action_pair_alias_source_join.assert_integrity()
        hard_face_margin_certificate.compact_handle.assert_live()
        hard_hash = hard_face_margin_certificate.receipt.assert_integrity()
        alpha_hash = alpha_observation_family_closure.assert_live()
    except V1SampledV7DDecisionStabilityRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.PARENT_INTEGRITY,
            "one D-stability parent failed live replay: "
            f"{type(error).__name__}: {error}",
        )
    return q_hash, robust_hash, source_join_hash, hard_hash, alpha_hash


def _selected_action_and_aliases(
    *,
    q: V1PhaseATrueQDDecisionWitnessReceipt,
    source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
) -> tuple[
    V1SampledV7PhysicalActionCustody,
    tuple[V1SampledV7BranchActionAlias, ...],
    V1SignedWorkPackedActionDedupIdentity,
]:
    pair = source_join.action_pair
    try:
        pair.assert_integrity()
        matches = tuple(
            action
            for action in pair.physical_actions
            if action.physical_action_identity_hash
            == q.selected_physical_action_identity_hash
            and action.packed_action_hash == q.selected_packed_action_hash
            and action.resulting_mask_hash == q.selected_resulting_mask_hash
        )
    except Exception as error:
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.PARENT_INTEGRITY,
            f"selected action pair failed replay: {type(error).__name__}: {error}",
        )
    if len(matches) != 1:
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.SELECTED_ACTION_MISSING_OR_AMBIGUOUS,
            "the Q-selected action does not identify exactly one packed custody object",
        )
    selected = matches[0]
    try:
        selected.assert_integrity()
        action = selected.packed_action
        dedup = v1_signed_work_packed_action_dedup_identity(action)
    except Exception as error:
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.PARENT_INTEGRITY,
            f"selected physical custody failed replay: {type(error).__name__}: {error}",
        )
    if (
        selected.dedup_identity != dedup
        or selected.physical_action_identity_hash
        != q.selected_physical_action_identity_hash
        or selected.packed_action_hash != q.selected_packed_action_hash
        or selected.resulting_mask_hash != q.selected_resulting_mask_hash
        or dedup[0] != selected.resulting_mask_packed_sha256
        or dedup[1] != selected.packed_action_hash
        or dedup[2] != selected.resulting_mask_hash
        or dedup[3] != selected.removed_route_index_hash
        or dedup[4] != selected.removed_count
    ):
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.ACTION_IDENTITY_MISMATCH,
            "Q selection and packed custody differ in their complete dedup identity",
        )
    aliases = tuple(
        alias
        for alias in pair.branch_aliases
        if alias.physical_action_ordinal == selected.physical_action_ordinal
    )
    try:
        tuple(alias.assert_integrity() for alias in aliases)
    except Exception as error:
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.PARENT_INTEGRITY,
            f"one selected branch alias failed replay: {type(error).__name__}: {error}",
        )
    if (
        not aliases
        or tuple(alias.branch for alias in aliases) != selected.branch_alias_axis
        or tuple(alias.alias_hash for alias in aliases) != selected.branch_alias_hashes
        or any(
            alias.physical_action_identity_hash
            != selected.physical_action_identity_hash
            or alias.packed_action_hash != dedup[1]
            or alias.resulting_mask_hash != dedup[2]
            or alias.resulting_mask_packed_sha256 != dedup[0]
            or alias.removed_route_index_hash != dedup[3]
            or alias.removed_count != dedup[4]
            for alias in aliases
        )
    ):
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.ALIAS_COVERAGE_INCOMPLETE,
            "selected physical custody and its complete branch-alias partition diverged",
        )
    return selected, aliases, dedup


def _selected_structural_rows(
    *,
    source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
    aliases: tuple[V1SampledV7BranchActionAlias, ...],
    dedup: V1SignedWorkPackedActionDedupIdentity,
) -> tuple[V1SampledV7StructuralActionCrosswalkRow, ...]:
    alias_map = source_join.alias_map
    source_image = alias_map.source_image
    if type(source_image) is not V1SampledV7DSourceImageBindingReceipt:
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.INVALID_TYPE,
            "alias map lost its exact D source-image receipt",
        )
    complete = source_image._complete_crosswalk
    if type(complete) is not V1SampledV7CompleteActionCoordinatePackedCrosswalkReceipt:
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.INVALID_TYPE,
            "D source image lost its exact complete action crosswalk",
        )
    selected_branches = tuple(alias.branch for alias in aliases)
    matching_hashes = set(source_join.matching_structural_row_hashes)
    rows = tuple(
        row
        for row in complete.structural_rows
        if row.structural_row_hash in matching_hashes
        and row.branch in selected_branches
        and row.resource_coordinate_hash == source_join.resource_coordinate_hash
        and row.packed_dedup_identity == dedup
    )
    expected_pairs = {
        (cell_hash, branch)
        for cell_hash in source_join.matching_registered_cell_hashes
        for branch in selected_branches
    }
    observed_pairs = {
        (row.qualification_registered_cell_hash, row.branch) for row in rows
    }
    alias_rows = {
        row.source_structural_row_hash: row for row in alias_map.alias_rows
    }
    if (
        not rows
        or any(type(row) is not V1SampledV7StructuralActionCrosswalkRow for row in rows)
        or observed_pairs != expected_pairs
        or len(rows) != len(expected_pairs)
        or any(row.structural_row_hash not in alias_rows for row in rows)
        or any(
            alias_rows[row.structural_row_hash].packed_dedup_identity != dedup
            or alias_rows[row.structural_row_hash].branch is not row.branch
            for row in rows
        )
    ):
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.ALIAS_COVERAGE_INCOMPLETE,
            "the selected exact dedup identity does not cover every registered cell/branch alias",
        )
    return rows


def _signed_f_branch_evidence(
    *,
    source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
    aliases: tuple[V1SampledV7BranchActionAlias, ...],
    dedup: V1SignedWorkPackedActionDedupIdentity,
) -> tuple[V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt, ...]:
    source_image = source_join.alias_map.source_image
    branch_axis = tuple(source_image._branch_evidence)
    if (
        len(branch_axis) != 2
        or any(
            type(value) is not V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt
            for value in branch_axis
        )
        or tuple(value.branch for value in branch_axis) != _CANONICAL_BRANCHES
    ):
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.SIGNED_F_EVIDENCE_MISSING,
            "D source image lost its canonical exact branch evidence axis",
        )
    by_branch = {value.branch: value for value in branch_axis}
    selected: list[V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt] = []
    for alias in aliases:
        evidence = by_branch.get(alias.branch)
        if evidence is None:
            _refuse(
                V1SampledV7DDecisionStabilityRefusalCode.SIGNED_F_EVIDENCE_MISSING,
                f"selected {alias.branch.value} alias has no corrected-F evidence",
            )
        try:
            evidence.assert_integrity()
            decision = require_v1_block_support_production_witness_decision_v2(
                evidence._decision
            )
            action = evidence.packed_action
            observed_identity = v1_signed_work_packed_action_dedup_identity(action)
        except Exception as error:
            _refuse(
                V1SampledV7DDecisionStabilityRefusalCode.SIGNED_F_SEPARATION_UNSTABLE,
                "selected corrected-F evidence did not replay a positive strict "
                f"separation: {type(error).__name__}: {error}",
            )
        if (
            observed_identity != dedup
            or action.resulting_mask.packed_bits
            != source_join.action_pair.physical_actions[
                alias.physical_action_ordinal - 1  # type: ignore[operator]
            ].packed_action.resulting_mask.packed_bits
            or evidence.preferred_packed_action_hash != dedup[1]
            or evidence.preferred_resulting_mask_hash != dedup[2]
            or evidence.preferred_resulting_mask_packed_sha256 != dedup[0]
            or evidence.preferred_removed_route_index_hash != dedup[3]
            or evidence.preferred_removed_count != dedup[4]
            or decision.preferred_action_hash != dedup[1]
            or evidence.production_witness_decision_hash != decision.decision_hash
        ):
            _refuse(
                V1SampledV7DDecisionStabilityRefusalCode.ACTION_IDENTITY_MISMATCH,
                "corrected-F evidence names a different packed action",
            )
        selected.append(evidence)
    return tuple(selected)


def _alpha_source_coordinates(
    *,
    closure: V1SampledV7DAlphaObservationFamilyClosureReceipt,
    robust: V1SampledV7PhaseARobustShellDecisionReceipt,
    source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
    branch_evidence: tuple[V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt, ...],
) -> tuple[V1SampledFullPopulationCompactSourceReceipt, V1SampledV7DAlphaMeasurementOwnerBindingReceipt]:
    observations = (
        closure._corrected_a_observation,
        closure._signed_f_observation,
        closure._literal_j_observation,
    )
    if (
        type(observations[0]) is not V1SampledV7CorrectedAObservationBindingReceipt
        or type(observations[1]) is not V1SampledV7SignedFObservationBindingReceipt
        or type(observations[2]) is not V1SampledV7LiteralJObservationBindingReceipt
    ):
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.ALPHA_FAMILY_MISMATCH,
            "A/F/J closure lost one exact observation parent",
        )
    a_observation = observations[0]
    f_observation = observations[1]
    j_observation = observations[2]
    owner = j_observation._alpha_registration._owner_binding
    if (
        type(owner) is not V1SampledV7DAlphaMeasurementOwnerBindingReceipt
        or a_observation._owner_binding is not owner
        or f_observation._alpha_registration._owner_binding is not owner
    ):
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.ALPHA_FAMILY_MISMATCH,
            "A/F/J closure no longer retains one exact alpha-owner instance",
        )
    source_image = source_join.alias_map.source_image
    source_handle = source_image._common_source
    if type(source_handle) is not V1SampledFullPopulationCompactSourceHandle:
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.ALPHA_FAMILY_MISMATCH,
            "D source image lost its exact live common-population source handle",
        )
    try:
        source = source_handle.receipt
        source_hash = source.assert_integrity()
    except Exception as error:
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.PARENT_INTEGRITY,
            f"common D source failed replay: {type(error).__name__}: {error}",
        )
    expected_look = tuple(
        source.d1_stratum_counts[index] + source.final_dw2_stratum_counts[index]
        for index in (0, 1)
    )
    source_pairs = (
        (owner.generation_frame_receipt_hash, source.generation_frame_hash),
        (owner.discovery_stream_receipt_hash, source.discovery_stream_hash),
        (owner.residual_population_measure_design_hash, source.residual_measure_design_hash),
        (owner.suffix_registration_hash, source.suffix_registration_hash),
        (owner.model_identity_hash, source.model_identity_hash),
        (owner.chart_identity_hash, source.chart_identity_hash),
        (owner.current_mask_hash, source.current_mask_hash),
        (owner.proposal_generation, source.proposal_generation),
        (owner.physical_route_axis_hash, source.route_axis_hash),
        (
            owner._measurement_owner.measurement_source_lineage.route_id_axis_hash,
            source.route_id_axis_hash,
        ),
        (owner.route_count, source.route_count),
        (
            owner.corrected_evidence_stream_source_hash,
            source.corrected_evidence_stream_source_hash,
        ),
        (owner.literal_j_stream_source_hash, source.literal_j_stream_source_hash),
        (
            owner.same_capture_vjp_source_hash,
            source.residual_same_capture_vjp_source_hash,
        ),
        (owner.population_stratum_counts, source.population_stratum_counts),
        (owner.registered_look_stratum_counts, expected_look),
        (closure.population_stratum_counts, source.population_stratum_counts),
        (closure.registered_look_stratum_counts, expected_look),
        (sum(expected_look), source.context_count),
        (owner.historical_owner_alpha_ledger_hash, source.alpha_ledger_id),
        (owner.historical_owner_alpha_coordinate_hash, source.alpha_coordinate_id),
        (owner.historical_owner_alpha_lineage_hash, source.alpha_lineage_hash),
        (robust.inherited_alpha_ledger_hash, source.alpha_ledger_id),
        (robust.inherited_alpha_coordinate_hash, source.alpha_coordinate_id),
        (robust.inherited_alpha_lineage_hash, source.alpha_lineage_hash),
        (robust.inherited_familywise_alpha, owner.historical_owner_familywise_alpha),
        (f_observation.legacy_alpha_ledger_id, source.alpha_ledger_id),
        (f_observation.legacy_alpha_coordinate_id, source.alpha_coordinate_id),
        (source_image.common_source_receipt_hash, source_hash),
        (source_image.common_source_replay_hash, source.source_replay_hash),
        (source_image.parent_mask_hash, source.current_mask_hash),
        (source_image.route_axis_hash, source.route_axis_hash),
        (source_image.route_count, source.route_count),
    )
    if any(left != right for left, right in source_pairs):
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.ALPHA_FAMILY_MISMATCH,
            "consumed A/F/J family and Q/F source graph do not name one D look",
        )
    complete_branch_axis = tuple(source_image._branch_evidence)
    if any(
        evidence._source is not source
        or evidence.source_receipt_hash != source_hash
        or evidence.source_replay_hash != source.source_replay_hash
        or evidence.alpha_ledger_id != source.alpha_ledger_id
        or evidence.alpha_coordinate_id != source.alpha_coordinate_id
        or evidence.alpha_lineage_hash != source.alpha_lineage_hash
        for evidence in complete_branch_axis
    ):
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.ALPHA_FAMILY_MISMATCH,
            "branch corrected-F evidence is not the exact common A/F/J D source",
        )
    if any(
        not any(evidence is candidate for candidate in complete_branch_axis)
        for evidence in branch_evidence
    ):
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.SOURCE_MISMATCH,
            "derived selected F evidence left the source image branch axis",
        )
    return source, owner


def _derive_decision_coordinates(
    *,
    true_q_decision_witness: V1PhaseATrueQDDecisionWitnessReceipt,
    robust_shell_decision: V1SampledV7PhaseARobustShellDecisionReceipt,
    action_pair_alias_source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
    hard_face_margin_certificate: V1PhaseAGlobalQHardFaceMarginCertificateResult,
    alpha_observation_family_closure: V1SampledV7DAlphaObservationFamilyClosureReceipt,
) -> tuple[
    dict[str, object],
    tuple[V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt, ...],
]:
    q_hash, robust_hash, source_join_hash, hard_hash, alpha_hash = _assert_exact_parents(
        true_q_decision_witness=true_q_decision_witness,
        robust_shell_decision=robust_shell_decision,
        action_pair_alias_source_join=action_pair_alias_source_join,
        hard_face_margin_certificate=hard_face_margin_certificate,
        alpha_observation_family_closure=alpha_observation_family_closure,
    )
    q = true_q_decision_witness
    hard = hard_face_margin_certificate.receipt
    if (
        q.q_side_d_decision_component is not True
        or q.selected_action_matches_best_certified_hard_face_vertex is not True
        or q.signed_F_decision_component_still_required is not True
        or q.A_F_J_alpha_family_closure_still_required is not True
        or q.independent_C_still_required is not True
        or q.d_stability_authority is not False
        or hard.status
        is not V1PhaseAGlobalQHardFaceMarginCertificateStatus.CERTIFIED_RESOURCE_COMPETITIVE_HARD_OUTSIDE_FACE
        or hard.all_resource_competitive_hard_actions_share_outside_membership
        is not True
    ):
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.Q_COMPONENT_UNCERTIFIED,
            "true-Q input is not the exact certified Q-side D component",
        )
    if (
        q.production_cuda_execution is not True
        or q.private_cpu_reference_execution is not False
        or hard.production_cuda_execution is not True
        or hard.private_cpu_reference_execution is not False
    ):
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.PRODUCTION_CUDA_REQUIRED,
            "private CPU true-Q evidence is semantic reference only and cannot authorize D stability",
        )
    pair = action_pair_alias_source_join.action_pair
    if (
        robust_shell_decision.status
        is not V1SampledV7PhaseARobustDecisionStatus.STABLE_PHYSICAL_ACTION_PREMISE
        or q.robust_shell_decision_receipt_hash != robust_hash
        or q.action_pair_alias_source_join_receipt_hash != source_join_hash
        or q.action_pair_hash != pair.pair_hash
        or q.hard_face_margin_receipt_hash != hard_hash
        or hard.action_pair_alias_source_join_receipt_hash != source_join_hash
        or hard.robust_shell_decision_receipt_hash != robust_hash
        or robust_shell_decision.action_pair_alias_source_join_receipt_hash
        != source_join_hash
        or robust_shell_decision.action_pair_hash != pair.pair_hash
        or q.parent_mask_hash != pair.parent_mask_hash
        or q.route_axis_hash != pair.route_axis_hash
        or q.route_count != pair.route_count
    ):
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.SOURCE_MISMATCH,
            "Q, robust-shell, hard-face, and action-pair sources do not replay one graph",
        )
    selected, aliases, dedup = _selected_action_and_aliases(
        q=q,
        source_join=action_pair_alias_source_join,
    )
    structural_rows = _selected_structural_rows(
        source_join=action_pair_alias_source_join,
        aliases=aliases,
        dedup=dedup,
    )
    branch_evidence = _signed_f_branch_evidence(
        source_join=action_pair_alias_source_join,
        aliases=aliases,
        dedup=dedup,
    )
    common_source, alpha_owner = _alpha_source_coordinates(
        closure=alpha_observation_family_closure,
        robust=robust_shell_decision,
        source_join=action_pair_alias_source_join,
        branch_evidence=branch_evidence,
    )
    source_image = action_pair_alias_source_join.alias_map.source_image
    if (
        common_source.current_mask_hash != q.parent_mask_hash
        or common_source.route_axis_hash != q.route_axis_hash
        or common_source.route_id_axis_hash != q.route_id_axis_hash
        or common_source.route_count != q.route_count
        or selected.removed_count != q.selected_total_count
        or robust_shell_decision.selected_physical_action_identity_hash
        != selected.physical_action_identity_hash
    ):
        _refuse(
            V1SampledV7DDecisionStabilityRefusalCode.SOURCE_MISMATCH,
            "selected action, common D source, and Q witness axes do not close",
        )
    return (
        {
            "true_q_d_decision_witness_receipt_hash": q_hash,
            "robust_shell_decision_receipt_hash": robust_hash,
            "hard_face_margin_certificate_receipt_hash": hard_hash,
            "action_pair_alias_source_join_receipt_hash": source_join_hash,
            "action_pair_hash": pair.pair_hash,
            "alias_map_receipt_hash": action_pair_alias_source_join.alias_map_receipt_hash,
            "source_image_receipt_hash": source_image.receipt_hash,
            "common_d_source_receipt_hash": common_source.receipt_hash,
            "common_d_source_replay_hash": common_source.source_replay_hash,
            "alpha_observation_family_closure_receipt_hash": alpha_hash,
            "alpha_owner_binding_receipt_hash": (
                alpha_observation_family_closure.owner_binding_receipt_hash
            ),
            "alpha_family_plan_hash": alpha_observation_family_closure.alpha_family_plan_hash,
            "alpha_family_registry_receipt_hash": alpha_observation_family_closure.alpha_family_registry_receipt_hash,
            "alpha_head_cas_receipt_hash": alpha_observation_family_closure.alpha_head_cas_receipt_hash,
            "published_alpha_ledger_head_hash": alpha_observation_family_closure.published_ledger_head_hash,
            "registered_d_prefix_axis_hash": alpha_observation_family_closure.registered_prefix_axis_hash,
            "registered_look_index": alpha_observation_family_closure.registered_look_index,
            "registered_look_stratum_counts": alpha_observation_family_closure.registered_look_stratum_counts,
            "population_stratum_counts": alpha_observation_family_closure.population_stratum_counts,
            "a_f_j_cell_role_axis": alpha_observation_family_closure.cell_role_axis,
            "a_f_j_cell_id_axis": alpha_observation_family_closure.cell_id_axis,
            "a_f_j_cell_hash_axis": alpha_observation_family_closure.cell_hash_axis,
            "a_f_j_cell_scope_hash_axis": alpha_observation_family_closure.cell_scope_hash_axis,
            "a_f_j_cell_alpha_axis": alpha_observation_family_closure.cell_alpha_axis,
            "a_f_j_observation_receipt_hash_axis": alpha_observation_family_closure.observation_receipt_hash_axis,
            "selected_physical_action_identity_hash": selected.physical_action_identity_hash,
            "selected_packed_dedup_identity": dedup,
            "selected_packed_action_hash": dedup[1],
            "selected_resulting_mask_hash": dedup[2],
            "selected_resulting_mask_packed_sha256": dedup[0],
            "selected_removed_route_index_hash": dedup[3],
            "selected_removed_count": dedup[4],
            "parent_mask_hash": q.parent_mask_hash,
            "route_axis_hash": q.route_axis_hash,
            "route_id_axis_hash": q.route_id_axis_hash,
            "route_count": q.route_count,
            "selected_total_count": q.selected_total_count,
            "selected_outside_count": q.selected_outside_count,
            "selected_shell_count": q.selected_shell_count,
            "selected_branch_axis": tuple(alias.branch for alias in aliases),
            "selected_branch_alias_hashes": tuple(alias.alias_hash for alias in aliases),
            "selected_structural_row_hashes": tuple(row.structural_row_hash for row in structural_rows),
            "selected_signed_f_evidence_receipt_hashes": tuple(evidence.receipt_hash for evidence in branch_evidence),
            "selected_signed_f_decision_hashes": tuple(evidence.production_witness_decision_hash for evidence in branch_evidence),
            "selected_strict_full_population_f_lower_axis": tuple(float(evidence.strict_full_population_F_lower_outward) for evidence in branch_evidence),
            "q_best_hard_face_resource_lower": q.best_hard_face_resource_lower,
            "q_structural_dual_resource_upper": q.structural_dual_resource_upper,
            "q_dual_gap_upper": q.dual_gap_upper,
            "q_strict_outside_bit_margin_lower": q.strict_outside_bit_margin_lower,
        },
        branch_evidence,
    )


@dataclass(frozen=True, slots=True)
class V1SampledV7DDecisionStabilityReceipt:
    true_q_d_decision_witness_receipt_hash: str
    robust_shell_decision_receipt_hash: str
    hard_face_margin_certificate_receipt_hash: str
    action_pair_alias_source_join_receipt_hash: str
    action_pair_hash: str
    alias_map_receipt_hash: str
    source_image_receipt_hash: str
    common_d_source_receipt_hash: str
    common_d_source_replay_hash: str
    alpha_observation_family_closure_receipt_hash: str
    alpha_owner_binding_receipt_hash: str
    alpha_family_plan_hash: str
    alpha_family_registry_receipt_hash: str
    alpha_head_cas_receipt_hash: str
    published_alpha_ledger_head_hash: str
    registered_d_prefix_axis_hash: str
    registered_look_index: int
    registered_look_stratum_counts: tuple[int, int]
    population_stratum_counts: tuple[int, int]
    a_f_j_cell_role_axis: tuple[str, str, str]
    a_f_j_cell_id_axis: tuple[str, str, str]
    a_f_j_cell_hash_axis: tuple[str, str, str]
    a_f_j_cell_scope_hash_axis: tuple[str, str, str]
    a_f_j_cell_alpha_axis: tuple[float, float, float]
    a_f_j_observation_receipt_hash_axis: tuple[str, str, str]
    selected_physical_action_identity_hash: str
    selected_packed_dedup_identity: V1SignedWorkPackedActionDedupIdentity
    selected_packed_action_hash: str
    selected_resulting_mask_hash: str
    selected_resulting_mask_packed_sha256: str
    selected_removed_route_index_hash: str
    selected_removed_count: int
    parent_mask_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    selected_total_count: int
    selected_outside_count: int
    selected_shell_count: int
    selected_branch_axis: tuple[V1RelaxedKKTBranch, ...]
    selected_branch_alias_hashes: tuple[str, ...]
    selected_structural_row_hashes: tuple[str, ...]
    selected_signed_f_evidence_receipt_hashes: tuple[str, ...]
    selected_signed_f_decision_hashes: tuple[str, ...]
    selected_strict_full_population_f_lower_axis: tuple[float, ...]
    q_best_hard_face_resource_lower: float
    q_structural_dual_resource_upper: float
    q_dual_gap_upper: float
    q_strict_outside_bit_margin_lower: float
    q_component_certified: bool = True
    signed_f_component_certified: bool = True
    nonrecycling_a_f_j_family_closed: bool = True
    exact_packed_action_and_alias_image_joined: bool = True
    production_cuda_q_required: bool = True
    d_decision_stability_authority: bool = True
    independent_C_still_required: bool = True
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    action_execution_authority: bool = False
    mask_commit_authority: bool = False
    relinearization_authority: bool = False
    runtime_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = SAMPLED_V7_D_DECISION_STABILITY_ALGORITHM
    authority_scope: str = SAMPLED_V7_D_DECISION_STABILITY_AUTHORITY
    schema: str = SAMPLED_V7_D_DECISION_STABILITY_SCHEMA
    _true_q_decision_witness: V1PhaseATrueQDDecisionWitnessReceipt = field(repr=False, compare=False, default=None)  # type: ignore[assignment]
    _robust_shell_decision: V1SampledV7PhaseARobustShellDecisionReceipt = field(repr=False, compare=False, default=None)  # type: ignore[assignment]
    _action_pair_alias_source_join: V1SampledV7ActionPairAliasSourceJoinReceipt = field(repr=False, compare=False, default=None)  # type: ignore[assignment]
    _hard_face_margin_certificate: V1PhaseAGlobalQHardFaceMarginCertificateResult = field(repr=False, compare=False, default=None)  # type: ignore[assignment]
    _alpha_observation_family_closure: V1SampledV7DAlphaObservationFamilyClosureReceipt = field(repr=False, compare=False, default=None)  # type: ignore[assignment]
    _selected_signed_f_evidence: tuple[V1SampledV7BranchProductionUnjoinedFQEvidenceReceipt, ...] = field(repr=False, compare=False, default=())
    _issuance_token: InitVar[object | None] = None
    _sealed_hash: str | None = field(init=False, default=None, repr=False)
    _parent_identity_axis: tuple[int, int, int, int, int] = field(init=False, default=(0, 0, 0, 0, 0), repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError("sampled-v7 D-stability receipts require their issuer")
        object.__setattr__(
            self,
            "_parent_identity_axis",
            (
                id(self._true_q_decision_witness),
                id(self._robust_shell_decision),
                id(self._action_pair_alias_source_join),
                id(self._hard_face_margin_certificate),
                id(self._alpha_observation_family_closure),
            ),
        )
        self._validate_static()
        observed = self._compute_hash()
        object.__setattr__(self, "_sealed_hash", observed)

    def _validate_static(self) -> None:
        for item in fields(self):
            if item.name.startswith("_"):
                continue
            value = getattr(self, item.name)
            if item.name.endswith("_hash") and item.name not in {
                "selected_packed_dedup_identity"
            }:
                _identity(value, name=item.name)
        for name in (
            "registered_look_index",
            "selected_removed_count",
            "route_count",
            "selected_total_count",
        ):
            _positive_int(getattr(self, name), name=name)
        _count_pair(
            self.registered_look_stratum_counts,
            name="registered look stratum counts",
        )
        populations = _count_pair(
            self.population_stratum_counts,
            name="population stratum counts",
        )
        if any(
            self.registered_look_stratum_counts[index] > populations[index]
            for index in (0, 1)
        ):
            _refuse(
                V1SampledV7DDecisionStabilityRefusalCode.SOURCE_MISMATCH,
                "registered D look exceeds its population",
            )
        if (
            self.selected_removed_count != self.selected_total_count
            or self.selected_total_count
            != self.selected_outside_count + self.selected_shell_count
            or type(self.selected_outside_count) is not int
            or self.selected_outside_count < 0
            or type(self.selected_shell_count) is not int
            or self.selected_shell_count < 0
        ):
            _refuse(
                V1SampledV7DDecisionStabilityRefusalCode.SOURCE_MISMATCH,
                "selected action counts do not close",
            )
        if (
            type(self.selected_packed_dedup_identity) is not tuple
            or len(self.selected_packed_dedup_identity) != 5
            or self.selected_packed_dedup_identity
            != (
                self.selected_resulting_mask_packed_sha256,
                self.selected_packed_action_hash,
                self.selected_resulting_mask_hash,
                self.selected_removed_route_index_hash,
                self.selected_removed_count,
            )
        ):
            _refuse(
                V1SampledV7DDecisionStabilityRefusalCode.ACTION_IDENTITY_MISMATCH,
                "receipt lost the complete selected packed-action identity",
            )
        axes = (
            self.a_f_j_cell_role_axis,
            self.a_f_j_cell_id_axis,
            self.a_f_j_cell_hash_axis,
            self.a_f_j_cell_scope_hash_axis,
            self.a_f_j_cell_alpha_axis,
            self.a_f_j_observation_receipt_hash_axis,
        )
        if (
            any(type(axis) is not tuple or len(axis) != 3 for axis in axes)
            or self.a_f_j_cell_role_axis
            != ("corrected_A_order", "signed_F_witness", "literal_J_sampling")
            or any(
                type(value) is not str or not value
                for axis in (
                    self.a_f_j_cell_role_axis,
                    self.a_f_j_cell_id_axis,
                    self.a_f_j_cell_hash_axis,
                    self.a_f_j_cell_scope_hash_axis,
                    self.a_f_j_observation_receipt_hash_axis,
                )
                for value in axis
            )
            or any(
                type(value) is not float or not math.isfinite(value) or value < 0.0
                for value in self.a_f_j_cell_alpha_axis
            )
            or not self.selected_branch_axis
            or any(
                type(branch) is not V1RelaxedKKTBranch
                for branch in self.selected_branch_axis
            )
            or tuple(branch for branch in _CANONICAL_BRANCHES if branch in self.selected_branch_axis)
            != self.selected_branch_axis
            or not (
                len(self.selected_branch_axis)
                == len(self.selected_branch_alias_hashes)
                == len(self.selected_signed_f_evidence_receipt_hashes)
                == len(self.selected_signed_f_decision_hashes)
                == len(self.selected_strict_full_population_f_lower_axis)
            )
            or any(
                type(value) is not float or not math.isfinite(value) or value <= 0.0
                for value in self.selected_strict_full_population_f_lower_axis
            )
            or not self.selected_structural_row_hashes
            or any(
                type(value) is not str or not value
                for axis in (
                    self.selected_branch_alias_hashes,
                    self.selected_structural_row_hashes,
                    self.selected_signed_f_evidence_receipt_hashes,
                    self.selected_signed_f_decision_hashes,
                )
                for value in axis
            )
        ):
            _refuse(
                V1SampledV7DDecisionStabilityRefusalCode.AUTHORITY_VIOLATION,
                "D-stability evidence axes are incomplete or non-strict",
            )
        q_scalars = (
            self.q_best_hard_face_resource_lower,
            self.q_structural_dual_resource_upper,
            self.q_dual_gap_upper,
            self.q_strict_outside_bit_margin_lower,
        )
        if (
            any(type(value) is not float or not math.isfinite(value) for value in q_scalars)
            or self.q_best_hard_face_resource_lower < 0.0
            or self.q_structural_dual_resource_upper < 0.0
            or self.q_dual_gap_upper < 0.0
            or self.q_strict_outside_bit_margin_lower <= 0.0
            or not self.q_dual_gap_upper
            < self.q_strict_outside_bit_margin_lower
        ):
            _refuse(
                V1SampledV7DDecisionStabilityRefusalCode.Q_COMPONENT_UNCERTIFIED,
                "receipt lost the strict global true-Q hard-face inequality",
            )
        positive = (
            self.q_component_certified,
            self.signed_f_component_certified,
            self.nonrecycling_a_f_j_family_closed,
            self.exact_packed_action_and_alias_image_joined,
            self.production_cuda_q_required,
            self.d_decision_stability_authority,
            self.independent_C_still_required,
        )
        closed = (
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.action_execution_authority,
            self.mask_commit_authority,
            self.relinearization_authority,
            self.runtime_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool or value is not True for value in positive)
            or any(type(value) is not bool or value is not False for value in closed)
            or self.algorithm != SAMPLED_V7_D_DECISION_STABILITY_ALGORITHM
            or self.authority_scope != SAMPLED_V7_D_DECISION_STABILITY_AUTHORITY
            or self.schema != SAMPLED_V7_D_DECISION_STABILITY_SCHEMA
        ):
            _refuse(
                V1SampledV7DDecisionStabilityRefusalCode.AUTHORITY_VIOLATION,
                "D-stability receipt crossed its D-only authority boundary",
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
        return _payload_hash(self._payload(), role="receipt")

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if self._parent_identity_axis != (
            id(self._true_q_decision_witness),
            id(self._robust_shell_decision),
            id(self._action_pair_alias_source_join),
            id(self._hard_face_margin_certificate),
            id(self._alpha_observation_family_closure),
        ):
            _refuse(
                V1SampledV7DDecisionStabilityRefusalCode.MUTATED_CUSTODY,
                "one live D-stability parent was substituted",
            )
        public, evidence = _derive_decision_coordinates(
            true_q_decision_witness=self._true_q_decision_witness,
            robust_shell_decision=self._robust_shell_decision,
            action_pair_alias_source_join=self._action_pair_alias_source_join,
            hard_face_margin_certificate=self._hard_face_margin_certificate,
            alpha_observation_family_closure=self._alpha_observation_family_closure,
        )
        if (
            evidence != self._selected_signed_f_evidence
            or any(getattr(self, name) != value for name, value in public.items())
        ):
            _refuse(
                V1SampledV7DDecisionStabilityRefusalCode.MUTATED_CUSTODY,
                "D-stability public image or selected F custody no longer replays",
            )
        self._validate_static()
        observed = self._compute_hash()
        if observed != self._sealed_hash or (
            expected_hash is not None and observed != expected_hash
        ):
            _refuse(
                V1SampledV7DDecisionStabilityRefusalCode.MUTATED_CUSTODY,
                "D-stability receipt changed after sealing",
            )
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    @property
    def selected_packed_action(self) -> V1PackedRemovalAction:
        """Return the exact retained selected packed action after full replay."""

        self.assert_integrity()
        pair = self._action_pair_alias_source_join.action_pair
        matches = tuple(
            value
            for value in pair.physical_actions
            if value.dedup_identity == self.selected_packed_dedup_identity
            and value.packed_action_hash == self.selected_packed_action_hash
            and value.resulting_mask_hash == self.selected_resulting_mask_hash
            and value.resulting_mask_packed_sha256
            == self.selected_resulting_mask_packed_sha256
            and value.removed_route_index_hash
            == self.selected_removed_route_index_hash
            and value.removed_count == self.selected_removed_count
        )
        if len(matches) != 1:
            _refuse(
                V1SampledV7DDecisionStabilityRefusalCode.MUTATED_CUSTODY,
                "selected packed action custody is no longer unique",
            )
        return matches[0].packed_action

    def __reduce__(self) -> NoReturn:
        raise TypeError("sampled-v7 D-stability receipts are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("sampled-v7 D-stability receipts are nonserializable")


def issue_v1_sampled_v7_d_decision_stability(
    *,
    true_q_decision_witness: V1PhaseATrueQDDecisionWitnessReceipt,
    robust_shell_decision: V1SampledV7PhaseARobustShellDecisionReceipt,
    action_pair_alias_source_join: V1SampledV7ActionPairAliasSourceJoinReceipt,
    hard_face_margin_certificate: V1PhaseAGlobalQHardFaceMarginCertificateResult,
    alpha_observation_family_closure: V1SampledV7DAlphaObservationFamilyClosureReceipt,
) -> V1SampledV7DDecisionStabilityReceipt:
    """Issue the sole D-stability authority from exact Q/F/alpha parents."""

    public, evidence = _derive_decision_coordinates(
        true_q_decision_witness=true_q_decision_witness,
        robust_shell_decision=robust_shell_decision,
        action_pair_alias_source_join=action_pair_alias_source_join,
        hard_face_margin_certificate=hard_face_margin_certificate,
        alpha_observation_family_closure=alpha_observation_family_closure,
    )
    return V1SampledV7DDecisionStabilityReceipt(
        **public,  # type: ignore[arg-type]
        _true_q_decision_witness=true_q_decision_witness,
        _robust_shell_decision=robust_shell_decision,
        _action_pair_alias_source_join=action_pair_alias_source_join,
        _hard_face_margin_certificate=hard_face_margin_certificate,
        _alpha_observation_family_closure=alpha_observation_family_closure,
        _selected_signed_f_evidence=evidence,
        _issuance_token=_RECEIPT_TOKEN,
    )


__all__ = (
    "SAMPLED_V7_D_DECISION_STABILITY_ALGORITHM",
    "SAMPLED_V7_D_DECISION_STABILITY_AUTHORITY",
    "SAMPLED_V7_D_DECISION_STABILITY_SCHEMA",
    "V1SampledV7DDecisionStabilityReceipt",
    "V1SampledV7DDecisionStabilityRefusal",
    "V1SampledV7DDecisionStabilityRefusalCode",
    "issue_v1_sampled_v7_d_decision_stability",
)
