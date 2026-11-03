"""Neutral pre-outcome F/Q artifact owner for a geometric v1 ladder.

The historical F/Q artifact graph is rooted in a response-exactification
freeze and its score-zero-regret decision scope.  A sampled conditional-
response-geometry selector has a different composition authority.  This
module binds that selector's immutable packed ladder to the already generic
route envelope, endpoint envelope, alpha, common-panel, and certification
objects without relabeling any endpoint as score-resolved.

This is deliberately a binding joint, not an acquisition engine.  Every
component must already have been issued before F/Q outcomes.  The resulting
object grants no F/Q execution, BCE, candidate-freeze, mask, validation, or
terminal authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from .contracts import content_hash
from .modern_v1_composition_scope import (
    SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND,
    V1CompositionDecisionScopeReceipt,
)
from .modern_v1_ladder_statistics import (
    V1FrozenCandidateLadder,
    V1FrozenLadderCertificationDesign,
)
from .modern_v1_proposal_statistics import (
    V1CorrectedAttributionEnvelopeCertificate,
)
from .modern_v1_response_exactification import (
    V1CancellationAwareEndpointEnvelopeBundle,
    V1FQAlphaFamilyAllocationReceipt,
    V1FQCommonCertificationPanelReceipt,
)
from .modern_v1_sampled_geometric_freeze import V1SampledGeometricFreeze


COMPOSITION_FQ_ARTIFACTS_SCHEMA: Final = (
    "partition_free_v1_frozen_composition_fq_artifacts_v1"
)
COMPOSITION_FQ_ARTIFACTS_HASH_PREFIX: Final = (
    "partition-free-v1-frozen-composition-fq-artifacts-"
)


def _coordinates_match(left: object, right: object) -> bool:
    return all(
        getattr(left, name) == getattr(right, name)
        for name in (
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "proposal_generation",
            "route_count",
            "route_axis_hash",
        )
    )


@dataclass(frozen=True, slots=True)
class V1CompositionFQArtifacts:
    """Complete proposal-only F/Q graph for one geometric composition."""

    freeze: V1SampledGeometricFreeze = field(repr=False, compare=False)
    decision_scope: V1CompositionDecisionScopeReceipt
    physical_fq_envelope_certificate: (
        V1CorrectedAttributionEnvelopeCertificate
    ) = field(repr=False, compare=False)
    endpoint_envelope_bundle: V1CancellationAwareEndpointEnvelopeBundle = field(
        repr=False,
        compare=False,
    )
    fq_alpha_allocation: V1FQAlphaFamilyAllocationReceipt
    common_panel_receipt: V1FQCommonCertificationPanelReceipt
    certification_design: V1FrozenLadderCertificationDesign = field(
        repr=False,
        compare=False,
    )
    issued_before_fq_outcomes: bool = True
    composition_frozen_before_fq_outcomes: bool = True
    fq_outcomes_observed: bool = False
    fq_execution_authority: bool = False
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    terminal_validation_access_authority: bool = False
    terminal_authority: bool = False
    schema: str = COMPOSITION_FQ_ARTIFACTS_SCHEMA
    _artifacts_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.freeze) is not V1SampledGeometricFreeze:
            raise TypeError("composition F/Q artifacts require the geometric freeze")
        if type(self.decision_scope) is not V1CompositionDecisionScopeReceipt:
            raise TypeError("composition F/Q artifacts require the neutral scope")
        if (
            type(self.physical_fq_envelope_certificate)
            is not V1CorrectedAttributionEnvelopeCertificate
        ):
            raise TypeError("composition F/Q artifacts require the physical envelope")
        if (
            type(self.endpoint_envelope_bundle)
            is not V1CancellationAwareEndpointEnvelopeBundle
        ):
            raise TypeError("composition F/Q artifacts require the endpoint bundle")
        if type(self.fq_alpha_allocation) is not V1FQAlphaFamilyAllocationReceipt:
            raise TypeError("composition F/Q artifacts require the F/Q alpha receipt")
        if (
            type(self.common_panel_receipt)
            is not V1FQCommonCertificationPanelReceipt
        ):
            raise TypeError("composition F/Q artifacts require the common-panel receipt")
        if type(self.certification_design) is not V1FrozenLadderCertificationDesign:
            raise TypeError("composition F/Q artifacts require the certification design")

        self.freeze.assert_integrity()
        scope_hash = self.decision_scope.assert_integrity()
        self.physical_fq_envelope_certificate.assert_integrity()
        self.endpoint_envelope_bundle.assert_integrity()
        alpha_hash = self.fq_alpha_allocation.assert_integrity()
        self.common_panel_receipt.assert_integrity()
        self.certification_design.assert_integrity()

        freeze = self.freeze
        identity = freeze.identity
        replay = freeze.replay_receipt
        ladder = freeze.ladder
        scope = self.decision_scope
        alpha = self.fq_alpha_allocation
        common = self.common_panel_receipt
        design = self.certification_design
        physical = self.physical_fq_envelope_certificate
        bundle = self.endpoint_envelope_bundle

        if scope.composition_kind != (
            SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND
        ):
            raise ValueError("composition F/Q artifacts require geometry composition")
        scope_graph = (
            (scope.discovery_score_registration_hash, identity.score_discovery_registration_hash),
            (scope.discovery_score_design_hash, identity.score_discovery_design_hash),
            (scope.discovery_score_look_receipt_hash, identity.score_discovery_look_receipt_hash),
            (scope.discovery_score_look_index, identity.score_discovery_look_index),
            (
                scope.maximum_registered_discovery_score_looks,
                identity.maximum_registered_score_looks,
            ),
            (scope.sampled_geometry_state_hash, identity.sampled_geometry_state_hash),
            (scope.sampled_geometry_result_hash, identity.sampled_geometry_result_hash),
            (scope.sampled_geometry_assembly_hash, identity.sampled_geometry_assembly_hash),
            (scope.sampled_selector_problem_hash, freeze.selector_problem.problem_hash),
            (scope.sampled_selector_policy_hash, freeze.selector_policy.policy_hash),
            (
                scope.sampled_selector_receipt_hash,
                freeze.selector_result.receipt.receipt_hash,
            ),
            (scope.sampled_selector_result_hash, freeze.selector_result.result_hash),
            (
                scope.sampled_proposal_position_certificate_hash,
                identity.sampled_proposal_position_certificate_hash,
            ),
            (scope.composition_freeze_identity_hash, identity.identity_hash),
            (scope.composition_freeze_replay_receipt_hash, replay.receipt_hash),
            (scope.frozen_candidate_ladder_hash, ladder.ladder_hash),
            (scope.endpoint_action_hashes, ladder.action_hashes),
            (scope.endpoint_new_removal_counts, ladder.endpoint_removed_counts),
            (scope.current_mask_hash, ladder.current_mask_hash),
            (scope.proposal_generation, ladder.proposal_generation),
            (scope.route_count, ladder.route_count),
            (scope.physical_route_axis_hash, ladder.route_axis_hash),
            (scope.ordered_route_axis_hash, identity.ordered_route_axis_hash),
        )
        if any(left != right for left, right in scope_graph):
            raise ValueError("composition scope differs from the geometric freeze")

        if (
            not _coordinates_match(ladder, physical)
            or bundle.ladder_hash != ladder.ladder_hash
            or bundle.action_drive_certificate.ladder_hash != ladder.ladder_hash
            or bundle.action_drive_certificate.endpoint_action_hashes
            != ladder.action_hashes
            or bundle.route_sum_fallback_certificate.ladder_hash
            != ladder.ladder_hash
            or bundle.route_sum_fallback_certificate.endpoint_action_hashes
            != ladder.action_hashes
        ):
            raise ValueError("composition F/Q envelope graph differs from the ladder")

        if (
            alpha.proposal_generation != ladder.proposal_generation
            or alpha.score_order_look_index != scope.discovery_score_look_index
            or alpha.maximum_fq_ladder_redesigns
            != scope.maximum_registered_discovery_score_looks
            or alpha.fq_ladder_redesign_generation
            != scope.discovery_score_look_index - 1
        ):
            raise ValueError("composition F/Q alpha belongs to another discovery look")

        if (
            common.proposal_generation != ladder.proposal_generation
            or common.active_score_order_look_index
            != scope.discovery_score_look_index
            or common.maximum_fq_ladder_redesigns
            != scope.maximum_registered_discovery_score_looks
            or common.proposal_order_design_hash
            != scope.discovery_score_design_hash
            or common.score_order_registration_hash
            != scope.discovery_score_registration_hash
            or common.score_order_look_receipt_hash
            != scope.discovery_score_look_receipt_hash
            or common.decision_scope_receipt_hash != scope_hash
            or common.fq_alpha_allocation_receipt_hash != alpha_hash
            or common.fq_redesign_alpha_schedule_hash
            != alpha.fq_redesign_alpha_schedule_hash
            or common.fq_redesign_alpha_allocations
            != alpha.fq_redesign_alpha_allocations
            or common.proposal_order_look_alpha_weights
            != alpha.proposal_order_look_alpha_weights
        ):
            raise ValueError("composition F/Q common-panel graph does not close")

        if (
            design.ladder.ladder_hash != ladder.ladder_hash
            or design.envelope_certificate.assert_integrity()
            != physical.assert_integrity()
            or design.endpoint_envelope_certificate is None
            or design.endpoint_envelope_certificate.assert_integrity()
            != bundle.action_drive_certificate.assert_integrity()
            or design.trajectory_alpha_ledger_hash != alpha_hash
            or design.trajectory_alpha_ledger_generation
            != ladder.proposal_generation
            or design.trajectory_familywise_alpha != alpha.fq_familywise_alpha
            or design.trajectory_alpha_committed_before_generation
            != alpha.fq_alpha_committed_before_generation
            or design.generation_alpha_allocation
            != alpha.fq_generation_alpha_allocation
            or design.ladder_redesign_generation
            != alpha.fq_ladder_redesign_generation
            or design.generation_alpha_committed_before_redesign
            != alpha.fq_alpha_committed_before_redesign
            or design.redesign_alpha_allocation
            != alpha.fq_redesign_alpha_allocation
            or common.certification_context_population_hash
            != design.certification_context_population_hash
            or common.certification_stream_commitment_hash
            != design.certification_stream_commitment_hash
        ):
            raise ValueError("composition F/Q certification graph does not close")

        boolean_fields = (
            "issued_before_fq_outcomes",
            "composition_frozen_before_fq_outcomes",
            "fq_outcomes_observed",
            "fq_execution_authority",
            "bce_oracle_authority",
            "candidate_freeze_authority",
            "mask_commit_authority",
            "terminal_validation_access_authority",
            "terminal_authority",
        )
        if any(type(getattr(self, name)) is not bool for name in boolean_fields):
            raise TypeError("composition F/Q authority flags must be exact booleans")
        if (
            self.issued_before_fq_outcomes is not True
            or self.composition_frozen_before_fq_outcomes is not True
            or self.fq_outcomes_observed is not False
            or self.fq_execution_authority is not False
            or self.bce_oracle_authority is not False
            or self.candidate_freeze_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_validation_access_authority is not False
            or self.terminal_authority is not False
        ):
            raise PermissionError("composition F/Q artifacts crossed proposal authority")
        if self.schema != COMPOSITION_FQ_ARTIFACTS_SCHEMA:
            raise ValueError("unsupported composition F/Q artifacts schema")
        self.assert_integrity()

    @property
    def ladder(self) -> V1FrozenCandidateLadder:
        return self.freeze.ladder

    @property
    def composition_kind(self) -> str:
        return self.decision_scope.composition_kind

    @property
    def physical_route_axis_hash(self) -> str:
        return self.freeze.ladder.route_axis_hash

    @property
    def ordered_route_axis_hash(self) -> str:
        return self.freeze.identity.ordered_route_axis_hash

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "composition_kind": self.composition_kind,
            "composition_freeze_hash": self.freeze.assert_integrity(),
            "composition_freeze_identity_hash": self.freeze.identity.identity_hash,
            "composition_replay_receipt_hash": self.freeze.replay_receipt.receipt_hash,
            "frozen_candidate_ladder_hash": self.freeze.ladder.assert_integrity(),
            "composition_scope_receipt_hash": self.decision_scope.assert_integrity(),
            "physical_fq_envelope_certificate_hash": (
                self.physical_fq_envelope_certificate.assert_integrity()
            ),
            "endpoint_envelope_bundle_hash": (
                self.endpoint_envelope_bundle.assert_integrity()
            ),
            "fq_alpha_allocation_receipt_hash": (
                self.fq_alpha_allocation.assert_integrity()
            ),
            "common_panel_receipt_hash": self.common_panel_receipt.assert_integrity(),
            "certification_design_hash": self.certification_design.assert_integrity(),
            "issued_before_fq_outcomes": self.issued_before_fq_outcomes,
            "composition_frozen_before_fq_outcomes": (
                self.composition_frozen_before_fq_outcomes
            ),
            "fq_outcomes_observed": self.fq_outcomes_observed,
            "fq_execution_authority": self.fq_execution_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_validation_access_authority": (
                self.terminal_validation_access_authority
            ),
            "terminal_authority": self.terminal_authority,
        }

    @property
    def artifacts_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.freeze.assert_integrity()
        self.decision_scope.assert_integrity()
        self.physical_fq_envelope_certificate.assert_integrity()
        self.endpoint_envelope_bundle.assert_integrity()
        self.fq_alpha_allocation.assert_integrity()
        self.common_panel_receipt.assert_integrity()
        self.certification_design.assert_integrity()
        observed = content_hash(
            self._payload(),
            prefix=COMPOSITION_FQ_ARTIFACTS_HASH_PREFIX,
        )
        if self._artifacts_hash is not None and observed != self._artifacts_hash:
            raise ValueError("composition F/Q artifacts were tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("composition F/Q artifacts hash does not match")
        object.__setattr__(self, "_artifacts_hash", observed)
        return observed

    def to_json_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifacts_hash": self.artifacts_hash}


def build_v1_composition_fq_artifacts(
    *,
    freeze: V1SampledGeometricFreeze,
    decision_scope: V1CompositionDecisionScopeReceipt,
    physical_fq_envelope_certificate: V1CorrectedAttributionEnvelopeCertificate,
    endpoint_envelope_bundle: V1CancellationAwareEndpointEnvelopeBundle,
    fq_alpha_allocation: V1FQAlphaFamilyAllocationReceipt,
    common_panel_receipt: V1FQCommonCertificationPanelReceipt,
    certification_design: V1FrozenLadderCertificationDesign,
) -> V1CompositionFQArtifacts:
    """Bind a complete neutral pre-outcome F/Q graph.

    The function deliberately performs no allocation, sampling, projection,
    classification, oracle execution, or mask transition.  It accepts only
    already-issued typed components and closes their immutable coordinates.
    """

    return V1CompositionFQArtifacts(
        freeze=freeze,
        decision_scope=decision_scope,
        physical_fq_envelope_certificate=physical_fq_envelope_certificate,
        endpoint_envelope_bundle=endpoint_envelope_bundle,
        fq_alpha_allocation=fq_alpha_allocation,
        common_panel_receipt=common_panel_receipt,
        certification_design=certification_design,
    )


__all__ = [
    "COMPOSITION_FQ_ARTIFACTS_HASH_PREFIX",
    "COMPOSITION_FQ_ARTIFACTS_SCHEMA",
    "V1CompositionFQArtifacts",
    "build_v1_composition_fq_artifacts",
]
