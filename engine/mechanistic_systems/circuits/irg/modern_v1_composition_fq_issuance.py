"""Production issuance of neutral F/Q artifacts for geometric v1.

This joint consumes an already-frozen sampled-geometric composition and the
pre-outcome statistical registrations needed to measure it.  It uses the
factory-issued proposal-position certificate to transport an analytic route
envelope from the ordered discovery axis to the physical mask axis.  The
certificate supplies positional correspondence only; no legacy axis bridge
or irreversible application authority is invented.

The builder allocates the exact preregistered discovery-look F/Q share, closes
the causal D-prefix/C-stream graph, and issues the generic common-panel and
ladder-certification receipts.  It observes no F/Q, BCE, oracle, calibration,
or validation outcomes and returns proposal-only neutral artifacts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from mechanistic_systems.evidence.glocal_contracts import (
    content_hash as sampling_content_hash,
)

from .contracts import content_hash
from .modern_v1_alexnet_fq_registration import (
    V1AlexNetFQCausalGenerationFrameReceipt,
    V1AlexNetFQCausalRoleSamplingStreamReceipt,
    V1AlexNetFQDiscoveryPrefixFreezeReceipt,
)
from .modern_v1_composition_fq import (
    V1CompositionFQArtifacts,
    build_v1_composition_fq_artifacts,
)
from .modern_v1_composition_scope import V1CompositionDecisionScopeReceipt
from .modern_v1_ladder_statistics import V1FrozenLadderCertificationDesign
from .modern_v1_proposal_alpha_ledger import V1ProposalOrderAlphaSchedule
from .modern_v1_proposal_statistics import (
    V1CorrectedAttributionEnvelopeCertificate,
    V1GroupSequentialSerflingDesign,
)
from .modern_v1_response_exactification import (
    V1CancellationAwareEndpointEnvelopeBundle,
    V1FQCommonCertificationPanelReceipt,
    V1MasterDecisionAlphaRegistration,
    allocate_v1_fq_alpha_family,
    freeze_v1_fq_certification_context_axis,
    validate_v1_fq_causal_role_graph,
)
from .modern_v1_sampled_geometric_freeze import (
    V1SampledGeometricFreeze,
    composition_scope_from_v1_sampled_geometric_freeze,
)
from .modern_v1_sampled_geometric_provenance import (
    SAMPLED_PROPOSAL_POSITION_RULE,
    V1SampledProposalPositionCertificate,
)


COMPOSITION_PHYSICAL_FQ_ENVELOPE_PROOF_SCHEMA: Final = (
    "partition_free_v1_composition_physical_fq_envelope_binding_proof_v1"
)
COMPOSITION_FQ_DESIGN_ID_PREFIX: Final = (
    "partition-free-v1-composition-fq-design-id-"
)
_PHYSICAL_ENVELOPE_PROOF_PREFIX: Final = (
    "partition-free-v1-composition-physical-fq-envelope-binding-"
)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _ordered_source_coordinates_close(
    *,
    freeze: V1SampledGeometricFreeze,
    proposal_design: V1GroupSequentialSerflingDesign,
    ordered_envelope: V1CorrectedAttributionEnvelopeCertificate,
) -> bool:
    identity = freeze.identity
    ladder = freeze.ladder
    common = (
        "campaign_contract_hash",
        "model_identity_hash",
        "query_contract_hash",
        "preprocessing_hash",
        "current_mask_hash",
        "raw_logit_stream_source_hash",
        "sampling_law_hash",
        "proposal_generation",
        "route_count",
    )
    return (
        all(
            getattr(proposal_design, name) == getattr(ladder, name)
            and getattr(ordered_envelope, name) == getattr(ladder, name)
            for name in common
        )
        and proposal_design.route_axis_hash == identity.ordered_route_axis_hash
        and ordered_envelope.route_axis_hash == identity.ordered_route_axis_hash
        and proposal_design.stratum_names == ordered_envelope.stratum_names
        and proposal_design.population_stratum_counts
        == ordered_envelope.population_stratum_counts
        and proposal_design.sealed_population_order_hashes
        == ordered_envelope.sealed_population_order_hashes
    )


def _validate_position_certificate(
    freeze: V1SampledGeometricFreeze,
) -> V1SampledProposalPositionCertificate:
    certificate = freeze.selector_problem.sampled_proposal_position_certificate
    if type(certificate) is not V1SampledProposalPositionCertificate:
        raise TypeError(
            "composition F/Q issuance requires the typed proposal-position "
            "certificate"
        )
    identity = freeze.identity
    if (
        certificate.certificate_hash
        != identity.sampled_proposal_position_certificate_hash
        or certificate.current_mask_hash != identity.current_mask_hash
        or certificate.physical_route_axis_hash
        != identity.physical_route_axis_hash
        or certificate.ordered_route_axis_hash != identity.ordered_route_axis_hash
        or certificate.route_id_axis_hash != identity.ordered_route_axis_hash
        or certificate.route_count != freeze.ladder.route_count
        or certificate.position_rule != SAMPLED_PROPOSAL_POSITION_RULE
        or certificate.full_position_bijection_verified is not True
        or certificate.current_mask_coordinate_bound is not True
        or certificate.proposal_positional_correspondence_authority is not True
        or certificate.pre_freeze_ordered_to_physical_application_authority
        is not False
        or certificate.irreversible_order_authority is not False
        or certificate.irreversible_application_authority is not False
        or certificate.fq_certification_authority is not False
        or certificate.bce_oracle_authority is not False
        or certificate.mask_commit_authority is not False
        or certificate.production_execution_authority is not False
        or certificate.terminal_authority is not False
    ):
        raise PermissionError(
            "proposal-position certificate does not preserve neutral correspondence"
        )
    return certificate


def _bind_composition_physical_fq_envelope(
    *,
    freeze: V1SampledGeometricFreeze,
    proposal_design: V1GroupSequentialSerflingDesign,
    ordered_envelope: V1CorrectedAttributionEnvelopeCertificate,
    position_certificate: V1SampledProposalPositionCertificate,
    contexts: tuple[tuple[str, tuple[str, ...]], ...],
) -> V1CorrectedAttributionEnvelopeCertificate:
    if not _ordered_source_coordinates_close(
        freeze=freeze,
        proposal_design=proposal_design,
        ordered_envelope=ordered_envelope,
    ):
        raise ValueError(
            "ordered route envelope or proposal design differs from the "
            "sampled-geometric freeze"
        )
    if tuple(name for name, _ in contexts) != ordered_envelope.stratum_names:
        raise ValueError("F/Q certification strata differ from the route envelope")
    order_hashes = tuple(
        sampling_content_hash({"stratum_id": name, "context_ids": axis})
        for name, axis in contexts
    )
    proof_artifact_hash = content_hash(
        {
            "schema": COMPOSITION_PHYSICAL_FQ_ENVELOPE_PROOF_SCHEMA,
            "source_ordered_envelope_certificate_hash": (
                ordered_envelope.certificate_hash
            ),
            "proposal_position_certificate_hash": (
                position_certificate.certificate_hash
            ),
            "physical_route_axis_declaration_hash": (
                position_certificate.physical_route_axis_declaration_hash
            ),
            "position_rule": position_certificate.position_rule,
            "certification_panel_context_order_hashes": order_hashes,
            "universal_preoutcome_input_support_unchanged": True,
            "route_envelope_values_position_preserved": True,
            "proposal_positional_correspondence_only": True,
            "ordered_to_physical_application_authority": False,
            "fq_certification_authority": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
            "terminal_authority": False,
        },
        prefix=_PHYSICAL_ENVELOPE_PROOF_PREFIX,
    )
    return V1CorrectedAttributionEnvelopeCertificate(
        campaign_contract_hash=ordered_envelope.campaign_contract_hash,
        model_identity_hash=ordered_envelope.model_identity_hash,
        query_contract_hash=ordered_envelope.query_contract_hash,
        preprocessing_hash=ordered_envelope.preprocessing_hash,
        current_mask_hash=ordered_envelope.current_mask_hash,
        raw_logit_stream_source_hash=(
            ordered_envelope.raw_logit_stream_source_hash
        ),
        sampling_law_hash=ordered_envelope.sampling_law_hash,
        proposal_generation=ordered_envelope.proposal_generation,
        route_count=ordered_envelope.route_count,
        route_axis_hash=position_certificate.physical_route_axis_hash,
        stratum_names=ordered_envelope.stratum_names,
        population_stratum_counts=tuple(len(axis) for _, axis in contexts),
        sealed_population_order_hashes=order_hashes,  # type: ignore[arg-type]
        absolute_upper_envelope_by_stratum=(
            ordered_envelope.absolute_upper_envelope_by_stratum
        ),
        verifier_source_hash=ordered_envelope.verifier_source_hash,
        proof_artifact_hash=proof_artifact_hash,
    )


def _freeze_stratum_weights(
    values: Sequence[tuple[str, float]],
    *,
    expected_names: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    weights = tuple(
        (
            _identity(name, name="F/Q stratum weight name"),
            float(value),
        )
        for name, value in values
    )
    total = math.fsum(value for _, value in weights)
    tolerance = 64.0 * math.ulp(max(1.0, abs(total)))
    if (
        tuple(name for name, _ in weights) != expected_names
        or any(not math.isfinite(value) or value <= 0.0 for _, value in weights)
        or abs(total - 1.0) > tolerance
    ):
        raise ValueError("F/Q stratum weights must be positive and sum to one")
    return weights


def issue_v1_composition_fq_artifacts(
    *,
    freeze: V1SampledGeometricFreeze,
    decision_scope: V1CompositionDecisionScopeReceipt,
    ordered_envelope_certificate: V1CorrectedAttributionEnvelopeCertificate,
    proposal_order_design: V1GroupSequentialSerflingDesign,
    endpoint_envelope_bundle: V1CancellationAwareEndpointEnvelopeBundle,
    master_alpha_registration: V1MasterDecisionAlphaRegistration,
    proposal_alpha_schedule: V1ProposalOrderAlphaSchedule,
    stratum_weights: Sequence[tuple[str, float]],
    projection_kernel_source_hash: str,
    causal_generation_frame: V1AlexNetFQCausalGenerationFrameReceipt,
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
    discovery_prefix_freeze: V1AlexNetFQDiscoveryPrefixFreezeReceipt,
    certification_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
) -> V1CompositionFQArtifacts:
    """Issue a complete neutral pre-outcome F/Q artifact graph."""

    if type(freeze) is not V1SampledGeometricFreeze:
        raise TypeError("composition F/Q issuance requires a geometric freeze")
    if type(decision_scope) is not V1CompositionDecisionScopeReceipt:
        raise TypeError("composition F/Q issuance requires the neutral scope")
    if (
        type(ordered_envelope_certificate)
        is not V1CorrectedAttributionEnvelopeCertificate
    ):
        raise TypeError("composition F/Q issuance requires the ordered envelope")
    if type(proposal_order_design) is not V1GroupSequentialSerflingDesign:
        raise TypeError("composition F/Q issuance requires the discovery design")
    if (
        type(endpoint_envelope_bundle)
        is not V1CancellationAwareEndpointEnvelopeBundle
    ):
        raise TypeError("composition F/Q issuance requires the endpoint envelope")
    if type(master_alpha_registration) is not V1MasterDecisionAlphaRegistration:
        raise TypeError("composition F/Q issuance requires the master alpha ledger")
    if type(proposal_alpha_schedule) is not V1ProposalOrderAlphaSchedule:
        raise TypeError("composition F/Q issuance requires the proposal alpha schedule")

    freeze_hash = freeze.assert_integrity()
    canonical_scope = composition_scope_from_v1_sampled_geometric_freeze(freeze)
    scope_hash = decision_scope.assert_integrity()
    if scope_hash != canonical_scope.receipt_hash:
        raise ValueError("composition decision scope differs from its freeze")
    if proposal_order_design.design_hash != decision_scope.discovery_score_design_hash:
        raise ValueError("discovery design differs from the neutral decision scope")
    ordered_envelope_certificate.assert_integrity()
    endpoint_envelope_bundle.assert_integrity()
    if endpoint_envelope_bundle.ladder_hash != freeze.ladder.ladder_hash:
        raise ValueError("endpoint envelope bundle names another frozen ladder")

    position_certificate = _validate_position_certificate(freeze)
    if type(certification_stream) is not V1AlexNetFQCausalRoleSamplingStreamReceipt:
        raise TypeError("composition F/Q issuance requires the exact C stream")
    contexts = freeze_v1_fq_certification_context_axis(
        certification_stream.context_ids_by_stratum,
    )
    validate_v1_fq_causal_role_graph(
        ladder=freeze.ladder,
        certification_context_ids_by_stratum=contexts,
        certification_stream_commitment_hash=certification_stream.receipt_hash,
        causal_generation_frame=causal_generation_frame,
        discovery_stream=discovery_stream,
        discovery_prefix_freeze=discovery_prefix_freeze,
        certification_stream=certification_stream,
        expected_discovery_evidence_state_hash=(
            freeze.identity.score_discovery_evidence_hash
        ),
        require_complete_graph=True,
    )
    look_index = decision_scope.discovery_score_look_index
    d_counts = discovery_prefix_freeze.cumulative_stratum_counts
    d_contexts = discovery_stream.context_ids_by_stratum
    d_order_hashes = tuple(
        sampling_content_hash({"stratum_id": name, "context_ids": axis})
        for name, axis in d_contexts
    )
    if (
        d_counts != proposal_order_design.registered_look_stratum_counts[
            look_index - 1
        ]
        or tuple(len(axis) for _, axis in d_contexts)
        != proposal_order_design.population_stratum_counts
        or tuple(name for name, _ in d_contexts)
        != proposal_order_design.stratum_names
        or d_order_hashes != proposal_order_design.sealed_population_order_hashes
    ):
        raise ValueError("causal D stream differs from the registered discovery look")

    physical_envelope = _bind_composition_physical_fq_envelope(
        freeze=freeze,
        proposal_design=proposal_order_design,
        ordered_envelope=ordered_envelope_certificate,
        position_certificate=position_certificate,
        contexts=contexts,
    )
    allocation = allocate_v1_fq_alpha_family(
        master=master_alpha_registration,
        proposal_schedule=proposal_alpha_schedule,
        proposal_design=proposal_order_design,
        decision_scope=decision_scope,
    )
    weights = _freeze_stratum_weights(
        stratum_weights,
        expected_names=tuple(name for name, _ in contexts),
    )
    population_hash = content_hash(
        {"certification_context_ids_by_stratum": contexts}
    )
    common_panel = V1FQCommonCertificationPanelReceipt(
        proposal_generation=freeze.ladder.proposal_generation,
        active_score_order_look_index=allocation.score_order_look_index,
        maximum_fq_ladder_redesigns=allocation.maximum_fq_ladder_redesigns,
        proposal_order_design_hash=proposal_order_design.design_hash,
        score_order_registration_hash=(
            decision_scope.discovery_score_registration_hash
        ),
        score_order_look_receipt_hash=(
            decision_scope.discovery_score_look_receipt_hash
        ),
        decision_scope_receipt_hash=scope_hash,
        fq_alpha_allocation_receipt_hash=allocation.receipt_hash,
        fq_redesign_alpha_schedule_hash=(
            allocation.fq_redesign_alpha_schedule_hash
        ),
        fq_redesign_alpha_allocations=allocation.fq_redesign_alpha_allocations,
        proposal_order_look_alpha_weights=(
            allocation.proposal_order_look_alpha_weights
        ),
        certification_context_population_hash=population_hash,
        certification_stream_commitment_hash=certification_stream.receipt_hash,
    )
    design = V1FrozenLadderCertificationDesign(
        design_id=content_hash(
            {
                "composition_freeze_hash": freeze_hash,
                "composition_scope_receipt_hash": scope_hash,
                "proposal_position_certificate_hash": (
                    position_certificate.certificate_hash
                ),
                "ordered_envelope_certificate_hash": (
                    ordered_envelope_certificate.certificate_hash
                ),
                "physical_envelope_certificate_hash": (
                    physical_envelope.certificate_hash
                ),
                "endpoint_envelope_bundle_hash": (
                    endpoint_envelope_bundle.bundle_hash
                ),
                "fq_alpha_allocation_receipt_hash": allocation.receipt_hash,
                "fq_common_panel_receipt_hash": common_panel.receipt_hash,
                "certification_context_population_hash": population_hash,
                "certification_stream_commitment_hash": (
                    certification_stream.receipt_hash
                ),
            },
            prefix=COMPOSITION_FQ_DESIGN_ID_PREFIX,
        ),
        ladder=freeze.ladder,
        envelope_certificate=physical_envelope,
        certification_context_ids_by_stratum=contexts,
        stratum_weights=weights,
        certification_context_population_hash=population_hash,
        certification_stream_commitment_hash=certification_stream.receipt_hash,
        projection_kernel_source_hash=_identity(
            projection_kernel_source_hash,
            name="projection kernel source hash",
        ),
        trajectory_alpha_ledger_hash=allocation.receipt_hash,
        trajectory_alpha_ledger_generation=freeze.ladder.proposal_generation,
        trajectory_familywise_alpha=allocation.fq_familywise_alpha,
        trajectory_alpha_committed_before_generation=(
            allocation.fq_alpha_committed_before_generation
        ),
        generation_alpha_allocation=allocation.fq_generation_alpha_allocation,
        ladder_redesign_generation=allocation.fq_ladder_redesign_generation,
        generation_alpha_committed_before_redesign=(
            allocation.fq_alpha_committed_before_redesign
        ),
        redesign_alpha_allocation=float(allocation.fq_redesign_alpha_allocation),
        endpoint_envelope_certificate=(
            endpoint_envelope_bundle.action_drive_certificate
        ),
        causal_generation_frame=causal_generation_frame,
        causal_discovery_stream=discovery_stream,
        causal_discovery_prefix_freeze=discovery_prefix_freeze,
        causal_certification_stream=certification_stream,
    )
    return build_v1_composition_fq_artifacts(
        freeze=freeze,
        decision_scope=decision_scope,
        physical_fq_envelope_certificate=physical_envelope,
        endpoint_envelope_bundle=endpoint_envelope_bundle,
        fq_alpha_allocation=allocation,
        common_panel_receipt=common_panel,
        certification_design=design,
    )


__all__ = [
    "COMPOSITION_FQ_DESIGN_ID_PREFIX",
    "COMPOSITION_PHYSICAL_FQ_ENVELOPE_PROOF_SCHEMA",
    "issue_v1_composition_fq_artifacts",
]
