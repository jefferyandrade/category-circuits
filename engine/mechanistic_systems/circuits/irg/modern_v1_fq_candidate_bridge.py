"""Pure bridge from selected-shell F/Q evidence to an ordinary v1 queue.

Response exactification freezes a nested family of packed v1-order actions.
The selected-shell F/Q store then classifies every endpoint independently.
This module is the narrow authority bridge between those artifacts and the
ordinary proposal-only acquisition surface:

* only endpoints explicitly certified within both registered F/Q budgets can
  be admitted;
* the admitted endpoints are presented in nonincreasing resource order for
  direct-oracle backtracking, without a prefix-monotonicity inference;
* a registered scalar comparator rung is a prerequisite for entering a deeper
  endpoint, with deeper old-chart endpoints explicitly deferred until the
  post-milestone relinearization; and
* no F/Q evidence acquires BCE-oracle, mask-commit, or terminal authority.

The bridge is deliberately pure.  It performs no sampling, persistence,
oracle execution, mask transition, or terminal disclosure.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Integral
from typing import Any, Final, NoReturn

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .contracts import content_hash
from .modern_v1_acquisition import (
    ResidualDiagonalStatus,
    SAMPLED_FQ_COMPOSITION_GATE_POLICY_ID,
    V1AnalyticExactGateCorrectionAcquisition,
    V1CorrectionAcquisition,
    V1FDStreamAcquisition,
    V1FQAuthorizedAcquisition,
    V1RelinearizedAcquisition,
    V1SampledCorrectedPointNumericLineage,
    compose_v1_relinearized_acquisition,
    issue_v1_fq_authorized_acquisition,
)
from .modern_v1_campaign_contract import V1OptionACampaignBinding
from .modern_v1_candidate_composition import (
    FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID,
    V1CandidateCompositionConstraints,
    V1FQCandidateCompositionResult,
    V1FiniteRemovalDomain,
    V1PackedRemovalAction,
    compose_v1_fq_candidates_in_authorized_resource_descending_order,
    issue_v1_fq_action_numeric_states,
)
from .modern_v1_core import (
    V1ChartThresholds,
    V1MixedDifferenceEnvelope,
)
from .modern_v1_composition_fq import V1CompositionFQArtifacts
from .modern_v1_composition_scope import (
    LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND,
    SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND,
)
from .modern_v1_intervention_adapter import V1AtomCostState
from .modern_v1_ladder_statistics import V1FQProposalBudget
from .modern_v1_response_exactification import V1ResponseExactificationArtifacts
from .modern_v1_response_exactification_store import (
    V1FQAcquisitionRegistration,
    V1FQRegisteredLookCommit,
)


FQ_CANDIDATE_QUEUE_AUTHORIZATION_SCHEMA: Final = (
    "partition_free_v1_fq_candidate_queue_authorization_v3"
)
FQ_COMPOSITION_CANDIDATE_QUEUE_AUTHORIZATION_SCHEMA: Final = (
    "partition_free_v1_composition_fq_candidate_queue_authorization_v1"
)
FQ_CANDIDATE_DOMAIN_DECLARATION_SCHEMA: Final = (
    "partition_free_v1_fq_certified_candidate_domain_declaration_v1"
)
FQ_CANDIDATE_QUEUE_AUTHORITY_SCOPE: Final = (
    "composition_scoped_fq_certified_endpoint_to_proposal_queue_only_"
    "no_oracle_mask_commit_or_terminal_authority_v1"
)

_AUTHORIZATION_PREFIX: Final = "partition-free-v1-fq-candidate-queue-authorization-"
_COMPOSITION_AUTHORIZATION_PREFIX: Final = (
    "partition-free-v1-composition-fq-candidate-queue-authorization-"
)
_GEOMETRY_LINEAGE_PREFIX: Final = (
    "partition-free-v1-fq-sampled-geometry-lineage-"
)
_DOMAIN_DECLARATION_PREFIX: Final = (
    "partition-free-v1-fq-certified-candidate-domain-declaration-"
)
_RESERVED_METADATA_KEYS: Final = frozenset(
    {
        "fq_candidate_queue_authorization_hash",
        "fq_registered_look_commit_hash",
        "fq_acquisition_registration_hash",
        "fq_artifact_family",
        "fq_artifacts_hash",
        "fq_freeze_hash",
        "fq_response_artifacts_hash",
        "fq_response_freeze_hash",
        "fq_composition_artifacts_hash",
        "fq_composition_freeze_hash",
        "fq_composition_scope_receipt_hash",
        "fq_geometric_freeze_identity_hash",
        "fq_geometric_replay_receipt_hash",
        "fq_geometric_selector_problem_hash",
        "fq_geometric_selector_policy_hash",
        "fq_geometric_selector_receipt_hash",
        "fq_geometric_selector_result_hash",
        "fq_sampled_geometry_source_hash",
        "fq_sampled_geometry_state_hash",
        "fq_sampled_geometry_result_hash",
        "fq_sampled_geometry_assembly_hash",
        "fq_sampled_proposal_position_certificate_hash",
        "fq_geometry_lineage_hash",
        "fq_legacy_sampled_numeric_lineage_receipt",
        "fq_geometry_lineage_authority",
        "fq_frozen_ladder_hash",
        "fq_budget_hash",
        "fq_composition_kind",
        "fq_certified_endpoint_indices_descending",
        "fq_certified_action_hashes_descending",
        "fq_queue_endpoint_indices_descending",
        "fq_queue_action_hashes_descending",
        "fq_comparator_deferred_endpoint_indices",
        "fq_selected_endpoint_index",
        "fq_selected_action_hash",
        "fq_queue_head_endpoint_index",
        "fq_queue_head_action_hash",
        "fq_candidate_queue_authority_scope",
        "fq_ordinary_queue_materialization_authority",
        "fq_monotonicity_assumed",
        "fq_bce_oracle_authority",
        "fq_candidate_freeze_authority",
        "fq_mask_commit_authority",
        "fq_terminal_authority",
        "sampled_corrected_point_lineage_hash",
        "sampled_diagnostic_acquisition_hash",
        "sampled_summary_source_hash",
        "sampled_source_geometry_state_hash",
        "sampled_source_geometry_result_hash",
        "sampled_corrected_point_estimate_hash",
        "sampled_corrected_point_status",
        "sampled_score_order_regret_receipt_axis_hash",
        "sampled_fq_composition_gate_policy_id",
        "sampled_point_estimate",
        "sampled_census_authority",
        "sampled_exact_stream_rows_available",
        "sampled_exact_candidate_numeric_state_issued",
        "sampled_separate_atom_cost_state_required",
        "sampled_decisive_fq_commit_required",
        "sampled_score_order_authority",
        "sampled_fq_certification_authority",
        "sampled_candidate_composition_authority",
        "sampled_candidate_freeze_authority",
        "sampled_bce_oracle_authority",
        "sampled_mask_commit_authority",
        "sampled_terminal_authority",
        "sampled_validation_inputs_present",
        "sampled_proposal_only",
        "sampled_authority_scope",
    }
)


class V1FQCandidateQueueRefusalCode(str, Enum):
    """Stable fail-closed reasons at the F/Q-to-queue boundary."""

    INVALID_TYPE = "invalid_type"
    COORDINATE_MISMATCH = "coordinate_mismatch"
    NONDECISIVE_FQ_LOOK = "nondecisive_fq_look"
    COMPARATOR_RUNG_NOT_CERTIFIED = "comparator_rung_not_certified"
    AUTHORITY_VIOLATION = "authority_violation"
    NO_ELIGIBLE_QUEUE = "no_eligible_queue"
    SILENT_FILTER = "silent_filter"


class V1FQCandidateQueueRefusal(ValueError):
    """Typed refusal raised before an ordinary proposal queue can be used."""

    def __init__(self, code: V1FQCandidateQueueRefusalCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1FQCandidateQueueRefusalCode, message: str) -> NoReturn:
    raise V1FQCandidateQueueRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _identity_axis(
    values: Sequence[str],
    *,
    name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    result = tuple(_identity(value, name=f"{name} item") for value in values)
    if not allow_empty and not result:
        _refuse(
            V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be nonempty",
        )
    if len(result) != len(set(result)):
        _refuse(
            V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            f"{name} must not repeat identities",
        )
    return result


def _endpoint_axis(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    result = tuple(_positive_int(value, name=f"{name} item") for value in values)
    if not result or tuple(sorted(result)) != result or len(set(result)) != len(result):
        _refuse(
            V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be a unique increasing nonempty endpoint axis",
        )
    return result


def _same_hard_mask(left: HardMask, right: HardMask) -> bool:
    return (
        type(left) is HardMask
        and type(right) is HardMask
        and left.route_count == right.route_count
        and left.route_axis_hash == right.route_axis_hash
        and left.packed_bits == right.packed_bits
        and left.mask_hash == right.mask_hash
    )


SelectedFQArtifacts = V1ResponseExactificationArtifacts | V1CompositionFQArtifacts


def _artifact_family(artifacts: object) -> str:
    if type(artifacts) is V1ResponseExactificationArtifacts:
        return "response_exactification"
    if type(artifacts) is V1CompositionFQArtifacts:
        return "composition"
    _refuse(
        V1FQCandidateQueueRefusalCode.INVALID_TYPE,
        "F/Q queue authorization requires exact legacy-response or composition artifacts",
    )


def _artifact_is_composition(artifacts: object) -> bool:
    return _artifact_family(artifacts) == "composition"


def _authorization_schema_for_artifacts(artifacts: SelectedFQArtifacts) -> str:
    return (
        FQ_CANDIDATE_QUEUE_AUTHORIZATION_SCHEMA
        if _artifact_family(artifacts) == "response_exactification"
        else FQ_COMPOSITION_CANDIDATE_QUEUE_AUTHORIZATION_SCHEMA
    )


def _authorization_prefix_for_artifacts(artifacts: SelectedFQArtifacts) -> str:
    return (
        _AUTHORIZATION_PREFIX
        if _artifact_family(artifacts) == "response_exactification"
        else _COMPOSITION_AUTHORIZATION_PREFIX
    )


def _artifacts_hash(artifacts: SelectedFQArtifacts) -> str:
    return artifacts.assert_integrity()


def _artifacts_freeze_hash(artifacts: SelectedFQArtifacts) -> str:
    return artifacts.freeze.assert_integrity()


def _artifacts_ladder(artifacts: SelectedFQArtifacts):
    return artifacts.freeze.ladder


def _artifacts_decision_scope(artifacts: SelectedFQArtifacts):
    if type(artifacts) is V1ResponseExactificationArtifacts:
        return artifacts.freeze.replay_receipt.decision_scope_receipt
    return artifacts.decision_scope


def _artifacts_physical_route_axis_hash(artifacts: SelectedFQArtifacts) -> str:
    if type(artifacts) is V1ResponseExactificationArtifacts:
        return artifacts.freeze.axis_bridge.physical_route_axis_hash
    return artifacts.physical_route_axis_hash


def _artifacts_ordered_route_axis_hash(artifacts: SelectedFQArtifacts) -> str:
    if type(artifacts) is V1ResponseExactificationArtifacts:
        return artifacts.freeze.axis_bridge.ordered_route_axis_hash
    return artifacts.ordered_route_axis_hash


def _ensure_legacy_authorization(
    authorization: "V1FQCandidateQueueAuthorization",
    *,
    property_name: str,
) -> None:
    if _artifact_is_composition(authorization.artifacts):
        _refuse(
            V1FQCandidateQueueRefusalCode.AUTHORITY_VIOLATION,
            f"{property_name} is a legacy response-exactification identity, "
            "not a composition F/Q identity",
        )


def _ensure_composition_authorization(
    authorization: "V1FQCandidateQueueAuthorization",
    *,
    property_name: str,
) -> None:
    if not _artifact_is_composition(authorization.artifacts):
        _refuse(
            V1FQCandidateQueueRefusalCode.AUTHORITY_VIOLATION,
            f"{property_name} is a composition F/Q identity, not a legacy "
            "response-exactification identity",
        )


@dataclass(frozen=True, slots=True)
class V1FQCandidateQueueAuthorization:
    """Exact F/Q proposal-screening authority for one ordinary v1 queue.

    The typed registration is retained so this object transitively binds the
    full response-artifact graph, its pre-row freeze, the packed ladder, the
    registered F/Q design/budget, and the decisive no-replace look commit.
    Certified endpoints remain an arbitrary subset of the ladder; no prefix
    closure is inferred.
    """

    binding: V1OptionACampaignBinding = field(repr=False, compare=False)
    registration: V1FQAcquisitionRegistration = field(repr=False, compare=False)
    final_commit: V1FQRegisteredLookCommit = field(repr=False, compare=False)
    budget: V1FQProposalBudget = field(repr=False, compare=False)
    generation_index: int
    current_mask: HardMask = field(repr=False, compare=False)
    committed_mask_lineage_hashes: tuple[str, ...]
    committed_mask_transition_receipt_hashes: tuple[str, ...]
    certified_endpoint_indices: tuple[int, ...]
    certified_action_hashes: tuple[str, ...]
    selected_endpoint_index: int
    selected_action_hash: str
    composition_kind: str
    fq_proposal_screening_authority: bool = True
    ordinary_candidate_queue_materialization_authority: bool = True
    fq_monotonicity_assumed: bool = False
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQ_CANDIDATE_QUEUE_AUTHORITY_SCOPE
    schema: str = FQ_CANDIDATE_QUEUE_AUTHORIZATION_SCHEMA
    _authorization_hash: str | None = field(init=False, default=None, repr=False)
    _artifacts_hash: str | None = field(init=False, default=None, repr=False)
    _freeze_hash: str | None = field(init=False, default=None, repr=False)
    _response_artifacts_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _response_freeze_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _composition_scope_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _ladder_hash: str | None = field(init=False, default=None, repr=False)
    _registration_hash: str | None = field(init=False, default=None, repr=False)
    _commit_hash: str | None = field(init=False, default=None, repr=False)
    _budget_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.binding) is not V1OptionACampaignBinding:
            _refuse(
                V1FQCandidateQueueRefusalCode.INVALID_TYPE,
                "queue authorization requires an exact campaign binding",
            )
        if type(self.registration) is not V1FQAcquisitionRegistration:
            _refuse(
                V1FQCandidateQueueRefusalCode.INVALID_TYPE,
                "queue authorization requires an exact F/Q registration",
            )
        if type(self.final_commit) is not V1FQRegisteredLookCommit:
            _refuse(
                V1FQCandidateQueueRefusalCode.INVALID_TYPE,
                "queue authorization requires an exact registered-look commit",
            )
        if type(self.budget) is not V1FQProposalBudget:
            _refuse(
                V1FQCandidateQueueRefusalCode.INVALID_TYPE,
                "queue authorization requires the exact F/Q budget",
            )
        if type(self.current_mask) is not HardMask:
            _refuse(
                V1FQCandidateQueueRefusalCode.INVALID_TYPE,
                "queue authorization requires an exact current HardMask",
            )
        artifacts = self.registration.artifacts
        expected_schema = _authorization_schema_for_artifacts(artifacts)
        if (
            _artifact_is_composition(artifacts)
            and self.schema == FQ_CANDIDATE_QUEUE_AUTHORIZATION_SCHEMA
        ):
            object.__setattr__(self, "schema", expected_schema)

        generation = _nonnegative_int(self.generation_index, name="generation index")
        lineage = _identity_axis(
            self.committed_mask_lineage_hashes,
            name="committed mask lineage",
            allow_empty=False,
        )
        transitions = _identity_axis(
            self.committed_mask_transition_receipt_hashes,
            name="committed mask transition lineage",
            allow_empty=True,
        )
        certified_indices = _endpoint_axis(
            self.certified_endpoint_indices,
            name="certified endpoint indices",
        )
        certified_hashes = _identity_axis(
            self.certified_action_hashes,
            name="certified action hashes",
            allow_empty=False,
        )
        selected_index = _positive_int(
            self.selected_endpoint_index,
            name="selected endpoint index",
        )
        selected_hash = _identity(
            self.selected_action_hash,
            name="selected action hash",
        )
        composition_kind = _identity(
            self.composition_kind,
            name="composition kind",
        )
        if len(certified_indices) != len(certified_hashes):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "certified endpoint and action axes do not align",
            )
        if (
            lineage[-1] != self.current_mask.mask_hash
            or len(transitions) != len(lineage) - 1
        ):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "queue authorization has incomplete current-mask ancestry",
            )
        flags = (
            self.fq_proposal_screening_authority,
            self.ordinary_candidate_queue_materialization_authority,
            not self.fq_monotonicity_assumed,
            not self.bce_oracle_authority,
            not self.candidate_freeze_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
        )
        if any(type(flag) is not bool or not flag for flag in flags):
            _refuse(
                V1FQCandidateQueueRefusalCode.AUTHORITY_VIOLATION,
                "F/Q queue authorization crossed its proposal-only boundary",
            )
        if (
            self.authority_scope != FQ_CANDIDATE_QUEUE_AUTHORITY_SCOPE
            or self.schema != expected_schema
        ):
            _refuse(
                V1FQCandidateQueueRefusalCode.AUTHORITY_VIOLATION,
                "F/Q queue authorization scope or schema drifted",
            )
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "committed_mask_lineage_hashes", lineage)
        object.__setattr__(
            self,
            "committed_mask_transition_receipt_hashes",
            transitions,
        )
        object.__setattr__(self, "certified_endpoint_indices", certified_indices)
        object.__setattr__(self, "certified_action_hashes", certified_hashes)
        object.__setattr__(self, "selected_endpoint_index", selected_index)
        object.__setattr__(self, "selected_action_hash", selected_hash)
        object.__setattr__(self, "composition_kind", composition_kind)
        self.assert_integrity()

    @property
    def artifacts(self):
        return self.registration.artifacts

    @property
    def artifact_family(self) -> str:
        return _artifact_family(self.registration.artifacts)

    @property
    def freeze(self):
        return self.registration.artifacts.freeze

    @property
    def ladder(self):
        return self.registration.artifacts.freeze.ladder

    @property
    def artifacts_hash(self) -> str:
        if self._artifacts_hash is None:
            self.assert_integrity()
        assert self._artifacts_hash is not None
        return self._artifacts_hash

    @property
    def freeze_hash(self) -> str:
        if self._freeze_hash is None:
            self.assert_integrity()
        assert self._freeze_hash is not None
        return self._freeze_hash

    @property
    def response_artifacts_hash(self) -> str:
        _ensure_legacy_authorization(self, property_name="response_artifacts_hash")
        if self._response_artifacts_hash is None:
            self.assert_integrity()
        assert self._response_artifacts_hash is not None
        return self._response_artifacts_hash

    @property
    def response_freeze_hash(self) -> str:
        _ensure_legacy_authorization(self, property_name="response_freeze_hash")
        if self._response_freeze_hash is None:
            self.assert_integrity()
        assert self._response_freeze_hash is not None
        return self._response_freeze_hash

    @property
    def composition_artifacts_hash(self) -> str:
        _ensure_composition_authorization(
            self,
            property_name="composition_artifacts_hash",
        )
        return self.artifacts_hash

    @property
    def composition_freeze_hash(self) -> str:
        _ensure_composition_authorization(
            self,
            property_name="composition_freeze_hash",
        )
        return self.freeze_hash

    @property
    def composition_scope_receipt_hash(self) -> str:
        _ensure_composition_authorization(
            self,
            property_name="composition_scope_receipt_hash",
        )
        if self._composition_scope_hash is None:
            self.assert_integrity()
        assert self._composition_scope_hash is not None
        return self._composition_scope_hash

    @property
    def ladder_hash(self) -> str:
        if self._ladder_hash is None:
            self.assert_integrity()
        assert self._ladder_hash is not None
        return self._ladder_hash

    @property
    def registration_hash(self) -> str:
        if self._registration_hash is None:
            self.assert_integrity()
        assert self._registration_hash is not None
        return self._registration_hash

    @property
    def commit_hash(self) -> str:
        if self._commit_hash is None:
            self.assert_integrity()
        assert self._commit_hash is not None
        return self._commit_hash

    @property
    def budget_hash(self) -> str:
        if self._budget_hash is None:
            self.assert_integrity()
        assert self._budget_hash is not None
        return self._budget_hash

    @property
    def comparator_endpoint_index(self) -> int | None:
        if _artifact_is_composition(self.artifacts):
            return None
        replay = self.freeze.replay_receipt
        rung = replay.registered_comparator_rung_new_removal_count
        if rung is None or replay.registered_comparator_rung_included is not True:
            return None
        try:
            return replay.endpoint_new_removal_counts.index(rung) + 1
        except ValueError:
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "registered comparator rung left the frozen endpoint axis",
            )

    @property
    def certified_endpoint_indices_descending(self) -> tuple[int, ...]:
        return tuple(reversed(self.certified_endpoint_indices))

    @property
    def certified_action_hashes_descending(self) -> tuple[str, ...]:
        by_index = dict(
            zip(
                self.certified_endpoint_indices,
                self.certified_action_hashes,
                strict=True,
            )
        )
        return tuple(
            by_index[index] for index in self.certified_endpoint_indices_descending
        )

    @property
    def comparator_deferred_endpoint_indices(self) -> tuple[int, ...]:
        """Certified endpoints held for the post-milestone relinearization.

        The ordinary controller must encounter the exact comparator mask before
        any deeper irreversible proposal.  Once that mask is accepted, dynamic
        relinearization invalidates every remaining action from this chart, so
        retaining a deeper old-chart proposal would provide no legitimate
        runtime opportunity.
        """

        comparator = self.comparator_endpoint_index
        if comparator is None or comparator not in self.certified_endpoint_indices:
            return ()
        return tuple(
            endpoint
            for endpoint in self.certified_endpoint_indices
            if endpoint > comparator
        )

    @property
    def queue_endpoint_indices_descending(self) -> tuple[int, ...]:
        deferred = set(self.comparator_deferred_endpoint_indices)
        return tuple(
            endpoint
            for endpoint in self.certified_endpoint_indices_descending
            if endpoint not in deferred
        )

    @property
    def queue_action_hashes_descending(self) -> tuple[str, ...]:
        by_index = dict(
            zip(
                self.certified_endpoint_indices,
                self.certified_action_hashes,
                strict=True,
            )
        )
        return tuple(by_index[index] for index in self.queue_endpoint_indices_descending)

    @property
    def queue_head_endpoint_index(self) -> int:
        return self.queue_endpoint_indices_descending[0]

    @property
    def queue_head_action_hash(self) -> str:
        return self.queue_action_hashes_descending[0]

    @property
    def authorization_hash(self) -> str:
        if self._authorization_hash is None:
            return self.assert_integrity()
        return self._authorization_hash

    def _validate_graph(self) -> dict[str, str]:
        registration = self.registration
        commit = self.final_commit
        budget = self.budget
        binding = self.binding
        artifacts = registration.artifacts
        _artifact_family(artifacts)
        freeze = artifacts.freeze
        ladder = _artifacts_ladder(artifacts)
        assessment = commit.assessment
        receipt = commit.certification_receipt
        decision = commit.decision

        try:
            artifacts_hash = _artifacts_hash(artifacts)
            freeze_hash = _artifacts_freeze_hash(artifacts)
            ladder_hash = ladder.assert_integrity()
            registration_hash = registration.assert_integrity()
            commit_hash = commit.assert_integrity()
            budget_hash = budget.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                f"F/Q artifact integrity failed: {error}",
            )
        # Authority is carried by immutable content, not by process-local
        # Python object identity.  A checkpoint/resume reconstruction may
        # legitimately rebuild the exact registered budget as a distinct
        # object while preserving its content-addressed identity.
        if registration.budget.budget_hash != budget_hash:
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "authorization budget differs from the exact registered F/Q budget",
            )
        if commit.registration_hash != registration_hash:
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "registered-look commit names another F/Q registration",
            )
        if commit.look_index > len(registration.registered_looks):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "registered-look commit left the pre-outcome look schedule",
            )
        registered_look = registration.registered_looks[commit.look_index - 1]
        request = registration.projection_request(
            look_index=commit.look_index,
            prior_commit_hash=commit.prior_commit_hash,
        )
        if (
            commit.cumulative_stratum_counts
            != registered_look.cumulative_stratum_counts
            or commit.projection_request_hash != request.request_hash
            or commit.projection_csr_hash != request.csr.csr_hash
            or commit.scorer_request_identity_hash
            != registered_look.scorer_request_identity_hash
            or commit.acquisition_lineage_hash != request.acquisition_lineage_hash
        ):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "registered-look commit does not replay at its exact chain position",
            )

        if assessment.status != "certified_largest_endpoint":
            _refuse(
                V1FQCandidateQueueRefusalCode.NONDECISIVE_FQ_LOOK,
                f"F/Q status {assessment.status!r} cannot materialize a queue",
            )
        if decision.status != "certified_largest_endpoint":
            _refuse(
                V1FQCandidateQueueRefusalCode.NONDECISIVE_FQ_LOOK,
                "canonical F/Q decision did not certify an endpoint",
            )

        contract = binding.contract
        current_mask = self.current_mask
        physical_route_axis_hash = _artifacts_physical_route_axis_hash(artifacts)
        ordered_route_axis_hash = _artifacts_ordered_route_axis_hash(artifacts)
        coordinate_values = (
            ladder.campaign_contract_hash,
            budget.campaign_contract_hash,
            receipt.campaign_contract_hash,
        )
        if any(value != contract.campaign_contract_hash for value in coordinate_values):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "F/Q graph belongs to another campaign contract",
            )
        if (
            ladder.proposal_generation != self.generation_index
            or budget.proposal_generation != self.generation_index
            or receipt.proposal_generation != self.generation_index
        ):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "F/Q graph belongs to another relinearization generation",
            )
        if (
            current_mask.route_axis_hash != contract.route_axis_hash
            or ladder.route_axis_hash != current_mask.route_axis_hash
            or budget.route_axis_hash != current_mask.route_axis_hash
            or receipt.route_axis_hash != current_mask.route_axis_hash
            or physical_route_axis_hash != current_mask.route_axis_hash
            or ladder.route_count != current_mask.route_count
            or budget.route_count != current_mask.route_count
            or receipt.route_count != current_mask.route_count
        ):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "F/Q graph differs from the current physical route axis",
            )
        if (
            ladder.current_mask_hash != current_mask.mask_hash
            or budget.current_mask_hash != current_mask.mask_hash
            or receipt.current_mask_hash != current_mask.mask_hash
            or not ladder.is_packed
            or ladder.parent_mask is None
            or not _same_hard_mask(ladder.parent_mask, current_mask)
        ):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "F/Q graph is stale for the exact current hard mask",
            )
        if registration.ordered_score_route_axis_hash != ordered_route_axis_hash:
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "F/Q registration ordered score-route axis differs from artifacts",
            )

        design = artifacts.certification_design
        ladder_action_hashes = ladder.action_hashes
        classification_axis = tuple(
            (row.endpoint_index, row.action_hash) for row in assessment.classifications
        )
        expected_axis = tuple(enumerate(ladder_action_hashes, start=1))
        receipt_axis = tuple(
            (row.endpoint_index, row.action_hash) for row in receipt.endpoint_intervals
        )
        decision_axis = tuple(
            (row[0], row[1]) for row in decision.endpoint_outcomes
        )
        if (
            receipt.design_hash != design.design_hash
            or receipt.ladder_hash != ladder_hash
            or decision.design_hash != design.design_hash
            or decision.ladder_hash != ladder_hash
            or decision.budget_hash != budget_hash
            or assessment.budget_hash != budget_hash
            or classification_axis != expected_axis
            or receipt_axis != expected_axis
            or decision_axis != expected_axis
        ):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "F/Q classification, receipt, decision, or budget axis differs",
            )

        # F/Q safety is necessary but not sufficient for an irreversible
        # oracle candidate.  Admit only the F/Q-safe endpoint set recorded by
        # the pre-outcome composition scope.  Legacy score-zero-regret ladders
        # expose the same set through their retained compatibility ledger;
        # sampled-geometry ladders deliberately keep that legacy ledger empty.
        certified_rows = tuple(
            row
            for row in assessment.classifications
            if row.endpoint_index
            in set(assessment.composition_fq_certified_endpoint_indices)
        )
        certified_indices = tuple(row.endpoint_index for row in certified_rows)
        certified_hashes = tuple(row.action_hash for row in certified_rows)
        if (
            self.certified_endpoint_indices != certified_indices
            or self.certified_action_hashes != certified_hashes
            or assessment.composition_fq_certified_endpoint_indices
            != certified_indices
            or assessment.composition_fq_certified_action_hashes
            != certified_hashes
            or assessment.composition_kind != self.composition_kind
        ):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "authorization changed the composition-scoped F/Q-certified endpoint set",
            )
        if not certified_rows:
            _refuse(
                V1FQCandidateQueueRefusalCode.NO_ELIGIBLE_QUEUE,
                "the decisive F/Q look contains no certified endpoint",
            )
        assessment_selected_index = _positive_int(
            assessment.selected_endpoint_index,
            name="assessment selected endpoint index",
        )
        selected = assessment.classifications[assessment_selected_index - 1]
        if (
            selected.endpoint_index not in certified_indices
            or self.selected_endpoint_index != selected.endpoint_index
            or self.selected_action_hash != selected.action_hash
            or assessment.selected_action_hash != selected.action_hash
            or decision.selected_endpoint_index != selected.endpoint_index
            or decision.selected_action_hash != selected.action_hash
        ):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "authorization changed the decision-scoped selected endpoint",
            )

        for endpoint in certified_indices:
            action = ladder.actions[endpoint - 1]
            if not action.is_packed or action.packed_removal_action is None:
                _refuse(
                    V1FQCandidateQueueRefusalCode.INVALID_TYPE,
                    "ordinary queue materialization requires packed certified actions",
                )

        scope = _artifacts_decision_scope(artifacts)
        scope_hash = scope.assert_integrity()
        if (
            assessment.decision_scope_receipt_hash != scope_hash
            or tuple(scope.registered_endpoint_new_removal_counts)
            != tuple(ladder.endpoint_removed_counts)
            or tuple(scope.decision_competitive_endpoint_indices)
            != tuple(assessment.decision_competitive_endpoint_indices)
        ):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "F/Q assessment is not bound to the registered composition scope",
            )
        if type(artifacts) is V1CompositionFQArtifacts:
            identity = freeze.identity
            replay = freeze.replay_receipt
            problem = freeze.selector_problem
            policy = freeze.selector_policy
            result = freeze.selector_result
            position_certificate = problem.sampled_proposal_position_certificate
            position_certificate_hash = (
                None
                if position_certificate is None
                else position_certificate.certificate_hash
            )
            if (
                self.composition_kind
                != SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND
                or scope.composition_kind != self.composition_kind
                or assessment.score_resolved_endpoint_indices
                or assessment.provisional_endpoint_indices
                or assessment.score_resolved_fq_certified_endpoint_indices
                or assessment.score_resolved_fq_certified_action_hashes
                or assessment.score_scope_blocking_endpoint_indices
                or assessment.selected_endpoint_score_resolution_required is not False
                or assessment.smaller_score_provisional_endpoints_oracle_ineligible
                is not False
                or assessment.composition_frozen_endpoint_indices
                != scope.composition_frozen_endpoint_indices
                or assessment.composition_provisional_endpoint_indices
                != scope.composition_provisional_endpoint_indices
                or tuple(assessment.composition_fq_certified_endpoint_indices)
                != certified_indices
                or tuple(assessment.composition_fq_certified_action_hashes)
                != certified_hashes
                or scope.sampled_geometry_state_hash
                != identity.sampled_geometry_state_hash
                or scope.sampled_geometry_result_hash
                != identity.sampled_geometry_result_hash
                or scope.sampled_geometry_assembly_hash
                != identity.sampled_geometry_assembly_hash
                or scope.sampled_selector_problem_hash != problem.problem_hash
                or scope.sampled_selector_policy_hash != policy.policy_hash
                or scope.sampled_selector_receipt_hash
                != result.receipt.receipt_hash
                or scope.sampled_selector_result_hash != result.result_hash
                or scope.sampled_proposal_position_certificate_hash
                != identity.sampled_proposal_position_certificate_hash
                or identity.sampled_proposal_position_certificate_hash
                != position_certificate_hash
                or scope.composition_freeze_identity_hash != identity.identity_hash
                or scope.composition_freeze_replay_receipt_hash
                != replay.receipt_hash
                or scope.frozen_candidate_ladder_hash != ladder_hash
                or scope.endpoint_action_hashes != ladder.action_hashes
                or scope.current_mask_hash != current_mask.mask_hash
                or scope.proposal_generation != self.generation_index
                or scope.route_count != current_mask.route_count
                or scope.physical_route_axis_hash != current_mask.route_axis_hash
                or scope.ordered_route_axis_hash != ordered_route_axis_hash
            ):
                _refuse(
                    V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                    "composition authorization is not bound to its neutral "
                    "sampled-geometric scope and freeze lineage",
                )
        else:
            replay = freeze.replay_receipt
        comparator_endpoint = self.comparator_endpoint_index
        if comparator_endpoint is not None:
            comparator_count = replay.registered_comparator_retained_route_count
            if (
                replay.registered_comparator_rung_included is not True
                or comparator_count is None
                or ladder.endpoint_mask(endpoint_index=comparator_endpoint).retained_count
                != comparator_count
            ):
                _refuse(
                    V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                    "registered scalar comparator rung does not close on its exact mask",
                )
            if (
                selected.endpoint_index > comparator_endpoint
                and comparator_endpoint not in certified_indices
            ):
                _refuse(
                    V1FQCandidateQueueRefusalCode.COMPARATOR_RUNG_NOT_CERTIFIED,
                    "a deeper endpoint cannot enter while the registered comparator "
                    "rung is unresolved or unsafe",
                )

        return {
            "artifact_family": _artifact_family(artifacts),
            "artifacts_hash": artifacts_hash,
            "freeze_hash": freeze_hash,
            "response_artifacts_hash": (
                artifacts_hash if type(artifacts) is V1ResponseExactificationArtifacts else ""
            ),
            "response_freeze_hash": (
                freeze_hash if type(artifacts) is V1ResponseExactificationArtifacts else ""
            ),
            "composition_scope_hash": scope_hash,
            "ladder_hash": ladder_hash,
            "registration_hash": registration_hash,
            "commit_hash": commit_hash,
            "budget_hash": budget_hash,
        }

    def _payload(self, hashes: Mapping[str, str]) -> dict[str, object]:
        if _artifact_is_composition(self.artifacts):
            freeze = self.freeze
            identity = freeze.identity
            replay = freeze.replay_receipt
            problem = freeze.selector_problem
            policy = freeze.selector_policy
            result = freeze.selector_result
            scope = self.artifacts.decision_scope
            position_certificate = problem.sampled_proposal_position_certificate
            position_certificate_hash = (
                None
                if position_certificate is None
                else position_certificate.certificate_hash
            )
            return {
                "schema": self.schema,
                "campaign_contract_hash": self.binding.contract.campaign_contract_hash,
                "campaign_binding_hash": self.binding.binding_hash,
                "artifact_family": hashes["artifact_family"],
                "composition_artifacts_hash": hashes["artifacts_hash"],
                "composition_freeze_hash": hashes["freeze_hash"],
                "composition_scope_receipt_hash": hashes["composition_scope_hash"],
                "ladder_hash": hashes["ladder_hash"],
                "composition_freeze_identity_hash": identity.identity_hash,
                "composition_replay_receipt_hash": replay.receipt_hash,
                "geometric_selector_problem_hash": problem.problem_hash,
                "geometric_selector_policy_hash": policy.policy_hash,
                "geometric_selector_receipt_hash": result.receipt.receipt_hash,
                "geometric_selector_result_hash": result.result_hash,
                "sampled_geometry_source_hash": identity.sampled_geometry_source_hash,
                "sampled_geometry_state_hash": identity.sampled_geometry_state_hash,
                "sampled_geometry_result_hash": identity.sampled_geometry_result_hash,
                "sampled_geometry_assembly_hash": identity.sampled_geometry_assembly_hash,
                "sampled_proposal_position_certificate_hash": (
                    identity.sampled_proposal_position_certificate_hash
                ),
                "selector_problem_position_certificate_hash": (
                    position_certificate_hash
                ),
                "fq_registration_hash": hashes["registration_hash"],
                "fq_registered_look_commit_hash": hashes["commit_hash"],
                "fq_budget_hash": hashes["budget_hash"],
                "generation_index": self.generation_index,
                "current_mask_hash": self.current_mask.mask_hash,
                "route_axis_hash": self.current_mask.route_axis_hash,
                "ordered_score_route_axis_hash": scope.ordered_route_axis_hash,
                "route_count": self.current_mask.route_count,
                "committed_mask_lineage_hashes": self.committed_mask_lineage_hashes,
                "committed_mask_transition_receipt_hashes": (
                    self.committed_mask_transition_receipt_hashes
                ),
                "composition_kind": self.composition_kind,
                "composition_frozen_endpoint_indices": (
                    scope.composition_frozen_endpoint_indices
                ),
                "composition_provisional_endpoint_indices": (
                    scope.composition_provisional_endpoint_indices
                ),
                "decision_competitive_endpoint_indices": (
                    scope.decision_competitive_endpoint_indices
                ),
                "certified_endpoint_indices": self.certified_endpoint_indices,
                "certified_action_hashes": self.certified_action_hashes,
                "selected_endpoint_index": self.selected_endpoint_index,
                "selected_action_hash": self.selected_action_hash,
                "queue_endpoint_indices_descending": (
                    self.queue_endpoint_indices_descending
                ),
                "queue_action_hashes_descending": (
                    self.queue_action_hashes_descending
                ),
                "comparator_deferred_endpoint_indices": (
                    self.comparator_deferred_endpoint_indices
                ),
                "queue_head_endpoint_index": self.queue_head_endpoint_index,
                "queue_head_action_hash": self.queue_head_action_hash,
                "registered_comparator_endpoint_index": None,
                "registered_comparator_retained_route_count": None,
                "score_refinement_alias_present": False,
                "legacy_sampled_numeric_lineage_receipt_permitted": False,
                "fq_proposal_screening_authority": self.fq_proposal_screening_authority,
                "ordinary_candidate_queue_materialization_authority": (
                    self.ordinary_candidate_queue_materialization_authority
                ),
                "fq_monotonicity_assumed": self.fq_monotonicity_assumed,
                "bce_oracle_authority": self.bce_oracle_authority,
                "candidate_freeze_authority": self.candidate_freeze_authority,
                "mask_commit_authority": self.mask_commit_authority,
                "terminal_authority": self.terminal_authority,
                "authority_scope": self.authority_scope,
            }
        replay = self.freeze.replay_receipt
        return {
            "schema": self.schema,
            "campaign_contract_hash": self.binding.contract.campaign_contract_hash,
            "campaign_binding_hash": self.binding.binding_hash,
            "response_artifacts_hash": hashes["response_artifacts_hash"],
            "response_freeze_hash": hashes["response_freeze_hash"],
            "ladder_hash": hashes["ladder_hash"],
            "response_replay_receipt_hash": replay.receipt_hash,
            "axis_bridge_receipt_hash": self.freeze.axis_bridge.receipt_hash,
            "fq_registration_hash": hashes["registration_hash"],
            "fq_registered_look_commit_hash": hashes["commit_hash"],
            "fq_budget_hash": hashes["budget_hash"],
            "generation_index": self.generation_index,
            "current_mask_hash": self.current_mask.mask_hash,
            "route_axis_hash": self.current_mask.route_axis_hash,
            "route_count": self.current_mask.route_count,
            "committed_mask_lineage_hashes": self.committed_mask_lineage_hashes,
            "committed_mask_transition_receipt_hashes": (
                self.committed_mask_transition_receipt_hashes
            ),
            "composition_kind": self.composition_kind,
            "certified_endpoint_indices": self.certified_endpoint_indices,
            "certified_action_hashes": self.certified_action_hashes,
            "selected_endpoint_index": self.selected_endpoint_index,
            "selected_action_hash": self.selected_action_hash,
            "queue_endpoint_indices_descending": (
                self.queue_endpoint_indices_descending
            ),
            "queue_action_hashes_descending": self.queue_action_hashes_descending,
            "comparator_deferred_endpoint_indices": (
                self.comparator_deferred_endpoint_indices
            ),
            "queue_head_endpoint_index": self.queue_head_endpoint_index,
            "queue_head_action_hash": self.queue_head_action_hash,
            "registered_comparator_endpoint_index": self.comparator_endpoint_index,
            "registered_comparator_retained_route_count": (
                replay.registered_comparator_retained_route_count
            ),
            "fq_proposal_screening_authority": self.fq_proposal_screening_authority,
            "ordinary_candidate_queue_materialization_authority": (
                self.ordinary_candidate_queue_materialization_authority
            ),
            "fq_monotonicity_assumed": self.fq_monotonicity_assumed,
            "bce_oracle_authority": self.bce_oracle_authority,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "authority_scope": self.authority_scope,
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        hashes = self._validate_graph()
        observed = content_hash(
            self._payload(hashes),
            prefix=_authorization_prefix_for_artifacts(self.artifacts),
        )
        if self._authorization_hash is not None and observed != self._authorization_hash:
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "F/Q candidate-queue authorization was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "F/Q candidate-queue authorization hash does not match",
            )
        object.__setattr__(self, "_artifacts_hash", hashes["artifacts_hash"])
        object.__setattr__(self, "_freeze_hash", hashes["freeze_hash"])
        object.__setattr__(
            self,
            "_response_artifacts_hash",
            hashes["response_artifacts_hash"] or None,
        )
        object.__setattr__(
            self,
            "_response_freeze_hash",
            hashes["response_freeze_hash"] or None,
        )
        object.__setattr__(
            self,
            "_composition_scope_hash",
            hashes["composition_scope_hash"],
        )
        object.__setattr__(self, "_ladder_hash", hashes["ladder_hash"])
        object.__setattr__(
            self,
            "_registration_hash",
            hashes["registration_hash"],
        )
        object.__setattr__(self, "_commit_hash", hashes["commit_hash"])
        object.__setattr__(self, "_budget_hash", hashes["budget_hash"])
        object.__setattr__(self, "_authorization_hash", observed)
        return observed


def authorize_v1_fq_candidate_queue(
    *,
    binding: V1OptionACampaignBinding,
    registration: V1FQAcquisitionRegistration,
    final_commit: V1FQRegisteredLookCommit,
    generation_index: int,
    current_mask: HardMask,
    committed_mask_lineage_hashes: Sequence[str],
    committed_mask_transition_receipt_hashes: Sequence[str],
) -> V1FQCandidateQueueAuthorization:
    """Issue proposal-queue authority from one decisive registered F/Q look."""

    if type(registration) is not V1FQAcquisitionRegistration:
        _refuse(
            V1FQCandidateQueueRefusalCode.INVALID_TYPE,
            "queue authorization requires an exact F/Q registration",
        )
    if type(final_commit) is not V1FQRegisteredLookCommit:
        _refuse(
            V1FQCandidateQueueRefusalCode.INVALID_TYPE,
            "queue authorization requires an exact registered-look commit",
        )
    assessment = final_commit.assessment
    if (
        assessment.status != "certified_largest_endpoint"
        or assessment.selected_endpoint_index is None
        or assessment.selected_action_hash is None
    ):
        _refuse(
            V1FQCandidateQueueRefusalCode.NONDECISIVE_FQ_LOOK,
            f"F/Q status {assessment.status!r} cannot materialize a queue",
        )
    return V1FQCandidateQueueAuthorization(
        binding=binding,
        registration=registration,
        final_commit=final_commit,
        budget=registration.budget,
        generation_index=generation_index,
        current_mask=current_mask,
        committed_mask_lineage_hashes=tuple(committed_mask_lineage_hashes),
        committed_mask_transition_receipt_hashes=tuple(
            committed_mask_transition_receipt_hashes
        ),
        certified_endpoint_indices=(
            assessment.composition_fq_certified_endpoint_indices
        ),
        certified_action_hashes=(
            assessment.composition_fq_certified_action_hashes
        ),
        selected_endpoint_index=assessment.selected_endpoint_index,
        selected_action_hash=assessment.selected_action_hash,
        composition_kind=assessment.composition_kind,
    )


def _authorized_packed_actions(
    authorization: V1FQCandidateQueueAuthorization,
) -> tuple[V1PackedRemovalAction, ...]:
    actions: list[V1PackedRemovalAction] = []
    for endpoint_index in authorization.queue_endpoint_indices_descending:
        frozen = authorization.ladder.actions[endpoint_index - 1]
        packed = frozen.packed_removal_action
        if type(packed) is not V1PackedRemovalAction:
            _refuse(
                V1FQCandidateQueueRefusalCode.INVALID_TYPE,
                "certified endpoint is not an exact packed removal action",
            )
        resolved_count, resolved_hash = (
            packed.resulting_mask.removed_index_count_and_hash_from(
                authorization.current_mask
            )
        )
        if (
            resolved_count != packed.removed_count
            or resolved_hash != packed.removed_route_index_hash
        ):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "certified packed action differs from the current-mask delta",
            )
        actions.append(packed)
    return tuple(actions)


def _assert_authorized_current_coordinate(
    *,
    authorization: V1FQCandidateQueueAuthorization,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    committed_mask_lineage_hashes: Sequence[str],
    committed_mask_transition_receipt_hashes: Sequence[str],
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    generation = _nonnegative_int(generation_index, name="generation index")
    lineage = tuple(committed_mask_lineage_hashes)
    transitions = tuple(committed_mask_transition_receipt_hashes)
    if (
        type(binding) is not V1OptionACampaignBinding
        or binding.binding_hash != authorization.binding.binding_hash
        or generation != authorization.generation_index
        or type(current_mask) is not HardMask
        or not _same_hard_mask(current_mask, authorization.current_mask)
        or lineage != authorization.committed_mask_lineage_hashes
        or transitions
        != authorization.committed_mask_transition_receipt_hashes
    ):
        _refuse(
            V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            "F/Q composition inputs differ from the authorized generation, "
            "mask, axis, or ancestry",
        )
    return generation, lineage, transitions


def _exact_resource_gain_rows_from_atom_costs(
    *,
    authorization: V1FQCandidateQueueAuthorization,
    atom_costs: V1AtomCostState,
) -> tuple[tuple[tuple[int, str, float], ...], str]:
    if type(atom_costs) is not V1AtomCostState:
        _refuse(
            V1FQCandidateQueueRefusalCode.INVALID_TYPE,
            "compact F/Q composition requires an exact atom-cost state",
        )
    if _artifact_is_composition(authorization.artifacts):
        freeze = authorization.freeze
        identity = freeze.identity
        problem = freeze.selector_problem
        result = freeze.selector_result
        ordered_route_axis_hash = identity.ordered_route_axis_hash
        if (
            atom_costs.route_axis_hash != ordered_route_axis_hash
            or authorization.registration.ordered_score_route_axis_hash
            != ordered_route_axis_hash
            or authorization.current_mask.route_axis_hash != problem.route_axis_hash
            or authorization.current_mask.route_count != problem.route_count
            or len(atom_costs.route_ids) != problem.route_count
        ):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "composition atom-cost ordered axis or selector physical axis "
                "differs from the authorized geometric coordinates",
            )
        position_certificate = problem.sampled_proposal_position_certificate
        position_certificate_hash = (
            None if position_certificate is None else position_certificate.certificate_hash
        )
        atom_cost_source_hash = content_hash(
            {
                "schema": "partition_free_v1_composition_fq_selector_resource_replay_v1",
                "atom_cost_state_hash": atom_costs.state_hash,
                "selector_problem_hash": problem.problem_hash,
                "selector_policy_hash": freeze.selector_policy.policy_hash,
                "selector_receipt_hash": result.receipt.receipt_hash,
                "selector_result_hash": result.result_hash,
                "sampled_proposal_position_certificate_hash": (
                    identity.sampled_proposal_position_certificate_hash
                ),
                "selector_problem_position_certificate_hash": (
                    position_certificate_hash
                ),
                "ordered_route_axis_hash": ordered_route_axis_hash,
                "physical_route_axis_hash": problem.route_axis_hash,
                "route_count": problem.route_count,
                "resource_plane": problem.resource_plane,
                "resource_cost_policy_hash": problem.resource_cost_policy_hash,
                "resource_gain_source": (
                    "selector_result_exact_physical_resource_gain"
                ),
            },
            prefix="partition-free-v1-fq-composition-resource-source-",
        )
        normalized = tuple(
            candidate
            for _index, candidate in sorted(
                enumerate(result.frontier),
                key=lambda item: item[1].removal_count,
            )
        )
        rows: list[tuple[int, str, float]] = []
        for endpoint_index, action_hash in zip(
            authorization.queue_endpoint_indices_descending,
            authorization.queue_action_hashes_descending,
            strict=True,
        ):
            frozen = authorization.ladder.actions[endpoint_index - 1]
            if frozen.action_hash != action_hash:
                _refuse(
                    V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                    "authorization queue action hash differs from the frozen ladder",
                )
            candidate = normalized[endpoint_index - 1]
            packed = frozen.packed_removal_action
            if (
                type(packed) is not V1PackedRemovalAction
                or candidate.action.action_hash != packed.action_hash
                or candidate.action.resulting_mask.mask_hash
                != packed.resulting_mask.mask_hash
                or candidate.removal_count != packed.removed_count
            ):
                _refuse(
                    V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                    "selector resource replay differs from the frozen packed action",
                )
            gain = float(candidate.resource_gain)
            if not math.isfinite(gain) or gain < 0.0:
                _refuse(
                    V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                    "selector resource gain is invalid",
                )
            rows.append((endpoint_index, action_hash, gain))
        return tuple(rows), atom_cost_source_hash

    bridge = authorization.freeze.axis_bridge
    bridge.assert_integrity()
    if (
        atom_costs.route_axis_hash != bridge.ordered_route_axis_hash
        or authorization.registration.ordered_score_route_axis_hash
        != bridge.ordered_route_axis_hash
        or authorization.current_mask.route_axis_hash
        != bridge.physical_route_axis_hash
        or len(atom_costs.route_ids) != bridge.route_count
        or authorization.current_mask.route_count != bridge.route_count
    ):
        _refuse(
            V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            "atom-cost ordered route axis or its position-preserving physical "
            "bridge differs from the authorized route coordinates",
        )
    atom_cost_source_hash = content_hash(
        {
            "atom_cost_state_hash": atom_costs.state_hash,
            "axis_bridge_receipt_hash": bridge.receipt_hash,
            "ordered_route_axis_hash": bridge.ordered_route_axis_hash,
            "physical_route_axis_hash": bridge.physical_route_axis_hash,
            "route_count": bridge.route_count,
            "position_rule": bridge.position_rule,
            "full_position_bijection_verified": (
                bridge.full_position_bijection_verified
            ),
        },
        prefix="partition-free-v1-fq-atom-cost-axis-bridge-source-",
    )
    costs = atom_costs.costs
    rows: list[tuple[int, str, float]] = []
    for endpoint_index, action_hash in zip(
        authorization.queue_endpoint_indices_descending,
        authorization.queue_action_hashes_descending,
        strict=True,
    ):
        frozen = authorization.ladder.actions[endpoint_index - 1]
        if frozen.action_hash != action_hash:
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "authorization queue action hash differs from the frozen ladder",
            )
        packed = frozen.packed_removal_action
        if type(packed) is not V1PackedRemovalAction:
            _refuse(
                V1FQCandidateQueueRefusalCode.INVALID_TYPE,
                "certified endpoint is not an exact packed removal action",
            )
        chunk_sums = (
            float(np.sum(costs[chunk], dtype=np.float64))
            for chunk in packed.resulting_mask.iter_removed_index_chunks_from(
                authorization.current_mask
            )
        )
        gain = math.fsum(chunk_sums)
        if not math.isfinite(gain) or gain < 0.0:
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "atom-cost resource gain is invalid",
            )
        rows.append((endpoint_index, action_hash, gain))
    return tuple(rows), atom_cost_source_hash


def _geometry_lineage_metadata(
    authorization: V1FQCandidateQueueAuthorization,
) -> dict[str, object]:
    _ensure_composition_authorization(
        authorization,
        property_name="geometry_lineage_metadata",
    )
    artifacts = authorization.artifacts
    assert type(artifacts) is V1CompositionFQArtifacts
    freeze = artifacts.freeze
    identity = freeze.identity
    problem = freeze.selector_problem
    policy = freeze.selector_policy
    result = freeze.selector_result
    scope_hash = artifacts.decision_scope.assert_integrity()
    position_certificate = problem.sampled_proposal_position_certificate
    position_certificate_hash = (
        None if position_certificate is None else position_certificate.certificate_hash
    )
    lineage_payload = {
        "schema": "partition_free_v1_fq_sampled_geometry_lineage_v1",
        "authorization_hash": authorization.authorization_hash,
        "composition_artifacts_hash": authorization.composition_artifacts_hash,
        "composition_freeze_hash": authorization.composition_freeze_hash,
        "composition_scope_receipt_hash": scope_hash,
        "composition_freeze_identity_hash": identity.identity_hash,
        "composition_replay_receipt_hash": freeze.replay_receipt.receipt_hash,
        "selector_problem_hash": problem.problem_hash,
        "selector_policy_hash": policy.policy_hash,
        "selector_receipt_hash": result.receipt.receipt_hash,
        "selector_result_hash": result.result_hash,
        "sampled_geometry_source_hash": identity.sampled_geometry_source_hash,
        "sampled_geometry_state_hash": identity.sampled_geometry_state_hash,
        "sampled_geometry_result_hash": identity.sampled_geometry_result_hash,
        "sampled_geometry_assembly_hash": identity.sampled_geometry_assembly_hash,
        "sampled_proposal_position_certificate_hash": (
            identity.sampled_proposal_position_certificate_hash
        ),
        "selector_problem_position_certificate_hash": position_certificate_hash,
        "legacy_sampled_numeric_lineage_receipt": False,
        "proposal_only": True,
    }
    lineage_hash = content_hash(lineage_payload, prefix=_GEOMETRY_LINEAGE_PREFIX)
    return {
        "fq_geometry_lineage_hash": lineage_hash,
        "fq_sampled_geometry_source_hash": identity.sampled_geometry_source_hash,
        "fq_sampled_geometry_state_hash": identity.sampled_geometry_state_hash,
        "fq_sampled_geometry_result_hash": identity.sampled_geometry_result_hash,
        "fq_sampled_geometry_assembly_hash": identity.sampled_geometry_assembly_hash,
        "fq_geometric_freeze_identity_hash": identity.identity_hash,
        "fq_geometric_replay_receipt_hash": freeze.replay_receipt.receipt_hash,
        "fq_geometric_selector_problem_hash": problem.problem_hash,
        "fq_geometric_selector_policy_hash": policy.policy_hash,
        "fq_geometric_selector_receipt_hash": result.receipt.receipt_hash,
        "fq_geometric_selector_result_hash": result.result_hash,
        "fq_sampled_proposal_position_certificate_hash": (
            identity.sampled_proposal_position_certificate_hash
        ),
        "fq_legacy_sampled_numeric_lineage_receipt": False,
        "fq_geometry_lineage_authority": "sampled_geometry_proposal_lineage_only",
    }


def _sampled_lineage_metadata(
    *,
    authorization: V1FQCandidateQueueAuthorization,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    context_mixture_hash: str,
    sampled_numeric_lineage: V1SampledCorrectedPointNumericLineage | None,
) -> dict[str, object]:
    if _artifact_is_composition(authorization.artifacts):
        if sampled_numeric_lineage is not None:
            _refuse(
                V1FQCandidateQueueRefusalCode.AUTHORITY_VIOLATION,
                "composition F/Q authorization cannot consume a legacy sampled "
                "response-exactification numeric-lineage receipt",
            )
        return _geometry_lineage_metadata(authorization)
    if sampled_numeric_lineage is None:
        return {}
    if type(sampled_numeric_lineage) is not V1SampledCorrectedPointNumericLineage:
        _refuse(
            V1FQCandidateQueueRefusalCode.INVALID_TYPE,
            "sampled numeric lineage must be the exact diagnostic lineage type",
        )
    lineage = sampled_numeric_lineage
    lineage_hash = lineage.lineage_hash
    replay_hash = authorization.registration.artifacts.freeze.replay_receipt.receipt_hash
    if (
        lineage.campaign_contract_hash != binding.contract.campaign_contract_hash
        or lineage.campaign_binding_hash != binding.binding_hash
        or lineage.registration_hash != binding.registration.registration_hash
        or lineage.generation_index != generation_index
        or lineage.parent_mask_hash != current_mask.mask_hash
        or lineage.physical_route_axis_hash != current_mask.route_axis_hash
        or lineage.route_count != current_mask.route_count
        or lineage.referent_id != binding.contract.referent_id
        or lineage.context_mixture_hash != context_mixture_hash
        or lineage.calibration_identity != binding.contract.calibration_identity
        or lineage.response_exactification_replay_receipt_hash != replay_hash
        or lineage.fq_composition_gate_policy_id
        != SAMPLED_FQ_COMPOSITION_GATE_POLICY_ID
    ):
        _refuse(
            V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            "sampled numeric lineage differs from the authorized F/Q generation, "
            "mask, coordinate, or replay receipt",
        )
    if (
        lineage.sampled_point_estimate is not True
        or lineage.census_authority is not False
        or lineage.exact_stream_rows_available is not False
        or lineage.standard_fd_stream_acquisition_compatible is not False
        or lineage.exact_tangent_price_acquisition_compatible is not False
        or lineage.exact_candidate_numeric_state_issued is not False
        or lineage.separate_atom_cost_state_required is not True
        or lineage.decisive_fq_commit_required is not True
        or lineage.score_order_authority is not False
        or lineage.fq_certification_authority is not False
        or lineage.candidate_composition_authority is not False
        or lineage.candidate_freeze_authority is not False
        or lineage.bce_oracle_authority is not False
        or lineage.mask_commit_authority is not False
        or lineage.terminal_authority is not False
        or lineage.validation_inputs_present is not False
        or lineage.proposal_only is not True
    ):
        _refuse(
            V1FQCandidateQueueRefusalCode.AUTHORITY_VIOLATION,
            "sampled numeric lineage crossed its diagnostic authority boundary",
        )
    return {
        "sampled_corrected_point_lineage_hash": lineage_hash,
        "sampled_diagnostic_acquisition_hash": lineage.diagnostic_acquisition_hash,
        "sampled_summary_source_hash": lineage.sampled_summary_source_hash,
        "sampled_source_geometry_state_hash": lineage.source_geometry_state_hash,
        "sampled_source_geometry_result_hash": lineage.source_geometry_result_hash,
        "sampled_corrected_point_estimate_hash": (
            lineage.corrected_point_estimate_hash
        ),
        "sampled_corrected_point_status": lineage.corrected_point_status,
        "sampled_score_order_regret_receipt_axis_hash": (
            lineage.score_order_regret_receipt_axis_hash
        ),
        "sampled_fq_composition_gate_policy_id": (
            lineage.fq_composition_gate_policy_id
        ),
        "sampled_point_estimate": True,
        "sampled_census_authority": False,
        "sampled_exact_stream_rows_available": False,
        "sampled_exact_candidate_numeric_state_issued": False,
        "sampled_separate_atom_cost_state_required": True,
        "sampled_decisive_fq_commit_required": True,
        "sampled_score_order_authority": False,
        "sampled_fq_certification_authority": False,
        "sampled_candidate_composition_authority": False,
        "sampled_candidate_freeze_authority": False,
        "sampled_bce_oracle_authority": False,
        "sampled_mask_commit_authority": False,
        "sampled_terminal_authority": False,
        "sampled_validation_inputs_present": False,
        "sampled_proposal_only": True,
        "sampled_authority_scope": lineage.authority_scope,
    }


def _authorization_bridge_metadata(
    authorization: V1FQCandidateQueueAuthorization,
) -> dict[str, object]:
    common = {
        "fq_candidate_queue_authorization_hash": authorization.authorization_hash,
        "fq_registered_look_commit_hash": authorization.commit_hash,
        "fq_acquisition_registration_hash": authorization.registration_hash,
        "fq_frozen_ladder_hash": authorization.ladder_hash,
        "fq_budget_hash": authorization.budget_hash,
        "fq_composition_kind": authorization.composition_kind,
    }
    if _artifact_is_composition(authorization.artifacts):
        artifacts = authorization.artifacts
        assert type(artifacts) is V1CompositionFQArtifacts
        freeze = artifacts.freeze
        identity = freeze.identity
        result = freeze.selector_result
        return {
            **common,
            "fq_artifact_family": authorization.artifact_family,
            "fq_artifacts_hash": authorization.artifacts_hash,
            "fq_freeze_hash": authorization.freeze_hash,
            "fq_composition_artifacts_hash": authorization.composition_artifacts_hash,
            "fq_composition_freeze_hash": authorization.composition_freeze_hash,
            "fq_composition_scope_receipt_hash": (
                authorization.composition_scope_receipt_hash
            ),
            "fq_geometric_freeze_identity_hash": identity.identity_hash,
            "fq_geometric_replay_receipt_hash": freeze.replay_receipt.receipt_hash,
            "fq_geometric_selector_problem_hash": (
                freeze.selector_problem.problem_hash
            ),
            "fq_geometric_selector_policy_hash": freeze.selector_policy.policy_hash,
            "fq_geometric_selector_receipt_hash": result.receipt.receipt_hash,
            "fq_geometric_selector_result_hash": result.result_hash,
            "fq_sampled_geometry_source_hash": identity.sampled_geometry_source_hash,
            "fq_sampled_geometry_state_hash": identity.sampled_geometry_state_hash,
            "fq_sampled_geometry_result_hash": identity.sampled_geometry_result_hash,
            "fq_sampled_geometry_assembly_hash": (
                identity.sampled_geometry_assembly_hash
            ),
            "fq_sampled_proposal_position_certificate_hash": (
                identity.sampled_proposal_position_certificate_hash
            ),
        }
    return {
        **common,
        "fq_response_artifacts_hash": authorization.response_artifacts_hash,
        "fq_response_freeze_hash": authorization.response_freeze_hash,
    }


def compose_v1_fq_authorized_compact_candidates(
    *,
    authorization: V1FQCandidateQueueAuthorization,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    committed_mask_lineage_hashes: Sequence[str],
    committed_mask_transition_receipt_hashes: Sequence[str],
    atom_costs: V1AtomCostState,
    context_mixture_hash: str,
    sampled_numeric_lineage: V1SampledCorrectedPointNumericLineage | None = None,
    sampled_diagnostic_lineage_hash: str | None = None,
) -> V1FQCandidateCompositionResult:
    """Compose the compact, F/Q-authorized proposal queue.

    This is the production-shaped selected-shell bridge: it binds the decisive
    F/Q authorization, independently derives exact scalar resource release
    from the separate atom-cost state, and emits compact proposal witnesses.
    It does not construct the legacy full-route numeric state, evaluate the
    registered FD surrogate, invoke an oracle, freeze a candidate, or commit a
    mask.
    """

    if type(authorization) is not V1FQCandidateQueueAuthorization:
        _refuse(
            V1FQCandidateQueueRefusalCode.INVALID_TYPE,
            "compact F/Q composition requires an exact authorization",
        )
    authorization.assert_integrity()
    generation, _lineage, _transitions = _assert_authorized_current_coordinate(
        authorization=authorization,
        binding=binding,
        generation_index=generation_index,
        current_mask=current_mask,
        committed_mask_lineage_hashes=committed_mask_lineage_hashes,
        committed_mask_transition_receipt_hashes=(
            committed_mask_transition_receipt_hashes
        ),
    )
    if generation != authorization.generation_index:  # pragma: no cover - held above
        _refuse(
            V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            "compact F/Q generation changed during validation",
        )

    sampled_hash: str
    if _artifact_is_composition(authorization.artifacts):
        sampled_metadata = _sampled_lineage_metadata(
            authorization=authorization,
            binding=binding,
            generation_index=generation,
            current_mask=current_mask,
            context_mixture_hash=context_mixture_hash,
            sampled_numeric_lineage=sampled_numeric_lineage,
        )
        sampled_hash = str(sampled_metadata["fq_geometry_lineage_hash"])
        if (
            sampled_diagnostic_lineage_hash is not None
            and sampled_diagnostic_lineage_hash != sampled_hash
        ):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "sampled geometry lineage hash disagrees with the supplied "
                "diagnostic lineage hash",
            )
    elif sampled_numeric_lineage is not None:
        sampled_metadata = _sampled_lineage_metadata(
            authorization=authorization,
            binding=binding,
            generation_index=generation,
            current_mask=current_mask,
            context_mixture_hash=context_mixture_hash,
            sampled_numeric_lineage=sampled_numeric_lineage,
        )
        sampled_hash = str(sampled_metadata["sampled_corrected_point_lineage_hash"])
        if (
            sampled_diagnostic_lineage_hash is not None
            and sampled_diagnostic_lineage_hash != sampled_hash
        ):
            _refuse(
                V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
                "sampled lineage hash disagrees with the supplied typed lineage",
            )
    elif sampled_diagnostic_lineage_hash is None:
        _refuse(
            V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            "compact F/Q composition requires a sampled diagnostic lineage hash",
        )
    else:
        sampled_hash = _identity(
            sampled_diagnostic_lineage_hash,
            name="sampled diagnostic lineage hash",
        )

    rows, atom_cost_source_hash = _exact_resource_gain_rows_from_atom_costs(
        authorization=authorization,
        atom_costs=atom_costs,
    )
    states = issue_v1_fq_action_numeric_states(
        authorization=authorization,
        sampled_diagnostic_lineage_hash=sampled_hash,
        atom_cost_source_hash=atom_cost_source_hash,
        exact_resource_gain_rows=rows,
    )
    result = compose_v1_fq_candidates_in_authorized_resource_descending_order(
        authorization=authorization,
        action_numeric_states=states,
    )
    if (
        result.queue_endpoint_indices != authorization.queue_endpoint_indices_descending
        or result.queue_action_hashes != authorization.queue_action_hashes_descending
        or result.selected_proposal.resulting_mask_hash
        != authorization.ladder.endpoint_mask(
            endpoint_index=authorization.queue_head_endpoint_index
        ).mask_hash
        or result.exact_registered_surrogate_evaluation
        or result.exact_intervention_damage_evaluated
        or result.bce_oracle_authority
        or result.mask_commit_authority
        or result.terminal_authority
    ):
        _refuse(
            V1FQCandidateQueueRefusalCode.SILENT_FILTER,
            "compact F/Q composition filtered, reordered, or over-authorized a queue",
        )
    return result


def compose_v1_fq_authorized_compact_acquisition(
    *,
    authorization: V1FQCandidateQueueAuthorization,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    committed_mask_lineage_hashes: Sequence[str],
    committed_mask_transition_receipt_hashes: Sequence[str],
    atom_costs: V1AtomCostState,
    context_mixture_hash: str,
    sampled_numeric_lineage: V1SampledCorrectedPointNumericLineage | None = None,
    refresh_reason: str = "fresh_current_mask_fq_authorized_relinearization",
    logical_sequence: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> V1FQAuthorizedAcquisition:
    """Issue the compact controller acquisition from transient source graphs.

    The returned carrier retains neither the sampled summary graph, the full
    F/Q authorization graph, nor the route-wide atom-cost vector.  Their
    content identities are bound into the compact composition and receipt.
    """

    caller_metadata = dict(metadata or {})
    conflicts = sorted(_RESERVED_METADATA_KEYS.intersection(caller_metadata))
    if conflicts:
        _refuse(
            V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            f"caller metadata shadows reserved F/Q bindings: {conflicts}",
        )
    if (
        not _artifact_is_composition(authorization.artifacts)
        and type(sampled_numeric_lineage) is not V1SampledCorrectedPointNumericLineage
    ):
        _refuse(
            V1FQCandidateQueueRefusalCode.INVALID_TYPE,
            "compact acquisition requires the typed sampled numeric lineage",
        )
    generation, lineage, transitions = _assert_authorized_current_coordinate(
        authorization=authorization,
        binding=binding,
        generation_index=generation_index,
        current_mask=current_mask,
        committed_mask_lineage_hashes=committed_mask_lineage_hashes,
        committed_mask_transition_receipt_hashes=(
            committed_mask_transition_receipt_hashes
        ),
    )
    sampled_metadata = _sampled_lineage_metadata(
        authorization=authorization,
        binding=binding,
        generation_index=generation,
        current_mask=current_mask,
        context_mixture_hash=context_mixture_hash,
        sampled_numeric_lineage=sampled_numeric_lineage,
    )
    composition = compose_v1_fq_authorized_compact_candidates(
        authorization=authorization,
        binding=binding,
        generation_index=generation,
        current_mask=current_mask,
        committed_mask_lineage_hashes=lineage,
        committed_mask_transition_receipt_hashes=transitions,
        atom_costs=atom_costs,
        context_mixture_hash=context_mixture_hash,
        sampled_numeric_lineage=sampled_numeric_lineage,
    )
    bridge_metadata = {
        **sampled_metadata,
        **_authorization_bridge_metadata(authorization),
        "fq_queue_endpoint_indices_descending": (
            authorization.queue_endpoint_indices_descending
        ),
        "fq_queue_action_hashes_descending": (
            authorization.queue_action_hashes_descending
        ),
        "fq_comparator_deferred_endpoint_indices": (
            authorization.comparator_deferred_endpoint_indices
        ),
        "atom_cost_source_hash": composition.atom_cost_source_hash,
        "full_route_atom_cost_array_embedded": False,
        "full_sampled_summary_graph_embedded": False,
        "fq_additive_surrogate_used": False,
        "fq_bce_oracle_authority": False,
        "fq_candidate_freeze_authority": False,
        "fq_mask_commit_authority": False,
        "fq_terminal_authority": False,
    }
    return issue_v1_fq_authorized_acquisition(
        binding=binding,
        generation_index=generation,
        current_mask=current_mask,
        committed_mask_lineage_hashes=lineage,
        committed_mask_transition_receipt_hashes=transitions,
        composition=composition,
        refresh_reason=refresh_reason,
        logical_sequence=logical_sequence,
        metadata={**caller_metadata, **bridge_metadata},
    )


def compose_v1_fq_authorized_relinearized_acquisition(
    *,
    authorization: V1FQCandidateQueueAuthorization,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    committed_mask_lineage_hashes: Sequence[str],
    committed_mask_transition_receipt_hashes: Sequence[str],
    fd: V1FDStreamAcquisition,
    correction: (
        V1CorrectionAcquisition | V1AnalyticExactGateCorrectionAcquisition
    ),
    thresholds: V1ChartThresholds,
    constraints: V1CandidateCompositionConstraints,
    context_mixture_hash: str,
    residual_diagonal_status: ResidualDiagonalStatus = "proposal_only",
    residual_bound_proof_hash: str | None = None,
    mixed_difference_envelope: V1MixedDifferenceEnvelope | None = None,
    refresh_reason: str = "fresh_current_mask_fq_authorized_relinearization",
    logical_sequence: int = 0,
    sampled_numeric_lineage: V1SampledCorrectedPointNumericLineage | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> V1RelinearizedAcquisition:
    """Compose the exact currently admissible certified endpoints as a v1 queue."""

    if type(authorization) is not V1FQCandidateQueueAuthorization:
        _refuse(
            V1FQCandidateQueueRefusalCode.INVALID_TYPE,
            "F/Q queue composition requires an exact authorization",
        )
    authorization.assert_integrity()
    generation, lineage, transitions = _assert_authorized_current_coordinate(
        authorization=authorization,
        binding=binding,
        generation_index=generation_index,
        current_mask=current_mask,
        committed_mask_lineage_hashes=committed_mask_lineage_hashes,
        committed_mask_transition_receipt_hashes=(
            committed_mask_transition_receipt_hashes
        ),
    )

    caller_metadata = dict(metadata or {})
    conflicts = sorted(_RESERVED_METADATA_KEYS.intersection(caller_metadata))
    if conflicts:
        _refuse(
            V1FQCandidateQueueRefusalCode.COORDINATE_MISMATCH,
            f"caller metadata shadows reserved F/Q bindings: {conflicts}",
        )

    packed_actions = _authorized_packed_actions(authorization)
    if not packed_actions:
        _refuse(
            V1FQCandidateQueueRefusalCode.NO_ELIGIBLE_QUEUE,
            "F/Q authorization contains no currently admissible certified action",
        )
    sampled_metadata = _sampled_lineage_metadata(
        authorization=authorization,
        binding=binding,
        generation_index=generation,
        current_mask=current_mask,
        context_mixture_hash=context_mixture_hash,
        sampled_numeric_lineage=sampled_numeric_lineage,
    )
    declaration_hash = content_hash(
        {
            "schema": FQ_CANDIDATE_DOMAIN_DECLARATION_SCHEMA,
            "authorization_hash": authorization.authorization_hash,
            "fq_registered_look_commit_hash": authorization.commit_hash,
            "current_mask_hash": current_mask.mask_hash,
            "route_axis_hash": current_mask.route_axis_hash,
            "route_count": current_mask.route_count,
            "certified_endpoint_indices_descending": (
                authorization.certified_endpoint_indices_descending
            ),
            "certified_action_hashes_descending": (
                authorization.certified_action_hashes_descending
            ),
            "queue_endpoint_indices_descending": (
                authorization.queue_endpoint_indices_descending
            ),
            "queue_action_hashes_descending": (
                authorization.queue_action_hashes_descending
            ),
            "comparator_deferred_endpoint_indices": (
                authorization.comparator_deferred_endpoint_indices
            ),
            "packed_action_hashes": tuple(action.action_hash for action in packed_actions),
            "resulting_mask_hashes": tuple(
                action.resulting_mask.mask_hash for action in packed_actions
            ),
            "removed_route_index_hashes": tuple(
                action.removed_route_index_hash for action in packed_actions
            ),
            "fq_monotonicity_assumed": False,
        },
        prefix=_DOMAIN_DECLARATION_PREFIX,
    )
    domain = V1FiniteRemovalDomain(
        route_axis_hash=current_mask.route_axis_hash,
        route_count=current_mask.route_count,
        removal_sets=packed_actions,
        declaration_receipt_hash=declaration_hash,
    )
    bridge_metadata: dict[str, object] = {
        **_authorization_bridge_metadata(authorization),
        "fq_certified_endpoint_indices_descending": (
            authorization.certified_endpoint_indices_descending
        ),
        "fq_certified_action_hashes_descending": (
            authorization.certified_action_hashes_descending
        ),
        "fq_queue_endpoint_indices_descending": (
            authorization.queue_endpoint_indices_descending
        ),
        "fq_queue_action_hashes_descending": (
            authorization.queue_action_hashes_descending
        ),
        "fq_comparator_deferred_endpoint_indices": (
            authorization.comparator_deferred_endpoint_indices
        ),
        "fq_selected_endpoint_index": authorization.selected_endpoint_index,
        "fq_selected_action_hash": authorization.selected_action_hash,
        "fq_queue_head_endpoint_index": authorization.queue_head_endpoint_index,
        "fq_queue_head_action_hash": authorization.queue_head_action_hash,
        "fq_candidate_queue_authority_scope": authorization.authority_scope,
        "fq_ordinary_queue_materialization_authority": True,
        "fq_monotonicity_assumed": False,
        "fq_bce_oracle_authority": False,
        "fq_candidate_freeze_authority": False,
        "fq_mask_commit_authority": False,
        "fq_terminal_authority": False,
        **sampled_metadata,
    }
    acquisition = compose_v1_relinearized_acquisition(
        binding=binding,
        generation_index=generation,
        current_mask=current_mask,
        committed_mask_lineage_hashes=lineage,
        committed_mask_transition_receipt_hashes=transitions,
        fd=fd,
        correction=correction,
        thresholds=thresholds,
        domain=domain,
        constraints=constraints,
        context_mixture_hash=context_mixture_hash,
        residual_diagonal_status=residual_diagonal_status,
        residual_bound_proof_hash=residual_bound_proof_hash,
        mixed_difference_envelope=mixed_difference_envelope,
        refresh_reason=refresh_reason,
        logical_sequence=logical_sequence,
        candidate_composition_mode="frozen_resource_descending",
        metadata={**caller_metadata, **bridge_metadata},
    )

    # ``compose_v1_relinearized_acquisition`` binds caller metadata into the
    # chart contract, and therefore transitively into the acquisition hash,
    # while retaining a deliberately small top-level metadata projection.  At
    # this bridge the authorization and decisive commit must also be directly
    # inspectable on the ordinary acquisition and its generation receipt.
    # Reissue those two content-addressed carriers; no scientific coordinate
    # or proposal is changed.
    bound_metadata = {**dict(acquisition.metadata), **bridge_metadata}
    bound_acquisition_hash = content_hash(
        {
            "schema": acquisition.schema,
            "campaign_binding_hash": acquisition.binding.binding_hash,
            "generation_index": acquisition.generation_index,
            "current_mask_hash": acquisition.current_mask.mask_hash,
            "committed_mask_lineage_hashes": (
                acquisition.committed_mask_lineage_hashes
            ),
            "committed_mask_transition_receipt_hashes": (
                acquisition.committed_mask_transition_receipt_hashes
            ),
            "fd_acquisition_hash": acquisition.fd.acquisition_hash,
            "correction_acquisition_hash": acquisition.correction.acquisition_hash,
            "chart_hash": acquisition.chart.chart_hash,
            "numeric_state_hash": acquisition.numeric_state.state_hash,
            "domain_hash": acquisition.domain.domain_hash,
            "constraints_hash": acquisition.constraints.constraints_hash,
            "composition_result_hash": acquisition.composition.result_hash,
            "metadata": bound_metadata,
        },
        prefix="partition-free-v1-relinearized-acquisition-",
    )
    bound_receipt = replace(
        acquisition.receipt,
        acquisition_hash=bound_acquisition_hash,
        metadata={**dict(acquisition.receipt.metadata), **bridge_metadata},
    )
    acquisition = V1RelinearizedAcquisition(
        binding=acquisition.binding,
        generation_index=acquisition.generation_index,
        current_mask=acquisition.current_mask,
        committed_mask_lineage_hashes=acquisition.committed_mask_lineage_hashes,
        committed_mask_transition_receipt_hashes=(
            acquisition.committed_mask_transition_receipt_hashes
        ),
        fd=acquisition.fd,
        correction=acquisition.correction,
        chart=acquisition.chart,
        numeric_state=acquisition.numeric_state,
        domain=acquisition.domain,
        constraints=acquisition.constraints,
        composition=acquisition.composition,
        receipt=bound_receipt,
        metadata=bound_metadata,
    )

    expected_masks = tuple(action.resulting_mask.mask_hash for action in packed_actions)
    expected_counts = tuple(action.removed_count for action in packed_actions)
    expected_removal_hashes = tuple(
        action.removed_route_index_hash for action in packed_actions
    )
    proposals = acquisition.queue
    observed_masks = tuple(item.resulting_mask.mask_hash for item in proposals)
    observed_counts = tuple(item.removed_count for item in proposals)
    observed_removal_hashes = tuple(
        item.removed_route_index_hash for item in proposals
    )
    if not proposals:
        _refuse(
            V1FQCandidateQueueRefusalCode.NO_ELIGIBLE_QUEUE,
            "ordinary composition filtered every admitted certified endpoint",
        )
    if (
        acquisition.composition.composer_id
        != FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID
        or acquisition.composition.evaluated_action_count != len(packed_actions)
        or acquisition.composition.eligible_action_count != len(packed_actions)
        or len(proposals) != len(packed_actions)
        or observed_masks != expected_masks
        or observed_counts != expected_counts
        or observed_removal_hashes != expected_removal_hashes
    ):
        _refuse(
            V1FQCandidateQueueRefusalCode.SILENT_FILTER,
            "ordinary composition filtered, reordered, or changed an admitted endpoint",
        )
    if (
        proposals[0].resulting_mask.mask_hash
        != authorization.ladder.endpoint_mask(
            endpoint_index=authorization.queue_head_endpoint_index
        ).mask_hash
    ):
        _refuse(
            V1FQCandidateQueueRefusalCode.SILENT_FILTER,
            "ordinary queue no longer begins with the selected F/Q endpoint",
        )

    chart_metadata = acquisition.chart.build_contract.metadata
    if any(
        carrier.get(key) != value
        for carrier in (
            chart_metadata,
            acquisition.metadata,
            acquisition.receipt.metadata,
        )
        for key, value in bridge_metadata.items()
    ):
        _refuse(
            V1FQCandidateQueueRefusalCode.SILENT_FILTER,
            "ordinary acquisition did not bind the authorization graph in metadata",
        )
    if acquisition.claim_execution_certified or acquisition.mask_commit_authority:
        _refuse(
            V1FQCandidateQueueRefusalCode.AUTHORITY_VIOLATION,
            "ordinary acquisition converted F/Q evidence into execution authority",
        )
    return acquisition


__all__ = [
    "FQ_CANDIDATE_QUEUE_AUTHORIZATION_SCHEMA",
    "FQ_COMPOSITION_CANDIDATE_QUEUE_AUTHORIZATION_SCHEMA",
    "FQ_CANDIDATE_QUEUE_AUTHORITY_SCOPE",
    "V1FQCandidateQueueAuthorization",
    "V1FQCandidateQueueRefusal",
    "V1FQCandidateQueueRefusalCode",
    "authorize_v1_fq_candidate_queue",
    "compose_v1_fq_authorized_compact_acquisition",
    "compose_v1_fq_authorized_compact_candidates",
    "compose_v1_fq_authorized_relinearized_acquisition",
]
