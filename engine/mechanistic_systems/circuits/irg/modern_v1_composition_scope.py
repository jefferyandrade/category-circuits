"""Generic pre-outcome decision scope for a frozen modern-v1 ladder.

The historical response-exactification path used score-zero-regret evidence to
decide which ladder endpoints were decision-competitive.  A sampled
conditional-response-geometry selector has a different semantics: it may
freeze every emitted endpoint for later independent F/Q measurement without
claiming that any endpoint is score-resolved or globally optimal.

This module records that distinction without acquiring any downstream
authority.  The receipt is immutable and content addressed.  It binds the
discovery, geometry, selector, freeze, ladder, mask, generation, and route
coordinates that existed before fresh F/Q, BCE, or validation outcomes.  The
small selection helpers consume later endpoint classifications ephemerally;
they never add those outcomes to, or mutate, the pre-outcome receipt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Final, Literal

from .contracts import content_hash


COMPOSITION_DECISION_SCOPE_SCHEMA: Final = (
    "partition_free_v1_frozen_ladder_composition_decision_scope_v1"
)
LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND: Final = (
    "legacy_score_zero_regret_v1"
)
SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND: Final = (
    "sampled_conditional_response_geometry_v1"
)
COMPOSITION_DECISION_SCOPE_HASH_PREFIX: Final = (
    "partition-free-v1-frozen-ladder-composition-decision-scope-"
)
COMPOSITION_ACTION_AXIS_HASH_PREFIX: Final = (
    "partition-free-v1-composition-action-axis-"
)
COMPOSITION_COUNT_AXIS_HASH_PREFIX: Final = (
    "partition-free-v1-composition-count-axis-"
)

CompositionKind = Literal[
    "legacy_score_zero_regret_v1",
    "sampled_conditional_response_geometry_v1",
]

_COMPOSITION_KINDS: Final = {
    LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND,
    SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND,
}


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _optional_identity(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _identity(value, name=name)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one exact nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _identity_axis(values: Sequence[object], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an ordered identity sequence")
    axis = tuple(
        _identity(value, name=f"{name} item") for value in tuple(values)
    )
    if not axis or len(set(axis)) != len(axis):
        raise ValueError(f"{name} must be unique and nonempty")
    return axis


def _endpoint_axis(
    values: Sequence[object],
    *,
    endpoint_count: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an ordered integer sequence")
    axis = tuple(
        _positive_int(value, name=f"{name} item") for value in tuple(values)
    )
    if tuple(sorted(axis)) != axis or len(set(axis)) != len(axis):
        raise ValueError(f"{name} must be strictly increasing and unique")
    if axis and axis[-1] > endpoint_count:
        raise ValueError(f"{name} left the frozen ladder")
    return axis


@dataclass(frozen=True, slots=True)
class V1CompositionDecisionScopeReceipt:
    """Proposal-only decision scope for one immutable candidate ladder.

    ``composition_frozen_endpoint_indices`` and
    ``composition_provisional_endpoint_indices`` are the neutral partition.
    Geometry-composed endpoints are *not* exposed through a score-resolved
    alias.  Legacy callers must explicitly request
    :attr:`legacy_score_resolved_endpoint_indices`, which refuses on a
    geometry receipt.
    """

    composition_kind: CompositionKind
    discovery_score_registration_hash: str
    discovery_score_design_hash: str
    discovery_score_look_receipt_hash: str
    discovery_score_look_index: int
    maximum_registered_discovery_score_looks: int
    composition_freeze_identity_hash: str
    composition_freeze_replay_receipt_hash: str
    frozen_candidate_ladder_hash: str
    endpoint_action_hashes: tuple[str, ...]
    endpoint_new_removal_counts: tuple[int, ...]
    composition_frozen_endpoint_indices: tuple[int, ...]
    composition_provisional_endpoint_indices: tuple[int, ...]
    decision_competitive_endpoint_indices: tuple[int, ...]
    current_mask_hash: str
    proposal_generation: int
    route_count: int
    physical_route_axis_hash: str
    ordered_route_axis_hash: str

    sampled_geometry_state_hash: str | None = None
    sampled_geometry_result_hash: str | None = None
    sampled_geometry_assembly_hash: str | None = None
    sampled_selector_problem_hash: str | None = None
    sampled_selector_policy_hash: str | None = None
    sampled_selector_receipt_hash: str | None = None
    sampled_selector_result_hash: str | None = None
    sampled_proposal_position_certificate_hash: str | None = None
    legacy_score_zero_regret_receipt_hash: str | None = None
    legacy_score_zero_regret_endpoint_indices: tuple[int, ...] = ()

    proposal_only: bool = True
    proposal_optimality_authority: bool = False
    score_zero_regret_authority: bool = False
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    terminal_validation_access_authority: bool = False
    terminal_authority: bool = False
    frozen_before_fq_outcomes: bool = True
    fq_outcomes_observed: bool = False
    bce_outcomes_observed: bool = False
    validation_outcomes_observed: bool = False
    every_larger_frozen_endpoint_certified_infeasible_required: bool = True
    schema: str = COMPOSITION_DECISION_SCOPE_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        kind = _identity(self.composition_kind, name="composition kind")
        if kind not in _COMPOSITION_KINDS:
            raise ValueError("unsupported composition kind")
        object.__setattr__(self, "composition_kind", kind)
        for name in (
            "discovery_score_registration_hash",
            "discovery_score_design_hash",
            "discovery_score_look_receipt_hash",
            "composition_freeze_identity_hash",
            "composition_freeze_replay_receipt_hash",
            "frozen_candidate_ladder_hash",
            "current_mask_hash",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name),
            )
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        discovery_look_index = _positive_int(
            self.discovery_score_look_index,
            name="discovery score-look index",
        )
        maximum_discovery_looks = _positive_int(
            self.maximum_registered_discovery_score_looks,
            name="maximum registered discovery score looks",
        )
        if discovery_look_index > maximum_discovery_looks:
            raise ValueError(
                "discovery score-look index exceeds its preregistered axis"
            )
        route_count = _positive_int(self.route_count, name="route count")
        if self.physical_route_axis_hash == self.ordered_route_axis_hash:
            raise ValueError(
                "physical and ordered route axes must remain explicitly distinct"
            )
        action_hashes = _identity_axis(
            self.endpoint_action_hashes,
            name="endpoint action hashes",
        )
        counts = tuple(
            _positive_int(value, name="endpoint new removal count")
            for value in tuple(self.endpoint_new_removal_counts)
        )
        if (
            len(counts) != len(action_hashes)
            or tuple(sorted(counts)) != counts
            or len(set(counts)) != len(counts)
            or counts[-1] > route_count
        ):
            raise ValueError(
                "endpoint action/count axes must align and removal counts must be "
                "strictly increasing within the route axis"
            )
        endpoint_count = len(action_hashes)
        registered = tuple(range(1, endpoint_count + 1))
        frozen = _endpoint_axis(
            self.composition_frozen_endpoint_indices,
            endpoint_count=endpoint_count,
            name="composition-frozen endpoint indices",
        )
        provisional = _endpoint_axis(
            self.composition_provisional_endpoint_indices,
            endpoint_count=endpoint_count,
            name="composition-provisional endpoint indices",
        )
        competitive = _endpoint_axis(
            self.decision_competitive_endpoint_indices,
            endpoint_count=endpoint_count,
            name="decision-competitive endpoint indices",
        )
        if (
            tuple(sorted((*frozen, *provisional))) != registered
            or set(frozen).intersection(provisional)
        ):
            raise ValueError(
                "composition-frozen and provisional axes must exactly partition "
                "the frozen ladder"
            )
        if not set(competitive).issubset(frozen):
            raise PermissionError(
                "a decision-competitive endpoint lacks frozen composition"
            )

        sampled_names = (
            "sampled_geometry_state_hash",
            "sampled_geometry_result_hash",
            "sampled_geometry_assembly_hash",
            "sampled_selector_problem_hash",
            "sampled_selector_policy_hash",
            "sampled_selector_receipt_hash",
            "sampled_selector_result_hash",
        )
        sampled_hashes = tuple(
            _optional_identity(getattr(self, name), name=name)
            for name in sampled_names
        )
        for name, value in zip(sampled_names, sampled_hashes, strict=True):
            object.__setattr__(self, name, value)
        position_certificate_hash = _optional_identity(
            self.sampled_proposal_position_certificate_hash,
            name="sampled proposal-position certificate hash",
        )
        object.__setattr__(
            self,
            "sampled_proposal_position_certificate_hash",
            position_certificate_hash,
        )
        legacy_hash = _optional_identity(
            self.legacy_score_zero_regret_receipt_hash,
            name="legacy score-zero-regret receipt hash",
        )
        legacy_indices = _endpoint_axis(
            self.legacy_score_zero_regret_endpoint_indices,
            endpoint_count=endpoint_count,
            name="legacy score-zero-regret endpoint indices",
        )

        geometry_kind = (
            kind == SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND
        )
        if geometry_kind:
            if any(value is None for value in sampled_hashes):
                raise ValueError(
                    "sampled geometry composition requires complete geometry and "
                    "selector lineage"
                )
            if legacy_hash is not None or legacy_indices:
                raise ValueError(
                    "sampled geometry composition cannot carry legacy zero-regret "
                    "lineage"
                )
            if frozen != registered or provisional or competitive != registered:
                raise PermissionError(
                    "every geometry-selector endpoint must be composition-frozen "
                    "and decision-competitive"
                )
            if self.score_zero_regret_authority is not False:
                raise PermissionError(
                    "sampled geometry composition has no score-zero-regret authority"
                )
        else:
            if any(value is not None for value in sampled_hashes) or (
                position_certificate_hash is not None
            ):
                raise ValueError(
                    "legacy score-zero-regret composition cannot carry sampled "
                    "geometry or selector lineage"
                )
            if frozen != legacy_indices:
                raise ValueError(
                    "legacy composition-frozen endpoints must equal the explicit "
                    "zero-regret endpoint axis"
                )
            if bool(legacy_indices) is not (legacy_hash is not None):
                raise ValueError(
                    "legacy zero-regret endpoint and receipt identities are incomplete"
                )
            if self.score_zero_regret_authority is not bool(legacy_indices):
                raise PermissionError(
                    "legacy score-zero-regret authority differs from its certified axis"
                )

        exact_false_flags = (
            "proposal_optimality_authority",
            "fq_certification_authority",
            "bce_oracle_authority",
            "candidate_freeze_authority",
            "mask_commit_authority",
            "terminal_validation_access_authority",
            "terminal_authority",
            "fq_outcomes_observed",
            "bce_outcomes_observed",
            "validation_outcomes_observed",
        )
        for name in (
            "proposal_only",
            "score_zero_regret_authority",
            "frozen_before_fq_outcomes",
            "every_larger_frozen_endpoint_certified_infeasible_required",
            *exact_false_flags,
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be one exact boolean")
        if (
            self.proposal_only is not True
            or self.frozen_before_fq_outcomes is not True
            or self.every_larger_frozen_endpoint_certified_infeasible_required
            is not True
            or any(getattr(self, name) is not False for name in exact_false_flags)
        ):
            raise PermissionError(
                "composition decision scope crossed its pre-outcome proposal boundary"
            )
        if self.schema != COMPOSITION_DECISION_SCOPE_SCHEMA:
            raise ValueError("unsupported composition decision-scope schema")

        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(
            self,
            "discovery_score_look_index",
            discovery_look_index,
        )
        object.__setattr__(
            self,
            "maximum_registered_discovery_score_looks",
            maximum_discovery_looks,
        )
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "endpoint_action_hashes", action_hashes)
        object.__setattr__(self, "endpoint_new_removal_counts", counts)
        object.__setattr__(self, "composition_frozen_endpoint_indices", frozen)
        object.__setattr__(
            self,
            "composition_provisional_endpoint_indices",
            provisional,
        )
        object.__setattr__(
            self,
            "decision_competitive_endpoint_indices",
            competitive,
        )
        object.__setattr__(
            self,
            "legacy_score_zero_regret_receipt_hash",
            legacy_hash,
        )
        object.__setattr__(
            self,
            "legacy_score_zero_regret_endpoint_indices",
            legacy_indices,
        )
        self.assert_integrity()

    @property
    def endpoint_count(self) -> int:
        return len(self.endpoint_action_hashes)

    @property
    def registered_endpoint_indices(self) -> tuple[int, ...]:
        return tuple(range(1, self.endpoint_count + 1))

    @property
    def registered_endpoint_new_removal_counts(self) -> tuple[int, ...]:
        """Neutral count-axis spelling used by downstream F/Q consumers."""

        return self.endpoint_new_removal_counts

    @property
    def endpoint_action_axis_hash(self) -> str:
        return content_hash(
            self.endpoint_action_hashes,
            prefix=COMPOSITION_ACTION_AXIS_HASH_PREFIX,
        )

    @property
    def endpoint_count_axis_hash(self) -> str:
        return content_hash(
            self.endpoint_new_removal_counts,
            prefix=COMPOSITION_COUNT_AXIS_HASH_PREFIX,
        )

    @property
    def legacy_score_resolved_endpoint_indices(self) -> tuple[int, ...]:
        """Return the legacy axis, refusing the alias for geometry composition."""

        if self.composition_kind != LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND:
            raise PermissionError(
                "sampled geometry endpoints are composition-frozen, not "
                "legacy score-resolved"
            )
        return self.legacy_score_zero_regret_endpoint_indices

    def _later_fq_axes(
        self,
        *,
        within_budget_endpoint_indices: Sequence[object],
        certified_infeasible_endpoint_indices: Sequence[object],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        within = _endpoint_axis(
            within_budget_endpoint_indices,
            endpoint_count=self.endpoint_count,
            name="F/Q within-budget endpoint indices",
        )
        infeasible = _endpoint_axis(
            certified_infeasible_endpoint_indices,
            endpoint_count=self.endpoint_count,
            name="F/Q certified-infeasible endpoint indices",
        )
        if set(within).intersection(infeasible):
            raise ValueError(
                "an F/Q endpoint cannot be both within budget and certified infeasible"
            )
        return within, infeasible

    def fq_selection_blocking_endpoint_indices(
        self,
        *,
        within_budget_endpoint_indices: Sequence[object],
        certified_infeasible_endpoint_indices: Sequence[object],
    ) -> tuple[int, ...]:
        """Return suffix blockers for the largest currently eligible endpoint.

        An empty tuple means that :meth:`select_largest_fq_admissible_endpoint`
        may return the candidate.  Legacy provisional endpoints above the
        candidate remain score-scope blockers even if F/Q happened to measure
        them; geometry receipts have no provisional endpoints.
        """

        within, infeasible = self._later_fq_axes(
            within_budget_endpoint_indices=within_budget_endpoint_indices,
            certified_infeasible_endpoint_indices=(
                certified_infeasible_endpoint_indices
            ),
        )
        competitive = set(self.decision_competitive_endpoint_indices)
        candidates = tuple(index for index in within if index in competitive)
        if not candidates:
            return ()
        candidate = candidates[-1]
        unresolved_larger_frozen = tuple(
            index
            for index in self.composition_frozen_endpoint_indices
            if index > candidate and index not in set(infeasible)
        )
        larger_provisional = tuple(
            index
            for index in self.composition_provisional_endpoint_indices
            if index > candidate
        )
        return tuple(sorted((*unresolved_larger_frozen, *larger_provisional)))

    def select_largest_fq_admissible_endpoint(
        self,
        *,
        within_budget_endpoint_indices: Sequence[object],
        certified_infeasible_endpoint_indices: Sequence[object],
    ) -> int | None:
        """Safely select from later F/Q evidence without mutating this receipt.

        The largest decision-competitive within-budget endpoint is returned
        only after every larger composition-frozen endpoint is certified
        infeasible.  Unresolved larger endpoints therefore fail closed.
        """

        within, infeasible = self._later_fq_axes(
            within_budget_endpoint_indices=within_budget_endpoint_indices,
            certified_infeasible_endpoint_indices=(
                certified_infeasible_endpoint_indices
            ),
        )
        competitive = set(self.decision_competitive_endpoint_indices)
        candidates = tuple(index for index in within if index in competitive)
        if not candidates:
            return None
        candidate = candidates[-1]
        blockers = self.fq_selection_blocking_endpoint_indices(
            within_budget_endpoint_indices=within,
            certified_infeasible_endpoint_indices=infeasible,
        )
        if blockers:
            raise PermissionError(
                "F/Q selection is blocked until every larger frozen endpoint "
                f"is certified infeasible; blocking endpoints={blockers}"
            )
        return candidate

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "composition_kind": self.composition_kind,
            "discovery_score_registration_hash": (
                self.discovery_score_registration_hash
            ),
            "discovery_score_design_hash": self.discovery_score_design_hash,
            "discovery_score_look_receipt_hash": (
                self.discovery_score_look_receipt_hash
            ),
            "discovery_score_look_index": self.discovery_score_look_index,
            "maximum_registered_discovery_score_looks": (
                self.maximum_registered_discovery_score_looks
            ),
            "sampled_geometry_state_hash": self.sampled_geometry_state_hash,
            "sampled_geometry_result_hash": self.sampled_geometry_result_hash,
            "sampled_geometry_assembly_hash": self.sampled_geometry_assembly_hash,
            "sampled_selector_problem_hash": self.sampled_selector_problem_hash,
            "sampled_selector_policy_hash": self.sampled_selector_policy_hash,
            "sampled_selector_receipt_hash": self.sampled_selector_receipt_hash,
            "sampled_selector_result_hash": self.sampled_selector_result_hash,
            "sampled_proposal_position_certificate_hash": (
                self.sampled_proposal_position_certificate_hash
            ),
            "composition_freeze_identity_hash": (
                self.composition_freeze_identity_hash
            ),
            "composition_freeze_replay_receipt_hash": (
                self.composition_freeze_replay_receipt_hash
            ),
            "frozen_candidate_ladder_hash": self.frozen_candidate_ladder_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "endpoint_action_axis_hash": self.endpoint_action_axis_hash,
            "endpoint_new_removal_counts": self.endpoint_new_removal_counts,
            "endpoint_count_axis_hash": self.endpoint_count_axis_hash,
            "composition_frozen_endpoint_indices": (
                self.composition_frozen_endpoint_indices
            ),
            "composition_provisional_endpoint_indices": (
                self.composition_provisional_endpoint_indices
            ),
            "decision_competitive_endpoint_indices": (
                self.decision_competitive_endpoint_indices
            ),
            "legacy_score_zero_regret_receipt_hash": (
                self.legacy_score_zero_regret_receipt_hash
            ),
            "legacy_score_zero_regret_endpoint_indices": (
                self.legacy_score_zero_regret_endpoint_indices
            ),
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "route_count": self.route_count,
            "physical_route_axis_hash": self.physical_route_axis_hash,
            "ordered_route_axis_hash": self.ordered_route_axis_hash,
            "proposal_only": self.proposal_only,
            "proposal_optimality_authority": (
                self.proposal_optimality_authority
            ),
            "score_zero_regret_authority": self.score_zero_regret_authority,
            "fq_certification_authority": self.fq_certification_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_validation_access_authority": (
                self.terminal_validation_access_authority
            ),
            "terminal_authority": self.terminal_authority,
            "frozen_before_fq_outcomes": self.frozen_before_fq_outcomes,
            "fq_outcomes_observed": self.fq_outcomes_observed,
            "bce_outcomes_observed": self.bce_outcomes_observed,
            "validation_outcomes_observed": self.validation_outcomes_observed,
            "every_larger_frozen_endpoint_certified_infeasible_required": (
                self.every_larger_frozen_endpoint_certified_infeasible_required
            ),
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(),
            prefix=COMPOSITION_DECISION_SCOPE_HASH_PREFIX,
        )
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("composition decision-scope receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError(
                "composition decision-scope receipt hash does not match"
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def to_json_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


__all__ = [
    "COMPOSITION_ACTION_AXIS_HASH_PREFIX",
    "COMPOSITION_COUNT_AXIS_HASH_PREFIX",
    "COMPOSITION_DECISION_SCOPE_HASH_PREFIX",
    "COMPOSITION_DECISION_SCOPE_SCHEMA",
    "LEGACY_SCORE_ZERO_REGRET_COMPOSITION_KIND",
    "SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND",
    "CompositionKind",
    "V1CompositionDecisionScopeReceipt",
]
