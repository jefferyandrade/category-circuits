"""Pre-outcome response-exactification artifacts for sampled modern-v1.

Corrected-attribution sampling resolves an *ordered ordinal* prefix family.
It does not by itself create a physical mask, spend response-geometry alpha,
or authorize an irreversible transition.  This module closes precisely the
next artifact joint:

* replay the registered Serfling box and every action-regret identity;
* bridge ordinal position ``i`` to physical packed-mask bit ``i`` under an
  explicit, content-addressed two-axis receipt;
* freeze strictly nested packed child masks containing only new removals from
  the current mask;
* retain an exact comparator-density rung only when that scalar rung was
  registered before the score outcomes; and
* bind a selected-shell F/Q design to a common random certification panel whose
  physical rows may be reprojected across the finite precommitted score-look
  redesign family, while keeping every old statistic, certificate, ladder, and
  alpha allocation nontransferable.

The output remains proposal-only.  F/Q observations, candidate calibration,
the paired BCE oracle, mask commit, and terminal validation all remain false.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Final

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.evidence.glocal_contracts import (
    content_hash as sampling_content_hash,
)

from .contracts import content_hash
from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_composition_scope import (
    SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND,
    V1CompositionDecisionScopeReceipt,
)
from .modern_v1_current_look_fq_alpha_scope import (
    V1CurrentLookFQAlphaCoordinateScopeReceipt,
)
from .modern_v1_ladder_statistics import (
    ACTION_ROUTE_SUM_ENDPOINT_ENVELOPE_KIND,
    ACTION_SPECIFIC_ENDPOINT_ENVELOPE_KIND,
    V1FrozenCandidateLadder,
    V1FrozenLadderAction,
    V1FrozenLadderCertificationDesign,
    V1FrozenLadderEndpointEnvelopeCertificate,
)
from .modern_v1_alexnet_fq_registration import (
    V1AlexNetFQCausalGenerationFrameReceipt,
    V1AlexNetFQCausalRoleSamplingStreamReceipt,
    V1AlexNetFQDiscoveryPrefixFreezeReceipt,
)
from .modern_v1_proposal_alpha_ledger import V1ProposalOrderAlphaSchedule
from .modern_v1_proposal_statistics import (
    CORRECTED_ATTRIBUTION_CONFIDENCE_BOX_V2_SCHEMA,
    V1CorrectedAttributionActionFamily,
    V1CorrectedAttributionActionRegretRequest,
    V1CorrectedAttributionConfidenceBoxV2Receipt,
    V1CorrectedAttributionEnvelopeCertificate,
    V1GroupSequentialEmpiricalBernsteinSerflingDesign,
    V1GroupSequentialSerflingDesign,
    V1StratifiedCorrectedAttributionMoments,
    certify_corrected_attribution_action_regret,
    corrected_attribution_confidence_box,
    corrected_attribution_confidence_box_v2,
)
from .modern_v1_nested_action_regret import (
    V1NestedPrefixZeroRegretBatchRequest,
    certify_nested_prefix_zero_regret,
    nested_prefix_zero_regret_already_removed_hash,
    nested_prefix_zero_regret_live_hash,
    validate_nested_prefix_zero_regret_receipt_payload,
)
from .modern_v1_score_order_gpu import (
    NESTED_BOTTOM_K_ORDERED_PREFIX_LOGICAL_NAME,
    NESTED_BOTTOM_K_SELECTOR_ID,
    nested_bottom_k_ordered_prefix_hash,
    nested_bottom_k_physical_action_hash,
    open_nested_bottom_k_ordered_prefix,
    validate_nested_bottom_k_ordered_prefix_descriptor,
    validate_nested_bottom_k_receipt_payload,
)


MASTER_DECISION_ALPHA_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_master_decision_alpha_registration_v2"
)
FQ_ALPHA_ALLOCATION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_fq_alpha_family_allocation_receipt_v3"
)
FQ_SCORE_LOOK_PROPORTIONAL_REDESIGN_POLICY: Final = (
    "preoutcome_proposal_look_alpha_proportional_nonrecycling_v1"
)
FQ_COMMON_CERTIFICATION_PANEL_POLICY: Final = (
    "common_random_certification_panel_reprojected_across_"
    "precommitted_score_look_ladders_v1"
)
FQ_COMMON_CERTIFICATION_PANEL_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_fq_common_certification_panel_receipt_v1"
)
ORACLE_ATTEMPT_ALPHA_AXIS_SCHEMA: Final = (
    "partition_free_v1_oracle_attempt_alpha_axis_v1"
)
ORDINAL_PHYSICAL_AXIS_BRIDGE_SCHEMA: Final = (
    "partition_free_v1_ordinal_physical_axis_bridge_receipt_v1"
)
RESPONSE_EXACTIFICATION_REPLAY_SCHEMA: Final = (
    "partition_free_v1_response_exactification_replay_receipt_v2"
)
RESPONSE_EXACTIFICATION_HYBRID_REPLAY_SCHEMA: Final = (
    "partition_free_v1_response_exactification_replay_receipt_v3"
)
RESPONSE_EXACTIFICATION_FREEZE_SCHEMA: Final = (
    "partition_free_v1_response_exactification_freeze_v2"
)
RESPONSE_EXACTIFICATION_HYBRID_FREEZE_SCHEMA: Final = (
    "partition_free_v1_response_exactification_freeze_v3"
)
RESPONSE_EXACTIFICATION_DECISION_SCOPE_SCHEMA: Final = (
    "partition_free_v1_response_exactification_decision_scope_v1"
)
CANCELLATION_ENDPOINT_BUNDLE_SCHEMA: Final = (
    "partition_free_v1_cancellation_endpoint_envelope_bundle_v1"
)
RESPONSE_EXACTIFICATION_ARTIFACTS_SCHEMA: Final = (
    "partition_free_v1_response_exactification_artifacts_v2"
)

_MASTER_ALPHA_PREFIX: Final = "partition-free-v1-master-decision-alpha-v2-"
_FQ_ALPHA_PREFIX: Final = "partition-free-v1-fq-alpha-family-allocation-v3-"
_FQ_COMMON_PANEL_PREFIX: Final = (
    "partition-free-v1-fq-common-certification-panel-v1-"
)
_ORACLE_ATTEMPT_ALPHA_AXIS_PREFIX: Final = (
    "partition-free-v1-oracle-attempt-alpha-axis-"
)
_AXIS_BRIDGE_PREFIX: Final = "partition-free-v1-ordinal-physical-axis-bridge-"
_REPLAY_PREFIX: Final = "partition-free-v1-response-exactification-replay-v2-"
_HYBRID_REPLAY_PREFIX: Final = (
    "partition-free-v1-response-exactification-replay-v3-"
)
_FREEZE_PREFIX: Final = "partition-free-v1-response-exactification-freeze-v2-"
_HYBRID_FREEZE_PREFIX: Final = (
    "partition-free-v1-response-exactification-freeze-v3-"
)
_DECISION_SCOPE_PREFIX: Final = (
    "partition-free-v1-response-exactification-decision-scope-"
)
_ENDPOINT_BUNDLE_PREFIX: Final = (
    "partition-free-v1-cancellation-endpoint-envelope-bundle-"
)
_ARTIFACTS_PREFIX: Final = "partition-free-v1-response-exactification-artifacts-v2-"
_SCORE_REGISTRATION_PREFIX: Final = (
    "partition-free-v1-alexnet-score-order-registration-"
)
_SCORE_LOOK_PREFIX: Final = (
    "partition-free-v1-alexnet-score-order-look-receipt-"
)
_SCORE_PROPOSAL_SOURCE_PREFIX: Final = (
    "partition-free-v1-alexnet-score-order-proposal-source-"
)
_ORDERED_ACTION_AXIS_PREFIX: Final = (
    "partition-free-v1-alexnet-score-order-action-axis-"
)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be one positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return result


def _endpoint_index_axis(
    values: Sequence[object],
    *,
    endpoint_count: int,
    name: str,
) -> tuple[int, ...]:
    axis = tuple(_positive_int(value, name=f"{name} item") for value in values)
    if tuple(sorted(axis)) != axis or len(set(axis)) != len(axis):
        raise ValueError(f"{name} must be strictly increasing and unique")
    if axis and axis[-1] > endpoint_count:
        raise ValueError(f"{name} left the registered endpoint axis")
    return axis


def _alpha(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be one finite alpha")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result >= 1.0:
        raise ValueError(f"{name} must lie in [0,1)")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _alpha_tolerance(*values: float) -> float:
    return 64.0 * math.ulp(max(1.0, *(abs(value) for value in values)))


def _alpha_matches(left: float, right: float) -> bool:
    return abs(left - right) <= _alpha_tolerance(left, right)


def _direct_mask_hash(mask: HardMask, *, name: str) -> str:
    if type(mask) is not HardMask:
        raise TypeError(f"{name} must use the exact HardMask type")
    count = _positive_int(mask.route_count, name=f"{name} route count")
    axis_hash = _identity(mask.route_axis_hash, name=f"{name} route axis hash")
    packed = mask.packed_bits
    if type(packed) is not bytes or len(packed) != (count + 7) // 8:
        raise ValueError(f"{name} packed bytes differ from its route count")
    padding = len(packed) * 8 - count
    if padding and packed[-1] & ((1 << padding) - 1):
        raise ValueError(f"{name} has noncanonical padding bits")
    return content_hash(
        {
            "route_axis_hash": axis_hash,
            "route_count": count,
            "packed_bits": packed,
        },
        prefix="irg-mask-",
    )


def _mapping_hash(
    value: Mapping[str, object],
    *,
    hash_field: str,
    prefix: str,
    name: str,
) -> str:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    payload = dict(value)
    expected = _identity(payload.pop(hash_field, None), name=f"{name} hash")
    observed = content_hash(payload, prefix=prefix)
    if observed != expected:
        raise ValueError(f"{name} was tampered")
    return observed


def _frozen_context_axis(
    values: Sequence[tuple[str, Sequence[str]]],
    *,
    name: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    rows: list[tuple[str, tuple[str, ...]]] = []
    for stratum, contexts in values:
        stratum_id = _identity(stratum, name=f"{name} stratum")
        axis = tuple(_identity(item, name=f"{name} context") for item in contexts)
        if not axis or len(axis) != len(set(axis)):
            raise ValueError(f"{name} contexts must be nonempty and unique")
        rows.append((stratum_id, axis))
    if not rows or len({name for name, _ in rows}) != len(rows):
        raise ValueError(f"{name} strata must be nonempty and unique")
    flat = tuple(context for _, axis in rows for context in axis)
    if len(flat) != len(set(flat)):
        raise ValueError(f"{name} contexts must be globally unique")
    return tuple(rows)


def freeze_v1_fq_certification_context_axis(
    values: Sequence[tuple[str, Sequence[str]]],
    *,
    name: str = "F/Q certification common panel",
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Freeze one globally unique, nonempty certification context axis."""

    return _frozen_context_axis(values, name=name)


@dataclass(frozen=True, slots=True)
class V1MasterDecisionAlphaRegistration:
    """One pre-outcome, nonrecycling alpha split across decision families.

    Proposal-order and F/Q allocations live on the finite proposal-generation
    axis.  Paired-oracle allocations instead live on the finite *global
    candidate-attempt* axis: rejected candidates still consume an attempt and
    must never cause oracle alpha to be recycled or silently reindexed by the
    next proposal generation.
    """

    campaign_contract_hash: str
    controller_run_hash: str
    preoutcome_registration_receipt_hash: str
    proposal_order_alpha_schedule_hash: str
    maximum_proposal_generations: int
    maximum_oracle_attempts: int
    master_familywise_alpha: float
    proposal_order_familywise_alpha: float
    fq_familywise_alpha: float
    oracle_familywise_alpha: float
    proposal_order_generation_allocations: tuple[float, ...]
    fq_generation_allocations: tuple[float, ...]
    oracle_attempt_allocations: tuple[float, ...]
    unallocated_alpha_reserve: float = 0.0
    registered_before_any_decision_outcomes: bool = True
    family_allocations_nontransferable: bool = True
    unused_alpha_nonrecycling: bool = True
    oracle_reserve_nontransferable: bool = True
    schema: str = MASTER_DECISION_ALPHA_REGISTRATION_SCHEMA
    _registration_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "controller_run_hash",
            "preoutcome_registration_receipt_hash",
            "proposal_order_alpha_schedule_hash",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name)
            )
        horizon = _positive_int(
            self.maximum_proposal_generations,
            name="maximum proposal generations",
        )
        oracle_horizon = _positive_int(
            self.maximum_oracle_attempts,
            name="maximum oracle attempts",
        )
        master = _alpha(
            self.master_familywise_alpha,
            name="master familywise alpha",
            positive=True,
        )
        proposal_total = _alpha(
            self.proposal_order_familywise_alpha,
            name="proposal-order familywise alpha",
            positive=True,
        )
        fq_total = _alpha(
            self.fq_familywise_alpha,
            name="F/Q familywise alpha",
            positive=True,
        )
        oracle_total = _alpha(
            self.oracle_familywise_alpha,
            name="oracle familywise alpha",
            positive=True,
        )
        reserve = _alpha(
            self.unallocated_alpha_reserve,
            name="unallocated alpha reserve",
        )

        generation_axes: dict[str, tuple[float, ...]] = {}
        for name in (
            "proposal_order_generation_allocations",
            "fq_generation_allocations",
        ):
            axis = tuple(
                _alpha(value, name=f"{name} item", positive=True)
                for value in getattr(self, name)
            )
            if len(axis) != horizon:
                raise ValueError(f"{name} must cover the complete finite horizon")
            generation_axes[name] = axis
            object.__setattr__(self, name, axis)

        oracle_axis = tuple(
            _alpha(value, name="oracle_attempt_allocations item", positive=True)
            for value in self.oracle_attempt_allocations
        )
        if len(oracle_axis) != oracle_horizon:
            raise ValueError(
                "oracle_attempt_allocations must cover the complete finite "
                "candidate-attempt horizon"
            )
        object.__setattr__(self, "oracle_attempt_allocations", oracle_axis)

        totals = (proposal_total, fq_total)
        allocations = (
            generation_axes["proposal_order_generation_allocations"],
            generation_axes["fq_generation_allocations"],
        )
        for label, total, axis in zip(
            ("proposal-order", "F/Q"),
            totals,
            allocations,
            strict=True,
        ):
            committed = math.fsum(axis)
            if committed > total + _alpha_tolerance(committed, total):
                raise ValueError(f"{label} generation allocations overspend its family")
        oracle_committed = math.fsum(oracle_axis)
        if not _alpha_matches(oracle_committed, oracle_total):
            raise ValueError(
                "oracle attempt allocations must exactly close the "
                "nontransferable oracle family"
            )
        registered_total = math.fsum(
            (proposal_total, fq_total, oracle_total, reserve)
        )
        if not _alpha_matches(registered_total, master):
            raise ValueError(
                "master alpha must equal the three nontransferable families plus reserve"
            )
        if (
            self.registered_before_any_decision_outcomes is not True
            or self.family_allocations_nontransferable is not True
            or self.unused_alpha_nonrecycling is not True
            or self.oracle_reserve_nontransferable is not True
        ):
            raise PermissionError("master decision-alpha registration crossed its boundary")
        if self.schema != MASTER_DECISION_ALPHA_REGISTRATION_SCHEMA:
            raise ValueError("unsupported master decision-alpha registration schema")
        object.__setattr__(self, "maximum_proposal_generations", horizon)
        object.__setattr__(self, "maximum_oracle_attempts", oracle_horizon)
        object.__setattr__(self, "master_familywise_alpha", master)
        object.__setattr__(self, "proposal_order_familywise_alpha", proposal_total)
        object.__setattr__(self, "fq_familywise_alpha", fq_total)
        object.__setattr__(self, "oracle_familywise_alpha", oracle_total)
        object.__setattr__(self, "unallocated_alpha_reserve", reserve)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "campaign_contract_hash": self.campaign_contract_hash,
            "controller_run_hash": self.controller_run_hash,
            "preoutcome_registration_receipt_hash": (
                self.preoutcome_registration_receipt_hash
            ),
            "proposal_order_alpha_schedule_hash": (
                self.proposal_order_alpha_schedule_hash
            ),
            "maximum_proposal_generations": self.maximum_proposal_generations,
            "maximum_oracle_attempts": self.maximum_oracle_attempts,
            "master_familywise_alpha": self.master_familywise_alpha,
            "proposal_order_familywise_alpha": (
                self.proposal_order_familywise_alpha
            ),
            "fq_familywise_alpha": self.fq_familywise_alpha,
            "oracle_familywise_alpha": self.oracle_familywise_alpha,
            "proposal_order_generation_allocations": (
                self.proposal_order_generation_allocations
            ),
            "fq_generation_allocations": self.fq_generation_allocations,
            "oracle_attempt_allocations": self.oracle_attempt_allocations,
            "unallocated_alpha_reserve": self.unallocated_alpha_reserve,
            "registered_before_any_decision_outcomes": (
                self.registered_before_any_decision_outcomes
            ),
            "family_allocations_nontransferable": (
                self.family_allocations_nontransferable
            ),
            "unused_alpha_nonrecycling": self.unused_alpha_nonrecycling,
            "oracle_reserve_nontransferable": (
                self.oracle_reserve_nontransferable
            ),
        }

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()

    def to_payload(self) -> Mapping[str, object]:
        """Return the complete integrity-checked preregistration payload."""

        self.assert_integrity()
        return MappingProxyType(self._payload())

    @property
    def oracle_attempt_alpha_axis_hash(self) -> str:
        """Identity of the complete global oracle-attempt allocation axis."""

        self.assert_integrity()
        return content_hash(
            {
                "schema": ORACLE_ATTEMPT_ALPHA_AXIS_SCHEMA,
                "campaign_contract_hash": self.campaign_contract_hash,
                "controller_run_hash": self.controller_run_hash,
                "preoutcome_registration_receipt_hash": (
                    self.preoutcome_registration_receipt_hash
                ),
                "maximum_oracle_attempts": self.maximum_oracle_attempts,
                "oracle_familywise_alpha": self.oracle_familywise_alpha,
                "oracle_attempt_allocations": self.oracle_attempt_allocations,
                "allocation_axis": "global_candidate_attempt_index",
                "unused_alpha_nonrecycling": self.unused_alpha_nonrecycling,
                "oracle_reserve_nontransferable": (
                    self.oracle_reserve_nontransferable
                ),
            },
            prefix=_ORACLE_ATTEMPT_ALPHA_AXIS_PREFIX,
        )

    def oracle_alpha_for_attempt(self, candidate_attempt_index: int) -> float:
        """Return the preregistered alpha for one global candidate attempt."""

        attempt = _nonnegative_int(
            candidate_attempt_index,
            name="candidate attempt index",
        )
        if attempt >= self.maximum_oracle_attempts:
            raise IndexError("oracle attempt alpha axis is exhausted")
        return self.oracle_attempt_allocations[attempt]

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_MASTER_ALPHA_PREFIX)
        if self._registration_hash is not None and observed != self._registration_hash:
            raise ValueError("master decision-alpha registration was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("master decision-alpha registration hash does not match")
        object.__setattr__(self, "_registration_hash", observed)
        return observed

    def _constructor_payload(self) -> dict[str, object]:
        return dict(self._payload())

    def __reduce__(self) -> tuple[object, tuple[dict[str, object], str]]:
        return _rebuild_master_decision_alpha_registration, (
            self._constructor_payload(),
            self.assert_integrity(),
        )


@dataclass(frozen=True, slots=True)
class V1FQAlphaFamilyAllocationReceipt:
    """The independent selected-shell F/Q share for one generation.

    Oracle alpha is recorded only as a nontransferable family reserve plus the
    identity of its global candidate-attempt allocation axis.  This receipt
    deliberately contains no proposal-generation-indexed oracle allocation.
    """

    master_registration_hash: str
    proposal_order_alpha_schedule_hash: str
    proposal_generation: int
    fq_familywise_alpha: float
    fq_alpha_committed_before_generation: float
    fq_generation_alpha_allocation: float
    proposal_order_generation_alpha_allocation: float
    oracle_familywise_alpha_reserve: float
    maximum_oracle_attempts: int
    oracle_attempt_alpha_axis_hash: str
    fq_ladder_redesign_generation: int = 0
    fq_alpha_committed_before_redesign: float = 0.0
    fq_redesign_alpha_allocation: float | None = None
    score_order_look_index: int = 1
    maximum_fq_ladder_redesigns: int = 1
    proposal_order_look_alpha_weights: tuple[float, ...] = ()
    fq_redesign_alpha_allocations: tuple[float, ...] = ()
    fq_redesign_alpha_schedule_hash: str | None = None
    fq_redesign_allocation_policy: str = (
        FQ_SCORE_LOOK_PROPORTIONAL_REDESIGN_POLICY
    )
    redesign_alpha_unused_nonrecycling: bool = True
    redesign_alpha_nontransferable_across_score_looks: bool = True
    proposal_alpha_reused_for_fq: bool = False
    oracle_alpha_transferable_to_fq: bool = False
    mask_commit_authority: bool = False
    schema: str = FQ_ALPHA_ALLOCATION_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "master_registration_hash",
            "proposal_order_alpha_schedule_hash",
            "oracle_attempt_alpha_axis_hash",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name)
            )
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        redesign = _nonnegative_int(
            self.fq_ladder_redesign_generation, name="F/Q ladder redesign generation"
        )
        score_look = _positive_int(
            self.score_order_look_index, name="score-order look index"
        )
        maximum_redesigns = _positive_int(
            self.maximum_fq_ladder_redesigns,
            name="maximum F/Q ladder redesigns",
        )
        if score_look > maximum_redesigns or redesign != score_look - 1:
            raise ValueError(
                "F/Q ladder redesign generation must equal score look index minus one"
            )
        familywise = _alpha(
            self.fq_familywise_alpha, name="F/Q familywise alpha", positive=True
        )
        committed = _alpha(
            self.fq_alpha_committed_before_generation,
            name="F/Q alpha committed before generation",
        )
        generation_alpha = _alpha(
            self.fq_generation_alpha_allocation,
            name="F/Q generation alpha allocation",
            positive=True,
        )
        proposal_alpha = _alpha(
            self.proposal_order_generation_alpha_allocation,
            name="proposal-order generation alpha allocation",
            positive=True,
        )
        oracle_reserve = _alpha(
            self.oracle_familywise_alpha_reserve,
            name="oracle familywise alpha reserve",
            positive=True,
        )
        oracle_horizon = _positive_int(
            self.maximum_oracle_attempts,
            name="maximum oracle attempts",
        )
        committed_redesign = _alpha(
            self.fq_alpha_committed_before_redesign,
            name="F/Q alpha committed before redesign",
        )
        redesign_alpha = _alpha(
            (
                generation_alpha
                if self.fq_redesign_alpha_allocation is None
                else self.fq_redesign_alpha_allocation
            ),
            name="F/Q redesign alpha allocation",
            positive=True,
        )
        if committed + generation_alpha > familywise + _alpha_tolerance(
            committed, generation_alpha, familywise
        ):
            raise ValueError("F/Q generation allocation overspends its family")
        redesign_axis = tuple(
            _alpha(value, name="F/Q redesign alpha schedule item", positive=True)
            for value in self.fq_redesign_alpha_allocations
        )
        if not redesign_axis:
            if maximum_redesigns != 1 or score_look != 1 or redesign != 0:
                raise ValueError(
                    "multi-look F/Q redesigns require an explicit preoutcome schedule"
                )
            redesign_axis = (generation_alpha,)
        if len(redesign_axis) != maximum_redesigns:
            raise ValueError(
                "F/Q redesign alpha schedule must cover every registered score look"
            )
        proposal_look_weights = tuple(
            _alpha(value, name="proposal look alpha weight", positive=True)
            for value in self.proposal_order_look_alpha_weights
        )
        if not proposal_look_weights:
            if maximum_redesigns != 1:
                raise ValueError(
                    "multi-look F/Q redesigns require proposal-look alpha weights"
                )
            proposal_look_weights = (generation_alpha,)
        if len(proposal_look_weights) != maximum_redesigns:
            raise ValueError(
                "proposal-look alpha weights must cover every F/Q redesign"
            )
        weight_total = math.fsum(proposal_look_weights)
        expected_redesign_axis_list = [
            generation_alpha * weight / weight_total
            for weight in proposal_look_weights[:-1]
        ]
        expected_redesign_axis_list.append(
            generation_alpha - math.fsum(expected_redesign_axis_list)
        )
        expected_redesign_axis = tuple(expected_redesign_axis_list)
        if redesign_axis != expected_redesign_axis:
            raise ValueError(
                "F/Q redesign schedule is not proportional to the registered "
                "proposal-look alpha axis"
            )
        schedule_hash = content_hash(
            {
                "proposal_generation": generation,
                "maximum_fq_ladder_redesigns": maximum_redesigns,
                "fq_generation_alpha_allocation": generation_alpha,
                "proposal_order_look_alpha_weights": proposal_look_weights,
                "fq_redesign_alpha_allocations": redesign_axis,
                "allocation_policy": self.fq_redesign_allocation_policy,
                "proposal_alpha_reused_for_fq": False,
            },
            prefix="partition-free-v1-fq-redesign-alpha-schedule-",
        )
        if (
            self.fq_redesign_alpha_schedule_hash is not None
            and _identity(
                self.fq_redesign_alpha_schedule_hash,
                name="F/Q redesign alpha schedule hash",
            )
            != schedule_hash
        ):
            raise ValueError("F/Q redesign alpha schedule hash does not match")
        expected_committed_redesign = math.fsum(redesign_axis[:redesign])
        if (
            not _alpha_matches(math.fsum(redesign_axis), generation_alpha)
            or committed_redesign != expected_committed_redesign
            or redesign_alpha != redesign_axis[redesign]
        ):
            raise ValueError(
                "F/Q redesign allocation differs from its preregistered score-look axis"
            )
        if committed_redesign + redesign_alpha > generation_alpha + _alpha_tolerance(
            committed_redesign, redesign_alpha, generation_alpha
        ):
            raise ValueError("F/Q redesign allocation overspends its generation")
        if (
            self.proposal_alpha_reused_for_fq is not False
            or self.fq_redesign_allocation_policy
            != FQ_SCORE_LOOK_PROPORTIONAL_REDESIGN_POLICY
            or self.redesign_alpha_unused_nonrecycling is not True
            or self.redesign_alpha_nontransferable_across_score_looks is not True
            or self.oracle_alpha_transferable_to_fq is not False
            or self.mask_commit_authority is not False
        ):
            raise PermissionError("F/Q alpha allocation crossed its authority boundary")
        if self.schema != FQ_ALPHA_ALLOCATION_RECEIPT_SCHEMA:
            raise ValueError("unsupported F/Q alpha allocation receipt schema")
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "fq_ladder_redesign_generation", redesign)
        object.__setattr__(self, "score_order_look_index", score_look)
        object.__setattr__(
            self, "maximum_fq_ladder_redesigns", maximum_redesigns
        )
        object.__setattr__(self, "fq_familywise_alpha", familywise)
        object.__setattr__(self, "fq_alpha_committed_before_generation", committed)
        object.__setattr__(self, "fq_generation_alpha_allocation", generation_alpha)
        object.__setattr__(
            self, "proposal_order_generation_alpha_allocation", proposal_alpha
        )
        object.__setattr__(
            self, "oracle_familywise_alpha_reserve", oracle_reserve
        )
        object.__setattr__(self, "maximum_oracle_attempts", oracle_horizon)
        object.__setattr__(
            self, "fq_alpha_committed_before_redesign", committed_redesign
        )
        object.__setattr__(self, "fq_redesign_alpha_allocation", redesign_alpha)
        object.__setattr__(self, "fq_redesign_alpha_allocations", redesign_axis)
        object.__setattr__(
            self, "proposal_order_look_alpha_weights", proposal_look_weights
        )
        object.__setattr__(
            self, "fq_redesign_alpha_schedule_hash", schedule_hash
        )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "master_registration_hash": self.master_registration_hash,
            "proposal_order_alpha_schedule_hash": (
                self.proposal_order_alpha_schedule_hash
            ),
            "proposal_generation": self.proposal_generation,
            "fq_familywise_alpha": self.fq_familywise_alpha,
            "fq_alpha_committed_before_generation": (
                self.fq_alpha_committed_before_generation
            ),
            "fq_generation_alpha_allocation": self.fq_generation_alpha_allocation,
            "proposal_order_generation_alpha_allocation": (
                self.proposal_order_generation_alpha_allocation
            ),
            "oracle_familywise_alpha_reserve": (
                self.oracle_familywise_alpha_reserve
            ),
            "maximum_oracle_attempts": self.maximum_oracle_attempts,
            "oracle_attempt_alpha_axis_hash": self.oracle_attempt_alpha_axis_hash,
            "fq_ladder_redesign_generation": self.fq_ladder_redesign_generation,
            "fq_alpha_committed_before_redesign": (
                self.fq_alpha_committed_before_redesign
            ),
            "fq_redesign_alpha_allocation": self.fq_redesign_alpha_allocation,
            "score_order_look_index": self.score_order_look_index,
            "maximum_fq_ladder_redesigns": self.maximum_fq_ladder_redesigns,
            "proposal_order_look_alpha_weights": (
                self.proposal_order_look_alpha_weights
            ),
            "fq_redesign_alpha_allocations": self.fq_redesign_alpha_allocations,
            "fq_redesign_alpha_schedule_hash": (
                self.fq_redesign_alpha_schedule_hash
            ),
            "fq_redesign_allocation_policy": (
                self.fq_redesign_allocation_policy
            ),
            "redesign_alpha_unused_nonrecycling": (
                self.redesign_alpha_unused_nonrecycling
            ),
            "redesign_alpha_nontransferable_across_score_looks": (
                self.redesign_alpha_nontransferable_across_score_looks
            ),
            "proposal_alpha_reused_for_fq": self.proposal_alpha_reused_for_fq,
            "oracle_alpha_transferable_to_fq": self.oracle_alpha_transferable_to_fq,
            "mask_commit_authority": self.mask_commit_authority,
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_FQ_ALPHA_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("F/Q alpha allocation receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("F/Q alpha allocation receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def _constructor_payload(self) -> dict[str, object]:
        return dict(self._payload())

    def __reduce__(self) -> tuple[object, tuple[dict[str, object], str]]:
        return _rebuild_fq_alpha_family_allocation_receipt, (
            self._constructor_payload(),
            self.assert_integrity(),
        )


def _rebuild_master_decision_alpha_registration(
    payload: Mapping[str, object],
    expected_hash: str,
) -> V1MasterDecisionAlphaRegistration:
    rebuilt = V1MasterDecisionAlphaRegistration(**dict(payload))  # type: ignore[arg-type]
    rebuilt.assert_integrity(_identity(expected_hash, name="expected master hash"))
    return rebuilt


def _rebuild_fq_alpha_family_allocation_receipt(
    payload: Mapping[str, object],
    expected_hash: str,
) -> V1FQAlphaFamilyAllocationReceipt:
    rebuilt = V1FQAlphaFamilyAllocationReceipt(**dict(payload))  # type: ignore[arg-type]
    rebuilt.assert_integrity(_identity(expected_hash, name="expected F/Q receipt hash"))
    return rebuilt


def _rebuild_fq_common_certification_panel_receipt(
    payload: Mapping[str, object],
    expected_hash: str,
) -> V1FQCommonCertificationPanelReceipt:
    rebuilt = V1FQCommonCertificationPanelReceipt(**dict(payload))  # type: ignore[arg-type]
    rebuilt.assert_integrity(
        _identity(expected_hash, name="expected F/Q common-panel receipt hash")
    )
    return rebuilt


@dataclass(frozen=True, slots=True)
class V1OrdinalPhysicalAxisBridgeReceipt:
    """Explicit position-preserving bridge from score ordinals to mask bits."""

    route_count: int
    ordered_route_axis_hash: str
    physical_route_axis_hash: str
    current_mask_hash: str
    already_removed_route_index_hash: str
    ordered_endpoint_index_hashes: tuple[str, ...]
    physical_endpoint_index_hashes: tuple[str, ...]
    endpoint_new_removal_counts: tuple[int, ...]
    position_rule: str = "ordered_ordinal_i_maps_to_physical_mask_bit_i"
    full_position_bijection_verified: bool = True
    prior_removed_routes_fixed: bool = True
    endpoint_actions_contain_new_current_mask_deltas_only: bool = True
    mask_commit_authority: bool = False
    schema: str = ORDINAL_PHYSICAL_AXIS_BRIDGE_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        count = _positive_int(self.route_count, name="bridge route count")
        for name in (
            "ordered_route_axis_hash",
            "physical_route_axis_hash",
            "current_mask_hash",
            "already_removed_route_index_hash",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name)
            )
        ordered = tuple(
            _identity(item, name="ordered endpoint index hash")
            for item in self.ordered_endpoint_index_hashes
        )
        physical = tuple(
            _identity(item, name="physical endpoint index hash")
            for item in self.physical_endpoint_index_hashes
        )
        counts = tuple(
            _positive_int(item, name="endpoint new removal count")
            for item in self.endpoint_new_removal_counts
        )
        if (
            not ordered
            or len(ordered) != len(physical)
            or len(ordered) != len(counts)
            or tuple(sorted(counts)) != counts
            or len(set(counts)) != len(counts)
        ):
            raise ValueError("axis bridge endpoint axes are not aligned and increasing")
        if self.position_rule != "ordered_ordinal_i_maps_to_physical_mask_bit_i":
            raise ValueError("unsupported ordinal-to-physical bridge rule")
        if (
            self.full_position_bijection_verified is not True
            or self.prior_removed_routes_fixed is not True
            or self.endpoint_actions_contain_new_current_mask_deltas_only is not True
            or self.mask_commit_authority is not False
        ):
            raise PermissionError("ordinal-to-physical bridge crossed its boundary")
        if self.schema != ORDINAL_PHYSICAL_AXIS_BRIDGE_SCHEMA:
            raise ValueError("unsupported ordinal-to-physical bridge schema")
        object.__setattr__(self, "route_count", count)
        object.__setattr__(self, "ordered_endpoint_index_hashes", ordered)
        object.__setattr__(self, "physical_endpoint_index_hashes", physical)
        object.__setattr__(self, "endpoint_new_removal_counts", counts)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "route_count": self.route_count,
            "ordered_route_axis_hash": self.ordered_route_axis_hash,
            "physical_route_axis_hash": self.physical_route_axis_hash,
            "current_mask_hash": self.current_mask_hash,
            "already_removed_route_index_hash": (
                self.already_removed_route_index_hash
            ),
            "ordered_endpoint_index_hashes": self.ordered_endpoint_index_hashes,
            "physical_endpoint_index_hashes": self.physical_endpoint_index_hashes,
            "endpoint_new_removal_counts": self.endpoint_new_removal_counts,
            "position_rule": self.position_rule,
            "full_position_bijection_verified": self.full_position_bijection_verified,
            "prior_removed_routes_fixed": self.prior_removed_routes_fixed,
            "endpoint_actions_contain_new_current_mask_deltas_only": (
                self.endpoint_actions_contain_new_current_mask_deltas_only
            ),
            "mask_commit_authority": self.mask_commit_authority,
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_AXIS_BRIDGE_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("ordinal-to-physical bridge receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("ordinal-to-physical bridge receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1ResponseExactificationDecisionScopeReceipt:
    """Per-endpoint score-order scope for one provisional packed ladder.

    Exact replay and scientific resolution are deliberately different claims.
    Every registered rung is replayed against the same confidence box, while
    only ``score_resolved_endpoint_indices`` has rung-level action-regret
    authority.  The complement remains a deterministic, authority-inert
    provisional mask family.  F/Q may measure that family, but a later
    candidate decision must still enforce the score-resolution suffix rule.
    """

    score_order_registration_hash: str
    score_order_look_receipt_hash: str
    registered_endpoint_new_removal_counts: tuple[int, ...]
    score_resolved_endpoint_indices: tuple[int, ...]
    provisional_endpoint_indices: tuple[int, ...]
    decision_competitive_endpoint_indices: tuple[int, ...]
    registered_comparator_endpoint_index: int | None
    registered_comparator_rung_new_removal_count: int | None
    score_order_look_index: int = 1
    maximum_registered_score_looks: int = 1
    every_registered_rung_exactly_replayed: bool = True
    every_registered_rung_score_resolved: bool = True
    unresolved_rungs_have_provisional_composition_only: bool = True
    decision_competitive_rungs_score_resolved: bool = True
    selected_endpoint_must_be_score_resolved: bool = True
    larger_provisional_endpoints_block_selection: bool = True
    smaller_provisional_endpoints_block_selection: bool = False
    provisional_ladder_freeze_authority: bool = True
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    schema: str = RESPONSE_EXACTIFICATION_DECISION_SCOPE_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "score_order_registration_hash",
            "score_order_look_receipt_hash",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name)
            )
        counts = tuple(
            _positive_int(value, name="registered endpoint new removal count")
            for value in self.registered_endpoint_new_removal_counts
        )
        if (
            not counts
            or tuple(sorted(counts)) != counts
            or len(set(counts)) != len(counts)
        ):
            raise ValueError(
                "decision scope requires a strictly increasing registered endpoint axis"
            )
        endpoint_count = len(counts)
        look_index = _positive_int(
            self.score_order_look_index, name="score-order look index"
        )
        maximum_looks = _positive_int(
            self.maximum_registered_score_looks,
            name="maximum registered score looks",
        )
        if look_index > maximum_looks:
            raise ValueError("score-order look index exceeds its preregistered axis")
        registered = tuple(range(1, endpoint_count + 1))
        resolved = _endpoint_index_axis(
            self.score_resolved_endpoint_indices,
            endpoint_count=endpoint_count,
            name="score-resolved endpoint indices",
        )
        provisional = _endpoint_index_axis(
            self.provisional_endpoint_indices,
            endpoint_count=endpoint_count,
            name="provisional endpoint indices",
        )
        competitive = _endpoint_index_axis(
            self.decision_competitive_endpoint_indices,
            endpoint_count=endpoint_count,
            name="decision-competitive endpoint indices",
        )
        if (
            tuple(sorted((*resolved, *provisional))) != registered
            or set(resolved).intersection(provisional)
        ):
            raise ValueError(
                "score-resolved and provisional axes must exactly partition the ladder"
            )
        if not set(competitive).issubset(resolved):
            raise PermissionError(
                "a decision-competitive endpoint lacks score-order resolution"
            )
        comparator_endpoint = self.registered_comparator_endpoint_index
        comparator_count = self.registered_comparator_rung_new_removal_count
        if comparator_endpoint is None:
            if comparator_count is not None:
                raise ValueError("comparator endpoint/count scope is incomplete")
        else:
            comparator_endpoint = _positive_int(
                comparator_endpoint, name="registered comparator endpoint index"
            )
            if comparator_endpoint > endpoint_count:
                raise ValueError("registered comparator endpoint left the ladder")
            comparator_count = _positive_int(
                comparator_count, name="registered comparator rung removal count"
            )
            if counts[comparator_endpoint - 1] != comparator_count:
                raise ValueError("registered comparator endpoint/count do not align")
            if (
                comparator_endpoint not in resolved
                or comparator_endpoint not in competitive
            ):
                raise PermissionError(
                    "the preregistered comparator rung must remain score-resolved "
                    "and decision-competitive"
                )
        if self.every_registered_rung_score_resolved is not (not provisional):
            raise ValueError(
                "global score-resolution claim differs from the per-endpoint axes"
            )
        if (
            self.every_registered_rung_exactly_replayed is not True
            or self.unresolved_rungs_have_provisional_composition_only is not True
            or self.decision_competitive_rungs_score_resolved is not True
            or self.selected_endpoint_must_be_score_resolved is not True
            or self.larger_provisional_endpoints_block_selection is not True
            or self.smaller_provisional_endpoints_block_selection is not False
            or self.provisional_ladder_freeze_authority is not True
            or self.fq_certification_authority is not False
            or self.bce_oracle_authority is not False
            or self.candidate_freeze_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
        ):
            raise PermissionError("decision-scoped score receipt crossed its boundary")
        if self.schema != RESPONSE_EXACTIFICATION_DECISION_SCOPE_SCHEMA:
            raise ValueError("unsupported response-exactification decision scope")
        object.__setattr__(
            self, "registered_endpoint_new_removal_counts", counts
        )
        object.__setattr__(self, "score_resolved_endpoint_indices", resolved)
        object.__setattr__(self, "provisional_endpoint_indices", provisional)
        object.__setattr__(
            self, "decision_competitive_endpoint_indices", competitive
        )
        object.__setattr__(
            self, "registered_comparator_endpoint_index", comparator_endpoint
        )
        object.__setattr__(
            self,
            "registered_comparator_rung_new_removal_count",
            comparator_count,
        )
        object.__setattr__(self, "score_order_look_index", look_index)
        object.__setattr__(
            self, "maximum_registered_score_looks", maximum_looks
        )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "score_order_registration_hash": self.score_order_registration_hash,
            "score_order_look_receipt_hash": self.score_order_look_receipt_hash,
            "registered_endpoint_new_removal_counts": (
                self.registered_endpoint_new_removal_counts
            ),
            "score_resolved_endpoint_indices": self.score_resolved_endpoint_indices,
            "provisional_endpoint_indices": self.provisional_endpoint_indices,
            "decision_competitive_endpoint_indices": (
                self.decision_competitive_endpoint_indices
            ),
            "registered_comparator_endpoint_index": (
                self.registered_comparator_endpoint_index
            ),
            "registered_comparator_rung_new_removal_count": (
                self.registered_comparator_rung_new_removal_count
            ),
            "score_order_look_index": self.score_order_look_index,
            "maximum_registered_score_looks": (
                self.maximum_registered_score_looks
            ),
            "every_registered_rung_exactly_replayed": (
                self.every_registered_rung_exactly_replayed
            ),
            "every_registered_rung_score_resolved": (
                self.every_registered_rung_score_resolved
            ),
            "unresolved_rungs_have_provisional_composition_only": (
                self.unresolved_rungs_have_provisional_composition_only
            ),
            "decision_competitive_rungs_score_resolved": (
                self.decision_competitive_rungs_score_resolved
            ),
            "selected_endpoint_must_be_score_resolved": (
                self.selected_endpoint_must_be_score_resolved
            ),
            "larger_provisional_endpoints_block_selection": (
                self.larger_provisional_endpoints_block_selection
            ),
            "smaller_provisional_endpoints_block_selection": (
                self.smaller_provisional_endpoints_block_selection
            ),
            "provisional_ladder_freeze_authority": (
                self.provisional_ladder_freeze_authority
            ),
            "fq_certification_authority": self.fq_certification_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_DECISION_SCOPE_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("response decision-scope receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("response decision-scope receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1ResponseExactificationReplayReceipt:
    """Replay closure for the score box and every provisional frozen prefix."""

    score_order_registration_hash: str
    score_order_look_receipt_hash: str
    proposal_order_design_hash: str
    proposal_order_envelope_certificate_hash: str
    statistics_hash: str
    confidence_box_hash: str
    action_family_hashes: tuple[str, ...]
    proposal_hashes: tuple[str, ...]
    action_regret_request_hashes: tuple[str, ...]
    action_regret_receipt_hashes: tuple[str, ...]
    endpoint_new_removal_counts: tuple[int, ...]
    axis_bridge_receipt_hash: str
    registered_comparator_retained_route_count: int | None
    registered_comparator_rung_new_removal_count: int | None
    registered_comparator_rung_included: bool
    decision_scope_receipt: V1ResponseExactificationDecisionScopeReceipt | None = field(
        default=None, repr=False, compare=False
    )
    every_registered_rung_exactly_replayed: bool = True
    every_registered_rung_replayed_and_resolved: bool = True
    score_order_authority_consumed_only_for_ladder_composition: bool = True
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = RESPONSE_EXACTIFICATION_REPLAY_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )
    # Appended after every v2 slot so historical slotted pickle state keeps
    # its exact positional ABI.  Legacy receipts normalize this to the sole
    # proposal-order design hash; hybrid v3 receipts bind a distinct base.
    proposal_order_base_serfling_design_hash: str | None = None
    nested_prefix_zero_regret_batch_request_hash: str | None = None
    nested_prefix_zero_regret_batch_receipt_hash: str | None = None
    nested_prefix_zero_regret_rung_certificate_hashes: tuple[str, ...] = ()
    proposal_identity_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "score_order_registration_hash",
            "score_order_look_receipt_hash",
            "proposal_order_design_hash",
            "proposal_order_envelope_certificate_hash",
            "statistics_hash",
            "confidence_box_hash",
            "axis_bridge_receipt_hash",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name)
            )
        base_design_hash_was_provided = (
            self.proposal_order_base_serfling_design_hash is not None
        )
        base_design_hash = self.proposal_order_base_serfling_design_hash
        if base_design_hash is None:
            base_design_hash = self.proposal_order_design_hash
        object.__setattr__(
            self,
            "proposal_order_base_serfling_design_hash",
            _identity(
                base_design_hash,
                name="proposal-order base Serfling design hash",
            ),
        )
        families = tuple(
            _identity(value, name="action family hash")
            for value in self.action_family_hashes
        )
        proposals = tuple(
            _identity(value, name="proposal hash") for value in self.proposal_hashes
        )
        requests = tuple(
            _identity(value, name="regret request hash")
            for value in self.action_regret_request_hashes
        )
        receipts = tuple(
            _identity(value, name="regret receipt hash")
            for value in self.action_regret_receipt_hashes
        )
        counts = tuple(
            _positive_int(value, name="endpoint new removal count")
            for value in self.endpoint_new_removal_counts
        )
        batch_request_hash = (
            None
            if self.nested_prefix_zero_regret_batch_request_hash is None
            else _identity(
                self.nested_prefix_zero_regret_batch_request_hash,
                name="nested-prefix zero-regret batch request hash",
            )
        )
        batch_receipt_hash = (
            None
            if self.nested_prefix_zero_regret_batch_receipt_hash is None
            else _identity(
                self.nested_prefix_zero_regret_batch_receipt_hash,
                name="nested-prefix zero-regret batch receipt hash",
            )
        )
        batch_rung_hashes = tuple(
            _identity(value, name="nested-prefix zero-regret rung certificate hash")
            for value in self.nested_prefix_zero_regret_rung_certificate_hashes
        )
        proposal_identity_hashes = tuple(
            _identity(value, name="proposal identity hash")
            for value in self.proposal_identity_hashes
        )
        if not families or len(families) != len(counts):
            raise ValueError("response-exactification replay axes are not aligned")
        batch_fields_present = (
            batch_request_hash is not None
            or batch_receipt_hash is not None
            or bool(batch_rung_hashes)
            or bool(proposal_identity_hashes)
        )
        if batch_fields_present:
            if (
                batch_request_hash is None
                or batch_receipt_hash is None
                or len(batch_rung_hashes) != len(counts)
                or len(proposal_identity_hashes) != len(counts)
                or proposals
                or requests
                or receipts
            ):
                raise ValueError(
                    "nested-prefix zero-regret replay axis is incomplete"
                )
        elif not (
            len(proposals) == len(requests) == len(receipts) == len(counts)
        ):
            raise ValueError("response-exactification replay axes are not aligned")
        comparator = self.registered_comparator_retained_route_count
        comparator_rung = self.registered_comparator_rung_new_removal_count
        if comparator is not None:
            comparator = _positive_int(comparator, name="comparator retained count")
        if comparator_rung is not None:
            comparator_rung = _positive_int(
                comparator_rung, name="comparator rung new removal count"
            )
        if self.registered_comparator_rung_included != (
            comparator_rung is not None and comparator_rung in counts
        ):
            raise ValueError("comparator rung inclusion receipt is inconsistent")
        comparator_endpoint = (
            None
            if comparator_rung is None
            else counts.index(comparator_rung) + 1
        )
        scope = self.decision_scope_receipt
        if scope is None:
            scope = V1ResponseExactificationDecisionScopeReceipt(
                score_order_registration_hash=self.score_order_registration_hash,
                score_order_look_receipt_hash=self.score_order_look_receipt_hash,
                registered_endpoint_new_removal_counts=counts,
                score_resolved_endpoint_indices=tuple(range(1, len(counts) + 1)),
                provisional_endpoint_indices=(),
                decision_competitive_endpoint_indices=tuple(
                    range(1, len(counts) + 1)
                ),
                registered_comparator_endpoint_index=comparator_endpoint,
                registered_comparator_rung_new_removal_count=comparator_rung,
            )
        if type(scope) is not V1ResponseExactificationDecisionScopeReceipt:
            raise TypeError("response replay requires a typed decision-scope receipt")
        scope.assert_integrity()
        if (
            scope.score_order_registration_hash
            != self.score_order_registration_hash
            or scope.score_order_look_receipt_hash
            != self.score_order_look_receipt_hash
            or scope.registered_endpoint_new_removal_counts != counts
            or scope.registered_comparator_endpoint_index != comparator_endpoint
            or scope.registered_comparator_rung_new_removal_count
            != comparator_rung
        ):
            raise ValueError("response replay and decision scope do not close")
        if self.every_registered_rung_exactly_replayed is not True:
            raise PermissionError("response replay omitted a registered rung")
        if self.every_registered_rung_replayed_and_resolved is not (
            scope.every_registered_rung_score_resolved
        ):
            raise ValueError(
                "global replay/resolution claim differs from the per-endpoint scope"
            )
        if (
            self.score_order_authority_consumed_only_for_ladder_composition
            is not True
            or self.fq_certification_authority is not False
            or self.bce_oracle_authority is not False
            or self.candidate_freeze_authority is not False
            or self.mask_commit_authority is not False
        ):
            raise PermissionError("response-exactification replay crossed its boundary")
        if self.schema not in {
            RESPONSE_EXACTIFICATION_REPLAY_SCHEMA,
            RESPONSE_EXACTIFICATION_HYBRID_REPLAY_SCHEMA,
        }:
            raise ValueError("unsupported response-exactification replay schema")
        if (
            self.schema == RESPONSE_EXACTIFICATION_REPLAY_SCHEMA
            and self.proposal_order_base_serfling_design_hash
            != self.proposal_order_design_hash
        ):
            raise ValueError("legacy response replay cannot name two designs")
        if (
            self.schema == RESPONSE_EXACTIFICATION_HYBRID_REPLAY_SCHEMA
            and not base_design_hash_was_provided
        ):
            raise ValueError("hybrid response replay requires its base design hash")
        if (
            self.schema == RESPONSE_EXACTIFICATION_HYBRID_REPLAY_SCHEMA
            and self.proposal_order_base_serfling_design_hash
            == self.proposal_order_design_hash
        ):
            raise ValueError(
                "hybrid response replay requires distinct wrapper and base designs"
            )
        object.__setattr__(self, "action_family_hashes", families)
        object.__setattr__(self, "proposal_hashes", proposals)
        object.__setattr__(self, "action_regret_request_hashes", requests)
        object.__setattr__(self, "action_regret_receipt_hashes", receipts)
        object.__setattr__(self, "proposal_identity_hashes", proposal_identity_hashes)
        object.__setattr__(
            self,
            "nested_prefix_zero_regret_batch_request_hash",
            batch_request_hash,
        )
        object.__setattr__(
            self,
            "nested_prefix_zero_regret_batch_receipt_hash",
            batch_receipt_hash,
        )
        object.__setattr__(
            self,
            "nested_prefix_zero_regret_rung_certificate_hashes",
            batch_rung_hashes,
        )
        object.__setattr__(self, "endpoint_new_removal_counts", counts)
        object.__setattr__(self, "decision_scope_receipt", scope)
        object.__setattr__(
            self, "registered_comparator_retained_route_count", comparator
        )
        object.__setattr__(
            self, "registered_comparator_rung_new_removal_count", comparator_rung
        )
        self.assert_integrity()

    def _assert_normalized_schema_design_pair(self) -> None:
        if self.schema == RESPONSE_EXACTIFICATION_REPLAY_SCHEMA:
            if (
                self.proposal_order_base_serfling_design_hash
                != self.proposal_order_design_hash
            ):
                raise ValueError("legacy response replay cannot name two designs")
            return
        if self.schema == RESPONSE_EXACTIFICATION_HYBRID_REPLAY_SCHEMA:
            if (
                self.proposal_order_base_serfling_design_hash
                == self.proposal_order_design_hash
            ):
                raise ValueError(
                    "hybrid response replay requires distinct wrapper and base designs"
                )
            return
        raise ValueError("unsupported response-exactification replay schema")

    def _payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "score_order_registration_hash": self.score_order_registration_hash,
            "score_order_look_receipt_hash": self.score_order_look_receipt_hash,
            "proposal_order_design_hash": self.proposal_order_design_hash,
            "proposal_order_envelope_certificate_hash": (
                self.proposal_order_envelope_certificate_hash
            ),
            "statistics_hash": self.statistics_hash,
            "confidence_box_hash": self.confidence_box_hash,
            "action_family_hashes": self.action_family_hashes,
            "proposal_hashes": self.proposal_hashes,
            "action_regret_request_hashes": self.action_regret_request_hashes,
            "action_regret_receipt_hashes": self.action_regret_receipt_hashes,
            "endpoint_new_removal_counts": self.endpoint_new_removal_counts,
            "axis_bridge_receipt_hash": self.axis_bridge_receipt_hash,
            "registered_comparator_retained_route_count": (
                self.registered_comparator_retained_route_count
            ),
            "registered_comparator_rung_new_removal_count": (
                self.registered_comparator_rung_new_removal_count
            ),
            "registered_comparator_rung_included": (
                self.registered_comparator_rung_included
            ),
            "decision_scope_receipt_hash": (
                self.decision_scope_receipt.assert_integrity()
            ),
            "every_registered_rung_exactly_replayed": (
                self.every_registered_rung_exactly_replayed
            ),
            "every_registered_rung_replayed_and_resolved": (
                self.every_registered_rung_replayed_and_resolved
            ),
            "score_order_authority_consumed_only_for_ladder_composition": (
                self.score_order_authority_consumed_only_for_ladder_composition
            ),
            "fq_certification_authority": self.fq_certification_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "mask_commit_authority": self.mask_commit_authority,
        }
        if self.schema == RESPONSE_EXACTIFICATION_HYBRID_REPLAY_SCHEMA:
            payload["proposal_order_base_serfling_design_hash"] = (
                self.proposal_order_base_serfling_design_hash
            )
        if self.nested_prefix_zero_regret_batch_receipt_hash is not None:
            payload.update(
                {
                    "nested_prefix_zero_regret_batch_request_hash": (
                        self.nested_prefix_zero_regret_batch_request_hash
                    ),
                    "nested_prefix_zero_regret_batch_receipt_hash": (
                        self.nested_prefix_zero_regret_batch_receipt_hash
                    ),
                    "nested_prefix_zero_regret_rung_certificate_hashes": (
                        self.nested_prefix_zero_regret_rung_certificate_hashes
                    ),
                    "proposal_identity_hashes": (
                        self.proposal_identity_hashes
                    ),
                    "action_regret_identity_kind": (
                        "nested_prefix_batch_zero_regret_certificate_v1"
                    ),
                }
            )
        return payload

    @property
    def every_registered_rung_replayed(self) -> bool:
        """Literal replay claim retained as a public compatibility seam."""

        return self.every_registered_rung_exactly_replayed

    @property
    def every_registered_rung_resolved(self) -> bool:
        """Literal all-rung resolution; false for an honest partial ladder."""

        return self.every_registered_rung_replayed_and_resolved

    @property
    def score_resolved_endpoint_indices(self) -> tuple[int, ...]:
        return self.decision_scope_receipt.score_resolved_endpoint_indices

    @property
    def provisional_endpoint_indices(self) -> tuple[int, ...]:
        return self.decision_scope_receipt.provisional_endpoint_indices

    @property
    def decision_competitive_endpoint_indices(self) -> tuple[int, ...]:
        return self.decision_scope_receipt.decision_competitive_endpoint_indices

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._assert_normalized_schema_design_pair()
        self.decision_scope_receipt.assert_integrity()
        prefix = (
            _HYBRID_REPLAY_PREFIX
            if self.schema == RESPONSE_EXACTIFICATION_HYBRID_REPLAY_SCHEMA
            else _REPLAY_PREFIX
        )
        observed = content_hash(self._payload(), prefix=prefix)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("response-exactification replay receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("response-exactification replay hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @classmethod
    def _pickle_state_field_names(
        cls,
        *,
        schema: str,
        include_batch: bool,
    ) -> tuple[str, ...]:
        names = tuple(item.name for item in fields(cls))
        if schema == RESPONSE_EXACTIFICATION_REPLAY_SCHEMA:
            names = tuple(
                name
                for name in names
                if name != "proposal_order_base_serfling_design_hash"
            )
        elif schema != RESPONSE_EXACTIFICATION_HYBRID_REPLAY_SCHEMA:
            raise ValueError("unsupported response replay pickle schema")
        if not include_batch:
            names = tuple(
                name
                for name in names
                if name
                not in {
                    "nested_prefix_zero_regret_batch_request_hash",
                    "nested_prefix_zero_regret_batch_receipt_hash",
                    "nested_prefix_zero_regret_rung_certificate_hashes",
                    "proposal_identity_hashes",
                }
            )
        return names

    def __getstate__(self) -> list[object]:
        self.assert_integrity()
        field_names = type(self)._pickle_state_field_names(
            schema=self.schema,
            include_batch=(
                self.nested_prefix_zero_regret_batch_receipt_hash is not None
            ),
        )
        return [getattr(self, name) for name in field_names]

    def __setstate__(self, state: Sequence[object]) -> None:
        if type(state) not in (list, tuple):
            raise TypeError("response replay pickle state must be a list or tuple")
        values = list(state)
        state_names: tuple[str, ...] | None = None
        for schema in (
            RESPONSE_EXACTIFICATION_REPLAY_SCHEMA,
            RESPONSE_EXACTIFICATION_HYBRID_REPLAY_SCHEMA,
        ):
            for include_batch in (False, True):
                candidate = type(self)._pickle_state_field_names(
                    schema=schema,
                    include_batch=include_batch,
                )
                if len(values) == len(candidate) and (
                    values[candidate.index("schema")] == schema
                ):
                    state_names = candidate
                    break
            if state_names is not None:
                break
        legacy_v2_names = type(self)._pickle_state_field_names(
            schema=RESPONSE_EXACTIFICATION_REPLAY_SCHEMA,
            include_batch=False,
        )
        if state_names is None:
            if len(values) == len(legacy_v2_names):
                raise ValueError(
                    "a 25-slot response replay pickle must use the legacy v2 schema"
                )
            raise ValueError("response replay pickle state has invalid arity")
        payload = dict(zip(state_names, values, strict=True))
        payload.setdefault(
            "proposal_order_base_serfling_design_hash",
            payload["proposal_order_design_hash"],
        )
        payload.setdefault("nested_prefix_zero_regret_batch_request_hash", None)
        payload.setdefault("nested_prefix_zero_regret_batch_receipt_hash", None)
        payload.setdefault("nested_prefix_zero_regret_rung_certificate_hashes", ())
        payload.setdefault("proposal_identity_hashes", ())
        for name in tuple(item.name for item in fields(type(self))):
            object.__setattr__(self, name, payload[name])
        # Pickle restoration bypasses dataclass construction.  Re-run the
        # complete constructor validation, including any cached receipt hash,
        # before exposing the restored authority object.
        self.__post_init__()


@dataclass(frozen=True, slots=True)
class V1ResponseExactificationFreeze:
    """Packed v1-order ladder frozen before any fresh response rows."""

    ladder: V1FrozenCandidateLadder = field(repr=False, compare=False)
    axis_bridge: V1OrdinalPhysicalAxisBridgeReceipt
    replay_receipt: V1ResponseExactificationReplayReceipt
    ordered_envelope_certificate: V1CorrectedAttributionEnvelopeCertificate = field(
        repr=False, compare=False
    )
    frozen_before_fq_rows: bool = True
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = RESPONSE_EXACTIFICATION_FREEZE_SCHEMA
    _freeze_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def _assert_schema_pair(self) -> None:
        expected_replay_schema = {
            RESPONSE_EXACTIFICATION_FREEZE_SCHEMA: (
                RESPONSE_EXACTIFICATION_REPLAY_SCHEMA
            ),
            RESPONSE_EXACTIFICATION_HYBRID_FREEZE_SCHEMA: (
                RESPONSE_EXACTIFICATION_HYBRID_REPLAY_SCHEMA
            ),
        }.get(self.schema)
        if expected_replay_schema is None:
            raise ValueError("unsupported response-exactification freeze schema")
        if self.replay_receipt.schema != expected_replay_schema:
            raise ValueError(
                "response-exactification replay and freeze schemas do not pair"
            )

    def __post_init__(self) -> None:
        if type(self.ladder) is not V1FrozenCandidateLadder:
            raise TypeError("response exactification requires a typed frozen ladder")
        if type(self.axis_bridge) is not V1OrdinalPhysicalAxisBridgeReceipt:
            raise TypeError("response exactification requires the typed axis bridge")
        if type(self.replay_receipt) is not V1ResponseExactificationReplayReceipt:
            raise TypeError("response exactification requires the typed replay receipt")
        if (
            type(self.ordered_envelope_certificate)
            is not V1CorrectedAttributionEnvelopeCertificate
        ):
            raise TypeError("response exactification requires the ordered envelope")
        self.ladder.assert_integrity()
        self.axis_bridge.assert_integrity()
        self.replay_receipt.assert_integrity()
        self.ordered_envelope_certificate.assert_integrity()
        self._assert_schema_pair()
        replay_hash = self.replay_receipt.receipt_hash
        expected_ladder_freeze_receipt_hash = content_hash(
            {
                "axis_bridge_receipt_hash": self.axis_bridge.receipt_hash,
                "replay_receipt_hash": replay_hash,
                "frozen_before_fq_rows": True,
                "candidate_or_mask_authority": False,
            },
            prefix="partition-free-v1-response-ladder-freeze-receipt-",
        )
        if (
            self.replay_receipt.axis_bridge_receipt_hash
            != self.axis_bridge.receipt_hash
            or self.replay_receipt.proposal_order_envelope_certificate_hash
            != self.ordered_envelope_certificate.certificate_hash
            or self.ladder.discovery_evidence_hash != replay_hash
            or self.ladder.freeze_receipt_hash
            != expected_ladder_freeze_receipt_hash
            or self.ladder.route_axis_hash != self.axis_bridge.physical_route_axis_hash
            or self.ladder.route_count != self.axis_bridge.route_count
            or self.ladder.current_mask_hash != self.axis_bridge.current_mask_hash
            or self.ordered_envelope_certificate.route_axis_hash
            != self.axis_bridge.ordered_route_axis_hash
            or self.ladder.endpoint_removed_counts
            != self.axis_bridge.endpoint_new_removal_counts
            or self.ladder.endpoint_removed_counts
            != self.replay_receipt.endpoint_new_removal_counts
        ):
            raise ValueError("response-exactification freeze coordinates do not close")
        if (
            self.frozen_before_fq_rows is not True
            or self.fq_certification_authority is not False
            or self.bce_oracle_authority is not False
            or self.candidate_freeze_authority is not False
            or self.mask_commit_authority is not False
        ):
            raise PermissionError("response-exactification freeze crossed its boundary")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "ladder_hash": self.ladder.assert_integrity(),
            "axis_bridge_receipt_hash": self.axis_bridge.assert_integrity(),
            "replay_receipt_hash": self.replay_receipt.assert_integrity(),
            "ordered_envelope_certificate_hash": (
                self.ordered_envelope_certificate.assert_integrity()
            ),
            "frozen_before_fq_rows": self.frozen_before_fq_rows,
            "fq_certification_authority": self.fq_certification_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "mask_commit_authority": self.mask_commit_authority,
        }

    @property
    def freeze_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.ladder.assert_integrity()
        self.axis_bridge.assert_integrity()
        self.replay_receipt.assert_integrity()
        self.ordered_envelope_certificate.assert_integrity()
        self._assert_schema_pair()
        prefix = (
            _HYBRID_FREEZE_PREFIX
            if self.schema == RESPONSE_EXACTIFICATION_HYBRID_FREEZE_SCHEMA
            else _FREEZE_PREFIX
        )
        observed = content_hash(self._payload(), prefix=prefix)
        if self._freeze_hash is not None and observed != self._freeze_hash:
            raise ValueError("response-exactification freeze was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("response-exactification freeze hash does not match")
        object.__setattr__(self, "_freeze_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1CancellationAwareEndpointEnvelopeBundle:
    """Sharper action envelope plus its explicit sound route-sum fallback."""

    ladder_hash: str
    action_drive_certificate: V1FrozenLadderEndpointEnvelopeCertificate = field(
        repr=False, compare=False
    )
    route_sum_fallback_certificate: V1FrozenLadderEndpointEnvelopeCertificate = field(
        repr=False, compare=False
    )
    cancellation_before_absolute_value: bool = True
    nonvacuous_route_sum_fallback_present: bool = True
    issued_before_fq_outcomes: bool = True
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = CANCELLATION_ENDPOINT_BUNDLE_SCHEMA
    _bundle_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ladder_hash", _identity(self.ladder_hash, name="ladder hash")
        )
        action = self.action_drive_certificate
        fallback = self.route_sum_fallback_certificate
        if (
            type(action) is not V1FrozenLadderEndpointEnvelopeCertificate
            or type(fallback) is not V1FrozenLadderEndpointEnvelopeCertificate
        ):
            raise TypeError("endpoint envelope bundle requires typed certificates")
        action.assert_integrity()
        fallback.assert_integrity()
        if action.derivation_kind != ACTION_SPECIFIC_ENDPOINT_ENVELOPE_KIND:
            raise ValueError("endpoint bundle requires the cancellation-aware envelope")
        if fallback.derivation_kind != ACTION_ROUTE_SUM_ENDPOINT_ENVELOPE_KIND:
            raise ValueError("endpoint bundle requires an explicit route-sum fallback")
        coordinates = (
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
            "ladder_hash",
            "endpoint_action_hashes",
            "stratum_names",
        )
        if any(getattr(action, name) != getattr(fallback, name) for name in coordinates):
            raise ValueError("action-drive and route-sum endpoint axes differ")
        if action.ladder_hash != self.ladder_hash:
            raise ValueError("endpoint envelope bundle names another ladder")
        action_bounds = np.asarray(
            action.endpoint_absolute_upper_bounds_by_stratum, dtype=np.float64
        )
        fallback_bounds = np.asarray(
            fallback.endpoint_absolute_upper_bounds_by_stratum, dtype=np.float64
        )
        if (
            action_bounds.shape != fallback_bounds.shape
            or action_bounds.size == 0
            or not np.isfinite(action_bounds).all()
            or not np.isfinite(fallback_bounds).all()
            or np.any(action_bounds < 0.0)
            or np.any(fallback_bounds < 0.0)
            or np.any(action_bounds > fallback_bounds)
        ):
            raise ValueError("cancellation-aware envelope is not dominated by fallback")
        if (
            self.cancellation_before_absolute_value is not True
            or self.nonvacuous_route_sum_fallback_present is not True
            or self.issued_before_fq_outcomes is not True
            or self.fq_certification_authority is not False
            or self.bce_oracle_authority is not False
            or self.mask_commit_authority is not False
        ):
            raise PermissionError("endpoint envelope bundle crossed its boundary")
        if self.schema != CANCELLATION_ENDPOINT_BUNDLE_SCHEMA:
            raise ValueError("unsupported endpoint envelope bundle schema")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "ladder_hash": self.ladder_hash,
            "action_drive_certificate_hash": (
                self.action_drive_certificate.assert_integrity()
            ),
            "route_sum_fallback_certificate_hash": (
                self.route_sum_fallback_certificate.assert_integrity()
            ),
            "cancellation_before_absolute_value": (
                self.cancellation_before_absolute_value
            ),
            "nonvacuous_route_sum_fallback_present": (
                self.nonvacuous_route_sum_fallback_present
            ),
            "issued_before_fq_outcomes": self.issued_before_fq_outcomes,
            "fq_certification_authority": self.fq_certification_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
        }

    @property
    def bundle_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.action_drive_certificate.assert_integrity()
        self.route_sum_fallback_certificate.assert_integrity()
        observed = content_hash(self._payload(), prefix=_ENDPOINT_BUNDLE_PREFIX)
        if self._bundle_hash is not None and observed != self._bundle_hash:
            raise ValueError("endpoint envelope bundle was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("endpoint envelope bundle hash does not match")
        object.__setattr__(self, "_bundle_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQCommonCertificationPanelReceipt:
    """Pre-outcome theorem receipt for common-panel F/Q redesign reuse.

    The receipt distinguishes physical measurement rows from derived F/Q
    objects.  Context/draw identities from the sealed certification panel may
    be reprojected for each score-look ladder in the finite preregistered
    redesign family.  Old F/Q statistics, certificates, frozen ladders, and
    alpha allocations remain nontransferable.
    """

    proposal_generation: int
    active_score_order_look_index: int
    maximum_fq_ladder_redesigns: int
    proposal_order_design_hash: str
    score_order_registration_hash: str
    score_order_look_receipt_hash: str
    decision_scope_receipt_hash: str
    fq_alpha_allocation_receipt_hash: str
    fq_redesign_alpha_schedule_hash: str
    fq_redesign_alpha_allocations: tuple[float, ...]
    proposal_order_look_alpha_weights: tuple[float, ...]
    certification_context_population_hash: str
    certification_stream_commitment_hash: str
    certification_panel_policy: str = FQ_COMMON_CERTIFICATION_PANEL_POLICY
    ladder_family_measurability: str = (
        "each_ladder_is_function_of_sealed_score_discovery_prefix_only_v1"
    )
    finite_family_union_bound_axis: str = (
        "preallocated_score_look_redesign_alpha_axis_v1"
    )
    physical_c_rows_reusable_as_common_panel: bool = True
    old_fq_statistics_transferable: bool = False
    old_fq_certificates_transferable: bool = False
    old_fq_ladders_transferable: bool = False
    old_fq_alpha_transferable: bool = False
    fq_numeric_outcomes_enter_ladder_composition: bool = False
    fq_classifications_enter_ladder_composition: bool = False
    fq_selected_endpoint_enters_ladder_composition: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    schema: str = FQ_COMMON_CERTIFICATION_PANEL_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        active_look = _positive_int(
            self.active_score_order_look_index,
            name="active score-order look index",
        )
        maximum_redesigns = _positive_int(
            self.maximum_fq_ladder_redesigns,
            name="maximum F/Q ladder redesigns",
        )
        if active_look > maximum_redesigns:
            raise ValueError("active F/Q common-panel look exceeds its family")
        for name in (
            "proposal_order_design_hash",
            "score_order_registration_hash",
            "score_order_look_receipt_hash",
            "decision_scope_receipt_hash",
            "fq_alpha_allocation_receipt_hash",
            "fq_redesign_alpha_schedule_hash",
            "certification_context_population_hash",
            "certification_stream_commitment_hash",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name)
            )
        redesign_axis = tuple(
            _alpha(value, name="F/Q redesign alpha schedule item", positive=True)
            for value in self.fq_redesign_alpha_allocations
        )
        look_weights = tuple(
            _alpha(value, name="proposal look alpha weight", positive=True)
            for value in self.proposal_order_look_alpha_weights
        )
        if (
            len(redesign_axis) != maximum_redesigns
            or len(look_weights) != maximum_redesigns
        ):
            raise ValueError(
                "common-panel theorem receipt must cover every score-look redesign"
            )
        if (
            self.certification_panel_policy != FQ_COMMON_CERTIFICATION_PANEL_POLICY
            or self.ladder_family_measurability
            != "each_ladder_is_function_of_sealed_score_discovery_prefix_only_v1"
            or self.finite_family_union_bound_axis
            != "preallocated_score_look_redesign_alpha_axis_v1"
            or self.physical_c_rows_reusable_as_common_panel is not True
            or self.old_fq_statistics_transferable is not False
            or self.old_fq_certificates_transferable is not False
            or self.old_fq_ladders_transferable is not False
            or self.old_fq_alpha_transferable is not False
            or self.fq_numeric_outcomes_enter_ladder_composition is not False
            or self.fq_classifications_enter_ladder_composition is not False
            or self.fq_selected_endpoint_enters_ladder_composition is not False
            or self.bce_oracle_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
            or self.schema != FQ_COMMON_CERTIFICATION_PANEL_RECEIPT_SCHEMA
        ):
            raise PermissionError(
                "F/Q common-panel receipt crossed its theorem boundary"
            )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "active_score_order_look_index", active_look)
        object.__setattr__(
            self, "maximum_fq_ladder_redesigns", maximum_redesigns
        )
        object.__setattr__(self, "fq_redesign_alpha_allocations", redesign_axis)
        object.__setattr__(self, "proposal_order_look_alpha_weights", look_weights)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "proposal_generation": self.proposal_generation,
            "active_score_order_look_index": self.active_score_order_look_index,
            "maximum_fq_ladder_redesigns": self.maximum_fq_ladder_redesigns,
            "proposal_order_design_hash": self.proposal_order_design_hash,
            "score_order_registration_hash": self.score_order_registration_hash,
            "score_order_look_receipt_hash": self.score_order_look_receipt_hash,
            "decision_scope_receipt_hash": self.decision_scope_receipt_hash,
            "fq_alpha_allocation_receipt_hash": (
                self.fq_alpha_allocation_receipt_hash
            ),
            "fq_redesign_alpha_schedule_hash": self.fq_redesign_alpha_schedule_hash,
            "fq_redesign_alpha_allocations": self.fq_redesign_alpha_allocations,
            "proposal_order_look_alpha_weights": (
                self.proposal_order_look_alpha_weights
            ),
            "certification_context_population_hash": (
                self.certification_context_population_hash
            ),
            "certification_stream_commitment_hash": (
                self.certification_stream_commitment_hash
            ),
            "certification_panel_policy": self.certification_panel_policy,
            "ladder_family_measurability": self.ladder_family_measurability,
            "finite_family_union_bound_axis": self.finite_family_union_bound_axis,
            "physical_c_rows_reusable_as_common_panel": (
                self.physical_c_rows_reusable_as_common_panel
            ),
            "old_fq_statistics_transferable": self.old_fq_statistics_transferable,
            "old_fq_certificates_transferable": self.old_fq_certificates_transferable,
            "old_fq_ladders_transferable": self.old_fq_ladders_transferable,
            "old_fq_alpha_transferable": self.old_fq_alpha_transferable,
            "fq_numeric_outcomes_enter_ladder_composition": (
                self.fq_numeric_outcomes_enter_ladder_composition
            ),
            "fq_classifications_enter_ladder_composition": (
                self.fq_classifications_enter_ladder_composition
            ),
            "fq_selected_endpoint_enters_ladder_composition": (
                self.fq_selected_endpoint_enters_ladder_composition
            ),
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_FQ_COMMON_PANEL_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("F/Q common-panel receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("F/Q common-panel receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def _constructor_payload(self) -> dict[str, object]:
        return dict(self._payload())

    def __reduce__(self) -> tuple[object, tuple[dict[str, object], str]]:
        return _rebuild_fq_common_certification_panel_receipt, (
            self._constructor_payload(),
            self.assert_integrity(),
        )


@dataclass(frozen=True, slots=True)
class V1ResponseExactificationArtifacts:
    """Complete pre-row ladder, endpoint envelope, alpha, and F/Q design."""

    freeze: V1ResponseExactificationFreeze = field(repr=False, compare=False)
    physical_fq_envelope_certificate: V1CorrectedAttributionEnvelopeCertificate = field(
        repr=False, compare=False
    )
    endpoint_envelope_bundle: V1CancellationAwareEndpointEnvelopeBundle = field(
        repr=False, compare=False
    )
    fq_alpha_allocation: V1FQAlphaFamilyAllocationReceipt
    common_panel_receipt: V1FQCommonCertificationPanelReceipt
    certification_design: V1FrozenLadderCertificationDesign = field(
        repr=False, compare=False
    )
    issued_before_fq_outcomes: bool = True
    fq_execution_authority: bool = False
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    schema: str = RESPONSE_EXACTIFICATION_ARTIFACTS_SCHEMA
    _artifacts_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.freeze) is not V1ResponseExactificationFreeze:
            raise TypeError("response artifacts require the typed freeze")
        if (
            type(self.physical_fq_envelope_certificate)
            is not V1CorrectedAttributionEnvelopeCertificate
        ):
            raise TypeError("response artifacts require the physical F/Q envelope")
        if (
            type(self.endpoint_envelope_bundle)
            is not V1CancellationAwareEndpointEnvelopeBundle
        ):
            raise TypeError("response artifacts require the endpoint envelope bundle")
        if type(self.fq_alpha_allocation) is not V1FQAlphaFamilyAllocationReceipt:
            raise TypeError("response artifacts require the F/Q alpha allocation")
        if (
            type(self.common_panel_receipt)
            is not V1FQCommonCertificationPanelReceipt
        ):
            raise TypeError("response artifacts require the common-panel receipt")
        if type(self.certification_design) is not V1FrozenLadderCertificationDesign:
            raise TypeError("response artifacts require the typed F/Q design")
        self.freeze.assert_integrity()
        envelope_hash = self.physical_fq_envelope_certificate.assert_integrity()
        self.endpoint_envelope_bundle.assert_integrity()
        self.fq_alpha_allocation.assert_integrity()
        self.common_panel_receipt.assert_integrity()
        self.certification_design.assert_integrity()
        decision_scope = self.freeze.replay_receipt.decision_scope_receipt
        if (
            self.certification_design.ladder.ladder_hash
            != self.freeze.ladder.ladder_hash
            or self.certification_design.envelope_certificate_hash != envelope_hash
            or self.certification_design.endpoint_envelope_certificate
            is not self.endpoint_envelope_bundle.action_drive_certificate
            or self.certification_design.trajectory_alpha_ledger_hash
            != self.fq_alpha_allocation.receipt_hash
            or self.common_panel_receipt.fq_alpha_allocation_receipt_hash
            != self.fq_alpha_allocation.receipt_hash
            or self.common_panel_receipt.fq_redesign_alpha_schedule_hash
            != self.fq_alpha_allocation.fq_redesign_alpha_schedule_hash
            or self.common_panel_receipt.fq_redesign_alpha_allocations
            != self.fq_alpha_allocation.fq_redesign_alpha_allocations
            or self.common_panel_receipt.proposal_order_look_alpha_weights
            != self.fq_alpha_allocation.proposal_order_look_alpha_weights
            or self.common_panel_receipt.maximum_fq_ladder_redesigns
            != self.fq_alpha_allocation.maximum_fq_ladder_redesigns
            or self.common_panel_receipt.active_score_order_look_index
            != self.fq_alpha_allocation.score_order_look_index
            or self.common_panel_receipt.proposal_order_design_hash
            != self.freeze.replay_receipt.proposal_order_base_serfling_design_hash
            or self.common_panel_receipt.decision_scope_receipt_hash
            != decision_scope.receipt_hash
            or self.common_panel_receipt.score_order_registration_hash
            != decision_scope.score_order_registration_hash
            or self.common_panel_receipt.score_order_look_receipt_hash
            != decision_scope.score_order_look_receipt_hash
            or self.common_panel_receipt.certification_context_population_hash
            != self.certification_design.certification_context_population_hash
            or self.common_panel_receipt.certification_stream_commitment_hash
            != self.certification_design.certification_stream_commitment_hash
        ):
            raise ValueError("response-exactification artifact graph does not close")
        if (
            self.issued_before_fq_outcomes is not True
            or self.fq_execution_authority is not False
            or self.bce_oracle_authority is not False
            or self.candidate_freeze_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
        ):
            raise PermissionError("response-exactification artifacts crossed authority")
        if self.schema != RESPONSE_EXACTIFICATION_ARTIFACTS_SCHEMA:
            raise ValueError("unsupported response-exactification artifacts schema")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "freeze_hash": self.freeze.assert_integrity(),
            "ladder_hash": self.freeze.ladder.assert_integrity(),
            "axis_bridge_receipt_hash": self.freeze.axis_bridge.assert_integrity(),
            "replay_receipt_hash": self.freeze.replay_receipt.assert_integrity(),
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
            "fq_execution_authority": self.fq_execution_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
        }

    @property
    def artifacts_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.freeze.assert_integrity()
        self.physical_fq_envelope_certificate.assert_integrity()
        self.endpoint_envelope_bundle.assert_integrity()
        self.fq_alpha_allocation.assert_integrity()
        self.certification_design.assert_integrity()
        observed = content_hash(self._payload(), prefix=_ARTIFACTS_PREFIX)
        if self._artifacts_hash is not None and observed != self._artifacts_hash:
            raise ValueError("response-exactification artifacts were tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("response-exactification artifacts hash does not match")
        object.__setattr__(self, "_artifacts_hash", observed)
        return observed


def _coordinates_match_score_order(
    statistics: V1StratifiedCorrectedAttributionMoments,
    certificate: V1CorrectedAttributionEnvelopeCertificate,
    design: V1GroupSequentialSerflingDesign,
) -> None:
    names = (
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
        "stratum_names",
        "population_stratum_counts",
        "sealed_population_order_hashes",
    )
    for name in names:
        values = (
            getattr(statistics, name),
            getattr(certificate, name),
            getattr(design, name),
        )
        if not (values[0] == values[1] == values[2]):
            raise ValueError(f"score-order owners disagree on {name}")


def freeze_v1_response_exactified_ladder(
    *,
    current_mask: HardMask,
    physical_route_axis_hash: str,
    ordered_route_axis_hash: str,
    score_order_registration: Mapping[str, object],
    resolved_score_order_look: Mapping[str, object],
    statistics: V1StratifiedCorrectedAttributionMoments,
    ordered_envelope_certificate: V1CorrectedAttributionEnvelopeCertificate,
    proposal_order_design: (
        V1GroupSequentialSerflingDesign
        | V1GroupSequentialEmpiricalBernsteinSerflingDesign
    ),
    action_families: Sequence[V1CorrectedAttributionActionFamily],
    confidence_box: (
        V1CorrectedAttributionConfidenceBoxV2Receipt | None
    ) = None,
    prior_confidence_box: (
        V1CorrectedAttributionConfidenceBoxV2Receipt | None
    ) = None,
    score_order_prefix_external_state_root: str | os.PathLike[str] | None = None,
) -> V1ResponseExactificationFreeze:
    """Replay a resolved score look and freeze its physical packed ladder."""

    if type(statistics) is not V1StratifiedCorrectedAttributionMoments:
        raise TypeError("response exactification requires typed score moments")
    if (
        type(ordered_envelope_certificate)
        is not V1CorrectedAttributionEnvelopeCertificate
    ):
        raise TypeError("response exactification requires the ordered envelope")
    design_type = type(proposal_order_design)
    if design_type is V1GroupSequentialSerflingDesign:
        base_serfling_design = proposal_order_design
        if confidence_box is not None or prior_confidence_box is not None:
            raise TypeError("legacy response replay cannot consume a v2 box chain")
    elif design_type is V1GroupSequentialEmpiricalBernsteinSerflingDesign:
        base_serfling_design = proposal_order_design.serfling_design
        if type(confidence_box) is not V1CorrectedAttributionConfidenceBoxV2Receipt:
            raise TypeError("hybrid response replay requires its exact v2 confidence box")
        if confidence_box.look_index == 1:
            if prior_confidence_box is not None:
                raise ValueError("the first hybrid response look cannot have a prior box")
        elif type(prior_confidence_box) is not (
            V1CorrectedAttributionConfidenceBoxV2Receipt
        ):
            raise TypeError("a later hybrid response look requires its immediate prior box")
    else:
        raise TypeError("response exactification requires an exact v1 or v2 design")
    families = tuple(action_families)
    if not families or any(
        type(item) is not V1CorrectedAttributionActionFamily for item in families
    ):
        raise TypeError("response exactification requires typed action families")
    statistics_hash = statistics.assert_integrity()
    certificate_hash = ordered_envelope_certificate.assert_integrity()
    design_hash = proposal_order_design.assert_integrity()
    _coordinates_match_score_order(
        statistics, ordered_envelope_certificate, base_serfling_design
    )
    mask_hash = _direct_mask_hash(current_mask, name="response current mask")
    physical_axis = _identity(
        physical_route_axis_hash, name="physical route axis hash"
    )
    ordered_axis = _identity(
        ordered_route_axis_hash, name="ordered route axis hash"
    )
    if (
        current_mask.route_axis_hash != physical_axis
        or current_mask.route_count != base_serfling_design.route_count
        or mask_hash != base_serfling_design.current_mask_hash
        or ordered_axis != base_serfling_design.route_axis_hash
        or ordered_axis == physical_axis
    ):
        raise ValueError("ordinal and physical score/mask coordinates do not align")

    registration_hash = _mapping_hash(
        score_order_registration,
        hash_field="registration_hash",
        prefix=_SCORE_REGISTRATION_PREFIX,
        name="score-order registration",
    )
    look_hash = _mapping_hash(
        resolved_score_order_look,
        hash_field="score_order_look_receipt_hash",
        prefix=_SCORE_LOOK_PREFIX,
        name="resolved score-order look",
    )
    registration = dict(score_order_registration)
    look = dict(resolved_score_order_look)
    base_design_hash = base_serfling_design.assert_integrity()
    hybrid_design = (
        design_type is V1GroupSequentialEmpiricalBernsteinSerflingDesign
    )
    hybrid_graph_valid = (
        registration.get("proposal_order_base_serfling_design_hash")
        == base_design_hash
        and look.get("proposal_order_base_serfling_design_hash")
        == base_design_hash
        and registration.get("proposal_order_confidence_design_kind")
        == "hybrid_empirical_bernstein_serfling_v2"
        and look.get("proposal_order_confidence_design_kind")
        == "hybrid_empirical_bernstein_serfling_v2"
        and look.get("hybrid_serfling_confidence_authority") is True
        and look.get("confidence_box_schema")
        == CORRECTED_ATTRIBUTION_CONFIDENCE_BOX_V2_SCHEMA
    )
    if (
        look.get("registration_hash") != registration_hash
        or registration.get("proposal_order_design_hash") != design_hash
        or look.get("proposal_order_design_hash") != design_hash
        or (hybrid_design and not hybrid_graph_valid)
        or registration.get("proposal_order_envelope_certificate_hash")
        != certificate_hash
        or look.get("proposal_order_envelope_certificate_hash") != certificate_hash
        or look.get("statistics_hash") != statistics_hash
        or look.get("look_index") != statistics.look_index
        or tuple(look.get("stratum_counts", ())) != statistics.stratum_counts
        or look.get("proposal_order_inference_stage") != "response_exactification"
        or look.get("every_registered_rung_evaluated") is not True
        or type(look.get("every_registered_rung_resolved")) is not bool
        or look.get("serfling_confidence_authority") is not True
        or type(look.get("action_regret_stopping_authority")) is not bool
        or look.get("exact_action_order_selector") != NESTED_BOTTOM_K_SELECTOR_ID
        or look.get("score_order_nested_selector_id") != NESTED_BOTTOM_K_SELECTOR_ID
        or registration.get("physical_route_axis_hash") != physical_axis
        or registration.get("ordered_route_axis_hash") != ordered_axis
        or registration.get("candidate_freeze_authority") is not False
        or registration.get("fq_certification_authority") is not False
        or registration.get("bce_oracle_authority") is not False
        or registration.get("mask_commit_authority") is not False
        or look.get("candidate_freeze_authority") is not False
        or look.get("fq_certification_authority") is not False
        or look.get("bce_oracle_authority") is not False
        or look.get("mask_commit_authority") is not False
        or look.get("terminal_authority") is not False
    ):
        raise ValueError("resolved score-order receipt graph is incoherent")
    registered_looks = tuple(base_serfling_design.registered_look_stratum_counts)
    if (
        statistics.look_index > len(registered_looks)
        or registered_looks[statistics.look_index - 1] != statistics.stratum_counts
        or tuple(registration.get("registered_look_stratum_counts", ()))
        != registered_looks
    ):
        raise ValueError("score look left its preregistered sampling/alpha axis")

    new_counts = tuple(int(value) for value in registration["new_removal_counts"])
    registered_family_hashes = tuple(
        _identity(value, name="registered action family hash")
        for value in registration["action_family_hashes"]
    )
    cumulative_action_sizes = tuple(
        _positive_int(value, name="registered cumulative action size")
        for value in registration["cumulative_action_sizes"]
    )
    batch_identity_kind = (
        look.get("score_order_action_regret_identity_kind")
        == "nested_prefix_batch_zero_regret_certificate_v1"
    )
    family_hashes = (
        registered_family_hashes
        if batch_identity_kind
        else tuple(item.assert_integrity() for item in families)
    )
    if (
        family_hashes != registered_family_hashes
        or len(families) != len(new_counts)
        or tuple(item.action_size for item in families)
        != cumulative_action_sizes
        or tuple(sorted(new_counts)) != new_counts
        or len(set(new_counts)) != len(new_counts)
    ):
        raise ValueError("registered action family axis was changed")
    rung_receipts = tuple(dict(item) for item in look["rung_receipts"])
    if len(rung_receipts) != len(families):
        raise ValueError("resolved score look omitted a registered rung")

    removed = np.flatnonzero(~current_mask.to_bool_array()).astype(
        np.dtype("<u8"), copy=False
    )
    live = np.flatnonzero(current_mask.to_bool_array()).astype(
        np.dtype("<u8"), copy=False
    )
    already_removed_hash = content_hash(
        removed,
        prefix="partition-free-v1-alexnet-already-removed-route-axis-",
    )
    if (
        int(registration["already_removed_route_count"]) != int(removed.size)
        or registration["already_removed_route_index_hash"] != already_removed_hash
    ):
        raise ValueError("registered prior-pruned coordinates changed")
    zero_arrays = (
        statistics.signed_mean_by_stratum,
        statistics.signed_m2_by_stratum,
        statistics.absolute_mean_by_stratum,
        statistics.absolute_m2_by_stratum,
        ordered_envelope_certificate.absolute_upper_envelope_by_stratum,
    )
    zero_block = np.zeros(
        (len(statistics.stratum_names), int(removed.size)),
        dtype=np.float64,
    )
    if any(
        not np.array_equal(
            np.asarray(array)[:, removed],
            zero_block,
        )
        for array in zero_arrays
    ):
        raise ValueError("a prior-pruned score coordinate is not an exact zero")

    if hybrid_design:
        assert type(proposal_order_design) is (
            V1GroupSequentialEmpiricalBernsteinSerflingDesign
        )
        assert type(confidence_box) is V1CorrectedAttributionConfidenceBoxV2Receipt
        box = corrected_attribution_confidence_box_v2(
            statistics=statistics,
            envelope_certificate=ordered_envelope_certificate,
            design=proposal_order_design,
            prior_confidence_box=prior_confidence_box,
            expected_statistics_hash=statistics_hash,
            expected_envelope_certificate_hash=certificate_hash,
            expected_design_hash=design_hash,
            expected_prior_confidence_box_hash=(
                None
                if prior_confidence_box is None
                else prior_confidence_box.receipt_hash
            ),
        )
        if confidence_box.assert_integrity() != box.receipt_hash:
            raise ValueError("external hybrid confidence box failed theorem replay")
    else:
        assert type(proposal_order_design) is V1GroupSequentialSerflingDesign
        box = corrected_attribution_confidence_box(
            statistics=statistics,
            envelope_certificate=ordered_envelope_certificate,
            design=proposal_order_design,
            expected_statistics_hash=statistics_hash,
            expected_envelope_certificate_hash=certificate_hash,
            expected_design_hash=design_hash,
        )
    if look.get("confidence_box_hash") != box.receipt_hash:
        raise ValueError("resolved score look names another confidence box")
    if any(
        not np.array_equal(
            np.asarray(array)[removed],
            np.zeros(int(removed.size), dtype=np.float64),
        )
        for array in (
            box.class_balanced_absolute_point,
            box.class_balanced_absolute_lower,
            box.class_balanced_absolute_upper,
        )
    ):
        raise ValueError("confidence-box replay changed a pruned coordinate")

    actions: list[V1FrozenLadderAction] = []
    proposal_hashes: list[str] = []
    proposal_identity_hashes: list[str] = []
    request_hashes: list[str] = []
    regret_hashes: list[str] = []
    score_resolved_indices: list[int] = []
    ordered_index_hashes: list[str] = []
    physical_index_hashes: list[str] = []
    selector_receipt = validate_nested_bottom_k_receipt_payload(
        look.get("score_order_nested_selector_receipt"),
        expected_hash=look.get("score_order_nested_selector_receipt_hash"),
    )
    if (
        selector_receipt.get("request_hash") is None
        or selector_receipt.get("confidence_box_hash") != box.receipt_hash
        or selector_receipt.get("route_axis_hash") != physical_axis
        or tuple(selector_receipt.get("counts_in_request_order", ())) != new_counts
        or tuple(selector_receipt.get("action_family_hashes", ())) != family_hashes
        or int(selector_receipt.get("live_count", -1)) != int(live.size)
        or int(selector_receipt.get("kmax", -1)) != max(new_counts)
        or int(selector_receipt.get("already_removed_count", -1)) != int(removed.size)
        or tuple(selector_receipt.get("cumulative_action_sizes", ()))
        != tuple(family.action_size for family in families)
        or selector_receipt.get("action_family_registry_hash")
        != registration.get("proposal_order_action_family_registry_hash")
        or selector_receipt.get("ordered_route_axis_hash") != ordered_axis
        or look.get("score_order_ordered_kmax_prefix_hash")
        != selector_receipt.get("ordered_prefix_index_hash")
        or look.get("score_order_ordered_witness_prefix_hash")
        != selector_receipt.get("ordered_witness_prefix_index_hash")
    ):
        raise ValueError("nested score-order selector receipt is not bound to the look")
    if score_order_prefix_external_state_root is None:
        raise ValueError("response exactification requires the score-order prefix store")
    prefix_descriptor = validate_nested_bottom_k_ordered_prefix_descriptor(
        look.get("score_order_ordered_prefix_external_descriptor"),
        expected_hash=look.get("score_order_ordered_prefix_external_descriptor_hash"),
    )
    if (
        prefix_descriptor.get("request_hash") != selector_receipt["request_hash"]
        or prefix_descriptor.get("selector_receipt_hash")
        != selector_receipt["receipt_hash"]
        or prefix_descriptor.get("ordered_prefix_index_hash")
        != selector_receipt["ordered_prefix_index_hash"]
        or int(prefix_descriptor.get("route_count", -1))
        != int(selector_receipt["route_count"])
        or int(prefix_descriptor.get("kmax", -1)) != int(selector_receipt["kmax"])
    ):
        raise ValueError("ordered prefix external descriptor is not bound to the look")
    selector_rung_hashes = tuple(selector_receipt["rung_hashes"])
    selector_action_hashes = tuple(selector_receipt["physical_action_index_hashes"])
    selector_witness_hashes = tuple(selector_receipt["boundary_witness_hashes"])
    with open_nested_bottom_k_ordered_prefix(
        score_order_prefix_external_state_root,
        look.get("score_order_ordered_prefix_external_descriptor"),
        expected_hash=look.get("score_order_ordered_prefix_external_descriptor_hash"),
    ) as prefix_view:
        ordered_prefix = np.asarray(
            prefix_view.arrays[NESTED_BOTTOM_K_ORDERED_PREFIX_LOGICAL_NAME]
        )
        if (
            ordered_prefix.dtype != np.dtype("<u8")
            or ordered_prefix.ndim != 1
            or int(ordered_prefix.size) != max(new_counts)
            or nested_bottom_k_ordered_prefix_hash(ordered_prefix)
            != selector_receipt["ordered_prefix_index_hash"]
        ):
            raise ValueError("opened ordered prefix does not match selector receipt")
        batch_regret = None
        batch_regret_payload = None
        batch_rung_certificates = ()
        batch_rung_payloads: tuple[dict[str, object], ...] = ()
        raw_batch_payload = look.get(
            "score_order_nested_prefix_zero_regret_receipt"
        )
        if raw_batch_payload is not None:
            batch_regret_payload = (
                validate_nested_prefix_zero_regret_receipt_payload(
                    raw_batch_payload,
                    expected_hash=look.get(
                        "score_order_nested_prefix_zero_regret_receipt_hash"
                    ),
                )
            )
            batch_regret_source_hash = content_hash(
                {
                    "confidence_box_hash": box.receipt_hash,
                    "selector_request_hash": selector_receipt["request_hash"],
                    "selector_receipt_hash": selector_receipt["receipt_hash"],
                    "selector_result_hash": look.get(
                        "score_order_nested_selector_result_hash"
                    ),
                    "selector_ordered_prefix_hash": (
                        selector_receipt["ordered_prefix_index_hash"]
                    ),
                    "action_family_registry_hash": registration[
                        "proposal_order_action_family_registry_hash"
                    ],
                    "already_removed_route_index_hash": already_removed_hash,
                    "live_physical_indices_hash": (
                        selector_receipt["live_physical_indices_hash"]
                    ),
                    "action_family_hashes": family_hashes,
                    "new_removal_counts": new_counts,
                    "cumulative_action_sizes": tuple(
                        int(value)
                        for value in registration["cumulative_action_sizes"]
                    ),
                    "registered_maximum_action_regret": 0.0,
                    "composition_rule": (
                        "batched_nested_prefix_zero_regret_over_selector_order"
                    ),
                },
                prefix="partition-free-v1-alexnet-score-order-batch-regret-source-",
            )
            batch_rung_certificates = tuple(
                dict(item)
                for item in batch_regret_payload.get("rung_certificates", ())
            )
            batch_request = V1NestedPrefixZeroRegretBatchRequest(
                confidence_box=box,
                ordered_prefix_physical_indices=ordered_prefix,
                live_physical_indices=live,
                new_removal_counts=new_counts,
                already_removed_physical_indices=removed,
                prevalidated_action_family_hashes=family_hashes,
                selector_receipt_hash=selector_receipt["receipt_hash"],
                selector_result_hash=look.get(
                    "score_order_nested_selector_result_hash"
                ),
                selector_ordered_prefix_hash=(
                    selector_receipt["ordered_prefix_index_hash"]
                ),
                batch_source_hash=batch_regret_source_hash,
                action_family_registry_hash=registration[
                    "proposal_order_action_family_registry_hash"
                ],
                cumulative_action_sizes=cumulative_action_sizes,
                registered_maximum_action_regret=0.0,
            )
            batch_regret = certify_nested_prefix_zero_regret(batch_request)
            batch_rung_payloads = batch_rung_certificates
            batch_rung_certificates = batch_regret.rung_certificates
            if (
                look.get("score_order_action_regret_identity_kind")
                != "nested_prefix_batch_zero_regret_certificate_v1"
                or look.get(
                    "score_order_nested_prefix_zero_regret_batch_source_hash"
                )
                != batch_regret_source_hash
                or look.get("score_order_nested_prefix_zero_regret_request_hash")
                != batch_regret.request_hash
                or batch_regret_payload["request_hash"] != batch_regret.request_hash
                or look.get("score_order_nested_prefix_zero_regret_receipt_hash")
                != batch_regret.receipt_hash
                or batch_regret_payload["receipt_hash"] != batch_regret.receipt_hash
                or dict(batch_regret_payload) != batch_regret.to_json_dict()
                or batch_regret_payload["confidence_box_hash"] != box.receipt_hash
                or batch_regret_payload["route_axis_hash"] != box.route_axis_hash
                or tuple(batch_regret_payload["new_removal_counts"]) != new_counts
                or tuple(batch_regret_payload["action_family_hashes"])
                != family_hashes
                or batch_regret_payload["action_family_registry_hash"]
                != registration["proposal_order_action_family_registry_hash"]
                or tuple(batch_regret_payload["cumulative_action_sizes"])
                != cumulative_action_sizes
                or batch_regret_payload["selector_receipt_hash"]
                != selector_receipt["receipt_hash"]
                or batch_regret_payload["selector_result_hash"]
                != look.get("score_order_nested_selector_result_hash")
                or batch_regret_payload["selector_ordered_prefix_hash"]
                != selector_receipt["ordered_prefix_index_hash"]
                or batch_regret_payload["batch_source_hash"]
                != batch_regret_source_hash
                or batch_regret_payload["registered_maximum_action_regret"] != 0.0
                or batch_regret_payload["ordered_prefix_hash"]
                != selector_receipt["ordered_prefix_index_hash"]
                or batch_regret_payload["live_hash"]
                != nested_prefix_zero_regret_live_hash(live)
                or batch_regret_payload["already_removed_hash"]
                != nested_prefix_zero_regret_already_removed_hash(removed)
                or len(batch_rung_payloads) != len(new_counts)
                or look.get(
                    "score_order_nested_prefix_zero_regret_every_rung_resolved"
                )
                is not batch_regret.every_rung_resolved
                or tuple(
                    look.get(
                        "score_order_nested_prefix_zero_regret_resolved_new_removal_counts",
                        (),
                    )
                )
                != batch_regret.resolved_new_removal_counts
            ):
                raise ValueError("nested-prefix zero-regret receipt failed replay")
        for endpoint, (new_count, family, registered_rung) in enumerate(
            zip(new_counts, families, rung_receipts, strict=True),
            start=1,
        ):
            ordered_rung_prefix = ordered_prefix[:new_count]
            ordered_selector_hash = nested_bottom_k_ordered_prefix_hash(
                ordered_rung_prefix
            )
            new_routes = np.array(
                ordered_rung_prefix,
                dtype=np.dtype("<u8"),
                copy=True,
            )
            new_routes.sort(kind="stable")
            new_routes.setflags(write=False)
            physical_selector_hash = nested_bottom_k_physical_action_hash(new_routes)
            selector_index = endpoint - 1
            if (
                ordered_selector_hash
                != registered_rung.get("nested_score_order_ordered_prefix_hash")
                or physical_selector_hash
                != registered_rung.get("nested_score_order_physical_action_hash")
                or selector_rung_hashes[selector_index]
                != registered_rung.get("nested_score_order_rung_hash")
                or selector_action_hashes[selector_index] != physical_selector_hash
                or selector_witness_hashes[selector_index]
                != registered_rung.get("nested_score_order_boundary_witness_hash")
            ):
                raise ValueError("registered nested score rung failed selector replay")
            ordered_index_hash = content_hash(
                ordered_rung_prefix,
                prefix="partition-free-v1-response-ordered-prefix-axis-",
            )
            physical_index_hash = content_hash(
                new_routes,
                prefix="partition-free-v1-response-physical-delta-axis-",
            )
            if batch_regret is not None:
                regret_rung = batch_rung_certificates[selector_index]
                stored_regret_rung = batch_rung_payloads[selector_index]
                family_hash = family_hashes[selector_index]
                cumulative_size = cumulative_action_sizes[selector_index]
                if (
                    stored_regret_rung != regret_rung.to_json_dict()
                    or regret_rung.new_removal_count != new_count
                    or regret_rung.cumulative_action_size != cumulative_size
                    or regret_rung.action_family_hash != family_hash
                    or regret_rung.ordered_prefix_hash != ordered_selector_hash
                    or regret_rung.proposal_identity_hash
                    != registered_rung.get("proposal_identity_hash")
                    or regret_rung.physical_delta_hash
                    != registered_rung.get("physical_delta_hash")
                    or regret_rung.action_regret_resolved
                    is not registered_rung.get("action_regret_resolved")
                    or regret_rung.certified_maximum_action_regret
                    != registered_rung.get("certified_maximum_action_regret")
                ):
                    raise ValueError(
                        "nested-prefix zero-regret rung failed selector replay"
                    )
                proposal_identity_hash = regret_rung.proposal_identity_hash
                certified_regret = regret_rung.certified_maximum_action_regret
                action_regret_resolved = regret_rung.action_regret_resolved
                expected_registered = {
                    "new_removal_count": new_count,
                    "cumulative_action_size": cumulative_size,
                    "action_family_hash": family_hash,
                    "proposal_identity_hash": proposal_identity_hash,
                    "physical_delta_hash": regret_rung.physical_delta_hash,
                    "action_regret_certificate_kind": (
                        "nested_prefix_zero_regret_rung_certificate_v1"
                    ),
                    "certified_maximum_action_regret": certified_regret,
                    "action_regret_resolved": action_regret_resolved,
                    "nested_prefix_zero_regret_batch_request_hash": (
                        batch_regret.request_hash
                    ),
                    "nested_prefix_zero_regret_batch_receipt_hash": (
                        batch_regret.receipt_hash
                    ),
                    "nested_prefix_zero_regret_rung_certificate_hash": (
                        regret_rung.certificate_hash
                    ),
                    "nested_prefix_zero_regret_proposal_identity_hash": (
                        proposal_identity_hash
                    ),
                    "nested_prefix_zero_regret_physical_delta_hash": (
                        regret_rung.physical_delta_hash
                    ),
                    "nested_prefix_zero_regret_ordered_prefix_hash": (
                        regret_rung.ordered_prefix_hash
                    ),
                    "nested_score_order_rung_hash": (
                        selector_rung_hashes[selector_index]
                    ),
                    "nested_score_order_ordered_prefix_hash": ordered_selector_hash,
                    "nested_score_order_physical_action_hash": physical_selector_hash,
                    "nested_score_order_boundary_witness_hash": (
                        selector_witness_hashes[selector_index]
                    ),
                }
            else:
                proposal = np.concatenate((removed, new_routes)).astype(
                    np.dtype("<u8"), copy=False
                )
                proposal.sort(kind="stable")
                proposal_axis_hash = content_hash(
                    proposal, prefix=_ORDERED_ACTION_AXIS_PREFIX
                )
                proposal_source_hash = content_hash(
                    {
                        "confidence_box_hash": box.receipt_hash,
                        "action_family_hash": family.family_hash,
                        "already_removed_route_index_hash": already_removed_hash,
                        "new_routes": new_routes,
                        "composition_rule": (
                            "cumulative_removed_union_deterministic_live_bottom_k"
                        ),
                    },
                    prefix=_SCORE_PROPOSAL_SOURCE_PREFIX,
                )
                regret_request = V1CorrectedAttributionActionRegretRequest(
                    confidence_box=box,
                    action_family=family,
                    proposed_route_indices=proposal,
                    proposal_source_hash=proposal_source_hash,
                )
                regret = certify_corrected_attribution_action_regret(regret_request)
                proposal_hash = regret.proposal_hash
                regret_request_hash = regret.request_hash
                regret_receipt_hash = regret.receipt_hash
                certified_regret = regret.certified_maximum_action_regret
                action_regret_resolved = regret.action_regret_resolved
                expected_registered = {
                    "new_removal_count": new_count,
                    "cumulative_action_size": family.action_size,
                    "action_family_hash": family.family_hash,
                    "proposal_hash": proposal_hash,
                    "proposal_route_index_hash": proposal_axis_hash,
                    "action_regret_request_hash": regret_request_hash,
                    "action_regret_receipt_hash": regret_receipt_hash,
                    "certified_maximum_action_regret": certified_regret,
                    "action_regret_resolved": action_regret_resolved,
                    "nested_score_order_rung_hash": (
                        selector_rung_hashes[selector_index]
                    ),
                    "nested_score_order_ordered_prefix_hash": ordered_selector_hash,
                    "nested_score_order_physical_action_hash": physical_selector_hash,
                    "nested_score_order_boundary_witness_hash": (
                        selector_witness_hashes[selector_index]
                    ),
                }
                if registered_rung.get("action_regret_certificate_kind") is not None:
                    expected_registered["action_regret_certificate_kind"] = (
                        "legacy_single_action_regret_v1"
                    )
            if registered_rung != expected_registered:
                raise ValueError("registered score rung failed exact response replay")
            if action_regret_resolved:
                score_resolved_indices.append(endpoint)
            packed = V1PackedRemovalAction.from_removed_indices(
                parent_mask=current_mask,
                removed_indices=new_routes,
            )
            if (
                packed.removed_count != new_count
                or np.any(packed.resulting_mask.to_bool_array()[removed])
            ):
                raise RuntimeError("packed response action changed prior-pruned routes")
            actions.append(
                V1FrozenLadderAction(
                    endpoint_index=endpoint,
                    action_id=content_hash(
                        {
                            "registration_hash": registration_hash,
                            "look_hash": look_hash,
                            "endpoint_index": endpoint,
                            "new_removal_count": new_count,
                            "ordered_index_hash": ordered_index_hash,
                            "physical_index_hash": physical_index_hash,
                        },
                        prefix="partition-free-v1-response-ladder-action-id-",
                    ),
                    packed_removal_action=packed,
                )
            )
            if batch_regret is not None:
                proposal_identity_hashes.append(proposal_identity_hash)
            else:
                proposal_hashes.append(proposal_hash)
                request_hashes.append(regret_request_hash)
                regret_hashes.append(regret_receipt_hash)
            ordered_index_hashes.append(ordered_index_hash)
            physical_index_hashes.append(physical_index_hash)

    comparator_raw = registration.get(
        "registered_comparator_retained_route_count"
    )
    comparator = None if comparator_raw is None else int(comparator_raw)
    comparator_rung_raw = registration.get(
        "registered_comparator_rung_new_removal_count"
    )
    comparator_rung = (
        None if comparator_rung_raw is None else int(comparator_rung_raw)
    )
    comparator_included = registration.get(
        "registered_comparator_rung_included"
    ) is True
    comparator_endpoint: int | None = None
    if comparator is not None:
        expected_crossing = (
            current_mask.retained_count > comparator
            and current_mask.retained_count - max(new_counts) <= comparator
        )
        expected_rung = current_mask.retained_count - comparator if expected_crossing else None
        if (
            comparator <= 0
            or comparator > current_mask.route_count
            or comparator_rung != expected_rung
            or comparator_included != expected_crossing
            or (expected_crossing and comparator_rung not in new_counts)
            or registration.get("registered_comparator_scalar_only") is not True
            or registration.get(
                "registered_comparator_scalar_checkpoint_influence"
            )
            != "rung_cardinality_only"
            or registration.get(
                "historical_comparator_mask_order_or_trajectory_supplied"
            )
            is not False
        ):
            raise ValueError("registered comparator rung is not a scalar v1 prefix")
        if expected_crossing:
            index = new_counts.index(comparator_rung)
            comparator_endpoint = index + 1
            if actions[index].resulting_mask.retained_count != comparator:
                raise RuntimeError("comparator rung did not land on its exact density")
    elif (
        comparator_rung is not None
        or comparator_included
        or registration.get("registered_comparator_scalar_only") is not False
        or registration.get("registered_comparator_scalar_checkpoint_influence")
        is not None
        or registration.get(
            "historical_comparator_mask_order_or_trajectory_supplied"
        )
        is not False
    ):
        raise ValueError("unregistered comparator rung appeared after outcomes")

    registered_indices = tuple(range(1, len(new_counts) + 1))
    resolved_indices = tuple(score_resolved_indices)
    provisional_indices = tuple(
        index for index in registered_indices if index not in set(resolved_indices)
    )
    explicit_scope_fields = (
        "score_resolved_endpoint_indices",
        "score_resolved_new_removal_counts",
        "provisional_endpoint_indices",
        "provisional_new_removal_counts",
        "decision_competitive_endpoint_indices",
        "decision_competitive_new_removal_counts",
    )
    scope_presence = tuple(field_name in look for field_name in explicit_scope_fields)
    if any(scope_presence) and not all(scope_presence):
        raise ValueError("score look has an incomplete decision-scope axis")
    explicit_scope = all(scope_presence)
    if explicit_scope:
        declared_resolved = _endpoint_index_axis(
            look["score_resolved_endpoint_indices"],
            endpoint_count=len(new_counts),
            name="declared score-resolved endpoint indices",
        )
        declared_provisional = _endpoint_index_axis(
            look["provisional_endpoint_indices"],
            endpoint_count=len(new_counts),
            name="declared provisional endpoint indices",
        )
        competitive_indices = _endpoint_index_axis(
            look["decision_competitive_endpoint_indices"],
            endpoint_count=len(new_counts),
            name="declared decision-competitive endpoint indices",
        )
        declared_resolved_counts = tuple(
            _positive_int(value, name="declared score-resolved removal count")
            for value in look["score_resolved_new_removal_counts"]
        )
        declared_provisional_counts = tuple(
            _positive_int(value, name="declared provisional removal count")
            for value in look["provisional_new_removal_counts"]
        )
        declared_competitive_counts = tuple(
            _positive_int(value, name="declared competitive removal count")
            for value in look["decision_competitive_new_removal_counts"]
        )
    else:
        if provisional_indices:
            raise ValueError(
                "a partial score look lacks explicit per-endpoint decision scope"
            )
        declared_resolved = resolved_indices
        declared_provisional = provisional_indices
        competitive_indices = registered_indices
        declared_resolved_counts = tuple(
            new_counts[index - 1] for index in declared_resolved
        )
        declared_provisional_counts = tuple(
            new_counts[index - 1] for index in declared_provisional
        )
        declared_competitive_counts = tuple(
            new_counts[index - 1] for index in competitive_indices
        )
    all_resolved = not provisional_indices
    expected_resolved_counts = tuple(
        new_counts[index - 1] for index in resolved_indices
    )
    expected_provisional_counts = tuple(
        new_counts[index - 1] for index in provisional_indices
    )
    expected_competitive_counts = tuple(
        new_counts[index - 1] for index in competitive_indices
    )
    expected_comparator_required = comparator_endpoint is not None
    expected_comparator_resolved = bool(
        not expected_comparator_required
        or comparator_endpoint in resolved_indices
    )
    expected_decision_scope_authority = bool(
        resolved_indices and expected_comparator_resolved
    )
    if (
        declared_resolved != resolved_indices
        or declared_provisional != provisional_indices
        or declared_resolved_counts != expected_resolved_counts
        or declared_provisional_counts != expected_provisional_counts
        or declared_competitive_counts != expected_competitive_counts
        or look.get("every_registered_rung_resolved") is not all_resolved
        or look.get("action_regret_stopping_authority") is not all_resolved
        or (
            explicit_scope
            and (
                look.get("registered_comparator_endpoint_index")
                != comparator_endpoint
                or look.get("registered_comparator_resolution_required")
                is not expected_comparator_required
                or look.get("registered_comparator_resolved")
                is not expected_comparator_resolved
                or look.get("decision_scoped_action_regret_authority")
                is not expected_decision_scope_authority
                or look.get("decision_scope_rule")
                != (
                    "selected_zero_regret_endpoint_and_every_larger_endpoint_"
                    "certified_infeasible_smaller_unresolved_nonblocking"
                )
                or competitive_indices
                != (
                    resolved_indices if expected_decision_scope_authority else ()
                )
            )
        )
    ):
        raise ValueError("score look overstates its decision-scoped resolution")

    decision_scope = V1ResponseExactificationDecisionScopeReceipt(
        score_order_registration_hash=registration_hash,
        score_order_look_receipt_hash=look_hash,
        registered_endpoint_new_removal_counts=new_counts,
        score_resolved_endpoint_indices=resolved_indices,
        provisional_endpoint_indices=provisional_indices,
        decision_competitive_endpoint_indices=competitive_indices,
        registered_comparator_endpoint_index=comparator_endpoint,
        registered_comparator_rung_new_removal_count=comparator_rung,
        score_order_look_index=statistics.look_index,
        maximum_registered_score_looks=len(base_serfling_design.alpha_by_look),
        every_registered_rung_score_resolved=all_resolved,
    )

    bridge = V1OrdinalPhysicalAxisBridgeReceipt(
        route_count=current_mask.route_count,
        ordered_route_axis_hash=ordered_axis,
        physical_route_axis_hash=physical_axis,
        current_mask_hash=mask_hash,
        already_removed_route_index_hash=already_removed_hash,
        ordered_endpoint_index_hashes=tuple(ordered_index_hashes),
        physical_endpoint_index_hashes=tuple(physical_index_hashes),
        endpoint_new_removal_counts=new_counts,
    )
    replay = V1ResponseExactificationReplayReceipt(
        score_order_registration_hash=registration_hash,
        score_order_look_receipt_hash=look_hash,
        proposal_order_design_hash=design_hash,
        proposal_order_base_serfling_design_hash=base_design_hash,
        proposal_order_envelope_certificate_hash=certificate_hash,
        statistics_hash=statistics_hash,
        confidence_box_hash=box.receipt_hash,
        action_family_hashes=family_hashes,
        proposal_hashes=tuple(proposal_hashes),
        action_regret_request_hashes=tuple(request_hashes),
        action_regret_receipt_hashes=tuple(regret_hashes),
        nested_prefix_zero_regret_batch_request_hash=(
            None if batch_regret is None else batch_regret.request_hash
        ),
        nested_prefix_zero_regret_batch_receipt_hash=(
            None if batch_regret is None else batch_regret.receipt_hash
        ),
        nested_prefix_zero_regret_rung_certificate_hashes=(
            ()
            if batch_regret is None
            else tuple(
                rung.certificate_hash for rung in batch_rung_certificates
            )
        ),
        proposal_identity_hashes=tuple(proposal_identity_hashes),
        endpoint_new_removal_counts=new_counts,
        axis_bridge_receipt_hash=bridge.receipt_hash,
        registered_comparator_retained_route_count=comparator,
        registered_comparator_rung_new_removal_count=comparator_rung,
        registered_comparator_rung_included=comparator_included,
        decision_scope_receipt=decision_scope,
        every_registered_rung_replayed_and_resolved=all_resolved,
        schema=(
            RESPONSE_EXACTIFICATION_HYBRID_REPLAY_SCHEMA
            if hybrid_design
            else RESPONSE_EXACTIFICATION_REPLAY_SCHEMA
        ),
    )
    discovery_contexts = tuple(
        context
        for axis in box.observed_context_ids_by_stratum
        for context in axis
    )
    discovery_draws = tuple(
        draw for axis in box.observed_draw_ids_by_stratum for draw in axis
    )
    freeze_receipt_hash = content_hash(
        {
            "axis_bridge_receipt_hash": bridge.receipt_hash,
            "replay_receipt_hash": replay.receipt_hash,
            "frozen_before_fq_rows": True,
            "candidate_or_mask_authority": False,
        },
        prefix="partition-free-v1-response-ladder-freeze-receipt-",
    )
    ladder = V1FrozenCandidateLadder(
        campaign_contract_hash=base_serfling_design.campaign_contract_hash,
        model_identity_hash=base_serfling_design.model_identity_hash,
        query_contract_hash=base_serfling_design.query_contract_hash,
        preprocessing_hash=base_serfling_design.preprocessing_hash,
        current_mask_hash=mask_hash,
        raw_logit_stream_source_hash=(
            base_serfling_design.raw_logit_stream_source_hash
        ),
        sampling_law_hash=base_serfling_design.sampling_law_hash,
        proposal_generation=base_serfling_design.proposal_generation,
        route_count=current_mask.route_count,
        route_axis_hash=physical_axis,
        actions=tuple(actions),
        discovery_context_ids=discovery_contexts,
        discovery_draw_ids=discovery_draws,
        discovery_evidence_hash=replay.receipt_hash,
        freeze_receipt_hash=freeze_receipt_hash,
        parent_mask=current_mask,
    )
    return V1ResponseExactificationFreeze(
        ladder=ladder,
        axis_bridge=bridge,
        replay_receipt=replay,
        ordered_envelope_certificate=ordered_envelope_certificate,
        schema=(
            RESPONSE_EXACTIFICATION_HYBRID_FREEZE_SCHEMA
            if hybrid_design
            else RESPONSE_EXACTIFICATION_FREEZE_SCHEMA
        ),
    )


def bind_v1_cancellation_aware_endpoint_envelopes(
    *,
    ladder: V1FrozenCandidateLadder,
    action_drive_certificate: V1FrozenLadderEndpointEnvelopeCertificate,
    route_sum_fallback_certificate: V1FrozenLadderEndpointEnvelopeCertificate,
) -> V1CancellationAwareEndpointEnvelopeBundle:
    """Bind a sharper endpoint envelope to its explicit sound fallback."""

    if type(ladder) is not V1FrozenCandidateLadder:
        raise TypeError("endpoint envelope binding requires a typed ladder")
    ladder_hash = ladder.assert_integrity()
    if (
        action_drive_certificate.ladder_hash != ladder_hash
        or route_sum_fallback_certificate.ladder_hash != ladder_hash
        or action_drive_certificate.endpoint_action_hashes != ladder.action_hashes
        or route_sum_fallback_certificate.endpoint_action_hashes != ladder.action_hashes
    ):
        raise ValueError("endpoint envelopes differ from the frozen ladder")
    return V1CancellationAwareEndpointEnvelopeBundle(
        ladder_hash=ladder_hash,
        action_drive_certificate=action_drive_certificate,
        route_sum_fallback_certificate=route_sum_fallback_certificate,
    )


def validate_v1_fq_causal_role_graph(
    *,
    ladder: V1FrozenCandidateLadder,
    certification_context_ids_by_stratum: Sequence[
        tuple[str, Sequence[str]]
    ],
    certification_stream_commitment_hash: str,
    causal_generation_frame: V1AlexNetFQCausalGenerationFrameReceipt | None,
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt | None,
    discovery_prefix_freeze: V1AlexNetFQDiscoveryPrefixFreezeReceipt | None,
    certification_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt | None,
    expected_discovery_evidence_state_hash: str | None = None,
    require_complete_graph: bool = False,
) -> bool:
    """Validate the outcome-independent D-freeze/C-stream graph for one ladder.

    ``False`` means that no causal receipts were supplied and the caller did
    not require them.  Any partial graph fails closed.  The optional discovery
    evidence coordinate lets a modern composition bind the D prefix to its
    exact registered discovery state without changing historical receipts.
    """

    if type(ladder) is not V1FrozenCandidateLadder:
        raise TypeError("causal F/Q role graph requires a typed frozen ladder")
    ladder_hash = ladder.assert_integrity()
    contexts = freeze_v1_fq_certification_context_axis(
        certification_context_ids_by_stratum,
    )
    stream_commitment = _identity(
        certification_stream_commitment_hash,
        name="certification stream commitment hash",
    )
    receipts = (
        causal_generation_frame,
        discovery_stream,
        discovery_prefix_freeze,
        certification_stream,
    )
    complete = all(value is not None for value in receipts)
    if any(value is not None for value in receipts) and not complete:
        raise ValueError("causal F/Q role receipt graph must be supplied atomically")
    if not complete:
        if require_complete_graph:
            raise ValueError("production F/Q issuance requires the complete D/C graph")
        return False
    if (
        type(causal_generation_frame)
        is not V1AlexNetFQCausalGenerationFrameReceipt
        or type(discovery_stream)
        is not V1AlexNetFQCausalRoleSamplingStreamReceipt
        or type(discovery_prefix_freeze)
        is not V1AlexNetFQDiscoveryPrefixFreezeReceipt
        or type(certification_stream)
        is not V1AlexNetFQCausalRoleSamplingStreamReceipt
    ):
        raise TypeError("causal F/Q role graph requires exact receipt types")
    assert causal_generation_frame is not None
    assert discovery_stream is not None
    assert discovery_prefix_freeze is not None
    assert certification_stream is not None
    causal_generation_frame.assert_integrity()
    discovery_stream.assert_integrity()
    discovery_prefix_freeze.assert_integrity()
    certification_stream.assert_integrity()
    d_counts = discovery_prefix_freeze.cumulative_stratum_counts
    d_prefix_contexts = (
        discovery_stream.target_assignment_receipt.generated_order[: d_counts[0]],
        discovery_stream.complement_assignment_receipt.generated_order[
            : d_counts[1]
        ],
    )
    expected_discovery_hash = (
        None
        if expected_discovery_evidence_state_hash is None
        else _identity(
            expected_discovery_evidence_state_hash,
            name="expected discovery evidence-state hash",
        )
    )
    if (
        causal_generation_frame.generation_index != ladder.proposal_generation
        or causal_generation_frame.current_mask_hash != ladder.current_mask_hash
        or causal_generation_frame.frozen_transform_hash
        != ladder.raw_logit_stream_source_hash
        or discovery_stream.role != "discovery"
        or discovery_stream.generation_frame_hash
        != causal_generation_frame.receipt_hash
        or discovery_prefix_freeze.generation_frame_hash
        != causal_generation_frame.receipt_hash
        or discovery_prefix_freeze.discovery_stream_hash
        != discovery_stream.receipt_hash
        or discovery_prefix_freeze.generation_index != ladder.proposal_generation
        or (
            expected_discovery_hash is not None
            and discovery_prefix_freeze.discovery_evidence_state_hash
            != expected_discovery_hash
        )
        or certification_stream.role != "certification"
        or certification_stream.generation_frame_hash
        != causal_generation_frame.receipt_hash
        or certification_stream.discovery_prefix_freeze_hash
        != discovery_prefix_freeze.receipt_hash
        or certification_stream.selected_shell_hash
        != discovery_prefix_freeze.selected_shell_hash
        or discovery_prefix_freeze.selected_shell_hash != ladder_hash
        or set(ladder.discovery_context_ids)
        != set((*d_prefix_contexts[0], *d_prefix_contexts[1]))
        or contexts != certification_stream.context_ids_by_stratum
        or stream_commitment != certification_stream.receipt_hash
    ):
        raise ValueError("causal F/Q role receipts differ from frozen D/C rows")
    return True


def _bind_physical_fq_envelope(
    *,
    freeze: V1ResponseExactificationFreeze,
    contexts: tuple[tuple[str, tuple[str, ...]], ...],
) -> V1CorrectedAttributionEnvelopeCertificate:
    source = freeze.ordered_envelope_certificate
    if tuple(name for name, _ in contexts) != source.stratum_names:
        raise ValueError("F/Q certification context strata differ from score strata")
    order_hashes = tuple(
        sampling_content_hash({"stratum_id": name, "context_ids": axis})
        for name, axis in contexts
    )
    return V1CorrectedAttributionEnvelopeCertificate(
        campaign_contract_hash=source.campaign_contract_hash,
        model_identity_hash=source.model_identity_hash,
        query_contract_hash=source.query_contract_hash,
        preprocessing_hash=source.preprocessing_hash,
        current_mask_hash=source.current_mask_hash,
        raw_logit_stream_source_hash=source.raw_logit_stream_source_hash,
        sampling_law_hash=source.sampling_law_hash,
        proposal_generation=source.proposal_generation,
        route_count=source.route_count,
        route_axis_hash=freeze.axis_bridge.physical_route_axis_hash,
        stratum_names=source.stratum_names,
        population_stratum_counts=tuple(len(axis) for _, axis in contexts),
        sealed_population_order_hashes=order_hashes,  # type: ignore[arg-type]
        absolute_upper_envelope_by_stratum=(
            source.absolute_upper_envelope_by_stratum
        ),
        verifier_source_hash=source.verifier_source_hash,
        proof_artifact_hash=content_hash(
            {
                "source_ordered_envelope_certificate_hash": (
                    source.certificate_hash
                ),
                "axis_bridge_receipt_hash": freeze.axis_bridge.receipt_hash,
                "certification_panel_context_order_hashes": order_hashes,
                "universal_preoutcome_input_support_unchanged": True,
                "route_envelope_values_position_preserved": True,
            },
            prefix="partition-free-v1-physical-fq-envelope-binding-",
        ),
    )


def allocate_v1_fq_alpha_family(
    *,
    master: V1MasterDecisionAlphaRegistration,
    proposal_schedule: V1ProposalOrderAlphaSchedule,
    proposal_design: V1GroupSequentialSerflingDesign,
    decision_scope: (
        V1ResponseExactificationDecisionScopeReceipt
        | V1CompositionDecisionScopeReceipt
        | V1CurrentLookFQAlphaCoordinateScopeReceipt
    ),
) -> V1FQAlphaFamilyAllocationReceipt:
    """Allocate the exact preregistered discovery-look F/Q family share.

    Historical response scopes retain their original field interpretation and
    payload.  Neutral composition scopes expose the same finite discovery-look
    alpha axis through explicit neutral names; they never acquire a
    score-resolved endpoint alias.  Minimal current-look scopes bind only the
    exact registered D coordinate and likewise leave the allocation payload
    unchanged.
    """

    master_hash = master.assert_integrity()
    schedule_hash = proposal_schedule.assert_integrity()
    proposal_design.assert_integrity()
    if type(decision_scope) is V1ResponseExactificationDecisionScopeReceipt:
        decision_scope.assert_integrity()
        score_look_index = decision_scope.score_order_look_index
        registered_maximum_looks = decision_scope.maximum_registered_score_looks
        neutral_scope_mismatch = False
    elif type(decision_scope) is V1CompositionDecisionScopeReceipt:
        decision_scope.assert_integrity()
        if decision_scope.composition_kind != (
            SAMPLED_CONDITIONAL_RESPONSE_GEOMETRY_COMPOSITION_KIND
        ):
            raise ValueError("neutral F/Q alpha requires geometry composition")
        score_look_index = decision_scope.discovery_score_look_index
        registered_maximum_looks = (
            decision_scope.maximum_registered_discovery_score_looks
        )
        neutral_scope_mismatch = (
            proposal_design.design_hash
            != decision_scope.discovery_score_design_hash
            or proposal_design.proposal_generation
            != decision_scope.proposal_generation
            or proposal_design.current_mask_hash != decision_scope.current_mask_hash
            or proposal_design.route_count != decision_scope.route_count
            or proposal_design.route_axis_hash
            != decision_scope.ordered_route_axis_hash
        )
    elif type(decision_scope) is V1CurrentLookFQAlphaCoordinateScopeReceipt:
        decision_scope.assert_integrity()
        score_look_index = decision_scope.discovery_score_look_index
        registered_maximum_looks = (
            decision_scope.maximum_registered_discovery_score_looks
        )
        neutral_scope_mismatch = (
            proposal_design.design_hash
            != decision_scope.discovery_score_design_hash
            or proposal_design.proposal_generation
            != decision_scope.proposal_generation
            or proposal_design.current_mask_hash != decision_scope.current_mask_hash
            or proposal_design.route_count != decision_scope.route_count
            or proposal_design.route_axis_hash
            != decision_scope.ordered_route_axis_hash
        )
    else:
        raise TypeError(
            "F/Q alpha allocation requires an exact legacy-response, "
            "neutral-composition, or outcome-free current-look coordinate scope"
        )
    generation = proposal_design.proposal_generation
    maximum_redesigns = len(proposal_design.alpha_by_look)
    if (
        master.proposal_order_alpha_schedule_hash != schedule_hash
        or master.campaign_contract_hash != proposal_schedule.campaign_contract_hash
        or master.campaign_contract_hash != proposal_design.campaign_contract_hash
        or master.controller_run_hash != proposal_schedule.controller_run_hash
        or master.maximum_proposal_generations
        != proposal_schedule.maximum_proposal_generations
        or master.maximum_proposal_generations <= generation
        or not _alpha_matches(
            master.proposal_order_familywise_alpha,
            proposal_schedule.trajectory_familywise_alpha,
        )
        or tuple(master.proposal_order_generation_allocations)
        != tuple(proposal_schedule.generation_alpha_allocations)
        or not _alpha_matches(
            master.proposal_order_generation_allocations[generation],
            proposal_design.generation_alpha_allocation,
        )
        or registered_maximum_looks != maximum_redesigns
        or score_look_index > maximum_redesigns
        or neutral_scope_mismatch
    ):
        raise ValueError("master alpha registration differs from proposal schedule")
    fq_generation_alpha = master.fq_generation_allocations[generation]
    proposal_look_weights = tuple(proposal_design.alpha_by_look)
    weight_total = math.fsum(proposal_look_weights)
    redesign_axis_list = [
        fq_generation_alpha * weight / weight_total
        for weight in proposal_look_weights[:-1]
    ]
    redesign_axis_list.append(
        fq_generation_alpha - math.fsum(redesign_axis_list)
    )
    redesign_axis = tuple(redesign_axis_list)
    redesign_generation = score_look_index - 1
    return V1FQAlphaFamilyAllocationReceipt(
        master_registration_hash=master_hash,
        proposal_order_alpha_schedule_hash=schedule_hash,
        proposal_generation=generation,
        fq_familywise_alpha=master.fq_familywise_alpha,
        fq_alpha_committed_before_generation=math.fsum(
            master.fq_generation_allocations[:generation]
        ),
        fq_generation_alpha_allocation=fq_generation_alpha,
        proposal_order_generation_alpha_allocation=(
            master.proposal_order_generation_allocations[generation]
        ),
        oracle_familywise_alpha_reserve=master.oracle_familywise_alpha,
        maximum_oracle_attempts=master.maximum_oracle_attempts,
        oracle_attempt_alpha_axis_hash=master.oracle_attempt_alpha_axis_hash,
        fq_ladder_redesign_generation=redesign_generation,
        fq_alpha_committed_before_redesign=math.fsum(
            redesign_axis[:redesign_generation]
        ),
        fq_redesign_alpha_allocation=redesign_axis[redesign_generation],
        score_order_look_index=score_look_index,
        maximum_fq_ladder_redesigns=maximum_redesigns,
        proposal_order_look_alpha_weights=proposal_look_weights,
        fq_redesign_alpha_allocations=redesign_axis,
    )


def _allocate_fq_alpha(
    *,
    master: V1MasterDecisionAlphaRegistration,
    proposal_schedule: V1ProposalOrderAlphaSchedule,
    proposal_design: V1GroupSequentialSerflingDesign,
    decision_scope: V1ResponseExactificationDecisionScopeReceipt,
) -> V1FQAlphaFamilyAllocationReceipt:
    """Historical internal spelling retained for byte-identical replay."""

    return allocate_v1_fq_alpha_family(
        master=master,
        proposal_schedule=proposal_schedule,
        proposal_design=proposal_design,
        decision_scope=decision_scope,
    )


def build_v1_response_exactification_artifacts(
    *,
    freeze: V1ResponseExactificationFreeze,
    proposal_order_design: V1GroupSequentialSerflingDesign,
    endpoint_envelope_bundle: V1CancellationAwareEndpointEnvelopeBundle,
    master_alpha_registration: V1MasterDecisionAlphaRegistration,
    proposal_alpha_schedule: V1ProposalOrderAlphaSchedule,
    certification_context_ids_by_stratum: Sequence[
        tuple[str, Sequence[str]]
    ],
    stratum_weights: Sequence[tuple[str, float]],
    certification_stream_commitment_hash: str,
    projection_kernel_source_hash: str,
    causal_generation_frame: V1AlexNetFQCausalGenerationFrameReceipt | None = None,
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt | None = None,
    discovery_prefix_freeze: V1AlexNetFQDiscoveryPrefixFreezeReceipt | None = None,
    certification_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt | None = None,
) -> V1ResponseExactificationArtifacts:
    """Issue the pre-outcome selected-shell F/Q design.

    Certification rows remain fresh from score discovery, but are registered as
    a common random panel that may be reprojected across the finite
    score-look-redesign family under the explicit receipt below.
    """

    if type(freeze) is not V1ResponseExactificationFreeze:
        raise TypeError("response artifacts require a typed exactification freeze")
    if type(proposal_order_design) is not V1GroupSequentialSerflingDesign:
        raise TypeError("response artifacts require the proposal-order design")
    if (
        type(endpoint_envelope_bundle)
        is not V1CancellationAwareEndpointEnvelopeBundle
    ):
        raise TypeError("response artifacts require the endpoint envelope bundle")
    if type(master_alpha_registration) is not V1MasterDecisionAlphaRegistration:
        raise TypeError("response artifacts require the master alpha registration")
    if type(proposal_alpha_schedule) is not V1ProposalOrderAlphaSchedule:
        raise TypeError("response artifacts require the proposal alpha schedule")
    freeze.assert_integrity()
    proposal_order_design.assert_integrity()
    if (
        proposal_order_design.design_hash
        != freeze.replay_receipt.proposal_order_base_serfling_design_hash
    ):
        raise ValueError("proposal-order design differs from the replay freeze")
    endpoint_envelope_bundle.assert_integrity()
    if endpoint_envelope_bundle.ladder_hash != freeze.ladder.ladder_hash:
        raise ValueError("endpoint envelope bundle names another frozen ladder")
    contexts = _frozen_context_axis(
        certification_context_ids_by_stratum,
        name="F/Q certification common panel",
    )
    causal_full_frame = validate_v1_fq_causal_role_graph(
        ladder=freeze.ladder,
        certification_context_ids_by_stratum=contexts,
        certification_stream_commitment_hash=(
            certification_stream_commitment_hash
        ),
        causal_generation_frame=causal_generation_frame,
        discovery_stream=discovery_stream,
        discovery_prefix_freeze=discovery_prefix_freeze,
        certification_stream=certification_stream,
    )
    discovery_overlap = set(freeze.ladder.discovery_context_ids).intersection(
        context for _, axis in contexts for context in axis
    )
    if not causal_full_frame and discovery_overlap:
        raise ValueError(
            "legacy F/Q certification rows must be fresh and disjoint from discovery"
        )
    weights = tuple(
        (
            _identity(name, name="F/Q stratum weight name"),
            float(value),
        )
        for name, value in stratum_weights
    )
    if (
        tuple(name for name, _ in weights) != tuple(name for name, _ in contexts)
        or any(not math.isfinite(value) or value <= 0.0 for _, value in weights)
        or not _alpha_matches(math.fsum(value for _, value in weights), 1.0)
    ):
        raise ValueError("F/Q stratum weights must be positive and sum to one")
    physical_envelope = _bind_physical_fq_envelope(
        freeze=freeze,
        contexts=contexts,
    )
    allocation = _allocate_fq_alpha(
        master=master_alpha_registration,
        proposal_schedule=proposal_alpha_schedule,
        proposal_design=proposal_order_design,
        decision_scope=freeze.replay_receipt.decision_scope_receipt,
    )
    population_hash = content_hash(
        {"certification_context_ids_by_stratum": contexts}
    )
    common_panel = V1FQCommonCertificationPanelReceipt(
        proposal_generation=proposal_order_design.proposal_generation,
        active_score_order_look_index=allocation.score_order_look_index,
        maximum_fq_ladder_redesigns=allocation.maximum_fq_ladder_redesigns,
        proposal_order_design_hash=proposal_order_design.design_hash,
        score_order_registration_hash=(
            freeze.replay_receipt.decision_scope_receipt
            .score_order_registration_hash
        ),
        score_order_look_receipt_hash=(
            freeze.replay_receipt.decision_scope_receipt
            .score_order_look_receipt_hash
        ),
        decision_scope_receipt_hash=(
            freeze.replay_receipt.decision_scope_receipt.receipt_hash
        ),
        fq_alpha_allocation_receipt_hash=allocation.receipt_hash,
        fq_redesign_alpha_schedule_hash=(
            allocation.fq_redesign_alpha_schedule_hash
        ),
        fq_redesign_alpha_allocations=allocation.fq_redesign_alpha_allocations,
        proposal_order_look_alpha_weights=(
            allocation.proposal_order_look_alpha_weights
        ),
        certification_context_population_hash=population_hash,
        certification_stream_commitment_hash=_identity(
            certification_stream_commitment_hash,
            name="certification stream commitment hash",
        ),
    )
    design = V1FrozenLadderCertificationDesign(
        design_id=content_hash(
            {
                "freeze_hash": freeze.freeze_hash,
                "endpoint_envelope_bundle_hash": endpoint_envelope_bundle.bundle_hash,
                "fq_alpha_allocation_receipt_hash": allocation.receipt_hash,
                "fq_common_panel_receipt_hash": common_panel.receipt_hash,
                "fq_common_panel_policy": (
                    common_panel.certification_panel_policy
                ),
                "certification_context_population_hash": population_hash,
                "certification_stream_commitment_hash": (
                    certification_stream_commitment_hash
                ),
            },
            prefix="partition-free-v1-response-fq-design-id-",
        ),
        ladder=freeze.ladder,
        envelope_certificate=physical_envelope,
        certification_context_ids_by_stratum=contexts,
        stratum_weights=weights,
        certification_context_population_hash=population_hash,
        certification_stream_commitment_hash=_identity(
            certification_stream_commitment_hash,
            name="certification stream commitment hash",
        ),
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
    return V1ResponseExactificationArtifacts(
        freeze=freeze,
        physical_fq_envelope_certificate=physical_envelope,
        endpoint_envelope_bundle=endpoint_envelope_bundle,
        fq_alpha_allocation=allocation,
        common_panel_receipt=common_panel,
        certification_design=design,
    )


__all__ = [
    "CANCELLATION_ENDPOINT_BUNDLE_SCHEMA",
    "FQ_ALPHA_ALLOCATION_RECEIPT_SCHEMA",
    "FQ_COMMON_CERTIFICATION_PANEL_POLICY",
    "FQ_COMMON_CERTIFICATION_PANEL_RECEIPT_SCHEMA",
    "FQ_SCORE_LOOK_PROPORTIONAL_REDESIGN_POLICY",
    "MASTER_DECISION_ALPHA_REGISTRATION_SCHEMA",
    "ORACLE_ATTEMPT_ALPHA_AXIS_SCHEMA",
    "ORDINAL_PHYSICAL_AXIS_BRIDGE_SCHEMA",
    "RESPONSE_EXACTIFICATION_ARTIFACTS_SCHEMA",
    "RESPONSE_EXACTIFICATION_DECISION_SCOPE_SCHEMA",
    "RESPONSE_EXACTIFICATION_FREEZE_SCHEMA",
    "RESPONSE_EXACTIFICATION_HYBRID_REPLAY_SCHEMA",
    "RESPONSE_EXACTIFICATION_HYBRID_FREEZE_SCHEMA",
    "RESPONSE_EXACTIFICATION_REPLAY_SCHEMA",
    "V1CancellationAwareEndpointEnvelopeBundle",
    "V1FQAlphaFamilyAllocationReceipt",
    "V1FQCommonCertificationPanelReceipt",
    "V1MasterDecisionAlphaRegistration",
    "V1OrdinalPhysicalAxisBridgeReceipt",
    "V1ResponseExactificationArtifacts",
    "V1ResponseExactificationDecisionScopeReceipt",
    "V1ResponseExactificationFreeze",
    "V1ResponseExactificationReplayReceipt",
    "allocate_v1_fq_alpha_family",
    "bind_v1_cancellation_aware_endpoint_envelopes",
    "build_v1_response_exactification_artifacts",
    "freeze_v1_fq_certification_context_axis",
    "freeze_v1_response_exactified_ladder",
    "validate_v1_fq_causal_role_graph",
]
