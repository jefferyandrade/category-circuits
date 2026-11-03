"""Validation-independent Pareto milestones and scientific-horizon dispositions.

The historical comparator contributes exactly two decision coordinates to this
surface: its retained-route count and the content hash of the sealed artifact
that registered that scalar.  In particular, no comparator mask, score order,
or trajectory is accepted by any type in this module.

The milestone protocol is deliberately split at the ordinary controller
checkpoint seam.  A transition that lands on the registered count creates a
pending milestone.  Only a subsequent, content-addressed checkpoint binding
can turn that pending value into the durable milestone receipt.  This keeps
the comparison checkpoint recoverable without making it a terminal endpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Any, Final, Literal

from .contracts import content_hash
from .modern_v1_fqr_horizon import FQR_NO_CANDIDATE_HORIZON_STOP_REASON

_REGISTRATION_SCHEMA: Final = "partition_free_v1_pareto_milestone_registration_v1"
_PREFIX_RUNG_SCHEMA: Final = "partition_free_v1_exact_prefix_rung_registration_v1"
_PENDING_SCHEMA: Final = "partition_free_v1_pareto_milestone_pending_v1"
_CHECKPOINT_BINDING_SCHEMA: Final = (
    "partition_free_v1_pareto_milestone_checkpoint_binding_v1"
)
_RECEIPT_SCHEMA: Final = "partition_free_v1_pareto_milestone_receipt_v1"
_HORIZON_SCHEMA: Final = "partition_free_v1_scientific_horizon_disposition_v1"

REGISTERED_MINIMUM_RETAINED_ROUTE_COUNT_REACHED: Final = (
    "registered_minimum_retained_route_count_reached"
)
ALL_REGISTERED_ACTIONS_INFEASIBLE_AT_FULL_LOOK: Final = (
    "all_registered_current_generation_actions_certified_infeasible_at_full_look"
)
ALL_REGISTERED_FQR_ACTIONS_REJECTED_BY_PAIRED_BCE_BARRIER: Final = (
    "all_registered_current_generation_fqr_actions_rejected_by_paired_bce_barrier"
)

ScientificHorizonClass = Literal[
    "active",
    "scientific_endpoint",
    "resource_exhaustion_indeterminate",
    "owner_registered_stop",
]


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _unique_strings(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(_nonempty(value, label) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{label} must be nonempty and unique")
    return result


def _payload(instance: object) -> Mapping[str, Any]:
    fields = instance.__dataclass_fields__  # type: ignore[attr-defined]
    return MappingProxyType({name: getattr(instance, name) for name in fields})


def _strict_values(cls: type[Any], payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} payload must be a mapping")
    expected = set(cls.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError(f"{label} payload fields differ")
    return dict(payload)


@dataclass(frozen=True, slots=True)
class V1ParetoMilestoneRegistration:
    """Pre-outcome scalar comparison milestone with no comparator structure."""

    campaign_contract_hash: str
    route_axis_hash: str
    route_count: int
    comparator_retained_route_count: int
    comparator_registration_artifact_hash: str
    validation_independent: bool = True
    comparator_mask_supplied: bool = False
    comparator_order_supplied: bool = False
    comparator_trajectory_supplied: bool = False
    schema: str = _REGISTRATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "route_axis_hash",
            "comparator_registration_artifact_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        route_count = _positive_int(self.route_count, "route_count")
        retained = _positive_int(
            self.comparator_retained_route_count,
            "comparator_retained_route_count",
        )
        if retained >= route_count:
            raise ValueError(
                "comparator retained-route count must be below the v1 route count"
            )
        if (
            self.validation_independent is not True
            or self.comparator_mask_supplied is not False
            or self.comparator_order_supplied is not False
            or self.comparator_trajectory_supplied is not False
            or self.schema != _REGISTRATION_SCHEMA
        ):
            raise ValueError(
                "Pareto milestone registration imported comparator structure or outcomes"
            )
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "comparator_retained_route_count", retained)

    @property
    def registration_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-pareto-milestone-registration-")

    def to_payload(self) -> Mapping[str, Any]:
        return _payload(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> V1ParetoMilestoneRegistration:
        return cls(**_strict_values(cls, payload, "Pareto milestone registration"))


@dataclass(frozen=True, slots=True)
class V1ExactPrefixRungRegistration:
    """Pre-oracle identity of the exact v1-order rung that lands on ``k_cmp``."""

    campaign_contract_hash: str
    milestone_registration_hash: str
    acquisition_receipt_hash: str
    generation_index: int
    parent_mask_hash: str
    parent_retained_route_count: int
    comparator_retained_route_count: int
    required_removed_route_count: int
    queue_index: int
    candidate_id: str
    candidate_mask_hash: str
    removed_route_index_hash: str
    v1_order_receipt_hash: str
    validation_independent: bool = True
    exact_v1_order_prefix: bool = True
    comparator_mask_used: bool = False
    comparator_order_used: bool = False
    comparator_trajectory_used: bool = False
    oracle_outcomes_opened: bool = False
    schema: str = _PREFIX_RUNG_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "milestone_registration_hash",
            "acquisition_receipt_hash",
            "parent_mask_hash",
            "candidate_id",
            "candidate_mask_hash",
            "removed_route_index_hash",
            "v1_order_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        generation = _nonnegative_int(self.generation_index, "generation_index")
        parent_count = _positive_int(
            self.parent_retained_route_count,
            "parent_retained_route_count",
        )
        comparator_count = _positive_int(
            self.comparator_retained_route_count,
            "comparator_retained_route_count",
        )
        removed = _positive_int(
            self.required_removed_route_count,
            "required_removed_route_count",
        )
        queue_index = _nonnegative_int(self.queue_index, "queue_index")
        if parent_count <= comparator_count or removed != parent_count - comparator_count:
            raise ValueError("exact prefix rung does not land on the comparator count")
        if self.parent_mask_hash == self.candidate_mask_hash:
            raise ValueError("exact prefix rung must change the v1 mask")
        if (
            self.validation_independent is not True
            or self.exact_v1_order_prefix is not True
            or self.comparator_mask_used is not False
            or self.comparator_order_used is not False
            or self.comparator_trajectory_used is not False
            or self.oracle_outcomes_opened is not False
            or self.schema != _PREFIX_RUNG_SCHEMA
        ):
            raise ValueError("exact prefix-rung registration crossed its causal firewall")
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "parent_retained_route_count", parent_count)
        object.__setattr__(self, "comparator_retained_route_count", comparator_count)
        object.__setattr__(self, "required_removed_route_count", removed)
        object.__setattr__(self, "queue_index", queue_index)

    @property
    def registration_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-exact-prefix-rung-registration-")

    def to_payload(self) -> Mapping[str, Any]:
        return _payload(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> V1ExactPrefixRungRegistration:
        return cls(**_strict_values(cls, payload, "exact prefix-rung registration"))


@dataclass(frozen=True, slots=True)
class V1ParetoMilestonePending:
    """Exact post-transition milestone state awaiting ordinary checkpointing."""

    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    milestone_registration_hash: str
    comparator_retained_route_count: int
    milestone_mask_hash: str
    mask_lineage_hashes: tuple[str, ...]
    transition_receipt_hashes: tuple[str, ...]
    accepted_trajectory_hash: str
    accepted_step_count: int
    candidate_attempt_count: int
    milestone_transition_receipt_hash: str
    logical_sequence: int
    validation_independent: bool = True
    terminal_validation_opened: bool = False
    schema: str = _PENDING_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "milestone_registration_hash",
            "milestone_mask_hash",
            "accepted_trajectory_hash",
            "milestone_transition_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        retained = _positive_int(
            self.comparator_retained_route_count,
            "comparator_retained_route_count",
        )
        lineage = _unique_strings(self.mask_lineage_hashes, "mask_lineage_hash")
        transitions = tuple(
            _nonempty(value, "transition_receipt_hash")
            for value in self.transition_receipt_hashes
        )
        steps = _positive_int(self.accepted_step_count, "accepted_step_count")
        attempts = _positive_int(self.candidate_attempt_count, "candidate_attempt_count")
        sequence = _nonnegative_int(self.logical_sequence, "logical_sequence")
        if (
            len(lineage) != steps + 1
            or len(transitions) != steps
            or len(set(transitions)) != len(transitions)
            or lineage[-1] != self.milestone_mask_hash
            or transitions[-1] != self.milestone_transition_receipt_hash
            or attempts < steps
        ):
            raise ValueError("pending Pareto milestone has incomplete controller lineage")
        if (
            self.validation_independent is not True
            or self.terminal_validation_opened is not False
            or self.schema != _PENDING_SCHEMA
        ):
            raise ValueError("pending Pareto milestone crossed the validation firewall")
        object.__setattr__(self, "comparator_retained_route_count", retained)
        object.__setattr__(self, "mask_lineage_hashes", lineage)
        object.__setattr__(self, "transition_receipt_hashes", transitions)
        object.__setattr__(self, "accepted_step_count", steps)
        object.__setattr__(self, "candidate_attempt_count", attempts)
        object.__setattr__(self, "logical_sequence", sequence)

    @property
    def pending_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-pareto-milestone-pending-")

    def to_payload(self) -> Mapping[str, Any]:
        return _payload(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> V1ParetoMilestonePending:
        return cls(**_strict_values(cls, payload, "pending Pareto milestone"))


@dataclass(frozen=True, slots=True)
class V1ParetoMilestoneCheckpointBinding:
    """Evidence that the pending transition crossed the ordinary checkpoint seam."""

    pending_milestone_hash: str
    controller_state_hash: str
    controller_snapshot_receipt_hash: str
    checkpoint_artifact_hash: str
    checkpoint_manifest_hash: str
    checkpoint_codec_schema: str
    atomic_checkpoint_write_completed: bool = True
    restore_identity_verified: bool = True
    schema: str = _CHECKPOINT_BINDING_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "pending_milestone_hash",
            "controller_state_hash",
            "controller_snapshot_receipt_hash",
            "checkpoint_artifact_hash",
            "checkpoint_manifest_hash",
            "checkpoint_codec_schema",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if (
            self.atomic_checkpoint_write_completed is not True
            or self.restore_identity_verified is not True
            or self.schema != _CHECKPOINT_BINDING_SCHEMA
        ):
            raise ValueError("milestone checkpoint binding lacks completed restore evidence")

    @property
    def binding_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-pareto-checkpoint-binding-")

    def to_payload(self) -> Mapping[str, Any]:
        return _payload(self)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> V1ParetoMilestoneCheckpointBinding:
        return cls(**_strict_values(cls, payload, "Pareto milestone checkpoint binding"))


@dataclass(frozen=True, slots=True)
class V1ParetoMilestoneReceipt:
    """Immutable, validation-independent, nonterminal matched-sparsity checkpoint."""

    pending: V1ParetoMilestonePending
    checkpoint_binding: V1ParetoMilestoneCheckpointBinding
    continue_after_milestone: bool = True
    validation_independent: bool = True
    terminal_validation_opened: bool = False
    comparator_mask_used: bool = False
    comparator_order_used: bool = False
    comparator_trajectory_used: bool = False
    schema: str = _RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.pending) is not V1ParetoMilestonePending:
            raise TypeError("milestone receipt requires an exact pending milestone")
        if type(self.checkpoint_binding) is not V1ParetoMilestoneCheckpointBinding:
            raise TypeError("milestone receipt requires an exact checkpoint binding")
        if self.checkpoint_binding.pending_milestone_hash != self.pending.pending_hash:
            raise ValueError("milestone checkpoint binding names another transition")
        if (
            self.continue_after_milestone is not True
            or self.validation_independent is not True
            or self.terminal_validation_opened is not False
            or self.comparator_mask_used is not False
            or self.comparator_order_used is not False
            or self.comparator_trajectory_used is not False
            or self.schema != _RECEIPT_SCHEMA
        ):
            raise ValueError("Pareto milestone receipt crossed its nonterminal firewall")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-pareto-milestone-receipt-")

    def to_payload(self) -> Mapping[str, Any]:
        payload = dict(_payload(self))
        payload["pending"] = self.pending.to_payload()
        payload["checkpoint_binding"] = self.checkpoint_binding.to_payload()
        return MappingProxyType(payload)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> V1ParetoMilestoneReceipt:
        values = _strict_values(cls, payload, "Pareto milestone receipt")
        values["pending"] = V1ParetoMilestonePending.from_payload(values["pending"])
        values["checkpoint_binding"] = V1ParetoMilestoneCheckpointBinding.from_payload(
            values["checkpoint_binding"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class V1ScientificHorizonDisposition:
    """Typed meaning of a controller stop; caps never become scientific endpoints."""

    stop_reason: str | None
    horizon_class: ScientificHorizonClass
    scientific_endpoint: bool
    resource_exhaustion_indeterminate: bool
    terminal_science_authority: bool = False
    schema: str = _HORIZON_SCHEMA

    def __post_init__(self) -> None:
        if self.stop_reason is not None:
            object.__setattr__(self, "stop_reason", _nonempty(self.stop_reason, "stop_reason"))
        if self.horizon_class not in {
            "active",
            "scientific_endpoint",
            "resource_exhaustion_indeterminate",
            "owner_registered_stop",
        }:
            raise ValueError("unsupported scientific-horizon class")
        expected = {
            "active": (False, False),
            "scientific_endpoint": (True, False),
            "resource_exhaustion_indeterminate": (False, True),
            "owner_registered_stop": (False, False),
        }[self.horizon_class]
        if (
            (self.scientific_endpoint, self.resource_exhaustion_indeterminate)
            != expected
            or self.terminal_science_authority is not False
            or self.schema != _HORIZON_SCHEMA
        ):
            raise ValueError("scientific-horizon flags disagree with their typed class")
        if (self.stop_reason is None) != (self.horizon_class == "active"):
            raise ValueError("active horizon and stop reason disagree")

    @property
    def disposition_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-scientific-horizon-")


def classify_v1_scientific_horizon(
    stop_reason: str | None,
    *,
    registered_minimum_is_scientific_endpoint: bool = True,
) -> V1ScientificHorizonDisposition:
    """Classify a controller reason without granting terminal-reveal authority.

    The keyword preserves the historical generic-controller interpretation
    while allowing the canonical Pareto terminal gate to refuse a configured
    density floor that lacks a natural-legality proof.
    """

    if type(registered_minimum_is_scientific_endpoint) is not bool:
        raise TypeError(
            "registered-minimum scientific disposition must be one boolean"
        )

    if stop_reason is None:
        horizon_class: ScientificHorizonClass = "active"
    elif (
        stop_reason == REGISTERED_MINIMUM_RETAINED_ROUTE_COUNT_REACHED
        and registered_minimum_is_scientific_endpoint
    ) or stop_reason in {
        ALL_REGISTERED_ACTIONS_INFEASIBLE_AT_FULL_LOOK,
        ALL_REGISTERED_FQR_ACTIONS_REJECTED_BY_PAIRED_BCE_BARRIER,
        FQR_NO_CANDIDATE_HORIZON_STOP_REASON,
    }:
        horizon_class = "scientific_endpoint"
    elif stop_reason == "explicit_registered_stop":
        horizon_class = "owner_registered_stop"
    else:
        horizon_class = "resource_exhaustion_indeterminate"
    return V1ScientificHorizonDisposition(
        stop_reason=stop_reason,
        horizon_class=horizon_class,
        scientific_endpoint=horizon_class == "scientific_endpoint",
        resource_exhaustion_indeterminate=(
            horizon_class == "resource_exhaustion_indeterminate"
        ),
    )


__all__ = [
    "ALL_REGISTERED_ACTIONS_INFEASIBLE_AT_FULL_LOOK",
    "ALL_REGISTERED_FQR_ACTIONS_REJECTED_BY_PAIRED_BCE_BARRIER",
    "FQR_NO_CANDIDATE_HORIZON_STOP_REASON",
    "REGISTERED_MINIMUM_RETAINED_ROUTE_COUNT_REACHED",
    "ScientificHorizonClass",
    "V1ExactPrefixRungRegistration",
    "V1ParetoMilestoneCheckpointBinding",
    "V1ParetoMilestonePending",
    "V1ParetoMilestoneReceipt",
    "V1ParetoMilestoneRegistration",
    "V1ScientificHorizonDisposition",
    "classify_v1_scientific_horizon",
]
