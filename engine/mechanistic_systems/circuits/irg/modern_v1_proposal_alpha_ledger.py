"""Trajectory-wide alpha control for sampled modern-v1 proposal ordering.

The proposal statistics owner intentionally requires an external ledger.  This
module supplies that missing control-plane object.  A schedule is frozen before
proposal outcomes and allocates a strictly positive amount of one declared
trajectory familywise-error budget to every possible proposal generation.  A
ledger head then registers designs consecutively, before their outcomes, and
commits each generation's *full* scheduled allocation, whether or not all of
its registered look alpha is eventually used.  Consequently stopping early
cannot replenish a later generation.

The ledger contains identities, registered design hashes, and alpha arithmetic
only.  It contains no samples, scores, response geometry, losses, or masks and
therefore grants no F/Q, BCE-oracle, or mask authority.  Persisting exactly one
head per ``controller_run_hash`` remains the controller's compare-and-swap
responsibility; content addressing makes forks detectable but cannot choose an
authoritative external head by itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Any, Final

from mechanistic_systems.circuits.irg.modern_v1_proposal_statistics import (
    FIXED_TRAJECTORY_FWER_POLICY,
    V1GroupSequentialSerflingDesign,
)
from mechanistic_systems.contracts.content_address import content_hash


PROPOSAL_ORDER_ALPHA_SCHEDULE_SCHEMA: Final = (
    "partition_free_v1_proposal_order_trajectory_alpha_schedule_v1"
)
PROPOSAL_ORDER_ALPHA_LEDGER_SCHEMA: Final = (
    "partition_free_v1_proposal_order_trajectory_alpha_ledger_v1"
)
PROPOSAL_ORDER_ALPHA_REPLAY_SCHEMA: Final = (
    "partition_free_v1_proposal_order_trajectory_alpha_replay_v1"
)
EXPLICIT_PREOUTCOME_ALLOCATION_POLICY: Final = (
    "explicit_preoutcome_non_data_dependent_generation_allocation_v1"
)
PROPOSAL_ORDER_ALPHA_AUTHORITY: Final = (
    "trajectory_fwer_budget_for_sampled_corrected_attribution_order_only_v1"
)

_SCHEDULE_PREFIX: Final = "partition-free-v1-proposal-order-alpha-schedule-"
_LEDGER_PREFIX: Final = "partition-free-v1-proposal-order-alpha-ledger-"
_ALPHA_MATCH_ULPS: Final = 8
_LEDGER_ISSUER_TOKEN: Final = object()


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be one positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return result


def _finite_alpha(value: object, *, name: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        qualifier = "positive " if positive else "nonnegative "
        raise ValueError(f"{name} must be one finite {qualifier}scalar")
    result = float(value)
    lower_bound_ok = result > 0.0 if positive else result >= 0.0
    if not math.isfinite(result) or not lower_bound_ok:
        qualifier = "positive " if positive else "nonnegative "
        raise ValueError(f"{name} must be one finite {qualifier}scalar")
    return result


def _alpha_tolerance(*values: float) -> float:
    """Return a small, scale-local tolerance for duplicate alpha arithmetic.

    Budget inequalities themselves are strict in represented binary64 values.
    The tolerance is used only when two independently materialized fields are
    required to describe the same registered number (for example ``sum`` versus
    ``math.fsum`` at a ledger boundary).
    """

    scale = max((abs(float(value)) for value in values), default=0.0)
    return _ALPHA_MATCH_ULPS * math.ulp(scale)


def _alpha_matches(left: float, right: float) -> bool:
    if left == 0.0 or right == 0.0:
        return left == right
    return abs(left - right) <= _alpha_tolerance(left, right)


def _require_false(value: object, *, name: str) -> None:
    if value is not False:
        raise ValueError(f"{name} must remain false")


def _require_true(value: object, *, name: str) -> None:
    if value is not True:
        raise ValueError(f"{name} must remain true")


@dataclass(frozen=True, slots=True)
class V1ProposalOrderAlphaSchedule:
    """Pre-outcome finite allocation of proposal-order trajectory FWER.

    ``allocation_policy_kind`` and ``allocation_assumption`` are mandatory so
    that a human-selected allocation is never mislabeled as first-principles or
    silently replaced by a default significance level.
    """

    campaign_contract_hash: str
    controller_run_hash: str
    preoutcome_registration_receipt_hash: str
    maximum_proposal_generations: int
    trajectory_familywise_alpha: float
    generation_alpha_allocations: tuple[float, ...]
    allocation_policy_kind: str
    allocation_assumption: str
    issued_before_outcomes: bool = True
    allocation_depends_on_outcomes: bool = False
    contains_outcomes: bool = False
    f_q_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    alpha_policy_kind: str = FIXED_TRAJECTORY_FWER_POLICY
    authority_scope: str = PROPOSAL_ORDER_ALPHA_AUTHORITY
    schema: str = PROPOSAL_ORDER_ALPHA_SCHEDULE_SCHEMA
    _schedule_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "controller_run_hash",
            "preoutcome_registration_receipt_hash",
            "allocation_policy_kind",
            "allocation_assumption",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        maximum = _positive_int(
            self.maximum_proposal_generations,
            name="maximum proposal generations",
        )
        familywise = _finite_alpha(
            self.trajectory_familywise_alpha,
            name="trajectory familywise alpha",
            positive=True,
        )
        if familywise >= 1.0:
            raise ValueError("trajectory familywise alpha must lie in (0,1)")
        allocations = tuple(
            _finite_alpha(
                value,
                name=f"generation alpha allocation {index}",
                positive=True,
            )
            for index, value in enumerate(self.generation_alpha_allocations)
        )
        if len(allocations) != maximum:
            raise ValueError(
                "generation alpha allocations must explicitly cover every "
                "possible proposal generation"
            )
        # This is deliberately strict.  Matching/replay fields receive a small
        # ULP tolerance, but the represented allocation itself may not exceed
        # the represented trajectory FWER budget.
        if math.fsum(allocations) > familywise:
            raise ValueError(
                "generation alpha allocations exceed trajectory familywise alpha"
            )
        _require_true(
            self.issued_before_outcomes,
            name="issued_before_outcomes",
        )
        for name in (
            "allocation_depends_on_outcomes",
            "contains_outcomes",
            "f_q_authority",
            "bce_oracle_authority",
            "mask_commit_authority",
        ):
            _require_false(getattr(self, name), name=name)
        if self.allocation_policy_kind != EXPLICIT_PREOUTCOME_ALLOCATION_POLICY:
            raise ValueError(
                "proposal alpha allocations require the explicit pre-outcome policy"
            )
        if self.alpha_policy_kind != FIXED_TRAJECTORY_FWER_POLICY:
            raise ValueError(
                "proposal alpha schedule requires fixed non-recycling trajectory FWER"
            )
        if self.authority_scope != PROPOSAL_ORDER_ALPHA_AUTHORITY:
            raise ValueError("proposal alpha schedule authority scope is invalid")
        if self.schema != PROPOSAL_ORDER_ALPHA_SCHEDULE_SCHEMA:
            raise ValueError("proposal alpha schedule schema is invalid")
        object.__setattr__(self, "maximum_proposal_generations", maximum)
        object.__setattr__(self, "trajectory_familywise_alpha", familywise)
        object.__setattr__(self, "generation_alpha_allocations", allocations)
        # Eagerly seal the freshly validated object.  This closes the usual
        # lazy-hash gap in which ``object.__setattr__`` could mutate a frozen
        # instance before its first integrity check established a baseline.
        object.__setattr__(self, "_schedule_hash", self._compute_schedule_hash())

    @property
    def total_allocated_alpha(self) -> float:
        return math.fsum(self.generation_alpha_allocations)

    @property
    def permanently_unallocated_alpha(self) -> float:
        return self.trajectory_familywise_alpha - self.total_allocated_alpha

    def allocation_for_generation(self, proposal_generation: int) -> float:
        generation = _nonnegative_int(
            proposal_generation,
            name="proposal generation",
        )
        if generation >= self.maximum_proposal_generations:
            raise IndexError("proposal alpha schedule is exhausted")
        return self.generation_alpha_allocations[generation]

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "campaign_contract_hash": self.campaign_contract_hash,
            "controller_run_hash": self.controller_run_hash,
            "preoutcome_registration_receipt_hash": (
                self.preoutcome_registration_receipt_hash
            ),
            "maximum_proposal_generations": self.maximum_proposal_generations,
            "trajectory_familywise_alpha": self.trajectory_familywise_alpha,
            "generation_alpha_allocations": self.generation_alpha_allocations,
            "allocation_policy_kind": self.allocation_policy_kind,
            "allocation_assumption": self.allocation_assumption,
            "issued_before_outcomes": self.issued_before_outcomes,
            "allocation_depends_on_outcomes": self.allocation_depends_on_outcomes,
            "contains_outcomes": self.contains_outcomes,
            "f_q_authority": self.f_q_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "alpha_policy_kind": self.alpha_policy_kind,
            "authority_scope": self.authority_scope,
        }

    def _compute_schedule_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_SCHEDULE_PREFIX)

    @property
    def schedule_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = self._compute_schedule_hash()
        if self._schedule_hash is not None and observed != self._schedule_hash:
            raise ValueError("proposal alpha schedule was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("proposal alpha schedule hash does not match")
        object.__setattr__(self, "_schedule_hash", observed)
        return observed

    def require_f_q_authority(self) -> None:
        raise PermissionError("proposal alpha schedule has no F/Q authority")

    def require_bce_oracle_authority(self) -> None:
        raise PermissionError("proposal alpha schedule has no BCE-oracle authority")

    def require_mask_commit_authority(self) -> None:
        raise PermissionError("proposal alpha schedule has no mask-commit authority")

    def _constructor_payload(self) -> dict[str, Any]:
        return dict(self._hash_payload())

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any], str]]:
        return _rebuild_schedule, (
            self._constructor_payload(),
            self.assert_integrity(),
        )


def _ledger_hash_payload(
    *,
    schedule_hash: str,
    proposal_generation: int,
    registered_design_hashes: tuple[str, ...],
    committed_generation_allocations: tuple[float, ...],
    prior_state_hash: str | None,
    cumulative_committed_alpha: float,
    contains_outcomes: bool = False,
    f_q_authority: bool = False,
    bce_oracle_authority: bool = False,
    mask_commit_authority: bool = False,
    alpha_policy_kind: str = FIXED_TRAJECTORY_FWER_POLICY,
    authority_scope: str = PROPOSAL_ORDER_ALPHA_AUTHORITY,
    schema: str = PROPOSAL_ORDER_ALPHA_LEDGER_SCHEMA,
) -> dict[str, Any]:
    return {
        "schema": schema,
        "schedule_hash": schedule_hash,
        "proposal_generation": proposal_generation,
        "registered_design_hashes": registered_design_hashes,
        "committed_generation_allocations": committed_generation_allocations,
        "prior_state_hash": prior_state_hash,
        "cumulative_committed_alpha": cumulative_committed_alpha,
        "contains_outcomes": contains_outcomes,
        "f_q_authority": f_q_authority,
        "bce_oracle_authority": bce_oracle_authority,
        "mask_commit_authority": mask_commit_authority,
        "alpha_policy_kind": alpha_policy_kind,
        "authority_scope": authority_scope,
    }


def _expected_ledger_hash_for_generation(
    *,
    schedule: V1ProposalOrderAlphaSchedule,
    proposal_generation: int,
    registered_design_hashes: tuple[str, ...],
) -> str:
    """Replay the complete prefix chain and return its terminal state hash."""

    schedule_hash = schedule.assert_integrity()
    prior_hash: str | None = None
    current_hash = ""
    for generation in range(proposal_generation + 1):
        allocations = schedule.generation_alpha_allocations[:generation]
        payload = _ledger_hash_payload(
            schedule_hash=schedule_hash,
            proposal_generation=generation,
            registered_design_hashes=registered_design_hashes[:generation],
            committed_generation_allocations=allocations,
            prior_state_hash=prior_hash,
            cumulative_committed_alpha=math.fsum(allocations),
        )
        current_hash = content_hash(payload, prefix=_LEDGER_PREFIX)
        prior_hash = current_hash
    return current_hash


@dataclass(frozen=True, slots=True)
class V1ProposalOrderAlphaLedger:
    """Immutable head of one consecutive, non-recycling proposal-alpha chain.

    ``proposal_generation`` is both the number of registrations closed before
    outcomes and the generation index required from the next Serfling design.
    """

    schedule: V1ProposalOrderAlphaSchedule
    proposal_generation: int
    registered_design_hashes: tuple[str, ...]
    committed_generation_allocations: tuple[float, ...]
    prior_state_hash: str | None
    cumulative_committed_alpha: float
    _issuer_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    contains_outcomes: bool = False
    f_q_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    alpha_policy_kind: str = FIXED_TRAJECTORY_FWER_POLICY
    authority_scope: str = PROPOSAL_ORDER_ALPHA_AUTHORITY
    schema: str = PROPOSAL_ORDER_ALPHA_LEDGER_SCHEMA
    _ledger_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self._issuer_token is not _LEDGER_ISSUER_TOKEN:
            raise TypeError(
                "proposal alpha ledgers are issued only by initialize/advance/replay"
            )
        if type(self.schedule) is not V1ProposalOrderAlphaSchedule:
            raise TypeError("proposal alpha ledger requires the exact schedule type")
        self.schedule.assert_integrity()
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        if generation > self.schedule.maximum_proposal_generations:
            raise ValueError("proposal alpha ledger exceeds its frozen schedule")
        design_hashes = tuple(
            _identity(value, name=f"registered design hash {index}")
            for index, value in enumerate(self.registered_design_hashes)
        )
        if len(design_hashes) != generation:
            raise ValueError(
                "registered design hash ledger must be complete and consecutive"
            )
        if len(set(design_hashes)) != len(design_hashes):
            raise ValueError("a registered proposal design cannot be replayed")
        allocations = tuple(
            _finite_alpha(
                value,
                name=f"committed generation allocation {index}",
                positive=True,
            )
            for index, value in enumerate(self.committed_generation_allocations)
        )
        expected_allocations = self.schedule.generation_alpha_allocations[:generation]
        if allocations != expected_allocations:
            raise ValueError(
                "closed generations must commit their full scheduled allocations"
            )
        expected_cumulative = math.fsum(expected_allocations)
        supplied_cumulative = _finite_alpha(
            self.cumulative_committed_alpha,
            name="cumulative committed alpha",
            positive=False,
        )
        if not _alpha_matches(supplied_cumulative, expected_cumulative):
            raise ValueError("cumulative committed alpha is not canonical")
        expected_prior = (
            None
            if generation == 0
            else _expected_ledger_hash_for_generation(
                schedule=self.schedule,
                proposal_generation=generation - 1,
                registered_design_hashes=design_hashes[:-1],
            )
        )
        if generation == 0:
            if self.prior_state_hash is not None:
                raise ValueError(
                    "generation-zero alpha ledger cannot name a prior state"
                )
        else:
            prior = _identity(self.prior_state_hash, name="prior state hash")
            if prior != expected_prior:
                raise ValueError("proposal alpha ledger prior-state chain is invalid")
        for name in (
            "contains_outcomes",
            "f_q_authority",
            "bce_oracle_authority",
            "mask_commit_authority",
        ):
            _require_false(getattr(self, name), name=name)
        if self.alpha_policy_kind != FIXED_TRAJECTORY_FWER_POLICY:
            raise ValueError(
                "proposal alpha ledger requires fixed non-recycling trajectory FWER"
            )
        if self.authority_scope != PROPOSAL_ORDER_ALPHA_AUTHORITY:
            raise ValueError("proposal alpha ledger authority scope is invalid")
        if self.schema != PROPOSAL_ORDER_ALPHA_LEDGER_SCHEMA:
            raise ValueError("proposal alpha ledger schema is invalid")
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "registered_design_hashes", design_hashes)
        object.__setattr__(
            self,
            "committed_generation_allocations",
            expected_allocations,
        )
        object.__setattr__(self, "cumulative_committed_alpha", expected_cumulative)
        object.__setattr__(self, "_ledger_hash", self._compute_ledger_hash())

    @property
    def schedule_hash(self) -> str:
        return self.schedule.assert_integrity()

    @property
    def campaign_contract_hash(self) -> str:
        return self.schedule.campaign_contract_hash

    @property
    def controller_run_hash(self) -> str:
        return self.schedule.controller_run_hash

    @property
    def trajectory_alpha_ledger_hash(self) -> str:
        """The exact hash a new Serfling design must register."""

        return self.ledger_hash

    @property
    def trajectory_alpha_ledger_generation(self) -> int:
        return self.proposal_generation

    @property
    def trajectory_familywise_alpha(self) -> float:
        return self.schedule.trajectory_familywise_alpha

    @property
    def trajectory_alpha_committed_before_generation(self) -> float:
        return self.cumulative_committed_alpha

    @property
    def generation_alpha_allocation(self) -> float:
        if self.is_exhausted:
            raise RuntimeError("proposal alpha schedule is exhausted")
        return self.schedule.allocation_for_generation(self.proposal_generation)

    def next_design_alpha_coordinates(self) -> dict[str, str | int | float]:
        """Return the five ledger coordinates required by the next design."""

        return {
            "trajectory_alpha_ledger_hash": self.trajectory_alpha_ledger_hash,
            "trajectory_alpha_ledger_generation": (
                self.trajectory_alpha_ledger_generation
            ),
            "trajectory_familywise_alpha": self.trajectory_familywise_alpha,
            "trajectory_alpha_committed_before_generation": (
                self.trajectory_alpha_committed_before_generation
            ),
            "generation_alpha_allocation": self.generation_alpha_allocation,
        }

    @property
    def remaining_scheduled_alpha(self) -> float:
        return math.fsum(
            self.schedule.generation_alpha_allocations[self.proposal_generation :]
        )

    @property
    def remaining_trajectory_alpha(self) -> float:
        return (
            self.schedule.trajectory_familywise_alpha - self.cumulative_committed_alpha
        )

    @property
    def is_exhausted(self) -> bool:
        return self.proposal_generation == self.schedule.maximum_proposal_generations

    def _hash_payload(self) -> dict[str, Any]:
        return _ledger_hash_payload(
            schedule_hash=self.schedule.assert_integrity(),
            proposal_generation=self.proposal_generation,
            registered_design_hashes=self.registered_design_hashes,
            committed_generation_allocations=(self.committed_generation_allocations),
            prior_state_hash=self.prior_state_hash,
            cumulative_committed_alpha=self.cumulative_committed_alpha,
            contains_outcomes=self.contains_outcomes,
            f_q_authority=self.f_q_authority,
            bce_oracle_authority=self.bce_oracle_authority,
            mask_commit_authority=self.mask_commit_authority,
            alpha_policy_kind=self.alpha_policy_kind,
            authority_scope=self.authority_scope,
            schema=self.schema,
        )

    def _compute_ledger_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_LEDGER_PREFIX)

    @property
    def ledger_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        # The nested schedule is independently sealed.  The ledger's terminal
        # payload binds its complete design prefix, allocation prefix, and
        # recursively checked prior-state hash.
        self.schedule.assert_integrity()
        observed = self._compute_ledger_hash()
        if self._ledger_hash is not None and observed != self._ledger_hash:
            raise ValueError("proposal alpha ledger was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("proposal alpha ledger hash does not match")
        object.__setattr__(self, "_ledger_hash", observed)
        return observed

    def require_f_q_authority(self) -> None:
        raise PermissionError("proposal alpha ledger has no F/Q authority")

    def require_bce_oracle_authority(self) -> None:
        raise PermissionError("proposal alpha ledger has no BCE-oracle authority")

    def require_mask_commit_authority(self) -> None:
        raise PermissionError("proposal alpha ledger has no mask-commit authority")

    def _constructor_payload(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule,
            **self._state_constructor_payload(),
        }

    def _state_constructor_payload(self) -> dict[str, Any]:
        return {
            "proposal_generation": self.proposal_generation,
            "registered_design_hashes": self.registered_design_hashes,
            "committed_generation_allocations": (self.committed_generation_allocations),
            "prior_state_hash": self.prior_state_hash,
            "cumulative_committed_alpha": self.cumulative_committed_alpha,
            "contains_outcomes": self.contains_outcomes,
            "f_q_authority": self.f_q_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "alpha_policy_kind": self.alpha_policy_kind,
            "authority_scope": self.authority_scope,
            "schema": self.schema,
        }

    def __reduce__(self) -> tuple[Any, tuple[dict[str, Any], str]]:
        return _rebuild_ledger, (
            self._constructor_payload(),
            self.assert_integrity(),
        )


def initialize_v1_proposal_order_alpha_ledger(
    schedule: V1ProposalOrderAlphaSchedule,
) -> V1ProposalOrderAlphaLedger:
    """Return the unique generation-zero ledger head for ``schedule``."""

    if type(schedule) is not V1ProposalOrderAlphaSchedule:
        raise TypeError(
            "proposal alpha ledger initialization requires an exact schedule"
        )
    schedule.assert_integrity()
    return V1ProposalOrderAlphaLedger(
        _issuer_token=_LEDGER_ISSUER_TOKEN,
        schedule=schedule,
        proposal_generation=0,
        registered_design_hashes=(),
        committed_generation_allocations=(),
        prior_state_hash=None,
        cumulative_committed_alpha=0.0,
    )


def validate_v1_proposal_order_design(
    ledger: V1ProposalOrderAlphaLedger,
    design: V1GroupSequentialSerflingDesign,
    *,
    expected_ledger_hash: str | None = None,
    expected_design_hash: str | None = None,
) -> str:
    """Validate one design against the current immutable alpha-ledger head."""

    if type(ledger) is not V1ProposalOrderAlphaLedger:
        raise TypeError("proposal alpha validation requires the exact ledger type")
    if type(design) is not V1GroupSequentialSerflingDesign:
        raise TypeError("proposal alpha validation requires the exact design type")
    ledger_hash = ledger.assert_integrity(expected_ledger_hash)
    if ledger.is_exhausted:
        raise RuntimeError("proposal alpha schedule is exhausted")
    design_hash = design.assert_integrity(expected_design_hash)
    generation = ledger.proposal_generation
    expected_allocation = ledger.schedule.allocation_for_generation(generation)
    if design.proposal_generation != generation:
        raise ValueError("proposal design skips or replays a ledger generation")
    if design.trajectory_alpha_ledger_generation != generation:
        raise ValueError("proposal design names the wrong alpha-ledger generation")
    if design.trajectory_alpha_ledger_hash != ledger_hash:
        raise ValueError("proposal design names the wrong alpha-ledger head")
    if design.campaign_contract_hash != ledger.campaign_contract_hash:
        raise ValueError("proposal design names the wrong campaign contract")
    if (
        design.trajectory_familywise_alpha
        != ledger.schedule.trajectory_familywise_alpha
    ):
        raise ValueError("proposal design names the wrong trajectory FWER alpha")
    if not _alpha_matches(
        design.trajectory_alpha_committed_before_generation,
        ledger.cumulative_committed_alpha,
    ):
        raise ValueError("proposal design names the wrong committed-before alpha")
    if design.generation_alpha_allocation != expected_allocation:
        raise ValueError("proposal design names the wrong generation allocation")
    # The imported design permits a small arithmetic tolerance.  At this
    # production boundary the actual registered look probabilities must not
    # exceed the represented scheduled allocation at all.
    if math.fsum(design.alpha_by_look) > expected_allocation:
        raise ValueError("proposal design look alpha exceeds its scheduled allocation")
    if design.trajectory_alpha_policy_kind != FIXED_TRAJECTORY_FWER_POLICY:
        raise ValueError("proposal design does not use the registered FWER policy")
    return design_hash


def advance_v1_proposal_order_alpha_ledger(
    ledger: V1ProposalOrderAlphaLedger,
    design: V1GroupSequentialSerflingDesign,
    *,
    expected_ledger_hash: str | None = None,
    expected_design_hash: str | None = None,
) -> V1ProposalOrderAlphaLedger:
    """Register a design pre-outcome and charge its full generation allocation.

    The returned state must become the controller run's unique persisted head
    before any row covered by ``design`` is observed.  The pure object cannot
    perform that external compare-and-swap itself.
    """

    design_hash = validate_v1_proposal_order_design(
        ledger,
        design,
        expected_ledger_hash=expected_ledger_hash,
        expected_design_hash=expected_design_hash,
    )
    ledger_hash = ledger.assert_integrity(expected_ledger_hash)
    generation = ledger.proposal_generation
    allocation = ledger.schedule.allocation_for_generation(generation)
    committed = ledger.committed_generation_allocations + (allocation,)
    return V1ProposalOrderAlphaLedger(
        _issuer_token=_LEDGER_ISSUER_TOKEN,
        schedule=ledger.schedule,
        proposal_generation=generation + 1,
        registered_design_hashes=ledger.registered_design_hashes + (design_hash,),
        committed_generation_allocations=committed,
        prior_state_hash=ledger_hash,
        cumulative_committed_alpha=math.fsum(committed),
    )


def _rebuild_schedule(
    payload: dict[str, Any],
    expected_hash: str,
) -> V1ProposalOrderAlphaSchedule:
    result = V1ProposalOrderAlphaSchedule(**payload)
    result.assert_integrity(expected_hash)
    return result


def _rebuild_ledger(
    payload: dict[str, Any],
    expected_hash: str,
) -> V1ProposalOrderAlphaLedger:
    result = V1ProposalOrderAlphaLedger(
        _issuer_token=_LEDGER_ISSUER_TOKEN,
        **payload,
    )
    result.assert_integrity(expected_hash)
    return result


def export_v1_proposal_order_alpha_ledger_replay(
    ledger: V1ProposalOrderAlphaLedger,
) -> dict[str, Any]:
    """Export one small, JSON-compatible, content-addressed replay envelope."""

    if type(ledger) is not V1ProposalOrderAlphaLedger:
        raise TypeError("proposal alpha replay export requires the exact ledger type")
    ledger_hash = ledger.assert_integrity()
    schedule_hash = ledger.schedule.assert_integrity()
    return {
        "schema": PROPOSAL_ORDER_ALPHA_REPLAY_SCHEMA,
        "schedule_payload": ledger.schedule._constructor_payload(),
        "schedule_hash": schedule_hash,
        "ledger_payload": ledger._state_constructor_payload(),
        "ledger_hash": ledger_hash,
    }


def restore_v1_proposal_order_alpha_ledger_replay(
    envelope: Mapping[str, Any],
) -> V1ProposalOrderAlphaLedger:
    """Restore and verify an exported ledger without pickle or private tokens."""

    if not isinstance(envelope, Mapping):
        raise TypeError("proposal alpha replay envelope must be one mapping")
    payload = dict(envelope)
    expected_keys = {
        "schema",
        "schedule_payload",
        "schedule_hash",
        "ledger_payload",
        "ledger_hash",
    }
    if set(payload) != expected_keys:
        raise ValueError("proposal alpha replay envelope fields are invalid")
    if payload["schema"] != PROPOSAL_ORDER_ALPHA_REPLAY_SCHEMA:
        raise ValueError("proposal alpha replay envelope schema is invalid")
    if not isinstance(payload["schedule_payload"], Mapping):
        raise TypeError("proposal alpha replay schedule payload must be one mapping")
    if not isinstance(payload["ledger_payload"], Mapping):
        raise TypeError("proposal alpha replay ledger payload must be one mapping")
    schedule_hash = _identity(payload["schedule_hash"], name="schedule hash")
    ledger_hash = _identity(payload["ledger_hash"], name="ledger hash")
    schedule = _rebuild_schedule(dict(payload["schedule_payload"]), schedule_hash)
    ledger_payload = dict(payload["ledger_payload"])
    if "schedule" in ledger_payload or "_issuer_token" in ledger_payload:
        raise ValueError("proposal alpha replay ledger payload crossed its boundary")
    return _rebuild_ledger(
        {"schedule": schedule, **ledger_payload},
        ledger_hash,
    )


# Short aliases keep downstream wiring readable without creating a second type.
V1ProposalAlphaSchedule = V1ProposalOrderAlphaSchedule
V1ProposalAlphaLedger = V1ProposalOrderAlphaLedger
initialize_v1_proposal_alpha_ledger = initialize_v1_proposal_order_alpha_ledger
validate_v1_proposal_alpha_design = validate_v1_proposal_order_design
advance_v1_proposal_alpha_ledger = advance_v1_proposal_order_alpha_ledger


__all__ = [
    "EXPLICIT_PREOUTCOME_ALLOCATION_POLICY",
    "PROPOSAL_ORDER_ALPHA_AUTHORITY",
    "PROPOSAL_ORDER_ALPHA_LEDGER_SCHEMA",
    "PROPOSAL_ORDER_ALPHA_REPLAY_SCHEMA",
    "PROPOSAL_ORDER_ALPHA_SCHEDULE_SCHEMA",
    "V1ProposalAlphaLedger",
    "V1ProposalAlphaSchedule",
    "V1ProposalOrderAlphaLedger",
    "V1ProposalOrderAlphaSchedule",
    "advance_v1_proposal_alpha_ledger",
    "advance_v1_proposal_order_alpha_ledger",
    "export_v1_proposal_order_alpha_ledger_replay",
    "initialize_v1_proposal_alpha_ledger",
    "initialize_v1_proposal_order_alpha_ledger",
    "restore_v1_proposal_order_alpha_ledger_replay",
    "validate_v1_proposal_alpha_design",
    "validate_v1_proposal_order_design",
]
