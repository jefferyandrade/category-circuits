"""Production source-bound scalar witness for full-population signed work.

The complete-union V4 receipt is a raw lower enclosure of the shared-action
support slope.  It is not itself an observation: the D-fixed and residual
preferred-plane deficits still have to be charged.  This module performs that
last deterministic step from live, issuer-owned source receipts and then runs
the registered without-replacement confidence sequence over the sealed equal
block order.

The authority boundary is intentionally narrow.  A stable V2 decision proves
only strict full-population corrected-work separation of the preferred hard
action from the registered complete alternate domain.  Q safety, action
invariance, C/K/O, commit, terminal, performance, and Pareto authority remain
closed and require their own exact typed joins.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Final, NoReturn, Sequence

from mechanistic_systems.evidence.finite_population import BoundedWoRBettingCS

from .contracts import content_hash
from .modern_v1_signed_work_block_support_complete_union_production import (
    V1BlockSupportCompleteConstraintCoverBindingReceiptV3,
    V1BlockSupportCompleteUnionRawSlopeReceiptV4,
    require_v1_block_support_complete_union_v4_canonical_observation_raw_union_authority,
)
from .modern_v1_signed_work_block_support_domain_binding import (
    V1BlockSupportBoundWitnessSource,
    V1BlockSupportCanonicalBlockContextIdentity,
    require_v1_block_support_canonical_context_identity,
)
from .modern_v1_signed_work_block_support_source_cuda import (
    V1SignedWorkBlockSupportExecutionMode,
    V1SignedWorkBlockSupportSourceRole,
)
from .modern_v1_signed_work_block_support_witness import (
    V1BlockSupportWitnessReason,
    V1BlockSupportWitnessStatus,
)
from .modern_v1_signed_work_physical_alternate_domain_cuda import (
    V1SignedWorkPhysicalAlternateBlockSupportBinding,
)


PRODUCTION_BLOCK_OBSERVATION_V2_SCHEMA: Final = (
    "partition_free_v1_block_support_source_bound_margin_observation_v2"
)
PRODUCTION_BLOCK_STATE_V2_SCHEMA: Final = (
    "partition_free_v1_block_support_source_bound_state_v2"
)
PRODUCTION_BLOCK_DECISION_V2_SCHEMA: Final = (
    "partition_free_v1_block_support_source_bound_decision_v2"
)
PRODUCTION_BLOCK_OBSERVATION_V2_RULE: Final = (
    "source_native_complete_raw_union_minus_exact_registered_D_and_residual_"
    "preferred_plane_deficits_once_with_outward_binary64_enclosure_v2"
)
PRODUCTION_BLOCK_DECISION_V2_RULE: Final = (
    "sealed_equal_block_without_replacement_confidence_sequence_or_exact_"
    "binary64_rational_census_over_source_bound_margin_lower_observations_v2"
)
PRODUCTION_BLOCK_DECISION_V2_AUTHORITY: Final = (
    "strict_full_population_corrected_F_separation_only_no_Q_action_"
    "invariance_C_K_O_commit_terminal_performance_or_pareto_authority_v2"
)

_PREFIX: Final = "partition-free-v1-block-support-production-witness-"
_OBSERVATION_TOKEN: Final = object()
_DECISION_TOKEN: Final = object()
_EMPTY_CHAIN_HASH: Final = content_hash(
    {"schema": "partition_free_v1_block_support_production_empty_chain_v2"},
    prefix=_PREFIX + "chain-",
)


class V1BlockSupportProductionWitnessRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    SOURCE_MISMATCH = "source_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    CUDA_CUSTODY_REQUIRED = "cuda_custody_required"
    INCOMPLETE_COMPLETE_UNION = "incomplete_complete_union"
    OUT_OF_ORDER_BLOCK = "out_of_order_block"
    INVALID_RANGE = "invalid_range"
    NUMERICAL_REFUSAL = "numerical_refusal"
    RECEIPT_TAMPERING = "receipt_tampering"
    AUTHORITY_VIOLATION = "authority_violation"
    STRICT_SEPARATION_REQUIRED = "strict_separation_required"


class V1BlockSupportProductionWitnessRefusal(RuntimeError):
    def __init__(
        self,
        code: V1BlockSupportProductionWitnessRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1BlockSupportProductionWitnessRefusalCode:
            raise TypeError("production witness refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1BlockSupportProductionWitnessRefusalCode,
    message: str,
) -> NoReturn:
    raise V1BlockSupportProductionWitnessRefusal(code, message)


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one exact integer",
        )
    result = int(value)
    if result < minimum:
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be at least {minimum}",
        )
    return result


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be finite",
        )
    return result


def _nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be nonnegative",
        )
    return result


def _fraction_up(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.NUMERICAL_REFUSAL,
            "exact upper enclosure does not fit binary64",
        )
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result) or Fraction.from_float(result) < value:
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.NUMERICAL_REFUSAL,
            "outward upper conversion failed",
        )
    return result


def _fraction_down(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.NUMERICAL_REFUSAL,
            "exact lower enclosure does not fit binary64",
        )
    if Fraction.from_float(result) > value:
        result = math.nextafter(result, -math.inf)
    if not math.isfinite(result) or Fraction.from_float(result) > value:
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.NUMERICAL_REFUSAL,
            "outward lower conversion failed",
        )
    return result


def _payload(value: object, *, hash_field: str) -> dict[str, object]:
    return {
        item.name: (
            getattr(value, item.name).value
            if isinstance(getattr(value, item.name), Enum)
            else getattr(value, item.name)
        )
        for item in fields(value)
        if not item.name.startswith("_") and item.name != hash_field
    }


def _closed(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool or value for value in values):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} attempted to open downstream authority",
        )


@dataclass(frozen=True, slots=True)
class V1BlockSupportProductionMarginLowerBoundObservationV2:
    """One source-bound, deficit-complete lower-margin observation."""

    bound_registration_receipt_hash: str
    legacy_registration_hash: str
    complete_cover_v3_receipt_hash: str
    complete_union_raw_v4_receipt_hash: str
    canonical_context_receipt_hash: str
    physical_block_support_binding_receipt_hash: str
    d_fixed_support_receipt_hash: str
    residual_support_receipt_hash: str
    block_index: int
    block_id: str
    block_context_axis_hash: str
    block_context_count: int
    sealed_block_order_hash: str
    complete_alternate_domain_hash: str
    alias_axis_hash: str
    preferred_action_hash: str
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
    corrected_evidence_source_hash: str
    complete_union_raw_lower_outward: float
    d_preferred_plane_deficit_upper: float
    residual_preferred_plane_deficit_upper: float
    d_weight_numerator: int
    residual_weight_numerator: int
    population_weight_denominator: int
    weighted_preferred_plane_deficit_upper_outward: float
    raw_margin_lower_outward: float
    margin_lower_bound: float
    observation_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    production_source_bound_complete_union: bool = True
    exact_D_and_residual_deficit_debit_once: bool = True
    canonical_block_observation_authority: bool = True
    population_separation_authority: bool = False
    q_authority: bool = False
    action_invariance_authority: bool = False
    c_authority: bool = False
    k_authority: bool = False
    o_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    array_free_receipt: bool = True
    rule: str = PRODUCTION_BLOCK_OBSERVATION_V2_RULE
    schema: str = PRODUCTION_BLOCK_OBSERVATION_V2_SCHEMA
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _OBSERVATION_TOKEN:
            raise TypeError("production block observations are issuer-only")
        for item in fields(self):
            if (
                not item.name.startswith("_")
                and item.name.endswith("_hash")
                and item.name != "observation_hash"
            ):
                object.__setattr__(
                    self,
                    item.name,
                    _text(getattr(self, item.name), name=item.name),
                )
        for name in ("block_id", "rule", "schema"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name=name),
            )
        for name in (
            "block_index",
            "proposal_generation",
            "d_weight_numerator",
            "residual_weight_numerator",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name),
            )
        for name in (
            "block_context_count",
            "population_weight_denominator",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name=name, minimum=1),
            )
        if (
            self.d_weight_numerator + self.residual_weight_numerator
            != self.population_weight_denominator
        ):
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.SOURCE_MISMATCH,
                "observation population weights do not close exactly",
            )
        for name in (
            "complete_union_raw_lower_outward",
            "raw_margin_lower_outward",
            "margin_lower_bound",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name=name),
            )
        for name in (
            "d_preferred_plane_deficit_upper",
            "residual_preferred_plane_deficit_upper",
            "weighted_preferred_plane_deficit_upper_outward",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name=name),
            )
        expected_deficit = _fraction_up(
            Fraction(self.d_weight_numerator, self.population_weight_denominator)
            * Fraction.from_float(self.d_preferred_plane_deficit_upper)
            + Fraction(
                self.residual_weight_numerator,
                self.population_weight_denominator,
            )
            * Fraction.from_float(self.residual_preferred_plane_deficit_upper)
        )
        expected_raw_margin = _fraction_down(
            Fraction.from_float(self.complete_union_raw_lower_outward)
            - Fraction.from_float(expected_deficit)
        )
        if (
            self.weighted_preferred_plane_deficit_upper_outward != expected_deficit
            or self.raw_margin_lower_outward != expected_raw_margin
            or self.margin_lower_bound < self.raw_margin_lower_outward
        ):
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.NUMERICAL_REFUSAL,
                "production observation deficit debit or outward margin does not replay",
            )
        required = (
            self.production_source_bound_complete_union,
            self.exact_D_and_residual_deficit_debit_once,
            self.canonical_block_observation_authority,
            self.array_free_receipt,
        )
        if any(type(value) is not bool or value is not True for value in required):
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.AUTHORITY_VIOLATION,
                "production observation lost a mandatory invariant",
            )
        _closed(
            (
                self.population_separation_authority,
                self.q_authority,
                self.action_invariance_authority,
                self.c_authority,
                self.k_authority,
                self.o_authority,
                self.commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="production block observation",
        )
        if (
            self.rule != PRODUCTION_BLOCK_OBSERVATION_V2_RULE
            or self.schema != PRODUCTION_BLOCK_OBSERVATION_V2_SCHEMA
        ):
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.AUTHORITY_VIOLATION,
                "unsupported production observation semantics",
            )
        observed = content_hash(
            _payload(self, hash_field="observation_hash"),
            prefix=_PREFIX + "observation-integrity-",
        )
        if self._integrity_seal and self._integrity_seal != observed:
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.RECEIPT_TAMPERING,
                "production observation was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed)
        object.__setattr__(
            self,
            "observation_hash",
            content_hash(
                _payload(self, hash_field="observation_hash"),
                prefix=_PREFIX + "observation-",
            ),
        )

    def assert_integrity(self) -> str:
        payload = _payload(self, hash_field="observation_hash")
        observed = content_hash(payload, prefix=_PREFIX + "observation-integrity-")
        expected = content_hash(payload, prefix=_PREFIX + "observation-")
        if observed != self._integrity_seal or expected != self.observation_hash:
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.RECEIPT_TAMPERING,
                "production observation no longer replays",
            )
        return expected


@dataclass(frozen=True, slots=True)
class V1BlockSupportProductionWitnessStateV2:
    bound_registration_receipt_hash: str
    legacy_registration_hash: str
    observed_block_count: int
    last_observation_hash: str | None
    receipt_chain_hash: str
    state_hash: str = field(init=False)
    schema: str = PRODUCTION_BLOCK_STATE_V2_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bound_registration_receipt_hash",
            _text(
                self.bound_registration_receipt_hash,
                name="bound registration receipt hash",
            ),
        )
        object.__setattr__(
            self,
            "legacy_registration_hash",
            _text(self.legacy_registration_hash, name="legacy registration hash"),
        )
        count = _integer(
            self.observed_block_count,
            name="observed block count",
        )
        object.__setattr__(self, "observed_block_count", count)
        if count == 0:
            if (
                self.last_observation_hash is not None
                or self.receipt_chain_hash != _EMPTY_CHAIN_HASH
            ):
                _refuse(
                    V1BlockSupportProductionWitnessRefusalCode.RECEIPT_TAMPERING,
                    "empty production state has a nonempty chain",
                )
        else:
            object.__setattr__(
                self,
                "last_observation_hash",
                _text(self.last_observation_hash, name="last observation hash"),
            )
            object.__setattr__(
                self,
                "receipt_chain_hash",
                _text(self.receipt_chain_hash, name="receipt chain hash"),
            )
        if self.schema != PRODUCTION_BLOCK_STATE_V2_SCHEMA:
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.AUTHORITY_VIOLATION,
                "unsupported production state schema",
            )
        object.__setattr__(self, "state_hash", self._compute_hash())

    def _compute_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "bound_registration_receipt_hash": (
                    self.bound_registration_receipt_hash
                ),
                "legacy_registration_hash": self.legacy_registration_hash,
                "observed_block_count": self.observed_block_count,
                "last_observation_hash": self.last_observation_hash,
                "receipt_chain_hash": self.receipt_chain_hash,
            },
            prefix=_PREFIX + "state-",
        )

    def assert_integrity(self) -> str:
        expected = self._compute_hash()
        if self.state_hash != expected:
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.RECEIPT_TAMPERING,
                "production state no longer replays",
            )
        return expected


@dataclass(frozen=True, slots=True)
class V1BlockSupportProductionWitnessDecisionV2:
    bound_registration_receipt_hash: str
    legacy_registration_hash: str
    preoutcome_domain_receipt_hash: str
    complete_alternate_domain_hash: str
    complete_alternate_domain_receipt_hash: str
    alternate_action_count: int
    preferred_action_hash: str
    alias_axis_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    corrected_evidence_source_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    state_hash: str
    receipt_chain_hash: str
    status: V1BlockSupportWitnessStatus
    reason: V1BlockSupportWitnessReason
    observed_block_count: int
    block_population_count: int
    confidence_lower: float
    confidence_upper: float
    sample_mean_of_downward_bounds: float | None
    full_census: bool
    exact_census_sum_numerator: int | None
    exact_census_sum_denominator: int | None
    decision_hash: str = field(init=False)
    _issuer_token: InitVar[object] = None
    production_source_bound_witness: bool = True
    source_native_decision_identity_bound: bool = True
    sampled_full_population_F_strict_separation_authority: bool = False
    q_authority: bool = False
    action_invariance_authority: bool = False
    c_authority: bool = False
    k_authority: bool = False
    o_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    performance_authority: bool = False
    pareto_authority: bool = False
    alpha_allocation_nonrecycling: bool = True
    array_free_receipt: bool = True
    rule: str = PRODUCTION_BLOCK_DECISION_V2_RULE
    authority_scope: str = PRODUCTION_BLOCK_DECISION_V2_AUTHORITY
    schema: str = PRODUCTION_BLOCK_DECISION_V2_SCHEMA
    _integrity_seal: str = field(default="", repr=False, compare=False)

    def __post_init__(self, _issuer_token: object) -> None:
        if _issuer_token is not _DECISION_TOKEN:
            raise TypeError("production block decisions are issuer-only")
        for name in (
            "bound_registration_receipt_hash",
            "legacy_registration_hash",
            "preoutcome_domain_receipt_hash",
            "complete_alternate_domain_hash",
            "complete_alternate_domain_receipt_hash",
            "preferred_action_hash",
            "alias_axis_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "corrected_evidence_source_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "state_hash",
            "receipt_chain_hash",
            "rule",
            "authority_scope",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name=name),
            )
        if (
            type(self.status) is not V1BlockSupportWitnessStatus
            or type(self.reason) is not V1BlockSupportWitnessReason
        ):
            raise TypeError("production decision status/reason enums must be exact")
        observed = _integer(
            self.observed_block_count,
            name="observed block count",
        )
        population = _integer(
            self.block_population_count,
            name="block population count",
            minimum=1,
        )
        if observed > population:
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.RECEIPT_TAMPERING,
                "production decision observes more blocks than exist",
            )
        object.__setattr__(self, "observed_block_count", observed)
        object.__setattr__(self, "block_population_count", population)
        object.__setattr__(
            self,
            "alternate_action_count",
            _integer(
                self.alternate_action_count,
                name="alternate action count",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _integer(self.proposal_generation, name="proposal generation"),
        )
        object.__setattr__(
            self,
            "route_count",
            _integer(self.route_count, name="route count", minimum=1),
        )
        lower = _finite(self.confidence_lower, name="confidence lower")
        upper = _finite(self.confidence_upper, name="confidence upper")
        if lower > upper:
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.NUMERICAL_REFUSAL,
                "production confidence bounds are inverted",
            )
        object.__setattr__(self, "confidence_lower", lower)
        object.__setattr__(self, "confidence_upper", upper)
        if self.sample_mean_of_downward_bounds is not None:
            object.__setattr__(
                self,
                "sample_mean_of_downward_bounds",
                _finite(
                    self.sample_mean_of_downward_bounds,
                    name="sample mean",
                ),
            )
        if type(self.full_census) is not bool or self.full_census != (
            observed == population
        ):
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.RECEIPT_TAMPERING,
                "production decision census marker differs from counts",
            )
        if self.full_census:
            if (
                isinstance(self.exact_census_sum_numerator, bool)
                or not isinstance(self.exact_census_sum_numerator, int)
                or isinstance(self.exact_census_sum_denominator, bool)
                or not isinstance(self.exact_census_sum_denominator, int)
                or self.exact_census_sum_denominator <= 0
            ):
                _refuse(
                    V1BlockSupportProductionWitnessRefusalCode.RECEIPT_TAMPERING,
                    "full census must retain its exact rational sum",
                )
        elif (
            self.exact_census_sum_numerator is not None
            or self.exact_census_sum_denominator is not None
        ):
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.RECEIPT_TAMPERING,
                "partial decision cannot carry a census sum",
            )
        stable = self.status is V1BlockSupportWitnessStatus.STABLE_STRICT_SEPARATION
        if self.sampled_full_population_F_strict_separation_authority is not stable:
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.AUTHORITY_VIOLATION,
                "strict-F authority must equal the stable decision status",
            )
        if (
            stable
            and lower <= 0.0
            and not (
                self.full_census
                and self.exact_census_sum_numerator is not None
                and self.exact_census_sum_numerator > 0
            )
        ):
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.AUTHORITY_VIOLATION,
                "stable decision lacks a positive confidence or census result",
            )
        if (
            self.production_source_bound_witness is not True
            or self.source_native_decision_identity_bound is not True
            or self.alpha_allocation_nonrecycling is not True
            or self.array_free_receipt is not True
        ):
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.AUTHORITY_VIOLATION,
                "production decision weakened source/alpha/compact guarantees",
            )
        _closed(
            (
                self.q_authority,
                self.action_invariance_authority,
                self.c_authority,
                self.k_authority,
                self.o_authority,
                self.commit_authority,
                self.terminal_authority,
                self.performance_authority,
                self.pareto_authority,
            ),
            owner="production block decision",
        )
        if (
            self.rule != PRODUCTION_BLOCK_DECISION_V2_RULE
            or self.authority_scope != PRODUCTION_BLOCK_DECISION_V2_AUTHORITY
            or self.schema != PRODUCTION_BLOCK_DECISION_V2_SCHEMA
        ):
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.AUTHORITY_VIOLATION,
                "unsupported production decision semantics",
            )
        observed_hash = content_hash(
            _payload(self, hash_field="decision_hash"),
            prefix=_PREFIX + "decision-integrity-",
        )
        if self._integrity_seal and self._integrity_seal != observed_hash:
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.RECEIPT_TAMPERING,
                "production decision was mutated",
            )
        object.__setattr__(self, "_integrity_seal", observed_hash)
        object.__setattr__(
            self,
            "decision_hash",
            content_hash(
                _payload(self, hash_field="decision_hash"),
                prefix=_PREFIX + "decision-",
            ),
        )

    def assert_integrity(self) -> str:
        payload = _payload(self, hash_field="decision_hash")
        observed = content_hash(payload, prefix=_PREFIX + "decision-integrity-")
        expected = content_hash(payload, prefix=_PREFIX + "decision-")
        if observed != self._integrity_seal or expected != self.decision_hash:
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.RECEIPT_TAMPERING,
                "production decision no longer replays",
            )
        return expected


def _require_source_ready(
    source: V1BlockSupportBoundWitnessSource,
) -> V1BlockSupportBoundWitnessSource:
    if type(source) is not V1BlockSupportBoundWitnessSource:
        raise TypeError("production witness requires the exact bound source")
    try:
        source.assert_live()
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.STALE_SOURCE,
            f"bound source is stale: {error}",
        )
    if (
        source.receipt.production_cuda_source_bound is not True
        or source.receipt.source_derived_registration_path is not True
        or source.preoutcome_domain.receipt.production_cuda_source_bound is not True
        or source.preoutcome_domain.receipt.private_reference_test_only is not False
    ):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.CUDA_CUSTODY_REQUIRED,
            "production witness requires source-derived CUDA pre-outcome custody",
        )
    return source


def issue_v1_signed_work_block_support_production_observation_v2(
    *,
    bound_witness_source: V1BlockSupportBoundWitnessSource,
    canonical_context_identity: V1BlockSupportCanonicalBlockContextIdentity,
    physical_binding: V1SignedWorkPhysicalAlternateBlockSupportBinding,
    complete_cover: V1BlockSupportCompleteConstraintCoverBindingReceiptV3,
    complete_union_raw: V1BlockSupportCompleteUnionRawSlopeReceiptV4,
) -> V1BlockSupportProductionMarginLowerBoundObservationV2:
    """Issue one deficit-complete observation without caller numerical inputs."""

    source = _require_source_ready(bound_witness_source)
    if (
        type(canonical_context_identity)
        is not V1BlockSupportCanonicalBlockContextIdentity
        or type(physical_binding)
        is not V1SignedWorkPhysicalAlternateBlockSupportBinding
        or type(complete_cover)
        is not V1BlockSupportCompleteConstraintCoverBindingReceiptV3
        or type(complete_union_raw) is not V1BlockSupportCompleteUnionRawSlopeReceiptV4
    ):
        raise TypeError("production observation requires exact live/receipt types")
    try:
        identity = require_v1_block_support_canonical_context_identity(
            source=source,
            identity=canonical_context_identity,
        )
        physical_binding.assert_live()
        complete_cover.assert_integrity()
        require_v1_block_support_complete_union_v4_canonical_observation_raw_union_authority(
            complete_union_raw
        )
        d_receipt = physical_binding.d_fixed_support.receipt
        residual_receipt = physical_binding.residual_support.receipt
        d_receipt.assert_integrity()
        residual_receipt.assert_integrity()
    except (RuntimeError, PermissionError, TypeError, ValueError) as error:
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.STALE_SOURCE,
            f"one production observation parent is stale: {error}",
        )
    registration = source.registration
    pre = source.preoutcome_domain.receipt
    physical = physical_binding.receipt
    if (
        complete_cover.bound_registration_receipt_hash != source.receipt.receipt_hash
        or complete_cover.preoutcome_domain_receipt_hash != pre.receipt_hash
        or complete_cover.canonical_block_context_receipt_hash != identity.receipt_hash
        or complete_cover.canonical_context_tuple_hash
        != identity.canonical_context_tuple_hash
        or complete_cover.block_support_context_axis_hash
        != identity.block_support_context_axis_hash
        or complete_cover.block_index != identity.block_index
        or complete_cover.block_id != identity.sealed_source_block_id
        or complete_union_raw.complete_cover_v3_receipt_hash
        != complete_cover.receipt_hash
    ):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.CONTEXT_MISMATCH,
            "complete cover/raw union names another bound block",
        )
    if (
        complete_cover.complete_constraint_cover_source_bound is not True
        or complete_cover.production_cuda_custody is not True
        or complete_cover.canonical_block_observation_authority is not False
        or complete_union_raw.canonical_block_observation_raw_union_authority
        is not True
        or complete_union_raw.canonical_block_observation_authority is not False
        or complete_union_raw.d_plane_deficit_debited is not False
        or complete_union_raw.residual_plane_deficit_debited is not False
    ):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.INCOMPLETE_COMPLETE_UNION,
            "complete cover/raw receipt crossed or missed its raw-union boundary",
        )
    if (
        complete_cover.physical_block_support_binding_receipt_hash
        != physical.receipt_hash
        or complete_cover.d_fixed_support_receipt_hash != d_receipt.receipt_hash
        or complete_cover.residual_support_receipt_hash != residual_receipt.receipt_hash
        or physical.d_fixed_support_receipt_hash != d_receipt.receipt_hash
        or physical.residual_support_receipt_hash != residual_receipt.receipt_hash
        or d_receipt.source_role is not V1SignedWorkBlockSupportSourceRole.D_FIXED
        or residual_receipt.source_role
        is not V1SignedWorkBlockSupportSourceRole.RESIDUAL_BLOCK
        or d_receipt.execution_mode
        is not V1SignedWorkBlockSupportExecutionMode.CUDA_PRODUCTION
        or residual_receipt.execution_mode
        is not V1SignedWorkBlockSupportExecutionMode.CUDA_PRODUCTION
        or not physical.resident_device_identity.startswith("cuda:")
    ):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.CUDA_CUSTODY_REQUIRED,
            "live D/residual sources do not equal the complete CUDA cover parents",
        )
    if (
        d_receipt.corrected_evidence_source_hash
        != residual_receipt.corrected_evidence_source_hash
    ):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.SOURCE_MISMATCH,
            "D and residual sources name different corrected-evidence streams",
        )
    if (
        residual_receipt.context_axis_hash != identity.block_support_context_axis_hash
        or residual_receipt.context_count != len(identity.context_ids)
        or residual_receipt.block_ordinal != identity.block_index
        or residual_receipt.context_count != registration.block_size
        or d_receipt.context_count != registration.d_union_count
        or d_receipt.block_partition_hash != residual_receipt.block_partition_hash
        or physical.d_fixed_block_partition_hash
        != residual_receipt.block_partition_hash
        or physical.residual_block_partition_hash
        != residual_receipt.block_partition_hash
    ):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.CONTEXT_MISMATCH,
            "D/residual source population or sealed block identity differs",
        )
    common = (
        "model_identity_hash",
        "chart_identity_hash",
        "current_mask_hash",
        "proposal_generation",
        "route_axis_hash",
        "route_id_axis_hash",
        "native_fiber_axis_receipt_hash",
        "native_fiber_measure_hash",
        "registered_d_prefix_hash",
        "registered_d_w2_prefix_hash",
    )
    if any(
        getattr(complete_cover, name) != getattr(physical, name)
        or getattr(complete_cover, name) != getattr(d_receipt, name)
        or getattr(complete_cover, name) != getattr(residual_receipt, name)
        for name in common
    ):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.SOURCE_MISMATCH,
            "model/chart/mask/route/native-fiber/D-prefix custody differs",
        )
    if (
        complete_cover.preferred_action_hash != registration.preferred_action_hash
        or complete_cover.preferred_action_hash != physical.preferred_action_hash
        or complete_cover.preferred_action_binding_hash
        != physical.preferred_action_binding_hash
        or complete_cover.preferred_selected_count
        != registration.preferred_action_cardinality
        or complete_cover.ordered_alias_binding_hashes
        != tuple(
            value.binding_hash for value in source.preoutcome_domain.alias_bindings
        )
        or complete_union_raw.ordered_alias_binding_hashes
        != complete_cover.ordered_alias_binding_hashes
    ):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.SOURCE_MISMATCH,
            "preferred P or complete registered alias axis differs",
        )
    d_deficit = _nonnegative(
        d_receipt.preferred_plane_deficit_upper,
        name="D preferred-plane deficit upper",
    )
    residual_deficit = _nonnegative(
        residual_receipt.preferred_plane_deficit_upper,
        name="residual preferred-plane deficit upper",
    )
    weighted_fraction = Fraction(
        registration.d_weight_numerator,
        registration.population_weight_denominator,
    ) * Fraction.from_float(d_deficit) + Fraction(
        registration.residual_weight_numerator,
        registration.population_weight_denominator,
    ) * Fraction.from_float(
        residual_deficit
    )
    weighted = _fraction_up(weighted_fraction)
    raw_margin = _fraction_down(
        Fraction.from_float(complete_union_raw.complete_union_raw_lower_outward)
        - Fraction.from_float(weighted)
    )
    if raw_margin > registration.analytic_margin_upper_bound:
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.INVALID_RANGE,
            "source-bound lower margin exceeds the registered analytic upper bound",
        )
    clipped = max(raw_margin, registration.analytic_margin_lower_bound)
    return V1BlockSupportProductionMarginLowerBoundObservationV2(
        bound_registration_receipt_hash=source.receipt.receipt_hash,
        legacy_registration_hash=registration.registration_hash,
        complete_cover_v3_receipt_hash=complete_cover.receipt_hash,
        complete_union_raw_v4_receipt_hash=complete_union_raw.receipt_hash,
        canonical_context_receipt_hash=identity.receipt_hash,
        physical_block_support_binding_receipt_hash=physical.receipt_hash,
        d_fixed_support_receipt_hash=d_receipt.receipt_hash,
        residual_support_receipt_hash=residual_receipt.receipt_hash,
        block_index=identity.block_index,
        block_id=identity.sealed_source_block_id,
        block_context_axis_hash=identity.block_support_context_axis_hash,
        block_context_count=len(identity.context_ids),
        sealed_block_order_hash=registration.sealed_block_order_hash,
        complete_alternate_domain_hash=registration.complete_alternate_domain_hash,
        alias_axis_hash=source.receipt.alias_axis_hash,
        preferred_action_hash=registration.preferred_action_hash,
        model_identity_hash=complete_cover.model_identity_hash,
        chart_identity_hash=complete_cover.chart_identity_hash,
        current_mask_hash=complete_cover.current_mask_hash,
        proposal_generation=complete_cover.proposal_generation,
        route_axis_hash=complete_cover.route_axis_hash,
        route_id_axis_hash=complete_cover.route_id_axis_hash,
        native_fiber_axis_receipt_hash=(complete_cover.native_fiber_axis_receipt_hash),
        native_fiber_measure_hash=complete_cover.native_fiber_measure_hash,
        registered_d_prefix_hash=complete_cover.registered_d_prefix_hash,
        registered_d_w2_prefix_hash=complete_cover.registered_d_w2_prefix_hash,
        corrected_evidence_source_hash=(d_receipt.corrected_evidence_source_hash),
        complete_union_raw_lower_outward=(
            complete_union_raw.complete_union_raw_lower_outward
        ),
        d_preferred_plane_deficit_upper=d_deficit,
        residual_preferred_plane_deficit_upper=residual_deficit,
        d_weight_numerator=registration.d_weight_numerator,
        residual_weight_numerator=registration.residual_weight_numerator,
        population_weight_denominator=registration.population_weight_denominator,
        weighted_preferred_plane_deficit_upper_outward=weighted,
        raw_margin_lower_outward=raw_margin,
        margin_lower_bound=clipped,
        _issuer_token=_OBSERVATION_TOKEN,
    )


class V1BlockSupportProductionWitnessCertifier:
    """Append-only controller over exact source-bound block observations."""

    def __init__(self, source: V1BlockSupportBoundWitnessSource) -> None:
        self.source = _require_source_ready(source)
        self._observations: list[
            V1BlockSupportProductionMarginLowerBoundObservationV2
        ] = []
        self._state = V1BlockSupportProductionWitnessStateV2(
            bound_registration_receipt_hash=self.source.receipt.receipt_hash,
            legacy_registration_hash=self.source.registration.registration_hash,
            observed_block_count=0,
            last_observation_hash=None,
            receipt_chain_hash=_EMPTY_CHAIN_HASH,
        )

    @property
    def state(self) -> V1BlockSupportProductionWitnessStateV2:
        self._state.assert_integrity()
        return self._state

    @property
    def observations(
        self,
    ) -> tuple[V1BlockSupportProductionMarginLowerBoundObservationV2, ...]:
        return tuple(self._observations)

    def record(
        self,
        observation: V1BlockSupportProductionMarginLowerBoundObservationV2,
    ) -> V1BlockSupportProductionWitnessStateV2:
        self.source.assert_live()
        if (
            type(observation)
            is not V1BlockSupportProductionMarginLowerBoundObservationV2
        ):
            raise TypeError("production certifier accepts exact V2 observations")
        observation.assert_integrity()
        registration = self.source.registration
        pre = self.source.preoutcome_domain.receipt
        expected_index = len(self._observations)
        if expected_index >= registration.block_count:
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.OUT_OF_ORDER_BLOCK,
                "production certifier already consumed the full block census",
            )
        expected_identity = self.source.canonical_block_context_identity(expected_index)
        if (
            observation.bound_registration_receipt_hash
            != self.source.receipt.receipt_hash
            or observation.legacy_registration_hash != registration.registration_hash
            or observation.block_index != expected_index
            or observation.block_id != expected_identity.sealed_source_block_id
            or observation.block_context_axis_hash
            != expected_identity.block_support_context_axis_hash
            or observation.block_context_count != len(expected_identity.context_ids)
            or observation.sealed_block_order_hash
            != registration.sealed_block_order_hash
            or observation.complete_alternate_domain_hash
            != registration.complete_alternate_domain_hash
            or observation.alias_axis_hash != self.source.receipt.alias_axis_hash
            or observation.preferred_action_hash != pre.preferred_action_hash
            or observation.model_identity_hash != pre.model_identity_hash
            or observation.chart_identity_hash != pre.chart_identity_hash
            or observation.current_mask_hash != pre.current_mask_hash
            or observation.proposal_generation != pre.proposal_generation
            or observation.route_axis_hash != pre.route_axis_hash
            or observation.route_id_axis_hash != pre.route_id_axis_hash
            or observation.registered_d_prefix_hash != pre.registered_d_prefix_hash
            or observation.registered_d_w2_prefix_hash
            != pre.registered_d_w2_prefix_hash
            or observation.corrected_evidence_source_hash
            != registration.corrected_evidence_source_hash
            or (
                self._observations
                and observation.corrected_evidence_source_hash
                != self._observations[0].corrected_evidence_source_hash
            )
        ):
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.OUT_OF_ORDER_BLOCK,
                "observation is not the exact next sealed source-bound block",
            )
        if not (
            registration.analytic_margin_lower_bound
            <= observation.margin_lower_bound
            <= registration.analytic_margin_upper_bound
        ):
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.INVALID_RANGE,
                "production observation left its pre-outcome analytic range",
            )
        chain = content_hash(
            {
                "schema": "partition_free_v1_block_support_production_chain_link_v2",
                "prior_receipt_chain_hash": self._state.receipt_chain_hash,
                "block_index": expected_index,
                "observation_hash": observation.observation_hash,
            },
            prefix=_PREFIX + "chain-",
        )
        self._observations.append(observation)
        self._state = V1BlockSupportProductionWitnessStateV2(
            bound_registration_receipt_hash=self.source.receipt.receipt_hash,
            legacy_registration_hash=registration.registration_hash,
            observed_block_count=len(self._observations),
            last_observation_hash=observation.observation_hash,
            receipt_chain_hash=chain,
        )
        return self._state

    def decision(self) -> V1BlockSupportProductionWitnessDecisionV2:
        self.source.assert_live()
        self._state.assert_integrity()
        registration = self.source.registration
        pre = self.source.preoutcome_domain.receipt
        block_ids = self.source.legacy_source.block_ids
        sequence = BoundedWoRBettingCS(
            stratum_id=registration.betting_stratum_id,
            population_size=registration.block_count,
            alpha=registration.attempt_alpha_allocation,
            lower_bound=registration.analytic_margin_lower_bound,
            upper_bound=registration.analytic_margin_upper_bound,
            sealed_context_order=block_ids,
            sealed_order_hash=registration.sealed_block_order_hash,
            claim_bearing=True,
        )
        try:
            for observation in self._observations:
                sequence.record(
                    observation.margin_lower_bound,
                    context_id=observation.block_id,
                )
            snapshot = sequence.interval()
        except (ArithmeticError, FloatingPointError, ValueError) as error:
            _refuse(
                V1BlockSupportProductionWitnessRefusalCode.NUMERICAL_REFUSAL,
                f"bounded-WoR confidence sequence refused: {error}",
            )
        full_census = len(self._observations) == registration.block_count
        exact_sum: Fraction | None = None
        if full_census:
            exact_sum = sum(
                (
                    Fraction.from_float(value.margin_lower_bound)
                    for value in self._observations
                ),
                Fraction(0, 1),
            )
            if exact_sum > 0:
                status = V1BlockSupportWitnessStatus.STABLE_STRICT_SEPARATION
                reason = V1BlockSupportWitnessReason.EXACT_CENSUS_MEAN_STRICTLY_POSITIVE
            else:
                status = V1BlockSupportWitnessStatus.FAIL_CLOSED
                reason = V1BlockSupportWitnessReason.EXACT_CENSUS_MEAN_NONPOSITIVE
        elif (
            snapshot.empty
            or not math.isfinite(snapshot.lower)
            or not math.isfinite(snapshot.upper)
            or snapshot.lower > snapshot.upper
        ):
            status = V1BlockSupportWitnessStatus.FAIL_CLOSED
            reason = V1BlockSupportWitnessReason.NUMERICAL_OR_EMPTY_CONFIDENCE_SET
        elif snapshot.lower > 0.0:
            status = V1BlockSupportWitnessStatus.STABLE_STRICT_SEPARATION
            reason = (
                V1BlockSupportWitnessReason.LOWER_CONFIDENCE_BOUND_STRICTLY_POSITIVE
            )
        elif snapshot.upper <= 0.0:
            status = V1BlockSupportWitnessStatus.FAIL_CLOSED
            reason = V1BlockSupportWitnessReason.UPPER_CONFIDENCE_BOUND_NONPOSITIVE
        else:
            status = V1BlockSupportWitnessStatus.ACQUIRE_MORE_BLOCKS
            reason = V1BlockSupportWitnessReason.UNRESOLVED_FINITE_POPULATION_INTERVAL
        return V1BlockSupportProductionWitnessDecisionV2(
            bound_registration_receipt_hash=self.source.receipt.receipt_hash,
            legacy_registration_hash=registration.registration_hash,
            preoutcome_domain_receipt_hash=pre.receipt_hash,
            complete_alternate_domain_hash=pre.complete_alternate_domain_hash,
            complete_alternate_domain_receipt_hash=(
                pre.complete_alternate_domain_receipt_hash
            ),
            alternate_action_count=pre.hard_alternate_action_count,
            preferred_action_hash=pre.preferred_action_hash,
            alias_axis_hash=pre.alias_axis_hash,
            model_identity_hash=pre.model_identity_hash,
            chart_identity_hash=pre.chart_identity_hash,
            current_mask_hash=pre.current_mask_hash,
            proposal_generation=pre.proposal_generation,
            route_axis_hash=pre.route_axis_hash,
            route_id_axis_hash=pre.route_id_axis_hash,
            route_count=pre.route_count,
            corrected_evidence_source_hash=(
                self._observations[0].corrected_evidence_source_hash
                if self._observations
                else registration.corrected_evidence_source_hash
            ),
            registered_d_prefix_hash=pre.registered_d_prefix_hash,
            registered_d_w2_prefix_hash=pre.registered_d_w2_prefix_hash,
            state_hash=self._state.state_hash,
            receipt_chain_hash=self._state.receipt_chain_hash,
            status=status,
            reason=reason,
            observed_block_count=len(self._observations),
            block_population_count=registration.block_count,
            confidence_lower=snapshot.lower,
            confidence_upper=snapshot.upper,
            sample_mean_of_downward_bounds=snapshot.sample_mean,
            full_census=full_census,
            exact_census_sum_numerator=(
                exact_sum.numerator if exact_sum is not None else None
            ),
            exact_census_sum_denominator=(
                exact_sum.denominator if exact_sum is not None else None
            ),
            sampled_full_population_F_strict_separation_authority=(
                status is V1BlockSupportWitnessStatus.STABLE_STRICT_SEPARATION
            ),
            _issuer_token=_DECISION_TOKEN,
        )


def require_v1_block_support_production_witness_decision_v2(
    decision: V1BlockSupportProductionWitnessDecisionV2,
) -> V1BlockSupportProductionWitnessDecisionV2:
    """Open only the strict full-population corrected-F separation gate."""

    if type(decision) is not V1BlockSupportProductionWitnessDecisionV2:
        raise TypeError("production witness gate requires the exact V2 decision")
    decision.assert_integrity()
    if (
        decision.status is not V1BlockSupportWitnessStatus.STABLE_STRICT_SEPARATION
        or decision.sampled_full_population_F_strict_separation_authority is not True
        or decision.production_source_bound_witness is not True
    ):
        _refuse(
            V1BlockSupportProductionWitnessRefusalCode.STRICT_SEPARATION_REQUIRED,
            "production decision has not proved strict full-population F separation",
        )
    return decision


__all__ = (
    "PRODUCTION_BLOCK_DECISION_V2_AUTHORITY",
    "PRODUCTION_BLOCK_DECISION_V2_RULE",
    "PRODUCTION_BLOCK_DECISION_V2_SCHEMA",
    "PRODUCTION_BLOCK_OBSERVATION_V2_RULE",
    "PRODUCTION_BLOCK_OBSERVATION_V2_SCHEMA",
    "PRODUCTION_BLOCK_STATE_V2_SCHEMA",
    "V1BlockSupportProductionMarginLowerBoundObservationV2",
    "V1BlockSupportProductionWitnessCertifier",
    "V1BlockSupportProductionWitnessDecisionV2",
    "V1BlockSupportProductionWitnessRefusal",
    "V1BlockSupportProductionWitnessRefusalCode",
    "V1BlockSupportProductionWitnessStateV2",
    "issue_v1_signed_work_block_support_production_observation_v2",
    "require_v1_block_support_production_witness_decision_v2",
)
