"""Branchless post-C certification for unique signed-work physical actions.

This module is the first decision seam downstream of
``modern_v1_signed_work_physical_action_six_role_c``.  It consumes the exact
unique physical-action C axis and its six-role progress receipt.  It does not
reintroduce KKT branches, ladder endpoints, nesting, or monotonicity: every
candidate is an independent hard action, and the only order is the immutable
pre-C priority already registered by the physical-action axis.

The per-action roles remain deliberately separated:

* native-fiber signed ``F`` is replayed as the Phase-B least-work objective;
  no loss-derived ``tau_F`` is invented and ``F`` is not a C-stage
  functional-admission veto;
* literal-J ``Q`` is reported against ``tau_Q=tau_L**2``;
* directional ``R`` consumes the remaining response-length slack through
  ``eta*sqrt(Q)+R_eta <= eta*tau_L`` at both registered doses; and
* finite endpoint ``D_one`` is the action-admission authority.

The measurement design owns the norm and population measure. Its balanced-RMS
profile reconstructs R/D lengths from equal-stratum second-moment bounds;
the historical profile reconstructs population-weighted absolute means. This
decision layer must not reinterpret one profile's receipt as the other.

The result is either ``acquire_more_C``, ``stable_no_action``, or
``stable_action``.  A stable action carries only a receipt-only prefix of the
pre-frozen independent-action fallback order.  It cannot open calibration K,
open paired oracle O, materialize a mask, commit, or claim terminal/Pareto
authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral, Real
from typing import Final, Literal, NoReturn, Sequence

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_fqr_sequential_acquisition import (
    _response_burden,
    _response_burden_lower,
)
from .modern_v1_signed_work_physical_action_six_role_c import (
    D_HALF_ROLE,
    D_ONE_ROLE,
    F_NATIVE_ROLE,
    Q_J_ROLE,
    R_HALF_ROLE,
    R_ONE_ROLE,
    SIGNED_WORK_SIX_ROLE_DOSES,
    SIGNED_WORK_SIX_ROLES,
    V1PhysicalActionSixRoleActionSummary,
    V1PhysicalActionSixRoleCDesign,
    V1PhysicalActionSixRoleCRefusal,
    V1PhysicalActionSixRoleCellSummary,
    V1PhysicalActionSixRoleProgressReceipt,
)
from .modern_v1_signed_work_pre_c_packed_actions import (
    V1SignedWorkPackedActionDedupIdentity,
)


PHYSICAL_ACTION_SIX_ROLE_DECISION_DESIGN_SCHEMA: Final = (
    "partition_free_v1_physical_action_post_C_decision_design_v1"
)
PHYSICAL_ACTION_SIX_ROLE_ACTION_COORDINATE_SCHEMA: Final = (
    "partition_free_v1_physical_action_post_C_action_coordinate_v1"
)
PHYSICAL_ACTION_SIX_ROLE_ACTION_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_physical_action_post_C_action_certificate_v2"
)
PHYSICAL_ACTION_SIX_ROLE_CERTIFICATION_SCHEMA: Final = (
    "partition_free_v1_physical_action_post_C_certification_v1"
)
PHYSICAL_ACTION_SIX_ROLE_DECISION_SCHEMA: Final = (
    "partition_free_v1_physical_action_post_C_decision_v1"
)
PHYSICAL_ACTION_SIX_ROLE_DECISION_RULE: Final = (
    "first_direct_D_certified_action_after_all_prior_prefrozen_independent_"
    "actions_are_refuted_with_resolved_prefix_fallback_v1"
)
PHYSICAL_ACTION_SIX_ROLE_DECISION_AUTHORITY: Final = (
    "branchless_unique_physical_action_post_C_decision_evidence_only_no_K_O_"
    "calibration_oracle_materialization_commit_terminal_or_pareto_authority_v1"
)

_DESIGN_PREFIX: Final = "modern-v1-physical-action-post-C-design-"
_COORDINATE_PREFIX: Final = "modern-v1-physical-action-post-C-coordinate-"
_CERTIFICATE_PREFIX: Final = "modern-v1-physical-action-post-C-certificate-"
_RECEIPT_PREFIX: Final = "modern-v1-physical-action-post-C-receipt-"
_DECISION_PREFIX: Final = "modern-v1-physical-action-post-C-decision-"

PhysicalActionPostCDecisionStatus = Literal[
    "acquire_more_C",
    "stable_no_action",
    "stable_action",
]
PhysicalActionPostCAdmissionBasis = Literal[
    "direct_D_finite_endpoint",
]
PhysicalActionPostCEndpointClassification = Literal[
    "chart_certified",
    "direct_D_exactified",
]
PhysicalActionFallbackExhaustionStatus = Literal[
    "blocked_on_more_C",
    "fully_resolved",
]


class V1PhysicalActionIntervalStatus(str, Enum):
    CERTIFIED = "certified"
    REFUTED = "refuted"
    UNRESOLVED = "unresolved"


class V1PhysicalActionChartTransportStatus(str, Enum):
    CHART_CERTIFIED = "chart_certified"
    CHART_REFUTED = "chart_refuted"
    CHART_UNRESOLVED = "chart_unresolved"


class V1PhysicalActionPostCRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    HASH_MISMATCH = "hash_mismatch"
    STALE_IDENTITY = "stale_identity"
    ACTION_AXIS_MISMATCH = "action_axis_mismatch"
    POSTSELECTION = "postselection"
    INTERVAL_INVALID = "interval_invalid"
    UNSTABLE_DECISION = "unstable_decision"
    AUTHORITY_VIOLATION = "authority_violation"


class V1PhysicalActionPostCRefusal(ValueError):
    """Typed fail-closed refusal at the branchless post-C seam."""

    def __init__(
        self,
        code: V1PhysicalActionPostCRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1PhysicalActionPostCRefusalCode:
            raise TypeError("physical-action post-C refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1PhysicalActionPostCRefusalCode,
    message: str,
) -> NoReturn:
    raise V1PhysicalActionPostCRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1PhysicalActionPostCRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1PhysicalActionPostCRefusalCode.INVALID_TYPE,
            f"{name} must be one exact positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1PhysicalActionPostCRefusalCode.ACTION_AXIS_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1PhysicalActionPostCRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1PhysicalActionPostCRefusalCode.STALE_IDENTITY,
            f"{name} must be nonnegative",
        )
    return result


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1PhysicalActionPostCRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1PhysicalActionPostCRefusalCode.INTERVAL_INVALID,
            f"{name} must be finite and positive",
        )
    return result


def _interval(value: object, *, name: str) -> tuple[float, float]:
    if type(value) is not tuple or len(value) != 2:
        _refuse(
            V1PhysicalActionPostCRefusalCode.INVALID_TYPE,
            f"{name} must be one exact two-scalar tuple",
        )
    try:
        lower, upper = (float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise V1PhysicalActionPostCRefusal(
            V1PhysicalActionPostCRefusalCode.INTERVAL_INVALID,
            f"{name} is not numeric",
        ) from error
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower < 0.0
        or lower > upper
    ):
        _refuse(
            V1PhysicalActionPostCRefusalCode.INTERVAL_INVALID,
            f"{name} must be finite, nonnegative, and ordered",
        )
    return (0.0 if lower == 0.0 else lower, 0.0 if upper == 0.0 else upper)


def _dedup_identity(
    value: object,
) -> V1SignedWorkPackedActionDedupIdentity:
    if type(value) is not tuple or len(value) != 5:
        _refuse(
            V1PhysicalActionPostCRefusalCode.INVALID_TYPE,
            "physical-action dedup identity must be the canonical five-part tuple",
        )
    if any(type(item) is not str or not item for item in value[:4]):
        _refuse(
            V1PhysicalActionPostCRefusalCode.INVALID_TYPE,
            "physical-action dedup hash coordinates must be nonempty strings",
        )
    count = _positive_int(value[4], name="dedup removed count")
    return (value[0], value[1], value[2], value[3], count)


def _single_interval_status(
    interval: tuple[float, float],
    *,
    budget: float,
) -> V1PhysicalActionIntervalStatus:
    lower, upper = interval
    if upper <= budget:
        return V1PhysicalActionIntervalStatus.CERTIFIED
    if lower > budget:
        return V1PhysicalActionIntervalStatus.REFUTED
    return V1PhysicalActionIntervalStatus.UNRESOLVED


def _two_dose_status(
    *,
    half: tuple[float, float],
    one: tuple[float, float],
    tau_l: float,
) -> V1PhysicalActionIntervalStatus:
    if half[1] <= 0.5 * tau_l and one[1] <= tau_l:
        return V1PhysicalActionIntervalStatus.CERTIFIED
    if half[0] > 0.5 * tau_l or one[0] > tau_l:
        return V1PhysicalActionIntervalStatus.REFUTED
    return V1PhysicalActionIntervalStatus.UNRESOLVED


def _closed_authority(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool for value in values) or any(values):
        _refuse(
            V1PhysicalActionPostCRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed the post-C evidence boundary",
        )


@dataclass(frozen=True, slots=True)
class V1PhysicalActionPostCActionCoordinate:
    """Receipt-only identity for one independent pre-frozen hard action."""

    pre_C_priority: int
    physical_action_ordinal: int
    physical_action_entry_hash: str
    canonical_action_hash: str
    packed_action_hash: str
    dedup_identity: V1SignedWorkPackedActionDedupIdentity
    resulting_mask_hash: str
    resulting_mask_packed_sha256: str
    removed_route_index_hash: str
    removed_count: int
    alias_fanout_receipt_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    schema: str = PHYSICAL_ACTION_SIX_ROLE_ACTION_COORDINATE_SCHEMA

    def __post_init__(self) -> None:
        priority = _positive_int(self.pre_C_priority, name="pre-C priority")
        ordinal = _positive_int(
            self.physical_action_ordinal,
            name="physical-action ordinal",
        )
        removed = _positive_int(self.removed_count, name="removed count")
        routes = _positive_int(self.route_count, name="route count")
        for name in (
            "physical_action_entry_hash",
            "canonical_action_hash",
            "packed_action_hash",
            "resulting_mask_hash",
            "resulting_mask_packed_sha256",
            "removed_route_index_hash",
            "alias_fanout_receipt_hash",
            "parent_mask_hash",
            "route_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        dedup = _dedup_identity(self.dedup_identity)
        if dedup != (
            self.resulting_mask_packed_sha256,
            self.packed_action_hash,
            self.resulting_mask_hash,
            self.removed_route_index_hash,
            removed,
        ):
            _refuse(
                V1PhysicalActionPostCRefusalCode.HASH_MISMATCH,
                "post-C coordinate does not replay its canonical packed-action identity",
            )
        if removed > routes or priority != ordinal:
            _refuse(
                V1PhysicalActionPostCRefusalCode.ACTION_AXIS_MISMATCH,
                "physical-action priority/ordinal or removal cardinality is invalid",
            )
        if self.schema != PHYSICAL_ACTION_SIX_ROLE_ACTION_COORDINATE_SCHEMA:
            _refuse(
                V1PhysicalActionPostCRefusalCode.AUTHORITY_VIOLATION,
                "unsupported physical-action coordinate schema",
            )
        object.__setattr__(self, "pre_C_priority", priority)
        object.__setattr__(self, "physical_action_ordinal", ordinal)
        object.__setattr__(self, "dedup_identity", dedup)
        object.__setattr__(self, "removed_count", removed)
        object.__setattr__(self, "route_count", routes)

    @property
    def coordinate_hash(self) -> str:
        return content_hash(self, prefix=_COORDINATE_PREFIX)


@dataclass(frozen=True, slots=True)
class V1PhysicalActionSixRolePostCDecisionDesign:
    """Action order, identities, and canonical loss scale frozen before C."""

    physical_action_six_role_C_design_hash: str
    physical_action_axis_hash: str
    direction_provider_receipt_hash: str
    source_registration_hash: str
    chart_identity_hash: str
    current_parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    complete_train_population_hash: str
    c_stream_hash: str
    preoutcome_priority_axis_hash: str
    tau_l: float
    tau_l_source_hash: str
    action_coordinates: tuple[V1PhysicalActionPostCActionCoordinate, ...]
    doses: tuple[float, float] = (0.5, 1.0)
    no_default_tau_F: bool = True
    tau_Q_equals_tau_L_squared: bool = True
    no_independent_tau_R: bool = True
    F_is_objective_evidence_not_admission_veto: bool = True
    direct_D_owns_finite_action_admission: bool = True
    QR_owns_chart_transport_label_only: bool = True
    independent_actions_not_nested: bool = True
    branch_semantics_absent: bool = True
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_materialization_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    decision_rule: str = PHYSICAL_ACTION_SIX_ROLE_DECISION_RULE
    authority_scope: str = PHYSICAL_ACTION_SIX_ROLE_DECISION_AUTHORITY
    schema: str = PHYSICAL_ACTION_SIX_ROLE_DECISION_DESIGN_SCHEMA
    _design_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "physical_action_six_role_C_design_hash",
            "physical_action_axis_hash",
            "direction_provider_receipt_hash",
            "source_registration_hash",
            "chart_identity_hash",
            "current_parent_mask_hash",
            "route_axis_hash",
            "complete_train_population_hash",
            "c_stream_hash",
            "preoutcome_priority_axis_hash",
            "tau_l_source_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        tau_l = _positive_finite(self.tau_l, name="tau_L")
        if not math.isfinite(tau_l * tau_l):
            _refuse(
                V1PhysicalActionPostCRefusalCode.INTERVAL_INVALID,
                "tau_L squared is not finite",
            )
        coordinates = tuple(self.action_coordinates)
        if not coordinates or any(
            type(value) is not V1PhysicalActionPostCActionCoordinate
            for value in coordinates
        ):
            _refuse(
                V1PhysicalActionPostCRefusalCode.INVALID_TYPE,
                "post-C design requires exact nonempty physical-action coordinates",
            )
        if tuple(value.pre_C_priority for value in coordinates) != tuple(
            range(1, len(coordinates) + 1)
        ) or tuple(value.physical_action_ordinal for value in coordinates) != tuple(
            range(1, len(coordinates) + 1)
        ):
            _refuse(
                V1PhysicalActionPostCRefusalCode.POSTSELECTION,
                "physical actions were omitted, duplicated, or reordered after C opened",
            )
        if len({value.physical_action_entry_hash for value in coordinates}) != len(
            coordinates
        ) or len({value.dedup_identity for value in coordinates}) != len(coordinates):
            _refuse(
                V1PhysicalActionPostCRefusalCode.ACTION_AXIS_MISMATCH,
                "post-C design contains duplicate unique physical actions",
            )
        if any(
            value.parent_mask_hash != self.current_parent_mask_hash
            or value.route_axis_hash != self.route_axis_hash
            or value.route_count != routes
            for value in coordinates
        ):
            _refuse(
                V1PhysicalActionPostCRefusalCode.STALE_IDENTITY,
                "action coordinates do not share the exact chart parent and route axis",
            )
        required_true = (
            self.no_default_tau_F,
            self.tau_Q_equals_tau_L_squared,
            self.no_independent_tau_R,
            self.F_is_objective_evidence_not_admission_veto,
            self.direct_D_owns_finite_action_admission,
            self.QR_owns_chart_transport_label_only,
            self.independent_actions_not_nested,
            self.branch_semantics_absent,
            self.issued_before_C_outcomes,
        )
        if any(type(value) is not bool or not value for value in required_true):
            _refuse(
                V1PhysicalActionPostCRefusalCode.AUTHORITY_VIOLATION,
                "post-C design changed the canonical branchless role split",
            )
        if type(self.C_outcomes_seen) is not bool or self.C_outcomes_seen:
            _refuse(
                V1PhysicalActionPostCRefusalCode.POSTSELECTION,
                "post-C decision design was not frozen before C outcomes",
            )
        _closed_authority(
            (
                self.calibration_K_open_authority,
                self.paired_oracle_O_open_authority,
                self.mask_materialization_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical-action post-C decision design",
        )
        if (
            self.doses != (0.5, 1.0)
            or self.decision_rule != PHYSICAL_ACTION_SIX_ROLE_DECISION_RULE
            or self.authority_scope != PHYSICAL_ACTION_SIX_ROLE_DECISION_AUTHORITY
            or self.schema != PHYSICAL_ACTION_SIX_ROLE_DECISION_DESIGN_SCHEMA
        ):
            _refuse(
                V1PhysicalActionPostCRefusalCode.AUTHORITY_VIOLATION,
                "unsupported post-C decision semantics",
            )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "tau_l", tau_l)
        object.__setattr__(self, "action_coordinates", coordinates)
        self.assert_integrity()

    @property
    def tau_q(self) -> float:
        return self.tau_l * self.tau_l

    @property
    def design_hash(self) -> str:
        return self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "_design_hash"
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_DESIGN_PREFIX)
        cached = self._design_hash
        if cached is not None and cached != observed:
            _refuse(
                V1PhysicalActionPostCRefusalCode.HASH_MISMATCH,
                "post-C decision design was tampered after pre-outcome issuance",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1PhysicalActionPostCRefusalCode.HASH_MISMATCH,
                "post-C decision-design hash differs",
            )
        object.__setattr__(self, "_design_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1PhysicalActionSixRoleActionCertificate:
    coordinate: V1PhysicalActionPostCActionCoordinate
    f_native_interval: tuple[float, float]
    q_j_interval: tuple[float, float]
    r_half_interval: tuple[float, float]
    r_one_interval: tuple[float, float]
    d_half_interval: tuple[float, float]
    d_one_interval: tuple[float, float]
    chart_half_interval: tuple[float, float]
    chart_one_interval: tuple[float, float]
    tau_l: float
    Q_status: V1PhysicalActionIntervalStatus
    R_slack_status: V1PhysicalActionIntervalStatus
    direct_D_status: V1PhysicalActionIntervalStatus
    chart_transport_status: V1PhysicalActionChartTransportStatus
    admission_basis: PhysicalActionPostCAdmissionBasis | None
    endpoint_classification: PhysicalActionPostCEndpointClassification | None
    signed_F_used_as_admission_veto: bool = False
    Q_used_as_independent_admission_veto: bool = False
    R_has_independent_tuned_budget: bool = False
    half_dose_D_used_as_endpoint_veto: bool = False
    schema: str = PHYSICAL_ACTION_SIX_ROLE_ACTION_CERTIFICATE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.coordinate) is not V1PhysicalActionPostCActionCoordinate:
            _refuse(
                V1PhysicalActionPostCRefusalCode.INVALID_TYPE,
                "action certificate requires its exact pre-C coordinate",
            )
        for name in (
            "f_native_interval",
            "q_j_interval",
            "r_half_interval",
            "r_one_interval",
            "d_half_interval",
            "d_one_interval",
            "chart_half_interval",
            "chart_one_interval",
        ):
            object.__setattr__(self, name, _interval(getattr(self, name), name=name))
        tau_l = _positive_finite(self.tau_l, name="certificate tau_L")
        expected_half = (
            _response_burden_lower(
                eta=0.5,
                q_lower=self.q_j_interval[0],
                r_lower=self.r_half_interval[0],
            ),
            _response_burden(
                eta=0.5,
                q_upper=self.q_j_interval[1],
                r_upper=self.r_half_interval[1],
            ),
        )
        expected_one = (
            _response_burden_lower(
                eta=1.0,
                q_lower=self.q_j_interval[0],
                r_lower=self.r_one_interval[0],
            ),
            _response_burden(
                eta=1.0,
                q_upper=self.q_j_interval[1],
                r_upper=self.r_one_interval[1],
            ),
        )
        expected_Q = _single_interval_status(
            self.q_j_interval,
            budget=tau_l * tau_l,
        )
        expected_R = _two_dose_status(
            half=expected_half,
            one=expected_one,
            tau_l=tau_l,
        )
        expected_D = _single_interval_status(self.d_one_interval, budget=tau_l)
        expected_chart = {
            V1PhysicalActionIntervalStatus.CERTIFIED: (
                V1PhysicalActionChartTransportStatus.CHART_CERTIFIED
            ),
            V1PhysicalActionIntervalStatus.REFUTED: (
                V1PhysicalActionChartTransportStatus.CHART_REFUTED
            ),
            V1PhysicalActionIntervalStatus.UNRESOLVED: (
                V1PhysicalActionChartTransportStatus.CHART_UNRESOLVED
            ),
        }[expected_R]
        expected_basis: PhysicalActionPostCAdmissionBasis | None = None
        expected_classification: PhysicalActionPostCEndpointClassification | None = None
        if expected_D is V1PhysicalActionIntervalStatus.CERTIFIED:
            expected_basis = "direct_D_finite_endpoint"
            expected_classification = (
                "chart_certified"
                if expected_chart
                is V1PhysicalActionChartTransportStatus.CHART_CERTIFIED
                else "direct_D_exactified"
            )
        if (
            self.chart_half_interval != expected_half
            or self.chart_one_interval != expected_one
            or self.Q_status is not expected_Q
            or self.R_slack_status is not expected_R
            or self.direct_D_status is not expected_D
            or self.chart_transport_status is not expected_chart
            or self.admission_basis != expected_basis
            or self.endpoint_classification != expected_classification
        ):
            _refuse(
                V1PhysicalActionPostCRefusalCode.AUTHORITY_VIOLATION,
                "action statuses do not replay from the six-role intervals and tau_L",
            )
        _closed_authority(
            (
                self.signed_F_used_as_admission_veto,
                self.Q_used_as_independent_admission_veto,
                self.R_has_independent_tuned_budget,
                self.half_dose_D_used_as_endpoint_veto,
            ),
            owner="physical-action certificate role split",
        )
        if self.schema != PHYSICAL_ACTION_SIX_ROLE_ACTION_CERTIFICATE_SCHEMA:
            _refuse(
                V1PhysicalActionPostCRefusalCode.AUTHORITY_VIOLATION,
                "unsupported physical-action certificate schema",
            )
        object.__setattr__(self, "tau_l", tau_l)

    @property
    def certificate_hash(self) -> str:
        return content_hash(self, prefix=_CERTIFICATE_PREFIX)


@dataclass(frozen=True, slots=True)
class V1PhysicalActionSixRoleCertificationReceipt:
    decision_design_hash: str
    physical_action_six_role_C_design_hash: str
    physical_action_six_role_C_progress_receipt_hash: str
    physical_action_axis_hash: str
    source_registration_hash: str
    chart_identity_hash: str
    current_parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    tau_l: float
    action_certificates: tuple[V1PhysicalActionSixRoleActionCertificate, ...]
    sampling_status: str
    branch_or_nesting_semantics_used: bool = False
    signed_F_used_as_admission_veto: bool = False
    direct_D_is_action_admission_authority: bool = True
    QR_is_chart_transport_label_only: bool = True
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_materialization_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = PHYSICAL_ACTION_SIX_ROLE_DECISION_AUTHORITY
    schema: str = PHYSICAL_ACTION_SIX_ROLE_CERTIFICATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "decision_design_hash",
            "physical_action_six_role_C_design_hash",
            "physical_action_six_role_C_progress_receipt_hash",
            "physical_action_axis_hash",
            "source_registration_hash",
            "chart_identity_hash",
            "current_parent_mask_hash",
            "route_axis_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="receipt route count")
        generation = _nonnegative_int(
            self.proposal_generation,
            name="receipt proposal generation",
        )
        tau_l = _positive_finite(self.tau_l, name="receipt tau_L")
        certificates = tuple(self.action_certificates)
        if not certificates or any(
            type(value) is not V1PhysicalActionSixRoleActionCertificate
            for value in certificates
        ):
            _refuse(
                V1PhysicalActionPostCRefusalCode.INVALID_TYPE,
                "certification receipt requires exact nonempty action certificates",
            )
        if tuple(value.coordinate.pre_C_priority for value in certificates) != tuple(
            range(1, len(certificates) + 1)
        ) or len(
            {value.coordinate.physical_action_entry_hash for value in certificates}
        ) != len(certificates):
            _refuse(
                V1PhysicalActionPostCRefusalCode.POSTSELECTION,
                "certified physical-action axis was omitted, duplicated, or reordered",
            )
        if any(value.tau_l != tau_l for value in certificates):
            _refuse(
                V1PhysicalActionPostCRefusalCode.INTERVAL_INVALID,
                "action certificate tau_L differs from the family receipt",
            )
        _identity(self.sampling_status, name="sampling status")
        _closed_authority(
            (
                self.branch_or_nesting_semantics_used,
                self.signed_F_used_as_admission_veto,
                self.calibration_K_open_authority,
                self.paired_oracle_O_open_authority,
                self.mask_materialization_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical-action certification receipt",
        )
        if (
            type(self.direct_D_is_action_admission_authority) is not bool
            or not self.direct_D_is_action_admission_authority
            or type(self.QR_is_chart_transport_label_only) is not bool
            or not self.QR_is_chart_transport_label_only
            or self.authority_scope != PHYSICAL_ACTION_SIX_ROLE_DECISION_AUTHORITY
            or self.schema != PHYSICAL_ACTION_SIX_ROLE_CERTIFICATION_SCHEMA
        ):
            _refuse(
                V1PhysicalActionPostCRefusalCode.AUTHORITY_VIOLATION,
                "certification receipt changed the canonical authority split",
            )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "tau_l", tau_l)
        object.__setattr__(self, "action_certificates", certificates)

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_RECEIPT_PREFIX)


@dataclass(frozen=True, slots=True)
class V1PhysicalActionSixRoleDecision:
    certification_receipt_hash: str
    decision_design_hash: str
    status: PhysicalActionPostCDecisionStatus
    selected_action_coordinate: V1PhysicalActionPostCActionCoordinate | None
    selected_certificate_hash: str | None
    fallback_action_coordinates: tuple[V1PhysicalActionPostCActionCoordinate, ...]
    fallback_certificate_hashes: tuple[str, ...]
    unresolved_decision_action_entry_hashes: tuple[str, ...]
    blocked_fallback_action_entry_hash: str | None
    fallback_exhaustion_status: PhysicalActionFallbackExhaustionStatus
    stable_action_evidence: bool
    actions_are_independent_not_nested: bool = True
    fallback_is_prefrozen_order_subsequence: bool = True
    candidate_composition_mutable_after_C: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_materialization_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    paired_bce_oracle_still_required: bool = True
    decision_rule: str = PHYSICAL_ACTION_SIX_ROLE_DECISION_RULE
    authority_scope: str = PHYSICAL_ACTION_SIX_ROLE_DECISION_AUTHORITY
    schema: str = PHYSICAL_ACTION_SIX_ROLE_DECISION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "certification_receipt_hash",
            _identity(self.certification_receipt_hash, name="certification hash"),
        )
        object.__setattr__(
            self,
            "decision_design_hash",
            _identity(self.decision_design_hash, name="decision-design hash"),
        )
        if self.status not in {
            "acquire_more_C",
            "stable_no_action",
            "stable_action",
        }:
            _refuse(
                V1PhysicalActionPostCRefusalCode.UNSTABLE_DECISION,
                "unknown physical-action post-C decision status",
            )
        queue = tuple(self.fallback_action_coordinates)
        queue_hashes = tuple(
            _identity(value, name="fallback certificate hash")
            for value in self.fallback_certificate_hashes
        )
        unresolved = tuple(
            _identity(value, name="unresolved physical-action entry hash")
            for value in self.unresolved_decision_action_entry_hashes
        )
        if any(
            type(value) is not V1PhysicalActionPostCActionCoordinate
            for value in queue
        ) or len(queue) != len(queue_hashes):
            _refuse(
                V1PhysicalActionPostCRefusalCode.INVALID_TYPE,
                "fallback queue coordinate/certificate axes differ",
            )
        priorities = tuple(value.pre_C_priority for value in queue)
        if priorities != tuple(sorted(priorities)) or len(set(priorities)) != len(
            priorities
        ):
            _refuse(
                V1PhysicalActionPostCRefusalCode.POSTSELECTION,
                "fallback queue reordered or duplicated its pre-C action order",
            )
        selected = self.selected_action_coordinate
        if self.status == "stable_action":
            if (
                type(selected) is not V1PhysicalActionPostCActionCoordinate
                or not queue
                or queue[0].coordinate_hash != selected.coordinate_hash
                or self.selected_certificate_hash is None
                or queue_hashes[0] != self.selected_certificate_hash
                or type(self.stable_action_evidence) is not bool
                or not self.stable_action_evidence
                or unresolved
            ):
                _refuse(
                    V1PhysicalActionPostCRefusalCode.UNSTABLE_DECISION,
                    "stable action does not replay the first resolved fallback action",
                )
            _identity(self.selected_certificate_hash, name="selected certificate hash")
        else:
            if (
                selected is not None
                or self.selected_certificate_hash is not None
                or queue
                or queue_hashes
                or type(self.stable_action_evidence) is not bool
                or self.stable_action_evidence
            ):
                _refuse(
                    V1PhysicalActionPostCRefusalCode.AUTHORITY_VIOLATION,
                    "non-action decision named or authorized a physical action",
                )
        if self.status == "acquire_more_C":
            if not unresolved or self.blocked_fallback_action_entry_hash is not None:
                _refuse(
                    V1PhysicalActionPostCRefusalCode.UNSTABLE_DECISION,
                    "acquire-more-C decision must name decision-changing blockers only",
                )
        elif unresolved:
            _refuse(
                V1PhysicalActionPostCRefusalCode.UNSTABLE_DECISION,
                "stable decision retained decision-changing blockers",
            )
        if self.blocked_fallback_action_entry_hash is not None:
            _identity(
                self.blocked_fallback_action_entry_hash,
                name="blocked fallback action entry hash",
            )
        expected_exhaustion: PhysicalActionFallbackExhaustionStatus = (
            "blocked_on_more_C"
            if self.status == "acquire_more_C"
            or self.blocked_fallback_action_entry_hash is not None
            else "fully_resolved"
        )
        if self.fallback_exhaustion_status != expected_exhaustion:
            _refuse(
                V1PhysicalActionPostCRefusalCode.UNSTABLE_DECISION,
                "fallback exhaustion label differs from unresolved evidence",
            )
        required_true = (
            self.actions_are_independent_not_nested,
            self.fallback_is_prefrozen_order_subsequence,
            self.paired_bce_oracle_still_required,
        )
        if any(type(value) is not bool or not value for value in required_true):
            _refuse(
                V1PhysicalActionPostCRefusalCode.AUTHORITY_VIOLATION,
                "decision reintroduced nesting or weakened paired-oracle custody",
            )
        _closed_authority(
            (
                self.candidate_composition_mutable_after_C,
                self.calibration_K_open_authority,
                self.paired_oracle_O_open_authority,
                self.mask_materialization_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical-action post-C decision",
        )
        if (
            self.decision_rule != PHYSICAL_ACTION_SIX_ROLE_DECISION_RULE
            or self.authority_scope != PHYSICAL_ACTION_SIX_ROLE_DECISION_AUTHORITY
            or self.schema != PHYSICAL_ACTION_SIX_ROLE_DECISION_SCHEMA
        ):
            _refuse(
                V1PhysicalActionPostCRefusalCode.AUTHORITY_VIOLATION,
                "unsupported physical-action decision semantics",
            )
        object.__setattr__(self, "fallback_action_coordinates", queue)
        object.__setattr__(self, "fallback_certificate_hashes", queue_hashes)
        object.__setattr__(self, "unresolved_decision_action_entry_hashes", unresolved)

    @property
    def decision_hash(self) -> str:
        return content_hash(self, prefix=_DECISION_PREFIX)


def _coordinate_from_entry(
    *,
    pre_C_priority: int,
    entry: object,
) -> V1PhysicalActionPostCActionCoordinate:
    # Local shape validation avoids importing the axis entry merely to permit
    # subclasses or duck-typed reminting.  The enclosing exact axis has already
    # validated the exact entry type and its integrity.
    packed = entry.packed_action
    return V1PhysicalActionPostCActionCoordinate(
        pre_C_priority=pre_C_priority,
        physical_action_ordinal=entry.physical_action_ordinal,
        physical_action_entry_hash=entry.entry_hash,
        canonical_action_hash=entry.canonical_action_hash,
        packed_action_hash=packed.action_hash,
        dedup_identity=entry.dedup_identity,
        resulting_mask_hash=packed.resulting_mask.mask_hash,
        resulting_mask_packed_sha256=entry.actual_packed_sha256,
        removed_route_index_hash=packed.removed_route_index_hash,
        removed_count=packed.removed_count,
        alias_fanout_receipt_hash=entry.alias_fanout_receipt_hash,
        parent_mask_hash=entry.parent_mask_hash,
        route_axis_hash=entry.route_axis_hash,
        route_count=entry.route_count,
    )


def register_v1_signed_work_physical_action_six_role_post_C_decision_design(
    *,
    six_role_C_design: object,
    tau_l: float,
    tau_l_source_hash: str,
) -> V1PhysicalActionSixRolePostCDecisionDesign:
    """Freeze exact action identities and the canonical loss scale pre-outcome."""

    if type(six_role_C_design) is not V1PhysicalActionSixRoleCDesign:
        _refuse(
            V1PhysicalActionPostCRefusalCode.INVALID_TYPE,
            "post-C registration requires the exact physical-action C design",
        )
    design = six_role_C_design
    try:
        design_hash = design.assert_integrity()
        axis = design.physical_action_axis
        axis_hash = axis.assert_integrity()
        source_hash = design.source_registration.assert_integrity()
        provider_hash = design.direction_provider_receipt.assert_integrity()
    except (TypeError, ValueError, RuntimeError) as error:
        raise V1PhysicalActionPostCRefusal(
            V1PhysicalActionPostCRefusalCode.HASH_MISMATCH,
            "physical-action C source graph failed integrity replay",
        ) from error
    if tuple(axis.preoutcome_priority_action_hashes) != tuple(
        value.entry_hash for value in axis.physical_actions
    ):
        _refuse(
            V1PhysicalActionPostCRefusalCode.POSTSELECTION,
            "physical-action axis priority changed before post-C registration",
        )
    coordinates = tuple(
        _coordinate_from_entry(pre_C_priority=index, entry=entry)
        for index, entry in enumerate(axis.physical_actions, start=1)
    )
    return V1PhysicalActionSixRolePostCDecisionDesign(
        physical_action_six_role_C_design_hash=design_hash,
        physical_action_axis_hash=axis_hash,
        direction_provider_receipt_hash=provider_hash,
        source_registration_hash=source_hash,
        chart_identity_hash=design.source_registration.chart_identity_hash,
        current_parent_mask_hash=design.parent_mask_hash,
        route_axis_hash=design.route_axis_hash,
        route_count=design.route_count,
        proposal_generation=design.proposal_generation,
        complete_train_population_hash=design.complete_train_population_hash,
        c_stream_hash=design.c_stream_hash,
        preoutcome_priority_axis_hash=axis.preoutcome_priority_axis_hash,
        tau_l=tau_l,
        tau_l_source_hash=tau_l_source_hash,
        action_coordinates=coordinates,
    )


def _validated_action_summaries(
    *,
    design: V1PhysicalActionSixRoleCDesign,
    progress: V1PhysicalActionSixRoleProgressReceipt,
) -> tuple[V1PhysicalActionSixRoleActionSummary, ...]:
    """Rebuild every action summary from the bounded C-cell intervals."""

    action_hashes = design.physical_action_entry_hashes
    strata = design.strata
    counts = progress.observed_counts_by_stratum
    stratum_index_by_id = {
        stratum.stratum_id: index for index, stratum in enumerate(strata)
    }
    action_index_by_hash = {
        action_hash: index for index, action_hash in enumerate(action_hashes)
    }
    role_index_by_name = {
        role: index for index, role in enumerate(SIGNED_WORK_SIX_ROLES)
    }
    expected_cells = tuple(
        (stratum.stratum_id, action_hash, role)
        for stratum in strata
        for action_hash in action_hashes
        for role in SIGNED_WORK_SIX_ROLES
    )
    observed_cells = tuple(
        (cell.stratum_id, cell.physical_action_entry_hash, cell.role)
        for cell in progress.cell_summaries
    )
    if observed_cells != expected_cells:
        _refuse(
            V1PhysicalActionPostCRefusalCode.POSTSELECTION,
            "physical-action C cells were omitted, duplicated, or reordered",
        )
    cell_by_key: dict[tuple[str, str, str], V1PhysicalActionSixRoleCellSummary] = {}
    for cell in progress.cell_summaries:
        if type(cell) is not V1PhysicalActionSixRoleCellSummary:
            _refuse(
                V1PhysicalActionPostCRefusalCode.INVALID_TYPE,
                "post-C progress contains a noncanonical cell",
            )
        # These maps keep public replay linear in the number of registered
        # six-role cells.  Repeated tuple.index scans would make a multi-action
        # C family quadratic even though the cell axis is already sealed.
        stratum_index = stratum_index_by_id[cell.stratum_id]
        action_index = action_index_by_hash[cell.physical_action_entry_hash]
        role_index = role_index_by_name[cell.role]
        expected_count = counts[stratum_index]
        expected_population = strata[stratum_index].population_size
        expected_support = float(
            design.analytic_support_upper_by_stratum_action_role[
                stratum_index,
                action_index,
                role_index,
            ]
        )
        expected_alpha = (
            0.0
            if expected_count in {0, expected_population}
            else design.look_alpha(
                stratum_index=stratum_index,
                sample_size=expected_count,
            )
        )
        if (
            cell.dose != SIGNED_WORK_SIX_ROLE_DOSES[cell.role]
            or cell.population_size != expected_population
            or cell.sample_size != expected_count
            or cell.value_upper_bound != expected_support
            or cell.current_lower_look_alpha != expected_alpha
            or cell.current_upper_look_alpha != expected_alpha
            or cell.observation_chain_hash != progress.observation_chain_hash
        ):
            _refuse(
                V1PhysicalActionPostCRefusalCode.STALE_IDENTITY,
                "C-cell sufficient statistics left the sealed prefix or support",
            )
        cell_by_key[(cell.stratum_id, cell.physical_action_entry_hash, cell.role)] = cell

    reconstructed: list[V1PhysicalActionSixRoleActionSummary] = []
    for action_hash in action_hashes:
        lowers: dict[str, float] = {}
        uppers: dict[str, float] = {}
        for role in SIGNED_WORK_SIX_ROLES:
            cells = tuple(
                cell_by_key[(stratum.stratum_id, action_hash, role)]
                for stratum in strata
            )
            lowers[role] = design.aggregate_role_bound(
                role=role,
                values=tuple(value.lower for value in cells),
                direction="lower",
            )
            uppers[role] = design.aggregate_role_bound(
                role=role,
                values=tuple(value.upper for value in cells),
                direction="upper",
            )
        reconstructed.append(
            V1PhysicalActionSixRoleActionSummary(
                physical_action_entry_hash=action_hash,
                f_native_lower=lowers[F_NATIVE_ROLE],
                f_native_upper=uppers[F_NATIVE_ROLE],
                q_j_lower=lowers[Q_J_ROLE],
                q_j_upper=uppers[Q_J_ROLE],
                r_half_lower=lowers[R_HALF_ROLE],
                r_half_upper=uppers[R_HALF_ROLE],
                r_one_lower=lowers[R_ONE_ROLE],
                r_one_upper=uppers[R_ONE_ROLE],
                d_half_lower=lowers[D_HALF_ROLE],
                d_half_upper=uppers[D_HALF_ROLE],
                d_one_lower=lowers[D_ONE_ROLE],
                d_one_upper=uppers[D_ONE_ROLE],
                chart_half_burden_lower=_response_burden_lower(
                    eta=0.5,
                    q_lower=lowers[Q_J_ROLE],
                    r_lower=lowers[R_HALF_ROLE],
                ),
                chart_half_burden_upper=_response_burden(
                    eta=0.5,
                    q_upper=uppers[Q_J_ROLE],
                    r_upper=uppers[R_HALF_ROLE],
                ),
                chart_one_burden_lower=_response_burden_lower(
                    eta=1.0,
                    q_lower=lowers[Q_J_ROLE],
                    r_lower=lowers[R_ONE_ROLE],
                ),
                chart_one_burden_upper=_response_burden(
                    eta=1.0,
                    q_upper=uppers[Q_J_ROLE],
                    r_upper=uppers[R_ONE_ROLE],
                ),
            )
        )
    expected = tuple(reconstructed)
    if tuple(progress.action_summaries) != expected:
        _refuse(
            V1PhysicalActionPostCRefusalCode.POSTSELECTION,
            "aggregate action summaries do not replay their complete bounded cells",
        )
    return expected


def _certificate(
    *,
    coordinate: V1PhysicalActionPostCActionCoordinate,
    summary: V1PhysicalActionSixRoleActionSummary,
    tau_l: float,
) -> V1PhysicalActionSixRoleActionCertificate:
    if summary.physical_action_entry_hash != coordinate.physical_action_entry_hash:
        _refuse(
            V1PhysicalActionPostCRefusalCode.ACTION_AXIS_MISMATCH,
            "C action summary left its exact physical-action coordinate",
        )
    f_interval = (summary.f_native_lower, summary.f_native_upper)
    q_interval = (summary.q_j_lower, summary.q_j_upper)
    r_half = (summary.r_half_lower, summary.r_half_upper)
    r_one = (summary.r_one_lower, summary.r_one_upper)
    d_half = (summary.d_half_lower, summary.d_half_upper)
    d_one = (summary.d_one_lower, summary.d_one_upper)
    chart_half = (
        _response_burden_lower(
            eta=0.5,
            q_lower=q_interval[0],
            r_lower=r_half[0],
        ),
        _response_burden(
            eta=0.5,
            q_upper=q_interval[1],
            r_upper=r_half[1],
        ),
    )
    chart_one = (
        _response_burden_lower(
            eta=1.0,
            q_lower=q_interval[0],
            r_lower=r_one[0],
        ),
        _response_burden(
            eta=1.0,
            q_upper=q_interval[1],
            r_upper=r_one[1],
        ),
    )
    if chart_half != (
        summary.chart_half_burden_lower,
        summary.chart_half_burden_upper,
    ) or chart_one != (
        summary.chart_one_burden_lower,
        summary.chart_one_burden_upper,
    ):
        _refuse(
            V1PhysicalActionPostCRefusalCode.INTERVAL_INVALID,
            "Q/R chart burden does not replay from its literal components",
        )
    Q_status = _single_interval_status(q_interval, budget=tau_l * tau_l)
    R_status = _two_dose_status(half=chart_half, one=chart_one, tau_l=tau_l)
    D_status = _single_interval_status(d_one, budget=tau_l)
    chart_status = {
        V1PhysicalActionIntervalStatus.CERTIFIED: (
            V1PhysicalActionChartTransportStatus.CHART_CERTIFIED
        ),
        V1PhysicalActionIntervalStatus.REFUTED: (
            V1PhysicalActionChartTransportStatus.CHART_REFUTED
        ),
        V1PhysicalActionIntervalStatus.UNRESOLVED: (
            V1PhysicalActionChartTransportStatus.CHART_UNRESOLVED
        ),
    }[R_status]
    basis: PhysicalActionPostCAdmissionBasis | None = None
    classification: PhysicalActionPostCEndpointClassification | None = None
    if D_status is V1PhysicalActionIntervalStatus.CERTIFIED:
        basis = "direct_D_finite_endpoint"
        classification = (
            "chart_certified"
            if chart_status is V1PhysicalActionChartTransportStatus.CHART_CERTIFIED
            else "direct_D_exactified"
        )
    return V1PhysicalActionSixRoleActionCertificate(
        coordinate=coordinate,
        f_native_interval=f_interval,
        q_j_interval=q_interval,
        r_half_interval=r_half,
        r_one_interval=r_one,
        d_half_interval=d_half,
        d_one_interval=d_one,
        chart_half_interval=chart_half,
        chart_one_interval=chart_one,
        tau_l=tau_l,
        Q_status=Q_status,
        R_slack_status=R_status,
        direct_D_status=D_status,
        chart_transport_status=chart_status,
        admission_basis=basis,
        endpoint_classification=classification,
    )


def _resolve_prefrozen_action_queue(
    *,
    receipt: V1PhysicalActionSixRoleCertificationReceipt,
    decision_design: V1PhysicalActionSixRolePostCDecisionDesign,
) -> V1PhysicalActionSixRoleDecision:
    """Resolve only the already-frozen independent-action total order."""

    certificates = receipt.action_certificates
    certified = V1PhysicalActionIntervalStatus.CERTIFIED
    refuted = V1PhysicalActionIntervalStatus.REFUTED
    unresolved = V1PhysicalActionIntervalStatus.UNRESOLVED

    selected: V1PhysicalActionSixRoleActionCertificate | None = None
    decision_blockers: list[str] = []
    for certificate in certificates:
        status = certificate.direct_D_status
        if status is certified:
            selected = certificate
            break
        if status is unresolved:
            decision_blockers.append(
                certificate.coordinate.physical_action_entry_hash
            )
        elif status is not refuted:  # pragma: no cover - exact enum exhaustiveness
            _refuse(
                V1PhysicalActionPostCRefusalCode.UNSTABLE_DECISION,
                "unknown direct-D interval status",
            )

    if selected is None:
        if decision_blockers:
            return V1PhysicalActionSixRoleDecision(
                certification_receipt_hash=receipt.receipt_hash,
                decision_design_hash=decision_design.design_hash,
                status="acquire_more_C",
                selected_action_coordinate=None,
                selected_certificate_hash=None,
                fallback_action_coordinates=(),
                fallback_certificate_hashes=(),
                unresolved_decision_action_entry_hashes=tuple(decision_blockers),
                blocked_fallback_action_entry_hash=None,
                fallback_exhaustion_status="blocked_on_more_C",
                stable_action_evidence=False,
            )
        return V1PhysicalActionSixRoleDecision(
            certification_receipt_hash=receipt.receipt_hash,
            decision_design_hash=decision_design.design_hash,
            status="stable_no_action",
            selected_action_coordinate=None,
            selected_certificate_hash=None,
            fallback_action_coordinates=(),
            fallback_certificate_hashes=(),
            unresolved_decision_action_entry_hashes=(),
            blocked_fallback_action_entry_hash=None,
            fallback_exhaustion_status="fully_resolved",
            stable_action_evidence=False,
        )

    # Any unresolved action before the first certified action can still become
    # the first admissible action and therefore changes the irreversible choice.
    if decision_blockers:
        return V1PhysicalActionSixRoleDecision(
            certification_receipt_hash=receipt.receipt_hash,
            decision_design_hash=decision_design.design_hash,
            status="acquire_more_C",
            selected_action_coordinate=None,
            selected_certificate_hash=None,
            fallback_action_coordinates=(),
            fallback_certificate_hashes=(),
            unresolved_decision_action_entry_hashes=tuple(decision_blockers),
            blocked_fallback_action_entry_hash=None,
            fallback_exhaustion_status="blocked_on_more_C",
            stable_action_evidence=False,
        )

    queue: list[V1PhysicalActionSixRoleActionCertificate] = []
    blocked_fallback: str | None = None
    selected_priority = selected.coordinate.pre_C_priority
    for certificate in certificates:
        if certificate.coordinate.pre_C_priority < selected_priority:
            continue
        status = certificate.direct_D_status
        if status is certified:
            queue.append(certificate)
        elif status is unresolved:
            blocked_fallback = certificate.coordinate.physical_action_entry_hash
            break
        elif status is not refuted:  # pragma: no cover - exact enum exhaustiveness
            _refuse(
                V1PhysicalActionPostCRefusalCode.UNSTABLE_DECISION,
                "unknown direct-D interval status in fallback queue",
            )
    if not queue or queue[0].certificate_hash != selected.certificate_hash:
        _refuse(
            V1PhysicalActionPostCRefusalCode.UNSTABLE_DECISION,
            "resolved fallback queue lost its selected physical action",
        )
    return V1PhysicalActionSixRoleDecision(
        certification_receipt_hash=receipt.receipt_hash,
        decision_design_hash=decision_design.design_hash,
        status="stable_action",
        selected_action_coordinate=selected.coordinate,
        selected_certificate_hash=selected.certificate_hash,
        fallback_action_coordinates=tuple(value.coordinate for value in queue),
        fallback_certificate_hashes=tuple(value.certificate_hash for value in queue),
        unresolved_decision_action_entry_hashes=(),
        blocked_fallback_action_entry_hash=blocked_fallback,
        fallback_exhaustion_status=(
            "blocked_on_more_C" if blocked_fallback is not None else "fully_resolved"
        ),
        stable_action_evidence=True,
    )


def certify_v1_signed_work_physical_action_six_role_post_C_decision(
    *,
    six_role_C_design: object,
    progress_receipt: object,
    decision_design: object,
    live_chart_identity_hash: str,
    live_current_parent_mask_hash: str,
    live_route_axis_hash: str,
    live_proposal_generation: int,
) -> tuple[
    V1PhysicalActionSixRoleCertificationReceipt,
    V1PhysicalActionSixRoleDecision,
]:
    """Replay exact physical-C evidence and resolve its pre-frozen action order."""

    if (
        type(six_role_C_design) is not V1PhysicalActionSixRoleCDesign
        or type(progress_receipt) is not V1PhysicalActionSixRoleProgressReceipt
        or type(decision_design) is not V1PhysicalActionSixRolePostCDecisionDesign
    ):
        _refuse(
            V1PhysicalActionPostCRefusalCode.INVALID_TYPE,
            "post-C certification requires exact design/progress/policy types",
        )
    design = six_role_C_design
    progress = progress_receipt
    policy = decision_design
    try:
        design.assert_integrity(policy.physical_action_six_role_C_design_hash)
        progress.assert_replays(design)
    except V1PhysicalActionSixRoleCRefusal as error:
        raise V1PhysicalActionPostCRefusal(
            V1PhysicalActionPostCRefusalCode.HASH_MISMATCH,
            "physical-action C design/progress failed exact source replay",
        ) from error
    except (TypeError, ValueError, RuntimeError) as error:
        raise V1PhysicalActionPostCRefusal(
            V1PhysicalActionPostCRefusalCode.HASH_MISMATCH,
            "physical-action C source graph is stale or tampered",
        ) from error
    expected_policy = (
        register_v1_signed_work_physical_action_six_role_post_C_decision_design(
            six_role_C_design=design,
            tau_l=policy.tau_l,
            tau_l_source_hash=policy.tau_l_source_hash,
        )
    )
    policy.assert_integrity()
    if policy.design_hash != expected_policy.design_hash:
        _refuse(
            V1PhysicalActionPostCRefusalCode.HASH_MISMATCH,
            "post-C decision design does not replay its exact pre-outcome source",
        )
    live = (
        _identity(live_chart_identity_hash, name="live chart identity"),
        _identity(live_current_parent_mask_hash, name="live parent-mask hash"),
        _identity(live_route_axis_hash, name="live route-axis hash"),
        _nonnegative_int(
            live_proposal_generation,
            name="live proposal generation",
        ),
    )
    registered = (
        policy.chart_identity_hash,
        policy.current_parent_mask_hash,
        policy.route_axis_hash,
        policy.proposal_generation,
    )
    if live != registered:
        _refuse(
            V1PhysicalActionPostCRefusalCode.STALE_IDENTITY,
            "physical-action C evidence is stale for the live chart/mask/generation",
        )
    if (
        progress.physical_action_axis_hash != policy.physical_action_axis_hash
        or progress.source_registration_hash != policy.source_registration_hash
        or progress.direction_provider_receipt_hash
        != policy.direction_provider_receipt_hash
        or progress.chart_identity_hash != policy.chart_identity_hash
        or progress.current_parent_mask_hash != policy.current_parent_mask_hash
        or progress.route_axis_hash != policy.route_axis_hash
        or progress.route_count != policy.route_count
        or progress.proposal_generation != policy.proposal_generation
    ):
        _refuse(
            V1PhysicalActionPostCRefusalCode.STALE_IDENTITY,
            "post-C design and progress name different chart/action identities",
        )
    summaries = _validated_action_summaries(design=design, progress=progress)
    if tuple(value.physical_action_entry_hash for value in summaries) != tuple(
        value.physical_action_entry_hash for value in policy.action_coordinates
    ):
        _refuse(
            V1PhysicalActionPostCRefusalCode.ACTION_AXIS_MISMATCH,
            "C summaries left the exact pre-frozen physical-action axis",
        )
    certificates = tuple(
        _certificate(coordinate=coordinate, summary=summary, tau_l=policy.tau_l)
        for coordinate, summary in zip(
            policy.action_coordinates,
            summaries,
            strict=True,
        )
    )
    receipt = V1PhysicalActionSixRoleCertificationReceipt(
        decision_design_hash=policy.design_hash,
        physical_action_six_role_C_design_hash=design.design_hash,
        physical_action_six_role_C_progress_receipt_hash=progress.receipt_hash,
        physical_action_axis_hash=policy.physical_action_axis_hash,
        source_registration_hash=policy.source_registration_hash,
        chart_identity_hash=policy.chart_identity_hash,
        current_parent_mask_hash=policy.current_parent_mask_hash,
        route_axis_hash=policy.route_axis_hash,
        route_count=policy.route_count,
        proposal_generation=policy.proposal_generation,
        tau_l=policy.tau_l,
        action_certificates=certificates,
        sampling_status=progress.sampling_status,
    )
    decision = _resolve_prefrozen_action_queue(
        receipt=receipt,
        decision_design=policy,
    )
    return receipt, decision


__all__ = [
    "PHYSICAL_ACTION_SIX_ROLE_DECISION_AUTHORITY",
    "PHYSICAL_ACTION_SIX_ROLE_DECISION_RULE",
    "V1PhysicalActionChartTransportStatus",
    "V1PhysicalActionIntervalStatus",
    "V1PhysicalActionPostCActionCoordinate",
    "V1PhysicalActionPostCRefusal",
    "V1PhysicalActionPostCRefusalCode",
    "V1PhysicalActionSixRoleActionCertificate",
    "V1PhysicalActionSixRoleCertificationReceipt",
    "V1PhysicalActionSixRoleDecision",
    "V1PhysicalActionSixRolePostCDecisionDesign",
    "certify_v1_signed_work_physical_action_six_role_post_C_decision",
    "register_v1_signed_work_physical_action_six_role_post_C_decision_design",
]
