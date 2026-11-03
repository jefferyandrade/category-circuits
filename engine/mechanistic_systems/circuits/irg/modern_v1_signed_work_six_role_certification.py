"""Typed C-stage decision authority for canonical signed-work modern v1.

This module is the versioned successor to the legacy F/Q/R feasibility
certifier.  It consumes the independent six-observable C receipt and keeps
three roles separate:

* native-fiber signed ``F`` is objective evidence, never a loss-derived veto;
* direct finite displacement ``D`` owns response-budget admission;
* ``Q_J + R`` labels the fidelity of the current tangent chart.

A directly certified finite displacement may therefore exactify a loose
triangle envelope.  Such an endpoint is explicitly labelled
``direct_D_exactified`` and must relinearize after an oracle-accepted quench;
it receives no claim that the old chart remained valid.  This contract grants
only one pre-frozen proposal.  Calibration, oracle, commit, terminal, and
Pareto authority remain closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral, Real
from typing import Final, Literal, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_fqr_sequential_acquisition import (
    _downward_weighted_sum,
    _outward_weighted_sum,
    _response_burden,
    _response_burden_lower,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_signed_work_six_role_c_acquisition import (
    D_HALF_ROLE,
    D_ONE_ROLE,
    F_NATIVE_ROLE,
    Q_J_ROLE,
    R_HALF_ROLE,
    R_ONE_ROLE,
    SIGNED_WORK_SIX_ROLES,
    V1SignedWorkSixRoleBranchDesign,
    V1SignedWorkSixRoleBranchProgress,
    V1SignedWorkSixRoleCDesign,
    V1SignedWorkSixRoleCellSummary,
    V1SignedWorkSixRoleEndpointSummary,
    V1SignedWorkSixRoleProgressReceipt,
    _role_bound_matrix,
)


SIGNED_WORK_SIX_ROLE_DECISION_DESIGN_SCHEMA: Final = (
    "partition_free_v1_signed_work_direct_D_chart_QR_decision_design_v1"
)
SIGNED_WORK_SIX_ROLE_ENDPOINT_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_signed_work_direct_D_chart_QR_endpoint_certificate_v1"
)
SIGNED_WORK_SIX_ROLE_CERTIFICATION_SCHEMA: Final = (
    "partition_free_v1_signed_work_direct_D_chart_QR_certification_v1"
)
SIGNED_WORK_SIX_ROLE_DECISION_SCHEMA: Final = (
    "partition_free_v1_signed_work_direct_D_chart_QR_decision_v1"
)
SIGNED_WORK_SIX_ROLE_AUTHORIZATION_SCHEMA: Final = (
    "partition_free_v1_signed_work_direct_D_chart_QR_single_proposal_v1"
)
SIGNED_WORK_SIX_ROLE_FALLBACK_AUTHORIZATION_SCHEMA: Final = (
    "partition_free_v1_signed_work_direct_D_chart_QR_pre_O_fallback_queue_v1"
)
SIGNED_WORK_SIX_ROLE_DECISION_RULE: Final = (
    "max_registered_resource_direct_D_certified_all_higher_or_prior_tie_"
    "coordinates_direct_D_refuted_v1"
)
SIGNED_WORK_SIX_ROLE_AUTHORITY_SCOPE: Final = (
    "independent_C_direct_D_finite_response_admission_QR_chart_label_"
    "single_prefrozen_proposal_only_no_K_O_commit_terminal_or_pareto_authority_v1"
)
SIGNED_WORK_SIX_ROLE_FALLBACK_AUTHORITY_SCOPE: Final = (
    "independent_C_direct_D_finite_response_admission_QR_chart_label_"
    "prefrozen_pre_O_fallback_queue_no_K_O_commit_terminal_or_pareto_authority_v1"
)

_DESIGN_PREFIX: Final = "modern-v1-signed-six-role-decision-design-"
_COORDINATE_PREFIX: Final = "modern-v1-signed-six-role-coordinate-"
_CERTIFICATE_PREFIX: Final = "modern-v1-signed-six-role-endpoint-certificate-"
_RECEIPT_PREFIX: Final = "modern-v1-signed-six-role-certification-"
_DECISION_PREFIX: Final = "modern-v1-signed-six-role-decision-"
_AUTHORIZATION_PREFIX: Final = "modern-v1-signed-six-role-authorization-"
_FALLBACK_AUTHORIZATION_PREFIX: Final = (
    "modern-v1-signed-six-role-fallback-authorization-"
)

SixRoleDecisionStatus = Literal[
    "acquire_more_C",
    "stable_endpoint",
    "stable_no_endpoint",
]
SixRoleAdmissionBasis = Literal["chart_certified", "direct_D_exactified"]
SixRoleFallbackExhaustionStatus = Literal[
    "blocked_on_more_C",
    "fully_resolved_no_more_candidate",
]


class V1SignedWorkIntervalStatus(str, Enum):
    CERTIFIED = "certified"
    REFUTED = "refuted"
    UNRESOLVED = "unresolved"


class V1SignedWorkChartTransportStatus(str, Enum):
    CHART_CERTIFIED = "chart_certified"
    CHART_REFUTED = "chart_refuted"
    CHART_UNRESOLVED = "chart_unresolved"


class V1SignedWorkSixRoleDecisionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    HASH_MISMATCH = "hash_mismatch"
    STALE_CHART = "stale_chart"
    COORDINATE_MISMATCH = "coordinate_mismatch"
    POST_SELECTION = "post_selection"
    UNSTABLE_DECISION = "unstable_decision"
    NO_CERTIFIED_ENDPOINT = "no_certified_endpoint"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkSixRoleDecisionRefusal(ValueError):
    def __init__(
        self,
        code: V1SignedWorkSixRoleDecisionRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkSixRoleDecisionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkSixRoleDecisionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one exact positive integer",
        )
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one exact nonnegative integer",
        )
    return int(value)


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be finite and positive",
        )
    return result


def _interval_status(
    *,
    lower_by_dose: tuple[float, float],
    upper_by_dose: tuple[float, float],
    tau_l: float,
) -> V1SignedWorkIntervalStatus:
    budgets = (0.5 * tau_l, tau_l)
    if all(upper <= budget for upper, budget in zip(upper_by_dose, budgets)):
        return V1SignedWorkIntervalStatus.CERTIFIED
    if any(lower > budget for lower, budget in zip(lower_by_dose, budgets)):
        return V1SignedWorkIntervalStatus.REFUTED
    return V1SignedWorkIntervalStatus.UNRESOLVED


def _endpoint_interval_status(
    *,
    lower: float,
    upper: float,
    tau_l: float,
) -> V1SignedWorkIntervalStatus:
    """Classify the committed hard endpoint, independent of path probes."""

    if upper <= tau_l:
        return V1SignedWorkIntervalStatus.CERTIFIED
    if lower > tau_l:
        return V1SignedWorkIntervalStatus.REFUTED
    return V1SignedWorkIntervalStatus.UNRESOLVED


@dataclass(frozen=True, slots=True)
class V1SignedWorkSixRoleEndpointCoordinate:
    """One C-eligible endpoint in a total order frozen before C outcomes."""

    pre_C_priority: int
    branch: V1RelaxedKKTBranch
    endpoint_index: int
    endpoint_label: int
    action_hash: str
    canonical_candidate_identity_hash: str
    cumulative_removed_count: int
    crosswalk_receipt_hash: str
    hard_replay_endpoint_receipt_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pre_C_priority",
            _positive_int(self.pre_C_priority, name="pre-C priority"),
        )
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
                "endpoint coordinate requires the exact KKT branch enum",
            )
        for name in (
            "endpoint_index",
            "endpoint_label",
            "cumulative_removed_count",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        for name in (
            "action_hash",
            "canonical_candidate_identity_hash",
            "crosswalk_receipt_hash",
            "hard_replay_endpoint_receipt_hash",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name)
            )

    @property
    def coordinate_hash(self) -> str:
        return content_hash(self, prefix=_COORDINATE_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkSixRoleDecisionDesign:
    """Loss-unit budget and endpoint order registered before C opens."""

    six_role_C_design_hash: str
    bridge_family_bundle_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    tau_l: float
    tau_l_source_hash: str
    endpoint_coordinates: tuple[V1SignedWorkSixRoleEndpointCoordinate, ...]
    complete_source_endpoint_decision_hashes: tuple[str, ...]
    deterministic_pre_C_closed_endpoint_count: int
    doses: tuple[float, float] = (0.5, 1.0)
    signed_F_is_objective_not_feasibility: bool = True
    direct_D_owns_finite_response_admission: bool = True
    QR_transport_is_label_not_hidden_C_veto: bool = True
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    decision_rule: str = SIGNED_WORK_SIX_ROLE_DECISION_RULE
    authority_scope: str = SIGNED_WORK_SIX_ROLE_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_SIX_ROLE_DECISION_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "six_role_C_design_hash",
            "bridge_family_bundle_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "tau_l_source_hash",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name)
            )
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route count")
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        object.__setattr__(self, "tau_l", _positive_finite(self.tau_l, name="tau_L"))
        coordinates = tuple(self.endpoint_coordinates)
        if not coordinates or any(
            type(value) is not V1SignedWorkSixRoleEndpointCoordinate
            for value in coordinates
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
                "decision design requires exact nonempty endpoint coordinates",
            )
        if tuple(value.pre_C_priority for value in coordinates) != tuple(
            range(1, len(coordinates) + 1)
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.POST_SELECTION,
                "pre-C endpoint priority must be one exact total order",
            )
        decisions = tuple(
            _identity(value, name="source endpoint decision hash")
            for value in self.complete_source_endpoint_decision_hashes
        )
        if not decisions:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                "complete source endpoint decision axis is empty",
            )
        object.__setattr__(self, "endpoint_coordinates", coordinates)
        object.__setattr__(self, "complete_source_endpoint_decision_hashes", decisions)
        object.__setattr__(
            self,
            "deterministic_pre_C_closed_endpoint_count",
            _nonnegative_int(
                self.deterministic_pre_C_closed_endpoint_count,
                name="deterministically closed endpoint count",
            ),
        )
        required_true = (
            self.signed_F_is_objective_not_feasibility,
            self.direct_D_owns_finite_response_admission,
            self.QR_transport_is_label_not_hidden_C_veto,
            self.issued_before_C_outcomes,
        )
        if any(type(value) is not bool or not value for value in required_true):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "decision design changed the canonical role split",
            )
        if type(self.C_outcomes_seen) is not bool or self.C_outcomes_seen:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.POST_SELECTION,
                "decision design must be issued before C outcomes",
            )
        if (
            self.doses != (0.5, 1.0)
            or self.decision_rule != SIGNED_WORK_SIX_ROLE_DECISION_RULE
            or self.authority_scope != SIGNED_WORK_SIX_ROLE_AUTHORITY_SCOPE
            or self.schema != SIGNED_WORK_SIX_ROLE_DECISION_DESIGN_SCHEMA
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "unsupported six-role decision semantics",
            )

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_DESIGN_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkSixRoleEndpointCertificate:
    coordinate: V1SignedWorkSixRoleEndpointCoordinate
    f_native_interval: tuple[float, float]
    q_j_interval: tuple[float, float]
    r_half_interval: tuple[float, float]
    r_one_interval: tuple[float, float]
    d_half_interval: tuple[float, float]
    d_one_interval: tuple[float, float]
    chart_half_interval: tuple[float, float]
    chart_one_interval: tuple[float, float]
    tau_l: float
    direct_D_status: V1SignedWorkIntervalStatus
    chart_transport_status: V1SignedWorkChartTransportStatus
    admission_basis: SixRoleAdmissionBasis | None
    signed_F_used_as_feasibility_veto: bool = False
    schema: str = SIGNED_WORK_SIX_ROLE_ENDPOINT_CERTIFICATE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.coordinate) is not V1SignedWorkSixRoleEndpointCoordinate:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
                "endpoint certificate requires its exact pre-C coordinate",
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
            value = tuple(float(item) for item in getattr(self, name))
            if (
                len(value) != 2
                or any(not math.isfinite(item) or item < 0.0 for item in value)
                or value[0] > value[1]
            ):
                _refuse(
                    V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                    f"{name} is not a finite nonnegative interval",
                )
            object.__setattr__(self, name, value)
        object.__setattr__(self, "tau_l", _positive_finite(self.tau_l, name="tau_L"))
        if type(self.direct_D_status) is not V1SignedWorkIntervalStatus:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
                "direct-D status must use the exact enum",
            )
        if type(self.chart_transport_status) is not V1SignedWorkChartTransportStatus:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
                "chart status must use the exact enum",
            )
        expected_chart_half = (
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
        expected_chart_one = (
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
        if (
            self.chart_half_interval != expected_chart_half
            or self.chart_one_interval != expected_chart_one
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                "endpoint chart intervals do not replay from Q/R",
            )
        expected_direct = _endpoint_interval_status(
            lower=self.d_one_interval[0],
            upper=self.d_one_interval[1],
            tau_l=self.tau_l,
        )
        expected_chart_interval = _interval_status(
            lower_by_dose=(expected_chart_half[0], expected_chart_one[0]),
            upper_by_dose=(expected_chart_half[1], expected_chart_one[1]),
            tau_l=self.tau_l,
        )
        expected_chart = {
            V1SignedWorkIntervalStatus.CERTIFIED: (
                V1SignedWorkChartTransportStatus.CHART_CERTIFIED
            ),
            V1SignedWorkIntervalStatus.REFUTED: (
                V1SignedWorkChartTransportStatus.CHART_REFUTED
            ),
            V1SignedWorkIntervalStatus.UNRESOLVED: (
                V1SignedWorkChartTransportStatus.CHART_UNRESOLVED
            ),
        }[expected_chart_interval]
        if (
            self.direct_D_status is not expected_direct
            or self.chart_transport_status is not expected_chart
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "endpoint statuses do not replay from registered intervals and tau_L",
            )
        expected_basis: SixRoleAdmissionBasis | None = None
        if self.direct_D_status is V1SignedWorkIntervalStatus.CERTIFIED:
            expected_basis = (
                "chart_certified"
                if self.chart_transport_status
                is V1SignedWorkChartTransportStatus.CHART_CERTIFIED
                else "direct_D_exactified"
            )
        if self.admission_basis != expected_basis:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "admission basis does not follow direct-D/chart status",
            )
        if (
            type(self.signed_F_used_as_feasibility_veto) is not bool
            or self.signed_F_used_as_feasibility_veto
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "signed F may not become a loss-derived feasibility veto",
            )
        if self.schema != SIGNED_WORK_SIX_ROLE_ENDPOINT_CERTIFICATE_SCHEMA:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "unsupported endpoint-certificate schema",
            )

    @property
    def certificate_hash(self) -> str:
        return content_hash(self, prefix=_CERTIFICATE_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkSixRoleCertificationReceipt:
    decision_design_hash: str
    six_role_C_design_hash: str
    six_role_C_progress_receipt_hash: str
    bridge_family_bundle_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    tau_l: float
    endpoint_certificates: tuple[V1SignedWorkSixRoleEndpointCertificate, ...]
    sampling_status: str
    signed_F_used_as_feasibility_veto: bool = False
    direct_D_is_functional_admission_authority: bool = True
    QR_is_chart_label_not_hidden_veto: bool = True
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_SIX_ROLE_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_SIX_ROLE_CERTIFICATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "decision_design_hash",
            "six_role_C_design_hash",
            "six_role_C_progress_receipt_hash",
            "bridge_family_bundle_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name)
            )
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route count")
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        object.__setattr__(self, "tau_l", _positive_finite(self.tau_l, name="tau_L"))
        certificates = tuple(self.endpoint_certificates)
        if not certificates or any(
            type(value) is not V1SignedWorkSixRoleEndpointCertificate
            for value in certificates
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
                "certification requires exact endpoint certificates",
            )
        object.__setattr__(self, "endpoint_certificates", certificates)
        if any(value.tau_l != self.tau_l for value in certificates):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                "endpoint certificate tau_L differs from its family receipt",
            )
        _identity(self.sampling_status, name="sampling status")
        if (
            self.signed_F_used_as_feasibility_veto
            or not self.direct_D_is_functional_admission_authority
            or not self.QR_is_chart_label_not_hidden_veto
            or any(
                (
                    self.calibration_K_open_authority,
                    self.paired_oracle_O_open_authority,
                    self.mask_commit_authority,
                    self.terminal_authority,
                    self.pareto_authority,
                )
            )
            or self.authority_scope != SIGNED_WORK_SIX_ROLE_AUTHORITY_SCOPE
            or self.schema != SIGNED_WORK_SIX_ROLE_CERTIFICATION_SCHEMA
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "certification crossed its evidence boundary",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_RECEIPT_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkSixRoleDecision:
    certification_receipt_hash: str
    decision_design_hash: str
    status: SixRoleDecisionStatus
    selected_coordinate_hash: str | None
    selected_branch: V1RelaxedKKTBranch | None
    selected_endpoint_index: int | None
    selected_action_hash: str | None
    selected_cumulative_removed_count: int | None
    admission_basis: SixRoleAdmissionBasis | None
    chart_transport_status: V1SignedWorkChartTransportStatus | None
    unresolved_coordinate_hashes: tuple[str, ...]
    single_endpoint_proposal_authority: bool
    candidate_composition_mutable_after_C: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    paired_bce_oracle_still_required: bool = True
    decision_rule: str = SIGNED_WORK_SIX_ROLE_DECISION_RULE
    authority_scope: str = SIGNED_WORK_SIX_ROLE_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_SIX_ROLE_DECISION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "certification_receipt_hash",
            _identity(self.certification_receipt_hash, name="certification receipt hash"),
        )
        object.__setattr__(
            self,
            "decision_design_hash",
            _identity(self.decision_design_hash, name="decision design hash"),
        )
        if self.status not in {
            "acquire_more_C",
            "stable_endpoint",
            "stable_no_endpoint",
        }:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.UNSTABLE_DECISION,
                "unknown six-role decision status",
            )
        selected = (
            self.selected_coordinate_hash,
            self.selected_branch,
            self.selected_endpoint_index,
            self.selected_action_hash,
            self.selected_cumulative_removed_count,
            self.admission_basis,
            self.chart_transport_status,
        )
        if self.status == "stable_endpoint":
            if any(value is None for value in selected):
                _refuse(
                    V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                    "stable endpoint decision is incomplete",
                )
            _identity(self.selected_coordinate_hash, name="selected coordinate hash")
            if type(self.selected_branch) is not V1RelaxedKKTBranch:
                _refuse(
                    V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
                    "selected branch must use the exact enum",
                )
            _positive_int(self.selected_endpoint_index, name="selected endpoint index")
            _identity(self.selected_action_hash, name="selected action hash")
            _positive_int(
                self.selected_cumulative_removed_count,
                name="selected cumulative removed count",
            )
            if self.admission_basis not in {"chart_certified", "direct_D_exactified"}:
                _refuse(
                    V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                    "stable endpoint lacks an exact admission basis",
                )
            if type(self.chart_transport_status) is not V1SignedWorkChartTransportStatus:
                _refuse(
                    V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
                    "stable endpoint lacks an exact chart status",
                )
            if not self.single_endpoint_proposal_authority:
                _refuse(
                    V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                    "stable endpoint must grant exactly proposal authority",
                )
        elif (
            any(value is not None for value in selected)
            or self.single_endpoint_proposal_authority
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "non-endpoint decision may not name or authorize an action",
            )
        unresolved = tuple(
            _identity(value, name="unresolved coordinate hash")
            for value in self.unresolved_coordinate_hashes
        )
        if self.status == "acquire_more_C" and not unresolved:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.UNSTABLE_DECISION,
                "acquire-more-C decision must name its blockers",
            )
        if self.status != "acquire_more_C" and unresolved:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.UNSTABLE_DECISION,
                "stable decision retained decision-changing blockers",
            )
        object.__setattr__(self, "unresolved_coordinate_hashes", unresolved)
        closed = (
            not self.candidate_composition_mutable_after_C,
            not self.calibration_K_open_authority,
            not self.paired_oracle_O_open_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            self.paired_bce_oracle_still_required,
        )
        if any(type(value) is not bool or not value for value in closed):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "six-role decision crossed proposal-only authority",
            )
        if (
            self.decision_rule != SIGNED_WORK_SIX_ROLE_DECISION_RULE
            or self.authority_scope != SIGNED_WORK_SIX_ROLE_AUTHORITY_SCOPE
            or self.schema != SIGNED_WORK_SIX_ROLE_DECISION_SCHEMA
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "unsupported decision rule or schema",
            )

    @property
    def decision_hash(self) -> str:
        return content_hash(self, prefix=_DECISION_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkSixRoleProposalAuthorization:
    certification_receipt_hash: str
    decision_hash: str
    decision_design_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    selected_coordinate: V1SignedWorkSixRoleEndpointCoordinate
    admission_basis: SixRoleAdmissionBasis
    chart_transport_status: V1SignedWorkChartTransportStatus
    single_endpoint_proposal_authority: bool = True
    must_materialize_from_frozen_device_prefix: bool = True
    complete_ladder_mask_materialization_forbidden: bool = True
    relinearization_required_after_oracle_accept: bool = True
    old_chart_validity_claim: bool = False
    omitted_mask_optimality_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    paired_bce_oracle_still_required: bool = True
    authority_scope: str = SIGNED_WORK_SIX_ROLE_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_SIX_ROLE_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "certification_receipt_hash",
            "decision_hash",
            "decision_design_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name)
            )
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route count")
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        if type(self.selected_coordinate) is not V1SignedWorkSixRoleEndpointCoordinate:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
                "authorization requires an exact selected coordinate",
            )
        expected_old_chart_claim = self.admission_basis == "chart_certified"
        if (
            self.admission_basis not in {"chart_certified", "direct_D_exactified"}
            or type(self.chart_transport_status) is not V1SignedWorkChartTransportStatus
            or self.old_chart_validity_claim is not expected_old_chart_claim
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "authorization overclaimed the tangent chart",
            )
        required_true = (
            self.single_endpoint_proposal_authority,
            self.must_materialize_from_frozen_device_prefix,
            self.complete_ladder_mask_materialization_forbidden,
            self.relinearization_required_after_oracle_accept,
            self.paired_bce_oracle_still_required,
        )
        required_false = (
            self.omitted_mask_optimality_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool or not value for value in required_true)
            or any(type(value) is not bool or value for value in required_false)
            or self.authority_scope != SIGNED_WORK_SIX_ROLE_AUTHORITY_SCOPE
            or self.schema != SIGNED_WORK_SIX_ROLE_AUTHORIZATION_SCHEMA
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "authorization crossed its single-proposal boundary",
            )

    @property
    def authorization_hash(self) -> str:
        return content_hash(self, prefix=_AUTHORIZATION_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkSixRoleFallbackQueueAuthorization:
    """Pre-O fallback queue over already frozen, direct-D-certified endpoints.

    The singleton proposal authorization is enough to try one endpoint.  This
    companion object records the compact causal information needed to try a
    descending sequence after paired-O rejection without changing composition
    from the oracle outcome.  The queue is a prefix of the pre-C resource/priority
    decision order after removing refuted endpoints and quotienting duplicate
    physical hard candidates by their first representative coordinate.  It stops
    before the first unresolved, not-yet-covered physical candidate.
    """

    certification_receipt_hash: str
    decision_hash: str
    decision_design_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    selected_coordinate: V1SignedWorkSixRoleEndpointCoordinate
    fallback_coordinates: tuple[V1SignedWorkSixRoleEndpointCoordinate, ...]
    fallback_certificate_hashes: tuple[str, ...]
    fallback_admission_bases: tuple[SixRoleAdmissionBasis, ...]
    fallback_chart_transport_statuses: tuple[V1SignedWorkChartTransportStatus, ...]
    blocked_unresolved_coordinate_hash: str | None
    queued_physical_candidate_identity_hashes: tuple[str, ...]
    fallback_queue_proposal_authority: bool = True
    resolved_prefix_queue_authorized_before_K_O: bool = True
    fallback_exhaustion_status: SixRoleFallbackExhaustionStatus = (
        "fully_resolved_no_more_candidate"
    )
    stops_before_first_unresolved_decision_changing_coordinate: bool = True
    duplicate_physical_actions_quotiented_by_first_coordinate: bool = True
    duplicate_physical_candidates_replay_identically: bool = True
    must_materialize_from_frozen_device_prefix: bool = True
    complete_ladder_mask_materialization_forbidden: bool = True
    relinearization_required_after_oracle_accept: bool = True
    old_chart_validity_claims_are_per_endpoint: bool = True
    omitted_mask_optimality_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    paired_bce_oracle_still_required: bool = True
    authority_scope: str = SIGNED_WORK_SIX_ROLE_FALLBACK_AUTHORITY_SCOPE
    schema: str = SIGNED_WORK_SIX_ROLE_FALLBACK_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "certification_receipt_hash",
            "decision_hash",
            "decision_design_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
        ):
            object.__setattr__(
                self, name, _identity(getattr(self, name), name=name)
            )
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route count")
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        if type(self.selected_coordinate) is not V1SignedWorkSixRoleEndpointCoordinate:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
                "fallback authorization requires an exact selected coordinate",
            )
        coordinates = tuple(self.fallback_coordinates)
        if not coordinates or any(
            type(value) is not V1SignedWorkSixRoleEndpointCoordinate
            for value in coordinates
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
                "fallback authorization requires exact nonempty coordinates",
            )
        if coordinates[0].coordinate_hash != self.selected_coordinate.coordinate_hash:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                "fallback queue must start with the selected coordinate",
            )
        order_key = tuple(
            (-value.cumulative_removed_count, value.pre_C_priority)
            for value in coordinates
        )
        if order_key != tuple(sorted(order_key)):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.POST_SELECTION,
                "fallback coordinates are not in pre-C resource/priority order",
            )
        coord_hashes = tuple(value.coordinate_hash for value in coordinates)
        if len(set(coord_hashes)) != len(coord_hashes):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                "fallback coordinates contain duplicates",
            )
        certificates = tuple(
            _identity(value, name="fallback certificate hash")
            for value in self.fallback_certificate_hashes
        )
        bases = tuple(self.fallback_admission_bases)
        chart = tuple(self.fallback_chart_transport_statuses)
        physical = tuple(
            _identity(value, name="physical candidate identity")
            for value in self.queued_physical_candidate_identity_hashes
        )
        if (
            len(certificates) != len(coordinates)
            or len(bases) != len(coordinates)
            or len(chart) != len(coordinates)
            or len(physical) != len(coordinates)
            or len(set(physical)) != len(physical)
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                "fallback queue axes have inconsistent length or duplicate hard candidates",
            )
        for basis, status in zip(bases, chart, strict=True):
            if basis not in {"chart_certified", "direct_D_exactified"}:
                _refuse(
                    V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                    "fallback endpoint lacks an exact admission basis",
                )
            if type(status) is not V1SignedWorkChartTransportStatus:
                _refuse(
                    V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
                    "fallback endpoint lacks an exact chart status",
                )
            if (
                basis == "chart_certified"
                and status is not V1SignedWorkChartTransportStatus.CHART_CERTIFIED
            ):
                _refuse(
                    V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                    "fallback endpoint overclaimed chart validity",
                )
        if self.blocked_unresolved_coordinate_hash is not None:
            _identity(
                self.blocked_unresolved_coordinate_hash,
                name="blocked unresolved coordinate hash",
            )
        required_true = (
            self.fallback_queue_proposal_authority,
            self.resolved_prefix_queue_authorized_before_K_O,
            self.stops_before_first_unresolved_decision_changing_coordinate,
            self.duplicate_physical_actions_quotiented_by_first_coordinate,
            self.duplicate_physical_candidates_replay_identically,
            self.must_materialize_from_frozen_device_prefix,
            self.complete_ladder_mask_materialization_forbidden,
            self.relinearization_required_after_oracle_accept,
            self.old_chart_validity_claims_are_per_endpoint,
            self.paired_bce_oracle_still_required,
        )
        required_false = (
            self.omitted_mask_optimality_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(value) is not bool or not value for value in required_true)
            or any(type(value) is not bool or value for value in required_false)
            or self.fallback_exhaustion_status
            not in {"blocked_on_more_C", "fully_resolved_no_more_candidate"}
            or (
                self.blocked_unresolved_coordinate_hash is None
                and self.fallback_exhaustion_status != "fully_resolved_no_more_candidate"
            )
            or (
                self.blocked_unresolved_coordinate_hash is not None
                and self.fallback_exhaustion_status != "blocked_on_more_C"
            )
            or self.authority_scope != SIGNED_WORK_SIX_ROLE_FALLBACK_AUTHORITY_SCOPE
            or self.schema != SIGNED_WORK_SIX_ROLE_FALLBACK_AUTHORIZATION_SCHEMA
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "fallback authorization crossed its proposal-only boundary",
            )
        object.__setattr__(self, "fallback_coordinates", coordinates)
        object.__setattr__(self, "fallback_certificate_hashes", certificates)
        object.__setattr__(self, "fallback_admission_bases", bases)
        object.__setattr__(self, "fallback_chart_transport_statuses", chart)
        object.__setattr__(
            self,
            "queued_physical_candidate_identity_hashes",
            physical,
        )

    @property
    def authorization_hash(self) -> str:
        return content_hash(self, prefix=_FALLBACK_AUTHORIZATION_PREFIX)


def register_v1_signed_work_six_role_decision_design(
    *,
    six_role_C_design: object,
    tau_l: float,
    tau_l_source_hash: str,
) -> V1SignedWorkSixRoleDecisionDesign:
    """Freeze the resource order and budget without inspecting C outcomes."""

    if type(six_role_C_design) is not V1SignedWorkSixRoleCDesign:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
            "decision registration requires the exact six-role C design",
        )
    design = six_role_C_design
    design.assert_integrity()
    first = design.branch_designs[0].secant_design
    coordinates: list[V1SignedWorkSixRoleEndpointCoordinate] = []
    source_decisions: list[str] = []
    closed = 0
    priority = 0
    for branch_design in design.branch_designs:
        bundle = branch_design.bridge_branch
        source_decisions.extend(value.receipt_hash for value in bundle.endpoint_decisions)
        closed += sum(not value.eligible_for_C for value in bundle.endpoint_decisions)
        if len(bundle.crosswalks) != len(branch_design.endpoint_action_hashes):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                "C-eligible crosswalk axis differs from six-role endpoints",
            )
        for endpoint_index, (action_hash, crosswalk) in enumerate(
            zip(branch_design.endpoint_action_hashes, bundle.crosswalks), start=1
        ):
            priority += 1
            if action_hash != crosswalk.frozen_action_hash:
                _refuse(
                    V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                    "six-role endpoint differs from its pre-C crosswalk",
                )
            coordinates.append(
                V1SignedWorkSixRoleEndpointCoordinate(
                    pre_C_priority=priority,
                    branch=branch_design.branch,
                    endpoint_index=endpoint_index,
                    endpoint_label=crosswalk.endpoint_label,
                    action_hash=action_hash,
                    canonical_candidate_identity_hash=(
                        crosswalk.canonical_candidate_identity_hash
                    ),
                    cumulative_removed_count=crosswalk.cumulative_removed_count,
                    crosswalk_receipt_hash=crosswalk.receipt_hash,
                    hard_replay_endpoint_receipt_hash=(
                        crosswalk.hard_replay_endpoint_receipt_hash
                    ),
                )
            )
    return V1SignedWorkSixRoleDecisionDesign(
        six_role_C_design_hash=design.design_hash,
        bridge_family_bundle_hash=design.bridge_family.bundle_hash,
        chart_identity_hash=first.chart_identity_hash,
        current_mask_hash=first.ladder.current_mask_hash,
        route_axis_hash=first.ladder.route_axis_hash,
        route_count=first.ladder.route_count,
        proposal_generation=first.ladder.proposal_generation,
        tau_l=tau_l,
        tau_l_source_hash=tau_l_source_hash,
        endpoint_coordinates=tuple(coordinates),
        complete_source_endpoint_decision_hashes=tuple(source_decisions),
        deterministic_pre_C_closed_endpoint_count=closed,
    )


def _certificate(
    *,
    coordinate: V1SignedWorkSixRoleEndpointCoordinate,
    summary: V1SignedWorkSixRoleEndpointSummary,
    tau_l: float,
) -> V1SignedWorkSixRoleEndpointCertificate:
    if (
        summary.branch is not coordinate.branch
        or summary.endpoint_index != coordinate.endpoint_index
        or summary.action_hash != coordinate.action_hash
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "C endpoint summary left its pre-C coordinate",
        )
    expected_chart_half = (
        _response_burden_lower(
            eta=0.5,
            q_lower=summary.q_j_lower,
            r_lower=summary.r_half_lower,
        ),
        _response_burden(
            eta=0.5,
            q_upper=summary.q_j_upper,
            r_upper=summary.r_half_upper,
        ),
    )
    expected_chart_one = (
        _response_burden_lower(
            eta=1.0,
            q_lower=summary.q_j_lower,
            r_lower=summary.r_one_lower,
        ),
        _response_burden(
            eta=1.0,
            q_upper=summary.q_j_upper,
            r_upper=summary.r_one_upper,
        ),
    )
    if expected_chart_half != (
        summary.chart_half_burden_lower,
        summary.chart_half_burden_upper,
    ) or expected_chart_one != (
        summary.chart_one_burden_lower,
        summary.chart_one_burden_upper,
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "Q/R chart burden does not replay from its registered components",
        )
    direct = _endpoint_interval_status(
        lower=summary.d_one_lower,
        upper=summary.d_one_upper,
        tau_l=tau_l,
    )
    chart_interval_status = _interval_status(
        lower_by_dose=expected_chart_half[:1] + expected_chart_one[:1],
        upper_by_dose=expected_chart_half[1:] + expected_chart_one[1:],
        tau_l=tau_l,
    )
    chart = {
        V1SignedWorkIntervalStatus.CERTIFIED: V1SignedWorkChartTransportStatus.CHART_CERTIFIED,
        V1SignedWorkIntervalStatus.REFUTED: V1SignedWorkChartTransportStatus.CHART_REFUTED,
        V1SignedWorkIntervalStatus.UNRESOLVED: V1SignedWorkChartTransportStatus.CHART_UNRESOLVED,
    }[chart_interval_status]
    basis: SixRoleAdmissionBasis | None = None
    if direct is V1SignedWorkIntervalStatus.CERTIFIED:
        basis = (
            "chart_certified"
            if chart is V1SignedWorkChartTransportStatus.CHART_CERTIFIED
            else "direct_D_exactified"
        )
    return V1SignedWorkSixRoleEndpointCertificate(
        coordinate=coordinate,
        f_native_interval=(summary.f_native_lower, summary.f_native_upper),
        q_j_interval=(summary.q_j_lower, summary.q_j_upper),
        r_half_interval=(summary.r_half_lower, summary.r_half_upper),
        r_one_interval=(summary.r_one_lower, summary.r_one_upper),
        d_half_interval=(summary.d_half_lower, summary.d_half_upper),
        d_one_interval=(summary.d_one_lower, summary.d_one_upper),
        chart_half_interval=expected_chart_half,
        chart_one_interval=expected_chart_one,
        tau_l=tau_l,
        direct_D_status=direct,
        chart_transport_status=chart,
        admission_basis=basis,
    )


def _validated_branch_summaries(
    *,
    family_design: V1SignedWorkSixRoleCDesign,
    branch_design: V1SignedWorkSixRoleBranchDesign,
    branch_progress: V1SignedWorkSixRoleBranchProgress,
) -> tuple[V1SignedWorkSixRoleEndpointSummary, ...]:
    """Rebuild aggregate endpoints from the bounded cell sufficient statistics."""

    if branch_progress.branch_design_hash != branch_design.design_hash:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.HASH_MISMATCH,
            "progress branch belongs to another six-role design",
        )
    stratum_names = branch_design.stratum_names
    endpoints = branch_design.endpoint_action_hashes
    counts = tuple(branch_progress.observed_counts_by_stratum)
    if (
        len(counts) != len(stratum_names)
        or any(
            count < 0 or count > population
            for count, population in zip(counts, branch_design.population_sizes)
        )
        or sum(counts) < 0
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "branch C prefix counts left the registered population",
        )
    expected_cells = tuple(
        (stratum, endpoint_index, action, role)
        for stratum in stratum_names
        for endpoint_index, action in enumerate(endpoints, start=1)
        for role in SIGNED_WORK_SIX_ROLES
    )
    observed_cells = tuple(
        (value.stratum_id, value.endpoint_index, value.action_hash, value.role)
        for value in branch_progress.cell_summaries
    )
    if observed_cells != expected_cells:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.POST_SELECTION,
            "six-role cell axis was reordered, omitted, or post-selected",
        )
    cell_by_key: dict[tuple[str, int, str], V1SignedWorkSixRoleCellSummary] = {}
    support = _role_bound_matrix(branch_design)
    for cell in branch_progress.cell_summaries:
        if type(cell) is not V1SignedWorkSixRoleCellSummary:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
                "branch progress contains a noncanonical C cell",
            )
        stratum_index = stratum_names.index(cell.stratum_id)
        expected_count = counts[stratum_index]
        expected_population = branch_design.population_sizes[stratum_index]
        endpoint_offset = cell.endpoint_index - 1
        role_offset = SIGNED_WORK_SIX_ROLES.index(cell.role)
        expected_bound = float(support[stratum_index, endpoint_offset, role_offset])
        expected_alpha = (
            0.0
            if expected_count in {0, expected_population}
            else family_design.look_alpha(
                branch=branch_design.branch,
                stratum_index=stratum_index,
                sample_size=expected_count,
            )
        )
        if (
            cell.branch is not branch_design.branch
            or cell.sample_size != expected_count
            or cell.population_size != expected_population
            or cell.value_upper_bound != expected_bound
            or cell.current_lower_look_alpha != expected_alpha
            or cell.current_upper_look_alpha != expected_alpha
            or cell.observation_chain_hash
            != branch_progress.branch_observation_chain_hash
        ):
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                "C cell sufficient statistics do not match branch custody",
            )
        cell_by_key[(cell.stratum_id, cell.endpoint_index, cell.role)] = cell
    contexts = branch_design.secant_design.certification_context_ids_by_stratum
    draws = branch_design.sealed_draw_ids_by_stratum
    expected_next_contexts = tuple(
        None if count == len(contexts[index][1]) else contexts[index][1][count]
        for index, count in enumerate(counts)
    )
    expected_next_draws = tuple(
        None if count == len(draws[index][1]) else draws[index][1][count]
        for index, count in enumerate(counts)
    )
    if (
        tuple(branch_progress.next_expected_context_by_stratum)
        != expected_next_contexts
        or tuple(branch_progress.next_expected_draw_by_stratum)
        != expected_next_draws
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "branch next-C coordinate does not replay its sealed prefix",
        )
    weights = tuple(
        weight for _name, weight in branch_design.secant_design.stratum_weights
    )
    reconstructed: list[V1SignedWorkSixRoleEndpointSummary] = []
    for endpoint_index, action in enumerate(endpoints, start=1):
        lowers: dict[str, float] = {}
        uppers: dict[str, float] = {}
        for role in SIGNED_WORK_SIX_ROLES:
            role_cells = tuple(
                cell_by_key[(stratum, endpoint_index, role)]
                for stratum in stratum_names
            )
            lowers[role] = _downward_weighted_sum(
                weights, tuple(value.lower for value in role_cells)
            )
            uppers[role] = _outward_weighted_sum(
                weights, tuple(value.upper for value in role_cells)
            )
        reconstructed.append(
            V1SignedWorkSixRoleEndpointSummary(
                branch=branch_design.branch,
                endpoint_index=endpoint_index,
                action_hash=action,
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
    observed = tuple(branch_progress.endpoint_summaries)
    expected = tuple(reconstructed)
    if observed != expected:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "aggregate endpoint summaries do not replay bounded C cells",
        )
    return expected


def certify_v1_signed_work_six_role_decision(
    *,
    six_role_C_design: object,
    progress_receipt: object,
    decision_design: object,
    live_chart_identity_hash: str,
    live_current_mask_hash: str,
    live_route_axis_hash: str,
    live_proposal_generation: int,
) -> tuple[
    V1SignedWorkSixRoleCertificationReceipt,
    V1SignedWorkSixRoleDecision,
]:
    """Replay C evidence and decide only when the exact next action is stable."""

    if (
        type(six_role_C_design) is not V1SignedWorkSixRoleCDesign
        or type(progress_receipt) is not V1SignedWorkSixRoleProgressReceipt
        or type(decision_design) is not V1SignedWorkSixRoleDecisionDesign
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
            "six-role certification requires exact design/progress/decision types",
        )
    design = six_role_C_design
    progress = progress_receipt
    policy = decision_design
    design.assert_integrity(policy.six_role_C_design_hash)
    expected_policy = register_v1_signed_work_six_role_decision_design(
        six_role_C_design=design,
        tau_l=policy.tau_l,
        tau_l_source_hash=policy.tau_l_source_hash,
    )
    if policy.design_hash != expected_policy.design_hash:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.HASH_MISMATCH,
            "decision design does not replay from the exact pre-C bridge",
        )
    if (
        progress.design_hash != design.design_hash
        or progress.bridge_family_bundle_hash != design.bridge_family.bundle_hash
        or policy.bridge_family_bundle_hash != design.bridge_family.bundle_hash
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.HASH_MISMATCH,
            "six-role source graph hashes do not close",
        )
    live = (
        _identity(live_chart_identity_hash, name="live chart identity"),
        _identity(live_current_mask_hash, name="live current-mask hash"),
        _identity(live_route_axis_hash, name="live route-axis hash"),
        _nonnegative_int(live_proposal_generation, name="live proposal generation"),
    )
    registered = (
        policy.chart_identity_hash,
        policy.current_mask_hash,
        policy.route_axis_hash,
        policy.proposal_generation,
    )
    if live != registered:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.STALE_CHART,
            "six-role evidence is stale for the live chart/mask/generation",
        )
    expected_source_decisions = tuple(
        value.receipt_hash
        for branch_design in design.branch_designs
        for value in branch_design.bridge_branch.endpoint_decisions
    )
    if expected_source_decisions != policy.complete_source_endpoint_decision_hashes:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.HASH_MISMATCH,
            "pre-C hard-Q closure axis changed after decision registration",
        )
    if tuple(value.branch for value in progress.branch_receipts) != design.branches:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "progress branch axis differs from its pre-C design",
        )
    progress_by_branch = {value.branch: value for value in progress.branch_receipts}
    summary_by_coordinate: dict[
        tuple[V1RelaxedKKTBranch, int], V1SignedWorkSixRoleEndpointSummary
    ] = {}
    observed_count_axis: tuple[int, ...] | None = None
    for branch_design in design.branch_designs:
        branch_progress = progress_by_branch[branch_design.branch]
        summaries = _validated_branch_summaries(
            family_design=design,
            branch_design=branch_design,
            branch_progress=branch_progress,
        )
        counts = tuple(branch_progress.observed_counts_by_stratum)
        if observed_count_axis is None:
            observed_count_axis = counts
        elif counts != observed_count_axis:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                "atomic six-role branches advanced different C prefixes",
            )
        expected_axis = tuple(
            enumerate(branch_design.endpoint_action_hashes, start=1)
        )
        observed_axis = tuple(
            (value.endpoint_index, value.action_hash)
            for value in summaries
        )
        if observed_axis != expected_axis:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.POST_SELECTION,
                "C endpoint summaries were reordered or post-selected",
            )
        for summary in summaries:
            summary_by_coordinate[(branch_design.branch, summary.endpoint_index)] = summary
    if observed_count_axis is None or sum(observed_count_axis) != progress.shared_record_count:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "shared atomic C count differs from branch prefix custody",
        )
    populations = design.branch_designs[0].population_sizes
    expected_sampling_status = (
        "awaiting_C"
        if sum(observed_count_axis) == 0
        else (
            "full_C_census"
            if observed_count_axis == populations
            else "partial_C_evidence"
        )
    )
    if (
        progress.sampling_status != expected_sampling_status
        or progress.tail_count != design.tail_count
        or progress.registered_tail_alpha_total
        != design.registered_tail_alpha_total
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "six-role family sampling/alpha summary does not replay its design",
        )
    certificates = tuple(
        _certificate(
            coordinate=coordinate,
            summary=summary_by_coordinate[(coordinate.branch, coordinate.endpoint_index)],
            tau_l=policy.tau_l,
        )
        for coordinate in policy.endpoint_coordinates
    )
    receipt = V1SignedWorkSixRoleCertificationReceipt(
        decision_design_hash=policy.design_hash,
        six_role_C_design_hash=design.design_hash,
        six_role_C_progress_receipt_hash=progress.receipt_hash,
        bridge_family_bundle_hash=design.bridge_family.bundle_hash,
        chart_identity_hash=policy.chart_identity_hash,
        current_mask_hash=policy.current_mask_hash,
        route_axis_hash=policy.route_axis_hash,
        route_count=policy.route_count,
        proposal_generation=policy.proposal_generation,
        tau_l=policy.tau_l,
        endpoint_certificates=certificates,
        sampling_status=progress.sampling_status,
    )

    certified = tuple(
        value
        for value in certificates
        if value.direct_D_status is V1SignedWorkIntervalStatus.CERTIFIED
    )
    unresolved = tuple(
        value
        for value in certificates
        if value.direct_D_status is V1SignedWorkIntervalStatus.UNRESOLVED
    )
    selected: V1SignedWorkSixRoleEndpointCertificate | None = None
    blockers: tuple[V1SignedWorkSixRoleEndpointCertificate, ...] = ()
    if certified:
        maximum_resource = max(
            value.coordinate.cumulative_removed_count for value in certified
        )
        selected = min(
            (
                value
                for value in certified
                if value.coordinate.cumulative_removed_count == maximum_resource
            ),
            key=lambda value: value.coordinate.pre_C_priority,
        )
        blockers = tuple(
            value
            for value in unresolved
            if (
                value.coordinate.cumulative_removed_count > maximum_resource
                or (
                    value.coordinate.cumulative_removed_count == maximum_resource
                    and value.coordinate.pre_C_priority
                    < selected.coordinate.pre_C_priority
                )
            )
        )
    elif unresolved:
        blockers = unresolved

    if blockers:
        decision = V1SignedWorkSixRoleDecision(
            certification_receipt_hash=receipt.receipt_hash,
            decision_design_hash=policy.design_hash,
            status="acquire_more_C",
            selected_coordinate_hash=None,
            selected_branch=None,
            selected_endpoint_index=None,
            selected_action_hash=None,
            selected_cumulative_removed_count=None,
            admission_basis=None,
            chart_transport_status=None,
            unresolved_coordinate_hashes=tuple(
                value.coordinate.coordinate_hash for value in blockers
            ),
            single_endpoint_proposal_authority=False,
        )
    elif selected is None:
        decision = V1SignedWorkSixRoleDecision(
            certification_receipt_hash=receipt.receipt_hash,
            decision_design_hash=policy.design_hash,
            status="stable_no_endpoint",
            selected_coordinate_hash=None,
            selected_branch=None,
            selected_endpoint_index=None,
            selected_action_hash=None,
            selected_cumulative_removed_count=None,
            admission_basis=None,
            chart_transport_status=None,
            unresolved_coordinate_hashes=(),
            single_endpoint_proposal_authority=False,
        )
    else:
        decision = V1SignedWorkSixRoleDecision(
            certification_receipt_hash=receipt.receipt_hash,
            decision_design_hash=policy.design_hash,
            status="stable_endpoint",
            selected_coordinate_hash=selected.coordinate.coordinate_hash,
            selected_branch=selected.coordinate.branch,
            selected_endpoint_index=selected.coordinate.endpoint_index,
            selected_action_hash=selected.coordinate.action_hash,
            selected_cumulative_removed_count=(
                selected.coordinate.cumulative_removed_count
            ),
            admission_basis=selected.admission_basis,
            chart_transport_status=selected.chart_transport_status,
            unresolved_coordinate_hashes=(),
            single_endpoint_proposal_authority=True,
        )
    return receipt, decision


def authorize_v1_signed_work_six_role_proposal(
    *,
    six_role_C_design: object,
    progress_receipt: object,
    certification_receipt: object,
    decision: object,
    decision_design: object,
    live_chart_identity_hash: str,
    live_current_mask_hash: str,
    live_route_axis_hash: str,
    live_proposal_generation: int,
) -> V1SignedWorkSixRoleProposalAuthorization:
    """Source-replay and authorize one stable frozen prefix coordinate."""

    if (
        type(six_role_C_design) is not V1SignedWorkSixRoleCDesign
        or type(progress_receipt) is not V1SignedWorkSixRoleProgressReceipt
        or type(certification_receipt)
        is not V1SignedWorkSixRoleCertificationReceipt
        or type(decision) is not V1SignedWorkSixRoleDecision
        or type(decision_design) is not V1SignedWorkSixRoleDecisionDesign
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
            "authorization requires exact certification, decision, and design",
        )
    receipt = certification_receipt
    policy = decision_design
    replayed_receipt, replayed_decision = certify_v1_signed_work_six_role_decision(
        six_role_C_design=six_role_C_design,
        progress_receipt=progress_receipt,
        decision_design=policy,
        live_chart_identity_hash=live_chart_identity_hash,
        live_current_mask_hash=live_current_mask_hash,
        live_route_axis_hash=live_route_axis_hash,
        live_proposal_generation=live_proposal_generation,
    )
    if (
        receipt.receipt_hash != replayed_receipt.receipt_hash
        or decision.decision_hash != replayed_decision.decision_hash
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.HASH_MISMATCH,
            "supplied six-role receipt/decision differs from source replay",
        )
    receipt = replayed_receipt
    decision = replayed_decision
    if decision.status != "stable_endpoint":
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.NO_CERTIFIED_ENDPOINT,
            "only a stable direct-D endpoint can be authorized",
        )
    coordinate = next(
        (
            value
            for value in policy.endpoint_coordinates
            if value.coordinate_hash == decision.selected_coordinate_hash
        ),
        None,
    )
    if (
        coordinate is None
        or coordinate.branch is not decision.selected_branch
        or coordinate.endpoint_index != decision.selected_endpoint_index
        or coordinate.action_hash != decision.selected_action_hash
        or coordinate.cumulative_removed_count
        != decision.selected_cumulative_removed_count
        or decision.admission_basis is None
        or decision.chart_transport_status is None
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "selected proposal differs from its pre-C coordinate",
        )
    return V1SignedWorkSixRoleProposalAuthorization(
        certification_receipt_hash=receipt.receipt_hash,
        decision_hash=decision.decision_hash,
        decision_design_hash=policy.design_hash,
        chart_identity_hash=policy.chart_identity_hash,
        current_mask_hash=policy.current_mask_hash,
        route_axis_hash=policy.route_axis_hash,
        route_count=policy.route_count,
        proposal_generation=policy.proposal_generation,
        selected_coordinate=coordinate,
        admission_basis=decision.admission_basis,
        chart_transport_status=decision.chart_transport_status,
        old_chart_validity_claim=decision.admission_basis == "chart_certified",
    )


def authorize_v1_signed_work_six_role_pre_O_fallback_queue(
    *,
    six_role_C_design: object,
    progress_receipt: object,
    certification_receipt: object,
    decision: object,
    decision_design: object,
    live_chart_identity_hash: str,
    live_current_mask_hash: str,
    live_route_axis_hash: str,
    live_proposal_generation: int,
) -> V1SignedWorkSixRoleFallbackQueueAuthorization:
    """Source-replay a compact fallback queue frozen before paired O opens.

    This does not ask the oracle which route composition to try next.  It
    computes, from the already registered C certificate, the maximal prefix of
    direct-D-certified physical candidates available before the first unresolved
    coordinate could change the next post-rejection choice.
    """

    singleton = authorize_v1_signed_work_six_role_proposal(
        six_role_C_design=six_role_C_design,
        progress_receipt=progress_receipt,
        certification_receipt=certification_receipt,
        decision=decision,
        decision_design=decision_design,
        live_chart_identity_hash=live_chart_identity_hash,
        live_current_mask_hash=live_current_mask_hash,
        live_route_axis_hash=live_route_axis_hash,
        live_proposal_generation=live_proposal_generation,
    )
    if (
        type(certification_receipt)
        is not V1SignedWorkSixRoleCertificationReceipt
        or type(decision_design) is not V1SignedWorkSixRoleDecisionDesign
        or type(decision) is not V1SignedWorkSixRoleDecision
    ):
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.INVALID_TYPE,
            "fallback authorization requires exact certification inputs",
        )
    receipt = certification_receipt
    policy = decision_design
    certificate_by_coordinate = {
        value.coordinate.coordinate_hash: value for value in receipt.endpoint_certificates
    }
    ordered = tuple(
        sorted(
            policy.endpoint_coordinates,
            key=lambda value: (-value.cumulative_removed_count, value.pre_C_priority),
        )
    )
    duplicate_signature_by_physical: dict[
        str,
        tuple[
            V1SignedWorkIntervalStatus,
            SixRoleAdmissionBasis | None,
            V1SignedWorkChartTransportStatus,
            tuple[float, float],
            tuple[float, float],
        ],
    ] = {}
    for coordinate in ordered:
        certificate = certificate_by_coordinate.get(coordinate.coordinate_hash)
        if certificate is None:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                "fallback coordinate lacks a replayed certificate",
            )
        physical = coordinate.canonical_candidate_identity_hash
        signature = (
            certificate.direct_D_status,
            certificate.admission_basis,
            certificate.chart_transport_status,
            certificate.d_half_interval,
            certificate.d_one_interval,
        )
        existing = duplicate_signature_by_physical.setdefault(physical, signature)
        if existing != signature:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                "duplicate physical fallback candidates have inconsistent D evidence",
            )
    selected_hash = singleton.selected_coordinate.coordinate_hash
    queued: list[V1SignedWorkSixRoleEndpointCoordinate] = []
    certificate_hashes: list[str] = []
    bases: list[SixRoleAdmissionBasis] = []
    chart_statuses: list[V1SignedWorkChartTransportStatus] = []
    physical_identities: list[str] = []
    seen_physical: set[str] = set()
    blocked_unresolved: str | None = None
    selected_seen = False
    for coordinate in ordered:
        certificate = certificate_by_coordinate.get(coordinate.coordinate_hash)
        if certificate is None:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
                "fallback coordinate lacks a replayed certificate",
            )
        physical = coordinate.canonical_candidate_identity_hash
        if physical in seen_physical:
            continue
        if certificate.direct_D_status is V1SignedWorkIntervalStatus.UNRESOLVED:
            blocked_unresolved = coordinate.coordinate_hash
            break
        if certificate.direct_D_status is V1SignedWorkIntervalStatus.REFUTED:
            seen_physical.add(physical)
            continue
        if certificate.direct_D_status is not V1SignedWorkIntervalStatus.CERTIFIED:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "fallback certificate has an unsupported direct-D status",
            )
        if certificate.admission_basis is None:
            _refuse(
                V1SignedWorkSixRoleDecisionRefusalCode.AUTHORITY_VIOLATION,
                "fallback certified endpoint lacks an admission basis",
            )
        queued.append(coordinate)
        certificate_hashes.append(certificate.certificate_hash)
        bases.append(certificate.admission_basis)
        chart_statuses.append(certificate.chart_transport_status)
        physical_identities.append(physical)
        seen_physical.add(physical)
        selected_seen = selected_seen or coordinate.coordinate_hash == selected_hash
    if not selected_seen or not queued:
        _refuse(
            V1SignedWorkSixRoleDecisionRefusalCode.COORDINATE_MISMATCH,
            "fallback queue does not replay the selected endpoint",
        )
    return V1SignedWorkSixRoleFallbackQueueAuthorization(
        certification_receipt_hash=receipt.receipt_hash,
        decision_hash=decision.decision_hash,
        decision_design_hash=policy.design_hash,
        chart_identity_hash=policy.chart_identity_hash,
        current_mask_hash=policy.current_mask_hash,
        route_axis_hash=policy.route_axis_hash,
        route_count=policy.route_count,
        proposal_generation=policy.proposal_generation,
        selected_coordinate=singleton.selected_coordinate,
        fallback_coordinates=tuple(queued),
        fallback_certificate_hashes=tuple(certificate_hashes),
        fallback_admission_bases=tuple(bases),
        fallback_chart_transport_statuses=tuple(chart_statuses),
        blocked_unresolved_coordinate_hash=blocked_unresolved,
        queued_physical_candidate_identity_hashes=tuple(physical_identities),
        fallback_exhaustion_status=(
            "blocked_on_more_C"
            if blocked_unresolved is not None
            else "fully_resolved_no_more_candidate"
        ),
    )


__all__ = [
    "SIGNED_WORK_SIX_ROLE_AUTHORIZATION_SCHEMA",
    "SIGNED_WORK_SIX_ROLE_AUTHORITY_SCOPE",
    "SIGNED_WORK_SIX_ROLE_CERTIFICATION_SCHEMA",
    "SIGNED_WORK_SIX_ROLE_DECISION_DESIGN_SCHEMA",
    "SIGNED_WORK_SIX_ROLE_DECISION_RULE",
    "SIGNED_WORK_SIX_ROLE_DECISION_SCHEMA",
    "SIGNED_WORK_SIX_ROLE_ENDPOINT_CERTIFICATE_SCHEMA",
    "SIGNED_WORK_SIX_ROLE_FALLBACK_AUTHORIZATION_SCHEMA",
    "SIGNED_WORK_SIX_ROLE_FALLBACK_AUTHORITY_SCOPE",
    "V1SignedWorkChartTransportStatus",
    "V1SignedWorkIntervalStatus",
    "V1SignedWorkSixRoleCertificationReceipt",
    "V1SignedWorkSixRoleDecision",
    "V1SignedWorkSixRoleDecisionDesign",
    "V1SignedWorkSixRoleDecisionRefusal",
    "V1SignedWorkSixRoleDecisionRefusalCode",
    "V1SignedWorkSixRoleEndpointCertificate",
    "V1SignedWorkSixRoleEndpointCoordinate",
    "V1SignedWorkSixRoleFallbackQueueAuthorization",
    "V1SignedWorkSixRoleProposalAuthorization",
    "authorize_v1_signed_work_six_role_proposal",
    "authorize_v1_signed_work_six_role_pre_O_fallback_queue",
    "certify_v1_signed_work_six_role_decision",
    "register_v1_signed_work_six_role_decision_design",
]
