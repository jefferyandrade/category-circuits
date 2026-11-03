"""Joint corrected-work, response-length, and remainder certification.

Canonical modern v1 admits a frozen endpoint only when its simultaneous upper
bounds satisfy

``F_U <= tau_F``

and, at each authoritative dose ``eta in {1/2, 1}``,

``eta * sqrt(Q_U) + R_eta,U <= eta * tau_L``.

The raw-logit radius is derived from the registered calibrated-BCE damage
budget and positive affine calibration slope.  Thus ``tau_F = tau_L = B/a``
and ``tau_Q = tau_L**2``; there is no independently tuned remainder budget.
The decision has proposal-screening authority only.  A paired calibrated-BCE
oracle remains mandatory before any irreversible mask commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from enum import Enum
import math
from numbers import Integral, Real
from typing import Final, Literal, Mapping, NoReturn

from .contracts import content_hash
from .modern_v1_directional_remainder import (
    V1DirectionalRemainderReceipt,
    V1DirectionalSecantDesign,
    verify_directional_remainder_receipt,
)
from .modern_v1_dual_chart_geometry import (
    CORRECTED_EVIDENCE_EQUATION,
    LITERAL_JACOBIAN_EQUATION,
)


FQR_CERTIFICATION_DESIGN_SCHEMA: Final = "partition_free_v1_fqr_certification_design_v1"
FQ_ENDPOINT_BOUND_SCHEMA: Final = "partition_free_v1_dual_stream_fq_endpoint_bound_v1"
FQR_CERTIFICATION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_fqr_certification_receipt_v1"
)
FQR_CERTIFICATION_DECISION_SCHEMA: Final = (
    "partition_free_v1_fqr_certification_decision_v1"
)
DECISION_STABILITY_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_decision_stability_certificate_v1"
)
FQR_BUDGET_DERIVATION: Final = (
    "tau_F_equals_tau_L_equals_B_BCE_over_positive_affine_slope_and_"
    "tau_Q_equals_tau_L_squared_v1"
)
FQR_FEASIBILITY_EQUATION: Final = (
    "F_upper_le_tau_F_and_for_eta_half_one_eta_sqrt_Q_upper_plus_"
    "R_eta_upper_le_eta_tau_L_v1"
)
FQR_PROPOSAL_AUTHORITY: Final = (
    "fqr_proposal_screening_only_paired_bce_oracle_required_before_commit_v1"
)
FQR_STABILITY_SOURCES: Final = (
    "sampling",
    "fd_rank",
    "kkt_numerics",
    "directional_remainder",
)

_DESIGN_PREFIX: Final = "partition-free-v1-fqr-design-"
_FQ_BOUND_PREFIX: Final = "partition-free-v1-dual-stream-fq-bound-"
_RECEIPT_PREFIX: Final = "partition-free-v1-fqr-receipt-"
_DECISION_PREFIX: Final = "partition-free-v1-fqr-decision-"
_STABILITY_PREFIX: Final = "partition-free-v1-decision-stability-"

DecisionStatus = Literal[
    "certified_largest_endpoint",
    "refuse_no_certified_endpoint",
    "refuse_inconsistent_simultaneous_bounds",
]


class V1FQRCertificationRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    HASH_MISMATCH = "hash_mismatch"
    COORDINATE_MISMATCH = "coordinate_mismatch"
    STALE_CHART = "stale_chart"
    POST_SELECTION = "post_selection"
    AUTHORITY_VIOLATION = "authority_violation"
    UNSTABLE_DECISION = "unstable_decision"


class V1FQRCertificationRefusal(ValueError):
    """Stable fail-closed error at the F/Q/R proposal boundary."""

    def __init__(self, code: V1FQRCertificationRefusalCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1FQRCertificationRefusalCode, message: str) -> NoReturn:
    raise V1FQRCertificationRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be finite and nonnegative",
        )
    return result


def _finite_positive(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result <= 0.0:
        _refuse(
            V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def _response_lhs(*, eta: float, q_upper: float, r_upper: float) -> Decimal:
    with localcontext() as context:
        context.prec = 80
        return _decimal(eta) * _decimal(q_upper).sqrt() + _decimal(r_upper)


def _response_rhs(*, eta: float, tau_l: float) -> Decimal:
    with localcontext() as context:
        context.prec = 80
        return _decimal(eta) * _decimal(tau_l)


def _outward_float(value: Decimal) -> float:
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
            "derived response burden is not representable in binary64",
        )
    if Decimal.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        _refuse(
            V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
            "outward response burden exceeds binary64",
        )
    return result


@dataclass(frozen=True, slots=True)
class V1FQRCertificationDesign:
    """Pre-outcome joint F/Q/R design and analytically derived trust radius."""

    design_id: str
    secant_design: V1DirectionalSecantDesign = field(repr=False, compare=False)
    bce_damage_budget: float
    positive_affine_calibration_slope: float
    corrected_work_statistical_design_hash: str
    literal_jacobian_statistical_design_hash: str
    joint_fqr_family_registration_hash: str
    corrected_evidence_equation: str = CORRECTED_EVIDENCE_EQUATION
    literal_jacobian_equation: str = LITERAL_JACOBIAN_EQUATION
    budget_derivation: str = FQR_BUDGET_DERIVATION
    feasibility_equation: str = FQR_FEASIBILITY_EQUATION
    no_independent_remainder_budget: bool = True
    issued_before_fqr_outcomes: bool = True
    ladder_frozen_before_fqr_outcomes: bool = True
    validation_outcomes_seen: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    proposal_authority: str = FQR_PROPOSAL_AUTHORITY
    schema: str = FQR_CERTIFICATION_DESIGN_SCHEMA
    tau_f: float = field(init=False)
    tau_l: float = field(init=False)
    tau_q: float = field(init=False)
    _design_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "design_id", _identity(self.design_id, name="design id")
        )
        if type(self.secant_design) is not V1DirectionalSecantDesign:
            _refuse(
                V1FQRCertificationRefusalCode.INVALID_TYPE,
                "F/Q/R design requires the exact directional secant design",
            )
        self.secant_design.assert_integrity()
        for name in (
            "corrected_work_statistical_design_hash",
            "literal_jacobian_statistical_design_hash",
            "joint_fqr_family_registration_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            self.joint_fqr_family_registration_hash
            != self.secant_design.joint_fqr_family_registration_hash
        ):
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "F/Q/R and secant designs name different joint families",
            )
        budget = _finite_nonnegative(self.bce_damage_budget, name="BCE damage budget")
        slope = _finite_positive(
            self.positive_affine_calibration_slope,
            name="positive affine calibration slope",
        )
        tau = budget / slope
        tau_q = tau * tau
        if not math.isfinite(tau) or not math.isfinite(tau_q):
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "derived F/Q/R trust radii are nonfinite",
            )
        if (
            self.corrected_evidence_equation != CORRECTED_EVIDENCE_EQUATION
            or self.literal_jacobian_equation != LITERAL_JACOBIAN_EQUATION
            or self.budget_derivation != FQR_BUDGET_DERIVATION
            or self.feasibility_equation != FQR_FEASIBILITY_EQUATION
            or self.no_independent_remainder_budget is not True
        ):
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "F/Q/R design semantics differ from canonical modern v1",
            )
        if (
            self.issued_before_fqr_outcomes is not True
            or self.ladder_frozen_before_fqr_outcomes is not True
            or self.validation_outcomes_seen is not False
        ):
            _refuse(
                V1FQRCertificationRefusalCode.POST_SELECTION,
                "F/Q/R design was issued after outcomes or validation disclosure",
            )
        if any(
            getattr(self, name) is not False
            for name in (
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            _refuse(
                V1FQRCertificationRefusalCode.AUTHORITY_VIOLATION,
                "F/Q/R design cannot grant action authority",
            )
        if self.proposal_authority != FQR_PROPOSAL_AUTHORITY:
            _refuse(
                V1FQRCertificationRefusalCode.AUTHORITY_VIOLATION,
                "F/Q/R design has proposal-screening authority only",
            )
        if self.schema != FQR_CERTIFICATION_DESIGN_SCHEMA:
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "F/Q/R design schema is invalid",
            )
        object.__setattr__(self, "bce_damage_budget", budget)
        object.__setattr__(self, "positive_affine_calibration_slope", slope)
        object.__setattr__(self, "tau_f", tau)
        object.__setattr__(self, "tau_l", tau)
        object.__setattr__(self, "tau_q", tau_q)
        self.assert_integrity()

    @property
    def ladder_hash(self) -> str:
        return self.secant_design.ladder_hash

    @property
    def chart_identity_hash(self) -> str:
        return self.secant_design.chart_identity_hash

    @property
    def endpoint_action_hashes(self) -> tuple[str, ...]:
        return self.secant_design.endpoint_action_hashes

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_id": self.design_id,
            "secant_design_hash": self.secant_design.design_hash,
            "ladder_hash": self.ladder_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "bce_damage_budget": self.bce_damage_budget,
            "positive_affine_calibration_slope": self.positive_affine_calibration_slope,
            "tau_f": self.tau_f,
            "tau_l": self.tau_l,
            "tau_q": self.tau_q,
            "corrected_work_statistical_design_hash": self.corrected_work_statistical_design_hash,
            "literal_jacobian_statistical_design_hash": self.literal_jacobian_statistical_design_hash,
            "joint_fqr_family_registration_hash": self.joint_fqr_family_registration_hash,
            "corrected_evidence_equation": self.corrected_evidence_equation,
            "literal_jacobian_equation": self.literal_jacobian_equation,
            "budget_derivation": self.budget_derivation,
            "feasibility_equation": self.feasibility_equation,
            "no_independent_remainder_budget": self.no_independent_remainder_budget,
            "issued_before_fqr_outcomes": self.issued_before_fqr_outcomes,
            "ladder_frozen_before_fqr_outcomes": self.ladder_frozen_before_fqr_outcomes,
            "validation_outcomes_seen": self.validation_outcomes_seen,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "proposal_authority": self.proposal_authority,
        }

    @property
    def design_hash(self) -> str:
        cached = self._design_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.secant_design.assert_integrity()
        observed = content_hash(self._payload(), prefix=_DESIGN_PREFIX)
        cached = self._design_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRCertificationRefusalCode.HASH_MISMATCH,
                "F/Q/R design was tampered after hashing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1FQRCertificationRefusalCode.HASH_MISMATCH,
                "F/Q/R design hash does not match",
            )
        object.__setattr__(self, "_design_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQEndpointBound:
    """Simultaneous corrected-F and literal-Jacobian-Q bounds for one rung."""

    endpoint_index: int
    action_hash: str
    f_upper: float
    q_upper: float
    f_value_upper_bound: float
    q_value_upper_bound: float
    corrected_work_bound_source_hash: str
    literal_jacobian_bound_source_hash: str
    joint_fqr_family_registration_hash: str
    fd_q_residual_included: bool = True
    numerical_error_included: bool = True
    sampling_uncertainty_included: bool = True
    f_uses_corrected_evidence_only: bool = True
    q_uses_literal_jacobian_only: bool = True
    empty: bool = False
    schema: str = FQ_ENDPOINT_BOUND_SCHEMA
    _bound_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        endpoint = _positive_int(self.endpoint_index, name="endpoint index")
        action = _identity(self.action_hash, name="action hash")
        f_upper = _finite_nonnegative(self.f_upper, name="F upper")
        q_upper = _finite_nonnegative(self.q_upper, name="Q upper")
        f_value_bound = _finite_nonnegative(
            self.f_value_upper_bound,
            name="F value upper bound",
        )
        q_value_bound = _finite_nonnegative(
            self.q_value_upper_bound,
            name="Q value upper bound",
        )
        if f_upper > f_value_bound or q_upper > q_value_bound:
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "F/Q upper confidence bound exceeds its analytic value bound",
            )
        for name in (
            "corrected_work_bound_source_hash",
            "literal_jacobian_bound_source_hash",
            "joint_fqr_family_registration_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if any(
            getattr(self, name) is not True
            for name in (
                "fd_q_residual_included",
                "numerical_error_included",
                "sampling_uncertainty_included",
                "f_uses_corrected_evidence_only",
                "q_uses_literal_jacobian_only",
            )
        ):
            _refuse(
                V1FQRCertificationRefusalCode.AUTHORITY_VIOLATION,
                "F/Q bound omitted required uncertainty or literal-Jacobian semantics",
            )
        if type(self.empty) is not bool:
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "F/Q empty flag must be one exact boolean",
            )
        if self.schema != FQ_ENDPOINT_BOUND_SCHEMA:
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "F/Q endpoint-bound schema is invalid",
            )
        object.__setattr__(self, "endpoint_index", endpoint)
        object.__setattr__(self, "action_hash", action)
        object.__setattr__(self, "f_upper", f_upper)
        object.__setattr__(self, "q_upper", q_upper)
        object.__setattr__(self, "f_value_upper_bound", f_value_bound)
        object.__setattr__(self, "q_value_upper_bound", q_value_bound)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "endpoint_index",
                "action_hash",
                "f_upper",
                "q_upper",
                "f_value_upper_bound",
                "q_value_upper_bound",
                "corrected_work_bound_source_hash",
                "literal_jacobian_bound_source_hash",
                "joint_fqr_family_registration_hash",
                "fd_q_residual_included",
                "numerical_error_included",
                "sampling_uncertainty_included",
                "f_uses_corrected_evidence_only",
                "q_uses_literal_jacobian_only",
                "empty",
            )
        }

    @property
    def bound_hash(self) -> str:
        cached = self._bound_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_FQ_BOUND_PREFIX)
        cached = self._bound_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRCertificationRefusalCode.HASH_MISMATCH,
                "F/Q endpoint bound was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1FQRCertificationRefusalCode.HASH_MISMATCH,
                "F/Q endpoint-bound hash does not match",
            )
        object.__setattr__(self, "_bound_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRCertificationReceipt:
    """Joint simultaneous upper-bound receipt on the frozen endpoint axis."""

    design_hash: str
    secant_design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    endpoint_action_hashes: tuple[str, ...]
    joint_fqr_family_registration_hash: str
    joint_fqr_family_receipt_hash: str
    fq_bounds: tuple[V1FQEndpointBound, ...]
    remainder_receipt: V1DirectionalRemainderReceipt = field(repr=False, compare=False)
    endpoint_order_unchanged_after_outcomes: bool = True
    postselection_occurred: bool = False
    proposal_authority: str = FQR_PROPOSAL_AUTHORITY
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    schema: str = FQR_CERTIFICATION_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "secant_design_hash",
            "ladder_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "joint_fqr_family_registration_hash",
            "joint_fqr_family_receipt_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        endpoint_hashes = tuple(
            _identity(value, name="endpoint action hash")
            for value in self.endpoint_action_hashes
        )
        if not endpoint_hashes or len(endpoint_hashes) != len(set(endpoint_hashes)):
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "F/Q/R receipt endpoint axis must be nonempty and unique",
            )
        bounds = tuple(self.fq_bounds)
        if any(type(bound) is not V1FQEndpointBound for bound in bounds):
            _refuse(
                V1FQRCertificationRefusalCode.INVALID_TYPE,
                "F/Q/R receipt requires typed F/Q bounds",
            )
        expected_axis = tuple(enumerate(endpoint_hashes, start=1))
        observed_axis = tuple(
            (bound.endpoint_index, bound.action_hash) for bound in bounds
        )
        if observed_axis != expected_axis:
            _refuse(
                V1FQRCertificationRefusalCode.POST_SELECTION,
                "F/Q bounds do not exactly preserve the frozen endpoint order",
            )
        for bound in bounds:
            bound.assert_integrity()
            if (
                bound.joint_fqr_family_registration_hash
                != self.joint_fqr_family_registration_hash
            ):
                _refuse(
                    V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                    "F/Q bound belongs to another registered family",
                )
        if type(self.remainder_receipt) is not V1DirectionalRemainderReceipt:
            _refuse(
                V1FQRCertificationRefusalCode.INVALID_TYPE,
                "F/Q/R receipt requires the typed remainder receipt",
            )
        self.remainder_receipt.assert_integrity()
        if (
            self.remainder_receipt.design_hash != self.secant_design_hash
            or self.remainder_receipt.ladder_hash != self.ladder_hash
            or self.remainder_receipt.chart_identity_hash != self.chart_identity_hash
            or self.remainder_receipt.endpoint_action_hashes != endpoint_hashes
            or self.remainder_receipt.joint_fqr_family_registration_hash
            != self.joint_fqr_family_registration_hash
        ):
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "remainder receipt was recontextualized in F/Q/R certification",
            )
        if (
            self.endpoint_order_unchanged_after_outcomes is not True
            or self.postselection_occurred is not False
        ):
            _refuse(
                V1FQRCertificationRefusalCode.POST_SELECTION,
                "F/Q/R endpoints were selected or reordered after outcomes",
            )
        if self.proposal_authority != FQR_PROPOSAL_AUTHORITY:
            _refuse(
                V1FQRCertificationRefusalCode.AUTHORITY_VIOLATION,
                "F/Q/R receipt has proposal-screening authority only",
            )
        if any(
            getattr(self, name) is not False
            for name in (
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            _refuse(
                V1FQRCertificationRefusalCode.AUTHORITY_VIOLATION,
                "F/Q/R receipt cannot grant action authority",
            )
        if self.schema != FQR_CERTIFICATION_RECEIPT_SCHEMA:
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "F/Q/R receipt schema is invalid",
            )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "endpoint_action_hashes", endpoint_hashes)
        object.__setattr__(self, "fq_bounds", bounds)
        self.assert_integrity()

    @property
    def empty(self) -> bool:
        return self.remainder_receipt.empty or any(
            bound.empty for bound in self.fq_bounds
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "secant_design_hash": self.secant_design_hash,
            "ladder_hash": self.ladder_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "joint_fqr_family_registration_hash": self.joint_fqr_family_registration_hash,
            "joint_fqr_family_receipt_hash": self.joint_fqr_family_receipt_hash,
            "fq_bound_hashes": tuple(bound.bound_hash for bound in self.fq_bounds),
            "remainder_receipt_hash": self.remainder_receipt.receipt_hash,
            "endpoint_order_unchanged_after_outcomes": self.endpoint_order_unchanged_after_outcomes,
            "postselection_occurred": self.postselection_occurred,
            "proposal_authority": self.proposal_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for bound in self.fq_bounds:
            bound.assert_integrity()
        self.remainder_receipt.assert_integrity()
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        cached = self._receipt_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRCertificationRefusalCode.HASH_MISMATCH,
                "F/Q/R receipt was tampered after hashing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1FQRCertificationRefusalCode.HASH_MISMATCH,
                "F/Q/R receipt hash does not match",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRCertificationDecision:
    receipt_hash: str
    design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    status: DecisionStatus
    endpoint_outcomes: tuple[
        tuple[
            int, str, float, float, float, float, float, float, bool, bool, bool, bool
        ],
        ...,
    ]
    selected_endpoint_index: int | None
    selected_action_hash: str | None
    reason: str
    proposal_authority: str = FQR_PROPOSAL_AUTHORITY
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    oracle_required_before_mask_commit: bool = True
    schema: str = FQR_CERTIFICATION_DECISION_SCHEMA
    _decision_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "receipt_hash",
            "design_hash",
            "ladder_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "reason",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        if self.status not in {
            "certified_largest_endpoint",
            "refuse_no_certified_endpoint",
            "refuse_inconsistent_simultaneous_bounds",
        }:
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "unknown F/Q/R decision status",
            )
        outcomes = tuple(self.endpoint_outcomes)
        if not outcomes:
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "F/Q/R decision requires endpoint outcomes",
            )
        for position, row in enumerate(outcomes, start=1):
            if len(row) != 12 or row[0] != position:
                _refuse(
                    V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                    "F/Q/R outcome rows must be contiguous twelve-field records",
                )
            _identity(row[1], name="outcome action hash")
            for index, name in zip(
                range(2, 8),
                (
                    "F upper",
                    "Q upper",
                    "R half upper",
                    "R one upper",
                    "half lhs",
                    "one lhs",
                ),
                strict=True,
            ):
                _finite_nonnegative(row[index], name=name)
            if not all(type(value) is bool for value in row[8:12]):
                _refuse(
                    V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                    "F/Q/R outcome flags must be exact booleans",
                )
            if row[11] != (row[8] and row[9] and row[10]):
                _refuse(
                    V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                    "F/Q/R certification must require F and both dose constraints",
                )
        certified = tuple(row for row in outcomes if row[11])
        if self.status == "certified_largest_endpoint":
            if not certified or (
                self.selected_endpoint_index,
                self.selected_action_hash,
            ) != (certified[-1][0], certified[-1][1]):
                _refuse(
                    V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                    "F/Q/R decision did not choose the largest certified endpoint",
                )
        elif (
            self.selected_endpoint_index is not None
            or self.selected_action_hash is not None
        ):
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "an F/Q/R refusal cannot select an endpoint",
            )
        if (
            self.proposal_authority != FQR_PROPOSAL_AUTHORITY
            or self.bce_oracle_authority is not False
            or self.mask_commit_authority is not False
            or self.oracle_required_before_mask_commit is not True
        ):
            _refuse(
                V1FQRCertificationRefusalCode.AUTHORITY_VIOLATION,
                "the paired BCE oracle remains mandatory after F/Q/R screening",
            )
        if self.schema != FQR_CERTIFICATION_DECISION_SCHEMA:
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "F/Q/R decision schema is invalid",
            )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "endpoint_outcomes", outcomes)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "receipt_hash",
                "design_hash",
                "ladder_hash",
                "chart_identity_hash",
                "current_mask_hash",
                "proposal_generation",
                "status",
                "endpoint_outcomes",
                "selected_endpoint_index",
                "selected_action_hash",
                "reason",
                "proposal_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "oracle_required_before_mask_commit",
            )
        }

    @property
    def decision_hash(self) -> str:
        cached = self._decision_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_DECISION_PREFIX)
        cached = self._decision_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRCertificationRefusalCode.HASH_MISMATCH,
                "F/Q/R decision was tampered after hashing",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1FQRCertificationRefusalCode.HASH_MISMATCH,
                "F/Q/R decision hash does not match",
            )
        object.__setattr__(self, "_decision_hash", observed)
        return observed


def verify_v1_fqr_certification_receipt(
    *,
    design: V1FQRCertificationDesign,
    receipt: V1FQRCertificationReceipt,
    expected_receipt_hash: str | None = None,
) -> str:
    if (
        type(design) is not V1FQRCertificationDesign
        or type(receipt) is not V1FQRCertificationReceipt
    ):
        _refuse(
            V1FQRCertificationRefusalCode.INVALID_TYPE,
            "F/Q/R replay requires typed design and receipt",
        )
    design_hash = design.assert_integrity()
    receipt_hash = receipt.assert_integrity(expected_receipt_hash)
    verify_directional_remainder_receipt(
        design=design.secant_design,
        receipt=receipt.remainder_receipt,
    )
    ladder = design.secant_design.ladder
    if (
        receipt.design_hash != design_hash
        or receipt.secant_design_hash != design.secant_design.design_hash
        or receipt.ladder_hash != design.ladder_hash
        or receipt.chart_identity_hash != design.chart_identity_hash
        or receipt.current_mask_hash != ladder.current_mask_hash
        or receipt.proposal_generation != ladder.proposal_generation
        or receipt.endpoint_action_hashes != design.endpoint_action_hashes
        or receipt.joint_fqr_family_registration_hash
        != design.joint_fqr_family_registration_hash
    ):
        _refuse(
            V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
            "F/Q/R receipt differs from its frozen design",
        )
    return receipt_hash


def decide_v1_fqr(
    *,
    design: V1FQRCertificationDesign,
    receipt: V1FQRCertificationReceipt,
    expected_chart_identity_hash: str,
    expected_current_mask_hash: str,
    expected_proposal_generation: int,
    expected_ladder_hash: str,
) -> V1FQRCertificationDecision:
    """Select the largest independently certified rung, or refuse safely."""

    receipt_hash = verify_v1_fqr_certification_receipt(
        design=design,
        receipt=receipt,
    )
    expected_chart = _identity(
        expected_chart_identity_hash, name="live chart identity hash"
    )
    expected_mask = _identity(expected_current_mask_hash, name="live current-mask hash")
    expected_generation = _nonnegative_int(
        expected_proposal_generation,
        name="live proposal generation",
    )
    expected_ladder = _identity(expected_ladder_hash, name="live ladder hash")
    if (
        design.chart_identity_hash != expected_chart
        or receipt.chart_identity_hash != expected_chart
        or receipt.current_mask_hash != expected_mask
        or receipt.proposal_generation != expected_generation
        or receipt.ladder_hash != expected_ladder
    ):
        _refuse(
            V1FQRCertificationRefusalCode.STALE_CHART,
            "F/Q/R evidence does not belong to the current chart/mask/generation",
        )

    remainder = receipt.remainder_receipt.interval_upper_by_endpoint_and_dose
    outcomes: list[
        tuple[
            int, str, float, float, float, float, float, float, bool, bool, bool, bool
        ]
    ] = []
    for bound in receipt.fq_bounds:
        endpoint = bound.endpoint_index
        r_half = float(remainder[endpoint - 1, 0])
        r_one = float(remainder[endpoint - 1, 1])
        half_lhs_decimal = _response_lhs(
            eta=0.5,
            q_upper=bound.q_upper,
            r_upper=r_half,
        )
        one_lhs_decimal = _response_lhs(
            eta=1.0,
            q_upper=bound.q_upper,
            r_upper=r_one,
        )
        f_ok = _decimal(bound.f_upper) <= _decimal(design.tau_f)
        half_ok = half_lhs_decimal <= _response_rhs(eta=0.5, tau_l=design.tau_l)
        one_ok = one_lhs_decimal <= _response_rhs(eta=1.0, tau_l=design.tau_l)
        certified = bool(
            not bound.empty
            and not receipt.remainder_receipt.empty
            and f_ok
            and half_ok
            and one_ok
        )
        outcomes.append(
            (
                endpoint,
                bound.action_hash,
                bound.f_upper,
                bound.q_upper,
                r_half,
                r_one,
                _outward_float(half_lhs_decimal),
                _outward_float(one_lhs_decimal),
                bool(f_ok),
                bool(half_ok),
                bool(one_ok),
                certified,
            )
        )
    if receipt.empty:
        status: DecisionStatus = "refuse_inconsistent_simultaneous_bounds"
        selected_endpoint = None
        selected_action = None
        reason = "at least one joint F/Q/R simultaneous bound is empty"
    else:
        certified_rows = tuple(row for row in outcomes if row[-1])
        if certified_rows:
            status = "certified_largest_endpoint"
            selected_endpoint = certified_rows[-1][0]
            selected_action = certified_rows[-1][1]
            reason = (
                "largest frozen endpoint satisfying corrected-work and both "
                "directional response-length-plus-remainder constraints"
            )
        else:
            status = "refuse_no_certified_endpoint"
            selected_endpoint = None
            selected_action = None
            reason = "no frozen endpoint satisfies all joint F/Q/R constraints"
    return V1FQRCertificationDecision(
        receipt_hash=receipt_hash,
        design_hash=design.design_hash,
        ladder_hash=design.ladder_hash,
        chart_identity_hash=design.chart_identity_hash,
        current_mask_hash=receipt.current_mask_hash,
        proposal_generation=receipt.proposal_generation,
        status=status,
        endpoint_outcomes=tuple(outcomes),
        selected_endpoint_index=selected_endpoint,
        selected_action_hash=selected_action,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class V1DecisionStabilityCertificate:
    """Proof that all registered pre-oracle uncertainty sets choose one rung."""

    design_hash: str
    receipt_hash: str
    decision_hash: str
    ladder_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    selected_endpoint_index: int | None
    selected_action_hash: str | None
    admissible_endpoint_sets_by_source: tuple[tuple[str, tuple[int | None, ...]], ...]
    uncertainty_receipt_hashes_by_source: tuple[tuple[str, str], ...]
    all_admissible_combinations_same: bool = True
    candidate_endpoint_stability_only: bool = True
    irreversible_decision_authority: bool = False
    paired_bce_oracle_still_required: bool = True
    schema: str = DECISION_STABILITY_CERTIFICATE_SCHEMA
    _certificate_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "receipt_hash",
            "decision_hash",
            "ladder_hash",
            "chart_identity_hash",
            "current_mask_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        if self.selected_endpoint_index is not None:
            endpoint = _positive_int(
                self.selected_endpoint_index,
                name="selected endpoint index",
            )
            if self.selected_action_hash is None:
                _refuse(
                    V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                    "stable selected endpoint requires an action hash",
                )
            object.__setattr__(self, "selected_endpoint_index", endpoint)
            object.__setattr__(
                self,
                "selected_action_hash",
                _identity(self.selected_action_hash, name="selected action hash"),
            )
        elif self.selected_action_hash is not None:
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "stable refusal cannot name an action hash",
            )
        axes: list[tuple[str, tuple[int | None, ...]]] = []
        for source, raw_values in self.admissible_endpoint_sets_by_source:
            source_id = _identity(source, name="uncertainty source")
            values: list[int | None] = []
            for raw in raw_values:
                values.append(
                    None
                    if raw is None
                    else _positive_int(raw, name="admissible endpoint index")
                )
            axis = tuple(values)
            if not axis or len(axis) != len(set(axis)):
                _refuse(
                    V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                    "admissible endpoint sets must be nonempty and unique",
                )
            axes.append((source_id, axis))
        normalized = tuple(axes)
        if tuple(source for source, _ in normalized) != FQR_STABILITY_SOURCES:
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "decision stability must cover sampling, FD rank, KKT, and remainder",
            )
        receipt_rows = tuple(
            (
                _identity(source, name="uncertainty receipt source"),
                _identity(receipt_hash, name="uncertainty receipt hash"),
            )
            for source, receipt_hash in self.uncertainty_receipt_hashes_by_source
        )
        if tuple(source for source, _ in receipt_rows) != FQR_STABILITY_SOURCES:
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "decision stability must bind one receipt per uncertainty source",
            )
        target = self.selected_endpoint_index
        if any(values != (target,) for _, values in normalized):
            _refuse(
                V1FQRCertificationRefusalCode.UNSTABLE_DECISION,
                "at least one uncertainty set permits a different candidate decision",
            )
        if (
            self.all_admissible_combinations_same is not True
            or self.candidate_endpoint_stability_only is not True
            or self.irreversible_decision_authority is not False
            or self.paired_bce_oracle_still_required is not True
        ):
            _refuse(
                V1FQRCertificationRefusalCode.AUTHORITY_VIOLATION,
                "decision-stability certificate cannot replace the paired oracle",
            )
        if self.schema != DECISION_STABILITY_CERTIFICATE_SCHEMA:
            _refuse(
                V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
                "decision-stability certificate schema is invalid",
            )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "admissible_endpoint_sets_by_source", normalized)
        object.__setattr__(self, "uncertainty_receipt_hashes_by_source", receipt_rows)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "design_hash",
                "receipt_hash",
                "decision_hash",
                "ladder_hash",
                "chart_identity_hash",
                "current_mask_hash",
                "proposal_generation",
                "selected_endpoint_index",
                "selected_action_hash",
                "admissible_endpoint_sets_by_source",
                "uncertainty_receipt_hashes_by_source",
                "all_admissible_combinations_same",
                "candidate_endpoint_stability_only",
                "irreversible_decision_authority",
                "paired_bce_oracle_still_required",
            )
        }

    @property
    def certificate_hash(self) -> str:
        cached = self._certificate_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_STABILITY_PREFIX)
        cached = self._certificate_hash
        if cached is not None and cached != observed:
            _refuse(
                V1FQRCertificationRefusalCode.HASH_MISMATCH,
                "decision-stability certificate was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1FQRCertificationRefusalCode.HASH_MISMATCH,
                "decision-stability certificate hash does not match",
            )
        object.__setattr__(self, "_certificate_hash", observed)
        return observed


def certify_v1_decision_stability(
    *,
    design: V1FQRCertificationDesign,
    receipt: V1FQRCertificationReceipt,
    decision: V1FQRCertificationDecision,
    admissible_endpoint_sets_by_source: Mapping[str, tuple[int | None, ...]],
    uncertainty_receipt_hashes_by_source: Mapping[str, str],
) -> V1DecisionStabilityCertificate:
    """Issue a candidate-stability certificate; ambiguity refuses loudly."""

    if type(decision) is not V1FQRCertificationDecision:
        _refuse(
            V1FQRCertificationRefusalCode.INVALID_TYPE,
            "decision stability requires the exact F/Q/R decision type",
        )
    receipt_hash = verify_v1_fqr_certification_receipt(design=design, receipt=receipt)
    decision_hash = decision.assert_integrity()
    if (
        decision.receipt_hash != receipt_hash
        or decision.design_hash != design.design_hash
        or decision.ladder_hash != design.ladder_hash
        or decision.chart_identity_hash != design.chart_identity_hash
        or decision.current_mask_hash != receipt.current_mask_hash
        or decision.proposal_generation != receipt.proposal_generation
    ):
        _refuse(
            V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
            "candidate decision differs from the F/Q/R evidence graph",
        )
    if set(admissible_endpoint_sets_by_source) != set(FQR_STABILITY_SOURCES):
        _refuse(
            V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
            "decision stability source set is incomplete or unknown",
        )
    if set(uncertainty_receipt_hashes_by_source) != set(FQR_STABILITY_SOURCES):
        _refuse(
            V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
            "decision stability receipt source set is incomplete or unknown",
        )
    endpoint_count = len(design.endpoint_action_hashes)
    if any(
        value is not None and (value <= 0 or value > endpoint_count)
        for values in admissible_endpoint_sets_by_source.values()
        for value in values
    ):
        _refuse(
            V1FQRCertificationRefusalCode.COORDINATE_MISMATCH,
            "an admissible uncertainty decision left the frozen endpoint axis",
        )
    axes = tuple(
        (source, tuple(admissible_endpoint_sets_by_source[source]))
        for source in FQR_STABILITY_SOURCES
    )
    receipt_axes = tuple(
        (source, uncertainty_receipt_hashes_by_source[source])
        for source in FQR_STABILITY_SOURCES
    )
    return V1DecisionStabilityCertificate(
        design_hash=design.design_hash,
        receipt_hash=receipt_hash,
        decision_hash=decision_hash,
        ladder_hash=design.ladder_hash,
        chart_identity_hash=design.chart_identity_hash,
        current_mask_hash=receipt.current_mask_hash,
        proposal_generation=receipt.proposal_generation,
        selected_endpoint_index=decision.selected_endpoint_index,
        selected_action_hash=decision.selected_action_hash,
        admissible_endpoint_sets_by_source=axes,
        uncertainty_receipt_hashes_by_source=receipt_axes,
    )


__all__ = [
    "DECISION_STABILITY_CERTIFICATE_SCHEMA",
    "FQR_BUDGET_DERIVATION",
    "FQR_CERTIFICATION_DECISION_SCHEMA",
    "FQR_CERTIFICATION_DESIGN_SCHEMA",
    "FQR_CERTIFICATION_RECEIPT_SCHEMA",
    "FQR_FEASIBILITY_EQUATION",
    "FQR_PROPOSAL_AUTHORITY",
    "FQR_STABILITY_SOURCES",
    "V1DecisionStabilityCertificate",
    "V1FQEndpointBound",
    "V1FQRCertificationDecision",
    "V1FQRCertificationDesign",
    "V1FQRCertificationReceipt",
    "V1FQRCertificationRefusal",
    "V1FQRCertificationRefusalCode",
    "certify_v1_decision_stability",
    "decide_v1_fqr",
    "verify_v1_fqr_certification_receipt",
]
