"""Compact proposal authority for bounded sequential modern-v1 F/Q/R evidence.

This is the production sibling of the record-retaining F/Q/R bridge.  It
accepts only the two-sided bounded C-stream receipt, independently replays the
registered F/Q/R inequalities, and can authorize exactly one stable endpoint
from the already frozen ladder.  It never infers other queue rungs.

The resource axis is registered before C opens and is content-addressed here
only to identify the selected action's predeclared physical value.  Neither a
compact decision nor its authorization opens calibration K, paired oracle O,
mask commit, or terminal-validation authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from enum import Enum
import math
from numbers import Integral, Real
from typing import Final, Literal, NoReturn

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_directional_remainder import (
    V1DirectionalRemainderReceipt,
    V1DirectionalSecantRecord,
)
from .modern_v1_fqr_certification import (
    V1FQRCertificationReceipt,
    V1FQRCertificationDecision,
)
from .modern_v1_fqr_sequential_acquisition import (
    FQR_SEQUENTIAL_RECEIPT_SCHEMA,
    V1FQRDecisionReadinessReceipt,
    V1FQRDeterministicUncertaintyReceipt,
    V1FQRSequentialAcquisitionDesign,
    V1FQRSequentialAcquisitionReceipt,
    V1FQRSequentialEndpointSummary,
)
from .modern_v1_ladder_statistics import (
    V1FrozenCandidateLadder,
    V1FrozenLadderCertificationReceipt,
    V1FrozenLadderDecision,
)


COMPACT_FQR_RESOURCE_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_pre_C_resource_registration_v1"
)
COMPACT_FQR_ENDPOINT_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_endpoint_certificate_v1"
)
COMPACT_FQR_CERTIFICATION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_certification_receipt_v1"
)
COMPACT_FQR_DECISION_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_decision_v1"
)
COMPACT_FQR_AUTHORIZATION_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_single_endpoint_authorization_v1"
)
COMPACT_FQR_AUTHORITY_SCOPE: Final = (
    "bounded_two_sided_C_FQR_single_prefrozen_endpoint_proposal_only_"
    "no_K_O_commit_or_terminal_authority_v1"
)
COMPACT_FQR_DECISION_RULE: Final = (
    "upper_certify_selected_and_lower_refute_every_larger_prefrozen_endpoint_v1"
)

_RESOURCE_PREFIX: Final = "modern-v1-compact-fqr-pre-C-resource-"
_ENDPOINT_PREFIX: Final = "modern-v1-compact-fqr-endpoint-"
_RECEIPT_PREFIX: Final = "modern-v1-compact-fqr-receipt-"
_DECISION_PREFIX: Final = "modern-v1-compact-fqr-decision-"
_AUTHORIZATION_PREFIX: Final = "modern-v1-compact-fqr-authorization-"

CompactDecisionStatus = Literal["stable_endpoint", "stable_no_endpoint"]


class V1CompactFQRRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    FQ_ONLY_EVIDENCE = "fq_only_evidence"
    RAW_RECORD_EVIDENCE = "raw_record_evidence"
    LEGACY_ONE_SIDED_EVIDENCE = "legacy_one_sided_evidence"
    HASH_MISMATCH = "hash_mismatch"
    COORDINATE_MISMATCH = "coordinate_mismatch"
    STALE_CHART = "stale_chart"
    POST_SELECTION = "post_selection"
    AMBIGUOUS_EVIDENCE = "ambiguous_evidence"
    UNRESOLVED_DETERMINISTIC = "unresolved_deterministic"
    RESOURCE_DRIFT = "resource_drift"
    NO_CERTIFIED_ENDPOINT = "no_certified_endpoint"
    AUTHORITY_VIOLATION = "authority_violation"


class V1CompactFQRRefusal(ValueError):
    def __init__(self, code: V1CompactFQRRefusalCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1CompactFQRRefusalCode, message: str) -> NoReturn:
    raise V1CompactFQRRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1CompactFQRRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1CompactFQRRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(V1CompactFQRRefusalCode.COORDINATE_MISMATCH, f"{name} is negative")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(V1CompactFQRRefusalCode.COORDINATE_MISMATCH, f"{name} is zero")
    return result


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1CompactFQRRefusalCode.RESOURCE_DRIFT,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1CompactFQRRefusalCode.RESOURCE_DRIFT,
            f"{name} must be finite and positive",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1CompactFQRRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1CompactFQRRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be finite and nonnegative",
        )
    return result


def _direct_mask_hash(mask: HardMask) -> str:
    if type(mask) is not HardMask:
        _refuse(
            V1CompactFQRRefusalCode.INVALID_TYPE,
            "compact F/Q/R requires the exact live HardMask",
        )
    return content_hash(
        {
            "route_axis_hash": mask.route_axis_hash,
            "route_count": mask.route_count,
            "packed_bits": mask.packed_bits,
        },
        prefix="irg-mask-",
    )


def _outward_response(*, eta: float, q: float, r: float) -> float:
    with localcontext() as context:
        context.prec = 100
        exact = Decimal.from_float(eta) * Decimal.from_float(q).sqrt()
        exact += Decimal.from_float(r)
    result = float(exact)
    if Decimal.from_float(result) < exact:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        _refuse(
            V1CompactFQRRefusalCode.COORDINATE_MISMATCH,
            "response upper burden overflowed binary64",
        )
    return result


def _downward_response(*, eta: float, q: float, r: float) -> float:
    with localcontext() as context:
        context.prec = 100
        exact = Decimal.from_float(eta) * Decimal.from_float(q).sqrt()
        exact += Decimal.from_float(r)
    result = float(exact)
    if Decimal.from_float(result) > exact:
        result = math.nextafter(result, -math.inf)
    if not math.isfinite(result):
        _refuse(
            V1CompactFQRRefusalCode.COORDINATE_MISMATCH,
            "response lower burden overflowed binary64",
        )
    return max(0.0, result)


@dataclass(frozen=True, slots=True)
class V1CompactFQRPreCResourceRegistration:
    """Nested endpoint resources frozen before any C-stream outcome exists."""

    acquisition_design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    endpoint_action_hashes: tuple[str, ...]
    resource_by_endpoint: tuple[tuple[int, str, float], ...]
    resource_units: str
    resource_source_hash: str
    issued_before_C_stream_open: bool = True
    C_outcomes_seen: bool = False
    endpoint_resource_mutable_after_C: bool = False
    schema: str = COMPACT_FQR_RESOURCE_REGISTRATION_SCHEMA
    _registration_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "acquisition_design_hash",
            "ladder_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "resource_units",
            "resource_source_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        endpoints = tuple(
            _identity(value, name="resource endpoint action hash")
            for value in self.endpoint_action_hashes
        )
        rows = tuple(
            (
                _positive_int(index, name="resource endpoint index"),
                _identity(action_hash, name="resource action hash"),
                _finite_positive(resource, name="endpoint resource"),
            )
            for index, action_hash, resource in self.resource_by_endpoint
        )
        expected_axis = tuple(enumerate(endpoints, start=1))
        if tuple((index, action) for index, action, _ in rows) != expected_axis:
            _refuse(
                V1CompactFQRRefusalCode.POST_SELECTION,
                "pre-C resource rows reordered the frozen endpoint axis",
            )
        resources = tuple(value for _index, _action, value in rows)
        if any(right <= left for left, right in zip(resources, resources[1:])):
            _refuse(
                V1CompactFQRRefusalCode.RESOURCE_DRIFT,
                "nested endpoint resource must increase strictly",
            )
        if (
            self.issued_before_C_stream_open is not True
            or self.C_outcomes_seen is not False
            or self.endpoint_resource_mutable_after_C is not False
        ):
            _refuse(
                V1CompactFQRRefusalCode.POST_SELECTION,
                "resource axis was not frozen before C outcomes",
            )
        if self.schema != COMPACT_FQR_RESOURCE_REGISTRATION_SCHEMA:
            _refuse(
                V1CompactFQRRefusalCode.COORDINATE_MISMATCH,
                "resource registration schema is invalid",
            )
        object.__setattr__(self, "proposal_generation", _nonnegative_int(
            self.proposal_generation, name="resource proposal generation"
        ))
        object.__setattr__(self, "route_count", _positive_int(
            self.route_count, name="resource route count"
        ))
        object.__setattr__(self, "endpoint_action_hashes", endpoints)
        object.__setattr__(self, "resource_by_endpoint", rows)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "acquisition_design_hash",
                "ladder_hash",
                "chart_identity_hash",
                "current_mask_hash",
                "proposal_generation",
                "route_count",
                "route_axis_hash",
                "endpoint_action_hashes",
                "resource_by_endpoint",
                "resource_units",
                "resource_source_hash",
                "issued_before_C_stream_open",
                "C_outcomes_seen",
                "endpoint_resource_mutable_after_C",
            )
        }

    @property
    def registration_hash(self) -> str:
        cached = self._registration_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RESOURCE_PREFIX)
        if self._registration_hash is not None and observed != self._registration_hash:
            _refuse(V1CompactFQRRefusalCode.HASH_MISMATCH, "resource registration tampered")
        if expected_hash is not None and observed != expected_hash:
            _refuse(V1CompactFQRRefusalCode.HASH_MISMATCH, "resource hash mismatch")
        object.__setattr__(self, "_registration_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1CompactFQREndpointCertificate:
    endpoint_index: int
    action_hash: str
    f_lower: float
    f_upper: float
    q_lower: float
    q_upper: float
    r_half_lower: float
    r_half_upper: float
    r_one_lower: float
    r_one_upper: float
    half_response_lower: float
    half_response_upper: float
    one_response_lower: float
    one_response_upper: float
    upper_certified: bool
    lower_refuted: bool
    schema: str = COMPACT_FQR_ENDPOINT_CERTIFICATE_SCHEMA
    _certificate_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint_index", _positive_int(
            self.endpoint_index, name="endpoint index"
        ))
        object.__setattr__(self, "action_hash", _identity(
            self.action_hash, name="endpoint action hash"
        ))
        for name in (
            "f_lower", "f_upper", "q_lower", "q_upper", "r_half_lower",
            "r_half_upper", "r_one_lower", "r_one_upper",
            "half_response_lower", "half_response_upper",
            "one_response_lower", "one_response_upper",
        ):
            object.__setattr__(self, name, _finite_nonnegative(
                getattr(self, name), name=name
            ))
        for lower, upper in (
            (self.f_lower, self.f_upper),
            (self.q_lower, self.q_upper),
            (self.r_half_lower, self.r_half_upper),
            (self.r_one_lower, self.r_one_upper),
            (self.half_response_lower, self.half_response_upper),
            (self.one_response_lower, self.one_response_upper),
        ):
            if lower > upper:
                _refuse(
                    V1CompactFQRRefusalCode.COORDINATE_MISMATCH,
                    "compact endpoint interval is inverted",
                )
        if (
            self.half_response_lower
            != _downward_response(eta=0.5, q=self.q_lower, r=self.r_half_lower)
            or self.half_response_upper
            != _outward_response(eta=0.5, q=self.q_upper, r=self.r_half_upper)
            or self.one_response_lower
            != _downward_response(eta=1.0, q=self.q_lower, r=self.r_one_lower)
            or self.one_response_upper
            != _outward_response(eta=1.0, q=self.q_upper, r=self.r_one_upper)
        ):
            _refuse(
                V1CompactFQRRefusalCode.COORDINATE_MISMATCH,
                "compact endpoint response burdens do not replay",
            )
        if type(self.upper_certified) is not bool or type(self.lower_refuted) is not bool:
            _refuse(
                V1CompactFQRRefusalCode.COORDINATE_MISMATCH,
                "compact endpoint decisions must be exact booleans",
            )
        if self.upper_certified and self.lower_refuted:
            _refuse(
                V1CompactFQRRefusalCode.COORDINATE_MISMATCH,
                "one endpoint cannot upper-certify and lower-refute",
            )
        if self.schema != COMPACT_FQR_ENDPOINT_CERTIFICATE_SCHEMA:
            _refuse(
                V1CompactFQRRefusalCode.COORDINATE_MISMATCH,
                "compact endpoint schema is invalid",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema", "endpoint_index", "action_hash", "f_lower", "f_upper",
                "q_lower", "q_upper", "r_half_lower", "r_half_upper",
                "r_one_lower", "r_one_upper", "half_response_lower",
                "half_response_upper", "one_response_lower", "one_response_upper",
                "upper_certified", "lower_refuted",
            )
        }

    @property
    def certificate_hash(self) -> str:
        cached = self._certificate_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_ENDPOINT_PREFIX)
        if self._certificate_hash is not None and observed != self._certificate_hash:
            _refuse(V1CompactFQRRefusalCode.HASH_MISMATCH, "endpoint certificate tampered")
        if expected_hash is not None and observed != expected_hash:
            _refuse(V1CompactFQRRefusalCode.HASH_MISMATCH, "endpoint hash mismatch")
        object.__setattr__(self, "_certificate_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1CompactFQRCertificationReceipt:
    acquisition_design_hash: str
    acquisition_receipt_hash: str
    deterministic_uncertainty_receipt_hash: str
    readiness_receipt_hash: str
    resource_registration_hash: str
    fqr_design_hash: str
    secant_design_hash: str
    ladder_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    endpoint_action_hashes: tuple[str, ...]
    joint_fqr_family_registration_hash: str
    endpoint_certificates: tuple[V1CompactFQREndpointCertificate, ...]
    sampling_decision: str
    selected_endpoint_index: int | None
    selected_action_hash: str | None
    all_sampling_and_deterministic_sources_resolved: bool = True
    endpoint_axis_frozen_pre_C: bool = True
    retained_context_ids: int = 0
    retained_record_arrays: int = 0
    raw_record_bearing_receipt_consumed: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    paired_bce_oracle_still_required: bool = True
    decision_rule: str = COMPACT_FQR_DECISION_RULE
    authority_scope: str = COMPACT_FQR_AUTHORITY_SCOPE
    schema: str = COMPACT_FQR_CERTIFICATION_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "acquisition_design_hash", "acquisition_receipt_hash",
            "deterministic_uncertainty_receipt_hash", "readiness_receipt_hash",
            "resource_registration_hash", "fqr_design_hash", "secant_design_hash",
            "ladder_hash", "chart_identity_hash", "current_mask_hash",
            "route_axis_hash", "joint_fqr_family_registration_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        endpoints = tuple(_identity(value, name="receipt endpoint action hash")
                          for value in self.endpoint_action_hashes)
        certificates = tuple(self.endpoint_certificates)
        if any(type(value) is not V1CompactFQREndpointCertificate for value in certificates):
            _refuse(V1CompactFQRRefusalCode.INVALID_TYPE, "compact receipt endpoint type drifted")
        for value in certificates:
            value.assert_integrity()
        if tuple((value.endpoint_index, value.action_hash) for value in certificates) != tuple(
            enumerate(endpoints, start=1)
        ):
            _refuse(V1CompactFQRRefusalCode.POST_SELECTION, "compact receipt endpoint axis drifted")
        if self.sampling_decision not in {"sampling_stable_endpoint", "sampling_stable_no_endpoint"}:
            _refuse(V1CompactFQRRefusalCode.AMBIGUOUS_EVIDENCE, "compact receipt is not stable")
        if self.sampling_decision == "sampling_stable_endpoint":
            if self.selected_endpoint_index is None or self.selected_action_hash is None:
                _refuse(V1CompactFQRRefusalCode.COORDINATE_MISMATCH, "stable endpoint is unnamed")
        elif self.selected_endpoint_index is not None or self.selected_action_hash is not None:
            _refuse(V1CompactFQRRefusalCode.COORDINATE_MISMATCH, "no-endpoint receipt names an action")
        flags = (
            self.all_sampling_and_deterministic_sources_resolved,
            self.endpoint_axis_frozen_pre_C,
            self.retained_context_ids == 0,
            self.retained_record_arrays == 0,
            not self.raw_record_bearing_receipt_consumed,
            not self.calibration_K_open_authority,
            not self.paired_oracle_O_open_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            self.paired_bce_oracle_still_required,
        )
        if any(type(value) is not bool or not value for value in flags):
            _refuse(V1CompactFQRRefusalCode.AUTHORITY_VIOLATION, "compact receipt crossed authority boundary")
        if (
            self.decision_rule != COMPACT_FQR_DECISION_RULE
            or self.authority_scope != COMPACT_FQR_AUTHORITY_SCOPE
            or self.schema != COMPACT_FQR_CERTIFICATION_RECEIPT_SCHEMA
        ):
            _refuse(V1CompactFQRRefusalCode.AUTHORITY_VIOLATION, "compact receipt schema or rule drifted")
        object.__setattr__(self, "proposal_generation", _nonnegative_int(
            self.proposal_generation, name="receipt proposal generation"
        ))
        object.__setattr__(self, "route_count", _positive_int(self.route_count, name="receipt route count"))
        object.__setattr__(self, "endpoint_action_hashes", endpoints)
        object.__setattr__(self, "endpoint_certificates", certificates)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "acquisition_design_hash": self.acquisition_design_hash,
            "acquisition_receipt_hash": self.acquisition_receipt_hash,
            "deterministic_uncertainty_receipt_hash": self.deterministic_uncertainty_receipt_hash,
            "readiness_receipt_hash": self.readiness_receipt_hash,
            "resource_registration_hash": self.resource_registration_hash,
            "fqr_design_hash": self.fqr_design_hash,
            "secant_design_hash": self.secant_design_hash,
            "ladder_hash": self.ladder_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "route_count": self.route_count,
            "route_axis_hash": self.route_axis_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "joint_fqr_family_registration_hash": self.joint_fqr_family_registration_hash,
            "endpoint_certificate_hashes": tuple(value.certificate_hash for value in self.endpoint_certificates),
            "sampling_decision": self.sampling_decision,
            "selected_endpoint_index": self.selected_endpoint_index,
            "selected_action_hash": self.selected_action_hash,
            "all_sampling_and_deterministic_sources_resolved": self.all_sampling_and_deterministic_sources_resolved,
            "endpoint_axis_frozen_pre_C": self.endpoint_axis_frozen_pre_C,
            "retained_context_ids": self.retained_context_ids,
            "retained_record_arrays": self.retained_record_arrays,
            "raw_record_bearing_receipt_consumed": self.raw_record_bearing_receipt_consumed,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "paired_bce_oracle_still_required": self.paired_bce_oracle_still_required,
            "decision_rule": self.decision_rule,
            "authority_scope": self.authority_scope,
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for value in self.endpoint_certificates:
            value.assert_integrity()
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(V1CompactFQRRefusalCode.HASH_MISMATCH, "compact receipt tampered")
        if expected_hash is not None and observed != expected_hash:
            _refuse(V1CompactFQRRefusalCode.HASH_MISMATCH, "compact receipt hash mismatch")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1CompactFQRDecision:
    certification_receipt_hash: str
    acquisition_design_hash: str
    acquisition_receipt_hash: str
    deterministic_uncertainty_receipt_hash: str
    readiness_receipt_hash: str
    resource_registration_hash: str
    ladder_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    status: CompactDecisionStatus
    selected_endpoint_index: int | None
    selected_action_hash: str | None
    paired_bce_oracle_still_required: bool = True
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = COMPACT_FQR_AUTHORITY_SCOPE
    schema: str = COMPACT_FQR_DECISION_SCHEMA
    _decision_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "certification_receipt_hash", "acquisition_design_hash",
            "acquisition_receipt_hash", "deterministic_uncertainty_receipt_hash",
            "readiness_receipt_hash", "resource_registration_hash", "ladder_hash",
            "chart_identity_hash", "current_mask_hash", "route_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.status not in {"stable_endpoint", "stable_no_endpoint"}:
            _refuse(V1CompactFQRRefusalCode.AMBIGUOUS_EVIDENCE, "compact decision is not stable")
        if self.status == "stable_endpoint":
            object.__setattr__(self, "selected_endpoint_index", _positive_int(
                self.selected_endpoint_index, name="selected endpoint index"
            ))
            object.__setattr__(self, "selected_action_hash", _identity(
                self.selected_action_hash, name="selected action hash"
            ))
        elif self.selected_endpoint_index is not None or self.selected_action_hash is not None:
            _refuse(V1CompactFQRRefusalCode.COORDINATE_MISMATCH, "no-endpoint decision names an action")
        flags = (
            self.paired_bce_oracle_still_required,
            not self.calibration_K_open_authority,
            not self.paired_oracle_O_open_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
        )
        if any(type(value) is not bool or not value for value in flags):
            _refuse(V1CompactFQRRefusalCode.AUTHORITY_VIOLATION, "compact decision crossed authority boundary")
        if self.authority_scope != COMPACT_FQR_AUTHORITY_SCOPE or self.schema != COMPACT_FQR_DECISION_SCHEMA:
            _refuse(V1CompactFQRRefusalCode.AUTHORITY_VIOLATION, "compact decision schema drifted")
        object.__setattr__(self, "proposal_generation", _nonnegative_int(
            self.proposal_generation, name="decision proposal generation"
        ))
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "schema", "certification_receipt_hash", "acquisition_design_hash",
            "acquisition_receipt_hash", "deterministic_uncertainty_receipt_hash",
            "readiness_receipt_hash", "resource_registration_hash", "ladder_hash",
            "chart_identity_hash", "current_mask_hash", "proposal_generation",
            "route_axis_hash", "status", "selected_endpoint_index",
            "selected_action_hash", "paired_bce_oracle_still_required",
            "calibration_K_open_authority", "paired_oracle_O_open_authority",
            "mask_commit_authority", "terminal_authority", "authority_scope",
        )}

    @property
    def decision_hash(self) -> str:
        cached = self._decision_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_DECISION_PREFIX)
        if self._decision_hash is not None and observed != self._decision_hash:
            _refuse(V1CompactFQRRefusalCode.HASH_MISMATCH, "compact decision tampered")
        if expected_hash is not None and observed != expected_hash:
            _refuse(V1CompactFQRRefusalCode.HASH_MISMATCH, "compact decision hash mismatch")
        object.__setattr__(self, "_decision_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1CompactFQRProposalAuthorization:
    certification_receipt_hash: str
    decision_hash: str
    acquisition_design_hash: str
    acquisition_receipt_hash: str
    deterministic_uncertainty_receipt_hash: str
    readiness_receipt_hash: str
    resource_registration_hash: str
    ladder_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_count: int
    route_axis_hash: str
    selected_endpoint_index: int
    selected_action_hash: str
    selected_packed_action: V1PackedRemovalAction = field(repr=False, compare=False)
    selected_resulting_mask_hash: str
    selected_resource: float
    resource_units: str
    resource_source_hash: str
    single_endpoint_proposal_authority: bool = True
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    paired_bce_oracle_still_required: bool = True
    inferred_other_queue_rungs: bool = False
    retained_context_ids: int = 0
    retained_record_arrays: int = 0
    authority_scope: str = COMPACT_FQR_AUTHORITY_SCOPE
    schema: str = COMPACT_FQR_AUTHORIZATION_SCHEMA
    _authorization_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "certification_receipt_hash", "decision_hash", "acquisition_design_hash",
            "acquisition_receipt_hash", "deterministic_uncertainty_receipt_hash",
            "readiness_receipt_hash", "resource_registration_hash", "ladder_hash",
            "chart_identity_hash", "current_mask_hash", "route_axis_hash",
            "selected_action_hash", "selected_resulting_mask_hash", "resource_units",
            "resource_source_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.selected_packed_action) is not V1PackedRemovalAction:
            _refuse(V1CompactFQRRefusalCode.INVALID_TYPE, "authorization requires exact packed action")
        if _direct_mask_hash(self.selected_packed_action.resulting_mask) != self.selected_resulting_mask_hash:
            _refuse(V1CompactFQRRefusalCode.COORDINATE_MISMATCH, "packed action mask hash drifted")
        flags = (
            self.single_endpoint_proposal_authority,
            not self.calibration_K_open_authority,
            not self.paired_oracle_O_open_authority,
            not self.irreversible_decision_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            self.paired_bce_oracle_still_required,
            not self.inferred_other_queue_rungs,
            self.retained_context_ids == 0,
            self.retained_record_arrays == 0,
        )
        if any(type(value) is not bool or not value for value in flags):
            _refuse(V1CompactFQRRefusalCode.AUTHORITY_VIOLATION, "authorization crossed proposal-only boundary")
        if self.authority_scope != COMPACT_FQR_AUTHORITY_SCOPE or self.schema != COMPACT_FQR_AUTHORIZATION_SCHEMA:
            _refuse(V1CompactFQRRefusalCode.AUTHORITY_VIOLATION, "authorization schema drifted")
        object.__setattr__(self, "proposal_generation", _nonnegative_int(
            self.proposal_generation, name="authorization proposal generation"
        ))
        object.__setattr__(self, "route_count", _positive_int(self.route_count, name="authorization route count"))
        object.__setattr__(self, "selected_endpoint_index", _positive_int(
            self.selected_endpoint_index, name="authorization endpoint index"
        ))
        object.__setattr__(self, "selected_resource", _finite_positive(
            self.selected_resource, name="authorization resource"
        ))
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in (
            "schema", "certification_receipt_hash", "decision_hash",
            "acquisition_design_hash", "acquisition_receipt_hash",
            "deterministic_uncertainty_receipt_hash", "readiness_receipt_hash",
            "resource_registration_hash", "ladder_hash", "chart_identity_hash",
            "current_mask_hash", "proposal_generation", "route_count",
            "route_axis_hash", "selected_endpoint_index", "selected_action_hash",
            "selected_resulting_mask_hash", "selected_resource", "resource_units",
            "resource_source_hash", "single_endpoint_proposal_authority",
            "calibration_K_open_authority", "paired_oracle_O_open_authority",
            "irreversible_decision_authority", "mask_commit_authority",
            "terminal_authority", "paired_bce_oracle_still_required",
            "inferred_other_queue_rungs", "retained_context_ids",
            "retained_record_arrays", "authority_scope",
        )} | {"selected_packed_action_hash": self.selected_packed_action.action_hash}

    @property
    def authorization_hash(self) -> str:
        cached = self._authorization_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if _direct_mask_hash(self.selected_packed_action.resulting_mask) != self.selected_resulting_mask_hash:
            _refuse(V1CompactFQRRefusalCode.HASH_MISMATCH, "authorization packed action tampered")
        observed = content_hash(self._payload(), prefix=_AUTHORIZATION_PREFIX)
        if self._authorization_hash is not None and observed != self._authorization_hash:
            _refuse(V1CompactFQRRefusalCode.HASH_MISMATCH, "compact authorization tampered")
        if expected_hash is not None and observed != expected_hash:
            _refuse(V1CompactFQRRefusalCode.HASH_MISMATCH, "authorization hash mismatch")
        object.__setattr__(self, "_authorization_hash", observed)
        return observed


def _classify_wrong_evidence_type(value: object) -> None:
    if type(value) in {
        V1FrozenLadderCertificationReceipt,
        V1FrozenLadderDecision,
        V1FQRCertificationReceipt,
        V1FQRCertificationDecision,
    }:
        _refuse(V1CompactFQRRefusalCode.FQ_ONLY_EVIDENCE, "record-retaining or F/Q-only evidence cannot substitute for compact sequential F/Q/R")
    if type(value) in {V1DirectionalRemainderReceipt, V1DirectionalSecantRecord}:
        _refuse(V1CompactFQRRefusalCode.RAW_RECORD_EVIDENCE, "raw secant/remainder evidence is forbidden at the compact boundary")


def _verify_source_graph(
    *,
    acquisition_design: object,
    acquisition_receipt: object,
    deterministic_uncertainty_receipt: object,
    readiness_receipt: object,
    resource_registration: object,
    frozen_ladder: object,
    current_mask: object,
    live_chart_identity_hash: str,
    live_proposal_generation: int,
    live_ladder_hash: str,
) -> tuple[
    V1FQRSequentialAcquisitionDesign,
    V1FQRSequentialAcquisitionReceipt,
    V1FQRDeterministicUncertaintyReceipt,
    V1FQRDecisionReadinessReceipt,
    V1CompactFQRPreCResourceRegistration,
    V1FrozenCandidateLadder,
    HardMask,
    tuple[V1CompactFQREndpointCertificate, ...],
]:
    for value in (acquisition_design, acquisition_receipt, deterministic_uncertainty_receipt, readiness_receipt):
        _classify_wrong_evidence_type(value)
    expected = (
        (acquisition_design, V1FQRSequentialAcquisitionDesign),
        (acquisition_receipt, V1FQRSequentialAcquisitionReceipt),
        (deterministic_uncertainty_receipt, V1FQRDeterministicUncertaintyReceipt),
        (readiness_receipt, V1FQRDecisionReadinessReceipt),
        (resource_registration, V1CompactFQRPreCResourceRegistration),
        (frozen_ladder, V1FrozenCandidateLadder),
        (current_mask, HardMask),
    )
    if any(type(value) is not expected_type for value, expected_type in expected):
        _refuse(V1CompactFQRRefusalCode.INVALID_TYPE, "compact F/Q/R source graph uses an inexact type")
    design = acquisition_design
    receipt = acquisition_receipt
    deterministic = deterministic_uncertainty_receipt
    readiness = readiness_receipt
    resources = resource_registration
    ladder = frozen_ladder
    mask = current_mask
    try:
        design_hash = design.assert_integrity()
        receipt_hash = receipt.assert_integrity()
        deterministic_hash = deterministic.assert_integrity()
        readiness.assert_integrity()
        resource_hash = resources.assert_integrity()
        ladder_hash = ladder.assert_integrity()
    except V1CompactFQRRefusal:
        raise
    except (TypeError, ValueError) as error:
        _refuse(V1CompactFQRRefusalCode.HASH_MISMATCH, f"source graph integrity failed: {error}")
    if design.schema.endswith("_v1") or receipt.schema != FQR_SEQUENTIAL_RECEIPT_SCHEMA:
        _refuse(V1CompactFQRRefusalCode.LEGACY_ONE_SIDED_EVIDENCE, "compact authority requires two-sided sequential schema v2")
    if receipt.retained_context_ids != 0 or receipt.retained_record_arrays != 0:
        _refuse(V1CompactFQRRefusalCode.RAW_RECORD_EVIDENCE, "sequential receipt retained raw C evidence")
    secant = design.fqr_design.secant_design
    embedded = secant.ladder
    mask_hash = _direct_mask_hash(mask)
    live_chart = _identity(live_chart_identity_hash, name="live chart identity hash")
    live_generation = _nonnegative_int(live_proposal_generation, name="live proposal generation")
    live_ladder = _identity(live_ladder_hash, name="live ladder hash")
    if (
        design.fqr_design.ladder_hash != ladder_hash
        or embedded.ladder_hash != ladder_hash
        or embedded.action_hashes != ladder.action_hashes
        or receipt.design_hash != design_hash
        or receipt.fqr_design_hash != design.fqr_design.design_hash
        or receipt.secant_design_hash != secant.design_hash
        or receipt.ladder_hash != ladder_hash
        or receipt.chart_identity_hash != secant.chart_identity_hash
        or receipt.endpoint_action_hashes != ladder.action_hashes
        or deterministic.acquisition_design_hash != design_hash
        or deterministic.fqr_design_hash != design.fqr_design.design_hash
        or deterministic.ladder_hash != ladder_hash
        or deterministic.chart_identity_hash != secant.chart_identity_hash
        or readiness.acquisition_receipt_hash != receipt_hash
        or readiness.deterministic_uncertainty_receipt_hash != deterministic_hash
        or readiness.design_hash != design_hash
        or readiness.fqr_design_hash != design.fqr_design.design_hash
        or readiness.ladder_hash != ladder_hash
        or readiness.chart_identity_hash != secant.chart_identity_hash
        or resources.acquisition_design_hash != design_hash
        or resources.ladder_hash != ladder_hash
        or resources.endpoint_action_hashes != ladder.action_hashes
        or resources.registration_hash != resource_hash
    ):
        _refuse(V1CompactFQRRefusalCode.COORDINATE_MISMATCH, "compact source graph coordinates differ")
    if (
        design.issued_before_C_outcomes is not True
        or design.endpoint_order_mutable_after_C_outcomes is not False
        or design.validation_outcomes_seen is not False
        or ladder.frozen_before_certification_outcomes is not True
        or resources.issued_before_C_stream_open is not True
        or resources.C_outcomes_seen is not False
        or resources.endpoint_resource_mutable_after_C is not False
    ):
        _refuse(V1CompactFQRRefusalCode.POST_SELECTION, "compact ladder or resource axis was post-selected")
    if not ladder.is_packed or type(ladder.parent_mask) is not HardMask:
        _refuse(V1CompactFQRRefusalCode.INVALID_TYPE, "compact authority requires a packed frozen ladder")
    parent = ladder.parent_mask
    if (
        parent.packed_bits != mask.packed_bits
        or _direct_mask_hash(parent) != mask_hash
        or ladder.current_mask_hash != mask_hash
        or resources.current_mask_hash != mask_hash
        or resources.route_count != ladder.route_count
        or resources.route_axis_hash != ladder.route_axis_hash
        or resources.chart_identity_hash != live_chart
        or resources.proposal_generation != live_generation
        or secant.chart_identity_hash != live_chart
        or ladder.proposal_generation != live_generation
        or ladder_hash != live_ladder
    ):
        _refuse(V1CompactFQRRefusalCode.STALE_CHART, "compact evidence is stale for the live chart/mask/generation")
    if receipt.sampling_ambiguity_present or receipt.sampling_decision == "acquire_more_C":
        _refuse(V1CompactFQRRefusalCode.AMBIGUOUS_EVIDENCE, "C-stream sampling decision is unresolved")
    deterministic_resolved = (
        deterministic.resolved_for_independent_C
        and deterministic.numerical_enclosure_valid
    )
    if not deterministic_resolved:
        _refuse(V1CompactFQRRefusalCode.UNRESOLVED_DETERMINISTIC, "FD/KKT/numerical uncertainty remains")
    if (
        readiness.status != "ready_for_fqr_proposal_screening"
        or readiness.ready_for_fqr_proposal_screening is not True
        or not all((readiness.sampling_resolved, readiness.fd_rank_resolved,
                    readiness.kkt_numerics_resolved,
                    readiness.numerical_enclosure_resolved))
        or readiness.selected_endpoint_index != receipt.selected_endpoint_index
        or readiness.selected_action_hash != receipt.selected_action_hash
    ):
        _refuse(V1CompactFQRRefusalCode.UNRESOLVED_DETERMINISTIC, "readiness receipt does not resolve every uncertainty source")
    certificates: list[V1CompactFQREndpointCertificate] = []
    for summary in receipt.endpoint_summaries:
        if type(summary) is not V1FQRSequentialEndpointSummary:
            _refuse(V1CompactFQRRefusalCode.INVALID_TYPE, "endpoint summary type drifted")
        half_lower = _downward_response(eta=0.5, q=summary.q_lower, r=summary.r_half_lower)
        half_upper = _outward_response(eta=0.5, q=summary.q_upper, r=summary.r_half_upper)
        one_lower = _downward_response(eta=1.0, q=summary.q_lower, r=summary.r_one_lower)
        one_upper = _outward_response(eta=1.0, q=summary.q_upper, r=summary.r_one_upper)
        upper_certified = bool(
            summary.f_upper <= design.fqr_design.tau_f
            and half_upper <= 0.5 * design.fqr_design.tau_l
            and one_upper <= design.fqr_design.tau_l
        )
        lower_refuted = bool(
            summary.f_lower > design.fqr_design.tau_f
            or half_lower > 0.5 * design.fqr_design.tau_l
            or one_lower > design.fqr_design.tau_l
        )
        if (
            summary.half_response_burden_lower != half_lower
            or summary.half_response_burden_upper != half_upper
            or summary.one_response_burden_lower != one_lower
            or summary.one_response_burden_upper != one_upper
            or summary.sampling_certified != upper_certified
            or summary.sampling_refuted != lower_refuted
        ):
            _refuse(V1CompactFQRRefusalCode.COORDINATE_MISMATCH, "endpoint F/Q/R inequalities do not replay")
        certificates.append(V1CompactFQREndpointCertificate(
            endpoint_index=summary.endpoint_index,
            action_hash=summary.action_hash,
            f_lower=summary.f_lower,
            f_upper=summary.f_upper,
            q_lower=summary.q_lower,
            q_upper=summary.q_upper,
            r_half_lower=summary.r_half_lower,
            r_half_upper=summary.r_half_upper,
            r_one_lower=summary.r_one_lower,
            r_one_upper=summary.r_one_upper,
            half_response_lower=half_lower,
            half_response_upper=half_upper,
            one_response_lower=one_lower,
            one_response_upper=one_upper,
            upper_certified=upper_certified,
            lower_refuted=lower_refuted,
        ))
    return design, receipt, deterministic, readiness, resources, ladder, mask, tuple(certificates)


def certify_v1_fqr_compact(
    *,
    acquisition_design: object,
    acquisition_receipt: object,
    deterministic_uncertainty_receipt: object,
    readiness_receipt: object,
    resource_registration: object,
    frozen_ladder: object,
    current_mask: object,
    live_chart_identity_hash: str,
    live_proposal_generation: int,
    live_ladder_hash: str,
) -> tuple[V1CompactFQRCertificationReceipt, V1CompactFQRDecision]:
    """Replay compact F/Q/R evidence and issue a stable proposal decision."""

    design, receipt, deterministic, readiness, resources, ladder, mask, certificates = _verify_source_graph(
        acquisition_design=acquisition_design,
        acquisition_receipt=acquisition_receipt,
        deterministic_uncertainty_receipt=deterministic_uncertainty_receipt,
        readiness_receipt=readiness_receipt,
        resource_registration=resource_registration,
        frozen_ladder=frozen_ladder,
        current_mask=current_mask,
        live_chart_identity_hash=live_chart_identity_hash,
        live_proposal_generation=live_proposal_generation,
        live_ladder_hash=live_ladder_hash,
    )
    compact_receipt = V1CompactFQRCertificationReceipt(
        acquisition_design_hash=design.design_hash,
        acquisition_receipt_hash=receipt.receipt_hash,
        deterministic_uncertainty_receipt_hash=deterministic.receipt_hash,
        readiness_receipt_hash=readiness.receipt_hash,
        resource_registration_hash=resources.registration_hash,
        fqr_design_hash=design.fqr_design.design_hash,
        secant_design_hash=design.fqr_design.secant_design.design_hash,
        ladder_hash=ladder.ladder_hash,
        chart_identity_hash=design.fqr_design.chart_identity_hash,
        current_mask_hash=_direct_mask_hash(mask),
        proposal_generation=ladder.proposal_generation,
        route_count=ladder.route_count,
        route_axis_hash=ladder.route_axis_hash,
        endpoint_action_hashes=ladder.action_hashes,
        joint_fqr_family_registration_hash=design.joint_fqr_family_registration_hash,
        endpoint_certificates=certificates,
        sampling_decision=receipt.sampling_decision,
        selected_endpoint_index=receipt.selected_endpoint_index,
        selected_action_hash=receipt.selected_action_hash,
    )
    status: CompactDecisionStatus = (
        "stable_endpoint"
        if receipt.sampling_decision == "sampling_stable_endpoint"
        else "stable_no_endpoint"
    )
    decision = V1CompactFQRDecision(
        certification_receipt_hash=compact_receipt.receipt_hash,
        acquisition_design_hash=design.design_hash,
        acquisition_receipt_hash=receipt.receipt_hash,
        deterministic_uncertainty_receipt_hash=deterministic.receipt_hash,
        readiness_receipt_hash=readiness.receipt_hash,
        resource_registration_hash=resources.registration_hash,
        ladder_hash=ladder.ladder_hash,
        chart_identity_hash=design.fqr_design.chart_identity_hash,
        current_mask_hash=_direct_mask_hash(mask),
        proposal_generation=ladder.proposal_generation,
        route_axis_hash=ladder.route_axis_hash,
        status=status,
        selected_endpoint_index=receipt.selected_endpoint_index,
        selected_action_hash=receipt.selected_action_hash,
    )
    return compact_receipt, decision


def authorize_v1_fqr_compact_proposal(
    *,
    acquisition_design: object,
    acquisition_receipt: object,
    deterministic_uncertainty_receipt: object,
    readiness_receipt: object,
    resource_registration: object,
    certification_receipt: object,
    decision: object,
    frozen_ladder: object,
    current_mask: object,
    live_chart_identity_hash: str,
    live_proposal_generation: int,
    live_ladder_hash: str,
) -> V1CompactFQRProposalAuthorization:
    """Authorize only the one stable packed endpoint; never synthesize a queue."""

    if type(certification_receipt) is not V1CompactFQRCertificationReceipt or type(decision) is not V1CompactFQRDecision:
        _classify_wrong_evidence_type(certification_receipt)
        _classify_wrong_evidence_type(decision)
        _refuse(V1CompactFQRRefusalCode.INVALID_TYPE, "compact authorization requires exact compact receipt and decision")
    supplied_receipt_hash = certification_receipt.assert_integrity()
    supplied_decision_hash = decision.assert_integrity()
    replayed_receipt, replayed_decision = certify_v1_fqr_compact(
        acquisition_design=acquisition_design,
        acquisition_receipt=acquisition_receipt,
        deterministic_uncertainty_receipt=deterministic_uncertainty_receipt,
        readiness_receipt=readiness_receipt,
        resource_registration=resource_registration,
        frozen_ladder=frozen_ladder,
        current_mask=current_mask,
        live_chart_identity_hash=live_chart_identity_hash,
        live_proposal_generation=live_proposal_generation,
        live_ladder_hash=live_ladder_hash,
    )
    if supplied_receipt_hash != replayed_receipt.receipt_hash or supplied_decision_hash != replayed_decision.decision_hash:
        _refuse(V1CompactFQRRefusalCode.HASH_MISMATCH, "supplied compact decision differs from source replay")
    if replayed_decision.status == "stable_no_endpoint":
        _refuse(V1CompactFQRRefusalCode.NO_CERTIFIED_ENDPOINT, "stable F/Q/R evidence authorizes no endpoint")
    endpoint = replayed_decision.selected_endpoint_index
    action_hash = replayed_decision.selected_action_hash
    if endpoint is None or action_hash is None:
        _refuse(V1CompactFQRRefusalCode.COORDINATE_MISMATCH, "stable endpoint disappeared")
    ladder = frozen_ladder
    resources = resource_registration
    mask = current_mask
    frozen = ladder.actions[endpoint - 1]
    packed = frozen.packed_removal_action
    if type(packed) is not V1PackedRemovalAction or frozen.action_hash != action_hash:
        _refuse(V1CompactFQRRefusalCode.COORDINATE_MISMATCH, "selected endpoint differs from frozen packed action")
    removed_count, removed_hash = packed.resulting_mask.removed_index_count_and_hash_from(mask)
    if removed_count != packed.removed_count or removed_hash != packed.removed_route_index_hash:
        _refuse(V1CompactFQRRefusalCode.COORDINATE_MISMATCH, "selected packed action differs from live parent")
    resource_row = resources.resource_by_endpoint[endpoint - 1]
    if resource_row[:2] != (endpoint, action_hash):
        _refuse(V1CompactFQRRefusalCode.RESOURCE_DRIFT, "selected resource row drifted")
    return V1CompactFQRProposalAuthorization(
        certification_receipt_hash=replayed_receipt.receipt_hash,
        decision_hash=replayed_decision.decision_hash,
        acquisition_design_hash=replayed_decision.acquisition_design_hash,
        acquisition_receipt_hash=replayed_decision.acquisition_receipt_hash,
        deterministic_uncertainty_receipt_hash=replayed_decision.deterministic_uncertainty_receipt_hash,
        readiness_receipt_hash=replayed_decision.readiness_receipt_hash,
        resource_registration_hash=replayed_decision.resource_registration_hash,
        ladder_hash=replayed_decision.ladder_hash,
        chart_identity_hash=replayed_decision.chart_identity_hash,
        current_mask_hash=replayed_decision.current_mask_hash,
        proposal_generation=replayed_decision.proposal_generation,
        route_count=ladder.route_count,
        route_axis_hash=replayed_decision.route_axis_hash,
        selected_endpoint_index=endpoint,
        selected_action_hash=action_hash,
        selected_packed_action=packed,
        selected_resulting_mask_hash=_direct_mask_hash(packed.resulting_mask),
        selected_resource=resource_row[2],
        resource_units=resources.resource_units,
        resource_source_hash=resources.resource_source_hash,
    )


__all__ = [
    "V1CompactFQRDecision",
    "V1CompactFQREndpointCertificate",
    "V1CompactFQRPreCResourceRegistration",
    "V1CompactFQRProposalAuthorization",
    "V1CompactFQRCertificationReceipt",
    "V1CompactFQRRefusal",
    "V1CompactFQRRefusalCode",
    "authorize_v1_fqr_compact_proposal",
    "certify_v1_fqr_compact",
]
