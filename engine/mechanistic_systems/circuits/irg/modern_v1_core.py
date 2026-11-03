"""Inert, content-addressed contracts for the modern v1 proposal plane.

This module deliberately stops before claim-bearing execution.  Owner-selected
A1-A/A2-A coordinates may be represented only when they carry the immutable J0
campaign-contract hash, but no production contract instance or
external authorization boundary is established here.  Consequently no
registration value, signature-shaped string, proposal chart, or evidence
packet can make v1 executable.

The implemented surface is the authority-inert substrate: immutable evidence
lineage, pre-draw certificate freezes, strict draw accounting, deterministic
Frequent-Directions proposal lineage, identified finite-intervention correction
lineage, and fail-closed dynamic relinearization.  Everything produced by the
chart builder remains a proposal instrument and has no mask-acceptance
authority.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np

from .contracts import content_hash

CertificateFamily = Literal["staged_alpha", "grow_eprocess"]
CrossQueryShrinkageDisposition = Literal["defer", "typed_amendment"]
V1StreamRole = Literal[
    "proposal_census",
    "calibration_certificate",
    "oracle_certificate",
]
V1ChartReuseAction = Literal["reuse", "refresh", "refuse"]
V1ResidualDiagonalStatus = Literal[
    "absent",
    "proposal_only",
    "global_loewner_bound",
    "action_class_quadratic_bound",
]
V1MetricGeometryKind = Literal["frequent_directions_deterministic_v1"]
V1FDImplementation = Literal[
    "canonical_numpy",
    "cuda_algebraically_equivalent",
    "cuda_gram_algebraically_equivalent",
]
V1CorrectionKind = Literal[
    "identified_finite_intervention_v1",
    "analytic_exact_gate_raw_target_logit_v1",
]
V1MixedDifferenceProofBasis = Literal[
    "smoothness_theorem",
    "bounded_mixed_finite_difference",
    "interventional_coverage",
]

CERTIFICATE_STREAM_ROLES: tuple[V1StreamRole, ...] = (
    "calibration_certificate",
    "oracle_certificate",
)
PROPOSAL_STREAM_ROLE: V1StreamRole = "proposal_census"
_FD_ALGORITHM_ID = "weighted_welford_frequent_directions_v1"

MODERN_V1_WORKFLOW_IDENTITY: Final = "irg/v1"

_OWNER_REGISTRATION_SCHEMA = "partition_free_v1_owner_registration_v3"
_CERTIFICATE_FREEZE_SCHEMA = "partition_free_v1_certificate_freeze_v2"
_EVIDENCE_PACKET_SCHEMA = "partition_free_v1_evidence_packet_v2"
_CERTIFICATE_LEDGER_SCHEMA = "partition_free_v1_certificate_ledger_v2"
_FD_LINEAGE_SCHEMA = "partition_free_v1_fd_lineage_v1"
_FD_VERIFICATION_RECEIPT_SCHEMA = "partition_free_v1_fd_lineage_verification_receipt_v1"
_CORRECTION_LINEAGE_SCHEMA = "partition_free_v1_interventional_correction_lineage_v1"
_ANALYTIC_EXACT_GATE_LINEAGE_SCHEMA = (
    "partition_free_v1_analytic_exact_gate_correction_lineage_v1"
)
_CORRECTION_VERIFICATION_RECEIPT_SCHEMA = "partition_free_v1_correction_verification_receipt_v1"
_ANALYTIC_EXACT_GATE_VERIFICATION_SCHEMA = (
    "partition_free_v1_analytic_exact_gate_verification_receipt_v1"
)
_CHART_THRESHOLDS_SCHEMA = "partition_free_v1_chart_thresholds_v1"
_FINITE_ACTION_CLASS_PROOF_SCHEMA = "partition_free_v1_finite_action_class_proof_v1"
_MIXED_DIFFERENCE_ENVELOPE_SCHEMA = "partition_free_v1_mixed_difference_envelope_v2"
_CHART_BUILD_CONTRACT_SCHEMA = "partition_free_v1_chart_build_contract_v1"
_PROPOSAL_CHART_SCHEMA = "partition_free_v1_proposal_chart_v2"
_CHART_REUSE_REQUEST_SCHEMA = "partition_free_v1_chart_reuse_request_v1"
_CHART_REUSE_DECISION_SCHEMA = "partition_free_v1_chart_reuse_decision_v2"


def _nonempty(value: str, label: str) -> str:
    text = str(value)
    if not text:
        raise ValueError(f"{label} must be nonempty")
    return text


def _exact_schema(value: str, expected: str, label: str) -> str:
    if type(value) is not str or value != expected:
        raise ValueError(f"unsupported {label} schema")
    return expected


def _finite_nonnegative(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, not bool")
    number = int(value)
    if number <= 0 or number != value:
        raise ValueError(f"{label} must be a positive integer")
    return number


def _strict_bool(value: bool, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be bool")
    return value


def _hash_or_none(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, label)


def _string_tuple(
    values: Sequence[str],
    label: str,
    *,
    unique: bool,
) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or any(not value for value in result):
        raise ValueError(f"{label} must be a nonempty sequence of nonempty strings")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _ordered_finite_removal_sets(
    values: Sequence[Sequence[int]],
    *,
    route_count: int,
    label: str,
) -> tuple[tuple[int, ...], ...]:
    """Canonicalize each set while preserving the declared action order."""

    actions: list[tuple[int, ...]] = []
    for action_index, raw_action in enumerate(tuple(values)):
        if isinstance(raw_action, (str, bytes)):
            raise TypeError(f"{label} actions must contain exact integer indices")
        raw_indices = tuple(raw_action)
        if not raw_indices:
            raise ValueError(f"{label} actions must be nonempty")
        if any(isinstance(index, bool) or not isinstance(index, Integral) for index in raw_indices):
            raise TypeError(f"{label} action {action_index} must contain exact integer indices")
        action = tuple(sorted(int(index) for index in raw_indices))
        if len(set(action)) != len(action) or any(
            index < 0 or index >= route_count for index in action
        ):
            raise ValueError(f"{label} actions require unique indices on the declared route axis")
        actions.append(action)
    if not actions:
        raise ValueError(f"{label} must declare at least one finite action")
    if len(set(actions)) != len(actions):
        raise ValueError(f"{label} must not duplicate actions")
    return tuple(actions)


def _deep_freeze(value: Any, label: str) -> Any:
    """Copy and recursively freeze the small metadata language used in hashes.

    Content-addressed objects must not change after their hash is observed.
    Rather than excluding potentially meaningful metadata from claim identity,
    this surface accepts only immutable canonical scalar/container values and
    copies every container before wrapping it.
    """

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise TypeError(f"{label} keys must be nonempty strings")
            if raw_key in frozen:
                raise ValueError(f"{label} contains a duplicate key")
            frozen[raw_key] = _deep_freeze(raw_value, f"{label}.{raw_key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item, f"{label}[{index}]") for index, item in enumerate(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain nonfinite floats")
        return float(value)
    if isinstance(value, (str, bytes, int, bool)) or value is None:
        return value
    raise TypeError(f"{label} contains unsupported mutable/noncanonical value {type(value)!r}")


def _immutable_metadata(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    frozen = _deep_freeze(value, label)
    if not isinstance(frozen, Mapping):  # pragma: no cover - defensive boundary
        raise TypeError(f"{label} must be a mapping")
    return frozen


def _validate_aligned_draw_lineage(
    *,
    draw_ids: Sequence[str],
    source_context_ids: Sequence[str],
    source_row_hashes: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...] | None]:
    draws = _string_tuple(draw_ids, "draw_ids", unique=True)
    # Physical sources may repeat under IID-with-replacement sampling.  Draw
    # identity, not physical source identity, is the unique accumulator axis.
    sources = _string_tuple(source_context_ids, "source_context_ids", unique=False)
    if len(draws) != len(sources):
        raise ValueError("draw_ids and source_context_ids must have equal length")
    rows: tuple[str, ...] | None = None
    if source_row_hashes is not None:
        rows = _string_tuple(source_row_hashes, "source_row_hashes", unique=False)
        if len(rows) != len(draws):
            raise ValueError("source_row_hashes must align one-to-one with draw_ids")
    return draws, sources, rows


def _validate_mask_lineage(
    values: Sequence[str],
    *,
    base_mask_hash: str,
    current_mask_hash: str,
) -> tuple[str, ...]:
    lineage = _string_tuple(values, "committed_mask_lineage_hashes", unique=True)
    if lineage[0] != base_mask_hash:
        raise ValueError("mask lineage must begin at the exact chart base mask")
    if lineage[-1] != current_mask_hash:
        raise ValueError("mask lineage must end at the exact current mask")
    if base_mask_hash == current_mask_hash and lineage != (base_mask_hash,):
        raise ValueError("an unchanged chart base has a singleton mask lineage")
    return lineage


def _validate_mask_transition_receipts(
    values: Sequence[str],
    *,
    lineage_length: int,
) -> tuple[str, ...]:
    receipts = tuple(str(value) for value in values)
    if any(not value for value in receipts) or len(set(receipts)) != len(receipts):
        raise ValueError("mask-transition receipt hashes must be unique and nonempty")
    expected = lineage_length - 1
    if len(receipts) != expected:
        raise ValueError(
            "exact mask ancestry requires one transition receipt for each parent-to-child edge"
        )
    return receipts


@dataclass(frozen=True, slots=True)
class V1OwnerRegistration:
    """Owner-resolved preregistration identity that is always non-executable."""

    registration_id: str
    referent_id: str
    route_axis_hash: str
    context_population_hash: str
    calibration_identity: str
    certificate_family: CertificateFamily | None = None
    cross_query_shrinkage: CrossQueryShrinkageDisposition | None = None
    a1_coordinates_hash: str | None = None
    a1_proof_hash: str | None = None
    a2_coordinates_hash: str | None = None
    owner_signature: str | None = None
    owner_signature_date: str | None = None
    owner_signature_verification_hash: str | None = None
    campaign_contract_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _OWNER_REGISTRATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "registration_id",
            "referent_id",
            "route_axis_hash",
            "context_population_hash",
            "calibration_identity",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "schema",
            _exact_schema(self.schema, _OWNER_REGISTRATION_SCHEMA, "owner registration"),
        )
        if self.certificate_family is not None and self.certificate_family not in {
            "staged_alpha",
            "grow_eprocess",
        }:
            raise ValueError("certificate_family must be staged_alpha or grow_eprocess")
        if self.cross_query_shrinkage is not None and self.cross_query_shrinkage not in {
            "defer",
            "typed_amendment",
        }:
            raise ValueError("cross_query_shrinkage must be defer or typed_amendment")
        for name in (
            "campaign_contract_hash",
            "a1_coordinates_hash",
            "a1_proof_hash",
            "a2_coordinates_hash",
            "owner_signature",
            "owner_signature_date",
            "owner_signature_verification_hash",
        ):
            object.__setattr__(self, name, _hash_or_none(getattr(self, name), name))
        selected_fields = (
            self.certificate_family,
            self.cross_query_shrinkage,
            self.a1_coordinates_hash,
            self.a1_proof_hash,
            self.a2_coordinates_hash,
            self.owner_signature,
            self.owner_signature_date,
            self.owner_signature_verification_hash,
        )
        if (
            any(value is not None for value in selected_fields)
            and self.campaign_contract_hash is None
        ):
            raise ValueError(
                "selected modern-v1 registration fields require a campaign_contract_hash"
            )
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata, "registration metadata"),
        )

    @property
    def registration_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-registration-")

    @property
    def execution_ready(self) -> bool:
        # This is intentionally not a function of strings on this object.
        return False

    @property
    def execution_refusal_reason(self) -> str:
        if self.certificate_family is None:
            return "A1 certificate_family is unresolved"
        if self.cross_query_shrinkage is None:
            return "A2 cross_query_shrinkage disposition is unresolved"
        if self.campaign_contract_hash is None:
            return "the immutable campaign contract is not attached"
        if self.a1_coordinates_hash is None:
            return "A1 coordinates are not attached"
        if self.a1_proof_hash is None:
            return "A1 proof artifact is not attached"
        if self.cross_query_shrinkage == "typed_amendment" and self.a2_coordinates_hash is None:
            return "A2 typed amendment coordinates are not attached"
        if self.owner_signature is None or self.owner_signature_date is None:
            return "owner signature is absent"
        if self.owner_signature_verification_hash is None:
            return "owner signature is not verified by an external authority gate"
        return (
            "v1 execution is not implemented or authorized by this inert contract; "
            "no registration field can enable it"
        )

    def assert_executable(self) -> None:
        raise PermissionError(f"modern v1 execution is prohibited: {self.execution_refusal_reason}")


@dataclass(frozen=True, slots=True)
class V1CertificateFreeze:
    """Pre-draw immutable identity for one candidate certificate."""

    registration_hash: str
    parent_mask_hash: str
    candidate_mask_hash: str
    candidate_definition_hash: str
    null_id: str
    loss_id: str
    sampling_law_hash: str
    importance_weight_hash: str
    certificate_factor_hash: str
    pre_draw_filtration_hash: str
    allowed_roles: tuple[V1StreamRole, ...] = CERTIFICATE_STREAM_ROLES
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _CERTIFICATE_FREEZE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "parent_mask_hash",
            "candidate_mask_hash",
            "candidate_definition_hash",
            "null_id",
            "loss_id",
            "sampling_law_hash",
            "importance_weight_hash",
            "certificate_factor_hash",
            "pre_draw_filtration_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "schema",
            _exact_schema(self.schema, _CERTIFICATE_FREEZE_SCHEMA, "certificate freeze"),
        )
        roles = tuple(self.allowed_roles)
        if (
            not roles
            or len(set(roles)) != len(roles)
            or any(role not in CERTIFICATE_STREAM_ROLES for role in roles)
        ):
            raise ValueError("certificate freeze requires unique certificate roles only")
        object.__setattr__(self, "allowed_roles", roles)
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata, "certificate freeze metadata"),
        )

    @property
    def freeze_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-freeze-")


@dataclass(frozen=True, slots=True)
class V1EvidencePacket:
    """Immutable evidence with distinct draw and physical-source lineage."""

    packet_id: str
    role: V1StreamRole
    draw_ids: tuple[str, ...]
    source_context_ids: tuple[str, ...]
    registration_hash: str
    parent_mask_hash: str
    candidate_mask_hash: str | None = None
    candidate_definition_hash: str | None = None
    null_id: str | None = None
    loss_id: str | None = None
    sampling_law_hash: str | None = None
    importance_weight_hash: str | None = None
    certificate_factor_hash: str | None = None
    pre_draw_filtration_hash: str | None = None
    outcome_observed: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _EVIDENCE_PACKET_SCHEMA

    def __post_init__(self) -> None:
        if self.role not in {
            PROPOSAL_STREAM_ROLE,
            "calibration_certificate",
            "oracle_certificate",
        }:
            raise ValueError("unknown v1 stream role")
        for name in ("packet_id", "registration_hash", "parent_mask_hash"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "schema",
            _exact_schema(self.schema, _EVIDENCE_PACKET_SCHEMA, "evidence packet"),
        )
        draws, sources, _ = _validate_aligned_draw_lineage(
            draw_ids=self.draw_ids,
            source_context_ids=self.source_context_ids,
        )
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "source_context_ids", sources)
        for name in (
            "candidate_mask_hash",
            "candidate_definition_hash",
            "null_id",
            "loss_id",
            "sampling_law_hash",
            "importance_weight_hash",
            "certificate_factor_hash",
            "pre_draw_filtration_hash",
        ):
            object.__setattr__(self, name, _hash_or_none(getattr(self, name), name))
        if self.role in CERTIFICATE_STREAM_ROLES:
            missing = tuple(
                name
                for name in (
                    "candidate_mask_hash",
                    "candidate_definition_hash",
                    "null_id",
                    "loss_id",
                    "sampling_law_hash",
                    "importance_weight_hash",
                    "certificate_factor_hash",
                    "pre_draw_filtration_hash",
                )
                if getattr(self, name) is None
            )
            if missing:
                raise ValueError(
                    "certificate-role evidence packet is missing frozen fields: "
                    + ", ".join(missing)
                )
        object.__setattr__(
            self,
            "outcome_observed",
            _strict_bool(self.outcome_observed, "outcome_observed"),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata, "evidence packet metadata"),
        )

    @property
    def packet_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-packet-")

    @property
    def can_update_proposal(self) -> bool:
        return self.outcome_observed

    @property
    def can_enter_certificate(self) -> bool:
        return self.role in CERTIFICATE_STREAM_ROLES and self.outcome_observed


def assert_packet_matches_certificate_freeze(
    packet: V1EvidencePacket,
    freeze: V1CertificateFreeze,
) -> None:
    """Fail closed unless observed certificate evidence matches every freeze."""

    if packet.role not in CERTIFICATE_STREAM_ROLES:
        raise PermissionError("proposal-only evidence cannot enter a certificate")
    if not packet.outcome_observed:
        raise PermissionError("unobserved evidence cannot enter a certificate")
    if packet.role not in freeze.allowed_roles:
        raise PermissionError("packet role is not allowed by the certificate freeze")
    comparisons = {
        "registration_hash": packet.registration_hash,
        "parent_mask_hash": packet.parent_mask_hash,
        "candidate_mask_hash": packet.candidate_mask_hash,
        "candidate_definition_hash": packet.candidate_definition_hash,
        "null_id": packet.null_id,
        "loss_id": packet.loss_id,
        "sampling_law_hash": packet.sampling_law_hash,
        "importance_weight_hash": packet.importance_weight_hash,
        "certificate_factor_hash": packet.certificate_factor_hash,
        "pre_draw_filtration_hash": packet.pre_draw_filtration_hash,
    }
    for name, packet_value in comparisons.items():
        if packet_value != getattr(freeze, name):
            raise ValueError(f"certificate packet changed frozen field {name}")


@dataclass(frozen=True, slots=True)
class V1CertificateLedger:
    """Immutable packet ledger with global, not merely per-packet, draw use."""

    freeze: V1CertificateFreeze
    admitted_packets: tuple[V1EvidencePacket, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _CERTIFICATE_LEDGER_SCHEMA

    def __post_init__(self) -> None:
        packets = tuple(self.admitted_packets)
        seen_hashes: set[str] = set()
        seen_draws: set[str] = set()
        for packet in packets:
            if not isinstance(packet, V1EvidencePacket):
                raise TypeError("admitted_packets must contain V1EvidencePacket values")
            assert_packet_matches_certificate_freeze(packet, self.freeze)
            if packet.packet_hash in seen_hashes:
                raise ValueError("certificate packet was already admitted")
            overlap = seen_draws.intersection(packet.draw_ids)
            if overlap:
                raise ValueError(
                    "certificate draw IDs cannot be reused across packets: "
                    + ", ".join(sorted(overlap))
                )
            seen_hashes.add(packet.packet_hash)
            seen_draws.update(packet.draw_ids)
        object.__setattr__(self, "admitted_packets", packets)
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata, "certificate ledger metadata"),
        )
        object.__setattr__(
            self,
            "schema",
            _exact_schema(
                self.schema,
                _CERTIFICATE_LEDGER_SCHEMA,
                "certificate ledger",
            ),
        )

    @property
    def admitted_packet_hashes(self) -> tuple[str, ...]:
        return tuple(packet.packet_hash for packet in self.admitted_packets)

    @property
    def admitted_draw_ids(self) -> tuple[str, ...]:
        return tuple(draw for packet in self.admitted_packets for draw in packet.draw_ids)

    @property
    def outcome_observed(self) -> bool:
        # Ledger state is derived from immutable admitted evidence; callers
        # cannot assert a contradictory outcome flag.
        return bool(self.admitted_packets)

    @property
    def ledger_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-certificate-ledger-")


def advance_certificate_ledger(
    ledger: V1CertificateLedger,
    packet: V1EvidencePacket,
) -> V1CertificateLedger:
    """Admit one observed packet without mutable caller-supplied outcome state."""

    assert_packet_matches_certificate_freeze(packet, ledger.freeze)
    if packet.packet_hash in ledger.admitted_packet_hashes:
        raise ValueError("certificate packet was already admitted")
    overlap = set(ledger.admitted_draw_ids).intersection(packet.draw_ids)
    if overlap:
        raise ValueError(
            "certificate draw IDs cannot be reused across packets: " + ", ".join(sorted(overlap))
        )
    return V1CertificateLedger(
        freeze=ledger.freeze,
        admitted_packets=ledger.admitted_packets + (packet,),
        metadata=ledger.metadata,
    )


def replace_certificate_freeze(
    ledger: V1CertificateLedger,
    replacement: V1CertificateFreeze,
) -> V1CertificateLedger:
    """Replace a candidate freeze only before any outcome packet is admitted."""

    if ledger.outcome_observed:
        raise PermissionError(
            "candidate-specific certificate outcomes were already observed; "
            "changing the candidate/null/loss/factor requires a fresh ledger"
        )
    return V1CertificateLedger(freeze=replacement, metadata=ledger.metadata)


@dataclass(frozen=True, slots=True)
class V1FrequentDirectionsLineage:
    """Deterministic FD proposal lineage, including IID draw/source separation."""

    algorithm_id: str
    implementation: V1FDImplementation
    route_axis_hash: str
    measured_mask_hash: str
    draw_ids: tuple[str, ...]
    source_context_ids: tuple[str, ...]
    source_row_hashes: tuple[str, ...]
    stream_weight_hash: str
    accumulator_context_axis_hash: str
    factor_hash: str
    factor_receipt_hash: str
    fd_shrink_receipt_hash: str
    exact_streamed_norm_receipt_hash: str
    fd_rank: int
    factor_row_count: int
    route_count: int
    fd_shrink: float
    exact_streamed_norm: float
    canonical_equivalence_receipt_hash: str | None = None
    retained_capacity_rank: int | None = None
    retained_capacity_factor_hash: str | None = None
    full_replay_source_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _FD_LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "algorithm_id",
            "route_axis_hash",
            "measured_mask_hash",
            "stream_weight_hash",
            "accumulator_context_axis_hash",
            "factor_hash",
            "factor_receipt_hash",
            "fd_shrink_receipt_hash",
            "exact_streamed_norm_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "schema",
            _exact_schema(self.schema, _FD_LINEAGE_SCHEMA, "FD lineage"),
        )
        if self.algorithm_id != _FD_ALGORITHM_ID:
            raise ValueError("unsupported or non-deterministic FD algorithm")
        if self.implementation not in {
            "canonical_numpy",
            "cuda_algebraically_equivalent",
            "cuda_gram_algebraically_equivalent",
        }:
            raise ValueError("unsupported FD implementation")
        equivalence = _hash_or_none(
            self.canonical_equivalence_receipt_hash,
            "canonical_equivalence_receipt_hash",
        )
        if self.implementation in {
            "cuda_algebraically_equivalent",
            "cuda_gram_algebraically_equivalent",
        } and equivalence is None:
            raise ValueError("CUDA FD lineage requires a canonical-equivalence receipt")
        object.__setattr__(self, "canonical_equivalence_receipt_hash", equivalence)
        draws, sources, rows = _validate_aligned_draw_lineage(
            draw_ids=self.draw_ids,
            source_context_ids=self.source_context_ids,
            source_row_hashes=self.source_row_hashes,
        )
        if rows is None:  # defensive: source_row_hashes is required for FD lineage
            raise ValueError("FD lineage requires source row hashes")
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "source_context_ids", sources)
        object.__setattr__(self, "source_row_hashes", rows)
        expected_axis = content_hash(draws, prefix="irg-trust-context-axis-")
        if self.accumulator_context_axis_hash != expected_axis:
            raise ValueError(
                "FD accumulator context axis must hash unique draw_ids, not "
                "possibly repeated physical source_context_ids"
            )
        rank = _positive_int(self.fd_rank, "fd_rank")
        factor_rows = _positive_int(self.factor_row_count, "factor_row_count")
        route_count = _positive_int(self.route_count, "route_count")
        if factor_rows > rank or rank > route_count:
            raise ValueError("FD factor dimensions must satisfy rows <= rank <= routes")
        object.__setattr__(self, "fd_rank", rank)
        object.__setattr__(self, "factor_row_count", factor_rows)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(
            self,
            "fd_shrink",
            _finite_nonnegative(self.fd_shrink, "fd_shrink"),
        )
        object.__setattr__(
            self,
            "exact_streamed_norm",
            _finite_nonnegative(self.exact_streamed_norm, "exact_streamed_norm"),
        )
        retained = self.retained_capacity_rank
        retained_factor = _hash_or_none(
            self.retained_capacity_factor_hash,
            "retained_capacity_factor_hash",
        )
        if retained is not None:
            retained = _positive_int(retained, "retained_capacity_rank")
            if retained < rank:
                raise ValueError("retained_capacity_rank cannot be below fd_rank")
            if retained > rank and retained_factor is None:
                raise ValueError("rank growth from retained capacity requires its factor hash")
        elif retained_factor is not None:
            raise ValueError("retained capacity factor requires retained_capacity_rank")
        object.__setattr__(self, "retained_capacity_rank", retained)
        object.__setattr__(self, "retained_capacity_factor_hash", retained_factor)
        object.__setattr__(
            self,
            "full_replay_source_hash",
            _hash_or_none(self.full_replay_source_hash, "full_replay_source_hash"),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata, "FD lineage metadata"),
        )

    @property
    def draw_axis_hash(self) -> str:
        return content_hash(self.draw_ids, prefix="partition-free-v1-draw-axis-")

    @property
    def physical_source_axis_hash(self) -> str:
        return content_hash(
            self.source_context_ids,
            prefix="partition-free-v1-physical-source-axis-",
        )

    @property
    def lineage_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-fd-lineage-")

    @property
    def can_certify_decision(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class V1FDLineageVerificationReceipt:
    """Adapter-issued, proposal-only verification of one FD lineage."""

    lineage_hash: str
    source_state_hash: str
    source_ledger_hash: str
    accumulator_derivation_receipt_hash: str
    ordered_source_rows_hash: str
    stream_weight_hash: str
    factor_derivation_receipt_hash: str
    fd_shrink_derivation_receipt_hash: str
    exact_norm_derivation_receipt_hash: str
    verified_algebra_hash: str
    proposal_only: bool = True
    claim_execution_certified: bool = False
    schema: str = _FD_VERIFICATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "lineage_hash",
            "source_state_hash",
            "source_ledger_hash",
            "accumulator_derivation_receipt_hash",
            "ordered_source_rows_hash",
            "stream_weight_hash",
            "factor_derivation_receipt_hash",
            "fd_shrink_derivation_receipt_hash",
            "exact_norm_derivation_receipt_hash",
            "verified_algebra_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "schema",
            _exact_schema(
                self.schema,
                _FD_VERIFICATION_RECEIPT_SCHEMA,
                "FD verification receipt",
            ),
        )
        if not _strict_bool(self.proposal_only, "proposal_only"):
            raise ValueError("FD verification receipts must remain proposal-only")
        if _strict_bool(
            self.claim_execution_certified,
            "claim_execution_certified",
        ):
            raise ValueError("FD verification receipts cannot certify execution")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-fd-lineage-verification-",
        )

    @property
    def can_certify_decision(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class V1InterventionalCorrectionLineage:
    """Identified coordinate correction anchored by finite interventions."""

    correction_kind: V1CorrectionKind
    base_mask_hash: str
    route_axis_hash: str
    referent_id: str
    context_mixture_hash: str
    calibration_identity: str
    feature_map_hash: str
    estimand_contrast_hash: str
    observation_model_hash: str
    interventional_support_hash: str
    design_matrix_hash: str
    identification_receipt_hash: str
    corrected_price_hash: str
    corrected_price_receipt_hash: str
    atom_cost_hash: str
    atom_cost_receipt_hash: str
    measured_action_hashes: tuple[str, ...]
    intervention_draw_ids: tuple[str, ...]
    identified_design_rank: int
    required_design_rank: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _CORRECTION_LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.correction_kind != "identified_finite_intervention_v1":
            raise ValueError("unsupported interventional correction")
        for name in (
            "base_mask_hash",
            "route_axis_hash",
            "referent_id",
            "context_mixture_hash",
            "calibration_identity",
            "feature_map_hash",
            "estimand_contrast_hash",
            "observation_model_hash",
            "interventional_support_hash",
            "design_matrix_hash",
            "identification_receipt_hash",
            "corrected_price_hash",
            "corrected_price_receipt_hash",
            "atom_cost_hash",
            "atom_cost_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "schema",
            _exact_schema(
                self.schema,
                _CORRECTION_LINEAGE_SCHEMA,
                "interventional correction lineage",
            ),
        )
        object.__setattr__(
            self,
            "measured_action_hashes",
            _string_tuple(
                self.measured_action_hashes,
                "measured_action_hashes",
                unique=True,
            ),
        )
        object.__setattr__(
            self,
            "intervention_draw_ids",
            _string_tuple(
                self.intervention_draw_ids,
                "intervention_draw_ids",
                unique=True,
            ),
        )
        identified = _positive_int(
            self.identified_design_rank,
            "identified_design_rank",
        )
        required = _positive_int(self.required_design_rank, "required_design_rank")
        if identified < required:
            raise ValueError("interventional correction is underidentified for its feature map")
        if identified > len(self.measured_action_hashes):
            raise ValueError(
                "identified design rank exceeds the number of independent measured actions"
            )
        object.__setattr__(self, "identified_design_rank", identified)
        object.__setattr__(self, "required_design_rank", required)
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata, "correction lineage metadata"),
        )

    @property
    def lineage_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-correction-lineage-")

    @property
    def can_certify_decision(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class V1AnalyticExactGateCorrectionLineage:
    """Exact raw-logit Aumann--Shapley proposal lineage.

    This is the canonical v1 correction: the post-activation conserving
    identity supplies ``-G*phi*c`` directly.  It contains no child mask,
    parent/child contrast, fitted residual, or oracle outcome.
    """

    base_mask_hash: str
    route_axis_hash: str
    referent_id: str
    context_mixture_hash: str
    calibration_identity: str
    source_attribution_hash: str
    streaming_receipt_hash: str
    tangent_price_state_hash: str
    corrected_price_hash: str
    corrected_price_receipt_hash: str
    absolute_salience_hash: str
    atom_cost_hash: str
    atom_cost_receipt_hash: str
    price_orientation_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    correction_kind: V1CorrectionKind = "analytic_exact_gate_raw_target_logit_v1"
    raw_target_logit_unit_cotangent: bool = True
    empirical_child_parent_mask_evaluation_count: int = 0
    schema: str = _ANALYTIC_EXACT_GATE_LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.correction_kind != "analytic_exact_gate_raw_target_logit_v1":
            raise ValueError("unsupported analytic exact-gate correction")
        for name in (
            "base_mask_hash",
            "route_axis_hash",
            "referent_id",
            "context_mixture_hash",
            "calibration_identity",
            "source_attribution_hash",
            "streaming_receipt_hash",
            "tangent_price_state_hash",
            "corrected_price_hash",
            "corrected_price_receipt_hash",
            "absolute_salience_hash",
            "atom_cost_hash",
            "atom_cost_receipt_hash",
            "price_orientation_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.raw_target_logit_unit_cotangent is not True:
            raise ValueError("analytic exact-gate correction requires the unit cotangent")
        if (
            isinstance(self.empirical_child_parent_mask_evaluation_count, bool)
            or int(self.empirical_child_parent_mask_evaluation_count) != 0
        ):
            raise ValueError(
                "analytic exact-gate correction forbids child/parent mask evaluations"
            )
        object.__setattr__(
            self,
            "schema",
            _exact_schema(
                self.schema,
                _ANALYTIC_EXACT_GATE_LINEAGE_SCHEMA,
                "analytic exact-gate correction lineage",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata, "analytic exact-gate metadata"),
        )

    @property
    def lineage_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-analytic-exact-gate-lineage-",
        )

    @property
    def can_certify_decision(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class V1AnalyticExactGateCorrectionVerificationReceipt:
    """Independent hash/algebra check for an analytic exact-gate result."""

    lineage_hash: str
    result_receipt_hash: str
    corrected_price_values_hash: str
    absolute_salience_values_hash: str
    source_attribution_hash: str
    streaming_receipt_hash: str
    verified_algebra_hash: str
    proposal_only: bool = True
    claim_execution_certified: bool = False
    schema: str = _ANALYTIC_EXACT_GATE_VERIFICATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "lineage_hash",
            "result_receipt_hash",
            "corrected_price_values_hash",
            "absolute_salience_values_hash",
            "source_attribution_hash",
            "streaming_receipt_hash",
            "verified_algebra_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.proposal_only is not True or self.claim_execution_certified is not False:
            raise ValueError("analytic exact-gate verification is proposal-only")
        object.__setattr__(
            self,
            "schema",
            _exact_schema(
                self.schema,
                _ANALYTIC_EXACT_GATE_VERIFICATION_SCHEMA,
                "analytic exact-gate verification receipt",
            ),
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-analytic-exact-gate-verification-",
        )

    @property
    def can_certify_decision(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class V1InterventionalCorrectionVerificationReceipt:
    """Locally checked algebra binding an adapter result to its lineage."""

    lineage_hash: str
    adapter_result_receipt_hash: str
    corrected_price_values_hash: str
    design_matrix_values_hash: str
    correction_coefficients_hash: str
    observed_action_residuals_hash: str
    fitted_action_residuals_hash: str
    verified_algebra_hash: str
    proposal_only: bool = True
    claim_execution_certified: bool = False
    schema: str = _CORRECTION_VERIFICATION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "lineage_hash",
            "adapter_result_receipt_hash",
            "corrected_price_values_hash",
            "design_matrix_values_hash",
            "correction_coefficients_hash",
            "observed_action_residuals_hash",
            "fitted_action_residuals_hash",
            "verified_algebra_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "schema",
            _exact_schema(
                self.schema,
                _CORRECTION_VERIFICATION_RECEIPT_SCHEMA,
                "correction verification receipt",
            ),
        )
        if not _strict_bool(self.proposal_only, "proposal_only"):
            raise ValueError("correction verification receipts must remain proposal-only")
        if _strict_bool(
            self.claim_execution_certified,
            "claim_execution_certified",
        ):
            raise ValueError("correction verification receipts cannot certify execution")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-correction-verification-",
        )

    @property
    def can_certify_decision(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class V1ChartThresholds:
    """Frozen registration coordinates for chart construction and reuse."""

    action_leverage_bound: float
    metric_path_length_bound: float
    layer_local_leverage_bound: float
    residual_bound: float
    fd_shrink_bound: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _CHART_THRESHOLDS_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "action_leverage_bound",
            "metric_path_length_bound",
            "layer_local_leverage_bound",
            "residual_bound",
            "fd_shrink_bound",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "schema",
            _exact_schema(
                self.schema,
                _CHART_THRESHOLDS_SCHEMA,
                "chart thresholds",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata, "chart threshold metadata"),
        )

    @property
    def threshold_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-chart-thresholds-")


@dataclass(frozen=True, slots=True)
class V1FiniteActionClassProof:
    """Finite proof declaration binding the exact ordered action membership.

    This object proves no inequality by construction.  It only declares the
    finite actions to which an external proof artifact applies, and binds that
    declaration to the exact chart mask and physical route axis.  Consumers
    must recompute ``proof_hash`` and membership before using an action-class
    bound.
    """

    route_axis_hash: str
    route_count: int
    base_mask_hash: str
    current_mask_hash: str
    ordered_removal_sets: tuple[tuple[int, ...], ...]
    proof_artifact_hash: str
    construction_receipt_hash: str
    schema: str = _FINITE_ACTION_CLASS_PROOF_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "route_axis_hash",
            "base_mask_hash",
            "current_mask_hash",
            "proof_artifact_hash",
            "construction_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        route_count = _positive_int(self.route_count, "route_count")
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(
            self,
            "ordered_removal_sets",
            _ordered_finite_removal_sets(
                self.ordered_removal_sets,
                route_count=route_count,
                label="finite action-class proof",
            ),
        )
        object.__setattr__(
            self,
            "schema",
            _exact_schema(
                self.schema,
                _FINITE_ACTION_CLASS_PROOF_SCHEMA,
                "finite action-class proof",
            ),
        )

    @property
    def proof_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-finite-action-class-proof-")

    def contains(self, removal_set: Sequence[int]) -> bool:
        action = _ordered_finite_removal_sets(
            (removal_set,),
            route_count=self.route_count,
            label="candidate action-class membership",
        )[0]
        return action in self.ordered_removal_sets

    @property
    def can_certify_decision(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class V1MixedDifferenceEnvelope:
    """Frozen, proof-bound envelope that alone may permit chart reuse."""

    proof_basis: V1MixedDifferenceProofBasis
    proof_artifact_hash: str
    action_class_proof: V1FiniteActionClassProof
    base_mask_hash: str
    route_axis_hash: str
    referent_id: str
    context_mixture_hash: str
    calibration_identity: str
    threshold_hash: str
    construction_receipt_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _MIXED_DIFFERENCE_ENVELOPE_SCHEMA

    def __post_init__(self) -> None:
        if self.proof_basis not in {
            "smoothness_theorem",
            "bounded_mixed_finite_difference",
            "interventional_coverage",
        }:
            raise ValueError("unsupported mixed-difference proof basis")
        for name in (
            "proof_artifact_hash",
            "base_mask_hash",
            "route_axis_hash",
            "referent_id",
            "context_mixture_hash",
            "calibration_identity",
            "threshold_hash",
            "construction_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if type(self.action_class_proof) is not V1FiniteActionClassProof:
            raise TypeError("mixed-difference envelope requires an exact finite action-class proof")
        if self.action_class_proof.proof_artifact_hash != self.proof_artifact_hash:
            raise ValueError("action-class proof names another proof artifact")
        if self.action_class_proof.route_axis_hash != self.route_axis_hash:
            raise ValueError("action-class proof names another route axis")
        if self.action_class_proof.base_mask_hash != self.base_mask_hash:
            raise ValueError("action-class proof names another chart base mask")
        object.__setattr__(
            self,
            "schema",
            _exact_schema(
                self.schema,
                _MIXED_DIFFERENCE_ENVELOPE_SCHEMA,
                "mixed-difference envelope",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata, "mixed-difference metadata"),
        )

    @property
    def envelope_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-mixed-difference-")

    @property
    def action_class_hash(self) -> str:
        """Compatibility name for the typed finite declaration's content hash."""

        return self.action_class_proof.proof_hash


@dataclass(frozen=True, slots=True)
class V1ProposalChartBuildContract:
    """Identity and frozen-coordinate input to the proposal-chart builder."""

    registration_hash: str
    base_mask_hash: str
    current_mask_hash: str
    cumulative_removal_hash: str
    committed_mask_lineage_hashes: tuple[str, ...]
    committed_mask_transition_receipt_hashes: tuple[str, ...]
    route_axis_hash: str
    live_route_axis_hash: str
    referent_id: str
    context_mixture_hash: str
    calibration_identity: str
    thresholds: V1ChartThresholds
    residual_diagonal_status: V1ResidualDiagonalStatus = "absent"
    residual_diagonal_receipt_hash: str | None = None
    residual_bound_proof_hash: str | None = None
    mixed_difference_envelope: V1MixedDifferenceEnvelope | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _CHART_BUILD_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "base_mask_hash",
            "current_mask_hash",
            "cumulative_removal_hash",
            "route_axis_hash",
            "live_route_axis_hash",
            "referent_id",
            "context_mixture_hash",
            "calibration_identity",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "schema",
            _exact_schema(
                self.schema,
                _CHART_BUILD_CONTRACT_SCHEMA,
                "chart build contract",
            ),
        )
        if not isinstance(self.thresholds, V1ChartThresholds):
            raise TypeError("thresholds must be a frozen V1ChartThresholds object")
        lineage = _validate_mask_lineage(
            self.committed_mask_lineage_hashes,
            base_mask_hash=self.base_mask_hash,
            current_mask_hash=self.current_mask_hash,
        )
        object.__setattr__(
            self,
            "committed_mask_lineage_hashes",
            lineage,
        )
        object.__setattr__(
            self,
            "committed_mask_transition_receipt_hashes",
            _validate_mask_transition_receipts(
                self.committed_mask_transition_receipt_hashes,
                lineage_length=len(lineage),
            ),
        )
        if self.residual_diagonal_status not in {
            "absent",
            "proposal_only",
            "global_loewner_bound",
            "action_class_quadratic_bound",
        }:
            raise ValueError("invalid residual_diagonal_status")
        receipt = _hash_or_none(
            self.residual_diagonal_receipt_hash,
            "residual_diagonal_receipt_hash",
        )
        proof = _hash_or_none(
            self.residual_bound_proof_hash,
            "residual_bound_proof_hash",
        )
        if self.residual_diagonal_status != "absent" and receipt is None:
            raise ValueError("non-absent residual diagonal requires a receipt")
        if self.residual_diagonal_status == "absent" and receipt is not None:
            raise ValueError("absent residual diagonal cannot carry a receipt")
        if (
            self.residual_diagonal_status
            in {
                "global_loewner_bound",
                "action_class_quadratic_bound",
            }
            and proof is None
        ):
            raise ValueError("a residual bound label requires its proof hash")
        if self.residual_diagonal_status in {"absent", "proposal_only"} and proof is not None:
            raise ValueError("proposal-only/absent residuals cannot carry a bound proof")
        object.__setattr__(self, "residual_diagonal_receipt_hash", receipt)
        object.__setattr__(self, "residual_bound_proof_hash", proof)
        envelope = self.mixed_difference_envelope
        if envelope is not None:
            if not isinstance(envelope, V1MixedDifferenceEnvelope):
                raise TypeError("mixed_difference_envelope must be a proof-bound envelope object")
            envelope_comparisons = (
                ("base_mask_hash", envelope.base_mask_hash),
                ("route_axis_hash", envelope.route_axis_hash),
                ("referent_id", envelope.referent_id),
                ("context_mixture_hash", envelope.context_mixture_hash),
                ("calibration_identity", envelope.calibration_identity),
            )
            for name, observed in envelope_comparisons:
                if observed != getattr(self, name):
                    raise ValueError(f"mixed-difference envelope identity mismatch for {name}")
            if envelope.threshold_hash != self.thresholds.threshold_hash:
                raise ValueError("mixed-difference envelope does not bind the frozen thresholds")
            if envelope.action_class_proof.current_mask_hash != self.current_mask_hash:
                raise ValueError(
                    "mixed-difference action class does not bind the chart current mask"
                )
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata, "chart build metadata"),
        )

    @property
    def contract_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-chart-build-")


@dataclass(frozen=True, slots=True)
class V1ProposalChart:
    """Content-addressed, proposal-only local chart."""

    build_contract: V1ProposalChartBuildContract
    fd_lineage: V1FrequentDirectionsLineage
    correction_lineage: (
        V1InterventionalCorrectionLineage
        | V1AnalyticExactGateCorrectionLineage
    )
    fd_verification_receipt: V1FDLineageVerificationReceipt
    correction_verification_receipt: (
        V1InterventionalCorrectionVerificationReceipt
        | V1AnalyticExactGateCorrectionVerificationReceipt
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _PROPOSAL_CHART_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.build_contract, V1ProposalChartBuildContract):
            raise TypeError("build_contract must be V1ProposalChartBuildContract")
        if not isinstance(self.fd_lineage, V1FrequentDirectionsLineage):
            raise TypeError("missing or unsupported Frequent-Directions lineage")
        if type(self.correction_lineage) not in {
            V1InterventionalCorrectionLineage,
            V1AnalyticExactGateCorrectionLineage,
        }:
            raise TypeError("missing or unsupported proposal correction")
        if type(self.fd_verification_receipt) is not V1FDLineageVerificationReceipt:
            raise TypeError("chart requires an exact FD verification receipt")
        if type(self.correction_verification_receipt) not in {
            V1InterventionalCorrectionVerificationReceipt,
            V1AnalyticExactGateCorrectionVerificationReceipt,
        }:
            raise TypeError("chart requires an exact correction verification receipt")
        if (
            type(self.correction_lineage)
            is V1AnalyticExactGateCorrectionLineage
        ) != (
            type(self.correction_verification_receipt)
            is V1AnalyticExactGateCorrectionVerificationReceipt
        ):
            raise TypeError("chart correction lineage and verification kinds differ")
        contract = self.build_contract
        fd = self.fd_lineage
        correction = self.correction_lineage
        if self.fd_verification_receipt.lineage_hash != fd.lineage_hash:
            raise ValueError("FD verification receipt names another lineage")
        if self.correction_verification_receipt.lineage_hash != correction.lineage_hash:
            raise ValueError("correction verification receipt names another lineage")
        # Keep the two producers separate: dict merging would let one lineage
        # silently overwrite the other's value for a shared identity field.
        comparisons = (
            ("route_axis_hash", fd.route_axis_hash),
            ("base_mask_hash", fd.measured_mask_hash),
            ("base_mask_hash", correction.base_mask_hash),
            ("route_axis_hash", correction.route_axis_hash),
            ("referent_id", correction.referent_id),
            ("context_mixture_hash", correction.context_mixture_hash),
            ("calibration_identity", correction.calibration_identity),
        )
        for name, observed in comparisons:
            expected = getattr(contract, name)
            if observed != expected:
                raise ValueError(f"proposal-chart lineage mismatch for {name}")
        if fd.fd_shrink > contract.thresholds.fd_shrink_bound:
            raise ValueError("FD shrink exceeds the frozen chart threshold")
        object.__setattr__(
            self,
            "schema",
            _exact_schema(self.schema, _PROPOSAL_CHART_SCHEMA, "proposal chart"),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata, "proposal chart metadata"),
        )

    @property
    def registration_hash(self) -> str:
        return self.build_contract.registration_hash

    @property
    def base_mask_hash(self) -> str:
        return self.build_contract.base_mask_hash

    @property
    def current_mask_hash(self) -> str:
        return self.build_contract.current_mask_hash

    @property
    def cumulative_removal_hash(self) -> str:
        return self.build_contract.cumulative_removal_hash

    @property
    def committed_mask_lineage_hashes(self) -> tuple[str, ...]:
        return self.build_contract.committed_mask_lineage_hashes

    @property
    def committed_mask_transition_receipt_hashes(self) -> tuple[str, ...]:
        return self.build_contract.committed_mask_transition_receipt_hashes

    @property
    def route_axis_hash(self) -> str:
        return self.build_contract.route_axis_hash

    @property
    def live_route_axis_hash(self) -> str:
        return self.build_contract.live_route_axis_hash

    @property
    def referent_id(self) -> str:
        return self.build_contract.referent_id

    @property
    def context_mixture_hash(self) -> str:
        return self.build_contract.context_mixture_hash

    @property
    def calibration_identity(self) -> str:
        return self.build_contract.calibration_identity

    @property
    def mixed_difference_envelope_hash(self) -> str | None:
        envelope = self.build_contract.mixed_difference_envelope
        return None if envelope is None else envelope.envelope_hash

    @property
    def residual_diagonal_status(self) -> V1ResidualDiagonalStatus:
        return self.build_contract.residual_diagonal_status

    @property
    def thresholds(self) -> V1ChartThresholds:
        return self.build_contract.thresholds

    @property
    def metric_geometry_kind(self) -> V1MetricGeometryKind:
        return "frequent_directions_deterministic_v1"

    @property
    def chart_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-chart-")

    @property
    def default_reuse_requires_refresh_after_commit(self) -> bool:
        return self.mixed_difference_envelope_hash is None

    @property
    def can_certify_decision(self) -> bool:
        return False

    def assert_can_certify_decision(self) -> None:
        raise PermissionError(
            "proposal charts, FD factors, and residual labels cannot certify decisions"
        )


def build_v1_proposal_chart(
    contract: V1ProposalChartBuildContract,
    *,
    fd_lineage: object | None,
    correction_lineage: object | None,
    metadata: Mapping[str, Any] | None = None,
) -> V1ProposalChart:
    """Build only from adapter results whose algebra can be rechecked.

    Bare lineage dataclasses are descriptive records and are intentionally
    insufficient: their hash-shaped fields can be populated by any caller.
    The FD path therefore requires the exact adapter verification bundle, and
    the correction path requires the exact result bundle with its numerical
    arrays so this boundary can recompute the result receipt and equations.
    """

    from .modern_v1_fd_adapter import V1FDAdapterResult
    from .modern_v1_intervention_adapter import (
        V1AnalyticExactGateCorrectionResult,
        V1InterventionalCorrectionResult,
    )

    if fd_lineage is None:
        raise ValueError("proposal chart requires deterministic FD lineage")
    if correction_lineage is None:
        raise ValueError("proposal chart requires a typed proposal correction")
    if type(fd_lineage) is not V1FDAdapterResult:
        raise TypeError(
            "proposal chart requires an exact FD adapter verification bundle; "
            "bare V1FrequentDirectionsLineage values are not producer proof"
        )
    if type(correction_lineage) not in {
        V1InterventionalCorrectionResult,
        V1AnalyticExactGateCorrectionResult,
    }:
        raise TypeError(
            "proposal chart requires an exact correction adapter result; bare "
            "lineage values are not producer proof"
        )
    fd_result = fd_lineage
    correction_result = correction_lineage
    if fd_result.schema != "partition_free_v1_fd_adapter_result_v1":
        raise ValueError("unsupported modern-v1 FD adapter-result schema")
    if type(fd_result.lineage) is not V1FrequentDirectionsLineage:
        raise TypeError("FD adapter result contains a noncanonical lineage")
    if type(fd_result.verification_receipt) is not V1FDLineageVerificationReceipt:
        raise TypeError("FD adapter result lacks an exact verification receipt")
    if fd_result.verification_receipt.lineage_hash != fd_result.lineage.lineage_hash:
        raise ValueError("FD adapter verification does not bind its lineage")
    if type(correction_result) is V1AnalyticExactGateCorrectionResult:
        if (
            correction_result.schema
            != "partition_free_v1_analytic_exact_gate_correction_result_v1"
            or type(correction_result.lineage)
            is not V1AnalyticExactGateCorrectionLineage
        ):
            raise TypeError("analytic exact-gate result contains another lineage")
        corrected_price_values_hash = content_hash(
            correction_result.corrected_prices,
            prefix="partition-free-v1-corrected-price-values-",
        )
        absolute_salience_values_hash = content_hash(
            correction_result.absolute_salience,
            prefix="partition-free-v1-exact-gate-absolute-salience-values-",
        )
        lineage = correction_result.lineage
        if (
            corrected_price_values_hash != lineage.corrected_price_hash
            or absolute_salience_values_hash != lineage.absolute_salience_hash
        ):
            raise ValueError("analytic exact-gate values differ from lineage")
        expected_result_receipt_hash = content_hash(
            {
                "schema": correction_result.schema,
                "lineage_hash": lineage.lineage_hash,
                "corrected_price_hash": corrected_price_values_hash,
                "absolute_salience_hash": absolute_salience_values_hash,
                "analytic_identity_no_fitted_residual": True,
            },
            prefix="partition-free-v1-analytic-exact-gate-result-",
        )
        if expected_result_receipt_hash != correction_result.result_receipt_hash:
            raise ValueError("analytic exact-gate result receipt is not reproducible")
        verified_algebra_hash = content_hash(
            {
                "schema": "partition_free_v1_analytic_exact_gate_algebra_check_v1",
                "lineage_hash": lineage.lineage_hash,
                "result_receipt_hash": correction_result.result_receipt_hash,
                "corrected_price_values_hash": corrected_price_values_hash,
                "absolute_salience_values_hash": absolute_salience_values_hash,
                "source_attribution_hash": lineage.source_attribution_hash,
                "streaming_receipt_hash": lineage.streaming_receipt_hash,
                "raw_target_logit_unit_cotangent": True,
                "empirical_child_parent_mask_evaluation_count": 0,
            },
            prefix="partition-free-v1-analytic-exact-gate-algebra-check-",
        )
        correction_verification = (
            V1AnalyticExactGateCorrectionVerificationReceipt(
                lineage_hash=lineage.lineage_hash,
                result_receipt_hash=correction_result.result_receipt_hash,
                corrected_price_values_hash=corrected_price_values_hash,
                absolute_salience_values_hash=absolute_salience_values_hash,
                source_attribution_hash=lineage.source_attribution_hash,
                streaming_receipt_hash=lineage.streaming_receipt_hash,
                verified_algebra_hash=verified_algebra_hash,
            )
        )
        return V1ProposalChart(
            build_contract=contract,
            fd_lineage=fd_result.lineage,
            correction_lineage=lineage,
            fd_verification_receipt=fd_result.verification_receipt,
            correction_verification_receipt=correction_verification,
            metadata={} if metadata is None else metadata,
        )
    if correction_result.schema != ("partition_free_v1_interventional_correction_result_v1"):
        raise ValueError("unsupported interventional adapter-result schema")
    if type(correction_result.lineage) is not V1InterventionalCorrectionLineage:
        raise TypeError("correction result contains a noncanonical lineage")

    corrected_price_values_hash = content_hash(
        correction_result.corrected_prices,
        prefix="partition-free-v1-corrected-price-values-",
    )
    if corrected_price_values_hash != correction_result.lineage.corrected_price_hash:
        raise ValueError("correction result values differ from its corrected-price hash")
    recomputed_fitted = np.asarray(correction_result.design_matrix, dtype=np.float64) @ np.asarray(
        correction_result.correction_coefficients, dtype=np.float64
    )
    fitted = np.asarray(correction_result.fitted_action_residuals, dtype=np.float64)
    if not np.allclose(recomputed_fitted, fitted, rtol=1e-12, atol=1e-12):
        raise ValueError("correction result fitted residuals violate its design algebra")
    expected_result_receipt_hash = content_hash(
        {
            "lineage_hash": correction_result.lineage.lineage_hash,
            "corrected_price_hash": corrected_price_values_hash,
            "correction_coefficients": correction_result.correction_coefficients,
            "observed_action_residuals": (correction_result.observed_action_residuals),
            "fitted_action_residuals": correction_result.fitted_action_residuals,
        },
        prefix="partition-free-v1-correction-result-",
    )
    if expected_result_receipt_hash != correction_result.result_receipt_hash:
        raise ValueError("correction adapter result receipt is not reproducible")
    design_matrix_values_hash = content_hash(
        correction_result.design_matrix,
        prefix="partition-free-v1-correction-design-values-",
    )
    correction_coefficients_hash = content_hash(
        correction_result.correction_coefficients,
        prefix="partition-free-v1-correction-coefficients-",
    )
    observed_residuals_hash = content_hash(
        correction_result.observed_action_residuals,
        prefix="partition-free-v1-correction-observed-residuals-",
    )
    fitted_residuals_hash = content_hash(
        correction_result.fitted_action_residuals,
        prefix="partition-free-v1-correction-fitted-residuals-",
    )
    verified_algebra_hash = content_hash(
        {
            "schema": "partition_free_v1_correction_algebra_check_v1",
            "lineage_hash": correction_result.lineage.lineage_hash,
            "adapter_result_receipt_hash": correction_result.result_receipt_hash,
            "corrected_price_values_hash": corrected_price_values_hash,
            "design_matrix_values_hash": design_matrix_values_hash,
            "correction_coefficients_hash": correction_coefficients_hash,
            "observed_action_residuals_hash": observed_residuals_hash,
            "fitted_action_residuals_hash": fitted_residuals_hash,
            "fitted_equals_design_times_coefficients": True,
        },
        prefix="partition-free-v1-correction-algebra-check-",
    )
    correction_verification = V1InterventionalCorrectionVerificationReceipt(
        lineage_hash=correction_result.lineage.lineage_hash,
        adapter_result_receipt_hash=correction_result.result_receipt_hash,
        corrected_price_values_hash=corrected_price_values_hash,
        design_matrix_values_hash=design_matrix_values_hash,
        correction_coefficients_hash=correction_coefficients_hash,
        observed_action_residuals_hash=observed_residuals_hash,
        fitted_action_residuals_hash=fitted_residuals_hash,
        verified_algebra_hash=verified_algebra_hash,
    )
    return V1ProposalChart(
        build_contract=contract,
        fd_lineage=fd_result.lineage,
        correction_lineage=correction_result.lineage,
        fd_verification_receipt=fd_result.verification_receipt,
        correction_verification_receipt=correction_verification,
        metadata={} if metadata is None else metadata,
    )


@dataclass(frozen=True, slots=True)
class V1ChartReuseRequest:
    """Observed proposal state supplied to the frozen chart-reuse gates."""

    current_mask_hash: str
    committed_mask_lineage_hashes: tuple[str, ...]
    committed_mask_transition_receipt_hashes: tuple[str, ...]
    route_axis_hash: str
    live_route_axis_hash: str
    referent_id: str
    context_mixture_hash: str
    calibration_identity: str
    candidate_action_hash: str
    candidate_action_class_hash: str
    cumulative_removal_hash: str
    action_leverage: float
    metric_path_length: float
    layer_local_leverage: float
    residual_interval_lower: float
    residual_interval_upper: float
    realized_residuals_compatible: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _CHART_REUSE_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "current_mask_hash",
            "route_axis_hash",
            "live_route_axis_hash",
            "referent_id",
            "context_mixture_hash",
            "calibration_identity",
            "candidate_action_hash",
            "candidate_action_class_hash",
            "cumulative_removal_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        lineage = _string_tuple(
            self.committed_mask_lineage_hashes,
            "committed_mask_lineage_hashes",
            unique=True,
        )
        if lineage[-1] != self.current_mask_hash:
            raise ValueError("reuse-request mask lineage must end at current_mask_hash")
        object.__setattr__(self, "committed_mask_lineage_hashes", lineage)
        object.__setattr__(
            self,
            "committed_mask_transition_receipt_hashes",
            _validate_mask_transition_receipts(
                self.committed_mask_transition_receipt_hashes,
                lineage_length=len(lineage),
            ),
        )
        for name in ("action_leverage", "metric_path_length", "layer_local_leverage"):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        lower = float(self.residual_interval_lower)
        upper = float(self.residual_interval_upper)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError("residual interval must be finite and ordered")
        object.__setattr__(self, "residual_interval_lower", lower)
        object.__setattr__(self, "residual_interval_upper", upper)
        object.__setattr__(
            self,
            "realized_residuals_compatible",
            _strict_bool(
                self.realized_residuals_compatible,
                "realized_residuals_compatible",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata, "chart reuse metadata"),
        )
        object.__setattr__(
            self,
            "schema",
            _exact_schema(
                self.schema,
                _CHART_REUSE_REQUEST_SCHEMA,
                "chart reuse request",
            ),
        )


@dataclass(frozen=True, slots=True)
class V1ChartReuseDecision:
    """Proposal-only relinearization result; never a mask decision."""

    action: V1ChartReuseAction
    reason: str
    chart_hash: str
    candidate_action_hash: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema: str = _CHART_REUSE_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if self.action not in {"reuse", "refresh", "refuse"}:
            raise ValueError("invalid chart reuse action")
        for name in ("reason", "chart_hash", "candidate_action_hash"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "schema",
            _exact_schema(
                self.schema,
                _CHART_REUSE_DECISION_SCHEMA,
                "chart reuse decision",
            ),
        )
        object.__setattr__(
            self,
            "diagnostics",
            _immutable_metadata(self.diagnostics, "chart reuse diagnostics"),
        )

    @property
    def decision_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-chart-reuse-")

    @property
    def can_accept_mask(self) -> bool:
        return False


def adjudicate_chart_reuse(
    chart: V1ProposalChart,
    request: V1ChartReuseRequest,
) -> V1ChartReuseDecision:
    """Evaluate dynamic relinearization without granting decision authority."""

    diagnostics: dict[str, Any] = {
        "chart_hash": chart.chart_hash,
        "mixed_difference_envelope_available": (chart.mixed_difference_envelope_hash is not None),
        "residual_diagonal_status": chart.residual_diagonal_status,
        "metric_geometry_kind": chart.metric_geometry_kind,
        "proposal_only": True,
    }
    identity_fields = (
        "current_mask_hash",
        "cumulative_removal_hash",
        "committed_mask_lineage_hashes",
        "committed_mask_transition_receipt_hashes",
        "route_axis_hash",
        "live_route_axis_hash",
        "referent_id",
        "context_mixture_hash",
        "calibration_identity",
    )
    for name in identity_fields:
        if getattr(chart, name) != getattr(request, name):
            diagnostics["mismatched_field"] = name
            return V1ChartReuseDecision(
                action="refuse",
                reason="chart identity differs from request",
                chart_hash=chart.chart_hash,
                candidate_action_hash=request.candidate_action_hash,
                diagnostics=diagnostics,
            )
    if chart.mixed_difference_envelope_hash is None:
        return V1ChartReuseDecision(
            action="refresh",
            reason="missing mixed-difference staleness envelope",
            chart_hash=chart.chart_hash,
            candidate_action_hash=request.candidate_action_hash,
            diagnostics=diagnostics,
        )
    envelope = chart.build_contract.mixed_difference_envelope
    if envelope is None:  # pragma: no cover - guarded by the hash check above
        raise AssertionError("mixed-difference envelope identity is inconsistent")
    if request.candidate_action_class_hash != envelope.action_class_hash:
        diagnostics["mismatched_field"] = "candidate_action_class_hash"
        return V1ChartReuseDecision(
            action="refuse",
            reason="candidate is outside the frozen envelope action class",
            chart_hash=chart.chart_hash,
            candidate_action_hash=request.candidate_action_hash,
            diagnostics=diagnostics,
        )
    thresholds = chart.thresholds
    threshold_checks = {
        "action_leverage": (request.action_leverage, thresholds.action_leverage_bound),
        "metric_path_length": (
            request.metric_path_length,
            thresholds.metric_path_length_bound,
        ),
        "layer_local_leverage": (
            request.layer_local_leverage,
            thresholds.layer_local_leverage_bound,
        ),
        "residual_interval": (
            max(
                abs(request.residual_interval_lower),
                abs(request.residual_interval_upper),
            ),
            thresholds.residual_bound,
        ),
    }
    diagnostics["threshold_checks"] = threshold_checks
    for name, (value, bound) in threshold_checks.items():
        if value > bound:
            diagnostics["failed_gate"] = name
            return V1ChartReuseDecision(
                action="refresh",
                reason=f"{name} exceeds registered reuse bound",
                chart_hash=chart.chart_hash,
                candidate_action_hash=request.candidate_action_hash,
                diagnostics=diagnostics,
            )
    if not request.realized_residuals_compatible:
        return V1ChartReuseDecision(
            action="refresh",
            reason="realized residuals violate the staleness model",
            chart_hash=chart.chart_hash,
            candidate_action_hash=request.candidate_action_hash,
            diagnostics=diagnostics,
        )
    return V1ChartReuseDecision(
        action="reuse",
        reason="all registered chart-reuse gates are satisfied",
        chart_hash=chart.chart_hash,
        candidate_action_hash=request.candidate_action_hash,
        diagnostics=diagnostics,
    )


__all__ = [
    "MODERN_V1_WORKFLOW_IDENTITY",
    "CERTIFICATE_STREAM_ROLES",
    "PROPOSAL_STREAM_ROLE",
    "CertificateFamily",
    "CrossQueryShrinkageDisposition",
    "V1AnalyticExactGateCorrectionLineage",
    "V1AnalyticExactGateCorrectionVerificationReceipt",
    "V1CertificateFreeze",
    "V1CertificateLedger",
    "V1ChartReuseAction",
    "V1ChartReuseDecision",
    "V1ChartReuseRequest",
    "V1ChartThresholds",
    "V1CorrectionKind",
    "V1EvidencePacket",
    "V1FDImplementation",
    "V1FiniteActionClassProof",
    "V1FDLineageVerificationReceipt",
    "V1FrequentDirectionsLineage",
    "V1InterventionalCorrectionLineage",
    "V1InterventionalCorrectionVerificationReceipt",
    "V1MetricGeometryKind",
    "V1MixedDifferenceEnvelope",
    "V1MixedDifferenceProofBasis",
    "V1OwnerRegistration",
    "V1ProposalChart",
    "V1ProposalChartBuildContract",
    "V1ResidualDiagonalStatus",
    "V1StreamRole",
    "adjudicate_chart_reuse",
    "advance_certificate_ledger",
    "assert_packet_matches_certificate_freeze",
    "build_v1_proposal_chart",
    "replace_certificate_freeze",
]
