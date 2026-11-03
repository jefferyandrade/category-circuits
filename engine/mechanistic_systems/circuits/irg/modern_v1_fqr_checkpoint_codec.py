"""Deterministic bounded-state checkpoint codec for the canonical F/Q/R seam.

The live terminal F/Q/R receipt owns every directional-secant record.  That is
useful while issuing the receipt, but it is deliberately *not* a production
checkpoint representation.  This module instead stores:

* the reduced dual-chart receipt;
* both pre-frozen residual-branch ladders;
* exact integer sufficient statistics for an in-flight C-stream acquisition;
* the bounded sequential receipt and deterministic-uncertainty receipts; and
* compact, independently replayed projections of any completed F/Q/R receipt
  and proposal-queue authorization.

The projections preserve the original content hashes and all causal
coordinates, but are different exact Python types from the live receipt and
authorization.  Restoring this codec therefore never recreates proposal,
oracle, commit, or terminal authority.  A caller must freshly verify the live
evidence graph before crossing the proposal-queue boundary.

The wire representation is strict JSON with tagged little-endian binary64
scalars/arrays, framed by a little-endian length and SHA-256 digest.  Pickle is
not used.
"""

from __future__ import annotations

import base64
from dataclasses import InitVar, dataclass, field, fields, is_dataclass
from decimal import Decimal, localcontext
from enum import Enum
import hashlib
import json
import math
from numbers import Integral
import struct
import threading
from typing import Any, Final, NoReturn, cast
import weakref

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_directional_remainder import (
    DIRECTIONAL_SECANT_DOSES,
    V1DirectionalRemainderInterval,
    V1DirectionalSecantDesign,
)
from .modern_v1_dual_chart_geometry import (
    V1DualChartGeometry,
    V1DualChartMismatchReceipt,
)
from .modern_v1_fqr_candidate_bridge import (
    FQR_CANDIDATE_QUEUE_AUTHORIZATION_SCHEMA,
    FQR_CANDIDATE_QUEUE_AUTHORITY_SCOPE,
    FQR_QUEUE_ORDER_POLICY,
    FQR_SINGLE_LADDER_SCOPE,
    V1FQRCandidateQueueAuthorization,
    V1FQRCandidateQueueEntry,
)
from .modern_v1_fqr_branch_family import (
    V1FQRKKTPreFrozenBranchFamily,
    V1FQRPreCDominanceCertificate,
)
from .modern_v1_fqr_branch_acquisition import (
    V1FQRBranchFamilyAccumulator,
    V1FQRBranchFamilyAccumulatorSnapshot,
)
from .modern_v1_fqr_branch_design import V1FQRBranchFamilyDesign
from .modern_v1_fqr_adaptive_rank import (
    V1FQRAdaptiveRankNoCRefusal,
    V1FQRAdaptiveRankPolicy,
    V1FQRAdaptiveRankRefusalCode,
    V1FQRAdaptiveRankReplayLedger,
    V1FQRAdaptiveRankReplayRecord,
    V1FQRAdaptiveRankStabilityCertificate,
)
from .modern_v1_fqr_certification import (
    FQR_CERTIFICATION_RECEIPT_SCHEMA,
    FQR_PROPOSAL_AUTHORITY,
    V1DecisionStabilityCertificate,
    V1FQEndpointBound,
    V1FQRCertificationDecision,
    V1FQRCertificationDesign,
    V1FQRCertificationReceipt,
    verify_v1_fqr_certification_receipt,
)
from .modern_v1_fqr_compact_branch_family import (
    COMPACT_FQR_BRANCH_FAMILY_MAX_SOURCE_ENDPOINTS,
    V1CompactFQRBranchEvidenceBinding,
    V1CompactFQRBranchNegativeEvidenceBinding,
    V1CompactFQRBranchFamilyAuthorization,
    V1CompactFQRBranchFamilyQueueEntry,
    V1CompactFQRBranchFamilySourceAlias,
)
from .modern_v1_fqr_compact_certification import (
    V1CompactFQRCertificationReceipt,
    V1CompactFQRDecision,
    V1CompactFQREndpointCertificate,
    V1CompactFQRPreCResourceRegistration,
    V1CompactFQRProposalAuthorization,
)
from .modern_v1_fqr_runtime_acquisition import V1FQRAuthorizedAcquisition
from .modern_v1_fqr_sequential_acquisition import (
    FQR_DETERMINISTIC_UNCERTAINTY_PROPOSAL_V3_SCHEMA,
    FQR_DETERMINISTIC_UNCERTAINTY_SCHEMA,
    FQR_DETERMINISTIC_UNCERTAINTY_V2_SCHEMA,
    _PROPOSAL_DETERMINISTIC_ISSUANCE_TOKEN,
    V1FQRDecisionReadinessReceipt,
    V1FQRDeterministicUncertaintyReceipt,
    V1FQRSequentialAccumulator,
    V1FQRSequentialAccumulatorSnapshot,
    V1FQRSequentialAcquisitionDesign,
    V1FQRSequentialAcquisitionReceipt,
    V1FQRSequentialCellSummary,
    V1FQRSequentialEndpointSummary,
)
from .modern_v1_signed_work_label_native_six_role_c import (
    V1LabelNativeSixRoleAccumulatorSnapshot,
    V1LabelNativeSixRoleCAccumulator,
    V1LabelNativeSixRoleCDesign,
    V1LabelNativeSixRoleCRefusal,
)
from .modern_v1_ladder_statistics import (
    V1FrozenCandidateLadder,
    V1FrozenLadderAction,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch

FQR_CHECKPOINT_SCHEMA: Final = "partition_free_v1_fqr_checkpoint_bundle_v1"
FQR_COMPACT_CHECKPOINT_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_checkpoint_bundle_v2"
)
FQR_INFLIGHT_C_CHECKPOINT_SCHEMA: Final = (
    "partition_free_v1_fqr_inflight_shared_c_checkpoint_bundle_v3"
)
FQR_LABEL_NATIVE_SIX_ROLE_INFLIGHT_C_CHECKPOINT_SCHEMA: Final = (
    "partition_free_v1_label_native_six_role_inflight_c_checkpoint_bundle_v2"
)
FQR_ADAPTIVE_RANK_CHECKPOINT_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_checkpoint_bundle_v2"
)
FQR_ADAPTIVE_RANK_REPLAY_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_replay_checkpoint_projection_v2"
)
FQR_CHECKPOINT_BRANCH_SCHEMA: Final = "partition_free_v1_fqr_checkpoint_branch_v1"
FQR_CHECKPOINT_RECEIPT_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_fqr_receipt_checkpoint_projection_v1"
)
FQR_CHECKPOINT_QUEUE_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_fqr_queue_checkpoint_projection_v1"
)
FQR_PRE_C_DOMINANCE_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_fqr_pre_c_dominance_checkpoint_projection_v1"
)
FQR_PRE_C_FAMILY_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_fqr_pre_c_family_checkpoint_projection_v1"
)
FQR_COMPACT_BRANCH_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_branch_checkpoint_projection_v1"
)
FQR_COMPACT_NEGATIVE_BRANCH_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_negative_branch_checkpoint_projection_v2"
)

_DETERMINISTIC_PROPOSAL_V3_FIELDS: Final = frozenset(
    {
        "proposal_freeze_certificate_hash",
        "proposal_freeze_graph_hash",
        "proposal_candidate_family_hash",
        "device_shell_packed_branch_binding_hash",
        "device_shell_source_branch_binding_hash",
        "proposal_branch",
        "candidate_family_frozen_before_independent_C",
        "D_FD_KKT_composition_uncertainty_efficiency_only",
        "independent_C_FQR_required",
    }
)
FQR_COMPACT_ALIAS_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_alias_checkpoint_projection_v1"
)
FQR_COMPACT_ENTRY_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_entry_checkpoint_projection_v1"
)
FQR_COMPACT_FAMILY_QUEUE_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_family_queue_checkpoint_projection_v1"
)
FQR_COMPACT_FAMILY_QUEUE_PROJECTION_V2_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_family_queue_checkpoint_projection_v2"
)
FQR_AUTHORIZED_ACQUISITION_PROJECTION_SCHEMA: Final = (
    "partition_free_v1_fqr_authorized_acquisition_checkpoint_projection_v1"
)
FQR_AUTHORIZED_ACQUISITION_PROJECTION_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_authorized_acquisition_checkpoint_projection_v2"
)
FQR_INFLIGHT_C_CHECKPOINT_AUTHORITY_SCOPE: Final = (
    "bounded_inflight_shared_c_prefix_replay_only_exact_pre_c_design_and_"
    "snapshot_hash_required_no_c_k_o_commit_terminal_or_pareto_authority_v1"
)
FQR_LABEL_NATIVE_SIX_ROLE_INFLIGHT_C_CHECKPOINT_AUTHORITY_SCOPE: Final = (
    "bounded_label_native_six_role_C_prefix_replay_only_exact_design_and_"
    "snapshot_hash_required_no_c_k_o_commit_terminal_or_pareto_authority_v1"
)
FQR_CHECKPOINT_AUTHORITY_SCOPE: Final = (
    "bounded_resume_and_inspection_only_fresh_live_fqr_verification_required_"
    "no_proposal_queue_oracle_commit_or_terminal_authority_v1"
)
CANONICAL_FQR_BRANCH_IDS: Final = ("spectral", "diagonal_l1")
FQR_COMPACT_REPLAY_REQUIREMENT: Final = (
    "decoded_checkpoint_is_inert_rebuild_exact_live_pre_c_family_compact_"
    "evidence_cross_branch_authorization_and_fqr_acquisition_then_match_all_"
    "source_hashes_before_queue_ready_resume_v1"
)

_MAGIC: Final = b"MV1FQR1"
_FRAME_PREFIX_BYTES: Final = len(_MAGIC) + 8 + 32
_MAX_PAYLOAD_BYTES: Final = 512 << 20
_MAX_ARRAY_BYTES: Final = 256 << 20
_MAX_BYTE_STRING_BYTES: Final = 256 << 20
_MAX_STRING_BYTES: Final = 16 << 20
_MAX_SEQUENCE_ITEMS: Final = 2_000_000
_MAX_NESTING_DEPTH: Final = 64
_MAX_BRANCHES: Final = 16
_MAX_ENDPOINTS_PER_BRANCH: Final = 4096
_LE_F64: Final = np.dtype("<f8")

_RECEIPT_PROJECTION_PREFIX: Final = (
    "partition-free-v1-fqr-checkpoint-receipt-projection-"
)
_QUEUE_PROJECTION_PREFIX: Final = "partition-free-v1-fqr-checkpoint-queue-projection-"
_BRANCH_PREFIX: Final = "partition-free-v1-fqr-checkpoint-branch-"
_BUNDLE_PREFIX: Final = "partition-free-v1-fqr-checkpoint-bundle-"
_DOMINANCE_PROJECTION_PREFIX: Final = (
    "partition-free-v1-fqr-checkpoint-dominance-projection-"
)
_FAMILY_PROJECTION_PREFIX: Final = (
    "partition-free-v1-fqr-checkpoint-family-projection-"
)
_COMPACT_BRANCH_PROJECTION_PREFIX: Final = (
    "partition-free-v1-fqr-checkpoint-compact-branch-projection-"
)
_COMPACT_NEGATIVE_BRANCH_PROJECTION_PREFIX: Final = (
    "partition-free-v1-fqr-checkpoint-compact-negative-branch-projection-"
)
_COMPACT_ALIAS_PROJECTION_PREFIX: Final = (
    "partition-free-v1-fqr-checkpoint-compact-alias-projection-"
)
_COMPACT_ENTRY_PROJECTION_PREFIX: Final = (
    "partition-free-v1-fqr-checkpoint-compact-entry-projection-"
)
_COMPACT_FAMILY_QUEUE_PROJECTION_PREFIX: Final = (
    "partition-free-v1-fqr-checkpoint-compact-family-queue-projection-"
)
_AUTHORIZED_ACQUISITION_PROJECTION_PREFIX: Final = (
    "partition-free-v1-fqr-checkpoint-authorized-acquisition-projection-"
)
_INFLIGHT_C_PRE_C_GRAPH_PREFIX: Final = (
    "partition-free-v1-fqr-inflight-shared-c-pre-c-graph-"
)
_INFLIGHT_C_BUNDLE_PREFIX: Final = (
    "partition-free-v1-fqr-inflight-shared-c-checkpoint-bundle-"
)
_LABEL_NATIVE_SIX_ROLE_INFLIGHT_C_BUNDLE_PREFIX: Final = (
    "partition-free-v1-label-native-six-role-inflight-c-checkpoint-bundle-"
)
_ADAPTIVE_RANK_REPLAY_PROJECTION_PREFIX: Final = (
    "partition-free-v1-fqr-adaptive-rank-replay-checkpoint-projection-"
)
_ADAPTIVE_RANK_CHECKPOINT_BUNDLE_PREFIX: Final = (
    "partition-free-v1-fqr-adaptive-rank-checkpoint-bundle-"
)
_COMPACT_BUNDLE_PREFIX: Final = (
    "partition-free-v1-fqr-checkpoint-compact-bundle-"
)
_METADATA_PROJECTION_PREFIX: Final = (
    "partition-free-v1-fqr-checkpoint-acquisition-metadata-"
)
_LIVE_FQR_RECEIPT_PREFIX: Final = "partition-free-v1-fqr-receipt-"
_LIVE_QUEUE_AUTHORIZATION_PREFIX: Final = (
    "partition-free-v1-fqr-candidate-queue-authorization-"
)

# These process-local tokens are deliberately absent from the wire format.
# The capture token closes direct construction of a checkpoint root; the
# decode token additionally identifies a root which crossed the strict framed
# decoder and may therefore be used by the explicit trusted-C restore seam.
_LABEL_NATIVE_SIX_ROLE_CAPTURE_ISSUANCE_TOKEN: Final = object()
_LABEL_NATIVE_SIX_ROLE_DECODE_ISSUANCE_TOKEN: Final = object()


class V1FQRCheckpointRefusalCode(str, Enum):
    """Stable fail-closed reasons for checkpoint encode/decode refusal."""

    INVALID_TYPE = "invalid_type"
    UNKNOWN_SCHEMA = "unknown_schema"
    HASH_MISMATCH = "hash_mismatch"
    STALE_COORDINATES = "stale_coordinates"
    POST_SELECTION = "post_selection"
    MISSING_BRANCH = "missing_branch"
    AUTHORITY_VIOLATION = "authority_violation"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    FQ_ONLY_LEGACY = "fq_only_legacy"
    RAW_RECORD_STATE_FORBIDDEN = "raw_record_state_forbidden"
    SNAPSHOT_HASH_REQUIRED = "snapshot_hash_required"
    MALFORMED = "malformed"


class V1FQRCheckpointRefusal(ValueError):
    """Typed refusal that never grants resume or action authority."""

    def __init__(self, code: V1FQRCheckpointRefusalCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1FQRCheckpointRefusalCode, message: str) -> NoReturn:
    raise V1FQRCheckpointRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1FQRCheckpointRefusalCode.STALE_COORDINATES,
            f"{name} must be one exact nonempty string",
        )
    return value


def _branch_id(value: object, *, name: str) -> str:
    branch = _identity(value, name=name)
    if branch not in CANONICAL_FQR_BRANCH_IDS:
        _refuse(
            V1FQRCheckpointRefusalCode.STALE_COORDINATES,
            f"{name} is not a canonical KKT branch",
        )
    return branch


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1FQRCheckpointRefusalCode.MALFORMED,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1FQRCheckpointRefusalCode.MALFORMED,
            f"{name} must be nonnegative",
        )
    return result


def _exact_positive_int(value: object, *, name: str) -> int:
    result = _exact_nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1FQRCheckpointRefusalCode.MALFORMED,
            f"{name} must be positive",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        _refuse(
            V1FQRCheckpointRefusalCode.MALFORMED,
            f"{name} must be one finite nonnegative binary64 value",
        )
    return value


def _finite_positive(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result <= 0.0:
        _refuse(
            V1FQRCheckpointRefusalCode.MALFORMED,
            f"{name} must be positive",
        )
    return result


def _canonical_retained_branch_ids(values: object) -> tuple[str, ...]:
    if type(values) is not tuple:
        _refuse(
            V1FQRCheckpointRefusalCode.INVALID_TYPE,
            "retained branch IDs must be one exact tuple",
        )
    observed = tuple(
        _branch_id(value, name="retained KKT branch") for value in values
    )
    expected = tuple(value for value in CANONICAL_FQR_BRANCH_IDS if value in observed)
    if len(observed) not in {1, 2} or len(set(observed)) != len(observed) or observed != expected:
        _refuse(
            V1FQRCheckpointRefusalCode.MISSING_BRANCH,
            "retained KKT branches must be the exact nonempty canonical-order subset",
        )
    return observed


def _direct_mask_hash(mask: HardMask) -> str:
    if type(mask) is not HardMask:
        _refuse(
            V1FQRCheckpointRefusalCode.INVALID_TYPE,
            "checkpoint masks must be exact HardMask values",
        )
    return content_hash(
        {
            "route_axis_hash": mask.route_axis_hash,
            "route_count": mask.route_count,
            "packed_bits": mask.packed_bits,
        },
        prefix="irg-mask-",
    )


def _response_lhs_decimal(*, eta: float, q_upper: float, r_upper: float) -> Decimal:
    with localcontext() as context:
        context.prec = 96
        exact = Decimal.from_float(eta) * Decimal.from_float(q_upper).sqrt()
        exact += Decimal.from_float(r_upper)
    return exact


def _outward_decimal(value: Decimal) -> float:
    result = float(value)
    if Decimal.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


@dataclass(frozen=True, slots=True)
class V1FQRPreCDominanceCheckpoint:
    """Inert bounded projection of one lawful pre-C branch omission proof."""

    source_certificate_hash: str
    source_complete_family_hash: str
    problem_hash: str
    policy_hash: str
    result_hash: str
    retained_branch_id: str
    omitted_branch_id: str
    retained_binding_hash: str
    omitted_binding_hash: str
    omitted_ladder_hash: str
    endpoint_coverage: tuple[tuple[int, int, str, float, float], ...]
    dominance_method: str
    issued_before_c_outcomes: bool = True
    fqr_outcomes_seen: bool = False
    hard_global_optimality_claim: bool = False
    live_certificate_reconstruction_authority: bool = False
    proposal_queue_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQR_CHECKPOINT_AUTHORITY_SCOPE
    schema: str = FQR_PRE_C_DOMINANCE_PROJECTION_SCHEMA
    _projection_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "source_certificate_hash",
            "source_complete_family_hash",
            "problem_hash",
            "policy_hash",
            "result_hash",
            "retained_binding_hash",
            "omitted_binding_hash",
            "omitted_ladder_hash",
            "dominance_method",
        ):
            _identity(getattr(self, name), name=name)
        retained = _branch_id(
            self.retained_branch_id, name="dominance retained branch"
        )
        omitted = _branch_id(
            self.omitted_branch_id, name="dominance omitted branch"
        )
        if retained == omitted or {retained, omitted} != set(CANONICAL_FQR_BRANCH_IDS):
            _refuse(
                V1FQRCheckpointRefusalCode.MISSING_BRANCH,
                "dominance projection must retain one and omit the other canonical branch",
            )
        coverage: list[tuple[int, int, str, float, float]] = []
        for raw in self.endpoint_coverage:
            if type(raw) is not tuple or len(raw) != 5:
                _refuse(
                    V1FQRCheckpointRefusalCode.MALFORMED,
                    "dominance coverage rows must be exact five-tuples",
                )
            omitted_endpoint = _exact_positive_int(
                raw[0], name="omitted endpoint index"
            )
            retained_endpoint = _exact_positive_int(
                raw[1], name="retained endpoint index"
            )
            action_hash = _identity(raw[2], name="dominated packed action hash")
            omitted_resource = _finite_nonnegative(
                raw[3], name="omitted registered resource"
            )
            retained_resource = _finite_nonnegative(
                raw[4], name="retained registered resource"
            )
            if retained_resource < omitted_resource:
                _refuse(
                    V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                    "dominance projection retained a smaller-resource duplicate",
                )
            coverage.append(
                (
                    omitted_endpoint,
                    retained_endpoint,
                    action_hash,
                    omitted_resource,
                    retained_resource,
                )
            )
        normalized = tuple(coverage)
        if (
            not normalized
            or len(normalized) > _MAX_ENDPOINTS_PER_BRANCH
            or tuple(row[0] for row in normalized)
            != tuple(range(1, len(normalized) + 1))
            or len({row[2] for row in normalized}) != len(normalized)
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
                "dominance projection is empty, incomplete, duplicate, or too large",
            )
        if (
            self.issued_before_c_outcomes is not True
            or self.fqr_outcomes_seen is not False
            or self.hard_global_optimality_claim is not False
            or any(
                getattr(self, name) is not False
                for name in (
                    "live_certificate_reconstruction_authority",
                    "proposal_queue_authority",
                    "calibration_authority",
                    "bce_oracle_authority",
                    "mask_commit_authority",
                    "terminal_authority",
                )
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "dominance checkpoint crossed its inert pre-C boundary",
            )
        if (
            self.authority_scope != FQR_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema != FQR_PRE_C_DOMINANCE_PROJECTION_SCHEMA
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "dominance checkpoint schema or authority is invalid",
            )
        object.__setattr__(self, "retained_branch_id", retained)
        object.__setattr__(self, "omitted_branch_id", omitted)
        object.__setattr__(self, "endpoint_coverage", normalized)
        self.assert_integrity()

    @classmethod
    def from_live(
        cls, certificate: V1FQRPreCDominanceCertificate
    ) -> "V1FQRPreCDominanceCheckpoint":
        if type(certificate) is not V1FQRPreCDominanceCertificate:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "dominance projection requires the exact typed pre-C certificate",
            )
        try:
            source_hash = certificate.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"pre-C dominance certificate failed source validation: {error}",
            ) from error
        return cls(
            source_certificate_hash=source_hash,
            source_complete_family_hash=certificate.source_complete_family_hash,
            problem_hash=certificate.problem_hash,
            policy_hash=certificate.policy_hash,
            result_hash=certificate.result_hash,
            retained_branch_id=certificate.retained_branch.value,
            omitted_branch_id=certificate.omitted_branch.value,
            retained_binding_hash=certificate.retained_binding_hash,
            omitted_binding_hash=certificate.omitted_binding_hash,
            omitted_ladder_hash=certificate.omitted_ladder_hash,
            endpoint_coverage=certificate.endpoint_coverage,
            dominance_method=certificate.dominance_method,
        )

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "source_certificate_hash",
                "source_complete_family_hash",
                "problem_hash",
                "policy_hash",
                "result_hash",
                "retained_branch_id",
                "omitted_branch_id",
                "retained_binding_hash",
                "omitted_binding_hash",
                "omitted_ladder_hash",
                "endpoint_coverage",
                "dominance_method",
                "issued_before_c_outcomes",
                "fqr_outcomes_seen",
                "hard_global_optimality_claim",
                "live_certificate_reconstruction_authority",
                "proposal_queue_authority",
                "calibration_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
                "authority_scope",
            )
        }

    @property
    def projection_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_DOMINANCE_PROJECTION_PREFIX)
        if self._projection_hash is not None and self._projection_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "dominance checkpoint projection was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "dominance checkpoint projection hash does not match",
            )
        object.__setattr__(self, "_projection_hash", observed)
        return observed

    def assert_matches_live(self, certificate: V1FQRPreCDominanceCertificate) -> str:
        replayed = type(self).from_live(certificate)
        if replayed.projection_hash != self.projection_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "fresh live dominance certificate differs from its checkpoint projection",
            )
        return replayed.source_certificate_hash


@dataclass(frozen=True, slots=True)
class V1FQRPreFrozenFamilyCheckpoint:
    """Hash-complete, route-array-free projection of the pre-C KKT family."""

    source_family_hash: str
    source_complete_family_hash: str
    problem_hash: str
    policy_hash: str
    result_hash: str
    relaxed_selected_branch_id: str | None
    retained_branch_ids: tuple[str, ...]
    retained_binding_hashes: tuple[tuple[str, str], ...]
    route_axis_hash: str
    route_count: int
    current_mask_hash: str
    chart_identity_hash: str
    proposal_generation: int
    discovery_prefix_hash: str
    discovery_evidence_hash: str
    dominance_checkpoint: V1FQRPreCDominanceCheckpoint | None = field(
        default=None, repr=False, compare=False
    )
    family_frozen_before_c: bool = True
    c_stream_outcomes_seen: bool = False
    live_family_reconstruction_authority: bool = False
    proposal_queue_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQR_CHECKPOINT_AUTHORITY_SCOPE
    schema: str = FQR_PRE_C_FAMILY_PROJECTION_SCHEMA
    _projection_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "source_family_hash",
            "source_complete_family_hash",
            "problem_hash",
            "policy_hash",
            "result_hash",
            "route_axis_hash",
            "current_mask_hash",
            "chart_identity_hash",
            "discovery_prefix_hash",
            "discovery_evidence_hash",
        ):
            _identity(getattr(self, name), name=name)
        relaxed = self.relaxed_selected_branch_id
        if relaxed is not None:
            relaxed = _branch_id(relaxed, name="relaxed selected branch")
        retained = _canonical_retained_branch_ids(self.retained_branch_ids)
        rows = tuple(self.retained_binding_hashes)
        if (
            len(rows) != len(retained)
            or tuple(row[0] for row in rows) != retained
            or any(type(row) is not tuple or len(row) != 2 for row in rows)
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.MISSING_BRANCH,
                "retained binding hashes do not follow the exact branch axis",
            )
        normalized_rows = tuple(
            (
                _branch_id(branch, name="binding branch"),
                _identity(value, name="retained binding hash"),
            )
            for branch, value in rows
        )
        dominance = self.dominance_checkpoint
        if len(retained) == 2:
            if dominance is not None or self.source_complete_family_hash != self.source_family_hash:
                _refuse(
                    V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                    "complete pre-C family cannot carry an omission projection",
                )
        else:
            if type(dominance) is not V1FQRPreCDominanceCheckpoint:
                _refuse(
                    V1FQRCheckpointRefusalCode.MISSING_BRANCH,
                    "one-branch checkpoint requires the exact pre-C dominance projection",
                )
            dominance.assert_integrity()
            if (
                dominance.retained_branch_id != retained[0]
                or dominance.retained_binding_hash != normalized_rows[0][1]
                or dominance.source_complete_family_hash
                != self.source_complete_family_hash
                or dominance.problem_hash != self.problem_hash
                or dominance.policy_hash != self.policy_hash
                or dominance.result_hash != self.result_hash
            ):
                _refuse(
                    V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                    "dominance projection does not bind the retained family",
                )
        object.__setattr__(
            self, "route_count", _exact_positive_int(self.route_count, name="route count")
        )
        object.__setattr__(
            self,
            "proposal_generation",
            _exact_nonnegative_int(
                self.proposal_generation, name="proposal generation"
            ),
        )
        if (
            self.family_frozen_before_c is not True
            or self.c_stream_outcomes_seen is not False
            or any(
                getattr(self, name) is not False
                for name in (
                    "live_family_reconstruction_authority",
                    "proposal_queue_authority",
                    "calibration_authority",
                    "bce_oracle_authority",
                    "mask_commit_authority",
                    "terminal_authority",
                )
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "pre-C family checkpoint crossed its inert boundary",
            )
        if (
            self.authority_scope != FQR_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema != FQR_PRE_C_FAMILY_PROJECTION_SCHEMA
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "pre-C family checkpoint schema or authority is invalid",
            )
        object.__setattr__(self, "relaxed_selected_branch_id", relaxed)
        object.__setattr__(self, "retained_branch_ids", retained)
        object.__setattr__(self, "retained_binding_hashes", normalized_rows)
        self.assert_integrity()

    @classmethod
    def from_live(
        cls, family: V1FQRKKTPreFrozenBranchFamily
    ) -> "V1FQRPreFrozenFamilyCheckpoint":
        if type(family) is not V1FQRKKTPreFrozenBranchFamily:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "family projection requires the exact pre-frozen KKT family",
            )
        try:
            family_hash = family.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"pre-C family failed source validation: {error}",
            ) from error
        retained = family.retained_bindings
        dominance = family.dominance_certificate
        dominance_checkpoint = (
            None
            if dominance is None
            else V1FQRPreCDominanceCheckpoint.from_live(dominance)
        )
        return cls(
            source_family_hash=family_hash,
            source_complete_family_hash=(
                family_hash
                if dominance is None
                else dominance.source_complete_family_hash
            ),
            problem_hash=family.problem_hash,
            policy_hash=family.policy_hash,
            result_hash=family.result_hash,
            relaxed_selected_branch_id=(
                None
                if family.relaxed_selected_branch is None
                else family.relaxed_selected_branch.value
            ),
            retained_branch_ids=tuple(value.branch.value for value in retained),
            retained_binding_hashes=tuple(
                (value.branch.value, value.binding_hash) for value in retained
            ),
            route_axis_hash=family.route_axis_hash,
            route_count=family.route_count,
            current_mask_hash=family.current_mask_hash,
            chart_identity_hash=family.chart_identity_hash,
            proposal_generation=family.proposal_generation,
            discovery_prefix_hash=family.discovery_prefix_hash,
            discovery_evidence_hash=family.discovery_evidence_hash,
            dominance_checkpoint=dominance_checkpoint,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_family_hash": self.source_family_hash,
            "source_complete_family_hash": self.source_complete_family_hash,
            "problem_hash": self.problem_hash,
            "policy_hash": self.policy_hash,
            "result_hash": self.result_hash,
            "relaxed_selected_branch_id": self.relaxed_selected_branch_id,
            "retained_branch_ids": self.retained_branch_ids,
            "retained_binding_hashes": self.retained_binding_hashes,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "current_mask_hash": self.current_mask_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "proposal_generation": self.proposal_generation,
            "discovery_prefix_hash": self.discovery_prefix_hash,
            "discovery_evidence_hash": self.discovery_evidence_hash,
            "dominance_projection_hash": (
                None
                if self.dominance_checkpoint is None
                else self.dominance_checkpoint.projection_hash
            ),
            "family_frozen_before_c": self.family_frozen_before_c,
            "c_stream_outcomes_seen": self.c_stream_outcomes_seen,
            "live_family_reconstruction_authority": (
                self.live_family_reconstruction_authority
            ),
            "proposal_queue_authority": self.proposal_queue_authority,
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def projection_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if self.dominance_checkpoint is not None:
            self.dominance_checkpoint.assert_integrity()
        observed = content_hash(self._payload(), prefix=_FAMILY_PROJECTION_PREFIX)
        if self._projection_hash is not None and self._projection_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "pre-C family checkpoint projection was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "pre-C family checkpoint projection hash does not match",
            )
        object.__setattr__(self, "_projection_hash", observed)
        return observed

    def assert_matches_live(self, family: V1FQRKKTPreFrozenBranchFamily) -> str:
        replayed = type(self).from_live(family)
        if replayed.projection_hash != self.projection_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "fresh live pre-C family differs from its checkpoint projection",
            )
        return replayed.source_family_hash


@dataclass(frozen=True, slots=True)
class V1CompactFQRBranchCheckpoint:
    """Bounded compact certification graph for one retained KKT branch.

    The live single-endpoint proposal authorization is projected to hashes,
    scalars, and its packed mask.  Decoding therefore cannot recreate branch
    evidence or proposal authority without a fresh exact live-graph replay.
    """

    source_branch_evidence_hash: str
    source_proposal_authorization_hash: str
    prefrozen_family_hash: str
    branch_id: str
    branch_binding_hash: str
    certification_receipt: V1CompactFQRCertificationReceipt = field(
        repr=False, compare=False
    )
    decision: V1CompactFQRDecision = field(repr=False, compare=False)
    resource_registration: V1CompactFQRPreCResourceRegistration = field(
        repr=False, compare=False
    )
    selected_packed_action: V1PackedRemovalAction = field(repr=False, compare=False)
    selected_endpoint_index: int
    selected_action_hash: str
    selected_resulting_mask_hash: str
    selected_resource: float
    resource_units: str
    resource_source_hash: str
    compact_only: bool = True
    raw_records_retained: bool = False
    live_branch_evidence_reconstruction_authority: bool = False
    proposal_queue_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    paired_bce_oracle_still_required: bool = True
    replay_requirement: str = FQR_COMPACT_REPLAY_REQUIREMENT
    authority_scope: str = FQR_CHECKPOINT_AUTHORITY_SCOPE
    schema: str = FQR_COMPACT_BRANCH_PROJECTION_SCHEMA
    _projection_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "source_branch_evidence_hash",
            "source_proposal_authorization_hash",
            "prefrozen_family_hash",
            "branch_binding_hash",
            "selected_action_hash",
            "selected_resulting_mask_hash",
            "resource_units",
            "resource_source_hash",
        ):
            _identity(getattr(self, name), name=name)
        branch = _branch_id(self.branch_id, name="compact evidence branch")
        if type(self.certification_receipt) is not V1CompactFQRCertificationReceipt:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "compact branch checkpoint requires the exact compact receipt",
            )
        if type(self.decision) is not V1CompactFQRDecision:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "compact branch checkpoint requires the exact compact decision",
            )
        if type(self.resource_registration) is not V1CompactFQRPreCResourceRegistration:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "compact branch checkpoint requires the exact pre-C resource registration",
            )
        if type(self.selected_packed_action) is not V1PackedRemovalAction:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "compact branch checkpoint requires one exact packed action",
            )
        receipt_hash = self.certification_receipt.assert_integrity()
        decision_hash = self.decision.assert_integrity()
        resource_hash = self.resource_registration.assert_integrity()
        endpoint = _exact_positive_int(
            self.selected_endpoint_index, name="selected endpoint index"
        )
        resource = _finite_positive(
            self.selected_resource, name="selected endpoint resource"
        )
        child = self.selected_packed_action.resulting_mask
        child_hash = _direct_mask_hash(child)
        if (
            child_hash != self.selected_resulting_mask_hash
            or endpoint > len(self.certification_receipt.endpoint_certificates)
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "compact branch packed action or selected endpoint is invalid",
            )
        certificate = self.certification_receipt.endpoint_certificates[endpoint - 1]
        resource_row = self.resource_registration.resource_by_endpoint[endpoint - 1]
        if (
            self.decision.certification_receipt_hash != receipt_hash
            or self.decision.resource_registration_hash != resource_hash
            or self.decision.selected_endpoint_index != endpoint
            or self.decision.selected_action_hash != self.selected_action_hash
            or certificate.endpoint_index != endpoint
            or certificate.action_hash != self.selected_action_hash
            or certificate.upper_certified is not True
            or resource_row
            != (endpoint, self.selected_action_hash, resource)
            or self.certification_receipt.resource_registration_hash != resource_hash
            or self.certification_receipt.current_mask_hash
            != self.resource_registration.current_mask_hash
            or self.certification_receipt.chart_identity_hash
            != self.resource_registration.chart_identity_hash
            or self.certification_receipt.proposal_generation
            != self.resource_registration.proposal_generation
            or self.certification_receipt.route_count
            != self.resource_registration.route_count
            or self.certification_receipt.route_axis_hash
            != self.resource_registration.route_axis_hash
            or self.certification_receipt.endpoint_action_hashes
            != self.resource_registration.endpoint_action_hashes
            or self.certification_receipt.sampling_decision
            != "sampling_stable_endpoint"
            or self.decision.status != "stable_endpoint"
            or decision_hash != self.decision.decision_hash
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "compact branch receipt, decision, resource, and selected action disagree",
            )
        if (
            self.compact_only is not True
            or self.raw_records_retained is not False
            or self.paired_bce_oracle_still_required is not True
            or any(
                getattr(self, name) is not False
                for name in (
                    "live_branch_evidence_reconstruction_authority",
                    "proposal_queue_authority",
                    "calibration_authority",
                    "bce_oracle_authority",
                    "mask_commit_authority",
                    "terminal_authority",
                )
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "compact branch checkpoint crossed its inert authority boundary",
            )
        if (
            self.replay_requirement != FQR_COMPACT_REPLAY_REQUIREMENT
            or self.authority_scope != FQR_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema != FQR_COMPACT_BRANCH_PROJECTION_SCHEMA
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "compact branch checkpoint schema or replay contract is invalid",
            )
        object.__setattr__(self, "branch_id", branch)
        object.__setattr__(self, "selected_endpoint_index", endpoint)
        object.__setattr__(self, "selected_resource", resource)
        self.assert_integrity()

    @classmethod
    def from_live(
        cls, evidence: V1CompactFQRBranchEvidenceBinding
    ) -> "V1CompactFQRBranchCheckpoint":
        if type(evidence) is not V1CompactFQRBranchEvidenceBinding:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "compact branch projection requires exact compact branch evidence",
            )
        try:
            evidence_hash = evidence.assert_integrity()
            authorization = evidence.proposal_authorization
            authorization_hash = authorization.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"compact branch evidence failed source validation: {error}",
            ) from error
        return cls(
            source_branch_evidence_hash=evidence_hash,
            source_proposal_authorization_hash=authorization_hash,
            prefrozen_family_hash=evidence.prefrozen_family_hash,
            branch_id=evidence.branch.value,
            branch_binding_hash=evidence.branch_binding.binding_hash,
            certification_receipt=evidence.certification_receipt,
            decision=evidence.decision,
            resource_registration=evidence.resource_registration,
            selected_packed_action=authorization.selected_packed_action,
            selected_endpoint_index=authorization.selected_endpoint_index,
            selected_action_hash=authorization.selected_action_hash,
            selected_resulting_mask_hash=authorization.selected_resulting_mask_hash,
            selected_resource=authorization.selected_resource,
            resource_units=authorization.resource_units,
            resource_source_hash=authorization.resource_source_hash,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_branch_evidence_hash": self.source_branch_evidence_hash,
            "source_proposal_authorization_hash": (
                self.source_proposal_authorization_hash
            ),
            "prefrozen_family_hash": self.prefrozen_family_hash,
            "branch_id": self.branch_id,
            "branch_binding_hash": self.branch_binding_hash,
            "certification_receipt_hash": self.certification_receipt.receipt_hash,
            "decision_hash": self.decision.decision_hash,
            "resource_registration_hash": (
                self.resource_registration.registration_hash
            ),
            "selected_packed_action_hash": self.selected_packed_action.action_hash,
            "selected_endpoint_index": self.selected_endpoint_index,
            "selected_action_hash": self.selected_action_hash,
            "selected_resulting_mask_hash": self.selected_resulting_mask_hash,
            "selected_resource": self.selected_resource,
            "resource_units": self.resource_units,
            "resource_source_hash": self.resource_source_hash,
            "compact_only": self.compact_only,
            "raw_records_retained": self.raw_records_retained,
            "live_branch_evidence_reconstruction_authority": (
                self.live_branch_evidence_reconstruction_authority
            ),
            "proposal_queue_authority": self.proposal_queue_authority,
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "paired_bce_oracle_still_required": (
                self.paired_bce_oracle_still_required
            ),
            "replay_requirement": self.replay_requirement,
            "authority_scope": self.authority_scope,
        }

    @property
    def projection_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.certification_receipt.assert_integrity()
        self.decision.assert_integrity()
        self.resource_registration.assert_integrity()
        if (
            _direct_mask_hash(self.selected_packed_action.resulting_mask)
            != self.selected_resulting_mask_hash
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "compact branch packed action was tampered",
            )
        observed = content_hash(
            self._payload(), prefix=_COMPACT_BRANCH_PROJECTION_PREFIX
        )
        if self._projection_hash is not None and self._projection_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "compact branch checkpoint projection was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "compact branch checkpoint projection hash does not match",
            )
        object.__setattr__(self, "_projection_hash", observed)
        return observed

    def assert_matches_live(self, evidence: V1CompactFQRBranchEvidenceBinding) -> str:
        replayed = type(self).from_live(evidence)
        if replayed.projection_hash != self.projection_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "fresh compact branch evidence differs from its checkpoint projection",
            )
        return replayed.source_branch_evidence_hash


@dataclass(frozen=True, slots=True)
class V1CompactFQRNegativeBranchCheckpoint:
    """Inert v2 projection of a complete stable-no-endpoint branch proof."""

    source_branch_negative_evidence_hash: str
    prefrozen_family_hash: str
    branch_id: str
    branch_binding_hash: str
    certification_receipt: V1CompactFQRCertificationReceipt = field(
        repr=False, compare=False
    )
    decision: V1CompactFQRDecision = field(repr=False, compare=False)
    resource_registration: V1CompactFQRPreCResourceRegistration = field(
        repr=False, compare=False
    )
    resource_units: str
    resource_source_hash: str
    compact_only: bool = True
    stable_no_endpoint: bool = True
    negative_evidence_only: bool = True
    raw_records_retained: bool = False
    live_branch_evidence_reconstruction_authority: bool = False
    proposal_queue_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    paired_bce_oracle_still_required: bool = True
    replay_requirement: str = FQR_COMPACT_REPLAY_REQUIREMENT
    authority_scope: str = FQR_CHECKPOINT_AUTHORITY_SCOPE
    schema: str = FQR_COMPACT_NEGATIVE_BRANCH_PROJECTION_SCHEMA
    _projection_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "source_branch_negative_evidence_hash",
            "prefrozen_family_hash",
            "branch_binding_hash",
            "resource_units",
            "resource_source_hash",
        ):
            _identity(getattr(self, name), name=name)
        branch = _branch_id(self.branch_id, name="negative evidence branch")
        if type(self.certification_receipt) is not V1CompactFQRCertificationReceipt:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "negative branch checkpoint requires the exact compact receipt",
            )
        if type(self.decision) is not V1CompactFQRDecision:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "negative branch checkpoint requires the exact compact decision",
            )
        if type(self.resource_registration) is not V1CompactFQRPreCResourceRegistration:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "negative branch checkpoint requires the exact resource registration",
            )
        receipt_hash = self.certification_receipt.assert_integrity()
        decision_hash = self.decision.assert_integrity()
        resource_hash = self.resource_registration.assert_integrity()
        certificates = self.certification_receipt.endpoint_certificates
        if (
            not certificates
            or self.certification_receipt.sampling_decision
            != "sampling_stable_no_endpoint"
            or self.decision.status != "stable_no_endpoint"
            or self.certification_receipt.selected_endpoint_index is not None
            or self.certification_receipt.selected_action_hash is not None
            or self.decision.selected_endpoint_index is not None
            or self.decision.selected_action_hash is not None
            or any(value.upper_certified for value in certificates)
            or any(not value.lower_refuted for value in certificates)
            or self.decision.certification_receipt_hash != receipt_hash
            or self.decision.resource_registration_hash != resource_hash
            or decision_hash != self.decision.decision_hash
            or self.certification_receipt.resource_registration_hash != resource_hash
            or self.certification_receipt.chart_identity_hash
            != self.resource_registration.chart_identity_hash
            or self.certification_receipt.current_mask_hash
            != self.resource_registration.current_mask_hash
            or self.certification_receipt.proposal_generation
            != self.resource_registration.proposal_generation
            or self.certification_receipt.route_count
            != self.resource_registration.route_count
            or self.certification_receipt.route_axis_hash
            != self.resource_registration.route_axis_hash
            or self.certification_receipt.endpoint_action_hashes
            != self.resource_registration.endpoint_action_hashes
            or self.resource_registration.resource_units != self.resource_units
            or self.resource_registration.resource_source_hash
            != self.resource_source_hash
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "negative branch receipt, decision, resource, and exclusion axis disagree",
            )
        if (
            self.compact_only is not True
            or self.stable_no_endpoint is not True
            or self.negative_evidence_only is not True
            or self.raw_records_retained is not False
            or self.paired_bce_oracle_still_required is not True
            or any(
                getattr(self, name) is not False
                for name in (
                    "live_branch_evidence_reconstruction_authority",
                    "proposal_queue_authority",
                    "calibration_authority",
                    "bce_oracle_authority",
                    "mask_commit_authority",
                    "terminal_authority",
                )
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "negative branch checkpoint crossed its inert no-proposal boundary",
            )
        if (
            self.replay_requirement != FQR_COMPACT_REPLAY_REQUIREMENT
            or self.authority_scope != FQR_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema != FQR_COMPACT_NEGATIVE_BRANCH_PROJECTION_SCHEMA
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "negative branch checkpoint schema or replay contract is invalid",
            )
        object.__setattr__(self, "branch_id", branch)
        self.assert_integrity()

    @classmethod
    def from_live(
        cls, evidence: V1CompactFQRBranchNegativeEvidenceBinding
    ) -> "V1CompactFQRNegativeBranchCheckpoint":
        if type(evidence) is not V1CompactFQRBranchNegativeEvidenceBinding:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "negative branch projection requires exact stable-negative evidence",
            )
        try:
            evidence_hash = evidence.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"stable-negative branch evidence failed source validation: {error}",
            ) from error
        return cls(
            source_branch_negative_evidence_hash=evidence_hash,
            prefrozen_family_hash=evidence.prefrozen_family_hash,
            branch_id=evidence.branch.value,
            branch_binding_hash=evidence.branch_binding.binding_hash,
            certification_receipt=evidence.certification_receipt,
            decision=evidence.decision,
            resource_registration=evidence.resource_registration,
            resource_units=evidence.resource_registration.resource_units,
            resource_source_hash=evidence.resource_registration.resource_source_hash,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_branch_negative_evidence_hash": (
                self.source_branch_negative_evidence_hash
            ),
            "prefrozen_family_hash": self.prefrozen_family_hash,
            "branch_id": self.branch_id,
            "branch_binding_hash": self.branch_binding_hash,
            "certification_receipt_hash": self.certification_receipt.receipt_hash,
            "decision_hash": self.decision.decision_hash,
            "resource_registration_hash": (
                self.resource_registration.registration_hash
            ),
            "resource_units": self.resource_units,
            "resource_source_hash": self.resource_source_hash,
            "compact_only": self.compact_only,
            "stable_no_endpoint": self.stable_no_endpoint,
            "negative_evidence_only": self.negative_evidence_only,
            "raw_records_retained": self.raw_records_retained,
            "live_branch_evidence_reconstruction_authority": (
                self.live_branch_evidence_reconstruction_authority
            ),
            "proposal_queue_authority": self.proposal_queue_authority,
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "paired_bce_oracle_still_required": (
                self.paired_bce_oracle_still_required
            ),
            "replay_requirement": self.replay_requirement,
            "authority_scope": self.authority_scope,
        }

    @property
    def projection_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.certification_receipt.assert_integrity()
        self.decision.assert_integrity()
        self.resource_registration.assert_integrity()
        observed = content_hash(
            self._payload(), prefix=_COMPACT_NEGATIVE_BRANCH_PROJECTION_PREFIX
        )
        if self._projection_hash is not None and self._projection_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "stable-negative branch checkpoint projection was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "stable-negative branch projection hash does not match",
            )
        object.__setattr__(self, "_projection_hash", observed)
        return observed

    def assert_matches_live(
        self, evidence: V1CompactFQRBranchNegativeEvidenceBinding
    ) -> str:
        replayed = type(self).from_live(evidence)
        if replayed.projection_hash != self.projection_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "fresh stable-negative evidence differs from checkpoint projection",
            )
        return replayed.source_branch_negative_evidence_hash


@dataclass(frozen=True, slots=True)
class V1CompactFQRSourceAliasCheckpoint:
    """Inert scalar projection of one independently certified source alias."""

    source_alias_hash: str
    branch_id: str
    endpoint_index: int
    frozen_action_hash: str
    packed_action_hash: str
    registered_resource: float
    branch_binding_hash: str
    branch_evidence_hash: str
    resource_registration_hash: str
    certification_receipt_hash: str
    decision_hash: str
    proposal_authorization_hash: str
    endpoint_certificate_hash: str
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
    independently_upper_certified: bool = True
    live_alias_reconstruction_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQR_CHECKPOINT_AUTHORITY_SCOPE
    schema: str = FQR_COMPACT_ALIAS_PROJECTION_SCHEMA
    _projection_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        branch = _branch_id(self.branch_id, name="source alias branch")
        object.__setattr__(
            self,
            "endpoint_index",
            _exact_positive_int(self.endpoint_index, name="source alias endpoint"),
        )
        for name in (
            "source_alias_hash",
            "frozen_action_hash",
            "packed_action_hash",
            "branch_binding_hash",
            "branch_evidence_hash",
            "resource_registration_hash",
            "certification_receipt_hash",
            "decision_hash",
            "proposal_authorization_hash",
            "endpoint_certificate_hash",
        ):
            _identity(getattr(self, name), name=name)
        object.__setattr__(
            self,
            "registered_resource",
            _finite_positive(
                self.registered_resource, name="source alias registered resource"
            ),
        )
        for name in (
            "f_lower",
            "f_upper",
            "q_lower",
            "q_upper",
            "r_half_lower",
            "r_half_upper",
            "r_one_lower",
            "r_one_upper",
            "half_response_lower",
            "half_response_upper",
            "one_response_lower",
            "one_response_upper",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        if any(
            lower > upper
            for lower, upper in (
                (self.f_lower, self.f_upper),
                (self.q_lower, self.q_upper),
                (self.r_half_lower, self.r_half_upper),
                (self.r_one_lower, self.r_one_upper),
                (self.half_response_lower, self.half_response_upper),
                (self.one_response_lower, self.one_response_upper),
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "source alias checkpoint contains an inverted interval",
            )
        if (
            self.independently_upper_certified is not True
            or any(
                getattr(self, name) is not False
                for name in (
                    "live_alias_reconstruction_authority",
                    "calibration_authority",
                    "bce_oracle_authority",
                    "mask_commit_authority",
                    "terminal_authority",
                )
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "source alias checkpoint crossed its inert boundary",
            )
        if (
            self.authority_scope != FQR_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema != FQR_COMPACT_ALIAS_PROJECTION_SCHEMA
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "source alias checkpoint schema is invalid",
            )
        object.__setattr__(self, "branch_id", branch)
        self.assert_integrity()

    @classmethod
    def from_live(
        cls, alias: V1CompactFQRBranchFamilySourceAlias
    ) -> "V1CompactFQRSourceAliasCheckpoint":
        if type(alias) is not V1CompactFQRBranchFamilySourceAlias:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "source alias projection requires the exact compact alias",
            )
        try:
            source_hash = alias.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"compact source alias failed source validation: {error}",
            ) from error
        return cls(
            source_alias_hash=source_hash,
            branch_id=alias.branch.value,
            endpoint_index=alias.endpoint_index,
            frozen_action_hash=alias.frozen_action_hash,
            packed_action_hash=alias.packed_action_hash,
            registered_resource=alias.registered_resource,
            branch_binding_hash=alias.branch_binding_hash,
            branch_evidence_hash=alias.branch_evidence_hash,
            resource_registration_hash=alias.resource_registration_hash,
            certification_receipt_hash=alias.certification_receipt_hash,
            decision_hash=alias.decision_hash,
            proposal_authorization_hash=alias.proposal_authorization_hash,
            endpoint_certificate_hash=alias.endpoint_certificate_hash,
            f_lower=alias.f_lower,
            f_upper=alias.f_upper,
            q_lower=alias.q_lower,
            q_upper=alias.q_upper,
            r_half_lower=alias.r_half_lower,
            r_half_upper=alias.r_half_upper,
            r_one_lower=alias.r_one_lower,
            r_one_upper=alias.r_one_upper,
            half_response_lower=alias.half_response_lower,
            half_response_upper=alias.half_response_upper,
            one_response_lower=alias.one_response_lower,
            one_response_upper=alias.one_response_upper,
        )

    def scalar_intervals(self) -> tuple[float, ...]:
        return (
            self.f_lower,
            self.f_upper,
            self.q_lower,
            self.q_upper,
            self.r_half_lower,
            self.r_half_upper,
            self.r_one_lower,
            self.r_one_upper,
            self.half_response_lower,
            self.half_response_upper,
            self.one_response_lower,
            self.one_response_upper,
        )

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "source_alias_hash",
                "branch_id",
                "endpoint_index",
                "frozen_action_hash",
                "packed_action_hash",
                "registered_resource",
                "branch_binding_hash",
                "branch_evidence_hash",
                "resource_registration_hash",
                "certification_receipt_hash",
                "decision_hash",
                "proposal_authorization_hash",
                "endpoint_certificate_hash",
                "f_lower",
                "f_upper",
                "q_lower",
                "q_upper",
                "r_half_lower",
                "r_half_upper",
                "r_one_lower",
                "r_one_upper",
                "half_response_lower",
                "half_response_upper",
                "one_response_lower",
                "one_response_upper",
                "independently_upper_certified",
                "live_alias_reconstruction_authority",
                "calibration_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
                "authority_scope",
            )
        }

    @property
    def projection_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(), prefix=_COMPACT_ALIAS_PROJECTION_PREFIX
        )
        if self._projection_hash is not None and self._projection_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "compact source alias projection was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "compact source alias projection hash does not match",
            )
        object.__setattr__(self, "_projection_hash", observed)
        return observed

    def assert_matches_live(
        self, alias: V1CompactFQRBranchFamilySourceAlias
    ) -> str:
        replayed = type(self).from_live(alias)
        if replayed.projection_hash != self.projection_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "fresh compact alias differs from its checkpoint projection",
            )
        return replayed.source_alias_hash


def _projected_alias_sort_key(
    alias: V1CompactFQRSourceAliasCheckpoint,
) -> tuple[float, str, str, int]:
    return (
        -alias.registered_resource,
        alias.branch_id,
        alias.packed_action_hash,
        alias.endpoint_index,
    )


def _projected_alias_intersection(
    aliases: tuple[V1CompactFQRSourceAliasCheckpoint, ...],
) -> tuple[float, ...]:
    if not aliases:
        _refuse(
            V1FQRCheckpointRefusalCode.MISSING_BRANCH,
            "compact queue entry has no source aliases",
        )
    rows = tuple(value.scalar_intervals() for value in aliases)
    values: list[float] = []
    for offset in range(0, 12, 2):
        lower = max(row[offset] for row in rows)
        upper = min(row[offset + 1] for row in rows)
        if lower > upper:
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "duplicate-action alias intervals have an empty intersection",
            )
        values.extend((lower, upper))
    return tuple(values)


@dataclass(frozen=True, slots=True)
class V1CompactFQRQueueEntryCheckpoint:
    """One unique packed physical action in an inert pre-K queue projection."""

    source_entry_hash: str
    representative_branch_id: str
    representative_endpoint_index: int
    representative_frozen_action_hash: str
    packed_action_hash: str
    packed_action: V1PackedRemovalAction = field(repr=False, compare=False)
    parent_mask_hash: str
    resulting_mask_hash: str
    route_axis_hash: str
    route_count: int
    registered_resource: float
    resource_units: str
    resource_source_hash: str
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
    source_aliases: tuple[V1CompactFQRSourceAliasCheckpoint, ...]
    live_entry_reconstruction_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQR_CHECKPOINT_AUTHORITY_SCOPE
    schema: str = FQR_COMPACT_ENTRY_PROJECTION_SCHEMA
    _projection_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        branch = _branch_id(
            self.representative_branch_id, name="queue representative branch"
        )
        endpoint = _exact_positive_int(
            self.representative_endpoint_index,
            name="queue representative endpoint",
        )
        for name in (
            "source_entry_hash",
            "representative_frozen_action_hash",
            "packed_action_hash",
            "parent_mask_hash",
            "resulting_mask_hash",
            "route_axis_hash",
            "resource_units",
            "resource_source_hash",
        ):
            _identity(getattr(self, name), name=name)
        route_count = _exact_positive_int(self.route_count, name="queue route count")
        resource = _finite_positive(
            self.registered_resource, name="queue registered resource"
        )
        for name in (
            "f_lower",
            "f_upper",
            "q_lower",
            "q_upper",
            "r_half_lower",
            "r_half_upper",
            "r_one_lower",
            "r_one_upper",
            "half_response_lower",
            "half_response_upper",
            "one_response_lower",
            "one_response_upper",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        aliases = tuple(self.source_aliases)
        if (
            not aliases
            or len(aliases) > len(CANONICAL_FQR_BRANCH_IDS)
            or any(type(value) is not V1CompactFQRSourceAliasCheckpoint for value in aliases)
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "queue entry requires one or two exact compact alias projections",
            )
        for value in aliases:
            value.assert_integrity()
        if (
            aliases != tuple(sorted(aliases, key=_projected_alias_sort_key))
            or len({(value.branch_id, value.endpoint_index) for value in aliases})
            != len(aliases)
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.POST_SELECTION,
                "queue alias projections are duplicated or reordered",
            )
        representative = aliases[0]
        scalar_values = (
            self.f_lower,
            self.f_upper,
            self.q_lower,
            self.q_upper,
            self.r_half_lower,
            self.r_half_upper,
            self.r_one_lower,
            self.r_one_upper,
            self.half_response_lower,
            self.half_response_upper,
            self.one_response_lower,
            self.one_response_upper,
        )
        if (
            representative.branch_id != branch
            or representative.endpoint_index != endpoint
            or representative.frozen_action_hash
            != self.representative_frozen_action_hash
            or representative.registered_resource != resource
            or any(value.packed_action_hash != self.packed_action_hash for value in aliases)
            or any(value.registered_resource != resource for value in aliases)
            or _projected_alias_intersection(aliases) != scalar_values
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "queue entry does not replay its source-alias intersection",
            )
        if type(self.packed_action) is not V1PackedRemovalAction:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "queue entry checkpoint requires one exact packed action",
            )
        child = self.packed_action.resulting_mask
        if (
            self.packed_action.action_hash != self.packed_action_hash
            or _direct_mask_hash(child) != self.resulting_mask_hash
            or child.route_axis_hash != self.route_axis_hash
            or child.route_count != route_count
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "queue entry packed action differs from its physical identity",
            )
        if any(
            getattr(self, name) is not False
            for name in (
                "live_entry_reconstruction_authority",
                "calibration_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "queue entry checkpoint crossed its inert boundary",
            )
        if (
            self.authority_scope != FQR_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema != FQR_COMPACT_ENTRY_PROJECTION_SCHEMA
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "queue entry checkpoint schema is invalid",
            )
        object.__setattr__(self, "representative_branch_id", branch)
        object.__setattr__(self, "representative_endpoint_index", endpoint)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "registered_resource", resource)
        object.__setattr__(self, "source_aliases", aliases)
        self.assert_integrity()

    @classmethod
    def from_live(
        cls, entry: V1CompactFQRBranchFamilyQueueEntry
    ) -> "V1CompactFQRQueueEntryCheckpoint":
        if type(entry) is not V1CompactFQRBranchFamilyQueueEntry:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "queue entry projection requires the exact compact entry",
            )
        try:
            source_hash = entry.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"compact queue entry failed source validation: {error}",
            ) from error
        return cls(
            source_entry_hash=source_hash,
            representative_branch_id=entry.representative_branch.value,
            representative_endpoint_index=entry.representative_endpoint_index,
            representative_frozen_action_hash=entry.representative_frozen_action_hash,
            packed_action_hash=entry.packed_action_hash,
            packed_action=entry.packed_action,
            parent_mask_hash=entry.parent_mask_hash,
            resulting_mask_hash=entry.resulting_mask_hash,
            route_axis_hash=entry.route_axis_hash,
            route_count=entry.route_count,
            registered_resource=entry.registered_resource,
            resource_units=entry.resource_units,
            resource_source_hash=entry.resource_source_hash,
            f_lower=entry.f_lower,
            f_upper=entry.f_upper,
            q_lower=entry.q_lower,
            q_upper=entry.q_upper,
            r_half_lower=entry.r_half_lower,
            r_half_upper=entry.r_half_upper,
            r_one_lower=entry.r_one_lower,
            r_one_upper=entry.r_one_upper,
            half_response_lower=entry.half_response_lower,
            half_response_upper=entry.half_response_upper,
            one_response_lower=entry.one_response_lower,
            one_response_upper=entry.one_response_upper,
            source_aliases=tuple(
                V1CompactFQRSourceAliasCheckpoint.from_live(value)
                for value in entry.source_aliases
            ),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_entry_hash": self.source_entry_hash,
            "representative_branch_id": self.representative_branch_id,
            "representative_endpoint_index": self.representative_endpoint_index,
            "representative_frozen_action_hash": (
                self.representative_frozen_action_hash
            ),
            "packed_action_hash": self.packed_action_hash,
            "packed_action_content_hash": self.packed_action.action_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "resulting_mask_hash": self.resulting_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "registered_resource": self.registered_resource,
            "resource_units": self.resource_units,
            "resource_source_hash": self.resource_source_hash,
            "f_lower": self.f_lower,
            "f_upper": self.f_upper,
            "q_lower": self.q_lower,
            "q_upper": self.q_upper,
            "r_half_lower": self.r_half_lower,
            "r_half_upper": self.r_half_upper,
            "r_one_lower": self.r_one_lower,
            "r_one_upper": self.r_one_upper,
            "half_response_lower": self.half_response_lower,
            "half_response_upper": self.half_response_upper,
            "one_response_lower": self.one_response_lower,
            "one_response_upper": self.one_response_upper,
            "source_alias_projection_hashes": tuple(
                value.projection_hash for value in self.source_aliases
            ),
            "live_entry_reconstruction_authority": (
                self.live_entry_reconstruction_authority
            ),
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def projection_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for value in self.source_aliases:
            value.assert_integrity()
        if (
            self.packed_action.action_hash != self.packed_action_hash
            or _direct_mask_hash(self.packed_action.resulting_mask)
            != self.resulting_mask_hash
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "queue entry checkpoint packed action was tampered",
            )
        observed = content_hash(
            self._payload(), prefix=_COMPACT_ENTRY_PROJECTION_PREFIX
        )
        if self._projection_hash is not None and self._projection_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "compact queue entry projection was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "compact queue entry projection hash does not match",
            )
        object.__setattr__(self, "_projection_hash", observed)
        return observed

    def assert_matches_live(
        self, entry: V1CompactFQRBranchFamilyQueueEntry
    ) -> str:
        replayed = type(self).from_live(entry)
        if replayed.projection_hash != self.projection_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "fresh compact queue entry differs from its checkpoint projection",
            )
        return replayed.source_entry_hash


def _compact_queue_entry_sort_key(
    entry: V1CompactFQRQueueEntryCheckpoint,
) -> tuple[float, str, str, int]:
    return (
        -entry.registered_resource,
        entry.representative_branch_id,
        entry.packed_action_hash,
        entry.representative_endpoint_index,
    )


def _normalize_source_branch_hashes(
    rows: object,
) -> tuple[tuple[str, str], ...]:
    if type(rows) is not tuple or len(rows) not in {1, 2}:
        _refuse(
            V1FQRCheckpointRefusalCode.MISSING_BRANCH,
            "source branch evidence must name one dominated family or both branches",
        )
    normalized: list[tuple[str, str]] = []
    for row in rows:
        if type(row) is not tuple or len(row) != 2:
            _refuse(
                V1FQRCheckpointRefusalCode.MALFORMED,
                "source branch evidence coordinates must be exact pairs",
            )
        normalized.append(
            (
                _branch_id(row[0], name="source evidence branch"),
                _identity(row[1], name="source branch evidence hash"),
            )
        )
    result = tuple(normalized)
    if result != tuple(sorted(result)) or len({row[0] for row in result}) != len(
        result
    ):
        _refuse(
            V1FQRCheckpointRefusalCode.POST_SELECTION,
            "source branch evidence rows were duplicated or reordered",
        )
    return result


def _normalize_source_branch_negative_hashes(
    rows: object,
) -> tuple[tuple[str, str], ...]:
    if type(rows) is not tuple or len(rows) not in {0, 1}:
        _refuse(
            V1FQRCheckpointRefusalCode.MISSING_BRANCH,
            "stable-negative source evidence must be empty or name one branch",
        )
    normalized: list[tuple[str, str]] = []
    for row in rows:
        if type(row) is not tuple or len(row) != 2:
            _refuse(
                V1FQRCheckpointRefusalCode.MALFORMED,
                "stable-negative source evidence coordinates must be exact pairs",
            )
        normalized.append(
            (
                _branch_id(row[0], name="stable-negative source branch"),
                _identity(row[1], name="stable-negative source evidence hash"),
            )
        )
    result = tuple(normalized)
    if result != tuple(sorted(result)) or len({row[0] for row in result}) != len(
        result
    ):
        _refuse(
            V1FQRCheckpointRefusalCode.POST_SELECTION,
            "stable-negative source evidence rows were duplicated or reordered",
        )
    return result


def _normalize_complete_source_branch_hashes(
    *, positive: object, negative: object
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    positives = _normalize_source_branch_hashes(positive)
    negatives = _normalize_source_branch_negative_hashes(negative)
    branches = tuple(row[0] for row in positives + negatives)
    if (
        len(branches) not in {1, 2}
        or len(set(branches)) != len(branches)
        or (negatives and (len(positives) != 1 or len(branches) != 2))
    ):
        _refuse(
            V1FQRCheckpointRefusalCode.MISSING_BRANCH,
            "queue provenance must contain distinct complete positive-or-stable-"
            "negative evidence for every retained branch",
        )
    return positives, negatives


@dataclass(frozen=True, slots=True)
class V1CompactFQRBranchFamilyQueueCheckpoint:
    """Inert, duplicate-quotiented projection of the complete pre-K queue.

    This object deliberately projects the live materialization grant away.  It
    retains every independently upper-certified physical endpoint and enough
    scalar/mask identity to compare a freshly rebuilt live authorization.
    """

    source_authorization_hash: str
    prefrozen_family_hash: str
    problem_hash: str
    policy_hash: str
    result_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_count: int
    discovery_prefix_hash: str
    discovery_evidence_hash: str
    joint_fqr_family_registration_hash: str
    resource_units: str
    resource_source_hash: str
    source_branch_evidence_hashes: tuple[tuple[str, str], ...]
    dominance_certificate_hash: str | None
    queue_entries: tuple[V1CompactFQRQueueEntryCheckpoint, ...] = field(
        repr=False, compare=False
    )
    source_endpoint_count: int
    certified_alias_count: int
    unique_physical_action_count: int
    maximum_source_endpoint_count: int
    per_branch_winner_coordinates: tuple[tuple[str, int, str, float], ...]
    queue_head_source_entry_hash: str
    queue_head_branch_id: str
    queue_head_endpoint_index: int
    queue_head_packed_action_hash: str
    queue_head_registered_resource: float
    source_branch_negative_evidence_hashes: tuple[tuple[str, str], ...] = ()
    compact_only: bool = True
    raw_context_ids_retained: int = 0
    raw_record_arrays_retained: int = 0
    route_scale_arrays_retained: int = 0
    live_authorization_reconstruction_authority: bool = False
    proposal_queue_materialization_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    paired_bce_oracle_still_required: bool = True
    replay_requirement: str = FQR_COMPACT_REPLAY_REQUIREMENT
    authority_scope: str = FQR_CHECKPOINT_AUTHORITY_SCOPE
    schema: str = FQR_COMPACT_FAMILY_QUEUE_PROJECTION_SCHEMA
    _projection_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "source_authorization_hash",
            "prefrozen_family_hash",
            "problem_hash",
            "policy_hash",
            "result_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "discovery_prefix_hash",
            "discovery_evidence_hash",
            "joint_fqr_family_registration_hash",
            "resource_units",
            "resource_source_hash",
            "queue_head_source_entry_hash",
            "queue_head_packed_action_hash",
        ):
            _identity(getattr(self, name), name=name)
        generation = _exact_nonnegative_int(
            self.proposal_generation, name="queue proposal generation"
        )
        route_count = _exact_positive_int(self.route_count, name="queue route count")
        source_count = _exact_positive_int(
            self.source_endpoint_count, name="source endpoint count"
        )
        alias_count = _exact_positive_int(
            self.certified_alias_count, name="certified alias count"
        )
        unique_count = _exact_positive_int(
            self.unique_physical_action_count, name="unique physical action count"
        )
        maximum = _exact_positive_int(
            self.maximum_source_endpoint_count,
            name="maximum source endpoint count",
        )
        head_endpoint = _exact_positive_int(
            self.queue_head_endpoint_index, name="queue head endpoint"
        )
        head_resource = _finite_positive(
            self.queue_head_registered_resource, name="queue head resource"
        )
        head_branch = _branch_id(self.queue_head_branch_id, name="queue head branch")
        source_hashes, negative_hashes = _normalize_complete_source_branch_hashes(
            positive=self.source_branch_evidence_hashes,
            negative=self.source_branch_negative_evidence_hashes,
        )
        expected_schema = (
            FQR_COMPACT_FAMILY_QUEUE_PROJECTION_V2_SCHEMA
            if negative_hashes
            else FQR_COMPACT_FAMILY_QUEUE_PROJECTION_SCHEMA
        )
        if self.schema != expected_schema:
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "compact queue schema does not match its positive/stable-negative "
                "provenance version",
            )
        source_branches = tuple(row[0] for row in source_hashes)
        complete_source_branches = tuple(
            row[0] for row in source_hashes + negative_hashes
        )
        if len(complete_source_branches) == 1:
            dominance = _identity(
                self.dominance_certificate_hash,
                name="pre-C dominance certificate hash",
            )
        else:
            if self.dominance_certificate_hash is not None:
                _refuse(
                    V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                    "complete two-branch queue cannot carry an omission certificate",
                )
            dominance = None
        entries = tuple(self.queue_entries)
        if (
            not entries
            or len(entries) > COMPACT_FQR_BRANCH_FAMILY_MAX_SOURCE_ENDPOINTS
            or any(type(value) is not V1CompactFQRQueueEntryCheckpoint for value in entries)
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
                "compact queue is empty, inexact, or exceeds its endpoint bound",
            )
        for entry in entries:
            entry.assert_integrity()
        if (
            entries != tuple(sorted(entries, key=_compact_queue_entry_sort_key))
            or len({entry.packed_action_hash for entry in entries}) != len(entries)
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.POST_SELECTION,
                "compact physical queue was reordered or not duplicate-quotiented",
            )
        if any(
            entry.parent_mask_hash != self.current_mask_hash
            or entry.route_axis_hash != self.route_axis_hash
            or entry.route_count != route_count
            or entry.resource_units != self.resource_units
            or entry.resource_source_hash != self.resource_source_hash
            for entry in entries
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "one compact queue entry left its family mask/resource coordinates",
            )
        aliases = tuple(alias for entry in entries for alias in entry.source_aliases)
        evidence_by_branch = dict(source_hashes)
        if (
            maximum != COMPACT_FQR_BRANCH_FAMILY_MAX_SOURCE_ENDPOINTS
            or source_count > maximum
            or alias_count != len(aliases)
            or unique_count != len(entries)
            or alias_count > source_count
            or set(alias.branch_id for alias in aliases) != set(source_branches)
            or any(
                evidence_by_branch.get(alias.branch_id) != alias.branch_evidence_hash
                for alias in aliases
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.MISSING_BRANCH,
                "compact queue counts or source-branch alias coverage disagree",
            )
        expected_winners: list[tuple[str, int, str, float]] = []
        winning_aliases: list[V1CompactFQRSourceAliasCheckpoint] = []
        for branch in source_branches:
            branch_aliases = tuple(alias for alias in aliases if alias.branch_id == branch)
            winner = min(branch_aliases, key=_projected_alias_sort_key)
            winning_aliases.append(winner)
            expected_winners.append(
                (
                    branch,
                    winner.endpoint_index,
                    winner.packed_action_hash,
                    winner.registered_resource,
                )
            )
        winners = tuple(sorted(expected_winners))
        if type(self.per_branch_winner_coordinates) is not tuple:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "per-branch winners must be one exact tuple",
            )
        observed_winners = tuple(self.per_branch_winner_coordinates)
        if observed_winners != winners:
            _refuse(
                V1FQRCheckpointRefusalCode.POST_SELECTION,
                "per-branch compact winners do not replay from the full alias union",
            )
        global_winner = min(winning_aliases, key=_projected_alias_sort_key)
        head = entries[0]
        if (
            head.source_entry_hash != self.queue_head_source_entry_hash
            or head.representative_branch_id != head_branch
            or head.representative_endpoint_index != head_endpoint
            or head.packed_action_hash != self.queue_head_packed_action_hash
            or head.registered_resource != head_resource
            or global_winner.branch_id != head_branch
            or global_winner.endpoint_index != head_endpoint
            or global_winner.packed_action_hash != head.packed_action_hash
            or global_winner.registered_resource != head.registered_resource
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.POST_SELECTION,
                "compact queue head differs from the registered global winner rule",
            )
        if (
            self.compact_only is not True
            or self.raw_context_ids_retained != 0
            or self.raw_record_arrays_retained != 0
            or self.route_scale_arrays_retained != 0
            or self.paired_bce_oracle_still_required is not True
            or any(
                getattr(self, name) is not False
                for name in (
                    "live_authorization_reconstruction_authority",
                    "proposal_queue_materialization_authority",
                    "calibration_authority",
                    "bce_oracle_authority",
                    "irreversible_decision_authority",
                    "mask_commit_authority",
                    "terminal_authority",
                    "pareto_authority",
                )
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "decoded compact queue crossed its inert checkpoint boundary",
            )
        if (
            self.replay_requirement != FQR_COMPACT_REPLAY_REQUIREMENT
            or self.authority_scope != FQR_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema
            != (
                FQR_COMPACT_FAMILY_QUEUE_PROJECTION_V2_SCHEMA
                if negative_hashes
                else FQR_COMPACT_FAMILY_QUEUE_PROJECTION_SCHEMA
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "compact queue checkpoint schema or replay contract is invalid",
            )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "source_endpoint_count", source_count)
        object.__setattr__(self, "certified_alias_count", alias_count)
        object.__setattr__(self, "unique_physical_action_count", unique_count)
        object.__setattr__(self, "maximum_source_endpoint_count", maximum)
        object.__setattr__(self, "queue_head_endpoint_index", head_endpoint)
        object.__setattr__(self, "queue_head_registered_resource", head_resource)
        object.__setattr__(self, "queue_head_branch_id", head_branch)
        object.__setattr__(self, "source_branch_evidence_hashes", source_hashes)
        object.__setattr__(
            self,
            "source_branch_negative_evidence_hashes",
            negative_hashes,
        )
        object.__setattr__(self, "dominance_certificate_hash", dominance)
        object.__setattr__(self, "queue_entries", entries)
        object.__setattr__(self, "per_branch_winner_coordinates", winners)
        self.assert_integrity()

    @classmethod
    def from_live(
        cls, authorization: V1CompactFQRBranchFamilyAuthorization
    ) -> "V1CompactFQRBranchFamilyQueueCheckpoint":
        if type(authorization) is not V1CompactFQRBranchFamilyAuthorization:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "compact queue projection requires exact live branch-family authority",
            )
        try:
            source_hash = authorization.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"compact branch-family authorization failed validation: {error}",
            ) from error
        return cls(
            source_authorization_hash=source_hash,
            prefrozen_family_hash=authorization.prefrozen_family_hash,
            problem_hash=authorization.problem_hash,
            policy_hash=authorization.policy_hash,
            result_hash=authorization.result_hash,
            chart_identity_hash=authorization.chart_identity_hash,
            current_mask_hash=authorization.current_mask_hash,
            proposal_generation=authorization.proposal_generation,
            route_axis_hash=authorization.route_axis_hash,
            route_count=authorization.route_count,
            discovery_prefix_hash=authorization.discovery_prefix_hash,
            discovery_evidence_hash=authorization.discovery_evidence_hash,
            joint_fqr_family_registration_hash=(
                authorization.joint_fqr_family_registration_hash
            ),
            resource_units=authorization.resource_units,
            resource_source_hash=authorization.resource_source_hash,
            source_branch_evidence_hashes=authorization.source_branch_evidence_hashes,
            dominance_certificate_hash=authorization.dominance_certificate_hash,
            queue_entries=tuple(
                V1CompactFQRQueueEntryCheckpoint.from_live(value)
                for value in authorization.queue_entries
            ),
            source_endpoint_count=authorization.source_endpoint_count,
            certified_alias_count=authorization.certified_alias_count,
            unique_physical_action_count=authorization.unique_physical_action_count,
            maximum_source_endpoint_count=authorization.maximum_source_endpoint_count,
            per_branch_winner_coordinates=authorization.per_branch_winner_coordinates,
            queue_head_source_entry_hash=authorization.queue_head_entry_hash,
            queue_head_branch_id=authorization.queue_head_branch.value,
            queue_head_endpoint_index=authorization.queue_head_endpoint_index,
            queue_head_packed_action_hash=authorization.queue_head_packed_action_hash,
            queue_head_registered_resource=authorization.queue_head_registered_resource,
            source_branch_negative_evidence_hashes=(
                authorization.source_branch_negative_evidence_hashes
            ),
            schema=(
                FQR_COMPACT_FAMILY_QUEUE_PROJECTION_V2_SCHEMA
                if authorization.source_branch_negative_evidence_hashes
                else FQR_COMPACT_FAMILY_QUEUE_PROJECTION_SCHEMA
            ),
        )

    def _payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "source_authorization_hash": self.source_authorization_hash,
            "prefrozen_family_hash": self.prefrozen_family_hash,
            "problem_hash": self.problem_hash,
            "policy_hash": self.policy_hash,
            "result_hash": self.result_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "discovery_prefix_hash": self.discovery_prefix_hash,
            "discovery_evidence_hash": self.discovery_evidence_hash,
            "joint_fqr_family_registration_hash": (
                self.joint_fqr_family_registration_hash
            ),
            "resource_units": self.resource_units,
            "resource_source_hash": self.resource_source_hash,
            "source_branch_evidence_hashes": self.source_branch_evidence_hashes,
            "dominance_certificate_hash": self.dominance_certificate_hash,
            "queue_entry_projection_hashes": tuple(
                value.projection_hash for value in self.queue_entries
            ),
            "source_endpoint_count": self.source_endpoint_count,
            "certified_alias_count": self.certified_alias_count,
            "unique_physical_action_count": self.unique_physical_action_count,
            "maximum_source_endpoint_count": self.maximum_source_endpoint_count,
            "per_branch_winner_coordinates": self.per_branch_winner_coordinates,
            "queue_head_source_entry_hash": self.queue_head_source_entry_hash,
            "queue_head_branch_id": self.queue_head_branch_id,
            "queue_head_endpoint_index": self.queue_head_endpoint_index,
            "queue_head_packed_action_hash": self.queue_head_packed_action_hash,
            "queue_head_registered_resource": self.queue_head_registered_resource,
            "compact_only": self.compact_only,
            "raw_context_ids_retained": self.raw_context_ids_retained,
            "raw_record_arrays_retained": self.raw_record_arrays_retained,
            "route_scale_arrays_retained": self.route_scale_arrays_retained,
            "live_authorization_reconstruction_authority": (
                self.live_authorization_reconstruction_authority
            ),
            "proposal_queue_materialization_authority": (
                self.proposal_queue_materialization_authority
            ),
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "paired_bce_oracle_still_required": self.paired_bce_oracle_still_required,
            "replay_requirement": self.replay_requirement,
            "authority_scope": self.authority_scope,
        }
        if self.source_branch_negative_evidence_hashes:
            payload["source_branch_negative_evidence_hashes"] = (
                self.source_branch_negative_evidence_hashes
            )
        return payload

    @property
    def projection_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for value in self.queue_entries:
            value.assert_integrity()
        observed = content_hash(
            self._payload(), prefix=_COMPACT_FAMILY_QUEUE_PROJECTION_PREFIX
        )
        if self._projection_hash is not None and self._projection_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "compact branch-family queue projection was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "compact branch-family queue projection hash does not match",
            )
        object.__setattr__(self, "_projection_hash", observed)
        return observed

    def assert_matches_live(
        self, authorization: V1CompactFQRBranchFamilyAuthorization
    ) -> str:
        replayed = type(self).from_live(authorization)
        if replayed.projection_hash != self.projection_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "fresh compact branch-family authorization differs from checkpoint",
            )
        return replayed.source_authorization_hash


@dataclass(frozen=True, slots=True)
class V1FQRAuthorizedAcquisitionCheckpoint:
    """Bounded queue-ready lineage projection with no live acquisition grant.

    Issuer-only numeric states and proposals are represented only by their
    content hashes and queue coordinates.  The exact current mask is retained
    because it is the bounded physical parent needed to revalidate a rebuilt
    live carrier; endpoint masks live in the compact queue projection.
    """

    source_acquisition_hash: str
    source_receipt_hash: str
    source_composition_hash: str
    campaign_contract_hash: str
    campaign_binding_hash: str
    registration_hash: str
    generation_index: int
    current_mask: HardMask = field(repr=False, compare=False)
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    committed_mask_lineage_hashes: tuple[str, ...]
    committed_mask_transition_receipt_hashes: tuple[str, ...]
    invalidation_chart_hash: str
    compact_domain_hash: str
    certification_scope_hash: str
    branch_family_authorization_hash: str
    prefrozen_family_hash: str
    joint_fqr_family_registration_hash: str
    source_branch_evidence_hashes: tuple[tuple[str, str], ...]
    queue_projection_hash: str
    action_numeric_state_hashes: tuple[str, ...]
    ordered_candidate_ids: tuple[str, ...]
    ordered_entry_hashes: tuple[str, ...]
    ordered_branch_endpoint_coordinates: tuple[tuple[str, int], ...]
    ordered_packed_action_hashes: tuple[str, ...]
    ordered_resulting_mask_hashes: tuple[str, ...]
    ordered_fqr_interval_hashes: tuple[str, ...]
    ordered_resource_receipt_hashes: tuple[str, ...]
    refresh_reason: str
    logical_sequence: int
    metadata_hash: str
    source_branch_negative_evidence_hashes: tuple[tuple[str, str], ...] = ()
    compact_only: bool = True
    raw_context_ids_retained: int = 0
    raw_record_arrays_retained: int = 0
    route_scale_arrays_retained: int = 0
    replay_required_before_queue_ready_resume: bool = True
    live_acquisition_reconstruction_authority: bool = False
    queue_ready_resume_authority: bool = False
    proposal_queue_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    paired_bce_oracle_still_required: bool = True
    replay_requirement: str = FQR_COMPACT_REPLAY_REQUIREMENT
    authority_scope: str = FQR_CHECKPOINT_AUTHORITY_SCOPE
    schema: str = FQR_AUTHORIZED_ACQUISITION_PROJECTION_SCHEMA
    _projection_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "source_acquisition_hash",
            "source_receipt_hash",
            "source_composition_hash",
            "campaign_contract_hash",
            "campaign_binding_hash",
            "registration_hash",
            "current_mask_hash",
            "route_axis_hash",
            "invalidation_chart_hash",
            "compact_domain_hash",
            "certification_scope_hash",
            "branch_family_authorization_hash",
            "prefrozen_family_hash",
            "joint_fqr_family_registration_hash",
            "queue_projection_hash",
            "refresh_reason",
            "metadata_hash",
        ):
            _identity(getattr(self, name), name=name)
        generation = _exact_nonnegative_int(
            self.generation_index, name="acquisition generation"
        )
        logical_sequence = _exact_nonnegative_int(
            self.logical_sequence, name="acquisition logical sequence"
        )
        route_count = _exact_positive_int(
            self.route_count, name="acquisition route count"
        )
        if type(self.current_mask) is not HardMask:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "acquisition checkpoint requires one exact current HardMask",
            )
        if (
            _direct_mask_hash(self.current_mask) != self.current_mask_hash
            or self.current_mask.route_axis_hash != self.route_axis_hash
            or self.current_mask.route_count != route_count
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "acquisition current mask differs from its direct physical identity",
            )
        lineage = tuple(
            _identity(value, name="committed mask lineage hash")
            for value in self.committed_mask_lineage_hashes
        )
        transitions = tuple(
            _identity(value, name="committed mask transition receipt hash")
            for value in self.committed_mask_transition_receipt_hashes
        )
        if (
            not lineage
            or len(lineage) > _MAX_SEQUENCE_ITEMS
            or lineage[-1] != self.current_mask_hash
            or len(transitions) != len(lineage) - 1
            or len(set(lineage)) != len(lineage)
            or len(set(transitions)) != len(transitions)
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "acquisition checkpoint has incomplete exact mask ancestry",
            )
        source_hashes, negative_hashes = _normalize_complete_source_branch_hashes(
            positive=self.source_branch_evidence_hashes,
            negative=self.source_branch_negative_evidence_hashes,
        )
        state_hashes = tuple(
            _identity(value, name="action numeric-state hash")
            for value in self.action_numeric_state_hashes
        )
        candidate_ids = tuple(
            _identity(value, name="ordered candidate ID")
            for value in self.ordered_candidate_ids
        )
        entry_hashes = tuple(
            _identity(value, name="ordered compact entry hash")
            for value in self.ordered_entry_hashes
        )
        packed_hashes = tuple(
            _identity(value, name="ordered packed action hash")
            for value in self.ordered_packed_action_hashes
        )
        resulting_hashes = tuple(
            _identity(value, name="ordered resulting mask hash")
            for value in self.ordered_resulting_mask_hashes
        )
        interval_hashes = tuple(
            _identity(value, name="ordered F/Q/R interval hash")
            for value in self.ordered_fqr_interval_hashes
        )
        resource_hashes = tuple(
            _identity(value, name="ordered resource receipt hash")
            for value in self.ordered_resource_receipt_hashes
        )
        coordinates: list[tuple[str, int]] = []
        if type(self.ordered_branch_endpoint_coordinates) is not tuple:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "branch/endpoint acquisition axis must be one exact tuple",
            )
        for row in self.ordered_branch_endpoint_coordinates:
            if type(row) is not tuple or len(row) != 2:
                _refuse(
                    V1FQRCheckpointRefusalCode.MALFORMED,
                    "branch/endpoint acquisition coordinates must be exact pairs",
                )
            coordinates.append(
                (
                    _branch_id(row[0], name="acquisition queue branch"),
                    _exact_positive_int(row[1], name="acquisition queue endpoint"),
                )
            )
        coordinate_axis = tuple(coordinates)
        width = len(candidate_ids)
        axes = (
            state_hashes,
            entry_hashes,
            coordinate_axis,
            packed_hashes,
            resulting_hashes,
            interval_hashes,
            resource_hashes,
        )
        if (
            width == 0
            or width > COMPACT_FQR_BRANCH_FAMILY_MAX_SOURCE_ENDPOINTS
            or any(len(axis) != width for axis in axes)
            or any(
                len(set(axis)) != len(axis)
                for axis in (state_hashes, candidate_ids, entry_hashes, packed_hashes)
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
                "acquisition queue axes are empty, duplicate, misaligned, or too large",
            )
        if (
            self.compact_only is not True
            or self.raw_context_ids_retained != 0
            or self.raw_record_arrays_retained != 0
            or self.route_scale_arrays_retained != 0
            or self.replay_required_before_queue_ready_resume is not True
            or self.paired_bce_oracle_still_required is not True
            or any(
                getattr(self, name) is not False
                for name in (
                    "live_acquisition_reconstruction_authority",
                    "queue_ready_resume_authority",
                    "proposal_queue_authority",
                    "calibration_authority",
                    "bce_oracle_authority",
                    "irreversible_decision_authority",
                    "mask_commit_authority",
                    "terminal_authority",
                    "pareto_authority",
                )
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "acquisition projection crossed its inert replay boundary",
            )
        if (
            self.replay_requirement != FQR_COMPACT_REPLAY_REQUIREMENT
            or self.authority_scope != FQR_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema
            != (
                FQR_AUTHORIZED_ACQUISITION_PROJECTION_V2_SCHEMA
                if negative_hashes
                else FQR_AUTHORIZED_ACQUISITION_PROJECTION_SCHEMA
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "acquisition checkpoint schema or replay contract is invalid",
            )
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "logical_sequence", logical_sequence)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "committed_mask_lineage_hashes", lineage)
        object.__setattr__(self, "committed_mask_transition_receipt_hashes", transitions)
        object.__setattr__(self, "source_branch_evidence_hashes", source_hashes)
        object.__setattr__(
            self,
            "source_branch_negative_evidence_hashes",
            negative_hashes,
        )
        object.__setattr__(self, "action_numeric_state_hashes", state_hashes)
        object.__setattr__(self, "ordered_candidate_ids", candidate_ids)
        object.__setattr__(self, "ordered_entry_hashes", entry_hashes)
        object.__setattr__(self, "ordered_branch_endpoint_coordinates", coordinate_axis)
        object.__setattr__(self, "ordered_packed_action_hashes", packed_hashes)
        object.__setattr__(self, "ordered_resulting_mask_hashes", resulting_hashes)
        object.__setattr__(self, "ordered_fqr_interval_hashes", interval_hashes)
        object.__setattr__(self, "ordered_resource_receipt_hashes", resource_hashes)
        self.assert_integrity()

    @classmethod
    def from_live(
        cls,
        acquisition: V1FQRAuthorizedAcquisition,
        queue: V1CompactFQRBranchFamilyQueueCheckpoint,
    ) -> "V1FQRAuthorizedAcquisitionCheckpoint":
        if type(acquisition) is not V1FQRAuthorizedAcquisition:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "acquisition projection requires the exact dedicated F/Q/R carrier",
            )
        if type(queue) is not V1CompactFQRBranchFamilyQueueCheckpoint:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "acquisition projection requires its exact compact queue projection",
            )
        try:
            queue_hash = queue.assert_integrity()
            composition_hash = acquisition.composition.assert_integrity()
            source_hash = acquisition.acquisition_hash
            receipt_hash = acquisition.receipt.receipt_hash
            for state, proposal in zip(
                acquisition.composition.action_numeric_states,
                acquisition.composition.proposals,
                strict=True,
            ):
                state.assert_integrity()
                proposal.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"live F/Q/R acquisition failed validation: {error}",
            ) from error
        composition = acquisition.composition
        receipt = acquisition.receipt
        if (
            queue.source_authorization_hash
            != composition.branch_family_authorization_hash
            or queue.prefrozen_family_hash != composition.prefrozen_family_hash
            or queue.joint_fqr_family_registration_hash
            != composition.joint_fqr_family_registration_hash
            or queue.source_branch_evidence_hashes
            != composition.source_branch_evidence_hashes
            or queue.source_branch_negative_evidence_hashes
            != getattr(composition, "source_branch_negative_evidence_hashes", ())
            or getattr(receipt, "source_branch_negative_evidence_hashes", ())
            != queue.source_branch_negative_evidence_hashes
            or tuple(value.source_entry_hash for value in queue.queue_entries)
            != composition.queue_entry_hashes
            or tuple(
                (
                    value.representative_branch_id,
                    value.representative_endpoint_index,
                )
                for value in queue.queue_entries
            )
            != composition.queue_branch_endpoint_coordinates
            or tuple(value.packed_action_hash for value in queue.queue_entries)
            != composition.queue_packed_action_hashes
            or tuple(value.resulting_mask_hash for value in queue.queue_entries)
            != composition.queue_resulting_mask_hashes
            or receipt.acquisition_hash != source_hash
            or receipt.composition_result_hash != composition_hash
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "live acquisition differs from the projected compact physical queue",
            )
        return cls(
            source_acquisition_hash=source_hash,
            source_receipt_hash=receipt_hash,
            source_composition_hash=composition_hash,
            campaign_contract_hash=receipt.campaign_contract_hash,
            campaign_binding_hash=receipt.campaign_binding_hash,
            registration_hash=receipt.registration_hash,
            generation_index=acquisition.generation_index,
            current_mask=acquisition.current_mask,
            current_mask_hash=receipt.current_mask_hash,
            route_axis_hash=receipt.route_axis_hash,
            route_count=acquisition.current_mask.route_count,
            committed_mask_lineage_hashes=acquisition.committed_mask_lineage_hashes,
            committed_mask_transition_receipt_hashes=(
                acquisition.committed_mask_transition_receipt_hashes
            ),
            invalidation_chart_hash=receipt.invalidation_chart_hash,
            compact_domain_hash=receipt.compact_domain_hash,
            certification_scope_hash=receipt.certification_scope_hash,
            branch_family_authorization_hash=(
                receipt.branch_family_authorization_hash
            ),
            prefrozen_family_hash=receipt.prefrozen_family_hash,
            joint_fqr_family_registration_hash=(
                receipt.joint_fqr_family_registration_hash
            ),
            source_branch_evidence_hashes=receipt.source_branch_evidence_hashes,
            queue_projection_hash=queue_hash,
            action_numeric_state_hashes=tuple(
                value.state_hash for value in composition.action_numeric_states
            ),
            ordered_candidate_ids=receipt.ordered_candidate_ids,
            ordered_entry_hashes=receipt.ordered_entry_hashes,
            ordered_branch_endpoint_coordinates=(
                receipt.ordered_branch_endpoint_coordinates
            ),
            ordered_packed_action_hashes=receipt.ordered_packed_action_hashes,
            ordered_resulting_mask_hashes=composition.queue_resulting_mask_hashes,
            ordered_fqr_interval_hashes=receipt.ordered_fqr_interval_hashes,
            ordered_resource_receipt_hashes=(
                composition.queue_resource_receipt_hashes
            ),
            refresh_reason=receipt.refresh_reason,
            logical_sequence=receipt.logical_sequence,
            metadata_hash=content_hash(
                dict(acquisition.metadata), prefix=_METADATA_PROJECTION_PREFIX
            ),
            source_branch_negative_evidence_hashes=(
                queue.source_branch_negative_evidence_hashes
            ),
            schema=(
                FQR_AUTHORIZED_ACQUISITION_PROJECTION_V2_SCHEMA
                if queue.source_branch_negative_evidence_hashes
                else FQR_AUTHORIZED_ACQUISITION_PROJECTION_SCHEMA
            ),
        )

    def assert_matches_queue(
        self, queue: V1CompactFQRBranchFamilyQueueCheckpoint
    ) -> str:
        if type(queue) is not V1CompactFQRBranchFamilyQueueCheckpoint:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "acquisition replay requires its exact compact queue projection",
            )
        queue_hash = queue.assert_integrity()
        if (
            self.queue_projection_hash != queue_hash
            or self.branch_family_authorization_hash
            != queue.source_authorization_hash
            or self.prefrozen_family_hash != queue.prefrozen_family_hash
            or self.joint_fqr_family_registration_hash
            != queue.joint_fqr_family_registration_hash
            or self.source_branch_evidence_hashes
            != queue.source_branch_evidence_hashes
            or self.source_branch_negative_evidence_hashes
            != queue.source_branch_negative_evidence_hashes
            or self.current_mask_hash != queue.current_mask_hash
            or self.route_axis_hash != queue.route_axis_hash
            or self.route_count != queue.route_count
            or self.generation_index != queue.proposal_generation
            or self.ordered_entry_hashes
            != tuple(value.source_entry_hash for value in queue.queue_entries)
            or self.ordered_branch_endpoint_coordinates
            != tuple(
                (
                    value.representative_branch_id,
                    value.representative_endpoint_index,
                )
                for value in queue.queue_entries
            )
            or self.ordered_packed_action_hashes
            != tuple(value.packed_action_hash for value in queue.queue_entries)
            or self.ordered_resulting_mask_hashes
            != tuple(value.resulting_mask_hash for value in queue.queue_entries)
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "acquisition projection differs from its compact queue projection",
            )
        return queue_hash

    def _payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "source_acquisition_hash": self.source_acquisition_hash,
            "source_receipt_hash": self.source_receipt_hash,
            "source_composition_hash": self.source_composition_hash,
            "campaign_contract_hash": self.campaign_contract_hash,
            "campaign_binding_hash": self.campaign_binding_hash,
            "registration_hash": self.registration_hash,
            "generation_index": self.generation_index,
            "current_mask_content_hash": _direct_mask_hash(self.current_mask),
            "current_mask_hash": self.current_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "committed_mask_lineage_hashes": self.committed_mask_lineage_hashes,
            "committed_mask_transition_receipt_hashes": (
                self.committed_mask_transition_receipt_hashes
            ),
            "invalidation_chart_hash": self.invalidation_chart_hash,
            "compact_domain_hash": self.compact_domain_hash,
            "certification_scope_hash": self.certification_scope_hash,
            "branch_family_authorization_hash": (
                self.branch_family_authorization_hash
            ),
            "prefrozen_family_hash": self.prefrozen_family_hash,
            "joint_fqr_family_registration_hash": (
                self.joint_fqr_family_registration_hash
            ),
            "source_branch_evidence_hashes": self.source_branch_evidence_hashes,
            "queue_projection_hash": self.queue_projection_hash,
            "action_numeric_state_hashes": self.action_numeric_state_hashes,
            "ordered_candidate_ids": self.ordered_candidate_ids,
            "ordered_entry_hashes": self.ordered_entry_hashes,
            "ordered_branch_endpoint_coordinates": (
                self.ordered_branch_endpoint_coordinates
            ),
            "ordered_packed_action_hashes": self.ordered_packed_action_hashes,
            "ordered_resulting_mask_hashes": self.ordered_resulting_mask_hashes,
            "ordered_fqr_interval_hashes": self.ordered_fqr_interval_hashes,
            "ordered_resource_receipt_hashes": (
                self.ordered_resource_receipt_hashes
            ),
            "refresh_reason": self.refresh_reason,
            "logical_sequence": self.logical_sequence,
            "metadata_hash": self.metadata_hash,
            "compact_only": self.compact_only,
            "raw_context_ids_retained": self.raw_context_ids_retained,
            "raw_record_arrays_retained": self.raw_record_arrays_retained,
            "route_scale_arrays_retained": self.route_scale_arrays_retained,
            "replay_required_before_queue_ready_resume": (
                self.replay_required_before_queue_ready_resume
            ),
            "live_acquisition_reconstruction_authority": (
                self.live_acquisition_reconstruction_authority
            ),
            "queue_ready_resume_authority": self.queue_ready_resume_authority,
            "proposal_queue_authority": self.proposal_queue_authority,
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "paired_bce_oracle_still_required": self.paired_bce_oracle_still_required,
            "replay_requirement": self.replay_requirement,
            "authority_scope": self.authority_scope,
        }
        if self.source_branch_negative_evidence_hashes:
            payload["source_branch_negative_evidence_hashes"] = (
                self.source_branch_negative_evidence_hashes
            )
        return payload

    @property
    def projection_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if _direct_mask_hash(self.current_mask) != self.current_mask_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "acquisition checkpoint current mask was tampered",
            )
        observed = content_hash(
            self._payload(), prefix=_AUTHORIZED_ACQUISITION_PROJECTION_PREFIX
        )
        if self._projection_hash is not None and self._projection_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "authorized-acquisition checkpoint projection was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "authorized-acquisition checkpoint projection hash does not match",
            )
        object.__setattr__(self, "_projection_hash", observed)
        return observed

    def assert_matches_live(
        self,
        acquisition: V1FQRAuthorizedAcquisition,
        queue: V1CompactFQRBranchFamilyQueueCheckpoint,
    ) -> str:
        replayed = type(self).from_live(acquisition, queue)
        if replayed.projection_hash != self.projection_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "fresh F/Q/R acquisition differs from its checkpoint projection",
            )
        return replayed.source_acquisition_hash


@dataclass(frozen=True, slots=True)
class V1CompactFQRCheckpointBundle:
    """Canonical v2 bounded checkpoint graph at the queue-ready seam.

    Decoding produces only inspection/replay evidence.  Queue-ready execution
    requires rebuilding every named live source and calling
    :meth:`assert_matches_live`; even a successful match does not open K/O or
    commit authority by itself.
    """

    dual_chart_geometry: V1DualChartGeometry = field(repr=False, compare=False)
    prefrozen_family: V1FQRPreFrozenFamilyCheckpoint = field(
        repr=False, compare=False
    )
    branch_ids: tuple[str, ...]
    branches: tuple[
        V1CompactFQRBranchCheckpoint | V1CompactFQRNegativeBranchCheckpoint,
        ...,
    ] = field(
        repr=False, compare=False
    )
    queue: V1CompactFQRBranchFamilyQueueCheckpoint = field(
        repr=False, compare=False
    )
    acquisition: V1FQRAuthorizedAcquisitionCheckpoint = field(
        repr=False, compare=False
    )
    raw_context_route_rows_serialized: bool = False
    raw_secant_records_serialized: bool = False
    route_scale_arrays_serialized: bool = False
    live_source_reconstruction_authority: bool = False
    queue_ready_resume_authority: bool = False
    proposal_queue_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    paired_bce_oracle_still_required: bool = True
    replay_requirement: str = FQR_COMPACT_REPLAY_REQUIREMENT
    authority_scope: str = FQR_CHECKPOINT_AUTHORITY_SCOPE
    schema: str = FQR_COMPACT_CHECKPOINT_SCHEMA
    _bundle_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.dual_chart_geometry) is not V1DualChartGeometry:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "compact checkpoint requires the exact reduced dual chart",
            )
        if type(self.prefrozen_family) is not V1FQRPreFrozenFamilyCheckpoint:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "compact checkpoint requires the exact pre-C family projection",
            )
        if type(self.queue) is not V1CompactFQRBranchFamilyQueueCheckpoint:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "compact checkpoint requires the exact physical queue projection",
            )
        if type(self.acquisition) is not V1FQRAuthorizedAcquisitionCheckpoint:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "compact checkpoint requires the exact acquisition projection",
            )
        try:
            self.dual_chart_geometry.assert_integrity()
            self.prefrozen_family.assert_integrity()
            queue_hash = self.queue.assert_integrity()
            self.acquisition.assert_integrity()
        except V1FQRCheckpointRefusal:
            raise
        except (TypeError, ValueError, PermissionError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"compact checkpoint source projection failed validation: {error}",
            ) from error
        branch_ids = _canonical_retained_branch_ids(self.branch_ids)
        branches = tuple(self.branches)
        if (
            len(branches) != len(branch_ids)
            or any(
                type(value)
                not in {
                    V1CompactFQRBranchCheckpoint,
                    V1CompactFQRNegativeBranchCheckpoint,
                }
                for value in branches
            )
            or tuple(value.branch_id for value in branches) != branch_ids
            or branch_ids != self.prefrozen_family.retained_branch_ids
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.MISSING_BRANCH,
                "compact branch projections do not follow the retained family axis",
            )
        for branch in branches:
            branch.assert_integrity()
        geometry = self.dual_chart_geometry
        family = self.prefrozen_family
        queue = self.queue
        if (
            geometry.current_mask_hash != family.current_mask_hash
            or geometry.proposal_generation != family.proposal_generation
            or geometry.route_count != family.route_count
            or geometry.route_axis_hash != family.route_axis_hash
            or queue.prefrozen_family_hash != family.source_family_hash
            or queue.problem_hash != family.problem_hash
            or queue.policy_hash != family.policy_hash
            or queue.result_hash != family.result_hash
            or queue.chart_identity_hash != family.chart_identity_hash
            or queue.current_mask_hash != family.current_mask_hash
            or queue.proposal_generation != family.proposal_generation
            or queue.route_axis_hash != family.route_axis_hash
            or queue.route_count != family.route_count
            or queue.discovery_prefix_hash != family.discovery_prefix_hash
            or queue.discovery_evidence_hash != family.discovery_evidence_hash
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "dual chart, pre-C family, and compact queue coordinates disagree",
            )
        dominance_hash = (
            None
            if family.dominance_checkpoint is None
            else family.dominance_checkpoint.source_certificate_hash
        )
        if queue.dominance_certificate_hash != dominance_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.MISSING_BRANCH,
                "compact queue branch omission is not bound to the pre-C certificate",
            )
        binding_by_branch = dict(family.retained_binding_hashes)
        evidence_by_branch = dict(
            queue.source_branch_evidence_hashes
            + queue.source_branch_negative_evidence_hashes
        )
        for branch in branches:
            source_evidence_hash = (
                branch.source_branch_evidence_hash
                if type(branch) is V1CompactFQRBranchCheckpoint
                else branch.source_branch_negative_evidence_hash
            )
            if (
                branch.prefrozen_family_hash != family.source_family_hash
                or binding_by_branch.get(branch.branch_id) != branch.branch_binding_hash
                or evidence_by_branch.get(branch.branch_id)
                != source_evidence_hash
                or branch.certification_receipt.chart_identity_hash
                != family.chart_identity_hash
                or branch.certification_receipt.current_mask_hash
                != family.current_mask_hash
                or branch.certification_receipt.proposal_generation
                != family.proposal_generation
                or branch.certification_receipt.route_count != family.route_count
                or branch.certification_receipt.route_axis_hash != family.route_axis_hash
                or branch.certification_receipt.joint_fqr_family_registration_hash
                != queue.joint_fqr_family_registration_hash
                or branch.resource_units != queue.resource_units
                or branch.resource_source_hash != queue.resource_source_hash
            ):
                _refuse(
                    V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                    "one compact branch projection left its family/queue coordinates",
                )
        if self.acquisition.queue_projection_hash != queue_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "acquisition projection is not bound to this compact queue",
            )
        self.acquisition.assert_matches_queue(queue)
        if (
            self.raw_context_route_rows_serialized is not False
            or self.raw_secant_records_serialized is not False
            or self.route_scale_arrays_serialized is not False
            or self.paired_bce_oracle_still_required is not True
            or any(
                getattr(self, name) is not False
                for name in (
                    "live_source_reconstruction_authority",
                    "queue_ready_resume_authority",
                    "proposal_queue_authority",
                    "calibration_authority",
                    "bce_oracle_authority",
                    "irreversible_decision_authority",
                    "mask_commit_authority",
                    "terminal_authority",
                    "pareto_authority",
                )
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "compact checkpoint bundle crossed its inert authority boundary",
            )
        if (
            self.replay_requirement != FQR_COMPACT_REPLAY_REQUIREMENT
            or self.authority_scope != FQR_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema != FQR_COMPACT_CHECKPOINT_SCHEMA
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "compact checkpoint bundle schema or replay contract is invalid",
            )
        object.__setattr__(self, "branch_ids", branch_ids)
        object.__setattr__(self, "branches", branches)
        self.assert_integrity()

    @classmethod
    def from_live(
        cls,
        *,
        dual_chart_geometry: V1DualChartGeometry,
        prefrozen_family: V1FQRKKTPreFrozenBranchFamily,
        branch_evidences: tuple[
            V1CompactFQRBranchEvidenceBinding
            | V1CompactFQRBranchNegativeEvidenceBinding,
            ...,
        ],
        queue_authorization: V1CompactFQRBranchFamilyAuthorization,
        acquisition: V1FQRAuthorizedAcquisition,
    ) -> "V1CompactFQRCheckpointBundle":
        if type(branch_evidences) is not tuple or any(
            type(value)
            not in {
                V1CompactFQRBranchEvidenceBinding,
                V1CompactFQRBranchNegativeEvidenceBinding,
            }
            for value in branch_evidences
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "compact checkpoint requires an exact tuple of branch evidence",
            )
        family_projection = V1FQRPreFrozenFamilyCheckpoint.from_live(
            prefrozen_family
        )
        evidence_by_branch: dict[
            str,
            V1CompactFQRBranchEvidenceBinding
            | V1CompactFQRBranchNegativeEvidenceBinding,
        ] = {}
        for evidence in branch_evidences:
            branch = evidence.branch.value
            if branch in evidence_by_branch:
                _refuse(
                    V1FQRCheckpointRefusalCode.MISSING_BRANCH,
                    "compact checkpoint repeats one branch evidence binding",
                )
            evidence_by_branch[branch] = evidence
        if set(evidence_by_branch) != set(family_projection.retained_branch_ids):
            _refuse(
                V1FQRCheckpointRefusalCode.MISSING_BRANCH,
                "compact checkpoint lacks an exact retained branch evidence set",
            )
        ordered_evidence = tuple(
            evidence_by_branch[branch]
            for branch in family_projection.retained_branch_ids
        )
        branches = tuple(
            (
                V1CompactFQRBranchCheckpoint.from_live(value)
                if type(value) is V1CompactFQRBranchEvidenceBinding
                else V1CompactFQRNegativeBranchCheckpoint.from_live(value)
            )
            for value in ordered_evidence
        )
        queue = V1CompactFQRBranchFamilyQueueCheckpoint.from_live(
            queue_authorization
        )
        acquisition_projection = V1FQRAuthorizedAcquisitionCheckpoint.from_live(
            acquisition, queue
        )
        return cls(
            dual_chart_geometry=dual_chart_geometry,
            prefrozen_family=family_projection,
            branch_ids=family_projection.retained_branch_ids,
            branches=branches,
            queue=queue,
            acquisition=acquisition_projection,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dual_chart_geometry_hash": self.dual_chart_geometry.geometry_hash,
            "prefrozen_family_projection_hash": self.prefrozen_family.projection_hash,
            "branch_ids": self.branch_ids,
            "branch_projection_hashes": tuple(
                value.projection_hash for value in self.branches
            ),
            "queue_projection_hash": self.queue.projection_hash,
            "acquisition_projection_hash": self.acquisition.projection_hash,
            "raw_context_route_rows_serialized": (
                self.raw_context_route_rows_serialized
            ),
            "raw_secant_records_serialized": self.raw_secant_records_serialized,
            "route_scale_arrays_serialized": self.route_scale_arrays_serialized,
            "live_source_reconstruction_authority": (
                self.live_source_reconstruction_authority
            ),
            "queue_ready_resume_authority": self.queue_ready_resume_authority,
            "proposal_queue_authority": self.proposal_queue_authority,
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "paired_bce_oracle_still_required": self.paired_bce_oracle_still_required,
            "replay_requirement": self.replay_requirement,
            "authority_scope": self.authority_scope,
        }

    @property
    def bundle_hash(self) -> str:
        return self.assert_integrity()

    @property
    def chart_identity_hash(self) -> str:
        return self.prefrozen_family.chart_identity_hash

    @property
    def current_mask_hash(self) -> str:
        return self.prefrozen_family.current_mask_hash

    @property
    def proposal_generation(self) -> int:
        return self.prefrozen_family.proposal_generation

    @property
    def route_axis_hash(self) -> str:
        return self.prefrozen_family.route_axis_hash

    @property
    def branch_family_registration_hash(self) -> str:
        return self.queue.joint_fqr_family_registration_hash

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.dual_chart_geometry.assert_integrity()
        self.prefrozen_family.assert_integrity()
        for branch in self.branches:
            branch.assert_integrity()
        self.queue.assert_integrity()
        self.acquisition.assert_integrity()
        observed = content_hash(self._payload(), prefix=_COMPACT_BUNDLE_PREFIX)
        if self._bundle_hash is not None and self._bundle_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "compact F/Q/R checkpoint bundle was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "compact F/Q/R checkpoint bundle hash does not match",
            )
        object.__setattr__(self, "_bundle_hash", observed)
        return observed

    def assert_matches_live(
        self,
        *,
        dual_chart_geometry: V1DualChartGeometry,
        prefrozen_family: V1FQRKKTPreFrozenBranchFamily,
        branch_evidences: tuple[
            V1CompactFQRBranchEvidenceBinding
            | V1CompactFQRBranchNegativeEvidenceBinding,
            ...,
        ],
        queue_authorization: V1CompactFQRBranchFamilyAuthorization,
        acquisition: V1FQRAuthorizedAcquisition,
    ) -> str:
        replayed = type(self).from_live(
            dual_chart_geometry=dual_chart_geometry,
            prefrozen_family=prefrozen_family,
            branch_evidences=branch_evidences,
            queue_authorization=queue_authorization,
            acquisition=acquisition,
        )
        if replayed.bundle_hash != self.bundle_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "fresh queue-ready live graph differs from the compact checkpoint",
            )
        return replayed.bundle_hash


def v1_fqr_inflight_pre_c_graph_hash(design: V1FQRBranchFamilyDesign) -> str:
    """Hash the exact pre-C design graph required to replay a family prefix."""

    if type(design) is not V1FQRBranchFamilyDesign:
        _refuse(
            V1FQRCheckpointRefusalCode.INVALID_TYPE,
            "in-flight pre-C graph hashing requires the exact family design",
        )
    try:
        design_hash = design.assert_integrity()
        registration = design.registration
        registration_hash = registration.assert_integrity()
        family = registration.prefrozen_family
        family_hash = family.assert_integrity()
    except (TypeError, ValueError, PermissionError) as error:
        raise V1FQRCheckpointRefusal(
            V1FQRCheckpointRefusalCode.HASH_MISMATCH,
            f"pre-C shared-C design graph failed validation: {error}",
        ) from error
    dominance = family.dominance_certificate
    dominance_hash = None
    complete_family_hash = family_hash
    if dominance is not None:
        try:
            dominance_hash = dominance.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"pre-C dominance certificate failed validation: {error}",
            ) from error
        complete_family_hash = dominance.source_complete_family_hash
    return content_hash(
        {
            "schema": "partition_free_v1_fqr_inflight_shared_c_pre_c_graph_v1",
            "branch_family_design_hash": design_hash,
            "joint_family_registration_hash": registration_hash,
            "prefrozen_family_hash": family_hash,
            "source_complete_family_hash": complete_family_hash,
            "dominance_certificate_hash": dominance_hash,
            "branch_bundle_hashes": tuple(
                value.bundle_hash for value in design.branch_bundles
            ),
            "branch_ids": tuple(
                value.branch.value for value in design.branch_bundles
            ),
            "chart_identity_hash": family.chart_identity_hash,
            "current_mask_hash": family.current_mask_hash,
            "proposal_generation": family.proposal_generation,
            "route_axis_hash": family.route_axis_hash,
            "route_count": family.route_count,
            "issued_before_C_outcomes": True,
            "C_outcomes_seen": False,
        },
        prefix=_INFLIGHT_C_PRE_C_GRAPH_PREFIX,
    )


@dataclass(frozen=True, slots=True)
class V1FQRInFlightSharedCCheckpointBundle:
    """Authority-inert v3 checkpoint for one atomic shared-C prefix.

    The exact bounded family snapshot contains integer sufficient statistics,
    never secant records or route rows.  It cannot be restored by hash alone:
    the caller must supply the exact pre-C family design, and a nonempty prefix
    additionally requires an exact expected snapshot hash.
    """

    pre_c_graph_hash: str
    prefrozen_family_hash: str
    source_complete_family_hash: str
    dominance_certificate_hash: str | None
    branch_family_design_hash: str
    branch_family_registration_hash: str
    branch_ids: tuple[str, ...]
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_count: int
    snapshot: V1FQRBranchFamilyAccumulatorSnapshot = field(
        repr=False, compare=False
    )
    snapshot_hash: str
    shared_record_count: int
    transaction_chain_hash: str
    exact_integer_sufficient_statistics_only: bool = True
    retained_context_ids: int = 0
    retained_secant_records: int = 0
    retained_record_arrays: int = 0
    retained_route_scale_arrays: int = 0
    exact_pre_c_design_required_for_restore: bool = True
    exact_expected_snapshot_hash_required_for_nonempty_restore: bool = True
    live_snapshot_reconstruction_authority: bool = False
    C_stream_resume_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    authority_scope: str = FQR_INFLIGHT_C_CHECKPOINT_AUTHORITY_SCOPE
    schema: str = FQR_INFLIGHT_C_CHECKPOINT_SCHEMA
    _bundle_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "pre_c_graph_hash",
            "prefrozen_family_hash",
            "source_complete_family_hash",
            "branch_family_design_hash",
            "branch_family_registration_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "snapshot_hash",
            "transaction_chain_hash",
        ):
            _identity(getattr(self, name), name=name)
        branch_ids = _canonical_retained_branch_ids(self.branch_ids)
        generation = _exact_nonnegative_int(
            self.proposal_generation, name="in-flight proposal generation"
        )
        route_count = _exact_positive_int(
            self.route_count, name="in-flight route count"
        )
        shared_count = _exact_nonnegative_int(
            self.shared_record_count, name="shared C record count"
        )
        dominance = self.dominance_certificate_hash
        if len(branch_ids) == 1:
            dominance = _identity(
                dominance, name="pre-C dominance certificate hash"
            )
            if self.source_complete_family_hash == self.prefrozen_family_hash:
                _refuse(
                    V1FQRCheckpointRefusalCode.MISSING_BRANCH,
                    "one-branch in-flight checkpoint lacks its complete-family identity",
                )
        elif dominance is not None:
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "complete two-branch in-flight checkpoint cannot claim branch omission",
            )
        if type(self.snapshot) is not V1FQRBranchFamilyAccumulatorSnapshot:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "in-flight checkpoint requires the exact bounded family snapshot",
            )
        try:
            observed_snapshot_hash = self.snapshot.assert_integrity()
        except PermissionError as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                f"shared-C snapshot crossed its authority boundary: {error}",
            ) from error
        except (TypeError, ValueError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"shared-C snapshot failed validation: {error}",
            ) from error
        snapshot_branch_ids = tuple(value.value for value in self.snapshot.branches)
        branch_snapshots = self.snapshot.branch_snapshots
        if (
            observed_snapshot_hash != self.snapshot_hash
            or self.snapshot.branch_family_design_hash
            != self.branch_family_design_hash
            or self.snapshot.joint_family_registration_hash
            != self.branch_family_registration_hash
            or snapshot_branch_ids != branch_ids
            or self.snapshot.shared_record_count != shared_count
            or self.snapshot.transaction_chain_hash != self.transaction_chain_hash
            or len(branch_snapshots) != len(branch_ids)
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "in-flight root differs from its exact atomic family snapshot",
            )
        for branch_snapshot in branch_snapshots:
            try:
                branch_snapshot.assert_integrity()
            except PermissionError as error:
                raise V1FQRCheckpointRefusal(
                    V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                    f"branch snapshot crossed its authority boundary: {error}",
                ) from error
            except (TypeError, ValueError) as error:
                raise V1FQRCheckpointRefusal(
                    V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                    f"branch snapshot failed validation: {error}",
                ) from error
            if (
                branch_snapshot.chart_identity_hash != self.chart_identity_hash
                or branch_snapshot.current_mask_hash != self.current_mask_hash
                or branch_snapshot.proposal_generation != generation
                or branch_snapshot.route_axis_hash != self.route_axis_hash
                or branch_snapshot.route_count != route_count
                or branch_snapshot.joint_fqr_family_registration_hash
                != self.branch_family_registration_hash
                or branch_snapshot.total_records_consumed != shared_count
            ):
                _refuse(
                    V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                    "branch snapshot differs from the shared family coordinates",
                )
            if (
                branch_snapshot.retained_context_ids != 0
                or branch_snapshot.retained_record_arrays != 0
                or branch_snapshot.raw_secant_records_retained is not False
            ):
                _refuse(
                    V1FQRCheckpointRefusalCode.RAW_RECORD_STATE_FORBIDDEN,
                    "branch snapshot retained raw C state",
                )
        if (
            self.exact_integer_sufficient_statistics_only is not True
            or self.retained_context_ids != 0
            or self.retained_secant_records != 0
            or self.retained_record_arrays != 0
            or self.retained_route_scale_arrays != 0
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.RAW_RECORD_STATE_FORBIDDEN,
                "in-flight checkpoint retained raw records, contexts, or route arrays",
            )
        if (
            self.exact_pre_c_design_required_for_restore is not True
            or self.exact_expected_snapshot_hash_required_for_nonempty_restore is not True
            or any(
                getattr(self, name) is not False
                for name in (
                    "live_snapshot_reconstruction_authority",
                    "C_stream_resume_authority",
                    "calibration_K_open_authority",
                    "paired_oracle_O_open_authority",
                    "irreversible_decision_authority",
                    "mask_commit_authority",
                    "terminal_authority",
                    "pareto_claim_authority",
                )
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "in-flight shared-C checkpoint crossed its inert authority boundary",
            )
        if (
            self.authority_scope != FQR_INFLIGHT_C_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema != FQR_INFLIGHT_C_CHECKPOINT_SCHEMA
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "in-flight shared-C checkpoint schema or scope is invalid",
            )
        object.__setattr__(self, "branch_ids", branch_ids)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "shared_record_count", shared_count)
        object.__setattr__(self, "dominance_certificate_hash", dominance)
        self.assert_integrity()

    @classmethod
    def from_live(
        cls,
        *,
        design: V1FQRBranchFamilyDesign,
        snapshot: V1FQRBranchFamilyAccumulatorSnapshot,
    ) -> "V1FQRInFlightSharedCCheckpointBundle":
        if type(design) is not V1FQRBranchFamilyDesign:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "in-flight capture requires the exact pre-C family design",
            )
        if type(snapshot) is not V1FQRBranchFamilyAccumulatorSnapshot:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "in-flight capture requires the exact family snapshot",
            )
        try:
            design_hash = design.assert_integrity()
            snapshot_hash = snapshot.assert_integrity()
            # This independently validates every branch sufficient statistic
            # against the pre-outcome design without retaining the live object.
            V1FQRBranchFamilyAccumulator.from_snapshot(
                design,
                snapshot,
                expected_snapshot_hash=snapshot_hash,
            )
        except PermissionError as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                f"in-flight source graph crossed its authority boundary: {error}",
            ) from error
        except (TypeError, ValueError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                f"in-flight source graph failed replay: {error}",
            ) from error
        registration = design.registration
        family = registration.prefrozen_family
        dominance = family.dominance_certificate
        return cls(
            pre_c_graph_hash=v1_fqr_inflight_pre_c_graph_hash(design),
            prefrozen_family_hash=family.family_hash,
            source_complete_family_hash=(
                family.family_hash
                if dominance is None
                else dominance.source_complete_family_hash
            ),
            dominance_certificate_hash=(
                None if dominance is None else dominance.certificate_hash
            ),
            branch_family_design_hash=design_hash,
            branch_family_registration_hash=design.common_family_hash,
            branch_ids=tuple(value.branch.value for value in design.branch_bundles),
            chart_identity_hash=family.chart_identity_hash,
            current_mask_hash=family.current_mask_hash,
            proposal_generation=family.proposal_generation,
            route_axis_hash=family.route_axis_hash,
            route_count=family.route_count,
            snapshot=snapshot,
            snapshot_hash=snapshot_hash,
            shared_record_count=snapshot.shared_record_count,
            transaction_chain_hash=snapshot.transaction_chain_hash,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "pre_c_graph_hash": self.pre_c_graph_hash,
            "prefrozen_family_hash": self.prefrozen_family_hash,
            "source_complete_family_hash": self.source_complete_family_hash,
            "dominance_certificate_hash": self.dominance_certificate_hash,
            "branch_family_design_hash": self.branch_family_design_hash,
            "branch_family_registration_hash": self.branch_family_registration_hash,
            "branch_ids": self.branch_ids,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "snapshot_hash": self.snapshot_hash,
            "shared_record_count": self.shared_record_count,
            "transaction_chain_hash": self.transaction_chain_hash,
            "exact_integer_sufficient_statistics_only": (
                self.exact_integer_sufficient_statistics_only
            ),
            "retained_context_ids": self.retained_context_ids,
            "retained_secant_records": self.retained_secant_records,
            "retained_record_arrays": self.retained_record_arrays,
            "retained_route_scale_arrays": self.retained_route_scale_arrays,
            "exact_pre_c_design_required_for_restore": (
                self.exact_pre_c_design_required_for_restore
            ),
            "exact_expected_snapshot_hash_required_for_nonempty_restore": (
                self.exact_expected_snapshot_hash_required_for_nonempty_restore
            ),
            "live_snapshot_reconstruction_authority": (
                self.live_snapshot_reconstruction_authority
            ),
            "C_stream_resume_authority": self.C_stream_resume_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_claim_authority": self.pareto_claim_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def bundle_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        try:
            self.snapshot.assert_integrity(self.snapshot_hash)
        except PermissionError as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                f"in-flight snapshot crossed its authority boundary: {error}",
            ) from error
        except (TypeError, ValueError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"in-flight snapshot was tampered: {error}",
            ) from error
        observed = content_hash(self._payload(), prefix=_INFLIGHT_C_BUNDLE_PREFIX)
        if self._bundle_hash is not None and self._bundle_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "in-flight shared-C checkpoint bundle was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "in-flight shared-C checkpoint bundle hash does not match",
            )
        object.__setattr__(self, "_bundle_hash", observed)
        return observed

    def snapshot_for_restore(
        self,
        *,
        design: V1FQRBranchFamilyDesign,
        expected_snapshot_hash: str | None = None,
    ) -> V1FQRBranchFamilyAccumulatorSnapshot:
        """Verify the external pre-C graph and return the inert exact snapshot."""

        if self.shared_record_count > 0 and expected_snapshot_hash is None:
            _refuse(
                V1FQRCheckpointRefusalCode.SNAPSHOT_HASH_REQUIRED,
                "nonempty shared-C restore requires an exact expected snapshot hash",
            )
        if type(design) is not V1FQRBranchFamilyDesign:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "shared-C restore requires the exact pre-C family design",
            )
        try:
            observed_snapshot_hash = self.snapshot.assert_integrity(
                expected_snapshot_hash
            )
        except PermissionError as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                f"shared-C restore crossed an authority boundary: {error}",
            ) from error
        except (TypeError, ValueError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"shared-C expected snapshot hash does not match: {error}",
            ) from error
        graph_hash = v1_fqr_inflight_pre_c_graph_hash(design)
        if (
            graph_hash != self.pre_c_graph_hash
            or design.design_hash != self.branch_family_design_hash
            or design.common_family_hash != self.branch_family_registration_hash
            or tuple(value.branch.value for value in design.branch_bundles)
            != self.branch_ids
            or observed_snapshot_hash != self.snapshot_hash
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "shared-C checkpoint belongs to another pre-C design graph",
            )
        return self.snapshot


@dataclass(frozen=True, slots=True)
class _V1LabelNativeSixRoleTrustedRestoreLineage:
    """Process-local proof that one accumulator came from a decoded bundle.

    This receipt is intentionally neither serializable nor exported.  The
    restored accumulator's object identity is the capability: subsequent
    rows still have to pass ``record_atomic``'s exact source/execution checks,
    and a later checkpoint capture replays the complete sufficient state.
    """

    source_bundle_hash: str
    design_hash: str
    branch_design_hashes: tuple[str, ...]
    joint_six_role_family_registration_hash: str
    source_snapshot_hash: str
    source_progress_receipt_hash: str
    source_shared_record_count: int
    source_transaction_chain_hash: str


_LABEL_NATIVE_SIX_ROLE_TRUSTED_RESTORE_LOCK: Final = threading.RLock()
_LABEL_NATIVE_SIX_ROLE_TRUSTED_RESTORE_LINEAGES: Final = (
    weakref.WeakKeyDictionary()
)


def _register_label_native_six_role_trusted_restore(
    *,
    accumulator: V1LabelNativeSixRoleCAccumulator,
    lineage: _V1LabelNativeSixRoleTrustedRestoreLineage,
) -> None:
    with _LABEL_NATIVE_SIX_ROLE_TRUSTED_RESTORE_LOCK:
        _LABEL_NATIVE_SIX_ROLE_TRUSTED_RESTORE_LINEAGES[accumulator] = lineage


def _require_label_native_six_role_trusted_restore_lineage(
    *,
    accumulator: V1LabelNativeSixRoleCAccumulator,
    design: V1LabelNativeSixRoleCDesign,
    snapshot: V1LabelNativeSixRoleAccumulatorSnapshot,
    source_progress_receipt_hash: str,
) -> _V1LabelNativeSixRoleTrustedRestoreLineage:
    with _LABEL_NATIVE_SIX_ROLE_TRUSTED_RESTORE_LOCK:
        lineage = _LABEL_NATIVE_SIX_ROLE_TRUSTED_RESTORE_LINEAGES.get(accumulator)
    if type(lineage) is not _V1LabelNativeSixRoleTrustedRestoreLineage:
        _refuse(
            V1FQRCheckpointRefusalCode.STALE_COORDINATES,
            "label-native C prefix was reconstructed outside the trusted "
            "decoded-checkpoint restore seam",
        )
    if (
        design.design_hash != lineage.design_hash
        or tuple(branch.design_hash for branch in design.branch_designs)
        != lineage.branch_design_hashes
        or design.joint_six_role_family_registration_hash
        != lineage.joint_six_role_family_registration_hash
        or accumulator.design is not design
        or snapshot.design_hash != lineage.design_hash
        or snapshot.shared_record_count < lineage.source_shared_record_count
    ):
        _refuse(
            V1FQRCheckpointRefusalCode.STALE_COORDINATES,
            "trusted label-native C restore lineage belongs to another design "
            "or a later prefix",
        )
    if snapshot.shared_record_count == lineage.source_shared_record_count:
        if (
            snapshot.snapshot_hash != lineage.source_snapshot_hash
            or source_progress_receipt_hash
            != lineage.source_progress_receipt_hash
            or snapshot.transaction_chain_hash
            != lineage.source_transaction_chain_hash
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "trusted label-native C restore prefix no longer matches its "
                "decoded checkpoint lineage",
            )
    elif snapshot.transaction_chain_hash == lineage.source_transaction_chain_hash:
        _refuse(
            V1FQRCheckpointRefusalCode.HASH_MISMATCH,
            "advanced label-native C restore retained its source transaction chain",
        )
    return lineage


@dataclass(frozen=True, slots=True)
class V1LabelNativeSixRoleInFlightCCheckpointBundle:
    """Authority-inert checkpoint for label-native six-role C sufficient state.

    The snapshot stores only exact integer sufficient statistics and hash
    chains.  Restoring a nonempty prefix requires both this snapshot hash and
    the exact live label-native six-role design, so checkpoint bytes cannot
    become an alternate source of C, K/O, commit, or terminal authority.
    """

    design_hash: str
    branch_design_hashes: tuple[str, ...]
    joint_six_role_family_registration_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_count: int
    snapshot: V1LabelNativeSixRoleAccumulatorSnapshot = field(
        repr=False,
        compare=False,
    )
    snapshot_hash: str
    source_progress_receipt_hash: str
    shared_record_count: int
    transaction_chain_hash: str
    _issuance_token: InitVar[object | None] = None
    branch_ids: tuple[str, ...] = CANONICAL_FQR_BRANCH_IDS
    exact_integer_sufficient_statistics_only: bool = True
    retained_context_ids: int = 0
    retained_secant_records: int = 0
    retained_record_arrays: int = 0
    retained_route_scale_arrays: int = 0
    exact_label_native_design_required_for_restore: bool = True
    exact_expected_snapshot_hash_required_for_nonempty_restore: bool = True
    live_snapshot_reconstruction_authority: bool = False
    C_stream_resume_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    authority_scope: str = (
        FQR_LABEL_NATIVE_SIX_ROLE_INFLIGHT_C_CHECKPOINT_AUTHORITY_SCOPE
    )
    schema: str = FQR_LABEL_NATIVE_SIX_ROLE_INFLIGHT_C_CHECKPOINT_SCHEMA
    _bundle_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    _decoded_for_trusted_restore: bool = field(
        init=False,
        default=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self, _issuance_token: object | None) -> None:
        if (
            _issuance_token is not _LABEL_NATIVE_SIX_ROLE_CAPTURE_ISSUANCE_TOKEN
            and _issuance_token is not _LABEL_NATIVE_SIX_ROLE_DECODE_ISSUANCE_TOKEN
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "label-native checkpoint roots may only be issued from a live "
                "authenticated accumulator or the strict checkpoint decoder",
            )
        object.__setattr__(
            self,
            "_decoded_for_trusted_restore",
            _issuance_token is _LABEL_NATIVE_SIX_ROLE_DECODE_ISSUANCE_TOKEN,
        )
        for name in (
            "design_hash",
            "joint_six_role_family_registration_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "snapshot_hash",
            "source_progress_receipt_hash",
            "transaction_chain_hash",
        ):
            _identity(getattr(self, name), name=name)
        branch_ids = tuple(
            _branch_id(value, name="label-native branch")
            for value in self.branch_ids
        )
        if branch_ids != CANONICAL_FQR_BRANCH_IDS:
            _refuse(
                V1FQRCheckpointRefusalCode.MISSING_BRANCH,
                "label-native six-role checkpoint must cover both canonical branches",
            )
        branch_hashes = tuple(
            _identity(value, name="label-native branch design hash")
            for value in self.branch_design_hashes
        )
        if len(branch_hashes) != len(branch_ids):
            _refuse(
                V1FQRCheckpointRefusalCode.MISSING_BRANCH,
                "label-native branch design hashes do not cover the branch axis",
            )
        generation = _exact_nonnegative_int(
            self.proposal_generation,
            name="label-native proposal generation",
        )
        route_count = _exact_positive_int(
            self.route_count,
            name="label-native route count",
        )
        shared = _exact_nonnegative_int(
            self.shared_record_count,
            name="label-native shared C record count",
        )
        if type(self.snapshot) is not V1LabelNativeSixRoleAccumulatorSnapshot:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "label-native checkpoint requires the exact six-role snapshot",
            )
        try:
            observed_snapshot_hash = self.snapshot.snapshot_hash
        except (TypeError, ValueError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"label-native six-role snapshot failed validation: {error}",
            ) from error
        if (
            observed_snapshot_hash != self.snapshot_hash
            or self.snapshot.design_hash != self.design_hash
            or tuple(value.value for value in self.snapshot.branches) != branch_ids
            or self.snapshot.shared_record_count != shared
            or self.snapshot.transaction_chain_hash != self.transaction_chain_hash
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "label-native checkpoint root differs from its six-role snapshot",
            )
        for name in (
            "retained_context_ids",
            "retained_secant_records",
            "retained_record_arrays",
            "retained_route_scale_arrays",
        ):
            object.__setattr__(
                self,
                name,
                _exact_nonnegative_int(getattr(self, name), name=name),
            )
        if (
            self.exact_integer_sufficient_statistics_only is not True
            or self.retained_context_ids != 0
            or self.retained_secant_records != 0
            or self.retained_record_arrays != 0
            or self.retained_route_scale_arrays != 0
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.RAW_RECORD_STATE_FORBIDDEN,
                "label-native checkpoint retained raw records, contexts, or route arrays",
            )
        if (
            self.exact_label_native_design_required_for_restore is not True
            or self.exact_expected_snapshot_hash_required_for_nonempty_restore is not True
            or any(
                getattr(self, name) is not False
                for name in (
                    "live_snapshot_reconstruction_authority",
                    "C_stream_resume_authority",
                    "calibration_K_open_authority",
                    "paired_oracle_O_open_authority",
                    "irreversible_decision_authority",
                    "mask_commit_authority",
                    "terminal_authority",
                    "pareto_claim_authority",
                )
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "label-native six-role checkpoint crossed its inert authority boundary",
            )
        if (
            self.authority_scope
            != FQR_LABEL_NATIVE_SIX_ROLE_INFLIGHT_C_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema != FQR_LABEL_NATIVE_SIX_ROLE_INFLIGHT_C_CHECKPOINT_SCHEMA
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "label-native six-role checkpoint schema or scope is invalid",
            )
        object.__setattr__(self, "branch_ids", branch_ids)
        object.__setattr__(self, "branch_design_hashes", branch_hashes)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "shared_record_count", shared)
        self.assert_integrity()

    @classmethod
    def from_live(
        cls,
        *,
        accumulator: V1LabelNativeSixRoleCAccumulator,
    ) -> "V1LabelNativeSixRoleInFlightCCheckpointBundle":
        if type(accumulator) is not V1LabelNativeSixRoleCAccumulator:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "label-native checkpoint capture requires the exact live six-role accumulator",
            )
        design = accumulator.design
        try:
            design_hash = design.assert_integrity()
            source_progress = accumulator.progress_receipt()
            source_progress.assert_replays(design)
            snapshot = accumulator.export_snapshot()
            snapshot_hash = snapshot.snapshot_hash
            replayed = V1LabelNativeSixRoleCAccumulator.from_snapshot(
                design,
                snapshot,
                expected_snapshot_hash=snapshot_hash,
            )
            if replayed.progress_receipt().receipt_hash != source_progress.receipt_hash:
                _refuse(
                    V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                    "label-native checkpoint snapshot does not replay its live progress receipt",
                )
            if accumulator.live_row_authenticated_prefix:
                if (
                    accumulator.require_live_row_authenticated_prefix()
                    != snapshot.transaction_chain_hash
                ):
                    _refuse(
                        V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                        "live label-native C transaction chain differs from its snapshot",
                    )
            else:
                _require_label_native_six_role_trusted_restore_lineage(
                    accumulator=accumulator,
                    design=design,
                    snapshot=snapshot,
                    source_progress_receipt_hash=source_progress.receipt_hash,
                )
        except V1LabelNativeSixRoleCRefusal as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                f"label-native six-role source graph failed replay: {error}",
            ) from error
        except (TypeError, ValueError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                f"label-native six-role source graph failed validation: {error}",
            ) from error
        return cls(
            design_hash=design_hash,
            branch_design_hashes=tuple(
                branch.design_hash for branch in design.branch_designs
            ),
            joint_six_role_family_registration_hash=(
                design.joint_six_role_family_registration_hash
            ),
            chart_identity_hash=design.chart_identity_hash,
            current_mask_hash=design.current_mask_hash,
            proposal_generation=design.proposal_generation,
            route_axis_hash=design.route_axis_hash,
            route_count=design.route_count,
            snapshot=snapshot,
            snapshot_hash=snapshot_hash,
            source_progress_receipt_hash=source_progress.receipt_hash,
            shared_record_count=snapshot.shared_record_count,
            transaction_chain_hash=snapshot.transaction_chain_hash,
            _issuance_token=_LABEL_NATIVE_SIX_ROLE_CAPTURE_ISSUANCE_TOKEN,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "branch_design_hashes": self.branch_design_hashes,
            "joint_six_role_family_registration_hash": (
                self.joint_six_role_family_registration_hash
            ),
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "snapshot_hash": self.snapshot_hash,
            "source_progress_receipt_hash": self.source_progress_receipt_hash,
            "shared_record_count": self.shared_record_count,
            "transaction_chain_hash": self.transaction_chain_hash,
            "branch_ids": self.branch_ids,
            "exact_integer_sufficient_statistics_only": (
                self.exact_integer_sufficient_statistics_only
            ),
            "retained_context_ids": self.retained_context_ids,
            "retained_secant_records": self.retained_secant_records,
            "retained_record_arrays": self.retained_record_arrays,
            "retained_route_scale_arrays": self.retained_route_scale_arrays,
            "exact_label_native_design_required_for_restore": (
                self.exact_label_native_design_required_for_restore
            ),
            "exact_expected_snapshot_hash_required_for_nonempty_restore": (
                self.exact_expected_snapshot_hash_required_for_nonempty_restore
            ),
            "live_snapshot_reconstruction_authority": (
                self.live_snapshot_reconstruction_authority
            ),
            "C_stream_resume_authority": self.C_stream_resume_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_claim_authority": self.pareto_claim_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def bundle_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        try:
            if self.snapshot.snapshot_hash != self.snapshot_hash:
                _refuse(
                    V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                    "label-native six-role snapshot was tampered",
                )
        except (TypeError, ValueError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"label-native six-role snapshot was tampered: {error}",
            ) from error
        observed = content_hash(
            self._payload(),
            prefix=_LABEL_NATIVE_SIX_ROLE_INFLIGHT_C_BUNDLE_PREFIX,
        )
        if self._bundle_hash is not None and self._bundle_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "label-native six-role checkpoint bundle was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "label-native six-role checkpoint bundle hash does not match",
            )
        object.__setattr__(self, "_bundle_hash", observed)
        return observed

    def snapshot_for_restore(
        self,
        *,
        design: V1LabelNativeSixRoleCDesign,
        expected_snapshot_hash: str | None = None,
    ) -> V1LabelNativeSixRoleAccumulatorSnapshot:
        if self.shared_record_count > 0 and expected_snapshot_hash is None:
            _refuse(
                V1FQRCheckpointRefusalCode.SNAPSHOT_HASH_REQUIRED,
                "nonempty label-native six-role restore requires an exact expected snapshot hash",
            )
        if type(design) is not V1LabelNativeSixRoleCDesign:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "label-native restore requires the exact six-role design",
            )
        try:
            observed_design_hash = design.assert_integrity()
            observed_snapshot_hash = self.snapshot.snapshot_hash
            if (
                expected_snapshot_hash is not None
                and observed_snapshot_hash != expected_snapshot_hash
            ):
                _refuse(
                    V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                    "label-native six-role expected snapshot hash does not match",
                )
            replayed = V1LabelNativeSixRoleCAccumulator.from_snapshot(
                design,
                self.snapshot,
                expected_snapshot_hash=observed_snapshot_hash,
            )
            if (
                replayed.progress_receipt().receipt_hash
                != self.source_progress_receipt_hash
            ):
                _refuse(
                    V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                    "label-native checkpoint snapshot does not replay its "
                    "captured progress receipt",
                )
        except V1FQRCheckpointRefusal:
            raise
        except V1LabelNativeSixRoleCRefusal as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                f"label-native six-role restore crossed its design boundary: {error}",
            ) from error
        except (TypeError, ValueError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"label-native six-role expected snapshot hash does not match: {error}",
            ) from error
        if (
            observed_design_hash != self.design_hash
            or tuple(branch.design_hash for branch in design.branch_designs)
            != self.branch_design_hashes
            or design.joint_six_role_family_registration_hash
            != self.joint_six_role_family_registration_hash
            or design.chart_identity_hash != self.chart_identity_hash
            or design.current_mask_hash != self.current_mask_hash
            or design.proposal_generation != self.proposal_generation
            or design.route_axis_hash != self.route_axis_hash
            or design.route_count != self.route_count
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "label-native checkpoint belongs to another six-role design",
            )
        return self.snapshot

    def restore_authenticated_accumulator(
        self,
        *,
        design: V1LabelNativeSixRoleCDesign,
        expected_bundle_hash: str,
        expected_snapshot_hash: str,
    ) -> V1LabelNativeSixRoleCAccumulator:
        """Restore one resumable C accumulator through the strict decode seam.

        A public ``V1LabelNativeSixRoleCAccumulator.from_snapshot`` result is
        intentionally never resumable checkpoint authority.  This method is
        narrower: the root must have been issued by the strict framed decoder,
        the caller must bind both manifest hashes, and the snapshot must replay
        the exact source-progress receipt under the exact design.  The returned
        accumulator is registered process-locally; subsequent rows still pass
        the normal live row/source checks before a new checkpoint can be made.
        """

        if self._decoded_for_trusted_restore is not True:
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "trusted label-native C restore requires a strictly decoded "
                "checkpoint root",
            )
        expected_bundle = _identity(
            expected_bundle_hash,
            name="expected label-native checkpoint bundle hash",
        )
        expected_snapshot = _identity(
            expected_snapshot_hash,
            name="expected label-native checkpoint snapshot hash",
        )
        observed_bundle_hash = self.assert_integrity(expected_bundle)
        snapshot = self.snapshot_for_restore(
            design=design,
            expected_snapshot_hash=expected_snapshot,
        )
        try:
            accumulator = V1LabelNativeSixRoleCAccumulator.from_snapshot(
                design,
                snapshot,
                expected_snapshot_hash=expected_snapshot,
            )
            progress = accumulator.progress_receipt()
            progress.assert_replays(design)
        except V1LabelNativeSixRoleCRefusal as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                f"trusted label-native C restore failed source replay: {error}",
            ) from error
        except (TypeError, ValueError) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                f"trusted label-native C restore failed validation: {error}",
            ) from error
        if (
            progress.receipt_hash != self.source_progress_receipt_hash
            or snapshot.snapshot_hash != self.snapshot_hash
            or snapshot.transaction_chain_hash != self.transaction_chain_hash
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "trusted label-native C restore does not replay the captured "
                "progress/snapshot lineage",
            )
        _register_label_native_six_role_trusted_restore(
            accumulator=accumulator,
            lineage=_V1LabelNativeSixRoleTrustedRestoreLineage(
                source_bundle_hash=observed_bundle_hash,
                design_hash=self.design_hash,
                branch_design_hashes=self.branch_design_hashes,
                joint_six_role_family_registration_hash=(
                    self.joint_six_role_family_registration_hash
                ),
                source_snapshot_hash=self.snapshot_hash,
                source_progress_receipt_hash=self.source_progress_receipt_hash,
                source_shared_record_count=self.shared_record_count,
                source_transaction_chain_hash=self.transaction_chain_hash,
            ),
        )
        return accumulator


@dataclass(frozen=True, slots=True)
class V1FQRCertificationReceiptCheckpoint:
    """Compact projection of a raw-record-bearing terminal F/Q/R receipt."""

    source_receipt_hash: str
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
    remainder_receipt_hash: str
    remainder_record_axis_hash: str
    remainder_statistical_proof_hash: str
    remainder_certification_stream_commitment_hash: str
    remainder_intervals: tuple[V1DirectionalRemainderInterval, ...]
    omitted_secant_record_count: int
    endpoint_order_unchanged_after_outcomes: bool = True
    postselection_occurred: bool = False
    source_proposal_authority: str = FQR_PROPOSAL_AUTHORITY
    raw_secant_records_serialized: bool = False
    live_receipt_reconstruction_authority: bool = False
    proposal_queue_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQR_CHECKPOINT_AUTHORITY_SCOPE
    schema: str = FQR_CHECKPOINT_RECEIPT_PROJECTION_SCHEMA
    _projection_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "source_receipt_hash",
            "design_hash",
            "secant_design_hash",
            "ladder_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "joint_fqr_family_registration_hash",
            "joint_fqr_family_receipt_hash",
            "remainder_receipt_hash",
            "remainder_record_axis_hash",
            "remainder_statistical_proof_hash",
            "remainder_certification_stream_commitment_hash",
        ):
            _identity(getattr(self, name), name=name)
        generation = _exact_nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        endpoints = tuple(
            _identity(value, name="endpoint action hash")
            for value in self.endpoint_action_hashes
        )
        if (
            not endpoints
            or len(endpoints) > _MAX_ENDPOINTS_PER_BRANCH
            or len(set(endpoints)) != len(endpoints)
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
                "checkpoint endpoint axis is empty, duplicate, or too large",
            )
        bounds = tuple(self.fq_bounds)
        if any(type(value) is not V1FQEndpointBound for value in bounds):
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "receipt projection requires exact F/Q bounds",
            )
        if tuple((x.endpoint_index, x.action_hash) for x in bounds) != tuple(
            enumerate(endpoints, start=1)
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.POST_SELECTION,
                "F/Q bounds reordered the frozen endpoint axis",
            )
        intervals = tuple(self.remainder_intervals)
        if any(
            type(value) is not V1DirectionalRemainderInterval for value in intervals
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "receipt projection requires exact remainder intervals",
            )
        expected_cells = tuple(
            (index, action, dose)
            for index, action in enumerate(endpoints, start=1)
            for dose in DIRECTIONAL_SECANT_DOSES
        )
        if (
            tuple(
                (value.endpoint_index, value.action_hash, value.dose)
                for value in intervals
            )
            != expected_cells
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.POST_SELECTION,
                "remainder intervals reordered or omitted a frozen endpoint/dose",
            )
        omitted = _exact_nonnegative_int(
            self.omitted_secant_record_count,
            name="omitted secant record count",
        )
        if omitted <= 0 or any(
            value.sample_size != omitted
            or value.record_axis_hash != self.remainder_record_axis_hash
            for value in intervals
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "remainder intervals differ from the omitted record axis",
            )
        if (
            self.endpoint_order_unchanged_after_outcomes is not True
            or self.postselection_occurred is not False
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.POST_SELECTION,
                "terminal F/Q/R evidence was post-selected",
            )
        if self.source_proposal_authority != FQR_PROPOSAL_AUTHORITY:
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "source receipt exceeded proposal-screening authority",
            )
        if self.raw_secant_records_serialized is not False:
            _refuse(
                V1FQRCheckpointRefusalCode.RAW_RECORD_STATE_FORBIDDEN,
                "terminal secant records cannot enter the checkpoint",
            )
        if any(
            getattr(self, name) is not False
            for name in (
                "live_receipt_reconstruction_authority",
                "proposal_queue_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "receipt projection cannot reconstruct live action authority",
            )
        if (
            self.authority_scope != FQR_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema != FQR_CHECKPOINT_RECEIPT_PROJECTION_SCHEMA
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "receipt projection schema or authority is invalid",
            )
        for value in (*bounds, *intervals):
            value.assert_integrity()
        live_payload = {
            "schema": FQR_CERTIFICATION_RECEIPT_SCHEMA,
            "design_hash": self.design_hash,
            "secant_design_hash": self.secant_design_hash,
            "ladder_hash": self.ladder_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": generation,
            "endpoint_action_hashes": endpoints,
            "joint_fqr_family_registration_hash": (
                self.joint_fqr_family_registration_hash
            ),
            "joint_fqr_family_receipt_hash": self.joint_fqr_family_receipt_hash,
            "fq_bound_hashes": tuple(value.bound_hash for value in bounds),
            "remainder_receipt_hash": self.remainder_receipt_hash,
            "endpoint_order_unchanged_after_outcomes": True,
            "postselection_occurred": False,
            "proposal_authority": FQR_PROPOSAL_AUTHORITY,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
            "terminal_authority": False,
        }
        if content_hash(live_payload, prefix=_LIVE_FQR_RECEIPT_PREFIX) != (
            self.source_receipt_hash
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "compact projection does not replay the outer live receipt hash",
            )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "endpoint_action_hashes", endpoints)
        object.__setattr__(self, "fq_bounds", bounds)
        object.__setattr__(self, "remainder_intervals", intervals)
        object.__setattr__(self, "omitted_secant_record_count", omitted)
        self.assert_integrity()

    @classmethod
    def from_live(
        cls,
        *,
        design: V1FQRCertificationDesign,
        receipt: V1FQRCertificationReceipt,
    ) -> "V1FQRCertificationReceiptCheckpoint":
        if (
            type(design) is not V1FQRCertificationDesign
            or type(receipt) is not V1FQRCertificationReceipt
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "receipt projection requires exact live F/Q/R types",
            )
        verify_v1_fqr_certification_receipt(design=design, receipt=receipt)
        remainder = receipt.remainder_receipt
        return cls(
            source_receipt_hash=receipt.receipt_hash,
            design_hash=receipt.design_hash,
            secant_design_hash=receipt.secant_design_hash,
            ladder_hash=receipt.ladder_hash,
            chart_identity_hash=receipt.chart_identity_hash,
            current_mask_hash=receipt.current_mask_hash,
            proposal_generation=receipt.proposal_generation,
            endpoint_action_hashes=receipt.endpoint_action_hashes,
            joint_fqr_family_registration_hash=(
                receipt.joint_fqr_family_registration_hash
            ),
            joint_fqr_family_receipt_hash=receipt.joint_fqr_family_receipt_hash,
            fq_bounds=receipt.fq_bounds,
            remainder_receipt_hash=remainder.receipt_hash,
            remainder_record_axis_hash=remainder.record_axis_hash,
            remainder_statistical_proof_hash=remainder.statistical_proof_hash,
            remainder_certification_stream_commitment_hash=(
                remainder.certification_stream_commitment_hash
            ),
            remainder_intervals=remainder.endpoint_intervals,
            omitted_secant_record_count=len(remainder.records),
        )

    @property
    def remainder_upper_by_endpoint_and_dose(self) -> tuple[tuple[float, float], ...]:
        values = tuple(value.upper for value in self.remainder_intervals)
        return tuple(
            (values[index], values[index + 1]) for index in range(0, len(values), 2)
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_receipt_hash": self.source_receipt_hash,
            "design_hash": self.design_hash,
            "secant_design_hash": self.secant_design_hash,
            "ladder_hash": self.ladder_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "current_mask_hash": self.current_mask_hash,
            "proposal_generation": self.proposal_generation,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "joint_fqr_family_registration_hash": (
                self.joint_fqr_family_registration_hash
            ),
            "joint_fqr_family_receipt_hash": self.joint_fqr_family_receipt_hash,
            "fq_bound_hashes": tuple(value.bound_hash for value in self.fq_bounds),
            "remainder_receipt_hash": self.remainder_receipt_hash,
            "remainder_record_axis_hash": self.remainder_record_axis_hash,
            "remainder_statistical_proof_hash": self.remainder_statistical_proof_hash,
            "remainder_certification_stream_commitment_hash": (
                self.remainder_certification_stream_commitment_hash
            ),
            "remainder_interval_hashes": tuple(
                value.interval_hash for value in self.remainder_intervals
            ),
            "omitted_secant_record_count": self.omitted_secant_record_count,
            "endpoint_order_unchanged_after_outcomes": (
                self.endpoint_order_unchanged_after_outcomes
            ),
            "postselection_occurred": self.postselection_occurred,
            "source_proposal_authority": self.source_proposal_authority,
            "raw_secant_records_serialized": self.raw_secant_records_serialized,
            "live_receipt_reconstruction_authority": (
                self.live_receipt_reconstruction_authority
            ),
            "proposal_queue_authority": self.proposal_queue_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def projection_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for value in (*self.fq_bounds, *self.remainder_intervals):
            value.assert_integrity()
        observed = content_hash(self._payload(), prefix=_RECEIPT_PROJECTION_PREFIX)
        if self._projection_hash is not None and self._projection_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "F/Q/R receipt projection was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "F/Q/R receipt projection hash does not match",
            )
        object.__setattr__(self, "_projection_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRCandidateQueueCheckpoint:
    """Non-authoritative replay projection of a live proposal-only queue."""

    source_authorization_hash: str
    design_hash: str
    receipt_hash: str
    decision_hash: str
    stability_certificate_hash: str
    ladder_hash: str
    current_mask_hash: str
    chart_identity_hash: str
    proposal_generation: int
    certified_endpoint_indices: tuple[int, ...]
    certified_action_hashes: tuple[str, ...]
    queue_entries: tuple[V1FQRCandidateQueueEntry, ...]
    source_fqr_proposal_screening_authority: bool = True
    source_proposal_queue_materialization_authority: bool = True
    source_frozen_ladder_order_preserved: bool = True
    source_fqr_monotonicity_assumed: bool = False
    source_paired_bce_oracle_still_required: bool = True
    source_cross_branch_completeness_authority: bool = False
    source_prefrozen_multiladder_family_required: bool = True
    source_ladder_family_scope: str = FQR_SINGLE_LADDER_SCOPE
    source_queue_order_policy: str = FQR_QUEUE_ORDER_POLICY
    source_authority_scope: str = FQR_CANDIDATE_QUEUE_AUTHORITY_SCOPE
    live_queue_reconstruction_authority: bool = False
    proposal_queue_materialization_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_validation_access_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQR_CHECKPOINT_AUTHORITY_SCOPE
    schema: str = FQR_CHECKPOINT_QUEUE_PROJECTION_SCHEMA
    _projection_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "source_authorization_hash",
            "design_hash",
            "receipt_hash",
            "decision_hash",
            "stability_certificate_hash",
            "ladder_hash",
            "current_mask_hash",
            "chart_identity_hash",
        ):
            _identity(getattr(self, name), name=name)
        generation = _exact_nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        indices = tuple(
            _exact_nonnegative_int(value, name="certified endpoint index")
            for value in self.certified_endpoint_indices
        )
        hashes = tuple(
            _identity(value, name="certified action hash")
            for value in self.certified_action_hashes
        )
        entries = tuple(self.queue_entries)
        if (
            not indices
            or any(value <= 0 for value in indices)
            or indices != tuple(sorted(set(indices)))
            or len(indices) != len(hashes)
            or any(type(value) is not V1FQRCandidateQueueEntry for value in entries)
            or tuple(value.endpoint_index for value in entries)
            != tuple(reversed(indices))
            or tuple(value.frozen_action_hash for value in reversed(entries)) != hashes
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.POST_SELECTION,
                "queue projection differs from the frozen certified subset",
            )
        for value in entries:
            value.assert_integrity()
        source_flags = (
            self.source_fqr_proposal_screening_authority,
            self.source_proposal_queue_materialization_authority,
            self.source_frozen_ladder_order_preserved,
            not self.source_fqr_monotonicity_assumed,
            self.source_paired_bce_oracle_still_required,
            not self.source_cross_branch_completeness_authority,
            self.source_prefrozen_multiladder_family_required,
        )
        if any(type(value) is not bool or not value for value in source_flags):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "source queue projection crossed its registered proposal boundary",
            )
        if (
            self.source_ladder_family_scope != FQR_SINGLE_LADDER_SCOPE
            or self.source_queue_order_policy != FQR_QUEUE_ORDER_POLICY
            or self.source_authority_scope != FQR_CANDIDATE_QUEUE_AUTHORITY_SCOPE
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "source queue scope or order policy changed",
            )
        if any(
            getattr(self, name) is not False
            for name in (
                "live_queue_reconstruction_authority",
                "proposal_queue_materialization_authority",
                "calibration_authority",
                "bce_oracle_authority",
                "irreversible_decision_authority",
                "mask_commit_authority",
                "terminal_validation_access_authority",
                "terminal_authority",
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "decoded queue projections grant no live authority",
            )
        if (
            self.authority_scope != FQR_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema != FQR_CHECKPOINT_QUEUE_PROJECTION_SCHEMA
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "queue projection schema or authority is invalid",
            )
        live_payload = {
            "schema": FQR_CANDIDATE_QUEUE_AUTHORIZATION_SCHEMA,
            "design_hash": self.design_hash,
            "receipt_hash": self.receipt_hash,
            "decision_hash": self.decision_hash,
            "stability_certificate_hash": self.stability_certificate_hash,
            "ladder_hash": self.ladder_hash,
            "current_mask_hash": self.current_mask_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "proposal_generation": generation,
            "certified_endpoint_indices": indices,
            "certified_action_hashes": hashes,
            "queue_entry_hashes": tuple(value.entry_hash for value in entries),
            "fqr_proposal_screening_authority": True,
            "proposal_queue_materialization_authority": True,
            "frozen_ladder_order_preserved": True,
            "fqr_monotonicity_assumed": False,
            "candidate_freeze_authority": False,
            "calibration_authority": False,
            "bce_oracle_authority": False,
            "irreversible_decision_authority": False,
            "mask_commit_authority": False,
            "terminal_validation_access_authority": False,
            "terminal_authority": False,
            "paired_bce_oracle_still_required": True,
            "cross_branch_completeness_authority": False,
            "prefrozen_multiladder_family_required_for_cross_branch_selection": True,
            "ladder_family_scope": FQR_SINGLE_LADDER_SCOPE,
            "queue_order_policy": FQR_QUEUE_ORDER_POLICY,
            "authority_scope": FQR_CANDIDATE_QUEUE_AUTHORITY_SCOPE,
        }
        if content_hash(live_payload, prefix=_LIVE_QUEUE_AUTHORIZATION_PREFIX) != (
            self.source_authorization_hash
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "queue projection does not replay its live authorization hash",
            )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "certified_endpoint_indices", indices)
        object.__setattr__(self, "certified_action_hashes", hashes)
        object.__setattr__(self, "queue_entries", entries)
        self.assert_integrity()

    @classmethod
    def from_live(
        cls,
        authorization: V1FQRCandidateQueueAuthorization,
    ) -> "V1FQRCandidateQueueCheckpoint":
        if type(authorization) is not V1FQRCandidateQueueAuthorization:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "queue projection requires the exact live authorization type",
            )
        authorization.assert_integrity()
        return cls(
            source_authorization_hash=authorization.authorization_hash,
            design_hash=authorization.design.design_hash,
            receipt_hash=authorization.receipt.receipt_hash,
            decision_hash=authorization.decision.decision_hash,
            stability_certificate_hash=(
                authorization.stability_certificate.certificate_hash
            ),
            ladder_hash=authorization.ladder.ladder_hash,
            current_mask_hash=_direct_mask_hash(authorization.current_mask),
            chart_identity_hash=authorization.chart_identity_hash,
            proposal_generation=authorization.proposal_generation,
            certified_endpoint_indices=authorization.certified_endpoint_indices,
            certified_action_hashes=authorization.certified_action_hashes,
            queue_entries=authorization.queue_entries,
        )

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "source_authorization_hash",
                "design_hash",
                "receipt_hash",
                "decision_hash",
                "stability_certificate_hash",
                "ladder_hash",
                "current_mask_hash",
                "chart_identity_hash",
                "proposal_generation",
                "certified_endpoint_indices",
                "certified_action_hashes",
                "source_fqr_proposal_screening_authority",
                "source_proposal_queue_materialization_authority",
                "source_frozen_ladder_order_preserved",
                "source_fqr_monotonicity_assumed",
                "source_paired_bce_oracle_still_required",
                "source_cross_branch_completeness_authority",
                "source_prefrozen_multiladder_family_required",
                "source_ladder_family_scope",
                "source_queue_order_policy",
                "source_authority_scope",
                "live_queue_reconstruction_authority",
                "proposal_queue_materialization_authority",
                "calibration_authority",
                "bce_oracle_authority",
                "irreversible_decision_authority",
                "mask_commit_authority",
                "terminal_validation_access_authority",
                "terminal_authority",
                "authority_scope",
            )
        } | {
            "queue_entry_hashes": tuple(
                value.entry_hash for value in self.queue_entries
            )
        }

    @property
    def projection_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for value in self.queue_entries:
            value.assert_integrity()
        observed = content_hash(self._payload(), prefix=_QUEUE_PROJECTION_PREFIX)
        if self._projection_hash is not None and self._projection_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "queue projection was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "queue projection hash does not match",
            )
        object.__setattr__(self, "_projection_hash", observed)
        return observed


def _replay_compact_decision(
    *,
    design: V1FQRCertificationDesign,
    receipt: V1FQRCertificationReceiptCheckpoint,
    decision: V1FQRCertificationDecision,
) -> None:
    remainder = receipt.remainder_upper_by_endpoint_and_dose
    outcomes: list[tuple[object, ...]] = []
    receipt_empty = any(value.empty for value in receipt.fq_bounds) or any(
        value.empty for value in receipt.remainder_intervals
    )
    for bound, (r_half, r_one) in zip(receipt.fq_bounds, remainder, strict=True):
        half_lhs_decimal = _response_lhs_decimal(
            eta=0.5, q_upper=bound.q_upper, r_upper=r_half
        )
        one_lhs_decimal = _response_lhs_decimal(
            eta=1.0, q_upper=bound.q_upper, r_upper=r_one
        )
        half_lhs = _outward_decimal(half_lhs_decimal)
        one_lhs = _outward_decimal(one_lhs_decimal)
        f_ok = Decimal.from_float(bound.f_upper) <= Decimal.from_float(design.tau_f)
        half_ok = half_lhs_decimal <= (
            Decimal.from_float(0.5) * Decimal.from_float(design.tau_l)
        )
        one_ok = one_lhs_decimal <= (
            Decimal.from_float(1.0) * Decimal.from_float(design.tau_l)
        )
        certified = bool(
            not bound.empty and not receipt_empty and f_ok and half_ok and one_ok
        )
        outcomes.append(
            (
                bound.endpoint_index,
                bound.action_hash,
                bound.f_upper,
                bound.q_upper,
                r_half,
                r_one,
                half_lhs,
                one_lhs,
                bool(f_ok),
                bool(half_ok),
                bool(one_ok),
                certified,
            )
        )
    if tuple(outcomes) != decision.endpoint_outcomes:
        _refuse(
            V1FQRCheckpointRefusalCode.HASH_MISMATCH,
            "compact F/Q/R values do not replay the supplied decision",
        )
    certified_rows = tuple(value for value in outcomes if value[-1])
    if receipt_empty:
        expected = ("refuse_inconsistent_simultaneous_bounds", None, None)
    elif certified_rows:
        expected = (
            "certified_largest_endpoint",
            certified_rows[-1][0],
            certified_rows[-1][1],
        )
    else:
        expected = ("refuse_no_certified_endpoint", None, None)
    if (
        decision.status,
        decision.selected_endpoint_index,
        decision.selected_action_hash,
    ) != expected:
        _refuse(
            V1FQRCheckpointRefusalCode.HASH_MISMATCH,
            "compact F/Q/R decision status does not replay",
        )


@dataclass(frozen=True, slots=True)
class V1FQRCheckpointBranch:
    """One independently solved and pre-frozen residual-envelope branch."""

    branch_id: str
    frozen_ladder: V1FrozenCandidateLadder = field(repr=False, compare=False)
    secant_design: V1DirectionalSecantDesign = field(repr=False, compare=False)
    fqr_design: V1FQRCertificationDesign = field(repr=False, compare=False)
    sequential_design: V1FQRSequentialAcquisitionDesign = field(
        repr=False, compare=False
    )
    sufficient_state: V1FQRSequentialAccumulatorSnapshot = field(
        repr=False, compare=False
    )
    sequential_receipt: V1FQRSequentialAcquisitionReceipt = field(
        repr=False, compare=False
    )
    deterministic_uncertainty_receipt: V1FQRDeterministicUncertaintyReceipt = field(
        repr=False, compare=False
    )
    readiness_receipt: V1FQRDecisionReadinessReceipt = field(repr=False, compare=False)
    certification_receipt_checkpoint: V1FQRCertificationReceiptCheckpoint | None = (
        field(default=None, repr=False, compare=False)
    )
    decision: V1FQRCertificationDecision | None = field(
        default=None, repr=False, compare=False
    )
    stability_certificate: V1DecisionStabilityCertificate | None = field(
        default=None, repr=False, compare=False
    )
    queue_checkpoint: V1FQRCandidateQueueCheckpoint | None = field(
        default=None, repr=False, compare=False
    )
    branch_frozen_before_C_outcomes: bool = True
    endpoint_order_mutable_after_C_outcomes: bool = False
    validation_outcomes_seen: bool = False
    live_receipt_authority: bool = False
    proposal_queue_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQR_CHECKPOINT_AUTHORITY_SCOPE
    schema: str = FQR_CHECKPOINT_BRANCH_SCHEMA
    _branch_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        branch = _identity(self.branch_id, name="F/Q/R residual branch")
        exact_types = (
            (self.frozen_ladder, V1FrozenCandidateLadder),
            (self.secant_design, V1DirectionalSecantDesign),
            (self.fqr_design, V1FQRCertificationDesign),
            (self.sequential_design, V1FQRSequentialAcquisitionDesign),
            (self.sufficient_state, V1FQRSequentialAccumulatorSnapshot),
            (self.sequential_receipt, V1FQRSequentialAcquisitionReceipt),
            (
                self.deterministic_uncertainty_receipt,
                V1FQRDeterministicUncertaintyReceipt,
            ),
            (self.readiness_receipt, V1FQRDecisionReadinessReceipt),
        )
        if any(type(value) is not expected for value, expected in exact_types):
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "checkpoint branch contains a noncanonical typed component",
            )
        ladder_hash = self.frozen_ladder.assert_integrity()
        secant_hash = self.secant_design.assert_integrity()
        design_hash = self.fqr_design.assert_integrity()
        sequential_design_hash = self.sequential_design.assert_integrity()
        self.sufficient_state.assert_integrity()
        sequential_receipt_hash = self.sequential_receipt.assert_integrity()
        deterministic_hash = self.deterministic_uncertainty_receipt.assert_integrity()
        self.readiness_receipt.assert_integrity()
        if (
            self.secant_design.ladder_hash != ladder_hash
            or self.fqr_design.secant_design.design_hash != secant_hash
            or self.sequential_design.fqr_design.design_hash != design_hash
            or self.sufficient_state.acquisition_design_hash != sequential_design_hash
            or self.sequential_receipt.design_hash != sequential_design_hash
            or self.deterministic_uncertainty_receipt.acquisition_design_hash
            != sequential_design_hash
            or self.readiness_receipt.acquisition_receipt_hash
            != sequential_receipt_hash
            or self.readiness_receipt.deterministic_uncertainty_receipt_hash
            != deterministic_hash
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "checkpoint branch components name different designs or receipts",
            )
        restored_accumulator = V1FQRSequentialAccumulator.from_snapshot(
            self.sequential_design,
            self.sufficient_state,
            expected_snapshot_hash=self.sufficient_state.snapshot_hash,
        )
        if restored_accumulator.receipt().receipt_hash != sequential_receipt_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "public accumulator snapshot does not replay its bounded receipt",
            )
        final_values = (
            self.certification_receipt_checkpoint,
            self.decision,
            self.stability_certificate,
            self.queue_checkpoint,
        )
        if any(value is None for value in final_values) and any(
            value is not None for value in final_values
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.MISSING_BRANCH,
                "completed F/Q/R graph must checkpoint receipt, decision, "
                "stability, and queue together",
            )
        if all(value is not None for value in final_values):
            receipt = cast(V1FQRCertificationReceiptCheckpoint, final_values[0])
            decision = cast(V1FQRCertificationDecision, final_values[1])
            stability = cast(V1DecisionStabilityCertificate, final_values[2])
            queue = cast(V1FQRCandidateQueueCheckpoint, final_values[3])
            if (
                type(receipt) is not V1FQRCertificationReceiptCheckpoint
                or type(decision) is not V1FQRCertificationDecision
                or type(stability) is not V1DecisionStabilityCertificate
                or type(queue) is not V1FQRCandidateQueueCheckpoint
            ):
                _refuse(
                    V1FQRCheckpointRefusalCode.INVALID_TYPE,
                    "completed branch graph contains a noncanonical type",
                )
            receipt.assert_integrity()
            decision_hash = decision.assert_integrity()
            stability.assert_integrity()
            queue.assert_integrity()
            if (
                receipt.design_hash != design_hash
                or receipt.secant_design_hash != secant_hash
                or receipt.ladder_hash != ladder_hash
                or receipt.chart_identity_hash != self.fqr_design.chart_identity_hash
                or receipt.current_mask_hash != self.frozen_ladder.current_mask_hash
                or receipt.proposal_generation != self.frozen_ladder.proposal_generation
                or decision.receipt_hash != receipt.source_receipt_hash
                or decision.design_hash != design_hash
                or decision.decision_hash != stability.decision_hash
                or stability.receipt_hash != receipt.source_receipt_hash
                or stability.certificate_hash != queue.stability_certificate_hash
                or queue.receipt_hash != receipt.source_receipt_hash
                or queue.decision_hash != decision_hash
                or queue.ladder_hash != ladder_hash
                or queue.chart_identity_hash != self.fqr_design.chart_identity_hash
                or queue.current_mask_hash != self.frozen_ladder.current_mask_hash
                or queue.proposal_generation != self.frozen_ladder.proposal_generation
                or self.sequential_receipt.sampling_decision
                != "sampling_stable_endpoint"
                or self.sequential_receipt.selected_endpoint_index
                != decision.selected_endpoint_index
                or self.sequential_receipt.selected_action_hash
                != decision.selected_action_hash
            ):
                _refuse(
                    V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                    "completed branch hashes or causal coordinates differ",
                )
            _replay_compact_decision(
                design=self.fqr_design,
                receipt=receipt,
                decision=decision,
            )
        if (
            self.branch_frozen_before_C_outcomes is not True
            or self.endpoint_order_mutable_after_C_outcomes is not False
            or self.validation_outcomes_seen is not False
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.POST_SELECTION,
                "checkpoint branch was selected or reordered after C outcomes",
            )
        if any(
            getattr(self, name) is not False
            for name in (
                "live_receipt_authority",
                "proposal_queue_authority",
                "calibration_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "checkpoint branch grants no live F/Q/R or action authority",
            )
        if (
            self.authority_scope != FQR_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema != FQR_CHECKPOINT_BRANCH_SCHEMA
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "checkpoint branch schema or authority is invalid",
            )
        object.__setattr__(self, "branch_id", branch)
        self.assert_integrity()

    @classmethod
    def from_live(
        cls,
        *,
        branch_id: str,
        accumulator: V1FQRSequentialAccumulator,
        deterministic_uncertainty_receipt: V1FQRDeterministicUncertaintyReceipt,
        readiness_receipt: V1FQRDecisionReadinessReceipt,
        certification_receipt: V1FQRCertificationReceipt | None = None,
        decision: V1FQRCertificationDecision | None = None,
        stability_certificate: V1DecisionStabilityCertificate | None = None,
        authorization: V1FQRCandidateQueueAuthorization | None = None,
    ) -> "V1FQRCheckpointBranch":
        if type(accumulator) is not V1FQRSequentialAccumulator:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "branch capture requires the exact sequential accumulator",
            )
        design = accumulator.design
        return cls(
            branch_id=branch_id,
            frozen_ladder=design.fqr_design.secant_design.ladder,
            secant_design=design.fqr_design.secant_design,
            fqr_design=design.fqr_design,
            sequential_design=design,
            sufficient_state=accumulator.export_snapshot(),
            sequential_receipt=accumulator.receipt(),
            deterministic_uncertainty_receipt=deterministic_uncertainty_receipt,
            readiness_receipt=readiness_receipt,
            certification_receipt_checkpoint=(
                None
                if certification_receipt is None
                else V1FQRCertificationReceiptCheckpoint.from_live(
                    design=design.fqr_design,
                    receipt=certification_receipt,
                )
            ),
            decision=decision,
            stability_certificate=stability_certificate,
            queue_checkpoint=(
                None
                if authorization is None
                else V1FQRCandidateQueueCheckpoint.from_live(authorization)
            ),
        )

    @property
    def branch_hash(self) -> str:
        return self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "branch_id": self.branch_id,
            "ladder_hash": self.frozen_ladder.ladder_hash,
            "secant_design_hash": self.secant_design.design_hash,
            "fqr_design_hash": self.fqr_design.design_hash,
            "sequential_design_hash": self.sequential_design.design_hash,
            "sufficient_state_hash": self.sufficient_state.snapshot_hash,
            "sequential_receipt_hash": self.sequential_receipt.receipt_hash,
            "deterministic_uncertainty_receipt_hash": (
                self.deterministic_uncertainty_receipt.receipt_hash
            ),
            "readiness_receipt_hash": self.readiness_receipt.receipt_hash,
            "certification_receipt_projection_hash": (
                None
                if self.certification_receipt_checkpoint is None
                else self.certification_receipt_checkpoint.projection_hash
            ),
            "source_certification_receipt_hash": (
                None
                if self.certification_receipt_checkpoint is None
                else self.certification_receipt_checkpoint.source_receipt_hash
            ),
            "decision_hash": (
                None if self.decision is None else self.decision.decision_hash
            ),
            "stability_certificate_hash": (
                None
                if self.stability_certificate is None
                else self.stability_certificate.certificate_hash
            ),
            "queue_projection_hash": (
                None
                if self.queue_checkpoint is None
                else self.queue_checkpoint.projection_hash
            ),
            "source_queue_authorization_hash": (
                None
                if self.queue_checkpoint is None
                else self.queue_checkpoint.source_authorization_hash
            ),
            "branch_frozen_before_C_outcomes": self.branch_frozen_before_C_outcomes,
            "endpoint_order_mutable_after_C_outcomes": (
                self.endpoint_order_mutable_after_C_outcomes
            ),
            "validation_outcomes_seen": self.validation_outcomes_seen,
            "live_receipt_authority": self.live_receipt_authority,
            "proposal_queue_authority": self.proposal_queue_authority,
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "authority_scope": self.authority_scope,
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_BRANCH_PREFIX)
        if self._branch_hash is not None and self._branch_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "F/Q/R checkpoint branch was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "F/Q/R checkpoint branch hash does not match",
            )
        object.__setattr__(self, "_branch_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankReplayCheckpoint:
    """Inert, bounded projection of one exact raw-D rank replay.

    The live source-derived certificate contains issuer-only objects and is
    intentionally not reconstructible from checkpoint bytes.  This projection
    is sufficient to verify contiguous-rank and identical-D-prefix restart
    coordinates; reopening C still requires fresh live source verification.
    """

    policy_hash: str
    rank: int
    record_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    cumulative_stratum_counts: tuple[int, ...]
    D_context_axis_hash: str
    D_draw_axis_hash: str
    D_source_measurement_axis_hash: str
    D_weight_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    sampled_geometry_source_hash: str
    kkt_problem_hash: str
    kkt_result_hash: str
    provisional_branch_family_hash: str
    decision_certificate_hash: str
    branch_decision_hashes: tuple[str, ...]
    branch_source_derivation_receipt_hashes: tuple[str, ...]
    branch_ambiguity_code_axes: tuple[tuple[str, ...], ...]
    rank_local_residual_decision_stable: bool
    robust_branch_family_hash: str | None
    C_stream_open_authority: bool = False
    schema: str = FQR_ADAPTIVE_RANK_REPLAY_PROJECTION_SCHEMA
    _projection_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    @classmethod
    def from_live(
        cls,
        record: V1FQRAdaptiveRankReplayRecord,
    ) -> "V1FQRAdaptiveRankReplayCheckpoint":
        if type(record) is not V1FQRAdaptiveRankReplayRecord:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "adaptive-rank replay projection requires an exact live record",
            )
        record.assert_integrity()
        decision = record.decision_certificate
        source_hashes = tuple(
            value.authoritative_typed_source_derivation_receipt_hash
            for value in decision.branch_decisions
        )
        if any(value is None for value in source_hashes):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "legacy/caller-authored rank decisions cannot enter the canonical "
                "adaptive-rank checkpoint",
            )
        return cls(
            policy_hash=record.policy_hash,
            rank=record.rank,
            record_hash=record.record_hash,
            current_mask_hash=record.current_mask_hash,
            route_axis_hash=record.route_axis_hash,
            route_count=record.route_count,
            proposal_generation=record.proposal_generation,
            cumulative_stratum_counts=record.cumulative_stratum_counts,
            D_context_axis_hash=record.D_context_axis_hash,
            D_draw_axis_hash=record.D_draw_axis_hash,
            D_source_measurement_axis_hash=record.D_source_measurement_axis_hash,
            D_weight_hash=record.D_weight_hash,
            fd_design_hash=record.fd_design_hash,
            fd_snapshot_hash=record.fd_snapshot_hash,
            sampled_geometry_source_hash=record.sampled_geometry_source_hash,
            kkt_problem_hash=record.kkt_problem_hash,
            kkt_result_hash=record.kkt_result_hash,
            provisional_branch_family_hash=record.provisional_branch_family_hash,
            decision_certificate_hash=decision.certificate_hash,
            branch_decision_hashes=tuple(
                value.decision_hash for value in decision.branch_decisions
            ),
            branch_source_derivation_receipt_hashes=cast(
                tuple[str, ...], source_hashes
            ),
            branch_ambiguity_code_axes=tuple(
                value.source_derivation_ambiguity_codes
                for value in decision.branch_decisions
            ),
            rank_local_residual_decision_stable=(
                decision.rank_local_residual_decision_stable
            ),
            robust_branch_family_hash=decision.robust_branch_family_hash,
        )

    def __post_init__(self) -> None:
        for name in (
            "policy_hash",
            "record_hash",
            "current_mask_hash",
            "route_axis_hash",
            "D_context_axis_hash",
            "D_draw_axis_hash",
            "D_source_measurement_axis_hash",
            "D_weight_hash",
            "fd_design_hash",
            "fd_snapshot_hash",
            "sampled_geometry_source_hash",
            "kkt_problem_hash",
            "kkt_result_hash",
            "provisional_branch_family_hash",
            "decision_certificate_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        rank = _exact_positive_int(self.rank, name="adaptive FD rank")
        routes = _exact_positive_int(self.route_count, name="adaptive route count")
        generation = _exact_nonnegative_int(
            self.proposal_generation,
            name="adaptive proposal generation",
        )
        counts = tuple(
            _exact_positive_int(value, name="adaptive D stratum count")
            for value in self.cumulative_stratum_counts
        )
        if len(counts) != 2:
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "adaptive-rank checkpoint requires both D strata",
            )
        branch_hashes = tuple(
            _identity(value, name="adaptive branch decision hash")
            for value in self.branch_decision_hashes
        )
        source_hashes = tuple(
            _identity(value, name="adaptive source derivation hash")
            for value in self.branch_source_derivation_receipt_hashes
        )
        ambiguity_axes = tuple(
            tuple(_identity(code, name="adaptive ambiguity code") for code in axis)
            for axis in self.branch_ambiguity_code_axes
        )
        if (
            len(branch_hashes) != 2
            or len(source_hashes) != 2
            or len(ambiguity_axes) != 2
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.MISSING_BRANCH,
                "adaptive-rank checkpoint requires exact spectral and diagonal-l1 sources",
            )
        if (
            type(self.rank_local_residual_decision_stable) is not bool
            or type(self.C_stream_open_authority) is not bool
            or self.C_stream_open_authority is not False
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "adaptive-rank replay checkpoint gained C authority",
            )
        family = self.robust_branch_family_hash
        if family is not None:
            family = _identity(family, name="robust branch-family hash")
        if (family is not None) is not self.rank_local_residual_decision_stable:
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "adaptive-rank robust family and stability flag differ",
            )
        if self.schema != FQR_ADAPTIVE_RANK_REPLAY_PROJECTION_SCHEMA:
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "adaptive-rank replay checkpoint schema is invalid",
            )
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "cumulative_stratum_counts", counts)
        object.__setattr__(self, "branch_decision_hashes", branch_hashes)
        object.__setattr__(
            self, "branch_source_derivation_receipt_hashes", source_hashes
        )
        object.__setattr__(self, "branch_ambiguity_code_axes", ambiguity_axes)
        object.__setattr__(self, "robust_branch_family_hash", family)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "policy_hash",
                "rank",
                "record_hash",
                "current_mask_hash",
                "route_axis_hash",
                "route_count",
                "proposal_generation",
                "cumulative_stratum_counts",
                "D_context_axis_hash",
                "D_draw_axis_hash",
                "D_source_measurement_axis_hash",
                "D_weight_hash",
                "fd_design_hash",
                "fd_snapshot_hash",
                "sampled_geometry_source_hash",
                "kkt_problem_hash",
                "kkt_result_hash",
                "provisional_branch_family_hash",
                "decision_certificate_hash",
                "branch_decision_hashes",
                "branch_source_derivation_receipt_hashes",
                "branch_ambiguity_code_axes",
                "rank_local_residual_decision_stable",
                "robust_branch_family_hash",
                "C_stream_open_authority",
            )
        }

    @property
    def projection_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(),
            prefix=_ADAPTIVE_RANK_REPLAY_PROJECTION_PREFIX,
        )
        if self._projection_hash is not None and self._projection_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "adaptive-rank replay projection was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "adaptive-rank replay projection hash does not match",
            )
        object.__setattr__(self, "_projection_hash", observed)
        return observed

    def validate_live_replay(self, record: V1FQRAdaptiveRankReplayRecord) -> str:
        """Reproject one freshly rebuilt raw-D replay before restart use."""

        rebuilt = V1FQRAdaptiveRankReplayCheckpoint.from_live(record)
        if rebuilt.projection_hash != self.projection_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "fresh adaptive-rank replay differs from its inert checkpoint",
            )
        return rebuilt.projection_hash


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankCheckpointBundle:
    """Durable inert rank-prefix/refusal carrier for crash-safe D replay."""

    policy_hash: str
    uncertainty_set_hash: str
    ledger_hash: str
    admitted_rank_ladder: tuple[int, ...]
    replays: tuple[V1FQRAdaptiveRankReplayCheckpoint, ...]
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    terminal_refusal: V1FQRAdaptiveRankNoCRefusal | None
    live_stability_certificate_hash: str | None
    robust_branch_family_hash: str | None
    raw_D_rows_serialized: bool = False
    live_stability_authority_serialized: bool = False
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    schema: str = FQR_ADAPTIVE_RANK_CHECKPOINT_SCHEMA
    _bundle_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    @classmethod
    def from_live(
        cls,
        *,
        policy: V1FQRAdaptiveRankPolicy,
        ledger: V1FQRAdaptiveRankReplayLedger,
        refusal: V1FQRAdaptiveRankNoCRefusal | None = None,
        stability: V1FQRAdaptiveRankStabilityCertificate | None = None,
    ) -> "V1FQRAdaptiveRankCheckpointBundle":
        if type(policy) is not V1FQRAdaptiveRankPolicy:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "adaptive-rank checkpoint requires the exact live policy",
            )
        if type(ledger) is not V1FQRAdaptiveRankReplayLedger:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "adaptive-rank checkpoint requires the exact live replay ledger",
            )
        policy_hash = policy.assert_integrity()
        ledger_hash = ledger.assert_integrity()
        if ledger.policy.policy_hash != policy_hash:
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "adaptive-rank ledger belongs to another policy",
            )
        if (refusal is None) == (stability is None):
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "adaptive-rank checkpoint requires exactly one refusal or stability outcome",
            )
        if refusal is not None:
            if type(refusal) is not V1FQRAdaptiveRankNoCRefusal:
                _refuse(
                    V1FQRCheckpointRefusalCode.INVALID_TYPE,
                    "adaptive-rank checkpoint refusal changed type",
                )
            if (
                refusal.policy_hash != policy_hash
                or refusal.ledger_hash != ledger_hash
            ):
                _refuse(
                    V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                    "adaptive-rank refusal differs from its replay ledger",
                )
        if stability is not None:
            if type(stability) is not V1FQRAdaptiveRankStabilityCertificate:
                _refuse(
                    V1FQRCheckpointRefusalCode.INVALID_TYPE,
                    "adaptive-rank checkpoint stability changed type",
                )
            stability.assert_integrity()
            if (
                stability.policy_hash != policy_hash
                or stability.replay_ledger_hash != ledger_hash
            ):
                _refuse(
                    V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                    "adaptive-rank stability differs from its replay ledger",
                )
        first = ledger.records[0]
        return cls(
            policy_hash=policy_hash,
            uncertainty_set_hash=policy.uncertainty_set_hash,
            ledger_hash=ledger_hash,
            admitted_rank_ladder=policy.admitted_rank_ladder,
            replays=tuple(
                V1FQRAdaptiveRankReplayCheckpoint.from_live(value)
                for value in ledger.records
            ),
            current_mask_hash=first.current_mask_hash,
            route_axis_hash=first.route_axis_hash,
            route_count=first.route_count,
            proposal_generation=first.proposal_generation,
            terminal_refusal=refusal,
            live_stability_certificate_hash=(
                None if stability is None else stability.certificate_hash
            ),
            robust_branch_family_hash=(
                None if stability is None else stability.robust_branch_family_hash
            ),
        )

    def __post_init__(self) -> None:
        for name in (
            "policy_hash",
            "uncertainty_set_hash",
            "ledger_hash",
            "current_mask_hash",
            "route_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        admitted = tuple(
            _exact_positive_int(value, name="admitted adaptive FD rank")
            for value in self.admitted_rank_ladder
        )
        replays = tuple(self.replays)
        if (
            not admitted
            or not replays
            or any(type(value) is not V1FQRAdaptiveRankReplayCheckpoint for value in replays)
            or tuple(value.rank for value in replays) != admitted[: len(replays)]
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "adaptive-rank checkpoint is not a contiguous admitted prefix",
            )
        routes = _exact_positive_int(self.route_count, name="adaptive route count")
        generation = _exact_nonnegative_int(
            self.proposal_generation,
            name="adaptive proposal generation",
        )
        shared = {
            (
                value.policy_hash,
                value.current_mask_hash,
                value.route_axis_hash,
                value.route_count,
                value.proposal_generation,
                value.cumulative_stratum_counts,
                value.D_context_axis_hash,
                value.D_draw_axis_hash,
                value.D_source_measurement_axis_hash,
                value.D_weight_hash,
            )
            for value in replays
        }
        if len(shared) != 1:
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "adaptive-rank checkpoint replays do not bind one sealed D prefix",
            )
        first = replays[0]
        if (
            first.policy_hash != self.policy_hash
            or first.current_mask_hash != self.current_mask_hash
            or first.route_axis_hash != self.route_axis_hash
            or first.route_count != routes
            or first.proposal_generation != generation
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "adaptive-rank checkpoint root differs from its replay prefix",
            )
        refusal = self.terminal_refusal
        live_hash = self.live_stability_certificate_hash
        family = self.robust_branch_family_hash
        if (refusal is None) == (live_hash is None):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "adaptive-rank checkpoint outcome is absent or conflicting",
            )
        if refusal is not None:
            if type(refusal) is not V1FQRAdaptiveRankNoCRefusal:
                _refuse(
                    V1FQRCheckpointRefusalCode.INVALID_TYPE,
                    "adaptive-rank checkpoint refusal changed type",
                )
            if (
                refusal.policy_hash != self.policy_hash
                or refusal.ledger_hash != self.ledger_hash
                or refusal.current_mask_hash != self.current_mask_hash
                or refusal.proposal_generation != generation
                or family is not None
            ):
                _refuse(
                    V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                    "adaptive-rank checkpoint refusal coordinates differ",
                )
            if refusal.code is V1FQRAdaptiveRankRefusalCode.RANK_PROMOTION_REQUIRED:
                if len(replays) >= len(admitted):
                    _refuse(
                        V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                        "rank-promotion refusal has no admitted successor",
                    )
            elif refusal.code is V1FQRAdaptiveRankRefusalCode.DECISION_FAMILY_AMBIGUOUS:
                if len(replays) != len(admitted):
                    _refuse(
                        V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                        "maximum-rank ambiguity did not exhaust the admitted ladder",
                    )
        else:
            live_hash = _identity(live_hash, name="live stability certificate hash")
            family = _identity(family, name="robust branch-family hash")
            if (
                not replays[-1].rank_local_residual_decision_stable
                or replays[-1].robust_branch_family_hash != family
            ):
                _refuse(
                    V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                    "stable adaptive-rank checkpoint lacks its inert family projection",
                )
        flags = (
            self.raw_D_rows_serialized,
            self.live_stability_authority_serialized,
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
        )
        if any(type(value) is not bool for value in flags) or any(flags):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "adaptive-rank checkpoint serialized raw rows or live authority",
            )
        if self.schema != FQR_ADAPTIVE_RANK_CHECKPOINT_SCHEMA:
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "adaptive-rank checkpoint schema is invalid",
            )
        object.__setattr__(self, "admitted_rank_ladder", admitted)
        object.__setattr__(self, "replays", replays)
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "live_stability_certificate_hash", live_hash)
        object.__setattr__(self, "robust_branch_family_hash", family)
        self.assert_integrity()

    @property
    def next_rank(self) -> int | None:
        if (
            self.terminal_refusal is None
            or self.terminal_refusal.code
            is not V1FQRAdaptiveRankRefusalCode.RANK_PROMOTION_REQUIRED
        ):
            return None
        return self.admitted_rank_ladder[len(self.replays)]

    @property
    def maximum_rank_no_c(self) -> bool:
        return (
            self.terminal_refusal is not None
            and self.terminal_refusal.code
            is V1FQRAdaptiveRankRefusalCode.DECISION_FAMILY_AMBIGUOUS
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_hash": self.policy_hash,
            "uncertainty_set_hash": self.uncertainty_set_hash,
            "ledger_hash": self.ledger_hash,
            "admitted_rank_ladder": self.admitted_rank_ladder,
            "replay_projection_hashes": tuple(
                value.projection_hash for value in self.replays
            ),
            "current_mask_hash": self.current_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "proposal_generation": self.proposal_generation,
            "terminal_refusal_hash": (
                None
                if self.terminal_refusal is None
                else self.terminal_refusal.refusal_hash
            ),
            "live_stability_certificate_hash": self.live_stability_certificate_hash,
            "robust_branch_family_hash": self.robust_branch_family_hash,
            "raw_D_rows_serialized": self.raw_D_rows_serialized,
            "live_stability_authority_serialized": (
                self.live_stability_authority_serialized
            ),
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
        }

    @property
    def bundle_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for value in self.replays:
            value.assert_integrity()
        observed = content_hash(
            self._payload(),
            prefix=_ADAPTIVE_RANK_CHECKPOINT_BUNDLE_PREFIX,
        )
        if self._bundle_hash is not None and self._bundle_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "adaptive-rank checkpoint bundle was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "adaptive-rank checkpoint bundle hash does not match",
            )
        object.__setattr__(self, "_bundle_hash", observed)
        return observed

    def validate_live_prefix(
        self,
        *,
        policy: V1FQRAdaptiveRankPolicy,
        ledger: V1FQRAdaptiveRankReplayLedger,
    ) -> str:
        """Require hash-identical live raw-D replays before rank resume."""

        if type(policy) is not V1FQRAdaptiveRankPolicy:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "adaptive-rank restart requires the exact live policy",
            )
        if type(ledger) is not V1FQRAdaptiveRankReplayLedger:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "adaptive-rank restart requires the exact live ledger",
            )
        if (
            policy.assert_integrity() != self.policy_hash
            or policy.uncertainty_set_hash != self.uncertainty_set_hash
            or ledger.assert_integrity() != self.ledger_hash
            or len(ledger.records) != len(self.replays)
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "fresh adaptive-rank policy/ledger differs from checkpoint",
            )
        for projection, record in zip(self.replays, ledger.records, strict=True):
            projection.validate_live_replay(record)
        return self.bundle_hash


@dataclass(frozen=True, slots=True)
class V1FQRCheckpointBundle:
    """Two-branch, content-addressed local F/Q/R resume bundle."""

    dual_chart_geometry: V1DualChartGeometry = field(repr=False, compare=False)
    chart_identity_hash: str
    branch_family_registration_hash: str
    branches: tuple[V1FQRCheckpointBranch, ...]
    branch_ids: tuple[str, ...] = CANONICAL_FQR_BRANCH_IDS
    branch_family_frozen_before_C_outcomes: bool = True
    cross_branch_postselection_occurred: bool = False
    validation_outcomes_seen: bool = False
    raw_context_route_rows_serialized: bool = False
    raw_secant_records_serialized: bool = False
    live_receipt_authority: bool = False
    proposal_queue_authority: bool = False
    calibration_authority: bool = False
    bce_oracle_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = FQR_CHECKPOINT_AUTHORITY_SCOPE
    schema: str = FQR_CHECKPOINT_SCHEMA
    _bundle_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.dual_chart_geometry) is not V1DualChartGeometry:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "F/Q/R checkpoint requires the reduced dual-chart geometry",
            )
        self.dual_chart_geometry.assert_integrity()
        chart = _identity(self.chart_identity_hash, name="chart identity hash")
        registration = _identity(
            self.branch_family_registration_hash,
            name="branch family registration hash",
        )
        branch_ids = tuple(
            _identity(value, name="registered branch ID") for value in self.branch_ids
        )
        branches = tuple(self.branches)
        if (
            branch_ids != CANONICAL_FQR_BRANCH_IDS
            or len(branches) != len(branch_ids)
            or len(branches) > _MAX_BRANCHES
            or any(type(value) is not V1FQRCheckpointBranch for value in branches)
            or tuple(value.branch_id for value in branches) != branch_ids
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.MISSING_BRANCH,
                "canonical checkpoints require spectral and diagonal_l1 branches in order",
            )
        geometry = self.dual_chart_geometry
        family_ids: list[str] = []
        alpha_ledgers: list[tuple[str, int]] = []
        for branch in branches:
            branch.assert_integrity()
            ladder = branch.frozen_ladder
            if (
                ladder.campaign_contract_hash != geometry.campaign_contract_hash
                or ladder.model_identity_hash != geometry.model_identity_hash
                or ladder.query_contract_hash != geometry.query_contract_hash
                or ladder.preprocessing_hash != geometry.preprocessing_hash
                or ladder.current_mask_hash != geometry.current_mask_hash
                or ladder.sampling_law_hash != geometry.sampling_law_hash
                or ladder.proposal_generation != geometry.proposal_generation
                or ladder.route_count != geometry.route_count
                or ladder.route_axis_hash != geometry.route_axis_hash
                or branch.secant_design.chart_identity_hash != chart
            ):
                _refuse(
                    V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                    "branch ladder differs from the current dual chart",
                )
            family_ids.append(branch.fqr_design.joint_fqr_family_registration_hash)
            alpha_ledgers.append(
                (
                    branch.sequential_design.trajectory_alpha_ledger_hash,
                    branch.sequential_design.trajectory_alpha_ledger_generation,
                )
            )
        if (
            any(value != registration for value in family_ids)
            or len(set(alpha_ledgers)) != 1
            or alpha_ledgers[0][1] != geometry.proposal_generation
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "branch family or alpha-ledger coordinates disagree",
            )
        if (
            self.branch_family_frozen_before_C_outcomes is not True
            or self.cross_branch_postselection_occurred is not False
            or self.validation_outcomes_seen is not False
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.POST_SELECTION,
                "checkpoint branch family was changed after C/validation outcomes",
            )
        if (
            self.raw_context_route_rows_serialized is not False
            or self.raw_secant_records_serialized is not False
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.RAW_RECORD_STATE_FORBIDDEN,
                "checkpoint bundle cannot contain raw chart rows or secant records",
            )
        if any(
            getattr(self, name) is not False
            for name in (
                "live_receipt_authority",
                "proposal_queue_authority",
                "calibration_authority",
                "bce_oracle_authority",
                "irreversible_decision_authority",
                "mask_commit_authority",
                "terminal_authority",
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "checkpoint bundle cannot grant action authority",
            )
        if (
            self.authority_scope != FQR_CHECKPOINT_AUTHORITY_SCOPE
            or self.schema != FQR_CHECKPOINT_SCHEMA
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "F/Q/R checkpoint schema or authority is invalid",
            )
        object.__setattr__(self, "chart_identity_hash", chart)
        object.__setattr__(self, "branch_family_registration_hash", registration)
        object.__setattr__(self, "branch_ids", branch_ids)
        object.__setattr__(self, "branches", branches)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "dual_chart_geometry_hash": self.dual_chart_geometry.geometry_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "branch_family_registration_hash": self.branch_family_registration_hash,
            "branch_ids": self.branch_ids,
            "branch_hashes": tuple(value.branch_hash for value in self.branches),
            "branch_family_frozen_before_C_outcomes": (
                self.branch_family_frozen_before_C_outcomes
            ),
            "cross_branch_postselection_occurred": (
                self.cross_branch_postselection_occurred
            ),
            "validation_outcomes_seen": self.validation_outcomes_seen,
            "raw_context_route_rows_serialized": (
                self.raw_context_route_rows_serialized
            ),
            "raw_secant_records_serialized": self.raw_secant_records_serialized,
            "live_receipt_authority": self.live_receipt_authority,
            "proposal_queue_authority": self.proposal_queue_authority,
            "calibration_authority": self.calibration_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "irreversible_decision_authority": (self.irreversible_decision_authority),
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def bundle_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.dual_chart_geometry.assert_integrity()
        for value in self.branches:
            value.assert_integrity()
        observed = content_hash(self._payload(), prefix=_BUNDLE_PREFIX)
        if self._bundle_hash is not None and self._bundle_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "F/Q/R checkpoint bundle was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "F/Q/R checkpoint bundle hash does not match",
            )
        object.__setattr__(self, "_bundle_hash", observed)
        return observed


_ALLOWED_TYPES: Final = {
    cls.__name__: cls
    for cls in (
        HardMask,
        V1PackedRemovalAction,
        V1FrozenLadderAction,
        V1FrozenCandidateLadder,
        V1DualChartMismatchReceipt,
        V1DualChartGeometry,
        V1DirectionalSecantDesign,
        V1DirectionalRemainderInterval,
        V1FQRCertificationDesign,
        V1FQEndpointBound,
        V1FQRCertificationDecision,
        V1DecisionStabilityCertificate,
        V1FQRSequentialAcquisitionDesign,
        V1FQRSequentialCellSummary,
        V1FQRSequentialEndpointSummary,
        V1FQRSequentialAcquisitionReceipt,
        V1FQRDeterministicUncertaintyReceipt,
        V1FQRDecisionReadinessReceipt,
        V1FQRCandidateQueueEntry,
        V1FQRSequentialAccumulatorSnapshot,
        V1FQRBranchFamilyAccumulatorSnapshot,
        V1LabelNativeSixRoleAccumulatorSnapshot,
        V1FQRAdaptiveRankNoCRefusal,
        V1FQRAdaptiveRankReplayCheckpoint,
        V1FQRAdaptiveRankCheckpointBundle,
        V1CompactFQREndpointCertificate,
        V1CompactFQRCertificationReceipt,
        V1CompactFQRDecision,
        V1CompactFQRPreCResourceRegistration,
        V1FQRPreCDominanceCheckpoint,
        V1FQRPreFrozenFamilyCheckpoint,
        V1CompactFQRBranchCheckpoint,
        V1CompactFQRNegativeBranchCheckpoint,
        V1CompactFQRSourceAliasCheckpoint,
        V1CompactFQRQueueEntryCheckpoint,
        V1CompactFQRBranchFamilyQueueCheckpoint,
        V1FQRAuthorizedAcquisitionCheckpoint,
        V1CompactFQRCheckpointBundle,
        V1FQRInFlightSharedCCheckpointBundle,
        V1LabelNativeSixRoleInFlightCCheckpointBundle,
        V1FQRCertificationReceiptCheckpoint,
        V1FQRCandidateQueueCheckpoint,
        V1FQRCheckpointBranch,
        V1FQRCheckpointBundle,
    )
}

_ALLOWED_ENUM_TYPES: Final = {
    V1RelaxedKKTBranch.__name__: V1RelaxedKKTBranch,
    V1FQRAdaptiveRankRefusalCode.__name__: V1FQRAdaptiveRankRefusalCode,
}


@dataclass(slots=True)
class _WireAggregate:
    items: int = 0
    array_bytes: int = 0
    byte_string_bytes: int = 0


@dataclass(slots=True)
class _WireBudget:
    """One recursive cursor over shared aggregate wire-resource counters."""

    depth: int = 0
    aggregate: _WireAggregate = field(default_factory=_WireAggregate)

    @property
    def items(self) -> int:
        return self.aggregate.items

    @property
    def array_bytes(self) -> int:
        return self.aggregate.array_bytes

    @property
    def byte_string_bytes(self) -> int:
        return self.aggregate.byte_string_bytes

    def claim_items(self, count: int) -> None:
        observed = self.aggregate.items + count
        if observed > _MAX_SEQUENCE_ITEMS:
            _refuse(
                V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
                "aggregate wire-container item count exceeds checkpoint bound",
            )
        self.aggregate.items = observed

    def claim_array_bytes(self, count: int) -> None:
        observed = self.aggregate.array_bytes + count
        if observed > _MAX_ARRAY_BYTES:
            _refuse(
                V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
                "aggregate numeric-array bytes exceed checkpoint bound",
            )
        self.aggregate.array_bytes = observed

    def claim_byte_string_bytes(self, count: int) -> None:
        observed = self.aggregate.byte_string_bytes + count
        if observed > _MAX_BYTE_STRING_BYTES:
            _refuse(
                V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
                "aggregate raw byte-string bytes exceed checkpoint bound",
            )
        self.aggregate.byte_string_bytes = observed

    def descend(self) -> "_WireBudget":
        if self.depth >= _MAX_NESTING_DEPTH:
            _refuse(
                V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
                "wire object exceeds the maximum nesting depth",
            )
        return _WireBudget(depth=self.depth + 1, aggregate=self.aggregate)


def _encode_value(value: object, budget: _WireBudget) -> object:
    if value is None or type(value) is bool:
        return value
    if type(value) is str:
        if len(value.encode("utf-8")) > _MAX_STRING_BYTES:
            _refuse(V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED, "string too large")
        return value
    enum_type = type(value)
    if _ALLOWED_ENUM_TYPES.get(enum_type.__name__) is enum_type:
        return {
            "__enum__": {
                "type": enum_type.__name__,
                "value": cast(Enum, value).value,
            }
        }
    if isinstance(value, Integral) and not isinstance(value, bool):
        return {"__int__": str(int(value))}
    if type(value) is float:
        if not math.isfinite(value):
            _refuse(V1FQRCheckpointRefusalCode.MALFORMED, "nonfinite wire float")
        return {"__f64le__": struct.pack("<d", value).hex()}
    if type(value) is bytes:
        if len(value) > _MAX_BYTE_STRING_BYTES:
            _refuse(
                V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
                "byte string exceeds checkpoint bound",
            )
        budget.claim_byte_string_bytes(len(value))
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if type(value) is np.ndarray:
        if value.dtype != _LE_F64 or not value.flags.c_contiguous:
            _refuse(
                V1FQRCheckpointRefusalCode.INVALID_TYPE,
                "checkpoint numeric arrays must be contiguous little-endian float64",
            )
        if value.nbytes > _MAX_ARRAY_BYTES or not np.isfinite(value).all():
            _refuse(
                V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
                "checkpoint numeric array is nonfinite or too large",
            )
        budget.claim_array_bytes(value.nbytes)
        return {
            "__ndarray__": {
                "dtype": "<f8",
                "shape": [int(item) for item in value.shape],
                "data": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
            }
        }
    if type(value) is tuple:
        if len(value) > _MAX_SEQUENCE_ITEMS:
            _refuse(
                V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
                "wire tuple exceeds item bound",
            )
        budget.claim_items(len(value))
        child = budget.descend()
        return {"__tuple__": [_encode_value(item, child) for item in value]}
    if type(value) is dict:
        if len(value) > _MAX_SEQUENCE_ITEMS or any(
            type(key) is not str for key in value
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
                "wire mapping is too large or has non-string keys",
            )
        budget.claim_items(len(value))
        child = budget.descend()
        return {
            "__dict__": {key: _encode_value(value[key], child) for key in sorted(value)}
        }
    cls = type(value)
    registered = _ALLOWED_TYPES.get(cls.__name__)
    if registered is not cls or not is_dataclass(value):
        if cls in {
            V1FQRCertificationReceipt,
            V1FQRCandidateQueueAuthorization,
        } or cls.__name__ in {
            "V1DirectionalSecantRecord",
            "V1DirectionalRemainderReceipt",
        }:
            _refuse(
                V1FQRCheckpointRefusalCode.RAW_RECORD_STATE_FORBIDDEN,
                "raw-record-bearing live F/Q/R objects require compact projections",
            )
        if cls in {
            V1CompactFQRBranchEvidenceBinding,
            V1CompactFQRBranchNegativeEvidenceBinding,
            V1CompactFQRProposalAuthorization,
            V1CompactFQRBranchFamilyAuthorization,
            V1FQRAuthorizedAcquisition,
            V1FQRBranchFamilyAccumulator,
            V1LabelNativeSixRoleCAccumulator,
        }:
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "live F/Q/R authority carriers require inert checkpoint projections",
            )
        if "FQ" in cls.__name__ and "FQR" not in cls.__name__:
            _refuse(
                V1FQRCheckpointRefusalCode.FQ_ONLY_LEGACY,
                "F/Q-only material cannot authorize canonical F/Q/R resume",
            )
        _refuse(
            V1FQRCheckpointRefusalCode.INVALID_TYPE,
            f"unsupported checkpoint object type {cls.__name__}",
        )
    init_fields = tuple(item for item in fields(cls) if item.init)
    # Optional fields are omitted from their frozen legacy wire schemas rather
    # than serialized as defaults.  This keeps historical payload bytes exact.
    if cls in {
        V1CompactFQRBranchFamilyQueueCheckpoint,
        V1FQRAuthorizedAcquisitionCheckpoint,
    } and not value.source_branch_negative_evidence_hashes:
        init_fields = tuple(
            item
            for item in init_fields
            if item.name != "source_branch_negative_evidence_hashes"
        )
    if cls is V1FQRDeterministicUncertaintyReceipt:
        if value.schema == FQR_DETERMINISTIC_UNCERTAINTY_SCHEMA:
            excluded_deterministic_fields = {
                *_DETERMINISTIC_PROPOSAL_V3_FIELDS,
                "fd_rank_decision_stability_certificate_hash",
            }
        elif value.schema == FQR_DETERMINISTIC_UNCERTAINTY_V2_SCHEMA:
            excluded_deterministic_fields = set(
                _DETERMINISTIC_PROPOSAL_V3_FIELDS
            )
        elif value.schema == FQR_DETERMINISTIC_UNCERTAINTY_PROPOSAL_V3_SCHEMA:
            excluded_deterministic_fields = {
                "fd_rank_decision_stability_certificate_hash"
            }
        else:
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                "deterministic uncertainty schema is not registered",
            )
        init_fields = tuple(
            item
            for item in init_fields
            if item.name not in excluded_deterministic_fields
        )
    if len(init_fields) > _MAX_SEQUENCE_ITEMS:
        _refuse(
            V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
            "typed checkpoint object exceeds the per-container item bound",
        )
    budget.claim_items(len(init_fields))
    child = budget.descend()
    payload = {
        item.name: _encode_value(getattr(value, item.name), child)
        for item in init_fields
    }
    return {"__type__": cls.__name__, "fields": payload}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_value(
    value: object, *, budget: _WireBudget | None = None
) -> object:
    cursor = _WireBudget() if budget is None else budget
    if cursor.depth > _MAX_NESTING_DEPTH:
        _refuse(
            V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
            "decoded object exceeds maximum nesting depth",
        )
    if value is None or type(value) in {bool, str}:
        return value
    if type(value) is not dict:
        _refuse(
            V1FQRCheckpointRefusalCode.MALFORMED,
            "wire values must use canonical tagged objects",
        )
    raw = cast(dict[str, Any], value)
    if set(raw) == {"__int__"}:
        text = raw["__int__"]
        if (
            type(text) is not str
            or not text
            or (text[0] == "-" and not text[1:].isdigit())
            or (text[0] != "-" and not text.isdigit())
        ):
            _refuse(V1FQRCheckpointRefusalCode.MALFORMED, "invalid integer tag")
        return int(text)
    if set(raw) == {"__f64le__"}:
        text = raw["__f64le__"]
        if type(text) is not str or len(text) != 16:
            _refuse(V1FQRCheckpointRefusalCode.MALFORMED, "invalid float tag")
        try:
            result = struct.unpack("<d", bytes.fromhex(text))[0]
        except (ValueError, struct.error) as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.MALFORMED, "invalid float bytes"
            ) from error
        if not math.isfinite(result):
            _refuse(V1FQRCheckpointRefusalCode.MALFORMED, "nonfinite float")
        return result
    if set(raw) == {"__enum__"}:
        spec = raw["__enum__"]
        if type(spec) is not dict or set(spec) != {"type", "value"}:
            _refuse(V1FQRCheckpointRefusalCode.MALFORMED, "invalid enum tag")
        enum_type = _ALLOWED_ENUM_TYPES.get(spec["type"])
        if enum_type is None or type(spec["value"]) is not str:
            _refuse(V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA, "unknown wire enum")
        try:
            return enum_type(spec["value"])
        except ValueError as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.MALFORMED,
                "invalid wire enum value",
            ) from error
    if set(raw) == {"__bytes__"}:
        text = raw["__bytes__"]
        if type(text) is not str:
            _refuse(V1FQRCheckpointRefusalCode.MALFORMED, "invalid byte tag")
        if len(text) > 4 * ((_MAX_BYTE_STRING_BYTES + 2) // 3):
            _refuse(V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED, "bytes too large")
        if len(text) % 4 != 0:
            _refuse(V1FQRCheckpointRefusalCode.MALFORMED, "invalid base64 byte size")
        padding = len(text) - len(text.rstrip("="))
        misplaced_padding = "=" in (text[:-padding] if padding else text)
        if padding > 2 or misplaced_padding:
            _refuse(V1FQRCheckpointRefusalCode.MALFORMED, "invalid base64 padding")
        decoded_size = (len(text) // 4) * 3 - padding
        cursor.claim_byte_string_bytes(decoded_size)
        try:
            result = base64.b64decode(text, validate=True)
        except ValueError as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.MALFORMED, "invalid base64 bytes"
            ) from error
        if len(result) > _MAX_BYTE_STRING_BYTES:
            _refuse(V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED, "bytes too large")
        return result
    if set(raw) == {"__tuple__"}:
        items = raw["__tuple__"]
        if type(items) is not list or len(items) > _MAX_SEQUENCE_ITEMS:
            _refuse(V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED, "tuple too large")
        cursor.claim_items(len(items))
        child = cursor.descend()
        return tuple(_decode_value(item, budget=child) for item in items)
    if set(raw) == {"__dict__"}:
        mapping = raw["__dict__"]
        if type(mapping) is not dict or len(mapping) > _MAX_SEQUENCE_ITEMS:
            _refuse(V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED, "mapping too large")
        cursor.claim_items(len(mapping))
        child = cursor.descend()
        return {
            key: _decode_value(item, budget=child)
            for key, item in cast(dict[str, Any], mapping).items()
        }
    if set(raw) == {"__ndarray__"}:
        spec = raw["__ndarray__"]
        if type(spec) is not dict or set(spec) != {"dtype", "shape", "data"}:
            _refuse(V1FQRCheckpointRefusalCode.MALFORMED, "invalid ndarray tag")
        if spec["dtype"] != "<f8" or type(spec["shape"]) is not list:
            _refuse(V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA, "unknown array dtype")
        shape = tuple(
            _exact_nonnegative_int(item, name="array dimension")
            for item in spec["shape"]
        )
        if len(shape) > 8:
            _refuse(V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED, "array rank too high")
        count = math.prod(shape)
        expected = count * 8
        if expected > _MAX_ARRAY_BYTES or type(spec["data"]) is not str:
            _refuse(V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED, "array too large")
        if len(spec["data"]) != 4 * ((expected + 2) // 3):
            _refuse(
                V1FQRCheckpointRefusalCode.MALFORMED,
                "array base64 size mismatch",
            )
        cursor.claim_array_bytes(expected)
        try:
            data = base64.b64decode(spec["data"], validate=True)
        except ValueError as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.MALFORMED, "invalid array base64"
            ) from error
        if len(data) != expected:
            _refuse(V1FQRCheckpointRefusalCode.MALFORMED, "array size mismatch")
        array = np.frombuffer(data, dtype=_LE_F64).reshape(shape)
        if not np.isfinite(array).all():
            _refuse(V1FQRCheckpointRefusalCode.MALFORMED, "nonfinite array")
        array.setflags(write=False)
        return array
    if set(raw) == {"__type__", "fields"}:
        name = raw["__type__"]
        payload = raw["fields"]
        if type(name) is not str or type(payload) is not dict:
            _refuse(V1FQRCheckpointRefusalCode.MALFORMED, "invalid typed object")
        cls = _ALLOWED_TYPES.get(name)
        if cls is None:
            if "FQ" in name and "FQR" not in name:
                _refuse(
                    V1FQRCheckpointRefusalCode.FQ_ONLY_LEGACY,
                    "F/Q-only checkpoint types cannot authorize canonical F/Q/R resume",
                )
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                f"unknown checkpoint type {name!r}",
            )
        expected_fields = tuple(item.name for item in fields(cls) if item.init)
        legacy_optional_fields: set[str] = set()
        if cls in {
            V1CompactFQRBranchFamilyQueueCheckpoint,
            V1FQRAuthorizedAcquisitionCheckpoint,
        }:
            legacy_optional_fields.add("source_branch_negative_evidence_hashes")
        if cls is V1FQRDeterministicUncertaintyReceipt:
            legacy_optional_fields.add(
                "fd_rank_decision_stability_certificate_hash"
            )
            legacy_optional_fields.update(_DETERMINISTIC_PROPOSAL_V3_FIELDS)
        observed_fields = set(payload)
        required_fields = set(expected_fields).difference(legacy_optional_fields)
        if (
            not required_fields.issubset(observed_fields)
            or not observed_fields.issubset(set(expected_fields))
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                f"checkpoint type {name!r} has missing or unknown fields",
            )
        if cls in {
            V1CompactFQRBranchFamilyQueueCheckpoint,
            V1FQRAuthorizedAcquisitionCheckpoint,
        }:
            schema_value = payload.get("schema")
            negative_present = (
                "source_branch_negative_evidence_hashes" in observed_fields
            )
            v2_schema = (
                FQR_COMPACT_FAMILY_QUEUE_PROJECTION_V2_SCHEMA
                if cls is V1CompactFQRBranchFamilyQueueCheckpoint
                else FQR_AUTHORIZED_ACQUISITION_PROJECTION_V2_SCHEMA
            )
            v1_schema = (
                FQR_COMPACT_FAMILY_QUEUE_PROJECTION_SCHEMA
                if cls is V1CompactFQRBranchFamilyQueueCheckpoint
                else FQR_AUTHORIZED_ACQUISITION_PROJECTION_SCHEMA
            )
            if (
                schema_value not in {v1_schema, v2_schema}
                or negative_present != (schema_value == v2_schema)
            ):
                _refuse(
                    V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                    f"checkpoint type {name!r} does not use its canonical v1/v2 "
                    "field set",
                )
        if cls is V1FQRDeterministicUncertaintyReceipt:
            schema_value = payload.get("schema")
            certificate_present = (
                "fd_rank_decision_stability_certificate_hash" in observed_fields
            )
            proposal_fields_present = observed_fields.intersection(
                _DETERMINISTIC_PROPOSAL_V3_FIELDS
            )
            canonical_field_set = (
                schema_value == FQR_DETERMINISTIC_UNCERTAINTY_SCHEMA
                and not certificate_present
                and not proposal_fields_present
            ) or (
                schema_value == FQR_DETERMINISTIC_UNCERTAINTY_V2_SCHEMA
                and certificate_present
                and not proposal_fields_present
            ) or (
                schema_value
                == FQR_DETERMINISTIC_UNCERTAINTY_PROPOSAL_V3_SCHEMA
                and not certificate_present
                and proposal_fields_present == _DETERMINISTIC_PROPOSAL_V3_FIELDS
            )
            if not canonical_field_set:
                _refuse(
                    V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
                    "deterministic uncertainty does not use its canonical v1/v2/v3 "
                    "field set",
                )
        if len(expected_fields) > _MAX_SEQUENCE_ITEMS:
            _refuse(
                V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
                "decoded typed object exceeds the per-container item bound",
            )
        cursor.claim_items(len(observed_fields))
        child = cursor.descend()
        kwargs = {
            field_name: _decode_value(payload[field_name], budget=child)
            for field_name in expected_fields
            if field_name in payload
        }
        if (
            cls is V1FQRDeterministicUncertaintyReceipt
            and payload.get("schema")
            == FQR_DETERMINISTIC_UNCERTAINTY_PROPOSAL_V3_SCHEMA
        ):
            kwargs["_proposal_freeze_issuance_token"] = (
                _PROPOSAL_DETERMINISTIC_ISSUANCE_TOKEN
            )
        if cls is V1LabelNativeSixRoleInFlightCCheckpointBundle:
            kwargs["_issuance_token"] = (
                _LABEL_NATIVE_SIX_ROLE_DECODE_ISSUANCE_TOKEN
            )
        try:
            return cls(**kwargs)
        except V1FQRCheckpointRefusal:
            raise
        except PermissionError as error:
            raise V1FQRCheckpointRefusal(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                f"decoded {name} crossed an authority boundary: {error}",
            ) from error
        except (TypeError, ValueError) as error:
            text = str(error).lower()
            code = (
                V1FQRCheckpointRefusalCode.POST_SELECTION
                if "post-select" in text or "reorder" in text
                else (
                    V1FQRCheckpointRefusalCode.HASH_MISMATCH
                    if "tamper" in text or "hash" in text
                    else V1FQRCheckpointRefusalCode.STALE_COORDINATES
                )
            )
            raise V1FQRCheckpointRefusal(
                code, f"decoded {name} failed independent validation: {error}"
            ) from error
    _refuse(V1FQRCheckpointRefusalCode.MALFORMED, "unknown wire tag")


def encode_v1_fqr_checkpoint(
    bundle: (
        V1FQRCheckpointBundle
        | V1CompactFQRCheckpointBundle
        | V1FQRInFlightSharedCCheckpointBundle
        | V1LabelNativeSixRoleInFlightCCheckpointBundle
        | V1FQRAdaptiveRankCheckpointBundle
    ),
) -> bytes:
    """Encode one validated bundle into deterministic, content-addressed bytes."""

    if type(bundle) not in {
        V1FQRCheckpointBundle,
        V1CompactFQRCheckpointBundle,
        V1FQRInFlightSharedCCheckpointBundle,
        V1LabelNativeSixRoleInFlightCCheckpointBundle,
        V1FQRAdaptiveRankCheckpointBundle,
    }:
        if type(bundle) in {
            V1FQRCertificationReceipt,
            V1FQRCandidateQueueAuthorization,
        } or type(bundle).__name__ in {
            "V1DirectionalSecantRecord",
            "V1DirectionalRemainderReceipt",
        }:
            _refuse(
                V1FQRCheckpointRefusalCode.RAW_RECORD_STATE_FORBIDDEN,
                "raw-record-bearing live F/Q/R state requires a bounded projection",
            )
        if type(bundle) in {
            V1CompactFQRBranchEvidenceBinding,
            V1CompactFQRBranchNegativeEvidenceBinding,
            V1CompactFQRProposalAuthorization,
            V1CompactFQRBranchFamilyAuthorization,
            V1FQRAuthorizedAcquisition,
            V1FQRBranchFamilyAccumulator,
            V1LabelNativeSixRoleCAccumulator,
        }:
            _refuse(
                V1FQRCheckpointRefusalCode.AUTHORITY_VIOLATION,
                "live F/Q/R authority cannot be encoded as a checkpoint root",
            )
        if "FQ" in type(bundle).__name__ and "FQR" not in type(bundle).__name__:
            _refuse(
                V1FQRCheckpointRefusalCode.FQ_ONLY_LEGACY,
                "F/Q-only material cannot authorize canonical F/Q/R resume",
            )
        _refuse(
            V1FQRCheckpointRefusalCode.INVALID_TYPE,
            "F/Q/R codec requires the exact checkpoint-bundle type",
        )
    bundle_hash = bundle.assert_integrity()
    schema = bundle.schema
    envelope = {
        "schema": schema,
        "bundle_hash": bundle_hash,
        "state": _encode_value(bundle, _WireBudget()),
    }
    body = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if len(body) > _MAX_PAYLOAD_BYTES:
        _refuse(
            V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
            "encoded F/Q/R checkpoint exceeds the payload bound",
        )
    digest = hashlib.sha256(body).digest()
    return _MAGIC + struct.pack("<Q", len(body)) + digest + body


def decode_v1_fqr_checkpoint(
    payload: bytes,
    *,
    expected_chart_identity_hash: str | None = None,
    expected_current_mask_hash: str | None = None,
    expected_proposal_generation: int | None = None,
    expected_route_axis_hash: str | None = None,
    expected_route_count: int | None = None,
    expected_branch_family_registration_hash: str | None = None,
    expected_pre_c_graph_hash: str | None = None,
    expected_branch_family_design_hash: str | None = None,
    expected_snapshot_hash: str | None = None,
) -> (
    V1FQRCheckpointBundle
    | V1CompactFQRCheckpointBundle
    | V1FQRInFlightSharedCCheckpointBundle
    | V1LabelNativeSixRoleInFlightCCheckpointBundle
    | V1FQRAdaptiveRankCheckpointBundle
):
    """Decode and independently replay a bounded F/Q/R checkpoint graph."""

    if type(payload) is not bytes:
        _refuse(
            V1FQRCheckpointRefusalCode.INVALID_TYPE,
            "checkpoint payload must be exact bytes",
        )
    if not payload.startswith(_MAGIC):
        if payload.startswith(b"MV1"):
            _refuse(
                V1FQRCheckpointRefusalCode.FQ_ONLY_LEGACY,
                "legacy/FQ-only checkpoint framing cannot enter canonical F/Q/R",
            )
        _refuse(
            V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA,
            "unknown F/Q/R checkpoint magic",
        )
    if len(payload) < _FRAME_PREFIX_BYTES:
        _refuse(V1FQRCheckpointRefusalCode.MALFORMED, "truncated checkpoint frame")
    declared = struct.unpack("<Q", payload[len(_MAGIC) : len(_MAGIC) + 8])[0]
    if declared > _MAX_PAYLOAD_BYTES or len(payload) != _FRAME_PREFIX_BYTES + declared:
        _refuse(
            V1FQRCheckpointRefusalCode.BOUNDS_EXCEEDED,
            "checkpoint length is inconsistent or outside bounds",
        )
    expected_digest = payload[len(_MAGIC) + 8 : _FRAME_PREFIX_BYTES]
    body = payload[_FRAME_PREFIX_BYTES:]
    if hashlib.sha256(body).digest() != expected_digest:
        _refuse(
            V1FQRCheckpointRefusalCode.HASH_MISMATCH,
            "checkpoint payload digest does not match",
        )
    try:
        envelope = json.loads(
            body.decode("ascii"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise V1FQRCheckpointRefusal(
            V1FQRCheckpointRefusalCode.MALFORMED,
            f"checkpoint JSON is not strict: {error}",
        ) from error
    if type(envelope) is not dict or set(envelope) != {
        "schema",
        "bundle_hash",
        "state",
    }:
        _refuse(V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA, "invalid envelope schema")
    schema = envelope["schema"]
    if schema not in {
        FQR_CHECKPOINT_SCHEMA,
        FQR_COMPACT_CHECKPOINT_SCHEMA,
        FQR_INFLIGHT_C_CHECKPOINT_SCHEMA,
        FQR_LABEL_NATIVE_SIX_ROLE_INFLIGHT_C_CHECKPOINT_SCHEMA,
        FQR_ADAPTIVE_RANK_CHECKPOINT_SCHEMA,
    }:
        _refuse(V1FQRCheckpointRefusalCode.UNKNOWN_SCHEMA, "unknown bundle schema")
    restored = _decode_value(envelope["state"])
    expected_root_by_schema = {
        FQR_CHECKPOINT_SCHEMA: V1FQRCheckpointBundle,
        FQR_COMPACT_CHECKPOINT_SCHEMA: V1CompactFQRCheckpointBundle,
        FQR_INFLIGHT_C_CHECKPOINT_SCHEMA: V1FQRInFlightSharedCCheckpointBundle,
        FQR_LABEL_NATIVE_SIX_ROLE_INFLIGHT_C_CHECKPOINT_SCHEMA: (
            V1LabelNativeSixRoleInFlightCCheckpointBundle
        ),
        FQR_ADAPTIVE_RANK_CHECKPOINT_SCHEMA: V1FQRAdaptiveRankCheckpointBundle,
    }
    expected_root = expected_root_by_schema[schema]
    if type(restored) is not expected_root or restored.schema != schema:
        _refuse(
            V1FQRCheckpointRefusalCode.INVALID_TYPE,
            "checkpoint root is not the exact bundle type named by its schema",
        )
    restored.assert_integrity(_identity(envelope["bundle_hash"], name="bundle hash"))
    if type(restored) is V1FQRInFlightSharedCCheckpointBundle:
        chart_identity_hash = restored.chart_identity_hash
        branch_family_registration_hash = restored.branch_family_registration_hash
        current_mask_hash = restored.current_mask_hash
        proposal_generation = restored.proposal_generation
        route_axis_hash = restored.route_axis_hash
        route_count = restored.route_count
        if restored.shared_record_count > 0 and expected_snapshot_hash is None:
            _refuse(
                V1FQRCheckpointRefusalCode.SNAPSHOT_HASH_REQUIRED,
                "nonempty shared-C checkpoint decode requires an exact expected snapshot hash",
            )
        if expected_snapshot_hash is not None:
            try:
                restored.snapshot.assert_integrity(expected_snapshot_hash)
            except (TypeError, ValueError, PermissionError) as error:
                raise V1FQRCheckpointRefusal(
                    V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                    f"checkpoint expected snapshot hash does not match: {error}",
                ) from error
    elif type(restored) is V1LabelNativeSixRoleInFlightCCheckpointBundle:
        chart_identity_hash = restored.chart_identity_hash
        branch_family_registration_hash = (
            restored.joint_six_role_family_registration_hash
        )
        current_mask_hash = restored.current_mask_hash
        proposal_generation = restored.proposal_generation
        route_axis_hash = restored.route_axis_hash
        route_count = restored.route_count
        if restored.shared_record_count > 0 and expected_snapshot_hash is None:
            _refuse(
                V1FQRCheckpointRefusalCode.SNAPSHOT_HASH_REQUIRED,
                "nonempty label-native six-role checkpoint decode requires an "
                "exact expected snapshot hash",
            )
        if (
            expected_snapshot_hash is not None
            and restored.snapshot.snapshot_hash != expected_snapshot_hash
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.HASH_MISMATCH,
                "checkpoint expected label-native six-role snapshot hash does not match",
            )
    elif type(restored) is V1FQRAdaptiveRankCheckpointBundle:
        chart_identity_hash = None
        branch_family_registration_hash = None
        current_mask_hash = restored.current_mask_hash
        proposal_generation = restored.proposal_generation
        route_axis_hash = restored.route_axis_hash
        route_count = restored.route_count
        if any(
            value is not None
            for value in (
                expected_chart_identity_hash,
                expected_branch_family_registration_hash,
                expected_pre_c_graph_hash,
                expected_branch_family_design_hash,
                expected_snapshot_hash,
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "C/branch/chart restore coordinates were supplied for an inert "
                "adaptive-rank D checkpoint",
            )
    else:
        geometry = restored.dual_chart_geometry
        chart_identity_hash = restored.chart_identity_hash
        branch_family_registration_hash = restored.branch_family_registration_hash
        current_mask_hash = geometry.current_mask_hash
        proposal_generation = geometry.proposal_generation
        route_axis_hash = geometry.route_axis_hash
        route_count = geometry.route_count
        if any(
            value is not None
            for value in (
                expected_pre_c_graph_hash,
                expected_branch_family_design_hash,
                expected_snapshot_hash,
            )
        ):
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                "shared-C restore coordinates were supplied for a legacy checkpoint root",
            )
    expected_coordinates = (
        (
            expected_chart_identity_hash,
            chart_identity_hash,
            "chart identity",
        ),
        (
            expected_current_mask_hash,
            current_mask_hash,
            "current mask",
        ),
        (
            expected_proposal_generation,
            proposal_generation,
            "proposal generation",
        ),
        (
            expected_route_axis_hash,
            route_axis_hash,
            "route axis",
        ),
        (
            expected_route_count,
            route_count,
            "route count",
        ),
        (
            expected_branch_family_registration_hash,
            branch_family_registration_hash,
            "branch family registration",
        ),
        (
            expected_pre_c_graph_hash,
            (
                restored.pre_c_graph_hash
                if type(restored) is V1FQRInFlightSharedCCheckpointBundle
                else None
            ),
            "pre-C graph",
        ),
        (
            expected_branch_family_design_hash,
            (
                restored.branch_family_design_hash
                if type(restored) is V1FQRInFlightSharedCCheckpointBundle
                else (
                    restored.design_hash
                    if type(restored)
                    is V1LabelNativeSixRoleInFlightCCheckpointBundle
                    else None
                )
            ),
            "branch-family design",
        ),
    )
    for expected, observed, name in expected_coordinates:
        if expected is not None and expected != observed:
            _refuse(
                V1FQRCheckpointRefusalCode.STALE_COORDINATES,
                f"checkpoint {name} is stale",
            )
    return restored


def v1_fqr_checkpoint_payload_hash(payload: bytes) -> str:
    """Return the exact framed-byte identity; this grants no decode authority."""

    if type(payload) is not bytes:
        raise TypeError("checkpoint payload must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CANONICAL_FQR_BRANCH_IDS",
    "FQR_ADAPTIVE_RANK_CHECKPOINT_SCHEMA",
    "FQR_ADAPTIVE_RANK_REPLAY_PROJECTION_SCHEMA",
    "FQR_CHECKPOINT_AUTHORITY_SCOPE",
    "FQR_CHECKPOINT_BRANCH_SCHEMA",
    "FQR_CHECKPOINT_QUEUE_PROJECTION_SCHEMA",
    "FQR_CHECKPOINT_RECEIPT_PROJECTION_SCHEMA",
    "FQR_CHECKPOINT_SCHEMA",
    "FQR_COMPACT_CHECKPOINT_SCHEMA",
    "FQR_COMPACT_FAMILY_QUEUE_PROJECTION_V2_SCHEMA",
    "FQR_COMPACT_NEGATIVE_BRANCH_PROJECTION_SCHEMA",
    "FQR_COMPACT_REPLAY_REQUIREMENT",
    "FQR_INFLIGHT_C_CHECKPOINT_AUTHORITY_SCOPE",
    "FQR_INFLIGHT_C_CHECKPOINT_SCHEMA",
    "FQR_LABEL_NATIVE_SIX_ROLE_INFLIGHT_C_CHECKPOINT_AUTHORITY_SCOPE",
    "FQR_LABEL_NATIVE_SIX_ROLE_INFLIGHT_C_CHECKPOINT_SCHEMA",
    "V1CompactFQRBranchCheckpoint",
    "V1CompactFQRNegativeBranchCheckpoint",
    "V1CompactFQRBranchFamilyQueueCheckpoint",
    "V1CompactFQRCheckpointBundle",
    "V1CompactFQRQueueEntryCheckpoint",
    "V1CompactFQRSourceAliasCheckpoint",
    "V1FQRAdaptiveRankCheckpointBundle",
    "V1FQRAdaptiveRankReplayCheckpoint",
    "V1FQRCandidateQueueCheckpoint",
    "V1FQRCertificationReceiptCheckpoint",
    "V1FQRAuthorizedAcquisitionCheckpoint",
    "V1FQRCheckpointBranch",
    "V1FQRCheckpointBundle",
    "V1FQRCheckpointRefusal",
    "V1FQRCheckpointRefusalCode",
    "V1FQRInFlightSharedCCheckpointBundle",
    "V1LabelNativeSixRoleInFlightCCheckpointBundle",
    "V1FQRPreCDominanceCheckpoint",
    "V1FQRPreFrozenFamilyCheckpoint",
    "decode_v1_fqr_checkpoint",
    "encode_v1_fqr_checkpoint",
    "v1_fqr_checkpoint_payload_hash",
    "v1_fqr_inflight_pre_c_graph_hash",
]
