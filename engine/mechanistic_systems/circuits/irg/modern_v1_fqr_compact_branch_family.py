"""Detached compact F/Q/R queue authorization across both KKT branches.

The relaxed KKT winner is not authoritative after finite-direction C-stream
certification.  This module therefore materializes the complete union of
*explicitly* upper-certified endpoints from every retained pre-C branch.  It
does not infer prefix feasibility: nonmonotone certified rungs remain in the
queue.  Exact duplicate packed physical actions are represented once while
all independently certified source coordinates remain content-addressed
aliases.  Because those aliases belong to one registered simultaneous F/Q/R
event, the physical entry carries their interval intersection; an empty
intersection or unequal registered resource fails closed.

The resulting authorization is a bounded, detached carrier.  It retains only
packed actions, compact F/Q/R interval/burden scalars, and source hashes.  It
is frozen before calibration K and oracle O and grants neither of them—nor
commit, terminal validation, or a Pareto claim—any authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral, Real
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_directional_remainder import (
    V1DirectionalRemainderReceipt,
    V1DirectionalSecantRecord,
)
from .modern_v1_fqr_branch_family import (
    V1FQRBranchFamilyRefusal,
    V1FQRBranchFamilyRefusalCode,
    V1FQRKKTBranchFrozenBinding,
    V1FQRKKTPreFrozenBranchFamily,
)
from .modern_v1_fqr_certification import (
    V1FQRCertificationDecision,
    V1FQRCertificationReceipt,
)
from .modern_v1_fqr_compact_certification import (
    V1CompactFQRCertificationReceipt,
    V1CompactFQRDecision,
    V1CompactFQREndpointCertificate,
    V1CompactFQRPreCResourceRegistration,
    V1CompactFQRProposalAuthorization,
    V1CompactFQRRefusal,
    V1CompactFQRRefusalCode,
)
from .modern_v1_fqr_device_shell_materialized_family import (
    V1FQRDeviceShellPackedBranchBinding,
    V1FQRDeviceShellPackedFamilyRefusal,
    V1FQRDeviceShellPackedFamilyRefusalCode,
    V1FQRDeviceShellPackedPreFrozenBranchFamily,
)
from .modern_v1_fqr_sequential_acquisition import (
    V1FQRDecisionReadinessReceipt,
    V1FQRDeterministicUncertaintyReceipt,
    V1FQRSequentialAcquisitionReceipt,
)
from .modern_v1_ladder_statistics import (
    V1FrozenLadderCertificationReceipt,
    V1FrozenLadderDecision,
)
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch


COMPACT_FQR_BRANCH_EVIDENCE_BINDING_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_branch_evidence_binding_v1"
)
COMPACT_FQR_BRANCH_NEGATIVE_EVIDENCE_BINDING_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_branch_negative_evidence_binding_v1"
)
COMPACT_FQR_BRANCH_FAMILY_SOURCE_ALIAS_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_branch_family_source_alias_v1"
)
COMPACT_FQR_BRANCH_FAMILY_QUEUE_ENTRY_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_branch_family_queue_entry_v1"
)
COMPACT_FQR_BRANCH_FAMILY_AUTHORIZATION_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_branch_family_queue_authorization_v1"
)
COMPACT_FQR_BRANCH_FAMILY_AUTHORIZATION_V2_SCHEMA: Final = (
    "partition_free_v1_compact_fqr_branch_family_queue_authorization_v2"
)
COMPACT_FQR_BRANCH_FAMILY_SELECTION_POLICY: Final = (
    "negative_registered_resource_then_branch_value_then_packed_action_hash_"
    "then_endpoint_index_with_exact_physical_duplicate_quotient_v1"
)
COMPACT_FQR_BRANCH_FAMILY_AUTHORITY_SCOPE: Final = (
    "bounded_prefrozen_cross_branch_compact_FQR_proposal_queue_materialization_"
    "only_no_K_O_commit_terminal_or_pareto_authority_v1"
)
COMPACT_FQR_BRANCH_FAMILY_MAX_SOURCE_ENDPOINTS: Final = 4096

_EVIDENCE_PREFIX: Final = "modern-v1-compact-fqr-branch-evidence-"
_NEGATIVE_EVIDENCE_PREFIX: Final = (
    "modern-v1-compact-fqr-branch-negative-evidence-"
)
_ALIAS_PREFIX: Final = "modern-v1-compact-fqr-branch-source-alias-"
_ENTRY_PREFIX: Final = "modern-v1-compact-fqr-branch-queue-entry-"
_AUTHORIZATION_PREFIX: Final = "modern-v1-compact-fqr-branch-family-queue-"

_EXACT_PREFROZEN_FAMILY_TYPES: Final = (
    V1FQRKKTPreFrozenBranchFamily,
    V1FQRDeviceShellPackedPreFrozenBranchFamily,
)
_EXACT_BRANCH_BINDING_TYPES: Final = (
    V1FQRKKTBranchFrozenBinding,
    V1FQRDeviceShellPackedBranchBinding,
)


class V1CompactFQRBranchFamilyRefusalCode(str, Enum):
    """Stable fail-closed reasons at the compact cross-branch boundary."""

    INVALID_TYPE = "invalid_type"
    FQ_ONLY_EVIDENCE = "fq_only_evidence"
    RECORD_RETAINING_EVIDENCE = "record_retaining_evidence"
    RAW_RECORD_EVIDENCE = "raw_record_evidence"
    WRONG_EVIDENCE_FAMILY = "wrong_evidence_family"
    HASH_MISMATCH = "hash_mismatch"
    COORDINATE_MISMATCH = "coordinate_mismatch"
    STALE_CHART = "stale_chart"
    POST_SELECTION = "post_selection"
    AMBIGUOUS_EVIDENCE = "ambiguous_evidence"
    MISSING_BRANCH = "missing_branch"
    INVALID_DOMINANCE = "invalid_dominance"
    NO_CERTIFIED_ENDPOINT = "no_certified_endpoint"
    QUEUE_BOUND_EXCEEDED = "queue_bound_exceeded"
    DUPLICATE_ACTION_CONFLICT = "duplicate_action_conflict"
    QUEUE_HEAD_MISMATCH = "queue_head_mismatch"
    AUTHORITY_VIOLATION = "authority_violation"


class V1CompactFQRBranchFamilyRefusal(ValueError):
    """Typed refusal before a compact cross-branch queue can exist."""

    def __init__(
        self,
        code: V1CompactFQRBranchFamilyRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1CompactFQRBranchFamilyRefusalCode,
    message: str,
) -> NoReturn:
    raise V1CompactFQRBranchFamilyRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be finite and nonnegative",
        )
    return result


def _finite_positive(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result == 0.0:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _direct_mask_hash(mask: object, *, name: str) -> str:
    if type(mask) is not HardMask:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
            f"{name} must be an exact HardMask",
        )
    if type(mask.packed_bits) is not bytes:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} packed storage must be immutable bytes",
        )
    route_count = _positive_int(mask.route_count, name=f"{name} route count")
    route_axis_hash = _identity(
        mask.route_axis_hash,
        name=f"{name} route axis hash",
    )
    expected_bytes = (route_count + 7) // 8
    if len(mask.packed_bits) != expected_bytes:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} packed byte length differs from its route count",
        )
    padding = expected_bytes * 8 - route_count
    if padding and mask.packed_bits[-1] & ((1 << padding) - 1):
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"{name} has nonzero padding bits",
        )
    return content_hash(
        {
            "route_axis_hash": route_axis_hash,
            "route_count": route_count,
            "packed_bits": mask.packed_bits,
        },
        prefix="irg-mask-",
    )


def _classify_compact_substitution(value: object) -> None:
    if type(value) in {
        V1FrozenLadderCertificationReceipt,
        V1FrozenLadderDecision,
    }:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.FQ_ONLY_EVIDENCE,
            "F/Q-only evidence cannot enter the compact F/Q/R branch seam",
        )
    if type(value) in {
        V1FQRCertificationReceipt,
        V1FQRCertificationDecision,
    }:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.RECORD_RETAINING_EVIDENCE,
            "record-retaining F/Q/R evidence cannot impersonate compact evidence",
        )
    if type(value) in {
        V1DirectionalRemainderReceipt,
        V1DirectionalSecantRecord,
    }:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.RAW_RECORD_EVIDENCE,
            "raw secant or remainder evidence is forbidden at this boundary",
        )
    if type(value) in {
        V1FQRSequentialAcquisitionReceipt,
        V1FQRDeterministicUncertaintyReceipt,
        V1FQRDecisionReadinessReceipt,
    }:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.WRONG_EVIDENCE_FAMILY,
            "a sequential source receipt is not a compact certification receipt",
        )


def _translate_source_refusal(error: Exception) -> NoReturn:
    if isinstance(error, V1CompactFQRRefusal):
        mapping = {
            V1CompactFQRRefusalCode.INVALID_TYPE: (
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE
            ),
            V1CompactFQRRefusalCode.FQ_ONLY_EVIDENCE: (
                V1CompactFQRBranchFamilyRefusalCode.FQ_ONLY_EVIDENCE
            ),
            V1CompactFQRRefusalCode.RAW_RECORD_EVIDENCE: (
                V1CompactFQRBranchFamilyRefusalCode.RAW_RECORD_EVIDENCE
            ),
            V1CompactFQRRefusalCode.HASH_MISMATCH: (
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH
            ),
            V1CompactFQRRefusalCode.COORDINATE_MISMATCH: (
                V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH
            ),
            V1CompactFQRRefusalCode.STALE_CHART: (
                V1CompactFQRBranchFamilyRefusalCode.STALE_CHART
            ),
            V1CompactFQRRefusalCode.POST_SELECTION: (
                V1CompactFQRBranchFamilyRefusalCode.POST_SELECTION
            ),
            V1CompactFQRRefusalCode.AMBIGUOUS_EVIDENCE: (
                V1CompactFQRBranchFamilyRefusalCode.AMBIGUOUS_EVIDENCE
            ),
            V1CompactFQRRefusalCode.UNRESOLVED_DETERMINISTIC: (
                V1CompactFQRBranchFamilyRefusalCode.AMBIGUOUS_EVIDENCE
            ),
            V1CompactFQRRefusalCode.RESOURCE_DRIFT: (
                V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH
            ),
            V1CompactFQRRefusalCode.NO_CERTIFIED_ENDPOINT: (
                V1CompactFQRBranchFamilyRefusalCode.NO_CERTIFIED_ENDPOINT
            ),
            V1CompactFQRRefusalCode.AUTHORITY_VIOLATION: (
                V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION
            ),
            V1CompactFQRRefusalCode.LEGACY_ONE_SIDED_EVIDENCE: (
                V1CompactFQRBranchFamilyRefusalCode.WRONG_EVIDENCE_FAMILY
            ),
        }
        _refuse(mapping[error.code], f"invalid compact source graph: {error}")
    if isinstance(error, V1FQRBranchFamilyRefusal):
        mapping = {
            V1FQRBranchFamilyRefusalCode.INVALID_TYPE: (
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE
            ),
            V1FQRBranchFamilyRefusalCode.FQ_ONLY_EVIDENCE: (
                V1CompactFQRBranchFamilyRefusalCode.FQ_ONLY_EVIDENCE
            ),
            V1FQRBranchFamilyRefusalCode.COORDINATE_MISMATCH: (
                V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH
            ),
            V1FQRBranchFamilyRefusalCode.HASH_MISMATCH: (
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH
            ),
            V1FQRBranchFamilyRefusalCode.STALE_CHART: (
                V1CompactFQRBranchFamilyRefusalCode.STALE_CHART
            ),
            V1FQRBranchFamilyRefusalCode.POST_SELECTION: (
                V1CompactFQRBranchFamilyRefusalCode.POST_SELECTION
            ),
            V1FQRBranchFamilyRefusalCode.UNSTABLE_DECISION: (
                V1CompactFQRBranchFamilyRefusalCode.AMBIGUOUS_EVIDENCE
            ),
            V1FQRBranchFamilyRefusalCode.MISSING_BRANCH: (
                V1CompactFQRBranchFamilyRefusalCode.MISSING_BRANCH
            ),
            V1FQRBranchFamilyRefusalCode.INVALID_DOMINANCE: (
                V1CompactFQRBranchFamilyRefusalCode.INVALID_DOMINANCE
            ),
            V1FQRBranchFamilyRefusalCode.NO_CERTIFIED_ENDPOINT: (
                V1CompactFQRBranchFamilyRefusalCode.NO_CERTIFIED_ENDPOINT
            ),
            V1FQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION: (
                V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION
            ),
        }
        _refuse(mapping[error.code], f"invalid pre-C branch family: {error}")
    if isinstance(error, V1FQRDeviceShellPackedFamilyRefusal):
        mapping = {
            V1FQRDeviceShellPackedFamilyRefusalCode.INVALID_TYPE: (
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE
            ),
            V1FQRDeviceShellPackedFamilyRefusalCode.COORDINATE_MISMATCH: (
                V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH
            ),
            V1FQRDeviceShellPackedFamilyRefusalCode.IDENTITY_MISMATCH: (
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH
            ),
            V1FQRDeviceShellPackedFamilyRefusalCode.POST_OUTCOME: (
                V1CompactFQRBranchFamilyRefusalCode.POST_SELECTION
            ),
            V1FQRDeviceShellPackedFamilyRefusalCode.AUTHORITY_VIOLATION: (
                V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION
            ),
        }
        _refuse(
            mapping[error.code],
            f"invalid packed device-shell pre-C family: {error}",
        )
    _refuse(
        V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
        f"source integrity replay failed: {error}",
    )


def _validate_packed_action(
    *,
    packed: object,
    parent: HardMask,
    expected_packed_hash: str,
    expected_resulting_mask_hash: str | None = None,
) -> tuple[V1PackedRemovalAction, str]:
    if type(packed) is not V1PackedRemovalAction:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
            "a compact queue endpoint requires an exact packed action",
        )
    child_hash = _direct_mask_hash(
        packed.resulting_mask,
        name="compact queue resulting mask",
    )
    if (
        packed.resulting_mask.route_axis_hash != parent.route_axis_hash
        or packed.resulting_mask.route_count != parent.route_count
        or packed.action_hash != expected_packed_hash
        or (
            expected_resulting_mask_hash is not None
            and child_hash != expected_resulting_mask_hash
        )
    ):
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
            "packed action differs from its registered physical identity",
        )
    try:
        removed_count, removed_hash = (
            packed.resulting_mask.removed_index_count_and_hash_from(parent)
        )
    except (TypeError, ValueError) as error:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            f"packed action is not an irreversible child of the live parent: {error}",
        )
    if (
        removed_count != packed.removed_count
        or removed_hash != packed.removed_route_index_hash
    ):
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
            "packed action delta differs from its direct parent-to-child replay",
        )
    return packed, child_hash


def _certificate_scalars(
    certificate: V1CompactFQREndpointCertificate,
) -> tuple[float, ...]:
    return (
        certificate.f_lower,
        certificate.f_upper,
        certificate.q_lower,
        certificate.q_upper,
        certificate.r_half_lower,
        certificate.r_half_upper,
        certificate.r_one_lower,
        certificate.r_one_upper,
        certificate.half_response_lower,
        certificate.half_response_upper,
        certificate.one_response_lower,
        certificate.one_response_upper,
    )


@dataclass(frozen=True, slots=True)
class V1CompactFQRBranchEvidenceBinding:
    """One exact compact F/Q/R graph bound to one retained KKT branch."""

    prefrozen_family_hash: str
    branch: V1RelaxedKKTBranch
    branch_binding: (
        V1FQRKKTBranchFrozenBinding | V1FQRDeviceShellPackedBranchBinding
    ) = field(
        repr=False,
        compare=False,
    )
    certification_receipt: V1CompactFQRCertificationReceipt = field(
        repr=False,
        compare=False,
    )
    decision: V1CompactFQRDecision = field(repr=False, compare=False)
    proposal_authorization: V1CompactFQRProposalAuthorization = field(
        repr=False,
        compare=False,
    )
    resource_registration: V1CompactFQRPreCResourceRegistration = field(
        repr=False,
        compare=False,
    )
    compact_only: bool = True
    raw_records_retained: bool = False
    postselected: bool = False
    downstream_authority: bool = False
    schema: str = COMPACT_FQR_BRANCH_EVIDENCE_BINDING_SCHEMA
    _evidence_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prefrozen_family_hash",
            _identity(self.prefrozen_family_hash, name="pre-frozen family hash"),
        )
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
                "compact branch binding requires the exact KKT branch enum",
            )
        for value in (
            self.certification_receipt,
            self.decision,
            self.proposal_authorization,
            self.resource_registration,
        ):
            _classify_compact_substitution(value)
        expected_types = (
            (self.certification_receipt, V1CompactFQRCertificationReceipt),
            (self.decision, V1CompactFQRDecision),
            (
                self.proposal_authorization,
                V1CompactFQRProposalAuthorization,
            ),
            (
                self.resource_registration,
                V1CompactFQRPreCResourceRegistration,
            ),
        )
        if (
            type(self.branch_binding) not in _EXACT_BRANCH_BINDING_TYPES
            or any(
                type(value) is not expected
                for value, expected in expected_types
            )
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
                "compact branch evidence requires exact versioned source types",
            )
        if (
            self.compact_only is not True
            or self.raw_records_retained is not False
            or self.postselected is not False
            or self.downstream_authority is not False
            or self.schema != COMPACT_FQR_BRANCH_EVIDENCE_BINDING_SCHEMA
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "compact branch evidence crossed its pre-oracle boundary",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "prefrozen_family_hash": self.prefrozen_family_hash,
            "branch": self.branch,
            "branch_binding_hash": self.branch_binding.binding_hash,
            "certification_receipt_hash": self.certification_receipt.receipt_hash,
            "decision_hash": self.decision.decision_hash,
            "proposal_authorization_hash": (
                self.proposal_authorization.authorization_hash
            ),
            "resource_registration_hash": (
                self.resource_registration.registration_hash
            ),
            "compact_only": self.compact_only,
            "raw_records_retained": self.raw_records_retained,
            "postselected": self.postselected,
            "downstream_authority": self.downstream_authority,
        }

    @property
    def evidence_hash(self) -> str:
        cached = self._evidence_hash
        return self.assert_integrity() if cached is None else cached

    @property
    def upper_certified_endpoint_indices(self) -> tuple[int, ...]:
        return tuple(
            value.endpoint_index
            for value in self.certification_receipt.endpoint_certificates
            if value.upper_certified
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.compact_only is not True
            or self.raw_records_retained is not False
            or self.postselected is not False
            or self.downstream_authority is not False
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "compact branch evidence crossed its pre-oracle boundary",
            )
        try:
            binding_hash = self.branch_binding.assert_integrity()
            receipt_hash = self.certification_receipt.assert_integrity()
            decision_hash = self.decision.assert_integrity()
            authorization_hash = self.proposal_authorization.assert_integrity()
            resource_hash = self.resource_registration.assert_integrity()
        except Exception as error:
            _translate_source_refusal(error)

        binding = self.branch_binding
        receipt = self.certification_receipt
        decision = self.decision
        authorization = self.proposal_authorization
        resources = self.resource_registration
        ladder = binding.frozen_ladder
        parent = ladder.parent_mask
        if type(parent) is not HardMask:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
                "branch ladder lost its exact parent mask",
            )
        parent_hash = _direct_mask_hash(parent, name="branch ladder parent mask")
        endpoint_count = len(ladder.actions)
        if endpoint_count > COMPACT_FQR_BRANCH_FAMILY_MAX_SOURCE_ENDPOINTS:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.QUEUE_BOUND_EXCEEDED,
                "one branch exceeds the fixed compact endpoint bound",
            )

        common = (
            self.branch is binding.branch
            and binding_hash == binding.binding_hash
            and receipt_hash == receipt.receipt_hash
            and decision_hash == decision.decision_hash
            and authorization_hash == authorization.authorization_hash
            and resource_hash == resources.registration_hash
            and receipt.ladder_hash == ladder.ladder_hash
            and decision.ladder_hash == ladder.ladder_hash
            and authorization.ladder_hash == ladder.ladder_hash
            and resources.ladder_hash == ladder.ladder_hash
            and receipt.chart_identity_hash == binding.chart_identity_hash
            and decision.chart_identity_hash == binding.chart_identity_hash
            and authorization.chart_identity_hash == binding.chart_identity_hash
            and resources.chart_identity_hash == binding.chart_identity_hash
            and receipt.current_mask_hash == parent_hash
            and decision.current_mask_hash == parent_hash
            and authorization.current_mask_hash == parent_hash
            and resources.current_mask_hash == parent_hash
            and binding.current_mask_hash == parent_hash
            and receipt.proposal_generation == binding.proposal_generation
            and decision.proposal_generation == binding.proposal_generation
            and authorization.proposal_generation == binding.proposal_generation
            and resources.proposal_generation == binding.proposal_generation
            and receipt.route_axis_hash == binding.route_axis_hash
            and decision.route_axis_hash == binding.route_axis_hash
            and authorization.route_axis_hash == binding.route_axis_hash
            and resources.route_axis_hash == binding.route_axis_hash
            and receipt.route_count == binding.route_count
            and authorization.route_count == binding.route_count
            and resources.route_count == binding.route_count
            and receipt.endpoint_action_hashes == ladder.action_hashes
            and resources.endpoint_action_hashes == ladder.action_hashes
            and receipt.resource_registration_hash == resource_hash
            and decision.resource_registration_hash == resource_hash
            and authorization.resource_registration_hash == resource_hash
            and decision.certification_receipt_hash == receipt_hash
            and authorization.certification_receipt_hash == receipt_hash
            and authorization.decision_hash == decision_hash
            and decision.acquisition_design_hash == receipt.acquisition_design_hash
            and authorization.acquisition_design_hash
            == receipt.acquisition_design_hash
            and decision.acquisition_receipt_hash
            == receipt.acquisition_receipt_hash
            and authorization.acquisition_receipt_hash
            == receipt.acquisition_receipt_hash
            and decision.deterministic_uncertainty_receipt_hash
            == receipt.deterministic_uncertainty_receipt_hash
            and authorization.deterministic_uncertainty_receipt_hash
            == receipt.deterministic_uncertainty_receipt_hash
            and decision.readiness_receipt_hash == receipt.readiness_receipt_hash
            and authorization.readiness_receipt_hash
            == receipt.readiness_receipt_hash
            and resources.acquisition_design_hash
            == receipt.acquisition_design_hash
            and authorization.resource_units == resources.resource_units
            and authorization.resource_source_hash
            == resources.resource_source_hash
            and tuple(row[2] for row in resources.resource_by_endpoint)
            == binding.endpoint_registered_resources
        )
        if not common:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "compact receipt/decision/authorization/resource axes differ from "
                "their frozen branch",
            )

        certificates = receipt.endpoint_certificates
        if len(certificates) != endpoint_count:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "compact certificate axis differs from the frozen branch ladder",
            )
        certified = tuple(
            value.endpoint_index for value in certificates if value.upper_certified
        )
        if not certified:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.NO_CERTIFIED_ENDPOINT,
                "a retained branch has no upper-certified compact endpoint",
            )
        winner = max(certified)
        if (
            decision.status != "stable_endpoint"
            or receipt.sampling_decision != "sampling_stable_endpoint"
            or decision.selected_endpoint_index != winner
            or receipt.selected_endpoint_index != winner
            or authorization.selected_endpoint_index != winner
            or decision.selected_action_hash != ladder.action_hashes[winner - 1]
            or receipt.selected_action_hash != ladder.action_hashes[winner - 1]
            or authorization.selected_action_hash != ladder.action_hashes[winner - 1]
            or not certificates[winner - 1].upper_certified
            or any(not value.lower_refuted for value in certificates[winner:])
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.AMBIGUOUS_EVIDENCE,
                "per-branch compact winner does not replay from the complete "
                "upper-certified/lower-refuted endpoint axis",
            )
        selected_frozen = ladder.actions[winner - 1]
        selected_packed = selected_frozen.packed_removal_action
        expected_packed_hash = binding.endpoint_packed_action_hashes[winner - 1]
        _validate_packed_action(
            packed=selected_packed,
            parent=parent,
            expected_packed_hash=expected_packed_hash,
            expected_resulting_mask_hash=(
                authorization.selected_resulting_mask_hash
            ),
        )
        if (
            authorization.selected_packed_action.action_hash
            != expected_packed_hash
            or authorization.selected_resource
            != binding.endpoint_registered_resources[winner - 1]
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.QUEUE_HEAD_MISMATCH,
                "the compact authorization is not the branch's greatest-resource "
                "upper-certified endpoint",
            )

        observed = content_hash(self._payload(), prefix=_EVIDENCE_PREFIX)
        cached = self._evidence_hash
        if cached is not None and cached != observed:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "compact branch evidence was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "compact branch evidence hash does not match",
            )
        object.__setattr__(self, "_evidence_hash", observed)
        return observed


def bind_v1_compact_fqr_branch_evidence(
    *,
    prefrozen_family: object,
    branch: object,
    certification_receipt: object,
    decision: object,
    proposal_authorization: object,
    resource_registration: object,
) -> V1CompactFQRBranchEvidenceBinding:
    """Bind one exact compact certification graph to one family branch."""

    for value in (
        certification_receipt,
        decision,
        proposal_authorization,
        resource_registration,
    ):
        _classify_compact_substitution(value)
    if type(prefrozen_family) not in _EXACT_PREFROZEN_FAMILY_TYPES:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
            "compact branch binding requires the exact pre-frozen family",
        )
    if type(branch) is not V1RelaxedKKTBranch:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
            "compact branch binding requires the exact KKT branch enum",
        )
    try:
        family_hash = prefrozen_family.assert_integrity()
        binding = prefrozen_family.binding_for_branch(branch)
    except Exception as error:
        _translate_source_refusal(error)
    return V1CompactFQRBranchEvidenceBinding(
        prefrozen_family_hash=family_hash,
        branch=branch,
        branch_binding=binding,
        certification_receipt=certification_receipt,
        decision=decision,
        proposal_authorization=proposal_authorization,
        resource_registration=resource_registration,
    )


@dataclass(frozen=True, slots=True)
class V1CompactFQRBranchNegativeEvidenceBinding:
    """Complete compact proof that one retained branch has no safe endpoint.

    A stable-negative branch is still part of the exact two-branch convex
    union.  It contributes provenance and branch-completeness evidence, but it
    contributes no proposal-queue entry and carries no downstream authority.
    """

    prefrozen_family_hash: str
    branch: V1RelaxedKKTBranch
    branch_binding: (
        V1FQRKKTBranchFrozenBinding | V1FQRDeviceShellPackedBranchBinding
    ) = field(
        repr=False,
        compare=False,
    )
    certification_receipt: V1CompactFQRCertificationReceipt = field(
        repr=False,
        compare=False,
    )
    decision: V1CompactFQRDecision = field(repr=False, compare=False)
    resource_registration: V1CompactFQRPreCResourceRegistration = field(
        repr=False,
        compare=False,
    )
    compact_only: bool = True
    stable_no_endpoint: bool = True
    negative_evidence_only: bool = True
    raw_records_retained: bool = False
    postselected: bool = False
    proposal_queue_entry_authority: bool = False
    downstream_authority: bool = False
    schema: str = COMPACT_FQR_BRANCH_NEGATIVE_EVIDENCE_BINDING_SCHEMA
    _evidence_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prefrozen_family_hash",
            _identity(self.prefrozen_family_hash, name="pre-frozen family hash"),
        )
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
                "compact negative branch binding requires the exact KKT branch enum",
            )
        for value in (
            self.certification_receipt,
            self.decision,
            self.resource_registration,
        ):
            _classify_compact_substitution(value)
        expected_types = (
            (self.certification_receipt, V1CompactFQRCertificationReceipt),
            (self.decision, V1CompactFQRDecision),
            (
                self.resource_registration,
                V1CompactFQRPreCResourceRegistration,
            ),
        )
        if (
            type(self.branch_binding) not in _EXACT_BRANCH_BINDING_TYPES
            or any(
                type(value) is not expected
                for value, expected in expected_types
            )
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
                "compact negative evidence requires exact versioned source types",
            )
        if (
            self.compact_only is not True
            or self.stable_no_endpoint is not True
            or self.negative_evidence_only is not True
            or self.raw_records_retained is not False
            or self.postselected is not False
            or self.proposal_queue_entry_authority is not False
            or self.downstream_authority is not False
            or self.schema
            != COMPACT_FQR_BRANCH_NEGATIVE_EVIDENCE_BINDING_SCHEMA
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "compact negative evidence crossed its no-proposal boundary",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "prefrozen_family_hash": self.prefrozen_family_hash,
            "branch": self.branch,
            "branch_binding_hash": self.branch_binding.binding_hash,
            "certification_receipt_hash": self.certification_receipt.receipt_hash,
            "decision_hash": self.decision.decision_hash,
            "resource_registration_hash": (
                self.resource_registration.registration_hash
            ),
            "compact_only": self.compact_only,
            "stable_no_endpoint": self.stable_no_endpoint,
            "negative_evidence_only": self.negative_evidence_only,
            "raw_records_retained": self.raw_records_retained,
            "postselected": self.postselected,
            "proposal_queue_entry_authority": (
                self.proposal_queue_entry_authority
            ),
            "downstream_authority": self.downstream_authority,
        }

    @property
    def evidence_hash(self) -> str:
        cached = self._evidence_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.compact_only is not True
            or self.stable_no_endpoint is not True
            or self.negative_evidence_only is not True
            or self.raw_records_retained is not False
            or self.postselected is not False
            or self.proposal_queue_entry_authority is not False
            or self.downstream_authority is not False
            or self.schema
            != COMPACT_FQR_BRANCH_NEGATIVE_EVIDENCE_BINDING_SCHEMA
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "compact negative evidence crossed its no-proposal boundary",
            )
        try:
            binding_hash = self.branch_binding.assert_integrity()
            receipt_hash = self.certification_receipt.assert_integrity()
            decision_hash = self.decision.assert_integrity()
            resource_hash = self.resource_registration.assert_integrity()
        except Exception as error:
            _translate_source_refusal(error)

        binding = self.branch_binding
        receipt = self.certification_receipt
        decision = self.decision
        resources = self.resource_registration
        ladder = binding.frozen_ladder
        parent = ladder.parent_mask
        if type(parent) is not HardMask:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
                "negative branch ladder lost its exact parent mask",
            )
        parent_hash = _direct_mask_hash(parent, name="negative branch parent mask")
        certificates = receipt.endpoint_certificates
        endpoint_count = len(ladder.actions)
        if endpoint_count > COMPACT_FQR_BRANCH_FAMILY_MAX_SOURCE_ENDPOINTS:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.QUEUE_BOUND_EXCEEDED,
                "one negative branch exceeds the compact endpoint bound",
            )
        common = (
            self.branch is binding.branch
            and binding_hash == binding.binding_hash
            and receipt_hash == receipt.receipt_hash
            and decision_hash == decision.decision_hash
            and resource_hash == resources.registration_hash
            and receipt.ladder_hash == ladder.ladder_hash
            and decision.ladder_hash == ladder.ladder_hash
            and resources.ladder_hash == ladder.ladder_hash
            and receipt.chart_identity_hash == binding.chart_identity_hash
            and decision.chart_identity_hash == binding.chart_identity_hash
            and resources.chart_identity_hash == binding.chart_identity_hash
            and receipt.current_mask_hash == parent_hash
            and decision.current_mask_hash == parent_hash
            and resources.current_mask_hash == parent_hash
            and binding.current_mask_hash == parent_hash
            and receipt.proposal_generation == binding.proposal_generation
            and decision.proposal_generation == binding.proposal_generation
            and resources.proposal_generation == binding.proposal_generation
            and receipt.route_axis_hash == binding.route_axis_hash
            and decision.route_axis_hash == binding.route_axis_hash
            and resources.route_axis_hash == binding.route_axis_hash
            and receipt.route_count == binding.route_count
            and resources.route_count == binding.route_count
            and receipt.endpoint_action_hashes == ladder.action_hashes
            and resources.endpoint_action_hashes == ladder.action_hashes
            and receipt.resource_registration_hash == resource_hash
            and decision.resource_registration_hash == resource_hash
            and decision.certification_receipt_hash == receipt_hash
            and decision.acquisition_design_hash == receipt.acquisition_design_hash
            and decision.acquisition_receipt_hash
            == receipt.acquisition_receipt_hash
            and decision.deterministic_uncertainty_receipt_hash
            == receipt.deterministic_uncertainty_receipt_hash
            and decision.readiness_receipt_hash == receipt.readiness_receipt_hash
            and resources.acquisition_design_hash
            == receipt.acquisition_design_hash
            and tuple(row[2] for row in resources.resource_by_endpoint)
            == binding.endpoint_registered_resources
        )
        if not common:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "negative compact receipt/decision/resource axes differ from "
                "their frozen branch",
            )
        if len(certificates) != endpoint_count:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "negative compact certificate axis differs from its frozen ladder",
            )
        if (
            decision.status != "stable_no_endpoint"
            or receipt.sampling_decision != "sampling_stable_no_endpoint"
            or decision.selected_endpoint_index is not None
            or decision.selected_action_hash is not None
            or receipt.selected_endpoint_index is not None
            or receipt.selected_action_hash is not None
            or any(value.upper_certified for value in certificates)
            or any(not value.lower_refuted for value in certificates)
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.AMBIGUOUS_EVIDENCE,
                "negative branch evidence must lower-refute every frozen endpoint",
            )

        observed = content_hash(
            self._payload(),
            prefix=_NEGATIVE_EVIDENCE_PREFIX,
        )
        cached = self._evidence_hash
        if cached is not None and cached != observed:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "compact negative branch evidence was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "compact negative branch evidence hash does not match",
            )
        object.__setattr__(self, "_evidence_hash", observed)
        return observed


def bind_v1_compact_fqr_branch_negative_evidence(
    *,
    prefrozen_family: object,
    branch: object,
    certification_receipt: object,
    decision: object,
    resource_registration: object,
) -> V1CompactFQRBranchNegativeEvidenceBinding:
    """Bind one exact stable-no-endpoint compact graph to a retained branch."""

    for value in (certification_receipt, decision, resource_registration):
        _classify_compact_substitution(value)
    if type(prefrozen_family) not in _EXACT_PREFROZEN_FAMILY_TYPES:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
            "negative branch binding requires the exact pre-frozen family",
        )
    if type(branch) is not V1RelaxedKKTBranch:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
            "negative branch binding requires the exact KKT branch enum",
        )
    try:
        family_hash = prefrozen_family.assert_integrity()
        binding = prefrozen_family.binding_for_branch(branch)
    except Exception as error:
        _translate_source_refusal(error)
    return V1CompactFQRBranchNegativeEvidenceBinding(
        prefrozen_family_hash=family_hash,
        branch=branch,
        branch_binding=binding,
        certification_receipt=certification_receipt,
        decision=decision,
        resource_registration=resource_registration,
    )


@dataclass(frozen=True, slots=True)
class V1CompactFQRBranchFamilySourceAlias:
    """One independently upper-certified source for a physical queue action."""

    branch: V1RelaxedKKTBranch
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
    schema: str = COMPACT_FQR_BRANCH_FAMILY_SOURCE_ALIAS_SCHEMA
    _alias_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
                "queue alias requires the exact KKT branch enum",
            )
        object.__setattr__(
            self,
            "endpoint_index",
            _positive_int(self.endpoint_index, name="alias endpoint index"),
        )
        for name in (
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
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "registered_resource",
            _finite_positive(
                self.registered_resource,
                name="alias registered resource",
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
                    V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                    "queue alias contains an inverted compact interval",
                )
        if (
            self.independently_upper_certified is not True
            or self.schema != COMPACT_FQR_BRANCH_FAMILY_SOURCE_ALIAS_SCHEMA
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "queue alias lacks independent compact upper certification",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "branch",
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
            )
        }

    @property
    def alias_hash(self) -> str:
        cached = self._alias_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if self.independently_upper_certified is not True:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "queue alias lost independent upper certification",
            )
        observed = content_hash(self._payload(), prefix=_ALIAS_PREFIX)
        cached = self._alias_hash
        if cached is not None and cached != observed:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "compact queue alias was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "compact queue alias hash does not match",
            )
        object.__setattr__(self, "_alias_hash", observed)
        return observed


def _alias_sort_key(
    alias: V1CompactFQRBranchFamilySourceAlias,
) -> tuple[float, str, str, int]:
    return (
        -alias.registered_resource,
        alias.branch.value,
        alias.packed_action_hash,
        alias.endpoint_index,
    )


@dataclass(frozen=True, slots=True)
class V1CompactFQRBranchFamilyQueueEntry:
    """One unique packed action in the immutable pre-K proposal queue."""

    representative_branch: V1RelaxedKKTBranch
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
    source_aliases: tuple[V1CompactFQRBranchFamilySourceAlias, ...]
    proposal_queue_entry_only: bool = True
    schema: str = COMPACT_FQR_BRANCH_FAMILY_QUEUE_ENTRY_SCHEMA
    _entry_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.representative_branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
                "queue entry requires the exact representative branch enum",
            )
        object.__setattr__(
            self,
            "representative_endpoint_index",
            _positive_int(
                self.representative_endpoint_index,
                name="representative endpoint index",
            ),
        )
        for name in (
            "representative_frozen_action_hash",
            "packed_action_hash",
            "parent_mask_hash",
            "resulting_mask_hash",
            "route_axis_hash",
            "resource_units",
            "resource_source_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="queue entry route count"),
        )
        object.__setattr__(
            self,
            "registered_resource",
            _finite_positive(
                self.registered_resource,
                name="queue entry registered resource",
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
        aliases = tuple(self.source_aliases)
        if (
            not aliases
            or any(
                type(value) is not V1CompactFQRBranchFamilySourceAlias
                for value in aliases
            )
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
                "queue entry requires at least one exact compact source alias",
            )
        for alias in aliases:
            alias.assert_integrity()
        if aliases != tuple(sorted(aliases, key=_alias_sort_key)):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.POST_SELECTION,
                "duplicate-action aliases are not in the registered order",
            )
        if len({(value.branch, value.endpoint_index) for value in aliases}) != len(
            aliases
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "a queue entry repeats one branch/endpoint source coordinate",
            )
        representative = aliases[0]
        if (
            any(value.packed_action_hash != self.packed_action_hash for value in aliases)
            or any(
                value.registered_resource != representative.registered_resource
                for value in aliases
            )
            or representative.branch is not self.representative_branch
            or representative.endpoint_index != self.representative_endpoint_index
            or representative.frozen_action_hash
            != self.representative_frozen_action_hash
            or representative.registered_resource != self.registered_resource
            or _intersect_alias_scalars(aliases) != _entry_scalars(self)
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.DUPLICATE_ACTION_CONFLICT,
                "duplicate physical-action aliases disagree on resource identity or "
                "the queue entry is not their interval intersection",
            )
        if type(self.packed_action) is not V1PackedRemovalAction:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
                "queue entry requires one exact packed action",
            )
        child = self.packed_action.resulting_mask
        child_hash = _direct_mask_hash(child, name="detached queue resulting mask")
        if (
            self.packed_action.action_hash != self.packed_action_hash
            or child_hash != self.resulting_mask_hash
            or child.route_axis_hash != self.route_axis_hash
            or child.route_count != self.route_count
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "detached queue packed action differs from its physical identity",
            )
        if (
            self.proposal_queue_entry_only is not True
            or self.schema != COMPACT_FQR_BRANCH_FAMILY_QUEUE_ENTRY_SCHEMA
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "queue entry crossed its proposal-materialization boundary",
            )
        object.__setattr__(self, "source_aliases", aliases)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "representative_branch",
                "representative_endpoint_index",
                "representative_frozen_action_hash",
                "packed_action_hash",
                "parent_mask_hash",
                "resulting_mask_hash",
                "route_axis_hash",
                "route_count",
                "registered_resource",
                "resource_units",
                "resource_source_hash",
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
                "proposal_queue_entry_only",
            )
        } | {
            "packed_action_content_hash": self.packed_action.action_hash,
            "source_alias_hashes": tuple(
                value.alias_hash for value in self.source_aliases
            ),
        }

    @property
    def entry_hash(self) -> str:
        cached = self._entry_hash
        return self.assert_integrity() if cached is None else cached

    @property
    def branch(self) -> V1RelaxedKKTBranch:
        """Canonical representative branch after physical-action quotienting."""

        return self.representative_branch

    @property
    def endpoint_index(self) -> int:
        """Canonical representative endpoint after physical-action quotienting."""

        return self.representative_endpoint_index

    @property
    def current_mask_hash(self) -> str:
        """Compatibility spelling for the entry's exact parent-mask identity."""

        return self.parent_mask_hash

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for value in self.source_aliases:
            value.assert_integrity()
        if self.proposal_queue_entry_only is not True:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "queue entry crossed its proposal-materialization boundary",
            )
        if (
            self.packed_action.action_hash != self.packed_action_hash
            or _direct_mask_hash(
                self.packed_action.resulting_mask,
                name="detached queue resulting mask",
            )
            != self.resulting_mask_hash
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "detached queue packed action was tampered",
            )
        observed = content_hash(self._payload(), prefix=_ENTRY_PREFIX)
        cached = self._entry_hash
        if cached is not None and cached != observed:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "compact queue entry was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "compact queue entry hash does not match",
            )
        object.__setattr__(self, "_entry_hash", observed)
        return observed


def _certificate_scalars_from_alias(
    value: V1CompactFQRBranchFamilySourceAlias,
) -> tuple[float, ...]:
    return (
        value.f_lower,
        value.f_upper,
        value.q_lower,
        value.q_upper,
        value.r_half_lower,
        value.r_half_upper,
        value.r_one_lower,
        value.r_one_upper,
        value.half_response_lower,
        value.half_response_upper,
        value.one_response_lower,
        value.one_response_upper,
    )


def _intersect_alias_scalars(
    aliases: tuple[V1CompactFQRBranchFamilySourceAlias, ...],
) -> tuple[float, ...]:
    """Intersect independently valid simultaneous-event interval receipts."""

    if not aliases:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.DUPLICATE_ACTION_CONFLICT,
            "a physical queue action has no compact source aliases",
        )
    scalar_rows = tuple(_certificate_scalars_from_alias(value) for value in aliases)
    result: list[float] = []
    for offset in range(0, 12, 2):
        lower = max(row[offset] for row in scalar_rows)
        upper = min(row[offset + 1] for row in scalar_rows)
        if lower > upper:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.DUPLICATE_ACTION_CONFLICT,
                "duplicate physical-action F/Q/R intervals have an empty "
                "simultaneous-event intersection",
            )
        result.extend((lower, upper))
    return tuple(result)


def _entry_scalars(value: V1CompactFQRBranchFamilyQueueEntry) -> tuple[float, ...]:
    return (
        value.f_lower,
        value.f_upper,
        value.q_lower,
        value.q_upper,
        value.r_half_lower,
        value.r_half_upper,
        value.r_one_lower,
        value.r_one_upper,
        value.half_response_lower,
        value.half_response_upper,
        value.one_response_lower,
        value.one_response_upper,
    )


def _entry_sort_key(
    entry: V1CompactFQRBranchFamilyQueueEntry,
) -> tuple[float, str, str, int]:
    return (
        -entry.registered_resource,
        entry.representative_branch.value,
        entry.packed_action_hash,
        entry.representative_endpoint_index,
    )


@dataclass(frozen=True, slots=True)
class V1CompactFQRBranchFamilyAuthorization:
    """Detached immutable queue materialized before K/O from compact F/Q/R."""

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
    queue_entries: tuple[V1CompactFQRBranchFamilyQueueEntry, ...]
    source_endpoint_count: int
    certified_alias_count: int
    unique_physical_action_count: int
    maximum_source_endpoint_count: int
    per_branch_winner_coordinates: tuple[tuple[str, int, str, float], ...]
    queue_head_entry_hash: str
    queue_head_branch: V1RelaxedKKTBranch
    queue_head_endpoint_index: int
    queue_head_packed_action_hash: str
    queue_head_registered_resource: float
    source_branch_negative_evidence_hashes: tuple[tuple[str, str], ...] = ()
    selection_policy: str = COMPACT_FQR_BRANCH_FAMILY_SELECTION_POLICY
    proposal_queue_materialization_authority: bool = True
    materialized_before_K: bool = True
    K_outcomes_seen: bool = False
    O_outcomes_seen: bool = False
    queue_mutable_after_K: bool = False
    oracle_conditioned_reordering_authority: bool = False
    backtracking_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_validation_access_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    hard_global_optimality_claim: bool = False
    paired_bce_oracle_still_required: bool = True
    retained_context_ids: int = 0
    retained_record_arrays: int = 0
    retained_route_scale_arrays: int = 0
    authority_scope: str = COMPACT_FQR_BRANCH_FAMILY_AUTHORITY_SCOPE
    schema: str = COMPACT_FQR_BRANCH_FAMILY_AUTHORIZATION_SCHEMA
    _authorization_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
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
            "queue_head_entry_hash",
            "queue_head_packed_action_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(
                self.proposal_generation,
                name="queue proposal generation",
            ),
        )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="queue route count"),
        )
        for name in (
            "source_endpoint_count",
            "certified_alias_count",
            "unique_physical_action_count",
            "maximum_source_endpoint_count",
            "queue_head_endpoint_index",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "queue_head_registered_resource",
            _finite_positive(
                self.queue_head_registered_resource,
                name="queue head registered resource",
            ),
        )
        if type(self.queue_head_branch) is not V1RelaxedKKTBranch:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
                "queue head requires the exact KKT branch enum",
            )
        source_hashes = tuple(self.source_branch_evidence_hashes)
        negative_hashes = tuple(self.source_branch_negative_evidence_hashes)
        if len(source_hashes) not in {1, 2} or len(negative_hashes) > 1:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.MISSING_BRANCH,
                "authorization requires one or two positive branches and at most "
                "one stable-negative branch",
            )
        normalized_groups: list[tuple[tuple[str, str], ...]] = []
        for rows, label in (
            (source_hashes, "positive branch evidence"),
            (negative_hashes, "negative branch evidence"),
        ):
            normalized_rows: list[tuple[str, str]] = []
            for row in rows:
                if len(row) != 2:
                    _refuse(
                        V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                        f"{label} rows must have two fields",
                    )
                try:
                    branch = V1RelaxedKKTBranch(row[0])
                except (TypeError, ValueError):
                    _refuse(
                        V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                        f"{label} names an unknown KKT branch",
                    )
                normalized_rows.append(
                    (branch.value, _identity(row[1], name=f"{label} hash"))
                )
            normalized_groups.append(tuple(normalized_rows))
        normalized_sources, normalized_negatives = normalized_groups
        if (
            normalized_sources != tuple(sorted(normalized_sources))
            or normalized_negatives != tuple(sorted(normalized_negatives))
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.POST_SELECTION,
                "branch evidence hashes are not in branch-value order",
            )
        all_source_branches = tuple(
            branch for branch, _value in normalized_sources + normalized_negatives
        )
        if (
            len(all_source_branches) not in {1, 2}
            or len(set(all_source_branches)) != len(all_source_branches)
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.AMBIGUOUS_EVIDENCE,
                "each retained branch must have exactly one positive-or-negative "
                "evidence object",
            )
        if normalized_negatives:
            if self.schema != COMPACT_FQR_BRANCH_FAMILY_AUTHORIZATION_V2_SCHEMA:
                _refuse(
                    V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                    "stable-negative provenance requires authorization schema v2",
                )
        elif self.schema != COMPACT_FQR_BRANCH_FAMILY_AUTHORIZATION_SCHEMA:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "positive-only authorization must retain schema v1",
            )
        if len(all_source_branches) == 1:
            object.__setattr__(
                self,
                "dominance_certificate_hash",
                _identity(
                    self.dominance_certificate_hash,
                    name="pre-C dominance certificate hash",
                ),
            )
        elif self.dominance_certificate_hash is not None:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
                "a complete two-branch queue cannot carry an omission certificate",
            )
        entries = tuple(self.queue_entries)
        if (
            not entries
            or any(type(value) is not V1CompactFQRBranchFamilyQueueEntry for value in entries)
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
                "authorization requires a nonempty exact compact queue",
            )
        for value in entries:
            value.assert_integrity()
        if entries != tuple(sorted(entries, key=_entry_sort_key)):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.POST_SELECTION,
                "compact branch-family queue is not in its registered order",
            )
        if len({value.packed_action_hash for value in entries}) != len(entries):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.DUPLICATE_ACTION_CONFLICT,
                "duplicate packed physical actions were not quotiented",
            )
        if any(
            value.parent_mask_hash != self.current_mask_hash
            or value.route_axis_hash != self.route_axis_hash
            or value.route_count != self.route_count
            or value.resource_units != self.resource_units
            or value.resource_source_hash != self.resource_source_hash
            for value in entries
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "one queue entry left the family mask/resource coordinates",
            )
        aliases = tuple(alias for entry in entries for alias in entry.source_aliases)
        if (
            self.source_endpoint_count > self.maximum_source_endpoint_count
            or self.maximum_source_endpoint_count
            != COMPACT_FQR_BRANCH_FAMILY_MAX_SOURCE_ENDPOINTS
            or self.certified_alias_count != len(aliases)
            or self.unique_physical_action_count != len(entries)
            or self.certified_alias_count > self.source_endpoint_count
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.QUEUE_BOUND_EXCEEDED,
                "compact queue counts differ from the fixed bounded union",
            )
        positive_source_branches = tuple(value[0] for value in normalized_sources)
        if set(value.branch.value for value in aliases) != set(
            positive_source_branches
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.MISSING_BRANCH,
                "the queue does not retain a certified alias from every positive "
                "branch or improperly materializes a negative branch",
            )
        winners = _winner_coordinates_from_aliases(aliases)
        if tuple(self.per_branch_winner_coordinates) != winners:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.QUEUE_HEAD_MISMATCH,
                "per-branch winners do not replay from the complete alias union",
            )
        head = entries[0]
        global_winner = min(
            (
                _alias_for_coordinate(aliases, coordinate)
                for coordinate in winners
            ),
            key=_alias_sort_key,
        )
        if (
            head.entry_hash != self.queue_head_entry_hash
            or head.representative_branch is not self.queue_head_branch
            or head.representative_endpoint_index != self.queue_head_endpoint_index
            or head.packed_action_hash != self.queue_head_packed_action_hash
            or head.registered_resource != self.queue_head_registered_resource
            or global_winner.branch is not head.representative_branch
            or global_winner.endpoint_index != head.representative_endpoint_index
            or global_winner.packed_action_hash != head.packed_action_hash
            or global_winner.registered_resource != head.registered_resource
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.QUEUE_HEAD_MISMATCH,
                "queue head differs from the per-branch compact-winner/global-"
                "greatest-resource rule",
            )
        if (
            self.selection_policy != COMPACT_FQR_BRANCH_FAMILY_SELECTION_POLICY
            or self.proposal_queue_materialization_authority is not True
            or self.materialized_before_K is not True
            or self.K_outcomes_seen is not False
            or self.O_outcomes_seen is not False
            or self.queue_mutable_after_K is not False
            or self.oracle_conditioned_reordering_authority is not False
            or self.backtracking_authority is not False
            or self.calibration_K_open_authority is not False
            or self.paired_oracle_O_open_authority is not False
            or self.irreversible_decision_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_validation_access_authority is not False
            or self.terminal_authority is not False
            or self.pareto_claim_authority is not False
            or self.hard_global_optimality_claim is not False
            or self.paired_bce_oracle_still_required is not True
            or self.retained_context_ids != 0
            or self.retained_record_arrays != 0
            or self.retained_route_scale_arrays != 0
            or self.authority_scope != COMPACT_FQR_BRANCH_FAMILY_AUTHORITY_SCOPE
            or self.schema
            not in {
                COMPACT_FQR_BRANCH_FAMILY_AUTHORIZATION_SCHEMA,
                COMPACT_FQR_BRANCH_FAMILY_AUTHORIZATION_V2_SCHEMA,
            }
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "compact branch-family queue crossed its materialization-only boundary",
            )
        object.__setattr__(self, "source_branch_evidence_hashes", normalized_sources)
        object.__setattr__(
            self,
            "source_branch_negative_evidence_hashes",
            normalized_negatives,
        )
        object.__setattr__(self, "queue_entries", entries)
        object.__setattr__(self, "per_branch_winner_coordinates", winners)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in (
                "schema",
                "prefrozen_family_hash",
                "problem_hash",
                "policy_hash",
                "result_hash",
                "chart_identity_hash",
                "current_mask_hash",
                "proposal_generation",
                "route_axis_hash",
                "route_count",
                "discovery_prefix_hash",
                "discovery_evidence_hash",
                "joint_fqr_family_registration_hash",
                "resource_units",
                "resource_source_hash",
                "source_branch_evidence_hashes",
                "dominance_certificate_hash",
                "source_endpoint_count",
                "certified_alias_count",
                "unique_physical_action_count",
                "maximum_source_endpoint_count",
                "per_branch_winner_coordinates",
                "queue_head_entry_hash",
                "queue_head_branch",
                "queue_head_endpoint_index",
                "queue_head_packed_action_hash",
                "queue_head_registered_resource",
                "selection_policy",
                "proposal_queue_materialization_authority",
                "materialized_before_K",
                "K_outcomes_seen",
                "O_outcomes_seen",
                "queue_mutable_after_K",
                "oracle_conditioned_reordering_authority",
                "backtracking_authority",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "irreversible_decision_authority",
                "mask_commit_authority",
                "terminal_validation_access_authority",
                "terminal_authority",
                "pareto_claim_authority",
                "hard_global_optimality_claim",
                "paired_bce_oracle_still_required",
                "retained_context_ids",
                "retained_record_arrays",
                "retained_route_scale_arrays",
                "authority_scope",
            )
        } | {"queue_entry_hashes": tuple(value.entry_hash for value in self.queue_entries)}
        if self.schema == COMPACT_FQR_BRANCH_FAMILY_AUTHORIZATION_V2_SCHEMA:
            payload["source_branch_negative_evidence_hashes"] = (
                self.source_branch_negative_evidence_hashes
            )
        return payload

    @property
    def authorization_hash(self) -> str:
        cached = self._authorization_hash
        return self.assert_integrity() if cached is None else cached

    @property
    def queue_head(self) -> V1CompactFQRBranchFamilyQueueEntry:
        return self.queue_entries[0]

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.proposal_queue_materialization_authority is not True
            or self.materialized_before_K is not True
            or self.K_outcomes_seen is not False
            or self.O_outcomes_seen is not False
            or self.queue_mutable_after_K is not False
            or self.oracle_conditioned_reordering_authority is not False
            or self.backtracking_authority is not False
            or self.calibration_K_open_authority is not False
            or self.paired_oracle_O_open_authority is not False
            or self.irreversible_decision_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_validation_access_authority is not False
            or self.terminal_authority is not False
            or self.pareto_claim_authority is not False
            or self.hard_global_optimality_claim is not False
            or self.paired_bce_oracle_still_required is not True
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.AUTHORITY_VIOLATION,
                "compact branch-family queue crossed its materialization-only boundary",
            )
        for value in self.queue_entries:
            value.assert_integrity()
        if self.queue_entries[0].entry_hash != self.queue_head_entry_hash:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.QUEUE_HEAD_MISMATCH,
                "compact queue head changed after materialization",
            )
        observed = content_hash(self._payload(), prefix=_AUTHORIZATION_PREFIX)
        cached = self._authorization_hash
        if cached is not None and cached != observed:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "compact branch-family authorization was tampered after hashing",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.HASH_MISMATCH,
                "compact branch-family authorization hash does not match",
            )
        object.__setattr__(self, "_authorization_hash", observed)
        return observed


def _winner_coordinates_from_aliases(
    aliases: tuple[V1CompactFQRBranchFamilySourceAlias, ...],
) -> tuple[tuple[str, int, str, float], ...]:
    winners: list[tuple[str, int, str, float]] = []
    for branch in (
        V1RelaxedKKTBranch.SPECTRAL,
        V1RelaxedKKTBranch.DIAGONAL_L1,
    ):
        branch_aliases = tuple(value for value in aliases if value.branch is branch)
        if not branch_aliases:
            continue
        winner = min(branch_aliases, key=_alias_sort_key)
        winners.append(
            (
                branch.value,
                winner.endpoint_index,
                winner.packed_action_hash,
                winner.registered_resource,
            )
        )
    return tuple(sorted(winners))


def _alias_for_coordinate(
    aliases: tuple[V1CompactFQRBranchFamilySourceAlias, ...],
    coordinate: tuple[str, int, str, float],
) -> V1CompactFQRBranchFamilySourceAlias:
    for value in aliases:
        if (
            value.branch.value,
            value.endpoint_index,
            value.packed_action_hash,
            value.registered_resource,
        ) == coordinate:
            return value
    _refuse(
        V1CompactFQRBranchFamilyRefusalCode.QUEUE_HEAD_MISMATCH,
        "per-branch winner coordinate is absent from the compact alias union",
    )


def _evidence_for_branch(
    *,
    branch: V1RelaxedKKTBranch,
    spectral_evidence: object | None,
    diagonal_l1_evidence: object | None,
) -> object | None:
    return (
        spectral_evidence
        if branch is V1RelaxedKKTBranch.SPECTRAL
        else diagonal_l1_evidence
    )


def authorize_v1_compact_fqr_branch_family_queue(
    *,
    prefrozen_family: object,
    spectral_evidence: object | None,
    diagonal_l1_evidence: object | None,
    current_mask: object,
    live_chart_identity_hash: str,
    live_current_mask_hash: str,
    live_proposal_generation: int,
    live_family_hash: str,
) -> V1CompactFQRBranchFamilyAuthorization:
    """Materialize the full compact certified physical union before K/O."""

    family_type = type(prefrozen_family)
    if family_type not in _EXACT_PREFROZEN_FAMILY_TYPES:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
            "compact queue authorization requires the exact pre-frozen family",
        )
    if type(current_mask) is not HardMask:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.INVALID_TYPE,
            "compact queue authorization requires the exact current HardMask",
        )
    try:
        family_hash = prefrozen_family.assert_integrity()
    except Exception as error:
        _translate_source_refusal(error)
    parent_hash = _direct_mask_hash(current_mask, name="live current mask")
    live_chart = _identity(live_chart_identity_hash, name="live chart identity hash")
    live_mask = _identity(live_current_mask_hash, name="live current-mask hash")
    live_generation = _nonnegative_int(
        live_proposal_generation,
        name="live proposal generation",
    )
    live_family = _identity(live_family_hash, name="live pre-frozen family hash")
    if (
        family_hash != live_family
        or prefrozen_family.chart_identity_hash != live_chart
        or prefrozen_family.current_mask_hash != live_mask
        or prefrozen_family.current_mask_hash != parent_hash
        or prefrozen_family.proposal_generation != live_generation
        or current_mask.route_axis_hash != prefrozen_family.route_axis_hash
        or current_mask.route_count != prefrozen_family.route_count
    ):
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.STALE_CHART,
            "compact branch family does not belong to the live chart/mask/generation",
        )

    evidences: list[
        V1CompactFQRBranchEvidenceBinding
        | V1CompactFQRBranchNegativeEvidenceBinding
    ] = []
    for branch in (
        V1RelaxedKKTBranch.SPECTRAL,
        V1RelaxedKKTBranch.DIAGONAL_L1,
    ):
        binding = (
            prefrozen_family.spectral
            if branch is V1RelaxedKKTBranch.SPECTRAL
            else prefrozen_family.diagonal_l1
        )
        evidence = _evidence_for_branch(
            branch=branch,
            spectral_evidence=spectral_evidence,
            diagonal_l1_evidence=diagonal_l1_evidence,
        )
        if binding is None:
            if evidence is not None:
                _refuse(
                    V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                    "evidence was supplied for a pre-C dominated branch",
                )
            continue
        _classify_compact_substitution(evidence)
        if type(evidence) not in {
            V1CompactFQRBranchEvidenceBinding,
            V1CompactFQRBranchNegativeEvidenceBinding,
        }:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.MISSING_BRANCH,
                f"{branch.value} lacks one exact positive-or-stable-negative "
                "compact F/Q/R evidence binding",
            )
        evidence.assert_integrity()
        if (
            evidence.branch is not branch
            or evidence.prefrozen_family_hash != family_hash
            or evidence.branch_binding.binding_hash != binding.binding_hash
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
                "compact evidence differs from its retained family branch",
            )
        evidences.append(evidence)

    if len(evidences) != len(prefrozen_family.retained_bindings):
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.MISSING_BRANCH,
            "every retained branch requires one compact F/Q/R evidence binding",
        )
    if family_type is V1FQRDeviceShellPackedPreFrozenBranchFamily and (
        len(evidences) != 2
        or len(prefrozen_family.retained_bindings) != 2
        or prefrozen_family.spectral is None
        or prefrozen_family.diagonal_l1 is None
        or prefrozen_family.dominance_certificate is not None
    ):
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
            "a packed device-shell family requires complete evidence for both "
            "branches and cannot omit one by dominance",
        )
    if len(evidences) == 1:
        certificate = prefrozen_family.dominance_certificate
        if certificate is None:
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.MISSING_BRANCH,
                "one branch may be omitted only by exact pre-C action dominance",
            )
        try:
            certificate.assert_integrity()
        except Exception as error:
            _translate_source_refusal(error)
    elif prefrozen_family.dominance_certificate is not None:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.INVALID_DOMINANCE,
            "a complete two-branch family cannot carry an omission certificate",
        )

    source_endpoint_count = sum(
        len(value.certification_receipt.endpoint_certificates) for value in evidences
    )
    if source_endpoint_count > COMPACT_FQR_BRANCH_FAMILY_MAX_SOURCE_ENDPOINTS:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.QUEUE_BOUND_EXCEEDED,
            "pre-frozen branch family exceeds the compact endpoint bound",
        )
    joint_hashes = {
        value.certification_receipt.joint_fqr_family_registration_hash
        for value in evidences
    }
    resource_coordinates = {
        (
            value.resource_registration.resource_units,
            value.resource_registration.resource_source_hash,
        )
        for value in evidences
    }
    if len(joint_hashes) != 1 or len(resource_coordinates) != 1:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.COORDINATE_MISMATCH,
            "retained branches do not share one registered F/Q/R/resource family",
        )
    joint_fqr_hash = next(iter(joint_hashes))
    resource_units, resource_source_hash = next(iter(resource_coordinates))

    positive_evidences = tuple(
        value
        for value in evidences
        if type(value) is V1CompactFQRBranchEvidenceBinding
    )
    negative_evidences = tuple(
        value
        for value in evidences
        if type(value) is V1CompactFQRBranchNegativeEvidenceBinding
    )

    aliases: list[V1CompactFQRBranchFamilySourceAlias] = []
    packed_by_source: dict[tuple[V1RelaxedKKTBranch, int], V1PackedRemovalAction] = {}
    branch_authorized_coordinates: list[tuple[str, int, str, float]] = []
    for evidence in positive_evidences:
        binding = evidence.branch_binding
        receipt = evidence.certification_receipt
        resources = evidence.resource_registration
        authorization = evidence.proposal_authorization
        branch_authorized_coordinates.append(
            (
                evidence.branch.value,
                authorization.selected_endpoint_index,
                authorization.selected_packed_action.action_hash,
                authorization.selected_resource,
            )
        )
        for certificate in receipt.endpoint_certificates:
            if not certificate.upper_certified:
                continue
            endpoint = certificate.endpoint_index
            frozen = binding.frozen_ladder.actions[endpoint - 1]
            packed = frozen.packed_removal_action
            expected_packed_hash = binding.endpoint_packed_action_hashes[endpoint - 1]
            packed, _child_hash = _validate_packed_action(
                packed=packed,
                parent=current_mask,
                expected_packed_hash=expected_packed_hash,
            )
            packed_by_source[(evidence.branch, endpoint)] = packed
            scalars = _certificate_scalars(certificate)
            aliases.append(
                V1CompactFQRBranchFamilySourceAlias(
                    branch=evidence.branch,
                    endpoint_index=endpoint,
                    frozen_action_hash=frozen.action_hash,
                    packed_action_hash=expected_packed_hash,
                    registered_resource=(
                        binding.endpoint_registered_resources[endpoint - 1]
                    ),
                    branch_binding_hash=binding.binding_hash,
                    branch_evidence_hash=evidence.evidence_hash,
                    resource_registration_hash=resources.registration_hash,
                    certification_receipt_hash=receipt.receipt_hash,
                    decision_hash=evidence.decision.decision_hash,
                    proposal_authorization_hash=authorization.authorization_hash,
                    endpoint_certificate_hash=certificate.certificate_hash,
                    f_lower=scalars[0],
                    f_upper=scalars[1],
                    q_lower=scalars[2],
                    q_upper=scalars[3],
                    r_half_lower=scalars[4],
                    r_half_upper=scalars[5],
                    r_one_lower=scalars[6],
                    r_one_upper=scalars[7],
                    half_response_lower=scalars[8],
                    half_response_upper=scalars[9],
                    one_response_lower=scalars[10],
                    one_response_upper=scalars[11],
                )
            )
    if not aliases:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.NO_CERTIFIED_ENDPOINT,
            "every retained branch is stably lower-refuted and no endpoint is "
            "certified",
        )
    alias_tuple = tuple(aliases)
    replayed_branch_winners = _winner_coordinates_from_aliases(alias_tuple)
    if tuple(sorted(branch_authorized_coordinates)) != replayed_branch_winners:
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.QUEUE_HEAD_MISMATCH,
            "compact per-branch authorizations differ from their complete certified axes",
        )

    aliases_by_action: dict[
        str,
        list[V1CompactFQRBranchFamilySourceAlias],
    ] = {}
    for alias in sorted(aliases, key=_alias_sort_key):
        aliases_by_action.setdefault(alias.packed_action_hash, []).append(alias)

    entries: list[V1CompactFQRBranchFamilyQueueEntry] = []
    for packed_hash, raw_group in aliases_by_action.items():
        group = tuple(sorted(raw_group, key=_alias_sort_key))
        representative = group[0]
        packed = packed_by_source[
            (representative.branch, representative.endpoint_index)
        ]
        if any(
            alias.registered_resource != representative.registered_resource
            for alias in group
        ):
            _refuse(
                V1CompactFQRBranchFamilyRefusalCode.DUPLICATE_ACTION_CONFLICT,
                "identical packed physical actions have different registered resource",
            )
        for alias in group[1:]:
            other = packed_by_source[(alias.branch, alias.endpoint_index)]
            if (
                other.action_hash != packed.action_hash
                or other.removed_count != packed.removed_count
                or other.removed_route_index_hash
                != packed.removed_route_index_hash
                or other.resulting_mask.route_axis_hash
                != packed.resulting_mask.route_axis_hash
                or other.resulting_mask.route_count
                != packed.resulting_mask.route_count
                or other.resulting_mask.packed_bits
                != packed.resulting_mask.packed_bits
            ):
                _refuse(
                    V1CompactFQRBranchFamilyRefusalCode.DUPLICATE_ACTION_CONFLICT,
                    f"packed action hash {packed_hash} maps to distinct physical actions",
                )
        child_hash = _direct_mask_hash(
            packed.resulting_mask,
            name="queue resulting mask",
        )
        scalars = _intersect_alias_scalars(group)
        entries.append(
            V1CompactFQRBranchFamilyQueueEntry(
                representative_branch=representative.branch,
                representative_endpoint_index=representative.endpoint_index,
                representative_frozen_action_hash=(
                    representative.frozen_action_hash
                ),
                packed_action_hash=packed_hash,
                packed_action=packed,
                parent_mask_hash=parent_hash,
                resulting_mask_hash=child_hash,
                route_axis_hash=prefrozen_family.route_axis_hash,
                route_count=prefrozen_family.route_count,
                registered_resource=representative.registered_resource,
                resource_units=resource_units,
                resource_source_hash=resource_source_hash,
                f_lower=scalars[0],
                f_upper=scalars[1],
                q_lower=scalars[2],
                q_upper=scalars[3],
                r_half_lower=scalars[4],
                r_half_upper=scalars[5],
                r_one_lower=scalars[6],
                r_one_upper=scalars[7],
                half_response_lower=scalars[8],
                half_response_upper=scalars[9],
                one_response_lower=scalars[10],
                one_response_upper=scalars[11],
                source_aliases=group,
            )
        )
    queue = tuple(sorted(entries, key=_entry_sort_key))
    head = queue[0]
    global_winner = min(
        (
            _alias_for_coordinate(alias_tuple, coordinate)
            for coordinate in replayed_branch_winners
        ),
        key=_alias_sort_key,
    )
    if (
        head.representative_branch is not global_winner.branch
        or head.representative_endpoint_index != global_winner.endpoint_index
        or head.packed_action_hash != global_winner.packed_action_hash
        or head.registered_resource != global_winner.registered_resource
    ):
        _refuse(
            V1CompactFQRBranchFamilyRefusalCode.QUEUE_HEAD_MISMATCH,
            "queue head differs from the global greatest-resource compact winner",
        )

    dominance_hash = (
        None
        if prefrozen_family.dominance_certificate is None
        else prefrozen_family.dominance_certificate.certificate_hash
    )
    return V1CompactFQRBranchFamilyAuthorization(
        prefrozen_family_hash=family_hash,
        problem_hash=prefrozen_family.problem_hash,
        policy_hash=prefrozen_family.policy_hash,
        result_hash=prefrozen_family.result_hash,
        chart_identity_hash=prefrozen_family.chart_identity_hash,
        current_mask_hash=parent_hash,
        proposal_generation=prefrozen_family.proposal_generation,
        route_axis_hash=prefrozen_family.route_axis_hash,
        route_count=prefrozen_family.route_count,
        discovery_prefix_hash=prefrozen_family.discovery_prefix_hash,
        discovery_evidence_hash=prefrozen_family.discovery_evidence_hash,
        joint_fqr_family_registration_hash=joint_fqr_hash,
        resource_units=resource_units,
        resource_source_hash=resource_source_hash,
        source_branch_evidence_hashes=tuple(
            sorted(
                (value.branch.value, value.evidence_hash)
                for value in positive_evidences
            )
        ),
        dominance_certificate_hash=dominance_hash,
        queue_entries=queue,
        source_endpoint_count=source_endpoint_count,
        certified_alias_count=len(aliases),
        unique_physical_action_count=len(queue),
        maximum_source_endpoint_count=(
            COMPACT_FQR_BRANCH_FAMILY_MAX_SOURCE_ENDPOINTS
        ),
        per_branch_winner_coordinates=replayed_branch_winners,
        queue_head_entry_hash=head.entry_hash,
        queue_head_branch=head.representative_branch,
        queue_head_endpoint_index=head.representative_endpoint_index,
        queue_head_packed_action_hash=head.packed_action_hash,
        queue_head_registered_resource=head.registered_resource,
        source_branch_negative_evidence_hashes=tuple(
            sorted(
                (value.branch.value, value.evidence_hash)
                for value in negative_evidences
            )
        ),
        schema=(
            COMPACT_FQR_BRANCH_FAMILY_AUTHORIZATION_V2_SCHEMA
            if negative_evidences
            else COMPACT_FQR_BRANCH_FAMILY_AUTHORIZATION_SCHEMA
        ),
    )


__all__ = [
    "COMPACT_FQR_BRANCH_FAMILY_AUTHORITY_SCOPE",
    "COMPACT_FQR_BRANCH_FAMILY_MAX_SOURCE_ENDPOINTS",
    "COMPACT_FQR_BRANCH_FAMILY_SELECTION_POLICY",
    "V1CompactFQRBranchEvidenceBinding",
    "V1CompactFQRBranchNegativeEvidenceBinding",
    "V1CompactFQRBranchFamilyAuthorization",
    "V1CompactFQRBranchFamilyQueueEntry",
    "V1CompactFQRBranchFamilyRefusal",
    "V1CompactFQRBranchFamilyRefusalCode",
    "V1CompactFQRBranchFamilySourceAlias",
    "authorize_v1_compact_fqr_branch_family_queue",
    "bind_v1_compact_fqr_branch_evidence",
    "bind_v1_compact_fqr_branch_negative_evidence",
]
