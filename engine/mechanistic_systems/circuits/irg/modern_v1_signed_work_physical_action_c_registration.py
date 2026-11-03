"""Canonical pre-outcome registration for branchless physical-action C.

The physical-action C runtime has several independently valid ingredients:
the complete ImageNet population, causal D/C permutations, the frozen D
shell, a deduplicated physical-action axis, a finite-dose provider, analytic
observation supports, and an independent F/Q-family alpha allocation.  None
of those ingredients, in isolation, proves that they describe the same chart
and proposal generation.  This module is the narrow typed joint that closes
that gap before the first C outcome is observed.

Two statistical ledgers deliberately remain distinct.  The proposal-order
ledger carried by :class:`V1SignedWorkPhysicalActionCAxis` is D lineage only.
Physical C consumes the current score-look allocation from the independent
F/Q family, whose receipt explicitly forbids reuse of proposal-order alpha.
The joint never advances either ledger and grants no K, O, commit, terminal,
or Pareto authority.

Coordinate registration is compact.  The exact target/complement permutation
receipts are retained once, and context, draw, and packet identities are
derived deterministically on demand.  Consequently the production object
does not materialize another 1.28-million-element coordinate table.
"""

from __future__ import annotations

from .modern_v1_physical_metric import (
    LEGACY_UNIFORM_ABSOLUTE, BALANCED_RMS, profile_coordinates,
    squared_support_upper, validate_physical_measurement_profile, restore_profile_state,
)

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Final, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.experimental_design.finite_population import (
    SealedUniformPermutationReceipt,
)

from .modern_v1_alexnet_fq_registration import (
    V1AlexNetFQAcquisitionRegistration,
    V1AlexNetFQCausalRoleSamplingStreamReceipt,
    V1AlexNetFQDiscoveryPrefixFreezeReceipt,
)
from .modern_v1_alexnet_fqr_bounds import V1AlexNetFQRBoundDerivationReceipt
from .modern_v1_imagenet_population import (
    CompleteImageNetPopulationPairReceipt,
    IMAGENET1K_TRAIN_CONTEXT_COUNT,
)
from .modern_v1_response_exactification import V1FQAlphaFamilyAllocationReceipt
from .modern_v1_signed_work_physical_action_c_axis import (
    V1SignedWorkPhysicalActionCAxis,
)
from .modern_v1_signed_work_physical_action_direction_provider import (
    V1PhysicalActionDirectionReceipt,
)
from .modern_v1_signed_work_physical_action_six_role_c import (
    D_HALF_ROLE,
    D_ONE_ROLE,
    F_NATIVE_ROLE,
    Q_J_ROLE,
    R_HALF_ROLE,
    R_ONE_ROLE,
    SIGNED_WORK_SIX_ROLES,
    V1PhysicalActionSixRoleCCoordinateRegistration,
    V1PhysicalActionSixRoleCDesign,
    V1PhysicalActionSixRoleCSourceRegistration,
    V1PhysicalActionSixRoleCStratum,
)

PHYSICAL_ACTION_C_ALPHA_SCHEMA: Final = (
    "partition_free_v1_physical_action_C_independent_fq_alpha_v1"
)
PHYSICAL_ACTION_C_COORDINATE_PLAN_SCHEMA: Final = (
    "partition_free_v1_physical_action_C_coordinate_plan_v1"
)
PHYSICAL_ACTION_C_ANALYTIC_SUPPORT_SCHEMA: Final = (
    "partition_free_v1_physical_action_C_six_role_analytic_support_v1"
)
PHYSICAL_ACTION_C_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_physical_action_C_canonical_registration_v2"
)
PHYSICAL_ACTION_C_ALPHA_POLICY: Final = (
    "current_score_look_independent_fq_redesign_allocation_nonrecycling_"
    "proposal_order_D_alpha_is_lineage_only_v1"
)
PHYSICAL_ACTION_C_SUPPORT_POLICY: Final = (
    "preoutcome_exact_singleton_branch_action_mask_AlexNet_interval_"
    "envelope_alias_max_with_outward_binary64_v1"
)
PHYSICAL_ACTION_C_AUTHORITY: Final = (
    "preoutcome_complete_train_physical_action_C_registration_and_analytic_"
    "support_only_no_outcome_selection_K_O_commit_terminal_or_pareto_authority_v1"
)

_ALPHA_PREFIX: Final = "modern-v1-physical-action-C-alpha-"
_ORDER_PREFIX: Final = "modern-v1-physical-action-C-generated-order-"
_DOMAIN_PREFIX: Final = "modern-v1-physical-action-D-C-domain-separation-"
_PLAN_PREFIX: Final = "modern-v1-physical-action-C-coordinate-plan-"
_DRAW_PREFIX: Final = "modern-v1-physical-action-C-draw-"
_PACKET_PREFIX: Final = "modern-v1-physical-action-C-packet-"
_COORDINATE_PROOF_PREFIX: Final = "modern-v1-physical-action-C-coordinate-proof-"
_SUPPORT_SOURCE_PREFIX: Final = "modern-v1-physical-action-C-support-source-"
_SUPPORT_PREFIX: Final = "modern-v1-physical-action-C-support-"
_REGISTRATION_PREFIX: Final = "modern-v1-physical-action-C-registration-"
_RAW_LOGIT_EXECUTION_STREAM_PREFIX: Final = (
    "modern-v1-physical-action-C-finite-direction-raw-logit-stream-"
)
_LE_F64: Final = np.dtype("<f8")
_ALPHA_TOKEN: Final = object()
_PLAN_TOKEN: Final = object()
_SUPPORT_TOKEN: Final = object()
_REGISTRATION_TOKEN: Final = object()


class V1PhysicalActionCRegistrationRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_POPULATION = "stale_population"
    STALE_STREAM = "stale_stream"
    DOMAIN_SEPARATION_FAILURE = "domain_separation_failure"
    STALE_D_FREEZE = "stale_D_freeze"
    ACTION_AXIS_MISMATCH = "action_axis_mismatch"
    ALPHA_GENERATION_MISMATCH = "alpha_generation_mismatch"
    ALPHA_DOUBLE_SPEND = "alpha_double_spend"
    STALE_SOURCE = "stale_source"
    STALE_COORDINATE = "stale_coordinate"
    ANALYTIC_SUPPORT_INVALID = "analytic_support_invalid"
    POSTSELECTION = "postselection"
    AUTHORITY_VIOLATION = "authority_violation"


class V1PhysicalActionCRegistrationRefusal(ValueError):
    """Typed fail-closed refusal for canonical physical-action C issuance."""

    def __init__(
        self,
        code: V1PhysicalActionCRegistrationRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1PhysicalActionCRegistrationRefusalCode:
            raise TypeError("physical-C registration refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1PhysicalActionCRegistrationRefusalCode,
    message: str,
) -> NoReturn:
    raise V1PhysicalActionCRegistrationRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def derive_v1_physical_action_c_finite_direction_raw_logit_execution_stream_identity(
    *,
    raw_logit_coordinate_identity_hash: str,
    certification_stream_hash: str,
    physical_action_axis_hash: str,
    direction_provider_receipt_hash: str,
    chart_identity_hash: str,
    C_execution_registration_hash: str,
) -> str:
    """Name finite-direction C logits without confusing them with their coordinate.

    The D chart freezes *which* raw target-logit coordinate is being studied.
    Physical C subsequently evaluates that coordinate along a registered family
    of finite relaxed-gate directions.  Those executions are a new measurement
    stream, even when corrected evidence uses the same underlying logit
    coordinate.  Domain-separated derivation makes that distinction explicit
    while binding the new stream causally to its D chart, C permutation, action
    axis, direction provider, and pre-outcome execution design.
    """

    payload = {
        "schema": (
            "partition_free_v1_physical_action_C_finite_direction_"
            "raw_logit_execution_stream_v1"
        ),
        "raw_logit_coordinate_identity_hash": _identity(
            raw_logit_coordinate_identity_hash,
            name="raw-logit coordinate identity hash",
        ),
        "certification_stream_hash": _identity(
            certification_stream_hash,
            name="certification stream hash",
        ),
        "physical_action_axis_hash": _identity(
            physical_action_axis_hash,
            name="physical-action axis hash",
        ),
        "direction_provider_receipt_hash": _identity(
            direction_provider_receipt_hash,
            name="direction-provider receipt hash",
        ),
        "chart_identity_hash": _identity(
            chart_identity_hash,
            name="chart identity hash",
        ),
        "C_execution_registration_hash": _identity(
            C_execution_registration_hash,
            name="C execution registration hash",
        ),
        "issued_before_C_outcomes": True,
    }
    return content_hash(payload, prefix=_RAW_LOGIT_EXECUTION_STREAM_PREFIX)


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            f"{name} must be one real scalar",
        )
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.ALPHA_GENERATION_MISMATCH,
            f"{name} must lie strictly inside (0, 1)",
        )
    return result


def _immutable_f64(value: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=_LE_F64)
    except (TypeError, ValueError, OverflowError) as error:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
            f"{name} cannot be represented as binary64: {error}",
        )
    if (
        array.shape != shape
        or not bool(np.isfinite(array).all())
        or bool(np.any(array < 0.0))
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
            f"{name} must be finite, nonnegative, and have shape {shape}",
        )
    snapshot = np.frombuffer(
        np.ascontiguousarray(array, dtype=_LE_F64).tobytes(order="C"),
        dtype=_LE_F64,
    ).reshape(shape)
    snapshot.setflags(write=False)
    return snapshot


def _outward_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
            f"{name} must be one real scalar",
        )
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
            f"{name} must be finite and nonnegative",
        )
    enclosed = math.nextafter(scalar, math.inf)
    if not math.isfinite(enclosed):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
            f"{name} cannot be outward-enclosed in binary64",
        )
    return enclosed


def _closed_authority(values: Sequence[object], *, owner: str) -> None:
    if any(type(value) is not bool for value in values) or any(values):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed its pre-outcome registration boundary",
        )


def v1_physical_action_c_generated_order_hash(
    receipt: SealedUniformPermutationReceipt,
    *,
    role: str,
    stratum: str,
) -> str:
    """Bind the exact generated reveal, not merely its population."""

    if type(receipt) is not SealedUniformPermutationReceipt:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            "generated-order hashing requires an exact permutation receipt",
        )
    receipt.verify()
    return content_hash(
        {
            "permutation_receipt_hash": receipt.content_hash,
            "role": _identity(role, name="causal stream role"),
            "stratum": _identity(stratum, name="causal stream stratum"),
            # The exact receipt hash already binds both the canonical
            # population and its generated reveal.  Re-serializing the full
            # 1.28M-context tuple here would add an avoidable O(N) transient
            # without strengthening the identity proof.
            "generated_order_count": len(receipt.generated_order),
            "algorithm_id": receipt.algorithm_id,
        },
        prefix=_ORDER_PREFIX,
    )


def v1_physical_action_c_stream_order_hash(
    stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
) -> str:
    """Return one compact exact hash for target then complement order."""

    if type(stream) is not V1AlexNetFQCausalRoleSamplingStreamReceipt:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            "stream-order hashing requires an exact causal role stream",
        )
    stream_hash = stream.assert_integrity()
    order_hashes = tuple(
        v1_physical_action_c_generated_order_hash(
            receipt,
            role=stream.role,
            stratum=stratum,
        )
        for stratum, receipt in (
            ("target", stream.target_assignment_receipt),
            ("complement", stream.complement_assignment_receipt),
        )
    )
    return content_hash(
        {
            "stream_hash": stream_hash,
            "role": stream.role,
            "order_hashes": order_hashes,
            "stratum_order": ("target", "complement"),
        },
        prefix=_ORDER_PREFIX,
    )


def v1_physical_action_dc_domain_separation_proof_hash(
    *,
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
    discovery_freeze: V1AlexNetFQDiscoveryPrefixFreezeReceipt,
    certification_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
) -> str:
    """Verify and bind exact D/C KDF-domain separation and causal order."""

    if (
        type(discovery_stream) is not V1AlexNetFQCausalRoleSamplingStreamReceipt
        or type(certification_stream) is not V1AlexNetFQCausalRoleSamplingStreamReceipt
        or type(discovery_freeze) is not V1AlexNetFQDiscoveryPrefixFreezeReceipt
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            "D/C separation requires exact role streams and D freeze",
        )
    d_hash = discovery_stream.assert_integrity()
    c_hash = certification_stream.assert_integrity()
    freeze_hash = discovery_freeze.assert_integrity()
    if (
        discovery_stream.role != "discovery"
        or certification_stream.role != "certification"
        or discovery_freeze.discovery_stream_hash != d_hash
        or certification_stream.causal_parent_receipt_hash != freeze_hash
        or certification_stream.discovery_prefix_freeze_hash != freeze_hash
        or certification_stream.selected_shell_hash
        != discovery_freeze.selected_shell_hash
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.DOMAIN_SEPARATION_FAILURE,
            "C was not opened from the exact frozen D shell",
        )
    common_names = (
        "registration_hash",
        "generation_frame_hash",
        "generation_index",
        "current_mask_hash",
        "frozen_transform_hash",
        "target_population_membership_hash",
        "complement_population_membership_hash",
    )
    if any(
        getattr(discovery_stream, name) != getattr(certification_stream, name)
        for name in common_names
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.DOMAIN_SEPARATION_FAILURE,
            "D and C do not share one registered chart/population generation",
        )
    d_receipts = (
        discovery_stream.target_assignment_receipt,
        discovery_stream.complement_assignment_receipt,
    )
    c_receipts = (
        certification_stream.target_assignment_receipt,
        certification_stream.complement_assignment_receipt,
    )
    for d_receipt, c_receipt in zip(d_receipts, c_receipts, strict=True):
        d_receipt.verify()
        c_receipt.verify()
        if (
            d_receipt.canonical_population != c_receipt.canonical_population
            or d_receipt.entropy_hex == c_receipt.entropy_hex
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.DOMAIN_SEPARATION_FAILURE,
                "D/C strata changed population or reused randomization entropy",
            )
    if (
        c_receipts[0].entropy_hex == c_receipts[1].entropy_hex
        or discovery_stream.domain_separated_from_other_roles is not True
        or certification_stream.domain_separated_from_other_roles is not True
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.DOMAIN_SEPARATION_FAILURE,
            "causal role/stratum randomization domains are not distinct",
        )
    return content_hash(
        {
            "discovery_stream_hash": d_hash,
            "discovery_order_hash": v1_physical_action_c_stream_order_hash(
                discovery_stream
            ),
            "discovery_freeze_hash": freeze_hash,
            "certification_stream_hash": c_hash,
            "certification_order_hash": v1_physical_action_c_stream_order_hash(
                certification_stream
            ),
            "D_entropy_commitments": tuple(
                content_hash(value.entropy_hex, prefix="sealed-D-entropy-")
                for value in d_receipts
            ),
            "C_entropy_commitments": tuple(
                content_hash(value.entropy_hex, prefix="sealed-C-entropy-")
                for value in c_receipts
            ),
            "same_complete_population_different_KDF_domains": True,
            "C_opened_only_after_exact_D_freeze": True,
        },
        prefix=_DOMAIN_PREFIX,
    )


@dataclass(frozen=True, slots=True)
class V1PhysicalActionCAlphaAllocationReceipt:
    """Independent current-look C alpha; proposal-order alpha is lineage only."""

    source_fq_allocation: V1FQAlphaFamilyAllocationReceipt = field(
        repr=False,
        compare=False,
    )
    physical_action_axis_hash: str
    d_freeze_hash: str
    certification_stream_hash: str
    proposal_order_ledger_lineage_hash: str
    proposal_order_alpha_schedule_hash: str
    proposal_generation: int
    score_order_look_index: int
    fq_familywise_alpha: float
    fq_alpha_committed_before_generation: float
    fq_alpha_committed_before_score_look: float
    physical_c_family_alpha_allocation: float
    source_fq_allocation_receipt_hash: str
    _issuer_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    proposal_order_D_alpha_reused_for_C: bool = False
    next_proposal_generation_alpha_consumed: bool = False
    allocation_nonrecycling: bool = True
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    downstream_authority: bool = False
    alpha_policy: str = PHYSICAL_ACTION_C_ALPHA_POLICY
    authority_scope: str = PHYSICAL_ACTION_C_AUTHORITY
    schema: str = PHYSICAL_ACTION_C_ALPHA_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuer_token is not _ALPHA_TOKEN:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "physical-C alpha receipts are issued only by the typed allocator",
            )
        if type(self.source_fq_allocation) is not V1FQAlphaFamilyAllocationReceipt:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
                "physical C requires the exact independent F/Q alpha receipt",
            )
        source_hash = self.source_fq_allocation.assert_integrity()
        for name in (
            "physical_action_axis_hash",
            "d_freeze_hash",
            "certification_stream_hash",
            "proposal_order_ledger_lineage_hash",
            "proposal_order_alpha_schedule_hash",
            "source_fq_allocation_receipt_hash",
            "alpha_policy",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        look = _positive_int(self.score_order_look_index, name="score-order look")
        familywise = _probability(self.fq_familywise_alpha, name="F/Q familywise alpha")
        committed_generation = float(self.fq_alpha_committed_before_generation)
        committed_look = float(self.fq_alpha_committed_before_score_look)
        allocation = _probability(
            self.physical_c_family_alpha_allocation,
            name="physical-C family alpha",
        )
        if (
            not math.isfinite(committed_generation)
            or committed_generation < 0.0
            or not math.isfinite(committed_look)
            or committed_look < 0.0
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ALPHA_GENERATION_MISMATCH,
                "physical-C committed alpha coordinates must be finite/nonnegative",
            )
        source = self.source_fq_allocation
        if (
            self.source_fq_allocation_receipt_hash != source_hash
            or generation != source.proposal_generation
            or look != source.score_order_look_index
            or self.proposal_order_alpha_schedule_hash
            != source.proposal_order_alpha_schedule_hash
            or familywise != source.fq_familywise_alpha
            or committed_generation != source.fq_alpha_committed_before_generation
            or committed_look != source.fq_alpha_committed_before_redesign
            or allocation != source.fq_redesign_alpha_allocation
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ALPHA_GENERATION_MISMATCH,
                "physical-C alpha differs from its exact current score-look receipt",
            )
        if (
            source.proposal_alpha_reused_for_fq is not False
            or source.redesign_alpha_unused_nonrecycling is not True
            or source.redesign_alpha_nontransferable_across_score_looks is not True
            or source.oracle_alpha_transferable_to_fq is not False
            or self.proposal_order_D_alpha_reused_for_C is not False
            or self.next_proposal_generation_alpha_consumed is not False
            or self.allocation_nonrecycling is not True
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ALPHA_DOUBLE_SPEND,
                "physical C attempted to recycle D, another look, or oracle alpha",
            )
        if Fraction.from_float(committed_generation) + Fraction.from_float(
            committed_look
        ) + Fraction.from_float(allocation) > Fraction.from_float(familywise):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ALPHA_DOUBLE_SPEND,
                "physical-C allocation overspends the independent F/Q family",
            )
        if (
            self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
            or self.downstream_authority is not False
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.POSTSELECTION,
                "physical-C alpha was not frozen before C outcomes",
            )
        if (
            self.alpha_policy != PHYSICAL_ACTION_C_ALPHA_POLICY
            or self.authority_scope != PHYSICAL_ACTION_C_AUTHORITY
            or self.schema != PHYSICAL_ACTION_C_ALPHA_SCHEMA
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "physical-C alpha policy/schema changed",
            )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "score_order_look_index", look)
        object.__setattr__(self, "fq_familywise_alpha", familywise)
        object.__setattr__(
            self, "fq_alpha_committed_before_generation", committed_generation
        )
        object.__setattr__(self, "fq_alpha_committed_before_score_look", committed_look)
        object.__setattr__(self, "physical_c_family_alpha_allocation", allocation)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"source_fq_allocation", "_issuer_token", "_receipt_hash"}
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        source_hash = self.source_fq_allocation.assert_integrity()
        if source_hash != self.source_fq_allocation_receipt_hash:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ALPHA_GENERATION_MISMATCH,
                "physical-C source alpha receipt changed",
            )
        observed = content_hash(self._payload(), prefix=_ALPHA_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ALPHA_GENERATION_MISMATCH,
                "physical-C alpha receipt was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ALPHA_GENERATION_MISMATCH,
                "physical-C alpha receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


def allocate_v1_physical_action_c_alpha(
    *,
    physical_action_axis: V1SignedWorkPhysicalActionCAxis,
    certification_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
    source_fq_allocation: V1FQAlphaFamilyAllocationReceipt,
) -> V1PhysicalActionCAlphaAllocationReceipt:
    """Allocate C from the independent current-look F/Q family, never D."""

    if type(physical_action_axis) is not V1SignedWorkPhysicalActionCAxis:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            "physical-C alpha requires the exact action axis",
        )
    if type(certification_stream) is not V1AlexNetFQCausalRoleSamplingStreamReceipt:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            "physical-C alpha requires the exact C stream",
        )
    if type(source_fq_allocation) is not V1FQAlphaFamilyAllocationReceipt:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            "physical-C alpha requires the exact independent F/Q allocation",
        )
    axis_hash = physical_action_axis.assert_integrity()
    c_hash = certification_stream.assert_integrity()
    source_hash = source_fq_allocation.assert_integrity()
    ledger = physical_action_axis.alpha_ledger_head
    ledger_hash = ledger.assert_integrity()
    generation = physical_action_axis.alpha_ledger_generation
    if (
        certification_stream.role != "certification"
        or certification_stream.generation_index != generation
        or source_fq_allocation.proposal_generation != generation
        or source_fq_allocation.proposal_order_alpha_schedule_hash
        != ledger.schedule_hash
        or source_fq_allocation.proposal_order_generation_alpha_allocation
        != ledger.schedule.allocation_for_generation(generation)
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.ALPHA_GENERATION_MISMATCH,
            "independent F/Q alpha does not share the frozen D generation/schedule",
        )
    return V1PhysicalActionCAlphaAllocationReceipt(
        _issuer_token=_ALPHA_TOKEN,
        source_fq_allocation=source_fq_allocation,
        physical_action_axis_hash=axis_hash,
        d_freeze_hash=physical_action_axis.d_freeze_hash,
        certification_stream_hash=c_hash,
        proposal_order_ledger_lineage_hash=ledger_hash,
        proposal_order_alpha_schedule_hash=ledger.schedule_hash,
        proposal_generation=generation,
        score_order_look_index=source_fq_allocation.score_order_look_index,
        fq_familywise_alpha=source_fq_allocation.fq_familywise_alpha,
        fq_alpha_committed_before_generation=(
            source_fq_allocation.fq_alpha_committed_before_generation
        ),
        fq_alpha_committed_before_score_look=(
            source_fq_allocation.fq_alpha_committed_before_redesign
        ),
        physical_c_family_alpha_allocation=(
            source_fq_allocation.fq_redesign_alpha_allocation
        ),
        source_fq_allocation_receipt_hash=source_hash,
    )


@dataclass(frozen=True, slots=True)
class V1PhysicalActionCCoordinatePlan:
    """Compact, exact target/complement C order/draw/packet registration."""

    certification_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt = field(
        repr=False,
        compare=False,
    )
    complete_population: CompleteImageNetPopulationPairReceipt = field(
        repr=False,
        compare=False,
    )
    certification_stream_hash: str
    complete_population_hash: str
    target_permutation_receipt_hash: str
    complement_permutation_receipt_hash: str
    generated_order_hashes: tuple[str, str]
    population_sizes: tuple[int, int]
    maximum_contexts_per_packet: int
    packet_counts: tuple[int, int]
    D_C_domain_separation_proof_hash: str
    _issuer_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    stratum_ids: tuple[str, str] = ("target", "complement")
    draw_ids_derived_preoutcome: bool = True
    packet_axis_derived_preoutcome: bool = True
    outcomes_seen: bool = False
    downstream_authority: bool = False
    authority_scope: str = PHYSICAL_ACTION_C_AUTHORITY
    schema: str = PHYSICAL_ACTION_C_COORDINATE_PLAN_SCHEMA
    _plan_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuer_token is not _PLAN_TOKEN:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "coordinate plans are issued only by the typed C registrar",
            )
        if (
            type(self.certification_stream)
            is not V1AlexNetFQCausalRoleSamplingStreamReceipt
            or type(self.complete_population)
            is not CompleteImageNetPopulationPairReceipt
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
                "coordinate plan requires exact stream and population receipts",
            )
        c_hash = self.certification_stream.assert_integrity()
        population_hash = self.complete_population.content_hash
        for name in (
            "certification_stream_hash",
            "complete_population_hash",
            "target_permutation_receipt_hash",
            "complement_permutation_receipt_hash",
            "D_C_domain_separation_proof_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        receipts = (
            self.certification_stream.target_assignment_receipt,
            self.certification_stream.complement_assignment_receipt,
        )
        expected_receipt_hashes = tuple(value.content_hash for value in receipts)
        expected_order_hashes = tuple(
            v1_physical_action_c_generated_order_hash(
                receipt,
                role="certification",
                stratum=stratum,
            )
            for stratum, receipt in zip(self.stratum_ids, receipts, strict=True)
        )
        sizes = tuple(len(value.generated_order) for value in receipts)
        packet_capacity = _positive_int(
            self.maximum_contexts_per_packet,
            name="maximum contexts per C packet",
        )
        expected_packet_counts = tuple(
            (size + packet_capacity - 1) // packet_capacity for size in sizes
        )
        if (
            self.certification_stream_hash != c_hash
            or self.complete_population_hash != population_hash
            or (
                self.target_permutation_receipt_hash,
                self.complement_permutation_receipt_hash,
            )
            != expected_receipt_hashes
            or tuple(self.generated_order_hashes) != expected_order_hashes
            or tuple(self.population_sizes) != sizes
            or tuple(self.packet_counts) != expected_packet_counts
            or self.stratum_ids != ("target", "complement")
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.STALE_COORDINATE,
                "coordinate plan does not replay the exact C permutations/packets",
            )
        if (
            self.draw_ids_derived_preoutcome is not True
            or self.packet_axis_derived_preoutcome is not True
            or self.outcomes_seen is not False
            or self.downstream_authority is not False
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.POSTSELECTION,
                "coordinate plan contains post-outcome state",
            )
        if (
            self.authority_scope != PHYSICAL_ACTION_C_AUTHORITY
            or self.schema != PHYSICAL_ACTION_C_COORDINATE_PLAN_SCHEMA
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "coordinate-plan schema or authority changed",
            )
        object.__setattr__(self, "generated_order_hashes", expected_order_hashes)
        object.__setattr__(self, "population_sizes", sizes)
        object.__setattr__(self, "maximum_contexts_per_packet", packet_capacity)
        object.__setattr__(self, "packet_counts", expected_packet_counts)
        self.assert_integrity()

    @property
    def assignment_receipts(self) -> tuple[SealedUniformPermutationReceipt, ...]:
        return (
            self.certification_stream.target_assignment_receipt,
            self.certification_stream.complement_assignment_receipt,
        )

    @property
    def plan_hash(self) -> str:
        return self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name
            not in {
                "certification_stream",
                "complete_population",
                "_issuer_token",
                "_plan_hash",
            }
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.certification_stream.assert_integrity()
            != self.certification_stream_hash
            or self.complete_population.content_hash != self.complete_population_hash
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.STALE_COORDINATE,
                "coordinate-plan sources changed",
            )
        observed = content_hash(self._payload(), prefix=_PLAN_PREFIX)
        if self._plan_hash is not None and observed != self._plan_hash:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.STALE_COORDINATE,
                "coordinate plan was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.STALE_COORDINATE,
                "coordinate plan hash differs",
            )
        object.__setattr__(self, "_plan_hash", observed)
        return observed

    def coordinate(
        self,
        *,
        source_registration_hash: str,
        stratum_index: int,
        sealed_order_index: int,
    ) -> V1PhysicalActionSixRoleCCoordinateRegistration:
        """Materialize one exact pre-registered coordinate in O(1) storage."""

        source_hash = _identity(
            source_registration_hash,
            name="physical-C source registration hash",
        )
        stratum_offset = _nonnegative_int(stratum_index, name="C stratum index")
        if stratum_offset >= 2:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.STALE_COORDINATE,
                "C stratum index left target/complement",
            )
        index = _nonnegative_int(sealed_order_index, name="sealed C order index")
        receipt = self.assignment_receipts[stratum_offset]
        if index >= len(receipt.generated_order):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.STALE_COORDINATE,
                "sealed C order index exceeds its complete stratum",
            )
        stratum = self.stratum_ids[stratum_offset]
        context_id = receipt.generated_order[index]
        draw_id = content_hash(
            {
                "coordinate_plan_hash": self.plan_hash,
                "certification_stream_hash": self.certification_stream_hash,
                "permutation_receipt_hash": receipt.content_hash,
                "generated_order_hash": self.generated_order_hashes[stratum_offset],
                "stratum": stratum,
                "sealed_order_index": index,
                "context_id": context_id,
                "outcomes_seen": False,
            },
            prefix=_DRAW_PREFIX,
        )
        start = (index // self.maximum_contexts_per_packet) * (
            self.maximum_contexts_per_packet
        )
        stop = min(
            start + self.maximum_contexts_per_packet, len(receipt.generated_order)
        )
        packet_contexts = receipt.generated_order[start:stop]
        packet_hash = content_hash(
            {
                "coordinate_plan_hash": self.plan_hash,
                "stratum": stratum,
                "packet_index": start // self.maximum_contexts_per_packet,
                "start": start,
                "stop": stop,
                "context_ids": packet_contexts,
                "outcomes_seen": False,
            },
            prefix=_PACKET_PREFIX,
        )
        proof_hash = content_hash(
            {
                "coordinate_plan_hash": self.plan_hash,
                "source_registration_hash": source_hash,
                "stratum": stratum,
                "sealed_order_index": index,
                "context_id": context_id,
                "draw_id": draw_id,
                "expected_context_packet_hash": packet_hash,
                "issued_before_C_outcomes": True,
            },
            prefix=_COORDINATE_PROOF_PREFIX,
        )
        return V1PhysicalActionSixRoleCCoordinateRegistration(
            source_registration_hash=source_hash,
            stratum_id=stratum,
            sealed_order_index=index,
            context_id=context_id,
            draw_id=draw_id,
            C_coordinate_proof_hash=proof_hash,
            expected_context_packet_hash=packet_hash,
        )


def _single_action_analytic_key(
    source: V1AlexNetFQRBoundDerivationReceipt,
) -> tuple[object, str, str]:
    """Return the exact branch/action/mask key for one analytic source."""

    if (
        len(source.endpoint_action_hashes) != 1
        or len(source.endpoint_packed_action_hashes) != 1
        or len(source.endpoint_mask_hashes) != 1
        or len(source.relaxed_logit_receipts) != 1
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
            "physical-action C requires one-action analytic receipts; a prior "
            "nested branch ladder cannot cover independent signed-work actions",
        )
    return (
        source.branch,
        source.endpoint_packed_action_hashes[0],
        source.endpoint_mask_hashes[0],
    )


def _serialized_analytic_key(key: tuple[object, str, str]) -> tuple[str, str, str]:
    branch = key[0]
    value = getattr(branch, "value", None)
    if type(value) is not str or not value:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
            "analytic action key lost its exact Q branch",
        )
    return (value, key[1], key[2])


@dataclass(frozen=True, slots=True)
class V1PhysicalActionCSixRoleAnalyticSupportReceipt:
    """Unique-action six-role support derived only from analytic receipts."""

    physical_action_axis: V1SignedWorkPhysicalActionCAxis = field(
        repr=False,
        compare=False,
    )
    causal_registration: V1AlexNetFQAcquisitionRegistration = field(
        repr=False,
        compare=False,
    )
    source_registration: V1PhysicalActionSixRoleCSourceRegistration = field(
        repr=False,
        compare=False,
    )
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt = field(
        repr=False,
        compare=False,
    )
    discovery_freeze: V1AlexNetFQDiscoveryPrefixFreezeReceipt = field(
        repr=False,
        compare=False,
    )
    analytic_sources: tuple[V1AlexNetFQRBoundDerivationReceipt, ...] = field(
        repr=False,
        compare=False,
    )
    physical_action_axis_hash: str
    coordinate_plan_hash: str
    causal_registration_hash: str
    source_registration_hash: str
    discovery_stream_hash: str
    discovery_freeze_hash: str
    discovery_population_order_receipt_hashes: tuple[str, str]
    chart_identity_hash: str
    proposal_generation: int
    source_receipt_hashes: tuple[str, ...]
    source_action_keys: tuple[tuple[str, str, str], ...]
    analytic_support_upper_by_stratum_action_role: np.ndarray = field(
        repr=False,
        compare=False,
    )
    support_bound_source_hashes_by_role: tuple[tuple[str, str], ...]
    alias_source_crosswalk_hashes_by_action: tuple[tuple[str, ...], ...]
    _issuer_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    support_policy: str = PHYSICAL_ACTION_C_SUPPORT_POLICY
    observed_values_used: bool = False
    hand_entered_support_used: bool = False
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    downstream_authority: bool = False
    authority_scope: str = PHYSICAL_ACTION_C_AUTHORITY
    schema: str = PHYSICAL_ACTION_C_ANALYTIC_SUPPORT_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )
    physical_measurement_profile: str = field(default=LEGACY_UNIFORM_ABSOLUTE, kw_only=True)

    def __setstate__(self, state: list[object]) -> None:
        restore_profile_state(self, state)

    def __post_init__(self) -> None:
        validate_physical_measurement_profile(self.physical_measurement_profile)
        if self._issuer_token is not _SUPPORT_TOKEN:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "analytic supports are issued only by the source-bound derivation",
            )
        if type(self.physical_action_axis) is not V1SignedWorkPhysicalActionCAxis:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
                "analytic support requires the exact physical-action axis",
            )
        axis_hash = self.physical_action_axis.assert_integrity()
        if (
            type(self.causal_registration) is not V1AlexNetFQAcquisitionRegistration
            or type(self.source_registration)
            is not V1PhysicalActionSixRoleCSourceRegistration
            or type(self.discovery_stream)
            is not V1AlexNetFQCausalRoleSamplingStreamReceipt
            or type(self.discovery_freeze)
            is not V1AlexNetFQDiscoveryPrefixFreezeReceipt
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
                "analytic support requires the exact causal D stream and freeze",
            )
        causal_hash = self.causal_registration.assert_integrity()
        source_registration_hash = self.source_registration.assert_integrity()
        d_hash = self.discovery_stream.assert_integrity()
        freeze_hash = self.discovery_freeze.assert_integrity()
        sources = tuple(self.analytic_sources)
        if not sources or any(
            type(value) is not V1AlexNetFQRBoundDerivationReceipt for value in sources
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
                "analytic support requires exact AlexNet F/Q/R derivation receipts",
            )
        observed_hashes = tuple(value.assert_integrity() for value in sources)
        for name in (
            "physical_action_axis_hash",
            "coordinate_plan_hash",
            "causal_registration_hash",
            "source_registration_hash",
            "discovery_stream_hash",
            "discovery_freeze_hash",
            "chart_identity_hash",
            "support_policy",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        actions = self.physical_action_axis.physical_actions
        support = _immutable_f64(
            self.analytic_support_upper_by_stratum_action_role,
            shape=(2, len(actions), len(SIGNED_WORK_SIX_ROLES)),
            name="physical-action six-role analytic support",
        )
        source_axis = tuple(
            (_identity(role, name="support role"), _identity(value, name="source hash"))
            for role, value in self.support_bound_source_hashes_by_role
        )
        aliases = tuple(
            tuple(row) for row in self.alias_source_crosswalk_hashes_by_action
        )
        expected_aliases = tuple(
            tuple(value.crosswalk_hash for value in fanout.source_aliases)
            for fanout in self.physical_action_axis.alias_fanouts
        )
        expected_order_receipts = (
            self.discovery_stream.target_assignment_receipt.content_hash,
            self.discovery_stream.complement_assignment_receipt.content_hash,
        )
        source_keys = tuple(tuple(value) for value in self.source_action_keys)
        expected_source_keys = tuple(
            _serialized_analytic_key(_single_action_analytic_key(source))
            for source in sources
        )
        if (
            self.physical_action_axis_hash != axis_hash
            or self.causal_registration_hash != causal_hash
            or self.source_registration_hash != source_registration_hash
            or self.discovery_stream_hash != d_hash
            or self.discovery_freeze_hash != freeze_hash
            or self.discovery_freeze.discovery_stream_hash != d_hash
            or self.discovery_stream.role != "discovery"
            or tuple(self.discovery_population_order_receipt_hashes)
            != expected_order_receipts
            or tuple(self.source_receipt_hashes) != observed_hashes
            or source_keys != expected_source_keys
            or tuple(role for role, _hash in source_axis) != SIGNED_WORK_SIX_ROLES
            or aliases != expected_aliases
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
                "analytic support lost a source, role, action, or alias coordinate",
            )
        if (
            self.support_policy != PHYSICAL_ACTION_C_SUPPORT_POLICY
            or self.observed_values_used is not False
            or self.hand_entered_support_used is not False
            or self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
            or self.downstream_authority is not False
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.POSTSELECTION,
                "analytic support is not a closed pre-outcome derivation",
            )
        if (
            self.authority_scope != PHYSICAL_ACTION_C_AUTHORITY
            or self.schema != PHYSICAL_ACTION_C_ANALYTIC_SUPPORT_SCHEMA
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "analytic-support schema or authority changed",
            )
        object.__setattr__(self, "analytic_sources", sources)
        object.__setattr__(
            self,
            "discovery_population_order_receipt_hashes",
            expected_order_receipts,
        )
        object.__setattr__(self, "source_receipt_hashes", observed_hashes)
        object.__setattr__(self, "source_action_keys", source_keys)
        object.__setattr__(
            self, "analytic_support_upper_by_stratum_action_role", support
        )
        object.__setattr__(self, "support_bound_source_hashes_by_role", source_axis)
        object.__setattr__(self, "alias_source_crosswalk_hashes_by_action", aliases)
        object.__setattr__(self, "proposal_generation", generation)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "physical_action_axis_hash": self.physical_action_axis_hash,
            "coordinate_plan_hash": self.coordinate_plan_hash,
            "causal_registration_hash": self.causal_registration_hash,
            "source_registration_hash": self.source_registration_hash,
            "discovery_stream_hash": self.discovery_stream_hash,
            "discovery_freeze_hash": self.discovery_freeze_hash,
            "discovery_population_order_receipt_hashes": (
                self.discovery_population_order_receipt_hashes
            ),
            "chart_identity_hash": self.chart_identity_hash,
            "proposal_generation": self.proposal_generation,
            "source_receipt_hashes": self.source_receipt_hashes,
            "source_action_keys": self.source_action_keys,
            "analytic_support_upper_by_stratum_action_role": (
                self.analytic_support_upper_by_stratum_action_role
            ),
            "support_bound_source_hashes_by_role": (
                self.support_bound_source_hashes_by_role
            ),
            "alias_source_crosswalk_hashes_by_action": (
                self.alias_source_crosswalk_hashes_by_action
            ),
            "support_policy": self.support_policy,
            "observed_values_used": self.observed_values_used,
            "hand_entered_support_used": self.hand_entered_support_used,
            "issued_before_C_outcomes": self.issued_before_C_outcomes,
            "C_outcomes_seen": self.C_outcomes_seen,
            "downstream_authority": self.downstream_authority,
            "authority_scope": self.authority_scope,
        } | profile_coordinates(self.physical_measurement_profile)

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.physical_action_axis.assert_integrity()
            != self.physical_action_axis_hash
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
                "physical-action axis changed after support derivation",
            )
        if tuple(value.assert_integrity() for value in self.analytic_sources) != (
            self.source_receipt_hashes
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
                "analytic source receipt changed after support derivation",
            )
        if (
            self.causal_registration.assert_integrity() != self.causal_registration_hash
            or self.source_registration.assert_integrity()
            != self.source_registration_hash
            or self.discovery_stream.assert_integrity() != self.discovery_stream_hash
            or self.discovery_freeze.assert_integrity() != self.discovery_freeze_hash
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
                "causal D source changed after support derivation",
            )
        observed = content_hash(self._payload(), prefix=_SUPPORT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
                "analytic support receipt was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
                "analytic support receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


def derive_v1_physical_action_c_six_role_analytic_support(
    *,
    physical_action_axis: V1SignedWorkPhysicalActionCAxis,
    coordinate_plan: V1PhysicalActionCCoordinatePlan,
    causal_registration: V1AlexNetFQAcquisitionRegistration,
    source_registration: V1PhysicalActionSixRoleCSourceRegistration,
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
    discovery_freeze: V1AlexNetFQDiscoveryPrefixFreezeReceipt,
    chart_identity_hash: str,
    analytic_sources: Sequence[V1AlexNetFQRBoundDerivationReceipt],
    physical_measurement_profile: str = LEGACY_UNIFORM_ABSOLUTE,
) -> V1PhysicalActionCSixRoleAnalyticSupportReceipt:
    """Transport branch analytic proofs to the unique physical-action quotient.

    A duplicated physical action is measured once.  If more than one branch
    proves a bound for that same action, the quotient takes the outward maximum
    of those proofs; it never averages them or chooses the tighter proof after
    seeing C.  ``F_native`` is safely bounded by corrected-work absolute mass,
    because native-fiber signed cancellation can only reduce that mass.
    """

    validate_physical_measurement_profile(physical_measurement_profile)
    if type(physical_action_axis) is not V1SignedWorkPhysicalActionCAxis:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            "support derivation requires the exact physical-action axis",
        )
    if type(coordinate_plan) is not V1PhysicalActionCCoordinatePlan:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            "support derivation requires the exact coordinate plan",
        )
    if (
        type(causal_registration) is not V1AlexNetFQAcquisitionRegistration
        or type(source_registration) is not V1PhysicalActionSixRoleCSourceRegistration
        or type(discovery_stream) is not V1AlexNetFQCausalRoleSamplingStreamReceipt
        or type(discovery_freeze) is not V1AlexNetFQDiscoveryPrefixFreezeReceipt
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            "support derivation requires exact causal, source, D-stream, and "
            "D-freeze owners",
        )
    axis_hash = physical_action_axis.assert_integrity()
    plan_hash = coordinate_plan.assert_integrity()
    causal_hash = causal_registration.assert_integrity()
    source_registration_hash = source_registration.assert_integrity()
    d_hash = discovery_stream.assert_integrity()
    d_freeze_hash = discovery_freeze.assert_integrity()
    chart = _identity(chart_identity_hash, name="chart identity hash")
    supplied_sources = tuple(analytic_sources)
    if not supplied_sources or any(
        type(value) is not V1AlexNetFQRBoundDerivationReceipt
        for value in supplied_sources
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            "support derivation requires exact AlexNet analytic receipts",
        )
    if (
        discovery_stream.role != "discovery"
        or discovery_freeze.discovery_stream_hash != d_hash
        or discovery_freeze.generation_frame_hash
        != discovery_stream.generation_frame_hash
        or discovery_freeze.generation_index != discovery_stream.generation_index
        or physical_action_axis.d_freeze.discovery_stream_commitment_hash != d_hash
        or physical_action_axis.d_freeze.packed_action_family_hash
        != discovery_freeze.selected_shell_hash
        or source_registration.physical_action_axis_hash != axis_hash
        or source_registration.C_coordinate_proof_source_registration_hash != plan_hash
        or source_registration.chart_identity_hash != chart
        or source_registration.raw_logit_coordinate_identity_hash
        != discovery_stream.frozen_transform_hash
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
            "analytic support does not descend from the exact frozen D chart/action axis",
        )
    generation = physical_action_axis.alpha_ledger_generation
    expected_counts = coordinate_plan.population_sizes
    expected_d_order_receipts = (
        discovery_stream.target_assignment_receipt.content_hash,
        discovery_stream.complement_assignment_receipt.content_hash,
    )
    indexed_sources: dict[
        tuple[object, str, str], V1AlexNetFQRBoundDerivationReceipt
    ] = {}
    for source in supplied_sources:
        source.assert_integrity()
        key = _single_action_analytic_key(source)
        if key in indexed_sources:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
                "duplicate analytic receipts name the same branch/action/mask key",
            )
        if (
            source.campaign_contract_hash != causal_registration.campaign_contract_hash
            or source.model_identity_hash
            != causal_registration.expected_model_identity_hash
            or source.model_state_hash != causal_registration.expected_model_state_hash
            or source.query_contract_hash != causal_registration.query_contract_hash
            or source.preprocessing_hash != causal_registration.preprocessing_hash
            or source.current_mask_hash != physical_action_axis.parent_mask_hash
            or source.route_axis_hash != physical_action_axis.route_axis_hash
            or source.route_count != physical_action_axis.route_count
            or source.proposal_generation != generation
            or source.chart_identity_hash != chart
            or source.raw_logit_stream_source_hash
            != source_registration.raw_logit_coordinate_identity_hash
            or source.input_box_hash != causal_registration.normalized_input_box_hash
            or source.stratum_names != ("target", "complement")
            or source.population_stratum_counts != expected_counts
            or source.sealed_population_order_hashes != expected_d_order_receipts
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
                "an analytic receipt belongs to another chart/action/population axis",
            )
        indexed_sources[key] = source
    expected_keys: set[tuple[object, str, str]] = set()
    for fanout in physical_action_axis.alias_fanouts:
        for alias in fanout.source_aliases:
            packed = alias.packed_action
            if packed is None:
                _refuse(
                    V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
                    "a Q-safe physical alias lost its exact packed action",
                )
            expected_keys.add(
                (
                    alias.q_branch,
                    packed.action_hash,
                    packed.resulting_mask.mask_hash,
                )
            )
    if set(indexed_sources) != expected_keys:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
            "one-action analytic receipts do not exactly cover every unique "
            "branch/action/mask alias",
        )
    canonical_keys = tuple(sorted(expected_keys, key=_serialized_analytic_key))
    sources = tuple(indexed_sources[key] for key in canonical_keys)
    source_hashes = tuple(value.receipt_hash for value in sources)
    actions = physical_action_axis.physical_actions
    support = np.empty((2, len(actions), len(SIGNED_WORK_SIX_ROLES)), dtype=_LE_F64)
    alias_rows: list[tuple[str, ...]] = []
    support_sources: dict[str, list[dict[str, object]]] = {
        role: [] for role in SIGNED_WORK_SIX_ROLES
    }
    role_index = {role: index for index, role in enumerate(SIGNED_WORK_SIX_ROLES)}
    for action_offset, fanout in enumerate(physical_action_axis.alias_fanouts):
        alias_rows.append(
            tuple(value.crosswalk_hash for value in fanout.source_aliases)
        )
        values_by_role: dict[str, list[object]] = {
            role: [] for role in SIGNED_WORK_SIX_ROLES
        }
        for alias in fanout.source_aliases:
            packed = alias.packed_action
            if packed is None:
                _refuse(
                    V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
                    "a physical-action alias lost its packed action",
                )
            source_key = (
                alias.q_branch,
                packed.action_hash,
                packed.resulting_mask.mask_hash,
            )
            source = indexed_sources.get(source_key)
            if source is None:
                _refuse(
                    V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
                    "a physical-action alias lacks its branch analytic receipt",
                )
            # Every source is a newly frozen one-action ladder.  The upstream
            # Phase-B endpoint label remains provenance only; it is not an
            # index into a different (possibly nested) branch ladder.
            endpoint = 0
            if (
                source.endpoint_packed_action_hashes[0] != packed.action_hash
                or source.endpoint_mask_hashes[0] != packed.resulting_mask.mask_hash
            ):
                _refuse(
                    V1PhysicalActionCRegistrationRefusalCode.ANALYTIC_SUPPORT_INVALID,
                    "analytic endpoint does not describe the exact packed physical action",
                )
            parent_lower, parent_upper = source.parent_raw_target_logit_interval
            f_values = tuple(
                source.corrected_f_upper_by_stratum_endpoint[stratum][endpoint]
                for stratum in range(2)
            )
            q_value = source.q_upper_by_endpoint[endpoint]
            r_values = source.raw_interval_remainder_upper_by_endpoint_and_dose[
                endpoint
            ]
            d_values: list[float] = []
            for relaxed in source.relaxed_logit_receipts[endpoint]:
                direct = max(
                    abs(relaxed.raw_target_logit_lower - parent_upper),
                    abs(relaxed.raw_target_logit_upper - parent_lower),
                )
                d_values.append(direct)
            values_by_role[F_NATIVE_ROLE].append(f_values)
            values_by_role[Q_J_ROLE].append(q_value)
            values_by_role[R_HALF_ROLE].append(r_values[0])
            values_by_role[R_ONE_ROLE].append(r_values[1])
            values_by_role[D_HALF_ROLE].append(d_values[0])
            values_by_role[D_ONE_ROLE].append(d_values[1])
            source_common = {
                "analytic_receipt_hash": source.receipt_hash,
                "alias_crosswalk_hash": alias.crosswalk_hash,
                "branch": alias.q_branch,
                "source_endpoint_label": alias.source_endpoint_label,
                "packed_action_hash": packed.action_hash,
                "resulting_mask_hash": packed.resulting_mask.mask_hash,
            }
            for role in SIGNED_WORK_SIX_ROLES:
                support_sources[role].append(source_common)
        for stratum in range(2):
            f_max = max(
                tuple(value)[stratum] for value in values_by_role[F_NATIVE_ROLE]
            )
            support[stratum, action_offset, role_index[F_NATIVE_ROLE]] = (
                _outward_nonnegative(f_max, name="native-F analytic support")
            )
            for role in SIGNED_WORK_SIX_ROLES[1:]:
                role_max = max(float(value) for value in values_by_role[role])
                role_bound = _outward_nonnegative(role_max, name=f"{role} analytic support")
                if physical_measurement_profile == BALANCED_RMS and role in (
                    R_HALF_ROLE, R_ONE_ROLE, D_HALF_ROLE, D_ONE_ROLE
                ):
                    role_bound = squared_support_upper(role_bound)
                support[stratum, action_offset, role_index[role]] = role_bound
    source_hashes_by_role = tuple(
        (
            role,
            content_hash(
                {
                    "policy": PHYSICAL_ACTION_C_SUPPORT_POLICY,
                    "role": role,
                    "physical_action_axis_hash": axis_hash,
                    "coordinate_plan_hash": plan_hash,
                    "source_receipts": source_hashes,
                    "alias_sources": support_sources[role],
                    "F_native_bounded_by_route_absolute_corrected_work": (
                        role == F_NATIVE_ROLE
                    ),
                    "unique_action_alias_reduction": "outward_max_preoutcome",
                    "observed_values_used": False,
                } | profile_coordinates(physical_measurement_profile),
                prefix=_SUPPORT_SOURCE_PREFIX,
            ),
        )
        for role in SIGNED_WORK_SIX_ROLES
    )
    return V1PhysicalActionCSixRoleAnalyticSupportReceipt(
        physical_measurement_profile=physical_measurement_profile,
        _issuer_token=_SUPPORT_TOKEN,
        physical_action_axis=physical_action_axis,
        causal_registration=causal_registration,
        source_registration=source_registration,
        discovery_stream=discovery_stream,
        discovery_freeze=discovery_freeze,
        analytic_sources=sources,
        physical_action_axis_hash=axis_hash,
        coordinate_plan_hash=plan_hash,
        causal_registration_hash=causal_hash,
        source_registration_hash=source_registration_hash,
        discovery_stream_hash=d_hash,
        discovery_freeze_hash=d_freeze_hash,
        discovery_population_order_receipt_hashes=expected_d_order_receipts,
        chart_identity_hash=chart,
        proposal_generation=generation,
        source_receipt_hashes=source_hashes,
        source_action_keys=tuple(
            _serialized_analytic_key(key) for key in canonical_keys
        ),
        analytic_support_upper_by_stratum_action_role=support,
        support_bound_source_hashes_by_role=source_hashes_by_role,
        alias_source_crosswalk_hashes_by_action=tuple(alias_rows),
    )


@dataclass(frozen=True, slots=True)
class V1PhysicalActionCCanonicalRegistration:
    """Closed canonical C registration and its ready six-role design."""

    complete_population: CompleteImageNetPopulationPairReceipt = field(
        repr=False,
        compare=False,
    )
    causal_registration: V1AlexNetFQAcquisitionRegistration = field(
        repr=False,
        compare=False,
    )
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt = field(
        repr=False,
        compare=False,
    )
    discovery_freeze: V1AlexNetFQDiscoveryPrefixFreezeReceipt = field(
        repr=False,
        compare=False,
    )
    certification_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt = field(
        repr=False,
        compare=False,
    )
    physical_action_axis: V1SignedWorkPhysicalActionCAxis = field(
        repr=False,
        compare=False,
    )
    direction_provider_receipt: V1PhysicalActionDirectionReceipt = field(
        repr=False,
        compare=False,
    )
    alpha_allocation: V1PhysicalActionCAlphaAllocationReceipt
    coordinate_plan: V1PhysicalActionCCoordinatePlan
    analytic_support: V1PhysicalActionCSixRoleAnalyticSupportReceipt
    source_registration: V1PhysicalActionSixRoleCSourceRegistration
    six_role_design: V1PhysicalActionSixRoleCDesign
    complete_population_hash: str
    causal_registration_hash: str
    discovery_stream_hash: str
    discovery_freeze_hash: str
    certification_stream_hash: str
    physical_action_axis_hash: str
    direction_provider_receipt_hash: str
    alpha_allocation_receipt_hash: str
    coordinate_plan_hash: str
    analytic_support_receipt_hash: str
    source_registration_hash: str
    six_role_design_hash: str
    D_C_domain_separation_proof_hash: str
    _issuer_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    production_execution: bool = True
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    K_open_authority: bool = False
    O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = PHYSICAL_ACTION_C_AUTHORITY
    schema: str = PHYSICAL_ACTION_C_REGISTRATION_SCHEMA
    _registration_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuer_token is not _REGISTRATION_TOKEN:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "canonical physical-C registrations require the typed factory",
            )
        exact_types = (
            (self.complete_population, CompleteImageNetPopulationPairReceipt),
            (self.causal_registration, V1AlexNetFQAcquisitionRegistration),
            (self.discovery_stream, V1AlexNetFQCausalRoleSamplingStreamReceipt),
            (self.discovery_freeze, V1AlexNetFQDiscoveryPrefixFreezeReceipt),
            (self.certification_stream, V1AlexNetFQCausalRoleSamplingStreamReceipt),
            (self.physical_action_axis, V1SignedWorkPhysicalActionCAxis),
            (self.direction_provider_receipt, V1PhysicalActionDirectionReceipt),
            (self.alpha_allocation, V1PhysicalActionCAlphaAllocationReceipt),
            (self.coordinate_plan, V1PhysicalActionCCoordinatePlan),
            (self.analytic_support, V1PhysicalActionCSixRoleAnalyticSupportReceipt),
            (self.source_registration, V1PhysicalActionSixRoleCSourceRegistration),
            (self.six_role_design, V1PhysicalActionSixRoleCDesign),
        )
        if any(type(value) is not expected for value, expected in exact_types):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
                "canonical physical-C registration lost an exact typed owner",
            )
        observed = {
            "complete_population_hash": self.complete_population.content_hash,
            "causal_registration_hash": self.causal_registration.assert_integrity(),
            "discovery_stream_hash": self.discovery_stream.assert_integrity(),
            "discovery_freeze_hash": self.discovery_freeze.assert_integrity(),
            "certification_stream_hash": self.certification_stream.assert_integrity(),
            "physical_action_axis_hash": self.physical_action_axis.assert_integrity(),
            "direction_provider_receipt_hash": (
                self.direction_provider_receipt.assert_integrity()
            ),
            "alpha_allocation_receipt_hash": self.alpha_allocation.assert_integrity(),
            "coordinate_plan_hash": self.coordinate_plan.assert_integrity(),
            "analytic_support_receipt_hash": self.analytic_support.assert_integrity(),
            "source_registration_hash": self.source_registration.assert_integrity(),
            "six_role_design_hash": self.six_role_design.assert_integrity(),
        }
        for name in (*observed, "D_C_domain_separation_proof_hash", "authority_scope"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        stale = tuple(
            name for name, value in observed.items() if getattr(self, name) != value
        )
        if stale:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.STALE_SOURCE,
                "canonical physical-C owner hashes differ at: " + ", ".join(stale),
            )
        if (self.six_role_design.physical_measurement_profile
                != self.analytic_support.physical_measurement_profile):
            _refuse(V1PhysicalActionCRegistrationRefusalCode.STALE_SOURCE,
                    "physical-C design and analytic support use different moment laws")
        if self.coordinate_plan.D_C_domain_separation_proof_hash != (
            self.D_C_domain_separation_proof_hash
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.DOMAIN_SEPARATION_FAILURE,
                "canonical registration changed the D/C separation proof",
            )
        if (
            type(self.production_execution) is not bool
            or type(self.issued_before_C_outcomes) is not bool
            or not self.issued_before_C_outcomes
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.POSTSELECTION,
                "canonical physical-C registration was not pre-outcome",
            )
        _closed_authority(
            (
                self.C_outcomes_seen,
                self.K_open_authority,
                self.O_open_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="canonical physical-action C registration",
        )
        if (
            self.authority_scope != PHYSICAL_ACTION_C_AUTHORITY
            or self.schema != PHYSICAL_ACTION_C_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.AUTHORITY_VIOLATION,
                "canonical physical-C schema or authority changed",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name
            not in {
                "complete_population",
                "causal_registration",
                "discovery_stream",
                "discovery_freeze",
                "certification_stream",
                "physical_action_axis",
                "direction_provider_receipt",
                "alpha_allocation",
                "coordinate_plan",
                "analytic_support",
                "source_registration",
                "six_role_design",
                "_issuer_token",
                "_registration_hash",
            }
        }

    @property
    def registration_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        nested = (
            self.complete_population.content_hash,
            self.causal_registration.assert_integrity(),
            self.discovery_stream.assert_integrity(),
            self.discovery_freeze.assert_integrity(),
            self.certification_stream.assert_integrity(),
            self.physical_action_axis.assert_integrity(),
            self.direction_provider_receipt.assert_integrity(),
            self.alpha_allocation.assert_integrity(),
            self.coordinate_plan.assert_integrity(),
            self.analytic_support.assert_integrity(),
            self.source_registration.assert_integrity(),
            self.six_role_design.assert_integrity(),
        )
        recorded = tuple(
            getattr(self, name)
            for name in (
                "complete_population_hash",
                "causal_registration_hash",
                "discovery_stream_hash",
                "discovery_freeze_hash",
                "certification_stream_hash",
                "physical_action_axis_hash",
                "direction_provider_receipt_hash",
                "alpha_allocation_receipt_hash",
                "coordinate_plan_hash",
                "analytic_support_receipt_hash",
                "source_registration_hash",
                "six_role_design_hash",
            )
        )
        if nested != recorded:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.STALE_SOURCE,
                "canonical physical-C nested source changed",
            )
        observed = content_hash(self._payload(), prefix=_REGISTRATION_PREFIX)
        if self._registration_hash is not None and observed != self._registration_hash:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.STALE_SOURCE,
                "canonical physical-C registration was tampered",
            )
        if expected_hash is not None and observed != expected_hash:
            _refuse(
                V1PhysicalActionCRegistrationRefusalCode.STALE_SOURCE,
                "canonical physical-C registration hash differs",
            )
        object.__setattr__(self, "_registration_hash", observed)
        return observed

    def coordinate(
        self,
        *,
        stratum_index: int,
        sealed_order_index: int,
    ) -> V1PhysicalActionSixRoleCCoordinateRegistration:
        return self.coordinate_plan.coordinate(
            source_registration_hash=self.source_registration_hash,
            stratum_index=stratum_index,
            sealed_order_index=sealed_order_index,
        )


def register_v1_signed_work_physical_action_c(
    *,
    design_id: str,
    complete_population: CompleteImageNetPopulationPairReceipt,
    causal_registration: V1AlexNetFQAcquisitionRegistration,
    discovery_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
    discovery_freeze: V1AlexNetFQDiscoveryPrefixFreezeReceipt,
    certification_stream: V1AlexNetFQCausalRoleSamplingStreamReceipt,
    physical_action_axis: V1SignedWorkPhysicalActionCAxis,
    direction_provider_receipt: V1PhysicalActionDirectionReceipt,
    source_fq_allocation: V1FQAlphaFamilyAllocationReceipt,
    analytic_sources: Sequence[V1AlexNetFQRBoundDerivationReceipt],
    maximum_contexts_per_packet: int,
    chart_identity_hash: str,
    corrected_evidence_stream_identity_hash: str,
    literal_jacobian_stream_identity_hash: str,
    same_capture_scorer_registration_hash: str,
    native_fiber_axis_receipt_hash: str,
    native_fiber_measure_hash: str,
    C_execution_registration_hash: str,
    physical_measurement_profile: str = LEGACY_UNIFORM_ABSOLUTE,
    production_execution: bool = True,
    _allow_private_small_population_reference: bool = False,
) -> V1PhysicalActionCCanonicalRegistration:
    """Close the canonical D -> physical-action C design before C outcomes."""

    validate_physical_measurement_profile(physical_measurement_profile)
    if type(complete_population) is not CompleteImageNetPopulationPairReceipt:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            "canonical physical C requires the exact ImageNet population pair",
        )
    if type(causal_registration) is not V1AlexNetFQAcquisitionRegistration:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            "canonical physical C requires the exact causal source registration",
        )
    if (
        type(discovery_stream) is not V1AlexNetFQCausalRoleSamplingStreamReceipt
        or type(discovery_freeze) is not V1AlexNetFQDiscoveryPrefixFreezeReceipt
        or type(certification_stream) is not V1AlexNetFQCausalRoleSamplingStreamReceipt
        or type(physical_action_axis) is not V1SignedWorkPhysicalActionCAxis
        or type(direction_provider_receipt) is not V1PhysicalActionDirectionReceipt
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            "canonical physical C requires exact D/C/action/provider owners",
        )
    population_hash = complete_population.content_hash
    causal_hash = causal_registration.assert_integrity()
    d_hash = discovery_stream.assert_integrity()
    freeze_hash = discovery_freeze.assert_integrity()
    c_hash = certification_stream.assert_integrity()
    axis_hash = physical_action_axis.assert_integrity()
    provider_hash = direction_provider_receipt.assert_integrity()
    if (
        causal_registration.source_context_axis_hash
        != complete_population.train_axis_hash
        or causal_registration.source_label_map_hash
        != complete_population.label_map_hash
        or causal_registration.preprocessing_hash
        != complete_population.preprocessing_hash
        or discovery_stream.registration_hash != causal_hash
        or certification_stream.registration_hash != causal_hash
        or discovery_stream.target_population_membership_hash
        != causal_registration.target_population_membership_hash
        or discovery_stream.complement_population_membership_hash
        != causal_registration.complement_population_membership_hash
        or certification_stream.target_population_membership_hash
        != causal_registration.target_population_membership_hash
        or certification_stream.complement_population_membership_hash
        != causal_registration.complement_population_membership_hash
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.STALE_POPULATION,
            "causal source registration differs from complete ImageNet train",
        )
    d_receipts = (
        discovery_stream.target_assignment_receipt,
        discovery_stream.complement_assignment_receipt,
    )
    c_receipts = (
        certification_stream.target_assignment_receipt,
        certification_stream.complement_assignment_receipt,
    )
    counts = tuple(len(value.canonical_population) for value in c_receipts)
    # The exact causal stream factory proves intra- and cross-stratum
    # uniqueness before issuing these immutable permutation receipts.  Their
    # integrity hashes and the registered membership hashes above carry that
    # proof forward; do not rebuild a second 1.28M-element tuple and set here.
    if (
        tuple(len(value.canonical_population) for value in d_receipts) != counts
        or counts
        != (
            causal_registration.target_population_count,
            causal_registration.complement_population_count,
        )
        or (
            not _allow_private_small_population_reference
            and sum(counts) != IMAGENET1K_TRAIN_CONTEXT_COUNT
        )
        or (
            not _allow_private_small_population_reference
            and sum(counts) != complete_population.train_context_count
        )
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.STALE_POPULATION,
            "D/C strata do not form the exact complete ImageNet train population",
        )
    if type(production_execution) is not bool:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.INVALID_TYPE,
            "production execution flag must be exact",
        )
    if production_execution and (
        direction_provider_receipt.production_execution is not True
        or direction_provider_receipt.backend_kind != "torch_cuda"
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.AUTHORITY_VIOLATION,
            "canonical production C requires the bounded CUDA direction provider",
        )
    if not production_execution and not _allow_private_small_population_reference:
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.AUTHORITY_VIOLATION,
            "nonproduction physical C is available only to the private small reference",
        )
    dc_proof = v1_physical_action_dc_domain_separation_proof_hash(
        discovery_stream=discovery_stream,
        discovery_freeze=discovery_freeze,
        certification_stream=certification_stream,
    )
    d_order_hash = v1_physical_action_c_stream_order_hash(discovery_stream)
    c_order_hash = v1_physical_action_c_stream_order_hash(certification_stream)
    axis_d = physical_action_axis.d_freeze
    axis_c = physical_action_axis.c_stream
    if (
        axis_d.discovery_stream_commitment_hash != d_hash
        or axis_d.discovery_sampling_order_hash != d_order_hash
        or axis_d.parent_mask_hash != discovery_stream.current_mask_hash
        or axis_d.alpha_ledger_generation != discovery_stream.generation_index
        or axis_c.certification_stream_commitment_hash != c_hash
        or axis_c.certification_sampling_order_hash != c_order_hash
        or axis_c.D_C_domain_separation_receipt_hash != dc_proof
        or axis_c.complete_population_hash != population_hash
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.STALE_D_FREEZE,
            "physical action axis is not descended from the exact typed D/C streams",
        )
    if (
        direction_provider_receipt.source_family_hash
        != physical_action_axis.packed_action_family_hash
        or direction_provider_receipt.parent_mask_hash
        != physical_action_axis.parent_mask_hash
        or direction_provider_receipt.unique_physical_action_count
        != physical_action_axis.physical_action_count
        or direction_provider_receipt.alias_count
        != physical_action_axis.upstream_q_safe_alias_count
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.ACTION_AXIS_MISMATCH,
            "direction provider differs from the deduplicated physical-action axis",
        )
    alpha = allocate_v1_physical_action_c_alpha(
        physical_action_axis=physical_action_axis,
        certification_stream=certification_stream,
        source_fq_allocation=source_fq_allocation,
    )
    generation = physical_action_axis.alpha_ledger_generation
    if (
        causal_registration.maximum_proposal_generations <= generation
        or source_fq_allocation.fq_familywise_alpha
        != causal_registration.fq_familywise_alpha
        or source_fq_allocation.fq_generation_alpha_allocation
        != causal_registration.fq_generation_allocations[generation]
        or source_fq_allocation.proposal_order_generation_alpha_allocation
        != causal_registration.proposal_order_generation_allocations[generation]
        or physical_action_axis.alpha_ledger_head.schedule.trajectory_familywise_alpha
        != causal_registration.proposal_order_familywise_alpha
    ):
        _refuse(
            V1PhysicalActionCRegistrationRefusalCode.ALPHA_GENERATION_MISMATCH,
            "physical-C alpha/lineage differs from the causal source registration",
        )
    packet_capacity = _positive_int(
        maximum_contexts_per_packet,
        name="maximum contexts per packet",
    )
    order_hashes = tuple(
        v1_physical_action_c_generated_order_hash(
            receipt,
            role="certification",
            stratum=stratum,
        )
        for stratum, receipt in zip(("target", "complement"), c_receipts, strict=True)
    )
    coordinate_plan = V1PhysicalActionCCoordinatePlan(
        _issuer_token=_PLAN_TOKEN,
        certification_stream=certification_stream,
        complete_population=complete_population,
        certification_stream_hash=c_hash,
        complete_population_hash=population_hash,
        target_permutation_receipt_hash=c_receipts[0].content_hash,
        complement_permutation_receipt_hash=c_receipts[1].content_hash,
        generated_order_hashes=(order_hashes[0], order_hashes[1]),
        population_sizes=(counts[0], counts[1]),
        maximum_contexts_per_packet=packet_capacity,
        packet_counts=tuple(
            (value + packet_capacity - 1) // packet_capacity for value in counts
        ),
        D_C_domain_separation_proof_hash=dc_proof,
    )
    chart_identity = _identity(chart_identity_hash, name="chart identity hash")
    execution_registration = _identity(
        C_execution_registration_hash,
        name="C execution registration hash",
    )
    raw_logit_coordinate_identity = discovery_stream.frozen_transform_hash
    raw_logit_execution_stream_identity = derive_v1_physical_action_c_finite_direction_raw_logit_execution_stream_identity(
        raw_logit_coordinate_identity_hash=raw_logit_coordinate_identity,
        certification_stream_hash=c_hash,
        physical_action_axis_hash=axis_hash,
        direction_provider_receipt_hash=provider_hash,
        chart_identity_hash=chart_identity,
        C_execution_registration_hash=execution_registration,
    )
    source = V1PhysicalActionSixRoleCSourceRegistration(
        physical_action_axis_hash=axis_hash,
        direction_provider_receipt_hash=provider_hash,
        c_stream_hash=physical_action_axis.c_stream_hash,
        complete_train_population_hash=population_hash,
        chart_identity_hash=chart_identity,
        current_parent_mask_hash=physical_action_axis.parent_mask_hash,
        route_axis_hash=physical_action_axis.route_axis_hash,
        route_count=physical_action_axis.route_count,
        proposal_generation=physical_action_axis.alpha_ledger_generation,
        corrected_evidence_stream_identity_hash=_identity(
            corrected_evidence_stream_identity_hash,
            name="corrected evidence stream identity hash",
        ),
        literal_jacobian_stream_identity_hash=_identity(
            literal_jacobian_stream_identity_hash,
            name="literal Jacobian stream identity hash",
        ),
        raw_logit_coordinate_identity_hash=raw_logit_coordinate_identity,
        raw_logit_stream_identity_hash=raw_logit_execution_stream_identity,
        same_capture_scorer_registration_hash=_identity(
            same_capture_scorer_registration_hash,
            name="same-capture scorer registration hash",
        ),
        native_fiber_axis_receipt_hash=_identity(
            native_fiber_axis_receipt_hash,
            name="native-fiber axis receipt hash",
        ),
        native_fiber_measure_hash=_identity(
            native_fiber_measure_hash,
            name="native-fiber measure hash",
        ),
        C_coordinate_proof_source_registration_hash=coordinate_plan.plan_hash,
        C_execution_registration_hash=execution_registration,
    )
    support = derive_v1_physical_action_c_six_role_analytic_support(
        physical_measurement_profile=physical_measurement_profile,
        physical_action_axis=physical_action_axis,
        coordinate_plan=coordinate_plan,
        causal_registration=causal_registration,
        source_registration=source,
        discovery_stream=discovery_stream,
        discovery_freeze=discovery_freeze,
        chart_identity_hash=source.chart_identity_hash,
        analytic_sources=analytic_sources,
    )
    strata = tuple(
        V1PhysicalActionSixRoleCStratum(
            stratum_id=stratum,
            population_size=counts[index],
            ordered_context_axis_hash=order_hashes[index],
            sealed_draw_axis_hash=c_receipts[index].content_hash,
            coordinate_proof_commitment_hash=content_hash(
                {
                    "coordinate_plan_hash": coordinate_plan.plan_hash,
                    "source_registration_hash": source.registration_hash,
                    "stratum": stratum,
                    "population_size": counts[index],
                    "generated_order_hash": order_hashes[index],
                    "issued_before_C_outcomes": True,
                },
                prefix=_COORDINATE_PROOF_PREFIX,
            ),
        )
        for index, stratum in enumerate(("target", "complement"))
    )
    design = V1PhysicalActionSixRoleCDesign(
        physical_measurement_profile=physical_measurement_profile,
        design_id=_identity(design_id, name="physical-C design id"),
        physical_action_axis=physical_action_axis,
        direction_provider_receipt=direction_provider_receipt,
        source_registration=source,
        strata=strata,
        analytic_support_upper_by_stratum_action_role=(
            support.analytic_support_upper_by_stratum_action_role
        ),
        support_bound_source_hashes_by_role=(
            support.support_bound_source_hashes_by_role
        ),
        six_role_family_alpha_allocation=(alpha.physical_c_family_alpha_allocation),
        physical_c_alpha_allocation_receipt=alpha,
        private_small_population_reference=_allow_private_small_population_reference,
    )
    return V1PhysicalActionCCanonicalRegistration(
        _issuer_token=_REGISTRATION_TOKEN,
        complete_population=complete_population,
        causal_registration=causal_registration,
        discovery_stream=discovery_stream,
        discovery_freeze=discovery_freeze,
        certification_stream=certification_stream,
        physical_action_axis=physical_action_axis,
        direction_provider_receipt=direction_provider_receipt,
        alpha_allocation=alpha,
        coordinate_plan=coordinate_plan,
        analytic_support=support,
        source_registration=source,
        six_role_design=design,
        complete_population_hash=population_hash,
        causal_registration_hash=causal_hash,
        discovery_stream_hash=d_hash,
        discovery_freeze_hash=freeze_hash,
        certification_stream_hash=c_hash,
        physical_action_axis_hash=axis_hash,
        direction_provider_receipt_hash=provider_hash,
        alpha_allocation_receipt_hash=alpha.receipt_hash,
        coordinate_plan_hash=coordinate_plan.plan_hash,
        analytic_support_receipt_hash=support.receipt_hash,
        source_registration_hash=source.registration_hash,
        six_role_design_hash=design.design_hash,
        D_C_domain_separation_proof_hash=dc_proof,
        production_execution=production_execution,
    )


__all__ = [
    "PHYSICAL_ACTION_C_ALPHA_POLICY",
    "PHYSICAL_ACTION_C_AUTHORITY",
    "PHYSICAL_ACTION_C_SUPPORT_POLICY",
    "V1PhysicalActionCAlphaAllocationReceipt",
    "V1PhysicalActionCCanonicalRegistration",
    "V1PhysicalActionCCoordinatePlan",
    "V1PhysicalActionCRegistrationRefusal",
    "V1PhysicalActionCRegistrationRefusalCode",
    "V1PhysicalActionCSixRoleAnalyticSupportReceipt",
    "allocate_v1_physical_action_c_alpha",
    "derive_v1_physical_action_c_finite_direction_raw_logit_execution_stream_identity",
    "derive_v1_physical_action_c_six_role_analytic_support",
    "register_v1_signed_work_physical_action_c",
    "v1_physical_action_c_generated_order_hash",
    "v1_physical_action_c_stream_order_hash",
    "v1_physical_action_dc_domain_separation_proof_hash",
]
