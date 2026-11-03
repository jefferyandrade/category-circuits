"""Freeze unique signed-work physical actions as an authority-inert C axis.

The endogenous signed-work coordinator preserves coordinate and Q-branch
aliases, but C measures a *physical mask action*.  This module is the narrow
identity bridge between those representations.  Every exact physical action
is registered once, while a separate fanout receipt retains every upstream
alias.  The axis deliberately has no nesting or branch semantics: aliases are
provenance, not extra C measurements.

The bridge binds an exact pre-C execution bundle, the complete ImageNet
population receipt, one immutable non-recycling alpha-ledger head, and typed D
and C stream freezes.  It sees no C result and grants no C, K, O, commit,
terminal, or Pareto authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from numbers import Integral
from typing import Final, NoReturn

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_candidate_composition import V1PackedRemovalAction
from .modern_v1_imagenet_population import CompleteImageNetPopulationPairReceipt
from .modern_v1_proposal_alpha_ledger import V1ProposalOrderAlphaLedger
from .modern_v1_phase_a_true_q_d_decision_witness import (
    V1PhaseATrueQDDecisionWitnessReceipt,
)
from .modern_v1_sampled_v7_d_decision_stability import (
    V1SampledV7DDecisionStabilityReceipt,
)
from .modern_v1_signed_work_pre_c_coordinator import (
    V1SignedWorkPreCExecutionBundle,
)
from .modern_v1_signed_work_pre_c_packed_actions import (
    V1SignedWorkCanonicalPackedPhysicalAction,
    V1SignedWorkCoordinateCandidateCrosswalk,
    V1SignedWorkPackedActionDedupIdentity,
    v1_signed_work_packed_action_dedup_identity,
)


SIGNED_WORK_PHYSICAL_ACTION_D_FREEZE_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_action_D_freeze_v1"
)
SIGNED_WORK_PHYSICAL_ACTION_C_STREAM_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_action_C_stream_freeze_v1"
)
SIGNED_WORK_TRUE_Q_PHYSICAL_ACTION_C_ADMISSION_SCHEMA: Final = (
    "partition_free_v1_signed_work_true_q_physical_action_C_admission_v1"
)
SIGNED_WORK_D_STABLE_PHYSICAL_ACTION_C_ADMISSION_SCHEMA: Final = (
    "partition_free_v1_signed_work_full_D_stable_physical_action_C_admission_v2"
)
SIGNED_WORK_PHYSICAL_ACTION_C_ENTRY_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_action_C_axis_entry_v1"
)
SIGNED_WORK_PHYSICAL_ACTION_ALIAS_FANOUT_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_action_alias_fanout_v1"
)
SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_SCHEMA: Final = (
    "partition_free_v1_signed_work_physical_action_C_axis_v1"
)
SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY: Final = (
    "pre_C_unique_physical_action_axis_and_complete_alias_fanout_identity_only_"
    "no_C_outcome_measurement_K_O_calibration_oracle_commit_terminal_or_pareto_"
    "authority_v1"
)
SIGNED_WORK_PHYSICAL_ACTION_PRIORITY_POLICY: Final = (
    "deterministic_pre_C_canonical_first_physical_occurrence_no_C_outcome_v1"
)

_D_FREEZE_PREFIX: Final = "partition-free-v1-signed-work-physical-D-freeze-"
_C_STREAM_PREFIX: Final = "partition-free-v1-signed-work-physical-C-stream-"
_TRUE_Q_C_ADMISSION_PREFIX: Final = (
    "partition-free-v1-signed-work-true-q-physical-C-admission-"
)
_D_STABLE_C_ADMISSION_PREFIX: Final = (
    "partition-free-v1-signed-work-full-D-stable-physical-C-admission-"
)
_ENTRY_PREFIX: Final = "partition-free-v1-signed-work-physical-C-entry-"
_FANOUT_PREFIX: Final = "partition-free-v1-signed-work-physical-alias-fanout-"
_AXIS_PREFIX: Final = "partition-free-v1-signed-work-physical-C-axis-"
_PRIORITY_AXIS_PREFIX: Final = (
    "partition-free-v1-signed-work-physical-pre-C-priority-axis-"
)
_D_FREEZE_TOKEN: Final = object()
_C_STREAM_TOKEN: Final = object()
_TRUE_Q_C_ADMISSION_TOKEN: Final = object()
_D_STABLE_C_ADMISSION_TOKEN: Final = object()
_AXIS_TOKEN: Final = object()


class V1SignedWorkPhysicalActionCAxisRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    HASH_MISMATCH = "hash_mismatch"
    AMBIGUOUS_HARD_FQ = "ambiguous_hard_fq"
    EMPTY_ACTION_AXIS = "empty_action_axis"
    STALE_PARENT_OR_AXIS = "stale_parent_or_axis"
    DUPLICATE_PHYSICAL_ACTION = "duplicate_physical_action"
    INCOMPLETE_ALIAS_FANOUT = "incomplete_alias_fanout"
    POPULATION_MISMATCH = "population_mismatch"
    STREAM_CAUSALITY_MISMATCH = "stream_causality_mismatch"
    ALPHA_LEDGER_MISMATCH = "alpha_ledger_mismatch"
    TRUE_Q_D_WITNESS_MISMATCH = "true_q_d_witness_mismatch"
    Q_ONLY_D_WITNESS_RETIRED = "q_only_d_witness_retired"
    D_STABILITY_WITNESS_MISMATCH = "D_stability_witness_mismatch"
    POST_OUTCOME = "post_outcome"
    AUTHORITY_VIOLATION = "authority_violation"


class V1SignedWorkPhysicalActionCAxisRefusal(RuntimeError):
    """Typed fail-closed refusal at the physical-action C-axis seam."""

    def __init__(
        self,
        code: V1SignedWorkPhysicalActionCAxisRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkPhysicalActionCAxisRefusalCode:
            raise TypeError("physical-action C-axis refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkPhysicalActionCAxisRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkPhysicalActionCAxisRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _dedup_identity(
    value: object,
    *,
    name: str,
) -> V1SignedWorkPackedActionDedupIdentity:
    if type(value) is not tuple or len(value) != 5:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            f"{name} must be the exact canonical five-part tuple",
        )
    head = value[:4]
    if any(type(item) is not str or not item for item in head):
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            f"{name} hash coordinates must be nonempty strings",
        )
    count = _positive_int(value[4], name=f"{name} removed count")
    return (head[0], head[1], head[2], head[3], count)


def _closed_authority(values: tuple[object, ...], *, owner: str) -> None:
    if any(type(value) is not bool for value in values):
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} authority flags must be exact booleans",
        )
    if any(values):
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed the pre-C identity boundary",
        )


def _validate_population(
    population: CompleteImageNetPopulationPairReceipt,
) -> str:
    if type(population) is not CompleteImageNetPopulationPairReceipt:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            "physical-action C axis requires the exact complete ImageNet receipt",
        )
    # ``content_hash`` replays the complete canonical population coordinates.
    return population.content_hash


def _validate_ledger(ledger: V1ProposalOrderAlphaLedger) -> str:
    if type(ledger) is not V1ProposalOrderAlphaLedger:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            "physical-action C axis requires the exact non-recycling ledger head",
        )
    try:
        ledger_hash = ledger.assert_integrity()
    except (TypeError, ValueError, RuntimeError) as error:
        raise V1SignedWorkPhysicalActionCAxisRefusal(
            V1SignedWorkPhysicalActionCAxisRefusalCode.ALPHA_LEDGER_MISMATCH,
            "alpha-ledger head failed exact integrity replay",
        ) from error
    return ledger_hash


def _downstream_authority_values(owner: object) -> tuple[object, ...]:
    names = (
        "C_outcomes_seen",
        "c_outcomes_seen",
        "validation_data_seen",
        "C_measurement_authority",
        "c_measurement_authority",
        "c_stream_authority",
        "nonlinear_remainder_authority",
        "calibration_authority",
        "oracle_authority",
        "mask_commit_authority",
        "terminal_authority",
        "pareto_authority",
    )
    return tuple(getattr(owner, name) for name in names if hasattr(owner, name))


def _validate_pre_c_bundle(bundle: V1SignedWorkPreCExecutionBundle) -> tuple[str, str]:
    if type(bundle) is not V1SignedWorkPreCExecutionBundle:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            "physical-action C axis requires an exact pre-C execution bundle",
        )
    family = bundle.packed_action_family
    receipt = bundle.receipt_family
    # Classify the three C-admission failures before replaying the family.
    # This also makes fail-closed behavior stable under adversarial mutation of
    # an otherwise frozen upstream object.
    if family.q_ambiguous_crosswalk_hashes:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.AMBIGUOUS_HARD_FQ,
            "an unresolved hard-F/Q alias cannot enter the physical C axis",
        )
    if not family.canonical_actions:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.EMPTY_ACTION_AXIS,
            "pre-C execution contains no Q-safe physical action",
        )
    seen_identities: set[V1SignedWorkPackedActionDedupIdentity] = set()
    packed_by_sha256: dict[str, bytes] = {}
    for value in family.canonical_actions:
        if type(value) is not V1SignedWorkCanonicalPackedPhysicalAction:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
                "canonical physical-action family changed type",
            )
        identity = v1_signed_work_packed_action_dedup_identity(value.packed_action)
        payload = value.packed_action.resulting_mask.packed_bits
        prior_payload = packed_by_sha256.get(identity[0])
        if identity in seen_identities or prior_payload == payload:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.DUPLICATE_PHYSICAL_ACTION,
                "canonical family repeats one physical action or exact payload",
            )
        if prior_payload is not None:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "matching packed-byte digests carried unequal actual bytes",
            )
        seen_identities.add(identity)
        packed_by_sha256[identity[0]] = payload
    try:
        receipt.__post_init__()
        family.__post_init__()
        bundle.__post_init__()
    except V1SignedWorkPhysicalActionCAxisRefusal:
        raise
    except (TypeError, ValueError, RuntimeError, PermissionError) as error:
        raise V1SignedWorkPhysicalActionCAxisRefusal(
            V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
            "pre-C execution bundle failed exact integrity replay",
        ) from error
    owners: tuple[object, ...] = (
        bundle,
        receipt,
        family,
        *family.candidate_crosswalks,
        *family.canonical_actions,
    )
    for owner in owners:
        _closed_authority(
            _downstream_authority_values(owner),
            owner=type(owner).__name__,
        )
    return bundle.bundle_hash, family.family_hash


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhysicalActionDFreezeIdentity:
    """Exact D/pre-C lineage owner consumed by the physical C-axis freeze."""

    pre_c_execution_bundle: V1SignedWorkPreCExecutionBundle = field(
        repr=False,
        compare=False,
    )
    complete_population: CompleteImageNetPopulationPairReceipt = field(
        repr=False,
        compare=False,
    )
    alpha_ledger_head: V1ProposalOrderAlphaLedger = field(
        repr=False,
        compare=False,
    )
    discovery_stream_commitment_hash: str
    discovery_sampling_order_hash: str
    pre_c_execution_bundle_hash: str
    packed_action_family_hash: str
    source_registration_hash: str
    registered_d_coordinate_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    complete_population_hash: str
    alpha_ledger_head_hash: str
    alpha_ledger_generation: int
    _issuer_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    frozen_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY
    schema: str = SIGNED_WORK_PHYSICAL_ACTION_D_FREEZE_SCHEMA
    _identity_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuer_token is not _D_FREEZE_TOKEN:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.AUTHORITY_VIOLATION,
                "D-freeze identities are issued only by the typed freeze factory",
            )
        bundle_hash, family_hash = _validate_pre_c_bundle(self.pre_c_execution_bundle)
        population_hash = _validate_population(self.complete_population)
        ledger_hash = _validate_ledger(self.alpha_ledger_head)
        for name in (
            "discovery_stream_commitment_hash",
            "discovery_sampling_order_hash",
            "pre_c_execution_bundle_hash",
            "packed_action_family_hash",
            "source_registration_hash",
            "registered_d_coordinate_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "complete_population_hash",
            "alpha_ledger_head_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        generation = _nonnegative_int(
            self.alpha_ledger_generation,
            name="alpha-ledger generation",
        )
        receipt = self.pre_c_execution_bundle.receipt_family
        if (
            self.pre_c_execution_bundle_hash != bundle_hash
            or self.packed_action_family_hash != family_hash
            or self.source_registration_hash != receipt.source_registration_hash
            or self.registered_d_coordinate_hash != receipt.registered_d_coordinate_hash
            or self.parent_mask_hash != receipt.parent_mask_hash
            or self.route_axis_hash != receipt.route_axis_hash
            or routes != receipt.route_count
            or self.complete_population_hash != population_hash
            or self.alpha_ledger_head_hash != ledger_hash
            or generation != self.alpha_ledger_head.proposal_generation
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.STALE_PARENT_OR_AXIS,
                "D-freeze identity does not replay its pre-C bundle, population, or ledger",
            )
        if (
            type(self.frozen_before_C_outcomes) is not bool
            or not self.frozen_before_C_outcomes
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.POST_OUTCOME,
                "D-freeze identity was not closed before C outcomes",
            )
        _closed_authority(
            (
                self.C_outcomes_seen,
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical-action D freeze",
        )
        if (
            self.authority_scope != SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY
            or self.schema != SIGNED_WORK_PHYSICAL_ACTION_D_FREEZE_SCHEMA
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.AUTHORITY_VIOLATION,
                "physical-action D-freeze schema or authority changed",
            )
        object.__setattr__(self, "route_count", receipt.route_count)
        object.__setattr__(self, "alpha_ledger_generation", generation)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "discovery_stream_commitment_hash": self.discovery_stream_commitment_hash,
            "discovery_sampling_order_hash": self.discovery_sampling_order_hash,
            "pre_c_execution_bundle_hash": self.pre_c_execution_bundle_hash,
            "packed_action_family_hash": self.packed_action_family_hash,
            "source_registration_hash": self.source_registration_hash,
            "registered_d_coordinate_hash": self.registered_d_coordinate_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "complete_population_hash": self.complete_population_hash,
            "alpha_ledger_head_hash": self.alpha_ledger_head_hash,
            "alpha_ledger_generation": self.alpha_ledger_generation,
            "frozen_before_C_outcomes": self.frozen_before_C_outcomes,
            "C_outcomes_seen": self.C_outcomes_seen,
            "C_measurement_authority": self.C_measurement_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def identity_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        bundle_hash, family_hash = _validate_pre_c_bundle(self.pre_c_execution_bundle)
        population_hash = _validate_population(self.complete_population)
        ledger_hash = _validate_ledger(self.alpha_ledger_head)
        if (
            bundle_hash != self.pre_c_execution_bundle_hash
            or family_hash != self.packed_action_family_hash
            or population_hash != self.complete_population_hash
            or ledger_hash != self.alpha_ledger_head_hash
            or self.alpha_ledger_head.proposal_generation
            != self.alpha_ledger_generation
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical-action D-freeze source changed after issuance",
            )
        observed = content_hash(self._payload(), prefix=_D_FREEZE_PREFIX)
        cached = self._identity_hash
        if cached is not None and cached != observed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical-action D-freeze identity was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical-action D-freeze hash does not match",
            )
        object.__setattr__(self, "_identity_hash", observed)
        return observed


def freeze_v1_signed_work_physical_action_d_identity(
    *,
    pre_c_execution_bundle: V1SignedWorkPreCExecutionBundle,
    complete_population: CompleteImageNetPopulationPairReceipt,
    alpha_ledger_head: V1ProposalOrderAlphaLedger,
    discovery_stream_commitment_hash: str,
    discovery_sampling_order_hash: str,
) -> V1SignedWorkPhysicalActionDFreezeIdentity:
    """Freeze exact D/pre-C lineage before any independent C observation."""

    bundle_hash, family_hash = _validate_pre_c_bundle(pre_c_execution_bundle)
    population_hash = _validate_population(complete_population)
    ledger_hash = _validate_ledger(alpha_ledger_head)
    receipt = pre_c_execution_bundle.receipt_family
    return V1SignedWorkPhysicalActionDFreezeIdentity(
        _issuer_token=_D_FREEZE_TOKEN,
        pre_c_execution_bundle=pre_c_execution_bundle,
        complete_population=complete_population,
        alpha_ledger_head=alpha_ledger_head,
        discovery_stream_commitment_hash=_identity(
            discovery_stream_commitment_hash,
            name="discovery stream commitment hash",
        ),
        discovery_sampling_order_hash=_identity(
            discovery_sampling_order_hash,
            name="discovery sampling-order hash",
        ),
        pre_c_execution_bundle_hash=bundle_hash,
        packed_action_family_hash=family_hash,
        source_registration_hash=receipt.source_registration_hash,
        registered_d_coordinate_hash=receipt.registered_d_coordinate_hash,
        parent_mask_hash=receipt.parent_mask_hash,
        route_axis_hash=receipt.route_axis_hash,
        route_count=receipt.route_count,
        complete_population_hash=population_hash,
        alpha_ledger_head_hash=ledger_hash,
        alpha_ledger_generation=alpha_ledger_head.proposal_generation,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhysicalActionCStreamIdentity:
    """Independent complete-train C stream frozen after D and before outcomes."""

    d_freeze: V1SignedWorkPhysicalActionDFreezeIdentity = field(
        repr=False,
        compare=False,
    )
    complete_population: CompleteImageNetPopulationPairReceipt = field(
        repr=False,
        compare=False,
    )
    certification_stream_commitment_hash: str
    certification_sampling_order_hash: str
    D_C_domain_separation_receipt_hash: str
    causal_parent_d_freeze_hash: str
    complete_population_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    alpha_ledger_head_hash: str
    alpha_ledger_generation: int
    _issuer_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    stream_role: str = "certification"
    complete_train_population_permuted: bool = True
    sampling_order_outcome_independent: bool = True
    opened_after_D_freeze: bool = True
    C_outcomes_seen: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY
    schema: str = SIGNED_WORK_PHYSICAL_ACTION_C_STREAM_SCHEMA
    _identity_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuer_token is not _C_STREAM_TOKEN:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.AUTHORITY_VIOLATION,
                "C-stream identities are issued only by the typed freeze factory",
            )
        if type(self.d_freeze) is not V1SignedWorkPhysicalActionDFreezeIdentity:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
                "C stream requires the exact physical-action D freeze",
            )
        d_hash = self.d_freeze.assert_integrity()
        population_hash = _validate_population(self.complete_population)
        for name in (
            "certification_stream_commitment_hash",
            "certification_sampling_order_hash",
            "D_C_domain_separation_receipt_hash",
            "causal_parent_d_freeze_hash",
            "complete_population_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "alpha_ledger_head_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        generation = _nonnegative_int(
            self.alpha_ledger_generation,
            name="alpha-ledger generation",
        )
        d = self.d_freeze
        if (
            self.causal_parent_d_freeze_hash != d_hash
            or self.complete_population_hash != population_hash
            or self.complete_population_hash != d.complete_population_hash
            or self.parent_mask_hash != d.parent_mask_hash
            or self.route_axis_hash != d.route_axis_hash
            or routes != d.route_count
            or self.alpha_ledger_head_hash != d.alpha_ledger_head_hash
            or generation != d.alpha_ledger_generation
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.STREAM_CAUSALITY_MISMATCH,
                "C stream does not descend from the exact D freeze and population",
            )
        if (
            self.stream_role != "certification"
            or type(self.complete_train_population_permuted) is not bool
            or not self.complete_train_population_permuted
            or type(self.sampling_order_outcome_independent) is not bool
            or not self.sampling_order_outcome_independent
            or type(self.opened_after_D_freeze) is not bool
            or not self.opened_after_D_freeze
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.STREAM_CAUSALITY_MISMATCH,
                "C stream weakened its complete-population pre-outcome sampling law",
            )
        _closed_authority(
            (
                self.C_outcomes_seen,
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical-action C stream",
        )
        if (
            self.authority_scope != SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY
            or self.schema != SIGNED_WORK_PHYSICAL_ACTION_C_STREAM_SCHEMA
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.AUTHORITY_VIOLATION,
                "physical-action C-stream schema or authority changed",
            )
        object.__setattr__(self, "route_count", d.route_count)
        object.__setattr__(self, "alpha_ledger_generation", generation)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "certification_stream_commitment_hash": (
                self.certification_stream_commitment_hash
            ),
            "certification_sampling_order_hash": self.certification_sampling_order_hash,
            "D_C_domain_separation_receipt_hash": (
                self.D_C_domain_separation_receipt_hash
            ),
            "causal_parent_d_freeze_hash": self.causal_parent_d_freeze_hash,
            "complete_population_hash": self.complete_population_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "alpha_ledger_head_hash": self.alpha_ledger_head_hash,
            "alpha_ledger_generation": self.alpha_ledger_generation,
            "stream_role": self.stream_role,
            "complete_train_population_permuted": (
                self.complete_train_population_permuted
            ),
            "sampling_order_outcome_independent": self.sampling_order_outcome_independent,
            "opened_after_D_freeze": self.opened_after_D_freeze,
            "C_outcomes_seen": self.C_outcomes_seen,
            "C_measurement_authority": self.C_measurement_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def identity_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        d_hash = self.d_freeze.assert_integrity()
        population_hash = _validate_population(self.complete_population)
        if (
            d_hash != self.causal_parent_d_freeze_hash
            or population_hash != self.complete_population_hash
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical-action C-stream source changed after issuance",
            )
        observed = content_hash(self._payload(), prefix=_C_STREAM_PREFIX)
        cached = self._identity_hash
        if cached is not None and cached != observed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical-action C-stream identity was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical-action C-stream hash does not match",
            )
        object.__setattr__(self, "_identity_hash", observed)
        return observed


def freeze_v1_signed_work_physical_action_c_stream_identity(
    *,
    d_freeze: V1SignedWorkPhysicalActionDFreezeIdentity,
    complete_population: CompleteImageNetPopulationPairReceipt,
    certification_stream_commitment_hash: str,
    certification_sampling_order_hash: str,
    D_C_domain_separation_receipt_hash: str,
) -> V1SignedWorkPhysicalActionCStreamIdentity:
    """Freeze one independent C permutation before observing any C row."""

    if type(d_freeze) is not V1SignedWorkPhysicalActionDFreezeIdentity:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            "C-stream freeze requires the exact D-freeze owner",
        )
    d_hash = d_freeze.assert_integrity()
    population_hash = _validate_population(complete_population)
    if population_hash != d_freeze.complete_population_hash:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.POPULATION_MISMATCH,
            "D and C do not name the same complete ImageNet population",
        )
    return V1SignedWorkPhysicalActionCStreamIdentity(
        _issuer_token=_C_STREAM_TOKEN,
        d_freeze=d_freeze,
        complete_population=complete_population,
        certification_stream_commitment_hash=_identity(
            certification_stream_commitment_hash,
            name="certification stream commitment hash",
        ),
        certification_sampling_order_hash=_identity(
            certification_sampling_order_hash,
            name="certification sampling-order hash",
        ),
        D_C_domain_separation_receipt_hash=_identity(
            D_C_domain_separation_receipt_hash,
            name="D/C domain-separation receipt hash",
        ),
        causal_parent_d_freeze_hash=d_hash,
        complete_population_hash=population_hash,
        parent_mask_hash=d_freeze.parent_mask_hash,
        route_axis_hash=d_freeze.route_axis_hash,
        route_count=d_freeze.route_count,
        alpha_ledger_head_hash=d_freeze.alpha_ledger_head_hash,
        alpha_ledger_generation=d_freeze.alpha_ledger_generation,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkTrueQPhysicalActionCAdmission:
    """True-Q D-stable precondition for opening physical-action C.

    This is stricter than ordinary packed-action custody.  It binds the exact
    physical-action D freeze to a compact true-Q D decision witness and verifies
    that the witness-selected packed child is one canonical Q-safe physical
    action in the pre-C bundle.  It is still pre-outcome identity evidence: C
    may be opened by a separately registered stream freeze, but this receipt
    itself observes no C rows and grants no K/O/commit/terminal/Pareto authority.
    """

    d_freeze: V1SignedWorkPhysicalActionDFreezeIdentity = field(
        repr=False,
        compare=False,
    )
    true_q_d_decision_witness: V1PhaseATrueQDDecisionWitnessReceipt = field(
        repr=False,
        compare=False,
    )
    d_freeze_hash: str
    true_q_d_decision_witness_hash: str
    pre_c_execution_bundle_hash: str
    packed_action_family_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    selected_physical_action_identity_hash: str
    selected_packed_dedup_identity: V1SignedWorkPackedActionDedupIdentity
    selected_packed_action_hash: str
    selected_resulting_mask_hash: str
    matched_canonical_action_hash: str
    matched_physical_action_ordinal: int
    selected_total_count: int
    selected_outside_count: int
    selected_shell_count: int
    dual_gap_upper: float
    strict_outside_bit_margin_lower: float
    _issuer_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    true_q_d_stability_witness: bool = True
    selected_action_is_canonical_q_safe_physical_action: bool = True
    frozen_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY
    schema: str = SIGNED_WORK_TRUE_Q_PHYSICAL_ACTION_C_ADMISSION_SCHEMA
    _admission_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuer_token is not _TRUE_Q_C_ADMISSION_TOKEN:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.AUTHORITY_VIOLATION,
                "true-Q C-admission receipts are issued only by the typed factory",
            )
        if type(self.d_freeze) is not V1SignedWorkPhysicalActionDFreezeIdentity:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
                "true-Q C admission requires the exact physical-action D freeze",
            )
        if (
            type(self.true_q_d_decision_witness)
            is not V1PhaseATrueQDDecisionWitnessReceipt
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
                "true-Q C admission requires the exact true-Q D witness",
            )
        d_hash = self.d_freeze.assert_integrity()
        witness = self.true_q_d_decision_witness
        witness_hash = witness.assert_integrity()
        bundle = self.d_freeze.pre_c_execution_bundle
        bundle_hash, family_hash = _validate_pre_c_bundle(bundle)
        matching = tuple(
            action
            for action in bundle.packed_action_family.canonical_actions
            if (
                action.packed_action.action_hash == witness.selected_packed_action_hash
                and action.packed_action.resulting_mask.mask_hash
                == witness.selected_resulting_mask_hash
            )
        )
        if len(matching) != 1:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.TRUE_Q_D_WITNESS_MISMATCH,
                "true-Q D witness does not select exactly one canonical packed action",
            )
        canonical = matching[0]
        for name in (
            "d_freeze_hash",
            "true_q_d_decision_witness_hash",
            "pre_c_execution_bundle_hash",
            "packed_action_family_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "selected_physical_action_identity_hash",
            "selected_packed_action_hash",
            "selected_resulting_mask_hash",
            "matched_canonical_action_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        ordinal = _positive_int(
            self.matched_physical_action_ordinal,
            name="matched physical-action ordinal",
        )
        total = _positive_int(self.selected_total_count, name="selected total count")
        outside = _nonnegative_int(
            self.selected_outside_count,
            name="selected outside count",
        )
        shell = _nonnegative_int(
            self.selected_shell_count,
            name="selected shell count",
        )
        dedup = _dedup_identity(
            self.selected_packed_dedup_identity,
            name="selected packed dedup identity",
        )
        canonical_dedup = v1_signed_work_packed_action_dedup_identity(
            canonical.packed_action
        )
        if (
            self.d_freeze_hash != d_hash
            or self.true_q_d_decision_witness_hash != witness_hash
            or self.pre_c_execution_bundle_hash != bundle_hash
            or self.packed_action_family_hash != family_hash
            or self.parent_mask_hash != self.d_freeze.parent_mask_hash
            or self.parent_mask_hash != witness.parent_mask_hash
            or self.route_axis_hash != self.d_freeze.route_axis_hash
            or self.route_axis_hash != witness.route_axis_hash
            or routes != self.d_freeze.route_count
            or routes != witness.route_count
            or self.selected_physical_action_identity_hash
            != witness.selected_physical_action_identity_hash
            or dedup != canonical_dedup
            or dedup[1] != witness.selected_packed_action_hash
            or dedup[2] != witness.selected_resulting_mask_hash
            or self.selected_packed_action_hash != witness.selected_packed_action_hash
            or self.selected_resulting_mask_hash != witness.selected_resulting_mask_hash
            or self.matched_canonical_action_hash != canonical.canonical_action_hash
            or ordinal != canonical.canonical_ordinal
            or total != witness.selected_total_count
            or outside != witness.selected_outside_count
            or shell != witness.selected_shell_count
            or total != outside + shell
            or float(self.dual_gap_upper) != witness.dual_gap_upper
            or float(self.strict_outside_bit_margin_lower)
            != witness.strict_outside_bit_margin_lower
            or not witness.dual_gap_upper < witness.strict_outside_bit_margin_lower
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.TRUE_Q_D_WITNESS_MISMATCH,
                "true-Q C admission source graph does not replay one D-stable action",
            )
        if (
            type(self.true_q_d_stability_witness) is not bool
            or self.true_q_d_stability_witness is not True
            or type(self.selected_action_is_canonical_q_safe_physical_action)
            is not bool
            or self.selected_action_is_canonical_q_safe_physical_action is not True
            or type(self.frozen_before_C_outcomes) is not bool
            or self.frozen_before_C_outcomes is not True
            or self.authority_scope != SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY
            or self.schema != SIGNED_WORK_TRUE_Q_PHYSICAL_ACTION_C_ADMISSION_SCHEMA
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.AUTHORITY_VIOLATION,
                "true-Q C admission weakened its pre-outcome semantics",
            )
        _closed_authority(
            (
                self.C_outcomes_seen,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="true-Q physical-action C admission",
        )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "matched_physical_action_ordinal", ordinal)
        object.__setattr__(self, "selected_total_count", total)
        object.__setattr__(self, "selected_outside_count", outside)
        object.__setattr__(self, "selected_shell_count", shell)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "d_freeze_hash": self.d_freeze_hash,
            "true_q_d_decision_witness_hash": self.true_q_d_decision_witness_hash,
            "pre_c_execution_bundle_hash": self.pre_c_execution_bundle_hash,
            "packed_action_family_hash": self.packed_action_family_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "selected_physical_action_identity_hash": (
                self.selected_physical_action_identity_hash
            ),
            "selected_packed_dedup_identity": self.selected_packed_dedup_identity,
            "selected_packed_action_hash": self.selected_packed_action_hash,
            "selected_resulting_mask_hash": self.selected_resulting_mask_hash,
            "matched_canonical_action_hash": self.matched_canonical_action_hash,
            "matched_physical_action_ordinal": self.matched_physical_action_ordinal,
            "selected_total_count": self.selected_total_count,
            "selected_outside_count": self.selected_outside_count,
            "selected_shell_count": self.selected_shell_count,
            "dual_gap_upper": self.dual_gap_upper,
            "strict_outside_bit_margin_lower": self.strict_outside_bit_margin_lower,
            "true_q_d_stability_witness": self.true_q_d_stability_witness,
            "selected_action_is_canonical_q_safe_physical_action": (
                self.selected_action_is_canonical_q_safe_physical_action
            ),
            "frozen_before_C_outcomes": self.frozen_before_C_outcomes,
            "C_outcomes_seen": self.C_outcomes_seen,
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def admission_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        d_hash = self.d_freeze.assert_integrity()
        witness_hash = self.true_q_d_decision_witness.assert_integrity()
        if (
            d_hash != self.d_freeze_hash
            or witness_hash != self.true_q_d_decision_witness_hash
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "true-Q physical-action C admission source changed after issuance",
            )
        observed = content_hash(self._payload(), prefix=_TRUE_Q_C_ADMISSION_PREFIX)
        cached = self._admission_hash
        if cached is not None and cached != observed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "true-Q physical-action C admission was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "true-Q physical-action C admission hash does not match",
            )
        object.__setattr__(self, "_admission_hash", observed)
        return observed


def issue_v1_signed_work_true_q_physical_action_c_admission(
    *,
    d_freeze: V1SignedWorkPhysicalActionDFreezeIdentity,
    true_q_d_decision_witness: V1PhaseATrueQDDecisionWitnessReceipt,
) -> V1SignedWorkTrueQPhysicalActionCAdmission:
    """Refuse the superseded Q-only D-to-C admission path.

    The legacy receipt type remains import-readable for provenance, but its
    factory is permanently closed.  A Q component cannot substitute for the
    final corrected-F/true-Q/nonrecycling-A/F/J D decision.
    """

    del d_freeze, true_q_d_decision_witness
    _refuse(
        V1SignedWorkPhysicalActionCAxisRefusalCode.Q_ONLY_D_WITNESS_RETIRED,
        "Q-only D evidence cannot open physical C; provide the joined "
        "corrected-F/true-Q/nonrecycling-A/F/J D-stability receipt",
    )

    if type(d_freeze) is not V1SignedWorkPhysicalActionDFreezeIdentity:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            "true-Q physical-action C admission requires exact D freeze",
        )
    if type(true_q_d_decision_witness) is not V1PhaseATrueQDDecisionWitnessReceipt:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            "true-Q physical-action C admission requires exact true-Q witness",
        )
    d_hash = d_freeze.assert_integrity()
    witness_hash = true_q_d_decision_witness.assert_integrity()
    bundle_hash, family_hash = _validate_pre_c_bundle(d_freeze.pre_c_execution_bundle)
    family = d_freeze.pre_c_execution_bundle.packed_action_family
    matching = tuple(
        action
        for action in family.canonical_actions
        if (
            action.packed_action.action_hash
            == true_q_d_decision_witness.selected_packed_action_hash
            and action.packed_action.resulting_mask.mask_hash
            == true_q_d_decision_witness.selected_resulting_mask_hash
        )
    )
    if len(matching) != 1:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.TRUE_Q_D_WITNESS_MISMATCH,
            "true-Q D witness does not name one canonical Q-safe packed action",
        )
    canonical = matching[0]
    return V1SignedWorkTrueQPhysicalActionCAdmission(
        _issuer_token=_TRUE_Q_C_ADMISSION_TOKEN,
        d_freeze=d_freeze,
        true_q_d_decision_witness=true_q_d_decision_witness,
        d_freeze_hash=d_hash,
        true_q_d_decision_witness_hash=witness_hash,
        pre_c_execution_bundle_hash=bundle_hash,
        packed_action_family_hash=family_hash,
        parent_mask_hash=d_freeze.parent_mask_hash,
        route_axis_hash=d_freeze.route_axis_hash,
        route_count=d_freeze.route_count,
        selected_physical_action_identity_hash=(
            true_q_d_decision_witness.selected_physical_action_identity_hash
        ),
        selected_packed_dedup_identity=(
            v1_signed_work_packed_action_dedup_identity(canonical.packed_action)
        ),
        selected_packed_action_hash=true_q_d_decision_witness.selected_packed_action_hash,
        selected_resulting_mask_hash=(
            true_q_d_decision_witness.selected_resulting_mask_hash
        ),
        matched_canonical_action_hash=canonical.canonical_action_hash,
        matched_physical_action_ordinal=canonical.canonical_ordinal,
        selected_total_count=true_q_d_decision_witness.selected_total_count,
        selected_outside_count=true_q_d_decision_witness.selected_outside_count,
        selected_shell_count=true_q_d_decision_witness.selected_shell_count,
        dual_gap_upper=true_q_d_decision_witness.dual_gap_upper,
        strict_outside_bit_margin_lower=(
            true_q_d_decision_witness.strict_outside_bit_margin_lower
        ),
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkDStablePhysicalActionCAdmission:
    """Final D-stable precondition for opening physical-action C.

    This sibling to the legacy true-Q admission consumes the final sampled-v7
    D-stability receipt, not merely its Q component.  It binds the D freeze to
    one canonical physical packed action after true-Q, robust-shell,
    corrected signed-F, and non-recycling A/F/J evidence have all been joined.
    It is still pre-outcome identity evidence and grants no downstream
    authority.
    """

    d_freeze: V1SignedWorkPhysicalActionDFreezeIdentity = field(
        repr=False,
        compare=False,
    )
    d_decision_stability: V1SampledV7DDecisionStabilityReceipt = field(
        repr=False,
        compare=False,
    )
    d_freeze_hash: str
    d_decision_stability_hash: str
    true_q_d_decision_witness_hash: str
    alpha_observation_family_closure_hash: str
    pre_c_execution_bundle_hash: str
    packed_action_family_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    selected_physical_action_identity_hash: str
    selected_packed_dedup_identity: V1SignedWorkPackedActionDedupIdentity
    selected_packed_action_hash: str
    selected_resulting_mask_hash: str
    selected_resulting_mask_packed_sha256: str
    selected_removed_route_index_hash: str
    selected_removed_count: int
    matched_canonical_action_hash: str
    matched_physical_action_ordinal: int
    selected_total_count: int
    selected_outside_count: int
    selected_shell_count: int
    selected_branch_alias_hashes: tuple[str, ...]
    selected_signed_f_evidence_receipt_hashes: tuple[str, ...]
    q_dual_gap_upper: float
    q_strict_outside_bit_margin_lower: float
    _issuer_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    d_decision_stability_witness: bool = True
    selected_action_is_canonical_d_stable_physical_action: bool = True
    frozen_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY
    schema: str = SIGNED_WORK_D_STABLE_PHYSICAL_ACTION_C_ADMISSION_SCHEMA
    _admission_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuer_token is not _D_STABLE_C_ADMISSION_TOKEN:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.AUTHORITY_VIOLATION,
                "D-stable C-admission receipts are issued only by the typed factory",
            )
        if type(self.d_freeze) is not V1SignedWorkPhysicalActionDFreezeIdentity:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
                "D-stable C admission requires the exact physical-action D freeze",
            )
        if type(self.d_decision_stability) is not V1SampledV7DDecisionStabilityReceipt:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
                "D-stable C admission requires the exact sampled-v7 D-stability receipt",
            )
        d_hash = self.d_freeze.assert_integrity()
        stability = self.d_decision_stability
        stability_hash = stability.assert_integrity()
        selected_action = stability.selected_packed_action
        bundle_hash, family_hash = _validate_pre_c_bundle(
            self.d_freeze.pre_c_execution_bundle
        )
        family = self.d_freeze.pre_c_execution_bundle.packed_action_family
        matching = tuple(
            action
            for action in family.canonical_actions
            if (
                action.packed_action.action_hash == stability.selected_packed_action_hash
                and action.packed_action.resulting_mask.mask_hash
                == stability.selected_resulting_mask_hash
                and v1_signed_work_packed_action_dedup_identity(action.packed_action)
                == stability.selected_packed_dedup_identity
                and action.packed_action.resulting_mask.packed_bits
                == selected_action.resulting_mask.packed_bits
            )
        )
        if len(matching) != 1:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.D_STABILITY_WITNESS_MISMATCH,
                "D-stability receipt does not select exactly one canonical packed action",
            )
        canonical = matching[0]
        for name in (
            "d_freeze_hash",
            "d_decision_stability_hash",
            "true_q_d_decision_witness_hash",
            "alpha_observation_family_closure_hash",
            "pre_c_execution_bundle_hash",
            "packed_action_family_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "selected_physical_action_identity_hash",
            "selected_packed_action_hash",
            "selected_resulting_mask_hash",
            "selected_resulting_mask_packed_sha256",
            "selected_removed_route_index_hash",
            "matched_canonical_action_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        removed = _positive_int(
            self.selected_removed_count,
            name="selected removed count",
        )
        ordinal = _positive_int(
            self.matched_physical_action_ordinal,
            name="matched physical-action ordinal",
        )
        total = _positive_int(self.selected_total_count, name="selected total count")
        outside = _nonnegative_int(
            self.selected_outside_count,
            name="selected outside count",
        )
        shell = _nonnegative_int(
            self.selected_shell_count,
            name="selected shell count",
        )
        aliases = tuple(
            _identity(value, name="selected branch alias hash")
            for value in self.selected_branch_alias_hashes
        )
        evidence = tuple(
            _identity(value, name="selected signed-F evidence receipt hash")
            for value in self.selected_signed_f_evidence_receipt_hashes
        )
        dedup = _dedup_identity(
            self.selected_packed_dedup_identity,
            name="selected packed dedup identity",
        )
        if (
            self.d_freeze_hash != d_hash
            or self.d_decision_stability_hash != stability_hash
            or self.true_q_d_decision_witness_hash
            != stability.true_q_d_decision_witness_receipt_hash
            or self.alpha_observation_family_closure_hash
            != stability.alpha_observation_family_closure_receipt_hash
            or self.pre_c_execution_bundle_hash != bundle_hash
            or self.packed_action_family_hash != family_hash
            or self.parent_mask_hash != self.d_freeze.parent_mask_hash
            or self.parent_mask_hash != stability.parent_mask_hash
            or self.route_axis_hash != self.d_freeze.route_axis_hash
            or self.route_axis_hash != stability.route_axis_hash
            or routes != self.d_freeze.route_count
            or routes != stability.route_count
            or self.selected_physical_action_identity_hash
            != stability.selected_physical_action_identity_hash
            or dedup != stability.selected_packed_dedup_identity
            or self.selected_packed_action_hash != stability.selected_packed_action_hash
            or self.selected_resulting_mask_hash
            != stability.selected_resulting_mask_hash
            or self.selected_resulting_mask_packed_sha256
            != stability.selected_resulting_mask_packed_sha256
            or self.selected_removed_route_index_hash
            != stability.selected_removed_route_index_hash
            or removed != stability.selected_removed_count
            or self.matched_canonical_action_hash != canonical.canonical_action_hash
            or ordinal != canonical.canonical_ordinal
            or total != stability.selected_total_count
            or outside != stability.selected_outside_count
            or shell != stability.selected_shell_count
            or total != outside + shell
            or removed != total
            or aliases != stability.selected_branch_alias_hashes
            or evidence != stability.selected_signed_f_evidence_receipt_hashes
            or type(self.q_dual_gap_upper) is not float
            or type(self.q_strict_outside_bit_margin_lower) is not float
            or not math.isfinite(self.q_dual_gap_upper)
            or not math.isfinite(self.q_strict_outside_bit_margin_lower)
            or self.q_dual_gap_upper != stability.q_dual_gap_upper
            or self.q_strict_outside_bit_margin_lower
            != stability.q_strict_outside_bit_margin_lower
            or not stability.q_dual_gap_upper
            < stability.q_strict_outside_bit_margin_lower
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.D_STABILITY_WITNESS_MISMATCH,
                "D-stable C admission source graph does not replay one stable action",
            )
        if (
            stability.d_decision_stability_authority is not True
            or stability.q_component_certified is not True
            or stability.signed_f_component_certified is not True
            or stability.nonrecycling_a_f_j_family_closed is not True
            or stability.production_cuda_q_required is not True
            or stability.independent_C_still_required is not True
            or stability.C_stream_open_authority is not False
            or stability.calibration_K_open_authority is not False
            or stability.paired_oracle_O_open_authority is not False
            or stability.mask_commit_authority is not False
            or type(self.d_decision_stability_witness) is not bool
            or self.d_decision_stability_witness is not True
            or type(self.selected_action_is_canonical_d_stable_physical_action)
            is not bool
            or self.selected_action_is_canonical_d_stable_physical_action is not True
            or type(self.frozen_before_C_outcomes) is not bool
            or self.frozen_before_C_outcomes is not True
            or self.authority_scope != SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY
            or self.schema != SIGNED_WORK_D_STABLE_PHYSICAL_ACTION_C_ADMISSION_SCHEMA
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.AUTHORITY_VIOLATION,
                "D-stable C admission weakened its pre-outcome semantics",
            )
        _closed_authority(
            (
                self.C_outcomes_seen,
                self.C_stream_open_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="D-stable physical-action C admission",
        )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "selected_removed_count", removed)
        object.__setattr__(self, "matched_physical_action_ordinal", ordinal)
        object.__setattr__(self, "selected_total_count", total)
        object.__setattr__(self, "selected_outside_count", outside)
        object.__setattr__(self, "selected_shell_count", shell)
        object.__setattr__(self, "selected_branch_alias_hashes", aliases)
        object.__setattr__(self, "selected_signed_f_evidence_receipt_hashes", evidence)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "d_freeze_hash": self.d_freeze_hash,
            "d_decision_stability_hash": self.d_decision_stability_hash,
            "true_q_d_decision_witness_hash": self.true_q_d_decision_witness_hash,
            "alpha_observation_family_closure_hash": (
                self.alpha_observation_family_closure_hash
            ),
            "pre_c_execution_bundle_hash": self.pre_c_execution_bundle_hash,
            "packed_action_family_hash": self.packed_action_family_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "selected_physical_action_identity_hash": (
                self.selected_physical_action_identity_hash
            ),
            "selected_packed_dedup_identity": self.selected_packed_dedup_identity,
            "selected_packed_action_hash": self.selected_packed_action_hash,
            "selected_resulting_mask_hash": self.selected_resulting_mask_hash,
            "selected_resulting_mask_packed_sha256": (
                self.selected_resulting_mask_packed_sha256
            ),
            "selected_removed_route_index_hash": self.selected_removed_route_index_hash,
            "selected_removed_count": self.selected_removed_count,
            "matched_canonical_action_hash": self.matched_canonical_action_hash,
            "matched_physical_action_ordinal": self.matched_physical_action_ordinal,
            "selected_total_count": self.selected_total_count,
            "selected_outside_count": self.selected_outside_count,
            "selected_shell_count": self.selected_shell_count,
            "selected_branch_alias_hashes": self.selected_branch_alias_hashes,
            "selected_signed_f_evidence_receipt_hashes": (
                self.selected_signed_f_evidence_receipt_hashes
            ),
            "q_dual_gap_upper": self.q_dual_gap_upper,
            "q_strict_outside_bit_margin_lower": (
                self.q_strict_outside_bit_margin_lower
            ),
            "d_decision_stability_witness": self.d_decision_stability_witness,
            "selected_action_is_canonical_d_stable_physical_action": (
                self.selected_action_is_canonical_d_stable_physical_action
            ),
            "frozen_before_C_outcomes": self.frozen_before_C_outcomes,
            "C_outcomes_seen": self.C_outcomes_seen,
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def admission_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        d_hash = self.d_freeze.assert_integrity()
        stability_hash = self.d_decision_stability.assert_integrity()
        if (
            d_hash != self.d_freeze_hash
            or stability_hash != self.d_decision_stability_hash
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "D-stable physical-action C admission source changed after issuance",
            )
        observed = content_hash(self._payload(), prefix=_D_STABLE_C_ADMISSION_PREFIX)
        cached = self._admission_hash
        if cached is not None and cached != observed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "D-stable physical-action C admission was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "D-stable physical-action C admission hash does not match",
            )
        object.__setattr__(self, "_admission_hash", observed)
        return observed


def issue_v1_signed_work_d_stable_physical_action_c_admission(
    *,
    d_freeze: V1SignedWorkPhysicalActionDFreezeIdentity,
    d_decision_stability: V1SampledV7DDecisionStabilityReceipt,
) -> V1SignedWorkDStablePhysicalActionCAdmission:
    """Bind final D-stability to one packed physical C action."""

    if type(d_freeze) is not V1SignedWorkPhysicalActionDFreezeIdentity:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            "D-stable physical-action C admission requires exact D freeze",
        )
    if type(d_decision_stability) is not V1SampledV7DDecisionStabilityReceipt:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            "D-stable physical-action C admission requires exact D-stability receipt",
        )
    d_hash = d_freeze.assert_integrity()
    stability_hash = d_decision_stability.assert_integrity()
    selected_action = d_decision_stability.selected_packed_action
    bundle_hash, family_hash = _validate_pre_c_bundle(d_freeze.pre_c_execution_bundle)
    family = d_freeze.pre_c_execution_bundle.packed_action_family
    matching = tuple(
        action
        for action in family.canonical_actions
        if (
            action.packed_action.action_hash
            == d_decision_stability.selected_packed_action_hash
            and action.packed_action.resulting_mask.mask_hash
            == d_decision_stability.selected_resulting_mask_hash
            and v1_signed_work_packed_action_dedup_identity(action.packed_action)
            == d_decision_stability.selected_packed_dedup_identity
            and action.packed_action.resulting_mask.packed_bits
            == selected_action.resulting_mask.packed_bits
        )
    )
    if len(matching) != 1:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.D_STABILITY_WITNESS_MISMATCH,
            "D-stability receipt does not name one canonical stable packed action",
        )
    canonical = matching[0]
    return V1SignedWorkDStablePhysicalActionCAdmission(
        _issuer_token=_D_STABLE_C_ADMISSION_TOKEN,
        d_freeze=d_freeze,
        d_decision_stability=d_decision_stability,
        d_freeze_hash=d_hash,
        d_decision_stability_hash=stability_hash,
        true_q_d_decision_witness_hash=(
            d_decision_stability.true_q_d_decision_witness_receipt_hash
        ),
        alpha_observation_family_closure_hash=(
            d_decision_stability.alpha_observation_family_closure_receipt_hash
        ),
        pre_c_execution_bundle_hash=bundle_hash,
        packed_action_family_hash=family_hash,
        parent_mask_hash=d_freeze.parent_mask_hash,
        route_axis_hash=d_freeze.route_axis_hash,
        route_count=d_freeze.route_count,
        selected_physical_action_identity_hash=(
            d_decision_stability.selected_physical_action_identity_hash
        ),
        selected_packed_dedup_identity=(
            d_decision_stability.selected_packed_dedup_identity
        ),
        selected_packed_action_hash=d_decision_stability.selected_packed_action_hash,
        selected_resulting_mask_hash=(
            d_decision_stability.selected_resulting_mask_hash
        ),
        selected_resulting_mask_packed_sha256=(
            d_decision_stability.selected_resulting_mask_packed_sha256
        ),
        selected_removed_route_index_hash=(
            d_decision_stability.selected_removed_route_index_hash
        ),
        selected_removed_count=d_decision_stability.selected_removed_count,
        matched_canonical_action_hash=canonical.canonical_action_hash,
        matched_physical_action_ordinal=canonical.canonical_ordinal,
        selected_total_count=d_decision_stability.selected_total_count,
        selected_outside_count=d_decision_stability.selected_outside_count,
        selected_shell_count=d_decision_stability.selected_shell_count,
        selected_branch_alias_hashes=d_decision_stability.selected_branch_alias_hashes,
        selected_signed_f_evidence_receipt_hashes=(
            d_decision_stability.selected_signed_f_evidence_receipt_hashes
        ),
        q_dual_gap_upper=d_decision_stability.q_dual_gap_upper,
        q_strict_outside_bit_margin_lower=(
            d_decision_stability.q_strict_outside_bit_margin_lower
        ),
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhysicalActionCAxisEntry:
    """One unique physical action measured once on the C action axis."""

    physical_action_ordinal: int
    canonical_action_hash: str
    dedup_identity: V1SignedWorkPackedActionDedupIdentity
    packed_action: V1PackedRemovalAction = field(repr=False, compare=False)
    actual_packed_bytes: bytes = field(repr=False, compare=False)
    actual_packed_sha256: str
    actual_packed_byte_count: int
    alias_fanout_receipt_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    C_outcomes_seen: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY
    schema: str = SIGNED_WORK_PHYSICAL_ACTION_C_ENTRY_SCHEMA
    _entry_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        ordinal = _positive_int(
            self.physical_action_ordinal,
            name="physical-action ordinal",
        )
        if type(self.packed_action) is not V1PackedRemovalAction:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
                "physical C-axis entry requires an exact packed action",
            )
        actual = self.actual_packed_bytes
        if type(actual) is not bytes:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
                "physical C-axis entry requires canonical immutable packed bytes",
            )
        expected_identity = v1_signed_work_packed_action_dedup_identity(
            self.packed_action
        )
        identity = _dedup_identity(self.dedup_identity, name="entry dedup identity")
        if identity != expected_identity:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical C-axis entry changed the canonical five-part identity",
            )
        if actual != self.packed_action.resulting_mask.packed_bits:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical C-axis entry bytes differ from its packed action",
            )
        digest = hashlib.sha256(actual).hexdigest()
        packed_byte_count = _positive_int(
            self.actual_packed_byte_count,
            name="actual packed byte count",
        )
        if (
            self.actual_packed_sha256 != digest
            or packed_byte_count != len(actual)
            or expected_identity[0] != digest
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical C-axis entry digest or byte count does not replay",
            )
        for name in (
            "canonical_action_hash",
            "actual_packed_sha256",
            "alias_fanout_receipt_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        mask = self.packed_action.resulting_mask
        if mask.route_axis_hash != self.route_axis_hash or mask.route_count != routes:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.STALE_PARENT_OR_AXIS,
                "physical C-axis entry changed the route axis",
            )
        _closed_authority(
            (
                self.C_outcomes_seen,
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical C-axis entry",
        )
        if (
            self.authority_scope != SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY
            or self.schema != SIGNED_WORK_PHYSICAL_ACTION_C_ENTRY_SCHEMA
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.AUTHORITY_VIOLATION,
                "physical C-axis entry schema or authority changed",
            )
        object.__setattr__(self, "physical_action_ordinal", ordinal)
        object.__setattr__(self, "dedup_identity", expected_identity)
        object.__setattr__(self, "actual_packed_byte_count", len(actual))
        object.__setattr__(self, "route_count", routes)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "physical_action_ordinal": self.physical_action_ordinal,
            "canonical_action_hash": self.canonical_action_hash,
            "dedup_identity": self.dedup_identity,
            "actual_packed_bytes": self.actual_packed_bytes,
            "actual_packed_sha256": self.actual_packed_sha256,
            "actual_packed_byte_count": self.actual_packed_byte_count,
            "alias_fanout_receipt_hash": self.alias_fanout_receipt_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "C_outcomes_seen": self.C_outcomes_seen,
            "C_measurement_authority": self.C_measurement_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def entry_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.actual_packed_bytes != self.packed_action.resulting_mask.packed_bits
            or tuple(self.dedup_identity)
            != v1_signed_work_packed_action_dedup_identity(self.packed_action)
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical C-axis entry payload changed after issuance",
            )
        observed = content_hash(self._payload(), prefix=_ENTRY_PREFIX)
        cached = self._entry_hash
        if cached is not None and cached != observed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical C-axis entry was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical C-axis entry hash does not match",
            )
        object.__setattr__(self, "_entry_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhysicalActionAliasFanoutReceipt:
    """Complete upstream alias fanout for one unique C-axis action."""

    physical_action_ordinal: int
    canonical_action_hash: str
    dedup_identity: V1SignedWorkPackedActionDedupIdentity
    source_aliases: tuple[V1SignedWorkCoordinateCandidateCrosswalk, ...] = field(
        repr=False,
        compare=False,
    )
    alias_crosswalk_hashes: tuple[str, ...]
    coordinate_candidate_key_hashes: tuple[str, ...]
    alias_count: int
    all_aliases_retained: bool = True
    C_outcomes_seen: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY
    schema: str = SIGNED_WORK_PHYSICAL_ACTION_ALIAS_FANOUT_SCHEMA
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        ordinal = _positive_int(
            self.physical_action_ordinal,
            name="fanout physical-action ordinal",
        )
        for name in ("canonical_action_hash", "authority_scope"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.source_aliases) is not tuple:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
                "alias fanout source must be one exact tuple",
            )
        aliases = self.source_aliases
        if not aliases or any(
            type(value) is not V1SignedWorkCoordinateCandidateCrosswalk
            for value in aliases
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
                "alias fanout requires exact nonempty coordinate crosswalks",
            )
        expected_hashes = tuple(value.crosswalk_hash for value in aliases)
        expected_keys = tuple(value.coordinate_candidate_key_hash for value in aliases)
        if (
            tuple(self.alias_crosswalk_hashes) != expected_hashes
            or tuple(self.coordinate_candidate_key_hashes) != expected_keys
            or len(set(expected_hashes)) != len(expected_hashes)
            or len(set(expected_keys)) != len(expected_keys)
            or _positive_int(self.alias_count, name="alias count") != len(aliases)
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INCOMPLETE_ALIAS_FANOUT,
                "physical-action fanout lost, duplicated, or reordered an alias",
            )
        identity = _dedup_identity(self.dedup_identity, name="fanout dedup identity")
        for alias in aliases:
            if (
                type(alias.packed_action) is not V1PackedRemovalAction
                or v1_signed_work_packed_action_dedup_identity(alias.packed_action)
                != identity
            ):
                _refuse(
                    V1SignedWorkPhysicalActionCAxisRefusalCode.INCOMPLETE_ALIAS_FANOUT,
                    "fanout alias names another physical action",
                )
        if type(self.all_aliases_retained) is not bool or not self.all_aliases_retained:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INCOMPLETE_ALIAS_FANOUT,
                "fanout did not retain every upstream alias",
            )
        _closed_authority(
            (
                self.C_outcomes_seen,
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical-action alias fanout",
        )
        if (
            self.authority_scope != SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY
            or self.schema != SIGNED_WORK_PHYSICAL_ACTION_ALIAS_FANOUT_SCHEMA
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.AUTHORITY_VIOLATION,
                "alias-fanout schema or authority changed",
            )
        object.__setattr__(self, "physical_action_ordinal", ordinal)
        object.__setattr__(self, "dedup_identity", identity)
        object.__setattr__(self, "source_aliases", aliases)
        object.__setattr__(self, "alias_crosswalk_hashes", expected_hashes)
        object.__setattr__(self, "coordinate_candidate_key_hashes", expected_keys)
        object.__setattr__(self, "alias_count", len(aliases))
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "physical_action_ordinal": self.physical_action_ordinal,
            "canonical_action_hash": self.canonical_action_hash,
            "dedup_identity": self.dedup_identity,
            "alias_crosswalk_hashes": self.alias_crosswalk_hashes,
            "coordinate_candidate_key_hashes": self.coordinate_candidate_key_hashes,
            "alias_count": self.alias_count,
            "all_aliases_retained": self.all_aliases_retained,
            "C_outcomes_seen": self.C_outcomes_seen,
            "C_measurement_authority": self.C_measurement_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if tuple(value.crosswalk_hash for value in self.source_aliases) != (
            self.alias_crosswalk_hashes
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical-action alias source changed after issuance",
            )
        observed = content_hash(self._payload(), prefix=_FANOUT_PREFIX)
        cached = self._receipt_hash
        if cached is not None and cached != observed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical-action alias fanout was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical-action alias fanout hash does not match",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


def _build_fanout(
    canonical: V1SignedWorkCanonicalPackedPhysicalAction,
) -> V1SignedWorkPhysicalActionAliasFanoutReceipt:
    aliases = tuple(canonical.aliases)
    return V1SignedWorkPhysicalActionAliasFanoutReceipt(
        physical_action_ordinal=canonical.canonical_ordinal,
        canonical_action_hash=canonical.canonical_action_hash,
        dedup_identity=canonical.dedup_identity,
        source_aliases=aliases,
        alias_crosswalk_hashes=tuple(value.crosswalk_hash for value in aliases),
        coordinate_candidate_key_hashes=tuple(
            value.coordinate_candidate_key_hash for value in aliases
        ),
        alias_count=len(aliases),
    )


def _build_entry(
    canonical: V1SignedWorkCanonicalPackedPhysicalAction,
    fanout: V1SignedWorkPhysicalActionAliasFanoutReceipt,
    *,
    parent_mask_hash: str,
    route_axis_hash: str,
    route_count: int,
) -> V1SignedWorkPhysicalActionCAxisEntry:
    action = canonical.packed_action
    packed = action.resulting_mask.packed_bits
    return V1SignedWorkPhysicalActionCAxisEntry(
        physical_action_ordinal=canonical.canonical_ordinal,
        canonical_action_hash=canonical.canonical_action_hash,
        dedup_identity=canonical.dedup_identity,
        packed_action=action,
        actual_packed_bytes=packed,
        actual_packed_sha256=hashlib.sha256(packed).hexdigest(),
        actual_packed_byte_count=len(packed),
        alias_fanout_receipt_hash=fanout.receipt_hash,
        parent_mask_hash=parent_mask_hash,
        route_axis_hash=route_axis_hash,
        route_count=route_count,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkPhysicalActionCAxis:
    """One-measurement-per-physical-action C axis with complete alias fanout."""

    pre_c_execution_bundle: V1SignedWorkPreCExecutionBundle = field(
        repr=False,
        compare=False,
    )
    d_freeze: V1SignedWorkPhysicalActionDFreezeIdentity = field(
        repr=False,
        compare=False,
    )
    c_stream: V1SignedWorkPhysicalActionCStreamIdentity = field(
        repr=False,
        compare=False,
    )
    complete_population: CompleteImageNetPopulationPairReceipt = field(
        repr=False,
        compare=False,
    )
    alpha_ledger_head: V1ProposalOrderAlphaLedger = field(
        repr=False,
        compare=False,
    )
    physical_actions: tuple[V1SignedWorkPhysicalActionCAxisEntry, ...]
    alias_fanouts: tuple[V1SignedWorkPhysicalActionAliasFanoutReceipt, ...]
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    pre_c_execution_bundle_hash: str
    packed_action_family_hash: str
    d_freeze_hash: str
    c_stream_hash: str
    complete_population_hash: str
    alpha_ledger_head_hash: str
    alpha_ledger_generation: int
    preoutcome_priority_action_hashes: tuple[str, ...]
    preoutcome_priority_axis_hash: str
    physical_action_count: int
    upstream_q_safe_alias_count: int
    _issuer_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    priority_policy: str = SIGNED_WORK_PHYSICAL_ACTION_PRIORITY_POLICY
    priority_depends_on_C_outcomes: bool = False
    nesting_authority: bool = False
    branch_semantics_authority: bool = False
    C_outcomes_seen: bool = False
    C_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY
    schema: str = SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_SCHEMA
    _axis_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuer_token is not _AXIS_TOKEN:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.AUTHORITY_VIOLATION,
                "physical C axes are issued only by the typed freeze factory",
            )
        bundle_hash, family_hash = _validate_pre_c_bundle(self.pre_c_execution_bundle)
        if type(self.d_freeze) is not V1SignedWorkPhysicalActionDFreezeIdentity:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
                "physical C axis requires the exact D-freeze owner",
            )
        if type(self.c_stream) is not V1SignedWorkPhysicalActionCStreamIdentity:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
                "physical C axis requires the exact C-stream owner",
            )
        d_hash = self.d_freeze.assert_integrity()
        c_hash = self.c_stream.assert_integrity()
        population_hash = _validate_population(self.complete_population)
        ledger_hash = _validate_ledger(self.alpha_ledger_head)
        actions = tuple(self.physical_actions)
        fanouts = tuple(self.alias_fanouts)
        if not actions or any(
            type(value) is not V1SignedWorkPhysicalActionCAxisEntry for value in actions
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.EMPTY_ACTION_AXIS,
                "physical C axis must contain at least one exact action",
            )
        if any(
            type(value) is not V1SignedWorkPhysicalActionAliasFanoutReceipt
            for value in fanouts
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
                "physical C axis fanouts must use the exact receipt type",
            )
        canonical = self.pre_c_execution_bundle.packed_action_family.canonical_actions
        if len(actions) != len(canonical) or len(fanouts) != len(canonical):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INCOMPLETE_ALIAS_FANOUT,
                "physical C axis cardinality differs from the canonical action family",
            )
        expected_ordinals = tuple(range(1, len(actions) + 1))
        if (
            tuple(value.physical_action_ordinal for value in actions)
            != expected_ordinals
            or tuple(value.physical_action_ordinal for value in fanouts)
            != expected_ordinals
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INCOMPLETE_ALIAS_FANOUT,
                "physical C action and fanout axes are incomplete or reordered",
            )
        seen_identities: set[V1SignedWorkPackedActionDedupIdentity] = set()
        packed_by_sha256: dict[str, bytes] = {}
        for source, action, fanout in zip(canonical, actions, fanouts, strict=True):
            action.assert_integrity()
            fanout.assert_integrity()
            if (
                action.canonical_action_hash != source.canonical_action_hash
                or action.dedup_identity != source.dedup_identity
                or action.packed_action is not source.packed_action
                or fanout.canonical_action_hash != source.canonical_action_hash
                or fanout.dedup_identity != source.dedup_identity
                or fanout.receipt_hash != action.alias_fanout_receipt_hash
            ):
                _refuse(
                    V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                    "physical C action or fanout differs from its canonical source",
                )
            identity = action.dedup_identity
            packed = action.actual_packed_bytes
            prior_packed = packed_by_sha256.get(identity[0])
            if identity in seen_identities or prior_packed == packed:
                _refuse(
                    V1SignedWorkPhysicalActionCAxisRefusalCode.DUPLICATE_PHYSICAL_ACTION,
                    "one physical payload appears more than once on the C axis",
                )
            if prior_packed is not None:
                _refuse(
                    V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                    "matching C-axis packed digests carried unequal actual bytes",
                )
            seen_identities.add(identity)
            packed_by_sha256[identity[0]] = packed
        for name in (
            "parent_mask_hash",
            "route_axis_hash",
            "pre_c_execution_bundle_hash",
            "packed_action_family_hash",
            "d_freeze_hash",
            "c_stream_hash",
            "complete_population_hash",
            "alpha_ledger_head_hash",
            "preoutcome_priority_axis_hash",
            "priority_policy",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        generation = _nonnegative_int(
            self.alpha_ledger_generation,
            name="alpha-ledger generation",
        )
        physical_count = _positive_int(
            self.physical_action_count,
            name="physical-action count",
        )
        upstream_alias_count = _positive_int(
            self.upstream_q_safe_alias_count,
            name="upstream Q-safe alias count",
        )
        receipt = self.pre_c_execution_bundle.receipt_family
        priority = tuple(self.preoutcome_priority_action_hashes)
        expected_priority = tuple(value.entry_hash for value in actions)
        expected_priority_hash = content_hash(
            {
                "policy": SIGNED_WORK_PHYSICAL_ACTION_PRIORITY_POLICY,
                "physical_action_entry_hashes": expected_priority,
                "C_outcomes_seen": False,
            },
            prefix=_PRIORITY_AXIS_PREFIX,
        )
        alias_count = sum(value.alias_count for value in fanouts)
        if (
            self.parent_mask_hash != receipt.parent_mask_hash
            or self.route_axis_hash != receipt.route_axis_hash
            or routes != receipt.route_count
            or self.pre_c_execution_bundle_hash != bundle_hash
            or self.packed_action_family_hash != family_hash
            or self.d_freeze_hash != d_hash
            or self.c_stream_hash != c_hash
            or self.complete_population_hash != population_hash
            or self.alpha_ledger_head_hash != ledger_hash
            or generation != self.alpha_ledger_head.proposal_generation
            or self.d_freeze.pre_c_execution_bundle_hash != bundle_hash
            or self.c_stream.causal_parent_d_freeze_hash != d_hash
            or self.c_stream.complete_population_hash != population_hash
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.STALE_PARENT_OR_AXIS,
                "physical C axis does not bind one exact parent, route axis, or source lineage",
            )
        if (
            priority != expected_priority
            or self.preoutcome_priority_axis_hash != expected_priority_hash
            or self.priority_policy != SIGNED_WORK_PHYSICAL_ACTION_PRIORITY_POLICY
            or type(self.priority_depends_on_C_outcomes) is not bool
            or self.priority_depends_on_C_outcomes
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.POST_OUTCOME,
                "physical C priority is not the deterministic pre-outcome action axis",
            )
        if (
            physical_count != len(actions)
            or upstream_alias_count != alias_count
            or alias_count
            != len(
                self.pre_c_execution_bundle.packed_action_family.q_safe_crosswalk_hashes
            )
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INCOMPLETE_ALIAS_FANOUT,
                "physical action or alias cardinality does not close",
            )
        _closed_authority(
            (
                self.nesting_authority,
                self.branch_semantics_authority,
                self.C_outcomes_seen,
                self.C_measurement_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="physical-action C axis",
        )
        if (
            self.authority_scope != SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY
            or self.schema != SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_SCHEMA
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.AUTHORITY_VIOLATION,
                "physical-action C-axis schema or authority changed",
            )
        object.__setattr__(self, "physical_actions", actions)
        object.__setattr__(self, "alias_fanouts", fanouts)
        object.__setattr__(self, "route_count", receipt.route_count)
        object.__setattr__(self, "alpha_ledger_generation", generation)
        object.__setattr__(self, "preoutcome_priority_action_hashes", priority)
        object.__setattr__(self, "physical_action_count", len(actions))
        object.__setattr__(self, "upstream_q_safe_alias_count", alias_count)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "physical_action_entry_hashes": tuple(
                value.entry_hash for value in self.physical_actions
            ),
            "alias_fanout_receipt_hashes": tuple(
                value.receipt_hash for value in self.alias_fanouts
            ),
            "parent_mask_hash": self.parent_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "pre_c_execution_bundle_hash": self.pre_c_execution_bundle_hash,
            "packed_action_family_hash": self.packed_action_family_hash,
            "d_freeze_hash": self.d_freeze_hash,
            "c_stream_hash": self.c_stream_hash,
            "complete_population_hash": self.complete_population_hash,
            "alpha_ledger_head_hash": self.alpha_ledger_head_hash,
            "alpha_ledger_generation": self.alpha_ledger_generation,
            "preoutcome_priority_action_hashes": (
                self.preoutcome_priority_action_hashes
            ),
            "preoutcome_priority_axis_hash": self.preoutcome_priority_axis_hash,
            "physical_action_count": self.physical_action_count,
            "upstream_q_safe_alias_count": self.upstream_q_safe_alias_count,
            "priority_policy": self.priority_policy,
            "priority_depends_on_C_outcomes": self.priority_depends_on_C_outcomes,
            "nesting_authority": self.nesting_authority,
            "branch_semantics_authority": self.branch_semantics_authority,
            "C_outcomes_seen": self.C_outcomes_seen,
            "C_measurement_authority": self.C_measurement_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def axis_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        bundle_hash, family_hash = _validate_pre_c_bundle(self.pre_c_execution_bundle)
        d_hash = self.d_freeze.assert_integrity()
        c_hash = self.c_stream.assert_integrity()
        population_hash = _validate_population(self.complete_population)
        ledger_hash = _validate_ledger(self.alpha_ledger_head)
        for value in self.physical_actions:
            value.assert_integrity()
        for value in self.alias_fanouts:
            value.assert_integrity()
        if (
            bundle_hash != self.pre_c_execution_bundle_hash
            or family_hash != self.packed_action_family_hash
            or d_hash != self.d_freeze_hash
            or c_hash != self.c_stream_hash
            or population_hash != self.complete_population_hash
            or ledger_hash != self.alpha_ledger_head_hash
        ):
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical C-axis source changed after issuance",
            )
        observed = content_hash(self._payload(), prefix=_AXIS_PREFIX)
        cached = self._axis_hash
        if cached is not None and cached != observed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical C axis was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "physical C-axis hash does not match",
            )
        object.__setattr__(self, "_axis_hash", observed)
        return observed


def freeze_v1_signed_work_physical_action_c_axis(
    *,
    pre_c_execution_bundle: V1SignedWorkPreCExecutionBundle,
    d_freeze: V1SignedWorkPhysicalActionDFreezeIdentity,
    c_stream: V1SignedWorkPhysicalActionCStreamIdentity,
    complete_population: CompleteImageNetPopulationPairReceipt,
    alpha_ledger_head: V1ProposalOrderAlphaLedger,
) -> V1SignedWorkPhysicalActionCAxis:
    """Register every exact physical action once and every alias separately."""

    bundle_hash, family_hash = _validate_pre_c_bundle(pre_c_execution_bundle)
    if type(d_freeze) is not V1SignedWorkPhysicalActionDFreezeIdentity:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            "physical C-axis freeze requires the exact D-freeze owner",
        )
    if type(c_stream) is not V1SignedWorkPhysicalActionCStreamIdentity:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
            "physical C-axis freeze requires the exact C-stream owner",
        )
    d_hash = d_freeze.assert_integrity()
    c_hash = c_stream.assert_integrity()
    population_hash = _validate_population(complete_population)
    ledger_hash = _validate_ledger(alpha_ledger_head)
    if population_hash != d_freeze.complete_population_hash:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.POPULATION_MISMATCH,
            "physical C axis changed the complete train population after D freeze",
        )
    if (
        ledger_hash != d_freeze.alpha_ledger_head_hash
        or alpha_ledger_head.proposal_generation != d_freeze.alpha_ledger_generation
    ):
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.ALPHA_LEDGER_MISMATCH,
            "physical C axis changed the non-recycling alpha-ledger head",
        )
    if (
        d_freeze.pre_c_execution_bundle_hash != bundle_hash
        or d_freeze.packed_action_family_hash != family_hash
        or c_stream.causal_parent_d_freeze_hash != d_hash
    ):
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.STALE_PARENT_OR_AXIS,
            "physical C axis changed its exact pre-C or D lineage",
        )
    family = pre_c_execution_bundle.packed_action_family
    canonical = tuple(family.canonical_actions)
    if not canonical:
        _refuse(
            V1SignedWorkPhysicalActionCAxisRefusalCode.EMPTY_ACTION_AXIS,
            "physical C-axis freeze received no Q-safe canonical action",
        )
    identities: set[V1SignedWorkPackedActionDedupIdentity] = set()
    packed_by_sha256: dict[str, bytes] = {}
    for value in canonical:
        if type(value) is not V1SignedWorkCanonicalPackedPhysicalAction:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.INVALID_TYPE,
                "canonical physical-action family changed type",
            )
        identity = v1_signed_work_packed_action_dedup_identity(value.packed_action)
        packed = value.packed_action.resulting_mask.packed_bits
        prior_packed = packed_by_sha256.get(identity[0])
        if identity in identities or prior_packed == packed:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.DUPLICATE_PHYSICAL_ACTION,
                "canonical family repeats one physical action or exact payload",
            )
        if prior_packed is not None:
            _refuse(
                V1SignedWorkPhysicalActionCAxisRefusalCode.HASH_MISMATCH,
                "matching packed-byte digests carried unequal actual bytes",
            )
        identities.add(identity)
        packed_by_sha256[identity[0]] = packed
    fanouts = tuple(_build_fanout(value) for value in canonical)
    receipt = pre_c_execution_bundle.receipt_family
    actions = tuple(
        _build_entry(
            source,
            fanout,
            parent_mask_hash=receipt.parent_mask_hash,
            route_axis_hash=receipt.route_axis_hash,
            route_count=receipt.route_count,
        )
        for source, fanout in zip(canonical, fanouts, strict=True)
    )
    priority = tuple(value.entry_hash for value in actions)
    priority_hash = content_hash(
        {
            "policy": SIGNED_WORK_PHYSICAL_ACTION_PRIORITY_POLICY,
            "physical_action_entry_hashes": priority,
            "C_outcomes_seen": False,
        },
        prefix=_PRIORITY_AXIS_PREFIX,
    )
    return V1SignedWorkPhysicalActionCAxis(
        _issuer_token=_AXIS_TOKEN,
        pre_c_execution_bundle=pre_c_execution_bundle,
        d_freeze=d_freeze,
        c_stream=c_stream,
        complete_population=complete_population,
        alpha_ledger_head=alpha_ledger_head,
        physical_actions=actions,
        alias_fanouts=fanouts,
        parent_mask_hash=receipt.parent_mask_hash,
        route_axis_hash=receipt.route_axis_hash,
        route_count=receipt.route_count,
        pre_c_execution_bundle_hash=bundle_hash,
        packed_action_family_hash=family_hash,
        d_freeze_hash=d_hash,
        c_stream_hash=c_hash,
        complete_population_hash=population_hash,
        alpha_ledger_head_hash=ledger_hash,
        alpha_ledger_generation=alpha_ledger_head.proposal_generation,
        preoutcome_priority_action_hashes=priority,
        preoutcome_priority_axis_hash=priority_hash,
        physical_action_count=len(actions),
        upstream_q_safe_alias_count=sum(value.alias_count for value in fanouts),
    )


__all__ = [
    "SIGNED_WORK_PHYSICAL_ACTION_ALIAS_FANOUT_SCHEMA",
    "SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_AUTHORITY",
    "SIGNED_WORK_PHYSICAL_ACTION_C_AXIS_SCHEMA",
    "SIGNED_WORK_PHYSICAL_ACTION_C_ENTRY_SCHEMA",
    "SIGNED_WORK_PHYSICAL_ACTION_C_STREAM_SCHEMA",
    "SIGNED_WORK_PHYSICAL_ACTION_D_FREEZE_SCHEMA",
    "SIGNED_WORK_PHYSICAL_ACTION_PRIORITY_POLICY",
    "SIGNED_WORK_D_STABLE_PHYSICAL_ACTION_C_ADMISSION_SCHEMA",
    "SIGNED_WORK_TRUE_Q_PHYSICAL_ACTION_C_ADMISSION_SCHEMA",
    "V1SignedWorkDStablePhysicalActionCAdmission",
    "V1SignedWorkPhysicalActionAliasFanoutReceipt",
    "V1SignedWorkPhysicalActionCAxis",
    "V1SignedWorkPhysicalActionCAxisEntry",
    "V1SignedWorkPhysicalActionCAxisRefusal",
    "V1SignedWorkPhysicalActionCAxisRefusalCode",
    "V1SignedWorkPhysicalActionCStreamIdentity",
    "V1SignedWorkPhysicalActionDFreezeIdentity",
    "V1SignedWorkTrueQPhysicalActionCAdmission",
    "freeze_v1_signed_work_physical_action_c_axis",
    "freeze_v1_signed_work_physical_action_c_stream_identity",
    "freeze_v1_signed_work_physical_action_d_identity",
    "issue_v1_signed_work_d_stable_physical_action_c_admission",
    "issue_v1_signed_work_true_q_physical_action_c_admission",
]
