"""Compact canonical F/Q/R proposal carrier for the modern-v1 controller.

The bounded C-stream branch-family authorization is the only scientific input
accepted here.  This module copies its complete, already frozen physical queue
into a durable controller DTO while retaining each independently certified
F/Q/R interval, both directional-response burdens, packed hard action, and
source coordinate.  It does not recertify absent budgets and never converts an
F/Q-only object into F/Q/R authority.

The result is proposal authority only.  Candidate-specific calibration and the
paired BCE oracle remain mandatory before the controller can issue its private,
single-use mask-commit capability.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Final

from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .contracts import content_hash
from .modern_v1_campaign_contract import V1OptionACampaignBinding
from .modern_v1_candidate_composition import (
    V1PackedRemovalAction,
    V1RemovedRouteSetIdentity,
)
from .modern_v1_fqr_compact_branch_family import (
    V1CompactFQRBranchFamilyAuthorization,
    V1CompactFQRBranchFamilySourceAlias,
)


FQR_ACTION_NUMERIC_STATE_SCHEMA: Final = (
    "partition_free_v1_fqr_action_numeric_state_v1"
)
FQR_ACTION_NUMERIC_STATE_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_action_numeric_state_v2"
)
FQR_CANDIDATE_PROPOSAL_SCHEMA: Final = (
    "partition_free_v1_fqr_candidate_proposal_v1"
)
FQR_CANDIDATE_PROPOSAL_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_candidate_proposal_v2"
)
FQR_CANDIDATE_COMPOSITION_SCHEMA: Final = (
    "partition_free_v1_fqr_candidate_composition_result_v1"
)
FQR_CANDIDATE_COMPOSITION_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_candidate_composition_result_v2"
)
FQR_ACQUISITION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_fqr_acquisition_generation_receipt_v1"
)
FQR_ACQUISITION_RECEIPT_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_acquisition_generation_receipt_v2"
)
FQR_AUTHORIZED_ACQUISITION_SCHEMA: Final = (
    "partition_free_v1_fqr_authorized_acquisition_v1"
)
FQR_AUTHORIZED_ACQUISITION_V2_SCHEMA: Final = (
    "partition_free_v1_fqr_authorized_acquisition_v2"
)
FQR_RESOURCE_DESCENDING_COMPOSER_ID: Final = (
    "canonical_compact_fqr_cross_branch_resource_descending_queue_v1"
)
FQR_RUNTIME_AUTHORITY_SCOPE: Final = (
    "canonical_prefrozen_cross_branch_fqr_proposal_queue_only_"
    "paired_bce_required_no_calibration_oracle_commit_terminal_or_pareto_"
    "authority_v1"
)

_STATE_TOKEN: Final = object()
_PROPOSAL_TOKEN: Final = object()
_COMPOSITION_TOKEN: Final = object()


def _identity(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _freeze(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{name} keys must be nonempty strings")
            frozen[key] = _freeze(item, f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, f"{name}[{index}]") for index, item in enumerate(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a nonfinite float")
        return float(value)
    if isinstance(value, (str, bytes, int, bool)) or value is None:
        return value
    raise TypeError(f"{name} contains unsupported value {type(value)!r}")


def _fresh_mask_hash(mask: HardMask, name: str) -> str:
    if type(mask) is not HardMask:
        raise TypeError(f"{name} must be an exact HardMask")
    packed = mask.packed_bits
    if type(packed) is not bytes or len(packed) != (mask.route_count + 7) // 8:
        raise ValueError(f"{name} has an invalid packed representation")
    padding = len(packed) * 8 - mask.route_count
    if padding and packed[-1] & ((1 << padding) - 1):
        raise ValueError(f"{name} contains nonzero padding bits")
    return content_hash(
        {
            "route_axis_hash": mask.route_axis_hash,
            "route_count": mask.route_count,
            "packed_bits": packed,
        },
        prefix="irg-mask-",
    )


def _branch_value(value: object, name: str) -> str:
    raw = getattr(value, "value", value)
    if raw not in {"spectral", "diagonal_l1"}:
        raise ValueError(f"{name} is not a registered relaxed-KKT branch")
    return str(raw)


def _canonical_branch_evidence_hashes(
    *,
    positive: object,
    negative: object,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Validate the complete positive-or-stable-negative branch partition.

    A materialized queue necessarily has at least one positive branch.  A
    stable-negative row is valid only in the v2 two-branch union, where the
    other branch supplies the physical queue.  Keeping the two axes separate
    prevents exclusion evidence from silently impersonating proposal evidence.
    """

    if type(positive) is not tuple or type(negative) is not tuple:
        raise TypeError("F/Q/R branch evidence axes must be exact tuples")
    normalized: list[tuple[tuple[str, str], ...]] = []
    for rows, label in (
        (positive, "source evidence"),
        (negative, "source stable-negative evidence"),
    ):
        values: list[tuple[str, str]] = []
        for row in rows:
            if type(row) is not tuple or len(row) != 2:
                raise TypeError(f"{label} rows must be exact pairs")
            values.append(
                (
                    _branch_value(row[0], f"{label} branch"),
                    _identity(row[1], f"{label} hash"),
                )
            )
        result = tuple(values)
        if result != tuple(sorted(result)) or len(
            {branch for branch, _value in result}
        ) != len(result):
            raise ValueError(f"{label} rows are duplicated or reordered")
        normalized.append(result)
    positives, negatives = normalized
    all_branches = tuple(branch for branch, _value in positives + negatives)
    if (
        len(positives) not in {1, 2}
        or len(negatives) not in {0, 1}
        or len(all_branches) not in {1, 2}
        or len(set(all_branches)) != len(all_branches)
        or (negatives and (len(positives) != 1 or len(all_branches) != 2))
    ):
        raise ValueError(
            "F/Q/R materialized branch provenance must be one or two positive "
            "branches, or one positive plus one distinct stable-negative branch"
        )
    return positives, negatives


def _canonical_source_aliases(
    values: Sequence[object],
) -> tuple[V1CompactFQRBranchFamilySourceAlias, ...]:
    """Detach compact source aliases from their upstream receipt objects.

    The branch-family seam owns the exact alias type.  Its public aliases are
    immutable scalar coordinates.  We intentionally copy only those scalar
    coordinates so neither raw C records nor a record-bearing receipt can leak
    into the controller checkpoint.
    """

    result: list[V1CompactFQRBranchFamilySourceAlias] = []
    for alias in values:
        if type(alias) is not V1CompactFQRBranchFamilySourceAlias:
            raise TypeError("F/Q/R source alias must use the exact compact type")
        alias.assert_integrity()
        result.append(alias)
    if not result or len({value.alias_hash for value in result}) != len(result):
        raise ValueError("F/Q/R source aliases must be nonempty and unique")
    return tuple(result)


@dataclass(frozen=True, slots=True, init=False)
class V1FQRActionNumericState:
    """One compact hard action with explicit bounded F/Q/R coordinates."""

    queue_index: int
    representative_branch: str
    representative_endpoint_index: int
    representative_frozen_action_hash: str
    source_aliases: tuple[V1CompactFQRBranchFamilySourceAlias, ...]
    source_aliases_hash: str
    parent_mask: HardMask = field(repr=False, compare=False)
    packed_action: V1PackedRemovalAction = field(repr=False, compare=False)
    parent_mask_hash: str
    resulting_mask_hash: str
    packed_action_hash: str
    route_axis_hash: str
    route_count: int
    removed_count: int
    removed_route_index_hash: str
    registered_resource: float
    resource_units: str
    resource_source_hash: str
    resource_receipt_hash: str
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
    fqr_interval_hash: str
    branch_family_authorization_hash: str
    branch_family_entry_hash: str
    prefrozen_family_hash: str
    joint_fqr_family_registration_hash: str
    source_branch_evidence_hashes: tuple[tuple[str, str], ...]
    generation_index: int
    independently_certified_fqr_bounds: bool
    complete_queue_frozen_before_K_O: bool
    proposal_only: bool
    paired_bce_oracle_required: bool
    calibration_K_open_authority: bool
    paired_oracle_O_open_authority: bool
    candidate_freeze_authority: bool
    irreversible_decision_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    schema: str = FQR_ACTION_NUMERIC_STATE_SCHEMA
    _state_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _STATE_TOKEN:
            raise TypeError("V1FQRActionNumericState is issuer-only")
        expected = tuple(
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        )
        if set(values) != set(expected):
            missing = sorted(set(expected).difference(values))
            extra = sorted(set(values).difference(expected))
            raise TypeError(f"F/Q/R state fields differ: missing={missing}, extra={extra}")
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", FQR_ACTION_NUMERIC_STATE_SCHEMA)
        object.__setattr__(self, "_state_hash", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.parent_mask) is not HardMask:
            raise TypeError("F/Q/R state requires an exact parent mask")
        if type(self.packed_action) is not V1PackedRemovalAction:
            raise TypeError("F/Q/R state requires an exact packed action")
        parent_hash = _fresh_mask_hash(self.parent_mask, "F/Q/R parent mask")
        child = self.packed_action.resulting_mask
        child_hash = _fresh_mask_hash(child, "F/Q/R resulting mask")
        self.parent_mask.assert_same_axis(child)
        removed_count, removed_hash = child.removed_index_count_and_hash_from(
            self.parent_mask
        )
        if (
            self.parent_mask_hash != parent_hash
            or self.resulting_mask_hash != child_hash
            or self.packed_action_hash != self.packed_action.action_hash
            or self.route_axis_hash != self.parent_mask.route_axis_hash
            or self.route_count != self.parent_mask.route_count
            or self.removed_count != removed_count
            or self.removed_count != self.packed_action.removed_count
            or self.removed_route_index_hash != removed_hash
            or self.removed_route_index_hash != self.packed_action.removed_route_index_hash
        ):
            raise ValueError("F/Q/R action identity differs from its exact masks")
        object.__setattr__(self, "queue_index", _nonnegative_int(self.queue_index, "queue_index"))
        object.__setattr__(
            self,
            "representative_endpoint_index",
            _positive_int(self.representative_endpoint_index, "representative_endpoint_index"),
        )
        object.__setattr__(self, "generation_index", _nonnegative_int(self.generation_index, "generation_index"))
        object.__setattr__(self, "route_count", _positive_int(self.route_count, "route_count"))
        object.__setattr__(self, "removed_count", _positive_int(self.removed_count, "removed_count"))
        object.__setattr__(
            self,
            "representative_branch",
            _branch_value(self.representative_branch, "representative_branch"),
        )
        for name in (
            "representative_frozen_action_hash",
            "source_aliases_hash",
            "parent_mask_hash",
            "resulting_mask_hash",
            "packed_action_hash",
            "route_axis_hash",
            "removed_route_index_hash",
            "resource_units",
            "resource_source_hash",
            "resource_receipt_hash",
            "fqr_interval_hash",
            "branch_family_authorization_hash",
            "branch_family_entry_hash",
            "prefrozen_family_hash",
            "joint_fqr_family_registration_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        aliases = tuple(self.source_aliases)
        if not aliases or any(
            type(value) is not V1CompactFQRBranchFamilySourceAlias
            for value in aliases
        ):
            raise TypeError("F/Q/R state requires exact compact source aliases")
        for alias in aliases:
            alias.assert_integrity()
        alias_hashes = tuple(value.alias_hash for value in aliases)
        if len(set(alias_hashes)) != len(alias_hashes):
            raise ValueError("F/Q/R source aliases must be nonempty and unique")
        if self.source_aliases_hash != content_hash(
            alias_hashes,
            prefix="partition-free-v1-fqr-source-aliases-",
        ):
            raise ValueError("F/Q/R source-alias hash is not reproducible")
        evidence, negative_evidence = _canonical_branch_evidence_hashes(
            positive=self.source_branch_evidence_hashes,
            negative=getattr(self, "source_branch_negative_evidence_hashes", ()),
        )
        positive_branches = {branch for branch, _value in evidence}
        if (
            self.representative_branch not in positive_branches
            or any(alias.branch.value not in positive_branches for alias in aliases)
        ):
            raise ValueError(
                "F/Q/R physical action aliases must originate only from positive "
                "branch evidence"
            )
        values = {
            name: _finite_nonnegative(getattr(self, name), name)
            for name in (
                "registered_resource",
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
            )
        }
        for lower, upper in (
            (values["f_lower"], values["f_upper"]),
            (values["q_lower"], values["q_upper"]),
            (values["r_half_lower"], values["r_half_upper"]),
            (values["r_one_lower"], values["r_one_upper"]),
            (values["half_response_lower"], values["half_response_upper"]),
            (values["one_response_lower"], values["one_response_upper"]),
        ):
            if lower > upper:
                raise ValueError("F/Q/R state contains an inverted interval")
        interval_payload = self._interval_payload(values=values)
        if self.fqr_interval_hash != content_hash(
            interval_payload,
            prefix="partition-free-v1-fqr-certified-interval-",
        ):
            raise ValueError("F/Q/R interval hash is not reproducible")
        resource_payload = self._resource_payload(resource=values["registered_resource"])
        if self.resource_receipt_hash != content_hash(
            resource_payload,
            prefix="partition-free-v1-fqr-resource-",
        ):
            raise ValueError("F/Q/R resource receipt is not reproducible")
        firewall = (
            self.independently_certified_fqr_bounds,
            self.complete_queue_frozen_before_K_O,
            self.proposal_only,
            self.paired_bce_oracle_required,
            not self.calibration_K_open_authority,
            not self.paired_oracle_O_open_authority,
            not self.candidate_freeze_authority,
            not self.irreversible_decision_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            not self.execution_authority,
        )
        if any(type(flag) is not bool or not flag for flag in firewall):
            raise ValueError("F/Q/R state crossed its proposal-only authority firewall")
        expected_schema = (
            FQR_ACTION_NUMERIC_STATE_V2_SCHEMA
            if negative_evidence
            else FQR_ACTION_NUMERIC_STATE_SCHEMA
        )
        if self.schema != expected_schema:
            raise ValueError("unsupported F/Q/R action-state schema")
        object.__setattr__(self, "source_aliases", aliases)
        object.__setattr__(self, "source_branch_evidence_hashes", evidence)
        if hasattr(self, "source_branch_negative_evidence_hashes"):
            object.__setattr__(
                self,
                "source_branch_negative_evidence_hashes",
                negative_evidence,
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.assert_integrity()

    def _interval_payload(self, *, values: Mapping[str, float] | None = None) -> dict[str, object]:
        names = (
            "f_lower", "f_upper", "q_lower", "q_upper", "r_half_lower",
            "r_half_upper", "r_one_lower", "r_one_upper",
            "half_response_lower", "half_response_upper",
            "one_response_lower", "one_response_upper",
        )
        source = {
            name: getattr(self, name) if values is None else values[name]
            for name in names
        }
        return {
            "schema": "partition_free_v1_fqr_certified_interval_v1",
            "branch_family_authorization_hash": self.branch_family_authorization_hash,
            "branch_family_entry_hash": self.branch_family_entry_hash,
            "packed_action_hash": self.packed_action_hash,
            "representative_branch": self.representative_branch,
            "representative_endpoint_index": self.representative_endpoint_index,
            **dict(source),
            "independently_certified_fqr_bounds": True,
        }

    def _resource_payload(self, *, resource: float | None = None) -> dict[str, object]:
        return {
            "schema": "partition_free_v1_fqr_resource_receipt_v1",
            "branch_family_authorization_hash": self.branch_family_authorization_hash,
            "branch_family_entry_hash": self.branch_family_entry_hash,
            "packed_action_hash": self.packed_action_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "resulting_mask_hash": self.resulting_mask_hash,
            "registered_resource": self.registered_resource if resource is None else resource,
            "resource_units": self.resource_units,
            "resource_source_hash": self.resource_source_hash,
        }

    def _payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "queue_index": self.queue_index,
            "representative_branch": self.representative_branch,
            "representative_endpoint_index": self.representative_endpoint_index,
            "representative_frozen_action_hash": self.representative_frozen_action_hash,
            "source_alias_hashes": tuple(
                value.alias_hash for value in self.source_aliases
            ),
            "source_aliases_hash": self.source_aliases_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "resulting_mask_hash": self.resulting_mask_hash,
            "packed_action_hash": self.packed_action_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "removed_count": self.removed_count,
            "removed_route_index_hash": self.removed_route_index_hash,
            "registered_resource": self.registered_resource,
            "resource_units": self.resource_units,
            "resource_source_hash": self.resource_source_hash,
            "resource_receipt_hash": self.resource_receipt_hash,
            "fqr_interval_hash": self.fqr_interval_hash,
            "fqr_intervals": self._interval_payload(),
            "branch_family_authorization_hash": self.branch_family_authorization_hash,
            "branch_family_entry_hash": self.branch_family_entry_hash,
            "prefrozen_family_hash": self.prefrozen_family_hash,
            "joint_fqr_family_registration_hash": self.joint_fqr_family_registration_hash,
            "source_branch_evidence_hashes": self.source_branch_evidence_hashes,
            "generation_index": self.generation_index,
            "independently_certified_fqr_bounds": self.independently_certified_fqr_bounds,
            "complete_queue_frozen_before_K_O": self.complete_queue_frozen_before_K_O,
            "proposal_only": self.proposal_only,
            "paired_bce_oracle_required": self.paired_bce_oracle_required,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "execution_authority": self.execution_authority,
        }
        negative_evidence = getattr(
            self, "source_branch_negative_evidence_hashes", ()
        )
        if negative_evidence:
            payload["source_branch_negative_evidence_hashes"] = (
                negative_evidence
            )
        return payload

    @property
    def state_hash(self) -> str:
        return self.assert_integrity()

    @property
    def numeric_state_hash(self) -> str:
        return self.state_hash

    @property
    def resulting_mask(self) -> HardMask:
        return self.packed_action.resulting_mask

    @property
    def removed_routes(self) -> V1RemovedRouteSetIdentity:
        return V1RemovedRouteSetIdentity(
            route_count=self.route_count,
            count=self.removed_count,
            index_hash=self.removed_route_index_hash,
        )

    @property
    def compact_removed_indices_available(self) -> bool:
        return False

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            _fresh_mask_hash(self.parent_mask, "F/Q/R parent mask") != self.parent_mask_hash
            or _fresh_mask_hash(self.resulting_mask, "F/Q/R child mask") != self.resulting_mask_hash
            or self.packed_action.action_hash != self.packed_action_hash
        ):
            raise ValueError("F/Q/R state masks were tampered")
        for alias in self.source_aliases:
            alias.assert_integrity()
        observed = content_hash(self._payload(), prefix="partition-free-v1-fqr-action-state-")
        if self._state_hash is not None and observed != self._state_hash:
            raise ValueError("F/Q/R action state was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("F/Q/R action-state hash does not match")
        object.__setattr__(self, "_state_hash", observed)
        return observed

    def assert_can_certify_decision(self) -> None:
        raise PermissionError("F/Q/R proposal states are not scientific certificates")

    def assert_can_commit_mask(self) -> None:
        raise PermissionError("F/Q/R proposal states cannot commit a hard mask")


@dataclass(frozen=True, slots=True, init=False)
class V1FQRActionNumericStateV2(V1FQRActionNumericState):
    """Mixed positive/stable-negative action state with explicit v2 provenance."""

    source_branch_negative_evidence_hashes: tuple[tuple[str, str], ...] = ()

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _STATE_TOKEN:
            raise TypeError("V1FQRActionNumericStateV2 is issuer-only")
        base_expected = tuple(
            name
            for name, declaration in V1FQRActionNumericState.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        )
        expected = (*base_expected, "source_branch_negative_evidence_hashes")
        if set(values) != set(expected):
            missing = sorted(set(expected).difference(values))
            extra = sorted(set(values).difference(expected))
            raise TypeError(
                f"F/Q/R v2 state fields differ: missing={missing}, extra={extra}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", FQR_ACTION_NUMERIC_STATE_V2_SCHEMA)
        object.__setattr__(self, "_state_hash", None)
        self.__post_init__()


@dataclass(frozen=True, slots=True, init=False)
class V1FQRCandidateProposal:
    """One quotient-physical F/Q/R candidate in the immutable oracle queue."""

    numeric_state: V1FQRActionNumericState
    domain_hash: str
    certification_scope_hash: str
    evaluation_receipt_hash: str
    candidate_id: str
    proposal_only: bool
    proposal_witness_only: bool
    fresh_relinearization_required: bool
    paired_bce_oracle_required: bool
    candidate_freeze_authority: bool
    bce_oracle_authority: bool
    irreversible_decision_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    schema: str = FQR_CANDIDATE_PROPOSAL_SCHEMA

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _PROPOSAL_TOKEN:
            raise TypeError("V1FQRCandidateProposal is issuer-only")
        expected = tuple(
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        )
        if set(values) != set(expected):
            raise TypeError("F/Q/R proposal fields differ from their schema")
        for name in expected:
            object.__setattr__(self, name, values[name])
        state = values.get("numeric_state")
        object.__setattr__(
            self,
            "schema",
            (
                FQR_CANDIDATE_PROPOSAL_V2_SCHEMA
                if type(state) is V1FQRActionNumericStateV2
                else FQR_CANDIDATE_PROPOSAL_SCHEMA
            ),
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.numeric_state) not in {
            V1FQRActionNumericState,
            V1FQRActionNumericStateV2,
        }:
            raise TypeError("F/Q/R proposal requires its issued numeric state")
        self.numeric_state.assert_integrity()
        for name in (
            "domain_hash",
            "certification_scope_hash",
            "evaluation_receipt_hash",
            "candidate_id",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        firewall = (
            self.proposal_only,
            self.proposal_witness_only,
            self.fresh_relinearization_required,
            self.paired_bce_oracle_required,
            not self.candidate_freeze_authority,
            not self.bce_oracle_authority,
            not self.irreversible_decision_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            not self.execution_authority,
        )
        if any(type(flag) is not bool or not flag for flag in firewall):
            raise ValueError("F/Q/R proposal crossed its proposal-only authority firewall")
        expected_schema = (
            FQR_CANDIDATE_PROPOSAL_V2_SCHEMA
            if type(self.numeric_state) is V1FQRActionNumericStateV2
            else FQR_CANDIDATE_PROPOSAL_SCHEMA
        )
        if self.schema != expected_schema:
            raise ValueError("unsupported F/Q/R proposal schema")
        expected_receipt = content_hash(
            self._evaluation_payload(),
            prefix="partition-free-v1-fqr-candidate-evaluation-",
        )
        if self.evaluation_receipt_hash != expected_receipt:
            raise ValueError("F/Q/R evaluation receipt is not reproducible")
        expected_candidate = content_hash(
            {
                "evaluation_receipt_hash": expected_receipt,
                "resulting_mask_hash": self.resulting_mask_hash,
                "removed_route_index_hash": self.removed_route_index_hash,
                "fqr_interval_hash": self.fqr_interval_hash,
                "branch_family_entry_hash": self.branch_family_entry_hash,
            },
            prefix="partition-free-v1-fqr-candidate-",
        )
        if self.candidate_id != expected_candidate:
            raise ValueError("F/Q/R candidate identity is not reproducible")

    def _evaluation_payload(self) -> dict[str, object]:
        return {
            "schema": (
                "partition_free_v1_fqr_candidate_evaluation_v2"
                if type(self.numeric_state) is V1FQRActionNumericStateV2
                else "partition_free_v1_fqr_candidate_evaluation_v1"
            ),
            "numeric_state_hash": self.numeric_state_hash,
            "domain_hash": self.domain_hash,
            "certification_scope_hash": self.certification_scope_hash,
            "queue_index": self.queue_index,
            "representative_branch": self.representative_branch,
            "representative_endpoint_index": self.representative_endpoint_index,
            "packed_action_hash": self.packed_action_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "resulting_mask_hash": self.resulting_mask_hash,
            "removed_count": self.removed_count,
            "removed_route_index_hash": self.removed_route_index_hash,
            "registered_resource": self.resource_gain,
            "resource_receipt_hash": self.resource_receipt_hash,
            "fqr_interval_hash": self.fqr_interval_hash,
            "branch_family_authorization_hash": self.branch_family_authorization_hash,
            "branch_family_entry_hash": self.branch_family_entry_hash,
            "proposal_only": True,
            "paired_bce_oracle_required": True,
            "candidate_freeze_authority": False,
            "bce_oracle_authority": False,
            "irreversible_decision_authority": False,
            "mask_commit_authority": False,
            "terminal_authority": False,
            "pareto_authority": False,
            "execution_authority": False,
        }

    @property
    def constraints_hash(self) -> str:
        """Generic controller compatibility name; this is not an F/Q budget."""

        return self.certification_scope_hash

    @property
    def proposal_id(self) -> str:
        return self.candidate_id

    @property
    def proposal_hash(self) -> str:
        return self.assert_integrity()

    @property
    def numeric_state_hash(self) -> str:
        return self.numeric_state.state_hash

    @property
    def queue_index(self) -> int:
        return self.numeric_state.queue_index

    @property
    def representative_branch(self) -> str:
        return self.numeric_state.representative_branch

    @property
    def representative_endpoint_index(self) -> int:
        return self.numeric_state.representative_endpoint_index

    @property
    def endpoint_index(self) -> int:
        return self.representative_endpoint_index

    @property
    def action_hash(self) -> str:
        return self.numeric_state.packed_action_hash

    @property
    def packed_action_hash(self) -> str:
        return self.numeric_state.packed_action_hash

    @property
    def parent_mask(self) -> HardMask:
        return self.numeric_state.parent_mask

    @property
    def parent_mask_hash(self) -> str:
        return self.numeric_state.parent_mask_hash

    @property
    def resulting_mask(self) -> HardMask:
        return self.numeric_state.resulting_mask

    @property
    def resulting_mask_hash(self) -> str:
        return self.numeric_state.resulting_mask_hash

    @property
    def removed_routes(self) -> V1RemovedRouteSetIdentity:
        return self.numeric_state.removed_routes

    @property
    def removed_count(self) -> int:
        return self.numeric_state.removed_count

    @property
    def removed_route_index_hash(self) -> str:
        return self.numeric_state.removed_route_index_hash

    @property
    def compact_removed_indices_available(self) -> bool:
        return False

    @property
    def resource_gain(self) -> float:
        return self.numeric_state.registered_resource

    @property
    def resource_receipt_hash(self) -> str:
        return self.numeric_state.resource_receipt_hash

    @property
    def fqr_interval_hash(self) -> str:
        return self.numeric_state.fqr_interval_hash

    @property
    def branch_family_authorization_hash(self) -> str:
        return self.numeric_state.branch_family_authorization_hash

    @property
    def branch_family_entry_hash(self) -> str:
        return self.numeric_state.branch_family_entry_hash

    @property
    def exact_intervention_damage_evaluated(self) -> bool:
        return False

    def resolve_removed_routes(self, parent_mask: HardMask):
        if type(parent_mask) is not HardMask or _fresh_mask_hash(
            parent_mask, "F/Q/R proposal parent"
        ) != self.parent_mask_hash:
            raise ValueError("F/Q/R proposal parent differs from its bound identity")
        resolved = self.resulting_mask.removed_index_set_from(parent_mask)
        if (
            resolved.count != self.removed_count
            or resolved.index_hash != self.removed_route_index_hash
        ):
            raise ValueError("resolved F/Q/R delta differs from its identity")
        return resolved

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.numeric_state.assert_integrity()
        self.__post_init__()
        if expected_hash is not None and self.candidate_id != expected_hash:
            raise ValueError("F/Q/R proposal hash does not match")
        return self.candidate_id

    def assert_can_certify_decision(self) -> None:
        raise PermissionError("F/Q/R proposals cannot certify a pruning decision")

    def assert_can_commit_mask(self) -> None:
        raise PermissionError("F/Q/R proposals cannot commit a hard mask")


def _issue_proposal(
    *,
    state: V1FQRActionNumericState | V1FQRActionNumericStateV2,
    domain_hash: str,
    certification_scope_hash: str,
) -> V1FQRCandidateProposal:
    provisional = {
        "numeric_state": state,
        "domain_hash": domain_hash,
        "certification_scope_hash": certification_scope_hash,
        "proposal_only": True,
        "proposal_witness_only": True,
        "fresh_relinearization_required": True,
        "paired_bce_oracle_required": True,
        "candidate_freeze_authority": False,
        "bce_oracle_authority": False,
        "irreversible_decision_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
        "execution_authority": False,
    }
    shell = object.__new__(V1FQRCandidateProposal)
    for name, value in provisional.items():
        object.__setattr__(shell, name, value)
    object.__setattr__(
        shell,
        "schema",
        (
            FQR_CANDIDATE_PROPOSAL_V2_SCHEMA
            if type(state) is V1FQRActionNumericStateV2
            else FQR_CANDIDATE_PROPOSAL_SCHEMA
        ),
    )
    receipt_hash = content_hash(
        shell._evaluation_payload(),
        prefix="partition-free-v1-fqr-candidate-evaluation-",
    )
    candidate_id = content_hash(
        {
            "evaluation_receipt_hash": receipt_hash,
            "resulting_mask_hash": state.resulting_mask_hash,
            "removed_route_index_hash": state.removed_route_index_hash,
            "fqr_interval_hash": state.fqr_interval_hash,
            "branch_family_entry_hash": state.branch_family_entry_hash,
        },
        prefix="partition-free-v1-fqr-candidate-",
    )
    return V1FQRCandidateProposal(
        _issuer_token=_PROPOSAL_TOKEN,
        **provisional,
        evaluation_receipt_hash=receipt_hash,
        candidate_id=candidate_id,
    )


@dataclass(frozen=True, slots=True, init=False)
class V1FQRCandidateCompositionResult:
    """Exact compact materialization of the predeclared cross-branch queue."""

    action_numeric_states: tuple[V1FQRActionNumericState, ...]
    proposals: tuple[V1FQRCandidateProposal, ...]
    domain_hash: str
    certification_scope_hash: str
    branch_family_authorization_hash: str
    prefrozen_family_hash: str
    joint_fqr_family_registration_hash: str
    source_branch_evidence_hashes: tuple[tuple[str, str], ...]
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    queue_entry_hashes: tuple[str, ...]
    queue_branch_endpoint_coordinates: tuple[tuple[str, int], ...]
    queue_packed_action_hashes: tuple[str, ...]
    queue_resulting_mask_hashes: tuple[str, ...]
    queue_fqr_interval_hashes: tuple[str, ...]
    queue_resource_receipt_hashes: tuple[str, ...]
    validated_action_count: int
    proposal_count: int
    composer_id: str
    selection_receipt_hash: str
    complete_authorized_queue_materialized_exactly: bool
    duplicate_physical_actions_quotiented_upstream: bool
    resource_descending_order_verified: bool
    filtering_performed: bool
    reordering_performed: bool
    fqr_monotonicity_assumed: bool
    claims_hard_global_optimum: bool
    proposal_only: bool
    proposal_witness_only: bool
    fresh_relinearization_required: bool
    paired_bce_oracle_required: bool
    candidate_freeze_authority: bool
    bce_oracle_authority: bool
    irreversible_decision_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    schema: str = FQR_CANDIDATE_COMPOSITION_SCHEMA
    _result_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _COMPOSITION_TOKEN:
            raise TypeError("V1FQRCandidateCompositionResult is issuer-only")
        expected = tuple(
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        )
        if set(values) != set(expected):
            raise TypeError("F/Q/R composition fields differ from their schema")
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", FQR_CANDIDATE_COMPOSITION_SCHEMA)
        object.__setattr__(self, "_result_hash", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        states = tuple(self.action_numeric_states)
        proposals = tuple(self.proposals)
        if not states or any(
            type(value)
            not in {V1FQRActionNumericState, V1FQRActionNumericStateV2}
            for value in states
        ):
            raise TypeError("F/Q/R composition requires issued numeric states")
        if len(states) != len(proposals) or any(
            type(value) is not V1FQRCandidateProposal for value in proposals
        ):
            raise TypeError("F/Q/R composition has an incomplete proposal axis")
        for state, proposal in zip(states, proposals, strict=True):
            state.assert_integrity()
            proposal.assert_integrity()
            if proposal.numeric_state is not state:
                raise ValueError("F/Q/R proposal recontextualized its numeric state")
        first = states[0]
        if any(
            value.branch_family_authorization_hash != first.branch_family_authorization_hash
            or value.prefrozen_family_hash != first.prefrozen_family_hash
            or value.joint_fqr_family_registration_hash != first.joint_fqr_family_registration_hash
            or value.source_branch_evidence_hashes != first.source_branch_evidence_hashes
            or getattr(value, "source_branch_negative_evidence_hashes", ())
            != getattr(first, "source_branch_negative_evidence_hashes", ())
            or value.parent_mask_hash != first.parent_mask_hash
            or value.route_axis_hash != first.route_axis_hash
            or value.route_count != first.route_count
            for value in states
        ):
            raise ValueError("F/Q/R composition states do not share one source coordinate")
        expected_axes = (
            tuple(value.branch_family_entry_hash for value in states),
            tuple((value.representative_branch, value.representative_endpoint_index) for value in states),
            tuple(value.packed_action_hash for value in states),
            tuple(value.resulting_mask_hash for value in states),
            tuple(value.fqr_interval_hash for value in states),
            tuple(value.resource_receipt_hash for value in states),
        )
        observed_axes = (
            self.queue_entry_hashes,
            self.queue_branch_endpoint_coordinates,
            self.queue_packed_action_hashes,
            self.queue_resulting_mask_hashes,
            self.queue_fqr_interval_hashes,
            self.queue_resource_receipt_hashes,
        )
        if observed_axes != expected_axes:
            raise ValueError("F/Q/R composition changed its authorized queue axis")
        if any(
            right.registered_resource > left.registered_resource
            for left, right in zip(states, states[1:], strict=False)
        ):
            raise ValueError("F/Q/R composition is not resource descending")
        shared = (
            (self.branch_family_authorization_hash, first.branch_family_authorization_hash),
            (self.prefrozen_family_hash, first.prefrozen_family_hash),
            (self.joint_fqr_family_registration_hash, first.joint_fqr_family_registration_hash),
            (self.source_branch_evidence_hashes, first.source_branch_evidence_hashes),
            (
                getattr(self, "source_branch_negative_evidence_hashes", ()),
                getattr(first, "source_branch_negative_evidence_hashes", ()),
            ),
            (self.parent_mask_hash, first.parent_mask_hash),
            (self.route_axis_hash, first.route_axis_hash),
            (self.route_count, first.route_count),
        )
        if any(observed != expected for observed, expected in shared):
            raise ValueError("F/Q/R composition changed a source identity")
        if self.domain_hash != _domain_hash(states):
            raise ValueError("F/Q/R compact domain hash is not reproducible")
        if self.certification_scope_hash != _certification_scope_hash(first):
            raise ValueError("F/Q/R certification-scope hash is not reproducible")
        if self.validated_action_count != len(states) or self.proposal_count != len(proposals):
            raise ValueError("F/Q/R composition filtered an authorized action")
        if self.composer_id != FQR_RESOURCE_DESCENDING_COMPOSER_ID:
            raise ValueError("unsupported F/Q/R composition policy")
        firewall = (
            self.complete_authorized_queue_materialized_exactly,
            self.duplicate_physical_actions_quotiented_upstream,
            self.resource_descending_order_verified,
            not self.filtering_performed,
            not self.reordering_performed,
            not self.fqr_monotonicity_assumed,
            not self.claims_hard_global_optimum,
            self.proposal_only,
            self.proposal_witness_only,
            self.fresh_relinearization_required,
            self.paired_bce_oracle_required,
            not self.candidate_freeze_authority,
            not self.bce_oracle_authority,
            not self.irreversible_decision_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            not self.execution_authority,
        )
        if any(type(flag) is not bool or not flag for flag in firewall):
            raise ValueError("F/Q/R composition crossed its proposal-only authority firewall")
        evidence, negative_evidence = _canonical_branch_evidence_hashes(
            positive=self.source_branch_evidence_hashes,
            negative=getattr(self, "source_branch_negative_evidence_hashes", ()),
        )
        if evidence != first.source_branch_evidence_hashes or (
            negative_evidence
            != getattr(first, "source_branch_negative_evidence_hashes", ())
        ):
            raise ValueError("F/Q/R composition branch provenance is invalid")
        if {
            alias.branch.value
            for state in states
            for alias in state.source_aliases
        } != {branch for branch, _value in evidence}:
            raise ValueError(
                "F/Q/R composition does not cover every positive evidence branch"
            )
        expected_schema = (
            FQR_CANDIDATE_COMPOSITION_V2_SCHEMA
            if negative_evidence
            else FQR_CANDIDATE_COMPOSITION_SCHEMA
        )
        if self.schema != expected_schema:
            raise ValueError("unsupported F/Q/R composition schema")
        expected_selection = content_hash(
            self._selection_payload(),
            prefix="partition-free-v1-fqr-candidate-selection-",
        )
        if self.selection_receipt_hash != expected_selection:
            raise ValueError("F/Q/R selection receipt is not reproducible")
        object.__setattr__(self, "action_numeric_states", states)
        object.__setattr__(self, "proposals", proposals)
        self.assert_integrity()

    @property
    def constraints_hash(self) -> str:
        return self.certification_scope_hash

    @property
    def ranked_proposals(self) -> tuple[V1FQRCandidateProposal, ...]:
        return self.proposals

    @property
    def selected_proposal(self) -> V1FQRCandidateProposal:
        return self.proposals[0]

    @property
    def result_hash(self) -> str:
        return self.assert_integrity()

    def _selection_payload(self) -> dict[str, object]:
        return {
            "schema": (
                "partition_free_v1_fqr_candidate_selection_v2"
                if getattr(self, "source_branch_negative_evidence_hashes", ())
                else "partition_free_v1_fqr_candidate_selection_v1"
            ),
            "composer_id": self.composer_id,
            "domain_hash": self.domain_hash,
            "certification_scope_hash": self.certification_scope_hash,
            "branch_family_authorization_hash": self.branch_family_authorization_hash,
            "queue_entry_hashes": self.queue_entry_hashes,
            "queue_candidate_ids": tuple(value.candidate_id for value in self.proposals),
            "queue_branch_endpoint_coordinates": self.queue_branch_endpoint_coordinates,
            "queue_packed_action_hashes": self.queue_packed_action_hashes,
            "validated_action_count": self.validated_action_count,
            "proposal_count": self.proposal_count,
            "complete_authorized_queue_materialized_exactly": True,
            "duplicate_physical_actions_quotiented_upstream": True,
            "resource_descending_order_verified": True,
            "filtering_performed": False,
            "reordering_performed": False,
            "fqr_monotonicity_assumed": False,
            "claims_hard_global_optimum": False,
            "proposal_only": True,
            "paired_bce_oracle_required": True,
        }

    def _payload(self) -> dict[str, object]:
        payload = {
            "schema": self.schema,
            "action_numeric_state_hashes": tuple(value.state_hash for value in self.action_numeric_states),
            "candidate_ids": tuple(value.candidate_id for value in self.proposals),
            "domain_hash": self.domain_hash,
            "certification_scope_hash": self.certification_scope_hash,
            "branch_family_authorization_hash": self.branch_family_authorization_hash,
            "prefrozen_family_hash": self.prefrozen_family_hash,
            "joint_fqr_family_registration_hash": self.joint_fqr_family_registration_hash,
            "source_branch_evidence_hashes": self.source_branch_evidence_hashes,
            "parent_mask_hash": self.parent_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "queue_entry_hashes": self.queue_entry_hashes,
            "queue_branch_endpoint_coordinates": self.queue_branch_endpoint_coordinates,
            "queue_packed_action_hashes": self.queue_packed_action_hashes,
            "queue_resulting_mask_hashes": self.queue_resulting_mask_hashes,
            "queue_fqr_interval_hashes": self.queue_fqr_interval_hashes,
            "queue_resource_receipt_hashes": self.queue_resource_receipt_hashes,
            "validated_action_count": self.validated_action_count,
            "proposal_count": self.proposal_count,
            "composer_id": self.composer_id,
            "selection_receipt_hash": self.selection_receipt_hash,
            "complete_authorized_queue_materialized_exactly": self.complete_authorized_queue_materialized_exactly,
            "duplicate_physical_actions_quotiented_upstream": self.duplicate_physical_actions_quotiented_upstream,
            "resource_descending_order_verified": self.resource_descending_order_verified,
            "filtering_performed": self.filtering_performed,
            "reordering_performed": self.reordering_performed,
            "fqr_monotonicity_assumed": self.fqr_monotonicity_assumed,
            "claims_hard_global_optimum": self.claims_hard_global_optimum,
            "proposal_only": self.proposal_only,
            "proposal_witness_only": self.proposal_witness_only,
            "fresh_relinearization_required": self.fresh_relinearization_required,
            "paired_bce_oracle_required": self.paired_bce_oracle_required,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "irreversible_decision_authority": self.irreversible_decision_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "execution_authority": self.execution_authority,
        }
        negative_evidence = getattr(
            self, "source_branch_negative_evidence_hashes", ()
        )
        if negative_evidence:
            payload["source_branch_negative_evidence_hashes"] = (
                negative_evidence
            )
        return payload

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for state, proposal in zip(self.action_numeric_states, self.proposals, strict=True):
            state.assert_integrity()
            proposal.assert_integrity()
        observed = content_hash(self._payload(), prefix="partition-free-v1-fqr-composition-")
        if self._result_hash is not None and observed != self._result_hash:
            raise ValueError("F/Q/R composition was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("F/Q/R composition hash does not match")
        object.__setattr__(self, "_result_hash", observed)
        return observed


@dataclass(frozen=True, slots=True, init=False)
class V1FQRCandidateCompositionResultV2(V1FQRCandidateCompositionResult):
    """Mixed positive/stable-negative compact queue composition."""

    source_branch_negative_evidence_hashes: tuple[tuple[str, str], ...] = ()

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _COMPOSITION_TOKEN:
            raise TypeError("V1FQRCandidateCompositionResultV2 is issuer-only")
        base_expected = tuple(
            name
            for name, declaration in V1FQRCandidateCompositionResult.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        )
        expected = (*base_expected, "source_branch_negative_evidence_hashes")
        if set(values) != set(expected):
            missing = sorted(set(expected).difference(values))
            extra = sorted(set(values).difference(expected))
            raise TypeError(
                "F/Q/R v2 composition fields differ: "
                f"missing={missing}, extra={extra}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", FQR_CANDIDATE_COMPOSITION_V2_SCHEMA)
        object.__setattr__(self, "_result_hash", None)
        self.__post_init__()


def _domain_hash(
    states: Sequence[V1FQRActionNumericState | V1FQRActionNumericStateV2],
) -> str:
    return content_hash(
        {
            "schema": (
                "partition_free_v1_fqr_compact_physical_domain_v2"
                if type(states[0]) is V1FQRActionNumericStateV2
                else "partition_free_v1_fqr_compact_physical_domain_v1"
            ),
            "branch_family_authorization_hash": states[0].branch_family_authorization_hash,
            "parent_mask_hash": states[0].parent_mask_hash,
            "route_axis_hash": states[0].route_axis_hash,
            "route_count": states[0].route_count,
            "queue_entry_hashes": tuple(value.branch_family_entry_hash for value in states),
            "queue_packed_action_hashes": tuple(value.packed_action_hash for value in states),
            "queue_resulting_mask_hashes": tuple(value.resulting_mask_hash for value in states),
            "duplicate_physical_actions_quotiented_upstream": True,
            "filtering_performed": False,
            "reordering_performed": False,
        },
        prefix="partition-free-v1-fqr-compact-domain-",
    )


def _certification_scope_hash(
    state: V1FQRActionNumericState | V1FQRActionNumericStateV2,
) -> str:
    negative_evidence = getattr(
        state, "source_branch_negative_evidence_hashes", ()
    )
    payload = {
            "schema": (
                "partition_free_v1_fqr_certification_scope_v2"
                if negative_evidence
                else "partition_free_v1_fqr_certification_scope_v1"
            ),
            "branch_family_authorization_hash": state.branch_family_authorization_hash,
            "prefrozen_family_hash": state.prefrozen_family_hash,
            "joint_fqr_family_registration_hash": state.joint_fqr_family_registration_hash,
            "source_branch_evidence_hashes": state.source_branch_evidence_hashes,
            "independently_certified_fqr_bounds": True,
            "paired_bce_oracle_required": True,
        }
    if negative_evidence:
        payload["source_branch_negative_evidence_hashes"] = (
            negative_evidence
        )
    return content_hash(
        payload, prefix="partition-free-v1-fqr-certification-scope-"
    )


@dataclass(frozen=True, slots=True)
class V1FQRAcquisitionGenerationReceipt:
    campaign_contract_hash: str
    campaign_binding_hash: str
    registration_hash: str
    generation_index: int
    current_mask_hash: str
    route_axis_hash: str
    acquisition_hash: str
    invalidation_chart_hash: str
    composition_result_hash: str
    compact_domain_hash: str
    certification_scope_hash: str
    branch_family_authorization_hash: str
    prefrozen_family_hash: str
    joint_fqr_family_registration_hash: str
    source_branch_evidence_hashes: tuple[tuple[str, str], ...]
    ordered_candidate_ids: tuple[str, ...]
    ordered_entry_hashes: tuple[str, ...]
    ordered_branch_endpoint_coordinates: tuple[tuple[str, int], ...]
    ordered_packed_action_hashes: tuple[str, ...]
    ordered_fqr_interval_hashes: tuple[str, ...]
    refresh_reason: str
    logical_sequence: int
    proposal_only: bool = True
    complete_queue_frozen_before_K_O: bool = True
    paired_bce_oracle_required: bool = True
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    candidate_freeze_authority: bool = False
    irreversible_decision_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    claim_execution_certified: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = FQR_ACQUISITION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash", "campaign_binding_hash", "registration_hash",
            "current_mask_hash", "route_axis_hash", "acquisition_hash",
            "invalidation_chart_hash", "composition_result_hash", "compact_domain_hash",
            "certification_scope_hash", "branch_family_authorization_hash",
            "prefrozen_family_hash", "joint_fqr_family_registration_hash", "refresh_reason",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        object.__setattr__(self, "generation_index", _nonnegative_int(self.generation_index, "generation_index"))
        object.__setattr__(self, "logical_sequence", _nonnegative_int(self.logical_sequence, "logical_sequence"))
        candidate_ids = tuple(_identity(value, "ordered candidate id") for value in self.ordered_candidate_ids)
        entry_hashes = tuple(_identity(value, "ordered entry hash") for value in self.ordered_entry_hashes)
        packed_hashes = tuple(_identity(value, "ordered packed action hash") for value in self.ordered_packed_action_hashes)
        interval_hashes = tuple(_identity(value, "ordered F/Q/R interval hash") for value in self.ordered_fqr_interval_hashes)
        coordinates = tuple(
            (_branch_value(branch, "ordered branch"), _positive_int(endpoint, "ordered endpoint"))
            for branch, endpoint in self.ordered_branch_endpoint_coordinates
        )
        size = len(candidate_ids)
        if size == 0 or any(len(axis) != size for axis in (entry_hashes, packed_hashes, interval_hashes, coordinates)):
            raise ValueError("F/Q/R receipt queue axes do not align")
        if any(len(set(axis)) != len(axis) for axis in (candidate_ids, entry_hashes, packed_hashes)):
            raise ValueError("F/Q/R receipt physical queue contains duplicates")
        evidence, negative_evidence = _canonical_branch_evidence_hashes(
            positive=self.source_branch_evidence_hashes,
            negative=getattr(self, "source_branch_negative_evidence_hashes", ()),
        )
        firewall = (
            self.proposal_only,
            self.complete_queue_frozen_before_K_O,
            self.paired_bce_oracle_required,
            not self.calibration_K_open_authority,
            not self.paired_oracle_O_open_authority,
            not self.candidate_freeze_authority,
            not self.irreversible_decision_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            not self.claim_execution_certified,
        )
        if any(type(flag) is not bool or not flag for flag in firewall):
            raise ValueError("F/Q/R acquisition receipt crossed its authority firewall")
        expected_schema = (
            FQR_ACQUISITION_RECEIPT_V2_SCHEMA
            if negative_evidence
            else FQR_ACQUISITION_RECEIPT_SCHEMA
        )
        if self.schema != expected_schema:
            raise ValueError("unsupported F/Q/R acquisition-receipt schema")
        object.__setattr__(self, "source_branch_evidence_hashes", evidence)
        if hasattr(self, "source_branch_negative_evidence_hashes"):
            object.__setattr__(
                self,
                "source_branch_negative_evidence_hashes",
                negative_evidence,
            )
        object.__setattr__(self, "ordered_candidate_ids", candidate_ids)
        object.__setattr__(self, "ordered_entry_hashes", entry_hashes)
        object.__setattr__(self, "ordered_branch_endpoint_coordinates", coordinates)
        object.__setattr__(self, "ordered_packed_action_hashes", packed_hashes)
        object.__setattr__(self, "ordered_fqr_interval_hashes", interval_hashes)
        object.__setattr__(self, "metadata", _freeze(self.metadata, "F/Q/R receipt metadata"))

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self._payload(), prefix="partition-free-v1-fqr-acquisition-generation-"
        )

    def _payload(self) -> dict[str, object]:
        payload = {
            name: getattr(self, name)
            for name in (
                "campaign_contract_hash",
                "campaign_binding_hash",
                "registration_hash",
                "generation_index",
                "current_mask_hash",
                "route_axis_hash",
                "acquisition_hash",
                "invalidation_chart_hash",
                "composition_result_hash",
                "compact_domain_hash",
                "certification_scope_hash",
                "branch_family_authorization_hash",
                "prefrozen_family_hash",
                "joint_fqr_family_registration_hash",
                "source_branch_evidence_hashes",
                "ordered_candidate_ids",
                "ordered_entry_hashes",
                "ordered_branch_endpoint_coordinates",
                "ordered_packed_action_hashes",
                "ordered_fqr_interval_hashes",
                "refresh_reason",
                "logical_sequence",
                "proposal_only",
                "complete_queue_frozen_before_K_O",
                "paired_bce_oracle_required",
                "calibration_K_open_authority",
                "paired_oracle_O_open_authority",
                "candidate_freeze_authority",
                "irreversible_decision_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_authority",
                "claim_execution_certified",
                "metadata",
                "schema",
            )
        }
        negative_evidence = getattr(
            self, "source_branch_negative_evidence_hashes", ()
        )
        if negative_evidence:
            payload["source_branch_negative_evidence_hashes"] = (
                negative_evidence
            )
        return payload


@dataclass(frozen=True, slots=True)
class V1FQRAcquisitionGenerationReceiptV2(V1FQRAcquisitionGenerationReceipt):
    """Mixed branch acquisition receipt retaining stable-negative evidence."""

    source_branch_negative_evidence_hashes: tuple[tuple[str, str], ...] = ()


def _invalidation_chart_hash(
    *,
    generation_index: int,
    current_mask: HardMask,
    composition: V1FQRCandidateCompositionResult | V1FQRCandidateCompositionResultV2,
) -> str:
    negative_evidence = getattr(
        composition, "source_branch_negative_evidence_hashes", ()
    )
    payload = {
            "schema": (
                "partition_free_v1_fqr_invalidation_chart_v2"
                if negative_evidence
                else "partition_free_v1_fqr_invalidation_chart_v1"
            ),
            "generation_index": generation_index,
            "current_mask_hash": current_mask.mask_hash,
            "route_axis_hash": current_mask.route_axis_hash,
            "branch_family_authorization_hash": composition.branch_family_authorization_hash,
            "prefrozen_family_hash": composition.prefrozen_family_hash,
            "joint_fqr_family_registration_hash": composition.joint_fqr_family_registration_hash,
            "source_branch_evidence_hashes": composition.source_branch_evidence_hashes,
        }
    if negative_evidence:
        payload["source_branch_negative_evidence_hashes"] = (
            negative_evidence
        )
    return content_hash(
        payload, prefix="partition-free-v1-fqr-invalidation-chart-"
    )


def _acquisition_hash(
    *,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    lineage: tuple[str, ...],
    transitions: tuple[str, ...],
    invalidation_chart_hash: str,
    composition_result_hash: str,
    metadata: Mapping[str, Any],
    schema: str,
) -> str:
    return content_hash(
        {
            "schema": schema,
            "campaign_binding_hash": binding.binding_hash,
            "generation_index": generation_index,
            "current_mask_hash": current_mask.mask_hash,
            "committed_mask_lineage_hashes": lineage,
            "committed_mask_transition_receipt_hashes": transitions,
            "invalidation_chart_hash": invalidation_chart_hash,
            "composition_result_hash": composition_result_hash,
            "metadata": metadata,
        },
        prefix="partition-free-v1-fqr-authorized-acquisition-",
    )


@dataclass(frozen=True, slots=True)
class V1FQRAuthorizedAcquisition:
    binding: V1OptionACampaignBinding
    generation_index: int
    current_mask: HardMask
    committed_mask_lineage_hashes: tuple[str, ...]
    committed_mask_transition_receipt_hashes: tuple[str, ...]
    composition: V1FQRCandidateCompositionResult | V1FQRCandidateCompositionResultV2
    receipt: V1FQRAcquisitionGenerationReceipt | V1FQRAcquisitionGenerationReceiptV2
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = FQR_AUTHORIZED_ACQUISITION_SCHEMA

    def __post_init__(self) -> None:
        if type(self.binding) is not V1OptionACampaignBinding:
            raise TypeError("F/Q/R acquisition requires an exact campaign binding")
        if type(self.current_mask) is not HardMask:
            raise TypeError("F/Q/R acquisition requires an exact current mask")
        if type(self.composition) not in {
            V1FQRCandidateCompositionResult,
            V1FQRCandidateCompositionResultV2,
        }:
            raise TypeError("F/Q/R acquisition requires its compact composition")
        if type(self.receipt) not in {
            V1FQRAcquisitionGenerationReceipt,
            V1FQRAcquisitionGenerationReceiptV2,
        }:
            raise TypeError("F/Q/R acquisition requires its exact receipt")
        self.composition.assert_integrity()
        generation = _nonnegative_int(self.generation_index, "generation_index")
        lineage = tuple(_identity(value, "mask lineage hash") for value in self.committed_mask_lineage_hashes)
        transitions = tuple(_identity(value, "mask transition hash") for value in self.committed_mask_transition_receipt_hashes)
        if (
            not lineage
            or lineage[-1] != _fresh_mask_hash(self.current_mask, "F/Q/R current mask")
            or len(transitions) != len(lineage) - 1
            or len(set(lineage)) != len(lineage)
            or len(set(transitions)) != len(transitions)
        ):
            raise ValueError("F/Q/R acquisition has incomplete exact mask ancestry")
        composition = self.composition
        receipt = self.receipt
        contract = self.binding.contract
        if (
            contract.route_axis_hash != self.current_mask.route_axis_hash
            or composition.parent_mask_hash != self.current_mask.mask_hash
            or composition.route_axis_hash != self.current_mask.route_axis_hash
            or composition.route_count != self.current_mask.route_count
            or composition.action_numeric_states[0].generation_index != generation
        ):
            raise ValueError("F/Q/R acquisition differs from the current generation or axis")
        if (
            receipt.campaign_contract_hash != contract.campaign_contract_hash
            or receipt.campaign_binding_hash != self.binding.binding_hash
            or receipt.registration_hash != self.binding.registration.registration_hash
            or receipt.generation_index != generation
            or receipt.current_mask_hash != self.current_mask.mask_hash
            or receipt.route_axis_hash != self.current_mask.route_axis_hash
            or receipt.composition_result_hash != composition.result_hash
            or receipt.compact_domain_hash != composition.domain_hash
            or receipt.certification_scope_hash != composition.certification_scope_hash
            or receipt.branch_family_authorization_hash != composition.branch_family_authorization_hash
            or receipt.prefrozen_family_hash != composition.prefrozen_family_hash
            or receipt.joint_fqr_family_registration_hash != composition.joint_fqr_family_registration_hash
            or receipt.source_branch_evidence_hashes != composition.source_branch_evidence_hashes
            or getattr(receipt, "source_branch_negative_evidence_hashes", ())
            != getattr(composition, "source_branch_negative_evidence_hashes", ())
            or receipt.ordered_candidate_ids != tuple(value.candidate_id for value in composition.proposals)
            or receipt.ordered_entry_hashes != composition.queue_entry_hashes
            or receipt.ordered_branch_endpoint_coordinates != composition.queue_branch_endpoint_coordinates
            or receipt.ordered_packed_action_hashes != composition.queue_packed_action_hashes
            or receipt.ordered_fqr_interval_hashes != composition.queue_fqr_interval_hashes
            or receipt.invalidation_chart_hash != self.invalidation_chart_hash
            or receipt.acquisition_hash != self.acquisition_hash
        ):
            raise ValueError("F/Q/R acquisition receipt differs from its compact sources")
        expected_schema = (
            FQR_AUTHORIZED_ACQUISITION_V2_SCHEMA
            if getattr(composition, "source_branch_negative_evidence_hashes", ())
            else FQR_AUTHORIZED_ACQUISITION_SCHEMA
        )
        if self.schema != expected_schema:
            raise ValueError("unsupported F/Q/R acquisition schema")
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "committed_mask_lineage_hashes", lineage)
        object.__setattr__(self, "committed_mask_transition_receipt_hashes", transitions)
        object.__setattr__(self, "metadata", _freeze(self.metadata, "F/Q/R acquisition metadata"))

    @property
    def invalidation_chart_hash(self) -> str:
        return _invalidation_chart_hash(
            generation_index=self.generation_index,
            current_mask=self.current_mask,
            composition=self.composition,
        )

    @property
    def acquisition_hash(self) -> str:
        return _acquisition_hash(
            binding=self.binding,
            generation_index=self.generation_index,
            current_mask=self.current_mask,
            lineage=self.committed_mask_lineage_hashes,
            transitions=self.committed_mask_transition_receipt_hashes,
            invalidation_chart_hash=self.invalidation_chart_hash,
            composition_result_hash=self.composition.result_hash,
            metadata=self.metadata,
            schema=self.schema,
        )

    @property
    def queue(self) -> tuple[V1FQRCandidateProposal, ...]:
        return self.composition.proposals

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def mask_commit_authority(self) -> bool:
        return False


def issue_v1_fqr_authorized_acquisition(
    *,
    authorization: V1CompactFQRBranchFamilyAuthorization,
    binding: V1OptionACampaignBinding,
    generation_index: int,
    current_mask: HardMask,
    committed_mask_lineage_hashes: Sequence[str],
    committed_mask_transition_receipt_hashes: Sequence[str],
    refresh_reason: str = "fresh_current_mask_canonical_fqr_relinearization",
    logical_sequence: int = 0,
    metadata: Mapping[str, Any] | None = None,
) -> V1FQRAuthorizedAcquisition:
    """Detach one complete compact F/Q/R queue for controller execution."""

    if type(authorization) is not V1CompactFQRBranchFamilyAuthorization:
        raise TypeError("F/Q/R acquisition requires exact compact branch-family authority")
    if type(binding) is not V1OptionACampaignBinding:
        raise TypeError("F/Q/R acquisition requires an exact campaign binding")
    if type(current_mask) is not HardMask:
        raise TypeError("F/Q/R acquisition requires an exact current mask")
    authorization.assert_integrity()
    generation = _nonnegative_int(generation_index, "generation_index")
    sequence = _nonnegative_int(logical_sequence, "logical_sequence")
    reason = _identity(refresh_reason, "refresh_reason")
    lineage = tuple(committed_mask_lineage_hashes)
    transitions = tuple(committed_mask_transition_receipt_hashes)
    current_hash = _fresh_mask_hash(current_mask, "F/Q/R current mask")
    if (
        not lineage
        or lineage[-1] != current_hash
        or len(transitions) != len(lineage) - 1
        or binding.contract.route_axis_hash != current_mask.route_axis_hash
        or authorization.current_mask_hash != current_hash
        or authorization.route_axis_hash != current_mask.route_axis_hash
        or authorization.route_count != current_mask.route_count
        or authorization.proposal_generation != generation
    ):
        raise ValueError("F/Q/R acquisition coordinates are stale")
    entries = tuple(authorization.queue_entries)
    if not entries:
        raise ValueError("F/Q/R acquisition authorization has no physical queue")
    authorization_hash = authorization.authorization_hash
    evidence, negative_evidence = _canonical_branch_evidence_hashes(
        positive=authorization.source_branch_evidence_hashes,
        negative=authorization.source_branch_negative_evidence_hashes,
    )
    mixed_schema = bool(negative_evidence)
    state_type = (
        V1FQRActionNumericStateV2
        if mixed_schema
        else V1FQRActionNumericState
    )
    states: list[V1FQRActionNumericState | V1FQRActionNumericStateV2] = []
    for queue_index, entry in enumerate(entries):
        entry.assert_integrity()
        packed = entry.packed_action
        if type(packed) is not V1PackedRemovalAction:
            raise TypeError("F/Q/R queue entry lost its packed physical action")
        child = packed.resulting_mask
        removed_count, removed_hash = child.removed_index_count_and_hash_from(current_mask)
        aliases = _canonical_source_aliases(entry.source_aliases)
        source_aliases_hash = content_hash(
            tuple(value.alias_hash for value in aliases),
            prefix="partition-free-v1-fqr-source-aliases-",
        )
        interval_values = {
            name: _finite_nonnegative(getattr(entry, name), name)
            for name in (
                "f_lower", "f_upper", "q_lower", "q_upper", "r_half_lower",
                "r_half_upper", "r_one_lower", "r_one_upper",
                "half_response_lower", "half_response_upper",
                "one_response_lower", "one_response_upper",
            )
        }
        state_shell = object.__new__(state_type)
        prevalues = {
            "queue_index": queue_index,
            "representative_branch": _branch_value(entry.representative_branch, "representative branch"),
            "representative_endpoint_index": entry.representative_endpoint_index,
            "representative_frozen_action_hash": entry.representative_frozen_action_hash,
            "source_aliases": aliases,
            "source_aliases_hash": source_aliases_hash,
            "parent_mask": current_mask,
            "packed_action": packed,
            "parent_mask_hash": current_hash,
            "resulting_mask_hash": _fresh_mask_hash(child, "F/Q/R authorized child"),
            "packed_action_hash": packed.action_hash,
            "route_axis_hash": current_mask.route_axis_hash,
            "route_count": current_mask.route_count,
            "removed_count": removed_count,
            "removed_route_index_hash": removed_hash,
            "registered_resource": entry.registered_resource,
            "resource_units": entry.resource_units,
            "resource_source_hash": entry.resource_source_hash,
            **interval_values,
            "branch_family_authorization_hash": authorization_hash,
            "branch_family_entry_hash": entry.entry_hash,
            "prefrozen_family_hash": authorization.prefrozen_family_hash,
            "joint_fqr_family_registration_hash": authorization.joint_fqr_family_registration_hash,
            "source_branch_evidence_hashes": evidence,
            "generation_index": generation,
            "independently_certified_fqr_bounds": True,
            "complete_queue_frozen_before_K_O": True,
            "proposal_only": True,
            "paired_bce_oracle_required": True,
            "calibration_K_open_authority": False,
            "paired_oracle_O_open_authority": False,
            "candidate_freeze_authority": False,
            "irreversible_decision_authority": False,
            "mask_commit_authority": False,
            "terminal_authority": False,
            "pareto_authority": False,
            "execution_authority": False,
        }
        if mixed_schema:
            prevalues["source_branch_negative_evidence_hashes"] = negative_evidence
        for name, value in prevalues.items():
            object.__setattr__(state_shell, name, value)
        object.__setattr__(
            state_shell,
            "schema",
            (
                FQR_ACTION_NUMERIC_STATE_V2_SCHEMA
                if mixed_schema
                else FQR_ACTION_NUMERIC_STATE_SCHEMA
            ),
        )
        interval_hash = content_hash(
            state_shell._interval_payload(values=interval_values),
            prefix="partition-free-v1-fqr-certified-interval-",
        )
        object.__setattr__(state_shell, "fqr_interval_hash", interval_hash)
        resource_hash = content_hash(
            state_shell._resource_payload(resource=float(entry.registered_resource)),
            prefix="partition-free-v1-fqr-resource-",
        )
        states.append(
            state_type(
                _issuer_token=_STATE_TOKEN,
                **prevalues,
                fqr_interval_hash=interval_hash,
                resource_receipt_hash=resource_hash,
            )
        )
    frozen_states = tuple(states)
    domain_hash = _domain_hash(frozen_states)
    certification_scope_hash = _certification_scope_hash(frozen_states[0])
    proposals = tuple(
        _issue_proposal(
            state=state,
            domain_hash=domain_hash,
            certification_scope_hash=certification_scope_hash,
        )
        for state in frozen_states
    )
    provisional = {
        "action_numeric_states": frozen_states,
        "proposals": proposals,
        "domain_hash": domain_hash,
        "certification_scope_hash": certification_scope_hash,
        "branch_family_authorization_hash": authorization_hash,
        "prefrozen_family_hash": authorization.prefrozen_family_hash,
        "joint_fqr_family_registration_hash": authorization.joint_fqr_family_registration_hash,
        "source_branch_evidence_hashes": evidence,
        "parent_mask_hash": current_hash,
        "route_axis_hash": current_mask.route_axis_hash,
        "route_count": current_mask.route_count,
        "queue_entry_hashes": tuple(value.branch_family_entry_hash for value in frozen_states),
        "queue_branch_endpoint_coordinates": tuple(
            (value.representative_branch, value.representative_endpoint_index)
            for value in frozen_states
        ),
        "queue_packed_action_hashes": tuple(value.packed_action_hash for value in frozen_states),
        "queue_resulting_mask_hashes": tuple(value.resulting_mask_hash for value in frozen_states),
        "queue_fqr_interval_hashes": tuple(value.fqr_interval_hash for value in frozen_states),
        "queue_resource_receipt_hashes": tuple(value.resource_receipt_hash for value in frozen_states),
        "validated_action_count": len(frozen_states),
        "proposal_count": len(proposals),
        "composer_id": FQR_RESOURCE_DESCENDING_COMPOSER_ID,
        "complete_authorized_queue_materialized_exactly": True,
        "duplicate_physical_actions_quotiented_upstream": True,
        "resource_descending_order_verified": True,
        "filtering_performed": False,
        "reordering_performed": False,
        "fqr_monotonicity_assumed": False,
        "claims_hard_global_optimum": False,
        "proposal_only": True,
        "proposal_witness_only": True,
        "fresh_relinearization_required": True,
        "paired_bce_oracle_required": True,
        "candidate_freeze_authority": False,
        "bce_oracle_authority": False,
        "irreversible_decision_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
        "execution_authority": False,
    }
    if mixed_schema:
        provisional["source_branch_negative_evidence_hashes"] = negative_evidence
    composition_type = (
        V1FQRCandidateCompositionResultV2
        if mixed_schema
        else V1FQRCandidateCompositionResult
    )
    shell = object.__new__(composition_type)
    for name, value in provisional.items():
        object.__setattr__(shell, name, value)
    object.__setattr__(
        shell,
        "schema",
        (
            FQR_CANDIDATE_COMPOSITION_V2_SCHEMA
            if mixed_schema
            else FQR_CANDIDATE_COMPOSITION_SCHEMA
        ),
    )
    selection_hash = content_hash(
        shell._selection_payload(),
        prefix="partition-free-v1-fqr-candidate-selection-",
    )
    composition = composition_type(
        _issuer_token=_COMPOSITION_TOKEN,
        **provisional,
        selection_receipt_hash=selection_hash,
    )
    required_metadata = {
        "refresh_reason": reason,
        "logical_sequence": sequence,
        "generation_index": generation,
        "authority_scope": FQR_RUNTIME_AUTHORITY_SCOPE,
        "candidate_composition_mode": "canonical_fqr_cross_branch_resource_descending_compact",
        "complete_queue_frozen_before_K_O": True,
        "full_compact_branch_authorization_embedded": False,
        "raw_C_records_embedded": False,
        "record_retaining_receipts_embedded": False,
        "full_route_numeric_state_embedded": False,
        "fqr_monotonicity_assumed": False,
        "proposal_only": True,
        "paired_bce_oracle_required": True,
        "calibration_K_open_authority": False,
        "paired_oracle_O_open_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
    }
    caller_metadata = dict(metadata or {})
    collisions = sorted(set(required_metadata).intersection(caller_metadata))
    if collisions:
        raise ValueError(f"caller metadata shadows reserved F/Q/R fields: {collisions}")
    core_metadata = _freeze({**caller_metadata, **required_metadata}, "F/Q/R acquisition metadata")
    invalidation_hash = _invalidation_chart_hash(
        generation_index=generation,
        current_mask=current_mask,
        composition=composition,
    )
    acquisition_hash = _acquisition_hash(
        binding=binding,
        generation_index=generation,
        current_mask=current_mask,
        lineage=lineage,
        transitions=transitions,
        invalidation_chart_hash=invalidation_hash,
        composition_result_hash=composition.result_hash,
        metadata=core_metadata,
        schema=(
            FQR_AUTHORIZED_ACQUISITION_V2_SCHEMA
            if mixed_schema
            else FQR_AUTHORIZED_ACQUISITION_SCHEMA
        ),
    )
    receipt_type = (
        V1FQRAcquisitionGenerationReceiptV2
        if mixed_schema
        else V1FQRAcquisitionGenerationReceipt
    )
    receipt_kwargs = {
        "campaign_contract_hash": binding.contract.campaign_contract_hash,
        "campaign_binding_hash": binding.binding_hash,
        "registration_hash": binding.registration.registration_hash,
        "generation_index": generation,
        "current_mask_hash": current_hash,
        "route_axis_hash": current_mask.route_axis_hash,
        "acquisition_hash": acquisition_hash,
        "invalidation_chart_hash": invalidation_hash,
        "composition_result_hash": composition.result_hash,
        "compact_domain_hash": composition.domain_hash,
        "certification_scope_hash": composition.certification_scope_hash,
        "branch_family_authorization_hash": composition.branch_family_authorization_hash,
        "prefrozen_family_hash": composition.prefrozen_family_hash,
        "joint_fqr_family_registration_hash": composition.joint_fqr_family_registration_hash,
        "source_branch_evidence_hashes": composition.source_branch_evidence_hashes,
        "ordered_candidate_ids": tuple(value.candidate_id for value in proposals),
        "ordered_entry_hashes": composition.queue_entry_hashes,
        "ordered_branch_endpoint_coordinates": composition.queue_branch_endpoint_coordinates,
        "ordered_packed_action_hashes": composition.queue_packed_action_hashes,
        "ordered_fqr_interval_hashes": composition.queue_fqr_interval_hashes,
        "refresh_reason": reason,
        "logical_sequence": sequence,
        "schema": (
            FQR_ACQUISITION_RECEIPT_V2_SCHEMA
            if mixed_schema
            else FQR_ACQUISITION_RECEIPT_SCHEMA
        ),
        "metadata": {
            "queue_size": len(proposals),
            "queue_resource_values": tuple(value.resource_gain for value in proposals),
            "queue_f_bounds": tuple((value.numeric_state.f_lower, value.numeric_state.f_upper) for value in proposals),
            "queue_q_bounds": tuple((value.numeric_state.q_lower, value.numeric_state.q_upper) for value in proposals),
            "queue_r_half_bounds": tuple((value.numeric_state.r_half_lower, value.numeric_state.r_half_upper) for value in proposals),
            "queue_r_one_bounds": tuple((value.numeric_state.r_one_lower, value.numeric_state.r_one_upper) for value in proposals),
            "queue_half_response_bounds": tuple((value.numeric_state.half_response_lower, value.numeric_state.half_response_upper) for value in proposals),
            "queue_one_response_bounds": tuple((value.numeric_state.one_response_lower, value.numeric_state.one_response_upper) for value in proposals),
            "authority_scope": FQR_RUNTIME_AUTHORITY_SCOPE,
        },
    }
    if mixed_schema:
        receipt_kwargs["source_branch_negative_evidence_hashes"] = (
            composition.source_branch_negative_evidence_hashes
        )
    receipt = receipt_type(
        **receipt_kwargs,
    )
    return V1FQRAuthorizedAcquisition(
        binding=binding,
        generation_index=generation,
        current_mask=current_mask,
        committed_mask_lineage_hashes=lineage,
        committed_mask_transition_receipt_hashes=transitions,
        composition=composition,
        receipt=receipt,
        metadata=core_metadata,
        schema=(
            FQR_AUTHORIZED_ACQUISITION_V2_SCHEMA
            if mixed_schema
            else FQR_AUTHORIZED_ACQUISITION_SCHEMA
        ),
    )


__all__ = [
    "FQR_ACQUISITION_RECEIPT_SCHEMA",
    "FQR_ACQUISITION_RECEIPT_V2_SCHEMA",
    "FQR_ACTION_NUMERIC_STATE_SCHEMA",
    "FQR_ACTION_NUMERIC_STATE_V2_SCHEMA",
    "FQR_AUTHORIZED_ACQUISITION_SCHEMA",
    "FQR_AUTHORIZED_ACQUISITION_V2_SCHEMA",
    "FQR_CANDIDATE_COMPOSITION_SCHEMA",
    "FQR_CANDIDATE_COMPOSITION_V2_SCHEMA",
    "FQR_CANDIDATE_PROPOSAL_SCHEMA",
    "FQR_CANDIDATE_PROPOSAL_V2_SCHEMA",
    "FQR_RESOURCE_DESCENDING_COMPOSER_ID",
    "FQR_RUNTIME_AUTHORITY_SCOPE",
    "V1FQRAcquisitionGenerationReceipt",
    "V1FQRAcquisitionGenerationReceiptV2",
    "V1FQRActionNumericState",
    "V1FQRActionNumericStateV2",
    "V1FQRAuthorizedAcquisition",
    "V1FQRCandidateCompositionResult",
    "V1FQRCandidateCompositionResultV2",
    "V1FQRCandidateProposal",
    "issue_v1_fqr_authorized_acquisition",
]
