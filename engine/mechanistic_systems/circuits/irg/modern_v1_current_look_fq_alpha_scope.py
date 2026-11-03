"""Minimal current-look coordinate scope for independent F/Q alpha.

The sampled discovery backend has already registered a finite score-look axis
before observing corrected-attribution rows.  Physical-action C needs the F/Q
share assigned to the *current* registered look, but it does not need (and
must not inherit) a response-exactification or sampled-composition action
scope merely to name that alpha coordinate.

This module therefore extracts only the coordinate identity of the current D
look.  Issuance verifies the complete content hashes of the existing score
registration and look receipt, checks both against the exact proposal design,
and then discards their action/order/outcome payloads.  The resulting receipt
has authority only to select the already-preregistered F/Q alpha coordinate;
it cannot execute C, certify F/Q/R, open the oracle, or commit a mask.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Final

from .contracts import content_hash
from .modern_v1_proposal_statistics import V1GroupSequentialSerflingDesign


CURRENT_LOOK_FQ_ALPHA_COORDINATE_SCOPE_SCHEMA: Final = (
    "partition_free_v1_current_look_fq_alpha_coordinate_scope_v1"
)
SAMPLED_SCORE_ORDER_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_alexnet_sampled_score_order_registration_v3"
)
SAMPLED_SCORE_ORDER_LOOK_SCHEMA: Final = (
    "partition_free_v1_alexnet_sampled_score_order_look_receipt_v3"
)

_SCOPE_PREFIX: Final = "partition-free-v1-current-look-fq-alpha-coordinate-scope-"
_SCORE_REGISTRATION_PREFIX: Final = (
    "partition-free-v1-alexnet-score-order-registration-"
)
_SCORE_LOOK_PREFIX: Final = (
    "partition-free-v1-alexnet-score-order-look-receipt-"
)
_ISSUER_TOKEN: Final = object()


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one exact nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _hashed_mapping(
    value: Mapping[str, object],
    *,
    hash_field: str,
    prefix: str,
    name: str,
) -> tuple[dict[str, object], str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    payload = dict(value)
    expected = _identity(payload.pop(hash_field, None), name=f"{name} hash")
    observed = content_hash(payload, prefix=prefix)
    if observed != expected:
        raise ValueError(f"{name} was tampered")
    return payload, observed


def _require_false(mapping: Mapping[str, object], name: str, *, owner: str) -> None:
    if mapping.get(name) is not False:
        raise PermissionError(f"{owner} grants forbidden {name} authority")


@dataclass(frozen=True, slots=True)
class V1CurrentLookFQAlphaCoordinateScopeReceipt:
    """Outcome-free authority to name one preregistered F/Q alpha coordinate.

    The private issued-coordinate tuple is redundant custody, not additional
    scientific state.  It makes ``dataclasses.replace`` fail closed if a
    caller changes a mask, generation, route, design, registration, or look
    coordinate after factory issuance.
    """

    discovery_score_registration_hash: str
    discovery_score_design_hash: str
    discovery_score_look_receipt_hash: str
    discovery_score_look_index: int
    maximum_registered_discovery_score_looks: int
    current_mask_hash: str
    proposal_generation: int
    route_count: int
    ordered_route_axis_hash: str

    proposal_only: bool = True
    issued_before_fq_or_c_outcomes: bool = True
    outcome_free_alpha_coordinate_authority: bool = True
    contains_action_order: bool = False
    contains_c_outcomes: bool = False
    fq_outcomes_observed: bool = False
    bce_outcomes_observed: bool = False
    validation_outcomes_observed: bool = False
    c_execution_authority: bool = False
    fq_certification_authority: bool = False
    bce_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    downstream_authority: bool = False
    terminal_authority: bool = False
    schema: str = CURRENT_LOOK_FQ_ALPHA_COORDINATE_SCOPE_SCHEMA
    _issuer_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
        kw_only=True,
    )
    _issued_coordinates: tuple[object, ...] = field(
        default=(),
        repr=False,
        compare=False,
        kw_only=True,
    )
    _receipt_hash: str | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuer_token is not _ISSUER_TOKEN:
            raise TypeError(
                "current-look F/Q alpha scopes are issued only by the typed factory"
            )
        for name in (
            "discovery_score_registration_hash",
            "discovery_score_design_hash",
            "discovery_score_look_receipt_hash",
            "current_mask_hash",
            "ordered_route_axis_hash",
        ):
            object.__setattr__(
                self,
                name,
                _identity(getattr(self, name), name=name.replace("_", " ")),
            )
        look_index = _positive_int(
            self.discovery_score_look_index,
            name="discovery score-look index",
        )
        maximum_looks = _positive_int(
            self.maximum_registered_discovery_score_looks,
            name="maximum registered discovery score looks",
        )
        if look_index > maximum_looks:
            raise ValueError("discovery score look exceeds its preregistered axis")
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        route_count = _positive_int(self.route_count, name="route count")
        object.__setattr__(self, "discovery_score_look_index", look_index)
        object.__setattr__(
            self,
            "maximum_registered_discovery_score_looks",
            maximum_looks,
        )
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_count", route_count)

        exact_true = (
            "proposal_only",
            "issued_before_fq_or_c_outcomes",
            "outcome_free_alpha_coordinate_authority",
        )
        exact_false = (
            "contains_action_order",
            "contains_c_outcomes",
            "fq_outcomes_observed",
            "bce_outcomes_observed",
            "validation_outcomes_observed",
            "c_execution_authority",
            "fq_certification_authority",
            "bce_oracle_authority",
            "candidate_freeze_authority",
            "mask_commit_authority",
            "downstream_authority",
            "terminal_authority",
        )
        for name in (*exact_true, *exact_false):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be one exact boolean")
        if any(getattr(self, name) is not True for name in exact_true) or any(
            getattr(self, name) is not False for name in exact_false
        ):
            raise PermissionError(
                "current-look F/Q alpha scope crossed its outcome-free boundary"
            )
        if self.schema != CURRENT_LOOK_FQ_ALPHA_COORDINATE_SCOPE_SCHEMA:
            raise ValueError("unsupported current-look F/Q alpha scope schema")

        coordinates = self._coordinate_tuple()
        if tuple(self._issued_coordinates) != coordinates:
            raise ValueError(
                "current-look F/Q alpha coordinates differ from factory issuance"
            )
        self.assert_integrity()

    def _coordinate_tuple(self) -> tuple[object, ...]:
        return (
            self.discovery_score_registration_hash,
            self.discovery_score_design_hash,
            self.discovery_score_look_receipt_hash,
            self.discovery_score_look_index,
            self.maximum_registered_discovery_score_looks,
            self.current_mask_hash,
            self.proposal_generation,
            self.route_count,
            self.ordered_route_axis_hash,
        )

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"_issuer_token", "_issued_coordinates", "_receipt_hash"}
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if tuple(self._issued_coordinates) != self._coordinate_tuple():
            raise ValueError("current-look F/Q alpha scope coordinates were tampered")
        observed = content_hash(self._payload(), prefix=_SCOPE_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("current-look F/Q alpha scope was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("current-look F/Q alpha scope hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    def to_payload(self) -> Mapping[str, object]:
        self.assert_integrity()
        return MappingProxyType(self._payload())

    def to_json_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}

    def _constructor_payload(self) -> dict[str, object]:
        return self._payload()

    def __reduce__(
        self,
    ) -> tuple[object, tuple[dict[str, object], tuple[object, ...], str]]:
        return _rebuild_current_look_fq_alpha_coordinate_scope, (
            self._constructor_payload(),
            self._issued_coordinates,
            self.assert_integrity(),
        )


def register_v1_current_look_fq_alpha_coordinate_scope(
    *,
    proposal_design: V1GroupSequentialSerflingDesign,
    discovery_registration: Mapping[str, object],
    discovery_look_receipt: Mapping[str, object],
) -> V1CurrentLookFQAlphaCoordinateScopeReceipt:
    """Issue the minimal exact current-look scope from the existing D graph."""

    if type(proposal_design) is not V1GroupSequentialSerflingDesign:
        raise TypeError("current-look F/Q alpha scope requires the exact design type")
    design_hash = proposal_design.assert_integrity()
    registration, registration_hash = _hashed_mapping(
        discovery_registration,
        hash_field="registration_hash",
        prefix=_SCORE_REGISTRATION_PREFIX,
        name="discovery score registration",
    )
    look, look_hash = _hashed_mapping(
        discovery_look_receipt,
        hash_field="score_order_look_receipt_hash",
        prefix=_SCORE_LOOK_PREFIX,
        name="discovery score-look receipt",
    )
    if registration.get("schema") != SAMPLED_SCORE_ORDER_REGISTRATION_SCHEMA:
        raise ValueError("unsupported discovery score-registration schema")
    if look.get("schema") != SAMPLED_SCORE_ORDER_LOOK_SCHEMA:
        raise ValueError("unsupported discovery score-look schema")

    confidence_design_hash = _identity(
        registration.get("proposal_order_design_hash"),
        name="registered confidence design hash",
    )
    look_index = _positive_int(look.get("look_index"), name="score-look index")
    maximum_looks = proposal_design.registered_look_count
    if look_index > maximum_looks:
        raise ValueError("score-look index exceeds the registered design")
    expected_counts = proposal_design.registered_look_stratum_counts[look_index - 1]
    look_counts = tuple(look.get("stratum_counts", ()))
    if (
        registration.get("proposal_order_base_serfling_design_hash") != design_hash
        or look.get("proposal_order_base_serfling_design_hash") != design_hash
        or look.get("registration_hash") != registration_hash
        or look.get("proposal_order_design_hash") != confidence_design_hash
        or registration.get("ordered_route_axis_hash")
        != proposal_design.route_axis_hash
        or registration.get("current_mask_hash")
        not in (None, proposal_design.current_mask_hash)
        or registration.get("proposal_generation")
        not in (None, proposal_design.proposal_generation)
        or registration.get("route_count") not in (None, proposal_design.route_count)
        or registration.get("raw_logit_stream_source_hash")
        != proposal_design.raw_logit_stream_source_hash
        or registration.get("query_contract_hash")
        != proposal_design.query_contract_hash
        or tuple(registration.get("registered_look_stratum_counts", ()))
        != proposal_design.registered_look_stratum_counts
        or look_counts != expected_counts
    ):
        raise ValueError(
            "current discovery registration/design/look coordinates do not close"
        )

    for owner, mapping in (
        ("discovery score registration", registration),
        ("discovery score look", look),
    ):
        for authority in (
            "fq_certification_authority",
            "bce_oracle_authority",
            "candidate_freeze_authority",
            "mask_commit_authority",
        ):
            _require_false(mapping, authority, owner=owner)
    _require_false(look, "terminal_authority", owner="discovery score look")
    for owner, mapping in (
        ("discovery score registration", registration),
        ("discovery score look", look),
    ):
        for outcome in (
            "contains_c_outcomes",
            "c_outcomes_observed",
            "fq_outcomes_observed",
            "bce_outcomes_observed",
            "validation_outcomes_observed",
        ):
            if mapping.get(outcome, False) is not False:
                raise PermissionError(f"{owner} contains post-D outcome state")

    coordinates: tuple[object, ...] = (
        registration_hash,
        design_hash,
        look_hash,
        look_index,
        maximum_looks,
        proposal_design.current_mask_hash,
        proposal_design.proposal_generation,
        proposal_design.route_count,
        proposal_design.route_axis_hash,
    )
    return V1CurrentLookFQAlphaCoordinateScopeReceipt(
        discovery_score_registration_hash=registration_hash,
        discovery_score_design_hash=design_hash,
        discovery_score_look_receipt_hash=look_hash,
        discovery_score_look_index=look_index,
        maximum_registered_discovery_score_looks=maximum_looks,
        current_mask_hash=proposal_design.current_mask_hash,
        proposal_generation=proposal_design.proposal_generation,
        route_count=proposal_design.route_count,
        ordered_route_axis_hash=proposal_design.route_axis_hash,
        _issuer_token=_ISSUER_TOKEN,
        _issued_coordinates=coordinates,
    )


def _rebuild_current_look_fq_alpha_coordinate_scope(
    payload: Mapping[str, object],
    issued_coordinates: tuple[object, ...],
    expected_hash: str,
) -> V1CurrentLookFQAlphaCoordinateScopeReceipt:
    rebuilt = V1CurrentLookFQAlphaCoordinateScopeReceipt(
        **dict(payload),
        _issuer_token=_ISSUER_TOKEN,
        _issued_coordinates=tuple(issued_coordinates),
    )  # type: ignore[arg-type]
    rebuilt.assert_integrity(
        _identity(expected_hash, name="expected current-look F/Q alpha scope hash")
    )
    return rebuilt


__all__ = [
    "CURRENT_LOOK_FQ_ALPHA_COORDINATE_SCOPE_SCHEMA",
    "SAMPLED_SCORE_ORDER_LOOK_SCHEMA",
    "SAMPLED_SCORE_ORDER_REGISTRATION_SCHEMA",
    "V1CurrentLookFQAlphaCoordinateScopeReceipt",
    "register_v1_current_look_fq_alpha_coordinate_scope",
]
