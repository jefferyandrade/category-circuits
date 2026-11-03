"""Typed, pre-outcome ownership for the sampled-D to pre-C handoff.

Sampled D has two lawful consumers:

* the historical backend graph builder; or
* one source-bound post-registered-look signed-work consumer.

The consumer is selected before any D outcome.  The resulting registration
binds the live chart and grants only ownership of the D-to-pre-C transition;
it grants no C, calibration, oracle, commit, terminal, or Pareto authority.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .contracts import content_hash


SAMPLED_D_HANDOFF_CONSUMER_DECLARATION_SCHEMA: Final = (
    "partition_free_v1_sampled_D_handoff_consumer_declaration_v1"
)
SAMPLED_D_HANDOFF_CONSUMER_DECLARATION_PREFIX: Final = (
    "partition-free-v1-sampled-D-handoff-consumer-declaration-"
)
SAMPLED_D_HANDOFF_CONSUMER_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_sampled_D_handoff_consumer_registration_v1"
)
SAMPLED_D_HANDOFF_CONSUMER_REGISTRATION_PREFIX: Final = (
    "partition-free-v1-sampled-D-handoff-consumer-registration-"
)
SAMPLED_D_HANDOFF_CONSUMER_CLAIM_SCHEMA: Final = (
    "partition_free_v1_sampled_D_handoff_consumer_claim_v1"
)
SAMPLED_D_HANDOFF_CONSUMER_CLAIM_PREFIX: Final = (
    "partition-free-v1-sampled-D-handoff-consumer-claim-"
)
SAMPLED_D_HANDOFF_CONSUMER_SOURCE_PREFIX: Final = (
    "partition-free-v1-sampled-D-handoff-consumer-source-"
)


class V1SampledDHandoffConsumerIdentity(str, Enum):
    """The complete registered consumer axis for sampled-D."""

    LEGACY_BACKEND_GRAPH = "legacy_backend_graph"
    POST_REGISTERED_LOOK_SIGNED_WORK = "post_registered_look_signed_work"


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def v1_sampled_d_handoff_consumer_source_hash(consumer: Callable[..., Any]) -> str:
    """Bind a callable identity to the complete source file that defines it."""

    if not callable(consumer):
        raise TypeError("sampled-D handoff consumer source must be callable")
    target: Any = consumer
    if not inspect.isfunction(target) and not inspect.ismethod(target):
        target = type(target).__call__
    source_name = inspect.getsourcefile(target)
    if source_name is None:
        raise ValueError("sampled-D handoff consumer source file is unavailable")
    source_path = Path(source_name).resolve(strict=True)
    return content_hash(
        {
            "schema": "partition_free_v1_sampled_D_handoff_consumer_source_v1",
            "module": _identity(getattr(target, "__module__", None), name="module"),
            "qualname": _identity(
                getattr(target, "__qualname__", None),
                name="qualified callable name",
            ),
            "source_file_sha256": _file_sha256(source_path),
        },
        prefix=SAMPLED_D_HANDOFF_CONSUMER_SOURCE_PREFIX,
    )


@dataclass(frozen=True, slots=True)
class V1SampledDHandoffConsumerDeclaration:
    """Source-bound consumer selection made before chart coordinates resolve."""

    consumer_identity: V1SampledDHandoffConsumerIdentity
    consumer_source_hash: str

    def __post_init__(self) -> None:
        if type(self.consumer_identity) is not V1SampledDHandoffConsumerIdentity:
            raise TypeError("sampled-D consumer identity must be the exact enum")
        _identity(self.consumer_source_hash, name="consumer_source_hash")

    @property
    def declaration_hash(self) -> str:
        return content_hash(
            self.to_json_payload(),
            prefix=SAMPLED_D_HANDOFF_CONSUMER_DECLARATION_PREFIX,
        )

    def to_json_payload(self) -> dict[str, object]:
        return {
            "schema": SAMPLED_D_HANDOFF_CONSUMER_DECLARATION_SCHEMA,
            "consumer_identity": self.consumer_identity.value,
            "consumer_source_hash": self.consumer_source_hash,
        }


def declare_v1_sampled_d_handoff_consumer(
    *,
    consumer_identity: V1SampledDHandoffConsumerIdentity,
    consumer: Callable[..., Any],
) -> V1SampledDHandoffConsumerDeclaration:
    """Declare one exact consumer without observing any D outcome."""

    return V1SampledDHandoffConsumerDeclaration(
        consumer_identity=consumer_identity,
        consumer_source_hash=v1_sampled_d_handoff_consumer_source_hash(consumer),
    )


@dataclass(frozen=True, slots=True)
class V1SampledDHandoffConsumerRegistration:
    """Pre-D registration bound to one chart and one exact consumer source."""

    consumer_identity: V1SampledDHandoffConsumerIdentity
    consumer_source_hash: str
    consumer_declaration_hash: str
    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    model_identity_hash: str
    query_contract_hash: str
    train_population_hash: str
    current_mask_hash: str
    route_axis_hash: str
    ordered_route_axis_hash: str
    route_count: int
    proposal_generation: int
    registered_before_D_outcomes: bool = True
    D_outcomes_seen: bool = False
    C_authority: bool = False
    K_calibration_authority: bool = False
    O_oracle_authority: bool = False
    candidate_freeze_authority: bool = False
    mask_commit_authority: bool = False
    terminal_validation_authority: bool = False
    pareto_claim_authority: bool = False

    def __post_init__(self) -> None:
        if type(self.consumer_identity) is not V1SampledDHandoffConsumerIdentity:
            raise TypeError("sampled-D consumer identity must be the exact enum")
        for name in (
            "consumer_source_hash",
            "consumer_declaration_hash",
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "model_identity_hash",
            "query_contract_hash",
            "train_population_hash",
            "current_mask_hash",
            "route_axis_hash",
            "ordered_route_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        _positive_int(self.route_count, name="route_count")
        _nonnegative_int(self.proposal_generation, name="proposal_generation")
        authority = (
            self.registered_before_D_outcomes,
            self.D_outcomes_seen,
            self.C_authority,
            self.K_calibration_authority,
            self.O_oracle_authority,
            self.candidate_freeze_authority,
            self.mask_commit_authority,
            self.terminal_validation_authority,
            self.pareto_claim_authority,
        )
        if any(type(value) is not bool for value in authority):
            raise TypeError("sampled-D consumer authority fields must be exact bools")
        if authority != (True, False, False, False, False, False, False, False, False):
            raise PermissionError(
                "sampled-D consumer registration must precede D and close all "
                "downstream authority"
            )
        declaration = V1SampledDHandoffConsumerDeclaration(
            consumer_identity=self.consumer_identity,
            consumer_source_hash=self.consumer_source_hash,
        )
        if declaration.declaration_hash != self.consumer_declaration_hash:
            raise ValueError("sampled-D consumer declaration hash changed")

    @property
    def registration_hash(self) -> str:
        return content_hash(
            self.to_json_payload(),
            prefix=SAMPLED_D_HANDOFF_CONSUMER_REGISTRATION_PREFIX,
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        result = self.registration_hash
        if expected_hash is not None and result != _identity(
            expected_hash,
            name="expected registration hash",
        ):
            raise ValueError("sampled-D consumer registration hash changed")
        return result

    def to_json_payload(self) -> dict[str, object]:
        return {
            "schema": SAMPLED_D_HANDOFF_CONSUMER_REGISTRATION_SCHEMA,
            "consumer_identity": self.consumer_identity.value,
            "consumer_source_hash": self.consumer_source_hash,
            "consumer_declaration_hash": self.consumer_declaration_hash,
            "campaign_contract_hash": self.campaign_contract_hash,
            "campaign_binding_hash": self.campaign_binding_hash,
            "controller_run_id": self.controller_run_id,
            "model_identity_hash": self.model_identity_hash,
            "query_contract_hash": self.query_contract_hash,
            "train_population_hash": self.train_population_hash,
            "current_mask_hash": self.current_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "ordered_route_axis_hash": self.ordered_route_axis_hash,
            "route_count": self.route_count,
            "proposal_generation": self.proposal_generation,
            "registered_before_D_outcomes": self.registered_before_D_outcomes,
            "D_outcomes_seen": self.D_outcomes_seen,
            "C_authority": self.C_authority,
            "K_calibration_authority": self.K_calibration_authority,
            "O_oracle_authority": self.O_oracle_authority,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_validation_authority": self.terminal_validation_authority,
            "pareto_claim_authority": self.pareto_claim_authority,
        }

    @classmethod
    def from_json_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "V1SampledDHandoffConsumerRegistration":
        value = dict(payload)
        expected = {
            "schema",
            "consumer_identity",
            "consumer_source_hash",
            "consumer_declaration_hash",
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "model_identity_hash",
            "query_contract_hash",
            "train_population_hash",
            "current_mask_hash",
            "route_axis_hash",
            "ordered_route_axis_hash",
            "route_count",
            "proposal_generation",
            "registered_before_D_outcomes",
            "D_outcomes_seen",
            "C_authority",
            "K_calibration_authority",
            "O_oracle_authority",
            "candidate_freeze_authority",
            "mask_commit_authority",
            "terminal_validation_authority",
            "pareto_claim_authority",
        }
        if set(value) != expected or value.get("schema") != (
            SAMPLED_D_HANDOFF_CONSUMER_REGISTRATION_SCHEMA
        ):
            raise ValueError("sampled-D consumer registration payload is noncanonical")
        try:
            identity = V1SampledDHandoffConsumerIdentity(value["consumer_identity"])
        except (TypeError, ValueError) as error:
            raise ValueError("sampled-D consumer identity is unknown") from error
        return cls(
            consumer_identity=identity,
            consumer_source_hash=value["consumer_source_hash"],
            consumer_declaration_hash=value["consumer_declaration_hash"],
            campaign_contract_hash=value["campaign_contract_hash"],
            campaign_binding_hash=value["campaign_binding_hash"],
            controller_run_id=value["controller_run_id"],
            model_identity_hash=value["model_identity_hash"],
            query_contract_hash=value["query_contract_hash"],
            train_population_hash=value["train_population_hash"],
            current_mask_hash=value["current_mask_hash"],
            route_axis_hash=value["route_axis_hash"],
            ordered_route_axis_hash=value["ordered_route_axis_hash"],
            route_count=value["route_count"],
            proposal_generation=value["proposal_generation"],
            registered_before_D_outcomes=value["registered_before_D_outcomes"],
            D_outcomes_seen=value["D_outcomes_seen"],
            C_authority=value["C_authority"],
            K_calibration_authority=value["K_calibration_authority"],
            O_oracle_authority=value["O_oracle_authority"],
            candidate_freeze_authority=value["candidate_freeze_authority"],
            mask_commit_authority=value["mask_commit_authority"],
            terminal_validation_authority=value["terminal_validation_authority"],
            pareto_claim_authority=value["pareto_claim_authority"],
        )


def issue_v1_sampled_d_handoff_consumer_registration(
    *,
    declaration: V1SampledDHandoffConsumerDeclaration,
    campaign_contract_hash: str,
    campaign_binding_hash: str,
    controller_run_id: str,
    model_identity_hash: str,
    query_contract_hash: str,
    train_population_hash: str,
    current_mask_hash: str,
    route_axis_hash: str,
    ordered_route_axis_hash: str,
    route_count: int,
    proposal_generation: int,
) -> V1SampledDHandoffConsumerRegistration:
    """Issue the full chart-bound registration before D is observed."""

    if type(declaration) is not V1SampledDHandoffConsumerDeclaration:
        raise TypeError("sampled-D consumer declaration must be exact")
    return V1SampledDHandoffConsumerRegistration(
        consumer_identity=declaration.consumer_identity,
        consumer_source_hash=declaration.consumer_source_hash,
        consumer_declaration_hash=declaration.declaration_hash,
        campaign_contract_hash=campaign_contract_hash,
        campaign_binding_hash=campaign_binding_hash,
        controller_run_id=controller_run_id,
        model_identity_hash=model_identity_hash,
        query_contract_hash=query_contract_hash,
        train_population_hash=train_population_hash,
        current_mask_hash=current_mask_hash,
        route_axis_hash=route_axis_hash,
        ordered_route_axis_hash=ordered_route_axis_hash,
        route_count=route_count,
        proposal_generation=proposal_generation,
    )


def assert_v1_sampled_d_handoff_consumer_coordinates(
    registration: V1SampledDHandoffConsumerRegistration,
    *,
    campaign_contract_hash: str,
    campaign_binding_hash: str,
    controller_run_id: str,
    model_identity_hash: str,
    query_contract_hash: str,
    train_population_hash: str,
    current_mask_hash: str,
    route_axis_hash: str,
    ordered_route_axis_hash: str,
    route_count: int,
    proposal_generation: int,
) -> str:
    """Refuse stale, cross-chart, or cross-source consumer registrations."""

    if type(registration) is not V1SampledDHandoffConsumerRegistration:
        raise TypeError("sampled-D consumer registration must be exact")
    actual = (
        registration.campaign_contract_hash,
        registration.campaign_binding_hash,
        registration.controller_run_id,
        registration.model_identity_hash,
        registration.query_contract_hash,
        registration.train_population_hash,
        registration.current_mask_hash,
        registration.route_axis_hash,
        registration.ordered_route_axis_hash,
        registration.route_count,
        registration.proposal_generation,
    )
    expected = (
        campaign_contract_hash,
        campaign_binding_hash,
        controller_run_id,
        model_identity_hash,
        query_contract_hash,
        train_population_hash,
        current_mask_hash,
        route_axis_hash,
        ordered_route_axis_hash,
        route_count,
        proposal_generation,
    )
    if actual != expected:
        raise PermissionError("sampled-D consumer registration is stale or foreign")
    return registration.assert_integrity()


@dataclass(frozen=True, slots=True)
class V1SampledDHandoffConsumerClaimReceipt:
    """Post-D proof that the pre-registered source owns D-to-pre-C."""

    registration_hash: str
    consumer_identity: V1SampledDHandoffConsumerIdentity
    consumer_source_hash: str
    D_handoff_descriptor_hash: str
    D_handoff_receipt_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    proposal_generation: int
    D_outcomes_seen: bool = True
    C_authority: bool = False
    K_calibration_authority: bool = False
    O_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_validation_authority: bool = False
    pareto_claim_authority: bool = False

    def __post_init__(self) -> None:
        if (
            self.consumer_identity
            is not V1SampledDHandoffConsumerIdentity.POST_REGISTERED_LOOK_SIGNED_WORK
        ):
            raise PermissionError("only signed-work may claim the post-look handoff")
        for name in (
            "registration_hash",
            "consumer_source_hash",
            "D_handoff_descriptor_hash",
            "D_handoff_receipt_hash",
            "current_mask_hash",
            "route_axis_hash",
        ):
            _identity(getattr(self, name), name=name)
        _positive_int(self.route_count, name="route_count")
        _nonnegative_int(self.proposal_generation, name="proposal_generation")
        authority = (
            self.D_outcomes_seen,
            self.C_authority,
            self.K_calibration_authority,
            self.O_oracle_authority,
            self.mask_commit_authority,
            self.terminal_validation_authority,
            self.pareto_claim_authority,
        )
        if any(type(value) is not bool for value in authority):
            raise TypeError("sampled-D handoff claim authority must be exact bools")
        if authority != (True, False, False, False, False, False, False):
            raise PermissionError("sampled-D handoff claim opened downstream authority")

    @property
    def claim_hash(self) -> str:
        return content_hash(
            self.to_json_payload(),
            prefix=SAMPLED_D_HANDOFF_CONSUMER_CLAIM_PREFIX,
        )

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        result = self.claim_hash
        if expected_hash is not None and result != _identity(
            expected_hash,
            name="expected claim hash",
        ):
            raise ValueError("sampled-D handoff consumer claim hash changed")
        return result

    def to_json_payload(self) -> dict[str, object]:
        return {
            "schema": SAMPLED_D_HANDOFF_CONSUMER_CLAIM_SCHEMA,
            "registration_hash": self.registration_hash,
            "consumer_identity": self.consumer_identity.value,
            "consumer_source_hash": self.consumer_source_hash,
            "D_handoff_descriptor_hash": self.D_handoff_descriptor_hash,
            "D_handoff_receipt_hash": self.D_handoff_receipt_hash,
            "current_mask_hash": self.current_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "proposal_generation": self.proposal_generation,
            "D_outcomes_seen": self.D_outcomes_seen,
            "C_authority": self.C_authority,
            "K_calibration_authority": self.K_calibration_authority,
            "O_oracle_authority": self.O_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_validation_authority": self.terminal_validation_authority,
            "pareto_claim_authority": self.pareto_claim_authority,
        }

    @classmethod
    def from_json_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "V1SampledDHandoffConsumerClaimReceipt":
        value = dict(payload)
        expected = {
            "schema",
            "registration_hash",
            "consumer_identity",
            "consumer_source_hash",
            "D_handoff_descriptor_hash",
            "D_handoff_receipt_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_count",
            "proposal_generation",
            "D_outcomes_seen",
            "C_authority",
            "K_calibration_authority",
            "O_oracle_authority",
            "mask_commit_authority",
            "terminal_validation_authority",
            "pareto_claim_authority",
        }
        if set(value) != expected or value.get("schema") != (
            SAMPLED_D_HANDOFF_CONSUMER_CLAIM_SCHEMA
        ):
            raise ValueError("sampled-D handoff consumer claim is noncanonical")
        try:
            identity = V1SampledDHandoffConsumerIdentity(value["consumer_identity"])
        except (TypeError, ValueError) as error:
            raise ValueError("sampled-D claim consumer identity is unknown") from error
        return cls(
            registration_hash=value["registration_hash"],
            consumer_identity=identity,
            consumer_source_hash=value["consumer_source_hash"],
            D_handoff_descriptor_hash=value["D_handoff_descriptor_hash"],
            D_handoff_receipt_hash=value["D_handoff_receipt_hash"],
            current_mask_hash=value["current_mask_hash"],
            route_axis_hash=value["route_axis_hash"],
            route_count=value["route_count"],
            proposal_generation=value["proposal_generation"],
            D_outcomes_seen=value["D_outcomes_seen"],
            C_authority=value["C_authority"],
            K_calibration_authority=value["K_calibration_authority"],
            O_oracle_authority=value["O_oracle_authority"],
            mask_commit_authority=value["mask_commit_authority"],
            terminal_validation_authority=value["terminal_validation_authority"],
            pareto_claim_authority=value["pareto_claim_authority"],
        )


def claim_v1_sampled_d_handoff_for_post_registered_look(
    registration: V1SampledDHandoffConsumerRegistration,
    *,
    declaration: V1SampledDHandoffConsumerDeclaration,
    D_handoff_descriptor_hash: str,
    D_handoff_receipt_hash: str,
) -> V1SampledDHandoffConsumerClaimReceipt:
    """Let only the exact pre-registered signed-work source claim pre-C."""

    if type(registration) is not V1SampledDHandoffConsumerRegistration:
        raise TypeError("sampled-D consumer registration must be exact")
    if type(declaration) is not V1SampledDHandoffConsumerDeclaration:
        raise TypeError("sampled-D consumer declaration must be exact")
    if (
        registration.consumer_identity
        is not V1SampledDHandoffConsumerIdentity.POST_REGISTERED_LOOK_SIGNED_WORK
        or declaration.consumer_identity is not registration.consumer_identity
        or declaration.consumer_source_hash != registration.consumer_source_hash
        or declaration.declaration_hash != registration.consumer_declaration_hash
    ):
        raise PermissionError(
            "post-look signed-work source was not the pre-registered D consumer"
        )
    registration_hash = registration.assert_integrity()
    return V1SampledDHandoffConsumerClaimReceipt(
        registration_hash=registration_hash,
        consumer_identity=registration.consumer_identity,
        consumer_source_hash=registration.consumer_source_hash,
        D_handoff_descriptor_hash=D_handoff_descriptor_hash,
        D_handoff_receipt_hash=D_handoff_receipt_hash,
        current_mask_hash=registration.current_mask_hash,
        route_axis_hash=registration.route_axis_hash,
        route_count=registration.route_count,
        proposal_generation=registration.proposal_generation,
    )


__all__ = [
    "SAMPLED_D_HANDOFF_CONSUMER_CLAIM_SCHEMA",
    "SAMPLED_D_HANDOFF_CONSUMER_DECLARATION_SCHEMA",
    "SAMPLED_D_HANDOFF_CONSUMER_REGISTRATION_SCHEMA",
    "V1SampledDHandoffConsumerClaimReceipt",
    "V1SampledDHandoffConsumerDeclaration",
    "V1SampledDHandoffConsumerIdentity",
    "V1SampledDHandoffConsumerRegistration",
    "assert_v1_sampled_d_handoff_consumer_coordinates",
    "claim_v1_sampled_d_handoff_for_post_registered_look",
    "declare_v1_sampled_d_handoff_consumer",
    "issue_v1_sampled_d_handoff_consumer_registration",
    "v1_sampled_d_handoff_consumer_source_hash",
]
