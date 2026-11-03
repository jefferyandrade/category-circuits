"""Immutable Option-A campaign contract for modern partition-free v1.

This module owns the J0 registration object named by the modern-v1 landing
plan.  It freezes every campaign-wide, decision-bearing coordinate before any
candidate-specific evidence exists and exposes only content-addressed views of
that contract to the staged-alpha and owner-registration surfaces.

The contract is deliberately authority-inert.  Constructing it does not open
``irg/v1``, authorize a bounded GPU probe, certify a candidate, or permit a
mask commit.  It also does not read environment variables: runtime launchers
must parse an exact external configuration into this object and receipt the
source bytes separately.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

from .contracts import content_hash
from .modern_v1_core import MODERN_V1_WORKFLOW_IDENTITY, V1OwnerRegistration
from .modern_v1_staged_alpha import (
    STAGED_ALPHA_BOUND_ID,
    STAGED_ALPHA_EXHAUSTION_ACTION_ID,
    STAGED_ALPHA_LOOK_SPENDING_ID,
    STAGED_ALPHA_PROOF_ID,
    STAGED_ALPHA_SAMPLING_ID,
    STAGED_ALPHA_STEP_SPENDING_ID,
    V1StagedAlphaCoordinates,
)

OPTION_A_CAMPAIGN_CONTRACT_SCHEMA: Final = (
    "partition_free_v1_option_a_campaign_contract_v2"
)
OPTION_A_CAMPAIGN_BINDING_SCHEMA: Final = (
    "partition_free_v1_option_a_campaign_binding_v1"
)
DENSE_REFERENT_MASK_METADATA_KEY: Final = "dense_referent_mask_hash"
MODERN_V1_DESIGN_IDENTITY: Final = "irg_v1_partition_free_dynamic_relinearization_v1"
MODERN_V1_WORKFLOW_ALIAS: Final = "partition_free_v1"
DIRECT_RAW_BCE_INCREMENT_ID: Final = (
    "direct_raw_bce_candidate_minus_dense_baseline_mask_v1"
)
DIRECT_INCREMENT_ORIENTATION_ID: Final = (
    "candidate_mask_minus_dense_baseline_mask_v1"
)
EXHAUSTION_ACTION_ID: Final = STAGED_ALPHA_EXHAUSTION_ACTION_ID
ATOMIC_REFRESH_POLICY_ID: Final = "atomic_full_refresh_after_each_accepted_mask_v1"
D_MAX_DISPOSITION_ID: Final = "absent_not_used_by_option_a_v1"
FINITE_HORIZON_CROSS_ATTEMPT_SPENDING_ID: Final = (
    "finite_horizon_normalized_inverse_square_cross_attempt_spending_v1"
)
NO_CROSS_QUERY_BORROWING_ID: Final = "independent_querywise_no_borrowing_v1"
CAMPAIGN_WIDE_DENSE_RELATIVE_CONFIDENCE_CLAIM_ID: Final = (
    "campaign_wide_dense_relative_accepted_mask_union_bound_v1"
)
TERMINAL_REVEAL_POLICY_ID: Final = (
    "freeze_extraction_and_calibration_then_single_terminal_reveal_v1"
)
PROPOSAL_ONLY_AUTHORITY_ID: Final = "proposal_geometry_never_commits_mask_v1"
STUDENT_T_EVIDENCE_CEILING_ID: Final = (
    "exact_under_normal_rows_otherwise_model_or_clt_evidence_v1"
)
OPTION_A_FD_RANK: Final = 8


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _freeze(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str or not key:
                raise TypeError(f"{label} keys must be nonempty strings")
            frozen[key] = _freeze(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, f"{label}[{index}]") for index, item in enumerate(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a nonfinite float")
        return float(value)
    if isinstance(value, (str, bytes, int, bool)) or value is None:
        return value
    raise TypeError(f"{label} contains an unsupported value")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _strictly_increasing_counts(values: Sequence[int]) -> tuple[int, ...]:
    counts = tuple(_positive_int(value, "cumulative draw count") for value in values)
    if not counts or any(right <= left for left, right in zip(counts, counts[1:])):
        raise ValueError("cumulative draw counts must be nonempty and strictly increasing")
    return counts


@dataclass(frozen=True, slots=True)
class V1OptionACampaignContract:
    """One hashable source of truth for the initial Option-A campaign.

    Hash-valued fields bind full externally governed objects without copying
    large manifests into every runtime receipt.  The numeric and categorical
    fields here are the exact values that may affect candidate ordering,
    evidence acquisition, stopping, or terminal disclosure.
    """

    campaign_id: str
    registration_id: str
    model_seal_hash: str
    query_id: str
    referent_id: str
    route_axis_hash: str
    context_population_hash: str
    context_population_source_manifest_hash: str
    target_population_hash: str
    dense_reference_hash: str
    calibration_identity: str
    terminal_validation_population_hash: str
    campaign_wide_alpha: float
    per_step_alpha: float
    cumulative_draw_counts: tuple[int, ...]
    maximum_importance_weight: float
    maximum_step_count: int
    maximum_candidate_count: int
    maximum_removed_count: int
    look_schedule_policy_hash: str
    damage_target_derivation_hash: str
    sampling_law_hash: str
    support_construction_hash: str
    importance_weight_bound_proof_hash: str
    entropy_source_policy_hash: str
    entropy_replay_policy_hash: str
    candidate_domain_policy_hash: str
    resource_cost_policy_hash: str
    queue_exhaustion_policy_hash: str
    terminal_calibration_policy_hash: str
    terminal_validation_policy_hash: str
    student_t_evidence_disclosure_hash: str
    proof_artifact_hash: str
    owner_decision_record_hash: str
    config_source_receipt_hash: str
    design_identity: str = MODERN_V1_DESIGN_IDENTITY
    workflow_identity: str = MODERN_V1_WORKFLOW_IDENTITY
    workflow_alias: str = MODERN_V1_WORKFLOW_ALIAS
    certificate_family: Literal["staged_alpha"] = "staged_alpha"
    cross_query_shrinkage: Literal["defer"] = "defer"
    step_spending_id: str = STAGED_ALPHA_STEP_SPENDING_ID
    look_spending_id: str = STAGED_ALPHA_LOOK_SPENDING_ID
    interval_bound_id: str = STAGED_ALPHA_BOUND_ID
    sampling_algorithm_id: str = STAGED_ALPHA_SAMPLING_ID
    proof_id: str = STAGED_ALPHA_PROOF_ID
    direct_increment_target_id: str = DIRECT_RAW_BCE_INCREMENT_ID
    increment_orientation_id: str = DIRECT_INCREMENT_ORIENTATION_ID
    exhaustion_action_id: str = EXHAUSTION_ACTION_ID
    fd_rank: int = OPTION_A_FD_RANK
    refresh_policy_id: str = ATOMIC_REFRESH_POLICY_ID
    d_max_disposition_id: str = D_MAX_DISPOSITION_ID
    cross_step_spending_id: str = FINITE_HORIZON_CROSS_ATTEMPT_SPENDING_ID
    cross_query_borrowing_id: str = NO_CROSS_QUERY_BORROWING_ID
    trajectory_confidence_claim_id: str = (
        CAMPAIGN_WIDE_DENSE_RELATIVE_CONFIDENCE_CLAIM_ID
    )
    terminal_reveal_policy_id: str = TERMINAL_REVEAL_POLICY_ID
    proposal_authority_id: str = PROPOSAL_ONLY_AUTHORITY_ID
    student_t_evidence_ceiling_id: str = STUDENT_T_EVIDENCE_CEILING_ID
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = OPTION_A_CAMPAIGN_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_id",
            "registration_id",
            "model_seal_hash",
            "query_id",
            "referent_id",
            "route_axis_hash",
            "context_population_hash",
            "context_population_source_manifest_hash",
            "target_population_hash",
            "dense_reference_hash",
            "calibration_identity",
            "terminal_validation_population_hash",
            "look_schedule_policy_hash",
            "damage_target_derivation_hash",
            "sampling_law_hash",
            "support_construction_hash",
            "importance_weight_bound_proof_hash",
            "entropy_source_policy_hash",
            "entropy_replay_policy_hash",
            "candidate_domain_policy_hash",
            "resource_cost_policy_hash",
            "queue_exhaustion_policy_hash",
            "terminal_calibration_policy_hash",
            "terminal_validation_policy_hash",
            "student_t_evidence_disclosure_hash",
            "proof_artifact_hash",
            "owner_decision_record_hash",
            "config_source_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))

        campaign_alpha = float(self.campaign_wide_alpha)
        alpha = float(self.per_step_alpha)
        if not math.isfinite(campaign_alpha) or not 0.0 < campaign_alpha < 1.0:
            raise ValueError("campaign_wide_alpha must lie in (0,1)")
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("per_step_alpha must lie in (0,1)")
        object.__setattr__(self, "campaign_wide_alpha", campaign_alpha)
        object.__setattr__(self, "per_step_alpha", alpha)
        counts = _strictly_increasing_counts(self.cumulative_draw_counts)
        if counts[0] < 2:
            raise ValueError("the first Student-t look requires at least two draws")
        object.__setattr__(self, "cumulative_draw_counts", counts)
        maximum_importance_weight = _finite_positive(
            self.maximum_importance_weight,
            "maximum_importance_weight",
        )
        if maximum_importance_weight < 1.0:
            raise ValueError("maximum_importance_weight cannot be below one")
        object.__setattr__(
            self,
            "maximum_importance_weight",
            maximum_importance_weight,
        )
        for name in (
            "maximum_step_count",
            "maximum_candidate_count",
            "maximum_removed_count",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if self.maximum_candidate_count < self.maximum_step_count:
            raise ValueError(
                "maximum_candidate_count cannot be below maximum_step_count"
            )
        expected_attempt_alpha = campaign_alpha / math.fsum(
            1.0 / float((index + 1) ** 2)
            for index in range(self.maximum_candidate_count)
        )
        if alpha != expected_attempt_alpha:
            raise ValueError(
                "per_step_alpha must be the registered first-attempt "
                "inverse-square campaign stake"
            )
        # The removal cap counts physical route coordinates; the step cap
        # counts accepted controller transitions.  A finite global action may
        # explicitly remove many routes in one transition, so these two
        # cardinalities are intentionally independent.  Coupling them made the
        # entropy tape scale with model size and ruled out AlexNet-scale
        # composite actions even though the domain already enumerates them.

        exact = {
            "design_identity": MODERN_V1_DESIGN_IDENTITY,
            "workflow_identity": MODERN_V1_WORKFLOW_IDENTITY,
            "workflow_alias": MODERN_V1_WORKFLOW_ALIAS,
            "certificate_family": "staged_alpha",
            "cross_query_shrinkage": "defer",
            "step_spending_id": STAGED_ALPHA_STEP_SPENDING_ID,
            "look_spending_id": STAGED_ALPHA_LOOK_SPENDING_ID,
            "interval_bound_id": STAGED_ALPHA_BOUND_ID,
            "sampling_algorithm_id": STAGED_ALPHA_SAMPLING_ID,
            "proof_id": STAGED_ALPHA_PROOF_ID,
            "direct_increment_target_id": DIRECT_RAW_BCE_INCREMENT_ID,
            "increment_orientation_id": DIRECT_INCREMENT_ORIENTATION_ID,
            "exhaustion_action_id": EXHAUSTION_ACTION_ID,
            "fd_rank": OPTION_A_FD_RANK,
            "refresh_policy_id": ATOMIC_REFRESH_POLICY_ID,
            "d_max_disposition_id": D_MAX_DISPOSITION_ID,
            "cross_step_spending_id": FINITE_HORIZON_CROSS_ATTEMPT_SPENDING_ID,
            "cross_query_borrowing_id": NO_CROSS_QUERY_BORROWING_ID,
            "trajectory_confidence_claim_id": (
                CAMPAIGN_WIDE_DENSE_RELATIVE_CONFIDENCE_CLAIM_ID
            ),
            "terminal_reveal_policy_id": TERMINAL_REVEAL_POLICY_ID,
            "proposal_authority_id": PROPOSAL_ONLY_AUTHORITY_ID,
            "student_t_evidence_ceiling_id": STUDENT_T_EVIDENCE_CEILING_ID,
            "schema": OPTION_A_CAMPAIGN_CONTRACT_SCHEMA,
        }
        for name, expected in exact.items():
            if getattr(self, name) != expected:
                raise ValueError(f"Option-A campaign fixed coordinate {name} drifted")
        object.__setattr__(self, "metadata", _freeze(self.metadata, "campaign metadata"))

    @property
    def maximum_looks_per_step(self) -> int:
        return len(self.cumulative_draw_counts)

    @property
    def campaign_contract_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-option-a-campaign-")

    def to_json_payload(self) -> dict[str, Any]:
        payload = {item.name: _thaw(getattr(self, item.name)) for item in fields(self)}
        payload["campaign_contract_hash"] = self.campaign_contract_hash
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-safe external representation."""

        return self.to_json_payload()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V1OptionACampaignContract":
        """Reconstruct and verify one serialized campaign object."""

        if not isinstance(payload, Mapping):
            raise TypeError("Option-A campaign payload must be a mapping")
        values = dict(payload)
        declared_hash = values.pop("campaign_contract_hash", None)
        contract = cls(**values)
        if declared_hash is not None and declared_hash != contract.campaign_contract_hash:
            raise ValueError("serialized Option-A campaign contract hash mismatch")
        return contract

    @classmethod
    def load(cls, path: str | Path) -> "V1OptionACampaignContract":
        return load_v1_option_a_campaign_contract(path)

    def write(self, path: str | Path) -> None:
        write_v1_option_a_campaign_contract(path, self)

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def mask_commit_authority(self) -> bool:
        return False

    def staged_alpha_coordinates(self) -> V1StagedAlphaCoordinates:
        """Project the full contract into the exact staged-alpha runtime view."""

        return V1StagedAlphaCoordinates(
            campaign_wide_alpha=self.campaign_wide_alpha,
            per_step_alpha=self.per_step_alpha,
            maximum_candidate_count=self.maximum_candidate_count,
            maximum_looks_per_step=self.maximum_looks_per_step,
            maximum_importance_weight=self.maximum_importance_weight,
            campaign_contract_hash=self.campaign_contract_hash,
            registered_cumulative_draw_counts=self.cumulative_draw_counts,
            look_schedule_policy_hash=self.look_schedule_policy_hash,
            proof_artifact_hash=self.proof_artifact_hash,
            owner_decision_record_hash=self.owner_decision_record_hash,
            importance_weight_bound_proof_hash=self.importance_weight_bound_proof_hash,
            entropy_source_policy_hash=self.entropy_source_policy_hash,
            metadata={
                "campaign_id": self.campaign_id,
                "registration_id": self.registration_id,
                "config_source_receipt_hash": self.config_source_receipt_hash,
                "direct_increment_target_id": self.direct_increment_target_id,
                "increment_orientation_id": self.increment_orientation_id,
                "damage_target_derivation_hash": self.damage_target_derivation_hash,
                "sampling_law_hash": self.sampling_law_hash,
                "support_construction_hash": self.support_construction_hash,
                "entropy_replay_policy_hash": self.entropy_replay_policy_hash,
                "queue_exhaustion_policy_hash": self.queue_exhaustion_policy_hash,
                "fd_rank": self.fd_rank,
                "refresh_policy_id": self.refresh_policy_id,
                "d_max_disposition_id": self.d_max_disposition_id,
            },
        )

    def owner_registration(self) -> V1OwnerRegistration:
        """Bind the contract to an authority-inert selected-family registration."""

        coordinates = self.staged_alpha_coordinates()
        return V1OwnerRegistration(
            registration_id=self.registration_id,
            referent_id=self.referent_id,
            route_axis_hash=self.route_axis_hash,
            context_population_hash=self.context_population_hash,
            calibration_identity=self.calibration_identity,
            campaign_contract_hash=self.campaign_contract_hash,
            certificate_family="staged_alpha",
            cross_query_shrinkage="defer",
            a1_coordinates_hash=coordinates.coordinates_hash,
            a1_proof_hash=self.proof_artifact_hash,
            metadata={
                "campaign_id": self.campaign_id,
                "model_seal_hash": self.model_seal_hash,
                "query_id": self.query_id,
                "target_population_hash": self.target_population_hash,
                "context_population_source_manifest_hash": (
                    self.context_population_source_manifest_hash
                ),
                "dense_reference_hash": self.dense_reference_hash,
                "terminal_validation_population_hash": self.terminal_validation_population_hash,
                "config_source_receipt_hash": self.config_source_receipt_hash,
            },
        )


def load_v1_option_a_campaign_contract(
    path: str | Path,
) -> V1OptionACampaignContract:
    """Load and rehash one exact external campaign JSON object."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Option-A campaign JSON must contain an object")
    try:
        return V1OptionACampaignContract.from_dict(payload)
    except ValueError as exc:
        if "serialized Option-A campaign contract hash mismatch" in str(exc):
            raise ValueError(
                "Option-A campaign JSON differs from its declared content hash"
            ) from exc
        raise


def write_v1_option_a_campaign_contract(
    path: str | Path,
    contract: V1OptionACampaignContract,
) -> None:
    """Atomically write one canonical external campaign configuration."""

    if type(contract) is not V1OptionACampaignContract:
        raise TypeError("campaign writer requires an exact Option-A contract")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            contract.to_json_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(destination)


@dataclass(frozen=True, slots=True)
class V1OptionACampaignBinding:
    """Recomputable tuple proving contract/coordinates/registration agreement."""

    contract: V1OptionACampaignContract
    coordinates: V1StagedAlphaCoordinates
    registration: V1OwnerRegistration
    schema: str = OPTION_A_CAMPAIGN_BINDING_SCHEMA

    def __post_init__(self) -> None:
        if type(self.contract) is not V1OptionACampaignContract:
            raise TypeError("campaign binding requires an exact Option-A contract")
        if type(self.coordinates) is not V1StagedAlphaCoordinates:
            raise TypeError("campaign binding requires exact staged-alpha coordinates")
        if type(self.registration) is not V1OwnerRegistration:
            raise TypeError("campaign binding requires an exact owner registration")
        expected_coordinates = self.contract.staged_alpha_coordinates()
        expected_registration = self.contract.owner_registration()
        if self.coordinates != expected_coordinates:
            raise ValueError("staged-alpha coordinates differ from the campaign contract")
        if self.registration != expected_registration:
            raise ValueError("owner registration differs from the campaign contract")
        if self.schema != OPTION_A_CAMPAIGN_BINDING_SCHEMA:
            raise ValueError("unsupported Option-A campaign-binding schema")

    @property
    def binding_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-option-a-campaign-binding-")

    @property
    def claim_execution_certified(self) -> bool:
        return False

    @property
    def mask_commit_authority(self) -> bool:
        return False


def bind_v1_option_a_campaign(
    contract: V1OptionACampaignContract,
) -> V1OptionACampaignBinding:
    """Issue the deterministic selected-family views of one exact contract."""

    if type(contract) is not V1OptionACampaignContract:
        raise TypeError("campaign binding requires an exact Option-A contract")
    return V1OptionACampaignBinding(
        contract=contract,
        coordinates=contract.staged_alpha_coordinates(),
        registration=contract.owner_registration(),
    )


def v1_dense_referent_mask_hash(contract: V1OptionACampaignContract) -> str:
    """Return the hard-mask identity used by paired-BCE evaluation.

    ``dense_reference_hash`` owns the terminal reference artifact/design and
    is not generally the hash of the dense :class:`HardMask`.  New contracts
    bind that mask separately in metadata.  The fallback keeps legacy
    contracts byte-readable without rewriting their hashes.
    """

    if type(contract) is not V1OptionACampaignContract:
        raise TypeError("dense referent lookup needs an exact campaign contract")
    value = contract.metadata.get(DENSE_REFERENT_MASK_METADATA_KEY)
    if value is None:
        return contract.dense_reference_hash
    return _nonempty(value, DENSE_REFERENT_MASK_METADATA_KEY)


__all__ = [
    "ATOMIC_REFRESH_POLICY_ID",
    "DIRECT_INCREMENT_ORIENTATION_ID",
    "DIRECT_RAW_BCE_INCREMENT_ID",
    "D_MAX_DISPOSITION_ID",
    "DENSE_REFERENT_MASK_METADATA_KEY",
    "EXHAUSTION_ACTION_ID",
    "MODERN_V1_DESIGN_IDENTITY",
    "MODERN_V1_WORKFLOW_ALIAS",
    "MODERN_V1_WORKFLOW_IDENTITY",
    "NO_CROSS_QUERY_BORROWING_ID",
    "FINITE_HORIZON_CROSS_ATTEMPT_SPENDING_ID",
    "CAMPAIGN_WIDE_DENSE_RELATIVE_CONFIDENCE_CLAIM_ID",
    "OPTION_A_CAMPAIGN_BINDING_SCHEMA",
    "OPTION_A_CAMPAIGN_CONTRACT_SCHEMA",
    "OPTION_A_FD_RANK",
    "PROPOSAL_ONLY_AUTHORITY_ID",
    "STUDENT_T_EVIDENCE_CEILING_ID",
    "TERMINAL_REVEAL_POLICY_ID",
    "V1OptionACampaignBinding",
    "V1OptionACampaignContract",
    "bind_v1_option_a_campaign",
    "load_v1_option_a_campaign_contract",
    "v1_dense_referent_mask_hash",
    "write_v1_option_a_campaign_contract",
]
