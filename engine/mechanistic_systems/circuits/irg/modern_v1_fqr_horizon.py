"""Typed proof and closure witness for a canonical F/Q/R endpoint horizon.

The ordinary proposal-freeze protocol answers whether a candidate queue may be
frozen.  A complete stable-negative F/Q/R branch family answers a different
question: it proves that the current registered frozen endpoint family contains
no certified endpoint.  The one-route rung closes the minimum-size endpoint of
each registered ladder; it does not enumerate or reject every legal singleton
mask outside those ladders.  These receipts keep those authorities separate.

The proof is compact, but it does not replace the proposal evidence graph.  A
terminal runtime must retain the existing proposal-progress sidecar so the
sampling, deterministic-uncertainty, and directional-remainder receipts remain
replayable after restart.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Final

from .contracts import content_hash


FQR_NO_CANDIDATE_HORIZON_STOP_REASON: Final = (
    "all_registered_frozen_current_generation_fqr_endpoints_certified_"
    "infeasible_"
    "at_stable_C_horizon"
)
FQR_NO_CANDIDATE_HORIZON_PROOF_HASH_PREFIX: Final = (
    "partition-free-v1-fqr-no-candidate-horizon-proof-"
)
FQR_NO_CANDIDATE_HORIZON_WITNESS_HASH_PREFIX: Final = (
    "partition-free-v1-controller-proposal-horizon-witness-"
)

_PROOF_SCHEMA: Final = "partition_free_v1_fqr_no_candidate_horizon_proof_v1"
_WITNESS_SCHEMA: Final = (
    "partition_free_v1_controller_proposal_horizon_witness_v1"
)
_READINESS_STATUS: Final = "ready_for_cross_branch_fqr_screening"
_FAMILY_REFUSAL_CODE: Final = "no_certified_endpoint"
_CANONICAL_BRANCH_AXES: Final = frozenset(
    {
        ("spectral",),
        ("diagonal_l1",),
        ("spectral", "diagonal_l1"),
    }
)


def _identity(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return result


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def _named_counts(
    values: tuple[tuple[str, int], ...],
    label: str,
) -> tuple[tuple[str, int], ...]:
    result = tuple(
        (
            _identity(name, f"{label} name"),
            _nonnegative_int(count, f"{label} count"),
        )
        for name, count in values
    )
    if not result or len({name for name, _count in result}) != len(result):
        raise ValueError(f"{label} must be nonempty with unique names")
    return result


def _branch_axis(values: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(_identity(value, "registered branch") for value in values)
    if result not in _CANONICAL_BRANCH_AXES:
        raise ValueError(
            "registered branch axis must be a retained canonical F/Q/R axis"
        )
    return result


def _negative_branch_hashes(
    values: tuple[tuple[str, str], ...],
    *,
    registered_branch_axis: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    result = tuple(
        (
            _identity(branch, "negative-evidence branch"),
            _identity(receipt_hash, "negative-evidence receipt hash"),
        )
        for branch, receipt_hash in values
    )
    if tuple(branch for branch, _value in result) != registered_branch_axis:
        raise ValueError(
            "no-candidate evidence must cover the exact registered branch axis"
        )
    if len({value for _branch, value in result}) != len(result):
        raise ValueError("no-candidate branch evidence hashes must be distinct")
    return result


def is_v1_fqr_no_candidate_horizon_proof_hash(value: object) -> bool:
    """Return whether an identity is in the dedicated proof namespace."""

    return type(value) is str and value.startswith(
        FQR_NO_CANDIDATE_HORIZON_PROOF_HASH_PREFIX
    ) and len(value) > len(FQR_NO_CANDIDATE_HORIZON_PROOF_HASH_PREFIX)


@dataclass(frozen=True, slots=True)
class V1FQRNoCandidateHorizonProofReceipt:
    """Compact proof that every frozen endpoint is stably negative.

    This receipt has scientific-horizon evidence authority only.  It cannot
    freeze a candidate, open K/O, mutate a mask, reveal validation rows, or
    make a Pareto claim.  Its scope is the exact registered branch ladders, not
    every legal mask of the same cardinalities.
    """

    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    proposal_backend_identity_hash: str
    generation_index: int
    current_mask_hash: str
    route_axis_hash: str
    mask_lineage_hash: str
    chart_identity_hash: str
    fqr_graph_hash: str
    fqr_policy_hash: str
    fqr_branch_family_design_hash: str
    prefrozen_branch_family_hash: str
    complete_train_population_hash: str
    causal_generation_frame_hash: str
    causal_discovery_stream_hash: str
    causal_discovery_prefix_freeze_hash: str
    causal_certification_stream_hash: str
    sampled_d_sufficient_state_hash: str
    fqr_acquisition_design_hash: str
    fqr_progress_receipt_hash: str
    fqr_readiness_receipt_hash: str
    fqr_accumulator_snapshot_hash: str
    fqr_transaction_chain_hash: str
    proposal_sufficient_state_hash: str
    proposal_sufficient_state_descriptor_hash: str
    fqr_c_state_payload_hash: str
    native_execution_binding_chain_hash: str
    registered_branch_axis: tuple[str, ...]
    branch_negative_evidence_hashes: tuple[tuple[str, str], ...]
    shared_observed_counts_by_stratum: tuple[tuple[str, int], ...]
    readiness_status: str = _READINESS_STATUS
    family_refusal_code: str = _FAMILY_REFUSAL_CODE
    minimum_nonempty_action_count: int = 1
    all_registered_branches_decision_stable: bool = True
    all_frozen_endpoints_lower_refuted: bool = True
    minimum_nonempty_action_registered_in_every_branch: bool = True
    adaptive_optional_stopping_valid: bool = True
    candidate_queue_materialized: bool = False
    calibration_K_opened: bool = False
    paired_oracle_O_opened: bool = False
    mask_mutated: bool = False
    validation_outcomes_observed: bool = False
    candidate_freeze_authority: bool = False
    calibration_or_oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_validation_authority: bool = False
    pareto_claim_authority: bool = False
    schema: str = _PROOF_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "proposal_backend_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "mask_lineage_hash",
            "chart_identity_hash",
            "fqr_graph_hash",
            "fqr_policy_hash",
            "fqr_branch_family_design_hash",
            "prefrozen_branch_family_hash",
            "complete_train_population_hash",
            "causal_generation_frame_hash",
            "causal_discovery_stream_hash",
            "causal_discovery_prefix_freeze_hash",
            "causal_certification_stream_hash",
            "sampled_d_sufficient_state_hash",
            "fqr_acquisition_design_hash",
            "fqr_progress_receipt_hash",
            "fqr_readiness_receipt_hash",
            "fqr_accumulator_snapshot_hash",
            "fqr_transaction_chain_hash",
            "proposal_sufficient_state_hash",
            "proposal_sufficient_state_descriptor_hash",
            "fqr_c_state_payload_hash",
            "native_execution_binding_chain_hash",
            "readiness_status",
            "family_refusal_code",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        object.__setattr__(
            self,
            "generation_index",
            _nonnegative_int(self.generation_index, "generation_index"),
        )
        minimum = _positive_int(
            self.minimum_nonempty_action_count,
            "minimum_nonempty_action_count",
        )
        if minimum != 1:
            raise ValueError(
                "canonical operational saturation requires the one-route rung"
            )
        object.__setattr__(self, "minimum_nonempty_action_count", minimum)
        object.__setattr__(
            self,
            "registered_branch_axis",
            _branch_axis(self.registered_branch_axis),
        )
        object.__setattr__(
            self,
            "shared_observed_counts_by_stratum",
            _named_counts(
                self.shared_observed_counts_by_stratum,
                "shared_observed_counts_by_stratum",
            ),
        )
        object.__setattr__(
            self,
            "branch_negative_evidence_hashes",
            _negative_branch_hashes(
                self.branch_negative_evidence_hashes,
                registered_branch_axis=self.registered_branch_axis,
            ),
        )
        required_true = (
            self.all_registered_branches_decision_stable,
            self.all_frozen_endpoints_lower_refuted,
            self.minimum_nonempty_action_registered_in_every_branch,
            self.adaptive_optional_stopping_valid,
        )
        required_false = (
            self.candidate_queue_materialized,
            self.calibration_K_opened,
            self.paired_oracle_O_opened,
            self.mask_mutated,
            self.validation_outcomes_observed,
            self.candidate_freeze_authority,
            self.calibration_or_oracle_authority,
            self.mask_commit_authority,
            self.terminal_validation_authority,
            self.pareto_claim_authority,
        )
        if (
            any(type(flag) is not bool for flag in (*required_true, *required_false))
            or not all(required_true)
            or any(required_false)
            or self.readiness_status != _READINESS_STATUS
            or self.family_refusal_code != _FAMILY_REFUSAL_CODE
            or self.schema != _PROOF_SCHEMA
        ):
            raise ValueError("F/Q/R no-candidate proof crossed its authority firewall")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix=FQR_NO_CANDIDATE_HORIZON_PROOF_HASH_PREFIX,
        )


@dataclass(frozen=True, slots=True)
class V1ControllerProposalHorizonWitness:
    """Post-decision witness closing one proposal generation without a mask."""

    proof: V1FQRNoCandidateHorizonProofReceipt
    campaign_contract_hash: str
    campaign_binding_hash: str
    controller_run_id: str
    proposal_backend_identity_hash: str
    generation_index: int
    current_mask_hash: str
    route_axis_hash: str
    mask_lineage_hash: str
    pre_stop_controller_state_hash: str
    pre_stop_controller_snapshot_hash: str
    pre_stop_logical_sequence: int
    terminal_proposal_progress_hash: str
    terminal_proposal_frontier_hash: str
    proposal_joint_state_hash: str
    proposal_phase_evidence_state_hash: str
    proposal_sufficient_state_hash: str
    proposal_sufficient_state_descriptor_hash: str
    verified_receipt_graph_hash: str
    freeze_evaluation_receipt_hash: str
    freeze_evidence_receipt_hash: str
    freeze_decision_receipt_hash: str
    post_stop_logical_sequence: int
    stop_reason: str = FQR_NO_CANDIDATE_HORIZON_STOP_REASON
    proposal_progress_retention_required: bool = True
    candidate_queue_materialized: bool = False
    calibration_K_opened: bool = False
    paired_oracle_O_opened: bool = False
    mask_mutated: bool = False
    validation_outcomes_observed: bool = False
    mask_commit_authority: bool = False
    terminal_validation_authority: bool = False
    pareto_claim_authority: bool = False
    schema: str = _WITNESS_SCHEMA

    def __post_init__(self) -> None:
        if type(self.proof) is not V1FQRNoCandidateHorizonProofReceipt:
            raise TypeError("proposal-horizon witness requires the exact F/Q/R proof")
        self.proof.__post_init__()
        for name in (
            "campaign_contract_hash",
            "campaign_binding_hash",
            "controller_run_id",
            "proposal_backend_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "mask_lineage_hash",
            "pre_stop_controller_state_hash",
            "pre_stop_controller_snapshot_hash",
            "terminal_proposal_progress_hash",
            "terminal_proposal_frontier_hash",
            "proposal_joint_state_hash",
            "proposal_phase_evidence_state_hash",
            "proposal_sufficient_state_hash",
            "proposal_sufficient_state_descriptor_hash",
            "verified_receipt_graph_hash",
            "freeze_evaluation_receipt_hash",
            "freeze_evidence_receipt_hash",
            "freeze_decision_receipt_hash",
            "stop_reason",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        generation = _nonnegative_int(self.generation_index, "generation_index")
        pre_sequence = _nonnegative_int(
            self.pre_stop_logical_sequence,
            "pre_stop_logical_sequence",
        )
        post_sequence = _nonnegative_int(
            self.post_stop_logical_sequence,
            "post_stop_logical_sequence",
        )
        if post_sequence != pre_sequence + 1:
            raise ValueError("proposal-horizon stop must advance exactly one sequence")
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "pre_stop_logical_sequence", pre_sequence)
        object.__setattr__(self, "post_stop_logical_sequence", post_sequence)
        proof = self.proof
        if (
            proof.campaign_contract_hash != self.campaign_contract_hash
            or proof.campaign_binding_hash != self.campaign_binding_hash
            or proof.controller_run_id != self.controller_run_id
            or proof.proposal_backend_identity_hash
            != self.proposal_backend_identity_hash
            or proof.generation_index != generation
            or proof.current_mask_hash != self.current_mask_hash
            or proof.route_axis_hash != self.route_axis_hash
            or proof.mask_lineage_hash != self.mask_lineage_hash
            or proof.proposal_sufficient_state_hash
            != self.proposal_sufficient_state_hash
            or proof.proposal_sufficient_state_descriptor_hash
            != self.proposal_sufficient_state_descriptor_hash
        ):
            raise ValueError("proposal-horizon witness changed its proof coordinates")
        required_true = (self.proposal_progress_retention_required,)
        required_false = (
            self.candidate_queue_materialized,
            self.calibration_K_opened,
            self.paired_oracle_O_opened,
            self.mask_mutated,
            self.validation_outcomes_observed,
            self.mask_commit_authority,
            self.terminal_validation_authority,
            self.pareto_claim_authority,
        )
        if (
            self.stop_reason != FQR_NO_CANDIDATE_HORIZON_STOP_REASON
            or any(type(flag) is not bool for flag in (*required_true, *required_false))
            or not all(required_true)
            or any(required_false)
            or self.schema != _WITNESS_SCHEMA
        ):
            raise ValueError("proposal-horizon witness crossed its authority firewall")

    @property
    def witness_hash(self) -> str:
        return content_hash(
            self,
            prefix=FQR_NO_CANDIDATE_HORIZON_WITNESS_HASH_PREFIX,
        )


def v1_fqr_no_candidate_freeze_verification_hash(
    *,
    proof: V1FQRNoCandidateHorizonProofReceipt,
    verified_receipt_graph_hash: str,
    freeze_criterion_hash: str,
    sufficient_state_hash: str,
) -> str:
    """Bind an unchanged freeze-evaluation schema to the typed horizon proof."""

    if type(proof) is not V1FQRNoCandidateHorizonProofReceipt:
        raise TypeError("F/Q/R horizon verification requires the exact proof")
    return content_hash(
        {
            "schema": "partition_free_v1_fqr_no_candidate_freeze_verification_v1",
            "proof_receipt_hash": proof.receipt_hash,
            "verified_receipt_graph_hash": _identity(
                verified_receipt_graph_hash,
                "verified_receipt_graph_hash",
            ),
            "freeze_criterion_hash": _identity(
                freeze_criterion_hash,
                "freeze_criterion_hash",
            ),
            "sufficient_state_hash": _identity(
                sufficient_state_hash,
                "sufficient_state_hash",
            ),
            "criterion_satisfied": False,
            "candidate_identity_hash": None,
        },
        prefix="partition-free-v1-fqr-no-candidate-freeze-verification-",
    )


def v1_fqr_no_candidate_freeze_evaluation_hash(
    *,
    proof: V1FQRNoCandidateHorizonProofReceipt,
    verification_hash: str,
) -> str:
    """Bind the ordinary negative evaluation to the same compact proof."""

    if type(proof) is not V1FQRNoCandidateHorizonProofReceipt:
        raise TypeError("F/Q/R horizon evaluation requires the exact proof")
    return content_hash(
        {
            "schema": "partition_free_v1_fqr_no_candidate_freeze_evaluation_v1",
            "proof_receipt_hash": proof.receipt_hash,
            "verification_hash": _identity(verification_hash, "verification_hash"),
            "criterion_satisfied": False,
            "authority": "stable_negative_fqr_horizon_only",
            "candidate_freeze_authority": False,
            "calibration_K_open_authority": False,
            "paired_oracle_O_open_authority": False,
            "mask_commit_authority": False,
        },
        prefix="partition-free-v1-fqr-no-candidate-freeze-evaluation-",
    )


__all__ = [
    "FQR_NO_CANDIDATE_HORIZON_PROOF_HASH_PREFIX",
    "FQR_NO_CANDIDATE_HORIZON_STOP_REASON",
    "FQR_NO_CANDIDATE_HORIZON_WITNESS_HASH_PREFIX",
    "V1ControllerProposalHorizonWitness",
    "V1FQRNoCandidateHorizonProofReceipt",
    "is_v1_fqr_no_candidate_horizon_proof_hash",
    "v1_fqr_no_candidate_freeze_evaluation_hash",
    "v1_fqr_no_candidate_freeze_verification_hash",
]
