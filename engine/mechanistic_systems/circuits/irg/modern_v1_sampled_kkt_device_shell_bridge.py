"""Compact sampled-D bridge for device-shell KKT endpoint identities.

This seam binds an already executed two-branch device-shell KKT solve and two
complete hard-F/Q replays to the array-free pre-C branch-family identity.  It
does not run the selector, replay F/Q, open C, or materialize an endpoint.

The durable graph contains only content-addressed receipts and scalar/tuple
coordinates.  The separate execution context owns the live label handles and
the array-backed KKT problem.  Those ephemeral objects are deliberately
nonserializable, noncopyable, and unavailable through a general-purpose
accessor.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from numbers import Integral
from typing import Any, Final, NoReturn

from .contracts import content_hash
from .modern_v1_device_shell_fq_replay import (
    V1DeviceShellHardFQReplayReceipt,
)
from .modern_v1_device_shell_fq_replay_cuda import (
    V1DeviceShellHardFQCUDAReplayReceipt,
)
from .modern_v1_device_shell_ladder import (
    V1DeviceShellBorrowOperation,
    V1DeviceShellLabelLease,
)
from .modern_v1_fqr_branch_family import (
    V1FQRDeviceShellBranchBinding,
    V1FQRDeviceShellPreFrozenBranchFamily,
    V1FQRDeviceShellSourceStabilityBinding,
    bind_v1_fqr_device_shell_source_stability,
    freeze_v1_fqr_device_shell_branch_family,
)
from .modern_v1_fqr_adaptive_rank import (
    FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA,
    V1FQRAdaptiveRankPolicy,
    V1FQRAdaptiveRankReplayLedger,
    V1FQRAdaptiveRankReplayRecord,
    V1FQRAdaptiveRankStabilityCertificate,
    V1FQRAdaptiveRankDecisionCertificate,
    V1FQREndogenousRankDPrefixV2,
    V1FQREndogenousRankNextAction,
    V1FQREndogenousRankNoCCode,
    V1FQREndogenousRankNoCRefusalV2,
    V1FQREndogenousRankPolicyV2,
    V1FQREndogenousRankSameVJPFanoutV2,
    certify_v1_fqr_adaptive_rank_stability,
)
from .modern_v1_fqr_adaptive_rank_derivation import (
    V1FQRAdaptiveRankDeviceShellDerivedDecision,
)
from .modern_v1_literal_j_operator_census import (
    V1LiteralJOperatorCensusReceipt,
)
from .modern_v1_literal_j_projection_inference import (
    V1LiteralJFixedDirectionProjectionReceipt,
)
from .modern_v1_relaxed_kkt import (
    V1RelaxedKKTAdaptiveRankBranchWitness,
    V1RelaxedKKTBranch,
    V1RelaxedKKTClosureMode,
    V1RelaxedKKTDecisionLevelRobustInvarianceCertificate,
    V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    V1RelaxedKKTDeviceShellExecutionBundle,
    V1RelaxedKKTDeviceShellExecutionReceipt,
    V1RelaxedKKTGeometryUncertaintySet,
    V1RelaxedKKTPolicy,
    V1RelaxedKKTProblem,
    V1RelaxedKKTResult,
)
from .modern_v1_sampled_acquisition_adapters import (
    V1SampledPreFreezeGeometrySource,
)
from .modern_v1_sampled_kkt_bridge import V1SampledKKTSourceProblemBundle


SAMPLED_KKT_DEVICE_SHELL_GRAPH_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_device_shell_graph_v1"
)
SAMPLED_KKT_DEVICE_SHELL_GRAPH_V2_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_device_shell_graph_v2"
)
SAMPLED_KKT_DEVICE_SHELL_PROPOSAL_GRAPH_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_device_shell_proposal_graph_v3"
)
SAMPLED_KKT_DEVICE_SHELL_DECISION_STABLE_GRAPH_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_device_shell_decision_stable_graph_v4"
)
SAMPLED_KKT_DEVICE_SHELL_RANK_BINDING_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_device_shell_rank_binding_v1"
)
SAMPLED_KKT_DEVICE_SHELL_D_TO_C_ADMISSION_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_device_shell_D_to_C_admission_v1"
)
SAMPLED_KKT_DEVICE_SHELL_D_TO_C_ADMISSION_V2_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_device_shell_D_to_C_admission_v2"
)
SAMPLED_KKT_ADAPTIVE_EXECUTION_RANK_FREEZE_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_adaptive_execution_rank_freeze_v1"
)
SAMPLED_KKT_PROPOSAL_RANK_POLICY_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_proposal_rank_policy_v1"
)
SAMPLED_KKT_DEVICE_SHELL_PROPOSAL_FREEZE_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_device_shell_proposal_freeze_for_C_v1"
)
SAMPLED_KKT_DEVICE_SHELL_ADAPTIVE_FREEZE_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_device_shell_adaptive_freeze_for_C_v2"
)
SAMPLED_KKT_DEVICE_SHELL_D_SOURCE_COORDINATE_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_device_shell_registered_D_source_coordinate_v1"
)
SAMPLED_KKT_DEVICE_SHELL_D_SOURCE_COORDINATE_V2_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_device_shell_registered_D_source_coordinate_v2"
)
SAMPLED_KKT_ENDOGENOUS_RANK_POLICY_V2_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_endogenous_rank_policy_v2"
)
SAMPLED_KKT_SOURCE_DECISION_IMAGE_V2_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_source_decision_image_v2"
)
SAMPLED_KKT_ENDOGENOUS_RANK_FREEZE_V2_SCHEMA: Final = (
    "partition_free_v1_sampled_kkt_endogenous_execution_rank_freeze_v2"
)
SAMPLED_KKT_DEVICE_SHELL_AUTHORITY_SCOPE: Final = (
    "sampled_D_device_shell_identity_and_complete_reference_hard_FQ_only_"
    "no_face_shell_C_K_O_calibration_oracle_commit_terminal_or_pareto_"
    "authority_v1"
)
_GRAPH_PREFIX: Final = "partition-free-v1-sampled-kkt-device-shell-graph-"
_RANK_BINDING_PREFIX: Final = (
    "partition-free-v1-sampled-kkt-device-shell-rank-binding-"
)
_D_TO_C_ADMISSION_PREFIX: Final = (
    "partition-free-v1-sampled-kkt-device-shell-D-to-C-admission-"
)
_ADAPTIVE_EXECUTION_RANK_FREEZE_PREFIX: Final = (
    "partition-free-v1-sampled-kkt-adaptive-execution-rank-freeze-"
)
_PROPOSAL_RANK_POLICY_PREFIX: Final = (
    "partition-free-v1-sampled-kkt-proposal-rank-policy-"
)
_PROPOSAL_FREEZE_PREFIX: Final = (
    "partition-free-v1-sampled-kkt-device-shell-proposal-freeze-for-C-"
)
_D_SOURCE_COORDINATE_PREFIX: Final = (
    "partition-free-v1-sampled-kkt-device-shell-D-source-coordinate-"
)
_ENDOGENOUS_RANK_POLICY_V2_PREFIX: Final = (
    "partition-free-v1-sampled-kkt-endogenous-rank-policy-v2-"
)
_SOURCE_DECISION_IMAGE_V2_PREFIX: Final = (
    "partition-free-v1-sampled-kkt-source-decision-image-v2-"
)
_ENDOGENOUS_RANK_FREEZE_V2_PREFIX: Final = (
    "partition-free-v1-sampled-kkt-endogenous-rank-freeze-v2-"
)
_D_TO_C_ADMISSION_ISSUANCE_TOKEN: Final = object()
_ADAPTIVE_EXECUTION_RANK_FREEZE_ISSUANCE_TOKEN: Final = object()
_PROPOSAL_RANK_POLICY_ISSUANCE_TOKEN: Final = object()
_PROPOSAL_FREEZE_ISSUANCE_TOKEN: Final = object()
_D_SOURCE_COORDINATE_ISSUANCE_TOKEN: Final = object()
_ENDOGENOUS_RANK_POLICY_V2_ISSUANCE_TOKEN: Final = object()
_SOURCE_DECISION_IMAGE_V2_ISSUANCE_TOKEN: Final = object()
_ENDOGENOUS_RANK_FREEZE_V2_ISSUANCE_TOKEN: Final = object()

DeviceShellHardFQReplayReceipt = (
    V1DeviceShellHardFQReplayReceipt | V1DeviceShellHardFQCUDAReplayReceipt
)


class V1SampledKKTDeviceShellBridgeRefusalCode(str, Enum):
    """Typed lifecycle and authority refusals for the ephemeral context."""

    CONTEXT_EVICTED = "context_evicted"
    CONTEXT_HAS_ACTIVE_LEASES = "context_has_active_leases"
    UNAUTHORIZED_LABEL_OPERATION = "unauthorized_label_operation"
    SOURCE_WITNESS_MISSING = "source_witness_missing"
    CUDA_HARD_FQ_REPLAY_MISSING = "cuda_hard_fq_replay_missing"
    RANK_REPLAY_MISMATCH = "rank_replay_mismatch"
    D_TO_C_STABILITY_MISSING = "d_to_c_stability_missing"
    REGISTERED_D_SOURCE_MISSING = "registered_d_source_missing"
    PRIVATE_PRIMAL_DIRECTION_LINEAGE_UNAVAILABLE = (
        "private_primal_direction_lineage_unavailable"
    )
    ROBUST_KKT_SENSITIVITY_MISSING = "robust_kkt_sensitivity_missing"


class V1SampledKKTDeviceShellBridgeRefusal(RuntimeError):
    """The ephemeral bridge context cannot perform the requested operation."""

    def __init__(
        self,
        code: V1SampledKKTDeviceShellBridgeRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SampledKKTDeviceShellBridgeRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SampledKKTDeviceShellBridgeRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be one exact integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class V1SampledKKTProposalRankPolicy:
    """Outcome-independent rank choice for the stochastic D/KKT proposal kernel.

    This is deliberately distinct from adaptive-rank *stability* authority.
    It chooses the largest rank admitted by the pre-outcome HBM forecast, then
    freezes that choice before D or C outcomes exist.  The resulting proposal
    may depend on D, but neither this policy nor D uncertainty can authorize an
    irreversible mask decision; independent C F/Q/R and K/O remain required.
    """

    policy_id: str
    source_memory_forecast_policy_hash: str
    route_count: int
    admitted_rank_ladder: tuple[int, ...]
    selected_rank: int
    _issuance_token: InitVar[object]
    selection_rule: str = "maximum_preoutcome_hbm_admitted_rank_v1"
    registered_before_D_or_C_outcomes: bool = True
    empirical_outcomes_used_for_rank_selection: bool = False
    candidate_composition_may_depend_on_D: bool = True
    D_uncertainty_has_no_commit_authority: bool = True
    C_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_KKT_PROPOSAL_RANK_POLICY_SCHEMA
    _policy_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _PROPOSAL_RANK_POLICY_ISSUANCE_TOKEN:
            raise PermissionError(
                "sampled-KKT proposal-rank policy requires its registered issuer"
            )
        object.__setattr__(
            self,
            "policy_id",
            _identity(self.policy_id, name="proposal-rank policy id"),
        )
        object.__setattr__(
            self,
            "source_memory_forecast_policy_hash",
            _identity(
                self.source_memory_forecast_policy_hash,
                name="source memory-forecast policy hash",
            ),
        )
        route_count = _nonnegative_int(self.route_count, name="route count")
        if route_count == 0:
            raise ValueError("proposal-rank route count must be positive")
        ranks = tuple(
            _nonnegative_int(value, name="admitted proposal rank")
            for value in self.admitted_rank_ladder
        )
        if (
            not ranks
            or any(value == 0 for value in ranks)
            or any(value > route_count for value in ranks)
            or ranks != tuple(sorted(set(ranks)))
        ):
            raise ValueError(
                "admitted proposal ranks must be positive, unique, increasing, "
                "and no larger than the route dimension"
            )
        selected = _nonnegative_int(self.selected_rank, name="selected proposal rank")
        if selected == 0 or selected != max(ranks):
            raise ValueError(
                "proposal rank must be the maximum pre-outcome HBM-admitted rank"
            )
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "admitted_rank_ladder", ranks)
        object.__setattr__(self, "selected_rank", selected)
        if self.selection_rule != "maximum_preoutcome_hbm_admitted_rank_v1":
            raise ValueError("unsupported sampled-KKT proposal-rank selection rule")
        flags = (
            self.registered_before_D_or_C_outcomes,
            self.empirical_outcomes_used_for_rank_selection,
            self.candidate_composition_may_depend_on_D,
            self.D_uncertainty_has_no_commit_authority,
            self.C_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("proposal-rank policy flags must be exact booleans")
        if flags != (True, False, True, True, False, False, False, False, False, False):
            raise PermissionError("proposal-rank policy crossed its authority boundary")
        if self.schema != SAMPLED_KKT_PROPOSAL_RANK_POLICY_SCHEMA:
            raise ValueError("sampled-KKT proposal-rank policy schema is invalid")
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "source_memory_forecast_policy_hash": (
                self.source_memory_forecast_policy_hash
            ),
            "route_count": self.route_count,
            "admitted_rank_ladder": self.admitted_rank_ladder,
            "selected_rank": self.selected_rank,
            "selection_rule": self.selection_rule,
            "registered_before_D_or_C_outcomes": (
                self.registered_before_D_or_C_outcomes
            ),
            "empirical_outcomes_used_for_rank_selection": (
                self.empirical_outcomes_used_for_rank_selection
            ),
            "candidate_composition_may_depend_on_D": (
                self.candidate_composition_may_depend_on_D
            ),
            "D_uncertainty_has_no_commit_authority": (
                self.D_uncertainty_has_no_commit_authority
            ),
            "C_stream_authority": self.C_stream_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
        }

    @property
    def policy_hash(self) -> str:
        cached = self._policy_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_PROPOSAL_RANK_POLICY_PREFIX)
        cached = self._policy_hash
        if cached is not None and cached != observed:
            raise ValueError("sampled-KKT proposal-rank policy was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("sampled-KKT proposal-rank policy hash does not match")
        object.__setattr__(self, "_policy_hash", observed)
        return observed


def register_v1_sampled_kkt_proposal_rank_policy(
    *,
    policy: V1FQRAdaptiveRankPolicy,
) -> V1SampledKKTProposalRankPolicy:
    """Freeze the maximum memory-admitted rank before empirical outcomes."""

    if type(policy) is not V1FQRAdaptiveRankPolicy:
        raise TypeError("proposal-rank registration requires the exact HBM policy")
    source_hash = policy.assert_integrity()
    if (
        not policy.registered_before_D_outcomes
        or policy.empirical_outcomes_used_for_filtering
        or not policy.admitted_rank_ladder
        or tuple(item.rank for item in policy.memory_forecasts if item.admitted)
        != policy.admitted_rank_ladder
    ):
        raise PermissionError(
            "proposal rank requires an outcome-independent, forecast-replayed HBM policy"
        )
    dimension_admitted = tuple(
        rank for rank in policy.admitted_rank_ladder if rank <= policy.route_count
    )
    if not dimension_admitted:
        raise ValueError(
            "sampled-KKT proposal execution requires at least one HBM-admitted "
            "rank no larger than the route dimension"
        )
    return V1SampledKKTProposalRankPolicy(
        policy_id=f"{policy.policy_id}:proposal-kernel",
        source_memory_forecast_policy_hash=source_hash,
        route_count=policy.route_count,
        admitted_rank_ladder=dimension_admitted,
        selected_rank=max(dimension_admitted),
        _issuance_token=_PROPOSAL_RANK_POLICY_ISSUANCE_TOKEN,
    )


def _relaxed_problem_perturbation_exact_zero(
    uncertainty: V1RelaxedKKTGeometryUncertaintySet,
) -> bool:
    """Return true only when ``A``, ``M_J``, and FD define one exact problem.

    Fixed-direction ``M_J z`` intervals do not control how the optimizer moves
    when the relaxed problem is perturbed.  Until a source-issued robust KKT
    solution-sensitivity theorem exists, only an exact-zero perturbation can
    promote the nominal face/shell witness toward executable C admission.
    Solver, gate, and ratio numerical radii stay under the existing outward
    margin certificates for the already registered problem.
    """

    return all(
        value == 0.0
        for value in (
            uncertainty.corrected_work_A_sampling_linf_radius,
            uncertainty.corrected_work_A_deterministic_linf_radius,
            uncertainty.literal_j_response_operator_MJ_sampling_linf_adjoint_radius,
            uncertainty.literal_j_response_operator_MJ_deterministic_linf_adjoint_radius,
            uncertainty.fd_q_residual_deterministic_linf_adjoint_radius,
        )
    )


def _relaxed_problem_sensitivity_closed(
    *,
    uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    witnesses: tuple[V1RelaxedKKTAdaptiveRankBranchWitness, ...],
    decision_level_certificates: tuple[
        V1RelaxedKKTDecisionLevelRobustInvarianceCertificate, ...
    ] = (),
) -> bool:
    if _relaxed_problem_perturbation_exact_zero(uncertainty):
        return True
    certificate_by_branch = {
        value.branch: value for value in decision_level_certificates
    }
    for witness in witnesses:
        if (
            type(witness) is not V1RelaxedKKTAdaptiveRankBranchWitness
            or not witness.decision_invariance_closed
        ):
            return False
        mode = witness.closure_mode
        if mode is V1RelaxedKKTClosureMode.FULL_SOLUTION_SENSITIVITY:
            if witness.relaxed_solution_sensitivity_certificate_hash is None:
                return False
            continue
        if mode is V1RelaxedKKTClosureMode.DECISION_LEVEL_SHELL_INVARIANCE:
            certificate = certificate_by_branch.get(witness.branch)
            if (
                type(certificate)
                is not V1RelaxedKKTDecisionLevelRobustInvarianceCertificate
                or certificate.certificate_hash
                != witness.decision_level_robust_invariance_certificate_hash
                or certificate.branch is not witness.branch
                or certificate.finite_population_geometry_uncertainty_set_hash
                != uncertainty.uncertainty_hash
            ):
                return False
            certificate.assert_integrity(certificate.certificate_hash)
            continue
        return False
    return bool(witnesses)


def _relaxed_problem_closure_modes(
    *,
    uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    witnesses: tuple[V1RelaxedKKTAdaptiveRankBranchWitness, ...],
) -> tuple[V1RelaxedKKTClosureMode, ...]:
    if _relaxed_problem_perturbation_exact_zero(uncertainty):
        return tuple(V1RelaxedKKTClosureMode.EXACT_ZERO for _ in witnesses)
    modes = tuple(witness.closure_mode for witness in witnesses)
    if any(value is None for value in modes):
        return ()
    return tuple(value for value in modes if value is not None)


def _graph_bound_source_sensitivity_closed(
    *,
    graph: V1SampledKKTDeviceShellGraph,
    binding: V1FQRDeviceShellSourceStabilityBinding,
) -> bool:
    """Replay one source binding's decision closure, including v4 theorem hashes."""

    if binding.relaxed_problem_sensitivity_closed:
        return True
    witness = binding.adaptive_rank_witness
    if (
        graph.schema != SAMPLED_KKT_DEVICE_SHELL_DECISION_STABLE_GRAPH_SCHEMA
        or witness.closure_mode
        is not V1RelaxedKKTClosureMode.DECISION_LEVEL_SHELL_INVARIANCE
        or witness.decision_level_robust_invariance_certificate_hash is None
    ):
        return False
    branch_offset = (
        0 if witness.branch is V1RelaxedKKTBranch.SPECTRAL else 1
    )
    hashes = graph.decision_level_robust_invariance_certificate_hashes
    return (
        len(hashes) == 2
        and hashes[branch_offset]
        == witness.decision_level_robust_invariance_certificate_hash
    )


def _replayed_problem_hash(problem: V1RelaxedKKTProblem) -> str:
    """Recompute the problem identity instead of trusting its cached digest."""

    return content_hash(
        {
            "schema": problem.schema,
            "problem_id": problem.problem_id,
            "resource": problem.resource,
            "corrected_work": problem.corrected_work,
            "low_rank_factor": problem.low_rank_factor,
            "residual_diagonal": problem.residual_diagonal,
            "residual_spectral_bound": problem.residual_spectral_bound,
            "eligible_mask": problem.eligible_mask,
            "work_budget": problem.work_budget,
            "quadratic_budget": problem.quadratic_budget,
            "route_axis_hash": problem.route_axis_hash,
            "source_geometry_hash": problem.source_geometry_hash,
            "block_capacities": problem.block_capacities,
            "input_array_policy": problem.input_array_policy,
            "corrected_work_envelope_id": problem.corrected_work_envelope_id,
        },
        prefix="partition-free-v1-relaxed-kkt-problem-",
    )


def _validated_replay_hash(replay: DeviceShellHardFQReplayReceipt) -> str:
    """Replay nested receipt invariants before admitting a durable hash."""

    try:
        if type(replay) not in {
            V1DeviceShellHardFQReplayReceipt,
            V1DeviceShellHardFQCUDAReplayReceipt,
        }:
            raise TypeError("hard-F/Q replay has an unsupported receipt type")
        for endpoint in replay.endpoints:
            endpoint.__post_init__()
        replay.__post_init__()
        return replay.receipt_hash
    except (TypeError, ValueError, PermissionError) as error:
        raise ValueError(f"hard-F/Q replay does not replay structurally: {error}") from error


def _replay_action_status_axis_hash(
    replay: DeviceShellHardFQReplayReceipt,
) -> str:
    """One status-axis identity shared by reference and CUDA replay receipts."""

    _validated_replay_hash(replay)
    return content_hash(
        tuple(
            (
                endpoint.endpoint_label,
                endpoint.packed_action_hash,
                endpoint.resulting_mask_hash,
                endpoint.status,
                endpoint.receipt_hash,
            )
            for endpoint in replay.endpoints
        ),
        prefix="partition-free-v1-device-shell-hard-fq-status-axis-",
    )


def _validated_execution_receipt_hash(
    receipt: V1RelaxedKKTDeviceShellExecutionReceipt,
) -> str:
    """Replay every nested array-free shell receipt before hashing it."""

    try:
        for branch in (receipt.spectral, receipt.diagonal_l1):
            for action in branch.frozen_ladder.actions:
                action.__post_init__()
            branch.frozen_ladder.__post_init__()
            branch.__post_init__()
        receipt.__post_init__()
        return receipt.receipt_hash
    except (TypeError, ValueError, PermissionError) as error:
        raise ValueError(
            f"device-shell execution receipt does not replay structurally: {error}"
        ) from error


def _validate_replay_branch(
    *,
    replay: DeviceShellHardFQReplayReceipt,
    branch: V1RelaxedKKTBranch,
    branch_receipt: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    execution_receipt: V1RelaxedKKTDeviceShellExecutionReceipt,
    family_binding: V1FQRDeviceShellBranchBinding,
    problem_hash: str,
    problem_id: str,
    chart_identity_hash: str,
    route_axis_hash: str,
    route_count: int,
    current_mask_hash: str,
) -> None:
    if type(replay) not in {
        V1DeviceShellHardFQReplayReceipt,
        V1DeviceShellHardFQCUDAReplayReceipt,
    }:
        raise TypeError("device-shell graph requires exact hard-F/Q replay receipts")
    if type(branch_receipt) is not V1RelaxedKKTDeviceShellBranchIdentityReceipt:
        raise TypeError("device-shell graph requires exact branch identity receipts")
    if type(family_binding) is not V1FQRDeviceShellBranchBinding:
        raise TypeError("device-shell graph requires exact compact branch bindings")
    _validated_replay_hash(replay)
    ladder = branch_receipt.frozen_ladder
    common_invalid = (
        replay.branch is not branch
        or branch_receipt.branch is not branch
        or family_binding.branch is not branch
        or replay.problem_hash != problem_hash
        or replay.problem_id != problem_id
        or replay.source_geometry_hash != chart_identity_hash
        or replay.source_branch_identity_receipt_hash != branch_receipt.receipt_hash
        or replay.frozen_ladder_receipt_hash != ladder.receipt_hash
        or replay.source_rounding_hash != branch_receipt.source_rounding_hash
        or replay.route_axis_hash != route_axis_hash
        or replay.route_count != route_count
        or replay.parent_mask_hash != current_mask_hash
        or replay.endpoint_count != ladder.endpoint_count
        or replay.label_commitment != ladder.label_commitment
        or replay.eligibility_commitment != ladder.eligibility_commitment
        or replay.block_capacity_hash != ladder.block_capacity_hash
        or replay.block_legality_scope != ladder.block_legality_scope
        or not replay.complete_registered_ladder_replayed
        or replay.nonmonotone_q_early_stop_used
        or not replay.label_membership_used_directly
    )
    if common_invalid:
        raise ValueError("device-shell hard-F/Q replay left its exact branch source")
    if type(replay) is V1DeviceShellHardFQReplayReceipt:
        if (
            replay.source_identity_kind != "branch_identity"
            or replay.source_identity_hash != branch_receipt.receipt_hash
            or replay.cuda_device_native_complete
        ):
            raise ValueError("reference hard-F/Q replay left its exact source")
    else:
        if (
            replay.device_shell_execution_receipt_hash
            != execution_receipt.receipt_hash
            or replay.device_shell_execution_design_hash
            != execution_receipt.device_shell_execution_design_hash
            or replay.cuda_execution_design_hash
            != execution_receipt.cuda_execution_design_hash
            or replay.base_result_hash != execution_receipt.base_result_hash
            or replay.policy_hash != execution_receipt.policy_hash
            or replay.numerical_policy_hash != execution_receipt.policy_hash
            or replay.branch_result_hash == ""
            or replay.branch_kkt_receipt_hash == ""
            or not replay.stable_physical_tie_order
            or replay.ordinary_route_order_d2h
            or replay.ordinary_removed_indices_materialized
            or replay.ordinary_hard_masks_materialized
            or replay.ordinary_hard_actions_materialized
        ):
            raise ValueError("CUDA hard-F/Q replay left its exact execution source")
    if replay.frozen_action_receipt_hashes != tuple(
        action.receipt_hash for action in ladder.actions
    ):
        raise ValueError("device-shell hard-F/Q replay omitted a frozen action")
    if len(family_binding.endpoints) != replay.endpoint_count:
        raise ValueError("compact family and hard-F/Q replay endpoint axes differ")
    for action, endpoint, compact in zip(
        ladder.actions,
        replay.endpoints,
        family_binding.endpoints,
        strict=True,
    ):
        source_axis = (
            action.endpoint_label,
            action.cumulative_removed_count,
            action.receipt_hash,
            action.packed_action_hash,
            action.removed_route_index_hash,
            action.resulting_mask_packed_sha256,
            action.resulting_mask_hash,
        )
        replay_axis = (
            endpoint.endpoint_label,
            endpoint.cumulative_removed_count,
            endpoint.frozen_action_receipt_hash,
            endpoint.packed_action_hash,
            endpoint.removed_route_index_hash,
            endpoint.resulting_mask_packed_sha256,
            endpoint.resulting_mask_hash,
        )
        compact_axis = (
            compact.endpoint_index,
            compact.cumulative_removed_count,
            compact.source_action_receipt_hash,
            compact.packed_action_hash,
            compact.removed_route_index_hash,
            compact.resulting_mask_packed_sha256,
            compact.resulting_mask_hash,
        )
        if source_axis != replay_axis or source_axis != compact_axis:
            raise ValueError("packed endpoint action axes do not close exactly")
        if compact.registered_resource != endpoint.resource_nominal:
            raise ValueError("compact endpoint resource differs from hard-F/Q replay")


@dataclass(frozen=True, slots=True)
class V1SampledKKTDeviceShellDSourceCoordinate:
    """Array-free coordinate derived from one exact registered-D source.

    Rank records may not self-assert that they replayed raw D.  Canonical
    device-shell graphs carry this private-issuer receipt, derived transiently
    from the exact sampled pre-freeze source, and every rank-local record must
    replay each coordinate below.
    """

    sampled_geometry_source_hash: str
    source_admission_hash: str
    source_discovery_receipt_hash: str
    source_assembly_receipt_hash: str
    source_geometry_result_hash: str
    model_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    current_mask_hash: str
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    route_count: int
    proposal_generation: int
    observed_stratum_counts: tuple[int, int]
    observed_draw_axis_hash: str
    observed_source_context_axis_hash: str
    source_measurement_axis_hash: str
    sample_weight_hash: str
    corrected_moments_state_hash: str
    fd_design_hash: str
    fd_snapshot_hash: str
    snapshot_numerical_backend_hash: str
    _issuance_token: InitVar[object]
    fd_sketch_rank: int | None = None
    corrected_evidence_stream_identity_hash: str | None = None
    same_capture_and_vjp_execution_proof_hash: str | None = None
    source_derived_from_exact_registered_D: bool = True
    C_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_KKT_DEVICE_SHELL_D_SOURCE_COORDINATE_SCHEMA
    _coordinate_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _D_SOURCE_COORDINATE_ISSUANCE_TOKEN:
            raise PermissionError(
                "registered-D source coordinates require the exact-source issuer"
            )
        for name in (
            "sampled_geometry_source_hash",
            "source_admission_hash",
            "source_discovery_receipt_hash",
            "source_assembly_receipt_hash",
            "source_geometry_result_hash",
            "model_identity_hash",
            "literal_jacobian_stream_identity_hash",
            "current_mask_hash",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
            "observed_draw_axis_hash",
            "observed_source_context_axis_hash",
            "source_measurement_axis_hash",
            "sample_weight_hash",
            "corrected_moments_state_hash",
            "fd_design_hash",
            "fd_snapshot_hash",
            "snapshot_numerical_backend_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        route_count = _nonnegative_int(self.route_count, name="route count")
        if route_count == 0:
            raise ValueError("registered-D route count must be positive")
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        counts = tuple(self.observed_stratum_counts)
        if (
            len(counts) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) <= 0
                for value in counts
            )
        ):
            raise ValueError("registered D requires one positive two-stratum prefix")
        object.__setattr__(
            self,
            "observed_stratum_counts",
            (int(counts[0]), int(counts[1])),
        )
        rank = self.fd_sketch_rank
        if rank is not None:
            rank = _nonnegative_int(rank, name="registered-D FD sketch rank")
            if rank == 0:
                raise ValueError("registered-D FD sketch rank must be positive")
        object.__setattr__(self, "fd_sketch_rank", rank)
        corrected_identity = self.corrected_evidence_stream_identity_hash
        same_vjp_proof = self.same_capture_and_vjp_execution_proof_hash
        if corrected_identity is not None:
            corrected_identity = _identity(
                corrected_identity,
                name="corrected-evidence stream identity hash",
            )
        if same_vjp_proof is not None:
            same_vjp_proof = _identity(
                same_vjp_proof,
                name="same-capture/VJP execution proof hash",
            )
        if (corrected_identity is None) is not (same_vjp_proof is None):
            raise ValueError(
                "registered-D corrected/J provenance must include both the "
                "corrected stream and same-capture/VJP proof"
            )
        object.__setattr__(
            self,
            "corrected_evidence_stream_identity_hash",
            corrected_identity,
        )
        object.__setattr__(
            self,
            "same_capture_and_vjp_execution_proof_hash",
            same_vjp_proof,
        )
        flags = (
            self.source_derived_from_exact_registered_D,
            self.C_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("registered-D coordinate flags must be exact booleans")
        if flags != (True, False, False, False, False, False, False):
            raise PermissionError("registered-D coordinate crossed its authority boundary")
        if self.schema not in {
            SAMPLED_KKT_DEVICE_SHELL_D_SOURCE_COORDINATE_SCHEMA,
            SAMPLED_KKT_DEVICE_SHELL_D_SOURCE_COORDINATE_V2_SCHEMA,
        }:
            raise ValueError("registered-D source-coordinate schema is invalid")
        if (
            self.schema == SAMPLED_KKT_DEVICE_SHELL_D_SOURCE_COORDINATE_SCHEMA
        ) != (corrected_identity is None):
            raise ValueError(
                "registered-D source-coordinate schema and corrected/J provenance differ"
            )
        self.assert_integrity()

    @property
    def raw_prefix_coordinate(self) -> tuple[object, ...]:
        """The population/order coordinates that must stay identical by rank."""

        return (
            self.current_mask_hash,
            self.physical_route_axis_hash,
            self.route_count,
            self.proposal_generation,
            self.observed_stratum_counts,
            self.observed_draw_axis_hash,
            self.observed_source_context_axis_hash,
            self.source_measurement_axis_hash,
            self.sample_weight_hash,
            self.corrected_evidence_stream_identity_hash,
            self.literal_jacobian_stream_identity_hash,
            self.same_capture_and_vjp_execution_proof_hash,
        )

    @property
    def rank_invariant_raw_prefix_coordinate(self) -> tuple[object, ...]:
        """Return the registered raw-D identity before rank-local FD work.

        The same-capture/VJP proof is deliberately *not* part of this
        coordinate.  It certifies one rank-local transaction and therefore
        changes when the identical raw prefix is reexecuted with another FD
        design.  Every v2 coordinate must still carry its own nonempty proof;
        equality here means only that model/chart, population membership,
        sealed order, source measurements, weights, and corrected/J stream
        semantics are unchanged.  Historical ``raw_prefix_coordinate`` bytes
        and their stricter same-proof comparison remain available for replay.
        """

        if self.schema != SAMPLED_KKT_DEVICE_SHELL_D_SOURCE_COORDINATE_V2_SCHEMA:
            return self.raw_prefix_coordinate
        return (
            self.model_identity_hash,
            self.current_mask_hash,
            self.physical_route_axis_hash,
            self.ordered_route_axis_hash,
            self.route_count,
            self.proposal_generation,
            self.observed_stratum_counts,
            self.observed_draw_axis_hash,
            self.observed_source_context_axis_hash,
            self.source_measurement_axis_hash,
            self.sample_weight_hash,
            self.corrected_moments_state_hash,
            self.corrected_evidence_stream_identity_hash,
            self.literal_jacobian_stream_identity_hash,
        )

    @property
    def rank_invariant_raw_prefix_hash(self) -> str:
        """Compact identity used by adaptive reexecution and cache keys."""

        return content_hash(
            self.rank_invariant_raw_prefix_coordinate,
            prefix="partition-free-v1-registered-D-rank-invariant-prefix-",
        )

    def _payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "sampled_geometry_source_hash": self.sampled_geometry_source_hash,
            "source_admission_hash": self.source_admission_hash,
            "source_discovery_receipt_hash": self.source_discovery_receipt_hash,
            "source_assembly_receipt_hash": self.source_assembly_receipt_hash,
            "source_geometry_result_hash": self.source_geometry_result_hash,
            "model_identity_hash": self.model_identity_hash,
            "literal_jacobian_stream_identity_hash": (
                self.literal_jacobian_stream_identity_hash
            ),
            "current_mask_hash": self.current_mask_hash,
            "physical_route_axis_hash": self.physical_route_axis_hash,
            "ordered_route_axis_hash": self.ordered_route_axis_hash,
            "route_count": self.route_count,
            "proposal_generation": self.proposal_generation,
            "observed_stratum_counts": self.observed_stratum_counts,
            "observed_draw_axis_hash": self.observed_draw_axis_hash,
            "observed_source_context_axis_hash": (
                self.observed_source_context_axis_hash
            ),
            "source_measurement_axis_hash": self.source_measurement_axis_hash,
            "sample_weight_hash": self.sample_weight_hash,
            "corrected_moments_state_hash": self.corrected_moments_state_hash,
            "fd_design_hash": self.fd_design_hash,
            "fd_snapshot_hash": self.fd_snapshot_hash,
            **(
                {"fd_sketch_rank": self.fd_sketch_rank}
                if self.fd_sketch_rank is not None
                else {}
            ),
            "snapshot_numerical_backend_hash": (
                self.snapshot_numerical_backend_hash
            ),
            "source_derived_from_exact_registered_D": (
                self.source_derived_from_exact_registered_D
            ),
            "C_stream_authority": self.C_stream_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
        }
        if self.schema == SAMPLED_KKT_DEVICE_SHELL_D_SOURCE_COORDINATE_V2_SCHEMA:
            payload["corrected_evidence_stream_identity_hash"] = (
                self.corrected_evidence_stream_identity_hash
            )
            payload["same_capture_and_vjp_execution_proof_hash"] = (
                self.same_capture_and_vjp_execution_proof_hash
            )
        return payload

    @property
    def coordinate_hash(self) -> str:
        cached = self._coordinate_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_D_SOURCE_COORDINATE_PREFIX)
        cached = self._coordinate_hash
        if cached is not None and cached != observed:
            raise ValueError("registered-D source coordinate was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("registered-D source-coordinate hash does not match")
        object.__setattr__(self, "_coordinate_hash", observed)
        return observed


def _bind_registered_d_source_coordinate(
    source: V1SampledPreFreezeGeometrySource,
) -> V1SampledKKTDeviceShellDSourceCoordinate:
    if type(source) is not V1SampledPreFreezeGeometrySource:
        raise TypeError("canonical device-shell D requires the exact sampled source")
    admission = source.admission
    receipt = admission.receipt
    dual_chart = admission.dual_chart_stream_binding
    if dual_chart is None or len(dual_chart) != 3:
        _refuse(
            V1SampledKKTDeviceShellBridgeRefusalCode.REGISTERED_D_SOURCE_MISSING,
            "exact registered D lacks its same-capture literal-J stream binding",
        )
    return V1SampledKKTDeviceShellDSourceCoordinate(
        sampled_geometry_source_hash=source.source_hash,
        source_admission_hash=admission.admission_hash,
        source_discovery_receipt_hash=receipt.receipt_hash,
        source_assembly_receipt_hash=source.receipt.receipt_hash,
        source_geometry_result_hash=source.geometry_result.result_hash,
        model_identity_hash=receipt.model_identity_hash,
        corrected_evidence_stream_identity_hash=dual_chart[0],
        literal_jacobian_stream_identity_hash=dual_chart[1],
        same_capture_and_vjp_execution_proof_hash=dual_chart[2],
        current_mask_hash=receipt.current_mask_hash,
        physical_route_axis_hash=receipt.physical_route_axis_hash,
        ordered_route_axis_hash=receipt.ordered_route_axis_hash,
        route_count=receipt.route_count,
        proposal_generation=receipt.proposal_generation,
        observed_stratum_counts=receipt.observed_stratum_counts,
        observed_draw_axis_hash=receipt.observed_draw_axis_hash,
        observed_source_context_axis_hash=(
            receipt.observed_source_context_axis_hash
        ),
        source_measurement_axis_hash=receipt.source_measurement_axis_hash,
        sample_weight_hash=receipt.sample_weight_hash,
        corrected_moments_state_hash=receipt.corrected_moments_state_hash,
        fd_design_hash=receipt.fd_design_hash,
        fd_snapshot_hash=receipt.fd_snapshot_hash,
        snapshot_numerical_backend_hash=receipt.snapshot_numerical_backend_hash,
        fd_sketch_rank=source.admission.fd_snapshot.sketch_rank,
        schema=SAMPLED_KKT_DEVICE_SHELL_D_SOURCE_COORDINATE_V2_SCHEMA,
        _issuance_token=_D_SOURCE_COORDINATE_ISSUANCE_TOKEN,
    )


LiteralJFixedPrimalSourceReceipt = (
    V1LiteralJOperatorCensusReceipt | V1LiteralJFixedDirectionProjectionReceipt
)


def _validated_private_primal_direction_descriptor_axis(
    census: LiteralJFixedPrimalSourceReceipt,
) -> str:
    """Return canonical private-primal lineage or emit the typed no-C refusal."""

    if type(census) not in (
        V1LiteralJOperatorCensusReceipt,
        V1LiteralJFixedDirectionProjectionReceipt,
    ):
        raise TypeError("fixed-primal lineage requires an exact literal-J receipt")
    census.assert_integrity()
    descriptor_axis_hash = getattr(
        census,
        "private_primal_direction_descriptor_axis_hash",
        None,
    )
    lineage_available = getattr(
        census,
        "private_primal_direction_lineage_available",
        False,
    )
    if lineage_available is not True or type(descriptor_axis_hash) is not str:
        _refuse(
            V1SampledKKTDeviceShellBridgeRefusalCode.PRIVATE_PRIMAL_DIRECTION_LINEAGE_UNAVAILABLE,
            "the literal-J direction axis lacks canonical branch/private-"
            "primal descriptors; generic direction ids cannot authorize C",
        )
    return _identity(
        descriptor_axis_hash,
        name="private-primal direction descriptor axis hash",
    )


@dataclass(frozen=True, slots=True)
class V1SampledKKTDeviceShellGraph:
    """Array-free durable lineage for the compact pre-C device-shell family."""

    sampled_geometry_source_hash: str
    problem_hash: str
    problem_id: str
    policy: V1RelaxedKKTPolicy
    base_result_hash: str
    execution_receipt: V1RelaxedKKTDeviceShellExecutionReceipt
    route_axis_hash: str
    route_count: int
    current_mask_hash: str
    chart_identity_hash: str
    proposal_generation: int
    discovery_prefix_hash: str
    discovery_evidence_hash: str
    spectral_hard_fq_replay: DeviceShellHardFQReplayReceipt
    diagonal_l1_hard_fq_replay: DeviceShellHardFQReplayReceipt
    branch_family: V1FQRDeviceShellPreFrozenBranchFamily
    spectral_action_status_axis_hash: str
    diagonal_l1_action_status_axis_hash: str
    registered_d_source_coordinate: (
        V1SampledKKTDeviceShellDSourceCoordinate | None
    ) = None
    source_problem_bundle_hash: str | None = None
    literal_j_fixed_primal_census_receipt_hash: str | None = None
    spectral_source_stability: V1FQRDeviceShellSourceStabilityBinding | None = None
    diagonal_l1_source_stability: (
        V1FQRDeviceShellSourceStabilityBinding | None
    ) = None
    decision_level_robust_invariance_certificate_hashes: tuple[str, ...] = ()
    kkt_fd_rank: int | None = None
    both_hard_fq_replays_complete: bool = True
    source_built_face_shell_witness_available: bool = False
    cuda_device_native_fq_replay_available: bool = False
    c_stream_outcomes_seen: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SAMPLED_KKT_DEVICE_SHELL_AUTHORITY_SCOPE
    schema: str = SAMPLED_KKT_DEVICE_SHELL_GRAPH_SCHEMA
    _graph_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "sampled_geometry_source_hash",
            "problem_hash",
            "problem_id",
            "base_result_hash",
            "route_axis_hash",
            "current_mask_hash",
            "chart_identity_hash",
            "discovery_prefix_hash",
            "discovery_evidence_hash",
            "spectral_action_status_axis_hash",
            "diagonal_l1_action_status_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _nonnegative_int(self.route_count, name="route count"),
        )
        if self.route_count == 0:
            raise ValueError("route count must be positive")
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        if type(self.policy) is not V1RelaxedKKTPolicy:
            raise TypeError("device-shell graph requires an exact KKT policy")
        if type(self.execution_receipt) is not V1RelaxedKKTDeviceShellExecutionReceipt:
            raise TypeError("device-shell graph requires an exact execution receipt")
        if type(self.branch_family) is not V1FQRDeviceShellPreFrozenBranchFamily:
            raise TypeError("device-shell graph requires an exact compact family")
        source_bindings = (
            self.spectral_source_stability,
            self.diagonal_l1_source_stability,
        )
        if any(
            value is not None
            and type(value) is not V1FQRDeviceShellSourceStabilityBinding
            for value in source_bindings
        ):
            raise TypeError("device-shell graph source bindings have invalid types")
        source_complete = all(value is not None for value in source_bindings)
        decision_hashes = tuple(
            _identity(value, name="decision-level theorem hash")
            for value in self.decision_level_robust_invariance_certificate_hashes
        )
        if decision_hashes and len(decision_hashes) != 2:
            raise ValueError(
                "decision-stable graph requires the canonical two-branch theorem axis"
            )
        object.__setattr__(
            self,
            "decision_level_robust_invariance_certificate_hashes",
            decision_hashes,
        )
        if decision_hashes and not source_complete:
            raise PermissionError(
                "decision-level theorem hashes cannot float free of both "
                "source-stability bindings"
            )
        census_hash = self.literal_j_fixed_primal_census_receipt_hash
        if census_hash is not None:
            census_hash = _identity(
                census_hash,
                name="literal-J fixed-primal census receipt hash",
            )
        object.__setattr__(
            self,
            "literal_j_fixed_primal_census_receipt_hash",
            census_hash,
        )
        d_source = self.registered_d_source_coordinate
        if d_source is not None and type(d_source) is not (
            V1SampledKKTDeviceShellDSourceCoordinate
        ):
            raise TypeError("device-shell graph requires an exact registered-D source")
        d_source_complete = d_source is not None
        source_problem_hash = self.source_problem_bundle_hash
        if source_problem_hash is not None:
            source_problem_hash = _identity(
                source_problem_hash,
                name="sampled source-problem bundle hash",
            )
        object.__setattr__(
            self,
            "source_problem_bundle_hash",
            source_problem_hash,
        )
        cuda_complete = (
            type(self.spectral_hard_fq_replay)
            is V1DeviceShellHardFQCUDAReplayReceipt
            and type(self.diagonal_l1_hard_fq_replay)
            is V1DeviceShellHardFQCUDAReplayReceipt
        )
        flags = (
            self.both_hard_fq_replays_complete,
            self.source_built_face_shell_witness_available,
            self.cuda_device_native_fq_replay_available,
            self.c_stream_outcomes_seen,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("device-shell graph authority fields must be exact booleans")
        expected_flags = (
            True,
            source_complete,
            cuda_complete,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        )
        if flags != expected_flags:
            raise PermissionError("device-shell graph crossed its pre-C authority boundary")
        proposal_only = d_source_complete and cuda_complete and not source_complete
        if source_complete and not d_source_complete:
            raise PermissionError(
                "canonical source witnesses require their exact registered-D source"
            )
        if (source_problem_hash is not None) is not d_source_complete:
            raise PermissionError(
                "registered-D device-shell graphs require their exact source problem"
            )
        if d_source_complete and not (source_complete or proposal_only):
            raise PermissionError(
                "registered-D source requires either source stability or a "
                "CUDA-native proposal graph"
            )
        if (census_hash is not None) is not source_complete:
            raise PermissionError(
                "canonical source witnesses require their exact literal-J census"
            )
        rank = self.kkt_fd_rank
        if rank is not None:
            rank = _nonnegative_int(rank, name="KKT FD rank")
            if rank == 0:
                raise ValueError("KKT FD rank must be positive")
        object.__setattr__(self, "kkt_fd_rank", rank)
        if proposal_only:
            if rank is None:
                raise ValueError("proposal-only device-shell graph requires its FD rank")
            expected_schema = SAMPLED_KKT_DEVICE_SHELL_PROPOSAL_GRAPH_SCHEMA
        elif source_complete:
            expected_schema = (
                SAMPLED_KKT_DEVICE_SHELL_DECISION_STABLE_GRAPH_SCHEMA
                if decision_hashes
                else SAMPLED_KKT_DEVICE_SHELL_GRAPH_V2_SCHEMA
            )
        else:
            if rank is not None:
                raise ValueError("legacy device-shell graph cannot carry proposal rank")
            expected_schema = SAMPLED_KKT_DEVICE_SHELL_GRAPH_SCHEMA
        if self.schema != expected_schema or self.authority_scope != SAMPLED_KKT_DEVICE_SHELL_AUTHORITY_SCOPE:
            raise ValueError("unsupported device-shell sampled graph contract")
        self.assert_integrity()

    def _validate_closure(self) -> None:
        execution = self.execution_receipt
        family = self.branch_family
        execution_hash = _validated_execution_receipt_hash(execution)
        family.assert_integrity()
        if (
            execution.problem_hash != self.problem_hash
            or execution.policy_hash != self.policy.policy_hash
            or execution.base_result_hash != self.base_result_hash
            or family.execution_receipt_hash != execution_hash
            or family.problem_hash != self.problem_hash
            or family.policy_hash != self.policy.policy_hash
            or family.base_result_hash != self.base_result_hash
            or family.route_axis_hash != self.route_axis_hash
            or family.route_count != self.route_count
            or family.current_mask_hash != self.current_mask_hash
            or family.chart_identity_hash != self.chart_identity_hash
            or family.proposal_generation != self.proposal_generation
            or family.discovery_prefix_hash != self.discovery_prefix_hash
            or family.discovery_evidence_hash != self.discovery_evidence_hash
            or execution.adaptive_shell_witness_available
            or self.spectral_action_status_axis_hash
            != _replay_action_status_axis_hash(self.spectral_hard_fq_replay)
            or self.diagonal_l1_action_status_axis_hash
            != _replay_action_status_axis_hash(self.diagonal_l1_hard_fq_replay)
        ):
            raise ValueError("sampled chart, KKT execution, replay, and family do not close")
        _validate_replay_branch(
            replay=self.spectral_hard_fq_replay,
            branch=V1RelaxedKKTBranch.SPECTRAL,
            branch_receipt=execution.spectral,
            execution_receipt=execution,
            family_binding=family.spectral,
            problem_hash=self.problem_hash,
            problem_id=self.problem_id,
            chart_identity_hash=self.chart_identity_hash,
            route_axis_hash=self.route_axis_hash,
            route_count=self.route_count,
            current_mask_hash=self.current_mask_hash,
        )
        _validate_replay_branch(
            replay=self.diagonal_l1_hard_fq_replay,
            branch=V1RelaxedKKTBranch.DIAGONAL_L1,
            branch_receipt=execution.diagonal_l1,
            execution_receipt=execution,
            family_binding=family.diagonal_l1,
            problem_hash=self.problem_hash,
            problem_id=self.problem_id,
            chart_identity_hash=self.chart_identity_hash,
            route_axis_hash=self.route_axis_hash,
            route_count=self.route_count,
            current_mask_hash=self.current_mask_hash,
        )
        source_bindings = (
            self.spectral_source_stability,
            self.diagonal_l1_source_stability,
        )
        if self.source_built_face_shell_witness_available:
            d_source = self.registered_d_source_coordinate
            if d_source is None:
                raise ValueError("source-stable graph omitted its registered-D source")
            d_source.assert_integrity()
            if (
                d_source.sampled_geometry_source_hash
                != self.sampled_geometry_source_hash
                or d_source.current_mask_hash != self.current_mask_hash
                or d_source.physical_route_axis_hash != self.route_axis_hash
                or d_source.route_count != self.route_count
                or d_source.proposal_generation != self.proposal_generation
            ):
                raise ValueError("registered-D source left the graph chart")
            if any(value is None for value in source_bindings):
                raise ValueError("source-stable graph omitted a branch source binding")
            for expected_branch, binding, family_binding, replay in zip(
                (V1RelaxedKKTBranch.SPECTRAL, V1RelaxedKKTBranch.DIAGONAL_L1),
                source_bindings,
                (family.spectral, family.diagonal_l1),
                (self.spectral_hard_fq_replay, self.diagonal_l1_hard_fq_replay),
                strict=True,
            ):
                assert binding is not None
                binding.assert_integrity()
                if (
                    binding.branch_binding.binding_hash != family_binding.binding_hash
                    or binding.adaptive_rank_witness.branch is not expected_branch
                    or binding.cuda_hard_fq_replay is not replay
                    or binding.literal_j_fixed_primal_census.receipt_hash
                    != self.literal_j_fixed_primal_census_receipt_hash
                ):
                    raise ValueError("source-stability binding left the graph branch")
            if self.decision_level_robust_invariance_certificate_hashes:
                witness_hashes = tuple(
                    binding.adaptive_rank_witness.decision_level_robust_invariance_certificate_hash
                    for binding in source_bindings
                    if binding is not None
                )
                if (
                    self.schema
                    != SAMPLED_KKT_DEVICE_SHELL_DECISION_STABLE_GRAPH_SCHEMA
                    or witness_hashes
                    != self.decision_level_robust_invariance_certificate_hashes
                ):
                    raise ValueError(
                        "decision-level theorem axis left its source witnesses"
                    )
            elif self.schema == SAMPLED_KKT_DEVICE_SHELL_DECISION_STABLE_GRAPH_SCHEMA:
                raise ValueError("decision-stable graph omitted its theorem axis")
        elif any(value is not None for value in source_bindings):
            raise ValueError("unstable source bindings cannot be retained as authority")
        elif self.registered_d_source_coordinate is not None:
            d_source = self.registered_d_source_coordinate
            d_source.assert_integrity()
            if (
                self.schema != SAMPLED_KKT_DEVICE_SHELL_PROPOSAL_GRAPH_SCHEMA
                or not self.cuda_device_native_fq_replay_available
                or d_source.sampled_geometry_source_hash
                != self.sampled_geometry_source_hash
                or d_source.current_mask_hash != self.current_mask_hash
                or d_source.physical_route_axis_hash != self.route_axis_hash
                or d_source.route_count != self.route_count
                or d_source.proposal_generation != self.proposal_generation
                or d_source.fd_sketch_rank != self.kkt_fd_rank
            ):
                raise ValueError("proposal-only registered-D source left its graph")

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "sampled_geometry_source_hash": self.sampled_geometry_source_hash,
            "problem_hash": self.problem_hash,
            "problem_id": self.problem_id,
            "policy_hash": self.policy.policy_hash,
            "base_result_hash": self.base_result_hash,
            "execution_receipt_hash": _validated_execution_receipt_hash(
                self.execution_receipt
            ),
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "current_mask_hash": self.current_mask_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "proposal_generation": self.proposal_generation,
            "discovery_prefix_hash": self.discovery_prefix_hash,
            "discovery_evidence_hash": self.discovery_evidence_hash,
            "spectral_hard_fq_replay_hash": _validated_replay_hash(
                self.spectral_hard_fq_replay
            ),
            "diagonal_l1_hard_fq_replay_hash": (
                _validated_replay_hash(self.diagonal_l1_hard_fq_replay)
            ),
            "branch_family_hash": self.branch_family.family_hash,
            "registered_d_source_coordinate_hash": (
                None
                if self.registered_d_source_coordinate is None
                else self.registered_d_source_coordinate.coordinate_hash
            ),
            **(
                {"source_problem_bundle_hash": self.source_problem_bundle_hash}
                if self.source_problem_bundle_hash is not None
                else {}
            ),
            "literal_j_fixed_primal_census_receipt_hash": (
                self.literal_j_fixed_primal_census_receipt_hash
            ),
            "spectral_source_stability_hash": (
                None
                if self.spectral_source_stability is None
                else self.spectral_source_stability.binding_hash
            ),
            "diagonal_l1_source_stability_hash": (
                None
                if self.diagonal_l1_source_stability is None
                else self.diagonal_l1_source_stability.binding_hash
            ),
            **(
                {
                    "decision_level_robust_invariance_certificate_hashes": (
                        self.decision_level_robust_invariance_certificate_hashes
                    )
                }
                if self.schema
                == SAMPLED_KKT_DEVICE_SHELL_DECISION_STABLE_GRAPH_SCHEMA
                else {}
            ),
            **(
                {"kkt_fd_rank": self.kkt_fd_rank}
                if self.schema == SAMPLED_KKT_DEVICE_SHELL_PROPOSAL_GRAPH_SCHEMA
                else {}
            ),
            "spectral_action_status_axis_hash": (
                self.spectral_action_status_axis_hash
            ),
            "diagonal_l1_action_status_axis_hash": (
                self.diagonal_l1_action_status_axis_hash
            ),
            "both_hard_fq_replays_complete": self.both_hard_fq_replays_complete,
            "source_built_face_shell_witness_available": (
                self.source_built_face_shell_witness_available
            ),
            "cuda_device_native_fq_replay_available": (
                self.cuda_device_native_fq_replay_available
            ),
            "c_stream_outcomes_seen": self.c_stream_outcomes_seen,
            "c_stream_authority": self.c_stream_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "authority_scope": self.authority_scope,
        }

    @property
    def graph_hash(self) -> str:
        cached = self._graph_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self._validate_closure()
        observed = content_hash(self._payload(), prefix=_GRAPH_PREFIX)
        cached = self._graph_hash
        if cached is not None and cached != observed:
            raise ValueError("sampled device-shell graph was tampered after hashing")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("sampled device-shell graph hash does not match")
        object.__setattr__(self, "_graph_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledKKTDeviceShellProposalFreezeForC:
    """Freeze a CUDA D/KKT candidate family for independent C measurement.

    The certificate authorizes only a lifetime-guarded borrow of the already
    frozen branch labels for directional C evaluation.  Historical v1 freezes
    bind the maximum pre-outcome HBM-admitted proposal rank and deliberately do
    not assert D stability.  Canonical v2 freezes instead bind the first
    identical-D decision-stable rank through
    :class:`V1SampledKKTAdaptiveExecutionRankFreeze`.  Neither form carries
    F/Q/R, calibration, oracle, commit, terminal, or Pareto authority; safety
    still requires independent C F/Q/R and the paired K/O oracle.
    """

    graph: V1SampledKKTDeviceShellGraph
    proposal_rank_policy: V1SampledKKTProposalRankPolicy
    _issuance_token: InitVar[object]
    adaptive_execution_rank_freeze: V1SampledKKTAdaptiveExecutionRankFreeze | None = None
    registered_D_source_bound: bool = True
    candidate_family_frozen_before_C: bool = True
    complete_cuda_hard_fq_replay: bool = True
    D_sampling_fd_kkt_uncertainty_affects_efficiency_only: bool = True
    independent_C_FQR_required: bool = True
    candidate_specific_K_and_paired_O_required: bool = True
    robust_kkt_sensitivity_required_for_C: bool = False
    C_direction_borrow_authority: bool = True
    C_FQR_decision_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_KKT_DEVICE_SHELL_PROPOSAL_FREEZE_SCHEMA
    _certificate_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _PROPOSAL_FREEZE_ISSUANCE_TOKEN:
            raise PermissionError("proposal freeze requires its registered issuer")
        graph = self.graph
        rank_policy = self.proposal_rank_policy
        if type(graph) is not V1SampledKKTDeviceShellGraph:
            raise TypeError("proposal freeze requires an exact device-shell graph")
        if type(rank_policy) is not V1SampledKKTProposalRankPolicy:
            raise TypeError("proposal freeze requires an exact proposal-rank policy")
        graph.assert_integrity()
        rank_policy.assert_integrity()
        d_source = graph.registered_d_source_coordinate
        common_closed = (
            type(d_source) is V1SampledKKTDeviceShellDSourceCoordinate
            and graph.both_hard_fq_replays_complete
            and graph.cuda_device_native_fq_replay_available
            and type(graph.spectral_hard_fq_replay)
            is V1DeviceShellHardFQCUDAReplayReceipt
            and type(graph.diagonal_l1_hard_fq_replay)
            is V1DeviceShellHardFQCUDAReplayReceipt
            and graph.route_count == rank_policy.route_count
            and not graph.c_stream_outcomes_seen
        )
        adaptive = self.adaptive_execution_rank_freeze
        if adaptive is None:
            mode_closed = (
                graph.schema == SAMPLED_KKT_DEVICE_SHELL_PROPOSAL_GRAPH_SCHEMA
                and not graph.source_built_face_shell_witness_available
                and graph.spectral_source_stability is None
                and graph.diagonal_l1_source_stability is None
                and graph.kkt_fd_rank == rank_policy.selected_rank
                and self.robust_kkt_sensitivity_required_for_C is False
                and self.schema == SAMPLED_KKT_DEVICE_SHELL_PROPOSAL_FREEZE_SCHEMA
            )
        else:
            if type(adaptive) is not V1SampledKKTAdaptiveExecutionRankFreeze:
                raise TypeError(
                    "adaptive proposal freeze requires the exact execution-rank freeze"
                )
            adaptive.assert_integrity()
            mode_closed = (
                adaptive.memory_admission_policy.policy_hash
                == rank_policy.policy_hash
                and adaptive.final_graph.graph_hash == graph.graph_hash
                and adaptive.final_execution_rank == d_source.fd_sketch_rank
                and graph.source_built_face_shell_witness_available
                and graph.spectral_source_stability is not None
                and graph.diagonal_l1_source_stability is not None
                and graph.schema
                in {
                    SAMPLED_KKT_DEVICE_SHELL_GRAPH_V2_SCHEMA,
                    SAMPLED_KKT_DEVICE_SHELL_DECISION_STABLE_GRAPH_SCHEMA,
                }
                and self.robust_kkt_sensitivity_required_for_C is True
                and self.schema == SAMPLED_KKT_DEVICE_SHELL_ADAPTIVE_FREEZE_SCHEMA
            )
        if not common_closed or not mode_closed:
            raise ValueError(
                "proposal graph, registered D, CUDA hard-F/Q replay, and frozen "
                "execution-rank owner do not close"
            )
        d_source.assert_integrity()
        flags = (
            self.registered_D_source_bound,
            self.candidate_family_frozen_before_C,
            self.complete_cuda_hard_fq_replay,
            self.D_sampling_fd_kkt_uncertainty_affects_efficiency_only,
            self.independent_C_FQR_required,
            self.candidate_specific_K_and_paired_O_required,
            self.robust_kkt_sensitivity_required_for_C,
            self.C_direction_borrow_authority,
            self.C_FQR_decision_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("proposal-freeze authority fields must be exact booleans")
        expected_robust = adaptive is not None
        if flags != (
            True,
            True,
            True,
            True,
            True,
            True,
            expected_robust,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            raise PermissionError("proposal freeze crossed its C-direction boundary")
        self.assert_integrity()

    @property
    def frozen_graph_hash(self) -> str:
        return self.graph.graph_hash

    @property
    def frozen_candidate_family_hash(self) -> str:
        return self.graph.branch_family.family_hash

    @property
    def execution_rank(self) -> int:
        adaptive = self.adaptive_execution_rank_freeze
        return (
            self.proposal_rank_policy.selected_rank
            if adaptive is None
            else adaptive.final_execution_rank
        )

    def _payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "frozen_graph_hash": self.frozen_graph_hash,
            "frozen_candidate_family_hash": self.frozen_candidate_family_hash,
            "registered_D_source_coordinate_hash": (
                self.graph.registered_d_source_coordinate.coordinate_hash
                if self.graph.registered_d_source_coordinate is not None
                else None
            ),
            "proposal_rank_policy_hash": self.proposal_rank_policy.policy_hash,
            "registered_D_source_bound": self.registered_D_source_bound,
            "candidate_family_frozen_before_C": (
                self.candidate_family_frozen_before_C
            ),
            "complete_cuda_hard_fq_replay": self.complete_cuda_hard_fq_replay,
            "D_sampling_fd_kkt_uncertainty_affects_efficiency_only": (
                self.D_sampling_fd_kkt_uncertainty_affects_efficiency_only
            ),
            "independent_C_FQR_required": self.independent_C_FQR_required,
            "candidate_specific_K_and_paired_O_required": (
                self.candidate_specific_K_and_paired_O_required
            ),
            "robust_kkt_sensitivity_required_for_C": (
                self.robust_kkt_sensitivity_required_for_C
            ),
            "C_direction_borrow_authority": self.C_direction_borrow_authority,
            "C_FQR_decision_authority": self.C_FQR_decision_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
        }
        if self.schema == SAMPLED_KKT_DEVICE_SHELL_PROPOSAL_FREEZE_SCHEMA:
            payload["selected_rank"] = self.proposal_rank_policy.selected_rank
        else:
            assert self.adaptive_execution_rank_freeze is not None
            payload["execution_rank"] = self.execution_rank
            payload["adaptive_execution_rank_freeze_hash"] = (
                self.adaptive_execution_rank_freeze.freeze_hash
            )
            payload["maximum_hbm_admitted_rank"] = (
                self.adaptive_execution_rank_freeze.maximum_hbm_admitted_rank
            )
        return payload

    @property
    def certificate_hash(self) -> str:
        cached = self._certificate_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.graph.assert_integrity()
        self.proposal_rank_policy.assert_integrity()
        if self.adaptive_execution_rank_freeze is not None:
            self.adaptive_execution_rank_freeze.assert_integrity()
        observed = content_hash(self._payload(), prefix=_PROPOSAL_FREEZE_PREFIX)
        cached = self._certificate_hash
        if cached is not None and cached != observed:
            raise ValueError("sampled-KKT proposal freeze was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("sampled-KKT proposal-freeze hash does not match")
        object.__setattr__(self, "_certificate_hash", observed)
        return observed


def freeze_v1_sampled_kkt_device_shell_proposal_for_independent_c(
    *,
    graph: V1SampledKKTDeviceShellGraph,
    rank_policy: V1SampledKKTProposalRankPolicy,
) -> V1SampledKKTDeviceShellProposalFreezeForC:
    """Issue C-direction borrow authority after a proposal-only graph freezes."""

    if type(graph) is not V1SampledKKTDeviceShellGraph:
        raise TypeError("proposal freeze requires the exact device-shell graph")
    if type(rank_policy) is not V1SampledKKTProposalRankPolicy:
        raise TypeError("proposal freeze requires the exact proposal-rank policy")
    return V1SampledKKTDeviceShellProposalFreezeForC(
        graph=graph,
        proposal_rank_policy=rank_policy,
        _issuance_token=_PROPOSAL_FREEZE_ISSUANCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledKKTDeviceShellRankBinding:
    """One rank-local source graph bound to its identical-D replay record.

    The binding is authority-inert even when its source-derived decision is
    stable: only the adaptive-prefix issuer may decide that this is the final
    replay and open C.  An unstable binding must nevertheless be a complete,
    typed replay.  In particular, ambiguity is a source-derived status, not a
    license to omit a branch, an endpoint, or the raw-D lineage.
    """

    graph: V1SampledKKTDeviceShellGraph
    derived_decision: V1FQRAdaptiveRankDeviceShellDerivedDecision
    replay_record: V1FQRAdaptiveRankReplayRecord
    C_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = SAMPLED_KKT_DEVICE_SHELL_RANK_BINDING_SCHEMA
    _binding_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        graph = self.graph
        derived = self.derived_decision
        record = self.replay_record
        if type(graph) is not V1SampledKKTDeviceShellGraph:
            raise TypeError("rank binding requires an exact device-shell graph")
        if type(derived) is not V1FQRAdaptiveRankDeviceShellDerivedDecision:
            raise TypeError("rank binding requires an exact source-derived decision")
        if type(record) is not V1FQRAdaptiveRankReplayRecord:
            raise TypeError("rank binding requires an exact adaptive replay record")
        graph_hash = graph.assert_integrity()
        record_hash = record.assert_integrity()
        del record_hash
        decision = derived.decision_certificate
        decision.assert_integrity()
        d_source = graph.registered_d_source_coordinate
        source_bindings = (
            graph.spectral_source_stability,
            graph.diagonal_l1_source_stability,
        )
        typed_branch_derivations = tuple(derived.branch_derivations)
        typed_complete_replay = (
            len(typed_branch_derivations) == 2
            and all(value is not None for value in source_bindings)
            and all(
                value is not None
                and value.complete_cuda_hard_fq_replay
                and value.cuda_hard_fq_replay.complete_registered_ladder_replayed
                and not value.cuda_hard_fq_replay.nonmonotone_q_early_stop_used
                and value.cuda_hard_fq_replay.label_membership_used_directly
                for value in source_bindings
            )
            and all(
                branch.authoritative_typed_source_derivation_available
                and branch.authoritative_typed_source_derivation_receipt_hash
                is not None
                for branch in decision.branch_decisions
            )
        )
        source_stable = all(
            value is not None and value.source_stable
            for value in source_bindings
        )
        if (
            not graph.source_built_face_shell_witness_available
            or not graph.cuda_device_native_fq_replay_available
            or not typed_complete_replay
            or d_source is None
            or derived.graph_hash != graph_hash
            or derived.rank != record.rank
            or derived.policy_hash != record.policy_hash
            or decision.certificate_hash
            != record.decision_certificate.certificate_hash
            or decision.rank_local_residual_decision_stable is not source_stable
            or decision.C_stream_open_authority
            is not decision.rank_local_residual_decision_stable
            or record.current_mask_hash != graph.current_mask_hash
            or record.route_axis_hash != graph.route_axis_hash
            or record.route_count != graph.route_count
            or record.proposal_generation != graph.proposal_generation
            or record.sampled_geometry_source_hash
            != graph.sampled_geometry_source_hash
            or record.sampled_geometry_source_hash
            != d_source.sampled_geometry_source_hash
            or record.cumulative_stratum_counts
            != d_source.observed_stratum_counts
            or record.D_context_axis_hash
            != d_source.observed_source_context_axis_hash
            or record.D_draw_axis_hash != d_source.observed_draw_axis_hash
            or record.D_source_measurement_axis_hash
            != d_source.source_measurement_axis_hash
            or record.D_weight_hash != d_source.sample_weight_hash
            or record.fd_design_hash != d_source.fd_design_hash
            or record.fd_snapshot_hash != d_source.fd_snapshot_hash
            or record.kkt_problem_hash != graph.problem_hash
            or record.kkt_result_hash != graph.base_result_hash
            or record.provisional_branch_family_hash
            != graph.branch_family.family_hash
        ):
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.RANK_REPLAY_MISMATCH,
                "rank-local graph, source derivation, and raw-D replay do not close",
            )
        flags = (
            self.C_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("rank-binding authority fields must be exact booleans")
        if any(flags):
            raise PermissionError("a rank-local binding cannot open workflow stages")
        if self.schema != SAMPLED_KKT_DEVICE_SHELL_RANK_BINDING_SCHEMA:
            raise ValueError("device-shell rank-binding schema is invalid")
        self.assert_integrity()

    @property
    def rank(self) -> int:
        return self.replay_record.rank

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "rank": self.rank,
            "graph_hash": self.graph.graph_hash,
            "derived_decision_hash": self.derived_decision.result_hash,
            "replay_record_hash": self.replay_record.record_hash,
            "C_stream_authority": self.C_stream_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
        }

    @property
    def binding_hash(self) -> str:
        cached = self._binding_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_RANK_BINDING_PREFIX)
        cached = self._binding_hash
        if cached is not None and cached != observed:
            raise ValueError("device-shell rank binding was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("device-shell rank-binding hash does not match")
        object.__setattr__(self, "_binding_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SampledKKTDeviceShellDToCAdmissionCertificate:
    """The sole typed device-shell authority to borrow frozen directions for C.

    Historical single-rank exact-geometry certificates retain their byte
    identity.  In a multi-rank prefix, every nonfinal replay is required to be
    a typed, complete, authority-inert ambiguity.  Only the final replay may
    be source-stable in face, shell, and hard-F/Q status and may carry the
    source-derived C-open bit consumed by this issuer.  A nonzero
    relaxed-problem perturbation may close when every affected branch binds a
    source-issued decision-sensitivity theorem; nominal fixed-direction
    witnesses remain diagnostic in either case.
    """

    policy_hash: str
    replay_ledger: V1FQRAdaptiveRankReplayLedger
    stability_certificate: V1FQRAdaptiveRankStabilityCertificate
    rank_bindings: tuple[V1SampledKKTDeviceShellRankBinding, ...]
    _issuance_token: InitVar[object]
    closure_modes_by_rank_and_branch: tuple[
        tuple[V1RelaxedKKTClosureMode, V1RelaxedKKTClosureMode], ...
    ] = ()
    exact_registered_D_sources_bound: bool = True
    identical_registered_D_prefix_replayed: bool = True
    source_branch_witnesses_stable: bool = True
    nominal_fixed_direction_witnesses_diagnostic_only: bool = True
    relaxed_problem_perturbation_exact_zero: bool = True
    complete_cuda_hard_fq_replay: bool = True
    hard_fq_status_axes_stable: bool = True
    adaptive_rank_decision_stable: bool = True
    C_stream_open_authority: bool = True
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_KKT_DEVICE_SHELL_D_TO_C_ADMISSION_SCHEMA
    _certificate_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _D_TO_C_ADMISSION_ISSUANCE_TOKEN:
            raise PermissionError("D-to-C admission requires the registered issuer")
        object.__setattr__(
            self,
            "policy_hash",
            _identity(self.policy_hash, name="adaptive-rank policy hash"),
        )
        ledger = self.replay_ledger
        stability = self.stability_certificate
        bindings = tuple(self.rank_bindings)
        if type(ledger) is not V1FQRAdaptiveRankReplayLedger:
            raise TypeError("D-to-C admission requires the exact replay ledger")
        if type(stability) is not V1FQRAdaptiveRankStabilityCertificate:
            raise TypeError("D-to-C admission requires the exact stability certificate")
        if (
            not bindings
            or any(type(value) is not V1SampledKKTDeviceShellRankBinding for value in bindings)
        ):
            raise TypeError("D-to-C admission requires exact rank bindings")
        ledger_hash = ledger.assert_integrity()
        stability_hash = stability.assert_integrity()
        del stability_hash
        for binding in bindings:
            binding.assert_integrity()
        d_sources = tuple(
            binding.graph.registered_d_source_coordinate for binding in bindings
        )
        exact_d_sources = all(
            type(value) is V1SampledKKTDeviceShellDSourceCoordinate
            for value in d_sources
        )
        identical_raw_prefix = exact_d_sources and len(
            {
                value.rank_invariant_raw_prefix_coordinate
                for value in d_sources
                if value is not None
            }
        ) == 1
        nonfinal_bindings = bindings[:-1]
        final_binding = bindings[-1]
        nonfinal_rank_local_ambiguity = all(
            not binding.derived_decision.decision_certificate.rank_local_residual_decision_stable
            and not binding.derived_decision.decision_certificate.C_stream_open_authority
            and any(
                branch.source_derivation_ambiguity_codes
                for branch in binding.derived_decision.decision_certificate.branch_decisions
            )
            for binding in nonfinal_bindings
        )
        final_decision = final_binding.derived_decision.decision_certificate
        source_witnesses_stable = (
            final_binding.graph.spectral_source_stability is not None
            and final_binding.graph.diagonal_l1_source_stability is not None
            and _graph_bound_source_sensitivity_closed(
                graph=final_binding.graph,
                binding=final_binding.graph.spectral_source_stability,
            )
            and _graph_bound_source_sensitivity_closed(
                graph=final_binding.graph,
                binding=final_binding.graph.diagonal_l1_source_stability,
            )
            and final_binding.graph.spectral_source_stability.routewise_kkt_face_stable
            and final_binding.graph.diagonal_l1_source_stability.routewise_kkt_face_stable
            and final_binding.graph.spectral_source_stability.packed_endpoint_shell_stable
            and final_binding.graph.diagonal_l1_source_stability.packed_endpoint_shell_stable
            and final_binding.graph.spectral_source_stability.complete_cuda_hard_fq_replay
            and final_binding.graph.diagonal_l1_source_stability.complete_cuda_hard_fq_replay
            and final_binding.graph.spectral_source_stability.hard_fq_status_axis_stable
            and final_binding.graph.diagonal_l1_source_stability.hard_fq_status_axis_stable
            and final_decision.rank_local_residual_decision_stable
            and final_decision.C_stream_open_authority
        )
        complete_cuda_fq = all(
            binding.graph.spectral_source_stability is not None
            and binding.graph.diagonal_l1_source_stability is not None
            and binding.graph.spectral_source_stability.complete_cuda_hard_fq_replay
            and binding.graph.diagonal_l1_source_stability.complete_cuda_hard_fq_replay
            for binding in bindings
        )
        relaxed_problem_exact_zero = all(
            binding.graph.spectral_source_stability is not None
            and binding.graph.diagonal_l1_source_stability is not None
            and binding.graph.spectral_source_stability.relaxed_problem_perturbation_exact_zero
            and binding.graph.diagonal_l1_source_stability.relaxed_problem_perturbation_exact_zero
            for binding in bindings
        )
        relaxed_problem_sensitivity_closed = all(
            binding.graph.spectral_source_stability is not None
            and binding.graph.diagonal_l1_source_stability is not None
            and _graph_bound_source_sensitivity_closed(
                graph=binding.graph,
                binding=binding.graph.spectral_source_stability,
            )
            and _graph_bound_source_sensitivity_closed(
                graph=binding.graph,
                binding=binding.graph.diagonal_l1_source_stability,
            )
            for binding in bindings
        )
        derived_closure_modes = tuple(
            (
                binding.graph.spectral_source_stability.adaptive_rank_witness.closure_mode,
                binding.graph.diagonal_l1_source_stability.adaptive_rank_witness.closure_mode,
            )
            for binding in bindings
            if binding.graph.spectral_source_stability is not None
            and binding.graph.diagonal_l1_source_stability is not None
        )
        if (
            len(derived_closure_modes) != len(bindings)
            or any(value is None for pair in derived_closure_modes for value in pair)
        ):
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.D_TO_C_STABILITY_MISSING,
                "one source branch has no registered relaxed-problem closure mode",
            )
        normalized_modes = tuple(
            (left, right)
            for left, right in derived_closure_modes
            if left is not None and right is not None
        )
        supplied_modes = tuple(self.closure_modes_by_rank_and_branch)
        decision_level_used = any(
            value is V1RelaxedKKTClosureMode.DECISION_LEVEL_SHELL_INVARIANCE
            for pair in normalized_modes
            for value in pair
        )
        if decision_level_used:
            if supplied_modes != normalized_modes:
                _refuse(
                    V1SampledKKTDeviceShellBridgeRefusalCode.D_TO_C_STABILITY_MISSING,
                    "decision-level closure-mode axis differs from source witnesses",
                )
        elif supplied_modes:
            raise ValueError(
                "legacy exact/full-sensitivity admission cannot change its hash schema"
            )
        hard_fq_status_axes_stable = (
            final_binding.graph.spectral_source_stability is not None
            and final_binding.graph.diagonal_l1_source_stability is not None
            and final_binding.graph.spectral_source_stability.hard_fq_status_axis_stable
            and final_binding.graph.diagonal_l1_source_stability.hard_fq_status_axis_stable
        )
        if (
            ledger.policy.policy_hash != self.policy_hash
            or stability.policy_hash != self.policy_hash
            or stability.replay_ledger_hash != ledger_hash
            or tuple(value.rank for value in bindings)
            != stability.executed_rank_prefix
            or tuple(value.replay_record.record_hash for value in bindings)
            != stability.replay_record_hashes
            or tuple(value.replay_record for value in bindings) != ledger.records
            or bindings[-1].rank != stability.frozen_rank
            or bindings[-1].replay_record.record_hash
            != stability.frozen_replay_record_hash
            or bindings[-1].derived_decision.decision_certificate.certificate_hash
            != stability.rank_local_decision_certificate_hash
            or bindings[-1].graph.branch_family.family_hash
            != stability.source_pre_c_branch_family_hash
            or bindings[-1].derived_decision.decision_certificate.robust_branch_family_hash
            != stability.robust_branch_family_hash
            or not all(
                value.graph.source_built_face_shell_witness_available
                and value.graph.cuda_device_native_fq_replay_available
                for value in bindings
            )
            or not exact_d_sources
            or not identical_raw_prefix
            or not nonfinal_rank_local_ambiguity
            or not source_witnesses_stable
            or not relaxed_problem_sensitivity_closed
            or not complete_cuda_fq
            or not hard_fq_status_axes_stable
        ):
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.D_TO_C_STABILITY_MISSING,
                "source ranks, identical-D ledger, and stability certificate differ",
            )
        if (
            not final_decision.rank_local_residual_decision_stable
            or not final_decision.C_stream_open_authority
            or not stability.C_stream_open_authority
        ):
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.D_TO_C_STABILITY_MISSING,
                "the frozen rank has no source-derived stable decision",
            )
        flags = (
            self.exact_registered_D_sources_bound,
            self.identical_registered_D_prefix_replayed,
            self.source_branch_witnesses_stable,
            self.nominal_fixed_direction_witnesses_diagnostic_only,
            self.relaxed_problem_perturbation_exact_zero,
            self.complete_cuda_hard_fq_replay,
            self.hard_fq_status_axes_stable,
            self.adaptive_rank_decision_stable,
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("D-to-C admission flags must be exact booleans")
        if (
            self.exact_registered_D_sources_bound is not True
            or self.identical_registered_D_prefix_replayed is not True
            or self.source_branch_witnesses_stable is not True
            or self.nominal_fixed_direction_witnesses_diagnostic_only is not True
            or self.relaxed_problem_perturbation_exact_zero
            is not relaxed_problem_exact_zero
            or self.complete_cuda_hard_fq_replay is not True
            or self.hard_fq_status_axes_stable is not True
            or self.adaptive_rank_decision_stable is not True
            or self.C_stream_open_authority is not True
            or any(
                (
                    self.calibration_K_open_authority,
                    self.paired_oracle_O_open_authority,
                    self.mask_commit_authority,
                    self.terminal_authority,
                    self.pareto_authority,
                )
            )
        ):
            raise PermissionError("D-to-C admission crossed its C-only authority")
        expected_schema = (
            SAMPLED_KKT_DEVICE_SHELL_D_TO_C_ADMISSION_V2_SCHEMA
            if decision_level_used
            else SAMPLED_KKT_DEVICE_SHELL_D_TO_C_ADMISSION_SCHEMA
        )
        if self.schema != expected_schema:
            raise ValueError("D-to-C admission schema is invalid")
        object.__setattr__(self, "rank_bindings", bindings)
        object.__setattr__(
            self,
            "closure_modes_by_rank_and_branch",
            normalized_modes if decision_level_used else (),
        )
        self.assert_integrity()

    @property
    def frozen_rank(self) -> int:
        return self.rank_bindings[-1].rank

    @property
    def frozen_graph_hash(self) -> str:
        return self.rank_bindings[-1].graph.graph_hash

    def _payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "policy_hash": self.policy_hash,
            "replay_ledger_hash": self.replay_ledger.ledger_hash,
            "stability_certificate_hash": (
                self.stability_certificate.certificate_hash
            ),
            "rank_binding_hashes": tuple(
                value.binding_hash for value in self.rank_bindings
            ),
            "registered_D_source_coordinate_hashes": tuple(
                value.graph.registered_d_source_coordinate.coordinate_hash
                for value in self.rank_bindings
                if value.graph.registered_d_source_coordinate is not None
            ),
            "frozen_rank": self.frozen_rank,
            "frozen_graph_hash": self.frozen_graph_hash,
            "exact_registered_D_sources_bound": (
                self.exact_registered_D_sources_bound
            ),
            "identical_registered_D_prefix_replayed": (
                self.identical_registered_D_prefix_replayed
            ),
            "source_branch_witnesses_stable": (
                self.source_branch_witnesses_stable
            ),
            "nominal_fixed_direction_witnesses_diagnostic_only": (
                self.nominal_fixed_direction_witnesses_diagnostic_only
            ),
            "relaxed_problem_perturbation_exact_zero": (
                self.relaxed_problem_perturbation_exact_zero
            ),
            "complete_cuda_hard_fq_replay": self.complete_cuda_hard_fq_replay,
            "hard_fq_status_axes_stable": self.hard_fq_status_axes_stable,
            "adaptive_rank_decision_stable": (
                self.adaptive_rank_decision_stable
            ),
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
        }
        if self.schema == SAMPLED_KKT_DEVICE_SHELL_D_TO_C_ADMISSION_V2_SCHEMA:
            payload["closure_modes_by_rank_and_branch"] = (
                self.closure_modes_by_rank_and_branch
            )
        return payload

    @property
    def certificate_hash(self) -> str:
        cached = self._certificate_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_D_TO_C_ADMISSION_PREFIX)
        cached = self._certificate_hash
        if cached is not None and cached != observed:
            raise ValueError("D-to-C admission certificate was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("D-to-C admission certificate hash does not match")
        object.__setattr__(self, "_certificate_hash", observed)
        return observed


def certify_v1_sampled_kkt_device_shell_d_to_c_admission(
    *,
    policy: V1FQRAdaptiveRankPolicy,
    rank_bindings: tuple[V1SampledKKTDeviceShellRankBinding, ...],
) -> V1SampledKKTDeviceShellDToCAdmissionCertificate:
    """Close the identical-D adaptive prefix and issue C-only authority."""

    if type(policy) is not V1FQRAdaptiveRankPolicy:
        raise TypeError("D-to-C admission requires the exact adaptive policy")
    bindings = tuple(rank_bindings)
    if not bindings or any(
        type(value) is not V1SampledKKTDeviceShellRankBinding
        for value in bindings
    ):
        raise TypeError("D-to-C admission requires exact rank bindings")
    ledger = V1FQRAdaptiveRankReplayLedger(
        policy=policy,
        records=tuple(value.replay_record for value in bindings),
    )
    stability = certify_v1_fqr_adaptive_rank_stability(
        policy=policy,
        ledger=ledger,
    )
    exact_zero = all(
        binding.graph.spectral_source_stability is not None
        and binding.graph.diagonal_l1_source_stability is not None
        and binding.graph.spectral_source_stability.relaxed_problem_perturbation_exact_zero
        and binding.graph.diagonal_l1_source_stability.relaxed_problem_perturbation_exact_zero
        for binding in bindings
    )
    closure_modes = tuple(
        (
            binding.graph.spectral_source_stability.adaptive_rank_witness.closure_mode,
            binding.graph.diagonal_l1_source_stability.adaptive_rank_witness.closure_mode,
        )
        for binding in bindings
        if binding.graph.spectral_source_stability is not None
        and binding.graph.diagonal_l1_source_stability is not None
    )
    decision_level_used = any(
        value is V1RelaxedKKTClosureMode.DECISION_LEVEL_SHELL_INVARIANCE
        for pair in closure_modes
        for value in pair
    )
    normalized_modes = tuple(
        (left, right)
        for left, right in closure_modes
        if left is not None and right is not None
    )
    return V1SampledKKTDeviceShellDToCAdmissionCertificate(
        policy_hash=policy.policy_hash,
        replay_ledger=ledger,
        stability_certificate=stability,
        rank_bindings=bindings,
        relaxed_problem_perturbation_exact_zero=exact_zero,
        closure_modes_by_rank_and_branch=(
            normalized_modes if decision_level_used else ()
        ),
        schema=(
            SAMPLED_KKT_DEVICE_SHELL_D_TO_C_ADMISSION_V2_SCHEMA
            if decision_level_used
            else SAMPLED_KKT_DEVICE_SHELL_D_TO_C_ADMISSION_SCHEMA
        ),
        _issuance_token=_D_TO_C_ADMISSION_ISSUANCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledKKTAdaptiveExecutionRankFreeze:
    """Bind pre-outcome HBM admission to the first D-stable execution rank.

    The memory policy answers only which ranks are admissible before any D or
    C outcomes exist.  The D-to-C certificate answers a different question:
    after replaying the identical registered D prefix, which first admitted
    rank makes the next two-branch irreversible decision invariant to the
    registered sampling, FD, and KKT uncertainty sets?  Keeping those owners
    separate prevents the largest rank that *fits* from being mistaken for the
    rank that must produce the frozen candidate family.

    This receipt grants C-open authority inherited from the exact D-to-C
    certificate and nothing beyond C.  It is durable and array-free; no FD
    snapshot, relaxed primal, route key, label vector, or context row is kept.
    """

    memory_admission_policy: V1SampledKKTProposalRankPolicy
    d_to_c_admission: V1SampledKKTDeviceShellDToCAdmissionCertificate
    _issuance_token: InitVar[object]
    memory_admission_registered_before_D_or_C_outcomes: bool = True
    empirical_outcomes_used_for_memory_admission: bool = False
    final_rank_selected_from_registered_D_only: bool = True
    identical_D_prefix_replayed_at_every_executed_rank: bool = True
    stopped_at_first_decision_stable_rank: bool = True
    C_stream_open_authority: bool = True
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_KKT_ADAPTIVE_EXECUTION_RANK_FREEZE_SCHEMA
    _freeze_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _ADAPTIVE_EXECUTION_RANK_FREEZE_ISSUANCE_TOKEN:
            raise PermissionError(
                "adaptive execution-rank freeze requires its registered issuer"
            )
        memory = self.memory_admission_policy
        admission = self.d_to_c_admission
        if type(memory) is not V1SampledKKTProposalRankPolicy:
            raise TypeError(
                "adaptive execution rank requires the exact HBM-admission policy"
            )
        if type(admission) is not V1SampledKKTDeviceShellDToCAdmissionCertificate:
            raise TypeError(
                "adaptive execution rank requires the exact D-to-C certificate"
            )
        memory.assert_integrity()
        admission.assert_integrity()
        stability = admission.stability_certificate
        ledger = admission.replay_ledger
        final_binding = admission.rank_bindings[-1]
        final_graph = final_binding.graph
        d_source = final_graph.registered_d_source_coordinate
        admitted = memory.admitted_rank_ladder
        executed = stability.executed_rank_prefix
        if (
            memory.source_memory_forecast_policy_hash != admission.policy_hash
            or ledger.policy.policy_hash != admission.policy_hash
            or stability.policy_hash != admission.policy_hash
            or memory.route_count != stability.route_count
            or memory.route_count != final_graph.route_count
            or admitted != stability.admitted_rank_ladder
            or executed != admitted[: len(executed)]
            or not executed
            or stability.frozen_rank != executed[-1]
            or admission.frozen_rank != stability.frozen_rank
            or final_binding.rank != stability.frozen_rank
            or not stability.stopped_at_first_robust_rank
            or not stability.C_stream_open_authority
            or not admission.C_stream_open_authority
            or not final_graph.source_built_face_shell_witness_available
            or not final_graph.cuda_device_native_fq_replay_available
            or d_source is None
            or d_source.fd_sketch_rank != stability.frozen_rank
            or final_graph.c_stream_outcomes_seen
        ):
            raise ValueError(
                "pre-outcome HBM admission and first identical-D stable rank "
                "do not close"
            )
        flags = (
            self.memory_admission_registered_before_D_or_C_outcomes,
            self.empirical_outcomes_used_for_memory_admission,
            self.final_rank_selected_from_registered_D_only,
            self.identical_D_prefix_replayed_at_every_executed_rank,
            self.stopped_at_first_decision_stable_rank,
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("adaptive execution-rank flags must be exact booleans")
        if flags != (
            True,
            False,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ):
            raise PermissionError(
                "adaptive execution-rank freeze crossed its C-only authority"
            )
        if self.schema != SAMPLED_KKT_ADAPTIVE_EXECUTION_RANK_FREEZE_SCHEMA:
            raise ValueError("adaptive execution-rank freeze schema is invalid")
        self.assert_integrity()

    @property
    def admitted_rank_ladder(self) -> tuple[int, ...]:
        return self.memory_admission_policy.admitted_rank_ladder

    @property
    def executed_rank_prefix(self) -> tuple[int, ...]:
        return self.d_to_c_admission.stability_certificate.executed_rank_prefix

    @property
    def final_execution_rank(self) -> int:
        return self.d_to_c_admission.frozen_rank

    @property
    def maximum_hbm_admitted_rank(self) -> int:
        return max(self.admitted_rank_ladder)

    @property
    def final_graph(self) -> V1SampledKKTDeviceShellGraph:
        return self.d_to_c_admission.rank_bindings[-1].graph

    @property
    def frozen_candidate_family_hash(self) -> str:
        return self.final_graph.branch_family.family_hash

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "memory_admission_policy_hash": (
                self.memory_admission_policy.policy_hash
            ),
            "adaptive_rank_policy_hash": self.d_to_c_admission.policy_hash,
            "d_to_c_admission_certificate_hash": (
                self.d_to_c_admission.certificate_hash
            ),
            "admitted_rank_ladder": self.admitted_rank_ladder,
            "executed_rank_prefix": self.executed_rank_prefix,
            "maximum_hbm_admitted_rank": self.maximum_hbm_admitted_rank,
            "final_execution_rank": self.final_execution_rank,
            "final_graph_hash": self.final_graph.graph_hash,
            "frozen_candidate_family_hash": self.frozen_candidate_family_hash,
            "memory_admission_registered_before_D_or_C_outcomes": (
                self.memory_admission_registered_before_D_or_C_outcomes
            ),
            "empirical_outcomes_used_for_memory_admission": (
                self.empirical_outcomes_used_for_memory_admission
            ),
            "final_rank_selected_from_registered_D_only": (
                self.final_rank_selected_from_registered_D_only
            ),
            "identical_D_prefix_replayed_at_every_executed_rank": (
                self.identical_D_prefix_replayed_at_every_executed_rank
            ),
            "stopped_at_first_decision_stable_rank": (
                self.stopped_at_first_decision_stable_rank
            ),
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
        }

    @property
    def freeze_hash(self) -> str:
        cached = self._freeze_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.memory_admission_policy.assert_integrity()
        self.d_to_c_admission.assert_integrity()
        observed = content_hash(
            self._payload(),
            prefix=_ADAPTIVE_EXECUTION_RANK_FREEZE_PREFIX,
        )
        cached = self._freeze_hash
        if cached is not None and cached != observed:
            raise ValueError("adaptive execution-rank freeze was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("adaptive execution-rank freeze hash does not match")
        object.__setattr__(self, "_freeze_hash", observed)
        return observed


def freeze_v1_sampled_kkt_adaptive_execution_rank(
    *,
    memory_admission_policy: V1SampledKKTProposalRankPolicy,
    d_to_c_admission: V1SampledKKTDeviceShellDToCAdmissionCertificate,
) -> V1SampledKKTAdaptiveExecutionRankFreeze:
    """Freeze the first source-stable rank without changing HBM admission."""

    return V1SampledKKTAdaptiveExecutionRankFreeze(
        memory_admission_policy=memory_admission_policy,
        d_to_c_admission=d_to_c_admission,
        _issuance_token=_ADAPTIVE_EXECUTION_RANK_FREEZE_ISSUANCE_TOKEN,
    )


def freeze_v1_sampled_kkt_device_shell_adaptive_for_independent_c(
    *,
    execution_rank_freeze: V1SampledKKTAdaptiveExecutionRankFreeze,
) -> V1SampledKKTDeviceShellProposalFreezeForC:
    """Freeze the first D-stable device-shell family for independent C."""

    if type(execution_rank_freeze) is not V1SampledKKTAdaptiveExecutionRankFreeze:
        raise TypeError(
            "adaptive C freeze requires the exact execution-rank freeze"
        )
    execution_rank_freeze.assert_integrity()
    return V1SampledKKTDeviceShellProposalFreezeForC(
        graph=execution_rank_freeze.final_graph,
        proposal_rank_policy=execution_rank_freeze.memory_admission_policy,
        adaptive_execution_rank_freeze=execution_rank_freeze,
        robust_kkt_sensitivity_required_for_C=True,
        schema=SAMPLED_KKT_DEVICE_SHELL_ADAPTIVE_FREEZE_SCHEMA,
        _issuance_token=_PROPOSAL_FREEZE_ISSUANCE_TOKEN,
    )


# ---------------------------------------------------------------------------
# Endogenous rank v2
# ---------------------------------------------------------------------------
#
# The v1 proposal-rank policy above is historical replay authority: it selects
# `max(admitted)` before D and its hashes must not change.  V2 is intentionally
# separate.  Its pre-D object admits an execution lattice only.  The first
# rank whose complete source-issued decision image is invariant is selected
# after D, from rank-local FD states sharing one capture/VJP lineage.


@dataclass(frozen=True, slots=True)
class V1SampledKKTEndogenousRankPolicyV2:
    """Sampled-KKT binding to physical rank admission, with no selected rank."""

    policy_id: str
    source_endogenous_rank_policy: V1FQREndogenousRankPolicyV2
    source_memory_forecast_policy_hash: str
    route_count: int
    admitted_execution_rank_lattice: tuple[int, ...]
    _issuance_token: InitVar[object]
    pre_D_selected_rank: None = None
    selection_rule: str = "minimum_source_decision_invariant_rank_after_D_v2"
    rank_lattice_is_execution_only: bool = True
    admission_registered_before_D_or_C_outcomes: bool = True
    empirical_outcomes_used_for_memory_admission: bool = False
    maximum_rank_selected_before_D: bool = False
    same_capture_VJP_fanout_required: bool = True
    C_stream_open_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_KKT_ENDOGENOUS_RANK_POLICY_V2_SCHEMA
    _policy_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _ENDOGENOUS_RANK_POLICY_V2_ISSUANCE_TOKEN:
            raise PermissionError("endogenous sampled-KKT policy requires its issuer")
        object.__setattr__(
            self,
            "policy_id",
            _identity(self.policy_id, name="endogenous sampled-KKT policy id"),
        )
        source = self.source_endogenous_rank_policy
        if type(source) is not V1FQREndogenousRankPolicyV2:
            raise TypeError("endogenous sampled-KKT policy requires the exact v2 source")
        source_hash = source.assert_integrity()
        supplied_source_hash = _identity(
            self.source_memory_forecast_policy_hash,
            name="source endogenous-rank policy hash",
        )
        routes = _nonnegative_int(self.route_count, name="route count")
        ranks = tuple(
            _nonnegative_int(value, name="admitted execution rank")
            for value in self.admitted_execution_rank_lattice
        )
        if (
            supplied_source_hash != source_hash
            or routes != source.route_count
            or ranks != source.admitted_rank_lattice
            or not ranks
        ):
            raise ValueError("endogenous sampled-KKT policy does not replay its source")
        flags = (
            self.rank_lattice_is_execution_only,
            self.admission_registered_before_D_or_C_outcomes,
            self.empirical_outcomes_used_for_memory_admission,
            self.maximum_rank_selected_before_D,
            self.same_capture_VJP_fanout_required,
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("endogenous sampled-KKT policy flags must be exact booleans")
        if flags != (
            True,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            raise PermissionError("endogenous sampled-KKT policy crossed its boundary")
        if self.pre_D_selected_rank is not None:
            raise PermissionError("endogenous sampled-KKT cannot select a rank before D")
        if self.selection_rule != "minimum_source_decision_invariant_rank_after_D_v2":
            raise ValueError("endogenous sampled-KKT selection rule is invalid")
        if self.schema != SAMPLED_KKT_ENDOGENOUS_RANK_POLICY_V2_SCHEMA:
            raise ValueError("endogenous sampled-KKT policy schema is invalid")
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "admitted_execution_rank_lattice", ranks)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "source_memory_forecast_policy_hash": (
                self.source_memory_forecast_policy_hash
            ),
            "route_count": self.route_count,
            "admitted_execution_rank_lattice": (
                self.admitted_execution_rank_lattice
            ),
            "pre_D_selected_rank": self.pre_D_selected_rank,
            "selection_rule": self.selection_rule,
            "rank_lattice_is_execution_only": self.rank_lattice_is_execution_only,
            "admission_registered_before_D_or_C_outcomes": (
                self.admission_registered_before_D_or_C_outcomes
            ),
            "empirical_outcomes_used_for_memory_admission": (
                self.empirical_outcomes_used_for_memory_admission
            ),
            "maximum_rank_selected_before_D": self.maximum_rank_selected_before_D,
            "same_capture_VJP_fanout_required": (
                self.same_capture_VJP_fanout_required
            ),
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
        }

    @property
    def policy_hash(self) -> str:
        cached = self._policy_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.source_endogenous_rank_policy.assert_integrity(
            self.source_memory_forecast_policy_hash
        )
        observed = content_hash(
            self._payload(),
            prefix=_ENDOGENOUS_RANK_POLICY_V2_PREFIX,
        )
        cached = self._policy_hash
        if cached is not None and cached != observed:
            raise ValueError("endogenous sampled-KKT policy was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("endogenous sampled-KKT policy hash does not match")
        object.__setattr__(self, "_policy_hash", observed)
        return observed


def register_v1_sampled_kkt_endogenous_rank_policy_v2(
    *,
    policy: V1FQREndogenousRankPolicyV2,
) -> V1SampledKKTEndogenousRankPolicyV2:
    """Bind the v2 physical lattice without choosing its maximum rank."""

    if type(policy) is not V1FQREndogenousRankPolicyV2:
        raise TypeError("v2 sampled-KKT registration requires the exact v2 policy")
    source_hash = policy.assert_integrity()
    return V1SampledKKTEndogenousRankPolicyV2(
        policy_id=f"{policy.policy_id}:sampled-KKT-endogenous-v2",
        source_endogenous_rank_policy=policy,
        source_memory_forecast_policy_hash=source_hash,
        route_count=policy.route_count,
        admitted_execution_rank_lattice=policy.admitted_rank_lattice,
        _issuance_token=_ENDOGENOUS_RANK_POLICY_V2_ISSUANCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledKKTSourceDecisionImageV2:
    """Complete source-issued D/KKT decision image for one rank-local state."""

    endogenous_rank_policy_hash: str
    source_endogenous_rank_policy_hash: str
    registered_d_prefix_hash: str
    same_vjp_fanout_hash: str
    rank: int
    rank_local_fd_state_hash: str
    source_decision_certificate: V1FQRAdaptiveRankDecisionCertificate
    source_component_receipt_hashes: tuple[str, str, str]
    D_sampling_decision_invariant: bool
    FD_truncation_decision_invariant: bool
    KKT_numerical_decision_invariant: bool
    complete_decision_image_invariant: bool
    _issuance_token: InitVar[object]
    source_issued: bool = True
    complete_two_branch_image: bool = True
    C_stream_open_authority: bool = False
    schema: str = SAMPLED_KKT_SOURCE_DECISION_IMAGE_V2_SCHEMA
    _image_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _SOURCE_DECISION_IMAGE_V2_ISSUANCE_TOKEN:
            raise PermissionError("v2 source decision image requires its issuer")
        for name in (
            "endogenous_rank_policy_hash",
            "source_endogenous_rank_policy_hash",
            "registered_d_prefix_hash",
            "same_vjp_fanout_hash",
            "rank_local_fd_state_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        rank = _nonnegative_int(self.rank, name="decision-image rank")
        if rank == 0:
            raise ValueError("decision-image rank must be positive")
        certificate = self.source_decision_certificate
        if type(certificate) is not V1FQRAdaptiveRankDecisionCertificate:
            raise TypeError("decision image requires the exact source certificate")
        certificate.assert_integrity()
        if (
            certificate.schema != FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA
            or certificate.rank != rank
            or certificate.policy_hash != self.source_endogenous_rank_policy_hash
        ):
            raise ValueError("source decision certificate has stale v2 coordinates")
        receipt_hashes = tuple(
            _identity(value, name="source component-invariance receipt hash")
            for value in self.source_component_receipt_hashes
        )
        if len(receipt_hashes) != 3:
            raise ValueError("decision image requires D, FD, and KKT source receipts")
        flags = (
            self.D_sampling_decision_invariant,
            self.FD_truncation_decision_invariant,
            self.KKT_numerical_decision_invariant,
            self.complete_decision_image_invariant,
            self.source_issued,
            self.complete_two_branch_image,
            self.C_stream_open_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("source decision-image flags must be exact booleans")
        component_invariant = all(flags[:3])
        certificate_invariant = (
            certificate.rank_local_residual_decision_stable
            and certificate.robust_branch_family_hash is not None
            and certificate.robust_next_decision_hash
            == certificate.robust_branch_family_hash
            and certificate.C_stream_open_authority
        )
        expected_invariant = component_invariant and certificate_invariant
        if self.complete_decision_image_invariant is not expected_invariant:
            raise ValueError("complete decision-image invariance does not replay")
        if certificate_invariant is not component_invariant:
            raise ValueError(
                "source certificate and D/FD/KKT component invariance disagree"
            )
        if flags[4:] != (True, True, False):
            raise PermissionError("a rank-local image cannot itself open C")
        if self.schema != SAMPLED_KKT_SOURCE_DECISION_IMAGE_V2_SCHEMA:
            raise ValueError("source decision-image schema is invalid")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "source_component_receipt_hashes", receipt_hashes)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "endogenous_rank_policy_hash": self.endogenous_rank_policy_hash,
            "source_endogenous_rank_policy_hash": (
                self.source_endogenous_rank_policy_hash
            ),
            "registered_d_prefix_hash": self.registered_d_prefix_hash,
            "same_vjp_fanout_hash": self.same_vjp_fanout_hash,
            "rank": self.rank,
            "rank_local_fd_state_hash": self.rank_local_fd_state_hash,
            "source_decision_certificate_hash": (
                self.source_decision_certificate.certificate_hash
            ),
            "source_component_receipt_hashes": (
                self.source_component_receipt_hashes
            ),
            "D_sampling_decision_invariant": self.D_sampling_decision_invariant,
            "FD_truncation_decision_invariant": (
                self.FD_truncation_decision_invariant
            ),
            "KKT_numerical_decision_invariant": (
                self.KKT_numerical_decision_invariant
            ),
            "complete_decision_image_invariant": (
                self.complete_decision_image_invariant
            ),
            "source_issued": self.source_issued,
            "complete_two_branch_image": self.complete_two_branch_image,
            "C_stream_open_authority": self.C_stream_open_authority,
        }

    @property
    def image_hash(self) -> str:
        cached = self._image_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.source_decision_certificate.assert_integrity()
        observed = content_hash(
            self._payload(),
            prefix=_SOURCE_DECISION_IMAGE_V2_PREFIX,
        )
        cached = self._image_hash
        if cached is not None and cached != observed:
            raise ValueError("source decision image was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("source decision image hash does not match")
        object.__setattr__(self, "_image_hash", observed)
        return observed


def issue_v1_sampled_kkt_source_decision_image_v2(
    *,
    policy: V1SampledKKTEndogenousRankPolicyV2,
    d_prefix: V1FQREndogenousRankDPrefixV2,
    fanout: V1FQREndogenousRankSameVJPFanoutV2,
    source_decision_certificate: V1FQRAdaptiveRankDecisionCertificate,
    source_component_receipt_hashes: tuple[str, str, str],
    D_sampling_decision_invariant: bool,
    FD_truncation_decision_invariant: bool,
    KKT_numerical_decision_invariant: bool,
) -> V1SampledKKTSourceDecisionImageV2:
    """Bind a typed source decision certificate to one rank-local fanout state."""

    if type(policy) is not V1SampledKKTEndogenousRankPolicyV2:
        raise TypeError("decision image requires the exact endogenous rank policy")
    if type(d_prefix) is not V1FQREndogenousRankDPrefixV2:
        raise TypeError("decision image requires the exact D prefix")
    if type(fanout) is not V1FQREndogenousRankSameVJPFanoutV2:
        raise TypeError("decision image requires the exact same-VJP fanout")
    if type(source_decision_certificate) is not V1FQRAdaptiveRankDecisionCertificate:
        raise TypeError("decision image requires the exact source certificate")
    policy_hash = policy.assert_integrity()
    source_policy_hash = policy.source_memory_forecast_policy_hash
    prefix_hash = d_prefix.assert_integrity()
    fanout_hash = fanout.assert_integrity()
    rank = source_decision_certificate.rank
    state_by_rank = {value.rank: value for value in fanout.rank_local_states}
    if (
        d_prefix.policy_hash != source_policy_hash
        or fanout.policy_hash != source_policy_hash
        or fanout.registered_d_prefix_hash != prefix_hash
        or rank not in state_by_rank
        or source_decision_certificate.policy_hash != source_policy_hash
        or source_decision_certificate.current_mask_hash != d_prefix.current_mask_hash
        or source_decision_certificate.route_axis_hash != d_prefix.route_axis_hash
        or source_decision_certificate.route_count != policy.route_count
        or source_decision_certificate.proposal_generation
        != d_prefix.proposal_generation
    ):
        raise ValueError("decision image source coordinates do not close")
    component_invariant = all(
        (
            D_sampling_decision_invariant,
            FD_truncation_decision_invariant,
            KKT_numerical_decision_invariant,
        )
    )
    certificate_invariant = (
        source_decision_certificate.rank_local_residual_decision_stable
        and source_decision_certificate.C_stream_open_authority
        and source_decision_certificate.robust_branch_family_hash is not None
    )
    return V1SampledKKTSourceDecisionImageV2(
        endogenous_rank_policy_hash=policy_hash,
        source_endogenous_rank_policy_hash=source_policy_hash,
        registered_d_prefix_hash=prefix_hash,
        same_vjp_fanout_hash=fanout_hash,
        rank=rank,
        rank_local_fd_state_hash=state_by_rank[rank].state_hash,
        source_decision_certificate=source_decision_certificate,
        source_component_receipt_hashes=source_component_receipt_hashes,
        D_sampling_decision_invariant=D_sampling_decision_invariant,
        FD_truncation_decision_invariant=FD_truncation_decision_invariant,
        KKT_numerical_decision_invariant=KKT_numerical_decision_invariant,
        complete_decision_image_invariant=(
            component_invariant and certificate_invariant
        ),
        _issuance_token=_SOURCE_DECISION_IMAGE_V2_ISSUANCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SampledKKTEndogenousExecutionRankFreezeV2:
    """C-only authority at the minimum invariant source decision rank."""

    policy: V1SampledKKTEndogenousRankPolicyV2
    d_prefix: V1FQREndogenousRankDPrefixV2
    fanout: V1FQREndogenousRankSameVJPFanoutV2
    evaluated_decision_images: tuple[V1SampledKKTSourceDecisionImageV2, ...]
    frozen_rank: int
    frozen_decision_image_hash: str
    _issuance_token: InitVar[object]
    minimum_invariant_rank_selected: bool = True
    no_new_D_acquired_for_rank_refinement: bool = True
    same_capture_VJP_fanout_used: bool = True
    C_stream_open_authority: bool = True
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SAMPLED_KKT_ENDOGENOUS_RANK_FREEZE_V2_SCHEMA
    _freeze_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self, _issuance_token: object) -> None:
        if _issuance_token is not _ENDOGENOUS_RANK_FREEZE_V2_ISSUANCE_TOKEN:
            raise PermissionError("v2 endogenous rank freeze requires its issuer")
        policy = self.policy
        prefix = self.d_prefix
        fanout = self.fanout
        if type(policy) is not V1SampledKKTEndogenousRankPolicyV2:
            raise TypeError("v2 rank freeze requires the exact policy")
        if type(prefix) is not V1FQREndogenousRankDPrefixV2:
            raise TypeError("v2 rank freeze requires the exact D prefix")
        if type(fanout) is not V1FQREndogenousRankSameVJPFanoutV2:
            raise TypeError("v2 rank freeze requires the exact same-VJP fanout")
        policy_hash = policy.assert_integrity()
        prefix_hash = prefix.assert_integrity()
        fanout_hash = fanout.assert_integrity()
        images = tuple(self.evaluated_decision_images)
        if not images or any(
            type(value) is not V1SampledKKTSourceDecisionImageV2
            for value in images
        ):
            raise TypeError("v2 rank freeze requires exact decision images")
        for image in images:
            image.assert_integrity()
        ranks = tuple(value.rank for value in images)
        first_invariant_index = next(
            (
                index
                for index, value in enumerate(images)
                if value.complete_decision_image_invariant
            ),
            None,
        )
        frozen = _nonnegative_int(self.frozen_rank, name="frozen rank")
        if frozen == 0:
            raise ValueError("frozen rank must be positive")
        expected_frozen = (
            None
            if first_invariant_index is None
            else images[first_invariant_index].rank
        )
        state_by_rank = {value.rank: value for value in fanout.rank_local_states}
        if (
            prefix.policy_hash != policy.source_memory_forecast_policy_hash
            or fanout.policy_hash != policy.source_memory_forecast_policy_hash
            or fanout.registered_d_prefix_hash != prefix_hash
            or ranks != fanout.fanned_ranks[: len(ranks)]
            or first_invariant_index is None
            or first_invariant_index != len(images) - 1
            or frozen != expected_frozen
            or frozen not in state_by_rank
            or self.frozen_decision_image_hash != images[-1].image_hash
            or any(
                value.endogenous_rank_policy_hash != policy_hash
                or value.registered_d_prefix_hash != prefix_hash
                or value.same_vjp_fanout_hash != fanout_hash
                or value.rank_local_fd_state_hash
                != state_by_rank[value.rank].state_hash
                for value in images
            )
        ):
            raise ValueError("v2 minimum invariant rank freeze does not close")
        flags = (
            self.minimum_invariant_rank_selected,
            self.no_new_D_acquired_for_rank_refinement,
            self.same_capture_VJP_fanout_used,
            self.C_stream_open_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("v2 rank-freeze flags must be exact booleans")
        if flags != (True, True, True, True, False, False, False, False, False):
            raise PermissionError("v2 rank freeze crossed its C-only authority")
        if self.schema != SAMPLED_KKT_ENDOGENOUS_RANK_FREEZE_V2_SCHEMA:
            raise ValueError("v2 rank-freeze schema is invalid")
        object.__setattr__(self, "evaluated_decision_images", images)
        object.__setattr__(self, "frozen_rank", frozen)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_hash": self.policy.policy_hash,
            "registered_d_prefix_hash": self.d_prefix.prefix_hash,
            "same_vjp_fanout_hash": self.fanout.fanout_hash,
            "evaluated_decision_image_hashes": tuple(
                value.image_hash for value in self.evaluated_decision_images
            ),
            "evaluated_rank_prefix": tuple(
                value.rank for value in self.evaluated_decision_images
            ),
            "frozen_rank": self.frozen_rank,
            "frozen_decision_image_hash": self.frozen_decision_image_hash,
            "minimum_invariant_rank_selected": self.minimum_invariant_rank_selected,
            "no_new_D_acquired_for_rank_refinement": (
                self.no_new_D_acquired_for_rank_refinement
            ),
            "same_capture_VJP_fanout_used": self.same_capture_VJP_fanout_used,
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_K_open_authority": self.calibration_K_open_authority,
            "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
        }

    @property
    def freeze_hash(self) -> str:
        cached = self._freeze_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(),
            prefix=_ENDOGENOUS_RANK_FREEZE_V2_PREFIX,
        )
        cached = self._freeze_hash
        if cached is not None and cached != observed:
            raise ValueError("v2 endogenous rank freeze was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("v2 endogenous rank freeze hash does not match")
        object.__setattr__(self, "_freeze_hash", observed)
        return observed


def freeze_or_refuse_v1_sampled_kkt_endogenous_rank_v2(
    *,
    policy: V1SampledKKTEndogenousRankPolicyV2,
    d_prefix: V1FQREndogenousRankDPrefixV2,
    fanout: V1FQREndogenousRankSameVJPFanoutV2,
    evaluated_decision_images: tuple[V1SampledKKTSourceDecisionImageV2, ...],
) -> V1SampledKKTEndogenousExecutionRankFreezeV2 | V1FQREndogenousRankNoCRefusalV2:
    """Freeze the minimum invariant rank or issue the unique lawful next action."""

    if type(policy) is not V1SampledKKTEndogenousRankPolicyV2:
        raise TypeError("v2 rank decision requires the exact policy")
    if type(d_prefix) is not V1FQREndogenousRankDPrefixV2:
        raise TypeError("v2 rank decision requires the exact D prefix")
    if type(fanout) is not V1FQREndogenousRankSameVJPFanoutV2:
        raise TypeError("v2 rank decision requires the exact same-VJP fanout")
    policy_hash = policy.assert_integrity()
    prefix_hash = d_prefix.assert_integrity()
    fanout_hash = fanout.assert_integrity()
    images = tuple(evaluated_decision_images)
    if any(
        type(value) is not V1SampledKKTSourceDecisionImageV2
        for value in images
    ):
        raise TypeError("v2 rank decision requires exact source decision images")
    for image in images:
        image.assert_integrity()
    ranks = tuple(value.rank for value in images)
    if (
        d_prefix.policy_hash != policy.source_memory_forecast_policy_hash
        or fanout.policy_hash != policy.source_memory_forecast_policy_hash
        or fanout.registered_d_prefix_hash != prefix_hash
        or ranks != fanout.fanned_ranks[: len(ranks)]
        or any(
            value.endogenous_rank_policy_hash != policy_hash
            or value.registered_d_prefix_hash != prefix_hash
            or value.same_vjp_fanout_hash != fanout_hash
            for value in images
        )
    ):
        return V1FQREndogenousRankNoCRefusalV2(
            code=V1FQREndogenousRankNoCCode.SOURCE_DECISION_IMAGE_INCOMPLETE,
            reason="source decision images are stale, post-selected, or non-prefix",
            required_action=V1FQREndogenousRankNextAction.PHYSICAL_NO_C_REFUSAL,
            policy_hash=policy.source_memory_forecast_policy_hash,
            registered_d_prefix_hash=prefix_hash,
            same_vjp_fanout_hash=fanout_hash,
        )
    invariant_indices = tuple(
        index
        for index, value in enumerate(images)
        if value.complete_decision_image_invariant
    )
    if invariant_indices:
        first = invariant_indices[0]
        if first != len(images) - 1:
            return V1FQREndogenousRankNoCRefusalV2(
                code=V1FQREndogenousRankNoCCode.SOURCE_DECISION_IMAGE_INCOMPLETE,
                reason="rank evaluation continued after the first invariant image",
                required_action=V1FQREndogenousRankNextAction.PHYSICAL_NO_C_REFUSAL,
                policy_hash=policy.source_memory_forecast_policy_hash,
                registered_d_prefix_hash=prefix_hash,
                same_vjp_fanout_hash=fanout_hash,
                current_rank=images[first].rank,
            )
        frozen = images[first]
        return V1SampledKKTEndogenousExecutionRankFreezeV2(
            policy=policy,
            d_prefix=d_prefix,
            fanout=fanout,
            evaluated_decision_images=images,
            frozen_rank=frozen.rank,
            frozen_decision_image_hash=frozen.image_hash,
            _issuance_token=_ENDOGENOUS_RANK_FREEZE_V2_ISSUANCE_TOKEN,
        )

    current = images[-1] if images else None
    current_rank = None if current is None else current.rank
    if current is not None and not current.D_sampling_decision_invariant:
        if d_prefix.full_D_census:
            return V1FQREndogenousRankNoCRefusalV2(
                code=V1FQREndogenousRankNoCCode.MAXIMUM_ADMITTED_RANK_AMBIGUOUS,
                reason="D sampling remains ambiguous at the registered full census",
                required_action=V1FQREndogenousRankNextAction.PHYSICAL_NO_C_REFUSAL,
                policy_hash=policy.source_memory_forecast_policy_hash,
                registered_d_prefix_hash=prefix_hash,
                same_vjp_fanout_hash=fanout_hash,
                current_rank=current_rank,
            )
        return V1FQREndogenousRankNoCRefusalV2(
            code=V1FQREndogenousRankNoCCode.D_SAMPLING_AMBIGUOUS,
            reason="source decision image is not invariant over D sampling uncertainty",
            required_action=V1FQREndogenousRankNextAction.ACQUIRE_MORE_D_FOR_SAMPLING,
            policy_hash=policy.source_memory_forecast_policy_hash,
            registered_d_prefix_hash=prefix_hash,
            same_vjp_fanout_hash=fanout_hash,
            current_rank=current_rank,
        )
    if current is not None and not current.KKT_numerical_decision_invariant:
        return V1FQREndogenousRankNoCRefusalV2(
            code=V1FQREndogenousRankNoCCode.KKT_NUMERICAL_AMBIGUOUS,
            reason="source decision image is not invariant over solver uncertainty",
            required_action=V1FQREndogenousRankNextAction.CONTINUE_REGISTERED_KKT_SOLVER,
            policy_hash=policy.source_memory_forecast_policy_hash,
            registered_d_prefix_hash=prefix_hash,
            same_vjp_fanout_hash=fanout_hash,
            current_rank=current_rank,
        )

    next_image_index = len(images)
    if next_image_index < len(fanout.fanned_ranks):
        next_rank = fanout.fanned_ranks[next_image_index]
        return V1FQREndogenousRankNoCRefusalV2(
            code=V1FQREndogenousRankNoCCode.FD_REFINEMENT_AVAILABLE,
            reason=(
                "the next identifiable rank-local FD state is already available "
                "from the same capture/VJP fanout"
            ),
            required_action=(
                V1FQREndogenousRankNextAction.REFINE_RANK_FROM_SAME_VJP_FANOUT
            ),
            policy_hash=policy.source_memory_forecast_policy_hash,
            registered_d_prefix_hash=prefix_hash,
            same_vjp_fanout_hash=fanout_hash,
            current_rank=current_rank,
            next_rank=next_rank,
        )

    admitted = policy.admitted_execution_rank_lattice
    next_admitted_index = len(fanout.fanned_ranks)
    if next_admitted_index < len(admitted):
        next_rank = admitted[next_admitted_index]
        if not d_prefix.full_D_census:
            return V1FQREndogenousRankNoCRefusalV2(
                code=V1FQREndogenousRankNoCCode.D_IDENTIFIABILITY_INCOMPLETE,
                reason=(
                    "the next physically admitted rank requires n_s >= r + 1 "
                    "in every positive-weight stratum"
                ),
                required_action=(
                    V1FQREndogenousRankNextAction.ACQUIRE_MORE_D_FOR_IDENTIFIABILITY
                ),
                policy_hash=policy.source_memory_forecast_policy_hash,
                registered_d_prefix_hash=prefix_hash,
                same_vjp_fanout_hash=fanout_hash,
                current_rank=current_rank,
                next_rank=next_rank,
            )

    return V1FQREndogenousRankNoCRefusalV2(
        code=V1FQREndogenousRankNoCCode.MAXIMUM_ADMITTED_RANK_AMBIGUOUS,
        reason=(
            "every identifiable physically admitted rank retains a noninvariant "
            "source decision image; C remains closed"
        ),
        required_action=V1FQREndogenousRankNextAction.PHYSICAL_NO_C_REFUSAL,
        policy_hash=policy.source_memory_forecast_policy_hash,
        registered_d_prefix_hash=prefix_hash,
        same_vjp_fanout_hash=fanout_hash,
        current_rank=current_rank,
    )


class V1SampledKKTDeviceShellExecutionContext:
    """Ephemeral owner of the live device-shell bundle and KKT arrays only."""

    __slots__ = ("_bundle", "_evicted", "_graph", "_problem")
    __hash__ = None

    def __init__(
        self,
        *,
        graph: V1SampledKKTDeviceShellGraph,
        execution_bundle: V1RelaxedKKTDeviceShellExecutionBundle,
        problem: V1RelaxedKKTProblem,
    ) -> None:
        if type(graph) is not V1SampledKKTDeviceShellGraph:
            raise TypeError("execution context requires an exact durable graph")
        if type(execution_bundle) is not V1RelaxedKKTDeviceShellExecutionBundle:
            raise TypeError("execution context requires an exact device-shell bundle")
        if type(problem) is not V1RelaxedKKTProblem:
            raise TypeError("execution context requires an exact KKT problem")
        if execution_bundle.is_evicted:
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.CONTEXT_EVICTED,
                "device-shell bundle was already evicted",
            )
        self._graph = graph
        self._bundle: V1RelaxedKKTDeviceShellExecutionBundle | None = (
            execution_bundle
        )
        self._problem: V1RelaxedKKTProblem | None = problem
        self._evicted = False
        self._assert_live_closure()

    @property
    def graph(self) -> V1SampledKKTDeviceShellGraph:
        """Return the durable array-free graph, including after eviction."""

        return self._graph

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def _require_live(
        self,
    ) -> tuple[V1RelaxedKKTDeviceShellExecutionBundle, V1RelaxedKKTProblem]:
        bundle = self._bundle
        problem = self._problem
        if self._evicted or bundle is None or problem is None or bundle.is_evicted:
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.CONTEXT_EVICTED,
                "sampled device-shell execution context is stale",
            )
        return bundle, problem

    def _assert_live_closure(self) -> None:
        bundle, problem = self._require_live()
        graph = self._graph
        if (
            _replayed_problem_hash(problem) != problem.problem_hash
            or problem.problem_hash != graph.problem_hash
            or problem.problem_id != graph.problem_id
            or problem.route_axis_hash != graph.route_axis_hash
            or problem.route_count != graph.route_count
            or problem.source_geometry_hash != graph.chart_identity_hash
            or bundle.receipt.receipt_hash != graph.execution_receipt.receipt_hash
            or bundle.base_execution.result.result_hash != graph.base_result_hash
        ):
            raise ValueError("ephemeral problem or execution bundle left its graph")
        graph.assert_integrity()

    def borrow_branch_labels(
        self,
        *,
        branch: V1RelaxedKKTBranch,
        operation: V1DeviceShellBorrowOperation,
        admission_certificate: (
            V1SampledKKTDeviceShellDToCAdmissionCertificate
            | V1SampledKKTDeviceShellProposalFreezeForC
            | None
        ) = None,
    ) -> V1DeviceShellLabelLease:
        """Issue a lifetime-guarded label borrow for the registered F/Q replay."""

        if type(branch) is not V1RelaxedKKTBranch:
            raise TypeError("branch must be an exact V1RelaxedKKTBranch")
        if type(operation) is not V1DeviceShellBorrowOperation:
            raise TypeError("operation must be an exact label-borrow operation")
        if operation is V1DeviceShellBorrowOperation.FQ_HARD_REPLAY:
            if admission_certificate is not None:
                raise ValueError("F/Q replay must not consume a downstream admission")
        else:
            certificate = admission_certificate
            if type(certificate) is V1SampledKKTDeviceShellDToCAdmissionCertificate:
                certificate.assert_integrity()
                authorized = (
                    certificate.C_stream_open_authority
                    and certificate.frozen_graph_hash == self._graph.graph_hash
                )
            elif type(certificate) is V1SampledKKTDeviceShellProposalFreezeForC:
                certificate.assert_integrity()
                authorized = (
                    certificate.C_direction_borrow_authority
                    and certificate.independent_C_FQR_required
                    and certificate.frozen_graph_hash == self._graph.graph_hash
                )
            else:
                _refuse(
                    V1SampledKKTDeviceShellBridgeRefusalCode.UNAUTHORIZED_LABEL_OPERATION,
                    "C label borrowing requires an exact live stability admission "
                    "or proposal-only independent-C freeze",
                )
            if not authorized:
                _refuse(
                    V1SampledKKTDeviceShellBridgeRefusalCode.UNAUTHORIZED_LABEL_OPERATION,
                    "C-direction admission belongs to another rank or source graph",
                )
        self._assert_live_closure()
        bundle, _ = self._require_live()
        handle = (
            bundle.spectral_label_handle
            if branch is V1RelaxedKKTBranch.SPECTRAL
            else bundle.diagonal_l1_label_handle
        )
        return handle.borrow_for_registered_operation(operation=operation)

    def evict(self) -> None:
        """Release both label axes and the problem, refusing partial eviction."""

        bundle, _ = self._require_live()
        # Use the bundle's cleanup-only owners directly.  Public handle
        # accessors correctly reject a poisoned invariant, but that must not
        # strand the allocation after all leases have closed.
        spectral = bundle._spectral_label_handle
        diagonal = bundle._diagonal_l1_label_handle
        if spectral.active_lease_count or diagonal.active_lease_count:
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.CONTEXT_HAS_ACTIVE_LEASES,
                "all operation-specific label leases must close before eviction",
            )
        bundle.evict()
        if not bundle.is_evicted:  # pragma: no cover - defensive lifecycle guard.
            raise RuntimeError("device-shell bundle did not evict both label axes")
        self._bundle = None
        self._problem = None
        self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("sampled device-shell contexts are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("sampled device-shell contexts are nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("sampled device-shell contexts are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("sampled device-shell contexts cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("sampled device-shell contexts cannot be copied")


def build_v1_sampled_kkt_device_shell_bridge(
    *,
    execution_bundle: V1RelaxedKKTDeviceShellExecutionBundle,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    sampled_geometry_source_hash: str | None = None,
    sampled_geometry_source: V1SampledPreFreezeGeometrySource | None = None,
    source_problem_bundle: V1SampledKKTSourceProblemBundle | None = None,
    current_mask_hash: str,
    chart_identity_hash: str,
    proposal_generation: int,
    discovery_prefix_hash: str,
    discovery_evidence_hash: str,
    spectral_hard_fq_replay: DeviceShellHardFQReplayReceipt,
    diagonal_l1_hard_fq_replay: DeviceShellHardFQReplayReceipt,
    adaptive_rank_witnesses: tuple[
        V1RelaxedKKTAdaptiveRankBranchWitness, ...
    ] | None = None,
    decision_level_robust_invariance_certificates: tuple[
        V1RelaxedKKTDecisionLevelRobustInvarianceCertificate, ...
    ] = (),
    literal_j_geometry_uncertainty: (
        V1RelaxedKKTGeometryUncertaintySet | None
    ) = None,
    literal_j_fixed_primal_census: LiteralJFixedPrimalSourceReceipt | None = None,
) -> V1SampledKKTDeviceShellExecutionContext:
    """Close exact sampled-D coordinates around a device-shell KKT execution."""

    if type(execution_bundle) is not V1RelaxedKKTDeviceShellExecutionBundle:
        raise TypeError("device-shell bridge requires an exact execution bundle")
    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("device-shell bridge requires an exact KKT problem")
    if type(policy) is not V1RelaxedKKTPolicy:
        raise TypeError("device-shell bridge requires an exact KKT policy")
    if source_problem_bundle is not None and type(source_problem_bundle) is not (
        V1SampledKKTSourceProblemBundle
    ):
        raise TypeError("source_problem_bundle must use the exact frozen source type")
    replay_types = (
        type(spectral_hard_fq_replay),
        type(diagonal_l1_hard_fq_replay),
    )
    admitted_replay_types = {
        V1DeviceShellHardFQReplayReceipt,
        V1DeviceShellHardFQCUDAReplayReceipt,
    }
    if any(value not in admitted_replay_types for value in replay_types):
        raise TypeError("device-shell bridge requires two exact hard-F/Q replays")
    if replay_types[0] is not replay_types[1]:
        raise TypeError("device-shell replay branches must use one exact backend")
    bundle_witnesses = tuple(execution_bundle.adaptive_rank_witnesses)
    if adaptive_rank_witnesses is None:
        witnesses = bundle_witnesses
    else:
        supplied_witnesses = tuple(adaptive_rank_witnesses)
        if (
            len(supplied_witnesses) != len(bundle_witnesses)
            or any(
                supplied is not source
                for supplied, source in zip(
                    supplied_witnesses,
                    bundle_witnesses,
                    strict=True,
                )
            )
        ):
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.SOURCE_WITNESS_MISSING,
                "caller-supplied witnesses are not the exact source-built bundle "
                "objects",
            )
        witnesses = bundle_witnesses
    if any(
        type(value) is not V1RelaxedKKTAdaptiveRankBranchWitness
        for value in witnesses
    ):
        raise TypeError("adaptive-rank witnesses must use the exact source type")
    if witnesses and (
        len(witnesses) != 2
        or tuple(value.branch for value in witnesses)
        != (V1RelaxedKKTBranch.SPECTRAL, V1RelaxedKKTBranch.DIAGONAL_L1)
    ):
        raise ValueError("adaptive-rank witnesses must form the two-branch axis")
    decision_certificates = tuple(
        decision_level_robust_invariance_certificates
    )
    if any(
        type(value)
        is not V1RelaxedKKTDecisionLevelRobustInvarianceCertificate
        for value in decision_certificates
    ):
        raise TypeError("decision-level theorem certificates must use exact types")
    if decision_certificates and (
        len(decision_certificates) != 2
        or tuple(value.branch for value in decision_certificates)
        != (V1RelaxedKKTBranch.SPECTRAL, V1RelaxedKKTBranch.DIAGONAL_L1)
    ):
        _refuse(
            V1SampledKKTDeviceShellBridgeRefusalCode.ROBUST_KKT_SENSITIVITY_MISSING,
            "decision-level theorem certificates must form the canonical branch axis",
        )
    if decision_certificates and not witnesses:
        _refuse(
            V1SampledKKTDeviceShellBridgeRefusalCode.ROBUST_KKT_SENSITIVITY_MISSING,
            "decision-level theorem certificates cannot float free of source witnesses",
        )
    cuda_replay = replay_types[0] is V1DeviceShellHardFQCUDAReplayReceipt
    prepared = source_problem_bundle
    if cuda_replay:
        if prepared is None:
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.REGISTERED_D_SOURCE_MISSING,
                "CUDA proposal construction requires the exact sampled-D "
                "source-problem bundle, not an independently supplied problem",
            )
        prepared.assert_integrity(prepared.bundle_hash)
        if (
            problem is not prepared.problem
            or policy is not prepared.policy
            or prepared.problem.problem_hash != problem.problem_hash
            or prepared.chart_identity_hash != problem.source_geometry_hash
        ):
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.REGISTERED_D_SOURCE_MISSING,
                "CUDA proposal problem is not the exact frozen sampled-D problem",
            )
        if sampled_geometry_source is not None and (
            sampled_geometry_source is not prepared.sampled_geometry_source
        ):
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.REGISTERED_D_SOURCE_MISSING,
                "CUDA proposal source differs from its frozen source problem",
            )
        sampled_geometry_source = prepared.sampled_geometry_source
        source_receipt = sampled_geometry_source.admission.receipt
        moments = sampled_geometry_source.admission.moments
        expected_discovery_prefix_hash = content_hash(
            {
                "context_ids": tuple(
                    item
                    for axis in moments.observed_context_ids_by_stratum
                    for item in axis
                ),
                "draw_ids": tuple(
                    item
                    for axis in moments.observed_draw_ids_by_stratum
                    for item in axis
                ),
            },
            prefix="partition-free-v1-fqr-kkt-discovery-prefix-",
        )
        if (
            current_mask_hash != source_receipt.current_mask_hash
            or chart_identity_hash != prepared.chart_identity_hash
            or proposal_generation != source_receipt.proposal_generation
            or discovery_prefix_hash != expected_discovery_prefix_hash
            or discovery_evidence_hash
            != source_receipt.corrected_moments_state_hash
            or (
                sampled_geometry_source_hash is not None
                and sampled_geometry_source_hash
                != sampled_geometry_source.source_hash
            )
        ):
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.REGISTERED_D_SOURCE_MISSING,
                "CUDA proposal chart or discovery coordinates differ from the "
                "exact frozen sampled-D source problem",
            )
    elif prepared is not None:
        raise ValueError("source-problem binding is reserved for CUDA proposal graphs")
    if witnesses and not cuda_replay:
        _refuse(
            V1SampledKKTDeviceShellBridgeRefusalCode.CUDA_HARD_FQ_REPLAY_MISSING,
            "source-stability propagation requires both CUDA-native complete "
            "hard-F/Q replays",
        )
    bundle_uncertainty = execution_bundle.geometry_uncertainty
    if literal_j_geometry_uncertainty is not None and (
        literal_j_geometry_uncertainty is not bundle_uncertainty
    ):
        _refuse(
            V1SampledKKTDeviceShellBridgeRefusalCode.SOURCE_WITNESS_MISSING,
            "caller-supplied geometry uncertainty is not the exact source-built "
            "bundle object",
        )
    geometry_uncertainty = bundle_uncertainty
    if witnesses:
        if type(geometry_uncertainty) is not V1RelaxedKKTGeometryUncertaintySet:
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.SOURCE_WITNESS_MISSING,
                "canonical face stability requires typed source-derived literal-J "
                "response-operator uncertainty",
            )
        if not _relaxed_problem_sensitivity_closed(
            uncertainty=geometry_uncertainty,
            witnesses=witnesses,
            decision_level_certificates=decision_certificates,
        ):
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.ROBUST_KKT_SENSITIVITY_MISSING,
                "fixed-direction M_J z and nominal KKT margins do not certify "
                "the optimizer face or rounded shell under nonzero A/M_J/FD "
                "relaxed-problem perturbation",
            )
        for witness, certificate, identity, branch_result in zip(
            witnesses,
            decision_certificates,
            (execution_bundle.receipt.spectral, execution_bundle.receipt.diagonal_l1),
            (
                execution_bundle.base_execution.result.spectral,
                execution_bundle.base_execution.result.diagonal_l1,
            ),
            strict=True,
        ) if decision_certificates else ():
            certificate.assert_integrity()
            if (
                certificate.certificate_hash
                != witness.decision_level_robust_invariance_certificate_hash
                or certificate.branch is not witness.branch
                or certificate.problem_hash != problem.problem_hash
                or certificate.policy_hash != policy.policy_hash
                or certificate.branch_result_hash != branch_result.result_hash
                or certificate.branch_receipt_hash
                != branch_result.receipt.receipt_hash
                or certificate.route_axis_hash != problem.route_axis_hash
                or certificate.route_count != problem.route_count
                or certificate.finite_population_geometry_uncertainty_set_hash
                != geometry_uncertainty.uncertainty_hash
                or certificate.rounded_ladder_hash != identity.source_rounding_hash
                or certificate.endpoint_action_axis_hash
                != identity.endpoint_action_axis_hash
                or tuple(certificate.worst_case_hard_fq_replay.endpoint_statuses)
                != tuple(
                    value.status.value
                    for value in (
                        spectral_hard_fq_replay.endpoints
                        if witness.branch is V1RelaxedKKTBranch.SPECTRAL
                        else diagonal_l1_hard_fq_replay.endpoints
                    )
                )
                or tuple(
                    certificate.worst_case_hard_fq_replay.endpoint_removed_index_hashes
                )
                != tuple(
                    value.removed_route_index_hash
                    for value in (
                        spectral_hard_fq_replay.endpoints
                        if witness.branch is V1RelaxedKKTBranch.SPECTRAL
                        else diagonal_l1_hard_fq_replay.endpoints
                    )
                )
            ):
                _refuse(
                    V1SampledKKTDeviceShellBridgeRefusalCode.ROBUST_KKT_SENSITIVITY_MISSING,
                    "decision-level theorem is stale, branch-swapped, or differs "
                    "from the complete hard-F/Q replay/action axis",
                )
        if type(literal_j_fixed_primal_census) not in (
            V1LiteralJOperatorCensusReceipt,
            V1LiteralJFixedDirectionProjectionReceipt,
        ):
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.SOURCE_WITNESS_MISSING,
                "canonical face stability requires the exact literal-J fixed-primal "
                "source receipt",
            )
        descriptor_axis_hash = _validated_private_primal_direction_descriptor_axis(
            literal_j_fixed_primal_census
        )
    elif geometry_uncertainty is not None:
        _refuse(
            V1SampledKKTDeviceShellBridgeRefusalCode.SOURCE_WITNESS_MISSING,
            "literal-J geometry uncertainty cannot float free of branch witnesses",
        )
    elif literal_j_fixed_primal_census is not None:
        _refuse(
            V1SampledKKTDeviceShellBridgeRefusalCode.SOURCE_WITNESS_MISSING,
            "literal-J fixed-primal source cannot float free of branch witnesses",
        )
    if sampled_geometry_source is not None and type(sampled_geometry_source) is not (
        V1SampledPreFreezeGeometrySource
    ):
        raise TypeError("sampled_geometry_source must use the exact source type")
    d_source_coordinate = (
        None
        if sampled_geometry_source is None
        else _bind_registered_d_source_coordinate(sampled_geometry_source)
    )
    d_source_required = bool(witnesses) or cuda_replay
    if d_source_required is not (d_source_coordinate is not None):
        _refuse(
            V1SampledKKTDeviceShellBridgeRefusalCode.REGISTERED_D_SOURCE_MISSING,
            "CUDA proposal or source-stability propagation requires the exact "
            "registered-D source; legacy hash-only graphs remain no-C",
        )
    if execution_bundle.is_evicted:
        _refuse(
            V1SampledKKTDeviceShellBridgeRefusalCode.CONTEXT_EVICTED,
            "device-shell bundle was evicted before graph construction",
        )
    if d_source_coordinate is None:
        sampled_hash = _identity(
            sampled_geometry_source_hash,
            name="sampled geometry source hash",
        )
    else:
        sampled_hash = d_source_coordinate.sampled_geometry_source_hash
        if sampled_geometry_source_hash is not None and (
            _identity(
                sampled_geometry_source_hash,
                name="sampled geometry source hash",
            )
            != sampled_hash
        ):
            raise ValueError("supplied sampled geometry hash differs from its source")
    mask_hash = _identity(current_mask_hash, name="current mask hash")
    chart_hash = _identity(chart_identity_hash, name="chart identity hash")
    generation = _nonnegative_int(
        proposal_generation,
        name="proposal generation",
    )
    prefix_hash = _identity(discovery_prefix_hash, name="discovery prefix hash")
    evidence_hash = _identity(
        discovery_evidence_hash,
        name="discovery evidence hash",
    )
    if _replayed_problem_hash(problem) != problem.problem_hash:
        raise ValueError("relaxed KKT problem arrays changed after registration")
    if d_source_coordinate is not None and (
        d_source_coordinate.current_mask_hash != mask_hash
        or d_source_coordinate.physical_route_axis_hash != problem.route_axis_hash
        or d_source_coordinate.route_count != problem.route_count
        or d_source_coordinate.proposal_generation != generation
    ):
        _refuse(
            V1SampledKKTDeviceShellBridgeRefusalCode.REGISTERED_D_SOURCE_MISSING,
            "the exact registered-D source left the shell chart coordinates",
        )
    if literal_j_fixed_primal_census is not None and (
        d_source_coordinate is None
        or literal_j_fixed_primal_census.model_identity_hash
        != d_source_coordinate.model_identity_hash
        or literal_j_fixed_primal_census.literal_j_stream_source_hash
        != d_source_coordinate.literal_jacobian_stream_identity_hash
        or literal_j_fixed_primal_census.chart_identity_hash != chart_hash
        or literal_j_fixed_primal_census.current_mask_hash != mask_hash
        or literal_j_fixed_primal_census.proposal_generation != generation
        or literal_j_fixed_primal_census.route_axis_hash != problem.route_axis_hash
        or literal_j_fixed_primal_census.route_count != problem.route_count
        or literal_j_fixed_primal_census.direction_count != 2
        or geometry_uncertainty is None
        or literal_j_fixed_primal_census.direction_axis_hash
        != geometry_uncertainty.literal_j_fixed_primal_direction_axis_hash
        or geometry_uncertainty.literal_j_fixed_primal_direction_count != 2
    ):
        _refuse(
            V1SampledKKTDeviceShellBridgeRefusalCode.PRIVATE_PRIMAL_DIRECTION_LINEAGE_UNAVAILABLE,
            "the exact fixed-primal census does not bind the two KKT branch directions",
        )

    bundle = execution_bundle
    receipt = bundle.receipt
    base_result = bundle.base_execution.result
    if type(base_result) is not V1RelaxedKKTResult:
        raise TypeError("device-shell bridge requires an exact base KKT result")
    if (
        problem.source_geometry_hash != chart_hash
        or receipt.problem_hash != problem.problem_hash
        or receipt.policy_hash != policy.policy_hash
        or receipt.base_result_hash != base_result.result_hash
        or base_result.problem_hash != problem.problem_hash
        or base_result.policy_hash != policy.policy_hash
        or not base_result.both_branches_converged
        or bundle.base_execution.forecast.problem_hash != problem.problem_hash
        or bundle.base_execution.telemetry.route_axis_hash != problem.route_axis_hash
        or bundle.base_execution.telemetry.source_geometry_hash != chart_hash
        or bundle.base_execution.telemetry.numerical_policy_hash != policy.policy_hash
        or bundle.base_execution.telemetry.hard_ladder_emitted
        or bundle.base_execution.telemetry.adaptive_shell_witness_available
        or receipt.adaptive_shell_witness_available
    ):
        raise ValueError("problem, policy, base result, and shell execution do not close")
    for branch_result in (base_result.spectral, base_result.diagonal_l1):
        if (
            branch_result.receipt.problem_hash != problem.problem_hash
            or branch_result.receipt.policy_hash != policy.policy_hash
            or branch_result.rounded_ladder.candidates
            or branch_result.rounded_ladder.ordered_route_indices.size
            or branch_result.relaxed_gate_values is not None
        ):
            raise PermissionError(
                "device-shell base result must contain no ordinary action, order, "
                "or relaxed route vector"
            )
    for branch_receipt in (receipt.spectral, receipt.diagonal_l1):
        ladder = branch_receipt.frozen_ladder
        if (
            ladder.route_axis_hash != problem.route_axis_hash
            or ladder.route_count != problem.route_count
            or ladder.parent_mask_hash != mask_hash
        ):
            raise ValueError("device-shell route or current-mask coordinates are stale")

    spectral_resources = tuple(
        endpoint.resource_nominal for endpoint in spectral_hard_fq_replay.endpoints
    )
    diagonal_resources = tuple(
        endpoint.resource_nominal for endpoint in diagonal_l1_hard_fq_replay.endpoints
    )
    family = freeze_v1_fqr_device_shell_branch_family(
        execution_receipt=receipt,
        chart_identity_hash=chart_hash,
        proposal_generation=generation,
        discovery_prefix_hash=prefix_hash,
        discovery_evidence_hash=evidence_hash,
        spectral_registered_resources=spectral_resources,
        diagonal_l1_registered_resources=diagonal_resources,
    )
    spectral_source: V1FQRDeviceShellSourceStabilityBinding | None = None
    diagonal_source: V1FQRDeviceShellSourceStabilityBinding | None = None
    if witnesses:
        spectral_axis_hash = getattr(receipt.spectral, "endpoint_action_axis_hash", None)
        diagonal_axis_hash = getattr(
            receipt.diagonal_l1,
            "endpoint_action_axis_hash",
            None,
        )
        if type(spectral_axis_hash) is not str or type(diagonal_axis_hash) is not str:
            _refuse(
                V1SampledKKTDeviceShellBridgeRefusalCode.SOURCE_WITNESS_MISSING,
                "source shell receipts lack the canonical endpoint action axis",
            )
        assert type(spectral_hard_fq_replay) is V1DeviceShellHardFQCUDAReplayReceipt
        assert type(diagonal_l1_hard_fq_replay) is V1DeviceShellHardFQCUDAReplayReceipt
        spectral_source = bind_v1_fqr_device_shell_source_stability(
            branch_binding=family.spectral,
            adaptive_rank_witness=witnesses[0],
            cuda_hard_fq_replay=spectral_hard_fq_replay,
            literal_j_geometry_uncertainty=geometry_uncertainty,
            literal_j_fixed_primal_census=literal_j_fixed_primal_census,
            shell_endpoint_action_axis_hash=spectral_axis_hash,
            private_primal_direction_descriptor_axis_hash=descriptor_axis_hash,
        )
        diagonal_source = bind_v1_fqr_device_shell_source_stability(
            branch_binding=family.diagonal_l1,
            adaptive_rank_witness=witnesses[1],
            cuda_hard_fq_replay=diagonal_l1_hard_fq_replay,
            literal_j_geometry_uncertainty=geometry_uncertainty,
            literal_j_fixed_primal_census=literal_j_fixed_primal_census,
            shell_endpoint_action_axis_hash=diagonal_axis_hash,
            private_primal_direction_descriptor_axis_hash=descriptor_axis_hash,
        )
    graph = V1SampledKKTDeviceShellGraph(
        sampled_geometry_source_hash=sampled_hash,
        problem_hash=problem.problem_hash,
        problem_id=problem.problem_id,
        policy=policy,
        base_result_hash=base_result.result_hash,
        execution_receipt=receipt,
        route_axis_hash=problem.route_axis_hash,
        route_count=problem.route_count,
        current_mask_hash=mask_hash,
        chart_identity_hash=chart_hash,
        proposal_generation=generation,
        discovery_prefix_hash=prefix_hash,
        discovery_evidence_hash=evidence_hash,
        spectral_hard_fq_replay=spectral_hard_fq_replay,
        diagonal_l1_hard_fq_replay=diagonal_l1_hard_fq_replay,
        branch_family=family,
        spectral_action_status_axis_hash=(
            _replay_action_status_axis_hash(spectral_hard_fq_replay)
        ),
        diagonal_l1_action_status_axis_hash=(
            _replay_action_status_axis_hash(diagonal_l1_hard_fq_replay)
        ),
        registered_d_source_coordinate=d_source_coordinate,
        source_problem_bundle_hash=(
            None if prepared is None else prepared.bundle_hash
        ),
        literal_j_fixed_primal_census_receipt_hash=(
            None
            if literal_j_fixed_primal_census is None
            else literal_j_fixed_primal_census.receipt_hash
        ),
        spectral_source_stability=spectral_source,
        diagonal_l1_source_stability=diagonal_source,
        decision_level_robust_invariance_certificate_hashes=tuple(
            value.certificate_hash for value in decision_certificates
        ),
        kkt_fd_rank=(
            d_source_coordinate.fd_sketch_rank
            if cuda_replay and not witnesses and d_source_coordinate is not None
            else None
        ),
        source_built_face_shell_witness_available=bool(witnesses),
        cuda_device_native_fq_replay_available=cuda_replay,
        schema=(
            (
                SAMPLED_KKT_DEVICE_SHELL_DECISION_STABLE_GRAPH_SCHEMA
                if decision_certificates
                else SAMPLED_KKT_DEVICE_SHELL_GRAPH_V2_SCHEMA
            )
            if witnesses
            else (
                SAMPLED_KKT_DEVICE_SHELL_PROPOSAL_GRAPH_SCHEMA
                if cuda_replay
                else SAMPLED_KKT_DEVICE_SHELL_GRAPH_SCHEMA
            )
        ),
    )
    return V1SampledKKTDeviceShellExecutionContext(
        graph=graph,
        execution_bundle=bundle,
        problem=problem,
    )


__all__ = [
    "SAMPLED_KKT_DEVICE_SHELL_AUTHORITY_SCOPE",
    "SAMPLED_KKT_DEVICE_SHELL_GRAPH_SCHEMA",
    "SAMPLED_KKT_DEVICE_SHELL_GRAPH_V2_SCHEMA",
    "SAMPLED_KKT_DEVICE_SHELL_DECISION_STABLE_GRAPH_SCHEMA",
    "SAMPLED_KKT_DEVICE_SHELL_PROPOSAL_GRAPH_SCHEMA",
    "SAMPLED_KKT_DEVICE_SHELL_RANK_BINDING_SCHEMA",
    "SAMPLED_KKT_DEVICE_SHELL_D_TO_C_ADMISSION_SCHEMA",
    "SAMPLED_KKT_DEVICE_SHELL_D_TO_C_ADMISSION_V2_SCHEMA",
    "SAMPLED_KKT_ADAPTIVE_EXECUTION_RANK_FREEZE_SCHEMA",
    "SAMPLED_KKT_PROPOSAL_RANK_POLICY_SCHEMA",
    "SAMPLED_KKT_DEVICE_SHELL_PROPOSAL_FREEZE_SCHEMA",
    "SAMPLED_KKT_DEVICE_SHELL_ADAPTIVE_FREEZE_SCHEMA",
    "SAMPLED_KKT_DEVICE_SHELL_D_SOURCE_COORDINATE_SCHEMA",
    "SAMPLED_KKT_DEVICE_SHELL_D_SOURCE_COORDINATE_V2_SCHEMA",
    "SAMPLED_KKT_ENDOGENOUS_RANK_FREEZE_V2_SCHEMA",
    "SAMPLED_KKT_ENDOGENOUS_RANK_POLICY_V2_SCHEMA",
    "SAMPLED_KKT_SOURCE_DECISION_IMAGE_V2_SCHEMA",
    "V1SampledKKTDeviceShellBridgeRefusal",
    "V1SampledKKTDeviceShellBridgeRefusalCode",
    "V1SampledKKTDeviceShellExecutionContext",
    "V1SampledKKTDeviceShellGraph",
    "V1SampledKKTProposalRankPolicy",
    "V1SampledKKTDeviceShellProposalFreezeForC",
    "V1SampledKKTDeviceShellRankBinding",
    "V1SampledKKTDeviceShellDToCAdmissionCertificate",
    "V1SampledKKTAdaptiveExecutionRankFreeze",
    "V1SampledKKTDeviceShellDSourceCoordinate",
    "V1SampledKKTEndogenousExecutionRankFreezeV2",
    "V1SampledKKTEndogenousRankPolicyV2",
    "V1SampledKKTSourceDecisionImageV2",
    "build_v1_sampled_kkt_device_shell_bridge",
    "certify_v1_sampled_kkt_device_shell_d_to_c_admission",
    "freeze_v1_sampled_kkt_device_shell_proposal_for_independent_c",
    "freeze_v1_sampled_kkt_device_shell_adaptive_for_independent_c",
    "freeze_v1_sampled_kkt_adaptive_execution_rank",
    "freeze_or_refuse_v1_sampled_kkt_endogenous_rank_v2",
    "issue_v1_sampled_kkt_source_decision_image_v2",
    "register_v1_sampled_kkt_endogenous_rank_policy_v2",
    "register_v1_sampled_kkt_proposal_rank_policy",
]
