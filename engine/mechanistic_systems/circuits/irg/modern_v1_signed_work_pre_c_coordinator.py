"""Receipt-only pre-C orchestration for endogenous signed-work proposals.

This module is the narrow execution joint between current-chart D geometry and
the independent C measurement stream.  It deliberately keeps the two phases
separate:

* Phase A sees only resource, literal-J Q geometry, eligibility, and legality;
* every certified Phase-A coordinate creates its own exact-floor Phase-B
  signed-work problem and its own endpoint plan;
* only the Q branches that Phase A explicitly witnessed at that coordinate are
  solved, rounded, and completely replayed at hard F/Q before any C outcome is
  visible; an excluded branch is never called or represented as infeasible by
  Phase B.

The returned object contains bounded receipts only.  Relaxed theorem state,
device labels, and the signed-work operator are consumed and evicted on both
success and failure.  Nothing here authorizes C, K, O, a mask commit, a
terminal claim, or a Pareto claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Final, NoReturn

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .contracts import content_hash
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch, V1RelaxedKKTProblem
from .modern_v1_signed_work_device_shell_cuda import (
    V1SignedWorkDeviceShellEndpointPlanReceipt,
    V1SignedWorkDeviceShellExecutionDesign,
    V1SignedWorkDeviceShellExecutionBundle,
    freeze_v1_signed_work_device_shell_single_branch,
)
from .modern_v1_signed_work_device_shell_c_bridge import (
    V1SignedWorkCBridgeDesign,
    V1SignedWorkCBridgeRefusal,
    V1SignedWorkCBridgeRefusalCode,
    materialize_v1_signed_work_device_shell_c_single_branch_actions,
)
from .modern_v1_signed_work_hard_fq_replay_cuda import (
    V1SignedWorkHardFQEndpointReceipt,
    V1SignedWorkHardFQReplayDesign,
    V1SignedWorkHardFQSingleBranchBundle,
    replay_v1_signed_work_hard_fq_single_branch_cuda,
)
from .modern_v1_signed_work_kkt_cuda import (
    V1SignedWorkLeastWorkCUDAPolicy,
    V1SignedWorkLeastWorkCUDAProblem,
    V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    solve_v1_signed_work_least_work_branch_cuda,
    solve_v1_signed_work_least_work_branch_private_cpu_reference,
)
from .modern_v1_signed_work_least_work_reference import (
    V1SignedWorkLeastWorkBranch,
)
from .modern_v1_signed_work_operator_cuda import V1SignedWorkOperatorCUDA
from .modern_v1_signed_work_pre_c_packed_actions import (
    V1SignedWorkCoordinateCandidateCrosswalk,
    V1SignedWorkPreCPackedActionFamily,
    build_v1_signed_work_pre_c_packed_action_family,
    crosswalk_v1_signed_work_coordinate_candidate,
)
from .modern_v1_signed_work_resource_frontier import (
    V1SignedWorkEndogenousResourceCoordinate,
    V1SignedWorkResourceUnionStatus,
)
from .modern_v1_signed_work_resource_frontier_cuda import (
    V1SignedWorkResourceFrontierCUDAPolicy,
    V1SignedWorkResourceFrontierCUDAExecutionBundle,
    V1SignedWorkResourceFrontierCUDAProblem,
    build_v1_signed_work_resource_frontier_cuda_problem_from_relaxed_kkt,
    derive_v1_signed_work_endpoint_plan_from_resource_coordinate,
    issue_v1_signed_work_resource_frontier_cuda_h200_qualification,
    solve_v1_signed_work_resource_frontier_cuda,
)


SIGNED_WORK_PRE_C_SOURCE_SCHEMA: Final = (
    "partition_free_v1_signed_work_pre_c_source_bundle_v2"
)
SIGNED_WORK_PRE_C_POLICY_SCHEMA: Final = (
    "partition_free_v1_signed_work_pre_c_policy_bundle_v1"
)
SIGNED_WORK_PRE_C_KEY_SCHEMA: Final = (
    "partition_free_v1_signed_work_pre_c_candidate_key_v1"
)
SIGNED_WORK_PRE_C_CANDIDATE_SCHEMA: Final = (
    "partition_free_v1_signed_work_pre_c_candidate_receipt_v1"
)
SIGNED_WORK_PRE_C_COORDINATE_SCHEMA: Final = (
    "partition_free_v1_signed_work_pre_c_coordinate_receipt_v1"
)
SIGNED_WORK_PRE_C_PHYSICAL_GROUP_SCHEMA: Final = (
    "partition_free_v1_signed_work_pre_c_physical_action_group_v1"
)
SIGNED_WORK_PRE_C_FAMILY_SCHEMA: Final = (
    "partition_free_v1_signed_work_pre_c_receipt_family_v2"
)
SIGNED_WORK_PRE_C_EXECUTION_BUNDLE_SCHEMA: Final = (
    "partition_free_v1_signed_work_pre_c_packed_execution_bundle_v1"
)
SIGNED_WORK_PRE_C_LIVE_COMMON_COORDINATE_SCHEMA: Final = (
    "partition_free_v1_signed_work_pre_c_live_common_coordinate_context_v1"
)
SIGNED_WORK_PRE_C_AUTHORITY: Final = (
    "bounded_pre_C_signed_work_phase_A_then_independent_exact_floor_phase_B_"
    "hard_FQ_receipts_only_no_C_K_O_commit_terminal_or_pareto_v1"
)
SIGNED_WORK_PRE_C_PACKED_EXECUTION_AUTHORITY: Final = (
    "bounded_pre_C_signed_work_phase_A_then_independent_exact_floor_phase_B_"
    "hard_FQ_receipts_plus_exact_Q_safe_packed_actions_no_C_K_O_commit_terminal_"
    "or_pareto_v1"
)

_SOURCE_PREFIX: Final = "partition-free-v1-signed-work-pre-c-source-"
_POLICY_PREFIX: Final = "partition-free-v1-signed-work-pre-c-policy-"
_KEY_PREFIX: Final = "partition-free-v1-signed-work-pre-c-key-"
_CANDIDATE_PREFIX: Final = "partition-free-v1-signed-work-pre-c-candidate-"
_COORDINATE_PREFIX: Final = "partition-free-v1-signed-work-pre-c-coordinate-"
_PHYSICAL_PREFIX: Final = "partition-free-v1-signed-work-pre-c-physical-"
_FAMILY_PREFIX: Final = "partition-free-v1-signed-work-pre-c-family-"
_EXECUTION_BUNDLE_PREFIX: Final = (
    "partition-free-v1-signed-work-pre-c-packed-execution-"
)
_PRODUCTION_SOURCE_ISSUANCE_TOKEN: Final = object()
_PRIVATE_REFERENCE_SOURCE_ISSUANCE_TOKEN: Final = object()


class V1SignedWorkPreCRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    SOURCE_OR_CHART_MISMATCH = "source_or_chart_mismatch"
    PRODUCTION_CUDA_REQUIRED = "production_cuda_required"
    PRIVATE_REFERENCE_REQUIRED = "private_reference_required"
    UNRESOLVED_PHASE_A = "unresolved_phase_a"
    INFEASIBLE_PHASE_A = "infeasible_phase_a"
    PHASE_B_UNRESOLVED = "phase_b_unresolved"
    HARD_FQ_AMBIGUOUS = "hard_fq_ambiguous"
    NO_Q_SAFE_ACTION = "no_q_safe_action"
    WORKSPACE_FORECAST_EXCEEDED = "workspace_forecast_exceeded"
    AMBIGUOUS_PHASE_A_BRANCH_MEMBERSHIP = "ambiguous_phase_a_branch_membership"
    HASH_SUBSTITUTION = "hash_substitution"
    RECEIPT_INCONSISTENT = "receipt_inconsistent"
    AUTHORITY_VIOLATION = "authority_violation"
    EXECUTION_FAILURE = "execution_failure"
    CLEANUP_FAILURE = "cleanup_failure"
    LIVE_COMMON_COORDINATE_UNAVAILABLE = "live_common_coordinate_unavailable"


class V1SignedWorkPreCRefusal(RuntimeError):
    def __init__(self, code: V1SignedWorkPreCRefusalCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1SignedWorkPreCRefusalCode, message: str) -> NoReturn:
    raise V1SignedWorkPreCRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkPreCRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _closed_authority(values: tuple[bool, ...], *, owner: str) -> None:
    if any(type(value) is not bool for value in values):
        _refuse(
            V1SignedWorkPreCRefusalCode.INVALID_TYPE,
            f"{owner} authority fields must be exact booleans",
        )
    if any(values):
        _refuse(
            V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} crossed the pre-C authority boundary",
        )


def _relaxed_branch(
    branch: V1SignedWorkLeastWorkBranch,
) -> V1RelaxedKKTBranch:
    if branch is V1SignedWorkLeastWorkBranch.SPECTRAL:
        return V1RelaxedKKTBranch.SPECTRAL
    if branch is V1SignedWorkLeastWorkBranch.DIAGONAL_L1:
        return V1RelaxedKKTBranch.DIAGONAL_L1
    _refuse(
        V1SignedWorkPreCRefusalCode.AMBIGUOUS_PHASE_A_BRANCH_MEMBERSHIP,
        "Phase A emitted an unknown Q branch",
    )


def _coordinate_branch_partition(
    coordinate: V1SignedWorkEndogenousResourceCoordinate,
) -> tuple[tuple[V1RelaxedKKTBranch, ...], tuple[V1RelaxedKKTBranch, ...]]:
    """Return one deterministic closed partition of the canonical Q union."""

    if type(coordinate) is not V1SignedWorkEndogenousResourceCoordinate:
        raise TypeError("Phase-A coordinate type is invalid")
    witnessed = tuple(coordinate.witnessed_feasible_branches)
    excluded = tuple(coordinate.certified_excluded_branches)
    canonical_source = (
        V1SignedWorkLeastWorkBranch.SPECTRAL,
        V1SignedWorkLeastWorkBranch.DIAGONAL_L1,
    )
    if (
        type(coordinate.branch_membership_closed) is not bool
        or not coordinate.branch_membership_closed
        or not witnessed
        or any(type(value) is not V1SignedWorkLeastWorkBranch for value in witnessed)
        or any(type(value) is not V1SignedWorkLeastWorkBranch for value in excluded)
        or len(set(witnessed)) != len(witnessed)
        or len(set(excluded)) != len(excluded)
        or set(witnessed).intersection(excluded)
        or set(witnessed).union(excluded) != set(canonical_source)
    ):
        _refuse(
            V1SignedWorkPreCRefusalCode.AMBIGUOUS_PHASE_A_BRANCH_MEMBERSHIP,
            "Phase A did not close an unambiguous Q-branch membership partition",
        )
    supporting = tuple(
        _relaxed_branch(value) for value in canonical_source if value in witnessed
    )
    certified_excluded = tuple(
        _relaxed_branch(value) for value in canonical_source if value in excluded
    )
    return supporting, certified_excluded


def _readonly_required_action(
    value: np.ndarray | None,
    *,
    route_count: int,
) -> np.ndarray | None:
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.dtype != np.dtype(np.bool_) or raw.shape != (route_count,):
        _refuse(
            V1SignedWorkPreCRefusalCode.INVALID_TYPE,
            "required hard action must be one exact bool physical route axis",
        )
    frozen = np.frombuffer(raw.tobytes(order="C"), dtype=np.bool_)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, slots=True)
class V1SignedWorkPreCSourceBundle:
    """One authoritative same-D response/signed-work source registration."""

    response_geometry: V1RelaxedKKTProblem = field(repr=False, compare=False)
    signed_work_operator: V1SignedWorkOperatorCUDA = field(
        repr=False,
        compare=False,
    )
    source_registration_hash: str
    registered_d_coordinate_hash: str
    source_bundle_authoritative: bool
    required_hard_action: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    c_stream_outcomes_seen: bool = False
    calibration_outcomes_seen: bool = False
    oracle_outcomes_seen: bool = False
    validation_outcomes_seen: bool = False
    schema: str = SIGNED_WORK_PRE_C_SOURCE_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)
    _source_issuance_mode: str = field(init=False, repr=False)
    _bundle_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.response_geometry) is not V1RelaxedKKTProblem:
            raise TypeError("pre-C source requires exact relaxed-KKT geometry")
        if type(self.signed_work_operator) is not V1SignedWorkOperatorCUDA:
            raise TypeError("pre-C source requires exact signed-work operator")
        for name in ("source_registration_hash", "registered_d_coordinate_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.source_bundle_authoritative) is not bool:
            raise TypeError("source_bundle_authoritative must be an exact bool")
        if not self.source_bundle_authoritative:
            _refuse(
                V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
                "pre-C execution requires an authoritative same-D source bundle",
            )
        if self._issuance_token is _PRODUCTION_SOURCE_ISSUANCE_TOKEN:
            issuance_mode = "sampled_same_D_production_registration"
        elif self._issuance_token is _PRIVATE_REFERENCE_SOURCE_ISSUANCE_TOKEN:
            issuance_mode = "private_cpu_reference_test_only"
        else:
            _refuse(
                V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
                "a public boolean cannot issue an authoritative pre-C source",
            )
        object.__setattr__(self, "_source_issuance_mode", issuance_mode)
        _closed_authority(
            (
                self.c_stream_outcomes_seen,
                self.calibration_outcomes_seen,
                self.oracle_outcomes_seen,
                self.validation_outcomes_seen,
            ),
            owner="pre-C source bundle",
        )
        if self.schema != SIGNED_WORK_PRE_C_SOURCE_SCHEMA:
            raise ValueError("unsupported pre-C source schema")
        geometry = self.response_geometry
        operator_receipt = self.signed_work_operator.receipt
        operator_receipt.assert_integrity()
        if (
            operator_receipt.route_count != geometry.route_count
            or operator_receipt.route_axis_hash != geometry.route_axis_hash
            or operator_receipt.source_geometry_hash != geometry.source_geometry_hash
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.SOURCE_OR_CHART_MISMATCH,
                "literal-J geometry and signed corrected-work operator do not share "
                "one registered chart",
            )
        required = _readonly_required_action(
            self.required_hard_action,
            route_count=geometry.route_count,
        )
        object.__setattr__(self, "required_hard_action", required)
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "response_geometry_problem_hash": self.response_geometry.problem_hash,
            "signed_work_operator_receipt_hash": self.signed_work_operator.receipt_hash,
            "source_registration_hash": self.source_registration_hash,
            "registered_d_coordinate_hash": self.registered_d_coordinate_hash,
            "source_bundle_authoritative": self.source_bundle_authoritative,
            "source_issuance_mode": self._source_issuance_mode,
            "required_hard_action_hash": (
                None
                if self.required_hard_action is None
                else content_hash(self.required_hard_action)
            ),
            "c_stream_outcomes_seen": self.c_stream_outcomes_seen,
            "calibration_outcomes_seen": self.calibration_outcomes_seen,
            "oracle_outcomes_seen": self.oracle_outcomes_seen,
            "validation_outcomes_seen": self.validation_outcomes_seen,
        }

    def assert_integrity(self) -> str:
        expected_mode = (
            "sampled_same_D_production_registration"
            if self._issuance_token is _PRODUCTION_SOURCE_ISSUANCE_TOKEN
            else (
                "private_cpu_reference_test_only"
                if self._issuance_token is _PRIVATE_REFERENCE_SOURCE_ISSUANCE_TOKEN
                else None
            )
        )
        if expected_mode is None or self._source_issuance_mode != expected_mode:
            _refuse(
                V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
                "pre-C source issuance custody changed after registration",
            )
        if (
            type(self.response_geometry) is not V1RelaxedKKTProblem
            or type(self.signed_work_operator) is not V1SignedWorkOperatorCUDA
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.SOURCE_OR_CHART_MISMATCH,
                "pre-C source types changed after registration",
            )
        self.signed_work_operator.receipt.assert_integrity()
        geometry = self.response_geometry
        operator_receipt = self.signed_work_operator.receipt
        if (
            operator_receipt.route_count != geometry.route_count
            or operator_receipt.route_axis_hash != geometry.route_axis_hash
            or operator_receipt.source_geometry_hash != geometry.source_geometry_hash
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.SOURCE_OR_CHART_MISMATCH,
                "literal-J geometry and signed corrected-work operator no longer share "
                "the registered chart",
            )
        observed = content_hash(self._payload(), prefix=_SOURCE_PREFIX)
        if self._bundle_hash is not None and observed != self._bundle_hash:
            _refuse(
                V1SignedWorkPreCRefusalCode.HASH_SUBSTITUTION,
                "pre-C source identity was substituted after registration",
            )
        object.__setattr__(self, "_bundle_hash", observed)
        return observed

    @property
    def bundle_hash(self) -> str:
        return self.assert_integrity()


def _issue_v1_signed_work_pre_c_source_bundle_from_sampled_registration(
    *,
    response_geometry: V1RelaxedKKTProblem,
    signed_work_operator: V1SignedWorkOperatorCUDA,
    registered_d_coordinate_hash: str,
    registration_receipt: object,
    required_hard_action: np.ndarray | None = None,
) -> V1SignedWorkPreCSourceBundle:
    """Issue production source custody from the closed sampled same-D receipt."""

    # This local import avoids an initialization cycle.  At issuance time the
    # bridge is loaded and its registration receipt constructor is token-closed.
    from .modern_v1_sampled_signed_work_pre_c_bridge import (
        V1SampledSignedWorkPreCJointRegistrationReceipt,
    )

    if (
        type(registration_receipt)
        is not V1SampledSignedWorkPreCJointRegistrationReceipt
    ):
        _refuse(
            V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
            "production pre-C source issuance requires the exact sampled registration receipt",
        )
    receipt_hash = registration_receipt.assert_integrity()
    if (
        not registration_receipt.production_cuda_proposal_input_authority
        or registration_receipt.private_cpu_reference_test_only
        or registration_receipt.response_geometry_problem_hash
        != response_geometry.problem_hash
        or registration_receipt.response_geometry_source_hash
        != response_geometry.source_geometry_hash
        or registration_receipt.signed_work_operator_receipt_hash
        != signed_work_operator.receipt_hash
        or registration_receipt.registered_d_coordinate_hash
        != registered_d_coordinate_hash
    ):
        _refuse(
            V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
            "sampled registration does not authorize this production pre-C source",
        )
    return V1SignedWorkPreCSourceBundle(
        response_geometry=response_geometry,
        signed_work_operator=signed_work_operator,
        source_registration_hash=receipt_hash,
        registered_d_coordinate_hash=registered_d_coordinate_hash,
        source_bundle_authoritative=True,
        required_hard_action=required_hard_action,
        _issuance_token=_PRODUCTION_SOURCE_ISSUANCE_TOKEN,
    )


def _issue_v1_signed_work_pre_c_source_bundle_private_cpu_reference_for_tests(
    *,
    response_geometry: V1RelaxedKKTProblem,
    signed_work_operator: V1SignedWorkOperatorCUDA,
    source_registration_hash: str,
    registered_d_coordinate_hash: str,
    required_hard_action: np.ndarray | None = None,
) -> V1SignedWorkPreCSourceBundle:
    """Issue tiny-domain test custody; production execution rejects this mode."""

    return V1SignedWorkPreCSourceBundle(
        response_geometry=response_geometry,
        signed_work_operator=signed_work_operator,
        source_registration_hash=source_registration_hash,
        registered_d_coordinate_hash=registered_d_coordinate_hash,
        source_bundle_authoritative=True,
        required_hard_action=required_hard_action,
        _issuance_token=_PRIVATE_REFERENCE_SOURCE_ISSUANCE_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkPreCPolicyBundle:
    phase_a: V1SignedWorkResourceFrontierCUDAPolicy
    phase_b: V1SignedWorkLeastWorkCUDAPolicy
    shell: V1SignedWorkDeviceShellExecutionDesign
    hard_fq: V1SignedWorkHardFQReplayDesign
    schema: str = SIGNED_WORK_PRE_C_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if type(self.phase_a) is not V1SignedWorkResourceFrontierCUDAPolicy:
            raise TypeError("pre-C Phase-A policy type is invalid")
        if type(self.phase_b) is not V1SignedWorkLeastWorkCUDAPolicy:
            raise TypeError("pre-C Phase-B policy type is invalid")
        if type(self.shell) is not V1SignedWorkDeviceShellExecutionDesign:
            raise TypeError("pre-C shell design type is invalid")
        if type(self.hard_fq) is not V1SignedWorkHardFQReplayDesign:
            raise TypeError("pre-C hard-F/Q design type is invalid")
        if self.schema != SIGNED_WORK_PRE_C_POLICY_SCHEMA:
            raise ValueError("unsupported pre-C policy schema")

    @property
    def policy_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "phase_a_policy_hash": self.phase_a.policy_hash,
                "phase_b_policy_hash": self.phase_b.policy_hash,
                "shell_design_hash": self.shell.design_hash,
                "hard_fq_design_hash": self.hard_fq.design_hash,
            },
            prefix=_POLICY_PREFIX,
        )


def bind_v1_signed_work_pre_c_policy_bundle_to_h200_qualification(
    *,
    benchmark_artifact: Mapping[str, object],
    phase_a_unqualified: V1SignedWorkResourceFrontierCUDAPolicy,
    phase_b: V1SignedWorkLeastWorkCUDAPolicy,
    shell: V1SignedWorkDeviceShellExecutionDesign,
    hard_fq: V1SignedWorkHardFQReplayDesign,
) -> V1SignedWorkPreCPolicyBundle:
    """Bind one pre-registered pre-C policy family to exact H200 evidence.

    Numerical policy is supplied explicitly and remains outcome-independent.  The
    benchmark can only mint the narrow source-specific Phase-A execution gate;
    it cannot alter Phase B, rounding, hard-F/Q replay, or downstream
    authority.  A serialized legacy boolean or postflight projection is never
    accepted in place of the complete benchmark artifact.
    """

    if not isinstance(benchmark_artifact, Mapping):
        raise TypeError("pre-C H200 binding requires a benchmark mapping")
    if type(phase_a_unqualified) is not V1SignedWorkResourceFrontierCUDAPolicy:
        raise TypeError("pre-C H200 binding requires an exact Phase-A policy")
    if (
        phase_a_unqualified.route_scale_cuda_qualification is not None
        or phase_a_unqualified.route_scale_cuda_solver_qualified
    ):
        _refuse(
            V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
            "pre-C H200 binding requires the closed unqualified Phase-A policy",
        )
    receipt = issue_v1_signed_work_resource_frontier_cuda_h200_qualification(
        benchmark_artifact=benchmark_artifact,
        policy=phase_a_unqualified,
    )
    receipt.assert_integrity()
    qualified_phase_a = replace(
        phase_a_unqualified,
        route_scale_cuda_qualification=receipt,
    )
    return V1SignedWorkPreCPolicyBundle(
        phase_a=qualified_phase_a,
        phase_b=phase_b,
        shell=shell,
        hard_fq=hard_fq,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkPreCCandidateKey:
    resource_coordinate_hash: str
    q_branch: V1RelaxedKKTBranch
    source_endpoint_label: int
    schema: str = SIGNED_WORK_PRE_C_KEY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resource_coordinate_hash",
            _identity(self.resource_coordinate_hash, name="resource coordinate hash"),
        )
        if type(self.q_branch) is not V1RelaxedKKTBranch:
            raise TypeError("pre-C candidate key requires one exact Q branch")
        if isinstance(self.source_endpoint_label, bool) or not isinstance(
            self.source_endpoint_label,
            int,
        ):
            raise TypeError("source endpoint label must be one positive integer")
        if self.source_endpoint_label <= 0:
            raise ValueError("source endpoint label must be positive")
        if self.schema != SIGNED_WORK_PRE_C_KEY_SCHEMA:
            raise ValueError("unsupported pre-C candidate-key schema")

    @property
    def key_hash(self) -> str:
        return content_hash(self, prefix=_KEY_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkPreCCandidateReceipt:
    key: V1SignedWorkPreCCandidateKey
    coordinate_ordinal: int
    coordinate_resource_floor: float
    least_work_problem_hash: str
    least_work_execution_hash: str
    least_work_branch_receipt_hash: str
    coordinate_endpoint_plan_hash: str
    shell_execution_hash: str
    hard_fq_bundle_hash: str
    hard_fq_branch_receipt_hash: str
    endpoint: V1SignedWorkHardFQEndpointReceipt
    validation_data_seen: bool = False
    c_stream_authority: bool = False
    nonlinear_remainder_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_PRE_C_PACKED_EXECUTION_AUTHORITY
    schema: str = SIGNED_WORK_PRE_C_CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.key) is not V1SignedWorkPreCCandidateKey:
            raise TypeError("pre-C candidate receipt requires an exact key")
        if type(self.endpoint) is not V1SignedWorkHardFQEndpointReceipt:
            raise TypeError(
                "pre-C candidate receipt requires an exact hard-F/Q endpoint"
            )
        if (
            self.endpoint.branch is not self.key.q_branch
            or self.endpoint.endpoint_label != self.key.source_endpoint_label
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                "candidate key and hard-F/Q source endpoint differ",
            )
        if (
            isinstance(self.coordinate_ordinal, bool)
            or not isinstance(
                self.coordinate_ordinal,
                int,
            )
            or self.coordinate_ordinal <= 0
        ):
            raise ValueError("coordinate ordinal must be one positive integer")
        floor = float(self.coordinate_resource_floor)
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("coordinate resource floor must be positive finite")
        object.__setattr__(self, "coordinate_resource_floor", floor)
        for name in (
            "least_work_problem_hash",
            "least_work_execution_hash",
            "least_work_branch_receipt_hash",
            "coordinate_endpoint_plan_hash",
            "shell_execution_hash",
            "hard_fq_bundle_hash",
            "hard_fq_branch_receipt_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        _closed_authority(
            (
                self.validation_data_seen,
                self.c_stream_authority,
                self.nonlinear_remainder_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="pre-C candidate receipt",
        )
        if (
            self.authority_scope != SIGNED_WORK_PRE_C_PACKED_EXECUTION_AUTHORITY
            or self.schema != SIGNED_WORK_PRE_C_CANDIDATE_SCHEMA
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
                "pre-C candidate semantics changed",
            )

    @property
    def physical_action_provenance_hash(self) -> str:
        return content_hash(
            {
                "packed_action_hash": self.endpoint.packed_action_hash,
                "resulting_mask_hash": self.endpoint.resulting_mask_hash,
                "removed_route_index_hash": self.endpoint.removed_route_index_hash,
            },
            prefix=_PHYSICAL_PREFIX,
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_CANDIDATE_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkPreCCoordinateReceipt:
    coordinate: V1SignedWorkEndogenousResourceCoordinate
    least_work_problem_hash: str
    supporting_q_branches: tuple[V1RelaxedKKTBranch, ...]
    excluded_q_branches: tuple[V1RelaxedKKTBranch, ...]
    least_work_execution_hashes: tuple[str, ...]
    least_work_branch_receipt_hashes: tuple[str, ...]
    endpoint_plan_hash: str
    shell_execution_hashes: tuple[str, ...]
    hard_fq_bundle_hashes: tuple[str, ...]
    hard_fq_branch_receipt_hashes: tuple[str, ...]
    candidate_keys: tuple[V1SignedWorkPreCCandidateKey, ...]
    phase_a_branch_membership_closed: bool = True
    exact_endogenous_resource_floor_used: bool = True
    coordinate_specific_solve: bool = True
    common_endpoint_plan_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_PRE_C_AUTHORITY
    schema: str = SIGNED_WORK_PRE_C_COORDINATE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.coordinate) is not V1SignedWorkEndogenousResourceCoordinate:
            raise TypeError(
                "pre-C coordinate receipt requires exact Phase-A coordinate"
            )
        for name in (
            "least_work_problem_hash",
            "endpoint_plan_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        supporting = tuple(self.supporting_q_branches)
        excluded = tuple(self.excluded_q_branches)
        canonical = (
            V1RelaxedKKTBranch.SPECTRAL,
            V1RelaxedKKTBranch.DIAGONAL_L1,
        )
        if (
            not supporting
            or any(type(value) is not V1RelaxedKKTBranch for value in supporting)
            or any(type(value) is not V1RelaxedKKTBranch for value in excluded)
            or tuple(value for value in canonical if value in supporting) != supporting
            or tuple(value for value in canonical if value in excluded) != excluded
            or set(supporting).intersection(excluded)
            or set(supporting).union(excluded) != set(canonical)
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.AMBIGUOUS_PHASE_A_BRANCH_MEMBERSHIP,
                "coordinate receipt does not carry one closed canonical Phase-A "
                "branch partition",
            )
        expected_supporting, expected_excluded = _coordinate_branch_partition(
            self.coordinate
        )
        if supporting != expected_supporting or excluded != expected_excluded:
            _refuse(
                V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                "coordinate receipt changed Phase-A branch membership",
            )
        object.__setattr__(self, "supporting_q_branches", supporting)
        object.__setattr__(self, "excluded_q_branches", excluded)
        hash_fields = (
            "least_work_execution_hashes",
            "least_work_branch_receipt_hashes",
            "shell_execution_hashes",
            "hard_fq_bundle_hashes",
            "hard_fq_branch_receipt_hashes",
        )
        for field_name in hash_fields:
            hashes = tuple(
                _identity(value, name=field_name) for value in getattr(self, field_name)
            )
            if len(hashes) != len(supporting) or len(set(hashes)) != len(hashes):
                raise ValueError(
                    f"{field_name} must bind every supporting branch exactly once"
                )
            object.__setattr__(self, field_name, hashes)
        keys = tuple(self.candidate_keys)
        if len(keys) != len(supporting) or any(
            type(value) is not V1SignedWorkPreCCandidateKey for value in keys
        ):
            raise TypeError("coordinate receipt requires one key per supporting branch")
        if tuple(value.q_branch for value in keys) != supporting or any(
            value.resource_coordinate_hash != self.coordinate.coordinate_hash
            for value in keys
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                "coordinate receipt candidate axis is stale or reordered",
            )
        object.__setattr__(self, "candidate_keys", keys)
        flags = (
            self.phase_a_branch_membership_closed,
            self.exact_endogenous_resource_floor_used,
            self.coordinate_specific_solve,
            self.common_endpoint_plan_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags) or flags != (
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
                "coordinate solve was not exact-floor, independent, and pre-C only",
            )
        if (
            self.authority_scope != SIGNED_WORK_PRE_C_AUTHORITY
            or self.schema != SIGNED_WORK_PRE_C_COORDINATE_SCHEMA
        ):
            raise ValueError("pre-C coordinate semantics changed")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_COORDINATE_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkPreCPhysicalActionGroup:
    provenance_hash: str
    packed_action_hash: str
    resulting_mask_hash: str
    removed_route_index_hash: str
    candidate_keys: tuple[V1SignedWorkPreCCandidateKey, ...]
    candidates_deduplicated: bool = False
    schema: str = SIGNED_WORK_PRE_C_PHYSICAL_GROUP_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "provenance_hash",
            "packed_action_hash",
            "resulting_mask_hash",
            "removed_route_index_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        keys = tuple(self.candidate_keys)
        if not keys or any(
            type(value) is not V1SignedWorkPreCCandidateKey for value in keys
        ):
            raise TypeError("physical-action provenance requires exact candidate keys")
        if len({value.key_hash for value in keys}) != len(keys):
            raise ValueError("physical-action provenance repeats a candidate key")
        object.__setattr__(self, "candidate_keys", keys)
        expected = content_hash(
            {
                "packed_action_hash": self.packed_action_hash,
                "resulting_mask_hash": self.resulting_mask_hash,
                "removed_route_index_hash": self.removed_route_index_hash,
            },
            prefix=_PHYSICAL_PREFIX,
        )
        if self.provenance_hash != expected:
            _refuse(
                V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                "physical-action provenance hash does not replay",
            )
        if (
            type(self.candidates_deduplicated) is not bool
            or self.candidates_deduplicated
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
                "duplicate physical actions must retain every coordinate/branch provenance",
            )
        if self.schema != SIGNED_WORK_PRE_C_PHYSICAL_GROUP_SCHEMA:
            raise ValueError("unsupported physical-action provenance schema")

    @property
    def group_hash(self) -> str:
        return content_hash(self, prefix=_PHYSICAL_PREFIX + "group-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkPreCReceiptFamily:
    source_bundle_hash: str
    response_geometry_problem_hash: str
    signed_work_operator_receipt_hash: str
    source_registration_hash: str
    registered_d_coordinate_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    required_hard_action_hash: str | None
    policy_bundle_hash: str
    phase_a_problem_hash: str
    phase_a_result_hash: str
    phase_a_execution_hash: str
    coordinate_receipts: tuple[V1SignedWorkPreCCoordinateReceipt, ...]
    candidates: tuple[V1SignedWorkPreCCandidateReceipt, ...]
    physical_action_groups: tuple[V1SignedWorkPreCPhysicalActionGroup, ...]
    receipt_only: bool = True
    all_coordinates_solved_independently: bool = True
    common_endpoint_plan_authority: bool = False
    relaxed_state_evicted: bool = True
    device_labels_evicted: bool = True
    signed_work_operator_evicted: bool = True
    validation_data_seen: bool = False
    c_stream_authority: bool = False
    nonlinear_remainder_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_PRE_C_AUTHORITY
    schema: str = SIGNED_WORK_PRE_C_FAMILY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "source_bundle_hash",
            "response_geometry_problem_hash",
            "signed_work_operator_receipt_hash",
            "source_registration_hash",
            "registered_d_coordinate_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "policy_bundle_hash",
            "phase_a_problem_hash",
            "phase_a_result_hash",
            "phase_a_execution_hash",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if isinstance(self.route_count, bool) or not isinstance(self.route_count, int):
            raise TypeError("pre-C receipt route count must be one positive integer")
        if self.route_count <= 0:
            raise ValueError("pre-C receipt route count must be positive")
        if self.required_hard_action_hash is not None:
            object.__setattr__(
                self,
                "required_hard_action_hash",
                _identity(self.required_hard_action_hash, name="required action hash"),
            )
        coordinates = tuple(self.coordinate_receipts)
        candidates = tuple(self.candidates)
        groups = tuple(self.physical_action_groups)
        if not coordinates or any(
            type(value) is not V1SignedWorkPreCCoordinateReceipt
            for value in coordinates
        ):
            raise TypeError("pre-C family requires exact coordinate receipts")
        if not candidates or any(
            type(value) is not V1SignedWorkPreCCandidateReceipt for value in candidates
        ):
            raise TypeError("pre-C family requires exact candidate receipts")
        if not groups or any(
            type(value) is not V1SignedWorkPreCPhysicalActionGroup for value in groups
        ):
            raise TypeError("pre-C family requires physical-action provenance groups")
        if tuple(value.coordinate.ordinal for value in coordinates) != tuple(
            range(1, len(coordinates) + 1)
        ):
            raise ValueError("pre-C coordinates are incomplete or reordered")
        execution_hashes = tuple(
            execution_hash
            for coordinate in coordinates
            for execution_hash in coordinate.least_work_execution_hashes
        )
        if len({value.least_work_problem_hash for value in coordinates}) != len(
            coordinates
        ) or len(set(execution_hashes)) != len(execution_hashes):
            _refuse(
                V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                "distinct resource coordinates did not receive independent Phase-B solves",
            )
        candidate_hashes = tuple(value.key.key_hash for value in candidates)
        expected_hashes = tuple(
            key.key_hash
            for coordinate in coordinates
            for key in coordinate.candidate_keys
        )
        if candidate_hashes != expected_hashes or len(set(candidate_hashes)) != len(
            candidate_hashes
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                "pre-C candidate family is incomplete, reordered, or key-collapsed",
            )
        candidates_by_key = {
            candidate.key.key_hash: candidate for candidate in candidates
        }
        for coordinate in coordinates:
            for index, key in enumerate(coordinate.candidate_keys):
                candidate = candidates_by_key[key.key_hash]
                if (
                    candidate.least_work_problem_hash
                    != coordinate.least_work_problem_hash
                    or candidate.least_work_execution_hash
                    != coordinate.least_work_execution_hashes[index]
                    or candidate.least_work_branch_receipt_hash
                    != coordinate.least_work_branch_receipt_hashes[index]
                    or candidate.coordinate_endpoint_plan_hash
                    != coordinate.endpoint_plan_hash
                    or candidate.shell_execution_hash
                    != coordinate.shell_execution_hashes[index]
                    or candidate.hard_fq_bundle_hash
                    != coordinate.hard_fq_bundle_hashes[index]
                    or candidate.hard_fq_branch_receipt_hash
                    != coordinate.hard_fq_branch_receipt_hashes[index]
                ):
                    _refuse(
                        V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                        "candidate and coordinate branch receipts differ",
                    )
        grouped_hashes = tuple(
            key.key_hash for group in groups for key in group.candidate_keys
        )
        if sorted(grouped_hashes) != sorted(candidate_hashes):
            _refuse(
                V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                "physical-action groups lost or duplicated candidate provenance",
            )
        group_by_provenance = {group.provenance_hash: group for group in groups}
        if len(group_by_provenance) != len(groups) or any(
            candidate.physical_action_provenance_hash not in group_by_provenance
            for candidate in candidates
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                "candidate physical provenance is not represented exactly once",
            )
        object.__setattr__(self, "coordinate_receipts", coordinates)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "physical_action_groups", groups)
        flags = (
            self.receipt_only,
            self.all_coordinates_solved_independently,
            self.common_endpoint_plan_authority,
            self.relaxed_state_evicted,
            self.device_labels_evicted,
            self.signed_work_operator_evicted,
            self.validation_data_seen,
            self.c_stream_authority,
            self.nonlinear_remainder_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags) or flags != (
            True,
            True,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
                "pre-C receipt family retained mutable state or downstream authority",
            )
        if (
            self.authority_scope != SIGNED_WORK_PRE_C_AUTHORITY
            or self.schema != SIGNED_WORK_PRE_C_FAMILY_SCHEMA
        ):
            raise ValueError("pre-C receipt-family semantics changed")

    @property
    def candidate_keys(self) -> tuple[V1SignedWorkPreCCandidateKey, ...]:
        return tuple(value.key for value in self.candidates)

    def candidate_for_key(
        self,
        key: V1SignedWorkPreCCandidateKey,
    ) -> V1SignedWorkPreCCandidateReceipt:
        if type(key) is not V1SignedWorkPreCCandidateKey:
            raise TypeError("candidate lookup requires an exact pre-C key")
        matches = tuple(value for value in self.candidates if value.key == key)
        if len(matches) != 1:
            raise KeyError(key)
        return matches[0]

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_FAMILY_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkPreCExecutionBundle:
    """Receipt family plus exact Q-safe physical actions for the C bridge.

    The bundle is created while every coordinate/branch label lease is still
    live, then returned only after relaxed state, labels, and the signed-work
    operator have been evicted.  It is an identity carrier, not permission to
    observe C, calibrate, query the oracle, commit, or claim terminality.
    """

    receipt_family: V1SignedWorkPreCReceiptFamily
    packed_action_family: V1SignedWorkPreCPackedActionFamily = field(
        repr=False,
        compare=False,
    )
    packed_actions_materialized_before_label_eviction: bool = True
    all_q_ambiguous_aliases_refused: bool = True
    c_outcomes_seen: bool = False
    c_measurement_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_PRE_C_AUTHORITY
    schema: str = SIGNED_WORK_PRE_C_EXECUTION_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.receipt_family) is not V1SignedWorkPreCReceiptFamily:
            raise TypeError("pre-C execution bundle requires an exact receipt family")
        if type(self.packed_action_family) is not V1SignedWorkPreCPackedActionFamily:
            raise TypeError("pre-C execution bundle requires an exact packed family")
        receipt = self.receipt_family
        packed = self.packed_action_family
        if (
            packed.parent_mask_hash != receipt.parent_mask_hash
            or packed.route_axis_hash != receipt.route_axis_hash
            or packed.route_count != receipt.route_count
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                "packed actions do not share the receipt family's parent route axis",
            )
        candidates = {value.key.key_hash: value for value in receipt.candidates}
        crosswalks = {
            value.coordinate_candidate_key_hash: value
            for value in packed.candidate_crosswalks
        }
        if set(candidates) != set(crosswalks) or len(crosswalks) != len(
            packed.candidate_crosswalks
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                "packed actions lost, duplicated, or invented a coordinate candidate",
            )
        for key_hash, candidate in candidates.items():
            crosswalk = crosswalks[key_hash]
            if (
                crosswalk.resource_coordinate_hash
                != candidate.key.resource_coordinate_hash
                or crosswalk.coordinate_ordinal != candidate.coordinate_ordinal
                or crosswalk.q_branch is not candidate.key.q_branch
                or crosswalk.source_endpoint_label
                != candidate.key.source_endpoint_label
                or crosswalk.endpoint_decision.hard_replay_endpoint_receipt_hash
                != candidate.endpoint.receipt_hash
                or crosswalk.q_status is not candidate.endpoint.status
            ):
                _refuse(
                    V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                    "packed-action crosswalk differs from its exact hard-F/Q receipt",
                )
        if packed.q_ambiguous_crosswalk_hashes:
            _refuse(
                V1SignedWorkPreCRefusalCode.HARD_FQ_AMBIGUOUS,
                "an ambiguous hard-F/Q endpoint cannot cross the C identity bridge",
            )
        flags = (
            self.packed_actions_materialized_before_label_eviction,
            self.all_q_ambiguous_aliases_refused,
            self.c_outcomes_seen,
            self.c_measurement_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags) or flags != (
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
                "pre-C packed execution retained live custody or downstream authority",
            )
        if (
            self.authority_scope != SIGNED_WORK_PRE_C_AUTHORITY
            or self.schema != SIGNED_WORK_PRE_C_EXECUTION_BUNDLE_SCHEMA
        ):
            raise ValueError("pre-C packed execution semantics changed")

    @property
    def has_q_safe_physical_action(self) -> bool:
        return bool(self.packed_action_family.canonical_actions)

    @property
    def bundle_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "receipt_family_hash": self.receipt_family.receipt_hash,
                "packed_action_family_hash": self.packed_action_family.family_hash,
                "packed_actions_materialized_before_label_eviction": (
                    self.packed_actions_materialized_before_label_eviction
                ),
                "all_q_ambiguous_aliases_refused": (
                    self.all_q_ambiguous_aliases_refused
                ),
                "authority_scope": self.authority_scope,
            },
            prefix=_EXECUTION_BUNDLE_PREFIX,
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkPreCLiveCommonCoordinateContext:
    """Ephemeral two-branch custody at the only valid production-D seam.

    The context exists only while one exact Phase-A coordinate has live KKT
    state and shell labels for both registered Q branches.  A consumer must
    finish synchronously.  The coordinator then evicts every retained owner;
    no handle is copied into the durable pre-C execution bundle.
    """

    source: V1SignedWorkPreCSourceBundle = field(repr=False, compare=False)
    parent_mask: HardMask = field(repr=False, compare=False)
    policies: V1SignedWorkPreCPolicyBundle = field(repr=False, compare=False)
    phase_a_problem: V1SignedWorkResourceFrontierCUDAProblem = field(
        repr=False,
        compare=False,
    )
    phase_a_execution: V1SignedWorkResourceFrontierCUDAExecutionBundle = field(
        repr=False,
        compare=False,
    )
    coordinate: V1SignedWorkEndogenousResourceCoordinate
    problem: V1SignedWorkLeastWorkCUDAProblem = field(repr=False, compare=False)
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt
    branch_executions: tuple[
        V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
        V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    ] = field(repr=False, compare=False)
    branch_shells: tuple[
        V1SignedWorkDeviceShellExecutionBundle,
        V1SignedWorkDeviceShellExecutionBundle,
    ] = field(repr=False, compare=False)
    branch_hard_fq: tuple[
        V1SignedWorkHardFQSingleBranchBundle,
        V1SignedWorkHardFQSingleBranchBundle,
    ] = field(repr=False, compare=False)
    candidate_crosswalks: tuple[V1SignedWorkCoordinateCandidateCrosswalk, ...]
    private_cpu_reference_execution: bool
    c_outcomes_seen: bool = False
    calibration_outcomes_seen: bool = False
    oracle_outcomes_seen: bool = False
    commit_outcomes_seen: bool = False
    durable_live_custody_authority: bool = False
    schema: str = SIGNED_WORK_PRE_C_LIVE_COMMON_COORDINATE_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.source) is not V1SignedWorkPreCSourceBundle
            or type(self.parent_mask) is not HardMask
            or type(self.policies) is not V1SignedWorkPreCPolicyBundle
            or type(self.phase_a_problem)
            is not V1SignedWorkResourceFrontierCUDAProblem
            or type(self.phase_a_execution)
            is not V1SignedWorkResourceFrontierCUDAExecutionBundle
            or type(self.coordinate) is not V1SignedWorkEndogenousResourceCoordinate
            or type(self.problem) is not V1SignedWorkLeastWorkCUDAProblem
            or type(self.endpoint_plan)
            is not V1SignedWorkDeviceShellEndpointPlanReceipt
            or type(self.private_cpu_reference_execution) is not bool
        ):
            raise TypeError("live common-coordinate context lost an exact owner")
        if (
            self.phase_a_problem.source_registration_hash
            != self.source.source_registration_hash
            or self.phase_a_problem.registered_d_coordinate_hash
            != self.source.registered_d_coordinate_hash
            or self.phase_a_problem.problem_hash
            != self.phase_a_execution.result.problem_hash
            or (
                not self.private_cpu_reference_execution
                and not self.phase_a_execution.theorem_state_available
            )
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.LIVE_COMMON_COORDINATE_UNAVAILABLE,
                "live production-D custody lacks its exact Phase-A theorem state",
            )
        executions = tuple(self.branch_executions)
        shells = tuple(self.branch_shells)
        hard_fq = tuple(self.branch_hard_fq)
        if (
            len(executions) != 2
            or len(shells) != 2
            or len(hard_fq) != 2
            or any(
                type(value) is not V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle
                for value in executions
            )
            or any(
                type(value) is not V1SignedWorkDeviceShellExecutionBundle
                for value in shells
            )
            or any(
                type(value) is not V1SignedWorkHardFQSingleBranchBundle
                for value in hard_fq
            )
        ):
            raise TypeError(
                "live common-coordinate context requires two exact branches"
            )
        branches = tuple(value.branch for value in executions)
        expected = (
            V1RelaxedKKTBranch.SPECTRAL,
            V1RelaxedKKTBranch.DIAGONAL_L1,
        )
        if branches != expected:
            _refuse(
                V1SignedWorkPreCRefusalCode.LIVE_COMMON_COORDINATE_UNAVAILABLE,
                "live production-D custody is not the exact two-branch axis",
            )
        for execution, shell, replay in zip(
            executions,
            shells,
            hard_fq,
            strict=True,
        ):
            state = execution.state_handle
            if (
                execution.branch_receipt.branch is not execution.branch
                or state is None
                or state.is_evicted
                or shell.branch_identity.branch is not execution.branch
                or shell.label_handle.is_evicted
                or replay.receipt.branch is not execution.branch
                or replay.receipt.problem_hash != self.problem.problem_hash
            ):
                _refuse(
                    V1SignedWorkPreCRefusalCode.LIVE_COMMON_COORDINATE_UNAVAILABLE,
                    "one common-coordinate KKT/shell/replay owner is stale or mismatched",
                )
        crosswalks = tuple(self.candidate_crosswalks)
        if (
            len(crosswalks) != 2
            or tuple(value.q_branch for value in crosswalks) != expected
            or any(
                value.resource_coordinate_hash != self.coordinate.coordinate_hash
                for value in crosswalks
            )
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.LIVE_COMMON_COORDINATE_UNAVAILABLE,
                "common-coordinate packed-action crosswalk is incomplete",
            )
        if (
            any(
                type(value) is not bool or value
                for value in (
                    self.c_outcomes_seen,
                    self.calibration_outcomes_seen,
                    self.oracle_outcomes_seen,
                    self.commit_outcomes_seen,
                    self.durable_live_custody_authority,
                )
            )
            or self.schema != SIGNED_WORK_PRE_C_LIVE_COMMON_COORDINATE_SCHEMA
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
                "live common-coordinate context crossed the pre-C boundary",
            )
        object.__setattr__(self, "branch_executions", executions)
        object.__setattr__(self, "branch_shells", shells)
        object.__setattr__(self, "branch_hard_fq", hard_fq)
        object.__setattr__(self, "candidate_crosswalks", crosswalks)

    def __reduce_ex__(self, protocol: int) -> None:
        del protocol
        raise TypeError("live common-coordinate contexts are nonserializable")


V1SignedWorkPreCLiveCommonCoordinateConsumer = Callable[
    [V1SignedWorkPreCLiveCommonCoordinateContext], None
]


def _validate_parent_mask(
    source: V1SignedWorkPreCSourceBundle, parent: HardMask
) -> None:
    if type(parent) is not HardMask:
        raise TypeError("pre-C execution requires an exact parent HardMask")
    geometry = source.response_geometry
    if (
        parent.route_count != geometry.route_count
        or parent.route_axis_hash != geometry.route_axis_hash
    ):
        _refuse(
            V1SignedWorkPreCRefusalCode.SOURCE_OR_CHART_MISMATCH,
            "parent mask does not share the registered physical route axis",
        )
    if source.required_hard_action is not None:
        parent_live = parent.to_bool_array()
        if bool(np.any(source.required_hard_action & ~parent_live)):
            _refuse(
                V1SignedWorkPreCRefusalCode.SOURCE_OR_CHART_MISMATCH,
                "required hard action contains a parent-dead route",
            )


def _validate_execution_mode(
    source: V1SignedWorkPreCSourceBundle,
    policies: V1SignedWorkPreCPolicyBundle,
    *,
    private_cpu_reference: bool,
) -> None:
    receipt = source.signed_work_operator.receipt
    if private_cpu_reference:
        if not (
            source._issuance_token is _PRIVATE_REFERENCE_SOURCE_ISSUANCE_TOKEN
            and receipt.private_cpu_reference_execution
            and not receipt.production_cuda_execution
            and not policies.phase_a.production_cuda_required
            and policies.phase_a.allow_private_cpu_reference
            and not policies.phase_b.production_cuda_required
            and policies.phase_b.allow_private_cpu_reference
            and not policies.shell.production_cuda_required
            and policies.shell.allow_private_cpu_reference
            and not policies.hard_fq.production_cuda_required
            and policies.hard_fq.allow_private_cpu_reference
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.PRIVATE_REFERENCE_REQUIRED,
                "the test-only entry point requires explicit private CPU custody at "
                "every execution seam",
            )
        return
    if not (
        source._issuance_token is _PRODUCTION_SOURCE_ISSUANCE_TOKEN
        and receipt.production_cuda_execution
        and not receipt.private_cpu_reference_execution
        and policies.phase_a.production_cuda_required
        and not policies.phase_a.allow_private_cpu_reference
        and policies.phase_b.production_cuda_required
        and not policies.phase_b.allow_private_cpu_reference
        and policies.shell.production_cuda_required
        and not policies.shell.allow_private_cpu_reference
        and policies.hard_fq.production_cuda_required
        and not policies.hard_fq.allow_private_cpu_reference
    ):
        _refuse(
            V1SignedWorkPreCRefusalCode.PRODUCTION_CUDA_REQUIRED,
            "production pre-C execution is CUDA-only and has no CPU fallback",
        )


def _evict_coordinate_custody(
    execution: V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle | None,
    shell: V1SignedWorkDeviceShellExecutionBundle | None,
) -> None:
    if shell is not None and not shell.label_handle.is_evicted:
        shell.label_handle.evict()
    if (
        execution is not None
        and execution.state_handle is not None
        and not execution.state_handle.is_evicted
    ):
        execution.state_handle.evict()


def _physical_groups(
    candidates: tuple[V1SignedWorkPreCCandidateReceipt, ...],
) -> tuple[V1SignedWorkPreCPhysicalActionGroup, ...]:
    grouped: dict[
        str,
        tuple[str, str, str, list[V1SignedWorkPreCCandidateKey]],
    ] = {}
    order: list[str] = []
    for candidate in candidates:
        provenance = candidate.physical_action_provenance_hash
        endpoint = candidate.endpoint
        if provenance not in grouped:
            order.append(provenance)
            grouped[provenance] = (
                endpoint.packed_action_hash,
                endpoint.resulting_mask_hash,
                endpoint.removed_route_index_hash,
                [],
            )
        packed, mask, removed, keys = grouped[provenance]
        if (
            packed != endpoint.packed_action_hash
            or mask != endpoint.resulting_mask_hash
            or removed != endpoint.removed_route_index_hash
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                "one physical provenance hash mapped to different hard actions",
            )
        keys.append(candidate.key)
    return tuple(
        V1SignedWorkPreCPhysicalActionGroup(
            provenance_hash=provenance,
            packed_action_hash=grouped[provenance][0],
            resulting_mask_hash=grouped[provenance][1],
            removed_route_index_hash=grouped[provenance][2],
            candidate_keys=tuple(grouped[provenance][3]),
        )
        for provenance in order
    )


def _execute_pre_c(
    *,
    source: V1SignedWorkPreCSourceBundle,
    parent_mask: HardMask,
    policies: V1SignedWorkPreCPolicyBundle,
    private_cpu_reference: bool,
    materialize_packed_actions: bool,
    c_bridge_design: V1SignedWorkCBridgeDesign,
    device_total_memory_bytes: int | None,
    current_reserved_bytes: int | None,
    live_common_coordinate_consumer: (
        V1SignedWorkPreCLiveCommonCoordinateConsumer | None
    ) = None,
) -> V1SignedWorkPreCReceiptFamily | V1SignedWorkPreCExecutionBundle:
    if type(source) is not V1SignedWorkPreCSourceBundle:
        raise TypeError("pre-C execution requires an exact source bundle")
    if type(policies) is not V1SignedWorkPreCPolicyBundle:
        raise TypeError("pre-C execution requires an exact frozen policy bundle")
    if type(materialize_packed_actions) is not bool:
        raise TypeError("packed-action materialization mode must be an exact bool")
    if type(c_bridge_design) is not V1SignedWorkCBridgeDesign:
        raise TypeError("pre-C execution requires an exact C-bridge design")
    if live_common_coordinate_consumer is not None and not callable(
        live_common_coordinate_consumer
    ):
        raise TypeError("live common-coordinate consumer must be callable")
    if live_common_coordinate_consumer is not None and not materialize_packed_actions:
        _refuse(
            V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
            "a live production-D consumer requires packed-action custody",
        )
    operator = source.signed_work_operator
    active_executions: list[V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle] = []
    active_shells: list[V1SignedWorkDeviceShellExecutionBundle] = []
    cleanup_error: Exception | None = None
    completed = False
    live_common_coordinate_consumed = False
    try:
        source_hash = source.assert_integrity()
        _validate_parent_mask(source, parent_mask)
        _validate_execution_mode(
            source,
            policies,
            private_cpu_reference=private_cpu_reference,
        )
        geometry = source.response_geometry
        phase_a_problem = (
            build_v1_signed_work_resource_frontier_cuda_problem_from_relaxed_kkt(
                problem_id=f"{source.source_registration_hash}-pre-c-phase-a",
                response_geometry=geometry,
                source_registration_hash=source.source_registration_hash,
                registered_d_coordinate_hash=source.registered_d_coordinate_hash,
                source_bundle_authoritative=source.source_bundle_authoritative,
                required_hard_action=source.required_hard_action,
            )
        )
        phase_a = solve_v1_signed_work_resource_frontier_cuda(
            phase_a_problem,
            policies.phase_a,
            # A common hard endpoint plan would conflate distinct resource
            # coordinates.  Every Phase-B solve derives its own one-coordinate
            # plan below.
            derive_endpoint_plan=False,
            device_total_memory_bytes=device_total_memory_bytes,
            current_reserved_bytes=current_reserved_bytes,
        )
        phase_a_result = phase_a.result.reference_result
        if phase_a.result.endpoint_plan is not None or (
            phase_a.result.coordinate_hashes_bound_to_endpoint_plan
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.AUTHORITY_VIOLATION,
                "Phase A unexpectedly emitted a shared endpoint plan",
            )
        if phase_a_result.union_status is V1SignedWorkResourceUnionStatus.UNRESOLVED:
            _refuse(
                V1SignedWorkPreCRefusalCode.UNRESOLVED_PHASE_A,
                "the Q-branch resource frontier did not close",
            )
        if (
            phase_a_result.union_status
            is not V1SignedWorkResourceUnionStatus.CERTIFIED_COORDINATES
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.INFEASIBLE_PHASE_A,
                "the Q-branch resource frontier certified no advancing action",
            )
        coordinates = phase_a_result.require_phase_b_coordinates()
        coordinate_receipts: list[V1SignedWorkPreCCoordinateReceipt] = []
        candidate_receipts: list[V1SignedWorkPreCCandidateReceipt] = []
        packed_crosswalks: list[V1SignedWorkCoordinateCandidateCrosswalk] = []
        packed_retained_host_bytes = 0
        for coordinate in coordinates:
            supporting_branches, excluded_branches = _coordinate_branch_partition(
                coordinate
            )
            retain_common_coordinate = (
                live_common_coordinate_consumer is not None
                and not live_common_coordinate_consumed
                and supporting_branches
                == (
                    V1RelaxedKKTBranch.SPECTRAL,
                    V1RelaxedKKTBranch.DIAGONAL_L1,
                )
            )
            coordinate_executions: list[
                V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle
            ] = []
            coordinate_shells: list[V1SignedWorkDeviceShellExecutionBundle] = []
            coordinate_hard_fq: list[V1SignedWorkHardFQSingleBranchBundle] = []
            coordinate_crosswalk_start = len(packed_crosswalks)
            # No carrier/placeholder problem is constructed.  The first
            # Phase-B problem for this coordinate already has the exact
            # endogenous Phase-A floor.
            problem = V1SignedWorkLeastWorkCUDAProblem(
                source_problem=geometry,
                signed_work_operator=operator,
                resource_floor=coordinate.total_resource_floor,
                source_registration_hash=source.source_registration_hash,
                registered_d_coordinate_hash=source.registered_d_coordinate_hash,
                source_bundle_authoritative=source.source_bundle_authoritative,
                required_hard_action=source.required_hard_action,
            )
            endpoint_plan = (
                derive_v1_signed_work_endpoint_plan_from_resource_coordinate(
                    problem=phase_a_problem,
                    coordinate=coordinate,
                )
            )
            keys: list[V1SignedWorkPreCCandidateKey] = []
            execution_hashes: list[str] = []
            branch_receipt_hashes: list[str] = []
            shell_hashes: list[str] = []
            hard_fq_bundle_hashes: list[str] = []
            hard_fq_receipt_hashes: list[str] = []
            for branch in supporting_branches:
                execution = (
                    solve_v1_signed_work_least_work_branch_private_cpu_reference(
                        problem=problem,
                        policy=policies.phase_b,
                        branch=branch,
                    )
                    if private_cpu_reference
                    else solve_v1_signed_work_least_work_branch_cuda(
                        problem=problem,
                        policy=policies.phase_b,
                        branch=branch,
                    )
                )
                active_executions.append(execution)
                coordinate_executions.append(execution)
                if (
                    execution.branch is not branch
                    or execution.branch_receipt.branch is not branch
                    or not execution.result.branch_converged
                    or not execution.result.relaxed_state_available
                    or execution.state_handle is None
                    or execution.result.nonrequested_branch_authority
                ):
                    _refuse(
                        V1SignedWorkPreCRefusalCode.PHASE_B_UNRESOLVED,
                        "one explicitly supported exact-coordinate Q branch did not close",
                    )
                shell = freeze_v1_signed_work_device_shell_single_branch(
                    problem=problem,
                    policy=policies.phase_b,
                    execution=execution,
                    endpoint_plan=endpoint_plan,
                    parent_mask=parent_mask,
                    design=policies.shell,
                    device_total_memory_bytes=device_total_memory_bytes,
                    current_reserved_bytes=current_reserved_bytes,
                )
                active_shells.append(shell)
                coordinate_shells.append(shell)
                hard_fq = replay_v1_signed_work_hard_fq_single_branch_cuda(
                    problem=problem,
                    policy=policies.phase_b,
                    execution=execution,
                    shell=shell,
                    design=policies.hard_fq,
                    device_total_memory_bytes=device_total_memory_bytes,
                    current_reserved_bytes=current_reserved_bytes,
                )
                branch_receipt = hard_fq.receipt
                coordinate_hard_fq.append(hard_fq)
                if branch_receipt.branch is not branch:
                    _refuse(
                        V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                        "single-branch hard replay changed its Phase-A membership",
                    )
                if len(branch_receipt.endpoints) != 1:
                    _refuse(
                        V1SignedWorkPreCRefusalCode.RECEIPT_INCONSISTENT,
                        "a coordinate-specific endpoint plan must replay exactly one source "
                        "endpoint for each supporting Q branch",
                    )
                endpoint = branch_receipt.endpoints[0]
                key = V1SignedWorkPreCCandidateKey(
                    resource_coordinate_hash=coordinate.coordinate_hash,
                    q_branch=branch,
                    source_endpoint_label=endpoint.endpoint_label,
                )
                if materialize_packed_actions:
                    remaining_host_cap = (
                        c_bridge_design.maximum_host_packed_payload_bytes
                        - packed_retained_host_bytes
                    )
                    if remaining_host_cap <= 0:
                        _refuse(
                            V1SignedWorkPreCRefusalCode.WORKSPACE_FORECAST_EXCEEDED,
                            "retained packed actions exhausted the registered host cap",
                        )
                    source_design = replace(
                        c_bridge_design,
                        maximum_host_packed_payload_bytes=remaining_host_cap,
                    )
                    try:
                        materialization = materialize_v1_signed_work_device_shell_c_single_branch_actions(
                            shell=shell,
                            hard_fq=hard_fq,
                            parent_mask=parent_mask,
                            design=source_design,
                        )
                    except V1SignedWorkCBridgeRefusal as error:
                        if (
                            error.code
                            is V1SignedWorkCBridgeRefusalCode.WORKSPACE_FORECAST_EXCEEDED
                        ):
                            _refuse(
                                V1SignedWorkPreCRefusalCode.WORKSPACE_FORECAST_EXCEEDED,
                                "coordinate-aware packed-action host/CUDA forecast "
                                f"refused before label borrow: {error}",
                            )
                        raise
                    packed_retained_host_bytes += (
                        materialization.retained_host_packed_payload_bytes
                    )
                    packed_crosswalks.append(
                        crosswalk_v1_signed_work_coordinate_candidate(
                            coordinate_candidate_key_hash=key.key_hash,
                            resource_coordinate_hash=coordinate.coordinate_hash,
                            coordinate_ordinal=coordinate.ordinal,
                            materialization=materialization,
                            source_endpoint_label=endpoint.endpoint_label,
                        )
                    )
                keys.append(key)
                execution_hashes.append(execution.execution_hash)
                branch_receipt_hashes.append(execution.branch_receipt.receipt_hash)
                shell_hashes.append(shell.execution_hash)
                hard_fq_bundle_hashes.append(hard_fq.bundle_hash)
                hard_fq_receipt_hashes.append(branch_receipt.receipt_hash)
                candidate_receipts.append(
                    V1SignedWorkPreCCandidateReceipt(
                        key=key,
                        coordinate_ordinal=coordinate.ordinal,
                        coordinate_resource_floor=coordinate.total_resource_floor,
                        least_work_problem_hash=problem.problem_hash,
                        least_work_execution_hash=execution.execution_hash,
                        least_work_branch_receipt_hash=(
                            execution.branch_receipt.receipt_hash
                        ),
                        coordinate_endpoint_plan_hash=endpoint_plan.receipt_hash,
                        shell_execution_hash=shell.execution_hash,
                        hard_fq_bundle_hash=hard_fq.bundle_hash,
                        hard_fq_branch_receipt_hash=branch_receipt.receipt_hash,
                        endpoint=endpoint,
                    )
                )
                # Each branch continuation owns independent retained state and
                # labels.  Release it immediately after the complete hard F/Q
                # replay so one coordinate/branch cannot become implicit state
                # for another.
                if not retain_common_coordinate:
                    _evict_coordinate_custody(execution, shell)
            if retain_common_coordinate:
                common_crosswalks = tuple(
                    packed_crosswalks[coordinate_crosswalk_start:]
                )
                context = V1SignedWorkPreCLiveCommonCoordinateContext(
                    source=source,
                    parent_mask=parent_mask,
                    policies=policies,
                    phase_a_problem=phase_a_problem,
                    phase_a_execution=phase_a,
                    coordinate=coordinate,
                    problem=problem,
                    endpoint_plan=endpoint_plan,
                    branch_executions=tuple(coordinate_executions),
                    branch_shells=tuple(coordinate_shells),
                    branch_hard_fq=tuple(coordinate_hard_fq),
                    candidate_crosswalks=common_crosswalks,
                    private_cpu_reference_execution=private_cpu_reference,
                )
                live_common_coordinate_consumer(context)
                live_common_coordinate_consumed = True
                for execution, shell in zip(
                    coordinate_executions,
                    coordinate_shells,
                    strict=True,
                ):
                    _evict_coordinate_custody(execution, shell)
            coordinate_receipts.append(
                V1SignedWorkPreCCoordinateReceipt(
                    coordinate=coordinate,
                    least_work_problem_hash=problem.problem_hash,
                    supporting_q_branches=supporting_branches,
                    excluded_q_branches=excluded_branches,
                    least_work_execution_hashes=tuple(execution_hashes),
                    least_work_branch_receipt_hashes=tuple(branch_receipt_hashes),
                    endpoint_plan_hash=endpoint_plan.receipt_hash,
                    shell_execution_hashes=tuple(shell_hashes),
                    hard_fq_bundle_hashes=tuple(hard_fq_bundle_hashes),
                    hard_fq_branch_receipt_hashes=tuple(hard_fq_receipt_hashes),
                    candidate_keys=tuple(keys),
                )
            )

        if (
            live_common_coordinate_consumer is not None
            and not live_common_coordinate_consumed
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.LIVE_COMMON_COORDINATE_UNAVAILABLE,
                "Phase A produced no exact common two-branch coordinate for the live hook",
            )
        operator_receipt_hash = operator.receipt_hash
        operator.evict()
        candidates = tuple(candidate_receipts)
        receipt_family = V1SignedWorkPreCReceiptFamily(
            source_bundle_hash=source_hash,
            response_geometry_problem_hash=geometry.problem_hash,
            signed_work_operator_receipt_hash=operator_receipt_hash,
            source_registration_hash=source.source_registration_hash,
            registered_d_coordinate_hash=source.registered_d_coordinate_hash,
            parent_mask_hash=parent_mask.mask_hash,
            route_axis_hash=parent_mask.route_axis_hash,
            route_count=parent_mask.route_count,
            required_hard_action_hash=(
                None
                if source.required_hard_action is None
                else content_hash(source.required_hard_action)
            ),
            policy_bundle_hash=policies.policy_hash,
            phase_a_problem_hash=phase_a_problem.problem_hash,
            phase_a_result_hash=phase_a_result.result_hash,
            phase_a_execution_hash=phase_a.execution_hash,
            coordinate_receipts=tuple(coordinate_receipts),
            candidates=candidates,
            physical_action_groups=_physical_groups(candidates),
        )
        if not materialize_packed_actions:
            completed = True
            return receipt_family
        packed_family = build_v1_signed_work_pre_c_packed_action_family(
            candidate_crosswalks=tuple(packed_crosswalks),
        )
        if (
            packed_family.maximum_source_peak_host_workspace_upper_bound_bytes
            > c_bridge_design.maximum_host_packed_payload_bytes
        ):
            _refuse(
                V1SignedWorkPreCRefusalCode.WORKSPACE_FORECAST_EXCEEDED,
                "coordinate-aware packed-action peak exceeds the registered host cap",
            )
        if packed_family.q_ambiguous_crosswalk_hashes:
            _refuse(
                V1SignedWorkPreCRefusalCode.HARD_FQ_AMBIGUOUS,
                "hard-F/Q ambiguity remained after complete endpoint replay",
            )
        if not packed_family.canonical_actions:
            _refuse(
                V1SignedWorkPreCRefusalCode.NO_Q_SAFE_ACTION,
                "complete hard-F/Q replay certified no nonempty action for C",
            )
        bundle = V1SignedWorkPreCExecutionBundle(
            receipt_family=receipt_family,
            packed_action_family=packed_family,
        )
        completed = True
        return bundle
    except V1SignedWorkPreCRefusal:
        raise
    except Exception as error:
        _refuse(
            V1SignedWorkPreCRefusalCode.EXECUTION_FAILURE,
            f"pre-C execution failed closed: {type(error).__name__}: {error}",
        )
    finally:
        for execution, shell in zip(active_executions, active_shells):
            try:
                _evict_coordinate_custody(execution, shell)
            except Exception as error:  # pragma: no cover - defensive cleanup path.
                cleanup_error = error
        # A failure may happen after a solver but before a shell exists.
        for execution in active_executions[len(active_shells) :]:
            try:
                _evict_coordinate_custody(execution, None)
            except Exception as error:  # pragma: no cover - defensive cleanup path.
                cleanup_error = error
        try:
            operator.evict()
        except Exception as error:  # pragma: no cover - defensive cleanup path.
            cleanup_error = error
        if cleanup_error is not None and not completed:
            raise V1SignedWorkPreCRefusal(
                V1SignedWorkPreCRefusalCode.CLEANUP_FAILURE,
                f"pre-C failure cleanup did not close: {cleanup_error}",
            ) from cleanup_error


def execute_v1_signed_work_pre_c(
    *,
    source: V1SignedWorkPreCSourceBundle,
    parent_mask: HardMask,
    policies: V1SignedWorkPreCPolicyBundle,
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1SignedWorkPreCReceiptFamily:
    """Execute the production CUDA-only Phase-A/Phase-B pre-C joint."""

    result = _execute_pre_c(
        source=source,
        parent_mask=parent_mask,
        policies=policies,
        private_cpu_reference=False,
        materialize_packed_actions=False,
        c_bridge_design=V1SignedWorkCBridgeDesign(),
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
    )
    if type(result) is not V1SignedWorkPreCReceiptFamily:  # pragma: no cover.
        raise AssertionError("receipt-only execution returned packed custody")
    return result


def execute_v1_signed_work_pre_c_with_packed_actions(
    *,
    source: V1SignedWorkPreCSourceBundle,
    parent_mask: HardMask,
    policies: V1SignedWorkPreCPolicyBundle,
    c_bridge_design: V1SignedWorkCBridgeDesign = V1SignedWorkCBridgeDesign(),
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
    live_common_coordinate_consumer: (
        V1SignedWorkPreCLiveCommonCoordinateConsumer | None
    ) = None,
) -> V1SignedWorkPreCExecutionBundle:
    """Execute production pre-C and retain exact Q-safe packed actions.

    This is the canonical identity bridge into a subsequently registered C
    design.  It fails closed when complete hard-F/Q replay leaves an ambiguous
    endpoint and carries no C, K, O, commit, terminal, or Pareto authority.
    """

    result = _execute_pre_c(
        source=source,
        parent_mask=parent_mask,
        policies=policies,
        private_cpu_reference=False,
        materialize_packed_actions=True,
        c_bridge_design=c_bridge_design,
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
        live_common_coordinate_consumer=live_common_coordinate_consumer,
    )
    if type(result) is not V1SignedWorkPreCExecutionBundle:  # pragma: no cover.
        raise AssertionError("packed pre-C execution returned a receipt-only family")
    return result


def _execute_v1_signed_work_pre_c_private_cpu_reference_for_tests(
    *,
    source: V1SignedWorkPreCSourceBundle,
    parent_mask: HardMask,
    policies: V1SignedWorkPreCPolicyBundle,
    device_total_memory_bytes: int = 1 << 30,
    current_reserved_bytes: int = 0,
) -> V1SignedWorkPreCReceiptFamily:
    """Focused tiny-domain reference; never selected by the production entry point."""

    result = _execute_pre_c(
        source=source,
        parent_mask=parent_mask,
        policies=policies,
        private_cpu_reference=True,
        materialize_packed_actions=False,
        c_bridge_design=V1SignedWorkCBridgeDesign(
            production_cuda_required=False,
            allow_private_cpu_reference=True,
        ),
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
    )
    if type(result) is not V1SignedWorkPreCReceiptFamily:  # pragma: no cover.
        raise AssertionError("private receipt-only execution returned packed custody")
    return result


def _execute_v1_signed_work_pre_c_with_packed_actions_private_cpu_reference_for_tests(
    *,
    source: V1SignedWorkPreCSourceBundle,
    parent_mask: HardMask,
    policies: V1SignedWorkPreCPolicyBundle,
    c_bridge_design: V1SignedWorkCBridgeDesign = V1SignedWorkCBridgeDesign(
        production_cuda_required=False,
        allow_private_cpu_reference=True,
    ),
    device_total_memory_bytes: int = 1 << 30,
    current_reserved_bytes: int = 0,
    live_common_coordinate_consumer: (
        V1SignedWorkPreCLiveCommonCoordinateConsumer | None
    ) = None,
) -> V1SignedWorkPreCExecutionBundle:
    """Tiny-domain exact action-custody reference; never a production fallback."""

    result = _execute_pre_c(
        source=source,
        parent_mask=parent_mask,
        policies=policies,
        private_cpu_reference=True,
        materialize_packed_actions=True,
        c_bridge_design=c_bridge_design,
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
        live_common_coordinate_consumer=live_common_coordinate_consumer,
    )
    if type(result) is not V1SignedWorkPreCExecutionBundle:  # pragma: no cover.
        raise AssertionError("private packed execution returned receipt-only custody")
    return result


__all__ = [
    "SIGNED_WORK_PRE_C_AUTHORITY",
    "SIGNED_WORK_PRE_C_CANDIDATE_SCHEMA",
    "SIGNED_WORK_PRE_C_COORDINATE_SCHEMA",
    "SIGNED_WORK_PRE_C_EXECUTION_BUNDLE_SCHEMA",
    "SIGNED_WORK_PRE_C_FAMILY_SCHEMA",
    "SIGNED_WORK_PRE_C_KEY_SCHEMA",
    "SIGNED_WORK_PRE_C_LIVE_COMMON_COORDINATE_SCHEMA",
    "SIGNED_WORK_PRE_C_PACKED_EXECUTION_AUTHORITY",
    "SIGNED_WORK_PRE_C_PHYSICAL_GROUP_SCHEMA",
    "SIGNED_WORK_PRE_C_POLICY_SCHEMA",
    "SIGNED_WORK_PRE_C_SOURCE_SCHEMA",
    "V1SignedWorkPreCCandidateKey",
    "V1SignedWorkPreCCandidateReceipt",
    "V1SignedWorkPreCCoordinateReceipt",
    "V1SignedWorkPreCExecutionBundle",
    "V1SignedWorkPreCLiveCommonCoordinateContext",
    "V1SignedWorkPreCPhysicalActionGroup",
    "V1SignedWorkPreCPolicyBundle",
    "V1SignedWorkPreCReceiptFamily",
    "V1SignedWorkPreCRefusal",
    "V1SignedWorkPreCRefusalCode",
    "V1SignedWorkPreCSourceBundle",
    "bind_v1_signed_work_pre_c_policy_bundle_to_h200_qualification",
    "execute_v1_signed_work_pre_c",
    "execute_v1_signed_work_pre_c_with_packed_actions",
]
