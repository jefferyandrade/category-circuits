"""Source-derived deterministic FD-rank stability for canonical F/Q/R D.

This module is the only bridge from the actual sampled FD/KKT graph to an
adaptive-rank decision certificate.  It never accepts caller-authored margin
scalars.  Hard F/Q classifications are recomputed with outward binary64
enclosures from the frozen KKT arrays, complete rounded ladders, residual
envelope, and legality blocks.

The true response matrix is represented by

``M = H.T @ H + E``, ``E >= 0``, ``lambda_max(E) <= rho``, ``E[j,j] <= d[j]``.

For a hard endpoint ``S`` this gives the certified interval

``Q_L = ||H 1_S||_L**2``

``Q_U = ||H 1_S||_U**2 + min(rho_U |S|,
                              (sum_{j in S} sqrt(d[j]_U))_U**2)``.

Consequently a ``Q_U`` budget violation is *not* an impossibility proof.  An
endpoint is robustly impossible only when ``F_L > tau_F`` or ``Q_L > tau_Q``;
the remaining unresolved case is FD ambiguity and triggers rank promotion (or
the maximum-rank no-C refusal).  Equality is retained as ambiguity.

The current v1 KKT receipt exposes scalar optimality summaries, not the
routewise reduced-advantage and shell-boundary enclosure required to prove a
relaxed face or rounded endpoint set under a nonunique convex face.  This
module therefore records those missing sources explicitly.  Hard-F/Q closure
alone cannot open C.  When the registered KKT v2 witness is available, the
same builder consumes it and freezes the stable *two-branch family*; spectral
and diagonal-l1 are not required to select the same action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral
from typing import Callable, Final, Sequence

import numpy as np

from .contracts import content_hash
from .modern_v1_fqr_adaptive_rank import (
    FQR_ADAPTIVE_RANK_BRANCH_DECISION_V2_SCHEMA,
    FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA,
    V1FQRAdaptiveRankBranchDecision,
    V1FQRAdaptiveRankDecisionCertificate,
    V1FQRAdaptiveRankPolicy,
    V1FQREndogenousRankPolicyV2,
    V1FQRAdaptiveRankRefusal,
    V1FQRAdaptiveRankRefusalCode,
    V1FQRAdaptiveRankReplayLedger,
    V1FQRAdaptiveRankReplayRecord,
    V1FQRAdaptiveRankStabilityCertificate,
    _DECISION_FAMILY_PREFIX,
    _SOURCE_DERIVATION_ISSUANCE_TOKEN,
    certify_v1_fqr_adaptive_rank_stability,
)
from .modern_v1_device_shell_fq_replay import V1DeviceShellHardFQStatus
from .modern_v1_fqr_branch_family import (
    V1FQRDeviceShellSourceStabilityBinding,
    V1FQRKKTBranchFrozenBinding,
)
from .modern_v1_relaxed_kkt import (
    V1RelaxedKKTAdaptiveRankBranchWitness,
    V1RelaxedKKTBranch,
    V1RelaxedKKTBranchResult,
    V1RelaxedKKTProblem,
)
from .modern_v1_sampled_kkt_bridge import V1SampledKKTBranchFamilyGraph
from .modern_v1_stratified_fd_gpu import V1StratifiedFDAssemblyReceipt


FQR_ADAPTIVE_RANK_HARD_ENDPOINT_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_hard_endpoint_interval_v1"
)
FQR_ADAPTIVE_RANK_HARD_REPLAY_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_complete_hard_fq_replay_v1"
)
FQR_ADAPTIVE_RANK_SOURCE_DERIVATION_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_source_derivation_v1"
)
FQR_ADAPTIVE_RANK_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_execution_v1"
)
FQR_ADAPTIVE_RANK_DEVICE_SHELL_SOURCE_DERIVATION_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_device_shell_source_derivation_v1"
)
FQR_ADAPTIVE_RANK_DEVICE_SHELL_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_fqr_adaptive_rank_device_shell_execution_v1"
)
FQR_ADAPTIVE_RANK_OUTWARD_ARITHMETIC_ID: Final = (
    "binary64_nextafter_gamma_n_absolute_sum_hard_fq_interval_v1"
)

_ENDPOINT_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-hard-endpoint-"
_HARD_REPLAY_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-hard-replay-"
_DERIVATION_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-derivation-"
_STABLE_BRANCH_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-stable-branch-"
_ENDPOINT_AXIS_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-endpoint-axis-"
_EXECUTION_PREFIX: Final = "partition-free-v1-fqr-adaptive-rank-execution-"
_DEVICE_SHELL_DERIVATION_PREFIX: Final = (
    "partition-free-v1-fqr-adaptive-rank-device-shell-derivation-"
)
_DEVICE_SHELL_EXECUTION_PREFIX: Final = (
    "partition-free-v1-fqr-adaptive-rank-device-shell-execution-"
)
_BRANCH_AXIS: Final = (V1RelaxedKKTBranch.SPECTRAL, V1RelaxedKKTBranch.DIAGONAL_L1)
_UNIT_ROUNDOFF: Final = float(np.finfo(np.float64).eps) / 2.0
_MIN_NORMAL: Final = float(np.finfo(np.float64).tiny)
_DEFAULT_CHUNK_SIZE: Final = 1 << 16


class V1FQRHardFQStatus(str, Enum):
    SAFE_FEASIBLE = "safe_feasible"
    ROBUST_IMPOSSIBLE = "robust_impossible"
    FD_AMBIGUOUS = "fd_ambiguous"


class V1FQRRankDerivationAmbiguityCode(str, Enum):
    FD_HARD_FQ_BOUNDARY = "fd_hard_fq_boundary"
    RELAXED_FACE_WITNESS_MISSING = "relaxed_face_witness_missing"
    RELAXED_FACE_UNRESOLVED = "relaxed_face_unresolved"
    ROUNDED_ENDPOINT_SET_WITNESS_MISSING = (
        "rounded_endpoint_set_witness_missing"
    )
    ROUNDED_ENDPOINT_SET_UNRESOLVED = "rounded_endpoint_set_unresolved"
    KKT_BRANCH_NOT_CONVERGED = "kkt_branch_not_converged"
    KKT_NUMERICAL_BOUNDARY = "kkt_numerical_boundary"
    RELAXED_PROBLEM_SENSITIVITY_UNCERTIFIED = (
        "relaxed_problem_sensitivity_uncertified"
    )


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _next_down(value: float) -> float:
    result = math.nextafter(float(value), -math.inf)
    if not math.isfinite(result):
        raise ValueError("outward lower endpoint left finite binary64")
    return result


def _next_up(value: float) -> float:
    result = math.nextafter(float(value), math.inf)
    if not math.isfinite(result):
        raise ValueError("outward upper endpoint left finite binary64")
    return result


def _up_add(left: float, right: float) -> float:
    if left == 0.0:
        return float(right)
    if right == 0.0:
        return float(left)
    return _next_up(float(left) + float(right))


def _up_multiply(left: float, right: float) -> float:
    if left == 0.0 or right == 0.0:
        return 0.0
    return _next_up(float(left) * float(right))


def _up_square(value: float) -> float:
    return _up_multiply(value, value)


def _down_square(value: float) -> float:
    if value == 0.0:
        return 0.0
    result = float(value) * float(value)
    if not math.isfinite(result):
        raise ValueError("outward squared lower endpoint overflowed binary64")
    return max(0.0, _next_down(result))


def _gamma_n(operation_count: int) -> float:
    count = _nonnegative_int(operation_count, name="rounding operation count")
    if count == 0:
        return 0.0
    product = _up_multiply(float(count), _UNIT_ROUNDOFF)
    if product >= 0.5:
        raise ValueError("route axis is too large for certified binary64 summation")
    denominator = _next_down(1.0 - product)
    if denominator <= 0.0:
        raise ValueError("gamma_n denominator is nonpositive")
    return _next_up(product / denominator)


def _nonnegative_sum_upper(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all() or bool(np.any(array < 0.0)):
        raise ValueError("outward nonnegative sum received invalid values")
    count = int(array.size)
    if count == 0:
        return 0.0
    nominal = float(np.sum(array, dtype=np.float64))
    if not math.isfinite(nominal):
        raise ValueError("outward nonnegative sum overflowed binary64")
    if count == 1:
        return nominal
    gamma = _gamma_n(count - 1)
    magnitude = _next_up(nominal / _next_down(1.0 - gamma))
    nonzero = int(np.count_nonzero(array.view(np.uint64) & np.uint64(0x7FFF_FFFF_FFFF_FFFF)))
    underflow = _up_multiply(float(nonzero), _MIN_NORMAL)
    return _up_add(magnitude, underflow)


def _signed_sum_interval_from_nominal(
    *,
    nominal: float,
    absolute_sum_upper: float,
    term_count: int,
) -> tuple[float, float]:
    count = _nonnegative_int(term_count, name="signed sum term count")
    center = float(nominal)
    absolute = _finite_nonnegative(
        absolute_sum_upper,
        name="signed sum absolute upper",
    )
    if not math.isfinite(center):
        raise ValueError("signed sum center is nonfinite")
    if count <= 1:
        return center, center
    gamma = _gamma_n(count - 1)
    roundoff = _up_multiply(gamma, absolute)
    underflow = _up_multiply(float(count), _MIN_NORMAL)
    radius = _up_add(roundoff, underflow)
    return _next_down(center - radius), _next_up(center + radius)


def _square_interval(lower: float, upper: float) -> tuple[float, float]:
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("invalid signed interval")
    if lower <= 0.0 <= upper:
        minimum = 0.0
    else:
        minimum = _down_square(min(abs(lower), abs(upper)))
    maximum = _up_square(max(abs(lower), abs(upper)))
    return minimum, maximum


def _nonnegative_interval_sum(
    lowers: Sequence[float],
    uppers: Sequence[float],
) -> tuple[float, float]:
    lower_values = np.asarray(tuple(lowers), dtype=np.float64)
    upper_values = np.asarray(tuple(uppers), dtype=np.float64)
    if lower_values.shape != upper_values.shape:
        raise ValueError("interval sum axes differ")
    upper = _nonnegative_sum_upper(upper_values)
    if not lower_values.size:
        return 0.0, upper
    nominal = float(np.sum(lower_values, dtype=np.float64))
    gamma = _gamma_n(max(0, int(lower_values.size) - 1))
    absolute_upper = _nonnegative_sum_upper(lower_values)
    error = _up_multiply(gamma, absolute_upper)
    lower = max(0.0, _next_down(nominal - error))
    return lower, upper


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankHardEndpointInterval:
    branch: V1RelaxedKKTBranch
    endpoint_index: int
    prefix_count: int
    packed_action_hash: str
    kkt_candidate_hash: str
    resource_lower: float
    resource_upper: float
    f_lower: float
    f_upper: float
    low_rank_q_lower: float
    low_rank_q_upper: float
    spectral_residual_q_upper: float
    diagonal_l1_residual_q_upper: float
    canonical_residual_q_upper: float
    q_upper: float
    work_budget: float
    quadratic_budget: float
    legal: bool
    status: V1FQRHardFQStatus
    equality_boundary: bool
    outward_arithmetic_id: str = FQR_ADAPTIVE_RANK_OUTWARD_ARITHMETIC_ID
    schema: str = FQR_ADAPTIVE_RANK_HARD_ENDPOINT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.branch, V1RelaxedKKTBranch):
            raise TypeError("hard endpoint branch is not registered")
        endpoint = _positive_int(self.endpoint_index, name="endpoint index")
        count = _positive_int(self.prefix_count, name="endpoint prefix count")
        for name in ("packed_action_hash", "kkt_candidate_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        numeric_names = (
            "resource_lower",
            "resource_upper",
            "f_lower",
            "f_upper",
            "low_rank_q_lower",
            "low_rank_q_upper",
            "spectral_residual_q_upper",
            "diagonal_l1_residual_q_upper",
            "canonical_residual_q_upper",
            "q_upper",
            "work_budget",
            "quadratic_budget",
        )
        for name in numeric_names:
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        if (
            self.resource_lower > self.resource_upper
            or self.f_lower > self.f_upper
            or self.low_rank_q_lower > self.low_rank_q_upper
        ):
            raise ValueError("hard endpoint interval is reversed")
        expected_residual = min(
            self.spectral_residual_q_upper,
            self.diagonal_l1_residual_q_upper,
        )
        expected_q = _up_add(self.low_rank_q_upper, expected_residual)
        if (
            self.canonical_residual_q_upper != expected_residual
            or self.q_upper != expected_q
        ):
            raise ValueError("hard endpoint Q upper does not replay its envelope")
        if type(self.legal) is not bool or type(self.equality_boundary) is not bool:
            raise TypeError("hard endpoint legality/boundary flags must be booleans")
        if not isinstance(self.status, V1FQRHardFQStatus):
            raise TypeError("hard endpoint status is not registered")
        equality = any(
            left == right
            for left, right in (
                (self.f_upper, self.work_budget),
                (self.f_lower, self.work_budget),
                (self.q_upper, self.quadratic_budget),
                (self.low_rank_q_lower, self.quadratic_budget),
            )
        )
        safe = (
            self.legal
            and self.f_upper < self.work_budget
            and self.q_upper < self.quadratic_budget
        )
        impossible = (
            not self.legal
            or self.f_lower > self.work_budget
            or self.low_rank_q_lower > self.quadratic_budget
        )
        expected_status = (
            V1FQRHardFQStatus.SAFE_FEASIBLE
            if safe
            else V1FQRHardFQStatus.ROBUST_IMPOSSIBLE
            if impossible
            else V1FQRHardFQStatus.FD_AMBIGUOUS
        )
        if self.status is not expected_status or self.equality_boundary is not equality:
            raise ValueError("hard endpoint status does not replay its intervals")
        if self.outward_arithmetic_id != FQR_ADAPTIVE_RANK_OUTWARD_ARITHMETIC_ID:
            raise ValueError("hard endpoint arithmetic owner changed")
        if self.schema != FQR_ADAPTIVE_RANK_HARD_ENDPOINT_SCHEMA:
            raise ValueError("hard endpoint schema is invalid")
        object.__setattr__(self, "endpoint_index", endpoint)
        object.__setattr__(self, "prefix_count", count)

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_ENDPOINT_PREFIX)


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankCompleteHardFQReplay:
    branch: V1RelaxedKKTBranch
    problem_hash: str
    kkt_result_hash: str
    kkt_branch_result_hash: str
    kkt_branch_receipt_hash: str
    kkt_primal_receipt_hash: str
    kkt_stationarity_receipt_hash: str
    kkt_gap_receipt_hash: str
    kkt_convergence_receipt_hash: str
    rounded_ladder_hash: str
    frozen_branch_binding_hash: str
    frozen_ladder_hash: str
    fd_assembly_receipt_hash: str
    fd_residual_diagonal_hash: str
    legality_hash: str
    eligibility_hash: str
    route_axis_hash: str
    route_count: int
    current_mask_hash: str
    chart_identity_hash: str
    proposal_generation: int
    numerical_enclosure_policy_hash: str
    endpoint_intervals: tuple[V1FQRAdaptiveRankHardEndpointInterval, ...]
    complete_registered_ladder_replayed: bool = True
    nonmonotone_q_early_stop_used: bool = False
    outward_numerical_enclosure_used: bool = True
    schema: str = FQR_ADAPTIVE_RANK_HARD_REPLAY_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.branch, V1RelaxedKKTBranch):
            raise TypeError("hard F/Q replay branch is not registered")
        for name in (
            "problem_hash",
            "kkt_result_hash",
            "kkt_branch_result_hash",
            "kkt_branch_receipt_hash",
            "kkt_primal_receipt_hash",
            "kkt_stationarity_receipt_hash",
            "kkt_gap_receipt_hash",
            "kkt_convergence_receipt_hash",
            "rounded_ladder_hash",
            "frozen_branch_binding_hash",
            "frozen_ladder_hash",
            "fd_assembly_receipt_hash",
            "fd_residual_diagonal_hash",
            "legality_hash",
            "eligibility_hash",
            "route_axis_hash",
            "current_mask_hash",
            "chart_identity_hash",
            "numerical_enclosure_policy_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        generation = _nonnegative_int(
            self.proposal_generation,
            name="proposal generation",
        )
        endpoints = tuple(self.endpoint_intervals)
        if (
            not endpoints
            or any(type(value) is not V1FQRAdaptiveRankHardEndpointInterval for value in endpoints)
            or tuple(value.endpoint_index for value in endpoints)
            != tuple(range(1, len(endpoints) + 1))
            or any(value.branch is not self.branch for value in endpoints)
        ):
            raise ValueError("complete hard F/Q replay endpoint axis is invalid")
        flags = (
            self.complete_registered_ladder_replayed,
            self.nonmonotone_q_early_stop_used,
            self.outward_numerical_enclosure_used,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("hard F/Q replay flags must be exact booleans")
        if flags != (True, False, True):
            raise PermissionError("hard F/Q replay omitted registered endpoints")
        if self.schema != FQR_ADAPTIVE_RANK_HARD_REPLAY_SCHEMA:
            raise ValueError("hard F/Q replay schema is invalid")
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "endpoint_intervals", endpoints)
        self.assert_integrity()

    @property
    def endpoint_status_axis_hash(self) -> str:
        return content_hash(
            tuple(
                (
                    value.endpoint_index,
                    value.packed_action_hash,
                    value.status,
                    value.receipt_hash,
                )
                for value in self.endpoint_intervals
            ),
            prefix=_ENDPOINT_AXIS_PREFIX,
        )

    @property
    def hard_fq_status_axis_stable(self) -> bool:
        # ``FD_AMBIGUOUS`` is a source-derived status, not an omitted action.
        # Such endpoints stay in the pre-frozen family for C to certify.  The
        # hard-F/Q axis is therefore complete when every registered endpoint
        # was replayed and every non-impossible endpoint is retained; it need
        # not pretend the residual envelope resolved finite-dose feasibility.
        retained = set(self.admissible_action_hashes)
        return all(
            (
                value.status is V1FQRHardFQStatus.ROBUST_IMPOSSIBLE
                or value.packed_action_hash in retained
            )
            for value in self.endpoint_intervals
        )

    @property
    def admissible_action_hashes(self) -> tuple[str, ...]:
        # Ambiguous endpoints are retained in the provisional family for C;
        # they may not be silently discarded as impossible.
        return tuple(
            value.packed_action_hash
            for value in self.endpoint_intervals
            if value.status is not V1FQRHardFQStatus.ROBUST_IMPOSSIBLE
        )

    @property
    def safe_action_hashes(self) -> tuple[str, ...]:
        return tuple(
            value.packed_action_hash
            for value in self.endpoint_intervals
            if value.status is V1FQRHardFQStatus.SAFE_FEASIBLE
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "branch": self.branch,
            "problem_hash": self.problem_hash,
            "kkt_result_hash": self.kkt_result_hash,
            "kkt_branch_result_hash": self.kkt_branch_result_hash,
            "kkt_branch_receipt_hash": self.kkt_branch_receipt_hash,
            "kkt_primal_receipt_hash": self.kkt_primal_receipt_hash,
            "kkt_stationarity_receipt_hash": self.kkt_stationarity_receipt_hash,
            "kkt_gap_receipt_hash": self.kkt_gap_receipt_hash,
            "kkt_convergence_receipt_hash": self.kkt_convergence_receipt_hash,
            "rounded_ladder_hash": self.rounded_ladder_hash,
            "frozen_branch_binding_hash": self.frozen_branch_binding_hash,
            "frozen_ladder_hash": self.frozen_ladder_hash,
            "fd_assembly_receipt_hash": self.fd_assembly_receipt_hash,
            "fd_residual_diagonal_hash": self.fd_residual_diagonal_hash,
            "legality_hash": self.legality_hash,
            "eligibility_hash": self.eligibility_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "current_mask_hash": self.current_mask_hash,
            "chart_identity_hash": self.chart_identity_hash,
            "proposal_generation": self.proposal_generation,
            "numerical_enclosure_policy_hash": self.numerical_enclosure_policy_hash,
            "endpoint_receipt_hashes": tuple(
                value.receipt_hash for value in self.endpoint_intervals
            ),
            "endpoint_status_axis_hash": self.endpoint_status_axis_hash,
            "complete_registered_ladder_replayed": (
                self.complete_registered_ladder_replayed
            ),
            "nonmonotone_q_early_stop_used": self.nonmonotone_q_early_stop_used,
            "outward_numerical_enclosure_used": self.outward_numerical_enclosure_used,
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_HARD_REPLAY_PREFIX)
        if self._receipt_hash is not None and self._receipt_hash != observed:
            raise ValueError("complete hard F/Q replay was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("complete hard F/Q replay hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankBranchSourceDerivation:
    policy_hash: str
    rank: int
    branch: V1RelaxedKKTBranch
    graph_hash: str
    sampled_geometry_source_hash: str
    fd_assembly_receipt_hash: str
    problem_hash: str
    kkt_result_hash: str
    branch_family_hash: str
    hard_fq_replay: V1FQRAdaptiveRankCompleteHardFQReplay
    relaxed_face_witness_hash: str | None
    rounded_endpoint_set_witness_hash: str | None
    relaxed_face_stable: bool
    rounded_endpoint_set_stable: bool
    ambiguity_codes: tuple[V1FQRRankDerivationAmbiguityCode, ...]
    stable_branch_identity_hash: str | None
    source_derived: bool = True
    fixed_z_face_perturbation_only: bool = True
    full_kkt_solution_sensitivity_theorem: bool = False
    C_stream_open_authority: bool = False
    schema: str = FQR_ADAPTIVE_RANK_SOURCE_DERIVATION_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "policy_hash",
            "graph_hash",
            "sampled_geometry_source_hash",
            "fd_assembly_receipt_hash",
            "problem_hash",
            "kkt_result_hash",
            "branch_family_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        rank = _positive_int(self.rank, name="FD rank")
        if not isinstance(self.branch, V1RelaxedKKTBranch):
            raise TypeError("source derivation branch is not registered")
        replay = self.hard_fq_replay
        if type(replay) is not V1FQRAdaptiveRankCompleteHardFQReplay:
            raise TypeError("source derivation requires exact hard F/Q replay")
        replay.assert_integrity()
        if replay.branch is not self.branch or replay.problem_hash != self.problem_hash:
            raise ValueError("source derivation hard replay coordinates differ")
        for name in (
            "relaxed_face_witness_hash",
            "rounded_endpoint_set_witness_hash",
            "stable_branch_identity_hash",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identity(value, name=name))
        if (
            type(self.relaxed_face_stable) is not bool
            or type(self.rounded_endpoint_set_stable) is not bool
            or type(self.source_derived) is not bool
            or type(self.C_stream_open_authority) is not bool
        ):
            raise TypeError("source derivation flags must be exact booleans")
        if self.source_derived is not True or self.C_stream_open_authority is not False:
            raise PermissionError("source derivation crossed its pre-C boundary")
        ambiguities = tuple(self.ambiguity_codes)
        if any(not isinstance(value, V1FQRRankDerivationAmbiguityCode) for value in ambiguities):
            raise TypeError("source derivation ambiguity code is not registered")
        if len(set(ambiguities)) != len(ambiguities):
            raise ValueError("source derivation ambiguity codes must be unique")
        stable = (
            self.relaxed_face_stable
            and self.rounded_endpoint_set_stable
            and replay.hard_fq_status_axis_stable
        )
        if self.relaxed_face_stable is not (
            self.relaxed_face_witness_hash is not None
        ):
            raise ValueError("relaxed-face stability requires its exact witness hash")
        if self.rounded_endpoint_set_stable is not (
            self.rounded_endpoint_set_witness_hash is not None
        ):
            raise ValueError(
                "rounded-endpoint stability requires its exact shell witness hash"
            )
        if stable != (self.stable_branch_identity_hash is not None):
            raise ValueError("source derivation stable identity does not replay")
        if stable == bool(ambiguities):
            raise ValueError("source derivation ambiguity/stability state is invalid")
        if self.schema != FQR_ADAPTIVE_RANK_SOURCE_DERIVATION_SCHEMA:
            raise ValueError("source derivation schema is invalid")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "ambiguity_codes", ambiguities)
        self.assert_integrity()

    @property
    def rank_local_stable(self) -> bool:
        return self.stable_branch_identity_hash is not None

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_hash": self.policy_hash,
            "rank": self.rank,
            "branch": self.branch,
            "graph_hash": self.graph_hash,
            "sampled_geometry_source_hash": self.sampled_geometry_source_hash,
            "fd_assembly_receipt_hash": self.fd_assembly_receipt_hash,
            "problem_hash": self.problem_hash,
            "kkt_result_hash": self.kkt_result_hash,
            "branch_family_hash": self.branch_family_hash,
            "hard_fq_replay_hash": self.hard_fq_replay.receipt_hash,
            "relaxed_face_witness_hash": self.relaxed_face_witness_hash,
            "rounded_endpoint_set_witness_hash": (
                self.rounded_endpoint_set_witness_hash
            ),
            "relaxed_face_stable": self.relaxed_face_stable,
            "rounded_endpoint_set_stable": self.rounded_endpoint_set_stable,
            "ambiguity_codes": self.ambiguity_codes,
            "stable_branch_identity_hash": self.stable_branch_identity_hash,
            "source_derived": self.source_derived,
            "C_stream_open_authority": self.C_stream_open_authority,
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_DERIVATION_PREFIX)
        if self._receipt_hash is not None and self._receipt_hash != observed:
            raise ValueError("source derivation receipt was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("source derivation receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankDerivedDecision:
    policy_hash: str
    rank: int
    graph_hash: str
    branch_derivations: tuple[V1FQRAdaptiveRankBranchSourceDerivation, ...]
    decision_certificate: V1FQRAdaptiveRankDecisionCertificate
    schema: str = FQR_ADAPTIVE_RANK_EXECUTION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_hash", _identity(self.policy_hash, name="policy hash"))
        object.__setattr__(self, "graph_hash", _identity(self.graph_hash, name="graph hash"))
        rank = _positive_int(self.rank, name="FD rank")
        branches = tuple(self.branch_derivations)
        if (
            len(branches) != 2
            or any(type(value) is not V1FQRAdaptiveRankBranchSourceDerivation for value in branches)
            or tuple(value.branch for value in branches) != _BRANCH_AXIS
        ):
            raise ValueError("derived decision requires exact two-branch sources")
        certificate = self.decision_certificate
        if type(certificate) is not V1FQRAdaptiveRankDecisionCertificate:
            raise TypeError("derived decision requires exact decision certificate")
        certificate.assert_integrity()
        if (
            certificate.policy_hash != self.policy_hash
            or certificate.rank != rank
            or any(value.policy_hash != self.policy_hash for value in branches)
            or any(value.rank != rank or value.graph_hash != self.graph_hash for value in branches)
            or tuple(value.receipt_hash for value in branches)
            != tuple(
                value.authoritative_typed_source_derivation_receipt_hash
                for value in certificate.branch_decisions
            )
        ):
            raise ValueError("derived decision source graph does not close")
        if self.schema != FQR_ADAPTIVE_RANK_EXECUTION_SCHEMA:
            raise ValueError("derived decision schema is invalid")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "branch_derivations", branches)

    @property
    def result_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "policy_hash": self.policy_hash,
                "rank": self.rank,
                "graph_hash": self.graph_hash,
                "branch_derivation_hashes": tuple(
                    value.receipt_hash for value in self.branch_derivations
                ),
                "decision_certificate_hash": self.decision_certificate.certificate_hash,
            },
            prefix=_EXECUTION_PREFIX,
        )


def _device_shell_endpoint_status_axis_hash(
    source: V1FQRDeviceShellSourceStabilityBinding,
) -> str:
    replay = source.cuda_hard_fq_replay
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


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankDeviceShellBranchSourceDerivation:
    """One source-built shell witness plus complete CUDA hard-F/Q replay."""

    policy_hash: str
    rank: int
    graph_hash: str
    branch_family_hash: str
    source_binding: V1FQRDeviceShellSourceStabilityBinding
    endpoint_status_axis_hash: str
    admissible_action_hashes: tuple[str, ...]
    safe_action_hashes: tuple[str, ...]
    ambiguity_codes: tuple[V1FQRRankDerivationAmbiguityCode, ...]
    stable_branch_identity_hash: str | None
    source_derived: bool = True
    fixed_z_face_perturbation_only: bool = True
    full_kkt_solution_sensitivity_theorem: bool = False
    C_stream_open_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = FQR_ADAPTIVE_RANK_DEVICE_SHELL_SOURCE_DERIVATION_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in ("policy_hash", "graph_hash", "branch_family_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "endpoint_status_axis_hash",
            _identity(
                self.endpoint_status_axis_hash,
                name="endpoint status axis hash",
            ),
        )
        rank = _positive_int(self.rank, name="FD rank")
        source = self.source_binding
        if type(source) is not V1FQRDeviceShellSourceStabilityBinding:
            raise TypeError("device-shell source derivation requires an exact binding")
        source.assert_integrity()
        replay = source.cuda_hard_fq_replay
        admissible = tuple(
            _identity(value, name="admissible packed action hash")
            for value in self.admissible_action_hashes
        )
        safe = tuple(
            _identity(value, name="safe packed action hash")
            for value in self.safe_action_hashes
        )
        ambiguities = tuple(self.ambiguity_codes)
        if any(
            not isinstance(value, V1FQRRankDerivationAmbiguityCode)
            for value in ambiguities
        ):
            raise TypeError("device-shell derivation ambiguity code is unregistered")
        if len(set(ambiguities)) != len(ambiguities):
            raise ValueError("device-shell derivation ambiguity codes are not unique")
        expected_admissible = tuple(replay.admissible_action_hashes)
        expected_safe = tuple(replay.safe_action_hashes)
        expected_axis = _device_shell_endpoint_status_axis_hash(source)
        stable = source.source_stable
        if (
            admissible != expected_admissible
            or safe != expected_safe
            or self.endpoint_status_axis_hash != expected_axis
        ):
            raise ValueError("device-shell derivation does not replay its hard F/Q axis")
        if stable != (self.stable_branch_identity_hash is not None):
            raise ValueError("device-shell stable branch identity does not replay")
        if stable == bool(ambiguities):
            raise ValueError("device-shell ambiguity/stability state is invalid")
        if self.stable_branch_identity_hash is not None:
            object.__setattr__(
                self,
                "stable_branch_identity_hash",
                _identity(
                    self.stable_branch_identity_hash,
                    name="stable branch identity hash",
                ),
            )
        flags = (
            self.source_derived,
            self.fixed_z_face_perturbation_only,
            self.full_kkt_solution_sensitivity_theorem,
            self.C_stream_open_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("device-shell derivation flags must be exact booleans")
        if (
            self.source_derived is not True
            or self.fixed_z_face_perturbation_only
            is not source.fixed_z_face_perturbation_only
            or self.full_kkt_solution_sensitivity_theorem
            is not source.full_kkt_solution_sensitivity_theorem
            or self.fixed_z_face_perturbation_only
            == self.full_kkt_solution_sensitivity_theorem
            or any(
                (
                    self.C_stream_open_authority,
                    self.calibration_authority,
                    self.oracle_authority,
                    self.mask_commit_authority,
                )
            )
        ):
            raise PermissionError("device-shell derivation crossed its pre-C boundary")
        if self.schema != FQR_ADAPTIVE_RANK_DEVICE_SHELL_SOURCE_DERIVATION_SCHEMA:
            raise ValueError("device-shell source-derivation schema is invalid")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "admissible_action_hashes", admissible)
        object.__setattr__(self, "safe_action_hashes", safe)
        object.__setattr__(self, "ambiguity_codes", ambiguities)
        self.assert_integrity()

    @property
    def branch(self) -> V1RelaxedKKTBranch:
        return self.source_binding.branch_binding.branch

    @property
    def rank_local_stable(self) -> bool:
        return self.stable_branch_identity_hash is not None

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "policy_hash": self.policy_hash,
            "rank": self.rank,
            "graph_hash": self.graph_hash,
            "branch_family_hash": self.branch_family_hash,
            "branch": self.branch,
            "source_binding_hash": self.source_binding.binding_hash,
            "endpoint_status_axis_hash": self.endpoint_status_axis_hash,
            "admissible_action_hashes": self.admissible_action_hashes,
            "safe_action_hashes": self.safe_action_hashes,
            "ambiguity_codes": self.ambiguity_codes,
            "stable_branch_identity_hash": self.stable_branch_identity_hash,
            "source_derived": self.source_derived,
            "fixed_z_face_perturbation_only": (
                self.fixed_z_face_perturbation_only
            ),
            "full_kkt_solution_sensitivity_theorem": (
                self.full_kkt_solution_sensitivity_theorem
            ),
            "C_stream_open_authority": self.C_stream_open_authority,
            "calibration_authority": self.calibration_authority,
            "oracle_authority": self.oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(
            self._payload(), prefix=_DEVICE_SHELL_DERIVATION_PREFIX
        )
        cached = self._receipt_hash
        if cached is not None and cached != observed:
            raise ValueError("device-shell source derivation was tampered")
        if expected_hash is not None and expected_hash != observed:
            raise ValueError("device-shell source derivation hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1FQRAdaptiveRankDeviceShellDerivedDecision:
    policy_hash: str
    rank: int
    graph_hash: str
    branch_derivations: tuple[
        V1FQRAdaptiveRankDeviceShellBranchSourceDerivation, ...
    ]
    decision_certificate: V1FQRAdaptiveRankDecisionCertificate
    schema: str = FQR_ADAPTIVE_RANK_DEVICE_SHELL_EXECUTION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_hash",
            _identity(self.policy_hash, name="policy hash"),
        )
        object.__setattr__(
            self,
            "graph_hash",
            _identity(self.graph_hash, name="graph hash"),
        )
        rank = _positive_int(self.rank, name="FD rank")
        derivations = tuple(self.branch_derivations)
        if (
            len(derivations) != 2
            or any(
                type(value)
                is not V1FQRAdaptiveRankDeviceShellBranchSourceDerivation
                for value in derivations
            )
            or tuple(value.branch for value in derivations) != _BRANCH_AXIS
        ):
            raise ValueError("device-shell decision requires the two source branches")
        decision = self.decision_certificate
        if type(decision) is not V1FQRAdaptiveRankDecisionCertificate:
            raise TypeError("device-shell decision requires an exact certificate")
        decision.assert_integrity()
        if (
            decision.policy_hash != self.policy_hash
            or decision.rank != rank
            or any(
                value.policy_hash != self.policy_hash
                or value.rank != rank
                or value.graph_hash != self.graph_hash
                for value in derivations
            )
            or tuple(value.receipt_hash for value in derivations)
            != tuple(
                value.authoritative_typed_source_derivation_receipt_hash
                for value in decision.branch_decisions
            )
        ):
            raise ValueError("device-shell derived decision graph does not close")
        if self.schema != FQR_ADAPTIVE_RANK_DEVICE_SHELL_EXECUTION_SCHEMA:
            raise ValueError("device-shell derived-decision schema is invalid")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "branch_derivations", derivations)

    @property
    def result_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "policy_hash": self.policy_hash,
                "rank": self.rank,
                "graph_hash": self.graph_hash,
                "branch_derivation_hashes": tuple(
                    value.receipt_hash for value in self.branch_derivations
                ),
                "decision_certificate_hash": (
                    self.decision_certificate.certificate_hash
                ),
            },
            prefix=_DEVICE_SHELL_EXECUTION_PREFIX,
        )


def _branch_result(
    graph: V1SampledKKTBranchFamilyGraph,
    branch: V1RelaxedKKTBranch,
) -> V1RelaxedKKTBranchResult:
    return graph.result.spectral if branch is V1RelaxedKKTBranch.SPECTRAL else graph.result.diagonal_l1


def _branch_binding(
    graph: V1SampledKKTBranchFamilyGraph,
    branch: V1RelaxedKKTBranch,
) -> V1FQRKKTBranchFrozenBinding:
    binding = (
        graph.branch_family.spectral
        if branch is V1RelaxedKKTBranch.SPECTRAL
        else graph.branch_family.diagonal_l1
    )
    if type(binding) is not V1FQRKKTBranchFrozenBinding:
        raise ValueError("adaptive rank requires both complete branch ladders")
    return binding


def _branch_witness(
    graph: V1SampledKKTBranchFamilyGraph,
    branch: V1RelaxedKKTBranch,
) -> V1RelaxedKKTAdaptiveRankBranchWitness | None:
    witnesses = tuple(graph.adaptive_rank_witnesses)
    if not witnesses:
        return None
    matches = tuple(value for value in witnesses if value.branch is branch)
    if len(matches) != 1:
        raise ValueError("adaptive-rank witness axis is not branch-unique")
    return matches[0]


def _legality_for_prefix(problem: V1RelaxedKKTProblem, indices: np.ndarray) -> bool:
    if indices.ndim != 1 or indices.dtype.kind not in {"i", "u"}:
        return False
    if indices.size and (
        int(np.min(indices)) < 0
        or int(np.max(indices)) >= problem.route_count
        or len(set(int(value) for value in indices)) != int(indices.size)
        or bool(np.any(~problem.eligible_mask[indices]))
    ):
        return False
    return all(
        int(np.count_nonzero((indices >= block.start) & (indices < block.stop)))
        <= block.removal_capacity
        for block in problem.block_capacities
    )


def _complete_hard_fq_replay(
    *,
    graph: V1SampledKKTBranchFamilyGraph,
    branch: V1RelaxedKKTBranch,
    fd_receipt: V1StratifiedFDAssemblyReceipt,
    numerical_enclosure_policy_hash: str,
    chunk_size: int,
) -> V1FQRAdaptiveRankCompleteHardFQReplay:
    problem = graph.problem
    branch_result = _branch_result(graph, branch)
    binding = _branch_binding(graph, branch)
    rounded = branch_result.rounded_ladder
    candidates = rounded.candidates
    if (
        rounded.every_registered_candidate_replayed is not True
        or rounded.nonmonotone_q_early_stop_used is not False
        or len(candidates) != len(binding.endpoint_packed_action_hashes)
    ):
        raise ValueError("adaptive rank requires complete nonmonotone-Q hard replay")
    route_order = np.asarray(rounded.ordered_route_indices, dtype=np.int64)
    factor = np.asarray(problem.low_rank_factor, dtype=np.float64)
    work = np.asarray(problem.corrected_work, dtype=np.float64)
    resource = np.asarray(problem.resource, dtype=np.float64)
    diagonal = np.asarray(problem.residual_diagonal, dtype=np.float64)
    rank_rows = int(factor.shape[0])
    running_factor_center = np.zeros(rank_rows, dtype=np.float64)
    running_factor_abs_upper = np.zeros(rank_rows, dtype=np.float64)
    running_work_center = 0.0
    running_work_abs_upper = 0.0
    running_resource_center = 0.0
    running_resource_abs_upper = 0.0
    running_sqrt_diagonal_upper = 0.0
    previous = 0
    endpoints: list[V1FQRAdaptiveRankHardEndpointInterval] = []
    for endpoint_index, candidate in enumerate(candidates, start=1):
        stop = candidate.prefix_count
        if stop <= previous or stop > int(route_order.size):
            raise ValueError("hard ladder prefix counts are not complete and nested")
        for start in range(previous, stop, chunk_size):
            shell = route_order[start : min(stop, start + chunk_size)]
            if shell.size == 0:
                continue
            if rank_rows:
                factor_shell = factor[:, shell]
                shell_center = np.sum(factor_shell, axis=1, dtype=np.float64)
                running_factor_center = running_factor_center + shell_center
                for row in range(rank_rows):
                    running_factor_abs_upper[row] = _up_add(
                        float(running_factor_abs_upper[row]),
                        _nonnegative_sum_upper(
                            np.asarray(np.abs(factor_shell[row]), dtype=np.float64)
                        ),
                    )
            work_shell = work[shell]
            resource_shell = resource[shell]
            running_work_center += float(np.sum(work_shell, dtype=np.float64))
            running_resource_center += float(np.sum(resource_shell, dtype=np.float64))
            running_work_abs_upper = _up_add(
                running_work_abs_upper,
                _nonnegative_sum_upper(work_shell),
            )
            running_resource_abs_upper = _up_add(
                running_resource_abs_upper,
                _nonnegative_sum_upper(resource_shell),
            )
            sqrt_diagonal = np.sqrt(diagonal[shell], dtype=np.float64)
            if sqrt_diagonal.size:
                np.nextafter(sqrt_diagonal, np.inf, out=sqrt_diagonal)
                running_sqrt_diagonal_upper = _up_add(
                    running_sqrt_diagonal_upper,
                    _nonnegative_sum_upper(sqrt_diagonal),
                )
        term_count = stop
        work_lower, work_upper = _signed_sum_interval_from_nominal(
            nominal=running_work_center,
            absolute_sum_upper=running_work_abs_upper,
            term_count=term_count,
        )
        resource_lower, resource_upper = _signed_sum_interval_from_nominal(
            nominal=running_resource_center,
            absolute_sum_upper=running_resource_abs_upper,
            term_count=term_count,
        )
        row_lowers: list[float] = []
        row_uppers: list[float] = []
        for row in range(rank_rows):
            lower, upper = _signed_sum_interval_from_nominal(
                nominal=float(running_factor_center[row]),
                absolute_sum_upper=float(running_factor_abs_upper[row]),
                term_count=term_count,
            )
            square_lower, square_upper = _square_interval(lower, upper)
            row_lowers.append(square_lower)
            row_uppers.append(square_upper)
        low_q_lower, low_q_upper = _nonnegative_interval_sum(row_lowers, row_uppers)
        spectral_residual = _up_multiply(
            problem.residual_spectral_bound,
            float(stop),
        )
        diagonal_residual = _up_square(running_sqrt_diagonal_upper)
        canonical_residual = min(spectral_residual, diagonal_residual)
        q_upper = _up_add(low_q_upper, canonical_residual)
        legal = _legality_for_prefix(problem, route_order[:stop])
        equality = any(
            left == right
            for left, right in (
                (work_upper, problem.work_budget),
                (work_lower, problem.work_budget),
                (q_upper, problem.quadratic_budget),
                (low_q_lower, problem.quadratic_budget),
            )
        )
        safe = (
            legal
            and work_upper < problem.work_budget
            and q_upper < problem.quadratic_budget
        )
        impossible = (
            not legal
            or work_lower > problem.work_budget
            or low_q_lower > problem.quadratic_budget
        )
        status = (
            V1FQRHardFQStatus.SAFE_FEASIBLE
            if safe
            else V1FQRHardFQStatus.ROBUST_IMPOSSIBLE
            if impossible
            else V1FQRHardFQStatus.FD_AMBIGUOUS
        )
        endpoint = V1FQRAdaptiveRankHardEndpointInterval(
            branch=branch,
            endpoint_index=endpoint_index,
            prefix_count=stop,
            packed_action_hash=binding.endpoint_packed_action_hashes[endpoint_index - 1],
            kkt_candidate_hash=candidate.candidate_hash,
            resource_lower=max(0.0, resource_lower),
            resource_upper=max(0.0, resource_upper),
            f_lower=max(0.0, work_lower),
            f_upper=max(0.0, work_upper),
            low_rank_q_lower=low_q_lower,
            low_rank_q_upper=low_q_upper,
            spectral_residual_q_upper=spectral_residual,
            diagonal_l1_residual_q_upper=diagonal_residual,
            canonical_residual_q_upper=canonical_residual,
            q_upper=q_upper,
            work_budget=problem.work_budget,
            quadratic_budget=problem.quadratic_budget,
            legal=legal,
            status=status,
            equality_boundary=equality,
        )
        # Bind the landed replay without trusting it as the interval proof.
        if (
            not (endpoint.resource_lower <= candidate.resource_value <= endpoint.resource_upper)
            or not (
                endpoint.f_lower
                <= candidate.corrected_work_value
                <= endpoint.f_upper
            )
            or not (
                endpoint.low_rank_q_lower
                <= candidate.low_rank_quadratic
                <= endpoint.low_rank_q_upper
            )
        ):
            raise ValueError("landed hard candidate lies outside source-derived enclosure")
        endpoints.append(endpoint)
        previous = stop
    receipt = branch_result.receipt
    return V1FQRAdaptiveRankCompleteHardFQReplay(
        branch=branch,
        problem_hash=problem.problem_hash,
        kkt_result_hash=graph.result.result_hash,
        kkt_branch_result_hash=branch_result.result_hash,
        kkt_branch_receipt_hash=receipt.receipt_hash,
        kkt_primal_receipt_hash=receipt.primal.receipt_hash,
        kkt_stationarity_receipt_hash=receipt.stationarity.receipt_hash,
        kkt_gap_receipt_hash=receipt.gap.receipt_hash,
        kkt_convergence_receipt_hash=receipt.convergence.receipt_hash,
        rounded_ladder_hash=rounded.ladder_hash,
        frozen_branch_binding_hash=binding.binding_hash,
        frozen_ladder_hash=binding.frozen_ladder.ladder_hash,
        fd_assembly_receipt_hash=fd_receipt.receipt_hash,
        fd_residual_diagonal_hash=fd_receipt.residual_diagonal_hash,
        legality_hash=graph.legality_hash,
        eligibility_hash=graph.eligibility_hash,
        route_axis_hash=problem.route_axis_hash,
        route_count=problem.route_count,
        current_mask_hash=graph.branch_family.current_mask_hash,
        chart_identity_hash=graph.chart_identity_hash,
        proposal_generation=graph.branch_family.proposal_generation,
        numerical_enclosure_policy_hash=numerical_enclosure_policy_hash,
        endpoint_intervals=tuple(endpoints),
    )


def derive_v1_fqr_adaptive_rank_decision(
    *,
    policy: V1FQRAdaptiveRankPolicy | V1FQREndogenousRankPolicyV2,
    rank: int,
    graph: V1SampledKKTBranchFamilyGraph,
    numerical_enclosure_policy_hash: str,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> V1FQRAdaptiveRankDerivedDecision:
    """Derive one rank-local decision solely from the closed source graph.

    Face and endpoint-set witnesses are intentionally not caller parameters.
    Until the KKT v2 object exists and is bound by the graph, those components
    remain typed ambiguity and the returned certificate has no C authority.

    ``policy`` is either the legacy pre-D adaptive-rank policy (rank must be
    on its memory-admitted ladder; certificate keyed by its hash) or the v2
    endogenous-rank policy (rank must be on its physically admitted dyadic
    lattice; certificate keyed by the v2 hash, which is the identity the v2
    source decision image and the device-native witness consume).  The
    derivation itself is identical for both; only the admission axis and the
    certificate's policy identity differ.
    """

    if type(policy) is V1FQRAdaptiveRankPolicy:
        admitted_ranks = policy.admitted_rank_ladder
    elif type(policy) is V1FQREndogenousRankPolicyV2:
        admitted_ranks = policy.admitted_rank_lattice
    else:
        raise TypeError("adaptive-rank derivation requires the exact policy")
    if type(graph) is not V1SampledKKTBranchFamilyGraph:
        raise TypeError("adaptive-rank derivation requires the exact sampled KKT graph")
    policy_hash = policy.assert_integrity()
    graph_hash = graph.assert_integrity()
    observed_rank = _positive_int(rank, name="FD rank")
    if observed_rank not in admitted_ranks:
        raise ValueError("derived FD rank is not memory-admitted")
    chunk = _positive_int(chunk_size, name="derivation chunk size")
    enclosure_policy = _identity(
        numerical_enclosure_policy_hash,
        name="numerical enclosure policy hash",
    )
    source = graph.sampled_geometry_source
    fd_receipt = source.geometry_result.receipt
    if type(fd_receipt) is not V1StratifiedFDAssemblyReceipt:
        raise TypeError("sampled graph lacks its exact FD assembly receipt")
    fd_receipt.assert_integrity()
    if (
        fd_receipt.maximum_rank != observed_rank
        or fd_receipt.route_count != policy.route_count
        or fd_receipt.route_count != graph.problem.route_count
        or fd_receipt.route_axis_hash
        != source.admission.fd_design.route_axis_hash
        or graph.problem.route_axis_hash != graph.branch_family.route_axis_hash
        or graph.problem.source_geometry_hash != graph.chart_identity_hash
        or graph.result.problem_hash != graph.problem.problem_hash
        or graph.result.policy_hash != graph.policy.policy_hash
        or graph.branch_family.problem_hash != graph.problem.problem_hash
        or graph.branch_family.result_hash != graph.result.result_hash
        or graph.branch_family.dominance_certificate is not None
        or graph.branch_family.spectral is None
        or graph.branch_family.diagonal_l1 is None
    ):
        raise ValueError("adaptive-rank source graph coordinates do not close")
    branch_derivations: list[V1FQRAdaptiveRankBranchSourceDerivation] = []
    branch_decisions: list[V1FQRAdaptiveRankBranchDecision] = []
    for branch in _BRANCH_AXIS:
        branch_result = _branch_result(graph, branch)
        binding = _branch_binding(graph, branch)
        witness = _branch_witness(graph, branch)
        hard = _complete_hard_fq_replay(
            graph=graph,
            branch=branch,
            fd_receipt=fd_receipt,
            numerical_enclosure_policy_hash=enclosure_policy,
            chunk_size=chunk,
        )
        ambiguities: list[V1FQRRankDerivationAmbiguityCode] = []
        branch_converged = branch_result.receipt.convergence.converged
        if not branch_converged:
            ambiguities.append(V1FQRRankDerivationAmbiguityCode.KKT_BRANCH_NOT_CONVERGED)
        if not hard.hard_fq_status_axis_stable:
            ambiguities.append(V1FQRRankDerivationAmbiguityCode.FD_HARD_FQ_BOUNDARY)
        if witness is None:
            # Scalar gap/stationarity summaries do not identify a nonunique
            # face or enclose the lexicographic shell boundaries.  Do not
            # manufacture a sensitivity formula from them.
            ambiguities.extend(
                (
                    V1FQRRankDerivationAmbiguityCode.RELAXED_FACE_WITNESS_MISSING,
                    V1FQRRankDerivationAmbiguityCode.ROUNDED_ENDPOINT_SET_WITNESS_MISSING,
                )
            )
            relaxed_face_stable = False
            rounded_endpoint_set_stable = False
            stable_branch_identity_hash = None
            relaxed_face_witness_hash = None
            rounded_endpoint_set_witness_hash = None
            winning_face_margin = 0.0
            winning_face_perturbation = fd_receipt.combined_residual_spectral_bound
            selected_margin = 0.0
            selected_perturbation = fd_receipt.combined_residual_spectral_bound
        else:
            relaxed_face_stable = witness.relaxed_face_stable and branch_converged
            rounded_endpoint_set_stable = (
                witness.rounded_endpoint_set_stable and branch_converged
            )
            if branch_converged and not relaxed_face_stable:
                ambiguities.append(
                    V1FQRRankDerivationAmbiguityCode.RELAXED_FACE_UNRESOLVED
                )
            if branch_converged and not rounded_endpoint_set_stable:
                ambiguities.append(
                    V1FQRRankDerivationAmbiguityCode.ROUNDED_ENDPOINT_SET_UNRESOLVED
                )
            relaxed_face_witness_hash = (
                witness.relaxed_face_witness_hash if relaxed_face_stable else None
            )
            rounded_endpoint_set_witness_hash = (
                witness.rounded_endpoint_set_witness_hash
                if rounded_endpoint_set_stable
                else None
            )
            winning_face_margin = witness.winning_face_margin_lower_bound
            winning_face_perturbation = (
                witness.winning_face_residual_perturbation_upper_bound
                + witness.finite_population_face_perturbation_upper_bound
            )
            selected_margin = min(
                witness.selected_vs_runner_up_margin_lower_bound,
                witness.endpoint_boundary_margin_lower_bound,
            )
            selected_perturbation = (
                max(
                    witness.selected_vs_runner_up_residual_perturbation_upper_bound,
                    witness.endpoint_boundary_residual_perturbation_upper_bound,
                )
                + witness.finite_population_key_perturbation_upper_bound
            )
            stable_branch_identity_hash = (
                content_hash(
                    {
                        "schema": (
                            "partition_free_v1_fqr_adaptive_rank_stable_"
                            "branch_identity_v2"
                        ),
                        "branch": branch.value,
                        "kkt_witness_hash": witness.witness_hash,
                        "hard_fq_replay_hash": hard.receipt_hash,
                        "hard_ladder_hash": binding.frozen_ladder.ladder_hash,
                        "endpoint_status_axis_hash": hard.endpoint_status_axis_hash,
                        "admissible_action_hashes": hard.admissible_action_hashes,
                        "safe_action_hashes": hard.safe_action_hashes,
                        "source_pre_c_branch_family_hash": (
                            graph.branch_family.family_hash
                        ),
                    },
                    prefix=_STABLE_BRANCH_PREFIX,
                )
                if (
                    relaxed_face_stable
                    and rounded_endpoint_set_stable
                    and hard.hard_fq_status_axis_stable
                    and branch_result.receipt.convergence.converged
                )
                else None
            )
        derivation = V1FQRAdaptiveRankBranchSourceDerivation(
            policy_hash=policy_hash,
            rank=observed_rank,
            branch=branch,
            graph_hash=graph_hash,
            sampled_geometry_source_hash=source.source_hash,
            fd_assembly_receipt_hash=fd_receipt.receipt_hash,
            problem_hash=graph.problem.problem_hash,
            kkt_result_hash=graph.result.result_hash,
            branch_family_hash=graph.branch_family.family_hash,
            hard_fq_replay=hard,
            relaxed_face_witness_hash=relaxed_face_witness_hash,
            rounded_endpoint_set_witness_hash=rounded_endpoint_set_witness_hash,
            relaxed_face_stable=relaxed_face_stable,
            rounded_endpoint_set_stable=rounded_endpoint_set_stable,
            ambiguity_codes=tuple(dict.fromkeys(ambiguities)),
            stable_branch_identity_hash=stable_branch_identity_hash,
        )
        branch_derivations.append(derivation)
        admissible = hard.admissible_action_hashes
        selected = admissible[-1] if admissible else None
        # This scalar remains diagnostic only.  In particular an ambiguous
        # interval has no positive robust margin merely because both of its
        # endpoints happen to be far from a budget in absolute value.
        label_margins: list[float] = []
        for value in hard.endpoint_intervals:
            if value.status is V1FQRHardFQStatus.SAFE_FEASIBLE:
                label_margins.append(
                    min(
                        value.work_budget - value.f_upper,
                        value.quadratic_budget - value.q_upper,
                    )
                )
            elif value.status is V1FQRHardFQStatus.ROBUST_IMPOSSIBLE:
                label_margins.append(
                    max(
                        value.f_lower - value.work_budget,
                        value.low_rank_q_lower - value.quadratic_budget,
                        0.0,
                    )
                )
            else:
                label_margins.append(0.0)
        stable_margin = min(label_margins, default=0.0)
        branch_decisions.append(
            V1FQRAdaptiveRankBranchDecision(
                branch=branch.value,
                ordered_packed_action_hashes=tuple(
                    value.packed_action_hash for value in hard.endpoint_intervals
                ),
                admissible_packed_action_hashes=admissible,
                selected_packed_action_hash=selected,
                hard_ladder_hash=binding.frozen_ladder.ladder_hash,
                feasibility_receipt_hash=hard.receipt_hash,
                decision_receipt_hash=derivation.receipt_hash,
                fd_residual_envelope_receipt_hash=fd_receipt.receipt_hash,
                fd_residual_spectral_upper_bound=(
                    fd_receipt.combined_residual_spectral_bound
                ),
                fd_residual_diagonal_upper_bounds_hash=(
                    fd_receipt.residual_diagonal_hash
                ),
                winning_face_margin_lower_bound=winning_face_margin,
                winning_face_residual_perturbation_upper_bound=(
                    winning_face_perturbation
                ),
                hard_fq_feasibility_margin_lower_bound=stable_margin,
                hard_fq_residual_perturbation_upper_bound=0.0,
                selected_vs_runner_up_margin_lower_bound=selected_margin,
                selected_vs_runner_up_residual_perturbation_upper_bound=(
                    selected_perturbation
                ),
                next_larger_infeasible_margin_lower_bound=selected_margin,
                next_larger_residual_perturbation_upper_bound=selected_perturbation,
                selected_competitor_margin_receipt_hash=hard.receipt_hash,
                next_larger_feasibility_boundary_receipt_hash=hard.receipt_hash,
                authoritative_typed_source_derivation_available=True,
                authoritative_typed_source_derivation_receipt_hash=(
                    derivation.receipt_hash
                ),
                relaxed_face_source_stable=relaxed_face_stable,
                rounded_endpoint_set_source_stable=rounded_endpoint_set_stable,
                hard_fq_status_axis_source_stable=(
                    hard.hard_fq_status_axis_stable
                ),
                stable_branch_identity_hash=stable_branch_identity_hash,
                endpoint_status_axis_hash=hard.endpoint_status_axis_hash,
                source_derivation_ambiguity_codes=tuple(
                    value.value for value in derivation.ambiguity_codes
                ),
                _source_derivation_issuance_token=(
                    _SOURCE_DERIVATION_ISSUANCE_TOKEN
                ),
                schema=FQR_ADAPTIVE_RANK_BRANCH_DECISION_V2_SCHEMA,
            )
        )
    rank_stable = all(
        value.rank_local_residual_decision_stable for value in branch_decisions
    )
    robust_family = (
        content_hash(
            {
                "schema": (
                    "partition_free_v1_fqr_adaptive_rank_robust_"
                    "two_branch_family_v2"
                ),
                "branch_axis": tuple(value.branch for value in branch_decisions),
                "stable_branch_identity_hashes": tuple(
                    value.stable_branch_identity_hash for value in branch_decisions
                ),
                "ordered_action_axes": tuple(
                    value.ordered_packed_action_hashes for value in branch_decisions
                ),
                "endpoint_status_axis_hashes": tuple(
                    value.endpoint_status_axis_hash for value in branch_decisions
                ),
                "source_derivation_receipt_hashes": tuple(
                    value.authoritative_typed_source_derivation_receipt_hash
                    for value in branch_decisions
                ),
                "source_pre_c_branch_family_hash": graph.branch_family.family_hash,
            },
            prefix=_DECISION_FAMILY_PREFIX,
        )
        if rank_stable
        else None
    )
    certificate = V1FQRAdaptiveRankDecisionCertificate(
        policy_hash=policy_hash,
        rank=observed_rank,
        current_mask_hash=graph.branch_family.current_mask_hash,
        route_axis_hash=graph.problem.route_axis_hash,
        route_count=graph.problem.route_count,
        proposal_generation=graph.branch_family.proposal_generation,
        branch_decisions=tuple(branch_decisions),
        rank_local_residual_decision_stable=rank_stable,
        robust_next_decision_hash=robust_family,
        C_stream_open_authority=rank_stable,
        robust_branch_family_hash=robust_family,
        source_pre_c_branch_family_hash=graph.branch_family.family_hash,
        schema=FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA,
    )
    return V1FQRAdaptiveRankDerivedDecision(
        policy_hash=policy_hash,
        rank=observed_rank,
        graph_hash=graph_hash,
        branch_derivations=tuple(branch_derivations),
        decision_certificate=certificate,
    )


def derive_v1_fqr_device_shell_adaptive_rank_decision(
    *,
    policy: V1FQRAdaptiveRankPolicy,
    rank: int,
    graph: object,
) -> V1FQRAdaptiveRankDeviceShellDerivedDecision:
    """Derive one rank decision from exact shell witnesses and CUDA replay.

    The local import avoids making the device-shell bridge and this source
    derivation module cyclic owners.  Exact type equality remains mandatory.
    """

    from .modern_v1_sampled_kkt_device_shell_bridge import (
        V1SampledKKTDeviceShellGraph,
    )

    if type(policy) is not V1FQRAdaptiveRankPolicy:
        raise TypeError("device-shell adaptive rank requires the exact policy")
    if type(graph) is not V1SampledKKTDeviceShellGraph:
        raise TypeError("device-shell adaptive rank requires the exact source graph")
    policy_hash = policy.assert_integrity()
    graph_hash = graph.assert_integrity()
    observed_rank = _positive_int(rank, name="FD rank")
    if observed_rank not in policy.admitted_rank_ladder:
        raise ValueError("device-shell rank is not memory-admitted")
    if (
        graph.route_count != policy.route_count
        or not graph.source_built_face_shell_witness_available
        or not graph.cuda_device_native_fq_replay_available
        or graph.registered_d_source_coordinate is None
        or graph.registered_d_source_coordinate.fd_sketch_rank != observed_rank
        or graph.spectral_source_stability is None
        or graph.diagonal_l1_source_stability is None
        or graph.spectral_source_stability.cuda_hard_fq_replay.rank
        != graph.diagonal_l1_source_stability.cuda_hard_fq_replay.rank
    ):
        raise ValueError("device-shell graph lacks its rank-local source closure")

    derivations: list[V1FQRAdaptiveRankDeviceShellBranchSourceDerivation] = []
    decisions: list[V1FQRAdaptiveRankBranchDecision] = []
    for source in (
        graph.spectral_source_stability,
        graph.diagonal_l1_source_stability,
    ):
        assert source is not None
        source.assert_integrity()
        witness = source.adaptive_rank_witness
        replay = source.cuda_hard_fq_replay
        ambiguities: list[V1FQRRankDerivationAmbiguityCode] = []
        if not source.routewise_kkt_face_stable:
            ambiguities.append(
                V1FQRRankDerivationAmbiguityCode.RELAXED_FACE_UNRESOLVED
            )
        if not source.packed_endpoint_shell_stable:
            ambiguities.append(
                V1FQRRankDerivationAmbiguityCode.ROUNDED_ENDPOINT_SET_UNRESOLVED
            )
        if not source.complete_cuda_hard_fq_replay:
            ambiguities.append(
                V1FQRRankDerivationAmbiguityCode.FD_HARD_FQ_BOUNDARY
            )
        if (
            not source.hard_fq_status_axis_stable
            and V1FQRRankDerivationAmbiguityCode.FD_HARD_FQ_BOUNDARY
            not in ambiguities
        ):
            ambiguities.append(
                V1FQRRankDerivationAmbiguityCode.FD_HARD_FQ_BOUNDARY
            )
        if not source.decision_relevant_perturbation_closed:
            ambiguities.append(
                V1FQRRankDerivationAmbiguityCode.RELAXED_PROBLEM_SENSITIVITY_UNCERTIFIED
            )
        status_axis_hash = _device_shell_endpoint_status_axis_hash(source)
        stable_branch_hash = (
            content_hash(
                {
                    "schema": (
                        "partition_free_v1_fqr_adaptive_rank_device_shell_"
                        "stable_branch_identity_v1"
                    ),
                    "branch": source.branch_binding.branch,
                    "source_stability_binding_hash": source.binding_hash,
                    "endpoint_status_axis_hash": status_axis_hash,
                    "admissible_action_hashes": replay.admissible_action_hashes,
                    "safe_action_hashes": replay.safe_action_hashes,
                    "source_pre_c_branch_family_hash": graph.branch_family.family_hash,
                },
                prefix=_STABLE_BRANCH_PREFIX,
            )
            if source.source_stable
            else None
        )
        derivation = V1FQRAdaptiveRankDeviceShellBranchSourceDerivation(
            policy_hash=policy_hash,
            rank=observed_rank,
            graph_hash=graph_hash,
            branch_family_hash=graph.branch_family.family_hash,
            source_binding=source,
            endpoint_status_axis_hash=status_axis_hash,
            admissible_action_hashes=tuple(replay.admissible_action_hashes),
            safe_action_hashes=tuple(replay.safe_action_hashes),
            ambiguity_codes=tuple(ambiguities),
            stable_branch_identity_hash=stable_branch_hash,
            fixed_z_face_perturbation_only=(
                source.fixed_z_face_perturbation_only
            ),
            full_kkt_solution_sensitivity_theorem=(
                source.full_kkt_solution_sensitivity_theorem
            ),
        )
        derivations.append(derivation)
        label_margins: list[float] = []
        for endpoint in replay.endpoints:
            if endpoint.status is V1DeviceShellHardFQStatus.SAFE_FEASIBLE:
                label_margins.append(
                    min(
                        endpoint.work_budget - endpoint.f_upper,
                        endpoint.quadratic_budget - endpoint.q_upper,
                    )
                )
            elif endpoint.status is V1DeviceShellHardFQStatus.ROBUST_IMPOSSIBLE:
                label_margins.append(
                    max(
                        endpoint.f_lower - endpoint.work_budget,
                        endpoint.low_rank_q_lower - endpoint.quadratic_budget,
                        0.0,
                    )
                )
            else:
                label_margins.append(0.0)
        admissible = tuple(replay.admissible_action_hashes)
        deterministic_face = (
            witness.winning_face_residual_perturbation_upper_bound
            + witness.finite_population_face_perturbation_upper_bound
        )
        deterministic_key = (
            max(
                witness.selected_vs_runner_up_residual_perturbation_upper_bound,
                witness.endpoint_boundary_residual_perturbation_upper_bound,
            )
            + witness.finite_population_key_perturbation_upper_bound
        )
        decisions.append(
            V1FQRAdaptiveRankBranchDecision(
                branch=source.branch_binding.branch.value,
                ordered_packed_action_hashes=tuple(replay.packed_action_hashes),
                admissible_packed_action_hashes=admissible,
                selected_packed_action_hash=(
                    admissible[-1] if admissible else None
                ),
                hard_ladder_hash=source.branch_binding.source_ladder_receipt_hash,
                feasibility_receipt_hash=replay.receipt_hash,
                decision_receipt_hash=derivation.receipt_hash,
                fd_residual_envelope_receipt_hash=(
                    witness.finite_population_geometry_uncertainty_set_hash
                ),
                fd_residual_spectral_upper_bound=max(
                    deterministic_face,
                    deterministic_key,
                ),
                fd_residual_diagonal_upper_bounds_hash=content_hash(
                    tuple(
                        (
                            endpoint.endpoint_label,
                            endpoint.spectral_residual_q_upper,
                            endpoint.diagonal_l1_residual_q_upper,
                        )
                        for endpoint in replay.endpoints
                    ),
                    prefix="partition-free-v1-fqr-device-shell-residual-axis-",
                ),
                winning_face_margin_lower_bound=(
                    witness.winning_face_margin_lower_bound
                ),
                winning_face_residual_perturbation_upper_bound=(
                    deterministic_face
                ),
                hard_fq_feasibility_margin_lower_bound=min(
                    label_margins,
                    default=0.0,
                ),
                hard_fq_residual_perturbation_upper_bound=0.0,
                selected_vs_runner_up_margin_lower_bound=min(
                    witness.selected_vs_runner_up_margin_lower_bound,
                    witness.endpoint_boundary_margin_lower_bound,
                ),
                selected_vs_runner_up_residual_perturbation_upper_bound=(
                    deterministic_key
                ),
                next_larger_infeasible_margin_lower_bound=min(
                    witness.selected_vs_runner_up_margin_lower_bound,
                    witness.endpoint_boundary_margin_lower_bound,
                ),
                next_larger_residual_perturbation_upper_bound=deterministic_key,
                selected_competitor_margin_receipt_hash=derivation.receipt_hash,
                next_larger_feasibility_boundary_receipt_hash=replay.receipt_hash,
                authoritative_typed_source_derivation_available=True,
                authoritative_typed_source_derivation_receipt_hash=(
                    derivation.receipt_hash
                ),
                relaxed_face_source_stable=(
                    source.decision_relevant_perturbation_closed
                    and source.routewise_kkt_face_stable
                ),
                rounded_endpoint_set_source_stable=(
                    source.decision_relevant_perturbation_closed
                    and source.packed_endpoint_shell_stable
                ),
                hard_fq_status_axis_source_stable=(
                    source.hard_fq_status_axis_stable
                ),
                stable_branch_identity_hash=stable_branch_hash,
                endpoint_status_axis_hash=status_axis_hash,
                source_derivation_ambiguity_codes=tuple(
                    value.value for value in derivation.ambiguity_codes
                ),
                _source_derivation_issuance_token=(
                    _SOURCE_DERIVATION_ISSUANCE_TOKEN
                ),
                schema=FQR_ADAPTIVE_RANK_BRANCH_DECISION_V2_SCHEMA,
            )
        )
    rank_stable = all(
        value.rank_local_residual_decision_stable for value in decisions
    )
    robust_family = (
        content_hash(
            {
                "schema": (
                    "partition_free_v1_fqr_adaptive_rank_robust_"
                    "two_branch_family_v2"
                ),
                "branch_axis": tuple(value.branch for value in decisions),
                "stable_branch_identity_hashes": tuple(
                    value.stable_branch_identity_hash for value in decisions
                ),
                "ordered_action_axes": tuple(
                    value.ordered_packed_action_hashes for value in decisions
                ),
                "endpoint_status_axis_hashes": tuple(
                    value.endpoint_status_axis_hash for value in decisions
                ),
                "source_derivation_receipt_hashes": tuple(
                    value.authoritative_typed_source_derivation_receipt_hash
                    for value in decisions
                ),
                "source_pre_c_branch_family_hash": graph.branch_family.family_hash,
            },
            prefix=_DECISION_FAMILY_PREFIX,
        )
        if rank_stable
        else None
    )
    certificate = V1FQRAdaptiveRankDecisionCertificate(
        policy_hash=policy_hash,
        rank=observed_rank,
        current_mask_hash=graph.current_mask_hash,
        route_axis_hash=graph.route_axis_hash,
        route_count=graph.route_count,
        proposal_generation=graph.proposal_generation,
        branch_decisions=tuple(decisions),
        rank_local_residual_decision_stable=rank_stable,
        robust_next_decision_hash=robust_family,
        C_stream_open_authority=rank_stable,
        robust_branch_family_hash=robust_family,
        source_pre_c_branch_family_hash=graph.branch_family.family_hash,
        schema=FQR_ADAPTIVE_RANK_DECISION_CERTIFICATE_V2_SCHEMA,
    )
    return V1FQRAdaptiveRankDeviceShellDerivedDecision(
        policy_hash=policy_hash,
        rank=observed_rank,
        graph_hash=graph_hash,
        branch_derivations=tuple(derivations),
        decision_certificate=certificate,
    )


RankReplayExecutor = Callable[[int], V1FQRAdaptiveRankReplayRecord]


def execute_v1_fqr_adaptive_rank_loop(
    *,
    policy: V1FQRAdaptiveRankPolicy,
    execute_rank: RankReplayExecutor,
) -> tuple[V1FQRAdaptiveRankReplayLedger, V1FQRAdaptiveRankStabilityCertificate]:
    """Execute only the ambiguity-required contiguous rank prefix.

    ``execute_rank`` owns physical raw-D/KKT execution.  Every returned record
    is independently checked by the replay ledger for identical sealed D
    coordinates, no snapshot padding, no D extension, and no C outcome.  A
    typed promotion refusal advances exactly one admitted rank; every other
    refusal propagates.  Maximum-rank ambiguity therefore remains no-C.
    """

    if type(policy) is not V1FQRAdaptiveRankPolicy:
        raise TypeError("adaptive-rank loop requires the exact policy")
    if not callable(execute_rank):
        raise TypeError("adaptive-rank loop requires a rank executor")
    records: list[V1FQRAdaptiveRankReplayRecord] = []
    for rank in policy.admitted_rank_ladder:
        record = execute_rank(rank)
        if type(record) is not V1FQRAdaptiveRankReplayRecord:
            raise TypeError("adaptive-rank executor returned another record type")
        if record.rank != rank:
            raise ValueError("adaptive-rank executor returned another rank")
        records.append(record)
        ledger = V1FQRAdaptiveRankReplayLedger(
            policy=policy,
            records=tuple(records),
        )
        try:
            certificate = certify_v1_fqr_adaptive_rank_stability(
                policy=policy,
                ledger=ledger,
            )
        except V1FQRAdaptiveRankRefusal as error:
            if error.receipt.code is V1FQRAdaptiveRankRefusalCode.RANK_PROMOTION_REQUIRED:
                continue
            raise
        return ledger, certificate
    raise AssertionError("maximum-rank ambiguity should have emitted a typed refusal")


__all__ = [
    "FQR_ADAPTIVE_RANK_DEVICE_SHELL_EXECUTION_SCHEMA",
    "FQR_ADAPTIVE_RANK_DEVICE_SHELL_SOURCE_DERIVATION_SCHEMA",
    "FQR_ADAPTIVE_RANK_EXECUTION_SCHEMA",
    "FQR_ADAPTIVE_RANK_HARD_ENDPOINT_SCHEMA",
    "FQR_ADAPTIVE_RANK_HARD_REPLAY_SCHEMA",
    "FQR_ADAPTIVE_RANK_OUTWARD_ARITHMETIC_ID",
    "FQR_ADAPTIVE_RANK_SOURCE_DERIVATION_SCHEMA",
    "V1FQRAdaptiveRankBranchSourceDerivation",
    "V1FQRAdaptiveRankCompleteHardFQReplay",
    "V1FQRAdaptiveRankDerivedDecision",
    "V1FQRAdaptiveRankDeviceShellBranchSourceDerivation",
    "V1FQRAdaptiveRankDeviceShellDerivedDecision",
    "V1FQRAdaptiveRankHardEndpointInterval",
    "V1FQRHardFQStatus",
    "V1FQRRankDerivationAmbiguityCode",
    "derive_v1_fqr_adaptive_rank_decision",
    "derive_v1_fqr_device_shell_adaptive_rank_decision",
    "execute_v1_fqr_adaptive_rank_loop",
]
