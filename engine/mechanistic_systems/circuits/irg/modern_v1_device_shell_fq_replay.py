"""Exact small-domain hard-F/Q replay from device-shell membership labels.

The production KKT path can retain one ``uint8`` first-inclusion label per
physical route instead of returning an ordered route axis or materializing an
``endpoint x route`` mask matrix.  This module defines the mathematical
reference joint for replaying every frozen endpoint directly from those
labels.

The implementation is deliberately limited to an immutable NumPy label
handle on a registered small domain.  A CUDA handle fails closed: the future
production kernel must use a deterministic segmented reduction without
``scatter_add``, ``bincount``, or floating-point atomics.  A CPU reference is
not a production fallback.

The resulting receipts certify only the registered hard F/Q replay and its
exact frozen action identities.  They grant no C-stream, nonlinear-remainder,
calibration, oracle, mask-commit, production, or Pareto authority.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Final, Literal, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.content_address import canonical_json, content_hash

from .modern_v1_device_shell_ladder import (
    DEVICE_SHELL_ELIGIBILITY_COMMITMENT_DOMAIN,
    DEVICE_SHELL_LABEL_COMMITMENT_DOMAIN,
    V1DeviceShellBorrowOperation,
    V1DeviceShellFrozenLadderReceipt,
    V1DeviceShellLabelHandle,
)
from .modern_v1_relaxed_kkt import (
    RELAXED_KKT_CANONICAL_HARD_Q_ID,
    V1RelaxedKKTBranch,
    V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    V1RelaxedKKTProblem,
)


DEVICE_SHELL_HARD_FQ_REPLAY_DESIGN_SCHEMA: Final = (
    "partition_free_v1_device_shell_hard_fq_replay_design_v1"
)
DEVICE_SHELL_HARD_FQ_ENDPOINT_SCHEMA: Final = (
    "partition_free_v1_device_shell_hard_fq_endpoint_v1"
)
DEVICE_SHELL_HARD_FQ_REPLAY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_device_shell_hard_fq_replay_receipt_v1"
)
DEVICE_SHELL_HARD_FQ_OUTWARD_ARITHMETIC_ID: Final = (
    "binary64_nextafter_gamma_n_absolute_sum_hard_fq_interval_v1"
)
DEVICE_SHELL_HARD_FQ_MEMBERSHIP_ALGORITHM_ID: Final = (
    "numpy_physical_axis_first_inclusion_membership_complete_replay_v1"
)
DEVICE_SHELL_HARD_FQ_DEFAULT_REFERENCE_ROUTE_LIMIT: Final = 65_536

_DESIGN_PREFIX: Final = "partition-free-v1-device-shell-hard-fq-design-"
_ENDPOINT_PREFIX: Final = "partition-free-v1-device-shell-hard-fq-endpoint-"
_REPLAY_PREFIX: Final = "partition-free-v1-device-shell-hard-fq-replay-"
_STATUS_AXIS_PREFIX: Final = (
    "partition-free-v1-device-shell-hard-fq-status-axis-"
)
_UNIT_ROUNDOFF: Final = float(np.finfo(np.float64).eps) / 2.0
_MIN_NORMAL: Final = float(np.finfo(np.float64).tiny)

DeviceShellFQSourceKind = Literal["branch_identity", "frozen_ladder"]


class V1DeviceShellHardFQStatus(str, Enum):
    """Strict hard-F/Q classification shared with adaptive-rank replay."""

    SAFE_FEASIBLE = "safe_feasible"
    ROBUST_IMPOSSIBLE = "robust_impossible"
    FD_AMBIGUOUS = "fd_ambiguous"


class V1DeviceShellHardFQRefusalCode(str, Enum):
    """Typed failures that cannot become hard-F/Q replay authority."""

    NUMPY_REFERENCE_SCALE_EXCEEDED = "numpy_reference_scale_exceeded"
    CUDA_DEVICE_NATIVE_KERNEL_UNAVAILABLE = (
        "cuda_device_native_kernel_unavailable"
    )
    PROBLEM_IDENTITY_MISMATCH = "problem_identity_mismatch"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    LABEL_IDENTITY_MISMATCH = "label_identity_mismatch"
    ELIGIBILITY_IDENTITY_MISMATCH = "eligibility_identity_mismatch"
    BLOCK_IDENTITY_MISMATCH = "block_identity_mismatch"
    MALFORMED_REFERENCE_LABELS = "malformed_reference_labels"


class V1DeviceShellHardFQRefusal(RuntimeError):
    """The label-native hard-F/Q reference joint refused its input."""

    def __init__(
        self,
        code: V1DeviceShellHardFQRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1DeviceShellHardFQRefusalCode, message: str) -> NoReturn:
    raise V1DeviceShellHardFQRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an exact integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an exact integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
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
    count = _nonnegative_int(
        operation_count,
        name="rounding operation count",
    )
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
    nonzero = int(
        np.count_nonzero(
            array.view(np.uint64) & np.uint64(0x7FFF_FFFF_FFFF_FFFF)
        )
    )
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


def _new_stream_digest(*, domain: str, metadata: object) -> "hashlib._Hash":
    encoded_domain = domain.encode("ascii")
    encoded_metadata = canonical_json(metadata).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(encoded_domain).to_bytes(8, "big", signed=False))
    digest.update(encoded_domain)
    digest.update(len(encoded_metadata).to_bytes(8, "big", signed=False))
    digest.update(encoded_metadata)
    return digest


def _label_commitment(
    labels: np.ndarray,
    *,
    ladder: V1DeviceShellFrozenLadderReceipt,
) -> str:
    digest = _new_stream_digest(
        domain=DEVICE_SHELL_LABEL_COMMITMENT_DOMAIN,
        metadata={
            "route_axis_hash": ladder.route_axis_hash,
            "route_count": ladder.route_count,
            "endpoint_count": ladder.endpoint_count,
        },
    )
    digest.update(memoryview(labels).cast("B"))
    return "partition-free-v1-device-shell-labels-" + digest.hexdigest()


def _eligibility_commitment(problem: V1RelaxedKKTProblem) -> str:
    digest = _new_stream_digest(
        domain=DEVICE_SHELL_ELIGIBILITY_COMMITMENT_DOMAIN,
        metadata={
            "route_axis_hash": problem.route_axis_hash,
            "route_count": problem.route_count,
        },
    )
    packed = np.packbits(problem.eligible_mask, bitorder="big")
    digest.update(memoryview(packed).cast("B"))
    return "partition-free-v1-device-shell-eligibility-" + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class V1DeviceShellHardFQReplayDesign:
    """Hashed design for the bounded, exact NumPy reference implementation."""

    branch: V1RelaxedKKTBranch
    maximum_numpy_reference_routes: int = (
        DEVICE_SHELL_HARD_FQ_DEFAULT_REFERENCE_ROUTE_LIMIT
    )
    design_id: str = "modern-v1-device-shell-hard-fq-reference-v1"
    backend: Literal["numpy_reference"] = "numpy_reference"
    membership_algorithm_id: str = DEVICE_SHELL_HARD_FQ_MEMBERSHIP_ALGORITHM_ID
    outward_arithmetic_id: str = DEVICE_SHELL_HARD_FQ_OUTWARD_ARITHMETIC_ID
    replay_every_registered_endpoint: bool = True
    allow_nonmonotone_q_early_stop: bool = False
    production_cuda_enabled: bool = False
    schema: str = DEVICE_SHELL_HARD_FQ_REPLAY_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("hard-F/Q replay design requires an exact KKT branch")
        object.__setattr__(
            self,
            "maximum_numpy_reference_routes",
            _positive_int(
                self.maximum_numpy_reference_routes,
                name="maximum NumPy reference routes",
            ),
        )
        object.__setattr__(self, "design_id", _identity(self.design_id, name="design ID"))
        if self.backend != "numpy_reference":
            raise ValueError("the landed hard-F/Q replay joint is NumPy-reference only")
        if self.membership_algorithm_id != DEVICE_SHELL_HARD_FQ_MEMBERSHIP_ALGORITHM_ID:
            raise ValueError("unsupported device-shell membership algorithm")
        if self.outward_arithmetic_id != DEVICE_SHELL_HARD_FQ_OUTWARD_ARITHMETIC_ID:
            raise ValueError("unsupported hard-F/Q outward arithmetic")
        flags = (
            self.replay_every_registered_endpoint,
            self.allow_nonmonotone_q_early_stop,
            self.production_cuda_enabled,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("hard-F/Q replay design flags must be exact booleans")
        if flags != (True, False, False):
            raise PermissionError(
                "the reference design must replay all endpoints without CUDA authority"
            )
        if self.schema != DEVICE_SHELL_HARD_FQ_REPLAY_DESIGN_SCHEMA:
            raise ValueError("unsupported device-shell hard-F/Q design schema")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_DESIGN_PREFIX)


@dataclass(frozen=True, slots=True)
class V1DeviceShellHardFQEndpointReceipt:
    """Outward hard-F/Q interval and exact frozen identity for one endpoint."""

    branch: V1RelaxedKKTBranch
    endpoint_label: int
    cumulative_removed_count: int
    frozen_action_receipt_hash: str
    packed_action_hash: str
    removed_route_index_hash: str
    resulting_mask_hash: str
    resulting_mask_packed_sha256: str
    block_prefix_counts: tuple[int, ...]
    resource_nominal: float
    resource_lower: float
    resource_upper: float
    f_nominal: float
    f_lower: float
    f_upper: float
    low_rank_q_nominal: float
    low_rank_q_lower: float
    low_rank_q_upper: float
    spectral_residual_q_upper: float
    diagonal_l1_residual_q_upper: float
    canonical_residual_q_upper: float
    q_upper: float
    work_budget: float
    quadratic_budget: float
    legal: bool
    status: V1DeviceShellHardFQStatus
    equality_boundary: bool
    canonical_hard_q_id: str = RELAXED_KKT_CANONICAL_HARD_Q_ID
    outward_arithmetic_id: str = DEVICE_SHELL_HARD_FQ_OUTWARD_ARITHMETIC_ID
    schema: str = DEVICE_SHELL_HARD_FQ_ENDPOINT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("hard-F/Q endpoint requires an exact KKT branch")
        object.__setattr__(
            self,
            "endpoint_label",
            _positive_int(self.endpoint_label, name="endpoint label"),
        )
        object.__setattr__(
            self,
            "cumulative_removed_count",
            _positive_int(
                self.cumulative_removed_count,
                name="cumulative removed count",
            ),
        )
        for name in (
            "frozen_action_receipt_hash",
            "packed_action_hash",
            "removed_route_index_hash",
            "resulting_mask_hash",
            "resulting_mask_packed_sha256",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        block_counts = tuple(
            _nonnegative_int(value, name="block prefix count")
            for value in self.block_prefix_counts
        )
        if block_counts and sum(block_counts) != self.cumulative_removed_count:
            raise ValueError(
                "hard-F/Q block prefixes do not partition cumulative removals"
            )
        object.__setattr__(self, "block_prefix_counts", block_counts)
        packed_digest = self.resulting_mask_packed_sha256
        if len(packed_digest) != 64 or packed_digest != packed_digest.lower():
            raise ValueError("resulting mask digest must be lowercase SHA-256")
        try:
            int(packed_digest, 16)
        except ValueError as error:
            raise ValueError(
                "resulting mask digest must be lowercase SHA-256"
            ) from error
        for name in (
            "resource_nominal",
            "resource_lower",
            "resource_upper",
            "f_nominal",
            "f_lower",
            "f_upper",
            "low_rank_q_nominal",
            "low_rank_q_lower",
            "low_rank_q_upper",
            "spectral_residual_q_upper",
            "diagonal_l1_residual_q_upper",
            "canonical_residual_q_upper",
            "q_upper",
            "work_budget",
            "quadratic_budget",
        ):
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
            raise ValueError("hard-F/Q endpoint interval is reversed")
        if not (
            self.resource_lower <= self.resource_nominal <= self.resource_upper
            and self.f_lower <= self.f_nominal <= self.f_upper
            and self.low_rank_q_lower
            <= self.low_rank_q_nominal
            <= self.low_rank_q_upper
        ):
            raise ValueError("hard-F/Q endpoint nominal lies outside its enclosure")
        expected_residual = min(
            self.spectral_residual_q_upper,
            self.diagonal_l1_residual_q_upper,
        )
        expected_q = _up_add(self.low_rank_q_upper, expected_residual)
        if (
            self.canonical_residual_q_upper != expected_residual
            or self.q_upper != expected_q
        ):
            raise ValueError("hard-F/Q endpoint does not replay its Q envelope")
        if type(self.legal) is not bool or type(self.equality_boundary) is not bool:
            raise TypeError("hard-F/Q legality and boundary flags must be exact booleans")
        if type(self.status) is not V1DeviceShellHardFQStatus:
            raise TypeError("hard-F/Q endpoint status is not registered")
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
            V1DeviceShellHardFQStatus.SAFE_FEASIBLE
            if safe
            else V1DeviceShellHardFQStatus.ROBUST_IMPOSSIBLE
            if impossible
            else V1DeviceShellHardFQStatus.FD_AMBIGUOUS
        )
        if self.status is not expected_status or self.equality_boundary is not equality:
            raise ValueError("hard-F/Q endpoint status does not replay its intervals")
        if self.canonical_hard_q_id != RELAXED_KKT_CANONICAL_HARD_Q_ID:
            raise ValueError("unsupported canonical hard-Q equation")
        if self.outward_arithmetic_id != DEVICE_SHELL_HARD_FQ_OUTWARD_ARITHMETIC_ID:
            raise ValueError("hard-F/Q endpoint arithmetic owner changed")
        if self.schema != DEVICE_SHELL_HARD_FQ_ENDPOINT_SCHEMA:
            raise ValueError("unsupported device-shell hard-F/Q endpoint schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_ENDPOINT_PREFIX)


@dataclass(frozen=True, slots=True)
class V1DeviceShellHardFQReplayReceipt:
    """Array-free receipt for a complete label-native hard-F/Q replay."""

    branch: V1RelaxedKKTBranch
    design_hash: str
    problem_hash: str
    problem_id: str
    source_geometry_hash: str
    source_identity_kind: DeviceShellFQSourceKind
    source_identity_hash: str
    source_branch_identity_receipt_hash: str | None
    frozen_ladder_receipt_hash: str
    source_rounding_hash: str
    route_axis_hash: str
    parent_mask_hash: str
    route_count: int
    rank: int
    endpoint_count: int
    label_commitment: str
    eligibility_commitment: str
    block_capacity_hash: str
    block_legality_scope: Literal[
        "registered_compact_capacities",
        "explicit_unconstrained_reference",
    ]
    frozen_action_receipt_hashes: tuple[str, ...]
    endpoints: tuple[V1DeviceShellHardFQEndpointReceipt, ...]
    membership_algorithm_id: str = DEVICE_SHELL_HARD_FQ_MEMBERSHIP_ALGORITHM_ID
    outward_arithmetic_id: str = DEVICE_SHELL_HARD_FQ_OUTWARD_ARITHMETIC_ID
    complete_registered_ladder_replayed: bool = True
    nonmonotone_q_early_stop_used: bool = False
    label_membership_used_directly: bool = True
    ordinary_route_order_materialized: bool = False
    ordinary_removed_indices_materialized: bool = False
    ordinary_hard_masks_materialized: bool = False
    cuda_device_native_complete: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    production_authority: bool = False
    pareto_authority: bool = False
    schema: str = DEVICE_SHELL_HARD_FQ_REPLAY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("hard-F/Q replay receipt requires an exact KKT branch")
        for name in (
            "design_hash",
            "problem_hash",
            "problem_id",
            "source_geometry_hash",
            "source_identity_hash",
            "frozen_ladder_receipt_hash",
            "source_rounding_hash",
            "route_axis_hash",
            "parent_mask_hash",
            "label_commitment",
            "eligibility_commitment",
            "block_capacity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.source_identity_kind not in {"branch_identity", "frozen_ladder"}:
            raise ValueError("unsupported device-shell F/Q source identity kind")
        if self.source_identity_kind == "branch_identity":
            branch_hash = _identity(
                self.source_branch_identity_receipt_hash,
                name="source branch identity receipt hash",
            )
            if branch_hash != self.source_identity_hash:
                raise ValueError("source branch identity hashes differ")
        elif self.source_branch_identity_receipt_hash is not None:
            raise ValueError("standalone ladder replay cannot name a branch receipt")
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        object.__setattr__(self, "rank", _nonnegative_int(self.rank, name="rank"))
        object.__setattr__(
            self,
            "endpoint_count",
            _positive_int(self.endpoint_count, name="endpoint count"),
        )
        if self.block_legality_scope not in {
            "registered_compact_capacities",
            "explicit_unconstrained_reference",
        }:
            raise ValueError("unsupported hard-F/Q block-legality scope")
        action_receipt_hashes = tuple(
            _identity(value, name="frozen action receipt hash")
            for value in self.frozen_action_receipt_hashes
        )
        object.__setattr__(
            self,
            "frozen_action_receipt_hashes",
            action_receipt_hashes,
        )
        endpoints = tuple(self.endpoints)
        if (
            not endpoints
            or len(endpoints) != self.endpoint_count
            or any(type(value) is not V1DeviceShellHardFQEndpointReceipt for value in endpoints)
            or tuple(value.endpoint_label for value in endpoints)
            != tuple(range(1, len(endpoints) + 1))
            or any(value.branch is not self.branch for value in endpoints)
            or tuple(value.cumulative_removed_count for value in endpoints)
            != tuple(sorted(set(value.cumulative_removed_count for value in endpoints)))
        ):
            raise ValueError("hard-F/Q replay endpoint axis is incomplete or nonnested")
        if action_receipt_hashes != tuple(
            value.frozen_action_receipt_hash for value in endpoints
        ):
            raise ValueError("hard-F/Q replay action-receipt axis differs")
        object.__setattr__(self, "endpoints", endpoints)
        if self.membership_algorithm_id != DEVICE_SHELL_HARD_FQ_MEMBERSHIP_ALGORITHM_ID:
            raise ValueError("hard-F/Q replay membership algorithm changed")
        if self.outward_arithmetic_id != DEVICE_SHELL_HARD_FQ_OUTWARD_ARITHMETIC_ID:
            raise ValueError("hard-F/Q replay arithmetic owner changed")
        flags = (
            self.complete_registered_ladder_replayed,
            self.nonmonotone_q_early_stop_used,
            self.label_membership_used_directly,
            self.ordinary_route_order_materialized,
            self.ordinary_removed_indices_materialized,
            self.ordinary_hard_masks_materialized,
            self.cuda_device_native_complete,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.production_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("hard-F/Q replay flags must be exact booleans")
        if flags != (
            True,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ):
            raise PermissionError("hard-F/Q replay crossed its reference authority boundary")
        if self.schema != DEVICE_SHELL_HARD_FQ_REPLAY_RECEIPT_SCHEMA:
            raise ValueError("unsupported device-shell hard-F/Q replay schema")

    @property
    def endpoint_status_axis_hash(self) -> str:
        return content_hash(
            tuple(
                (
                    value.endpoint_label,
                    value.packed_action_hash,
                    value.resulting_mask_hash,
                    value.status,
                    value.receipt_hash,
                )
                for value in self.endpoints
            ),
            prefix=_STATUS_AXIS_PREFIX,
        )

    @property
    def admissible_action_hashes(self) -> tuple[str, ...]:
        return tuple(
            value.packed_action_hash
            for value in self.endpoints
            if value.status is not V1DeviceShellHardFQStatus.ROBUST_IMPOSSIBLE
        )

    @property
    def safe_action_hashes(self) -> tuple[str, ...]:
        return tuple(
            value.packed_action_hash
            for value in self.endpoints
            if value.status is V1DeviceShellHardFQStatus.SAFE_FEASIBLE
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_REPLAY_PREFIX)


def _normalize_source_identity(
    *,
    source_identity: (
        V1RelaxedKKTDeviceShellBranchIdentityReceipt
        | V1DeviceShellFrozenLadderReceipt
    ),
    design: V1DeviceShellHardFQReplayDesign,
) -> tuple[
    DeviceShellFQSourceKind,
    str,
    str | None,
    V1DeviceShellFrozenLadderReceipt,
]:
    if type(source_identity) is V1RelaxedKKTDeviceShellBranchIdentityReceipt:
        if source_identity.branch is not design.branch:
            _refuse(
                V1DeviceShellHardFQRefusalCode.SOURCE_IDENTITY_MISMATCH,
                "device-shell branch receipt differs from the replay design",
            )
        return (
            "branch_identity",
            source_identity.receipt_hash,
            source_identity.receipt_hash,
            source_identity.frozen_ladder,
        )
    if type(source_identity) is V1DeviceShellFrozenLadderReceipt:
        return (
            "frozen_ladder",
            source_identity.receipt_hash,
            None,
            source_identity,
        )
    raise TypeError(
        "hard-F/Q replay requires an exact device-shell branch or ladder receipt"
    )


def _verify_problem_ladder_identity(
    *,
    problem: V1RelaxedKKTProblem,
    ladder: V1DeviceShellFrozenLadderReceipt,
) -> None:
    replayed_problem_hash = content_hash(
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
    if replayed_problem_hash != problem.problem_hash:
        _refuse(
            V1DeviceShellHardFQRefusalCode.PROBLEM_IDENTITY_MISMATCH,
            "relaxed KKT arrays or coordinates changed after problem registration",
        )
    if (
        problem.route_count != ladder.route_count
        or problem.route_axis_hash != ladder.route_axis_hash
    ):
        _refuse(
            V1DeviceShellHardFQRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "KKT problem and shell ladder differ on the physical route axis",
        )
    problem_blocks = tuple(
        (block.block_id, block.start, block.stop, block.removal_capacity)
        for block in problem.block_capacities
    )
    ladder_blocks = tuple(
        (block.block_id, block.start, block.stop, block.removal_capacity)
        for block in ladder.block_capacities
    )
    if problem_blocks != ladder_blocks:
        _refuse(
            V1DeviceShellHardFQRefusalCode.BLOCK_IDENTITY_MISMATCH,
            "KKT problem blocks differ from the frozen shell legality receipt",
        )
    if _eligibility_commitment(problem) != ladder.eligibility_commitment:
        _refuse(
            V1DeviceShellHardFQRefusalCode.ELIGIBILITY_IDENTITY_MISMATCH,
            "KKT eligibility differs from the frozen shell receipt",
        )


def _sum_interval(values: np.ndarray) -> tuple[float, float, float]:
    nominal = float(np.sum(values, dtype=np.float64))
    absolute_upper = _nonnegative_sum_upper(values)
    lower, upper = _signed_sum_interval_from_nominal(
        nominal=nominal,
        absolute_sum_upper=absolute_upper,
        term_count=int(values.size),
    )
    return nominal, max(0.0, lower), max(0.0, upper)


def _replay_numpy_labels(
    *,
    problem: V1RelaxedKKTProblem,
    ladder: V1DeviceShellFrozenLadderReceipt,
    labels: np.ndarray,
    branch: V1RelaxedKKTBranch,
) -> tuple[V1DeviceShellHardFQEndpointReceipt, ...]:
    endpoints: list[V1DeviceShellHardFQEndpointReceipt] = []
    factor = problem.low_rank_factor
    for endpoint_label, action in enumerate(ladder.actions, start=1):
        membership = (labels != 0) & (labels <= endpoint_label)
        cumulative_count = int(np.count_nonzero(membership))
        shell_count = int(np.count_nonzero(labels == endpoint_label))
        if (
            cumulative_count != action.cumulative_removed_count
            or shell_count != action.shell_count
        ):
            _refuse(
                V1DeviceShellHardFQRefusalCode.LABEL_IDENTITY_MISMATCH,
                "label membership histogram differs from the frozen action receipt",
            )
        if bool(np.any(membership & ~problem.eligible_mask)):
            _refuse(
                V1DeviceShellHardFQRefusalCode.ELIGIBILITY_IDENTITY_MISMATCH,
                "a label endpoint includes a route ineligible in the KKT problem",
            )

        resource_nominal, resource_lower, resource_upper = _sum_interval(
            problem.resource[membership]
        )
        f_nominal, f_lower, f_upper = _sum_interval(
            problem.corrected_work[membership]
        )

        row_lowers: list[float] = []
        row_nominals: list[float] = []
        row_uppers: list[float] = []
        for row in range(problem.rank):
            row_values = factor[row, membership]
            nominal = float(np.sum(row_values, dtype=np.float64))
            absolute_upper = _nonnegative_sum_upper(np.abs(row_values))
            lower, upper = _signed_sum_interval_from_nominal(
                nominal=nominal,
                absolute_sum_upper=absolute_upper,
                term_count=cumulative_count,
            )
            square_lower, square_upper = _square_interval(lower, upper)
            row_nominals.append(nominal)
            row_lowers.append(square_lower)
            row_uppers.append(square_upper)
        nominal_response = np.asarray(row_nominals, dtype=np.float64)
        low_rank_q_nominal = float(np.dot(nominal_response, nominal_response))
        low_rank_q_lower, low_rank_q_upper = _nonnegative_interval_sum(
            row_lowers,
            row_uppers,
        )

        spectral_residual = _up_multiply(
            problem.residual_spectral_bound,
            float(cumulative_count),
        )
        sqrt_diagonal = np.sqrt(
            problem.residual_diagonal[membership],
            dtype=np.float64,
        )
        if sqrt_diagonal.size:
            np.nextafter(sqrt_diagonal, np.inf, out=sqrt_diagonal)
        diagonal_sum_upper = _nonnegative_sum_upper(sqrt_diagonal)
        diagonal_residual = _up_square(diagonal_sum_upper)
        canonical_residual = min(spectral_residual, diagonal_residual)
        q_upper = _up_add(low_rank_q_upper, canonical_residual)

        block_prefix_counts = tuple(
            row[endpoint_label - 1] for row in ladder.block_prefix_counts
        )
        legal = all(
            count <= block.removal_capacity
            for count, block in zip(
                block_prefix_counts,
                ladder.block_capacities,
                strict=True,
            )
        )
        equality = any(
            left == right
            for left, right in (
                (f_upper, problem.work_budget),
                (f_lower, problem.work_budget),
                (q_upper, problem.quadratic_budget),
                (low_rank_q_lower, problem.quadratic_budget),
            )
        )
        safe = (
            legal
            and f_upper < problem.work_budget
            and q_upper < problem.quadratic_budget
        )
        impossible = (
            not legal
            or f_lower > problem.work_budget
            or low_rank_q_lower > problem.quadratic_budget
        )
        status = (
            V1DeviceShellHardFQStatus.SAFE_FEASIBLE
            if safe
            else V1DeviceShellHardFQStatus.ROBUST_IMPOSSIBLE
            if impossible
            else V1DeviceShellHardFQStatus.FD_AMBIGUOUS
        )
        endpoints.append(
            V1DeviceShellHardFQEndpointReceipt(
                branch=branch,
                endpoint_label=endpoint_label,
                cumulative_removed_count=cumulative_count,
                frozen_action_receipt_hash=action.receipt_hash,
                packed_action_hash=action.packed_action_hash,
                removed_route_index_hash=action.removed_route_index_hash,
                resulting_mask_hash=action.resulting_mask_hash,
                resulting_mask_packed_sha256=(
                    action.resulting_mask_packed_sha256
                ),
                block_prefix_counts=block_prefix_counts,
                resource_nominal=resource_nominal,
                resource_lower=resource_lower,
                resource_upper=resource_upper,
                f_nominal=f_nominal,
                f_lower=f_lower,
                f_upper=f_upper,
                low_rank_q_nominal=low_rank_q_nominal,
                low_rank_q_lower=low_rank_q_lower,
                low_rank_q_upper=low_rank_q_upper,
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
        )
    return tuple(endpoints)


def replay_v1_device_shell_hard_fq_reference(
    *,
    problem: V1RelaxedKKTProblem,
    source_identity: (
        V1RelaxedKKTDeviceShellBranchIdentityReceipt
        | V1DeviceShellFrozenLadderReceipt
    ),
    handle: V1DeviceShellLabelHandle,
    design: V1DeviceShellHardFQReplayDesign,
) -> V1DeviceShellHardFQReplayReceipt:
    """Replay every frozen endpoint directly from resident first-inclusion labels.

    The function intentionally has no CUDA fallback and emits no ordinary
    route order, removed-index set, packed mask, or hard action.  The exact
    identities for those objects are copied from the already-frozen ladder
    receipt after the resident label and eligibility commitments replay.
    """

    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("hard-F/Q replay requires an exact relaxed KKT problem")
    if type(handle) is not V1DeviceShellLabelHandle:
        raise TypeError("hard-F/Q replay requires an exact shell-label handle")
    if type(design) is not V1DeviceShellHardFQReplayDesign:
        raise TypeError("hard-F/Q replay requires an exact reference design")
    (
        source_kind,
        source_hash,
        branch_receipt_hash,
        ladder,
    ) = _normalize_source_identity(
        source_identity=source_identity,
        design=design,
    )
    _verify_problem_ladder_identity(problem=problem, ladder=ladder)

    operation = V1DeviceShellBorrowOperation.FQ_HARD_REPLAY
    with handle.borrow_for_registered_operation(operation=operation) as lease:
        if lease.backend_kind != "numpy_reference":
            _refuse(
                V1DeviceShellHardFQRefusalCode.CUDA_DEVICE_NATIVE_KERNEL_UNAVAILABLE,
                "CUDA hard-F/Q replay awaits a qualified deterministic segmented "
                "kernel; CPU transfer or floating atomics are forbidden",
            )
        if problem.route_count > design.maximum_numpy_reference_routes:
            _refuse(
                V1DeviceShellHardFQRefusalCode.NUMPY_REFERENCE_SCALE_EXCEEDED,
                "route axis exceeds the registered small-domain NumPy reference",
            )
        if (
            lease.route_count != problem.route_count
            or ladder.handle_kind != lease.backend_kind
            or ladder.device_identity != lease.device_identity
        ):
            _refuse(
                V1DeviceShellHardFQRefusalCode.SOURCE_IDENTITY_MISMATCH,
                "shell handle differs from the KKT problem or frozen ladder receipt",
            )
        resident = lease._labels_for_registered_operation(operation=operation)
        if type(resident) is not np.ndarray:
            _refuse(
                V1DeviceShellHardFQRefusalCode.MALFORMED_REFERENCE_LABELS,
                "NumPy reference labels must be an exact ndarray",
            )
        labels = np.asarray(resident)
        if (
            labels.ndim != 1
            or labels.shape != (problem.route_count,)
            or labels.dtype != np.dtype(np.uint8)
            or not labels.flags.c_contiguous
            or labels.flags.writeable
        ):
            _refuse(
                V1DeviceShellHardFQRefusalCode.MALFORMED_REFERENCE_LABELS,
                "NumPy reference labels must be immutable contiguous uint8",
            )
        if labels.size and int(np.max(labels)) > ladder.endpoint_count:
            _refuse(
                V1DeviceShellHardFQRefusalCode.MALFORMED_REFERENCE_LABELS,
                "a first-inclusion label lies outside the frozen endpoint axis",
            )
        if _label_commitment(labels, ladder=ladder) != ladder.label_commitment:
            _refuse(
                V1DeviceShellHardFQRefusalCode.LABEL_IDENTITY_MISMATCH,
                "resident first-inclusion labels differ from their frozen commitment",
            )
        endpoints = _replay_numpy_labels(
            problem=problem,
            ladder=ladder,
            labels=labels,
            branch=design.branch,
        )

    return V1DeviceShellHardFQReplayReceipt(
        branch=design.branch,
        design_hash=design.design_hash,
        problem_hash=problem.problem_hash,
        problem_id=problem.problem_id,
        source_geometry_hash=problem.source_geometry_hash,
        source_identity_kind=source_kind,
        source_identity_hash=source_hash,
        source_branch_identity_receipt_hash=branch_receipt_hash,
        frozen_ladder_receipt_hash=ladder.receipt_hash,
        source_rounding_hash=ladder.source_ladder_hash,
        route_axis_hash=problem.route_axis_hash,
        parent_mask_hash=ladder.parent_mask_hash,
        route_count=problem.route_count,
        rank=problem.rank,
        endpoint_count=ladder.endpoint_count,
        label_commitment=ladder.label_commitment,
        eligibility_commitment=ladder.eligibility_commitment,
        block_capacity_hash=ladder.block_capacity_hash,
        block_legality_scope=ladder.block_legality_scope,
        frozen_action_receipt_hashes=tuple(
            action.receipt_hash for action in ladder.actions
        ),
        endpoints=endpoints,
    )
