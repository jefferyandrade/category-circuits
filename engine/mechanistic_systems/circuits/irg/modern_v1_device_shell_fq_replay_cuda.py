"""Deterministic CUDA hard-F/Q replay for an exact device-shell execution.

This production-candidate sibling consumes the exact
``V1RelaxedKKTDeviceShellExecutionBundle`` that owns a frozen branch label
axis.  It does not accept a caller-spliced handle/receipt pair.  First-
inclusion labels are grouped on CUDA by a stable integer argsort; physical
route order is therefore the exact tie order within each shell.  At most one
route-scale int64 permutation is retained, and only compact
``O(endpoint_count * (rank + constant))`` FP64 prefix summaries cross D2H.

No label axis, route order, removed-index axis, hard mask, or hard action is
materialized on the host.  No ``scatter_add``, ``bincount``, or floating
atomic reduction is used.  The existing reference endpoint type remains the
owner of outward interval arithmetic and strict F/Q classifications.

The receipts are authority-inert beyond this hard-F/Q measurement.  In
particular they do not authorize face stability, C, R, calibration, oracle,
commit, production, terminal, or Pareto claims.  A local primitive probe is
not full-axis H200 repeat/parity qualification; that gate remains explicitly
open.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

import math
import time
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any, Final, Literal, NoReturn

import numpy as np

from .contracts import content_hash
from .modern_v1_device_shell_fq_replay import (
    DEVICE_SHELL_HARD_FQ_OUTWARD_ARITHMETIC_ID,
    V1DeviceShellHardFQEndpointReceipt,
    V1DeviceShellHardFQRefusal,
    V1DeviceShellHardFQStatus,
)
from . import modern_v1_device_shell_fq_replay as _reference
from .modern_v1_device_shell_ladder import (
    V1DeviceShellBorrowOperation,
    V1DeviceShellFrozenLadderReceipt,
    V1DeviceShellLabelHandle,
)
from .modern_v1_relaxed_kkt import (
    RELAXED_KKT_CANONICAL_HARD_Q_ID,
    RELAXED_KKT_FUSED_CUDA_ALGORITHM_ID,
    RELAXED_KKT_ROUNDING_KEY_ID,
    V1RelaxedKKTBranch,
    V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    V1RelaxedKKTDeviceShellExecutionBundle,
    V1RelaxedKKTProblem,
)


DEVICE_SHELL_HARD_FQ_CUDA_REPLAY_DESIGN_SCHEMA: Final = (
    "partition_free_v1_device_shell_hard_fq_cuda_replay_design_v1"
)
DEVICE_SHELL_HARD_FQ_CUDA_HBM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_device_shell_hard_fq_cuda_hbm_forecast_v1"
)
DEVICE_SHELL_HARD_FQ_CUDA_REPLAY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_device_shell_hard_fq_cuda_replay_receipt_v1"
)
DEVICE_SHELL_HARD_FQ_CUDA_TELEMETRY_SCHEMA: Final = (
    "partition_free_v1_device_shell_hard_fq_cuda_telemetry_v1"
)
DEVICE_SHELL_HARD_FQ_CUDA_BUNDLE_SCHEMA: Final = (
    "partition_free_v1_device_shell_hard_fq_cuda_bundle_v1"
)
DEVICE_SHELL_HARD_FQ_CUDA_ALGORITHM_ID: Final = (
    "torch_cuda_stable_int16_first_inclusion_argsort_physical_tie_"
    "streamed_fp64_cumsum_v1"
)
DEVICE_SHELL_HARD_FQ_CUDA_PRIMITIVE_PROBE_ID: Final = (
    "local_stable_argsort_and_bitwise_repeat_cumsum_probe_v1"
)
DEVICE_SHELL_HARD_FQ_CUDA_HBM_CEILING_FRACTION: Final = 0.85
DEVICE_SHELL_HARD_FQ_CUDA_ARGSORT_WORKSPACE_BYTES_PER_ROUTE: Final = 24
DEVICE_SHELL_HARD_FQ_CUDA_MINIMUM_ALLOCATOR_GUARD_BYTES: Final = 64 * 1024 * 1024
DEVICE_SHELL_HARD_FQ_CUDA_ALLOCATOR_GUARD_FRACTION: Final = 0.10

_DESIGN_PREFIX: Final = "partition-free-v1-device-shell-hard-fq-cuda-design-"
_FORECAST_PREFIX: Final = (
    "partition-free-v1-device-shell-hard-fq-cuda-hbm-forecast-"
)
_ACTION_AXIS_PREFIX: Final = (
    "partition-free-v1-device-shell-hard-fq-cuda-action-axis-"
)
_ENDPOINT_AXIS_PREFIX: Final = (
    "partition-free-v1-device-shell-hard-fq-cuda-endpoint-axis-"
)
_RECEIPT_PREFIX: Final = "partition-free-v1-device-shell-hard-fq-cuda-replay-"
_TELEMETRY_PREFIX: Final = (
    "partition-free-v1-device-shell-hard-fq-cuda-telemetry-"
)
_BUNDLE_PREFIX: Final = "partition-free-v1-device-shell-hard-fq-cuda-bundle-"


class V1DeviceShellHardFQCUDARefusalCode(str, Enum):
    """Typed failures that cannot become CUDA replay evidence."""

    CUDA_UNAVAILABLE = "cuda_unavailable"
    CPU_HANDLE_FORBIDDEN = "cpu_handle_forbidden"
    DETERMINISM_DISABLED = "determinism_disabled"
    CUDA_DETERMINISM_PRIMITIVE_UNQUALIFIED = (
        "cuda_determinism_primitive_unqualified"
    )
    EXECUTION_IDENTITY_MISMATCH = "execution_identity_mismatch"
    PROBLEM_IDENTITY_MISMATCH = "problem_identity_mismatch"
    BRANCH_IDENTITY_MISMATCH = "branch_identity_mismatch"
    MALFORMED_CUDA_LABELS = "malformed_cuda_labels"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    HBM_AVAILABLE_MEMORY_EXCEEDED = "hbm_available_memory_exceeded"
    MEASURED_HBM_CEILING_EXCEEDED = "measured_hbm_ceiling_exceeded"
    MEASURED_HBM_FORECAST_INVALIDATED = "measured_hbm_forecast_invalidated"
    CUDA_OUT_OF_MEMORY = "cuda_out_of_memory"
    LABEL_SUMMARY_MISMATCH = "label_summary_mismatch"
    NONFINITE_CUDA_SUMMARY = "nonfinite_cuda_summary"


class V1DeviceShellHardFQCUDARefusal(RuntimeError):
    """The CUDA path refused rather than weakening its contract."""

    def __init__(
        self,
        code: V1DeviceShellHardFQCUDARefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1DeviceShellHardFQCUDARefusalCode:
            raise TypeError("CUDA replay refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1DeviceShellHardFQCUDARefusalCode,
    message: str,
) -> NoReturn:
    raise V1DeviceShellHardFQCUDARefusal(code, message)


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


@dataclass(frozen=True, slots=True)
class V1DeviceShellHardFQCUDAReplayDesign:
    """Hashed, non-weakenable deterministic CUDA execution design."""

    design_id: str = "modern-v1-device-shell-hard-fq-cuda-v1"
    backend: Literal["torch_cuda"] = "torch_cuda"
    algorithm_id: str = DEVICE_SHELL_HARD_FQ_CUDA_ALGORITHM_ID
    primitive_probe_id: str = DEVICE_SHELL_HARD_FQ_CUDA_PRIMITIVE_PROBE_ID
    outward_arithmetic_id: str = DEVICE_SHELL_HARD_FQ_OUTWARD_ARITHMETIC_ID
    hbm_ceiling_fraction: float = (
        DEVICE_SHELL_HARD_FQ_CUDA_HBM_CEILING_FRACTION
    )
    require_torch_deterministic_algorithms: bool = True
    require_stable_integer_argsort: bool = True
    require_bitwise_repeat_cumsum_probe: bool = True
    replay_every_registered_endpoint: bool = True
    allow_nonmonotone_q_early_stop: bool = False
    allow_cpu_fallback: bool = False
    schema: str = DEVICE_SHELL_HARD_FQ_CUDA_REPLAY_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "design_id",
            _identity(self.design_id, name="design ID"),
        )
        if self.backend != "torch_cuda":
            raise ValueError("CUDA hard-F/Q replay has no alternate backend")
        if self.algorithm_id != DEVICE_SHELL_HARD_FQ_CUDA_ALGORITHM_ID:
            raise ValueError("unsupported CUDA hard-F/Q algorithm")
        if self.primitive_probe_id != DEVICE_SHELL_HARD_FQ_CUDA_PRIMITIVE_PROBE_ID:
            raise ValueError("unsupported CUDA primitive probe")
        if self.outward_arithmetic_id != DEVICE_SHELL_HARD_FQ_OUTWARD_ARITHMETIC_ID:
            raise ValueError("CUDA replay cannot change endpoint arithmetic")
        ceiling = _finite_nonnegative(
            self.hbm_ceiling_fraction,
            name="HBM ceiling fraction",
        )
        if ceiling != DEVICE_SHELL_HARD_FQ_CUDA_HBM_CEILING_FRACTION:
            raise ValueError("CUDA replay HBM ceiling is frozen at 85 percent")
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        flags = (
            self.require_torch_deterministic_algorithms,
            self.require_stable_integer_argsort,
            self.require_bitwise_repeat_cumsum_probe,
            self.replay_every_registered_endpoint,
            self.allow_nonmonotone_q_early_stop,
            self.allow_cpu_fallback,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("CUDA replay design flags must be exact booleans")
        if flags != (True, True, True, True, False, False):
            raise PermissionError("CUDA replay design weakened a closed safety gate")
        if self.schema != DEVICE_SHELL_HARD_FQ_CUDA_REPLAY_DESIGN_SCHEMA:
            raise ValueError("unsupported CUDA replay design schema")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_DESIGN_PREFIX)


@dataclass(frozen=True, slots=True)
class V1DeviceShellHardFQCUDAHBMForecast:
    """Pre-outcome allocation forecast under the fixed 85% HBM ceiling."""

    design_hash: str
    execution_receipt_hash: str
    problem_hash: str
    policy_hash: str
    branch: V1RelaxedKKTBranch
    branch_identity_receipt_hash: str
    device: str
    route_count: int
    rank: int
    endpoint_count: int
    maximum_selected_count: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    process_allocated_before_bytes: int
    process_reserved_before_bytes: int
    hbm_ceiling_fraction: float
    hbm_ceiling_bytes: int
    resident_label_bytes: int
    grouping_key_bytes: int
    retained_permutation_bytes: int
    streamed_input_vector_bytes: int
    selected_pair_peak_bytes: int
    compact_summary_bytes: int
    compact_endpoint_index_bytes: int
    stable_argsort_workspace_upper_bytes: int
    allocator_guard_bytes: int
    estimated_incremental_peak_bytes: int
    estimated_process_peak_bytes: int
    admitted: bool
    schema: str = DEVICE_SHELL_HARD_FQ_CUDA_HBM_FORECAST_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "execution_receipt_hash",
            "problem_hash",
            "policy_hash",
            "branch_identity_receipt_hash",
            "device",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("CUDA HBM forecast branch must be exact")
        for name in ("route_count", "endpoint_count", "maximum_selected_count"):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        object.__setattr__(self, "rank", _nonnegative_int(self.rank, name="rank"))
        byte_fields = (
            "device_total_memory_bytes",
            "device_free_memory_bytes",
            "process_allocated_before_bytes",
            "process_reserved_before_bytes",
            "hbm_ceiling_bytes",
            "resident_label_bytes",
            "grouping_key_bytes",
            "retained_permutation_bytes",
            "streamed_input_vector_bytes",
            "selected_pair_peak_bytes",
            "compact_summary_bytes",
            "compact_endpoint_index_bytes",
            "stable_argsort_workspace_upper_bytes",
            "allocator_guard_bytes",
            "estimated_incremental_peak_bytes",
            "estimated_process_peak_bytes",
        )
        for name in byte_fields:
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        ceiling_fraction = _finite_nonnegative(
            self.hbm_ceiling_fraction,
            name="HBM ceiling fraction",
        )
        if ceiling_fraction != DEVICE_SHELL_HARD_FQ_CUDA_HBM_CEILING_FRACTION:
            raise ValueError("CUDA HBM forecast must use the fixed 85% ceiling")
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling_fraction)
        n = self.route_count
        endpoint_count = self.endpoint_count
        selected = self.maximum_selected_count
        if selected > n:
            raise ValueError("selected endpoint exceeds the route axis")
        expected_ceiling = math.floor(
            self.device_total_memory_bytes * ceiling_fraction
        )
        expected_key = 2 * n
        expected_permutation = 8 * n
        expected_input = 8 * n
        expected_selected = 16 * selected
        expected_summary = 8 * endpoint_count * (2 * self.rank + 3)
        expected_endpoint_index = 8 * endpoint_count
        expected_sort = (
            DEVICE_SHELL_HARD_FQ_CUDA_ARGSORT_WORKSPACE_BYTES_PER_ROUTE * n
        )
        subtotal = max(
            expected_key + expected_permutation + expected_sort,
            expected_permutation + expected_input + expected_selected,
        ) + expected_summary + expected_endpoint_index
        expected_guard = max(
            DEVICE_SHELL_HARD_FQ_CUDA_MINIMUM_ALLOCATOR_GUARD_BYTES,
            math.ceil(
                DEVICE_SHELL_HARD_FQ_CUDA_ALLOCATOR_GUARD_FRACTION * subtotal
            ),
        )
        expected_incremental = subtotal + expected_guard
        expected_process = self.process_reserved_before_bytes + expected_incremental
        expected_admission = (
            self.process_allocated_before_bytes + expected_incremental <= expected_ceiling
            and expected_incremental <= self.device_free_memory_bytes + max(0, self.process_reserved_before_bytes - self.process_allocated_before_bytes)
        )
        observed = (
            self.hbm_ceiling_bytes,
            self.resident_label_bytes,
            self.grouping_key_bytes,
            self.retained_permutation_bytes,
            self.streamed_input_vector_bytes,
            self.selected_pair_peak_bytes,
            self.compact_summary_bytes,
            self.compact_endpoint_index_bytes,
            self.stable_argsort_workspace_upper_bytes,
            self.allocator_guard_bytes,
            self.estimated_incremental_peak_bytes,
            self.estimated_process_peak_bytes,
            self.admitted,
        )
        expected = (
            expected_ceiling,
            n,
            expected_key,
            expected_permutation,
            expected_input,
            expected_selected,
            expected_summary,
            expected_endpoint_index,
            expected_sort,
            expected_guard,
            expected_incremental,
            expected_process,
            expected_admission,
        )
        if observed != expected:
            raise ValueError("CUDA HBM forecast does not replay its allocation model")
        if self.device_free_memory_bytes > self.device_total_memory_bytes:
            raise ValueError("free CUDA memory cannot exceed total memory")
        if self.process_allocated_before_bytes > self.process_reserved_before_bytes:
            raise ValueError("allocated CUDA memory cannot exceed reserved memory")
        if type(self.admitted) is not bool:
            raise TypeError("HBM admission must be an exact bool")
        if self.schema != DEVICE_SHELL_HARD_FQ_CUDA_HBM_FORECAST_SCHEMA:
            raise ValueError("unsupported CUDA HBM forecast schema")

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_FORECAST_PREFIX)


@dataclass(frozen=True, slots=True)
class V1DeviceShellHardFQCUDAReplayReceipt:
    """Array-free complete hard-F/Q receipt for one exact branch."""

    branch: V1RelaxedKKTBranch
    design_hash: str
    hbm_forecast_hash: str
    device_shell_execution_receipt_hash: str
    device_shell_execution_design_hash: str
    cuda_execution_bundle_hash: str
    cuda_execution_design_hash: str
    base_result_hash: str
    branch_result_hash: str
    branch_kkt_receipt_hash: str
    problem_hash: str
    problem_id: str
    source_geometry_hash: str
    policy_hash: str
    numerical_policy_hash: str
    source_branch_identity_receipt_hash: str
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
    rounding_key_id: str
    frozen_action_axis_hash: str
    frozen_action_receipt_hashes: tuple[str, ...]
    packed_action_hashes: tuple[str, ...]
    endpoint_axis_hash: str
    endpoints: tuple[V1DeviceShellHardFQEndpointReceipt, ...]
    algorithm_id: str = DEVICE_SHELL_HARD_FQ_CUDA_ALGORITHM_ID
    primitive_probe_id: str = DEVICE_SHELL_HARD_FQ_CUDA_PRIMITIVE_PROBE_ID
    canonical_hard_q_id: str = RELAXED_KKT_CANONICAL_HARD_Q_ID
    outward_arithmetic_id: str = DEVICE_SHELL_HARD_FQ_OUTWARD_ARITHMETIC_ID
    complete_registered_ladder_replayed: bool = True
    nonmonotone_q_early_stop_used: bool = False
    label_membership_used_directly: bool = True
    stable_physical_tie_order: bool = True
    full_axis_h200_repeat_parity_required: bool = True
    full_axis_h200_repeat_parity_qualified: bool = False
    ordinary_route_order_d2h: bool = False
    ordinary_removed_indices_materialized: bool = False
    ordinary_hard_masks_materialized: bool = False
    ordinary_hard_actions_materialized: bool = False
    face_stability_authority: bool = False
    c_stream_authority: bool = False
    nonlinear_remainder_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    production_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = DEVICE_SHELL_HARD_FQ_CUDA_REPLAY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("CUDA replay receipt branch must be exact")
        for name in (
            "design_hash",
            "hbm_forecast_hash",
            "device_shell_execution_receipt_hash",
            "device_shell_execution_design_hash",
            "cuda_execution_bundle_hash",
            "cuda_execution_design_hash",
            "base_result_hash",
            "branch_result_hash",
            "branch_kkt_receipt_hash",
            "problem_hash",
            "problem_id",
            "source_geometry_hash",
            "policy_hash",
            "numerical_policy_hash",
            "source_branch_identity_receipt_hash",
            "frozen_ladder_receipt_hash",
            "source_rounding_hash",
            "route_axis_hash",
            "parent_mask_hash",
            "label_commitment",
            "eligibility_commitment",
            "block_capacity_hash",
            "frozen_action_axis_hash",
            "endpoint_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in ("route_count", "endpoint_count"):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        object.__setattr__(self, "rank", _nonnegative_int(self.rank, name="rank"))
        if self.block_legality_scope not in {
            "registered_compact_capacities",
            "explicit_unconstrained_reference",
        }:
            raise ValueError("unsupported CUDA replay block-legality scope")
        if self.rounding_key_id != RELAXED_KKT_ROUNDING_KEY_ID:
            raise ValueError("CUDA replay rounding key differs from KKT")
        endpoints = tuple(self.endpoints)
        action_hashes = tuple(
            _identity(value, name="frozen action receipt hash")
            for value in self.frozen_action_receipt_hashes
        )
        packed_hashes = tuple(
            _identity(value, name="packed action hash")
            for value in self.packed_action_hashes
        )
        if (
            not endpoints
            or len(endpoints) != self.endpoint_count
            or any(
                type(value) is not V1DeviceShellHardFQEndpointReceipt
                for value in endpoints
            )
            or tuple(value.endpoint_label for value in endpoints)
            != tuple(range(1, self.endpoint_count + 1))
            or any(value.branch is not self.branch for value in endpoints)
            or action_hashes
            != tuple(value.frozen_action_receipt_hash for value in endpoints)
            or packed_hashes != tuple(value.packed_action_hash for value in endpoints)
        ):
            raise ValueError("CUDA replay endpoint/action axis is incomplete")
        expected_action_axis = content_hash(
            tuple(
                (
                    value.frozen_action_receipt_hash,
                    value.packed_action_hash,
                    value.removed_route_index_hash,
                    value.resulting_mask_hash,
                )
                for value in endpoints
            ),
            prefix=_ACTION_AXIS_PREFIX,
        )
        expected_endpoint_axis = content_hash(
            tuple(value.receipt_hash for value in endpoints),
            prefix=_ENDPOINT_AXIS_PREFIX,
        )
        if (
            self.frozen_action_axis_hash != expected_action_axis
            or self.endpoint_axis_hash != expected_endpoint_axis
        ):
            raise ValueError("CUDA replay action or endpoint axis does not replay")
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "frozen_action_receipt_hashes", action_hashes)
        object.__setattr__(self, "packed_action_hashes", packed_hashes)
        if self.algorithm_id != DEVICE_SHELL_HARD_FQ_CUDA_ALGORITHM_ID:
            raise ValueError("unsupported CUDA replay algorithm")
        if self.primitive_probe_id != DEVICE_SHELL_HARD_FQ_CUDA_PRIMITIVE_PROBE_ID:
            raise ValueError("unsupported CUDA primitive probe")
        if self.canonical_hard_q_id != RELAXED_KKT_CANONICAL_HARD_Q_ID:
            raise ValueError("unsupported canonical hard-Q equation")
        if self.outward_arithmetic_id != DEVICE_SHELL_HARD_FQ_OUTWARD_ARITHMETIC_ID:
            raise ValueError("CUDA replay endpoint arithmetic owner changed")
        required = (
            self.complete_registered_ladder_replayed,
            self.label_membership_used_directly,
            self.stable_physical_tie_order,
            self.full_axis_h200_repeat_parity_required,
        )
        forbidden = (
            self.full_axis_h200_repeat_parity_qualified,
            self.nonmonotone_q_early_stop_used,
            self.ordinary_route_order_d2h,
            self.ordinary_removed_indices_materialized,
            self.ordinary_hard_masks_materialized,
            self.ordinary_hard_actions_materialized,
            self.face_stability_authority,
            self.c_stream_authority,
            self.nonlinear_remainder_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.production_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in (*required, *forbidden)):
            raise TypeError("CUDA replay flags must be exact booleans")
        if required != (True, True, True, True) or any(forbidden):
            raise PermissionError("CUDA hard-F/Q replay crossed its authority boundary")
        if self.schema != DEVICE_SHELL_HARD_FQ_CUDA_REPLAY_RECEIPT_SCHEMA:
            raise ValueError("unsupported CUDA replay receipt schema")

    @property
    def safe_action_hashes(self) -> tuple[str, ...]:
        return tuple(
            value.packed_action_hash
            for value in self.endpoints
            if value.status is V1DeviceShellHardFQStatus.SAFE_FEASIBLE
        )

    @property
    def admissible_action_hashes(self) -> tuple[str, ...]:
        return tuple(
            value.packed_action_hash
            for value in self.endpoints
            if value.status is not V1DeviceShellHardFQStatus.ROBUST_IMPOSSIBLE
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_RECEIPT_PREFIX)


@dataclass(frozen=True, slots=True)
class V1DeviceShellHardFQCUDATelemetry:
    """Array-free HBM, scan, D2H, synchronization, and cleanup evidence."""

    replay_receipt_hash: str
    forecast_hash: str
    design_hash: str
    execution_receipt_hash: str
    problem_hash: str
    policy_hash: str
    branch: V1RelaxedKKTBranch
    branch_identity_receipt_hash: str
    algorithm_id: str
    outward_arithmetic_id: str
    device: str
    route_count: int
    rank: int
    endpoint_count: int
    maximum_selected_count: int
    label_grouping_scan_count: int
    label_route_visit_count: int
    streamed_problem_h2d_scan_count: int
    streamed_problem_value_visit_count: int
    fp64_prefix_cumsum_count: int
    fp64_prefix_value_visit_count: int
    endpoint_summary_gather_count: int
    route_scale_int64_permutation_peak_count: int
    bulk_h2d_tensor_payload_bytes: int
    compact_summary_d2h_tensor_payload_bytes: int
    scalar_validation_d2h_tensor_payload_bytes: int
    label_axis_d2h_tensor_payload_bytes: int
    route_order_d2h_tensor_payload_bytes: int
    peak_allocated_hbm_bytes: int
    peak_reserved_hbm_bytes: int
    grouping_seconds: float
    prefix_replay_seconds: float
    total_wall_seconds: float
    primitive_probe_scalar_d2h_tensor_payload_bytes: int = 1
    label_validation_scalar_d2h_tensor_payload_bytes: int = 1
    scalar_device_to_host_synchronization_count: int = 2
    compact_summary_device_to_host_synchronization_count: int = 1
    explicit_cuda_synchronize_count: int = 3
    total_device_synchronization_count: int = 6
    torch_deterministic_algorithms_enabled: bool = True
    torch_deterministic_warn_only: bool = False
    stable_integer_argsort_probe_passed: bool = True
    bitwise_repeat_cumsum_probe_passed: bool = True
    primitive_probe_local_qualification_only: bool = True
    full_axis_h200_repeat_parity_qualified: bool = False
    floating_scatter_add_used: bool = False
    floating_bincount_used: bool = False
    floating_atomics_used: bool = False
    route_scale_workspace_released: bool = True
    bundle_label_handles_evicted: bool = False
    face_stability_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    production_authority: bool = False
    pareto_authority: bool = False
    schema: str = DEVICE_SHELL_HARD_FQ_CUDA_TELEMETRY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "replay_receipt_hash",
            "forecast_hash",
            "design_hash",
            "execution_receipt_hash",
            "problem_hash",
            "policy_hash",
            "branch_identity_receipt_hash",
            "algorithm_id",
            "outward_arithmetic_id",
            "device",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("CUDA telemetry branch must be exact")
        positive_fields = (
            "route_count",
            "endpoint_count",
            "maximum_selected_count",
            "label_grouping_scan_count",
            "label_route_visit_count",
            "streamed_problem_h2d_scan_count",
            "streamed_problem_value_visit_count",
            "fp64_prefix_cumsum_count",
            "fp64_prefix_value_visit_count",
            "endpoint_summary_gather_count",
            "route_scale_int64_permutation_peak_count",
            "bulk_h2d_tensor_payload_bytes",
            "compact_summary_d2h_tensor_payload_bytes",
            "scalar_validation_d2h_tensor_payload_bytes",
            "primitive_probe_scalar_d2h_tensor_payload_bytes",
            "label_validation_scalar_d2h_tensor_payload_bytes",
            "scalar_device_to_host_synchronization_count",
            "compact_summary_device_to_host_synchronization_count",
            "explicit_cuda_synchronize_count",
            "total_device_synchronization_count",
        )
        for name in positive_fields:
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        object.__setattr__(self, "rank", _nonnegative_int(self.rank, name="rank"))
        for name in (
            "label_axis_d2h_tensor_payload_bytes",
            "route_order_d2h_tensor_payload_bytes",
            "peak_allocated_hbm_bytes",
            "peak_reserved_hbm_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        for name in ("grouping_seconds", "prefix_replay_seconds", "total_wall_seconds"):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        prefix_count = 2 * self.rank + 3
        if (
            self.maximum_selected_count > self.route_count
            or self.label_grouping_scan_count != 1
            or self.label_route_visit_count != self.route_count
            or self.streamed_problem_h2d_scan_count != self.rank + 3
            or self.streamed_problem_value_visit_count
            != self.route_count * (self.rank + 3)
            or self.fp64_prefix_cumsum_count != prefix_count
            or self.fp64_prefix_value_visit_count
            != self.maximum_selected_count * prefix_count
            or self.endpoint_summary_gather_count != prefix_count
            or self.route_scale_int64_permutation_peak_count != 1
            or self.bulk_h2d_tensor_payload_bytes
            != 8 * self.route_count * (self.rank + 3)
            or self.compact_summary_d2h_tensor_payload_bytes
            != 8 * self.endpoint_count * prefix_count
            or self.scalar_validation_d2h_tensor_payload_bytes != 2
            or self.primitive_probe_scalar_d2h_tensor_payload_bytes != 1
            or self.label_validation_scalar_d2h_tensor_payload_bytes != 1
            or self.scalar_device_to_host_synchronization_count != 2
            or self.compact_summary_device_to_host_synchronization_count != 1
            or self.explicit_cuda_synchronize_count != 3
            or self.total_device_synchronization_count != 6
            or self.label_axis_d2h_tensor_payload_bytes != 0
            or self.route_order_d2h_tensor_payload_bytes != 0
        ):
            raise ValueError("CUDA replay telemetry counters do not replay")
        if self.peak_allocated_hbm_bytes > self.peak_reserved_hbm_bytes:
            raise ValueError("allocated HBM cannot exceed reserved HBM")
        required = (
            self.torch_deterministic_algorithms_enabled,
            self.stable_integer_argsort_probe_passed,
            self.bitwise_repeat_cumsum_probe_passed,
            self.primitive_probe_local_qualification_only,
            self.route_scale_workspace_released,
        )
        forbidden = (
            self.full_axis_h200_repeat_parity_qualified,
            self.torch_deterministic_warn_only,
            self.floating_scatter_add_used,
            self.floating_bincount_used,
            self.floating_atomics_used,
            self.bundle_label_handles_evicted,
            self.face_stability_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.production_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in (*required, *forbidden)):
            raise TypeError("CUDA replay telemetry flags must be exact booleans")
        if required != (True, True, True, True, True) or any(forbidden):
            raise PermissionError("CUDA replay telemetry crossed a closed gate")
        if self.algorithm_id != DEVICE_SHELL_HARD_FQ_CUDA_ALGORITHM_ID:
            raise ValueError("CUDA telemetry algorithm changed")
        if self.outward_arithmetic_id != DEVICE_SHELL_HARD_FQ_OUTWARD_ARITHMETIC_ID:
            raise ValueError("CUDA telemetry arithmetic owner changed")
        if self.schema != DEVICE_SHELL_HARD_FQ_CUDA_TELEMETRY_SCHEMA:
            raise ValueError("unsupported CUDA telemetry schema")

    @property
    def telemetry_hash(self) -> str:
        return content_hash(self, prefix=_TELEMETRY_PREFIX)


@dataclass(frozen=True, slots=True)
class V1DeviceShellHardFQCUDAReplayBundle:
    """Array-free replay receipt plus preflight and execution evidence."""

    receipt: V1DeviceShellHardFQCUDAReplayReceipt
    forecast: V1DeviceShellHardFQCUDAHBMForecast
    telemetry: V1DeviceShellHardFQCUDATelemetry
    design: V1DeviceShellHardFQCUDAReplayDesign
    schema: str = DEVICE_SHELL_HARD_FQ_CUDA_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1DeviceShellHardFQCUDAReplayReceipt:
            raise TypeError("CUDA replay bundle requires an exact receipt")
        if type(self.forecast) is not V1DeviceShellHardFQCUDAHBMForecast:
            raise TypeError("CUDA replay bundle requires an exact HBM forecast")
        if type(self.telemetry) is not V1DeviceShellHardFQCUDATelemetry:
            raise TypeError("CUDA replay bundle requires exact telemetry")
        if type(self.design) is not V1DeviceShellHardFQCUDAReplayDesign:
            raise TypeError("CUDA replay bundle requires an exact design")
        if (
            self.receipt.design_hash != self.design.design_hash
            or self.receipt.hbm_forecast_hash != self.forecast.forecast_hash
            or self.telemetry.replay_receipt_hash != self.receipt.receipt_hash
            or self.telemetry.forecast_hash != self.forecast.forecast_hash
            or self.telemetry.design_hash != self.design.design_hash
            or self.receipt.branch is not self.forecast.branch
            or self.receipt.branch is not self.telemetry.branch
            or self.receipt.problem_hash != self.forecast.problem_hash
            or self.receipt.problem_hash != self.telemetry.problem_hash
        ):
            raise ValueError("CUDA replay receipt, forecast, and telemetry do not close")
        if self.schema != DEVICE_SHELL_HARD_FQ_CUDA_BUNDLE_SCHEMA:
            raise ValueError("unsupported CUDA replay bundle schema")

    @property
    def bundle_hash(self) -> str:
        return content_hash(self, prefix=_BUNDLE_PREFIX)


def _branch_pair_from_exact_bundle(
    *,
    execution: V1RelaxedKKTDeviceShellExecutionBundle,
    branch: V1RelaxedKKTBranch,
) -> tuple[V1DeviceShellLabelHandle, V1RelaxedKKTDeviceShellBranchIdentityReceipt]:
    execution_receipt = execution.receipt
    if branch is V1RelaxedKKTBranch.SPECTRAL:
        source_receipt = execution_receipt.spectral
        handle = execution.spectral_label_handle
    else:
        source_receipt = execution_receipt.diagonal_l1
        handle = execution.diagonal_l1_label_handle
    if execution.receipt is not execution_receipt:
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.EXECUTION_IDENTITY_MISMATCH,
            "execution receipt changed while resolving its owned branch pair",
        )
    if (
        type(handle) is not V1DeviceShellLabelHandle
        or type(source_receipt)
        is not V1RelaxedKKTDeviceShellBranchIdentityReceipt
        or source_receipt.branch is not branch
    ):
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.BRANCH_IDENTITY_MISMATCH,
            "execution did not own one exact branch handle/receipt pair",
        )
    return handle, source_receipt


def _verify_execution_identity(
    *,
    execution: V1RelaxedKKTDeviceShellExecutionBundle,
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    source_receipt: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
) -> None:
    receipt = execution.receipt
    base = execution.base_execution
    branch_result = (
        base.result.spectral
        if branch is V1RelaxedKKTBranch.SPECTRAL
        else base.result.diagonal_l1
    )
    registered_source = (
        receipt.spectral
        if branch is V1RelaxedKKTBranch.SPECTRAL
        else receipt.diagonal_l1
    )
    if (
        receipt.problem_hash != problem.problem_hash
        or base.result.problem_hash != problem.problem_hash
        or base.result.policy_hash != receipt.policy_hash
        or base.telemetry.numerical_policy_hash != receipt.policy_hash
        or base.telemetry.route_axis_hash != problem.route_axis_hash
        or base.telemetry.source_geometry_hash != problem.source_geometry_hash
        or registered_source.receipt_hash != source_receipt.receipt_hash
        or branch_result.branch is not branch
        or base.telemetry.algorithm_id != RELAXED_KKT_FUSED_CUDA_ALGORITHM_ID
    ):
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.EXECUTION_IDENTITY_MISMATCH,
            "problem, policy, geometry, algorithm, or branch differs from execution",
        )
    try:
        _reference._verify_problem_ladder_identity(
            problem=problem,
            ladder=source_receipt.frozen_ladder,
        )
    except V1DeviceShellHardFQRefusal as error:
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.PROBLEM_IDENTITY_MISMATCH,
            str(error),
        )


def _build_hbm_forecast(
    *,
    execution: V1RelaxedKKTDeviceShellExecutionBundle,
    problem: V1RelaxedKKTProblem,
    source_receipt: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    branch: V1RelaxedKKTBranch,
    design: V1DeviceShellHardFQCUDAReplayDesign,
    torch: Any,
    device: Any,
) -> V1DeviceShellHardFQCUDAHBMForecast:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    ladder = source_receipt.frozen_ladder
    n = problem.route_count
    endpoint_count = ladder.endpoint_count
    selected = ladder.cumulative_counts[-1]
    key_bytes = 2 * n
    permutation_bytes = 8 * n
    input_bytes = 8 * n
    selected_bytes = 16 * selected
    summary_bytes = 8 * endpoint_count * (2 * problem.rank + 3)
    endpoint_index_bytes = 8 * endpoint_count
    sort_bytes = DEVICE_SHELL_HARD_FQ_CUDA_ARGSORT_WORKSPACE_BYTES_PER_ROUTE * n
    subtotal = max(
        key_bytes + permutation_bytes + sort_bytes,
        permutation_bytes + input_bytes + selected_bytes,
    ) + summary_bytes + endpoint_index_bytes
    guard = max(
        DEVICE_SHELL_HARD_FQ_CUDA_MINIMUM_ALLOCATOR_GUARD_BYTES,
        math.ceil(DEVICE_SHELL_HARD_FQ_CUDA_ALLOCATOR_GUARD_FRACTION * subtotal),
    )
    incremental = subtotal + guard
    process_peak = reserved + incremental
    ceiling = math.floor(total_bytes * design.hbm_ceiling_fraction)
    return V1DeviceShellHardFQCUDAHBMForecast(
        design_hash=design.design_hash,
        execution_receipt_hash=execution.receipt.receipt_hash,
        problem_hash=problem.problem_hash,
        policy_hash=execution.receipt.policy_hash,
        branch=branch,
        branch_identity_receipt_hash=source_receipt.receipt_hash,
        device=str(device),
        route_count=n,
        rank=problem.rank,
        endpoint_count=endpoint_count,
        maximum_selected_count=selected,
        device_total_memory_bytes=int(total_bytes),
        device_free_memory_bytes=int(free_bytes),
        process_allocated_before_bytes=allocated,
        process_reserved_before_bytes=reserved,
        hbm_ceiling_fraction=design.hbm_ceiling_fraction,
        hbm_ceiling_bytes=ceiling,
        resident_label_bytes=n,
        grouping_key_bytes=key_bytes,
        retained_permutation_bytes=permutation_bytes,
        streamed_input_vector_bytes=input_bytes,
        selected_pair_peak_bytes=selected_bytes,
        compact_summary_bytes=summary_bytes,
        compact_endpoint_index_bytes=endpoint_index_bytes,
        stable_argsort_workspace_upper_bytes=sort_bytes,
        allocator_guard_bytes=guard,
        estimated_incremental_peak_bytes=incremental,
        estimated_process_peak_bytes=process_peak,
        admitted=(allocated + incremental <= ceiling and incremental <= cuda_reusable_memory_bytes(device, framework=torch, free_bytes=free_bytes)),
    )


def _verify_measured_hbm(
    *,
    forecast: V1DeviceShellHardFQCUDAHBMForecast,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> None:
    allocated = _nonnegative_int(peak_allocated_bytes, name="measured peak allocated")
    reserved = _nonnegative_int(peak_reserved_bytes, name="measured peak reserved")
    if allocated > reserved:
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.MEASURED_HBM_FORECAST_INVALIDATED,
            "measured allocated HBM exceeds measured reserved HBM",
        )
    if allocated > forecast.hbm_ceiling_bytes:
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.MEASURED_HBM_CEILING_EXCEEDED,
            "measured replay HBM crossed the registered 85% ceiling",
        )
    allocated_limit = (
        forecast.process_allocated_before_bytes
        + forecast.estimated_incremental_peak_bytes
    )
    if (
        allocated > allocated_limit
        or reserved > forecast.estimated_process_peak_bytes
    ):
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.MEASURED_HBM_FORECAST_INVALIDATED,
            "measured replay HBM exceeded its pre-outcome allocation forecast",
        )


def _require_cuda_determinism(torch: Any, device: Any) -> None:
    if not bool(torch.are_deterministic_algorithms_enabled()):
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.DETERMINISM_DISABLED,
            "torch deterministic algorithms must already be enabled",
        )
    warn_only = getattr(
        torch,
        "is_deterministic_algorithms_warn_only_enabled",
        lambda: False,
    )
    if bool(warn_only()):
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.DETERMINISM_DISABLED,
            "warn-only deterministic mode cannot authorize replay",
        )
    try:
        keys = torch.tensor([2, 1, 2, 1, 3, 1], device=device, dtype=torch.int16)
        expected = torch.tensor(
            [1, 3, 5, 0, 2, 4],
            device=device,
            dtype=torch.int64,
        )
        stable = torch.all(torch.argsort(keys, stable=True) == expected)
        values = torch.tensor(
            [1.0, -0.25, 0.125, -0.0625, 0.03125],
            device=device,
            dtype=torch.float64,
        )
        first = torch.cumsum(values, dim=0, dtype=torch.float64)
        second = torch.cumsum(values, dim=0, dtype=torch.float64)
        repeated = torch.all(first.view(torch.int64) == second.view(torch.int64))
        qualified = bool(torch.logical_and(stable, repeated).item())
        del keys, expected, stable, values, first, second, repeated
    except Exception as error:
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.CUDA_DETERMINISM_PRIMITIVE_UNQUALIFIED,
            f"registered stable argsort/cumsum probe failed: {error}",
        )
    if not qualified:
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.CUDA_DETERMINISM_PRIMITIVE_UNQUALIFIED,
            "stable argsort or repeat FP64 cumsum did not replay bitwise",
        )


def _assert_problem_array(
    value: np.ndarray,
    *,
    shape: tuple[int, ...],
    name: str,
) -> None:
    if (
        type(value) is not np.ndarray
        or value.shape != shape
        or value.dtype != np.dtype(np.float64)
        or not value.flags.c_contiguous
        or value.flags.writeable
        or not np.isfinite(value).all()
    ):
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.PROBLEM_IDENTITY_MISMATCH,
            f"{name} is not registered immutable contiguous float64",
        )


def _validate_label_tensor(
    *,
    labels: Any,
    problem: V1RelaxedKKTProblem,
    ladder: V1DeviceShellFrozenLadderReceipt,
    lease: Any,
    torch: Any,
) -> Any:
    if (
        type(labels) is not torch.Tensor
        or labels.device.type != "cuda"
        or labels.dtype is not torch.uint8
        or labels.ndim != 1
        or tuple(labels.shape) != (problem.route_count,)
        or not labels.is_contiguous()
        or str(labels.device) != lease.device_identity
        or ladder.handle_kind != "torch_cuda"
        or ladder.device_identity != str(labels.device)
    ):
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.MALFORMED_CUDA_LABELS,
            "labels must be exact contiguous CUDA uint8 on the frozen device",
        )
    return labels.device


def _nonnegative_upper_from_prefix(*, nominal: float, count: int) -> float:
    center = _finite_nonnegative(nominal, name="nonnegative CUDA prefix")
    terms = _nonnegative_int(count, name="CUDA prefix term count")
    if terms == 0:
        return 0.0
    if terms == 1:
        return center
    gamma = _reference._gamma_n(terms - 1)
    magnitude = _reference._next_up(
        center / _reference._next_down(1.0 - gamma)
    )
    # Count is a conservative nonzero-count upper bound; this can only widen
    # the existing reference enclosure.
    underflow = _reference._up_multiply(float(terms), _reference._MIN_NORMAL)
    return _reference._up_add(magnitude, underflow)


def _upload_vector(torch: Any, value: np.ndarray, *, device: Any) -> Any:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if (
        tensor.dtype is not torch.float64
        or tensor.device != device
        or tensor.ndim != 1
        or not tensor.is_contiguous()
    ):
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.NONFINITE_CUDA_SUMMARY,
            "streamed CUDA vector changed dtype/device/layout",
        )
    return tensor


def _prefix_endpoints(
    *,
    torch: Any,
    source: Any,
    selected: Any,
    endpoint_indices: Any,
) -> Any:
    gathered = torch.index_select(source, 0, selected)
    prefix = torch.cumsum(gathered, dim=0, dtype=torch.float64)
    result = torch.index_select(prefix, 0, endpoint_indices)
    del gathered, prefix
    return result


def _cuda_summaries(
    *,
    torch: Any,
    device: Any,
    labels: Any,
    problem: V1RelaxedKKTProblem,
    ladder: V1DeviceShellFrozenLadderReceipt,
) -> tuple[np.ndarray, float, float, int, int]:
    """Compute compact summaries; every route-scale local dies on return."""

    endpoint_count = ladder.endpoint_count
    cumulative = ladder.cumulative_counts
    selected_count = cumulative[-1]
    rank = problem.rank

    grouping_started = time.perf_counter()
    key = labels.to(dtype=torch.int16)
    zero_mask = key == 0
    key.masked_fill_(zero_mask, endpoint_count + 1)
    del zero_mask
    try:
        permutation = torch.argsort(key, stable=True)
    except Exception as error:
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.CUDA_DETERMINISM_PRIMITIVE_UNQUALIFIED,
            f"stable CUDA integer argsort is unavailable: {error}",
        )
    if (
        permutation.dtype is not torch.int64
        or permutation.device != device
        or tuple(permutation.shape) != (problem.route_count,)
        or not permutation.is_contiguous()
    ):
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.MALFORMED_CUDA_LABELS,
            "stable grouping did not emit one CUDA int64 permutation",
        )
    del key

    starts = (0, *cumulative[:-1])
    boundary_positions_host = np.asarray(
        tuple(
            value
            for pair in zip(
                starts,
                (value - 1 for value in cumulative),
                strict=True,
            )
            for value in pair
        ),
        dtype=np.int64,
    )
    boundary_positions = torch.tensor(
        boundary_positions_host,
        device=device,
        dtype=torch.int64,
    )
    expected = torch.tensor(
        np.repeat(np.arange(1, endpoint_count + 1, dtype=np.uint8), 2),
        device=device,
        dtype=torch.uint8,
    )
    boundary_routes = torch.index_select(permutation, 0, boundary_positions)
    observed = torch.index_select(labels, 0, boundary_routes)
    valid = torch.logical_and(
        torch.all(labels <= endpoint_count),
        torch.all(observed == expected),
    )
    if selected_count < problem.route_count:
        valid = torch.logical_and(valid, labels[permutation[selected_count]] == 0)
    valid_host = bool(valid.item())
    del boundary_positions, expected, boundary_routes, observed, valid
    if not valid_host:
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.LABEL_SUMMARY_MISMATCH,
            "label shells differ from frozen cumulative counts",
        )
    grouping_seconds = time.perf_counter() - grouping_started

    selected = permutation[:selected_count]
    endpoint_indices = torch.tensor(
        np.asarray(cumulative, dtype=np.int64) - 1,
        device=device,
        dtype=torch.int64,
    )
    summary = torch.empty(
        (2 * rank + 3, endpoint_count),
        device=device,
        dtype=torch.float64,
    )
    replay_started = time.perf_counter()
    resource = _upload_vector(torch, problem.resource, device=device)
    summary[0].copy_(
        _prefix_endpoints(
            torch=torch,
            source=resource,
            selected=selected,
            endpoint_indices=endpoint_indices,
        )
    )
    del resource
    work = _upload_vector(torch, problem.corrected_work, device=device)
    summary[1].copy_(
        _prefix_endpoints(
            torch=torch,
            source=work,
            selected=selected,
            endpoint_indices=endpoint_indices,
        )
    )
    del work
    diagonal = _upload_vector(torch, problem.residual_diagonal, device=device)
    gathered_diagonal = torch.index_select(diagonal, 0, selected)
    del diagonal
    gathered_diagonal.sqrt_()
    infinity = torch.tensor(math.inf, device=device, dtype=torch.float64)
    torch.nextafter(gathered_diagonal, infinity, out=gathered_diagonal)
    diagonal_prefix = torch.cumsum(
        gathered_diagonal,
        dim=0,
        dtype=torch.float64,
    )
    summary[2].copy_(torch.index_select(diagonal_prefix, 0, endpoint_indices))
    del gathered_diagonal, infinity, diagonal_prefix
    for row in range(rank):
        factor_row = _upload_vector(
            torch,
            problem.low_rank_factor[row],
            device=device,
        )
        gathered = torch.index_select(factor_row, 0, selected)
        del factor_row
        prefix = torch.cumsum(gathered, dim=0, dtype=torch.float64)
        summary[3 + row].copy_(torch.index_select(prefix, 0, endpoint_indices))
        del prefix
        gathered.abs_()
        absolute_prefix = torch.cumsum(gathered, dim=0, dtype=torch.float64)
        summary[3 + rank + row].copy_(
            torch.index_select(absolute_prefix, 0, endpoint_indices)
        )
        del gathered, absolute_prefix
    torch.cuda.synchronize(device)
    prefix_seconds = time.perf_counter() - replay_started
    host = summary.detach().to(device="cpu", non_blocking=False).numpy().copy()
    if (
        host.shape != (2 * rank + 3, endpoint_count)
        or host.dtype != np.dtype(np.float64)
        or not np.isfinite(host).all()
    ):
        _refuse(
            V1DeviceShellHardFQCUDARefusalCode.NONFINITE_CUDA_SUMMARY,
            "compact CUDA summary is malformed or nonfinite",
        )
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    del summary, endpoint_indices, selected, permutation
    return host, grouping_seconds, prefix_seconds, peak_allocated, peak_reserved


def _endpoint_receipts(
    *,
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    ladder: V1DeviceShellFrozenLadderReceipt,
    summaries: np.ndarray,
) -> tuple[V1DeviceShellHardFQEndpointReceipt, ...]:
    rank = problem.rank
    results: list[V1DeviceShellHardFQEndpointReceipt] = []
    for offset, action in enumerate(ladder.actions):
        endpoint_label = offset + 1
        count = action.cumulative_removed_count
        resource_nominal = float(summaries[0, offset])
        resource_abs = _nonnegative_upper_from_prefix(
            nominal=resource_nominal,
            count=count,
        )
        resource_lower, resource_upper = (
            _reference._signed_sum_interval_from_nominal(
                nominal=resource_nominal,
                absolute_sum_upper=resource_abs,
                term_count=count,
            )
        )
        f_nominal = float(summaries[1, offset])
        f_abs = _nonnegative_upper_from_prefix(nominal=f_nominal, count=count)
        f_lower, f_upper = _reference._signed_sum_interval_from_nominal(
            nominal=f_nominal,
            absolute_sum_upper=f_abs,
            term_count=count,
        )
        row_nominals = summaries[3 : 3 + rank, offset]
        row_abs = summaries[3 + rank : 3 + 2 * rank, offset]
        row_lowers: list[float] = []
        row_uppers: list[float] = []
        for row in range(rank):
            absolute_upper = _nonnegative_upper_from_prefix(
                nominal=float(row_abs[row]),
                count=count,
            )
            lower, upper = _reference._signed_sum_interval_from_nominal(
                nominal=float(row_nominals[row]),
                absolute_sum_upper=absolute_upper,
                term_count=count,
            )
            square_lower, square_upper = _reference._square_interval(lower, upper)
            row_lowers.append(square_lower)
            row_uppers.append(square_upper)
        low_q_nominal = float(np.dot(row_nominals, row_nominals))
        low_q_lower, low_q_upper = _reference._nonnegative_interval_sum(
            row_lowers,
            row_uppers,
        )
        if not low_q_lower <= low_q_nominal <= low_q_upper:
            _refuse(
                V1DeviceShellHardFQCUDARefusalCode.NONFINITE_CUDA_SUMMARY,
                "nominal low-rank Q escaped the registered outward enclosure",
            )
        spectral_residual = _reference._up_multiply(
            problem.residual_spectral_bound,
            float(count),
        )
        diagonal_upper = _nonnegative_upper_from_prefix(
            nominal=float(summaries[2, offset]),
            count=count,
        )
        diagonal_residual = _reference._up_square(diagonal_upper)
        canonical_residual = min(spectral_residual, diagonal_residual)
        q_upper = _reference._up_add(low_q_upper, canonical_residual)
        block_prefix_counts = tuple(row[offset] for row in ladder.block_prefix_counts)
        legal = all(
            observed <= block.removal_capacity
            for observed, block in zip(
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
                (low_q_lower, problem.quadratic_budget),
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
            or low_q_lower > problem.quadratic_budget
        )
        status = (
            V1DeviceShellHardFQStatus.SAFE_FEASIBLE
            if safe
            else V1DeviceShellHardFQStatus.ROBUST_IMPOSSIBLE
            if impossible
            else V1DeviceShellHardFQStatus.FD_AMBIGUOUS
        )
        results.append(
            V1DeviceShellHardFQEndpointReceipt(
                branch=branch,
                endpoint_label=endpoint_label,
                cumulative_removed_count=count,
                frozen_action_receipt_hash=action.receipt_hash,
                packed_action_hash=action.packed_action_hash,
                removed_route_index_hash=action.removed_route_index_hash,
                resulting_mask_hash=action.resulting_mask_hash,
                resulting_mask_packed_sha256=action.resulting_mask_packed_sha256,
                block_prefix_counts=block_prefix_counts,
                resource_nominal=resource_nominal,
                resource_lower=max(0.0, resource_lower),
                resource_upper=max(0.0, resource_upper),
                f_nominal=f_nominal,
                f_lower=max(0.0, f_lower),
                f_upper=max(0.0, f_upper),
                low_rank_q_nominal=low_q_nominal,
                low_rank_q_lower=max(0.0, low_q_lower),
                low_rank_q_upper=max(0.0, low_q_upper),
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
    return tuple(results)


def _require_hbm_admission(forecast: V1DeviceShellHardFQCUDAHBMForecast) -> None:
    if forecast.admitted:
        return
    reusable = forecast.device_free_memory_bytes + max(
        0,
        forecast.process_reserved_before_bytes - forecast.process_allocated_before_bytes,
    )
    code = (
        V1DeviceShellHardFQCUDARefusalCode.HBM_AVAILABLE_MEMORY_EXCEEDED
        if forecast.estimated_incremental_peak_bytes > reusable
        else V1DeviceShellHardFQCUDARefusalCode.HBM_FORECAST_EXCEEDED
    )
    _refuse(code, "replay exceeds its registered pre-outcome 85% HBM envelope")


def replay_v1_device_shell_hard_fq_cuda(
    *,
    execution: V1RelaxedKKTDeviceShellExecutionBundle,
    problem: V1RelaxedKKTProblem,
    branch: V1RelaxedKKTBranch,
    design: V1DeviceShellHardFQCUDAReplayDesign,
) -> V1DeviceShellHardFQCUDAReplayBundle:
    """Replay all frozen endpoints without route-scale D2H or CPU fallback."""

    if type(execution) is not V1RelaxedKKTDeviceShellExecutionBundle:
        raise TypeError("CUDA hard-F/Q replay requires an exact shell execution")
    if type(problem) is not V1RelaxedKKTProblem:
        raise TypeError("CUDA hard-F/Q replay requires an exact KKT problem")
    if type(branch) is not V1RelaxedKKTBranch:
        raise TypeError("CUDA hard-F/Q replay requires an exact KKT branch")
    if type(design) is not V1DeviceShellHardFQCUDAReplayDesign:
        raise TypeError("CUDA hard-F/Q replay requires an exact CUDA design")
    handle, source_receipt = _branch_pair_from_exact_bundle(
        execution=execution,
        branch=branch,
    )
    _verify_execution_identity(
        execution=execution,
        problem=problem,
        branch=branch,
        source_receipt=source_receipt,
    )
    ladder = source_receipt.frozen_ladder
    operation = V1DeviceShellBorrowOperation.FQ_HARD_REPLAY
    total_started = time.perf_counter()
    with handle.borrow_for_registered_operation(operation=operation) as lease:
        if lease.backend_kind != "torch_cuda":
            _refuse(
                V1DeviceShellHardFQCUDARefusalCode.CPU_HANDLE_FORBIDDEN,
                "production CUDA replay has no CPU or NumPy fallback",
            )
        try:
            import torch
        except ImportError as error:  # pragma: no cover - environment dependent.
            _refuse(
                V1DeviceShellHardFQCUDARefusalCode.CUDA_UNAVAILABLE,
                f"torch is unavailable: {error}",
            )
        if not torch.cuda.is_available():
            _refuse(
                V1DeviceShellHardFQCUDARefusalCode.CUDA_UNAVAILABLE,
                "torch reports no CUDA device",
            )
        labels = lease._labels_for_registered_operation(operation=operation)
        device = _validate_label_tensor(
            labels=labels,
            problem=problem,
            ladder=ladder,
            lease=lease,
            torch=torch,
        )
        base = execution.base_execution
        if (
            base.telemetry.device != str(device)
            or base.forecast.device != str(device)
        ):
            _refuse(
                V1DeviceShellHardFQCUDARefusalCode.EXECUTION_IDENTITY_MISMATCH,
                "resident label device differs from the fused CUDA execution",
            )
        _assert_problem_array(
            problem.resource,
            shape=(problem.route_count,),
            name="resource",
        )
        _assert_problem_array(
            problem.corrected_work,
            shape=(problem.route_count,),
            name="corrected_work",
        )
        _assert_problem_array(
            problem.residual_diagonal,
            shape=(problem.route_count,),
            name="residual_diagonal",
        )
        _assert_problem_array(
            problem.low_rank_factor,
            shape=(problem.rank, problem.route_count),
            name="low_rank_factor",
        )
        forecast = _build_hbm_forecast(
            execution=execution,
            problem=problem,
            source_receipt=source_receipt,
            branch=branch,
            design=design,
            torch=torch,
            device=device,
        )
        _require_hbm_admission(forecast)
        _require_cuda_determinism(torch, device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        try:
            (
                summaries,
                grouping_seconds,
                prefix_seconds,
                peak_allocated,
                peak_reserved,
            ) = _cuda_summaries(
                torch=torch,
                device=device,
                labels=labels,
                problem=problem,
                ladder=ladder,
            )
        except Exception as error:
            oom_type = getattr(torch, "OutOfMemoryError", ())
            if (oom_type and isinstance(error, oom_type)) or (
                "out of memory" in str(error).lower()
            ):
                _refuse(
                    V1DeviceShellHardFQCUDARefusalCode.CUDA_OUT_OF_MEMORY,
                    str(error),
                )
            raise
        finally:
            # Route-scale worker locals have died on return/exception.  Release
            # cached blocks without evicting the bundle-owned label handles.
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        _verify_measured_hbm(
            forecast=forecast,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )

    endpoints = _endpoint_receipts(
        problem=problem,
        branch=branch,
        ladder=ladder,
        summaries=summaries,
    )
    del summaries
    action_axis_hash = content_hash(
        tuple(
            (
                value.frozen_action_receipt_hash,
                value.packed_action_hash,
                value.removed_route_index_hash,
                value.resulting_mask_hash,
            )
            for value in endpoints
        ),
        prefix=_ACTION_AXIS_PREFIX,
    )
    endpoint_axis_hash = content_hash(
        tuple(value.receipt_hash for value in endpoints),
        prefix=_ENDPOINT_AXIS_PREFIX,
    )
    base = execution.base_execution
    branch_result = (
        base.result.spectral
        if branch is V1RelaxedKKTBranch.SPECTRAL
        else base.result.diagonal_l1
    )
    receipt = V1DeviceShellHardFQCUDAReplayReceipt(
        branch=branch,
        design_hash=design.design_hash,
        hbm_forecast_hash=forecast.forecast_hash,
        device_shell_execution_receipt_hash=execution.receipt.receipt_hash,
        device_shell_execution_design_hash=execution.execution_design.design_hash,
        cuda_execution_bundle_hash=base.bundle_hash,
        cuda_execution_design_hash=base.execution_design.design_hash,
        base_result_hash=base.result.result_hash,
        branch_result_hash=branch_result.result_hash,
        branch_kkt_receipt_hash=branch_result.receipt.receipt_hash,
        problem_hash=problem.problem_hash,
        problem_id=problem.problem_id,
        source_geometry_hash=problem.source_geometry_hash,
        policy_hash=execution.receipt.policy_hash,
        numerical_policy_hash=base.telemetry.numerical_policy_hash,
        source_branch_identity_receipt_hash=source_receipt.receipt_hash,
        frozen_ladder_receipt_hash=ladder.receipt_hash,
        source_rounding_hash=source_receipt.source_rounding_hash,
        route_axis_hash=problem.route_axis_hash,
        parent_mask_hash=ladder.parent_mask_hash,
        route_count=problem.route_count,
        rank=problem.rank,
        endpoint_count=ladder.endpoint_count,
        label_commitment=ladder.label_commitment,
        eligibility_commitment=ladder.eligibility_commitment,
        block_capacity_hash=ladder.block_capacity_hash,
        block_legality_scope=ladder.block_legality_scope,
        rounding_key_id=source_receipt.rounding_key_id,
        frozen_action_axis_hash=action_axis_hash,
        frozen_action_receipt_hashes=tuple(
            action.receipt_hash for action in ladder.actions
        ),
        packed_action_hashes=tuple(
            action.packed_action_hash for action in ladder.actions
        ),
        endpoint_axis_hash=endpoint_axis_hash,
        endpoints=endpoints,
    )
    telemetry = V1DeviceShellHardFQCUDATelemetry(
        replay_receipt_hash=receipt.receipt_hash,
        forecast_hash=forecast.forecast_hash,
        design_hash=design.design_hash,
        execution_receipt_hash=execution.receipt.receipt_hash,
        problem_hash=problem.problem_hash,
        policy_hash=execution.receipt.policy_hash,
        branch=branch,
        branch_identity_receipt_hash=source_receipt.receipt_hash,
        algorithm_id=design.algorithm_id,
        outward_arithmetic_id=design.outward_arithmetic_id,
        device=forecast.device,
        route_count=problem.route_count,
        rank=problem.rank,
        endpoint_count=ladder.endpoint_count,
        maximum_selected_count=ladder.cumulative_counts[-1],
        label_grouping_scan_count=1,
        label_route_visit_count=problem.route_count,
        streamed_problem_h2d_scan_count=problem.rank + 3,
        streamed_problem_value_visit_count=problem.route_count * (problem.rank + 3),
        fp64_prefix_cumsum_count=2 * problem.rank + 3,
        fp64_prefix_value_visit_count=(
            ladder.cumulative_counts[-1] * (2 * problem.rank + 3)
        ),
        endpoint_summary_gather_count=2 * problem.rank + 3,
        route_scale_int64_permutation_peak_count=1,
        bulk_h2d_tensor_payload_bytes=8 * problem.route_count * (problem.rank + 3),
        compact_summary_d2h_tensor_payload_bytes=(
            8 * ladder.endpoint_count * (2 * problem.rank + 3)
        ),
        scalar_validation_d2h_tensor_payload_bytes=2,
        label_axis_d2h_tensor_payload_bytes=0,
        route_order_d2h_tensor_payload_bytes=0,
        peak_allocated_hbm_bytes=peak_allocated,
        peak_reserved_hbm_bytes=peak_reserved,
        grouping_seconds=grouping_seconds,
        prefix_replay_seconds=prefix_seconds,
        total_wall_seconds=time.perf_counter() - total_started,
    )
    return V1DeviceShellHardFQCUDAReplayBundle(
        receipt=receipt,
        forecast=forecast,
        telemetry=telemetry,
        design=design,
    )


__all__ = [
    "DEVICE_SHELL_HARD_FQ_CUDA_ALGORITHM_ID",
    "DEVICE_SHELL_HARD_FQ_CUDA_HBM_CEILING_FRACTION",
    "DEVICE_SHELL_HARD_FQ_CUDA_PRIMITIVE_PROBE_ID",
    "V1DeviceShellHardFQCUDAHBMForecast",
    "V1DeviceShellHardFQCUDARefusal",
    "V1DeviceShellHardFQCUDARefusalCode",
    "V1DeviceShellHardFQCUDAReplayBundle",
    "V1DeviceShellHardFQCUDAReplayDesign",
    "V1DeviceShellHardFQCUDAReplayReceipt",
    "V1DeviceShellHardFQCUDATelemetry",
    "replay_v1_device_shell_hard_fq_cuda",
]
