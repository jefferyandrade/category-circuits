"""Device-shell rounding for signed-work least-work proposals.

This module consumes the opaque theorem state produced by
``modern_v1_signed_work_kkt_cuda`` and freezes a nested hard-ladder identity
using uint8 first-inclusion labels.  It is a proposal bridge only: the labels
and receipts here have no F/Q/R, calibration, oracle, commit, terminal, or
Pareto authority.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import InitVar, dataclass, field
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
from threading import RLock
from typing import Any, Final, Literal, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask

from .modern_v1_device_shell_ladder import (
    DEVICE_SHELL_DEFAULT_PHYSICAL_CHUNK_SIZE,
    DEVICE_SHELL_MAXIMUM_ENDPOINTS,
    V1DeviceShellCompactBlockCapacity,
    V1DeviceShellFrozenLadderReceipt,
    V1DeviceShellLabelHandle,
    freeze_v1_device_shell_ladder,
)
from .modern_v1_device_prefix_rank import V1DevicePrefixRankHandle
from .modern_v1_relaxed_kkt import V1RelaxedKKTBranch
from .modern_v1_compact_device_structural_source import (
    V1CompactDeviceStructuralSourceReceipt,
)
from .modern_v1_signed_work_kkt_cuda import (
    V1SignedWorkLeastWorkCUDABranchReceipt,
    V1SignedWorkLeastWorkCUDAPolicy,
    V1SignedWorkLeastWorkCUDAExecutionBundle,
    V1SignedWorkLeastWorkCUDADeviceProblemV2,
    V1SignedWorkLeastWorkCUDAProblem,
    V1SignedWorkLeastWorkCUDARefusal,
    V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
)

try:  # pragma: no cover - availability is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SIGNED_WORK_DEVICE_SHELL_ENDPOINT_PLAN_SCHEMA: Final = (
    "partition_free_v1_signed_work_device_shell_endpoint_plan_receipt_v1"
)
SIGNED_WORK_DEVICE_SHELL_DESIGN_SCHEMA: Final = (
    "partition_free_v1_signed_work_device_shell_design_v1"
)
SIGNED_WORK_DEVICE_SHELL_BRANCH_SCHEMA: Final = (
    "partition_free_v1_signed_work_device_shell_branch_identity_receipt_v1"
)
SIGNED_WORK_DEVICE_SHELL_EXECUTION_SCHEMA: Final = (
    "partition_free_v1_signed_work_device_shell_execution_receipt_v1"
)
SIGNED_WORK_DEVICE_SHELL_BUNDLE_SCHEMA: Final = (
    "partition_free_v1_signed_work_device_shell_execution_bundle_v1"
)
SIGNED_WORK_DEVICE_SHELL_FAMILY_SCHEMA: Final = (
    "partition_free_v1_signed_work_two_Q_branch_device_shell_family_v1"
)
SIGNED_WORK_DEVICE_SHELL_HBM_SCHEMA: Final = (
    "partition_free_v1_signed_work_device_shell_hbm_forecast_v1"
)
SIGNED_WORK_DEVICE_SHELL_KEY_ID: Final = (
    "negative_relaxed_gate_negative_kkt_reduced_advantage_"
    "singleton_corrected_work_per_resource_physical_index_v1"
)
SIGNED_WORK_DEVICE_SHELL_AUTHORITY: Final = (
    "signed_work_relaxed_state_to_uint8_first_inclusion_identity_only_"
    "no_additive_F_replay_no_FQR_C_K_O_commit_terminal_or_pareto_v1"
)
SIGNED_WORK_DEVICE_SHELL_PLAN_ORIGIN: Final = (
    "external_common_endpoint_plan_from_endogenous_frontier_pending_v1"
)

_PLAN_PREFIX: Final = "partition-free-v1-signed-work-device-shell-plan-"
_DESIGN_PREFIX: Final = "partition-free-v1-signed-work-device-shell-design-"
_ROUNDING_PREFIX: Final = "partition-free-v1-signed-work-device-shell-rounding-"
_BRANCH_PREFIX: Final = "partition-free-v1-signed-work-device-shell-branch-"
_EXECUTION_PREFIX: Final = "partition-free-v1-signed-work-device-shell-execution-"
_FAMILY_PREFIX: Final = "partition-free-v1-signed-work-device-shell-family-"
_HBM_PREFIX: Final = "partition-free-v1-signed-work-device-shell-hbm-"
_F64_BYTES: Final = np.dtype(np.float64).itemsize
_I32_BYTES: Final = np.dtype(np.int32).itemsize
_I64_BYTES: Final = np.dtype(np.int64).itemsize
_U8_BYTES: Final = np.dtype(np.uint8).itemsize
_BOOL_BYTES: Final = np.dtype(np.bool_).itemsize


class V1SignedWorkDeviceShellRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    AXIS_MISMATCH = "axis_mismatch"
    ENDPOINT_LIMIT_EXCEEDED = "endpoint_limit_exceeded"
    ENDPOINT_PLAN_NOT_PRE_C = "endpoint_plan_not_pre_c"
    NO_RELAXED_STATE = "no_relaxed_state"
    BRANCH_UNAVAILABLE = "branch_unavailable"
    CUDA_REQUIRED = "cuda_required"
    DETERMINISM_REQUIRED = "determinism_required"
    NONFINITE = "nonfinite"
    INFEASIBLE_ENDPOINT_PLAN = "infeasible_endpoint_plan"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    STALE_SOURCE = "stale_source"
    AUTHORITY_VIOLATION = "authority_violation"
    LABEL_FREEZE_FAILED = "label_freeze_failed"
    ENDPOINT_QUERY_MISMATCH = "endpoint_query_mismatch"


class V1SignedWorkDeviceShellRefusal(RuntimeError):
    def __init__(
        self,
        code: V1SignedWorkDeviceShellRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkDeviceShellRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkDeviceShellRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INVALID_TYPE,
            f"{name} must be one positive finite real",
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.NONFINITE,
            f"{name} must be positive and finite",
        )
    return result


def _authority_flags_are_closed(values: tuple[bool, ...], *, owner: str) -> None:
    if any(type(value) is not bool for value in values):
        raise TypeError(f"{owner} authority fields must be exact booleans")
    if any(values):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
            f"{owner} cannot carry downstream authority",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkDeviceShellEndpointPlanReceipt:
    """Pre-C endpoint counts shared by signed-work device-shell branches.

    The plan is intentionally external to this module.  It records the
    currently pending endogenous-frontier origin rather than inventing a
    powers-of-two, geometric, or otherwise ad hoc endpoint schedule here.
    """

    endpoint_counts: tuple[int, ...]
    route_axis_hash: str
    route_count: int
    source_registration_hash: str
    registered_d_coordinate_hash: str
    endpoint_plan_origin: str = SIGNED_WORK_DEVICE_SHELL_PLAN_ORIGIN
    common_across_q_branches: bool = True
    c_stream_outcomes_seen: bool = False
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_DEVICE_SHELL_AUTHORITY
    schema: str = SIGNED_WORK_DEVICE_SHELL_ENDPOINT_PLAN_SCHEMA

    def __post_init__(self) -> None:
        counts = tuple(
            _positive_int(value, name="endpoint count")
            for value in self.endpoint_counts
        )
        if not counts or tuple(sorted(set(counts))) != counts:
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
                "endpoint counts must be nonempty, unique, and increasing",
            )
        if len(counts) > DEVICE_SHELL_MAXIMUM_ENDPOINTS:
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.ENDPOINT_LIMIT_EXCEEDED,
                "uint8 first-inclusion labels support at most 255 endpoints",
            )
        route_count = _positive_int(self.route_count, name="route count")
        if counts[-1] > route_count:
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
                "the largest endpoint exceeds the physical route axis",
            )
        object.__setattr__(self, "endpoint_counts", counts)
        object.__setattr__(
            self,
            "route_axis_hash",
            _identity(self.route_axis_hash, name="route axis hash"),
        )
        object.__setattr__(self, "route_count", route_count)
        for name in ("source_registration_hash", "registered_d_coordinate_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.endpoint_plan_origin != SIGNED_WORK_DEVICE_SHELL_PLAN_ORIGIN:
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.ENDPOINT_PLAN_NOT_PRE_C,
                "endpoint plan origin is not the registered pending frontier seam",
            )
        if type(self.common_across_q_branches) is not bool or (
            not self.common_across_q_branches
        ):
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.ENDPOINT_PLAN_NOT_PRE_C,
                "signed-work shell rounding requires one common endpoint plan",
            )
        _authority_flags_are_closed(
            (
                self.c_stream_outcomes_seen,
                self.fqr_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="endpoint plan",
        )
        if (
            self.authority_scope != SIGNED_WORK_DEVICE_SHELL_AUTHORITY
            or self.schema != SIGNED_WORK_DEVICE_SHELL_ENDPOINT_PLAN_SCHEMA
        ):
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
                "endpoint plan semantics changed",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PLAN_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkDeviceShellExecutionDesign:
    rounding_tile_size: int = 1 << 20
    physical_chunk_size: int = DEVICE_SHELL_DEFAULT_PHYSICAL_CHUNK_SIZE
    hbm_ceiling_fraction: float = 0.85
    production_cuda_required: bool = True
    allow_private_cpu_reference: bool = False
    private_reference_route_limit: int = 4096
    allow_unconstrained_blocks: bool = False
    require_deterministic_algorithms: bool = True
    output_mode: Literal["device_resident_uint8_first_inclusion_v1"] = (
        "device_resident_uint8_first_inclusion_v1"
    )
    schema: str = SIGNED_WORK_DEVICE_SHELL_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rounding_tile_size",
            _positive_int(self.rounding_tile_size, name="rounding tile size"),
        )
        object.__setattr__(
            self,
            "physical_chunk_size",
            _positive_int(self.physical_chunk_size, name="physical chunk size"),
        )
        ceiling = _positive(self.hbm_ceiling_fraction, name="HBM ceiling fraction")
        if ceiling > 0.85:
            raise ValueError("canonical device-shell HBM ceiling cannot exceed 0.85")
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        for name in (
            "production_cuda_required",
            "allow_private_cpu_reference",
            "allow_unconstrained_blocks",
            "require_deterministic_algorithms",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.production_cuda_required and self.allow_private_cpu_reference:
            raise ValueError("production device-shell design cannot enable CPU reference")
        object.__setattr__(
            self,
            "private_reference_route_limit",
            _nonnegative_int(
                self.private_reference_route_limit,
                name="private reference route limit",
            ),
        )
        if self.output_mode != "device_resident_uint8_first_inclusion_v1":
            raise ValueError("unsupported signed-work device-shell output mode")
        if self.schema != SIGNED_WORK_DEVICE_SHELL_DESIGN_SCHEMA:
            raise ValueError("unsupported signed-work device-shell design schema")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_DESIGN_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkDeviceShellHBMForecast:
    problem_hash: str
    policy_hash: str
    execution_hash: str
    endpoint_plan_hash: str
    design_hash: str
    branch: V1RelaxedKKTBranch
    device: str
    route_count: int
    maximum_endpoint_count: int
    rounding_tile_size: int
    resident_ratio_bytes: int
    resident_boolean_label_bytes: int
    bounded_order_bytes: int
    tile_workspace_bytes: int
    forecast_incremental_bytes: int
    current_reserved_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    route_scale_cpu_order_materialized: bool = False
    schema: str = SIGNED_WORK_DEVICE_SHELL_HBM_SCHEMA

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_HBM_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkDeviceShellBranchIdentityReceipt:
    branch: V1RelaxedKKTBranch
    source_rounding_hash: str
    frozen_ladder: V1DeviceShellFrozenLadderReceipt
    signed_work_branch_receipt_hash: str
    endpoint_plan_hash: str
    key_id: str = SIGNED_WORK_DEVICE_SHELL_KEY_ID
    authority_scope: str = SIGNED_WORK_DEVICE_SHELL_AUTHORITY
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SIGNED_WORK_DEVICE_SHELL_BRANCH_SCHEMA

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("branch identity requires an exact relaxed-KKT branch")
        if type(self.frozen_ladder) is not V1DeviceShellFrozenLadderReceipt:
            raise TypeError("branch identity requires an exact frozen shell receipt")
        for name in (
            "source_rounding_hash",
            "signed_work_branch_receipt_hash",
            "endpoint_plan_hash",
            "key_id",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        _authority_flags_are_closed(
            (
                self.fqr_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="signed-work device-shell branch",
        )
        if (
            self.key_id != SIGNED_WORK_DEVICE_SHELL_KEY_ID
            or self.authority_scope != SIGNED_WORK_DEVICE_SHELL_AUTHORITY
            or self.schema != SIGNED_WORK_DEVICE_SHELL_BRANCH_SCHEMA
        ):
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
                "branch identity semantics changed",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_BRANCH_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkDeviceShellExecutionReceipt:
    problem_hash: str
    policy_hash: str
    least_work_execution_hash: str
    state_identity_hash: str
    endpoint_plan_hash: str
    design_hash: str
    parent_mask_hash: str
    source_geometry_hash: str
    source_registration_hash: str
    registered_d_coordinate_hash: str
    selected_branch: V1RelaxedKKTBranch
    branch_identity: V1SignedWorkDeviceShellBranchIdentityReceipt
    hbm_forecast_hash: str
    required_hard_action_hash: str | None
    route_scale_cpu_order_materialized: bool = False
    additive_f_replay_performed: bool = False
    key_id: str = SIGNED_WORK_DEVICE_SHELL_KEY_ID
    authority_scope: str = SIGNED_WORK_DEVICE_SHELL_AUTHORITY
    fqr_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SIGNED_WORK_DEVICE_SHELL_EXECUTION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "problem_hash",
            "policy_hash",
            "least_work_execution_hash",
            "state_identity_hash",
            "endpoint_plan_hash",
            "design_hash",
            "parent_mask_hash",
            "source_geometry_hash",
            "source_registration_hash",
            "registered_d_coordinate_hash",
            "hbm_forecast_hash",
            "key_id",
            "authority_scope",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identity(value, name=name))
        if type(self.selected_branch) is not V1RelaxedKKTBranch:
            raise TypeError("execution receipt requires an exact branch")
        if type(self.branch_identity) is not V1SignedWorkDeviceShellBranchIdentityReceipt:
            raise TypeError("execution receipt requires exact branch identity")
        if self.branch_identity.branch is not self.selected_branch:
            raise ValueError("branch identity and execution branch differ")
        if self.branch_identity.endpoint_plan_hash != self.endpoint_plan_hash:
            raise ValueError("branch identity and execution endpoint plan differ")
        if self.branch_identity.frozen_ladder.parent_mask_hash != self.parent_mask_hash:
            raise ValueError("frozen ladder and execution parent mask differ")
        if type(self.route_scale_cpu_order_materialized) is not bool or (
            self.route_scale_cpu_order_materialized
        ):
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
                "signed-work device shell may not materialize route-scale CPU order",
            )
        if type(self.additive_f_replay_performed) is not bool or (
            self.additive_f_replay_performed
        ):
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
                "signed-work device shell may not replay additive F",
            )
        _authority_flags_are_closed(
            (
                self.fqr_authority,
                self.c_stream_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="signed-work device-shell execution",
        )
        if (
            self.key_id != SIGNED_WORK_DEVICE_SHELL_KEY_ID
            or self.authority_scope != SIGNED_WORK_DEVICE_SHELL_AUTHORITY
            or self.schema != SIGNED_WORK_DEVICE_SHELL_EXECUTION_SCHEMA
        ):
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
                "execution receipt semantics changed",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_EXECUTION_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkDeviceShellExecutionBundle:
    label_handle: V1DeviceShellLabelHandle = field(repr=False, compare=False)
    execution_receipt: V1SignedWorkDeviceShellExecutionReceipt
    hbm_forecast: V1SignedWorkDeviceShellHBMForecast
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    schema: str = SIGNED_WORK_DEVICE_SHELL_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.label_handle) is not V1DeviceShellLabelHandle:
            raise TypeError("signed-work device-shell bundle requires exact labels")
        if type(self.execution_receipt) is not V1SignedWorkDeviceShellExecutionReceipt:
            raise TypeError("signed-work device-shell bundle requires exact receipt")
        if type(self.hbm_forecast) is not V1SignedWorkDeviceShellHBMForecast:
            raise TypeError("signed-work device-shell bundle requires exact forecast")
        if self.hbm_forecast.forecast_hash != self.execution_receipt.hbm_forecast_hash:
            raise ValueError("device-shell forecast identity differs from receipt")
        if self.production_cuda_execution == self.private_cpu_reference_execution:
            raise ValueError("exactly one device-shell execution mode must be active")
        if self.schema != SIGNED_WORK_DEVICE_SHELL_BUNDLE_SCHEMA:
            raise ValueError("unsupported signed-work device-shell bundle schema")

    @property
    def frozen_ladder(self) -> V1DeviceShellFrozenLadderReceipt:
        return self.execution_receipt.branch_identity.frozen_ladder

    @property
    def branch_identity(
        self,
    ) -> V1SignedWorkDeviceShellBranchIdentityReceipt:
        return self.execution_receipt.branch_identity

    @property
    def execution_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "execution_receipt_hash": self.execution_receipt.receipt_hash,
                "hbm_forecast_hash": self.hbm_forecast.forecast_hash,
                "production_cuda_execution": self.production_cuda_execution,
                "private_cpu_reference_execution": self.private_cpu_reference_execution,
            },
            prefix=_EXECUTION_PREFIX + "bundle-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkDeviceShellFamilyReceipt:
    """Both independent Q-branch ladders frozen before any C outcome."""

    problem_hash: str
    policy_hash: str
    least_work_execution_hash: str
    endpoint_plan_hash: str
    design_hash: str
    parent_mask_hash: str
    spectral_execution_hash: str
    diagonal_l1_execution_hash: str
    both_q_branches_frozen: bool = True
    c_stream_outcomes_seen: bool = False
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_DEVICE_SHELL_AUTHORITY
    schema: str = SIGNED_WORK_DEVICE_SHELL_FAMILY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "problem_hash",
            "policy_hash",
            "least_work_execution_hash",
            "endpoint_plan_hash",
            "design_hash",
            "parent_mask_hash",
            "spectral_execution_hash",
            "diagonal_l1_execution_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.both_q_branches_frozen) is not bool or not self.both_q_branches_frozen:
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
                "signed-work shell family must retain both Q branches",
            )
        _authority_flags_are_closed(
            (
                self.c_stream_outcomes_seen,
                self.fqr_authority,
                self.calibration_authority,
                self.oracle_authority,
                self.mask_commit_authority,
                self.terminal_authority,
                self.pareto_authority,
            ),
            owner="signed-work device-shell family",
        )
        if (
            self.authority_scope != SIGNED_WORK_DEVICE_SHELL_AUTHORITY
            or self.schema != SIGNED_WORK_DEVICE_SHELL_FAMILY_SCHEMA
        ):
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
                "signed-work shell-family semantics changed",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_FAMILY_PREFIX)


@dataclass(frozen=True, slots=True)
class V1SignedWorkDeviceShellFamilyBundle:
    spectral: V1SignedWorkDeviceShellExecutionBundle
    diagonal_l1: V1SignedWorkDeviceShellExecutionBundle
    receipt: V1SignedWorkDeviceShellFamilyReceipt
    schema: str = SIGNED_WORK_DEVICE_SHELL_FAMILY_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.spectral) is not V1SignedWorkDeviceShellExecutionBundle
            or type(self.diagonal_l1) is not V1SignedWorkDeviceShellExecutionBundle
            or type(self.receipt) is not V1SignedWorkDeviceShellFamilyReceipt
        ):
            raise TypeError("signed-work shell family requires exact branch bundles")
        spectral_receipt = self.spectral.execution_receipt
        diagonal_receipt = self.diagonal_l1.execution_receipt
        if (
            spectral_receipt.selected_branch is not V1RelaxedKKTBranch.SPECTRAL
            or diagonal_receipt.selected_branch
            is not V1RelaxedKKTBranch.DIAGONAL_L1
            or self.receipt.spectral_execution_hash != self.spectral.execution_hash
            or self.receipt.diagonal_l1_execution_hash
            != self.diagonal_l1.execution_hash
            or spectral_receipt.problem_hash != self.receipt.problem_hash
            or diagonal_receipt.problem_hash != self.receipt.problem_hash
            or spectral_receipt.policy_hash != self.receipt.policy_hash
            or diagonal_receipt.policy_hash != self.receipt.policy_hash
            or spectral_receipt.least_work_execution_hash
            != self.receipt.least_work_execution_hash
            or diagonal_receipt.least_work_execution_hash
            != self.receipt.least_work_execution_hash
            or spectral_receipt.endpoint_plan_hash
            != self.receipt.endpoint_plan_hash
            or diagonal_receipt.endpoint_plan_hash
            != self.receipt.endpoint_plan_hash
            or spectral_receipt.design_hash != self.receipt.design_hash
            or diagonal_receipt.design_hash != self.receipt.design_hash
            or spectral_receipt.parent_mask_hash != self.receipt.parent_mask_hash
            or diagonal_receipt.parent_mask_hash != self.receipt.parent_mask_hash
            or self.schema != SIGNED_WORK_DEVICE_SHELL_FAMILY_SCHEMA
        ):
            raise ValueError("signed-work shell family branch identity is inconsistent")

    @property
    def family_hash(self) -> str:
        return self.receipt.receipt_hash

    def evict_labels_atomically(self) -> None:
        V1DeviceShellLabelHandle.evict_all_atomically(
            (self.spectral.label_handle, self.diagonal_l1.label_handle)
        )


def _branch_receipt_map(
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
) -> dict[V1RelaxedKKTBranch, V1SignedWorkLeastWorkCUDABranchReceipt]:
    receipts = execution.branch_receipts
    if any(type(receipt) is not V1SignedWorkLeastWorkCUDABranchReceipt for receipt in receipts):
        raise TypeError("signed-work execution carries a malformed branch receipt")
    return {receipt.branch: receipt for receipt in receipts}


def _selected_branch(
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    branch: V1RelaxedKKTBranch | None,
) -> tuple[V1RelaxedKKTBranch, V1SignedWorkLeastWorkCUDABranchReceipt]:
    branch_map = _branch_receipt_map(execution)
    selected = execution.selected_branch if branch is None else branch
    if type(selected) is not V1RelaxedKKTBranch:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.BRANCH_UNAVAILABLE,
            "no certified relaxed branch is available for hard rounding",
        )
    receipt = branch_map.get(selected)
    if receipt is None or not receipt.converged:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.BRANCH_UNAVAILABLE,
            "selected branch does not have a converged signed-work receipt",
        )
    return selected, receipt


def _parent_live_numpy(parent_mask: HardMask) -> np.ndarray:
    result = parent_mask.to_bool_array()
    result.setflags(write=False)
    return result


def _required_numpy(problem: V1SignedWorkLeastWorkCUDAProblem) -> np.ndarray:
    if problem.required_hard_action is None:
        result = np.zeros(problem.source_problem.route_count, dtype=np.bool_)
    else:
        result = np.asarray(problem.required_hard_action, dtype=np.bool_).reshape(-1)
    frozen = np.array(result, dtype=np.bool_, order="C", copy=True)
    frozen.setflags(write=False)
    return frozen


def _ratio_numpy(problem: V1SignedWorkLeastWorkCUDAProblem) -> np.ndarray:
    source = problem.source_problem
    result = np.divide(
        source.corrected_work,
        source.resource,
        out=np.zeros(source.route_count, dtype=np.float64),
        where=source.resource > 0.0,
    )
    if not np.all(np.isfinite(result)):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.NONFINITE,
            "corrected-work/resource key coordinate is nonfinite",
        )
    result.setflags(write=False)
    return result


def _validate_static_inputs(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt,
    parent_mask: HardMask,
    design: V1SignedWorkDeviceShellExecutionDesign,
) -> None:
    if type(problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("device-shell rounding requires exact signed-work problem")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("device-shell rounding requires exact signed-work policy")
    if type(execution) is not V1SignedWorkLeastWorkCUDAExecutionBundle:
        raise TypeError("device-shell rounding requires exact signed-work execution")
    if type(endpoint_plan) is not V1SignedWorkDeviceShellEndpointPlanReceipt:
        raise TypeError("device-shell rounding requires exact endpoint plan")
    if type(parent_mask) is not HardMask:
        raise TypeError("device-shell rounding requires exact parent mask")
    if type(design) is not V1SignedWorkDeviceShellExecutionDesign:
        raise TypeError("device-shell rounding requires exact execution design")
    problem.assert_integrity()
    if execution.result.problem_hash != problem.problem_hash:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "least-work execution belongs to a different signed-work problem",
        )
    if execution.result.policy_hash != policy.policy_hash:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "least-work execution belongs to a different signed-work policy",
        )
    if not problem.source_bundle_authoritative:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "signed-work source bundle is not authoritative for rounding",
        )
    source = problem.source_problem
    if (
        endpoint_plan.route_axis_hash != source.route_axis_hash
        or endpoint_plan.route_count != source.route_count
        or endpoint_plan.source_registration_hash != problem.source_registration_hash
        or endpoint_plan.registered_d_coordinate_hash
        != problem.registered_d_coordinate_hash
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "endpoint plan does not bind the signed-work D/source identity",
        )
    if (
        parent_mask.route_axis_hash != source.route_axis_hash
        or parent_mask.route_count != source.route_count
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.AXIS_MISMATCH,
            "parent mask differs from the signed-work physical route axis",
        )
    if (
        execution.hard_mask_global_optimality_claimed
        or execution.mask_commit_authority
        or execution.result.c_stream_authority
        or execution.result.calibration_authority
        or execution.result.bce_oracle_authority
        or execution.result.terminal_authority
        or execution.result.pareto_claim_authority
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
            "upstream signed-work execution crossed proposal-only authority",
        )
    if not execution.relaxed_state_available or execution.relaxed_state_handle is None:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.NO_RELAXED_STATE,
            "device-shell rounding requires retained relaxed theorem state",
        )


def _compact_blocks(
    problem: V1SignedWorkLeastWorkCUDAProblem,
) -> tuple[V1DeviceShellCompactBlockCapacity, ...]:
    return tuple(
        V1DeviceShellCompactBlockCapacity.from_registered_block(block)
        for block in problem.source_problem.block_capacities
    )


def _validate_endpoint_capacity(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    endpoint_counts: tuple[int, ...],
    parent_live: np.ndarray,
    required: np.ndarray,
    design: V1SignedWorkDeviceShellExecutionDesign,
) -> int:
    source = problem.source_problem
    selectable = np.asarray(source.eligible_mask, dtype=np.bool_) & parent_live
    if bool(np.any(required & ~selectable)):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
            "required hard action is not parent-live and eligible",
        )
    required_total = int(np.count_nonzero(required))
    if endpoint_counts[0] < required_total:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
            "the first endpoint cannot contain the required prior hard action",
        )
    if source.block_capacities:
        free_capacity = 0
        for block in source.block_capacities:
            required_block = int(np.count_nonzero(required[block.start : block.stop]))
            remaining = block.removal_capacity - required_block
            if remaining < 0:
                _refuse(
                    V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
                    "required hard action exceeds a compact block capacity",
                )
            selectable_block = int(
                np.count_nonzero(
                    selectable[block.start : block.stop]
                    & ~required[block.start : block.stop]
                )
            )
            free_capacity += min(remaining, selectable_block)
    else:
        if not design.allow_unconstrained_blocks:
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
                "compact block capacities are required unless reference use is explicit",
            )
        free_capacity = int(np.count_nonzero(selectable & ~required))
    requested_free = endpoint_counts[-1] - required_total
    if requested_free < 0 or requested_free > free_capacity:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
            "endpoint plan exceeds selectable routes after required/capacity deduction",
        )
    return required_total


def estimate_v1_signed_work_device_shell_cuda_hbm(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt,
    design: V1SignedWorkDeviceShellExecutionDesign,
    branch: V1RelaxedKKTBranch | None = None,
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1SignedWorkDeviceShellHBMForecast:
    """Forecast bridge HBM before allocating route-resident rounding buffers."""

    if type(problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("device-shell HBM forecast requires exact signed-work problem")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("device-shell HBM forecast requires exact signed-work policy")
    if type(execution) is not V1SignedWorkLeastWorkCUDAExecutionBundle:
        raise TypeError("device-shell HBM forecast requires exact signed-work execution")
    if type(endpoint_plan) is not V1SignedWorkDeviceShellEndpointPlanReceipt:
        raise TypeError("device-shell HBM forecast requires exact endpoint plan")
    if type(design) is not V1SignedWorkDeviceShellExecutionDesign:
        raise TypeError("device-shell HBM forecast requires exact execution design")
    problem.assert_integrity()
    if (
        execution.result.problem_hash != problem.problem_hash
        or execution.result.policy_hash != policy.policy_hash
        or endpoint_plan.route_axis_hash != problem.source_problem.route_axis_hash
        or endpoint_plan.route_count != problem.source_problem.route_count
        or endpoint_plan.source_registration_hash != problem.source_registration_hash
        or endpoint_plan.registered_d_coordinate_hash
        != problem.registered_d_coordinate_hash
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "device-shell HBM forecast identities do not share one signed-work source",
        )
    selected, _ = _selected_branch(execution, branch)
    state = execution.relaxed_state_handle
    if state is None:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.NO_RELAXED_STATE,
            "device-shell HBM forecast requires retained theorem state",
        )
    route_count = problem.source_problem.route_count
    maximum_endpoint = endpoint_plan.endpoint_counts[-1]
    tile = design.rounding_tile_size
    resident_ratio = route_count * _F64_BYTES
    resident_boolean_label = route_count * (
        _U8_BYTES + 3 * _BOOL_BYTES
    )
    bounded_order = (
        maximum_endpoint * _I64_BYTES
        + 2 * min(maximum_endpoint, tile) * _I64_BYTES
        + 2 * maximum_endpoint * _I64_BYTES
    )
    tile_workspace = tile * (
        4 * _I64_BYTES + 4 * _F64_BYTES + 2 * _BOOL_BYTES
    )
    incremental = resident_ratio + resident_boolean_label + bounded_order + tile_workspace
    device = state.device
    admission_memory = None
    if device_total_memory_bytes is None or current_reserved_bytes is None:
        if device.startswith("cuda"):
            if torch is None or not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkDeviceShellRefusalCode.CUDA_REQUIRED,
                    "CUDA memory forecast requested without an available CUDA runtime",
                )
            torch_device = torch.device(device)
            admission_memory = (cuda_reusable_memory_bytes(torch_device, framework=torch), int(torch.cuda.memory_allocated(torch_device)))
            if device_total_memory_bytes is None:
                device_total_memory_bytes = int(
                    torch.cuda.get_device_properties(torch_device).total_memory
                )
            if current_reserved_bytes is None:
                current_reserved_bytes = int(torch.cuda.memory_reserved(torch_device))
        else:
            if device_total_memory_bytes is None:
                device_total_memory_bytes = max(1 << 30, 4 * incremental)
            if current_reserved_bytes is None:
                current_reserved_bytes = 0
    total = _positive_int(device_total_memory_bytes, name="device total memory")
    current = _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
    ceiling = int(math.floor(design.hbm_ceiling_fraction * total))
    peak = current + incremental
    available, allocated = admission_memory or (total, current)
    return V1SignedWorkDeviceShellHBMForecast(
        problem_hash=problem.problem_hash,
        policy_hash=execution.result.policy_hash,
        execution_hash=execution.execution_hash,
        endpoint_plan_hash=endpoint_plan.receipt_hash,
        design_hash=design.design_hash,
        branch=selected,
        device=device,
        route_count=route_count,
        maximum_endpoint_count=maximum_endpoint,
        rounding_tile_size=tile,
        resident_ratio_bytes=resident_ratio,
        resident_boolean_label_bytes=resident_boolean_label,
        bounded_order_bytes=bounded_order,
        tile_workspace_bytes=tile_workspace,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=allocated + incremental <= ceiling and incremental <= available,
    )


def _cpu_lexicographic_order(
    *,
    relaxed: np.ndarray,
    reduced: np.ndarray,
    ratio: np.ndarray,
    candidate: np.ndarray,
    maximum_count: int,
    block_capacities: tuple[Any, ...],
    required: np.ndarray,
) -> np.ndarray:
    physical = np.arange(relaxed.size, dtype=np.int64)
    order = np.lexsort((physical, ratio, -reduced, -relaxed))
    if block_capacities:
        remaining = {
            block.block_id: block.removal_capacity
            - int(np.count_nonzero(required[block.start : block.stop]))
            for block in block_capacities
        }
        block_for_route = np.empty(relaxed.size, dtype=object)
        for block in block_capacities:
            block_for_route[block.start : block.stop] = block.block_id
        accepted: list[int] = []
        for item in order:
            route = int(item)
            if not candidate[route]:
                continue
            block_id = block_for_route[route]
            if remaining[block_id] <= 0:
                continue
            remaining[block_id] -= 1
            accepted.append(route)
            if len(accepted) == maximum_count:
                break
        result = np.asarray(accepted, dtype=np.int64)
    else:
        result = order[candidate[order]][:maximum_count].astype(np.int64, copy=False)
    result.setflags(write=False)
    return result


def _cpu_first_inclusion_labels(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    endpoint_counts: tuple[int, ...],
    parent_live: np.ndarray,
    required: np.ndarray,
    relaxed: np.ndarray,
    reduced: np.ndarray,
    ratio: np.ndarray,
) -> np.ndarray:
    if relaxed.size > 4096:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
            "private CPU label reference is limited to tiny focused tests",
        )
    source = problem.source_problem
    candidate = np.asarray(source.eligible_mask, dtype=np.bool_) & parent_live & ~required
    maximum_free = endpoint_counts[-1] - int(np.count_nonzero(required))
    ordered = _cpu_lexicographic_order(
        relaxed=relaxed,
        reduced=reduced,
        ratio=ratio,
        candidate=candidate,
        maximum_count=maximum_free,
        block_capacities=source.block_capacities,
        required=required,
    )
    if int(ordered.size) < maximum_free:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
            "private CPU order cannot cover the endpoint plan",
        )
    labels = np.zeros(relaxed.size, dtype=np.uint8)
    if bool(np.any(required)):
        labels[required] = 1
    required_total = int(np.count_nonzero(required))
    previous = 0
    for endpoint_label, count in enumerate(endpoint_counts, start=1):
        stop = count - required_total
        if stop < previous:
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
                "endpoint plan is not increasing after required deduction",
            )
        if stop > previous:
            labels[ordered[previous:stop]] = endpoint_label
        previous = stop
    labels.setflags(write=False)
    return labels


def _cpu_prefix_ranks(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    maximum_free: int,
    parent_live: np.ndarray,
    required: np.ndarray,
    relaxed: np.ndarray,
    reduced: np.ndarray,
    ratio: np.ndarray,
) -> np.ndarray:
    if relaxed.size > 4096:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
            "private CPU prefix-rank reference is limited to tiny focused tests",
        )
    if maximum_free > np.iinfo(np.uint32).max:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
            "prefix path exceeds uint32 rank custody",
        )
    source = problem.source_problem
    candidate = np.asarray(source.eligible_mask, dtype=np.bool_) & parent_live & ~required
    ordered = _cpu_lexicographic_order(
        relaxed=relaxed,
        reduced=reduced,
        ratio=ratio,
        candidate=candidate,
        maximum_count=maximum_free,
        block_capacities=source.block_capacities,
        required=required,
    )
    if int(ordered.size) < maximum_free:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
            "private CPU order cannot cover the prefix-rank path",
        )
    ranks = np.zeros(relaxed.size, dtype=np.uint32)
    if maximum_free:
        ranks[ordered[:maximum_free]] = np.arange(
            1,
            maximum_free + 1,
            dtype=np.uint32,
        )
    ranks.setflags(write=False)
    return ranks


def _stable_lexicographic_order_from_indices_torch(
    *,
    indices: Any,
    relaxed: Any,
    reduced: Any,
    ratio: Any,
    target: int,
) -> Any:
    if int(indices.numel()) <= 0 or target <= 0:
        return indices[:0]
    limit = min(_positive_int(target, name="target"), int(indices.numel()))
    ordered = indices
    _, permutation = torch.sort(ordered, stable=True)
    ordered = ordered[permutation]
    # Reindex every less-to-more-significant coordinate *after* the previous
    # stable sort.  Precomputing this tuple would bind later values to the old
    # order and silently break the registered lexicographic key.
    for coordinate in (ratio, -reduced, -relaxed):
        values = coordinate[ordered]
        _, permutation = torch.sort(values, stable=True)
        ordered = ordered[permutation]
    return ordered[:limit].contiguous()


def _lexicographic_topk_from_indices_torch(
    *,
    indices: Any,
    relaxed: Any,
    reduced: Any,
    ratio: Any,
    target: int,
) -> Any:
    """Select the exact lexicographic top-k with full sorting only on k rows.

    The canonical key is ``(-relaxed, -reduced, ratio, physical_index)``.  A
    route-scale production call should not stable-sort every candidate merely
    to keep a small prefix.  This routine performs exact threshold selection
    one key coordinate at a time, then applies the established stable sort only
    to the retained top-k set so the emitted order is byte-for-byte compatible
    with the reference key.
    """

    if int(indices.numel()) <= 0 or target <= 0:
        return indices[:0]
    limit = min(_positive_int(target, name="target"), int(indices.numel()))
    if limit == int(indices.numel()):
        return _stable_lexicographic_order_from_indices_torch(
            indices=indices,
            relaxed=relaxed,
            reduced=reduced,
            ratio=ratio,
            target=target,
        )

    active = indices
    need = limit
    selected: list[Any] = []
    for coordinate, largest in (
        (relaxed, True),
        (reduced, True),
        (ratio, False),
    ):
        values = coordinate[active]
        if not bool(torch.isfinite(values).all().item()):
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.NONFINITE,
                "lexicographic CUDA key contains a nonfinite value",
            )
        boundary_values = torch.topk(
            values,
            k=need,
            largest=largest,
            sorted=False,
        ).values
        boundary = (
            torch.min(boundary_values)
            if largest
            else torch.max(boundary_values)
        )
        better = values > boundary if largest else values < boundary
        better_count = int(torch.count_nonzero(better).item())
        if better_count:
            better_indices = active[better]
            selected.append(better_indices)
            need -= better_count
            if need <= 0:  # pragma: no cover - threshold construction forbids this.
                break
        active = active[values == boundary]
        if need == 0:
            break
        if int(active.numel()) <= need:
            selected.append(active)
            need -= int(active.numel())
            break

    if need > 0:
        # The last key is the physical index.  The surviving tie set is already
        # equal on all floating coordinates, so the smallest physical indices
        # complete the exact lexicographic prefix.
        if int(active.numel()) > need:
            selected.append(
                torch.topk(active, k=need, largest=False, sorted=False).values
            )
        else:
            selected.append(active)

    retained = torch.cat(tuple(selected), dim=0) if selected else indices[:0]
    if int(retained.numel()) != limit:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.LABEL_FREEZE_FAILED,
            "lexicographic CUDA multiselect did not retain the requested prefix",
        )
    return _stable_lexicographic_order_from_indices_torch(
        indices=retained,
        relaxed=relaxed,
        reduced=reduced,
        ratio=ratio,
        target=limit,
    )


def _tiled_lexicographic_topk_torch(
    *,
    relaxed: Any,
    reduced: Any,
    ratio: Any,
    candidate: Any,
    start: int,
    stop: int,
    target: int,
    tile_size: int,
) -> Any:
    if target <= 0:
        return torch.empty(0, dtype=torch.int64, device=relaxed.device)
    accepted = torch.empty(0, dtype=torch.int64, device=relaxed.device)
    for begin in range(start, stop, tile_size):
        end = min(stop, begin + tile_size)
        relative = torch.nonzero(candidate[begin:end], as_tuple=False).reshape(-1)
        if int(relative.numel()) <= 0:
            continue
        active = (relative + begin).to(dtype=torch.int64)
        tile_winners = _lexicographic_topk_from_indices_torch(
            indices=active,
            relaxed=relaxed,
            reduced=reduced,
            ratio=ratio,
            target=target,
        )
        merged = torch.cat((accepted, tile_winners), dim=0)
        accepted = _lexicographic_topk_from_indices_torch(
            indices=merged,
            relaxed=relaxed,
            reduced=reduced,
            ratio=ratio,
            target=target,
        )
    return accepted


def _blockwise_lexicographic_topk_torch(
    *,
    source: Any,
    required: np.ndarray,
    relaxed: Any,
    reduced: Any,
    ratio: Any,
    candidate: Any,
    target: int,
    tile_size: int,
) -> Any:
    """Select the exact global prefix after compact block-capacity pruning.

    Each compact block contributes only routes that could legally participate
    under its one-live-survivor capacity.  The expensive global key refinement
    is then applied once to the union, instead of repeatedly stable-sorting a
    growing route-scale prefix after every block.
    """

    if target <= 0:
        return torch.empty(0, dtype=torch.int64, device=relaxed.device)
    block_candidates: list[Any] = []
    for block in source.block_capacities:
        required_block = int(np.count_nonzero(required[block.start : block.stop]))
        block_target = min(target, block.removal_capacity - required_block)
        if block_target <= 0:
            continue
        block_winners = _tiled_lexicographic_topk_torch(
            relaxed=relaxed,
            reduced=reduced,
            ratio=ratio,
            candidate=candidate,
            start=block.start,
            stop=block.stop,
            target=block_target,
            tile_size=tile_size,
        )
        if int(block_winners.numel()):
            block_candidates.append(block_winners)
    if not block_candidates:
        return torch.empty(0, dtype=torch.int64, device=relaxed.device)
    merged = torch.cat(tuple(block_candidates), dim=0)
    return _lexicographic_topk_from_indices_torch(
        indices=merged,
        relaxed=relaxed,
        reduced=reduced,
        ratio=ratio,
        target=target,
    )


def _torch_first_inclusion_labels(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    endpoint_counts: tuple[int, ...],
    parent_live: np.ndarray,
    required: np.ndarray,
    relaxed: Any,
    reduced: Any,
    ratio: Any,
    eligible: Any,
    required_t: Any,
    tile_size: int,
) -> Any:
    device = relaxed.device
    route_count = problem.source_problem.route_count
    parent = torch.tensor(parent_live, dtype=torch.bool, device=device).contiguous()
    if (
        ratio.dtype != torch.float64
        or eligible.dtype != torch.bool
        or required_t.dtype != torch.bool
        or tuple(ratio.shape) != (route_count,)
        or tuple(eligible.shape) != (route_count,)
        or tuple(required_t.shape) != (route_count,)
        or ratio.device != device
        or eligible.device != device
        or required_t.device != device
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "retained device-shell source tensors do not match the route axis",
        )
    candidate = eligible & parent & ~required_t
    required_total = int(np.count_nonzero(required))
    maximum_free = endpoint_counts[-1] - required_total
    source = problem.source_problem
    if source.block_capacities:
        accepted = _blockwise_lexicographic_topk_torch(
            source=source,
            required=required,
            relaxed=relaxed,
            reduced=reduced,
            ratio=ratio,
            candidate=candidate,
            target=maximum_free,
            tile_size=tile_size,
        )
    else:
        accepted = _tiled_lexicographic_topk_torch(
            relaxed=relaxed,
            reduced=reduced,
            ratio=ratio,
            candidate=candidate,
            start=0,
            stop=route_count,
            target=maximum_free,
            tile_size=tile_size,
        )
    if int(accepted.numel()) < maximum_free:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
            "CUDA order cannot cover the endpoint plan",
        )
    labels = torch.zeros(route_count, dtype=torch.uint8, device=device)
    if required_total:
        labels[required_t] = 1
    previous = 0
    for endpoint_label, count in enumerate(endpoint_counts, start=1):
        stop = count - required_total
        if stop > previous:
            labels[accepted[previous:stop]] = endpoint_label
        previous = stop
    return labels.contiguous()


def _torch_prefix_ranks(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    maximum_free: int,
    parent_live: np.ndarray,
    required: np.ndarray,
    relaxed: Any,
    reduced: Any,
    ratio: Any,
    eligible: Any,
    required_t: Any,
    tile_size: int,
) -> Any:
    if maximum_free > np.iinfo(np.int32).max:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
            "logical uint32 prefix path exceeds canonical CUDA int32 storage",
        )
    device = relaxed.device
    route_count = problem.source_problem.route_count
    parent = torch.tensor(parent_live, dtype=torch.bool, device=device).contiguous()
    if (
        ratio.dtype != torch.float64
        or eligible.dtype != torch.bool
        or required_t.dtype != torch.bool
        or tuple(ratio.shape) != (route_count,)
        or tuple(eligible.shape) != (route_count,)
        or tuple(required_t.shape) != (route_count,)
        or ratio.device != device
        or eligible.device != device
        or required_t.device != device
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "retained device-shell source tensors do not match the route axis",
        )
    candidate = eligible & parent & ~required_t
    source = problem.source_problem
    if source.block_capacities:
        accepted = _blockwise_lexicographic_topk_torch(
            source=source,
            required=required,
            relaxed=relaxed,
            reduced=reduced,
            ratio=ratio,
            candidate=candidate,
            target=maximum_free,
            tile_size=tile_size,
        )
    else:
        accepted = _tiled_lexicographic_topk_torch(
            relaxed=relaxed,
            reduced=reduced,
            ratio=ratio,
            candidate=candidate,
            start=0,
            stop=route_count,
            target=maximum_free,
            tile_size=tile_size,
        )
    if int(accepted.numel()) < maximum_free:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
            "CUDA order cannot cover the prefix-rank path",
        )
    ranks = torch.zeros(route_count, dtype=torch.int32, device=device)
    if maximum_free:
        ranks[accepted[:maximum_free]] = torch.arange(
            1,
            maximum_free + 1,
            dtype=torch.int32,
            device=device,
        )
    return ranks.contiguous()


def _source_rounding_hash(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt,
    design: V1SignedWorkDeviceShellExecutionDesign,
    parent_mask: HardMask,
    branch: V1RelaxedKKTBranch,
    branch_receipt: V1SignedWorkLeastWorkCUDABranchReceipt,
) -> str:
    required_hash = (
        None
        if problem.required_hard_action is None
        else content_hash(problem.required_hard_action)
    )
    return content_hash(
        {
            "schema": "partition_free_v1_signed_work_device_shell_rounding_source_v1",
            "problem_hash": problem.problem_hash,
            "policy_hash": policy.policy_hash,
            "least_work_execution_hash": execution.execution_hash,
            "state_identity_hash": execution.state_identity_hash,
            "branch": branch,
            "branch_receipt_hash": branch_receipt.receipt_hash,
            "endpoint_plan_hash": endpoint_plan.receipt_hash,
            "parent_mask_hash": parent_mask.mask_hash,
            "source_geometry_hash": problem.source_problem.source_geometry_hash,
            "signed_work_operator_receipt_hash": (
                problem.signed_work_operator.receipt_hash
            ),
            "source_registration_hash": problem.source_registration_hash,
            "registered_d_coordinate_hash": problem.registered_d_coordinate_hash,
            "required_hard_action_hash": required_hash,
            "key_id": SIGNED_WORK_DEVICE_SHELL_KEY_ID,
            "endpoint_plan_origin": endpoint_plan.endpoint_plan_origin,
            "authority_scope": SIGNED_WORK_DEVICE_SHELL_AUTHORITY,
        },
        prefix=_ROUNDING_PREFIX,
    )


def estimate_v1_signed_work_device_shell_single_branch_hbm(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt,
    design: V1SignedWorkDeviceShellExecutionDesign,
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1SignedWorkDeviceShellHBMForecast:
    """Forecast one Phase-A-supported branch without inventing its complement."""

    if type(problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("single-branch shell forecast requires exact problem")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("single-branch shell forecast requires exact policy")
    if type(execution) is not V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle:
        raise TypeError("single-branch shell forecast requires exact execution")
    if type(endpoint_plan) is not V1SignedWorkDeviceShellEndpointPlanReceipt:
        raise TypeError("single-branch shell forecast requires exact endpoint plan")
    if type(design) is not V1SignedWorkDeviceShellExecutionDesign:
        raise TypeError("single-branch shell forecast requires exact design")
    problem.assert_integrity()
    if (
        execution.result.problem_hash != problem.problem_hash
        or execution.result.policy_hash != policy.policy_hash
        or endpoint_plan.route_axis_hash != problem.source_problem.route_axis_hash
        or endpoint_plan.route_count != problem.source_problem.route_count
        or endpoint_plan.source_registration_hash != problem.source_registration_hash
        or endpoint_plan.registered_d_coordinate_hash
        != problem.registered_d_coordinate_hash
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "single-branch shell forecast identities do not share one source",
        )
    state = execution.state_handle
    if state is None or not execution.result.branch_converged:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.NO_RELAXED_STATE,
            "single-branch shell forecast requires converged retained state",
        )
    route_count = problem.source_problem.route_count
    maximum_endpoint = endpoint_plan.endpoint_counts[-1]
    tile = design.rounding_tile_size
    resident_ratio = route_count * _F64_BYTES
    resident_boolean_label = route_count * (_U8_BYTES + 3 * _BOOL_BYTES)
    bounded_order = (
        maximum_endpoint * _I64_BYTES
        + 2 * min(maximum_endpoint, tile) * _I64_BYTES
        + 2 * maximum_endpoint * _I64_BYTES
    )
    tile_workspace = tile * (
        4 * _I64_BYTES + 4 * _F64_BYTES + 2 * _BOOL_BYTES
    )
    incremental = resident_ratio + resident_boolean_label + bounded_order + tile_workspace
    device = state.device
    admission_memory = None
    if device_total_memory_bytes is None or current_reserved_bytes is None:
        if device.startswith("cuda"):
            if torch is None or not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkDeviceShellRefusalCode.CUDA_REQUIRED,
                    "CUDA shell forecast requested without CUDA",
                )
            torch_device = torch.device(device)
            admission_memory = (cuda_reusable_memory_bytes(torch_device, framework=torch), int(torch.cuda.memory_allocated(torch_device)))
            if device_total_memory_bytes is None:
                device_total_memory_bytes = int(
                    torch.cuda.get_device_properties(torch_device).total_memory
                )
            if current_reserved_bytes is None:
                current_reserved_bytes = int(torch.cuda.memory_reserved(torch_device))
        else:
            if device_total_memory_bytes is None:
                device_total_memory_bytes = max(1 << 30, 4 * incremental)
            if current_reserved_bytes is None:
                current_reserved_bytes = 0
    total = _positive_int(device_total_memory_bytes, name="device total memory")
    current = _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
    ceiling = int(math.floor(design.hbm_ceiling_fraction * total))
    peak = current + incremental
    available, allocated = admission_memory or (total, current)
    return V1SignedWorkDeviceShellHBMForecast(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        execution_hash=execution.execution_hash,
        endpoint_plan_hash=endpoint_plan.receipt_hash,
        design_hash=design.design_hash,
        branch=execution.branch,
        device=device,
        route_count=route_count,
        maximum_endpoint_count=maximum_endpoint,
        rounding_tile_size=tile,
        resident_ratio_bytes=resident_ratio,
        resident_boolean_label_bytes=resident_boolean_label,
        bounded_order_bytes=bounded_order,
        tile_workspace_bytes=tile_workspace,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=allocated + incremental <= ceiling and incremental <= available,
    )


def _single_branch_source_rounding_hash(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt,
    design: V1SignedWorkDeviceShellExecutionDesign,
    parent_mask: HardMask,
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_signed_work_device_shell_rounding_source_v1",
            "problem_hash": problem.problem_hash,
            "policy_hash": policy.policy_hash,
            "least_work_execution_hash": execution.execution_hash,
            "state_identity_hash": execution.state_identity_hash,
            "branch": execution.branch,
            "branch_receipt_hash": execution.branch_receipt.receipt_hash,
            "endpoint_plan_hash": endpoint_plan.receipt_hash,
            "parent_mask_hash": parent_mask.mask_hash,
            "source_geometry_hash": problem.source_problem.source_geometry_hash,
            "signed_work_operator_receipt_hash": (
                problem.signed_work_operator.receipt_hash
            ),
            "source_registration_hash": problem.source_registration_hash,
            "registered_d_coordinate_hash": problem.registered_d_coordinate_hash,
            "required_hard_action_hash": (
                None
                if problem.required_hard_action is None
                else content_hash(problem.required_hard_action)
            ),
            "key_id": SIGNED_WORK_DEVICE_SHELL_KEY_ID,
            "endpoint_plan_origin": endpoint_plan.endpoint_plan_origin,
            "authority_scope": SIGNED_WORK_DEVICE_SHELL_AUTHORITY,
        },
        prefix=_ROUNDING_PREFIX,
    )


def freeze_v1_signed_work_device_shell_single_branch(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle,
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt,
    parent_mask: HardMask,
    design: V1SignedWorkDeviceShellExecutionDesign = (
        V1SignedWorkDeviceShellExecutionDesign()
    ),
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1SignedWorkDeviceShellExecutionBundle:
    """Freeze one explicitly supported Q branch into one hard action ladder."""

    if type(problem) is not V1SignedWorkLeastWorkCUDAProblem:
        raise TypeError("single-branch shell requires exact problem")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("single-branch shell requires exact policy")
    if type(execution) is not V1SignedWorkLeastWorkCUDASingleBranchExecutionBundle:
        raise TypeError("single-branch shell requires exact execution")
    if type(endpoint_plan) is not V1SignedWorkDeviceShellEndpointPlanReceipt:
        raise TypeError("single-branch shell requires exact endpoint plan")
    if type(parent_mask) is not HardMask:
        raise TypeError("single-branch shell requires exact parent mask")
    if type(design) is not V1SignedWorkDeviceShellExecutionDesign:
        raise TypeError("single-branch shell requires exact design")
    problem.assert_integrity()
    source = problem.source_problem
    if (
        execution.result.problem_hash != problem.problem_hash
        or execution.result.policy_hash != policy.policy_hash
        or endpoint_plan.route_axis_hash != source.route_axis_hash
        or endpoint_plan.route_count != source.route_count
        or endpoint_plan.source_registration_hash != problem.source_registration_hash
        or endpoint_plan.registered_d_coordinate_hash
        != problem.registered_d_coordinate_hash
        or parent_mask.route_axis_hash != source.route_axis_hash
        or parent_mask.route_count != source.route_count
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "single-branch problem, plan, execution, and parent do not share one chart",
        )
    if not problem.source_bundle_authoritative:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "single-branch shell requires authoritative source custody",
        )
    if (
        not execution.result.branch_converged
        or execution.state_handle is None
        or execution.result.nonrequested_branch_authority
        or execution.result.c_stream_authority
        or execution.result.calibration_authority
        or execution.result.bce_oracle_authority
        or execution.result.mask_commit_authority
        or execution.result.terminal_authority
        or execution.result.pareto_claim_authority
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.NO_RELAXED_STATE,
            "single-branch execution is unresolved or crossed pre-C authority",
        )
    parent_live = _parent_live_numpy(parent_mask)
    required = _required_numpy(problem)
    _validate_endpoint_capacity(
        problem=problem,
        endpoint_counts=endpoint_plan.endpoint_counts,
        parent_live=parent_live,
        required=required,
        design=design,
    )
    forecast = estimate_v1_signed_work_device_shell_single_branch_hbm(
        problem=problem,
        policy=policy,
        execution=execution,
        endpoint_plan=endpoint_plan,
        design=design,
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.HBM_FORECAST_EXCEEDED,
            "single-branch shell forecast exceeds the HBM ceiling",
        )
    if design.production_cuda_required and not execution.production_cuda_execution:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.CUDA_REQUIRED,
            "production single-branch shell requires CUDA execution",
        )
    if (
        execution.private_cpu_reference_execution
        and not design.allow_private_cpu_reference
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.CUDA_REQUIRED,
            "private single-branch shell was not explicitly enabled",
        )
    if (
        design.require_deterministic_algorithms
        and execution.production_cuda_execution
        and torch is not None
        and not torch.are_deterministic_algorithms_enabled()
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.DETERMINISM_REQUIRED,
            "single-branch CUDA shell requires deterministic torch algorithms",
        )
    source_hash = _single_branch_source_rounding_hash(
        problem=problem,
        policy=policy,
        execution=execution,
        endpoint_plan=endpoint_plan,
        design=design,
        parent_mask=parent_mask,
    )
    state = execution.state_handle
    try:
        with state.borrow_for_device_shell_rounding(
            expected_problem_hash=problem.problem_hash,
            expected_policy_hash=policy.policy_hash,
            expected_branch_receipt_hashes=(
                execution.result.solver_state_binding_hashes
            ),
        ) as lease:
            relaxed = lease.primal(execution.branch)
            reduced = lease.reduced_advantage(execution.branch)
            if execution.private_cpu_reference_execution:
                if source.route_count > design.private_reference_route_limit:
                    _refuse(
                        V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
                        "private single-branch shell is limited to focused tests",
                    )
                labels_np = _cpu_first_inclusion_labels(
                    problem=problem,
                    endpoint_counts=endpoint_plan.endpoint_counts,
                    parent_live=parent_live,
                    required=required,
                    relaxed=relaxed.detach().cpu().numpy().copy(),
                    reduced=reduced.detach().cpu().numpy().copy(),
                    ratio=_ratio_numpy(problem),
                )
                handle = V1DeviceShellLabelHandle.from_numpy_reference(labels_np)
            else:
                resource_t = lease.source_tensor("resource")
                singleton_work_t = lease.source_tensor("singleton_work")
                ratio_t = torch.where(
                    resource_t > 0.0,
                    singleton_work_t / resource_t,
                    torch.zeros_like(resource_t),
                )
                handle = V1DeviceShellLabelHandle.from_torch_cuda(
                    _torch_first_inclusion_labels(
                        problem=problem,
                        endpoint_counts=endpoint_plan.endpoint_counts,
                        parent_live=parent_live,
                        required=required,
                        relaxed=relaxed,
                        reduced=reduced,
                        ratio=ratio_t,
                        eligible=lease.source_tensor("eligible"),
                        required_t=lease.source_tensor("required"),
                        tile_size=design.rounding_tile_size,
                    )
                )
    except V1SignedWorkDeviceShellRefusal:
        raise
    except V1SignedWorkLeastWorkCUDARefusal as error:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            f"single-branch theorem-state custody refused rounding: {error}",
        )
    blocks = _compact_blocks(problem)
    try:
        frozen = freeze_v1_device_shell_ladder(
            handle=handle,
            parent_mask=parent_mask,
            eligible_mask=source.eligible_mask,
            block_capacities=blocks,
            endpoint_count=len(endpoint_plan.endpoint_counts),
            source_ladder_hash=source_hash,
            physical_chunk_size=design.physical_chunk_size,
            allow_unconstrained_blocks=design.allow_unconstrained_blocks,
        )
    except Exception:
        handle.evict()
        raise
    branch_identity = V1SignedWorkDeviceShellBranchIdentityReceipt(
        branch=execution.branch,
        source_rounding_hash=source_hash,
        frozen_ladder=frozen,
        signed_work_branch_receipt_hash=execution.branch_receipt.receipt_hash,
        endpoint_plan_hash=endpoint_plan.receipt_hash,
    )
    receipt = V1SignedWorkDeviceShellExecutionReceipt(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        least_work_execution_hash=execution.execution_hash,
        state_identity_hash=state.identity_hash,
        endpoint_plan_hash=endpoint_plan.receipt_hash,
        design_hash=design.design_hash,
        parent_mask_hash=parent_mask.mask_hash,
        source_geometry_hash=source.source_geometry_hash,
        source_registration_hash=problem.source_registration_hash,
        registered_d_coordinate_hash=problem.registered_d_coordinate_hash,
        selected_branch=execution.branch,
        branch_identity=branch_identity,
        hbm_forecast_hash=forecast.forecast_hash,
        required_hard_action_hash=(
            None
            if problem.required_hard_action is None
            else content_hash(problem.required_hard_action)
        ),
    )
    return V1SignedWorkDeviceShellExecutionBundle(
        label_handle=handle,
        execution_receipt=receipt,
        hbm_forecast=forecast,
        production_cuda_execution=execution.production_cuda_execution,
        private_cpu_reference_execution=execution.private_cpu_reference_execution,
    )


def freeze_v1_signed_work_device_shell_ladder(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt,
    parent_mask: HardMask,
    design: V1SignedWorkDeviceShellExecutionDesign = (
        V1SignedWorkDeviceShellExecutionDesign()
    ),
    branch: V1RelaxedKKTBranch | None = None,
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1SignedWorkDeviceShellExecutionBundle:
    """Freeze a signed-work relaxed proposal into uint8 first-inclusion labels."""

    _validate_static_inputs(
        problem=problem,
        policy=policy,
        execution=execution,
        endpoint_plan=endpoint_plan,
        parent_mask=parent_mask,
        design=design,
    )
    selected, branch_receipt = _selected_branch(execution, branch)
    parent_live = _parent_live_numpy(parent_mask)
    required = _required_numpy(problem)
    required_total = _validate_endpoint_capacity(
        problem=problem,
        endpoint_counts=endpoint_plan.endpoint_counts,
        parent_live=parent_live,
        required=required,
        design=design,
    )
    del required_total
    forecast = estimate_v1_signed_work_device_shell_cuda_hbm(
        problem=problem,
        policy=policy,
        execution=execution,
        endpoint_plan=endpoint_plan,
        design=design,
        branch=selected,
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.HBM_FORECAST_EXCEEDED,
            "signed-work device-shell forecast exceeds the registered HBM ceiling",
        )
    state = execution.relaxed_state_handle
    if state is None:  # pragma: no cover - already checked by static validation.
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.NO_RELAXED_STATE,
            "device-shell rounding requires retained theorem state",
        )
    if design.production_cuda_required and not execution.production_cuda_execution:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.CUDA_REQUIRED,
            "production signed-work device shell requires CUDA execution",
        )
    if (
        execution.private_cpu_reference_execution
        and not design.allow_private_cpu_reference
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.CUDA_REQUIRED,
            "private CPU signed-work device shell was not explicitly enabled",
        )
    if (
        design.require_deterministic_algorithms
        and execution.production_cuda_execution
        and torch is not None
        and not torch.are_deterministic_algorithms_enabled()
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.DETERMINISM_REQUIRED,
            "CUDA shell rounding requires deterministic torch algorithms",
        )
    source_hash = _source_rounding_hash(
        problem=problem,
        policy=policy,
        execution=execution,
        endpoint_plan=endpoint_plan,
        design=design,
        parent_mask=parent_mask,
        branch=selected,
        branch_receipt=branch_receipt,
    )
    expected_receipts = tuple(receipt.receipt_hash for receipt in execution.branch_receipts)
    try:
        with state.borrow_for_device_shell_rounding(
            expected_problem_hash=problem.problem_hash,
            expected_policy_hash=policy.policy_hash,
            expected_branch_receipt_hashes=expected_receipts,  # type: ignore[arg-type]
        ) as lease:
            relaxed = lease.primal(selected)
            reduced = lease.reduced_advantage(selected)
            if execution.private_cpu_reference_execution:
                ratio = _ratio_numpy(problem)
                if problem.source_problem.route_count > design.private_reference_route_limit:
                    _refuse(
                        V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
                        "private CPU label reference is limited to focused tests",
                    )
                labels_np = _cpu_first_inclusion_labels(
                    problem=problem,
                    endpoint_counts=endpoint_plan.endpoint_counts,
                    parent_live=parent_live,
                    required=required,
                    relaxed=relaxed.detach().cpu().numpy().copy(),
                    reduced=reduced.detach().cpu().numpy().copy(),
                    ratio=ratio,
                )
                handle = V1DeviceShellLabelHandle.from_numpy_reference(labels_np)
            else:
                resource_t = lease.source_tensor("resource")
                singleton_work_t = lease.source_tensor("singleton_work")
                eligible_t = lease.source_tensor("eligible")
                required_t = lease.source_tensor("required")
                ratio_t = torch.where(
                    resource_t > 0.0,
                    singleton_work_t / resource_t,
                    torch.zeros_like(resource_t),
                )
                labels_t = _torch_first_inclusion_labels(
                    problem=problem,
                    endpoint_counts=endpoint_plan.endpoint_counts,
                    parent_live=parent_live,
                    required=required,
                    relaxed=relaxed,
                    reduced=reduced,
                    ratio=ratio_t,
                    eligible=eligible_t,
                    required_t=required_t,
                    tile_size=design.rounding_tile_size,
                )
                handle = V1DeviceShellLabelHandle.from_torch_cuda(labels_t)
    except V1SignedWorkDeviceShellRefusal:
        raise
    except V1SignedWorkLeastWorkCUDARefusal as error:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            f"signed-work theorem-state custody refused rounding: {error}",
        )
    except Exception as error:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.LABEL_FREEZE_FAILED,
            f"signed-work device-shell label construction failed: {error}",
        )

    blocks = _compact_blocks(problem)
    try:
        frozen = freeze_v1_device_shell_ladder(
            handle=handle,
            parent_mask=parent_mask,
            eligible_mask=problem.source_problem.eligible_mask,
            block_capacities=blocks,
            endpoint_count=len(endpoint_plan.endpoint_counts),
            source_ladder_hash=source_hash,
            physical_chunk_size=design.physical_chunk_size,
            allow_unconstrained_blocks=design.allow_unconstrained_blocks,
        )
    except Exception:
        handle.evict()
        raise
    branch_identity = V1SignedWorkDeviceShellBranchIdentityReceipt(
        branch=selected,
        source_rounding_hash=source_hash,
        frozen_ladder=frozen,
        signed_work_branch_receipt_hash=branch_receipt.receipt_hash,
        endpoint_plan_hash=endpoint_plan.receipt_hash,
    )
    receipt = V1SignedWorkDeviceShellExecutionReceipt(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        least_work_execution_hash=execution.execution_hash,
        state_identity_hash=state.identity_hash,
        endpoint_plan_hash=endpoint_plan.receipt_hash,
        design_hash=design.design_hash,
        parent_mask_hash=parent_mask.mask_hash,
        source_geometry_hash=problem.source_problem.source_geometry_hash,
        source_registration_hash=problem.source_registration_hash,
        registered_d_coordinate_hash=problem.registered_d_coordinate_hash,
        selected_branch=selected,
        branch_identity=branch_identity,
        hbm_forecast_hash=forecast.forecast_hash,
        required_hard_action_hash=(
            None
            if problem.required_hard_action is None
            else content_hash(problem.required_hard_action)
        ),
    )
    return V1SignedWorkDeviceShellExecutionBundle(
        label_handle=handle,
        execution_receipt=receipt,
        hbm_forecast=forecast,
        production_cuda_execution=execution.production_cuda_execution,
        private_cpu_reference_execution=execution.private_cpu_reference_execution,
    )


def freeze_v1_signed_work_device_prefix_rank_path(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt,
    parent_mask: HardMask,
    design: V1SignedWorkDeviceShellExecutionDesign = (
        V1SignedWorkDeviceShellExecutionDesign()
    ),
    branch: V1RelaxedKKTBranch | None = None,
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1DevicePrefixRankHandle:
    """Freeze the complete D-rounding prefix order as rank custody.

    The returned handle is not an F/Q/R certificate and cannot authorize
    calibration, oracle execution, commits, terminal claims, or Pareto claims.
    It only preserves the branch-specific deterministic rounding path so later
    C-stage endpoint-count queries can materialize bounded uint8 labels without
    recomputing or reordering the proposal after seeing C outcomes.
    """

    _validate_static_inputs(
        problem=problem,
        policy=policy,
        execution=execution,
        endpoint_plan=endpoint_plan,
        parent_mask=parent_mask,
        design=design,
    )
    selected, branch_receipt = _selected_branch(execution, branch)
    parent_live = _parent_live_numpy(parent_mask)
    required = _required_numpy(problem)
    required_total = _validate_endpoint_capacity(
        problem=problem,
        endpoint_counts=endpoint_plan.endpoint_counts,
        parent_live=parent_live,
        required=required,
        design=design,
    )
    maximum_free = endpoint_plan.endpoint_counts[-1] - required_total
    forecast = estimate_v1_signed_work_device_shell_cuda_hbm(
        problem=problem,
        policy=policy,
        execution=execution,
        endpoint_plan=endpoint_plan,
        design=design,
        branch=selected,
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
    )
    # Logical ranks are uint32.  CUDA custody intentionally uses torch.int32,
    # the smallest broadly supported comparison/indexing representation that
    # safely covers the registered 57M-route axis.
    extra_rank_bytes = max(
        0,
        problem.source_problem.route_count * (_I32_BYTES - _U8_BYTES),
    )
    incremental = forecast.forecast_incremental_bytes + extra_rank_bytes
    if execution.production_cuda_execution and (current_reserved_bytes is None or device_total_memory_bytes is None):
        device = torch.device(forecast.device)
        admitted = (int(torch.cuda.memory_allocated(device)) + incremental <= forecast.hbm_ceiling_bytes
                    and incremental <= cuda_reusable_memory_bytes(device, framework=torch))
    else:
        admitted = forecast.forecast_peak_reserved_bytes + extra_rank_bytes <= forecast.hbm_ceiling_bytes
    if not admitted:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.HBM_FORECAST_EXCEEDED,
            "prefix-rank custody forecast exceeds the registered HBM ceiling",
        )
    state = execution.relaxed_state_handle
    if state is None:  # pragma: no cover - already checked by static validation.
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.NO_RELAXED_STATE,
            "prefix-rank custody requires retained theorem state",
        )
    if design.production_cuda_required and not execution.production_cuda_execution:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.CUDA_REQUIRED,
            "production prefix-rank custody requires CUDA execution",
        )
    if (
        execution.private_cpu_reference_execution
        and not design.allow_private_cpu_reference
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.CUDA_REQUIRED,
            "private CPU prefix-rank custody was not explicitly enabled",
        )
    if (
        design.require_deterministic_algorithms
        and execution.production_cuda_execution
        and torch is not None
        and not torch.are_deterministic_algorithms_enabled()
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.DETERMINISM_REQUIRED,
            "CUDA prefix-rank custody requires deterministic torch algorithms",
        )
    source_hash = _source_rounding_hash(
        problem=problem,
        policy=policy,
        execution=execution,
        endpoint_plan=endpoint_plan,
        design=design,
        parent_mask=parent_mask,
        branch=selected,
        branch_receipt=branch_receipt,
    )
    expected_receipts = tuple(receipt.receipt_hash for receipt in execution.branch_receipts)
    try:
        with state.borrow_for_device_shell_rounding(
            expected_problem_hash=problem.problem_hash,
            expected_policy_hash=policy.policy_hash,
            expected_branch_receipt_hashes=expected_receipts,  # type: ignore[arg-type]
        ) as lease:
            relaxed = lease.primal(selected)
            reduced = lease.reduced_advantage(selected)
            if execution.private_cpu_reference_execution:
                if problem.source_problem.route_count > design.private_reference_route_limit:
                    _refuse(
                        V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
                        "private CPU prefix-rank reference is limited to focused tests",
                    )
                ranks_np = _cpu_prefix_ranks(
                    problem=problem,
                    maximum_free=maximum_free,
                    parent_live=parent_live,
                    required=required,
                    relaxed=relaxed.detach().cpu().numpy().copy(),
                    reduced=reduced.detach().cpu().numpy().copy(),
                    ratio=_ratio_numpy(problem),
                )
                return V1DevicePrefixRankHandle.from_numpy_reference(
                    ranks=ranks_np,
                    required=required,
                    route_axis_hash=problem.source_problem.route_axis_hash,
                    source_rounding_hash=source_hash,
                )
            resource_t = lease.source_tensor("resource")
            singleton_work_t = lease.source_tensor("singleton_work")
            eligible_t = lease.source_tensor("eligible")
            required_t = lease.source_tensor("required")
            ratio_t = torch.where(
                resource_t > 0.0,
                singleton_work_t / resource_t,
                torch.zeros_like(resource_t),
            )
            ranks_t = _torch_prefix_ranks(
                problem=problem,
                maximum_free=maximum_free,
                parent_live=parent_live,
                required=required,
                relaxed=relaxed,
                reduced=reduced,
                ratio=ratio_t,
                eligible=eligible_t,
                required_t=required_t,
                tile_size=design.rounding_tile_size,
            )
            return V1DevicePrefixRankHandle.from_torch_cuda(
                ranks=ranks_t,
                required=required,
                route_axis_hash=problem.source_problem.route_axis_hash,
                source_rounding_hash=source_hash,
            )
    except V1SignedWorkDeviceShellRefusal:
        raise
    except V1SignedWorkLeastWorkCUDARefusal as error:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            f"signed-work theorem-state custody refused prefix-rank freeze: {error}",
        )
    except Exception as error:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.LABEL_FREEZE_FAILED,
            f"signed-work prefix-rank construction failed: {error}",
        )


def freeze_v1_signed_work_device_shell_family(
    *,
    problem: V1SignedWorkLeastWorkCUDAProblem,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt,
    parent_mask: HardMask,
    design: V1SignedWorkDeviceShellExecutionDesign = (
        V1SignedWorkDeviceShellExecutionDesign()
    ),
    device_total_memory_bytes: int | None = None,
    current_reserved_bytes: int | None = None,
) -> V1SignedWorkDeviceShellFamilyBundle:
    """Freeze both convex Q-branch rounded ladders before independent C.

    Relaxed branch selection does not imply that deterministic hard rounding
    preserves the same winner.  Canonical proposal freeze therefore retains
    both converged branches and delegates elimination to complete signed F/Q
    replay and later independent certification.
    """

    spectral = freeze_v1_signed_work_device_shell_ladder(
        problem=problem,
        policy=policy,
        execution=execution,
        endpoint_plan=endpoint_plan,
        parent_mask=parent_mask,
        design=design,
        branch=V1RelaxedKKTBranch.SPECTRAL,
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
    )
    try:
        diagonal = freeze_v1_signed_work_device_shell_ladder(
            problem=problem,
            policy=policy,
            execution=execution,
            endpoint_plan=endpoint_plan,
            parent_mask=parent_mask,
            design=design,
            branch=V1RelaxedKKTBranch.DIAGONAL_L1,
            device_total_memory_bytes=device_total_memory_bytes,
            # Query the live allocator after the first persistent label vector
            # when the caller did not freeze a synthetic reference budget.
            current_reserved_bytes=(
                None
                if current_reserved_bytes is None
                else current_reserved_bytes
                # Only the first branch's uint8 first-inclusion labels remain
                # live across the second freeze.  Its candidate/eligibility/
                # required masks are transient and have already been released.
                + spectral.hbm_forecast.route_count * _U8_BYTES
            ),
        )
    except Exception:
        spectral.label_handle.evict()
        raise
    try:
        receipt = V1SignedWorkDeviceShellFamilyReceipt(
            problem_hash=problem.problem_hash,
            policy_hash=policy.policy_hash,
            least_work_execution_hash=execution.execution_hash,
            endpoint_plan_hash=endpoint_plan.receipt_hash,
            design_hash=design.design_hash,
            parent_mask_hash=parent_mask.mask_hash,
            spectral_execution_hash=spectral.execution_hash,
            diagonal_l1_execution_hash=diagonal.execution_hash,
        )
        return V1SignedWorkDeviceShellFamilyBundle(
            spectral=spectral,
            diagonal_l1=diagonal,
            receipt=receipt,
        )
    except Exception:
        V1DeviceShellLabelHandle.evict_all_atomically(
            (spectral.label_handle, diagonal.label_handle)
        )
        raise


# ---------------------------------------------------------------------------
# Zero-copy DeviceProblemV2 shell custody
# ---------------------------------------------------------------------------

SIGNED_WORK_DEVICE_SHELL_DEVICE_SCHEMA_V2: Final = (
    "partition_free_v1_signed_work_device_shell_live_device_receipt_v2"
)
SIGNED_WORK_DEVICE_SHELL_DEVICE_BUNDLE_SCHEMA_V2: Final = (
    "partition_free_v1_signed_work_device_shell_live_device_bundle_v2"
)
SIGNED_WORK_DEVICE_SHELL_DEVICE_FAMILY_SCHEMA_V2: Final = (
    "partition_free_v1_signed_work_device_shell_live_device_family_v2"
)
SIGNED_WORK_DEVICE_SHELL_DEVICE_HBM_SCHEMA_V2: Final = (
    "partition_free_v1_signed_work_device_shell_live_device_hbm_v2"
)
SIGNED_WORK_DEVICE_ACTION_COMMITMENT_SCHEMA_V2: Final = (
    "partition_free_v1_device_action_algorithmic_commitment_not_packed_mask_v2"
)

_DEVICE_V2_PREFIX: Final = "partition-free-v1-signed-work-device-shell-live-v2-"
_DEVICE_V2_HANDLE_TOKEN: Final = object()


class V1SignedWorkDeviceShellTensorOperationV2(str, Enum):
    """Closed uses of a live v2 label tensor.

    ``BOUNDED_ENDPOINT_QUERY`` deliberately preserves the later path from a
    complete frozen prefix to one selected packed endpoint.  It does not
    materialize that endpoint here and grants no C or commit authority.
    """

    HARD_FQ_REPLAY = "hard_fq_replay"
    BOUNDED_ENDPOINT_QUERY = "bounded_endpoint_query"


def _torch_storage_invariant_v2(value: Any) -> tuple[object, ...]:
    if torch is None or type(value) is not torch.Tensor:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "v2 shell custody lost its exact torch tensor",
        )
    storage = value.untyped_storage()
    return (
        str(value.device),
        str(value.dtype),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
        int(value.storage_offset()),
        int(value.data_ptr()),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(value._version),
        bool(value.requires_grad),
        bool(value.is_contiguous()),
    )


class V1SignedWorkDeviceShellTensorLeaseV2:
    """Nonserializable exact-operation borrow of one live label tensor."""

    __slots__ = ("_closed", "_handle", "_operation", "_token")
    __hash__ = None

    def __init__(
        self,
        *,
        _issuance_token: object,
        handle: "V1SignedWorkDeviceShellTensorHandleV2",
        operation: V1SignedWorkDeviceShellTensorOperationV2,
        token: object,
    ) -> None:
        if _issuance_token is not _DEVICE_V2_HANDLE_TOKEN:
            raise TypeError("v2 shell tensor leases are handle-issued only")
        self._handle: V1SignedWorkDeviceShellTensorHandleV2 | None = handle
        self._operation = operation
        self._token: object | None = token
        self._closed = False

    @property
    def tensor(self) -> Any:
        handle = self._handle
        token = self._token
        if self._closed or handle is None or token is None:
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
                "v2 shell label lease is closed",
            )
        return handle._tensor_for_lease(token=token, operation=self._operation)

    def close(self) -> None:
        if self._closed:
            return
        handle = self._handle
        token = self._token
        if handle is None or token is None:  # pragma: no cover - defensive.
            raise RuntimeError("v2 shell label lease lost its owner")
        try:
            handle._release(token=token, operation=self._operation)
        finally:
            self._handle = None
            self._token = None
            self._closed = True

    def __enter__(self) -> "V1SignedWorkDeviceShellTensorLeaseV2":
        # Touch the borrow once so a closed or mutated custody refuses at
        # entry (``del`` on the deleter-less property would raise instead).
        _ = self.tensor
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __reduce__(self) -> NoReturn:
        raise TypeError("v2 shell tensor leases are nonserializable")


class V1SignedWorkDeviceShellTensorHandleV2:
    """Process-local label storage with pointer/version custody and no D2H."""

    __slots__ = (
        "_active",
        "_evicted",
        "_invariant",
        "_lock",
        "_tensor",
    )
    __hash__ = None

    def __init__(self, *, tensor: Any, _issuance_token: object) -> None:
        if _issuance_token is not _DEVICE_V2_HANDLE_TOKEN:
            raise TypeError("v2 shell tensor handles are issuer-only")
        if (
            torch is None
            or type(tensor) is not torch.Tensor
            or tensor.dtype != torch.uint8
            or tensor.ndim != 1
            or int(tensor.numel()) <= 0
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise TypeError("v2 shell labels require one exact contiguous uint8 tensor")
        self._tensor: Any | None = tensor
        self._invariant = _torch_storage_invariant_v2(tensor)
        self._active: dict[object, V1SignedWorkDeviceShellTensorOperationV2] = {}
        self._evicted = False
        self._lock = RLock()

    def _require_live(self) -> Any:
        tensor = self._tensor
        if self._evicted or tensor is None:
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
                "v2 shell labels were evicted",
            )
        if _torch_storage_invariant_v2(tensor) != self._invariant:
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
                "v2 shell label storage or mutation version changed",
            )
        return tensor

    @property
    def route_count(self) -> int:
        with self._lock:
            return int(self._require_live().numel())

    @property
    def device(self) -> str:
        with self._lock:
            return str(self._require_live().device)

    @property
    def is_evicted(self) -> bool:
        return self._evicted

    def borrow(
        self, *, operation: V1SignedWorkDeviceShellTensorOperationV2
    ) -> V1SignedWorkDeviceShellTensorLeaseV2:
        if type(operation) is not V1SignedWorkDeviceShellTensorOperationV2:
            raise TypeError("v2 shell borrow operation must be exact")
        with self._lock:
            self._require_live()
            token = object()
            self._active[token] = operation
            return V1SignedWorkDeviceShellTensorLeaseV2(
                _issuance_token=_DEVICE_V2_HANDLE_TOKEN,
                handle=self,
                operation=operation,
                token=token,
            )

    def _tensor_for_lease(
        self,
        *,
        token: object,
        operation: V1SignedWorkDeviceShellTensorOperationV2,
    ) -> Any:
        with self._lock:
            if self._active.get(token) is not operation:
                _refuse(
                    V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
                    "v2 shell label lease operation or token is stale",
                )
            return self._require_live()

    def _release(
        self,
        *,
        token: object,
        operation: V1SignedWorkDeviceShellTensorOperationV2,
    ) -> None:
        with self._lock:
            if self._active.pop(token, None) is not operation:
                _refuse(
                    V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
                    "v2 shell label lease release is stale",
                )
            self._require_live()

    def evict(self) -> None:
        with self._lock:
            self._require_live()
            if self._active:
                _refuse(
                    V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
                    "v2 shell labels cannot be evicted while borrowed",
                )
            self._tensor = None
            self._evicted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("v2 shell tensor handles are nonserializable")


@dataclass(frozen=True, slots=True)
class V1SignedWorkDeviceShellDeviceHBMForecastV2:
    problem_hash: str
    policy_hash: str
    least_work_execution_hash: str
    endpoint_plan_hash: str
    design_hash: str
    branch: V1RelaxedKKTBranch
    device: str
    route_count: int
    endpoint_count: int
    maximum_endpoint_count: int
    new_label_storage_bytes: int
    bounded_order_workspace_bytes: int
    tile_workspace_bytes: int
    compact_histogram_bytes: int
    forecast_incremental_bytes: int
    current_reserved_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_bytes: int
    admitted: bool
    borrowed_route_source_bytes_counted_as_new: bool = False
    route_scale_d2h_bytes: int = 0
    host_route_mask_expansion_bytes: int = 0
    schema: str = SIGNED_WORK_DEVICE_SHELL_DEVICE_HBM_SCHEMA_V2
    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _admission_memory: tuple[int, int] | None) -> None:
        available, allocated = _admission_memory or (self.device_total_memory_bytes, self.current_reserved_bytes)
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("v2 shell forecast branch must be exact")
        for name in (
            "problem_hash",
            "policy_hash",
            "least_work_execution_hash",
            "endpoint_plan_hash",
            "design_hash",
            "device",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        routes = _positive_int(self.route_count, name="route count")
        endpoints = _positive_int(self.endpoint_count, name="endpoint count")
        maximum = _positive_int(
            self.maximum_endpoint_count, name="maximum endpoint count"
        )
        if self.new_label_storage_bytes != routes * _U8_BYTES:
            raise ValueError("v2 shell label forecast does not replay")
        incremental = (
            self.new_label_storage_bytes
            + self.bounded_order_workspace_bytes
            + self.tile_workspace_bytes
            + self.compact_histogram_bytes
        )
        if (
            endpoints > DEVICE_SHELL_MAXIMUM_ENDPOINTS
            or maximum > routes
            or any(
                _nonnegative_int(getattr(self, name), name=name) != getattr(self, name)
                for name in (
                    "bounded_order_workspace_bytes",
                    "tile_workspace_bytes",
                    "compact_histogram_bytes",
                    "current_reserved_bytes",
                    "device_total_memory_bytes",
                    "hbm_ceiling_bytes",
                )
            )
            or self.forecast_incremental_bytes != incremental
            or self.forecast_peak_reserved_bytes
            != self.current_reserved_bytes + incremental
            or self.admitted
            is not (allocated + incremental <= self.hbm_ceiling_bytes and incremental <= available)
            or self.borrowed_route_source_bytes_counted_as_new
            or self.route_scale_d2h_bytes != 0
            or self.host_route_mask_expansion_bytes != 0
            or self.schema != SIGNED_WORK_DEVICE_SHELL_DEVICE_HBM_SCHEMA_V2
        ):
            raise ValueError("v2 shell HBM or zero-copy semantics do not replay")
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "endpoint_count", endpoints)
        object.__setattr__(self, "maximum_endpoint_count", maximum)

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_DEVICE_V2_PREFIX + "hbm-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkDeviceShellDeviceReceiptV2:
    problem_hash: str
    policy_hash: str
    least_work_execution_hash: str
    state_identity_hash: str
    endpoint_plan_hash: str
    design_hash: str
    current_mask_hash: str
    structural_source_receipt_hash: str
    source_geometry_hash: str
    source_registration_hash: str
    registered_d_coordinate_hash: str
    branch: V1RelaxedKKTBranch
    signed_work_branch_receipt_hash: str
    route_axis_hash: str
    route_count: int
    endpoint_counts: tuple[int, ...]
    shell_counts: tuple[int, ...]
    block_prefix_counts: tuple[tuple[int, ...], ...]
    device_action_commitment_hashes: tuple[str, ...]
    device: str
    hbm_forecast_hash: str
    action_commitment_schema: str = SIGNED_WORK_DEVICE_ACTION_COMMITMENT_SCHEMA_V2
    legacy_packed_action_hashes_available: bool = False
    complete_prefix_frozen_for_bounded_endpoint_query: bool = True
    device_storage_version_custody: bool = True
    route_scale_d2h_bytes: int = 0
    host_route_mask_expansion_bytes: int = 0
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = SIGNED_WORK_DEVICE_SHELL_AUTHORITY
    schema: str = SIGNED_WORK_DEVICE_SHELL_DEVICE_SCHEMA_V2

    def __post_init__(self) -> None:
        for name in (
            "problem_hash",
            "policy_hash",
            "least_work_execution_hash",
            "state_identity_hash",
            "endpoint_plan_hash",
            "design_hash",
            "current_mask_hash",
            "structural_source_receipt_hash",
            "source_geometry_hash",
            "source_registration_hash",
            "registered_d_coordinate_hash",
            "signed_work_branch_receipt_hash",
            "route_axis_hash",
            "device",
            "hbm_forecast_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("v2 shell receipt branch must be exact")
        routes = _positive_int(self.route_count, name="route count")
        counts = tuple(_positive_int(item, name="endpoint count") for item in self.endpoint_counts)
        shells = tuple(_positive_int(item, name="shell count") for item in self.shell_counts)
        if (
            not counts
            or tuple(sorted(set(counts))) != counts
            or len(counts) != len(shells)
            or tuple(int(item) for item in np.cumsum(shells)) != counts
            or counts[-1] > routes
        ):
            raise ValueError("v2 shell endpoint histogram is inconsistent")
        block_prefix = tuple(tuple(int(item) for item in row) for row in self.block_prefix_counts)
        if any(
            len(row) != len(counts)
            or any(item < 0 for item in row)
            or tuple(sorted(row)) != row
            for row in block_prefix
        ):
            raise ValueError("v2 shell compact block prefixes are malformed")
        if block_prefix and tuple(
            sum(row[index] for row in block_prefix)
            for index in range(len(counts))
        ) != counts:
            raise ValueError("v2 shell compact blocks do not partition endpoints")
        actions = tuple(
            _identity(item, name="device action commitment hash")
            for item in self.device_action_commitment_hashes
        )
        if len(actions) != len(counts):
            raise ValueError("v2 shell action commitments do not cover endpoints")
        closed = (
            self.legacy_packed_action_hashes_available,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if (
            any(type(item) is not bool for item in closed)
            or any(closed)
            or self.complete_prefix_frozen_for_bounded_endpoint_query is not True
            or self.device_storage_version_custody is not True
            or self.route_scale_d2h_bytes != 0
            or self.host_route_mask_expansion_bytes != 0
            or self.action_commitment_schema
            != SIGNED_WORK_DEVICE_ACTION_COMMITMENT_SCHEMA_V2
            or self.authority_scope != SIGNED_WORK_DEVICE_SHELL_AUTHORITY
            or self.schema != SIGNED_WORK_DEVICE_SHELL_DEVICE_SCHEMA_V2
        ):
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
                "v2 device shell crossed proposal-only identity authority",
            )
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "endpoint_counts", counts)
        object.__setattr__(self, "shell_counts", shells)
        object.__setattr__(self, "block_prefix_counts", block_prefix)
        object.__setattr__(self, "device_action_commitment_hashes", actions)

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_DEVICE_V2_PREFIX + "receipt-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkDeviceShellDeviceBundleV2:
    label_handle: V1SignedWorkDeviceShellTensorHandleV2 = field(
        repr=False, compare=False
    )
    receipt: V1SignedWorkDeviceShellDeviceReceiptV2
    hbm_forecast: V1SignedWorkDeviceShellDeviceHBMForecastV2
    schema: str = SIGNED_WORK_DEVICE_SHELL_DEVICE_BUNDLE_SCHEMA_V2

    def __post_init__(self) -> None:
        if type(self.label_handle) is not V1SignedWorkDeviceShellTensorHandleV2:
            raise TypeError("v2 shell bundle requires exact live label custody")
        if type(self.receipt) is not V1SignedWorkDeviceShellDeviceReceiptV2:
            raise TypeError("v2 shell bundle requires exact receipt")
        if type(self.hbm_forecast) is not V1SignedWorkDeviceShellDeviceHBMForecastV2:
            raise TypeError("v2 shell bundle requires exact HBM forecast")
        if (
            self.receipt.hbm_forecast_hash != self.hbm_forecast.forecast_hash
            or self.label_handle.route_count != self.receipt.route_count
            or self.label_handle.device != self.receipt.device
            or self.schema != SIGNED_WORK_DEVICE_SHELL_DEVICE_BUNDLE_SCHEMA_V2
        ):
            raise ValueError("v2 shell live storage and receipt differ")

    @property
    def bundle_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "receipt_hash": self.receipt.receipt_hash,
                "forecast_hash": self.hbm_forecast.forecast_hash,
                "live_device_commitment_only": True,
            },
            prefix=_DEVICE_V2_PREFIX + "bundle-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkDeviceShellDeviceFamilyV2:
    spectral: V1SignedWorkDeviceShellDeviceBundleV2
    diagonal_l1: V1SignedWorkDeviceShellDeviceBundleV2
    problem_hash: str
    policy_hash: str
    least_work_execution_hash: str
    endpoint_plan_hash: str
    current_mask_hash: str
    c_stream_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = SIGNED_WORK_DEVICE_SHELL_DEVICE_FAMILY_SCHEMA_V2

    def __post_init__(self) -> None:
        if (
            type(self.spectral) is not V1SignedWorkDeviceShellDeviceBundleV2
            or type(self.diagonal_l1) is not V1SignedWorkDeviceShellDeviceBundleV2
        ):
            raise TypeError("v2 shell family requires exact branch bundles")
        for name in (
            "problem_hash",
            "policy_hash",
            "least_work_execution_hash",
            "endpoint_plan_hash",
            "current_mask_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        left = self.spectral.receipt
        right = self.diagonal_l1.receipt
        if (
            left.branch is not V1RelaxedKKTBranch.SPECTRAL
            or right.branch is not V1RelaxedKKTBranch.DIAGONAL_L1
            or left.problem_hash != self.problem_hash
            or right.problem_hash != self.problem_hash
            or left.policy_hash != self.policy_hash
            or right.policy_hash != self.policy_hash
            or left.least_work_execution_hash != self.least_work_execution_hash
            or right.least_work_execution_hash != self.least_work_execution_hash
            or left.endpoint_plan_hash != self.endpoint_plan_hash
            or right.endpoint_plan_hash != self.endpoint_plan_hash
            or left.current_mask_hash != self.current_mask_hash
            or right.current_mask_hash != self.current_mask_hash
            or left.endpoint_counts != right.endpoint_counts
            or self.c_stream_authority
            or self.mask_commit_authority
            or self.schema != SIGNED_WORK_DEVICE_SHELL_DEVICE_FAMILY_SCHEMA_V2
        ):
            raise ValueError("v2 shell family identity or authority differs")

    @property
    def family_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "problem_hash": self.problem_hash,
                "policy_hash": self.policy_hash,
                "least_work_execution_hash": self.least_work_execution_hash,
                "endpoint_plan_hash": self.endpoint_plan_hash,
                "current_mask_hash": self.current_mask_hash,
                "spectral": self.spectral.bundle_hash,
                "diagonal_l1": self.diagonal_l1.bundle_hash,
            },
            prefix=_DEVICE_V2_PREFIX + "family-",
        )

    def evict_labels(self) -> None:
        # Fail closed before releasing either member.
        if self.spectral.label_handle.is_evicted or self.diagonal_l1.label_handle.is_evicted:
            raise RuntimeError("v2 shell family is already partially evicted")
        self.spectral.label_handle.evict()
        self.diagonal_l1.label_handle.evict()


def _validate_device_v2_inputs(
    *,
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt,
    current_mask: HardMask,
    structural_receipt: V1CompactDeviceStructuralSourceReceipt,
    design: V1SignedWorkDeviceShellExecutionDesign,
) -> None:
    if type(problem) is not V1SignedWorkLeastWorkCUDADeviceProblemV2:
        raise TypeError("v2 device shell requires exact DeviceProblemV2")
    if type(policy) is not V1SignedWorkLeastWorkCUDAPolicy:
        raise TypeError("v2 device shell requires exact policy")
    if type(execution) is not V1SignedWorkLeastWorkCUDAExecutionBundle:
        raise TypeError("v2 device shell requires exact execution")
    if type(endpoint_plan) is not V1SignedWorkDeviceShellEndpointPlanReceipt:
        raise TypeError("v2 device shell requires exact endpoint plan")
    if type(current_mask) is not HardMask:
        raise TypeError("v2 device shell requires exact packed current mask")
    if type(structural_receipt) is not V1CompactDeviceStructuralSourceReceipt:
        raise TypeError("v2 device shell requires exact structural receipt")
    if type(design) is not V1SignedWorkDeviceShellExecutionDesign:
        raise TypeError("v2 device shell requires exact design")
    structural_receipt.assert_integrity()
    state = execution.state_handle
    branch_receipts = tuple(execution.branch_receipts)
    if (
        execution.result.problem_hash != problem.problem_hash
        or execution.result.policy_hash != policy.policy_hash
        or state is None
        or not execution.result.all_branches_converged
        or not execution.result.relaxed_state_available
        or state.problem_hash != problem.problem_hash
        or state.policy_hash != policy.policy_hash
        or state.branch_receipt_hashes
        != tuple(item.receipt_hash for item in branch_receipts)
        or any(
            item.problem_hash != problem.problem_hash
            or item.policy_hash != policy.policy_hash
            or item.operator_receipt_hash != problem.signed_work_operator.receipt_hash
            for item in branch_receipts
        )
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "v2 problem, solver, and retained theorem state differ",
        )
    if (
        # The sealed structural receipt hash binds the structural tensor
        # identities transitively.  Resource and required-action tensors are
        # intrinsic scalar broadcasts, for which DeviceProblemV2 derives fresh
        # identities keyed on this exact receipt hash (so per-tensor
        # re-equality is unsatisfiable by construction); the eligible mask is
        # a real unpacked tensor whose receipt identity passes through
        # exactly, so its equality is kept as an independent leg.
        structural_receipt.receipt_hash != problem.structural_source_receipt_hash
        or structural_receipt.eligible_mask_identity_hash
        != problem.eligible_mask_identity_hash
        # The structural receipt lives on the physical block axis; it joins
        # the route-ID proposal plane only through its ordered route-ID
        # identity.  The packed current mask is a physical-axis object and is
        # bound to the plane through the structural receipt.
        or structural_receipt.route_id_axis_hash != problem.route_axis_hash
        or structural_receipt.route_count != problem.route_count
        or structural_receipt.current_mask_hash != current_mask.mask_hash
        or not structural_receipt.resource_is_intrinsic_unit
        or not structural_receipt.eligible_mask_is_exact_current_mask
        or not structural_receipt.required_hard_action_is_intrinsic_empty
        or current_mask.route_axis_hash != structural_receipt.route_axis_hash
        or current_mask.route_count != problem.route_count
        or endpoint_plan.route_axis_hash != problem.route_axis_hash
        or endpoint_plan.route_count != problem.route_count
        or endpoint_plan.source_registration_hash != problem.source_registration_hash
        or endpoint_plan.registered_d_coordinate_hash
        != problem.registered_d_coordinate_hash
        or endpoint_plan.endpoint_counts[-1]
        > int(structural_receipt.registered_legal_resource_maximum)
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "v2 shell structural, packed-mask, endpoint, or route identity differs",
        )
    if (
        not problem.source_bundle_authoritative
        or execution.result.hard_mask_global_optimality_claimed
        or execution.result.c_stream_authority
        or execution.result.calibration_authority
        or execution.result.bce_oracle_authority
        or execution.result.mask_commit_authority
        or execution.result.terminal_authority
        or execution.result.pareto_claim_authority
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.AUTHORITY_VIOLATION,
            "v2 shell input crossed proposal-only authority",
        )
    if design.production_cuda_required:
        if not execution.production_cuda_execution or not state.device.startswith("cuda:"):
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.CUDA_REQUIRED,
                "production v2 shell requires CUDA theorem state",
            )
    elif (
        not design.allow_private_cpu_reference
        or not execution.private_cpu_reference_execution
        or state.device != "cpu"
        or problem.route_count > design.private_reference_route_limit
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.CUDA_REQUIRED,
            "v2 CPU shell is restricted to an explicit tiny reference",
        )
    if (
        design.require_deterministic_algorithms
        and execution.production_cuda_execution
        and (torch is None or not torch.are_deterministic_algorithms_enabled())
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.DETERMINISM_REQUIRED,
            "v2 CUDA shell requires deterministic torch algorithms",
        )


def estimate_v1_signed_work_device_shell_device_hbm_v2(
    *,
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt,
    current_mask: HardMask,
    structural_receipt: V1CompactDeviceStructuralSourceReceipt,
    design: V1SignedWorkDeviceShellExecutionDesign,
    branch: V1RelaxedKKTBranch,
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkDeviceShellDeviceHBMForecastV2:
    """Forecast only allocations introduced after retained-state custody."""

    _validate_device_v2_inputs(
        problem=problem,
        policy=policy,
        execution=execution,
        endpoint_plan=endpoint_plan,
        current_mask=current_mask,
        structural_receipt=structural_receipt,
        design=design,
    )
    if type(branch) is not V1RelaxedKKTBranch:
        raise TypeError("v2 shell forecast branch must be exact")
    state = execution.state_handle
    if state is None:  # pragma: no cover - validation closes this.
        raise RuntimeError("v2 shell state disappeared")
    routes = problem.route_count
    endpoints = len(endpoint_plan.endpoint_counts)
    maximum = endpoint_plan.endpoint_counts[-1]
    tile = min(routes, design.rounding_tile_size)
    label_bytes = routes * _U8_BYTES
    bounded_order = (5 * maximum + 2 * min(maximum, tile)) * _I64_BYTES
    tile_workspace = tile * (
        4 * _I64_BYTES + 4 * _F64_BYTES + 2 * _BOOL_BYTES
    )
    compact_histogram = (
        (len(problem.block_capacities) + 1) * (endpoints + 1) * _I64_BYTES
    )
    incremental = label_bytes + bounded_order + tile_workspace + compact_histogram
    device = state.device
    admission_memory = None
    if current_reserved_bytes is None or device_total_memory_bytes is None:
        if device.startswith("cuda:"):
            if torch is None or not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkDeviceShellRefusalCode.CUDA_REQUIRED,
                    "v2 CUDA forecast requires an available runtime",
                )
            torch_device = torch.device(device)
            admission_memory = (cuda_reusable_memory_bytes(torch_device, framework=torch), int(torch.cuda.memory_allocated(torch_device)))
            if current_reserved_bytes is None:
                current_reserved_bytes = int(torch.cuda.memory_reserved(torch_device))
            if device_total_memory_bytes is None:
                device_total_memory_bytes = int(
                    torch.cuda.get_device_properties(torch_device).total_memory
                )
        else:
            if current_reserved_bytes is None:
                current_reserved_bytes = 0
            if device_total_memory_bytes is None:
                device_total_memory_bytes = max(1 << 30, 4 * incremental)
    current = _nonnegative_int(current_reserved_bytes, name="current reserved bytes")
    total = _positive_int(device_total_memory_bytes, name="device total memory")
    ceiling = math.floor(design.hbm_ceiling_fraction * total)
    available, allocated = admission_memory or (total, current)
    return V1SignedWorkDeviceShellDeviceHBMForecastV2(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        least_work_execution_hash=execution.execution_hash,
        endpoint_plan_hash=endpoint_plan.receipt_hash,
        design_hash=design.design_hash,
        branch=branch,
        device=device,
        route_count=routes,
        endpoint_count=endpoints,
        maximum_endpoint_count=maximum,
        new_label_storage_bytes=label_bytes,
        bounded_order_workspace_bytes=bounded_order,
        tile_workspace_bytes=tile_workspace,
        compact_histogram_bytes=compact_histogram,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current,
        forecast_peak_reserved_bytes=current + incremental,
        device_total_memory_bytes=total,
        hbm_ceiling_bytes=ceiling,
        admitted=allocated + incremental <= ceiling and incremental <= available,
        _admission_memory=admission_memory,
    )


def _device_blockwise_topk_v2(
    *,
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    relaxed: Any,
    reduced: Any,
    ratio: Any,
    eligible: Any,
    target: int,
    tile_size: int,
) -> Any:
    block_candidates: list[Any] = []
    for block in problem.block_capacities:
        block_target = min(target, block.removal_capacity)
        if block_target <= 0:
            continue
        winners = _tiled_lexicographic_topk_torch(
            relaxed=relaxed,
            reduced=reduced,
            ratio=ratio,
            candidate=eligible,
            start=block.start,
            stop=block.stop,
            target=block_target,
            tile_size=tile_size,
        )
        if int(winners.numel()):
            block_candidates.append(winners)
    if not block_candidates:
        return torch.empty(0, dtype=torch.int64, device=relaxed.device)
    merged = torch.cat(tuple(block_candidates), dim=0)
    return _lexicographic_topk_from_indices_torch(
        indices=merged,
        relaxed=relaxed,
        reduced=reduced,
        ratio=ratio,
        target=target,
    )


def _device_labels_v2(
    *,
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    endpoint_counts: tuple[int, ...],
    relaxed: Any,
    reduced: Any,
    ratio: Any,
    eligible: Any,
    tile_size: int,
) -> Any:
    routes = problem.route_count
    device = relaxed.device
    if (
        relaxed.dtype != torch.float64
        or reduced.dtype != torch.float64
        or ratio.dtype != torch.float64
        or eligible.dtype != torch.bool
        or any(tuple(item.shape) != (routes,) for item in (relaxed, reduced, ratio, eligible))
        or any(item.device != device for item in (reduced, ratio, eligible))
    ):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "v2 shell route tensors differ in axis, dtype, or device",
        )
    maximum = endpoint_counts[-1]
    accepted = _device_blockwise_topk_v2(
        problem=problem,
        relaxed=relaxed,
        reduced=reduced,
        ratio=ratio,
        eligible=eligible,
        target=maximum,
        tile_size=tile_size,
    )
    if int(accepted.numel()) != maximum:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
            "v2 device order cannot cover the frozen endpoint plan",
        )
    labels = torch.zeros(routes, dtype=torch.uint8, device=device)
    previous = 0
    for endpoint_label, stop in enumerate(endpoint_counts, start=1):
        labels[accepted[previous:stop]] = endpoint_label
        previous = stop
    return labels.contiguous()


def _compact_device_histogram_v2(
    *,
    labels: Any,
    endpoint_count: int,
    blocks: tuple[Any, ...],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    histogram = torch.bincount(labels.to(dtype=torch.int64), minlength=endpoint_count + 1)
    shell_counts = tuple(int(item) for item in histogram[1:].tolist())
    if any(item <= 0 for item in shell_counts):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.LABEL_FREEZE_FAILED,
            "v2 device shell contains an empty registered endpoint",
        )
    prefixes: list[tuple[int, ...]] = []
    for block in blocks:
        block_histogram = torch.bincount(
            labels[block.start : block.stop].to(dtype=torch.int64),
            minlength=endpoint_count + 1,
        )
        prefix = tuple(int(item) for item in torch.cumsum(block_histogram[1:], dim=0).tolist())
        if prefix[-1] > block.removal_capacity:
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.INFEASIBLE_ENDPOINT_PLAN,
                "v2 device shell exceeds a registered block capacity",
            )
        prefixes.append(prefix)
    return shell_counts, tuple(prefixes)


def _device_v2_source_rounding_hash(
    *,
    problem_hash: str,
    policy_hash: str,
    least_work_execution_hash: str,
    state_identity_hash: str,
    branch: V1RelaxedKKTBranch,
    branch_receipt_hash: str,
    endpoint_plan_hash: str,
    current_mask_hash: str,
    structural_source_receipt_hash: str,
    source_geometry_hash: str,
    source_registration_hash: str,
    registered_d_coordinate_hash: str,
) -> str:
    """Rounding-source identity shared by the freeze and the bounded query."""

    return content_hash(
        {
            "schema": "partition_free_v1_device_shell_rounding_source_v2",
            "problem_hash": problem_hash,
            "policy_hash": policy_hash,
            "least_work_execution_hash": least_work_execution_hash,
            "state_identity_hash": state_identity_hash,
            "branch": branch,
            "branch_receipt_hash": branch_receipt_hash,
            "endpoint_plan_hash": endpoint_plan_hash,
            "current_mask_hash": current_mask_hash,
            "structural_source_receipt_hash": structural_source_receipt_hash,
            "source_geometry_hash": source_geometry_hash,
            "source_registration_hash": source_registration_hash,
            "registered_d_coordinate_hash": registered_d_coordinate_hash,
            "key_id": SIGNED_WORK_DEVICE_SHELL_KEY_ID,
            "no_route_scale_d2h": True,
        },
        prefix=_DEVICE_V2_PREFIX + "source-",
    )


def _device_v2_action_commitment_hash(
    *,
    source_rounding_hash: str,
    endpoint_label: int,
    cumulative_removed_count: int,
) -> str:
    return content_hash(
        {
            "schema": SIGNED_WORK_DEVICE_ACTION_COMMITMENT_SCHEMA_V2,
            "source_rounding_hash": source_rounding_hash,
            "endpoint_label": endpoint_label,
            "cumulative_removed_count": cumulative_removed_count,
            "live_device_storage_version_custody": True,
            "legacy_packed_mask_hash": None,
        },
        prefix=_DEVICE_V2_PREFIX + "action-",
    )


def freeze_v1_signed_work_device_shell_device_branch_v2(
    *,
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt,
    current_mask: HardMask,
    structural_receipt: V1CompactDeviceStructuralSourceReceipt,
    branch: V1RelaxedKKTBranch,
    design: V1SignedWorkDeviceShellExecutionDesign = V1SignedWorkDeviceShellExecutionDesign(),
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkDeviceShellDeviceBundleV2:
    """Freeze one DeviceProblemV2 branch without NumPy or label D2H."""

    forecast = estimate_v1_signed_work_device_shell_device_hbm_v2(
        problem=problem,
        policy=policy,
        execution=execution,
        endpoint_plan=endpoint_plan,
        current_mask=current_mask,
        structural_receipt=structural_receipt,
        design=design,
        branch=branch,
        current_reserved_bytes=current_reserved_bytes,
        device_total_memory_bytes=device_total_memory_bytes,
    )
    if not forecast.admitted:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.HBM_FORECAST_EXCEEDED,
            "v2 device shell exceeds the registered HBM ceiling",
        )
    state = execution.state_handle
    if state is None:  # pragma: no cover - forecast validation closes this.
        raise RuntimeError("v2 shell theorem state disappeared")
    receipt_map = {item.branch: item for item in execution.branch_receipts}
    branch_receipt = receipt_map.get(branch)
    if branch_receipt is None or not branch_receipt.converged:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.BRANCH_UNAVAILABLE,
            "v2 shell branch lacks one converged KKT receipt",
        )
    expected_receipts = tuple(item.receipt_hash for item in execution.branch_receipts)
    try:
        with state.borrow_for_device_shell_rounding(
            expected_problem_hash=problem.problem_hash,
            expected_policy_hash=policy.policy_hash,
            expected_branch_receipt_hashes=expected_receipts,
        ) as lease:
            relaxed = lease.primal(branch)
            reduced = lease.reduced_advantage(branch)
            resource = lease.source_tensor("resource")
            singleton = lease.source_tensor("singleton_work")
            eligible = lease.source_tensor("eligible")
            required = lease.source_tensor("required")
            if (
                # The required-action identity is the problem's intrinsic
                # derivation keyed on the sealed receipt hash (unsatisfiable
                # to re-equate against the issuer's hash); the eligible mask is
                # a real tensor whose receipt identity passes through exactly.
                state.eligible_mask_identity_hash
                != structural_receipt.eligible_mask_identity_hash
                or int(resource.untyped_storage().nbytes()) > resource.element_size()
                or int(required.untyped_storage().nbytes()) > required.element_size()
                or bool(required[0].item())
                or float(resource[0].item()) != 1.0
            ):
                _refuse(
                    V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
                    "v2 shell state does not retain intrinsic unit/empty structural coordinates",
                )
            labels = _device_labels_v2(
                problem=problem,
                endpoint_counts=endpoint_plan.endpoint_counts,
                relaxed=relaxed,
                reduced=reduced,
                # Unit resource makes corrected-work/resource the borrowed
                # singleton tensor itself; no route-sized ratio is allocated.
                ratio=singleton,
                eligible=eligible,
                tile_size=design.rounding_tile_size,
            )
    except V1SignedWorkDeviceShellRefusal:
        raise
    except V1SignedWorkLeastWorkCUDARefusal as error:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            f"v2 theorem-state custody refused shell freeze: {error}",
        )
    handle = V1SignedWorkDeviceShellTensorHandleV2(
        tensor=labels,
        _issuance_token=_DEVICE_V2_HANDLE_TOKEN,
    )
    try:
        shell_counts, block_prefixes = _compact_device_histogram_v2(
            labels=labels,
            endpoint_count=len(endpoint_plan.endpoint_counts),
            blocks=problem.block_capacities,
        )
        source_hash = _device_v2_source_rounding_hash(
            problem_hash=problem.problem_hash,
            policy_hash=policy.policy_hash,
            least_work_execution_hash=execution.execution_hash,
            state_identity_hash=state.identity_hash,
            branch=branch,
            branch_receipt_hash=branch_receipt.receipt_hash,
            endpoint_plan_hash=endpoint_plan.receipt_hash,
            current_mask_hash=current_mask.mask_hash,
            structural_source_receipt_hash=structural_receipt.receipt_hash,
            source_geometry_hash=problem.source_geometry_hash,
            source_registration_hash=problem.source_registration_hash,
            registered_d_coordinate_hash=problem.registered_d_coordinate_hash,
        )
        actions = tuple(
            _device_v2_action_commitment_hash(
                source_rounding_hash=source_hash,
                endpoint_label=index,
                cumulative_removed_count=count,
            )
            for index, count in enumerate(endpoint_plan.endpoint_counts, start=1)
        )
        receipt = V1SignedWorkDeviceShellDeviceReceiptV2(
            problem_hash=problem.problem_hash,
            policy_hash=policy.policy_hash,
            least_work_execution_hash=execution.execution_hash,
            state_identity_hash=state.identity_hash,
            endpoint_plan_hash=endpoint_plan.receipt_hash,
            design_hash=design.design_hash,
            current_mask_hash=current_mask.mask_hash,
            structural_source_receipt_hash=structural_receipt.receipt_hash,
            source_geometry_hash=problem.source_geometry_hash,
            source_registration_hash=problem.source_registration_hash,
            registered_d_coordinate_hash=problem.registered_d_coordinate_hash,
            branch=branch,
            signed_work_branch_receipt_hash=branch_receipt.receipt_hash,
            route_axis_hash=problem.route_axis_hash,
            route_count=problem.route_count,
            endpoint_counts=endpoint_plan.endpoint_counts,
            shell_counts=shell_counts,
            block_prefix_counts=block_prefixes,
            device_action_commitment_hashes=actions,
            device=state.device,
            hbm_forecast_hash=forecast.forecast_hash,
        )
        return V1SignedWorkDeviceShellDeviceBundleV2(
            label_handle=handle,
            receipt=receipt,
            hbm_forecast=forecast,
        )
    except BaseException:
        handle.evict()
        raise


def freeze_v1_signed_work_device_shell_device_family_v2(
    *,
    problem: V1SignedWorkLeastWorkCUDADeviceProblemV2,
    policy: V1SignedWorkLeastWorkCUDAPolicy,
    execution: V1SignedWorkLeastWorkCUDAExecutionBundle,
    endpoint_plan: V1SignedWorkDeviceShellEndpointPlanReceipt,
    current_mask: HardMask,
    structural_receipt: V1CompactDeviceStructuralSourceReceipt,
    design: V1SignedWorkDeviceShellExecutionDesign = V1SignedWorkDeviceShellExecutionDesign(),
    current_reserved_bytes: int | None = None,
    device_total_memory_bytes: int | None = None,
) -> V1SignedWorkDeviceShellDeviceFamilyV2:
    """Freeze both exact Q branches while retaining only two device labels."""

    spectral = freeze_v1_signed_work_device_shell_device_branch_v2(
        problem=problem,
        policy=policy,
        execution=execution,
        endpoint_plan=endpoint_plan,
        current_mask=current_mask,
        structural_receipt=structural_receipt,
        branch=V1RelaxedKKTBranch.SPECTRAL,
        design=design,
        current_reserved_bytes=current_reserved_bytes,
        device_total_memory_bytes=device_total_memory_bytes,
    )
    try:
        diagonal = freeze_v1_signed_work_device_shell_device_branch_v2(
            problem=problem,
            policy=policy,
            execution=execution,
            endpoint_plan=endpoint_plan,
            current_mask=current_mask,
            structural_receipt=structural_receipt,
            branch=V1RelaxedKKTBranch.DIAGONAL_L1,
            design=design,
            current_reserved_bytes=(
                None
                if current_reserved_bytes is None
                else current_reserved_bytes + problem.route_count * _U8_BYTES
            ),
            device_total_memory_bytes=device_total_memory_bytes,
        )
        return V1SignedWorkDeviceShellDeviceFamilyV2(
            spectral=spectral,
            diagonal_l1=diagonal,
            problem_hash=problem.problem_hash,
            policy_hash=policy.policy_hash,
            least_work_execution_hash=execution.execution_hash,
            endpoint_plan_hash=endpoint_plan.receipt_hash,
            current_mask_hash=current_mask.mask_hash,
        )
    except BaseException:
        spectral.label_handle.evict()
        raise


# ---------------------------------------------------------------------------
# Bounded endpoint packed-identity query
# ---------------------------------------------------------------------------

SIGNED_WORK_DEVICE_SHELL_ENDPOINT_PACKED_IDENTITY_SCHEMA_V2: Final = (
    "partition_free_v1_signed_work_device_shell_endpoint_packed_identity_v2"
)
_BIT_SHIFTS_BIG: Final = (7, 6, 5, 4, 3, 2, 1, 0)


@dataclass(frozen=True, slots=True)
class V1SignedWorkDeviceShellEndpointPackedIdentityReceiptV2:
    """One frozen rung resolved to a packed physical action, bounded D2H.

    The only host transfer is the packed child mask (``ceil(n/8)`` bytes,
    declared as ``packed_payload_d2h_bytes``) plus one count scalar; the
    current mask's packed bytes travel host-to-device once.  The device
    action commitment is recomputed from the shell receipt and matched, so
    this receipt is the lawful crosswalk between a live-storage rung and the
    packed identity the C plane and dedup consume.
    """

    shell_receipt_hash: str
    branch: V1RelaxedKKTBranch
    endpoint_label: int
    cumulative_removed_count: int
    device_action_commitment_hash: str
    current_mask_hash: str
    route_axis_hash: str
    route_count: int
    resulting_mask_hash: str
    resulting_mask_packed_sha256: str
    removed_route_index_hash: str
    action_hash: str
    packed_payload_d2h_bytes: int
    host_mask_upload_bytes: int
    scalar_d2h_bytes: int
    scalar_d2h_count: int = 2
    query_operation: str = "bounded_endpoint_query"
    route_scale_d2h_bytes: int = 0
    route_scale_host_index_materialized: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = SIGNED_WORK_DEVICE_SHELL_ENDPOINT_PACKED_IDENTITY_SCHEMA_V2

    def __post_init__(self) -> None:
        if type(self.branch) is not V1RelaxedKKTBranch:
            raise TypeError("packed-identity receipt branch must be exact")
        for name in (
            "shell_receipt_hash",
            "device_action_commitment_hash",
            "current_mask_hash",
            "route_axis_hash",
            "resulting_mask_hash",
            "resulting_mask_packed_sha256",
            "removed_route_index_hash",
            "action_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "endpoint_label",
            "cumulative_removed_count",
            "route_count",
            "packed_payload_d2h_bytes",
            "host_mask_upload_bytes",
            "scalar_d2h_bytes",
            "scalar_d2h_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        # Exactly two scalars leave the device: the removed count and the
        # retained-superset check.
        if self.scalar_d2h_count != 2 or self.scalar_d2h_bytes != 8 * self.scalar_d2h_count:
            raise ValueError("packed identity query transfers exactly two scalars")
        if self.cumulative_removed_count > self.route_count:
            raise ValueError("packed identity removes more routes than the axis holds")
        expected_bytes = (self.route_count + 7) // 8
        if (
            self.packed_payload_d2h_bytes != expected_bytes
            or self.host_mask_upload_bytes != expected_bytes
        ):
            raise ValueError("packed identity transfer must be exactly the packed mask")
        if int(self.route_scale_d2h_bytes) != 0 or self.route_scale_host_index_materialized:
            raise ValueError("packed identity query is bounded by construction")
        if self.query_operation != "bounded_endpoint_query":
            raise ValueError("packed identity query operation changed")
        flags = (
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags) or any(flags):
            raise ValueError("packed identity receipt carries no authority")
        if self.schema != SIGNED_WORK_DEVICE_SHELL_ENDPOINT_PACKED_IDENTITY_SCHEMA_V2:
            raise ValueError("packed identity semantics changed")

    @property
    def dedup_identity(self) -> tuple[str, str, str, str, int]:
        """The five-part identity ``v1_signed_work_packed_action_dedup_identity`` returns."""

        return (
            self.resulting_mask_packed_sha256,
            self.action_hash,
            self.resulting_mask_hash,
            self.removed_route_index_hash,
            self.cumulative_removed_count,
        )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_DEVICE_V2_PREFIX + "endpoint-packed-identity-")


def _device_unpack_big_endian_bits(packed: Any, *, route_count: int) -> Any:
    shifts = torch.tensor(_BIT_SHIFTS_BIG, dtype=torch.uint8, device=packed.device)
    bits = (packed.reshape(-1, 1) >> shifts.reshape(1, -1)) & 1
    return bits.reshape(-1)[:route_count].to(dtype=torch.bool)


def _device_pack_big_endian_bits(bits: Any) -> Any:
    route_count = int(bits.numel())
    padded_count = ((route_count + 7) // 8) * 8
    padded = torch.zeros(padded_count, dtype=torch.uint8, device=bits.device)
    padded[:route_count] = bits.to(dtype=torch.uint8)
    weights = torch.tensor(
        tuple(1 << shift for shift in _BIT_SHIFTS_BIG),
        dtype=torch.int64,
        device=bits.device,
    )
    return (padded.reshape(-1, 8).to(dtype=torch.int64) * weights).sum(dim=1).to(
        dtype=torch.uint8
    )


def query_v1_signed_work_device_shell_endpoint_packed_identity_v2(
    *,
    bundle: V1SignedWorkDeviceShellDeviceBundleV2,
    endpoint_label: int,
    current_mask: HardMask,
):
    """Resolve one frozen rung to its packed physical action under a bounded borrow.

    Returns ``(receipt, action)`` where ``action`` is the exact
    ``V1PackedRemovalAction`` (child mask, removed count, delta hash) and the
    receipt binds it to the shell's device action commitment.  Refuses on a
    label outside the ladder, a foreign current mask, a count or commitment
    that does not replay, a removed route the current mask does not retain,
    or evicted label custody.
    """

    from .modern_v1_candidate_composition import V1PackedRemovalAction

    if type(bundle) is not V1SignedWorkDeviceShellDeviceBundleV2:
        raise TypeError("packed identity query requires the exact v2 shell bundle")
    if type(current_mask) is not HardMask:
        raise TypeError("packed identity query requires an exact HardMask")
    if type(endpoint_label) is not int:
        raise TypeError("endpoint label must be an exact int")
    receipt = bundle.receipt
    ladder = len(receipt.endpoint_counts)
    if endpoint_label < 1 or endpoint_label > ladder:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.ENDPOINT_QUERY_MISMATCH,
            "endpoint label lies outside the frozen ladder",
        )
    if current_mask.mask_hash != receipt.current_mask_hash:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
            "packed identity query received a mask other than the shell's current mask",
        )
    if int(current_mask.route_count) != int(receipt.route_count):
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.AXIS_MISMATCH,
            "current mask and shell route axes differ",
        )
    expected_count = int(receipt.endpoint_counts[endpoint_label - 1])
    commitment = _device_v2_action_commitment_hash(
        source_rounding_hash=_device_v2_source_rounding_hash(
            problem_hash=receipt.problem_hash,
            policy_hash=receipt.policy_hash,
            least_work_execution_hash=receipt.least_work_execution_hash,
            state_identity_hash=receipt.state_identity_hash,
            branch=receipt.branch,
            branch_receipt_hash=receipt.signed_work_branch_receipt_hash,
            endpoint_plan_hash=receipt.endpoint_plan_hash,
            current_mask_hash=receipt.current_mask_hash,
            structural_source_receipt_hash=receipt.structural_source_receipt_hash,
            source_geometry_hash=receipt.source_geometry_hash,
            source_registration_hash=receipt.source_registration_hash,
            registered_d_coordinate_hash=receipt.registered_d_coordinate_hash,
        ),
        endpoint_label=endpoint_label,
        cumulative_removed_count=expected_count,
    )
    if commitment != receipt.device_action_commitment_hashes[endpoint_label - 1]:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.ENDPOINT_QUERY_MISMATCH,
            "device action commitment does not replay from the shell receipt",
        )
    route_count = int(receipt.route_count)
    with bundle.label_handle.borrow(
        operation=V1SignedWorkDeviceShellTensorOperationV2.BOUNDED_ENDPOINT_QUERY
    ) as lease:
        labels = lease.tensor
        if labels.dtype != torch.uint8 or tuple(labels.shape) != (route_count,):
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
                "frozen labels differ from the shell's route axis",
            )
        removed = (labels > 0) & (labels <= endpoint_label)
        count = int(removed.sum().item())
        if count != expected_count:
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.ENDPOINT_QUERY_MISMATCH,
                "frozen labels do not reproduce the registered endpoint count",
            )
        parent_packed = torch.frombuffer(
            bytearray(current_mask.packed_bits), dtype=torch.uint8
        ).to(labels.device)
        retained = _device_unpack_big_endian_bits(parent_packed, route_count=route_count)
        if bool((removed & ~retained).any().item()):
            _refuse(
                V1SignedWorkDeviceShellRefusalCode.STALE_SOURCE,
                "frozen shell removes a route the current mask does not retain",
            )
        child_packed = _device_pack_big_endian_bits(retained & ~removed)
        child_bytes = bytes(child_packed.cpu().numpy().tobytes())
        del removed, retained, child_packed, parent_packed
    child = HardMask(
        route_axis_hash=current_mask.route_axis_hash,
        route_count=route_count,
        packed_bits=child_bytes,
    )
    action = V1PackedRemovalAction.from_resulting_mask(
        parent_mask=current_mask,
        resulting_mask=child,
    )
    if action.removed_count != expected_count:
        _refuse(
            V1SignedWorkDeviceShellRefusalCode.ENDPOINT_QUERY_MISMATCH,
            "packed delta count disagrees with the frozen endpoint count",
        )
    identity = V1SignedWorkDeviceShellEndpointPackedIdentityReceiptV2(
        shell_receipt_hash=receipt.receipt_hash,
        branch=receipt.branch,
        endpoint_label=endpoint_label,
        cumulative_removed_count=expected_count,
        device_action_commitment_hash=commitment,
        current_mask_hash=current_mask.mask_hash,
        route_axis_hash=receipt.route_axis_hash,
        route_count=route_count,
        resulting_mask_hash=child.mask_hash,
        resulting_mask_packed_sha256=hashlib.sha256(child.packed_bits).hexdigest(),
        removed_route_index_hash=action.removed_route_index_hash,
        action_hash=action.action_hash,
        packed_payload_d2h_bytes=len(child_bytes),
        host_mask_upload_bytes=len(current_mask.packed_bits),
        scalar_d2h_bytes=16,
    )
    return identity, action


__all__ = [
    "SIGNED_WORK_DEVICE_SHELL_ENDPOINT_PACKED_IDENTITY_SCHEMA_V2",
    "V1SignedWorkDeviceShellEndpointPackedIdentityReceiptV2",
    "query_v1_signed_work_device_shell_endpoint_packed_identity_v2",
    "SIGNED_WORK_DEVICE_SHELL_AUTHORITY",
    "SIGNED_WORK_DEVICE_SHELL_BRANCH_SCHEMA",
    "SIGNED_WORK_DEVICE_SHELL_BUNDLE_SCHEMA",
    "SIGNED_WORK_DEVICE_SHELL_DESIGN_SCHEMA",
    "SIGNED_WORK_DEVICE_SHELL_ENDPOINT_PLAN_SCHEMA",
    "SIGNED_WORK_DEVICE_SHELL_EXECUTION_SCHEMA",
    "SIGNED_WORK_DEVICE_SHELL_FAMILY_SCHEMA",
    "SIGNED_WORK_DEVICE_SHELL_HBM_SCHEMA",
    "SIGNED_WORK_DEVICE_SHELL_KEY_ID",
    "SIGNED_WORK_DEVICE_SHELL_PLAN_ORIGIN",
    "V1SignedWorkDeviceShellBranchIdentityReceipt",
    "V1SignedWorkDeviceShellEndpointPlanReceipt",
    "V1SignedWorkDeviceShellExecutionBundle",
    "V1SignedWorkDeviceShellExecutionDesign",
    "V1SignedWorkDeviceShellExecutionReceipt",
    "V1SignedWorkDeviceShellFamilyBundle",
    "V1SignedWorkDeviceShellFamilyReceipt",
    "V1SignedWorkDeviceShellHBMForecast",
    "V1SignedWorkDeviceShellRefusal",
    "V1SignedWorkDeviceShellRefusalCode",
    "SIGNED_WORK_DEVICE_ACTION_COMMITMENT_SCHEMA_V2",
    "SIGNED_WORK_DEVICE_SHELL_DEVICE_BUNDLE_SCHEMA_V2",
    "SIGNED_WORK_DEVICE_SHELL_DEVICE_FAMILY_SCHEMA_V2",
    "SIGNED_WORK_DEVICE_SHELL_DEVICE_HBM_SCHEMA_V2",
    "SIGNED_WORK_DEVICE_SHELL_DEVICE_SCHEMA_V2",
    "V1SignedWorkDeviceShellDeviceBundleV2",
    "V1SignedWorkDeviceShellDeviceFamilyV2",
    "V1SignedWorkDeviceShellDeviceHBMForecastV2",
    "V1SignedWorkDeviceShellDeviceReceiptV2",
    "V1SignedWorkDeviceShellTensorHandleV2",
    "V1SignedWorkDeviceShellTensorLeaseV2",
    "V1SignedWorkDeviceShellTensorOperationV2",
    "estimate_v1_signed_work_device_shell_cuda_hbm",
    "estimate_v1_signed_work_device_shell_device_hbm_v2",
    "estimate_v1_signed_work_device_shell_single_branch_hbm",
    "freeze_v1_signed_work_device_prefix_rank_path",
    "freeze_v1_signed_work_device_shell_device_branch_v2",
    "freeze_v1_signed_work_device_shell_device_family_v2",
    "freeze_v1_signed_work_device_shell_ladder",
    "freeze_v1_signed_work_device_shell_family",
    "freeze_v1_signed_work_device_shell_single_branch",
]
