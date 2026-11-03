"""Label-native packed directions for route-scale modern-v1 secants.

The historical directional provider accepts a ladder of materialized
``HardMask`` endpoints.  That representation is convenient for tiny tests but
is the wrong route-scale execution object: every endpoint repeats a complete
packed mask and each dose rebuilds another complete gate axis.

This module consumes the D-frozen prefix-rank custody through its public
bounded endpoint query, retains only the resulting uint8 first-inclusion
labels, and lazily constructs one packed dose-lane packet at a time.  It never
reads or transfers the prefix-rank vector, never materializes endpoint hard
masks, and grants no F/Q/R, calibration, oracle, commit, terminal, or Pareto
authority.  Production construction is CUDA-only.  A bounded CPU reference is
available solely for focused parity tests.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import InitVar, dataclass, field, fields as dataclass_fields
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from typing import Any, Final, Iterator, NoReturn, Sequence

import numpy as np

from mechanistic_systems.backends.torch_masks import CompiledTorchCircuitMask
from mechanistic_systems.contracts.content_address import canonical_json, content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.core.storage import RouteBlockSpec

from .modern_v1_device_prefix_rank import (
    DEVICE_PREFIX_RANK_DEFAULT_HBM_CEILING_FRACTION,
    DEVICE_PREFIX_RANK_DEFAULT_STREAM_CHUNK_SIZE,
    V1DevicePrefixRankEndpointQueryReceipt,
    V1DevicePrefixRankHandle,
    V1DevicePrefixRankReceipt,
)
from .modern_v1_device_shell_ladder import (
    DEVICE_SHELL_LABEL_COMMITMENT_DOMAIN,
    V1DeviceShellBorrowOperation,
    V1DeviceShellFrozenActionReceipt,
    V1DeviceShellFrozenLadderReceipt,
    V1DeviceShellLabelHandle,
    V1DeviceShellRefusal,
)
from .modern_v1_directional_remainder import DIRECTIONAL_SECANT_DOSES
from .modern_v1_signed_work_label_native_c_bridge import (
    V1SignedWorkLabelNativeCBranchReceipt,
)

try:  # pragma: no cover - availability is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


LABEL_NATIVE_DIRECTION_POLICY_SCHEMA: Final = (
    "partition_free_v1_label_native_packed_direction_policy_v1"
)
LABEL_NATIVE_DIRECTION_DESIGN_SCHEMA: Final = (
    "partition_free_v1_label_native_packed_direction_design_v1"
)
LABEL_NATIVE_DIRECTION_HBM_SCHEMA: Final = (
    "partition_free_v1_label_native_packed_direction_hbm_forecast_v1"
)
LABEL_NATIVE_DIRECTION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_label_native_packed_direction_receipt_v1"
)
LABEL_NATIVE_QUALIFIED_DIRECTION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_label_native_packed_direction_receipt_v2_qualified_lanes"
)
LABEL_NATIVE_QUALIFIED_LANE_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_label_native_pre_C_qualified_lane_registration_v1"
)
LABEL_NATIVE_DIRECTION_AUTHORITY: Final = (
    "label_native_direction_execution_only_no_FQR_CKO_commit_terminal_or_pareto_v1"
)
LABEL_NATIVE_DIRECTION_ALGORITHM_ID: Final = (
    "frozen_uint8_first_inclusion_to_lazy_complete_action_dose_packets_v1"
)
LABEL_NATIVE_CANONICAL_ACTION_IDENTITY: Final = (
    "pre_C_frozen_shell_canonical_packed_action_axis_v1"
)
LABEL_NATIVE_ACTION_AXIS_SCHEMA: Final = (
    "partition_free_v1_label_native_canonical_action_axis_v1"
)
LABEL_NATIVE_SELECTED_CARRIER_SCHEMA: Final = (
    "partition_free_v1_label_native_selected_packed_carrier_v1"
)
LABEL_NATIVE_SELECTED_CARRIER_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_label_native_selected_packed_carrier_receipt_v1"
)

_POLICY_PREFIX: Final = "partition-free-v1-label-native-direction-policy-"
_DESIGN_PREFIX: Final = "partition-free-v1-label-native-direction-design-"
_FORECAST_PREFIX: Final = "partition-free-v1-label-native-direction-forecast-"
_DIRECTION_PREFIX: Final = "partition-free-v1-label-native-direction-"
_ACTION_AXIS_PREFIX: Final = "partition-free-v1-label-native-action-axis-"
_SELECTED_CARRIER_RECEIPT_PREFIX: Final = (
    "partition-free-v1-label-native-selected-carrier-receipt-"
)
_RECEIPT_PREFIX: Final = "partition-free-v1-label-native-direction-receipt-"
_QUALIFIED_LANE_PREFIX: Final = (
    "partition-free-v1-label-native-qualified-lane-registration-"
)
_LABEL_COMMITMENT_DOMAIN: Final = (
    "partition_free_v1_device_prefix_rank_query_labels_v1"
)
_LABEL_COMMITMENT_PREFIX: Final = (
    "partition-free-v1-device-prefix-query-labels-"
)
_SHELL_LABEL_COMMITMENT_PREFIX: Final = (
    "partition-free-v1-device-shell-labels-"
)
_F32_BYTES: Final = np.dtype(np.float32).itemsize
_I64_BYTES: Final = np.dtype(np.int64).itemsize
_BOOL_BYTES: Final = np.dtype(np.bool_).itemsize


class V1LabelNativeDirectionRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    STALE_SOURCE = "stale_source"
    CUDA_REQUIRED = "cuda_required"
    CPU_REFERENCE_LIMIT_EXCEEDED = "cpu_reference_limit_exceeded"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    LABEL_IDENTITY_MISMATCH = "label_identity_mismatch"
    CAUSAL_ORDER_VIOLATION = "causal_order_violation"
    ACTION_IDENTITY_MISMATCH = "action_identity_mismatch"
    QUALIFIED_LANE_MISMATCH = "qualified_lane_mismatch"
    PACKET_LEASE_ACTIVE = "packet_lease_active"
    PACKET_RELEASED = "packet_released"
    HANDLE_EVICTED = "handle_evicted"
    AUTHORITY_VIOLATION = "authority_violation"


class V1LabelNativeDirectionRefusal(RuntimeError):
    def __init__(
        self,
        code: V1LabelNativeDirectionRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1LabelNativeDirectionRefusalCode:
            raise TypeError("label-native refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1LabelNativeDirectionRefusalCode,
    message: str,
) -> NoReturn:
    raise V1LabelNativeDirectionRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1LabelNativeDirectionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1LabelNativeDirectionRefusalCode.INVALID_TYPE,
            f"{name} must be one positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1LabelNativeDirectionRefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1LabelNativeDirectionRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1LabelNativeDirectionRefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be one real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0 or result > 0.85:
        raise ValueError(f"{name} must lie in (0, 0.85]")
    return result


def label_native_direction_provider_source_hash() -> str:
    """Return the exact source identity registered by a direction design."""

    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class V1LabelNativePackedDirectionPolicy:
    maximum_requested_packed_lanes: int = 8
    route_validation_chunk_size: int = (
        DEVICE_PREFIX_RANK_DEFAULT_STREAM_CHUNK_SIZE
    )
    hbm_ceiling_fraction: float = (
        DEVICE_PREFIX_RANK_DEFAULT_HBM_CEILING_FRACTION
    )
    private_cpu_route_limit: int = 4096
    private_cpu_workspace_ceiling_bytes: int = 1 << 30
    production_cuda_required: bool = True
    allow_private_cpu_reference: bool = False
    schema: str = LABEL_NATIVE_DIRECTION_POLICY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "maximum_requested_packed_lanes",
            "route_validation_chunk_size",
            "private_cpu_route_limit",
            "private_cpu_workspace_ceiling_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "hbm_ceiling_fraction",
            _fraction(self.hbm_ceiling_fraction, name="HBM ceiling fraction"),
        )
        for name in ("production_cuda_required", "allow_private_cpu_reference"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an exact bool")
        if self.production_cuda_required and self.allow_private_cpu_reference:
            raise ValueError("production CUDA policy cannot enable CPU reference mode")
        if self.schema != LABEL_NATIVE_DIRECTION_POLICY_SCHEMA:
            raise ValueError("unsupported label-native direction policy schema")

    @property
    def policy_hash(self) -> str:
        return content_hash(self, prefix=_POLICY_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LabelNativeCanonicalActionAxisRegistration:
    """Exact canonical packed-action axis frozen before C measurement."""

    source_shell_ladder: V1DeviceShellFrozenLadderReceipt
    action_identity_scheme: str = LABEL_NATIVE_CANONICAL_ACTION_IDENTITY
    authority_scope: str = LABEL_NATIVE_DIRECTION_AUTHORITY
    schema: str = LABEL_NATIVE_ACTION_AXIS_SCHEMA

    def __post_init__(self) -> None:
        if type(self.source_shell_ladder) is not V1DeviceShellFrozenLadderReceipt:
            raise TypeError("action-axis registration requires an exact shell receipt")
        for name in ("action_identity_scheme", "authority_scope"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if (
            self.action_identity_scheme != LABEL_NATIVE_CANONICAL_ACTION_IDENTITY
            or self.authority_scope != LABEL_NATIVE_DIRECTION_AUTHORITY
            or self.schema != LABEL_NATIVE_ACTION_AXIS_SCHEMA
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.ACTION_IDENTITY_MISMATCH,
                "canonical action-axis registration changed its pre-C identity",
            )

    @property
    def endpoint_counts(self) -> tuple[int, ...]:
        return self.source_shell_ladder.cumulative_counts

    @property
    def endpoint_action_hashes(self) -> tuple[str, ...]:
        return tuple(
            action.packed_action_hash for action in self.source_shell_ladder.actions
        )

    @property
    def source_action_receipt_hashes(self) -> tuple[str, ...]:
        return tuple(action.receipt_hash for action in self.source_shell_ladder.actions)

    @property
    def registration_hash(self) -> str:
        return content_hash(self, prefix=_ACTION_AXIS_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LabelNativeQualifiedLaneRegistration:
    """Pre-C exact Q-safe execution subset over one complete source ladder.

    The resident uint8 labels retain the complete first-inclusion source
    ladder.  This compact registration merely maps the provider's contiguous
    execution labels ``1..E`` onto the source action labels that the complete
    hard-F/Q replay proved Q-safe.  It is derived entirely from the pre-C
    bridge and therefore cannot widen, reorder, or reinterpret that set.
    """

    action_axis_registration: V1LabelNativeCanonicalActionAxisRegistration
    compact_C_bridge_branch_receipt: V1SignedWorkLabelNativeCBranchReceipt = field(
        repr=False,
        compare=False,
    )
    branch_value: str = field(init=False)
    compact_C_bridge_branch_receipt_hash: str = field(init=False)
    action_axis_registration_hash: str = field(init=False)
    complete_source_endpoint_counts: tuple[int, ...] = field(init=False)
    complete_source_action_receipt_hashes: tuple[str, ...] = field(init=False)
    complete_source_action_hashes: tuple[str, ...] = field(init=False)
    compact_provider_endpoint_labels: tuple[int, ...] = field(init=False)
    source_endpoint_labels: tuple[int, ...] = field(init=False)
    endpoint_counts: tuple[int, ...] = field(init=False)
    endpoint_action_receipt_hashes: tuple[str, ...] = field(init=False)
    endpoint_action_hashes: tuple[str, ...] = field(init=False)
    action_crosswalk_receipt_hashes: tuple[str, ...] = field(init=False)
    canonical_physical_identity_hashes: tuple[str, ...] = field(init=False)
    route_axis_hash: str = field(init=False)
    route_count: int = field(init=False)
    parent_mask_hash: str = field(init=False)
    eligible_axis_frozen_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    robustly_Q_infeasible_actions_included: bool = False
    provider_route_scale_label_or_index_D2H_bytes: int = 0
    candidate_composition_recomputed: bool = False
    authority_scope: str = LABEL_NATIVE_DIRECTION_AUTHORITY
    schema: str = LABEL_NATIVE_QUALIFIED_LANE_REGISTRATION_SCHEMA

    @staticmethod
    def _derive(
        action_axis: V1LabelNativeCanonicalActionAxisRegistration,
        bridge: V1SignedWorkLabelNativeCBranchReceipt,
    ) -> dict[str, object]:
        if type(action_axis) is not V1LabelNativeCanonicalActionAxisRegistration:
            raise TypeError("qualified lanes require an exact canonical action axis")
        if type(bridge) is not V1SignedWorkLabelNativeCBranchReceipt:
            raise TypeError("qualified lanes require an exact compact C branch receipt")
        bridge.assert_integrity()
        source = action_axis.source_shell_ladder
        if (
            action_axis.endpoint_counts != bridge.complete_source_endpoint_counts
            or action_axis.endpoint_action_hashes
            != bridge.complete_source_action_hashes
            or action_axis.source_action_receipt_hashes
            != bridge.complete_source_action_receipt_hashes
            or source.receipt_hash != bridge.shell_ladder_receipt_hash
            or source.route_axis_hash != bridge.route_axis_hash
            or source.route_count != bridge.route_count
            or source.parent_mask_hash != bridge.parent_mask_hash
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.QUALIFIED_LANE_MISMATCH,
                "qualified lanes differ from their complete pre-C shell source",
            )
        crosswalks = bridge.eligible_action_crosswalks
        if not crosswalks:
            _refuse(
                V1LabelNativeDirectionRefusalCode.QUALIFIED_LANE_MISMATCH,
                "qualified execution requires at least one pre-C Q-safe action",
            )
        source_labels = tuple(value.source_endpoint_label for value in crosswalks)
        if tuple(sorted(set(source_labels))) != source_labels:
            _refuse(
                V1LabelNativeDirectionRefusalCode.QUALIFIED_LANE_MISMATCH,
                "qualified source labels must preserve exact source order",
            )
        return {
            "branch_value": bridge.branch.value,
            "compact_C_bridge_branch_receipt_hash": bridge.receipt_hash,
            "action_axis_registration_hash": action_axis.registration_hash,
            "complete_source_endpoint_counts": bridge.complete_source_endpoint_counts,
            "complete_source_action_receipt_hashes": (
                bridge.complete_source_action_receipt_hashes
            ),
            "complete_source_action_hashes": bridge.complete_source_action_hashes,
            "compact_provider_endpoint_labels": tuple(
                range(1, len(crosswalks) + 1)
            ),
            "source_endpoint_labels": source_labels,
            "endpoint_counts": tuple(
                value.cumulative_removed_count for value in crosswalks
            ),
            "endpoint_action_receipt_hashes": tuple(
                value.source_action_receipt_hash for value in crosswalks
            ),
            "endpoint_action_hashes": tuple(
                value.packed_action_hash for value in crosswalks
            ),
            "action_crosswalk_receipt_hashes": tuple(
                value.receipt_hash for value in crosswalks
            ),
            "canonical_physical_identity_hashes": tuple(
                value.canonical_physical_identity_hash for value in crosswalks
            ),
            "route_axis_hash": bridge.route_axis_hash,
            "route_count": bridge.route_count,
            "parent_mask_hash": bridge.parent_mask_hash,
        }

    def __post_init__(self) -> None:
        expected = self._derive(
            self.action_axis_registration,
            self.compact_C_bridge_branch_receipt,
        )
        for name, value in expected.items():
            object.__setattr__(self, name, value)
        flags = (
            self.eligible_axis_frozen_before_C_outcomes,
            self.C_outcomes_seen,
            self.robustly_Q_infeasible_actions_included,
            self.candidate_composition_recomputed,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("qualified-lane flags must be exact booleans")
        object.__setattr__(
            self,
            "provider_route_scale_label_or_index_D2H_bytes",
            _nonnegative_int(
                self.provider_route_scale_label_or_index_D2H_bytes,
                name="qualified-provider route-scale D2H bytes",
            ),
        )
        if (
            flags != (True, False, False, False)
            or self.provider_route_scale_label_or_index_D2H_bytes != 0
            or self.authority_scope != LABEL_NATIVE_DIRECTION_AUTHORITY
            or self.schema != LABEL_NATIVE_QUALIFIED_LANE_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.AUTHORITY_VIOLATION,
                "qualified lanes crossed their pre-C execution-only authority",
            )

    def _payload(self) -> dict[str, object]:
        return {
            "action_axis_registration_hash": self.action_axis_registration_hash,
            "compact_C_bridge_branch_receipt_hash": (
                self.compact_C_bridge_branch_receipt_hash
            ),
            "branch_value": self.branch_value,
            "complete_source_endpoint_counts": self.complete_source_endpoint_counts,
            "complete_source_action_receipt_hashes": (
                self.complete_source_action_receipt_hashes
            ),
            "complete_source_action_hashes": self.complete_source_action_hashes,
            "compact_provider_endpoint_labels": (
                self.compact_provider_endpoint_labels
            ),
            "source_endpoint_labels": self.source_endpoint_labels,
            "endpoint_counts": self.endpoint_counts,
            "endpoint_action_receipt_hashes": self.endpoint_action_receipt_hashes,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "action_crosswalk_receipt_hashes": (
                self.action_crosswalk_receipt_hashes
            ),
            "canonical_physical_identity_hashes": (
                self.canonical_physical_identity_hashes
            ),
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "parent_mask_hash": self.parent_mask_hash,
            "eligible_axis_frozen_before_C_outcomes": (
                self.eligible_axis_frozen_before_C_outcomes
            ),
            "C_outcomes_seen": self.C_outcomes_seen,
            "robustly_Q_infeasible_actions_included": (
                self.robustly_Q_infeasible_actions_included
            ),
            "provider_route_scale_label_or_index_D2H_bytes": (
                self.provider_route_scale_label_or_index_D2H_bytes
            ),
            "candidate_composition_recomputed": (
                self.candidate_composition_recomputed
            ),
            "authority_scope": self.authority_scope,
            "schema": self.schema,
        }

    @property
    def registration_hash(self) -> str:
        return content_hash(self._payload(), prefix=_QUALIFIED_LANE_PREFIX)

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        expected = self._derive(
            self.action_axis_registration,
            self.compact_C_bridge_branch_receipt,
        )
        if any(getattr(self, name) != value for name, value in expected.items()):
            _refuse(
                V1LabelNativeDirectionRefusalCode.QUALIFIED_LANE_MISMATCH,
                "qualified lane registration was mutated or substituted",
            )
        flags = (
            self.eligible_axis_frozen_before_C_outcomes,
            self.C_outcomes_seen,
            self.robustly_Q_infeasible_actions_included,
            self.candidate_composition_recomputed,
        )
        if (
            any(type(value) is not bool for value in flags)
            or flags != (True, False, False, False)
            or isinstance(
                self.provider_route_scale_label_or_index_D2H_bytes,
                bool,
            )
            or not isinstance(
                self.provider_route_scale_label_or_index_D2H_bytes,
                Integral,
            )
            or self.provider_route_scale_label_or_index_D2H_bytes != 0
            or self.authority_scope != LABEL_NATIVE_DIRECTION_AUTHORITY
            or self.schema != LABEL_NATIVE_QUALIFIED_LANE_REGISTRATION_SCHEMA
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.AUTHORITY_VIOLATION,
                "qualified lane registration no longer has pre-C authority",
            )
        observed = self.registration_hash
        if expected_hash is not None and observed != _identity(
            expected_hash,
            name="expected qualified-lane registration hash",
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.QUALIFIED_LANE_MISMATCH,
                "qualified lane registration hash differs from its frozen source",
            )
        return observed

    def source_label_for_provider_label(self, provider_label: int) -> int:
        label = _positive_int(provider_label, name="compact provider endpoint label")
        if label > len(self.source_endpoint_labels):
            _refuse(
                V1LabelNativeDirectionRefusalCode.QUALIFIED_LANE_MISMATCH,
                "compact provider endpoint label lies outside the qualified axis",
            )
        return self.source_endpoint_labels[label - 1]


def register_v1_label_native_qualified_lanes(
    *,
    action_axis_registration: V1LabelNativeCanonicalActionAxisRegistration,
    compact_C_bridge_branch_receipt: V1SignedWorkLabelNativeCBranchReceipt,
    conditioned_on_C_outcomes: bool,
) -> V1LabelNativeQualifiedLaneRegistration:
    """Freeze the exact pre-C Q-safe source subset for provider execution."""

    if type(conditioned_on_C_outcomes) is not bool:
        raise TypeError("C-conditioning flag must be exact")
    if conditioned_on_C_outcomes:
        _refuse(
            V1LabelNativeDirectionRefusalCode.CAUSAL_ORDER_VIOLATION,
            "qualified provider lanes must be registered before C outcomes",
        )
    return V1LabelNativeQualifiedLaneRegistration(
        action_axis_registration=action_axis_registration,
        compact_C_bridge_branch_receipt=compact_C_bridge_branch_receipt,
    )


@dataclass(frozen=True, slots=True)
class V1LabelNativePackedDirectionDesign:
    design_id: str
    prefix_rank_receipt_hash: str
    source_rounding_hash: str
    route_axis_hash: str
    route_count: int
    parent_mask_hash: str
    endpoint_counts: tuple[int, ...]
    provider_source_hash: str
    action_axis_registration: V1LabelNativeCanonicalActionAxisRegistration
    action_identity_scheme: str = LABEL_NATIVE_CANONICAL_ACTION_IDENTITY
    doses: tuple[float, ...] = DIRECTIONAL_SECANT_DOSES
    endpoint_hard_masks_registered: bool = False
    candidate_composition_recomputed: bool = False
    authority_scope: str = LABEL_NATIVE_DIRECTION_AUTHORITY
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = LABEL_NATIVE_DIRECTION_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "design_id",
            "prefix_rank_receipt_hash",
            "source_rounding_hash",
            "route_axis_hash",
            "parent_mask_hash",
            "provider_source_hash",
            "action_identity_scheme",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        counts = tuple(
            _positive_int(value, name="endpoint count")
            for value in self.endpoint_counts
        )
        if (
            not counts
            or tuple(sorted(set(counts))) != counts
            or counts[-1] > self.route_count
        ):
            raise ValueError("label-native endpoint axis is invalid")
        object.__setattr__(self, "endpoint_counts", counts)
        if type(self.action_axis_registration) is not (
            V1LabelNativeCanonicalActionAxisRegistration
        ):
            raise TypeError("direction design requires an exact canonical action axis")
        source = self.action_axis_registration.source_shell_ladder
        if (
            self.action_axis_registration.endpoint_counts != counts
            or source.source_ladder_hash != self.source_rounding_hash
            or source.route_axis_hash != self.route_axis_hash
            or source.route_count != self.route_count
            or source.parent_mask_hash != self.parent_mask_hash
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.ACTION_IDENTITY_MISMATCH,
                "direction design differs from its exact frozen shell action axis",
            )
        doses = tuple(float(value) for value in self.doses)
        if doses != tuple(float(value) for value in DIRECTIONAL_SECANT_DOSES):
            raise ValueError("label-native directions require canonical secant doses")
        object.__setattr__(self, "doses", doses)
        flags = (
            self.endpoint_hard_masks_registered,
            self.candidate_composition_recomputed,
            self.fqr_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("label-native design flags must be exact booleans")
        if (
            any(flags)
            or self.provider_source_hash
            != label_native_direction_provider_source_hash()
            or self.action_identity_scheme
            != LABEL_NATIVE_CANONICAL_ACTION_IDENTITY
            or self.authority_scope != LABEL_NATIVE_DIRECTION_AUTHORITY
            or self.schema != LABEL_NATIVE_DIRECTION_DESIGN_SCHEMA
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.AUTHORITY_VIOLATION,
                "label-native direction design crossed its proposal boundary",
            )

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_DESIGN_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LabelNativeDirectionHBMForecast:
    backend_kind: str
    device_identity: str
    route_count: int
    endpoint_count: int
    maximum_route_block_size: int
    validation_chunk_size: int
    requested_packed_lanes: int
    selected_packed_lanes: int
    parent_gate_bytes: int
    packet_gate_bytes: int
    consumer_stacked_gate_bytes: int
    cpu_label_tensor_bytes: int
    compact_count_bytes: int
    execution_scratch_bytes: int
    construction_peak_bytes: int
    execution_peak_bytes: int
    forecast_incremental_bytes: int
    current_reserved_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_or_workspace_ceiling_bytes: int
    hbm_ceiling_fraction: float
    admitted: bool
    schema: str = LABEL_NATIVE_DIRECTION_HBM_SCHEMA
    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _admission_memory: tuple[int, int] | None) -> None:
        available, allocated = _admission_memory or (
            self.device_total_or_workspace_ceiling_bytes, self.current_reserved_bytes,
        )
        if self.backend_kind not in {"numpy_reference", "torch_cuda"}:
            raise ValueError("unsupported direction forecast backend")
        _identity(self.device_identity, name="device identity")
        for name in (
            "route_count",
            "endpoint_count",
            "maximum_route_block_size",
            "validation_chunk_size",
            "requested_packed_lanes",
            "selected_packed_lanes",
            "parent_gate_bytes",
            "packet_gate_bytes",
            "consumer_stacked_gate_bytes",
            "cpu_label_tensor_bytes",
            "compact_count_bytes",
            "execution_scratch_bytes",
            "construction_peak_bytes",
            "execution_peak_bytes",
            "forecast_incremental_bytes",
            "current_reserved_bytes",
            "forecast_peak_reserved_bytes",
            "device_total_or_workspace_ceiling_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if (
            self.route_count <= 0
            or self.endpoint_count <= 0
            or self.maximum_route_block_size <= 0
            or self.requested_packed_lanes <= 0
            or self.selected_packed_lanes <= 0
            or self.selected_packed_lanes > self.requested_packed_lanes
        ):
            raise ValueError("direction forecast axes/lanes are invalid")
        fraction = _fraction(
            self.hbm_ceiling_fraction,
            name="forecast HBM ceiling fraction",
        )
        expected_parent = self.route_count * _F32_BYTES
        expected_packet = (
            self.route_count * self.selected_packed_lanes * _F32_BYTES
        )
        expected_cpu_label = (
            self.route_count if self.backend_kind == "numpy_reference" else 0
        )
        expected_compact = (self.endpoint_count + 1) * _I64_BYTES
        expected_scratch = self.maximum_route_block_size * 4 * _BOOL_BYTES
        expected_construction = self.route_count * (
            _F32_BYTES + _BOOL_BYTES
        ) + self.validation_chunk_size * (
            _I64_BYTES + 4 * _BOOL_BYTES
        ) + expected_compact
        expected_execution = (
            expected_parent
            + 2 * expected_packet
            + expected_scratch
            + expected_cpu_label
            + expected_compact
        )
        expected_incremental = max(expected_construction, expected_execution)
        if (
            self.parent_gate_bytes != expected_parent
            or self.packet_gate_bytes != expected_packet
            or self.consumer_stacked_gate_bytes != expected_packet
            or self.cpu_label_tensor_bytes != expected_cpu_label
            or self.compact_count_bytes != expected_compact
            or self.execution_scratch_bytes != expected_scratch
            or self.construction_peak_bytes != expected_construction
            or self.execution_peak_bytes != expected_execution
            or self.forecast_incremental_bytes != expected_incremental
            or self.forecast_peak_reserved_bytes
            != self.current_reserved_bytes + expected_incremental
            or self.admitted
            != (
                allocated + expected_incremental
                <= self.device_total_or_workspace_ceiling_bytes
                and expected_incremental <= available
            )
            or fraction != self.hbm_ceiling_fraction
            or self.schema != LABEL_NATIVE_DIRECTION_HBM_SCHEMA
        ):
            raise ValueError("label-native direction forecast does not replay")

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_FORECAST_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LabelNativeDirectionLaneIdentity:
    endpoint_label: int
    endpoint_count: int
    endpoint_action_hash: str
    dose: float
    direction_hash: str
    lane_key: str
    source_endpoint_label: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "endpoint_label",
            _positive_int(self.endpoint_label, name="endpoint label"),
        )
        object.__setattr__(
            self,
            "endpoint_count",
            _positive_int(self.endpoint_count, name="endpoint count"),
        )
        source_label = (
            self.endpoint_label
            if self.source_endpoint_label is None
            else _positive_int(
                self.source_endpoint_label,
                name="source endpoint label",
            )
        )
        object.__setattr__(self, "source_endpoint_label", source_label)
        for name in ("endpoint_action_hash", "direction_hash", "lane_key"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        dose = float(self.dose)
        if dose not in DIRECTIONAL_SECANT_DOSES:
            raise ValueError("direction lane dose is not canonical")
        object.__setattr__(self, "dose", dose)


_PACKET_CONSTRUCTION_KEY: Final = object()
_PARENT_PACKET_CONSTRUCTION_KEY: Final = object()


class V1LabelNativeParentLanePacket:
    """One explicit lifetime lease over the provider's compiled parent lane."""

    __slots__ = (
        "_compiled_mask",
        "_lane_key",
        "_owner",
        "_owner_token",
        "_released",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        _construction_key: object,
        owner: "V1LabelNativePackedDirectionProvider",
        owner_token: object,
        lane_key: str,
        compiled_mask: CompiledTorchCircuitMask,
    ) -> None:
        if _construction_key is not _PARENT_PACKET_CONSTRUCTION_KEY:
            raise TypeError("parent packets must be issued by their provider")
        if type(owner) is not V1LabelNativePackedDirectionProvider:
            raise TypeError("parent packet owner must be the exact provider")
        if type(compiled_mask) is not CompiledTorchCircuitMask:
            raise TypeError("parent packet requires one exact compiled mask")
        self._owner = owner
        self._owner_token = owner_token
        self._lane_key = _identity(lane_key, name="parent lane key")
        self._compiled_mask = compiled_mask
        self._released = False

    def _require_live(self) -> None:
        if self._released:
            _refuse(
                V1LabelNativeDirectionRefusalCode.PACKET_RELEASED,
                "parent lane packet was already released",
            )

    @property
    def compiled_mask(self) -> CompiledTorchCircuitMask:
        self._require_live()
        return self._compiled_mask

    @property
    def lane_key(self) -> str:
        return self._lane_key

    @property
    def is_released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        compiled = self._compiled_mask
        seen: set[int] = set()
        for mapping in (
            compiled.parameter_overrides,
            compiled.residual_gates,
            compiled.gate_values,
            compiled.float_gate_values,
        ):
            if type(mapping) is dict and id(mapping) not in seen:
                mapping.clear()
                seen.add(id(mapping))
        owner = self._owner
        token = self._owner_token
        owner._release_packet(token)
        self._compiled_mask = None
        self._owner = None
        self._owner_token = None
        self._released = True

    def __enter__(self) -> "V1LabelNativeParentLanePacket":
        self._require_live()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.release()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("parent lane packet leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("parent lane packet leases are nonserializable")


class V1LabelNativeDirectionLanePacket:
    """One explicit memory lease over a bounded compiled direction packet.

    Releasing the packet clears its route-scale gate references and unlocks
    construction of the next packet.  The provider deliberately keeps no
    implicit auto-release path: retaining a packet must fail closed instead
    of silently invalidating gates still held by a consumer.
    """

    __slots__ = (
        "_compiled_masks",
        "_lane_identities",
        "_owner",
        "_owner_token",
        "_packet_index",
        "_released",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        _construction_key: object,
        owner: "V1LabelNativePackedDirectionProvider",
        owner_token: object,
        packet_index: int,
        lane_identities: tuple[V1LabelNativeDirectionLaneIdentity, ...],
        compiled_masks: tuple[CompiledTorchCircuitMask, ...],
    ) -> None:
        if _construction_key is not _PACKET_CONSTRUCTION_KEY:
            raise TypeError("direction packets must be issued by their provider")
        if type(owner) is not V1LabelNativePackedDirectionProvider:
            raise TypeError("direction packet owner must be the exact provider")
        index = _nonnegative_int(packet_index, name="packet index")
        identities = tuple(lane_identities)
        masks = tuple(compiled_masks)
        if (
            not identities
            or len(identities) != len(masks)
            or any(
                type(value) is not V1LabelNativeDirectionLaneIdentity
                for value in identities
            )
            or any(type(value) is not CompiledTorchCircuitMask for value in masks)
        ):
            raise TypeError("direction packet requires aligned exact lane values")
        self._packet_index = index
        self._lane_identities = identities
        self._compiled_masks = masks
        self._owner = owner
        self._owner_token = owner_token
        self._released = False

    def _require_live(self) -> None:
        if self._released:
            _refuse(
                V1LabelNativeDirectionRefusalCode.PACKET_RELEASED,
                "direction packet was already released",
            )

    @property
    def packet_index(self) -> int:
        return self._packet_index

    @property
    def lane_identities(self) -> tuple[V1LabelNativeDirectionLaneIdentity, ...]:
        return self._lane_identities

    @property
    def compiled_masks(self) -> tuple[CompiledTorchCircuitMask, ...]:
        self._require_live()
        return self._compiled_masks

    @property
    def is_released(self) -> bool:
        return self._released

    @property
    def lane_keys(self) -> tuple[str, ...]:
        return tuple(value.lane_key for value in self._lane_identities)

    def release(self) -> None:
        if self._released:
            return
        owner = self._owner
        token = self._owner_token
        for compiled in self._compiled_masks:
            seen: set[int] = set()
            for mapping in (
                compiled.parameter_overrides,
                compiled.residual_gates,
                compiled.gate_values,
                compiled.float_gate_values,
            ):
                if type(mapping) is dict and id(mapping) not in seen:
                    mapping.clear()
                    seen.add(id(mapping))
        owner._release_packet(token)
        self._compiled_masks = ()
        self._owner = None
        self._owner_token = None
        self._released = True

    def __enter__(self) -> "V1LabelNativeDirectionLanePacket":
        self._require_live()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.release()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("direction packet leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("direction packet leases are nonserializable")


@dataclass(frozen=True, slots=True)
class V1LabelNativePackedDirectionReceipt:
    design_hash: str
    policy_hash: str
    prefix_rank_receipt_hash: str
    endpoint_query_hash: str
    resulting_label_commitment: str
    forecast_hash: str
    route_axis_hash: str
    route_count: int
    parent_mask_hash: str
    endpoint_counts: tuple[int, ...]
    endpoint_action_hashes: tuple[str, ...]
    endpoint_direction_hashes: tuple[str, ...]
    action_axis_registration: V1LabelNativeCanonicalActionAxisRegistration
    backend_kind: str
    device_identity: str
    selected_packed_lanes: int
    packet_sizes: tuple[int, ...]
    provider_owns_label_handle: bool
    qualified_lane_registration: V1LabelNativeQualifiedLaneRegistration | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    action_identity_scheme: str = LABEL_NATIVE_CANONICAL_ACTION_IDENTITY
    endpoint_query_conditioned_on_c_outcomes: bool = False
    label_commitment_verified: bool = True
    canonical_action_axis_label_identity_verified: bool = True
    label_native_first_inclusion_used: bool = True
    endpoint_hard_masks_materialized: bool = False
    full_prefix_ranking_copied_to_host: bool = False
    candidate_composition_recomputed: bool = False
    authority_scope: str = LABEL_NATIVE_DIRECTION_AUTHORITY
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = LABEL_NATIVE_DIRECTION_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "policy_hash",
            "prefix_rank_receipt_hash",
            "endpoint_query_hash",
            "resulting_label_commitment",
            "forecast_hash",
            "route_axis_hash",
            "parent_mask_hash",
            "device_identity",
            "action_identity_scheme",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        object.__setattr__(
            self,
            "selected_packed_lanes",
            _positive_int(self.selected_packed_lanes, name="selected packed lanes"),
        )
        counts = tuple(
            _positive_int(value, name="endpoint count")
            for value in self.endpoint_counts
        )
        actions = tuple(
            _identity(value, name="endpoint action hash")
            for value in self.endpoint_action_hashes
        )
        directions = tuple(
            _identity(value, name="endpoint direction hash")
            for value in self.endpoint_direction_hashes
        )
        packets = tuple(
            _positive_int(value, name="packet size") for value in self.packet_sizes
        )
        qualified = self.qualified_lane_registration
        if qualified is None:
            executed_endpoint_count = len(counts)
            expected_schema = LABEL_NATIVE_DIRECTION_RECEIPT_SCHEMA
        else:
            if type(qualified) is not V1LabelNativeQualifiedLaneRegistration:
                raise TypeError(
                    "qualified direction receipt requires an exact lane registration"
                )
            qualified.assert_integrity()
            executed_endpoint_count = len(qualified.source_endpoint_labels)
            expected_schema = LABEL_NATIVE_QUALIFIED_DIRECTION_RECEIPT_SCHEMA
        expected_packets = tuple(
            min(
                self.selected_packed_lanes,
                executed_endpoint_count * len(DIRECTIONAL_SECANT_DOSES) - start,
            )
            for start in range(
                0,
                executed_endpoint_count * len(DIRECTIONAL_SECANT_DOSES),
                self.selected_packed_lanes,
            )
        )
        if (
            not counts
            or tuple(sorted(set(counts))) != counts
            or counts[-1] > self.route_count
            or len(counts) != len(actions)
            or len(counts) != len(directions)
            or packets != expected_packets
            or self.backend_kind not in {"numpy_reference", "torch_cuda"}
        ):
            raise ValueError("label-native direction receipt axes do not replay")
        if type(self.action_axis_registration) is not (
            V1LabelNativeCanonicalActionAxisRegistration
        ):
            raise TypeError("direction receipt requires an exact canonical action axis")
        action_axis = self.action_axis_registration
        source = action_axis.source_shell_ladder
        expected_actions = action_axis.endpoint_action_hashes
        expected_directions = _direction_hashes_for_coordinates(
            design_hash=self.design_hash,
            endpoint_query_hash=self.endpoint_query_hash,
            resulting_label_commitment=self.resulting_label_commitment,
            parent_mask_hash=self.parent_mask_hash,
            endpoint_counts=counts,
            endpoint_action_hashes=expected_actions,
        )
        if (
            self.action_identity_scheme
            != LABEL_NATIVE_CANONICAL_ACTION_IDENTITY
            or action_axis.endpoint_counts != counts
            or source.route_axis_hash != self.route_axis_hash
            or source.route_count != self.route_count
            or source.parent_mask_hash != self.parent_mask_hash
            or actions != expected_actions
            or directions != expected_directions
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.ACTION_IDENTITY_MISMATCH,
                "action/direction axes do not derive from the frozen pre-C query",
            )
        if qualified is not None and (
            qualified.action_axis_registration_hash
            != action_axis.registration_hash
            or qualified.complete_source_endpoint_counts != counts
            or qualified.complete_source_action_hashes != actions
            or qualified.route_axis_hash != self.route_axis_hash
            or qualified.route_count != self.route_count
            or qualified.parent_mask_hash != self.parent_mask_hash
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.QUALIFIED_LANE_MISMATCH,
                "qualified execution receipt differs from its complete source axis",
            )
        object.__setattr__(self, "endpoint_counts", counts)
        object.__setattr__(self, "endpoint_action_hashes", actions)
        object.__setattr__(self, "endpoint_direction_hashes", directions)
        object.__setattr__(self, "packet_sizes", packets)
        flags = (
            self.provider_owns_label_handle,
            self.endpoint_query_conditioned_on_c_outcomes,
            self.label_commitment_verified,
            self.canonical_action_axis_label_identity_verified,
            self.label_native_first_inclusion_used,
            self.endpoint_hard_masks_materialized,
            self.full_prefix_ranking_copied_to_host,
            self.candidate_composition_recomputed,
            self.fqr_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("label-native receipt flags must be exact booleans")
        if (
            self.endpoint_query_conditioned_on_c_outcomes
            or not self.label_commitment_verified
            or not self.canonical_action_axis_label_identity_verified
            or not self.label_native_first_inclusion_used
            or any(flags[5:])
            or self.authority_scope != LABEL_NATIVE_DIRECTION_AUTHORITY
            or self.schema != expected_schema
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.AUTHORITY_VIOLATION,
                "label-native direction receipt crossed measurement preparation authority",
            )

    @property
    def receipt_hash(self) -> str:
        if self.qualified_lane_registration is not None:
            return content_hash(self, prefix=_RECEIPT_PREFIX)
        # Preserve the exact historical v1 identity when the additive
        # qualified-lane field is absent.
        payload = {
            item.name: getattr(self, item.name)
            for item in dataclass_fields(self)
            if item.name != "qualified_lane_registration"
        }
        return content_hash(payload, prefix=_RECEIPT_PREFIX)

    @property
    def executed_source_endpoint_labels(self) -> tuple[int, ...]:
        qualified = self.qualified_lane_registration
        if qualified is None:
            return tuple(range(1, len(self.endpoint_counts) + 1))
        qualified.assert_integrity()
        return qualified.source_endpoint_labels

    @property
    def compact_provider_to_source_endpoint_labels(self) -> tuple[int, ...]:
        return self.executed_source_endpoint_labels

    @property
    def executed_endpoint_counts(self) -> tuple[int, ...]:
        return tuple(
            self.endpoint_counts[label - 1]
            for label in self.executed_source_endpoint_labels
        )

    @property
    def executed_endpoint_action_hashes(self) -> tuple[str, ...]:
        return tuple(
            self.endpoint_action_hashes[label - 1]
            for label in self.executed_source_endpoint_labels
        )


@dataclass(frozen=True, slots=True)
class V1LabelNativeSelectedEndpointPackedReceipt:
    provider_receipt_hash: str
    design_hash: str
    prefix_rank_receipt_hash: str
    endpoint_query_hash: str
    action_axis_registration_hash: str
    source_shell_ladder_receipt_hash: str
    source_action_receipt: V1DeviceShellFrozenActionReceipt
    endpoint_label: int
    endpoint_count: int
    expected_action_hash: str
    route_axis_hash: str
    route_count: int
    parent_mask_hash: str
    resulting_mask_packed_sha256: str
    resulting_mask_hash: str
    removed_route_index_hash: str
    backend_kind: str
    device_identity: str
    packed_payload_d2h_bytes: int
    raw_label_d2h_bytes: int = 0
    canonical_action_identity_replayed: bool = True
    query_and_shell_label_commitments_verified: bool = True
    no_packet_lease_active_at_materialization: bool = True
    hard_mask_object_returned: bool = False
    packed_child_carrier_materialized: bool = True
    authority_scope: str = LABEL_NATIVE_DIRECTION_AUTHORITY
    fqr_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = LABEL_NATIVE_SELECTED_CARRIER_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "provider_receipt_hash",
            "design_hash",
            "prefix_rank_receipt_hash",
            "endpoint_query_hash",
            "action_axis_registration_hash",
            "source_shell_ladder_receipt_hash",
            "expected_action_hash",
            "route_axis_hash",
            "parent_mask_hash",
            "resulting_mask_hash",
            "removed_route_index_hash",
            "device_identity",
            "authority_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "endpoint_label",
            _positive_int(self.endpoint_label, name="selected endpoint label"),
        )
        object.__setattr__(
            self,
            "endpoint_count",
            _positive_int(self.endpoint_count, name="selected endpoint count"),
        )
        object.__setattr__(
            self,
            "route_count",
            _positive_int(self.route_count, name="route count"),
        )
        for name in ("packed_payload_d2h_bytes", "raw_label_d2h_bytes"):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if type(self.source_action_receipt) is not V1DeviceShellFrozenActionReceipt:
            raise TypeError("selected carrier requires an exact source action receipt")
        action = self.source_action_receipt
        digest = self.resulting_mask_packed_sha256
        if (
            type(digest) is not str
            or len(digest) != 64
            or digest != digest.lower()
        ):
            raise ValueError("selected packed carrier requires a lowercase SHA-256")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError("selected packed carrier requires a lowercase SHA-256") from error
        expected_d2h = (
            (self.route_count + 7) // 8 if self.backend_kind == "torch_cuda" else 0
        )
        if (
            self.backend_kind not in {"numpy_reference", "torch_cuda"}
            or action.endpoint_label != self.endpoint_label
            or action.cumulative_removed_count != self.endpoint_count
            or action.packed_action_hash != self.expected_action_hash
            or action.resulting_mask_packed_sha256 != digest
            or action.resulting_mask_hash != self.resulting_mask_hash
            or action.removed_route_index_hash != self.removed_route_index_hash
            or self.packed_payload_d2h_bytes != expected_d2h
            or self.raw_label_d2h_bytes != 0
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.ACTION_IDENTITY_MISMATCH,
                "selected packed carrier does not replay its canonical source action",
            )
        flags = (
            self.canonical_action_identity_replayed,
            self.query_and_shell_label_commitments_verified,
            self.no_packet_lease_active_at_materialization,
            self.hard_mask_object_returned,
            self.packed_child_carrier_materialized,
            self.fqr_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.mask_commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("selected carrier flags must be exact booleans")
        if (
            flags[:5] != (True, True, True, False, True)
            or any(flags[5:])
            or self.authority_scope != LABEL_NATIVE_DIRECTION_AUTHORITY
            or self.schema != LABEL_NATIVE_SELECTED_CARRIER_RECEIPT_SCHEMA
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.AUTHORITY_VIOLATION,
                "selected packed carrier crossed its identity-only authority",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_SELECTED_CARRIER_RECEIPT_PREFIX)


@dataclass(frozen=True, slots=True)
class V1LabelNativeSelectedEndpointPackedCarrier:
    receipt: V1LabelNativeSelectedEndpointPackedReceipt
    packed_child_bits: bytes = field(repr=False, compare=False)
    schema: str = LABEL_NATIVE_SELECTED_CARRIER_SCHEMA

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1LabelNativeSelectedEndpointPackedReceipt:
            raise TypeError("selected packed carrier requires its exact receipt")
        packed = bytes(self.packed_child_bits)
        if (
            len(packed) != (self.receipt.route_count + 7) // 8
            or hashlib.sha256(packed).hexdigest()
            != self.receipt.resulting_mask_packed_sha256
            or HardMask(
                route_axis_hash=self.receipt.route_axis_hash,
                route_count=self.receipt.route_count,
                packed_bits=packed,
            ).mask_hash
            != self.receipt.resulting_mask_hash
            or self.schema != LABEL_NATIVE_SELECTED_CARRIER_SCHEMA
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.ACTION_IDENTITY_MISMATCH,
                "selected packed child bytes differ from their canonical source action",
            )
        object.__setattr__(self, "packed_child_bits", packed)


def _normalize_route_blocks(
    route_blocks: Sequence[RouteBlockSpec],
    *,
    route_count: int,
) -> tuple[RouteBlockSpec, ...]:
    blocks = tuple(route_blocks)
    if not blocks or any(type(block) is not RouteBlockSpec for block in blocks):
        raise TypeError("label-native directions require exact route blocks")
    expected = 0
    layer_ids: set[str] = set()
    for block in blocks:
        block.validate()
        if block.offset != expected or block.stop > route_count:
            raise ValueError("route blocks must be contiguous on the direction axis")
        if str(block.layer_id) in layer_ids:
            raise ValueError("route-block layer IDs must be unique")
        layer_ids.add(str(block.layer_id))
        expected = block.stop
    if expected != route_count:
        raise ValueError("route blocks do not cover the complete direction axis")
    return blocks


def _direct_mask_hash(mask: HardMask) -> str:
    if type(mask) is not HardMask:
        raise TypeError("label-native directions require an exact parent HardMask")
    return HardMask(
        route_axis_hash=mask.route_axis_hash,
        route_count=mask.route_count,
        packed_bits=mask.packed_bits,
    ).mask_hash


def _stream_digest(*, domain: str, metadata: object) -> "hashlib._Hash":
    encoded_domain = domain.encode("ascii")
    encoded_metadata = canonical_json(metadata).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(encoded_domain).to_bytes(8, "big", signed=False))
    digest.update(encoded_domain)
    digest.update(len(encoded_metadata).to_bytes(8, "big", signed=False))
    digest.update(encoded_metadata)
    return digest


def _replay_label_commitment_bounded(
    handle: V1DeviceShellLabelHandle,
    *,
    route_count: int,
    chunk_size: int,
) -> str:
    digest = _stream_digest(
        domain=_LABEL_COMMITMENT_DOMAIN,
        metadata={"dtype": "uint8", "shape": (route_count,)},
    )
    chunk = min(_positive_int(chunk_size, name="label chunk size"), route_count)
    for start in range(0, route_count, chunk):
        stop = min(route_count, start + chunk)
        labels = handle.copy_host_chunk(start, stop)
        digest.update(memoryview(np.ascontiguousarray(labels)).cast("B"))
    return _LABEL_COMMITMENT_PREFIX + digest.hexdigest()


def _replay_shell_label_commitment_bounded(
    handle: V1DeviceShellLabelHandle,
    *,
    route_axis_hash: str,
    route_count: int,
    endpoint_count: int,
    chunk_size: int,
) -> str:
    digest = _stream_digest(
        domain=DEVICE_SHELL_LABEL_COMMITMENT_DOMAIN,
        metadata={
            "route_axis_hash": route_axis_hash,
            "route_count": route_count,
            "endpoint_count": endpoint_count,
        },
    )
    chunk = min(_positive_int(chunk_size, name="label chunk size"), route_count)
    for start in range(0, route_count, chunk):
        stop = min(route_count, start + chunk)
        labels = handle.copy_host_chunk(start, stop)
        digest.update(memoryview(np.ascontiguousarray(labels)).cast("B"))
    return _SHELL_LABEL_COMMITMENT_PREFIX + digest.hexdigest()


def _direction_hashes_for_coordinates(
    *,
    design_hash: str,
    endpoint_query_hash: str,
    resulting_label_commitment: str,
    parent_mask_hash: str,
    endpoint_counts: tuple[int, ...],
    endpoint_action_hashes: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        content_hash(
            {
                "design_hash": design_hash,
                "prefix_rank_query_hash": endpoint_query_hash,
                "resulting_label_commitment": resulting_label_commitment,
                "parent_mask_hash": parent_mask_hash,
                "endpoint_label": label,
                "endpoint_count": count,
                "endpoint_action_hash": action_hash,
            },
            prefix=_DIRECTION_PREFIX,
        )
        for label, (count, action_hash) in enumerate(
            zip(endpoint_counts, endpoint_action_hashes, strict=True),
            start=1,
        )
    )


def _forecast_for_lanes(
    *,
    backend_kind: str,
    device_identity: str,
    route_count: int,
    endpoint_count: int,
    maximum_route_block_size: int,
    validation_chunk_size: int,
    requested_lanes: int,
    selected_lanes: int,
    current_reserved_bytes: int,
    ceiling_bytes: int,
    hbm_ceiling_fraction: float,
    _admission_memory: tuple[int, int] | None = None,
) -> V1LabelNativeDirectionHBMForecast:
    parent = route_count * _F32_BYTES
    packet = route_count * selected_lanes * _F32_BYTES
    cpu_labels = route_count if backend_kind == "numpy_reference" else 0
    compact_counts = (endpoint_count + 1) * _I64_BYTES
    scratch = maximum_route_block_size * 4 * _BOOL_BYTES
    construction = route_count * (_F32_BYTES + _BOOL_BYTES) + (
        validation_chunk_size * (_I64_BYTES + 4 * _BOOL_BYTES)
    ) + compact_counts
    execution = parent + 2 * packet + scratch + cpu_labels + compact_counts
    incremental = max(construction, execution)
    available, allocated = _admission_memory or (ceiling_bytes, current_reserved_bytes)
    return V1LabelNativeDirectionHBMForecast(
        backend_kind=backend_kind,
        device_identity=device_identity,
        route_count=route_count,
        endpoint_count=endpoint_count,
        maximum_route_block_size=maximum_route_block_size,
        validation_chunk_size=validation_chunk_size,
        requested_packed_lanes=requested_lanes,
        selected_packed_lanes=selected_lanes,
        parent_gate_bytes=parent,
        packet_gate_bytes=packet,
        consumer_stacked_gate_bytes=packet,
        cpu_label_tensor_bytes=cpu_labels,
        compact_count_bytes=compact_counts,
        execution_scratch_bytes=scratch,
        construction_peak_bytes=construction,
        execution_peak_bytes=execution,
        forecast_incremental_bytes=incremental,
        current_reserved_bytes=current_reserved_bytes,
        forecast_peak_reserved_bytes=current_reserved_bytes + incremental,
        device_total_or_workspace_ceiling_bytes=ceiling_bytes,
        hbm_ceiling_fraction=hbm_ceiling_fraction,
        admitted=allocated + incremental <= ceiling_bytes and incremental <= available,
        _admission_memory=_admission_memory,
    )


def _select_forecast(
    *,
    backend_kind: str,
    device_identity: str,
    route_count: int,
    endpoint_count: int,
    maximum_route_block_size: int,
    policy: V1LabelNativePackedDirectionPolicy,
    device_total_memory_bytes: int | None,
    current_reserved_bytes: int | None,
) -> V1LabelNativeDirectionHBMForecast:
    admission_memory = None
    if backend_kind == "torch_cuda":
        if torch is None:  # pragma: no cover - CUDA construction already guarded.
            raise RuntimeError("CUDA direction forecast requires PyTorch")
        device = torch.device(device_identity)
        total = (
            int(torch.cuda.get_device_properties(device).total_memory)
            if device_total_memory_bytes is None
            else _positive_int(
                device_total_memory_bytes,
                name="device total memory bytes",
            )
        )
        current = (
            int(torch.cuda.memory_reserved(device))
            if current_reserved_bytes is None
            else _nonnegative_int(
                current_reserved_bytes,
                name="current reserved bytes",
            )
        )
        ceiling = int(math.floor(policy.hbm_ceiling_fraction * total))
        if device_total_memory_bytes is None or current_reserved_bytes is None:
            admission_memory = (
                cuda_reusable_memory_bytes(device, framework=torch),
                int(torch.cuda.memory_allocated(device)),
            )
    else:
        current = 0
        ceiling = policy.private_cpu_workspace_ceiling_bytes
    validation_chunk_size = min(
        policy.route_validation_chunk_size,
        route_count,
    )
    reusable_forecast = None
    for lanes in range(policy.maximum_requested_packed_lanes, 0, -1):
        forecast = _forecast_for_lanes(
            backend_kind=backend_kind,
            device_identity=device_identity,
            route_count=route_count,
            endpoint_count=endpoint_count,
            maximum_route_block_size=maximum_route_block_size,
            validation_chunk_size=validation_chunk_size,
            requested_lanes=policy.maximum_requested_packed_lanes,
            selected_lanes=lanes,
            current_reserved_bytes=current,
            ceiling_bytes=ceiling,
            hbm_ceiling_fraction=policy.hbm_ceiling_fraction,
            _admission_memory=admission_memory,
        )
        if forecast.admitted:
            if current + forecast.forecast_incremental_bytes <= ceiling:
                return forecast
            if reusable_forecast is None:
                reusable_forecast = forecast
    if reusable_forecast is not None:
        return reusable_forecast
    _refuse(
        V1LabelNativeDirectionRefusalCode.HBM_FORECAST_EXCEEDED,
        "even one label-native direction lane exceeds the registered workspace",
    )


_PROVIDER_CONSTRUCTION_KEY: Final = object()
_PACKET_ITERATOR_CONSTRUCTION_KEY: Final = object()


class _V1LabelNativeDirectionPacketIterator:
    __slots__ = ("_next_start", "_packet_index", "_pairs", "_provider", "_width")
    __hash__ = None

    def __init__(
        self,
        *,
        _construction_key: object,
        provider: "V1LabelNativePackedDirectionProvider",
        pairs: tuple[tuple[int, int, int, str, str, float], ...],
        width: int,
    ) -> None:
        if _construction_key is not _PACKET_ITERATOR_CONSTRUCTION_KEY:
            raise TypeError("direction packet iterators must be issued by a provider")
        self._provider = provider
        self._pairs = pairs
        self._width = _positive_int(width, name="packet iterator width")
        self._next_start = 0
        self._packet_index = 0

    def __iter__(self) -> "_V1LabelNativeDirectionPacketIterator":
        return self

    def __next__(self) -> V1LabelNativeDirectionLanePacket:
        if self._next_start >= len(self._pairs):
            raise StopIteration
        packet = self._provider._build_packet(
            packet_index=self._packet_index,
            pairs=self._pairs[self._next_start : self._next_start + self._width],
        )
        self._next_start += len(packet.lane_identities)
        self._packet_index += 1
        return packet

    def __reduce__(self) -> NoReturn:
        raise TypeError("direction packet iterators are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("direction packet iterators are nonserializable")


class V1LabelNativePackedDirectionProvider:
    """Ephemeral owner of one parent gate and one lazy label-native query."""

    __slots__ = (
        "_active_packet_token",
        "_closed",
        "_label_handle",
        "_owns_label_handle",
        "_operation_lock",
        "_parent_gate",
        "_parent_packed_bits",
        "_qualified_lane_registration",
        "_qualified_lane_registration_hash",
        "_route_blocks",
        "design",
        "forecast",
        "policy",
        "prefix_rank_receipt",
        "query_receipt",
        "receipt",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        _construction_key: object,
        label_handle: V1DeviceShellLabelHandle,
        prefix_rank_receipt: V1DevicePrefixRankReceipt,
        query_receipt: V1DevicePrefixRankEndpointQueryReceipt,
        design: V1LabelNativePackedDirectionDesign,
        policy: V1LabelNativePackedDirectionPolicy,
        parent_mask: HardMask,
        route_blocks: tuple[RouteBlockSpec, ...],
        forecast: V1LabelNativeDirectionHBMForecast,
        label_commitment_verified: bool,
        owns_label_handle: bool,
        qualified_lane_registration: V1LabelNativeQualifiedLaneRegistration | None,
    ) -> None:
        if _construction_key is not _PROVIDER_CONSTRUCTION_KEY:
            raise TypeError("label-native providers must be issued by a factory")
        if torch is None:
            raise RuntimeError("label-native packed directions require PyTorch")
        self._label_handle = label_handle
        self._owns_label_handle = owns_label_handle
        self._operation_lock = RLock()
        self.prefix_rank_receipt = prefix_rank_receipt
        self.query_receipt = query_receipt
        self.design = design
        self.policy = policy
        self.forecast = forecast
        self._route_blocks = route_blocks
        self._active_packet_token = None
        self._closed = False
        self._parent_packed_bits = bytes(parent_mask.packed_bits)
        if qualified_lane_registration is not None and type(
            qualified_lane_registration
        ) is not V1LabelNativeQualifiedLaneRegistration:
            raise TypeError("provider requires an exact qualified-lane registration")
        self._qualified_lane_registration = qualified_lane_registration
        self._qualified_lane_registration_hash = (
            None
            if qualified_lane_registration is None
            else qualified_lane_registration.assert_integrity()
        )

        parent_axis = parent_mask.to_bool_array()
        device = torch.device(label_handle.device_identity)
        parent_bool = torch.as_tensor(
            np.array(parent_axis, dtype=np.bool_, order="C", copy=True),
            dtype=torch.bool,
            device=device,
        ).contiguous()
        try:
            self._validate_resident_labels(parent_bool)
            self._parent_gate = parent_bool.to(dtype=torch.float32)
        finally:
            del parent_bool

        source_ladder = design.action_axis_registration.source_shell_ladder
        if qualified_lane_registration is None:
            observed_shell_commitment = _replay_shell_label_commitment_bounded(
                label_handle,
                route_axis_hash=design.route_axis_hash,
                route_count=design.route_count,
                endpoint_count=len(design.endpoint_counts),
                chunk_size=policy.route_validation_chunk_size,
            )
            if observed_shell_commitment != source_ladder.label_commitment:
                _refuse(
                    V1LabelNativeDirectionRefusalCode.ACTION_IDENTITY_MISMATCH,
                    "query labels do not reproduce the canonical packed-action source",
                )
        elif (
            qualified_lane_registration.action_axis_registration_hash
            != design.action_axis_registration.registration_hash
            or qualified_lane_registration.complete_source_endpoint_counts
            != design.endpoint_counts
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.QUALIFIED_LANE_MISMATCH,
                "qualified provider registration differs from its source design",
            )

        actions = design.action_axis_registration.endpoint_action_hashes
        directions = _direction_hashes_for_coordinates(
            design_hash=design.design_hash,
            endpoint_query_hash=query_receipt.query_hash,
            resulting_label_commitment=query_receipt.resulting_label_commitment,
            parent_mask_hash=design.parent_mask_hash,
            endpoint_counts=design.endpoint_counts,
            endpoint_action_hashes=actions,
        )
        executed_endpoint_count = (
            len(design.endpoint_counts)
            if qualified_lane_registration is None
            else len(qualified_lane_registration.source_endpoint_labels)
        )
        total_lanes = executed_endpoint_count * len(design.doses)
        packets = tuple(
            min(forecast.selected_packed_lanes, total_lanes - start)
            for start in range(0, total_lanes, forecast.selected_packed_lanes)
        )
        self.receipt = V1LabelNativePackedDirectionReceipt(
            design_hash=design.design_hash,
            policy_hash=policy.policy_hash,
            prefix_rank_receipt_hash=prefix_rank_receipt.receipt_hash,
            endpoint_query_hash=query_receipt.query_hash,
            resulting_label_commitment=query_receipt.resulting_label_commitment,
            forecast_hash=forecast.forecast_hash,
            route_axis_hash=design.route_axis_hash,
            route_count=design.route_count,
            parent_mask_hash=design.parent_mask_hash,
            endpoint_counts=design.endpoint_counts,
            endpoint_action_hashes=actions,
            endpoint_direction_hashes=directions,
            action_axis_registration=design.action_axis_registration,
            backend_kind=label_handle.backend_kind,
            device_identity=label_handle.device_identity,
            selected_packed_lanes=forecast.selected_packed_lanes,
            packet_sizes=packets,
            provider_owns_label_handle=owns_label_handle,
            qualified_lane_registration=qualified_lane_registration,
            endpoint_query_conditioned_on_c_outcomes=(
                query_receipt.endpoint_query_conditioned_on_c_outcomes
            ),
            label_commitment_verified=label_commitment_verified,
            schema=(
                LABEL_NATIVE_DIRECTION_RECEIPT_SCHEMA
                if qualified_lane_registration is None
                else LABEL_NATIVE_QUALIFIED_DIRECTION_RECEIPT_SCHEMA
            ),
        )

    @classmethod
    def from_prefix_rank(
        cls,
        *,
        prefix_rank_handle: V1DevicePrefixRankHandle,
        design: V1LabelNativePackedDirectionDesign,
        parent_mask: HardMask,
        route_blocks: Sequence[RouteBlockSpec],
        conditioned_on_c_outcomes: bool,
        qualified_lane_registration: (
            V1LabelNativeQualifiedLaneRegistration | None
        ) = None,
        policy: V1LabelNativePackedDirectionPolicy = (
            V1LabelNativePackedDirectionPolicy()
        ),
        device_total_memory_bytes: int | None = None,
        current_reserved_bytes: int | None = None,
    ) -> "V1LabelNativePackedDirectionProvider":
        if type(prefix_rank_handle) is not V1DevicePrefixRankHandle:
            raise TypeError("prefix factory requires an exact rank handle")
        if type(conditioned_on_c_outcomes) is not bool:
            raise TypeError("C-conditioning flag must be exact")
        if conditioned_on_c_outcomes:
            _refuse(
                V1LabelNativeDirectionRefusalCode.CAUSAL_ORDER_VIOLATION,
                "direction labels must be frozen before any C-stream outcome",
            )
        prefix = prefix_rank_handle.receipt
        _preflight_sources(
            prefix=prefix,
            design=design,
            parent_mask=parent_mask,
            route_blocks=route_blocks,
            policy=policy,
        )
        if qualified_lane_registration is not None:
            if type(qualified_lane_registration) is not (
                V1LabelNativeQualifiedLaneRegistration
            ):
                raise TypeError(
                    "qualified provider construction requires an exact registration"
                )
            qualified_lane_registration.assert_integrity()
            if (
                qualified_lane_registration.action_axis_registration_hash
                != design.action_axis_registration.registration_hash
            ):
                _refuse(
                    V1LabelNativeDirectionRefusalCode.QUALIFIED_LANE_MISMATCH,
                    "qualified registration belongs to another source ladder",
                )
        labels, query = prefix_rank_handle.derive_endpoint_labels(
            design.endpoint_counts,
            conditioned_on_c_outcomes=conditioned_on_c_outcomes,
            stream_chunk_size=policy.route_validation_chunk_size,
            hbm_ceiling_fraction=policy.hbm_ceiling_fraction,
            device_total_memory_bytes=device_total_memory_bytes,
            current_reserved_bytes=current_reserved_bytes,
        )
        try:
            return _construct_provider(
                label_handle=labels,
                prefix=prefix,
                query=query,
                design=design,
                policy=policy,
                parent_mask=parent_mask,
                route_blocks=route_blocks,
                label_commitment_verified=True,
                owns_label_handle=True,
                qualified_lane_registration=qualified_lane_registration,
                device_total_memory_bytes=device_total_memory_bytes,
                current_reserved_bytes=None,
            )
        except BaseException:
            if not labels.is_evicted:
                labels.evict()
            raise

    @classmethod
    def from_qualified_prefix_rank(
        cls,
        *,
        prefix_rank_handle: V1DevicePrefixRankHandle,
        design: V1LabelNativePackedDirectionDesign,
        qualified_lane_registration: V1LabelNativeQualifiedLaneRegistration,
        parent_mask: HardMask,
        route_blocks: Sequence[RouteBlockSpec],
        conditioned_on_c_outcomes: bool,
        policy: V1LabelNativePackedDirectionPolicy = (
            V1LabelNativePackedDirectionPolicy()
        ),
        device_total_memory_bytes: int | None = None,
        current_reserved_bytes: int | None = None,
    ) -> "V1LabelNativePackedDirectionProvider":
        """Build the exact pre-C hard-Q-safe provider subset.

        The full source endpoint query remains resident; only the lanes bound
        by ``qualified_lane_registration`` are executed.
        """

        if type(qualified_lane_registration) is not (
            V1LabelNativeQualifiedLaneRegistration
        ):
            raise TypeError("qualified factory requires an exact registration")
        return cls.from_prefix_rank(
            prefix_rank_handle=prefix_rank_handle,
            design=design,
            parent_mask=parent_mask,
            route_blocks=route_blocks,
            conditioned_on_c_outcomes=conditioned_on_c_outcomes,
            qualified_lane_registration=qualified_lane_registration,
            policy=policy,
            device_total_memory_bytes=device_total_memory_bytes,
            current_reserved_bytes=current_reserved_bytes,
        )

    @classmethod
    def from_shell_query(
        cls,
        *,
        label_handle: V1DeviceShellLabelHandle,
        prefix_rank_receipt: V1DevicePrefixRankReceipt,
        query_receipt: V1DevicePrefixRankEndpointQueryReceipt,
        design: V1LabelNativePackedDirectionDesign,
        parent_mask: HardMask,
        route_blocks: Sequence[RouteBlockSpec],
        policy: V1LabelNativePackedDirectionPolicy = (
            V1LabelNativePackedDirectionPolicy()
        ),
        device_total_memory_bytes: int | None = None,
        current_reserved_bytes: int | None = None,
    ) -> "V1LabelNativePackedDirectionProvider":
        if type(label_handle) is not V1DeviceShellLabelHandle:
            raise TypeError("shell-query factory requires an exact label handle")
        if type(prefix_rank_receipt) is not V1DevicePrefixRankReceipt:
            raise TypeError("shell-query factory requires an exact rank receipt")
        if type(query_receipt) is not V1DevicePrefixRankEndpointQueryReceipt:
            raise TypeError("shell-query factory requires an exact query receipt")
        if query_receipt.endpoint_query_conditioned_on_c_outcomes:
            _refuse(
                V1LabelNativeDirectionRefusalCode.CAUSAL_ORDER_VIOLATION,
                "a C-conditioned shell query cannot authorize C measurement",
            )
        _preflight_sources(
            prefix=prefix_rank_receipt,
            design=design,
            parent_mask=parent_mask,
            route_blocks=route_blocks,
            policy=policy,
        )
        observed = _replay_label_commitment_bounded(
            label_handle,
            route_count=design.route_count,
            chunk_size=policy.route_validation_chunk_size,
        )
        if observed != query_receipt.resulting_label_commitment:
            _refuse(
                V1LabelNativeDirectionRefusalCode.LABEL_IDENTITY_MISMATCH,
                "shell-query labels differ from their frozen commitment",
            )
        return _construct_provider(
            label_handle=label_handle,
            prefix=prefix_rank_receipt,
            query=query_receipt,
            design=design,
            policy=policy,
            parent_mask=parent_mask,
            route_blocks=route_blocks,
            label_commitment_verified=True,
            owns_label_handle=False,
            qualified_lane_registration=None,
            device_total_memory_bytes=device_total_memory_bytes,
            current_reserved_bytes=current_reserved_bytes,
        )

    def _require_live(self) -> None:
        if self._closed or self._parent_gate is None:
            _refuse(
                V1LabelNativeDirectionRefusalCode.HANDLE_EVICTED,
                "label-native direction provider is closed",
            )
        registration = self._qualified_lane_registration
        if registration is not None:
            registration.assert_integrity(self._qualified_lane_registration_hash)

    def _require_packet_slot(self) -> None:
        if self._active_packet_token is not None:
            _refuse(
                V1LabelNativeDirectionRefusalCode.PACKET_LEASE_ACTIVE,
                "release the current direction packet before constructing another",
            )

    def _reserve_packet_slot(self) -> object:
        with self._operation_lock:
            self._require_live()
            self._require_packet_slot()
            token = object()
            self._active_packet_token = token
            return token

    def _issue_reserved_packet(
        self,
        *,
        token: object,
        packet_index: int,
        lane_identities: tuple[V1LabelNativeDirectionLaneIdentity, ...],
        compiled_masks: tuple[CompiledTorchCircuitMask, ...],
    ) -> V1LabelNativeDirectionLanePacket:
        with self._operation_lock:
            self._require_live()
            if self._active_packet_token is not token:
                _refuse(
                    V1LabelNativeDirectionRefusalCode.PACKET_RELEASED,
                    "packet construction lost its reserved memory lease",
                )
            return V1LabelNativeDirectionLanePacket(
                _construction_key=_PACKET_CONSTRUCTION_KEY,
                owner=self,
                owner_token=token,
                packet_index=packet_index,
                lane_identities=lane_identities,
                compiled_masks=compiled_masks,
            )

    def _cancel_packet_reservation(self, token: object) -> None:
        with self._operation_lock:
            if self._active_packet_token is token:
                self._active_packet_token = None

    def _release_packet(self, token: object) -> None:
        with self._operation_lock:
            if self._active_packet_token is not token:
                _refuse(
                    V1LabelNativeDirectionRefusalCode.PACKET_RELEASED,
                    "direction packet does not own the provider's active memory lease",
                )
            self._active_packet_token = None

    @property
    def has_active_packet_lease(self) -> bool:
        with self._operation_lock:
            return self._active_packet_token is not None

    @property
    def qualified_lane_registration(
        self,
    ) -> V1LabelNativeQualifiedLaneRegistration | None:
        self._require_live()
        return self._qualified_lane_registration

    def source_endpoint_label_for_provider_label(self, provider_label: int) -> int:
        """Replay the compact-provider-label to source-action-label mapping."""

        self._require_live()
        label = _positive_int(provider_label, name="compact provider endpoint label")
        source_labels = self.receipt.executed_source_endpoint_labels
        if label > len(source_labels):
            _refuse(
                V1LabelNativeDirectionRefusalCode.QUALIFIED_LANE_MISMATCH,
                "provider endpoint label lies outside its registered execution axis",
            )
        return source_labels[label - 1]

    def _validate_resident_labels(self, parent_bool: Any) -> None:
        expected_cumulative = self.design.endpoint_counts
        shell_counts = np.zeros(len(expected_cumulative), dtype=np.int64)
        try:
            with self._label_handle.borrow_for_registered_operation(
                operation=V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
            ) as lease:
                labels = lease._labels_for_registered_operation(
                    operation=(
                        V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
                    )
                )
                if lease.backend_kind == "numpy_reference":
                    values = np.asarray(labels, dtype=np.uint8)
                    parent = parent_bool.detach().cpu().numpy()
                    if values.shape != parent.shape or bool(
                        np.any((values != 0) & ~parent)
                    ):
                        _refuse(
                            V1LabelNativeDirectionRefusalCode.LABEL_IDENTITY_MISMATCH,
                            "direction labels leave the parent-live route domain",
                        )
                    if values.size and int(np.max(values)) > len(expected_cumulative):
                        _refuse(
                            V1LabelNativeDirectionRefusalCode.LABEL_IDENTITY_MISMATCH,
                            "direction labels exceed the frozen endpoint axis",
                        )
                    shell_counts[:] = np.bincount(
                        values,
                        minlength=len(expected_cumulative) + 1,
                    )[1:]
                else:
                    chunk = min(
                        self.policy.route_validation_chunk_size,
                        self.design.route_count,
                    )
                    counts = torch.zeros(
                        len(expected_cumulative) + 1,
                        dtype=torch.int64,
                        device=labels.device,
                    )
                    for start in range(0, self.design.route_count, chunk):
                        stop = min(self.design.route_count, start + chunk)
                        slab = labels[start:stop]
                        parent_slab = parent_bool[start:stop]
                        if bool(torch.any((slab != 0) & ~parent_slab).item()):
                            _refuse(
                                V1LabelNativeDirectionRefusalCode.LABEL_IDENTITY_MISMATCH,
                                "direction labels leave the parent-live route domain",
                            )
                        if bool(torch.any(slab > len(expected_cumulative)).item()):
                            _refuse(
                                V1LabelNativeDirectionRefusalCode.LABEL_IDENTITY_MISMATCH,
                                "direction labels exceed the frozen endpoint axis",
                            )
                        counts.add_(
                            torch.bincount(
                                slab.to(dtype=torch.int64),
                                minlength=len(expected_cumulative) + 1,
                            )
                        )
                    shell_counts[:] = counts[1:].detach().cpu().numpy()
        except V1DeviceShellRefusal as error:
            _refuse(
                V1LabelNativeDirectionRefusalCode.LABEL_IDENTITY_MISMATCH,
                f"shell-label custody refused direction validation: {error}",
            )
        if tuple(np.cumsum(shell_counts).tolist()) != expected_cumulative:
            _refuse(
                V1LabelNativeDirectionRefusalCode.LABEL_IDENTITY_MISMATCH,
                "label shell counts do not replay the frozen endpoint counts",
            )

    def compiled_parent(self) -> CompiledTorchCircuitMask:
        """Return the legacy unleased parent view.

        New execution code must use :meth:`lease_compiled_parent`; this method
        remains only for historical small-domain consumers.
        """

        self._require_live()
        return self._compiled_from_flat_gate(self._parent_gate)

    def lease_compiled_parent(self) -> V1LabelNativeParentLanePacket:
        """Issue one lifetime-safe compiled parent lane lease."""

        token = self._reserve_packet_slot()
        try:
            return V1LabelNativeParentLanePacket(
                _construction_key=_PARENT_PACKET_CONSTRUCTION_KEY,
                owner=self,
                owner_token=token,
                lane_key=self.parent_lane_key,
                compiled_mask=self._compiled_from_flat_gate(self._parent_gate),
            )
        except BaseException:
            self._cancel_packet_reservation(token)
            raise

    @property
    def parent_lane_key(self) -> str:
        self._require_live()
        return f"label-native-parent:{self.receipt.receipt_hash}"

    def _compiled_from_flat_gate(self, gate: Any) -> CompiledTorchCircuitMask:
        gates: dict[str, Any] = {}
        residual: dict[str, Any] = {}
        for block in self._route_blocks:
            value = gate[block.offset : block.stop].reshape(block.shape)
            key = str(block.layer_id)
            gates[key] = value
            if block.route_kind == "residual_stream_gate":
                residual[key] = value
        return CompiledTorchCircuitMask(
            parameter_overrides={},
            residual_gates=residual,
            gate_values=gates,
            float_gate_values=gates,
        )

    def iter_packed_direction_packets(
        self,
    ) -> Iterator[V1LabelNativeDirectionLanePacket]:
        """Return a retryable iterator over explicit one-at-a-time leases."""

        with self._operation_lock:
            self._require_live()
            source_labels = self.receipt.executed_source_endpoint_labels
            pairs = tuple(
                (
                    provider_label,
                    source_label,
                    self.design.endpoint_counts[source_label - 1],
                    self.receipt.endpoint_action_hashes[source_label - 1],
                    self.receipt.endpoint_direction_hashes[source_label - 1],
                    dose,
                )
                for provider_label, source_label in enumerate(
                    source_labels,
                    start=1,
                )
                for dose in self.design.doses
            )
        return _V1LabelNativeDirectionPacketIterator(
            _construction_key=_PACKET_ITERATOR_CONSTRUCTION_KEY,
            provider=self,
            pairs=pairs,
            width=self.forecast.selected_packed_lanes,
        )

    def materialize_selected_endpoint_packed_carrier(
        self,
        *,
        endpoint_label: int,
        endpoint_count: int,
        expected_action_hash: str,
    ) -> V1LabelNativeSelectedEndpointPackedCarrier:
        """Materialize one source-registered packed child, without authority."""

        label = _positive_int(endpoint_label, name="selected endpoint label")
        count = _positive_int(endpoint_count, name="selected endpoint count")
        action_hash = _identity(
            expected_action_hash,
            name="expected canonical packed action hash",
        )
        source_labels = self.receipt.executed_source_endpoint_labels
        if label > len(source_labels):
            _refuse(
                V1LabelNativeDirectionRefusalCode.ACTION_IDENTITY_MISMATCH,
                "selected provider label lies outside its registered execution axis",
            )
        source_label = source_labels[label - 1]
        if (
            self.receipt.endpoint_counts[source_label - 1] != count
            or self.receipt.endpoint_action_hashes[source_label - 1] != action_hash
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.ACTION_IDENTITY_MISMATCH,
                "selected endpoint coordinates differ from the canonical action axis",
            )
        token = self._reserve_packet_slot()
        try:
            source_ladder = self.design.action_axis_registration.source_shell_ladder
            source_action = source_ladder.actions[source_label - 1]
            with self._label_handle.borrow_for_registered_operation(
                operation=(
                    V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
                )
            ) as lease:
                labels = lease._labels_for_registered_operation(
                    operation=(
                        V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
                    )
                )
                if lease.backend_kind == "numpy_reference":
                    packed, packed_d2h = self._pack_selected_numpy_reference(
                        labels=labels,
                        endpoint_label=source_label,
                    )
                else:
                    packed, packed_d2h = self._pack_selected_torch_cuda(
                        labels=labels,
                        endpoint_label=source_label,
                    )
            digest = hashlib.sha256(packed).hexdigest()
            if digest != source_action.resulting_mask_packed_sha256:
                _refuse(
                    V1LabelNativeDirectionRefusalCode.ACTION_IDENTITY_MISMATCH,
                    "selected packed child differs from its frozen shell action",
                )
            receipt = V1LabelNativeSelectedEndpointPackedReceipt(
                provider_receipt_hash=self.receipt.receipt_hash,
                design_hash=self.design.design_hash,
                prefix_rank_receipt_hash=self.prefix_rank_receipt.receipt_hash,
                endpoint_query_hash=self.query_receipt.query_hash,
                action_axis_registration_hash=(
                    self.design.action_axis_registration.registration_hash
                ),
                source_shell_ladder_receipt_hash=source_ladder.receipt_hash,
                source_action_receipt=source_action,
                endpoint_label=source_label,
                endpoint_count=count,
                expected_action_hash=action_hash,
                route_axis_hash=self.receipt.route_axis_hash,
                route_count=self.receipt.route_count,
                parent_mask_hash=self.receipt.parent_mask_hash,
                resulting_mask_packed_sha256=digest,
                resulting_mask_hash=source_action.resulting_mask_hash,
                removed_route_index_hash=source_action.removed_route_index_hash,
                backend_kind=self.receipt.backend_kind,
                device_identity=self.receipt.device_identity,
                packed_payload_d2h_bytes=packed_d2h,
            )
            return V1LabelNativeSelectedEndpointPackedCarrier(
                receipt=receipt,
                packed_child_bits=packed,
            )
        except V1DeviceShellRefusal as error:
            _refuse(
                V1LabelNativeDirectionRefusalCode.LABEL_IDENTITY_MISMATCH,
                "shell-label custody refused selected materialization: "
                f"{error}",
            )
        finally:
            self._cancel_packet_reservation(token)

    def _pack_selected_numpy_reference(
        self,
        *,
        labels: object,
        endpoint_label: int,
    ) -> tuple[bytes, int]:
        values = np.asarray(labels, dtype=np.uint8)
        parent_live = np.unpackbits(
            np.frombuffer(self._parent_packed_bits, dtype=np.uint8),
            bitorder="big",
            count=self.receipt.route_count,
        ).astype(np.bool_, copy=False)
        if values.shape != parent_live.shape or bool(
            np.any((values != 0) & ~parent_live)
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.LABEL_IDENTITY_MISMATCH,
                "selected carrier labels leave the parent-live route domain",
            )
        child = parent_live & ~((values != 0) & (values <= endpoint_label))
        return np.packbits(child, bitorder="big").tobytes(), 0

    def _pack_selected_torch_cuda(
        self,
        *,
        labels: object,
        endpoint_label: int,
    ) -> tuple[bytes, int]:
        if (
            torch is None
            or not isinstance(labels, torch.Tensor)
            or labels.device.type != "cuda"
            or labels.dtype is not torch.uint8
            or labels.ndim != 1
            or int(labels.numel()) != self.receipt.route_count
            or not labels.is_contiguous()
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.CUDA_REQUIRED,
                "selected packed carrier requires contiguous CUDA uint8 labels",
            )
        device = labels.device
        chunk = min(
            self.policy.route_validation_chunk_size,
            self.receipt.route_count,
        )
        tile = max(8, (chunk // 8) * 8)
        workspace = min(tile, self.receipt.route_count) * (
            _I64_BYTES + 4 * _BOOL_BYTES
        )
        available_bytes = cuda_reusable_memory_bytes(device, framework=torch)
        allocated = int(torch.cuda.memory_allocated(device))
        if (
            workspace > available_bytes
            or allocated + workspace
            > self.forecast.device_total_or_workspace_ceiling_bytes
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.HBM_FORECAST_EXCEEDED,
                "current CUDA occupancy invalidates selected-carrier workspace",
            )
        parent_bytes = np.frombuffer(self._parent_packed_bits, dtype=np.uint8)
        output = bytearray(len(self._parent_packed_bits))
        bit_masks = torch.tensor(
            (128, 64, 32, 16, 8, 4, 2, 1),
            dtype=torch.uint8,
            device=device,
        )
        bit_weights = bit_masks.to(dtype=torch.int16)
        d2h = 0
        for start in range(0, self.receipt.route_count, tile):
            stop = min(self.receipt.route_count, start + tile)
            count = stop - start
            byte_start = start // 8
            byte_stop = byte_start + (count + 7) // 8
            parent_host = np.array(
                parent_bytes[byte_start:byte_stop],
                dtype=np.uint8,
                copy=True,
            )
            parent_t = torch.as_tensor(parent_host, dtype=torch.uint8, device=device)
            label_tile = labels[start:stop]
            padded_count = ((count + 7) // 8) * 8
            if padded_count != count:
                padded = torch.zeros(
                    padded_count,
                    dtype=torch.uint8,
                    device=device,
                )
                padded[:count].copy_(label_tile)
                label_tile = padded
            label_bits = label_tile.reshape(-1, 8)
            parent_live = (
                torch.bitwise_and(parent_t.reshape(-1, 1), bit_masks) != 0
            )
            invalid = torch.any(label_tile[:count] > len(self.receipt.endpoint_counts))
            dead_selected = torch.any((label_bits != 0) & ~parent_live)
            if bool(invalid.item()) or bool(dead_selected.item()):
                _refuse(
                    V1LabelNativeDirectionRefusalCode.LABEL_IDENTITY_MISMATCH,
                    "CUDA labels left the frozen endpoint or parent-live domain",
                )
            included = (label_bits != 0) & (label_bits <= endpoint_label)
            clear = torch.sum(
                included.to(torch.int16) * bit_weights.reshape(1, 8),
                dim=1,
                dtype=torch.int16,
            ).to(torch.uint8)
            child = torch.bitwise_and(parent_t, torch.bitwise_not(clear))
            host = child.detach().to(device="cpu", non_blocking=False).numpy()
            output[byte_start:byte_stop] = host.tobytes()
            d2h += int(host.nbytes)
            del (
                child,
                clear,
                dead_selected,
                host,
                included,
                invalid,
                label_bits,
                label_tile,
                parent_live,
                parent_t,
            )
        return bytes(output), d2h

    def _build_packet(
        self,
        *,
        packet_index: int,
        pairs: tuple[tuple[int, int, int, str, str, float], ...],
    ) -> V1LabelNativeDirectionLanePacket:
        token = self._reserve_packet_slot()
        try:
            with self._label_handle.borrow_for_registered_operation(
                operation=(
                    V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
                )
            ) as lease:
                labels = lease._labels_for_registered_operation(
                    operation=(
                        V1DeviceShellBorrowOperation.C_DIRECTIONAL_CERTIFICATION
                    )
                )
                labels_t = (
                    torch.tensor(
                        np.asarray(labels),
                        dtype=torch.uint8,
                        device=self._parent_gate.device,
                    )
                    if lease.backend_kind == "numpy_reference"
                    else labels
                )
                identities: list[V1LabelNativeDirectionLaneIdentity] = []
                compiled: list[CompiledTorchCircuitMask] = []
                for (
                    provider_label,
                    source_label,
                    count,
                    action,
                    direction,
                    dose,
                ) in pairs:
                    gate = self._parent_gate.clone()
                    for block in self._route_blocks:
                        label_block = labels_t[block.offset : block.stop]
                        selected = torch.logical_and(
                            label_block != 0,
                            label_block <= source_label,
                        )
                        gate[block.offset : block.stop].masked_fill_(
                            selected,
                            1.0 - float(dose),
                        )
                        del selected
                    lane_key = (
                        f"label-native:{direction}:eta:{float(dose).hex()}"
                    )
                    identities.append(
                        V1LabelNativeDirectionLaneIdentity(
                            endpoint_label=provider_label,
                            endpoint_count=count,
                            endpoint_action_hash=action,
                            dose=dose,
                            direction_hash=direction,
                            lane_key=lane_key,
                            source_endpoint_label=source_label,
                        )
                    )
                    compiled.append(self._compiled_from_flat_gate(gate))
                    del gate
            return self._issue_reserved_packet(
                token=token,
                packet_index=packet_index,
                lane_identities=tuple(identities),
                compiled_masks=tuple(compiled),
            )
        except V1DeviceShellRefusal as error:
            self._cancel_packet_reservation(token)
            _refuse(
                V1LabelNativeDirectionRefusalCode.LABEL_IDENTITY_MISMATCH,
                f"shell-label custody refused direction execution: {error}",
            )
        except BaseException:
            self._cancel_packet_reservation(token)
            raise

    def close(self) -> None:
        with self._operation_lock:
            if self._closed:
                return
            self._require_packet_slot()
            if self._owns_label_handle and not self._label_handle.is_evicted:
                self._label_handle.evict()
            self._parent_gate = None
            self._parent_packed_bits = None
            self._closed = True

    def __enter__(self) -> "V1LabelNativePackedDirectionProvider":
        self._require_live()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("label-native direction providers are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("label-native direction providers are nonserializable")


def _preflight_sources(
    *,
    prefix: V1DevicePrefixRankReceipt,
    design: V1LabelNativePackedDirectionDesign,
    parent_mask: HardMask,
    route_blocks: Sequence[RouteBlockSpec],
    policy: V1LabelNativePackedDirectionPolicy,
) -> tuple[RouteBlockSpec, ...]:
    if type(prefix) is not V1DevicePrefixRankReceipt:
        raise TypeError("label-native directions require an exact rank receipt")
    if type(design) is not V1LabelNativePackedDirectionDesign:
        raise TypeError("label-native directions require an exact design")
    if type(policy) is not V1LabelNativePackedDirectionPolicy:
        raise TypeError("label-native directions require an exact policy")
    blocks = _normalize_route_blocks(route_blocks, route_count=design.route_count)
    if (
        prefix.receipt_hash != design.prefix_rank_receipt_hash
        or prefix.source_rounding_hash != design.source_rounding_hash
        or prefix.route_axis_hash != design.route_axis_hash
        or prefix.route_count != design.route_count
        or parent_mask.route_axis_hash != design.route_axis_hash
        or parent_mask.route_count != design.route_count
        or _direct_mask_hash(parent_mask) != design.parent_mask_hash
    ):
        _refuse(
            V1LabelNativeDirectionRefusalCode.STALE_SOURCE,
            "prefix rank, parent mask, route blocks, and direction design differ",
        )
    if prefix.handle_kind == "numpy_reference":
        if policy.production_cuda_required or not policy.allow_private_cpu_reference:
            _refuse(
                V1LabelNativeDirectionRefusalCode.CUDA_REQUIRED,
                "production label-native directions require CUDA label custody",
            )
        if design.route_count > policy.private_cpu_route_limit:
            _refuse(
                V1LabelNativeDirectionRefusalCode.CPU_REFERENCE_LIMIT_EXCEEDED,
                "CPU label-native execution exceeds its focused-test route limit",
            )
    elif prefix.handle_kind != "torch_cuda":
        _refuse(
            V1LabelNativeDirectionRefusalCode.INVALID_TYPE,
            "prefix-rank receipt has an unsupported backend",
        )
    return blocks


def _construct_provider(
    *,
    label_handle: V1DeviceShellLabelHandle,
    prefix: V1DevicePrefixRankReceipt,
    query: V1DevicePrefixRankEndpointQueryReceipt,
    design: V1LabelNativePackedDirectionDesign,
    policy: V1LabelNativePackedDirectionPolicy,
    parent_mask: HardMask,
    route_blocks: Sequence[RouteBlockSpec],
    label_commitment_verified: bool,
    owns_label_handle: bool,
    qualified_lane_registration: V1LabelNativeQualifiedLaneRegistration | None,
    device_total_memory_bytes: int | None,
    current_reserved_bytes: int | None,
) -> V1LabelNativePackedDirectionProvider:
    blocks = _preflight_sources(
        prefix=prefix,
        design=design,
        parent_mask=parent_mask,
        route_blocks=route_blocks,
        policy=policy,
    )
    ladder = design.action_axis_registration.source_shell_ladder
    stale_legs = tuple(
        name
        for name, failed in (
            ("query.prefix_rank_receipt_hash", query.prefix_rank_receipt_hash != prefix.receipt_hash),
            ("query.route_axis_hash", query.route_axis_hash != design.route_axis_hash),
            ("query.route_count", query.route_count != design.route_count),
            ("query.endpoint_counts", query.endpoint_counts != design.endpoint_counts),
            ("label_handle.route_count", label_handle.route_count != design.route_count),
            ("label_handle.backend_kind", label_handle.backend_kind != prefix.handle_kind),
            (
                "label_handle.device_identity",
                label_handle.device_identity != prefix.device_identity,
            ),
            ("ladder.handle_kind", ladder.handle_kind != label_handle.backend_kind),
            (
                "ladder.device_identity",
                ladder.device_identity != label_handle.device_identity,
            ),
            (
                "query.endpoint_query_conditioned_on_c_outcomes",
                bool(query.endpoint_query_conditioned_on_c_outcomes),
            ),
        )
        if failed
    )
    if stale_legs:
        _refuse(
            V1LabelNativeDirectionRefusalCode.STALE_SOURCE,
            "shell query, label custody, and frozen direction design differ: "
            + ", ".join(stale_legs)
            + f" (label {label_handle.device_identity!r}, prefix {prefix.device_identity!r}, "
            f"ladder {ladder.device_identity!r})",
        )
    if qualified_lane_registration is not None:
        qualified_lane_registration.assert_integrity()
        if (
            qualified_lane_registration.action_axis_registration_hash
            != design.action_axis_registration.registration_hash
            or qualified_lane_registration.complete_source_endpoint_counts
            != design.endpoint_counts
            or qualified_lane_registration.complete_source_action_hashes
            != design.action_axis_registration.endpoint_action_hashes
        ):
            _refuse(
                V1LabelNativeDirectionRefusalCode.QUALIFIED_LANE_MISMATCH,
                "qualified execution source differs from the full endpoint query",
            )
    maximum_block = max(block.stop - block.offset for block in blocks)
    forecast = _select_forecast(
        backend_kind=label_handle.backend_kind,
        device_identity=label_handle.device_identity,
        route_count=design.route_count,
        endpoint_count=len(design.endpoint_counts),
        maximum_route_block_size=maximum_block,
        policy=policy,
        device_total_memory_bytes=device_total_memory_bytes,
        current_reserved_bytes=current_reserved_bytes,
    )
    return V1LabelNativePackedDirectionProvider(
        _construction_key=_PROVIDER_CONSTRUCTION_KEY,
        label_handle=label_handle,
        prefix_rank_receipt=prefix,
        query_receipt=query,
        design=design,
        policy=policy,
        parent_mask=parent_mask,
        route_blocks=blocks,
        forecast=forecast,
        label_commitment_verified=label_commitment_verified,
        owns_label_handle=owns_label_handle,
        qualified_lane_registration=qualified_lane_registration,
    )


__all__ = [
    "LABEL_NATIVE_ACTION_AXIS_SCHEMA",
    "LABEL_NATIVE_CANONICAL_ACTION_IDENTITY",
    "LABEL_NATIVE_DIRECTION_ALGORITHM_ID",
    "LABEL_NATIVE_DIRECTION_AUTHORITY",
    "LABEL_NATIVE_DIRECTION_DESIGN_SCHEMA",
    "LABEL_NATIVE_DIRECTION_HBM_SCHEMA",
    "LABEL_NATIVE_DIRECTION_POLICY_SCHEMA",
    "LABEL_NATIVE_DIRECTION_RECEIPT_SCHEMA",
    "LABEL_NATIVE_QUALIFIED_DIRECTION_RECEIPT_SCHEMA",
    "LABEL_NATIVE_QUALIFIED_LANE_REGISTRATION_SCHEMA",
    "LABEL_NATIVE_SELECTED_CARRIER_RECEIPT_SCHEMA",
    "LABEL_NATIVE_SELECTED_CARRIER_SCHEMA",
    "V1LabelNativeCanonicalActionAxisRegistration",
    "V1LabelNativeDirectionHBMForecast",
    "V1LabelNativeDirectionLaneIdentity",
    "V1LabelNativeDirectionLanePacket",
    "V1LabelNativeParentLanePacket",
    "V1LabelNativeDirectionRefusal",
    "V1LabelNativeDirectionRefusalCode",
    "V1LabelNativePackedDirectionDesign",
    "V1LabelNativePackedDirectionPolicy",
    "V1LabelNativePackedDirectionProvider",
    "V1LabelNativePackedDirectionReceipt",
    "V1LabelNativeQualifiedLaneRegistration",
    "V1LabelNativeSelectedEndpointPackedCarrier",
    "V1LabelNativeSelectedEndpointPackedReceipt",
    "label_native_direction_provider_source_hash",
    "register_v1_label_native_qualified_lanes",
]
