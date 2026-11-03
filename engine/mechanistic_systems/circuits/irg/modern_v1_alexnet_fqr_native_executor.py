"""Native AlexNet shared-C executor for canonical v1 F/Q/R certification.

The executor is deliberately narrower than the F/Q/R statistical controller.
It binds one already frozen branch family and one exact registered C slice,
projects corrected Aumann--Shapley evidence and the literal relaxed-gate
Jacobian through one scorer capture/VJP, and then measures the finite
directional secants at ``eta in {1/2, 1}``.  It selects no endpoint, spends no
alpha, opens no calibration/oracle stream, and grants no mask-commit authority.

The only route-scale live objects are the existing packed projection consumers
and the directional gate provider.  The durable registration contains hashes,
compact axes, counts, and a pre-outcome memory forecast; it never serializes
context-by-route rows or the analytic resolver's live route geometry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from pathlib import Path
from string import hexdigits
from types import MappingProxyType
from typing import Any, Final, NoReturn

import numpy as np

from mechanistic_systems.circuits.irg.contracts import MeasurementRequest
from mechanistic_systems.circuits.irg.modern_v1_alexnet_fqr_bounds import (
    V1AlexNetFQRBranchAnalyticBounds,
    build_v1_alexnet_fqr_raw_logit_interval_verifier,
    directional_observation_bounds_from_v1_alexnet_fqr_receipt,
)
from mechanistic_systems.circuits.irg.modern_v1_directional_dual_projection import (
    V1DirectionalDualProjectionPlan,
    bind_v1_directional_dual_projection_scorer_proof,
    close_v1_directional_dual_projection,
)
from mechanistic_systems.circuits.irg.modern_v1_directional_remainder import (
    DIRECTIONAL_SECANT_DOSES,
    V1DirectionalSecantRecord,
)
from mechanistic_systems.circuits.irg.modern_v1_directional_secant_bridge import (
    build_v1_native_directional_secant_provider,
    build_v1_native_fibered_directional_secant_provider,
)
from mechanistic_systems.circuits.irg.modern_v1_fibered_directional_record import (
    V1FiberedDirectionalRecord,
    build_v1_fibered_directional_projection_closures,
)
from mechanistic_systems.circuits.irg.modern_v1_fibered_work_projection_gpu import (
    V1FiberedWorkProjectionGPUConsumer,
    V1FiberedWorkProjectionRequest,
    fibered_work_projection_kernel_source_hash,
)
from mechanistic_systems.circuits.irg.modern_v1_fqr_branch_design import (
    V1FQRBranchFamilyDesign,
)
from mechanistic_systems.circuits.irg.modern_v1_ladder_projection_gpu import (
    V1DirectionalActionProjectionRequest,
    V1LadderProjectionGPUConsumer,
    projection_kernel_source_hash,
)
from mechanistic_systems.circuits.irg.modern_v1_label_native_projection_scorer_adapter import (
    LABEL_NATIVE_PROJECTION_SCORER_CHILD_RECEIPT_SCHEMA,
)
from mechanistic_systems.circuits.irg.modern_v1_relaxed_kkt import (
    V1RelaxedKKTBranch,
)
from mechanistic_systems.circuits.irg.modern_v1_native_fiber_axis import (
    V1NativeDestinationFiberAxisReceipt,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.core.axis_identity import (
    ConstructiveOrdinalRouteAxis,
    ordered_string_axis_hash,
)
from mechanistic_systems.measurement.modern_v1_directional_secant_provider import (
    V1FiberedDirectionalSecantObservationBounds,
    V1PackedDirectionalSecantExecutionReceipt,
    V1PackedFiberedDirectionalSecantExecutionReceipt,
    directional_secant_provider_source_hash,
)
from mechanistic_systems.measurement.optimized_irg_providers import (
    build_route_chunked_dual_chart_fanout,
)

try:  # pragma: no cover - import availability is environment-dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


NATIVE_FQR_EXECUTION_POLICY_SCHEMA: Final = (
    "partition_free_v1_alexnet_native_fqr_execution_policy_v1"
)
NATIVE_FQR_MEMORY_FORECAST_SCHEMA: Final = (
    "partition_free_v1_alexnet_native_fqr_memory_forecast_v1"
)
NATIVE_FQR_ANALYTIC_CACHE_SCHEMA: Final = (
    "partition_free_v1_alexnet_fqr_analytic_cache_v1"
)
NATIVE_FQR_PRE_C_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_alexnet_native_fqr_pre_c_registration_v1"
)
NATIVE_FQR_EXECUTION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_alexnet_native_fqr_execution_receipt_v1"
)
NATIVE_FQR_EXECUTION_RESULT_SCHEMA: Final = (
    "partition_free_v1_alexnet_native_fqr_execution_result_v1"
)
NATIVE_FIBERED_FQR_EXECUTION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_alexnet_native_fibered_fqr_execution_receipt_v1"
)
NATIVE_FIBERED_FQR_EXECUTION_RESULT_SCHEMA: Final = (
    "partition_free_v1_alexnet_native_fibered_fqr_execution_result_v1"
)
NATIVE_FIBERED_FQR_MODE_BINDING_SCHEMA: Final = (
    "partition_free_v1_alexnet_native_fibered_fqr_mode_binding_v1"
)
NATIVE_FQR_AUTHORITY_SCOPE: Final = (
    "shared_C_projection_and_directional_secant_measurement_only_"
    "no_fqr_decision_or_calibration_or_oracle_or_commit_or_terminal_authority_v1"
)

_POLICY_PREFIX: Final = "partition-free-v1-alexnet-native-fqr-policy-"
_FORECAST_PREFIX: Final = "partition-free-v1-alexnet-native-fqr-forecast-"
_CACHE_PREFIX: Final = "partition-free-v1-alexnet-fqr-analytic-cache-"
_REGISTRATION_PREFIX: Final = (
    "partition-free-v1-alexnet-native-fqr-pre-c-registration-"
)
_EXECUTION_PREFIX: Final = "partition-free-v1-alexnet-native-fqr-execution-"
_RESULT_PREFIX: Final = "partition-free-v1-alexnet-native-fqr-result-"
_FIBERED_EXECUTION_PREFIX: Final = (
    "partition-free-v1-alexnet-native-fibered-fqr-execution-"
)
_FIBERED_RESULT_PREFIX: Final = (
    "partition-free-v1-alexnet-native-fibered-fqr-result-"
)
_FIBERED_MODE_BINDING_PREFIX: Final = (
    "partition-free-v1-alexnet-native-fibered-fqr-mode-binding-"
)
_FIBERED_BOUNDS_ADAPTER_PREFIX: Final = (
    "partition-free-v1-alexnet-native-fibered-fqr-bounds-adapter-"
)
_SCORER_RECEIPT_PREFIX: Final = "partition-free-v1-raw-target-logit-stream-"
_SCORER_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_raw_target_logit_stream_receipt_v1"
)
_SCORER_PROOF_PREFIX: Final = (
    "partition-free-v1-alexnet-native-fqr-same-capture-vjp-"
)
_PROJECTION_LINEAGE_PREFIX: Final = (
    "partition-free-v1-alexnet-native-fqr-projection-lineage-"
)
_RAW_TARGET_LOGIT_COORDINATE: Final = "raw_target_logit"
_RAW_TARGET_LOGIT_COTANGENT: Final = "raw_target_logit_unit_cotangent"
_FLOAT64_BYTES: Final = 8
_INT64_BYTES: Final = 8


class V1AlexNetFQRNativeExecutionMode(str, Enum):
    """Execution classes; the CPU class is authority-inert test evidence."""

    PRODUCTION_CUDA = "production_cuda"
    REFERENCE_CPU_TEST_ONLY = "reference_cpu_test_only"


class V1AlexNetFQRNativeRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    COORDINATE_MISMATCH = "coordinate_mismatch"
    HASH_MISMATCH = "hash_mismatch"
    STALE_GRAPH = "stale_graph"
    POST_OUTCOME = "post_outcome"
    AUTHORITY_VIOLATION = "authority_violation"
    CUDA_REQUIRED = "cuda_required"
    WORKSPACE_EXCEEDED = "workspace_exceeded"
    SCORER_CONTRACT = "scorer_contract"
    ANALYTIC_CACHE_MISMATCH = "analytic_cache_mismatch"
    PARTIAL_PROJECTION = "partial_projection"
    NUMERICAL_SUPPORT_MISMATCH = "numerical_support_mismatch"


class V1AlexNetFQRNativeRefusal(PermissionError):
    def __init__(
        self,
        code: V1AlexNetFQRNativeRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1AlexNetFQRNativeRefusalCode, message: str) -> NoReturn:
    raise V1AlexNetFQRNativeRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _identity(value, name=name)
    if (
        len(result) != 64
        or result.lower() != result
        or any(character not in hexdigits.lower() for character in result)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256 identity")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be one positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return result


def _identity_axis(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(_identity(value, name=f"{name} item") for value in tuple(values))
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not repeat")
    return result


def _route_identity_axis(
    values: Sequence[str],
    *,
    name: str,
) -> Sequence[str]:
    """Validate a route axis without defeating constructive storage.

    ``ConstructiveOrdinalRouteAxis`` proves the type, uniqueness, and order
    of its exact ``route:<ordinal>`` grammar by construction.  Replaying the
    generic tuple/set validator over the 56,994,496-route AlexNet axis would
    allocate tens of millions of Python strings before observing C.  Ordinary
    finite axes retain the strict historical validator.
    """

    if type(values) is ConstructiveOrdinalRouteAxis:
        return values
    return _identity_axis(values, name=name)


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def alexnet_fqr_native_executor_source_hash() -> str:
    """Hash the exact orchestration source sealed by the execution policy."""

    return _source_sha256()


@dataclass(frozen=True, slots=True)
class V1AlexNetFQRNativeExecutionPolicy:
    """Immutable resource/source policy fixed before any C observation."""

    policy_id: str
    execution_mode: V1AlexNetFQRNativeExecutionMode
    registered_route_chunk_size: int
    registered_context_packet_size: int
    maximum_packed_lanes: int
    maximum_gate_workspace_bytes: int
    maximum_native_workspace_bytes: int
    executor_source_hash: str
    projection_kernel_source_hash: str
    packed_direction_provider_source_hash: str
    relaxed_gate_executor_source_hash: str
    raw_logit_interval_verifier_source_hash: str
    analytic_resolver_source_hash: str
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    production_cpu_fallback_allowed: bool = False
    context_route_host_rows_allowed: bool = False
    fqr_decision_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = NATIVE_FQR_AUTHORITY_SCOPE
    schema: str = NATIVE_FQR_EXECUTION_POLICY_SCHEMA
    _policy_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _identity(self.policy_id, name="policy id"))
        if type(self.execution_mode) is not V1AlexNetFQRNativeExecutionMode:
            raise TypeError("native F/Q/R execution mode must be the exact enum")
        for name in (
            "registered_route_chunk_size",
            "registered_context_packet_size",
            "maximum_packed_lanes",
            "maximum_gate_workspace_bytes",
            "maximum_native_workspace_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        for name in (
            "executor_source_hash",
            "projection_kernel_source_hash",
            "packed_direction_provider_source_hash",
            "relaxed_gate_executor_source_hash",
            "raw_logit_interval_verifier_source_hash",
            "analytic_resolver_source_hash",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        true_flags = (self.issued_before_C_outcomes,)
        false_flags = (
            self.C_outcomes_seen,
            self.production_cpu_fallback_allowed,
            self.context_route_host_rows_allowed,
            self.fqr_decision_authority,
            self.calibration_K_open_authority,
            self.paired_oracle_O_open_authority,
            self.mask_commit_authority,
            self.terminal_authority,
        )
        if any(type(value) is not bool for value in (*true_flags, *false_flags)):
            raise TypeError("native F/Q/R policy flags must be exact booleans")
        if not all(true_flags) or any(false_flags):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.AUTHORITY_VIOLATION,
                "native F/Q/R policy crossed its pre-C authority boundary",
            )
        if (
            self.executor_source_hash != _source_sha256()
            or self.projection_kernel_source_hash != projection_kernel_source_hash()
            or self.packed_direction_provider_source_hash
            != directional_secant_provider_source_hash()
        ):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "native F/Q/R execution or numerical source changed after registration",
            )
        if (
            self.schema != NATIVE_FQR_EXECUTION_POLICY_SCHEMA
            or self.authority_scope != NATIVE_FQR_AUTHORITY_SCOPE
        ):
            raise ValueError("native F/Q/R execution policy schema/scope is invalid")
        self.assert_integrity()

    @property
    def production_cuda_authority(self) -> bool:
        return self.execution_mode is V1AlexNetFQRNativeExecutionMode.PRODUCTION_CUDA

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "policy_id": self.policy_id,
                "execution_mode": self.execution_mode.value,
                "registered_route_chunk_size": self.registered_route_chunk_size,
                "registered_context_packet_size": self.registered_context_packet_size,
                "maximum_packed_lanes": self.maximum_packed_lanes,
                "maximum_gate_workspace_bytes": self.maximum_gate_workspace_bytes,
                "maximum_native_workspace_bytes": self.maximum_native_workspace_bytes,
                "executor_source_hash": self.executor_source_hash,
                "projection_kernel_source_hash": self.projection_kernel_source_hash,
                "packed_direction_provider_source_hash": (
                    self.packed_direction_provider_source_hash
                ),
                "relaxed_gate_executor_source_hash": (
                    self.relaxed_gate_executor_source_hash
                ),
                "raw_logit_interval_verifier_source_hash": (
                    self.raw_logit_interval_verifier_source_hash
                ),
                "analytic_resolver_source_hash": self.analytic_resolver_source_hash,
                "issued_before_C_outcomes": self.issued_before_C_outcomes,
                "C_outcomes_seen": self.C_outcomes_seen,
                "production_cpu_fallback_allowed": (
                    self.production_cpu_fallback_allowed
                ),
                "context_route_host_rows_allowed": self.context_route_host_rows_allowed,
                "fqr_decision_authority": self.fqr_decision_authority,
                "calibration_K_open_authority": self.calibration_K_open_authority,
                "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
                "mask_commit_authority": self.mask_commit_authority,
                "terminal_authority": self.terminal_authority,
                "authority_scope": self.authority_scope,
            }
        )

    @property
    def policy_hash(self) -> str:
        cached = self._policy_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        if (
            self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
            or self.production_cpu_fallback_allowed is not False
            or self.context_route_host_rows_allowed is not False
            or self.fqr_decision_authority is not False
            or self.calibration_K_open_authority is not False
            or self.paired_oracle_O_open_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
        ):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.AUTHORITY_VIOLATION,
                "native F/Q/R execution policy gained downstream authority",
            )
        if (
            self.executor_source_hash != _source_sha256()
            or self.projection_kernel_source_hash != projection_kernel_source_hash()
            or self.packed_direction_provider_source_hash
            != directional_secant_provider_source_hash()
        ):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "native F/Q/R source binding is stale",
            )
        observed = content_hash(self._payload(), prefix=_POLICY_PREFIX)
        if self._policy_hash is not None and self._policy_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "native F/Q/R execution policy was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "native F/Q/R execution policy hash does not match",
            )
        object.__setattr__(self, "_policy_hash", observed)
        return observed


def build_v1_alexnet_fqr_native_execution_policy(
    *,
    policy_id: str,
    registered_route_chunk_size: int,
    registered_context_packet_size: int,
    maximum_packed_lanes: int,
    maximum_gate_workspace_bytes: int,
    maximum_native_workspace_bytes: int,
    relaxed_gate_executor_source_hash: str,
    raw_logit_interval_verifier_source_hash: str,
    analytic_resolver_source_hash: str,
    reference_cpu_test_only: bool = False,
) -> V1AlexNetFQRNativeExecutionPolicy:
    """Seal a production-CUDA policy or an authority-inert CPU test policy."""

    if type(reference_cpu_test_only) is not bool:
        raise TypeError("reference_cpu_test_only must be bool")
    return V1AlexNetFQRNativeExecutionPolicy(
        policy_id=policy_id,
        execution_mode=(
            V1AlexNetFQRNativeExecutionMode.REFERENCE_CPU_TEST_ONLY
            if reference_cpu_test_only
            else V1AlexNetFQRNativeExecutionMode.PRODUCTION_CUDA
        ),
        registered_route_chunk_size=registered_route_chunk_size,
        registered_context_packet_size=registered_context_packet_size,
        maximum_packed_lanes=maximum_packed_lanes,
        maximum_gate_workspace_bytes=maximum_gate_workspace_bytes,
        maximum_native_workspace_bytes=maximum_native_workspace_bytes,
        executor_source_hash=_source_sha256(),
        projection_kernel_source_hash=projection_kernel_source_hash(),
        packed_direction_provider_source_hash=(
            directional_secant_provider_source_hash()
        ),
        relaxed_gate_executor_source_hash=relaxed_gate_executor_source_hash,
        raw_logit_interval_verifier_source_hash=(
            raw_logit_interval_verifier_source_hash
        ),
        analytic_resolver_source_hash=analytic_resolver_source_hash,
    )


@dataclass(frozen=True, slots=True)
class V1AlexNetFQRNativeMemoryForecast:
    """Pre-outcome bound for executor-owned route-scale materialization."""

    policy_hash: str
    branch_family_design_hash: str
    route_count: int
    branch_count: int
    endpoint_counts_by_branch: tuple[int, ...]
    final_removed_counts_by_branch: tuple[int, ...]
    context_slice_count: int
    projection_fanout_peak_bytes: int
    gate_workspace_forecast_bytes: int
    secant_peak_bytes: int
    peak_native_workspace_bytes: int
    maximum_gate_workspace_bytes: int
    maximum_native_workspace_bytes: int
    forecast_scope: str = (
        "executor_owned_projection_CSR_shells_dual_route_chunks_projection_outputs_"
        "and_directional_gate_fields_excluding_model_scorer_and_compiler_allocators_v1"
    )
    registered_before_C_outcomes: bool = True
    schema: str = NATIVE_FQR_MEMORY_FORECAST_SCHEMA
    _forecast_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in ("policy_hash", "branch_family_design_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "route_count",
            "branch_count",
            "context_slice_count",
            "projection_fanout_peak_bytes",
            "gate_workspace_forecast_bytes",
            "secant_peak_bytes",
            "peak_native_workspace_bytes",
            "maximum_gate_workspace_bytes",
            "maximum_native_workspace_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        endpoints = tuple(
            _positive_int(value, name="branch endpoint count")
            for value in self.endpoint_counts_by_branch
        )
        removed = tuple(
            _positive_int(value, name="branch final removed count")
            for value in self.final_removed_counts_by_branch
        )
        if (
            len(endpoints) != self.branch_count
            or len(removed) != self.branch_count
            or self.peak_native_workspace_bytes
            != max(self.projection_fanout_peak_bytes, self.secant_peak_bytes)
            or self.registered_before_C_outcomes is not True
            or self.schema != NATIVE_FQR_MEMORY_FORECAST_SCHEMA
        ):
            raise ValueError("native F/Q/R memory forecast axes/schema are invalid")
        if (
            self.gate_workspace_forecast_bytes
            > self.maximum_gate_workspace_bytes
            or self.peak_native_workspace_bytes > self.maximum_native_workspace_bytes
        ):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.WORKSPACE_EXCEEDED,
                "native F/Q/R pre-outcome memory forecast exceeds its ceiling",
            )
        object.__setattr__(self, "endpoint_counts_by_branch", endpoints)
        object.__setattr__(self, "final_removed_counts_by_branch", removed)
        object.__setattr__(
            self,
            "forecast_scope",
            _identity(self.forecast_scope, name="forecast scope"),
        )
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "policy_hash": self.policy_hash,
                "branch_family_design_hash": self.branch_family_design_hash,
                "route_count": self.route_count,
                "branch_count": self.branch_count,
                "endpoint_counts_by_branch": self.endpoint_counts_by_branch,
                "final_removed_counts_by_branch": (
                    self.final_removed_counts_by_branch
                ),
                "context_slice_count": self.context_slice_count,
                "projection_fanout_peak_bytes": self.projection_fanout_peak_bytes,
                "gate_workspace_forecast_bytes": (
                    self.gate_workspace_forecast_bytes
                ),
                "secant_peak_bytes": self.secant_peak_bytes,
                "peak_native_workspace_bytes": self.peak_native_workspace_bytes,
                "maximum_gate_workspace_bytes": self.maximum_gate_workspace_bytes,
                "maximum_native_workspace_bytes": self.maximum_native_workspace_bytes,
                "forecast_scope": self.forecast_scope,
                "registered_before_C_outcomes": self.registered_before_C_outcomes,
            }
        )

    @property
    def forecast_hash(self) -> str:
        cached = self._forecast_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_FORECAST_PREFIX)
        if self._forecast_hash is not None and self._forecast_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "native F/Q/R memory forecast was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "native F/Q/R memory forecast hash does not match",
            )
        object.__setattr__(self, "_forecast_hash", observed)
        return observed


def forecast_v1_alexnet_fqr_native_memory(
    *,
    policy: V1AlexNetFQRNativeExecutionPolicy,
    family_design: V1FQRBranchFamilyDesign,
    context_slice_count: int,
) -> V1AlexNetFQRNativeMemoryForecast:
    """Compute an outcome-independent upper bound from frozen compact axes only."""

    if type(policy) is not V1AlexNetFQRNativeExecutionPolicy:
        raise TypeError("native memory forecast requires the exact policy")
    if type(family_design) is not V1FQRBranchFamilyDesign:
        raise TypeError("native memory forecast requires the exact branch family")
    policy.assert_integrity()
    family_design.assert_integrity()
    contexts = _positive_int(context_slice_count, name="C slice context count")
    route_count = family_design.registration.prefrozen_family.route_count
    endpoints = tuple(
        len(bundle.secant_design.endpoint_action_hashes)
        for bundle in family_design.branch_bundles
    )
    removed = tuple(
        bundle.secant_design.ladder.endpoint_removed_counts[-1]
        for bundle in family_design.branch_bundles
    )
    packet_contexts = min(contexts, policy.registered_context_packet_size)
    selected_total = sum(removed)
    endpoint_total = sum(endpoints)

    # Two scientific streams each retain their own selected-shell CSR and
    # packet staging.  Only a registered route window of each stream exists at
    # the producer boundary; no full context x route tensor is forecast.
    static_csr = 2 * selected_total * _INT64_BYTES
    shell_packet = 2 * packet_contexts * selected_total * _FLOAT64_BYTES
    endpoint_packet = 2 * packet_contexts * endpoint_total * 2 * _FLOAT64_BYTES
    dual_route_windows = (
        2
        * packet_contexts
        * policy.registered_route_chunk_size
        * _FLOAT64_BYTES
    )
    projection_outputs = (
        2 * contexts * endpoint_total * 6 * _FLOAT64_BYTES
    )
    projection_peak = (
        static_csr
        + shell_packet
        + endpoint_packet
        + dual_route_windows
        + projection_outputs
    )

    max_endpoints = max(endpoints)
    gate_fields = route_count * (2 + 4 * policy.maximum_packed_lanes)
    secant_outputs = contexts * max_endpoints * 2 * 4 * _FLOAT64_BYTES
    secant_peak = gate_fields + secant_outputs
    return V1AlexNetFQRNativeMemoryForecast(
        policy_hash=policy.policy_hash,
        branch_family_design_hash=family_design.design_hash,
        route_count=route_count,
        branch_count=len(endpoints),
        endpoint_counts_by_branch=endpoints,
        final_removed_counts_by_branch=removed,
        context_slice_count=contexts,
        projection_fanout_peak_bytes=projection_peak,
        gate_workspace_forecast_bytes=gate_fields,
        secant_peak_bytes=secant_peak,
        peak_native_workspace_bytes=max(projection_peak, secant_peak),
        maximum_gate_workspace_bytes=policy.maximum_gate_workspace_bytes,
        maximum_native_workspace_bytes=policy.maximum_native_workspace_bytes,
    )


@dataclass(frozen=True, slots=True)
class V1AlexNetFQRAnalyticBoundsResolver:
    """Ephemeral exact-result cache identified durably only by receipt hashes."""

    resolver_source_hash: str
    expected_receipt_hashes_by_branch: tuple[tuple[V1RelaxedKKTBranch, str], ...]
    _results: tuple[V1AlexNetFQRBranchAnalyticBounds, ...] = field(
        repr=False, compare=False
    )
    schema: str = NATIVE_FQR_ANALYTIC_CACHE_SCHEMA
    _cache_identity_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        source = _sha256(self.resolver_source_hash, name="analytic resolver source hash")
        expected = tuple(self.expected_receipt_hashes_by_branch)
        results = tuple(self._results)
        if (
            not expected
            or len(expected) != len(results)
            or any(type(branch) is not V1RelaxedKKTBranch for branch, _ in expected)
            or len({branch for branch, _ in expected}) != len(expected)
            or any(type(result) is not V1AlexNetFQRBranchAnalyticBounds for result in results)
        ):
            raise TypeError("analytic resolver requires aligned exact branch results")
        normalized = tuple(
            (branch, _identity(receipt_hash, name="analytic receipt hash"))
            for branch, receipt_hash in expected
        )
        for (branch, receipt_hash), result in zip(normalized, results, strict=True):
            result.assert_integrity()
            if (
                result.derivation_receipt.branch is not branch
                or result.derivation_receipt.receipt_hash != receipt_hash
            ):
                _refuse(
                    V1AlexNetFQRNativeRefusalCode.ANALYTIC_CACHE_MISMATCH,
                    "analytic result differs from its registered branch receipt",
                )
        if self.schema != NATIVE_FQR_ANALYTIC_CACHE_SCHEMA:
            raise ValueError("analytic resolver schema is invalid")
        object.__setattr__(self, "resolver_source_hash", source)
        object.__setattr__(self, "expected_receipt_hashes_by_branch", normalized)
        object.__setattr__(self, "_results", results)
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "resolver_source_hash": self.resolver_source_hash,
                "expected_receipt_hashes_by_branch": tuple(
                    (branch.value, receipt_hash)
                    for branch, receipt_hash in self.expected_receipt_hashes_by_branch
                ),
                "raw_route_geometry_serialized": False,
            }
        )

    @property
    def cache_identity_hash(self) -> str:
        cached = self._cache_identity_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for (branch, receipt_hash), result in zip(
            self.expected_receipt_hashes_by_branch,
            self._results,
            strict=True,
        ):
            result.assert_integrity()
            if (
                result.derivation_receipt.branch is not branch
                or result.derivation_receipt.receipt_hash != receipt_hash
            ):
                _refuse(
                    V1AlexNetFQRNativeRefusalCode.ANALYTIC_CACHE_MISMATCH,
                    "analytic cache changed after registration",
                )
        observed = content_hash(self._payload(), prefix=_CACHE_PREFIX)
        if self._cache_identity_hash is not None and self._cache_identity_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.ANALYTIC_CACHE_MISMATCH,
                "analytic cache identity was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.ANALYTIC_CACHE_MISMATCH,
                "analytic cache identity does not match",
            )
        object.__setattr__(self, "_cache_identity_hash", observed)
        return observed

    def resolve(
        self,
        branch: V1RelaxedKKTBranch,
        *,
        expected_receipt_hash: str,
    ) -> V1AlexNetFQRBranchAnalyticBounds:
        self.assert_integrity()
        expected = _identity(expected_receipt_hash, name="expected analytic receipt hash")
        for (candidate, receipt_hash), result in zip(
            self.expected_receipt_hashes_by_branch,
            self._results,
            strict=True,
        ):
            if candidate is branch:
                if receipt_hash != expected:
                    _refuse(
                        V1AlexNetFQRNativeRefusalCode.ANALYTIC_CACHE_MISMATCH,
                        "analytic branch was requested under another receipt hash",
                    )
                return result
        _refuse(
            V1AlexNetFQRNativeRefusalCode.ANALYTIC_CACHE_MISMATCH,
            "analytic cache omits a retained F/Q/R branch",
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "live AlexNet F/Q/R analytic caches are not serializable; persist only "
            "cache identity and derivation receipt hashes"
        )


@dataclass(frozen=True, slots=True)
class V1AlexNetFQRNativeCPreExecutionRegistration:
    """Compact pre-C proof binding graph, family, slice, streams, and policy."""

    graph_hash: str
    family_design: V1FQRBranchFamilyDesign = field(repr=False, compare=False)
    current_mask: HardMask = field(repr=False, compare=False)
    chart_identity_hash: str
    proposal_generation: int
    sampled_d_geometry_hash: str
    upstream_d_same_vjp_execution_proof_hash: str
    certification_context_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    certification_draw_ids_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    sealed_order_start_indices_by_stratum: tuple[tuple[str, int], ...]
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    same_capture_and_vjp_registration_hash: str
    scorer_request_identity_hash: str
    scorer_request_state_hash: str
    ordered_score_route_axis_hash: str
    destination_transport_hash: str
    analytic_cache_identity_hash: str
    analytic_receipt_hashes_by_branch: tuple[
        tuple[V1RelaxedKKTBranch, str], ...
    ]
    execution_policy: V1AlexNetFQRNativeExecutionPolicy = field(
        repr=False, compare=False
    )
    memory_forecast: V1AlexNetFQRNativeMemoryForecast = field(
        repr=False, compare=False
    )
    issued_before_C_outcomes: bool = True
    C_outcomes_seen: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = NATIVE_FQR_AUTHORITY_SCOPE
    schema: str = NATIVE_FQR_PRE_C_REGISTRATION_SCHEMA
    _registration_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.family_design) is not V1FQRBranchFamilyDesign:
            raise TypeError("native C registration requires the exact family design")
        if type(self.current_mask) is not HardMask:
            raise TypeError("native C registration requires the exact current mask")
        if type(self.execution_policy) is not V1AlexNetFQRNativeExecutionPolicy:
            raise TypeError("native C registration requires the exact execution policy")
        if type(self.memory_forecast) is not V1AlexNetFQRNativeMemoryForecast:
            raise TypeError("native C registration requires the exact memory forecast")
        self.family_design.assert_integrity()
        self.execution_policy.assert_integrity()
        self.memory_forecast.assert_integrity()
        for name in (
            "graph_hash",
            "sampled_d_geometry_hash",
            "upstream_d_same_vjp_execution_proof_hash",
            "corrected_evidence_stream_identity_hash",
            "literal_jacobian_stream_identity_hash",
            "same_capture_and_vjp_registration_hash",
            "scorer_request_identity_hash",
            "scorer_request_state_hash",
            "ordered_score_route_axis_hash",
            "destination_transport_hash",
            "analytic_cache_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "chart_identity_hash",
            _identity(self.chart_identity_hash, name="chart identity hash"),
        )
        generation = _nonnegative_int(
            self.proposal_generation, name="proposal generation"
        )
        contexts = tuple(
            (
                _identity(name, name="C stratum"),
                _identity_axis(axis, name=f"{name} C context axis"),
            )
            for name, axis in self.certification_context_ids_by_stratum
        )
        draws = tuple(
            (
                _identity(name, name="C draw stratum"),
                _identity_axis(axis, name=f"{name} C draw axis"),
            )
            for name, axis in self.certification_draw_ids_by_stratum
        )
        starts = tuple(
            (
                _identity(name, name="C slice-start stratum"),
                _nonnegative_int(value, name="C slice-start index"),
            )
            for name, value in self.sealed_order_start_indices_by_stratum
        )
        names = tuple(name for name, _ in contexts)
        if (
            not contexts
            or tuple(name for name, _ in draws) != names
            or tuple(name for name, _ in starts) != names
            or any(not axis for _name, axis in contexts)
            or any(
                len(context_axis) != len(draw_axis)
                for (_name, context_axis), (_draw_name, draw_axis) in zip(
                    contexts, draws, strict=True
                )
            )
        ):
            raise ValueError("native C registration axes are empty or misaligned")
        complete_contexts = dict(
            self.family_design.registration.certification_context_ids_by_stratum
        )
        complete_draws = dict(self.family_design.registration.sealed_draw_ids_by_stratum)
        if tuple(complete_contexts) != names or tuple(complete_draws) != names:
            raise ValueError("native C registration names another family C stream")
        for (name, context_axis), (_draw_name, draw_axis), (_start_name, start) in zip(
            contexts, draws, starts, strict=True
        ):
            stop = start + len(context_axis)
            if (
                stop > len(complete_contexts[name])
                or stop > len(complete_draws[name])
                or tuple(complete_contexts[name][start:stop]) != context_axis
                or tuple(complete_draws[name][start:stop]) != draw_axis
            ):
                _refuse(
                    V1AlexNetFQRNativeRefusalCode.COORDINATE_MISMATCH,
                    "native C registration is not an exact sealed family slice",
                )
        expected_analytics = tuple(self.analytic_receipt_hashes_by_branch)
        branch_axis = tuple(bundle.branch for bundle in self.family_design.branch_bundles)
        if tuple(branch for branch, _hash in expected_analytics) != branch_axis:
            raise ValueError("native C analytic receipt axis differs from retained branches")
        analytics = tuple(
            (branch, _identity(value, name="analytic receipt hash"))
            for branch, value in expected_analytics
        )
        family = self.family_design.registration.prefrozen_family
        if (
            self.chart_identity_hash != family.chart_identity_hash
            or generation != family.proposal_generation
            or self.current_mask.mask_hash != family.current_mask_hash
            or self.current_mask.route_axis_hash != family.route_axis_hash
            or self.current_mask.route_count != family.route_count
            or self.scorer_request_state_hash != self.current_mask.mask_hash
            or self.memory_forecast.policy_hash != self.execution_policy.policy_hash
            or self.memory_forecast.branch_family_design_hash
            != self.family_design.design_hash
            or self.memory_forecast.context_slice_count
            != sum(len(axis) for _name, axis in contexts)
            or self.corrected_evidence_stream_identity_hash
            == self.literal_jacobian_stream_identity_hash
            or any(
                bundle.secant_design.ladder.raw_logit_stream_source_hash
                != self.corrected_evidence_stream_identity_hash
                for bundle in self.family_design.branch_bundles
            )
            or len(
                {
                    self.sampled_d_geometry_hash,
                    self.upstream_d_same_vjp_execution_proof_hash,
                    self.family_design.registration.certification_stream_commitment_hash,
                    self.same_capture_and_vjp_registration_hash,
                }
            )
            != 4
        ):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.COORDINATE_MISMATCH,
                "native C graph/family/mask/stream coordinates differ",
            )
        registration = self.family_design.registration
        policy = self.execution_policy
        if (
            registration.packed_direction_provider_source_hash
            != policy.packed_direction_provider_source_hash
            or registration.relaxed_gate_executor_source_hash
            != policy.relaxed_gate_executor_source_hash
            or registration.raw_logit_interval_verifier_source_hash
            != policy.raw_logit_interval_verifier_source_hash
        ):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "native C policy source hashes differ from the F/Q/R family",
            )
        if (
            self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
            or self.calibration_K_open_authority is not False
            or self.paired_oracle_O_open_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
        ):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.AUTHORITY_VIOLATION,
                "native C registration crossed its pre-C boundary",
            )
        if (
            self.schema != NATIVE_FQR_PRE_C_REGISTRATION_SCHEMA
            or self.authority_scope != NATIVE_FQR_AUTHORITY_SCOPE
        ):
            raise ValueError("native C registration schema/scope is invalid")
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "certification_context_ids_by_stratum", contexts)
        object.__setattr__(self, "certification_draw_ids_by_stratum", draws)
        object.__setattr__(self, "sealed_order_start_indices_by_stratum", starts)
        object.__setattr__(self, "analytic_receipt_hashes_by_branch", analytics)
        self.assert_integrity()

    @property
    def flat_context_ids(self) -> tuple[str, ...]:
        return tuple(
            value for _name, axis in self.certification_context_ids_by_stratum for value in axis
        )

    @property
    def flat_draw_ids(self) -> tuple[str, ...]:
        return tuple(
            value for _name, axis in self.certification_draw_ids_by_stratum for value in axis
        )

    @property
    def flat_stratum_ids(self) -> tuple[str, ...]:
        return tuple(
            name for name, axis in self.certification_context_ids_by_stratum for _ in axis
        )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "graph_hash": self.graph_hash,
                "family_design_hash": self.family_design.design_hash,
                "family_registration_hash": self.family_design.common_family_hash,
                "current_mask_hash": self.current_mask.mask_hash,
                "route_axis_hash": self.current_mask.route_axis_hash,
                "route_count": self.current_mask.route_count,
                "chart_identity_hash": self.chart_identity_hash,
                "proposal_generation": self.proposal_generation,
                "sampled_d_geometry_hash": self.sampled_d_geometry_hash,
                "upstream_d_same_vjp_execution_proof_hash": (
                    self.upstream_d_same_vjp_execution_proof_hash
                ),
                "certification_context_ids_by_stratum": (
                    self.certification_context_ids_by_stratum
                ),
                "certification_draw_ids_by_stratum": (
                    self.certification_draw_ids_by_stratum
                ),
                "sealed_order_start_indices_by_stratum": (
                    self.sealed_order_start_indices_by_stratum
                ),
                "corrected_evidence_stream_identity_hash": (
                    self.corrected_evidence_stream_identity_hash
                ),
                "literal_jacobian_stream_identity_hash": (
                    self.literal_jacobian_stream_identity_hash
                ),
                "same_capture_and_vjp_registration_hash": (
                    self.same_capture_and_vjp_registration_hash
                ),
                "scorer_request_identity_hash": self.scorer_request_identity_hash,
                "scorer_request_state_hash": self.scorer_request_state_hash,
                "ordered_score_route_axis_hash": self.ordered_score_route_axis_hash,
                "destination_transport_hash": self.destination_transport_hash,
                "analytic_cache_identity_hash": self.analytic_cache_identity_hash,
                "analytic_receipt_hashes_by_branch": tuple(
                    (branch.value, value)
                    for branch, value in self.analytic_receipt_hashes_by_branch
                ),
                "execution_policy_hash": self.execution_policy.policy_hash,
                "memory_forecast_hash": self.memory_forecast.forecast_hash,
                "issued_before_C_outcomes": self.issued_before_C_outcomes,
                "C_outcomes_seen": self.C_outcomes_seen,
                "calibration_K_open_authority": self.calibration_K_open_authority,
                "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
                "mask_commit_authority": self.mask_commit_authority,
                "terminal_authority": self.terminal_authority,
                "authority_scope": self.authority_scope,
                "raw_context_route_rows_serialized": False,
                "raw_analytic_route_geometry_serialized": False,
            }
        )

    @property
    def registration_hash(self) -> str:
        cached = self._registration_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        try:
            self.family_design.assert_integrity()
            self.execution_policy.assert_integrity()
            self.memory_forecast.assert_integrity()
        except (TypeError, ValueError, PermissionError) as error:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                f"native C registration child integrity failed: {error}",
            )
        if (
            self.issued_before_C_outcomes is not True
            or self.C_outcomes_seen is not False
            or self.calibration_K_open_authority is not False
            or self.paired_oracle_O_open_authority is not False
            or self.mask_commit_authority is not False
            or self.terminal_authority is not False
        ):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.AUTHORITY_VIOLATION,
                "native C registration gained downstream authority",
            )
        observed = content_hash(self._payload(), prefix=_REGISTRATION_PREFIX)
        if self._registration_hash is not None and self._registration_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "native C registration was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "native C registration hash does not match",
            )
        object.__setattr__(self, "_registration_hash", observed)
        return observed


def register_v1_alexnet_fqr_native_c_execution(
    *,
    graph_hash: str,
    family_design: V1FQRBranchFamilyDesign,
    current_mask: HardMask,
    chart_identity_hash: str,
    proposal_generation: int,
    sampled_d_geometry_hash: str,
    upstream_d_same_vjp_execution_proof_hash: str,
    certification_context_ids_by_stratum: Sequence[tuple[str, Sequence[str]]],
    certification_draw_ids_by_stratum: Sequence[tuple[str, Sequence[str]]],
    sealed_order_start_indices_by_stratum: Sequence[tuple[str, int]],
    corrected_evidence_stream_identity_hash: str,
    literal_jacobian_stream_identity_hash: str,
    same_capture_and_vjp_registration_hash: str,
    scorer_request_identity_hash: str,
    scorer_request_state_hash: str,
    ordered_score_route_axis_hash: str,
    destination_transport_hash: str,
    analytic_resolver: V1AlexNetFQRAnalyticBoundsResolver,
    execution_policy: V1AlexNetFQRNativeExecutionPolicy,
) -> V1AlexNetFQRNativeCPreExecutionRegistration:
    """Register the exact compact C slice before opening scorer execution."""

    if type(analytic_resolver) is not V1AlexNetFQRAnalyticBoundsResolver:
        raise TypeError("native C registration requires the exact analytic resolver")
    analytic_resolver.assert_integrity()
    execution_policy.assert_integrity()
    if analytic_resolver.resolver_source_hash != execution_policy.analytic_resolver_source_hash:
        _refuse(
            V1AlexNetFQRNativeRefusalCode.ANALYTIC_CACHE_MISMATCH,
            "analytic resolver source differs from the execution policy",
        )
    context_count = sum(
        len(tuple(axis)) for _name, axis in certification_context_ids_by_stratum
    )
    forecast = forecast_v1_alexnet_fqr_native_memory(
        policy=execution_policy,
        family_design=family_design,
        context_slice_count=context_count,
    )
    return V1AlexNetFQRNativeCPreExecutionRegistration(
        graph_hash=graph_hash,
        family_design=family_design,
        current_mask=current_mask,
        chart_identity_hash=chart_identity_hash,
        proposal_generation=proposal_generation,
        sampled_d_geometry_hash=sampled_d_geometry_hash,
        upstream_d_same_vjp_execution_proof_hash=(
            upstream_d_same_vjp_execution_proof_hash
        ),
        certification_context_ids_by_stratum=tuple(
            (name, tuple(axis))
            for name, axis in certification_context_ids_by_stratum
        ),
        certification_draw_ids_by_stratum=tuple(
            (name, tuple(axis)) for name, axis in certification_draw_ids_by_stratum
        ),
        sealed_order_start_indices_by_stratum=tuple(
            sealed_order_start_indices_by_stratum
        ),
        corrected_evidence_stream_identity_hash=(
            corrected_evidence_stream_identity_hash
        ),
        literal_jacobian_stream_identity_hash=literal_jacobian_stream_identity_hash,
        same_capture_and_vjp_registration_hash=(
            same_capture_and_vjp_registration_hash
        ),
        scorer_request_identity_hash=scorer_request_identity_hash,
        scorer_request_state_hash=scorer_request_state_hash,
        ordered_score_route_axis_hash=ordered_score_route_axis_hash,
        destination_transport_hash=destination_transport_hash,
        analytic_cache_identity_hash=analytic_resolver.cache_identity_hash,
        analytic_receipt_hashes_by_branch=(
            analytic_resolver.expected_receipt_hashes_by_branch
        ),
        execution_policy=execution_policy,
        memory_forecast=forecast,
    )


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be one mapping")
    return value


def _projection_children(value: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    matches: list[Mapping[str, object]] = []

    def visit(node: Mapping[str, object]) -> None:
        if (
            node.get("frozen_action_projection_streaming") is True
            or node.get("fibered_corrected_work_projection_streaming") is True
            or node.get("schema")
            == LABEL_NATIVE_PROJECTION_SCORER_CHILD_RECEIPT_SCHEMA
        ):
            matches.append(node)
        for key in (
            "loss_only_stream_tee_child_receipts",
            "child_streaming_receipts",
            "dual_chart_evidence_receipt",
            "dual_chart_jacobian_receipt",
        ):
            child = node.get(key, ())
            if isinstance(child, Mapping):
                visit(child)
            elif isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
                for item in child:
                    if isinstance(item, Mapping):
                        visit(item)

    visit(value)
    return tuple(matches)


def _projection_derivation_receipt_hash(
    child: Mapping[str, object],
) -> object:
    """Return the exact versioned scalar-projection derivation identity."""

    if child.get("fibered_corrected_work_projection_streaming") is True:
        return child.get("fibered_work_derivation_receipt_hash")
    if child.get("frozen_action_projection_streaming") is True:
        return child.get("projection_derivation_receipt_hash")
    if (
        child.get("schema")
        == LABEL_NATIVE_PROJECTION_SCORER_CHILD_RECEIPT_SCHEMA
    ):
        return child.get("projection_derivation_receipt_hash")
    return None


def _validate_measurement_request(
    registration: V1AlexNetFQRNativeCPreExecutionRegistration,
    request: MeasurementRequest,
) -> None:
    if type(request) is not MeasurementRequest:
        raise TypeError("native F/Q/R execution requires the exact measurement request")
    payload = _mapping(request.payload, name="native C measurement payload")
    contexts = tuple(request.context_selector.context_ids or ())
    if (
        request.request_id != registration.scorer_request_identity_hash
        or request.state_hash != registration.scorer_request_state_hash
        or request.state_hash != registration.current_mask.mask_hash
        or request.kind != "response_moments"
        or contexts != registration.flat_context_ids
        or payload.get("mask_hash") != registration.current_mask.mask_hash
        or payload.get("whole_physical_route_axis") is not True
        or payload.get("positive_removal_stream_coordinate")
        != _RAW_TARGET_LOGIT_COORDINATE
        or payload.get("positive_removal_stream_identity_hash")
        != registration.corrected_evidence_stream_identity_hash
        or payload.get("literal_jacobian_stream_identity_hash")
        != registration.literal_jacobian_stream_identity_hash
        or payload.get("response_pullback") != _RAW_TARGET_LOGIT_COTANGENT
        or payload.get("dual_chart_geometry") is not True
        or payload.get("corrected_work_rows_signed_raw_negative_G_phi_c") is not True
        or payload.get("fd_rows_signed_raw_negative_G_indicator_a_positive_c")
        is not True
        or payload.get("fd_rows_signed_raw_negative_G_phi_c") is not False
        or payload.get("absolute_rows_used_for_covariance") is not False
        or payload.get("affine_calibration_applied_to_attribution") is not False
        or payload.get("calibration_slope_applied_to_attribution") is not False
        or payload.get("calibration_offset_applied_to_attribution") is not False
        or payload.get("context_specific_bce_pullback_applied") is not False
        or payload.get("positive_removal_orientation") is not True
        or payload.get("terminal_outcomes_opened") is not False
    ):
        _refuse(
            V1AlexNetFQRNativeRefusalCode.COORDINATE_MISMATCH,
            "native C measurement request differs from its registered raw-logit slice",
        )


def _close_projection_scorer_receipt(
    *,
    registration: V1AlexNetFQRNativeCPreExecutionRegistration,
    request: MeasurementRequest,
    provider_receipt: Mapping[str, object],
    expected_evidence_projection_receipts: Sequence[Mapping[str, object]],
    expected_jacobian_projection_receipts: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], str]:
    """Verify one dual branch-fanout scorer call and emit the closure ABI."""

    contexts = registration.flat_context_ids
    corrected = registration.corrected_evidence_stream_identity_hash
    jacobian = registration.literal_jacobian_stream_identity_hash
    receipt = _mapping(provider_receipt, name="native C provider receipt")
    raw_packets = tuple(
        _mapping(value, name="native C raw packet receipt")
        for value in receipt.get("exact_conserving_target_logit_stream_receipts", ())
    )
    packet_count = int(receipt.get("context_packet_count", 0))
    capture_count = sum(int(value.get("capture_batch_count", 0)) for value in raw_packets)
    vjp_count = sum(int(value.get("vjp_count", 0)) for value in raw_packets)
    route_chunks = sum(int(value.get("route_chunk_count", 0)) for value in raw_packets)
    evidence_root = _mapping(
        receipt.get("dual_chart_evidence_receipt", {}),
        name="dual corrected-evidence receipt",
    )
    jacobian_root = _mapping(
        receipt.get("dual_chart_jacobian_receipt", {}),
        name="dual literal-J receipt",
    )
    evidence_children = _projection_children(evidence_root)
    jacobian_children = _projection_children(jacobian_root)
    expected_evidence = tuple(
        dict(_mapping(value, name="expected corrected projection receipt"))
        for value in expected_evidence_projection_receipts
    )
    expected_jacobian = tuple(
        dict(_mapping(value, name="expected Jacobian projection receipt"))
        for value in expected_jacobian_projection_receipts
    )
    branch_count = len(expected_evidence)
    common_invalid = (
        branch_count <= 0
        or len(expected_jacobian) != branch_count
        or receipt.get("schema") != "optimized_device_tripartite_panel_receipt_v1"
        or receipt.get("request_id") != request.request_id
        or receipt.get("exact_conserving_single_pass_target_logit_streaming")
        is not True
        or receipt.get("exact_conserving_single_pass_positive_removal_streaming")
        is not True
        or receipt.get("exact_conserving_single_pass_loss_streaming") is not False
        or receipt.get("full_tripartite_rows_materialized") is not False
        or int(receipt.get("context_count", -1)) != len(contexts)
        or int(receipt.get("route_count", -1)) != registration.current_mask.route_count
        or receipt.get("context_ids_hash")
        != content_hash(contexts, prefix="irg-device-panel-contexts-")
        or packet_count <= 0
        or len(raw_packets) != packet_count
        or capture_count != packet_count
        or vjp_count != packet_count
        or route_chunks <= 0
        or receipt.get("positive_removal_stream_coordinate")
        != _RAW_TARGET_LOGIT_COORDINATE
        or receipt.get("positive_removal_stream_identity_hash") != corrected
        or receipt.get("dual_chart_streaming") is not True
        or int(receipt.get("dual_chart_child_count", -1)) != 2
        or receipt.get("same_capture_and_vjp_required") is not True
        or receipt.get("shared_positive_removal_rows") is not False
        or receipt.get("corrected_evidence_stream_identity_hash") != corrected
        or receipt.get("literal_jacobian_stream_identity_hash") != jacobian
        or receipt.get("host_context_route_rows_transferred") is not False
        or int(receipt.get("host_context_route_row_transfer_count", -1)) != 0
        or int(receipt.get("host_context_route_row_transfer_bytes", -1)) != 0
        or int(receipt.get("route_chunk_size", -1))
        != registration.execution_policy.registered_route_chunk_size
        or len(evidence_children) != branch_count
        or len(jacobian_children) != branch_count
        or tuple(dict(child) for child in evidence_children) != expected_evidence
        or tuple(dict(child) for child in jacobian_children) != expected_jacobian
    )
    packet_invalid = any(
        int(value.get("capture_batch_count", 0)) != 1
        or int(value.get("vjp_count", 0)) != 1
        or int(value.get("context_count", 0)) <= 0
        or int(value.get("route_count", -1)) != registration.current_mask.route_count
        or int(value.get("route_chunk_count", 0)) <= 0
        or value.get("stream_coordinate") != _RAW_TARGET_LOGIT_COORDINATE
        or value.get("stream_identity_hash") != corrected
        or value.get("cotangent_coordinate") != _RAW_TARGET_LOGIT_COTANGENT
        or value.get("raw_target_logit_unit_pullback") is not True
        or value.get("affine_calibration_applied") is not False
        or value.get("context_specific_loss_pullback_applied") is not False
        or value.get("dual_corrected_jacobian_streaming") is not True
        or value.get("same_capture_and_vjp") is not True
        or value.get("literal_jacobian_stream_identity_hash") != jacobian
        or value.get("corrected_rows_excluded_from_response_geometry") is not True
        or value.get("scorer_adapter_host_context_route_rows_transferred") is not False
        for value in raw_packets
    )
    def invalid_projection_child(
        child: Mapping[str, object],
        *,
        stream: str,
    ) -> bool:
        is_label_native = (
            child.get("schema")
            == LABEL_NATIVE_PROJECTION_SCORER_CHILD_RECEIPT_SCHEMA
        )
        if (
            child.get("positive_removal_stream_coordinate")
            != _RAW_TARGET_LOGIT_COORDINATE
            or child.get("positive_removal_stream_identity_hash") != stream
            or child.get("host_context_route_rows_transferred") is not False
            or int(child.get("route_chunk_size", -1))
            != registration.execution_policy.registered_route_chunk_size
            or child.get("physical_mask_axis_hash")
            != registration.current_mask.route_axis_hash
            or child.get("ordered_score_route_axis_hash")
            != registration.ordered_score_route_axis_hash
        ):
            return True
        if is_label_native:
            expected_role = (
                "corrected_evidence" if stream == corrected else "literal_jacobian"
            )
            return (
                child.get("stream_role") != expected_role
                or child.get("stream_identity_hash") != stream
                or child.get("same_capture_and_vjp_registration_hash")
                != registration.same_capture_and_vjp_registration_hash
                or child.get("same_capture_and_vjp_execution_proof_available")
                is not False
                or child.get("registration_only_projection") is not True
                or child.get("C_projection_row_derivation_authority") is not False
                or child.get("corrected_then_literal_pairing_proved") is not True
                or child.get("shared_positive_removal_rows") is not False
                or child.get("full_tripartite_rows_materialized") is not False
                or int(child.get("host_context_route_row_transfer_count", -1)) != 0
                or int(child.get("host_context_route_row_transfer_bytes", -1)) != 0
                or child.get("raw_route_rows_retained") is not False
                or child.get("endpoint_hard_masks_materialized") is not False
                or child.get("candidate_selection_authority") is not False
                or child.get("mask_commit_authority") is not False
                or type(child.get("projection_derivation_receipt_hash")) is not str
                or not child.get("projection_derivation_receipt_hash")
            )
        return (
            (
                not (
                    child.get("fibered_corrected_work_projection_streaming") is True
                    or child.get("frozen_action_projection_streaming") is True
                )
                if stream == corrected
                else child.get("frozen_action_projection_streaming") is not True
            )
            or (
                child.get("within_native_fiber_cancellation_only") is not True
                if (
                    stream == corrected
                    and child.get("fibered_corrected_work_projection_streaming")
                    is True
                )
                else False
            )
            or (
                child.get("cross_native_fiber_work_cancellation_used") is not False
                if (
                    stream == corrected
                    and child.get("fibered_corrected_work_projection_streaming")
                    is True
                )
                else False
            )
            or (
                child.get("global_signed_projection_is_diagnostic_only") is not True
                if (
                    stream == corrected
                    and child.get("fibered_corrected_work_projection_streaming")
                    is True
                )
                else False
            )
            or (
                child.get("full_tripartite_rows_materialized") is not False
                if child.get("fibered_corrected_work_projection_streaming") is not True
                else (
                    child.get("scalar_fibered_work_derivation_authority") is not True
                    or child.get("route_scale_fiber_id_array_materialized") is not False
                )
            )
        )

    child_invalid = any(
        invalid_projection_child(child, stream=stream)
        for stream, children in (
            (corrected, evidence_children),
            (jacobian, jacobian_children),
        )
        for child in children
    )
    receipt_hashes = tuple(
        _projection_derivation_receipt_hash(child)
        for child in (*evidence_children, *jacobian_children)
    )
    if (
        common_invalid
        or packet_invalid
        or child_invalid
        or len(set(receipt_hashes)) != 2 * branch_count
    ):
        _refuse(
            V1AlexNetFQRNativeRefusalCode.SCORER_CONTRACT,
            "one-VJP dual branch-fanout scorer receipt did not close",
        )
    same_vjp_material_hash = content_hash(
        {
            "registration_hash": registration.registration_hash,
            "measurement_request_id": request.request_id,
            "provider_receipt": dict(receipt),
            "corrected_projection_receipt_hashes": tuple(
                _projection_derivation_receipt_hash(child)
                for child in evidence_children
            ),
            "literal_jacobian_projection_receipt_hashes": tuple(
                _projection_derivation_receipt_hash(child)
                for child in jacobian_children
            ),
            "one_scorer_invocation": True,
            "capture_batch_count": capture_count,
            "vjp_count": vjp_count,
            "same_capture_and_vjp_registration_hash": (
                registration.same_capture_and_vjp_registration_hash
            ),
            "upstream_d_same_vjp_execution_proof_hash": (
                registration.upstream_d_same_vjp_execution_proof_hash
            ),
        },
        prefix=_SCORER_PROOF_PREFIX,
    )
    # The existing typed dual-projection proof ABI deliberately accepts only
    # a bare SHA-256 source identity.  Hash the content-addressed material
    # receipt once more rather than discarding its complete binding.
    same_vjp_proof = hashlib.sha256(
        same_vjp_material_hash.encode("utf-8")
    ).hexdigest()
    payload: dict[str, object] = {
        "schema": _SCORER_RECEIPT_SCHEMA,
        "request_id": request.request_id,
        "context_ids_hash": receipt.get("context_ids_hash"),
        "context_count": len(contexts),
        "context_packet_count": packet_count,
        "route_count": registration.current_mask.route_count,
        "positive_removal_stream_coordinate": _RAW_TARGET_LOGIT_COORDINATE,
        "positive_removal_stream_identity_hash": corrected,
        "corrected_evidence_stream_identity_hash": corrected,
        "literal_jacobian_stream_identity_hash": jacobian,
        "same_capture_and_vjp_registration_hash": (
            registration.same_capture_and_vjp_registration_hash
        ),
        "same_capture_and_vjp_execution_proof_hash": same_vjp_proof,
        "same_capture_and_vjp_execution_material_hash": same_vjp_material_hash,
        "raw_target_logit_unit_pullback": True,
        "cotangent_coordinate": _RAW_TARGET_LOGIT_COTANGENT,
        "affine_calibration_applied": False,
        "context_specific_loss_pullback_applied": False,
        "calibration_offset_applied": False,
        "one_physical_scorer_invocation": True,
        "capture_batch_count": capture_count,
        "vjp_count": vjp_count,
        "dual_chart_streaming": True,
        "same_capture_and_vjp_required": True,
        "shared_positive_removal_rows": False,
        "loss_only_stream_tee": False,
        "child_streaming_receipts": (dict(evidence_root), dict(jacobian_root)),
        "packet_stream_receipts": tuple(dict(value) for value in raw_packets),
        "host_context_route_rows_transferred": False,
        "host_context_route_row_transfer_count": 0,
        "host_context_route_row_transfer_bytes": 0,
        "context_route_host_rows_forbidden": True,
    }
    canonical = MappingProxyType(
        {
            **payload,
            "receipt_hash": content_hash(payload, prefix=_SCORER_RECEIPT_PREFIX),
        }
    )
    return canonical, same_vjp_proof


def _device_type(device: Any) -> str:
    if torch is None:
        return "unavailable"
    try:
        return str(torch.device(device).type)
    except (TypeError, ValueError, RuntimeError):
        return "invalid"


def _fibered_directional_observation_bounds_from_v1_alexnet_fqr_receipt(
    *,
    result: V1AlexNetFQRBranchAnalyticBounds,
    design: Any,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
) -> V1FiberedDirectionalSecantObservationBounds:
    """Lower one pre-C AlexNet proof into the canonical fibered bounds ABI.

    The existing endpoint certificate is the minimum of independently sound
    absolute envelopes: the singleton route triangle fallback and, when
    available, the action-drive envelope that cancels only within a
    destination unit before summing nonnegative unit/layer burdens.  Each
    dominates the uniform cut-average native-fiber total variation used by
    canonical ``F``.  Importantly, this adapter never uses the observed or
    bounded magnitude of one globally signed corrected projection as fibered
    work.  The literal-J and raw-secant bounds retain their already registered
    global/complete-action meanings.
    """

    if type(result) is not V1AlexNetFQRBranchAnalyticBounds:
        raise TypeError("fibered AlexNet bounds require the exact branch result")
    if type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
        raise TypeError("fibered AlexNet bounds require the exact native-fiber receipt")
    native_fiber_axis_receipt.assert_integrity()
    _input_hash, analytic_receipt_hash, certificate_hash = result.assert_integrity()
    legacy = directional_observation_bounds_from_v1_alexnet_fqr_receipt(
        result=result,
        design=design,
    )
    certificate = result.corrected_work_endpoint_certificate
    endpoint_rows = np.asarray(
        certificate.endpoint_absolute_upper_bounds_by_stratum,
        dtype=np.float64,
    )
    endpoint_count = len(legacy.endpoint_action_hashes)
    if (
        endpoint_rows.ndim != 2
        or endpoint_rows.shape[1] != endpoint_count
        or not np.isfinite(endpoint_rows).all()
        or bool(np.any(endpoint_rows < 0.0))
    ):
        _refuse(
            V1AlexNetFQRNativeRefusalCode.NUMERICAL_SUPPORT_MISMATCH,
            "routewise corrected-work certificate cannot bound native-fiber F",
        )
    fibered_work_upper = np.nextafter(
        np.max(endpoint_rows, axis=0),
        np.full(endpoint_count, np.inf, dtype=np.float64),
    )
    if not np.isfinite(fibered_work_upper).all():
        _refuse(
            V1AlexNetFQRNativeRefusalCode.NUMERICAL_SUPPORT_MISMATCH,
            "native-fiber F support cannot be outward-enclosed in binary64",
        )
    bound_source_hash = content_hash(
        {
            "schema": "partition_free_v1_alexnet_native_fibered_fqr_bounds_adapter_v1",
            "analytic_receipt_hash": analytic_receipt_hash,
            "corrected_work_endpoint_certificate_hash": certificate_hash,
            "corrected_work_endpoint_derivation_kind": (
                certificate.derivation_kind
            ),
            "secant_design_hash": legacy.design_hash,
            "endpoint_action_hashes": legacy.endpoint_action_hashes,
            "native_fiber_axis_receipt_hash": (
                native_fiber_axis_receipt.receipt_hash
            ),
            "native_fiber_measure_hash": native_fiber_axis_receipt.measure_hash,
            "fibered_work_upper_by_endpoint": fibered_work_upper,
            "global_literal_jacobian_absolute_upper_by_endpoint": (
                legacy.jacobian_projection_absolute_upper_by_endpoint
            ),
            "remainder_absolute_upper_by_endpoint_and_dose": (
                legacy.remainder_absolute_upper_by_endpoint_and_dose
            ),
            "endpoint_absolute_envelope_dominates_native_fiber_total_variation": True,
            "uniform_cut_average_cannot_increase_cross_cut_nonnegative_sum": True,
            "global_corrected_scalar_projection_used": False,
            "registered_before_secant_outcomes": True,
        },
        prefix=_FIBERED_BOUNDS_ADAPTER_PREFIX,
    )
    return V1FiberedDirectionalSecantObservationBounds(
        design_hash=legacy.design_hash,
        endpoint_action_hashes=legacy.endpoint_action_hashes,
        native_fiber_axis_receipt_hash=native_fiber_axis_receipt.receipt_hash,
        native_fiber_measure_hash=native_fiber_axis_receipt.measure_hash,
        raw_logit_absolute_upper=legacy.raw_logit_absolute_upper,
        fibered_work_upper_by_endpoint=fibered_work_upper,
        global_literal_jacobian_absolute_upper_by_endpoint=(
            legacy.jacobian_projection_absolute_upper_by_endpoint
        ),
        remainder_absolute_upper_by_endpoint_and_dose=(
            legacy.remainder_absolute_upper_by_endpoint_and_dose
        ),
        bound_source_hash=bound_source_hash,
    )


@dataclass(frozen=True, slots=True)
class V1AlexNetFQRNativeCExecutionReceipt:
    registration_hash: str
    policy_hash: str
    memory_forecast_hash: str
    scorer_receipt_hash: str
    scorer_same_capture_and_vjp_execution_proof_hash: str
    branch_execution_receipt_hashes: tuple[tuple[V1RelaxedKKTBranch, str], ...]
    branch_record_hashes: tuple[tuple[V1RelaxedKKTBranch, tuple[str, ...]], ...]
    execution_mode: V1AlexNetFQRNativeExecutionMode
    device_type: str
    scorer_invocation_count: int = 1
    context_route_host_rows_transferred: bool = False
    fqr_decision_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = NATIVE_FQR_AUTHORITY_SCOPE
    schema: str = NATIVE_FQR_EXECUTION_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "policy_hash",
            "memory_forecast_hash",
            "scorer_receipt_hash",
            "scorer_same_capture_and_vjp_execution_proof_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if type(self.execution_mode) is not V1AlexNetFQRNativeExecutionMode:
            raise TypeError("native F/Q/R receipt has another execution mode type")
        executions = tuple(
            (branch, _identity(value, name="branch execution receipt hash"))
            for branch, value in self.branch_execution_receipt_hashes
        )
        records = tuple(
            (
                branch,
                tuple(_identity(value, name="directional record hash") for value in axis),
            )
            for branch, axis in self.branch_record_hashes
        )
        if (
            not executions
            or tuple(branch for branch, _ in executions)
            != tuple(branch for branch, _ in records)
            or any(not axis for _branch, axis in records)
            or self.scorer_invocation_count != 1
            or self.context_route_host_rows_transferred is not False
            or any(
                value is not False
                for value in (
                    self.fqr_decision_authority,
                    self.calibration_K_open_authority,
                    self.paired_oracle_O_open_authority,
                    self.mask_commit_authority,
                    self.terminal_authority,
                )
            )
            or self.schema != NATIVE_FQR_EXECUTION_RECEIPT_SCHEMA
            or self.authority_scope != NATIVE_FQR_AUTHORITY_SCOPE
        ):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.AUTHORITY_VIOLATION,
                "native F/Q/R execution receipt axes or authority are invalid",
            )
        object.__setattr__(self, "branch_execution_receipt_hashes", executions)
        object.__setattr__(self, "branch_record_hashes", records)
        object.__setattr__(self, "device_type", _identity(self.device_type, name="device type"))
        self.assert_integrity()

    @property
    def production_cuda_execution(self) -> bool:
        return bool(
            self.execution_mode is V1AlexNetFQRNativeExecutionMode.PRODUCTION_CUDA
            and self.device_type == "cuda"
        )

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "registration_hash": self.registration_hash,
                "policy_hash": self.policy_hash,
                "memory_forecast_hash": self.memory_forecast_hash,
                "scorer_receipt_hash": self.scorer_receipt_hash,
                "scorer_same_capture_and_vjp_execution_proof_hash": (
                    self.scorer_same_capture_and_vjp_execution_proof_hash
                ),
                "branch_execution_receipt_hashes": tuple(
                    (branch.value, value)
                    for branch, value in self.branch_execution_receipt_hashes
                ),
                "branch_record_hashes": tuple(
                    (branch.value, values) for branch, values in self.branch_record_hashes
                ),
                "execution_mode": self.execution_mode.value,
                "device_type": self.device_type,
                "production_cuda_execution": self.production_cuda_execution,
                "scorer_invocation_count": self.scorer_invocation_count,
                "context_route_host_rows_transferred": (
                    self.context_route_host_rows_transferred
                ),
                "fqr_decision_authority": self.fqr_decision_authority,
                "calibration_K_open_authority": self.calibration_K_open_authority,
                "paired_oracle_O_open_authority": self.paired_oracle_O_open_authority,
                "mask_commit_authority": self.mask_commit_authority,
                "terminal_authority": self.terminal_authority,
                "authority_scope": self.authority_scope,
            }
        )

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_EXECUTION_PREFIX)
        if self._receipt_hash is not None and self._receipt_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "native F/Q/R execution receipt was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "native F/Q/R execution receipt hash does not match",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1AlexNetFQRNativeCExecutionResult:
    receipt: V1AlexNetFQRNativeCExecutionReceipt
    branch_records: tuple[
        tuple[V1RelaxedKKTBranch, tuple[V1DirectionalSecantRecord, ...]], ...
    ] = field(repr=False, compare=False)
    schema: str = NATIVE_FQR_EXECUTION_RESULT_SCHEMA
    _result_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1AlexNetFQRNativeCExecutionReceipt:
            raise TypeError("native F/Q/R result requires the exact execution receipt")
        rows = tuple((branch, tuple(records)) for branch, records in self.branch_records)
        if (
            tuple(branch for branch, _ in rows)
            != tuple(branch for branch, _ in self.receipt.branch_record_hashes)
            or any(
                any(type(record) is not V1DirectionalSecantRecord for record in records)
                for _branch, records in rows
            )
            or tuple(
                (branch, tuple(record.record_hash for record in records))
                for branch, records in rows
            )
            != self.receipt.branch_record_hashes
            or self.schema != NATIVE_FQR_EXECUTION_RESULT_SCHEMA
        ):
            raise ValueError("native F/Q/R result differs from its receipt")
        object.__setattr__(self, "branch_records", rows)
        self.assert_integrity()

    @property
    def result_hash(self) -> str:
        cached = self._result_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.receipt.assert_integrity()
        for _branch, records in self.branch_records:
            for record in records:
                record.assert_integrity()
        observed = content_hash(
            {
                "schema": self.schema,
                "receipt_hash": self.receipt.receipt_hash,
                "branch_record_hashes": tuple(
                    (branch.value, tuple(record.record_hash for record in records))
                    for branch, records in self.branch_records
                ),
            },
            prefix=_RESULT_PREFIX,
        )
        if self._result_hash is not None and self._result_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "native F/Q/R execution result was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "native F/Q/R execution result hash does not match",
            )
        object.__setattr__(self, "_result_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1AlexNetFiberedFQRNativeCExecutionReceipt:
    """Exact receipt for native-fiber F, global literal-J Q, and whole-action R."""

    registration_hash: str
    policy_hash: str
    memory_forecast_hash: str
    fibered_mode_binding_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    fibered_work_projection_kernel_source_hash: str
    scorer_receipt_hash: str
    scorer_same_capture_and_vjp_execution_proof_hash: str
    branch_fibered_work_projection_receipt_hashes: tuple[
        tuple[V1RelaxedKKTBranch, str], ...
    ]
    branch_literal_jacobian_projection_receipt_hashes: tuple[
        tuple[V1RelaxedKKTBranch, str], ...
    ]
    branch_observation_bounds_hashes: tuple[
        tuple[V1RelaxedKKTBranch, str], ...
    ]
    branch_execution_receipt_hashes: tuple[tuple[V1RelaxedKKTBranch, str], ...]
    branch_record_hashes: tuple[
        tuple[V1RelaxedKKTBranch, tuple[str, ...]], ...
    ]
    execution_mode: V1AlexNetFQRNativeExecutionMode
    device_type: str
    scorer_invocation_count: int = 1
    native_fibered_work_projection: bool = True
    global_corrected_scalar_projection_used: bool = False
    literal_jacobian_projection_is_global_signed: bool = True
    secants_execute_complete_actions: bool = True
    context_route_host_rows_transferred: bool = False
    fqr_decision_authority: bool = False
    calibration_K_open_authority: bool = False
    paired_oracle_O_open_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    authority_scope: str = NATIVE_FQR_AUTHORITY_SCOPE
    schema: str = NATIVE_FIBERED_FQR_EXECUTION_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        for name in (
            "registration_hash",
            "policy_hash",
            "memory_forecast_hash",
            "fibered_mode_binding_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "scorer_receipt_hash",
            "scorer_same_capture_and_vjp_execution_proof_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "fibered_work_projection_kernel_source_hash",
            _sha256(
                self.fibered_work_projection_kernel_source_hash,
                name="fibered work projection kernel source hash",
            ),
        )
        if type(self.execution_mode) is not V1AlexNetFQRNativeExecutionMode:
            raise TypeError("fibered native F/Q/R receipt has another mode type")

        def hash_axis(
            values: Sequence[tuple[V1RelaxedKKTBranch, str]],
            *,
            name: str,
        ) -> tuple[tuple[V1RelaxedKKTBranch, str], ...]:
            rows = tuple(values)
            if not rows or any(
                type(branch) is not V1RelaxedKKTBranch for branch, _value in rows
            ):
                raise TypeError(f"{name} requires exact registered branch values")
            return tuple(
                (branch, _identity(value, name=f"{name} hash"))
                for branch, value in rows
            )

        fibered = hash_axis(
            self.branch_fibered_work_projection_receipt_hashes,
            name="fibered-work projection receipt",
        )
        jacobian = hash_axis(
            self.branch_literal_jacobian_projection_receipt_hashes,
            name="literal-J projection receipt",
        )
        bounds = hash_axis(
            self.branch_observation_bounds_hashes,
            name="fibered observation bounds",
        )
        executions = hash_axis(
            self.branch_execution_receipt_hashes,
            name="fibered secant execution receipt",
        )
        records = tuple(
            (
                branch,
                tuple(
                    _identity(value, name="fibered directional record hash")
                    for value in tuple(axis)
                ),
            )
            for branch, axis in tuple(self.branch_record_hashes)
        )
        branch_axes = tuple(
            tuple(branch for branch, _value in values)
            for values in (fibered, jacobian, bounds, executions, records)
        )
        if (
            any(axis != branch_axes[0] for axis in branch_axes[1:])
            or any(not axis for _branch, axis in records)
            or self.scorer_invocation_count != 1
            or self.native_fibered_work_projection is not True
            or self.global_corrected_scalar_projection_used is not False
            or self.literal_jacobian_projection_is_global_signed is not True
            or self.secants_execute_complete_actions is not True
            or self.context_route_host_rows_transferred is not False
            or any(
                value is not False
                for value in (
                    self.fqr_decision_authority,
                    self.calibration_K_open_authority,
                    self.paired_oracle_O_open_authority,
                    self.mask_commit_authority,
                    self.terminal_authority,
                )
            )
            or self.schema != NATIVE_FIBERED_FQR_EXECUTION_RECEIPT_SCHEMA
            or self.authority_scope != NATIVE_FQR_AUTHORITY_SCOPE
        ):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.AUTHORITY_VIOLATION,
                "fibered native F/Q/R receipt axes or authority are invalid",
            )
        object.__setattr__(
            self, "branch_fibered_work_projection_receipt_hashes", fibered
        )
        object.__setattr__(
            self, "branch_literal_jacobian_projection_receipt_hashes", jacobian
        )
        object.__setattr__(self, "branch_observation_bounds_hashes", bounds)
        object.__setattr__(self, "branch_execution_receipt_hashes", executions)
        object.__setattr__(self, "branch_record_hashes", records)
        object.__setattr__(
            self, "device_type", _identity(self.device_type, name="device type")
        )
        self.assert_integrity()

    @property
    def production_cuda_execution(self) -> bool:
        return bool(
            self.execution_mode is V1AlexNetFQRNativeExecutionMode.PRODUCTION_CUDA
            and self.device_type == "cuda"
        )

    def _payload(self) -> Mapping[str, object]:
        def serialized(
            values: Sequence[tuple[V1RelaxedKKTBranch, object]],
        ) -> tuple[tuple[str, object], ...]:
            return tuple((branch.value, value) for branch, value in values)

        return MappingProxyType(
            {
                "schema": self.schema,
                "registration_hash": self.registration_hash,
                "policy_hash": self.policy_hash,
                "memory_forecast_hash": self.memory_forecast_hash,
                "fibered_mode_binding_hash": self.fibered_mode_binding_hash,
                "native_fiber_axis_receipt_hash": (
                    self.native_fiber_axis_receipt_hash
                ),
                "native_fiber_measure_hash": self.native_fiber_measure_hash,
                "fibered_work_projection_kernel_source_hash": (
                    self.fibered_work_projection_kernel_source_hash
                ),
                "scorer_receipt_hash": self.scorer_receipt_hash,
                "scorer_same_capture_and_vjp_execution_proof_hash": (
                    self.scorer_same_capture_and_vjp_execution_proof_hash
                ),
                "branch_fibered_work_projection_receipt_hashes": serialized(
                    self.branch_fibered_work_projection_receipt_hashes
                ),
                "branch_literal_jacobian_projection_receipt_hashes": serialized(
                    self.branch_literal_jacobian_projection_receipt_hashes
                ),
                "branch_observation_bounds_hashes": serialized(
                    self.branch_observation_bounds_hashes
                ),
                "branch_execution_receipt_hashes": serialized(
                    self.branch_execution_receipt_hashes
                ),
                "branch_record_hashes": serialized(self.branch_record_hashes),
                "execution_mode": self.execution_mode.value,
                "device_type": self.device_type,
                "production_cuda_execution": self.production_cuda_execution,
                "scorer_invocation_count": self.scorer_invocation_count,
                "native_fibered_work_projection": (
                    self.native_fibered_work_projection
                ),
                "global_corrected_scalar_projection_used": (
                    self.global_corrected_scalar_projection_used
                ),
                "literal_jacobian_projection_is_global_signed": (
                    self.literal_jacobian_projection_is_global_signed
                ),
                "secants_execute_complete_actions": (
                    self.secants_execute_complete_actions
                ),
                "context_route_host_rows_transferred": (
                    self.context_route_host_rows_transferred
                ),
                "fqr_decision_authority": self.fqr_decision_authority,
                "calibration_K_open_authority": self.calibration_K_open_authority,
                "paired_oracle_O_open_authority": (
                    self.paired_oracle_O_open_authority
                ),
                "mask_commit_authority": self.mask_commit_authority,
                "terminal_authority": self.terminal_authority,
                "authority_scope": self.authority_scope,
            }
        )

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._payload(), prefix=_FIBERED_EXECUTION_PREFIX)
        if self._receipt_hash is not None and self._receipt_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "fibered native F/Q/R execution receipt was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "fibered native F/Q/R execution receipt hash does not match",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1AlexNetFiberedFQRNativeCExecutionResult:
    """Exact branch record carrier for canonical native-fiber observations."""

    receipt: V1AlexNetFiberedFQRNativeCExecutionReceipt
    branch_records: tuple[
        tuple[V1RelaxedKKTBranch, tuple[V1FiberedDirectionalRecord, ...]], ...
    ] = field(repr=False, compare=False)
    schema: str = NATIVE_FIBERED_FQR_EXECUTION_RESULT_SCHEMA
    _result_hash: str | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1AlexNetFiberedFQRNativeCExecutionReceipt:
            raise TypeError("fibered native result requires its exact receipt")
        rows = tuple((branch, tuple(records)) for branch, records in self.branch_records)
        if (
            tuple(branch for branch, _records in rows)
            != tuple(branch for branch, _hashes in self.receipt.branch_record_hashes)
            or any(
                any(type(record) is not V1FiberedDirectionalRecord for record in records)
                for _branch, records in rows
            )
            or tuple(
                (branch, tuple(record.record_hash for record in records))
                for branch, records in rows
            )
            != self.receipt.branch_record_hashes
            or self.schema != NATIVE_FIBERED_FQR_EXECUTION_RESULT_SCHEMA
        ):
            raise ValueError("fibered native result differs from its receipt")
        object.__setattr__(self, "branch_records", rows)
        self.assert_integrity()

    @property
    def result_hash(self) -> str:
        cached = self._result_hash
        return self.assert_integrity() if cached is None else cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.receipt.assert_integrity()
        for _branch, records in self.branch_records:
            for record in records:
                record.assert_integrity()
        observed = content_hash(
            {
                "schema": self.schema,
                "receipt_hash": self.receipt.receipt_hash,
                "branch_record_hashes": tuple(
                    (branch.value, tuple(record.record_hash for record in records))
                    for branch, records in self.branch_records
                ),
            },
            prefix=_FIBERED_RESULT_PREFIX,
        )
        if self._result_hash is not None and self._result_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "fibered native F/Q/R execution result was tampered",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "fibered native F/Q/R execution result hash does not match",
            )
        object.__setattr__(self, "_result_hash", observed)
        return observed


class V1AlexNetFQRNativeCExecutor:
    """Execute one registered shared-C slice without selecting an endpoint."""

    def __init__(
        self,
        *,
        scorer: Any,
        evaluator: Any,
        batch_factory: Callable[[tuple[str, ...]], Any],
        route_ids: Sequence[str],
        device: Any,
        target_readout_index: int = 0,
        native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt | None = None,
    ) -> None:
        execute = getattr(scorer, "accumulate_device_tripartite", None)
        if not callable(execute):
            raise TypeError("native F/Q/R scorer lacks device tripartite execution")
        if not callable(batch_factory):
            raise TypeError("native F/Q/R executor requires a batch factory")
        routes = _route_identity_axis(route_ids, name="native executor route axis")
        target = _nonnegative_int(target_readout_index, name="target readout index")
        if native_fiber_axis_receipt is not None:
            if (
                type(native_fiber_axis_receipt)
                is not V1NativeDestinationFiberAxisReceipt
            ):
                raise TypeError(
                    "canonical fibered execution requires the exact source-issued "
                    "native-fiber receipt"
                )
            native_fiber_axis_receipt.assert_integrity()
            if (
                native_fiber_axis_receipt.route_count != len(routes)
                or native_fiber_axis_receipt.route_id_axis_hash
                != ordered_string_axis_hash(
                    routes,
                    name="native-fiber executor ordered score-route axis",
                )
            ):
                raise ValueError(
                    "native-fiber receipt differs from the executor route transport"
                )
        self.scorer = scorer
        self.evaluator = evaluator
        self.batch_factory = batch_factory
        self.route_ids = routes
        self.device = device
        self.target_readout_index = target
        self.native_fiber_axis_receipt = native_fiber_axis_receipt
        self._fibered_projection_source_hash = (
            None
            if native_fiber_axis_receipt is None
            else fibered_work_projection_kernel_source_hash()
        )
        self._native_fiber_mode_binding_hash = (
            None
            if native_fiber_axis_receipt is None
            else content_hash(
                {
                    "schema": NATIVE_FIBERED_FQR_MODE_BINDING_SCHEMA,
                    "native_fiber_axis_receipt_hash": (
                        native_fiber_axis_receipt.receipt_hash
                    ),
                    "native_fiber_measure_hash": (
                        native_fiber_axis_receipt.measure_hash
                    ),
                    "route_count": len(routes),
                    "ordered_score_route_axis_hash": (
                        native_fiber_axis_receipt.route_id_axis_hash
                    ),
                    "executor_source_hash": _source_sha256(),
                    "fibered_work_projection_kernel_source_hash": (
                        self._fibered_projection_source_hash
                    ),
                    "native_fibered_work_projection": True,
                    "global_corrected_scalar_projection_used": False,
                    "literal_jacobian_projection_is_global_signed": True,
                    "secants_execute_complete_actions": True,
                    "authority_scope": NATIVE_FQR_AUTHORITY_SCOPE,
                },
                prefix=_FIBERED_MODE_BINDING_PREFIX,
            )
        )
        self._last_result: V1AlexNetFQRNativeCExecutionResult | None = None
        self._last_fibered_result: (
            V1AlexNetFiberedFQRNativeCExecutionResult | None
        ) = None

    @property
    def last_result(self) -> V1AlexNetFQRNativeCExecutionResult | None:
        return self._last_result

    @property
    def last_fibered_result(
        self,
    ) -> V1AlexNetFiberedFQRNativeCExecutionResult | None:
        return self._last_fibered_result

    @property
    def native_fiber_mode_binding_hash(self) -> str | None:
        return self._native_fiber_mode_binding_hash

    @staticmethod
    def _assert_live(
        registration: V1AlexNetFQRNativeCPreExecutionRegistration,
        *,
        live_graph_hash: str,
        live_current_mask_hash: str,
        live_chart_identity_hash: str,
        live_proposal_generation: int,
    ) -> None:
        if (
            _identity(live_graph_hash, name="live graph hash")
            != registration.graph_hash
            or _identity(live_current_mask_hash, name="live current-mask hash")
            != registration.current_mask.mask_hash
            or _identity(live_chart_identity_hash, name="live chart identity")
            != registration.chart_identity_hash
            or _nonnegative_int(
                live_proposal_generation, name="live proposal generation"
            )
            != registration.proposal_generation
        ):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.STALE_GRAPH,
                "native F/Q/R execution registration is stale",
            )

    @staticmethod
    def _projection_request(
        *,
        registration: V1AlexNetFQRNativeCPreExecutionRegistration,
        branch_index: int,
        role: str,
    ) -> V1DirectionalActionProjectionRequest:
        bundle = registration.family_design.branch_bundles[branch_index]
        return V1DirectionalActionProjectionRequest(
            design=bundle.secant_design,
            expected_context_ids=registration.flat_context_ids,
            expected_draw_ids=registration.flat_draw_ids,
            expected_stratum_ids=registration.flat_stratum_ids,
            sealed_order_start_indices_by_stratum=(
                registration.sealed_order_start_indices_by_stratum
            ),
            projection_role=role,
            corrected_evidence_stream_identity_hash=(
                registration.corrected_evidence_stream_identity_hash
            ),
            literal_jacobian_stream_identity_hash=(
                registration.literal_jacobian_stream_identity_hash
            ),
            same_capture_and_vjp_registration_hash=(
                registration.same_capture_and_vjp_registration_hash
            ),
            destination_transport_hash=registration.destination_transport_hash,
            scorer_request_identity_hash=registration.scorer_request_identity_hash,
            acquisition_lineage_hash=content_hash(
                {
                    "registration_hash": registration.registration_hash,
                    "branch": bundle.branch.value,
                    "role": role,
                    "secant_design_hash": bundle.secant_design.design_hash,
                    "scorer_request_identity_hash": (
                        registration.scorer_request_identity_hash
                    ),
                },
                prefix=_PROJECTION_LINEAGE_PREFIX,
            ),
            ordered_score_route_axis_hash=(
                registration.ordered_score_route_axis_hash
            ),
            registered_route_chunk_size=(
                registration.execution_policy.registered_route_chunk_size
            ),
            projection_kernel_source_hash=(
                registration.execution_policy.projection_kernel_source_hash
            ),
        )

    def _resolve_analytics(
        self,
        *,
        registration: V1AlexNetFQRNativeCPreExecutionRegistration,
        resolver: V1AlexNetFQRAnalyticBoundsResolver,
    ) -> tuple[V1AlexNetFQRBranchAnalyticBounds, ...]:
        if type(resolver) is not V1AlexNetFQRAnalyticBoundsResolver:
            raise TypeError("native F/Q/R execution requires the exact analytic resolver")
        resolver.assert_integrity(registration.analytic_cache_identity_hash)
        if resolver.resolver_source_hash != registration.execution_policy.analytic_resolver_source_hash:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.ANALYTIC_CACHE_MISMATCH,
                "live analytic resolver source differs from the pre-C policy",
            )
        results: list[V1AlexNetFQRBranchAnalyticBounds] = []
        for bundle, (branch, receipt_hash) in zip(
            registration.family_design.branch_bundles,
            registration.analytic_receipt_hashes_by_branch,
            strict=True,
        ):
            result = resolver.resolve(branch, expected_receipt_hash=receipt_hash)
            registered_input = next(
                (
                    value
                    for value in registration.family_design.registration.branch_inputs
                    if value.branch is branch
                ),
                None,
            )
            if (
                type(result) is not V1AlexNetFQRBranchAnalyticBounds
                or registered_input is None
                or result.analytic_input.input_hash
                != registered_input.input_hash
                or result.derivation_receipt.ladder_hash
                != bundle.secant_design.ladder_hash
                or result.derivation_receipt.chart_identity_hash
                != registration.chart_identity_hash
                or result.derivation_receipt.current_mask_hash
                != registration.current_mask.mask_hash
                or result.derivation_receipt.proposal_generation
                != registration.proposal_generation
            ):
                _refuse(
                    V1AlexNetFQRNativeRefusalCode.ANALYTIC_CACHE_MISMATCH,
                    "resolved analytic result differs from the registered branch chart",
                )
            results.append(result)
        return tuple(results)

    @staticmethod
    def _assert_sequential_support(
        *,
        bundle: Any,
        records: Sequence[V1DirectionalSecantRecord],
    ) -> None:
        """Refuse numerical observations outside the registered C support.

        In particular, a mathematical exact-linear ``R=0`` proof does not by
        itself cover the one-ulp radii carried by independently reported
        binary64 parent/dosed centers.  Those radii may only enter through an
        explicitly registered deterministic numerical uncertainty set.  The
        executor never clips a radius or silently replaces the sequential
        design's exact support with the looser raw interval envelope.
        """

        design = bundle.sequential_design
        names = tuple(design.stratum_names)
        for record in records:
            try:
                stratum = names.index(record.stratum_id)
            except ValueError as error:
                _refuse(
                    V1AlexNetFQRNativeRefusalCode.COORDINATE_MISMATCH,
                    f"secant record names an unregistered C stratum: {error}",
                )
            observed = np.stack(
                (
                    record.f_upper_observation_by_endpoint,
                    record.q_upper_observation_by_endpoint,
                    record.r_upper_observation_by_endpoint[:, 0],
                    record.r_upper_observation_by_endpoint[:, 1],
                ),
                axis=1,
            )
            support = np.stack(
                (
                    design.analytic_f_upper_by_stratum_endpoint[stratum],
                    design.analytic_q_upper_by_stratum_endpoint[stratum],
                    design.analytic_r_upper_by_stratum_endpoint_and_dose[
                        stratum, :, 0
                    ],
                    design.analytic_r_upper_by_stratum_endpoint_and_dose[
                        stratum, :, 1
                    ],
                ),
                axis=1,
            )
            if (
                observed.shape != support.shape
                or not np.isfinite(observed).all()
                or bool(np.any(observed < 0.0))
                or bool(np.any(observed > support))
            ):
                r_mismatch = bool(np.any(observed[:, 2:] > support[:, 2:]))
                detail = (
                    "mathematical exact-linear R support does not cover the "
                    "binary64 enclosure; register deterministic numerical "
                    "uncertainty without spending alpha"
                    if r_mismatch and bool(np.any(support[:, 2:] == 0.0))
                    else "secant F/Q/R observation exceeded its registered "
                    "sequential support"
                )
                _refuse(
                    V1AlexNetFQRNativeRefusalCode.NUMERICAL_SUPPORT_MISMATCH,
                    detail,
                )

    @staticmethod
    def _assert_fibered_sequential_support(
        *,
        bundle: Any,
        records: Sequence[V1FiberedDirectionalRecord],
    ) -> None:
        """Refuse any fibered interval that exceeds the frozen base support."""

        design = bundle.sequential_design
        names = tuple(design.stratum_names)
        for record in records:
            if type(record) is not V1FiberedDirectionalRecord:
                raise TypeError("canonical fibered C requires exact fibered records")
            try:
                stratum = names.index(record.stratum_id)
            except ValueError as error:
                _refuse(
                    V1AlexNetFQRNativeRefusalCode.COORDINATE_MISMATCH,
                    f"fibered secant record names an unregistered C stratum: {error}",
                )
            lower = np.stack(
                (
                    record.f_lower_observation_by_endpoint,
                    record.q_lower_observation_by_endpoint,
                    record.r_lower_observation_by_endpoint[:, 0],
                    record.r_lower_observation_by_endpoint[:, 1],
                ),
                axis=1,
            )
            upper = np.stack(
                (
                    record.f_upper_observation_by_endpoint,
                    record.q_upper_observation_by_endpoint,
                    record.r_upper_observation_by_endpoint[:, 0],
                    record.r_upper_observation_by_endpoint[:, 1],
                ),
                axis=1,
            )
            support = np.stack(
                (
                    design.analytic_f_upper_by_stratum_endpoint[stratum],
                    design.analytic_q_upper_by_stratum_endpoint[stratum],
                    design.analytic_r_upper_by_stratum_endpoint_and_dose[
                        stratum, :, 0
                    ],
                    design.analytic_r_upper_by_stratum_endpoint_and_dose[
                        stratum, :, 1
                    ],
                ),
                axis=1,
            )
            if (
                lower.shape != support.shape
                or upper.shape != support.shape
                or not np.isfinite(lower).all()
                or not np.isfinite(upper).all()
                or bool(np.any(lower < 0.0))
                or bool(np.any(lower > upper))
                or bool(np.any(upper > support))
            ):
                _refuse(
                    V1AlexNetFQRNativeRefusalCode.NUMERICAL_SUPPORT_MISMATCH,
                    "native-fiber F/global-J Q/whole-action R interval exceeded "
                    "its registered sequential support",
                )

    def execute(
        self,
        *,
        registration: V1AlexNetFQRNativeCPreExecutionRegistration,
        measurement_request: MeasurementRequest,
        analytic_resolver: V1AlexNetFQRAnalyticBoundsResolver,
        live_graph_hash: str,
        live_current_mask_hash: str,
        live_chart_identity_hash: str,
        live_proposal_generation: int,
    ) -> V1AlexNetFQRNativeCExecutionResult:
        if type(registration) is not V1AlexNetFQRNativeCPreExecutionRegistration:
            raise TypeError("native F/Q/R executor requires the exact registration")
        registration.assert_integrity()
        self._assert_live(
            registration,
            live_graph_hash=live_graph_hash,
            live_current_mask_hash=live_current_mask_hash,
            live_chart_identity_hash=live_chart_identity_hash,
            live_proposal_generation=live_proposal_generation,
        )
        _validate_measurement_request(registration, measurement_request)
        policy = registration.execution_policy
        device_type = _device_type(self.device)
        if policy.execution_mode is V1AlexNetFQRNativeExecutionMode.PRODUCTION_CUDA:
            cuda_available = bool(torch is not None and torch.cuda.is_available())
            if device_type != "cuda" or not cuda_available:
                _refuse(
                    V1AlexNetFQRNativeRefusalCode.CUDA_REQUIRED,
                    "production native F/Q/R execution has no CPU fallback",
                )
        elif device_type != "cpu":
            _refuse(
                V1AlexNetFQRNativeRefusalCode.CUDA_REQUIRED,
                "authority-inert reference execution is CPU-test-only",
            )
        if (
            int(getattr(self.scorer, "response_route_chunk_size", -1))
            != policy.registered_route_chunk_size
            or int(getattr(self.scorer, "context_packet_size", -1))
            != policy.registered_context_packet_size
            or len(self.route_ids) != registration.current_mask.route_count
            or ordered_string_axis_hash(
                self.route_ids, name="native F/Q/R ordered score-route axis"
            )
            != registration.ordered_score_route_axis_hash
        ):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.COORDINATE_MISMATCH,
                "native scorer or route-axis execution geometry changed",
            )
        registration.memory_forecast.assert_integrity()

        # Analytic results, observation supports, and interval verifiers are
        # resolved before the scorer is called.  A stale cache can therefore
        # never inspect C and then decide which bound to apply.
        analytics = self._resolve_analytics(
            registration=registration,
            resolver=analytic_resolver,
        )
        bounds = tuple(
            directional_observation_bounds_from_v1_alexnet_fqr_receipt(
                result=result,
                design=bundle.secant_design,
            )
            for bundle, result in zip(
                registration.family_design.branch_bundles,
                analytics,
                strict=True,
            )
        )
        verifiers = tuple(
            build_v1_alexnet_fqr_raw_logit_interval_verifier(
                result=result,
                design=bundle.secant_design,
            )
            for bundle, result in zip(
                registration.family_design.branch_bundles,
                analytics,
                strict=True,
            )
        )

        evidence_requests: list[V1DirectionalActionProjectionRequest] = []
        jacobian_requests: list[V1DirectionalActionProjectionRequest] = []
        evidence_consumers: list[V1LadderProjectionGPUConsumer] = []
        jacobian_consumers: list[V1LadderProjectionGPUConsumer] = []
        for branch_index, _bundle in enumerate(
            registration.family_design.branch_bundles
        ):
            evidence_request = self._projection_request(
                registration=registration,
                branch_index=branch_index,
                role="corrected_evidence",
            )
            jacobian_request = self._projection_request(
                registration=registration,
                branch_index=branch_index,
                role="literal_jacobian",
            )
            evidence_requests.append(evidence_request)
            jacobian_requests.append(jacobian_request)
            evidence_consumers.append(
                V1LadderProjectionGPUConsumer(
                    evidence_request,
                    route_ids=self.route_ids,
                    device=self.device,
                )
            )
            jacobian_consumers.append(
                V1LadderProjectionGPUConsumer(
                    jacobian_request,
                    route_ids=self.route_ids,
                    device=self.device,
                )
            )
        fanout = build_route_chunked_dual_chart_fanout(
            evidence_consumers=tuple(evidence_consumers),
            jacobian_consumers=tuple(jacobian_consumers),
        )
        provider_receipt = self.scorer.accumulate_device_tripartite(
            measurement_request,
            accumulator=fanout,
        )
        if not isinstance(provider_receipt, Mapping):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.SCORER_CONTRACT,
                "native F/Q/R scorer returned no typed receipt",
            )
        scorer_receipt, scorer_proof_hash = _close_projection_scorer_receipt(
            registration=registration,
            request=measurement_request,
            provider_receipt=provider_receipt,
            expected_evidence_projection_receipts=tuple(
                consumer.route_streaming_receipt()
                for consumer in evidence_consumers
            ),
            expected_jacobian_projection_receipts=tuple(
                consumer.route_streaming_receipt()
                for consumer in jacobian_consumers
            ),
        )

        branch_rows: list[
            tuple[V1RelaxedKKTBranch, tuple[V1DirectionalSecantRecord, ...]]
        ] = []
        branch_execution_receipts: list[
            tuple[V1RelaxedKKTBranch, V1PackedDirectionalSecantExecutionReceipt]
        ] = []
        for bundle, evidence_request, jacobian_request, evidence_consumer, jacobian_consumer, bound, verifier in zip(
            registration.family_design.branch_bundles,
            evidence_requests,
            jacobian_requests,
            evidence_consumers,
            jacobian_consumers,
            bounds,
            verifiers,
            strict=True,
        ):
            plan = V1DirectionalDualProjectionPlan(
                requests=(evidence_request, jacobian_request)
            )
            evidence_acquisition = evidence_consumer.close_acquisition(scorer_receipt)
            jacobian_acquisition = jacobian_consumer.close_acquisition(scorer_receipt)
            proof = bind_v1_directional_dual_projection_scorer_proof(
                plan=plan,
                corrected_evidence_acquisition=evidence_acquisition,
                literal_jacobian_acquisition=jacobian_acquisition,
                scorer_same_capture_and_vjp_execution_proof_hash=scorer_proof_hash,
            )
            dual = close_v1_directional_dual_projection(
                plan=plan,
                corrected_evidence_acquisition=evidence_acquisition,
                literal_jacobian_acquisition=jacobian_acquisition,
                scorer_same_capture_and_vjp_execution_proof=proof,
            )
            provider = build_v1_native_directional_secant_provider(
                dual_projection=dual,
                design=bundle.secant_design,
                evaluator=self.evaluator,
                batch_factory=self.batch_factory,
                parent_mask=registration.current_mask,
                relaxed_gate_executor_source_hash=(
                    policy.relaxed_gate_executor_source_hash
                ),
                raw_logit_interval_verifier_source_hash=(
                    policy.raw_logit_interval_verifier_source_hash
                ),
                raw_logit_interval_verifier=verifier,
                observation_bounds=bound,
                maximum_packed_lanes=policy.maximum_packed_lanes,
                maximum_gate_workspace_bytes=policy.maximum_gate_workspace_bytes,
                target_readout_index=self.target_readout_index,
            )
            execution = provider.execute()
            execution.assert_integrity()
            if (
                tuple(DIRECTIONAL_SECANT_DOSES) != (0.5, 1.0)
                or execution.endpoint_action_hashes
                != bundle.secant_design.endpoint_action_hashes
                or execution.context_ids_by_stratum
                != registration.certification_context_ids_by_stratum
                or execution.sealed_order_start_indices_by_stratum
                != registration.sealed_order_start_indices_by_stratum
            ):
                _refuse(
                    V1AlexNetFQRNativeRefusalCode.PARTIAL_PROJECTION,
                    "directional secant execution changed endpoints, doses, or C order",
                )
            self._assert_sequential_support(
                bundle=bundle,
                records=execution.records,
            )
            branch_execution_receipts.append((bundle.branch, execution))
            branch_rows.append((bundle.branch, tuple(execution.records)))

        execution_receipt = V1AlexNetFQRNativeCExecutionReceipt(
            registration_hash=registration.registration_hash,
            policy_hash=policy.policy_hash,
            memory_forecast_hash=registration.memory_forecast.forecast_hash,
            scorer_receipt_hash=_identity(
                scorer_receipt["receipt_hash"], name="scorer receipt hash"
            ),
            scorer_same_capture_and_vjp_execution_proof_hash=scorer_proof_hash,
            branch_execution_receipt_hashes=tuple(
                (branch, execution.receipt_hash)
                for branch, execution in branch_execution_receipts
            ),
            branch_record_hashes=tuple(
                (branch, tuple(record.record_hash for record in records))
                for branch, records in branch_rows
            ),
            execution_mode=policy.execution_mode,
            device_type=device_type,
        )
        result = V1AlexNetFQRNativeCExecutionResult(
            receipt=execution_receipt,
            branch_records=tuple(branch_rows),
        )
        self._last_result = result
        return result

    def execute_fqr_directional_secants(
        self,
        **kwargs: Any,
    ) -> tuple[
        tuple[V1RelaxedKKTBranch, tuple[V1DirectionalSecantRecord, ...]], ...
    ]:
        """Backend ABI: return the exact typed ``(branch, records)`` axis."""

        return self.execute(**kwargs).branch_records

    def execute_fibered(
        self,
        *,
        registration: V1AlexNetFQRNativeCPreExecutionRegistration,
        measurement_request: MeasurementRequest,
        analytic_resolver: V1AlexNetFQRAnalyticBoundsResolver,
        live_graph_hash: str,
        live_current_mask_hash: str,
        live_chart_identity_hash: str,
        live_proposal_generation: int,
    ) -> V1AlexNetFiberedFQRNativeCExecutionResult:
        """Execute canonical native-fiber F/global-J Q/whole-action R.

        This is a strict sibling of :meth:`execute`.  It deliberately returns
        another exact result type so a legacy globally scalar corrected-F row
        cannot be mistaken for a native-fiber work observation during replay.
        """

        if type(registration) is not V1AlexNetFQRNativeCPreExecutionRegistration:
            raise TypeError("fibered native F/Q/R requires the exact registration")
        registration.assert_integrity()
        self._assert_live(
            registration,
            live_graph_hash=live_graph_hash,
            live_current_mask_hash=live_current_mask_hash,
            live_chart_identity_hash=live_chart_identity_hash,
            live_proposal_generation=live_proposal_generation,
        )
        _validate_measurement_request(registration, measurement_request)
        native_fibers = self.native_fiber_axis_receipt
        if type(native_fibers) is not V1NativeDestinationFiberAxisReceipt:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.AUTHORITY_VIOLATION,
                "canonical fibered execution lacks a source-issued native-fiber axis",
            )
        native_fibers.assert_integrity()
        live_fibered_source = fibered_work_projection_kernel_source_hash()
        if (
            self._fibered_projection_source_hash != live_fibered_source
            or self._native_fiber_mode_binding_hash is None
            or native_fibers.route_axis_hash
            != registration.current_mask.route_axis_hash
            or native_fibers.route_id_axis_hash
            != registration.ordered_score_route_axis_hash
            or native_fibers.route_count != registration.current_mask.route_count
        ):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.HASH_MISMATCH,
                "native-fiber source, measure, or route transport changed",
            )

        policy = registration.execution_policy
        device_type = _device_type(self.device)
        if policy.execution_mode is V1AlexNetFQRNativeExecutionMode.PRODUCTION_CUDA:
            cuda_available = bool(torch is not None and torch.cuda.is_available())
            if device_type != "cuda" or not cuda_available:
                _refuse(
                    V1AlexNetFQRNativeRefusalCode.CUDA_REQUIRED,
                    "production fibered native F/Q/R execution has no CPU fallback",
                )
        elif device_type != "cpu":
            _refuse(
                V1AlexNetFQRNativeRefusalCode.CUDA_REQUIRED,
                "authority-inert fibered reference execution is CPU-test-only",
            )
        if (
            int(getattr(self.scorer, "response_route_chunk_size", -1))
            != policy.registered_route_chunk_size
            or int(getattr(self.scorer, "context_packet_size", -1))
            != policy.registered_context_packet_size
            or len(self.route_ids) != registration.current_mask.route_count
            or ordered_string_axis_hash(
                self.route_ids,
                name="fibered native F/Q/R ordered score-route axis",
            )
            != registration.ordered_score_route_axis_hash
        ):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.COORDINATE_MISMATCH,
                "fibered native scorer or route execution geometry changed",
            )
        registration.memory_forecast.assert_integrity()

        # All support objects are selected and closed before C is observed.
        analytics = self._resolve_analytics(
            registration=registration,
            resolver=analytic_resolver,
        )
        bounds = tuple(
            _fibered_directional_observation_bounds_from_v1_alexnet_fqr_receipt(
                result=result,
                design=bundle.secant_design,
                native_fiber_axis_receipt=native_fibers,
            )
            for bundle, result in zip(
                registration.family_design.branch_bundles,
                analytics,
                strict=True,
            )
        )
        verifiers = tuple(
            build_v1_alexnet_fqr_raw_logit_interval_verifier(
                result=result,
                design=bundle.secant_design,
            )
            for bundle, result in zip(
                registration.family_design.branch_bundles,
                analytics,
                strict=True,
            )
        )

        branch_count = len(registration.family_design.branch_bundles)
        per_branch_fiber_cap = policy.maximum_native_workspace_bytes // branch_count
        if per_branch_fiber_cap <= 0:
            _refuse(
                V1AlexNetFQRNativeRefusalCode.WORKSPACE_EXCEEDED,
                "registered workspace cannot admit one fibered branch consumer",
            )
        evidence_requests: list[V1DirectionalActionProjectionRequest] = []
        fibered_requests: list[V1FiberedWorkProjectionRequest] = []
        jacobian_requests: list[V1DirectionalActionProjectionRequest] = []
        evidence_consumers: list[V1FiberedWorkProjectionGPUConsumer] = []
        jacobian_consumers: list[V1LadderProjectionGPUConsumer] = []
        for branch_index, _bundle in enumerate(
            registration.family_design.branch_bundles
        ):
            evidence_request = self._projection_request(
                registration=registration,
                branch_index=branch_index,
                role="corrected_evidence",
            )
            fibered_request = V1FiberedWorkProjectionRequest(
                projection_request=evidence_request,
                native_fiber_axis=native_fibers,
                maximum_temporary_device_bytes=per_branch_fiber_cap,
                maximum_registered_device_bytes=per_branch_fiber_cap,
            )
            jacobian_request = self._projection_request(
                registration=registration,
                branch_index=branch_index,
                role="literal_jacobian",
            )
            evidence_requests.append(evidence_request)
            fibered_requests.append(fibered_request)
            jacobian_requests.append(jacobian_request)
            evidence_consumers.append(
                V1FiberedWorkProjectionGPUConsumer(
                    fibered_request,
                    route_ids=self.route_ids,
                    device=self.device,
                    production=policy.production_cuda_authority,
                )
            )
            jacobian_consumers.append(
                V1LadderProjectionGPUConsumer(
                    jacobian_request,
                    route_ids=self.route_ids,
                    device=self.device,
                )
            )
        fanout = build_route_chunked_dual_chart_fanout(
            evidence_consumers=tuple(evidence_consumers),
            jacobian_consumers=tuple(jacobian_consumers),
        )
        provider_receipt = self.scorer.accumulate_device_tripartite(
            measurement_request,
            accumulator=fanout,
        )
        if not isinstance(provider_receipt, Mapping):
            _refuse(
                V1AlexNetFQRNativeRefusalCode.SCORER_CONTRACT,
                "fibered native scorer returned no typed receipt",
            )
        scorer_receipt, scorer_proof_hash = _close_projection_scorer_receipt(
            registration=registration,
            request=measurement_request,
            provider_receipt=provider_receipt,
            expected_evidence_projection_receipts=tuple(
                consumer.route_streaming_receipt()
                for consumer in evidence_consumers
            ),
            expected_jacobian_projection_receipts=tuple(
                consumer.route_streaming_receipt()
                for consumer in jacobian_consumers
            ),
        )

        branch_rows: list[
            tuple[V1RelaxedKKTBranch, tuple[V1FiberedDirectionalRecord, ...]]
        ] = []
        branch_executions: list[
            tuple[
                V1RelaxedKKTBranch,
                V1PackedFiberedDirectionalSecantExecutionReceipt,
            ]
        ] = []
        branch_fibered_projection_hashes: list[
            tuple[V1RelaxedKKTBranch, str]
        ] = []
        branch_jacobian_projection_hashes: list[
            tuple[V1RelaxedKKTBranch, str]
        ] = []
        branch_bound_hashes: list[tuple[V1RelaxedKKTBranch, str]] = []
        for (
            bundle,
            fibered_request,
            jacobian_request,
            fibered_consumer,
            jacobian_consumer,
            bound,
            verifier,
        ) in zip(
            registration.family_design.branch_bundles,
            fibered_requests,
            jacobian_requests,
            evidence_consumers,
            jacobian_consumers,
            bounds,
            verifiers,
            strict=True,
        ):
            fibered_projection = fibered_consumer.close()
            jacobian_acquisition = jacobian_consumer.close_acquisition(
                scorer_receipt
            )
            closures = build_v1_fibered_directional_projection_closures(
                design=bundle.secant_design,
                native_fiber_axis_receipt=native_fibers,
                fibered_work_projection_request=fibered_request,
                fibered_work_projection=fibered_projection,
                literal_jacobian_projection_request=jacobian_request,
                literal_jacobian_acquisition=jacobian_acquisition,
                corrected_evidence_stream_identity_hash=(
                    registration.corrected_evidence_stream_identity_hash
                ),
                literal_jacobian_stream_identity_hash=(
                    registration.literal_jacobian_stream_identity_hash
                ),
                same_capture_and_vjp_execution_proof_hash=scorer_proof_hash,
            )
            provider = build_v1_native_fibered_directional_secant_provider(
                projection_closures=closures,
                native_fiber_axis_receipt=native_fibers,
                sealed_order_start_indices_by_stratum=(
                    registration.sealed_order_start_indices_by_stratum
                ),
                design=bundle.secant_design,
                evaluator=self.evaluator,
                batch_factory=self.batch_factory,
                parent_mask=registration.current_mask,
                relaxed_gate_executor_source_hash=(
                    policy.relaxed_gate_executor_source_hash
                ),
                raw_logit_interval_verifier_source_hash=(
                    policy.raw_logit_interval_verifier_source_hash
                ),
                raw_logit_interval_verifier=verifier,
                observation_bounds=bound,
                maximum_packed_lanes=policy.maximum_packed_lanes,
                maximum_gate_workspace_bytes=policy.maximum_gate_workspace_bytes,
                target_readout_index=self.target_readout_index,
            )
            execution = provider.execute()
            if type(execution) is not V1PackedFiberedDirectionalSecantExecutionReceipt:
                raise TypeError("fibered secant provider returned another receipt type")
            execution.assert_integrity()
            if (
                tuple(DIRECTIONAL_SECANT_DOSES) != (0.5, 1.0)
                or execution.endpoint_action_hashes
                != bundle.secant_design.endpoint_action_hashes
                or execution.context_ids_by_stratum
                != registration.certification_context_ids_by_stratum
                or execution.sealed_order_start_indices_by_stratum
                != registration.sealed_order_start_indices_by_stratum
                or execution.native_fiber_axis_receipt_hash
                != native_fibers.receipt_hash
                or execution.native_fiber_measure_hash != native_fibers.measure_hash
                or execution.global_corrected_scalar_projection_used is not False
                or execution.literal_jacobian_projection_is_global_signed is not True
                or execution.secants_execute_complete_actions is not True
            ):
                _refuse(
                    V1AlexNetFQRNativeRefusalCode.PARTIAL_PROJECTION,
                    "fibered secant execution changed F/J/R identity, C order, or doses",
                )
            self._assert_fibered_sequential_support(
                bundle=bundle,
                records=execution.records,
            )
            branch_executions.append((bundle.branch, execution))
            branch_rows.append((bundle.branch, tuple(execution.records)))
            branch_fibered_projection_hashes.append(
                (bundle.branch, fibered_projection.receipt.receipt_hash)
            )
            branch_jacobian_projection_hashes.append(
                (
                    bundle.branch,
                    jacobian_acquisition.result.receipt.receipt_hash,
                )
            )
            branch_bound_hashes.append((bundle.branch, bound.bounds_hash))

        execution_receipt = V1AlexNetFiberedFQRNativeCExecutionReceipt(
            registration_hash=registration.registration_hash,
            policy_hash=policy.policy_hash,
            memory_forecast_hash=registration.memory_forecast.forecast_hash,
            fibered_mode_binding_hash=self._native_fiber_mode_binding_hash,
            native_fiber_axis_receipt_hash=native_fibers.receipt_hash,
            native_fiber_measure_hash=native_fibers.measure_hash,
            fibered_work_projection_kernel_source_hash=live_fibered_source,
            scorer_receipt_hash=_identity(
                scorer_receipt["receipt_hash"], name="scorer receipt hash"
            ),
            scorer_same_capture_and_vjp_execution_proof_hash=scorer_proof_hash,
            branch_fibered_work_projection_receipt_hashes=tuple(
                branch_fibered_projection_hashes
            ),
            branch_literal_jacobian_projection_receipt_hashes=tuple(
                branch_jacobian_projection_hashes
            ),
            branch_observation_bounds_hashes=tuple(branch_bound_hashes),
            branch_execution_receipt_hashes=tuple(
                (branch, execution.receipt_hash)
                for branch, execution in branch_executions
            ),
            branch_record_hashes=tuple(
                (branch, tuple(record.record_hash for record in records))
                for branch, records in branch_rows
            ),
            execution_mode=policy.execution_mode,
            device_type=device_type,
        )
        result = V1AlexNetFiberedFQRNativeCExecutionResult(
            receipt=execution_receipt,
            branch_records=tuple(branch_rows),
        )
        self._last_fibered_result = result
        return result

    def execute_fibered_fqr_directional_secants(
        self,
        **kwargs: Any,
    ) -> tuple[
        tuple[V1RelaxedKKTBranch, tuple[V1FiberedDirectionalRecord, ...]], ...
    ]:
        """Canonical sibling ABI returning only exact native-fiber records."""

        return self.execute_fibered(**kwargs).branch_records


__all__ = [
    "NATIVE_FQR_AUTHORITY_SCOPE",
    "NATIVE_FQR_ANALYTIC_CACHE_SCHEMA",
    "NATIVE_FQR_EXECUTION_POLICY_SCHEMA",
    "NATIVE_FQR_EXECUTION_RECEIPT_SCHEMA",
    "NATIVE_FQR_EXECUTION_RESULT_SCHEMA",
    "NATIVE_FQR_MEMORY_FORECAST_SCHEMA",
    "NATIVE_FQR_PRE_C_REGISTRATION_SCHEMA",
    "NATIVE_FIBERED_FQR_EXECUTION_RECEIPT_SCHEMA",
    "NATIVE_FIBERED_FQR_EXECUTION_RESULT_SCHEMA",
    "NATIVE_FIBERED_FQR_MODE_BINDING_SCHEMA",
    "V1AlexNetFQRAnalyticBoundsResolver",
    "V1AlexNetFiberedFQRNativeCExecutionReceipt",
    "V1AlexNetFiberedFQRNativeCExecutionResult",
    "V1AlexNetFQRNativeCExecutionReceipt",
    "V1AlexNetFQRNativeCExecutionResult",
    "V1AlexNetFQRNativeCExecutor",
    "V1AlexNetFQRNativeCPreExecutionRegistration",
    "V1AlexNetFQRNativeExecutionMode",
    "V1AlexNetFQRNativeExecutionPolicy",
    "V1AlexNetFQRNativeMemoryForecast",
    "V1AlexNetFQRNativeRefusal",
    "V1AlexNetFQRNativeRefusalCode",
    "alexnet_fqr_native_executor_source_hash",
    "build_v1_alexnet_fqr_native_execution_policy",
    "forecast_v1_alexnet_fqr_native_memory",
    "register_v1_alexnet_fqr_native_c_execution",
]
