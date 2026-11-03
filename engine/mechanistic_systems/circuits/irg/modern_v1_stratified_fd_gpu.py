"""Stratified device FD assembly for modern-v1 raw-logit response geometry.

The production sampling law may stratify train contexts, but the response
metric is still one global raw-logit geometry.  This module keeps those two
facts separate: each frozen stratum owns its own un-reweighted FD stream, then
the final geometry is assembled by the law of total covariance with the
pre-outcome macro weights.

The assembly certificate is intentionally not a source-row proof.  Existing
``DeviceGramFrequentDirectionsTrustAccumulator`` receipts remain the authority
surface for row-level FD derivations; reduced proposal packets and this
stratified combiner cannot be promoted into that proof by naming convention.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.trust_geometry import GlobalTrustGeometrySpec
from mechanistic_systems.core.axis_identity import (
    canonical_unique_string_axis,
    ordered_string_axis_hash,
)
from mechanistic_systems.geometry.trust import GlobalTrustGeometryState
from mechanistic_systems.measurement.device_route_moments import (
    DeviceFrequentDirectionsTrustAccumulatorSnapshot,
    DeviceFrequentDirectionsTrustGeometryDeviceState,
    DeviceGramFrequentDirectionsTrustAccumulator,
    _copy_immutable_fp64_host_array_into_tensor,
)

try:  # pragma: no cover - import behavior is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


STRATIFIED_FD_DESIGN_SCHEMA: Final = "partition_free_v1_stratified_fd_design_v1"
STRATIFIED_FD_ASSEMBLY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_stratified_fd_assembly_receipt_v1"
)
STRATIFIED_FD_RESULT_SCHEMA: Final = "partition_free_v1_stratified_fd_result_v1"
STRATIFIED_FD_DEVICE_ASSEMBLY_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_stratified_fd_device_assembly_receipt_v1"
)
STRATIFIED_FD_ACCUMULATOR_SNAPSHOT_SCHEMA: Final = (
    "partition_free_v1_stratified_fd_accumulator_snapshot_v2"
)
STRATIFIED_FD_MEMORY_FORECAST_SCHEMA: Final = (
    "partition_free_v1_stratified_fd_memory_forecast_v2"
)

STRATIFIED_FD_ROW_ARITHMETIC_ID: Final = (
    "separate_stratum_weighted_raw_logit_fd_rows_no_reweighted_mixture_v1"
)
STRATIFIED_FD_RECOMPRESSION_ARITHMETIC_ID: Final = (
    "law_total_covariance_gram_eigh_fd_recompression_float64_outward_v2"
)
STRATIFIED_FD_SOURCE_PROOF_STATUS: Final = (
    "not_a_source_row_proof_and_not_promotable_from_reduced_packets_v1"
)
STRATIFIED_FD_MEMORY_SCOPE: Final = (
    "registered_lifetime_separated_bulk_tensor_bytes_excluding_scorer_rows_v2"
)

_DESIGN_PREFIX: Final = "partition-free-v1-stratified-fd-design-"
_RECEIPT_PREFIX: Final = "partition-free-v1-stratified-fd-receipt-"
_RESULT_PREFIX: Final = "partition-free-v1-stratified-fd-result-"
_SNAPSHOT_PREFIX: Final = "partition-free-v1-stratified-fd-snapshot-"
_CONTEXT_AXIS_PREFIX: Final = "partition-free-v1-stratified-fd-context-axis-"
_SOURCE_MEASUREMENT_PREFIX: Final = (
    "partition-free-v1-stratified-fd-source-measurements-"
)
_DEVICE_RECEIPT_PREFIX: Final = "partition-free-v1-stratified-fd-device-receipt-"
_FLOAT64_BYTES: Final = 8
_PSD_TOLERANCE_MULTIPLIER: Final = 128.0
_STRATIFIED_FD_DEVICE_STATE_ISSUER: Final = object()


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _stratum_names(values: Sequence[str], *, name: str = "stratum names") -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(_identity(item, name=f"{name} item") for item in values)
    if len(result) < 2 or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain at least two unique names")
    return result


def _weights(values: Sequence[float], *, expected: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("stratum weights must be a numeric sequence")
    result = tuple(float(item) for item in values)
    if len(result) != expected:
        raise ValueError("stratum weights must align with stratum names")
    if any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError("stratum weights must be finite and positive")
    total = math.fsum(result)
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("stratum weights must be frozen and normalized before outcomes")
    return result


def _torch_device(device: Any) -> Any:
    if torch is None:
        raise RuntimeError("stratified FD assembly requires PyTorch")
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda" and runtime_device.index is None:
        try:
            return torch.device("cuda", int(torch.cuda.current_device()))
        except (AssertionError, RuntimeError):
            return runtime_device
    return runtime_device


def _device_identity(device: Any) -> str:
    if torch is None:
        return "torch-unavailable"
    runtime_device = torch.device(device)
    if runtime_device.type != "cuda":
        return str(runtime_device)
    index = runtime_device.index
    if index is None:
        try:
            index = int(torch.cuda.current_device())
        except (AssertionError, RuntimeError):
            index = -1
    try:
        name = torch.cuda.get_device_name(runtime_device)
    except (AssertionError, RuntimeError):
        name = "cuda-device-name-unavailable"
    return f"cuda:{index}:{name}"


def _as_device_matrix(value: np.ndarray, *, device: Any, name: str) -> Any:
    if torch is None:
        raise RuntimeError("stratified FD assembly requires PyTorch")
    array = np.asarray(value)
    if array.dtype != np.dtype("<f8") or array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix")
    result = torch.empty(array.shape, dtype=torch.float64, device=device)
    _copy_immutable_fp64_host_array_into_tensor(result, array, name=name)
    return result


def _as_device_vector(value: np.ndarray, *, device: Any, name: str) -> Any:
    if torch is None:
        raise RuntimeError("stratified FD assembly requires PyTorch")
    array = np.asarray(value)
    if array.dtype != np.dtype("<f8") or array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector")
    result = torch.empty(array.shape, dtype=torch.float64, device=device)
    _copy_immutable_fp64_host_array_into_tensor(result, array, name=name)
    return result


def _finite_tensor(name: str, value: Any) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True, slots=True)
class V1StratifiedFDDesign:
    """Frozen pre-outcome identity for one stratified response geometry."""

    campaign_contract_hash: str
    model_identity_hash: str
    query_contract_hash: str
    preprocessing_hash: str
    current_mask_hash: str
    raw_logit_stream_source_hash: str
    sampling_law_hash: str
    proposal_generation: int
    route_ids: tuple[str, ...]
    stratum_names: tuple[str, ...]
    stratum_weights: tuple[float, ...]
    spec: GlobalTrustGeometrySpec
    measured_mask_hash: str
    calibration_state_id: str
    row_arithmetic_id: str = STRATIFIED_FD_ROW_ARITHMETIC_ID
    recompression_arithmetic_id: str = STRATIFIED_FD_RECOMPRESSION_ARITHMETIC_ID
    source_proof_status: str = STRATIFIED_FD_SOURCE_PROOF_STATUS
    schema: str = STRATIFIED_FD_DESIGN_SCHEMA
    _design_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "campaign_contract_hash",
            "model_identity_hash",
            "query_contract_hash",
            "preprocessing_hash",
            "current_mask_hash",
            "raw_logit_stream_source_hash",
            "sampling_law_hash",
            "measured_mask_hash",
            "calibration_state_id",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        generation = _nonnegative_int(self.proposal_generation, name="proposal generation")
        route_ids = canonical_unique_string_axis(
            self.route_ids,
            name="stratified FD route axis",
        )
        names = _stratum_names(self.stratum_names)
        weights = _weights(self.stratum_weights, expected=len(names))
        if type(self.spec) is not GlobalTrustGeometrySpec:
            raise TypeError("stratified FD design requires an exact trust spec")
        if self.spec.maximum_rank > len(route_ids):
            raise ValueError("trust spec maximum rank cannot exceed the route axis")
        if self.row_arithmetic_id != STRATIFIED_FD_ROW_ARITHMETIC_ID:
            raise ValueError("stratified FD row arithmetic identity is invalid")
        if self.recompression_arithmetic_id != STRATIFIED_FD_RECOMPRESSION_ARITHMETIC_ID:
            raise ValueError("stratified FD recompression arithmetic identity is invalid")
        if self.source_proof_status != STRATIFIED_FD_SOURCE_PROOF_STATUS:
            raise ValueError("stratified FD source-proof status is invalid")
        if self.schema != STRATIFIED_FD_DESIGN_SCHEMA:
            raise ValueError("stratified FD design schema is invalid")
        object.__setattr__(self, "proposal_generation", generation)
        object.__setattr__(self, "route_ids", route_ids)
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "stratum_weights", weights)

    @property
    def route_axis_hash(self) -> str:
        return ordered_string_axis_hash(
            self.route_ids,
            name="stratified FD route axis",
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "campaign_contract_hash": self.campaign_contract_hash,
            "model_identity_hash": self.model_identity_hash,
            "query_contract_hash": self.query_contract_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "current_mask_hash": self.current_mask_hash,
            "raw_logit_stream_source_hash": self.raw_logit_stream_source_hash,
            "sampling_law_hash": self.sampling_law_hash,
            "proposal_generation": self.proposal_generation,
            "route_axis_hash": self.route_axis_hash,
            "route_count": len(self.route_ids),
            "stratum_names": self.stratum_names,
            "stratum_weights": self.stratum_weights,
            "spec": self.spec,
            "measured_mask_hash": self.measured_mask_hash,
            "calibration_state_id": self.calibration_state_id,
            "row_arithmetic_id": self.row_arithmetic_id,
            "recompression_arithmetic_id": self.recompression_arithmetic_id,
            "source_proof_status": self.source_proof_status,
        }

    @property
    def design_hash(self) -> str:
        cached = self._design_hash
        if cached is None:
            cached = content_hash(self._hash_payload(), prefix=_DESIGN_PREFIX)
            object.__setattr__(self, "_design_hash", cached)
        return cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_DESIGN_PREFIX)
        if self._design_hash is not None and observed != self._design_hash:
            raise ValueError("stratified FD design was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("stratified FD design hash does not match")
        object.__setattr__(self, "_design_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1StratifiedFDMemoryForecast:
    """Allocation-free phase ledger for streamed FD restore and assembly."""

    route_count: int
    stratum_count: int
    response_rows: int
    per_stratum_sketch_rank: int
    combined_maximum_rank: int
    live_stratum_fd_device_bytes: int
    combined_pre_recompression_factor_device_bytes: int
    combined_output_state_device_bytes: int
    gram_workspace_device_bytes: int
    restore_peak_device_bytes: int
    stream_compression_workspace_device_bytes: int
    stream_compression_peak_device_bytes: int
    single_child_finalize_peak_device_bytes: int
    combined_assembly_peak_device_bytes: int
    peak_device_bytes: int
    aggregate_host_transfer_bytes: int
    host_context_route_rows_transferred: bool = False
    source_context_route_rows_retained_bytes: int = 0
    memory_accounting_scope: str = STRATIFIED_FD_MEMORY_SCOPE
    schema: str = STRATIFIED_FD_MEMORY_FORECAST_SCHEMA

    def __post_init__(self) -> None:
        routes = _positive_int(self.route_count, name="route count")
        strata = _positive_int(self.stratum_count, name="stratum count")
        response_rows = _positive_int(self.response_rows, name="response rows")
        sketch_rank = _positive_int(
            self.per_stratum_sketch_rank,
            name="per-stratum sketch rank",
        )
        maximum_rank = _positive_int(
            self.combined_maximum_rank,
            name="combined maximum rank",
        )
        input_factor_rows = strata * (sketch_rank + response_rows)
        expected_live = strata * (2 * sketch_rank + response_rows + 1) * routes * _FLOAT64_BYTES
        expected_pre = input_factor_rows * routes * _FLOAT64_BYTES
        expected_output = (response_rows + maximum_rank + 1) * routes * _FLOAT64_BYTES
        expected_gram = input_factor_rows * input_factor_rows * _FLOAT64_BYTES
        stream_gram = (2 * sketch_rank) ** 2 * _FLOAT64_BYTES
        expected_restore = expected_live
        expected_stream_workspace = (
            2 * sketch_rank * routes * _FLOAT64_BYTES + stream_gram
        )
        expected_stream_peak = expected_live + expected_stream_workspace
        one_child = (
            2 * sketch_rank + response_rows + 1
        ) * routes * _FLOAT64_BYTES
        # After clone-free, one-child-at-a-time finalization, the two largest
        # route surfaces are either the registered scaled Gram input or the
        # normalized factor plus its exact coordinate-square reduction. The
        # response/vector allowance covers the eager fp64 diagonal expression.
        expected_finalize = one_child + (
            2 * sketch_rank + response_rows + 4
        ) * routes * _FLOAT64_BYTES + stream_gram
        # Assembly preallocates the complete input factor exactly once. Its
        # largest co-live surface is the registered `pre_factor * pre_factor`
        # (or identically sized scaled Gram input), plus means and vector state.
        expected_assembly = (
            2 * input_factor_rows
            + strata * response_rows
            + response_rows
            + 3
        ) * routes * _FLOAT64_BYTES + expected_gram
        expected_peak = max(
            expected_restore,
            expected_stream_peak,
            expected_finalize,
            expected_assembly,
        )
        expected_host = expected_output
        expected = {
            "live_stratum_fd_device_bytes": expected_live,
            "combined_pre_recompression_factor_device_bytes": expected_pre,
            "combined_output_state_device_bytes": expected_output,
            "gram_workspace_device_bytes": expected_gram,
            "restore_peak_device_bytes": expected_restore,
            "stream_compression_workspace_device_bytes": expected_stream_workspace,
            "stream_compression_peak_device_bytes": expected_stream_peak,
            "single_child_finalize_peak_device_bytes": expected_finalize,
            "combined_assembly_peak_device_bytes": expected_assembly,
            "peak_device_bytes": expected_peak,
            "aggregate_host_transfer_bytes": expected_host,
            "source_context_route_rows_retained_bytes": 0,
        }
        for name, expected_value in expected.items():
            observed = _nonnegative_int(getattr(self, name), name=name)
            if observed != expected_value:
                raise ValueError(f"{name} does not replay from forecast inputs")
            object.__setattr__(self, name, observed)
        if self.host_context_route_rows_transferred is not False:
            raise ValueError("stratified FD assembly cannot transfer context-route rows")
        if self.memory_accounting_scope != STRATIFIED_FD_MEMORY_SCOPE:
            raise ValueError("stratified FD memory scope is invalid")
        if self.schema != STRATIFIED_FD_MEMORY_FORECAST_SCHEMA:
            raise ValueError("stratified FD memory forecast schema is invalid")
        object.__setattr__(self, "route_count", routes)
        object.__setattr__(self, "stratum_count", strata)
        object.__setattr__(self, "response_rows", response_rows)
        object.__setattr__(self, "per_stratum_sketch_rank", sketch_rank)
        object.__setattr__(self, "combined_maximum_rank", maximum_rank)


def forecast_stratified_fd_gpu_memory(
    *,
    route_count: int,
    stratum_count: int,
    response_rows: int,
    per_stratum_sketch_rank: int,
    combined_maximum_rank: int,
) -> V1StratifiedFDMemoryForecast:
    """Return the registered byte ledger without allocating route tensors."""

    routes = _positive_int(route_count, name="route count")
    strata = _positive_int(stratum_count, name="stratum count")
    responses = _positive_int(response_rows, name="response rows")
    sketch = _positive_int(per_stratum_sketch_rank, name="per-stratum sketch rank")
    maximum = _positive_int(combined_maximum_rank, name="combined maximum rank")
    input_factor_rows = strata * (sketch + responses)
    live = strata * (2 * sketch + responses + 1) * routes * _FLOAT64_BYTES
    pre = input_factor_rows * routes * _FLOAT64_BYTES
    output = (responses + maximum + 1) * routes * _FLOAT64_BYTES
    gram = input_factor_rows * input_factor_rows * _FLOAT64_BYTES
    restore_peak = live
    stream_gram = (2 * sketch) ** 2 * _FLOAT64_BYTES
    stream_workspace = 2 * sketch * routes * _FLOAT64_BYTES + stream_gram
    stream_peak = live + stream_workspace
    one_child = (2 * sketch + responses + 1) * routes * _FLOAT64_BYTES
    finalize_peak = one_child + (
        2 * sketch + responses + 4
    ) * routes * _FLOAT64_BYTES + stream_gram
    assembly_peak = (
        2 * input_factor_rows
        + strata * responses
        + responses
        + 3
    ) * routes * _FLOAT64_BYTES + gram
    peak = max(restore_peak, stream_peak, finalize_peak, assembly_peak)
    return V1StratifiedFDMemoryForecast(
        route_count=routes,
        stratum_count=strata,
        response_rows=responses,
        per_stratum_sketch_rank=sketch,
        combined_maximum_rank=maximum,
        live_stratum_fd_device_bytes=live,
        combined_pre_recompression_factor_device_bytes=pre,
        combined_output_state_device_bytes=output,
        gram_workspace_device_bytes=gram,
        restore_peak_device_bytes=restore_peak,
        stream_compression_workspace_device_bytes=stream_workspace,
        stream_compression_peak_device_bytes=stream_peak,
        single_child_finalize_peak_device_bytes=finalize_peak,
        combined_assembly_peak_device_bytes=assembly_peak,
        peak_device_bytes=peak,
        aggregate_host_transfer_bytes=output,
    )


@dataclass(frozen=True, slots=True)
class V1StratifiedFDAssemblyReceipt:
    """Numerical certificate for law-of-total-covariance assembly."""

    design_hash: str
    combined_geometry_hash: str
    stratum_geometry_hashes: tuple[tuple[str, str], ...]
    route_axis_hash: str
    device_identity: str
    response_rows: int
    route_count: int
    input_factor_rows: int
    retained_rank: int
    maximum_rank: int
    source_residual_spectral_bound: float
    recompression_residual_spectral_bound: float
    combined_residual_spectral_bound: float
    mean_component_hash: str
    centered_factor_hash: str
    residual_diagonal_hash: str
    total_weight: float
    context_count: int
    peak_device_bytes: int
    aggregate_host_transfer_bytes: int
    host_context_route_rows_transferred: bool = False
    source_proof_status: str = STRATIFIED_FD_SOURCE_PROOF_STATUS
    row_arithmetic_id: str = STRATIFIED_FD_ROW_ARITHMETIC_ID
    recompression_arithmetic_id: str = STRATIFIED_FD_RECOMPRESSION_ARITHMETIC_ID
    schema: str = STRATIFIED_FD_ASSEMBLY_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "combined_geometry_hash",
            "route_axis_hash",
            "device_identity",
            "mean_component_hash",
            "centered_factor_hash",
            "residual_diagonal_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        pairs = tuple(
            (_identity(name, name="receipt stratum name"), _identity(value, name="stratum geometry hash"))
            for name, value in self.stratum_geometry_hashes
        )
        if not pairs or len({name for name, _value in pairs}) != len(pairs):
            raise ValueError("receipt stratum geometry hashes must be named and unique")
        for name in (
            "response_rows",
            "route_count",
            "input_factor_rows",
            "retained_rank",
            "maximum_rank",
            "context_count",
            "peak_device_bytes",
            "aggregate_host_transfer_bytes",
        ):
            value = (
                _nonnegative_int(getattr(self, name), name=name)
                if name == "retained_rank"
                else _positive_int(getattr(self, name), name=name)
            )
            object.__setattr__(self, name, value)
        if self.retained_rank > self.maximum_rank:
            raise ValueError("receipt retained rank exceeds maximum rank")
        source = _finite_nonnegative(
            self.source_residual_spectral_bound,
            name="source residual spectral bound",
        )
        extra = _finite_nonnegative(
            self.recompression_residual_spectral_bound,
            name="recompression residual spectral bound",
        )
        combined = _finite_nonnegative(
            self.combined_residual_spectral_bound,
            name="combined residual spectral bound",
        )
        if not math.isclose(combined, source + extra, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("combined residual spectral bound does not replay")
        total_weight = float(self.total_weight)
        if not math.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("receipt total weight must be finite and positive")
        if self.host_context_route_rows_transferred is not False:
            raise ValueError("stratified FD assembly cannot transfer context-route rows")
        if self.source_proof_status != STRATIFIED_FD_SOURCE_PROOF_STATUS:
            raise ValueError("receipt source-proof status is invalid")
        if self.row_arithmetic_id != STRATIFIED_FD_ROW_ARITHMETIC_ID:
            raise ValueError("receipt row arithmetic identity is invalid")
        if self.recompression_arithmetic_id != STRATIFIED_FD_RECOMPRESSION_ARITHMETIC_ID:
            raise ValueError("receipt recompression arithmetic identity is invalid")
        if self.schema != STRATIFIED_FD_ASSEMBLY_RECEIPT_SCHEMA:
            raise ValueError("stratified FD receipt schema is invalid")
        object.__setattr__(self, "stratum_geometry_hashes", pairs)
        object.__setattr__(self, "source_residual_spectral_bound", source)
        object.__setattr__(self, "recompression_residual_spectral_bound", extra)
        object.__setattr__(self, "combined_residual_spectral_bound", combined)
        object.__setattr__(self, "total_weight", total_weight)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "combined_geometry_hash": self.combined_geometry_hash,
            "stratum_geometry_hashes": self.stratum_geometry_hashes,
            "route_axis_hash": self.route_axis_hash,
            "device_identity": self.device_identity,
            "response_rows": self.response_rows,
            "route_count": self.route_count,
            "input_factor_rows": self.input_factor_rows,
            "retained_rank": self.retained_rank,
            "maximum_rank": self.maximum_rank,
            "source_residual_spectral_bound": self.source_residual_spectral_bound,
            "recompression_residual_spectral_bound": (
                self.recompression_residual_spectral_bound
            ),
            "combined_residual_spectral_bound": self.combined_residual_spectral_bound,
            "mean_component_hash": self.mean_component_hash,
            "centered_factor_hash": self.centered_factor_hash,
            "residual_diagonal_hash": self.residual_diagonal_hash,
            "total_weight": self.total_weight,
            "context_count": self.context_count,
            "peak_device_bytes": self.peak_device_bytes,
            "aggregate_host_transfer_bytes": self.aggregate_host_transfer_bytes,
            "host_context_route_rows_transferred": (
                self.host_context_route_rows_transferred
            ),
            "source_proof_status": self.source_proof_status,
            "row_arithmetic_id": self.row_arithmetic_id,
            "recompression_arithmetic_id": self.recompression_arithmetic_id,
        }

    @property
    def receipt_hash(self) -> str:
        cached = self._receipt_hash
        if cached is None:
            cached = content_hash(self._hash_payload(), prefix=_RECEIPT_PREFIX)
            object.__setattr__(self, "_receipt_hash", cached)
        return cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("stratified FD receipt was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("stratified FD receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1StratifiedFDDeviceAssemblyReceipt:
    """Compact lineage for a fully device-resident covariance assembly.

    The receipt binds scalar identities, source axes, tensor roles, and
    numerical envelopes.  It intentionally does not hash or materialize any
    route-scale tensor; live storage/version custody belongs to the issuing
    device handle at the next joint.
    """

    design_hash: str
    route_axis_hash: str
    device_identity: str
    stratum_names: tuple[str, ...]
    stratum_weights: tuple[float, ...]
    stratum_context_counts: tuple[int, ...]
    stratum_context_axis_hashes: tuple[str, ...]
    stratum_source_measurement_axis_hashes: tuple[str, ...]
    response_rows: int
    route_count: int
    input_factor_rows: int
    retained_rank: int
    maximum_rank: int
    source_residual_spectral_bound: float
    recompression_residual_spectral_bound: float
    combined_residual_spectral_bound: float
    total_weight: float
    context_count: int
    mean_component_shape: tuple[int, int]
    centered_factor_shape: tuple[int, int]
    residual_diagonal_shape: tuple[int, ...]
    dtype_identity: str = "torch.float64"
    route_scale_d2h_bytes: int = 0
    route_scale_content_digest_bytes: int = 0
    host_context_route_rows_transferred: bool = False
    source_proof_status: str = STRATIFIED_FD_SOURCE_PROOF_STATUS
    row_arithmetic_id: str = STRATIFIED_FD_ROW_ARITHMETIC_ID
    recompression_arithmetic_id: str = STRATIFIED_FD_RECOMPRESSION_ARITHMETIC_ID
    downstream_decision_authority: bool = False
    schema: str = STRATIFIED_FD_DEVICE_ASSEMBLY_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in ("design_hash", "route_axis_hash", "device_identity"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        names = _stratum_names(self.stratum_names)
        weights = _weights(self.stratum_weights, expected=len(names))
        counts = tuple(
            _positive_int(value, name="device FD stratum context count")
            for value in self.stratum_context_counts
        )
        context_axes = tuple(
            _identity(value, name="device FD stratum context-axis hash")
            for value in self.stratum_context_axis_hashes
        )
        measurement_axes = tuple(
            _identity(value, name="device FD source-measurement-axis hash")
            for value in self.stratum_source_measurement_axis_hashes
        )
        if not (len(counts) == len(context_axes) == len(measurement_axes) == len(names)):
            raise ValueError("device FD receipt stratum ledgers must align")
        response_rows = _positive_int(self.response_rows, name="response rows")
        route_count = _positive_int(self.route_count, name="route count")
        input_rows = _positive_int(self.input_factor_rows, name="input factor rows")
        retained = _nonnegative_int(self.retained_rank, name="retained rank")
        maximum = _positive_int(self.maximum_rank, name="maximum rank")
        if retained > maximum:
            raise ValueError("device FD retained rank exceeds its maximum")
        source = _finite_nonnegative(
            self.source_residual_spectral_bound,
            name="source residual spectral bound",
        )
        extra = _finite_nonnegative(
            self.recompression_residual_spectral_bound,
            name="recompression residual spectral bound",
        )
        combined = _finite_nonnegative(
            self.combined_residual_spectral_bound,
            name="combined residual spectral bound",
        )
        if not math.isclose(combined, source + extra, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("device FD combined residual does not replay")
        total_weight = float(self.total_weight)
        context_count = _positive_int(self.context_count, name="context count")
        if (
            not math.isfinite(total_weight)
            or total_weight <= 0.0
            or context_count != sum(counts)
            or tuple(self.mean_component_shape) != (response_rows, route_count)
            or tuple(self.centered_factor_shape) != (retained, route_count)
            or tuple(self.residual_diagonal_shape) != (route_count,)
            or self.dtype_identity != "torch.float64"
            or self.route_scale_d2h_bytes != 0
            or self.route_scale_content_digest_bytes != 0
            or self.host_context_route_rows_transferred is not False
            or self.source_proof_status != STRATIFIED_FD_SOURCE_PROOF_STATUS
            or self.row_arithmetic_id != STRATIFIED_FD_ROW_ARITHMETIC_ID
            or self.recompression_arithmetic_id
            != STRATIFIED_FD_RECOMPRESSION_ARITHMETIC_ID
            or self.downstream_decision_authority is not False
            or self.schema != STRATIFIED_FD_DEVICE_ASSEMBLY_RECEIPT_SCHEMA
        ):
            raise ValueError("device FD receipt crossed its numerical or authority boundary")
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "stratum_context_counts", counts)
        object.__setattr__(self, "stratum_context_axis_hashes", context_axes)
        object.__setattr__(self, "stratum_source_measurement_axis_hashes", measurement_axes)
        object.__setattr__(self, "response_rows", response_rows)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "input_factor_rows", input_rows)
        object.__setattr__(self, "retained_rank", retained)
        object.__setattr__(self, "maximum_rank", maximum)
        object.__setattr__(self, "source_residual_spectral_bound", source)
        object.__setattr__(self, "recompression_residual_spectral_bound", extra)
        object.__setattr__(self, "combined_residual_spectral_bound", combined)
        object.__setattr__(self, "total_weight", total_weight)
        object.__setattr__(self, "context_count", context_count)

    def _hash_payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.startswith("_")
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_DEVICE_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("device FD assembly receipt was mutated")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("device FD assembly receipt hash differs")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()


class V1StratifiedFDDeviceGeometryState:
    """One-shot transient owner of combined device FD tensors."""

    __slots__ = (
        "receipt",
        "_mean_component",
        "_centered_factor",
        "_residual_diagonal",
        "_consumed",
    )

    def __init__(
        self,
        *,
        receipt: V1StratifiedFDDeviceAssemblyReceipt,
        mean_component: Any,
        centered_factor: Any,
        residual_diagonal: Any,
        _issuance_token: object,
    ) -> None:
        if _issuance_token is not _STRATIFIED_FD_DEVICE_STATE_ISSUER:
            raise PermissionError("stratified device FD states are source-issued only")
        if type(receipt) is not V1StratifiedFDDeviceAssemblyReceipt:
            raise TypeError("stratified device FD state requires its exact receipt")
        receipt.assert_integrity()
        tensors = (mean_component, centered_factor, residual_diagonal)
        if torch is None or any(not torch.is_tensor(value) for value in tensors):
            raise TypeError("stratified device FD state requires torch tensors")
        if (
            any(value.dtype != torch.float64 for value in tensors)
            or any(value.requires_grad for value in tensors)
            or any(not value.is_contiguous() for value in tensors)
            or len({str(value.device) for value in tensors}) != 1
            or str(mean_component.device) != receipt.device_identity
            or tuple(mean_component.shape) != receipt.mean_component_shape
            or tuple(centered_factor.shape) != receipt.centered_factor_shape
            or tuple(residual_diagonal.shape) != receipt.residual_diagonal_shape
        ):
            raise ValueError("stratified device FD tensors differ from their receipt")
        _finite_tensor("combined device FD mean", mean_component)
        _finite_tensor("combined device FD factor", centered_factor)
        _finite_tensor("combined device FD residual diagonal", residual_diagonal)
        if bool(torch.any(residual_diagonal < 0.0).item()):
            raise ValueError("combined device FD residual diagonal is negative")
        self.receipt = receipt
        self._mean_component = mean_component
        self._centered_factor = centered_factor
        self._residual_diagonal = residual_diagonal
        self._consumed = False

    def take_tensors_once(self) -> tuple[Any, Any, Any]:
        if self._consumed:
            raise RuntimeError("combined device FD tensors were already consumed")
        self.receipt.assert_integrity()
        self._consumed = True
        result = (
            self._mean_component,
            self._centered_factor,
            self._residual_diagonal,
        )
        self._mean_component = None
        self._centered_factor = None
        self._residual_diagonal = None
        return result

    def __reduce__(self) -> NoReturn:
        raise TypeError("combined device FD states are nonserializable")

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("combined device FD states are nonserializable")


@dataclass(frozen=True, slots=True)
class V1StratifiedFDGeometryResult:
    """Closed result for a stratified raw-logit trust geometry."""

    design: V1StratifiedFDDesign
    geometry: GlobalTrustGeometryState
    receipt: V1StratifiedFDAssemblyReceipt
    memory_forecast: V1StratifiedFDMemoryForecast
    schema: str = STRATIFIED_FD_RESULT_SCHEMA
    _result_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.design) is not V1StratifiedFDDesign:
            raise TypeError("stratified FD result requires its exact design type")
        self.design.assert_integrity()
        if type(self.geometry) is not GlobalTrustGeometryState:
            raise TypeError("stratified FD result requires a trust geometry state")
        if type(self.receipt) is not V1StratifiedFDAssemblyReceipt:
            raise TypeError("stratified FD result requires its exact receipt type")
        self.receipt.assert_integrity()
        if type(self.memory_forecast) is not V1StratifiedFDMemoryForecast:
            raise TypeError("stratified FD result requires its memory forecast")
        if self.receipt.design_hash != self.design.design_hash:
            raise ValueError("stratified FD receipt does not bind its design")
        if self.receipt.combined_geometry_hash != self.geometry.content_hash:
            raise ValueError("stratified FD receipt does not bind its geometry")
        if self.geometry.route_ids != self.design.route_ids:
            raise ValueError("stratified FD result route axis differs from design")
        if self.geometry.spec != self.design.spec:
            raise ValueError("stratified FD result spec differs from design")
        if self.schema != STRATIFIED_FD_RESULT_SCHEMA:
            raise ValueError("stratified FD result schema is invalid")

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design.design_hash,
            "geometry_hash": self.geometry.content_hash,
            "receipt_hash": self.receipt.receipt_hash,
            "memory_forecast": self.memory_forecast,
        }

    @property
    def result_hash(self) -> str:
        cached = self._result_hash
        if cached is None:
            cached = content_hash(self._hash_payload(), prefix=_RESULT_PREFIX)
            object.__setattr__(self, "_result_hash", cached)
        return cached


@dataclass(frozen=True, slots=True)
class V1StratifiedFDAccumulatorSnapshot:
    """Packet-boundary exact continuation state for separate stratum FD streams."""

    design_hash: str
    design_stratum_names: tuple[str, ...]
    route_axis_hash: str
    sketch_rank: int
    packet_boundary_index: int
    stratum_snapshots: tuple[tuple[str, DeviceFrequentDirectionsTrustAccumulatorSnapshot], ...]
    source_measurement_hashes_by_stratum: tuple[tuple[str, tuple[str, ...]], ...]
    schema: str = STRATIFIED_FD_ACCUMULATOR_SNAPSHOT_SCHEMA
    _snapshot_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "design_hash",
            _identity(self.design_hash, name="snapshot design hash"),
        )
        names = _stratum_names(self.design_stratum_names, name="snapshot stratum names")
        route_axis_hash = _identity(self.route_axis_hash, name="snapshot route axis hash")
        sketch_rank = _positive_int(self.sketch_rank, name="snapshot sketch rank")
        boundary = _nonnegative_int(
            self.packet_boundary_index,
            name="packet boundary index",
        )
        pairs = tuple(self.stratum_snapshots)
        if tuple(name for name, _snapshot in pairs) != names:
            raise ValueError("stratified FD snapshot is partial or out of order")
        source_pairs = tuple(
            (
                _identity(name, name="snapshot source-measurement stratum name"),
                tuple(
                    _identity(value, name="snapshot source-measurement hash")
                    for value in values
                ),
            )
            for name, values in self.source_measurement_hashes_by_stratum
        )
        if tuple(name for name, _values in source_pairs) != names:
            raise ValueError(
                "stratified FD source-measurement snapshot is partial or out of order"
            )
        all_context_ids: list[str] = []
        for name, snapshot in pairs:
            _identity(name, name="snapshot stratum name")
            if type(snapshot) is not DeviceFrequentDirectionsTrustAccumulatorSnapshot:
                raise TypeError("stratified FD snapshot requires exact device snapshots")
            snapshot.validate_integrity()
            if snapshot.ordered_route_axis_hash != route_axis_hash:
                raise ValueError("stratified FD snapshot route axis differs by stratum")
            if snapshot.sketch_rank != sketch_rank:
                raise ValueError("stratified FD snapshot sketch rank differs by stratum")
            all_context_ids.extend(snapshot.context_ids)
        if len(set(all_context_ids)) != len(all_context_ids):
            raise ValueError("stratified FD snapshot repeats a context across strata")
        source_by_name = dict(source_pairs)
        for name, snapshot in pairs:
            if len(source_by_name[name]) != snapshot.context_count:
                raise ValueError(
                    "stratified FD source-measurement ledger differs from its stratum contexts"
                )
        context_count = len(all_context_ids)
        if boundary > context_count or (boundary == 0) != (context_count == 0):
            raise ValueError(
                "stratified FD packet-boundary count is inconsistent with observed contexts"
            )
        if self.schema != STRATIFIED_FD_ACCUMULATOR_SNAPSHOT_SCHEMA:
            raise ValueError("stratified FD snapshot schema is invalid")
        object.__setattr__(self, "design_stratum_names", names)
        object.__setattr__(self, "route_axis_hash", route_axis_hash)
        object.__setattr__(self, "sketch_rank", sketch_rank)
        object.__setattr__(self, "packet_boundary_index", boundary)
        object.__setattr__(self, "stratum_snapshots", pairs)
        object.__setattr__(self, "source_measurement_hashes_by_stratum", source_pairs)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "design_stratum_names": self.design_stratum_names,
            "route_axis_hash": self.route_axis_hash,
            "sketch_rank": self.sketch_rank,
            "packet_boundary_index": self.packet_boundary_index,
            "stratum_snapshot_hashes": tuple(
                (name, snapshot.snapshot_hash)
                for name, snapshot in self.stratum_snapshots
            ),
            "source_measurement_hashes_by_stratum": (
                self.source_measurement_hashes_by_stratum
            ),
        }

    def _compute_snapshot_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_SNAPSHOT_PREFIX)

    @property
    def snapshot_hash(self) -> str:
        cached = self._snapshot_hash
        if cached is None:
            cached = self._compute_snapshot_hash()
            object.__setattr__(self, "_snapshot_hash", cached)
        return cached

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for _name, stratum_snapshot in self.stratum_snapshots:
            stratum_snapshot.validate_integrity()
        observed = self._compute_snapshot_hash()
        if self._snapshot_hash is not None and observed != self._snapshot_hash:
            raise ValueError("stratified FD snapshot was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("stratified FD snapshot hash does not match")
        object.__setattr__(self, "_snapshot_hash", observed)
        return observed

    def __reduce__(self) -> tuple[Any, tuple[dict[str, object], str]]:
        payload = {
            "design_hash": self.design_hash,
            "design_stratum_names": self.design_stratum_names,
            "route_axis_hash": self.route_axis_hash,
            "sketch_rank": self.sketch_rank,
            "packet_boundary_index": self.packet_boundary_index,
            "stratum_snapshots": self.stratum_snapshots,
            "source_measurement_hashes_by_stratum": (
                self.source_measurement_hashes_by_stratum
            ),
            "schema": self.schema,
        }
        return _rebuild_stratified_fd_snapshot, (
            payload,
            self.assert_integrity(),
        )


def _rebuild_stratified_fd_snapshot(
    payload: dict[str, object],
    expected_hash: str,
) -> V1StratifiedFDAccumulatorSnapshot:
    snapshot = V1StratifiedFDAccumulatorSnapshot(**payload)  # type: ignore[arg-type]
    snapshot.assert_integrity(expected_hash)
    return snapshot


def _ordered_geometries(
    geometries_by_stratum: Mapping[str, GlobalTrustGeometryState],
    design: V1StratifiedFDDesign,
) -> tuple[tuple[str, GlobalTrustGeometryState], ...]:
    design.assert_integrity()
    provided = {str(key): value for key, value in geometries_by_stratum.items()}
    if set(provided) != set(design.stratum_names):
        raise ValueError("stratified FD assembly requires every frozen stratum exactly once")
    ordered = tuple((name, provided[name]) for name in design.stratum_names)
    reference = ordered[0][1]
    for _name, geometry in ordered:
        if type(geometry) is not GlobalTrustGeometryState:
            raise TypeError("stratified FD assembly requires trust geometry states")
        if (
            geometry.route_ids != design.route_ids
            or geometry.spec != design.spec
            or geometry.measured_mask_hash != design.measured_mask_hash
            or geometry.calibration_state_id != design.calibration_state_id
            or geometry.residual_diagonal_bound is None
        ):
            raise ValueError(
                "stratified FD geometries must share route/spec/mask/calibration "
                "and certified residual diagonals"
            )
        if geometry.mean_component.shape != reference.mean_component.shape:
            raise ValueError("stratified FD geometries have different response axes")
    return ordered


def _compress_factor_rows_with_gram(
    rows: Any,
    *,
    maximum_rank: int,
) -> tuple[Any, float]:
    """Deterministically FD-compress a small row set on the registered device."""

    row_count = int(rows.shape[0])
    if row_count <= maximum_rank:
        return rows, 0.0
    _finite_tensor("stratified FD recompression rows", rows)
    scale = float(torch.max(torch.abs(rows)).item())
    if scale == 0.0:
        return rows[:0], 0.0
    scaled = rows / scale
    gram = scaled @ scaled.T
    gram = 0.5 * (gram + gram.T)
    # Preserve the registered arithmetic and release only the completed Gram
    # input workspace before allocating the retained projection.
    del scaled
    eigenvalues, left = torch.linalg.eigh(gram)
    _finite_tensor("stratified FD recompression eigenvalues", eigenvalues)
    _finite_tensor("stratified FD recompression eigenvectors", left)
    max_eigenvalue = (
        float(torch.max(torch.abs(eigenvalues)).item()) if eigenvalues.numel() else 0.0
    )
    tolerance = (
        _PSD_TOLERANCE_MULTIPLIER
        * float(torch.finfo(eigenvalues.dtype).eps)
        * max(1, row_count, int(rows.shape[1]))
        * max(1.0, max_eigenvalue)
    )
    if bool(torch.any(eigenvalues < -tolerance).item()):
        raise ValueError("stratified FD recompression Gram matrix is not PSD")
    eigenvalues = torch.clamp(torch.flip(eigenvalues, dims=(0,)), min=0.0)
    left = torch.flip(left, dims=(1,))
    pivots = torch.argmax(torch.abs(left), dim=0)
    columns = torch.arange(left.shape[1], device=left.device)
    signs = torch.sign(left[pivots, columns])
    signs = torch.where(signs == 0.0, torch.ones_like(signs), signs)
    left = left * signs
    delta_scaled = (
        float(eigenvalues[maximum_rank].item())
        if int(eigenvalues.numel()) > maximum_rank
        else 0.0
    )
    retain = min(maximum_rank, int(eigenvalues.numel()))
    singular = torch.sqrt(eigenvalues[:retain]) * scale
    shrunk = (
        torch.sqrt(torch.clamp(eigenvalues[:retain] - delta_scaled, min=0.0))
        * scale
    )
    projected = left[:, :retain].T @ rows
    gains = torch.where(
        singular > 0.0,
        shrunk / singular,
        torch.zeros_like(shrunk),
    )
    projected.mul_(gains[:, None])
    retained_count = int(torch.count_nonzero(shrunk > 0.0).item())
    retained = projected[:retained_count]
    _finite_tensor("stratified FD recompressed rows", retained)
    # The FD shrinkage identity is exact over real arithmetic.  The receipt,
    # however, is an outward fp64 certificate: inflate the discarded eigenvalue
    # by the same Gram PSD tolerance used to distinguish roundoff from a real
    # negative eigenvalue, so a slightly underestimated pivot cannot understate
    # the residual spectral envelope.
    delta = (delta_scaled + tolerance) * scale * scale
    if not math.isfinite(delta) or delta < 0.0:
        raise ValueError("stratified FD recompression residual is invalid")
    return retained, float(delta)


def _recompression_residual_diagonal_bound(
    pre_diagonal: Any,
    post_diagonal: Any,
    *,
    input_factor_rows: int,
    retained_rank: int,
    spectral_residual_bound: float,
) -> Any:
    """Return an fp64-outward coordinate residual bound for recompression.

    ``pre_diagonal - post_diagonal`` is the real-arithmetic coordinate loss
    from the FD recompression.  Computing both diagonals through independent
    fp64 sums and then clamping negative entries is not outward-conservative:
    a tiny post-diagonal overshoot would erase positive omitted energy for
    sparse one-route actions.  The returned vector therefore adds a local
    subtraction/summation tolerance before the nonnegative clamp.
    """

    rows = _positive_int(input_factor_rows, name="input factor rows")
    retained = _nonnegative_int(retained_rank, name="retained rank")
    spectral = _finite_nonnegative(
        spectral_residual_bound,
        name="recompression residual spectral bound",
    )
    if pre_diagonal.shape != post_diagonal.shape:
        raise ValueError("recompression diagonals must share the route axis")
    _finite_tensor("stratified FD pre-recompression diagonal", pre_diagonal)
    _finite_tensor("stratified FD post-recompression diagonal", post_diagonal)
    dtype = pre_diagonal.dtype
    eps = float(torch.finfo(dtype).eps)
    work = max(1, rows, retained)
    local_scale = torch.abs(pre_diagonal) + torch.abs(post_diagonal) + spectral
    tolerance = (
        4.0
        * _PSD_TOLERANCE_MULTIPLIER
        * eps
        * float(work)
        * torch.clamp(local_scale, min=torch.finfo(dtype).tiny)
    )
    residual = torch.clamp(pre_diagonal - post_diagonal + tolerance, min=0.0)
    _finite_tensor("stratified FD recompression residual diagonal", residual)
    return residual


def _assemble_stratified_device_fd_consuming(
    *,
    accumulators_by_stratum: dict[str, DeviceGramFrequentDirectionsTrustAccumulator],
    source_measurements_by_stratum: Mapping[str, Sequence[str]],
    design: V1StratifiedFDDesign,
    sketch_rank: int,
    device: Any,
) -> V1StratifiedFDDeviceGeometryState:
    """Consume live child sketches into one device-only total-covariance state."""

    if torch is None:
        raise RuntimeError("stratified device FD assembly requires PyTorch")
    design.assert_integrity()
    runtime_device = _torch_device(device)
    names = design.stratum_names
    if set(accumulators_by_stratum) != set(names):
        raise ValueError("device FD assembly requires every frozen stratum exactly once")
    source_measurements = {
        str(name): tuple(map(str, values))
        for name, values in source_measurements_by_stratum.items()
    }
    if set(source_measurements) != set(names):
        raise ValueError("device FD assembly source ledgers are incomplete")
    if any(not source_measurements[name] for name in names):
        raise ValueError("cannot assemble device FD geometry with an empty stratum")

    route_count = len(design.route_ids)
    maximum_rank = min(design.spec.maximum_rank, route_count)
    weights = design.stratum_weights
    mean_stack: Any | None = None
    pre_factor: Any | None = None
    source_residual_diagonal: Any | None = None
    response_rows: int | None = None
    factor_cursor = 0
    source_residual = 0.0
    total_weight = 0.0
    context_count = 0
    stratum_counts: list[int] = []
    context_axis_hashes: list[str] = []
    measurement_axis_hashes: list[str] = []

    for index, name in enumerate(names):
        accumulator = accumulators_by_stratum.pop(name)
        if (
            type(accumulator) is not DeviceGramFrequentDirectionsTrustAccumulator
            or accumulator.sketch_rank != sketch_rank
            or accumulator.route_ids != design.route_ids
            or accumulator.device != runtime_device
        ):
            raise ValueError("device FD child differs from the registered design")
        child: DeviceFrequentDirectionsTrustGeometryDeviceState = (
            accumulator.finalize_device_consuming()
        )
        del accumulator
        if (
            child.route_axis_hash != design.route_axis_hash
            or child.route_count != route_count
            or child.sketch_rank != sketch_rank
            or child.device_identity != str(runtime_device)
            or len(source_measurements[name]) != child.context_count
        ):
            raise ValueError("consumed device FD child has stale source identity")
        mean_component, centered_factor, residual_diagonal = child.take_tensors_once()
        child_response_rows = int(mean_component.shape[0])
        if response_rows is None:
            response_rows = child_response_rows
            input_capacity = len(names) * (sketch_rank + response_rows)
            mean_stack = torch.empty(
                (len(names), response_rows, route_count),
                dtype=torch.float64,
                device=runtime_device,
            )
            pre_factor = torch.empty(
                (input_capacity, route_count),
                dtype=torch.float64,
                device=runtime_device,
            )
            source_residual_diagonal = torch.zeros(
                route_count,
                dtype=torch.float64,
                device=runtime_device,
            )
        elif child_response_rows != response_rows:
            raise ValueError("device FD children have different response axes")
        assert mean_stack is not None
        assert pre_factor is not None
        assert source_residual_diagonal is not None
        mean_stack[index].copy_(mean_component)
        factor_rows = int(centered_factor.shape[0])
        if factor_rows:
            pre_factor[factor_cursor : factor_cursor + factor_rows].copy_(
                centered_factor
            )
            pre_factor[factor_cursor : factor_cursor + factor_rows].mul_(
                math.sqrt(weights[index])
            )
            factor_cursor += factor_rows
        source_residual_diagonal.add_(residual_diagonal, alpha=weights[index])
        source_residual += weights[index] * child.residual_spectral_bound
        total_weight += weights[index] * child.total_weight
        context_count += child.context_count
        stratum_counts.append(child.context_count)
        context_axis_hashes.append(
            content_hash(
                child.context_ids,
                prefix=_CONTEXT_AXIS_PREFIX + "device-stratum-",
            )
        )
        measurement_axis_hashes.append(
            content_hash(
                source_measurements[name],
                prefix=_SOURCE_MEASUREMENT_PREFIX + "device-stratum-axis-",
            )
        )
        del mean_component, centered_factor, residual_diagonal, child

    if response_rows is None or mean_stack is None or pre_factor is None:
        raise RuntimeError("device FD assembly finalized no stratum")
    if source_residual_diagonal is None:  # pragma: no cover - coupled allocation.
        raise RuntimeError("device FD residual assembly was not initialized")
    combined_mean = torch.zeros(
        (response_rows, route_count),
        dtype=torch.float64,
        device=runtime_device,
    )
    for index, weight in enumerate(weights):
        combined_mean.add_(mean_stack[index], alpha=weight)
    for index, weight in enumerate(weights):
        destination = pre_factor[
            factor_cursor : factor_cursor + response_rows
        ]
        torch.sub(mean_stack[index], combined_mean, out=destination)
        destination.mul_(math.sqrt(weight))
        factor_cursor += response_rows
    del mean_stack
    if factor_cursor <= 0:
        raise RuntimeError("device FD total-covariance factor is empty")
    active_pre_factor = pre_factor[:factor_cursor]
    pre_diagonal = torch.sum(active_pre_factor * active_pre_factor, dim=0)
    if factor_cursor <= maximum_rank:
        retained_factor = active_pre_factor.clone()
        extra_residual = 0.0
    else:
        retained_factor, extra_residual = _compress_factor_rows_with_gram(
            active_pre_factor,
            maximum_rank=maximum_rank,
        )
    del active_pre_factor, pre_factor
    retained_factor = retained_factor.detach().contiguous()
    post_diagonal = (
        torch.zeros(route_count, dtype=torch.float64, device=runtime_device)
        if int(retained_factor.shape[0]) == 0
        else torch.sum(retained_factor * retained_factor, dim=0)
    )
    recompression_diagonal = (
        torch.zeros_like(pre_diagonal)
        if extra_residual == 0.0
        else _recompression_residual_diagonal_bound(
            pre_diagonal,
            post_diagonal,
            input_factor_rows=factor_cursor,
            retained_rank=int(retained_factor.shape[0]),
            spectral_residual_bound=extra_residual,
        )
    )
    if extra_residual != 0.0:
        # One scalar synchronization makes the coordinate envelope dominate
        # the scalar spectral envelope.  No route tensor crosses the boundary.
        extra_residual = max(
            extra_residual,
            float(torch.max(recompression_diagonal).item()),
        )
    residual_diagonal = torch.clamp(
        source_residual_diagonal + recompression_diagonal,
        min=0.0,
    ).detach().contiguous()
    combined_residual = float(source_residual + extra_residual)
    receipt = V1StratifiedFDDeviceAssemblyReceipt(
        design_hash=design.design_hash,
        route_axis_hash=design.route_axis_hash,
        device_identity=str(runtime_device),
        stratum_names=names,
        stratum_weights=weights,
        stratum_context_counts=tuple(stratum_counts),
        stratum_context_axis_hashes=tuple(context_axis_hashes),
        stratum_source_measurement_axis_hashes=tuple(measurement_axis_hashes),
        response_rows=response_rows,
        route_count=route_count,
        input_factor_rows=factor_cursor,
        retained_rank=int(retained_factor.shape[0]),
        maximum_rank=maximum_rank,
        source_residual_spectral_bound=float(source_residual),
        recompression_residual_spectral_bound=float(extra_residual),
        combined_residual_spectral_bound=combined_residual,
        total_weight=total_weight,
        context_count=context_count,
        mean_component_shape=tuple(int(value) for value in combined_mean.shape),
        centered_factor_shape=tuple(int(value) for value in retained_factor.shape),
        residual_diagonal_shape=tuple(int(value) for value in residual_diagonal.shape),
    )
    return V1StratifiedFDDeviceGeometryState(
        receipt=receipt,
        mean_component=combined_mean.detach().contiguous(),
        centered_factor=retained_factor,
        residual_diagonal=residual_diagonal,
        _issuance_token=_STRATIFIED_FD_DEVICE_STATE_ISSUER,
    )


def assemble_stratified_global_trust_geometry_on_device(
    *,
    geometries_by_stratum: Mapping[str, GlobalTrustGeometryState],
    design: V1StratifiedFDDesign,
    device: Any = "cpu",
    metadata: Mapping[str, object] | None = None,
) -> V1StratifiedFDGeometryResult:
    """Assemble a global trust geometry by exact law of total covariance."""

    if torch is None:
        raise RuntimeError("stratified FD assembly requires PyTorch")
    runtime_device = _torch_device(device)
    ordered = _ordered_geometries(geometries_by_stratum, design)
    weights = tuple(float(weight) for weight in design.stratum_weights)
    reference = ordered[0][1]
    route_count = len(design.route_ids)
    response_rows = int(reference.mean_component.shape[0])
    maximum_rank = min(design.spec.maximum_rank, route_count)
    sketch_rank = min(design.spec.maximum_rank, route_count)
    memory_forecast = forecast_stratified_fd_gpu_memory(
        route_count=route_count,
        stratum_count=len(ordered),
        response_rows=response_rows,
        per_stratum_sketch_rank=max(1, sketch_rank),
        combined_maximum_rank=max(1, maximum_rank),
    )

    mean_stack = torch.empty(
        (len(ordered), response_rows, route_count),
        dtype=torch.float64,
        device=runtime_device,
    )
    for index, (name, geometry) in enumerate(ordered):
        _copy_immutable_fp64_host_array_into_tensor(
            mean_stack[index],
            np.asarray(geometry.mean_component),
            name=f"{name} mean component",
        )
    weight_tensor = torch.tensor(weights, dtype=torch.float64, device=runtime_device)
    combined_mean = torch.sum(weight_tensor[:, None, None] * mean_stack, dim=0)
    input_factor_rows = sum(
        int(geometry.centered_factor.shape[0]) + response_rows
        for _name, geometry in ordered
    )
    pre_factor = torch.empty(
        (input_factor_rows, route_count),
        dtype=torch.float64,
        device=runtime_device,
    )
    factor_cursor = 0
    source_residual = 0.0
    source_residual_diagonal = torch.zeros(
        route_count,
        dtype=torch.float64,
        device=runtime_device,
    )
    total_weight = 0.0
    context_count = 0
    for index, (name, geometry) in enumerate(ordered):
        weight = weights[index]
        root = math.sqrt(weight)
        factor_rows = int(geometry.centered_factor.shape[0])
        if factor_rows:
            factor_destination = pre_factor[
                factor_cursor : factor_cursor + factor_rows
            ]
            _copy_immutable_fp64_host_array_into_tensor(
                factor_destination,
                np.asarray(geometry.centered_factor),
                name=f"{name} centered factor",
            )
            factor_destination.mul_(root)
            factor_cursor += factor_rows
        mean_destination = pre_factor[
            factor_cursor : factor_cursor + response_rows
        ]
        torch.sub(mean_stack[index], combined_mean, out=mean_destination)
        mean_destination.mul_(root)
        factor_cursor += response_rows
        residual_diagonal = _as_device_vector(
            geometry.residual_diagonal_bound,
            device=runtime_device,
            name=f"{name} residual diagonal",
        )
        residual_diagonal.mul_(weight)
        source_residual_diagonal.add_(residual_diagonal)
        del residual_diagonal
        source_residual += weight * float(geometry.residual_spectral_bound)
        total_weight += weight * float(geometry.total_weight)
        context_count += int(geometry.context_count)
    if factor_cursor != input_factor_rows:
        raise RuntimeError("stratified FD factor workspace was not filled exactly")
    pre_diagonal = torch.sum(pre_factor * pre_factor, dim=0)
    retained_factor, extra_residual = _compress_factor_rows_with_gram(
        pre_factor,
        maximum_rank=maximum_rank,
    )
    del pre_factor
    post_diagonal = (
        torch.zeros(route_count, dtype=torch.float64, device=runtime_device)
        if int(retained_factor.shape[0]) == 0
        else torch.sum(retained_factor * retained_factor, dim=0)
    )
    recompression_diagonal = (
        torch.zeros_like(pre_diagonal)
        if extra_residual == 0.0
        else _recompression_residual_diagonal_bound(
            pre_diagonal,
            post_diagonal,
                input_factor_rows=input_factor_rows,
            retained_rank=int(retained_factor.shape[0]),
            spectral_residual_bound=extra_residual,
        )
    )
    if extra_residual != 0.0:
        diagonal_floor = float(torch.max(recompression_diagonal).item())
        extra_residual = max(extra_residual, diagonal_floor)
    residual_diagonal = torch.clamp(
        source_residual_diagonal + recompression_diagonal,
        min=0.0,
    )
    combined_residual = float(source_residual + extra_residual)
    mean_host = combined_mean.detach().cpu().numpy()
    factor_host = retained_factor.detach().cpu().numpy()
    residual_diagonal_host = residual_diagonal.detach().cpu().numpy()

    stratum_hashes = tuple((name, geometry.content_hash) for name, geometry in ordered)
    context_axis_hash = content_hash(
        {
            "design_hash": design.design_hash,
            "stratum_context_axis_hashes": tuple(
                (name, geometry.context_axis_hash) for name, geometry in ordered
            ),
            "stratum_weights": design.stratum_weights,
        },
        prefix=_CONTEXT_AXIS_PREFIX,
    )
    source_measurement_hash = content_hash(
        tuple((name, geometry.source_measurement_hashes) for name, geometry in ordered),
        prefix=_SOURCE_MEASUREMENT_PREFIX,
    )
    caller_metadata = dict(metadata or {})
    protected = {
        "stratified_fd_design_hash",
        "stratified_fd_stratum_geometry_hashes",
        "stratified_fd_law_of_total_covariance",
        "stratified_fd_no_reweighted_mixture_stream",
        "stratified_fd_source_proof_status",
        "host_context_route_rows_transferred",
        "recompression_residual_spectral_bound",
        "residual_diagonal_includes_recompression_loss",
    }
    if protected.intersection(caller_metadata):
        raise ValueError("stratified FD metadata cannot override protected fields")
    geometry = GlobalTrustGeometryState(
        spec=design.spec,
        route_ids=design.route_ids,
        mean_component=mean_host,
        centered_factor=factor_host,
        residual_spectral_bound=combined_residual,
        residual_diagonal_bound=residual_diagonal_host,
        total_weight=total_weight,
        context_count=context_count,
        context_axis_hash=context_axis_hash,
        measured_mask_hash=design.measured_mask_hash,
        calibration_state_id=design.calibration_state_id,
        source_measurement_hashes=(source_measurement_hash,),
        derivation_receipt=None,
        metadata={
            **caller_metadata,
            "stratified_fd_design_hash": design.design_hash,
            "stratified_fd_stratum_geometry_hashes": stratum_hashes,
            "stratified_fd_law_of_total_covariance": True,
            "stratified_fd_no_reweighted_mixture_stream": True,
            "stratified_fd_source_proof_status": STRATIFIED_FD_SOURCE_PROOF_STATUS,
            "host_context_route_rows_transferred": False,
            "recompression_residual_spectral_bound": float(extra_residual),
            "residual_diagonal_includes_recompression_loss": True,
        },
    )
    receipt = V1StratifiedFDAssemblyReceipt(
        design_hash=design.design_hash,
        combined_geometry_hash=geometry.content_hash,
        stratum_geometry_hashes=stratum_hashes,
        route_axis_hash=design.route_axis_hash,
        device_identity=_device_identity(runtime_device),
        response_rows=response_rows,
        route_count=route_count,
        input_factor_rows=input_factor_rows,
        retained_rank=int(factor_host.shape[0]),
        maximum_rank=maximum_rank,
        source_residual_spectral_bound=float(source_residual),
        recompression_residual_spectral_bound=float(extra_residual),
        combined_residual_spectral_bound=combined_residual,
        mean_component_hash=content_hash(
            mean_host,
            prefix="partition-free-v1-stratified-fd-mean-",
        ),
        centered_factor_hash=content_hash(
            factor_host,
            prefix="partition-free-v1-stratified-fd-factor-",
        ),
        residual_diagonal_hash=content_hash(
            residual_diagonal_host,
            prefix="partition-free-v1-stratified-fd-residual-diagonal-",
        ),
        total_weight=total_weight,
        context_count=context_count,
        peak_device_bytes=memory_forecast.peak_device_bytes,
        aggregate_host_transfer_bytes=memory_forecast.aggregate_host_transfer_bytes,
    )
    return V1StratifiedFDGeometryResult(
        design=design,
        geometry=geometry,
        receipt=receipt,
        memory_forecast=memory_forecast,
    )


class V1StratifiedDeviceFDTrustAccumulator:
    """Separate device FD streams for fixed train-context strata."""

    def __init__(
        self,
        design: V1StratifiedFDDesign,
        *,
        sketch_rank: int | None = None,
        device: Any = "cpu",
        emit_derivation_receipt: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("stratified device FD accumulation requires PyTorch")
        if type(design) is not V1StratifiedFDDesign:
            raise TypeError("stratified device FD accumulator requires its design")
        design.assert_integrity()
        rank = (
            min(design.spec.maximum_rank, len(design.route_ids))
            if sketch_rank is None
            else _positive_int(sketch_rank, name="sketch rank")
        )
        if rank > len(design.route_ids):
            raise ValueError("sketch rank cannot exceed the route axis")
        self.design = design
        self.sketch_rank = int(rank)
        self.device = _torch_device(device)
        self.emit_derivation_receipt = bool(emit_derivation_receipt)
        self._accumulators = {
            name: DeviceGramFrequentDirectionsTrustAccumulator(
                route_ids=design.route_ids,
                sketch_rank=self.sketch_rank,
                device=self.device,
                emit_derivation_receipt=self.emit_derivation_receipt,
            )
            for name in design.stratum_names
        }
        self._all_context_ids: set[str] = set()
        self._source_measurements: dict[str, list[str]] = {
            name: [] for name in design.stratum_names
        }
        self._packet_boundary_index = 0
        self._consumed = False

    def append_labeled_rows(
        self,
        *,
        rows: Any,
        context_ids: Sequence[str],
        stratum_labels: Sequence[str],
        source_measurement_hashes: Sequence[str],
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        """Append one all-same-stratum row packet to exactly one FD stream."""

        if self._consumed:
            raise RuntimeError("consumed stratified FD state cannot accept rows")

        labels = tuple(_identity(item, name="stratum label") for item in stratum_labels)
        contexts = tuple(_identity(item, name="context id") for item in context_ids)
        measurements = tuple(
            _identity(item, name="source measurement hash")
            for item in source_measurement_hashes
        )
        if not contexts or len(contexts) != len(labels) or len(contexts) != len(measurements):
            raise ValueError("stratified FD row ledgers must align and be nonempty")
        if len(set(labels)) != 1:
            raise ValueError("one FD update cannot mix strata")
        stratum = labels[0]
        if stratum not in self._accumulators:
            raise ValueError("row packet names an unregistered stratum")
        if len(set(contexts)) != len(contexts) or self._all_context_ids.intersection(contexts):
            raise ValueError("stratified FD contexts must be globally unique")
        self._accumulators[stratum].update(
            rows,
            context_ids=contexts,
            weights=weights,
        )
        self._all_context_ids.update(contexts)
        self._source_measurements[stratum].extend(measurements)
        self._packet_boundary_index += 1

    def snapshot(self) -> V1StratifiedFDAccumulatorSnapshot:
        if self._consumed:
            raise RuntimeError("consumed stratified FD state has no continuation snapshot")
        return V1StratifiedFDAccumulatorSnapshot(
            design_hash=self.design.design_hash,
            design_stratum_names=self.design.stratum_names,
            route_axis_hash=self.design.route_axis_hash,
            sketch_rank=self.sketch_rank,
            packet_boundary_index=self._packet_boundary_index,
            stratum_snapshots=tuple(
                (name, self._accumulators[name].snapshot())
                for name in self.design.stratum_names
            ),
            source_measurement_hashes_by_stratum=tuple(
                (name, tuple(self._source_measurements[name]))
                for name in self.design.stratum_names
            ),
        )

    @classmethod
    def _validate_snapshot_binding(
        cls,
        snapshot: V1StratifiedFDAccumulatorSnapshot,
        *,
        design: V1StratifiedFDDesign,
        sketch_rank: int,
    ) -> tuple[int, bool]:
        if type(snapshot) is not V1StratifiedFDAccumulatorSnapshot:
            raise TypeError("stratified FD restoration requires its exact snapshot type")
        snapshot.assert_integrity()
        if type(design) is not V1StratifiedFDDesign:
            raise TypeError("stratified FD restoration requires its exact design type")
        design.assert_integrity()
        rank = _positive_int(sketch_rank, name="restored sketch rank")
        if (
            snapshot.design_hash != design.design_hash
            or snapshot.design_stratum_names != design.stratum_names
            or snapshot.route_axis_hash != design.route_axis_hash
            or snapshot.sketch_rank != rank
        ):
            raise ValueError(
                "stratified FD snapshot differs from the current design or rank"
            )
        proof_statuses = {
            stratum_snapshot.bounded_source_proof is not None
            for _name, stratum_snapshot in snapshot.stratum_snapshots
        }
        if len(proof_statuses) != 1:
            raise ValueError(
                "stratified FD snapshot mixes source-proof emission policies"
            )
        return rank, proof_statuses.pop()

    @classmethod
    def from_snapshot(
        cls,
        snapshot: V1StratifiedFDAccumulatorSnapshot,
        *,
        design: V1StratifiedFDDesign,
        sketch_rank: int,
        device: Any,
    ) -> "V1StratifiedDeviceFDTrustAccumulator":
        """Restore exact packet-boundary state against current campaign identity.

        The caller must restate the design and sketch rank.  A checkpoint is
        evidence for continuation state; it is never authority to change the
        route axis, sampling strata, geometry specification, or numerical
        fidelity.  Source-measurement ledgers are restored alongside the
        sketches so final geometry cannot acquire provenance that was absent
        from the persisted state.
        """

        rank, emit_derivation_receipt = cls._validate_snapshot_binding(
            snapshot,
            design=design,
            sketch_rank=sketch_rank,
        )
        restored = cls(
            design,
            sketch_rank=rank,
            device=device,
            emit_derivation_receipt=emit_derivation_receipt,
        )
        # The constructor has already allocated exactly one private FD child
        # per stratum. Restore into those children. Replacing this dictionary
        # with newly restored children would transiently keep two complete
        # 57M-route FD states alive (an H200 out-of-memory failure).
        for name, stratum_snapshot in snapshot.stratum_snapshots:
            restored._accumulators[name].restore_snapshot_into_empty(
                stratum_snapshot
            )
        restored._source_measurements = {
            name: list(values)
            for name, values in snapshot.source_measurement_hashes_by_stratum
        }
        restored._all_context_ids = {
            context_id
            for accumulator in restored._accumulators.values()
            for context_id in accumulator.context_ids
        }
        restored._packet_boundary_index = snapshot.packet_boundary_index
        if restored.snapshot().snapshot_hash != snapshot.snapshot_hash:
            raise RuntimeError(
                "restored stratified FD state does not replay its checkpoint"
            )
        return restored

    @classmethod
    def finalize_snapshot_combined_consuming(
        cls,
        snapshot: V1StratifiedFDAccumulatorSnapshot,
        *,
        design: V1StratifiedFDDesign,
        sketch_rank: int,
        device: Any,
        metadata: Mapping[str, object] | None = None,
    ) -> V1StratifiedFDGeometryResult:
        """Finalize an immutable snapshot with only one live child at a time."""

        rank, _emit_derivation_receipt = cls._validate_snapshot_binding(
            snapshot,
            design=design,
            sketch_rank=sketch_rank,
        )
        runtime_device = _torch_device(device)
        source_measurements = dict(
            snapshot.source_measurement_hashes_by_stratum
        )
        geometries: dict[str, GlobalTrustGeometryState] = {}
        for name, stratum_snapshot in snapshot.stratum_snapshots:
            source_hashes = tuple(source_measurements[name])
            if not source_hashes:
                raise ValueError("cannot finalize stratified FD with an empty stratum")
            accumulator = DeviceGramFrequentDirectionsTrustAccumulator.from_snapshot(
                stratum_snapshot,
                route_ids=design.route_ids,
                sketch_rank=rank,
                device=runtime_device,
            )
            geometries[name] = accumulator.finalize(
                spec=design.spec,
                measured_mask_hash=design.measured_mask_hash,
                calibration_state_id=design.calibration_state_id,
                source_measurement_hashes=source_hashes,
                metadata={
                    "stratified_fd_design_hash": design.design_hash,
                    "stratified_fd_stratum_name": name,
                    "stratified_fd_stream_is_unreweighted": True,
                },
            )
            del accumulator
        if runtime_device.type == "cuda":
            # Reuse the first child's identical cached blocks for the second,
            # then evict once before differently shaped combined assembly.
            torch.cuda.empty_cache()
        return assemble_stratified_global_trust_geometry_on_device(
            geometries_by_stratum=geometries,
            design=design,
            device=runtime_device,
            metadata=metadata,
        )

    def finalize_combined(
        self,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> V1StratifiedFDGeometryResult:
        if self._consumed:
            raise RuntimeError("consumed stratified FD state cannot be finalized")
        geometries: dict[str, GlobalTrustGeometryState] = {}
        for name in self.design.stratum_names:
            source_hashes = tuple(self._source_measurements[name])
            if not source_hashes:
                raise ValueError("cannot finalize stratified FD with an empty stratum")
            geometries[name] = self._accumulators[name].finalize_non_mutating(
                spec=self.design.spec,
                measured_mask_hash=self.design.measured_mask_hash,
                calibration_state_id=self.design.calibration_state_id,
                source_measurement_hashes=source_hashes,
                metadata={
                    "stratified_fd_design_hash": self.design.design_hash,
                    "stratified_fd_stratum_name": name,
                    "stratified_fd_stream_is_unreweighted": True,
                },
            )
        return assemble_stratified_global_trust_geometry_on_device(
            geometries_by_stratum=geometries,
            design=self.design,
            device=self.device,
            metadata=metadata,
        )

    def finalize_combined_device_consuming(
        self,
    ) -> V1StratifiedFDDeviceGeometryState:
        """Consume all strata into a one-shot device-only covariance state.

        This is the live proposal/KKT handoff.  It neither invokes the
        portable ``GlobalTrustGeometryState`` constructor nor computes hashes
        over route tensors, so its route-scale D2H and content-digest byte
        counts are exactly zero.  Any failure after consumption is fail-closed;
        callers must replay the sealed D prefix rather than guess or reuse a
        partially consumed chart.
        """

        if self._consumed:
            raise RuntimeError("consumed stratified FD state cannot be finalized")
        if any(not self._source_measurements[name] for name in self.design.stratum_names):
            raise ValueError("cannot finalize stratified FD with an empty stratum")
        self._consumed = True
        try:
            return _assemble_stratified_device_fd_consuming(
                accumulators_by_stratum=self._accumulators,
                source_measurements_by_stratum=self._source_measurements,
                design=self.design,
                sketch_rank=self.sketch_rank,
                device=self.device,
            )
        finally:
            self._accumulators.clear()

    def finalize_combined_consuming(
        self,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> V1StratifiedFDGeometryResult:
        """Consume an ephemeral restored state into one combined geometry.

        Production summary assembly restores from an immutable caller-owned
        checkpoint and never needs a second continuation from the device copy.
        Consuming each stratum in place therefore preserves the checkpoint
        authority while avoiding ``finalize_non_mutating``'s complete child
        clone. Each finalized child is evicted before the next stratum and all
        FD state is evicted before law-of-total-covariance recompression.
        """

        if self._consumed:
            raise RuntimeError("consumed stratified FD state cannot be finalized")
        self._consumed = True
        geometries: dict[str, GlobalTrustGeometryState] = {}
        for name in self.design.stratum_names:
            source_hashes = tuple(self._source_measurements[name])
            if not source_hashes:
                raise ValueError("cannot finalize stratified FD with an empty stratum")
            accumulator = self._accumulators.pop(name)
            geometries[name] = accumulator.finalize(
                spec=self.design.spec,
                measured_mask_hash=self.design.measured_mask_hash,
                calibration_state_id=self.design.calibration_state_id,
                source_measurement_hashes=source_hashes,
                metadata={
                    "stratified_fd_design_hash": self.design.design_hash,
                    "stratified_fd_stratum_name": name,
                    "stratified_fd_stream_is_unreweighted": True,
                },
            )
            del accumulator
        if self.device.type == "cuda":
            # This is a registered phase boundary, not a hot-path polling call:
            # assembly must not retain the consumed FD allocator cache.
            torch.cuda.empty_cache()
        return assemble_stratified_global_trust_geometry_on_device(
            geometries_by_stratum=geometries,
            design=self.design,
            device=self.device,
            metadata=metadata,
        )


__all__ = [
    "STRATIFIED_FD_RECOMPRESSION_ARITHMETIC_ID",
    "STRATIFIED_FD_ROW_ARITHMETIC_ID",
    "STRATIFIED_FD_SOURCE_PROOF_STATUS",
    "V1StratifiedDeviceFDTrustAccumulator",
    "V1StratifiedFDAccumulatorSnapshot",
    "V1StratifiedFDAssemblyReceipt",
    "V1StratifiedFDDesign",
    "V1StratifiedFDDeviceAssemblyReceipt",
    "V1StratifiedFDDeviceGeometryState",
    "V1StratifiedFDGeometryResult",
    "V1StratifiedFDMemoryForecast",
    "assemble_stratified_global_trust_geometry_on_device",
    "forecast_stratified_fd_gpu_memory",
]
