"""CUDA-native native-fiber signed corrected-work operator for modern v1.

This module is the production-scale linear primitive that the route-scale
signed-F proposal solver needs.  It deliberately does **not** solve KKT/PDHG by
itself and it carries no calibration, oracle, mask-commit, or Pareto authority.

The operator represents corrected evidence rows through either

* exact row identity factors, ``A = I_context @ rows``; or
* generic finite-rank factors, ``A = U @ B``, plus a nonnegative route residual
  envelope.

For route actions ``z`` it exposes the native-fiber signed map

    forward(z)[x, f, lane] = sum_{j in native fiber f} A[x, j] z[j, lane].

The adjoint is the exact binary64 transpose of this same factored map over the
registered target-major native-fiber cuts.  Route-to-fiber membership is
replayed from the O(cuts) ``V1NativeDestinationFiberAxisReceipt``; this file
never materializes route-scale fiber IDs or layer strings.

Production construction requires CUDA and refuses CPU fallback.  A private CPU
reference mode exists only for focused tests and uses the same cut/fiber tiling
loop as the CUDA path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Final

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_native_fiber_axis import (
    V1NativeDestinationFiberAxisReceipt,
)
from mechanistic_systems.circuits.irg.modern_v1_signed_work_row_spool_cuda import (
    V1SignedWorkRowBankDeviceHandle,
    V1SignedWorkRowBankLease,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

try:  # pragma: no cover - import availability is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


SIGNED_WORK_OPERATOR_CUDA_DESIGN_SCHEMA: Final = (
    "partition_free_v1_native_fiber_signed_work_operator_cuda_design_v1"
)
SIGNED_WORK_OPERATOR_CUDA_HBM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_native_fiber_signed_work_operator_cuda_hbm_forecast_v1"
)
SIGNED_WORK_OPERATOR_CUDA_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_native_fiber_signed_work_operator_cuda_receipt_v1"
)
SIGNED_WORK_OPERATOR_CUDA_TELEMETRY_SCHEMA: Final = (
    "partition_free_v1_native_fiber_signed_work_operator_cuda_telemetry_v1"
)
SIGNED_WORK_OPERATOR_CUDA_ALGORITHM: Final = (
    "target_major_native_fiber_factored_B_forward_adjoint_no_floating_atomics_"
    "fp64_tiled_v1"
)
SIGNED_WORK_OPERATOR_CUDA_AUTHORITY: Final = (
    "signed_corrected_work_linear_operator_only_no_solver_no_C_K_O_no_oracle_"
    "no_mask_commit_no_pareto_authority_v1"
)
SIGNED_WORK_OPERATOR_CUDA_MEMORY_SCOPE: Final = (
    "factored_corrected_evidence_plus_context_fiber_lane_outputs_no_route_scale_"
    "fiber_id_or_layer_string_materialization_v1"
)

_DESIGN_PREFIX: Final = "partition-free-v1-signed-work-cuda-design-"
_FORECAST_PREFIX: Final = "partition-free-v1-signed-work-cuda-hbm-"
_RECEIPT_PREFIX: Final = "partition-free-v1-signed-work-cuda-receipt-"
_TELEMETRY_PREFIX: Final = "partition-free-v1-signed-work-cuda-telemetry-"
_LE_F64: Final = np.dtype("<f8")
_F64_BYTES: Final = 8
_EXACT_ROW_FACTORIZATION_KINDS: Final = (
    "exact_rows_identity",
    "exact_rows_spool_alias",
)
_SIGNED_WORK_FACTORIZATION_KINDS: Final = (
    *_EXACT_ROW_FACTORIZATION_KINDS,
    "generic_factors",
)


class V1SignedWorkOperatorCUDARefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    AXIS_MISMATCH = "axis_mismatch"
    NUMERICAL_REFUSAL = "numerical_refusal"
    DEVICE_REFUSAL = "device_refusal"
    MEMORY_REFUSAL = "memory_refusal"
    EVICTED = "evicted"


class V1SignedWorkOperatorCUDARefusal(RuntimeError):
    """Typed refusal for the production signed-work CUDA operator."""

    def __init__(
        self,
        code: V1SignedWorkOperatorCUDARefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(code: V1SignedWorkOperatorCUDARefusalCode, message: str) -> None:
    raise V1SignedWorkOperatorCUDARefusal(code, message)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one exact positive integer",
        )
    result = int(value)
    if result <= 0:
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
            f"{name} must be positive",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
            f"{name} must be nonnegative",
        )
    return result


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _finite_fraction(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
            f"{name} must be one finite real value",
        )
    result = float(value)
    if not math.isfinite(result):
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be finite",
        )
    return result


def _immutable_f64_matrix(
    value: object,
    *,
    name: str,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    try:
        array = np.array(value, dtype=_LE_F64, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise V1SignedWorkOperatorCUDARefusal(
            V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
            f"{name} must be a finite float64 matrix",
        ) from error
    if array.ndim != 2:
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
            f"{name} must be two-dimensional",
        )
    if shape is not None and array.shape != shape:
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
            f"{name} must have shape {shape}",
        )
    if not np.isfinite(array).all():
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.NUMERICAL_REFUSAL,
            f"{name} must contain only finite values",
        )
    array[array == 0.0] = 0.0
    result = np.frombuffer(array.tobytes(order="C"), dtype=_LE_F64).reshape(
        array.shape
    )
    result.setflags(write=False)
    return result


def _immutable_f64_vector(
    value: object,
    *,
    name: str,
    size: int | None = None,
    nonnegative: bool = False,
) -> np.ndarray:
    try:
        array = np.array(value, dtype=_LE_F64, order="C", copy=True).reshape(-1)
    except (TypeError, ValueError, OverflowError) as error:
        raise V1SignedWorkOperatorCUDARefusal(
            V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
            f"{name} must be a finite float64 vector",
        ) from error
    if size is not None and array.shape != (size,):
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
            f"{name} must have shape ({size},)",
        )
    if not np.isfinite(array).all():
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.NUMERICAL_REFUSAL,
            f"{name} must contain only finite values",
        )
    if nonnegative and bool(np.any(array < 0.0)):
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.NUMERICAL_REFUSAL,
            f"{name} must be nonnegative",
        )
    array[array == 0.0] = 0.0
    result = np.frombuffer(array.tobytes(order="C"), dtype=_LE_F64).reshape(
        array.shape
    )
    result.setflags(write=False)
    return result


def _array_digest(array: np.ndarray) -> dict[str, object]:
    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": tuple(int(item) for item in value.shape),
        "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest(),
        "bytes": int(value.nbytes),
    }


def signed_work_operator_cuda_source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class V1SignedWorkOperatorCUDAExecutionDesign:
    route_tile_size: int = 131_072
    fiber_tile_size: int = 4096
    maximum_lanes: int = 16
    hbm_ceiling_fraction: float = 0.85
    production_cuda_required: bool = True
    allow_private_cpu_reference: bool = False
    schema: str = SIGNED_WORK_OPERATOR_CUDA_DESIGN_SCHEMA
    algorithm_id: str = SIGNED_WORK_OPERATOR_CUDA_ALGORITHM
    forbid_route_scale_fiber_ids: bool = True
    floating_atomics_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_tile_size",
            _positive_int(self.route_tile_size, name="route tile size"),
        )
        object.__setattr__(
            self,
            "fiber_tile_size",
            _positive_int(self.fiber_tile_size, name="fiber tile size"),
        )
        object.__setattr__(
            self,
            "maximum_lanes",
            _positive_int(self.maximum_lanes, name="maximum lanes"),
        )
        ceiling = _finite_fraction(
            self.hbm_ceiling_fraction,
            name="HBM ceiling fraction",
        )
        if not (0.0 < ceiling <= 0.85):
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.MEMORY_REFUSAL,
                "signed-work CUDA HBM ceiling must be in (0, 0.85]",
            )
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        if type(self.production_cuda_required) is not bool:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "production_cuda_required must be an exact bool",
            )
        if type(self.allow_private_cpu_reference) is not bool:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "allow_private_cpu_reference must be an exact bool",
            )
        if self.production_cuda_required and self.allow_private_cpu_reference:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
                "private CPU reference mode cannot be enabled on a production "
                "CUDA-required design",
            )
        if self.forbid_route_scale_fiber_ids is not True:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                "route-scale fiber ID materialization is a closed gate",
            )
        if self.floating_atomics_allowed is not False:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.NUMERICAL_REFUSAL,
                "floating atomics are forbidden for signed-work operator math",
            )
        if self.schema != SIGNED_WORK_OPERATOR_CUDA_DESIGN_SCHEMA:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "unsupported signed-work CUDA execution design schema",
            )
        if self.algorithm_id != SIGNED_WORK_OPERATOR_CUDA_ALGORITHM:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "unsupported signed-work CUDA algorithm",
            )

    @property
    def design_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "route_tile_size": self.route_tile_size,
                "fiber_tile_size": self.fiber_tile_size,
                "maximum_lanes": self.maximum_lanes,
                "hbm_ceiling_fraction": self.hbm_ceiling_fraction,
                "production_cuda_required": self.production_cuda_required,
                "allow_private_cpu_reference": self.allow_private_cpu_reference,
                "algorithm_id": self.algorithm_id,
                "forbid_route_scale_fiber_ids": self.forbid_route_scale_fiber_ids,
                "floating_atomics_allowed": self.floating_atomics_allowed,
            },
            prefix=_DESIGN_PREFIX,
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkOperatorCUDAHBMForecast:
    design_hash: str
    native_fiber_axis_receipt_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    context_count: int
    rank_count: int
    fiber_count: int
    lane_count: int
    persistent_factor_bytes: int
    persistent_axis_bytes: int
    result_cell_bytes: int
    adjoint_result_bytes: int
    temporary_tile_bytes_upper: int
    registered_device_bytes_upper: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    admitted_under_85_percent_h200_style_ceiling: bool
    borrowed_row_bank_bytes: int = 0
    logical_zero_residual_storage_bytes: int = 0
    route_scale_fiber_id_array_materialized: bool = False
    schema: str = SIGNED_WORK_OPERATOR_CUDA_HBM_FORECAST_SCHEMA
    _forecast_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "design_hash",
            "native_fiber_axis_receipt_hash",
            "route_axis_hash",
            "route_id_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        for name in (
            "route_count",
            "context_count",
            "rank_count",
            "fiber_count",
            "lane_count",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        for name in (
            "persistent_factor_bytes",
            "persistent_axis_bytes",
            "result_cell_bytes",
            "adjoint_result_bytes",
            "temporary_tile_bytes_upper",
            "registered_device_bytes_upper",
            "device_total_memory_bytes",
            "device_free_memory_bytes",
            "borrowed_row_bank_bytes",
            "logical_zero_residual_storage_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name=name),
            )
        if type(self.admitted_under_85_percent_h200_style_ceiling) is not bool:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "HBM admission flag must be an exact bool",
            )
        if self.route_scale_fiber_id_array_materialized is not False:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                "signed-work CUDA forecast cannot materialize route-scale fiber IDs",
            )
        if self.schema != SIGNED_WORK_OPERATOR_CUDA_HBM_FORECAST_SCHEMA:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "unsupported HBM forecast schema",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "native_fiber_axis_receipt_hash": self.native_fiber_axis_receipt_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_id_axis_hash": self.route_id_axis_hash,
            "route_count": self.route_count,
            "context_count": self.context_count,
            "rank_count": self.rank_count,
            "fiber_count": self.fiber_count,
            "lane_count": self.lane_count,
            "persistent_factor_bytes": self.persistent_factor_bytes,
            "persistent_axis_bytes": self.persistent_axis_bytes,
            "result_cell_bytes": self.result_cell_bytes,
            "adjoint_result_bytes": self.adjoint_result_bytes,
            "temporary_tile_bytes_upper": self.temporary_tile_bytes_upper,
            "registered_device_bytes_upper": self.registered_device_bytes_upper,
            "device_total_memory_bytes": self.device_total_memory_bytes,
            "device_free_memory_bytes": self.device_free_memory_bytes,
            "admitted_under_85_percent_h200_style_ceiling": (
                self.admitted_under_85_percent_h200_style_ceiling
            ),
            "borrowed_row_bank_bytes": self.borrowed_row_bank_bytes,
            "logical_zero_residual_storage_bytes": (
                self.logical_zero_residual_storage_bytes
            ),
            "route_scale_fiber_id_array_materialized": (
                self.route_scale_fiber_id_array_materialized
            ),
        }

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self) -> str:
        observed = content_hash(self._payload(), prefix=_FORECAST_PREFIX)
        if self._forecast_hash is not None and observed != self._forecast_hash:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.MEMORY_REFUSAL,
                "signed-work CUDA HBM forecast was tampered",
            )
        object.__setattr__(self, "_forecast_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SignedWorkOperatorCUDAReceipt:
    source_geometry_hash: str
    factorization_kind: str
    design_hash: str
    forecast_hash: str
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    context_count: int
    rank_count: int
    fiber_count: int
    cut_count: int
    context_weights_digest: dict[str, object]
    left_context_factor_digest: dict[str, object]
    right_route_factor_digest: dict[str, object]
    residual_linear_digest: dict[str, object]
    device_identity: str
    production_cuda_execution: bool
    private_cpu_reference_execution: bool
    row_spool_receipt_hash: str | None = None
    row_spool_zero_copy_alias: bool = False
    logical_zero_residual: bool = False
    borrowed_row_bank_bytes: int = 0
    algorithm_id: str = SIGNED_WORK_OPERATOR_CUDA_ALGORITHM
    authority_scope: str = SIGNED_WORK_OPERATOR_CUDA_AUTHORITY
    memory_scope: str = SIGNED_WORK_OPERATOR_CUDA_MEMORY_SCOPE
    route_scale_fiber_id_array_materialized: bool = False
    floating_atomics_used: bool = False
    schema: str = SIGNED_WORK_OPERATOR_CUDA_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "source_geometry_hash",
            "factorization_kind",
            "design_hash",
            "forecast_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "device_identity",
            "algorithm_id",
            "authority_scope",
            "memory_scope",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if self.factorization_kind not in _SIGNED_WORK_FACTORIZATION_KINDS:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "unsupported signed-work factorization kind",
            )
        for name in (
            "route_count",
            "context_count",
            "rank_count",
            "fiber_count",
            "cut_count",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        for name in ("production_cuda_execution", "private_cpu_reference_execution"):
            if type(getattr(self, name)) is not bool:
                _refuse(
                    V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                    f"{name} must be an exact bool",
                )
        for name in ("row_spool_zero_copy_alias", "logical_zero_residual"):
            if type(getattr(self, name)) is not bool:
                _refuse(
                    V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                    f"{name} must be an exact bool",
                )
        object.__setattr__(
            self,
            "borrowed_row_bank_bytes",
            _nonnegative_int(self.borrowed_row_bank_bytes, name="borrowed row-bank bytes"),
        )
        if self.row_spool_receipt_hash is not None:
            object.__setattr__(
                self,
                "row_spool_receipt_hash",
                _identity(
                    self.row_spool_receipt_hash,
                    name="row-spool receipt hash",
                ),
            )
        if self.row_spool_zero_copy_alias:
            if (
                self.factorization_kind != "exact_rows_spool_alias"
                or self.row_spool_receipt_hash is None
                or self.borrowed_row_bank_bytes <= 0
            ):
                _refuse(
                    V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                    "row-spool zero-copy alias requires exact row-spool custody",
                )
        elif self.row_spool_receipt_hash is not None or self.borrowed_row_bank_bytes:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "row-spool custody fields require a zero-copy alias receipt",
            )
        if self.production_cuda_execution == self.private_cpu_reference_execution:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
                "exactly one execution custody mode must be active",
            )
        if self.algorithm_id != SIGNED_WORK_OPERATOR_CUDA_ALGORITHM:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "unsupported signed-work operator algorithm",
            )
        if self.authority_scope != SIGNED_WORK_OPERATOR_CUDA_AUTHORITY:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "signed-work CUDA receipt cannot claim broader authority",
            )
        if self.memory_scope != SIGNED_WORK_OPERATOR_CUDA_MEMORY_SCOPE:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "signed-work CUDA receipt memory scope is invalid",
            )
        if self.route_scale_fiber_id_array_materialized is not False:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                "route-scale fiber IDs are forbidden",
            )
        if self.floating_atomics_used is not False:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.NUMERICAL_REFUSAL,
                "floating atomics are forbidden",
            )
        if self.schema != SIGNED_WORK_OPERATOR_CUDA_RECEIPT_SCHEMA:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "unsupported signed-work CUDA receipt schema",
            )
        self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "source_geometry_hash",
                "factorization_kind",
                "design_hash",
                "forecast_hash",
                "native_fiber_axis_receipt_hash",
                "native_fiber_measure_hash",
                "route_axis_hash",
                "route_id_axis_hash",
                "route_count",
                "context_count",
                "rank_count",
                "fiber_count",
                "cut_count",
                "context_weights_digest",
                "left_context_factor_digest",
                "right_route_factor_digest",
                "residual_linear_digest",
                "device_identity",
                "production_cuda_execution",
                "private_cpu_reference_execution",
                "row_spool_receipt_hash",
                "row_spool_zero_copy_alias",
                "logical_zero_residual",
                "borrowed_row_bank_bytes",
                "algorithm_id",
                "authority_scope",
                "memory_scope",
                "route_scale_fiber_id_array_materialized",
                "floating_atomics_used",
            )
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self) -> str:
        observed = content_hash(self._payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.NUMERICAL_REFUSAL,
                "signed-work CUDA receipt was tampered",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1SignedWorkOperatorCUDATelemetry:
    receipt_hash: str
    operation: str
    device_identity: str
    route_count: int
    context_count: int
    rank_count: int
    fiber_count: int
    lane_count: int
    cut_count: int
    fiber_tile_count: int
    route_value_reads: int
    factor_value_reads_upper: int
    temporary_tile_bytes_upper: int
    result_bytes: int
    evicted_after_operation: bool = False
    route_scale_fiber_id_array_materialized: bool = False
    floating_atomics_used: bool = False
    schema: str = SIGNED_WORK_OPERATOR_CUDA_TELEMETRY_SCHEMA

    @property
    def telemetry_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in (
                    "schema",
                    "receipt_hash",
                    "operation",
                    "device_identity",
                    "route_count",
                    "context_count",
                    "rank_count",
                    "fiber_count",
                    "lane_count",
                    "cut_count",
                    "fiber_tile_count",
                    "route_value_reads",
                    "factor_value_reads_upper",
                    "temporary_tile_bytes_upper",
                    "result_bytes",
                    "evicted_after_operation",
                    "route_scale_fiber_id_array_materialized",
                    "floating_atomics_used",
                )
            },
            prefix=_TELEMETRY_PREFIX,
        )


def build_v1_signed_work_operator_cuda_hbm_forecast(
    *,
    design: V1SignedWorkOperatorCUDAExecutionDesign,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    route_axis_hash: str,
    route_id_axis_hash: str,
    context_count: int,
    rank_count: int,
    lane_count: int,
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
    borrowed_row_bank_bytes: int = 0,
    logical_zero_residual: bool = False,
    _admission_available_bytes: int | None = None,
) -> V1SignedWorkOperatorCUDAHBMForecast:
    """Return the pre-outcome HBM forecast for this operator custody envelope."""

    if type(design) is not V1SignedWorkOperatorCUDAExecutionDesign:
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
            "HBM forecast requires the exact signed-work CUDA design",
        )
    if type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
            "HBM forecast requires the exact native-fiber axis receipt",
        )
    native_fiber_axis_receipt.assert_integrity()
    route_axis_hash = _identity(route_axis_hash, name="route axis hash")
    route_id_axis_hash = _identity(route_id_axis_hash, name="route-ID axis hash")
    if (
        route_axis_hash != native_fiber_axis_receipt.route_axis_hash
        or route_id_axis_hash != native_fiber_axis_receipt.route_id_axis_hash
    ):
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
            "signed-work operator route axes differ from the native-fiber receipt",
        )
    context_count = _positive_int(context_count, name="context count")
    rank_count = _positive_int(rank_count, name="rank count")
    lane_count = _positive_int(lane_count, name="lane count")
    if lane_count > design.maximum_lanes:
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.MEMORY_REFUSAL,
            "lane count exceeds the registered signed-work CUDA design",
        )
    route_count = native_fiber_axis_receipt.route_count
    fiber_count = native_fiber_axis_receipt.fiber_count
    borrowed = _nonnegative_int(
        borrowed_row_bank_bytes,
        name="borrowed row-bank bytes",
    )
    right_factor_bytes = rank_count * route_count * _F64_BYTES
    if borrowed > right_factor_bytes:
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.MEMORY_REFUSAL,
            "borrowed row-bank bytes exceed the right-factor envelope",
        )
    if type(logical_zero_residual) is not bool:
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
            "logical_zero_residual must be an exact bool",
        )
    residual_storage_bytes = _F64_BYTES if logical_zero_residual else route_count * _F64_BYTES
    persistent_factor_bytes = (
        context_count * rank_count * _F64_BYTES
        + max(0, right_factor_bytes - borrowed)
        + context_count * _F64_BYTES
        + residual_storage_bytes
    )
    persistent_axis_bytes = native_fiber_axis_receipt.cut_count * 6 * _F64_BYTES
    result_cell_bytes = context_count * fiber_count * lane_count * _F64_BYTES
    adjoint_result_bytes = route_count * lane_count * _F64_BYTES
    maximum_routes_per_tile = max(1, int(design.route_tile_size))
    maximum_fibers_per_tile = max(1, int(design.fiber_tile_size))
    temporary_tile_bytes_upper = (
        rank_count * maximum_routes_per_tile
        + maximum_routes_per_tile * lane_count
        + rank_count * maximum_fibers_per_tile * lane_count
        + context_count * maximum_fibers_per_tile * lane_count
    ) * _F64_BYTES
    registered = (
        persistent_factor_bytes
        + persistent_axis_bytes
        + max(result_cell_bytes, adjoint_result_bytes)
        + temporary_tile_bytes_upper
    )
    total = _nonnegative_int(
        device_total_memory_bytes,
        name="device total memory bytes",
    )
    free = _nonnegative_int(
        device_free_memory_bytes,
        name="device free memory bytes",
    )
    ceiling = math.floor(total * design.hbm_ceiling_fraction)
    available = free if _admission_available_bytes is None else _admission_available_bytes
    admitted = bool(registered <= ceiling and registered <= available)
    return V1SignedWorkOperatorCUDAHBMForecast(
        design_hash=design.design_hash,
        native_fiber_axis_receipt_hash=native_fiber_axis_receipt.receipt_hash,
        route_axis_hash=route_axis_hash,
        route_id_axis_hash=route_id_axis_hash,
        route_count=route_count,
        context_count=context_count,
        rank_count=rank_count,
        fiber_count=fiber_count,
        lane_count=lane_count,
        persistent_factor_bytes=int(persistent_factor_bytes),
        persistent_axis_bytes=int(persistent_axis_bytes),
        result_cell_bytes=int(result_cell_bytes),
        adjoint_result_bytes=int(adjoint_result_bytes),
        temporary_tile_bytes_upper=int(temporary_tile_bytes_upper),
        registered_device_bytes_upper=int(registered),
        device_total_memory_bytes=int(total),
        device_free_memory_bytes=int(free),
        admitted_under_85_percent_h200_style_ceiling=admitted,
        borrowed_row_bank_bytes=borrowed,
        logical_zero_residual_storage_bytes=(
            residual_storage_bytes if logical_zero_residual else 0
        ),
    )


class V1SignedWorkOperatorCUDA:
    """Factored native-fiber signed corrected-work forward/adjoint operator."""

    def __init__(
        self,
        *,
        left_context_factor: np.ndarray,
        right_route_factor: object,
        residual_by_route: object,
        context_weights: np.ndarray,
        native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
        route_axis_hash: str,
        route_id_axis_hash: str,
        design: V1SignedWorkOperatorCUDAExecutionDesign,
        device: object | None,
        source_geometry_hash: str,
        factorization_kind: str,
        right_route_factor_tensor: object | None = None,
        residual_by_route_tensor: object | None = None,
        context_weights_tensor: object | None = None,
        row_bank_lease: V1SignedWorkRowBankLease | None = None,
        row_spool_receipt_hash: str | None = None,
        borrowed_row_bank_bytes: int = 0,
        right_route_factor_digest_override: dict[str, object] | None = None,
        residual_linear_digest_override: dict[str, object] | None = None,
        logical_zero_residual: bool = False,
    ) -> None:
        if torch is None:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
                "PyTorch is required for signed-work operator device custody",
            )
        if type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "signed-work operator requires the exact native-fiber receipt",
            )
        native_fiber_axis_receipt.assert_integrity()
        if type(design) is not V1SignedWorkOperatorCUDAExecutionDesign:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "signed-work operator requires the exact CUDA execution design",
            )
        route_axis_hash = _identity(route_axis_hash, name="route axis hash")
        route_id_axis_hash = _identity(route_id_axis_hash, name="route-ID axis hash")
        if (
            route_axis_hash != native_fiber_axis_receipt.route_axis_hash
            or route_id_axis_hash != native_fiber_axis_receipt.route_id_axis_hash
        ):
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                "signed-work route axes differ from native-fiber receipt axes",
            )
        source_geometry_hash = _identity(
            source_geometry_hash,
            name="source geometry hash",
        )
        if factorization_kind not in _SIGNED_WORK_FACTORIZATION_KINDS:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "unsupported signed-work factorization kind",
            )
        context_count, rank_count = left_context_factor.shape
        if right_route_factor.shape != (rank_count, native_fiber_axis_receipt.route_count):
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                "right route factor shape is not rank×route",
            )
        residual_shape = (
            tuple(residual_by_route.shape)
            if residual_by_route_tensor is None
            else tuple(residual_by_route_tensor.shape)  # type: ignore[attr-defined]
        )
        if residual_shape != (native_fiber_axis_receipt.route_count,):
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                "residual route envelope shape is invalid",
            )
        if context_weights.shape != (context_count,):
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                "context weights are not aligned with the context factor",
            )
        context_mass = float(np.sum(context_weights, dtype=np.float64))
        if bool(np.any(context_weights < 0.0)) or context_mass <= 0.0:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.NUMERICAL_REFUSAL,
                "context weights must be nonnegative with positive total mass",
            )
        if not math.isclose(context_mass, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.NUMERICAL_REFUSAL,
                "context weights must sum to one for expectation-valued signed F",
            )
        device_obj = self._resolve_device(device, design=design)
        production_cuda = device_obj.type == "cuda"
        private_cpu = device_obj.type == "cpu"
        if production_cuda and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(device_obj)
            admission_available = cuda_reusable_memory_bytes(
                device_obj, framework=torch, free_bytes=int(free)
            )
        else:
            persistent = (
                context_count * rank_count
                + rank_count * native_fiber_axis_receipt.route_count
                + native_fiber_axis_receipt.route_count
                + context_count
            ) * _F64_BYTES
            total = max(1 << 30, persistent * 4)
            free = total
            admission_available = None
        forecast = build_v1_signed_work_operator_cuda_hbm_forecast(
            design=design,
            native_fiber_axis_receipt=native_fiber_axis_receipt,
            route_axis_hash=route_axis_hash,
            route_id_axis_hash=route_id_axis_hash,
            context_count=context_count,
            rank_count=rank_count,
            lane_count=design.maximum_lanes,
            device_total_memory_bytes=int(total),
            device_free_memory_bytes=int(free),
            borrowed_row_bank_bytes=borrowed_row_bank_bytes,
            logical_zero_residual=logical_zero_residual,
            _admission_available_bytes=admission_available,
        )
        if production_cuda and not forecast.admitted_under_85_percent_h200_style_ceiling:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.MEMORY_REFUSAL,
                "signed-work operator factor custody exceeds registered HBM forecast",
            )
        self._design = design
        self._axis = native_fiber_axis_receipt
        self._route_axis_hash = route_axis_hash
        self._route_id_axis_hash = route_id_axis_hash
        self._source_geometry_hash = source_geometry_hash
        self._factorization_kind = factorization_kind
        self._logical_zero_residual = bool(logical_zero_residual)
        self._row_bank_lease = row_bank_lease
        self._device_obj = device_obj
        self._device_identity = self._device_identity_for(device_obj)
        self._left = torch.tensor(
            np.ascontiguousarray(left_context_factor),
            dtype=torch.float64,
            device=device_obj,
        ).contiguous()
        if right_route_factor_tensor is None:
            self._right = torch.tensor(
                np.ascontiguousarray(right_route_factor),
                dtype=torch.float64,
                device=device_obj,
            ).contiguous()
        else:
            self._right = torch.as_tensor(
                right_route_factor_tensor,
                dtype=torch.float64,
                device=device_obj,
            )
            if (
                self._right.dtype != torch.float64
                or tuple(self._right.shape)
                != (rank_count, native_fiber_axis_receipt.route_count)
                or not self._right.is_contiguous()
                or self._right.requires_grad
            ):
                _refuse(
                    V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
                    "borrowed signed-work rows do not satisfy exact tensor custody",
                )
        if residual_by_route_tensor is None:
            self._residual = torch.tensor(
                np.ascontiguousarray(residual_by_route),
                dtype=torch.float64,
                device=device_obj,
            ).contiguous()
        else:
            self._residual = torch.as_tensor(
                residual_by_route_tensor,
                dtype=torch.float64,
                device=device_obj,
            )
            if (
                self._residual.dtype != torch.float64
                or tuple(self._residual.shape)
                != (native_fiber_axis_receipt.route_count,)
                or self._residual.requires_grad
            ):
                _refuse(
                    V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
                    "logical signed-work residual has invalid tensor custody",
                )
        if context_weights_tensor is None:
            self._context_weights = torch.tensor(
                np.ascontiguousarray(context_weights),
                dtype=torch.float64,
                device=device_obj,
            ).contiguous()
        else:
            self._context_weights = torch.as_tensor(
                context_weights_tensor,
                dtype=torch.float64,
                device=device_obj,
            ).contiguous()
        self._evicted = False
        self._last_telemetry: V1SignedWorkOperatorCUDATelemetry | None = None
        receipt = V1SignedWorkOperatorCUDAReceipt(
            source_geometry_hash=source_geometry_hash,
            factorization_kind=factorization_kind,
            design_hash=design.design_hash,
            forecast_hash=forecast.forecast_hash,
            native_fiber_axis_receipt_hash=native_fiber_axis_receipt.receipt_hash,
            native_fiber_measure_hash=native_fiber_axis_receipt.measure_hash,
            route_axis_hash=route_axis_hash,
            route_id_axis_hash=route_id_axis_hash,
            route_count=native_fiber_axis_receipt.route_count,
            context_count=context_count,
            rank_count=rank_count,
            fiber_count=native_fiber_axis_receipt.fiber_count,
            cut_count=native_fiber_axis_receipt.cut_count,
            context_weights_digest=_array_digest(context_weights),
            left_context_factor_digest=_array_digest(left_context_factor),
            right_route_factor_digest=(
                _array_digest(right_route_factor)
                if right_route_factor_digest_override is None
                else right_route_factor_digest_override
            ),
            residual_linear_digest=(
                _array_digest(residual_by_route)
                if residual_linear_digest_override is None
                else residual_linear_digest_override
            ),
            device_identity=self._device_identity,
            production_cuda_execution=production_cuda,
            private_cpu_reference_execution=private_cpu,
            row_spool_receipt_hash=row_spool_receipt_hash,
            row_spool_zero_copy_alias=row_bank_lease is not None,
            logical_zero_residual=self._logical_zero_residual,
            borrowed_row_bank_bytes=borrowed_row_bank_bytes,
        )
        self._forecast = forecast
        self._receipt = receipt

    @classmethod
    def from_exact_rows(
        cls,
        *,
        corrected_evidence_rows: object,
        context_weights: object,
        native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
        route_axis_hash: str,
        route_id_axis_hash: str,
        design: V1SignedWorkOperatorCUDAExecutionDesign = (
            V1SignedWorkOperatorCUDAExecutionDesign()
        ),
        device: object | None = None,
        source_geometry_hash: str = "unregistered-source-geometry",
    ) -> "V1SignedWorkOperatorCUDA":
        rows = _immutable_f64_matrix(
            corrected_evidence_rows,
            name="corrected evidence rows",
        )
        if type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "exact row construction requires the exact native-fiber receipt",
            )
        native_fiber_axis_receipt.assert_integrity()
        if rows.shape[1] != native_fiber_axis_receipt.route_count:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                "corrected evidence rows are not context×route for this axis",
            )
        context_count = int(rows.shape[0])
        weights = _immutable_f64_vector(
            context_weights,
            name="context weights",
            size=context_count,
            nonnegative=True,
        )
        left = np.eye(context_count, dtype=_LE_F64)
        residual = np.zeros(native_fiber_axis_receipt.route_count, dtype=_LE_F64)
        residual.setflags(write=False)
        return cls(
            left_context_factor=left,
            right_route_factor=rows,
            residual_by_route=residual,
            context_weights=weights,
            native_fiber_axis_receipt=native_fiber_axis_receipt,
            route_axis_hash=route_axis_hash,
            route_id_axis_hash=route_id_axis_hash,
            design=design,
            device=device,
            source_geometry_hash=source_geometry_hash,
            factorization_kind="exact_rows_identity",
        )

    @classmethod
    def from_row_bank_device_handle(
        cls,
        *,
        row_bank_handle: V1SignedWorkRowBankDeviceHandle,
        expected_custody_receipt_hash: str,
        expected_registered_d_prefix_hash: str,
        expected_chart_identity_hash: str,
        expected_raw_logit_stream_identity_hash: str,
        expected_native_fiber_axis_receipt_hash: str,
        expected_route_axis_hash: str,
        route_id_axis_hash: str,
        design: V1SignedWorkOperatorCUDAExecutionDesign = (
            V1SignedWorkOperatorCUDAExecutionDesign()
        ),
        device: object | None = None,
        source_geometry_hash: str | None = None,
    ) -> "V1SignedWorkOperatorCUDA":
        if type(row_bank_handle) is not V1SignedWorkRowBankDeviceHandle:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "row-bank construction requires exact signed-work row custody",
            )
        if type(design) is not V1SignedWorkOperatorCUDAExecutionDesign:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "row-bank construction requires the exact CUDA execution design",
            )
        lease = row_bank_handle.borrow_for_signed_work_operator(
            expected_custody_receipt_hash=expected_custody_receipt_hash,
            expected_registered_d_prefix_hash=expected_registered_d_prefix_hash,
            expected_chart_identity_hash=expected_chart_identity_hash,
            expected_raw_logit_stream_identity_hash=expected_raw_logit_stream_identity_hash,
            expected_native_fiber_axis_receipt_hash=(
                expected_native_fiber_axis_receipt_hash
            ),
            expected_route_axis_hash=expected_route_axis_hash,
        )
        try:
            receipt = lease.receipt
            native = lease.native_fiber_axis_receipt
            rows = lease.rows
            if rows.dtype != torch.float64 or not rows.is_contiguous() or rows.requires_grad:
                _refuse(
                    V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
                    "borrowed row bank is not exact contiguous FP64 custody",
                )
            requested = torch.device(rows.device if device is None else device)
            if requested != rows.device:
                _refuse(
                    V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
                    "row-bank operator device must equal the borrowed row device",
                )
            if route_id_axis_hash != native.route_id_axis_hash:
                _refuse(
                    V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                    "row-bank operator route-ID axis differs from native custody",
                )
            if design.production_cuda_required:
                if rows.device.type != "cuda" or not receipt.production_cuda_proposal_input_authority:
                    _refuse(
                        V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
                        "production row-bank operator requires CUDA-authoritative row custody",
                    )
            elif not (
                design.allow_private_cpu_reference
                and rows.device.type == "cpu"
                and receipt.cpu_reference_test_only
            ):
                _refuse(
                    V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
                    "non-production row-bank operator requires explicit CPU-reference custody",
                )
            context_count = int(rows.shape[0])
            weights = np.asarray(lease.context_weights, dtype=_LE_F64)
            left = np.eye(context_count, dtype=_LE_F64)
            zero_scalar = torch.zeros((), dtype=torch.float64, device=rows.device)
            residual = zero_scalar.expand(native.route_count)
            operator = cls(
                left_context_factor=left,
                right_route_factor=rows,
                residual_by_route=np.zeros(1, dtype=_LE_F64),
                context_weights=weights,
                native_fiber_axis_receipt=native,
                route_axis_hash=receipt.route_axis_hash,
                route_id_axis_hash=route_id_axis_hash,
                design=design,
                device=rows.device,
                source_geometry_hash=(
                    expected_chart_identity_hash
                    if source_geometry_hash is None
                    else source_geometry_hash
                ),
                factorization_kind="exact_rows_spool_alias",
                right_route_factor_tensor=rows,
                residual_by_route_tensor=residual,
                context_weights_tensor=torch.as_tensor(
                    weights,
                    dtype=torch.float64,
                    device=rows.device,
                ),
                row_bank_lease=lease,
                row_spool_receipt_hash=receipt.receipt_hash,
                borrowed_row_bank_bytes=receipt.row_bank_tensor_payload_bytes,
                right_route_factor_digest_override={
                    "source": "signed_work_row_spool_zero_copy_alias",
                    "row_spool_receipt_hash": receipt.receipt_hash,
                    "dtype": str(rows.dtype),
                    "shape": tuple(int(item) for item in rows.shape),
                    "bytes": receipt.row_bank_tensor_payload_bytes,
                    "host_payload_digest_materialized": False,
                },
                residual_linear_digest_override={
                    "source": "logical_zero_residual_scalar_expand",
                    "dtype": str(rows.dtype),
                    "shape": (native.route_count,),
                    "storage_bytes": _F64_BYTES,
                    "route_scale_zero_materialized": False,
                },
                logical_zero_residual=True,
            )
        except BaseException:
            lease.close()
            raise
        return operator

    @classmethod
    def from_factors(
        cls,
        *,
        left_context_factor: object,
        right_route_factor: object,
        residual_by_route: object,
        context_weights: object,
        native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
        route_axis_hash: str,
        route_id_axis_hash: str,
        design: V1SignedWorkOperatorCUDAExecutionDesign = (
            V1SignedWorkOperatorCUDAExecutionDesign()
        ),
        device: object | None = None,
        source_geometry_hash: str = "unregistered-source-geometry",
    ) -> "V1SignedWorkOperatorCUDA":
        left = _immutable_f64_matrix(
            left_context_factor,
            name="left context factor",
        )
        if type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "factor construction requires the exact native-fiber receipt",
            )
        native_fiber_axis_receipt.assert_integrity()
        rank_count = int(left.shape[1])
        right = _immutable_f64_matrix(
            right_route_factor,
            name="right route factor",
            shape=(rank_count, native_fiber_axis_receipt.route_count),
        )
        residual = _immutable_f64_vector(
            residual_by_route,
            name="residual linear envelope",
            size=native_fiber_axis_receipt.route_count,
            nonnegative=True,
        )
        weights = _immutable_f64_vector(
            context_weights,
            name="context weights",
            size=int(left.shape[0]),
            nonnegative=True,
        )
        return cls(
            left_context_factor=left,
            right_route_factor=right,
            residual_by_route=residual,
            context_weights=weights,
            native_fiber_axis_receipt=native_fiber_axis_receipt,
            route_axis_hash=route_axis_hash,
            route_id_axis_hash=route_id_axis_hash,
            design=design,
            device=device,
            source_geometry_hash=source_geometry_hash,
            factorization_kind="generic_factors",
        )

    @staticmethod
    def _resolve_device(
        device: object | None,
        *,
        design: V1SignedWorkOperatorCUDAExecutionDesign,
    ):
        if torch is None:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
                "PyTorch is not available",
            )
        if device is None:
            if design.production_cuda_required:
                if not torch.cuda.is_available():
                    _refuse(
                        V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
                        "production signed-work operator requires CUDA and has no "
                        "CPU fallback",
                    )
                return torch.device("cuda")
            if design.allow_private_cpu_reference:
                return torch.device("cpu")
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
                "non-production construction requires explicit private CPU "
                "reference authorization",
            )
        device_obj = torch.device(device)
        if device_obj.type == "cuda":
            if not torch.cuda.is_available():
                _refuse(
                    V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
                    "requested CUDA device is unavailable",
                )
            return device_obj
        if device_obj.type == "cpu":
            if design.production_cuda_required or not design.allow_private_cpu_reference:
                _refuse(
                    V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
                    "production signed-work operator refuses CPU fallback",
                )
            return device_obj
        _refuse(
            V1SignedWorkOperatorCUDARefusalCode.DEVICE_REFUSAL,
            f"unsupported signed-work operator device {device_obj.type!r}",
        )

    @staticmethod
    def _device_identity_for(device_obj: object) -> str:
        if torch is None:
            return "torch-unavailable"
        if getattr(device_obj, "type", None) == "cuda":
            index = 0 if device_obj.index is None else int(device_obj.index)
            return (
                f"torch_cuda:{index}:"
                f"{torch.cuda.get_device_name(index).replace(' ', '_')}"
            )
        return "private_cpu_reference"

    @property
    def receipt(self) -> V1SignedWorkOperatorCUDAReceipt:
        return self._receipt

    @property
    def receipt_hash(self) -> str:
        return self._receipt.receipt_hash

    @property
    def forecast(self) -> V1SignedWorkOperatorCUDAHBMForecast:
        return self._forecast

    @property
    def design(self) -> V1SignedWorkOperatorCUDAExecutionDesign:
        return self._design

    @property
    def device(self) -> str:
        return self._device_identity

    @property
    def native_fiber_axis_receipt(self) -> V1NativeDestinationFiberAxisReceipt:
        return self._axis

    @property
    def context_weights(self):
        self._assert_live()
        return self._context_weights

    @property
    def residual_linear(self):
        self._assert_live()
        return self._residual

    @property
    def residual_is_logical_zero(self) -> bool:
        return bool(getattr(self, "_logical_zero_residual", False))

    @property
    def last_telemetry(self) -> V1SignedWorkOperatorCUDATelemetry | None:
        return self._last_telemetry

    @property
    def route_count(self) -> int:
        return self._axis.route_count

    @property
    def context_count(self) -> int:
        return int(self._left.shape[0]) if not self._evicted else self._receipt.context_count

    @property
    def rank_count(self) -> int:
        return int(self._left.shape[1]) if not self._evicted else self._receipt.rank_count

    @property
    def fiber_count(self) -> int:
        return self._axis.fiber_count

    @property
    def operator_frobenius_norm_upper(self) -> float:
        """Return a source-bound upper bound for the linear operator norm.

        Native-fiber grouping only assigns each route column to one output
        row, so the grouped operator has the same Frobenius norm as the
        context-by-route matrix ``A``.  Exact rows therefore use ``||A||_F``;
        a generic factorization uses the certified submultiplicative bound
        ``||U||_F ||B||_F``.  This is evaluated once per chart, never inside
        the PDHG loop.
        """

        self._assert_live()
        if self._factorization_kind in _EXACT_ROW_FACTORIZATION_KINDS:
            value = torch.linalg.vector_norm(self._right)
        else:
            value = torch.linalg.vector_norm(self._left) * torch.linalg.vector_norm(
                self._right
            )
        result = float(value.item())
        if not math.isfinite(result):
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.NUMERICAL_REFUSAL,
                "signed-work Frobenius bound is not finite",
            )
        return result

    def _assert_live(self) -> None:
        if self._evicted:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.EVICTED,
                "signed-work operator device custody has been evicted",
            )

    def _canonical_route_lanes(self, value: object, *, name: str):
        self._assert_live()
        tensor = torch.as_tensor(value, dtype=torch.float64, device=self._device_obj)
        if tensor.ndim == 1:
            tensor = tensor.reshape(-1, 1)
        if tensor.ndim != 2 or int(tensor.shape[0]) != self.route_count:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                f"{name} must have shape route×lane",
            )
        if int(tensor.shape[1]) <= 0 or int(tensor.shape[1]) > self._design.maximum_lanes:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.MEMORY_REFUSAL,
                f"{name} lane count exceeds the registered design",
            )
        if not bool(torch.isfinite(tensor).all().item()):
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.NUMERICAL_REFUSAL,
                f"{name} contains nonfinite values",
            )
        return tensor.contiguous()

    def _canonical_dual_cells(self, value: object, *, name: str):
        self._assert_live()
        tensor = torch.as_tensor(value, dtype=torch.float64, device=self._device_obj)
        if tensor.ndim == 2:
            tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], 1)
        if (
            tensor.ndim != 3
            or int(tensor.shape[0]) != self.context_count
            or int(tensor.shape[1]) != self.fiber_count
        ):
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                f"{name} must have shape context×fiber×lane",
            )
        if int(tensor.shape[2]) <= 0 or int(tensor.shape[2]) > self._design.maximum_lanes:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.MEMORY_REFUSAL,
                f"{name} lane count exceeds the registered design",
            )
        if not bool(torch.isfinite(tensor).all().item()):
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.NUMERICAL_REFUSAL,
                f"{name} contains nonfinite values",
            )
        return tensor.contiguous()

    def _tile_bounds_for_cut(self, cut) -> tuple[int, int, int]:
        max_by_routes = max(1, self._design.route_tile_size // cut.routes_per_fiber)
        fibers_per_tile = max(1, min(self._design.fiber_tile_size, max_by_routes))
        tile_count = (cut.fiber_count + fibers_per_tile - 1) // fibers_per_tile
        return fibers_per_tile, tile_count, cut.routes_per_fiber

    def _iter_cut_fiber_tiles(self):
        for cut in self._axis.cuts:
            fibers_per_tile, _, routes_per_fiber = self._tile_bounds_for_cut(cut)
            for local_start in range(0, cut.fiber_count, fibers_per_tile):
                local_stop = min(cut.fiber_count, local_start + fibers_per_tile)
                fiber_start = cut.fiber_offset + local_start
                fiber_stop = cut.fiber_offset + local_stop
                route_start = cut.route_offset + local_start * routes_per_fiber
                route_stop = cut.route_offset + local_stop * routes_per_fiber
                yield (
                    cut,
                    local_start,
                    local_stop,
                    fiber_start,
                    fiber_stop,
                    route_start,
                    route_stop,
                    routes_per_fiber,
                )

    def _fiber_tile_count(self) -> int:
        return sum(self._tile_bounds_for_cut(cut)[1] for cut in self._axis.cuts)

    def _temporary_bytes_upper(self, lane_count: int) -> int:
        max_routes = 0
        max_fibers = 0
        for cut in self._axis.cuts:
            fibers_per_tile, _, routes_per_fiber = self._tile_bounds_for_cut(cut)
            max_fibers = max(max_fibers, fibers_per_tile)
            max_routes = max(max_routes, fibers_per_tile * routes_per_fiber)
        return int(
            (
                self.rank_count * max_routes
                + max_routes * lane_count
                + self.rank_count * max_fibers * lane_count
                + self.context_count * max_fibers * lane_count
            )
            * _F64_BYTES
        )

    def _forward_prevalidated(self, z, *, record_telemetry: bool = False):
        """Hot-loop forward for solver-owned finite route-major tensors.

        The enclosing solver validates dtype, shape, device, and finiteness at
        registered receipt boundaries.  Avoiding ``Tensor.item`` here is
        essential: a host synchronization on every PDHG application would
        dominate route-scale execution.
        """

        lane_count = int(z.shape[1])
        cells = torch.zeros(
            (self.context_count, self.fiber_count, lane_count),
            dtype=torch.float64,
            device=self._device_obj,
        )
        route_reads = 0
        factor_reads = 0
        for (
            _cut,
            _local_start,
            local_stop,
            fiber_start,
            fiber_stop,
            route_start,
            route_stop,
            routes_per_fiber,
        ) in self._iter_cut_fiber_tiles():
            tile_fibers = int(fiber_stop - fiber_start)
            z_tile = z[route_start:route_stop, :].reshape(
                tile_fibers,
                routes_per_fiber,
                lane_count,
            )
            right = self._right[:, route_start:route_stop].reshape(
                self.rank_count,
                tile_fibers,
                routes_per_fiber,
            )
            rank_fiber_lane = torch.einsum("rfs,fsl->rfl", right, z_tile)
            if self._factorization_kind in _EXACT_ROW_FACTORIZATION_KINDS:
                # ``left`` is an identity matrix in this branch.  Skipping the
                # otherwise redundant context×context contraction is material
                # for every route-scale solver iteration.
                cells[:, fiber_start:fiber_stop, :] = rank_fiber_lane
            else:
                cells[:, fiber_start:fiber_stop, :] = torch.einsum(
                    "cr,rfl->cfl",
                    self._left,
                    rank_fiber_lane,
                )
            route_reads += int(route_stop - route_start) * lane_count
            factor_reads += int(right.numel()) + int(self._left.numel())
            del right, z_tile, rank_fiber_lane
        if record_telemetry:
            self._last_telemetry = V1SignedWorkOperatorCUDATelemetry(
                receipt_hash=self.receipt_hash,
                operation="forward",
                device_identity=self.device,
                route_count=self.route_count,
                context_count=self.context_count,
                rank_count=self.rank_count,
                fiber_count=self.fiber_count,
                lane_count=lane_count,
                cut_count=self._axis.cut_count,
                fiber_tile_count=self._fiber_tile_count(),
                route_value_reads=route_reads,
                factor_value_reads_upper=factor_reads,
                temporary_tile_bytes_upper=self._temporary_bytes_upper(lane_count),
                result_bytes=int(cells.numel() * _F64_BYTES),
            )
        return cells

    def forward(self, z_columns: object):
        """Apply the native-fiber signed map, returning context×fiber×lane."""

        z = self._canonical_route_lanes(z_columns, name="z columns")
        return self._forward_prevalidated(z, record_telemetry=True)

    def _absolute_forward_envelope_prevalidated(self, z):
        """Bound the absolute primitive-term mass of every forward cell.

        This is the companion needed for outward numerical certification of a
        hard signed-work replay.  It preserves the same native-fiber tiling as
        :meth:`_forward_prevalidated`, but replaces each factor and route lane
        by its absolute value.  The result therefore bounds both the exact
        cell magnitude and the total magnitude against which floating-point
        contraction error must be charged.  It is never used as the pruning
        objective and does not erase signed cancellation from ``F`` itself.
        """

        lane_count = int(z.shape[1])
        cells = torch.zeros(
            (self.context_count, self.fiber_count, lane_count),
            dtype=torch.float64,
            device=self._device_obj,
        )
        z_abs = torch.abs(z)
        for (
            _cut,
            _local_start,
            local_stop,
            fiber_start,
            fiber_stop,
            route_start,
            route_stop,
            routes_per_fiber,
        ) in self._iter_cut_fiber_tiles():
            tile_fibers = int(fiber_stop - fiber_start)
            z_tile = z_abs[route_start:route_stop, :].reshape(
                tile_fibers,
                routes_per_fiber,
                lane_count,
            )
            right = torch.abs(self._right[:, route_start:route_stop]).reshape(
                self.rank_count,
                tile_fibers,
                routes_per_fiber,
            )
            rank_fiber_lane = torch.einsum("rfs,fsl->rfl", right, z_tile)
            if self._factorization_kind in _EXACT_ROW_FACTORIZATION_KINDS:
                cells[:, fiber_start:fiber_stop, :] = rank_fiber_lane
            else:
                cells[:, fiber_start:fiber_stop, :] = torch.einsum(
                    "cr,rfl->cfl",
                    torch.abs(self._left),
                    rank_fiber_lane,
                )
            del right, z_tile, rank_fiber_lane
        del z_abs
        return cells

    def absolute_forward_envelope(self, z_columns: object):
        """Return a device-resident absolute contraction envelope.

        The returned ``context×fiber×lane`` tensor is an error-analysis
        object only.  Canonical signed work continues to use
        ``abs(forward(z))`` so cancellation within one context/native fiber is
        retained exactly.
        """

        z = self._canonical_route_lanes(z_columns, name="z columns")
        return self._absolute_forward_envelope_prevalidated(z)

    def within_cut_route_permutation(self, *, seed: bytes | str):
        """Draw the fiber-shuffle null's structural permutation on device.

        Routes are permuted only inside their own cut (every cut has one
        ``routes_per_fiber``), which reassigns routes to fibers of equal size
        while preserving every route's own row, singleton magnitude, and cut
        membership.  The draw is a pure function of ``seed``; the returned
        ``int64`` route vector is the only temporary (one route-length int64).
        """

        self._assert_live()
        if isinstance(seed, str):
            seed = seed.encode("ascii")
        if type(seed) is not bytes or not seed:
            raise TypeError("permutation seed must be nonempty bytes or ascii text")
        digest = hashlib.sha256(seed).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:16], "big"))
        pieces = []
        for cut in self._axis.cuts:
            count = int(cut.fiber_count) * int(cut.routes_per_fiber)
            pieces.append(generator.permutation(count) + int(cut.route_offset))
        host = np.concatenate(pieces).astype(np.int64, copy=False)
        if host.shape != (self.route_count,):
            raise RuntimeError("cut structure does not cover the route axis")
        return torch.as_tensor(host, dtype=torch.int64, device=self._device_obj).contiguous()

    def _validate_route_permutation(self, route_permutation):
        permutation = torch.as_tensor(route_permutation, device=self._device_obj)
        if (
            permutation.dtype != torch.int64
            or permutation.ndim != 1
            or tuple(permutation.shape) != (self.route_count,)
            or not permutation.is_contiguous()
        ):
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                "route permutation must be one contiguous int64 route vector",
            )
        for cut in self._axis.cuts:
            start = int(cut.route_offset)
            stop = start + int(cut.fiber_count) * int(cut.routes_per_fiber)
            segment = torch.sort(permutation[start:stop]).values
            expected = torch.arange(start, stop, dtype=torch.int64, device=permutation.device)
            if not bool(torch.equal(segment, expected)):
                _refuse(
                    V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                    "route permutation must permute routes within their own cut",
                )
        return permutation

    def forward_from_first_inclusion_labels(
        self,
        labels: object,
        endpoint_labels: object,
        route_permutation: object | None = None,
    ):
        """Replay signed cells and their error envelope without route×lane z.

        ``labels[j]`` is zero for an unselected route and otherwise records the
        first frozen endpoint containing route ``j``.  Membership for a small
        endpoint batch is formed only inside each registered native-fiber tile;
        no full ``route×lane`` boolean or FP64 action matrix is materialized.
        The method is computational only.  The caller remains responsible for
        obtaining the labels through the typed device-shell replay lease and
        for binding the frozen-ladder identities in its receipt.

        ``route_permutation`` (optional, from ``within_cut_route_permutation``)
        reassigns routes to fibers inside each cut: position ``i`` of every
        tile carries route ``perm[i]``'s row and label.  Membership, singleton
        magnitudes, and per-cut counts are unchanged; only the fiber structure
        moves.  This is the fiber-shuffle null of the discriminating-physics
        preregistration and is never a certified quantity.
        """

        self._assert_live()
        permutation = (
            None
            if route_permutation is None
            else self._validate_route_permutation(route_permutation)
        )
        resident = torch.as_tensor(labels, device=self._device_obj)
        endpoints_source = torch.as_tensor(
            endpoint_labels, device=self._device_obj
        )
        if (
            resident.dtype != torch.uint8
            or resident.ndim != 1
            or tuple(resident.shape) != (self.route_count,)
            or not resident.is_contiguous()
        ):
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                "first-inclusion labels must be one contiguous uint8 route vector",
            )
        if endpoints_source.ndim != 1:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.AXIS_MISMATCH,
                "endpoint labels must be one vector",
            )
        if endpoints_source.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "endpoint labels must use an exact integer dtype",
            )
        endpoint_host = (
            endpoints_source.detach().cpu().numpy().astype(np.int64, copy=False)
        )
        lane_count = int(endpoint_host.size)
        if lane_count <= 0 or lane_count > self._design.maximum_lanes:
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.MEMORY_REFUSAL,
                "endpoint-label lane count exceeds the registered design",
            )
        if (
            bool(np.any(endpoint_host <= 0))
            or bool(np.any(endpoint_host > 255))
            or bool(np.any(endpoint_host[1:] <= endpoint_host[:-1]))
        ):
            _refuse(
                V1SignedWorkOperatorCUDARefusalCode.INVALID_TYPE,
                "endpoint labels must be strictly increasing positive uint8 values",
            )
        endpoints = torch.as_tensor(
            endpoint_host,
            dtype=torch.uint8,
            device=self._device_obj,
        ).contiguous()

        signed_cells = torch.zeros(
            (self.context_count, self.fiber_count, lane_count),
            dtype=torch.float64,
            device=self._device_obj,
        )
        absolute_envelope = torch.zeros_like(signed_cells)
        for (
            _cut,
            _local_start,
            local_stop,
            fiber_start,
            fiber_stop,
            route_start,
            route_stop,
            routes_per_fiber,
        ) in self._iter_cut_fiber_tiles():
            tile_fibers = int(fiber_stop - fiber_start)
            if permutation is None:
                label_tile = resident[route_start:route_stop].reshape(-1, 1)
                right_tile = self._right[:, route_start:route_stop]
            else:
                tile_index = permutation[route_start:route_stop]
                label_tile = resident[tile_index].reshape(-1, 1)
                right_tile = self._right[:, tile_index]
            membership = (
                (label_tile > 0) & (label_tile <= endpoints.reshape(1, -1))
            ).to(dtype=torch.float64)
            z_tile = membership.reshape(
                tile_fibers,
                routes_per_fiber,
                lane_count,
            )
            right = right_tile.reshape(
                self.rank_count,
                tile_fibers,
                routes_per_fiber,
            )
            signed_rank = torch.einsum("rfs,fsl->rfl", right, z_tile)
            absolute_rank = torch.einsum(
                "rfs,fsl->rfl", torch.abs(right), z_tile
            )
            if self._factorization_kind in _EXACT_ROW_FACTORIZATION_KINDS:
                signed_cells[:, fiber_start:fiber_stop, :] = signed_rank
                absolute_envelope[:, fiber_start:fiber_stop, :] = absolute_rank
            else:
                signed_cells[:, fiber_start:fiber_stop, :] = torch.einsum(
                    "cr,rfl->cfl", self._left, signed_rank
                )
                absolute_envelope[:, fiber_start:fiber_stop, :] = torch.einsum(
                    "cr,rfl->cfl", torch.abs(self._left), absolute_rank
                )
            del (
                label_tile,
                membership,
                z_tile,
                right,
                signed_rank,
                absolute_rank,
            )
        return signed_cells, absolute_envelope

    def _adjoint_prevalidated(self, dual, *, record_telemetry: bool = False):
        """Hot-loop transpose for solver-owned finite cell tensors."""

        lane_count = int(dual.shape[2])
        out = torch.zeros(
            (self.route_count, lane_count),
            dtype=torch.float64,
            device=self._device_obj,
        )
        route_writes = 0
        factor_reads = 0
        for (
            _cut,
            _local_start,
            _local_stop,
            fiber_start,
            fiber_stop,
            route_start,
            route_stop,
            routes_per_fiber,
        ) in self._iter_cut_fiber_tiles():
            tile_fibers = int(fiber_stop - fiber_start)
            dual_tile = dual[:, fiber_start:fiber_stop, :]
            if self._factorization_kind in _EXACT_ROW_FACTORIZATION_KINDS:
                rank_fiber_lane = dual_tile
            else:
                rank_fiber_lane = torch.einsum(
                    "cr,cfl->rfl", self._left, dual_tile
                )
            right = self._right[:, route_start:route_stop].reshape(
                self.rank_count,
                tile_fibers,
                routes_per_fiber,
            )
            out[route_start:route_stop, :] = torch.einsum(
                "rfs,rfl->fsl",
                right,
                rank_fiber_lane,
            ).reshape(tile_fibers * routes_per_fiber, lane_count)
            route_writes += int(route_stop - route_start) * lane_count
            factor_reads += int(right.numel()) + int(self._left.numel())
            del dual_tile, rank_fiber_lane, right
        if record_telemetry:
            self._last_telemetry = V1SignedWorkOperatorCUDATelemetry(
                receipt_hash=self.receipt_hash,
                operation="adjoint",
                device_identity=self.device,
                route_count=self.route_count,
                context_count=self.context_count,
                rank_count=self.rank_count,
                fiber_count=self.fiber_count,
                lane_count=lane_count,
                cut_count=self._axis.cut_count,
                fiber_tile_count=self._fiber_tile_count(),
                route_value_reads=route_writes,
                factor_value_reads_upper=factor_reads,
                temporary_tile_bytes_upper=self._temporary_bytes_upper(lane_count),
                result_bytes=int(out.numel() * _F64_BYTES),
            )
        return out

    def adjoint(self, dual_cells: object):
        """Apply the exact transpose of :meth:`forward`, returning route×lane."""

        dual = self._canonical_dual_cells(dual_cells, name="dual cells")
        return self._adjoint_prevalidated(dual, record_telemetry=True)

    def residual_linear_value(self, z_columns: object):
        """Return ``epsilon^T z`` lane-wise for the residual envelope."""

        z = self._canonical_route_lanes(z_columns, name="z columns")
        if self.residual_is_logical_zero:
            return torch.zeros(
                int(z.shape[1]),
                dtype=torch.float64,
                device=self._device_obj,
            )
        return torch.matmul(self._residual.reshape(1, -1), z).reshape(-1)

    def evict(self) -> V1SignedWorkOperatorCUDATelemetry:
        """Release private tensor custody and make further operator use illegal."""

        if self._evicted:
            return V1SignedWorkOperatorCUDATelemetry(
                receipt_hash=self.receipt_hash,
                operation="evict",
                device_identity=self.device,
                route_count=self._receipt.route_count,
                context_count=self._receipt.context_count,
                rank_count=self._receipt.rank_count,
                fiber_count=self._receipt.fiber_count,
                lane_count=0 + 1,
                cut_count=self._receipt.cut_count,
                fiber_tile_count=0,
                route_value_reads=0,
                factor_value_reads_upper=0,
                temporary_tile_bytes_upper=0,
                result_bytes=0,
                evicted_after_operation=True,
            )
        del self._left
        del self._right
        del self._residual
        del self._context_weights
        lease = getattr(self, "_row_bank_lease", None)
        if lease is not None:
            lease.close()
            self._row_bank_lease = None
        self._evicted = True
        if torch is not None and self._device_obj.type == "cuda":
            torch.cuda.empty_cache()
        telemetry = V1SignedWorkOperatorCUDATelemetry(
            receipt_hash=self.receipt_hash,
            operation="evict",
            device_identity=self.device,
            route_count=self._receipt.route_count,
            context_count=self._receipt.context_count,
            rank_count=self._receipt.rank_count,
            fiber_count=self._receipt.fiber_count,
            lane_count=1,
            cut_count=self._receipt.cut_count,
            fiber_tile_count=0,
            route_value_reads=0,
            factor_value_reads_upper=0,
            temporary_tile_bytes_upper=0,
            result_bytes=0,
            evicted_after_operation=True,
        )
        self._last_telemetry = telemetry
        return telemetry


__all__ = [
    "SIGNED_WORK_OPERATOR_CUDA_ALGORITHM",
    "SIGNED_WORK_OPERATOR_CUDA_AUTHORITY",
    "SIGNED_WORK_OPERATOR_CUDA_DESIGN_SCHEMA",
    "SIGNED_WORK_OPERATOR_CUDA_HBM_FORECAST_SCHEMA",
    "SIGNED_WORK_OPERATOR_CUDA_MEMORY_SCOPE",
    "SIGNED_WORK_OPERATOR_CUDA_RECEIPT_SCHEMA",
    "SIGNED_WORK_OPERATOR_CUDA_TELEMETRY_SCHEMA",
    "V1SignedWorkOperatorCUDA",
    "V1SignedWorkOperatorCUDAExecutionDesign",
    "V1SignedWorkOperatorCUDAHBMForecast",
    "V1SignedWorkOperatorCUDAReceipt",
    "V1SignedWorkOperatorCUDARefusal",
    "V1SignedWorkOperatorCUDARefusalCode",
    "V1SignedWorkOperatorCUDATelemetry",
    "build_v1_signed_work_operator_cuda_hbm_forecast",
    "signed_work_operator_cuda_source_hash",
]
