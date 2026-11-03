"""Source-owned analytic literal-J supports for canonical AlexNet.

For a current relaxed-gate chart, the literal response Jacobian of route
``j`` is ``-G 1[a > 0] c_j``.  This module supplies a pre-outcome absolute
support for every AlexNet route by combining

* current-mask interval forward bounds ``X``;
* the possible-active ReLU indicator ``I = 1[U_a > 0]``;
* a reverse absolute cotangent envelope ``C``; and
* exact convolutional spatial incidence or linear contraction.

For the two frozen KKT private primals it also evaluates the strictly tighter
fractional-action bound

``sum_cut <C I, abs(interval_drive(W * m * z))>``.

The latter performs the signed incoming-route contraction before magnitude,
so within-destination cancellation is retained.  The generic residual CUDA
owner later takes the elementwise minimum of this direct bound and the
independent route-sum bound.  Neither bound grants proposal, C, oracle,
commit, terminal, or Pareto authority.

Production issuance is deliberately narrow.  It requires exact source,
model-load, evaluator, query, current-mask, declared-route, native-cut,
registered-input-box, population-chart, and private-primal custody.  Callers
cannot provide a bound tensor or any of the durable source/derivation hashes.
The private CPU entry point is limited to small exhaustive tests and cannot
mint generic production custody.
"""

from __future__ import annotations

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from dataclasses import InitVar, dataclass, field
from enum import Enum
import hashlib
import inspect
import math
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Mapping, NoReturn

import numpy as np

from mechanistic_systems.circuits.torch_evaluator import TorchCircuitEvaluator
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.contracts.science import QuerySpec
from mechanistic_systems.measurement.models import ModelLoadReceipt, state_dict_digest

from . import modern_v1_alexnet_envelope as _envelope
from .modern_v1_alexnet import V1AlexNetBackendAdmission
from .modern_v1_alexnet_native_fiber import (
    build_alexnet_v1_native_destination_fiber_axis,
)
from .modern_v1_literal_j_d_fixed_residual_population_projection import (
    V1LiteralJDFixedResidualPopulationProjectionRegistration,
)
from .modern_v1_literal_j_projection_cuda import (
    V1LiteralJPrivatePrimalDirectionCUDAHandle,
)
from .modern_v1_literal_j_residual_population_projection_cuda import (
    V1LiteralJResidualAnalyticSupportSourceCUDAResult,
    _ANALYTIC_SUPPORT_SOURCE_OWNER_TOKEN,
    _issue_v1_literal_j_residual_analytic_support_source_cuda_from_owner,
)
from .modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt
from .modern_v1_neutral_runtime import DeclaredRouteAxis

if TYPE_CHECKING:
    from .modern_v1_alexnet_backend import V1AlexNetFQAcquisitionPlan

try:  # pragma: no cover - runtime image dependent.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


ALEXNET_LITERAL_J_SUPPORT_DERIVATION_SCHEMA: Final = (
    "partition_free_v1_alexnet_literal_j_analytic_support_derivation_v1"
)
ALEXNET_LITERAL_J_SUPPORT_SPAN_SCHEMA: Final = (
    "partition_free_v1_alexnet_literal_j_analytic_support_span_v1"
)
ALEXNET_LITERAL_J_SUPPORT_HBM_FORECAST_SCHEMA: Final = (
    "partition_free_v1_alexnet_literal_j_analytic_support_hbm_forecast_v1"
)
ALEXNET_LITERAL_J_SUPPORT_RESULT_SCHEMA: Final = (
    "partition_free_v1_alexnet_literal_j_analytic_support_result_v1"
)
ALEXNET_LITERAL_J_SUPPORT_ALGORITHM: Final = (
    "current_gated_box_forward_possible_active_relu_reverse_cotangent_"
    "exact_spatial_incidence_and_fractional_private_primal_drive_v1"
)
ALEXNET_LITERAL_J_SUPPORT_AUTHORITY: Final = (
    "pre_outcome_literal_j_support_source_only_no_rank_kkt_face_C_K_O_commit_"
    "terminal_or_pareto_authority_v1"
)
ALEXNET_LITERAL_J_DIRECT_BOUND_EQUATION: Final = (
    "sum_cut_inner_product_C_times_possible_active_I_with_absolute_interval_"
    "drive_of_W_times_current_mask_times_private_primal_v1"
)
ALEXNET_LITERAL_J_ROUTE_BOUND_EQUATION: Final = (
    "current_mask_times_abs_weight_times_valid_spatial_sum_of_C_times_"
    "possible_active_I_times_source_absolute_box_bound_v1"
)
ALEXNET_LITERAL_J_RELU_KINK_POLICY: Final = (
    "literal_relaxed_gate_jacobian_uses_indicator_preactivation_strictly_"
    "greater_than_zero_v1"
)

_PREFIX: Final = "partition-free-v1-alexnet-literal-j-support-"
_F64_BYTES: Final = 8
_BOOL_BYTES: Final = 1
_CPU_REFERENCE_ROUTE_LIMIT: Final = 4096
_DEFAULT_HBM_CEILING_FRACTION: Final = 0.85
_DEFAULT_AFFINE_OUTPUT_ROW_CHUNK: Final = 256
_DEFAULT_ROUTE_OUTPUT_ROW_CHUNK: Final = 256
_RECEIPT_TOKEN: Final = object()
_FORECAST_TOKEN: Final = object()

_LAYER_IDS: Final = (
    "features.stage1.conv",
    "features.stage2.conv",
    "features.stage3.conv",
    "features.stage4.conv",
    "features.stage5.conv",
    "readout.hidden1",
    "readout.hidden2",
)
_AFFINE_PATHS: Final = (
    "features.0",
    "features.3",
    "features.6",
    "features.8",
    "features.10",
    "classifier.1",
    "classifier.4",
)
_CANONICAL_SPANS: Final = (
    (
        "features.stage1.conv",
        "features.0",
        "conv_kernel_weight",
        (64, 3, 11, 11),
        0,
        23_232,
        363,
    ),
    (
        "features.stage2.conv",
        "features.3",
        "conv_kernel_weight",
        (192, 64, 5, 5),
        23_232,
        330_432,
        1_600,
    ),
    (
        "features.stage3.conv",
        "features.6",
        "conv_kernel_weight",
        (384, 192, 3, 3),
        330_432,
        993_984,
        1_728,
    ),
    (
        "features.stage4.conv",
        "features.8",
        "conv_kernel_weight",
        (256, 384, 3, 3),
        993_984,
        1_878_720,
        3_456,
    ),
    (
        "features.stage5.conv",
        "features.10",
        "conv_kernel_weight",
        (256, 256, 3, 3),
        1_878_720,
        2_468_544,
        2_304,
    ),
    (
        "readout.hidden1",
        "classifier.1",
        "linear_weight",
        (4_096, 9_216),
        2_468_544,
        40_217_280,
        9_216,
    ),
    (
        "readout.hidden2",
        "classifier.4",
        "linear_weight",
        (4_096, 4_096),
        40_217_280,
        56_994_496,
        4_096,
    ),
)

_DEPENDENCY_FUNCTIONS: Final = (
    _envelope.admit_v1_alexnet_envelope_layout,
    _envelope._forward_bounds,
    _envelope._conv_transpose_absolute_bound,
    _envelope._linear_transpose_absolute_bound,
    _envelope._reverse_maxpool_absolute_bound,
    _envelope._reverse_adaptive_avg_absolute_bound,
    _envelope._outward_interval,
    _envelope._rounding_factor,
    _envelope._direct_hard_mask_hash,
    _envelope._outward_reverse_absolute_bound,
    _envelope._execution_underflow_radius,
)
_EXPECTED_DEPENDENCY_FUNCTIONS: Final = tuple(_DEPENDENCY_FUNCTIONS)


class V1AlexNetLiteralJSupportRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    SOURCE_IDENTITY_MISMATCH = "source_identity_mismatch"
    MODEL_STATE_MISMATCH = "model_state_mismatch"
    EVALUATOR_MISMATCH = "evaluator_mismatch"
    QUERY_MISMATCH = "query_mismatch"
    CHART_MISMATCH = "chart_mismatch"
    MASK_MISMATCH = "mask_mismatch"
    ROUTE_AXIS_MISMATCH = "route_axis_mismatch"
    NATIVE_CUT_MISMATCH = "native_cut_mismatch"
    INPUT_BOX_MISMATCH = "input_box_mismatch"
    ACQUISITION_PLAN_MISMATCH = "acquisition_plan_mismatch"
    DIRECTION_AXIS_MISMATCH = "direction_axis_mismatch"
    DEVICE_REQUIRED = "device_required"
    DETERMINISM_DISABLED = "determinism_disabled"
    NUMERICAL_REFUSAL = "numerical_refusal"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    SOURCE_MUTATED = "source_mutated"
    AUTHORITY_VIOLATION = "authority_violation"


class V1AlexNetLiteralJSupportRefusal(RuntimeError):
    """Typed fail-closed refusal from the AlexNet analytic support owner."""

    def __init__(
        self,
        code: V1AlexNetLiteralJSupportRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1AlexNetLiteralJSupportRefusalCode:
            raise TypeError("AlexNet literal-J refusal code must be exact")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1AlexNetLiteralJSupportRefusalCode,
    message: str,
) -> NoReturn:
    raise V1AlexNetLiteralJSupportRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty exact string",
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return int(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.INVALID_TYPE,
            f"{name} must be one nonnegative exact integer",
        )
    return int(value)


def _finite_fraction(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 < value < 1.0:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.INVALID_TYPE,
            f"{name} must be one exact finite float in (0, 1)",
        )
    return value


def _payload_hash(payload: Mapping[str, object], *, role: str) -> str:
    return content_hash(payload, prefix=_PREFIX + role + "-")


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _callable_source_sha256(value: object) -> str:
    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as error:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.SOURCE_IDENTITY_MISMATCH,
            f"registered dependency source is unavailable: {error}",
        )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _assert_dependency_owners() -> None:
    current = (
        _envelope.admit_v1_alexnet_envelope_layout,
        _envelope._forward_bounds,
        _envelope._conv_transpose_absolute_bound,
        _envelope._linear_transpose_absolute_bound,
        _envelope._reverse_maxpool_absolute_bound,
        _envelope._reverse_adaptive_avg_absolute_bound,
        _envelope._outward_interval,
        _envelope._rounding_factor,
        _envelope._direct_hard_mask_hash,
        _envelope._outward_reverse_absolute_bound,
        _envelope._execution_underflow_radius,
    )
    if any(
        left is not right
        for left, right in zip(current, _EXPECTED_DEPENDENCY_FUNCTIONS, strict=True)
    ):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "an in-process analytic-bound dependency was replaced",
        )


def v1_alexnet_literal_j_support_owner_identity_hash() -> str:
    """Content-address the exact deterministic analytic-source implementation."""

    _assert_dependency_owners()
    return _payload_hash(
        {
            "schema": "alexnet_literal_j_support_owner_identity_v1",
            "owner_source_sha256": _source_sha256(),
            "dependency_source_sha256s": tuple(
                _callable_source_sha256(item) for item in _EXPECTED_DEPENDENCY_FUNCTIONS
            ),
            "algorithm": ALEXNET_LITERAL_J_SUPPORT_ALGORITHM,
            "route_equation": ALEXNET_LITERAL_J_ROUTE_BOUND_EQUATION,
            "direct_equation": ALEXNET_LITERAL_J_DIRECT_BOUND_EQUATION,
            "relu_kink_policy": ALEXNET_LITERAL_J_RELU_KINK_POLICY,
            "canonical_spans": _CANONICAL_SPANS,
        },
        role="owner-identity",
    )


def _up(value: Any) -> Any:
    return torch.nextafter(value, torch.full_like(value, math.inf))


def _require_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    device: Any | None = None,
    nonnegative: bool = False,
) -> Any:
    if torch is None or type(value) is not torch.Tensor:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.INVALID_TYPE,
            f"{name} must be one exact torch.Tensor",
        )
    if (
        value.dtype != torch.float64
        or tuple(int(item) for item in value.shape) != shape
        or not value.is_contiguous()
        or value.requires_grad
        or (device is not None and value.device != device)
    ):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.DEVICE_REQUIRED,
            f"{name} left its registered contiguous FP64 device/shape",
        )
    if not bool(torch.isfinite(value).all().item()):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
            f"{name} contains a nonfinite value",
        )
    if nonnegative and bool((value < 0.0).any().item()):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
            f"{name} contains a negative bound",
        )
    return value


def _direct_mask_hash(mask: HardMask) -> str:
    try:
        return _envelope._direct_hard_mask_hash(
            mask,
            name="AlexNet literal-J current mask",
        )
    except TypeError as error:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.INVALID_TYPE,
            str(error),
        )
    except ValueError as error:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.MASK_MISMATCH,
            str(error),
        )


@dataclass(frozen=True, slots=True)
class V1AlexNetLiteralJSupportSpanReceipt:
    layer_id: str
    module_path: str
    route_kind: str
    route_shape: tuple[int, ...]
    route_offset: int
    route_stop: int
    routes_per_native_fiber: int
    possible_active_unit_count: int
    destination_unit_count: int
    live_route_count: int
    source_absolute_max: float
    cotangent_absolute_max: float
    route_bound_max: float
    route_rounding_factor: float
    schema: str = ALEXNET_LITERAL_J_SUPPORT_SPAN_SCHEMA

    def __post_init__(self) -> None:
        for name in ("layer_id", "module_path", "route_kind"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        shape = tuple(
            _positive_int(int(item), name="route shape extent")
            for item in self.route_shape
        )
        object.__setattr__(self, "route_shape", shape)
        for name in (
            "route_offset",
            "possible_active_unit_count",
            "live_route_count",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name=name)
            )
        for name in (
            "route_stop",
            "routes_per_native_fiber",
            "destination_unit_count",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        if (
            self.route_stop - self.route_offset != math.prod(shape)
            or self.routes_per_native_fiber != math.prod(shape[1:])
            or self.possible_active_unit_count > self.destination_unit_count
            or self.live_route_count > math.prod(shape)
        ):
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.ROUTE_AXIS_MISMATCH,
                "literal-J span geometry is inconsistent",
            )
        for name in (
            "source_absolute_max",
            "cotangent_absolute_max",
            "route_bound_max",
            "route_rounding_factor",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                _refuse(
                    V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
                    f"{name} must be finite and nonnegative",
                )
            object.__setattr__(self, name, value)
        if self.schema != ALEXNET_LITERAL_J_SUPPORT_SPAN_SCHEMA:
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.AUTHORITY_VIOLATION,
                "unsupported AlexNet literal-J span schema",
            )


@dataclass(frozen=True, slots=True)
class V1AlexNetLiteralJSupportHBMForecast:
    route_count: int
    direction_count: int
    largest_route_block_count: int
    retained_interval_state_bytes: int
    retained_cotangent_and_activity_bytes: int
    largest_affine_workspace_bytes: int
    owner_route_vector_bytes: int
    owner_direct_vector_bytes: int
    generic_handoff_vector_and_guard_bytes: int
    derivation_incremental_peak_bytes: int
    handoff_incremental_peak_bytes: int
    projected_incremental_peak_bytes: int
    device_identity: str
    current_allocated_device_bytes: int
    current_reserved_device_bytes: int
    device_free_memory_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_fraction: float
    hbm_ceiling_bytes: int
    projected_total_reserved_peak_bytes: int
    admitted: bool
    source_vector_and_guard_transfer_counted: bool = True
    external_model_and_private_primal_custody_excluded: bool = True
    schema: str = ALEXNET_LITERAL_J_SUPPORT_HBM_FORECAST_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _forecast_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _FORECAST_TOKEN:
            raise PermissionError(
                "AlexNet literal-J HBM forecasts require their issuer"
            )
        for name in (
            "route_count",
            "direction_count",
            "largest_route_block_count",
            "owner_route_vector_bytes",
            "owner_direct_vector_bytes",
            "generic_handoff_vector_and_guard_bytes",
            "device_total_memory_bytes",
            "hbm_ceiling_bytes",
        ):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name=name)
            )
        for name in (
            "retained_interval_state_bytes",
            "retained_cotangent_and_activity_bytes",
            "largest_affine_workspace_bytes",
            "derivation_incremental_peak_bytes",
            "handoff_incremental_peak_bytes",
            "projected_incremental_peak_bytes",
            "current_allocated_device_bytes",
            "current_reserved_device_bytes",
            "device_free_memory_bytes",
            "projected_total_reserved_peak_bytes",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name=name)
            )
        fraction = _finite_fraction(
            self.hbm_ceiling_fraction, name="hbm_ceiling_fraction"
        )
        expected_derivation = (
            self.retained_interval_state_bytes
            + self.retained_cotangent_and_activity_bytes
            + self.largest_affine_workspace_bytes
            + self.owner_route_vector_bytes
            + self.owner_direct_vector_bytes
        )
        expected_handoff = (
            self.owner_route_vector_bytes
            + self.owner_direct_vector_bytes
            + self.generic_handoff_vector_and_guard_bytes
        )
        expected_peak = max(expected_derivation, expected_handoff)
        expected_ceiling = math.floor(fraction * self.device_total_memory_bytes)
        expected_reserved = self.current_reserved_device_bytes + expected_peak
        expected_admitted = (
            expected_peak <= self.device_free_memory_bytes + max(0, self.current_reserved_device_bytes - self.current_allocated_device_bytes)
            and self.current_allocated_device_bytes + expected_peak <= expected_ceiling
        )
        if (
            self.derivation_incremental_peak_bytes != expected_derivation
            or self.handoff_incremental_peak_bytes != expected_handoff
            or self.projected_incremental_peak_bytes != expected_peak
            or self.hbm_ceiling_bytes != expected_ceiling
            or self.projected_total_reserved_peak_bytes != expected_reserved
            or self.admitted is not expected_admitted
            or self.current_allocated_device_bytes > self.current_reserved_device_bytes
            or self.current_reserved_device_bytes > self.device_total_memory_bytes
            or self.device_free_memory_bytes > self.device_total_memory_bytes
            or self.source_vector_and_guard_transfer_counted is not True
            or self.external_model_and_private_primal_custody_excluded is not True
            or self.schema != ALEXNET_LITERAL_J_SUPPORT_HBM_FORECAST_SCHEMA
        ):
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.HBM_FORECAST_EXCEEDED,
                "AlexNet literal-J HBM forecast arithmetic or closed flags differ",
            )
        object.__setattr__(self, "hbm_ceiling_fraction", fraction)
        object.__setattr__(
            self,
            "device_identity",
            _identity(self.device_identity, name="device_identity"),
        )
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    @property
    def forecast_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(self._payload(), role="hbm-forecast")
        if self._forecast_hash is not None and self._forecast_hash != observed:
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.SOURCE_MUTATED,
                "AlexNet literal-J HBM forecast mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.SOURCE_MUTATED,
                "AlexNet literal-J HBM forecast hash differs",
            )
        object.__setattr__(self, "_forecast_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1AlexNetLiteralJSupportDerivationReceipt:
    population_projection_registration_hash: str
    fq_acquisition_plan_hash: str
    master_alpha_registration_hash: str
    campaign_contract_hash: str
    controller_run_hash: str
    train_dataset_identity_hash: str
    train_split_identity_hash: str
    source_role_registration_hash: str
    preprocessing_hash: str
    acquisition_alpha_closure_hash: str
    backend_admission_hash: str
    model_load_receipt_hash: str
    model_identity_hash: str
    model_state_hash: str
    evaluator_id: str
    evaluator_parity_receipt_hash: str
    evaluator_source_identity_hash: str
    query_contract_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    route_axis_content_hash: str
    route_count: int
    native_fiber_axis_receipt_hash: str
    input_box_hash: str
    population_alpha_ledger_hash: str
    population_alpha_coordinate_hash: str
    population_alpha_lineage_hash: str
    fq_generation_alpha_allocation: float
    population_familywise_alpha: float
    private_primal_direction_axis_hash: str
    direction_ids: tuple[str, str]
    source_owner_identity_hash: str
    backend_execution_receipt_hash: str
    literal_j_abs_bound_source_hash: str
    direct_projection_abs_bound_derivation_receipt_hash: str
    hbm_forecast_hash: str
    span_receipts: tuple[V1AlexNetLiteralJSupportSpanReceipt, ...]
    route_sum_projection_abs_bounds: tuple[float, float]
    direct_projection_abs_bounds: tuple[float, float]
    selected_projection_abs_bounds: tuple[float, float]
    compact_tensor_fingerprint_is_content_authority: bool = False
    exact_inputs_and_deterministic_implementation_are_content_authority: bool = True
    current_gated_forward_bounds: bool = True
    possible_active_relu_masks: bool = True
    literal_jacobian_not_corrected_attribution: bool = True
    direct_bound_preserves_within_destination_signed_cancellation: bool = True
    no_route_scale_d2h_array: bool = True
    proposal_authority: bool = False
    c_stream_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    algorithm: str = ALEXNET_LITERAL_J_SUPPORT_ALGORITHM
    route_bound_equation: str = ALEXNET_LITERAL_J_ROUTE_BOUND_EQUATION
    direct_bound_equation: str = ALEXNET_LITERAL_J_DIRECT_BOUND_EQUATION
    relu_kink_policy: str = ALEXNET_LITERAL_J_RELU_KINK_POLICY
    authority_scope: str = ALEXNET_LITERAL_J_SUPPORT_AUTHORITY
    schema: str = ALEXNET_LITERAL_J_SUPPORT_DERIVATION_SCHEMA
    _issuance_token: InitVar[object | None] = None
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self, _issuance_token: object | None) -> None:
        if _issuance_token is not _RECEIPT_TOKEN:
            raise PermissionError(
                "AlexNet literal-J receipts require their source owner"
            )
        for name in (
            "population_projection_registration_hash",
            "fq_acquisition_plan_hash",
            "master_alpha_registration_hash",
            "campaign_contract_hash",
            "controller_run_hash",
            "train_dataset_identity_hash",
            "train_split_identity_hash",
            "source_role_registration_hash",
            "preprocessing_hash",
            "acquisition_alpha_closure_hash",
            "backend_admission_hash",
            "model_load_receipt_hash",
            "model_identity_hash",
            "model_state_hash",
            "evaluator_id",
            "evaluator_parity_receipt_hash",
            "evaluator_source_identity_hash",
            "query_contract_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "route_axis_content_hash",
            "native_fiber_axis_receipt_hash",
            "input_box_hash",
            "population_alpha_ledger_hash",
            "population_alpha_coordinate_hash",
            "population_alpha_lineage_hash",
            "private_primal_direction_axis_hash",
            "source_owner_identity_hash",
            "backend_execution_receipt_hash",
            "literal_j_abs_bound_source_hash",
            "direct_projection_abs_bound_derivation_receipt_hash",
            "hbm_forecast_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal_generation"),
        )
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route_count")
        )
        for name in (
            "fq_generation_alpha_allocation",
            "population_familywise_alpha",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                _refuse(
                    V1AlexNetLiteralJSupportRefusalCode.ACQUISITION_PLAN_MISMATCH,
                    f"{name} must be a finite probability in (0, 1)",
                )
            object.__setattr__(self, name, value)
        alpha_tolerance = 32.0 * math.ulp(
            max(
                1.0,
                self.population_familywise_alpha,
                self.fq_generation_alpha_allocation,
            )
        )
        if (
            self.population_familywise_alpha
            > self.fq_generation_alpha_allocation + alpha_tolerance
        ):
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.ACQUISITION_PLAN_MISMATCH,
                "population alpha exceeds its registered generation allocation",
            )
        directions = tuple(
            _identity(item, name="direction_id") for item in self.direction_ids
        )
        if len(directions) != 2 or directions[0] == directions[1]:
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.DIRECTION_AXIS_MISMATCH,
                "AlexNet support receipt requires two distinct directions",
            )
        object.__setattr__(self, "direction_ids", directions)
        spans = tuple(self.span_receipts)
        if len(spans) != 7 or any(
            type(item) is not V1AlexNetLiteralJSupportSpanReceipt for item in spans
        ):
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.NATIVE_CUT_MISMATCH,
                "AlexNet literal-J derivation must cover seven exact spans",
            )
        object.__setattr__(self, "span_receipts", spans)
        for name in (
            "route_sum_projection_abs_bounds",
            "direct_projection_abs_bounds",
            "selected_projection_abs_bounds",
        ):
            values = tuple(float(item) for item in getattr(self, name))
            if len(values) != 2 or any(
                not math.isfinite(item) or item < 0.0 for item in values
            ):
                _refuse(
                    V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
                    f"{name} must contain two finite nonnegative bounds",
                )
            object.__setattr__(self, name, values)
        flags = (
            self.compact_tensor_fingerprint_is_content_authority,
            self.exact_inputs_and_deterministic_implementation_are_content_authority,
            self.current_gated_forward_bounds,
            self.possible_active_relu_masks,
            self.literal_jacobian_not_corrected_attribution,
            self.direct_bound_preserves_within_destination_signed_cancellation,
            self.no_route_scale_d2h_array,
            self.proposal_authority,
            self.c_stream_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(item) is not bool for item in flags):
            raise TypeError("AlexNet literal-J receipt flags must be exact booleans")
        if (
            self.compact_tensor_fingerprint_is_content_authority
            or not self.exact_inputs_and_deterministic_implementation_are_content_authority
            or not self.current_gated_forward_bounds
            or not self.possible_active_relu_masks
            or not self.literal_jacobian_not_corrected_attribution
            or not self.direct_bound_preserves_within_destination_signed_cancellation
            or not self.no_route_scale_d2h_array
            or any(flags[7:])
            or self.algorithm != ALEXNET_LITERAL_J_SUPPORT_ALGORITHM
            or self.route_bound_equation != ALEXNET_LITERAL_J_ROUTE_BOUND_EQUATION
            or self.direct_bound_equation != ALEXNET_LITERAL_J_DIRECT_BOUND_EQUATION
            or self.relu_kink_policy != ALEXNET_LITERAL_J_RELU_KINK_POLICY
            or self.authority_scope != ALEXNET_LITERAL_J_SUPPORT_AUTHORITY
            or self.schema != ALEXNET_LITERAL_J_SUPPORT_DERIVATION_SCHEMA
        ):
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.AUTHORITY_VIOLATION,
                "AlexNet literal-J derivation crossed its source-only authority",
            )
        self.assert_integrity()

    def _payload(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            }
        )

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = _payload_hash(self._payload(), role="derivation-receipt")
        if self._receipt_hash is not None and self._receipt_hash != observed:
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.SOURCE_MUTATED,
                "AlexNet literal-J derivation receipt mutated",
            )
        if expected_hash is not None and expected_hash != observed:
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.SOURCE_MUTATED,
                "AlexNet literal-J derivation receipt hash differs",
            )
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1AlexNetLiteralJAnalyticSupportSourceResult:
    derivation_receipt: V1AlexNetLiteralJSupportDerivationReceipt
    hbm_forecast: V1AlexNetLiteralJSupportHBMForecast
    analytic_support_source: V1LiteralJResidualAnalyticSupportSourceCUDAResult = field(
        repr=False, compare=False
    )
    production_cuda_execution: bool = True
    schema: str = ALEXNET_LITERAL_J_SUPPORT_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.derivation_receipt)
            is not V1AlexNetLiteralJSupportDerivationReceipt
            or type(self.hbm_forecast) is not V1AlexNetLiteralJSupportHBMForecast
            or type(self.analytic_support_source)
            is not V1LiteralJResidualAnalyticSupportSourceCUDAResult
        ):
            raise TypeError(
                "AlexNet literal-J source result requires exact typed owners"
            )
        self.derivation_receipt.assert_integrity()
        self.hbm_forecast.assert_integrity(self.derivation_receipt.hbm_forecast_hash)
        self.analytic_support_source.source_handle.assert_live()
        source = self.analytic_support_source.receipt
        if (
            source.analytic_bound_derivation_receipt_hash
            != self.derivation_receipt.receipt_hash
            or source.backend_execution_receipt_hash
            != self.derivation_receipt.backend_execution_receipt_hash
            or source.literal_j_abs_bound_source_hash
            != self.derivation_receipt.literal_j_abs_bound_source_hash
            or source.direct_projection_abs_bound_derivation_receipt_hash
            != self.derivation_receipt.direct_projection_abs_bound_derivation_receipt_hash
            or self.production_cuda_execution is not True
            or self.schema != ALEXNET_LITERAL_J_SUPPORT_RESULT_SCHEMA
        ):
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.AUTHORITY_VIOLATION,
                "AlexNet and generic analytic-support custody do not close",
            )

    @property
    def result_hash(self) -> str:
        return _payload_hash(
            {
                "schema": self.schema,
                "derivation_receipt_hash": self.derivation_receipt.receipt_hash,
                "hbm_forecast_hash": self.hbm_forecast.forecast_hash,
                "generic_source_receipt_hash": self.analytic_support_source.receipt.receipt_hash,
                "production_cuda_execution": self.production_cuda_execution,
            },
            role="result",
        )


@dataclass(frozen=True, slots=True)
class _V1AlexNetLiteralJSupportNumericResult:
    route_abs_bound: Any = field(repr=False, compare=False)
    route_sum_projection_abs_bounds: Any = field(repr=False, compare=False)
    direct_projection_abs_bounds: Any = field(repr=False, compare=False)
    selected_projection_abs_bounds: Any = field(repr=False, compare=False)
    span_receipts: tuple[V1AlexNetLiteralJSupportSpanReceipt, ...]


@dataclass(frozen=True, slots=True)
class _V1AlexNetLiteralJSupportCPUReferenceResult:
    route_abs_bound: Any = field(repr=False, compare=False)
    route_sum_projection_abs_bounds: Any = field(repr=False, compare=False)
    direct_projection_abs_bounds: Any = field(repr=False, compare=False)
    selected_projection_abs_bounds: Any = field(repr=False, compare=False)
    possible_active_by_layer: Mapping[str, Any] = field(repr=False, compare=False)
    span_receipts: tuple[V1AlexNetLiteralJSupportSpanReceipt, ...]
    private_cpu_reference_only: bool = True
    production_authority: bool = False


@dataclass(frozen=True, slots=True)
class _V1AlexNetLiteralJAcquisitionClosure:
    fq_acquisition_plan_hash: str
    master_alpha_registration_hash: str
    campaign_contract_hash: str
    controller_run_hash: str
    train_dataset_identity_hash: str
    train_split_identity_hash: str
    source_role_registration_hash: str
    preprocessing_hash: str
    fq_generation_alpha_allocation: float
    acquisition_alpha_closure_hash: str


def _possible_active_masks(
    preactivation_interval_by_layer: Mapping[str, tuple[Any, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer in _LAYER_IDS:
        pair = preactivation_interval_by_layer.get(layer)
        if pair is None or len(pair) != 2:
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
                f"forward interval omitted AlexNet cut {layer!r}",
            )
        lower, upper = pair
        if lower.shape != upper.shape:
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
                "preactivation interval shape changed",
            )
        # The registered literal-J kink convention is 1[a > 0].  If the
        # outward upper endpoint is exactly zero, no admitted point has a>0.
        result[layer] = (upper > 0.0).contiguous()
    return result


def _apply_possible_active(cotangent: Any, active: Any) -> Any:
    if cotangent.shape != active.shape or active.dtype != torch.bool:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
            "cotangent and possible-active axes differ",
        )
    return torch.where(active, cotangent, torch.zeros_like(cotangent)).contiguous()


def _literal_j_backward_bounds(
    *,
    model: Any,
    gates: Mapping[str, np.ndarray],
    target_class_id: int,
    source_by_layer: Mapping[str, Any],
    possible_active_by_layer: Mapping[str, Any],
    pool_input_shapes: Mapping[int, tuple[int, int, int]],
    avgpool_input_shape: tuple[int, int, int],
    profile: Any,
    output_row_chunk: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    readout = model.classifier[6]
    destination = readout.weight.detach()[target_class_id].to(torch.float64).abs()
    readout_factor = _envelope._rounding_factor(
        profile, term_count=1, multiplication_count=1
    )
    destination = _envelope._outward_reverse_absolute_bound(
        destination, factor=readout_factor, support=destination != 0,
        profile=profile, operation_count=2, name="literal-J readout cotangent bound")
    cotangent_by_layer: dict[str, Any] = {"readout.hidden2": destination}
    factors: dict[str, float] = {"readout.hidden2": readout_factor}

    destination, factor = _envelope._linear_transpose_absolute_bound(
        _apply_possible_active(
            destination, possible_active_by_layer["readout.hidden2"]
        ),
        model.classifier[4],
        gates["readout.hidden2"],
        profile,
        output_row_chunk=output_row_chunk,
    )
    cotangent_by_layer["readout.hidden1"] = destination
    factors["readout.hidden1"] = factor
    destination, _ = _envelope._linear_transpose_absolute_bound(
        _apply_possible_active(
            destination, possible_active_by_layer["readout.hidden1"]
        ),
        model.classifier[1],
        gates["readout.hidden1"],
        profile,
        output_row_chunk=output_row_chunk,
    )

    avg_output_shape = tuple(int(item) for item in model.avgpool.output_size)
    if len(avg_output_shape) == 1:
        avg_output_shape = (avg_output_shape[0], avg_output_shape[0])
    expected_flat = avgpool_input_shape[0] * avg_output_shape[0] * avg_output_shape[1]
    if int(destination.numel()) != expected_flat:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
            "classifier cotangent cannot reshape to adaptive-pool output",
        )
    destination = destination.reshape(avgpool_input_shape[0], *avg_output_shape)
    destination, _ = _envelope._reverse_adaptive_avg_absolute_bound(
        destination, avgpool_input_shape, profile
    )
    destination, pool_factor = _envelope._reverse_maxpool_absolute_bound(
        destination, model.features[12], pool_input_shapes[12], profile
    )
    cotangent_by_layer["features.stage5.conv"] = destination
    factors["features.stage5.conv"] = pool_factor

    destination, factor = _envelope._conv_transpose_absolute_bound(
        _apply_possible_active(
            destination, possible_active_by_layer["features.stage5.conv"]
        ),
        model.features[10],
        gates["features.stage5.conv"],
        tuple(int(item) for item in source_by_layer["features.stage5.conv"].shape),
        profile,
    )
    cotangent_by_layer["features.stage4.conv"] = destination
    factors["features.stage4.conv"] = factor
    destination, factor = _envelope._conv_transpose_absolute_bound(
        _apply_possible_active(
            destination, possible_active_by_layer["features.stage4.conv"]
        ),
        model.features[8],
        gates["features.stage4.conv"],
        tuple(int(item) for item in source_by_layer["features.stage4.conv"].shape),
        profile,
    )
    cotangent_by_layer["features.stage3.conv"] = destination
    factors["features.stage3.conv"] = factor
    destination, _ = _envelope._conv_transpose_absolute_bound(
        _apply_possible_active(
            destination, possible_active_by_layer["features.stage3.conv"]
        ),
        model.features[6],
        gates["features.stage3.conv"],
        tuple(int(item) for item in source_by_layer["features.stage3.conv"].shape),
        profile,
    )
    destination, pool_factor = _envelope._reverse_maxpool_absolute_bound(
        destination, model.features[5], pool_input_shapes[5], profile
    )
    cotangent_by_layer["features.stage2.conv"] = destination
    factors["features.stage2.conv"] = pool_factor
    destination, _ = _envelope._conv_transpose_absolute_bound(
        _apply_possible_active(
            destination, possible_active_by_layer["features.stage2.conv"]
        ),
        model.features[3],
        gates["features.stage2.conv"],
        tuple(int(item) for item in source_by_layer["features.stage2.conv"].shape),
        profile,
    )
    destination, pool_factor = _envelope._reverse_maxpool_absolute_bound(
        destination, model.features[2], pool_input_shapes[2], profile
    )
    cotangent_by_layer["features.stage1.conv"] = destination
    factors["features.stage1.conv"] = pool_factor
    if set(cotangent_by_layer) != set(_LAYER_IDS):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
            "literal-J reverse envelope omitted an AlexNet cut",
        )
    return cotangent_by_layer, factors


def _conv_route_abs_bound(
    *,
    source: Any,
    destination: Any,
    module: Any,
    gate: np.ndarray,
    profile: Any,
    output_row_chunk: int,
) -> tuple[Any, float]:
    kernel_h, kernel_w = _envelope._pair(module.kernel_size)
    output_count = int(module.out_channels)
    input_count = int(module.in_channels)
    rows = max(1, min(output_count, int(output_row_chunk)))
    patches = F.unfold(
        source.unsqueeze(0),
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )[0]
    spatial = int(patches.shape[1])
    if (
        destination.ndim != 3
        or int(destination.shape[0]) != output_count
        or int(destination.shape[1] * destination.shape[2]) != spatial
    ):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
            "convolution source/destination incidence differs",
        )
    factor = _envelope._rounding_factor(
        profile, term_count=spatial, multiplication_count=3
    )
    result = torch.empty(
        tuple(int(item) for item in module.weight.shape),
        dtype=torch.float64,
        device=module.weight.device,
    )
    for start in range(0, output_count, rows):
        stop = min(output_count, start + rows)
        incidence = torch.mm(
            destination[start:stop].reshape(stop - start, spatial), patches.T
        )
        weight = (
            module.weight.detach()[start:stop]
            .to(torch.float64)
            .abs()
            .reshape(stop - start, -1)
        )
        gate_tensor = torch.as_tensor(
            gate[start:stop], dtype=torch.float64, device=weight.device
        ).reshape(stop - start, -1)
        base = incidence * weight * gate_tensor
        values = torch.where(
            base == 0.0,
            torch.zeros_like(base),
            _up(base * factor),
        )
        if not bool(torch.isfinite(values).all().item()) or bool(
            (values < 0.0).any().item()
        ):
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
                "convolution literal-J route bound is invalid",
            )
        result[start:stop] = values.reshape(
            stop - start, input_count, kernel_h, kernel_w
        )
    return result.contiguous(), factor


def _linear_route_abs_bound(
    *,
    source: Any,
    destination: Any,
    module: Any,
    gate: np.ndarray,
    profile: Any,
    output_row_chunk: int,
) -> tuple[Any, float]:
    rows = max(1, min(int(module.out_features), int(output_row_chunk)))
    factor = _envelope._rounding_factor(profile, term_count=1, multiplication_count=3)
    result = torch.empty(
        tuple(int(item) for item in module.weight.shape),
        dtype=torch.float64,
        device=module.weight.device,
    )
    for start in range(0, int(module.out_features), rows):
        stop = min(int(module.out_features), start + rows)
        weight = module.weight.detach()[start:stop].to(torch.float64).abs()
        gate_tensor = torch.as_tensor(
            gate[start:stop], dtype=torch.float64, device=weight.device
        )
        base = (
            weight
            * source.reshape(1, -1)
            * destination[start:stop].reshape(-1, 1)
            * gate_tensor
        )
        values = torch.where(
            base == 0.0,
            torch.zeros_like(base),
            _up(base * factor),
        )
        if not bool(torch.isfinite(values).all().item()) or bool(
            (values < 0.0).any().item()
        ):
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
                "linear literal-J route bound is invalid",
            )
        result[start:stop] = values
    return result.contiguous(), factor


def _conv_fractional_drive_interval(
    *,
    source_lower: Any,
    source_upper: Any,
    module: Any,
    gate: np.ndarray,
    direction: Any,
    profile: Any,
) -> tuple[Any, Any]:
    weight = module.weight.detach().to(torch.float64)
    gate_tensor = torch.as_tensor(gate, dtype=torch.float64, device=weight.device)
    effective = weight * gate_tensor * direction.reshape(weight.shape)
    positive = effective.clamp_min(0.0)
    negative = effective.clamp_max(0.0)
    common = (module.stride, module.padding, module.dilation, module.groups)
    nominal_lower = (
        F.conv2d(source_lower.unsqueeze(0), positive, None, *common)[0]
        + F.conv2d(source_upper.unsqueeze(0), negative, None, *common)[0]
    )
    nominal_upper = (
        F.conv2d(source_upper.unsqueeze(0), positive, None, *common)[0]
        + F.conv2d(source_lower.unsqueeze(0), negative, None, *common)[0]
    )
    source_absolute = _envelope._interval_absolute_bound(source_lower, source_upper)
    absolute_sum = F.conv2d(
        source_absolute.unsqueeze(0), effective.abs(), None, *common
    )[0]
    support = (
        F.conv2d(
            (source_absolute != 0.0).to(torch.float64).unsqueeze(0),
            (effective != 0.0).to(torch.float64),
            None,
            *common,
        )[0]
        > 0.0
    )
    terms = (
        int(module.in_channels // module.groups)
        * int(module.kernel_size[0])
        * int(module.kernel_size[1])
    )
    factor = _envelope._rounding_factor(
        profile, term_count=2 * terms, multiplication_count=1
    )
    return _envelope._outward_interval(
        nominal_lower,
        nominal_upper,
        absolute_sum,
        factor,
        profile=profile,
        name="literal-J fractional convolution drive",
        nonzero_support=support,
        absolute_underflow_radius=_envelope._execution_underflow_radius(
            profile, operation_count=2 * (terms + 1)
        ),
    )


def _linear_fractional_drive_interval(
    *,
    source_lower: Any,
    source_upper: Any,
    module: Any,
    gate: np.ndarray,
    direction: Any,
    profile: Any,
    output_row_chunk: int,
) -> tuple[Any, Any]:
    rows = max(1, min(int(module.out_features), int(output_row_chunk)))
    lower = torch.zeros(
        int(module.out_features), dtype=torch.float64, device=module.weight.device
    )
    upper = torch.zeros_like(lower)
    source_absolute = _envelope._interval_absolute_bound(source_lower, source_upper)
    terms = int(module.in_features)
    factor = _envelope._rounding_factor(
        profile, term_count=2 * terms, multiplication_count=1
    )
    underflow = _envelope._execution_underflow_radius(
        profile, operation_count=2 * (terms + 1)
    )
    for start in range(0, int(module.out_features), rows):
        stop = min(int(module.out_features), start + rows)
        weight = module.weight.detach()[start:stop].to(torch.float64)
        gate_tensor = torch.as_tensor(
            gate[start:stop], dtype=torch.float64, device=weight.device
        )
        effective = weight * gate_tensor * direction[start:stop]
        positive = effective.clamp_min(0.0)
        negative = effective.clamp_max(0.0)
        nominal_lower = torch.mv(positive, source_lower) + torch.mv(
            negative, source_upper
        )
        nominal_upper = torch.mv(positive, source_upper) + torch.mv(
            negative, source_lower
        )
        absolute_sum = torch.mv(effective.abs(), source_absolute)
        support = (
            torch.mv(
                (effective != 0.0).to(torch.float64),
                (source_absolute != 0.0).to(torch.float64),
            )
            > 0.0
        )
        chunk_lower, chunk_upper = _envelope._outward_interval(
            nominal_lower,
            nominal_upper,
            absolute_sum,
            factor,
            profile=profile,
            name="literal-J fractional linear drive",
            nonzero_support=support,
            absolute_underflow_radius=underflow,
        )
        lower[start:stop] = chunk_lower
        upper[start:stop] = chunk_upper
    return lower.contiguous(), upper.contiguous()


def _outward_nonnegative_tensor_sum(values: Any) -> Any:
    flat = values.reshape(-1)
    if int(flat.numel()) == 0:
        return torch.zeros((), dtype=torch.float64, device=values.device)
    nominal = torch.sum(flat)
    count = 2 * int(flat.numel()) + 1
    eps = 2.0**-53
    product = count * eps
    if product >= 0.5:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
            "literal-J reduction exceeds the finite gamma regime",
        )
    gamma = product / (1.0 - product)
    rounding = _up(
        torch.abs(nominal) * gamma + count * float.fromhex("0x0.0000000000001p-1022")
    )
    result = torch.where(
        nominal == 0.0, torch.zeros_like(nominal), _up(nominal + rounding)
    )
    if not bool(torch.isfinite(result).item()) or bool((result < 0.0).item()):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
            "literal-J nonnegative reduction overflowed",
        )
    return result


def _projection_route_sum_bound(directions: Any, route_bound: Any) -> Any:
    values = []
    for index in range(int(directions.shape[0])):
        values.append(
            _outward_nonnegative_tensor_sum(_up(torch.abs(directions[index]) * route_bound))
        )
    return torch.stack(values).contiguous()


def _direct_projection_bound(
    *,
    model: Any,
    route_axis: DeclaredRouteAxis,
    gates: Mapping[str, np.ndarray],
    source_interval_by_layer: Mapping[str, tuple[Any, Any]],
    cotangent_by_layer: Mapping[str, Any],
    possible_active_by_layer: Mapping[str, Any],
    directions: Any,
    profile: Any,
    output_row_chunk: int,
) -> Any:
    totals: list[Any] = []
    for direction_index in range(int(directions.shape[0])):
        total = torch.zeros((), dtype=torch.float64, device=directions.device)
        for block in route_axis.route_blocks:
            layer = str(block.layer_id)
            module = model.get_submodule(
                dict(zip(_LAYER_IDS, _AFFINE_PATHS, strict=True))[layer]
            )
            source_lower, source_upper = source_interval_by_layer[layer]
            direction = directions[
                direction_index, int(block.offset) : int(block.stop)
            ].reshape(block.shape)
            if block.route_kind == "conv_kernel_weight":
                drive_lower, drive_upper = _conv_fractional_drive_interval(
                    source_lower=source_lower,
                    source_upper=source_upper,
                    module=module,
                    gate=gates[layer],
                    direction=direction,
                    profile=profile,
                )
            elif block.route_kind == "linear_weight":
                drive_lower, drive_upper = _linear_fractional_drive_interval(
                    source_lower=source_lower,
                    source_upper=source_upper,
                    module=module,
                    gate=gates[layer],
                    direction=direction,
                    profile=profile,
                    output_row_chunk=output_row_chunk,
                )
            else:
                _refuse(
                    V1AlexNetLiteralJSupportRefusalCode.ROUTE_AXIS_MISMATCH,
                    "unsupported literal-J route kind",
                )
            drive_absolute = _envelope._interval_absolute_bound(
                drive_lower, drive_upper
            )
            local_cotangent = _apply_possible_active(
                cotangent_by_layer[layer], possible_active_by_layer[layer]
            )
            layer_bound = _outward_nonnegative_tensor_sum(
                local_cotangent * drive_absolute
            )
            total = torch.where(
                torch.logical_and(total == 0.0, layer_bound == 0.0),
                torch.zeros_like(total),
                _up(total + layer_bound),
            )
        totals.append(total)
    result = torch.stack(totals).contiguous()
    _require_tensor(
        result,
        name="direct projection absolute bounds",
        shape=(int(directions.shape[0]),),
        device=directions.device,
        nonnegative=True,
    )
    return result


def _legacy_fp32_row_block_abs_bound(
    *, source, destination, module, gate, scale, profile, output_row_chunk,
):
    """Uniform bound for the actual FP32 gradient/scale → correlation → W chain.

    Gates apply only after checking the ungated intermediate products: the
    legacy producer forms those rows even when a route's final gate is zero.
    Local normal-minimum floors cover FTZ and gradual underflow, and propagate
    through the subsequent multiplication by W.
    """
    if profile.storage_dtype != "torch.float32":
        _refuse(V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
                "factored legacy row support currently requires a FP32 model")
    effective = 1.0 if abs(float(scale)) < 1e-12 else float(scale)
    with np.errstate(over="ignore", under="ignore"):
        effective = abs(float(np.float32(effective)))
    if not math.isfinite(effective) or effective == 0:
        _refuse(V1AlexNetLiteralJSupportRefusalCode.NUMERICAL_REFUSAL,
                "legacy scale is not finite and nonzero after FP32 resolution")
    factor = _envelope._rounding_factor(profile, term_count=1, multiplication_count=1)

    def close(base, *, factor=factor, operations=2, name):
        # Unconditional floors conservatively cover every route. Structural
        # absence is recovered by the exact final gate, after admission.
        return _envelope._outward_reverse_absolute_bound(
            base, factor=factor, support=torch.ones_like(base, dtype=torch.bool),
            profile=profile, operation_count=operations, name=name)

    cotangent = close(_up(destination / effective), name="legacy scaled FP32 cotangent")
    rows = max(1, min(int(module.weight.shape[0]), int(output_row_chunk)))
    result = torch.empty_like(module.weight, dtype=torch.float64)
    convolution = isinstance(module, nn.Conv2d)
    if convolution:
        if module.groups != 1:
            _refuse(V1AlexNetLiteralJSupportRefusalCode.ROUTE_AXIS_MISMATCH,
                    "canonical AlexNet support requires ungrouped convolution")
        patches = F.unfold(source.unsqueeze(0), kernel_size=module.kernel_size,
                           dilation=module.dilation, padding=module.padding, stride=module.stride)[0]
        spatial = int(patches.shape[1])
        correlation_factor = _envelope._rounding_factor(
            profile, term_count=spatial, multiplication_count=1)
    for start in range(0, int(module.weight.shape[0]), rows):
        stop = min(int(module.weight.shape[0]), start + rows)
        if convolution:
            incidence = cotangent[start:stop].reshape(stop - start, -1) @ patches.T
            contracted = close(incidence, factor=correlation_factor,
                               operations=2 * spatial + 1, name="legacy ungated FP32 correlation")
        else:
            contracted = close(cotangent[start:stop, None] * source[None, :],
                               name="legacy ungated FP32 outer product")
        weight = module.weight.detach()[start:stop].to(torch.float64).abs().reshape(stop - start, -1)
        weighted = close(_up(contracted * weight), name="legacy ungated FP32 weighted row")
        gate_tensor = torch.as_tensor(gate[start:stop], device=result.device, dtype=torch.bool)
        result[start:stop] = torch.where(gate_tensor, weighted.reshape(gate_tensor.shape), 0.0)
    return result.contiguous(), factor


def _derive_numeric_support(
    *,
    model: Any,
    route_axis: DeclaredRouteAxis,
    current_mask: HardMask,
    query: QuerySpec,
    input_box: _envelope.V1RegisteredNormalizedInputBox,
    directions: Any,
    route_output_row_chunk: int,
    affine_output_row_chunk: int,
    legacy_row_scales_by_layer: Mapping[str, float] | None = None,
) -> tuple[_V1AlexNetLiteralJSupportNumericResult, dict[str, Any]]:
    profile = _envelope._arithmetic_profile(model)
    gates = _envelope._split_gates(current_mask, route_axis)
    target_class_id = _envelope._target_class(
        query, int(model.classifier[6].out_features)
    )
    if legacy_row_scales_by_layer is not None and set(legacy_row_scales_by_layer) != set(_LAYER_IDS):
        _refuse(V1AlexNetLiteralJSupportRefusalCode.NATIVE_CUT_MISMATCH,
                "legacy scale source must cover every canonical cut exactly")
    with torch.no_grad():
        (
            source_by_layer,
            source_interval_by_layer,
            preactivation_interval_by_layer,
            _destination_shapes,
            _forward_factors,
            pool_input_shapes,
            avgpool_input_shape,
            _final_hidden_interval,
        ) = _envelope._forward_bounds(
            model=model,
            gates=gates,
            input_box=input_box,
            profile=profile,
            output_row_chunk=affine_output_row_chunk,
        )
        possible_active = _possible_active_masks(preactivation_interval_by_layer)
        cotangent_by_layer, _backward_factors = _literal_j_backward_bounds(
            model=model,
            gates=gates,
            target_class_id=target_class_id,
            source_by_layer=source_by_layer,
            possible_active_by_layer=possible_active,
            pool_input_shapes=pool_input_shapes,
            avgpool_input_shape=avgpool_input_shape,
            profile=profile,
            output_row_chunk=affine_output_row_chunk,
        )

        route_bound = torch.empty(
            route_axis.route_count, dtype=torch.float64, device=directions.device
        )
        span_receipts: list[V1AlexNetLiteralJSupportSpanReceipt] = []
        for block in route_axis.route_blocks:
            layer = str(block.layer_id)
            module = model.get_submodule(
                dict(zip(_LAYER_IDS, _AFFINE_PATHS, strict=True))[layer]
            )
            destination = _apply_possible_active(
                cotangent_by_layer[layer], possible_active[layer]
            )
            if legacy_row_scales_by_layer is not None:
                block_bound, route_factor = _legacy_fp32_row_block_abs_bound(
                    source=source_by_layer[layer], destination=destination, module=module,
                    gate=gates[layer], scale=legacy_row_scales_by_layer[layer], profile=profile,
                    output_row_chunk=route_output_row_chunk)
            elif block.route_kind == "conv_kernel_weight":
                block_bound, route_factor = _conv_route_abs_bound(
                    source=source_by_layer[layer],
                    destination=destination,
                    module=module,
                    gate=gates[layer],
                    profile=profile,
                    output_row_chunk=route_output_row_chunk,
                )
            elif block.route_kind == "linear_weight":
                block_bound, route_factor = _linear_route_abs_bound(
                    source=source_by_layer[layer],
                    destination=destination,
                    module=module,
                    gate=gates[layer],
                    profile=profile,
                    output_row_chunk=route_output_row_chunk,
                )
            else:
                _refuse(
                    V1AlexNetLiteralJSupportRefusalCode.ROUTE_AXIS_MISMATCH,
                    "unsupported literal-J route kind",
                )
            if tuple(int(item) for item in block_bound.shape) != tuple(block.shape):
                _refuse(
                    V1AlexNetLiteralJSupportRefusalCode.ROUTE_AXIS_MISMATCH,
                    "literal-J block bound changed the route shape",
                )
            route_bound[int(block.offset) : int(block.stop)].copy_(
                block_bound.reshape(-1)
            )
            span_receipts.append(
                V1AlexNetLiteralJSupportSpanReceipt(
                    layer_id=layer,
                    module_path=dict(zip(_LAYER_IDS, _AFFINE_PATHS, strict=True))[
                        layer
                    ],
                    route_kind=str(block.route_kind),
                    route_shape=tuple(int(item) for item in block.shape),
                    route_offset=int(block.offset),
                    route_stop=int(block.stop),
                    routes_per_native_fiber=math.prod(
                        tuple(int(item) for item in block.shape)[1:]
                    ),
                    possible_active_unit_count=int(possible_active[layer].sum().item()),
                    destination_unit_count=int(possible_active[layer].numel()),
                    live_route_count=int(np.count_nonzero(gates[layer])),
                    source_absolute_max=float(source_by_layer[layer].max().item()),
                    cotangent_absolute_max=float(
                        cotangent_by_layer[layer].max().item()
                    ),
                    route_bound_max=float(block_bound.max().item()),
                    route_rounding_factor=float(route_factor),
                )
            )
        _require_tensor(
            route_bound,
            name="literal_j_abs_bound",
            shape=(route_axis.route_count,),
            device=directions.device,
            nonnegative=True,
        )
        # Round each product before reduction: support may itself contain
        # subnormal entries; a rounded-to-zero product is not an exact zero.
        route_sum = _projection_route_sum_bound(directions, route_bound)
        direct = route_sum.clone() if legacy_row_scales_by_layer is not None else _direct_projection_bound(
            model=model,
            route_axis=route_axis,
            gates=gates,
            source_interval_by_layer=source_interval_by_layer,
            cotangent_by_layer=cotangent_by_layer,
            possible_active_by_layer=possible_active,
            directions=directions,
            profile=profile,
            output_row_chunk=affine_output_row_chunk,
        )
        # The old cancellation envelope concerns mathematical derivatives.
        # It cannot tighten rounded legacy rows without an additive row-error
        # projection. Factored support therefore uses the route-sum law only.
        selected = torch.minimum(route_sum, direct).contiguous()
    return (
        _V1AlexNetLiteralJSupportNumericResult(
            route_abs_bound=route_bound,
            route_sum_projection_abs_bounds=route_sum,
            direct_projection_abs_bounds=direct,
            selected_projection_abs_bounds=selected,
            span_receipts=tuple(span_receipts),
        ),
        possible_active,
    )


def _canonical_span_axis(
    route_axis: DeclaredRouteAxis,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            str(block.layer_id),
            dict(zip(_LAYER_IDS, _AFFINE_PATHS, strict=True))[str(block.layer_id)],
            str(block.route_kind),
            tuple(int(item) for item in block.shape),
            int(block.offset),
            int(block.stop),
            math.prod(tuple(int(item) for item in block.shape)[1:]),
        )
        for block in route_axis.route_blocks
    )


def _validate_canonical_spans(
    route_axis: DeclaredRouteAxis,
    native_fibers: V1NativeDestinationFiberAxisReceipt,
) -> None:
    if _canonical_span_axis(route_axis) != _CANONICAL_SPANS:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.ROUTE_AXIS_MISMATCH,
            "literal-J production source requires the seven canonical AlexNet spans",
        )
    native_fibers.assert_integrity()
    observed_cuts = tuple(
        (
            cut.layer_id,
            cut.route_kind,
            tuple(cut.route_shape),
            cut.route_offset,
            cut.route_stop,
            cut.routes_per_fiber,
        )
        for cut in native_fibers.cuts
    )
    expected_cuts = tuple(
        (layer, kind, shape, start, stop, routes_per_fiber)
        for layer, _path, kind, shape, start, stop, routes_per_fiber in _CANONICAL_SPANS
    )
    if observed_cuts != expected_cuts:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.NATIVE_CUT_MISMATCH,
            "native-fiber cuts differ from the canonical AlexNet literal-J spans",
        )


def _shape_ledger(
    model: Any, input_box: _envelope.V1RegisteredNormalizedInputBox
) -> tuple[int, int]:
    channels = len(input_box.lower_by_channel)
    height, width = input_box.spatial_shape
    source_elements = 0
    destination_elements = 0
    for module in model.features:
        if type(module) is nn.Conv2d:
            source_elements += channels * height * width
            kernel_h, kernel_w = _envelope._pair(module.kernel_size)
            stride_h, stride_w = _envelope._pair(module.stride)
            pad_h, pad_w = _envelope._pair(module.padding)
            dilation_h, dilation_w = _envelope._pair(module.dilation)
            height = math.floor(
                (height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1
            )
            width = math.floor(
                (width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1
            )
            channels = int(module.out_channels)
            destination_elements += channels * height * width
        elif type(module) is nn.MaxPool2d:
            height, width = _envelope._maxpool_output_shape(module, (height, width))
        elif type(module) is not nn.ReLU:
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.SOURCE_IDENTITY_MISMATCH,
                "unsupported feature module in HBM shape forecast",
            )
    output_size = _envelope._pair(model.avgpool.output_size)
    features = channels * output_size[0] * output_size[1]
    for index, module in enumerate(model.classifier):
        if type(module) is nn.Linear and index != 6:
            source_elements += features
            features = int(module.out_features)
            destination_elements += features
    return source_elements, destination_elements


def _cuda_device_identity(device: Any) -> str:
    if torch is None or device.type != "cuda" or not torch.cuda.is_available():
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.DEVICE_REQUIRED,
            "CUDA device identity requires one live CUDA device",
        )
    properties = torch.cuda.get_device_properties(device)
    return _payload_hash(
        {
            "schema": "alexnet_literal_j_cuda_device_identity_v1",
            "device": str(device),
            "name": str(properties.name),
            "total_memory": int(properties.total_memory),
            "compute_capability": (
                int(properties.major),
                int(properties.minor),
            ),
            "multi_processor_count": int(properties.multi_processor_count),
            "uuid": str(getattr(properties, "uuid", "unavailable")),
        },
        role="cuda-device-identity",
    )


def build_v1_alexnet_literal_j_support_hbm_forecast(
    *,
    model: Any,
    route_axis: DeclaredRouteAxis,
    input_box: _envelope.V1RegisteredNormalizedInputBox,
    device_identity: str,
    device_free_memory_bytes: int,
    device_total_memory_bytes: int,
    current_allocated_device_bytes: int,
    current_reserved_device_bytes: int,
    hbm_ceiling_fraction: float = _DEFAULT_HBM_CEILING_FRACTION,
    _available_memory_bytes: int | None = None,
) -> V1AlexNetLiteralJSupportHBMForecast:
    if (
        type(route_axis) is not DeclaredRouteAxis
        or type(input_box) is not _envelope.V1RegisteredNormalizedInputBox
    ):
        raise TypeError(
            "AlexNet literal-J HBM forecast requires exact route/input owners"
        )
    source_elements, destination_elements = _shape_ledger(model, input_box)
    route_count = route_axis.route_count
    largest_block = max(int(block.route_count) for block in route_axis.route_blocks)
    retained_interval = (3 * source_elements + 2 * destination_elements) * _F64_BYTES
    retained_cotangent = destination_elements * (_F64_BYTES + _BOOL_BYTES)
    # Ten route-block-sized FP64 slabs conservatively cover effective weights,
    # sign splits, gate/direction staging, incidence, absolute mass, and output
    # temporaries for the largest affine contraction.
    workspace = 10 * largest_block * _F64_BYTES
    owner_route = route_count * _F64_BYTES
    owner_direct = 2 * _F64_BYTES
    # Generic handoff clones the route/direct vectors and retains one mutation
    # guard for each; the owner vectors remain live until construction returns.
    generic_handoff = 2 * owner_route + 2 * owner_direct
    derivation_peak = (
        retained_interval + retained_cotangent + workspace + owner_route + owner_direct
    )
    handoff_peak = owner_route + owner_direct + generic_handoff
    peak = max(derivation_peak, handoff_peak)
    fraction = _finite_fraction(hbm_ceiling_fraction, name="hbm_ceiling_fraction")
    total = _positive_int(device_total_memory_bytes, name="device_total_memory_bytes")
    free = _nonnegative_int(device_free_memory_bytes, name="device_free_memory_bytes")
    allocated = _nonnegative_int(
        current_allocated_device_bytes,
        name="current_allocated_device_bytes",
    )
    reserved = _nonnegative_int(
        current_reserved_device_bytes, name="current_reserved_device_bytes"
    )
    identity = _identity(device_identity, name="device_identity")
    ceiling = math.floor(fraction * total)
    projected_reserved = reserved + peak
    available = free + max(0, reserved - allocated) if _available_memory_bytes is None else _available_memory_bytes
    admitted = peak <= available and allocated + peak <= ceiling
    return V1AlexNetLiteralJSupportHBMForecast(
        route_count=route_count,
        direction_count=2,
        largest_route_block_count=largest_block,
        retained_interval_state_bytes=retained_interval,
        retained_cotangent_and_activity_bytes=retained_cotangent,
        largest_affine_workspace_bytes=workspace,
        owner_route_vector_bytes=owner_route,
        owner_direct_vector_bytes=owner_direct,
        generic_handoff_vector_and_guard_bytes=generic_handoff,
        derivation_incremental_peak_bytes=derivation_peak,
        handoff_incremental_peak_bytes=handoff_peak,
        projected_incremental_peak_bytes=peak,
        device_identity=identity,
        current_allocated_device_bytes=allocated,
        current_reserved_device_bytes=reserved,
        device_free_memory_bytes=free,
        device_total_memory_bytes=total,
        hbm_ceiling_fraction=fraction,
        hbm_ceiling_bytes=ceiling,
        projected_total_reserved_peak_bytes=projected_reserved,
        admitted=admitted,
        _issuance_token=_FORECAST_TOKEN,
    )


def _validate_fq_acquisition_custody(
    *,
    fq_acquisition_plan: Any,
    backend_admission: V1AlexNetBackendAdmission,
    model_load_receipt: ModelLoadReceipt,
    query: QuerySpec,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    input_box: _envelope.V1RegisteredNormalizedInputBox,
) -> _V1AlexNetLiteralJAcquisitionClosure:
    # Import lazily: the production backend may itself install this source
    # owner, whereas the exact plan class lives in that backend module.
    from .modern_v1_alexnet_backend import V1AlexNetFQAcquisitionPlan

    if type(fq_acquisition_plan) is not V1AlexNetFQAcquisitionPlan:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.INVALID_TYPE,
            "fq_acquisition_plan must use exact pre-outcome custody",
        )
    try:
        plan_hash = fq_acquisition_plan.assert_integrity()
        master = fq_acquisition_plan.master_alpha_registration
        master_hash = master.assert_integrity()
    except (TypeError, ValueError, PermissionError) as error:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.ACQUISITION_PLAN_MISMATCH,
            f"F/Q acquisition custody failed integrity: {error}",
        )

    generation = population_registration.proposal_generation
    if generation >= len(master.fq_generation_allocations):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.ACQUISITION_PLAN_MISMATCH,
            "population generation exceeds the registered F/Q alpha horizon",
        )
    generation_alpha = float(master.fq_generation_allocations[generation])
    alpha_tolerance = 32.0 * math.ulp(
        max(1.0, generation_alpha, population_registration.familywise_alpha)
    )
    plan_axis = tuple(
        (name, len(context_ids))
        for name, context_ids in fq_acquisition_plan.certification_context_ids_by_stratum
    )
    population_axis = tuple(
        zip(
            population_registration.stratum_ids,
            population_registration.population_stratum_counts,
            strict=True,
        )
    )

    envelope_source_path = Path(
        inspect.getsourcefile(_envelope.build_v1_alexnet_corrected_attribution_envelope)
        or ""
    ).resolve()
    if not envelope_source_path.is_file():
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "AlexNet analytic-envelope source is unavailable",
        )
    envelope_source_hash = hashlib.sha256(envelope_source_path.read_bytes()).hexdigest()
    envelope_factory_identity_hash = content_hash(
        {
            "module_source_sha256": envelope_source_hash,
            "callable": (
                "mechanistic_systems.circuits.irg.modern_v1_alexnet_envelope."
                "build_v1_alexnet_corrected_attribution_envelope"
            ),
        },
        prefix="partition-free-v1-alexnet-envelope-factory-identity-",
    )
    admitted_checkpoint_hash = backend_admission.metadata.get("model_checkpoint_sha256")
    if (
        fq_acquisition_plan.source_kind != "causal_full_population_roles"
        or fq_acquisition_plan.discovery_context_ids_by_stratum
        != fq_acquisition_plan.certification_context_ids_by_stratum
        or plan_axis != population_axis
        or sum(count for _, count in plan_axis)
        != population_registration.population_count
        or fq_acquisition_plan.train_split_identity_hash
        != backend_admission.context_population_hash
        or fq_acquisition_plan.expected_model_identity_hash
        != population_registration.model_identity_hash
        or fq_acquisition_plan.expected_model_identity_hash
        != backend_admission.model_seal_hash
        or fq_acquisition_plan.expected_model_state_hash
        != model_load_receipt.state_dict_hash
        or fq_acquisition_plan.query_contract_hash != query.content_hash
        or fq_acquisition_plan.preprocessing_hash != input_box.preprocessing_hash
        or fq_acquisition_plan.normalized_input_box_hash != input_box.box_hash
        or fq_acquisition_plan.expected_envelope_verifier_source_hash
        != envelope_source_hash
        or fq_acquisition_plan.analytic_envelope_factory_identity_hash
        != envelope_factory_identity_hash
        or master.campaign_contract_hash != backend_admission.campaign_contract_hash
        or master.controller_run_hash != backend_admission.controller_run_id
        or master.maximum_proposal_generations
        != fq_acquisition_plan.maximum_proposal_generations
        or population_registration.familywise_alpha > generation_alpha + alpha_tolerance
        or model_load_receipt.claim_eligible_source is not True
        or admitted_checkpoint_hash != model_load_receipt.checkpoint_hash
    ):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.ACQUISITION_PLAN_MISMATCH,
            "F/Q plan, full-train population, alpha, model, query, or input box differ",
        )

    alpha_closure_hash = _payload_hash(
        {
            "schema": "alexnet_literal_j_fq_population_alpha_closure_v1",
            "fq_acquisition_plan_hash": plan_hash,
            "master_alpha_registration_hash": master_hash,
            "campaign_contract_hash": master.campaign_contract_hash,
            "controller_run_hash": master.controller_run_hash,
            "proposal_generation": generation,
            "fq_generation_alpha_allocation": generation_alpha,
            "population_projection_registration_hash": (
                population_registration.registration_hash
            ),
            "population_alpha_ledger_hash": population_registration.alpha_ledger_hash,
            "population_alpha_coordinate_hash": (
                population_registration.alpha_coordinate_hash
            ),
            "population_alpha_lineage_hash": (
                population_registration.alpha_lineage_hash
            ),
            "population_familywise_alpha": (population_registration.familywise_alpha),
            "child_allocation_may_be_stricter_but_never_exceeds_generation": True,
        },
        role="fq-population-alpha-closure",
    )
    return _V1AlexNetLiteralJAcquisitionClosure(
        fq_acquisition_plan_hash=plan_hash,
        master_alpha_registration_hash=master_hash,
        campaign_contract_hash=master.campaign_contract_hash,
        controller_run_hash=master.controller_run_hash,
        train_dataset_identity_hash=fq_acquisition_plan.train_dataset_identity_hash,
        train_split_identity_hash=fq_acquisition_plan.train_split_identity_hash,
        source_role_registration_hash=(
            fq_acquisition_plan.source_role_registration_hash
        ),
        preprocessing_hash=fq_acquisition_plan.preprocessing_hash,
        fq_generation_alpha_allocation=generation_alpha,
        acquisition_alpha_closure_hash=alpha_closure_hash,
    )


def _validate_alexnet_support_environment(
    *, backend_admission, model_load_receipt, model, evaluator, query,
    current_mask, route_axis, native_fiber_axis, input_box, model_identity_hash,
):
    """Shared exact model/chart admission, without inventing a residual owner."""
    exact_types = (
        (backend_admission, V1AlexNetBackendAdmission, "backend_admission"),
        (model_load_receipt, ModelLoadReceipt, "model_load_receipt"),
        (evaluator, TorchCircuitEvaluator, "evaluator"),
        (query, QuerySpec, "query"),
        (current_mask, HardMask, "current_mask"),
        (route_axis, DeclaredRouteAxis, "route_axis"),
        (native_fiber_axis, V1NativeDestinationFiberAxisReceipt, "native_fiber_axis"),
        (input_box, _envelope.V1RegisteredNormalizedInputBox, "input_box"),
    )
    for value, expected, name in exact_types:
        if type(value) is not expected:
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.INVALID_TYPE,
                f"{name} must use exact typed custody",
            )
    if model is not evaluator.source_model:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.EVALUATOR_MISMATCH,
            "live model is not the evaluator source model",
        )
    input_box.assert_integrity()
    native_fiber_axis.assert_integrity()
    model_load_receipt.validate()
    if (
        backend_admission.model_seal_hash != model_identity_hash
        or backend_admission.route_axis_hash != route_axis.route_axis_hash
        or backend_admission.query_id != query.query_id
        or backend_admission.architecture_family != "alexnet"
    ):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "backend admission differs from population/model/query coordinates",
        )
    if current_mask.route_axis_hash != route_axis.route_axis_hash or current_mask.route_count != route_axis.route_count:
        _refuse(V1AlexNetLiteralJSupportRefusalCode.CHART_MISMATCH,
                "current mask differs from declared route custody")
    if query.content_hash not in tuple(item.content_hash for item in evaluator.queries):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.QUERY_MISMATCH,
            "query is not one exact evaluator query",
        )
    state_hash = state_dict_digest(model)
    if (
        state_hash != model_load_receipt.state_dict_hash
        or model_load_receipt.architecture_family != "alexnet_conv_kernel_weight"
    ):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.MODEL_STATE_MISMATCH,
            "live AlexNet differs from its model-load receipt",
        )
    rederived_native = build_alexnet_v1_native_destination_fiber_axis(
        evaluator, route_axis
    )
    if rederived_native.receipt_hash != native_fiber_axis.receipt_hash:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.NATIVE_CUT_MISMATCH,
            "provided native cuts differ from the live evaluator derivation",
        )
    _validate_canonical_spans(route_axis, native_fiber_axis)
    layout = _envelope.admit_v1_alexnet_envelope_layout(
        model=model,
        route_axis=route_axis,
        current_mask=current_mask,
        query=query,
        model_identity_hash=model_identity_hash,
        expected_model_state_hash=model_load_receipt.state_dict_hash,
        preprocessing_hash=input_box.preprocessing_hash,
        input_box=input_box,
    )
    if (
        layout.route_axis_hash != route_axis.route_axis_hash
        or layout.route_count != route_axis.route_count
        or layout.query_contract_hash != query.content_hash
    ):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.SOURCE_IDENTITY_MISMATCH,
            "analytic layout admission differs from registered owners",
        )
    evaluator_source_identity = _payload_hash(
        {
            "schema": "alexnet_literal_j_evaluator_source_identity_v1",
            "evaluator_id": evaluator.evaluator_id,
            "parity_receipt_hash": evaluator.parity_receipt_hash,
            "native_fiber_receipt_hash": native_fiber_axis.receipt_hash,
            "model_load_receipt_hash": model_load_receipt.content_hash,
            "route_axis_content_hash": route_axis.content_hash,
        },
        role="evaluator-source",
    )
    return layout, state_hash, evaluator_source_identity


def _validate_production_owners(
    *,
    fq_acquisition_plan: Any,
    backend_admission: V1AlexNetBackendAdmission,
    model_load_receipt: ModelLoadReceipt,
    model: Any,
    evaluator: TorchCircuitEvaluator,
    query: QuerySpec,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    current_mask: HardMask,
    route_axis: DeclaredRouteAxis,
    native_fiber_axis: V1NativeDestinationFiberAxisReceipt,
    input_box: _envelope.V1RegisteredNormalizedInputBox,
) -> tuple[Any, str, str, str, _V1AlexNetLiteralJAcquisitionClosure]:
    if type(population_registration) is not V1LiteralJDFixedResidualPopulationProjectionRegistration:
        _refuse(V1AlexNetLiteralJSupportRefusalCode.INVALID_TYPE,
                "population_registration must use exact typed custody")
    population_registration.assert_integrity()
    layout, state_hash, evaluator_source_identity = _validate_alexnet_support_environment(
        backend_admission=backend_admission, model_load_receipt=model_load_receipt,
        model=model, evaluator=evaluator, query=query, current_mask=current_mask,
        route_axis=route_axis, native_fiber_axis=native_fiber_axis, input_box=input_box,
        model_identity_hash=population_registration.model_identity_hash)
    if (
        population_registration.route_axis_hash != route_axis.route_axis_hash
        or population_registration.route_id_axis_hash != route_axis.route_id_axis_hash
        or population_registration.route_count != route_axis.route_count
        or population_registration.current_mask_hash != _direct_mask_hash(current_mask)
        or current_mask.route_axis_hash != route_axis.route_axis_hash
        or current_mask.route_count != route_axis.route_count
    ):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.CHART_MISMATCH,
            "population chart differs from mask or declared route custody",
        )
    acquisition = _validate_fq_acquisition_custody(
        fq_acquisition_plan=fq_acquisition_plan, backend_admission=backend_admission,
        model_load_receipt=model_load_receipt, query=query,
        population_registration=population_registration, input_box=input_box)
    backend_execution = _payload_hash(
        {
            "schema": "alexnet_literal_j_backend_execution_owner_v1",
            "backend_admission_hash": backend_admission.admission_hash,
            "fq_acquisition_plan_hash": acquisition.fq_acquisition_plan_hash,
            "acquisition_alpha_closure_hash": (
                acquisition.acquisition_alpha_closure_hash
            ),
            "train_dataset_identity_hash": (acquisition.train_dataset_identity_hash),
            "train_split_identity_hash": acquisition.train_split_identity_hash,
            "source_role_registration_hash": (
                acquisition.source_role_registration_hash
            ),
            "evaluator_source_identity_hash": evaluator_source_identity,
            "population_projection_registration_hash": population_registration.registration_hash,
            "current_mask_hash": population_registration.current_mask_hash,
            "proposal_generation": population_registration.proposal_generation,
        },
        role="backend-execution",
    )
    return (
        layout,
        state_hash,
        evaluator_source_identity,
        backend_execution,
        acquisition,
    )


def build_v1_alexnet_literal_j_analytic_support_source_cuda(
    *,
    fq_acquisition_plan: "V1AlexNetFQAcquisitionPlan",
    backend_admission: V1AlexNetBackendAdmission,
    model_load_receipt: ModelLoadReceipt,
    model: Any,
    evaluator: TorchCircuitEvaluator,
    query: QuerySpec,
    population_registration: V1LiteralJDFixedResidualPopulationProjectionRegistration,
    current_mask: HardMask,
    route_axis: DeclaredRouteAxis,
    native_fiber_axis: V1NativeDestinationFiberAxisReceipt,
    input_box: _envelope.V1RegisteredNormalizedInputBox,
    direction_custody: V1LiteralJPrivatePrimalDirectionCUDAHandle,
    route_output_row_chunk: int = _DEFAULT_ROUTE_OUTPUT_ROW_CHUNK,
    affine_output_row_chunk: int = _DEFAULT_AFFINE_OUTPUT_ROW_CHUNK,
    hbm_ceiling_fraction: float = _DEFAULT_HBM_CEILING_FRACTION,
) -> V1AlexNetLiteralJAnalyticSupportSourceResult:
    """Build guarded production CUDA support from exact AlexNet owners only."""

    if torch is None or nn is None or F is None:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.DEVICE_REQUIRED,
            "AlexNet literal-J support requires PyTorch",
        )
    if type(direction_custody) is not V1LiteralJPrivatePrimalDirectionCUDAHandle:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.INVALID_TYPE,
            "direction_custody must be the exact live KKT CUDA handle",
        )
    _assert_dependency_owners()
    (
        layout,
        state_hash,
        evaluator_source_identity,
        backend_execution,
        acquisition,
    ) = _validate_production_owners(
        fq_acquisition_plan=fq_acquisition_plan,
        backend_admission=backend_admission,
        model_load_receipt=model_load_receipt,
        model=model,
        evaluator=evaluator,
        query=query,
        population_registration=population_registration,
        current_mask=current_mask,
        route_axis=route_axis,
        native_fiber_axis=native_fiber_axis,
        input_box=input_box,
    )
    device = next(model.parameters()).device
    if (
        device.type != "cuda"
        or evaluator.device != device
        or not torch.cuda.is_available()
    ):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.DEVICE_REQUIRED,
            "production AlexNet literal-J support has no CPU fallback",
        )
    if not torch.are_deterministic_algorithms_enabled():
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.DETERMINISM_DISABLED,
            "deterministic PyTorch algorithms must be enabled before support derivation",
        )
    if bool(model.training) or any(
        parameter.device != device for parameter in model.parameters()
    ):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.DEVICE_REQUIRED,
            "AlexNet model must remain eval-mode on one CUDA device",
        )
    route_rows = _positive_int(route_output_row_chunk, name="route_output_row_chunk")
    affine_rows = _positive_int(affine_output_row_chunk, name="affine_output_row_chunk")
    free, total = torch.cuda.mem_get_info(device)
    forecast = build_v1_alexnet_literal_j_support_hbm_forecast(
        model=model,
        route_axis=route_axis,
        input_box=input_box,
        device_identity=_cuda_device_identity(device),
        device_free_memory_bytes=int(free),
        _available_memory_bytes=cuda_reusable_memory_bytes(device, framework=torch, free_bytes=free),
        device_total_memory_bytes=int(total),
        current_allocated_device_bytes=int(torch.cuda.memory_allocated(device)),
        current_reserved_device_bytes=int(torch.cuda.memory_reserved(device)),
        hbm_ceiling_fraction=hbm_ceiling_fraction,
    )
    if not forecast.admitted:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.HBM_FORECAST_EXCEEDED,
            "analytic support derivation exceeds the registered HBM envelope",
        )

    opening = (
        acquisition.fq_acquisition_plan_hash,
        acquisition.acquisition_alpha_closure_hash,
        backend_admission.admission_hash,
        model_load_receipt.content_hash,
        state_hash,
        evaluator_source_identity,
        query.content_hash,
        population_registration.registration_hash,
        _direct_mask_hash(current_mask),
        route_axis.content_hash,
        native_fiber_axis.receipt_hash,
        input_box.box_hash,
    )
    owner_identity = v1_alexnet_literal_j_support_owner_identity_hash()
    direction_lease = direction_custody.borrow_for_literal_j_projection()
    lease_transferred = False
    try:
        descriptor_axis = direction_lease.descriptor_axis
        directions = direction_lease.directions_for_literal_j_projection()
        if (
            descriptor_axis.descriptor_axis_hash
            != population_registration.private_primal_direction_axis_hash
            or descriptor_axis.direction_ids != population_registration.direction_ids
            or descriptor_axis.route_axis_hash != route_axis.route_axis_hash
            or descriptor_axis.route_count != route_axis.route_count
            or directions.device != device
            or bool((directions < 0.0).any().item())
            or bool((directions > 1.0).any().item())
        ):
            _refuse(
                V1AlexNetLiteralJSupportRefusalCode.DIRECTION_AXIS_MISMATCH,
                "private-primal custody differs from the population chart",
            )
        numeric, _possible_active = _derive_numeric_support(
            model=model,
            route_axis=route_axis,
            current_mask=current_mask,
            query=query,
            input_box=input_box,
            directions=directions,
            route_output_row_chunk=route_rows,
            affine_output_row_chunk=affine_rows,
        )
        route_sum_values = tuple(
            float(item)
            for item in numeric.route_sum_projection_abs_bounds.detach().cpu().tolist()
        )
        direct_values = tuple(
            float(item)
            for item in numeric.direct_projection_abs_bounds.detach().cpu().tolist()
        )
        selected_values = tuple(
            float(item)
            for item in numeric.selected_projection_abs_bounds.detach().cpu().tolist()
        )
        source_hash = _payload_hash(
            {
                "schema": "alexnet_literal_j_route_bound_source_v1",
                "owner_identity_hash": owner_identity,
                "population_registration_hash": population_registration.registration_hash,
                "backend_execution_receipt_hash": backend_execution,
                "input_box_hash": input_box.box_hash,
                "fq_acquisition_plan_hash": acquisition.fq_acquisition_plan_hash,
                "train_dataset_identity_hash": (
                    acquisition.train_dataset_identity_hash
                ),
                "preprocessing_hash": acquisition.preprocessing_hash,
                "native_fiber_receipt_hash": native_fiber_axis.receipt_hash,
                "equation": ALEXNET_LITERAL_J_ROUTE_BOUND_EQUATION,
            },
            role="route-bound-source",
        )
        direct_hash = _payload_hash(
            {
                "schema": "alexnet_literal_j_direct_private_primal_bound_v1",
                "owner_identity_hash": owner_identity,
                "population_registration_hash": population_registration.registration_hash,
                "private_primal_direction_axis_hash": descriptor_axis.descriptor_axis_hash,
                "direction_ids": descriptor_axis.direction_ids,
                "equation": ALEXNET_LITERAL_J_DIRECT_BOUND_EQUATION,
                "within_destination_signed_cancellation_before_magnitude": True,
            },
            role="direct-bound-derivation",
        )
        receipt = V1AlexNetLiteralJSupportDerivationReceipt(
            population_projection_registration_hash=population_registration.registration_hash,
            fq_acquisition_plan_hash=acquisition.fq_acquisition_plan_hash,
            master_alpha_registration_hash=(acquisition.master_alpha_registration_hash),
            campaign_contract_hash=acquisition.campaign_contract_hash,
            controller_run_hash=acquisition.controller_run_hash,
            train_dataset_identity_hash=acquisition.train_dataset_identity_hash,
            train_split_identity_hash=acquisition.train_split_identity_hash,
            source_role_registration_hash=acquisition.source_role_registration_hash,
            preprocessing_hash=acquisition.preprocessing_hash,
            acquisition_alpha_closure_hash=(acquisition.acquisition_alpha_closure_hash),
            backend_admission_hash=backend_admission.admission_hash,
            model_load_receipt_hash=model_load_receipt.content_hash,
            model_identity_hash=population_registration.model_identity_hash,
            model_state_hash=state_hash,
            evaluator_id=evaluator.evaluator_id,
            evaluator_parity_receipt_hash=evaluator.parity_receipt_hash,
            evaluator_source_identity_hash=evaluator_source_identity,
            query_contract_hash=query.content_hash,
            current_mask_hash=population_registration.current_mask_hash,
            proposal_generation=population_registration.proposal_generation,
            route_axis_hash=route_axis.route_axis_hash,
            route_id_axis_hash=route_axis.route_id_axis_hash,
            route_axis_content_hash=route_axis.content_hash,
            route_count=route_axis.route_count,
            native_fiber_axis_receipt_hash=native_fiber_axis.receipt_hash,
            input_box_hash=input_box.box_hash,
            population_alpha_ledger_hash=population_registration.alpha_ledger_hash,
            population_alpha_coordinate_hash=(
                population_registration.alpha_coordinate_hash
            ),
            population_alpha_lineage_hash=(population_registration.alpha_lineage_hash),
            fq_generation_alpha_allocation=(acquisition.fq_generation_alpha_allocation),
            population_familywise_alpha=population_registration.familywise_alpha,
            private_primal_direction_axis_hash=descriptor_axis.descriptor_axis_hash,
            direction_ids=descriptor_axis.direction_ids,
            source_owner_identity_hash=owner_identity,
            backend_execution_receipt_hash=backend_execution,
            literal_j_abs_bound_source_hash=source_hash,
            direct_projection_abs_bound_derivation_receipt_hash=direct_hash,
            hbm_forecast_hash=forecast.forecast_hash,
            span_receipts=numeric.span_receipts,
            route_sum_projection_abs_bounds=route_sum_values,  # type: ignore[arg-type]
            direct_projection_abs_bounds=direct_values,  # type: ignore[arg-type]
            selected_projection_abs_bounds=selected_values,  # type: ignore[arg-type]
            _issuance_token=_RECEIPT_TOKEN,
        )
        generic = _issue_v1_literal_j_residual_analytic_support_source_cuda_from_owner(
            owner_token=_ANALYTIC_SUPPORT_SOURCE_OWNER_TOKEN,
            analytic_bound_owner_receipt=receipt,
            population_registration=population_registration,
            direction_lease=direction_lease,
            literal_j_abs_bound=numeric.route_abs_bound,
            direct_projection_abs_bounds=numeric.direct_projection_abs_bounds,
        )
        lease_transferred = True
    except BaseException:
        if not lease_transferred:
            direction_lease.close()
        raise

    try:
        closing_state = state_dict_digest(model)
        closing_native = build_alexnet_v1_native_destination_fiber_axis(
            evaluator, route_axis
        )
        closing = (
            fq_acquisition_plan.assert_integrity(),
            acquisition.acquisition_alpha_closure_hash,
            backend_admission.admission_hash,
            model_load_receipt.content_hash,
            closing_state,
            evaluator_source_identity,
            query.content_hash,
            population_registration.registration_hash,
            _direct_mask_hash(current_mask),
            route_axis.content_hash,
            closing_native.receipt_hash,
            input_box.assert_integrity(),
        )
    except BaseException:
        generic.source_handle.evict()
        raise
    if opening != closing:
        generic.source_handle.evict()
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.SOURCE_MUTATED,
            "AlexNet support source changed during analytic derivation",
        )
    try:
        return V1AlexNetLiteralJAnalyticSupportSourceResult(
            derivation_receipt=receipt,
            hbm_forecast=forecast,
            analytic_support_source=generic,
        )
    except BaseException:
        generic.source_handle.evict()
        raise


def _derive_v1_alexnet_literal_j_support_bounds_cpu_reference(
    *,
    model: Any,
    route_axis: DeclaredRouteAxis,
    current_mask: HardMask,
    query: QuerySpec,
    input_box: _envelope.V1RegisteredNormalizedInputBox,
    directions: Any,
    route_output_row_chunk: int = 16,
    affine_output_row_chunk: int = 16,
) -> _V1AlexNetLiteralJSupportCPUReferenceResult:
    """Small exact-graph CPU oracle; never emits production source custody."""

    if torch is None or nn is None or F is None:
        raise RuntimeError("private AlexNet literal-J reference requires PyTorch")
    if (
        type(route_axis) is not DeclaredRouteAxis
        or type(current_mask) is not HardMask
        or type(query) is not QuerySpec
        or type(input_box) is not _envelope.V1RegisteredNormalizedInputBox
    ):
        raise TypeError(
            "private literal-J reference requires exact route/mask/query/box"
        )
    input_box.assert_integrity()
    if (
        current_mask.route_axis_hash != route_axis.route_axis_hash
        or current_mask.route_count != route_axis.route_count
    ):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.MASK_MISMATCH,
            "private literal-J reference mask differs from its declared route axis",
        )
    if route_axis.route_count > _CPU_REFERENCE_ROUTE_LIMIT:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.AUTHORITY_VIOLATION,
            "private CPU literal-J reference refuses route-scale execution",
        )
    if next(model.parameters()).device.type != "cpu" or bool(model.training):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.DEVICE_REQUIRED,
            "private literal-J reference requires an eval-mode CPU model",
        )
    z = _require_tensor(
        directions,
        name="private CPU directions",
        shape=(2, route_axis.route_count),
    )
    if z.device.type != "cpu":
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.DEVICE_REQUIRED,
            "private literal-J reference directions must remain on CPU",
        )
    if bool((z < 0.0).any().item()) or bool((z > 1.0).any().item()):
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.DIRECTION_AXIS_MISMATCH,
            "private-primal directions must lie in [0,1]",
        )
    opening_state = state_dict_digest(model)
    numeric, possible_active = _derive_numeric_support(
        model=model,
        route_axis=route_axis,
        current_mask=current_mask,
        query=query,
        input_box=input_box,
        directions=z,
        route_output_row_chunk=_positive_int(
            route_output_row_chunk, name="route_output_row_chunk"
        ),
        affine_output_row_chunk=_positive_int(
            affine_output_row_chunk, name="affine_output_row_chunk"
        ),
    )
    if state_dict_digest(model) != opening_state:
        _refuse(
            V1AlexNetLiteralJSupportRefusalCode.SOURCE_MUTATED,
            "CPU reference model changed during derivation",
        )
    return _V1AlexNetLiteralJSupportCPUReferenceResult(
        route_abs_bound=numeric.route_abs_bound,
        route_sum_projection_abs_bounds=numeric.route_sum_projection_abs_bounds,
        direct_projection_abs_bounds=numeric.direct_projection_abs_bounds,
        selected_projection_abs_bounds=numeric.selected_projection_abs_bounds,
        possible_active_by_layer=MappingProxyType(dict(possible_active)),
        span_receipts=numeric.span_receipts,
    )


__all__ = [
    "ALEXNET_LITERAL_J_DIRECT_BOUND_EQUATION",
    "ALEXNET_LITERAL_J_RELU_KINK_POLICY",
    "ALEXNET_LITERAL_J_ROUTE_BOUND_EQUATION",
    "ALEXNET_LITERAL_J_SUPPORT_ALGORITHM",
    "ALEXNET_LITERAL_J_SUPPORT_AUTHORITY",
    "V1AlexNetLiteralJAnalyticSupportSourceResult",
    "V1AlexNetLiteralJSupportDerivationReceipt",
    "V1AlexNetLiteralJSupportHBMForecast",
    "V1AlexNetLiteralJSupportRefusal",
    "V1AlexNetLiteralJSupportRefusalCode",
    "V1AlexNetLiteralJSupportSpanReceipt",
    "build_v1_alexnet_literal_j_analytic_support_source_cuda",
    "build_v1_alexnet_literal_j_support_hbm_forecast",
    "v1_alexnet_literal_j_support_owner_identity_hash",
]
