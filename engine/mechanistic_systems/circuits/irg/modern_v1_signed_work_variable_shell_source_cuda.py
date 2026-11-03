"""Authority-inert source representation for a frozen variable KKT shell.

The missing D-side solution-sensitivity theorem needs to evaluate a *variable*
bounded shell after that shell has been frozen.  Reopening or retaining a
``context x route`` bank would defeat the production geometry.  This module
therefore consumes predictable same-capture ``D_W2`` corrected-evidence and
literal-J chunks and immediately lowers them to

``b_f(x), A_B(x), d(x), j_B(x)``

such that, for every shell gate vector ``z_B``, the retained source state
evaluates exactly (up to enclosed binary64 arithmetic)

``F_B(z_B) = E_x [ (1 / cuts) sum_f |b_f(x) + A_{B,f}(x) z_B| ]``

and

``Q_B(z_B) = E_x [ (d(x) + j_B(x)^T z_B)^2 ]``.

``A_B`` is stored sparsely as one corrected coefficient and one native-fiber
ordinal per shell route.  The only context-by-fiber tensor is the fixed-
outside offset ``b``.  Route rows are paired, transient, CUDA-resident source
inputs; route-to-fiber lowering uses the compact cut records and never creates
route-scale fiber labels or performs route-scale D2H.

This is deliberately a source-representation joint, not a statistical or
optimization theorem.  Its durable receipt is array-free and every C/K/O,
commit, terminal, Pareto, uniform-confidence, normal-cone, and action-
invariance authority flag is closed.  A CPU mode and an exact rational
evaluator exist only for tiny reference tests.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral, Real
import os
from typing import Any, Final, NoReturn, Sequence

from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .contracts import content_hash
from .modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt

try:  # pragma: no cover - availability depends on the runtime image.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


VARIABLE_SHELL_SOURCE_DESIGN_SCHEMA: Final = (
    "partition_free_v1_signed_work_variable_shell_source_cuda_design_v1"
)
VARIABLE_SHELL_SOURCE_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_signed_work_variable_shell_source_registration_v1"
)
VARIABLE_SHELL_SOURCE_HBM_SCHEMA: Final = (
    "partition_free_v1_signed_work_variable_shell_source_hbm_v1"
)
VARIABLE_SHELL_SOURCE_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_signed_work_variable_shell_source_receipt_v2"
)
VARIABLE_SHELL_SOURCE_TENSOR_LEASE_SCHEMA: Final = (
    "partition_free_v1_signed_work_variable_shell_source_tensor_lease_v1"
)
VARIABLE_SHELL_SOURCE_CHUNK_BINDING_SCHEMA: Final = (
    "partition_free_v1_signed_work_variable_shell_source_chunk_binding_v1"
)
VARIABLE_SHELL_EVALUATION_SCHEMA: Final = (
    "partition_free_v1_signed_work_variable_shell_evaluation_v1"
)
VARIABLE_SHELL_SOURCE_ALGORITHM: Final = (
    "post_shell_freeze_same_capture_D_W2_chunk_lowering_to_native_fiber_"
    "outside_offsets_and_bounded_shell_coefficients_fp64_v1"
)
VARIABLE_SHELL_SOURCE_AUTHORITY_SCOPE: Final = (
    "source_representation_and_sample_function_evaluation_only_no_uniform_"
    "population_confidence_solution_sensitivity_action_invariance_C_K_O_"
    "commit_terminal_or_pareto_authority_v1"
)
VARIABLE_SHELL_F_EQUATION: Final = (
    "E_context_sum_cut_uniform_probability_sum_native_fiber_abs_b_f_plus_"
    "sum_shell_route_in_fiber_A_j_z_j_v1"
)
VARIABLE_SHELL_Q_EQUATION: Final = (
    "E_context_square_d_plus_literal_J_shell_dot_z_shell_v1"
)
VARIABLE_SHELL_MEMORY_SCOPE: Final = (
    "registered_reserved_device_bytes_plus_compact_context_by_fiber_outside_"
    "offsets_plus_two_context_by_shell_coefficient_banks_plus_context_"
    "outside_J_and_weights_plus_one_route_fixed_outside_vector_plus_one_active_"
    "compact_packet_plus_two_source_owned_route_chunks_plus_all_chunk_lowering_"
    "temporaries_or_tiled_arbitrary_shell_evaluation_whichever_is_larger_v3"
)

_PREFIX: Final = "partition-free-v1-variable-shell-source-"
_F64_BYTES: Final = 8
_I64_BYTES: Final = 8
_F64_EPS: Final = 2.0**-52
_F64_MIN_SUBNORMAL: Final = math.nextafter(0.0, math.inf)
_MAXIMUM_CANONICAL_SHELL: Final = 8
_ISSUER_TOKEN: Final = object()


class V1SignedWorkVariableShellExecutionMode(str, Enum):
    CUDA_PRODUCTION = "cuda_production"
    CPU_REFERENCE_TEST_ONLY = "cpu_reference_test_only"


class V1SignedWorkVariableShellSourceRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    INVALID_DESIGN = "invalid_design"
    STALE_IDENTITY = "stale_identity"
    AXIS_MISMATCH = "axis_mismatch"
    ORDER_MISMATCH = "order_mismatch"
    SHELL_TOO_LARGE = "shell_too_large"
    DEVICE_REQUIRED = "device_required"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    HBM_FORECAST_EXCEEDED = "hbm_forecast_exceeded"
    NONFINITE_SOURCE = "nonfinite_source"
    PARTIAL_PACKET = "partial_packet"
    DUPLICATE_PACKET = "duplicate_packet"
    MUTATED_CUSTODY = "mutated_custody"
    EVICTED = "evicted"
    ACTIVE_PACKET = "active_packet"
    ACTIVE_TENSOR_LEASE = "active_tensor_lease"
    AUTHORITY_VIOLATION = "authority_violation"
    NUMERICAL_REFUSAL = "numerical_refusal"
    DETERMINISM_DISABLED = "determinism_disabled"


class V1SignedWorkVariableShellSourceRefusal(RuntimeError):
    """Typed, fail-closed refusal for bounded-shell source custody."""

    def __init__(
        self,
        code: V1SignedWorkVariableShellSourceRefusalCode,
        message: str,
    ) -> None:
        if type(code) is not V1SignedWorkVariableShellSourceRefusalCode:
            raise TypeError("variable-shell refusal code must have exact enum type")
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1SignedWorkVariableShellSourceRefusalCode,
    message: str,
) -> NoReturn:
    raise V1SignedWorkVariableShellSourceRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _identity_axis(
    value: object,
    *,
    name: str,
    unique: bool = True,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if type(value) not in {tuple, list}:
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one ordered string sequence",
        )
    result = tuple(_identity(item, name=f"{name} item") for item in value)
    if not result and not allow_empty:
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.AXIS_MISMATCH,
            f"{name} cannot be empty",
        )
    if unique and len(result) != len(set(result)):
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.AXIS_MISMATCH,
            f"{name} must be unique",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
            f"{name} must be nonnegative",
        )
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
            f"{name} must be positive",
        )
    return result


def _finite_positive(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one finite positive real",
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
            f"{name} must be finite and positive",
        )
    return result


def _weights(value: object, *, count: int) -> tuple[float, ...]:
    if type(value) not in {tuple, list} or len(value) != count:
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.AXIS_MISMATCH,
            "context weights must align with the registered context axis",
        )
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "context weights must be real scalars",
            )
        scalar = float(item)
        if not math.isfinite(scalar) or scalar < 0.0:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "context weights must be finite and nonnegative",
            )
        result.append(0.0 if scalar == 0.0 else scalar)
    if not math.isclose(
        math.fsum(result),
        1.0,
        rel_tol=0.0,
        abs_tol=128.0 * _F64_EPS,
    ):
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
            "context weights must sum to one",
        )
    return tuple(result)


def _gamma(operation_count: int) -> float:
    count = max(1, _positive_int(operation_count, name="operation count"))
    product = count * _F64_EPS
    if product >= 0.25:
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.NUMERICAL_REFUSAL,
            "binary64 operation depth is outside the registered gamma regime",
        )
    return math.nextafter(product / (1.0 - product), math.inf)


def _up(value: float) -> float:
    if not math.isfinite(value):
        return value
    return math.nextafter(value, math.inf)


def _down(value: float) -> float:
    if not math.isfinite(value):
        return value
    return math.nextafter(value, -math.inf)


def _tensor_invariant(value: Any) -> tuple[object, ...]:
    return (
        id(value),
        int(value._version),
        int(value.data_ptr()),
        tuple(int(item) for item in value.shape),
        str(value.dtype),
        str(value.device),
    )


def _up_tensor(value: Any) -> Any:
    return torch.nextafter(
        value,
        torch.full((), math.inf, dtype=torch.float64, device=value.device),
    )


def _require_fp64_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    device: object | None = None,
    finite: bool = True,
) -> Any:
    if torch is None or type(value) is not torch.Tensor:
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
            f"{name} must have exact torch.Tensor type",
        )
    if (
        value.dtype != torch.float64
        or tuple(int(item) for item in value.shape) != shape
        or not value.is_contiguous()
        or bool(value.requires_grad)
    ):
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
            f"{name} must be detached contiguous float64 with shape {shape}",
        )
    if device is not None and value.device != device:
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.STALE_IDENTITY,
            f"{name} is on another device",
        )
    if finite and not bool(torch.isfinite(value).all().item()):
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.NONFINITE_SOURCE,
            f"{name} contains nonfinite source material",
        )
    return value


def _axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_variable_shell_registered_axis_v1",
            "role": role,
            "values": tuple(values),
        },
        prefix=_PREFIX + "axis-",
    )


def v1_variable_shell_population_axis_hash(role: str, values: Sequence[str]) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_variable_shell_canonical_population_axis_v1",
            "role": _identity(role, name="population axis role"),
            "values": tuple(_identity_axis(values, name=f"{role} population axis")),
        },
        prefix=_PREFIX + "population-axis-",
    )


def _finite_census_axis_hash(
    registration: "V1SignedWorkVariableShellSourceRegistration",
) -> str:
    return content_hash(
        {
            "schema": "partition_free_v1_variable_shell_finite_census_axis_custody_v1",
            "registration_hash": registration.registration_hash,
            "population_count": registration.population_count,
            "registered_count": registration.context_count,
            "count_complete": registration.count_complete,
            "finite_census_axis_custody_complete": (
                registration.finite_census_axis_custody_complete
            ),
            "finite_census_complete": registration.finite_census_complete,
            "population_context_axis_hash": registration.population_context_axis_hash,
            "population_draw_axis_hash": registration.population_draw_axis_hash,
            "canonical_population_context_axis_hash": (
                v1_variable_shell_population_axis_hash(
                    "context", registration.registered_context_ids
                )
            ),
            "canonical_population_draw_axis_hash": (
                v1_variable_shell_population_axis_hash(
                    "draw", registration.registered_draw_ids
                )
            ),
            "registered_context_axis_hash": _axis_hash(
                "context", registration.registered_context_ids
            ),
            "registered_draw_axis_hash": _axis_hash(
                "draw", registration.registered_draw_ids
            ),
            "registered_measurement_axis_hash": _axis_hash(
                "measurement", registration.registered_measurement_ids
            ),
            "registered_capture_axis_hash": _axis_hash(
                "capture", registration.registered_capture_ids
            ),
            "registered_source_packet_axis_hash": _axis_hash(
                "source_packet", registration.registered_source_packet_hashes
            ),
            "census_claim_requires_canonical_population_axis_equality": True,
        },
        prefix=_PREFIX + "finite-census-axis-",
    )


def _tensor_source_identity_hash(value: object, *, name: str) -> str:
    if torch is None or type(value) is not torch.Tensor:
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
            f"{name} must be one live torch tensor",
        )
    tensor = value
    return content_hash(
        {
            "schema": "partition_free_v1_variable_shell_source_local_tensor_identity_v1",
            "name": name,
            "shape": tuple(int(item) for item in tensor.shape),
            "stride": tuple(int(item) for item in tensor.stride()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "data_ptr": int(tensor.data_ptr()),
            "storage_offset": int(tensor.storage_offset()),
            "version": int(getattr(tensor, "_version", -1)),
            "python_object_id": int(id(tensor)),
            "source_local_identity_only": True,
            "upstream_dual_stream_custody_authority": False,
        },
        prefix=_PREFIX + "tensor-identity-",
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSourceCUDAExecutionDesign:
    route_chunk_size: int = 131_072
    maximum_batch_size: int = 8
    maximum_shell_size: int = _MAXIMUM_CANONICAL_SHELL
    hbm_ceiling_fraction: float = 0.85
    execution_mode: V1SignedWorkVariableShellExecutionMode = (
        V1SignedWorkVariableShellExecutionMode.CUDA_PRODUCTION
    )
    deterministic_fp64_required: bool = True
    forbid_route_scale_row_retention: bool = True
    forbid_route_scale_d2h: bool = True
    forbid_route_scale_fiber_labels: bool = True
    schema: str = VARIABLE_SHELL_SOURCE_DESIGN_SCHEMA
    algorithm_id: str = VARIABLE_SHELL_SOURCE_ALGORITHM

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_chunk_size",
            _positive_int(self.route_chunk_size, name="route chunk size"),
        )
        object.__setattr__(
            self,
            "maximum_batch_size",
            _positive_int(self.maximum_batch_size, name="maximum batch size"),
        )
        maximum_shell = _positive_int(
            self.maximum_shell_size,
            name="maximum shell size",
        )
        if maximum_shell > _MAXIMUM_CANONICAL_SHELL:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.SHELL_TOO_LARGE,
                "canonical variable-shell source representation is bounded by B=8",
            )
        object.__setattr__(self, "maximum_shell_size", maximum_shell)
        ceiling = _finite_positive(
            self.hbm_ceiling_fraction,
            name="HBM ceiling fraction",
        )
        if ceiling > 0.85:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "canonical HBM ceiling cannot exceed 0.85",
            )
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        if type(self.execution_mode) is not V1SignedWorkVariableShellExecutionMode:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "execution mode must have exact enum type",
            )
        flags = (
            self.deterministic_fp64_required,
            self.forbid_route_scale_row_retention,
            self.forbid_route_scale_d2h,
            self.forbid_route_scale_fiber_labels,
        )
        if any(type(item) is not bool for item in flags) or any(
            item is not True for item in flags
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "deterministic FP64 and all route-scale prohibitions are mandatory",
            )
        if (
            self.schema != VARIABLE_SHELL_SOURCE_DESIGN_SCHEMA
            or self.algorithm_id != VARIABLE_SHELL_SOURCE_ALGORITHM
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "unsupported variable-shell source design",
            )

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "design-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSourceRegistration:
    registration_id: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    parent_d_source_receipt_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    kkt_branch_id: str
    kkt_branch_source_receipt_hash: str
    resource_axis_hash: str
    resource_source_receipt_hash: str
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    alpha_ledger_id: str
    alpha_coordinate_id: str
    alpha_lineage_hash: str
    population_count: int
    population_context_axis_hash: str
    population_draw_axis_hash: str
    registered_context_ids: tuple[str, ...]
    registered_draw_ids: tuple[str, ...]
    registered_measurement_ids: tuple[str, ...]
    registered_capture_ids: tuple[str, ...]
    registered_source_packet_hashes: tuple[str, ...]
    context_weights: tuple[float, ...]
    shell_frozen_before_d_w2_outcomes: bool = True
    predictable_without_replacement_prefix: bool = True
    same_capture_dual_stream_required: bool = True
    alpha_coordinate_nonrecyclable: bool = True
    schema: str = VARIABLE_SHELL_SOURCE_REGISTRATION_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "registration_id",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "parent_d_source_receipt_hash",
            "shell_freeze_receipt_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "kkt_branch_id",
            "kkt_branch_source_receipt_hash",
            "resource_axis_hash",
            "resource_source_receipt_hash",
            "witness_stream_source_hash",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
            "alpha_lineage_hash",
            "population_context_axis_hash",
            "population_draw_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "proposal_generation",
            _nonnegative_int(self.proposal_generation, name="proposal generation"),
        )
        population = _positive_int(self.population_count, name="population count")
        object.__setattr__(self, "population_count", population)
        if type(self.shell_indices) not in {tuple, list}:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "shell indices must be one ordered integer sequence",
            )
        indices = tuple(
            _nonnegative_int(item, name="shell route index")
            for item in self.shell_indices
        )
        if (
            not indices
            or len(indices) > _MAXIMUM_CANONICAL_SHELL
            or len(indices) != len(set(indices))
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.SHELL_TOO_LARGE,
                "the frozen shell must contain one through eight unique routes",
            )
        object.__setattr__(self, "shell_indices", indices)
        shell_ids = _identity_axis(self.shell_route_ids, name="shell route IDs")
        if len(shell_ids) != len(indices):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AXIS_MISMATCH,
                "shell route IDs must align with shell indices",
            )
        object.__setattr__(self, "shell_route_ids", shell_ids)
        axes: dict[str, tuple[str, ...]] = {}
        for name in (
            "registered_context_ids",
            "registered_draw_ids",
            "registered_measurement_ids",
            "registered_capture_ids",
            "registered_source_packet_hashes",
        ):
            axis = _identity_axis(getattr(self, name), name=name)
            object.__setattr__(self, name, axis)
            axes[name] = axis
        lengths = {len(item) for item in axes.values()}
        if len(lengths) != 1 or next(iter(lengths)) > population:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AXIS_MISMATCH,
                "all registered D_W2 axes must have one prefix length no larger than the population",
            )
        object.__setattr__(
            self,
            "context_weights",
            _weights(self.context_weights, count=next(iter(lengths))),
        )
        flags = (
            self.shell_frozen_before_d_w2_outcomes,
            self.predictable_without_replacement_prefix,
            self.same_capture_dual_stream_required,
            self.alpha_coordinate_nonrecyclable,
        )
        if any(type(item) is not bool for item in flags) or any(
            item is not True for item in flags
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "shell freeze, predictable prefix, same capture, and nonrecyclable alpha lineage are mandatory",
            )
        if self.schema != VARIABLE_SHELL_SOURCE_REGISTRATION_SCHEMA:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "unsupported variable-shell source registration",
            )

    @property
    def context_count(self) -> int:
        return len(self.registered_context_ids)

    @property
    def shell_size(self) -> int:
        return len(self.shell_indices)

    @property
    def count_complete(self) -> bool:
        return self.context_count == self.population_count

    @property
    def finite_census_axis_custody_complete(self) -> bool:
        return (
            self.count_complete
            and self.population_context_axis_hash
            == v1_variable_shell_population_axis_hash(
                "context", self.registered_context_ids
            )
            and self.population_draw_axis_hash
            == v1_variable_shell_population_axis_hash("draw", self.registered_draw_ids)
        )

    @property
    def finite_census_complete(self) -> bool:
        return self.count_complete and self.finite_census_axis_custody_complete

    @property
    def registration_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "registration-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSourceChunkBinding:
    """Typed identity for one route chunk of one active same-capture packet."""

    registration_hash: str
    active_packet_ordinal: int
    active_packet_context_axis_hash: str
    active_packet_draw_axis_hash: str
    active_packet_measurement_axis_hash: str
    active_packet_capture_axis_hash: str
    active_packet_source_packet_axis_hash: str
    route_start: int
    route_stop: int
    route_axis_hash: str
    route_id_axis_hash: str
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    corrected_evidence_tensor_identity_hash: str
    literal_j_tensor_identity_hash: str
    same_capture_corrected_and_literal_rows: bool = True
    source_local_tensor_identity_only: bool = True
    upstream_dual_stream_custody_authority: bool = False
    predictable_before_chunk_outcomes: bool = True
    schema: str = VARIABLE_SHELL_SOURCE_CHUNK_BINDING_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "chunk bindings must be issued by the active variable-shell accumulator",
            )
        for name in (
            "registration_hash",
            "active_packet_context_axis_hash",
            "active_packet_draw_axis_hash",
            "active_packet_measurement_axis_hash",
            "active_packet_capture_axis_hash",
            "active_packet_source_packet_axis_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "witness_stream_source_hash",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "corrected_evidence_tensor_identity_hash",
            "literal_j_tensor_identity_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "active_packet_ordinal",
            _nonnegative_int(self.active_packet_ordinal, name="active packet ordinal"),
        )
        start = _nonnegative_int(self.route_start, name="chunk route start")
        stop = _positive_int(self.route_stop, name="chunk route stop")
        if start >= stop:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AXIS_MISMATCH,
                "chunk binding route interval is empty or reversed",
            )
        object.__setattr__(self, "route_start", start)
        object.__setattr__(self, "route_stop", stop)
        if (
            type(self.same_capture_corrected_and_literal_rows) is not bool
            or type(self.source_local_tensor_identity_only) is not bool
            or type(self.upstream_dual_stream_custody_authority) is not bool
            or type(self.predictable_before_chunk_outcomes) is not bool
            or self.same_capture_corrected_and_literal_rows is not True
            or self.source_local_tensor_identity_only is not True
            or self.upstream_dual_stream_custody_authority is not False
            or self.predictable_before_chunk_outcomes is not True
            or self.schema != VARIABLE_SHELL_SOURCE_CHUNK_BINDING_SCHEMA
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AUTHORITY_VIOLATION,
                "chunk binding lost same-capture, source-local, or predictable identity",
            )

    @property
    def binding_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "chunk-binding-",
        )

    @property
    def durable_binding_hash(self) -> str:
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
                and name
                not in {
                    "corrected_evidence_tensor_identity_hash",
                    "literal_j_tensor_identity_hash",
                }
            }
            | {
                "source_local_tensor_identity_checked_at_consume_time": True,
                "source_local_tensor_identity_not_durable_replay_authority": True,
            },
            prefix=_PREFIX + "durable-chunk-binding-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSourceCUDAHBMForecast:
    registration_hash: str
    design_hash: str
    route_count: int
    fiber_count: int
    context_count: int
    shell_size: int
    route_chunk_size: int
    maximum_batch_size: int
    persistent_representation_bytes: int
    construction_fixed_outside_bytes: int
    active_compact_packet_bytes: int
    source_colive_route_chunk_bytes: int
    reduction_workspace_bytes: int
    evaluation_workspace_bytes: int
    incremental_peak_bytes: int
    current_reserved_device_bytes: int
    projected_peak_reserved_bytes: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    hbm_ceiling_fraction: float
    hbm_ceiling_bytes: int
    admitted: bool
    memory_scope: str = VARIABLE_SHELL_MEMORY_SCOPE
    schema: str = VARIABLE_SHELL_SOURCE_HBM_SCHEMA

    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _admission_memory: tuple[int, int] | None) -> None:
        for name in ("registration_hash", "design_hash"):
            _identity(getattr(self, name), name=name)
        for name in (
            "route_count",
            "fiber_count",
            "context_count",
            "shell_size",
            "route_chunk_size",
            "maximum_batch_size",
            "persistent_representation_bytes",
            "construction_fixed_outside_bytes",
            "active_compact_packet_bytes",
            "source_colive_route_chunk_bytes",
            "reduction_workspace_bytes",
            "evaluation_workspace_bytes",
            "incremental_peak_bytes",
            "current_reserved_device_bytes",
            "projected_peak_reserved_bytes",
            "device_total_memory_bytes",
            "device_free_memory_bytes",
            "hbm_ceiling_bytes",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name=name)
            )
        if (
            self.route_count <= 0
            or self.fiber_count <= 0
            or self.context_count <= 0
            or self.shell_size <= 0
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "HBM axis counts must be positive",
            )
        ceiling = _finite_positive(
            self.hbm_ceiling_fraction, name="HBM ceiling fraction"
        )
        object.__setattr__(self, "hbm_ceiling_fraction", ceiling)
        if type(self.admitted) is not bool:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "HBM admission must be exact bool",
            )
        expected_persistent = (
            _F64_BYTES
            * (
                2 * self.context_count * self.fiber_count
                + 2 * self.context_count * self.shell_size
                + 3 * self.context_count
            )
            + _I64_BYTES * self.shell_size
        )
        expected_fixed = _F64_BYTES * self.route_count
        expected_active = (
            _F64_BYTES
            * self.maximum_batch_size
            * (2 * self.fiber_count + 2 * self.shell_size + 2)
        )
        expected_colive = (
            2 * _F64_BYTES * self.maximum_batch_size * self.route_chunk_size
        )
        chunk_fibers = min(self.fiber_count, self.route_chunk_size)
        expected_workspace = (
            _F64_BYTES
            * self.maximum_batch_size
            * (4 * self.route_chunk_size + 4 * chunk_fibers + 6)
        )
        expected_evaluation = (
            _F64_BYTES
            * self.maximum_batch_size
            * (4 * self.fiber_count + 4 * self.shell_size + 20)
        )
        expected_construction_ephemeral = (
            expected_fixed + expected_active + expected_colive + expected_workspace
        )
        expected_incremental = expected_persistent + max(
            expected_construction_ephemeral,
            expected_evaluation,
        )
        expected_projected = self.current_reserved_device_bytes + expected_incremental
        expected_ceiling = math.floor(self.device_total_memory_bytes * ceiling)
        available, allocated = _admission_memory or (
            self.device_free_memory_bytes, self.current_reserved_device_bytes
        )
        expected_admitted = (
            allocated + expected_incremental <= expected_ceiling
            and expected_incremental <= available
        )
        observed = (
            self.persistent_representation_bytes,
            self.construction_fixed_outside_bytes,
            self.active_compact_packet_bytes,
            self.source_colive_route_chunk_bytes,
            self.reduction_workspace_bytes,
            self.evaluation_workspace_bytes,
            self.incremental_peak_bytes,
            self.projected_peak_reserved_bytes,
            self.hbm_ceiling_bytes,
            self.admitted,
        )
        expected = (
            expected_persistent,
            expected_fixed,
            expected_active,
            expected_colive,
            expected_workspace,
            expected_evaluation,
            expected_incremental,
            expected_projected,
            expected_ceiling,
            expected_admitted,
        )
        if observed != expected:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "HBM forecast does not replay its registered formula",
            )
        if (
            self.memory_scope != VARIABLE_SHELL_MEMORY_SCOPE
            or self.schema != VARIABLE_SHELL_SOURCE_HBM_SCHEMA
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "unsupported HBM forecast schema or scope",
            )

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "hbm-")


def build_v1_signed_work_variable_shell_source_cuda_hbm_forecast(
    *,
    registration: V1SignedWorkVariableShellSourceRegistration,
    native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
    design: V1SignedWorkVariableShellSourceCUDAExecutionDesign,
    device_total_memory_bytes: int,
    device_free_memory_bytes: int,
    current_reserved_device_bytes: int,
    _admission_memory: tuple[int, int] | None = None,
) -> V1SignedWorkVariableShellSourceCUDAHBMForecast:
    if (
        type(registration) is not V1SignedWorkVariableShellSourceRegistration
        or type(native_fiber_axis_receipt) is not V1NativeDestinationFiberAxisReceipt
        or type(design) is not V1SignedWorkVariableShellSourceCUDAExecutionDesign
    ):
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
            "HBM forecast requires exact registration, native-fiber, and design types",
        )
    native_fiber_axis_receipt.assert_integrity()
    if registration.shell_size > design.maximum_shell_size:
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.SHELL_TOO_LARGE,
            "registered shell exceeds the frozen execution design",
        )
    if (
        registration.native_fiber_axis_receipt_hash
        != native_fiber_axis_receipt.receipt_hash
        or registration.native_fiber_measure_hash
        != native_fiber_axis_receipt.measure_hash
        or registration.route_axis_hash != native_fiber_axis_receipt.route_axis_hash
        or registration.route_id_axis_hash
        != native_fiber_axis_receipt.route_id_axis_hash
        or any(
            index >= native_fiber_axis_receipt.route_count
            for index in registration.shell_indices
        )
    ):
        _refuse(
            V1SignedWorkVariableShellSourceRefusalCode.STALE_IDENTITY,
            "registration does not bind the supplied native-fiber route axis",
        )
    route_count = native_fiber_axis_receipt.route_count
    fiber_count = native_fiber_axis_receipt.fiber_count
    contexts = registration.context_count
    shell = registration.shell_size
    batch = design.maximum_batch_size
    chunk = design.route_chunk_size
    persistent = (
        _F64_BYTES * (2 * contexts * fiber_count + 2 * contexts * shell + 3 * contexts)
        + _I64_BYTES * shell
    )
    fixed = _F64_BYTES * route_count
    active = _F64_BYTES * batch * (2 * fiber_count + 2 * shell + 2)
    colive = 2 * _F64_BYTES * batch * chunk
    chunk_fibers = min(fiber_count, chunk)
    workspace = _F64_BYTES * batch * (4 * chunk + 4 * chunk_fibers + 6)
    evaluation = _F64_BYTES * batch * (4 * fiber_count + 4 * shell + 20)
    incremental = persistent + max(fixed + active + colive + workspace, evaluation)
    total = _positive_int(device_total_memory_bytes, name="device total memory bytes")
    free = _nonnegative_int(device_free_memory_bytes, name="device free memory bytes")
    reserved = _nonnegative_int(
        current_reserved_device_bytes, name="current reserved device bytes"
    )
    projected = reserved + incremental
    ceiling = math.floor(total * design.hbm_ceiling_fraction)
    available, allocated = _admission_memory or (free, reserved)
    admitted = allocated + incremental <= ceiling and incremental <= available
    return V1SignedWorkVariableShellSourceCUDAHBMForecast(
        registration_hash=registration.registration_hash,
        design_hash=design.design_hash,
        route_count=route_count,
        fiber_count=fiber_count,
        context_count=contexts,
        shell_size=shell,
        route_chunk_size=chunk,
        maximum_batch_size=batch,
        persistent_representation_bytes=persistent,
        construction_fixed_outside_bytes=fixed,
        active_compact_packet_bytes=active,
        source_colive_route_chunk_bytes=colive,
        reduction_workspace_bytes=workspace,
        evaluation_workspace_bytes=evaluation,
        incremental_peak_bytes=incremental,
        current_reserved_device_bytes=reserved,
        projected_peak_reserved_bytes=projected,
        device_total_memory_bytes=total,
        device_free_memory_bytes=free,
        hbm_ceiling_fraction=design.hbm_ceiling_fraction,
        hbm_ceiling_bytes=ceiling,
        admitted=admitted,
        _admission_memory=_admission_memory,
    )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellSourceReceipt:
    registration_hash: str
    design_hash: str
    hbm_forecast_hash: str
    model_identity_hash: str
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    route_axis_hash: str
    route_id_axis_hash: str
    registered_d_prefix_hash: str
    registered_d_w2_prefix_hash: str
    shell_freeze_receipt_hash: str
    shell_indices: tuple[int, ...]
    shell_route_ids: tuple[str, ...]
    shell_native_fiber_indices: tuple[int, ...]
    native_fiber_axis_receipt_hash: str
    native_fiber_measure_hash: str
    kkt_branch_id: str
    kkt_branch_source_receipt_hash: str
    resource_axis_hash: str
    resource_source_receipt_hash: str
    witness_stream_source_hash: str
    same_capture_vjp_source_hash: str
    corrected_evidence_stream_source_hash: str
    literal_j_stream_source_hash: str
    alpha_ledger_id: str
    alpha_coordinate_id: str
    alpha_lineage_hash: str
    population_context_axis_hash: str
    population_draw_axis_hash: str
    context_axis_hash: str
    draw_axis_hash: str
    measurement_axis_hash: str
    capture_axis_hash: str
    source_packet_axis_hash: str
    finite_census_axis_hash: str
    chunk_binding_axis_hash: str
    context_count: int
    population_count: int
    route_count: int
    fiber_count: int
    cut_count: int
    retained_float64_count: int
    retained_int64_count: int
    retained_tensor_bytes: int
    source_replay_hash: str
    count_complete: bool
    finite_census_complete: bool
    finite_census_axis_custody_complete: bool = False
    per_chunk_typed_identity_bound: bool = True
    source_local_paired_rows_consumed: bool = True
    upstream_dual_stream_custody_authority: bool = False
    route_scale_rows_retained: bool = False
    route_scale_d2h_performed: bool = False
    route_scale_fiber_labels_materialized: bool = False
    array_free_durable_receipt: bool = True
    statistical_uniform_confidence_authority: bool = False
    solution_sensitivity_authority: bool = False
    normal_cone_authority: bool = False
    action_invariance_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    f_equation: str = VARIABLE_SHELL_F_EQUATION
    q_equation: str = VARIABLE_SHELL_Q_EQUATION
    authority_scope: str = VARIABLE_SHELL_SOURCE_AUTHORITY_SCOPE
    schema: str = VARIABLE_SHELL_SOURCE_RECEIPT_SCHEMA
    _issuance_token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUER_TOKEN:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "variable-shell receipts must be issued by the sealed accumulator",
            )
        for name in (
            "registration_hash",
            "design_hash",
            "hbm_forecast_hash",
            "model_identity_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "registered_d_prefix_hash",
            "registered_d_w2_prefix_hash",
            "shell_freeze_receipt_hash",
            "native_fiber_axis_receipt_hash",
            "native_fiber_measure_hash",
            "kkt_branch_id",
            "kkt_branch_source_receipt_hash",
            "resource_axis_hash",
            "resource_source_receipt_hash",
            "witness_stream_source_hash",
            "same_capture_vjp_source_hash",
            "corrected_evidence_stream_source_hash",
            "literal_j_stream_source_hash",
            "alpha_ledger_id",
            "alpha_coordinate_id",
            "alpha_lineage_hash",
            "population_context_axis_hash",
            "population_draw_axis_hash",
            "context_axis_hash",
            "draw_axis_hash",
            "measurement_axis_hash",
            "capture_axis_hash",
            "source_packet_axis_hash",
            "finite_census_axis_hash",
            "chunk_binding_axis_hash",
            "source_replay_hash",
        ):
            _identity(getattr(self, name), name=name)
        if (
            type(self.shell_indices) is not tuple
            or type(self.shell_route_ids) is not tuple
            or type(self.shell_native_fiber_indices) is not tuple
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "receipt shell axes must be immutable tuples",
            )
        if not (
            1 <= len(self.shell_indices) <= _MAXIMUM_CANONICAL_SHELL
            and len(self.shell_indices)
            == len(self.shell_route_ids)
            == len(self.shell_native_fiber_indices)
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AXIS_MISMATCH,
                "receipt shell axes do not align",
            )
        for name in (
            "proposal_generation",
            "context_count",
            "population_count",
            "route_count",
            "fiber_count",
            "cut_count",
            "retained_float64_count",
            "retained_int64_count",
            "retained_tensor_bytes",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name=name)
            )
        if (
            min(
                self.context_count,
                self.population_count,
                self.route_count,
                self.fiber_count,
                self.cut_count,
            )
            <= 0
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "receipt population and geometry counts must be positive",
            )
        if type(self.count_complete) is not bool:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "count-complete marker must be an exact boolean",
            )
        if self.count_complete is not (self.context_count == self.population_count):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AUTHORITY_VIOLATION,
                "count-complete marker does not replay source coverage",
            )
        if self.finite_census_complete is not (
            self.count_complete and self.finite_census_axis_custody_complete
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AUTHORITY_VIOLATION,
                "finite-census marker must bind both count completeness and population-axis custody",
            )
        exact_true = (
            self.finite_census_axis_custody_complete,
            self.per_chunk_typed_identity_bound,
            self.source_local_paired_rows_consumed,
            self.array_free_durable_receipt,
        )
        closed = (
            self.upstream_dual_stream_custody_authority,
            self.route_scale_rows_retained,
            self.route_scale_d2h_performed,
            self.route_scale_fiber_labels_materialized,
            self.statistical_uniform_confidence_authority,
            self.solution_sensitivity_authority,
            self.normal_cone_authority,
            self.action_invariance_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(
            type(item) is not bool
            for item in exact_true
            + closed
            + (self.count_complete, self.finite_census_complete)
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "receipt authority markers must be exact booleans",
            )
        if (
            self.per_chunk_typed_identity_bound is not True
            or self.source_local_paired_rows_consumed is not True
            or self.array_free_durable_receipt is not True
            or any(item is not False for item in closed)
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AUTHORITY_VIOLATION,
                "source receipt leaked route custody or downstream authority",
            )
        if (
            self.f_equation != VARIABLE_SHELL_F_EQUATION
            or self.q_equation != VARIABLE_SHELL_Q_EQUATION
            or self.authority_scope != VARIABLE_SHELL_SOURCE_AUTHORITY_SCOPE
            or self.schema != VARIABLE_SHELL_SOURCE_RECEIPT_SCHEMA
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "unsupported source receipt semantics",
            )

    @property
    def receipt_hash(self) -> str:
        # The private issuer capability is intentionally excluded from the
        # durable content address; it is process identity, not evidence.
        return content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if not name.startswith("_")
            },
            prefix=_PREFIX + "receipt-",
        )


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellEvaluationReceipt:
    source_receipt_hash: str
    source_replay_hash: str
    shell_gate_hash: str
    shell_gates: tuple[float, ...]
    f_lower: float
    f_center: float
    f_upper: float
    q_lower: float
    q_center: float
    q_upper: float
    finite_census_complete: bool
    sample_function_evaluation_only: bool = True
    population_function_exact_except_numerical_enclosure: bool = False
    statistical_uniform_confidence_authority: bool = False
    action_invariance_authority: bool = False
    c_stream_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = VARIABLE_SHELL_SOURCE_AUTHORITY_SCOPE
    schema: str = VARIABLE_SHELL_EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        for name in ("source_receipt_hash", "source_replay_hash", "shell_gate_hash"):
            _identity(getattr(self, name), name=name)
        if type(self.shell_gates) is not tuple or not self.shell_gates:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "evaluation gates must be one nonempty immutable tuple",
            )
        for value in self.shell_gates:
            if type(value) is not float or not math.isfinite(value):
                _refuse(
                    V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                    "evaluation gates must be finite binary64 scalars",
                )
        for lower_name, center_name, upper_name in (
            ("f_lower", "f_center", "f_upper"),
            ("q_lower", "q_center", "q_upper"),
        ):
            lower = float(getattr(self, lower_name))
            center = float(getattr(self, center_name))
            upper = float(getattr(self, upper_name))
            if not (
                math.isfinite(lower)
                and math.isfinite(center)
                and math.isfinite(upper)
                and 0.0 <= lower <= center <= upper
            ):
                _refuse(
                    V1SignedWorkVariableShellSourceRefusalCode.NUMERICAL_REFUSAL,
                    "evaluation interval is nonfinite or not outward ordered",
                )
        flags = (
            self.finite_census_complete,
            self.sample_function_evaluation_only,
            self.population_function_exact_except_numerical_enclosure,
            self.statistical_uniform_confidence_authority,
            self.action_invariance_authority,
            self.c_stream_authority,
            self.calibration_authority,
            self.oracle_authority,
            self.commit_authority,
            self.terminal_authority,
            self.pareto_authority,
        )
        if any(type(item) is not bool for item in flags):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "evaluation authority markers must be exact booleans",
            )
        if (
            self.sample_function_evaluation_only is not True
            or self.population_function_exact_except_numerical_enclosure
            is not self.finite_census_complete
            or any(item is not False for item in flags[3:])
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AUTHORITY_VIOLATION,
                "evaluation receipt leaked statistical or downstream authority",
            )
        if (
            self.authority_scope != VARIABLE_SHELL_SOURCE_AUTHORITY_SCOPE
            or self.schema != VARIABLE_SHELL_EVALUATION_SCHEMA
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "unsupported evaluation schema",
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_PREFIX + "evaluation-")


@dataclass(frozen=True, slots=True)
class V1SignedWorkVariableShellExactReference:
    """Exact rational value of the binary64 sample inputs (test-only)."""

    f_value: Fraction
    q_value: Fraction


class V1SignedWorkVariableShellSourceTensorLease:
    """Narrow process-local borrow of the compact solver tensors.

    The lease exposes only the already-retained bounded-shell representation:
    ``b_fiber``, its radius, ``a_shell``, ``d_response``, its radius,
    ``j_shell``, normalized context weights, and the shell-to-fiber map.  It
    neither copies nor serializes tensors and grants no evidential or
    downstream authority.  The owning source refuses eviction while a lease
    is active and rechecks every tensor version/pointer when it closes.

    Borrowers must not retain tensor references after ``close``.  PyTorch has
    no const tensor view, so mutation is policed by the sealed version
    invariants and fails the source closed on the next custody check.
    """

    __slots__ = (
        "_owner",
        "_lease_token",
        "_closed",
        "source_receipt_hash",
        "source_replay_hash",
        "borrow_purpose_hash",
        "lease_hash",
        "sample_function_source_only",
        "solver_authority",
        "action_invariance_authority",
        "c_stream_authority",
        "commit_authority",
        "schema",
    )
    __hash__ = None

    def __init__(
        self,
        *,
        owner: "V1SignedWorkVariableShellSourceDeviceHandle",
        lease_token: object,
        borrow_purpose_hash: str,
    ) -> None:
        if type(owner) is not V1SignedWorkVariableShellSourceDeviceHandle:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "tensor lease requires the exact source owner",
            )
        receipt = owner.receipt
        purpose = _identity(borrow_purpose_hash, name="tensor borrow purpose hash")
        self._owner = owner
        self._lease_token = lease_token
        self._closed = False
        self.source_receipt_hash = receipt.receipt_hash
        self.source_replay_hash = receipt.source_replay_hash
        self.borrow_purpose_hash = purpose
        self.sample_function_source_only = True
        self.solver_authority = False
        self.action_invariance_authority = False
        self.c_stream_authority = False
        self.commit_authority = False
        self.schema = VARIABLE_SHELL_SOURCE_TENSOR_LEASE_SCHEMA
        self.lease_hash = content_hash(
            {
                "schema": self.schema,
                "source_receipt_hash": self.source_receipt_hash,
                "source_replay_hash": self.source_replay_hash,
                "borrow_purpose_hash": purpose,
                "sample_function_source_only": True,
                "all_downstream_authority": False,
            },
            prefix=_PREFIX + "tensor-lease-",
        )

    def compact_tensors(self) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        if self._closed:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.EVICTED,
                "tensor lease was closed",
            )
        if self._owner.receipt.receipt_hash != self.source_receipt_hash:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.STALE_IDENTITY,
                "tensor lease source identity changed",
            )
        return self._owner._require_live()

    def close(self) -> None:
        if self._closed:
            return
        self._owner._close_tensor_lease(self._lease_token)
        self._closed = True

    def __enter__(self) -> "V1SignedWorkVariableShellSourceTensorLease":
        self.compact_tensors()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        del exc_type, exc, traceback
        self.close()
        return False

    def __reduce__(self) -> NoReturn:
        raise TypeError("live variable-shell tensor leases are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("live variable-shell tensor leases are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("live variable-shell tensor leases cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("live variable-shell tensor leases cannot be copied")


class V1SignedWorkVariableShellSourceDeviceHandle:
    """Sole process-local owner of the compact bounded-shell tensors."""

    __slots__ = (
        "_b_fiber",
        "_b_radius",
        "_a_shell",
        "_d_response",
        "_d_radius",
        "_j_shell",
        "_weights",
        "_shell_fibers",
        "_receipt",
        "_forecast",
        "_design",
        "_registered_context_ids",
        "_invariants",
        "_active_tensor_lease_tokens",
    )

    def __init__(
        self,
        *,
        b_fiber: Any,
        b_radius: Any,
        a_shell: Any,
        d_response: Any,
        d_radius: Any,
        j_shell: Any,
        weights: Any,
        shell_fibers: Any,
        receipt: V1SignedWorkVariableShellSourceReceipt,
        forecast: V1SignedWorkVariableShellSourceCUDAHBMForecast,
        design: V1SignedWorkVariableShellSourceCUDAExecutionDesign,
        registered_context_ids: Sequence[str] | None = None,
    ) -> None:
        if torch is None:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.DEVICE_REQUIRED,
                "PyTorch is unavailable",
            )
        if (
            type(receipt) is not V1SignedWorkVariableShellSourceReceipt
            or type(forecast) is not V1SignedWorkVariableShellSourceCUDAHBMForecast
            or type(design) is not V1SignedWorkVariableShellSourceCUDAExecutionDesign
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "live handle requires exact receipt, forecast, and design types",
            )
        count, fibers, shell = (
            receipt.context_count,
            receipt.fiber_count,
            len(receipt.shell_indices),
        )
        device = b_fiber.device if type(b_fiber) is torch.Tensor else None
        _require_fp64_tensor(
            b_fiber, name="fixed outside fiber offsets", shape=(count, fibers)
        )
        _require_fp64_tensor(
            b_radius,
            name="fixed outside fiber numerical radii",
            shape=(count, fibers),
            device=device,
        )
        _require_fp64_tensor(
            a_shell,
            name="shell corrected coefficients",
            shape=(count, shell),
            device=device,
        )
        _require_fp64_tensor(
            d_response,
            name="fixed outside literal-J projection",
            shape=(count,),
            device=device,
        )
        _require_fp64_tensor(
            d_radius,
            name="fixed outside literal-J numerical radii",
            shape=(count,),
            device=device,
        )
        _require_fp64_tensor(
            j_shell,
            name="shell literal-J coefficients",
            shape=(count, shell),
            device=device,
        )
        _require_fp64_tensor(
            weights, name="context weights", shape=(count,), device=device
        )
        if (
            type(shell_fibers) is not torch.Tensor
            or shell_fibers.dtype != torch.int64
            or tuple(shell_fibers.shape) != (shell,)
            or not shell_fibers.is_contiguous()
            or shell_fibers.device != device
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "shell fiber axis must be contiguous int64 on the representation device",
            )
        if (
            tuple(int(item) for item in shell_fibers.detach().cpu().tolist())
            != receipt.shell_native_fiber_indices
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.MUTATED_CUSTODY,
                "shell fiber tensor differs from its bounded receipt axis",
            )
        context_ids = (
            ()
            if registered_context_ids is None
            else _identity_axis(
                registered_context_ids,
                name="registered context IDs",
                unique=True,
                allow_empty=False,
            )
        )
        if context_ids and (
            len(context_ids) != count
            or _axis_hash("context", context_ids) != receipt.context_axis_hash
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AXIS_MISMATCH,
                "registered context IDs do not replay the source receipt axis",
            )
        self._b_fiber = b_fiber
        self._b_radius = b_radius
        self._a_shell = a_shell
        self._d_response = d_response
        self._d_radius = d_radius
        self._j_shell = j_shell
        self._weights = weights
        self._shell_fibers = shell_fibers
        self._receipt = receipt
        self._forecast = forecast
        self._design = design
        self._registered_context_ids = context_ids
        self._invariants = tuple(
            _tensor_invariant(item)
            for item in (
                b_fiber,
                b_radius,
                a_shell,
                d_response,
                d_radius,
                j_shell,
                weights,
                shell_fibers,
            )
        )
        self._active_tensor_lease_tokens: set[object] = set()

    def _require_live(self) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        tensors = (
            self._b_fiber,
            self._b_radius,
            self._a_shell,
            self._d_response,
            self._d_radius,
            self._j_shell,
            self._weights,
            self._shell_fibers,
        )
        if any(item is None for item in tensors):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.EVICTED,
                "variable-shell source representation was evicted",
            )
        if tuple(_tensor_invariant(item) for item in tensors) != self._invariants:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.MUTATED_CUSTODY,
                "compact variable-shell source tensors changed after sealing",
            )
        return tensors  # type: ignore[return-value]

    @property
    def receipt(self) -> V1SignedWorkVariableShellSourceReceipt:
        self._require_live()
        return self._receipt

    @property
    def forecast(self) -> V1SignedWorkVariableShellSourceCUDAHBMForecast:
        self._require_live()
        return self._forecast

    @property
    def is_evicted(self) -> bool:
        return self._b_fiber is None

    @property
    def retained_tensor_bytes(self) -> int:
        tensors = self._require_live()
        return sum(int(item.numel()) * int(item.element_size()) for item in tensors)

    def registered_context_ids(self) -> tuple[str, ...]:
        """Return the source-owned observed context prefix for confidence joins."""

        self._require_live()
        if not self._registered_context_ids:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AUTHORITY_VIOLATION,
                "variable-shell source lacks live context-axis custody",
            )
        return self._registered_context_ids

    def borrow_compact_tensors(
        self,
        *,
        borrow_purpose_hash: str,
    ) -> V1SignedWorkVariableShellSourceTensorLease:
        """Borrow compact live tensors without copying or gaining authority."""

        self._require_live()
        token = object()
        self._active_tensor_lease_tokens.add(token)
        return V1SignedWorkVariableShellSourceTensorLease(
            owner=self,
            lease_token=token,
            borrow_purpose_hash=borrow_purpose_hash,
        )

    def _close_tensor_lease(self, token: object) -> None:
        if token not in self._active_tensor_lease_tokens:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.STALE_IDENTITY,
                "tensor lease token is not active on this source",
            )
        self._require_live()
        self._active_tensor_lease_tokens.remove(token)

    def _validated_z(self, shell_gates: object) -> Any:
        tensors = self._require_live()
        device = tensors[0].device
        shell = len(self._receipt.shell_indices)
        if type(shell_gates) is torch.Tensor:
            z = _require_fp64_tensor(
                shell_gates, name="shell gates", shape=(shell,), device=device
            )
        else:
            if (
                isinstance(shell_gates, (str, bytes))
                or not isinstance(shell_gates, Sequence)
                or len(shell_gates) != shell
            ):
                _refuse(
                    V1SignedWorkVariableShellSourceRefusalCode.AXIS_MISMATCH,
                    "shell gates must align with the frozen shell",
                )
            values: list[float] = []
            for item in shell_gates:
                if (
                    isinstance(item, bool)
                    or not isinstance(item, Real)
                    or not math.isfinite(float(item))
                ):
                    _refuse(
                        V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                        "shell gates must be finite reals",
                    )
                values.append(float(item))
            z = torch.tensor(values, dtype=torch.float64, device=device)
        if bool(((z < 0.0) | (z > 1.0)).any().item()):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "relaxed shell gates must lie in [0,1]",
            )
        return z

    def evaluate(
        self, shell_gates: object
    ) -> V1SignedWorkVariableShellEvaluationReceipt:
        (
            b_fiber,
            b_radius,
            a_shell,
            d_response,
            d_radius,
            j_shell,
            weights,
            shell_fibers,
        ) = self._require_live()
        z = self._validated_z(shell_gates)
        contexts, fibers = b_fiber.shape
        shell = z.numel()
        affine_gamma = _gamma(2 * int(shell) + 1)
        f_center_tensor = torch.zeros((), dtype=torch.float64, device=b_fiber.device)
        q_center_tensor = torch.zeros((), dtype=torch.float64, device=b_fiber.device)
        f_local_total = torch.zeros((), dtype=torch.float64, device=b_fiber.device)
        q_local_total = torch.zeros((), dtype=torch.float64, device=b_fiber.device)

        # Evaluation is tiled over contexts.  Thus arbitrary z never allocates
        # another complete context-by-fiber bank, and the registered evaluation
        # workspace is independent of the acquired D_W2 prefix length.
        for context_start in range(0, int(contexts), self._design.maximum_batch_size):
            context_stop = min(
                int(contexts), context_start + self._design.maximum_batch_size
            )
            b_tile = b_fiber[context_start:context_stop]
            b_radius_tile = b_radius[context_start:context_stop]
            a_tile = a_shell[context_start:context_stop]
            d_tile = d_response[context_start:context_stop]
            d_radius_tile = d_radius[context_start:context_stop]
            j_tile = j_shell[context_start:context_stop]
            weight_tile = weights[context_start:context_stop]

            # Copy only one compact context tile.  Sparse shell additions use
            # B<=8 scalar indices; no route-scale label is materialized.
            fiber_response = b_tile.clone()
            for shell_index in range(shell):
                fiber_response[:, shell_fibers[shell_index]] += (
                    a_tile[:, shell_index] * z[shell_index]
                )
            literal_response = d_tile + torch.sum(j_tile * z.unsqueeze(0), dim=1)
            f_terms = torch.sum(torch.abs(fiber_response), dim=1) / float(
                self._receipt.cut_count
            )
            q_terms = literal_response * literal_response
            f_center_tensor += torch.sum(weight_tile * f_terms)
            q_center_tensor += torch.sum(weight_tile * q_terms)

            # Conservative standard-model enclosure.  Local affine errors are
            # bounded by gamma_(2B+1) times the absolute operand mass.  The Q
            # square uses |(r+e)^2-r^2| <= 2|r|e+e^2.
            f_abs_operands = torch.abs(b_tile)
            for shell_index in range(shell):
                f_abs_operands[:, shell_fibers[shell_index]] += torch.abs(
                    a_tile[:, shell_index] * z[shell_index]
                )
            underflow_f = _F64_MIN_SUBNORMAL * float(2 * int(shell) + 2)
            f_local_error = b_radius_tile + affine_gamma * f_abs_operands + underflow_f
            f_error_terms = torch.sum(f_local_error, dim=1) / float(
                self._receipt.cut_count
            )
            j_abs_operands = torch.abs(d_tile) + torch.sum(
                torch.abs(j_tile * z.unsqueeze(0)), dim=1
            )
            underflow_q = _F64_MIN_SUBNORMAL * float(2 * int(shell) + 2)
            j_local_error = d_radius_tile + affine_gamma * j_abs_operands + underflow_q
            q_local_error = (
                2.0 * torch.abs(literal_response) * j_local_error
                + j_local_error * j_local_error
            )
            f_local_total += torch.sum(weight_tile * f_error_terms)
            q_local_total += torch.sum(weight_tile * q_local_error)

        if not bool(torch.isfinite(f_center_tensor).item()) or not bool(
            torch.isfinite(q_center_tensor).item()
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.NUMERICAL_REFUSAL,
                "variable-shell evaluation overflowed binary64",
            )
        reduction_gamma = _gamma(max(4, 2 * int(contexts) * max(1, int(fibers))))
        f_center = float(f_center_tensor.item())
        q_center = float(q_center_tensor.item())
        f_local_scalar = float(f_local_total.item())
        q_local_scalar = float(q_local_total.item())
        f_radius = _up(f_local_scalar + reduction_gamma * (f_center + f_local_scalar))
        q_radius = _up(q_local_scalar + reduction_gamma * (q_center + q_local_scalar))
        f_lower = max(0.0, _down(f_center - f_radius))
        f_upper = _up(f_center + f_radius)
        q_lower = max(0.0, _down(q_center - q_radius))
        q_upper = _up(q_center + q_radius)
        z_values = tuple(float(item) for item in z.detach().cpu().tolist())
        z_hash = content_hash(
            {"shell_indices": self._receipt.shell_indices, "gates": z_values},
            prefix=_PREFIX + "z-",
        )
        return V1SignedWorkVariableShellEvaluationReceipt(
            source_receipt_hash=self._receipt.receipt_hash,
            source_replay_hash=self._receipt.source_replay_hash,
            shell_gate_hash=z_hash,
            shell_gates=z_values,
            f_lower=f_lower,
            f_center=f_center,
            f_upper=f_upper,
            q_lower=q_lower,
            q_center=q_center,
            q_upper=q_upper,
            finite_census_complete=self._receipt.finite_census_complete,
            population_function_exact_except_numerical_enclosure=self._receipt.finite_census_complete,
        )

    def evaluate_exact_cpu_reference(
        self, shell_gates: Sequence[float]
    ) -> V1SignedWorkVariableShellExactReference:
        if (
            self._design.execution_mode
            is not V1SignedWorkVariableShellExecutionMode.CPU_REFERENCE_TEST_ONLY
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.CPU_PRODUCTION_FALLBACK,
                "exact rational evaluation is private to CPU reference tests",
            )
        (
            b_fiber,
            _b_radius,
            a_shell,
            d_response,
            _d_radius,
            j_shell,
            weights,
            shell_fibers,
        ) = self._require_live()
        z = self._validated_z(shell_gates)
        arrays = tuple(
            item.detach().cpu().tolist()
            for item in (
                b_fiber,
                a_shell,
                d_response,
                j_shell,
                weights,
                shell_fibers,
                z,
            )
        )
        (
            b_values,
            a_values,
            d_values,
            j_values,
            weight_values,
            shell_fiber_values,
            z_values,
        ) = arrays
        z_f = [Fraction.from_float(float(item)) for item in z_values]
        f_total = Fraction(0)
        q_total = Fraction(0)
        for context_index in range(self._receipt.context_count):
            fibers = [
                Fraction.from_float(float(item)) for item in b_values[context_index]
            ]
            for shell_index, fiber_index in enumerate(shell_fiber_values):
                fibers[int(fiber_index)] += (
                    Fraction.from_float(float(a_values[context_index][shell_index]))
                    * z_f[shell_index]
                )
            response = Fraction.from_float(float(d_values[context_index]))
            for shell_index in range(len(z_f)):
                response += (
                    Fraction.from_float(float(j_values[context_index][shell_index]))
                    * z_f[shell_index]
                )
            weight = Fraction.from_float(float(weight_values[context_index]))
            f_total += (
                weight * sum(abs(item) for item in fibers) / self._receipt.cut_count
            )
            q_total += weight * response * response
        return V1SignedWorkVariableShellExactReference(f_value=f_total, q_value=q_total)

    def evict(self) -> None:
        if self._active_tensor_lease_tokens:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.ACTIVE_TENSOR_LEASE,
                "cannot evict variable-shell source while a tensor lease is active",
            )
        self._b_fiber = None
        self._b_radius = None
        self._a_shell = None
        self._d_response = None
        self._d_radius = None
        self._j_shell = None
        self._weights = None
        self._shell_fibers = None
        self._invariants = ()

    def __reduce__(self) -> NoReturn:
        raise TypeError("variable-shell source handles are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("variable-shell source handles are nonserializable")

    def __getstate__(self) -> NoReturn:
        raise TypeError("variable-shell source handles are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("variable-shell source handles cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> NoReturn:
        del memo
        raise TypeError("variable-shell source handles cannot be copied")


class V1SignedWorkVariableShellSourceCUDAAccumulator:
    """Predictable D_W2 packet reducer with no route-row custody."""

    __slots__ = (
        "registration",
        "native_fiber_axis_receipt",
        "design",
        "forecast",
        "_device",
        "_fixed_outside",
        "_b_fiber",
        "_a_shell",
        "_d_response",
        "_b_radius",
        "_d_radius",
        "_j_shell",
        "_weights",
        "_shell_fibers",
        "_context_cursor",
        "_active_count",
        "_active_context_start",
        "_active_route_cursor",
        "_active_chunk_count",
        "_active_b",
        "_active_b_radius",
        "_active_a",
        "_active_d",
        "_active_d_abs_mass",
        "_active_j",
        "_active_packet_hashes",
        "_chunk_binding_hashes",
        "_aborted",
        "_sealed",
    )

    def __init__(
        self,
        *,
        registration: V1SignedWorkVariableShellSourceRegistration,
        native_fiber_axis_receipt: V1NativeDestinationFiberAxisReceipt,
        frozen_primal: Any,
        design: V1SignedWorkVariableShellSourceCUDAExecutionDesign,
        declared_device: str,
        device_total_memory_bytes: int | None = None,
        device_free_memory_bytes: int | None = None,
        current_reserved_device_bytes: int | None = None,
    ) -> None:
        if torch is None:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.DEVICE_REQUIRED,
                "PyTorch is unavailable",
            )
        if (
            type(registration) is not V1SignedWorkVariableShellSourceRegistration
            or type(native_fiber_axis_receipt)
            is not V1NativeDestinationFiberAxisReceipt
            or type(design) is not V1SignedWorkVariableShellSourceCUDAExecutionDesign
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "accumulator requires exact registration, native-fiber, and design types",
            )
        native_fiber_axis_receipt.assert_integrity()
        try:
            device = torch.device(_identity(declared_device, name="declared device"))
        except (RuntimeError, ValueError) as error:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.DEVICE_REQUIRED,
                f"invalid declared device: {error}",
            )
        if (
            design.execution_mode
            is V1SignedWorkVariableShellExecutionMode.CUDA_PRODUCTION
        ):
            if (
                device.type != "cuda"
                or device.index is None
                or not torch.cuda.is_available()
            ):
                _refuse(
                    V1SignedWorkVariableShellSourceRefusalCode.DEVICE_REQUIRED,
                    "production variable-shell source execution requires an explicit live CUDA device",
                )
            warn_only = getattr(
                torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False
            )
            if (
                not bool(torch.are_deterministic_algorithms_enabled())
                or bool(warn_only())
                or os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}
            ):
                _refuse(
                    V1SignedWorkVariableShellSourceRefusalCode.DETERMINISM_DISABLED,
                    "production CUDA requires strict deterministic algorithms and a qualified CUBLAS workspace",
                )
            if any(
                item is not None
                for item in (
                    device_total_memory_bytes,
                    device_free_memory_bytes,
                    current_reserved_device_bytes,
                )
            ):
                _refuse(
                    V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                    "production CUDA HBM facts must come from the live device, not caller overrides",
                )
            free, total = torch.cuda.mem_get_info(device)
            admission_memory = (
                cuda_reusable_memory_bytes(device, framework=torch, free_bytes=int(free)),
                int(torch.cuda.memory_allocated(device)),
            )
            total_memory = int(total)
            free_memory = int(free)
            reserved = int(torch.cuda.memory_reserved(device))
        else:
            if device.type != "cpu":
                _refuse(
                    V1SignedWorkVariableShellSourceRefusalCode.CPU_PRODUCTION_FALLBACK,
                    "private reference mode is CPU-only",
                )
            if (
                device_total_memory_bytes is None
                or device_free_memory_bytes is None
                or current_reserved_device_bytes is None
            ):
                _refuse(
                    V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                    "CPU reference mode requires explicit synthetic memory facts",
                )
            admission_memory = None
            total_memory, free_memory, reserved = (
                int(device_total_memory_bytes),
                int(device_free_memory_bytes),
                int(current_reserved_device_bytes),
            )
        forecast = build_v1_signed_work_variable_shell_source_cuda_hbm_forecast(
            registration=registration,
            native_fiber_axis_receipt=native_fiber_axis_receipt,
            design=design,
            device_total_memory_bytes=total_memory,
            device_free_memory_bytes=free_memory,
            current_reserved_device_bytes=reserved,
            _admission_memory=admission_memory,
        )
        if not forecast.admitted:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.HBM_FORECAST_EXCEEDED,
                "variable-shell compact representation exceeds the registered HBM envelope",
            )
        route_count = native_fiber_axis_receipt.route_count
        frozen = _require_fp64_tensor(
            frozen_primal, name="frozen primal", shape=(route_count,), device=device
        )
        if bool(((frozen < 0.0) | (frozen > 1.0)).any().item()):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "frozen primal gates must lie in [0,1]",
            )
        self.registration = registration
        self.native_fiber_axis_receipt = native_fiber_axis_receipt
        self.design = design
        self.forecast = forecast
        self._device = device
        self._fixed_outside = frozen.clone()
        shell_tensor = torch.tensor(
            registration.shell_indices, dtype=torch.int64, device=device
        )
        self._fixed_outside[shell_tensor] = 0.0
        contexts, fibers, shell = (
            registration.context_count,
            native_fiber_axis_receipt.fiber_count,
            registration.shell_size,
        )
        self._b_fiber = torch.zeros(
            (contexts, fibers), dtype=torch.float64, device=device
        )
        self._b_radius = torch.zeros(
            (contexts, fibers), dtype=torch.float64, device=device
        )
        self._a_shell = torch.zeros(
            (contexts, shell), dtype=torch.float64, device=device
        )
        self._d_response = torch.zeros((contexts,), dtype=torch.float64, device=device)
        self._d_radius = torch.zeros((contexts,), dtype=torch.float64, device=device)
        self._j_shell = torch.zeros(
            (contexts, shell), dtype=torch.float64, device=device
        )
        self._weights = torch.tensor(
            registration.context_weights, dtype=torch.float64, device=device
        )
        self._shell_fibers = torch.tensor(
            tuple(
                native_fiber_axis_receipt.route_to_fiber(index).global_fiber_index
                for index in registration.shell_indices
            ),
            dtype=torch.int64,
            device=device,
        )
        self._context_cursor = 0
        self._active_count = 0
        self._active_context_start = 0
        self._active_route_cursor = 0
        self._active_chunk_count = 0
        self._active_b = None
        self._active_b_radius = None
        self._active_a = None
        self._active_d = None
        self._active_d_abs_mass = None
        self._active_j = None
        self._active_packet_hashes: dict[str, str] | None = None
        self._chunk_binding_hashes: list[str] = []
        self._aborted = False
        self._sealed = False

    @property
    def device(self) -> Any:
        """Return the registered execution device without exposing tensor custody."""

        return self._device

    @property
    def is_aborted(self) -> bool:
        """Report whether unsealed tensor custody was explicitly released."""

        return bool(self._aborted)

    def _require_unaborted(self) -> None:
        if self._aborted:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.EVICTED,
                "variable-shell source accumulator was aborted",
            )

    def begin_packet(
        self,
        *,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        measurement_ids: Sequence[str],
        capture_ids: Sequence[str],
        source_packet_hashes: Sequence[str],
    ) -> None:
        self._require_unaborted()
        if self._sealed:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.DUPLICATE_PACKET,
                "accumulator was already sealed",
            )
        if self._active_count:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.ACTIVE_PACKET,
                "one D_W2 packet is already active",
            )
        axes = tuple(
            tuple(item)
            for item in (
                context_ids,
                draw_ids,
                measurement_ids,
                capture_ids,
                source_packet_hashes,
            )
        )
        counts = {len(item) for item in axes}
        if len(counts) != 1:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AXIS_MISMATCH,
                "packet source axes must align",
            )
        count = next(iter(counts))
        if (
            count <= 0
            or count > self.design.maximum_batch_size
            or self._context_cursor + count > self.registration.context_count
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.ORDER_MISMATCH,
                "packet size is empty, too large, or exceeds the registered prefix",
            )
        start, stop = self._context_cursor, self._context_cursor + count
        registered_axes = (
            self.registration.registered_context_ids,
            self.registration.registered_draw_ids,
            self.registration.registered_measurement_ids,
            self.registration.registered_capture_ids,
            self.registration.registered_source_packet_hashes,
        )
        if any(
            tuple(axis) != tuple(registered[start:stop])
            for axis, registered in zip(axes, registered_axes)
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.ORDER_MISMATCH,
                "D_W2 packet identities differ from the predictable registered prefix",
            )
        self._active_packet_hashes = {
            "context": _axis_hash("active_packet_context", axes[0]),
            "draw": _axis_hash("active_packet_draw", axes[1]),
            "measurement": _axis_hash("active_packet_measurement", axes[2]),
            "capture": _axis_hash("active_packet_capture", axes[3]),
            "source_packet": _axis_hash("active_packet_source_packet", axes[4]),
        }
        self._active_count = count
        self._active_context_start = start
        self._active_route_cursor = 0
        self._active_chunk_count = 0
        self._active_b = torch.zeros(
            (count, self.native_fiber_axis_receipt.fiber_count),
            dtype=torch.float64,
            device=self._device,
        )
        self._active_b_radius = torch.zeros(
            (count, self.native_fiber_axis_receipt.fiber_count),
            dtype=torch.float64,
            device=self._device,
        )
        self._active_a = torch.zeros(
            (count, self.registration.shell_size),
            dtype=torch.float64,
            device=self._device,
        )
        self._active_d = torch.zeros((count,), dtype=torch.float64, device=self._device)
        self._active_d_abs_mass = torch.zeros(
            (count,), dtype=torch.float64, device=self._device
        )
        self._active_j = torch.zeros(
            (count, self.registration.shell_size),
            dtype=torch.float64,
            device=self._device,
        )

    def issue_route_chunk_binding(
        self,
        *,
        route_start: int,
        corrected_evidence_rows: Any,
        literal_j_rows: Any,
        route_axis_hash: str,
        route_id_axis_hash: str,
        same_capture_vjp_source_hash: str,
        corrected_evidence_stream_source_hash: str,
        literal_j_stream_source_hash: str,
    ) -> V1SignedWorkVariableShellSourceChunkBinding:
        self._require_unaborted()
        if not self._active_count:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.PARTIAL_PACKET,
                "begin_packet must precede chunk-binding issuance",
            )
        start = _nonnegative_int(route_start, name="route chunk start")
        if start != self._active_route_cursor:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.ORDER_MISMATCH,
                "chunk bindings are issued only for the next contiguous route span",
            )
        if (
            _identity(route_axis_hash, name="route axis hash")
            != self.registration.route_axis_hash
            or _identity(route_id_axis_hash, name="route ID axis hash")
            != self.registration.route_id_axis_hash
            or _identity(
                same_capture_vjp_source_hash, name="same-capture VJP source hash"
            )
            != self.registration.same_capture_vjp_source_hash
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.STALE_IDENTITY,
                "route chunk source identity is stale or permuted",
            )
        if (
            torch is None
            or type(corrected_evidence_rows) is not torch.Tensor
            or corrected_evidence_rows.ndim != 2
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "corrected rows must be one rank-two tensor",
            )
        count, width = tuple(int(item) for item in corrected_evidence_rows.shape)
        if width <= 0 or width > self.design.route_chunk_size:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "route chunk exceeds the registered physical chunk size",
            )
        corrected = _require_fp64_tensor(
            corrected_evidence_rows,
            name="corrected evidence rows",
            shape=(self._active_count, width),
            device=self._device,
        )
        literal = _require_fp64_tensor(
            literal_j_rows,
            name="literal-J rows",
            shape=(self._active_count, width),
            device=self._device,
        )
        stop = start + width
        corrected_source = _identity(
            corrected_evidence_stream_source_hash,
            name="corrected evidence stream source hash",
        )
        literal_source = _identity(
            literal_j_stream_source_hash,
            name="literal-J stream source hash",
        )
        if (
            corrected_source != self.registration.corrected_evidence_stream_source_hash
            or literal_source != self.registration.literal_j_stream_source_hash
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.STALE_IDENTITY,
                "route chunk corrected/J stream identity is stale",
            )
        packet_hashes = self._active_packet_hashes
        if packet_hashes is None:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.PARTIAL_PACKET,
                "active packet identity hashes are unavailable",
            )
        return V1SignedWorkVariableShellSourceChunkBinding(
            registration_hash=self.registration.registration_hash,
            active_packet_ordinal=self._active_context_start,
            active_packet_context_axis_hash=packet_hashes["context"],
            active_packet_draw_axis_hash=packet_hashes["draw"],
            active_packet_measurement_axis_hash=packet_hashes["measurement"],
            active_packet_capture_axis_hash=packet_hashes["capture"],
            active_packet_source_packet_axis_hash=packet_hashes["source_packet"],
            route_start=start,
            route_stop=stop,
            route_axis_hash=self.registration.route_axis_hash,
            route_id_axis_hash=self.registration.route_id_axis_hash,
            witness_stream_source_hash=self.registration.witness_stream_source_hash,
            same_capture_vjp_source_hash=self.registration.same_capture_vjp_source_hash,
            corrected_evidence_stream_source_hash=corrected_source,
            literal_j_stream_source_hash=literal_source,
            corrected_evidence_tensor_identity_hash=_tensor_source_identity_hash(
                corrected, name="corrected evidence rows"
            ),
            literal_j_tensor_identity_hash=_tensor_source_identity_hash(
                literal, name="literal-J rows"
            ),
            _issuance_token=_ISSUER_TOKEN,
        )

    def consume_route_chunk(
        self,
        *,
        route_start: int,
        corrected_evidence_rows: Any,
        literal_j_rows: Any,
        route_axis_hash: str,
        route_id_axis_hash: str,
        same_capture_vjp_source_hash: str,
        corrected_evidence_stream_source_hash: str | None = None,
        literal_j_stream_source_hash: str | None = None,
        chunk_binding: V1SignedWorkVariableShellSourceChunkBinding | None = None,
    ) -> None:
        self._require_unaborted()
        if not self._active_count:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.PARTIAL_PACKET,
                "begin_packet must precede route chunks",
            )
        start = _nonnegative_int(route_start, name="route chunk start")
        if start != self._active_route_cursor:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.ORDER_MISMATCH,
                "route chunks must cover the registered route axis contiguously",
            )
        if (
            _identity(route_axis_hash, name="route axis hash")
            != self.registration.route_axis_hash
            or _identity(route_id_axis_hash, name="route ID axis hash")
            != self.registration.route_id_axis_hash
            or _identity(
                same_capture_vjp_source_hash, name="same-capture VJP source hash"
            )
            != self.registration.same_capture_vjp_source_hash
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.STALE_IDENTITY,
                "route chunk source identity is stale or permuted",
            )
        if (
            torch is None
            or type(corrected_evidence_rows) is not torch.Tensor
            or corrected_evidence_rows.ndim != 2
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "corrected rows must be one rank-two tensor",
            )
        count, width = tuple(int(item) for item in corrected_evidence_rows.shape)
        if width <= 0 or width > self.design.route_chunk_size:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_DESIGN,
                "route chunk exceeds the registered physical chunk size",
            )
        stop = start + width
        corrected_source = (
            self.registration.corrected_evidence_stream_source_hash
            if corrected_evidence_stream_source_hash is None
            else _identity(
                corrected_evidence_stream_source_hash,
                name="corrected evidence stream source hash",
            )
        )
        literal_source = (
            self.registration.literal_j_stream_source_hash
            if literal_j_stream_source_hash is None
            else _identity(
                literal_j_stream_source_hash,
                name="literal-J stream source hash",
            )
        )
        if (
            corrected_source != self.registration.corrected_evidence_stream_source_hash
            or literal_source != self.registration.literal_j_stream_source_hash
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.STALE_IDENTITY,
                "route chunk corrected/J stream identity is stale",
            )
        corrected = _require_fp64_tensor(
            corrected_evidence_rows,
            name="corrected evidence rows",
            shape=(self._active_count, width),
            device=self._device,
        )
        literal = _require_fp64_tensor(
            literal_j_rows,
            name="literal-J rows",
            shape=(self._active_count, width),
            device=self._device,
        )
        packet_hashes = self._active_packet_hashes
        if packet_hashes is None:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.PARTIAL_PACKET,
                "active packet identity hashes are unavailable",
            )
        if chunk_binding is None:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AUTHORITY_VIOLATION,
                "route chunks require an issuer-sealed source-local tensor binding",
            )
        if type(chunk_binding) is not V1SignedWorkVariableShellSourceChunkBinding:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.INVALID_TYPE,
                "chunk binding must have exact variable-shell chunk-binding type",
            )
        expected_binding = self.issue_route_chunk_binding(
            route_start=start,
            corrected_evidence_rows=corrected,
            literal_j_rows=literal,
            route_axis_hash=route_axis_hash,
            route_id_axis_hash=route_id_axis_hash,
            same_capture_vjp_source_hash=same_capture_vjp_source_hash,
            corrected_evidence_stream_source_hash=corrected_source,
            literal_j_stream_source_hash=literal_source,
        )
        if chunk_binding.binding_hash != expected_binding.binding_hash:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.STALE_IDENTITY,
                "route chunk typed source binding differs from the active packet, route span, or live tensor identity",
            )
        if stop > self.native_fiber_axis_receipt.route_count:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AXIS_MISMATCH,
                "route chunk exceeds the registered route axis",
            )
        cut = self.native_fiber_axis_receipt.cut_for_route(start)
        if stop > cut.route_stop:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AXIS_MISMATCH,
                "one route chunk cannot cross a compact native-fiber cut",
            )
        local_start = start - cut.route_offset
        if local_start % cut.routes_per_fiber or width % cut.routes_per_fiber:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AXIS_MISMATCH,
                "route chunks must begin and end on native-fiber boundaries",
            )
        fiber_local_start = local_start // cut.routes_per_fiber
        fiber_count = width // cut.routes_per_fiber
        fiber_start = cut.fiber_offset + fiber_local_start
        fiber_stop = fiber_start + fiber_count
        outside = self._fixed_outside[start:stop].reshape(
            fiber_count, cut.routes_per_fiber
        )
        corrected_view = corrected.reshape(
            self._active_count, fiber_count, cut.routes_per_fiber
        )
        b_value = torch.einsum("bfr,fr->bf", corrected_view, outside)
        b_abs_mass = torch.einsum(
            "bfr,fr->bf", torch.abs(corrected_view), torch.abs(outside)
        )
        b_gamma = _gamma(2 * cut.routes_per_fiber + 1)
        b_abs_upper = _up_tensor(b_abs_mass / (1.0 - b_gamma))
        b_radius = _up_tensor(
            b_gamma * b_abs_upper
            + _F64_MIN_SUBNORMAL * float(2 * cut.routes_per_fiber + 1)
        )
        self._active_b[:, fiber_start:fiber_stop] = b_value
        self._active_b_radius[:, fiber_start:fiber_stop] = b_radius
        self._active_d += torch.mv(literal, self._fixed_outside[start:stop])
        d_chunk_abs = torch.mv(
            torch.abs(literal), torch.abs(self._fixed_outside[start:stop])
        )
        d_chunk_gamma = _gamma(2 * width + 1)
        d_chunk_upper = _up_tensor(d_chunk_abs / (1.0 - d_chunk_gamma))
        self._active_d_abs_mass = _up_tensor(self._active_d_abs_mass + d_chunk_upper)
        for shell_position, route_index in enumerate(self.registration.shell_indices):
            if start <= route_index < stop:
                local = route_index - start
                self._active_a[:, shell_position] = corrected[:, local]
                self._active_j[:, shell_position] = literal[:, local]
        self._active_route_cursor = stop
        self._active_chunk_count += 1
        self._chunk_binding_hashes.append(expected_binding.durable_binding_hash)

    def end_packet(self) -> None:
        self._require_unaborted()
        if (
            not self._active_count
            or self._active_route_cursor != self.native_fiber_axis_receipt.route_count
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.PARTIAL_PACKET,
                "D_W2 packet did not cover the complete route axis",
            )
        start, stop = (
            self._active_context_start,
            self._active_context_start + self._active_count,
        )
        d_gamma = _gamma(
            2 * self.native_fiber_axis_receipt.route_count
            + self._active_chunk_count
            + 1
        )
        d_radius = _up_tensor(
            d_gamma * self._active_d_abs_mass
            + _F64_MIN_SUBNORMAL
            * float(2 * self.native_fiber_axis_receipt.route_count + 1)
        )
        compact = (
            self._active_b,
            self._active_b_radius,
            self._active_a,
            self._active_d,
            d_radius,
            self._active_j,
        )
        if not all(bool(torch.isfinite(item).all().item()) for item in compact):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.NUMERICAL_REFUSAL,
                "D_W2 compact lowering overflowed binary64",
            )
        self._b_fiber[start:stop].copy_(self._active_b)
        self._b_radius[start:stop].copy_(self._active_b_radius)
        self._a_shell[start:stop].copy_(self._active_a)
        self._d_response[start:stop].copy_(self._active_d)
        self._d_radius[start:stop].copy_(d_radius)
        self._j_shell[start:stop].copy_(self._active_j)
        self._context_cursor = stop
        self._active_count = 0
        self._active_route_cursor = 0
        self._active_chunk_count = 0
        self._active_b = None
        self._active_b_radius = None
        self._active_a = None
        self._active_d = None
        self._active_d_abs_mass = None
        self._active_j = None
        self._active_packet_hashes = None

    def seal(self) -> V1SignedWorkVariableShellSourceDeviceHandle:
        self._require_unaborted()
        if self._sealed:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.DUPLICATE_PACKET,
                "variable-shell source accumulator already sealed",
            )
        if (
            self._active_count
            or self._context_cursor != self.registration.context_count
        ):
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.PARTIAL_PACKET,
                "the registered D_W2 prefix is incomplete",
            )
        self._sealed = True
        self._fixed_outside = None
        shell_fibers = tuple(
            int(item) for item in self._shell_fibers.detach().cpu().tolist()
        )
        retained_f64 = int(
            self._b_fiber.numel()
            + self._b_radius.numel()
            + self._a_shell.numel()
            + self._d_response.numel()
            + self._d_radius.numel()
            + self._j_shell.numel()
            + self._weights.numel()
        )
        retained_i64 = int(self._shell_fibers.numel())
        finite_census_axis_hash = _finite_census_axis_hash(self.registration)
        chunk_binding_axis_hash = content_hash(
            {
                "schema": "partition_free_v1_variable_shell_chunk_binding_axis_v1",
                "registration_hash": self.registration.registration_hash,
                "chunk_binding_hashes": tuple(self._chunk_binding_hashes),
                "complete_packet_route_cover": True,
                "per_chunk_typed_identity_bound": True,
            },
            prefix=_PREFIX + "chunk-binding-axis-",
        )
        source_replay_hash = content_hash(
            {
                "schema": "partition_free_v1_variable_shell_source_replay_v1",
                "registration_hash": self.registration.registration_hash,
                "source_packet_axis_hash": _axis_hash(
                    "source_packet", self.registration.registered_source_packet_hashes
                ),
                "finite_census_axis_hash": finite_census_axis_hash,
                "chunk_binding_axis_hash": chunk_binding_axis_hash,
                "algorithm_id": self.design.algorithm_id,
                "shell_fibers": shell_fibers,
            },
            prefix=_PREFIX + "replay-",
        )
        receipt = V1SignedWorkVariableShellSourceReceipt(
            registration_hash=self.registration.registration_hash,
            design_hash=self.design.design_hash,
            hbm_forecast_hash=self.forecast.forecast_hash,
            model_identity_hash=self.registration.model_identity_hash,
            chart_identity_hash=self.registration.chart_identity_hash,
            current_mask_hash=self.registration.current_mask_hash,
            proposal_generation=self.registration.proposal_generation,
            route_axis_hash=self.registration.route_axis_hash,
            route_id_axis_hash=self.registration.route_id_axis_hash,
            registered_d_prefix_hash=self.registration.registered_d_prefix_hash,
            registered_d_w2_prefix_hash=self.registration.registered_d_w2_prefix_hash,
            shell_freeze_receipt_hash=self.registration.shell_freeze_receipt_hash,
            shell_indices=self.registration.shell_indices,
            shell_route_ids=self.registration.shell_route_ids,
            shell_native_fiber_indices=shell_fibers,
            native_fiber_axis_receipt_hash=self.registration.native_fiber_axis_receipt_hash,
            native_fiber_measure_hash=self.registration.native_fiber_measure_hash,
            kkt_branch_id=self.registration.kkt_branch_id,
            kkt_branch_source_receipt_hash=self.registration.kkt_branch_source_receipt_hash,
            resource_axis_hash=self.registration.resource_axis_hash,
            resource_source_receipt_hash=self.registration.resource_source_receipt_hash,
            witness_stream_source_hash=self.registration.witness_stream_source_hash,
            same_capture_vjp_source_hash=self.registration.same_capture_vjp_source_hash,
            corrected_evidence_stream_source_hash=self.registration.corrected_evidence_stream_source_hash,
            literal_j_stream_source_hash=self.registration.literal_j_stream_source_hash,
            alpha_ledger_id=self.registration.alpha_ledger_id,
            alpha_coordinate_id=self.registration.alpha_coordinate_id,
            alpha_lineage_hash=self.registration.alpha_lineage_hash,
            population_context_axis_hash=(
                self.registration.population_context_axis_hash
            ),
            population_draw_axis_hash=self.registration.population_draw_axis_hash,
            context_axis_hash=_axis_hash(
                "context", self.registration.registered_context_ids
            ),
            draw_axis_hash=_axis_hash("draw", self.registration.registered_draw_ids),
            measurement_axis_hash=_axis_hash(
                "measurement", self.registration.registered_measurement_ids
            ),
            capture_axis_hash=_axis_hash(
                "capture", self.registration.registered_capture_ids
            ),
            source_packet_axis_hash=_axis_hash(
                "source_packet", self.registration.registered_source_packet_hashes
            ),
            finite_census_axis_hash=finite_census_axis_hash,
            chunk_binding_axis_hash=chunk_binding_axis_hash,
            context_count=self.registration.context_count,
            population_count=self.registration.population_count,
            route_count=self.native_fiber_axis_receipt.route_count,
            fiber_count=self.native_fiber_axis_receipt.fiber_count,
            cut_count=self.native_fiber_axis_receipt.cut_count,
            retained_float64_count=retained_f64,
            retained_int64_count=retained_i64,
            retained_tensor_bytes=retained_f64 * _F64_BYTES + retained_i64 * _I64_BYTES,
            source_replay_hash=source_replay_hash,
            count_complete=self.registration.count_complete,
            finite_census_complete=self.registration.finite_census_complete,
            finite_census_axis_custody_complete=(
                self.registration.finite_census_axis_custody_complete
            ),
            _issuance_token=_ISSUER_TOKEN,
        )
        handle = V1SignedWorkVariableShellSourceDeviceHandle(
            b_fiber=self._b_fiber,
            b_radius=self._b_radius,
            a_shell=self._a_shell,
            d_response=self._d_response,
            d_radius=self._d_radius,
            j_shell=self._j_shell,
            weights=self._weights,
            shell_fibers=self._shell_fibers,
            receipt=receipt,
            forecast=self.forecast,
            design=self.design,
            registered_context_ids=self.registration.registered_context_ids,
        )
        self._b_fiber = None
        self._b_radius = None
        self._a_shell = None
        self._d_response = None
        self._d_radius = None
        self._j_shell = None
        self._weights = None
        self._shell_fibers = None
        return handle

    def abort_unsealed(self) -> None:
        """Idempotently release every tensor still owned before ``seal``.

        A sealed accumulator has already transferred tensor ownership to its
        returned device handle and therefore cannot be aborted through this
        API.  Callers must evict that handle instead.  This distinction keeps
        failure-atomic orchestration from silently destroying published live
        custody while still giving partial construction an explicit HBM
        unwind path.
        """

        if self._aborted:
            return
        if self._sealed:
            _refuse(
                V1SignedWorkVariableShellSourceRefusalCode.AUTHORITY_VIOLATION,
                "sealed accumulator transferred custody; evict its device handle",
            )
        for name in (
            "_fixed_outside",
            "_b_fiber",
            "_b_radius",
            "_a_shell",
            "_d_response",
            "_d_radius",
            "_j_shell",
            "_weights",
            "_shell_fibers",
            "_active_b",
            "_active_b_radius",
            "_active_a",
            "_active_d",
            "_active_d_abs_mass",
            "_active_j",
        ):
            setattr(self, name, None)
        self._active_packet_hashes = None
        self._chunk_binding_hashes.clear()
        self._active_count = 0
        self._active_route_cursor = 0
        self._active_chunk_count = 0
        self._aborted = True

    def __reduce__(self) -> NoReturn:
        raise TypeError("variable-shell source accumulators are nonserializable")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise TypeError("variable-shell source accumulators are nonserializable")


__all__ = [
    "VARIABLE_SHELL_SOURCE_ALGORITHM",
    "VARIABLE_SHELL_SOURCE_AUTHORITY_SCOPE",
    "VARIABLE_SHELL_F_EQUATION",
    "VARIABLE_SHELL_Q_EQUATION",
    "V1SignedWorkVariableShellExecutionMode",
    "V1SignedWorkVariableShellSourceRefusalCode",
    "V1SignedWorkVariableShellSourceRefusal",
    "V1SignedWorkVariableShellSourceCUDAExecutionDesign",
    "V1SignedWorkVariableShellSourceChunkBinding",
    "V1SignedWorkVariableShellSourceRegistration",
    "V1SignedWorkVariableShellSourceCUDAHBMForecast",
    "V1SignedWorkVariableShellSourceReceipt",
    "V1SignedWorkVariableShellEvaluationReceipt",
    "V1SignedWorkVariableShellExactReference",
    "V1SignedWorkVariableShellSourceTensorLease",
    "V1SignedWorkVariableShellSourceDeviceHandle",
    "V1SignedWorkVariableShellSourceCUDAAccumulator",
    "build_v1_signed_work_variable_shell_source_cuda_hbm_forecast",
    "v1_variable_shell_population_axis_hash",
]
