"""Factored (computation-graph) custody of the modern-v1 corrected-A and literal-J rows.

Per registered context ``x`` the D lane's two fp64 route-scale rows are, on every
Linear block, the Hadamard product of the layer weight with a rank-one outer
product of a per-unit cotangent and the shared layer input,

    A(x) = -M o ( W o ( gtilde(x) x(x)^T ) ),   J(x) = -M o ( W o ( delta(x) x(x)^T ) ),

with ``gtilde = G o phi(a, b)`` (conserving Aumann--Shapley cotangent) and
``delta = G o 1[a > 0]`` (literal gate cotangent); on every ``conv_kernel_weight``
block they are the weight-Hadamard of a correlation of the same two objects
over positions.  This module custodies the *factors*
``(x, G, a)`` per layer together with the chart's ``(W, b)`` and reproduces the
legacy rows on demand.

Floating-point law.  The legacy producer forms the row
in fp32 -- ``(cot / scale)`` then the outer product then the Hadamard ``W`` --
and only then promotes the route chunk to fp64, multiplies by the 0/1 hard gate
and negates (``mechanistic_systems/measurement/production.py``).  Materialization
here therefore replays the same FP32 elementwise operation order for linear
windows, and ``mechanistic_systems.backends.torch_runtime.
conv_kernel_route_measurement_with_context_target_batched`` on the original
``T = 2`` ``(literal, corrected)`` cotangent stack for convolution. Linear
destination slicing introduces no reduction or reassociation. Same-device
bitwise parity is tested against the original full kernels. The native-fiber
forward/adjoint pair uses these rounded coefficients, with rounded FP64
contractions. The older algebra
(``inner``/``gram``/``moments``/``price_matrix``/own-kink damage) works in fp64
from the fp32 factors and agrees with row-based computation to fp64 tolerance,
never bitwise.

This is a new V2 family (``partition_free_v1_factored_row_capture_v2``) keyed by
the parent packet plan / look boundary hash.  No field is added to any legacy
hashed family.  The packet is process-local and nonserializable; only its
receipt is durable.  Nothing here certifies F/Q/R, opens C/K/O, authorizes a
mask commit, or carries terminal, runtime, or Pareto authority.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from mechanistic_systems.core.numerics import CapturedRouteBlock, exact_batched_vjp
from mechanistic_systems.core.storage import RouteBlockSpec
from mechanistic_systems.measurement.queries import QueryCotangentBatch

from .modern_v1_conserving_attribution import conserving_relu_cotangent
from .modern_v1_conserving_route_score import (
    V1ConservingActivationCapture,
    literal_relu_gate_jacobian_cotangent,
)
from .modern_v1_sampled_dual_chart_row_custody import V1SampledDualChartRowRole

try:  # pragma: no cover - exercised wherever torch is installed.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


FACTORED_ROW_CAPTURE_FAMILY_SCHEMA: Final = "partition_free_v1_factored_row_capture_v2"
FACTORED_ROW_LAYER_FACTOR_DESCRIPTOR_SCHEMA: Final = (
    "partition_free_v1_factored_row_layer_factor_descriptor_v2"
)
FACTORED_ROW_OWN_KINK_SET_DAMAGE_SCHEMA: Final = (
    "partition_free_v1_factored_row_own_kink_set_damage_v2"
)
FACTORED_ROW_EQUALITY_LAW_ID: Final = (
    "legacy_fp32_kernel_order_reproduction_then_fp64_gate_negate_bitwise_same_device_v1"
)
# Bound on the linear-layer einsum intermediate ``[k_chunk, out, in]`` in ``combine_rows``.
DEFAULT_COMBINE_WORKSPACE_BYTES: Final = 2 << 30
FACTORED_ROW_SYMMETRY_COVARIANCE_LAW_ID: Final = (
    "per_unit_positive_rescaling_cotangent_input_covariant_route_drive_row_invariant_v1"
)
FACTORED_ROW_CAPTURE_AUTHORITY_SCOPE: Final = (
    "proposal_input_custody_only_no_D_stability_C_K_O_mask_commit_terminal_or_"
    "pareto_authority_v1"
)
# The legacy fused reducer receives ``cat((literal, corrected), dim=0)``:
# query index 0 is the literal-J role, query index 1 the corrected-A role.
FACTORED_ROW_KERNEL_QUERY_ORDER: Final = ("literal_jacobian", "corrected")
_ROLE_QUERY_INDEX: Final = {
    V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN: 0,
    V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY: 1,
}
_SUPPORTED_REDUCER_KINDS: Final = frozenset({"linear_weight", "conv_kernel_weight"})
_CAPTURE_PREFIX: Final = "partition-free-v1-factored-row-capture-"
_LAYER_PREFIX: Final = "partition-free-v1-factored-row-layer-"
_KINK_PREFIX: Final = "partition-free-v1-factored-row-own-kink-"
_F64_BYTES: Final = 8


class V1FactoredRowRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    AXIS_MISMATCH = "axis_mismatch"
    DTYPE_MISMATCH = "dtype_mismatch"
    DEVICE_MISMATCH = "device_mismatch"
    NONFINITE = "nonfinite"
    ROLE_MISMATCH = "role_mismatch"
    LAYER_KIND_UNSUPPORTED = "layer_kind_unsupported"
    STALE_IDENTITY = "stale_identity"
    MISSING_INPUT = "missing_input"
    FACTORED_PROVIDER_ABSENT = "factored_provider_absent"
    CPU_PRODUCTION_FALLBACK = "cpu_production_fallback"
    GRAM_NOT_PSD = "gram_not_psd"
    DENSE_FACTOR_NOT_ADMITTED = "dense_factor_not_admitted"
    ROW_WORKSPACE_NOT_ADMITTED = "row_workspace_not_admitted"
    EVICTED = "evicted"
    ACTIVE_PACKET = "active_packet"
    NO_ACTIVE_PACKET = "no_active_packet"
    ORDER_MISMATCH = "order_mismatch"
    MUTATED_CUSTODY = "mutated_custody"
    AUTHORITY_VIOLATION = "authority_violation"


class V1FactoredRowRefusal(RuntimeError):
    """Typed refusal of the factored-row family."""

    def __init__(self, code: V1FactoredRowRefusalCode, message: str) -> None:
        if type(code) is not V1FactoredRowRefusalCode:
            raise TypeError("factored-row refusal requires an exact code")
        super().__init__(f"{code.value}: {message}")
        self.code = code


def _refuse(code: V1FactoredRowRefusalCode, message: str) -> NoReturn:
    raise V1FactoredRowRefusal(code, message)


def _torch() -> Any:
    if torch is None:  # pragma: no cover
        _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "factored rows require PyTorch")
    return torch


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, f"{name} must be a nonempty string")
    return value


def _optional_identity(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _identity(value, name=name)


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, f"{name} must be a nonnegative int")
    return int(value)


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, f"{name} must be a positive int")
    return result


def _closed_flags(receipt: object, names: Sequence[str]) -> None:
    for name in names:
        if getattr(receipt, name) is not False:
            _refuse(
                V1FactoredRowRefusalCode.AUTHORITY_VIOLATION,
                f"factored-row receipts carry no {name}",
            )


def _plain(value: Any) -> Any:
    """Convert reducer parameters to hashable plain JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            _refuse(V1FactoredRowRefusalCode.NONFINITE, "reducer parameter is nonfinite")
        return number
    _refuse(
        V1FactoredRowRefusalCode.INVALID_TYPE,
        f"reducer parameter of type {type(value).__name__} is not hashable",
    )


def _reducer_parameters_for_kernel(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the geometric conv parameters the legacy kernel consumes."""

    return {
        name: value[name]
        for name in ("stride", "padding", "dilation", "groups")
        if name in value
    }


def _f32_tensor(value: Any, *, name: str, device: Any) -> Any:
    torch_module = _torch()
    if not isinstance(value, torch_module.Tensor):
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, f"{name} must be a torch tensor")
    if value.dtype != torch_module.float32:
        _refuse(
            V1FactoredRowRefusalCode.DTYPE_MISMATCH,
            f"{name} must be float32 (the legacy kernel dtype); got {value.dtype}",
        )
    if value.device != device:
        _refuse(V1FactoredRowRefusalCode.DEVICE_MISMATCH, f"{name} left the capture device")
    if not bool(torch_module.isfinite(value).all().item()):
        _refuse(V1FactoredRowRefusalCode.NONFINITE, f"{name} is nonfinite")
    return value.detach()


@dataclass(frozen=True)
class V1FactoredLayerFactors:
    """Factors of one route block for one context packet (process-local)."""

    layer_id: str
    reducer_kind: str
    route_block: RouteBlockSpec
    route_offset: int
    activation_kind: str
    scale: float
    abs_aggregation_mode: str
    reducer_parameters: Mapping[str, Any] = field(repr=False, compare=False)
    input: Any = field(repr=False, compare=False)
    post_activation_cotangent: Any = field(repr=False, compare=False)
    preactivation: Any = field(repr=False, compare=False)
    bias_baseline: Any = field(repr=False, compare=False)
    weight: Any = field(repr=False, compare=False)
    weight_identity_hash: str | None = None

    def __post_init__(self) -> None:
        torch_module = _torch()
        _identity(self.layer_id, name="layer id")
        if self.reducer_kind not in _SUPPORTED_REDUCER_KINDS:
            _refuse(
                V1FactoredRowRefusalCode.LAYER_KIND_UNSUPPORTED,
                f"factored rows support {sorted(_SUPPORTED_REDUCER_KINDS)}, not "
                f"{self.reducer_kind!r}",
            )
        if type(self.route_block) is not RouteBlockSpec:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "route block must be a RouteBlockSpec")
        self.route_block.validate()
        if self.route_block.route_kind != self.reducer_kind:
            _refuse(
                V1FactoredRowRefusalCode.AXIS_MISMATCH,
                "route block kind differs from the reducer kind",
            )
        object.__setattr__(self, "route_offset", _nonnegative_int(self.route_offset, name="route offset"))
        if self.activation_kind not in {"relu", "identity"}:
            _refuse(V1FactoredRowRefusalCode.LAYER_KIND_UNSUPPORTED, "unsupported activation kind")
        scale = float(self.scale)
        if not math.isfinite(scale) or scale == 0.0:
            _refuse(V1FactoredRowRefusalCode.NONFINITE, "capture scale must be finite and nonzero")
        object.__setattr__(self, "scale", scale)
        _identity(self.abs_aggregation_mode, name="abs aggregation mode")
        object.__setattr__(self, "reducer_parameters", dict(self.reducer_parameters))
        device = self.post_activation_cotangent.device if isinstance(
            self.post_activation_cotangent, torch_module.Tensor
        ) else None
        if device is None:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "cotangent must be a torch tensor")
        cot = _f32_tensor(self.post_activation_cotangent, name="post-activation cotangent", device=device)
        pre = _f32_tensor(self.preactivation, name="preactivation", device=device)
        source = _f32_tensor(self.input, name="layer input", device=device)
        weight = _f32_tensor(self.weight, name="layer weight", device=device)
        bias = _f32_tensor(self.bias_baseline, name="bias baseline", device=device)
        if tuple(cot.shape) != tuple(pre.shape):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "cotangent and preactivation shapes differ")
        contexts = int(pre.shape[0])
        if contexts <= 0 or int(source.shape[0]) != contexts:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "input and preactivation context axes differ")
        if tuple(self.route_block.shape) != tuple(int(v) for v in weight.shape):
            _refuse(
                V1FactoredRowRefusalCode.AXIS_MISMATCH,
                "route block shape differs from the layer weight",
            )
        if self.reducer_kind == "linear_weight":
            if source.ndim != 2 or weight.ndim != 2 or pre.ndim != 2:
                _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "linear factors require matrix geometry")
            if int(source.shape[1]) != int(weight.shape[1]) or int(pre.shape[1]) != int(weight.shape[0]):
                _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "linear factor axes do not compose")
        else:
            if source.ndim != 4 or weight.ndim != 4 or pre.ndim != 4:
                _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "conv factors require NCHW geometry")
            missing = sorted({"stride", "padding", "dilation", "groups"} - set(self.reducer_parameters))
            if missing:
                _refuse(
                    V1FactoredRowRefusalCode.MISSING_INPUT,
                    "conv factors lack reducer parameters: " + ",".join(missing),
                )
            groups = int(self.reducer_parameters["groups"])
            if groups <= 0 or int(source.shape[1]) != groups * int(weight.shape[1]) or int(pre.shape[1]) != int(weight.shape[0]):
                _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "conv factor axes do not compose")
        try:
            torch_module.broadcast_shapes(tuple(pre.shape), tuple(bias.shape))
        except RuntimeError as error:
            _refuse(
                V1FactoredRowRefusalCode.AXIS_MISMATCH,
                f"bias baseline does not broadcast over the preactivation: {error}",
            )
        object.__setattr__(self, "input", source)
        object.__setattr__(self, "post_activation_cotangent", cot)
        object.__setattr__(self, "preactivation", pre)
        object.__setattr__(self, "bias_baseline", bias)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(
            self,
            "weight_identity_hash",
            _optional_identity(self.weight_identity_hash, name="weight identity hash"),
        )

    @property
    def device(self) -> Any:
        return self.preactivation.device

    @property
    def context_count(self) -> int:
        return int(self.preactivation.shape[0])

    @property
    def route_count(self) -> int:
        return int(self.route_block.route_count)

    @property
    def route_stop(self) -> int:
        return self.route_offset + self.route_count

    def corrected_cotangent(self) -> Any:
        """``G * phi`` with the legacy helper, leading query axis of length one."""

        cot = self.post_activation_cotangent.unsqueeze(0)
        if self.activation_kind == "relu":
            return conserving_relu_cotangent(cot, self.preactivation, self.bias_baseline)
        return cot

    def literal_cotangent(self) -> Any:
        """``G * 1[a > 0]`` with the legacy helper, leading query axis of length one."""

        cot = self.post_activation_cotangent.unsqueeze(0)
        if self.activation_kind == "relu":
            return literal_relu_gate_jacobian_cotangent(cot, self.preactivation)
        return cot

    def stacked_cotangents(self) -> Any:
        """The legacy fused query stack ``cat((literal, corrected), dim=0)``."""

        torch_module = _torch()
        return torch_module.cat((self.literal_cotangent(), self.corrected_cotangent()), dim=0)

    def role_cotangent(self, role: V1SampledDualChartRowRole) -> Any:
        """Per-context cotangent of one role, ``[B, out(, H, W)]`` fp32."""

        if type(role) is not V1SampledDualChartRowRole:
            _refuse(V1FactoredRowRefusalCode.ROLE_MISMATCH, "row role must be exact")
        if role is V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY:
            return self.corrected_cotangent()[0]
        return self.literal_cotangent()[0]

    def row_cotangent(self, role: V1SampledDualChartRowRole) -> Any:
        """Replay the legacy fp32 gradient/scale operation before factor algebra."""

        cotangent = self.role_cotangent(role)
        # Both legacy row kernels replace tiny scales before their FP32 cast.
        # The descriptor retains the producer value; algebra replays its rule.
        effective_scale = 1.0 if abs(self.scale) < 1e-12 else self.scale
        if effective_scale == 1.0:
            return cotangent
        torch_module = _torch()
        scale = torch_module.as_tensor(effective_scale, dtype=torch_module.float32, device=self.device)
        result = cotangent / scale
        if not bool(torch_module.isfinite(result).all().item()):
            _refuse(V1FactoredRowRefusalCode.NONFINITE, "scaled row cotangent is nonfinite in fp32")
        return result

    @property
    def factor_bytes(self) -> int:
        """Bytes of the per-context factors (``x``, ``G``, ``a``); ``W``/``b`` are chart-level."""

        return int(
            sum(
                int(value.numel()) * int(value.element_size())
                for value in (self.input, self.post_activation_cotangent, self.preactivation)
            )
        )

    @property
    def chart_parameter_bytes(self) -> int:
        return int(
            int(self.weight.numel()) * int(self.weight.element_size())
            + int(self.bias_baseline.numel()) * int(self.bias_baseline.element_size())
        )

    def descriptor(self) -> dict[str, Any]:
        """Hashable, array-free description of this layer's factors."""

        return {
            "schema": FACTORED_ROW_LAYER_FACTOR_DESCRIPTOR_SCHEMA,
            "layer_id": self.layer_id,
            "reducer_kind": self.reducer_kind,
            "route_block": self.route_block.to_dict(),
            "route_offset": self.route_offset,
            "route_count": self.route_count,
            "activation_kind": self.activation_kind,
            "scale": self.scale,
            "abs_aggregation_mode": self.abs_aggregation_mode,
            "reducer_parameters": _plain(_reducer_parameters_for_kernel(self.reducer_parameters)),
            "context_count": self.context_count,
            "input_shape": [int(v) for v in self.input.shape],
            "cotangent_shape": [int(v) for v in self.post_activation_cotangent.shape],
            "preactivation_shape": [int(v) for v in self.preactivation.shape],
            "bias_shape": [int(v) for v in self.bias_baseline.shape],
            "weight_shape": [int(v) for v in self.weight.shape],
            "factor_dtype_identity": "torch.float32",
            "weight_identity_hash": self.weight_identity_hash,
            "weight_identity_source": (
                "caller_declared" if self.weight_identity_hash is not None else "chart_identity_hash_only"
            ),
            "factor_bytes": self.factor_bytes,
            "chart_parameter_bytes": self.chart_parameter_bytes,
            "kernel_query_order": list(FACTORED_ROW_KERNEL_QUERY_ORDER),
        }

    @property
    def descriptor_hash(self) -> str:
        return content_hash(self.descriptor(), prefix=_LAYER_PREFIX)

    def legacy_fp32_rows(self) -> Any:
        """Replay the legacy fp32 kernel; returns ``[B, 2, route_count]`` fp32.

        Query index 0 is the literal-J role, index 1 the corrected-A role.
        """

        torch_module = _torch()
        stacked = self.stacked_cotangents()
        if self.reducer_kind == "linear_weight":
            from packages.salience.ops.target_batched import (
                PER_CONTEXT_SCORE_HEAD,
                linear_frontier_aggregate_target_batched,
            )

            result = linear_frontier_aggregate_target_batched(
                self.input,
                stacked,
                weights=self.weight,
                scale=self.scale,
                device=stacked.device,
                abs_aggregation_mode=self.abs_aggregation_mode,
                compute_neuron_metrics=False,
                compute_legacy_oldfast_abs=False,
                retain_per_context=True,
                retain_local_destination=False,
            )
            rows = result[PER_CONTEXT_SCORE_HEAD]
        else:
            from mechanistic_systems.backends.torch_runtime import (
                conv_kernel_route_measurement_with_context_target_batched,
            )

            parameters = self.reducer_parameters
            _signed, _absolute, rows, _local = (
                conv_kernel_route_measurement_with_context_target_batched(
                    self.input,
                    stacked,
                    weights=self.weight,
                    scale=self.scale,
                    stride=parameters["stride"],
                    padding=parameters["padding"],
                    dilation=parameters["dilation"],
                    groups=int(parameters["groups"]),
                    device=stacked.device,
                    abs_aggregation_mode=self.abs_aggregation_mode,
                    retain_local_destination=False,
                    per_context_consumer=None,
                    profiler=None,
                )
            )
        rows = rows.detach()
        if rows.dtype != torch_module.float32:
            _refuse(V1FactoredRowRefusalCode.DTYPE_MISMATCH, "legacy kernel changed its fp32 row dtype")
        return rows.reshape(self.context_count, 2, self.route_count)

    def legacy_fp32_row_window(
        self, role: V1SampledDualChartRowRole, start: int, stop: int,
    ) -> Any:
        """Replay one role on a local route window without a full linear row.

        Linear row construction has no reduction: FP32 ``cot/scale``, outer
        product, then a separate FP32 weight multiplication. Destination
        slicing therefore preserves that operation order. At most two partial
        destination fibers surround the requested window. Convolution keeps
        the original full-layer kernel, whose reduction order is not changed.
        """
        if type(role) is not V1SampledDualChartRowRole:
            _refuse(V1FactoredRowRefusalCode.ROLE_MISMATCH, "exact row role required")
        start = _nonnegative_int(start, name="local route start")
        stop = _positive_int(stop, name="local route stop")
        if not start < stop <= self.route_count:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "local row window exceeds layer")
        if self.reducer_kind != "linear_weight":
            return self.legacy_fp32_rows()[:, _ROLE_QUERY_INDEX[role], start:stop]
        return self._legacy_fp32_linear_row_window_from_cotangent(
            self.row_cotangent(role), start, stop)

    def _legacy_fp32_linear_row_window_from_cotangent(
        self, row_cotangent: Any, start: int, stop: int,
    ) -> Any:
        """Replay a validated linear window with its role/scale-prepared cotangent.

        The caller owns the preparation and its lifetime. Keeping it outside a
        native tile loop avoids repeating whole-layer cotangent validation and
        construction; this helper neither caches it nor changes FP32 row order.
        """
        inner = int(self.weight.shape[1])
        first, last = start // inner, (stop + inner - 1) // inner
        cot = row_cotangent[:, first:last]
        # Keep these as separate eager FP32 operations. Reassociation or
        # fusion with W would change the registered rounded-row estimand.
        rows = cot[:, :, None] * self.input[:, None, :]
        rows.mul_(self.weight[None, first:last, :])
        return rows.reshape(self.context_count, -1)[:, start - first * inner:stop - first * inner]


@dataclass(frozen=True)
class V1FactoredRowPacketReceipt:
    """Durable, array-free receipt of one factored context packet (V2 family)."""

    physical_route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    context_ids: tuple[str, ...]
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    layer_descriptors: tuple[Mapping[str, Any], ...]
    layer_descriptor_hashes: tuple[str, ...]
    query_id: str
    vjp_backend_used: str
    factor_bytes: int
    chart_parameter_bytes: int
    avoided_dense_row_bytes: int
    parent_packet_plan_hash: str | None = None
    parent_look_boundary_hash: str | None = None
    factor_dtype_identity: str = "torch.float32"
    equality_law_id: str = FACTORED_ROW_EQUALITY_LAW_ID
    symmetry_covariance_law_id: str = FACTORED_ROW_SYMMETRY_COVARIANCE_LAW_ID
    kernel_query_order: tuple[str, ...] = FACTORED_ROW_KERNEL_QUERY_ORDER
    route_scale_d2h_bytes: int = 0
    route_scale_host_tensor_materializations: int = 0
    D_stability_authority: bool = False
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = FACTORED_ROW_CAPTURE_AUTHORITY_SCOPE
    schema: str = FACTORED_ROW_CAPTURE_FAMILY_SCHEMA
    _sealed_hash: str = field(init=False, default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "physical_route_axis_hash",
            "route_id_axis_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "corrected_a_stream_identity_hash",
            "literal_j_stream_identity_hash",
            "query_id",
            "vjp_backend_used",
        ):
            _identity(getattr(self, name), name=name)
        if self.physical_route_axis_hash == self.route_id_axis_hash:
            _refuse(
                V1FactoredRowRefusalCode.AXIS_MISMATCH,
                "physical and ordered route axes must remain distinct (two-axis law)",
            )
        if self.corrected_a_stream_identity_hash == self.literal_j_stream_identity_hash:
            _refuse(V1FactoredRowRefusalCode.ROLE_MISMATCH, "A and J streams must remain distinct")
        object.__setattr__(self, "route_count", _positive_int(self.route_count, name="route count"))
        object.__setattr__(
            self, "proposal_generation", _nonnegative_int(self.proposal_generation, name="proposal generation")
        )
        contexts = tuple(_identity(value, name="context id") for value in self.context_ids)
        if not contexts or len(set(contexts)) != len(contexts):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "context ids must be nonempty and unique")
        object.__setattr__(self, "context_ids", contexts)
        descriptors = tuple(dict(item) for item in self.layer_descriptors)
        hashes = tuple(_identity(value, name="layer descriptor hash") for value in self.layer_descriptor_hashes)
        if not descriptors or len(descriptors) != len(hashes):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "layer descriptors and hashes must align")
        if tuple(content_hash(item, prefix=_LAYER_PREFIX) for item in descriptors) != hashes:
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "layer descriptor hashes do not replay")
        offset = 0
        for item in descriptors:
            if int(item["route_offset"]) != offset:
                _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "layer descriptors are not contiguous")
            offset += int(item["route_count"])
        if offset != self.route_count:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "layer descriptors do not close the route axis")
        object.__setattr__(self, "layer_descriptors", descriptors)
        object.__setattr__(self, "layer_descriptor_hashes", hashes)
        for name in ("factor_bytes", "chart_parameter_bytes", "avoided_dense_row_bytes"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name=name))
        if self.avoided_dense_row_bytes != 2 * len(contexts) * self.route_count * _F64_BYTES:
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "avoided dense row bytes do not replay")
        object.__setattr__(
            self, "parent_packet_plan_hash", _optional_identity(self.parent_packet_plan_hash, name="parent packet plan hash")
        )
        object.__setattr__(
            self, "parent_look_boundary_hash", _optional_identity(self.parent_look_boundary_hash, name="parent look boundary hash")
        )
        if self.route_scale_d2h_bytes != 0 or self.route_scale_host_tensor_materializations != 0:
            _refuse(V1FactoredRowRefusalCode.AUTHORITY_VIOLATION, "factored capture performs no route-scale D2H")
        if (
            self.schema != FACTORED_ROW_CAPTURE_FAMILY_SCHEMA
            or self.equality_law_id != FACTORED_ROW_EQUALITY_LAW_ID
            or self.symmetry_covariance_law_id != FACTORED_ROW_SYMMETRY_COVARIANCE_LAW_ID
            or tuple(self.kernel_query_order) != FACTORED_ROW_KERNEL_QUERY_ORDER
            or self.authority_scope != FACTORED_ROW_CAPTURE_AUTHORITY_SCOPE
            or self.factor_dtype_identity != "torch.float32"
        ):
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored receipt schema or law drifted")
        _closed_flags(
            self,
            (
                "D_stability_authority",
                "C_authority",
                "calibration_authority",
                "oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_authority",
            ),
        )
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _hash_payload(self) -> dict[str, Any]:
        payload = {
            name: getattr(self, name)
            for name in (
                "physical_route_axis_hash",
                "route_id_axis_hash",
                "route_count",
                "context_ids",
                "chart_identity_hash",
                "current_mask_hash",
                "proposal_generation",
                "corrected_a_stream_identity_hash",
                "literal_j_stream_identity_hash",
                "layer_descriptors",
                "layer_descriptor_hashes",
                "query_id",
                "vjp_backend_used",
                "factor_bytes",
                "chart_parameter_bytes",
                "avoided_dense_row_bytes",
                "parent_packet_plan_hash",
                "parent_look_boundary_hash",
                "factor_dtype_identity",
                "equality_law_id",
                "symmetry_covariance_law_id",
                "kernel_query_order",
                "route_scale_d2h_bytes",
                "route_scale_host_tensor_materializations",
                "D_stability_authority",
                "C_authority",
                "calibration_authority",
                "oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_authority",
                "authority_scope",
                "schema",
            )
        }
        return payload

    def _compute_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_CAPTURE_PREFIX)

    @property
    def receipt_hash(self) -> str:
        return self._sealed_hash

    def assert_integrity(self) -> str:
        if self._compute_hash() != self._sealed_hash:
            _refuse(V1FactoredRowRefusalCode.MUTATED_CUSTODY, "factored receipt hash does not replay")
        return self._sealed_hash

    def to_json_payload(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload["receipt_hash"] = self.receipt_hash
        return payload

    @property
    def context_count(self) -> int:
        return len(self.context_ids)


@dataclass(frozen=True)
class V1FactoredOwnKinkSetDamage:
    """Exact own-kink, downstream-linearized change of ``s`` for one removal set."""

    context_count: int
    layer_ids: tuple[str, ...]
    total_change: np.ndarray = field(repr=False, compare=False)
    first_order_change: np.ndarray = field(repr=False, compare=False)
    kink_remainder: np.ndarray = field(repr=False, compare=False)
    flipped_gate_count: np.ndarray = field(repr=False, compare=False)
    in_layer_wall_dose: tuple[float | None, ...] = ()
    removed_route_count: int = 0
    orientation: str = "change_in_differentiated_scalar_positive_removal_work_is_negative"
    claim_scope: str = (
        "acted_layers_own_gates_only_downstream_linearized_by_frozen_G_no_pool_or_"
        "cross_layer_cascade"
    )
    schema: str = FACTORED_ROW_OWN_KINK_SET_DAMAGE_SCHEMA

    def __post_init__(self) -> None:
        count = _positive_int(self.context_count, name="context count")
        arrays = {}
        for name in ("total_change", "first_order_change", "kink_remainder"):
            array = np.ascontiguousarray(np.asarray(getattr(self, name), dtype=np.float64))
            if array.shape != (count,) or not np.isfinite(array).all():
                _refuse(V1FactoredRowRefusalCode.NONFINITE, f"{name} must be finite with one value per context")
            array.setflags(write=False)
            arrays[name] = array
        flips = np.ascontiguousarray(np.asarray(self.flipped_gate_count, dtype=np.int64))
        if flips.shape != (count,) or bool(np.any(flips < 0)):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "flipped gate counts must be nonnegative per context")
        flips.setflags(write=False)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "flipped_gate_count", flips)
        doses = tuple(self.in_layer_wall_dose)
        if len(doses) != count:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "one wall dose (or None) per context")
        for dose in doses:
            if dose is not None and (not isinstance(dose, float) or not math.isfinite(dose) or dose <= 0.0):
                _refuse(V1FactoredRowRefusalCode.NONFINITE, "wall dose must be a finite positive float or None")
        object.__setattr__(self, "in_layer_wall_dose", doses)
        object.__setattr__(self, "removed_route_count", _nonnegative_int(self.removed_route_count, name="removed route count"))
        object.__setattr__(self, "layer_ids", tuple(_identity(v, name="layer id") for v in self.layer_ids))

    def receipt_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "context_count": self.context_count,
            "layer_ids": list(self.layer_ids),
            "removed_route_count": self.removed_route_count,
            "total_change": [float(v) for v in self.total_change],
            "first_order_change": [float(v) for v in self.first_order_change],
            "kink_remainder": [float(v) for v in self.kink_remainder],
            "flipped_gate_count": [int(v) for v in self.flipped_gate_count],
            "in_layer_wall_dose": [None if v is None else float(v) for v in self.in_layer_wall_dose],
            "orientation": self.orientation,
            "claim_scope": self.claim_scope,
        }

    @property
    def receipt_hash(self) -> str:
        return content_hash(self.receipt_payload(), prefix=_KINK_PREFIX)


class V1FactoredRowPacket:
    """Process-local owner of one context packet's factors (nonserializable)."""

    __slots__ = ("_receipt", "_layers", "_device", "_evicted", "_factor_custody")

    def __init__(
        self,
        *,
        receipt: V1FactoredRowPacketReceipt,
        layers: Sequence[V1FactoredLayerFactors],
    ) -> None:
        if type(receipt) is not V1FactoredRowPacketReceipt:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "packet requires its exact receipt")
        receipt.assert_integrity()
        layer_tuple = tuple(layers)
        if not layer_tuple or any(type(item) is not V1FactoredLayerFactors for item in layer_tuple):
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "packet requires exact layer factors")
        if tuple(item.descriptor_hash for item in layer_tuple) != receipt.layer_descriptor_hashes:
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "layer factors differ from the receipt")
        devices = {str(item.device) for item in layer_tuple}
        if len(devices) != 1:
            _refuse(V1FactoredRowRefusalCode.DEVICE_MISMATCH, "packet layers span several devices")
        if any(item.context_count != receipt.context_count for item in layer_tuple):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "layer context counts differ from the receipt")
        self._receipt = receipt
        self._layers = layer_tuple
        self._device = layer_tuple[0].device
        self._evicted = False
        self._factor_custody = self._factor_tensor_custody()

    def _factor_tensor_custody(self) -> tuple[Any, ...]:
        """Process-local tensor version/storage guards; no tensor transfer or hashing.

        Like the resident-row custody guard, this detects ordinary PyTorch
        writes and storage/view replacement. It is not a content checksum for
        external writes that bypass PyTorch's version counter.
        """

        result = []
        for layer in self._layers:
            for name in ("input", "post_activation_cotangent", "preactivation", "bias_baseline", "weight"):
                value = getattr(layer, name)
                if not _torch().is_tensor(value):
                    _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, f"factored {name} lost tensor custody")
                try:
                    version = int(value._version)
                except RuntimeError:
                    _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored custody requires version-tracked tensors")
                result.append((id(value), str(value.device), value.dtype, tuple(value.shape),
                               tuple(value.stride()), int(value.storage_offset()),
                               int(value.data_ptr()), int(value.untyped_storage().data_ptr()), version))
        return tuple(result)

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("factored row packets are nonserializable; persist the receipt only")

    def __copy__(self) -> NoReturn:
        raise TypeError("factored row packets cannot be copied")

    def __deepcopy__(self, _memo: dict[int, object]) -> NoReturn:
        raise TypeError("factored row packets cannot be copied")

    def _assert_live(self) -> None:
        if self._evicted:
            _refuse(V1FactoredRowRefusalCode.EVICTED, "factored row packet was evicted")
        current = self._factor_tensor_custody()
        if current != self._factor_custody:
            fields = ("input", "post_activation_cotangent", "preactivation", "bias_baseline", "weight")
            changed = [
                f"{self._layers[index // len(fields)].layer_id}.{fields[index % len(fields)]}"
                for index, (before, after) in enumerate(zip(self._factor_custody, current, strict=True))
                if before != after
            ]
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY,
                    "factored packet tensors changed after capture: " + ", ".join(changed))

    def assert_integrity(self) -> str:
        """Reopen scalar descriptors and live tensor custody at a source join."""

        self._assert_live()
        receipt_hash = self._receipt.assert_integrity()
        if tuple(layer.descriptor_hash for layer in self._layers) != self._receipt.layer_descriptor_hashes:
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored packet layer descriptors changed after capture")
        return receipt_hash

    @property
    def receipt(self) -> V1FactoredRowPacketReceipt:
        self._assert_live()
        return self._receipt

    @property
    def layers(self) -> tuple[V1FactoredLayerFactors, ...]:
        self._assert_live()
        return self._layers

    @property
    def device(self) -> Any:
        return self._device

    @property
    def route_count(self) -> int:
        return self._receipt.route_count

    @property
    def context_count(self) -> int:
        return self._receipt.context_count

    @property
    def factor_bytes(self) -> int:
        return self._receipt.factor_bytes

    def evict(self) -> None:
        self._evicted = True
        self._layers = ()

    # -- gate ---------------------------------------------------------------

    def gate64(self, gate: Any) -> Any:
        """Return the 0/1 fp64 route gate on the packet device (exact)."""

        return self._gate64(gate)

    def _gate64(self, gate: Any) -> Any:

        torch_module = _torch()
        if isinstance(gate, HardMask):
            if gate.route_count != self.route_count:
                _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "hard mask route count differs from the packet axis")
            if gate.route_axis_hash != self._receipt.physical_route_axis_hash:
                _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "hard mask names another physical route axis")
            if gate.mask_hash != self._receipt.current_mask_hash:
                # The factors were captured on the chart of ``current_mask_hash``
                # (the masked network); another hard mask is another chart.
                _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "hard mask is not the packet's current mask")
            values = torch_module.as_tensor(gate.to_bool_array(), dtype=torch_module.bool)
        elif isinstance(gate, torch_module.Tensor):
            values = gate.detach().to(dtype=torch_module.bool).reshape(-1)
        else:
            values = torch_module.as_tensor(np.asarray(gate, dtype=bool).reshape(-1), dtype=torch_module.bool)
        if int(values.numel()) != self.route_count:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "route gate differs from the packet route axis")
        return values.to(device=self._device, dtype=torch_module.float64)

    @staticmethod
    def _role_index(role: V1SampledDualChartRowRole) -> int:
        if type(role) is not V1SampledDualChartRowRole:
            _refuse(V1FactoredRowRefusalCode.ROLE_MISMATCH, "row role must be an exact V1SampledDualChartRowRole")
        return _ROLE_QUERY_INDEX[role]

    # -- legacy-equal materialization ----------------------------------------

    def materialize_rows(
        self,
        role: V1SampledDualChartRowRole,
        route_start: int,
        route_stop: int,
        *,
        gate: Any,
    ) -> Any:
        """Reproduce the legacy fp64 rows of ``role`` on ``[route_start, route_stop)``.

        Bitwise equal (same device) to the legacy producer's chunk: the fp32
        kernel is replayed per block, then ``to(float64)``, ``mul_(gate)``,
        ``neg_()`` exactly as ``production.py`` does.
        """

        self._assert_live()
        torch_module = _torch()
        self._role_index(role)
        start = _nonnegative_int(route_start, name="route start")
        stop = _positive_int(route_stop, name="route stop")
        if stop <= start or stop > self.route_count:
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "route window is empty or exceeds the axis")
        gate64 = self._gate64(gate)
        parts = []
        for layer in self._layers:
            lo = max(start, layer.route_offset)
            hi = min(stop, layer.route_stop)
            if lo >= hi:
                continue
            rows32 = layer.legacy_fp32_row_window(
                role, lo - layer.route_offset, hi - layer.route_offset,
            )
            chunk = rows32.to(
                dtype=torch_module.float64
            ).contiguous()
            chunk.mul_(gate64[lo:hi][None, :])
            chunk.neg_()
            parts.append(chunk)
        return torch_module.cat(parts, dim=1).contiguous()

    def _native_fiber_axis(self, native_fiber_axis: Any) -> Any:
        """Constructively join native cuts to this packet's physical blocks."""
        from contracts.hashing import stable_digest
        from .modern_v1_native_fiber_axis import V1NativeDestinationFiberAxisReceipt

        self.assert_integrity()
        axis = native_fiber_axis
        if type(axis) is not V1NativeDestinationFiberAxisReceipt:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "exact native fiber axis required")
        axis.assert_integrity()
        if (axis.route_axis_hash != self.receipt.physical_route_axis_hash
                or axis.route_id_axis_hash != self.receipt.route_id_axis_hash
                or axis.route_count != self.route_count
                or len(axis.cuts) != len(self._layers)):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "native fiber and packet axes differ")
        for cut, layer in zip(axis.cuts, self._layers, strict=True):
            if (cut.layer_id != layer.layer_id
                    or cut.route_kind != layer.reducer_kind
                    or cut.route_block_content_hash != stable_digest(layer.route_block.to_dict())
                    or cut.route_shape != tuple(layer.weight.shape)
                    or cut.activation_kind != layer.activation_kind
                    or cut.route_offset != layer.route_offset
                    or cut.route_stop != layer.route_stop
                    or cut.fiber_count != int(layer.weight.shape[0])):
                _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "native cut differs from captured layer")
        return axis

    def _native_fiber_row_tiles(self, role: V1SampledDualChartRowRole, axis: Any,
                               gate64: Any, route_tile_size: int,
                               legacy_conv_row_budget_bytes: int):
        """Yield shared rounded coefficients for a linear map and its transpose.

        Scratch holds one packet/layer tile; no context-by-whole-model bank is
        created. A tile contains at least one complete destination fiber.
        Convolution's legacy full-layer scratch is retained and must be included
        in GPU resource qualification separately from linear tile storage.
        """
        self._role_index(role)
        size = _positive_int(route_tile_size, name="native route tile size")
        budget = _nonnegative_int(legacy_conv_row_budget_bytes, name="legacy conv row budget")
        torch_module = _torch()
        # Admission precedes every legacy convolution call. This budget covers
        # its retained two-role FP32 row payload, not unfold/BLAS workspace;
        # total peak allocation remains a separate hardware qualification.
        for layer in self._layers:
            if (layer.reducer_kind != "linear_weight"
                    and 2 * self.context_count * layer.route_count * 4 > budget):
                _refuse(V1FactoredRowRefusalCode.ROW_WORKSPACE_NOT_ADMITTED,
                        "legacy conv two-role row payload exceeds its budget")
        for cut, layer in zip(axis.cuts, self._layers, strict=True):
            step = max(1, size // cut.routes_per_fiber)
            conv_rows = None
            row_cotangent = None
            if layer.reducer_kind == "linear_weight":
                # Preparation and all legacy scale/finite checks occur once
                # per layer in this pass; no cotangent survives the layer loop.
                row_cotangent = layer.row_cotangent(role)
            else:
                conv_rows = layer.legacy_fp32_rows()[:, _ROLE_QUERY_INDEX[role], :]
            for first in range(0, cut.fiber_count, step):
                last = min(cut.fiber_count, first + step)
                lo, hi = first * cut.routes_per_fiber, last * cut.routes_per_fiber
                rows32 = (layer._legacy_fp32_linear_row_window_from_cotangent(
                              row_cotangent, lo, hi)
                          if conv_rows is None else conv_rows[:, lo:hi])
                rows = rows32.to(dtype=torch_module.float64).contiguous()
                rows.mul_(gate64[layer.route_offset + lo:layer.route_offset + hi][None, :])
                rows.neg_()
                yield (cut.fiber_offset + first, cut.fiber_offset + last,
                       layer.route_offset + lo, layer.route_offset + hi,
                       rows.reshape(self.context_count, last - first, cut.routes_per_fiber))
                del rows, rows32
            del conv_rows, row_cotangent

    def native_fiber_forward(self, role: V1SampledDualChartRowRole, directions: Any,
                             *, native_fiber_axis: Any, gate: Any,
                             route_tile_size: int = 262144,
                             legacy_conv_row_budget_bytes: int = 256 << 20) -> Any:
        """Rounded-row signed map ``[routes, lanes] -> [contexts, fibers, lanes]``.

        This is a numerical primitive, not a population or decision certificate.
        The FP64 contractions require their own outward error analysis at the
        certification boundary. Signed cancellation is retained within fibers.
        """
        axis = self._native_fiber_axis(native_fiber_axis)
        torch_module = _torch()
        z = torch_module.as_tensor(directions, dtype=torch_module.float64, device=self._device)
        if z.ndim == 1:
            z = z[:, None]
        if (z.ndim != 2 or int(z.shape[0]) != self.route_count or int(z.shape[1]) <= 0
                or not bool(torch_module.isfinite(z).all().item())):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "native directions must be finite [routes, lanes]")
        lanes = int(z.shape[1])
        result = torch_module.empty((self.context_count, axis.fiber_count, lanes),
                                    dtype=torch_module.float64, device=self._device)
        gate64 = self._gate64(gate)
        for first, last, lo, hi, rows in self._native_fiber_row_tiles(
                role, axis, gate64, route_tile_size, legacy_conv_row_budget_bytes):
            result[:, first:last, :] = torch_module.einsum(
                "bfs,fsl->bfl", rows, z[lo:hi].reshape(last - first, -1, lanes))
            del rows
        if not bool(torch_module.isfinite(result).all().item()):
            _refuse(V1FactoredRowRefusalCode.NONFINITE, "native forward is nonfinite")
        return result

    def native_fiber_adjoint(self, role: V1SampledDualChartRowRole, dual: Any,
                             *, native_fiber_axis: Any, gate: Any,
                             route_tile_size: int = 262144,
                             legacy_conv_row_budget_bytes: int = 256 << 20) -> Any:
        """Transpose of the same rounded-row map, returning ``[routes, lanes]``.

        Transpose means the transpose of the represented coefficient matrix;
        FP64 application is rounded and is not an exact-real arithmetic claim.
        Context weights and cut measures are applied by the enclosing solver,
        not implicitly inserted into either side of this linear pair.
        """
        axis = self._native_fiber_axis(native_fiber_axis)
        torch_module = _torch()
        y = torch_module.as_tensor(dual, dtype=torch_module.float64, device=self._device)
        if (y.ndim != 3 or tuple(y.shape[:2]) != (self.context_count, axis.fiber_count)
                or int(y.shape[2]) <= 0 or not bool(torch_module.isfinite(y).all().item())):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "native dual must be finite [contexts, fibers, lanes]")
        lanes = int(y.shape[2])
        result = torch_module.empty((self.route_count, lanes), dtype=torch_module.float64, device=self._device)
        gate64 = self._gate64(gate)
        for first, last, lo, hi, rows in self._native_fiber_row_tiles(
                role, axis, gate64, route_tile_size, legacy_conv_row_budget_bytes):
            result[lo:hi] = torch_module.einsum(
                "bfs,bfl->fsl", rows, y[:, first:last, :]).reshape(hi - lo, lanes)
            del rows
        if not bool(torch_module.isfinite(result).all().item()):
            _refuse(V1FactoredRowRefusalCode.NONFINITE, "native adjoint is nonfinite")
        return result

    # -- fp64 helpers -----------------------------------------------------------

    def _layer_rows64(self, layer: V1FactoredLayerFactors, role: V1SampledDualChartRowRole, gate64: Any) -> Any:
        """Dense fp64 rows of one layer (conv algebra; small blocks only)."""

        torch_module = _torch()
        index = self._role_index(role)
        rows = layer.legacy_fp32_rows()[:, index, :].to(dtype=torch_module.float64)
        rows = rows * gate64[layer.route_offset : layer.route_stop][None, :]
        return -rows

    @staticmethod
    def _linear_weight_gate(layer: V1FactoredLayerFactors, gate64: Any) -> Any:
        out, inner = tuple(int(v) for v in layer.weight.shape)
        return layer.weight.to(dtype=gate64.dtype) * gate64[layer.route_offset : layer.route_stop].reshape(out, inner)

    # -- bilinear reduction --------------------------------------------------

    def inner(self, role: V1SampledDualChartRowRole, directions: Any, *, gate: Any) -> Any:
        """``⟨row(x), V_k⟩`` for every context and direction, fp64 ``[B, k]``."""

        self._assert_live()
        torch_module = _torch()
        gate64 = self._gate64(gate)
        V = torch_module.as_tensor(directions, dtype=torch_module.float64, device=self._device)
        if V.ndim == 1:
            V = V.unsqueeze(0)
        if V.ndim != 2 or int(V.shape[1]) != self.route_count:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "directions must be [k, route_count]")
        result = torch_module.zeros((self.context_count, int(V.shape[0])), dtype=torch_module.float64, device=self._device)
        for layer in self._layers:
            block = V[:, layer.route_offset : layer.route_stop]
            if layer.reducer_kind == "linear_weight":
                out, inner = tuple(int(v) for v in layer.weight.shape)
                weighted = block.reshape(-1, out, inner) * self._linear_weight_gate(layer, gate64)[None]
                cot = layer.row_cotangent(role).to(dtype=torch_module.float64)
                source = layer.input.to(dtype=torch_module.float64)
                result -= torch_module.einsum("bo,koi,bi->bk", cot, weighted, source)
            else:
                result += self._layer_rows64(layer, role, gate64) @ block.T
        return result

    # -- Gram --------------------------------------------------------------------

    def gram(
        self,
        role: V1SampledDualChartRowRole,
        other: "V1FactoredRowPacket | None" = None,
        *,
        gate: Any,
    ) -> Any:
        """Exact row Gram ``K[x, x'] = ⟨row(x), row(x')⟩`` in fp64."""

        self._assert_live()
        torch_module = _torch()
        partner = self if other is None else other
        if type(partner) is not V1FactoredRowPacket:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "gram partner must be a factored row packet")
        partner._assert_live()
        if (
            partner._receipt.layer_descriptor_hashes != self._receipt.layer_descriptor_hashes
            and tuple(item["route_block"] for item in partner._receipt.layer_descriptors)
            != tuple(item["route_block"] for item in self._receipt.layer_descriptors)
        ):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "gram partner has another route-block axis")
        gate64 = self._gate64(gate)
        result = torch_module.zeros((self.context_count, partner.context_count), dtype=torch_module.float64, device=self._device)
        for mine, theirs in zip(self._layers, partner._layers, strict=True):
            # The Gram law is a same-chart law: both packets must share W (and
            # the gate).  A declared weight identity decides when present;
            # otherwise the device tensors are compared exactly.
            same_chart = (
                mine.weight_identity_hash == theirs.weight_identity_hash
                if mine.weight_identity_hash is not None and theirs.weight_identity_hash is not None
                else (
                    tuple(mine.weight.shape) == tuple(theirs.weight.shape)
                    and bool(torch_module.equal(mine.weight, theirs.weight))
                )
            )
            if not same_chart or mine.reducer_kind != theirs.reducer_kind or mine.scale != theirs.scale:
                _refuse(
                    V1FactoredRowRefusalCode.STALE_IDENTITY,
                    f"gram partner layer {mine.layer_id!r} belongs to another chart",
                )
            if mine.reducer_kind == "linear_weight":
                weight_gate = self._linear_weight_gate(mine, gate64)
                squared = weight_gate * weight_gate
                cot_a = mine.row_cotangent(role).to(dtype=torch_module.float64)
                cot_b = theirs.row_cotangent(role).to(dtype=torch_module.float64)
                x_a = mine.input.to(dtype=torch_module.float64)
                x_b = theirs.input.to(dtype=torch_module.float64)
                pair_inputs = x_a[:, None, :] * x_b[None, :, :]
                m = torch_module.einsum("oi,abi->abo", squared, pair_inputs)
                result += torch_module.einsum("ao,bo,abo->ab", cot_a, cot_b, m)
            else:
                result += self._layer_rows64(mine, role, gate64) @ partner._layer_rows64(theirs, role, gate64).T
        return result

    # -- moments ------------------------------------------------------------------

    def moments(
        self,
        role: V1SampledDualChartRowRole,
        power: int,
        weights: Any,
        *,
        gate: Any,
        absolute: bool = False,
    ) -> Any:
        """``Σ_x w_x row_r(x)^p`` (or of ``|row|``) for every route, fp64 ``[N]``."""

        self._assert_live()
        torch_module = _torch()
        p = _positive_int(power, name="moment power")
        w = torch_module.as_tensor(weights, dtype=torch_module.float64, device=self._device).reshape(-1)
        if int(w.numel()) != self.context_count or not bool(torch_module.isfinite(w).all().item()):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "one finite weight per context is required")
        gate64 = self._gate64(gate)
        result = torch_module.empty(self.route_count, dtype=torch_module.float64, device=self._device)
        sign = 1.0 if (absolute or p % 2 == 0) else -1.0
        for layer in self._layers:
            if layer.reducer_kind == "linear_weight":
                cot = layer.row_cotangent(role).to(dtype=torch_module.float64)
                source = layer.input.to(dtype=torch_module.float64)
                weight_gate = self._linear_weight_gate(layer, gate64)
                if absolute:
                    cot = cot.abs()
                    source = source.abs()
                    weight_gate = weight_gate.abs()
                block = torch_module.einsum("bo,b,bi->oi", cot**p, w, source**p) * weight_gate**p
                result[layer.route_offset : layer.route_stop] = sign * block.reshape(-1)
            else:
                rows = self._layer_rows64(layer, role, gate64)
                if absolute:
                    rows = rows.abs()
                result[layer.route_offset : layer.route_stop] = torch_module.einsum("b,br->r", w, rows**p)
        return result

    def combine_rows(
        self,
        role: V1SampledDualChartRowRole,
        coefficients: Any,
        *,
        gate: Any,
        out: Any | None = None,
        workspace_bytes: int = DEFAULT_COMBINE_WORKSPACE_BYTES,
    ) -> Any:
        """``Σ_x C[x, k] row(x)`` for every column ``k``; fp64 ``[k, N]``.

        The only route-scale object is the output.  ``coefficients`` is
        ``[B, k]`` (or ``[B]`` for one combination).  With ``out`` (an fp64
        ``[k, N]`` tensor or view on the packet's device) the combination is
        ACCUMULATED into it in place and ``out`` is returned, so a caller
        summing over packets never holds two route-scale tensors.  The linear
        einsum intermediate ``[k_chunk, out, in]`` is bounded by
        ``workspace_bytes`` by chunking over ``k`` (unchunked, the rank-128
        intermediate on fc1 was 36 GiB).
        """

        self._assert_live()
        torch_module = _torch()
        C = torch_module.as_tensor(coefficients, dtype=torch_module.float64, device=self._device)
        if C.ndim == 1:
            C = C.unsqueeze(1)
        if C.ndim != 2 or int(C.shape[0]) != self.context_count or not bool(torch_module.isfinite(C).all().item()):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "combination coefficients must be finite [B, k]")
        k = int(C.shape[1])
        workspace = _nonnegative_int(workspace_bytes, name="combine workspace bytes")
        if workspace <= 0:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "combine workspace must be positive")
        gate64 = self._gate64(gate)
        if out is None:
            result = torch_module.zeros((k, self.route_count), dtype=torch_module.float64, device=self._device)
        else:
            result = out
            if (
                not torch_module.is_tensor(result)
                or tuple(int(v) for v in result.shape) != (k, self.route_count)
                or result.dtype != torch_module.float64
                or result.device != torch_module.device(self._device)
            ):
                _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "combine output must be an fp64 [k, N] tensor on the packet device")
        for layer in self._layers:
            window = result[:, layer.route_offset : layer.route_stop]
            if layer.reducer_kind == "linear_weight":
                cot = layer.row_cotangent(role).to(dtype=torch_module.float64)
                source = layer.input.to(dtype=torch_module.float64)
                weight_gate = self._linear_weight_gate(layer, gate64)
                out_units, in_units = tuple(int(v) for v in layer.weight.shape)
                chunk = max(1, min(k, workspace // max(1, out_units * in_units * 8)))
                for ks in range(0, k, chunk):
                    ke = min(k, ks + chunk)
                    block = torch_module.einsum("bo,bk,bi->koi", cot, C[:, ks:ke], source)
                    block.mul_(weight_gate[None])
                    window[ks:ke].sub_(block.reshape(ke - ks, -1))
                    del block
            else:
                window.add_(C.T @ self._layer_rows64(layer, role, gate64))
        return result

    def pair_weighted_square_diagonal(
        self,
        role: V1SampledDualChartRowRole,
        pair_weights: Any,
        *,
        gate: Any,
    ) -> Any:
        """``Σ_{a,b} P[a, b] row(a) ∘ row(b)`` per route, fp64 ``[N]``.

        Used for the exact diagonal of a stacked covariance without
        materializing the stacked rows (``P = A Aᵀ`` for a combination matrix
        ``A``).  ``P`` must be ``[B, B]``.
        """

        self._assert_live()
        torch_module = _torch()
        P = torch_module.as_tensor(pair_weights, dtype=torch_module.float64, device=self._device)
        if tuple(int(v) for v in P.shape) != (self.context_count, self.context_count) or not bool(
            torch_module.isfinite(P).all().item()
        ):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "pair weights must be finite [B, B]")
        gate64 = self._gate64(gate)
        result = torch_module.empty(self.route_count, dtype=torch_module.float64, device=self._device)
        for layer in self._layers:
            if layer.reducer_kind == "linear_weight":
                cot = layer.row_cotangent(role).to(dtype=torch_module.float64)
                source = layer.input.to(dtype=torch_module.float64)
                weight_gate = self._linear_weight_gate(layer, gate64)
                block = torch_module.einsum("ab,ao,bo,ai,bi->oi", P, cot, cot, source, source) * (weight_gate * weight_gate)
                result[layer.route_offset : layer.route_stop] = block.reshape(-1)
            else:
                rows = self._layer_rows64(layer, role, gate64)
                result[layer.route_offset : layer.route_stop] = torch_module.einsum("ab,ar,br->r", P, rows, rows)
        return result

    def price_matrix(
        self,
        role: V1SampledDualChartRowRole,
        layer_id: str,
        weights: Any,
        *,
        gate: Any,
    ) -> Any:
        """Weighted first moment of one layer reshaped to its route block shape."""

        self._assert_live()
        for layer in self._layers:
            if layer.layer_id == layer_id:
                vector = self.moments(role, 1, weights, gate=gate)
                return vector[layer.route_offset : layer.route_stop].reshape(tuple(layer.route_block.shape))
        _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, f"packet has no layer {layer_id!r}")

    # -- exact own-kink set damage ----------------------------------------------

    def own_kink_set_damage(self, removal: Any, *, gate: Any) -> V1FactoredOwnKinkSetDamage:
        """``Δ_S(x) = Σ_i G_i (relu(a_i - c_{S,i}) - relu(a_i))`` per context.

        ``removal`` is a boolean route vector (the set ``S``); routes outside
        the gate are never removed twice.  fp64 from the fp32 factors; one
        masked matvec (Linear) or one masked convolution (conv) per layer.
        """

        self._assert_live()
        torch_module = _torch()
        gate64 = self._gate64(gate)
        removal_t = torch_module.as_tensor(np.asarray(removal, dtype=bool).reshape(-1), dtype=torch_module.bool, device=self._device)
        if int(removal_t.numel()) != self.route_count:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "removal set differs from the packet route axis")
        removal64 = removal_t.to(dtype=torch_module.float64) * gate64
        contexts = self.context_count
        total = torch_module.zeros(contexts, dtype=torch_module.float64, device=self._device)
        first = torch_module.zeros_like(total)
        flips = torch_module.zeros(contexts, dtype=torch_module.int64, device=self._device)
        best_dose = torch_module.full((contexts,), math.inf, dtype=torch_module.float64, device=self._device)
        layer_ids = []
        for layer in self._layers:
            mask = removal64[layer.route_offset : layer.route_stop].reshape(tuple(layer.route_block.shape))
            if not bool(torch_module.any(mask > 0).item()):
                continue
            layer_ids.append(layer.layer_id)
            weight = layer.weight.to(dtype=torch_module.float64) * mask
            source = layer.input.to(dtype=torch_module.float64)
            if layer.reducer_kind == "linear_weight":
                drive = source @ weight.T
                if layer.scale != 1.0:
                    drive = drive * layer.scale
            else:
                parameters = layer.reducer_parameters
                drive = torch_module.nn.functional.conv2d(
                    source,
                    weight,
                    stride=parameters["stride"],
                    padding=parameters["padding"],
                    dilation=parameters["dilation"],
                    groups=int(parameters["groups"]),
                )
                if layer.scale != 1.0:
                    drive = drive * layer.scale
            a = layer.preactivation.to(dtype=torch_module.float64)
            G = layer.post_activation_cotangent.to(dtype=torch_module.float64)
            if layer.activation_kind == "relu":
                delta = torch_module.relu(a - drive) - torch_module.relu(a)
                active = a > 0
                linear = -torch_module.where(active, drive, torch_module.zeros_like(drive))
                flipped = active != ((a - drive) > 0)
                toward_wall = torch_module.clamp(drive * torch_module.sign(a), min=0.0)
                dose = torch_module.where(
                    toward_wall > 0,
                    a.abs() / torch_module.where(toward_wall > 0, toward_wall, torch_module.ones_like(toward_wall)),
                    torch_module.full_like(a, math.inf),
                )
            else:
                delta = -drive
                linear = -drive
                flipped = torch_module.zeros_like(a, dtype=torch_module.bool)
                dose = torch_module.full_like(a, math.inf)
            reduce_dims = tuple(range(1, a.ndim))
            total += (G * delta).sum(dim=reduce_dims)
            first += (G * linear).sum(dim=reduce_dims)
            flips += flipped.sum(dim=reduce_dims)
            best_dose = torch_module.minimum(best_dose, dose.amin(dim=reduce_dims))
        doses = tuple(None if not math.isfinite(v) else float(v) for v in best_dose.tolist())
        return V1FactoredOwnKinkSetDamage(
            context_count=contexts,
            layer_ids=tuple(layer_ids),
            total_change=total.cpu().numpy(),
            first_order_change=first.cpu().numpy(),
            kink_remainder=(total - first).cpu().numpy(),
            flipped_gate_count=flips.cpu().numpy(),
            in_layer_wall_dose=doses,
            removed_route_count=int(torch_module.count_nonzero(removal64 > 0).item()),
        )


def capture_v1_factored_rows_from_conserving_route_blocks(
    response_matrix: Any,
    query_batch: QueryCotangentBatch,
    captured_blocks: Sequence[CapturedRouteBlock],
    activation_captures: Sequence[V1ConservingActivationCapture],
    *,
    context_ids: Sequence[str],
    physical_route_axis_hash: str,
    route_id_axis_hash: str,
    chart_identity_hash: str,
    current_mask_hash: str,
    proposal_generation: int,
    corrected_a_stream_identity_hash: str,
    literal_j_stream_identity_hash: str,
    backend: str = "auto",
    retain_graph: bool = False,
    parent_packet_plan_hash: str | None = None,
    parent_look_boundary_hash: str | None = None,
    layer_weight_identity_hashes: Mapping[str, str] | None = None,
) -> V1FactoredRowPacket:
    """Run the legacy VJP once and retain factors instead of rows.

    The VJP is the same ``exact_batched_vjp`` call the legacy producer makes
    (same ``G``); the cotangents are derived later by the same helper functions;
    the fp32 reducer is replayed only at materialization.  Exactly one query is
    admitted (the D lane's raw target logit).
    """

    blocks = tuple(captured_blocks)
    captures = tuple(activation_captures)
    if not blocks or len(blocks) != len(captures):
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "every captured route block requires one activation capture")
    if len({capture.layer_id for capture in captures}) != len(captures):
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "activation capture layer ids must be unique")
    if type(query_batch) is not QueryCotangentBatch or len(query_batch.query_ids) != 1:
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "factored capture admits exactly one query (the raw target logit)")
    for block, capture in zip(blocks, captures, strict=True):
        if type(block) is not CapturedRouteBlock or type(capture) is not V1ConservingActivationCapture:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "factored capture requires exact captured blocks and activation captures")
        if block.reducer_kind not in _SUPPORTED_REDUCER_KINDS:
            _refuse(
                V1FactoredRowRefusalCode.LAYER_KIND_UNSUPPORTED,
                f"route block {block.route_block.layer_id!r} has reducer kind {block.reducer_kind!r}",
            )
        capture.validate_for(block)
    contexts = tuple(_identity(value, name="context id") for value in context_ids)
    if len(contexts) != int(response_matrix.shape[0]):
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "context ids differ from the response batch")
    gradients, telemetry = exact_batched_vjp(
        response_matrix,
        [capture.post_activation_output for capture in captures],
        query_batch,
        retain_graph=retain_graph,
        backend=backend,
    )
    weight_hashes = {} if layer_weight_identity_hashes is None else dict(layer_weight_identity_hashes)
    layers: list[V1FactoredLayerFactors] = []
    offset = 0
    for block, capture, gradient in zip(blocks, captures, gradients, strict=True):
        if gradient is None:
            _refuse(
                V1FactoredRowRefusalCode.MISSING_INPUT,
                f"query response is disconnected from post-activation layer {capture.layer_id!r}",
            )
        if int(block.route_block.offset) != offset:
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "captured blocks are not contiguous on the route axis")
        if int(gradient.shape[0]) != 1:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "single-query VJP returned several cotangents")
        layers.append(
            V1FactoredLayerFactors(
                layer_id=capture.layer_id,
                reducer_kind=block.reducer_kind,
                route_block=block.route_block,
                route_offset=offset,
                activation_kind=capture.activation_kind,
                scale=float(block.scale),
                abs_aggregation_mode=str(block.abs_aggregation_mode),
                reducer_parameters=_reducer_parameters_for_kernel(dict(block.reducer_parameters)),
                # The native data plane releases/reuses its input ring after
                # this batch's consumer returns. Packet lifetime spans later
                # looks, so detach alone does not establish input custody.
                # Preserve the input layout and bytes before releasing that
                # lease; factor_bytes already includes all three per-context
                # factors. Chart parameters remain shared and version-guarded.
                input=block.input_source.detach().clone(),
                post_activation_cotangent=gradient[0].detach().clone(),
                preactivation=block.output.detach().clone(),
                bias_baseline=capture.bias_baseline.detach(),
                weight=block.weight.detach(),
                weight_identity_hash=weight_hashes.get(capture.layer_id),
            )
        )
        offset += int(block.route_block.route_count)
    descriptors = tuple(layer.descriptor() for layer in layers)
    receipt = V1FactoredRowPacketReceipt(
        physical_route_axis_hash=physical_route_axis_hash,
        route_id_axis_hash=route_id_axis_hash,
        route_count=offset,
        context_ids=contexts,
        chart_identity_hash=chart_identity_hash,
        current_mask_hash=current_mask_hash,
        proposal_generation=proposal_generation,
        corrected_a_stream_identity_hash=corrected_a_stream_identity_hash,
        literal_j_stream_identity_hash=literal_j_stream_identity_hash,
        layer_descriptors=descriptors,
        layer_descriptor_hashes=tuple(content_hash(item, prefix=_LAYER_PREFIX) for item in descriptors),
        query_id=str(query_batch.query_ids[0]),
        vjp_backend_used=str(telemetry.backend_used),
        factor_bytes=sum(layer.factor_bytes for layer in layers),
        chart_parameter_bytes=sum(layer.chart_parameter_bytes for layer in layers),
        avoided_dense_row_bytes=2 * len(contexts) * offset * _F64_BYTES,
        parent_packet_plan_hash=parent_packet_plan_hash,
        parent_look_boundary_hash=parent_look_boundary_hash,
    )
    return V1FactoredRowPacket(receipt=receipt, layers=layers)


def forecast_v1_factored_row_custody_bytes(
    *,
    layer_shapes: Sequence[tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    context_count: int,
    route_count: int,
    element_bytes: int = 4,
) -> dict[str, int]:
    """Closed-form custody forecast: factors vs dense rows.

    ``layer_shapes`` items are ``(layer_id, input_shape_per_context,
    cotangent_shape_per_context, preactivation_shape_per_context)``.
    """

    contexts = _positive_int(context_count, name="context count")
    routes = _positive_int(route_count, name="route count")
    per_context = 0
    for _layer_id, input_shape, cot_shape, pre_shape in layer_shapes:
        per_context += int(np.prod(input_shape, dtype=np.int64)) + int(np.prod(cot_shape, dtype=np.int64)) + int(
            np.prod(pre_shape, dtype=np.int64)
        )
    factor_bytes = per_context * int(element_bytes) * contexts
    dense_bytes = 2 * contexts * routes * _F64_BYTES
    return {
        "numbers_per_context": per_context,
        "factor_bytes": factor_bytes,
        "dense_row_bytes": dense_bytes,
        "context_count": contexts,
        "route_count": routes,
    }


__all__ = [
    "FACTORED_ROW_CAPTURE_AUTHORITY_SCOPE",
    "FACTORED_ROW_CAPTURE_FAMILY_SCHEMA",
    "FACTORED_ROW_EQUALITY_LAW_ID",
    "FACTORED_ROW_KERNEL_QUERY_ORDER",
    "FACTORED_ROW_LAYER_FACTOR_DESCRIPTOR_SCHEMA",
    "FACTORED_ROW_OWN_KINK_SET_DAMAGE_SCHEMA",
    "FACTORED_ROW_SYMMETRY_COVARIANCE_LAW_ID",
    "V1FactoredLayerFactors",
    "V1FactoredOwnKinkSetDamage",
    "V1FactoredRowPacket",
    "V1FactoredRowPacketReceipt",
    "V1FactoredRowRefusal",
    "V1FactoredRowRefusalCode",
    "capture_v1_factored_rows_from_conserving_route_blocks",
    "forecast_v1_factored_row_custody_bytes",
]


# =============================================================================
# Factored device-resident D session (sibling of the legacy session; V2 family)
# =============================================================================

FACTORED_DEVICE_RESIDENT_D_SESSION_DESIGN_SCHEMA: Final = (
    "partition_free_v1_factored_device_resident_D_session_design_v2"
)
FACTORED_DEVICE_RESIDENT_D_CUSTODY_FORECAST_SCHEMA: Final = (
    "partition_free_v1_factored_device_resident_D_custody_forecast_v2"
)
FACTORED_DEVICE_RESIDENT_D_LOOK_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_factored_device_resident_D_look_receipt_v2"
)
FACTORED_DEVICE_RESIDENT_D_REPLAY_COORDINATE_SCHEMA: Final = (
    "partition_free_v1_factored_device_resident_D_replay_coordinate_v2"
)
FACTORED_DEVICE_RESIDENT_D_FINALIZATION_SCHEMA: Final = (
    "partition_free_v1_factored_device_resident_D_finalization_v2"
)
FACTORED_DEVICE_RESIDENT_D_KKT_GEOMETRY_SCHEMA: Final = (
    "partition_free_v1_factored_device_resident_D_kkt_geometry_v2"
)
FACTORED_DEVICE_RESIDENT_D_LOOK_COMMIT_SCHEMA: Final = (
    "partition_free_v1_factored_device_resident_D_look_commit_v2"
)
FACTORED_DEVICE_RESIDENT_D_LOOK_REGISTRATION_SCHEMA: Final = (
    "partition_free_v1_factored_device_resident_D_look_registration_v2"
)
FACTORED_DEVICE_RESIDENT_D_REPLAY_BOUNDARY_SCHEMA: Final = (
    "partition_free_v1_alexnet_sampled_factored_device_resident_D_replay_boundary_v2"
)
FACTORED_DEVICE_RESIDENT_D_PACKET_STREAM_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_factored_device_resident_D_packet_stream_receipt_v2"
)
FACTORED_DEVICE_RESIDENT_D_AUTHORITY_SCOPE: Final = (
    "proposal_input_custody_only_no_D_stability_C_K_O_mask_commit_terminal_or_"
    "pareto_authority_v1"
)
FACTORED_DEVICE_RESIDENT_D_MOMENT_LAW_ID: Final = (
    "per_packet_transient_materialized_A_rows_legacy_welford_merge_bitwise_no_custody_v1"
)
_SESSION_PREFIX: Final = "partition-free-v1-factored-device-resident-D-"
_REPLAY_BOUNDARY_PREFIX: Final = (
    "partition-free-v1-alexnet-factored-device-resident-D-replay-boundary-"
)
_AUTHORITY_FLAGS: Final = (
    "D_stability_authority",
    "C_authority",
    "calibration_authority",
    "oracle_authority",
    "mask_commit_authority",
    "terminal_authority",
    "pareto_authority",
)


class V1FactoredDeviceResidentDSessionMode(str, Enum):
    CUDA_PRODUCTION = "cuda_production"
    CPU_REFERENCE_TEST_ONLY = "cpu_reference_test_only"


def _axis_hash(values: Sequence[str], *, role: str) -> str:
    return content_hash(tuple(values), prefix=_SESSION_PREFIX + role + "-axis-")


@dataclass(frozen=True)
class V1FactoredDeviceResidentDSessionDesign:
    """Frozen identity of one factored device-resident D session (V2 family)."""

    session_id: str
    physical_route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    chart_identity_hash: str
    current_mask_hash: str
    proposal_generation: int
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    stratum_names: tuple[str, ...]
    stratum_weights: tuple[float, ...]
    population_stratum_counts: tuple[int, ...]
    sealed_population_order_hashes: tuple[str, ...]
    maximum_contexts_per_look: int
    maximum_registered_contexts: int
    factor_numbers_per_context: int
    device_total_memory_bytes: int
    device_free_memory_bytes: int
    current_reserved_device_bytes: int = 0
    provider_peak_workspace_bytes: int = 0
    hbm_ceiling_fraction: float = 0.85
    execution_mode: V1FactoredDeviceResidentDSessionMode = (
        V1FactoredDeviceResidentDSessionMode.CUDA_PRODUCTION
    )
    factor_dtype_identity: str = "torch.float32"
    moment_law_id: str = FACTORED_DEVICE_RESIDENT_D_MOMENT_LAW_ID
    schema: str = FACTORED_DEVICE_RESIDENT_D_SESSION_DESIGN_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "session_id",
            "physical_route_axis_hash",
            "route_id_axis_hash",
            "chart_identity_hash",
            "current_mask_hash",
            "corrected_a_stream_identity_hash",
            "literal_j_stream_identity_hash",
        ):
            _identity(getattr(self, name), name=name)
        if self.physical_route_axis_hash == self.route_id_axis_hash:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "physical and ordered route axes must remain distinct")
        if self.corrected_a_stream_identity_hash == self.literal_j_stream_identity_hash:
            _refuse(V1FactoredRowRefusalCode.ROLE_MISMATCH, "A and J streams must remain distinct")
        names = tuple(_identity(v, name="stratum name") for v in self.stratum_names)
        if len(names) != 2 or len(set(names)) != 2:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "the factored session supports the canonical two-stratum split")
        weights = tuple(float(v) for v in self.stratum_weights)
        if len(weights) != 2 or any(not math.isfinite(w) or w < 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-12:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "stratum weights must be two nonnegative numbers summing to one")
        populations = tuple(_positive_int(v, name="population stratum count") for v in self.population_stratum_counts)
        orders = tuple(_identity(v, name="sealed population order hash") for v in self.sealed_population_order_hashes)
        if len(populations) != 2 or len(orders) != 2:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "population axes must align with the two strata")
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "population_stratum_counts", populations)
        object.__setattr__(self, "sealed_population_order_hashes", orders)
        for name in ("route_count", "maximum_contexts_per_look", "maximum_registered_contexts", "factor_numbers_per_context"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name=name))
        object.__setattr__(self, "proposal_generation", _nonnegative_int(self.proposal_generation, name="proposal generation"))
        for name in ("device_total_memory_bytes", "device_free_memory_bytes", "current_reserved_device_bytes", "provider_peak_workspace_bytes"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name=name))
        if self.device_total_memory_bytes <= 0:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "device total memory must be positive")
        fraction = float(self.hbm_ceiling_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 0.85:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "canonical HBM ceiling lies in (0, 0.85]")
        object.__setattr__(self, "hbm_ceiling_fraction", fraction)
        if type(self.execution_mode) is not V1FactoredDeviceResidentDSessionMode:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "execution mode must be exact")
        if (
            self.schema != FACTORED_DEVICE_RESIDENT_D_SESSION_DESIGN_SCHEMA
            or self.factor_dtype_identity != "torch.float32"
            or self.moment_law_id != FACTORED_DEVICE_RESIDENT_D_MOMENT_LAW_ID
        ):
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored session design schema drifted")

    @property
    def design_hash(self) -> str:
        return content_hash(self, prefix=_SESSION_PREFIX + "design-")


@dataclass(frozen=True)
class V1FactoredDeviceResidentDCustodyForecast:
    """Closed-form device byte ledger for the factored session."""

    design_hash: str
    route_count: int
    stratum_count: int
    maximum_contexts_per_look: int
    maximum_registered_contexts: int
    factor_numbers_per_context: int
    moment_accumulator_bytes: int
    factor_custody_bytes_at_endpoint: int
    transient_packet_row_bytes: int
    gram_workspace_bytes: int
    dense_factor_bytes_at_full_rank: int
    required_bytes: int
    ceiling_bytes: int
    admitted: bool
    dense_factor_admitted_within_ceiling: bool
    schema: str = FACTORED_DEVICE_RESIDENT_D_CUSTODY_FORECAST_SCHEMA

    @property
    def forecast_hash(self) -> str:
        return content_hash(self, prefix=_SESSION_PREFIX + "forecast-")


def forecast_v1_factored_device_resident_d_session(
    design: V1FactoredDeviceResidentDSessionDesign,
) -> V1FactoredDeviceResidentDCustodyForecast:
    """Forecast the factored session's device bytes from registered numerals only."""

    if type(design) is not V1FactoredDeviceResidentDSessionDesign:
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "forecast requires the exact factored session design")
    N = design.route_count
    S = len(design.stratum_names)
    n = design.maximum_registered_contexts
    B = design.maximum_contexts_per_look
    moments = 4 * S * N * _F64_BYTES
    factors = n * design.factor_numbers_per_context * 4
    transient = B * N * _F64_BYTES
    gram = (n + S) * (n + S) * _F64_BYTES + n * design.factor_numbers_per_context * _F64_BYTES
    dense = (1 + n + S) * N * _F64_BYTES
    required = moments + factors + transient + gram
    ceiling = int(math.floor(design.hbm_ceiling_fraction * (design.device_total_memory_bytes - design.current_reserved_device_bytes))) - design.provider_peak_workspace_bytes
    return V1FactoredDeviceResidentDCustodyForecast(
        design_hash=design.design_hash,
        route_count=N,
        stratum_count=S,
        maximum_contexts_per_look=B,
        maximum_registered_contexts=n,
        factor_numbers_per_context=design.factor_numbers_per_context,
        moment_accumulator_bytes=moments,
        factor_custody_bytes_at_endpoint=factors,
        transient_packet_row_bytes=transient,
        gram_workspace_bytes=gram,
        dense_factor_bytes_at_full_rank=dense,
        required_bytes=required,
        ceiling_bytes=max(ceiling, 0),
        admitted=required <= ceiling,
        dense_factor_admitted_within_ceiling=required + dense <= ceiling,
    )


@dataclass(frozen=True)
class V1FactoredDeviceResidentDLookReceipt:
    session_design_hash: str
    forecast_hash: str
    registered_look_index: int
    packet_ordinal: int
    stratum_name: str
    context_axis_hash: str
    draw_axis_hash: str
    source_measurement_axis_hash: str
    packet_receipt_hash: str
    stratum_counts_before: tuple[int, ...]
    stratum_counts_after: tuple[int, ...]
    factor_bytes: int
    avoided_dense_row_bytes: int
    moment_state_scalar_hash: str
    route_scale_d2h_bytes: int = 0
    schema: str = FACTORED_DEVICE_RESIDENT_D_LOOK_RECEIPT_SCHEMA

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_SESSION_PREFIX + "look-")

    def assert_integrity(self) -> str:
        if self.route_scale_d2h_bytes != 0 or self.schema != FACTORED_DEVICE_RESIDENT_D_LOOK_RECEIPT_SCHEMA:
            _refuse(V1FactoredRowRefusalCode.AUTHORITY_VIOLATION, "factored look receipt drifted")
        return self.receipt_hash


@dataclass(frozen=True)
class V1FactoredDeviceResidentDReplayCoordinateReceipt:
    session_design_hash: str
    forecast_hash: str
    sealed_model_context_ledger_hash: str
    route_id_axis_hash: str
    consumed_look_receipt_hashes: tuple[str, ...]
    packet_receipt_hashes: tuple[str, ...]
    next_registered_look_index: int
    schema: str = FACTORED_DEVICE_RESIDENT_D_REPLAY_COORDINATE_SCHEMA

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_SESSION_PREFIX + "replay-coordinate-")


@dataclass(frozen=True)
class V1FactoredDeviceResidentDFinalizationReceipt:
    session_design_hash: str
    forecast_hash: str
    final_logical_look_index: int
    stratum_counts: tuple[int, ...]
    context_axis_hashes_by_stratum: tuple[str, ...]
    draw_axis_hashes_by_stratum: tuple[str, ...]
    look_receipt_axis_hash: str
    packet_receipt_axis_hash: str
    factor_bytes_total: int
    avoided_dense_row_bytes_total: int
    singleton_work_l2_norm: float
    row_free_finalized_once: bool = True
    route_scale_d2h_bytes: int = 0
    D_stability_authority: bool = False
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = FACTORED_DEVICE_RESIDENT_D_AUTHORITY_SCOPE
    schema: str = FACTORED_DEVICE_RESIDENT_D_FINALIZATION_SCHEMA

    def __post_init__(self) -> None:
        _closed_flags(self, _AUTHORITY_FLAGS)
        if self.row_free_finalized_once is not True or self.route_scale_d2h_bytes != 0:
            _refuse(V1FactoredRowRefusalCode.AUTHORITY_VIOLATION, "factored finalization is row-free and once")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_SESSION_PREFIX + "finalization-")


@dataclass(frozen=True)
class V1FactoredDeviceResidentDKKTGeometryReceipt:
    session_design_hash: str
    finalization_receipt_hash: str
    exact_geometry_receipt_hash: str
    dense_adapter_hash: str | None
    dense_factor_admitted: bool
    dense_factor_budget_bytes: int
    retained_rank: int
    singleton_work_shape: tuple[int, ...]
    singleton_work_l2_norm: float
    residual_spectral_bound: float
    fd_sketch_used: bool = False
    rank_menu_registered: bool = False
    kkt_proposal_input_authority: bool = True
    native_fiber_signed_f_authority: bool = False
    D_stability_authority: bool = False
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = FACTORED_DEVICE_RESIDENT_D_AUTHORITY_SCOPE
    schema: str = FACTORED_DEVICE_RESIDENT_D_KKT_GEOMETRY_SCHEMA

    def __post_init__(self) -> None:
        _closed_flags(self, _AUTHORITY_FLAGS)
        if self.fd_sketch_used is not False or self.rank_menu_registered is not False or self.native_fiber_signed_f_authority is not False:
            _refuse(V1FactoredRowRefusalCode.AUTHORITY_VIOLATION, "factored KKT geometry uses no sketch and claims no signed-F authority")
        if (self.dense_adapter_hash is None) != (self.dense_factor_admitted is False):
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "dense adapter hash and admission flag disagree")

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_SESSION_PREFIX + "kkt-geometry-")


@dataclass
class _ActiveFactoredPacket:
    registered_look_index: int
    packet_ordinal: int
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    stratum_name: str
    stratum_index: int
    source_measurement_hashes: tuple[str, ...]
    sealed_order_indices: tuple[int, ...]
    stratum_counts_before: tuple[int, ...]
    packet: V1FactoredRowPacket | None = None


class V1FactoredDeviceResidentDSession:
    """One nonserializable live factored-D owner spanning registered look packets.

    Sibling of ``V1SampledDeviceResidentDSession``: same look protocol and
    refusals, but it consumes factored packets, retains factors instead of
    rows, keeps the per-stratum corrected moments by the bitwise legacy
    Welford merge on transiently materialized ``A`` rows (no custody), appends
    no FD rows, and freezes the exact Gram geometry at the endpoint.
    """

    def __init__(self, design: V1FactoredDeviceResidentDSessionDesign, *, device: Any | None = None, capture_admission: Any = None) -> None:
        torch_module = _torch()
        if type(design) is not V1FactoredDeviceResidentDSessionDesign:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "factored session requires its exact design")
        runtime_device = torch_module.device("cpu" if device is None else device)
        if design.execution_mode is V1FactoredDeviceResidentDSessionMode.CUDA_PRODUCTION:
            if runtime_device.type != "cuda":
                _refuse(V1FactoredRowRefusalCode.CPU_PRODUCTION_FALLBACK, "production factored session refuses CPU fallback")
            if not torch_module.cuda.is_available():
                _refuse(V1FactoredRowRefusalCode.DEVICE_MISMATCH, "production factored session requires available CUDA")
        elif runtime_device.type != "cpu":
            _refuse(V1FactoredRowRefusalCode.DEVICE_MISMATCH, "CPU reference factored session must run on CPU")
        from .modern_v1_alexnet_capture_admission import V1AlexNetFactoredCaptureAdmission
        if (design.execution_mode is V1FactoredDeviceResidentDSessionMode.CUDA_PRODUCTION
                or capture_admission is not None):
            if type(capture_admission) is not V1AlexNetFactoredCaptureAdmission:
                _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "CUDA factored D requires exact actual-capture admission")
            capture_admission.assert_live()
        forecast = forecast_v1_factored_device_resident_d_session(design)
        if not forecast.admitted:
            _refuse(
                V1FactoredRowRefusalCode.DENSE_FACTOR_NOT_ADMITTED,
                f"factored custody forecast {forecast.required_bytes} B exceeds the ceiling {forecast.ceiling_bytes} B",
            )
        self.design = design
        self.forecast = forecast
        self.device = runtime_device
        self._capture_admission = capture_admission
        self._capture_origin = (capture_admission, design.design_hash, str(runtime_device))
        self._capture_session_token = (None if capture_admission is None
                                       else capture_admission.bind_session(self))
        try:
            S = len(design.stratum_names)
            N = design.route_count
            self._signed_mean = torch_module.zeros((S, N), dtype=torch_module.float64, device=runtime_device)
            self._signed_m2 = torch_module.zeros_like(self._signed_mean)
            self._absolute_mean = torch_module.zeros_like(self._signed_mean)
            self._absolute_m2 = torch_module.zeros_like(self._signed_mean)
            self._stratum_counts = [0] * S
            self._context_ledgers: list[list[str]] = [[] for _ in range(S)]
            self._draw_ledgers: list[list[str]] = [[] for _ in range(S)]
            self._source_measurements: list[list[str]] = [[] for _ in range(S)]
            self._packets: list[tuple[int, V1FactoredRowPacket]] = []
            self._look_receipts: list[V1FactoredDeviceResidentDLookReceipt] = []
            self._active: _ActiveFactoredPacket | None = None
            self._evicted = False
            self._finalized: V1FactoredDeviceResidentDFinalizationReceipt | None = None
            self._kkt_frozen = False
            self._moment_custody = self._moment_tensor_custody()
        except BaseException:
            if capture_admission is not None:
                capture_admission.release_session(self)
            raise

    def _moment_tensor_custody(self) -> tuple[Any, ...]:
        result = []
        for name in ("_signed_mean", "_signed_m2", "_absolute_mean", "_absolute_m2"):
            value = getattr(self, name)
            if not _torch().is_tensor(value):
                _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored moment bank lost tensor custody")
            result.append((id(value), str(value.device), value.dtype, tuple(value.shape),
                           tuple(value.stride()), int(value.storage_offset()),
                           int(value.data_ptr()), int(value.untyped_storage().data_ptr()), int(value._version)))
        return tuple(result)

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("factored device-resident D sessions are nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("factored device-resident D sessions cannot be copied")

    def __deepcopy__(self, _memo: dict[int, object]) -> NoReturn:
        raise TypeError("factored device-resident D sessions cannot be copied")

    def _assert_live(self) -> None:
        if self._evicted:
            _refuse(V1FactoredRowRefusalCode.EVICTED, "factored device-resident D session was evicted")
        if (self._capture_admission is not self._capture_origin[0]
                or self.design.design_hash != self._capture_origin[1]
                or str(self.device) != self._capture_origin[2]):
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored capture source/design changed")
        if self._capture_admission is not None:
            self._capture_admission.assert_session(self)
        if self._moment_tensor_custody() != self._moment_custody:
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY,
                    "factored moment banks changed outside their registered merge")

    @property
    def stratum_counts(self) -> tuple[int, ...]:
        return tuple(self._stratum_counts)

    @property
    def look_receipts(self) -> tuple[V1FactoredDeviceResidentDLookReceipt, ...]:
        return tuple(self._look_receipts)

    @property
    def packets(self) -> tuple[tuple[int, V1FactoredRowPacket], ...]:
        self._assert_live()
        return tuple(self._packets)

    @property
    def is_finalized(self) -> bool:
        return self._finalized is not None

    def begin_registered_look_packet(
        self,
        *,
        registered_look_index: int,
        packet_ordinal: int,
        context_ids: Sequence[str],
        draw_ids: Sequence[str],
        stratum_labels: Sequence[str],
        sealed_order_indices: Sequence[int],
        source_measurement_hashes: Sequence[str],
    ) -> None:
        self._assert_live()
        if self._finalized is not None:
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "finalized factored sessions cannot acquire another look")
        if self._active is not None:
            _refuse(V1FactoredRowRefusalCode.ACTIVE_PACKET, "a factored D look packet is already active")
        look = _positive_int(registered_look_index, name="registered look index")
        packet = _nonnegative_int(packet_ordinal, name="packet ordinal")
        contexts = tuple(_identity(v, name="context id") for v in context_ids)
        draws = tuple(_identity(v, name="draw id") for v in draw_ids)
        labels = tuple(_identity(v, name="stratum label") for v in stratum_labels)
        sources = tuple(_identity(v, name="source measurement hash") for v in source_measurement_hashes)
        orders = tuple(_nonnegative_int(v, name="sealed order index") for v in sealed_order_indices)
        if not contexts or not (len(contexts) == len(draws) == len(labels) == len(sources) == len(orders)):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "look ledgers must align and be nonempty")
        if len(set(contexts)) != len(contexts) or len(set(draws)) != len(draws):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "look contexts and draws must be unique")
        if len(contexts) > self.design.maximum_contexts_per_look:
            _refuse(V1FactoredRowRefusalCode.DENSE_FACTOR_NOT_ADMITTED, "look exceeds its pre-outcome packet forecast")
        if len(set(labels)) != 1 or labels[0] not in self.design.stratum_names:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "one same-stratum packet at a time, naming a registered stratum")
        stratum = self.design.stratum_names.index(labels[0])
        if set(contexts).intersection(c for ledger in self._context_ledgers for c in ledger):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "look packet repeats an already consumed context")
        if set(draws).intersection(d for ledger in self._draw_ledgers for d in ledger):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "look packet repeats an already consumed draw")
        start = self._stratum_counts[stratum]
        if tuple(range(start, start + len(contexts))) != orders:
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "look packet is not the next sealed stratum prefix")
        if start + len(contexts) > self.design.population_stratum_counts[stratum]:
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "look packet exceeds its finite-population stratum")
        self._active = _ActiveFactoredPacket(
            registered_look_index=look,
            packet_ordinal=packet,
            context_ids=contexts,
            draw_ids=draws,
            stratum_name=labels[0],
            stratum_index=stratum,
            source_measurement_hashes=sources,
            sealed_order_indices=orders,
            stratum_counts_before=tuple(self._stratum_counts),
        )

    def consume_factored_packet(self, packet: V1FactoredRowPacket, *, gate: Any) -> None:
        """Retain one factored packet and merge its ``A`` moments bitwise-legacy."""

        self._assert_live()
        torch_module = _torch()
        active = self._active
        if active is None:
            _refuse(V1FactoredRowRefusalCode.NO_ACTIVE_PACKET, "no active factored look packet")
        if active.packet is not None:
            _refuse(V1FactoredRowRefusalCode.ACTIVE_PACKET, "the active look packet already holds its factors")
        if type(packet) is not V1FactoredRowPacket:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "factored session consumes factored row packets")
        receipt = packet.receipt
        design = self.design
        if (
            receipt.context_ids != active.context_ids
            or receipt.physical_route_axis_hash != design.physical_route_axis_hash
            or receipt.route_id_axis_hash != design.route_id_axis_hash
            or receipt.route_count != design.route_count
            or receipt.chart_identity_hash != design.chart_identity_hash
            or receipt.current_mask_hash != design.current_mask_hash
            or receipt.proposal_generation != design.proposal_generation
            or receipt.corrected_a_stream_identity_hash != design.corrected_a_stream_identity_hash
            or receipt.literal_j_stream_identity_hash != design.literal_j_stream_identity_hash
        ):
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored packet differs from the session design or active look")
        if design.execution_mode is V1FactoredDeviceResidentDSessionMode.CUDA_PRODUCTION and packet.device.type != "cuda":
            _refuse(V1FactoredRowRefusalCode.CPU_PRODUCTION_FALLBACK, "production factored packets must be device-resident")
        if packet.device != self.device:
            _refuse(V1FactoredRowRefusalCode.DEVICE_MISMATCH, "factored packet left the session device")
        admission = self._capture_admission
        with (nullcontext() if admission is None else admission.consuming(self, packet, gate)):
            # bitwise-legacy Welford merge on transiently materialized corrected-A rows
            corrected = packet.materialize_rows(
                V1SampledDualChartRowRole.CORRECTED_AUMANN_SHAPLEY, 0, design.route_count, gate=gate
            )
            stratum = active.stratum_index
            count0 = self._stratum_counts[stratum]
            count1 = int(corrected.shape[0])
            total = count0 + count1
            for source, mean_bank, m2_bank in (
                (corrected, self._signed_mean, self._signed_m2),
                (torch_module.abs(corrected), self._absolute_mean, self._absolute_m2),
            ):
                packet_mean = torch_module.mean(source, dim=0)
                packet_m2 = torch_module.sum((source - packet_mean) ** 2, dim=0)
                mean0 = mean_bank[stratum]
                m20 = m2_bank[stratum]
                if count0 == 0:
                    mean0.copy_(packet_mean)
                    m20.copy_(packet_m2)
                else:
                    delta = packet_mean - mean0
                    mean0.add_(delta, alpha=count1 / total)
                    m20.add_(packet_m2 + delta * delta * (count0 * count1 / total))
            del corrected
            self._moment_custody = self._moment_tensor_custody()
            active.packet = packet

    def end_registered_look_packet(self) -> V1FactoredDeviceResidentDLookReceipt:
        self._assert_live()
        active = self._active
        if active is None:
            _refuse(V1FactoredRowRefusalCode.NO_ACTIVE_PACKET, "no active factored look packet can be closed")
        if active.packet is None:
            _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "the active look packet received no factors")
        admission = self._capture_admission
        gate = None if admission is None else admission._records.get(id(active.packet), (None, None, None, None))[3]
        with (nullcontext() if admission is None else admission.committing(self, active.packet, gate)):
            stratum = active.stratum_index
            self._stratum_counts[stratum] += len(active.context_ids)
            self._context_ledgers[stratum].extend(active.context_ids)
            self._draw_ledgers[stratum].extend(active.draw_ids)
            self._source_measurements[stratum].extend(active.source_measurement_hashes)
            self._packets.append((stratum, active.packet))
            counts_after = tuple(self._stratum_counts)
            moment_scalar_hash = content_hash(
                {
                    "session_design_hash": self.design.design_hash,
                    "stratum_counts": counts_after,
                    "look_receipts_before": tuple(r.receipt_hash for r in self._look_receipts),
                    "latest_packet_receipt_hash": active.packet.receipt.receipt_hash,
                    "moment_law_id": FACTORED_DEVICE_RESIDENT_D_MOMENT_LAW_ID,
                    "route_scale_values_not_hashed": True,
                },
                prefix=_SESSION_PREFIX + "moment-state-scalar-",
            )
            receipt = V1FactoredDeviceResidentDLookReceipt(
                session_design_hash=self.design.design_hash,
                forecast_hash=self.forecast.forecast_hash,
                registered_look_index=active.registered_look_index,
                packet_ordinal=active.packet_ordinal,
                stratum_name=active.stratum_name,
                context_axis_hash=_axis_hash(active.context_ids, role="look-context"),
                draw_axis_hash=_axis_hash(active.draw_ids, role="look-draw"),
                source_measurement_axis_hash=_axis_hash(active.source_measurement_hashes, role="look-source-measurement"),
                packet_receipt_hash=active.packet.receipt.receipt_hash,
                stratum_counts_before=active.stratum_counts_before,
                stratum_counts_after=counts_after,
                factor_bytes=active.packet.factor_bytes,
                avoided_dense_row_bytes=active.packet.receipt.avoided_dense_row_bytes,
                moment_state_scalar_hash=moment_scalar_hash,
            )
            self._look_receipts.append(receipt)
            self._active = None
            return receipt

    def replay_coordinate_receipt(self, *, sealed_model_context_ledger_hash: str) -> V1FactoredDeviceResidentDReplayCoordinateReceipt:
        self._assert_live()
        if self._active is not None:
            _refuse(V1FactoredRowRefusalCode.ACTIVE_PACKET, "cannot emit a replay coordinate with an active packet")
        return V1FactoredDeviceResidentDReplayCoordinateReceipt(
            session_design_hash=self.design.design_hash,
            forecast_hash=self.forecast.forecast_hash,
            sealed_model_context_ledger_hash=_identity(sealed_model_context_ledger_hash, name="sealed model context ledger hash"),
            route_id_axis_hash=self.design.route_id_axis_hash,
            consumed_look_receipt_hashes=tuple(r.assert_integrity() for r in self._look_receipts),
            packet_receipt_hashes=tuple(p.receipt.receipt_hash for _s, p in self._packets),
            next_registered_look_index=(1 if not self._look_receipts else self._look_receipts[-1].registered_look_index + 1),
        )

    def singleton_work(self) -> Any:
        """``Σ_s w_s absolute-mean_s`` (route-scale fp64), the KKT tie-break vector."""

        self._assert_live()
        torch_module = _torch()
        work = torch_module.zeros(self.design.route_count, dtype=torch_module.float64, device=self.device)
        for weight, absolute_mean in zip(self.design.stratum_weights, self._absolute_mean, strict=True):
            work.add_(absolute_mean, alpha=weight)
        return work.detach().contiguous()

    def finalize_row_free_once(self) -> V1FactoredDeviceResidentDFinalizationReceipt:
        self._assert_live()
        if self._capture_admission is not None:
            self._capture_admission.session_payload(self)
        if self._active is not None:
            _refuse(V1FactoredRowRefusalCode.ACTIVE_PACKET, "cannot finalize with an active packet")
        if self._finalized is not None:
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "factored session already finalized")
        if not self._look_receipts or any(c <= 0 for c in self._stratum_counts):
            _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "finalization requires at least one packet per stratum")
        receipt = V1FactoredDeviceResidentDFinalizationReceipt(
            session_design_hash=self.design.design_hash,
            forecast_hash=self.forecast.forecast_hash,
            final_logical_look_index=self._look_receipts[-1].registered_look_index,
            stratum_counts=tuple(self._stratum_counts),
            context_axis_hashes_by_stratum=tuple(_axis_hash(ledger, role="stratum-context") for ledger in self._context_ledgers),
            draw_axis_hashes_by_stratum=tuple(_axis_hash(ledger, role="stratum-draw") for ledger in self._draw_ledgers),
            look_receipt_axis_hash=_axis_hash(tuple(r.receipt_hash for r in self._look_receipts), role="look-receipt"),
            packet_receipt_axis_hash=_axis_hash(tuple(p.receipt.receipt_hash for _s, p in self._packets), role="packet-receipt"),
            factor_bytes_total=sum(p.factor_bytes for _s, p in self._packets),
            avoided_dense_row_bytes_total=sum(p.receipt.avoided_dense_row_bytes for _s, p in self._packets),
            singleton_work_l2_norm=float(_torch().linalg.vector_norm(self.singleton_work()).item()),
        )
        self._finalized = receipt
        return receipt

    def freeze_exact_kkt_geometry_once(self, *, gate: Any, dense_factor_budget_bytes: int) -> tuple[Any, Any, Any, V1FactoredDeviceResidentDKKTGeometryReceipt]:
        """Exact Gram geometry plus the dense KKT adapter when admitted.

        Returns ``(geometry, adapter_or_None, singleton_work, receipt)``.  A dense
        factor over budget is a typed absence (``adapter is None``), never a
        fallback to a sketch.
        """

        self._assert_live()
        if self._finalized is None:
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "KKT geometry requires the row-free finalization first")
        if self._kkt_frozen:
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "factored session already froze its KKT geometry")
        from .modern_v1_factored_row_geometry import (
            adapt_v1_factored_exact_geometry_to_kkt_arrays,
            compute_v1_factored_row_exact_geometry,
        )

        self._kkt_frozen = True
        packets = tuple(p for _s, p in self._packets)
        labels = tuple(self.design.stratum_names[s] for s, _p in self._packets)
        geometry = compute_v1_factored_row_exact_geometry(
            packets=packets,
            packet_stratum_labels=labels,
            stratum_names=self.design.stratum_names,
            stratum_weights=self.design.stratum_weights,
            role=V1SampledDualChartRowRole.LITERAL_RELAXED_GATE_JACOBIAN,
            gate=gate,
        )
        budget = _nonnegative_int(dense_factor_budget_bytes, name="dense factor budget bytes")
        adapter = None
        try:
            adapter = adapt_v1_factored_exact_geometry_to_kkt_arrays(geometry, budget_bytes=budget)
        except V1FactoredRowRefusal as error:
            if error.code is not V1FactoredRowRefusalCode.DENSE_FACTOR_NOT_ADMITTED:
                raise
        work = self.singleton_work()
        receipt = V1FactoredDeviceResidentDKKTGeometryReceipt(
            session_design_hash=self.design.design_hash,
            finalization_receipt_hash=self._finalized.receipt_hash,
            exact_geometry_receipt_hash=geometry.receipt.receipt_hash,
            dense_adapter_hash=None if adapter is None else adapter.adapter_hash,
            dense_factor_admitted=adapter is not None,
            dense_factor_budget_bytes=budget,
            retained_rank=geometry.retained_rank,
            singleton_work_shape=tuple(int(v) for v in work.shape),
            singleton_work_l2_norm=float(_torch().linalg.vector_norm(work).item()),
            residual_spectral_bound=geometry.receipt.residual_spectral_bound,
        )
        return geometry, adapter, work, receipt

    def evict(self) -> None:
        if self._capture_origin[0] is not None:
            self._capture_origin[0].evict_session(self)
        else:
            self._evict_payload()

    def _evict_payload(self) -> None:
        """Drop factors and moments; the capture owner holds its lock when bound."""
        self._evicted = True
        if self._active is not None and self._active.packet is not None:
            self._active.packet.evict()
        for _s, packet in self._packets:
            packet.evict()
        self._packets = []
        self._active = None
        self._signed_mean = None
        self._signed_m2 = None
        self._absolute_mean = None
        self._absolute_m2 = None


# --- provider protocol -----------------------------------------------------------------


class V1FactoredRowPacketAccumulator:
    """Provider-facing accumulator for one factored packet (``accumulate_device_factored``)."""

    def __init__(
        self,
        *,
        session: V1FactoredDeviceResidentDSession,
        registered_look_index: int,
        packet_ordinal: int,
        expected_context_ids: Sequence[str],
        expected_draw_ids: Sequence[str],
        stratum_label: str,
        sealed_order_indices: Sequence[int],
        source_measurement_hashes: Sequence[str],
        route_ids: Any,
        destination_transport_hash: str,
        gate: Any,
        provider_plan_identity_hash: str,
        source_capture_identity_hash: str,
    ) -> None:
        if type(session) is not V1FactoredDeviceResidentDSession:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "accumulator requires the exact factored session")
        self.session = session
        self.registered_look_index = _positive_int(registered_look_index, name="registered look index")
        self.packet_ordinal = _nonnegative_int(packet_ordinal, name="packet ordinal")
        self.expected_context_ids = tuple(_identity(v, name="context id") for v in expected_context_ids)
        self.expected_draw_ids = tuple(_identity(v, name="draw id") for v in expected_draw_ids)
        self.stratum_label = _identity(stratum_label, name="stratum label")
        self.sealed_order_indices = tuple(int(v) for v in sealed_order_indices)
        self.source_measurement_hashes = tuple(_identity(v, name="source measurement hash") for v in source_measurement_hashes)
        self.route_ids = route_ids
        self.destination_transport_hash = _identity(destination_transport_hash, name="destination transport hash")
        self.gate = gate
        self.provider_plan_identity_hash = _identity(provider_plan_identity_hash, name="provider plan identity hash")
        self.source_capture_identity_hash = _identity(source_capture_identity_hash, name="source capture identity hash")
        self._packet: V1FactoredRowPacket | None = None
        self._look_receipt: V1FactoredDeviceResidentDLookReceipt | None = None
        self._begun = False

    @property
    def design(self) -> V1FactoredDeviceResidentDSessionDesign:
        return self.session.design

    def packet_identities(self) -> dict[str, Any]:
        """The identity kwargs a provider passes to the factored capture."""

        d = self.session.design
        return {
            "context_ids": self.expected_context_ids,
            "physical_route_axis_hash": d.physical_route_axis_hash,
            "route_id_axis_hash": d.route_id_axis_hash,
            "chart_identity_hash": d.chart_identity_hash,
            "current_mask_hash": d.current_mask_hash,
            "proposal_generation": d.proposal_generation,
            "corrected_a_stream_identity_hash": d.corrected_a_stream_identity_hash,
            "literal_j_stream_identity_hash": d.literal_j_stream_identity_hash,
            "parent_packet_plan_hash": self.provider_plan_identity_hash,
        }

    def capture_packet(
        self,
        response_matrix: Any,
        query_batch: QueryCotangentBatch,
        captured_blocks: Sequence[CapturedRouteBlock],
        activation_captures: Sequence[V1ConservingActivationCapture],
        *,
        backend: str = "auto",
        retain_graph: bool = False,
    ) -> V1FactoredRowPacket:
        """Run the one VJP and build the packet keyed to this accumulator's plan.

        Provider-facing: the measurement layer hands its captured batch here
        instead of importing this module (owner-dependency law), so the irg
        dependency stays inside irg.  Identities come from
        :meth:`packet_identities`; the capture is the same as the direct call.
        """

        admission = self.session._capture_admission
        self.session._assert_live()
        if admission is not None:
            admission.require_capture(self)
        packet = capture_v1_factored_rows_from_conserving_route_blocks(
            response_matrix, query_batch, captured_blocks, activation_captures,
            backend=backend, retain_graph=retain_graph, **self.packet_identities(),
        )
        try:
            if admission is not None:
                admission.observe(self, packet, response_matrix=response_matrix,
                    query_batch=query_batch, captured_blocks=captured_blocks,
                    activation_captures=activation_captures, backend=backend)
            return packet
        except BaseException:
            packet.evict()
            raise

    def begin_capture_source(self, **sources) -> None:
        self.session._assert_live()
        admission = self.session._capture_admission
        if admission is not None:
            admission.begin_capture(self, **sources)

    @staticmethod
    def refuse(code: str, message: str) -> NoReturn:
        """Raise the typed factored refusal named by ``code`` (provider-facing)."""

        try:
            member = V1FactoredRowRefusalCode[str(code)]
        except KeyError:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, f"unknown factored refusal code {code!r}: {message}")
        _refuse(member, message)

    def begin_packet(
        self,
        contexts: Sequence[str],
        *,
        route_ids: Any,
        destination_transport_hash: str,
        corrected_stream_identity_hash: str,
        literal_j_stream_identity_hash: str,
    ) -> None:
        if self._begun:
            _refuse(V1FactoredRowRefusalCode.ACTIVE_PACKET, "factored packet already begun")
        if tuple(contexts) != self.expected_context_ids:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "provider contexts differ from the frozen packet plan")
        if len(route_ids) != len(self.route_ids) or str(destination_transport_hash) != self.destination_transport_hash:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "provider route axis or transport differs from the plan")
        d = self.session.design
        if (
            str(corrected_stream_identity_hash) != d.corrected_a_stream_identity_hash
            or str(literal_j_stream_identity_hash) != d.literal_j_stream_identity_hash
        ):
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "provider stream identities differ from the session")
        self.session.begin_registered_look_packet(
            registered_look_index=self.registered_look_index,
            packet_ordinal=self.packet_ordinal,
            context_ids=self.expected_context_ids,
            draw_ids=self.expected_draw_ids,
            stratum_labels=(self.stratum_label,) * len(self.expected_context_ids),
            sealed_order_indices=self.sealed_order_indices,
            source_measurement_hashes=self.source_measurement_hashes,
        )
        self._begun = True

    def consume_factored_packet(self, packet: V1FactoredRowPacket) -> None:
        if not self._begun:
            _refuse(V1FactoredRowRefusalCode.NO_ACTIVE_PACKET, "begin_packet must precede the factored packet")
        if self._packet is not None:
            _refuse(V1FactoredRowRefusalCode.ACTIVE_PACKET, "one factored packet per accumulator")
        if type(packet) is not V1FactoredRowPacket or packet.receipt.parent_packet_plan_hash != self.provider_plan_identity_hash:
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored packet is not keyed by this packet plan")
        self.session.consume_factored_packet(packet, gate=self.gate)
        self._packet = packet

    def end_packet(self) -> V1FactoredDeviceResidentDLookReceipt:
        if self._packet is None:
            _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "end_packet before any factored packet")
        if self._look_receipt is not None:
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "factored packet already closed")
        self._look_receipt = self.session.end_registered_look_packet()
        return self._look_receipt

    @property
    def packet(self) -> V1FactoredRowPacket | None:
        return self._packet

    @property
    def look_receipt(self) -> V1FactoredDeviceResidentDLookReceipt:
        if self._look_receipt is None:
            _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "factored packet has no closed look receipt")
        return self._look_receipt

    def route_streaming_receipt(self) -> Mapping[str, object]:
        packet = self._packet
        receipt = self.look_receipt
        assert packet is not None
        return {
            "schema": FACTORED_DEVICE_RESIDENT_D_PACKET_STREAM_RECEIPT_SCHEMA,
            "factored_packet_receipt_hash": packet.receipt.receipt_hash,
            "factored_look_receipt_hash": receipt.receipt_hash,
            "provider_plan_identity_hash": self.provider_plan_identity_hash,
            "source_capture_identity_hash": self.source_capture_identity_hash,
            "context_count": packet.context_count,
            "route_count": packet.route_count,
            "factor_bytes": packet.factor_bytes,
            "avoided_dense_row_bytes": packet.receipt.avoided_dense_row_bytes,
            "route_scale_d2h_bytes": 0,
            "same_capture_and_vjp": True,
            "dual_corrected_jacobian_factored": True,
            "raw_dual_row_custody_retained": False,
            "equality_law_id": FACTORED_ROW_EQUALITY_LAW_ID,
        }


@dataclass(frozen=True)
class V1FactoredDeviceDLookRegistration:
    """Pre-outcome scalar registration of one logical look on the factored lane.

    Sibling of the legacy ``V1SampledRegisteredLookTransactionRequest``: the
    packet plans and measurement requests are built by the same helpers, but
    no legacy prior result is required or impersonated.  Only identities are
    retained here; the plan objects travel beside it.
    """

    runtime_request_hash: str
    registered_look_index: int
    registered_look_cumulative_stratum_counts: tuple[int, ...]
    first_packet_index: int
    packet_plan_hashes: tuple[str, ...]
    provider_plan_identity_hashes: tuple[str, ...]
    source_capture_identity_hashes: tuple[str, ...]
    packet_stratum_labels: tuple[str, ...]
    packet_context_counts: tuple[int, ...]
    corrected_a_stream_identity_hash: str
    literal_j_stream_identity_hash: str
    session_design_hash: str
    schema: str = FACTORED_DEVICE_RESIDENT_D_LOOK_REGISTRATION_SCHEMA

    def __post_init__(self) -> None:
        for name in ("runtime_request_hash", "corrected_a_stream_identity_hash", "literal_j_stream_identity_hash", "session_design_hash"):
            _identity(getattr(self, name), name=name)
        axes = (self.packet_plan_hashes, self.provider_plan_identity_hashes, self.source_capture_identity_hashes, self.packet_stratum_labels, self.packet_context_counts)
        if not axes[0] or any(len(a) != len(axes[0]) for a in axes):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "look registration packet ledgers must align")
        if len(set(self.packet_plan_hashes)) != len(self.packet_plan_hashes):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "duplicate packet plan in one look")
        _positive_int(self.registered_look_index, name="registered look index")
        _nonnegative_int(self.first_packet_index, name="first packet index")

    @property
    def registration_hash(self) -> str:
        return content_hash(self, prefix=_SESSION_PREFIX + "look-registration-")

    @property
    def request_hash(self) -> str:
        """Alias so boundary code can treat registrations like requests."""

        return self.registration_hash


@dataclass(frozen=True)
class V1FactoredDeviceDLookCommit:
    """Scalar commit of one logical look executed through the factored protocol."""

    look_registration_hash: str
    registered_look_index: int
    packet_plan_hashes: tuple[str, ...]
    provider_plan_identity_hashes: tuple[str, ...]
    source_capture_identity_hashes: tuple[str, ...]
    look_receipt_hashes: tuple[str, ...]
    packet_receipt_hashes: tuple[str, ...]
    provider_receipt_hashes: tuple[str, ...]
    stratum_counts_after: tuple[int, ...]
    source_packet_ordinal_start: int
    source_packet_ordinal_stop: int
    row_payload_present: bool = False
    route_scale_d2h_bytes: int = 0
    schema: str = FACTORED_DEVICE_RESIDENT_D_LOOK_COMMIT_SCHEMA

    def __post_init__(self) -> None:
        axes = (self.packet_plan_hashes, self.provider_plan_identity_hashes, self.source_capture_identity_hashes, self.look_receipt_hashes, self.packet_receipt_hashes, self.provider_receipt_hashes)
        if not axes[0] or any(len(a) != len(axes[0]) for a in axes):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "look commit ledgers must align")
        if self.source_packet_ordinal_stop - self.source_packet_ordinal_start != len(axes[0]):
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "look commit packet cursor does not cover its packets")
        if self.row_payload_present is not False or self.route_scale_d2h_bytes != 0:
            _refuse(V1FactoredRowRefusalCode.AUTHORITY_VIOLATION, "factored look commits carry no rows")

    @property
    def commit_hash(self) -> str:
        return content_hash(self, prefix=_SESSION_PREFIX + "look-commit-")


def execute_v1_factored_device_d_look(
    *,
    registration: V1FactoredDeviceDLookRegistration,
    packet_plans: Sequence[Any],
    measurement_requests: Sequence[Any],
    provider: Any,
    session: V1FactoredDeviceResidentDSession,
    gate: Any,
    source_packet_ordinal_start: int,
) -> V1FactoredDeviceDLookCommit:
    """Drive one frozen logical look through the provider's factored protocol.

    The provider must expose ``accumulate_device_factored(request, *, accumulator)``;
    a provider without it refuses ``FACTORED_PROVIDER_ABSENT`` (no fallback to
    the row protocol).  Packet plans are the frozen plans of the legacy
    builder's helpers; only their scalar identities are read, and they must
    match the pre-outcome registration exactly.
    """

    from .modern_v1_sampled_look_transaction import (
        _provider_plan_identity_hash,
        _source_capture_identity_hash,
    )

    execute = getattr(provider, "accumulate_device_factored", None)
    if not callable(execute):
        _refuse(
            V1FactoredRowRefusalCode.FACTORED_PROVIDER_ABSENT,
            "provider lacks accumulate_device_factored; the factored mode does not fall back to rows",
        )
    if type(registration) is not V1FactoredDeviceDLookRegistration:
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "factored look execution requires its exact registration")
    if registration.session_design_hash != session.design.design_hash:
        _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "look registration names another factored session")
    plans = tuple(packet_plans)
    requests = tuple(measurement_requests)
    if not plans or len(plans) != len(requests) or len(plans) != len(registration.packet_plan_hashes):
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "measurement requests must cover every registered packet plan")
    if tuple(plan.plan_hash for plan in plans) != registration.packet_plan_hashes:
        _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "packet plans differ from the look registration")
    look_index = int(registration.registered_look_index)
    start = _nonnegative_int(source_packet_ordinal_start, name="source packet ordinal start")
    plan_hashes = []
    provider_ids = []
    capture_ids = []
    look_hashes = []
    packet_hashes = []
    provider_receipt_hashes = []
    for local, (plan, measurement) in enumerate(zip(plans, requests, strict=True)):
        packet_request = plan.reduced_request.packet_request
        stratum_label = plan.fd_request.stratum_labels[0]
        stratum_index = tuple(plan.fd_request.design.stratum_names).index(stratum_label)
        order_start = int(packet_request.sealed_order_start_indices[stratum_index])
        provider_id = _provider_plan_identity_hash(measurement)
        capture_id = _source_capture_identity_hash(measurement)
        if (
            provider_id != registration.provider_plan_identity_hashes[local]
            or capture_id != registration.source_capture_identity_hashes[local]
        ):
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "measurement request differs from its pre-outcome registration")
        accumulator = V1FactoredRowPacketAccumulator(
            session=session,
            registered_look_index=look_index,
            packet_ordinal=local,
            expected_context_ids=packet_request.expected_context_ids,
            expected_draw_ids=packet_request.expected_draw_ids,
            stratum_label=stratum_label,
            sealed_order_indices=tuple(range(order_start, order_start + len(packet_request.expected_context_ids))),
            source_measurement_hashes=plan.fd_request.source_measurement_hashes,
            route_ids=plan.reduced_request.route_ids,
            destination_transport_hash=plan.reduced_request.destination_transport_hash,
            gate=gate,
            provider_plan_identity_hash=provider_id,
            source_capture_identity_hash=capture_id,
        )
        provider_receipt = execute(measurement, accumulator=accumulator)
        if not isinstance(provider_receipt, Mapping):
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "factored provider must return one receipt mapping")
        receipt = accumulator.look_receipt
        packet = accumulator.packet
        assert packet is not None
        if provider_receipt.get("factored_packet_receipt_hash") != packet.receipt.receipt_hash:
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "provider receipt does not name the consumed factored packet")
        plan_hashes.append(plan.plan_hash)
        provider_ids.append(provider_id)
        capture_ids.append(capture_id)
        look_hashes.append(receipt.receipt_hash)
        packet_hashes.append(packet.receipt.receipt_hash)
        provider_receipt_hashes.append(content_hash(dict(provider_receipt), prefix=_SESSION_PREFIX + "provider-receipt-"))
    return V1FactoredDeviceDLookCommit(
        look_registration_hash=registration.registration_hash,
        registered_look_index=look_index,
        packet_plan_hashes=tuple(plan_hashes),
        provider_plan_identity_hashes=tuple(provider_ids),
        source_capture_identity_hashes=tuple(capture_ids),
        look_receipt_hashes=tuple(look_hashes),
        packet_receipt_hashes=tuple(packet_hashes),
        provider_receipt_hashes=tuple(provider_receipt_hashes),
        stratum_counts_after=session.stratum_counts,
        source_packet_ordinal_start=start,
        source_packet_ordinal_stop=start + len(plans),
    )


@dataclass(frozen=True)
class V1FactoredDeviceResidentDReplayBoundary:
    """Scalar restart coordinate for one factored live-D prefix (V2 storage boundary)."""

    session_replay_coordinate: V1FactoredDeviceResidentDReplayCoordinateReceipt
    registered_runtime_request_hashes: tuple[str, ...]
    look_registration_hashes: tuple[str, ...]
    look_commit_hashes: tuple[str, ...]
    physical_route_axis_hash: str
    route_id_axis_hash: str
    source_packet_count: int
    provider_workspace_measurement_hash: str
    provider_peak_workspace_bytes: int
    finalization_receipt_hash: str | None = None
    kkt_geometry_receipt_hash: str | None = None
    exact_geometry_receipt_hash: str | None = None
    live_endpoint_registered: bool = False
    deterministic_replay_required: bool = True
    factored_custody: bool = True
    row_payload_present: bool = False
    row_scratch_bytes: int = 0
    raw_row_d2h_bytes: int = 0
    live_session_resume_authority: bool = False
    D_stability_authority: bool = False
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    schema: str = FACTORED_DEVICE_RESIDENT_D_REPLAY_BOUNDARY_SCHEMA

    def __post_init__(self) -> None:
        if type(self.session_replay_coordinate) is not V1FactoredDeviceResidentDReplayCoordinateReceipt:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "factored replay boundary requires its exact coordinate")
        axes = (self.registered_runtime_request_hashes, self.look_registration_hashes, self.look_commit_hashes)
        if not axes[0] or any(len(a) != len(axes[0]) for a in axes) or any(len(set(a)) != len(a) for a in axes):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "factored replay boundary has an incomplete look chain")
        if self.session_replay_coordinate.next_registered_look_index != len(axes[0]) + 1:
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "replay coordinate does not sit at the chain's next look")
        if self.physical_route_axis_hash == self.route_id_axis_hash or self.session_replay_coordinate.route_id_axis_hash != self.route_id_axis_hash:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "factored replay boundary collapsed route axes")
        packets = _positive_int(self.source_packet_count, name="source packet count")
        if len(self.session_replay_coordinate.consumed_look_receipt_hashes) != packets or len(self.session_replay_coordinate.packet_receipt_hashes) != packets:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "factored replay boundary packet custody does not close")
        if self.live_endpoint_registered:
            if self.finalization_receipt_hash is None or self.kkt_geometry_receipt_hash is None or self.exact_geometry_receipt_hash is None:
                _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "a registered factored endpoint names its finalization and geometry")
        elif any(v is not None for v in (self.finalization_receipt_hash, self.kkt_geometry_receipt_hash, self.exact_geometry_receipt_hash)):
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "endpoint receipts before the endpoint")
        if self.deterministic_replay_required is not True or self.factored_custody is not True or self.row_payload_present is not False or self.row_scratch_bytes != 0 or self.raw_row_d2h_bytes != 0 or self.live_session_resume_authority is not False:
            _refuse(V1FactoredRowRefusalCode.AUTHORITY_VIOLATION, "factored replay boundary custody flags drifted")
        _closed_flags(self, _AUTHORITY_FLAGS)
        if self.schema != FACTORED_DEVICE_RESIDENT_D_REPLAY_BOUNDARY_SCHEMA:
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored replay boundary schema drifted")

    @property
    def boundary_hash(self) -> str:
        return content_hash(self, prefix=_REPLAY_BOUNDARY_PREFIX)

    def to_json_payload(self) -> dict[str, Any]:
        coordinate = self.session_replay_coordinate
        payload = {
            "schema": self.schema,
            "session_replay_coordinate": {
                "schema": coordinate.schema,
                "session_design_hash": coordinate.session_design_hash,
                "forecast_hash": coordinate.forecast_hash,
                "sealed_model_context_ledger_hash": coordinate.sealed_model_context_ledger_hash,
                "route_id_axis_hash": coordinate.route_id_axis_hash,
                "consumed_look_receipt_hashes": list(coordinate.consumed_look_receipt_hashes),
                "packet_receipt_hashes": list(coordinate.packet_receipt_hashes),
                "next_registered_look_index": coordinate.next_registered_look_index,
                "receipt_hash": coordinate.receipt_hash,
            },
        }
        for name in (
            "registered_runtime_request_hashes",
            "look_registration_hashes",
            "look_commit_hashes",
            "physical_route_axis_hash",
            "route_id_axis_hash",
            "source_packet_count",
            "provider_workspace_measurement_hash",
            "provider_peak_workspace_bytes",
            "finalization_receipt_hash",
            "kkt_geometry_receipt_hash",
            "exact_geometry_receipt_hash",
            "live_endpoint_registered",
            "deterministic_replay_required",
            "factored_custody",
            "row_payload_present",
            "row_scratch_bytes",
            "raw_row_d2h_bytes",
            "live_session_resume_authority",
            *_AUTHORITY_FLAGS,
        ):
            value = getattr(self, name)
            payload[name] = list(value) if isinstance(value, tuple) else value
        payload["boundary_hash"] = self.boundary_hash
        # The backend's authority-inert receipt writer requires the typed hash
        # under ``receipt_hash``; it is the same boundary hash, never a re-hash.
        payload["receipt_hash"] = self.boundary_hash
        return payload

    @classmethod
    def from_json_payload(cls, payload: Mapping[str, Any]) -> "V1FactoredDeviceResidentDReplayBoundary":
        data = dict(payload)
        declared = data.pop("boundary_hash", None)
        receipt_alias = data.pop("receipt_hash", None)
        if receipt_alias is not None and receipt_alias != declared:
            raise ValueError("factored replay boundary receipt alias disagrees with its boundary hash")
        coordinate_payload = dict(data.pop("session_replay_coordinate"))
        declared_coordinate = coordinate_payload.pop("receipt_hash", None)
        coordinate_payload.pop("schema", None)
        coordinate = V1FactoredDeviceResidentDReplayCoordinateReceipt(
            **{
                **coordinate_payload,
                "consumed_look_receipt_hashes": tuple(coordinate_payload["consumed_look_receipt_hashes"]),
                "packet_receipt_hashes": tuple(coordinate_payload["packet_receipt_hashes"]),
            }
        )
        if coordinate.receipt_hash != declared_coordinate:
            raise ValueError("factored replay coordinate does not replay its stored hash")
        data.pop("schema", None)
        for name in ("registered_runtime_request_hashes", "look_registration_hashes", "look_commit_hashes"):
            data[name] = tuple(data[name])
        boundary = cls(session_replay_coordinate=coordinate, **data)
        if boundary.boundary_hash != declared:
            raise ValueError("factored replay boundary does not replay its stored hash")
        return boundary


@dataclass(frozen=True)
class V1FactoredDeviceResidentDLiveHandoff:
    """Nonserializable live factored-D owner (session plus scalar chain)."""

    session: V1FactoredDeviceResidentDSession = field(repr=False)
    look_registrations: tuple[V1FactoredDeviceDLookRegistration, ...]
    look_commits: tuple[V1FactoredDeviceDLookCommit, ...]
    replay_boundary: V1FactoredDeviceResidentDReplayBoundary
    finalization_receipt: V1FactoredDeviceResidentDFinalizationReceipt | None = None
    kkt_geometry_receipt: V1FactoredDeviceResidentDKKTGeometryReceipt | None = None
    exact_geometry: Any = field(default=None, repr=False, compare=False)
    kkt_dense_adapter: Any = field(default=None, repr=False, compare=False)
    singleton_work: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.assert_live()

    def assert_live(self, expected_boundary_hash: str | None = None) -> str:
        """Join the live packets and completed look chain to the exact replay boundary."""

        if type(self.session) is not V1FactoredDeviceResidentDSession:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "factored live handoff requires its exact session")
        self.session._assert_live()
        if type(self.replay_boundary) is not V1FactoredDeviceResidentDReplayBoundary:
            _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "factored handoff requires its exact replay boundary")
        self.replay_boundary.__post_init__()
        commits = tuple(self.look_commits)
        if (
            not commits
            or len(commits) != len(self.look_registrations)
            or any(type(c) is not V1FactoredDeviceDLookCommit for c in commits)
            or any(type(r) is not V1FactoredDeviceDLookRegistration for r in self.look_registrations)
            or tuple(c.commit_hash for c in commits) != self.replay_boundary.look_commit_hashes
            or tuple(r.registration_hash for r in self.look_registrations) != self.replay_boundary.look_registration_hashes
        ):
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored live handoff lost its exact look chain")
        if (self.finalization_receipt is None) != (self.kkt_geometry_receipt is None) or (self.finalization_receipt is None) != (not self.replay_boundary.live_endpoint_registered):
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "factored handoff endpoint receipts and boundary disagree")
        boundary = self.replay_boundary
        design = self.session.design
        if (boundary.physical_route_axis_hash != design.physical_route_axis_hash
                or boundary.route_id_axis_hash != design.route_id_axis_hash):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "factored handoff and live session route axes differ")
        replay = self.session.replay_coordinate_receipt(
            sealed_model_context_ledger_hash=boundary.session_replay_coordinate.sealed_model_context_ledger_hash,
        )
        if replay.receipt_hash != boundary.session_replay_coordinate.receipt_hash:
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored handoff no longer names its live session prefix")
        packet_hashes = tuple(packet.assert_integrity() for _, packet in self.session.packets)
        if packet_hashes != tuple(h for commit in commits for h in commit.packet_receipt_hashes):
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored commits and live packet custody differ")
        look_receipts = self.session.look_receipts
        if tuple(r.receipt_hash for r in look_receipts) != tuple(
                h for commit in commits for h in commit.look_receipt_hashes):
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored commits and look receipt custody differ")
        offsets = [0] * len(design.stratum_names)
        for (stratum, packet), receipt in zip(self.session.packets, look_receipts, strict=True):
            if type(stratum) is not int or not 0 <= stratum < len(offsets):
                _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "factored packet has invalid stratum")
            start = offsets[stratum]
            stop = start + packet.receipt.context_count
            contexts = tuple(self.session._context_ledgers[stratum][start:stop])
            draws = tuple(self.session._draw_ledgers[stratum][start:stop])
            sources = tuple(self.session._source_measurements[stratum][start:stop])
            if (contexts != packet.receipt.context_ids
                    or len(draws) != stop - start or len(sources) != stop - start
                    or receipt.stratum_name != design.stratum_names[stratum]
                    or receipt.packet_receipt_hash != packet.receipt.receipt_hash
                    or receipt.session_design_hash != design.design_hash
                    or receipt.forecast_hash != self.session.forecast.forecast_hash
                    or receipt.stratum_counts_before != tuple(offsets)
                    or receipt.context_axis_hash != _axis_hash(contexts, role="look-context")
                    or receipt.draw_axis_hash != _axis_hash(draws, role="look-draw")
                    or receipt.source_measurement_axis_hash != _axis_hash(sources, role="look-source-measurement")):
                _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored source ledgers do not replay captured packet axes")
            offsets[stratum] = stop
            if receipt.stratum_counts_after != tuple(offsets):
                _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "factored look counts do not replay packet lengths")
        if (tuple(offsets) != self.session.stratum_counts
                or any(tuple(map(len, ledgers)) != tuple(offsets) for ledgers in (
                    self.session._context_ledgers, self.session._draw_ledgers, self.session._source_measurements))):
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "factored source ledgers have an uncommitted tail")
        cursor = 0
        for index, (registration, commit) in enumerate(zip(self.look_registrations, commits, strict=True), 1):
            if (registration.registered_look_index != index or commit.registered_look_index != index
                    or commit.look_registration_hash != registration.registration_hash
                    or registration.runtime_request_hash != boundary.registered_runtime_request_hashes[index - 1]
                    or registration.session_design_hash != design.design_hash
                    or registration.corrected_a_stream_identity_hash != design.corrected_a_stream_identity_hash
                    or registration.literal_j_stream_identity_hash != design.literal_j_stream_identity_hash
                    or commit.stratum_counts_after != registration.registered_look_cumulative_stratum_counts
                    or commit.source_packet_ordinal_start != cursor
                    or commit.packet_plan_hashes != registration.packet_plan_hashes
                    or commit.provider_plan_identity_hashes != registration.provider_plan_identity_hashes
                    or commit.source_capture_identity_hashes != registration.source_capture_identity_hashes):
                _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored registered look does not join its exact commit")
            cursor = commit.source_packet_ordinal_stop
        if cursor != boundary.source_packet_count or commits[-1].stratum_counts_after != self.session.stratum_counts:
            _refuse(V1FactoredRowRefusalCode.ORDER_MISMATCH, "factored handoff packet/count prefix does not close")
        if self.finalization_receipt is not None:
            if (self.finalization_receipt.receipt_hash != boundary.finalization_receipt_hash
                    or self.kkt_geometry_receipt.receipt_hash != boundary.kkt_geometry_receipt_hash
                    or self.kkt_geometry_receipt.exact_geometry_receipt_hash != boundary.exact_geometry_receipt_hash):
                _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored endpoint receipts no longer replay their boundary")
        if self.session._capture_admission is not None:
            self.session._capture_admission.session_payload(self.session)
        observed = boundary.boundary_hash
        if expected_boundary_hash is not None and observed != expected_boundary_hash:
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "factored handoff boundary differs from the expected source")
        return observed

    @property
    def evicted(self) -> bool:
        return self.session._evicted

    def evict(self) -> None:
        if self.exact_geometry is not None:
            self.exact_geometry.evict()
        self.session.evict()

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("factored live handoffs are nonserializable")


__all__ += [
    "FACTORED_DEVICE_RESIDENT_D_AUTHORITY_SCOPE",
    "FACTORED_DEVICE_RESIDENT_D_CUSTODY_FORECAST_SCHEMA",
    "FACTORED_DEVICE_RESIDENT_D_FINALIZATION_SCHEMA",
    "FACTORED_DEVICE_RESIDENT_D_KKT_GEOMETRY_SCHEMA",
    "FACTORED_DEVICE_RESIDENT_D_LOOK_COMMIT_SCHEMA",
    "FACTORED_DEVICE_RESIDENT_D_LOOK_RECEIPT_SCHEMA",
    "FACTORED_DEVICE_RESIDENT_D_MOMENT_LAW_ID",
    "FACTORED_DEVICE_RESIDENT_D_PACKET_STREAM_RECEIPT_SCHEMA",
    "FACTORED_DEVICE_RESIDENT_D_REPLAY_BOUNDARY_SCHEMA",
    "FACTORED_DEVICE_RESIDENT_D_REPLAY_COORDINATE_SCHEMA",
    "FACTORED_DEVICE_RESIDENT_D_SESSION_DESIGN_SCHEMA",
    "V1FactoredDeviceDLookCommit",
    "V1FactoredDeviceDLookRegistration",
    "FACTORED_DEVICE_RESIDENT_D_LOOK_REGISTRATION_SCHEMA",
    "V1FactoredDeviceResidentDCustodyForecast",
    "V1FactoredDeviceResidentDFinalizationReceipt",
    "V1FactoredDeviceResidentDKKTGeometryReceipt",
    "V1FactoredDeviceResidentDLiveHandoff",
    "V1FactoredDeviceResidentDLookReceipt",
    "V1FactoredDeviceResidentDReplayBoundary",
    "V1FactoredDeviceResidentDReplayCoordinateReceipt",
    "V1FactoredDeviceResidentDSession",
    "V1FactoredDeviceResidentDSessionDesign",
    "V1FactoredDeviceResidentDSessionMode",
    "V1FactoredRowPacketAccumulator",
    "execute_v1_factored_device_d_look",
    "forecast_v1_factored_device_resident_d_session",
]
