"""Route-chunked frozen-cotangent gate-removal disclosure for modern v1.

This module computes the diagnostic singleton quantity

``G * (activation(a - c_r) - activation(a))``

for every executable route ``r`` of one captured affine block.  ``G`` is a
*frozen post-activation cotangent*.  Consequently the result is an exact
one-layer consequence at the captured state, but it is neither an exact
end-to-end loss difference nor the conserving Aumann--Shapley ordering key
``G * phi * c_r``.

Convolutional execution is output-channel and route chunked.  The spatial
contraction uses a matrix product between one output channel's cotangent and
the chunk's activation changes; it never constructs a
``context x query x full-route x spatial`` tensor.  Only weighted signed and
absolute context sums are retained.  A caller that needs context rows may
consume ephemeral ``[context, query, route_chunk]`` tensors through a
callback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any, Final, Literal

from mechanistic_systems.core.numerics import CapturedRouteBlock
from mechanistic_systems.core.storage import RouteBlockSpec

from .modern_v1_conserving_attribution import EXACT_SINGLETON_DISCLOSURE_ID


GATE_DISCLOSURE_SCHEMA: Final = (
    "partition_free_v1_frozen_gate_removal_disclosure_v1"
)
GATE_DISCLOSURE_EQUATION_ID: Final = (
    "frozen_postactivation_cotangent_times_activation_a_minus_c_minus_"
    "activation_a_v1"
)
GATE_DISCLOSURE_ROLE: Final = (
    "diagnostic_singleton_sidecar_not_conserving_ordering_key_v1"
)

ActivationKind = Literal["relu", "identity"]


def _torch() -> Any:
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
        raise RuntimeError("gate-removal disclosure requires PyTorch") from exc
    return torch


@dataclass(frozen=True, slots=True)
class V1GateDisclosureChunk:
    """One ephemeral context/query/route chunk delivered to a consumer.

    ``values`` has shape ``[context, query, flat_route]`` and positive-removal
    orientation.  Consumers must copy or aggregate it during the callback if
    they need to retain it; the result object never stores these rows.  Any
    consumer output remains provisional until the stream function returns,
    because final finite-value validation occurs after all chunks are reduced.
    """

    route_block: RouteBlockSpec
    flat_start: int
    flat_stop: int
    output_index: int | None
    values: Any
    activation_kind: ActivationKind
    equation_id: str = GATE_DISCLOSURE_EQUATION_ID
    semantic_role: str = GATE_DISCLOSURE_ROLE
    schema: str = GATE_DISCLOSURE_SCHEMA

    def __post_init__(self) -> None:
        torch = _torch()
        self.route_block.validate()
        start = int(self.flat_start)
        stop = int(self.flat_stop)
        if (
            isinstance(self.flat_start, bool)
            or isinstance(self.flat_stop, bool)
            or start < 0
            or stop <= start
            or stop > self.route_block.route_count
        ):
            raise ValueError("gate-disclosure chunk has invalid route bounds")
        if not isinstance(self.values, torch.Tensor) or self.values.ndim != 3:
            raise ValueError(
                "gate-disclosure callback values must be [context,query,route]"
            )
        if int(self.values.shape[2]) != stop - start:
            raise ValueError(
                "gate-disclosure callback route axis is misaligned"
            )
        if self.output_index is None:
            if self.route_block.route_kind != "linear_weight":
                raise ValueError(
                    "mixed-output disclosure chunks are linear-only"
                )
        elif (
            isinstance(self.output_index, bool)
            or int(self.output_index) < 0
            or int(self.output_index) >= int(self.route_block.shape[0])
        ):
            raise ValueError("gate-disclosure output index is invalid")
        if not self.values.is_floating_point():
            raise ValueError(
                "gate-disclosure callback values must be floating"
            )
        if self.activation_kind not in {"relu", "identity"}:
            raise ValueError("unsupported gate-disclosure activation")
        if self.equation_id != GATE_DISCLOSURE_EQUATION_ID:
            raise ValueError("unsupported gate-disclosure equation")
        if self.semantic_role != GATE_DISCLOSURE_ROLE:
            raise ValueError("unsupported gate-disclosure semantic role")
        if self.schema != GATE_DISCLOSURE_SCHEMA:
            raise ValueError("unsupported gate-disclosure schema")


V1GateDisclosureConsumer = Callable[[V1GateDisclosureChunk], None]
V1GateKinkConsumer = Callable[[int, int, Any], None]


@dataclass(frozen=True, slots=True)
class V1GateDisclosureMemoryReceipt:
    """Analytical and observed tensor-extent receipt for one streamed block.

    The element counts describe logical fields, not allocator peak memory.
    PyTorch workspace and convolution-unfold implementation details can add
    temporary storage beyond these fields, so this receipt must not be
    presented as a measured CUDA peak.
    """

    context_count: int
    query_count: int
    route_count: int
    output_count: int
    spatial_extent: int
    requested_route_chunk_size: int
    maximum_observed_route_chunk_size: int
    chunk_count: int
    callback_chunk_count: int
    source_patch_elements: int
    maximum_route_drive_elements: int
    maximum_selected_cotangent_elements: int
    maximum_callback_elements: int
    aggregate_elements: int
    theoretical_full_context_query_route_spatial_elements: int
    full_context_query_route_spatial_tensor_materialized: bool = False
    per_context_full_route_field_retained: bool = False
    per_route_moments_retained: bool = True
    bound_kind: str = "logical_tensor_extent_not_allocator_peak_v1"
    schema: str = GATE_DISCLOSURE_SCHEMA

    def __post_init__(self) -> None:
        positive = (
            "context_count",
            "query_count",
            "route_count",
            "output_count",
            "spatial_extent",
            "requested_route_chunk_size",
            "maximum_observed_route_chunk_size",
            "chunk_count",
        )
        nonnegative = (
            "callback_chunk_count",
            "source_patch_elements",
            "maximum_route_drive_elements",
            "maximum_selected_cotangent_elements",
            "maximum_callback_elements",
            "aggregate_elements",
            "theoretical_full_context_query_route_spatial_elements",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in nonnegative:
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            self.maximum_observed_route_chunk_size
            > self.requested_route_chunk_size
        ):
            raise ValueError(
                "observed disclosure chunk exceeds its declared bound"
            )
        if self.callback_chunk_count not in {0, self.chunk_count}:
            raise ValueError(
                "disclosure callback must consume every chunk or none"
            )
        if self.full_context_query_route_spatial_tensor_materialized:
            raise PermissionError(
                "full context-query-route-spatial disclosure is prohibited"
            )
        if self.per_context_full_route_field_retained:
            raise PermissionError(
                "per-context full-route disclosure is prohibited"
            )
        expected_aggregate_elements = (
            2 * self.query_count * self.route_count
            if self.per_route_moments_retained
            else 0
        )
        if self.aggregate_elements != expected_aggregate_elements:
            raise ValueError(
                "gate-disclosure aggregate extent differs from retention mode"
            )
        if self.bound_kind != "logical_tensor_extent_not_allocator_peak_v1":
            raise ValueError("unsupported gate-disclosure memory-bound kind")
        if self.schema != GATE_DISCLOSURE_SCHEMA:
            raise ValueError("unsupported gate-disclosure receipt schema")


@dataclass(frozen=True, slots=True)
class V1GateDisclosureMoments:
    """Weighted sufficient statistics for the singleton disclosure sidecar."""

    route_block: RouteBlockSpec
    signed_weighted_sum: Any
    absolute_weighted_sum: Any
    total_weight: float
    context_count: int
    query_count: int
    activation_kind: ActivationKind
    memory_receipt: V1GateDisclosureMemoryReceipt
    ordering_key_eligible: bool = False
    end_to_end_loss_exact: bool = False
    equation_id: str = GATE_DISCLOSURE_EQUATION_ID
    relu_equation_id: str = EXACT_SINGLETON_DISCLOSURE_ID
    semantic_role: str = GATE_DISCLOSURE_ROLE
    schema: str = GATE_DISCLOSURE_SCHEMA

    def __post_init__(self) -> None:
        torch = _torch()
        self.route_block.validate()
        expected = (int(self.query_count), *self.route_block.shape)
        for name in ("signed_weighted_sum", "absolute_weighted_sum"):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != expected
                or value.requires_grad
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(
                    f"{name} must be a finite detached {expected} tensor"
                )
        if bool(torch.any(self.absolute_weighted_sum < 0.0).item()):
            raise ValueError(
                "absolute gate-disclosure moment must be nonnegative"
            )
        if (
            isinstance(self.context_count, bool)
            or int(self.context_count) <= 0
            or isinstance(self.query_count, bool)
            or int(self.query_count) <= 0
            or not math.isfinite(float(self.total_weight))
            or float(self.total_weight) <= 0.0
        ):
            raise ValueError(
                "gate-disclosure aggregate axes and weight must be positive"
            )
        if self.activation_kind not in {"relu", "identity"}:
            raise ValueError("unsupported gate-disclosure activation")
        if self.ordering_key_eligible or self.end_to_end_loss_exact:
            raise PermissionError(
                "frozen-cotangent disclosure cannot claim ordering or loss "
                "authority"
            )
        if self.equation_id != GATE_DISCLOSURE_EQUATION_ID:
            raise ValueError("unsupported gate-disclosure equation")
        if self.relu_equation_id != EXACT_SINGLETON_DISCLOSURE_ID:
            raise ValueError("unsupported ReLU singleton equation")
        if self.semantic_role != GATE_DISCLOSURE_ROLE:
            raise ValueError("unsupported gate-disclosure semantic role")
        if self.schema != GATE_DISCLOSURE_SCHEMA:
            raise ValueError("unsupported gate-disclosure schema")
        if (
            self.memory_receipt.context_count != int(self.context_count)
            or self.memory_receipt.query_count != int(self.query_count)
            or self.memory_receipt.route_count != self.route_block.route_count
        ):
            raise ValueError(
                "gate-disclosure moments and memory receipt differ"
            )
        if not self.memory_receipt.per_route_moments_retained:
            raise ValueError(
                "gate-disclosure moments require retained per-route aggregates"
            )

    @property
    def signed_weighted_mean(self) -> Any:
        return self.signed_weighted_sum / float(self.total_weight)

    @property
    def absolute_weighted_mean(self) -> Any:
        return self.absolute_weighted_sum / float(self.total_weight)


@dataclass(frozen=True, slots=True)
class V1GateDisclosureStreamResult:
    """Receipt-only result for an externally consumed disclosure stream."""

    route_block: RouteBlockSpec
    context_count: int
    query_count: int
    activation_kind: ActivationKind
    memory_receipt: V1GateDisclosureMemoryReceipt
    ordering_key_eligible: bool = False
    end_to_end_loss_exact: bool = False
    equation_id: str = GATE_DISCLOSURE_EQUATION_ID
    relu_equation_id: str = EXACT_SINGLETON_DISCLOSURE_ID
    semantic_role: str = GATE_DISCLOSURE_ROLE
    schema: str = GATE_DISCLOSURE_SCHEMA

    def __post_init__(self) -> None:
        self.route_block.validate()
        if (
            isinstance(self.context_count, bool)
            or int(self.context_count) <= 0
            or isinstance(self.query_count, bool)
            or int(self.query_count) <= 0
            or self.memory_receipt.context_count != int(self.context_count)
            or self.memory_receipt.query_count != int(self.query_count)
            or self.memory_receipt.route_count != self.route_block.route_count
            or self.memory_receipt.per_route_moments_retained
            or self.memory_receipt.aggregate_elements != 0
        ):
            raise ValueError(
                "stream-only gate-disclosure result has invalid axes"
            )
        if self.activation_kind not in {"relu", "identity"}:
            raise ValueError("unsupported gate-disclosure activation")
        if self.ordering_key_eligible or self.end_to_end_loss_exact:
            raise PermissionError(
                "frozen-cotangent disclosure cannot claim ordering or loss "
                "authority"
            )
        if self.equation_id != GATE_DISCLOSURE_EQUATION_ID:
            raise ValueError("unsupported gate-disclosure equation")
        if self.relu_equation_id != EXACT_SINGLETON_DISCLOSURE_ID:
            raise ValueError("unsupported ReLU singleton equation")
        if self.semantic_role != GATE_DISCLOSURE_ROLE:
            raise ValueError("unsupported gate-disclosure semantic role")
        if self.schema != GATE_DISCLOSURE_SCHEMA:
            raise ValueError("unsupported gate-disclosure schema")


def _validate_inputs(
    captured: CapturedRouteBlock,
    post_activation_cotangent: Any,
    preactivation: Any,
    activation_kind: str,
) -> tuple[Any, int, int, int, int]:
    torch = _torch()
    if captured.reducer_kind not in {
        "linear_weight",
        "conv_channel_pair",
        "conv_kernel_weight",
    }:
        raise ValueError("gate disclosure supports only affine route families")
    if captured.route_block.route_kind != captured.reducer_kind:
        raise ValueError(
            "gate-disclosure route kind differs from its affine reducer"
        )
    if captured.reducer_kind == "linear_weight":
        captured.validate()
    else:
        # The canonical pruning axis may retain the full logical input axis
        # for grouped convolutions, including structurally disconnected
        # cross-group coordinates.  The incumbent route reducer validates
        # only stored input-within-group axes, so disclosure validates its
        # broader, explicitly zero-extended axis locally instead.
        captured.route_block.validate()
        if (
            not isinstance(captured.output, torch.Tensor)
            or not captured.output.requires_grad
            or captured.output.ndim != 4
            or not math.isfinite(float(captured.scale))
            or float(captured.scale) == 0.0
            or not isinstance(captured.input_source, torch.Tensor)
            or captured.input_source.ndim != 4
            or not isinstance(captured.weight, torch.Tensor)
            or captured.weight.ndim != 4
            or not {"stride", "padding", "dilation", "groups"}.issubset(
                captured.reducer_parameters
            )
        ):
            raise ValueError(
                "gate disclosure has an invalid convolutional capture"
            )
    if activation_kind not in {"relu", "identity"}:
        raise ValueError("activation_kind must be explicitly relu or identity")
    if not isinstance(preactivation, torch.Tensor):
        raise TypeError("preactivation must be a torch tensor")
    if (
        preactivation.shape != captured.output.shape
        or preactivation.dtype != captured.output.dtype
        or preactivation.device != captured.output.device
        or not torch.equal(preactivation, captured.output)
    ):
        raise ValueError(
            "preactivation must exactly match the captured affine output"
        )
    if (
        not preactivation.is_floating_point()
        or preactivation.dtype not in {torch.float32, torch.float64}
        or not bool(torch.isfinite(preactivation).all().item())
    ):
        raise ValueError("preactivation must be finite float32 or float64")
    if (
        not isinstance(post_activation_cotangent, torch.Tensor)
        or post_activation_cotangent.ndim != preactivation.ndim + 1
        or tuple(post_activation_cotangent.shape[1:])
        != tuple(preactivation.shape)
        or post_activation_cotangent.dtype != preactivation.dtype
        or post_activation_cotangent.device != preactivation.device
        or not bool(torch.isfinite(post_activation_cotangent).all().item())
    ):
        raise ValueError(
            "post-activation cotangent must be finite [query,context,*unit] "
            "and share preactivation dtype/device"
        )
    source = captured.input_source
    weight = captured.weight
    if not all(isinstance(value, torch.Tensor) for value in (source, weight)):
        raise TypeError("affine route source and weight must be torch tensors")
    if not all(
        value.is_floating_point()
        and value.dtype == preactivation.dtype
        and value.device == preactivation.device
        and bool(torch.isfinite(value).all().item())
        for value in (source, weight)
    ):
        raise ValueError(
            "affine route source, weight, and preactivation must be "
            "finite and "
            "share dtype/device"
        )
    contexts = int(preactivation.shape[0])
    queries = int(post_activation_cotangent.shape[0])
    outputs = int(preactivation.shape[1])
    spatial = math.prod(int(value) for value in preactivation.shape[2:]) or 1
    if contexts <= 0 or queries <= 0 or outputs <= 0 or spatial <= 0:
        raise ValueError("gate-disclosure axes must be nonempty")
    return torch, contexts, queries, outputs, spatial


def _context_weights(
    *,
    torch: Any,
    value: Any | None,
    contexts: int,
    source_device: Any,
    aggregate_device: Any,
    aggregate_dtype: Any,
) -> tuple[Any, float]:
    if value is None:
        weights = torch.ones(
            contexts,
            device=aggregate_device,
            dtype=aggregate_dtype,
        )
    else:
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 1
            or tuple(value.shape) != (contexts,)
            or value.device != source_device
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
            or bool(torch.any(value < 0.0).item())
        ):
            raise ValueError(
                "context_weights must be finite nonnegative [context] on the "
                "capture device"
            )
        weights = value.detach().to(
            device=aggregate_device,
            dtype=aggregate_dtype,
        )
    total = float(weights.sum().item())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("context_weights must have positive finite mass")
    return weights, total


def _conv_patches(captured: CapturedRouteBlock) -> Any:
    from torch.nn import functional as functional

    parameters = captured.reducer_parameters
    weight = captured.weight.detach()
    source = captured.input_source.detach()
    patches = functional.unfold(
        source,
        kernel_size=tuple(int(value) for value in weight.shape[-2:]),
        dilation=parameters["dilation"],
        padding=parameters["padding"],
        stride=parameters["stride"],
    )
    contexts = int(source.shape[0])
    groups = int(parameters["groups"])
    input_per_group = int(weight.shape[1])
    kernel_elements = int(weight.shape[2] * weight.shape[3])
    if int(weight.shape[0]) % groups:
        raise ValueError(
            "convolutional output channels do not divide into groups"
        )
    return patches.reshape(
        contexts,
        groups,
        input_per_group,
        kernel_elements,
        -1,
    )


def compute_frozen_gate_removal_disclosure(
    captured: CapturedRouteBlock,
    post_activation_cotangent: Any,
    preactivation: Any,
    *,
    activation_kind: ActivationKind,
    context_weights: Any | None = None,
    route_chunk_size: int = 4096,
    consumer: V1GateDisclosureConsumer | None = None,
    aggregate_device: Any | None = None,
    aggregate_dtype: Any | None = None,
    retain_moments: bool = True,
    kink_consumer: V1GateKinkConsumer | None = None,
) -> V1GateDisclosureMoments | V1GateDisclosureStreamResult:
    """Stream exact singleton gate disclosures for one affine route block.

    Retained moments are weighted sums over contexts.  ReLU uses
    ``G*(relu(a-c_r)-relu(a))``; identity uses the exact simplification
    ``-G*c_r``.  Spatial unit axes are contracted before the contextwise
    absolute value, matching the scalar route-disclosure estimand.

    ``consumer`` is invoked once per route chunk with detached values in
    positive-removal orientation.  Exceptions propagate and no partial
    result is returned.  Consumer-side writes must remain provisional until
    this function returns successfully: aggregate validation is deliberately
    deferred to the end to avoid a device-to-host synchronization per chunk.
    Set ``retain_moments=False`` only with a consumer to avoid allocating the
    duplicate query-by-route signed and absolute arrays when that consumer
    owns the sufficient statistics.  The returned stream result then retains
    only its bounded-memory receipt.

    ``kink_consumer`` (optional, diagnostic) receives, per route chunk,
    ``(flat_start, flat_stop, rho)`` where ``rho[r]`` is the in-layer kink
    rate of route ``r``: the largest, over contexts and spatial units, of
    ``(c_r * sign(a))^+ / |a|`` for that route's drive ``c_r`` against its
    own unit's preactivation ``a``.  Removing the route by a fraction ``t``
    flips no gate of its own unit while ``t * rho[r] < 1``; an exactly zero
    preactivation yields ``+inf``.  Identity activations yield zeros.  The
    consumer sees a detached float64 device tensor and must reduce it in
    place; nothing route-scale is retained here.
    """

    torch, contexts, queries, outputs, spatial = _validate_inputs(
        captured,
        post_activation_cotangent,
        preactivation,
        activation_kind,
    )
    if isinstance(route_chunk_size, bool) or int(route_chunk_size) <= 0:
        raise ValueError("route_chunk_size must be a positive integer")
    route_chunk_size = int(route_chunk_size)
    if consumer is not None and not callable(consumer):
        raise TypeError("gate-disclosure consumer must be callable")
    if kink_consumer is not None and not callable(kink_consumer):
        raise TypeError("gate-disclosure kink consumer must be callable")
    if type(retain_moments) is not bool:
        raise TypeError("retain_moments must be bool")
    if not retain_moments and consumer is None:
        raise ValueError("stream-only gate disclosure requires a consumer")
    if aggregate_dtype is None:
        aggregate_dtype = torch.float64
    if aggregate_dtype not in {torch.float32, torch.float64}:
        raise TypeError(
            "aggregate_dtype must be torch.float32 or torch.float64"
        )
    resolved_aggregate_device = (
        preactivation.device
        if aggregate_device is None
        else torch.device(aggregate_device)
    )
    weights = None
    total_weight = None
    if retain_moments or context_weights is not None:
        weights, total_weight = _context_weights(
            torch=torch,
            value=context_weights,
            contexts=contexts,
            source_device=preactivation.device,
            aggregate_device=resolved_aggregate_device,
            aggregate_dtype=aggregate_dtype,
        )

    block = captured.route_block
    routes = block.route_count
    if int(block.shape[0]) != outputs or routes % outputs:
        raise ValueError("route axis does not partition by affine output unit")
    routes_per_output = routes // outputs
    signed_flat = None
    absolute_flat = None
    if retain_moments:
        signed_flat = torch.empty(
            (queries, routes),
            device=resolved_aggregate_device,
            dtype=aggregate_dtype,
        )
        absolute_flat = torch.empty_like(signed_flat)

    patches = None
    source_patch_elements = 0
    if captured.reducer_kind in {"conv_channel_pair", "conv_kernel_weight"}:
        patches = _conv_patches(captured)
        source_patch_elements = int(patches.numel())
        if int(patches.shape[-1]) != spatial:
            raise ValueError(
                "captured convolution output differs from stride/padding/"
                "dilation patch geometry"
            )

    pre = preactivation.detach().reshape(contexts, outputs, spatial)
    cotangent = post_activation_cotangent.detach().reshape(
        queries,
        contexts,
        outputs,
        spatial,
    )
    source = captured.input_source.detach()
    weight = captured.weight.detach()
    scale = float(captured.scale)
    conv_groups = 1
    conv_input_per_group = 0
    conv_logical_input_count = 0
    conv_full_logical_axis = False
    conv_kernel_elements = 1
    if captured.reducer_kind in {"conv_channel_pair", "conv_kernel_weight"}:
        conv_groups = int(captured.reducer_parameters["groups"])
        conv_input_per_group = int(weight.shape[1])
        physical_input_count = int(source.shape[1])
        expected_axis_rank = (
            4 if captured.reducer_kind == "conv_kernel_weight" else 2
        )
        if len(block.shape) != expected_axis_rank:
            raise ValueError(
                "gate disclosure has an invalid convolutional route-axis rank"
            )
        conv_logical_input_count = int(block.shape[1])
        conv_full_logical_axis = (
            conv_logical_input_count == physical_input_count
        )
        if (
            conv_groups <= 0
            or int(source.shape[0]) != contexts
            or int(weight.shape[0]) != outputs
            or outputs % conv_groups
            or conv_groups * conv_input_per_group != physical_input_count
            or conv_logical_input_count
            not in {conv_input_per_group, physical_input_count}
        ):
            raise ValueError(
                "gate disclosure has invalid grouped logical-input geometry"
            )
        if captured.reducer_kind == "conv_kernel_weight":
            if tuple(block.shape[2:]) != tuple(
                int(value) for value in weight.shape[2:]
            ):
                raise ValueError(
                    "gate disclosure kernel taps differ from the logical axis"
                )
            conv_kernel_elements = int(weight.shape[2] * weight.shape[3])
    chunk_count = 0
    callback_count = 0
    maximum_chunk = 0
    maximum_drive_elements = 0
    maximum_selected_cotangent_elements = 0
    maximum_callback_elements = 0
    stream_all_finite = (
        torch.ones((), dtype=torch.bool, device=preactivation.device)
        if not retain_moments
        else None
    )

    def emit_kink(*, flat_start: int, flat_stop: int, drive: Any, pre_units: Any) -> None:
        """In-layer kink rate per route of this chunk (diagnostic only)."""

        span = int(flat_stop) - int(flat_start)
        if activation_kind != "relu":
            rho = torch.zeros(span, dtype=torch.float64, device=drive.device)
        else:
            pre64 = pre_units.to(dtype=torch.float64)
            drive64 = drive.to(dtype=torch.float64)
            magnitude = pre64.abs()
            toward_wall = torch.clamp(drive64 * torch.sign(pre64), min=0.0)
            ratio = torch.where(
                magnitude > 0,
                toward_wall / torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude)),
                torch.full_like(magnitude, math.inf),
            )
            # Reduce over contexts (dim 0) and spatial units (dim 2).
            rho = ratio.amax(dim=2).amax(dim=0)
            if tuple(rho.shape) != (span,):
                raise RuntimeError("kink reduction lost the route chunk axis")
        kink_consumer(int(flat_start), int(flat_stop), rho.detach())

    def commit_chunk(
        *,
        flat_start: int,
        flat_stop: int,
        output_index: int | None,
        drive: Any,
        selected_cotangent: Any,
        values: Any,
        pre_units: Any = None,
    ) -> None:
        nonlocal chunk_count
        nonlocal callback_count
        nonlocal maximum_chunk
        nonlocal maximum_drive_elements
        nonlocal maximum_selected_cotangent_elements
        nonlocal maximum_callback_elements
        span = int(flat_stop) - int(flat_start)
        if signed_flat is not None and absolute_flat is not None:
            assert weights is not None
            aggregate_values = values.to(
                device=resolved_aggregate_device,
                dtype=aggregate_dtype,
            )
            weighted = aggregate_values * weights[:, None, None]
            signed_flat[:, flat_start:flat_stop] = weighted.sum(dim=0)
            absolute_flat[:, flat_start:flat_stop] = weighted.abs().sum(
                dim=0
            )
        if stream_all_finite is not None:
            stream_all_finite.logical_and_(torch.isfinite(values).all())
        chunk_count += 1
        maximum_chunk = max(maximum_chunk, span)
        maximum_drive_elements = max(
            maximum_drive_elements,
            int(drive.numel()),
        )
        maximum_selected_cotangent_elements = max(
            maximum_selected_cotangent_elements,
            int(selected_cotangent.numel()),
        )
        maximum_callback_elements = max(
            maximum_callback_elements,
            int(values.numel()),
        )
        if consumer is not None:
            consumer(
                V1GateDisclosureChunk(
                    route_block=block,
                    flat_start=flat_start,
                    flat_stop=flat_stop,
                    output_index=output_index,
                    values=values.detach(),
                    activation_kind=activation_kind,
                )
            )
            callback_count += 1
        if kink_consumer is not None:
            if pre_units is None:
                raise RuntimeError("kink consumer requires the chunk preactivation")
            emit_kink(
                flat_start=flat_start,
                flat_stop=flat_stop,
                drive=drive,
                pre_units=pre_units,
            )

    with torch.no_grad():
        if captured.reducer_kind == "linear_weight":
            if spatial != 1 or source.ndim != 2 or weight.ndim != 2:
                raise ValueError(
                    "linear gate disclosure requires matrix affine geometry"
                )
            for flat_start in range(0, routes, route_chunk_size):
                flat_stop = min(flat_start + route_chunk_size, routes)
                flat_indices = torch.arange(
                    flat_start,
                    flat_stop,
                    device=preactivation.device,
                )
                output_indices = torch.div(
                    flat_indices,
                    routes_per_output,
                    rounding_mode="floor",
                )
                input_indices = torch.remainder(
                    flat_indices,
                    routes_per_output,
                )
                drive = (
                    source[:, input_indices]
                    * weight[output_indices, input_indices][None, :]
                    * scale
                )[:, :, None]
                selected_pre = pre[:, output_indices, :]
                selected_cotangent = cotangent[:, :, output_indices, :]
                delta = (
                    torch.relu(selected_pre - drive)
                    - torch.relu(selected_pre)
                    if activation_kind == "relu"
                    else -drive
                )
                values = (
                    selected_cotangent * delta[None, :, :, :]
                ).sum(dim=-1).permute(1, 0, 2)
                commit_chunk(
                    flat_start=flat_start,
                    flat_stop=flat_stop,
                    output_index=None,
                    drive=drive,
                    selected_cotangent=selected_cotangent,
                    values=values,
                    pre_units=selected_pre,
                )
        else:
            for output_index in range(outputs):
                pre_output = pre[:, output_index]
                cotangent_output = cotangent[:, :, output_index]
                output_base = output_index * routes_per_output
                for local_start in range(
                    0,
                    routes_per_output,
                    route_chunk_size,
                ):
                    local_stop = min(
                        local_start + route_chunk_size,
                        routes_per_output,
                    )
                    span = local_stop - local_start
                    flat_start = output_base + local_start
                    flat_stop = output_base + local_stop

                    if captured.reducer_kind == "conv_channel_pair":
                        assert patches is not None
                        output_per_group = outputs // conv_groups
                        group_index = output_index // output_per_group
                        group_start = (
                            group_index * conv_input_per_group
                            if conv_full_logical_axis
                            else 0
                        )
                        connected_start = max(local_start, group_start)
                        connected_stop = min(
                            local_stop,
                            group_start + conv_input_per_group,
                        )
                        drive = torch.zeros(
                            (contexts, span, spatial),
                            dtype=preactivation.dtype,
                            device=preactivation.device,
                        )
                        if connected_start < connected_stop:
                            connected_positions = torch.arange(
                                connected_start - local_start,
                                connected_stop - local_start,
                                device=preactivation.device,
                            )
                            stored_inputs = torch.arange(
                                connected_start - group_start,
                                connected_stop - group_start,
                                device=preactivation.device,
                            )
                            selected_patches = patches[
                                :,
                                group_index,
                                stored_inputs,
                            ]
                            selected_weight = weight[
                                output_index,
                                stored_inputs,
                            ].reshape(int(stored_inputs.numel()), -1)
                            drive[:, connected_positions, :] = torch.einsum(
                                "cskl,sk->csl",
                                selected_patches,
                                selected_weight,
                            ).mul(scale)
                    else:
                        assert captured.reducer_kind == "conv_kernel_weight"
                        assert patches is not None
                        output_per_group = outputs // conv_groups
                        group_index = output_index // output_per_group
                        group_start = (
                            group_index * conv_input_per_group
                            if conv_full_logical_axis
                            else 0
                        )
                        connected_start = max(
                            local_start,
                            group_start * conv_kernel_elements,
                        )
                        connected_stop = min(
                            local_stop,
                            (group_start + conv_input_per_group)
                            * conv_kernel_elements,
                        )
                        drive = torch.zeros(
                            (contexts, span, spatial),
                            dtype=preactivation.dtype,
                            device=preactivation.device,
                        )
                        if connected_start < connected_stop:
                            connected_positions = torch.arange(
                                connected_start - local_start,
                                connected_stop - local_start,
                                device=preactivation.device,
                            )
                            connected_flat = torch.arange(
                                connected_start,
                                connected_stop,
                                device=preactivation.device,
                            )
                            stored_inputs = torch.div(
                                connected_flat,
                                conv_kernel_elements,
                                rounding_mode="floor",
                            ) - group_start
                            connected_taps = torch.remainder(
                                connected_flat,
                                conv_kernel_elements,
                            )
                            route_source = patches[
                                :,
                                group_index,
                                stored_inputs,
                                connected_taps,
                            ]
                            flat_weight = weight[output_index].reshape(
                                conv_input_per_group,
                                conv_kernel_elements,
                            )
                            route_weight = flat_weight[
                                stored_inputs,
                                connected_taps,
                            ]
                            drive[:, connected_positions, :] = (
                                route_source
                                .mul(route_weight[None, :, None])
                                .mul(scale)
                            )

                    if activation_kind == "relu":
                        delta = torch.relu(
                            pre_output[:, None, :] - drive
                        ) - torch.relu(
                            pre_output[:, None, :]
                        )
                    else:
                        delta = -drive
                    # Batched matrix multiplication contracts only the
                    # spatial axis.  The cotangent is not replicated over
                    # routes.
                    values = torch.bmm(
                        cotangent_output.permute(1, 0, 2),
                        delta.transpose(1, 2),
                    )
                    commit_chunk(
                        flat_start=flat_start,
                        flat_stop=flat_stop,
                        output_index=output_index,
                        drive=drive,
                        selected_cotangent=cotangent_output,
                        values=values,
                        pre_units=pre_output[:, None, :],
                    )

    if stream_all_finite is not None and not bool(stream_all_finite.item()):
        raise ValueError("streamed gate-disclosure values are nonfinite")

    receipt = V1GateDisclosureMemoryReceipt(
        context_count=contexts,
        query_count=queries,
        route_count=routes,
        output_count=outputs,
        spatial_extent=spatial,
        requested_route_chunk_size=route_chunk_size,
        maximum_observed_route_chunk_size=maximum_chunk,
        chunk_count=chunk_count,
        callback_chunk_count=callback_count,
        source_patch_elements=source_patch_elements,
        maximum_route_drive_elements=maximum_drive_elements,
        maximum_selected_cotangent_elements=(
            maximum_selected_cotangent_elements
        ),
        maximum_callback_elements=maximum_callback_elements,
        aggregate_elements=(2 * queries * routes if retain_moments else 0),
        theoretical_full_context_query_route_spatial_elements=(
            contexts * queries * routes * spatial
        ),
        per_route_moments_retained=retain_moments,
    )
    if not retain_moments:
        return V1GateDisclosureStreamResult(
            route_block=block,
            context_count=contexts,
            query_count=queries,
            activation_kind=activation_kind,
            memory_receipt=receipt,
        )
    assert signed_flat is not None
    assert absolute_flat is not None
    assert total_weight is not None
    return V1GateDisclosureMoments(
        route_block=block,
        signed_weighted_sum=signed_flat.reshape(
            queries,
            *block.shape,
        ).detach(),
        absolute_weighted_sum=absolute_flat.reshape(
            queries,
            *block.shape,
        ).detach(),
        total_weight=total_weight,
        context_count=contexts,
        query_count=queries,
        activation_kind=activation_kind,
        memory_receipt=receipt,
    )


__all__ = [
    "ActivationKind",
    "GATE_DISCLOSURE_EQUATION_ID",
    "GATE_DISCLOSURE_ROLE",
    "GATE_DISCLOSURE_SCHEMA",
    "V1GateDisclosureChunk",
    "V1GateDisclosureConsumer",
    "V1GateDisclosureMemoryReceipt",
    "V1GateDisclosureMoments",
    "V1GateDisclosureStreamResult",
    "compute_frozen_gate_removal_disclosure",
]
