"""Architecture-neutral Torch implementations of canonical backend protocols."""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

# IRG_WALK_REDUCE_OVERHEAD (default OFF): reduce-overhead (CUDA-graph tree)
# compilation of the structural walk runners.  ``perf_gates`` is a pure
# stdlib module (no package-internal imports), so this import can never
# cycle back into the backends package.
from mechanistic_systems.perf_gates import (
    walk_reduce_overhead_active,
    walk_reduce_overhead_requested,
    walk_phase_armed,
)

from .calibration_runtime import RuntimeExecutionProfile
from .contracts import BackendProfile
from .cuda_graphs import initialize_cuda_graph_state
from .profiling import PhaseProfiler
from .torch_masks import HostGateIdentity, tensor_mutation_identity, resident_lane_stack
from mechanistic_systems.measurement.gpu_runtime import process_device_memory_budget_bytes

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - import smoke without Torch.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("Torch backends require PyTorch")


def _walk_forward_active() -> bool:
    """The current look selects graphs; the older phase flag remains valid."""
    from mechanistic_systems.inference.device_bce import DEVICE_LOOK_ACTIVE

    return bool(DEVICE_LOOK_ACTIVE.get() or walk_phase_armed())

# FORWARD prefix-reuse is a default-on runtime restructuring of the fused
# candidate forward.  The split preserves the complete graph and every tensor
# coordinate. Production uses the prepared execution directly; numerical
# comparisons belong to tests.
_FORWARD_PREFIX_REUSE_ENV = "CIRCUIT_FORWARD_PREFIX_REUSE"
_PREFIX_REUSE_KILL_ENV = "IRG_PREFIX_REUSE"
_TRUE_ENV_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_ENV_TOKENS = frozenset({"", "0", "false", "no", "off"})


def _forward_prefix_reuse_from_env(environ: Mapping[str, str] | None = None) -> bool:
    """Prefix reuse is ON BY DEFAULT.

    The mechanism computes the same conceptual forward; rounding
    differences are telemetry, not a gate.  ``IRG_PREFIX_REUSE=0`` is the
    kill switch (takes precedence); an explicit
    ``CIRCUIT_FORWARD_PREFIX_REUSE`` boolean is still honored for
    profiles that set it.
    """

    values = os.environ if environ is None else environ
    kill = str(values.get(_PREFIX_REUSE_KILL_ENV, "")).strip().lower()
    if kill:
        if kill in _TRUE_ENV_TOKENS:
            return True
        if kill in _FALSE_ENV_TOKENS:
            return False
        raise ValueError(
            f"{_PREFIX_REUSE_KILL_ENV} must be a boolean environment value"
        )
    token = str(values.get(_FORWARD_PREFIX_REUSE_ENV, "")).strip().lower()
    if not token:
        return True
    if token in _TRUE_ENV_TOKENS:
        return True
    if token in _FALSE_ENV_TOKENS:
        return False
    raise ValueError(
        f"{_FORWARD_PREFIX_REUSE_ENV} must be a boolean environment value"
    )


def _lanes_share_column(column: Any) -> bool:
    """True when every lane row of a stacked [K, ...] gate buffer is identical."""

    return bool(torch.equal(column, column[:1].expand_as(column)))


def _output_tree_unsqueeze_lane(value: Any) -> Any:
    """Add a leading K=1 axis to every tensor leaf in an output pytree."""

    if torch.is_tensor(value):
        return value.unsqueeze(0)
    if isinstance(value, Mapping):
        return type(value)(
            (key, _output_tree_unsqueeze_lane(item)) for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(_output_tree_unsqueeze_lane(item) for item in value)
    if isinstance(value, list):
        return [_output_tree_unsqueeze_lane(item) for item in value]
    raise TypeError(
        f"prepared execution returned unsupported output leaf {type(value)!r}"
    )


def _output_tree_stack_lanes(values: Sequence[Any]) -> Any:
    """Stack complete per-lane output pytrees without dropping any leaf."""

    if not values:
        raise ValueError("cannot stack an empty lane-output sequence")
    first = values[0]
    if torch.is_tensor(first):
        if not all(torch.is_tensor(value) for value in values):
            raise TypeError("lane outputs have inconsistent tensor structure")
        return torch.stack(tuple(values), dim=0)
    if isinstance(first, Mapping):
        keys = tuple(first)
        if not all(isinstance(value, Mapping) and tuple(value) == keys for value in values):
            raise TypeError("lane outputs have inconsistent mapping structure")
        return type(first)(
            (
                key,
                _output_tree_stack_lanes(tuple(value[key] for value in values)),
            )
            for key in keys
        )
    if isinstance(first, tuple):
        if not all(isinstance(value, tuple) and len(value) == len(first) for value in values):
            raise TypeError("lane outputs have inconsistent tuple structure")
        return tuple(
            _output_tree_stack_lanes(tuple(value[index] for value in values))
            for index in range(len(first))
        )
    if isinstance(first, list):
        if not all(isinstance(value, list) and len(value) == len(first) for value in values):
            raise TypeError("lane outputs have inconsistent list structure")
        return [
            _output_tree_stack_lanes(tuple(value[index] for value in values))
            for index in range(len(first))
        ]
    raise TypeError(
        f"prepared execution returned unsupported output leaf {type(first)!r}"
    )


def _fetch(source: Any, start: int, end: int, *, device: Any) -> Any:
    getter = getattr(source, "get", None)
    if callable(getter):
        return getter(start, end, device=device, dtype=torch.float32)
    return source[start:end].to(device=device, dtype=torch.float32, non_blocking=True)


def _batch_size(source: Any, fallback: int) -> int:
    return int(getattr(source, "batch_size", fallback))


def _frontier_abs(mode: str) -> bool:
    """True when the absolute head is |E[aG]| rather than E[|aG|].

    The token sets live in ``core.numerics`` so the environment resolver and
    this consumer cannot drift apart: a mode the resolver accepts but this
    rejects would fail deep inside a reducer, mid-census.
    """

    from mechanistic_systems.core.numerics import (
        EXACT_ABS_AGGREGATION_MODES,
        FRONTIER_ABS_AGGREGATION_MODES,
    )

    token = str(mode).strip().lower()
    if token in EXACT_ABS_AGGREGATION_MODES:
        return False
    if token in FRONTIER_ABS_AGGREGATION_MODES:
        return True
    raise ValueError(f"unsupported absolute aggregation mode: {mode!r}")


def _positive_scale(scale: Any, device: Any) -> tuple[Any, Any]:
    value = float(scale.detach().cpu().item()) if torch.is_tensor(scale) else float(scale)
    if abs(value) < 1e-12:
        value = 1.0
    return (
        torch.as_tensor(value, dtype=torch.float32, device=device),
        torch.as_tensor(abs(value), dtype=torch.float32, device=device),
    )


def _linear_block_size(batch: int, inputs: int, outputs: int) -> int:
    target_mb = max(1, int(os.getenv("MECHANISTIC_LINEAR_REDUCER_TARGET_MB", "512")))
    max_samples = max(1, int(os.getenv("MECHANISTIC_LINEAR_REDUCER_MAX_SAMPLES", "4096")))
    bytes_per_sample = max(1, (inputs + outputs) * torch.float32.itemsize * 3)
    return max(1, min(batch, max_samples, target_mb * 1024 * 1024 // bytes_per_sample))


def linear_route_measurement_target_batched(
    inputs_source: Any,
    target_gradients: Any,
    *,
    weights: Any,
    scale: Any,
    device: Any,
    abs_aggregation_mode: str = "exact_abs",
) -> tuple[Any, Any]:
    """Exact Linear route-gate reduction for a query tile."""

    _require_torch()
    gradients = target_gradients.to(device=device, dtype=torch.float32)
    weight = weights.to(device=device, dtype=torch.float32)
    if gradients.ndim != 3 or weight.ndim != 2:
        raise ValueError("Linear reduction expects gradients [Q,B,O] and weights [O,I]")
    query_count, batch, outputs = gradients.shape
    if int(weight.shape[0]) != int(outputs) or _batch_size(inputs_source, batch) != int(batch):
        raise ValueError("Linear route reduction axes disagree")
    inputs = int(weight.shape[1])
    signed = torch.zeros((query_count, outputs, inputs), device=device, dtype=torch.float32)
    absolute = torch.zeros_like(signed)
    frontier_abs = _frontier_abs(abs_aggregation_mode)
    scale_value, absolute_scale = _positive_scale(scale, device)
    for start in range(0, int(batch), _linear_block_size(int(batch), inputs, outputs)):
        end = min(int(batch), start + _linear_block_size(int(batch), inputs, outputs))
        source = _fetch(inputs_source, start, end, device=device)
        if source.ndim > 2:
            source = source.reshape(source.shape[0], -1)
        contribution = torch.einsum(
            "qbo,bi->qoi", gradients[:, start:end] / scale_value, source
        ) * weight.unsqueeze(0)
        signed.add_(contribution)
        if not frontier_abs:
            exact = torch.einsum(
                "qbo,bi->qoi",
                gradients[:, start:end].abs() / absolute_scale,
                source.abs(),
            ) * weight.abs().unsqueeze(0)
            absolute.add_(exact)
    return signed, signed.abs() if frontier_abs else absolute


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"expected a scalar or pair, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _conv_direct_block_size(
    *,
    batch: int,
    queries: int,
    input_channels: int,
    outputs: int,
    inputs_per_group: int,
    kernel_elements: int,
    spatial_locations: int,
) -> int:
    """Bound the two direct-contraction intermediates under a runtime budget."""

    target_mb = max(1, int(os.getenv("MECHANISTIC_CONV_DIRECT_TARGET_MB", "768")))
    max_samples = max(1, int(os.getenv("MECHANISTIC_CONV_DIRECT_MAX_SAMPLES", "16")))
    patch_values = input_channels * kernel_elements * spatial_locations
    correlation_values = queries * outputs * inputs_per_group * kernel_elements
    route_values = queries * outputs * inputs_per_group
    bytes_per_sample = max(
        1,
        (patch_values + correlation_values + route_values) * torch.float32.itemsize,
    )
    return max(
        1,
        min(batch, max_samples, target_mb * 1024 * 1024 // bytes_per_sample),
    )


def conv_channel_route_measurement_direct_target_batched(
    inputs_source: Any,
    target_gradients: Any,
    *,
    weights: Any,
    scale: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: int,
    device: Any,
    abs_aggregation_mode: str = "exact_abs",
) -> tuple[Any, Any]:
    """Contract Conv input patches, weights, and VJPs without pre-add materialization.

    The reduction preserves the accepted route-gate-v1 order: kernel and
    spatial coordinates are summed within each context, then the signed and
    absolute context moments are accumulated. Runtime blocking therefore does
    not change the scientific estimand.
    """

    _require_torch()
    gradients = target_gradients.to(device=device, dtype=torch.float32)
    weight = weights.to(device=device, dtype=torch.float32)
    if gradients.ndim != 5 or weight.ndim != 4:
        raise ValueError("Conv reduction expects gradients [Q,B,O,H,W] and Conv2d weights")
    queries, batch, outputs, height, width = map(int, gradients.shape)
    groups = int(groups)
    if groups <= 0 or outputs % groups:
        raise ValueError("Conv outputs must be divisible by groups")
    source_batch = _batch_size(inputs_source, batch)
    if int(weight.shape[0]) != outputs or source_batch != batch:
        raise ValueError("Conv route reduction axes disagree")
    inputs_per_group = int(weight.shape[1])
    outputs_per_group = outputs // groups
    kernel_h, kernel_w = map(int, weight.shape[-2:])
    kernel_elements = kernel_h * kernel_w
    stride_pair = _pair(stride)
    padding_pair = _pair(padding)
    dilation_pair = _pair(dilation)
    frontier_abs = _frontier_abs(abs_aggregation_mode)
    scale_value, _absolute_scale = _positive_scale(scale, device)
    signed = torch.zeros(
        (queries, outputs, inputs_per_group), device=device, dtype=torch.float32
    )
    absolute = torch.zeros_like(signed)

    # Input channels are inferred from the grouped route universe. This also
    # rejects a replay source whose channel layout disagrees with the weights.
    input_channels = groups * inputs_per_group
    block = _conv_direct_block_size(
        batch=batch,
        queries=queries,
        input_channels=input_channels,
        outputs=outputs,
        inputs_per_group=inputs_per_group,
        kernel_elements=kernel_elements,
        spatial_locations=height * width,
    )
    grouped_weight = weight.reshape(
        groups, outputs_per_group, inputs_per_group, kernel_elements
    )
    for start in range(0, batch, block):
        end = min(batch, start + block)
        source = _fetch(inputs_source, start, end, device=device)
        if source.ndim != 4 or int(source.shape[1]) != input_channels:
            raise ValueError("Conv replay inputs disagree with grouped route coordinates")
        chunk = end - start
        if (
            kernel_elements == 1
            and stride_pair == (1, 1)
            and padding_pair == (0, 0)
            and dilation_pair == (1, 1)
            and tuple(source.shape[-2:]) == (height, width)
        ):
            patches = source.reshape(chunk, groups, inputs_per_group, 1, height * width)
        else:
            patches = F.unfold(
                source,
                kernel_size=(kernel_h, kernel_w),
                dilation=dilation_pair,
                padding=padding_pair,
                stride=stride_pair,
            ).reshape(
                chunk,
                groups,
                inputs_per_group,
                kernel_elements,
                -1,
            )
        if int(patches.shape[-1]) != height * width:
            raise ValueError("Conv input patches disagree with gradient spatial coordinates")
        gradient_block = (gradients[:, start:end] / scale_value).reshape(
            queries, chunk, groups, outputs_per_group, height * width
        )
        # [Q,B,G,O,L] @ [1,B,G,L,I*K] -> [Q,B,G,O,I*K].
        correlation = torch.matmul(
            gradient_block,
            patches.reshape(
                chunk, groups, inputs_per_group * kernel_elements, height * width
            )
            .transpose(-1, -2)
            .unsqueeze(0),
        )
        per_context = (
            correlation.reshape(
                queries,
                chunk,
                groups,
                outputs_per_group,
                inputs_per_group,
                kernel_elements,
            )
            * grouped_weight.reshape(
                1, 1, groups, outputs_per_group, inputs_per_group, kernel_elements
            )
        ).sum(dim=-1)
        per_context = per_context.reshape(queries, chunk, outputs, inputs_per_group)
        signed.add_(per_context.sum(dim=1))
        if not frontier_abs:
            absolute.add_(per_context.abs().sum(dim=1))
    return signed, signed.abs() if frontier_abs else absolute


def _conv_valid_tap_slices(
    *,
    input_size: int,
    output_size: int,
    kernel_index: int,
    stride: int,
    dilation: int,
    padding: int,
) -> tuple[slice, slice] | None:
    """Map one unpadded input tap to its valid output-coordinate slice."""

    if (
        input_size <= 0
        or output_size <= 0
        or kernel_index < 0
        or stride <= 0
        or dilation <= 0
        or padding < 0
    ):
        raise ValueError("Conv tap geometry is invalid")
    lower_numerator = padding - kernel_index * dilation
    output_start = max(0, -((-lower_numerator) // stride))
    upper_numerator = (
        input_size - 1 + padding - kernel_index * dilation
    )
    output_stop = min(output_size, upper_numerator // stride + 1)
    if output_start >= output_stop:
        return None
    input_start = output_start * stride - padding + kernel_index * dilation
    input_stop = input_start + (output_stop - output_start - 1) * stride + 1
    return (
        slice(input_start, input_stop, stride),
        slice(output_start, output_stop),
    )


def census_conv_local_source_factor(inputs, weight, *, output_shape, stride, padding, dilation):
    """Compact query-independent input-tap sums for local destination rows.

    Use the same valid tap views as the streamed Conv reducers. There is no
    output-channel or route axis in this retained factor.
    """
    source = inputs.detach().to(torch.float32)
    kernel_h, kernel_w = map(int, weight.shape[-2:])
    height, width = map(int, output_shape[-2:])
    stride, padding, dilation = _pair(stride), _pair(padding), _pair(dilation)
    result = torch.zeros((*source.shape[:2], kernel_h, kernel_w),
                         device=source.device, dtype=torch.float32)
    for row in range(kernel_h):
        h = _conv_valid_tap_slices(input_size=int(source.shape[-2]), output_size=height,
            kernel_index=row, stride=stride[0], dilation=dilation[0], padding=padding[0])
        if h is None:
            continue
        for column in range(kernel_w):
            w = _conv_valid_tap_slices(input_size=int(source.shape[-1]), output_size=width,
                kernel_index=column, stride=stride[1], dilation=dilation[1], padding=padding[1])
            if w is not None:
                result[:, :, row, column] = source[:, :, h[0], w[0]].sum(dim=(-1, -2))
    return result


def _validate_census_local_factor(factor, *, batch, channels, kernel_h, kernel_w, device):
    if factor is not None and (
        tuple(factor.shape) != (batch, channels, kernel_h, kernel_w)
        or factor.dtype != torch.float32 or factor.device != torch.device(device)
    ):
        raise ValueError("census local source factor differs from its Conv capture")


def _stream_conv_channel_context_rows(
    inputs_source: Any,
    gradients: Any,
    *,
    signed: Any,
    absolute: Any,
    grouped_weight: Any,
    scale_value: Any,
    stride_pair: tuple[int, int],
    padding_pair: tuple[int, int],
    dilation_pair: tuple[int, int],
    groups: int,
    outputs_per_group: int,
    inputs_per_group: int,
    kernel_h: int,
    kernel_w: int,
    height: int,
    width: int,
    batch: int,
    queries: int,
    outputs: int,
    block: int,
    frontier_abs: bool,
    per_context_consumer: Any,
    local_source_factor: Any | None = None,
) -> tuple[Any, Any, None, None]:
    """Emit bounded whole-context channel-route chunks without a full row bank."""

    from packages.salience.ops.target_batched import _census_output_row_chunk

    input_channels = groups * inputs_per_group
    row_chunk = _census_output_row_chunk(
        batch=batch,
        targets=queries,
        row_width=inputs_per_group,
        out_rows=outputs,
    )
    for out_start in range(0, outputs, row_chunk):
        out_stop = min(outputs, out_start + row_chunk)
        output_chunk = out_stop - out_start
        global_chunk = torch.zeros(
            (batch, queries, output_chunk, inputs_per_group),
            device=gradients.device,
            dtype=torch.float32,
        )
        local_chunk = torch.zeros(
            (batch, output_chunk, inputs_per_group),
            device=gradients.device,
            dtype=torch.float32,
        )
        for batch_start in range(0, batch, block):
            batch_stop = min(batch, batch_start + block)
            source = _fetch(
                inputs_source,
                batch_start,
                batch_stop,
                device=gradients.device,
            )
            if source.ndim != 4 or int(source.shape[1]) != input_channels:
                raise ValueError(
                    "Conv replay inputs disagree with grouped route coordinates"
                )
            context_chunk = batch_stop - batch_start
            input_height, input_width = map(int, source.shape[-2:])
            for group_index in range(groups):
                group_start = group_index * outputs_per_group
                group_stop = group_start + outputs_per_group
                segment_start = max(out_start, group_start)
                segment_stop = min(out_stop, group_stop)
                if segment_start >= segment_stop:
                    continue
                local_start = segment_start - group_start
                local_stop = segment_stop - group_start
                chunk_start = segment_start - out_start
                chunk_stop = segment_stop - out_start
                gradient_segment = gradients[
                    :,
                    batch_start:batch_stop,
                    segment_start:segment_stop,
                ]
                weight_segment = grouped_weight[
                    group_index, local_start:local_stop
                ].reshape(
                    segment_stop - segment_start,
                    inputs_per_group,
                    kernel_h,
                    kernel_w,
                )
                per_context = torch.zeros(
                    queries,
                    context_chunk,
                    segment_stop - segment_start,
                    inputs_per_group,
                    device=gradients.device,
                    dtype=torch.float32,
                )
                local = torch.zeros(
                    context_chunk,
                    segment_stop - segment_start,
                    inputs_per_group,
                    device=gradients.device,
                    dtype=torch.float32,
                )
                source_channel_start = group_index * inputs_per_group
                source_channel_stop = source_channel_start + inputs_per_group
                source_group = source[
                    :,
                    source_channel_start:source_channel_stop,
                ]
                for kernel_row in range(kernel_h):
                    height_slices = _conv_valid_tap_slices(
                        input_size=input_height,
                        output_size=height,
                        kernel_index=kernel_row,
                        stride=stride_pair[0],
                        dilation=dilation_pair[0],
                        padding=padding_pair[0],
                    )
                    if height_slices is None:
                        continue
                    input_rows, output_rows = height_slices
                    for kernel_column in range(kernel_w):
                        width_slices = _conv_valid_tap_slices(
                            input_size=input_width,
                            output_size=width,
                            kernel_index=kernel_column,
                            stride=stride_pair[1],
                            dilation=dilation_pair[1],
                            padding=padding_pair[1],
                        )
                        if width_slices is None:
                            continue
                        input_columns, output_columns = width_slices
                        source_tap = source_group[
                            :, :, input_rows, input_columns
                        ]
                        gradient_tap = (
                            gradient_segment[
                                :, :, :, output_rows, output_columns
                            ]
                            / scale_value
                        )
                        tap_weight = weight_segment[
                            :, :, kernel_row, kernel_column
                        ]
                        per_context.add_(
                            torch.einsum(
                                "bihw,qbohw->qboi",
                                source_tap,
                                gradient_tap,
                            )
                            * tap_weight.reshape(
                                1,
                                1,
                                segment_stop - segment_start,
                                inputs_per_group,
                            )
                        )
                        local.add_(
                            (source_tap.sum(dim=(-1, -2)) if local_source_factor is None else
                             local_source_factor[batch_start:batch_stop,
                                 source_channel_start:source_channel_stop, kernel_row, kernel_column]).unsqueeze(1)
                            * tap_weight.unsqueeze(0)
                        )
                global_chunk[
                    batch_start:batch_stop, :, chunk_start:chunk_stop
                ].copy_(per_context.transpose(0, 1))
                signed[:, segment_start:segment_stop].add_(
                    per_context.sum(dim=1)
                )
                if not frontier_abs:
                    absolute[:, segment_start:segment_stop].add_(
                        per_context.abs().sum(dim=1)
                    )
                local_chunk[
                    batch_start:batch_stop, chunk_start:chunk_stop
                ].copy_(local * scale_value)
        per_context_consumer(
            out_start * inputs_per_group,
            global_chunk,
            local_chunk,
        )
        del global_chunk, local_chunk
    return signed, signed.abs() if frontier_abs else absolute, None, None


def conv_channel_route_measurement_direct_with_context_target_batched(
    inputs_source: Any,
    target_gradients: Any,
    *,
    weights: Any,
    scale: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: int,
    device: Any,
    abs_aggregation_mode: str = "exact_abs",
    retain_local_destination: bool = True,
    local_regime_tee: Any | None = None,
    per_context_consumer: Any | None = None,
    local_source_factor: Any | None = None,
) -> tuple[Any, ...]:
    """Direct Conv reduction with fused global and local context rows.

    The input-patch tensor and grouped weights are shared by both heads.
    ``global_rows`` is ``[B,Q,O,I]`` and ``local_rows`` is ``[B,O,I]``.
    The latter is the signed spatial sum of the contribution at the immediate
    destination channel, before query cotangent contraction.

    ``per_context_consumer`` materializes and emits one bounded
    whole-context output-channel chunk at a time. The complete
    ``[B,Q,O,I]`` and ``[B,O,I]`` row banks are never allocated.
    """

    _require_torch()
    if per_context_consumer is not None:
        if not bool(
            getattr(per_context_consumer, "wants_conv_rows", True)
        ):
            # Capability negotiation: an in-backward capture sink
            # declines conv emissions — those blocks keep retained-tile
            # semantics and are tile-fed by the provider.
            per_context_consumer = None
        elif not retain_local_destination:
            raise ValueError(
                "census-backward streaming requires local-destination "
                "retention semantics"
            )
        elif local_regime_tee is not None:
            raise ValueError(
                "census-backward streaming cannot share the unchunked local-regime tee"
            )
    gradients = target_gradients.to(device=device, dtype=torch.float32)
    weight = weights.to(device=device, dtype=torch.float32)
    if gradients.ndim != 5 or weight.ndim != 4:
        raise ValueError(
            "Conv reduction expects gradients [Q,B,O,H,W] and Conv2d weights"
        )
    queries, batch, outputs, height, width = map(int, gradients.shape)
    groups = int(groups)
    if groups <= 0 or outputs % groups:
        raise ValueError("Conv outputs must be divisible by groups")
    if int(weight.shape[0]) != outputs or _batch_size(inputs_source, batch) != batch:
        raise ValueError("Conv route reduction axes disagree")
    inputs_per_group = int(weight.shape[1])
    outputs_per_group = outputs // groups
    kernel_h, kernel_w = map(int, weight.shape[-2:])
    kernel_elements = kernel_h * kernel_w
    stride_pair = _pair(stride)
    padding_pair = _pair(padding)
    dilation_pair = _pair(dilation)
    frontier_abs = _frontier_abs(abs_aggregation_mode)
    scale_value, _absolute_scale = _positive_scale(scale, device)
    signed = torch.zeros(
        (queries, outputs, inputs_per_group), device=device, dtype=torch.float32
    )
    absolute = torch.zeros_like(signed)
    input_channels = groups * inputs_per_group
    _validate_census_local_factor(local_source_factor, batch=batch, channels=input_channels,
        kernel_h=kernel_h, kernel_w=kernel_w, device=device)
    block = _conv_direct_block_size(
        batch=batch,
        queries=queries,
        input_channels=input_channels,
        outputs=outputs,
        inputs_per_group=inputs_per_group,
        kernel_elements=kernel_elements,
        spatial_locations=height * width,
    )
    grouped_weight = weight.reshape(
        groups, outputs_per_group, inputs_per_group, kernel_elements
    )
    if per_context_consumer is not None:
        return _stream_conv_channel_context_rows(
            inputs_source,
            gradients,
            signed=signed,
            absolute=absolute,
            grouped_weight=grouped_weight,
            scale_value=scale_value,
            stride_pair=stride_pair,
            padding_pair=padding_pair,
            dilation_pair=dilation_pair,
            groups=groups,
            outputs_per_group=outputs_per_group,
            inputs_per_group=inputs_per_group,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            height=height,
            width=width,
            batch=batch,
            queries=queries,
            outputs=outputs,
            block=block,
            frontier_abs=frontier_abs,
            per_context_consumer=per_context_consumer,
            local_source_factor=local_source_factor,
        )
    global_rows = torch.empty(
        (batch, queries, outputs, inputs_per_group),
        device=device,
        dtype=torch.float32,
    )
    local_rows = (
        torch.empty(
            (batch, outputs, inputs_per_group),
            device=device,
            dtype=torch.float32,
        )
        if retain_local_destination
        else None
    )
    for start in range(0, batch, block):
        end = min(batch, start + block)
        source = _fetch(inputs_source, start, end, device=device)
        if source.ndim != 4 or int(source.shape[1]) != input_channels:
            raise ValueError(
                "Conv replay inputs disagree with grouped route coordinates"
            )
        chunk = end - start
        if (
            kernel_elements == 1
            and stride_pair == (1, 1)
            and padding_pair == (0, 0)
            and dilation_pair == (1, 1)
            and tuple(source.shape[-2:]) == (height, width)
        ):
            patches = source.reshape(
                chunk, groups, inputs_per_group, 1, height * width
            )
        else:
            patches = F.unfold(
                source,
                kernel_size=(kernel_h, kernel_w),
                dilation=dilation_pair,
                padding=padding_pair,
                stride=stride_pair,
            ).reshape(
                chunk,
                groups,
                inputs_per_group,
                kernel_elements,
                -1,
            )
        if int(patches.shape[-1]) != height * width:
            raise ValueError(
                "Conv input patches disagree with gradient spatial coordinates"
            )
        gradient_block = (gradients[:, start:end] / scale_value).reshape(
            queries, chunk, groups, outputs_per_group, height * width
        )
        correlation = torch.matmul(
            gradient_block,
            patches.reshape(
                chunk,
                groups,
                inputs_per_group * kernel_elements,
                height * width,
            )
            .transpose(-1, -2)
            .unsqueeze(0),
        )
        per_context = (
            correlation.reshape(
                queries,
                chunk,
                groups,
                outputs_per_group,
                inputs_per_group,
                kernel_elements,
            )
            * grouped_weight.reshape(
                1,
                1,
                groups,
                outputs_per_group,
                inputs_per_group,
                kernel_elements,
            )
        ).sum(dim=-1)
        per_context = per_context.reshape(
            queries, chunk, outputs, inputs_per_group
        )
        global_rows[start:end].copy_(per_context.transpose(0, 1))
        signed.add_(per_context.sum(dim=1))
        if not frontier_abs:
            absolute.add_(per_context.abs().sum(dim=1))
        # Signed sum over destination spatial sites of the exact route-resolved
        # pre-add contribution.  This contraction reuses ``patches`` and
        # ``grouped_weight`` and never constructs a second convolution.
        if local_rows is not None:
            local = (torch.einsum("bgikl,goik->bgoi", patches, grouped_weight)
                     if local_source_factor is None else torch.einsum(
                         "bgik,goik->bgoi", local_source_factor[start:end].reshape(
                             chunk, groups, inputs_per_group, kernel_elements), grouped_weight))
            local_rows[start:end].copy_(
                local.reshape(chunk, outputs, inputs_per_group) * scale_value
            )
        if local_regime_tee is not None:
            # Diagnostic-only branch: form the route-resolved spatial field
            # from the already resident patch and weight tensors.  The
            # canonical four heads above are unchanged.
            local_field = torch.einsum(
                "bgikl,goik->bgoil",
                patches,
                grouped_weight,
            ).reshape(
                chunk,
                outputs,
                inputs_per_group,
                height,
                width,
            )
            local_regime_tee.update(start, local_field)
    result = (
        signed,
        signed.abs() if frontier_abs else absolute,
        global_rows,
        local_rows,
    )
    if local_regime_tee is None:
        return result
    return (*result, local_regime_tee.finalize())


def conv_channel_route_measurement_target_batched(
    inputs_source: Any,
    target_gradients: Any,
    *,
    weights: Any,
    scale: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: int,
    device: Any,
    abs_aggregation_mode: str = "exact_abs",
) -> tuple[Any, Any]:
    """Reduce channel-pair effects through the bounded patch contraction.

    The spatial input/gradient contraction is followed by the weighted tap
    sum within each context. This avoids a virtual grouped convolution and
    its full spatial pre-add bank while preserving the route-gate estimand.
    """

    return conv_channel_route_measurement_direct_target_batched(
        inputs_source,
        target_gradients,
        weights=weights,
        scale=scale,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        device=device,
        abs_aggregation_mode=abs_aggregation_mode,
    )


def conv_channel_route_measurement_per_image(
    inputs_source: Any,
    target_gradients: Any,
    *,
    weights: Any,
    scale: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: int,
    device: Any,
) -> Any:
    """Per-image convolutional route contributions [B, Q, O, I]."""

    _signed, _absolute, rows, _local = (
        conv_channel_route_measurement_direct_with_context_target_batched(
            inputs_source,
            target_gradients,
            weights=weights,
            scale=scale,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            device=device,
            abs_aggregation_mode="frontier_aggregate_abs",
            retain_local_destination=False,
        )
    )
    return rows


def _conv_kernel_block_size(
    *,
    batch: int,
    queries: int,
    input_channels: int,
    outputs: int,
    inputs_per_group: int,
    kernel_elements: int,
    spatial_locations: int,
) -> int:
    """Bound the unfold patch buffer and per-tap rows under a runtime budget."""

    target_mb = max(
        1, int(os.getenv("MECHANISTIC_CONV_KERNEL_REDUCER_TARGET_MB", "768"))
    )
    max_samples = max(
        1, int(os.getenv("MECHANISTIC_CONV_KERNEL_REDUCER_MAX_SAMPLES", "16"))
    )
    patch_values = input_channels * kernel_elements * spatial_locations
    row_values = queries * outputs * inputs_per_group * kernel_elements
    bytes_per_sample = max(
        1, (patch_values + row_values) * torch.float32.itemsize
    )
    return max(
        1,
        min(batch, max_samples, target_mb * 1024 * 1024 // bytes_per_sample),
    )


def _conv_kernel_per_context(
    source: Any,
    gradient_block: Any,
    weight: Any,
    *,
    groups: int,
    stride: Any,
    padding: Any,
    dilation: Any,
) -> Any:
    """Per-context per-tap contributions ``[Q, b, O, I_pg, KH, KW]``.

    ``unfold`` materializes the padded input patches; the einsum contracts
    output spatial positions while RETAINING the kernel-tap axis; the tap's
    own weight coefficient multiplies each retained tap afterwards. This is
    the channel-pair patch contraction
    with its final kernel-tap summation removed: summing the result over
    ``(KH, KW)`` recovers the channel-pair per-context contribution.
    """

    outputs, inputs_per_group, kernel_h, kernel_w = (
        int(value) for value in weight.shape
    )
    groups = int(groups)
    if groups <= 0 or outputs % groups:
        raise ValueError("Conv outputs must be divisible by groups")
    outputs_per_group = outputs // groups
    kernel_elements = kernel_h * kernel_w
    queries = int(gradient_block.shape[0])
    chunk = int(source.shape[0])
    patches = F.unfold(
        source,
        kernel_size=(kernel_h, kernel_w),
        dilation=dilation,
        padding=padding,
        stride=stride,
    ).reshape(chunk, groups, inputs_per_group, kernel_elements, -1)
    locations = int(patches.shape[-1])
    if int(gradient_block.shape[-2] * gradient_block.shape[-1]) != locations:
        raise ValueError(
            "Conv kernel route gradient spatial geometry disagrees with "
            "its unfold patch geometry"
        )
    gradient = gradient_block.reshape(
        queries, chunk, groups, outputs_per_group, locations
    )
    correlation = torch.einsum("bgikl,qbgol->qbgoik", patches, gradient)
    per_context = correlation * weight.reshape(
        groups, outputs_per_group, inputs_per_group, kernel_elements
    ).unsqueeze(0).unsqueeze(0)
    return per_context.reshape(
        queries, chunk, outputs, inputs_per_group, kernel_h, kernel_w
    )


def conv_kernel_route_measurement_target_batched(
    inputs_source: Any,
    target_gradients: Any,
    *,
    weights: Any,
    scale: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: int,
    device: Any,
    abs_aggregation_mode: str = "exact_abs",
) -> tuple[Any, Any]:
    """Exact per-tap Conv kernel-weight route-gate reduction for a query tile."""

    _require_torch()
    gradients = target_gradients.to(device=device, dtype=torch.float32)
    weight = weights.to(device=device, dtype=torch.float32)
    if gradients.ndim != 5 or weight.ndim != 4:
        raise ValueError("Conv reduction expects gradients [Q,B,O,H,W] and Conv2d weights")
    queries, batch, outputs, height, width = (int(value) for value in gradients.shape)
    if int(weight.shape[0]) != outputs or _batch_size(inputs_source, batch) != batch:
        raise ValueError("Conv route reduction axes disagree")
    inputs_per_group, kernel_h, kernel_w = (int(value) for value in weight.shape[1:])
    signed = torch.zeros(
        (queries, outputs, inputs_per_group, kernel_h, kernel_w),
        device=device,
        dtype=torch.float32,
    )
    absolute = torch.zeros_like(signed)
    frontier_abs = _frontier_abs(abs_aggregation_mode)
    scale_value, _absolute_scale = _positive_scale(scale, device)
    block = _conv_kernel_block_size(
        batch=batch,
        queries=queries,
        input_channels=int(groups) * inputs_per_group,
        outputs=outputs,
        inputs_per_group=inputs_per_group,
        kernel_elements=kernel_h * kernel_w,
        spatial_locations=height * width,
    )
    for start in range(0, batch, block):
        end = min(batch, start + block)
        source = _fetch(inputs_source, start, end, device=device)
        per_context = _conv_kernel_per_context(
            source,
            gradients[:, start:end] / scale_value,
            weight,
            groups=int(groups),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        signed.add_(per_context.sum(dim=1))
        if not frontier_abs:
            absolute.add_(per_context.abs().sum(dim=1))
    return signed, signed.abs() if frontier_abs else absolute


def _stream_conv_kernel_context_rows(
    inputs_source: Any,
    gradients: Any,
    *,
    signed: Any,
    absolute: Any,
    grouped_weight: Any,
    scale_value: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: int,
    outputs_per_group: int,
    inputs_per_group: int,
    kernel_h: int,
    kernel_w: int,
    height: int,
    width: int,
    batch: int,
    queries: int,
    outputs: int,
    block: int,
    frontier_abs: bool,
    per_context_consumer: Any,
    profiler: Any | None,
    local_source_factor: Any | None = None,
) -> tuple[Any, Any, None, None]:
    """Emit bounded whole-context kernel-route chunks without a full row bank."""

    from packages.salience.ops.target_batched import _census_output_row_chunk

    kernel_elements = kernel_h * kernel_w
    input_channels = groups * inputs_per_group
    stride_pair = _pair(stride)
    padding_pair = _pair(padding)
    dilation_pair = _pair(dilation)
    row_chunk = _census_output_row_chunk(
        batch=batch,
        targets=queries,
        row_width=inputs_per_group * kernel_elements,
        out_rows=outputs,
    )
    for out_start in range(0, outputs, row_chunk):
        out_stop = min(outputs, out_start + row_chunk)
        output_chunk = out_stop - out_start
        global_chunk = torch.zeros(
            (
                batch,
                queries,
                output_chunk,
                inputs_per_group,
                kernel_h,
                kernel_w,
            ),
            device=gradients.device,
            dtype=torch.float32,
        )
        local_chunk = torch.zeros(
            (
                batch,
                output_chunk,
                inputs_per_group,
                kernel_h,
                kernel_w,
            ),
            device=gradients.device,
            dtype=torch.float32,
        )
        for batch_start in range(0, batch, block):
            batch_stop = min(batch, batch_start + block)
            with (
                profiler.phase("measurement.conv_kernel.streamed_source_fetch")
                if profiler is not None
                else nullcontext()
            ):
                source = _fetch(
                    inputs_source,
                    batch_start,
                    batch_stop,
                    device=gradients.device,
                )
                if source.ndim != 4 or int(source.shape[1]) != input_channels:
                    raise ValueError(
                        "Conv replay inputs disagree with grouped route coordinates"
                    )
                context_chunk = batch_stop - batch_start
                input_height, input_width = map(int, source.shape[-2:])
            with (
                profiler.phase("measurement.conv_kernel.streamed_tap_contract")
                if profiler is not None
                else nullcontext()
            ):
                for group_index in range(groups):
                    group_start = group_index * outputs_per_group
                    group_stop = group_start + outputs_per_group
                    segment_start = max(out_start, group_start)
                    segment_stop = min(out_stop, group_stop)
                    if segment_start >= segment_stop:
                        continue
                    local_start = segment_start - group_start
                    local_stop = segment_stop - group_start
                    chunk_start = segment_start - out_start
                    chunk_stop = segment_stop - out_start
                    gradient_segment = gradients[
                        :,
                        batch_start:batch_stop,
                        segment_start:segment_stop,
                    ]
                    weight_segment = grouped_weight[
                        group_index, local_start:local_stop
                    ].reshape(
                        segment_stop - segment_start,
                        inputs_per_group,
                        kernel_h,
                        kernel_w,
                    )
                    source_channel_start = group_index * inputs_per_group
                    source_channel_stop = source_channel_start + inputs_per_group
                    source_group = source[
                        :,
                        source_channel_start:source_channel_stop,
                    ]
                    for kernel_row in range(kernel_h):
                        height_slices = _conv_valid_tap_slices(
                            input_size=input_height,
                            output_size=height,
                            kernel_index=kernel_row,
                            stride=stride_pair[0],
                            dilation=dilation_pair[0],
                            padding=padding_pair[0],
                        )
                        if height_slices is None:
                            continue
                        input_rows, output_rows = height_slices
                        for kernel_column in range(kernel_w):
                            width_slices = _conv_valid_tap_slices(
                                input_size=input_width,
                                output_size=width,
                                kernel_index=kernel_column,
                                stride=stride_pair[1],
                                dilation=dilation_pair[1],
                                padding=padding_pair[1],
                            )
                            if width_slices is None:
                                continue
                            input_columns, output_columns = width_slices
                            source_tap = source_group[
                                :, :, input_rows, input_columns
                            ]
                            gradient_tap = (
                                gradient_segment[
                                    :, :, :, output_rows, output_columns
                                ]
                                / scale_value
                            )
                            tap_weight = weight_segment[
                                :, :, kernel_row, kernel_column
                            ]
                            per_context = (
                                torch.einsum(
                                    "bihw,qbohw->qboi",
                                    source_tap,
                                    gradient_tap,
                                )
                                * tap_weight.reshape(
                                    1,
                                    1,
                                    segment_stop - segment_start,
                                    inputs_per_group,
                                )
                            )
                            global_chunk[
                                batch_start:batch_stop,
                                :,
                                chunk_start:chunk_stop,
                                :,
                                kernel_row,
                                kernel_column,
                            ].copy_(per_context.transpose(0, 1))
                            signed[
                                :,
                                segment_start:segment_stop,
                                :,
                                kernel_row,
                                kernel_column,
                            ].add_(per_context.sum(dim=1))
                            if not frontier_abs:
                                absolute[
                                    :,
                                    segment_start:segment_stop,
                                    :,
                                    kernel_row,
                                    kernel_column,
                                ].add_(per_context.abs().sum(dim=1))
                            local = (
                                (source_tap.sum(dim=(-1, -2)) if local_source_factor is None else
                                 local_source_factor[batch_start:batch_stop,
                                     source_channel_start:source_channel_stop, kernel_row, kernel_column]).unsqueeze(1)
                                * tap_weight.unsqueeze(0)
                            )
                            local_chunk[
                                batch_start:batch_stop,
                                chunk_start:chunk_stop,
                                :,
                                kernel_row,
                                kernel_column,
                            ].copy_(local * scale_value)
        with (
            profiler.phase("measurement.conv_kernel.consumer_stream")
            if profiler is not None
            else nullcontext()
        ):
            per_context_consumer(
                out_start * inputs_per_group * kernel_elements,
                global_chunk,
                local_chunk,
            )
        del global_chunk, local_chunk
    return signed, signed.abs() if frontier_abs else absolute, None, None


def conv_kernel_route_measurement_with_context_target_batched(
    inputs_source: Any,
    target_gradients: Any,
    *,
    weights: Any,
    scale: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: int,
    device: Any,
    abs_aggregation_mode: str = "exact_abs",
    retain_local_destination: bool = True,
    per_context_consumer: Any | None = None,
    profiler: Any | None = None,
    local_source_factor: Any | None = None,
) -> tuple[Any, Any, Any | None, Any | None]:
    """Exact per-tap Conv kernel-weight reduction with shared context rows.

    This is the per-tap form of the channel-pair with-context reducer. Without
    a consumer, the unfolded input patch and per-tap contraction are built once
    per batch chunk and the complete context rows are returned. With an armed
    consumer, one bounded whole-context output-channel chunk is materialized
    and emitted at a time; complete row banks are never allocated.
    """

    _require_torch()
    if per_context_consumer is not None:
        if not bool(
            getattr(per_context_consumer, "wants_conv_rows", True)
        ):
            per_context_consumer = None
        elif not retain_local_destination:
            raise ValueError(
                "census-backward streaming requires local-destination "
                "retention semantics"
            )
    gradients = target_gradients.to(device=device, dtype=torch.float32)
    weight = weights.to(device=device, dtype=torch.float32)
    if gradients.ndim != 5 or weight.ndim != 4:
        raise ValueError(
            "Conv reduction expects gradients [Q,B,O,H,W] and Conv2d weights"
        )
    queries, batch, outputs, height, width = (
        int(value) for value in gradients.shape
    )
    groups = int(groups)
    if groups <= 0 or outputs % groups:
        raise ValueError("Conv outputs must be divisible by groups")
    if int(weight.shape[0]) != outputs or _batch_size(inputs_source, batch) != batch:
        raise ValueError("Conv route reduction axes disagree")
    inputs_per_group, kernel_h, kernel_w = (int(value) for value in weight.shape[1:])
    kernel_elements = kernel_h * kernel_w
    input_channels = groups * inputs_per_group
    outputs_per_group = outputs // groups
    signed = torch.zeros(
        (queries, outputs, inputs_per_group, kernel_h, kernel_w),
        device=device,
        dtype=torch.float32,
    )
    absolute = torch.zeros_like(signed)
    frontier_abs = _frontier_abs(abs_aggregation_mode)
    scale_value, _absolute_scale = _positive_scale(scale, device)
    block = _conv_kernel_block_size(
        batch=batch,
        queries=queries,
        input_channels=input_channels,
        outputs=outputs,
        inputs_per_group=inputs_per_group,
        kernel_elements=kernel_elements,
        spatial_locations=height * width,
    )
    grouped_weight = weight.reshape(
        groups, outputs_per_group, inputs_per_group, kernel_elements
    )
    _validate_census_local_factor(local_source_factor, batch=batch, channels=groups * inputs_per_group,
        kernel_h=kernel_h, kernel_w=kernel_w, device=device)
    if per_context_consumer is not None:
        return _stream_conv_kernel_context_rows(
            inputs_source,
            gradients,
            signed=signed,
            absolute=absolute,
            grouped_weight=grouped_weight,
            scale_value=scale_value,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            outputs_per_group=outputs_per_group,
            inputs_per_group=inputs_per_group,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            height=height,
            width=width,
            batch=batch,
            queries=queries,
            outputs=outputs,
            block=block,
            frontier_abs=frontier_abs,
            per_context_consumer=per_context_consumer,
            profiler=profiler,
            local_source_factor=local_source_factor,
        )
    global_rows = torch.empty(
        (batch, queries, outputs, inputs_per_group, kernel_h, kernel_w),
        device=device,
        dtype=torch.float32,
    )
    local_rows = (
        torch.empty(
            (batch, outputs, inputs_per_group, kernel_h, kernel_w),
            device=device,
            dtype=torch.float32,
        )
        if retain_local_destination
        else None
    )
    for start in range(0, batch, block):
        end = min(batch, start + block)
        with (
            profiler.phase("measurement.conv_kernel.shared_patch_extract")
            if profiler is not None
            else nullcontext()
        ):
            source = _fetch(inputs_source, start, end, device=device)
            if source.ndim != 4 or int(source.shape[1]) != input_channels:
                raise ValueError(
                    "Conv replay inputs disagree with grouped route coordinates"
                )
            chunk = end - start
            patches = F.unfold(
                source,
                kernel_size=(kernel_h, kernel_w),
                dilation=dilation,
                padding=padding,
                stride=stride,
            ).reshape(
                chunk,
                groups,
                inputs_per_group,
                kernel_elements,
                -1,
            )
            locations = int(patches.shape[-1])
            if int(height * width) != locations:
                raise ValueError(
                    "Conv kernel route gradient spatial geometry disagrees "
                    "with its unfold patch geometry"
                )
        with (
            profiler.phase("measurement.conv_kernel.fused_contract")
            if profiler is not None
            else nullcontext()
        ):
            gradient = (gradients[:, start:end] / scale_value).reshape(
                queries, chunk, groups, outputs_per_group, locations
            )
            correlation = torch.einsum("bgikl,qbgol->qbgoik", patches, gradient)
            per_context = correlation * grouped_weight.unsqueeze(0).unsqueeze(0)
            per_context = per_context.reshape(
                queries,
                chunk,
                outputs,
                inputs_per_group,
                kernel_h,
                kernel_w,
            )
            global_rows[start:end].copy_(per_context.transpose(0, 1))
            signed.add_(per_context.sum(dim=1))
            if not frontier_abs:
                absolute.add_(per_context.abs().sum(dim=1))
            if local_rows is not None:
                local = (
                    (patches.sum(dim=-1) if local_source_factor is None else
                     local_source_factor[start:end].reshape(
                         chunk, groups, inputs_per_group, kernel_elements)).unsqueeze(2)
                    * grouped_weight.unsqueeze(0)
                ).reshape(
                    chunk,
                    outputs,
                    inputs_per_group,
                    kernel_h,
                    kernel_w,
                )
                local_rows[start:end].copy_(local * scale_value)
        del patches, per_context
    return signed, signed.abs() if frontier_abs else absolute, global_rows, local_rows


def conv_kernel_route_measurement_per_image(
    inputs_source: Any,
    target_gradients: Any,
    *,
    weights: Any,
    scale: Any,
    stride: Any,
    padding: Any,
    dilation: Any,
    groups: int,
    device: Any,
) -> Any:
    """Per-image convolutional route contributions [B, Q, O, I, KH, KW]."""

    _require_torch()
    gradients = target_gradients.to(device=device, dtype=torch.float32)
    weight = weights.to(device=device, dtype=torch.float32)
    if gradients.ndim != 5 or weight.ndim != 4:
        raise ValueError("Conv reduction expects gradients [Q,B,O,H,W] and Conv2d weights")
    queries, batch, outputs, height, width = (int(value) for value in gradients.shape)
    if int(weight.shape[0]) != outputs or _batch_size(inputs_source, batch) != batch:
        raise ValueError("Conv route reduction axes disagree")
    inputs_per_group, kernel_h, kernel_w = (int(value) for value in weight.shape[1:])
    per_image = torch.zeros(
        (batch, queries, outputs, inputs_per_group, kernel_h, kernel_w),
        device=device,
        dtype=torch.float32,
    )
    scale_value, _absolute_scale = _positive_scale(scale, device)
    block = _conv_kernel_block_size(
        batch=batch,
        queries=queries,
        input_channels=int(groups) * inputs_per_group,
        outputs=outputs,
        inputs_per_group=inputs_per_group,
        kernel_elements=kernel_h * kernel_w,
        spatial_locations=height * width,
    )
    for start in range(0, batch, block):
        end = min(batch, start + block)
        source = _fetch(inputs_source, start, end, device=device)
        per_context = _conv_kernel_per_context(
            source,
            gradients[:, start:end] / scale_value,
            weight,
            groups=int(groups),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        per_image[start:end] = per_context.transpose(0, 1)
    return per_image


class TorchRouteReducerBackend:
    backend_id = "mechanistic_systems.torch_route_reducer.v1"
    implementation_sources = {
        "linear_weight": "packages.salience.ops.target_batched.linear_route_measurement_target_batched",
        "conv_channel_pair": "packages.salience.ops.target_batched.conv_channel_route_measurement_target_batched",
        "residual_stream_gate": "mechanistic_systems.backends.torch_runtime.residual_stream_gate_exact",
    }
    supported_route_families = frozenset(
        {"linear_weight", "conv_channel_pair", "residual_stream_gate"}
    )

    def reduce_vjp(
        self,
        *,
        route_family: str,
        activation: Any,
        output_gradient: Any,
        weight: Any,
        metadata: Mapping[str, Any],
    ) -> Any:
        if route_family == "linear_weight":
            # The production scorer and the mechanistic measurement adapter
            # share this exact target-batched kernel owner.  Keep topology and
            # capture outside ``packages.salience``; do not fork its route
            # contraction math in IRG.
            from packages.salience.ops.target_batched import (
                linear_route_measurement_target_batched as canonical_linear_reduce,
            )

            return canonical_linear_reduce(
                activation,
                output_gradient,
                weights=weight,
                scale=metadata.get("scale", 1.0),
                device=output_gradient.device,
                abs_aggregation_mode=str(metadata.get("abs_aggregation_mode", "exact_abs")),
            )
        if route_family == "conv_channel_pair":
            from packages.salience.ops.target_batched import (
                conv_channel_route_measurement_target_batched as canonical_conv_reduce,
            )

            return canonical_conv_reduce(
                activation,
                output_gradient,
                weights=weight,
                scale=metadata.get("scale", 1.0),
                stride=metadata["stride"],
                padding=metadata["padding"],
                dilation=metadata["dilation"],
                groups=int(metadata["groups"]),
                device=output_gradient.device,
                abs_aggregation_mode=str(metadata.get("abs_aggregation_mode", "exact_abs")),
            )
        if route_family == "residual_stream_gate":
            contribution = (
                activation.unsqueeze(0).to(torch.float32)
                * output_gradient.to(torch.float32)
                * float(metadata.get("scale", 1.0))
            )
            per_context = contribution.reshape(
                int(output_gradient.shape[0]), int(activation.shape[0]), 2, -1
            ).sum(dim=-1)
            return per_context.sum(dim=1), per_context.abs().sum(dim=1)
        raise ValueError(f"unsupported route family: {route_family!r}")

    def reduce_vjp_with_context(
        self,
        *,
        route_family: str,
        activation: Any,
        output_gradient: Any,
        weight: Any,
        metadata: Mapping[str, Any],
    ) -> Any:
        """Return pooled scores plus global/local context rows in one pass."""

        if route_family == "linear_weight":
            from packages.salience.ops.target_batched import (
                linear_route_measurement_with_context_target_batched,
            )

            return linear_route_measurement_with_context_target_batched(
                activation,
                output_gradient,
                weights=weight,
                scale=metadata.get("scale", 1.0),
                device=output_gradient.device,
                abs_aggregation_mode=str(
                    metadata.get("abs_aggregation_mode", "exact_abs")
                ),
                retain_local_destination=bool(
                    metadata.get("retain_local_attribution", False)
                ),
                per_context_consumer=metadata.get("per_context_consumer"),
            )
        if route_family == "conv_channel_pair":
            local_regime_tee = metadata.get("local_regime_tee")
            if local_regime_tee is not None:
                from packages.salience.ops.target_batched import (
                    conv_channel_route_measurement_with_context_and_regime_target_batched,
                )

                reducer = (
                    conv_channel_route_measurement_with_context_and_regime_target_batched
                )
            else:
                from packages.salience.ops.target_batched import (
                    conv_channel_route_measurement_with_context_target_batched,
                )

                reducer = conv_channel_route_measurement_with_context_target_batched
            return reducer(
                activation,
                output_gradient,
                weights=weight,
                scale=metadata.get("scale", 1.0),
                stride=metadata["stride"],
                padding=metadata["padding"],
                dilation=metadata["dilation"],
                groups=int(metadata["groups"]),
                device=output_gradient.device,
                abs_aggregation_mode=str(
                    metadata.get("abs_aggregation_mode", "exact_abs")
                ),
                retain_local_destination=bool(
                    metadata.get("retain_local_attribution", False)
                ),
                per_context_consumer=metadata.get("per_context_consumer"),
                **(
                    {"local_regime_tee": local_regime_tee}
                    if local_regime_tee is not None
                    else {}
                ),
            )
        if route_family == "residual_stream_gate":
            contribution = (
                activation.unsqueeze(0).to(torch.float32)
                * output_gradient.to(torch.float32)
                * float(metadata.get("scale", 1.0))
            )
            per_context = contribution.reshape(
                int(output_gradient.shape[0]), int(activation.shape[0]), 2, -1
            ).sum(dim=-1)
            global_rows = per_context.transpose(0, 1)
            local_rows = (
                (
                    activation.to(torch.float32)
                    .flatten(start_dim=2)
                    .mean(dim=2)
                    .mul(float(metadata.get("scale", 1.0)))
                )
                if bool(metadata.get("retain_local_attribution", False))
                else None
            )
            return (
                per_context.sum(dim=1),
                per_context.abs().sum(dim=1),
                global_rows,
                local_rows,
            )
        raise ValueError(f"unsupported route family: {route_family!r}")


class DirectTorchConvRouteReducerBackend:
    """Exact direct-contraction implementation for Conv channel-pair routes."""

    backend_id = "mechanistic_systems.direct_torch_conv_route_reducer.v1"
    implementation_sources = {
        "conv_channel_pair": (
            "mechanistic_systems.backends.torch_runtime."
            "conv_channel_route_measurement_direct_target_batched"
        )
    }
    supported_route_families = frozenset({"conv_channel_pair"})

    def reduce_vjp(
        self,
        *,
        route_family: str,
        activation: Any,
        output_gradient: Any,
        weight: Any,
        metadata: Mapping[str, Any],
    ) -> Any:
        if route_family != "conv_channel_pair":
            raise ValueError(f"unsupported route family: {route_family!r}")
        return conv_channel_route_measurement_direct_target_batched(
            activation,
            output_gradient,
            weights=weight,
            scale=metadata.get("scale", 1.0),
            stride=metadata["stride"],
            padding=metadata["padding"],
            dilation=metadata["dilation"],
            groups=int(metadata["groups"]),
            device=output_gradient.device,
            abs_aggregation_mode=str(metadata.get("abs_aggregation_mode", "exact_abs")),
        )

    def reduce_vjp_with_context(
        self,
        *,
        route_family: str,
        activation: Any,
        output_gradient: Any,
        weight: Any,
        metadata: Mapping[str, Any],
    ) -> Any:
        if route_family != "conv_channel_pair":
            raise ValueError(f"unsupported route family: {route_family!r}")
        return conv_channel_route_measurement_direct_with_context_target_batched(
            activation,
            output_gradient,
            weights=weight,
            scale=metadata.get("scale", 1.0),
            stride=metadata["stride"],
            padding=metadata["padding"],
            dilation=metadata["dilation"],
            groups=int(metadata["groups"]),
            device=output_gradient.device,
            abs_aggregation_mode=str(
                metadata.get("abs_aggregation_mode", "exact_abs")
            ),
            retain_local_destination=bool(
                metadata.get("retain_local_attribution", False)
            ),
            local_regime_tee=metadata.get("local_regime_tee"),
            per_context_consumer=metadata.get("per_context_consumer"),
            local_source_factor=metadata.get("census_local_source_factor"),
        )


class TorchConvKernelWeightRouteReducerBackend:
    """Route-reducer glue for the per-tap Conv kernel-weight family.

    Delegates entirely to the frozen kernel
    ``conv_kernel_route_measurement_target_batched`` that already backs
    ``ConvKernelWeightGateOperator`` in ``mechanistic_systems.core.operators``.
    This class performs no numerics of its own: it only maps the
    registry's ``(activation, output_gradient, weight, metadata)``
    calling convention onto that frozen kernel's keyword arguments.

    ``reduce_vjp_with_context`` is implemented by the same per-tap contraction
    with the pooled and per-context heads sharing the unfolded input patches.
    The route axis remains ``(out, in_per_group, kh, kw)`` throughout; no
    kernel tap is collapsed into a channel-pair route.
    """

    backend_id = "mechanistic_systems.torch_conv_kernel_weight_route_reducer.v1"
    implementation_sources = {
        "conv_kernel_weight": (
            "mechanistic_systems.backends.torch_runtime."
            "conv_kernel_route_measurement_target_batched"
        )
    }
    supported_route_families = frozenset({"conv_kernel_weight"})

    def reduce_vjp(
        self,
        *,
        route_family: str,
        activation: Any,
        output_gradient: Any,
        weight: Any,
        metadata: Mapping[str, Any],
    ) -> Any:
        if route_family != "conv_kernel_weight":
            raise ValueError(f"unsupported route family: {route_family!r}")
        return conv_kernel_route_measurement_target_batched(
            activation,
            output_gradient,
            weights=weight,
            scale=metadata.get("scale", 1.0),
            stride=metadata["stride"],
            padding=metadata["padding"],
            dilation=metadata["dilation"],
            groups=int(metadata["groups"]),
            device=output_gradient.device,
            abs_aggregation_mode=str(
                metadata.get("abs_aggregation_mode", "exact_abs")
            ),
        )

    def reduce_vjp_with_context(
        self,
        *,
        route_family: str,
        activation: Any,
        output_gradient: Any,
        weight: Any,
        metadata: Mapping[str, Any],
    ) -> Any:
        if route_family != "conv_kernel_weight":
            raise ValueError(f"unsupported route family: {route_family!r}")
        return conv_kernel_route_measurement_with_context_target_batched(
            activation,
            output_gradient,
            weights=weight,
            scale=metadata.get("scale", 1.0),
            stride=metadata["stride"],
            padding=metadata["padding"],
            dilation=metadata["dilation"],
            groups=int(metadata["groups"]),
            device=output_gradient.device,
            abs_aggregation_mode=str(
                metadata.get("abs_aggregation_mode", "exact_abs")
            ),
            retain_local_destination=bool(
                metadata.get("retain_local_attribution", False)
            ),
            per_context_consumer=metadata.get("per_context_consumer"),
            profiler=metadata.get("profiler"),
            local_source_factor=metadata.get("census_local_source_factor"),
        )


class PreparedLanes:
    """Gate lanes prepared once for repeated batched forwards.

    Stacking the per-lane gate buffers and building the vmap runner are
    input-independent; preparing them once per lane-set (instead of per
    forward call) removes the dominant per-call Python/dispatch cost of the
    fused path. ``lane_keys`` carry the mask identities for telemetry.
    """

    __slots__ = (
        "lane_keys",
        "buffer_names",
        "single_overrides",
        "stacked_overrides",
        "runner",
        "prefix_gm",
        "prefix_runner",
        "suffix_gm",
        "prefix_overrides",
        "stacked_suffix",
        "suffix_runner",
        "suffix_compile_applied",
        "prefix_fork_layer_id",
        "prefix_dense",
        "prefix_gate_versions",
        "prefix_fallback_reason",
        "resident_leases",
        "execution_weights_prepared",
    )

    def __init__(
        self,
        *,
        lane_keys: tuple[str, ...],
        buffer_names: tuple[str, ...],
        single_overrides: dict[str, Any] | None,
        stacked_overrides: dict[str, Any] | None,
        runner: Any | None = None,
        prefix_gm: Any | None = None,
        prefix_runner: Any | None = None,
        suffix_gm: Any | None = None,
        prefix_overrides: dict[str, Any] | None = None,
        stacked_suffix: dict[str, Any] | None = None,
        suffix_runner: Any | None = None,
        suffix_compile_applied: bool = False,
        prefix_fork_layer_id: str | None = None,
        prefix_dense: bool = False,
        prefix_gate_versions: tuple | None = None,
        prefix_fallback_reason: str = "",
        resident_leases: tuple | None = None,
        execution_weights_prepared: bool = False,
    ) -> None:
        self.lane_keys = lane_keys
        self.buffer_names = buffer_names
        self.single_overrides = single_overrides
        self.stacked_overrides = stacked_overrides
        self.resident_leases = resident_leases
        self.execution_weights_prepared = bool(execution_weights_prepared)
        # Input-independent batched runner: vmap over the stacked gate
        # overrides with inputs broadcast, built (and optionally
        # torch.compile'd) ONCE per lane-set so compile warmup amortizes
        # across every batch and look that reuses the lanes.
        self.runner = runner
        # A legal graph cut shares the complete lane-invariant prefix and
        # executes every differing and downstream coordinate in the suffix.
        self.prefix_gm = prefix_gm
        self.prefix_runner = prefix_runner
        self.suffix_gm = suffix_gm
        self.prefix_overrides = prefix_overrides
        self.stacked_suffix = stacked_suffix
        self.suffix_runner = suffix_runner
        self.suffix_compile_applied = bool(suffix_compile_applied)
        self.prefix_fork_layer_id = prefix_fork_layer_id
        self.prefix_dense = bool(prefix_dense)
        self.prefix_gate_versions = prefix_gate_versions
        self.prefix_fallback_reason = prefix_fallback_reason
    @property
    def lane_count(self) -> int:
        return len(self.lane_keys)


def _v6_runner_kind(runner: Any) -> str:
    if runner is None:
        return "none"
    return str(getattr(runner, "__qualname__", type(runner).__qualname__))


class TorchCandidateExecutionBackend:
    backend_id = "mechanistic_systems.torch_candidate_execution.v1"

    def __init__(
        self,
        *,
        compile_vmap: bool | None = None,
        forward_prefix_reuse: bool | None = None,
        runtime_execution_profile: RuntimeExecutionProfile | None = None,
    ) -> None:
        self.runtime_execution_profile = RuntimeExecutionProfile.resolve(
            runtime_execution_profile
        )
        if compile_vmap is None:
            # Benchmark-decided A/B knob: with the runner now built once per
            # lane-set, compile warmup amortizes across the whole unit.
            compile_vmap = self.runtime_execution_profile.vmap_compile
        self.compile_vmap = bool(compile_vmap)
        self.compile_vmap_applied = False
        # FORWARD prefix-reuse defaults on. ``None`` resolves the choice
        # from the environment; an explicit bool overrides it. Resolved
        # once here so the whole evaluator shares one immutable choice.
        if forward_prefix_reuse is None:
            forward_prefix_reuse = _forward_prefix_reuse_from_env()
        self.forward_prefix_reuse = bool(forward_prefix_reuse)
        # Per-call: did the most recent evaluate_prepared take the reuse path?
        self.prefix_reuse_applied = False
        # Aggregate telemetry (never scientific identity).
        self.prefix_reuse_prepared = 0
        self.prefix_reuse_forwards = 0
        self.last_prefix_fallback_reason = ""
        # Split GraphModules are a function of (model, fork module path) only,
        # not the concrete gate values, so they memoize across lane-sets that
        # fork at the same layer. A str entry records a cached non-split reason.
        self._prefix_split_cache: dict[tuple, Any] = {}
        self._activation_prefix_cache: OrderedDict[tuple, dict[str, Any]] = (
            OrderedDict()
        )
        self._activation_prefix_cache_bytes = 0
        self.activation_prefix_cache_hits = 0
        self.activation_prefix_cache_misses = 0
        self.activation_prefix_cache_invalidations = 0
        self.activation_prefix_saved_forwards = 0
        self.activation_prefix_cache_fallbacks: dict[str, int] = {}
        self.last_fallback_reason = ""
        self.profiler = PhaseProfiler(
            cuda_events_enabled=self.runtime_execution_profile.cuda_events
        )
        self.calls = 0
        self.processed_candidates = 0
        self.processed_contexts = 0
        self.device_type = "unknown"
        # named_buffers() walks every module; the gate graph is compiled once
        # per evaluator, so the base-buffer map is a per-model constant.
        # set_gate mutates buffers in place (copy_), so cached references stay
        # current.
        self._base_buffers_cache: dict[int, dict[str, Any]] = {}
        if torch is not None:
            from .torch_execution_weights import ExecutionWeightCache
            self._execution_weight_cache = ExecutionWeightCache()
        # A vmapped functional runner depends on the model graph, ordered
        # override axis, lane count, and device—not on candidate gate values.
        # Compiling a fresh closure for every parent/candidate mask pair made
        # TorchDynamo create hundreds of equivalent graphs during v0
        # bisection.  Retain a small structural cache so changing masks remain
        # ordinary tensor inputs to one compiled runner.
        self._prepared_vmap_runner_cache: OrderedDict[tuple, Any] = (
            OrderedDict()
        )
        self._prepared_single_runner_cache: OrderedDict[tuple, Any] = (
            OrderedDict()
        )
        # Suffix structures vary with the fork. Their vmap runners remain
        # eager; full-model and shared-prefix runners use stable compilation.
        self.prefix_reuse_intentional_eager_suffix_prepares = 0
        self.prepared_vmap_runner_cache_hits = 0
        self.prepared_vmap_runner_cache_misses = 0
        self.prepared_single_runner_cache_hits = 0
        self.prepared_single_runner_cache_misses = 0
        self.compile_single_lane_applied = False
        # One capture per (model, override-name-set, row shape, dtype).
        # Bisection keeps these constant within a unit, so the cache stays
        # single-digit; entries hold the graph plus its static tensors.
        self._cuda_graph_cache: dict[tuple, dict[str, Any]] = {}
        self.cuda_graphs_applied = False
        # IRG_WALK_REDUCE_OVERHEAD telemetry: counts of structural
        # runners built under the armed walk phase with
        # ``torch.compile(mode="reduce-overhead", dynamic=False)``.  A canary
        # verifies engagement by asserting ``runners_compiled >= 1`` in the
        # receipt's backend metadata when the env gate was set.
        self.walk_reduce_overhead_vmap_runners = 0
        self.walk_reduce_overhead_single_runners = 0
        # Worker-lifetime walk graphs share gate/weight workspaces and a
        # graph memory pool. Concrete proposals only update those workspaces;
        # their values never enter the compilation or capture key.
        self._walk_graph_cache: dict[tuple, dict[str, Any]] = {}
        self._walk_graph_buffers: dict[tuple, dict[str, Any]] = {}
        self._walk_graph_models: set[int] = set()
        self._walk_graph_replay_streams: dict[str, int] = {}
        self._walk_graph_pool = None
        self.walk_graph_captures = 0
        self.walk_graph_replays = 0
        self.walk_graph_buffer_copies = 0

    def release_item_resources(self) -> dict[str, int]:
        """Release mask-owned storage once all work for an item has finished.

        Prewarmed walk graphs own reusable static inputs, outputs and weight
        workspaces. Their shapes and model stay fixed across worker items.
        The source tensors remembered to avoid repeated workspace copies,
        however, belong to the current item's resident masks. Forget their
        version tokens with their references: allocator address reuse must
        never make a later item's first copy look redundant.
        """

        released = {
            "cuda_graphs": len(self._cuda_graph_cache),
            "activation_prefix_rows": len(self._activation_prefix_cache),
            "execution_weight_entries": sum(
                len(entries) for entries in self._execution_weight_cache._layers.values()
            ),
            "walk_source_tensors": sum(
                len(entry["sources"]) for entry in self._walk_graph_buffers.values()
            ),
            "reused_walk_graphs": len(self._walk_graph_cache),
        }
        self._cuda_graph_cache.clear()
        self.invalidate_activation_prefix_cache("item_end")
        self._execution_weight_cache._layers.clear()
        for entry in self._walk_graph_buffers.values():
            entry["sources"].clear()
            entry["versions"].clear()
        return released

    @staticmethod
    def _prefix_activation_bytes(value: Any) -> int:
        if isinstance(value, (tuple, list)):
            return sum(
                TorchCandidateExecutionBackend._prefix_activation_bytes(item)
                for item in value
            )
        if not torch.is_tensor(value):
            return 0
        return int(value.numel() * value.element_size())

    @staticmethod
    def _prefix_gate_versions(overrides: Mapping[str, Any]) -> tuple | None:
        versions = tuple(
            (name, tensor_mutation_identity(value)) for name, value in overrides.items()
        )
        return versions if all(identity is not None for _, identity in versions) else None

    def _activation_prefix_eligible(self, prepared: PreparedLanes) -> bool:
        if prepared.prefix_gate_versions is None:
            return False
        if prepared.prefix_gate_versions != self._prefix_gate_versions(
            prepared.prefix_overrides
        ):
            # A detached tensor shares storage and its mutation counter with
            # the original. Never let an in-place edit preserve old rows.
            self.invalidate_activation_prefix_cache("prefix_gate_mutation")
            prepared.prefix_dense = False
            # A prefix override is a view of lane zero. After an edit the
            # other lanes may differ before the old fork, so the complete
            # full-model runner must execute their current stacked gates.
            prepared.suffix_runner = None
            prepared.prefix_fallback_reason = (
                "prefix_reuse_skipped:prefix_gate_mutation"
            )
            self.last_prefix_fallback_reason = prepared.prefix_fallback_reason
            return False
        return prepared.prefix_dense

    def invalidate_activation_prefix_cache(self, reason: str) -> None:
        """Retire every cached activation at an explicit state boundary."""

        removed = len(self._activation_prefix_cache)
        if removed:
            self.activation_prefix_cache_invalidations += removed
        self._activation_prefix_cache.clear()
        self._activation_prefix_cache_bytes = 0
        # No row is left to protect: forget freed state tensors and the old
        # state tokens (token numbers are never reused).
        records = self.__dict__.setdefault("_activation_prefix_state_records", {})
        for tensor_id in [k for k, record in records.items() if record[0]() is None]:
            del records[tensor_id]
        self.__dict__.setdefault("_activation_prefix_state_tokens", {}).clear()
        token = str(reason) or "unspecified"
        self.activation_prefix_cache_fallbacks[f"invalidated:{token}"] = (
            self.activation_prefix_cache_fallbacks.get(
                f"invalidated:{token}", 0
            )
            + removed
        )

    @staticmethod
    def _activation_prefix_state(prepared: PreparedLanes) -> tuple:
        """Prefix weights and the buffers that are not gate overrides."""

        return tuple(
            (name, value)
            for kind, iterator in (
                ("parameter", prepared.prefix_gm.named_parameters()),
                ("buffer", prepared.prefix_gm.named_buffers()),
            )
            for name, value in iterator
            if kind == "parameter" or name not in prepared.prefix_overrides
        )

    def _activation_prefix_state_token(self, prepared: PreparedLanes) -> int | None:
        """Host identity of the prefix weights and non-gate buffers.

        Each state tensor is identified by its object, its storage, data
        pointer and layout, and its PyTorch version counter.  In-place
        updates (also under ``no_grad`` or inference mode), ``copy_`` and
        ``load_state_dict``, ``.data`` assignment and dtype or device moves
        change a recorded tensor's identity, and that clears every row.  A
        replaced tensor is a new object, so it changes the row key: older
        rows are never served and age out under the byte limit.  Weak
        references to each tensor and its storage keep identities exact after
        either is freed.  No device work or host synchronization is involved.

        Not seen: an in-place write through ``.data`` (or through an alias
        taken from ``.data``, or a NumPy view), because that skips the
        version counter.  Engine sources must not write model state that
        way.  BatchNorm's
        own training-mode forward is seen through ``num_batches_tracked``;
        its running mean and variance updates do not bump their counters.

        Returns an exact small-integer token, or None when a state tensor
        has no version counter (an inference tensor) or no storage.
        """

        import weakref

        from torch.multiprocessing.reductions import StorageWeakRef

        records = self.__dict__.setdefault("_activation_prefix_state_records", {})
        signature = []
        current = {}
        changed = False
        for name, value in self._activation_prefix_state(prepared):
            try:
                storage = value.untyped_storage()
                identity = (
                    id(value),
                    int(storage._cdata),
                    int(value.data_ptr()),
                    int(value.storage_offset()),
                    tuple(value.shape),
                    tuple(value.stride()),
                    str(value.dtype),
                    str(value.device),
                    int(value._version),
                )
                record = records.get(id(value))
                if record is None or record[0]() is not value or record[2] != identity:
                    changed = changed or record is not None
                    record = (weakref.ref(value), StorageWeakRef(storage), identity)
            except (AttributeError, NotImplementedError, RuntimeError, TypeError):
                return None
            current[id(value)] = record
            signature.append((str(name), identity))
        if changed:
            self.invalidate_activation_prefix_cache("prefix_state_changed")
        records.update(current)
        tokens = self.__dict__.setdefault("_activation_prefix_state_tokens", {})
        token = tokens.get(tuple(signature))
        if token is None and len(tokens) >= 256:
            # A replaced tensor gets a new token rather than a clear; bound
            # the host bookkeeping (one token per live prefix state suffices).
            self.invalidate_activation_prefix_cache("prefix_state_token_limit")
        if token is None:
            token = int(self.__dict__.get("_activation_prefix_next_state_token", 0))
            self._activation_prefix_next_state_token = token + 1
            tokens[tuple(signature)] = token
        return token

    def _activation_prefix_key(
        self,
        compiled_graph: Any,
        prepared: PreparedLanes,
        inputs: Any,
        *,
        context_ids: Sequence[str] | None,
        input_identity_hash: str | None,
    ) -> tuple | None:
        if (
            context_ids is None
            or input_identity_hash is None
            or prepared.prefix_fork_layer_id is None
            or prepared.prefix_overrides is None
            or not self._activation_prefix_eligible(prepared)
        ):
            return None
        layout = (
            "channels_last"
            if inputs.ndim == 4
            and inputs.is_contiguous(memory_format=torch.channels_last)
            else "contiguous"
        )
        if len(context_ids) != int(inputs.shape[0]):
            return None
        # The graph module shares the original weight tensors.  Rows are keyed
        # on their host identities and version counters; gate eligibility
        # comes from host mask identity and mutation metadata.
        state_token = self._activation_prefix_state_token(prepared)
        if state_token is None:
            # A state tensor without a mutation counter cannot support the
            # persistent cache. Keep ordinary within-batch reuse available.
            return None
        common = (
            id(compiled_graph.model),
            str(prepared.prefix_fork_layer_id),
            str(input_identity_hash),
            state_token,
            # BatchNorm epsilon enters the eval map without being a tensor,
            # so the state token cannot see an edit to it. Other
            # forward-configuration edits need an explicit invalidation.
            tuple(
                (name, float(module.eps))
                for name, module in prepared.prefix_gm.named_modules()
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            ),
            tuple(int(value) for value in inputs.shape[1:]),
            tuple(int(value) for value in inputs.stride()[1:]),
            str(inputs.dtype),
            str(inputs.device),
            layout,
            _v6_runner_kind(prepared.prefix_runner),
            bool(torch.backends.cuda.matmul.allow_tf32),
            bool(torch.backends.cudnn.enabled),
            bool(torch.backends.cudnn.allow_tf32),
            bool(torch.backends.cudnn.deterministic),
            bool(torch.backends.cudnn.benchmark),
            bool(torch.are_deterministic_algorithms_enabled()),
            str(torch.get_float32_matmul_precision()),
            bool(torch.is_autocast_enabled(inputs.device.type)),
            str(torch.get_autocast_dtype(inputs.device.type)),
        )
        return tuple(
            (common, str(context_id))
            for context_id in context_ids
        )

    def _activation_prefix_lookup(
        self,
        key: tuple | None,
        prepared: PreparedLanes,
    ) -> Any | None:
        if key is None or not self._activation_prefix_eligible(prepared):
            return None
        entries = []
        for row_key in key:
            entry = self._activation_prefix_cache.get(row_key)
            if entry is None:
                self.activation_prefix_cache_misses += 1
                return None
            entries.append(entry)
        if not entries:
            return None
        for row_key in key:
            self._activation_prefix_cache.move_to_end(row_key)
        self.activation_prefix_cache_hits += 1
        self.activation_prefix_saved_forwards += 1
        first = entries[0]["activation"]
        if isinstance(first, tuple):
            return tuple(
                torch.cat(tuple(entry["activation"][index] for entry in entries), dim=0)
                for index in range(len(first))
            )
        return torch.cat(tuple(entry["activation"] for entry in entries), dim=0)

    def _activation_prefix_store(
        self,
        key: tuple | None,
        prepared: PreparedLanes,
        activation: Any,
    ) -> None:
        if key is None or not self._activation_prefix_eligible(prepared):
            return
        values = (
            tuple(activation) if isinstance(activation, (tuple, list)) else (activation,)
        )
        if not all(
            torch.is_tensor(value)
            and value.ndim >= 1
            and int(value.shape[0]) == len(key)
            for value in values
        ):
            return
        limit = process_device_memory_budget_bytes(
            int(self.runtime_execution_profile.activation_prefix_cache_bytes)
        )
        for index, row_key in enumerate(key):
            # Compute payload size before allocating a row snapshot. Evict
            # before inserting so the resident payload never exceeds the
            # byte limit, including while a large batch is being stored.
            byte_count = sum(
                self._prefix_activation_bytes(value[index:index + 1]) for value in values
            )
            if byte_count <= 0 or byte_count > limit:
                self.activation_prefix_cache_fallbacks[
                    "entry_exceeds_byte_limit"
                ] = self.activation_prefix_cache_fallbacks.get(
                    "entry_exceeds_byte_limit", 0
                ) + 1
                continue
            previous = self._activation_prefix_cache.pop(row_key, None)
            if previous is not None:
                self._activation_prefix_cache_bytes -= int(previous["bytes"])
            while self._activation_prefix_cache_bytes + byte_count > limit:
                _old_key, old = self._activation_prefix_cache.popitem(last=False)
                self._activation_prefix_cache_bytes -= int(old["bytes"])
            row_values = tuple(value[index:index + 1].detach().clone() for value in values)
            retained = row_values if isinstance(activation, (tuple, list)) else row_values[0]
            self._activation_prefix_cache[row_key] = {
                "activation": retained,
                "bytes": byte_count,
            }
            self._activation_prefix_cache_bytes += byte_count

    def _cuda_graph_forward(
        self, compiled_graph: Any, overrides: Mapping[str, Any], inputs: Any
    ) -> Any:
        """Replay the K=1 forward as a captured CUDA graph.

        Ragged tails pad to the capture shape and slice the output rows.
        Evaluation rows are independent under frozen normalization.
        """

        model = compiled_graph.model
        if getattr(model, "training", False):
            raise RuntimeError("cuda_graphs requires an eval-mode model")
        rows = int(inputs.shape[0])
        names = tuple(sorted(str(name) for name in overrides))
        key = (
            id(model),
            names,
            tuple(int(d) for d in inputs.shape[1:]),
            str(inputs.dtype),
        )
        entry = self._cuda_graph_cache.get(key)
        if entry is not None and rows > entry["rows"]:
            entry = None  # cannot pad down; recapture at the larger shape
        if entry is None:
            initialize_cuda_graph_state(inputs.device)
            with torch.inference_mode(False):
                static_input = torch.zeros(
                    (rows, *inputs.shape[1:]), dtype=inputs.dtype, device=inputs.device
                )
                static_overrides = {
                    name: overrides[name].detach().clone() for name in names
                }
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream), torch.inference_mode():
                for _ in range(3):
                    torch.func.functional_call(
                        model, static_overrides, (static_input,), strict=False
                    )
            torch.cuda.current_stream().wait_stream(side_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(graph):
                static_output = torch.func.functional_call(
                    model, static_overrides, (static_input,), strict=False
                )
            entry = {
                "graph": graph,
                "input": static_input,
                "overrides": static_overrides,
                "output": static_output,
                "rows": rows,
            }
            self._cuda_graph_cache[key] = entry
        entry["input"][:rows].copy_(inputs)
        if rows < entry["rows"]:
            entry["input"][rows:].zero_()
        for name in names:
            entry["overrides"][name].copy_(overrides[name])
        entry["graph"].replay()
        self.cuda_graphs_applied = True
        # clone is mandatory: downstream views of the static output would
        # alias the next replay's results.
        return entry["output"][:rows].clone()

    @staticmethod
    def _walk_graph_layout(value: Any) -> tuple:
        return tuple(value.shape), str(value.dtype), str(value.device)

    def _walk_graph_key(self, compiled_graph: Any, prepared: PreparedLanes, inputs: Any) -> tuple:
        overrides = prepared.single_overrides or prepared.stacked_overrides
        autocast_enabled = bool(torch.is_autocast_enabled(inputs.device.type))
        return (
            id(compiled_graph.model), prepared.prefix_fork_layer_id,
            prepared.lane_count, tuple(inputs.shape[1:]), str(inputs.dtype),
            str(inputs.device), autocast_enabled,
            # A disabled autocast scope still changes the reported dtype,
            # but it does not change the forward. Startup uses such a BF16
            # scope at tier 0; decisions execute outside it.
            str(torch.get_autocast_dtype(inputs.device.type)) if autocast_enabled else None,
            tuple((name, self._walk_graph_layout(value)) for name, value in overrides.items()),
        )

    def _walk_static_buffers(self, compiled_graph: Any, prepared: PreparedLanes) -> dict[str, Any]:
        overrides = prepared.single_overrides or prepared.stacked_overrides
        key = (id(compiled_graph.model), prepared.lane_count,
               tuple((name, self._walk_graph_layout(value)) for name, value in overrides.items()))
        entry = self._walk_graph_buffers.get(key)
        if entry is None:
            # Keep original layouts: effective convolution weights may be
            # channels-last lane views, and shared weights may have stride 0.
            with torch.inference_mode(False), torch.no_grad():
                buffers = {}
                for name, value in overrides.items():
                    if value.ndim == 5 and name.endswith(".execution_weight"):
                        buffer = torch.empty(
                            (value.shape[0] * value.shape[1], *value.shape[2:]),
                            device=value.device, dtype=value.dtype,
                            memory_format=torch.channels_last,
                        ).view(value.shape)
                        buffers[name] = buffer.copy_(value)
                    else:
                        buffers[name] = value.detach().clone(memory_format=torch.preserve_format)
            entry = {"buffers": buffers, "versions": {}, "sources": {}}
            self._walk_graph_buffers[key] = entry
        versions = entry["versions"]
        for name, value in overrides.items():
            identity = tensor_mutation_identity(value)
            token = (int(value.data_ptr()), tuple(value.stride()),
                     None if identity is None else identity[1])
            if identity is None or versions.get(name) != token:
                entry["buffers"][name].copy_(value)
                versions[name] = token
                self.walk_graph_buffer_copies += 1
            # Retain the source storage while its version is remembered;
            # allocator address reuse then cannot imitate a current buffer.
            entry["sources"][name] = value
        return entry["buffers"]

    @staticmethod
    def _walk_output_snapshot(value: Any, rows: int) -> Any:
        if torch.is_tensor(value):
            return value[:, :rows].clone()
        if isinstance(value, Mapping):
            return type(value)((name, TorchCandidateExecutionBackend._walk_output_snapshot(item, rows))
                               for name, item in value.items())
        if isinstance(value, tuple):
            return tuple(TorchCandidateExecutionBackend._walk_output_snapshot(item, rows) for item in value)
        if isinstance(value, list):
            return [TorchCandidateExecutionBackend._walk_output_snapshot(item, rows) for item in value]
        raise TypeError(f"walk forward returned unsupported output leaf {type(value)!r}")

    def _capture_walk_graph(self, compiled_graph: Any, prepared: PreparedLanes, inputs: Any) -> dict[str, Any]:
        """Compile and capture one padded prefix/suffix pair at worker start.

        The compiled function contains both forwards; manual capture owns
        replay, so Inductor's separate CUDA graph trees are disabled. Each
        fork gets its own code object to avoid Dynamo's per-code cache limit
        when all layers and networks are prepared in the same worker.
        """
        import types
        from .torch_execution_weights import remap_execution_buffers

        if compiled_graph.model.training:
            raise RuntimeError("walk graph capture requires an eval-mode model")
        key = self._walk_graph_key(compiled_graph, prepared, inputs)
        rows = int(inputs.shape[0])
        cache_key = (key, rows)
        cached = self._walk_graph_cache.get(cache_key)
        if cached is not None:
            return cached
        initialize_cuda_graph_state(inputs.device)
        buffers = self._walk_static_buffers(compiled_graph, prepared)
        with torch.inference_mode(False):
            static_input = torch.empty_like(inputs).copy_(inputs)
        model = compiled_graph.model
        prefix_gm, suffix_gm = prepared.prefix_gm, prepared.suffix_gm
        if prefix_gm is not None:
            _, _, prefix_map, suffix_map = self._prefix_split_cache[
                (id(model), prepared.prefix_fork_layer_id)
            ]
            # Remap execution-weight names once, without reading any tensor
            # value. FX partitions retain the original leaf modules.
            named_extras = {name: name for name in buffers if ".execution_" in name}
            prefix_extra_names = remap_execution_buffers(compiled_graph, prefix_gm, named_extras)
            suffix_extra_names = remap_execution_buffers(compiled_graph, suffix_gm, named_extras)
            prefix_names = {**prefix_map, **prefix_extra_names}
            suffix_names = {**suffix_map, **suffix_extra_names}

            def run_walk(gates: Mapping[str, Any], images: Any) -> Any:
                prefix = torch.func.functional_call(
                    prefix_gm, {name: gates[original][0] for name, original in prefix_names.items()},
                    (images,), strict=False,
                )

                def suffix(lane: Mapping[str, Any], activation: Any) -> Any:
                    arguments = tuple(activation) if isinstance(activation, (tuple, list)) else (activation,)
                    return torch.func.functional_call(suffix_gm, dict(lane), arguments, strict=False)

                suffix_gates = {name: gates[original] for name, original in suffix_names.items()}
                return torch.vmap(suffix, in_dims=(0, None))(suffix_gates, prefix)
        elif prepared.lane_count == 1:
            def run_walk(gates: Mapping[str, Any], images: Any) -> Any:
                return _output_tree_unsqueeze_lane(
                    torch.func.functional_call(model, dict(gates), (images,), strict=False)
                )
        else:
            def run_walk(gates: Mapping[str, Any], images: Any) -> Any:
                def lane_forward(lane: Mapping[str, Any], batch: Any) -> Any:
                    return torch.func.functional_call(model, dict(lane), (batch,), strict=False)
                return torch.vmap(lane_forward, in_dims=(0, None))(gates, images)

        runner = types.FunctionType(
            run_walk.__code__.replace(co_name=f"walk_{id(model)}_{prepared.prefix_fork_layer_id}_{prepared.lane_count}"),
            run_walk.__globals__, run_walk.__name__, run_walk.__defaults__, run_walk.__closure__,
        )
        runner = torch.compile(runner, dynamic=False, fullgraph=True,
                               options={"triton.cudagraphs": False})
        stream = torch.cuda.Stream(device=inputs.device)
        stream.wait_stream(torch.cuda.current_stream(inputs.device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                runner(buffers, static_input)
        # Compilation and warmup finish before the worker serves decisions.
        stream.synchronize()
        if self._walk_graph_pool is None:
            self._walk_graph_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=self._walk_graph_pool, stream=stream):
            output = runner(buffers, static_input)
        torch.cuda.current_stream(inputs.device).wait_stream(stream)
        entry = {"graph": graph, "input": static_input, "output": output,
                 "rows": rows, "key": key, "runner": runner}
        self._walk_graph_cache[cache_key] = entry
        self.walk_graph_captures += 1
        return entry

    def prewarm_walk_graphs(self, compiled_graph: Any, inputs: Any,
                            prepared_lanes: Sequence[PreparedLanes]) -> None:
        """Prepare all supplied layer pairs once for this worker and shape."""
        if getattr(inputs.device, "type", None) != "cuda":
            return
        for prepared in prepared_lanes:
            self._capture_walk_graph(compiled_graph, prepared, inputs)
        self._walk_graph_models.add(id(compiled_graph.model))

    def carry_worker_walk_graphs(self, previous: TorchCandidateExecutionBackend) -> None:
        """Carry executable graph addresses when per-run counters are reset."""
        self._walk_graph_cache = previous._walk_graph_cache
        self._walk_graph_buffers = previous._walk_graph_buffers
        self._walk_graph_models = previous._walk_graph_models
        self._walk_graph_pool = previous._walk_graph_pool
        self._walk_graph_replay_streams = previous._walk_graph_replay_streams
        self.walk_graph_captures = previous.walk_graph_captures

    def _walk_graph_forward(self, compiled_graph: Any, prepared: PreparedLanes, inputs: Any) -> Any:
        stream = torch.cuda.current_stream(inputs.device)
        stream_id = int(stream.cuda_stream)
        device_key = str(inputs.device)
        expected_stream = self._walk_graph_replay_streams.setdefault(device_key, stream_id)
        if stream_id != expected_stream:
            raise RuntimeError("worker walk graphs require one ordered compute stream")
        key = self._walk_graph_key(compiled_graph, prepared, inputs)
        rows = int(inputs.shape[0])
        choices = [entry for (entry_key, capacity), entry in self._walk_graph_cache.items()
                   if entry_key == key and capacity >= rows]
        if not choices:
            raise RuntimeError("walk forward shape was not prepared at worker start")
        entry = min(choices, key=lambda item: item["rows"])
        self._walk_static_buffers(compiled_graph, prepared)
        entry["input"][:rows].copy_(inputs)
        if rows < entry["rows"]:
            entry["input"][rows:].zero_()
        entry["graph"].replay()
        self.walk_graph_replays += 1
        self.cuda_graphs_applied = True
        self.prefix_reuse_applied = prepared.prefix_gm is not None
        self.prefix_reuse_forwards += int(self.prefix_reuse_applied)
        return self._walk_output_snapshot(entry["output"], rows)

    def _maybe_compile_vmap(
        self, runner: Any, *, device: Any, reduce_overhead: bool = False
    ) -> Any:
        """Compile only on CUDA; CPU keeps the same uncompiled vmap path.

        ``reduce_overhead=True`` is the IRG_WALK_REDUCE_OVERHEAD lane: the
        runner is compiled with ``mode="reduce-overhead"`` (inductor CUDA
        graph trees) and ``dynamic=False`` so a steady walk shape replays as
        one captured graph.  Safety of the capture boundary:

        * It can only be requested while the pilot has ARMED the walk phase
          (see ``perf_gates.walk_reduce_overhead_active``), which excludes the
          census phase whose ``device_route_moments`` offload/restore issues
          side-stream ``record_stream`` transfers.  Cudagraph trees capture
          with thread-local capture error mode, but excluding concurrent
          side-stream traffic by phase keeps the capture window inert by
          construction, not by allocator luck.
        * The walk's physical batches are padded to fixed shapes
          (``_fixed_shape_execution_sources``), so ``dynamic=False`` sees a
          bounded shape set; TorchDynamo's cache limit bounds any residual
          per-shape specialization churn.
        * Replay re-executes the captured kernels on the current stream after
          the data plane's H2D ready-event wait, so input staging ordering is
          consistent with the input staging order used by the runner.
        """

        if (
            self.compile_vmap
            and getattr(device, "type", None) == "cuda"
            and hasattr(torch, "compile")
        ):
            self.compile_vmap_applied = True
            if reduce_overhead:
                initialize_cuda_graph_state(device)
                self.walk_reduce_overhead_vmap_runners += 1
                return torch.compile(
                    runner,
                    mode="reduce-overhead",
                    dynamic=False,
                )
            return torch.compile(
                runner,
                dynamic=bool(
                    self.runtime_execution_profile.compile_dynamic_batch
                ),
            )
        return runner

    def _prepared_vmap_runner(
        self,
        model: Any,
        *,
        buffer_names: tuple[str, ...],
        lane_count: int,
        device: Any,
        compile_vmap: bool | None = None,
    ) -> Any:
        """Return one structural vmap runner shared across concrete masks."""

        device_key = (
            getattr(device, "type", str(device)),
            getattr(device, "index", None),
        )
        effective_compile = (
            bool(self.compile_vmap) if compile_vmap is None else bool(compile_vmap)
        )
        # IRG_WALK_REDUCE_OVERHEAD: build a SEPARATE runner variant while the
        # walk phase is armed so census/terminal phases keep the default
        # dynamic-batch compile and never touch a cudagraph-tree runner.
        reduce_overhead = (
            effective_compile
            and getattr(device, "type", None) == "cuda"
            and walk_reduce_overhead_active()
        )
        key = (
            id(model),
            tuple(buffer_names),
            int(lane_count),
            device_key,
            bool(effective_compile),
            bool(self.runtime_execution_profile.compile_dynamic_batch),
            bool(reduce_overhead),
        )
        cached = self._prepared_vmap_runner_cache.get(key)
        if cached is not None:
            self.prepared_vmap_runner_cache_hits += 1
            self._prepared_vmap_runner_cache.move_to_end(key)
            return cached
        self.prepared_vmap_runner_cache_misses += 1

        def run_lane(buffer_overrides: Mapping[str, Any], inputs: Any) -> Any:
            return torch.func.functional_call(
                model,
                dict(buffer_overrides),
                inputs if isinstance(inputs, tuple) else (inputs,),
                strict=False,
            )

        runner = torch.vmap(run_lane, in_dims=(0, None))
        if effective_compile:
            # The keyword rides only on the armed walk-phase path so callers
            # (and tests) that stub ``_maybe_compile_vmap`` with the default
            # ``(runner, *, device)`` signature are untouched when the gate
            # is off.
            runner = (
                self._maybe_compile_vmap(
                    runner, device=device, reduce_overhead=True
                )
                if reduce_overhead
                else self._maybe_compile_vmap(runner, device=device)
            )
        self._prepared_vmap_runner_cache[key] = runner
        self._prepared_vmap_runner_cache.move_to_end(key)
        return runner

    def _prepared_single_runner(
        self,
        model: Any,
        *,
        buffer_names: tuple[str, ...],
        device: Any,
    ) -> Any:
        """Return one structural K=1 runner shared by changing gate values."""

        device_key = (
            getattr(device, "type", str(device)),
            getattr(device, "index", None),
        )
        # IRG_WALK_REDUCE_OVERHEAD: separate cudagraph-tree K=1 variant while
        # the walk phase is armed (see ``_maybe_compile_vmap`` for the safety
        # argument); census/terminal keep the default dynamic-batch runner.
        reduce_overhead = (
            self.runtime_execution_profile.k1_compile
            and getattr(device, "type", None) == "cuda"
            and walk_reduce_overhead_active()
        )
        key = (
            id(model),
            tuple(buffer_names),
            device_key,
            bool(self.runtime_execution_profile.k1_compile),
            bool(self.runtime_execution_profile.compile_dynamic_batch),
            bool(reduce_overhead),
        )
        cached = self._prepared_single_runner_cache.get(key)
        if cached is not None:
            self.prepared_single_runner_cache_hits += 1
            self._prepared_single_runner_cache.move_to_end(key)
            return cached
        self.prepared_single_runner_cache_misses += 1

        def run_single(buffer_overrides: Mapping[str, Any], inputs: Any) -> Any:
            return torch.func.functional_call(
                model,
                dict(buffer_overrides),
                (inputs,),
                strict=False,
            )

        runner = run_single
        if (
            self.runtime_execution_profile.k1_compile
            and getattr(device, "type", None) == "cuda"
            and hasattr(torch, "compile")
        ):
            if reduce_overhead:
                initialize_cuda_graph_state(device)
                runner = torch.compile(
                    runner,
                    mode="reduce-overhead",
                    dynamic=False,
                )
                self.walk_reduce_overhead_single_runners += 1
            else:
                runner = torch.compile(
                    runner,
                    dynamic=bool(
                        self.runtime_execution_profile.compile_dynamic_batch
                    ),
                )
            self.compile_single_lane_applied = True
        self._prepared_single_runner_cache[key] = runner
        self._prepared_single_runner_cache.move_to_end(key)
        return runner

    def _build_prefix_split(
        self,
        model: Any,
        fork_idx: int,
        graph_nodes: list[Any],
        buffer_names: tuple[str, ...],
        base_buffers: Mapping[str, Any],
    ) -> Any:
        """FX-split ``model`` at the fork block input; return artifacts or a reason.

        Everything strictly before the fork node (in the graph's topological
        node order) is partition 0 (the shared prefix); the fork node and all
        downstream compute is partition 1 (the vmapped suffix). ``split_module``
        wires the single crossing activation as submod_1's lone placeholder, so
        the prefix computes ``fork-block input`` once and the suffix consumes
        it. Gate buffers are routed to their partition by TENSOR IDENTITY (the
        split moves, never copies, the leaf gate modules), which is robust to
        any qualified-name rewriting the splitter may apply.
        """

        from torch.fx.passes.split_module import split_module

        partition: dict[Any, int] = {}
        for index, node in enumerate(graph_nodes):
            if node.op == "placeholder":
                partition[node] = 0
            elif node.op == "output":
                partition[node] = 1
            else:
                partition[node] = 0 if index < fork_idx else 1
        split = split_module(model, model, lambda node: partition[node])
        child_names = {name for name, _ in split.named_children()}
        if child_names - {"submod_0", "submod_1"}:
            # A non-contiguous cut (e.g. a graph-level skip that spans the fork)
            # yields extra partitions; the simple two-phase handoff cannot model
            # it, so fall back to the full-vmap path.
            return "extra_partitions"
        prefix_gm = getattr(split, "submod_0", None)
        suffix_gm = getattr(split, "submod_1", None)
        if prefix_gm is None or suffix_gm is None:
            return "unexpected_partition_layout"
        prefix_inputs = [n for n in prefix_gm.graph.nodes if n.op == "placeholder"]
        if len(prefix_inputs) != 1:
            return "multi_input_prefix"
        # Any number of tensors may cross the boundary: split_module wires
        # them as submod_1 placeholders in the same order submod_0 returns
        # them, so the prefix output tuple feeds the suffix positionally.
        # Real gated ResNet graphs always cross more than one activation at
        # a block boundary (the branch-capture contract), which is why the
        # former single-handoff rule silently disabled reuse everywhere.
        gate_ids = {id(base_buffers[name]): name for name in buffer_names}
        prefix_map = {
            sub_name: gate_ids[id(tensor)]
            for sub_name, tensor in prefix_gm.named_buffers()
            if id(tensor) in gate_ids
        }
        suffix_map = {
            sub_name: gate_ids[id(tensor)]
            for sub_name, tensor in suffix_gm.named_buffers()
            if id(tensor) in gate_ids
        }
        return prefix_gm, suffix_gm, prefix_map, suffix_map

    def _prefix_reuse_split(
        self,
        compiled_graph: Any,
        buffer_names: tuple[str, ...],
        base_buffers: Mapping[str, Any],
        stacked: Mapping[str, Any],
        dense_buffers: frozenset[str] = frozenset(),
        verified_differing: frozenset[str] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        """Derive the forward prefix/suffix artifacts for the current lane-set.

        Returns ``(artifacts, "")`` when a valid fork exists, else
        ``(None, reason)``. A successful split reproduces the full-vmap graph:
        the suffix runs the identical fork-through-readout subgraph over the
        SAME differing + downstream gate buffers, fed the prefix activation.
        """

        model = compiled_graph.model
        graph = getattr(model, "graph", None)
        if graph is None:
            return None, "prefix_reuse_skipped:model_not_fx_graph"
        differing = tuple(
            name for name in stacked if (
                not _lanes_share_column(stacked[name]) if verified_differing is None
                else name in verified_differing
            )
        )
        if not differing:
            return None, "prefix_reuse_skipped:no_differing_lanes"
        suffix_token = ".route_gate"
        graph_nodes = list(graph.nodes)
        index_of = {node: index for index, node in enumerate(graph_nodes)}
        node_by_target = {
            node.target: node for node in graph_nodes if node.op == "call_module"
        }
        diff_positions: list[tuple[int, Any, str]] = []
        for name in differing:
            if not name.endswith(suffix_token):
                return None, f"prefix_reuse_skipped:unmapped_gate_buffer:{name}"
            module_path = name[: -len(suffix_token)]
            node = node_by_target.get(module_path)
            fork_module_path = module_path
            if node is None:
                # Canonical ResNet execution keeps ``GatedResidualBlock`` as
                # one FX leaf so its branch-capture contract remains intact.
                # Convolution gates inside that leaf therefore have no
                # same-name call_module node.  The clean exact fork is the
                # input of the nearest enclosing leaf: run the whole block in
                # each suffix lane, but hoist every preceding block out of K.
                # This avoids retracing or changing the scientific graph and
                # still captures the dominant late-layer prefix reuse.
                ancestors = tuple(
                    str(target)
                    for target in node_by_target
                    if isinstance(target, str)
                    and module_path.startswith(f"{target}.")
                )
                if ancestors:
                    fork_module_path = max(ancestors, key=len)
                    node = node_by_target[fork_module_path]
            if node is None:
                return None, f"prefix_reuse_skipped:missing_fx_node:{module_path}"
            diff_positions.append(
                (index_of[node], node, fork_module_path)
            )
        diff_positions.sort(key=lambda item: item[0])
        fork_idx, fork_node, fork_module_path = diff_positions[0]
        fork_input = fork_node.args[0] if fork_node.args else None
        fx_node = getattr(getattr(torch, "fx", None), "Node", None)
        if (
            fx_node is None
            or not isinstance(fork_input, fx_node)
            or fork_input.op == "placeholder"
        ):
            # The earliest differing layer consumes the network input directly:
            # there is no shared prefix to hoist, so reuse buys nothing.
            return None, "prefix_reuse_skipped:fork_at_network_input"
        cache_key = (id(model), fork_module_path)
        cached = self._prefix_split_cache.get(cache_key)
        if cached is None:
            cached = self._build_prefix_split(
                model, fork_idx, graph_nodes, buffer_names, base_buffers
            )
            self._prefix_split_cache[cache_key] = cached
        if isinstance(cached, str):
            return None, f"prefix_reuse_skipped:{cached}"
        prefix_gm, suffix_gm, prefix_map, suffix_map = cached
        suffix_originals = set(suffix_map.values())
        if not all(name in suffix_originals for name in differing):
            return None, "prefix_reuse_skipped:differing_buffer_in_prefix"
        for original in prefix_map.values():
            if (not _lanes_share_column(stacked[original]) if verified_differing is None
                    else original in verified_differing):
                return None, "prefix_reuse_skipped:prefix_buffer_not_shared"
        # Lane-invariant prefix buffers collapse to a single shared row; the
        # differing + downstream buffers keep their full [K, ...] stack.
        prefix_overrides = {
            sub_name: stacked[original][0]
            for sub_name, original in prefix_map.items()
        }
        stacked_suffix = {
            sub_name: stacked[original]
            for sub_name, original in suffix_map.items()
        }
        return (
            {
                "prefix_gm": prefix_gm,
                "suffix_gm": suffix_gm,
                "prefix_overrides": prefix_overrides,
                "stacked_suffix": stacked_suffix,
                "fork_layer_id": fork_module_path,
                "prefix_dense": all(
                    name in dense_buffers for name in prefix_map.values()
                ),
                "prefix_gate_versions": self._prefix_gate_versions(prefix_overrides),
            },
            "",
        )

    def evaluate(
        self,
        compiled_graph: Any,
        inputs: Any,
        candidate_gates: Any,
        *,
        trusted_gates: bool = False,
    ) -> Any:
        """Run candidate gate assignments; trusted_gates skips per-call gate
        validation for callers holding compiled (shape-checked, finite-by-
        construction) masks — the per-gate isfinite() forced one device sync
        per gate per forward."""
        _require_torch()
        candidates = (
            (candidate_gates,)
            if isinstance(candidate_gates, Mapping)
            else tuple(candidate_gates)
        )
        if not candidates:
            raise ValueError("candidate execution requires at least one gate assignment")
        input_device = getattr(inputs, "device", None)
        if input_device is not None:
            self.device_type = str(input_device.type)
        self.calls += 1
        self.processed_candidates += len(candidates)
        self.processed_contexts += int(getattr(inputs, "shape", (0,))[0]) * len(
            candidates
        )
        with self.profiler.phase("candidate.prepare"):
            model_key = id(compiled_graph.model)
            base_buffers = self._base_buffers_cache.get(model_key)
            if base_buffers is None:
                base_buffers = dict(compiled_graph.model.named_buffers())
                self._base_buffers_cache[model_key] = base_buffers

        def overrides(candidate: Mapping[str, Any]) -> dict[str, Any]:
            unknown = set(candidate) - set(compiled_graph.gate_buffer_names)
            if unknown:
                raise KeyError(f"unknown gate layers: {sorted(unknown)!r}")
            result: dict[str, Any] = {}
            for layer_id, buffer_name in compiled_graph.gate_buffer_names.items():
                reference = base_buffers[buffer_name]
                value = candidate.get(layer_id, reference)
                value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
                if not trusted_gates and (
                    value.shape != reference.shape or not torch.isfinite(value).all()
                ):
                    raise ValueError(f"invalid candidate gate for {layer_id!r}")
                result[buffer_name] = value
            return result

        def run_one(buffer_overrides: Mapping[str, Any]) -> Any:
            return torch.func.functional_call(
                compiled_graph.model,
                dict(buffer_overrides),
                (inputs,),
                strict=False,
            )

        with self.profiler.phase("candidate.prepare"):
            prepared = tuple(overrides(candidate) for candidate in candidates)
            extras = self._execution_weight_cache.for_lanes(compiled_graph, candidates, trusted_gates)
            for buffers, extra in zip(prepared, extras):
                buffers.update(extra)
        if len(prepared) == 1:
            self.last_fallback_reason = ""
            with self.profiler.phase("candidate.forward"):
                return run_one(prepared[0])
        stacked = {
            name: torch.stack(tuple(item[name] for item in prepared), dim=0)
            for name in compiled_graph.gate_buffer_names.values()
        }
        from .torch_execution_weights import stack_execution_buffers
        stacked.update(stack_execution_buffers(extras))
        runner = torch.vmap(run_one)
        runner = self._maybe_compile_vmap(
            runner, device=next(iter(stacked.values())).device
        )
        with self.profiler.phase("candidate.vmap_forward"):
            output = runner(stacked)
        self.last_fallback_reason = ""
        return output

    def prepare_lanes(
        self,
        compiled_graph: Any,
        lane_gate_values: Sequence[Mapping[str, Any]],
        *,
        lane_keys: Sequence[str],
        trusted_gates: bool = False,
        lane_gate_identities: (
            Sequence[Mapping[str, HostGateIdentity] | None] | None
        ) = None,
        differing_layers: Sequence[str] | None = None,
    ) -> PreparedLanes:
        """Prepare K gate lanes once for repeated ``evaluate_prepared`` calls."""

        _require_torch()
        if len(lane_gate_values) != len(lane_keys) or not lane_gate_values:
            raise ValueError("prepared lanes require aligned, nonempty gate lanes")
        if (
            lane_gate_identities is not None
            and len(lane_gate_identities) != len(lane_keys)
        ):
            raise ValueError("host gate identities must align with prepared lanes")
        cache_enabled = self.runtime_execution_profile.activation_prefix_cache_bytes > 0
        walk_graph_preparing = (differing_layers is not None or
                                (id(compiled_graph.model) in self._walk_graph_models
                                 and _walk_forward_active()))
        with self.profiler.phase("candidate.lane_prepare_build"):
            model_key = id(compiled_graph.model)
            base_buffers = self._base_buffers_cache.get(model_key)
            if base_buffers is None:
                base_buffers = dict(compiled_graph.model.named_buffers())
                self._base_buffers_cache[model_key] = base_buffers
            buffer_names = tuple(compiled_graph.gate_buffer_names.values())
            per_lane: list[dict[str, Any]] = []
            dense_buffers = set(buffer_names) if cache_enabled else set()
            for lane_index, candidate in enumerate(lane_gate_values):
                unknown = set(candidate) - set(compiled_graph.gate_buffer_names)
                if unknown:
                    raise KeyError(f"unknown gate layers: {sorted(unknown)!r}")
                overrides: dict[str, Any] = {}
                for layer_id, buffer_name in compiled_graph.gate_buffer_names.items():
                    reference = base_buffers[buffer_name]
                    source_value = candidate.get(layer_id, reference)
                    value = torch.as_tensor(
                        source_value, dtype=reference.dtype, device=reference.device
                    )
                    if not trusted_gates and (
                        value.shape != reference.shape
                        or not torch.isfinite(value).all()
                    ):
                        raise ValueError(f"invalid candidate gate for {layer_id!r}")
                    overrides[buffer_name] = value
                    if cache_enabled and buffer_name in dense_buffers:
                        identities = (
                            None
                            if lane_gate_identities is None
                            else lane_gate_identities[lane_index]
                        )
                        identity = None if identities is None else identities.get(layer_id)
                        if identity is not None:
                            dense = identity.dense and identity.matches(source_value)
                        elif value.device.type == "cpu" and lane_gate_identities is None:
                            # Direct CPU callers already hold their mask on
                            # the host. Resolve once at preparation, never at
                            # lookup/store. Device callers need host lowering
                            # metadata; no gate transfer is used as a fallback.
                            dense = bool(torch.all(value == 1))
                        else:
                            dense = False
                        if not dense:
                            dense_buffers.discard(buffer_name)
                per_lane.append(overrides)
            from .torch_execution_weights import remap_execution_buffers, stack_execution_buffers
            extras = self._execution_weight_cache.for_lanes(compiled_graph, lane_gate_values, trusted_gates)
            if len(per_lane) == 1:
                per_lane[0].update(extras[0])
                single_names = tuple(per_lane[0])
                runner = None if walk_graph_preparing else self._prepared_single_runner(
                    compiled_graph.model,
                    buffer_names=single_names,
                    device=next(iter(per_lane[0].values())).device,
                )
                return PreparedLanes(
                    lane_keys=tuple(lane_keys),
                    buffer_names=single_names,
                    single_overrides=per_lane[0],
                    stacked_overrides=None,
                    runner=runner,
                    resident_leases=(tuple(lane_gate_values) if getattr(lane_gate_values[0], "owner", None) is not None else None),
                    execution_weights_prepared=bool(extras[0]),
                )
            with torch.inference_mode(False) if cache_enabled else nullcontext():
                stacked = resident_lane_stack(
                    lane_gate_values, per_lane, compiled_graph.gate_buffer_names,
                )
                resident = stacked is not None
                verified_differing = None
                if stacked is not None:
                    owner = lane_gate_values[0].owner
                    verified_differing = frozenset(
                        compiled_graph.gate_buffer_names[layer] for layer in owner._indices
                    )
                if stacked is None:
                    stacked = {
                        name: torch.stack(tuple(item[name] for item in per_lane), dim=0)
                        for name in per_lane[0]
                    }
                if differing_layers is not None:
                    verified_differing = frozenset(
                        compiled_graph.gate_buffer_names[layer] for layer in differing_layers
                    )
                elif walk_graph_preparing and verified_differing is None:
                    # A historical/nonresident pair supplies no shared-gate
                    # premise. Its complete two-lane graph is the direct
                    # representation; selecting it needs no device equality.
                    verified_differing = frozenset(compiled_graph.gate_buffer_names.values())
            # Keep the route axis for fork selection. Effective weights are
            # execution inputs, and do not define which gate lanes differ.
            gate_stacked = dict(stacked)
            route_buffer_names = buffer_names
            weight_state = (owner._execution_weight_states.get(model_key)
                            if resident else None)
            stacked_extras = (weight_state.stacked_buffers(compiled_graph)
                              if weight_state is not None else stack_execution_buffers(extras))
            stacked = {**stacked, **stacked_extras}
            buffer_names = tuple(stacked)
            model = compiled_graph.model
            device = next(iter(stacked.values())).device
            runner = (
                None
                if walk_graph_preparing
                else self._prepared_vmap_runner(
                    model,
                    buffer_names=buffer_names,
                    lane_count=len(per_lane),
                    device=device,
                )
            )
            prefix_kwargs: dict[str, Any] = {}
            if self.forward_prefix_reuse:
                artifacts, reason = self._prefix_reuse_split(
                    compiled_graph, route_buffer_names, base_buffers, gate_stacked,
                    frozenset(dense_buffers), verified_differing,
                )
                if artifacts is not None:
                    prefix_gm = artifacts["prefix_gm"]
                    suffix_gm = artifacts["suffix_gm"]
                    artifacts["prefix_overrides"].update(
                        remap_execution_buffers(compiled_graph, prefix_gm, extras[0])
                    )
                    artifacts["prefix_gate_versions"] = self._prefix_gate_versions(artifacts["prefix_overrides"])
                    artifacts["stacked_suffix"].update(
                        remap_execution_buffers(compiled_graph, suffix_gm, stacked_extras)
                    )
                    prefix_runner = None if walk_graph_preparing else self._prepared_single_runner(
                        prefix_gm,
                        buffer_names=tuple(artifacts["prefix_overrides"]),
                        device=next(
                            iter(artifacts["stacked_suffix"].values())
                        ).device,
                    )

                    suffix_runner = self._prepared_vmap_runner(
                        suffix_gm,
                        buffer_names=tuple(
                            artifacts["stacked_suffix"]
                        ),
                        lane_count=len(per_lane),
                        device=next(
                            iter(artifacts["stacked_suffix"].values())
                        ).device,
                        compile_vmap=False,
                    )
                    self.prefix_reuse_intentional_eager_suffix_prepares += 1
                    self.prefix_reuse_prepared += 1
                    prefix_kwargs = {
                        "prefix_gm": prefix_gm,
                        "prefix_runner": prefix_runner,
                        "suffix_gm": suffix_gm,
                        "prefix_overrides": artifacts["prefix_overrides"],
                        "stacked_suffix": artifacts["stacked_suffix"],
                        "suffix_runner": suffix_runner,
                        "suffix_compile_applied": False,
                        "prefix_fork_layer_id": artifacts["fork_layer_id"],
                        "prefix_dense": artifacts["prefix_dense"],
                        "prefix_gate_versions": artifacts["prefix_gate_versions"],
                    }
                else:
                    prefix_kwargs = {"prefix_fallback_reason": reason}
                    self.last_prefix_fallback_reason = reason
            return PreparedLanes(
                lane_keys=tuple(lane_keys),
                buffer_names=buffer_names,
                single_overrides=None,
                stacked_overrides=stacked,
                runner=runner,
                resident_leases=(tuple(lane_gate_values) if resident else None),
                execution_weights_prepared=bool(extras[0]),
                **prefix_kwargs,
            )

    def evaluate_prepared(
        self,
        compiled_graph: Any,
        prepared: PreparedLanes,
        inputs: Any,
        *,
        context_ids: Sequence[str] | None = None,
        input_identity_hash: str | None = None,
    ) -> Any:
        """Execute the prepared graph schedule for every lane: [K, batch, ...]."""

        _require_torch()
        leases = prepared.resident_leases
        if leases is not None and any(not lease._live() for lease in leases):
            leases[0].owner.historical_generation_prepares += 1
            prepared = self.prepare_lanes(
                compiled_graph,
                tuple(lease.execution_gates(lease.epoch) for lease in leases),
                lane_keys=prepared.lane_keys,
                trusted_gates=True,
            )
        input_device = getattr(inputs, "device", None)
        if input_device is not None:
            self.device_type = str(input_device.type)
        self.calls += 1
        self.processed_candidates += prepared.lane_count
        self.processed_contexts += int(getattr(inputs, "shape", (0,))[0]) * prepared.lane_count
        self.prefix_reuse_applied = False
        self.last_fallback_reason = ""

        if (getattr(input_device, "type", None) == "cuda"
                and id(compiled_graph.model) in self._walk_graph_models
                and _walk_forward_active()):
            with self.profiler.phase("candidate.walk_graph_forward"):
                return self._walk_graph_forward(compiled_graph, prepared, inputs)

        if prepared.single_overrides is not None:
            with self.profiler.phase("candidate.forward"):
                if self.runtime_execution_profile.cuda_graphs and input_device.type == "cuda":
                    output = self._cuda_graph_forward(compiled_graph, prepared.single_overrides, inputs)
                else:
                    if prepared.runner is None:
                        prepared.runner = self._prepared_single_runner(
                            compiled_graph.model, buffer_names=prepared.buffer_names,
                            device=input_device,
                        )
                    output = prepared.runner(prepared.single_overrides, inputs)
                return _output_tree_unsqueeze_lane(output)

        if prepared.suffix_runner is not None:
            self._activation_prefix_eligible(prepared)
        if prepared.suffix_runner is not None:
            prefix_cache_key = self._activation_prefix_key(
                compiled_graph, prepared, inputs,
                context_ids=context_ids, input_identity_hash=input_identity_hash,
            )
            prefix_activation = self._activation_prefix_lookup(prefix_cache_key, prepared)
            prefix_cache_hit = prefix_activation is not None
            if prefix_activation is None:
                with self.profiler.phase("candidate.prefix_forward"):
                    prefix_activation = prepared.prefix_runner(prepared.prefix_overrides, inputs)
            with self.profiler.phase("candidate.vmap_suffix_forward"):
                output = prepared.suffix_runner(prepared.stacked_suffix, prefix_activation)
            if prefix_cache_key is not None and not prefix_cache_hit:
                self._activation_prefix_store(prefix_cache_key, prepared, prefix_activation)
            self.prefix_reuse_applied = True
            self.prefix_reuse_forwards += 1
            return output

        assert prepared.stacked_overrides is not None
        with self.profiler.phase("candidate.vmap_forward"):
            if prepared.runner is None:
                prepared.runner = self._prepared_vmap_runner(
                    compiled_graph.model, buffer_names=prepared.buffer_names,
                    lane_count=prepared.lane_count, device=input_device,
                )
            return prepared.runner(prepared.stacked_overrides, inputs)

    def runtime_profile(self, *, wait_for_device: bool = True) -> BackendProfile:
        """Return runtime-only candidate telemetry without scientific identity."""

        snapshot = self.profiler.snapshot(wait=wait_for_device)
        from mechanistic_systems.measurement.run_products import effective_settings
        settings = effective_settings(resolved={
            "CIRCUIT_FORWARD_PREFIX_REUSE": "1" if self.forward_prefix_reuse else "0",
        })
        return BackendProfile(
            backend_id=self.backend_id,
            capability="candidate_execution",
            implementation=(
                "torch_func_vmap_compile"
                if self.compile_vmap_applied
                else "torch_func_vmap"
            ),
            device_type=self.device_type,
            numerical_mode="float32_native_model",
            calls=self.calls,
            wall_time_s=float(sum(snapshot["phase_time_s"].values())),
            processed_contexts=self.processed_contexts,
            fallback_reason=self.last_fallback_reason,
            phase_time_s=snapshot["phase_time_s"],
            cuda_phase_time_s=snapshot.get("cuda_phase_time_s", {}),
            cuda_phase_calls=snapshot.get("cuda_phase_calls", {}),
            metadata={
                "processed_candidates": self.processed_candidates,
                "effective_settings": settings,
                "phase_calls": snapshot["phase_calls"],
                "compile_vmap_requested": self.compile_vmap,
                "compile_vmap_applied": self.compile_vmap_applied,
                "compile_single_lane_requested": bool(
                    self.runtime_execution_profile.k1_compile
                ),
                "compile_dynamic_batch": bool(
                    self.runtime_execution_profile.compile_dynamic_batch
                ),
                "compile_single_lane_applied": bool(
                    self.compile_single_lane_applied
                ),
                "prepared_single_runner_cache_entries": len(
                    self._prepared_single_runner_cache
                ),
                "prepared_single_runner_cache_hits": int(
                    self.prepared_single_runner_cache_hits
                ),
                "prepared_single_runner_cache_misses": int(
                    self.prepared_single_runner_cache_misses
                ),
                "prepared_vmap_runner_cache_entries": len(
                    self._prepared_vmap_runner_cache
                ),
                "prepared_vmap_runner_cache_hits": int(
                    self.prepared_vmap_runner_cache_hits
                ),
                "prepared_vmap_runner_cache_misses": int(
                    self.prepared_vmap_runner_cache_misses
                ),
                "walk_reduce_overhead_requested": bool(
                    walk_reduce_overhead_requested()
                ),
                "walk_reduce_overhead_vmap_runners": int(
                    self.walk_reduce_overhead_vmap_runners
                ),
                "walk_reduce_overhead_single_runners": int(
                    self.walk_reduce_overhead_single_runners
                ),
                "walk_graph_captures": self.walk_graph_captures,
                "walk_graph_replays": self.walk_graph_replays,
                "walk_graph_buffer_copies": self.walk_graph_buffer_copies,
                "walk_graph_entries": len(self._walk_graph_cache),
                "walk_graph_compile_dynamic": False,
                "walk_graph_prefix_suffix_compiled_together": True,
                "walk_graph_static_buffer_bytes": sum(
                    value.numel() * value.element_size()
                    for entry in self._walk_graph_buffers.values()
                    for value in entry["buffers"].values()
                ),
                "forward_prefix_suffix_vmap_policy": "intentional_eager_vmap",
                "forward_prefix_suffix_compile_applied": False,
                "forward_prefix_intentional_eager_suffix_prepares": int(
                    self.prefix_reuse_intentional_eager_suffix_prepares
                ),
                "forward_prefix_reuse_requested": self.forward_prefix_reuse,
                "forward_prefix_reuse_prepared": int(self.prefix_reuse_prepared),
                "forward_prefix_reuse_forwards": int(self.prefix_reuse_forwards),
                "forward_prefix_last_fallback_reason": str(
                    self.last_prefix_fallback_reason
                )[:400],
                "activation_prefix_cache_enabled": bool(
                    True
                ),
                "activation_prefix_cache_byte_limit": int(
                    process_device_memory_budget_bytes(
                        self.runtime_execution_profile.activation_prefix_cache_bytes
                    )
                ),
                "activation_prefix_cache_resident_bytes": int(
                    self._activation_prefix_cache_bytes
                ),
                "activation_prefix_cache_entries": len(
                    self._activation_prefix_cache
                ),
                "activation_prefix_cache_hits": int(
                    self.activation_prefix_cache_hits
                ),
                "activation_prefix_cache_misses": int(
                    self.activation_prefix_cache_misses
                ),
                "activation_prefix_cache_invalidations": int(
                    self.activation_prefix_cache_invalidations
                ),
                "activation_prefix_saved_forwards": int(
                    self.activation_prefix_saved_forwards
                ),
                "activation_prefix_cache_fallbacks": dict(
                    self.activation_prefix_cache_fallbacks
                ),
                "runtime_execution_profile": self.runtime_execution_profile.to_runtime_dict(),
                "scientific_identity": "excluded",
            },
        )


def prefix_reuse_backend(
    backend: Any,
) -> TorchCandidateExecutionBackend:
    """Enable structural prefix reuse before physical execution begins."""

    if type(backend) is not TorchCandidateExecutionBackend:
        raise TypeError("prefix reuse requires TorchCandidateExecutionBackend")
    if backend.forward_prefix_reuse:
        return backend
    result = TorchCandidateExecutionBackend(
        runtime_execution_profile=backend.runtime_execution_profile,
        compile_vmap=bool(backend.compile_vmap),
        forward_prefix_reuse=True,
    )
    result.carry_worker_walk_graphs(backend)
    return result


class TorchContractionBackend:
    backend_id = "mechanistic_systems.torch_contraction.v1"

    def gram(self, left: Any, right: Any | None = None, **options: Any) -> Any:
        _require_torch()
        lhs = torch.as_tensor(left, device=options.get("device", None), dtype=torch.float32)
        rhs = lhs if right is None else torch.as_tensor(
            right, device=lhs.device, dtype=torch.float32
        )
        if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[1]:
            raise ValueError("Gram contraction requires [item,feature] matrices")
        return lhs @ rhs.T


__all__ = [
    "DirectTorchConvRouteReducerBackend",
    "TorchCandidateExecutionBackend",
    "TorchContractionBackend",
    "TorchConvKernelWeightRouteReducerBackend",
    "TorchRouteReducerBackend",
    "prefix_reuse_backend",
    "conv_channel_route_measurement_target_batched",
    "conv_channel_route_measurement_direct_target_batched",
    "conv_channel_route_measurement_direct_with_context_target_batched",
    "conv_kernel_route_measurement_per_image",
    "conv_kernel_route_measurement_target_batched",
    "conv_kernel_route_measurement_with_context_target_batched",
    "linear_route_measurement_target_batched",
]
