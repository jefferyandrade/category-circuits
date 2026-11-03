"""Universal executable route-gate operators.

Architectures declare where a route lives.  These operators alone define what
the route gate means and how its identity-gate derivative is reduced.  The
optimized production reducer and the coordinate-level reference calculation
are intentionally independent implementations of the same estimand.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

try:  # Optional for package import smoke; required by measurement execution.
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


RouteReducerKind = Literal[
    "linear_weight",
    "conv_channel_pair",
    "conv_kernel_weight",
    "residual_stream_gate",
]


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("route-gate operators require PyTorch")


def _reference_gate_operands(output: Any, alpha: Any) -> tuple[Any, Any]:
    """Promote reference-gate arithmetic without changing production numerics."""

    if torch.is_tensor(alpha):
        dtype = torch.promote_types(output.dtype, alpha.dtype)
        alpha_tensor = alpha.to(device=output.device, dtype=dtype)
    else:
        dtype = output.dtype
        alpha_tensor = torch.tensor(float(alpha), device=output.device, dtype=dtype)
    return output.to(dtype=dtype), alpha_tensor


@dataclass(frozen=True)
class RouteGateDescription:
    route_kind: str
    gate_coordinate: str
    derivative_estimand: str
    bias_gated: bool = False


class RouteGateOperator(ABC):
    """One architecture-neutral executable route-gate definition."""

    route_kind: RouteReducerKind

    @property
    @abstractmethod
    def description(self) -> RouteGateDescription:
        ...

    @abstractmethod
    def validate_capture(self, captured: Any) -> None:
        ...

    @abstractmethod
    def reduce_vjp(self, captured: Any, gradients: Any) -> tuple[Any, Any]:
        """Return signed image sums and sums of per-image absolute effects."""

    def reduce_vjp_per_image(
        self, captured: Any, gradients: Any
    ) -> Any:
        """Return per-image contributions [context, query, *route_shape]."""

        raise NotImplementedError(
            f"route kind {self.route_kind!r} has no per-image reducer yet; "
            "per-image retention refuses rather than approximates"
        )

    @abstractmethod
    def apply_reference_gate(
        self,
        captured: Any,
        coordinate: tuple[int, ...],
        alpha: Any,
    ) -> Any:
        """Apply one scalar gate to the captured local route output exactly."""

    @abstractmethod
    def reference_coordinate_per_context(
        self,
        captured: Any,
        gradients: Any,
        coordinate: tuple[int, ...],
    ) -> Any:
        """Independent chain-rule contribution with shape ``[query,context]``."""

    def reference_coordinate_scores(
        self,
        captured: Any,
        gradients: Any,
        coordinate: tuple[int, ...],
    ) -> tuple[Any, Any]:
        per_context = self.reference_coordinate_per_context(
            captured,
            gradients,
            coordinate,
        ).to(torch.float32)
        return per_context.sum(dim=1), per_context.abs().sum(dim=1)

    @abstractmethod
    def local_target_contribution_per_context(self, captured: Any) -> Any:
        """Return the route's signed contribution at its local destination.

        The result has shape ``[context, *route_shape]``.  Spatial destination
        coordinates are reduced by the route-family contract, so convolutional
        channel pairs and residual streams each have one scalar local
        attribution per route and context.  Convolutional channel-pair routes
        use a signed spatial sum before any absolute value is taken.
        A downstream affine transport (for example a following BatchNorm in
        evaluation mode) is intentionally applied by the architecture adapter,
        not by this architecture-neutral operator.
        """

    @staticmethod
    def _validate_coordinate(shape: tuple[int, ...], coordinate: tuple[int, ...]) -> None:
        if len(shape) != len(coordinate) or any(
            index < 0 or index >= extent
            for index, extent in zip(coordinate, shape, strict=True)
        ):
            raise IndexError(f"route coordinate {coordinate!r} is outside shape {shape!r}")


class LinearEdgeGateOperator(RouteGateOperator):
    route_kind = "linear_weight"
    description = RouteGateDescription(
        route_kind=route_kind,
        gate_coordinate="one output-by-input weight edge",
        derivative_estimand="weight[o,i] * input[i] * dF/doutput[o]",
    )

    def validate_capture(self, captured: Any) -> None:
        _require_torch()
        if not torch.is_tensor(captured.output) or captured.output.ndim != 2:
            raise ValueError("linear route output must be [context,out_feature]")
        if not torch.is_tensor(captured.input_source) or captured.input_source.ndim != 2:
            raise ValueError("linear route input must be [context,in_feature]")
        if not torch.is_tensor(captured.weight) or captured.weight.ndim != 2:
            raise ValueError("linear route weight must be [out_feature,in_feature]")
        expected = (int(captured.weight.shape[0]), int(captured.weight.shape[1]))
        if tuple(captured.route_block.shape) != expected:
            raise ValueError("linear route axis differs from its executable weight")

    def reduce_vjp(self, captured: Any, gradients: Any) -> tuple[Any, Any]:
        from mechanistic_systems.backends.torch_runtime import (
            linear_route_measurement_target_batched,
        )

        return linear_route_measurement_target_batched(
            captured.input_source,
            gradients,
            weights=captured.weight,
            scale=captured.scale,
            device=captured.output.device,
            abs_aggregation_mode=captured.abs_aggregation_mode,
        )

    def reduce_vjp_per_image(
        self, captured: Any, gradients: Any
    ) -> Any:
        """Per-context linear-route outer products using the scorer's blocking."""

        from mechanistic_systems.backends.torch_runtime import (
            _linear_block_size,
        )

        self.validate_capture(captured)
        inputs = captured.input_source
        device = captured.output.device
        gradients = gradients.to(device=device, dtype=torch.float32)
        inputs = inputs.to(device=device, dtype=torch.float32)
        weight = captured.weight.to(device=device, dtype=torch.float32)
        query_count, batch_count, output_count = gradients.shape
        input_count = int(inputs.shape[1])
        scale = float(captured.scale)
        if abs(scale) < 1e-12:
            scale = 1.0
        scale_tensor = torch.as_tensor(
            scale, device=device, dtype=torch.float32
        )
        rows = torch.empty(
            (batch_count, query_count, output_count, input_count),
            device=device,
            dtype=torch.float32,
        )
        block = _linear_block_size(
            int(batch_count), int(input_count), int(output_count)
        )
        for start in range(0, int(batch_count), int(block)):
            end = min(int(batch_count), start + int(block))
            source = inputs[start:end]
            gradient = gradients[:, start:end] / scale_tensor
            rows[start:end] = (
                torch.einsum("qbo,bi->bqoi", gradient, source)
                * weight.reshape(1, 1, output_count, input_count)
            )
        return rows

    def local_target_contribution_per_context(self, captured: Any) -> Any:
        self.validate_capture(captured)
        return (
            captured.input_source.to(torch.float32).unsqueeze(1)
            * captured.weight.to(torch.float32).unsqueeze(0)
            * float(captured.scale)
        )

    def apply_reference_gate(
        self,
        captured: Any,
        coordinate: tuple[int, ...],
        alpha: Any,
    ) -> Any:
        self.validate_capture(captured)
        self._validate_coordinate(tuple(captured.route_block.shape), coordinate)
        out_index, in_index = coordinate
        output, alpha_tensor = _reference_gate_operands(captured.output, alpha)
        result = output.clone()
        contribution = (
            captured.input_source[:, in_index].to(dtype=output.dtype)
            * captured.weight[out_index, in_index].to(dtype=output.dtype)
            * float(captured.scale)
        )
        result[:, out_index] = (
            result[:, out_index] + (alpha_tensor - 1.0) * contribution
        )
        return result

    def reference_coordinate_per_context(
        self,
        captured: Any,
        gradients: Any,
        coordinate: tuple[int, ...],
    ) -> Any:
        self.validate_capture(captured)
        self._validate_coordinate(tuple(captured.route_block.shape), coordinate)
        out_index, in_index = coordinate
        return (
            gradients[:, :, out_index].to(torch.float32)
            * captured.input_source[:, in_index].to(torch.float32).unsqueeze(0)
            * captured.weight[out_index, in_index].to(torch.float32)
            * float(captured.scale)
        )


class ConvChannelPairGateOperator(RouteGateOperator):
    route_kind = "conv_channel_pair"
    description = RouteGateDescription(
        route_kind=route_kind,
        gate_coordinate="all kernel coefficients for one grouped input-output channel pair",
        derivative_estimand=(
            "sum_kernel_and_spatial(preadd_route_contribution * dF/dconv_output)"
        ),
    )

    def validate_capture(self, captured: Any) -> None:
        _require_torch()
        if not torch.is_tensor(captured.output) or captured.output.ndim != 4:
            raise ValueError("conv route output must be [context,out_channel,height,width]")
        if not torch.is_tensor(captured.input_source) or captured.input_source.ndim != 4:
            raise ValueError("conv route input must be [context,in_channel,height,width]")
        if not torch.is_tensor(captured.weight) or captured.weight.ndim != 4:
            raise ValueError("conv route weight must be [out,in_per_group,kernel_h,kernel_w]")
        required = {"stride", "padding", "dilation", "groups"}
        missing = sorted(required - set(captured.reducer_parameters))
        if missing:
            raise ValueError("conv route reducer parameters missing: " + ",".join(missing))
        groups = int(captured.reducer_parameters["groups"])
        if groups <= 0 or int(captured.input_source.shape[1]) % groups:
            raise ValueError("conv route capture has invalid grouped-channel geometry")
        expected = (int(captured.weight.shape[0]), int(captured.weight.shape[1]))
        if tuple(captured.route_block.shape) != expected:
            raise ValueError("conv route axis differs from its executable grouped weight")

    def reduce_vjp(self, captured: Any, gradients: Any) -> tuple[Any, Any]:
        from mechanistic_systems.backends.torch_runtime import (
            conv_channel_route_measurement_target_batched,
        )

        parameters = captured.reducer_parameters
        return conv_channel_route_measurement_target_batched(
            captured.input_source,
            gradients,
            weights=captured.weight,
            scale=captured.scale,
            stride=parameters["stride"],
            padding=parameters["padding"],
            dilation=parameters["dilation"],
            groups=int(parameters["groups"]),
            device=captured.output.device,
            abs_aggregation_mode=captured.abs_aggregation_mode,
        )

    def reduce_vjp_per_image(
        self, captured: Any, gradients: Any
    ) -> Any:
        from mechanistic_systems.backends.torch_runtime import (
            conv_channel_route_measurement_per_image,
        )

        parameters = captured.reducer_parameters
        return conv_channel_route_measurement_per_image(
            captured.input_source,
            gradients,
            weights=captured.weight,
            scale=captured.scale,
            stride=parameters["stride"],
            padding=parameters["padding"],
            dilation=parameters["dilation"],
            groups=int(parameters["groups"]),
            device=captured.output.device,
        )

    def local_target_contribution_per_context(self, captured: Any) -> Any:
        """Exact contextwise spatially summed contribution of every channel pair.

        ``unfold`` materializes input patches once for the entire route block.
        The contraction then evaluates all output-input channel pairs together;
        no coordinate loop or replacement autograd path is introduced.
        """

        self.validate_capture(captured)
        parameters = captured.reducer_parameters
        source = captured.input_source.to(torch.float32)
        weight = captured.weight.to(torch.float32)
        patches = F.unfold(
            source,
            kernel_size=tuple(int(value) for value in weight.shape[-2:]),
            dilation=parameters["dilation"],
            padding=parameters["padding"],
            stride=parameters["stride"],
        )
        batch_count = int(source.shape[0])
        groups = int(parameters["groups"])
        input_per_group = int(weight.shape[1])
        output_count = int(weight.shape[0])
        if output_count % groups:
            raise ValueError("convolutional local attribution has invalid groups")
        output_per_group = output_count // groups
        kernel_elements = int(weight.shape[2] * weight.shape[3])
        patch_sum = patches.reshape(
            batch_count,
            groups,
            input_per_group,
            kernel_elements,
            -1,
        ).sum(dim=-1)
        grouped_weight = weight.reshape(
            groups,
            output_per_group,
            input_per_group,
            kernel_elements,
        )
        return (
            torch.einsum("bgik,goik->bgoi", patch_sum, grouped_weight)
            .reshape(batch_count, output_count, input_per_group)
            .mul(float(captured.scale))
        )

    def _route_activation(self, captured: Any, coordinate: tuple[int, ...]) -> Any:
        self.validate_capture(captured)
        self._validate_coordinate(tuple(captured.route_block.shape), coordinate)
        out_index, local_in_index = coordinate
        groups = int(captured.reducer_parameters["groups"])
        out_per_group = int(captured.weight.shape[0]) // groups
        in_per_group = int(captured.weight.shape[1])
        group_index = out_index // out_per_group
        global_in_index = group_index * in_per_group + local_in_index
        route_input = captured.input_source[:, global_in_index : global_in_index + 1]
        route_weight = captured.weight[
            out_index : out_index + 1,
            local_in_index : local_in_index + 1,
        ]
        parameters = captured.reducer_parameters
        return F.conv2d(
            route_input,
            route_weight,
            bias=None,
            stride=parameters["stride"],
            padding=parameters["padding"],
            dilation=parameters["dilation"],
            groups=1,
        )[:, 0] * float(captured.scale)

    def apply_reference_gate(
        self,
        captured: Any,
        coordinate: tuple[int, ...],
        alpha: Any,
    ) -> Any:
        route_activation = self._route_activation(captured, coordinate)
        out_index = coordinate[0]
        output, alpha_tensor = _reference_gate_operands(captured.output, alpha)
        result = output.clone()
        result[:, out_index] = result[:, out_index] + (
            alpha_tensor - 1.0
        ) * route_activation.to(dtype=output.dtype)
        return result

    def reference_coordinate_per_context(
        self,
        captured: Any,
        gradients: Any,
        coordinate: tuple[int, ...],
    ) -> Any:
        route_activation = self._route_activation(captured, coordinate).to(torch.float32)
        out_index = coordinate[0]
        return (
            gradients[:, :, out_index].to(torch.float32)
            * route_activation.unsqueeze(0)
        ).flatten(start_dim=2).sum(dim=2)


class ConvKernelWeightGateOperator(RouteGateOperator):
    """Per-tap granularity: one gate per (o, i, kh, kw) kernel tap.

    The per-tap attribution on context ``x`` is exactly the channel-pair
    contraction WITHOUT the final kernel-tap summation:

        weight[o,i,kh,kw] * sum_p dF/dconv_output[o,p]
                                  * padded_input[i, p_stride_offset(kh,kw)]

    so the channel-pair attribution equals the sum of its taps (the mandated
    tap-sum consistency identity). Per-sample weight autograd is explicitly
    declined here — it would break the shared output-VJP batching.
    """

    route_kind = "conv_kernel_weight"
    description = RouteGateDescription(
        route_kind=route_kind,
        gate_coordinate="one kernel tap of one grouped input-output channel pair",
        derivative_estimand=(
            "sum_spatial(per_tap_route_contribution * dF/dconv_output)"
        ),
    )

    def validate_capture(self, captured: Any) -> None:
        _require_torch()
        if not torch.is_tensor(captured.output) or captured.output.ndim != 4:
            raise ValueError("conv route output must be [context,out_channel,height,width]")
        if not torch.is_tensor(captured.input_source) or captured.input_source.ndim != 4:
            raise ValueError("conv route input must be [context,in_channel,height,width]")
        if not torch.is_tensor(captured.weight) or captured.weight.ndim != 4:
            raise ValueError("conv route weight must be [out,in_per_group,kernel_h,kernel_w]")
        required = {"stride", "padding", "dilation", "groups"}
        missing = sorted(required - set(captured.reducer_parameters))
        if missing:
            raise ValueError("conv route reducer parameters missing: " + ",".join(missing))
        groups = int(captured.reducer_parameters["groups"])
        if groups <= 0 or int(captured.input_source.shape[1]) % groups:
            raise ValueError("conv route capture has invalid grouped-channel geometry")
        expected = tuple(int(value) for value in captured.weight.shape)
        if tuple(captured.route_block.shape) != expected:
            raise ValueError(
                "conv kernel route axis differs from its executable grouped weight taps"
            )

    def reduce_vjp(self, captured: Any, gradients: Any) -> tuple[Any, Any]:
        from mechanistic_systems.backends.torch_runtime import (
            conv_kernel_route_measurement_target_batched,
        )

        parameters = captured.reducer_parameters
        return conv_kernel_route_measurement_target_batched(
            captured.input_source,
            gradients,
            weights=captured.weight,
            scale=captured.scale,
            stride=parameters["stride"],
            padding=parameters["padding"],
            dilation=parameters["dilation"],
            groups=int(parameters["groups"]),
            device=captured.output.device,
            abs_aggregation_mode=captured.abs_aggregation_mode,
        )

    def reduce_vjp_per_image(
        self, captured: Any, gradients: Any
    ) -> Any:
        from mechanistic_systems.backends.torch_runtime import (
            conv_kernel_route_measurement_per_image,
        )

        parameters = captured.reducer_parameters
        return conv_kernel_route_measurement_per_image(
            captured.input_source,
            gradients,
            weights=captured.weight,
            scale=captured.scale,
            stride=parameters["stride"],
            padding=parameters["padding"],
            dilation=parameters["dilation"],
            groups=int(parameters["groups"]),
            device=captured.output.device,
        )

    def local_target_contribution_per_context(self, captured: Any) -> Any:
        """Exact contextwise spatially summed contribution of every kernel tap.

        Identical to the channel-pair local contraction except the kernel-tap
        axis is RETAINED: taps are never flattened into the source
        axis, and summing the result over ``(KH, KW)`` recovers the
        channel-pair local contribution.
        """

        self.validate_capture(captured)
        parameters = captured.reducer_parameters
        source = captured.input_source.to(torch.float32)
        weight = captured.weight.to(torch.float32)
        patches = F.unfold(
            source,
            kernel_size=tuple(int(value) for value in weight.shape[-2:]),
            dilation=parameters["dilation"],
            padding=parameters["padding"],
            stride=parameters["stride"],
        )
        batch_count = int(source.shape[0])
        groups = int(parameters["groups"])
        input_per_group = int(weight.shape[1])
        output_count = int(weight.shape[0])
        if output_count % groups:
            raise ValueError("convolutional local attribution has invalid groups")
        output_per_group = output_count // groups
        kernel_h, kernel_w = (int(value) for value in weight.shape[-2:])
        kernel_elements = kernel_h * kernel_w
        patch_sum = patches.reshape(
            batch_count,
            groups,
            input_per_group,
            kernel_elements,
            -1,
        ).sum(dim=-1)
        grouped_weight = weight.reshape(
            groups,
            output_per_group,
            input_per_group,
            kernel_elements,
        )
        return (
            (patch_sum.unsqueeze(2) * grouped_weight.unsqueeze(0))
            .reshape(
                batch_count,
                output_count,
                input_per_group,
                kernel_h,
                kernel_w,
            )
            .mul(float(captured.scale))
        )

    def _tap_activation(self, captured: Any, coordinate: tuple[int, ...]) -> Any:
        self.validate_capture(captured)
        self._validate_coordinate(tuple(captured.route_block.shape), coordinate)
        out_index, local_in_index, kernel_row, kernel_column = coordinate
        groups = int(captured.reducer_parameters["groups"])
        out_per_group = int(captured.weight.shape[0]) // groups
        in_per_group = int(captured.weight.shape[1])
        group_index = out_index // out_per_group
        global_in_index = group_index * in_per_group + local_in_index
        route_input = captured.input_source[:, global_in_index : global_in_index + 1]
        route_weight = torch.zeros_like(
            captured.weight[
                out_index : out_index + 1,
                local_in_index : local_in_index + 1,
            ]
        )
        route_weight[0, 0, kernel_row, kernel_column] = captured.weight[
            out_index, local_in_index, kernel_row, kernel_column
        ]
        parameters = captured.reducer_parameters
        return F.conv2d(
            route_input,
            route_weight,
            bias=None,
            stride=parameters["stride"],
            padding=parameters["padding"],
            dilation=parameters["dilation"],
            groups=1,
        )[:, 0] * float(captured.scale)

    def apply_reference_gate(
        self,
        captured: Any,
        coordinate: tuple[int, ...],
        alpha: Any,
    ) -> Any:
        tap_activation = self._tap_activation(captured, coordinate)
        out_index = coordinate[0]
        output, alpha_tensor = _reference_gate_operands(captured.output, alpha)
        result = output.clone()
        result[:, out_index] = result[:, out_index] + (
            alpha_tensor - 1.0
        ) * tap_activation.to(dtype=output.dtype)
        return result

    def reference_coordinate_per_context(
        self,
        captured: Any,
        gradients: Any,
        coordinate: tuple[int, ...],
    ) -> Any:
        tap_activation = self._tap_activation(captured, coordinate).to(torch.float32)
        out_index = coordinate[0]
        return (
            gradients[:, :, out_index].to(torch.float32)
            * tap_activation.unsqueeze(0)
        ).flatten(start_dim=2).sum(dim=2)


class ResidualStreamGateOperator(RouteGateOperator):
    route_kind = "residual_stream_gate"
    description = RouteGateDescription(
        route_kind=route_kind,
        gate_coordinate="main or skip branch immediately before residual addition",
        derivative_estimand="inner_product(premerge_branch, dF/dpremerge_branch)",
    )

    def validate_capture(self, captured: Any) -> None:
        _require_torch()
        if (
            not torch.is_tensor(captured.output)
            or captured.output.ndim < 3
            or int(captured.output.shape[1]) != 2
            or tuple(captured.route_block.shape) != (2,)
        ):
            raise ValueError(
                "residual stream output must be [context,main_or_skip,...] with two routes"
            )

    def _per_context(self, captured: Any, gradients: Any) -> Any:
        contribution = (
            captured.output.unsqueeze(0).to(torch.float32)
            * gradients.to(torch.float32)
            * float(captured.scale)
        )
        return contribution.reshape(
            int(gradients.shape[0]),
            int(captured.output.shape[0]),
            2,
            -1,
        ).sum(dim=-1)

    def reduce_vjp(self, captured: Any, gradients: Any) -> tuple[Any, Any]:
        per_context = self._per_context(captured, gradients)
        return per_context.sum(dim=1), per_context.abs().sum(dim=1)

    def reduce_vjp_per_image(
        self, captured: Any, gradients: Any
    ) -> Any:
        return self._per_context(captured, gradients).transpose(0, 1)

    def local_target_contribution_per_context(self, captured: Any) -> Any:
        self.validate_capture(captured)
        return (
            captured.output.to(torch.float32)
            .flatten(start_dim=2)
            .mean(dim=2)
            .mul(float(captured.scale))
        )

    def apply_reference_gate(
        self,
        captured: Any,
        coordinate: tuple[int, ...],
        alpha: Any,
    ) -> Any:
        self.validate_capture(captured)
        self._validate_coordinate((2,), coordinate)
        branch = coordinate[0]
        output, alpha_tensor = _reference_gate_operands(captured.output, alpha)
        factors = torch.stack(
            tuple(
                1.0 + (alpha_tensor - 1.0) * float(captured.scale)
                if index == branch
                else torch.ones_like(alpha_tensor)
                for index in range(2)
            )
        )
        shape = (1, 2, *((1,) * (output.ndim - 2)))
        return output * factors.reshape(shape)

    def reference_coordinate_per_context(
        self,
        captured: Any,
        gradients: Any,
        coordinate: tuple[int, ...],
    ) -> Any:
        self.validate_capture(captured)
        self._validate_coordinate((2,), coordinate)
        branch = coordinate[0]
        return (
            captured.output[:, branch].to(torch.float32).unsqueeze(0)
            * gradients[:, :, branch].to(torch.float32)
            * float(captured.scale)
        ).flatten(start_dim=2).sum(dim=2)


ROUTE_GATE_OPERATORS: dict[RouteReducerKind, RouteGateOperator] = {
    "linear_weight": LinearEdgeGateOperator(),
    "conv_channel_pair": ConvChannelPairGateOperator(),
    "conv_kernel_weight": ConvKernelWeightGateOperator(),
    "residual_stream_gate": ResidualStreamGateOperator(),
}


def route_gate_operator(route_kind: str) -> RouteGateOperator:
    try:
        return ROUTE_GATE_OPERATORS[route_kind]  # type: ignore[index]
    except KeyError as error:
        raise ValueError(f"unsupported route-gate kind {route_kind!r}") from error


__all__ = [
    "ROUTE_GATE_OPERATORS",
    "ConvChannelPairGateOperator",
    "ConvKernelWeightGateOperator",
    "LinearEdgeGateOperator",
    "ResidualStreamGateOperator",
    "RouteGateDescription",
    "RouteGateOperator",
    "RouteReducerKind",
    "route_gate_operator",
]
