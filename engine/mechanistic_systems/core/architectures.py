"""Canonical PyTorch architecture topology and route bindings.

Framework module paths are executable aliases, not scientific coordinates.
This module freezes their lowering to stable layer IDs and derives topology
identity from the complete module/parameter shape graph without hashing model
weights. Checkpoint identity remains part of ``StudySpec``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contracts.hashing import stable_digest
from mechanistic_systems.core.storage import (
    RouteBlockSpec,
    ordered_block_route_axis_hash,
    rebase_route_blocks,
)
from mechanistic_systems.core.topology import (
    RouteAlias,
    RouteAliasResolver,
)

try:  # Optional for import smoke; required when binding a live model.
    import torch.nn as nn
except Exception:  # pragma: no cover - environment dependent.
    nn = None  # type: ignore[assignment]


ARCHITECTURE_BINDING_SCHEMA = "mechanistic_torch_architecture_binding_v1"


def _require_torch() -> None:
    if nn is None:
        raise RuntimeError("canonical PyTorch architecture binding requires PyTorch")


@dataclass(frozen=True)
class ExecutableRouteBinding:
    layer_id: str
    module_path: str
    route_kind: str
    capture_policy: str = "clone_for_downstream"
    circuit_role: str = "prunable"

    def validate(self) -> None:
        if not self.layer_id or not self.module_path:
            raise ValueError("executable route binding identity must be nonempty")
        if self.route_kind not in {
            "conv_channel_pair",
            "conv_kernel_weight",
            "linear_weight",
            "residual_stream_gate",
        }:
            raise ValueError(f"unsupported executable route kind {self.route_kind!r}")
        if self.capture_policy not in {
            "clone_for_downstream",
            "direct_nonmutating",
            "residual_premerge_decomposition",
        }:
            raise ValueError(f"unsupported certified capture policy {self.capture_policy!r}")
        if self.circuit_role not in {"prunable", "fixed_readout"}:
            raise ValueError(f"unsupported circuit role {self.circuit_role!r}")
        if self.circuit_role == "fixed_readout" and self.route_kind != "linear_weight":
            raise ValueError("fixed readout bindings must be linear-weight measurements")


def historical_route_alias(binding: ExecutableRouteBinding) -> str:
    """Return the frozen ScoreStore/profile alias for one executable route.

    Residual-stream gates are synthetic coordinates attached to a parent block,
    so their historical alias includes an explicit suffix even though execution
    hooks the parent module itself.
    """

    binding.validate()
    if binding.route_kind == "residual_stream_gate":
        return f"{binding.module_path}/residual_stream_gate"
    return binding.module_path


@dataclass(frozen=True)
class CanonicalTorchArchitectureSpec:
    architecture_id: str
    family: str
    route_profile_id: str
    bindings: tuple[ExecutableRouteBinding, ...]

    def validate(self) -> None:
        if not self.architecture_id or not self.family or not self.route_profile_id or not self.bindings:
            raise ValueError("canonical architecture specification is incomplete")
        for binding in self.bindings:
            binding.validate()
        layer_ids = tuple(item.layer_id for item in self.bindings)
        module_paths = tuple(item.module_path for item in self.bindings)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("canonical architecture contains duplicate stable layer IDs")
        if len(set(module_paths)) != len(module_paths):
            raise ValueError("canonical architecture contains duplicate executable aliases")

    @property
    def alias_resolver(self) -> RouteAliasResolver:
        self.validate()
        return RouteAliasResolver(
            tuple(RouteAlias(item.module_path, item.layer_id) for item in self.bindings)
        )

    def bind(self, model: Any) -> BoundTorchArchitecture:
        _require_torch()
        self.validate()
        if not isinstance(model, nn.Module):
            raise ValueError("canonical architecture binding requires an nn.Module")
        module_signatures: list[dict[str, Any]] = []
        for module_path, module in model.named_modules():
            signature: dict[str, Any] = {
                "module_path": module_path,
                "module_type": f"{type(module).__module__}.{type(module).__qualname__}",
            }
            for field in (
                "in_channels",
                "out_channels",
                "groups",
                "in_features",
                "out_features",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
            ):
                value = getattr(module, field, None)
                if value is not None:
                    signature[field] = list(value) if isinstance(value, tuple) else int(value)
            module_signatures.append(signature)
        parameter_shapes = [
            {"name": name, "shape": list(parameter.shape)}
            for name, parameter in model.named_parameters()
        ]
        buffer_shapes = [
            {"name": name, "shape": list(buffer.shape)}
            for name, buffer in model.named_buffers()
        ]
        topology_hash = stable_digest(
            {
                "schema_name": ARCHITECTURE_BINDING_SCHEMA,
                "architecture_id": self.architecture_id,
                "family": self.family,
                "modules": module_signatures,
                "parameters": parameter_shapes,
                "buffers": buffer_shapes,
            }
        )
        offset = 0
        route_blocks: list[RouteBlockSpec] = []
        for binding in self.bindings:
            try:
                module = model.get_submodule(binding.module_path)
            except (AttributeError, KeyError) as error:
                raise ValueError(
                    f"canonical architecture alias {binding.module_path!r} is unresolved"
                ) from error
            block_extras: dict[str, str] = {}
            if binding.route_kind == "conv_channel_pair":
                if not isinstance(module, nn.Conv2d):
                    raise ValueError(f"route alias {binding.module_path!r} is not Conv2d")
                shape = (int(module.out_channels), int(module.in_channels // module.groups))
            elif binding.route_kind == "conv_kernel_weight":
                if not isinstance(module, nn.Conv2d):
                    raise ValueError(f"route alias {binding.module_path!r} is not Conv2d")
                shape = (
                    int(module.out_channels),
                    int(module.in_channels // module.groups),
                    int(module.kernel_size[0]),
                    int(module.kernel_size[1]),
                )
                block_extras["flattening"] = "c_order_target_source_taps_v1"
            elif binding.route_kind == "linear_weight":
                if not isinstance(module, nn.Linear):
                    raise ValueError(f"route alias {binding.module_path!r} is not Linear")
                shape = (int(module.out_features), int(module.in_features))
            else:
                required = ("conv1", "bn1", "relu", "conv2", "bn2")
                if any(not hasattr(module, name) for name in required):
                    raise ValueError(f"route alias {binding.module_path!r} is not a residual block")
                shape = (2,)
            block = RouteBlockSpec(
                architecture_id=self.architecture_id,
                topology_hash=topology_hash,
                layer_id=binding.layer_id,
                route_kind=binding.route_kind,
                shape=shape,
                offset=offset,
                **block_extras,
            )
            route_blocks.append(block)
            offset = block.stop
        blocks = tuple(route_blocks)
        return BoundTorchArchitecture(
            spec=self,
            topology_hash=topology_hash,
            route_schema_hash=ordered_block_route_axis_hash(blocks),
            route_blocks=blocks,
        )


@dataclass(frozen=True)
class BoundTorchArchitecture:
    spec: CanonicalTorchArchitectureSpec
    topology_hash: str
    route_schema_hash: str
    route_blocks: tuple[RouteBlockSpec, ...]

    @property
    def architecture_id(self) -> str:
        return self.spec.architecture_id

    @property
    def bindings(self) -> tuple[ExecutableRouteBinding, ...]:
        return self.spec.bindings

    @property
    def circuit_bindings(self) -> tuple[ExecutableRouteBinding, ...]:
        """Bindings that were eligible for the historical pruning frontier."""

        return tuple(binding for binding in self.bindings if binding.circuit_role == "prunable")

    @property
    def circuit_route_blocks(self) -> tuple[RouteBlockSpec, ...]:
        selected = tuple(
            block
            for binding, block in zip(self.bindings, self.route_blocks, strict=True)
            if binding.circuit_role == "prunable"
        )
        return rebase_route_blocks(selected)

    @property
    def circuit_route_schema_hash(self) -> str:
        return ordered_block_route_axis_hash(self.circuit_route_blocks)

    @property
    def fixed_measurement_layer_ids(self) -> tuple[str, ...]:
        return tuple(
            binding.layer_id for binding in self.bindings if binding.circuit_role != "prunable"
        )

    def validate(self) -> None:
        self.spec.validate()
        if len(self.topology_hash) != 64 or len(self.route_schema_hash) != 64:
            raise ValueError("bound architecture hashes must be SHA-256 digests")
        if ordered_block_route_axis_hash(self.route_blocks) != self.route_schema_hash:
            raise ValueError("bound architecture route schema is inconsistent")
        if tuple(block.layer_id for block in self.route_blocks) != tuple(
            binding.layer_id for binding in self.bindings
        ):
            raise ValueError("bound architecture route blocks differ from stable bindings")
        if not self.circuit_bindings:
            raise ValueError("bound architecture has no prunable circuit bindings")


def _binding(
    layer_id: str,
    module_path: str,
    route_kind: str | None = None,
    *,
    circuit_role: str = "prunable",
) -> ExecutableRouteBinding:
    resolved_kind = route_kind or (
        "linear_weight"
        if module_path in {"fc", "classifier.0", "classifier.1", "classifier.3", "classifier.4", "classifier.6"}
        else "conv_channel_pair"
    )
    return ExecutableRouteBinding(
        layer_id=layer_id,
        module_path=module_path,
        route_kind=resolved_kind,
        capture_policy=(
            "residual_premerge_decomposition"
            if resolved_kind == "residual_stream_gate"
            else "clone_for_downstream"
        ),
        circuit_role=circuit_role,
    )


@dataclass(frozen=True)
class _RouteDeclaration:
    binding: ExecutableRouteBinding; stage_id: str
    block_id: str; operator_id: str


def _declaration(
    layer_id: str, module_path: str, *, stage_id: str, block_id: str,
    operator_id: str | None = None,
    route_kind: str | None = None, circuit_role: str = "prunable",
) -> _RouteDeclaration:
    return _RouteDeclaration(
        binding=_binding(layer_id, module_path, route_kind, circuit_role=circuit_role),
        stage_id=stage_id, block_id=block_id, operator_id=operator_id or layer_id)


def _resnet_declarations(
    *,
    block_counts: tuple[int, ...],
    convolutions_per_block: int,
    conv_route_kind: str | None = None,
) -> tuple[_RouteDeclaration, ...]:
    declarations = [
        _declaration(
            "stem.conv",
            "conv1",
            stage_id="stem",
            block_id="stem",
            route_kind=conv_route_kind,
        )
    ]
    for stage_index, block_count in enumerate(block_counts, start=1):
        stage_id = f"stage{stage_index}"
        for block_index in range(block_count):
            block_id = f"{stage_id}.block{block_index}"
            for convolution_index in range(1, convolutions_per_block + 1):
                layer_id = f"{block_id}.conv{convolution_index}"
                declarations.append(
                    _declaration(
                        layer_id,
                        f"layer{stage_index}.{block_index}.conv{convolution_index}",
                        stage_id=stage_id,
                        block_id=block_id,
                        route_kind=conv_route_kind,
                    )
                )
            if block_index == 0 and (stage_index > 1 or convolutions_per_block == 3):
                layer_id = f"{block_id}.projection"
                declarations.append(
                    _declaration(
                        layer_id,
                        f"layer{stage_index}.{block_index}.downsample.0",
                        stage_id=stage_id,
                        block_id=block_id,
                        route_kind=conv_route_kind,
                    )
                )
            layer_id = f"{block_id}.residual_stream"
            declarations.append(
                _declaration(
                    layer_id,
                    f"layer{stage_index}.{block_index}",
                    stage_id=stage_id,
                    block_id=block_id,
                    route_kind="residual_stream_gate",
                )
            )
    declarations.append(
        _declaration(
            "readout.logits", "fc",
            stage_id="readout",
            block_id="readout.logits",
            route_kind="linear_weight",
            circuit_role="fixed_readout",
        )
    )
    return tuple(declarations)


def _canonical_route_declarations(family: str) -> tuple[str, str, tuple[_RouteDeclaration, ...]]:
    normalized = str(family).strip().lower().replace("-", "_")
    rows: tuple[tuple[str, str, str, str, str], ...] | None = None
    # The kernel-weight granularity is a distinct family sharing
    # the alexnet topology rows; conv bindings carry conv_kernel_weight
    # and every identity key differs, so no receipt, cache, or contract
    # can pool across granularities.
    conv_route_kind = (
        "conv_kernel_weight"
        if normalized in {
            "alexnet_conv_kernel_weight",
            "vgg11_conv_kernel_weight",
        }
        else None
    )
    if normalized in {"alexnet", "alexnet_ssl_w1_mlp4096", "alexnet_conv_kernel_weight"}:
        rows = (
            ("features.stage1.conv", "features.0", "features.stage1", "features.stage1", "features.stage1.conv"),
            ("features.stage2.conv", "features.3", "features.stage2", "features.stage2", "features.stage2.conv"),
            ("features.stage3.conv", "features.6", "features.stage3", "features.stage3", "features.stage3.conv"),
            ("features.stage4.conv", "features.8", "features.stage4", "features.stage4", "features.stage4.conv"),
            ("features.stage5.conv", "features.10", "features.stage5", "features.stage5", "features.stage5.conv"),
            ("readout.hidden1", "classifier.1", "readout", "readout.hidden1", "readout.hidden1"),
            ("readout.hidden2", "classifier.4", "readout", "readout.hidden2", "readout.hidden2"),
            ("readout.logits", "classifier.6", "readout", "readout.logits", "readout.logits"),
        )
        if normalized == "alexnet":
            architecture_id = "torchvision.alexnet.v1"
            route_profile_id = "classic_target_circuit_v1"
        elif normalized == "alexnet_conv_kernel_weight":
            architecture_id = "torchvision.alexnet.conv_kernel_weight.v1"
            route_profile_id = (
                "classic_target_circuit_conv_kernel_weight_v1"
            )
        else:
            architecture_id = "visionlab.alexnet_w1_mlp4096.v1"
            route_profile_id = "ssl_alexnet_w1_mlp4096_target_circuit_v1"
    elif normalized in {"resnet18", "resnet18_conv_kernel_weight"}:
        declarations = _resnet_declarations(
            block_counts=(2, 2, 2, 2),
            convolutions_per_block=2,
            conv_route_kind=(
                "conv_kernel_weight"
                if normalized == "resnet18_conv_kernel_weight"
                else None
            ),
        )
        if normalized == "resnet18_conv_kernel_weight":
            architecture_id = "torchvision.resnet18.conv_kernel_weight.v1"
            route_profile_id = (
                "classic_residual_target_circuit_conv_kernel_weight_v1"
            )
        else:
            architecture_id = "torchvision.resnet18.v1"
            route_profile_id = "classic_residual_target_circuit_v1"
    elif normalized == "resnet50":
        declarations = _resnet_declarations(
            block_counts=(3, 4, 6, 3),
            convolutions_per_block=3,
        )
        architecture_id = "torchvision.resnet50.v1"
        route_profile_id = "classic_residual_target_circuit_v1"
    elif normalized == "resnet50_conv_kernel_weight":
        declarations = _resnet_declarations(
            block_counts=(3, 4, 6, 3),
            convolutions_per_block=3,
            conv_route_kind="conv_kernel_weight",
        )
        architecture_id = "torchvision.resnet50.conv_kernel_weight.v1"
        route_profile_id = (
            "classic_residual_target_circuit_conv_kernel_weight_v1"
        )
    elif normalized in {"vgg11", "vgg11_conv_kernel_weight"}:
        rows = (
            ("features.stage3.conv1", "features.6", "features.stage3", "features.stage3", "features.stage3.conv1"),
            ("features.stage3.conv2", "features.8", "features.stage3", "features.stage3", "features.stage3.conv2"),
            ("features.stage4.conv1", "features.11", "features.stage4", "features.stage4", "features.stage4.conv1"),
            ("features.stage4.conv2", "features.13", "features.stage4", "features.stage4", "features.stage4.conv2"),
            ("features.stage5.conv1", "features.16", "features.stage5", "features.stage5", "features.stage5.conv1"),
            ("features.stage5.conv2", "features.18", "features.stage5", "features.stage5", "features.stage5.conv2"),
            ("readout.hidden1", "classifier.0", "readout", "readout.hidden1", "readout.hidden1"),
            ("readout.hidden2", "classifier.3", "readout", "readout.hidden2", "readout.hidden2"),
            ("readout.logits", "classifier.6", "readout", "readout.logits", "readout.logits"),
        )
        if normalized == "vgg11":
            architecture_id = "torchvision.vgg11.v1"
            route_profile_id = "historical_classic_target_circuit_v1"
        else:
            architecture_id = "torchvision.vgg11.conv_kernel_weight.v1"
            route_profile_id = (
                "historical_classic_target_circuit_conv_kernel_weight_v1"
            )
    elif normalized == "vgg11_bn":
        rows = (
            ("features.stage3.conv1", "features.8", "features.stage3", "features.stage3", "features.stage3.conv1"),
            ("features.stage3.conv2", "features.11", "features.stage3", "features.stage3", "features.stage3.conv2"),
            ("features.stage4.conv1", "features.15", "features.stage4", "features.stage4", "features.stage4.conv1"),
            ("features.stage4.conv2", "features.18", "features.stage4", "features.stage4", "features.stage4.conv2"),
            ("features.stage5.conv1", "features.22", "features.stage5", "features.stage5", "features.stage5.conv1"),
            ("features.stage5.conv2", "features.25", "features.stage5", "features.stage5", "features.stage5.conv2"),
            ("readout.hidden1", "classifier.0", "readout", "readout.hidden1", "readout.hidden1"),
            ("readout.hidden2", "classifier.3", "readout", "readout.hidden2", "readout.hidden2"),
            ("readout.logits", "classifier.6", "readout", "readout.logits", "readout.logits"),
        )
        architecture_id = "torchvision.vgg11_bn.v1"
        route_profile_id = "historical_classic_target_circuit_v1"
    else:
        raise ValueError(f"unsupported canonical PyTorch architecture family {family!r}")
    if rows is not None:
        declarations = tuple(
            _declaration(
                layer_id,
                module_path,
                stage_id=stage_id,
                block_id=block_id,
                operator_id=operator_id,
                route_kind=(
                    conv_route_kind
                    if conv_route_kind is not None
                    and module_path.startswith("features.")
                    else None
                ),
                circuit_role=("fixed_readout" if layer_id == "readout.logits" else "prunable"),
            )
            for layer_id, module_path, stage_id, block_id, operator_id in rows
        )
    return architecture_id, route_profile_id, declarations


def canonical_torch_architecture_spec(family: str) -> CanonicalTorchArchitectureSpec:
    normalized = str(family).strip().lower().replace("-", "_")
    architecture_id, route_profile_id, declarations = _canonical_route_declarations(
        normalized
    )
    bindings = tuple(row.binding for row in declarations)
    spec = CanonicalTorchArchitectureSpec(
        architecture_id=architecture_id,
        family=normalized,
        route_profile_id=route_profile_id,
        bindings=bindings,
    )
    spec.validate()
    return spec


def canonical_v2_route_topology_receipt(architecture: BoundTorchArchitecture) -> Any:
    """Lower the architecture-owned declaration to the pure V2 receipt grammar."""

    from mechanistic_systems.circuits.operator_route_transport_v2 import (
        CanonicalRouteCoordinateReceipt,
        EligibleRouteBlockReceipt,
        EligibleTopologyAnnotationReceipt,
        TopologyAnnotationReceipt,
    )

    architecture.validate()
    _, _, declarations = _canonical_route_declarations(architecture.spec.family)
    if tuple(row.binding for row in declarations) != architecture.bindings:
        raise ValueError("bound architecture differs from its canonical route declaration")
    binding_hashes: list[str] = []
    annotation_hashes: list[str] = []
    eligible_blocks: list[EligibleRouteBlockReceipt] = []
    eligible_annotations: list[EligibleTopologyAnnotationReceipt] = []
    full_to_eligible: list[int] = []
    fixed_indices: list[int] = []
    eligible_offset = 0
    for full_index, (declaration, block) in enumerate(
        zip(declarations, architecture.route_blocks, strict=True)
    ):
        binding = declaration.binding
        binding_hash = stable_digest(
            {
                "schema_name": "mechanistic_production_route_binding_v1",
                "layer_id": binding.layer_id,
                "module_path": binding.module_path,
                "route_kind": binding.route_kind,
                "capture_policy": binding.capture_policy,
                "circuit_role": binding.circuit_role,
            }
        )
        annotation_hash = stable_digest(
            {
                "schema_name": "mechanistic_v2_route_topology_annotation_v1",
                "layer_id": binding.layer_id,
                "stage_id": declaration.stage_id,
                "block_id": declaration.block_id,
                "operator_id": declaration.operator_id,
                "topology_rank": full_index,
            }
        )
        binding_hashes.append(binding_hash)
        annotation_hashes.append(annotation_hash)
        if binding.circuit_role != "prunable":
            full_to_eligible.append(-1)
            fixed_indices.append(full_index)
            continue
        eligible_index = len(eligible_blocks)
        full_to_eligible.append(eligible_index)

        def coordinate(flat_index: int) -> CanonicalRouteCoordinateReceipt:
            value = block.route_coordinate(flat_index)
            return CanonicalRouteCoordinateReceipt(
                architecture_id=value.architecture_id,
                topology_hash=value.topology_hash,
                layer_id=value.layer_id,
                route_kind=value.route_kind,
                source_coordinate=value.source_coordinate,
                target_coordinate=value.target_coordinate,
                parameter_coordinate=value.parameter_coordinate,
            )

        eligible_blocks.append(
            EligibleRouteBlockReceipt(
                architecture_id=block.architecture_id,
                topology_hash=block.topology_hash,
                layer_id=block.layer_id,
                route_kind=block.route_kind,
                shape=block.shape,
                eligible_offset=eligible_offset,
                full_block_index=full_index,
                production_binding_hash=binding_hash,
                annotation_hash=annotation_hash,
                first_coordinate=coordinate(0),
                last_coordinate=coordinate(block.route_count - 1),
            )
        )
        eligible_annotations.append(
            EligibleTopologyAnnotationReceipt(
                full_block_index=full_index,
                production_binding_hash=binding_hash,
                annotation_hash=annotation_hash,
                layer_id=block.layer_id,
                stage_id=declaration.stage_id,
                block_id=declaration.block_id,
                operator_id=declaration.operator_id,
                topology_rank=full_index,
            )
        )
        eligible_offset += block.route_count
    receipt = TopologyAnnotationReceipt(
        architecture_id=architecture.architecture_id,
        topology_hash=architecture.topology_hash,
        full_binding_axis_hash=stable_digest(
            {
                "schema_name": "mechanistic_production_binding_axis_v1",
                "binding_hashes": binding_hashes,
            }
        ),
        full_route_schema_hash=architecture.route_schema_hash,
        eligible_route_axis_hash=architecture.circuit_route_schema_hash,
        full_to_eligible_block_index=tuple(full_to_eligible),
        fixed_measurement_full_block_indices=tuple(fixed_indices),
        eligible_route_blocks=tuple(eligible_blocks),
        eligible_annotations=tuple(eligible_annotations),
        full_binding_hashes=tuple(binding_hashes),
        full_annotation_hashes=tuple(annotation_hashes),
        fixed_measurement_exclusion_hash=stable_digest(
            {
                "schema_name": "mechanistic_fixed_measurement_exclusion_v1",
                "full_block_indices": fixed_indices,
                "binding_hashes": [binding_hashes[index] for index in fixed_indices],
            }
        ),
    )
    receipt.validate()
    return receipt


__all__ = [
    "ARCHITECTURE_BINDING_SCHEMA",
    "BoundTorchArchitecture",
    "CanonicalTorchArchitectureSpec",
    "ExecutableRouteBinding",
    "canonical_v2_route_topology_receipt",
    "canonical_torch_architecture_spec",
    "historical_route_alias",
]
