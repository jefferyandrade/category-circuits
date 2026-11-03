"""Source-derived AlexNet authority for the modern-v1 native work fibers.

The generic native-fiber builder deliberately accepts typed source proofs.  A
production caller must not manufacture those proofs from layer labels.  This
module is the AlexNet-specific lowering boundary: it re-binds the exact live
``TorchCircuitEvaluator`` to the canonical torchvision AlexNet declaration,
checks the executable route graph, derives the immediate-activation registry,
and content-addresses the concrete activation/scoring/reducer implementations
that make the proofs true.

Convolutional ReLU gates remain site-local.  The channel-level *route-addressed
fiber* is lawful because a kernel parameter is shared over sites and the
registered reducer contracts its signed cotangent-weighted site contributions
before any magnitude is taken.  This module never claims a shared spatial
activation gate.
"""

from __future__ import annotations

import hashlib
import inspect
from enum import Enum
from typing import Any, Final

from contracts.hashing import stable_digest
from mechanistic_systems.backends import torch_runtime as _torch_runtime
from mechanistic_systems.circuits import torch_evaluator as _torch_evaluator
from mechanistic_systems.core import architectures as _architectures
from mechanistic_systems.core import numerics as _numerics
from mechanistic_systems.core import operators as _operators
from mechanistic_systems.core.storage import RouteBlockSpec
from mechanistic_systems.measurement import optimized_irg_providers as _providers
from mechanistic_systems.measurement import torch_adapter as _torch_adapter

from . import modern_v1_conserving_attribution as _conserving_attribution
from . import modern_v1_conserving_route_score as _conserving_score
from .modern_v1_native_fiber_axis import (
    _CAPTURE_ISSUANCE_TOKEN,
    V1NativeDestinationActivationProof,
    V1NativeDestinationFiberAxisReceipt,
    V1SignedSpatialCotangentReducerProof,
    build_v1_native_destination_fiber_axis,
    v1_native_activation_registry_hash,
    v1_signed_spatial_reducer_registry_hash,
)
from .modern_v1_neutral_runtime import DeclaredRouteAxis

try:  # Optional at import; required by the callable below.
    from torchvision.models.alexnet import AlexNet as _TorchvisionAlexNet
except (ImportError, ModuleNotFoundError):  # pragma: no cover - environment dependent.
    _TorchvisionAlexNet = None  # type: ignore[assignment,misc]


ALEXNET_NATIVE_FIBER_BINDER_SCHEMA: Final = (
    "partition_free_v1_alexnet_native_destination_fiber_source_binder_v1"
)
ALEXNET_NATIVE_FIBER_SOURCE_SCHEMA: Final = (
    "partition_free_v1_alexnet_native_fiber_exact_python_source_v1"
)
ALEXNET_CONV_SPATIAL_GEOMETRY_SCHEMA: Final = (
    "partition_free_v1_alexnet_conv_signed_site_contraction_geometry_v1"
)

_ALEXNET_FAMILY: Final = "alexnet_conv_kernel_weight"
_ALEXNET_ARCHITECTURE_ID: Final = (
    "torchvision.alexnet.conv_kernel_weight.v1"
)
_ALEXNET_ROUTE_PROFILE_ID: Final = (
    "classic_target_circuit_conv_kernel_weight_v1"
)
_ALEXNET_EVALUATOR_ID: Final = "torch-explicit-gate-graph-evaluator-v2"
_ALEXNET_FIXED_READOUT_PATH: Final = "classifier.6"

_ALEXNET_LAYER_IDS: Final = (
    "features.stage1.conv",
    "features.stage2.conv",
    "features.stage3.conv",
    "features.stage4.conv",
    "features.stage5.conv",
    "readout.hidden1",
    "readout.hidden2",
)
_ALEXNET_AFFINE_PATHS: Final = (
    "features.0",
    "features.3",
    "features.6",
    "features.8",
    "features.10",
    "classifier.1",
    "classifier.4",
)
_ALEXNET_ACTIVATION_PATHS: Final = (
    "features.1",
    "features.4",
    "features.7",
    "features.9",
    "features.11",
    "classifier.2",
    "classifier.5",
)
_ALEXNET_ROUTE_KINDS: Final = (
    "conv_kernel_weight",
    "conv_kernel_weight",
    "conv_kernel_weight",
    "conv_kernel_weight",
    "conv_kernel_weight",
    "linear_weight",
    "linear_weight",
)
_ALEXNET_ROUTE_SHAPES: Final = (
    (64, 3, 11, 11),
    (192, 64, 5, 5),
    (384, 192, 3, 3),
    (256, 384, 3, 3),
    (256, 256, 3, 3),
    (4096, 9216),
    (4096, 4096),
)
_ALEXNET_ROUTE_COUNT: Final = 56_994_496

# Freeze the imported objects once.  Source changes across a package build are
# content-addressed below; in-process monkeypatching of the registered path is
# refused rather than silently minted as a new authority.
_EXPECTED_CANONICAL_SPEC = _architectures.canonical_torch_architecture_spec
_EXPECTED_ROUTE_DECLARATIONS = _architectures._canonical_route_declarations
_EXPECTED_ARCHITECTURE_BIND = _architectures.CanonicalTorchArchitectureSpec.bind
_EXPECTED_BOUND_ARCHITECTURE = _architectures.BoundTorchArchitecture
_EXPECTED_ACTIVATION_REGISTRY = (
    _providers.alexnet_v1_conserving_activation_bindings
)
_EXPECTED_ACTIVATION_BINDING = _torch_adapter.TorchConservingActivationBinding
_EXPECTED_CAPTURE_ONE = _torch_adapter.TorchModuleMeasurementAdapter._capture_one
_EXPECTED_CAPTURE_IMMEDIATE = (
    _torch_adapter.TorchModuleMeasurementAdapter._capture_immediate_activation
)
_EXPECTED_CAPTURE_BATCHES = (
    _torch_adapter.TorchModuleMeasurementAdapter.iter_captured_batches
)
_EXPECTED_CONSERVING_CAPTURE = _conserving_score.V1ConservingActivationCapture
_EXPECTED_CONSERVING_SCORE = (
    _conserving_score.score_captured_conserving_route_blocks
)
_EXPECTED_CONSERVING_COTANGENT = (
    _conserving_attribution.conserving_relu_cotangent
)
_EXPECTED_RELU_PATH_FRACTION = (
    _conserving_attribution.relu_path_open_fraction
)
_EXPECTED_EXACT_BATCHED_VJP = _numerics.exact_batched_vjp
_EXPECTED_REDUCE_ROUTE_GRADIENTS = _numerics.reduce_route_gradients
_EXPECTED_CONV_OPERATOR = _operators.ConvKernelWeightGateOperator
_EXPECTED_CONV_OPERATOR_INSTANCE = _operators.ROUTE_GATE_OPERATORS.get(
    "conv_kernel_weight"
)
_EXPECTED_CONV_PER_CONTEXT = _torch_runtime._conv_kernel_per_context
_EXPECTED_CONV_BLOCK_SIZE = _torch_runtime._conv_kernel_block_size
_EXPECTED_TORCH_FETCH = _torch_runtime._fetch
_EXPECTED_POSITIVE_SCALE = _torch_runtime._positive_scale
_EXPECTED_CONV_PER_IMAGE = (
    _torch_runtime.conv_kernel_route_measurement_per_image
)
_EXPECTED_NUMERICS_ROUTE_OPERATOR_DISPATCH = _numerics.route_gate_operator
_EXPECTED_ROUTE_OPERATOR_DISPATCH = _operators.route_gate_operator
_EXPECTED_TORCHVISION_ALEXNET = _TorchvisionAlexNet
_EXPECTED_TORCHVISION_ALEXNET_FORWARD = (
    None if _TorchvisionAlexNet is None else _TorchvisionAlexNet.forward
)


class V1AlexNetNativeFiberRefusalCode(str, Enum):
    """Typed reasons an AlexNet source axis cannot issue fiber authority."""

    INVALID_EVALUATOR = "invalid_evaluator"
    SOURCE_TAMPER = "source_tamper"
    ARCHITECTURE_MISMATCH = "architecture_mismatch"
    ROUTE_AXIS_MISMATCH = "route_axis_mismatch"
    PARITY_UNCERTIFIED = "parity_uncertified"
    ACTIVATION_REGISTRY_MISMATCH = "activation_registry_mismatch"
    REDUCER_SOURCE_MISMATCH = "reducer_source_mismatch"


class V1AlexNetNativeFiberRefusal(ValueError):
    """Fail closed instead of inferring an AlexNet fiber from a label."""

    def __init__(
        self,
        code: V1AlexNetNativeFiberRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1AlexNetNativeFiberRefusalCode,
    message: str,
) -> None:
    raise V1AlexNetNativeFiberRefusal(code, message)


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        _refuse(
            V1AlexNetNativeFiberRefusalCode.PARITY_UNCERTIFIED,
            f"{name} must be one lowercase SHA-256 digest",
        )
    result = value.lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.PARITY_UNCERTIFIED,
            f"{name} must be one lowercase SHA-256 digest",
        )
    return result


def _source_record(value: object, *, role: str) -> dict[str, str]:
    """Content-address one exact Python owner without hashing host paths."""

    try:
        source = inspect.getsource(value)
    except (OSError, TypeError) as error:
        _refuse(
            V1AlexNetNativeFiberRefusalCode.SOURCE_TAMPER,
            f"registered source for {role!r} is unavailable: {error}",
        )
    module = str(getattr(value, "__module__", ""))
    qualname = str(
        getattr(value, "__qualname__", getattr(value, "__name__", ""))
    )
    if not source or not module or not qualname:
        _refuse(
            V1AlexNetNativeFiberRefusalCode.SOURCE_TAMPER,
            f"registered source identity for {role!r} is incomplete",
        )
    return {
        "role": role,
        "module": module,
        "qualname": qualname,
        "source_utf8_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def _source_bundle_hash(
    *,
    role: str,
    sources: tuple[tuple[str, object], ...],
) -> str:
    return stable_digest(
        {
            "schema": ALEXNET_NATIVE_FIBER_SOURCE_SCHEMA,
            "role": role,
            "sources": [
                _source_record(value, role=source_role)
                for source_role, value in sources
            ],
        }
    )


def _assert_registered_source_objects() -> None:
    exact = (
        (
            _architectures.canonical_torch_architecture_spec,
            _EXPECTED_CANONICAL_SPEC,
            "canonical AlexNet architecture factory",
        ),
        (
            _architectures.CanonicalTorchArchitectureSpec.bind,
            _EXPECTED_ARCHITECTURE_BIND,
            "canonical architecture bind",
        ),
        (
            _architectures._canonical_route_declarations,
            _EXPECTED_ROUTE_DECLARATIONS,
            "canonical route declarations",
        ),
        (
            _architectures.BoundTorchArchitecture,
            _EXPECTED_BOUND_ARCHITECTURE,
            "bound architecture receipt",
        ),
        (
            _providers.alexnet_v1_conserving_activation_bindings,
            _EXPECTED_ACTIVATION_REGISTRY,
            "AlexNet activation registry",
        ),
        (
            _torch_adapter.TorchConservingActivationBinding,
            _EXPECTED_ACTIVATION_BINDING,
            "conserving activation binding",
        ),
        (
            _torch_adapter.TorchModuleMeasurementAdapter._capture_one,
            _EXPECTED_CAPTURE_ONE,
            "affine capture",
        ),
        (
            _torch_adapter.TorchModuleMeasurementAdapter._capture_immediate_activation,
            _EXPECTED_CAPTURE_IMMEDIATE,
            "immediate-activation capture",
        ),
        (
            _torch_adapter.TorchModuleMeasurementAdapter.iter_captured_batches,
            _EXPECTED_CAPTURE_BATCHES,
            "captured-batch execution",
        ),
        (
            _conserving_score.V1ConservingActivationCapture,
            _EXPECTED_CONSERVING_CAPTURE,
            "conserving activation capture receipt",
        ),
        (
            _conserving_score.score_captured_conserving_route_blocks,
            _EXPECTED_CONSERVING_SCORE,
            "conserving score path",
        ),
        (
            _conserving_attribution.conserving_relu_cotangent,
            _EXPECTED_CONSERVING_COTANGENT,
            "Aumann--Shapley ReLU cotangent",
        ),
        (
            _conserving_attribution.relu_path_open_fraction,
            _EXPECTED_RELU_PATH_FRACTION,
            "Aumann--Shapley ReLU path fraction",
        ),
        (
            _numerics.exact_batched_vjp,
            _EXPECTED_EXACT_BATCHED_VJP,
            "exact target-batched VJP",
        ),
        (
            _numerics.reduce_route_gradients,
            _EXPECTED_REDUCE_ROUTE_GRADIENTS,
            "route-gradient reducer dispatch",
        ),
        (
            _numerics.route_gate_operator,
            _EXPECTED_NUMERICS_ROUTE_OPERATOR_DISPATCH,
            "numerics route-operator dispatch binding",
        ),
        (
            _operators.route_gate_operator,
            _EXPECTED_ROUTE_OPERATOR_DISPATCH,
            "route-operator registry dispatch",
        ),
        (
            _operators.ConvKernelWeightGateOperator,
            _EXPECTED_CONV_OPERATOR,
            "conv-kernel gate operator",
        ),
        (
            _torch_runtime._conv_kernel_per_context,
            _EXPECTED_CONV_PER_CONTEXT,
            "conv-kernel context contraction",
        ),
        (
            _torch_runtime._conv_kernel_block_size,
            _EXPECTED_CONV_BLOCK_SIZE,
            "conv-kernel block sizing",
        ),
        (
            _torch_runtime._fetch,
            _EXPECTED_TORCH_FETCH,
            "conv-kernel source fetch",
        ),
        (
            _torch_runtime._positive_scale,
            _EXPECTED_POSITIVE_SCALE,
            "conv-kernel positive scale",
        ),
        (
            _torch_runtime.conv_kernel_route_measurement_per_image,
            _EXPECTED_CONV_PER_IMAGE,
            "conv-kernel per-image reducer",
        ),
    )
    changed = [name for observed, expected, name in exact if observed is not expected]
    operator = _operators.ROUTE_GATE_OPERATORS.get("conv_kernel_weight")
    if (
        changed
        or operator is not _EXPECTED_CONV_OPERATOR_INSTANCE
        or type(operator) is not _EXPECTED_CONV_OPERATOR
    ):
        coordinates = changed + (
            []
            if operator is _EXPECTED_CONV_OPERATOR_INSTANCE
            and type(operator) is _EXPECTED_CONV_OPERATOR
            else ["conv-kernel operator registry"]
        )
        _refuse(
            V1AlexNetNativeFiberRefusalCode.SOURCE_TAMPER,
            "registered implementation object changed in process: "
            + ", ".join(coordinates),
        )


def _pair(value: object) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            _refuse(
                V1AlexNetNativeFiberRefusalCode.ARCHITECTURE_MISMATCH,
                "AlexNet spatial parameter is not a pair",
            )
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _module_type_id(module: object) -> str:
    return f"{type(module).__module__}.{type(module).__qualname__}"


def _live_alexnet_graph_payload(model: Any) -> dict[str, object]:
    """Validate and serialize the exact registered torchvision AlexNet graph."""

    try:
        import torch.nn as nn
        from torchvision.models.alexnet import AlexNet
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ARCHITECTURE_MISMATCH,
            f"torchvision AlexNet is unavailable: {error}",
        )
    if (
        _EXPECTED_TORCHVISION_ALEXNET is None
        or AlexNet is not _EXPECTED_TORCHVISION_ALEXNET
        or type(model) is not _EXPECTED_TORCHVISION_ALEXNET
        or "forward" in vars(model)
    ):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ARCHITECTURE_MISMATCH,
            "source model is not the exact unmodified torchvision AlexNet class",
        )
    if (
        type(model).forward is not _EXPECTED_TORCHVISION_ALEXNET_FORWARD
        or AlexNet.forward is not _EXPECTED_TORCHVISION_ALEXNET_FORWARD
    ):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.SOURCE_TAMPER,
            "torchvision AlexNet forward implementation changed in process",
        )
    if tuple(model._modules) != ("features", "avgpool", "classifier"):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ARCHITECTURE_MISMATCH,
            "AlexNet top-level module order differs from the registered graph",
        )
    if type(model.features) is not nn.Sequential or type(model.classifier) is not nn.Sequential:
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ARCHITECTURE_MISMATCH,
            "AlexNet feature/classifier containers must be exact Sequential modules",
        )
    expected_feature_types = (
        nn.Conv2d,
        nn.ReLU,
        nn.MaxPool2d,
        nn.Conv2d,
        nn.ReLU,
        nn.MaxPool2d,
        nn.Conv2d,
        nn.ReLU,
        nn.Conv2d,
        nn.ReLU,
        nn.Conv2d,
        nn.ReLU,
        nn.MaxPool2d,
    )
    expected_classifier_types = (
        nn.Dropout,
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
    )
    if (
        tuple(type(module) for module in model.features) != expected_feature_types
        or tuple(type(module) for module in model.classifier)
        != expected_classifier_types
    ):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ARCHITECTURE_MISMATCH,
            "AlexNet child module types/order differ from the registered graph",
        )
    if (
        type(model.avgpool) is not nn.AdaptiveAvgPool2d
        or tuple(model.avgpool.output_size) != (6, 6)
    ):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ARCHITECTURE_MISMATCH,
            "AlexNet adaptive-pool geometry differs from the registered graph",
        )

    convolution_expectations = (
        (0, 3, 64, (11, 11), (4, 4), (2, 2)),
        (3, 64, 192, (5, 5), (1, 1), (2, 2)),
        (6, 192, 384, (3, 3), (1, 1), (1, 1)),
        (8, 384, 256, (3, 3), (1, 1), (1, 1)),
        (10, 256, 256, (3, 3), (1, 1), (1, 1)),
    )
    convolution_rows: list[dict[str, object]] = []
    for index, in_channels, out_channels, kernel, stride, padding in convolution_expectations:
        module = model.features[index]
        observed = (
            int(module.in_channels),
            int(module.out_channels),
            _pair(module.kernel_size),
            _pair(module.stride),
            _pair(module.padding),
            _pair(module.dilation),
            int(module.groups),
            module.bias is not None,
        )
        expected = (
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            (1, 1),
            1,
            True,
        )
        if observed != expected:
            _refuse(
                V1AlexNetNativeFiberRefusalCode.ARCHITECTURE_MISMATCH,
                f"AlexNet convolution features.{index} geometry differs",
            )
        convolution_rows.append(
            {
                "module_path": f"features.{index}",
                "module_type": _module_type_id(module),
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel,
                "stride": stride,
                "padding": padding,
                "dilation": (1, 1),
                "groups": 1,
                "bias": True,
            }
        )

    linear_expectations = (
        (1, 9216, 4096),
        (4, 4096, 4096),
        (6, 4096, 1000),
    )
    linear_rows: list[dict[str, object]] = []
    for index, in_features, out_features in linear_expectations:
        module = model.classifier[index]
        if (
            int(module.in_features) != in_features
            or int(module.out_features) != out_features
            or module.bias is None
        ):
            _refuse(
                V1AlexNetNativeFiberRefusalCode.ARCHITECTURE_MISMATCH,
                f"AlexNet linear classifier.{index} geometry differs",
            )
        linear_rows.append(
            {
                "module_path": f"classifier.{index}",
                "module_type": _module_type_id(module),
                "in_features": in_features,
                "out_features": out_features,
                "bias": True,
            }
        )
    relu_modules = (
        *tuple(model.features[index] for index in (1, 4, 7, 9, 11)),
        model.classifier[2],
        model.classifier[5],
    )
    if any(module.inplace is not True for module in relu_modules) or any(
        float(model.classifier[index].p) != 0.5
        or model.classifier[index].inplace is not False
        for index in (0, 3)
    ):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ARCHITECTURE_MISMATCH,
            "AlexNet ReLU/Dropout declarations differ from torchvision defaults",
        )
    return {
        "schema": "torchvision_alexnet_exact_live_graph_v1",
        "model_type": _module_type_id(model),
        "top_level_module_order": tuple(model._modules),
        "feature_child_types": tuple(
            _module_type_id(module) for module in model.features
        ),
        "classifier_child_types": tuple(
            _module_type_id(module) for module in model.classifier
        ),
        "convolutions": convolution_rows,
        "linears": linear_rows,
        "adaptive_pool_output_size": (6, 6),
        "relu_inplace": True,
        "dropout_probability": 0.5,
        "dropout_inplace": False,
    }


def _binding_payload(binding: object) -> dict[str, object]:
    return {
        "layer_id": str(getattr(binding, "layer_id", "")),
        "module_path": str(getattr(binding, "module_path", "")),
        "route_kind": str(getattr(binding, "route_kind", "")),
        "capture_policy": str(getattr(binding, "capture_policy", "")),
        "circuit_role": str(getattr(binding, "circuit_role", "")),
    }


def _validate_structural_evaluator_and_axis(
    evaluator: object,
    route_axis: object,
) -> tuple[Any, Any, tuple[RouteBlockSpec, ...], dict[str, object]]:
    if type(evaluator) is not _torch_evaluator.TorchCircuitEvaluator:
        _refuse(
            V1AlexNetNativeFiberRefusalCode.INVALID_EVALUATOR,
            "binder requires an exact live TorchCircuitEvaluator",
        )
    if type(route_axis) is not DeclaredRouteAxis:
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ROUTE_AXIS_MISMATCH,
            "binder requires an exact DeclaredRouteAxis",
        )
    if (
        getattr(evaluator, "evaluator_id", "") != _ALEXNET_EVALUATOR_ID
    ):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.INVALID_EVALUATOR,
            "evaluator lacks the registered execution identity/certification",
        )
    parity_hash = _sha256(
        getattr(evaluator, "parity_receipt_hash", ""),
        name="execution parity receipt hash",
    )
    del parity_hash
    model = getattr(evaluator, "source_model", None)
    live_graph = _live_alexnet_graph_payload(model)
    spec = _architectures.canonical_torch_architecture_spec(_ALEXNET_FAMILY)
    if (
        spec.architecture_id != _ALEXNET_ARCHITECTURE_ID
        or spec.family != _ALEXNET_FAMILY
        or spec.route_profile_id != _ALEXNET_ROUTE_PROFILE_ID
    ):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ARCHITECTURE_MISMATCH,
            "canonical AlexNet declaration identity changed",
        )
    bound = spec.bind(model)
    bound.validate()
    blocks = tuple(getattr(evaluator, "route_blocks", ()))
    bindings = tuple(getattr(evaluator, "mask_bindings", ()))
    expected_blocks = bound.circuit_route_blocks
    expected_bindings = bound.circuit_bindings
    expected_binding_axis = tuple(
        (item.layer_id, item.module_path, item.route_kind)
        for item in expected_bindings
    )
    observed_binding_axis = tuple(
        (item.layer_id, item.module_path, item.route_kind)
        for item in bindings
        if type(item) is _torch_evaluator.TorchMaskBinding
    )
    if (
        len(bindings) != len(expected_bindings)
        or len(observed_binding_axis) != len(bindings)
        or observed_binding_axis != expected_binding_axis
        or blocks != expected_blocks
        or getattr(evaluator, "topology_hash", "") != bound.topology_hash
        or getattr(evaluator, "route_schema_hash", "")
        != bound.circuit_route_schema_hash
        or getattr(evaluator, "readout_module_path", None)
        != _ALEXNET_FIXED_READOUT_PATH
    ):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ARCHITECTURE_MISMATCH,
            "evaluator blocks/bindings/topology differ from live canonical AlexNet",
        )
    observed_route_facts = tuple(
        (block.layer_id, block.route_kind, tuple(block.shape)) for block in blocks
    )
    if (
        tuple(item[0] for item in observed_route_facts) != _ALEXNET_LAYER_IDS
        or tuple(item[1] for item in observed_route_facts) != _ALEXNET_ROUTE_KINDS
        or tuple(item[2] for item in observed_route_facts) != _ALEXNET_ROUTE_SHAPES
        or blocks[-1].stop != _ALEXNET_ROUTE_COUNT
    ):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ARCHITECTURE_MISMATCH,
            "evaluator route shapes differ from the 56,994,496-route AlexNet axis",
        )
    graph = getattr(evaluator, "gate_graph", None)
    if (
        graph is None
        or getattr(graph, "model", None) is not getattr(evaluator, "model", None)
        or getattr(graph, "architecture_id", "") != bound.architecture_id
        or getattr(graph, "topology_hash", "") != bound.topology_hash
        or getattr(graph, "route_schema_hash", "")
        != bound.circuit_route_schema_hash
        or tuple(getattr(graph, "gates_by_layer", {})) != _ALEXNET_LAYER_IDS
    ):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.INVALID_EVALUATOR,
            "compiled evaluator gate graph differs from its declared AlexNet axis",
        )
    if (
        route_axis.route_blocks != blocks
        or route_axis.route_axis_hash != bound.circuit_route_schema_hash
        or route_axis.route_count != _ALEXNET_ROUTE_COUNT
    ):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ROUTE_AXIS_MISMATCH,
            "declared route axis differs from the exact evaluator route axis",
        )
    return model, bound, blocks, live_graph


def _validate_evaluator_and_axis(evaluator: object, route_axis: object):
    """Legacy execution admission remains closed without its global certificate."""
    if getattr(evaluator, "claim_execution_certified", False) is not True:
        _refuse(V1AlexNetNativeFiberRefusalCode.INVALID_EVALUATOR,
                "evaluator lacks the registered execution identity/certification")
    return _validate_structural_evaluator_and_axis(evaluator, route_axis)


def _activation_source_bundle_hash(model_type: type[Any]) -> str:
    return _source_bundle_hash(
        role="alexnet_immediate_activation_capture",
        sources=(
            ("torchvision AlexNet class", model_type),
            ("torchvision AlexNet forward", model_type.forward),
            ("AlexNet activation registry", _EXPECTED_ACTIVATION_REGISTRY),
            ("activation binding contract", _EXPECTED_ACTIVATION_BINDING),
            ("affine capture hook", _EXPECTED_CAPTURE_ONE),
            ("immediate activation hook", _EXPECTED_CAPTURE_IMMEDIATE),
            ("captured batch execution", _EXPECTED_CAPTURE_BATCHES),
            ("activation capture receipt", _EXPECTED_CONSERVING_CAPTURE),
        ),
    )


def _reducer_source_bundle_hash() -> str:
    return _source_bundle_hash(
        role="alexnet_signed_conv_site_contraction_before_magnitude",
        sources=(
            ("conserving score path", _EXPECTED_CONSERVING_SCORE),
            ("Aumann--Shapley cotangent", _EXPECTED_CONSERVING_COTANGENT),
            ("ReLU path fraction", _EXPECTED_RELU_PATH_FRACTION),
            ("exact target-batched VJP", _EXPECTED_EXACT_BATCHED_VJP),
            ("route-gradient reducer dispatch", _EXPECTED_REDUCE_ROUTE_GRADIENTS),
            ("route-operator registry dispatch", _EXPECTED_ROUTE_OPERATOR_DISPATCH),
            ("conv-kernel route operator", _EXPECTED_CONV_OPERATOR),
            ("conv-kernel block sizing", _EXPECTED_CONV_BLOCK_SIZE),
            ("conv-kernel source fetch", _EXPECTED_TORCH_FETCH),
            ("conv-kernel positive scale", _EXPECTED_POSITIVE_SCALE),
            ("conv-kernel per-image reducer", _EXPECTED_CONV_PER_IMAGE),
            ("conv-kernel context contraction", _EXPECTED_CONV_PER_CONTEXT),
        ),
    )


def _architecture_source_bundle_hash(model_type: type[Any]) -> str:
    return _source_bundle_hash(
        role="alexnet_canonical_architecture_and_route_binding",
        sources=(
            ("torchvision AlexNet class", model_type),
            ("canonical architecture factory", _EXPECTED_CANONICAL_SPEC),
            ("canonical route declarations", _EXPECTED_ROUTE_DECLARATIONS),
            ("canonical architecture bind", _EXPECTED_ARCHITECTURE_BIND),
            ("bound architecture receipt", _EXPECTED_BOUND_ARCHITECTURE),
        ),
    )


def _activation_proofs(
    *,
    evaluator: Any,
    blocks: tuple[RouteBlockSpec, ...],
    activation_source_bundle_hash: str,
) -> tuple[V1NativeDestinationActivationProof, ...]:
    try:
        activation_bindings = tuple(
            _providers.alexnet_v1_conserving_activation_bindings(
                evaluator.mask_bindings
            )
        )
    except (TypeError, ValueError) as error:
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ACTIVATION_REGISTRY_MISMATCH,
            f"AlexNet activation registry refused the evaluator: {error}",
        )
    observed = tuple(
        (
            item.layer_id,
            item.affine_module_path,
            item.activation_kind,
            item.activation_module_path,
        )
        for item in activation_bindings
        if type(item) is _EXPECTED_ACTIVATION_BINDING
    )
    expected = tuple(
        (layer, affine, "relu", activation)
        for layer, affine, activation in zip(
            _ALEXNET_LAYER_IDS,
            _ALEXNET_AFFINE_PATHS,
            _ALEXNET_ACTIVATION_PATHS,
            strict=True,
        )
    )
    if len(observed) != len(activation_bindings) or observed != expected:
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ACTIVATION_REGISTRY_MISMATCH,
            "activation registry differs from the seven exact AlexNet cuts",
        )
    module_ids: list[int] = []
    proofs: list[V1NativeDestinationActivationProof] = []
    for block, binding in zip(blocks, activation_bindings, strict=True):
        affine = evaluator.source_model.get_submodule(binding.affine_module_path)
        assert binding.activation_module_path is not None
        activation = evaluator.source_model.get_submodule(
            binding.activation_module_path
        )
        affine_parent_path, affine_child = binding.affine_module_path.rsplit(".", 1)
        activation_parent_path, activation_child = (
            binding.activation_module_path.rsplit(".", 1)
        )
        if (
            affine_parent_path != activation_parent_path
            or not affine_child.isdecimal()
            or not activation_child.isdecimal()
            or int(activation_child) != int(affine_child) + 1
        ):
            _refuse(
                V1AlexNetNativeFiberRefusalCode.ACTIVATION_REGISTRY_MISMATCH,
                f"activation for {block.layer_id!r} is not the next Sequential child",
            )
        parent = evaluator.source_model.get_submodule(affine_parent_path)
        children = tuple(parent._modules.items())
        affine_index = int(affine_child)
        activation_index = int(activation_child)
        if (
            affine_index >= len(children)
            or activation_index >= len(children)
            or children[affine_index] != (affine_child, affine)
            or children[activation_index] != (activation_child, activation)
        ):
            _refuse(
                V1AlexNetNativeFiberRefusalCode.ACTIVATION_REGISTRY_MISMATCH,
                f"activation container order for {block.layer_id!r} is stale",
            )
        module_ids.extend((id(affine), id(activation)))
        block_hash = stable_digest(block.to_dict())
        activation_content_hash = stable_digest(
            {
                "schema": "alexnet_native_destination_activation_source_v1",
                "implementation_bundle_hash": activation_source_bundle_hash,
                "layer_id": block.layer_id,
                "route_block_content_hash": block_hash,
                "affine_module_path": binding.affine_module_path,
                "affine_module_type": _module_type_id(affine),
                "activation_kind": binding.activation_kind,
                "activation_module_path": binding.activation_module_path,
                "activation_module_type": _module_type_id(activation),
                "sequential_parent_path": affine_parent_path,
                "affine_child_index": affine_index,
                "activation_child_index": activation_index,
                "runtime_identity_check": (
                    "arguments_zero_is_exact_affine_downstream_tensor_and_"
                    "invocation_ordinal_is_affine_plus_one_v1"
                ),
            }
        )
        proofs.append(
            V1NativeDestinationActivationProof(
                layer_id=block.layer_id,
                route_block_content_hash=block_hash,
                affine_module_path=binding.affine_module_path,
                activation_kind=binding.activation_kind,
                activation_module_path=binding.activation_module_path,
                affine_occurrence_ordinal=0,
                affine_occurrence_count=1,
                immediate_consumer_tensor_identity_certified=True,
                execution_parity_receipt_hash=evaluator.parity_receipt_hash,
                activation_source_content_hash=activation_content_hash,
            )
        )
    if len(set(module_ids)) != len(module_ids):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ACTIVATION_REGISTRY_MISMATCH,
            "AlexNet affine/activation paths share a module occurrence",
        )
    return tuple(proofs)


def _spatial_reducer_proofs(
    *,
    evaluator: Any,
    blocks: tuple[RouteBlockSpec, ...],
    reducer_source_bundle_hash: str,
) -> tuple[V1SignedSpatialCotangentReducerProof, ...]:
    proofs: list[V1SignedSpatialCotangentReducerProof] = []
    for block, binding in zip(blocks, evaluator.mask_bindings, strict=True):
        if block.route_kind != "conv_kernel_weight":
            continue
        module = evaluator.source_model.get_submodule(binding.module_path)
        spatial_geometry_hash = stable_digest(
            {
                "schema": ALEXNET_CONV_SPATIAL_GEOMETRY_SCHEMA,
                "layer_id": block.layer_id,
                "route_block": block.to_dict(),
                "affine_module_path": binding.module_path,
                "affine_module_type": _module_type_id(module),
                "weight_shape": tuple(int(value) for value in module.weight.shape),
                "in_channels": int(module.in_channels),
                "out_channels": int(module.out_channels),
                "groups": int(module.groups),
                "kernel_size": _pair(module.kernel_size),
                "stride": _pair(module.stride),
                "padding": _pair(module.padding),
                "dilation": _pair(module.dilation),
                "route_parameter_shared_across_spatial_sites": True,
                "site_local_activation_gates": True,
                "signed_cotangent_weighted_site_sum_before_magnitude": True,
                "context_axis_retained_until_after_spatial_contraction": True,
            }
        )
        proofs.append(
            V1SignedSpatialCotangentReducerProof(
                layer_id=block.layer_id,
                route_block_content_hash=stable_digest(block.to_dict()),
                affine_module_path=binding.module_path,
                route_kind="conv_kernel_weight",
                route_parameter_shared_across_spatial_sites=True,
                signed_site_contributions_summed_before_magnitude=True,
                sitewise_gate_axis_materialized=False,
                reducer_implementation_hash=reducer_source_bundle_hash,
                spatial_geometry_content_hash=spatial_geometry_hash,
                execution_parity_receipt_hash=evaluator.parity_receipt_hash,
            )
        )
    if len(proofs) != 5:
        _refuse(
            V1AlexNetNativeFiberRefusalCode.REDUCER_SOURCE_MISMATCH,
            "AlexNet requires five exact convolution spatial-reducer proofs",
        )
    return tuple(proofs)


def _build_alexnet_native_axis(
    evaluator: object,
    route_axis: object,
    *,
    structural_capture: tuple[str, str] | None = None,
) -> V1NativeDestinationFiberAxisReceipt:
    """Issue the compact native-fiber axis from one live AlexNet evaluator.

    Required arguments are the registered ``TorchCircuitEvaluator`` and its
    exact ``DeclaredRouteAxis``.  Every other identity is derived from those
    sources and the imported implementation owners; callers cannot supply a
    layer map, proof flag, source hash, architecture hash, or parity override.
    """

    _assert_registered_source_objects()
    validate = (_validate_evaluator_and_axis if structural_capture is None
                else _validate_structural_evaluator_and_axis)
    model, bound, blocks, live_graph = validate(evaluator, route_axis)
    architecture_source_hash = _architecture_source_bundle_hash(type(model))
    activation_source_hash = _activation_source_bundle_hash(type(model))
    reducer_source_hash = _reducer_source_bundle_hash()
    source_architecture_schema_hash = stable_digest(
        {
            "schema": ALEXNET_NATIVE_FIBER_BINDER_SCHEMA,
            "architecture_binding_schema": (
                _architectures.ARCHITECTURE_BINDING_SCHEMA
            ),
            "architecture_id": _ALEXNET_ARCHITECTURE_ID,
            "family": _ALEXNET_FAMILY,
            "route_profile_id": _ALEXNET_ROUTE_PROFILE_ID,
            "layer_ids": _ALEXNET_LAYER_IDS,
            "affine_module_paths": _ALEXNET_AFFINE_PATHS,
            "activation_module_paths": _ALEXNET_ACTIVATION_PATHS,
            "route_kinds": _ALEXNET_ROUTE_KINDS,
            "route_shapes": _ALEXNET_ROUTE_SHAPES,
            "route_count": _ALEXNET_ROUTE_COUNT,
            "architecture_source_bundle_hash": architecture_source_hash,
            "activation_source_bundle_hash": activation_source_hash,
            "reducer_source_bundle_hash": reducer_source_hash,
        }
    )
    source_architecture_content_hash = stable_digest(
        {
            "schema": "alexnet_native_fiber_live_architecture_content_v1",
            "architecture_schema_hash": source_architecture_schema_hash,
            "architecture_id": bound.architecture_id,
            "family": bound.spec.family,
            "route_profile_id": bound.spec.route_profile_id,
            "topology_hash": bound.topology_hash,
            "full_route_schema_hash": bound.route_schema_hash,
            "circuit_route_schema_hash": bound.circuit_route_schema_hash,
            "full_bindings": [
                _binding_payload(binding) for binding in bound.bindings
            ],
            "full_route_blocks": [
                block.to_dict() for block in bound.route_blocks
            ],
            "circuit_route_blocks": [block.to_dict() for block in blocks],
            "live_graph": live_graph,
            "evaluator_id": evaluator.evaluator_id,
            "compiled_gate_graph_identity": {
                "architecture_id": evaluator.gate_graph.architecture_id,
                "topology_hash": evaluator.gate_graph.topology_hash,
                "route_schema_hash": evaluator.gate_graph.route_schema_hash,
                "layer_axis": tuple(evaluator.gate_graph.gates_by_layer),
            },
        }
    )
    activation_proofs = _activation_proofs(
        evaluator=evaluator,
        blocks=blocks,
        activation_source_bundle_hash=activation_source_hash,
    )
    reducer_proofs = _spatial_reducer_proofs(
        evaluator=evaluator,
        blocks=blocks,
        reducer_source_bundle_hash=reducer_source_hash,
    )
    receipt = build_v1_native_destination_fiber_axis(
        route_axis,
        source_route_blocks=blocks,
        activation_proofs=activation_proofs,
        spatial_reducer_proofs=reducer_proofs,
        source_architecture_schema_hash=source_architecture_schema_hash,
        source_architecture_content_hash=source_architecture_content_hash,
        execution_parity_receipt_hash=evaluator.parity_receipt_hash,
        _structural_capture=structural_capture,
        _capture_issuance_token=_CAPTURE_ISSUANCE_TOKEN,
        activation_registry_hash=v1_native_activation_registry_hash(
            activation_proofs
        ),
        spatial_reducer_registry_hash=(
            v1_signed_spatial_reducer_registry_hash(reducer_proofs)
        ),
    )
    if (
        receipt.architecture_id != _ALEXNET_ARCHITECTURE_ID
        or receipt.topology_hash != evaluator.topology_hash
        or receipt.route_axis_hash != route_axis.route_axis_hash
        or receipt.route_count != _ALEXNET_ROUTE_COUNT
        or receipt.cut_count != 7
        or receipt.fiber_count != 9_344
    ):
        _refuse(
            V1AlexNetNativeFiberRefusalCode.ROUTE_AXIS_MISMATCH,
            "issued native-fiber receipt does not close the AlexNet source axis",
        )
    receipt.assert_integrity()
    return receipt


def build_alexnet_v1_native_destination_fiber_axis(evaluator, route_axis):
    """Legacy public binder; an opaque parity reference cannot grant admission."""
    return _build_alexnet_native_axis(evaluator, route_axis)


__all__ = [
    "ALEXNET_CONV_SPATIAL_GEOMETRY_SCHEMA",
    "ALEXNET_NATIVE_FIBER_BINDER_SCHEMA",
    "ALEXNET_NATIVE_FIBER_SOURCE_SCHEMA",
    "V1AlexNetNativeFiberRefusal",
    "V1AlexNetNativeFiberRefusalCode",
    "build_alexnet_v1_native_destination_fiber_axis",
]
