"""Mutation-free PyTorch evaluator for native circuit identification."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from mechanistic_systems.core.determinism import (
    configure_deterministic_backend,
    deterministic_backend_record,
)
from circuitflow.contracts import DataView
from mechanistic_systems.backends.readout_projection import (
    RuntimeReadoutProjectionSpec,
)
from mechanistic_systems.circuits.preservation import (
    AffineCalibrationFit,
    balanced_bce_damage,
    balanced_binary_bce,
    fit_balanced_affine_calibration,
)
from mechanistic_systems.circuits.search import CircuitEvaluation
from mechanistic_systems.core.storage import RouteBlockSpec
from mechanistic_systems.contracts import CircuitPlan, QuerySpec
from mechanistic_systems.measurement.queries import evaluate_query
from mechanistic_systems.measurement.torch_adapter import (
    MeasurementInputBatch,
    ResponseSelector,
    default_response_selector,
)
from mechanistic_systems.backends.torch_gate_graph import (
    GatedResidualBlock,
    TorchGateGraphCompiler,
)
from mechanistic_systems.backends.calibration_runtime import (
    RuntimeExecutionProfile,
)
from mechanistic_systems.backends.torch_runtime import (
    TorchCandidateExecutionBackend,
    _walk_forward_active,
)
from mechanistic_systems.backends.torch_masks import (
    CompiledTorchCircuitMask,
    HostGateIdentity,
    tensor_mutation_identity,
)

try:  # Optional for import smoke; required by native masked execution.
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("native torch circuit evaluation requires PyTorch")


class CircuitBatchProvider(Protocol):
    provider_id: str

    def iter_batches(
        self,
        view: DataView,
        *,
        query_id: str,
    ) -> Iterable[MeasurementInputBatch]: ...


@dataclass(frozen=True)
class TorchMaskBinding:
    layer_id: str
    module_path: str
    route_kind: str

    def validate(self) -> None:
        if not self.layer_id or not self.module_path:
            raise ValueError("torch mask binding requires canonical and module paths")
        if self.route_kind not in {
            "linear_weight",
            "conv_channel_pair",
            "conv_kernel_weight",
            "residual_stream_gate",
        }:
            raise ValueError(f"unsupported torch mask route kind {self.route_kind!r}")


@dataclass(frozen=True)
class CircuitResponse:
    """One ordered target-logit response axis for circuit identification."""

    context_ids: tuple[str, ...]
    values: np.ndarray
    labels: np.ndarray | None

    def validate(self, *, labels_required: bool = False) -> None:
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        if not self.context_ids or len(set(self.context_ids)) != len(self.context_ids):
            raise ValueError("circuit response context IDs must be nonempty and unique")
        if len(self.context_ids) != values.size or not np.all(np.isfinite(values)):
            raise ValueError("circuit response values differ from their context axis")
        if labels_required and self.labels is None:
            raise ValueError("calibrated circuit response requires labels")
        if (
            self.labels is not None
            and np.asarray(self.labels).reshape(-1).size != values.size
        ):
            raise ValueError("circuit response labels differ from their context axis")


@dataclass(frozen=True)
class MaskedPostactivationFeatureBank:
    """Host or memory-mapped states from the production masked executor."""

    context_ids: tuple[str, ...]
    module_axis: tuple[str, ...]
    module_paths: Mapping[str, str]
    module_states: Mapping[str, np.ndarray]
    execution_backend_id: str
    execution_profile: Mapping[str, Any]
    storage_kind: str

    def validate(self) -> None:
        if (
            not self.context_ids
            or len(set(self.context_ids)) != len(self.context_ids)
            or not self.module_axis
            or len(set(self.module_axis)) != len(self.module_axis)
            or set(self.module_paths) != set(self.module_axis)
            or set(self.module_states) != set(self.module_axis)
            or self.storage_kind not in {"host_arrays", "numpy_open_memmap"}
            or not self.execution_backend_id
        ):
            raise ValueError("masked postactivation feature bank is invalid")
        for module in self.module_axis:
            values = np.asarray(self.module_states[module])
            if (
                values.ndim != 2
                or values.shape[0] != len(self.context_ids)
                or not np.issubdtype(values.dtype, np.number)
                or not np.isfinite(values).all()
            ):
                raise ValueError("masked postactivation state is invalid")


class ReadoutResponse:
    """One ordered, uncollapsed model-readout response matrix.

    ``device_values`` optionally carries the same matrix as a device-resident
    float32 tensor so hot-loop consumers skip the host->device re-upload and
    float64->float32 cast per fit. It is runtime carriage, not scientific
    identity: excluded from durable caching; cache-replayed responses carry
    host values only.

    ``values`` may be constructed lazily (``values=None`` with a
    device tensor present) — the host float64 matrix then materializes ONCE
    on first access; consumers that only
    read ``device_values`` never pay the D2H transfer. ``validate`` checks
    the device tensor directly in that state (same shape/finiteness
    semantics) so validation never forces materialization.
    """

    __slots__ = (
        "context_ids",
        "labels",
        "device_labels",
        "device_values",
        "_values_host",
        "allow_duplicate_context_ids",
    )

    def __init__(
        self,
        context_ids: tuple[str, ...],
        values: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        device_labels: Any | None = None,
        device_values: Any | None = None,
        allow_duplicate_context_ids: bool = False,
    ) -> None:
        if values is None and device_values is None:
            raise ValueError("readout response requires values or device_values")
        if labels is None and device_labels is None:
            raise ValueError("readout response requires labels")
        self.context_ids = context_ids
        self.labels = labels
        self.device_labels = device_labels
        self.device_values = device_values
        self._values_host = values
        self.allow_duplicate_context_ids = bool(allow_duplicate_context_ids)

    @property
    def values(self) -> np.ndarray:
        if self._values_host is None:
            self._values_host = (
                self.device_values.detach()
                .to(device="cpu", dtype=torch.float64)
                .numpy()
            )
        return self._values_host

    def validate(self, *, check_finite: bool = True) -> None:
        labels = (
            np.asarray(self.labels).reshape(-1)
            if self.device_labels is None
            else self.device_labels.reshape(-1)
        )
        if self._values_host is None:
            rows, ndim = int(self.device_values.shape[0]), self.device_values.ndim
            finite = (
                bool(self.device_values.isfinite().all().item())
                if check_finite
                else True
            )
        else:
            values = np.asarray(self._values_host, dtype=np.float64)
            rows, ndim = values.shape[0] if values.ndim else 0, values.ndim
            finite = bool(np.isfinite(values).all()) if check_finite else True
        label_count = (
            int(labels.numel()) if self.device_labels is not None else int(labels.size)
        )
        if ndim != 2 or rows != label_count:
            raise ValueError("readout response must have shape [context, readout]")
        if rows != len(self.context_ids) or not finite:
            raise ValueError("readout response differs from its context axis")
        if not self.context_ids or (
            not self.allow_duplicate_context_ids
            and len(set(self.context_ids)) != len(self.context_ids)
        ):
            raise ValueError("readout response context IDs must be nonempty and unique")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64) - float(np.mean(left))
    y = np.asarray(right, dtype=np.float64) - float(np.mean(right))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denominator) if denominator > 1e-15 else 0.0


class TorchCircuitEvaluator:
    """Evaluate query-specific masks without mutating the source model."""

    @classmethod
    def from_bound_architecture(
        cls,
        bound_architecture: Any,
        *,
        model: Any,
        queries: Sequence[QuerySpec],
        batch_provider: CircuitBatchProvider,
        parity_receipt_hash: str,
        device: Any = "cpu",
        response_selector: ResponseSelector = default_response_selector,
        require_fixed_readout: bool = False,
        candidate_execution_backend: Any | None = None,
        runtime_execution_profile: RuntimeExecutionProfile | None = None,
        batch_norm_mode: str = "frozen_affine_v2",
    ) -> TorchCircuitEvaluator:
        bound_architecture.validate()
        fixed_readouts = tuple(
            binding
            for binding in bound_architecture.bindings
            if binding.circuit_role == "fixed_readout"
        )
        if len(fixed_readouts) > 1:
            raise ValueError(
                "circuit execution permits at most one declared fixed readout"
            )
        if require_fixed_readout and len(fixed_readouts) != 1:
            raise ValueError(
                "sampled-path execution requires one declared fixed readout"
            )
        return cls(
            model=model,
            queries=queries,
            route_blocks=bound_architecture.circuit_route_blocks,
            mask_bindings=tuple(
                TorchMaskBinding(
                    binding.layer_id,
                    binding.module_path,
                    binding.route_kind,
                )
                for binding in bound_architecture.circuit_bindings
            ),
            batch_provider=batch_provider,
            topology_hash=bound_architecture.topology_hash,
            route_schema_hash=bound_architecture.circuit_route_schema_hash,
            parity_receipt_hash=parity_receipt_hash,
            device=device,
            response_selector=response_selector,
            readout_module_path=(
                fixed_readouts[0].module_path if fixed_readouts else None
            ),
            candidate_execution_backend=candidate_execution_backend,
            runtime_execution_profile=runtime_execution_profile,
            batch_norm_mode=batch_norm_mode,
        )

    def __init__(
        self,
        *,
        model: Any,
        queries: Sequence[QuerySpec],
        route_blocks: Sequence[RouteBlockSpec],
        mask_bindings: Sequence[TorchMaskBinding],
        batch_provider: CircuitBatchProvider,
        topology_hash: str,
        route_schema_hash: str,
        parity_receipt_hash: str,
        device: Any = "cpu",
        response_selector: ResponseSelector = default_response_selector,
        readout_module_path: str | None = None,
        candidate_execution_backend: Any | None = None,
        runtime_execution_profile: RuntimeExecutionProfile | None = None,
        batch_norm_mode: str = "frozen_affine_v2",
    ) -> None:
        _require_torch()
        configure_deterministic_backend()
        if not isinstance(model, nn.Module):
            raise ValueError("native torch circuit evaluator requires an nn.Module")
        from mechanistic_systems.backends.deterministic_pooling import install_deterministic_adaptive_avg_pool2d
        install_deterministic_adaptive_avg_pool2d(model)
        self.source_model = model
        self.queries = tuple(queries)
        self.query_ids = tuple(query.query_id for query in self.queries)
        if not self.queries or len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("circuit evaluator query IDs must be nonempty and unique")
        self._query_by_id = {query.query_id: query for query in self.queries}
        self.route_blocks = tuple(route_blocks)
        self.mask_bindings = tuple(mask_bindings)
        if len(self.route_blocks) != len(self.mask_bindings):
            raise ValueError("route blocks and mask bindings must have identical axes")
        self.batch_provider = batch_provider
        self.topology_hash = str(topology_hash)
        self.route_schema_hash = str(route_schema_hash)
        self.parity_receipt_hash = str(parity_receipt_hash)
        self.device = torch.device(device)
        from mechanistic_systems.backends.cuda_graphs import initialize_cuda_graph_state
        initialize_cuda_graph_state(self.device)
        self.response_selector = response_selector
        self.readout_module_path = (
            None if readout_module_path is None else str(readout_module_path)
        )
        self.runtime_execution_profile = RuntimeExecutionProfile.resolve(
            runtime_execution_profile
        )
        self.candidate_execution_backend = (
            candidate_execution_backend
            or TorchCandidateExecutionBackend(
                runtime_execution_profile=self.runtime_execution_profile,
            )
        )
        self.readout_bias = self._resolve_readout_bias()
        self.evaluator_id = "torch-explicit-gate-graph-evaluator-v2"
        self._baseline_cache: dict[
            tuple[str, str], tuple[tuple[str, ...], np.ndarray, np.ndarray | None]
        ] = {}
        self._fully_pruned_cache: dict[
            tuple[str, str], tuple[tuple[str, ...], np.ndarray, np.ndarray | None]
        ] = {}
        self._validate_bindings()
        compiler = TorchGateGraphCompiler(batch_norm_mode=batch_norm_mode)
        if batch_norm_mode != "frozen_affine_v2":
            self.evaluator_id = "torch-explicit-gate-graph-evaluator-v3-native-lane-batchnorm"
        self._gate_graph = compiler.compile_bindings(
            self.source_model,
            self.mask_bindings,
            architecture_id=self.route_blocks[0].architecture_id,
            topology_hash=self.topology_hash,
            route_schema_hash=self.route_schema_hash,
            device=self.device,
        )
        self.model = self._gate_graph.model
        # NHWC uses fixed deterministic CUDA kernels. The forward tier controls
        # precision only: tier >=1 opts into TF32, tier 2 into bf16.
        self.forward_tier = int(self.runtime_execution_profile.forward_tier)
        self._forward_tier_applied = False

    def release_item_resources(self) -> dict[str, int]:
        """Drop completed-item storage while retaining the compiled model.

        The shared worker rebinds its batch provider before the next item.
        Retire that item's source binding with its cached responses. Model
        state, structural runners and prewarmed walk graph addresses remain
        reusable.
        """

        released = {
            "baseline_entries": len(self._baseline_cache),
            "fully_pruned_entries": len(self._fully_pruned_cache),
            **self.candidate_execution_backend.release_item_resources(),
        }
        self._baseline_cache.clear()
        self._fully_pruned_cache.clear()
        self.batch_provider = None
        return released

    def _execution_inputs(self, inputs: Any) -> Any:
        values = inputs.to(self.device, non_blocking=True)
        binding = getattr(self, "_uint8_folded_stem_binding", None)
        raw = values.dtype == torch.uint8 and binding is not None and binding["engaged"]
        if values.dtype == torch.uint8 and not raw:
            # Raw bank codes are decoded only by the bound folded stem; a
            # float cast would feed undecoded codes to the ordinary conv.
            raise ValueError(
                "raw uint8 images reached the evaluator without a bound folded "
                "conv1 stem; deliver normalized float images or bind the stem"
            )
        if not values.is_floating_point() and not raw:
            values = values.to(dtype=torch.float32)
        return values

    @property
    def claim_execution_certified(self) -> bool:
        """Opaque parity references carry provenance, not execution authority."""

        return False

    def bind_uint8_folded_stem(self, data_planes: Sequence[Any]) -> dict[str, Any]:
        """Bind raw bank delivery and the existing gated first conv together.

        Normalized float callers still use the ordinary conv. Raw uint8
        callers use the folded weights of their own mask, without changing
        the source network or the route-buffer names used by stateless lanes.
        This transaction must precede panel serving and prepared forwards.
        """
        from mechanistic_systems.backends.torch_gate_graph import GatedConv2d
        planes = tuple(data_planes)
        enabled = bool(planes) and all(p._uint8_folded_stem_env for p in planes)
        receipt = {"enabled": enabled, "engaged": False, "compile_mode": "gated_conv1_fold",
                   "acceptance": "same_estimand_algorithm_numerical_equivalence",
                   "consumer_count": len(planes)}
        stem = None
        added_buffers = False
        entered = []
        try:
            if not enabled:
                receipt["reason"] = "disabled"
            elif not all(p.optimized_bank for p in planes):
                receipt["reason"] = "consumer_requires_normalized_fallback"
            else:
                graph = getattr(self.model, "graph", None)
                placeholders = [] if graph is None else [n for n in graph.nodes if n.op == "placeholder"]
                if len(placeholders) != 1 or len(placeholders[0].users) != 1:
                    raise ValueError("folded stem requires one image input and one first consumer")
                node = next(iter(placeholders[0].users))
                if node.op != "call_module" or node.args != (placeholders[0],) or node.kwargs:
                    raise ValueError("folded stem requires conv1 directly on the image input")
                stem = self.model.get_submodule(str(node.target))
                if (not isinstance(stem, GatedConv2d) or stem.module.groups != 1
                        or stem.module.padding_mode != "zeros" or isinstance(stem.module.padding, str)
                        or stem.module.weight.dtype != torch.float32):
                    raise ValueError("folded stem requires a float32 gated conv with zero padding and groups=1")
                scale, bias = planes[0]._decode_scale, planes[0]._decode_bias
                if not all(torch.equal(scale, p._decode_scale) and torch.equal(bias, p._decode_bias) for p in planes):
                    raise ValueError("folded stem consumers have different decode constants")
                from mechanistic_systems.measurement.native_panel_data_plane import compute_uint8_folded_stem
                compute_uint8_folded_stem(stem.module.weight, stem.module.bias,
                                         decode_scale=scale, decode_bias=bias)
                if any(p.receipts for p in planes):
                    raise ValueError("folded stem must bind before serving")
                if hasattr(stem, "uint8_decode_scale"):
                    if not (torch.equal(stem.uint8_decode_scale, scale.to(stem.module.weight.device))
                            and torch.equal(stem.uint8_decode_bias, bias.to(stem.module.weight.device))):
                        raise ValueError("folded stem is already bound to other decode constants")
                else:
                    stem.register_buffer("uint8_decode_scale", scale.to(stem.module.weight.device).clone(), persistent=False)
                    stem.register_buffer("uint8_decode_bias", bias.to(stem.module.weight.device).clone(), persistent=False)
                    added_buffers = True
                for plane in planes:
                    if not plane.enter_raw_panel_delivery():
                        raise ValueError("consumer refused raw bank delivery")
                    entered.append(plane)
                receipt.update(engaged=True, reason="bound", module_path=str(node.target),
                               raw_input_dtype="uint8", scoring_input_space="normalized",
                               source_parameters_changed=False)
        except (ValueError, RuntimeError) as error:
            for plane in entered:
                plane.exit_raw_panel_delivery()
            if added_buffers:
                del stem.uint8_decode_scale
                del stem.uint8_decode_bias
            receipt["reason"] = str(error)
        for plane in planes:
            record = getattr(plane, "record_uint8_folded_stem_binding", None)
            if callable(record):
                record(receipt)
        self._uint8_folded_stem_binding = receipt
        return receipt

    @property
    def execution_certification_status(self) -> str:
        return "unverified_opaque_parity_reference"

    def require_certified_execution(self) -> None:
        """Refuse promotion until a scoped, source-verified receipt is bound."""

        raise ValueError(
            "masked execution is unverified: an opaque parity reference does not "
            "certify compiler, model, backend, or numerical scope; a verified "
            "execution receipt interface is not yet available"
        )

    def _resolve_readout_bias(self) -> Any | None:
        """Freeze the fixed readout bias used by historical C2 centering."""

        if self.readout_module_path is None:
            return None
        module = self._module(self.readout_module_path)
        if not isinstance(module, nn.Linear):
            raise ValueError("declared fixed readout must resolve to nn.Linear")
        if module.bias is None:
            return torch.zeros(int(module.out_features), dtype=torch.float32)
        return module.bias.detach().to(device="cpu", dtype=torch.float32).clone()

    @property
    def gate_graph(self) -> Any:
        """The explicit compiled graph shared by pruning and execution."""

        return self._gate_graph

    def candidate_execution_profile(self) -> Mapping[str, Any]:
        """Return runtime telemetry from the active stateless mask backend."""

        profile = getattr(self.candidate_execution_backend, "runtime_profile", None)
        if not callable(profile):
            return {
                "backend_id": str(
                    getattr(
                        self.candidate_execution_backend,
                        "backend_id",
                        type(self.candidate_execution_backend).__name__,
                    )
                ),
                "scientific_identity": "excluded",
                "deterministic_backend": deterministic_backend_record(),
            }
        return {
            **profile().to_runtime_dict(),
            "deterministic_backend": deterministic_backend_record(),
        }

    def _module(self, path: str) -> Any:
        try:
            return self.source_model.get_submodule(path)
        except (AttributeError, KeyError) as error:
            raise ValueError(f"unresolved circuit mask module path {path!r}") from error

    def _validate_bindings(self) -> None:
        expected_offset = 0
        for block, binding in zip(self.route_blocks, self.mask_bindings, strict=True):
            block.validate()
            binding.validate()
            if (
                block.layer_id != binding.layer_id
                or block.route_kind != binding.route_kind
            ):
                raise ValueError(
                    "circuit route block and mask binding semantics differ"
                )
            if (
                block.topology_hash != self.topology_hash
                or block.offset != expected_offset
            ):
                raise ValueError("circuit route block topology or ordering differs")
            module = self._module(binding.module_path)
            if binding.route_kind == "linear_weight":
                if (
                    not isinstance(module, nn.Linear)
                    or tuple(module.weight.shape) != block.shape
                ):
                    raise ValueError(
                        "linear mask binding shape differs from its route block"
                    )
            elif binding.route_kind == "conv_channel_pair":
                if not isinstance(module, nn.Conv2d):
                    raise ValueError("conv mask binding does not resolve to nn.Conv2d")
                expected = (
                    int(module.out_channels),
                    int(module.in_channels // module.groups),
                )
                if expected != block.shape:
                    raise ValueError(
                        "conv mask binding shape differs from its route block"
                    )
            elif binding.route_kind == "conv_kernel_weight":
                if not isinstance(module, nn.Conv2d):
                    raise ValueError(
                        "conv kernel-tap mask binding does not resolve to nn.Conv2d"
                    )
                if tuple(module.weight.shape) != block.shape:
                    raise ValueError(
                        "conv kernel-tap mask binding shape differs from its route block"
                    )
            else:
                required = ("conv1", "bn1", "relu", "conv2", "bn2")
                if any(
                    not hasattr(module, name) for name in required
                ) or block.shape != (2,):
                    raise ValueError(
                        "residual mask binding differs from its two-route block"
                    )
            expected_offset = block.stop

    def _validate_mask_axis(self, masks_by_layer: Mapping[str, np.ndarray]) -> None:
        """Require exact route-block coverage with an informative error.

        The previous validation materialized ``module.weight * mask`` for
        every route block and discarded the products — a full masked-weight
        build per compiled mask paid purely for this check.
        """

        expected = {block.layer_id for block in self.route_blocks}
        if set(masks_by_layer) != expected:
            missing = sorted(expected - set(masks_by_layer))
            extra = sorted(set(masks_by_layer) - expected)
            raise ValueError(
                "circuit mask does not cover the exact route-block axis: "
                f"missing={missing!r} extra={extra!r}"
            )

    def compile_mask(
        self,
        masks_by_layer: Mapping[str, np.ndarray],
    ) -> CompiledTorchCircuitMask:
        """Compile one query mask once for repeated exact forwards."""

        self._validate_mask_axis(masks_by_layer)
        residual_gates: dict[str, Any] = {}
        gate_values: dict[str, Any] = {}
        float_gate_values: dict[str, Any] = {}
        gate_identities = {}
        for block, binding in zip(self.route_blocks, self.mask_bindings, strict=True):
            mask = np.asarray(masks_by_layer[block.layer_id], dtype=bool)
            if mask.shape != block.shape:
                raise ValueError(
                    f"circuit mask for {block.layer_id!r} has the wrong shape"
                )
            # CPU torch.as_tensor aliases NumPy storage. Compile an isolated
            # snapshot so a tensor consumer cannot mutate a committed state.
            mask = np.array(mask, dtype=bool, copy=True, order="C")
            # A persistent cache needs versioned gates even when its caller
            # is in inference_mode. This changes no tensor values.
            with torch.inference_mode(False):
                value = torch.as_tensor(
                    mask.reshape(2)
                    if binding.route_kind == "residual_stream_gate"
                    else mask,
                    device=self.device,
                )
                float_value = value.to(dtype=torch.float32)
            gate_values[block.layer_id] = value
            # Pre-cast once: gate buffers are floating; converting per forward
            # allocated a float copy of every gate on every call.
            float_gate_values[block.layer_id] = float_value
            gate_identities[block.layer_id] = HostGateIdentity(
                dense=bool(mask.all()),
                tensor_identity=tensor_mutation_identity(float_value),
                tensor=float_value,
            )
            if binding.route_kind == "residual_stream_gate":
                residual_gates[block.layer_id] = value
        return CompiledTorchCircuitMask(
            parameter_overrides={},
            residual_gates=residual_gates,
            gate_values=gate_values,
            float_gate_values=float_gate_values,
            gate_identities=gate_identities,
        )

    def compile_packed_mask(self, mask: Any) -> CompiledTorchCircuitMask:
        """Lower the canonical big-endian HardMask bits on the execution device."""

        count = sum(int(np.prod(block.shape)) for block in self.route_blocks)
        if int(mask.route_count) != count or len(mask.packed_bits) != (count + 7) // 8:
            raise ValueError("packed circuit mask does not match the route axis")
        packed = torch.as_tensor(
            np.frombuffer(mask.packed_bits, dtype=np.uint8).copy(), device=self.device
        )
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=self.device)
        values = ((packed[:, None] >> shifts[None, :]) & 1).reshape(-1)[:count].to(torch.bool)
        gate_values: dict[str, Any] = {}
        float_gate_values: dict[str, Any] = {}
        residual_gates: dict[str, Any] = {}
        gate_identities = {}
        host_bits = (
            np.unpackbits(np.frombuffer(mask.packed_bits, dtype=np.uint8), count=count)
        )
        offset = 0
        for block, binding in zip(self.route_blocks, self.mask_bindings, strict=True):
            size = int(np.prod(block.shape))
            value = values[offset : offset + size].reshape(block.shape)
            if binding.route_kind == "residual_stream_gate":
                value = value.reshape(2)
                residual_gates[block.layer_id] = value
            gate_values[block.layer_id] = value
            with torch.inference_mode(False):
                float_value = value.to(dtype=torch.float32)
            float_gate_values[block.layer_id] = float_value
            gate_identities[block.layer_id] = HostGateIdentity(
                dense=bool(host_bits[offset : offset + size].all()),
                tensor_identity=tensor_mutation_identity(float_value),
                tensor=float_value,
            )
            offset += size
        return CompiledTorchCircuitMask(
            parameter_overrides={}, residual_gates=residual_gates,
            gate_values=gate_values, float_gate_values=float_gate_values,
            gate_identities=gate_identities,
        )

    def compile_gate_values(
        self,
        gates_by_layer: Mapping[str, np.ndarray],
    ) -> CompiledTorchCircuitMask:
        """Compile finite route gates on the exact circuit coordinate axis.

        Circuit masks are the binary special case.  Explicit floating gates
        are needed for dose interventions, but remain constrained to [0, 1]
        so this boundary cannot silently become an unconstrained reweighting
        model.
        """

        expected = {block.layer_id for block in self.route_blocks}
        if set(gates_by_layer) != expected:
            raise ValueError("route gates do not cover the exact route-block axis")
        gate_values: dict[str, Any] = {}
        residual_gates: dict[str, Any] = {}
        gate_identities = {}
        for block, binding in zip(self.route_blocks, self.mask_bindings, strict=True):
            values = np.asarray(gates_by_layer[block.layer_id], dtype=np.float32)
            if values.shape != block.shape or not np.isfinite(values).all():
                raise ValueError(
                    f"route gate for {block.layer_id!r} has an invalid shape or value"
                )
            if np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError(
                    "canonical intervention route gates must lie in [0, 1]"
                )
            module = self._module(binding.module_path)
            reference = next(module.parameters(), None)
            dtype = reference.dtype if reference is not None else torch.float32
            with torch.inference_mode(False):
                tensor = torch.as_tensor(values, device=self.device, dtype=dtype)
                # Own the snapshot; NumPy writes do not update _version.
                tensor = tensor.clone()
            gate_values[block.layer_id] = tensor
            gate_identities[block.layer_id] = HostGateIdentity(
                dense=bool(np.all(values == 1)),
                tensor_identity=tensor_mutation_identity(tensor),
                tensor=tensor,
            )
            if binding.route_kind == "residual_stream_gate":
                residual_gates[block.layer_id] = tensor
        return CompiledTorchCircuitMask(
            parameter_overrides={},
            residual_gates=residual_gates,
            gate_values=gate_values,
            # Dose gates are compiled in parameter dtype already.
            float_gate_values=gate_values,
            gate_identities=gate_identities,
        )

    def _response_values(
        self,
        view: DataView,
        *,
        query_id: str,
        masks_by_layer: Mapping[str, np.ndarray] | None,
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray | None]:
        query = self._query_by_id.get(query_id)
        if query is None:
            raise KeyError(query_id)
        self.model.to(self.device)
        prior_training = bool(self.model.training)
        self.model.eval()
        ids: list[str] = []
        values: list[np.ndarray] = []
        label_values: list[np.ndarray] = []
        labels_available: bool | None = None
        try:
            with torch.inference_mode():
                for batch in self.batch_provider.iter_batches(view, query_id=query_id):
                    batch.validate()
                    batch.wait_until_ready(self.device)
                    labels = (
                        None
                        if batch.labels is None
                        else torch.as_tensor(batch.labels, device=self.device)
                    )
                    has_labels = labels is not None
                    if labels_available is None:
                        labels_available = has_labels
                    elif labels_available != has_labels:
                        raise ValueError(
                            "circuit evaluation batches inconsistently provide labels"
                        )
                    output = self.forward_with_mask(
                        batch.inputs,
                        masks_by_layer=masks_by_layer,
                    )
                    response = self.response_selector(output)
                    packet = evaluate_query(response, query, labels=labels)
                    ids.extend(batch.context_ids)
                    values.append(
                        packet.values.detach()
                        .to(device="cpu", dtype=torch.float32)
                        .numpy()
                    )
                    if labels is not None:
                        label_values.append(
                            labels.detach().to(device="cpu").numpy().reshape(-1)
                        )
                    batch.mark_consumed(self.device)
        finally:
            self.model.train(prior_training)
        context_ids = tuple(ids)
        if not context_ids or len(set(context_ids)) != len(context_ids):
            raise ValueError("circuit evaluation context IDs are empty or repeated")
        return (
            context_ids,
            np.concatenate(values).astype(np.float64, copy=False),
            np.concatenate(label_values) if label_values else None,
        )

    def response_values(
        self,
        view: DataView,
        *,
        query_id: str,
        masks_by_layer: Mapping[str, np.ndarray] | None,
    ) -> CircuitResponse:
        """Return exact ordered query logits for a full or masked execution."""

        context_ids, values, labels = self._response_values(
            view,
            query_id=query_id,
            masks_by_layer=masks_by_layer,
        )
        result = CircuitResponse(context_ids, values, labels)
        result.validate()
        return result

    def _backend_span(self, name: str):
        """Time an evaluator-interior span on the execution backend's ledger.

        Spans land in ``candidate_execution_backend.phase_time_s`` next to
        ``candidate.forward`` so receipts decompose the masked-forward phase
        (producer wait vs staging vs packing). Timing only — never semantics.
        """

        profiler = getattr(self.candidate_execution_backend, "profiler", None)
        if profiler is None:
            return nullcontext()
        return profiler.phase(name)

    def readout_response_from_batches(
        self,
        batches: Iterable[MeasurementInputBatch],
        *,
        compiled_mask: CompiledTorchCircuitMask | None,
        readout_projection: RuntimeReadoutProjectionSpec | None = None,
    ) -> ReadoutResponse:
        """Execute an exact mask once and retain the complete readout basis.

        Statistical sampled-path calibration is defined in the model's full
        readout space.  Query reduction therefore happens only after an
        off-fold calibration has been fitted.
        """

        # The gate graph is moved to the device once at compile time; walking
        # every parameter per readout call was pure per-call overhead.
        prior_training = bool(self.model.training)
        self.model.eval()
        self._apply_forward_tier()
        gates = {} if compiled_mask is None else compiled_mask.execution_gates()
        if gates is None:
            raise ValueError("compiled circuit mask lacks explicit gate values")
        prepared = self.candidate_execution_backend.prepare_lanes(
            self._gate_graph, (gates,), lane_keys=("readout",), trusted_gates=True,
        )
        ids: list[str] = []
        device_values: list[Any] = []
        device_labels: list[Any] = []
        batches_iter = iter(batches)
        _end = object()
        try:
            with torch.inference_mode():
                while True:
                    with self._backend_span("candidate.batch_wait"):
                        batch = next(batches_iter, _end)
                    if batch is _end:
                        break
                    batch.validate()
                    batch.wait_until_ready(self.device)
                    if batch.labels is None:
                        raise ValueError(
                            "sampled-path calibration requires class labels"
                        )
                    output = self.forward_with_prepared_lanes(
                        batch.inputs, compiled_graph=self._gate_graph,
                        prepared_lanes=prepared,
                    )[0]
                    if compiled_mask is not None:
                        compiled_mask.mark_read_complete()
                    readout = self.response_selector(output)
                    if readout_projection is not None:
                        readout = readout_projection.apply(readout)
                    if readout.ndim != 2 or readout.shape[0] != len(batch.context_ids):
                        raise ValueError(
                            "model readout must have shape [context, coordinate]"
                        )
                    ids.extend(batch.context_ids)
                    # Accumulate on device; a per-batch .to("cpu") forces a
                    # stream sync per batch, so the single device-to-host
                    # move happens once below.
                    device_values.append(readout.detach().to(dtype=torch.float32))
                    device_labels.append(
                        torch.as_tensor(batch.labels, device=readout.device)
                        .detach()
                        .reshape(-1)
                        .clone()
                    )
                    batch.mark_consumed(self.device)
        finally:
            self.model.train(prior_training)
        with self._backend_span("candidate.response_materialize"):
            device_matrix = torch.cat(device_values, dim=0)
            labels_device = torch.cat(device_labels, dim=0)
            device_walk = self.device.type == "cuda" and _walk_forward_active()
            result = ReadoutResponse(
                context_ids=tuple(ids),
                values=(
                    None
                    if (
                        self.runtime_execution_profile.lazy_readout
                        or device_walk
                        or (
                            readout_projection is not None
                            and not readout_projection.retain_host_values
                        )
                    )
                    else device_matrix.to(device="cpu", dtype=torch.float64).numpy()
                ),
                labels=None if device_walk else labels_device.to(device="cpu").numpy(),
                device_labels=labels_device if device_walk else None,
                device_values=device_matrix,
            )
            result.validate(check_finite=not device_walk)
        return result

    def readout_responses_from_batches(
        self,
        batches: Iterable[MeasurementInputBatch],
        *,
        compiled_masks: Sequence[CompiledTorchCircuitMask],
        lane_keys: Sequence[str],
        readout_projection: RuntimeReadoutProjectionSpec | None = None,
    ) -> tuple[ReadoutResponse, ...]:
        """Execute K exact masks over ONE pass of the panel stream.

        The CRN pair (and any future proposal set) shares every input batch:
        gate lanes are prepared once and each batch runs a single fused
        forward producing ``[K, batch, readout]``. K == 1 degenerates to the
        plain path. Retention semantics match
        :meth:`readout_response_from_batches` per lane.
        """

        if not compiled_masks or len(compiled_masks) != len(lane_keys):
            raise ValueError("fused execution requires aligned compiled masks")
        expected = {block.layer_id for block in self.route_blocks}
        lane_gates = []
        for compiled in compiled_masks:
            gates = compiled.execution_gates()
            if gates is None:
                raise ValueError("compiled circuit mask lacks explicit gate values")
            if set(gates) != expected:
                raise ValueError(
                    "compiled circuit mask differs from the route-block axis"
                )
            lane_gates.append(gates)
        self._apply_forward_tier()
        prepared = self.candidate_execution_backend.prepare_lanes(
            self._gate_graph,
            lane_gates,
            lane_keys=tuple(lane_keys),
            trusted_gates=True,
            lane_gate_identities=tuple(compiled.gate_identities for compiled in compiled_masks),
        )
        lane_count = len(lane_gates)
        prior_training = bool(self.model.training)
        self.model.eval()
        tiered = self.device.type == "cuda"
        autocast_enabled = self.forward_tier >= 2 and self.device.type == "cuda"
        ids: list[str] = []
        device_values: list[list[Any]] = [[] for _ in range(lane_count)]
        device_labels: list[Any] = []
        batches_iter = iter(batches)
        _end = object()
        try:
            with torch.inference_mode():
                while True:
                    with self._backend_span("candidate.batch_wait"):
                        batch = next(batches_iter, _end)
                    if batch is _end:
                        break
                    batch.validate()
                    batch.wait_until_ready(self.device)
                    if batch.labels is None:
                        raise ValueError(
                            "sampled-path calibration requires class labels"
                        )
                    with self._backend_span("candidate.batch_stage"):
                        values = self._execution_inputs(batch.inputs)
                        if tiered and values.dim() == 4 and values.dtype != torch.uint8:
                            values = values.to(memory_format=torch.channels_last)
                    if autocast_enabled:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            output = self.candidate_execution_backend.evaluate_prepared(
                                self._gate_graph,
                                prepared,
                                values,
                                context_ids=batch.context_ids,
                                input_identity_hash=batch.input_identity_hash,
                            )
                        if torch.is_tensor(output) and output.dtype != torch.float32:
                            output = output.to(dtype=torch.float32)
                    else:
                        output = self.candidate_execution_backend.evaluate_prepared(
                            self._gate_graph,
                            prepared,
                            values,
                            context_ids=batch.context_ids,
                            input_identity_hash=batch.input_identity_hash,
                        )
                    ids.extend(batch.context_ids)
                    for lane in range(lane_count):
                        readout = self.response_selector(output[lane])
                        if readout_projection is not None:
                            readout = readout_projection.apply(readout)
                        if readout.ndim != 2 or readout.shape[0] != len(
                            batch.context_ids
                        ):
                            raise ValueError(
                                "model readout must have shape [context, coordinate]"
                            )
                        device_values[lane].append(
                            readout.detach().to(dtype=torch.float32)
                        )
                    device_labels.append(
                        torch.as_tensor(batch.labels, device=self.device)
                        .detach()
                        .reshape(-1)
                        # NativePanelDataPlane labels live in a reusable
                        # device workspace.  The forward may release that
                        # workspace immediately after this batch, but the
                        # response materializer retains labels until the full
                        # panel completes.  Own one compact per-batch copy so
                        # a later staged chunk cannot silently rewrite earlier
                        # response labels.
                        .clone()
                    )
                    batch.mark_consumed(self.device)
        finally:
            self.model.train(prior_training)
        with self._backend_span("candidate.response_materialize"):
            for compiled in compiled_masks:
                compiled.mark_read_complete()
            labels_device = torch.cat(device_labels, dim=0)
            responses = []
            for lane in range(lane_count):
                device_matrix = torch.cat(device_values[lane], dim=0)
                result = ReadoutResponse(
                    context_ids=tuple(ids),
                    values=(
                        None
                        if (
                            self.runtime_execution_profile.lazy_readout
                            or (self.device.type == "cuda" and _walk_forward_active())
                            or (
                                readout_projection is not None
                                and not readout_projection.retain_host_values
                            )
                        )
                        else device_matrix.to(device="cpu", dtype=torch.float64).numpy()
                    ),
                    labels=None,
                    device_labels=labels_device,
                    device_values=device_matrix,
                )
                # Native calibrated-BCE consumers retain both logits and labels
                # on device.  Finiteness is checked once at the completed
                # statistical look rather than forcing a synchronization for
                # every physical batch response.
                result.validate(check_finite=False)
                responses.append(result)
        return tuple(responses)

    def calibrated_bce_fit(
        self,
        view: DataView,
        *,
        query_id: str,
        masks_by_layer: Mapping[str, np.ndarray] | None,
    ) -> tuple[CircuitResponse, AffineCalibrationFit]:
        """Fit the historical per-candidate class-balanced affine readout."""

        response = self.response_values(
            view,
            query_id=query_id,
            masks_by_layer=masks_by_layer,
        )
        response.validate(labels_required=True)
        query = self._query_by_id[query_id]
        target_class_id = query.evaluator.get(
            "class_id",
            query.evaluator.get("unit", query.evaluator.get("coordinate")),
        )
        if target_class_id is None:
            raise ValueError("affine BCE calibration requires a class-query coordinate")
        assert response.labels is not None
        fit = fit_balanced_affine_calibration(
            response.values,
            response.labels,
            target_class_id=int(target_class_id),
        )
        return response, fit

    def calibrated_bce_loss(
        self,
        view: DataView,
        *,
        query_id: str,
        masks_by_layer: Mapping[str, np.ndarray] | None,
        fit: AffineCalibrationFit | None = None,
    ) -> tuple[CircuitResponse, AffineCalibrationFit, float]:
        """Return a fitted or fixed-affine class-balanced BCE loss."""

        if fit is None:
            response, resolved_fit = self.calibrated_bce_fit(
                view,
                query_id=query_id,
                masks_by_layer=masks_by_layer,
            )
        else:
            response = self.response_values(
                view,
                query_id=query_id,
                masks_by_layer=masks_by_layer,
            )
            response.validate(labels_required=True)
            resolved_fit = fit
        query = self._query_by_id[query_id]
        target_class_id = query.evaluator.get(
            "class_id",
            query.evaluator.get("unit", query.evaluator.get("coordinate")),
        )
        if target_class_id is None or response.labels is None:
            raise ValueError("affine BCE evaluation requires class-query labels")
        loss = balanced_binary_bce(
            resolved_fit.calibrated_logits(response.values),
            response.labels,
            target_class_id=int(target_class_id),
        )
        return response, resolved_fit, float(loss)

    def forward_with_mask(
        self,
        inputs: Any,
        *,
        masks_by_layer: Mapping[str, np.ndarray] | None,
    ) -> Any:
        """Run one exact full or masked forward without changing model parameters.

        This is the shared executable boundary for circuit identification and
        mechanism-response materialization. Residual streams are recombined
        before the block nonlinearity, while parameter masks are applied by
        ``functional_call`` so the registered source model remains immutable.
        """

        values = self._execution_inputs(inputs)
        compiled = None if masks_by_layer is None else self.compile_mask(masks_by_layer)
        return self.forward_with_compiled_mask(values, compiled_mask=compiled)

    def _apply_forward_tier(self) -> None:
        if self._forward_tier_applied:
            return
        self._forward_tier_applied = True
        configure_deterministic_backend()
        if self.device.type != "cuda":
            return
        self.model.to(memory_format=torch.channels_last)
        if self.forward_tier >= 1:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def forward_with_compiled_mask(
        self,
        inputs: Any,
        *,
        compiled_mask: CompiledTorchCircuitMask | None,
    ) -> Any:
        """Run one forward with an already device-resident circuit mask."""

        self._apply_forward_tier()
        values = self._execution_inputs(inputs)
        tiered = self.device.type == "cuda"
        if tiered and values.dim() == 4 and values.dtype != torch.uint8:
            values = values.to(memory_format=torch.channels_last)
        gate_values: Mapping[str, Any] = {}
        trusted = False
        if compiled_mask is not None:
            if compiled_mask.gate_values is None:
                raise ValueError("compiled circuit mask lacks explicit gate values")
            # Prefer the pre-cast float gates; compiled masks were shape-
            # checked at compile time and boolean/[0,1] gates are finite by
            # construction, so per-call gate validation (with its per-gate
            # device sync) is skipped.
            gate_values = compiled_mask.execution_gates()
            trusted = True
            expected = {block.layer_id for block in self.route_blocks}
            if set(gate_values) != expected:
                raise ValueError(
                    "compiled circuit mask differs from the route-block axis"
                )
        if self.forward_tier >= 2 and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.candidate_execution_backend.evaluate(
                    self._gate_graph,
                    values,
                    gate_values,
                    trusted_gates=trusted,
                )
            if torch.is_tensor(output) and output.dtype != torch.float32:
                output = output.to(dtype=torch.float32)
            if compiled_mask is not None:
                compiled_mask.mark_read_complete()
            return output
        output = self.candidate_execution_backend.evaluate(
            self._gate_graph,
            values,
            gate_values,
            trusted_gates=trusted,
        )
        if compiled_mask is not None:
            compiled_mask.mark_read_complete()
        return output

    def forward_with_prepared_lanes(
        self,
        inputs: Any,
        *,
        compiled_graph: Any,
        prepared_lanes: Any,
        context_ids: Sequence[str] | None = None,
        input_identity_hash: str | None = None,
    ) -> Any:
        """Execute an already prepared exact mask lane-set on one input batch.

        This is the shared pruning/runtime multi-mask backend exposed to
        response materializers.  It retains the same forward-tier semantics
        as :meth:`forward_with_compiled_mask`; only the runtime scheduling is
        different.  The returned tree carries a leading lane axis.
        """

        self._apply_forward_tier()
        values = self._execution_inputs(inputs)
        if self.device.type == "cuda" and values.dim() == 4 and values.dtype != torch.uint8:
            values = values.to(memory_format=torch.channels_last)

        def execute() -> Any:
            return self.candidate_execution_backend.evaluate_prepared(
                compiled_graph,
                prepared_lanes,
                values,
                context_ids=context_ids,
                input_identity_hash=input_identity_hash,
            )

        if self.forward_tier >= 2 and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return execute()
        return execute()

    def prewarm_walk_graphs(self, inputs: Any, *, include_single: bool = True,
                            include_pairs: bool = True) -> Mapping[str, Any]:
        """Capture every layer's walk forward once before serving decisions.

        Synthetic gates choose a graph structure only. They supply no rows
        to calibration or adjudication and change no committed mask. A
        declared differing layer chooses each fork through host metadata,
        so preparing the seven AlexNet pairs performs no device comparison.
        """
        backend = self.candidate_execution_backend
        before = backend.walk_graph_captures
        if self.device.type != "cuda":
            return {"status": "cpu", "captures": 0}
        rows = int(inputs.shape[0])
        if all(not enabled or (self.walk_graph_batch_capacity(lanes) or 0) >= rows
               for enabled, lanes in ((include_single, 1), (include_pairs, 2))):
            return {"status": "reused", "rows": rows, "captures": 0,
                    "total_captures": before,
                    "layers": tuple(self._gate_graph.gates_by_layer) if include_pairs else (),
                    "single": bool(include_single), "pairs": bool(include_pairs)}
        self.model.eval()
        self._apply_forward_tier()
        values = self._execution_inputs(inputs)
        if values.dim() == 4 and values.dtype != torch.uint8:
            values = values.to(memory_format=torch.channels_last)
        dense = {layer: torch.ones_like(module.route_gate)
                 for layer, module in self._gate_graph.gates_by_layer.items()}
        prepared = []
        if include_single:
            prepared.append(backend.prepare_lanes(
                self._gate_graph, (dense,), lane_keys=("worker:dense",), trusted_gates=True,
                differing_layers=(),
            ))
        if include_pairs:
            for layer, gate in dense.items():
                candidate = dict(dense)
                candidate[layer] = gate.clone()
                candidate[layer].reshape(-1)[0] = 0
                prepared.append(backend.prepare_lanes(
                    self._gate_graph, (dense, candidate),
                    lane_keys=("worker:parent", f"worker:{layer}"),
                    trusted_gates=True, differing_layers=(layer,),
                ))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=self.forward_tier >= 2):
            backend.prewarm_walk_graphs(self._gate_graph, values, prepared)
        return {"status": "prepared", "rows": int(values.shape[0]),
                "layers": tuple(dense) if include_pairs else (),
                "captures": backend.walk_graph_captures - before,
                "total_captures": backend.walk_graph_captures,
                "single": bool(include_single), "pairs": bool(include_pairs)}

    def walk_graph_batch_capacity(self, lane_count: int) -> int | None:
        """Prepared physical capacity; longer statistical looks are chunked."""
        entries = self.candidate_execution_backend._walk_graph_cache
        capacities = [capacity for (key, capacity) in entries
                      if key[0] == id(self._gate_graph.model) and key[2] == int(lane_count)]
        return max(capacities) if capacities else None

    def masked_postactivation_feature_bank_from_batches(
        self,
        batches: Iterable[MeasurementInputBatch],
        *,
        compiled_mask: CompiledTorchCircuitMask,
        capture_module_paths: Mapping[str, str],
        sample_rows: int,
        bank_directory: str | Path | None = None,
    ) -> MaskedPostactivationFeatureBank:
        """Capture registered postactivations through the pruning executor.

        This observation-only route invokes :meth:`forward_with_compiled_mask`
        for every physical batch, so masking, forward tiers, and candidate-
        backend behavior are exactly the production pruning behavior.  Hooks
        only observe caller-declared postactivation module boundaries.  Each
        batch is moved to host immediately; a full activation hierarchy is
        never resident on the GPU.  Supplying ``bank_directory`` additionally
        bounds host RAM by writing each layer into one ``.npy`` memory map.
        """

        modules = tuple(str(value) for value in capture_module_paths)
        paths = {str(key): str(value) for key, value in capture_module_paths.items()}
        if (
            sample_rows < 1
            or not modules
            or len(set(modules)) != len(modules)
            or len(set(paths.values())) != len(paths)
            or any(not module or not paths[module] for module in modules)
        ):
            raise ValueError("postactivation capture axis is invalid")
        if self.readout_module_path is not None and self.readout_module_path in set(
            paths.values()
        ):
            raise ValueError("the fixed readout is not an admissible hierarchy predictor")
        gate_values = compiled_mask.execution_gates()
        if gate_values is None or set(gate_values) != set(
            self._gate_graph.gates_by_layer
        ):
            raise ValueError("postactivation capture mask differs from the gate graph")
        capture_modules: dict[str, Any] = {}
        for module, module_path in paths.items():
            try:
                capture_modules[module] = self.model.get_submodule(module_path)
            except (AttributeError, KeyError) as error:
                raise ValueError(
                    f"unresolved postactivation module path {module_path!r}"
                ) from error

        output_root = None if bank_directory is None else Path(bank_directory).resolve()
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
        active: dict[str, Any] = {}
        handles: list[Any] = []
        host_chunks: dict[str, list[np.ndarray]] = {module: [] for module in modules}
        writers: dict[str, Any] = {}
        bank_paths: dict[str, Path] = {}
        context_ids: list[str] = []
        cursor = 0

        def capture_hook(module_name: str):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                if not torch.is_tensor(output):
                    raise RuntimeError("registered postactivation boundary did not emit a tensor")
                if module_name in active:
                    raise RuntimeError("registered postactivation module executed more than once")
                # Own the boundary value before a downstream in-place operation
                # can mutate the tensor observed by the hook.
                active[module_name] = output.detach().clone()

            return hook

        for module in modules:
            handles.append(
                capture_modules[module].register_forward_hook(capture_hook(module))
            )
        prior_training = bool(self.model.training)
        self.model.eval()
        try:
            with torch.inference_mode():
                for batch in batches:
                    batch.validate()
                    batch.wait_until_ready(self.device)
                    active.clear()
                    try:
                        self.forward_with_compiled_mask(
                            batch.inputs, compiled_mask=compiled_mask
                        )
                        if set(active) != set(modules):
                            raise RuntimeError(
                                "masked forward did not visit every registered postactivation"
                            )
                        batch_rows = len(batch.context_ids)
                        if cursor + batch_rows > sample_rows:
                            raise ValueError("postactivation batches exceed the declared sample axis")
                        if any(
                            not isinstance(value, str) or not value
                            for value in batch.context_ids
                        ):
                            raise ValueError("postactivation context IDs must be nonempty strings")
                        context_ids.extend(batch.context_ids)
                        for index, module in enumerate(modules):
                            tensor = active[module]
                            if int(tensor.shape[0]) != batch_rows:
                                raise RuntimeError("postactivation changed the physical batch axis")
                            matrix = (
                                tensor.reshape(batch_rows, -1)
                                .to(device="cpu", dtype=torch.float32)
                                .numpy()
                            )
                            if not np.isfinite(matrix).all():
                                raise FloatingPointError("postactivation capture became nonfinite")
                            if output_root is None:
                                host_chunks[module].append(np.ascontiguousarray(matrix))
                            else:
                                writer = writers.get(module)
                                if writer is None:
                                    path = output_root / f"{index:03d}_postactivation.npy"
                                    if path.exists():
                                        raise FileExistsError(
                                            f"refusing to overwrite postactivation bank {path}"
                                        )
                                    writer = np.lib.format.open_memmap(
                                        path,
                                        mode="w+",
                                        dtype=np.float32,
                                        shape=(sample_rows, matrix.shape[1]),
                                    )
                                    writers[module] = writer
                                    bank_paths[module] = path
                                if int(writer.shape[1]) != int(matrix.shape[1]):
                                    raise RuntimeError("postactivation feature width changed by batch")
                                writer[cursor : cursor + batch_rows] = matrix
                        cursor += batch_rows
                    finally:
                        batch.mark_consumed(self.device)
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(prior_training)
        if cursor != sample_rows or len(context_ids) != sample_rows:
            compiled_mask.mark_read_complete()
            raise ValueError("postactivation batches do not complete the declared sample axis")
        compiled_mask.mark_read_complete()
        if len(set(context_ids)) != len(context_ids):
            raise ValueError("postactivation context axis contains duplicates")
        if output_root is None:
            states = {
                module: np.concatenate(host_chunks[module], axis=0)
                for module in modules
            }
            storage_kind = "host_arrays"
        else:
            for writer in writers.values():
                writer.flush()
            writers.clear()
            states = {
                module: np.load(bank_paths[module], mmap_mode="r", allow_pickle=False)
                for module in modules
            }
            storage_kind = "numpy_open_memmap"
        result = MaskedPostactivationFeatureBank(
            context_ids=tuple(context_ids),
            module_axis=modules,
            module_paths=paths,
            module_states=states,
            execution_backend_id=str(
                getattr(
                    self.candidate_execution_backend,
                    "backend_id",
                    type(self.candidate_execution_backend).__name__,
                )
            ),
            execution_profile=self.candidate_execution_profile(),
            storage_kind=storage_kind,
        )
        result.validate()
        return result

    def residual_preactivations_from_batches(
        self,
        batches: Iterable[MeasurementInputBatch],
        *,
        compiled_mask: CompiledTorchCircuitMask,
        layer_ids: Iterable[str] | None = None,
    ) -> Mapping[str, np.ndarray]:
        """Capture residual terminal preactivations on the plain lane path.

        This observation-only path is deliberately separate from
        :meth:`readout_responses_from_batches`: capture mutates Python module
        attributes and therefore must never run inside the vmap-fused lane
        execution used by the decision statistic.

        ``layer_ids`` restricts which residual layers are copied to the host
        (the forward itself is unchanged); ``None`` copies them all.
        """

        residual_gates = {
            str(layer_id): module
            for layer_id, module in self._gate_graph.gates_by_layer.items()
            if isinstance(module, GatedResidualBlock)
        }
        if layer_ids is not None:
            wanted = {str(value) for value in layer_ids}
            residual_gates = {
                layer_id: module
                for layer_id, module in residual_gates.items()
                if layer_id in wanted
            }
        if not residual_gates:
            return {}
        gate_values = compiled_mask.execution_gates()
        if gate_values is None or set(gate_values) != set(
            self._gate_graph.gates_by_layer
        ):
            raise ValueError("observation mask differs from the gate-graph axis")
        captured: dict[str, list[Any]] = {layer_id: [] for layer_id in residual_gates}
        prior_training = bool(self.model.training)
        cuda_devices = (
            [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
            if self.device.type == "cuda" and torch.cuda.is_available()
            else []
        )
        self.model.eval()
        try:
            self._apply_forward_tier()
            for layer_id, module in self._gate_graph.gates_by_layer.items():
                module.set_gate(gate_values[str(layer_id)])
            for module in residual_gates.values():
                module.reset_capture()
            with (
                torch.random.fork_rng(devices=cuda_devices),
                torch.inference_mode(),
            ):
                for batch in batches:
                    batch.validate()
                    batch.wait_until_ready(self.device)
                    try:
                        values = self._execution_inputs(batch.inputs)
                        if (
                            self.forward_tier >= 1
                            and self.device.type == "cuda"
                            and values.dim() == 4
                        ):
                            values = values.to(memory_format=torch.channels_last)
                        if self.forward_tier >= 2 and self.device.type == "cuda":
                            with torch.autocast(
                                device_type="cuda",
                                dtype=torch.bfloat16,
                            ):
                                self._gate_graph.forward(values)
                        else:
                            self._gate_graph.forward(values)
                        for layer_id, module in residual_gates.items():
                            values = module.last_output_preactivation
                            if values is None:
                                continue
                            captured[layer_id].append(
                                values.detach()
                                .to(device="cpu", dtype=torch.float32)
                                .clone()
                            )
                    finally:
                        batch.mark_consumed(self.device)
        finally:
            for module in residual_gates.values():
                module.disarm_capture()
            self._gate_graph.reset()
            self.model.train(prior_training)
            compiled_mask.mark_read_complete()
        return {
            layer_id: torch.cat(values, dim=0).numpy()
            for layer_id, values in captured.items()
            if values
        }

    def stream_residual_preactivations(
        self,
        batches: Iterable[MeasurementInputBatch],
        *,
        compiled_masks: Sequence[CompiledTorchCircuitMask],
        consumer: Callable[[tuple[Mapping[str, Any], ...]], None],
        capture_reducers: Sequence[Callable[[Any], Any] | None] | None = None,
    ) -> int:
        """Device-resident residual captures, one batch at a time.

        The same plain-lane, observation-only forward as
        :meth:`residual_preactivations_from_batches`, but every batch runs
        under each of ``compiled_masks`` in turn and ``consumer`` receives the
        per-mask ``{layer_id: float32 device tensor}`` captures of that batch
        (never copied to the host).  Capture state is reset before every
        forward; a residual module that does not execute in a forward raises,
        so the caller falls back to the v5 host path.  Returns the number of
        batches consumed.

        ``capture_reducers[i]`` (optional) maps mask ``i``'s float32 capture
        of each layer right after that mask's forward -- for example the
        parent to its exact bool ``x > 0`` map -- so the full float32 capture
        is not held while the next mask's forward runs.  Each block's retained
        branch stack (``last_branches``, read by no consumer here) is released
        as soon as its pre-activation has been taken.  Memory only: the forward
        and the captured values are unchanged.
        """

        reducers = tuple(capture_reducers or ())
        if reducers and len(reducers) != len(compiled_masks):
            raise ValueError("capture_reducers must match compiled_masks")

        residual_gates = {
            str(layer_id): module
            for layer_id, module in self._gate_graph.gates_by_layer.items()
            if isinstance(module, GatedResidualBlock)
        }
        if not residual_gates:
            return 0
        gate_value_sets = []
        for compiled_mask in compiled_masks:
            gate_values = compiled_mask.execution_gates()
            if gate_values is None or set(gate_values) != set(
                self._gate_graph.gates_by_layer
            ):
                raise ValueError("observation mask differs from the gate-graph axis")
            gate_value_sets.append(gate_values)
        prior_training = bool(self.model.training)
        cuda_devices = (
            [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
            if self.device.type == "cuda" and torch.cuda.is_available()
            else []
        )
        consumed = 0
        self.model.eval()
        try:
            self._apply_forward_tier()
            for module in residual_gates.values():
                module.reset_capture()
            with (
                torch.random.fork_rng(devices=cuda_devices),
                torch.inference_mode(),
            ):
                for batch in batches:
                    batch.validate()
                    batch.wait_until_ready(self.device)
                    try:
                        values = self._execution_inputs(batch.inputs)
                        if (
                            self.forward_tier >= 1
                            and self.device.type == "cuda"
                            and values.dim() == 4
                        ):
                            values = values.to(memory_format=torch.channels_last)
                        captures: list[dict[str, Any]] = []
                        for mask_index, gate_values in enumerate(gate_value_sets):
                            reducer = reducers[mask_index] if reducers else None
                            for layer_id, module in self._gate_graph.gates_by_layer.items():
                                module.set_gate(gate_values[str(layer_id)])
                            for module in residual_gates.values():
                                module.last_output_preactivation = None
                            if self.forward_tier >= 2 and self.device.type == "cuda":
                                with torch.autocast(
                                    device_type="cuda",
                                    dtype=torch.bfloat16,
                                ):
                                    self._gate_graph.forward(values)
                            else:
                                self._gate_graph.forward(values)
                            capture: dict[str, Any] = {}
                            for layer_id, module in residual_gates.items():
                                preactivation = module.last_output_preactivation
                                if preactivation is None:
                                    raise RuntimeError(
                                        f"residual layer {layer_id!r} did not execute"
                                    )
                                value = preactivation.detach().to(dtype=torch.float32)
                                module.last_output_preactivation = None
                                module.last_branches = None
                                capture[layer_id] = (
                                    value if reducer is None else reducer(value)
                                )
                                del value
                            captures.append(capture)
                        consumer(tuple(captures))
                        consumed += 1
                        del captures
                    finally:
                        batch.mark_consumed(self.device)
        finally:
            for module in residual_gates.values():
                module.disarm_capture()
            self._gate_graph.reset()
            self.model.train(prior_training)
            for compiled in compiled_masks:
                compiled.mark_read_complete()
        return consumed

    def _baseline(
        self,
        view: DataView,
        query_id: str,
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray | None]:
        key = (view.view_hash, query_id)
        if key not in self._baseline_cache:
            self._baseline_cache[key] = self._response_values(
                view,
                query_id=query_id,
                masks_by_layer=None,
            )
        return self._baseline_cache[key]

    def _fully_pruned(
        self,
        view: DataView,
        query_id: str,
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray | None]:
        key = (view.view_hash, query_id)
        if key not in self._fully_pruned_cache:
            masks = {
                block.layer_id: np.zeros(block.shape, dtype=bool)
                for block in self.route_blocks
            }
            self._fully_pruned_cache[key] = self._response_values(
                view,
                query_id=query_id,
                masks_by_layer=masks,
            )
        return self._fully_pruned_cache[key]

    def evaluate(
        self,
        plan: CircuitPlan,
        *,
        query_id: str,
        masks_by_layer: Mapping[str, np.ndarray],
        split: str,
    ) -> CircuitEvaluation:
        if plan.route_universe_hash != self.route_schema_hash:
            raise ValueError(
                "CircuitPlan route universe differs from the torch evaluator"
            )
        view = (
            plan.optimization_view
            if split == "optimization"
            else plan.evaluation_view if split == "evaluation" else None
        )
        if view is None:
            raise ValueError(
                "circuit evaluation split must be optimization or evaluation"
            )
        baseline_ids, baseline, baseline_labels = self._baseline(view, query_id)
        masked_ids, masked, masked_labels = self._response_values(
            view,
            query_id=query_id,
            masks_by_layer=masks_by_layer,
        )
        if masked_ids != baseline_ids:
            raise ValueError("masked and full circuit evaluation context axes differ")
        if not (
            (baseline_labels is None and masked_labels is None)
            or (
                baseline_labels is not None
                and masked_labels is not None
                and np.array_equal(baseline_labels, masked_labels)
            )
        ):
            raise ValueError("masked and full circuit evaluation labels differ")
        residual = masked - baseline
        mse = float(np.mean(residual * residual))
        reference_energy = float(np.mean(baseline * baseline))
        normalized_mse = mse / max(reference_energy, 1e-12)
        relative_l2 = float(
            np.linalg.norm(residual) / max(np.linalg.norm(baseline), 1e-12)
        )
        correlation = _correlation(baseline, masked)
        metric = str(plan.response_distortion.get("metric", "normalized_mse"))
        balanced_damage = None
        if (
            metric == "balanced_bce_damage"
            or str(plan.response_distortion.get("preservation_metric", ""))
            == "one_minus_balanced_bce_damage"
        ):
            query = self._query_by_id[query_id]
            target_class_id = query.evaluator.get(
                "class_id",
                query.evaluator.get("unit", query.evaluator.get("coordinate")),
            )
            if target_class_id is None or baseline_labels is None:
                raise ValueError(
                    "balanced BCE target preservation requires class-query coordinates and labels"
                )
            null_ids, fully_pruned, null_labels = self._fully_pruned(view, query_id)
            if (
                null_ids != baseline_ids
                or null_labels is None
                or not np.array_equal(baseline_labels, null_labels)
            ):
                raise ValueError("fully pruned and full circuit evaluation axes differ")
            balanced_damage = balanced_bce_damage(
                baseline,
                masked,
                fully_pruned,
                baseline_labels,
                target_class_id=int(target_class_id),
            )
        if metric == "mse":
            distortion = mse
        elif metric == "normalized_mse":
            distortion = normalized_mse
        elif metric == "relative_l2":
            distortion = relative_l2
        elif metric == "one_minus_correlation":
            distortion = max(0.0, 1.0 - correlation)
        elif metric == "balanced_bce_damage":
            assert balanced_damage is not None
            distortion = balanced_damage.response_distortion
        else:
            raise ValueError(
                f"unsupported circuit response distortion metric {metric!r}"
            )
        preservation_metric = str(
            plan.response_distortion.get(
                "preservation_metric", "inverse_normalized_mse"
            )
        )
        if preservation_metric == "inverse_normalized_mse":
            preservation = 1.0 / (1.0 + normalized_mse)
        elif preservation_metric == "correlation":
            preservation = correlation
        elif preservation_metric == "one_minus_balanced_bce_damage":
            assert balanced_damage is not None
            preservation = balanced_damage.target_preservation
        else:
            raise ValueError(
                f"unsupported target preservation metric {preservation_metric!r}"
            )
        metrics = {
            "mse": mse,
            "normalized_mse": normalized_mse,
            "relative_l2": relative_l2,
            "correlation": correlation,
            "reference_energy": reference_energy,
        }
        if balanced_damage is not None:
            metrics.update(balanced_damage.to_metrics())
        result = CircuitEvaluation(
            response_distortion=float(distortion),
            target_preservation=float(preservation),
            context_count=len(baseline_ids),
            metrics=metrics,
        )
        result.validate()
        return result


__all__ = [
    "CircuitBatchProvider",
    "CircuitResponse",
    "CompiledTorchCircuitMask",
    "MaskedPostactivationFeatureBank",
    "ReadoutResponse",
    "RuntimeReadoutProjectionSpec",
    "TorchCircuitEvaluator",
    "TorchMaskBinding",
]
