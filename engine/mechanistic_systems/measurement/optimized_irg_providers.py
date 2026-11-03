"""IRG response geometry backed by the canonical optimized scoring runtime.

This module is intentionally an adapter, not another attribution
implementation.  Query VJPs, route-gate semantics, route-family lowering, and
custom-kernel selection remain owned by ``NativeLayerBlockedMeasurementAdapter``.
IRG only selects a nested context prefix, installs the current hard mask on the
evaluator's already-compiled gate graph, and consumes the resulting whole-route
response rows.  The adapter must not compile a second masked model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
import time
from threading import RLock
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from mechanistic_systems.core.axis_identity import (
    canonical_unique_string_axis,
    compact_string_axis_layout,
    ordered_string_axis_hash,
)

try:  # Optional for import smoke; required by the native optimized provider.
    import torch
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]

from mechanistic_systems.contracts.interventions import HardMaskView
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.measurement import (
    MeasurementOutcome,
    MeasurementRequest,
)
from mechanistic_systems.contracts.measurement_costs import RealizedCost
from mechanistic_systems.contracts import MeasurementPlan, QuerySpec
from mechanistic_systems.contracts.trust_geometry import GlobalTrustGeometrySpec
from mechanistic_systems.measurement.production import (
    NativeLayerBlockedMeasurementAdapter,
)
from mechanistic_systems.measurement.device_route_moments import (
    DeviceFrequentDirectionsTrustAccumulator,
    DeviceTripartiteMomentAccumulator,
)
from mechanistic_systems.measurement.flip_margin_geometry import (
    DenseFlipMarginAccumulator,
)
from mechanistic_systems.measurement.streaming_response_provider import (
    LocalAttributionMomentTensor,
    ResponseJacobianBatch,
    StreamingResponseGeometryProvider,
    TripartitePullbackMomentTensor,
)
from mechanistic_systems.measurement.response_geometry import (
    ResponseMomentTensor,
)
from mechanistic_systems.measurement.pruning_loss import (
    CALIBRATED_OVR_BCE,
    PRUNING_GUIDING_LOSSES,
    census_loss_pullback_hash,
    census_loss_path_hash,
    dense_calibration_state_hash,
    frozen_affine_parameters,
    frozen_pruning_loss_multiplier,
    frozen_pruning_loss_path,
)
from mechanistic_systems.measurement.torch_adapter import (
    MeasurementInputBatch,
    TorchConservingActivationBinding,
    TorchLayerBinding,
    TorchModuleMeasurementAdapter,
)
from mechanistic_systems.measurement.local_regime import (
    LocalRegimeCaptureSpec,
    REGIME_STATISTIC_NAMES,
    SMOOTH_TRANSMISSION_STATISTIC_NAMES,
)
from mechanistic_systems.measurement.gate_geometry import (
    LocalRegimeRouteSummary,
    SmoothTransmissionRouteSummary,
)
from mechanistic_systems.measurement.in_process_response_summary import (
    InProcessResponseSummary,
    register_in_process_response_summary,
)
from mechanistic_systems.circuits.irg.modern_v1_conserving_route_score import (
    V1GateDisclosureRequest,
)


# ---------------------------------------------------------------------------
# Route-chunked streaming (runtime-only physical execution policy).
#
# The census-scale OOM is the per-packet materialization of
# ``contexts × route`` float64 tensors on the whole physical route axis
# (AlexNet: 32 × 54,776,000 × 8 B ≈ 14 GB per head, with several heads alive
# concurrently).  The chunk size below bounds the route extent of every
# transient ``contexts × route`` tensor this module creates.  Like
# ``CIRCUIT_SCORING_BATCH_ROWS``/``CIRCUIT_SAMPLED_PREFETCH_DEPTH`` in
# ``RuntimeExecutionProfile``, it is a runtime-only physical execution knob:
# it must never change delivered bytes, and it is excluded from scientific
# identity.  The default keeps every registered ResNet18 request on the
# original single-span code path (axis ≈ 1.39 M routes < default).
# ---------------------------------------------------------------------------

RESPONSE_ROUTE_CHUNK_ROUTES_ENVIRONMENT = "CIRCUIT_RESPONSE_ROUTE_CHUNK_ROUTES"
RESPONSE_ROUTE_CHUNK_ROUTES_DEFAULT = 2_097_152

_BCE_POSITIVE_REMOVAL_COORDINATE = "frozen_affine_bce_loss"
_RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE = "raw_target_logit"
_RAW_TARGET_LOGIT_UNIT_COTANGENT = "raw_target_logit_unit_cotangent"


def _exact_raw_target_logit_class_id(query: QuerySpec) -> int:
    """Return the exact class-logit coordinate or refuse aliases/coercions."""

    objective = str(
        query.evaluator.get("objective", query.evaluator.get("kind", ""))
    ).strip().lower()
    target = query.evaluator.get("class_id")
    if (
        query.query_kind != "class_logit"
        or objective != "class_logit"
        or isinstance(target, bool)
        or not isinstance(target, (int, np.integer))
        or int(target) < 0
    ):
        raise ValueError(
            "raw target-logit streaming requires an exact pre-softmax "
            "class_logit query with an integer target class"
        )
    return int(target)


def _raw_target_logit_positive_removal_stream_identity(
    *,
    query_id: str,
    target_class_id: int,
) -> str:
    """Identify the exact unit pullback of one raw target-logit query."""

    return content_hash(
        {
            "schema": "raw_target_logit_positive_removal_stream_v2",
            "query_id": str(query_id),
            "target_class_id": int(target_class_id),
            "response_coordinate": _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE,
            "cotangent_coordinate": _RAW_TARGET_LOGIT_UNIT_COTANGENT,
            "response_pullback": _RAW_TARGET_LOGIT_UNIT_COTANGENT,
            "affine_calibration_applied": False,
            "context_specific_loss_pullback_applied": False,
        },
        prefix="irg-raw-target-logit-positive-removal-",
    )


def _raw_target_logit_literal_jacobian_stream_identity(
    *,
    query_id: str,
    target_class_id: int,
) -> str:
    """Identify the literal current-chart relaxed-gate Jacobian stream."""

    return content_hash(
        {
            "schema": "raw_target_logit_literal_gate_jacobian_stream_v1",
            "query_id": str(query_id),
            "target_class_id": int(target_class_id),
            "response_pullback": _RAW_TARGET_LOGIT_UNIT_COTANGENT,
            "orientation": "positive_removal",
            "relu_derivative": "indicator_preactivation_strictly_positive",
            "relu_zero_subgradient": "zero",
        },
        prefix="irg-raw-target-logit-literal-jacobian-",
    )


def _optimized_response_context_block_id(
    *,
    provider_plan_identity_hash: str,
    context_ids: Sequence[str],
) -> str:
    """Derive one ordered context-packet identity under a provider plan."""

    plan_identity = str(provider_plan_identity_hash)
    ordered_context_ids = tuple(str(value) for value in context_ids)
    if not plan_identity:
        raise ValueError("provider-plan identity must be nonempty")
    if not ordered_context_ids or any(not value for value in ordered_context_ids):
        raise ValueError("response context packet must contain nonempty context IDs")
    return (
        f"irg-prefix:{plan_identity}:"
        f"{content_hash(ordered_context_ids, prefix='irg-context-packet-')[-16:]}"
    )


def _bind_preregistered_source_capture_receipts(
    *,
    request: MeasurementRequest,
    exact_stream_receipts: Sequence[Mapping[str, object]],
    expected_context_block_ids: Sequence[str] | None = None,
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    str | None,
]:
    """Bind canonical V5 capture-plan identity without changing legacy rows."""

    receipts = tuple(exact_stream_receipts)
    source_capture_identity = request.payload.get(
        "source_capture_identity_hash"
    )
    if source_capture_identity is None:
        raw = tuple(
            item
            for item in receipts
            if str(item.get("stream_coordinate", ""))
            == _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
        )
        return receipts, raw, None
    if type(source_capture_identity) is not str or not source_capture_identity:
        raise ValueError("source-capture identity must be one nonempty string")
    if expected_context_block_ids is None:
        raise ValueError(
            "preregistered source capture requires its ordered context-packet ledger"
        )
    expected_block_ledger = tuple(str(value) for value in expected_context_block_ids)
    if (
        not expected_block_ledger
        or any(not value for value in expected_block_ledger)
        or len(set(expected_block_ledger)) != len(expected_block_ledger)
    ):
        raise ValueError(
            "source-capture context-packet ledger must be nonempty and unique"
        )
    bound_raw: list[Mapping[str, object]] = []
    bound_receipts: list[Mapping[str, object]] = []
    for item in receipts:
        if (
            str(item.get("stream_coordinate", ""))
            != _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
        ):
            bound_receipts.append(item)
            continue
        provider_plan_identity = item.get("provider_plan_identity_hash")
        if (
            provider_plan_identity is not None
            and provider_plan_identity != request.request_id
        ):
            raise RuntimeError(
                "raw target-logit capture differs from its provider plan"
            )
        raw_request_identity = item.get("request_id")
        if (
            raw_request_identity is not None
            and raw_request_identity != request.request_id
        ):
            raise RuntimeError(
                "raw target-logit capture differs from its provider request"
            )
        raw_source_capture_identity = item.get("source_capture_identity_hash")
        if (
            raw_source_capture_identity is not None
            and raw_source_capture_identity != source_capture_identity
        ):
            raise RuntimeError(
                "raw target-logit capture differs from its source-capture plan"
            )
        raw_index = len(bound_raw)
        if raw_index >= len(expected_block_ledger):
            raise RuntimeError(
                "raw target-logit capture exceeds its ordered context-packet plan"
            )
        if str(item.get("context_block_id", "")) != expected_block_ledger[raw_index]:
            raise RuntimeError(
                "raw target-logit capture differs from its ordered context-packet plan"
            )
        bound = {
            **dict(item),
            "request_id": request.request_id,
            "provider_plan_identity_hash": request.request_id,
            "source_capture_identity_hash": source_capture_identity,
        }
        bound_raw.append(bound)
        bound_receipts.append(bound)
    if not bound_raw:
        raise RuntimeError(
            "source-capture preregistration produced no raw target-logit receipt"
        )
    if len(bound_raw) != len(expected_block_ledger):
        raise RuntimeError(
            "raw target-logit capture is incomplete for its ordered context-packet plan"
        )
    return tuple(bound_receipts), tuple(bound_raw), source_capture_identity


_ALEXNET_V1_ACTIVATION_PATH_BY_AFFINE_PATH = {
    "features.0": ("relu", "features.1"),
    "features.3": ("relu", "features.4"),
    "features.6": ("relu", "features.7"),
    "features.8": ("relu", "features.9"),
    "features.10": ("relu", "features.11"),
    "classifier.1": ("relu", "classifier.2"),
    "classifier.4": ("relu", "classifier.5"),
    "classifier.6": ("identity", None),
}


def alexnet_v1_conserving_activation_bindings(
    mask_bindings: Sequence[Any],
) -> tuple[TorchConservingActivationBinding, ...]:
    """Bind the registered AlexNet affines to explicit immediate activations.

    The complete seven-affine pruning axis or the complete eight-affine
    measurement axis (including the fixed identity-logit readout) is
    required.  This helper does not infer activation topology from layer
    names; it validates the caller's bound affine paths against the frozen
    AlexNet map and refuses every other partial or extended route axis.
    """

    bindings = tuple(mask_bindings)
    module_paths = tuple(str(binding.module_path) for binding in bindings)
    full_paths = frozenset(_ALEXNET_V1_ACTIVATION_PATH_BY_AFFINE_PATH)
    prunable_paths = full_paths - {"classifier.6"}
    observed_paths = frozenset(module_paths)
    if (
        len(set(module_paths)) != len(module_paths)
        or observed_paths not in {prunable_paths, full_paths}
    ):
        raise ValueError(
            "canonical AlexNet conserving activation capture requires the "
            "complete registered seven-prunable or eight-measurement affine "
            "route axis"
        )
    result = []
    for binding in bindings:
        module_path = str(binding.module_path)
        activation_kind, activation_path = (
            _ALEXNET_V1_ACTIVATION_PATH_BY_AFFINE_PATH[module_path]
        )
        result.append(
            TorchConservingActivationBinding(
                layer_id=str(binding.layer_id),
                affine_module_path=module_path,
                activation_kind=activation_kind,
                activation_module_path=activation_path,
            )
        )
    return tuple(result)


def _resolve_response_route_chunk_size(explicit: int | None) -> int:
    """Resolve the route-chunk execution knob once at construction.

    Mirrors ``RuntimeExecutionProfile.from_environment``'s ``CIRCUIT_*``
    integer bridging: an explicit constructor value wins, then the
    environment, then the default.
    """

    if explicit is not None:
        resolved = int(explicit)
    else:
        raw = os.environ.get(RESPONSE_ROUTE_CHUNK_ROUTES_ENVIRONMENT)
        if raw is None or not str(raw).strip():
            resolved = int(RESPONSE_ROUTE_CHUNK_ROUTES_DEFAULT)
        else:
            try:
                resolved = int(str(raw).strip())
            except ValueError as error:
                raise ValueError(
                    f"{RESPONSE_ROUTE_CHUNK_ROUTES_ENVIRONMENT} must be an "
                    "integer environment value"
                ) from error
    if resolved < 1:
        raise ValueError("response route chunk size must be positive")
    return resolved


def _block_chunk_bounds(route_count: int, chunk_size: int) -> tuple[tuple[int, int], ...]:
    """Within-block ``[start, stop)`` spans covering the block exactly."""

    count = int(route_count)
    width = int(chunk_size)
    if count <= 0 or width <= 0:
        raise ValueError("route chunking requires positive extents")
    return tuple((start, min(start + width, count)) for start in range(0, count, width))


def route_chunk_spans(
    route_counts: Sequence[int],
    chunk_size: int,
) -> tuple[tuple[int, int, int, int, int], ...]:
    """Block-aligned route-chunk spans over a concatenated route axis.

    Returns ``(block_index, local_start, local_stop, global_start,
    global_stop)`` tuples.  Spans never cross a route-block boundary, are
    emitted in ascending global order, and cover the axis exactly (no
    remainder is ever dropped), so streaming chunk-by-chunk visits every
    route coordinate exactly once and in the canonical order.
    """

    counts = tuple(int(count) for count in route_counts)
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("route chunk spans require positive block counts")
    spans: list[tuple[int, int, int, int, int]] = []
    offset = 0
    for block_index, count in enumerate(counts):
        for start, stop in _block_chunk_bounds(count, chunk_size):
            spans.append((block_index, start, stop, offset + start, offset + stop))
        offset += count
    return tuple(spans)


def _route_batched_context_reduction(values: Any, mixing: Any) -> Any:
    """Contract ``[context, head, route]`` rows over contexts, route-batched.

    Computes ``out[k, h, r] = Σ_n mixing[k, n] · values[n, h, r]`` via the
    route-BATCHED contraction ``einsum("nhr,nkr->rhk")`` so each route's
    reduction over the context axis runs in a fixed-shape per-route kernel
    whose arithmetic does not depend on how many routes are present in the
    call.  Measured on torch 2.10 CPU (f64): this form is bitwise
    slice-stable for n ∈ {1, 2, 3, 32}, k ∈ {1, 3, 8} across adversarial
    chunkings, whereas the GEMV/GEMM-shaped forms ``einsum("n,nhr->hr")`` and
    ``einsum("kn,n,nhr->khr")`` are not (their context-reduction order
    changes with the column extent).  The chunked streaming path therefore
    uses this form for every context reduction it owns.
    """

    if torch is None:  # pragma: no cover - environment dependent.
        raise RuntimeError("route-batched context reduction requires PyTorch")
    contexts, _, span = (
        int(values.shape[0]),
        int(values.shape[1]),
        int(values.shape[2]),
    )
    probes = int(mixing.shape[0])
    expanded = mixing.transpose(0, 1)[:, :, None].expand(contexts, probes, span)
    return torch.einsum("nhr,nkr->rhk", values, expanded).permute(2, 1, 0)


class _ChunkedFcInputReceiptBuilder:
    """Chunk-streamed, exactly-equal device FC sign-equivalence receipt.

    Reproduces :func:`_fc_nonnegative_input_device_receipt` without ever
    holding a ``contexts × block`` tensor: the receipt's scalars are a
    maximum, a minimum, and integer counts — all order-free selections and
    exact integer sums — so combining per-chunk partials is bitwise
    identical to the one-shot reduction.
    """

    def __init__(
        self,
        *,
        weight: Any,
        context_ids: Sequence[str],
        architecture_id: str,
        layer_id: str,
        module_path: str,
    ) -> None:
        if torch is None:
            raise RuntimeError("device FC sign receipt requires PyTorch")
        self._weights = torch.as_tensor(weight, dtype=torch.float64).reshape(-1)
        self._contexts = tuple(map(str, context_ids))
        if not self._contexts or not bool(torch.isfinite(self._weights).all().item()):
            raise ValueError("device FC sign-equivalence receipt has invalid axes")
        self._architecture_id = str(architecture_id)
        self._layer_id = str(layer_id)
        self._module_path = str(module_path)
        self._nonzero = torch.abs(self._weights) > torch.finfo(torch.float64).tiny
        self._route_absolute_maximum = torch.empty_like(self._weights)
        self._filled = 0
        self._maximum_absolute_reconstructed: float | None = None
        self._minimum_reconstructed: float | None = None

    def consume(self, local_chunk: Any, start: int, stop: int) -> None:
        span = int(stop) - int(start)
        if (
            local_chunk.ndim != 2
            or tuple(local_chunk.shape) != (len(self._contexts), span)
            or int(start) != self._filled
            or not bool(torch.isfinite(local_chunk).all().item())
        ):
            raise ValueError("device FC sign-equivalence receipt has invalid axes")
        self._route_absolute_maximum[start:stop] = torch.amax(torch.abs(local_chunk), dim=0)
        nonzero = self._nonzero[start:stop]
        if bool(nonzero.any().item()):
            reconstructed = local_chunk[:, nonzero] / self._weights[start:stop][nonzero][None, :]
            chunk_maximum = float(torch.max(torch.abs(reconstructed)).item())
            chunk_minimum = float(torch.min(reconstructed).item())
            self._maximum_absolute_reconstructed = (
                chunk_maximum
                if self._maximum_absolute_reconstructed is None
                else max(self._maximum_absolute_reconstructed, chunk_maximum)
            )
            self._minimum_reconstructed = (
                chunk_minimum
                if self._minimum_reconstructed is None
                else min(self._minimum_reconstructed, chunk_minimum)
            )
        self._filled = int(stop)

    def finalize(self) -> Mapping[str, object]:
        if self._filled != int(self._weights.numel()):
            raise RuntimeError("chunked FC receipt did not cover its route block exactly")
        observed = self._maximum_absolute_reconstructed is not None
        maximum = self._maximum_absolute_reconstructed if observed else 1.0
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, maximum)
        verified = bool(observed and self._minimum_reconstructed >= -tolerance)
        payload = {
            "schema": "fc_nonnegative_input_receipt_v1",
            "architecture_scope": self._architecture_id,
            "layer_id": self._layer_id,
            "module_path": self._module_path,
            "context_ids": self._contexts,
            "verified": verified,
            "minimum_reconstructed_input": (
                None if not observed else float(self._minimum_reconstructed)
            ),
            "numerical_tolerance": tolerance,
            "nonzero_weight_route_count": int(torch.sum(self._nonzero).item()),
            "zero_activity_route_count": int(
                torch.sum(self._route_absolute_maximum <= tolerance).item()
            ),
            "zero_activity_tie_rule": "historical_weight_sign",
            "signed_fc_input_architectures_not_covered": True,
            "device_reduced": True,
            "host_context_route_rows_transferred": False,
        }
        return {
            **payload,
            "content_hash": content_hash(
                payload,
                prefix="irg-fc-nonnegative-input-",
            ),
        }


class _RouteChunkedLossOnlyConsumer(ABC):
    """Nominal protocol for the one-head, whole-axis streaming fast path.

    This is deliberately a nominal (``isinstance``) boundary instead of a
    structural/duck-typed one.  The optimized provider hands raw response
    derivatives to this consumer before the ordinary tripartite row tensor
    exists, so only an explicit nominal subclass may opt in.
    """

    # Compatibility default for consumers that do not know the distinct
    # raw-target-logit unit-pullback stream.  Production raw-logit consumers
    # must override this declaration explicitly.
    positive_removal_stream_coordinate = _BCE_POSITIVE_REMOVAL_COORDINATE

    def bind_route_chunk_size(self, chunk_size: int) -> None:
        """Bind the provider's physical route-window geometry.

        Historical consumers treat this coordinate as execution telemetry and
        therefore accept the provider value.  Consumers whose scientific
        request already seals a route-window geometry must override this
        method and fail closed on a mismatch; the provider must never mutate
        such a hash-bound coordinate by assigning ``chunk_size`` directly.
        """

        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("loss-only route chunk size must be an integer")
        if chunk_size <= 0:
            raise ValueError("loss-only route chunk size must be positive")
        self.chunk_size = int(chunk_size)

    @abstractmethod
    def begin_packet(
        self,
        context_ids: Sequence[str],
        *,
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
    ) -> None:
        """Open one observation-major packet on the declared route axis."""

    @abstractmethod
    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        """Consume the final loss head from a formed tripartite chunk."""

    @abstractmethod
    def consume_chunk_fused(
        self,
        start: int,
        stop: int,
        *,
        global_raw: Any,
        gate: Any,
        pullback: Any,
        affine_scale: float,
        score_epilogue: Any | None = None,
    ) -> None:
        """Form and stage only the loss head from raw route derivatives."""

    def consume_positive_removal_chunk(
        self,
        start: int,
        stop: int,
        rows: Any,
    ) -> None:
        """Consume canonical positive-removal rows from the one-VJP path.

        Compatibility consumers receive current-loss
        contribution rows and apply any pruning-orientation sign internally.
        The compatibility implementation therefore restores that orientation
        and delegates through ``consume_chunk``.  Production consumers may
        override this method to consume ``-G*phi*c`` directly without the
        bounded sign-conversion allocation.
        """

        self.consume_chunk(start, stop, -rows[:, None, :])

    @abstractmethod
    def end_packet(self) -> None:
        """Commit one exactly covered packet to the underlying accumulator."""

    @abstractmethod
    def route_streaming_receipt(self) -> Mapping[str, object]:
        """Return non-scientific execution telemetry for this invocation."""


class _RouteChunkedLossOnlyTee(_RouteChunkedLossOnlyConsumer):
    """Tee one emitted positive-removal stream to nominal loss-only sinks.

    The same chunk tensor object is offered to every child through
    ``consume_positive_removal_chunk``.  Children retain their own draw,
    weighting, hashing, and sufficient-statistic semantics; this adapter owns
    only the shared physical score stream.
    """

    def __init__(
        self,
        consumers: Sequence[_RouteChunkedLossOnlyConsumer],
    ) -> None:
        children = tuple(consumers)
        if len(children) < 2 or any(
            not isinstance(child, _RouteChunkedLossOnlyConsumer)
            for child in children
        ):
            raise TypeError(
                "loss-only tee requires at least two nominal reviewed consumers"
            )
        first = children[0]
        route_ids = getattr(first, "route_ids", ())
        device = torch.device(getattr(first, "device", "cpu")) if torch is not None else None
        coordinate = str(
            getattr(
                first,
                "positive_removal_stream_coordinate",
                _BCE_POSITIVE_REMOVAL_COORDINATE,
            )
        )
        stream_identity = str(getattr(first, "loss_pullback_hash", ""))
        if len(route_ids) <= 0:
            raise ValueError("loss-only tee requires a nonempty route axis")
        first_axis_hash: str | None = None
        for child in children[1:]:
            child_routes = getattr(child, "route_ids", ())
            axes_differ = len(child_routes) != len(route_ids)
            if not axes_differ and child_routes is not route_ids:
                if first_axis_hash is None:
                    first_axis_hash = ordered_string_axis_hash(
                        route_ids,
                        name="loss-only tee route axis",
                    )
                axes_differ = ordered_string_axis_hash(
                    child_routes,
                    name="loss-only tee child route axis",
                ) != first_axis_hash
            devices_differ = (
                torch is not None
                and torch.device(getattr(child, "device", "cpu")) != device
            )
            child_coordinate = str(
                getattr(
                    child,
                    "positive_removal_stream_coordinate",
                    _BCE_POSITIVE_REMOVAL_COORDINATE,
                )
            )
            child_identity = str(getattr(child, "loss_pullback_hash", ""))
            identities_differ = bool(stream_identity or child_identity) and (
                child_identity != stream_identity
            )
            if (
                axes_differ
                or devices_differ
                or child_coordinate != coordinate
                or identities_differ
            ):
                raise ValueError(
                    "loss-only tee children must share route axis, device, and "
                    "positive-removal coordinate identity"
                )
        self.consumers = children
        self.route_ids = route_ids
        self.device = device
        self.positive_removal_stream_coordinate = coordinate
        self.loss_pullback_hash = stream_identity
        self.head_ids = (
            "positive_removal:raw_target_logit"
            if coordinate == _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
            else "loss:bce",
        )
        self.chunk_size = 0
        self._packet_open = False

    def bind_route_chunk_size(self, chunk_size: int) -> None:
        if self._packet_open:
            raise RuntimeError("cannot rebind an open loss-only tee packet")
        super().bind_route_chunk_size(chunk_size)
        for child in self.consumers:
            child.bind_route_chunk_size(int(self.chunk_size))

    def begin_packet(
        self,
        context_ids: Sequence[str],
        *,
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
    ) -> None:
        if self._packet_open:
            raise RuntimeError("loss-only tee packet is already open")
        for child in self.consumers:
            child.begin_packet(
                context_ids,
                route_ids=route_ids,
                destination_transport_hash=destination_transport_hash,
                loss_pullback_hash=loss_pullback_hash,
            )
        self._packet_open = True

    def consume_positive_removal_chunk(
        self,
        start: int,
        stop: int,
        rows: Any,
    ) -> None:
        if not self._packet_open:
            raise RuntimeError("loss-only tee has no open packet")
        for child in self.consumers:
            child.consume_positive_removal_chunk(start, stop, rows)

    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        if not self._packet_open:
            raise RuntimeError("loss-only tee has no open packet")
        for child in self.consumers:
            child.consume_chunk(start, stop, rows)

    def consume_chunk_fused(
        self,
        start: int,
        stop: int,
        *,
        global_raw: Any,
        gate: Any,
        pullback: Any,
        affine_scale: float,
        score_epilogue: Any | None = None,
    ) -> None:
        if not self._packet_open:
            raise RuntimeError("loss-only tee has no open packet")
        for child in self.consumers:
            child.consume_chunk_fused(
                start,
                stop,
                global_raw=global_raw,
                gate=gate,
                pullback=pullback,
                affine_scale=affine_scale,
                score_epilogue=score_epilogue,
            )

    def end_packet(self) -> None:
        if not self._packet_open:
            raise RuntimeError("loss-only tee has no open packet")
        for child in self.consumers:
            child.end_packet()
        self._packet_open = False

    def route_streaming_receipt(self) -> Mapping[str, object]:
        if self._packet_open:
            raise RuntimeError("cannot receipt an open loss-only tee packet")
        child_receipts = tuple(
            dict(child.route_streaming_receipt())
            for child in self.consumers
        )
        host_transfer_bytes = sum(
            int(child.get("host_context_route_row_transfer_bytes", 0))
            for child in child_receipts
        )
        host_transfer_count = sum(
            int(child.get("host_context_route_row_transfer_count", 0))
            for child in child_receipts
        )
        host_transfer_roles = tuple(
            sorted(
                {
                    str(child.get("host_context_route_transfer_role", "none"))
                    for child in child_receipts
                    if str(child.get("host_context_route_transfer_role", "none"))
                    != "none"
                }
            )
        )
        return {
            "route_chunked_streaming": True,
            "route_chunk_size": int(self.chunk_size),
            "loss_only_route_streaming": True,
            "loss_only_stream_tee": True,
            "loss_only_stream_tee_child_count": len(self.consumers),
            "loss_only_stream_tee_child_receipts": child_receipts,
            "shared_positive_removal_rows": True,
            "positive_removal_stream_coordinate": (
                self.positive_removal_stream_coordinate
            ),
            "positive_removal_stream_identity_hash": self.loss_pullback_hash,
            "full_tripartite_rows_materialized": False,
            "host_context_route_rows_transferred": bool(host_transfer_bytes),
            "host_context_route_row_transfer_count": int(host_transfer_count),
            "host_context_route_row_transfer_bytes": int(host_transfer_bytes),
            "host_context_route_transfer_role": (
                "none"
                if not host_transfer_roles
                else "+".join(host_transfer_roles)
            ),
            "fused_route_moments_requested": False,
            "fused_route_moments_engaged": False,
            "fused_route_moment_update_count": 0,
            "fused_score_epilogue_receipts": (),
        }


class _RouteChunkedDualChartConsumer(_RouteChunkedLossOnlyConsumer):
    """Bind distinct corrected-work and literal-Jacobian streaming sinks.

    This nominal adapter is intentionally not a tee.  It prevents the two
    scientific objects from silently sharing row values while allowing the
    producer to derive both reductions from one capture and one VJP.
    ``evidence_consumer`` receives ``-G*phi*c`` and ``jacobian_consumer``
    receives ``-G*1[a>0]*c``.
    """

    def __init__(
        self,
        *,
        evidence_consumer: _RouteChunkedLossOnlyConsumer,
        jacobian_consumer: _RouteChunkedLossOnlyConsumer,
    ) -> None:
        if (
            not isinstance(evidence_consumer, _RouteChunkedLossOnlyConsumer)
            or not isinstance(jacobian_consumer, _RouteChunkedLossOnlyConsumer)
            or evidence_consumer is jacobian_consumer
        ):
            raise TypeError(
                "dual chart requires distinct nominal reviewed consumers"
            )
        evidence_coordinate = str(
            getattr(evidence_consumer, "positive_removal_stream_coordinate", "")
        )
        jacobian_coordinate = str(
            getattr(jacobian_consumer, "positive_removal_stream_coordinate", "")
        )
        if (
            evidence_coordinate != _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
            or jacobian_coordinate != evidence_coordinate
        ):
            raise ValueError(
                "dual-chart consumers require the raw target-logit coordinate"
            )
        evidence_routes = getattr(evidence_consumer, "route_ids", ())
        jacobian_routes = getattr(jacobian_consumer, "route_ids", ())
        axes_differ = len(evidence_routes) != len(jacobian_routes)
        if not axes_differ and evidence_routes is not jacobian_routes:
            axes_differ = ordered_string_axis_hash(
                evidence_routes,
                name="dual-chart evidence route axis",
            ) != ordered_string_axis_hash(
                jacobian_routes,
                name="dual-chart Jacobian route axis",
            )
        evidence_device = (
            torch.device(getattr(evidence_consumer, "device", "cpu"))
            if torch is not None
            else None
        )
        jacobian_device = (
            torch.device(getattr(jacobian_consumer, "device", "cpu"))
            if torch is not None
            else None
        )
        if axes_differ or evidence_device != jacobian_device:
            raise ValueError(
                "dual-chart consumers must share the route axis and device"
            )
        evidence_identity = str(
            getattr(evidence_consumer, "loss_pullback_hash", "")
        )
        jacobian_identity = str(
            getattr(jacobian_consumer, "loss_pullback_hash", "")
        )
        if (
            not evidence_identity
            or not jacobian_identity
            or evidence_identity == jacobian_identity
        ):
            raise ValueError(
                "dual-chart evidence and Jacobian stream identities must be distinct"
            )
        self.evidence_consumer = evidence_consumer
        self.jacobian_consumer = jacobian_consumer
        self.route_ids = evidence_routes
        self.device = evidence_device
        self.positive_removal_stream_coordinate = evidence_coordinate
        self.loss_pullback_hash = evidence_identity
        self.literal_jacobian_stream_identity_hash = jacobian_identity
        self.head_ids = (
            "corrected_evidence:raw_target_logit",
            "literal_gate_jacobian:raw_target_logit",
        )
        self.chunk_size = 0

    def bind_route_chunk_size(self, chunk_size: int) -> None:
        super().bind_route_chunk_size(chunk_size)
        self.evidence_consumer.bind_route_chunk_size(int(self.chunk_size))
        self.jacobian_consumer.bind_route_chunk_size(int(self.chunk_size))

    @staticmethod
    def _refuse_collapsed_stream() -> None:
        raise RuntimeError(
            "dual-chart adapter cannot consume one collapsed row stream"
        )

    def begin_packet(
        self,
        context_ids: Sequence[str],
        *,
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
    ) -> None:
        self._refuse_collapsed_stream()

    def consume_positive_removal_chunk(
        self,
        start: int,
        stop: int,
        rows: Any,
    ) -> None:
        self._refuse_collapsed_stream()

    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        self._refuse_collapsed_stream()

    def consume_chunk_fused(
        self,
        start: int,
        stop: int,
        *,
        global_raw: Any,
        gate: Any,
        pullback: Any,
        affine_scale: float,
        score_epilogue: Any | None = None,
    ) -> None:
        self._refuse_collapsed_stream()

    def end_packet(self) -> None:
        self._refuse_collapsed_stream()

    def route_streaming_receipt(self) -> Mapping[str, object]:
        evidence = dict(self.evidence_consumer.route_streaming_receipt())
        jacobian = dict(self.jacobian_consumer.route_streaming_receipt())
        host_transfer_bytes = sum(
            int(child.get("host_context_route_row_transfer_bytes", 0))
            for child in (evidence, jacobian)
        )
        host_transfer_count = sum(
            int(child.get("host_context_route_row_transfer_count", 0))
            for child in (evidence, jacobian)
        )
        host_transfer_roles = tuple(
            sorted(
                {
                    str(child.get("host_context_route_transfer_role", "none"))
                    for child in (evidence, jacobian)
                    if str(child.get("host_context_route_transfer_role", "none"))
                    != "none"
                }
            )
        )
        return {
            "route_chunked_streaming": True,
            "route_chunk_size": int(self.chunk_size),
            "loss_only_route_streaming": True,
            "dual_chart_streaming": True,
            "dual_chart_child_count": 2,
            "dual_chart_evidence_receipt": evidence,
            "dual_chart_jacobian_receipt": jacobian,
            "loss_only_stream_tee": False,
            "loss_only_stream_tee_child_count": 0,
            "loss_only_stream_tee_child_receipts": (),
            "shared_positive_removal_rows": False,
            "same_capture_and_vjp_required": True,
            "corrected_evidence_stream_identity_hash": self.loss_pullback_hash,
            "literal_jacobian_stream_identity_hash": (
                self.literal_jacobian_stream_identity_hash
            ),
            "positive_removal_stream_coordinate": (
                self.positive_removal_stream_coordinate
            ),
            "positive_removal_stream_identity_hash": self.loss_pullback_hash,
            "full_tripartite_rows_materialized": False,
            "host_context_route_rows_transferred": bool(host_transfer_bytes),
            "host_context_route_row_transfer_count": int(host_transfer_count),
            "host_context_route_row_transfer_bytes": int(host_transfer_bytes),
            "host_context_route_transfer_role": (
                "none"
                if not host_transfer_roles
                else "+".join(host_transfer_roles)
            ),
            "fused_route_moments_requested": False,
            "fused_route_moments_engaged": False,
            "fused_route_moment_update_count": 0,
            "fused_score_epilogue_receipts": (),
        }


def build_route_chunked_dual_chart_fanout(
    *,
    evidence_consumers: Sequence[_RouteChunkedLossOnlyConsumer],
    jacobian_consumers: Sequence[_RouteChunkedLossOnlyConsumer],
) -> _RouteChunkedLossOnlyConsumer:
    """Build one dual stream with independent branch fan-out.

    Every evidence child receives the same corrected ``-G*phi*c`` chunk and
    every Jacobian child receives the same literal ``-G*1[a>0]*c`` chunk.  The
    outer dual adapter remains the only producer-facing object, so one capture
    and one VJP feed all registered branch projections without ever collapsing
    the two scientific row streams.  A singleton side is passed through
    directly; a multi-child side uses the bounded route-chunk tee.
    """

    evidence = tuple(evidence_consumers)
    jacobian = tuple(jacobian_consumers)
    if (
        not evidence
        or not jacobian
        or len(evidence) != len(jacobian)
        or any(
            not isinstance(value, _RouteChunkedLossOnlyConsumer)
            for value in (*evidence, *jacobian)
        )
    ):
        raise TypeError(
            "dual-chart fan-out requires aligned nonempty reviewed consumer axes"
        )
    if len({id(value) for value in (*evidence, *jacobian)}) != (
        len(evidence) + len(jacobian)
    ):
        raise ValueError("dual-chart fan-out consumers must be distinct objects")
    evidence_sink = (
        evidence[0]
        if len(evidence) == 1
        else _RouteChunkedLossOnlyTee(evidence)
    )
    jacobian_sink = (
        jacobian[0]
        if len(jacobian) == 1
        else _RouteChunkedLossOnlyTee(jacobian)
    )
    return _RouteChunkedDualChartConsumer(
        evidence_consumer=evidence_sink,
        jacobian_consumer=jacobian_sink,
    )


class _RouteChunkedTripartiteUpdate:
    """Apply one packet's tripartite update from route-chunked row slabs.

    Streams the exact per-packet sufficient-statistic update of
    :class:`DeviceTripartiteMomentAccumulator.update` (unit context
    weights) into the accumulator's state slice-by-slice, so no
    ``contexts × full-route-axis`` tensor is ever formed.  Per-route context
    reductions use :func:`_route_batched_context_reduction` (and, for the
    route-batched second-moment recipes, the accumulator's own einsum
    string), whose arithmetic is independent of the route extent of a call;
    all other per-route operations are elementwise.  Packet-scoped scalars
    (total weight, sample count, sketch constant, context identifiers) are
    committed exactly once per packet, after the axis is covered exactly.
    """

    def __init__(
        self,
        *,
        accumulator: Any,
        route_total: int,
        chunk_size: int,
        identity_error: str,
    ) -> None:
        if torch is None:
            raise RuntimeError("route-chunked streaming requires PyTorch")
        self._accumulator = accumulator
        self._route_total = int(route_total)
        self.chunk_size = int(chunk_size)
        self._identity_error = str(identity_error)
        self._context_ids: tuple[str, ...] = ()
        self._weights: Any | None = None
        self._weight_total = 0.0
        self._weight_squared_total = 0.0
        self._signs: Any | None = None
        self._root: Any | None = None
        self._sign_mixing: Any | None = None
        self._ones_mixing: Any | None = None
        self._next_route = 0
        self._packet_open = False
        self._first_packet = False
        self.packet_count = 0
        self.chunk_update_count = 0
        self.fused_chunk_update_count = 0
        self.fused_requested = False
        self.fused_score_epilogue_receipts: list[Mapping[str, object]] = []

    def begin_packet(
        self,
        context_ids: Sequence[str],
        *,
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
    ) -> None:
        accumulator = self._accumulator
        if (
            str(destination_transport_hash or "") != accumulator.destination_transport_hash
            or str(loss_pullback_hash or "") != accumulator.loss_pullback_hash
        ):
            raise ValueError(self._identity_error)
        ids = tuple(map(str, context_ids))
        if (
            not ids
            or len(set(ids)) != len(ids)
            or set(ids).intersection(accumulator.context_ids)
            or self._packet_open
        ):
            raise ValueError("device tripartite rows/context axis is invalid")
        self._first_packet = accumulator.sample_count == 0
        accumulator._ensure_storage()
        # The chunked path carries the census consumers' weights=None
        # semantics only: every context weighs exactly one, as in the
        # legacy ``consume`` closures.
        weights = torch.ones(len(ids), dtype=torch.float64, device=accumulator.device)
        self._context_ids = ids
        self._weights = weights
        self._weight_total = float(weights.sum().item())
        self._weight_squared_total = float(torch.sum(weights * weights).item())
        self._ones_mixing = torch.ones(
            (1, len(ids)), dtype=torch.float64, device=accumulator.device
        )
        if accumulator.sketch_rank:
            self._signs = accumulator._signs(ids)
            self._root = torch.sqrt(weights)
            # root is exactly one per context here, so folding it into the
            # sign matrix is a bitwise no-op and preserves the summand
            # values ``signs[k, n] · (√w[n] · values[n, h, r])`` exactly.
            self._sign_mixing = self._signs * self._root[None, :]
        else:
            self._signs = None
            self._root = None
            self._sign_mixing = None
        self._next_route = 0
        self._packet_open = True
        self.packet_count += 1

    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        accumulator = self._accumulator
        # This chunked path bypasses ``DeviceTripartiteMomentAccumulator.update``,
        # whose sink-level detach is the authority that prevents a persistent
        # moment tensor from retaining the census autograd graph.  Detach here
        # before any copy_/add_: otherwise each persistent slice acquires a
        # CopySlices/AddBackward lineage and keeps full-route CUDA activations
        # alive across strata and looks.
        rows = rows.detach()
        head_count = len(accumulator.head_ids)
        span = int(stop) - int(start)
        if (
            not self._packet_open
            or int(start) != self._next_route
            or span <= 0
            or int(stop) > self._route_total
            or tuple(rows.shape) != (len(self._context_ids), head_count, span)
            or rows.dtype != torch.float64
            or not bool(torch.isfinite(rows).all().item())
        ):
            raise ValueError("device tripartite rows/context axis is invalid")
        weights = self._weights
        write = "copy_" if self._first_packet else "add_"
        targets = accumulator.windowed_slabs(int(start), int(stop))
        getattr(targets["sum_rows"], write)(
            _route_batched_context_reduction(rows, self._ones_mixing)[0]
        )
        getattr(targets["sum_outer"], write)(torch.einsum("n,nhr,njr->rhj", weights, rows, rows))
        squared = rows * rows
        getattr(targets["sum_outer_squared"], write)(
            torch.einsum("n,nhr,njr->rhj", weights, squared, squared)
        )
        activity = torch.stack(
            (
                torch.abs(rows[:, 0, :]),
                torch.sqrt(torch.sum(rows[:, 1:-1, :] ** 2, dim=1)),
                torch.abs(rows[:, -1, :]),
            ),
            dim=1,
        )
        getattr(targets["sum_view_activity"], write)(
            _route_batched_context_reduction(activity, self._ones_mixing)[0]
        )
        getattr(targets["sum_view_activity_squared"], write)(
            _route_batched_context_reduction(activity * activity, self._ones_mixing)[0]
        )
        if accumulator.sketch_rank:
            getattr(targets["sketch_sum"], write)(
                _route_batched_context_reduction(rows, self._sign_mixing)
            )
        if accumulator.host_primary:
            accumulator.commit_window(int(start), int(stop))
        self._next_route = int(stop)
        self.chunk_update_count += 1

    def consume_chunk_fused(
        self,
        start: int,
        stop: int,
        *,
        local_raw: Any,
        global_raw: Any,
        dest_scale: Any,
        gate: Any,
        pullback: Any,
        affine_scale: float,
        score_epilogue: Any | None = None,
    ) -> None:
        """Fuse head formation and all moment updates for one route chunk."""

        from mechanistic_systems.measurement.fused_route_moments import (
            fused_route_chunk_update,
        )

        accumulator = self._accumulator
        span = int(stop) - int(start)
        local_raw = local_raw.detach()
        global_raw = global_raw.detach()
        if (
            not self._packet_open
            or int(start) != self._next_route
            or span <= 0
            or int(stop) > self._route_total
            or tuple(local_raw.shape) != (len(self._context_ids), span)
            or tuple(global_raw.shape) != (len(self._context_ids), span)
            or local_raw.dtype != torch.float64
            or global_raw.dtype != torch.float64
        ):
            raise ValueError("device tripartite rows/context axis is invalid")
        if score_epilogue is None and (
            not bool(torch.isfinite(local_raw).all().item())
            or not bool(torch.isfinite(global_raw).all().item())
        ):
            raise ValueError("device tripartite rows/context axis is invalid")
        fused_route_chunk_update(
            accumulator,
            local_raw=local_raw,
            global_raw=global_raw,
            dest_scale=dest_scale,
            gate=gate,
            pullback=pullback,
            weights=self._weights,
            affine_scale=float(affine_scale),
            route_start=int(start),
            signs=self._signs,
            score_state=score_epilogue,
            initialize=self._first_packet,
        )
        self._next_route = int(stop)
        self.chunk_update_count += 1
        self.fused_chunk_update_count += 1

    def end_packet(self) -> None:
        accumulator = self._accumulator
        if not self._packet_open or self._next_route != self._route_total:
            raise RuntimeError("route-chunked streaming did not cover the route axis exactly")
        accumulator.total_weight += self._weight_total
        accumulator.sum_weight_squared += self._weight_squared_total
        accumulator.sample_count += len(self._context_ids)
        if accumulator.sketch_rank:
            write = "copy_" if self._first_packet else "add_"
            getattr(accumulator.sketch_constant, write)(
                torch.einsum("kn,n->k", self._signs, self._root)
            )
        elif self._first_packet:
            accumulator.sketch_sum.zero_()
            accumulator.sketch_constant.zero_()
        accumulator._context_ids.extend(self._context_ids)
        self._packet_open = False
        self._context_ids = ()
        self._weights = None
        self._signs = None
        self._root = None
        self._sign_mixing = None
        self._ones_mixing = None
        self._next_route = 0
        self._first_packet = False


def resnet_local_destination_transport(
    evaluator: Any,
) -> tuple[dict[str, np.ndarray], str]:
    """Resolve each route block's immediate post-affine destination scale."""

    route_blocks = tuple(getattr(evaluator, "route_blocks", ()))
    bindings = tuple(getattr(evaluator, "mask_bindings", ()))
    model = getattr(evaluator, "source_model", None)
    if not route_blocks or len(route_blocks) != len(bindings) or model is None:
        raise ValueError("local destination transport requires aligned route bindings")
    strict_resnet = "resnet" in str(route_blocks[0].architecture_id).lower()
    scales: dict[str, np.ndarray] = {}
    identities: list[tuple[str, str]] = []
    for block, binding in zip(route_blocks, bindings, strict=True):
        layer_id = str(block.layer_id)
        route_kind = str(binding.route_kind)
        if route_kind not in {"conv_channel_pair", "conv_kernel_weight"}:
            scales[layer_id] = np.ones(int(block.shape[0]), dtype=np.float64)
            identities.append(
                (
                    layer_id,
                    (
                        "identity_preadd_branch_v1"
                        if route_kind == "residual_stream_gate"
                        else "identity_direct_destination_v1"
                    ),
                )
            )
            continue
        path = str(binding.module_path)
        parent, _, leaf = path.rpartition(".")
        if leaf.startswith("conv") and leaf[4:].isdigit():
            bn_leaf = "bn" + leaf[4:]
            bn_path = f"{parent}.{bn_leaf}" if parent else bn_leaf
        elif leaf == "0" and parent.endswith("downsample"):
            bn_path = f"{parent}.1"
        elif not strict_resnet:
            scales[layer_id] = np.ones(int(block.shape[0]), dtype=np.float64)
            identities.append((layer_id, "identity_direct_destination_v1"))
            continue
        else:
            raise ValueError(
                "registered ResNet local attribution cannot resolve the "
                f"destination normalization for {path!r}"
            )
        batch_norm = model.get_submodule(bn_path)
        running_var = getattr(batch_norm, "running_var", None)
        if running_var is None:
            raise ValueError(f"local destination normalization {bn_path!r} lacks running variance")
        affine_weight = getattr(batch_norm, "weight", None)
        gamma = (
            np.ones(int(running_var.numel()), dtype=np.float64)
            if affine_weight is None
            else affine_weight.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        variance = running_var.detach().cpu().numpy().astype(np.float64, copy=False)
        scale = gamma / np.sqrt(variance + float(getattr(batch_norm, "eps")))
        if scale.shape != (int(block.shape[0]),) or not np.isfinite(scale).all():
            raise ValueError("local destination affine transport is invalid")
        scales[layer_id] = np.ascontiguousarray(scale)
        identities.append((layer_id, f"eval_batchnorm_linear_transport_v1::{bn_path}"))
    return scales, content_hash(
        {
            "schema": "resnet_local_destination_transport_v1",
            "identities": tuple(identities),
            "scales": tuple((layer_id, scales[layer_id]) for layer_id, _ in identities),
        },
        prefix="irg-local-destination-transport-",
    )


def resnet_local_regime_capture_specs(
    evaluator: Any,
) -> dict[str, LocalRegimeCaptureSpec]:
    """Bind eligible ResNet Conv routes to their exact terminal ReLU input.

    Intermediate convolutions use their following eval-mode BatchNorm affine
    coordinate. Final/projection convolutions use the containing gated
    residual block's post-add, pre-ReLU coordinate. Readout and residual-stream
    routes intentionally receive no singleton Conv-route diagnostic here.
    """

    route_blocks = tuple(getattr(evaluator, "route_blocks", ()))
    bindings = tuple(getattr(evaluator, "mask_bindings", ()))
    model = getattr(evaluator, "source_model", None)
    if (
        not route_blocks
        or len(route_blocks) != len(bindings)
        or model is None
        or "resnet" not in str(route_blocks[0].architecture_id).lower()
    ):
        return {}
    result: dict[str, LocalRegimeCaptureSpec] = {}
    for block, binding in zip(route_blocks, bindings, strict=True):
        if str(binding.route_kind) != "conv_channel_pair":
            continue
        path = str(binding.module_path)
        parent, _, leaf = path.rpartition(".")
        residual_block_path: str | None = None
        if leaf.startswith("conv") and leaf[4:].isdigit():
            convolution_index = int(leaf[4:])
            bn_leaf = f"bn{convolution_index}"
            bn_path = f"{parent}.{bn_leaf}" if parent else bn_leaf
            if parent:
                parent_module = model.get_submodule(parent)
                is_final = not hasattr(parent_module, f"conv{convolution_index + 1}")
                if is_final:
                    capture_source = "residual_output_preactivation"
                    residual_block_path = parent
                else:
                    capture_source = "affine_bound_output"
            else:
                capture_source = "affine_bound_output"
        elif leaf == "0" and parent.endswith("downsample"):
            bn_path = f"{parent}.1"
            capture_source = "residual_output_preactivation"
            residual_block_path = parent.rsplit(".downsample", 1)[0]
        else:
            continue
        batch_norm = model.get_submodule(bn_path)
        running_mean = getattr(batch_norm, "running_mean", None)
        running_var = getattr(batch_norm, "running_var", None)
        if running_mean is None or running_var is None:
            raise ValueError(f"local regime destination {bn_path!r} lacks eval statistics")
        channels = int(running_var.numel())
        affine_weight = getattr(batch_norm, "weight", None)
        affine_bias = getattr(batch_norm, "bias", None)
        gamma = (
            np.ones(channels, dtype=np.float64)
            if affine_weight is None
            else affine_weight.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        beta = (
            np.zeros(channels, dtype=np.float64)
            if affine_bias is None
            else affine_bias.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        mean = running_mean.detach().cpu().numpy().astype(np.float64, copy=False)
        variance = running_var.detach().cpu().numpy().astype(np.float64, copy=False)
        scale = gamma / np.sqrt(variance + float(getattr(batch_norm, "eps")))
        offset = beta - scale * mean
        if (
            scale.shape != (int(block.shape[0]),)
            or offset.shape != scale.shape
            or not np.isfinite(scale).all()
            or not np.isfinite(offset).all()
        ):
            raise ValueError("local regime affine destination is invalid")
        geometry_hash = content_hash(
            {
                "schema": "resnet_relu_local_regime_geometry_v1",
                "layer_id": str(block.layer_id),
                "module_path": path,
                "normalization_path": bn_path,
                "capture_source": capture_source,
                "residual_block_path": residual_block_path,
                "scale": scale,
                "offset": offset,
                "activation_kind": "relu",
            },
            prefix="irg-local-regime-geometry-",
        )
        result[str(block.layer_id)] = LocalRegimeCaptureSpec(
            layer_id=str(block.layer_id),
            route_scale=tuple(scale),
            affine_offset=tuple(offset),
            capture_source=capture_source,
            residual_block_path=residual_block_path,
            geometry_hash=geometry_hash,
        )
    return result


def fc_nonnegative_input_receipt(
    *,
    local_context_rows: np.ndarray,
    weight: np.ndarray,
    context_ids: Sequence[str],
    architecture_id: str,
    layer_id: str,
    module_path: str,
) -> Mapping[str, object]:
    """Verify the exact scope in which FC local and weight signs coincide."""

    local = np.asarray(local_context_rows, dtype=np.float64)
    weights = np.asarray(weight, dtype=np.float64).reshape(-1)
    contexts = tuple(map(str, context_ids))
    if (
        local.ndim != 2
        or local.shape != (len(contexts), weights.size)
        or not contexts
        or not np.isfinite(local).all()
        or not np.isfinite(weights).all()
    ):
        raise ValueError("FC sign-equivalence receipt has invalid axes")
    nonzero = np.abs(weights) > np.finfo(np.float64).tiny
    reconstructed = np.divide(
        local[:, nonzero],
        weights[nonzero][None, :],
    )
    tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            float(np.max(np.abs(reconstructed))) if reconstructed.size else 1.0,
        )
    )
    verified = bool(reconstructed.size and np.all(reconstructed >= -tolerance))
    zero_activity = np.all(np.abs(local) <= tolerance, axis=0)
    payload = {
        "schema": "fc_nonnegative_input_receipt_v1",
        "architecture_scope": str(architecture_id),
        "layer_id": str(layer_id),
        "module_path": str(module_path),
        "context_ids": contexts,
        "verified": verified,
        "minimum_reconstructed_input": (
            None if not reconstructed.size else float(np.min(reconstructed))
        ),
        "numerical_tolerance": tolerance,
        "nonzero_weight_route_count": int(np.sum(nonzero)),
        "zero_activity_route_count": int(np.sum(zero_activity)),
        "zero_activity_tie_rule": "historical_weight_sign",
        "signed_fc_input_architectures_not_covered": True,
    }
    return {
        **payload,
        "content_hash": content_hash(
            payload,
            prefix="irg-fc-nonnegative-input-",
        ),
    }


def _fc_nonnegative_input_device_receipt(
    *,
    local_context_rows: Any,
    weight: Any,
    context_ids: Sequence[str],
    architecture_id: str,
    layer_id: str,
    module_path: str,
) -> Mapping[str, object]:
    """GPU-reduced equivalent of :func:`fc_nonnegative_input_receipt`."""

    if torch is None:
        raise RuntimeError("device FC sign receipt requires PyTorch")
    local = torch.as_tensor(local_context_rows, dtype=torch.float64)
    weights = torch.as_tensor(weight, device=local.device, dtype=torch.float64).reshape(-1)
    contexts = tuple(map(str, context_ids))
    if (
        local.ndim != 2
        or tuple(local.shape) != (len(contexts), int(weights.numel()))
        or not contexts
        or not bool(torch.isfinite(local).all().item())
        or not bool(torch.isfinite(weights).all().item())
    ):
        raise ValueError("device FC sign-equivalence receipt has invalid axes")
    nonzero = torch.abs(weights) > torch.finfo(torch.float64).tiny
    reconstructed = local[:, nonzero] / weights[nonzero][None, :]
    maximum = (
        float(torch.max(torch.abs(reconstructed)).item()) if int(reconstructed.numel()) else 1.0
    )
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, maximum)
    verified = bool(int(reconstructed.numel()) and torch.all(reconstructed >= -tolerance).item())
    zero_activity = torch.all(torch.abs(local) <= tolerance, dim=0)
    payload = {
        "schema": "fc_nonnegative_input_receipt_v1",
        "architecture_scope": str(architecture_id),
        "layer_id": str(layer_id),
        "module_path": str(module_path),
        "context_ids": contexts,
        "verified": verified,
        "minimum_reconstructed_input": (
            None if not int(reconstructed.numel()) else float(torch.min(reconstructed).item())
        ),
        "numerical_tolerance": tolerance,
        "nonzero_weight_route_count": int(torch.sum(nonzero).item()),
        "zero_activity_route_count": int(torch.sum(zero_activity).item()),
        "zero_activity_tie_rule": "historical_weight_sign",
        "signed_fc_input_architectures_not_covered": True,
        "device_reduced": True,
        "host_context_route_rows_transferred": False,
    }
    return {
        **payload,
        "content_hash": content_hash(
            payload,
            prefix="irg-fc-nonnegative-input-",
        ),
    }


class _RequestScopedContextBatchProvider:
    """Expose one frozen IRG context prefix through the scoring protocol."""

    provider_id = "irg-request-scoped-context-prefix-v1"

    def __init__(
        self,
        batch_factory: Callable[[tuple[str, ...]], Iterable[MeasurementInputBatch]],
    ) -> None:
        self._batch_factory = batch_factory
        self._active_block_id = ""
        self._active_context_ids: tuple[str, ...] = ()

    def activate(
        self,
        *,
        context_block_id: str,
        context_ids: Sequence[str],
    ) -> None:
        identifiers = tuple(map(str, context_ids))
        if (
            not str(context_block_id)
            or not identifiers
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError("optimized scoring requires a unique nonempty context prefix")
        self._active_block_id = str(context_block_id)
        self._active_context_ids = identifiers

    def iter_batches(
        self,
        _plan: MeasurementPlan,
        *,
        context_block_id: str,
    ) -> Iterable[MeasurementInputBatch]:
        del _plan
        if str(context_block_id) != self._active_block_id or not self._active_context_ids:
            raise RuntimeError("optimized scoring context prefix was not activated")
        offset = 0
        for batch in self._batch_factory(self._active_context_ids):
            batch.validate()
            expected = self._active_context_ids[offset : offset + len(batch.context_ids)]
            if tuple(batch.context_ids) != expected:
                raise ValueError(
                    "optimized scoring batch factory changed context order or coverage"
                )
            offset += len(batch.context_ids)
            yield batch
        if offset != len(self._active_context_ids):
            raise ValueError("optimized scoring batch factory omitted requested contexts")


class OptimizedScoringResponseProvider:
    """Stream v1/v2 response geometry through registered scoring backends.

    The default ``is_grads_batched`` VJP policy is the production scoring
    policy.  Route reduction is resolved by the backend registry on
    ``(route_family, device, dtype)``; eligible CUDA families therefore use
    their registered custom kernels and fail/fallback according to that
    registry's policy.  This class never calls ``torch.autograd`` directly.
    """

    supported_kinds = StreamingResponseGeometryProvider.supported_kinds

    def __init__(
        self,
        *,
        evaluator: Any,
        measurement_view: Any,
        context_ids: Sequence[str],
        batch_factory: Callable[[tuple[str, ...]], Iterable[MeasurementInputBatch]],
        mask_resolver: Callable[[str], HardMaskView],
        calibration_resolver: Callable[[HardMaskView], Any],
        route_ids: Sequence[str],
        queries: Sequence[QuerySpec] | None,
        response_spec_hash: str,
        response_metric_id: str,
        study_hash: str,
        autograd_backend: str = "is_grads_batched",
        backend_registry: Any | None = None,
        batch_size: int = 32,
        response_route_chunk_size: int | None = None,
        artifact_root: str | None = None,
        projection_residual_bound: float = 0.0,
        sketch_residual_spectral_bound: float = 0.0,
        omitted_route_residual_bound: float = 0.0,
        apply_affine_calibration_to_response: bool = False,
        pruning_guiding_loss_id: str = CALIBRATED_OVR_BCE,
        census_dense_calibration_state_hash: str | None = None,
        decision_fixed_offset_resolver: Callable[[Sequence[str]], Any] | None = None,
        local_destination_scales_by_layer: Mapping[str, np.ndarray] | None = None,
        local_destination_transport_hash: str | None = None,
        local_covariance_sketch_rank: int = 8,
        local_covariance_sketch_seed: int = 0,
        local_regime_specs_by_layer: Mapping[str, LocalRegimeCaptureSpec] | None = None,
        conserving_activation_bindings: Sequence[
            TorchConservingActivationBinding
        ] | None = None,
        in_process_typed_response_fast_path: bool = False,
        progress_callback: Callable[..., None] | None = None,
        factored_numbers_per_context: int | None = None,
    ) -> None:
        if factored_numbers_per_context is not None and (
            isinstance(factored_numbers_per_context, bool)
            or not isinstance(factored_numbers_per_context, int)
            or factored_numbers_per_context <= 0
        ):
            raise ValueError(
                "factored_numbers_per_context must be a positive integer when declared"
            )
        # Declared custody-forecast numeral for the factored D protocol:
        # per-context factor numbers summed over the bound
        # affine layers (input + post-activation cotangent + preactivation).
        # ``None`` means undeclared; the factored backend then refuses with
        # ``FACTORED_PROVIDER_ABSENT`` instead of inferring a forecast.
        self._factored_numbers_per_context = factored_numbers_per_context
        # Installed by the source-scoped IRG owner; no numerical authority flag.
        self._factored_capture_admission = None
        route_axis = canonical_unique_string_axis(
            route_ids,
            name="optimized IRG scoring route axis",
        )
        populations = tuple(map(str, context_ids))
        selected_queries = tuple(queries or getattr(evaluator, "queries", ()))
        route_blocks = tuple(getattr(evaluator, "route_blocks", ()))
        mask_bindings = tuple(getattr(evaluator, "mask_bindings", ()))
        if (
            not route_axis
            or not populations
            or not selected_queries
            or not route_blocks
            or len(route_blocks) != len(mask_bindings)
        ):
            raise ValueError(
                "optimized IRG scoring requires contexts, queries, and aligned route bindings"
            )
        if sum(block.route_count for block in route_blocks) != len(route_axis):
            raise ValueError("optimized scorer route blocks differ from the IRG route axis")
        if len(str(study_hash)) != 64:
            raise ValueError("optimized scoring requires a 64-character study identity")

        self.evaluator = evaluator
        self.measurement_view = measurement_view
        self.context_ids = populations
        self.route_ids = route_axis
        self.queries = selected_queries
        self.mask_resolver = mask_resolver
        self.calibration_resolver = calibration_resolver
        self.study_hash = str(study_hash)
        self.progress_callback = progress_callback
        self.in_process_typed_response_fast_path = bool(in_process_typed_response_fast_path)
        self._route_blocks = route_blocks
        self._mask_bindings = mask_bindings
        self._lock = RLock()
        self._active_mask: HardMaskView | None = None
        self._active_calibration: Any | None = None
        self._active_request_id = ""
        self._census_guarded_request_id = ""
        self._census_pullback_identities: dict[tuple[str, int, str, str], str] = {}
        self._active_fc_nonnegative_input_receipts: tuple[Mapping[str, object], ...] = ()
        self._active_flip_margin_accumulator: DenseFlipMarginAccumulator | None = None
        self._active_context_packet_count = 0
        self._active_host_context_route_rows_transfer_count = 0
        self._active_exact_loss_stream_receipts: list[Mapping[str, object]] = []
        self._active_gate_disclosure_accumulator: Any | None = None
        self._sequential_device_states: dict[
            tuple[str, str, str, str, int],
            dict[str, Any],
        ] = {}
        self.response_route_chunk_size = _resolve_response_route_chunk_size(
            response_route_chunk_size
        )
        self._device_chunk_state: (
            _RouteChunkedTripartiteUpdate | _RouteChunkedLossOnlyConsumer | None
        ) = None
        self.apply_affine_calibration_to_response = bool(apply_affine_calibration_to_response)
        if pruning_guiding_loss_id not in PRUNING_GUIDING_LOSSES:
            raise ValueError("unsupported pruning-guiding census loss")
        self.pruning_guiding_loss_id = str(pruning_guiding_loss_id)
        self.decision_fixed_offset_resolver = decision_fixed_offset_resolver
        self.census_dense_calibration_state_hash = (
            None if census_dense_calibration_state_hash is None
            else str(census_dense_calibration_state_hash)
        )
        if self.census_dense_calibration_state_hash is not None and (
            not self.census_dense_calibration_state_hash
            or not self.apply_affine_calibration_to_response
        ):
            raise ValueError("dense census loss requires its calibration hash and affine response wiring")
        self.local_destination_scales_by_layer = (
            None
            if local_destination_scales_by_layer is None
            else {
                str(layer_id): np.asarray(scale, dtype=np.float64).reshape(-1)
                for layer_id, scale in local_destination_scales_by_layer.items()
            }
        )
        self.local_destination_transport_hash = (
            None
            if local_destination_transport_hash is None
            else str(local_destination_transport_hash)
        )
        self.local_regime_specs_by_layer = dict(local_regime_specs_by_layer or {})
        self.conserving_activation_bindings = tuple(
            conserving_activation_bindings or ()
        )
        if not set(self.local_regime_specs_by_layer).issubset(
            {str(block.layer_id) for block in route_blocks}
        ):
            raise ValueError("local regime specifications differ from the route blocks")
        if (self.local_destination_scales_by_layer is None) != (
            self.local_destination_transport_hash is None
        ):
            raise ValueError("local attribution requires destination scales and their identity")
        if self.local_destination_scales_by_layer is not None:
            expected_layers = {str(block.layer_id) for block in route_blocks}
            if set(self.local_destination_scales_by_layer) != expected_layers:
                raise ValueError("local destination scales differ from the route blocks")
            for block in route_blocks:
                local_scale = self.local_destination_scales_by_layer[str(block.layer_id)]
                if (
                    local_scale.shape != (int(block.shape[0]),)
                    or not np.isfinite(local_scale).all()
                ):
                    raise ValueError("local destination scale differs from its output axis")
        if self.apply_affine_calibration_to_response and len(selected_queries) != 1:
            raise ValueError(
                "positive-affine calibrated response geometry currently "
                "requires exactly one target-logit query"
            )

        self._batch_provider = _RequestScopedContextBatchProvider(batch_factory)
        layer_bindings = tuple(
            TorchLayerBinding(
                layer_id=str(binding.layer_id),
                module_path=str(binding.module_path),
                route_kind=str(binding.route_kind),
                capture_policy=(
                    "residual_premerge_decomposition"
                    if str(binding.route_kind) == "residual_stream_gate"
                    else "clone_for_downstream"
                ),
            )
            for binding in mask_bindings
        )
        self.model_adapter = TorchModuleMeasurementAdapter(
            model=evaluator.source_model,
            architecture_id=str(route_blocks[0].architecture_id),
            topology_hash=str(evaluator.topology_hash),
            route_schema_hash=str(evaluator.route_schema_hash),
            layer_bindings=layer_bindings,
            batch_provider=self._batch_provider,
            device=evaluator.device,
            response_selector=evaluator.response_selector,
            parity_receipt_hash=str(getattr(evaluator, "parity_receipt_hash", "")),
            compiled_gate_graph=evaluator.gate_graph,
            local_regime_specs_by_layer=self.local_regime_specs_by_layer,
            conserving_activation_bindings=(
                self.conserving_activation_bindings
            ),
        )
        self.scoring_backend = NativeLayerBlockedMeasurementAdapter(
            self.model_adapter,
            query_tile_size=len(self.queries),
            autograd_backend=autograd_backend,
            backend_registry=backend_registry,
            retain_context_factor=True,
            retain_local_attribution=(self.local_destination_scales_by_layer is not None),
            retain_local_regime=bool(self.local_regime_specs_by_layer),
            retain_context_observations=(
                self.local_destination_scales_by_layer is not None and len(self.queries) == 1
            ),
            # Whole-route rows are ephemeral inputs to the route-moment
            # reducer.  Keep them on the accelerator; the normal callback path
            # converts only its explicitly requested rows, while the pilot
            # sufficient-statistic path transfers route-level summaries only.
            context_row_transport="device",
            context_factor_mode="native",
            use_conserving_activation_attribution=bool(
                self.conserving_activation_bindings
            ),
            gate_disclosure_request_factory=(
                self._gate_disclosure_request_for_batch
                if self.conserving_activation_bindings
                else None
            ),
        )
        self._device_summary_consumer: Callable[..., None] | None = None
        self._delegate = StreamingResponseGeometryProvider(
            context_ids=self.context_ids,
            channel_ids=tuple(query.query_id for query in self.queries),
            route_ids=self.route_ids,
            batch_callback=self._response_batches,
            response_spec_hash=str(response_spec_hash),
            metric_id=str(response_metric_id),
            calibration_state_id="uninitialized-calibration",
            batch_size=int(batch_size),
            artifact_root=artifact_root,
            projection_residual_bound=float(projection_residual_bound),
            sketch_residual_spectral_bound=float(sketch_residual_spectral_bound),
            omitted_route_residual_bound=float(omitted_route_residual_bound),
            local_covariance_sketch_rank=int(local_covariance_sketch_rank),
            local_covariance_sketch_seed=int(local_covariance_sketch_seed),
            realized_cost=RealizedCost(forward_passes=1, backward_passes=1),
        )

    def residency_breakdown(
        self, *, additional_roots: Mapping[str, Any] | None = None
    ) -> Mapping[str, object]:
        """Bytes and counts held by the CENSUS's growing state, per look.

        MEASURED CONTEXT: an arm died OUT_OF_MEMORY at 337.6 GiB **inside the
        census** with zero decisions made, host RSS climbing 2.96 GiB/min against
        66.2 GiB `cuda_reserved`. Attribution could not be settled by reading:
        the first search found four unbounded stores in the BCE oracle provider,
        which serves the WALK -- the census is served here. That miss is the
        reason this reports measurements rather than an argument.

        `_sequential_device_states` IS evicted, but only for **stale**
        mask/calibration keys; the ACTIVE entry accumulates for the whole census.
        Its `tripartite` and `trust` accumulators are route-scale and fixed by
        construction, so the panel-scaled terms can only be the per-context
        `context_ids` and the per-request `request_ids` / `fc_receipts` lists.
        This weighs all of them so the dominant one is named, not inferred.

        Cost is O(entries + contexts) and it is intended to be called PER LOOK,
        ten to twenty times over an arm. Callers should gate it.
        """

        def _tensor_bytes(value: object) -> int:
            if value is None or torch is None:
                return 0
            try:
                return int(value.numel()) * int(value.element_size())
            except Exception:  # pragma: no cover - defensive on foreign types
                return 0

        states = []
        total_context_ids = 0
        total_request_ids = 0
        total_fc_receipts = 0
        for key, entry in self._sequential_device_states.items():
            tripartite = entry.get("tripartite")
            context_ids = tuple(getattr(tripartite, "context_ids", ()) or ())
            request_ids = entry.get("request_ids") or ()
            fc_receipts = entry.get("fc_receipts") or ()
            total_context_ids += len(context_ids)
            total_request_ids += len(request_ids)
            total_fc_receipts += len(fc_receipts)
            device_bytes = 0
            for attribute in ("_moments", "_values", "_buffer", "_state"):
                device_bytes += _tensor_bytes(getattr(tripartite, attribute, None))
            states.append(
                {
                    "stratum_id": key[1],
                    "mask_hash": key[2],
                    "context_ids": len(context_ids),
                    "context_id_bytes_estimated": sum(len(value) + 49 + 8 for value in context_ids),
                    "request_ids": len(request_ids),
                    "fc_receipts": len(fc_receipts),
                    "accumulator_device_bytes": device_bytes,
                }
            )
        from mechanistic_systems.measurement.memory_probe import tensor_storage_breakdown

        evaluator = getattr(self, "evaluator", None)
        return {
            "schema": "census_residency_breakdown_v1",
            "device_memory": tensor_storage_breakdown({
                "model_parameters_and_buffers": (
                    getattr(evaluator, "model", None), getattr(evaluator, "source_model", None)
                ),
                "census_state": self._sequential_device_states,
                **dict(additional_roots or {}),
                "candidate_execution_state": getattr(evaluator, "candidate_execution_backend", None),
                "panel_data_plane": getattr(
                    getattr(getattr(self, "_batch_provider", None), "_batch_factory", None),
                    "native_panel_data_plane", None
                ),
            }, device=getattr(evaluator, "device", "cpu")),
            "sequential_device_states": len(self._sequential_device_states),
            "total_context_ids": total_context_ids,
            "total_request_ids": total_request_ids,
            "total_fc_receipts": total_fc_receipts,
            "states": tuple(states),
        }

    @property
    def context_packet_size(self) -> int:
        """Maximum contexts retained with the whole physical route field."""

        return int(self._delegate.batch_size)

    def _masks_by_layer(self, mask: HardMaskView) -> Mapping[str, np.ndarray]:
        if mask.route_count != len(self.route_ids):
            raise ValueError("optimized scoring mask differs from the IRG route axis")
        values = mask.to_bool_array()
        result: dict[str, np.ndarray] = {}
        offset = 0
        for block in self._route_blocks:
            stop = offset + block.route_count
            result[str(block.layer_id)] = values[offset:stop].reshape(block.shape)
            offset = stop
        return result

    def _loss_pullback_identity(self, query: QuerySpec, target_index: int) -> str:
        if self.census_dense_calibration_state_hash is not None:
            key = (
                str(query.query_id), target_index,
                self.census_dense_calibration_state_hash, self.pruning_guiding_loss_id,
            )
            identity = self._census_pullback_identities.get(key)
            if identity is None:
                identity = census_loss_pullback_hash(
                    query_id=str(query.query_id), target_class_id=target_index,
                    fit=self._active_calibration, loss_id=self.pruning_guiding_loss_id,
                )
                self._census_pullback_identities[key] = identity
            return identity
        if self.pruning_guiding_loss_id != CALIBRATED_OVR_BCE:
            return census_loss_pullback_hash(
                query_id=str(query.query_id), target_class_id=target_index,
                fit=self._active_calibration, loss_id=self.pruning_guiding_loss_id,
            )
        return content_hash(
            {
                "schema": "frozen_affine_bce_loss_pullback_v1",
                "query_id": query.query_id,
                "target_class_id": target_index,
                "calibration_state_id": getattr(self._active_calibration, "calibration_state_id", ""),
            },
            prefix="irg-loss-pullback-",
        )

    def _require_dense_census_calibration(self) -> None:
        if self.census_dense_calibration_state_hash is None:
            return
        if self._active_request_id and self._census_guarded_request_id == self._active_request_id:
            return
        fit = self._active_calibration
        frozen_affine_parameters(fit)
        if self._active_mask is None or not bool(self._active_mask.to_bool_array().all()):
            raise RuntimeError("census loss must be measured on the dense network")
        if str(getattr(fit, "mask_hash", "")) != str(self._active_mask.mask_hash):
            raise RuntimeError("dense census calibration belongs to another mask")
        if dense_calibration_state_hash(fit) != self.census_dense_calibration_state_hash:
            raise RuntimeError("dense census calibration state differs from the installed objective")
        self._census_guarded_request_id = self._active_request_id

    def _decision_census_fixed_offsets(self, context_ids: Sequence[str]) -> Any:
        """Gather the substituted arm's fixed dense competitors by row identity."""
        resolver = getattr(self, "decision_fixed_offset_resolver", None)
        if resolver is None:
            from mechanistic_systems.measurement.run_products import pruning_functional
            if pruning_functional() == "substituted_softmax":
                raise ValueError("substituted census requires the resident dense table offset resolver")
            return None
        offsets = resolver(tuple(map(str, context_ids)))
        if offsets is None:
            raise ValueError("substituted census offsets are absent from the resident dense table")
        return offsets

    def _conserving_loss_path_identity(self, query: QuerySpec, target_index: int) -> str:
        key = (str(query.query_id), target_index, str(self.census_dense_calibration_state_hash),
               self.pruning_guiding_loss_id + ":activation_path")
        identity = self._census_pullback_identities.get(key)
        if identity is None:
            identity = census_loss_path_hash(
                query_id=str(query.query_id), target_class_id=target_index,
                fit=self._active_calibration, loss_id=self.pruning_guiding_loss_id,
            )
            self._census_pullback_identities[key] = identity
        return identity

    def _measurement_plan(
        self,
        *,
        context_block_id: str,
    ) -> MeasurementPlan:
        return MeasurementPlan(
            plan_id=f"irg-optimized-response:{self._active_request_id}",
            study_hash=self.study_hash,
            queries=self.queries,
            measurement_view=self.measurement_view,
            estimator_id="exact_route_gate",
            estimator_version="irg_optimized_nested_pullback_route_gate_v2",
            measurement_semantics=(
                "nested_local_destination_global_response_bce_loss_pullback"
                if self.local_destination_scales_by_layer is not None
                else "global_query_response_route_derivative"
            ),
            route_schema_hash=str(self.evaluator.route_schema_hash),
            context_block_ids=(str(context_block_id),),
            status="exploratory",
            notes={
                "physical_provider": "optimized_scoring_backend",
                "request_id": self._active_request_id,
            },
        )

    def _response_batches(
        self,
        selected: tuple[str, ...],
        route_indices: Sequence[int],
        batch_size: int,
    ) -> Iterable[ResponseJacobianBatch]:
        if self._active_mask is None or not self._active_request_id:
            raise RuntimeError("optimized response request is not active")
        # Guard before plan execution: iter_moment_tiles owns the first VJP.
        self._require_dense_census_calibration()
        if self.local_destination_scales_by_layer is not None and self.apply_affine_calibration_to_response:
            frozen_affine_parameters(self._active_calibration)
        if batch_size <= 0:
            raise ValueError("optimized response context packet must be positive")
        # The scoring backend's context-factor tiles are dense on
        # ``context × response × route``.  Yield bounded context packets here
        # instead of asking the scorer to retain the complete context
        # population at once.  This is the adapter boundary that makes the
        # downstream moment accumulator genuinely streaming.
        if len(selected) > batch_size:
            for start in range(0, len(selected), batch_size):
                yield from self._response_batches(
                    selected[start : start + batch_size],
                    route_indices,
                    batch_size,
                )
            return
        self._active_context_packet_count += 1
        context_block_id = _optimized_response_context_block_id(
            provider_plan_identity_hash=self._active_request_id,
            context_ids=selected,
        )
        self._batch_provider.activate(
            context_block_id=context_block_id,
            context_ids=selected,
        )
        from .phase_device_orchestration import install_scoring_gates
        install_scoring_gates(self, self._active_mask)
        plan = self._measurement_plan(context_block_id=context_block_id)
        plan.validate().require_valid()
        if self._device_chunk_state is not None:
            # Route-chunked streaming: capture stays per route block, but
            # every ``contexts × route`` float64 tensor this provider forms
            # is bounded to ``response_route_chunk_size`` routes and reduced
            # into the device accumulator chunk-by-chunk.  The legacy body
            # below (which concatenates whole-axis pieces) is bypassed
            # entirely for such packets.
            self._stream_route_chunked_packet(
                selected=selected,
                context_block_id=context_block_id,
                plan=plan,
            )
            return
        tiles = tuple(self.scoring_backend.iter_moment_tiles(plan))
        observation_ids: tuple[str, ...] = ()
        raw_responses: np.ndarray | None = None
        labels: np.ndarray | None = None
        if self.local_destination_scales_by_layer is not None:
            observation_ids, raw_responses, labels = self.scoring_backend.context_observations(
                context_block_id
            )
            if observation_ids != selected:
                raise RuntimeError("loss pullback observations changed the scoring context order")
        by_layer = {tile.route_block.layer_id: tile for tile in tiles}
        expected_layers = tuple(block.layer_id for block in self._route_blocks)
        if set(by_layer) != set(expected_layers):
            missing = sorted(set(expected_layers) - set(by_layer))
            raise RuntimeError(
                "optimized scoring cannot form whole-route context rows; "
                f"missing route blocks={missing!r}"
            )
        if torch is None:
            raise RuntimeError("optimized scoring response provider requires PyTorch")
        pieces: list[Any] = []
        local_pieces: list[Any] = []
        capture_regime = bool(self.local_regime_specs_by_layer)
        all_regime_names = tuple(
            dict.fromkeys((*REGIME_STATISTIC_NAMES, *SMOOTH_TRANSMISSION_STATISTIC_NAMES))
        )
        regime_pieces: dict[str, list[np.ndarray]] = {name: [] for name in all_regime_names}
        regime_geometry_by_route: list[str] = []
        regime_activation_by_route: list[str] = []
        fc_receipts: list[Mapping[str, object]] = []
        for block, binding in zip(self._route_blocks, self._mask_bindings, strict=True):
            tile = by_layer[block.layer_id]
            if tile.context_rows is None:
                raise RuntimeError(
                    "optimized scoring backend has no per-context reducer for "
                    f"route family {block.route_kind!r} at {block.layer_id!r}; "
                    "IRG refuses a partial route axis"
                )
            rows = torch.as_tensor(
                tile.context_rows,
                device=self.evaluator.device,
                dtype=torch.float64,
            )
            expected = (len(selected), len(self.queries), block.route_count)
            if rows.shape != expected:
                raise ValueError("optimized scoring context rows have an incompatible axis")
            pieces.append(rows)
            raw_regime = tile.local_regime_statistics
            if not capture_regime:
                pass
            elif raw_regime is None:
                for name in all_regime_names:
                    regime_pieces[name].append(np.zeros(block.route_count, dtype=np.float64))
                regime_geometry_by_route.extend("" for _ in range(block.route_count))
                regime_activation_by_route.extend("" for _ in range(block.route_count))
            else:
                statistics = raw_regime["statistics"]
                geometry_hash = str(raw_regime["geometry_hash"])
                activation_kind = str(raw_regime["activation_kind"])
                expected_names = (
                    REGIME_STATISTIC_NAMES
                    if activation_kind == "relu"
                    else SMOOTH_TRANSMISSION_STATISTIC_NAMES
                )
                for name in all_regime_names:
                    value = (
                        np.zeros(block.route_count, dtype=np.float64)
                        if name not in expected_names
                        else statistics[name]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64, copy=False)
                        .reshape(-1)
                    )
                    if value.shape != (block.route_count,):
                        raise ValueError("local regime statistic differs from its route block")
                    regime_pieces[name].append(value)
                regime_geometry_by_route.extend(geometry_hash for _ in range(block.route_count))
                regime_activation_by_route.extend(activation_kind for _ in range(block.route_count))
            if self.local_destination_scales_by_layer is not None:
                if tile.local_context_rows is None:
                    raise RuntimeError(
                        f"optimized scoring omitted fused local rows for {block.layer_id!r}"
                    )
                source_module = None
                if str(block.route_kind) == "linear_weight":
                    source_module = self.evaluator.source_model.get_submodule(
                        str(binding.module_path)
                    )
                if self._active_flip_margin_accumulator is not None:
                    if source_module is None:
                        self._active_flip_margin_accumulator.mark_not_captured(
                            layer_id=str(block.layer_id),
                            route_kind=str(block.route_kind),
                        )
                    else:
                        source_bias = source_module.bias
                        if source_bias is None:
                            source_bias = torch.zeros(
                                int(block.shape[0]),
                                device=source_module.weight.device,
                                dtype=source_module.weight.dtype,
                            )
                        self._active_flip_margin_accumulator.update_linear(
                            layer_id=str(block.layer_id),
                            local_context_rows=tile.local_context_rows,
                            bias=source_bias.detach(),
                            route_shape=tuple(block.shape),
                            flattening=str(block.flattening),
                            route_chunk_size=self.response_route_chunk_size,
                        )
                local = torch.as_tensor(
                    tile.local_context_rows,
                    device=self.evaluator.device,
                    dtype=torch.float64,
                ).clone()
                if local.shape != (len(selected), block.route_count):
                    raise ValueError("optimized local-attribution rows have an incompatible axis")
                if str(block.route_kind) == "linear_weight":
                    assert source_module is not None
                    weight_tensor = source_module.weight.detach().reshape(-1)
                    if tuple(weight_tensor.shape) != (block.route_count,):
                        raise ValueError("FC nonnegative-input receipt differs from the route axis")
                    if self._device_summary_consumer is None:
                        fc_receipts.append(
                            fc_nonnegative_input_receipt(
                                local_context_rows=(local.detach().cpu().numpy()),
                                weight=(
                                    weight_tensor.to(device="cpu")
                                    .numpy()
                                    .astype(np.float64, copy=False)
                                ),
                                context_ids=selected,
                                architecture_id=str(block.architecture_id),
                                layer_id=str(block.layer_id),
                                module_path=str(binding.module_path),
                            )
                        )
                    else:
                        fc_receipts.append(
                            _fc_nonnegative_input_device_receipt(
                                local_context_rows=local,
                                weight=weight_tensor,
                                context_ids=selected,
                                architecture_id=str(block.architecture_id),
                                layer_id=str(block.layer_id),
                                module_path=str(binding.module_path),
                            )
                        )
                destination_scale = self.local_destination_scales_by_layer[str(block.layer_id)]
                repeats = int(block.route_count // int(block.shape[0]))
                local *= torch.as_tensor(
                    np.repeat(destination_scale, repeats),
                    device=local.device,
                    dtype=local.dtype,
                )[None, :]
                local_pieces.append(local)
        self._active_fc_nonnegative_input_receipts = (
            *self._active_fc_nonnegative_input_receipts,
            *fc_receipts,
        )
        whole = torch.cat(pieces, dim=2)
        local_whole = None if not local_pieces else torch.cat(local_pieces, dim=1)
        if self.apply_affine_calibration_to_response:
            if self._active_calibration is None:
                raise RuntimeError("calibrated response geometry has no active affine fit")
            scale = float(getattr(self._active_calibration, "scale", float("nan")))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    "calibrated response geometry requires a finite positive affine scale"
                )
            # The affine offset has zero gate derivative.  Multiplying the
            # canonical raw target-logit gate derivative by the frozen
            # positive scale gives the exact derivative of the calibrated
            # target logit, without changing within-chart ordering or signs.
            whole *= scale
        loss_whole = None
        loss_pullback_hash = None
        if (
            local_whole is not None
            and raw_responses is not None
            and labels is not None
            and len(self.queries) == 1
            and self.apply_affine_calibration_to_response
        ):
            query = self.queries[0]
            target = query.evaluator.get(
                "class_id",
                query.evaluator.get("unit", query.evaluator.get("coordinate")),
            )
            if target is None:
                raise ValueError("BCE loss pullback requires a scalar target-logit query")
            target_index = int(target)
            loss_weight = frozen_pruning_loss_multiplier(
                raw_responses=raw_responses, labels=labels,
                target_class_id=target_index, fit=self._active_calibration,
                loss_id=self.pruning_guiding_loss_id, device=whole.device,
                fixed_offsets=self._decision_census_fixed_offsets(selected),
            )
            # ``whole`` is already d(calibrated_logit)/dg.  Multiplying by
            # p-y is the exact contextwise BCE pullback; no new backward pass.
            loss_whole = loss_weight[:, None] * whole[:, 0, :]
            loss_pullback_hash = self._loss_pullback_identity(query, target_index)
        active_routes_device = torch.as_tensor(
            self._active_mask.to_bool_array(),
            device=whole.device,
            dtype=whole.dtype,
        )
        whole *= active_routes_device[None, None, :]
        if local_whole is not None:
            local_whole *= active_routes_device[None, :]
        if loss_whole is not None:
            loss_whole *= active_routes_device[None, :]
        active_routes = self._active_mask.to_bool_array().astype(np.float64, copy=False)
        regime_whole = (
            {
                name: np.concatenate(parts, axis=0) * active_routes
                for name, parts in regime_pieces.items()
            }
            if capture_regime
            else {}
        )
        whole_ordinal_axis = (
            isinstance(route_indices, range)
            and route_indices.start == 0
            and route_indices.stop == len(self.route_ids)
            and route_indices.step == 1
        )
        selected_routes = whole if whole_ordinal_axis else None
        if self._device_summary_consumer is not None:
            if local_whole is None or loss_whole is None:
                raise RuntimeError(
                    "device tripartite summaries require fused local/global/loss rows"
                )
            if whole_ordinal_axis:
                device_rows = torch.cat(
                    (local_whole[:, None, :], whole, loss_whole[:, None, :]),
                    dim=1,
                )
            else:
                selected_index_device = torch.as_tensor(
                    np.asarray(route_indices, dtype=np.int64),
                    device=whole.device,
                    dtype=torch.long,
                )
                selected_routes = whole.index_select(2, selected_index_device)
                device_rows = torch.cat(
                    (
                        local_whole.index_select(1, selected_index_device)[:, None, :],
                        selected_routes,
                        loss_whole.index_select(1, selected_index_device)[:, None, :],
                    ),
                    dim=1,
                )
            self._device_summary_consumer(
                context_ids=selected,
                rows=device_rows,
                destination_transport_hash=self.local_destination_transport_hash,
                loss_pullback_hash=loss_pullback_hash,
            )
            return
        selected_index = np.asarray(route_indices, dtype=np.int64)
        selected_index_device = torch.as_tensor(
            selected_index, device=whole.device, dtype=torch.long
        )
        if selected_routes is None:
            selected_routes = whole.index_select(2, selected_index_device)
        selected_geometry = (
            tuple(regime_geometry_by_route[int(index)] for index in selected_index)
            if capture_regime
            else tuple("" for _ in selected_index)
        )
        selected_activation = (
            tuple(regime_activation_by_route[int(index)] for index in selected_index)
            if capture_regime
            else tuple("" for _ in selected_index)
        )
        selected_observation = (
            regime_whole["observation_mass"][selected_index]
            if capture_regime
            else np.zeros(len(selected_index), dtype=np.float64)
        )
        geometry_blocks: list[tuple[int, int, str]] = []
        block_start: int | None = None
        block_identity = ""
        for index, (identity, mass) in enumerate(
            zip(selected_geometry, selected_observation, strict=True)
        ):
            covered_identity = identity if mass > 0.0 else ""
            if covered_identity != block_identity:
                if block_start is not None:
                    geometry_blocks.append((block_start, index, block_identity))
                block_start = index if covered_identity else None
                block_identity = covered_identity
        if block_start is not None:
            geometry_blocks.append((block_start, len(selected_geometry), block_identity))
        covered_activation_kinds = {
            activation
            for activation, mass in zip(selected_activation, selected_observation, strict=True)
            if mass > 0.0
        }
        if len(covered_activation_kinds) > 1:
            raise ValueError(
                "one streamed response batch cannot conflate different "
                "destination activation transmission laws"
            )
        covered_activation = (
            None if not covered_activation_kinds else next(iter(covered_activation_kinds))
        )
        summary_names = (
            REGIME_STATISTIC_NAMES
            if covered_activation == "relu"
            else SMOOTH_TRANSMISSION_STATISTIC_NAMES
        )
        summary_type = (
            LocalRegimeRouteSummary
            if covered_activation == "relu"
            else SmoothTransmissionRouteSummary
        )
        local_regime_summary = (
            None
            if not geometry_blocks or covered_activation is None
            else summary_type(
                route_ids=tuple(self.route_ids[int(index)] for index in selected_index),
                source_context_count=len(selected),
                context_axis_hash=content_hash(selected, prefix="local-regime-context-axis-"),
                measured_mask_hash=self._active_mask.mask_hash,
                source_measurement_hash=content_hash(
                    {
                        "request_id": self._active_request_id,
                        "context_ids": selected,
                        "mask_hash": self._active_mask.mask_hash,
                        "route_ids": tuple(self.route_ids[int(index)] for index in selected_index),
                        "geometry_blocks": tuple(geometry_blocks),
                        "statistics": {
                            name: values[selected_index] for name, values in regime_whole.items()
                        },
                    },
                    prefix="irg-local-regime-measurement-",
                ),
                geometry_blocks=tuple(geometry_blocks),
                activation_kind=covered_activation,
                **{
                    name: values[selected_index]
                    for name, values in regime_whole.items()
                    if name in summary_names
                },
            )
        )
        # This is the only branch in this provider that materializes full
        # context-by-route response rows on the host. Device consumers return
        # above, and route-chunked device accumulation bypasses this body.
        self._active_host_context_route_rows_transfer_count += 1
        yield ResponseJacobianBatch(
            context_ids=selected,
            route_rows=selected_routes.detach().cpu().numpy(),
            local_route_rows=(
                None
                if local_whole is None
                else local_whole.index_select(1, selected_index_device).detach().cpu().numpy()
            ),
            local_destination_transport_hash=(self.local_destination_transport_hash),
            loss_route_rows=(
                None
                if loss_whole is None
                else loss_whole.index_select(1, selected_index_device).detach().cpu().numpy()
            ),
            loss_pullback_hash=loss_pullback_hash,
            local_regime_summary=local_regime_summary,
        )

    def _route_chunked_update_for(
        self,
        accumulator: Any,
        *,
        identity_error: str,
    ) -> _RouteChunkedTripartiteUpdate | None:
        """Chunk-capable state updater for one device accumulator, or None.

        Route-chunked streaming reproduces
        :class:`DeviceTripartiteMomentAccumulator.update` state arithmetic
        directly, so it engages only when the target is exactly that class
        (unshadowed), lives on the evaluator device, and the packet carries
        the fused local/global/loss heads with unit weights.  Everything
        else — including duck-typed fan-outs whose Frequent-Directions trust
        sketch requires whole-axis observations — keeps the original
        full-axis consumer path unchanged.
        """

        if (
            torch is None
            or type(accumulator) is not DeviceTripartiteMomentAccumulator
            or "update" in vars(accumulator)
            or int(self.response_route_chunk_size) >= len(self.route_ids)
            or self.local_destination_scales_by_layer is None
            or not self.apply_affine_calibration_to_response
            or len(self.queries) != 1
            or (
                accumulator.route_ids is not self.route_ids
                and (
                    len(accumulator.route_ids) != len(self.route_ids)
                    or ordered_string_axis_hash(
                        accumulator.route_ids,
                        name="device accumulator route axis",
                    )
                    != ordered_string_axis_hash(
                        self.route_ids,
                        name="optimized IRG scoring route axis",
                    )
                )
            )
            or len(accumulator.head_ids) != 2 + len(self.queries)
            or accumulator.device != torch.device(self.evaluator.device)
        ):
            return None
        return _RouteChunkedTripartiteUpdate(
            accumulator=accumulator,
            route_total=len(self.route_ids),
            chunk_size=int(self.response_route_chunk_size),
            identity_error=identity_error,
        )

    def _loss_only_chunk_consumer_for(
        self,
        accumulator: Any,
    ) -> _RouteChunkedLossOnlyConsumer | None:
        """Return the nominal loss-only consumer, refusing invalid opt-ins.

        Unlike tripartite moment streaming, this path remains useful even
        when the execution chunk spans the complete axis: it prevents the
        ordinary ``[packet, local/global/loss heads, route]`` concatenation
        and retains only one ``[packet, route]`` loss staging tensor.
        """

        if not isinstance(accumulator, _RouteChunkedLossOnlyConsumer):
            return None
        if torch is None:
            raise RuntimeError("loss-only route streaming requires PyTorch")
        coordinate = str(
            getattr(
                accumulator,
                "positive_removal_stream_coordinate",
                _BCE_POSITIVE_REMOVAL_COORDINATE,
            )
        )
        if coordinate not in {
            _BCE_POSITIVE_REMOVAL_COORDINATE,
            _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE,
        }:
            raise ValueError("loss-only consumer declared an unsupported coordinate")
        if self.census_dense_calibration_state_hash is not None and coordinate != _BCE_POSITIVE_REMOVAL_COORDINATE:
            raise ValueError("census corrected order requires the dense pruning-loss covector")
        if len(self.queries) != 1:
            raise ValueError(
                "loss-only route streaming requires one scalar target-logit response"
            )
        if (
            coordinate == _BCE_POSITIVE_REMOVAL_COORDINATE
            and (
                (self.local_destination_scales_by_layer is None and not self.conserving_activation_bindings)
                or not self.apply_affine_calibration_to_response
            )
        ):
            raise ValueError(
                "compatibility BCE loss-only streaming requires calibrated response wiring"
            )
        consumer_axis = getattr(accumulator, "route_ids", ())
        if consumer_axis is not self.route_ids and (
            len(consumer_axis) != len(self.route_ids)
            or ordered_string_axis_hash(
                consumer_axis,
                name="loss-only consumer route axis",
            )
            != ordered_string_axis_hash(
                self.route_ids,
                name="optimized IRG scoring route axis",
            )
        ):
            raise ValueError("loss-only consumer and optimized scorer route axes differ")
        if torch.device(getattr(accumulator, "device", "cpu")) != torch.device(
            self.evaluator.device
        ):
            raise ValueError("loss-only consumer and optimized scorer devices differ")
        accumulator.bind_route_chunk_size(int(self.response_route_chunk_size))
        return accumulator

    def _gate_disclosure_request_for_batch(
        self,
        batch: Any,
        query_batch: Any,
    ) -> V1GateDisclosureRequest | None:
        """Bind the same postactivation VJP to an explicit disclosure covector.

        The callback is active only for the explicit sidecar passed to
        :meth:`accumulate_device_tripartite`.  Canonical raw-target-logit
        disclosure uses an exact unit cotangent; the historical fixed-affine
        BCE cotangent remains compatibility-only.  The returned consumer is
        transactional per capture microbatch; callback failure propagates
        through scoring and aborts the acquisition.
        """

        accumulator = self._active_gate_disclosure_accumulator
        if accumulator is None:
            return None
        if torch is None:
            raise RuntimeError("gate-removal disclosure requires PyTorch")
        if (
            len(self.queries) != 1
            or tuple(query_batch.query_ids)
            != (str(self.queries[0].query_id),)
        ):
            raise ValueError(
                "target-scale gate disclosure requires one target-logit query"
            )
        query = self.queries[0]
        raw_query = query_batch.values[0]
        stream_coordinate = str(
            getattr(
                accumulator,
                "positive_removal_stream_coordinate",
                _BCE_POSITIVE_REMOVAL_COORDINATE,
            )
        )
        if stream_coordinate == _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE:
            target_index = _exact_raw_target_logit_class_id(query)
            multipliers = torch.ones(
                (1, int(raw_query.numel())),
                dtype=batch.route_blocks[0].output.dtype,
                device=raw_query.device,
            )
            target_scale_identity = (
                _raw_target_logit_positive_removal_stream_identity(
                    query_id=str(query.query_id),
                    target_class_id=target_index,
                )
            )
        elif stream_coordinate == _BCE_POSITIVE_REMOVAL_COORDINATE:
            target = query.evaluator.get(
                "class_id",
                query.evaluator.get(
                    "unit",
                    query.evaluator.get("coordinate"),
                ),
            )
            if target is None:
                raise ValueError(
                    "target-scale gate disclosure requires a target-logit query"
                )
            target_index = int(target)
            if self._active_calibration is None or batch.labels is None:
                raise ValueError(
                    "compatibility BCE gate disclosure requires calibration and labels"
                )
            scale, _, _ = frozen_affine_parameters(self._active_calibration)
            multipliers = frozen_pruning_loss_multiplier(
                raw_responses=batch.response_matrix, labels=batch.labels,
                target_class_id=target_index, fit=self._active_calibration,
                loss_id=self.pruning_guiding_loss_id, device=raw_query.device,
                fixed_offsets=self._decision_census_fixed_offsets(batch.context_ids),
            ).mul(scale).to(dtype=batch.route_blocks[0].output.dtype)[None, :]
            target_scale_identity = self._loss_pullback_identity(query, target_index)
        else:
            raise ValueError("gate disclosure declared an unsupported covector")
        expected_identity = str(
            getattr(accumulator, "loss_pullback_hash", "")
        )
        if expected_identity != target_scale_identity:
            raise ValueError(
                "gate-disclosure and score streams use different covectors"
            )
        return V1GateDisclosureRequest(
            context_ids=tuple(batch.context_ids),
            context_query_multipliers=multipliers,
            target_scale_identity=target_scale_identity,
            consumer=accumulator,
            # The response stream's multi-million-route chunk is suitable
            # only after spatial contraction.  Singleton activation changes
            # retain a spatial axis, so bound this diagnostic more tightly.
            route_chunk_size=min(int(self.response_route_chunk_size), 4096),
            # Linear singleton routes have spatial extent one.  The flattened
            # global-route kernel can therefore amortize launch overhead with
            # a larger bounded slab without inheriting convolutional spatial
            # residency.
            linear_route_chunk_size=min(
                int(self.response_route_chunk_size),
                65_536,
            ),
        )

    def _stream_route_chunked_packet(
        self,
        *,
        selected: tuple[str, ...],
        context_block_id: str,
        plan: MeasurementPlan,
    ) -> None:
        """Stream one context packet through route-chunked accumulation.

        Tripartite rows contain dense endpoint loss gradients. The census's
        conserving sink integrates that loss along the registered unitwise
        activation paths before route contraction. Other loss-only consumers
        retain their declared covector. Per-route context reductions happen
        inside the selected sink.
        """

        state = self._device_chunk_state
        if torch is None or state is None:
            raise RuntimeError("optimized scoring response provider requires PyTorch")
        loss_only_stream = isinstance(state, _RouteChunkedLossOnlyConsumer)
        if (
            loss_only_stream
            and self.conserving_activation_bindings
            and self._active_flip_margin_accumulator is None
        ):
            if self._active_mask is None or self._active_calibration is None:
                raise RuntimeError(
                    "exact loss-only streaming requires an active mask and calibration"
                )
            stream_coordinate = str(
                getattr(
                    state,
                    "positive_removal_stream_coordinate",
                    _BCE_POSITIVE_REMOVAL_COORDINATE,
                )
            )
            if stream_coordinate not in {
                _BCE_POSITIVE_REMOVAL_COORDINATE,
                _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE,
            }:
                raise ValueError(
                    "loss-only consumer declared an unsupported positive-removal "
                    "coordinate"
                )
            query = self.queries[0]
            loss_pullback_factory: Callable[[Any, Any], Any] | None
            loss_path_factory: Callable[[Any, Any], Any] | None = None
            loss_pullback_hash: str | None
            if stream_coordinate == _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE:
                target_index = _exact_raw_target_logit_class_id(query)
                # Canonical v1 proposal geometry and exact salience use the
                # raw target-logit covector.  Calibration affects only the
                # analytical trust-radius transport outside this scorer.
                scale = 1.0
                loss_pullback_factory = None
                loss_pullback_hash = None
                stream_identity_hash = (
                    _raw_target_logit_positive_removal_stream_identity(
                        query_id=str(query.query_id),
                        target_class_id=target_index,
                    )
                )
            else:
                target = query.evaluator.get(
                    "class_id",
                    query.evaluator.get(
                        "unit",
                        query.evaluator.get("coordinate"),
                    ),
                )
                if target is None:
                    raise ValueError(
                        "positive-removal streaming requires a scalar target-logit query"
                    )
                target_index = int(target)
                scale, _, _ = frozen_affine_parameters(self._active_calibration)
                def validate_labeled_query(batch: Any, query_batch: Any) -> None:
                    if (
                        batch.labels is None
                        or tuple(query_batch.query_ids) != (str(query.query_id),)
                    ):
                        raise ValueError(
                            "BCE positive-removal streaming requires one labeled query"
                        )
                if self.census_dense_calibration_state_hash is not None:
                    loss_pullback_factory = None
                    loss_pullback_hash = self._conserving_loss_path_identity(query, target_index)

                    def loss_path_factory(batch: Any, query_batch: Any) -> Any:
                        validate_labeled_query(batch, query_batch)
                        return frozen_pruning_loss_path(
                            raw_responses=batch.response_matrix, labels=batch.labels,
                            target_class_id=target_index, fit=self._active_calibration,
                            loss_id=self.pruning_guiding_loss_id, device=batch.response_matrix.device,
                            fixed_offsets=self._decision_census_fixed_offsets(batch.context_ids),
                        )
                else:
                    loss_pullback_hash = self._loss_pullback_identity(query, target_index)

                    def loss_pullback_factory(batch: Any, query_batch: Any) -> Any:
                        validate_labeled_query(batch, query_batch)
                        return frozen_pruning_loss_multiplier(
                            raw_responses=batch.response_matrix, labels=batch.labels,
                            target_class_id=target_index, fit=self._active_calibration,
                            loss_id=self.pruning_guiding_loss_id, device=batch.response_matrix.device,
                            fixed_offsets=self._decision_census_fixed_offsets(batch.context_ids),
                        )
                stream_identity_hash = loss_pullback_hash

            declared_stream_identity = str(
                getattr(state, "loss_pullback_hash", "")
            )
            if (
                declared_stream_identity
                and declared_stream_identity != stream_identity_hash
            ):
                raise ValueError(
                    "loss-only consumer names another positive-removal stream identity"
                )

            if isinstance(state, _RouteChunkedDualChartConsumer):
                evidence_consumer = state.evidence_consumer
                literal_jacobian_consumer = state.jacobian_consumer
                literal_jacobian_stream_identity_hash = (
                    state.literal_jacobian_stream_identity_hash
                )
                expected_literal_identity = (
                    _raw_target_logit_literal_jacobian_stream_identity(
                        query_id=str(query.query_id),
                        target_class_id=target_index,
                    )
                )
                if (
                    stream_coordinate
                    != _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
                    or literal_jacobian_stream_identity_hash
                    != expected_literal_identity
                ):
                    raise ValueError(
                        "dual-chart literal-Jacobian stream identity differs"
                    )
            else:
                evidence_consumer = state
                literal_jacobian_consumer = None
                literal_jacobian_stream_identity_hash = None

            stream_receipt = (
                self.scoring_backend.stream_conserving_positive_removal_rows(
                    plan,
                    context_block_id=context_block_id,
                    expected_context_ids=selected,
                    route_ids=self.route_ids,
                    route_gate=self._active_mask.to_bool_array(),
                    loss_pullback_factory=loss_pullback_factory,
                    loss_path_factory=loss_path_factory,
                    affine_scale=scale,
                    destination_transport_hash=(
                        self.local_destination_transport_hash
                    ),
                    loss_pullback_hash=loss_pullback_hash,
                    route_chunk_size=int(state.chunk_size),
                    consumer=evidence_consumer,
                    stream_coordinate=stream_coordinate,
                    stream_identity_hash=stream_identity_hash,
                    literal_jacobian_consumer=literal_jacobian_consumer,
                    literal_jacobian_stream_identity_hash=(
                        literal_jacobian_stream_identity_hash
                    ),
                    gate_disclosure_request_factory=(
                        self._gate_disclosure_request_for_batch
                        if self._active_gate_disclosure_accumulator is not None
                        else None
                    ),
                )
            )
            self._active_exact_loss_stream_receipts.append(stream_receipt)
            return
        by_layer = {
            tile.route_block.layer_id: tile for tile in self.scoring_backend.iter_moment_tiles(plan)
        }
        expected_layers = tuple(block.layer_id for block in self._route_blocks)
        if set(by_layer) != set(expected_layers):
            missing = sorted(set(expected_layers) - set(by_layer))
            raise RuntimeError(
                "optimized scoring cannot form whole-route context rows; "
                f"missing route blocks={missing!r}"
            )
        observation_ids, raw_responses, labels = self.scoring_backend.context_observations(
            context_block_id
        )
        if observation_ids != selected:
            raise RuntimeError("loss pullback observations changed the scoring context order")
        query = self.queries[0]
        target = query.evaluator.get(
            "class_id",
            query.evaluator.get("unit", query.evaluator.get("coordinate")),
        )
        if target is None:
            raise ValueError("response pullback requires a scalar target-logit query")
        target_index = int(target)
        device = self.evaluator.device
        stream_coordinate = (
            str(
                getattr(
                    state,
                    "positive_removal_stream_coordinate",
                    _BCE_POSITIVE_REMOVAL_COORDINATE,
                )
            )
            if loss_only_stream
            else _BCE_POSITIVE_REMOVAL_COORDINATE
        )
        if stream_coordinate == _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE:
            target_index = _exact_raw_target_logit_class_id(query)
            scale = 1.0
            loss_weight = torch.ones(
                len(selected),
                device=device,
                dtype=torch.float64,
            )
            loss_pullback_hash = (
                _raw_target_logit_positive_removal_stream_identity(
                    query_id=str(query.query_id),
                    target_class_id=target_index,
                )
            )
        elif stream_coordinate == _BCE_POSITIVE_REMOVAL_COORDINATE:
            if self._active_calibration is None:
                raise RuntimeError(
                    "calibrated response geometry has no active affine fit"
                )
            scale, _, _ = frozen_affine_parameters(self._active_calibration)
            loss_weight = frozen_pruning_loss_multiplier(
                raw_responses=raw_responses, labels=labels,
                target_class_id=target_index, fit=self._active_calibration,
                loss_id=self.pruning_guiding_loss_id, device=device,
                fixed_offsets=self._decision_census_fixed_offsets(selected),
            )
            loss_pullback_hash = self._loss_pullback_identity(query, target_index)
        else:
            raise ValueError("unsupported route-stream covector coordinate")
        if loss_only_stream:
            state.begin_packet(
                selected,
                route_ids=self.route_ids,
                destination_transport_hash=(self.local_destination_transport_hash),
                loss_pullback_hash=loss_pullback_hash,
            )
        else:
            state.begin_packet(
                selected,
                destination_transport_hash=(self.local_destination_transport_hash),
                loss_pullback_hash=loss_pullback_hash,
            )
        active_routes_device = torch.as_tensor(
            self._active_mask.to_bool_array(),
            device=device,
            dtype=torch.float64,
        )
        from mechanistic_systems.measurement.fused_route_moments import (
            ScoreEpilogueState,
            fused_kernel_available,
        )

        # The loss-only consumer performs a narrow elementwise head-formation
        # fast path.  It is not the tripartite fused moment kernel and must not
        # inherit either that capability claim or its score epilogue.
        fused_requested = not loss_only_stream
        if not loss_only_stream:
            state.fused_requested = fused_requested
        if (
            fused_requested
            and torch.device(device).type == "cuda"
            and not fused_kernel_available()
        ):
            raise RuntimeError(
                "IRG_FUSED_ROUTE_MOMENTS was requested but its CUDA/Triton "
                "kernel is unavailable"
            )
        score_epilogue = (
            ScoreEpilogueState(contexts=len(selected), device=device)
            if not loss_only_stream
            and fused_requested
            else None
        )
        fc_receipts: list[Mapping[str, object]] = []
        block_offset = 0
        for block, binding in zip(self._route_blocks, self._mask_bindings, strict=True):
            tile = by_layer.pop(block.layer_id)
            if tile.context_rows is None:
                raise RuntimeError(
                    "optimized scoring backend has no per-context reducer for "
                    f"route family {block.route_kind!r} at {block.layer_id!r}; "
                    "IRG refuses a partial route axis"
                )
            expected = (len(selected), len(self.queries), block.route_count)
            if tuple(tile.context_rows.shape) != expected:
                raise ValueError("optimized scoring context rows have an incompatible axis")
            if tile.local_context_rows is None:
                raise RuntimeError(
                    f"optimized scoring omitted fused local rows for {block.layer_id!r}"
                )
            if tuple(tile.local_context_rows.shape) != (
                len(selected),
                block.route_count,
            ):
                raise ValueError("optimized local-attribution rows have an incompatible axis")
            receipt_builder: _ChunkedFcInputReceiptBuilder | None = None
            source_module = None
            if str(block.route_kind) == "linear_weight":
                source_module = self.evaluator.source_model.get_submodule(str(binding.module_path))
                weight_tensor = source_module.weight.detach().reshape(-1)
                if tuple(weight_tensor.shape) != (block.route_count,):
                    raise ValueError("FC nonnegative-input receipt differs from the route axis")
                receipt_builder = _ChunkedFcInputReceiptBuilder(
                    weight=weight_tensor,
                    context_ids=selected,
                    architecture_id=str(block.architecture_id),
                    layer_id=str(block.layer_id),
                    module_path=str(binding.module_path),
                )
            if self._active_flip_margin_accumulator is not None:
                if source_module is None:
                    self._active_flip_margin_accumulator.mark_not_captured(
                        layer_id=str(block.layer_id),
                        route_kind=str(block.route_kind),
                    )
                else:
                    source_bias = source_module.bias
                    if source_bias is None:
                        source_bias = torch.zeros(
                            int(block.shape[0]),
                            device=source_module.weight.device,
                            dtype=source_module.weight.dtype,
                        )
                    self._active_flip_margin_accumulator.update_linear(
                        layer_id=str(block.layer_id),
                        local_context_rows=tile.local_context_rows,
                        bias=source_bias.detach(),
                        route_shape=tuple(block.shape),
                        flattening=str(block.flattening),
                        route_chunk_size=self.response_route_chunk_size,
                    )
            destination_scale = self.local_destination_scales_by_layer[str(block.layer_id)]
            repeats = int(block.route_count // int(block.shape[0]))
            destination_scale_routes = torch.as_tensor(
                np.repeat(destination_scale, repeats),
                device=device,
                dtype=torch.float64,
            )
            for start, stop in _block_chunk_bounds(block.route_count, state.chunk_size):
                local_rows = torch.as_tensor(
                    tile.local_context_rows[:, start:stop],
                    device=device,
                    dtype=torch.float64,
                )
                if receipt_builder is not None:
                    receipt_builder.consume(local_rows, start, stop)
                gate = active_routes_device[block_offset + start : block_offset + stop]
                if loss_only_stream:
                    state.consume_chunk_fused(
                        block_offset + start,
                        block_offset + stop,
                        global_raw=torch.as_tensor(
                            tile.context_rows[:, 0, start:stop],
                            device=device,
                            dtype=torch.float64,
                        ).contiguous(),
                        gate=gate.contiguous(),
                        pullback=loss_weight.contiguous(),
                        affine_scale=float(scale),
                        score_epilogue=None,
                    )
                    continue
                state.consume_chunk_fused(
                    block_offset + start,
                    block_offset + stop,
                    local_raw=local_rows.contiguous(),
                    global_raw=torch.as_tensor(
                        tile.context_rows[:, 0, start:stop],
                        device=device,
                        dtype=torch.float64,
                    ).contiguous(),
                    dest_scale=destination_scale_routes[start:stop].contiguous(),
                    gate=gate.contiguous(),
                    pullback=loss_weight.contiguous(),
                    affine_scale=float(scale),
                    score_epilogue=score_epilogue,
                )
            if receipt_builder is not None:
                fc_receipts.append(receipt_builder.finalize())
            block_offset += block.route_count
        if (
            not loss_only_stream
            and state.fused_requested
            and state.fused_chunk_update_count <= 0
        ):
            raise RuntimeError(
                "fused route moments were requested but no fused update engaged"
            )
        if loss_only_stream and score_epilogue is not None:
            raise RuntimeError("loss-only route streaming cannot finalize a score epilogue")
        if not loss_only_stream and score_epilogue is not None:
            state.fused_score_epilogue_receipts.append(
                score_epilogue.finalize_packet()
            )
        state.end_packet()
        self._active_fc_nonnegative_input_receipts = (
            *self._active_fc_nonnegative_input_receipts,
            *fc_receipts,
        )

    def execute(self, request: MeasurementRequest) -> MeasurementOutcome:
        if request.kind not in self.supported_kinds:
            raise ValueError("optimized scoring provider received an unsupported kind")
        mask_hash = str(request.payload.get("mask_hash", ""))
        if not mask_hash:
            raise ValueError("optimized response request requires mask_hash")
        # Explicit nested-prefix requests already carry these IDs.  The
        # compatibility selector may carry only a requested count, so derive
        # its actual deterministic selection before receipting it.
        selected_contexts = tuple(self._delegate._selected_contexts(request))
        prior_count = int(request.payload.get("response_acquisition_prior_count", 0))
        panel_receipt = {
            "schema": "optimized_response_panel_timing_receipt_v1",
            "request_id": request.request_id,
            "mask_hash": mask_hash,
            "stratum_id": request.payload.get("response_acquisition_stratum_id"),
            "prior_context_count": prior_count,
            "requested_cumulative_context_count": len(selected_contexts),
            "requested_new_context_count": max(0, len(selected_contexts) - prior_count),
            "response_acquisition_reset_count": int(
                request.payload.get("response_acquisition_reset_count", 0)
            ),
            "response_acquisition_last_reset_reason": request.payload.get(
                "response_acquisition_last_reset_reason"
            ),
            "response_acquisition_chart_republication_count": int(
                request.payload.get(
                    "response_acquisition_chart_republication_count",
                    0,
                )
            ),
            "context_prefix_hash": content_hash(
                selected_contexts,
                prefix="irg-response-panel-prefix-",
            ),
        }
        if self.decision_fixed_offset_resolver is not None:
            panel_receipt["pruning_functional"] = "substituted_softmax"
        if self.progress_callback is not None:
            self.progress_callback(
                "response_panel_started",
                **panel_receipt,
            )
        panel_started = time.perf_counter()
        with self._lock:
            mask = self.mask_resolver(mask_hash)
            calibration = self.calibration_resolver(mask)
            self._active_mask = mask
            self._active_calibration = calibration
            self._active_request_id = request.request_id
            self._census_guarded_request_id = ""
            self._active_fc_nonnegative_input_receipts = ()
            self._active_context_packet_count = 0
            self._active_host_context_route_rows_transfer_count = 0
            try:
                self._require_dense_census_calibration()
                self._delegate.calibration_state_id = str(calibration.calibration_state_id)
                if self._supports_device_sequential_response(request):
                    outcome = self._execute_device_sequential_response(
                        request,
                        selected_contexts=selected_contexts,
                        prior_count=prior_count,
                        mask=mask,
                        calibration=calibration,
                    )
                else:
                    outcome = self._delegate.execute(request)
                host_context_route_rows_transfer_count = (
                    self._active_host_context_route_rows_transfer_count
                )
            finally:
                self._active_mask = None
                self._active_calibration = None
                self._active_request_id = ""
        details = self.scoring_backend.runtime_details()
        completed_receipt = {
            **panel_receipt,
            "delivered_cumulative_context_count": len(outcome.context_ids),
            "delivered_new_context_count": int(
                outcome.metadata.get(
                    "physical_addition_count",
                    max(0, len(outcome.context_ids) - prior_count),
                )
            ),
            "sequential_sufficient_statistic_update": bool(
                outcome.metadata.get("device_sequential_sufficient_statistics", False)
            ),
            "sealed_prefix_suffix_only_update": bool(
                outcome.metadata.get("sealed_prefix_suffix_only_update", False)
            ),
            "prefix_replay_reason": outcome.metadata.get("prefix_replay_reason"),
            "elapsed_seconds": time.perf_counter() - panel_started,
            "scoring_context_packet_count": (self._active_context_packet_count),
            "scoring_context_packet_size": self._delegate.batch_size,
            "outcome_hash": outcome.integrity_hash,
        }
        result = MeasurementOutcome.create(
            request_id=outcome.request_id,
            state_hash=outcome.state_hash,
            context_ids=outcome.context_ids,
            candidate_ids=outcome.candidate_ids,
            values=outcome.values,
            calibration_state_id=outcome.calibration_state_id,
            realized_cost=outcome.realized_cost,
            cache_artifacts=outcome.cache_artifacts,
            metadata={
                **dict(outcome.metadata),
                "provider": "OptimizedScoringResponseProvider",
                "native_gpu_capable": True,
                "whole_physical_route_axis": True,
                "direct_torch_autograd_calls": False,
                "shared_canonical_gate_graph": (
                    self.model_adapter.gate_graph is self.evaluator.gate_graph
                ),
                "one_forward_shared_across_target_units": True,
                "target_unit_vjp_backend": "is_grads_batched",
                "affine_calibration_applied_to_response": (
                    self.apply_affine_calibration_to_response
                ),
                "affine_calibration_derivative_rule": (
                    "positive_scale_times_raw_target_logit_gate_derivative"
                    if self.apply_affine_calibration_to_response
                    else "none"
                ),
                "fused_local_attribution": (self.local_destination_scales_by_layer is not None),
                "tripartite_covariance_representation": (
                    "shared_context_rademacher_factor_v1"
                    if self.local_destination_scales_by_layer is not None
                    else "none"
                ),
                "tripartite_covariance_residual_bound_available": False,
                "tripartite_covariance_claim_status": (
                    "guidance_only_randomized_energy_approximation"
                    if self.local_destination_scales_by_layer is not None
                    else "not_measured"
                ),
                "local_destination_transport_hash": (self.local_destination_transport_hash),
                "fc_nonnegative_input_receipts": (self._active_fc_nonnegative_input_receipts),
                "scoring_context_packet_count": (self._active_context_packet_count),
                "scoring_context_packet_size": self._delegate.batch_size,
                "bounded_context_route_materialization": True,
                "host_context_route_rows_transferred": bool(host_context_route_rows_transfer_count),
                "host_context_route_row_transfer_count": int(
                    host_context_route_rows_transfer_count
                ),
                "response_panel_timing_receipt": completed_receipt,
                "fc_local_weight_sign_equivalence_scope": (
                    "verified_nonnegative_inputs_only"
                    if self._active_fc_nonnegative_input_receipts
                    and all(
                        bool(receipt["verified"])
                        for receipt in self._active_fc_nonnegative_input_receipts
                    )
                    else "not_established"
                ),
                "scoring_adapter_id": self.scoring_backend.adapter_id,
                "conserving_activation_attribution_enabled": bool(
                    self.conserving_activation_bindings
                ),
                "route_reducer_backends": details["route_reducer_backends"],
                "route_reducer_implementation_sources": details[
                    "route_reducer_implementation_sources"
                ],
                "scoring_runtime_details_hash": content_hash(
                    details, prefix="irg-optimized-scoring-runtime-"
                ),
            },
        )
        if self.progress_callback is not None:
            self.progress_callback(
                "response_panel_completed",
                **completed_receipt,
                result_integrity_hash=result.integrity_hash,
            )
        return result

    def _supports_device_sequential_response(self, request: MeasurementRequest) -> bool:
        """Whether one v1 request can remain entirely summary-streamed.

        v2+ atlas split folds and nuisance quotienting still require their
        richer provider path.  The ordinary v1 pilot needs only the fused
        U→G→D sufficient statistics and globally coupled FD trust factor, all
        of which have exact accelerator-resident accumulators.
        """

        payload = request.payload
        return bool(
            request.kind == "response_moments"
            and request.version == "v1"
            and payload.get("response_acquisition_design_hash")
            and payload.get("response_acquisition_stratum_id")
            and payload.get("whole_physical_route_axis") is True
            and "route_indices" not in payload
            and not payload.get("intrinsic_nuisance_quotient", False)
            and payload.get("functional_atlas_spec") is None
            and not self.local_regime_specs_by_layer
            and self.local_destination_scales_by_layer is not None
            and self.apply_affine_calibration_to_response
        )

    @staticmethod
    def _trust_spec_from_request(
        request: MeasurementRequest,
    ) -> GlobalTrustGeometrySpec | None:
        raw = request.payload.get("global_trust_geometry_spec")
        if raw is None:
            return None
        if isinstance(raw, GlobalTrustGeometrySpec):
            return raw
        if isinstance(raw, Mapping):
            return GlobalTrustGeometrySpec(**dict(raw))
        raise ValueError("global trust geometry request has an invalid spec")

    def _execute_device_sequential_response(
        self,
        request: MeasurementRequest,
        *,
        selected_contexts: tuple[str, ...],
        prior_count: int,
        mask: HardMaskView,
        calibration: Any,
    ) -> MeasurementOutcome:
        """Update one sealed stratum from only its newly exposed suffix."""

        design_hash = str(request.payload["response_acquisition_design_hash"])
        stratum_id = str(request.payload["response_acquisition_stratum_id"])
        calibration_id = str(calibration.calibration_state_id)
        trust_spec = self._trust_spec_from_request(request)
        requested_rank = (
            0
            if trust_spec is None
            else int(
                request.payload.get(
                    "global_trust_requested_rank",
                    trust_spec.initial_rank,
                )
            )
        )
        if (
            prior_count < 0
            or prior_count > len(selected_contexts)
            or (
                prior_count == len(selected_contexts)
                and not request.payload.get("response_acquisition_rank_replay", False)
            )
            or tuple(request.payload.get("response_acquisition_cumulative_prefix", ()))
            != selected_contexts
        ):
            raise ValueError("sequential response request has an invalid cumulative prefix")
        key = (
            design_hash,
            stratum_id,
            mask.mask_hash,
            calibration_id,
            requested_rank,
        )
        # A committed mask/calibration transition retires the prior device
        # chart. Keep both strata for the active chart, but do not leak one
        # full route accumulator per historical state.
        stale = [
            item
            for item in self._sequential_device_states
            if item[2] != mask.mask_hash or item[3] != calibration_id
        ]
        for item in stale:
            self._sequential_device_states.pop(item, None)
        entry = self._sequential_device_states.get(key)
        replay_reason: str | None = None
        if entry is None:
            if prior_count:
                # Rank growth deliberately creates a new FD sketch and must
                # replay the already sealed prefix at the higher rank. This is
                # the only supported cumulative replay; ordinary Neyman growth
                # always consumes the new suffix only.
                replay_reason = "global_trust_rank_growth"
            loss_pullback_hash = content_hash(
                {
                    "schema": "frozen_affine_bce_loss_pullback_v1",
                    "query_id": self.queries[0].query_id,
                    "target_class_id": int(
                        self.queries[0].evaluator.get(
                            "class_id",
                            self.queries[0].evaluator.get(
                                "unit",
                                self.queries[0].evaluator.get("coordinate"),
                            ),
                        )
                    ),
                    "calibration_state_id": calibration_id,
                },
                prefix="irg-loss-pullback-",
            )
            tripartite = DeviceTripartiteMomentAccumulator(
                head_ids=(
                    "local_destination",
                    *(f"global:{query.query_id}" for query in self.queries),
                    "loss:bce",
                ),
                route_ids=self.route_ids,
                destination_transport_hash=str(self.local_destination_transport_hash),
                loss_pullback_hash=loss_pullback_hash,
                sketch_rank=self._delegate.local_covariance_sketch_rank,
                sketch_seed=self._delegate.local_covariance_sketch_seed,
                device=self.evaluator.device,
            )
            trust = (
                None
                if trust_spec is None
                else DeviceFrequentDirectionsTrustAccumulator(
                    route_ids=self.route_ids,
                    sketch_rank=requested_rank,
                    device=self.evaluator.device,
                )
            )
            entry = {
                "tripartite": tripartite,
                "trust": trust,
                "request_ids": [],
                "fc_receipts": [],
            }
            self._sequential_device_states[key] = entry
        tripartite = entry["tripartite"]
        trust = entry["trust"]
        existing_ids = tuple(tripartite.context_ids)
        expected_prior = () if replay_reason is not None else selected_contexts[:prior_count]
        if existing_ids != expected_prior:
            raise RuntimeError("device response accumulator is not the sealed prefix")
        addition = (
            selected_contexts if replay_reason is not None else selected_contexts[prior_count:]
        )
        route_indices = tuple(range(len(self.route_ids)))

        def consume(
            *,
            context_ids: tuple[str, ...],
            rows: Any,
            destination_transport_hash: str | None,
            loss_pullback_hash: str | None,
        ) -> None:
            if (
                str(destination_transport_hash or "") != tripartite.destination_transport_hash
                or str(loss_pullback_hash or "") != tripartite.loss_pullback_hash
            ):
                raise ValueError("device sequential response pullback identity changed")
            tripartite.update(rows, context_ids=context_ids)
            if trust is not None:
                trust.update(
                    rows[:, 1:-1, :],
                    context_ids=context_ids,
                )

        # The Frequent-Directions trust sketch consumes whole-axis
        # observations (its SVD couples every route), so trust-bearing
        # requests keep the original full-axis consumer; the fused
        # tripartite-only update streams route chunks instead.
        chunk_state = (
            None
            if trust is not None
            else self._route_chunked_update_for(
                tripartite,
                identity_error=("device sequential response pullback identity changed"),
            )
        )
        if chunk_state is not None:
            self._device_chunk_state = chunk_state
        else:
            self._device_summary_consumer = consume
        self._active_host_context_route_rows_transfer_count = 0
        try:
            tuple(
                self._response_batches(
                    addition,
                    route_indices,
                    self.context_packet_size,
                )
            )
            host_transfer_count = self._active_host_context_route_rows_transfer_count
        finally:
            self._device_summary_consumer = None
            self._device_chunk_state = None
        route_chunk_metadata: dict[str, Any] = (
            {}
            if chunk_state is None
            else {
                "route_chunked_streaming": True,
                "route_chunk_size": int(self.response_route_chunk_size),
                "route_chunk_update_count": int(chunk_state.chunk_update_count),
                "fused_route_moments_requested": bool(chunk_state.fused_requested),
                "fused_route_moments_engaged": bool(chunk_state.fused_chunk_update_count),
                "fused_route_moment_update_count": int(chunk_state.fused_chunk_update_count),
                "fused_score_epilogue_receipts": tuple(
                    chunk_state.fused_score_epilogue_receipts
                ),
            }
        )
        if tuple(tripartite.context_ids) != selected_contexts:
            raise RuntimeError("device response update did not deliver the cumulative prefix")
        entry["request_ids"].append(request.request_id)
        entry["fc_receipts"].extend(self._active_fc_nonnegative_input_receipts)
        pullback: TripartitePullbackMomentTensor = tripartite.finalize()
        global_slice = slice(1, len(pullback.head_ids) - 1)
        mean_estimator = pullback.mean_estimator_covariance[:, global_slice, global_slice]
        response = ResponseMomentTensor(
            channel_ids=tuple(query.query_id for query in self.queries),
            route_ids=self.route_ids,
            mean=pullback.mean[global_slice],
            second_moment_diagonal=pullback.route_head_second_moment[:, global_slice, global_slice],
            mean_standard_error=np.sqrt(
                np.maximum(
                    np.diagonal(mean_estimator, axis1=1, axis2=2).T,
                    0.0,
                )
            ),
            second_moment_standard_error=(
                pullback.second_moment_standard_error[:, global_slice, global_slice]
            ),
            sample_count=pullback.sample_count,
            response_spec_hash=self._delegate.response_spec_hash,
            metric_id=self._delegate.metric_id,
            context_axis_hash=content_hash(selected_contexts, prefix="response-context-axis-"),
            mean_estimator_covariance_diagonal=mean_estimator,
            metadata={
                "streaming_sufficient_statistics": True,
                "device_sequential_sufficient_statistics": True,
                "contextual_variability_separate_from_estimation_uncertainty": True,
                "host_context_route_rows_transferred": bool(host_transfer_count),
                "host_context_route_row_transfer_count": int(host_transfer_count),
                "projection_residual_bound": (self._delegate.projection_residual_bound),
            },
        )
        local = LocalAttributionMomentTensor(
            route_ids=self.route_ids,
            mean=pullback.mean[0],
            second_moment=pullback.route_head_second_moment[:, 0, 0],
            mean_standard_error=np.sqrt(
                np.maximum(pullback.mean_estimator_covariance[:, 0, 0], 0.0)
            ),
            second_moment_standard_error=(pullback.second_moment_standard_error[:, 0, 0]),
            covariance_factor=pullback.covariance_factor[:, 0, :],
            sample_count=pullback.sample_count,
            context_axis_hash=pullback.context_axis_hash,
            destination_transport_hash=(pullback.destination_transport_hash),
            sketch_seed=pullback.sketch_seed,
        )
        geometry = (
            None
            if trust is None or trust_spec is None
            else trust.finalize(
                spec=trust_spec,
                measured_mask_hash=mask.mask_hash,
                calibration_state_id=calibration_id,
                source_measurement_hashes=tuple(entry["request_ids"]),
                metadata={
                    "requested_rank": requested_rank,
                    "sequential_prefix_updates": True,
                },
            )
        )
        if self.in_process_typed_response_fast_path:
            summary = InProcessResponseSummary(
                request_id=request.request_id,
                state_hash=request.state_hash,
                context_ids=selected_contexts,
                route_ids=self.route_ids,
                response_moments=response,
                local_attribution=local,
                tripartite_pullback=pullback,
                global_trust_geometry=geometry,
            )
            summary_token = register_in_process_response_summary(summary)
            return MeasurementOutcome.create(
                request_id=request.request_id,
                state_hash=request.state_hash,
                context_ids=selected_contexts,
                candidate_ids=request.candidate_ids,
                values=np.asarray([[float(pullback.sample_count)]], dtype=np.float64),
                calibration_state_id=calibration_id,
                realized_cost=RealizedCost(
                    forward_passes=float(len(addition)),
                    backward_passes=float(len(addition)),
                ),
                metadata={
                    "kind": "response_moments",
                    "mask_hash": mask.mask_hash,
                    "response_contract_hash": (self._delegate.response_spec_hash),
                    "response_metric_id": self._delegate.metric_id,
                    "response_dimension": len(self.queries),
                    "response_channel_ids": tuple(query.query_id for query in self.queries),
                    "full_route_count": len(self.route_ids),
                    "returned_route_count": len(self.route_ids),
                    "whole_physical_route_axis": True,
                    "sample_count": len(selected_contexts),
                    "statistic_rows": ("in_process_typed_device_response_summaries"),
                    "device_sequential_sufficient_statistics": True,
                    "sealed_prefix_suffix_only_update": (replay_reason is None),
                    "prefix_replay_reason": replay_reason,
                    "physical_addition_count": len(addition),
                    "physical_addition_hash": content_hash(
                        addition,
                        prefix="irg-response-panel-addition-",
                    ),
                    "host_context_route_rows_transferred": bool(host_transfer_count),
                    "host_context_route_row_transfer_count": int(host_transfer_count),
                    "route_level_summaries_only": True,
                    "in_process_typed_response_summary": True,
                    "in_process_summary_schema": summary.schema,
                    "in_process_summary_token": summary_token,
                    "in_process_summary_route_axis_hash": (summary.route_axis_hash),
                    "in_process_summary_component_hashes": (summary.component_hashes),
                    "compact_outcome_is_lineage_only": True,
                    "compact_outcome_replay_requires_recomputation": True,
                    "global_trust_cross_partition_terms_retained": (geometry is not None),
                    "global_trust_factorization_is_not_partitioning": (geometry is not None),
                    "global_trust_residual_certified": geometry is not None,
                    "fc_nonnegative_input_receipts": tuple(entry["fc_receipts"]),
                    **route_chunk_metadata,
                },
            )
        values, moment_layout = response.pack()
        local_values, local_layout = local.pack()
        moment_layout = compact_string_axis_layout(
            moment_layout,
            axis=self.route_ids,
            axis_key="route_ids",
            name="response moment route axis",
        )
        local_layout = compact_string_axis_layout(
            local_layout,
            axis=self.route_ids,
            axis_key="route_ids",
            name="local attribution route axis",
        )
        local_offset = int(values.shape[0])
        values = np.concatenate((values, local_values), axis=0)
        pullback_values, pullback_layout = pullback.pack()
        pullback_layout = compact_string_axis_layout(
            pullback_layout,
            axis=self.route_ids,
            axis_key="route_ids",
            name="tripartite pullback route axis",
        )
        pullback_offset = int(values.shape[0])
        values = np.concatenate((values, pullback_values), axis=0)
        metadata: dict[str, Any] = {
            "kind": "response_moments",
            "mask_hash": mask.mask_hash,
            "response_contract_hash": self._delegate.response_spec_hash,
            "response_metric_id": self._delegate.metric_id,
            "response_dimension": len(self.queries),
            "response_channel_ids": tuple(query.query_id for query in self.queries),
            "full_route_count": len(self.route_ids),
            "returned_route_count": len(self.route_ids),
            "whole_physical_route_axis": True,
            "moment_layout": moment_layout,
            "local_attribution_layout": {
                **local_layout,
                "row_offset": local_offset,
                "row_count": int(local_values.shape[0]),
            },
            "tripartite_pullback_layout": {
                **pullback_layout,
                "row_offset": pullback_offset,
                "row_count": int(pullback_values.shape[0]),
            },
            "sample_count": len(selected_contexts),
            "statistic_rows": "device_sequential_matrix_response_moments",
            "device_sequential_sufficient_statistics": True,
            "sealed_prefix_suffix_only_update": replay_reason is None,
            "prefix_replay_reason": replay_reason,
            "physical_addition_count": len(addition),
            "physical_addition_hash": content_hash(addition, prefix="irg-response-panel-addition-"),
            "host_context_route_rows_transferred": bool(host_transfer_count),
            "host_context_route_row_transfer_count": int(host_transfer_count),
            "route_level_summaries_only": True,
            "global_trust_cross_partition_terms_retained": geometry is not None,
            "global_trust_factorization_is_not_partitioning": geometry is not None,
            "global_trust_residual_certified": geometry is not None,
            "fc_nonnegative_input_receipts": tuple(entry["fc_receipts"]),
            **route_chunk_metadata,
        }
        if geometry is not None:
            trust_values, trust_layout = geometry.pack()
            trust_layout = compact_string_axis_layout(
                trust_layout,
                axis=self.route_ids,
                axis_key="route_ids",
                name="global trust route axis",
            )
            trust_offset = int(values.shape[0])
            values = np.concatenate((values, trust_values), axis=0)
            metadata["global_trust_geometry_layout"] = {
                **trust_layout,
                "row_offset": trust_offset,
                "row_count": int(trust_values.shape[0]),
            }
        return MeasurementOutcome.create(
            request_id=request.request_id,
            state_hash=request.state_hash,
            context_ids=selected_contexts,
            candidate_ids=request.candidate_ids,
            values=values,
            calibration_state_id=calibration_id,
            realized_cost=RealizedCost(
                forward_passes=float(len(addition)),
                backward_passes=float(len(addition)),
            ),
            metadata=metadata,
        )

    def accumulate_device_tripartite(
        self,
        request: MeasurementRequest,
        *,
        accumulator: Any,
        flip_margin_accumulator: DenseFlipMarginAccumulator | None = None,
        gate_disclosure_accumulator: Any | None = None,
    ) -> Mapping[str, object]:
        """Reduce one nested context addition into an accelerator accumulator.

        The method is the v0/v1 partition-field fast path.  It consumes the
        canonical fused U→G→D rows before the device boundary and never
        transfers a context×route tensor to host memory.
        """

        if request.kind != "response_moments":
            raise ValueError("device tripartite accumulation requires response_moments")
        if (
            gate_disclosure_accumulator is not None
            and not self.conserving_activation_bindings
        ):
            raise ValueError(
                "gate-removal disclosure requires exact activation capture"
            )
        mask_hash = str(request.payload.get("mask_hash", ""))
        if not mask_hash:
            raise ValueError("device tripartite request requires mask_hash")
        selected = self._delegate._selected_contexts(request)
        route_indices, route_ids = self._delegate._routes(request)
        whole_ordinal_axis = (
            isinstance(route_indices, range)
            and route_indices.start == 0
            and route_indices.stop == len(self.route_ids)
            and route_indices.step == 1
        ) or (
            len(route_indices) == len(self.route_ids)
            and all(index == ordinal for ordinal, index in enumerate(route_indices))
        )
        if not whole_ordinal_axis:
            raise ValueError("device partition acquisition requires the whole physical route axis")
        accumulator_route_ids = getattr(accumulator, "route_ids", ())
        if accumulator_route_ids is not route_ids and (
            len(accumulator_route_ids) != len(route_ids)
            or ordered_string_axis_hash(
                accumulator_route_ids,
                name="device accumulator route axis",
            )
            != ordered_string_axis_hash(
                route_ids,
                name="optimized IRG scoring route axis",
            )
        ):
            raise ValueError("device accumulator and optimized scorer route axes differ")
        import time

        started = time.perf_counter()

        def consume(
            *,
            context_ids: tuple[str, ...],
            rows: Any,
            destination_transport_hash: str | None,
            loss_pullback_hash: str | None,
        ) -> None:
            if str(destination_transport_hash or "") != str(
                getattr(accumulator, "destination_transport_hash", "")
            ) or str(loss_pullback_hash or "") != str(
                getattr(accumulator, "loss_pullback_hash", "")
            ):
                raise ValueError("device scorer and accumulator pullback identities differ")
            accumulator.update(rows, context_ids=context_ids)

        chunk_state = self._loss_only_chunk_consumer_for(accumulator)
        if chunk_state is None:
            chunk_state = self._route_chunked_update_for(
                accumulator,
                identity_error=("device scorer and accumulator pullback identities differ"),
            )
        with self._lock:
            mask = self.mask_resolver(mask_hash)
            calibration = self.calibration_resolver(mask)
            self._active_mask = mask
            self._active_calibration = calibration
            self._active_request_id = request.request_id
            self._census_guarded_request_id = ""
            self._active_fc_nonnegative_input_receipts = ()
            self._active_flip_margin_accumulator = flip_margin_accumulator
            self._active_context_packet_count = 0
            self._active_host_context_route_rows_transfer_count = 0
            self._active_exact_loss_stream_receipts = []
            self._active_gate_disclosure_accumulator = (
                gate_disclosure_accumulator
            )
            if chunk_state is not None:
                self._device_chunk_state = chunk_state
            else:
                self._device_summary_consumer = consume
            try:
                tuple(
                    self._response_batches(
                        selected,
                        route_indices,
                        self.context_packet_size,
                    )
                )
                # The active fit is cleared on leaving this request; retain
                # the registered identity of the fit validated for its rows.
                dense_calibration_state_id = (
                    str(calibration.calibration_state_id)
                    if self.census_dense_calibration_state_hash is not None
                    else None
                )
                delivered_packets = self._active_context_packet_count
                fc_receipts = self._active_fc_nonnegative_input_receipts
                host_transfer_count = self._active_host_context_route_rows_transfer_count
                exact_loss_stream_receipts = tuple(
                    self._active_exact_loss_stream_receipts
                )
            except Exception:
                invalidate = getattr(
                    gate_disclosure_accumulator,
                    "invalidate_gate_disclosure_stream",
                    None,
                )
                if callable(invalidate):
                    invalidate("optimized_provider_acquisition_failure")
                raise
            finally:
                self._device_summary_consumer = None
                self._device_chunk_state = None
                self._active_flip_margin_accumulator = None
                self._active_mask = None
                self._active_calibration = None
                self._active_request_id = ""
                self._active_gate_disclosure_accumulator = None
        bce_stream_receipts = tuple(
            item
            for item in exact_loss_stream_receipts
            if str(
                item.get(
                    "stream_coordinate",
                    _BCE_POSITIVE_REMOVAL_COORDINATE,
                )
            )
            == _BCE_POSITIVE_REMOVAL_COORDINATE
        )
        source_capture_preregistered = (
            request.payload.get("source_capture_identity_hash") is not None
        )
        # Canonical sampled-D V5 preregisters a distinct source-capture
        # identity in the predictable MeasurementRequest payload.  Bind that
        # identity, together with the request's own content-derived plan ID,
        # into both the outer provider receipt and the exact raw dual-stream
        # child.  Older request schemas intentionally omit the payload field
        # and retain byte-for-byte legacy receipt semantics.
        (
            exact_loss_stream_receipts,
            raw_target_logit_stream_receipts,
            source_capture_identity,
        ) = _bind_preregistered_source_capture_receipts(
            request=request,
            exact_stream_receipts=exact_loss_stream_receipts,
            expected_context_block_ids=(
                tuple(
                    _optimized_response_context_block_id(
                        provider_plan_identity_hash=request.request_id,
                        context_ids=selected[start : start + self.context_packet_size],
                    )
                    for start in range(0, len(selected), self.context_packet_size)
                )
                if source_capture_preregistered
                else None
            ),
        )
        receipt = {
            "schema": "optimized_device_tripartite_panel_receipt_v1",
            "request_id": request.request_id,
            "context_ids_hash": content_hash(selected, prefix="irg-device-panel-contexts-"),
            "context_count": len(selected),
            "context_packet_count": int(delivered_packets),
            "context_packet_size": self.context_packet_size,
            "elapsed_seconds": float(time.perf_counter() - started),
            "route_count": len(route_ids),
            "head_ids": tuple(getattr(accumulator, "head_ids", ())),
            "device_reduction": True,
            "host_context_route_rows_transferred": bool(host_transfer_count),
            "host_context_route_row_transfer_count": int(host_transfer_count),
            "route_level_summaries_only": True,
            "local_regime_diagnostics_enabled": bool(self.local_regime_specs_by_layer),
            "conserving_activation_attribution_enabled": bool(
                self.conserving_activation_bindings
            ),
            "fc_nonnegative_input_receipts": fc_receipts,
            "flip_margin_geometry_observed": bool(flip_margin_accumulator is not None),
            "gate_removal_disclosure_enabled": bool(
                gate_disclosure_accumulator is not None
            ),
            "exact_conserving_single_pass_loss_streaming": bool(
                bce_stream_receipts
            ),
            "exact_conserving_single_pass_loss_path_streaming": any(
                bool(item.get("loss_path_integrated")) for item in bce_stream_receipts
            ),
            "exact_conserving_loss_stream_receipts": (
                bce_stream_receipts
            ),
            "exact_conserving_single_pass_positive_removal_streaming": bool(
                exact_loss_stream_receipts
            ),
            "exact_conserving_positive_removal_stream_receipts": (
                exact_loss_stream_receipts
            ),
            "exact_conserving_single_pass_target_logit_streaming": bool(
                raw_target_logit_stream_receipts
            ),
            "exact_conserving_target_logit_stream_receipts": (
                raw_target_logit_stream_receipts
            ),
        }
        if self.decision_fixed_offset_resolver is not None:
            receipt["pruning_functional"] = "substituted_softmax"
        if source_capture_identity is not None:
            receipt.update(
                {
                    "provider_plan_identity_hash": request.request_id,
                    "source_capture_identity_hash": source_capture_identity,
                }
            )
        if self.census_dense_calibration_state_hash is not None:
            receipt.update({
                "pruning_guiding_loss_id": self.pruning_guiding_loss_id,
                "dense_calibration_state_hash": self.census_dense_calibration_state_hash,
                "dense_calibration_state_id": dense_calibration_state_id,
                "loss_pullback_field": "loss_gradient",
                "competitor_gradient": "zero",
            })
        if gate_disclosure_accumulator is not None:
            disclosure_receipt = getattr(
                gate_disclosure_accumulator,
                "gate_disclosure_streaming_receipt",
                None,
            )
            if not callable(disclosure_receipt):
                raise TypeError(
                    "gate-disclosure accumulator lacks a receipt method"
                )
            receipt["gate_removal_disclosure_receipt"] = (
                disclosure_receipt()
            )
        if chunk_state is not None:
            if isinstance(chunk_state, _RouteChunkedLossOnlyConsumer):
                receipt.update(chunk_state.route_streaming_receipt())
            else:
                receipt.update(
                    {
                        "route_chunked_streaming": True,
                        "route_chunk_size": int(self.response_route_chunk_size),
                        "route_chunk_update_count": int(chunk_state.chunk_update_count),
                        "fused_route_moments_requested": bool(chunk_state.fused_requested),
                        "fused_route_moments_engaged": bool(
                            chunk_state.fused_chunk_update_count
                        ),
                        "fused_route_moment_update_count": int(
                            chunk_state.fused_chunk_update_count
                        ),
                        "fused_score_epilogue_receipts": tuple(
                            chunk_state.fused_score_epilogue_receipts
                        ),
                    }
                )
        telemetry = getattr(self.scoring_backend, "performance_telemetry", None)
        if isinstance(telemetry, list) and isinstance(
            chunk_state,
            _RouteChunkedLossOnlyConsumer,
        ):
            telemetry.append(
                {
                    "schema": "optimized_loss_only_host_transfer_telemetry_v1",
                    "request_id": request.request_id,
                    "positive_removal_stream_coordinate": receipt.get(
                        "positive_removal_stream_coordinate"
                    ),
                    "context_packet_count": int(delivered_packets),
                    "host_context_route_rows_transferred": bool(
                        receipt.get("host_context_route_rows_transferred")
                    ),
                    "host_context_route_row_transfer_count": int(
                        receipt.get("host_context_route_row_transfer_count", 0)
                    ),
                    "host_context_route_row_transfer_bytes": int(
                        receipt.get("host_context_route_row_transfer_bytes", 0)
                    ),
                    "host_context_route_transfer_role": receipt.get(
                        "host_context_route_transfer_role",
                        "none",
                    ),
                    "duplicate_source_row_hash_d2h_avoided": bool(
                        receipt.get("duplicate_source_row_hash_d2h_avoided")
                    ),
                }
            )
        return receipt

    # ------------------------------------------------------------------
    # Factored D protocol: factors instead of rows.
    # ------------------------------------------------------------------

    @property
    def factored_numbers_per_context(self) -> int | None:
        """Declared per-context factor count for the custody forecast, or ``None``."""

        return self._factored_numbers_per_context

    def accumulate_device_factored(
        self,
        request: MeasurementRequest,
        *,
        accumulator: Any,
    ) -> Mapping[str, object]:
        """Capture one nested context packet as factored rows (no route rows).

        Sibling of :meth:`accumulate_device_tripartite` for the factored
        device-resident D lane (``factored_device_resident_v8``).  The same
        request grammar, mask/calibration resolution, whole-axis check and
        context-block identity are used; the difference is that the scoring
        adapter hands the per-layer factors of one VJP to ``accumulator``
        (``V1FactoredRowPacketAccumulator`` protocol) and never forms a
        ``contexts x route`` tensor.  Raw target-logit coordinate only; the
        accumulator's stream identities must equal this provider's derived
        corrected and literal-Jacobian identities, or the call refuses.
        """

        if request.kind != "response_moments":
            raise ValueError("device factored accumulation requires response_moments")
        refuse = getattr(accumulator, "refuse", None)
        if not callable(refuse) or not callable(getattr(accumulator, "capture_packet", None)):
            raise TypeError(
                "factored accumulator lacks the reviewed packet protocol "
                "(capture_packet / refuse)"
            )
        if not self.conserving_activation_bindings:
            refuse(
                "FACTORED_PROVIDER_ABSENT",
                "factored capture requires exact conserving activation bindings",
            )
        mask_hash = str(request.payload.get("mask_hash", ""))
        if not mask_hash:
            raise ValueError("device factored request requires mask_hash")
        source_capture_identity = request.payload.get("source_capture_identity_hash")
        if not str(source_capture_identity or ""):
            raise ValueError(
                "device factored request requires a preregistered "
                "source_capture_identity_hash"
            )
        if len(self.queries) != 1:
            raise ValueError("factored capture requires one scalar target-logit query")
        query = self.queries[0]
        target_index = _exact_raw_target_logit_class_id(query)
        corrected_identity = _raw_target_logit_positive_removal_stream_identity(
            query_id=str(query.query_id),
            target_class_id=target_index,
        )
        literal_identity = _raw_target_logit_literal_jacobian_stream_identity(
            query_id=str(query.query_id),
            target_class_id=target_index,
        )
        design = getattr(accumulator, "design", None)
        if (
            design is None
            or str(getattr(design, "corrected_a_stream_identity_hash", ""))
            != corrected_identity
            or str(getattr(design, "literal_j_stream_identity_hash", ""))
            != literal_identity
        ):
            refuse(
                "STALE_IDENTITY",
                "factored accumulator stream identities differ from the provider's "
                "raw target-logit corrected/literal-Jacobian identities",
            )
        selected = self._delegate._selected_contexts(request)
        route_indices, route_ids = self._delegate._routes(request)
        whole_ordinal_axis = (
            isinstance(route_indices, range)
            and route_indices.start == 0
            and route_indices.stop == len(self.route_ids)
            and route_indices.step == 1
        ) or (
            len(route_indices) == len(self.route_ids)
            and all(index == ordinal for ordinal, index in enumerate(route_indices))
        )
        if not whole_ordinal_axis:
            raise ValueError("device factored acquisition requires the whole physical route axis")
        accumulator_route_ids = getattr(accumulator, "route_ids", ())
        if accumulator_route_ids is not route_ids and (
            len(accumulator_route_ids) != len(route_ids)
            or ordered_string_axis_hash(
                accumulator_route_ids,
                name="factored accumulator route axis",
            )
            != ordered_string_axis_hash(
                route_ids,
                name="optimized IRG scoring route axis",
            )
        ):
            raise ValueError("factored accumulator and optimized scorer route axes differ")
        expected_contexts = tuple(getattr(accumulator, "expected_context_ids", ()))
        if expected_contexts and tuple(selected) != expected_contexts:
            refuse(
                "AXIS_MISMATCH",
                "factored accumulator expects another context packet",
            )
        import time

        started = time.perf_counter()
        with self._lock:
            mask = self.mask_resolver(mask_hash)
            calibration = self.calibration_resolver(mask)
            self._active_mask = mask
            self._active_calibration = calibration
            self._active_request_id = request.request_id
            self._census_guarded_request_id = ""
            self._active_context_packet_count = 0
            try:
                self._require_dense_census_calibration()
                context_block_id = _optimized_response_context_block_id(
                    provider_plan_identity_hash=request.request_id,
                    context_ids=selected,
                )
                self._active_context_packet_count = 1
                self._batch_provider.activate(
                    context_block_id=context_block_id,
                    context_ids=selected,
                )
                # The chart is the masked network: gates are applied exactly as
                # the row protocol applies them before capture.
                for layer_id, gate in self._masks_by_layer(mask).items():
                    self.model_adapter.gate_graph.set_gate(layer_id, gate)
                plan = self._measurement_plan(context_block_id=context_block_id)
                plan.validate().require_valid()
                begin_capture = getattr(accumulator, "begin_capture_source", None)
                if callable(begin_capture):
                    begin_capture(provider=self, request=request, current_mask=mask,
                                  current_calibration=calibration)
                elif self.evaluator.device.type != "cpu" or self._factored_capture_admission is not None:
                    raise TypeError("production factored capture requires its admission-aware accumulator")
                capture_receipt = self.scoring_backend.capture_factored_rows(
                    plan,
                    context_block_id=context_block_id,
                    expected_context_ids=selected,
                    route_ids=self.route_ids,
                    route_gate=mask.to_bool_array(),
                    destination_transport_hash=str(
                        self.local_destination_transport_hash or ""
                    ),
                    corrected_stream_identity_hash=corrected_identity,
                    literal_jacobian_stream_identity_hash=literal_identity,
                    accumulator=accumulator,
                )
            finally:
                self._active_mask = None
                self._active_calibration = None
                self._active_request_id = ""
        receipt: dict[str, object] = {
            "schema": "optimized_device_factored_capture_receipt_v1",
            "request_id": request.request_id,
            "provider_plan_identity_hash": request.request_id,
            "source_capture_identity_hash": str(source_capture_identity),
            "context_ids_hash": content_hash(selected, prefix="irg-device-panel-contexts-"),
            "context_count": len(selected),
            "context_packet_count": 1,
            "elapsed_seconds": float(time.perf_counter() - started),
            "route_count": len(route_ids),
            "device_reduction": True,
            "host_context_route_rows_transferred": False,
            "host_context_route_row_transfer_count": 0,
            "route_level_summaries_only": True,
            "conserving_activation_attribution_enabled": True,
            "positive_removal_stream_coordinate": (
                _RAW_TARGET_LOGIT_POSITIVE_REMOVAL_COORDINATE
            ),
            "factored_custody": True,
            "raw_dual_row_custody_retained": False,
        }
        capture = dict(capture_receipt)
        receipt["capture_receipt_schema"] = str(capture.pop("schema", ""))
        receipt.update(capture)
        return receipt


def factored_numbers_per_context_from_probe(
    model: Any,
    *,
    affine_module_paths: Sequence[str],
    example_input: Any | None = None,
    example_shape: Sequence[int] | None = None,
) -> int:
    """Per-context factor count of the bound affine layers by one no-grad probe.

    The count is ``input.numel + 2 * output.numel`` per context per affine
    module (input factor, post-activation cotangent, preactivation), read
    from a forward pass of ``example_input`` (batch of one) with
    forward hooks on the named modules.  Exactly one of ``example_input`` and
    ``example_shape`` is given; a shape is probed with a float32 zero tensor on
    the model's parameter device.  It is an engineering forecast numeral for
    the factored custody admission, never a decision input.
    """

    if torch is None:
        raise RuntimeError("factored custody probe requires PyTorch")
    if (example_input is None) == (example_shape is None):
        raise ValueError("factored custody probe takes exactly one of example_input or example_shape")
    if example_shape is not None:
        shape = tuple(int(value) for value in example_shape)
        if not shape or any(value <= 0 for value in shape):
            raise ValueError("factored custody probe example shape must be positive")
        example_input = torch.zeros(
            shape,
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
    paths = tuple(str(path) for path in affine_module_paths)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("factored custody probe requires unique affine module paths")
    modules = dict(model.named_modules())
    missing = [path for path in paths if path not in modules]
    if missing:
        raise ValueError(f"factored custody probe: unknown module paths {missing!r}")
    example = torch.as_tensor(example_input)
    if example.ndim < 1 or int(example.shape[0]) != 1:
        raise ValueError("factored custody probe expects an example batch of exactly one context")
    seen: dict[str, int] = {}
    handles = []

    def hook_for(path: str):
        def hook(_module: Any, inputs: Any, output: Any) -> None:
            source = inputs[0] if isinstance(inputs, tuple) else inputs
            seen[path] = int(source.numel()) + 2 * int(output.numel())

        return hook

    try:
        for path in paths:
            handles.append(modules[path].register_forward_hook(hook_for(path)))
        was_training = bool(getattr(model, "training", False))
        model.eval()
        with torch.no_grad():
            model(example)
        if was_training:
            model.train()
    finally:
        for handle in handles:
            handle.remove()
    unseen = [path for path in paths if path not in seen]
    if unseen:
        raise ValueError(f"factored custody probe: modules not reached by the forward {unseen!r}")
    return int(sum(seen[path] for path in paths))


__all__ = [
    "alexnet_v1_conserving_activation_bindings",
    "factored_numbers_per_context_from_probe",
    "build_route_chunked_dual_chart_fanout",
    "fc_nonnegative_input_receipt",
    "OptimizedScoringResponseProvider",
    "resnet_local_destination_transport",
    "resnet_local_regime_capture_specs",
    "route_chunk_spans",
    "RESPONSE_ROUTE_CHUNK_ROUTES_DEFAULT",
    "RESPONSE_ROUTE_CHUNK_ROUTES_ENVIRONMENT",
]
