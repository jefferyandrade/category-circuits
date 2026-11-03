"""Production adapter boundary for exact, block-streamed measurement.

Model adapters own model construction, preprocessing, capture hooks, and
prefix/suffix replay.  The canonical producer owns query ordering, exact VJP
math, context-block aggregation, and sufficient-statistic emission.  Raw
context-by-route values never cross this boundary as durable data.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any, Protocol

import numpy as np

from mechanistic_systems.core.storage import RouteBlockSpec
from mechanistic_systems.contracts import MeasurementPlan, QuerySpec
from mechanistic_systems.core.numerics import (
    AutogradBackend,
    CapturedRouteBlock,
    CachedQueryVJP,
    VJPTelemetry,
    score_captured_route_blocks,
)
from mechanistic_systems.measurement.queries import (
    DirectionResolver,
    QueryCotangentBatch,
    evaluate_queries,
)
from mechanistic_systems.measurement.streaming import (
    MeasurementMomentTile,
    context_sketch_matrix,
)
from mechanistic_systems.measurement.local_regime import (
    merge_local_regime_block_statistics,
)
from mechanistic_systems.measurement.pruning_loss import FrozenPruningLossPath
from mechanistic_systems.circuits.irg.modern_v1_conserving_route_score import (
    LITERAL_GATE_JACOBIAN_EQUATION_ID,
    LITERAL_GATE_JACOBIAN_ROUTE_SCORE_ID,
    RELU_ZERO_SUBGRADIENT_POLICY_ID,
    V1ConservingActivationCapture,
    V1GateDisclosureRequest,
    score_captured_conserving_route_blocks,
)
from mechanistic_systems.backends import (
    BackendRegistry,
    PhaseProfiler,
    build_default_backend_registry,
)

try:  # Optional for import smoke; required by native measurement execution.
    import torch
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("native measurement production requires PyTorch")


def _native_float32_rows(
    rows: Any,
    *,
    work_device: Any,
    leading_shape: tuple[int, ...],
) -> Any:
    """Native-mode retained rows without the full-width float64 transient.

    The legacy computation materialized ``rows.to(float64)`` at full
    ``[*leading, block_routes]`` width and then cast that copy to float32,
    so one capture batch transiently held source + float64 + float32 rows
    simultaneously — a residency that scales with the CAPTURE batch and
    measured ~14.5 GB for AlexNet fc1 at batch 32
    (OOM at the float32 cast; 32 × 37,748,736 routes).

    Native mode performs no reduction — only an elementwise cast — and the
    float64 round-trip is elementwise exact for every floating source
    dtype: float16/bfloat16/float32 embed exactly into float64, and
    IEEE-754 round-to-nearest returns any exactly representable value
    unchanged, so ``rows.to(f64).to(f32)`` is bitwise ``rows.to(f32)``.
    Emitting the direct cast is therefore bit-identical while allocating
    nothing at all when the source is already float32 (every route
    reducer's emission dtype; the result aliases the reducer's own
    per-batch buffer, which nothing mutates after emission) and exactly
    the retained output otherwise. Non-floating sources (no route reducer
    emits them) keep the legacy double-rounding semantics verbatim.
    """

    detached = rows.detach()
    if not detached.dtype.is_floating_point:  # pragma: no cover - defensive
        return (
            detached.to(device=work_device, dtype=torch.float64)
            .reshape(*leading_shape, -1)
            .to(torch.float32)
        )
    return (
        detached.to(device=work_device)
        .reshape(*leading_shape, -1)
        .to(torch.float32)
    )


@dataclass(frozen=True)
class CapturedMeasurementBatch:
    """One ephemeral model execution batch inside a frozen context block."""

    context_block_id: str
    response_matrix: Any
    labels: Any
    route_blocks: tuple[CapturedRouteBlock, ...]
    context_ids: tuple[str, ...]
    conserving_activation_captures: (
        tuple[V1ConservingActivationCapture, ...] | None
    ) = None
    census_route_vjp: CachedQueryVJP | None = None
    census_activation_vjp: CachedQueryVJP | None = None

    def validate(self) -> None:
        _require_torch()
        if not self.context_block_id:
            raise ValueError("captured measurement batch requires a context-block ID")
        if not torch.is_tensor(self.response_matrix) or self.response_matrix.ndim != 2:
            raise ValueError("captured response matrix must be [context,response_coordinate]")
        if len(self.context_ids) != int(self.response_matrix.shape[0]):
            raise ValueError("captured context IDs do not match the response matrix")
        if not self.context_ids or len(set(self.context_ids)) != len(self.context_ids):
            raise ValueError("captured context IDs must be nonempty and unique within a batch")
        if not self.route_blocks:
            raise ValueError("captured measurement batch has no route blocks")
        for block in self.route_blocks:
            block.validate()
            if int(block.output.shape[0]) != int(self.response_matrix.shape[0]):
                raise ValueError("captured layer and response context axes differ")
        captures = self.conserving_activation_captures
        if captures is not None:
            if len(captures) != len(self.route_blocks):
                raise ValueError(
                    "conserving activation captures do not cover the route axis"
                )
            for block, capture in zip(self.route_blocks, captures, strict=True):
                capture.validate_for(block)


GateDisclosureRequestFactory = Callable[
    [CapturedMeasurementBatch, QueryCotangentBatch],
    V1GateDisclosureRequest | None,
]

LossPullbackFactory = Callable[
    [CapturedMeasurementBatch, QueryCotangentBatch],
    Any,
]
LossPathFactory = Callable[
    [CapturedMeasurementBatch, QueryCotangentBatch],
    FrozenPruningLossPath,
]


class NativeMeasurementModelAdapter(Protocol):
    """Model-specific capture boundary for canonical measurement."""

    adapter_id: str
    architecture_id: str
    topology_hash: str
    route_schema_hash: str

    def route_blocks(self, plan: MeasurementPlan) -> Sequence[RouteBlockSpec]:
        ...

    def iter_captured_batches(
        self,
        plan: MeasurementPlan,
        *,
        context_block_id: str,
        queries: Sequence[QuerySpec],
    ) -> Iterable[CapturedMeasurementBatch]:
        ...

    def direction_resolver(self) -> DirectionResolver | None:
        ...

    def scientific_metadata(self) -> dict[str, Any]:
        ...


class ContextBlockMomentAccumulator:
    """Accumulate exact image contributions into one declared context block."""

    def __init__(
        self,
        *,
        context_block_id: str,
        query_ids: Sequence[str],
        route_blocks: Sequence[RouteBlockSpec],
        estimator_id: str,
        reducer_backend: Any | None = None,
        profiler: PhaseProfiler | None = None,
        retain_context_factor: bool = False,
        retain_local_attribution: bool = False,
        retain_local_regime: bool = False,
        context_row_transport: str = "host",
        context_factor_mode: str = "sketch",
        sketch_probes: int = 64,
        sketch_seed: int = 0,
        use_conserving_activation_attribution: bool = False,
        gate_disclosure_request_factory: (
            GateDisclosureRequestFactory | None
        ) = None,
    ) -> None:
        _require_torch()
        self.context_block_id = str(context_block_id)
        self.query_ids = tuple(str(value) for value in query_ids)
        self.route_blocks = tuple(route_blocks)
        self.estimator_id = str(estimator_id)
        if not self.context_block_id or not self.query_ids or not self.route_blocks:
            raise ValueError("context moment accumulator has an empty declared axis")
        self._route_by_layer = {block.layer_id: block for block in self.route_blocks}
        if len(self._route_by_layer) != len(self.route_blocks):
            raise ValueError("context moment route layer IDs must be unique")
        self._score_sum: dict[str, Any] = {}
        self._absolute_score_sum: dict[str, Any] = {}
        self._context_ids: set[str] = set()
        self._telemetry: list[VJPTelemetry] = []
        self.reducer_backend = reducer_backend
        self.profiler = profiler or PhaseProfiler()
        # Context-factor tee (both large-R
        # modes, sketch default). Rows are reduced per microbatch on the
        # accumulation device and staged on the host at reduced width, so
        # native-width per-image data never crosses the PCIe bus except in
        # explicit native mode.
        if context_factor_mode not in {"native", "channel_marginal", "sketch"}:
            raise ValueError(
                f"unknown context-factor mode {context_factor_mode!r}"
            )
        self.retain_context_factor = bool(retain_context_factor)
        self.retain_local_attribution = bool(retain_local_attribution)
        self.retain_local_regime = bool(retain_local_regime)
        if str(context_row_transport) not in {"host", "device"}:
            raise ValueError("context-row transport must be 'host' or 'device'")
        self.context_row_transport = str(context_row_transport)
        self.context_factor_mode = str(context_factor_mode)
        self.sketch_probes = int(sketch_probes)
        self.sketch_seed = int(sketch_seed)
        self.use_conserving_activation_attribution = bool(
            use_conserving_activation_attribution
        )
        self.gate_disclosure_request_factory = (
            gate_disclosure_request_factory
        )
        if (
            self.gate_disclosure_request_factory is not None
            and not self.use_conserving_activation_attribution
        ):
            raise ValueError(
                "gate disclosure requires exact conserving attribution"
            )
        if self.use_conserving_activation_attribution and self.retain_local_regime:
            raise ValueError(
                "exact conserving attribution and legacy local-regime "
                "diagnostics cannot share one scoring pass"
            )
        self._context_row_chunks: dict[str, list[Any]] = {}
        self._local_context_row_chunks: dict[str, list[Any]] = {}
        self._local_regime_by_layer: dict[str, dict[str, Any]] = {}
        self._sketch_cache: dict[tuple[str, str], Any] = {}

    @property
    def context_count(self) -> int:
        return len(self._context_ids)

    @property
    def telemetry(self) -> tuple[VJPTelemetry, ...]:
        return tuple(self._telemetry)

    def add(
        self,
        batch: CapturedMeasurementBatch,
        *,
        query_batch: Any,
        backend: AutogradBackend,
        retain_graph: bool = False,
    ) -> None:
        batch.validate()
        if batch.context_block_id != self.context_block_id:
            raise ValueError("captured batch was added to the wrong context block")
        if tuple(query_batch.query_ids) != self.query_ids:
            raise ValueError("captured batch query axis differs from its accumulator")
        overlap = self._context_ids.intersection(batch.context_ids)
        if overlap:
            raise ValueError("context IDs were measured more than once: " + ",".join(sorted(overlap)[:8]))
        observed = tuple(block.route_block for block in batch.route_blocks)
        if tuple(block.to_dict() for block in observed) != tuple(
            block.to_dict() for block in self.route_blocks
        ):
            raise ValueError("captured route blocks changed within a frozen context block")
        with self.profiler.phase("measurement.vjp_and_route_reduce"):
            if self.use_conserving_activation_attribution:
                captures = batch.conserving_activation_captures
                if captures is None:
                    raise ValueError(
                        "exact conserving attribution was requested without "
                        "typed post-activation captures"
                    )
                result = score_captured_conserving_route_blocks(
                    batch.response_matrix,
                    query_batch,
                    batch.route_blocks,
                    captures,
                    backend=backend,
                    retain_graph=retain_graph,
                    reducer_backend=self.reducer_backend,
                    profiler=self.profiler,
                    per_image=(
                        self.retain_context_factor
                        or self.retain_local_attribution
                    ),
                    local_attribution=self.retain_local_attribution,
                    cached_vjp=batch.census_activation_vjp,
                    gate_disclosure=(
                        None
                        if self.gate_disclosure_request_factory is None
                        else self.gate_disclosure_request_factory(
                            batch,
                            query_batch,
                        )
                    ),
                )
                reduced, telemetry = result.scores, result.telemetry
            else:
                reduced, telemetry = score_captured_route_blocks(
                    batch.response_matrix,
                    query_batch,
                    batch.route_blocks,
                    backend=backend,
                    retain_graph=retain_graph,
                    reducer_backend=self.reducer_backend,
                    profiler=self.profiler,
                    per_image=(
                        self.retain_context_factor
                        or self.retain_local_attribution
                        or self.retain_local_regime
                    ),
                    local_attribution=self.retain_local_attribution,
                    local_regime=self.retain_local_regime,
                    cached_vjp=batch.census_route_vjp,
                )
        self._telemetry.append(telemetry)
        for route_scores in reduced:
            layer_id = route_scores.route_block.layer_id
            # The local derivatives are evaluated in the model's declared
            # precision, but batch and score-window reduction is a numerical
            # operation rather than part of the estimator. Accumulating on the
            # device in float64 makes runtime microbatching immaterial while
            # avoiding one host synchronization per image batch. Compact
            # accumulators are flushed only when the context block seals.
            accumulation_device = route_scores.signed_scores.device
            if accumulation_device.type == "mps":
                accumulation_device = torch.device("cpu")
            value = route_scores.signed_scores.detach().to(
                device=accumulation_device,
                dtype=torch.float64,
            )
            existing = self._score_sum.get(layer_id)
            if existing is None:
                self._score_sum[layer_id] = value.clone()
            else:
                existing.add_(value)
            absolute_value = route_scores.absolute_scores.detach().to(
                device=accumulation_device,
                dtype=torch.float64,
            )
            existing_absolute = self._absolute_score_sum.get(layer_id)
            if existing_absolute is None:
                self._absolute_score_sum[layer_id] = absolute_value.clone()
            else:
                existing_absolute.add_(absolute_value)
            if self.retain_context_factor and route_scores.per_image_scores is not None:
                # Blocks whose route kind has no per-image reducer (e.g.
                # the linear_weight readout) arrive with per_image_scores
                # = None (score_captured_route_blocks omitted them); the
                # factor simply does not cover such a layer.
                with self.profiler.phase("measurement.context_factor_reduce"):
                    self._context_row_chunks.setdefault(layer_id, []).append(
                        self._reduce_context_rows(route_scores)
                    )
            if (
                self.retain_local_attribution
                and route_scores.local_per_image_scores is not None
            ):
                with self.profiler.phase(
                    "measurement.local_attribution_context_reduce"
                ):
                    self._local_context_row_chunks.setdefault(
                        layer_id, []
                    ).append(self._reduce_local_context_rows(route_scores))
            if (
                self.retain_local_regime
                and route_scores.local_regime_statistics is not None
            ):
                with self.profiler.phase(
                    "measurement.local_regime_sufficient_statistics"
                ):
                    self._local_regime_by_layer[layer_id] = (
                        merge_local_regime_block_statistics(
                            self._local_regime_by_layer.get(layer_id),
                            route_scores.local_regime_statistics,
                        )
                    )
        self._context_ids.update(batch.context_ids)

    def _reduce_context_rows(self, route_scores: Any) -> Any:
        """Reduce one microbatch's per-image rows to the retained width.

        Semantics match ``MeasurementMomentTile.from_values`` mode for
        mode: reductions (channel marginalization, sketch projection)
        happen in float64 and the retained rows are float32. Native mode
        performs no reduction, so its rows are emitted as the direct
        elementwise float32 cast — bitwise identical to the legacy
        float64 round-trip (see ``_native_float32_rows``) — and the
        capture path never materializes a batch-scaled full-width
        ``[context, query, block-routes]`` float64 copy. Sketch rows are
        an ESTIMATE class; channel-marginal rows are the exact
        marginalization onto the decision axis.
        """

        per_image = route_scores.per_image_scores
        if per_image is None:
            raise RuntimeError(
                "context-factor retention requires per-image capture; the "
                "reducer returned no per-image rows"
            )
        block = route_scores.route_block
        work_device = per_image.device
        if work_device.type == "mps":
            work_device = torch.device("cpu")
        if self.context_factor_mode == "native":
            rows = _native_float32_rows(
                per_image,
                work_device=work_device,
                leading_shape=(
                    int(per_image.shape[0]),
                    int(per_image.shape[1]),
                ),
            )
            return (
                rows
                if self.context_row_transport == "device"
                else rows.cpu().numpy()
            )
        flat = per_image.detach().to(
            device=work_device, dtype=torch.float64
        ).reshape(per_image.shape[0], per_image.shape[1], -1)
        if self.context_factor_mode == "channel_marginal":
            rows = (
                flat.reshape(
                    flat.shape[0], flat.shape[1], int(block.shape[0]), -1
                )
                .sum(dim=-1)
                .to(torch.float32)
            )
        else:  # sketch — Ω deterministic in (sketch_seed, layer_id)
            key = (block.layer_id, str(work_device))
            omega = self._sketch_cache.get(key)
            if omega is None:
                omega = torch.from_numpy(
                    context_sketch_matrix(
                        block.layer_id,
                        routes=int(flat.shape[-1]),
                        probes=self.sketch_probes,
                        seed=self.sketch_seed,
                    )
                ).to(device=work_device)
                self._sketch_cache[key] = omega
            rows = (flat @ omega).to(torch.float32)
        return rows if self.context_row_transport == "device" else rows.cpu().numpy()

    def _reduce_local_context_rows(self, route_scores: Any) -> Any:
        """Retain local rows at the same route resolution as global rows.

        Native mode is a pure elementwise cast and is emitted directly
        (bitwise identical to the legacy float64 round-trip — see
        ``_native_float32_rows``); it was the measured batch-scaled OOM
        site at AlexNet-census scale. The reducing modes keep their
        declared float64 arithmetic unchanged.
        """

        rows = route_scores.local_per_image_scores
        if rows is None:
            raise RuntimeError(
                "local-attribution retention requires fused local rows"
            )
        block = route_scores.route_block
        work_device = rows.device
        if work_device.type == "mps":
            work_device = torch.device("cpu")
        if self.context_factor_mode == "native":
            retained = _native_float32_rows(
                rows,
                work_device=work_device,
                leading_shape=(int(rows.shape[0]),),
            )
            return (
                retained
                if self.context_row_transport == "device"
                else retained.cpu().numpy()
            )
        flat = rows.detach().to(
            device=work_device, dtype=torch.float64
        ).reshape(rows.shape[0], -1)
        if self.context_factor_mode == "channel_marginal":
            retained = (
                flat.reshape(flat.shape[0], int(block.shape[0]), -1)
                .sum(dim=-1)
                .to(torch.float32)
            )
        else:
            key = (f"local::{block.layer_id}", str(work_device))
            omega = self._sketch_cache.get(key)
            if omega is None:
                omega = torch.from_numpy(
                    context_sketch_matrix(
                        block.layer_id,
                        routes=int(flat.shape[-1]),
                        probes=self.sketch_probes,
                        seed=self.sketch_seed,
                    )
                ).to(device=work_device)
                self._sketch_cache[key] = omega
            retained = (flat @ omega).to(torch.float32)
        return (
            retained
            if self.context_row_transport == "device"
            else retained.cpu().numpy()
        )

    def _assemble_retained_rows(
        self,
        store: dict[str, list[Any]],
        layer_id: str,
    ) -> Any:
        """Join retained per-context row chunks without the concat doubling.

        ``torch.cat``/``np.concatenate`` allocate the joined result while
        every input chunk is still referenced by ``store``, so the peak is
        TWICE the retained rows.  At AlexNet fc6 census scale the retained
        rows alone are ``contexts x 37,748,736 x 4 B`` (0.1406 GiB per
        context), so that transient doubling is measured in hundreds of GiB
        and is invisible to ``IRG_SCORING_BATCH_ROWS`` -- the knob changes how
        many chunks there are, never their total.

        Copying into one preallocated destination and releasing each chunk as
        it is consumed holds the destination plus a single chunk instead.  The
        assembled buffer is written back as the store's only chunk, so a
        second read takes the single-chunk path and returns the SAME object:
        this is idempotent, not merely once-safe.  Only whole rows are copied
        -- no arithmetic -- so the result is bit-identical to the concatenate
        it replaces.
        """

        chunks = store.get(layer_id)
        if not chunks:
            return None
        if len(chunks) == 1:
            return chunks[0]
        first = chunks[0]
        total = sum(int(chunk.shape[0]) for chunk in chunks)
        tail = tuple(int(value) for value in first.shape[1:])
        if self.context_row_transport == "device":
            joined = torch.empty(
                (total, *tail), device=first.device, dtype=first.dtype
            )
        else:
            joined = np.empty((total, *tail), dtype=first.dtype)
        del first
        offset = 0
        while chunks:
            chunk = chunks.pop(0)
            span = int(chunk.shape[0])
            joined[offset : offset + span] = chunk
            offset += span
            del chunk
        if offset != total:
            raise RuntimeError(
                "retained row assembly covered "
                f"{offset} of {total} rows for layer {layer_id!r}"
            )
        store[layer_id] = [joined]
        return joined

    def tiles(self) -> tuple[MeasurementMomentTile, ...]:
        if self.context_count <= 0:
            raise ValueError("context moment accumulator received no contexts")
        missing = sorted(
            set(self._route_by_layer)
            - (set(self._score_sum) & set(self._absolute_score_sum))
        )
        if missing:
            raise ValueError("context moment accumulator omitted route blocks: " + ",".join(missing))
        tiles: list[MeasurementMomentTile] = []
        for block in self.route_blocks:
            with self.profiler.phase("measurement.context_block_flush"):
                source_score_sum = self._score_sum[block.layer_id].to(
                    device="cpu", dtype=torch.float64
                )
                source_absolute_score_sum = self._absolute_score_sum[block.layer_id].to(
                    device="cpu", dtype=torch.float64
                )
            aggregate = source_score_sum / float(self.context_count)
            absolute_aggregate = source_absolute_score_sum / float(self.context_count)
            # One declared block is the inferential sampling unit.  The route
            # score is the mean exact image contribution inside that block, so
            # changing runtime microbatching cannot change the measurement.
            tiles.append(
                MeasurementMomentTile.from_values(
                    aggregate.unsqueeze(0),
                    route_block=block,
                    query_ids=self.query_ids,
                    context_block_id=self.context_block_id,
                    estimator_id=self.estimator_id,
                    absolute_values=absolute_aggregate.unsqueeze(0),
                )
            )
            context_rows = None
            if self.retain_context_factor:
                # Single-chunk packaging is the common case (one capture
                # batch per packet at production batch sizes); torch.cat
                # of a one-element list is a pure copy that transiently
                # DOUBLES every large tile on device — the measured
                # tiles-time OOM at AlexNet census scale (a
                # 2.0 GiB ask at 77.7 GB in use). Returning the single
                # chunk is bitwise the same values with zero allocation.
                context_rows = self._assemble_retained_rows(
                    self._context_row_chunks, block.layer_id
                )
                if context_rows is not None and (
                    int(context_rows.shape[0]) != self.context_count
                ):
                    raise RuntimeError(
                        "context-factor rows disagree with the context "
                        f"count for layer {block.layer_id!r}: "
                        f"{int(context_rows.shape[0])} != {self.context_count}"
                    )
                # A None here means this block's route kind has no per-image
                # reducer; the factor legitimately omits it.
            local_context_rows = None
            if self.retain_local_attribution:
                local_context_rows = self._assemble_retained_rows(
                    self._local_context_row_chunks, block.layer_id
                )
                if local_context_rows is not None and (
                    int(local_context_rows.shape[0]) != self.context_count
                ):
                    raise RuntimeError(
                        "local-attribution rows disagree with the context "
                        f"count for layer {block.layer_id!r}"
                    )
            tiles[-1] = replace(
                tiles[-1],
                source_score_sum=source_score_sum,
                source_absolute_score_sum=source_absolute_score_sum,
                source_context_count=self.context_count,
                context_rows=context_rows,
                local_context_rows=local_context_rows,
                local_regime_statistics=self._local_regime_by_layer.get(
                    block.layer_id
                ),
            )
        return tuple(tiles)


class NativeLayerBlockedMeasurementAdapter:
    """Bind a model capture adapter to the generic streamed artifact writer."""

    def __init__(
        self,
        model_adapter: NativeMeasurementModelAdapter,
        *,
        query_tile_size: int = 8,
        autograd_backend: AutogradBackend = "auto",
        backend_registry: BackendRegistry | None = None,
        retain_context_factor: bool = False,
        retain_local_attribution: bool = False,
        retain_local_regime: bool = False,
        retain_context_observations: bool = False,
        context_row_transport: str = "host",
        context_factor_mode: str = "sketch",
        sketch_probes: int = 64,
        sketch_seed: int = 0,
        use_conserving_activation_attribution: bool = False,
        gate_disclosure_request_factory: (
            GateDisclosureRequestFactory | None
        ) = None,
    ) -> None:
        if query_tile_size <= 0:
            raise ValueError("query_tile_size must be positive")
        self.model_adapter = model_adapter
        self.query_tile_size = int(query_tile_size)
        self.autograd_backend = autograd_backend
        self.retain_context_factor = bool(retain_context_factor)
        self.retain_local_attribution = bool(retain_local_attribution)
        self.retain_local_regime = bool(retain_local_regime)
        self.retain_context_observations = bool(retain_context_observations)
        if str(context_row_transport) not in {"host", "device"}:
            raise ValueError("context-row transport must be 'host' or 'device'")
        self.context_row_transport = str(context_row_transport)
        self.context_factor_mode = str(context_factor_mode)
        self.sketch_probes = int(sketch_probes)
        self.sketch_seed = int(sketch_seed)
        self.use_conserving_activation_attribution = bool(
            use_conserving_activation_attribution
        )
        self.gate_disclosure_request_factory = (
            gate_disclosure_request_factory
        )
        if (
            self.gate_disclosure_request_factory is not None
            and not self.use_conserving_activation_attribution
        ):
            raise ValueError(
                "gate disclosure requires exact conserving attribution"
            )
        if self.use_conserving_activation_attribution and self.retain_local_regime:
            raise ValueError(
                "exact conserving attribution and legacy local-regime "
                "diagnostics cannot share one scoring pass"
            )
        device = getattr(model_adapter, "device", None)
        device_type = str(getattr(device, "type", "all"))
        self.device_type = device_type
        self.backend_registry = backend_registry or build_default_backend_registry(
            device_type=device_type
        )
        self.route_reducer_backend = self.backend_registry
        self.profiler = PhaseProfiler()
        self.adapter_id = f"native-target-batched::{model_adapter.adapter_id}"
        self.performance_telemetry: list[dict[str, Any]] = []
        self._context_observations: dict[
            str, tuple[tuple[str, ...], np.ndarray, np.ndarray]
        ] = {}

    def route_blocks(self, plan: MeasurementPlan) -> Sequence[RouteBlockSpec]:
        blocks = tuple(self.model_adapter.route_blocks(plan))
        if not blocks:
            raise ValueError("native measurement model adapter declared no route blocks")
        for block in blocks:
            block.validate()
            if block.architecture_id != self.model_adapter.architecture_id:
                raise ValueError("route block architecture differs from its model adapter")
            if block.topology_hash != self.model_adapter.topology_hash:
                raise ValueError("route block topology differs from its model adapter")
        if plan.route_schema_hash != self.model_adapter.route_schema_hash:
            raise ValueError("measurement plan route schema differs from its model adapter")
        return blocks

    def scientific_metadata(self) -> dict[str, Any]:
        metadata_factory = getattr(self.model_adapter, "scientific_metadata", None)
        model_metadata = dict(metadata_factory()) if callable(metadata_factory) else {}
        return {
            "adapter_id": self.adapter_id,
            "architecture_id": self.model_adapter.architecture_id,
            "topology_hash": self.model_adapter.topology_hash,
            "route_schema_hash": self.model_adapter.route_schema_hash,
            "model_adapter": model_metadata,
            "claim_capture_certified": model_metadata.get("claim_capture_certified") is True,
            "conserving_activation_attribution": (
                self.use_conserving_activation_attribution
            ),
            "gate_removal_disclosure_enabled": (
                self.gate_disclosure_request_factory is not None
            ),
        }

    def runtime_details(self) -> dict[str, Any]:
        model_runtime_factory = getattr(self.model_adapter, "runtime_details", None)
        resolved_reducers = {
            family: self.backend_registry.resolve(
                capability="route_reducer",
                route_family=family,
                device_type=self.device_type,
                dtype="float32",
            )
            for family in (
                "conv_channel_pair",
                "linear_weight",
                "residual_stream_gate",
            )
        }
        return {
            "model_adapter_id": self.model_adapter.adapter_id,
            "batch_provider_id": str(
                getattr(getattr(self.model_adapter, "batch_provider", None), "provider_id", "")
            ),
            "query_tile_size": self.query_tile_size,
            "autograd_backend_policy": self.autograd_backend,
            "conserving_activation_attribution": (
                self.use_conserving_activation_attribution
            ),
            "route_reducer_backends": {
                family: str(getattr(backend, "backend_id", ""))
                for family, backend in resolved_reducers.items()
            },
            "route_reducer_implementation_sources": {
                family: str(
                    getattr(backend, "implementation_sources", {}).get(
                        family, "undeclared"
                    )
                )
                for family, backend in resolved_reducers.items()
            },
            "profile": self.profiler.snapshot(),
            "model_adapter_runtime": (
                dict(model_runtime_factory()) if callable(model_runtime_factory) else {}
            ),
            "tiles": list(self.performance_telemetry),
        }

    def iter_moment_tiles(self, plan: MeasurementPlan) -> Iterator[MeasurementMomentTile]:
        plan.validate(claim_bearing=plan.status == "frozen").require_valid()
        blocks = tuple(self.route_blocks(plan))
        direction_resolver = self.model_adapter.direction_resolver()
        for context_block_id in plan.context_block_ids:
            observation_ids: list[str] = []
            observation_responses: list[np.ndarray] = []
            observation_labels: list[np.ndarray] = []
            query_tiles = tuple(
                plan.queries[start : start + self.query_tile_size]
                for start in range(0, len(plan.queries), self.query_tile_size)
            )
            accumulators = tuple(
                ContextBlockMomentAccumulator(
                    context_block_id=context_block_id,
                    query_ids=tuple(query.query_id for query in queries),
                    route_blocks=blocks,
                    estimator_id=plan.estimator_id,
                    reducer_backend=self.route_reducer_backend,
                    profiler=self.profiler,
                    retain_context_factor=self.retain_context_factor,
                    retain_local_attribution=self.retain_local_attribution,
                    # Regime diagnostics are query-independent. Tee them only
                    # into the first query tile so target blocking cannot
                    # duplicate their sufficient statistics.
                    retain_local_regime=(
                        self.retain_local_regime and tile_index == 0
                    ),
                    context_row_transport=self.context_row_transport,
                    context_factor_mode=self.context_factor_mode,
                    sketch_probes=self.sketch_probes,
                    sketch_seed=self.sketch_seed,
                    use_conserving_activation_attribution=(
                        self.use_conserving_activation_attribution
                    ),
                    gate_disclosure_request_factory=(
                        self.gate_disclosure_request_factory
                    ),
                )
                for tile_index, queries in enumerate(query_tiles)
            )
            batch_count = 0
            # Capture each image batch exactly once. Query blocking is an
            # execution/memory decision, so all tiles reuse the same forward
            # graph and differ only in their cotangent block. This removes the
            # architecture-specific repeated-forward behavior of the legacy
            # launchers while preserving the exact route-gate derivative.
            for batch in self.model_adapter.iter_captured_batches(
                plan,
                context_block_id=context_block_id,
                queries=plan.queries,
            ):
                if self.retain_context_observations:
                    if batch.labels is None:
                        raise ValueError(
                            "loss pullback retention requires context labels"
                        )
                    observation_ids.extend(batch.context_ids)
                    observation_responses.append(
                        batch.response_matrix.detach()
                        .to(device="cpu", dtype=torch.float32)
                        .numpy()
                    )
                    observation_labels.append(
                        torch.as_tensor(batch.labels)
                        .detach()
                        .to(device="cpu")
                        .reshape(-1)
                        .numpy()
                    )
                for tile_index, (queries, accumulator) in enumerate(
                    zip(query_tiles, accumulators, strict=True)
                ):
                    with self.profiler.phase("measurement.query_evaluation"):
                        query_batch = evaluate_queries(
                            batch.response_matrix,
                            queries,
                            labels=batch.labels,
                            direction_resolver=direction_resolver,
                        )
                    accumulator.add(
                        batch,
                        query_batch=query_batch,
                        backend=self.autograd_backend,
                        retain_graph=tile_index < len(query_tiles) - 1,
                    )
                batch_count += 1
            if batch_count <= 0:
                raise ValueError(
                    f"model adapter emitted no batches for context block {context_block_id!r}"
                )
            if self.retain_context_observations:
                if (
                    tuple(observation_ids)
                    and len(set(observation_ids)) == len(observation_ids)
                ):
                    self._context_observations[str(context_block_id)] = (
                        tuple(observation_ids),
                        np.concatenate(observation_responses, axis=0),
                        np.concatenate(observation_labels, axis=0),
                    )
                else:
                    raise RuntimeError(
                        "retained context observations have an invalid context axis"
                    )
            for accumulator in accumulators:
                self.performance_telemetry.append(
                    {
                        "context_block_id": context_block_id,
                        "query_ids": list(accumulator.query_ids),
                        "batch_count": batch_count,
                        "context_count": accumulator.context_count,
                        "vjp_backends": sorted(
                            {item.backend_used for item in accumulator.telemetry}
                        ),
                        "fallback_reasons": sorted(
                            {item.fallback_reason for item in accumulator.telemetry if item.fallback_reason}
                        ),
                    }
                )
                yield from accumulator.tiles()

    def capture_factored_rows(
        self,
        plan: MeasurementPlan,
        *,
        context_block_id: str,
        expected_context_ids: Sequence[str],
        route_ids: Sequence[str],
        route_gate: Any,
        destination_transport_hash: str,
        corrected_stream_identity_hash: str,
        literal_jacobian_stream_identity_hash: str,
        accumulator: Any,
    ) -> dict[str, Any]:
        """Capture one context packet as factored rows; form no route rows.

        Sibling of :meth:`stream_conserving_positive_removal_rows` restricted
        to the raw target-logit coordinate.  Exactly one captured
        batch, one query evaluation and one target-batched VJP run; the
        per-layer factors (input, post-activation cotangent, preactivation,
        bias baseline, weight) are handed to ``accumulator`` through
        ``begin_packet`` / ``consume_factored_packet`` / ``end_packet`` and no
        ``contexts x route`` tensor is ever formed.  The packet must arrive as
        exactly one captured batch in declared order; a batch factory that
        splits it refuses (typed absence, never a silent merge), because one
        factored packet is one VJP.

        The accumulator owns the factored packet type and its typed refusals
        (``capture_packet`` / ``refuse``); this measurement-layer method does
        not import the irg factored module (owner-dependency law).
        """

        _require_torch()
        plan.validate(claim_bearing=plan.status == "frozen").require_valid()
        if not self.use_conserving_activation_attribution:
            raise PermissionError(
                "factored capture requires exact conserving attribution"
            )
        if len(plan.queries) != 1:
            raise ValueError("factored capture requires exactly one response query")
        query = plan.queries[0]
        query_objective = str(
            query.evaluator.get("objective", query.evaluator.get("kind", ""))
        ).strip().lower()
        target = query.evaluator.get("class_id")
        if (
            query.query_kind != "class_logit"
            or query_objective != "class_logit"
            or isinstance(target, bool)
            or not isinstance(target, (int, np.integer))
            or int(target) < 0
        ):
            raise ValueError(
                "factored capture requires an exact pre-softmax class_logit query "
                "with an integer target class"
            )
        for name in (
            "packet_identities",
            "capture_packet",
            "refuse",
            "begin_packet",
            "consume_factored_packet",
            "end_packet",
            "route_streaming_receipt",
        ):
            if not callable(getattr(accumulator, name, None)):
                raise TypeError(
                    "factored accumulator lacks the reviewed packet protocol"
                )
        for label, value in (
            ("corrected stream", corrected_stream_identity_hash),
            ("literal-Jacobian stream", literal_jacobian_stream_identity_hash),
            ("destination transport", destination_transport_hash),
        ):
            if not str(value or ""):
                raise ValueError(f"factored capture {label} identity must be nonempty")
        if not str(context_block_id):
            raise ValueError("factored capture context block ID must be nonempty")
        contexts = tuple(str(value) for value in expected_context_ids)
        declared_route_ids = route_ids
        if (
            not contexts
            or len(set(contexts)) != len(contexts)
            or len(declared_route_ids) <= 0
        ):
            raise ValueError("factored capture axes are invalid")
        blocks = tuple(self.route_blocks(plan))
        route_count = sum(int(block.route_count) for block in blocks)
        if route_count != len(declared_route_ids):
            raise ValueError(
                "factored capture route IDs differ from the declared route blocks"
            )
        gate = torch.as_tensor(route_gate, dtype=torch.bool).reshape(-1)
        if int(gate.numel()) != route_count:
            raise ValueError("factored capture hard gate differs from the route axis")

        direction_resolver = self.model_adapter.direction_resolver()
        batches = iter(
            self.model_adapter.iter_captured_batches(
                plan,
                context_block_id=str(context_block_id),
                queries=plan.queries,
            )
        )
        try:
            batch = next(batches)
        except StopIteration:
            raise RuntimeError("factored capture produced no captured batch") from None
        batch.validate()
        if tuple(batch.context_ids) != contexts:
            accumulator.refuse(
                "AXIS_MISMATCH",
                "factored capture requires the packet to arrive as exactly one "
                "captured batch in declared order (a splitting batch factory is a "
                "typed absence, not a merge)",
            )
        with self.profiler.phase("measurement.query_evaluation"):
            query_batch = evaluate_queries(
                batch.response_matrix,
                plan.queries,
                labels=batch.labels,
                direction_resolver=direction_resolver,
            )
        if query_batch.statuses != ("class_logit_exact",):
            raise RuntimeError(
                "factored capture did not produce the exact unit cotangent"
            )
        captures = batch.conserving_activation_captures
        if captures is None:
            raise ValueError(
                "factored capture requires typed post-activation captures"
            )
        with self.profiler.phase("measurement.factored_capture_vjp"):
            packet = accumulator.capture_packet(
                batch.response_matrix,
                query_batch,
                batch.route_blocks,
                captures,
                backend=self.autograd_backend,
                retain_graph=False,
            )
        try:
            next(batches)
        except StopIteration:
            pass
        else:
            packet.evict()
            accumulator.refuse(
                "AXIS_MISMATCH",
                "factored capture packet was split across captured batches; one "
                "factored packet is one VJP",
            )
        receipt = packet.receipt
        summary = {
            "schema": "native_factored_capture_receipt_v1",
            "context_block_id": str(context_block_id),
            "context_count": len(contexts),
            "route_count": int(route_count),
            "vjp_count": 1,
            "vjp_backend_used": str(receipt.vjp_backend_used),
            "factored_packet_receipt_hash": receipt.receipt_hash,
            "factor_bytes": int(receipt.factor_bytes),
            "avoided_dense_row_bytes": int(receipt.avoided_dense_row_bytes),
            "corrected_stream_identity_hash": str(corrected_stream_identity_hash),
            "literal_jacobian_stream_identity_hash": str(
                literal_jacobian_stream_identity_hash
            ),
            "route_rows_formed": False,
            "route_scale_d2h_bytes": 0,
        }
        accumulator.begin_packet(
            contexts,
            route_ids=declared_route_ids,
            destination_transport_hash=str(destination_transport_hash),
            corrected_stream_identity_hash=str(corrected_stream_identity_hash),
            literal_j_stream_identity_hash=str(literal_jacobian_stream_identity_hash),
        )
        accumulator.consume_factored_packet(packet)
        accumulator.end_packet()
        summary.update(dict(accumulator.route_streaming_receipt()))
        return summary

    def stream_conserving_positive_removal_rows(
        self,
        plan: MeasurementPlan,
        *,
        context_block_id: str,
        expected_context_ids: Sequence[str],
        route_ids: Sequence[str],
        route_gate: Any,
        loss_pullback_factory: LossPullbackFactory | None,
        affine_scale: float,
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
        route_chunk_size: int,
        consumer: Any,
        stream_coordinate: str = "frozen_affine_bce_loss",
        stream_identity_hash: str | None = None,
        literal_jacobian_consumer: Any | None = None,
        literal_jacobian_stream_identity_hash: str | None = None,
        gate_disclosure_request_factory: (
            GateDisclosureRequestFactory | None
        ) = None,
        loss_path_factory: LossPathFactory | None = None,
    ) -> dict[str, Any]:
        """Stream exact ``-G*phi*c`` rows without generic moment tiles.

        This is the narrow production data plane for a nominal one-head
        consumer.  Each captured context batch executes exactly one query
        evaluation and one target-batched conserving VJP.  Two explicitly
        identified covector coordinates are supported:

        * ``frozen_affine_bce_loss`` applies the positive affine derivative,
          the caller's context-specific BCE pullback, the hard keep gate, and
          the removal sign. A typed ``loss_path_factory`` instead integrates
          that loss exactly on every registered unit activation path before
          contraction, with one dense VJP and no endpoint multiplication;
        * ``raw_target_logit`` applies only the hard keep gate and removal
          sign.  Its pullback is exactly one and calibration slope/offset are
          deliberately absent.

        The resulting per-context route rows remain on their native device,
        are converted to float64 only at the declared route-chunk width, and
        are sent to ``consumer.consume_positive_removal_chunk``.  In raw-logit
        mode an optional second nominal consumer receives the literal
        current-chart gate Jacobian ``-G*1[a>0]*c`` from the same capture and
        VJP.  The corrected Aumann--Shapley rows and literal Jacobian rows are
        deliberately distinct semantic streams.

        No :class:`ContextBlockMomentAccumulator`, local-attribution row,
        generic aggregate float64 signed/absolute array, or
        :class:`MeasurementMomentTile` participates in this path.  An explicit
        gate-removal diagnostic may be supplied transactionally; it shares the
        same target-logit VJP while remaining a separate, non-ordering sidecar.
        """

        _require_torch()
        plan.validate(claim_bearing=plan.status == "frozen").require_valid()
        if not self.use_conserving_activation_attribution:
            raise PermissionError(
                "positive-removal streaming requires exact conserving attribution"
            )
        if len(plan.queries) != 1:
            raise ValueError(
                "positive-removal streaming requires exactly one response query"
            )
        query = plan.queries[0]
        query_objective = str(
            query.evaluator.get("objective", query.evaluator.get("kind", ""))
        ).strip().lower()
        coordinate = str(stream_coordinate)
        if coordinate not in {
            "frozen_affine_bce_loss",
            "raw_target_logit",
        }:
            raise ValueError("unsupported positive-removal stream coordinate")
        required_consumer_methods = (
            "begin_packet",
            "consume_positive_removal_chunk",
            "end_packet",
        )
        if any(
            not callable(getattr(consumer, name, None))
            for name in required_consumer_methods
        ):
            raise TypeError(
                "positive-removal stream consumer lacks the reviewed packet protocol"
            )
        dual_chart_streaming = literal_jacobian_consumer is not None
        if dual_chart_streaming:
            if literal_jacobian_consumer is consumer:
                raise ValueError(
                    "corrected evidence and literal Jacobian require distinct consumers"
                )
            if any(
                not callable(getattr(literal_jacobian_consumer, name, None))
                for name in required_consumer_methods
            ):
                raise TypeError(
                    "literal-Jacobian consumer lacks the reviewed packet protocol"
                )
            if coordinate != "raw_target_logit":
                raise ValueError(
                    "literal gate-Jacobian streaming is defined only in raw-logit coordinates"
                )
            if not str(literal_jacobian_stream_identity_hash or ""):
                raise ValueError(
                    "literal gate-Jacobian stream identity must be nonempty"
                )
        elif literal_jacobian_stream_identity_hash is not None:
            raise ValueError(
                "literal gate-Jacobian identity requires its nominal consumer"
            )
        scale = float(affine_scale)
        chunk_size = int(route_chunk_size)
        if coordinate == "frozen_affine_bce_loss":
            if (loss_pullback_factory is None) == (loss_path_factory is None):
                raise TypeError("loss streaming requires exactly one endpoint or integrated-path factory")
            if not callable(loss_path_factory if loss_path_factory is not None else loss_pullback_factory):
                raise TypeError("loss streaming factory must be callable")
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    "positive-removal affine scale must be finite and positive"
                )
            if not str(loss_pullback_hash or ""):
                raise ValueError("BCE positive-removal identity must be nonempty")
            if (
                stream_identity_hash is not None
                and str(stream_identity_hash) != str(loss_pullback_hash)
            ):
                raise ValueError(
                    "BCE stream and loss-pullback identities must agree"
                )
            effective_stream_identity = str(loss_pullback_hash)
        else:
            target = query.evaluator.get("class_id")
            if (
                query.query_kind != "class_logit"
                or query_objective != "class_logit"
                or isinstance(target, bool)
                or not isinstance(target, (int, np.integer))
                or int(target) < 0
            ):
                raise ValueError(
                    "raw target-logit streaming requires an exact pre-softmax "
                    "class_logit query with an integer target class"
                )
            if loss_pullback_factory is not None or loss_pullback_hash is not None or loss_path_factory is not None:
                raise ValueError(
                    "raw target-logit streaming forbids a BCE loss pullback"
                )
            if not np.isfinite(scale) or scale != 1.0:
                raise ValueError(
                    "raw target-logit streaming requires exact unit response scale"
                )
            if not str(stream_identity_hash or ""):
                raise ValueError(
                    "raw target-logit stream identity must be nonempty"
                )
            effective_stream_identity = str(stream_identity_hash)
        if isinstance(route_chunk_size, bool) or chunk_size <= 0:
            raise ValueError("positive-removal route chunk size must be positive")
        if not str(context_block_id):
            raise ValueError("positive-removal context block ID must be nonempty")
        contexts = tuple(str(value) for value in expected_context_ids)
        # Preserve the provider's canonical/lazy whole-axis object by
        # identity.  Materializing or string-copying a 57M-route axis here
        # would dominate the score pass and defeat this data plane.
        declared_route_ids = route_ids
        if (
            not contexts
            or len(set(contexts)) != len(contexts)
            or len(declared_route_ids) <= 0
            or not effective_stream_identity
        ):
            raise ValueError("positive-removal stream axes or identity are invalid")

        blocks = tuple(self.route_blocks(plan))
        route_count = sum(int(block.route_count) for block in blocks)
        if route_count != len(declared_route_ids):
            raise ValueError(
                "positive-removal route IDs differ from the declared route blocks"
            )
        gate = torch.as_tensor(route_gate, dtype=torch.bool).reshape(-1)
        if int(gate.numel()) != route_count:
            raise ValueError("positive-removal hard gate differs from the route axis")

        direction_resolver = self.model_adapter.direction_resolver()
        seen_contexts: list[str] = []
        batch_count = 0
        vjp_count = 0
        chunk_count = 0
        vjp_backends: set[str] = set()
        fallback_reasons: set[str] = set()
        mismatch_entry_count = 0
        mismatch_nonzero: Any | None = None
        mismatch_abs_sum: Any | None = None
        mismatch_square_sum: Any | None = None
        mismatch_signed_sum: Any | None = None
        mismatch_linf: Any | None = None
        for batch in self.model_adapter.iter_captured_batches(
            plan,
            context_block_id=str(context_block_id),
            queries=plan.queries,
        ):
            batch.validate()
            batch_contexts = tuple(batch.context_ids)
            expected_batch_contexts = contexts[
                len(seen_contexts) : len(seen_contexts) + len(batch_contexts)
            ]
            if batch_contexts != expected_batch_contexts:
                raise RuntimeError(
                    "positive-removal capture changed the declared context order"
                )
            with self.profiler.phase("measurement.query_evaluation"):
                query_batch = evaluate_queries(
                    batch.response_matrix,
                    plan.queries,
                    labels=batch.labels,
                    direction_resolver=direction_resolver,
                )
                if (
                    coordinate == "raw_target_logit"
                    and query_batch.statuses != ("class_logit_exact",)
                ):
                    raise RuntimeError(
                        "raw target-logit stream did not produce the exact unit cotangent"
                    )
            with self.profiler.phase("measurement.vjp_and_route_reduce"):
                captures = batch.conserving_activation_captures
                if captures is None:
                    raise ValueError(
                        "positive-removal streaming requires typed post-activation captures"
                    )
                gate_disclosure_request = (
                    None
                    if gate_disclosure_request_factory is None
                    else gate_disclosure_request_factory(batch, query_batch)
                )
                loss_path = (
                    None if loss_path_factory is None
                    else loss_path_factory(batch, query_batch)
                )
                if loss_path_factory is not None and (
                    not isinstance(loss_path, FrozenPruningLossPath)
                    or loss_path.affine_scale != scale
                ):
                    raise ValueError("integrated loss streaming requires the matching typed frozen affine path")
                result = score_captured_conserving_route_blocks(
                    batch.response_matrix,
                    query_batch,
                    batch.route_blocks,
                    captures,
                    backend=self.autograd_backend,
                    retain_graph=False,
                    reducer_backend=self.route_reducer_backend,
                    profiler=self.profiler,
                    per_image=True,
                    local_attribution=False,
                    gate_disclosure=gate_disclosure_request,
                    include_literal_jacobian=dual_chart_streaming,
                    cached_vjp=batch.census_activation_vjp,
                    loss_path=loss_path,
                )
            pullback = None
            if coordinate == "frozen_affine_bce_loss" and loss_path_factory is None:
                assert loss_pullback_factory is not None
                pullback = loss_pullback_factory(batch, query_batch)
                if (
                    not isinstance(pullback, torch.Tensor)
                    or tuple(pullback.shape) != (len(batch_contexts),)
                    or pullback.dtype != torch.float64
                    or pullback.device != batch.response_matrix.device
                    or not bool(torch.isfinite(pullback).all().item())
                ):
                    raise ValueError(
                        "positive-removal loss pullback must be finite float64 "
                        "on the capture device"
                    )

            consumer.begin_packet(
                batch_contexts,
                route_ids=declared_route_ids,
                destination_transport_hash=destination_transport_hash,
                # ``loss_pullback_hash`` is the historical nominal protocol's
                # opaque one-head identity slot.  In raw-logit mode it carries
                # the distinct unit-pullback stream identity, never a BCE hash.
                loss_pullback_hash=effective_stream_identity,
            )
            if literal_jacobian_consumer is not None:
                literal_jacobian_consumer.begin_packet(
                    batch_contexts,
                    route_ids=declared_route_ids,
                    destination_transport_hash=destination_transport_hash,
                    loss_pullback_hash=str(
                        literal_jacobian_stream_identity_hash
                    ),
                )
                literal_scores = result.literal_jacobian_scores
                if literal_scores is None:
                    raise RuntimeError(
                        "dual-chart scorer omitted literal gate-Jacobian rows"
                    )
            else:
                literal_scores = None
            global_offset = 0
            jacobian_blocks = (
                literal_scores
                if literal_scores is not None
                else (None,) * len(blocks)
            )
            mismatch_entry_count_by_block = [0] * len(blocks)
            for block_index, (block, route_scores, jacobian_scores) in enumerate(
                zip(
                    blocks,
                    result.scores,
                    jacobian_blocks,
                    strict=True,
                )
            ):
                if route_scores.route_block.to_dict() != block.to_dict():
                    raise RuntimeError(
                        "positive-removal scorer changed the declared route-block order"
                    )
                per_image = route_scores.per_image_scores
                if per_image is None:
                    raise RuntimeError(
                        "positive-removal scorer omitted per-context route rows for "
                        f"{block.layer_id!r}"
                    )
                rows = per_image.detach().reshape(
                    len(batch_contexts),
                    len(plan.queries),
                    int(block.route_count),
                )
                if tuple(rows.shape) != (
                    len(batch_contexts),
                    1,
                    int(block.route_count),
                ):
                    raise RuntimeError(
                        "positive-removal route rows have an incompatible axis"
                    )
                jacobian_rows = None
                if jacobian_scores is not None:
                    if (
                        jacobian_scores.route_block.to_dict()
                        != block.to_dict()
                    ):
                        raise RuntimeError(
                            "literal-Jacobian scorer changed the route-block order"
                        )
                    jacobian_per_image = jacobian_scores.per_image_scores
                    if jacobian_per_image is None:
                        raise RuntimeError(
                            "literal-Jacobian scorer omitted per-context route rows"
                        )
                    jacobian_rows = jacobian_per_image.detach().reshape(
                        len(batch_contexts),
                        len(plan.queries),
                        int(block.route_count),
                    )
                    if tuple(jacobian_rows.shape) != tuple(rows.shape):
                        raise RuntimeError(
                            "corrected and literal-Jacobian route rows are misaligned"
                        )
                for local_start in range(0, int(block.route_count), chunk_size):
                    local_stop = min(
                        local_start + chunk_size,
                        int(block.route_count),
                    )
                    start = global_offset + local_start
                    stop = global_offset + local_stop
                    # Preserve the established BCE-head elementwise order when
                    # requested.  Raw target-logit mode is exactly the unit
                    # pullback: no calibration slope, offset, label, sigmoid,
                    # or other context multiplier participates.  Only this
                    # bounded chunk is promoted to f64.
                    positive_removal = rows[
                        :, 0, local_start:local_stop
                    ].to(dtype=torch.float64).contiguous()
                    if pullback is not None:
                        positive_removal.mul_(scale)
                        positive_removal.mul_(pullback[:, None])
                    gate_chunk = gate[start:stop].to(
                        device=positive_removal.device,
                        dtype=torch.float64,
                    )
                    positive_removal.mul_(gate_chunk[None, :])
                    positive_removal.neg_()
                    if jacobian_rows is not None:
                        literal_positive_removal = jacobian_rows[
                            :, 0, local_start:local_stop
                        ].to(dtype=torch.float64).contiguous()
                        literal_positive_removal.mul_(gate_chunk[None, :])
                        literal_positive_removal.neg_()
                        mismatch = positive_removal - literal_positive_removal
                        mismatch_absolute = torch.abs(mismatch)
                        # Per-route-block device accumulators: the e-J mismatch
                        # is a regime map, so it is kept per declared block
                        # (one bounded D2H of 5 x blocks scalars at the end).
                        if mismatch_nonzero is None:
                            mismatch_nonzero = torch.zeros(
                                (len(blocks),), dtype=torch.int64, device=mismatch.device
                            )
                            mismatch_abs_sum = torch.zeros(
                                (len(blocks),), dtype=torch.float64, device=mismatch.device
                            )
                            mismatch_square_sum = torch.zeros_like(
                                mismatch_abs_sum
                            )
                            mismatch_signed_sum = torch.zeros_like(
                                mismatch_abs_sum
                            )
                            mismatch_linf = torch.zeros_like(mismatch_abs_sum)
                        mismatch_nonzero[block_index] += torch.count_nonzero(mismatch)
                        assert mismatch_abs_sum is not None
                        assert mismatch_square_sum is not None
                        assert mismatch_signed_sum is not None
                        assert mismatch_linf is not None
                        mismatch_abs_sum[block_index] += torch.sum(mismatch_absolute)
                        mismatch_square_sum[block_index] += torch.sum(
                            mismatch * mismatch
                        )
                        mismatch_signed_sum[block_index] += torch.sum(mismatch)
                        mismatch_linf[block_index] = torch.maximum(
                            mismatch_linf[block_index],
                            torch.max(mismatch_absolute),
                        )
                        mismatch_entry_count += int(mismatch.numel())
                        mismatch_entry_count_by_block[block_index] += int(
                            mismatch.numel()
                        )
                        consumer.consume_positive_removal_chunk(
                            start,
                            stop,
                            positive_removal,
                        )
                        assert literal_jacobian_consumer is not None
                        literal_jacobian_consumer.consume_positive_removal_chunk(
                            start,
                            stop,
                            literal_positive_removal,
                        )
                    else:
                        consumer.consume_positive_removal_chunk(
                            start,
                            stop,
                            positive_removal,
                        )
                    chunk_count += 1
                global_offset += int(block.route_count)
            if global_offset != route_count:
                raise RuntimeError(
                    "positive-removal stream did not close the route axis"
                )
            consumer.end_packet()
            if literal_jacobian_consumer is not None:
                literal_jacobian_consumer.end_packet()
            seen_contexts.extend(batch_contexts)
            batch_count += 1
            vjp_count += 1
            vjp_backends.add(str(result.telemetry.backend_used))
            if result.telemetry.fallback_reason:
                fallback_reasons.add(str(result.telemetry.fallback_reason))

        if tuple(seen_contexts) != contexts or batch_count <= 0:
            raise RuntimeError(
                "positive-removal capture did not cover the context packet exactly"
            )
        if dual_chart_streaming:
            if (
                mismatch_nonzero is None
                or mismatch_abs_sum is None
                or mismatch_square_sum is None
                or mismatch_signed_sum is None
                or mismatch_linf is None
                or mismatch_entry_count <= 0
            ):
                raise RuntimeError(
                    "dual-chart acquisition omitted its mismatch diagnostic"
                )
            # One bounded D2H for every block statistic; totals are derived
            # on the host from the per-block values in declared block order.
            block_stack = (
                torch.stack(
                    (
                        mismatch_nonzero.to(dtype=torch.float64),
                        mismatch_abs_sum,
                        mismatch_square_sum,
                        mismatch_signed_sum,
                        mismatch_linf,
                    )
                )
                .detach()
                .cpu()
                .tolist()
            )
            nonzero_by_block = [int(round(value)) for value in block_stack[0]]
            abs_by_block = [float(value) for value in block_stack[1]]
            square_by_block = [float(value) for value in block_stack[2]]
            signed_by_block = [float(value) for value in block_stack[3]]
            linf_by_block = [float(value) for value in block_stack[4]]
            mismatch_nonzero_count = int(sum(nonzero_by_block))
            mismatch_abs_total = float(math.fsum(abs_by_block))
            mismatch_square_total = float(math.fsum(square_by_block))
            mismatch_signed_total = float(math.fsum(signed_by_block))
            mismatch_linf_value = float(max(linf_by_block)) if linf_by_block else 0.0
            mismatch_by_route_block = tuple(
                {
                    "layer_id": block.layer_id,
                    "route_count": int(block.route_count),
                    "entry_count": int(entries),
                    "nonzero_count": int(nonzero),
                    "mean_absolute": (
                        float(abs_sum / entries) if entries else 0.0
                    ),
                    "root_mean_square": (
                        float(math.sqrt(square_sum / entries)) if entries else 0.0
                    ),
                    "linf": float(linf),
                    "signed_mean": (
                        float(signed_sum / entries) if entries else 0.0
                    ),
                }
                for block, entries, nonzero, abs_sum, square_sum, signed_sum, linf in zip(
                    blocks,
                    mismatch_entry_count_by_block,
                    nonzero_by_block,
                    abs_by_block,
                    square_by_block,
                    signed_by_block,
                    linf_by_block,
                    strict=True,
                )
            )
            if any(
                not all(
                    math.isfinite(item[name])
                    for name in ("mean_absolute", "root_mean_square", "linf", "signed_mean")
                )
                for item in mismatch_by_route_block
            ):
                raise FloatingPointError(
                    "dual-chart per-block mismatch diagnostic is not finite"
                )
            mismatch_mean_absolute = mismatch_abs_total / mismatch_entry_count
            mismatch_rms = math.sqrt(mismatch_square_total / mismatch_entry_count)
            mismatch_signed_mean = mismatch_signed_total / mismatch_entry_count
            if (
                mismatch_nonzero_count < 0
                or mismatch_nonzero_count > mismatch_entry_count
                or not all(
                    math.isfinite(value)
                    for value in (
                        mismatch_mean_absolute,
                        mismatch_rms,
                        mismatch_linf_value,
                        mismatch_signed_mean,
                    )
                )
                or min(
                    mismatch_mean_absolute,
                    mismatch_rms,
                    mismatch_linf_value,
                )
                < 0.0
            ):
                raise FloatingPointError(
                    "dual-chart mismatch diagnostic is not finite"
                )
        else:
            mismatch_nonzero_count = 0
            mismatch_mean_absolute = 0.0
            mismatch_rms = 0.0
            mismatch_linf_value = 0.0
            mismatch_signed_mean = 0.0
            mismatch_by_route_block = ()
        receipt = {
            "schema": (
                "exact_dual_chart_positive_removal_stream_v1"
                if dual_chart_streaming
                else "exact_conserving_loss_activation_path_stream_v3"
                if loss_path_factory is not None
                else "exact_conserving_positive_removal_stream_v2"
            ),
            "context_block_id": str(context_block_id),
            "context_count": len(contexts),
            "capture_batch_count": int(batch_count),
            "vjp_count": int(vjp_count),
            "route_count": int(route_count),
            "route_chunk_count": int(chunk_count),
            "route_chunk_size": int(chunk_size),
            "vjp_backends": tuple(sorted(vjp_backends)),
            "fallback_reasons": tuple(sorted(fallback_reasons)),
            "positive_removal_orientation": True,
            "exact_conserving_attribution": True,
            "stream_coordinate": coordinate,
            "stream_identity_hash": effective_stream_identity,
            "cotangent_coordinate": (
                "raw_target_logit_unit_cotangent"
                if coordinate == "raw_target_logit"
                else "integrated_dense_pruning_loss_unit_activation_path"
                if loss_path_factory is not None
                else "frozen_affine_bce_loss_cotangent"
            ),
            "raw_target_logit_unit_pullback": (
                coordinate == "raw_target_logit"
            ),
            "affine_calibration_applied": (
                coordinate == "frozen_affine_bce_loss"
            ),
            "context_specific_loss_pullback_applied": (
                coordinate == "frozen_affine_bce_loss"
            ),
            "loss_path_integrated": loss_path_factory is not None,
            **({} if loss_path_factory is None else {
                "loss_path_scope": "unitwise_bias_to_activation_frozen_target_cotangent_v1",
                "loss_path_integration": "exact_softplus_divided_difference_v1",
                "loss_probability_frozen_along_path": False,
                "dense_competitors_frozen_along_path": True,
            }),
            "calibration_offset_applied": loss_path_factory is not None,
            "local_attribution_rows_requested": False,
            "gate_removal_disclosure_requested": (
                gate_disclosure_request_factory is not None
            ),
            "generic_context_block_moment_accumulator_constructed": False,
            "generic_aggregate_float64_route_arrays_constructed": False,
            "generic_measurement_moment_tiles_materialized": False,
            "scorer_adapter_host_context_route_rows_transferred": False,
            "dual_corrected_jacobian_streaming": bool(dual_chart_streaming),
            "same_capture_and_vjp": bool(dual_chart_streaming),
            "corrected_evidence_stream_role": "aumann_shapley_work_F",
            "corrected_rows_excluded_from_response_geometry": bool(
                dual_chart_streaming
            ),
            "literal_jacobian_stream_role": (
                "current_chart_response_geometry_Q"
                if dual_chart_streaming
                else None
            ),
            "literal_jacobian_stream_identity_hash": (
                str(literal_jacobian_stream_identity_hash)
                if dual_chart_streaming
                else None
            ),
            "literal_jacobian_route_score_id": (
                LITERAL_GATE_JACOBIAN_ROUTE_SCORE_ID
                if dual_chart_streaming
                else None
            ),
            "literal_jacobian_equation_id": (
                LITERAL_GATE_JACOBIAN_EQUATION_ID
                if dual_chart_streaming
                else None
            ),
            "relu_zero_subgradient_policy_id": (
                RELU_ZERO_SUBGRADIENT_POLICY_ID
                if dual_chart_streaming
                else None
            ),
            "corrected_minus_literal_mismatch_entry_count": (
                mismatch_entry_count if dual_chart_streaming else 0
            ),
            "corrected_minus_literal_mismatch_nonzero_count": (
                mismatch_nonzero_count
            ),
            "corrected_minus_literal_mismatch_mean_absolute": (
                mismatch_mean_absolute
            ),
            "corrected_minus_literal_mismatch_root_mean_square": mismatch_rms,
            "corrected_minus_literal_mismatch_linf": mismatch_linf_value,
            "corrected_minus_literal_mismatch_signed_mean": (
                mismatch_signed_mean
            ),
            "corrected_minus_literal_mismatch_route_block_count": (
                len(mismatch_by_route_block)
            ),
            "corrected_minus_literal_mismatch_by_route_block": (
                mismatch_by_route_block
            ),
            "corrected_minus_literal_mismatch_diagnostic_only": True,
        }
        self.performance_telemetry.append(dict(receipt))
        return receipt

    def context_observations(
        self,
        context_block_id: str,
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
        """Return the latest ephemeral responses/labels for a measured block."""

        try:
            identifiers, responses, labels = self._context_observations[
                str(context_block_id)
            ]
        except KeyError as error:
            raise KeyError(
                "no retained context observations for the requested block"
            ) from error
        return identifiers, responses.copy(), labels.copy()


__all__ = [
    "CapturedMeasurementBatch",
    "ContextBlockMomentAccumulator",
    "NativeLayerBlockedMeasurementAdapter",
    "NativeMeasurementModelAdapter",
]
