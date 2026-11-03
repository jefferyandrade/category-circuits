"""Dense-mask FC flip crossings and destination pre-activation margins.

This module is an observation-only companion to the IRG dense geometry
census.  It consumes the raw, untransported local-destination contribution
rows already materialized by the canonical route reducer.  It does not own
route contraction, masking, statistical decisions, or stopping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

try:  # Optional for import smoke; required by the production CUDA path.
    import torch
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]


MARGIN_PERCENTILE_LEVELS = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)
MARGIN_NOT_CAPTURED = "not_captured"
MARGIN_NOT_APPLICABLE = "not_applicable"
_DEFAULT_ROUTE_CHUNK = 2_097_152


@dataclass
class _LinearFlipMarginState:
    shape: tuple[int, int]
    flip_count_deactivate: np.ndarray
    flip_count_activate: np.ndarray
    margin_chunks: list[np.ndarray] = field(default_factory=list)
    context_count: int = 0


SOURCE_BLOCK_ELEMENT_TARGET = 1 << 24
"""Target elements per streamed (contexts x destinations x source-block) view.

Bounds the boolean/compare temporaries in the FC crossing test. Counts are
invariant to this value -- it is a memory knob only, asserted by the
block-invariance test.
"""


def _source_block_width(
    source_count: int, context_count: int, destination_count: int
) -> int:
    """Return a source-block width bounding the streamed view's element count.

    The streamed view is ``(context_count, destination_count, block)``, so the
    block must divide the element target by the leading extent -- bounding by
    ``source_count`` alone would be a no-op and defeat the purpose.
    """

    if source_count <= 0 or context_count <= 0 or destination_count <= 0:
        raise ValueError("flip-margin block extents must be positive")
    leading = int(context_count) * int(destination_count)
    allowed = int(SOURCE_BLOCK_ELEMENT_TARGET) // max(1, leading)
    return max(1, min(int(source_count), allowed))


class DenseFlipMarginAccumulator:
    """Accumulate exact FC crossing counts and pooled destination margins.

    Route chunks are rounded down to whole destination groups so the C-order
    ``(destination, source)`` axis can be reshaped without gather/scatter.
    The only persistent route-scale state is two int32 counter arrays.
    """

    def __init__(self) -> None:
        self._linear: dict[str, _LinearFlipMarginState] = {}
        self._margin_status: dict[str, str] = {}
        self._conv_layers: set[str] = set()

    def mark_not_captured(self, *, layer_id: str, route_kind: str) -> None:
        layer = str(layer_id)
        kind = str(route_kind)
        if kind in {"conv_channel_pair", "conv_kernel_weight"}:
            self._conv_layers.add(layer)
            self._margin_status[layer] = MARGIN_NOT_CAPTURED
        elif kind != "linear_weight":
            self._margin_status.setdefault(layer, MARGIN_NOT_APPLICABLE)

    def update_linear(
        self,
        *,
        layer_id: str,
        local_context_rows: Any,
        bias: Any,
        route_shape: tuple[int, ...],
        flattening: str,
        route_chunk_size: int | None = None,
    ) -> None:
        """Consume one context packet of raw ``x_i * w_ji`` rows."""

        layer = str(layer_id)
        shape = tuple(int(value) for value in route_shape)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("FC flip-margin geometry requires a 2-D route shape")
        if str(flattening) != "c_order_target_then_source":
            raise ValueError("FC flip-margin geometry requires C-order target/source routes")
        target_count, source_count = shape
        rows_shape = tuple(int(value) for value in local_context_rows.shape)
        if len(rows_shape) != 2 or rows_shape[1] != target_count * source_count:
            raise ValueError("FC flip-margin rows differ from the declared route axis")
        context_count = rows_shape[0]
        if context_count <= 0:
            raise ValueError("FC flip-margin geometry requires at least one context")

        state = self._linear.get(layer)
        if state is None:
            state = _LinearFlipMarginState(
                shape=(target_count, source_count),
                flip_count_deactivate=np.zeros(shape, dtype=np.int32),
                flip_count_activate=np.zeros(shape, dtype=np.int32),
            )
            self._linear[layer] = state
        elif state.shape != shape:
            raise ValueError("FC flip-margin route shape changed within the census")
        self._margin_status[layer] = "captured"

        chunk_routes = int(route_chunk_size or _DEFAULT_ROUTE_CHUNK)
        if chunk_routes <= 0:
            raise ValueError("FC flip-margin route chunk size must be positive")
        destinations_per_chunk = max(1, chunk_routes // source_count)
        for destination_start in range(0, target_count, destinations_per_chunk):
            destination_stop = min(
                destination_start + destinations_per_chunk, target_count
            )
            route_start = destination_start * source_count
            route_stop = destination_stop * source_count
            self._consume_destination_chunk(
                state=state,
                local_context_rows=local_context_rows,
                bias=bias,
                destination_start=destination_start,
                destination_stop=destination_stop,
                route_start=route_start,
                route_stop=route_stop,
            )
        state.context_count += context_count

    @staticmethod
    def _checked_add_counts(
        target: np.ndarray,
        addition: np.ndarray,
        destination_start: int,
        destination_stop: int,
    ) -> None:
        current = target[destination_start:destination_stop].astype(
            np.int64, copy=False
        )
        summed = current + np.asarray(addition, dtype=np.int64)
        if np.any(summed > np.iinfo(np.int32).max):
            raise OverflowError("FC flip count exceeds the int32 receipt contract")
        target[destination_start:destination_stop] = summed.astype(
            np.int32, copy=False
        )

    def _consume_destination_chunk(
        self,
        *,
        state: _LinearFlipMarginState,
        local_context_rows: Any,
        bias: Any,
        destination_start: int,
        destination_stop: int,
        route_start: int,
        route_stop: int,
    ) -> None:
        source_count = state.shape[1]
        destination_count = destination_stop - destination_start
        if torch is not None and torch.is_tensor(local_context_rows):
            contribution = local_context_rows[:, route_start:route_stop].reshape(
                int(local_context_rows.shape[0]), destination_count, source_count
            ).to(dtype=torch.float64)
            bias_chunk = torch.as_tensor(
                bias,
                device=contribution.device,
                dtype=torch.float64,
            ).reshape(-1)[destination_start:destination_stop]
            if int(bias_chunk.numel()) != destination_count:
                raise ValueError("FC bias differs from its destination axis")
            preactivation = contribution.sum(dim=2) + bias_chunk[None, :]
            # Crossing tests WITHOUT materializing (preactivation - contribution).
            # Exact, not approximate: for finite IEEE-754 doubles, a - b is
            # exactly 0 iff a == b and otherwise carries the sign of the true
            # difference, so rounding can never cross zero. Hence
            #   preact - contribution <= 0  <=>  contribution >= preact
            #   preact - contribution >  0  <=>  contribution <  preact
            # The subtraction allocated a second full
            # (contexts x destinations x sources) float64 array purely to take a
            # sign; comparing against the broadcast preactivation gives
            # bit-identical counts. Sources are streamed in blocks so the
            # boolean temporaries are never source-wide either: peak transient
            # falls from O(contexts x destinations x sources) to
            # O(contexts x destinations x block).
            positive = (preactivation > 0.0)[:, :, None]
            nonpositive = ~positive
            deactivate = torch.zeros(
                (destination_count, source_count),
                dtype=torch.int64,
                device=contribution.device,
            )
            activate = torch.zeros_like(deactivate)
            block = _source_block_width(
                source_count, int(contribution.shape[0]), destination_count
            )
            for begin in range(0, source_count, block):
                end = min(begin + block, source_count)
                window = contribution[:, :, begin:end]
                reference = preactivation[:, :, None]
                deactivate[:, begin:end] = (
                    positive & (window >= reference)
                ).sum(dim=0, dtype=torch.int64)
                activate[:, begin:end] = (
                    nonpositive & (window < reference)
                ).sum(dim=0, dtype=torch.int64)
            deactivate_host = deactivate.detach().cpu().numpy()
            activate_host = activate.detach().cpu().numpy()
            margin_host = (
                preactivation.detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
                .reshape(-1)
            )
        else:
            contribution = np.asarray(
                local_context_rows[:, route_start:route_stop], dtype=np.float64
            ).reshape(-1, destination_count, source_count)
            bias_chunk = np.asarray(bias, dtype=np.float64).reshape(-1)[
                destination_start:destination_stop
            ]
            if bias_chunk.size != destination_count:
                raise ValueError("FC bias differs from its destination axis")
            preactivation = contribution.sum(axis=2) + bias_chunk[None, :]
            # Same exact-comparison identity and source blocking as the torch
            # branch above; see that comment for the IEEE-754 argument.
            positive = (preactivation > 0.0)[:, :, None]
            nonpositive = ~positive
            deactivate_host = np.zeros(
                (destination_count, source_count), dtype=np.int64
            )
            activate_host = np.zeros_like(deactivate_host)
            block = _source_block_width(
                source_count, int(contribution.shape[0]), destination_count
            )
            for begin in range(0, source_count, block):
                end = min(begin + block, source_count)
                window = contribution[:, :, begin:end]
                reference = preactivation[:, :, None]
                deactivate_host[:, begin:end] = np.sum(
                    positive & (window >= reference), axis=0, dtype=np.int64
                )
                activate_host[:, begin:end] = np.sum(
                    nonpositive & (window < reference), axis=0, dtype=np.int64
                )
            margin_host = preactivation.astype(np.float32, copy=True).reshape(-1)

        self._checked_add_counts(
            state.flip_count_deactivate,
            deactivate_host,
            destination_start,
            destination_stop,
        )
        self._checked_add_counts(
            state.flip_count_activate,
            activate_host,
            destination_start,
            destination_stop,
        )
        state.margin_chunks.append(margin_host)

    def residency_bytes(self) -> dict[str, Any]:
        """Bytes held by the pooled-margin lists, per FC layer.

        THE STORE THE PROVIDER'S BREAKDOWN CANNOT SEE. `residency_breakdown`
        walks `_sequential_device_states` on the response provider; this
        accumulator is a LOCAL of the census loop, constructed once before the
        look loop and finalized only after the last look. So `margin_chunks`
        retains a float32 preactivation row for every context of the whole
        census -- growth indexed by CUMULATIVE contexts, which is exactly the
        signature the RSS series shows -- and it exists solely to feed seven
        percentiles at the end.

        This is the same class of gap that already cost one
        misattribution here (the census accumulates into an ARGUMENT, not into
        the provider's dict), so it is weighed rather than argued about.
        """

        per_layer: dict[str, Any] = {}
        for layer, state in self._linear.items():
            chunk_bytes = sum(int(chunk.nbytes) for chunk in state.margin_chunks)
            per_layer[layer] = {
                "margin_chunk_bytes": chunk_bytes,
                "margin_chunk_count": len(state.margin_chunks),
                "context_count": int(state.context_count),
                "counter_bytes": (
                    int(state.flip_count_deactivate.nbytes)
                    + int(state.flip_count_activate.nbytes)
                ),
            }
        return {
            "schema": "dense_flip_margin_residency_v1",
            "by_layer": per_layer,
            "total_margin_chunk_bytes": sum(
                int(row["margin_chunk_bytes"]) for row in per_layer.values()
            ),
        }

    def finalize(self) -> dict[str, Any]:
        deactivate: dict[str, np.ndarray] = {}
        activate: dict[str, np.ndarray] = {}
        margins: dict[str, np.ndarray | str] = dict(self._margin_status)
        contexts: dict[str, int] = {}
        for layer, state in self._linear.items():
            if state.context_count <= 0 or not state.margin_chunks:
                raise RuntimeError("FC flip-margin layer received no contexts")
            pooled = np.concatenate(state.margin_chunks)
            deactivate[layer] = np.array(
                state.flip_count_deactivate, dtype=np.int32, copy=True
            )
            activate[layer] = np.array(
                state.flip_count_activate, dtype=np.int32, copy=True
            )
            margins[layer] = np.percentile(
                pooled, MARGIN_PERCENTILE_LEVELS
            ).astype(np.float64, copy=False)
            contexts[layer] = int(state.context_count)
        return {
            "schema": "dense_flip_margin_geometry_v1",
            "flip_count_deactivate": deactivate,
            "flip_count_activate": activate,
            "margin_percentiles": margins,
            "margin_percentile_levels": MARGIN_PERCENTILE_LEVELS,
            "context_count_by_fc_layer": contexts,
            "conv_margins": {
                layer: MARGIN_NOT_CAPTURED for layer in sorted(self._conv_layers)
            },
            "conv_flip_counters": "not_emitted",
        }


def validate_flip_margin_geometry(
    geometry: Mapping[str, Any],
) -> None:
    """Validate the durable observation payload without widening semantics."""

    if geometry.get("schema") != "dense_flip_margin_geometry_v1":
        raise ValueError("dense flip-margin geometry schema is invalid")
    levels = tuple(float(value) for value in geometry.get("margin_percentile_levels", ()))
    if levels != MARGIN_PERCENTILE_LEVELS:
        raise ValueError("dense flip-margin percentile levels are invalid")
    deactivate = geometry.get("flip_count_deactivate")
    activate = geometry.get("flip_count_activate")
    margins = geometry.get("margin_percentiles")
    if not isinstance(deactivate, Mapping) or not isinstance(activate, Mapping):
        raise ValueError("dense flip-margin counters are missing")
    if set(deactivate) != set(activate) or not isinstance(margins, Mapping):
        raise ValueError("dense flip-margin layer axes are inconsistent")
    for layer in deactivate:
        left = np.asarray(deactivate[layer])
        right = np.asarray(activate[layer])
        if (
            left.dtype != np.int32
            or right.dtype != np.int32
            or left.shape != right.shape
            or np.any(left < 0)
            or np.any(right < 0)
        ):
            raise ValueError("dense flip-margin FC counters are invalid")
        margin = np.asarray(margins.get(layer), dtype=np.float64)
        if margin.shape != (len(MARGIN_PERCENTILE_LEVELS),) or not np.isfinite(margin).all():
            raise ValueError("dense flip-margin FC percentiles are invalid")


__all__ = [
    "DenseFlipMarginAccumulator",
    "MARGIN_NOT_APPLICABLE",
    "MARGIN_NOT_CAPTURED",
    "MARGIN_PERCENTILE_LEVELS",
    "validate_flip_margin_geometry",
]
