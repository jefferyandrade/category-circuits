"""Observation-only enrichments for paired v0 pruning decisions.

Nothing in this module participates in a decision statistic.  The helpers
summarize already-resolved panels and build the bounded per-context sidecar
that is flushed by the registered orchestration layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import os
from typing import Any, Mapping, Sequence

import numpy as np


STEP_OBSERVABLES_ENV = "CIRCUIT_STEP_OBSERVABLES"
STEP_CONTEXT_DELTA_SIDECAR_FILENAME = "step_context_deltas.npz"
STEP_CONTEXT_DELTA_SIDECAR_SCHEMA_V1 = "step_context_deltas_v1"
STEP_CONTEXT_DELTA_SIDECAR_SCHEMA = "step_context_deltas_v2"
STEP_CONTEXT_DELTA_MAX_BYTES = 256 * 1024 * 1024

# v2 dictionary-encodes the four label columns.  MEASURED motivation, from the
# landed fleet: v1 stored one fixed-width unicode value PER ROW for fields that
# are constant across a decision's whole panel, and NumPy unicode is UCS-4, so
# every character costs 4 bytes:
#
#   context_id <U68  272 B    layer_id <U20   80 B
#   action     <U22   88 B    partition <U8   32 B
#   proposal_index int64 8 B  delta float32    4 B   -> 484 B/row
#
# (Exact: 484 x 434,944 + 4,096 = 210,516,992, the estimate one landed arm
# recorded.)  An entire arm contains 7 distinct layers, 2 partitions and 3
# actions, so ~96% of those bytes are the same few strings rewritten millions
# of times.  Codes plus one small table carry the same information in 3 B.
#
# The cap is evaluated against this UNCOMPRESSED estimate and sealing is
# PERMANENT, so the inflation was not merely wasteful -- it silently truncated
# four of fourteen landed arms, one of which kept only 67 of its 154 decisions.
# On disk those files cost ~54 B/row because deflate collapses the repetition,
# so the truncation was charged against bytes that were never written.
STEP_CONTEXT_DELTA_FIXED_BYTES_PER_ROW = (
    4  # proposal_index, int32
    + 4  # paired_bce_delta, float32
    + 1  # layer code, uint8-equivalent
    + 1  # partition code
    + 1  # action code
    + 4  # context code, int32
)

# Which identity the context column carries.  ``draw_id`` is salted with the
# draw index, so it is unique per row BY CONSTRUCTION -- incompressible, and
# useless as a join key (no two decisions ever share one).  ``source_context_id``
# is the underlying image, drawn WITH replacement, so it both repeats (the
# codebook compresses it) and lets two decisions be paired on common contexts.
CONTEXT_IDENTITY_SOURCE = "source_context_id"
CONTEXT_IDENTITY_DRAW = "draw_id"
PREACTIVATION_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def step_observables_enabled() -> bool:
    """Return the default-on state of the single observation kill switch."""

    value = os.environ.get(STEP_OBSERVABLES_ENV)
    return value is None or value.strip().lower() not in {"0", "false", "no"}


def _residual_layer_outcome_host(
    parent_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    context_count: int,
) -> tuple[Any, ...]:
    """One eligible layer's v5 host outcome (both captures present).

    Returns ``("shape_mismatch",)`` or ``("observed", flip_count,
    percentiles)``; the arithmetic is exactly v5's.
    """

    if (
        parent_values.shape != candidate_values.shape
        or parent_values.ndim < 1
        or int(parent_values.shape[0]) != int(context_count)
    ):
        return ("shape_mismatch",)
    candidate_flat = np.asarray(candidate_values, dtype=np.float64).reshape(-1)
    if candidate_flat.size == 0 or not np.isfinite(candidate_flat).all():
        return ("shape_mismatch",)
    flip_count = int(
        np.count_nonzero((parent_values > 0) != (candidate_values > 0))
    )
    percentile_values = np.percentile(
        candidate_flat,
        PREACTIVATION_PERCENTILES,
    )
    return (
        "observed",
        flip_count,
        {
            str(percentile): float(value)
            for percentile, value in zip(
                PREACTIVATION_PERCENTILES,
                percentile_values,
                strict=True,
            )
        },
    )


def _residual_observables_payload(
    *,
    residual_ids: tuple[str, ...],
    out_of_scope_ids: tuple[str, ...],
    outcomes: Mapping[str, tuple[Any, ...]],
    context_count: int,
) -> dict[str, Any]:
    """Assemble the v5 payload from per-layer outcomes, in residual order."""

    missing_parent: list[str] = []
    missing_candidate: list[str] = []
    shape_mismatch: list[str] = []
    observed: list[str] = []
    flip_counts: dict[str, int] = {}
    percentiles: dict[str, dict[str, float]] = {}
    for layer_id in residual_ids:
        outcome = outcomes[layer_id]
        kind = outcome[0]
        if kind == "missing_parent":
            missing_parent.append(layer_id)
        elif kind == "missing_candidate":
            missing_candidate.append(layer_id)
        elif kind == "shape_mismatch":
            shape_mismatch.append(layer_id)
        elif kind == "observed":
            flip_counts[layer_id] = int(outcome[1])
            percentiles[layer_id] = dict(outcome[2])
            observed.append(layer_id)
        else:  # pragma: no cover - internal contract
            raise ValueError(f"unknown residual layer outcome {kind!r}")
    coverage = {
        "scope": "residual_stream_gate_terminal_relu_preactivation",
        "panel_context_count": int(context_count),
        "eligible_layer_ids": residual_ids,
        "observed_layer_ids": tuple(observed),
        "out_of_scope_layer_ids": out_of_scope_ids,
        "out_of_scope_reason": (
            "gated wrapper does not own an identifiable nonlinearity"
        ),
        "missing_parent_layer_ids": tuple(missing_parent),
        "missing_candidate_layer_ids": tuple(missing_candidate),
        "shape_or_value_invalid_layer_ids": tuple(shape_mismatch),
    }
    return {
        "activation_flip_counts": flip_counts,
        "activation_flip_context_count": int(context_count),
        "candidate_preactivation_percentiles": percentiles,
        "activation_coverage": coverage,
        "preactivation_percentile_coverage": dict(coverage),
    }


def _residual_host_outcomes(
    *,
    parent_preactivations: Mapping[str, Any],
    candidate_preactivations: Mapping[str, Any],
    residual_ids: tuple[str, ...],
    context_count: int,
) -> dict[str, tuple[Any, ...]]:
    parent = {
        str(key): np.asarray(value)
        for key, value in parent_preactivations.items()
    }
    candidate = {
        str(key): np.asarray(value) for key, value in candidate_preactivations.items()
    }
    outcomes: dict[str, tuple[Any, ...]] = {}
    for layer_id in residual_ids:
        if layer_id not in parent:
            outcomes[layer_id] = ("missing_parent",)
        elif layer_id not in candidate:
            outcomes[layer_id] = ("missing_candidate",)
        else:
            outcomes[layer_id] = _residual_layer_outcome_host(
                parent[layer_id],
                candidate[layer_id],
                context_count=int(context_count),
            )
    return outcomes


def summarize_residual_step_observables(
    *,
    parent_preactivations: Mapping[str, Any],
    candidate_preactivations: Mapping[str, Any],
    residual_layer_ids: Sequence[str],
    out_of_scope_layer_ids: Sequence[str],
    context_count: int,
) -> dict[str, Any]:
    """Summarize exact residual-boundary flips and candidate margins."""

    residual_ids = tuple(
        dict.fromkeys(str(value) for value in residual_layer_ids)
    )
    out_of_scope_ids = tuple(
        dict.fromkeys(str(value) for value in out_of_scope_layer_ids)
    )
    return _residual_observables_payload(
        residual_ids=residual_ids,
        out_of_scope_ids=out_of_scope_ids,
        outcomes=_residual_host_outcomes(
            parent_preactivations=parent_preactivations,
            candidate_preactivations=candidate_preactivations,
            residual_ids=residual_ids,
            context_count=int(context_count),
        ),
        context_count=int(context_count),
    )


# Two 16-bit radix histograms select the float32 order statistics on the
# capture device. Interpolation uses NumPy's linear-percentile helpers on
# the selected float64 scalar values. Signed-zero ambiguity is resolved by
# the host percentile algorithm for that layer.

RESIDUAL_OBSERVABLES_DEVICE_CACHE_BYTES_ENV = "IRG_RESIDUAL_OBSERVABLES_DEVICE_CACHE_BYTES"
RESIDUAL_OBSERVABLES_DEVICE_CACHE_BYTES_DEFAULT = 1 << 30
_KEY_BINS = 1 << 16
_HIGH_OFFSET = 1 << 15
_HISTOGRAM_CHUNK = 1 << 24
_INT32_MIN = -(1 << 31)
_FLOAT32_EXPONENT_MASK = 0x7F800000
_FLOAT32_POSITIVE_INFINITY_BITS = 0x7F800000
_NUMPY_PERCENTILE_INTERNALS: Any = None


def residual_observables_device_cache_bytes(
    environ: Mapping[str, str] | None = None,
) -> int:
    source = os.environ if environ is None else environ
    raw = str(source.get(RESIDUAL_OBSERVABLES_DEVICE_CACHE_BYTES_ENV, "")).strip()
    if not raw:
        return int(RESIDUAL_OBSERVABLES_DEVICE_CACHE_BYTES_DEFAULT)
    value = int(raw)
    if value < 0:
        raise ValueError(f"{RESIDUAL_OBSERVABLES_DEVICE_CACHE_BYTES_ENV} must be >= 0")
    return value


def _percentile_quantiles() -> np.ndarray:
    # np.percentile divides q by 100 in float64 (numpy 2.1: by a float64 100
    # for float64 input; numpy 2.4: by the integer 100) -- identical values.
    return np.true_divide(np.asarray(PREACTIVATION_PERCENTILES), 100)


def _numpy_percentile_internals() -> Any:
    """Load NumPy's linear-quantile indexing and interpolation helpers."""

    global _NUMPY_PERCENTILE_INTERNALS
    if _NUMPY_PERCENTILE_INTERNALS is not None:
        return _NUMPY_PERCENTILE_INTERNALS or None
    try:
        try:
            from numpy.lib import _function_base_impl as base
        except ImportError:  # pragma: no cover - numpy 1.22-1.26
            from numpy.lib import function_base as base
        internals = (base._QuantileMethods["linear"], base._get_indexes,
                     base._get_gamma, base._lerp)
    except (ImportError, AttributeError, KeyError):
        internals = False
    _NUMPY_PERCENTILE_INTERNALS = internals
    return internals or None


def _linear_percentile_plan(internals: Any, count: int) -> tuple[Any, Any, Any, Any]:
    method, get_indexes, get_gamma, _lerp = internals
    quantiles = _percentile_quantiles()
    virtual = np.asanyarray(method["get_virtual_index"](int(count), quantiles))
    previous, following = get_indexes(np.empty(1, dtype=np.float64), virtual, int(count))
    gamma = get_gamma(virtual, previous, method)
    gamma = np.asanyarray(gamma).reshape(virtual.shape)
    return virtual, previous, following, gamma


def linear_percentile_ranks(count: int) -> tuple[int, ...]:
    """The 0-based ranks np.percentile ('linear') reads for PREACTIVATION_PERCENTILES."""

    internals = _numpy_percentile_internals()
    if internals is None:
        raise RuntimeError("numpy percentile internals are unavailable")
    _virtual, previous, following, _gamma = _linear_percentile_plan(internals, count)
    return tuple(
        sorted({int(rank) % int(count) for rank in (*previous.tolist(), *following.tolist())})
    )


def _linear_percentiles_with(
    internals: Any,
    count: int,
    order_statistics: Any,
) -> tuple[float, ...]:
    _method, _get_indexes, _get_gamma, lerp = internals
    _virtual, previous, following, gamma = _linear_percentile_plan(internals, count)
    ranks = sorted(
        {int(rank) % int(count) for rank in (*previous.tolist(), *following.tolist())}
    )
    values = order_statistics(tuple(ranks))
    previous_values = np.asarray(
        [values[int(rank) % int(count)] for rank in previous.tolist()], dtype=np.float64
    )
    following_values = np.asarray(
        [values[int(rank) % int(count)] for rank in following.tolist()], dtype=np.float64
    )
    result = lerp(previous_values, following_values, gamma)
    return tuple(float(value) for value in np.asarray(result, dtype=np.float64).reshape(-1))


def linear_percentiles_from_order_statistics(
    count: int,
    order_statistics: Any,
) -> dict[str, float]:
    """np.percentile(x, PREACTIVATION_PERCENTILES) from x's order statistics.

    ``order_statistics(ranks)`` returns ``{rank: float64 value}`` for the
    0-based ranks of the sorted sample; numpy's own index, gamma and lerp code
    does the rest, so the result equals np.percentile bit for bit.
    """

    internals = _numpy_percentile_internals()
    if internals is None:
        raise RuntimeError("numpy percentile internals are unavailable")
    values = _linear_percentiles_with(internals, int(count), order_statistics)
    return {
        str(percentile): float(value)
        for percentile, value in zip(PREACTIVATION_PERCENTILES, values, strict=True)
    }


def _float32_from_key(key: int) -> float:
    bits = int(key) if key >= 0 else int(key) ^ 0x7FFFFFFF
    return float(np.asarray([bits], dtype=np.int32).view(np.float32)[0])


class _ResidualLayerStream:
    """Device accumulators of one residual layer across the panel's batches."""

    __slots__ = (
        "parent_rows",
        "parent_rest",
        "candidate_rows",
        "candidate_rest",
        "aligned",
        "flip",
        "nonfinite",
        "negative_zero",
        "positive_zero",
        "high",
        "keys",
        "pass_b_low",
        "target_bins",
        "slot_table",
    )

    def __init__(self) -> None:
        self.parent_rows: list[int | None] = []
        self.parent_rest: tuple[int, ...] | None = None
        self.candidate_rows: list[int | None] = []
        self.candidate_rest: tuple[int, ...] | None = None
        self.aligned = True
        self.flip: Any = None
        self.nonfinite: Any = None
        self.negative_zero: Any = None
        self.positive_zero: Any = None
        self.high: Any = None
        self.keys: list[Any] | None = []
        self.pass_b_low: Any = None
        self.target_bins: tuple[int, ...] = ()
        self.slot_table: Any = None


class _DeviceObservationFallback(RuntimeError):
    """The capture dtype or shape requires host percentile arithmetic."""


class ResidualDeviceObservation:
    """Stream residual activation flips and percentiles on the capture device.

    Pass A consumes each batch's (parent, candidate) captures; pass B (only
    when the candidate keys did not fit the device cache budget) consumes a
    re-run candidate capture.  ``outcomes()`` returns per-layer outcomes in
    the host summarizer's outcome shape, plus the layers that must use
    the host path.
    """

    def __init__(
        self,
        residual_layer_ids: Sequence[str],
        *,
        context_count: int,
        cache_budget_bytes: int = RESIDUAL_OBSERVABLES_DEVICE_CACHE_BYTES_DEFAULT,
    ) -> None:
        import torch

        self._torch = torch
        self.residual_ids = tuple(dict.fromkeys(str(value) for value in residual_layer_ids))
        self.context_count = int(context_count)
        self._layers = {layer_id: _ResidualLayerStream() for layer_id in self.residual_ids}
        from mechanistic_systems.measurement.gpu_runtime import process_device_memory_budget_bytes

        self._cache_budget = process_device_memory_budget_bytes(int(cache_budget_bytes))
        self._cache_bytes = 0
        self._cache_enabled = self._cache_budget > 0
        self._batches_a = 0
        self._batches_b = 0
        self._planned = False
        self.host_layers: set[str] = set()
        self.telemetry: dict[str, Any] = {}

    # -- device kernels ----------------------------------------------------
    def _bits(self, values: Any) -> Any:
        torch = self._torch
        if values.dtype != torch.float32:
            raise _DeviceObservationFallback("residual capture is not float32")
        if values.dim() < 1:
            raise _DeviceObservationFallback("zero-dimensional residual capture")
        return values.contiguous().view(torch.int32).reshape(-1)

    @staticmethod
    def _positive(bits: Any) -> Any:
        return (bits > 0) & (bits <= _FLOAT32_POSITIVE_INFINITY_BITS)

    def positive_map(self, values: Any) -> Any:
        """The exact ``values > 0`` map of a float32 parent capture, as bool.

        The parent is only ever read through this map (the flip count), so a
        capture stream may keep this 1-byte map in place of the 4-byte float32
        capture while the candidate forward runs.  Computed on the bit image,
        exact on every device; shape preserved for the v5 shape checks.
        """

        return self._positive(self._bits(values)).reshape(tuple(values.shape))

    def _key(self, bits: Any) -> Any:
        return bits ^ ((bits >> 31) & 0x7FFFFFFF)

    def _histogram_into(self, hist: Any, index: Any) -> Any:
        """``hist[i] += count(index == i)``, chunked, with no host sync.

        ``index_add_`` of an expanded ones vector: integer atomics on the
        device, exact, and (unlike ``bincount``) no ``max().item()`` sync.
        Chunking bounds the int64 temporaries to ~128 MiB per chunk.
        """

        torch = self._torch
        flat = index.reshape(-1)
        for start in range(0, int(flat.numel()), _HISTOGRAM_CHUNK):
            part = flat[start : start + _HISTOGRAM_CHUNK].to(torch.int64)
            hist.index_add_(
                0,
                part,
                torch.ones(1, dtype=torch.int64, device=part.device).expand(part.numel()),
            )
        return hist

    def _new_histogram(self, bins: int, device: Any) -> Any:
        return self._torch.zeros(int(bins), dtype=self._torch.int64, device=device)

    @staticmethod
    def _add(accumulator: Any, value: Any) -> Any:
        return value if accumulator is None else accumulator + value

    @staticmethod
    def _record_shape(rows: list[int | None], rest_attr: tuple[int, ...] | None, values: Any):
        shape = tuple(int(size) for size in values.shape)
        rows.append(shape[0])
        rest = shape[1:]
        if rest_attr is not None and rest_attr != rest:
            # torch.cat in the v5 capture would raise for this layer.
            raise _DeviceObservationFallback("residual capture rest shape changes")
        return rest

    # -- pass A ------------------------------------------------------------
    def consume_pair(
        self,
        parent_capture: Mapping[str, Any],
        candidate_capture: Mapping[str, Any],
    ) -> None:
        """One batch of both captures (same batch, parent then candidate).

        A parent capture is either the float32 pre-activation or its bool
        :meth:`positive_map` (what the provider's capture stream delivers).
        """

        torch = self._torch
        self._batches_a += 1
        for layer_id, state in self._layers.items():
            parent = parent_capture.get(layer_id)
            candidate = candidate_capture.get(layer_id)
            parent_positive = None
            if parent is None:
                state.parent_rows.append(None)
            else:
                state.parent_rest = self._record_shape(
                    state.parent_rows, state.parent_rest, parent
                )
                if parent.dtype == torch.bool:
                    if parent.dim() < 1:
                        raise _DeviceObservationFallback(
                            "zero-dimensional residual capture"
                        )
                    parent_positive = parent.reshape(-1)
                else:
                    parent_positive = self._positive(self._bits(parent))
            if candidate is None:
                state.candidate_rows.append(None)
                if parent is not None:
                    state.aligned = False
                continue
            state.candidate_rest = self._record_shape(
                state.candidate_rows, state.candidate_rest, candidate
            )
            bits = self._bits(candidate)
            if parent_positive is None or tuple(parent.shape) != tuple(candidate.shape):
                state.aligned = False
            else:
                state.flip = self._add(
                    state.flip,
                    (parent_positive != self._positive(bits)).sum(dtype=torch.int64),
                )
            del parent_positive
            state.nonfinite = self._add(
                state.nonfinite,
                ((bits & _FLOAT32_EXPONENT_MASK) == _FLOAT32_EXPONENT_MASK).sum(
                    dtype=torch.int64
                ),
            )
            state.negative_zero = self._add(
                state.negative_zero, (bits == _INT32_MIN).sum(dtype=torch.int64)
            )
            state.positive_zero = self._add(
                state.positive_zero, (bits == 0).sum(dtype=torch.int64)
            )
            key = self._key(bits)
            if state.high is None:
                state.high = self._new_histogram(_KEY_BINS, key.device)
            self._histogram_into(state.high, (key >> 16) + _HIGH_OFFSET)
            if self._cache_enabled and state.keys is not None:
                size = int(key.numel()) * 4
                if self._cache_bytes + size <= self._cache_budget:
                    state.keys.append(key)
                    self._cache_bytes += size
                else:
                    self._drop_cache()

    def _drop_cache(self) -> None:
        self._cache_enabled = False
        self._cache_bytes = 0
        for state in self._layers.values():
            state.keys = None

    # -- planning ------------------------------------------------------------
    def _shape(self, rows: list[int | None], rest: tuple[int, ...] | None) -> tuple[int, ...] | None:
        present = [value for value in rows if value is not None]
        if not present:
            return None
        return (int(sum(present)), *(rest or ()))

    def plan(self) -> bool:
        """Classify every layer after pass A; return True if pass B must re-run."""

        torch = self._torch
        self._planned = True
        self._outcomes: dict[str, tuple[Any, ...]] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        for layer_id, state in self._layers.items():
            parent_shape = self._shape(state.parent_rows, state.parent_rest)
            candidate_shape = self._shape(state.candidate_rows, state.candidate_rest)
            if parent_shape is None:
                self._outcomes[layer_id] = ("missing_parent",)
                continue
            if candidate_shape is None:
                self._outcomes[layer_id] = ("missing_candidate",)
                continue
            if (
                parent_shape != candidate_shape
                or len(parent_shape) < 1
                or int(parent_shape[0]) != self.context_count
            ):
                self._outcomes[layer_id] = ("shape_mismatch",)
                continue
            count = 1
            for size in candidate_shape:
                count *= int(size)
            nonfinite = 0 if state.nonfinite is None else int(state.nonfinite.item())
            if count == 0 or nonfinite:
                self._outcomes[layer_id] = ("shape_mismatch",)
                continue
            if not state.aligned or state.parent_rows != state.candidate_rows:
                self.host_layers.add(layer_id)
                continue
            high = state.high.to(device="cpu").numpy().astype(np.int64, copy=False)
            if int(high.sum()) != count:
                raise ValueError("residual histogram count differs from the capture shape")
            ranks = linear_percentile_ranks(count)
            cumulative = np.cumsum(high)
            bins: dict[int, int] = {}
            for rank in ranks:
                bin_index = int(np.searchsorted(cumulative, rank, side="right"))
                bins[rank] = bin_index
            target = tuple(sorted(set(bins.values())))
            table = torch.full((_KEY_BINS,), -1, dtype=torch.int64, device=state.high.device)
            table[torch.as_tensor(target, dtype=torch.int64, device=state.high.device)] = (
                torch.arange(len(target), dtype=torch.int64, device=state.high.device)
            )
            state.target_bins = target
            state.slot_table = table
            # One extra trailing bin absorbs every element outside the target
            # bins, so level 2 needs no boolean indexing (no host sync).
            state.pass_b_low = self._new_histogram(
                len(target) * _KEY_BINS + 1, state.high.device
            )
            self._pending[layer_id] = {
                "count": count,
                "ranks": ranks,
                "bin_of_rank": bins,
                "cumulative_high": cumulative,
                "flip": 0 if state.flip is None else int(state.flip.item()),
                "negative_zero": int(state.negative_zero.item()),
                "positive_zero": int(state.positive_zero.item()),
            }
        needs_forward = bool(self._pending) and not self._cache_enabled
        if self._pending and self._cache_enabled:
            for layer_id in self._pending:
                state = self._layers[layer_id]
                for key in state.keys or ():
                    self._accumulate_low(state, key)
                state.keys = None
        self.telemetry = {
            "pass_b_forward": bool(needs_forward),
            "cache_bytes": int(self._cache_bytes),
            "pending_layers": len(self._pending),
            "host_layers": sorted(self.host_layers),
        }
        return needs_forward

    def _accumulate_low(self, state: _ResidualLayerStream, key: Any) -> None:
        torch = self._torch
        dump = int(state.pass_b_low.numel()) - 1
        flat = key.reshape(-1)
        for start in range(0, int(flat.numel()), _HISTOGRAM_CHUNK):
            part = flat[start : start + _HISTOGRAM_CHUNK]
            slot = state.slot_table[((part >> 16) + _HIGH_OFFSET).to(torch.int64)]
            index = torch.where(
                slot >= 0,
                slot * _KEY_BINS + (part & 0xFFFF).to(torch.int64),
                torch.full((), dump, dtype=torch.int64, device=part.device),
            )
            self._histogram_into(state.pass_b_low, index)

    # -- pass B ------------------------------------------------------------
    def consume_candidate(self, candidate_capture: Mapping[str, Any]) -> None:
        """One batch of the re-run candidate capture (pass B)."""

        self._batches_b += 1
        for layer_id in self._pending:
            state = self._layers[layer_id]
            candidate = candidate_capture.get(layer_id)
            if candidate is None:
                raise ValueError("candidate capture omitted a residual percentile layer")
            if tuple(candidate.shape[1:]) != state.candidate_rest:
                raise ValueError("candidate residual capture shape changed")
            self._accumulate_low(state, self._key(self._bits(candidate)))

    # -- resolution --------------------------------------------------------
    def outcomes(self) -> dict[str, tuple[Any, ...]]:
        if not self._planned:
            raise RuntimeError("plan() must run before outcomes()")
        outcomes = dict(self._outcomes)
        for layer_id, pending in self._pending.items():
            state = self._layers[layer_id]
            low = state.pass_b_low.to(device="cpu").numpy().astype(np.int64, copy=False)
            low = low[: len(state.target_bins) * _KEY_BINS].reshape(
                len(state.target_bins), _KEY_BINS
            )
            slot_of_bin = {bin_index: slot for slot, bin_index in enumerate(state.target_bins)}
            cumulative_high = pending["cumulative_high"]
            values: dict[int, float] = {}
            ambiguous = False
            for rank in pending["ranks"]:
                bin_index = pending["bin_of_rank"][rank]
                below = 0 if bin_index == 0 else int(cumulative_high[bin_index - 1])
                within = int(rank) - below
                cumulative_low = np.cumsum(low[slot_of_bin[bin_index]])
                low_index = int(np.searchsorted(cumulative_low, within, side="right"))
                if low_index >= _KEY_BINS:
                    raise ValueError("residual radix histogram cannot resolve the selected rank")
                key = ((bin_index - _HIGH_OFFSET) << 16) | low_index
                if key in (-1, 0) and pending["negative_zero"] and pending["positive_zero"]:
                    # numpy's introselect may return either signed zero here.
                    ambiguous = True
                    break
                values[int(rank)] = _float32_from_key(key)
            if ambiguous:
                self.host_layers.add(layer_id)
                continue
            percentiles = linear_percentiles_from_order_statistics(
                pending["count"], lambda ranks, _values=values: {r: _values[r] for r in ranks}
            )
            outcomes[layer_id] = ("observed", int(pending["flip"]), percentiles)
        for layer_id in self.host_layers:
            outcomes.pop(layer_id, None)
        self.telemetry["host_layers"] = sorted(self.host_layers)
        return outcomes


def summarize_residual_step_observables_from_outcomes(
    *,
    outcomes: Mapping[str, tuple[Any, ...]],
    residual_layer_ids: Sequence[str],
    out_of_scope_layer_ids: Sequence[str],
    context_count: int,
) -> dict[str, Any]:
    """The v5 payload from per-layer outcomes (device and host layers mixed)."""

    residual_ids = tuple(dict.fromkeys(str(value) for value in residual_layer_ids))
    out_of_scope_ids = tuple(dict.fromkeys(str(value) for value in out_of_scope_layer_ids))
    return _residual_observables_payload(
        residual_ids=residual_ids,
        out_of_scope_ids=out_of_scope_ids,
        outcomes=outcomes,
        context_count=int(context_count),
    )


def _unicode_width(values: Sequence[str]) -> int:
    return max((len(str(value)) for value in values), default=1)


class _Codebook:
    """Insertion-ordered string interner backing one dictionary-encoded column."""

    __slots__ = ("_codes", "_values", "_max_width")

    def __init__(self) -> None:
        self._codes: dict[str, int] = {}
        self._values: list[str] = []
        self._max_width = 1  # mirrors _unicode_width's empty-sequence default

    def __len__(self) -> int:
        return len(self._values)

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(self._values)

    @property
    def max_width(self) -> int:
        return self._max_width

    def width_with(self, value: str) -> int:
        return max(self._max_width, len(value))

    def count_with(self, value: str) -> int:
        return len(self._values) + (0 if value in self._codes else 1)

    def intern(self, value: str) -> int:
        code = self._codes.get(value)
        if code is None:
            code = len(self._values)
            self._codes[value] = code
            self._values.append(value)
            if len(value) > self._max_width:
                self._max_width = len(value)
        return code

    def decode(self, codes: Sequence[int]) -> list[str]:
        return [self._values[int(code)] for code in codes]


@dataclass
class StepContextDeltaAccumulator:
    """Bounded batch-at-finalize storage for completed-decision BCE rows.

    Columns are dictionary-encoded (see the module header for the measured
    484 B/row -> 15 B/row motivation).  ``layer_ids``/``partition_ids``/
    ``actions``/``context_ids`` remain readable as decoded string sequences so
    the stored representation is an implementation detail, not an API change.
    """

    cap_bytes: int = STEP_CONTEXT_DELTA_MAX_BYTES
    proposal_indices: list[int] = field(default_factory=list)
    paired_bce_deltas: list[np.float32] = field(default_factory=list)
    truncated_row_count: int = 0
    unavailable_row_count: int = 0
    _sealed_by_cap: bool = False

    # Conservative allowance for npz member headers and scalar metadata, so
    # the estimate below UPPER-BOUNDS the physical file. MEASURED: the v2
    # payload has 27 members and an empty sidecar serializes to 5,684 bytes
    # (zip local header + npy header + central directory entry per member).
    # The old 4,096 predates the codebook and moment members and would let the
    # estimator authorize a write that sidecar_bytes() then rejects.
    _container_allowance_bytes: int = field(
        default=8192,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.cap_bytes = int(self.cap_bytes)
        if self.cap_bytes <= self._container_allowance_bytes:
            raise ValueError("step-context sidecar cap is too small for metadata")
        self._layers = _Codebook()
        self._partitions = _Codebook()
        self._actions = _Codebook()
        self._contexts = _Codebook()
        self._layer_codes: list[int] = []
        self._partition_codes: list[int] = []
        self._action_codes: list[int] = []
        self._context_codes: list[int] = []
        self._context_identity = CONTEXT_IDENTITY_DRAW
        self._decision_moments: list[tuple[Any, ...]] = []

    # -- decoded views -------------------------------------------------
    @property
    def layer_ids(self) -> list[str]:
        return self._layers.decode(self._layer_codes)

    @property
    def partition_ids(self) -> list[str]:
        return self._partitions.decode(self._partition_codes)

    @property
    def actions(self) -> list[str]:
        return self._actions.decode(self._action_codes)

    @property
    def context_ids(self) -> list[str]:
        return self._contexts.decode(self._context_codes)

    @property
    def context_identity(self) -> str:
        return self._context_identity

    def _estimated_bytes_with(
        self,
        *,
        layer_id: str | None = None,
        partition_id: str | None = None,
        action: str | None = None,
        context_id: str | None = None,
    ) -> int:
        """Bytes the encoded npz would occupy, including one more row.

        Measured: this runs once PER APPENDED CONTEXT (~2,300 per
        decision), and an earlier form rebuilt tuples of every value
        accumulated since step 1 and rescanned them for max width --
        O(total_rows) per context, O(step x panel^2) per decision. On a real
        187-decision arm that was the dominant walk cost: decision epilogues
        grew 35 s -> 210 s linearly at ~1.1 s/step while evidence stayed at
        ~3 s. Running maxima and interned counts keep this O(1) and EXACT:
        width is a max (associative) and the distinct count is a set size, so
        the cached values equal a full rescan and sealing occurs at identical
        points.

        The row term is now FIXED-WIDTH -- codes do not depend on string
        length -- and the only length-dependent term is each codebook, which
        is written once.
        """

        row_count = len(self.proposal_indices) + (
            1 if context_id is not None else 0
        )
        codebook_bytes = 0
        for book, pending in (
            (self._layers, layer_id),
            (self._partitions, partition_id),
            (self._actions, action),
            (self._contexts, context_id),
        ):
            if pending is None:
                width, count = book.max_width, len(book)
            else:
                text = str(pending)
                width, count = book.width_with(text), book.count_with(text)
            codebook_bytes += 4 * width * count
        # The moment table is O(decisions) and is never truncated, so it must
        # still be charged: 4 x int32 + int64 + 5 x float64 per decision.
        moment_bytes = len(self._decision_moments) * (4 * 4 + 8 + 5 * 8)
        return (
            self._container_allowance_bytes
            + row_count * STEP_CONTEXT_DELTA_FIXED_BYTES_PER_ROW
            + codebook_bytes
            + moment_bytes
        )

    @property
    def row_count(self) -> int:
        return len(self.proposal_indices)

    @property
    def estimated_uncompressed_bytes(self) -> int:
        return self._estimated_bytes_with()

    def append_decision(
        self,
        *,
        proposal_index: int,
        layer_id: str,
        partition_id: str,
        action: str,
        context_ids: Sequence[str],
        paired_bce_delta: Any,
        unavailable_context_count: int = 0,
        source_context_ids: Sequence[str] | None = None,
    ) -> None:
        """Append one resolved panel, sealing permanently at the first overflow.

        ``source_context_ids``, when supplied, replaces the draw identity in
        the stored context column.  Draw IDs hash in the draw index and so are
        unique per row by construction: they cannot compress and cannot join
        two decisions.  Source IDs are the underlying images, drawn with
        replacement, so they compress AND make cross-decision pairing possible.
        Omitting them preserves the previous behaviour exactly.
        """

        contexts = tuple(str(value) for value in context_ids)
        unavailable = int(unavailable_context_count)
        if unavailable < 0:
            raise ValueError("unavailable step-context count must be nonnegative")
        deltas = np.asarray(paired_bce_delta, dtype=np.float32).reshape(-1)
        if deltas.size != len(contexts) or not np.isfinite(deltas).all():
            raise ValueError("step-context deltas must be finite and panel-aligned")
        if source_context_ids is not None:
            identities = tuple(str(value) for value in source_context_ids)
            if len(identities) != len(contexts):
                raise ValueError(
                    "step-context source IDs must be panel-aligned"
                )
            if contexts and self._context_identity != CONTEXT_IDENTITY_SOURCE:
                if self._context_codes:
                    raise ValueError(
                        "step-context identity cannot change mid-accumulation"
                    )
                self._context_identity = CONTEXT_IDENTITY_SOURCE
        else:
            identities = contexts
            if contexts and self._context_identity != CONTEXT_IDENTITY_DRAW:
                raise ValueError(
                    "step-context identity cannot change mid-accumulation"
                )
        self.unavailable_row_count += unavailable
        # Per-decision sufficient statistics are recorded BEFORE the cap is
        # consulted, and are never truncated. The table is O(decisions), not
        # O(rows), so it cannot itself exhaust the cap -- which means a sealed
        # arm still carries exact moments for EVERY decision, including the
        # ones whose rows were dropped. Four of fourteen landed arms sealed,
        # one keeping 67 of 154 decisions; under this table their mean/variance
        # decomposition would have survived intact.
        #
        # Raw power sums (not Welford) are used deliberately: they are exactly
        # mergeable across decisions and let the analysis choose its own
        # estimator afterwards, including the bias correction
        # mu^2_unbiased = m^2 - v/n that the (E[delta])^2 vs Var(delta)
        # decomposition needs. Accumulated in float64 from float32 rows.
        if len(contexts):
            wide = deltas.astype(np.float64)
            self._decision_moments.append(
                (
                    int(proposal_index),
                    self._layers.intern(str(layer_id)),
                    self._partitions.intern(str(partition_id)),
                    self._actions.intern(str(action)),
                    int(wide.size),
                    float(wide.sum()),
                    float(np.square(wide).sum()),
                    float(np.abs(wide).sum()),
                    float((wide ** 3).sum()),
                    float((wide ** 4).sum()),
                )
            )
        if self._sealed_by_cap:
            self.truncated_row_count += len(contexts)
            return
        for index, (identity, delta) in enumerate(
            zip(identities, deltas, strict=True)
        ):
            estimated = self._estimated_bytes_with(
                layer_id=str(layer_id),
                partition_id=str(partition_id),
                action=str(action),
                context_id=identity,
            )
            if estimated > self.cap_bytes:
                self._sealed_by_cap = True
                self.truncated_row_count += len(contexts) - index
                break
            self.proposal_indices.append(int(proposal_index))
            self._layer_codes.append(self._layers.intern(str(layer_id)))
            self._partition_codes.append(
                self._partitions.intern(str(partition_id))
            )
            self._action_codes.append(self._actions.intern(str(action)))
            self._context_codes.append(self._contexts.intern(identity))
            self.paired_bce_deltas.append(np.float32(delta))

    @staticmethod
    def _codebook_array(book: "_Codebook") -> np.ndarray:
        return np.asarray(
            book.values,
            dtype=f"<U{_unicode_width(book.values)}",
        )

    def payload(self) -> dict[str, np.ndarray]:
        """Return the typed npz member mapping."""

        def _moment_column(index: int, dtype: Any) -> np.ndarray:
            return np.asarray(
                [row[index] for row in self._decision_moments],
                dtype=dtype,
            )

        return {
            "schema": np.asarray(STEP_CONTEXT_DELTA_SIDECAR_SCHEMA),
            "context_identity": np.asarray(self._context_identity),
            "proposal_index": np.asarray(self.proposal_indices, dtype=np.int32),
            "layer_code": np.asarray(self._layer_codes, dtype=np.int32),
            "partition_code": np.asarray(self._partition_codes, dtype=np.int32),
            "action_code": np.asarray(self._action_codes, dtype=np.int32),
            "context_code": np.asarray(self._context_codes, dtype=np.int32),
            "layer_codebook": self._codebook_array(self._layers),
            "partition_codebook": self._codebook_array(self._partitions),
            "action_codebook": self._codebook_array(self._actions),
            "context_codebook": self._codebook_array(self._contexts),
            "paired_bce_delta": np.asarray(
                self.paired_bce_deltas,
                dtype=np.float32,
            ),
            # Complete for every decision even when the row store sealed.
            "decision_proposal_index": _moment_column(0, np.int32),
            "decision_layer_code": _moment_column(1, np.int32),
            "decision_partition_code": _moment_column(2, np.int32),
            "decision_action_code": _moment_column(3, np.int32),
            "decision_context_count": _moment_column(4, np.int64),
            "decision_delta_sum": _moment_column(5, np.float64),
            "decision_delta_sum_squares": _moment_column(6, np.float64),
            "decision_delta_sum_absolute": _moment_column(7, np.float64),
            "decision_delta_sum_cubes": _moment_column(8, np.float64),
            "decision_delta_sum_quads": _moment_column(9, np.float64),
            "row_count": np.asarray(self.row_count, dtype=np.int64),
            "truncated_row_count": np.asarray(
                self.truncated_row_count,
                dtype=np.int64,
            ),
            "unavailable_row_count": np.asarray(
                self.unavailable_row_count,
                dtype=np.int64,
            ),
            "cap_bytes": np.asarray(self.cap_bytes, dtype=np.int64),
            "estimated_uncompressed_bytes": np.asarray(
                self.estimated_uncompressed_bytes,
                dtype=np.int64,
            ),
        }

    def sidecar_bytes(self) -> bytes:
        """Serialize once and enforce the physical-byte cap as a final guard."""

        stream = io.BytesIO()
        payload = self.payload()
        np.savez_compressed(
            stream,
            **{name: payload[name] for name in sorted(payload)},
        )
        value = stream.getvalue()
        if len(value) > self.cap_bytes:
            raise RuntimeError(
                "conservative step-context estimate failed its physical-byte cap"
            )
        return value


def decode_step_context_deltas(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return flat per-row columns from either sidecar schema.

    Analysis reads one shape regardless of how the file was written: v1 stored
    the label columns literally, v2 stores codes plus codebooks.  Decoding here
    rather than at every call site keeps the encoding an implementation detail.

    ``context_identity`` discloses what the ``context_id`` column MEANS --
    ``draw_id`` values are unique per row by construction and cannot be used to
    pair decisions, whereas ``source_context_id`` values can.  Callers that
    join across decisions must check it rather than assume.
    """

    schema = str(np.asarray(payload["schema"]).item())
    delta = np.asarray(payload["paired_bce_delta"], dtype=np.float32)
    if schema == STEP_CONTEXT_DELTA_SIDECAR_SCHEMA_V1:
        return {
            "schema": schema,
            "context_identity": CONTEXT_IDENTITY_DRAW,
            "proposal_index": np.asarray(payload["proposal_index"]),
            "layer_id": np.asarray(payload["layer_id"]),
            "partition_id": np.asarray(payload["partition_id"]),
            "action": np.asarray(payload["action"]),
            "context_id": np.asarray(payload["context_id"]),
            "paired_bce_delta": delta,
        }
    if schema != STEP_CONTEXT_DELTA_SIDECAR_SCHEMA:
        raise ValueError(f"unsupported step-context sidecar schema {schema!r}")

    def _expand(code_key: str, book_key: str) -> np.ndarray:
        codes = np.asarray(payload[code_key], dtype=np.int64)
        book = np.asarray(payload[book_key])
        if book.size == 0:
            return np.empty((0,), dtype=book.dtype)
        return book[codes]

    return {
        "schema": schema,
        "context_identity": str(
            np.asarray(payload["context_identity"]).item()
        ),
        "proposal_index": np.asarray(payload["proposal_index"]),
        "layer_id": _expand("layer_code", "layer_codebook"),
        "partition_id": _expand("partition_code", "partition_codebook"),
        "action": _expand("action_code", "action_codebook"),
        "context_id": _expand("context_code", "context_codebook"),
        "paired_bce_delta": delta,
    }


def decode_step_decision_moments(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return per-decision damage moments and the mean/variance split.

    ``mean_square_unbiased`` applies the correction the decomposition needs:
    the plug-in ``m^2`` estimates ``mu^2`` with bias ``+sigma^2/n``, so
    ``m^2 - v/n`` is the unbiased estimate.  It can go slightly negative when
    the true mean is near zero, which is information rather than an error, so
    it is reported as computed and left for the caller to floor.

    ``mean_share`` is ``(E[delta])^2 / E[delta^2]``: the fraction of the
    second moment carried by the mean term rather than the contextual
    covariance term, which is exactly the quantity that decides whether a
    mean-based ordering key tracks the dominant moment.
    """

    if "decision_delta_sum" not in payload:
        raise ValueError("sidecar predates the per-decision moment table")
    counts = np.asarray(payload["decision_context_count"], dtype=np.float64)
    total = np.asarray(payload["decision_delta_sum"], dtype=np.float64)
    squares = np.asarray(payload["decision_delta_sum_squares"], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, total / counts, np.nan)
        second = np.where(counts > 0, squares / counts, np.nan)
        variance = np.where(
            counts > 1,
            (squares - counts * mean * mean) / np.maximum(counts - 1.0, 1.0),
            0.0,
        )
        variance = np.maximum(variance, 0.0)
        mean_square_unbiased = mean * mean - np.where(
            counts > 0, variance / counts, 0.0
        )
        mean_share = np.where(second > 0, (mean * mean) / second, np.nan)

    def _codes(code_key: str, book_key: str) -> np.ndarray:
        codes = np.asarray(payload[code_key], dtype=np.int64)
        book = np.asarray(payload[book_key])
        if book.size == 0:
            return np.empty((0,), dtype=book.dtype)
        return book[codes]

    return {
        "proposal_index": np.asarray(payload["decision_proposal_index"]),
        "layer_id": _codes("decision_layer_code", "layer_codebook"),
        "partition_id": _codes("decision_partition_code", "partition_codebook"),
        "action": _codes("decision_action_code", "action_codebook"),
        "context_count": np.asarray(payload["decision_context_count"]),
        "mean": mean,
        "variance": variance,
        "second_moment": second,
        "mean_square": mean * mean,
        "mean_square_unbiased": mean_square_unbiased,
        "mean_share": mean_share,
        "mean_absolute": np.where(
            counts > 0,
            np.asarray(payload["decision_delta_sum_absolute"], np.float64)
            / np.maximum(counts, 1.0),
            np.nan,
        ),
    }


__all__ = [
    "CONTEXT_IDENTITY_DRAW",
    "CONTEXT_IDENTITY_SOURCE",
    "PREACTIVATION_PERCENTILES",
    "STEP_CONTEXT_DELTA_FIXED_BYTES_PER_ROW",
    "STEP_CONTEXT_DELTA_MAX_BYTES",
    "STEP_CONTEXT_DELTA_SIDECAR_FILENAME",
    "STEP_CONTEXT_DELTA_SIDECAR_SCHEMA",
    "STEP_CONTEXT_DELTA_SIDECAR_SCHEMA_V1",
    "STEP_OBSERVABLES_ENV",
    "StepContextDeltaAccumulator",
    "decode_step_context_deltas",
    "decode_step_decision_moments",
    "step_observables_enabled",
    "summarize_residual_step_observables",
]
