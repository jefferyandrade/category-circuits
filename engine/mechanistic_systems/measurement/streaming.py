"""Layer-blocked production measurement without context-by-route persistence."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from circuitflow.artifacts import (
    ArtifactManifest,
    ArtifactStore,
    ArtifactType,
    PayloadRef,
)
from mechanistic_systems.backends.profiling import PhaseProfiler
from mechanistic_systems.core.storage import (
    RouteBlockSpec,
    ordered_block_route_axis_hash,
    rebase_route_blocks,
)
from mechanistic_systems.contracts import MeasurementPlan
from mechanistic_systems.measurement.estimators import (
    posterior_positive_majority_probability,
)
from mechanistic_systems.storage import write_parquet_rows

try:  # Optional for import smoke; required for production measurement.
    import zarr
    from numcodecs import Blosc
except Exception:  # pragma: no cover - environment dependent.
    zarr = None  # type: ignore[assignment]
    Blosc = None  # type: ignore[assignment]


SUFFICIENT_STATISTICS_SCHEMA = "mechanistic_measurement_sufficient_statistics_zarr_v4"
READABLE_SUFFICIENT_STATISTICS_SCHEMAS = frozenset(
    {
        "mechanistic_measurement_sufficient_statistics_zarr_v3",
        SUFFICIENT_STATISTICS_SCHEMA,
    }
)
STATISTIC_DTYPES = {
    "count": np.dtype("uint32"),
    "score_sum": np.dtype("float32"),
    "score_sum_squares": np.dtype("float32"),
    "absolute_score_sum": np.dtype("float32"),
    "absolute_score_sum_squares": np.dtype("float32"),
    "source_score_sum": np.dtype("float32"),
    "source_absolute_score_sum": np.dtype("float32"),
    "positive_count": np.dtype("uint32"),
    "negative_count": np.dtype("uint32"),
    "block_support": np.dtype("uint16"),
}


def _require_zarr() -> None:
    if zarr is None or Blosc is None:
        raise RuntimeError("streamed measurement requires zarr>=2.18,<3 and numcodecs")


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def context_sketch_matrix(
    layer_id: str, *, routes: int, probes: int, seed: int
) -> np.ndarray:
    """Deterministic per-layer Gaussian sketch Ω (routes × probes)/√K.

    Ω is a pure function of (seed, layer_id) — part of the artifact's
    seed law, recorded in the store attrs so any reader reconstructs
    the identical projection.
    """

    if probes < 1 or routes < 1:
        raise ValueError("sketch requires positive routes and probes")
    digest = hashlib.sha256(f"{int(seed)}:{layer_id}".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return (
        generator.standard_normal((int(routes), int(probes)))
        / np.sqrt(float(probes))
    ).astype(np.float64)


@dataclass(frozen=True)
class MeasurementMomentTile:
    """One already-reduced context-block/query-tile measurement packet.

    Producers should reduce raw derivative values on the accelerator and move
    only these compact sufficient-statistic arrays to the writer. Every array has shape
    ``[query_tile, *route_block.shape]``.
    """

    route_block: RouteBlockSpec
    query_ids: tuple[str, ...]
    context_block_id: str
    estimator_id: str
    count: Any
    score_sum: Any
    score_sum_squares: Any
    absolute_score_sum: Any
    absolute_score_sum_squares: Any
    source_score_sum: Any
    source_absolute_score_sum: Any
    positive_count: Any
    negative_count: Any
    block_support: Any
    source_context_count: int | None = None
    # Per-query within-layer second-moment matrix
    # Σ_c x_c x_cᵀ, shape (Q, R, R) float64. Optional and EXCLUDED from
    # ``arrays()``/STATISTIC_DTYPES — it rides only the covariance tee.
    covariance_sum: Any | None = None
    # 4.8: raw per-image rows (n, Q, R) float32 for the context-factor
    # pilot artifact (receipt (o)); optional, tee-only, pilot-cell-scoped.
    context_rows: Any | None = None
    # Query-independent immediate-destination route contributions [n, R].
    # These are emitted by the same route-reducer chunk that emits
    # ``context_rows``; they are never reconstructed from global scores.
    local_context_rows: Any | None = None
    # Optional compact ReLU-regime sufficient statistics on the native route
    # block. The dense context×route×spatial contribution field is consumed
    # before reduction and is never retained here.
    local_regime_statistics: Mapping[str, Any] | None = None

    @classmethod
    def from_values(
        cls,
        values: Any,
        *,
        route_block: RouteBlockSpec,
        query_ids: Sequence[str],
        context_block_id: str,
        estimator_id: str,
        absolute_values: Any | None = None,
        covariance: bool = False,
        context_factor: bool = False,
        context_factor_mode: str = "sketch",
        sketch_probes: int = 64,
        sketch_seed: int = 0,
    ) -> MeasurementMomentTile:
        """Reduce ephemeral ``[context,query,*route_shape]`` values in memory."""

        array = _numpy(values)
        expected = (len(tuple(query_ids)), *route_block.shape)
        if array.ndim != len(expected) + 1 or tuple(array.shape[1:]) != expected:
            raise ValueError(
                f"measurement values must be [context,{expected}], got {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError("measurement values must be finite floating-point arrays")
        wide = array.astype(np.float64, copy=False)
        absolute = np.abs(wide) if absolute_values is None else _numpy(absolute_values)
        if absolute.shape != array.shape or not np.issubdtype(absolute.dtype, np.floating):
            raise ValueError("absolute measurement values must match signed values")
        if not np.isfinite(absolute).all() or np.any(absolute < 0.0):
            raise ValueError("absolute measurement values must be finite and nonnegative")
        absolute_wide = absolute.astype(np.float64, copy=False)
        tile_shape = expected
        covariance_sum = None
        if covariance:
            flat = wide.reshape(wide.shape[0], wide.shape[1], -1)
            covariance_sum = np.einsum(
                "cqr,cqs->qrs", flat, flat, dtype=np.float64
            )
        context_rows = None
        if context_factor:
            flat = wide.reshape(wide.shape[0], wide.shape[1], -1)
            if context_factor_mode == "native":
                # exact rows on the measured route axis (pair granularity
                # for conv) — retainable only at small R.
                context_rows = flat.astype(np.float32)
            elif context_factor_mode == "channel_marginal":
                # exact linear marginalization onto the leading route
                # axis (the pruning-decision axis for conv channels).
                context_rows = (
                    wide.reshape(
                        wide.shape[0],
                        wide.shape[1],
                        int(route_block.shape[0]),
                        -1,
                    )
                    .sum(axis=-1)
                    .astype(np.float32)
                )
            elif context_factor_mode == "sketch":
                # DEFAULT: K-probe random
                # projection at native granularity — estimate class;
                # Ω is deterministic in (sketch_seed, layer_id).
                omega = context_sketch_matrix(
                    route_block.layer_id,
                    routes=int(flat.shape[-1]),
                    probes=int(sketch_probes),
                    seed=int(sketch_seed),
                )
                context_rows = (flat @ omega).astype(np.float32)
            else:
                raise ValueError(
                    f"unknown context-factor mode {context_factor_mode!r}"
                )
        return cls(
            route_block=route_block,
            query_ids=tuple(str(value) for value in query_ids),
            context_block_id=str(context_block_id),
            estimator_id=str(estimator_id),
            count=np.full(tile_shape, int(array.shape[0]), dtype=np.uint32),
            score_sum=np.sum(wide, axis=0, dtype=np.float64).astype(np.float32),
            score_sum_squares=np.sum(wide * wide, axis=0, dtype=np.float64).astype(np.float32),
            absolute_score_sum=np.sum(absolute_wide, axis=0, dtype=np.float64).astype(np.float32),
            absolute_score_sum_squares=np.sum(
                absolute_wide * absolute_wide,
                axis=0,
                dtype=np.float64,
            ).astype(np.float32),
            source_score_sum=np.sum(wide, axis=0, dtype=np.float64).astype(np.float32),
            source_absolute_score_sum=np.sum(
                absolute_wide, axis=0, dtype=np.float64
            ).astype(np.float32),
            positive_count=np.sum(array > 0.0, axis=0, dtype=np.uint32),
            negative_count=np.sum(array < 0.0, axis=0, dtype=np.uint32),
            block_support=np.any(array != 0.0, axis=0).astype(np.uint16),
            source_context_count=int(array.shape[0]),
            covariance_sum=covariance_sum,
            context_rows=context_rows,
            local_context_rows=None,
            local_regime_statistics=None,
        )

    def arrays(self) -> dict[str, np.ndarray]:
        self.route_block.validate()
        if not self.query_ids or len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("measurement tile query IDs must be nonempty and unique")
        if not self.context_block_id or not self.estimator_id:
            raise ValueError("measurement tile context and estimator IDs must be nonempty")
        if self.source_context_count is not None and int(self.source_context_count) <= 0:
            raise ValueError("measurement tile source_context_count must be positive")
        expected = (len(self.query_ids), *self.route_block.shape)
        result: dict[str, np.ndarray] = {}
        for name, dtype in STATISTIC_DTYPES.items():
            array = _numpy(getattr(self, name))
            if tuple(array.shape) != expected:
                raise ValueError(f"measurement statistic {name!r} has shape {array.shape}, expected {expected}")
            if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(f"measurement statistic {name!r} contains non-finite values")
            if np.any(array < 0) and name in {
                "count",
                "score_sum_squares",
                "absolute_score_sum",
                "absolute_score_sum_squares",
                "source_absolute_score_sum",
                "positive_count",
                "negative_count",
                "block_support",
            }:
                raise ValueError(f"measurement statistic {name!r} must be nonnegative")
            cast = array.astype(dtype, copy=False)
            result[name] = cast.reshape(len(self.query_ids), self.route_block.route_count, order="C")
        if np.any(result["positive_count"] > result["count"]):
            raise ValueError("positive_count cannot exceed count")
        if np.any(result["negative_count"] > result["count"]):
            raise ValueError("negative_count cannot exceed count")
        if np.any(result["positive_count"] + result["negative_count"] > result["count"]):
            raise ValueError("signed counts cannot exceed count")
        return result


class LayerBlockedMeasurementAdapter(Protocol):
    adapter_id: str

    def route_blocks(self, plan: MeasurementPlan) -> Sequence[RouteBlockSpec]:
        ...

    def iter_moment_tiles(self, plan: MeasurementPlan) -> Iterable[MeasurementMomentTile]:
        ...

    def scientific_metadata(self) -> Mapping[str, Any]:
        ...


class SufficientStatisticsMeasurementReader:
    def __init__(self, path: str | Path):
        _require_zarr()
        self.path = Path(path)
        self.root = zarr.open_group(str(self.path), mode="r")
        self.schema_name = str(self.root.attrs.get("schema_name", ""))
        if self.schema_name not in READABLE_SUFFICIENT_STATISTICS_SCHEMAS:
            raise ValueError("unsupported sufficient-statistics measurement schema")
        self.statistics = tuple(str(value) for value in self.root.attrs.get("statistics", ()))
        required = set(STATISTIC_DTYPES) - {"absolute_score_sum_squares"}
        if not required.issubset(self.statistics):
            raise ValueError("sufficient-statistics payload omits required v3 statistics")
        self.query_ids = tuple(str(value) for value in self.root.attrs["query_ids"])
        self.context_block_ids = tuple(str(value) for value in self.root.attrs["context_block_ids"])
        self.route_blocks = tuple(
            RouteBlockSpec.from_mapping(dict(value)) for value in self.root.attrs["route_blocks"]
        )
        self.route_axis_hash = ordered_block_route_axis_hash(self.route_blocks)
        if self.route_axis_hash != str(self.root.attrs.get("route_axis_hash", "")):
            raise ValueError("measurement route-axis hash mismatch")
        self.uncertainty_sampling_unit = str(
            self.root.attrs.get("uncertainty_sampling_unit", "declared_context_block")
        )
        self.uncertainty_available = bool(
            self.root.attrs.get(
                "cross_block_uncertainty_available",
                len(self.context_block_ids) >= 2,
            )
        )
        # Per-context-block statistics retained only when the
        # materializer teed them (older stores: pooled-only, no retrofit).
        self.conditional_available = bool(
            self.root.attrs.get("conditional_statistics_available", False)
        )
        # Within-layer covariance blocks (Σ x xᵀ per query, float64).
        self.covariance_available = bool(
            self.root.attrs.get("covariance_blocks_available", False)
        )
        # 4.8: raw per-image rows for the context-factor pilot artifact.
        self.context_factor_available = bool(
            self.root.attrs.get("context_factor_available", False)
        )
        # Row-axis semantics (sketch default):
        # native = measured route axis (exact); channel_marginal = exact
        # marginalization onto the decision axis; sketch = K-probe
        # estimate with Ω deterministic in (seed, layer_id).
        self.context_factor_mode = str(
            self.root.attrs.get("context_factor_mode", "")
        )
        self.context_factor_sketch_probes = int(
            self.root.attrs.get("context_factor_sketch_probes", 0) or 0
        )
        self.context_factor_sketch_seed = int(
            self.root.attrs.get("context_factor_sketch_seed", 0) or 0
        )
        # Layers that actually retained factor rows, in chain order. A
        # block whose route kind has no per-image reducer (e.g. the
        # linear_weight readout) is legitimately absent — its factor was
        # skipped, not failed — so cross-layer views iterate THIS set
        # rather than all route blocks.
        _factor_span_layers = set(
            self.root.attrs.get("context_row_spans", {}) or {}
        )
        self.context_factor_layers = tuple(
            block.layer_id
            for block in self.route_blocks
            if block.layer_id in _factor_span_layers
        )

    def projected_route_axis_hash(
        self,
        *,
        excluded_layer_ids: Sequence[str],
    ) -> str:
        """Hash an explicitly declared projection such as fixed-readout removal."""

        excluded = {str(value) for value in excluded_layer_ids}
        available = {block.layer_id for block in self.route_blocks}
        if not excluded or not excluded.issubset(available):
            raise ValueError("route-axis projection must exclude declared measured layers")
        selected = tuple(
            block for block in self.route_blocks if block.layer_id not in excluded
        )
        if not selected:
            raise ValueError("route-axis projection removed every measured layer")
        projected = rebase_route_blocks(selected)
        return ordered_block_route_axis_hash(projected)

    def read_block(self, layer_id: str) -> dict[str, np.ndarray]:
        index = next(
            (value for value, block in enumerate(self.route_blocks) if block.layer_id == str(layer_id)),
            None,
        )
        if index is None:
            raise KeyError(layer_id)
        group = self.root[f"blocks/block_{index:04d}"]
        return {
            name: np.asarray(group[name])
            for name in self.statistics
            if name in group
        }

    def derived_covariance_block(self, layer_id: str) -> dict[str, np.ndarray]:
        """Within-layer covariance (ddof=1) per query: (Q, R, R) float64.

        cov_q = (Σ x xᵀ − n μ_q μ_qᵀ) / (n − 1) with μ from the pooled
        first moments — the exact streaming form;
        symmetrized against fp asymmetry. Requires ≥ 2 samples.
        """

        if not self.covariance_available:
            raise ValueError(
                "this measurement store did not retain covariance blocks "
                "(materialize with retain_covariance_blocks)"
            )
        index = next(
            (value for value, block in enumerate(self.route_blocks) if block.layer_id == str(layer_id)),
            None,
        )
        if index is None:
            raise KeyError(layer_id)
        group = self.root[f"blocks/block_{index:04d}"]
        second = np.asarray(group["covariance_sum"], dtype=np.float64)
        return self._covariance_from_second_moment(layer_id, second)

    def derived_covariance_block_from_factor(
        self, layer_id: str
    ) -> dict[str, np.ndarray]:
        """The 4.4 ladder made executable: within-layer covariance as a
        DERIVED VIEW of the context factor.

        Span-ordered float64 accumulation over the losslessly-retained
        f32 rows reproduces the streaming tee's per-tile einsum order,
        so on a store carrying both representations this is BITWISE
        equal to ``derived_covariance_block`` (test-certified).
        """

        if self.context_factor_available and self.context_factor_mode != "native":
            raise ValueError(
                "the bitwise factor-derived covariance view requires the "
                f"NATIVE row axis; this store retains "
                f"{self.context_factor_mode!r} rows"
            )
        rows, spans = self.read_context_rows(layer_id)
        wide = np.asarray(rows, dtype=np.float64)
        queries, routes = wide.shape[1], wide.shape[2]
        second = np.zeros((queries, routes, routes), dtype=np.float64)
        for _block_id, (start, stop) in sorted(
            spans.items(), key=lambda item: item[1][0]
        ):
            chunk = wide[start:stop]
            second += np.einsum("cqr,cqs->qrs", chunk, chunk, dtype=np.float64)
        return self._covariance_from_second_moment(layer_id, second)

    def _covariance_from_second_moment(
        self, layer_id: str, second: np.ndarray
    ) -> dict[str, np.ndarray]:
        # IMAGE-level bookkeeping, deliberately: the retained second
        # moments are per-image outer products, so the mean and sample
        # count must be per-image too. `score_sum`/`count` carry
        # BLOCK-MEAN semantics in production stores (count = #blocks);
        # `source_score_sum` + source_context_count carry the exact
        # per-image sums in both toy and production stores.
        values = self.read_block(layer_id)
        source_counts = self.root.attrs.get("source_context_count_by_block", {})
        if source_counts:
            image_count = float(sum(int(v) for v in source_counts.values()))
        else:
            image_count = float(
                values["count"].astype(np.float64).reshape(-1)[0]
            )
        samples = np.full(second.shape[0], image_count, dtype=np.float64)
        if np.any(samples < 2):
            raise ValueError("covariance requires at least two samples per query")
        mean = (
            values["source_score_sum"]
            .astype(np.float64)
            .reshape(second.shape[0], -1)
            / samples[:, None]
        )
        centered = second - samples[:, None, None] * np.einsum(
            "qr,qs->qrs", mean, mean
        )
        covariance = centered / (samples - 1.0)[:, None, None]
        covariance = 0.5 * (covariance + covariance.transpose(0, 2, 1))
        return {"covariance": covariance, "mean": mean, "count": samples}

    def read_context_rows(
        self, layer_id: str
    ) -> tuple[np.ndarray, dict[str, list[int]]]:
        """Raw per-image rows (N, Q, R_layer) float32 + per-block spans."""

        if not self.context_factor_available:
            raise ValueError(
                "this measurement store did not retain the context factor "
                "(materialize with retain_context_factor)"
            )
        index = next(
            (value for value, block in enumerate(self.route_blocks) if block.layer_id == str(layer_id)),
            None,
        )
        if index is None:
            raise KeyError(layer_id)
        rows = np.asarray(self.root[f"blocks/block_{index:04d}/context_rows"])
        spans = dict(self.root.attrs["context_row_spans"][str(layer_id)])
        return rows, spans

    def read_conditional_block(self, layer_id: str) -> dict[str, np.ndarray]:
        """Per-context-block statistics for one layer: name -> (C, Q, R)."""

        if not self.conditional_available:
            raise ValueError(
                "this measurement store did not retain per-context-block "
                "statistics (materialize with retain_conditional_statistics)"
            )
        index = next(
            (value for value, block in enumerate(self.route_blocks) if block.layer_id == str(layer_id)),
            None,
        )
        if index is None:
            raise KeyError(layer_id)
        group = self.root[f"blocks/block_{index:04d}"]
        return {
            name: np.asarray(group[f"conditional_{name}"])
            for name in self.statistics
            if f"conditional_{name}" in group
        }

    def derived_conditional_moments(
        self,
        layer_id: str,
        groups: Mapping[str, Sequence[str]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Per-group (per-stratum) derived moments from retained block rows.

        ``groups`` maps group id -> member context-block ids (a partition or
        any subset family). Each group's sufficient statistics are the SUM of
        its members' rows — identical algebra to the pooled store restricted
        to the group — and the derived moments reuse ``derived_block``'s
        exact math. Groups with a single member get point moments only
        (uncertainty fields refused, mirroring the pooled single-block rule).
        """

        conditional = self.read_conditional_block(layer_id)
        position = {
            context_id: index
            for index, context_id in enumerate(self.context_block_ids)
        }
        result: dict[str, dict[str, np.ndarray]] = {}
        for group_id, members in groups.items():
            member_ids = tuple(str(value) for value in members)
            if not member_ids:
                raise ValueError(f"conditional group {group_id!r} is empty")
            unknown = [m for m in member_ids if m not in position]
            if unknown:
                raise ValueError(
                    f"conditional group {group_id!r} names unknown context "
                    f"blocks {unknown!r}"
                )
            rows = [position[m] for m in member_ids]
            values = {
                name: array[rows].sum(axis=0)
                for name, array in conditional.items()
            }
            result[group_id] = self._derived_from_values(
                values, uncertainty=len(member_ids) >= 2
            )
        return result

    def derived_block(self, layer_id: str) -> dict[str, np.ndarray]:
        values = self.read_block(layer_id)
        return self._derived_from_values(
            values, uncertainty=self.uncertainty_available
        )

    def _derived_from_values(
        self, values: dict[str, np.ndarray], *, uncertainty: bool
    ) -> dict[str, np.ndarray]:
        count = values["count"].astype(np.float64)
        score_sum = values["score_sum"].astype(np.float64)
        mean = np.divide(score_sum, count, out=np.zeros_like(score_sum), where=count > 0)
        absolute_mean = np.divide(
            values["absolute_score_sum"].astype(np.float64),
            count,
            out=np.zeros_like(score_sum),
            where=count > 0,
        )
        result = {
            "score_mean": mean,
            "absolute_score_mean": absolute_mean,
            # A hard partition is a deterministic thresholded view of the
            # signed aggregate and does not imply repeated-block certainty.
            "hard_sign": np.sign(mean).astype(np.int8),
        }
        if not uncertainty:
            return result
        score_variance = np.divide(
            values["score_sum_squares"].astype(np.float64) - count * mean * mean,
            count - 1.0,
            out=np.zeros_like(score_sum),
            where=count > 1,
        )
        signed_count = (
            values["positive_count"].astype(np.float64)
            + values["negative_count"].astype(np.float64)
        )
        probability = posterior_positive_majority_probability(
            values["positive_count"].astype(np.float64),
            values["negative_count"].astype(np.float64),
        )
        result.update({
            "score_variance": np.maximum(score_variance, 0.0),
            "positive_probability": probability,
            "sign_stability": np.maximum(probability, 1.0 - probability),
            "hard_sign": np.where(
                signed_count == 0,
                0,
                np.where(probability >= 0.5, 1, -1),
            ).astype(np.int8),
        })
        if "absolute_score_sum_squares" in values:
            absolute_variance = np.divide(
                values["absolute_score_sum_squares"].astype(np.float64)
                - count * absolute_mean * absolute_mean,
                count - 1.0,
                out=np.zeros_like(score_sum),
                where=count > 1,
            )
            result["absolute_score_variance"] = np.maximum(
                absolute_variance,
                0.0,
            )
        return result

    def source_image_aggregate_block(
        self,
        layer_id: str,
        *,
        score_head: str,
    ) -> np.ndarray:
        """Read the exact image-sum head corresponding to a legacy ScoreStore.

        Canonical block statistics retain block means for dependency-aware
        inference while separate source aggregates preserve the exact image
        sums written by historical ScoreStores.
        """

        statistic = {
            "route_scores": "source_score_sum",
            "route_abs_scores": "source_absolute_score_sum",
        }.get(str(score_head))
        if statistic is None:
            raise ValueError(f"unsupported historical score head {score_head!r}")
        return self.read_block(layer_id)[statistic].astype(np.float64)

    def source_image_aggregate_query_block(
        self,
        layer_id: str,
        *,
        query_index: int,
        score_head: str,
    ) -> np.ndarray:
        """Read one exact query slice without materializing every query/statistic.

        This is the canonical path for deep ScoreStore projection audits. Large
        linear route blocks can contain tens of millions of coordinates, so a
        query-local read avoids multiplying host memory by the query-bank size.
        """

        statistic = {
            "route_scores": "source_score_sum",
            "route_abs_scores": "source_absolute_score_sum",
        }.get(str(score_head))
        if statistic is None:
            raise ValueError(f"unsupported historical score head {score_head!r}")
        if not 0 <= int(query_index) < len(self.query_ids):
            raise IndexError(query_index)
        block_index = next(
            (
                index
                for index, block in enumerate(self.route_blocks)
                if block.layer_id == str(layer_id)
            ),
            None,
        )
        if block_index is None:
            raise KeyError(layer_id)
        values = self.root[
            f"blocks/block_{block_index:04d}/{statistic}"
        ][int(query_index), :]
        return np.asarray(values, dtype=np.float32)


def materialize_streamed_measurement_artifact(
    plan: MeasurementPlan,
    adapter: LayerBlockedMeasurementAdapter,
    store: ArtifactStore,
    *,
    staging_dir: str | Path,
    query_chunk: int = 8,
    route_chunk: int = 262_144,
    runtime_profiler: PhaseProfiler | None = None,
    retain_conditional_statistics: bool = False,
    retain_covariance_blocks: bool = False,
    retain_context_factor: bool = False,
    context_factor_mode: str = "sketch",
    sketch_probes: int = 64,
    sketch_seed: int = 0,
) -> tuple[ArtifactManifest, Path]:
    """Stream reduced tiles into one compact, coordinate-aware measurement.

    Additive option: with
    ``retain_conditional_statistics`` the per-context-block tier is teed
    into sibling ``conditional_<statistic>`` datasets with a leading
    context-block axis, in the SAME pass. The pooled v1 arrays are
    byte-identical with the flag on or off (the tee only adds datasets);
    per-group (per-stratum) moments then derive from any grouping of the
    retained block rows. Default off = today's artifact exactly.
    """

    _require_zarr()
    profiler = runtime_profiler or PhaseProfiler()
    plan.validate(claim_bearing=plan.status == "frozen").require_valid()
    query_ids = tuple(query.query_id for query in plan.queries)
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    context_ids = tuple(plan.context_block_ids)
    blocks = tuple(adapter.route_blocks(plan))
    metadata_factory = getattr(adapter, "scientific_metadata", None)
    producer_metadata = (
        dict(metadata_factory())
        if callable(metadata_factory)
        else {"adapter_id": str(getattr(adapter, "adapter_id", "unspecified"))}
    )
    if plan.status == "frozen" and producer_metadata.get("claim_capture_certified") is not True:
        raise ValueError(
            "claim-bearing streamed measurement requires a certified capture adapter"
        )
    route_axis_hash = ordered_block_route_axis_hash(blocks)
    block_index = {block.layer_id: index for index, block in enumerate(blocks)}
    coverage = {
        (block.layer_id, context_id): np.zeros(len(query_ids), dtype=bool)
        for block in blocks
        for context_id in context_ids
    }
    source_context_counts: dict[str, int] = {}
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".streamed-measurement-", dir=staging) as temporary:
        initialization_started = time.perf_counter()
        temporary_path = Path(temporary)
        zarr_path = temporary_path / "measurement_sufficient_statistics.zarr"
        root = zarr.open_group(str(zarr_path), mode="w")
        root.attrs.update(
            {
                "schema_name": SUFFICIENT_STATISTICS_SCHEMA,
                "query_ids": list(query_ids),
                "context_block_ids": list(context_ids),
                "estimator_id": plan.estimator_id,
                "route_axis_hash": route_axis_hash,
                "route_blocks": [block.to_dict() for block in blocks],
                "statistics": list(STATISTIC_DTYPES),
                "accumulation_dtype": "float64",
                "storage_dtypes": {name: dtype.name for name, dtype in STATISTIC_DTYPES.items()},
                "raw_context_by_route_persisted": False,
                "conditional_statistics_available": bool(
                    retain_conditional_statistics
                ),
                "covariance_blocks_available": bool(retain_covariance_blocks),
                "context_factor_available": bool(retain_context_factor),
                "context_factor_mode": (
                    str(context_factor_mode) if retain_context_factor else ""
                ),
                "context_factor_sketch_probes": (
                    int(sketch_probes)
                    if retain_context_factor and context_factor_mode == "sketch"
                    else 0
                ),
                "context_factor_sketch_seed": (
                    int(sketch_seed)
                    if retain_context_factor and context_factor_mode == "sketch"
                    else 0
                ),
                "inferential_sampling_unit": "declared_context_block",
                "uncertainty_sampling_unit": "declared_context_block",
                "cross_block_uncertainty_available": len(context_ids) >= 2,
                "unavailable_statistics": (
                    []
                    if len(context_ids) >= 2
                    else [
                        "score_variance",
                        "absolute_score_variance",
                        "positive_probability",
                        "sign_stability",
                    ]
                ),
                "statistic_semantics": {
                    "score_sum": "sum_of_declared_context_block_means",
                    "score_sum_squares": "sum_of_squared_declared_context_block_means",
                    "absolute_score_sum": "sum_of_declared_context_block_absolute_means",
                    "absolute_score_sum_squares": (
                        "sum_of_squared_declared_context_block_absolute_means"
                    ),
                    "positive_count": "positive_declared_context_block_count",
                    "negative_count": "negative_declared_context_block_count",
                    "source_score_sum": "exact_sum_of_per_image_signed_route_derivatives",
                    "source_absolute_score_sum": (
                        "exact_sum_of_absolute_per_image_route_derivatives"
                    ),
                },
                "context_block_value_semantics": "mean_exact_signed_and_absolute_image_contribution",
                "legacy_scorestore_aggregate_view": (
                    "exact_source_score_sum_and_source_absolute_score_sum"
                ),
            }
        )
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        arrays: dict[tuple[str, str], Any] = {}
        groups = root.require_group("blocks")
        for index, block in enumerate(blocks):
            group = groups.require_group(f"block_{index:04d}")
            group.attrs.update(block.to_dict())
            for name, dtype in STATISTIC_DTYPES.items():
                arrays[(block.layer_id, name)] = group.create_dataset(
                    name,
                    shape=(len(query_ids), block.route_count),
                    chunks=(
                        min(max(1, int(query_chunk)), len(query_ids)),
                        min(max(1, int(route_chunk)), block.route_count),
                    ),
                    dtype=dtype,
                    compressor=compressor,
                    fill_value=0,
                    overwrite=False,
                )
                if retain_conditional_statistics:
                    arrays[(block.layer_id, f"conditional_{name}")] = (
                        group.create_dataset(
                            f"conditional_{name}",
                            shape=(
                                len(context_ids),
                                len(query_ids),
                                block.route_count,
                            ),
                            chunks=(
                                1,
                                min(max(1, int(query_chunk)), len(query_ids)),
                                min(
                                    max(1, int(route_chunk)),
                                    block.route_count,
                                ),
                            ),
                            dtype=dtype,
                            compressor=compressor,
                            fill_value=0,
                            overwrite=False,
                        )
                    )
            if retain_covariance_blocks:
                flat_routes = int(np.prod(block.shape))
                arrays[(block.layer_id, "covariance_sum")] = (
                    group.create_dataset(
                        "covariance_sum",
                        shape=(len(query_ids), flat_routes, flat_routes),
                        chunks=(1, flat_routes, flat_routes),
                        dtype=np.float64,
                        compressor=compressor,
                        fill_value=0,
                        overwrite=False,
                    )
                )
            if retain_context_factor:
                if context_factor_mode == "native":
                    row_width = int(np.prod(block.shape))
                elif context_factor_mode == "channel_marginal":
                    row_width = int(block.shape[0])
                elif context_factor_mode == "sketch":
                    row_width = int(sketch_probes)
                else:
                    raise ValueError(
                        f"unknown context-factor mode {context_factor_mode!r}"
                    )
                arrays[(block.layer_id, "context_rows")] = (
                    group.create_dataset(
                        "context_rows",
                        shape=(0, len(query_ids), row_width),
                        chunks=(64, len(query_ids), min(row_width, 262_144)),
                        dtype=np.float32,
                        compressor=compressor,
                        overwrite=False,
                    )
                )
        context_row_spans: dict[str, dict[str, list[int]]] = {}
        context_position = {
            context_id: index for index, context_id in enumerate(context_ids)
        }
        profiler.record(
            "measurement.stream.zarr_initialization",
            time.perf_counter() - initialization_started,
        )

        tile_count = 0
        tile_iterator = iter(adapter.iter_moment_tiles(plan))
        while True:
            production_started = time.perf_counter()
            try:
                tile = next(tile_iterator)
            except StopIteration:
                profiler.record(
                    "measurement.stream.tile_production",
                    time.perf_counter() - production_started,
                )
                break
            profiler.record(
                "measurement.stream.tile_production",
                time.perf_counter() - production_started,
            )
            write_started = time.perf_counter()
            if tile.estimator_id != plan.estimator_id:
                raise ValueError("measurement tile estimator does not match its plan")
            if tile.route_block.layer_id not in block_index:
                raise ValueError(f"measurement tile has unknown route block {tile.route_block.layer_id!r}")
            expected_block = blocks[block_index[tile.route_block.layer_id]]
            if tile.route_block.to_dict() != expected_block.to_dict():
                raise ValueError("measurement tile route block differs from the declared axis")
            if tile.context_block_id not in context_ids:
                raise ValueError("measurement tile has an undeclared context block")
            source_count = int(tile.source_context_count or 0)
            if source_count <= 0:
                raise ValueError("measurement tile must declare its source context count")
            prior_source_count = source_context_counts.get(tile.context_block_id)
            if prior_source_count is not None and prior_source_count != source_count:
                raise ValueError("measurement source context count changed within a declared block")
            source_context_counts[tile.context_block_id] = source_count
            indices = tuple(query_index.get(query_id, -1) for query_id in tile.query_ids)
            if any(index < 0 for index in indices):
                raise ValueError("measurement tile contains an undeclared query")
            if indices != tuple(range(indices[0], indices[0] + len(indices))):
                raise ValueError("measurement query tiles must be contiguous in canonical order")
            seen = coverage[(tile.route_block.layer_id, tile.context_block_id)]
            if np.any(seen[list(indices)]):
                raise ValueError("measurement tile duplicates query/context/route-block coverage")
            tile_arrays = tile.arrays()
            query_slice = slice(indices[0], indices[-1] + 1)
            for name, values in tile_arrays.items():
                target = arrays[(tile.route_block.layer_id, name)]
                existing = np.asarray(target[query_slice, :])
                target[query_slice, :] = existing + values
                if retain_conditional_statistics:
                    conditional = arrays[
                        (tile.route_block.layer_id, f"conditional_{name}")
                    ]
                    position = context_position[tile.context_block_id]
                    prior = np.asarray(conditional[position, query_slice, :])
                    conditional[position, query_slice, :] = prior + values
            if retain_covariance_blocks:
                if tile.covariance_sum is None:
                    raise ValueError(
                        "covariance retention requires tiles built with "
                        "covariance=True; refusing a partial covariance store"
                    )
                cov_target = arrays[(tile.route_block.layer_id, "covariance_sum")]
                cov_values = np.asarray(tile.covariance_sum, dtype=np.float64)
                prior_cov = np.asarray(cov_target[query_slice, :, :])
                cov_target[query_slice, :, :] = prior_cov + cov_values
            if retain_context_factor and tile.context_rows is not None:
                # A None factor here is legitimate: the block's route kind
                # has no per-image reducer (e.g. the linear_weight
                # readout), so the accumulator omitted it. Such a layer's
                # context_rows dataset simply stays empty; the reader
                # reports it via ``context_factor_layers`` and the
                # cross-layer views skip it.
                if tuple(tile.query_ids) != tuple(query_ids):
                    # The rows dataset is append-only along the image axis
                    # with the FULL query axis per row; a query-tiled
                    # producer would interleave partial rows. Run factor
                    # pilots with query_tile_size >= len(queries).
                    raise ValueError(
                        "context-factor retention requires full-query "
                        "tiles; got a tile covering "
                        f"{len(tile.query_ids)}/{len(query_ids)} queries"
                    )
                rows_target = arrays[(tile.route_block.layer_id, "context_rows")]
                rows = np.asarray(tile.context_rows, dtype=np.float32)
                start = int(rows_target.shape[0])
                rows_target.append(rows)
                context_row_spans.setdefault(
                    tile.route_block.layer_id, {}
                )[tile.context_block_id] = [start, start + int(rows.shape[0])]
            seen[list(indices)] = True
            tile_count += 1
            profiler.record(
                "measurement.stream.zarr_tile_write",
                time.perf_counter() - write_started,
            )
        missing = [
            f"{layer_id}:{context_id}"
            for (layer_id, context_id), seen in coverage.items()
            if not bool(np.all(seen))
        ]
        if missing:
            raise ValueError("measurement adapter omitted frozen coverage: " + ",".join(missing[:8]))
        root.attrs["source_context_count_by_block"] = dict(source_context_counts)
        if retain_context_factor:
            root.attrs["context_row_spans"] = {
                layer: dict(spans) for layer, spans in context_row_spans.items()
            }

        metadata_started = time.perf_counter()
        summary_path = write_parquet_rows(
            temporary_path / "measurement_summary.parquet",
            [
                {
                    "query_id": query_id,
                    "context_block_count": len(context_ids),
                    "route_block_count": len(blocks),
                    "total_route_count": int(sum(block.route_count for block in blocks)),
                    "estimator_id": plan.estimator_id,
                    "raw_context_by_route_persisted": False,
                    "inferential_sampling_unit": "declared_context_block",
                    "source_context_count": int(sum(source_context_counts.values())),
                }
                for query_id in query_ids
            ],
        )
        blocks_path = write_parquet_rows(
            temporary_path / "route_blocks.parquet",
            [
                {
                    "block_index": index,
                    "block_id": block.block_id,
                    "layer_id": block.layer_id,
                    "route_kind": block.route_kind,
                    "shape_json": json.dumps(list(block.shape), separators=(",", ":")),
                    "offset": block.offset,
                    "route_count": block.route_count,
                }
                for index, block in enumerate(blocks)
            ],
        )
        generated = (zarr_path, summary_path, blocks_path)
        payloads = tuple(
            PayloadRef.from_path(
                name=path.name,
                path=path,
                uri=f"payloads/{path.name}",
                storage_format=("zarr_v2" if path == zarr_path else "parquet"),
                media_type=(
                    "application/vnd+zarr"
                    if path == zarr_path
                    else "application/vnd.apache.parquet"
                ),
            )
            for path in generated
        )
        manifest = ArtifactManifest(
            artifact_type=ArtifactType.MEASUREMENT,
            study_hash=plan.study_hash,
            plan_hash=plan.content_hash,
            parent_hashes=tuple(),
            axes={"query": query_ids, "route": tuple(block.block_id for block in blocks)},
            coordinate_schema_hash=route_axis_hash,
            semantics={
                "persistence_policy": "sufficient_statistics_only",
                "measurement_semantics": plan.measurement_semantics,
                "estimator_id": plan.estimator_id,
                "estimator_version": plan.estimator_version,
                "compact_statistics": list(STATISTIC_DTYPES),
                "context_block_ids": list(context_ids),
                "sign_representation": (
                    "cross_block_probability_with_thresholded_diagnostic"
                    if len(context_ids) >= 2
                    else "aggregate_hard_sign_uncertainty_unavailable"
                ),
                "head_storage": "implicit_route_blocks_query_by_route_zarr",
                "route_axis_hash": route_axis_hash,
                "route_block_count": len(blocks),
                "total_route_count": int(sum(block.route_count for block in blocks)),
                "tile_count": tile_count,
                "raw_context_by_route_persisted": False,
                "inferential_sampling_unit": "declared_context_block",
                "cross_block_uncertainty_available": len(context_ids) >= 2,
                "unavailable_statistics": (
                    []
                    if len(context_ids) >= 2
                    else [
                        "score_variance",
                        "absolute_score_variance",
                        "positive_probability",
                        "sign_stability",
                    ]
                ),
                "source_context_count_by_block": dict(source_context_counts),
                "context_block_value_semantics": "mean_exact_signed_and_absolute_image_contribution",
                "legacy_scorestore_aggregate_view": (
                    "exact_source_score_sum_and_source_absolute_score_sum"
                ),
                "producer": producer_metadata,
            },
            payloads=payloads,
            unit_ids=context_ids,
            source_kind="native",
            valid_for_claims=plan.status == "frozen",
        )
        profiler.record(
            "measurement.stream.metadata_views",
            time.perf_counter() - metadata_started,
        )
        commit_started = time.perf_counter()
        store.seal_contract(
            contract_kind="measurement_plan",
            content_hash=plan.content_hash,
            payload=plan.to_dict(),
        )
        bundle = store.write_bundle(
            manifest,
            payload_files={path.name: path for path in generated},
            link_payloads=True,
        )
        profiler.record(
            "measurement.stream.artifact_commit",
            time.perf_counter() - commit_started,
        )
    certification_started = time.perf_counter()
    receipt = store.certify(manifest, validator_id="mechanistic_streamed_measurement.v1")
    profiler.record(
        "measurement.stream.artifact_certification",
        time.perf_counter() - certification_started,
    )
    if not receipt.passed:
        raise ValueError("streamed measurement certification failed: " + "; ".join(receipt.issues))
    return manifest, bundle


__all__ = [
    "LayerBlockedMeasurementAdapter",
    "MeasurementMomentTile",
    "STATISTIC_DTYPES",
    "SUFFICIENT_STATISTICS_SCHEMA",
    "SufficientStatisticsMeasurementReader",
    "context_sketch_matrix",
    "materialize_streamed_measurement_artifact",
]
