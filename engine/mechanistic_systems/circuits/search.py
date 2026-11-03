"""Native nested rate-distortion circuit identification.

The search consumes canonical route-score moments and delegates only masked
model evaluation to an architecture adapter.  It constructs one deterministic
nested route order per query and partition family; every density on the path
is therefore a true subset of the next density rather than an independently
optimized mask.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from circuitflow.artifacts import (
    ArtifactManifest,
    ArtifactStore,
    ArtifactType,
)
from mechanistic_systems.circuits.rate_distortion import (
    CircuitCandidate,
    nested_rate_distortion_frontier,
    select_circuit,
)
from mechanistic_systems.core.storage import (
    BlockBitPackedCircuitSet,
    RouteBlockSpec,
    materialize_circuit_artifact,
    ordered_block_route_axis_hash,
    rebase_route_blocks,
)
from mechanistic_systems.contracts import (
    CANONICAL_PARTITION_POLICIES,
    CircuitPlan,
)
from mechanistic_systems.measurement.streaming import (
    SufficientStatisticsMeasurementReader,
)


@dataclass(frozen=True)
class CircuitEvaluation:
    response_distortion: float
    target_preservation: float
    context_count: int
    metrics: Mapping[str, float]

    def validate(self) -> None:
        if not np.isfinite(self.response_distortion) or self.response_distortion < 0.0:
            raise ValueError("circuit response distortion must be finite and nonnegative")
        if not np.isfinite(self.target_preservation):
            raise ValueError("circuit target preservation must be finite")
        if int(self.context_count) <= 0:
            raise ValueError("circuit evaluation requires at least one context")
        if any(not np.isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError("circuit evaluation metrics must be finite")


class CircuitEvaluator(Protocol):
    evaluator_id: str
    query_ids: tuple[str, ...]
    route_blocks: tuple[RouteBlockSpec, ...]
    claim_execution_certified: bool

    def evaluate(
        self,
        plan: CircuitPlan,
        *,
        query_id: str,
        masks_by_layer: Mapping[str, np.ndarray],
        split: str,
    ) -> CircuitEvaluation:
        ...


class PartitionLabelProvider(Protocol):
    provider_id: str
    supported_policies: frozenset[str]
    certified_matching_dimensions: frozenset[str]

    def labels(
        self,
        *,
        policy_id: str,
        query_id: str,
        query_index: int,
        score_mean: np.ndarray,
        score_magnitude: np.ndarray,
        route_blocks: Sequence[RouteBlockSpec],
        random_seed: int,
    ) -> np.ndarray:
        ...


def circuit_route_blocks_from_measurement(
    plan: CircuitPlan,
    measurement_blocks: Sequence[RouteBlockSpec],
) -> tuple[RouteBlockSpec, ...]:
    """Project measured routes onto the declared prunable circuit universe."""

    measured = tuple(measurement_blocks)
    measured_ids = tuple(block.layer_id for block in measured)
    fixed = tuple(plan.fixed_measurement_layer_ids)
    unknown = sorted(set(fixed) - set(measured_ids))
    if unknown:
        raise ValueError(
            "CircuitPlan fixed measurement layers are absent: " + ",".join(unknown)
        )
    selected = rebase_route_blocks(
        tuple(block for block in measured if block.layer_id not in set(fixed))
    )
    if ordered_block_route_axis_hash(selected) != plan.route_universe_hash:
        raise ValueError(
            "prunable measurement projection differs from the CircuitPlan route universe"
        )
    return selected


@dataclass(frozen=True)
class CircuitSearchResult:
    circuits: BlockBitPackedCircuitSet
    candidates_by_query: Mapping[str, tuple[CircuitCandidate, ...]]
    frontier_by_query: Mapping[str, tuple[CircuitCandidate, ...]]
    selected_by_query: Mapping[str, CircuitCandidate]
    heldout_evaluation_by_query: Mapping[str, CircuitEvaluation]
    partition_policy: str
    search_algorithm: str = "global_nested_rate_distortion"
    search_telemetry_by_query: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def preservation_by_query(self) -> dict[str, float]:
        return {
            query_id: float(result.target_preservation)
            for query_id, result in self.heldout_evaluation_by_query.items()
        }

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for query_id, candidates in self.candidates_by_query.items():
            frontier_ids = {candidate.realization_id for candidate in self.frontier_by_query[query_id]}
            selected_id = self.selected_by_query[query_id].realization_id
            heldout = self.heldout_evaluation_by_query[query_id]
            for candidate in candidates:
                rows.append(
                    {
                        "query_id": query_id,
                        "realization_id": candidate.realization_id,
                        "partition_policy": self.partition_policy,
                        "search_algorithm": self.search_algorithm,
                        "retained_route_count": candidate.retained_route_count,
                        "route_cost": candidate.route_cost,
                        "optimization_response_distortion": candidate.response_distortion,
                        "optimization_target_preservation": candidate.target_preservation,
                        "on_nondominated_frontier": candidate.realization_id in frontier_ids,
                        "selected": candidate.realization_id == selected_id,
                        "heldout_response_distortion": (
                            heldout.response_distortion if candidate.realization_id == selected_id else np.nan
                        ),
                        "heldout_target_preservation": (
                            heldout.target_preservation if candidate.realization_id == selected_id else np.nan
                        ),
                    }
                )
        return rows


def _retained_count_path(plan: CircuitPlan, route_count: int) -> tuple[int, ...]:
    budget = dict(plan.mask_budget)
    counts: list[int] = []
    if budget.get("frontier_retained_counts") is not None:
        counts.extend(int(value) for value in budget["frontier_retained_counts"])
    if budget.get("frontier_retained_fractions") is not None:
        counts.extend(
            max(1, int(round(float(value) * route_count)))
            for value in budget["frontier_retained_fractions"]
        )
    if not counts and plan.status != "frozen":
        final = int(budget.get("routes", route_count))
        counts.extend(max(1, int(round(final * fraction))) for fraction in (0.25, 0.5, 0.75, 1.0))
    if not counts:
        raise ValueError("frozen circuit search requires an explicit retained-count frontier path")
    normalized = tuple(sorted({min(route_count, max(1, value)) for value in counts}))
    if not normalized:
        raise ValueError("circuit frontier retained-count path is empty")
    return normalized


def _block_index_axis(route_blocks: Sequence[RouteBlockSpec]) -> np.ndarray:
    return np.concatenate(
        [np.full(block.route_count, index, dtype=np.int16) for index, block in enumerate(route_blocks)]
    )


def _quantile_bins(values: np.ndarray, *, bin_count: int = 10) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("magnitude binning requires a vector")
    if len(values) == 0:
        return np.empty(0, dtype=np.int16)
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, int(bin_count) + 1))
    interior = np.unique(quantiles[1:-1])
    return np.searchsorted(interior, values, side="right").astype(np.int16)


def _default_partition_labels(
    *,
    policy_id: str,
    score_mean: np.ndarray,
    score_magnitude: np.ndarray,
    route_blocks: Sequence[RouteBlockSpec],
    random_seed: int,
) -> np.ndarray:
    policy = str(policy_id)
    route_count = int(score_mean.size)
    if policy == "unpartitioned":
        return np.zeros(route_count, dtype=np.int8)
    signs = np.where(score_mean > 0.0, 0, np.where(score_mean < 0.0, 1, 2)).astype(np.int8)
    if policy == "attributional_sign":
        return signs
    rng = np.random.default_rng(int(random_seed))
    if policy == "random_count_matched":
        return signs[rng.permutation(route_count)]
    if policy == "magnitude_stratified_random":
        layers = _block_index_axis(route_blocks)
        magnitude_bins = _quantile_bins(score_magnitude)
        shuffled = signs.copy()
        for layer in np.unique(layers):
            layer_mask = layers == layer
            for magnitude_bin in np.unique(magnitude_bins[layer_mask]):
                indices = np.flatnonzero(layer_mask & (magnitude_bins == magnitude_bin))
                shuffled[indices] = shuffled[indices][rng.permutation(len(indices))]
        return shuffled
    raise ValueError(
        f"partition policy {policy!r} requires an explicit coordinate or donor label provider"
    )


def _top_priority_indices(priority: np.ndarray, indices: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    limit = min(int(limit), len(indices))
    local_priority = priority[indices]
    if limit < len(indices):
        selected = np.argpartition(-local_priority, limit - 1)[:limit]
        selected_indices = indices[selected]
    else:
        selected_indices = indices.copy()
    order = np.lexsort((selected_indices, -priority[selected_indices]))
    return selected_indices[order].astype(np.int64, copy=False)


def stratified_nested_route_order(
    priority: np.ndarray,
    group_ids: np.ndarray,
    *,
    maximum_count: int,
) -> np.ndarray:
    """Build one deterministic approximately proportional nested route order."""

    priority = np.asarray(priority, dtype=np.float64).reshape(-1)
    groups = np.asarray(group_ids).reshape(-1)
    if priority.shape != groups.shape or not np.isfinite(priority).all():
        raise ValueError("route priority and partition axes are invalid")
    maximum = min(len(priority), max(1, int(maximum_count)))
    preferred_order = {"positive": 0, "negative": 1, "zero": 2, "all": 3, "0": 0, "1": 1, "2": 2, "3": 3}
    values = tuple(
        sorted(
            np.unique(groups).tolist(),
            key=lambda value: (
                preferred_order.get(str(value), 100),
                str(value),
            ),
        )
    )
    if not values:
        raise ValueError("route partition axis is empty")
    ranked: list[np.ndarray] = []
    sizes: list[int] = []
    for value in values:
        indices = np.flatnonzero(groups == value)
        sizes.append(len(indices))
        ranked.append(_top_priority_indices(priority, indices, min(maximum, len(indices))))
    heap: list[tuple[float, int, int]] = []
    for group_index, size in enumerate(sizes):
        if size:
            heapq.heappush(heap, (1.0 / size, group_index, 0))
    order: list[int] = []
    while heap and len(order) < maximum:
        _fraction, group_index, rank = heapq.heappop(heap)
        if rank >= len(ranked[group_index]):
            continue
        order.append(int(ranked[group_index][rank]))
        next_rank = rank + 1
        if next_rank < len(ranked[group_index]):
            heapq.heappush(
                heap,
                ((next_rank + 1) / sizes[group_index], group_index, next_rank),
            )
    if len(order) != maximum:
        raise RuntimeError("stratified route ordering failed to cover the requested budget")
    return np.asarray(order, dtype=np.int64)


_PROVIDER_REQUIREMENTS: Mapping[str, frozenset[str]] = {
    "route_coordinate_sign": frozenset({"route_coordinate_sign"}),
    "wrong_target_attributional_sign": frozenset(
        {"wrong_target_sign", "current_target_magnitude"}
    ),
    "nuisance_stratified_sign_permutation": frozenset(
        {"layer", "score_magnitude", "activation_energy", "route_degree"}
    ),
}


def _validate_partition_label_provider(
    policy_id: str,
    provider: PartitionLabelProvider | None,
) -> None:
    required = _PROVIDER_REQUIREMENTS.get(policy_id)
    if required is None:
        return
    if provider is None:
        raise ValueError(
            f"partition policy {policy_id!r} requires a certified partition label provider"
        )
    supported = frozenset(str(value) for value in getattr(provider, "supported_policies", ()))
    certified = frozenset(
        str(value) for value in getattr(provider, "certified_matching_dimensions", ())
    )
    if policy_id not in supported:
        raise ValueError(
            f"partition label provider does not declare support for policy {policy_id!r}"
        )
    missing = required - certified
    if missing:
        raise ValueError(
            "partition label provider is missing certified matching dimensions: "
            + ", ".join(sorted(missing))
        )


def _split_mask(mask: np.ndarray, blocks: Sequence[RouteBlockSpec]) -> dict[str, np.ndarray]:
    return {
        block.layer_id: mask[block.offset : block.stop].reshape(block.shape, order="C")
        for block in blocks
    }


def _selection_constraints(plan: CircuitPlan, maximum_count: int) -> tuple[float, float | None, float | None]:
    budget = dict(plan.mask_budget)
    maximum_cost = float(budget.get("maximum_routes", budget.get("routes", maximum_count)))
    maximum_distortion = plan.response_distortion.get("maximum")
    minimum_preservation = plan.response_distortion.get("minimum_preservation")
    return (
        maximum_cost,
        None if maximum_distortion is None else float(maximum_distortion),
        None if minimum_preservation is None else float(minimum_preservation),
    )


def identify_nested_rate_distortion_circuits(
    plan: CircuitPlan,
    measurement_manifest: ArtifactManifest,
    measurement: SufficientStatisticsMeasurementReader,
    evaluator: CircuitEvaluator,
    *,
    partition_label_provider: PartitionLabelProvider | None = None,
) -> CircuitSearchResult:
    """Identify one selected point mask per query on a held-out evaluated path."""

    plan.validate(claim_bearing=plan.status == "frozen").require_valid()
    if measurement_manifest.artifact_type != ArtifactType.MEASUREMENT:
        raise ValueError("circuit identification requires a MeasurementArtifact")
    if measurement_manifest.artifact_hash != plan.measurement_artifact_hash:
        raise ValueError("CircuitPlan measurement parent differs from the supplied artifact")
    if measurement_manifest.study_hash != plan.study_hash:
        raise ValueError("measurement and circuit plan studies differ")
    if plan.status == "frozen" and not measurement_manifest.valid_for_claims:
        raise ValueError("claim-bearing circuit identification requires a claim-valid measurement")
    if plan.status == "frozen" and not evaluator.claim_execution_certified:
        raise ValueError("claim-bearing circuit identification requires certified masked execution")
    if plan.partition_policy not in CANONICAL_PARTITION_POLICIES:
        raise ValueError(f"unsupported canonical partition policy {plan.partition_policy!r}")
    _validate_partition_label_provider(plan.partition_policy, partition_label_provider)
    route_blocks = circuit_route_blocks_from_measurement(
        plan,
        measurement.route_blocks,
    )
    if tuple(evaluator.route_blocks) != route_blocks:
        raise ValueError("circuit evaluator route blocks differ from measurement coordinates")
    query_ids = tuple(measurement.query_ids)
    if evaluator.query_ids != query_ids or tuple(measurement_manifest.axes.get("query", ())) != query_ids:
        raise ValueError("measurement and evaluator query axes differ")
    search_algorithm = str(plan.search_policy.get("algorithm", ""))
    if search_algorithm == "reverse_hierarchical_equalized_marginal":
        from mechanistic_systems.circuits.hierarchical_search import (
            identify_reverse_hierarchical_equalized_marginal_circuits,
        )

        return identify_reverse_hierarchical_equalized_marginal_circuits(
            plan,
            measurement,
            evaluator,
            partition_label_provider=partition_label_provider,
        )
    if search_algorithm != "global_nested_rate_distortion":
        raise ValueError(f"unsupported circuit search algorithm {search_algorithm!r}")
    route_count = sum(block.route_count for block in route_blocks)
    counts = _retained_count_path(plan, route_count)
    maximum_count = max(counts)
    derived_by_layer = {
        block.layer_id: measurement.derived_block(block.layer_id)
        for block in route_blocks
    }
    candidates_by_query: dict[str, tuple[CircuitCandidate, ...]] = {}
    frontier_by_query: dict[str, tuple[CircuitCandidate, ...]] = {}
    selected_by_query: dict[str, CircuitCandidate] = {}
    heldout_by_query: dict[str, CircuitEvaluation] = {}
    selected_packed_rows: list[np.ndarray] = []
    for query_index, query_id in enumerate(query_ids):
        score_mean = np.concatenate(
            [derived_by_layer[block.layer_id]["score_mean"][query_index] for block in route_blocks]
        ).astype(np.float64, copy=False)
        score_magnitude = np.concatenate(
            [
                derived_by_layer[block.layer_id]["absolute_score_mean"][query_index]
                for block in route_blocks
            ]
        ).astype(np.float64, copy=False)
        if partition_label_provider is not None:
            group_ids = np.asarray(
                partition_label_provider.labels(
                    policy_id=plan.partition_policy,
                    query_id=query_id,
                    query_index=query_index,
                    score_mean=score_mean,
                    score_magnitude=score_magnitude,
                    route_blocks=route_blocks,
                    random_seed=plan.random_seed,
                ),
            )
        else:
            group_ids = _default_partition_labels(
                policy_id=plan.partition_policy,
                score_mean=score_mean,
                score_magnitude=score_magnitude,
                route_blocks=route_blocks,
                random_seed=plan.random_seed + query_index,
            )
        if group_ids.shape != score_mean.shape:
            raise ValueError("partition label provider returned the wrong route axis")
        order = stratified_nested_route_order(
            score_magnitude,
            group_ids,
            maximum_count=maximum_count,
        )
        query_candidates: list[CircuitCandidate] = []
        retained_by_realization: dict[str, int] = {}
        for retained_count in counts:
            mask = np.zeros(route_count, dtype=bool)
            mask[order[:retained_count]] = True
            realization_id = f"{query_id}::{plan.partition_policy}::routes={retained_count}"
            evaluation = evaluator.evaluate(
                plan,
                query_id=query_id,
                masks_by_layer=_split_mask(mask, route_blocks),
                split="optimization",
            )
            evaluation.validate()
            candidate = CircuitCandidate(
                realization_id=realization_id,
                response_distortion=evaluation.response_distortion,
                route_cost=float(retained_count),
                target_preservation=evaluation.target_preservation,
                retained_route_count=retained_count,
                metadata={"optimization_metrics": dict(evaluation.metrics)},
            )
            candidate.validate()
            query_candidates.append(candidate)
            retained_by_realization[realization_id] = retained_count
        candidates = tuple(query_candidates)
        frontier = nested_rate_distortion_frontier(candidates)
        maximum_cost, maximum_distortion, minimum_preservation = _selection_constraints(
            plan,
            maximum_count,
        )
        selected = select_circuit(
            candidates,
            maximum_cost=maximum_cost,
            maximum_distortion=maximum_distortion,
            minimum_preservation=minimum_preservation,
        )
        selected_mask = np.zeros(route_count, dtype=bool)
        selected_mask[order[: retained_by_realization[selected.realization_id]]] = True
        heldout = evaluator.evaluate(
            plan,
            query_id=query_id,
            masks_by_layer=_split_mask(selected_mask, route_blocks),
            split="evaluation",
        )
        heldout.validate()
        candidates_by_query[query_id] = candidates
        frontier_by_query[query_id] = frontier
        selected_by_query[query_id] = selected
        heldout_by_query[query_id] = heldout
        selected_packed_rows.append(np.packbits(selected_mask, bitorder="little"))
    circuits = BlockBitPackedCircuitSet(
        query_ids=query_ids,
        route_blocks=route_blocks,
        packed_masks=np.stack(selected_packed_rows, axis=0).astype(np.uint8, copy=False),
    )
    circuits.validate()
    return CircuitSearchResult(
        circuits=circuits,
        candidates_by_query=candidates_by_query,
        frontier_by_query=frontier_by_query,
        selected_by_query=selected_by_query,
        heldout_evaluation_by_query=heldout_by_query,
        partition_policy=plan.partition_policy,
        search_algorithm="global_nested_rate_distortion",
    )


def materialize_identified_circuits(
    plan: CircuitPlan,
    measurement_manifest: ArtifactManifest,
    measurement: SufficientStatisticsMeasurementReader,
    evaluator: CircuitEvaluator,
    store: ArtifactStore,
    *,
    staging_dir: str | Path,
    partition_label_provider: PartitionLabelProvider | None = None,
) -> tuple[ArtifactManifest, Path, CircuitSearchResult]:
    """Run native search and seal its selected masks plus complete frontier."""

    result = identify_nested_rate_distortion_circuits(
        plan,
        measurement_manifest,
        measurement,
        evaluator,
        partition_label_provider=partition_label_provider,
    )
    manifest, bundle = materialize_circuit_artifact(
        result.circuits,
        plan,
        store,
        staging_dir=staging_dir,
        preservation_by_query=result.preservation_by_query,
        frontier_rows=result.rows(),
    )
    return manifest, bundle, result


__all__ = [
    "CircuitEvaluation",
    "CircuitEvaluator",
    "CircuitSearchResult",
    "PartitionLabelProvider",
    "circuit_route_blocks_from_measurement",
    "identify_nested_rate_distortion_circuits",
    "materialize_identified_circuits",
    "stratified_nested_route_order",
]
