"""Historical reverse-hierarchical equalized-marginal circuit search."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import numpy as np

from mechanistic_systems.circuits.marginal_frontier import (
    geometric_retention_schedule,
    trace_single_marginal_path,
    trace_symmetric_marginal_path,
)
from mechanistic_systems.circuits.rate_distortion import CircuitCandidate
from mechanistic_systems.circuits.search import (
    CircuitEvaluation,
    CircuitSearchResult,
    PartitionLabelProvider,
    circuit_route_blocks_from_measurement,
)
from mechanistic_systems.core.storage import BlockBitPackedCircuitSet
from mechanistic_systems.contracts import CircuitPlan


def _ranked_partition_mask(
    priority: np.ndarray,
    partition_labels: np.ndarray,
    *,
    positive_density: float,
    negative_density: float,
    unpartitioned: bool,
) -> np.ndarray:
    values = np.asarray(priority, dtype=np.float64).reshape(-1)
    labels = np.asarray(partition_labels, dtype=np.int8).reshape(-1)
    if values.shape != labels.shape or not np.all(np.isfinite(values)):
        raise ValueError("hierarchical priority and partition axes differ")
    mask = np.zeros(values.size, dtype=bool)
    branches = ((0, positive_density),) if unpartitioned else (
        (0, positive_density),
        (1, negative_density),
    )
    for branch_id, density in branches:
        indices = np.flatnonzero(labels == branch_id)
        if unpartitioned:
            indices = np.arange(values.size, dtype=np.int64)
        if not 0.0 <= float(density) <= 1.0:
            raise ValueError("hierarchical retained density must lie in [0,1]")
        retained = min(len(indices), max(0, int(np.ceil(float(density) * len(indices)))))
        if retained:
            order = np.lexsort((indices, -values[indices]))
            mask[indices[order[:retained]]] = True
    return mask


def _layer_schedule(plan: CircuitPlan, route_kind: str, route_count: int) -> tuple[float, ...]:
    schedules = dict(plan.search_policy.get("density_schedules", {}) or {})
    configuration = dict(schedules.get(route_kind, {}) or {})
    if not configuration:
        raise ValueError(f"historical search lacks a density schedule for {route_kind!r}")
    if configuration.get("values") is not None:
        values = tuple(float(value) for value in configuration["values"])
        if not values or values[0] != 1.0 or values[-1] != 0.0:
            raise ValueError("explicit historical density schedules require endpoints one and zero")
        return values
    return geometric_retention_schedule(
        decay=float(configuration["decay"]),
        final_density=float(configuration["final_density"]),
        group_size=int(route_count),
    )


def _split_masks(
    masks: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=bool).copy() for key, value in masks.items()}


def _build_pair_loss_evaluator(
    *,
    evaluator: Any,
    optimization_view: Any,
    query_id: str,
    committed_masks: Mapping[str, np.ndarray],
    layer_id: str,
    layer_shape: tuple[int, ...],
    priority: np.ndarray,
    partition_labels: np.ndarray,
    unpartitioned: bool,
) -> Callable[[float, float], float]:
    """Bind one layer stage so later loop iterations cannot change its closure."""

    loss_cache: dict[tuple[float, float], float] = {}

    def evaluate_pair(positive: float, negative: float) -> float:
        key = (float(positive), float(negative))
        if key not in loss_cache:
            candidate_masks = _split_masks(committed_masks)
            candidate_masks[layer_id] = _ranked_partition_mask(
                priority,
                partition_labels,
                positive_density=positive,
                negative_density=negative,
                unpartitioned=unpartitioned,
            ).reshape(layer_shape)
            _response, _fit, candidate_loss = evaluator.calibrated_bce_loss(
                optimization_view,
                query_id=query_id,
                masks_by_layer=candidate_masks,
            )
            loss_cache[key] = float(candidate_loss)
        return loss_cache[key]

    return evaluate_pair


def _normalized_damage(loss: float, dense_loss: float, null_loss: float) -> tuple[float, float]:
    span = abs(float(null_loss) - float(dense_loss))
    if not np.isfinite(span) or span <= 1e-12:
        raise ValueError("global calibrated BCE anchors do not define a finite damage span")
    raw = float((float(loss) - float(dense_loss)) / span)
    return max(0.0, raw), 1.0 - raw


def _evaluate_equal_density(
    evaluate_pair: Callable[[float, float], float],
    density: float,
) -> float:
    return evaluate_pair(density, density)


def identify_reverse_hierarchical_equalized_marginal_circuits(
    plan: CircuitPlan,
    measurement: Any,
    evaluator: Any,
    *,
    partition_label_provider: PartitionLabelProvider | None,
) -> CircuitSearchResult:
    """Run the historical controller with cumulative deep-to-shallow masks."""

    if str(plan.search_policy.get("algorithm")) != "reverse_hierarchical_equalized_marginal":
        raise ValueError("hierarchical search received another search policy")
    if not hasattr(evaluator, "calibrated_bce_loss"):
        raise ValueError("historical marginal search requires calibrated target-logit responses")
    route_blocks = circuit_route_blocks_from_measurement(
        plan,
        measurement.route_blocks,
    )
    query_ids = tuple(measurement.query_ids)
    derived_by_layer = {
        block.layer_id: measurement.derived_block(block.layer_id)
        for block in route_blocks
    }
    threshold = float(plan.search_policy["sensitivity_threshold"])
    unpartitioned = plan.partition_policy == "unpartitioned"
    if not unpartitioned and partition_label_provider is None:
        raise ValueError("partitioned historical search requires an explicit label provider")

    candidates_by_query: dict[str, tuple[CircuitCandidate, ...]] = {}
    selected_by_query: dict[str, CircuitCandidate] = {}
    heldout_by_query: dict[str, CircuitEvaluation] = {}
    telemetry_by_query: dict[str, Mapping[str, Any]] = {}
    packed_rows: list[np.ndarray] = []
    optimization_view = plan.optimization_view
    evaluation_view = plan.evaluation_view

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
        if unpartitioned:
            global_labels = np.zeros(score_magnitude.size, dtype=np.int8)
        else:
            assert partition_label_provider is not None
            global_labels = np.asarray(
                partition_label_provider.labels(
                    policy_id=plan.partition_policy,
                    query_id=query_id,
                    query_index=query_index,
                    score_mean=score_mean,
                    score_magnitude=score_magnitude,
                    route_blocks=route_blocks,
                    random_seed=plan.random_seed,
                ),
                dtype=np.int8,
            ).reshape(-1)
        if global_labels.shape != score_magnitude.shape:
            raise ValueError("historical partition labels differ from the route axis")
        if not unpartitioned and not set(np.unique(global_labels)).issubset({0, 1}):
            raise ValueError("historical sign partition must contain only positive/negative labels")

        committed = {
            block.layer_id: np.ones(block.shape, dtype=bool)
            for block in route_blocks
        }
        all_pruned = {
            block.layer_id: np.zeros(block.shape, dtype=bool)
            for block in route_blocks
        }
        _dense_response, dense_fit, dense_loss = evaluator.calibrated_bce_loss(
            optimization_view,
            query_id=query_id,
            masks_by_layer=committed,
        )
        _null_response, null_fit, null_loss = evaluator.calibrated_bce_loss(
            optimization_view,
            query_id=query_id,
            masks_by_layer=all_pruned,
        )
        initial = CircuitCandidate(
            realization_id=f"{query_id}::historical::dense",
            response_distortion=0.0,
            route_cost=float(score_magnitude.size),
            target_preservation=1.0,
            retained_route_count=int(score_magnitude.size),
            metadata={"calibrated_balanced_bce": dense_loss, "stage": "dense"},
        )
        candidates: list[CircuitCandidate] = [initial]
        stage_rows: list[dict[str, Any]] = []
        for reverse_index, block in enumerate(reversed(route_blocks)):
            block_slice = slice(block.offset, block.stop)
            block_priority = score_magnitude[block_slice]
            block_labels = global_labels[block_slice]
            schedule = _layer_schedule(plan, block.route_kind, block.route_count)
            evaluate_pair = _build_pair_loss_evaluator(
                evaluator=evaluator,
                optimization_view=optimization_view,
                query_id=query_id,
                committed_masks=committed,
                layer_id=block.layer_id,
                layer_shape=block.shape,
                priority=block_priority,
                partition_labels=block_labels,
                unpartitioned=unpartitioned,
            )

            stage_dense_loss = evaluate_pair(1.0, 1.0)
            stage_null_loss = evaluate_pair(0.0, 0.0)
            if abs(stage_null_loss - stage_dense_loss) <= 1e-12:
                stage_rows.append(
                    {
                        "layer_id": block.layer_id,
                        "reverse_index": reverse_index,
                        "status": "uninformative_stage_anchor",
                        "dense_loss": stage_dense_loss,
                        "fully_pruned_loss": stage_null_loss,
                    }
                )
                continue
            if unpartitioned:
                path = trace_single_marginal_path(
                    partial(_evaluate_equal_density, evaluate_pair),
                    schedule=schedule,
                    total_layer_route_count=block.route_count,
                    dense_anchor_loss=stage_dense_loss,
                    fully_pruned_anchor_loss=stage_null_loss,
                    sensitivity_threshold=threshold,
                )
            else:
                path = trace_symmetric_marginal_path(
                    evaluate_pair,
                    positive_schedule=schedule,
                    negative_schedule=schedule,
                    total_layer_route_count=block.route_count,
                    dense_anchor_loss=stage_dense_loss,
                    fully_pruned_anchor_loss=stage_null_loss,
                    sensitivity_threshold=threshold,
                    initial_turn=str(plan.search_policy["initial_partition"]),
                )
            committed[block.layer_id] = _ranked_partition_mask(
                block_priority,
                block_labels,
                positive_density=path.positive_density,
                negative_density=path.negative_density,
                unpartitioned=unpartitioned,
            ).reshape(block.shape)
            stage_loss = evaluate_pair(path.positive_density, path.negative_density)
            distortion, preservation = _normalized_damage(stage_loss, dense_loss, null_loss)
            retained_count = int(sum(np.count_nonzero(mask) for mask in committed.values()))
            candidate = CircuitCandidate(
                realization_id=f"{query_id}::historical::stage={reverse_index:04d}",
                response_distortion=distortion,
                route_cost=float(retained_count),
                target_preservation=preservation,
                retained_route_count=retained_count,
                metadata={
                    "calibrated_balanced_bce": stage_loss,
                    "layer_id": block.layer_id,
                    "positive_density": path.positive_density,
                    "negative_density": path.negative_density,
                    "stage_dense_loss": stage_dense_loss,
                    "stage_fully_pruned_loss": stage_null_loss,
                    "accepted_step_count": sum(step.accepted for step in path.steps),
                    "proposal_count": len(path.steps),
                    "stopping_reason": path.stopping_reason,
                },
            )
            candidate.validate()
            candidates.append(candidate)
            stage_rows.append(dict(candidate.metadata))

        _final_response, final_fit, final_loss = evaluator.calibrated_bce_loss(
            optimization_view,
            query_id=query_id,
            masks_by_layer=committed,
        )
        final_distortion, final_preservation = _normalized_damage(final_loss, dense_loss, null_loss)
        retained_count = int(sum(np.count_nonzero(mask) for mask in committed.values()))
        selected = CircuitCandidate(
            realization_id=f"{query_id}::historical::selected",
            response_distortion=final_distortion,
            route_cost=float(retained_count),
            target_preservation=final_preservation,
            retained_route_count=retained_count,
            metadata={
                "calibrated_balanced_bce": final_loss,
                "affine_scale": final_fit.scale,
                "affine_offset": final_fit.offset,
                "stage_count": len(route_blocks),
            },
        )
        selected.validate()
        candidates.append(selected)

        _eval_dense, _dense_eval_fit, eval_dense_loss = evaluator.calibrated_bce_loss(
            evaluation_view,
            query_id=query_id,
            masks_by_layer={block.layer_id: np.ones(block.shape, dtype=bool) for block in route_blocks},
            fit=dense_fit,
        )
        _eval_null, _null_eval_fit, eval_null_loss = evaluator.calibrated_bce_loss(
            evaluation_view,
            query_id=query_id,
            masks_by_layer=all_pruned,
            fit=null_fit,
        )
        eval_response, _eval_fit, eval_final_loss = evaluator.calibrated_bce_loss(
            evaluation_view,
            query_id=query_id,
            masks_by_layer=committed,
            fit=final_fit,
        )
        eval_distortion, eval_preservation = _normalized_damage(
            eval_final_loss,
            eval_dense_loss,
            eval_null_loss,
        )
        heldout = CircuitEvaluation(
            response_distortion=eval_distortion,
            target_preservation=eval_preservation,
            context_count=len(eval_response.context_ids),
            metrics={
                "calibrated_balanced_bce": eval_final_loss,
                "calibrated_balanced_bce_dense": eval_dense_loss,
                "calibrated_balanced_bce_fully_pruned": eval_null_loss,
                "affine_scale": final_fit.scale,
                "affine_offset": final_fit.offset,
            },
        )
        heldout.validate()
        flat_mask = np.concatenate(
            [committed[block.layer_id].reshape(-1) for block in route_blocks]
        )
        packed_rows.append(np.packbits(flat_mask, bitorder="little"))
        candidates_by_query[query_id] = tuple(candidates)
        selected_by_query[query_id] = selected
        heldout_by_query[query_id] = heldout
        telemetry_by_query[query_id] = {
            "algorithm": "reverse_hierarchical_equalized_marginal",
            "layer_order": "deep_to_shallow",
            "sensitivity_threshold": threshold,
            "denominator_policy": plan.search_policy["denominator_policy"],
            "stages": stage_rows,
        }

    circuits = BlockBitPackedCircuitSet(
        query_ids=query_ids,
        route_blocks=route_blocks,
        packed_masks=np.stack(packed_rows, axis=0).astype(np.uint8, copy=False),
    )
    circuits.validate()
    return CircuitSearchResult(
        circuits=circuits,
        candidates_by_query=candidates_by_query,
        frontier_by_query={query_id: (selected_by_query[query_id],) for query_id in query_ids},
        selected_by_query=selected_by_query,
        heldout_evaluation_by_query=heldout_by_query,
        partition_policy=plan.partition_policy,
        search_algorithm="reverse_hierarchical_equalized_marginal",
        search_telemetry_by_query=telemetry_by_query,
    )


__all__ = ["identify_reverse_hierarchical_equalized_marginal_circuits"]
