"""Exact query evaluation for target-batched functional measurements.

Queries are scientific objects.  Model adapters expose one differentiable
response matrix ``[context, response_coordinate]``; this module converts the
declared queries into scalar response values and exact cotangents without
knowing anything about model topology, route storage, or runtime policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mechanistic_systems.contracts import QuerySpec

try:  # Optional for import smoke; required by native measurement execution.
    import torch
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]


DirectionResolver = Callable[[QuerySpec, int, Any, Any], Any]


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("exact query evaluation requires PyTorch")


@dataclass(frozen=True)
class QueryCotangent:
    """One scalar query evaluated over a context batch."""

    query_id: str
    values: Any
    cotangent: Any
    status: str
    response_semantics: str


@dataclass(frozen=True)
class QueryCotangentBatch:
    """Query-major values and cotangents for one model response matrix."""

    query_ids: tuple[str, ...]
    values: Any
    cotangents: Any
    statuses: tuple[str, ...]
    response_semantics: tuple[str, ...]

    def validate(self, response_matrix: Any) -> None:
        _require_torch()
        if not torch.is_tensor(response_matrix) or response_matrix.ndim != 2:
            raise ValueError("query response matrix must be a rank-2 torch tensor")
        query_count = len(self.query_ids)
        context_count, response_width = map(int, response_matrix.shape)
        if tuple(self.values.shape) != (query_count, context_count):
            raise ValueError("query values must have shape [query,context]")
        if tuple(self.cotangents.shape) != (query_count, context_count, response_width):
            raise ValueError(
                "query cotangents must have shape [query,context,response_coordinate]"
            )
        if len(set(self.query_ids)) != query_count:
            raise ValueError("query IDs must be unique")
        if len(self.statuses) != query_count or len(self.response_semantics) != query_count:
            raise ValueError("query metadata does not match the query axis")
        if not bool(torch.isfinite(self.values).all().item()):
            raise ValueError("query values contain non-finite entries")
        if not bool(torch.isfinite(self.cotangents).all().item()):
            raise ValueError("query cotangents contain non-finite entries")


def _integer(evaluator: Mapping[str, Any], *names: str) -> int:
    for name in names:
        if name in evaluator:
            return int(evaluator[name])
    raise ValueError("query evaluator is missing " + " or ".join(names))


def _direction(
    query: QuerySpec,
    *,
    response_width: int,
    response_matrix: Any,
    labels: Any,
    resolver: DirectionResolver | None,
) -> Any:
    evaluator = query.evaluator
    raw = evaluator.get("direction")
    if raw is None:
        if resolver is None:
            raise ValueError(
                f"query {query.query_id!r} requires a declared direction resolver"
            )
        raw = resolver(query, response_width, response_matrix, labels)
    direction = torch.as_tensor(
        raw,
        device=response_matrix.device,
        dtype=response_matrix.dtype,
    )
    if direction.ndim != 1 or int(direction.numel()) != response_width:
        raise ValueError(
            f"query {query.query_id!r} direction has shape {tuple(direction.shape)}, "
            f"expected ({response_width},)"
        )
    if not bool(torch.isfinite(direction).all().item()):
        raise ValueError(f"query {query.query_id!r} direction contains non-finite values")
    norm = str(evaluator.get("direction_normalization", "none"))
    if norm == "l2":
        denominator = direction.norm().clamp_min(torch.finfo(direction.dtype).eps)
        direction = direction / denominator
    elif norm != "none":
        raise ValueError(f"unsupported direction_normalization {norm!r}")
    return direction


def evaluate_query(
    response_matrix: Any,
    query: QuerySpec,
    *,
    labels: Any = None,
    direction_resolver: DirectionResolver | None = None,
) -> QueryCotangent:
    """Evaluate one declared query and its exact response-space derivative."""

    _require_torch()
    query.validate().require_valid()
    if len(query.axis_ids) != 1:
        raise ValueError(
            "route_gate_v1 executes scalar QuerySpec objects only; vector-output "
            "metrics require an explicitly versioned estimator"
        )
    if not torch.is_tensor(response_matrix) or response_matrix.ndim != 2:
        raise ValueError("query response matrix must be [context,response_coordinate]")
    context_count, response_width = map(int, response_matrix.shape)
    evaluator = dict(query.evaluator)
    objective = str(
        evaluator.get("objective", evaluator.get("kind", query.query_kind))
    ).strip().lower()
    cotangent = torch.zeros_like(response_matrix)

    if objective in {"class_logit", "unit", "linear_component"}:
        coordinate = _integer(evaluator, "class_id", "unit", "coordinate")
        if coordinate < 0 or coordinate >= response_width:
            raise ValueError(f"query coordinate {coordinate} is outside response width {response_width}")
        values = response_matrix[:, coordinate]
        cotangent[:, coordinate] = 1.0
        status = "class_logit_exact"

    elif objective in {"margin", "logit_margin"}:
        positive = _integer(evaluator, "positive_class_id", "positive_coordinate")
        negative = _integer(evaluator, "negative_class_id", "negative_coordinate")
        if positive == negative or min(positive, negative) < 0 or max(positive, negative) >= response_width:
            raise ValueError("margin query requires two distinct in-range response coordinates")
        values = response_matrix[:, positive] - response_matrix[:, negative]
        cotangent[:, positive] = 1.0
        cotangent[:, negative] = -1.0
        status = "logit_margin_exact"

    elif objective in {"log_prob", "class_log_probability"}:
        coordinate = _integer(evaluator, "class_id", "unit", "coordinate")
        if coordinate < 0 or coordinate >= response_width:
            raise ValueError(f"query coordinate {coordinate} is outside response width {response_width}")
        probabilities = torch.softmax(response_matrix, dim=1)
        values = torch.log_softmax(response_matrix, dim=1)[:, coordinate]
        cotangent.copy_(-probabilities)
        cotangent[:, coordinate] += 1.0
        status = "log_probability_exact"

    elif objective in {
        "one_vs_rest_log_odds",
        "one_vs_rest_margin",
        "softmax_margin",
    }:
        coordinate = _integer(evaluator, "class_id", "unit", "coordinate")
        if response_width < 2:
            raise ValueError("one-vs-rest log odds requires at least two outputs")
        if coordinate < 0 or coordinate >= response_width:
            raise ValueError(
                f"query coordinate {coordinate} is outside response width {response_width}"
            )
        competitor_logits = response_matrix.clone()
        competitor_logits[:, coordinate] = -torch.inf
        competitor_logsumexp = torch.logsumexp(competitor_logits, dim=1)
        values = response_matrix[:, coordinate] - competitor_logsumexp
        competitor_probabilities = torch.softmax(competitor_logits, dim=1)
        cotangent.copy_(-competitor_probabilities)
        cotangent[:, coordinate] = 1.0
        status = "one_vs_rest_log_odds_exact"

    elif objective in {"cross_entropy", "supervised_loss"}:
        if labels is None:
            raise ValueError("cross-entropy query requires context labels")
        label_vector = torch.as_tensor(labels, device=response_matrix.device, dtype=torch.long).reshape(-1)
        if int(label_vector.numel()) != context_count:
            raise ValueError("cross-entropy labels do not match the context axis")
        if bool(torch.any((label_vector < 0) | (label_vector >= response_width)).item()):
            raise ValueError("cross-entropy labels are outside the response coordinate axis")
        probabilities = torch.softmax(response_matrix, dim=1)
        values = torch.nn.functional.cross_entropy(
            response_matrix,
            label_vector,
            reduction="none",
        )
        cotangent.copy_(probabilities)
        cotangent[torch.arange(context_count, device=response_matrix.device), label_vector] -= 1.0
        status = "cross_entropy_exact"

    elif objective in {
        "learned_head",
        "semantic_direction",
        "behavioral_readout",
        "roi_readout",
        "linear_direction",
        "prototype_direction",
        "adaptive_cluster_direction",
        "pca_direction",
        "whitened_pca_direction",
        "whitened_adaptive_cluster_direction",
        "natural_metric_category_direction",
        "linear_probe_category_direction",
        "representation_direction",
        "linear_readout_direction",
        "domain_contrast_direction",
        "brain_roi_readout",
        "brain_pc_readout",
        "behavior_or_meg_direction",
    }:
        direction = _direction(
            query,
            response_width=response_width,
            response_matrix=response_matrix,
            labels=labels,
            resolver=direction_resolver,
        )
        multiplier = float(evaluator.get("direction_multiplier", 1.0))
        values = multiplier * torch.matmul(response_matrix, direction)
        cotangent.copy_(multiplier * direction.reshape(1, -1).expand(context_count, -1))
        status = "linear_direction_exact"

    elif objective == "legacy_imagewise_activation_changes":
        coordinate = _integer(evaluator, "class_id", "unit", "coordinate")
        if coordinate < 0 or coordinate >= response_width:
            raise ValueError(f"query coordinate {coordinate} is outside response width {response_width}")
        current = response_matrix[:, coordinate]
        invert = bool(evaluator.get("invert_off_target", False))
        output_layer = bool(evaluator.get("output_layer", True))
        if output_layer and invert:
            if labels is None:
                raise ValueError("label-conditioned legacy query requires labels")
            label_vector = torch.as_tensor(labels, device=response_matrix.device).reshape(-1)
            if int(label_vector.numel()) != context_count:
                raise ValueError("legacy query labels do not match the context axis")
            multiplier = torch.where(
                label_vector == coordinate,
                torch.ones_like(current),
                -torch.ones_like(current),
            )
            values = current * multiplier
            cotangent[:, coordinate] = multiplier
        elif (not output_layer) and invert:
            values = current.abs()
            cotangent[:, coordinate] = current.sign()
        else:
            values = current
            cotangent[:, coordinate] = 1.0
        status = "legacy_imagewise_activation_changes_exact"

    else:
        raise ValueError(
            f"query {query.query_id!r} has no exact evaluator for objective {objective!r}"
        )

    packet = QueryCotangent(
        query_id=query.query_id,
        values=values,
        cotangent=cotangent,
        status=status,
        response_semantics=query.response_semantics,
    )
    if not bool(torch.isfinite(values).all().item()) or not bool(torch.isfinite(cotangent).all().item()):
        raise ValueError(f"query {query.query_id!r} produced non-finite values or cotangents")
    return packet


def evaluate_queries(
    response_matrix: Any,
    queries: Sequence[QuerySpec],
    *,
    labels: Any = None,
    direction_resolver: DirectionResolver | None = None,
) -> QueryCotangentBatch:
    """Evaluate an ordered query tile without changing its declared axis."""

    if not queries:
        raise ValueError("query evaluation requires at least one query")
    packets = tuple(
        evaluate_query(
            response_matrix,
            query,
            labels=labels,
            direction_resolver=direction_resolver,
        )
        for query in queries
    )
    batch = QueryCotangentBatch(
        query_ids=tuple(packet.query_id for packet in packets),
        values=torch.stack(tuple(packet.values for packet in packets), dim=0),
        cotangents=torch.stack(tuple(packet.cotangent for packet in packets), dim=0),
        statuses=tuple(packet.status for packet in packets),
        response_semantics=tuple(packet.response_semantics for packet in packets),
    )
    batch.validate(response_matrix)
    return batch


def legacy_analytic_target_cotangent(
    *,
    target_activations: Any,
    labels: Any,
    unit: int,
    output_layer: bool,
    invert_off_target: bool,
    target_type: str,
    prototype_direction: Any = None,
) -> QueryCotangent | None:
    """Compatibility evaluator kept for the historical scorer."""

    _require_torch()
    if not torch.is_tensor(target_activations) or target_activations.ndim != 2 or unit < 0:
        return None
    target_kind = str(target_type or "imagewise_activation_changes").strip().lower()
    if prototype_direction is not None:
        vector_target_types = {
            "prototype_direction",
            "adaptive_cluster_direction",
            "pca_direction",
            "whitened_pca_direction",
            "whitened_adaptive_cluster_direction",
            "natural_metric_category_direction",
            "linear_probe_category_direction",
            "representation_direction",
            "linear_readout_direction",
            "domain_contrast_direction",
            "brain_roi_readout",
            "brain_pc_readout",
            "behavior_or_meg_direction",
        }
        evaluator: dict[str, Any] = {
            "objective": target_kind if target_kind in vector_target_types else "linear_direction",
            "direction": prototype_direction,
            "direction_multiplier": -1.0
            if target_kind in {"prototype_direction", "adaptive_cluster_direction"}
            else 1.0,
        }
        kind = "learned_head"
    elif unit >= int(target_activations.shape[1]):
        return None
    elif target_kind == "imagewise_activation_changes":
        evaluator = {
            "objective": "legacy_imagewise_activation_changes",
            "unit": int(unit),
            "output_layer": bool(output_layer),
            "invert_off_target": bool(invert_off_target),
        }
        kind = "class_logit"
    elif target_kind == "log_prob":
        evaluator = {"objective": "log_prob", "unit": int(unit)}
        kind = "class_logit"
    elif target_kind == "unit_component_total_loss":
        evaluator = {"objective": "cross_entropy"}
        kind = "custom"
    else:
        return None
    query = QuerySpec(
        query_id=f"legacy-unit-{int(unit)}",
        query_kind=kind,
        ontology_id="legacy_target_axis",
        axis_ids=(str(int(unit)),),
        evaluator=evaluator,
        response_semantics=target_kind,
        source_hash="0" * 64,
    )
    try:
        result = evaluate_query(target_activations, query, labels=labels)
    except ValueError:
        return None
    legacy_status = {
        "legacy_imagewise_activation_changes_exact": "imagewise_activation_changes_exact",
        "log_probability_exact": "log_prob_exact",
        "cross_entropy_exact": "unit_component_total_loss_exact",
    }.get(result.status, f"{target_kind}_exact")
    return QueryCotangent(
        query_id=result.query_id,
        values=result.values,
        cotangent=result.cotangent,
        status=legacy_status,
        response_semantics=(
            target_kind
            if prototype_direction is None or target_kind in vector_target_types
            else "target_vector"
        ),
    )


__all__ = [
    "DirectionResolver",
    "QueryCotangent",
    "QueryCotangentBatch",
    "evaluate_queries",
    "evaluate_query",
    "legacy_analytic_target_cotangent",
]
