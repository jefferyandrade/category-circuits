"""Cost-aware sampled panels and paired-risk evidence.

The canonical v0-v2 design uses target-positive versus pooled target-negative
strata. The historical negative-category planner remains in this module as an
explicit comparator. Both are independent of model execution: a proposal
declares one immutable panel, then parent and candidate masks are evaluated on
the same contexts.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mechanistic_systems.measurement.contexts import ContextAxis
from mechanistic_systems.measurement.neyman_allocation import (
    BINARY_NEYMAN_POLICY_ID,
    BINARY_NEYMAN_STRATA,
    NeymanAllocationRequest,
    NeymanStratumSpec,
    UnifiedNeymanAllocator,
    binary_neyman_request,
)

from .glocal_sampled_path import (
    request_cumulative_target_images,
    request_per_stratum_target_images,
)
from .sampled_path import (
    CrossfitPanel,
    PairedCandidateEvidence,
    SampledComparisonRequest,
)

try:  # Torch's CPU Generator is part of the sealed v1 sampling identity.
    import torch
except Exception:  # pragma: no cover - import smoke without Torch.
    torch = None  # type: ignore[assignment]


NEYMAN_PANEL_SCHEMA = "mechanistic_cost_optimal_two_stage_neyman_panel_v1"
HIERARCHICAL_RISK_SCHEMA = "mechanistic_hierarchical_paired_risk_v1"


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("sampled_path_frontier_v1 panel allocation requires PyTorch")


def _seed(seed: int, query_id: str, stratum: str) -> int:
    payload = f"{int(seed)}:{query_id}:{stratum}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**63 - 1)


def _ordered_positions(
    positions: np.ndarray,
    *,
    seed: int,
    query_id: str,
    stratum: str,
) -> np.ndarray:
    _require_torch()
    values = np.asarray(positions, dtype=np.int64).reshape(-1)
    if values.size <= 1:
        return values.copy()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_seed(seed, query_id, stratum))
    order = torch.randperm(int(values.size), generator=generator).numpy()
    return values[order]


@dataclass(frozen=True)
class NeymanEvidenceState:
    """Allocation-relevant posterior state from the preceding evidence stage."""

    pooled_sd: float = 1.0
    positive_sample_count: int = 0
    negative_category_count_sampled: int = 0
    negative_category_between_variance: float = 0.0
    stratum_sds: Mapping[str, float] = field(default_factory=dict)
    class_counts: Mapping[int, int] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any] | None
    ) -> "NeymanEvidenceState | None":
        if not payload:
            return None
        return cls(
            pooled_sd=max(1e-6, float(payload.get("allocation_pooled_sd", 1.0) or 1.0)),
            positive_sample_count=max(
                0, int(payload.get("positive_sample_count", 0) or 0)
            ),
            negative_category_count_sampled=max(
                0, int(payload.get("negative_category_count_sampled", 0) or 0)
            ),
            negative_category_between_variance=max(
                0.0,
                float(payload.get("negative_category_between_variance", 0.0) or 0.0),
            ),
            stratum_sds={
                str(key): max(1e-6, float(value))
                for key, value in dict(
                    payload.get("allocation_stratum_sds", {}) or {}
                ).items()
            },
            class_counts={
                int(key): max(0, int(value))
                for key, value in dict(payload.get("class_counts", {}) or {}).items()
            },
        )


@dataclass(frozen=True)
class CostOptimalNeymanPolicy:
    """Frozen production subset of the historical panel allocator."""

    sampling_seed: int = 0
    fold_count: int = 2
    panel_image_budget_sequence: tuple[int, ...] = (2048,)
    min_positive_images: int = 64
    min_negative_categories: int = 256
    min_neg_images_per_category: int = 1
    min_neg_images_per_new_category: int = 2
    max_negative_categories: int = 999
    panel_batch_multiple: int = 256
    finite_population_correction: bool = True
    variance_prior_strength: float = 4.0
    storage_sort: bool = True
    on_target_context_cost: float = 1.0
    off_target_context_cost: float = 1.0

    def validate(self) -> None:
        if self.fold_count < 2:
            raise ValueError("sampled-path cross-fitting requires at least two folds")
        if not self.panel_image_budget_sequence or any(
            int(value) <= 0 for value in self.panel_image_budget_sequence
        ):
            raise ValueError("panel budget sequence must contain positive values")
        integer_fields = (
            self.min_positive_images,
            self.min_negative_categories,
            self.min_neg_images_per_category,
            self.min_neg_images_per_new_category,
            self.max_negative_categories,
            self.panel_batch_multiple,
        )
        if any(int(value) <= 0 for value in integer_fields):
            raise ValueError("Neyman panel counts and batch multiple must be positive")
        if self.max_negative_categories < self.min_negative_categories:
            raise ValueError("maximum negative categories cannot be below the minimum")
        if (
            not math.isfinite(self.variance_prior_strength)
            or self.variance_prior_strength < 0
        ):
            raise ValueError("variance prior strength must be finite and nonnegative")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in (
                self.on_target_context_cost,
                self.off_target_context_cost,
            )
        ):
            raise ValueError("binary Neyman context costs must be positive")


@dataclass(frozen=True)
class NeymanPanelAllocation:
    panel: CrossfitPanel
    target_class_id: int
    positive_count: int
    selected_negative_categories: tuple[int, ...]
    negative_counts: Mapping[int, int]
    expected_variance: float
    expected_regret_reduction: float
    budget: int
    allocation_trace: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class HierarchicalPairedRisk:
    posterior_mean_delta: float
    posterior_sd_delta: float
    posterior_interval_95: tuple[float, float]
    class_means: Mapping[int, float]
    class_counts: Mapping[int, int]
    class_variances: Mapping[int, float]
    metadata: Mapping[str, Any]
    schema_name: str = HIERARCHICAL_RISK_SCHEMA

    def to_allocation_state(self) -> Mapping[str, Any]:
        return {
            **dict(self.metadata),
            "class_counts": {
                str(key): int(value) for key, value in self.class_counts.items()
            },
        }


def integer_neyman_waterfill(
    *,
    weight_by_key: Mapping[Any, float],
    available_by_key: Mapping[Any, int],
    base_counts: Mapping[Any, int],
    remaining: int,
) -> dict[Any, int]:
    """Exact integer Neyman water-fill over pre-seeded, capped strata.

    Distributes ``remaining`` units *on top of* the occupancy ``base_counts``
    across the keys of ``weight_by_key`` (iterated in INSERTION ORDER),
    proportional to the Neyman weights, never exceeding ``available_by_key``
    (finite-population caps), with a ``weight / sqrt(n + 1)`` marginal-variance-
    reduction tie-break used only when the proportional-floor step stalls.
    Returns per-key ADDITIONS (deltas, not totals).

    This is the single source of truth for the engine's integer allocation:
    both the pruning planner (``CostOptimalNeymanPanelPlanner._allocate_remaining``,
    positive/negative strata) and the symmetric-category attribution consumer
    (``measurement.attribution_sampler``) call it. Two load-bearing invariants:
    (i) ``weight_by_key`` is iterated in insertion order, so the allocation is
    order-sensitive across equal-weight survivors of a binding cap (the caller
    fixes a canonical stratum order); (ii) the return value is deltas the caller
    adds to its own counts, never totals.
    """

    additions: dict[Any, int] = {key: 0 for key in weight_by_key}
    remaining_to_allocate = int(remaining)
    while remaining_to_allocate > 0:
        active = [
            key
            for key in additions
            if int(base_counts.get(key, 0)) + additions[key] < available_by_key[key]
        ]
        if not active:
            break
        current_total = sum(
            int(base_counts.get(key, 0)) + additions[key] for key in active
        )
        target_total = current_total + remaining_to_allocate
        weight_sum = max(1e-12, sum(weight_by_key[key] for key in active))
        proposed: dict[Any, int] = {}
        spent = 0
        for key in active:
            current = int(base_counts.get(key, 0)) + additions[key]
            desired = min(
                available_by_key[key],
                max(
                    current,
                    int(math.floor(target_total * weight_by_key[key] / weight_sum)),
                ),
            )
            add = max(0, min(remaining_to_allocate - spent, desired - current))
            proposed[key] = add
            spent += add
        if spent <= 0:
            best = max(
                active,
                key=lambda key: weight_by_key[key]
                / math.sqrt(int(base_counts.get(key, 0)) + additions[key] + 1.0),
            )
            proposed[best] = proposed.get(best, 0) + 1
        applied = 0
        for key, add in proposed.items():
            add = min(
                int(add),
                available_by_key[key] - int(base_counts.get(key, 0)) - additions[key],
                remaining_to_allocate,
            )
            if add <= 0:
                continue
            additions[key] += add
            remaining_to_allocate -= add
            applied += add
        if applied <= 0:
            break
    return additions


def binary_cost_aware_neyman_counts(
    *,
    total_count: int,
    available_by_stratum: Mapping[str, int],
    minimum_by_stratum: Mapping[str, int],
    existing_by_stratum: Mapping[str, int] | None = None,
    sd_by_stratum: Mapping[str, float] | None = None,
    weight_by_stratum: Mapping[str, float] | None = None,
    cost_by_stratum: Mapping[str, float] | None = None,
) -> tuple[dict[str, int], tuple[Mapping[str, Any], ...]]:
    """Compatibility wrapper over the unified measurement allocator.

    The two strata are deliberately coarse: target-positive and pooled
    target-negative.  Minimum support is filled first, then every additional
    observation is assigned to the stratum with largest

    ``w_h**2 * sigma_h**2 / (c_h * n_h * (n_h + 1))``.

    Equalizing this discrete priority has continuous solution
    ``n_h ∝ w_h sigma_h / sqrt(c_h)``.  Variance estimates affect allocation
    only; they do not alter or narrow the downstream claim interval.
    """

    result = UnifiedNeymanAllocator().allocate(
        binary_neyman_request(
            total_count=total_count,
            available_by_stratum=available_by_stratum,
            minimum_by_stratum=minimum_by_stratum,
            existing_by_stratum=existing_by_stratum,
            sd_by_stratum=sd_by_stratum,
            weight_by_stratum=weight_by_stratum,
            cost_by_stratum=cost_by_stratum,
            role="binary_compatibility_adapter",
        )
    )
    return dict(result.target_counts), tuple(result.trace)


class CostOptimalNeymanPanelPlanner:
    """Exact integer Neyman allocator over an immutable ContextAxis."""

    def __init__(
        self,
        axis: ContextAxis,
        *,
        query_id: str,
        target_class_id: int,
        policy: CostOptimalNeymanPolicy,
    ) -> None:
        axis.validate()
        policy.validate()
        self.axis = axis
        self.query_id = str(query_id)
        self.target_class_id = int(target_class_id)
        self.policy = policy
        records = tuple(sorted(axis.records, key=lambda item: int(item.ordinal)))
        if tuple(int(record.ordinal) for record in records) != tuple(
            range(len(records))
        ):
            raise ValueError(
                "Neyman allocation requires a contiguous ordinal context axis"
            )
        self._records = records
        self._labels = np.asarray(
            [int(record.label) for record in records], dtype=np.int64
        )
        self._positions_by_class = {
            int(label): np.flatnonzero(self._labels == int(label)).astype(np.int64)
            for label in np.unique(self._labels)
        }
        self._positive_positions = np.flatnonzero(
            self._labels == self.target_class_id
        ).astype(np.int64)
        self._negative_positions_by_class = {
            int(label): positions
            for label, positions in self._positions_by_class.items()
            if int(label) != self.target_class_id and positions.size
        }
        if not self._positive_positions.size or not self._negative_positions_by_class:
            raise ValueError(
                "Neyman allocation requires target-positive and target-negative contexts"
            )

    def _allocation_sd(self, key: str, state: NeymanEvidenceState | None) -> float:
        if state is None:
            return 1.0
        pooled = max(1e-6, float(state.pooled_sd))
        if key not in state.stratum_sds:
            return pooled
        raw = max(1e-6, float(state.stratum_sds[key]))
        prior = float(self.policy.variance_prior_strength)
        if prior <= 0.0:
            return raw
        if key == "positive":
            count = int(state.positive_sample_count)
        elif key.startswith("neg:"):
            count = int(state.class_counts.get(int(key.split(":", 1)[1]), 0))
        else:
            count = 0
        variance = (count * raw * raw + prior * pooled * pooled) / max(
            1e-12, count + prior
        )
        return max(1e-6, math.sqrt(max(0.0, variance)))

    def _negative_class_order(self, state: NeymanEvidenceState | None) -> list[int]:
        _require_torch()
        classes = list(self._negative_positions_by_class)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _seed(self.policy.sampling_seed, self.query_id, "negative_class_order")
        )
        shuffled = [
            classes[int(index)]
            for index in torch.randperm(len(classes), generator=generator)
        ]
        if state is None:
            return shuffled
        return sorted(
            shuffled,
            key=lambda class_id: float(
                state.stratum_sds.get(f"neg:{class_id}", state.pooled_sd)
            ),
            reverse=True,
        )

    def _variance_components(
        self,
        counts: Mapping[str, int],
        selected_negative_categories: Sequence[int],
        state: NeymanEvidenceState | None,
    ) -> dict[str, float]:
        positive_population = max(1, int(self._positive_positions.size))
        total_negative_classes = max(1, len(self._negative_positions_by_class))
        scale = float(self._labels.size - self._positive_positions.size) / float(
            max(1, self._labels.size)
        )
        pooled_sd = 1.0 if state is None else float(state.pooled_sd)
        pooled_variance = max(1e-12, pooled_sd * pooled_sd)
        n_pos = max(1, int(counts.get("positive", 0)))
        # The historical controller shrinks SDs when choosing a Neyman
        # allocation, but evaluates the resulting design with the raw posterior
        # stratum SDs. These are deliberately different statistical roles.
        stratum_sds = {} if state is None else state.stratum_sds
        positive_sd = float(stratum_sds.get("positive", pooled_sd))
        positive_fpc = (
            max(0.0, 1.0 - float(n_pos) / float(positive_population))
            if self.policy.finite_population_correction
            else 1.0
        )
        positive_variance = positive_fpc * positive_sd * positive_sd / float(n_pos)
        selected = [
            int(class_id)
            for class_id in selected_negative_categories
            if f"neg:{int(class_id)}" in counts
        ]
        category_count = max(1, len(selected))
        negative_within = 0.0
        for class_id in selected:
            key = f"neg:{class_id}"
            n_class = max(1, int(counts.get(key, 0)))
            class_sd = float(stratum_sds.get(key, pooled_sd))
            negative_within += class_sd * class_sd / float(n_class)
        negative_within /= float(category_count * category_count)
        sampled_fraction = min(
            1.0, float(category_count) / float(total_negative_classes)
        )
        between_variance = (
            pooled_variance
            if state is None
            else float(state.negative_category_between_variance)
        )
        if state is not None and self.policy.variance_prior_strength > 0.0:
            observed = float(state.negative_category_count_sampled or len(selected))
            prior = float(self.policy.variance_prior_strength)
            between_variance = (
                observed * between_variance + prior * pooled_variance
            ) / max(1e-12, observed + prior)
        if not self.policy.finite_population_correction:
            sampled_fraction = 0.0
        negative_between = (
            max(0.0, 1.0 - sampled_fraction) * between_variance / float(category_count)
        )
        return {
            "positive": float(scale * scale * positive_variance),
            "negative_within": float(scale * scale * negative_within),
            "negative_between": float(scale * scale * negative_between),
            "total": float(
                scale * scale * (positive_variance + negative_within + negative_between)
            ),
        }

    def _expected_variance(
        self,
        counts: Mapping[str, int],
        selected: Sequence[int],
        state: NeymanEvidenceState | None,
    ) -> float:
        return float(self._variance_components(counts, selected, state)["total"])

    def _allocate_remaining(
        self,
        *,
        counts: dict[str, int],
        selected: list[int],
        class_order: Sequence[int],
        positive_available: int,
        remaining: int,
        state: NeymanEvidenceState | None,
    ) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        if remaining <= 0:
            return trace
        selected_set = set(selected)
        pooled_sd = 1.0 if state is None else max(1e-12, state.pooled_sd)
        between_variance = (
            0.0 if state is None else state.negative_category_between_variance
        )
        between_dominates = bool(between_variance > pooled_sd * pooled_sd)
        open_target = int(self.policy.min_negative_categories)
        if between_dominates:
            open_target = min(
                int(self.policy.max_negative_categories), len(class_order)
            )
        while (
            remaining > 0
            and len(selected) < open_target
            and len(selected)
            < min(int(self.policy.max_negative_categories), len(class_order))
        ):
            next_class = next(
                (
                    int(class_id)
                    for class_id in class_order
                    if int(class_id) not in selected_set
                ),
                None,
            )
            if next_class is None:
                break
            seed_count = min(
                int(self.policy.min_neg_images_per_new_category),
                int(self._negative_positions_by_class[next_class].size),
                int(remaining),
            )
            if seed_count <= 0:
                break
            selected.append(next_class)
            selected_set.add(next_class)
            counts[f"neg:{next_class}"] = seed_count
            remaining -= seed_count
            trace.append(
                {
                    "action": "open_negative_category",
                    "class": next_class,
                    "add_count": seed_count,
                    "runtime_cost": seed_count,
                    "remaining_budget": remaining,
                    "allocation_rule": (
                        "between_category_variance_dominant_coverage"
                        if between_dominates
                        else "minimum_negative_category_coverage"
                    ),
                }
            )
        if remaining <= 0:
            return trace
        category_count = max(1, len(selected))
        strata: list[tuple[str, int | None, float, int]] = []
        if int(counts.get("positive", 0)) < positive_available:
            strata.append(
                (
                    "positive",
                    None,
                    self._allocation_sd("positive", state),
                    positive_available,
                )
            )
        for class_id in selected:
            key = f"neg:{class_id}"
            available = int(self._negative_positions_by_class[class_id].size)
            if int(counts.get(key, 0)) < available:
                strata.append(
                    (
                        key,
                        class_id,
                        self._allocation_sd(key, state) / float(category_count),
                        available,
                    )
                )
        if not strata:
            return trace
        available_by_key = {
            key: available for key, _class_id, _weight, available in strata
        }
        weight_by_key = {key: weight for key, _class_id, weight, _available in strata}
        additions = integer_neyman_waterfill(
            weight_by_key=weight_by_key,
            available_by_key=available_by_key,
            base_counts=counts,
            remaining=int(remaining),
        )
        class_by_key = {key: class_id for key, class_id, _weight, _available in strata}
        added_total = 0
        for key, add in additions.items():
            if add <= 0:
                continue
            counts[key] = int(counts.get(key, 0)) + int(add)
            added_total += int(add)
            trace.append(
                {
                    "action": "closed_form_neyman_add",
                    "class": class_by_key[key],
                    "stratum": key,
                    "add_count": int(add),
                    "neyman_weight": float(weight_by_key[key]),
                    "runtime_cost": int(add),
                    "remaining_budget": max(0, int(remaining) - added_total),
                    "allocation_rule": "closed_form_equal_cost_neyman",
                }
            )
        return trace

    def _round_to_batch_multiple(self, indices: np.ndarray) -> np.ndarray:
        values = np.unique(np.asarray(indices, dtype=np.int64))
        multiple = int(self.policy.panel_batch_multiple)
        if multiple <= 1 or values.size == 0 or values.size < multiple:
            return values
        remainder = int(values.size % multiple)
        if remainder == 0:
            return values
        need = multiple - remainder
        selected = np.zeros(self._labels.size, dtype=bool)
        selected[values] = True
        remaining = np.flatnonzero(~selected).astype(np.int64)
        positive = remaining[self._labels[remaining] == self.target_class_id]
        negative = remaining[self._labels[remaining] != self.target_class_id]
        chosen: list[np.ndarray] = []
        for name, pool in (
            ("positive_rounding", positive),
            ("negative_rounding", negative),
        ):
            if need <= 0 or not pool.size:
                continue
            take = min(need, int(pool.size))
            chosen.append(
                _ordered_positions(
                    pool,
                    seed=self.policy.sampling_seed,
                    query_id=self.query_id,
                    stratum=name,
                )[:take]
            )
            need -= take
        if need > 0:
            chosen.append(
                _ordered_positions(
                    remaining,
                    seed=self.policy.sampling_seed,
                    query_id=self.query_id,
                    stratum="fallback_rounding",
                )[:need]
            )
        return np.concatenate([values, *chosen]) if chosen else values

    def _design_weights(self, indices: np.ndarray) -> np.ndarray:
        labels = self._labels[indices]
        positive = labels == self.target_class_id
        positive_population = max(1, int(self._positive_positions.size))
        negative_population = max(1, int(self._labels.size - positive_population))
        scale = float(negative_population) / float(
            positive_population + negative_population
        )
        weights = np.zeros(indices.size, dtype=np.float64)
        positive_count = int(positive.sum())
        if positive_count:
            weights[positive] = scale / float(positive_count)
        negative_classes = np.unique(labels[~positive])
        class_count = max(1, int(negative_classes.size))
        for class_id in negative_classes:
            mask = (~positive) & (labels == int(class_id))
            count = int(mask.sum())
            if count:
                weights[mask] = scale / float(class_count * count)
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("Neyman design produced invalid sampling weights")
        return weights

    def _fold_ids(self, indices: np.ndarray) -> np.ndarray:
        labels = self._labels[indices]
        folds = np.empty(indices.size, dtype=np.int64)
        for class_id in np.unique(labels):
            locations = np.flatnonzero(labels == int(class_id))
            folds[locations] = np.arange(locations.size, dtype=np.int64) % int(
                self.policy.fold_count
            )
        return folds

    def _stable_prefix_fold_ids(self, indices: np.ndarray) -> np.ndarray:
        """Assign folds from sealed within-stratum ranks, not storage order.

        Storage sorting can insert newly revealed contexts before an earlier
        prefix and thereby change the historical location-based fold.  Exact
        cumulative requests need stronger nesting: once a context is exposed,
        its fold is immutable at every later look.
        """

        values = np.asarray(indices, dtype=np.int64).reshape(-1)
        folds = np.empty(values.size, dtype=np.int64)
        location_by_position = {
            int(position): location for location, position in enumerate(values)
        }
        for class_id in np.unique(self._labels[values]):
            positions = self._positions_by_class[int(class_id)]
            stratum = (
                "positive"
                if int(class_id) == self.target_class_id
                else f"neg:{int(class_id)}"
            )
            ordered = _ordered_positions(
                positions,
                seed=self.policy.sampling_seed,
                query_id=self.query_id,
                stratum=stratum,
            )
            for rank, position in enumerate(ordered):
                location = location_by_position.get(int(position))
                if location is not None:
                    folds[location] = int(rank % self.policy.fold_count)
        return folds

    def _exact_stratum_targets(
        self,
        request: SampledComparisonRequest,
        *,
        budget: int,
    ) -> dict[str, int] | None:
        cumulative_target = request_cumulative_target_images(request)
        requested_targets = request_per_stratum_target_images(request)
        if not requested_targets:
            return None
        if cumulative_target is None:
            raise ValueError("per-stratum targets require a cumulative panel target")
        targets = {str(key): int(value) for key, value in requested_targets.items()}
        if any(value < 0 for value in targets.values()):
            raise ValueError("per-stratum panel targets must be nonnegative")
        allowed = {
            "positive",
            *(f"neg:{class_id}" for class_id in self._negative_positions_by_class),
        }
        unknown = sorted(set(targets) - allowed)
        if unknown:
            raise ValueError(f"unknown Neyman strata in exact target: {unknown}")
        if sum(targets.values()) != int(budget):
            raise ValueError(
                "per-stratum panel targets must sum to cumulative target images"
            )
        capacities = {"positive": int(self._positive_positions.size)}
        capacities.update(
            {
                f"neg:{class_id}": int(positions.size)
                for class_id, positions in self._negative_positions_by_class.items()
            }
        )
        for key, count in targets.items():
            if count > capacities[key]:
                raise ValueError(f"Neyman target for {key} exceeds its population")
            if 0 < count < int(self.policy.fold_count):
                raise ValueError(
                    f"Neyman target for {key} must cover every cross-fit fold"
                )
        return targets

    def draw(
        self,
        request: SampledComparisonRequest,
        *,
        previous_state: NeymanEvidenceState | None = None,
    ) -> NeymanPanelAllocation:
        if (
            request.query_id != self.query_id
            or int(request.fold_count) != self.policy.fold_count
        ):
            raise ValueError("sample request differs from its Neyman planner")
        cumulative_target = request_cumulative_target_images(request)
        requested_stratum_targets = request_per_stratum_target_images(request)
        exact_request = cumulative_target is not None
        if exact_request:
            budget = int(cumulative_target)
            if budget <= 0:
                raise ValueError("cumulative panel target must be positive")
        else:
            budget = int(
                self.policy.panel_image_budget_sequence[
                    min(
                        int(request.stage_index),
                        len(self.policy.panel_image_budget_sequence) - 1,
                    )
                ]
            )
        budget = min(int(self._labels.size), budget)
        if exact_request and budget < 2 * int(self.policy.fold_count):
            raise ValueError(
                "exact cumulative target must cover every fold on both sides"
            )
        class_order = self._negative_class_order(previous_state)
        exact_targets = self._exact_stratum_targets(request, budget=budget)
        trace: list[dict[str, Any]] = []
        if exact_targets is not None:
            counts = {key: count for key, count in exact_targets.items() if count}
            selected = sorted(
                int(key.split(":", 1)[1]) for key in counts if key.startswith("neg:")
            )
            if counts.get("positive", 0) < int(self.policy.fold_count) or sum(
                count for key, count in counts.items() if key.startswith("neg:")
            ) < int(self.policy.fold_count):
                raise ValueError(
                    "exact Neyman targets must cover positive and negative folds"
                )
        elif exact_request and request.prior_panel is not None:
            prior = request.prior_panel
            if int(request.stage_index) <= int(prior.stage_index):
                raise ValueError("exact cumulative panel stages must advance")
            if budget < len(prior.context_ids):
                raise ValueError(
                    "cumulative panel target cannot shrink its prior prefix"
                )
            if not prior.labels:
                raise ValueError("nested Neyman prefixes require prior stratum labels")
            prior_labels = np.asarray(prior.labels, dtype=np.int64)
            counts = {"positive": int((prior_labels == self.target_class_id).sum())}
            selected = sorted(
                int(value)
                for value in np.unique(prior_labels)
                if int(value) != self.target_class_id
            )
            counts.update(
                {
                    f"neg:{class_id}": int((prior_labels == class_id).sum())
                    for class_id in selected
                }
            )
        else:
            # Exact first looks reserve at least one complete fold cycle for
            # the negative side. Legacy stage-indexed requests retain their
            # historical initialization byte-for-byte.
            reserve_negative = int(self.policy.fold_count) if exact_request else 0
            positive_count = min(
                int(self._positive_positions.size),
                max(
                    int(self.policy.fold_count) if exact_request else 1,
                    min(
                        int(self.policy.min_positive_images),
                        max(1, budget - reserve_negative),
                    ),
                ),
            )
            remaining_after_positive = max(0, budget - positive_count)
            min_new = max(1, int(self.policy.min_neg_images_per_new_category))
            max_classes_by_budget = (
                max(1, remaining_after_positive // min_new)
                if remaining_after_positive > 0
                else 1
            )
            desired_classes = min(
                len(class_order),
                int(self.policy.max_negative_categories),
                max_classes_by_budget,
                max(1, int(self.policy.min_negative_categories)),
            )
            selected = list(class_order[:desired_classes])
            counts = {"positive": positive_count}
            for class_id in selected:
                counts[f"neg:{class_id}"] = min(
                    int(self._negative_positions_by_class[class_id].size),
                    max(int(self.policy.min_neg_images_per_category), min_new),
                )
            while sum(counts.values()) > budget and selected:
                counts.pop(f"neg:{selected.pop()}", None)
            if not selected and remaining_after_positive > 0:
                class_id = class_order[0]
                selected = [class_id]
                counts[f"neg:{class_id}"] = min(
                    int(self._negative_positions_by_class[class_id].size),
                    remaining_after_positive,
                )
        initial_variance = self._expected_variance(counts, selected, previous_state)
        if exact_targets is None:
            trace = self._allocate_remaining(
                counts=counts,
                selected=selected,
                class_order=class_order,
                positive_available=int(self._positive_positions.size),
                remaining=max(0, budget - sum(counts.values())),
                state=previous_state,
            )
        selected_indices: list[np.ndarray] = []
        pos_count = min(
            int(counts.get("positive", 0)), int(self._positive_positions.size)
        )
        if pos_count:
            selected_indices.append(
                _ordered_positions(
                    self._positive_positions,
                    seed=self.policy.sampling_seed,
                    query_id=self.query_id,
                    stratum="positive",
                )[:pos_count]
            )
        negative_counts: dict[int, int] = {}
        for class_id in selected:
            key = f"neg:{class_id}"
            count = min(
                int(counts.get(key, 0)),
                int(self._negative_positions_by_class[class_id].size),
            )
            if count <= 0:
                continue
            negative_counts[class_id] = count
            selected_indices.append(
                _ordered_positions(
                    self._negative_positions_by_class[class_id],
                    seed=self.policy.sampling_seed,
                    query_id=self.query_id,
                    stratum=key,
                )[:count]
            )
        if not selected_indices:
            raise ValueError("cost-optimal Neyman allocation selected no contexts")
        indices = np.concatenate(selected_indices)
        if not exact_request:
            indices = self._round_to_batch_multiple(indices)
        indices = np.unique(indices)
        if self.policy.storage_sort:
            indices = np.sort(indices)
        if exact_request and int(indices.size) != int(budget):
            raise ValueError(
                "exact cumulative Neyman target could not be satisfied by its strata"
            )
        weights = self._design_weights(indices)
        folds = (
            self._stable_prefix_fold_ids(indices)
            if exact_request
            else self._fold_ids(indices)
        )
        labels = self._labels[indices]
        sampled_negative = tuple(
            int(value) for value in np.unique(labels[labels != self.target_class_id])
        )
        sampled_negative_counts = {
            class_id: int((labels == class_id).sum()) for class_id in sampled_negative
        }
        final_counts = {"positive": int((labels == self.target_class_id).sum())}
        final_counts.update(
            {
                f"neg:{class_id}": count
                for class_id, count in sampled_negative_counts.items()
            }
        )
        variance_components = self._variance_components(
            final_counts, sampled_negative, previous_state
        )
        panel = CrossfitPanel(
            context_ids=tuple(
                self._records[int(index)].context_id for index in indices
            ),
            fold_ids=tuple(int(value) for value in folds),
            sampling_weights=tuple(float(value) for value in weights),
            labels=tuple(int(value) for value in labels),
            draw_id=hashlib.sha256(
                (
                    f"{self.policy.sampling_seed}:{self.query_id}:{request.layer_id}:"
                    f"{request.partition_id}:{request.proposal_index}:{request.stage_index}"
                ).encode("utf-8")
            ).hexdigest(),
            stage_index=int(request.stage_index),
            allocation_metadata={
                "schema_name": NEYMAN_PANEL_SCHEMA,
                "allocation_rule": "cost_optimal_two_stage_neyman",
                "target_class_id": self.target_class_id,
                "panel_image_budget": budget,
                "positive_population": int(self._positive_positions.size),
                "negative_population": int(
                    self._labels.size - self._positive_positions.size
                ),
                "negative_category_count_total": len(self._negative_positions_by_class),
                "variance_components": variance_components,
                "allocation_trace": trace[-128:],
                **(
                    {
                        "requested_cumulative_target_images": int(cumulative_target),
                        "requested_per_stratum_target_images": dict(
                            requested_stratum_targets
                        ),
                        "delivered_stratum_counts": dict(final_counts),
                    }
                    if exact_request
                    else {}
                ),
            },
        )
        panel.validate(fold_count=self.policy.fold_count)
        if exact_request and request.prior_panel is not None:
            panel.assert_nested_extension(request.prior_panel)
        return NeymanPanelAllocation(
            panel=panel,
            target_class_id=self.target_class_id,
            positive_count=final_counts["positive"],
            selected_negative_categories=sampled_negative,
            negative_counts=sampled_negative_counts,
            expected_variance=float(variance_components["total"]),
            expected_regret_reduction=float(
                max(0.0, initial_variance - variance_components["total"])
            ),
            budget=budget,
            allocation_trace=tuple(trace),
        )


class BinaryCostOptimalNeymanPanelPlanner(CostOptimalNeymanPanelPlanner):
    """Canonical two-stratum planner for v0-v2.

    The parent planner is retained as the historical negative-category
    comparator.  Production dispatch uses this subclass so category labels
    cannot become extra pruning degrees of freedom.
    """

    def _binary_sd(
        self, key: str, state: NeymanEvidenceState | None
    ) -> float:
        if state is None:
            return 1.0
        aliases = (
            ("on_target", "positive")
            if key == "on_target"
            else ("off_target", "negative")
        )
        raw = next(
            (
                float(state.stratum_sds[name])
                for name in aliases
                if name in state.stratum_sds
            ),
            float(state.pooled_sd),
        )
        return max(1e-6, raw)

    def draw(
        self,
        request: SampledComparisonRequest,
        *,
        previous_state: NeymanEvidenceState | None = None,
    ) -> NeymanPanelAllocation:
        if (
            request.query_id != self.query_id
            or int(request.fold_count) != self.policy.fold_count
        ):
            raise ValueError("sample request differs from its binary Neyman planner")
        positive = self._positive_positions
        negative = np.flatnonzero(
            self._labels != self.target_class_id
        ).astype(np.int64)
        capacities = {
            "on_target": int(positive.size),
            "off_target": int(negative.size),
        }
        cumulative = request_cumulative_target_images(request)
        exact = cumulative is not None
        budget = (
            int(cumulative)
            if exact
            else int(
                self.policy.panel_image_budget_sequence[
                    min(
                        int(request.stage_index),
                        len(self.policy.panel_image_budget_sequence) - 1,
                    )
                ]
            )
        )
        budget = min(sum(capacities.values()), budget)
        fold_count = int(self.policy.fold_count)
        if any(value < fold_count for value in capacities.values()):
            raise ValueError(
                "binary Neyman populations cannot cover every cross-fit fold"
            )
        if budget < 2 * fold_count:
            raise ValueError(
                "binary Neyman target must cover every cross-fit fold on both sides"
            )
        prior_counts = {"on_target": 0, "off_target": 0}
        if request.prior_panel is not None:
            prior_labels = np.asarray(request.prior_panel.labels, dtype=np.int64)
            prior_counts = {
                "on_target": int((prior_labels == self.target_class_id).sum()),
                "off_target": int((prior_labels != self.target_class_id).sum()),
            }
            if budget < sum(prior_counts.values()):
                raise ValueError("binary Neyman cumulative target cannot shrink")
        requested_targets = request_per_stratum_target_images(request)
        trace: tuple[Mapping[str, Any], ...]
        if requested_targets:
            normalized_targets = {
                ("on_target" if key == "positive" else key): int(value)
                for key, value in requested_targets.items()
            }
            if set(normalized_targets) != set(BINARY_NEYMAN_STRATA):
                raise ValueError(
                    "binary exact targets require on_target and off_target"
                )
            if sum(normalized_targets.values()) != budget:
                raise ValueError("binary exact targets must sum to panel target")
            if any(
                normalized_targets[key] < max(fold_count, prior_counts[key])
                or normalized_targets[key] > capacities[key]
                for key in BINARY_NEYMAN_STRATA
            ):
                raise ValueError("binary exact target violates support or nesting")
            allocation_result = UnifiedNeymanAllocator().allocate(
                NeymanAllocationRequest(
                    strata=tuple(
                        NeymanStratumSpec(
                            stratum_id=key,
                            capacity=capacities[key],
                            minimum_count=fold_count,
                            estimand_weight=0.5,
                            allocation_standard_deviation=self._binary_sd(
                                key, previous_state
                            ),
                            unit_cost=(
                                self.policy.on_target_context_cost
                                if key == "on_target"
                                else self.policy.off_target_context_cost
                            ),
                        )
                        for key in BINARY_NEYMAN_STRATA
                    ),
                    existing_counts=prior_counts,
                    exact_target_counts=normalized_targets,
                    require_normalized_estimand_weights=True,
                    role="sampled_path_panel",
                    policy_id=BINARY_NEYMAN_POLICY_ID,
                )
            )
            counts = dict(allocation_result.target_counts)
            trace = tuple(allocation_result.trace)
        else:
            natural_core = min(
                int(self.policy.min_positive_images),
                max(fold_count, budget // 4),
            )
            minima = {
                key: max(
                    prior_counts[key],
                    min(capacities[key], max(fold_count, natural_core)),
                )
                for key in BINARY_NEYMAN_STRATA
            }
            if sum(minima.values()) > budget:
                minima = {
                    key: max(prior_counts[key], fold_count)
                    for key in BINARY_NEYMAN_STRATA
                }
            allocation_result = UnifiedNeymanAllocator().allocate(
                binary_neyman_request(
                    total_count=budget,
                    available_by_stratum=capacities,
                    minimum_by_stratum=minima,
                    existing_by_stratum=prior_counts,
                    sd_by_stratum={
                        key: self._binary_sd(key, previous_state)
                        for key in BINARY_NEYMAN_STRATA
                    },
                    weight_by_stratum={"on_target": 0.5, "off_target": 0.5},
                    cost_by_stratum={
                        "on_target": self.policy.on_target_context_cost,
                        "off_target": self.policy.off_target_context_cost,
                    },
                    role="sampled_path_panel",
                )
            )
            counts = dict(allocation_result.target_counts)
            trace = tuple(allocation_result.trace)
        ordered = {
            "on_target": _ordered_positions(
                positive,
                seed=self.policy.sampling_seed,
                query_id=self.query_id,
                stratum="on_target",
            ),
            "off_target": _ordered_positions(
                negative,
                seed=self.policy.sampling_seed,
                query_id=self.query_id,
                stratum="off_target",
            ),
        }
        indices = np.concatenate(
            [
                ordered[key][: counts[key]]
                for key in BINARY_NEYMAN_STRATA
            ]
        )
        if self.policy.storage_sort:
            indices = np.sort(indices)
        labels = self._labels[indices]
        binary = labels == self.target_class_id
        delivered = {
            "on_target": int(binary.sum()),
            "off_target": int((~binary).sum()),
        }
        weights = np.where(
            binary,
            0.5 / float(delivered["on_target"]),
            0.5 / float(delivered["off_target"]),
        )
        location = {
            int(position): index for index, position in enumerate(indices)
        }
        folds = np.empty(indices.size, dtype=np.int64)
        for key in BINARY_NEYMAN_STRATA:
            for rank, position in enumerate(ordered[key]):
                index = location.get(int(position))
                if index is not None:
                    folds[index] = int(rank % fold_count)
        sds = {
            key: self._binary_sd(key, previous_state)
            for key in BINARY_NEYMAN_STRATA
        }
        variance_components: dict[str, float] = {}
        for key in BINARY_NEYMAN_STRATA:
            fpc = (
                max(
                    0.0,
                    1.0
                    - float(delivered[key]) / float(capacities[key]),
                )
                if self.policy.finite_population_correction
                else 1.0
            )
            variance_components[key] = (
                0.25
                * fpc
                * sds[key] ** 2
                / float(delivered[key])
            )
        variance_components["total"] = sum(variance_components.values())
        sampled_negative = tuple(
            int(value) for value in np.unique(labels[~binary])
        )
        sampled_negative_counts = {
            class_id: int((labels == class_id).sum())
            for class_id in sampled_negative
        }
        panel = CrossfitPanel(
            context_ids=tuple(
                self._records[int(index)].context_id for index in indices
            ),
            fold_ids=tuple(int(value) for value in folds),
            sampling_weights=tuple(float(value) for value in weights),
            labels=tuple(int(value) for value in labels),
            draw_id=hashlib.sha256(
                (
                    f"{self.policy.sampling_seed}:{self.query_id}:binary:"
                    f"{request.layer_id}:{request.partition_id}:"
                    f"{request.proposal_index}:{request.stage_index}"
                ).encode("utf-8")
            ).hexdigest(),
            stage_index=int(request.stage_index),
            allocation_metadata={
                "schema_name": (
                    "mechanistic_binary_cost_aware_neyman_panel_v1"
                ),
                "allocation_rule": (
                    "binary_pooled_off_target_cost_aware_neyman"
                ),
                "allocator_policy_id": BINARY_NEYMAN_POLICY_ID,
                "allocator_receipt": allocation_result.to_receipt(),
                "target_class_id": self.target_class_id,
                "panel_image_budget": budget,
                "positive_population": capacities["on_target"],
                "negative_population": capacities["off_target"],
                "finite_population_correction": bool(
                    self.policy.finite_population_correction
                ),
                "stratum_weights": {
                    "on_target": 0.5,
                    "off_target": 0.5,
                },
                "stratum_costs": {
                    "on_target": self.policy.on_target_context_cost,
                    "off_target": self.policy.off_target_context_cost,
                },
                "delivered_stratum_counts": delivered,
                "variance_components": variance_components,
                "allocation_trace": trace[-128:],
                "continuous_optimum": (
                    "n_h proportional to w_h*sigma_h/sqrt(c_h)"
                ),
                "off_target_categories_are_descriptive_only": True,
                **(
                    {
                        "requested_cumulative_target_images": int(cumulative),
                        "requested_per_stratum_target_images": dict(
                            requested_targets
                        ),
                    }
                    if exact
                    else {}
                ),
            },
        )
        panel.validate(fold_count=fold_count)
        if request.prior_panel is not None:
            panel.assert_nested_extension(request.prior_panel)
        return NeymanPanelAllocation(
            panel=panel,
            target_class_id=self.target_class_id,
            positive_count=delivered["on_target"],
            selected_negative_categories=sampled_negative,
            negative_counts=sampled_negative_counts,
            expected_variance=float(variance_components["total"]),
            expected_regret_reduction=0.0,
            budget=budget,
            allocation_trace=tuple(trace),
        )


def summarize_hierarchical_paired_risk(
    evidence: PairedCandidateEvidence,
) -> HierarchicalPairedRisk:
    """Summarize paired cross-fit deltas with the sealed stratified estimator."""

    panel = evidence.panel
    labels = np.asarray(panel.labels, dtype=np.int64)
    if labels.size != len(panel.context_ids):
        raise ValueError("hierarchical risk requires labels on the sampled panel")
    deltas = np.asarray(evidence.candidate_loss, dtype=np.float64) - np.asarray(
        evidence.committed_loss, dtype=np.float64
    )
    target_class_id = int(panel.allocation_metadata.get("target_class_id", -1))
    if target_class_id < 0:
        raise ValueError("Neyman panel is missing target-class identity")
    positive = labels == target_class_id
    positive_values = deltas[positive]
    negative_labels = labels[~positive]
    class_means: dict[int, float] = {}
    class_counts: dict[int, int] = {}
    class_variances: dict[int, float] = {}
    for class_id in np.unique(labels):
        values = deltas[labels == int(class_id)]
        class_means[int(class_id)] = float(values.mean())
        class_counts[int(class_id)] = int(values.size)
        class_variances[int(class_id)] = (
            float(values.var(ddof=1)) if values.size > 1 else 0.0
        )
    if (
        panel.allocation_metadata.get("allocation_rule")
        == "binary_pooled_off_target_cost_aware_neyman"
    ):
        off_values = deltas[~positive]
        if positive_values.size < 1 or off_values.size < 1:
            raise ValueError(
                "binary Neyman evidence requires both target sides"
            )
        means = {
            "on_target": float(positive_values.mean()),
            "off_target": float(off_values.mean()),
        }
        variances = {
            "on_target": (
                float(positive_values.var(ddof=1))
                if positive_values.size > 1
                else 0.0
            ),
            "off_target": (
                float(off_values.var(ddof=1))
                if off_values.size > 1
                else 0.0
            ),
        }
        populations = {
            "on_target": max(
                1,
                int(
                    panel.allocation_metadata.get(
                        "positive_population", positive_values.size
                    )
                ),
            ),
            "off_target": max(
                1,
                int(
                    panel.allocation_metadata.get(
                        "negative_population", off_values.size
                    )
                ),
            ),
        }
        counts = {
            "on_target": int(positive_values.size),
            "off_target": int(off_values.size),
        }
        risk_variance = 0.0
        for key in BINARY_NEYMAN_STRATA:
            fpc = (
                max(
                    0.0,
                    1.0 - float(counts[key]) / float(populations[key]),
                )
                if bool(
                    panel.allocation_metadata.get(
                        "finite_population_correction", True
                    )
                )
                else 1.0
            )
            risk_variance += (
                0.25 * fpc * variances[key] / float(counts[key])
            )
        risk_mean = 0.5 * (means["on_target"] + means["off_target"])
        risk_sd = math.sqrt(max(risk_variance, 1e-18))
        pooled_sd = (
            float(deltas.std(ddof=1)) if deltas.size > 1 else 1.0
        )
        fold_ids = np.asarray(panel.fold_ids, dtype=np.int64)
        fold_support = [
            {
                "fold": int(fold),
                "positive": int(positive[fold_ids == fold].sum()),
                "negative": int((~positive[fold_ids == fold]).sum()),
                "sample_count": int((fold_ids == fold).sum()),
            }
            for fold in sorted(set(fold_ids.tolist()))
        ]
        metadata = {
            "schema_name": HIERARCHICAL_RISK_SCHEMA,
            "hierarchical_model": (
                "two_stratum_class_balanced_bce_with_neyman_allocation"
            ),
            "sample_design": (
                "binary_pooled_off_target_cost_aware_neyman_common_random_numbers"
            ),
            "balanced_bce_estimator": (
                "equal_weight_on_target_pooled_off_target_delta"
            ),
            "binary_stratum_weights": {
                "on_target": 0.5,
                "off_target": 0.5,
            },
            "positive_population": populations["on_target"],
            "negative_population": populations["off_target"],
            "positive_sample_count": counts["on_target"],
            "negative_sample_count": counts["off_target"],
            "on_target_delta_mean": means["on_target"],
            "off_target_delta_mean": means["off_target"],
            "on_target_delta_variance": variances["on_target"],
            "off_target_delta_variance": variances["off_target"],
            "allocation_pooled_sd": max(1e-6, pooled_sd),
            "allocation_stratum_sds": {
                "on_target": max(
                    1e-6, math.sqrt(max(0.0, variances["on_target"]))
                ),
                "off_target": max(
                    1e-6, math.sqrt(max(0.0, variances["off_target"]))
                ),
            },
            "allocation_rule": (
                "binary_pooled_off_target_cost_aware_neyman"
            ),
            "off_target_categories_are_descriptive_only": True,
            "row_loss_weighting_for_estimator": "unweighted",
            "fold_target_support": fold_support,
            "fold_min_positive_count": min(
                (item["positive"] for item in fold_support), default=0
            ),
            "fold_min_negative_count": min(
                (item["negative"] for item in fold_support), default=0
            ),
        }
        return HierarchicalPairedRisk(
            posterior_mean_delta=float(risk_mean),
            posterior_sd_delta=float(risk_sd),
            posterior_interval_95=(
                float(risk_mean - 1.96 * risk_sd),
                float(risk_mean + 1.96 * risk_sd),
            ),
            class_means=class_means,
            class_counts=class_counts,
            class_variances=class_variances,
            metadata=metadata,
        )
    pooled_sd = float(deltas.std(ddof=1)) if deltas.size > 1 else 1.0
    if not math.isfinite(pooled_sd):
        pooled_sd = 1.0
    if positive_values.size:
        positive_mean = float(positive_values.mean())
        positive_variance = (
            float(positive_values.var(ddof=1)) if positive_values.size > 1 else 0.0
        )
    else:
        positive_mean = 0.0
        positive_variance = 0.0
    allocation_sds: dict[str, float] = {
        "positive": (
            math.sqrt(max(0.0, positive_variance))
            if positive_values.size > 1
            else pooled_sd
        )
    }
    negative_means: list[float] = []
    negative_within_terms: list[float] = []
    for class_id in np.unique(negative_labels):
        values = deltas[(labels == int(class_id)) & (~positive)]
        if not values.size:
            continue
        mean = float(values.mean())
        variance = (
            float(values.var(ddof=1)) if values.size > 1 else pooled_sd * pooled_sd
        )
        negative_means.append(mean)
        negative_within_terms.append(variance / float(max(1, values.size)))
        allocation_sds[f"neg:{int(class_id)}"] = math.sqrt(max(0.0, variance))
    positive_population = max(
        1,
        int(panel.allocation_metadata.get("positive_population", positive_values.size)),
    )
    negative_population = max(
        1, int(panel.allocation_metadata.get("negative_population", (~positive).sum()))
    )
    total_negative_classes = max(
        1,
        int(
            panel.allocation_metadata.get(
                "negative_category_count_total", len(np.unique(negative_labels))
            )
        ),
    )
    scale = float(negative_population) / float(
        positive_population + negative_population
    )
    if negative_means:
        sampled_negative_count = len(negative_means)
        negative_mean = float(np.mean(negative_means))
        negative_within = float(sum(negative_within_terms)) / float(
            sampled_negative_count * sampled_negative_count
        )
        unsampled_fraction = max(
            0.0, 1.0 - float(sampled_negative_count) / float(total_negative_classes)
        )
        between = (
            float(np.var(negative_means, ddof=1)) if sampled_negative_count > 1 else 0.0
        )
        negative_variance = negative_within + (
            unsampled_fraction * between / float(sampled_negative_count)
        )
    else:
        sampled_negative_count = 0
        negative_mean = 0.0
        negative_variance = 0.0
        unsampled_fraction = 1.0
        between = 0.0
    positive_mean_variance = positive_variance / float(max(1, positive_values.size))
    risk_mean = scale * (positive_mean + negative_mean)
    risk_variance = scale * scale * (positive_mean_variance + negative_variance)
    risk_sd = math.sqrt(max(risk_variance, 1e-18))
    fold_ids = np.asarray(panel.fold_ids, dtype=np.int64)
    fold_support = []
    for fold in sorted(set(fold_ids.tolist())):
        mask = fold_ids == fold
        fold_support.append(
            {
                "fold": int(fold),
                "positive": int(positive[mask].sum()),
                "negative": int((~positive[mask]).sum()),
                "sample_count": int(mask.sum()),
            }
        )
    metadata = {
        "schema_name": HIERARCHICAL_RISK_SCHEMA,
        "hierarchical_model": "bayesian_stratified_balanced_bce_with_neyman_allocation",
        "sample_design": "cost_optimal_two_stage_neyman_common_random_numbers",
        "balanced_bce_estimator": "target_preservation_pos_neg_stratified_delta",
        "balanced_bce_scale": scale,
        "positive_population": positive_population,
        "negative_population": negative_population,
        "positive_sample_count": int(positive_values.size),
        "negative_sample_count": int((~positive).sum()),
        "negative_category_count_sampled": sampled_negative_count,
        "negative_category_count_total": total_negative_classes,
        "negative_category_unsampled_fraction": unsampled_fraction,
        "positive_delta_mean": positive_mean,
        "negative_category_delta_mean": negative_mean,
        "positive_delta_variance": positive_variance,
        "negative_mean_variance": negative_variance,
        "negative_category_between_variance": between,
        "allocation_pooled_sd": max(1e-6, pooled_sd),
        "allocation_stratum_sds": allocation_sds,
        "allocation_rule": "cost_optimal_two_stage_neyman",
        "row_loss_weighting_for_estimator": "unweighted",
        "fold_target_support": fold_support,
        "fold_min_positive_count": min(
            (item["positive"] for item in fold_support), default=0
        ),
        "fold_min_negative_count": min(
            (item["negative"] for item in fold_support), default=0
        ),
    }
    return HierarchicalPairedRisk(
        posterior_mean_delta=float(risk_mean),
        posterior_sd_delta=float(risk_sd),
        posterior_interval_95=(
            float(risk_mean - 1.96 * risk_sd),
            float(risk_mean + 1.96 * risk_sd),
        ),
        class_means=class_means,
        class_counts=class_counts,
        class_variances=class_variances,
        metadata=metadata,
    )


__all__ = [
    "BINARY_NEYMAN_STRATA",
    "BinaryCostOptimalNeymanPanelPlanner",
    "CostOptimalNeymanPanelPlanner",
    "CostOptimalNeymanPolicy",
    "HIERARCHICAL_RISK_SCHEMA",
    "HierarchicalPairedRisk",
    "NEYMAN_PANEL_SCHEMA",
    "NeymanEvidenceState",
    "NeymanPanelAllocation",
    "binary_cost_aware_neyman_counts",
    "summarize_hierarchical_paired_risk",
]
