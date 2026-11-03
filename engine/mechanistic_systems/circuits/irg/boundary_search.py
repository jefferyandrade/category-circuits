"""Gridless complete-mask boundary experimentation.

The search operates on a finite, version-induced family of complete hard masks.
It adaptively chooses which masks deserve exact oracle evaluation near the
quality boundary; it never invents a density grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from mechanistic_systems.inference.sequential_bce import (
    RunningMoment,
    stratified_student_t_summary_from_moments,
)
from mechanistic_systems.evidence import staged_alpha, student_t_interval

from .candidates import ActionCandidate
from .contracts import content_hash


@dataclass(frozen=True)
class BoundaryPoint:
    candidate: ActionCandidate
    released_cost: float
    predicted_quality_loss: float
    predicted_standard_error: float
    expected_oracle_cost: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "released_cost",
            "predicted_quality_loss",
            "predicted_standard_error",
            "expected_oracle_cost",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (
                name
                in {"released_cost", "predicted_standard_error", "expected_oracle_cost"}
                and value < 0.0
            ):
                raise ValueError(f"{name} has an invalid value")
        if self.expected_oracle_cost <= 0.0:
            raise ValueError("expected_oracle_cost must be positive")


@dataclass
class BoundaryObservation:
    candidate_id: str
    values: list[float] = field(default_factory=list)
    values_by_stratum: dict[str, list[float]] = field(default_factory=dict)
    moments_by_stratum: dict[str, RunningMoment] = field(default_factory=dict)
    stratum_weights: dict[str, float] = field(default_factory=dict)
    look_count: int = 0

    def append(
        self,
        values: Sequence[float] | np.ndarray,
        *,
        strata: Sequence[str] | None = None,
        stratum_weights: dict[str, float] | None = None,
    ) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError("boundary observation requires finite nonempty values")
        self.values.extend(float(value) for value in array)
        self.look_count += 1
        if strata is None:
            if self.values_by_stratum:
                raise ValueError(
                    "cannot append unstratified rows after stratified evidence"
                )
            return
        labels = tuple(str(value) for value in strata)
        if len(labels) != int(array.size):
            raise ValueError("boundary stratum labels must align with values")
        declared = {
            str(key): float(value)
            for key, value in dict(stratum_weights or {}).items()
        }
        if not set(labels).issubset(declared):
            raise ValueError(
                "boundary stratum weights must cover every observed stratum"
            )
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in declared.values())
            or not math.isclose(sum(declared.values()), 1.0, abs_tol=1e-12)
        ):
            raise ValueError(
                "boundary stratum weights must be positive and sum to one"
            )
        if self.stratum_weights and declared != self.stratum_weights:
            raise ValueError("boundary stratum weights changed between looks")
        self.stratum_weights = declared
        for label, value in zip(labels, array, strict=True):
            self.values_by_stratum.setdefault(label, []).append(float(value))
        for label in sorted(set(labels)):
            packet = RunningMoment.from_values(
                array[
                    np.fromiter(
                        (item == label for item in labels),
                        dtype=bool,
                    )
                ]
            )
            self.moments_by_stratum[label] = self.moments_by_stratum.get(
                label, RunningMoment.empty()
            ).merge(packet)

    def append_moments(
        self,
        moments: dict[str, RunningMoment],
        *,
        stratum_weights: dict[str, float],
    ) -> None:
        declared = {
            str(key): float(value)
            for key, value in dict(stratum_weights).items()
        }
        if set(moments) != set(declared):
            raise ValueError(
                "boundary moment strata must equal the declared weight strata"
            )
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in declared.values())
            or not math.isclose(sum(declared.values()), 1.0, abs_tol=1e-12)
        ):
            raise ValueError(
                "boundary stratum weights must be positive and sum to one"
            )
        if self.values and not self.moments_by_stratum:
            raise ValueError(
                "cannot append stratified moments after unstratified evidence"
            )
        if self.stratum_weights and declared != self.stratum_weights:
            raise ValueError("boundary stratum weights changed between looks")
        self.stratum_weights = declared
        self.look_count += 1
        for label, packet in moments.items():
            if packet.count <= 0:
                raise ValueError("boundary moment packets must be nonempty")
            self.moments_by_stratum[label] = self.moments_by_stratum.get(
                label, RunningMoment.empty()
            ).merge(packet)

    @property
    def count(self) -> int:
        if self.moments_by_stratum:
            return int(sum(value.count for value in self.moments_by_stratum.values()))
        return len(self.values)

    @property
    def mean(self) -> float:
        if self.moments_by_stratum:
            return float(
                sum(
                    self.stratum_weights[key] * moment.mean
                    for key, moment in self.moments_by_stratum.items()
                )
            )
        return float(np.mean(self.values))

    @property
    def standard_error(self) -> float:
        if self.moments_by_stratum:
            if any(moment.count < 2 for moment in self.moments_by_stratum.values()):
                return math.inf
            variance = sum(
                self.stratum_weights[key] ** 2
                * moment.sample_variance
                / float(moment.count)
                for key, moment in self.moments_by_stratum.items()
            )
            return float(math.sqrt(max(0.0, variance)))
        if len(self.values) < 2:
            return 0.0
        return float(np.std(self.values, ddof=1) / math.sqrt(len(self.values)))


@dataclass(frozen=True)
class BoundaryProposal:
    candidate_id: str | None
    action: str
    reason: str
    resample: bool = False


def weighted_isotonic_regression(
    values: Sequence[float], weights: Sequence[float] | None = None
) -> np.ndarray:
    """Nondecreasing weighted PAVA."""

    y = np.asarray(values, dtype=np.float64).reshape(-1)
    if y.size == 0 or not np.isfinite(y).all():
        raise ValueError("isotonic regression requires finite nonempty values")
    w = (
        np.ones_like(y)
        if weights is None
        else np.asarray(weights, dtype=np.float64).reshape(-1)
    )
    if w.shape != y.shape or not np.isfinite(w).all() or np.any(w <= 0.0):
        raise ValueError("isotonic weights must be finite, positive, and aligned")
    block_values: list[float] = []
    block_weights: list[float] = []
    block_starts: list[int] = []
    block_ends: list[int] = []
    for index, (value, weight) in enumerate(zip(y, w, strict=True)):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_starts.append(index)
        block_ends.append(index + 1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            total_weight = block_weights[-2] + block_weights[-1]
            pooled = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / total_weight
            block_values[-2:] = [pooled]
            block_weights[-2:] = [total_weight]
            block_ends[-2:] = [block_ends[-1]]
            block_starts.pop()
    fitted = np.empty_like(y)
    for value, start, end in zip(block_values, block_starts, block_ends, strict=True):
        fitted[start:end] = value
    return fitted


class AdaptiveBoundarySearch:
    """Adaptive oracle selection on a version-induced complete-mask path."""

    def __init__(
        self,
        points: Sequence[BoundaryPoint],
        *,
        maximum_quality_loss: float,
        practical_indifference: float = 0.0,
        cost_tolerance: float = 0.0,
        monotonicity_tolerance: float = 0.0,
        z_value: float = 2.0,
        resample_standard_error: float = 0.0,
        confidence_level: float | None = None,
        maximum_looks_per_candidate: int | None = 32,
        precision_only_iid: bool = False,
        resample_until_resolved: bool = False,
        accept_quality_loss: float | None = None,
        reject_quality_loss: float | None = None,
    ) -> None:
        ordered = tuple(
            sorted(
                points,
                key=lambda point: (
                    point.released_cost,
                    point.predicted_quality_loss,
                    point.candidate.candidate_id,
                ),
            )
        )
        if not ordered:
            raise ValueError("boundary search requires at least one complete mask")
        if len({point.candidate.candidate_id for point in ordered}) != len(ordered):
            raise ValueError("boundary points require unique candidate IDs")
        if any(
            right.released_cost <= left.released_cost + 1e-15
            for left, right in zip(ordered, ordered[1:])
        ):
            raise ValueError(
                "boundary points require strictly increasing released cost"
            )
        self.points = ordered
        self.maximum_quality_loss = float(maximum_quality_loss)
        self.practical_indifference = float(practical_indifference)
        self.cost_tolerance = float(cost_tolerance)
        self.monotonicity_tolerance = float(monotonicity_tolerance)
        self.z_value = float(z_value)
        self.resample_standard_error = float(resample_standard_error)
        self.confidence_level = (
            None if confidence_level is None else float(confidence_level)
        )
        self.maximum_looks_per_candidate = (
            None
            if maximum_looks_per_candidate is None
            else int(maximum_looks_per_candidate)
        )
        self.precision_only_iid = bool(precision_only_iid)
        self.resample_until_resolved = bool(resample_until_resolved)
        self.accept_quality_loss = (
            self.maximum_quality_loss - self.practical_indifference
            if accept_quality_loss is None
            else float(accept_quality_loss)
        )
        self.reject_quality_loss = (
            self.maximum_quality_loss + self.practical_indifference
            if reject_quality_loss is None
            else float(reject_quality_loss)
        )
        for name in (
            "maximum_quality_loss",
            "practical_indifference",
            "cost_tolerance",
            "monotonicity_tolerance",
            "z_value",
            "resample_standard_error",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or (
                name != "maximum_quality_loss" and value < 0.0
            ):
                raise ValueError(f"{name} has an invalid value")
        if self.confidence_level is not None and not (
            0.5 < self.confidence_level < 1.0
        ):
            raise ValueError("confidence_level must lie in (0.5,1)")
        if self.precision_only_iid:
            if self.maximum_looks_per_candidate is not None:
                raise ValueError(
                    "precision-only IID search cannot declare a look horizon"
                )
        elif (
            self.maximum_looks_per_candidate is None
            or self.maximum_looks_per_candidate <= 0
        ):
            raise ValueError("maximum_looks_per_candidate must be positive")
        if (
            not math.isfinite(self.accept_quality_loss)
            or not math.isfinite(self.reject_quality_loss)
            or self.accept_quality_loss > self.reject_quality_loss
        ):
            raise ValueError(
                "boundary accept threshold must not exceed reject threshold"
            )
        self._index = {
            point.candidate.candidate_id: index for index, point in enumerate(ordered)
        }
        self.observations: dict[str, BoundaryObservation] = {}
        self._lower_bound_adjustments: dict[str, float] = {}
        self._upper_bound_adjustments: dict[str, float] = {}
        self._decision_interval_overrides: dict[
            str, tuple[float, float, float, Mapping[str, object]]
        ] = {}
        self.proposal_history: list[BoundaryProposal] = []
        self._latest_proposals: tuple[BoundaryProposal, ...] = ()

    def observe(
        self,
        candidate_id: str,
        values: Sequence[float] | np.ndarray,
        *,
        strata: Sequence[str] | None = None,
        stratum_weights: dict[str, float] | None = None,
    ) -> None:
        candidate = str(candidate_id)
        if candidate not in self._index:
            raise KeyError(f"unknown boundary candidate {candidate!r}")
        observation = self.observations.setdefault(
            candidate, BoundaryObservation(candidate_id=candidate)
        )
        observation.append(
            values,
            strata=strata,
            stratum_weights=stratum_weights,
        )
        self._decision_interval_overrides.pop(candidate, None)

    def observe_moments(
        self,
        candidate_id: str,
        moments: dict[str, RunningMoment],
        *,
        stratum_weights: dict[str, float],
    ) -> None:
        candidate = str(candidate_id)
        if candidate not in self._index:
            raise KeyError(f"unknown boundary candidate {candidate!r}")
        observation = self.observations.setdefault(
            candidate, BoundaryObservation(candidate_id=candidate)
        )
        observation.append_moments(
            moments,
            stratum_weights=stratum_weights,
        )
        self._decision_interval_overrides.pop(candidate, None)

    def _bounds(self, candidate_id: str) -> tuple[float, float]:
        override = self._decision_interval_overrides.get(str(candidate_id))
        if override is not None:
            return float(override[1]), float(override[2])
        observation = self.observations[candidate_id]
        if self.confidence_level is not None:
            if self.precision_only_iid:
                index = max(0, int(observation.look_count) - 1)
                look_alpha = (
                    1.0 - self.confidence_level
                ) / float((index + 1) * (index + 2))
            else:
                assert self.maximum_looks_per_candidate is not None
                look_alpha = staged_alpha(
                    1.0 - self.confidence_level,
                    planned_looks=self.maximum_looks_per_candidate,
                )
            if observation.moments_by_stratum:
                if any(
                    moment.count < 2
                    for moment in observation.moments_by_stratum.values()
                ):
                    return -math.inf, math.inf
                summary = stratified_student_t_summary_from_moments(
                    observation.moments_by_stratum,
                    stratum_weights=observation.stratum_weights,
                    alpha=look_alpha,
                )
                lower, upper = summary.lower, summary.upper
                return (
                    lower + self._lower_bound_adjustments.get(candidate_id, 0.0),
                    upper + self._upper_bound_adjustments.get(candidate_id, 0.0),
                )
            values = np.asarray(observation.values, dtype=np.float64)
            if values.size < 2:
                return -math.inf, math.inf
            interval = student_t_interval(
                float(values.mean()),
                float(values.var(ddof=1)),
                int(values.size),
                alpha=look_alpha,
            )
            return (
                float(interval.lower[()])
                + self._lower_bound_adjustments.get(candidate_id, 0.0),
                float(interval.upper[()])
                + self._upper_bound_adjustments.get(candidate_id, 0.0),
            )
        radius = self.z_value * observation.standard_error
        return (
            observation.mean
            - radius
            + self._lower_bound_adjustments.get(candidate_id, 0.0),
            observation.mean
            + radius
            + self._upper_bound_adjustments.get(candidate_id, 0.0),
        )

    def set_joint_decision_interval(
        self,
        candidate_id: str,
        *,
        estimate: float,
        lower: float,
        upper: float,
        receipt: Mapping[str, object],
    ) -> None:
        """Install one confidence interval for the complete decision statistic."""

        candidate = str(candidate_id)
        values = (float(estimate), float(lower), float(upper))
        if (
            candidate not in self.observations
            or not all(math.isfinite(value) for value in values)
            or values[1] > values[0]
            or values[0] > values[2]
            or not receipt
        ):
            raise ValueError("invalid joint decision interval override")
        self._decision_interval_overrides[candidate] = (
            values[0],
            values[1],
            values[2],
            dict(receipt),
        )

    def joint_decision_interval(
        self,
        candidate_id: str,
    ) -> tuple[float, float, float, Mapping[str, object]] | None:
        return self._decision_interval_overrides.get(str(candidate_id))

    def set_calibration_improvement_bound(
        self,
        candidate_id: str,
        improvement_bound: float,
    ) -> None:
        """Account for further attainable calibration improvement.

        The oracle estimates risk under the currently frozen affine fit.
        Additional calibration can lower, but not raise, the population-optimal
        BCE.  The candidate-specific lower decision bound is therefore shifted
        down by the certified attainable-improvement bound.
        """

        candidate = str(candidate_id)
        bound = float(improvement_bound)
        if candidate not in self._index:
            raise KeyError(f"unknown boundary candidate {candidate!r}")
        if not math.isfinite(bound) or bound < 0.0:
            raise ValueError(
                "calibration improvement bound must be finite and nonnegative"
            )
        self._lower_bound_adjustments[candidate] = -bound
        self._upper_bound_adjustments[candidate] = 0.0

    def _observed_indices(self) -> list[int]:
        return sorted(self._index[candidate_id] for candidate_id in self.observations)

    def _monotonicity_violations(self) -> list[tuple[str, str]]:
        indices = self._observed_indices()
        violations: list[tuple[str, str]] = []
        for left_index, right_index in zip(indices, indices[1:]):
            left = self.points[left_index].candidate.candidate_id
            right = self.points[right_index].candidate.candidate_id
            if (
                self.observations[left].mean
                > self.observations[right].mean + self.monotonicity_tolerance
            ):
                violations.append((left, right))
        return violations

    @property
    def nonmonotone(self) -> bool:
        return bool(self._monotonicity_violations())

    def _predicted_curve(self) -> np.ndarray:
        indices = self._observed_indices()
        if not indices:
            return np.asarray(
                [point.predicted_quality_loss for point in self.points],
                dtype=np.float64,
            )
        observed_means = np.asarray(
            [
                self.observations[self.points[index].candidate.candidate_id].mean
                for index in indices
            ],
            dtype=np.float64,
        )
        weights = np.asarray(
            [
                1.0
                / max(
                    self.observations[
                        self.points[index].candidate.candidate_id
                    ].standard_error
                    ** 2,
                    1e-8,
                )
                for index in indices
            ],
            dtype=np.float64,
        )
        fitted = weighted_isotonic_regression(observed_means, weights)
        costs = np.asarray(
            [point.released_cost for point in self.points], dtype=np.float64
        )
        observed_costs = costs[indices]
        return np.interp(
            costs, observed_costs, fitted, left=fitted[0], right=fitted[-1]
        )

    def _feasible_observed(self) -> list[int]:
        threshold = self.accept_quality_loss
        feasible: list[int] = []
        for candidate_id in self.observations:
            _lower, upper = self._bounds(candidate_id)
            if upper <= threshold:
                feasible.append(self._index[candidate_id])
        return sorted(feasible)

    def _infeasible_observed(self) -> list[int]:
        threshold = self.reject_quality_loss
        infeasible: list[int] = []
        for candidate_id in self.observations:
            lower, _upper = self._bounds(candidate_id)
            if lower > threshold:
                infeasible.append(self._index[candidate_id])
        return sorted(infeasible)

    def _uncertain_observed(self) -> list[int]:
        feasible = set(self._feasible_observed())
        infeasible = set(self._infeasible_observed())
        return [
            index
            for index in self._observed_indices()
            if index not in feasible | infeasible
        ]

    def _record_proposals(
        self, proposals: Sequence[BoundaryProposal]
    ) -> tuple[BoundaryProposal, ...]:
        output = tuple(proposals)
        self.proposal_history.extend(output)
        self._latest_proposals = output
        return output

    @property
    def latest_proposals(self) -> tuple[BoundaryProposal, ...]:
        """Most recent predictable batch, excluding earlier search history."""

        return self._latest_proposals

    def update_decision_thresholds(
        self,
        *,
        accept_quality_loss: float,
        reject_quality_loss: float,
    ) -> None:
        """Install a refined predictable reference interval.

        Observed candidate rows remain valid because only the independently
        measured dense reference changed.  The new thresholds affect the next
        proposal decision and are recorded in ``decision_scope_hash``.
        """

        accept = float(accept_quality_loss)
        reject = float(reject_quality_loss)
        if (
            not math.isfinite(accept)
            or not math.isfinite(reject)
            or accept > reject
        ):
            raise ValueError("refined boundary thresholds are invalid")
        self.accept_quality_loss = accept
        self.reject_quality_loss = reject

    def decision_relevant_candidate_ids(self) -> tuple[str, ...]:
        """Return the active complete-mask bracket, not the global score order."""

        indices: set[int] = {
            self._index[proposal.candidate_id]
            for proposal in self._latest_proposals
            if proposal.action == "evaluate" and proposal.candidate_id is not None
        }
        feasible = self._feasible_observed()
        infeasible = self._infeasible_observed()
        uncertain = self._uncertain_observed()
        indices.update(uncertain)
        left = max(feasible) if feasible else None
        if left is not None:
            indices.add(left)
        right_candidates = (
            [index for index in infeasible if left is None or index > left]
            if infeasible
            else []
        )
        right = (
            min(right_candidates)
            if right_candidates
            else (min(infeasible) if infeasible else None)
        )
        if right is not None:
            indices.add(right)
        if left is not None and right is not None and left < right:
            indices.update(range(left, right + 1))
        elif self.nonmonotone:
            indices.update(range(len(self.points)))
        if not indices:
            closest = min(
                range(len(self.points)),
                key=lambda index: (
                    abs(
                        self.points[index].predicted_quality_loss
                        - self.maximum_quality_loss
                    ),
                    self.points[index].expected_oracle_cost,
                    self.points[index].candidate.candidate_id,
                ),
            )
            indices.add(closest)
        return tuple(
            self.points[index].candidate.candidate_id for index in sorted(indices)
        )

    @property
    def decision_scope_hash(self) -> str:
        relevant = self.decision_relevant_candidate_ids()
        return content_hash(
            {
                "maximum_quality_loss": self.maximum_quality_loss,
                "accept_quality_loss": self.accept_quality_loss,
                "reject_quality_loss": self.reject_quality_loss,
                "practical_indifference": self.practical_indifference,
                "cost_tolerance": self.cost_tolerance,
                "candidate_axis": tuple(
                    (
                        point.candidate.candidate_id,
                        point.candidate.resulting_mask.mask_hash,
                        point.released_cost,
                        point.predicted_quality_loss,
                        point.predicted_standard_error,
                    )
                    for point in self.points
                ),
                "relevant_candidate_ids": relevant,
                "observations": tuple(
                    (
                        candidate_id,
                        tuple(self.observations[candidate_id].values),
                        tuple(
                            (
                                key,
                                value.count,
                                value.mean,
                                value.m2,
                            )
                            for key, value in sorted(
                                self.observations[
                                    candidate_id
                                ].moments_by_stratum.items()
                            )
                        ),
                    )
                    for candidate_id in sorted(self.observations)
                ),
                "joint_decision_intervals": tuple(
                    (
                        candidate_id,
                        estimate,
                        lower,
                        upper,
                        dict(receipt),
                    )
                    for candidate_id, (
                        estimate,
                        lower,
                        upper,
                        receipt,
                    ) in sorted(self._decision_interval_overrides.items())
                ),
                "latest_proposals": tuple(
                    (
                        proposal.candidate_id,
                        proposal.action,
                        proposal.reason,
                        proposal.resample,
                    )
                    for proposal in self._latest_proposals
                ),
            },
            prefix="irg-boundary-decision-scope-",
        )

    def propose_batch(self, maximum_candidates: int) -> tuple[BoundaryProposal, ...]:
        """Return a diverse batch of consequential oracle experiments.

        Every proposal is computed from the same pre-request information state,
        making the batch safe for common-context packed execution.  The policy
        prioritizes bracket endpoints, uncertain observations, and diverse
        interior points near the active quality boundary rather than issuing a
        serial one-mask request after the endpoints.
        """

        limit = int(maximum_candidates)
        if limit <= 0:
            raise ValueError("maximum_candidates must be positive")
        observed = set(self.observations)
        proposals: list[BoundaryProposal] = []

        # The minimal bracket can be acquired in one packed request.
        for index, reason in (
            (0, "establish conservative endpoint"),
            (len(self.points) - 1, "establish aggressive endpoint"),
        ):
            candidate_id = self.points[index].candidate.candidate_id
            if candidate_id not in observed and all(
                item.candidate_id != candidate_id for item in proposals
            ):
                proposals.append(BoundaryProposal(candidate_id, "evaluate", reason))
                if len(proposals) >= limit:
                    return self._record_proposals(proposals)

        uncertain = self._uncertain_observed()
        if self.resample_until_resolved or self.resample_standard_error > 0.0:
            resamples = sorted(
                (
                    index
                    for index in uncertain
                    if self.resample_until_resolved
                    or self.observations[
                        self.points[index].candidate.candidate_id
                    ].standard_error > self.resample_standard_error
                ),
                key=lambda index: (
                    -self.observations[
                        self.points[index].candidate.candidate_id
                    ].standard_error,
                    self.points[index].expected_oracle_cost,
                    self.points[index].candidate.candidate_id,
                ),
            )
            for index in resamples:
                proposals.append(
                    BoundaryProposal(
                        self.points[index].candidate.candidate_id,
                        "evaluate",
                        "reduce uncertainty at the active quality boundary",
                        resample=True,
                    )
                )
                if len(proposals) >= limit:
                    return self._record_proposals(proposals)

        unobserved_indices = [
            index
            for index, point in enumerate(self.points)
            if point.candidate.candidate_id not in observed
            and all(
                item.candidate_id != point.candidate.candidate_id for item in proposals
            )
        ]
        feasible = self._feasible_observed()
        infeasible = self._infeasible_observed()
        left = max(feasible) if feasible else -1
        if infeasible:
            right_candidates = [index for index in infeasible if index > left]
            right = min(right_candidates) if right_candidates else min(infeasible)
        else:
            right = len(self.points)

        if left >= 0 and right < len(self.points):
            width = self.points[right].released_cost - self.points[left].released_cost
            between = [index for index in unobserved_indices if left < index < right]
            if width <= self.cost_tolerance or not between:
                if not proposals:
                    return self._record_proposals(
                        (
                            BoundaryProposal(
                                None,
                                "stop",
                                "feasible/infeasible complete-mask bracket resolved",
                            ),
                        )
                    )
                return self._record_proposals(proposals)

            # Pick cost-quantile representatives across the unresolved bracket.
            # This maximizes information per launch while avoiding near-duplicate
            # masks in one packed request.
            target_count = min(limit - len(proposals), len(between))
            left_cost = self.points[left].released_cost
            right_cost = self.points[right].released_cost
            target_costs = np.linspace(left_cost, right_cost, target_count + 2)[1:-1]
            remaining = set(between)
            for target_cost in target_costs:
                index = min(
                    remaining,
                    key=lambda item: (
                        abs(self.points[item].released_cost - float(target_cost)),
                        abs(
                            self.points[item].predicted_quality_loss
                            - self.maximum_quality_loss
                        ),
                        self.points[item].expected_oracle_cost,
                        self.points[item].candidate.candidate_id,
                    ),
                )
                remaining.remove(index)
                proposals.append(
                    BoundaryProposal(
                        self.points[index].candidate.candidate_id,
                        "evaluate",
                        "resolve a diverse interior point of the complete-mask boundary bracket",
                    )
                )
            return self._record_proposals(proposals)

        if not unobserved_indices:
            if proposals:
                return self._record_proposals(proposals)
            return self._record_proposals(
                (
                    BoundaryProposal(
                        None, "stop", "all version-induced masks have exact evaluations"
                    ),
                )
            )

        curve = self._predicted_curve()
        if self.nonmonotone:
            observed_indices = self._observed_indices()

            def relevance(index: int) -> tuple[float, float, str]:
                nearest = min(abs(index - item) for item in observed_indices)
                boundary = 1.0 / (abs(curve[index] - self.maximum_quality_loss) + 1e-6)
                value = (
                    boundary * (1.0 + nearest) / self.points[index].expected_oracle_cost
                )
                return (
                    value,
                    self.points[index].released_cost,
                    self.points[index].candidate.candidate_id,
                )

            ranked = sorted(unobserved_indices, key=relevance, reverse=True)
            reason = (
                "nonmonotone fallback: query high frontier relevance per oracle cost"
            )
        elif not feasible:
            ranked = sorted(
                unobserved_indices,
                key=lambda index: (
                    abs(curve[index] - self.maximum_quality_loss),
                    self.points[index].expected_oracle_cost,
                    index,
                ),
            )
            reason = "search for the first feasible complete mask near the predicted boundary"
        else:
            ranked = sorted(
                unobserved_indices,
                key=lambda index: (
                    abs(curve[index] - self.maximum_quality_loss),
                    -self.points[index].released_cost,
                    self.points[index].expected_oracle_cost,
                ),
            )
            reason = "extend the measured feasible frontier near the predicted quality boundary"
        for index in ranked[: max(0, limit - len(proposals))]:
            proposals.append(
                BoundaryProposal(
                    self.points[index].candidate.candidate_id, "evaluate", reason
                )
            )
        return self._record_proposals(proposals)

    def propose_next(self) -> BoundaryProposal:
        return self.propose_batch(1)[0]

    def best_feasible_candidate_id(self) -> str:
        feasible = self._feasible_observed()
        if not feasible:
            raise ValueError("no confidently feasible complete mask has been measured")
        index = max(feasible)
        return self.points[index].candidate.candidate_id

    def diagnostics(self) -> dict[str, object]:
        return {
            "observed_candidate_ids": sorted(self.observations),
            "feasible_indices": self._feasible_observed(),
            "infeasible_indices": self._infeasible_observed(),
            "uncertain_indices": self._uncertain_observed(),
            "nonmonotone": self.nonmonotone,
            "monotonicity_violations": self._monotonicity_violations(),
            "proposal_count": len(self.proposal_history),
            "confidence_level": self.confidence_level,
            "maximum_looks_per_candidate": self.maximum_looks_per_candidate,
            "precision_only_iid": self.precision_only_iid,
            "resample_until_resolved": self.resample_until_resolved,
            "accept_quality_loss": self.accept_quality_loss,
            "reject_quality_loss": self.reject_quality_loss,
        }


__all__ = [
    "AdaptiveBoundarySearch",
    "BoundaryObservation",
    "BoundaryPoint",
    "BoundaryProposal",
    "weighted_isotonic_regression",
]
