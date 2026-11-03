"""Exact candidate ledger and nondominated frontier for IRG pruning."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mechanistic_systems.inference.sequential_bce import RunningMoment

from .candidates import ActionCandidate
from .contracts import ActionPrediction, canonical_json, content_hash
from .costs import RealizedCost


class CandidateStatus(str, Enum):
    GENERATED = "generated"
    QUEUED = "queued"
    EVALUATED = "evaluated"
    ELIMINATED = "eliminated"
    COMMITTED = "committed"
    REFUSED = "refused"


@dataclass(frozen=True)
class OracleReceipt:
    candidate_id: str
    state_hash: str
    context_ids: tuple[str, ...]
    paired_quality_deltas: tuple[float, ...]
    calibration_state_id: str
    realized_cost: RealizedCost
    measurement_request_id: str
    metric_id: str
    stratified_moments: Mapping[str, Mapping[str, float | int]] = field(
        default_factory=dict
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        if not all(
            (
                self.candidate_id,
                self.state_hash,
                self.calibration_state_id,
                self.measurement_request_id,
                self.metric_id,
            )
        ):
            raise ValueError("oracle receipt has an empty required identity")
        values = np.asarray(self.paired_quality_deltas, dtype=np.float64)
        moments = {
            str(key): RunningMoment.from_mapping(value)
            for key, value in dict(self.stratified_moments).items()
        }
        if values.size == 0 and not moments:
            raise ValueError(
                "oracle receipt requires paired deltas or stratified moments"
            )
        if values.size and not np.isfinite(values).all():
            raise ValueError("oracle receipt paired deltas must be finite")
        evidence_count = (
            int(values.size)
            if values.size
            else sum(moment.count for moment in moments.values())
        )
        if self.context_ids and len(self.context_ids) != evidence_count:
            raise ValueError("oracle context count must match evidence count")
        if len(set(self.context_ids)) != len(self.context_ids):
            raise ValueError("oracle receipt context_ids must be unique")
        object.__setattr__(
            self,
            "stratified_moments",
            {key: value.to_dict() for key, value in moments.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        canonical_json(self.metadata)
        expected = self.compute_hash()
        if self.receipt_hash and self.receipt_hash != expected:
            raise ValueError("oracle receipt content hash mismatch")
        object.__setattr__(self, "receipt_hash", expected)

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        state_hash: str,
        context_ids: Sequence[str],
        paired_quality_deltas: Sequence[float] | np.ndarray = (),
        stratified_moments: Mapping[
            str, Mapping[str, float | int]
        ] | None = None,
        calibration_state_id: str,
        realized_cost: RealizedCost,
        measurement_request_id: str,
        metric_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "OracleReceipt":
        return cls(
            candidate_id=str(candidate_id),
            state_hash=str(state_hash),
            context_ids=tuple(str(item) for item in context_ids),
            paired_quality_deltas=tuple(
                float(value) for value in np.asarray(paired_quality_deltas).reshape(-1)
            ),
            stratified_moments=dict(stratified_moments or {}),
            calibration_state_id=str(calibration_state_id),
            realized_cost=realized_cost,
            measurement_request_id=str(measurement_request_id),
            metric_id=str(metric_id),
            metadata=dict(metadata or {}),
        )

    @property
    def count(self) -> int:
        if self.paired_quality_deltas:
            return len(self.paired_quality_deltas)
        return int(
            sum(
                RunningMoment.from_mapping(value).count
                for value in self.stratified_moments.values()
            )
        )

    @property
    def mean_quality_loss(self) -> float:
        if self.paired_quality_deltas:
            return float(np.mean(self.paired_quality_deltas))
        moments = {
            key: RunningMoment.from_mapping(value)
            for key, value in self.stratified_moments.items()
        }
        weights = {
            str(key): float(value)
            for key, value in dict(
                self.metadata.get("oracle_stratum_weights", {})
            ).items()
        }
        if set(weights) != set(moments):
            weights = {
                key: moment.count / float(self.count)
                for key, moment in moments.items()
            }
        return float(
            sum(weights[key] * moment.mean for key, moment in moments.items())
        )

    @property
    def variance(self) -> float:
        values = np.asarray(self.paired_quality_deltas, dtype=np.float64)
        if values.size:
            return float(np.var(values, ddof=1)) if values.size >= 2 else 0.0
        moments = {
            key: RunningMoment.from_mapping(value)
            for key, value in self.stratified_moments.items()
        }
        weights = {
            str(key): float(value)
            for key, value in dict(
                self.metadata.get("oracle_stratum_weights", {})
            ).items()
        }
        if set(weights) != set(moments):
            weights = {
                key: moment.count / float(self.count)
                for key, moment in moments.items()
            }
        # This is the variance of the stratified estimator's observations,
        # retained for the legacy ``standard_error`` property below.
        return float(
            sum(
                weights[key] ** 2
                * moment.sample_variance
                * float(self.count)
                / float(moment.count)
                for key, moment in moments.items()
            )
        )

    @property
    def standard_error(self) -> float:
        return math.sqrt(self.variance / self.count) if self.count else float("inf")

    def compute_hash(self) -> str:
        return content_hash(
            {
                "candidate_id": self.candidate_id,
                "state_hash": self.state_hash,
                "context_ids": self.context_ids,
                "paired_quality_deltas": self.paired_quality_deltas,
                "stratified_moments": self.stratified_moments,
                "calibration_state_id": self.calibration_state_id,
                "realized_cost": self.realized_cost,
                "measurement_request_id": self.measurement_request_id,
                "metric_id": self.metric_id,
                "metadata": self.metadata,
            },
            prefix="irg-oracle-",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "state_hash": self.state_hash,
            "context_ids": list(self.context_ids),
            "paired_quality_deltas": list(self.paired_quality_deltas),
            "stratified_moments": dict(self.stratified_moments),
            "calibration_state_id": self.calibration_state_id,
            "realized_cost": self.realized_cost.to_dict(),
            "measurement_request_id": self.measurement_request_id,
            "metric_id": self.metric_id,
            "metadata": dict(self.metadata),
            "receipt_hash": self.receipt_hash,
            "mean_quality_loss": self.mean_quality_loss,
            "variance": self.variance,
            "standard_error": self.standard_error,
        }


@dataclass
class CandidateLedgerEntry:
    candidate: ActionCandidate
    status: CandidateStatus = CandidateStatus.GENERATED
    prediction: ActionPrediction | None = None
    oracle_receipts: list[OracleReceipt] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        return bool(self.oracle_receipts)

    @property
    def mean_quality_loss(self) -> float | None:
        if not self.oracle_receipts:
            return None
        total = sum(receipt.count for receipt in self.oracle_receipts)
        return float(
            sum(
                receipt.mean_quality_loss * receipt.count
                for receipt in self.oracle_receipts
            )
            / float(total)
        )

    @property
    def standard_error(self) -> float | None:
        if not self.oracle_receipts:
            return None
        moments = RunningMoment.empty()
        for receipt in self.oracle_receipts:
            if receipt.paired_quality_deltas:
                packet = RunningMoment.from_values(
                    receipt.paired_quality_deltas
                )
            else:
                # Frontier uncertainty is descriptive; the decision engine
                # retains the exact stratified projection.  Merge a
                # count/mean/variance-equivalent packet here.
                packet = RunningMoment(
                    count=receipt.count,
                    mean=receipt.mean_quality_loss,
                    m2=receipt.variance * max(0, receipt.count - 1),
                )
            moments = moments.merge(packet)
        if moments.count < 2:
            return 0.0
        return moments.standard_error

    def record_decision(self, action: str, reason: str, **details: Any) -> None:
        if not action or not reason:
            raise ValueError("candidate decision requires action and reason")
        record = {
            "action": str(action),
            "reason": str(reason),
            "details": dict(details),
        }
        record["decision_hash"] = content_hash(record, prefix="irg-decision-")
        self.decisions.append(record)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "status": self.status.value,
            "prediction": (
                None
                if self.prediction is None
                else {
                    "candidate_id": self.prediction.candidate_id,
                    "version": self.prediction.version,
                    "predicted_quality_loss": self.prediction.predicted_quality_loss,
                    "standard_error": self.prediction.standard_error,
                    "model_component": self.prediction.model_component,
                    "state_hash": self.prediction.state_hash,
                    "details": dict(self.prediction.details),
                }
            ),
            "oracle_receipts": [receipt.to_dict() for receipt in self.oracle_receipts],
            "decisions": list(self.decisions),
            "mean_quality_loss": self.mean_quality_loss,
            "standard_error": self.standard_error,
        }


class FrontierLedger:
    """Append-only candidate registry and exact measured frontier."""

    def __init__(self, *, primary_cost_coordinate: str = "route_count") -> None:
        self.primary_cost_coordinate = str(primary_cost_coordinate)
        if not self.primary_cost_coordinate:
            raise ValueError("frontier requires a primary cost coordinate")
        self._entries: dict[str, CandidateLedgerEntry] = {}
        self._events: list[dict[str, Any]] = []

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._events)

    def register(self, candidate: ActionCandidate) -> CandidateLedgerEntry:
        existing = self._entries.get(candidate.candidate_id)
        if existing is not None:
            if (
                existing.candidate.resulting_mask.mask_hash
                != candidate.resulting_mask.mask_hash
            ):
                raise ValueError("candidate ID collision")
            return existing
        if any(
            entry.candidate.resulting_mask.mask_hash
            == candidate.resulting_mask.mask_hash
            and entry.candidate.parent_state_hash == candidate.parent_state_hash
            for entry in self._entries.values()
        ):
            raise ValueError(
                "duplicate complete mask entered under a different candidate ID "
                "for the same predictable parent state"
            )
        entry = CandidateLedgerEntry(candidate=candidate)
        self._entries[candidate.candidate_id] = entry
        self._event(
            "register",
            candidate.candidate_id,
            mask_hash=candidate.resulting_mask.mask_hash,
        )
        return entry

    def entry(self, candidate_id: str) -> CandidateLedgerEntry:
        try:
            return self._entries[str(candidate_id)]
        except KeyError as exc:
            raise KeyError(f"unknown candidate {candidate_id!r}") from exc

    def record_prediction(self, prediction: ActionPrediction) -> None:
        entry = self.entry(prediction.candidate_id)
        if prediction.state_hash != entry.candidate.parent_state_hash:
            raise ValueError("prediction state does not match candidate parent state")
        entry.prediction = prediction
        self._event(
            "prediction",
            prediction.candidate_id,
            model_component=prediction.model_component,
        )

    def queue(self, candidate_id: str, *, reason: str) -> None:
        entry = self.entry(candidate_id)
        if entry.status in {CandidateStatus.COMMITTED, CandidateStatus.REFUSED}:
            raise ValueError("terminal candidate cannot be queued")
        entry.status = CandidateStatus.QUEUED
        entry.record_decision("queue", reason)
        self._event("queue", candidate_id, reason=reason)

    def record_oracle(self, receipt: OracleReceipt) -> None:
        entry = self.entry(receipt.candidate_id)
        if receipt.state_hash != entry.candidate.parent_state_hash:
            raise ValueError(
                "oracle receipt state does not match candidate parent state"
            )
        if any(
            item.receipt_hash == receipt.receipt_hash for item in entry.oracle_receipts
        ):
            return
        existing_contexts = {
            context_id
            for item in entry.oracle_receipts
            for context_id in item.context_ids
        }
        overlap = existing_contexts.intersection(receipt.context_ids)
        if overlap:
            raise ValueError("oracle receipt repeats contexts for the same candidate")
        entry.oracle_receipts.append(receipt)
        entry.status = CandidateStatus.EVALUATED
        self._event(
            "oracle",
            receipt.candidate_id,
            receipt_hash=receipt.receipt_hash,
            count=receipt.count,
            mean_quality_loss=receipt.mean_quality_loss,
        )

    def eliminate(self, candidate_id: str, *, reason: str, **details: Any) -> None:
        entry = self.entry(candidate_id)
        if entry.status == CandidateStatus.COMMITTED:
            raise ValueError("committed candidate cannot be eliminated")
        entry.status = CandidateStatus.ELIMINATED
        entry.record_decision("eliminate", reason, **details)
        self._event("eliminate", candidate_id, reason=reason, **details)

    def refuse(self, candidate_id: str, *, reason: str, **details: Any) -> None:
        entry = self.entry(candidate_id)
        if entry.status == CandidateStatus.COMMITTED:
            raise ValueError("committed candidate cannot be refused")
        entry.status = CandidateStatus.REFUSED
        entry.record_decision("refuse", reason, **details)
        self._event("refuse", candidate_id, reason=reason, **details)

    def preflight_commit(self, candidate_id: str) -> CandidateLedgerEntry:
        """Validate every deterministic commit invariant without mutation."""

        entry = self.entry(candidate_id)
        if not entry.measured:
            raise ValueError("candidate requires an exact oracle receipt before commit")
        if entry.status in {
            CandidateStatus.ELIMINATED,
            CandidateStatus.COMMITTED,
            CandidateStatus.REFUSED,
        }:
            raise ValueError("terminal candidate cannot be committed")
        for other in self._entries.values():
            if (
                other.status == CandidateStatus.COMMITTED
                and other.candidate.parent_mask_hash == entry.candidate.parent_mask_hash
            ):
                raise ValueError("one parent mask may have only one committed child")
        return entry

    def commit(self, candidate_id: str, *, reason: str, **details: Any) -> None:
        entry = self.preflight_commit(candidate_id)
        entry.status = CandidateStatus.COMMITTED
        entry.record_decision("commit", reason, **details)
        self._event("commit", candidate_id, reason=reason, **details)

    def measured_entries(self) -> tuple[CandidateLedgerEntry, ...]:
        return tuple(
            sorted(
                (entry for entry in self._entries.values() if entry.measured),
                key=lambda item: item.candidate.candidate_id,
            )
        )

    def nondominated(
        self, *, tolerance: float = 1e-12
    ) -> tuple[CandidateLedgerEntry, ...]:
        """Return measured points nondominated in released-cost/quality space.

        More released cost is better; lower quality loss is better.
        """

        measured = self.measured_entries()
        frontier: list[CandidateLedgerEntry] = []
        for candidate in measured:
            candidate_cost = candidate.candidate.released_cost.coordinate(
                self.primary_cost_coordinate
            )
            candidate_loss = float(candidate.mean_quality_loss)
            dominated = False
            for other in measured:
                if other is candidate:
                    continue
                other_cost = other.candidate.released_cost.coordinate(
                    self.primary_cost_coordinate
                )
                other_loss = float(other.mean_quality_loss)
                no_worse = (
                    other_cost >= candidate_cost - tolerance
                    and other_loss <= candidate_loss + tolerance
                )
                strictly_better = (
                    other_cost > candidate_cost + tolerance
                    or other_loss < candidate_loss - tolerance
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                frontier.append(candidate)
        return tuple(
            sorted(
                frontier,
                key=lambda item: (
                    item.candidate.released_cost.coordinate(
                        self.primary_cost_coordinate
                    ),
                    float(item.mean_quality_loss),
                    item.candidate.candidate_id,
                ),
            )
        )

    def best_feasible(self, *, maximum_quality_loss: float) -> CandidateLedgerEntry:
        threshold = float(maximum_quality_loss)
        if not math.isfinite(threshold):
            raise ValueError("maximum_quality_loss must be finite")
        feasible = [
            entry
            for entry in self.nondominated()
            if float(entry.mean_quality_loss) <= threshold
            and entry.status
            not in {CandidateStatus.REFUSED, CandidateStatus.ELIMINATED}
        ]
        if not feasible:
            raise ValueError("no measured candidate satisfies the quality ceiling")
        return max(
            feasible,
            key=lambda entry: (
                entry.candidate.released_cost.coordinate(self.primary_cost_coordinate),
                -float(entry.mean_quality_loss),
                entry.candidate.candidate_id,
            ),
        )

    def _event(self, kind: str, candidate_id: str, **details: Any) -> None:
        event = {
            "sequence": len(self._events),
            "kind": str(kind),
            "candidate_id": str(candidate_id),
            "details": dict(details),
        }
        event["event_hash"] = content_hash(event, prefix="irg-event-")
        self._events.append(event)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "irg_frontier_ledger_v1",
            "primary_cost_coordinate": self.primary_cost_coordinate,
            "entries": [
                self._entries[candidate_id].to_dict()
                for candidate_id in self.candidate_ids
            ],
            "events": list(self._events),
            "ledger_hash": content_hash(
                {
                    "primary_cost_coordinate": self.primary_cost_coordinate,
                    "entries": [
                        self._entries[candidate_id].to_dict()
                        for candidate_id in self.candidate_ids
                    ],
                    "events": self._events,
                },
                prefix="irg-ledger-",
            ),
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


__all__ = [
    "CandidateLedgerEntry",
    "CandidateStatus",
    "FrontierLedger",
    "OracleReceipt",
]
