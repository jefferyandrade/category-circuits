"""Run-level alpha ledger: one explicit error budget across every accept.

The greedy search makes an a-priori-unknown number of irreversible accepts,
each certified by a one-sided bound.  The ledger pre-registers one total error
budget and spends it across accepts in proportion to their saved cost, so the
probability that ANY accepted decision violated its certified bound is at most
``total_alpha`` for the whole run, however many decisions were taken.  The
spent ledger serializes into the run receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlphaLedgerEntry:
    decision_id: str
    kind: str
    weight: float
    alpha_allocated: float
    outcome: str
    charged: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.decision_id or not self.kind or not self.outcome:
            raise ValueError("ledger entries require identity, kind, and outcome")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("ledger entry weight must be finite and positive")
        if not 0.0 < self.alpha_allocated < 1.0:
            raise ValueError("ledger entry alpha must lie in (0,1)")


class AlphaWealthLedger:
    """Cost-proportional alpha spending with a hard total budget."""

    def __init__(
        self,
        *,
        total_alpha: float,
        total_weight: float,
        minimum_alpha: float = 1e-7,
    ) -> None:
        if not 0.0 < float(total_alpha) < 1.0:
            raise ValueError("ledger requires total alpha in (0,1)")
        if not math.isfinite(float(total_weight)) or float(total_weight) <= 0.0:
            raise ValueError("ledger requires positive total weight")
        if not 0.0 < float(minimum_alpha) < float(total_alpha):
            raise ValueError("ledger minimum alpha must lie below the total budget")
        self.total_alpha = float(total_alpha)
        self.total_weight = float(total_weight)
        self.minimum_alpha = float(minimum_alpha)
        self.entries: list[AlphaLedgerEntry] = []

    @property
    def spent_alpha(self) -> float:
        return float(
            sum(entry.alpha_allocated for entry in self.entries if entry.charged)
        )

    @property
    def remaining_alpha(self) -> float:
        return max(0.0, self.total_alpha - self.spent_alpha)

    def preview(self, weight: float) -> float:
        """The alpha a decision of this weight would receive right now.

        Rejections consume no budget (the guarantee is one-sided, on accepted
        prunes), so bounds are computed at this previewed alpha and the ledger
        is charged only if the decision commits.
        """

        proportional = self.total_alpha * float(weight) / self.total_weight
        allocation = max(self.minimum_alpha, proportional)
        return float(min(allocation, max(self.remaining_alpha, self.minimum_alpha)))

    def allocate(self, weight: float, *, decision_id: str, kind: str) -> float:
        """Alpha for one committing decision, capped by the remaining budget."""

        del kind
        if any(
            entry.decision_id == decision_id and entry.charged
            for entry in self.entries
        ):
            raise ValueError(f"ledger already spent for decision {decision_id!r}")
        remaining = self.remaining_alpha
        if remaining <= self.minimum_alpha:
            raise RuntimeError(
                "alpha ledger exhausted: the run made more certified decisions "
                "than its pre-registered budget supports"
            )
        return self.preview(weight)

    def record(
        self,
        *,
        decision_id: str,
        kind: str,
        weight: float,
        alpha_allocated: float,
        outcome: str,
        charged: bool = True,
        metadata: dict[str, object] | None = None,
    ) -> AlphaLedgerEntry:
        entry = AlphaLedgerEntry(
            decision_id=str(decision_id),
            kind=str(kind),
            weight=float(weight),
            alpha_allocated=float(alpha_allocated),
            outcome=str(outcome),
            charged=bool(charged),
            metadata=dict(metadata or {}),
        )
        entry.validate()
        if charged and self.spent_alpha + entry.alpha_allocated > self.total_alpha + 1e-12:
            raise RuntimeError("alpha ledger overspent its pre-registered budget")
        self.entries.append(entry)
        return entry

    def receipt(self) -> dict[str, object]:
        payload = {
            "schema_name": "evidence_foundation_alpha_ledger_v1",
            "total_alpha": self.total_alpha,
            "total_weight": self.total_weight,
            "minimum_alpha": self.minimum_alpha,
            "spent_alpha": self.spent_alpha,
            "entry_count": len(self.entries),
            "entries": [
                {
                    "decision_id": entry.decision_id,
                    "kind": entry.kind,
                    "weight": entry.weight,
                    "alpha_allocated": entry.alpha_allocated,
                    "outcome": entry.outcome,
                    "charged": entry.charged,
                    "metadata": entry.metadata,
                }
                for entry in self.entries
            ],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()
        payload["ledger_hash"] = digest
        return payload


__all__ = ["AlphaLedgerEntry", "AlphaWealthLedger"]
