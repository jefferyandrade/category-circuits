"""Bounded process-local handoff for typed whole-route response summaries.

The optimized response provider and adaptive acquisition adapter execute in
one Python process. Serializing million-route sufficient statistics between
those two call frames is pure controller overhead. This registry carries the
already validated typed objects across that joint while the ordinary
``MeasurementOutcome`` retains a compact, hash-bound lineage receipt.

The registry is deliberately not replay authority: a token is claimed once,
missing tokens fail closed, and the packed outcome path remains available for
compatibility and external replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Sequence

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.core.axis_identity import (
    canonical_unique_string_axis,
    ordered_string_axis_hash,
)


def _typed_hash(value: object | None) -> str | None:
    if value is None:
        return None
    identity = getattr(value, "content_hash", None)
    if not isinstance(identity, str) or not identity:
        raise TypeError("in-process response component lacks content_hash")
    return identity


@dataclass(frozen=True)
class InProcessResponseSummary:
    """One physical stratum's immutable typed sufficient statistics."""

    request_id: str
    state_hash: str
    context_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    response_moments: object
    local_attribution: object | None
    tripartite_pullback: object | None
    global_trust_geometry: object | None
    intrinsic_atlas_geometry: object | None = None
    schema: str = "irg_in_process_response_summary_v1"

    def __post_init__(self) -> None:
        route_axis = canonical_unique_string_axis(
            self.route_ids,
            name="in-process response route_ids",
        )
        if (
            not self.request_id
            or not self.state_hash
            or not self.context_ids
            or len(set(self.context_ids)) != len(self.context_ids)
        ):
            raise ValueError("in-process response summary identity is invalid")
        object.__setattr__(self, "route_ids", route_axis)
        for component in (
            self.response_moments,
            self.local_attribution,
            self.tripartite_pullback,
            self.global_trust_geometry,
            self.intrinsic_atlas_geometry,
        ):
            if component is None:
                continue
            component_routes = getattr(component, "route_ids", ())
            if component_routes is not self.route_ids and component_routes != self.route_ids:
                raise ValueError(
                    "in-process response component uses another route axis"
                )
            _typed_hash(component)

    @property
    def route_axis_hash(self) -> str:
        return ordered_string_axis_hash(
            self.route_ids,
            name="in-process response route_ids",
        )

    @property
    def component_hashes(self) -> dict[str, str | None]:
        return {
            "response_moments": _typed_hash(self.response_moments),
            "local_attribution": _typed_hash(self.local_attribution),
            "tripartite_pullback": _typed_hash(self.tripartite_pullback),
            "global_trust_geometry": _typed_hash(self.global_trust_geometry),
            "intrinsic_atlas_geometry": _typed_hash(self.intrinsic_atlas_geometry),
        }

    @property
    def token(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "request_id": self.request_id,
                "state_hash": self.state_hash,
                "context_ids": self.context_ids,
                "route_count": len(self.route_ids),
                "route_axis_hash": self.route_axis_hash,
                "component_hashes": self.component_hashes,
            },
            prefix="irg-in-process-response-summary-",
        )


_LOCK = RLock()
_SUMMARIES: dict[str, InProcessResponseSummary] = {}
_MAXIMUM_PENDING_SUMMARIES = 256


def register_in_process_response_summary(
    summary: InProcessResponseSummary,
) -> str:
    """Register one summary for exactly one immediate adapter consumer."""

    if not isinstance(summary, InProcessResponseSummary):
        raise TypeError("response summary registry requires a typed summary")
    token = summary.token
    with _LOCK:
        existing = _SUMMARIES.get(token)
        if existing is not None and existing != summary:
            raise RuntimeError("response summary token collision")
        if existing is None and len(_SUMMARIES) >= _MAXIMUM_PENDING_SUMMARIES:
            raise RuntimeError(
                "in-process response summary producer outran its consumer"
            )
        _SUMMARIES[token] = summary
    return token


def claim_in_process_response_summary(
    token: str,
    *,
    request_id: str,
    state_hash: str,
    context_ids: Sequence[str],
    route_count: int,
    route_axis_hash: str,
    component_hashes: object,
) -> InProcessResponseSummary:
    """Claim and validate one process-local summary exactly once."""

    identity = str(token)
    with _LOCK:
        summary = _SUMMARIES.pop(identity, None)
    if summary is None:
        raise RuntimeError(
            "typed response summary is unavailable; compact in-process "
            "outcomes require recomputation when replayed"
        )
    if (
        summary.request_id != str(request_id)
        or summary.state_hash != str(state_hash)
        or summary.context_ids != tuple(map(str, context_ids))
        or len(summary.route_ids) != int(route_count)
        or summary.route_axis_hash != str(route_axis_hash)
        or not isinstance(component_hashes, dict)
        or dict(component_hashes) != summary.component_hashes
        or summary.token != identity
    ):
        raise RuntimeError("typed response summary lineage check failed")
    return summary


__all__ = [
    "InProcessResponseSummary",
    "claim_in_process_response_summary",
    "register_in_process_response_summary",
]
