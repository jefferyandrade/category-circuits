"""Measurement request, outcome, and provider contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Protocol, Sequence, TypeAlias

import numpy as np

from .content_address import canonical_json, content_hash
from .measurement_costs import CostEstimate, RealizedCost

V6_UNHASHED_OUTCOME_PREFIX = "irg-outcome-v6-unhashed:"


MeasurementKind: TypeAlias = Literal[
    "calibration_logits",
    "calibration_refit",
    "jacobian_rows",
    "marginal_statistics",
    "metric_factor",
    "directional_curvature",
    "hard_mask_oracle",
    "model_validity",
    "response_jacobian",
    "atomic_intervention",
    "atomic_pair_intervention",
    "atom_stability",
    "response_moments",
    "response_factor",
    "covariance_rank_expansion",
    "response_transition",
    "finite_dose_response",
]


@dataclass(frozen=True)
class ContextSelector:
    selector_id: str
    requested_count: int
    strata: tuple[str, ...] = ()
    context_ids: tuple[str, ...] = ()
    random_seed: int = 0
    minimum_per_stratum: int = 0
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.selector_id:
            raise ValueError("context selector requires selector_id")
        if int(self.requested_count) < 0:
            raise ValueError("requested_count must be nonnegative")
        if int(self.minimum_per_stratum) < 0:
            raise ValueError("minimum_per_stratum must be nonnegative")
        if self.context_ids and self.requested_count not in (0, len(self.context_ids)):
            raise ValueError("explicit context_ids must match requested_count")
        if len(set(self.context_ids)) != len(self.context_ids):
            raise ValueError("context_ids must be unique")
        object.__setattr__(self, "parameters", dict(self.parameters))
        canonical_json(self.parameters)


@dataclass(frozen=True)
class MeasurementProvenance:
    producer_id: str
    producer_version: str
    code_hash: str
    parent_request_ids: tuple[str, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.producer_id or not self.producer_version or not self.code_hash:
            raise ValueError("measurement provenance has an empty required field")
        object.__setattr__(self, "notes", dict(self.notes))
        canonical_json(self.notes)


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    uri: str
    content_hash: str

    def __post_init__(self) -> None:
        if not all((self.artifact_id, self.kind, self.uri, self.content_hash)):
            raise ValueError("artifact reference has an empty required field")


@dataclass(frozen=True)
class MeasurementRequest:
    request_id: str
    kind: MeasurementKind
    state_hash: str
    version: str
    context_selector: ContextSelector
    candidate_ids: tuple[str, ...]
    expected_cost: CostEstimate
    expected_decision_value: float
    reuse_keys: tuple[str, ...]
    provenance: MeasurementProvenance
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id or not self.state_hash:
            raise ValueError("measurement request requires request_id and state_hash")
        if (
            not math.isfinite(self.expected_decision_value)
            or self.expected_decision_value < 0.0
        ):
            raise ValueError("expected_decision_value must be finite and nonnegative")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("measurement request candidate_ids must be unique")
        payload = dict(self.payload)
        object.__setattr__(self, "payload", payload)
        canonical_json(payload)
        expected_request_id = content_hash(
            {
                "kind": self.kind,
                "state_hash": self.state_hash,
                "version": self.version,
                "context_selector": self.context_selector,
                "candidate_ids": tuple(self.candidate_ids),
                "expected_cost": self.expected_cost,
                "reuse_keys": tuple(self.reuse_keys),
                "provenance": self.provenance,
                "payload": payload,
            },
            prefix="irg-request-",
        )
        if self.request_id != expected_request_id:
            raise ValueError(
                "measurement request ID differs from its predictable content"
            )

    @classmethod
    def create(
        cls,
        *,
        kind: MeasurementKind,
        state_hash: str,
        version: str,
        context_selector: ContextSelector,
        candidate_ids: Sequence[str] = (),
        expected_cost: CostEstimate = CostEstimate(),
        expected_decision_value: float = 0.0,
        reuse_keys: Sequence[str] = (),
        provenance: MeasurementProvenance,
        payload: Mapping[str, Any] | None = None,
    ) -> "MeasurementRequest":
        body = {
            "kind": kind,
            "state_hash": state_hash,
            "version": version,
            "context_selector": context_selector,
            "candidate_ids": tuple(candidate_ids),
            "expected_cost": expected_cost,
            "reuse_keys": tuple(reuse_keys),
            "provenance": provenance,
            "payload": dict(payload or {}),
        }
        return cls(
            request_id=content_hash(body, prefix="irg-request-"),
            expected_decision_value=float(expected_decision_value),
            **body,
        )

    @property
    def requester_identity(self) -> str:
        """Capability-neutral alias for the historical serialized field name."""

        return self.version

    @property
    def cache_key(self) -> str:
        return content_hash(
            {
                "kind": self.kind,
                "state_hash": self.state_hash,
                "version": self.version,
                "selector": self.context_selector,
                "candidate_ids": self.candidate_ids,
                "reuse_keys": self.reuse_keys,
                "payload": self.payload,
            },
            prefix="irg-cache-",
        )


@dataclass(frozen=True)
class MeasurementOutcome:
    request_id: str
    state_hash: str
    context_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    values: np.ndarray
    calibration_state_id: str
    realized_cost: RealizedCost
    cache_artifacts: tuple[ArtifactRef, ...]
    integrity_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._check_fields()

    def _check_fields(self) -> None:
        """Validate and normalize outcome fields without recomputing its hash."""

        if not self.request_id or not self.state_hash or not self.calibration_state_id:
            raise ValueError("measurement outcome has an empty required identity")
        values = np.asarray(self.values)
        metadata = dict(self.metadata)
        summary_only = bool(metadata.get("summary_only", False))
        if values.size == 0:
            if not summary_only:
                raise ValueError(
                    "measurement outcome requires values unless it carries "
                    "typed summary-only sufficient statistics"
                )
            if (
                metadata.get("sufficient_statistics_schema")
                != "stratified_running_moments_v1"
            ):
                raise ValueError(
                    "summary-only measurement outcome lacks its statistics schema"
                )
            candidates = metadata.get("oracle_stratified_moments")
            if not isinstance(candidates, Mapping) or set(candidates) != set(
                self.candidate_ids
            ):
                raise ValueError(
                    "summary-only moments must cover every candidate exactly"
                )
            for candidate_id, strata in candidates.items():
                if not isinstance(strata, Mapping) or not strata:
                    raise ValueError(
                        f"summary-only candidate {candidate_id!r} lacks strata"
                    )
                total = 0
                for moment in strata.values():
                    if not isinstance(moment, Mapping):
                        raise ValueError("summary-only moment must be a mapping")
                    count = int(moment.get("count", -1))
                    mean = float(moment.get("mean", math.nan))
                    m2 = float(moment.get("m2", math.nan))
                    if (
                        count <= 0
                        or not math.isfinite(mean)
                        or not math.isfinite(m2)
                        or m2 < 0.0
                    ):
                        raise ValueError(
                            "summary-only outcome contains an invalid moment"
                        )
                    total += count
                if total != len(self.context_ids):
                    raise ValueError(
                        "summary-only candidate moment counts do not match contexts"
                    )
        elif not np.isfinite(values).all():
            raise ValueError("measurement outcome requires finite values")
        if len(set(self.context_ids)) != len(self.context_ids):
            raise ValueError("measurement outcome context_ids must be unique")
        # Preserve read-only memmaps and immutable provider buffers. Mutable or
        # non-contiguous inputs receive exactly one canonical C-order copy.
        if values.flags.writeable or not values.flags.c_contiguous:
            values = np.array(values, copy=True, order="C")
            values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        state_hash: str,
        context_ids: Sequence[str],
        candidate_ids: Sequence[str],
        values: Any,
        calibration_state_id: str,
        realized_cost: RealizedCost,
        cache_artifacts: Sequence[ArtifactRef] = (),
        metadata: Mapping[str, Any] | None = None,
        audit_integrity: bool = True,
    ) -> "MeasurementOutcome":
        array = np.asarray(values)
        provisional = cls.__new__(cls)
        fields = {
            "request_id": str(request_id),
            "state_hash": str(state_hash),
            "context_ids": tuple(str(item) for item in context_ids),
            "candidate_ids": tuple(str(item) for item in candidate_ids),
            "values": array,
            "calibration_state_id": str(calibration_state_id),
            "realized_cost": realized_cost,
            "cache_artifacts": tuple(cache_artifacts),
            "metadata": dict(metadata or {}),
        }
        for key, value in fields.items():
            object.__setattr__(provisional, key, value)
        if not audit_integrity:
            # The per-look hash is pure audit.  Keep a typed sentinel so
            # existing readers can still carry the field without a content
            # digest.  Structural validation still runs.
            object.__setattr__(
                provisional,
                "integrity_hash",
                V6_UNHASHED_OUTCOME_PREFIX + str(request_id),
            )
            provisional._check_fields()
            return provisional
        object.__setattr__(
            provisional,
            "integrity_hash",
            provisional.compute_integrity_hash(),
        )
        provisional._check_fields()
        return provisional

    @classmethod
    def lightweight(
        cls,
        *,
        request_id: str,
        state_hash: str,
        context_ids: Sequence[str],
        candidate_ids: Sequence[str],
        values: Any,
        calibration_state_id: str,
        realized_cost: RealizedCost,
        metadata: Mapping[str, Any],
    ) -> "MeasurementOutcome":
        """An outcome record with no integrity hash.

        For the fixed-calibration adapter's paired looks only: their single
        consumer (``_assemble_pair_evidence``) reads the rows, the moment
        packet and the allocation receipt, and no outcome hash seeds anything
        (it would cover a wall-clock field).  The metadata
        may hold array objects and is NOT canonicalized.
        """

        if not request_id or not state_hash or not calibration_state_id:
            raise ValueError("measurement outcome has an empty required identity")
        array = np.asarray(values)
        if array.flags.writeable or not array.flags.c_contiguous:
            array = np.array(array, copy=True, order="C")
            array.setflags(write=False)
        record = cls.__new__(cls)
        for key, value in {
            "request_id": str(request_id),
            "state_hash": str(state_hash),
            "context_ids": tuple(context_ids),
            "candidate_ids": tuple(str(item) for item in candidate_ids),
            "values": array,
            "calibration_state_id": str(calibration_state_id),
            "realized_cost": realized_cost,
            "cache_artifacts": (),
            "integrity_hash": V6_UNHASHED_OUTCOME_PREFIX + str(request_id),
            "metadata": dict(metadata),
        }.items():
            object.__setattr__(record, key, value)
        return record


    def compute_integrity_hash(self) -> str:
        return content_hash(
            {
                "request_id": self.request_id,
                "state_hash": self.state_hash,
                "context_ids": self.context_ids,
                "candidate_ids": self.candidate_ids,
                "values": np.asarray(self.values),
                "calibration_state_id": self.calibration_state_id,
                "realized_cost": self.realized_cost,
                "cache_artifacts": self.cache_artifacts,
                "metadata": self.metadata,
            },
            prefix="irg-outcome-",
        )


class MeasurementProvider(Protocol):
    supported_kinds: frozenset[MeasurementKind]

    def execute(self, request: MeasurementRequest) -> MeasurementOutcome: ...


class OutcomeCache(Protocol):
    def get(self, cache_key: str) -> MeasurementOutcome | None: ...

    def put(self, cache_key: str, outcome: MeasurementOutcome) -> None: ...


def replace_request_value(
    request: MeasurementRequest,
    value: float,
) -> MeasurementRequest:
    return replace(request, expected_decision_value=float(value))


__all__ = [
    "ArtifactRef",
    "ContextSelector",
    "MeasurementKind",
    "MeasurementOutcome",
    "MeasurementProvenance",
    "MeasurementProvider",
    "MeasurementRequest",
    "OutcomeCache",
    "replace_request_value",
]
