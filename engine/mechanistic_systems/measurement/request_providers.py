"""MeasurementRequest providers for the IRG orchestration layer.

This module is intentionally adapter-first.  Existing Torch/calibration
runtimes can be exposed with ``CallbackMeasurementProvider`` without moving
validated numerical code.  The registry enforces scientific state, candidate,
context, and integrity identities before any outcome reaches the decision
engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from mechanistic_systems.contracts.measurement import (
    ArtifactRef,
    MeasurementKind,
    MeasurementOutcome,
    MeasurementProvider,
    MeasurementRequest,
    OutcomeCache,
)
from mechanistic_systems.contracts.measurement_costs import RealizedCost


class MemoryOutcomeCache:
    def __init__(self) -> None:
        self._items: dict[str, MeasurementOutcome] = {}

    def get(self, cache_key: str) -> MeasurementOutcome | None:
        return self._items.get(str(cache_key))

    def put(self, cache_key: str, outcome: MeasurementOutcome) -> None:
        self._items[str(cache_key)] = outcome

    def __len__(self) -> int:
        return len(self._items)


class NpzOutcomeCache:
    """Safe filesystem cache using JSON metadata plus NumPy NPZ arrays."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, cache_key: str) -> tuple[Path, Path]:
        label = str(cache_key).replace("/", "_")
        return self.root / f"{label}.json", self.root / f"{label}.npz"

    def get(self, cache_key: str) -> MeasurementOutcome | None:
        metadata_path, array_path = self._paths(cache_key)
        if not metadata_path.exists() or not array_path.exists():
            return None
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(array_path, allow_pickle=False) as archive:
            values = archive["values"]
        realized = RealizedCost(**payload["realized_cost"])
        artifacts = tuple(ArtifactRef(**item) for item in payload["cache_artifacts"])
        outcome = MeasurementOutcome.create(
            request_id=payload["request_id"],
            state_hash=payload["state_hash"],
            context_ids=payload["context_ids"],
            candidate_ids=payload["candidate_ids"],
            values=values,
            calibration_state_id=payload["calibration_state_id"],
            realized_cost=realized,
            cache_artifacts=artifacts,
            metadata=payload["metadata"],
        )
        if outcome.integrity_hash != payload["integrity_hash"]:
            raise ValueError("cached measurement outcome failed its integrity check")
        return outcome

    def put(self, cache_key: str, outcome: MeasurementOutcome) -> None:
        metadata_path, array_path = self._paths(cache_key)
        np.savez_compressed(array_path, values=np.asarray(outcome.values))
        payload = {
            "request_id": outcome.request_id,
            "state_hash": outcome.state_hash,
            "context_ids": list(outcome.context_ids),
            "candidate_ids": list(outcome.candidate_ids),
            "calibration_state_id": outcome.calibration_state_id,
            "realized_cost": outcome.realized_cost.to_dict(),
            "cache_artifacts": [item.__dict__ for item in outcome.cache_artifacts],
            "metadata": dict(outcome.metadata),
            "integrity_hash": outcome.integrity_hash,
        }
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class NpyMemmapOutcomeCache:
    """Filesystem cache for large factors using read-only memory-mapped NPY arrays.

    Unlike compressed NPZ, this format can be reloaded without allocating a
    second full context-by-route matrix. JSON retains all scientific identity
    and integrity fields.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, cache_key: str) -> tuple[Path, Path]:
        label = str(cache_key).replace("/", "_")
        return self.root / f"{label}.json", self.root / f"{label}.npy"

    def get(self, cache_key: str) -> MeasurementOutcome | None:
        metadata_path, array_path = self._paths(cache_key)
        if not metadata_path.exists() or not array_path.exists():
            return None
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = np.load(array_path, mmap_mode="r", allow_pickle=False)
        realized = RealizedCost(**payload["realized_cost"])
        artifacts = tuple(ArtifactRef(**item) for item in payload["cache_artifacts"])
        outcome = MeasurementOutcome.create(
            request_id=payload["request_id"],
            state_hash=payload["state_hash"],
            context_ids=payload["context_ids"],
            candidate_ids=payload["candidate_ids"],
            values=values,
            calibration_state_id=payload["calibration_state_id"],
            realized_cost=realized,
            cache_artifacts=artifacts,
            metadata=payload["metadata"],
        )
        if outcome.integrity_hash != payload["integrity_hash"]:
            raise ValueError("memory-mapped cached outcome failed its integrity check")
        return outcome

    def put(self, cache_key: str, outcome: MeasurementOutcome) -> None:
        metadata_path, array_path = self._paths(cache_key)
        temporary_array = array_path.with_suffix(".npy.tmp")
        temporary_metadata = metadata_path.with_suffix(".json.tmp")
        with temporary_array.open("wb") as stream:
            np.save(stream, np.asarray(outcome.values), allow_pickle=False)
        payload = {
            "request_id": outcome.request_id,
            "state_hash": outcome.state_hash,
            "context_ids": list(outcome.context_ids),
            "candidate_ids": list(outcome.candidate_ids),
            "calibration_state_id": outcome.calibration_state_id,
            "realized_cost": outcome.realized_cost.to_dict(),
            "cache_artifacts": [item.__dict__ for item in outcome.cache_artifacts],
            "metadata": dict(outcome.metadata),
            "integrity_hash": outcome.integrity_hash,
        }
        temporary_metadata.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary_array.replace(array_path)
        temporary_metadata.replace(metadata_path)


def validate_outcome_for_request(
    request: MeasurementRequest,
    outcome: MeasurementOutcome,
) -> None:
    if outcome.request_id != request.request_id:
        raise ValueError("measurement provider returned an outcome for a different request")
    if outcome.state_hash != request.state_hash:
        raise ValueError("measurement provider changed the scientific pruning state")
    if tuple(outcome.candidate_ids) != tuple(request.candidate_ids):
        raise ValueError("measurement provider changed candidate identity or ordering")
    if request.context_selector.context_ids:
        if tuple(outcome.context_ids) != tuple(request.context_selector.context_ids):
            raise ValueError("measurement provider changed an explicit context selection")
    if not outcome.context_ids:
        raise ValueError("measurement provider must retain per-context identities")
    if len(set(outcome.context_ids)) != len(outcome.context_ids):
        raise ValueError("measurement provider returned duplicate context identities")
    if str(outcome.metadata.get("kind", "")) != request.kind:
        raise ValueError("measurement outcome kind does not match the request")
    if request.kind == "hard_mask_oracle":
        values = np.asarray(outcome.values)
        expected_candidates = max(1, len(request.candidate_ids))
        summary_only = bool(outcome.metadata.get("summary_only", False))
        if summary_only:
            moments = outcome.metadata.get("oracle_stratified_moments")
            if (
                values.shape != (0, expected_candidates)
                or not isinstance(moments, Mapping)
                or set(map(str, moments)) != set(map(str, request.candidate_ids))
                or outcome.metadata.get("sufficient_statistics_schema")
                != "stratified_running_moments_v1"
            ):
                raise ValueError(
                    "summary-only oracle outcome requires an empty row array "
                    "and complete candidate-indexed sufficient statistics"
                )
        elif values.ndim == 1 and expected_candidates != 1:
            raise ValueError("multi-candidate oracle outcome requires a candidate axis")
        elif values.ndim == 2 and values.shape != (
            len(outcome.context_ids),
            expected_candidates,
        ):
            raise ValueError("oracle values must have shape [contexts, candidates]")
        elif values.ndim not in {1, 2}:
            raise ValueError("oracle values must be one- or two-dimensional")
    elif request.kind == "marginal_statistics":
        values = np.asarray(outcome.values)
        expected_rows = tuple(outcome.metadata.get("statistic_rows", ()))
        if values.ndim != 2 or values.shape[0] != 4:
            raise ValueError("marginal statistics must have shape [4, routes]")
        if expected_rows != ("mean", "diagonal_energy", "mean_standard_error", "energy_standard_error"):
            raise ValueError("marginal statistics require the canonical statistic row order")
        if int(outcome.metadata.get("sample_count", 0)) != len(outcome.context_ids):
            raise ValueError("marginal statistics sample_count must match retained context identities")
    elif request.kind == "response_moments":
        values = np.asarray(outcome.values)
        typed_in_process_summary = bool(
            outcome.metadata.get("in_process_typed_response_summary", False)
        )
        if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
            raise ValueError("response moments must be a nonempty packed [statistics, routes] array")
        if typed_in_process_summary:
            if (
                tuple(values.shape) != (1, 1)
                or not outcome.metadata.get("in_process_summary_token")
                or not outcome.metadata.get(
                    "in_process_summary_route_axis_hash"
                )
                or not isinstance(
                    outcome.metadata.get(
                        "in_process_summary_component_hashes"
                    ),
                    Mapping,
                )
                or outcome.metadata.get("compact_outcome_is_lineage_only")
                is not True
            ):
                raise ValueError(
                    "typed response moments require a complete compact "
                    "in-process lineage contract"
                )
        elif not (
            outcome.metadata.get("moment_layout")
            or outcome.metadata.get("joint_moment_layout")
        ):
            raise ValueError("response moments require a declared packed moment layout")
        if int(outcome.metadata.get("sample_count", 0)) != len(outcome.context_ids):
            raise ValueError("response moments sample_count must match retained context identities")
    elif request.kind in {
        "response_factor",
        "response_jacobian",
        "covariance_rank_expansion",
    }:
        values = np.asarray(outcome.values)
        if values.ndim != 3 or values.shape[0] != len(outcome.context_ids):
            raise ValueError("response factors must retain shape [contexts,response,coordinates]")
        if int(outcome.metadata.get("response_dimension", 0)) <= 0:
            raise ValueError("response factors require a positive response_dimension")
    elif request.kind in {"jacobian_rows", "metric_factor"}:
        values = np.asarray(outcome.values)
        if values.ndim not in {2, 3} or values.shape[0] != len(outcome.context_ids):
            raise ValueError(
                "legacy or response factors must retain a leading context axis with two or three dimensions"
            )
    elif request.kind == "response_transition":
        values = np.asarray(outcome.values)
        response_dimension = int(outcome.metadata.get("response_dimension", 0))
        if response_dimension <= 0:
            raise ValueError("response transition requires a positive response_dimension")
        if values.ndim != 2 or values.shape != (len(outcome.context_ids), 2 * response_dimension):
            raise ValueError(
                "response transition values must concatenate predicted and exact [contexts,response] rows"
            )
        if len(outcome.candidate_ids) != 1:
            raise ValueError("response transition requires exactly one candidate")
    elif request.kind == "directional_curvature":
        values = np.asarray(outcome.values)
        if values.ndim not in {1, 2, 3}:
            raise ValueError(
                "directional curvature must be [route], [response,route], "
                "or [probe,response,route]"
            )
        if values.ndim == 3:
            probe_ids = tuple(map(str, outcome.metadata.get("probe_ids", ())))
            route_ids = tuple(map(str, outcome.metadata.get("route_ids", ())))
            response_dimension = int(outcome.metadata.get("response_dimension", 0))
            statistic_rows = tuple(
                map(str, outcome.metadata.get("statistic_rows", ()))
            )
            packed_response_dimension = int(
                outcome.metadata.get(
                    "packed_response_dimension",
                    response_dimension,
                )
            )
            if (
                values.shape[0] != len(probe_ids)
                or values.shape[1] != packed_response_dimension
                or values.shape[2] != len(route_ids)
                or not probe_ids
                or not route_ids
                or response_dimension <= 0
                or (
                    statistic_rows
                    and (
                        statistic_rows != ("mean", "estimator_variance")
                        or packed_response_dimension != 2 * response_dimension
                    )
                )
            ):
                raise ValueError(
                    "control phenotype requires declared probe, response, "
                    "and physical-route axes"
                )
    elif request.kind == "finite_dose_response":
        values = np.asarray(outcome.values)
        layout = dict(outcome.metadata.get("packed_layout", {}))
        statistic_count = int(layout.get("statistic_count", 0))
        dose_count = int(layout.get("dose_count", 0))
        response_dimension = int(layout.get("response_dimension", 0))
        route_count = int(layout.get("route_count", 0))
        if (
            values.ndim != 2
            or values.shape
            != (
                statistic_count * dose_count * response_dimension,
                route_count,
            )
            or statistic_count != 4
            or dose_count <= 0
            or response_dimension <= 0
            or route_count <= 0
        ):
            raise ValueError(
                "finite-dose response has an incompatible packed statistic layout"
            )
    else:
        values = np.asarray(outcome.values)
        if values.ndim == 0 or values.shape[0] != len(outcome.context_ids):
            raise ValueError("measurement values must retain the leading context axis")


class ProviderRegistry:
    """Kind-indexed provider registry with content-addressed outcome reuse."""

    def __init__(self, *, cache: OutcomeCache | None = None) -> None:
        self.cache = cache if cache is not None else MemoryOutcomeCache()
        self._providers: dict[str, MeasurementProvider] = {}

    def register(self, provider: MeasurementProvider, *, replace_existing: bool = False) -> None:
        for kind in provider.supported_kinds:
            if kind in self._providers and not replace_existing:
                raise ValueError(f"a provider is already registered for {kind!r}")
            self._providers[kind] = provider

    def supports(self, kind: MeasurementKind) -> bool:
        """Return whether a provider is installed for the exact measurement kind."""
        return str(kind) in self._providers

    def provider(self, kind: MeasurementKind) -> MeasurementProvider:
        try:
            return self._providers[str(kind)]
        except KeyError as exc:
            raise KeyError(f"no measurement provider is registered for {kind!r}") from exc

    def execute(self, request: MeasurementRequest, *, allow_cache: bool = True) -> MeasurementOutcome:
        if allow_cache:
            cached = self.cache.get(request.cache_key)
            if cached is not None:
                validate_outcome_for_request(request, cached)
                cached_cost = RealizedCost(
                    packing_degree=cached.realized_cost.packing_degree,
                    cache_hit=True,
                )
                return MeasurementOutcome.create(
                    request_id=cached.request_id,
                    state_hash=cached.state_hash,
                    context_ids=cached.context_ids,
                    candidate_ids=cached.candidate_ids,
                    values=cached.values,
                    calibration_state_id=cached.calibration_state_id,
                    realized_cost=cached_cost,
                    cache_artifacts=cached.cache_artifacts,
                    metadata={**dict(cached.metadata), "registry_cache_hit": True},
                )
        provider = self.provider(request.kind)
        outcome = provider.execute(request)
        validate_outcome_for_request(request, outcome)
        self.cache.put(request.cache_key, outcome)
        return outcome


@dataclass
class CallbackMeasurementProvider:
    """Thin adapter around an existing validated runtime callback."""

    supported_kinds: frozenset[MeasurementKind]
    callback: Callable[[MeasurementRequest], MeasurementOutcome]

    def __init__(
        self,
        supported_kinds: Sequence[MeasurementKind],
        callback: Callable[[MeasurementRequest], MeasurementOutcome],
    ) -> None:
        kinds = frozenset(supported_kinds)
        if not kinds:
            raise ValueError("callback provider requires at least one measurement kind")
        self.supported_kinds = kinds
        self.callback = callback

    def execute(self, request: MeasurementRequest) -> MeasurementOutcome:
        if request.kind not in self.supported_kinds:
            raise ValueError(f"callback provider does not support {request.kind!r}")
        return self.callback(request)


class ArrayJacobianProvider:
    """Deterministic offline provider for retained context-by-route factors.

    This provider is useful for unit tests, replaying stored factors, and
    integrating preserved score stores without model execution.  Production
    model execution should be supplied through ``CallbackMeasurementProvider``.
    """

    supported_kinds = frozenset({"jacobian_rows", "marginal_statistics", "metric_factor"})

    def __init__(
        self,
        rows: Sequence[Sequence[float]] | np.ndarray,
        *,
        context_ids: Sequence[str] | None = None,
        calibration_state_id: str = "fixed-calibration",
        realized_cost: RealizedCost | None = None,
        nuisance_rows: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> None:
        route = np.asarray(rows, dtype=np.float64)
        if route.ndim != 2 or route.shape[0] == 0 or route.shape[1] == 0 or not np.isfinite(route).all():
            raise ValueError("ArrayJacobianProvider rows must be a finite nonempty matrix")
        self.route_rows = route
        self.nuisance_rows = None
        if nuisance_rows is not None:
            nuisance = np.asarray(nuisance_rows, dtype=np.float64)
            if nuisance.ndim != 2 or nuisance.shape[0] != route.shape[0] or not np.isfinite(nuisance).all():
                raise ValueError("nuisance rows must share the route-row context axis")
            self.nuisance_rows = nuisance
        self.context_ids = tuple(
            str(item) for item in (
                context_ids if context_ids is not None else (f"context-{index:08d}" for index in range(route.shape[0]))
            )
        )
        if len(self.context_ids) != route.shape[0] or len(set(self.context_ids)) != len(self.context_ids):
            raise ValueError("context_ids must uniquely align with rows")
        self.calibration_state_id = str(calibration_state_id)
        self.realized_cost = realized_cost or RealizedCost(backward_passes=1)

    def _indices(self, request: MeasurementRequest) -> np.ndarray:
        selector = request.context_selector
        if selector.context_ids:
            lookup = {context_id: index for index, context_id in enumerate(self.context_ids)}
            try:
                return np.asarray([lookup[item] for item in selector.context_ids], dtype=np.int64)
            except KeyError as exc:
                raise ValueError("requested context is absent from ArrayJacobianProvider") from exc
        excluded = set(str(item) for item in selector.parameters.get("exclude_context_ids", ()))
        available = np.asarray(
            [index for index, context_id in enumerate(self.context_ids) if context_id not in excluded],
            dtype=np.int64,
        )
        count = min(int(selector.requested_count or len(available)), len(available))
        if count <= 0:
            raise ValueError("no unexcluded contexts remain for the measurement")
        generator = np.random.default_rng(int(selector.random_seed))
        chosen = generator.choice(available, size=count, replace=False)
        return np.sort(chosen)

    def execute(self, request: MeasurementRequest) -> MeasurementOutcome:
        if request.kind not in self.supported_kinds:
            raise ValueError("ArrayJacobianProvider received an unsupported request")
        indices = self._indices(request)
        route_values = self.route_rows[indices]
        values = route_values
        metadata: dict[str, object] = {
            "kind": request.kind,
            "mask_hash": request.payload.get("mask_hash", "unknown-mask"),
            "route_dimension": self.route_rows.shape[1],
            "source": "retained_array_factor",
        }
        if request.kind == "marginal_statistics":
            n = route_values.shape[0]
            energy_rows = route_values * route_values
            mean = np.mean(route_values, axis=0)
            energy = np.mean(energy_rows, axis=0)
            mean_se = (
                np.std(route_values, axis=0, ddof=1) / np.sqrt(n)
                if n >= 2
                else np.zeros(route_values.shape[1], dtype=np.float64)
            )
            energy_se = (
                np.std(energy_rows, axis=0, ddof=1) / np.sqrt(n)
                if n >= 2
                else np.zeros(route_values.shape[1], dtype=np.float64)
            )
            values = np.vstack((mean, energy, mean_se, energy_se))
            metadata["statistic_rows"] = (
                "mean",
                "diagonal_energy",
                "mean_standard_error",
                "energy_standard_error",
            )
            metadata["sample_count"] = n
            metadata["streamable_sufficient_statistics"] = True
        else:
            retain = tuple(request.payload.get("retain", ())) if not isinstance(request.payload.get("retain"), str) else ()
            if self.nuisance_rows is not None and "nuisance_factor" in retain:
                values = np.concatenate([values, self.nuisance_rows[indices]], axis=1)
                metadata["nuisance_dimension"] = self.nuisance_rows.shape[1]
        return MeasurementOutcome.create(
            request_id=request.request_id,
            state_hash=request.state_hash,
            context_ids=tuple(self.context_ids[index] for index in indices),
            candidate_ids=request.candidate_ids,
            values=values,
            calibration_state_id=self.calibration_state_id,
            realized_cost=self.realized_cost,
            metadata=metadata,
        )


MaskOracleValues = Sequence[float] | np.ndarray
LegacyMaskOracleEvaluator = Callable[
    [str, tuple[str, ...]],
    MaskOracleValues,
]
ParentAwareMaskOracleEvaluator = Callable[
    [str, str, tuple[str, ...]],
    MaskOracleValues,
]


class PairedMaskOracleProvider:
    """Per-context exact hard-mask oracle adapter with common random numbers.

    ``evaluator`` preserves the legacy ``(candidate, contexts)`` contract.
    Dynamic pruning runtimes should instead supply ``parent_aware_evaluator``
    with the explicit ``(parent, candidate, contexts)`` contract.  Keeping the
    two call paths distinct makes parent propagation reviewable and prevents a
    callback-signature guess from silently changing scientific state.
    """

    supported_kinds = frozenset({"hard_mask_oracle"})

    def __init__(
        self,
        evaluator: LegacyMaskOracleEvaluator | None = None,
        *,
        parent_aware_evaluator: ParentAwareMaskOracleEvaluator | None = None,
        context_ids: Sequence[str],
        calibration_state_id: str = "fixed-calibration",
        realized_cost: RealizedCost | None = None,
    ) -> None:
        if (evaluator is None) == (parent_aware_evaluator is None):
            raise ValueError(
                "paired mask oracle requires exactly one of evaluator or "
                "parent_aware_evaluator"
            )
        self.evaluator = evaluator
        self.parent_aware_evaluator = parent_aware_evaluator
        self.context_ids = tuple(str(item) for item in context_ids)
        self._context_id_set = frozenset(self.context_ids)
        if not self.context_ids or len(self._context_id_set) != len(self.context_ids):
            raise ValueError("oracle context_ids must be finite, nonempty, and unique")
        self.calibration_state_id = str(calibration_state_id)
        self.realized_cost = realized_cost or RealizedCost(forward_passes=1)

    def _selected_contexts(self, request: MeasurementRequest) -> tuple[str, ...]:
        selector = request.context_selector
        if selector.context_ids:
            selected = tuple(selector.context_ids)
            if any(item not in self._context_id_set for item in selected):
                raise ValueError("oracle request includes an unknown explicit context")
            return selected
        excluded = set(str(item) for item in selector.parameters.get("exclude_context_ids", ()))
        available = np.asarray(
            [index for index, context_id in enumerate(self.context_ids) if context_id not in excluded],
            dtype=np.int64,
        )
        count = min(int(selector.requested_count or len(available)), len(available))
        if count <= 0:
            raise ValueError("no unexcluded oracle contexts remain")
        generator = np.random.default_rng(int(selector.random_seed))
        chosen = np.sort(generator.choice(available, size=count, replace=False))
        return tuple(self.context_ids[int(index)] for index in chosen)

    @staticmethod
    def _bound_parent_mask_hash(request: MeasurementRequest) -> str:
        parent = request.payload.get("parent_mask_hash")
        if not isinstance(parent, str) or not parent:
            raise ValueError(
                "parent-bound mask oracle requires a nonempty parent_mask_hash"
            )
        if parent != request.state_hash:
            raise ValueError(
                "certificate mask-oracle parent_mask_hash differs from "
                "request state_hash"
            )
        return parent

    def execute(self, request: MeasurementRequest) -> MeasurementOutcome:
        if request.kind != "hard_mask_oracle" or not request.candidate_ids:
            raise ValueError("paired mask oracle requires at least one candidate")
        certificate_sampling = (
            request.payload.get("draw_reuse_status")
            == "fresh_candidate_certificate"
        )
        parent_mask_hash: str | None = None
        if certificate_sampling or self.parent_aware_evaluator is not None:
            parent_mask_hash = self._bound_parent_mask_hash(request)
        if certificate_sampling:
            raw_draws = request.payload.get("draw_ids")
            raw_sources = request.payload.get("source_context_ids")
            if (
                isinstance(raw_draws, (str, bytes))
                or not isinstance(raw_draws, Sequence)
                or isinstance(raw_sources, (str, bytes))
                or not isinstance(raw_sources, Sequence)
            ):
                raise TypeError("certificate mask oracle requires draw/source sequences")
            outcome_contexts = tuple(str(value) for value in raw_draws)
            selected = tuple(str(value) for value in raw_sources)
            if (
                not outcome_contexts
                or len(set(outcome_contexts)) != len(outcome_contexts)
                or len(outcome_contexts) != len(selected)
                or any(value not in self._context_id_set for value in selected)
            ):
                raise ValueError("certificate mask-oracle draw/source lineage is invalid")
        else:
            selected = self._selected_contexts(request)
            outcome_contexts = selected
        columns = []
        for candidate_id in request.candidate_ids:
            if self.parent_aware_evaluator is not None:
                if parent_mask_hash is None:  # pragma: no cover - guarded above
                    raise RuntimeError("parent-aware oracle lost its bound parent")
                raw_values = self.parent_aware_evaluator(
                    parent_mask_hash,
                    candidate_id,
                    selected,
                )
            else:
                if self.evaluator is None:  # pragma: no cover - constructor invariant
                    raise RuntimeError("legacy oracle evaluator is unavailable")
                raw_values = self.evaluator(candidate_id, selected)
            values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
            if values.size != len(selected) or not np.isfinite(values).all():
                raise ValueError("oracle evaluator must return one finite paired delta per context")
            columns.append(values)
        matrix = np.column_stack(columns)
        output = matrix[:, 0] if matrix.shape[1] == 1 else matrix
        metadata: dict[str, object] = {
            "kind": "hard_mask_oracle",
            "paired_common_contexts": True,
            "packing_degree": max(1, len(request.candidate_ids)),
            "candidate_damage_rows_direct": True,
            "physical_source_context_ids": selected,
            "counter_addressed_draw_axis": certificate_sampling,
        }
        if parent_mask_hash is not None:
            metadata["parent_mask_hash"] = parent_mask_hash
            metadata["parent_mask_state_bound"] = True
        if self.parent_aware_evaluator is not None:
            metadata["parent_aware_evaluator"] = True
        return MeasurementOutcome.create(
            request_id=request.request_id,
            state_hash=request.state_hash,
            context_ids=outcome_contexts,
            candidate_ids=request.candidate_ids,
            values=output,
            calibration_state_id=self.calibration_state_id,
            realized_cost=replace(self.realized_cost, packing_degree=max(1, len(request.candidate_ids))),
            metadata=metadata,
        )


class PairedResponseTransitionProvider:
    """Paired predicted-versus-exact vector-response transition adapter.

    The provider evaluates the local response model and the exact hard-mask
    transition on identical contexts.  It transmits the frozen response-safe
    normalization contract unchanged so downstream validity is measured in
    scientific response units rather than scalar oracle-loss units.
    """

    supported_kinds = frozenset({"response_transition"})

    def __init__(
        self,
        predictor: Callable[[tuple[int, ...], tuple[float, ...], tuple[str, ...]], Sequence[Sequence[float]] | np.ndarray],
        exact_evaluator: Callable[[str, tuple[str, ...]], Sequence[Sequence[float]] | np.ndarray],
        *,
        context_ids: Sequence[str],
        response_dimension: int,
        normalization_metric: Sequence[Sequence[float]] | np.ndarray | None = None,
        calibration_state_id: str = "fixed-calibration",
        realized_cost: RealizedCost | None = None,
        risk_functional: (
            Callable[[np.ndarray, tuple[str, ...]], float] | None
        ) = None,
        risk_functional_hash: str | None = None,
    ) -> None:
        self.predictor = predictor
        self.exact_evaluator = exact_evaluator
        self.context_ids = tuple(str(item) for item in context_ids)
        self._context_id_set = frozenset(self.context_ids)
        if not self.context_ids or len(self._context_id_set) != len(self.context_ids):
            raise ValueError("response-transition context_ids must be finite, nonempty, and unique")
        self.response_dimension = int(response_dimension)
        if self.response_dimension <= 0:
            raise ValueError("response_dimension must be positive")
        metric = np.eye(self.response_dimension, dtype=np.float64) if normalization_metric is None else np.asarray(normalization_metric, dtype=np.float64)
        if metric.shape != (self.response_dimension, self.response_dimension) or not np.isfinite(metric).all():
            raise ValueError("normalization_metric must be finite and align with response_dimension")
        metric = 0.5 * (metric + metric.T)
        if float(np.min(np.linalg.eigvalsh(metric))) < -1e-8 * max(1.0, float(np.max(np.abs(metric)))):
            raise ValueError("normalization_metric must be positive semidefinite")
        self.normalization_metric = metric
        self.calibration_state_id = str(calibration_state_id)
        self.realized_cost = realized_cost or RealizedCost(forward_passes=2)
        if (risk_functional is None) != (risk_functional_hash is None):
            raise ValueError(
                "risk_functional and risk_functional_hash must be supplied together"
            )
        self.risk_functional = risk_functional
        self.risk_functional_hash = (
            None if risk_functional_hash is None else str(risk_functional_hash)
        )
        if self.risk_functional_hash == "":
            raise ValueError("risk_functional_hash must be nonempty")

    def _selected_contexts(self, request: MeasurementRequest) -> tuple[str, ...]:
        selector = request.context_selector
        if selector.context_ids:
            selected = tuple(selector.context_ids)
            if any(item not in self._context_id_set for item in selected):
                raise ValueError("response-transition request includes an unknown explicit context")
            return selected
        excluded = set(str(item) for item in selector.parameters.get("exclude_context_ids", ()))
        available = np.asarray(
            [index for index, context_id in enumerate(self.context_ids) if context_id not in excluded],
            dtype=np.int64,
        )
        count = min(int(selector.requested_count or len(available)), len(available))
        if count <= 0:
            raise ValueError("no unexcluded response-transition contexts remain")
        generator = np.random.default_rng(int(selector.random_seed))
        chosen = np.sort(generator.choice(available, size=count, replace=False))
        return tuple(self.context_ids[int(index)] for index in chosen)

    def execute(self, request: MeasurementRequest) -> MeasurementOutcome:
        if request.kind != "response_transition" or len(request.candidate_ids) != 1:
            raise ValueError("paired response transition requires exactly one candidate")
        certificate_draws = request.payload.get("draw_ids")
        certificate_sources = request.payload.get("source_context_ids")
        certificate_sampling = (
            request.payload.get("draw_reuse_status")
            == "fresh_candidate_certificate"
        )
        if certificate_sampling:
            if (
                isinstance(certificate_draws, (str, bytes))
                or not isinstance(certificate_draws, Sequence)
                or isinstance(certificate_sources, (str, bytes))
                or not isinstance(certificate_sources, Sequence)
            ):
                raise TypeError(
                    "certificate response transition requires draw/source sequences"
                )
            outcome_contexts = tuple(str(value) for value in certificate_draws)
            selected = tuple(str(value) for value in certificate_sources)
            if (
                not outcome_contexts
                or len(set(outcome_contexts)) != len(outcome_contexts)
                or len(outcome_contexts) != len(selected)
                or any(value not in self._context_id_set for value in selected)
            ):
                raise ValueError(
                    "certificate response transition draw/source lineage is invalid"
                )
        else:
            selected = self._selected_contexts(request)
            outcome_contexts = selected
        raw_action_indices = request.payload.get("action_indices", ())
        if isinstance(raw_action_indices, (str, bytes)) or not isinstance(
            raw_action_indices, Sequence
        ):
            raise TypeError("response-transition action_indices must be an integer sequence")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < 0
            for value in raw_action_indices
        ):
            raise TypeError(
                "response-transition action_indices must contain only nonnegative integers"
            )
        action_indices = tuple(int(value) for value in raw_action_indices)
        coefficients = tuple(float(value) for value in request.payload.get("coefficients", ()))
        if not action_indices or len(action_indices) != len(coefficients):
            raise ValueError("response-transition request requires aligned nonempty action indices and coefficients")
        predicted = np.asarray(self.predictor(action_indices, coefficients, selected), dtype=np.float64)
        exact = np.asarray(self.exact_evaluator(request.candidate_ids[0], selected), dtype=np.float64)
        expected_shape = (len(selected), self.response_dimension)
        if predicted.shape != expected_shape or exact.shape != expected_shape:
            raise ValueError("response-transition callbacks must return [contexts,response] arrays")
        if not np.isfinite(predicted).all() or not np.isfinite(exact).all():
            raise ValueError("response-transition callbacks returned nonfinite values")
        predicted_risk = None
        exact_risk = None
        if self.risk_functional is not None:
            predicted_risk = float(
                self.risk_functional(predicted, outcome_contexts)
            )
            exact_risk = float(self.risk_functional(exact, outcome_contexts))
            if (
                not np.isfinite(predicted_risk)
                or predicted_risk < 0.0
                or not np.isfinite(exact_risk)
                or exact_risk < 0.0
            ):
                raise ValueError(
                    "risk_functional must return one finite nonnegative risk"
                )
        metadata = {
            "kind": "response_transition",
            "response_dimension": self.response_dimension,
            "action_indices": action_indices,
            "coefficients": coefficients,
            "candidate_mask_hash": request.payload.get("candidate_mask_hash", request.candidate_ids[0]),
            "parent_mask_hash": request.payload.get("parent_mask_hash", ""),
            "normalization_metric": self.normalization_metric,
            "response_scale": float(request.payload.get("response_scale", 1.0)),
            "paired_common_contexts": True,
            "predicted_and_exact_rows_aligned": True,
            "risk_functional_hash": self.risk_functional_hash,
            "risk_values_derived_from_measured_rows": (
                self.risk_functional is not None
            ),
            "physical_source_context_ids": selected,
            "counter_addressed_draw_axis": certificate_sampling,
        }
        for key in (
            "referent_id",
            "context_mixture_hash",
            "risk_estimand_hash",
            "estimand_contrast_hash",
            "sampling_law_hash",
            "pre_outcome_lock_hash",
            "normalization_directions",
            "normalization_tolerances",
            "normalization_energy_metric",
            "normalization_energy_tolerance",
            "normalization_whitener",
            "context_weights",
            "response_constraint_hash",
            "active_competitor_changed",
        ):
            if key in request.payload and request.payload[key] is not None:
                metadata[key] = request.payload[key]
        if predicted_risk is not None and exact_risk is not None:
            metadata["predicted_risk"] = predicted_risk
            metadata["exact_risk"] = exact_risk
        return MeasurementOutcome.create(
            request_id=request.request_id,
            state_hash=request.state_hash,
            context_ids=outcome_contexts,
            candidate_ids=request.candidate_ids,
            values=np.concatenate((predicted, exact), axis=1),
            calibration_state_id=self.calibration_state_id,
            realized_cost=self.realized_cost,
            metadata=metadata,
        )


__all__ = [
    "ArrayJacobianProvider",
    "CallbackMeasurementProvider",
    "MeasurementProvider",
    "MemoryOutcomeCache",
    "NpzOutcomeCache",
    "NpyMemmapOutcomeCache",
    "OutcomeCache",
    "PairedMaskOracleProvider",
    "PairedResponseTransitionProvider",
    "ProviderRegistry",
    "validate_outcome_for_request",
]
