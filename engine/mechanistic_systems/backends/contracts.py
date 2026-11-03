"""Backend protocols, autotuning policy, and promotion gates.

Dispatch keys are capability, route family, device, and dtype.  Architecture
names are intentionally absent: architecture adapters lower stable route
coordinates into one of the shared executable route families.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GateGraphCompiler(Protocol):
    compiler_id: str

    def compile(self, model: Any, route_bindings: Sequence[Any], *, device: Any) -> Any:
        """Return a stateless graph with explicit route and residual gates."""


@runtime_checkable
class RouteReducerBackend(Protocol):
    backend_id: str
    supported_route_families: frozenset[str]

    def reduce_vjp(
        self,
        *,
        route_family: str,
        activation: Any,
        output_gradient: Any,
        weight: Any,
        metadata: Mapping[str, Any],
    ) -> Any:
        """Reduce one exact gate derivative without architecture dispatch."""


@runtime_checkable
class CalibrationBackend(Protocol):
    backend_id: str
    supported_families: frozenset[str]

    def fit(self, family: str, logits: Any, targets: Any, **options: Any) -> Any:
        ...

    def apply(self, fit: Any, logits: Any) -> Any:
        ...


@runtime_checkable
class CandidateExecutionBackend(Protocol):
    backend_id: str

    def evaluate(self, compiled_graph: Any, inputs: Any, candidate_gates: Any) -> Any:
        """Evaluate a candidate tile with no mutation of the source model."""


@runtime_checkable
class ContractionBackend(Protocol):
    backend_id: str

    def gram(self, left: Any, right: Any | None = None, **options: Any) -> Any:
        ...


@dataclass(frozen=True)
class BackendProfile:
    """Operational telemetry that never participates in scientific identity."""

    backend_id: str
    capability: str
    implementation: str
    device_type: str
    numerical_mode: str
    calls: int = 0
    wall_time_s: float = 0.0
    compile_time_s: float = 0.0
    loader_time_s: float = 0.0
    transfer_time_s: float = 0.0
    peak_hbm_bytes: int | None = None
    processed_contexts: int = 0
    processed_queries: int = 0
    fallback_reason: str = ""
    phase_time_s: Mapping[str, float] = field(default_factory=dict)
    cuda_phase_time_s: Mapping[str, float] = field(default_factory=dict)
    cuda_phase_calls: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.backend_id or not self.capability or not self.implementation:
            raise ValueError("backend profile identity must be nonempty")
        values = (
            self.wall_time_s,
            self.compile_time_s,
            self.loader_time_s,
            self.transfer_time_s,
            *self.phase_time_s.values(),
            *self.cuda_phase_time_s.values(),
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError("backend profile times must be finite and nonnegative")
        if self.calls < 0 or self.processed_contexts < 0 or self.processed_queries < 0:
            raise ValueError("backend profile counts must be nonnegative")
        if self.peak_hbm_bytes is not None and int(self.peak_hbm_bytes) < 0:
            raise ValueError("peak HBM must be nonnegative")

    def to_runtime_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "backend_id": self.backend_id,
            "capability": self.capability,
            "implementation": self.implementation,
            "device_type": self.device_type,
            "numerical_mode": self.numerical_mode,
            "calls": int(self.calls),
            "wall_time_s": float(self.wall_time_s),
            "compile_time_s": float(self.compile_time_s),
            "loader_time_s": float(self.loader_time_s),
            "transfer_time_s": float(self.transfer_time_s),
            "peak_hbm_bytes": self.peak_hbm_bytes,
            "processed_contexts": int(self.processed_contexts),
            "processed_queries": int(self.processed_queries),
            "fallback_reason": self.fallback_reason,
            "phase_time_s": {str(key): float(value) for key, value in self.phase_time_s.items()},
            "cuda_phase_time_s": {
                str(key): float(value) for key, value in self.cuda_phase_time_s.items()
            },
            "cuda_phase_calls": {
                str(key): int(value) for key, value in self.cuda_phase_calls.items()
            },
            "metadata": dict(self.metadata),
            "scientific_identity": "excluded",
        }


@dataclass(frozen=True, order=True)
class RuntimeTile:
    image_batch: int
    query_tile: int
    candidate_tile: int = 1
    route_block: int = 1

    def validate(self) -> None:
        if min(self.image_batch, self.query_tile, self.candidate_tile, self.route_block) <= 0:
            raise ValueError("runtime tile dimensions must be positive")

    @property
    def work(self) -> int:
        return self.image_batch * self.query_tile * self.candidate_tile * self.route_block


@dataclass(frozen=True)
class RuntimeTuningSpace:
    candidates: tuple[RuntimeTile, ...]
    hbm_fraction_limit: float = 0.70

    def validate(self) -> None:
        if not self.candidates:
            raise ValueError("runtime tuning space requires candidates")
        if not 0.0 < float(self.hbm_fraction_limit) <= 1.0:
            raise ValueError("HBM fraction limit must lie in (0, 1]")
        for candidate in self.candidates:
            candidate.validate()


def select_feasible_runtime_tile(
    space: RuntimeTuningSpace,
    *,
    total_hbm_bytes: int,
    preflight: Callable[[RuntimeTile], tuple[int, float]],
) -> tuple[RuntimeTile, tuple[dict[str, Any], ...]]:
    """Choose the fastest feasible tile without changing scientific identity.

    ``preflight`` returns ``(peak_hbm_bytes, wall_time_s)``.  All candidates
    are evaluated in stable order, making the tuning receipt reproducible.
    """

    space.validate()
    if int(total_hbm_bytes) <= 0:
        raise ValueError("total_hbm_bytes must be positive")
    limit = int(float(space.hbm_fraction_limit) * int(total_hbm_bytes))
    rows: list[dict[str, Any]] = []
    feasible: list[tuple[float, int, RuntimeTile]] = []
    for tile in sorted(space.candidates):
        peak, elapsed = preflight(tile)
        valid = int(peak) <= limit and math.isfinite(float(elapsed)) and float(elapsed) > 0.0
        rows.append(
            {
                "image_batch": tile.image_batch,
                "query_tile": tile.query_tile,
                "candidate_tile": tile.candidate_tile,
                "route_block": tile.route_block,
                "peak_hbm_bytes": int(peak),
                "wall_time_s": float(elapsed),
                "feasible": bool(valid),
            }
        )
        if valid:
            feasible.append((float(elapsed), -tile.work, tile))
    if not feasible:
        raise RuntimeError("no runtime tile satisfied the HBM and numerical preflight")
    return min(feasible)[2], tuple(rows)


@dataclass(frozen=True)
class BackendPromotionDecision:
    promoted: bool
    numerical_parity: bool
    kernel_speedup: float
    affected_cell_speedup: float
    reasons: tuple[str, ...]


def assess_backend_promotion(
    *,
    scores_numerically_equal: bool,
    reference_kernel_time_s: float,
    candidate_kernel_time_s: float,
    reference_cell_time_s: float,
    candidate_cell_time_s: float,
    pruning_decisions_preserved: bool,
) -> BackendPromotionDecision:
    """Apply the frozen numerical and performance promotion gate."""

    times = (
        reference_kernel_time_s,
        candidate_kernel_time_s,
        reference_cell_time_s,
        candidate_cell_time_s,
    )
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in times):
        raise ValueError("promotion timing inputs must be finite and positive")
    numerical = bool(scores_numerically_equal) and bool(pruning_decisions_preserved)
    kernel_speedup = float(reference_kernel_time_s) / float(candidate_kernel_time_s)
    cell_speedup = float(reference_cell_time_s) / float(candidate_cell_time_s)
    performance = kernel_speedup >= 1.5 or cell_speedup >= 1.25
    reasons: list[str] = []
    if not numerical:
        reasons.append("numerical_or_pruning_parity_failed")
    if not performance:
        reasons.append("performance_threshold_not_met")
    return BackendPromotionDecision(
        promoted=numerical and performance,
        numerical_parity=numerical,
        kernel_speedup=kernel_speedup,
        affected_cell_speedup=cell_speedup,
        reasons=tuple(reasons),
    )


class BackendRegistry:
    """Dispatch equivalent implementations without architecture-name keys."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str, str], Any] = {}

    def register(
        self,
        backend: Any,
        *,
        capability: str,
        route_family: str = "all",
        device_type: str = "all",
        dtype: str = "all",
    ) -> None:
        key = tuple(str(value) for value in (capability, route_family, device_type, dtype))
        if any(not value for value in key):
            raise ValueError("backend dispatch coordinates must be nonempty")
        if key in self._entries:
            raise ValueError(f"duplicate backend dispatch key {key!r}")
        self._entries[key] = backend

    def resolve(
        self,
        *,
        capability: str,
        route_family: str,
        device_type: str,
        dtype: str,
    ) -> Any:
        candidates = (
            (capability, route_family, device_type, dtype),
            (capability, route_family, device_type, "all"),
            (capability, route_family, "all", dtype),
            (capability, route_family, "all", "all"),
            (capability, "all", device_type, dtype),
            (capability, "all", "all", "all"),
        )
        for key in candidates:
            if key in self._entries:
                return self._entries[key]
        raise KeyError(
            f"no backend for capability={capability!r}, route_family={route_family!r}, "
            f"device={device_type!r}, dtype={dtype!r}"
        )


__all__ = [
    "BackendProfile",
    "BackendPromotionDecision",
    "BackendRegistry",
    "CalibrationBackend",
    "CandidateExecutionBackend",
    "ContractionBackend",
    "GateGraphCompiler",
    "RouteReducerBackend",
    "RuntimeTile",
    "RuntimeTuningSpace",
    "assess_backend_promotion",
    "select_feasible_runtime_tile",
]
