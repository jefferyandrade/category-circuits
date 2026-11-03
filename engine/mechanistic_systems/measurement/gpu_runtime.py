"""Hardware-aware execution planning for IRG measurement providers.

The module is deliberately independent of any particular ResNet evaluator.  It
learns safe, fast batch/packing configurations from realized executions and can
be used by any Torch-backed provider.  CUDA is optional; CPU execution remains
available for tests and deterministic replay.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def cuda_reusable_memory_bytes(
    device: Any = None, *, framework: Any = None, free_bytes: int | None = None,
) -> int:
    """Driver-free bytes plus this allocator's reusable, unallocated cache.

    Use this for allocation admission, keeping raw driver counters for
    telemetry. CUDA allocations can consume cached blocks or release them
    before retrying cudaMalloc. Callers still apply their process budget,
    allocator fraction and workspace margin to the result.
    """

    if framework is None:
        import torch as framework
    if free_bytes is None:
        free_bytes, _ = framework.cuda.mem_get_info(device)
    allocated = int(framework.cuda.memory_allocated(device))
    reserved = int(framework.cuda.memory_reserved(device))
    return int(free_bytes) + max(0, reserved - allocated)


def gpu_process_count(env: Mapping[str, str] | None = None) -> int:
    """Processes sharing this device; an omitted or empty setting means one."""

    source = os.environ if env is None else env
    raw = str(source.get("IRG_GPU_PROCESS_COUNT", "")).strip()
    count = int(raw) if raw else 1
    if count < 1:
        raise ValueError("IRG_GPU_PROCESS_COUNT must be positive")
    return count


def process_device_memory_budget_bytes(
    total_bytes: int, *, env: Mapping[str, str] | None = None
) -> int:
    """Share a GPU's memory budget among processes, with an optional client cap."""

    source = os.environ if env is None else env
    count = gpu_process_count(source)
    budget = int(total_bytes) // count
    raw = str(source.get("IRG_GPU_PROCESS_MEMORY_GIB", "")).strip()
    if raw:
        cap = float(raw)
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError("IRG_GPU_PROCESS_MEMORY_GIB must be positive")
        budget = min(budget, int(cap * (1 << 30)))
    return budget


def full_look_batch_rows(requested_rows: int, *, configured_rows: int,
                         memory_budget_bytes: int | None, lane_count: int) -> int:
    """Choose physical rows independently of the statistical alignment quantum.

    A full fixed look fills the device when its estimated transient working
    set fits. The existing per-row estimate and memory margin also bound
    smaller devices. With no device measurement, retain configured batching.
    """
    from mechanistic_systems.perf_gates import (
        DEFAULT_INPUT_ROW_BYTES, DEFAULT_ACTIVATION_MULTIPLIER,
        estimated_forward_row_bytes, memory_ceiling_rows,
    )

    if requested_rows < 1 or configured_rows < 1 or lane_count < 1:
        raise ValueError("look execution requires positive row and lane counts")
    if memory_budget_bytes is None:
        return configured_rows
    per_row = estimated_forward_row_bytes(
        input_row_bytes=DEFAULT_INPUT_ROW_BYTES, lane_count=lane_count,
        activation_multiplier=DEFAULT_ACTIVATION_MULTIPLIER,
    )
    ceiling = memory_ceiling_rows(free_bytes=memory_budget_bytes, per_row_bytes=per_row)
    if ceiling < 1:
        raise MemoryError("the look execution budget cannot hold one forward row")
    # Padding changes physical execution only. Its repeated tail rows are
    # removed by the existing source-to-logical-row assembly.
    requested = ((requested_rows + 7) // 8) * 8
    admitted = ceiling if ceiling < 8 else (ceiling // 8) * 8
    return min(requested, admitted)


@dataclass(frozen=True, order=True)
class ExecutionPlan:
    batch_size: int
    packing_degree: int
    dtype: str = "float32"
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0 or int(self.packing_degree) <= 0:
            raise ValueError("batch_size and packing_degree must be positive")
        if not self.dtype:
            raise ValueError("dtype must be nonempty")

    @property
    def key(self) -> str:
        return f"b{self.batch_size}-k{self.packing_degree}-{self.dtype}-gc{int(self.gradient_checkpointing)}"


@dataclass(frozen=True)
class HardwareTelemetry:
    plan: ExecutionPlan
    wall_seconds: float
    device_seconds: float
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    transferred_bytes: int = 0
    contexts: int = 1
    candidates: int = 1
    successful: bool = True
    oom: bool = False
    device_name: str = "cpu"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("wall_seconds", "device_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "transferred_bytes",
            "contexts",
            "candidates",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be nonnegative")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def useful_work(self) -> int:
        return max(1, int(self.contexts)) * max(1, int(self.candidates))

    @property
    def throughput(self) -> float:
        return self.useful_work / max(float(self.wall_seconds), 1e-12)


@dataclass(frozen=True)
class RuntimeLimits:
    memory_fraction: float = 0.85
    memory_reserve_bytes: int = 512 * 1024 * 1024
    minimum_batch_size: int = 1
    minimum_packing_degree: int = 1
    exploration_bonus: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 < float(self.memory_fraction) <= 1.0:
            raise ValueError("memory_fraction must lie in (0, 1]")
        if int(self.memory_reserve_bytes) < 0:
            raise ValueError("memory_reserve_bytes must be nonnegative")
        if int(self.minimum_batch_size) <= 0 or int(self.minimum_packing_degree) <= 0:
            raise ValueError("minimum execution dimensions must be positive")
        if float(self.exploration_bonus) < 0.0:
            raise ValueError("exploration_bonus must be nonnegative")


def configure_process_device_memory(
    device: Any, *, limits: RuntimeLimits = RuntimeLimits()
) -> dict[str, Any] | None:
    """Bound the co-resident process's PyTorch allocator, including caches.

    The existing runtime fraction and reserve leave room for CUDA allocations
    outside PyTorch. Such allocations are measured separately, not covered by
    this limit. Single-process execution keeps its existing allocator policy.
    """

    import torch

    resolved = torch.device(device)
    count = gpu_process_count()
    if count == 1 or resolved.type != "cuda":
        return None
    total = int(torch.cuda.get_device_properties(resolved).total_memory)
    share = process_device_memory_budget_bytes(total)
    limit = max(0, int(share * limits.memory_fraction) - limits.memory_reserve_bytes)
    if limit <= 0:
        raise ValueError("process device memory budget leaves no allocator space")
    get_fraction = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    previous = 1.0 if get_fraction is None else float(get_fraction(resolved))
    fraction = min(previous, float(limit) / total)
    torch.cuda.set_per_process_memory_fraction(fraction, resolved)
    return {
        "process_count": count,
        "device_capacity_bytes": total,
        "process_device_memory_budget_bytes": share,
        "allocator_limit_bytes": int(total * fraction),
        "memory_fraction": limits.memory_fraction,
        "memory_reserve_bytes": limits.memory_reserve_bytes,
        "scope": "pytorch_allocator",
    }


class AdaptiveExecutionTuner:
    """Online selector for batch size and candidate packing degree.

    The tuner never assumes that the largest fitting configuration is fastest.
    It scores plans by observed throughput, applies a small optimism bonus to
    under-explored plans, rejects configurations with recorded OOMs at the same
    or larger shape, and uses a simple peak-memory model before execution.
    """

    def __init__(
        self,
        plans: Sequence[ExecutionPlan],
        *,
        limits: RuntimeLimits = RuntimeLimits(),
        preferred_initial_plan: ExecutionPlan | None = None,
    ) -> None:
        unique = tuple(dict.fromkeys(plans))
        if not unique:
            raise ValueError("at least one execution plan is required")
        self.plans = tuple(sorted(unique, key=lambda item: (item.batch_size * item.packing_degree, item.batch_size)))
        self.limits = limits
        if (
            preferred_initial_plan is not None
            and preferred_initial_plan not in self.plans
        ):
            raise ValueError("preferred initial plan is absent from the tuner grid")
        self.preferred_initial_plan = preferred_initial_plan
        self._telemetry: dict[str, list[HardwareTelemetry]] = {plan.key: [] for plan in self.plans}

    @classmethod
    def grid(
        cls,
        *,
        batch_sizes: Iterable[int],
        packing_degrees: Iterable[int],
        dtype: str = "float32",
        limits: RuntimeLimits = RuntimeLimits(),
        preferred_initial_plan: ExecutionPlan | None = None,
    ) -> "AdaptiveExecutionTuner":
        return cls(
            [ExecutionPlan(int(batch), int(pack), dtype=dtype) for batch in batch_sizes for pack in packing_degrees],
            limits=limits,
            preferred_initial_plan=preferred_initial_plan,
        )

    def observe(self, telemetry: HardwareTelemetry) -> None:
        if telemetry.plan.key not in self._telemetry:
            raise ValueError("telemetry plan is absent from this tuner")
        self._telemetry[telemetry.plan.key].append(telemetry)

    @staticmethod
    def _representative_throughput(values: Sequence[HardwareTelemetry]) -> float:
        eligible = [
            item.throughput
            for item in values
            if not bool(item.metadata.get("warmup", False))
        ]
        if not eligible:
            eligible = [item.throughput for item in values]
        return float(np.median(np.asarray(eligible, dtype=np.float64)))

    def snapshot(self, *, workload_signature: Mapping[str, Any]) -> dict[str, object]:
        """Return a portable tuning receipt bound to one hardware/workload signature."""

        signature = {str(key): workload_signature[key] for key in sorted(workload_signature)}
        # Fail early if a caller tries to persist opaque runtime objects.
        json.dumps(signature, sort_keys=True)
        return {
            "schema": "irg_adaptive_execution_tuner_snapshot_v1",
            "workload_signature": signature,
            "limits": self.limits.__dict__,
            "plans": [plan.__dict__ for plan in self.plans],
            "observations": [
                {
                    "plan_key": key,
                    "wall_seconds": item.wall_seconds,
                    "device_seconds": item.device_seconds,
                    "peak_allocated_bytes": item.peak_allocated_bytes,
                    "peak_reserved_bytes": item.peak_reserved_bytes,
                    "transferred_bytes": item.transferred_bytes,
                    "contexts": item.contexts,
                    "candidates": item.candidates,
                    "successful": item.successful,
                    "oom": item.oom,
                    "device_name": item.device_name,
                    "metadata": dict(item.metadata),
                }
                for key, values in sorted(self._telemetry.items())
                for item in values
            ],
        }

    def restore_snapshot(
        self,
        payload: Mapping[str, Any],
        *,
        expected_workload_signature: Mapping[str, Any],
    ) -> None:
        """Restore observations only when scientific/runtime identity matches exactly."""

        if str(payload.get("schema")) != "irg_adaptive_execution_tuner_snapshot_v1":
            raise ValueError("unsupported adaptive execution tuner snapshot schema")
        expected = {str(key): expected_workload_signature[key] for key in sorted(expected_workload_signature)}
        if dict(payload.get("workload_signature", {})) != expected:
            raise ValueError("tuner snapshot workload signature does not match the current runtime")
        by_key = {plan.key: plan for plan in self.plans}
        for row in payload.get("observations", []):
            item = dict(row)
            key = str(item.pop("plan_key"))
            plan = by_key.get(key)
            if plan is None:
                continue
            self.observe(HardwareTelemetry(plan=plan, **item))

    def save(self, path: str | Path, *, workload_signature: Mapping[str, Any]) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.snapshot(workload_signature=workload_signature), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_workload_signature: Mapping[str, Any],
    ) -> "AdaptiveExecutionTuner":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        plans = tuple(ExecutionPlan(**row) for row in payload["plans"])
        tuner = cls(plans, limits=RuntimeLimits(**payload["limits"]))
        tuner.restore_snapshot(payload, expected_workload_signature=expected_workload_signature)
        return tuner

    def _predicted_peak(self, plan: ExecutionPlan) -> float | None:
        samples = [
            item
            for values in self._telemetry.values()
            for item in values
            if item.successful and not item.oom and item.peak_reserved_bytes > 0
        ]
        if not samples:
            return None
        # Fit baseline + context-batch + candidate-pack + interaction memory.
        # This distinguishes persistent model memory from the two tunable axes.
        design = np.asarray(
            [
                [
                    1.0,
                    float(item.plan.batch_size),
                    float(item.plan.packing_degree),
                    float(item.plan.batch_size * item.plan.packing_degree),
                ]
                for item in samples
            ],
            dtype=np.float64,
        )
        targets = np.asarray([item.peak_reserved_bytes for item in samples], dtype=np.float64)
        query = np.asarray(
            [1.0, float(plan.batch_size), float(plan.packing_degree), float(plan.batch_size * plan.packing_degree)],
            dtype=np.float64,
        )
        if len(samples) >= 4 and np.linalg.matrix_rank(design) >= 3:
            coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
            fitted = design @ coefficients
            positive_residual = max(0.0, float(np.max(targets - fitted)))
            prediction = float(query @ coefficients + positive_residual)
            return max(prediction, 0.0)
        # Conservative fallback with a separately estimated persistent baseline.
        smallest = min(samples, key=lambda item: item.plan.batch_size * item.plan.packing_degree)
        baseline = 0.5 * float(smallest.peak_reserved_bytes)
        rates = [
            max(0.0, item.peak_reserved_bytes - baseline)
            / max(1, item.plan.batch_size * item.plan.packing_degree)
            for item in samples
        ]
        return baseline + max(rates) * plan.batch_size * plan.packing_degree

    def _blocked_by_oom(self, plan: ExecutionPlan) -> bool:
        for values in self._telemetry.values():
            for item in values:
                if not item.oom:
                    continue
                if (
                    plan.batch_size >= item.plan.batch_size
                    and plan.packing_degree >= item.plan.packing_degree
                ):
                    return True
        return False

    def feasible_plans(
        self,
        *,
        candidate_count: int,
        context_count: int,
        free_memory_bytes: int | None = None,
    ) -> tuple[ExecutionPlan, ...]:
        candidate_count = max(1, int(candidate_count))
        context_count = max(1, int(context_count))
        allowed = None
        if free_memory_bytes is not None:
            allowed = max(
                0.0,
                float(free_memory_bytes) * self.limits.memory_fraction - self.limits.memory_reserve_bytes,
            )
        output = []
        for plan in self.plans:
            if plan.batch_size > context_count or plan.packing_degree > candidate_count:
                continue
            if self._blocked_by_oom(plan):
                continue
            predicted = self._predicted_peak(plan)
            if allowed is not None and predicted is not None and predicted > allowed:
                continue
            output.append(plan)
        if output:
            return tuple(output)
        # Guaranteed floor for a request that is executable at all.  The caller
        # can still fail cleanly if even this floor is OOM.
        floor = ExecutionPlan(
            min(context_count, self.limits.minimum_batch_size),
            min(candidate_count, self.limits.minimum_packing_degree),
            dtype=self.plans[0].dtype,
        )
        return (floor,)

    def choose(
        self,
        *,
        candidate_count: int,
        context_count: int,
        free_memory_bytes: int | None = None,
    ) -> ExecutionPlan:
        feasible = self.feasible_plans(
            candidate_count=candidate_count,
            context_count=context_count,
            free_memory_bytes=free_memory_bytes,
        )
        successful = {
            plan.key: [item for item in self._telemetry.get(plan.key, ()) if item.successful and not item.oom]
            for plan in feasible
        }
        measured = [plan for plan in feasible if successful[plan.key]]
        unknown = [plan for plan in feasible if not successful[plan.key] and not self._telemetry.get(plan.key)]
        if not measured:
            # Establish a known-safe floor before monotone expansion.
            if (
                self.preferred_initial_plan is not None
                and self.preferred_initial_plan in feasible
            ):
                return self.preferred_initial_plan
            return min(feasible, key=lambda item: (item.batch_size * item.packing_degree, item.packing_degree, item.batch_size))

        def mean_throughput(plan: ExecutionPlan) -> float:
            return self._representative_throughput(successful[plan.key])

        best_known = max(measured, key=lambda item: (mean_throughput(item), item.packing_degree, item.batch_size))
        if unknown:
            # Explore the nearest safe expansion around the current best rather
            # than jumping immediately to the largest unmeasured configuration.
            expansions = [
                plan
                for plan in unknown
                if plan.batch_size >= best_known.batch_size
                and plan.packing_degree >= best_known.packing_degree
            ]
            pool = expansions or unknown
            return min(
                pool,
                key=lambda item: (
                    abs(math.log(item.batch_size / best_known.batch_size))
                    + abs(math.log(item.packing_degree / best_known.packing_degree)),
                    item.batch_size * item.packing_degree,
                    item.packing_degree,
                    item.batch_size,
                ),
            )

        total_observations = 1 + sum(len(items) for items in self._telemetry.values())

        def score(plan: ExecutionPlan) -> tuple[float, int, int]:
            values = successful[plan.key]
            throughput = self._representative_throughput(values)
            bonus = self.limits.exploration_bonus * math.sqrt(
                math.log(total_observations + 1.0) / len(values)
            )
            return (throughput * (1.0 + bonus), plan.packing_degree, plan.batch_size)

        return max(feasible, key=score)

    def backoff(self, failed: ExecutionPlan, *, candidate_count: int, context_count: int) -> ExecutionPlan | None:
        candidates = [
            plan
            for plan in self.plans
            if plan.batch_size <= min(context_count, failed.batch_size)
            and plan.packing_degree <= min(candidate_count, failed.packing_degree)
            and (plan.batch_size < failed.batch_size or plan.packing_degree < failed.packing_degree)
            and not self._blocked_by_oom(plan)
        ]
        if not candidates:
            return None
        lower_packing = [plan for plan in candidates if plan.packing_degree < failed.packing_degree]
        if lower_packing:
            # Keep the context batch when possible and reduce the mask-pack axis
            # first, the empirically dominant activation-memory multiplier.
            return max(
                lower_packing,
                key=lambda item: (
                    item.batch_size == failed.batch_size,
                    item.batch_size,
                    item.packing_degree,
                ),
            )
        return max(candidates, key=lambda item: (item.packing_degree, item.batch_size))

    def diagnostics(self) -> dict[str, object]:
        return {
            "schema": "irg_adaptive_execution_tuner_v2",
            "limits": self.limits.__dict__,
            "plans": [plan.__dict__ for plan in self.plans],
            "preferred_initial_plan": (
                None
                if self.preferred_initial_plan is None
                else self.preferred_initial_plan.__dict__
            ),
            "observations": {
                key: [
                    {
                        "wall_seconds": item.wall_seconds,
                        "device_seconds": item.device_seconds,
                        "peak_allocated_bytes": item.peak_allocated_bytes,
                        "peak_reserved_bytes": item.peak_reserved_bytes,
                        "contexts": item.contexts,
                        "candidates": item.candidates,
                        "successful": item.successful,
                        "oom": item.oom,
                        "throughput": item.throughput,
                    }
                    for item in values
                ]
                for key, values in sorted(self._telemetry.items())
            },
        }

    def occupancy_alignment_quantum(
        self,
        *,
        candidate_count: int,
        throughput_fraction: float = 0.95,
        registered_quantum: int | None = None,
    ) -> int:
        """Return the smallest measured near-peak context batch.

        This is intentionally not the largest configured batch.  A configured
        batch may itself span several GPU scheduling waves, so rounding a
        statistical request to that value can buy compute rather than fill
        otherwise-idle lanes.  Once telemetry exists, the smallest batch whose
        median throughput is within ``throughput_fraction`` of the measured
        peak is the conservative occupancy quantum.  No evidence means no
        opportunistic oversampling.

        A registered statistical quantum remains independent of runtime timing.
        """

        fraction = float(throughput_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("throughput_fraction must lie in (0,1]")
        if registered_quantum is not None:
            quantum = int(registered_quantum)
            if quantum <= 0:
                raise ValueError("registered alignment quantum must be positive")
            return quantum
        eligible: list[tuple[ExecutionPlan, float]] = []
        for plan in self.plans:
            if plan.packing_degree > max(1, int(candidate_count)):
                continue
            observations = [
                item
                for item in self._telemetry.get(plan.key, ())
                if item.successful
                and not item.oom
                and not bool(item.metadata.get("warmup", False))
            ]
            if observations:
                eligible.append(
                    (plan, self._representative_throughput(observations))
                )
        if not eligible:
            return 1
        peak = max(value for _plan, value in eligible)
        near_peak = [
            plan.batch_size
            for plan, value in eligible
            if value + 1e-12 >= fraction * peak
        ]
        return max(1, min(near_peak))


class GpuActivitySampler:
    """Outcome-independent background GPU activity sampler.

    Utilization is operational telemetry only.  Sampling never changes
    contexts, execution plans, or stopping decisions, and failure to access
    NVML is represented explicitly rather than guessed.
    """

    def __init__(
        self,
        *,
        device: object,
        interval_seconds: float = 0.5,
        sample_callback: Any | None = None,
    ) -> None:
        interval = float(interval_seconds)
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError("GPU activity sampling interval must be positive")
        self.device = device
        self.interval_seconds = interval
        self._sample_callback = sample_callback
        self._samples: list[dict[str, float]] = []
        self._errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self._stopped = 0.0

    def _sample(self) -> dict[str, float]:
        if self._sample_callback is not None:
            return {
                str(key): float(value)
                for key, value in dict(self._sample_callback()).items()
            }
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            utilization = float(torch.cuda.utilization(self.device))
            allocated = float(torch.cuda.memory_allocated(self.device))
            reserved = float(torch.cuda.memory_reserved(self.device))
            return {
                "gpu_utilization_percent": utilization,
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
            }
        except Exception as error:
            raise RuntimeError(
                f"GPU telemetry unavailable: {type(error).__name__}: {error}"
            ) from error

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sample = self._sample()
                sample["elapsed_seconds"] = time.perf_counter() - self._started
                self._samples.append(sample)
            except Exception as error:
                if not self._errors:
                    self._errors.append(str(error))
            self._stop.wait(self.interval_seconds)

    def start(self) -> "GpuActivitySampler":
        if self._thread is not None:
            raise RuntimeError("GPU activity sampler was already started")
        self._started = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run,
            name="irg-gpu-activity-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> Mapping[str, object]:
        if self._thread is None:
            raise RuntimeError("GPU activity sampler was not started")
        self._stop.set()
        self._thread.join(timeout=max(1.0, 2.0 * self.interval_seconds))
        self._stopped = time.perf_counter()
        return self.receipt

    @property
    def receipt(self) -> Mapping[str, object]:
        utilizations = np.asarray(
            [
                value["gpu_utilization_percent"]
                for value in self._samples
                if "gpu_utilization_percent" in value
            ],
            dtype=np.float64,
        )
        allocated = [
            value.get("allocated_bytes", 0.0) for value in self._samples
        ]
        reserved = [
            value.get("reserved_bytes", 0.0) for value in self._samples
        ]
        return {
            "schema": "irg_gpu_activity_receipt_v1",
            "interval_seconds": self.interval_seconds,
            "duration_seconds": max(
                0.0,
                (self._stopped or time.perf_counter()) - self._started,
            ),
            "sample_count": len(self._samples),
            "utilization_available": bool(utilizations.size),
            "gpu_utilization_mean_percent": (
                None
                if not utilizations.size
                else float(np.mean(utilizations))
            ),
            "gpu_utilization_p50_percent": (
                None
                if not utilizations.size
                else float(np.quantile(utilizations, 0.5))
            ),
            "gpu_utilization_p95_percent": (
                None
                if not utilizations.size
                else float(np.quantile(utilizations, 0.95))
            ),
            "peak_allocated_bytes": int(max(allocated, default=0.0)),
            "peak_reserved_bytes": int(max(reserved, default=0.0)),
            "errors": tuple(self._errors),
            "outcome_independent": True,
        }


__all__ = [
    "AdaptiveExecutionTuner",
    "ExecutionPlan",
    "GpuActivitySampler",
    "HardwareTelemetry",
    "RuntimeLimits",
]
