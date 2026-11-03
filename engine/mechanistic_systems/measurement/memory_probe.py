"""In-process host/device memory sampling for OOM attribution.

WHY THIS EXISTS, and why the obvious alternatives do not work:

* ``sstat`` mid-run is WRONG here, measured: it reported 8.3 GiB for a job
  whose ``sacct`` MaxRSS at completion was 358.7 GiB. Only the completed-job
  ``sacct`` figure is reliable, and it arrives too late to attribute anything.
* An OOM kill is SIGKILL from the cgroup. There is no unwinding, no atexit, no
  final flush. Anything buffered at the moment of death is lost, so every
  sample is written and flushed INDIVIDUALLY as one JSON line. The trace of a
  killed run is exactly the trace up to the sample before the kill, which is
  the part that matters.
* Peak RSS must come from the kernel's own high-water mark (``VmHWM`` /
  ``ru_maxrss``), not from sampled ``VmRSS``: a transient that is allocated and
  freed between two ticks is invisible to sampling but is precisely the kind of
  allocation that kills a job. Reporting both makes a sampling miss visible as
  a gap between them.

``tracemalloc`` is deliberately OPT-IN and separate from RSS sampling: it costs
roughly 2-3x on allocation-heavy paths, while reading ``/proc/self/status`` is
a few microseconds. The default configuration is therefore cheap enough to arm
on every run, and the expensive attribution is turned on only when a run is
already known to be under investigation.

tracemalloc also sees ONLY Python allocations. Torch tensor payloads and NumPy
buffers are largely invisible to it, so a large gap between RSS and
``tracemalloc_current`` is itself the finding: it says the growth is in native
buffers, and the torch counters in the same record say whether it is device or
host.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import threading
import time
import tracemalloc
from typing import Any, Mapping
from types import ModuleType

MEMORY_PROBE_ENV = "IRG_MEMORY_PROBE"
MEMORY_PROBE_INTERVAL_ENV = "IRG_MEMORY_PROBE_INTERVAL_SECONDS"
MEMORY_PROBE_TRACEMALLOC_ENV = "IRG_MEMORY_PROBE_TRACEMALLOC"
MEMORY_PROBE_TOP_ENV = "IRG_MEMORY_PROBE_TOP"
MEMORY_PROBE_SCHEMA = "irg_memory_probe_sample_v1"

DEFAULT_INTERVAL_SECONDS = 15.0
DEFAULT_TOP_ALLOCATIONS = 12


def _truthy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() not in {"0", "false", "no", ""}


def _maxrss_bytes() -> int:
    """Kernel high-water RSS, normalized to bytes across platforms.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS/BSD. Getting this
    wrong silently reports a 1024x error, so the platform test is explicit.
    """

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _proc_status_bytes() -> dict[str, int]:
    """Current and peak RSS from /proc (Linux only; empty elsewhere)."""

    result: dict[str, int] = {}
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if not line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                    continue
                name, _, rest = line.partition(":")
                parts = rest.split()
                if len(parts) >= 2 and parts[1].lower() == "kb":
                    result[name.strip()] = int(parts[0]) * 1024
    except (OSError, ValueError):
        return {}
    return result


def _torch_device_bytes(device: Any = None) -> dict[str, int]:
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    try:
        if not torch.cuda.is_available():
            return {}
        arguments = () if device is None else (device,)
        return {
            "cuda_allocated": int(torch.cuda.memory_allocated(*arguments)),
            "cuda_reserved": int(torch.cuda.memory_reserved(*arguments)),
            "cuda_max_allocated": int(torch.cuda.max_memory_allocated(*arguments)),
            "cuda_max_reserved": int(torch.cuda.max_memory_reserved(*arguments)),
        }
    except Exception:
        # Never let instrumentation take down a run it is only observing.
        return {}


def worker_memory_boundary(roots: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Observe an item boundary without retaining tensors or changing execution.

    Allocator statistics distinguish live allocations from cached or split
    blocks. The optional owner walk is diagnostic only and counts aliases once.
    The driver settles and collects item resources before the cleanup sample.
    """

    row: dict[str, Any] = {"label": str(label), "wall_seconds": time.time()}
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        row["cuda_available"] = False
        return row
    row.update(
        cuda_available=True,
        allocated_bytes=int(torch.cuda.memory_allocated()),
        reserved_bytes=int(torch.cuda.memory_reserved()),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        memory_stats={str(key): int(value) for key, value in torch.cuda.memory_stats().items()},
    )
    if _truthy(os.environ.get("IRG_WORKER_MEMORY_DIAGNOSTICS")):
        from .device_image_gather import _REGISTRATIONS, RegisteredBank

        populations = {
            str(id(bank)): bank.resident
            for bank in _REGISTRATIONS.values() if isinstance(bank, RegisteredBank)
        }
        mapped_storages = {
            (str(view.device), int(view.untyped_storage().data_ptr()))
            for bank in _REGISTRATIONS.values() if isinstance(bank, RegisteredBank)
            for view in bank.views.values()
        }
        row["tensor_owners"] = tensor_storage_breakdown(
            {"resident_image_populations": populations, "worker_inputs": roots}, device="cuda",
            exclude_storages=mapped_storages,
        )
    return row


def tensor_storage_breakdown(
    roots: Mapping[str, Any], *, device: Any = "cpu", node_cap: int = 100000,
    exclude_storages: set[tuple[str, int]] | frozenset[tuple[str, int]] = frozenset(),
) -> dict[str, Any]:
    """Attribute live tensor storage to its first named owner, counting aliases once.

    Walk declared owners, not the whole interpreter. Allocator totals expose
    the remaining live allocation and cached blocks. Autograd saved tensors,
    compiled graph pools and allocations outside these owners are not inferred
    from tensor shapes; their bytes stay in the unattributed remainder.
    Non-owning views of CUDA-mapped host banks can be excluded by storage key;
    their address space is not an HBM allocation.
    """

    torch = sys.modules.get("torch")
    seen_objects: set[int] = set()
    seen_storages: set[tuple[str, int]] = set()
    components = {}
    examined = 0
    skipped = 0
    truncated = False
    for name, root in roots.items():
        row = {"device_bytes": 0, "host_tensor_bytes": 0, "storages": 0}
        pending = [root]
        while pending:
            value = pending.pop()
            # Never let instrumentation take down a run it is only observing:
            # an object the walk cannot read (a sparse tensor without a
            # storage, a proxy that refuses attribute access) is counted and
            # skipped. Its bytes stay in the unattributed remainder.
            try:
                if value is None or isinstance(value, (str, bytes, int, float, bool, type, ModuleType)):
                    continue
                if id(value) in seen_objects:
                    continue
                seen_objects.add(id(value))
                examined += 1
                if examined > node_cap:
                    truncated = True
                    break
                if torch is not None and isinstance(value, torch.Tensor):
                    storage = value.untyped_storage()
                    key = (str(value.device), int(storage.data_ptr()))
                    if key in exclude_storages:
                        continue
                    if key not in seen_storages:
                        seen_storages.add(key)
                        field = "device_bytes" if value.device.type == "cuda" else "host_tensor_bytes"
                        row[field] += int(storage.nbytes())
                        row["storages"] += 1
                    continue
                if isinstance(value, Mapping):
                    pending.extend(value.values())
                elif isinstance(value, (tuple, list, set, frozenset)):
                    pending.extend(value)
                elif hasattr(value, "__dict__") and (
                    not callable(value) or (torch is not None and isinstance(value, torch.nn.Module))
                ):
                    pending.extend(vars(value).values())
            except Exception:
                skipped += 1
        components[str(name)] = row
        if truncated:
            break
    report: dict[str, Any] = {
        "schema": "irg_tensor_storage_breakdown_v1",
        "components": components,
        "accounted_device_bytes": sum(row["device_bytes"] for row in components.values()),
        "nodes_examined": examined,
        "skipped_objects": skipped,
        "truncated": truncated,
        "attribution": "first_named_owner_per_storage",
    }
    try:
        on_cuda = torch is not None and torch.device(device).type == "cuda"
    except Exception as error:
        report["allocator_counters_error"] = repr(error)
        on_cuda = False
    if on_cuda:
        try:
            from mechanistic_systems.measurement.gpu_runtime import (
                gpu_process_count, process_device_memory_budget_bytes,
            )

            counters = _torch_device_bytes(device)
            report.update(counters)
            if counters:
                allocated, reserved = counters["cuda_allocated"], counters["cuda_reserved"]
                free, total = torch.cuda.mem_get_info(device)
                report.update({
                    "process_count": gpu_process_count(),
                    "process_device_memory_budget_bytes": process_device_memory_budget_bytes(int(total)),
                    "allocator_cached_bytes": max(0, reserved - allocated),
                    "unattributed_allocated_bytes": max(0, allocated - report["accounted_device_bytes"]),
                    "device_used_bytes": int(total) - int(free),
                    "outside_process_allocator_bytes": max(0, int(total) - int(free) - reserved),
                    "outside_process_allocator_scope": "CUDA context, libraries, driver and other processes",
                })
        except Exception as error:
            report["allocator_counters_error"] = repr(error)
    return report


def sample(
    *,
    label: str = "",
    top_allocations: int = 0,
) -> dict[str, Any]:
    """Return one memory sample. Safe to call from any thread."""

    status = _proc_status_bytes()
    record: dict[str, Any] = {
        "schema": MEMORY_PROBE_SCHEMA,
        "monotonic_seconds": round(time.monotonic(), 6),
        "wall_seconds": round(time.time(), 6),
        "pid": os.getpid(),
        "label": str(label),
        "maxrss_bytes": _maxrss_bytes(),
    }
    if status:
        record["vmrss_bytes"] = status.get("VmRSS", 0)
        record["vmhwm_bytes"] = status.get("VmHWM", 0)
        record["vmsize_bytes"] = status.get("VmSize", 0)
    record.update(_torch_device_bytes())
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        record["tracemalloc_current_bytes"] = int(current)
        record["tracemalloc_peak_bytes"] = int(peak)
        if top_allocations > 0:
            snapshot = tracemalloc.take_snapshot()
            statistics = snapshot.statistics("lineno")[:top_allocations]
            record["tracemalloc_top"] = [
                {
                    "location": str(entry.traceback[0]) if entry.traceback else "?",
                    "size_bytes": int(entry.size),
                    "count": int(entry.count),
                }
                for entry in statistics
            ]
    return record


class MemoryProbe:
    """Timer-sampled memory trace written as flushed JSON lines.

    Each sample is one line, flushed immediately, so a SIGKILL leaves a usable
    trace ending at the last completed sample.
    """

    def __init__(
        self,
        output_path: str,
        *,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        trace_allocations: bool = False,
        top_allocations: int = DEFAULT_TOP_ALLOCATIONS,
    ) -> None:
        interval = float(interval_seconds)
        if interval <= 0.0:
            raise ValueError("memory probe interval must be positive")
        self.output_path = str(output_path)
        self.interval_seconds = interval
        self.trace_allocations = bool(trace_allocations)
        self.top_allocations = max(0, int(top_allocations))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle: Any = None
        self._started_tracemalloc = False
        self.sample_count = 0

    def _emit(self, record: Mapping[str, Any]) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        # Flush per record: a killed process keeps every completed sample.
        self._handle.flush()
        self.sample_count += 1

    def mark(self, label: str) -> None:
        """Record one labelled sample immediately (phase boundaries)."""

        self._emit(
            sample(
                label=label,
                top_allocations=(
                    self.top_allocations if self.trace_allocations else 0
                ),
            )
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.mark("tick")
            except Exception:
                # Observation must never be able to fail the run it observes.
                return

    def start(self) -> "MemoryProbe":
        if self._thread is not None:
            return self
        directory = os.path.dirname(self.output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._handle = open(self.output_path, "a", encoding="utf-8")
        if self.trace_allocations and not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_tracemalloc = True
        self.mark("start")
        self._thread = threading.Thread(
            target=self._run,
            name="irg-memory-probe",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds))
        self._thread = None
        try:
            self.mark("stop")
        finally:
            if self._started_tracemalloc and tracemalloc.is_tracing():
                tracemalloc.stop()
                self._started_tracemalloc = False
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def __enter__(self) -> "MemoryProbe":
        return self.start()

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


def probe_from_environment(output_path: str) -> MemoryProbe | None:
    """Build a probe from the environment, or None when disarmed.

    Returns None rather than a no-op probe so callers can skip the file
    entirely when the probe is off.
    """

    if not _truthy(os.environ.get(MEMORY_PROBE_ENV)):
        return None
    raw_interval = os.environ.get(MEMORY_PROBE_INTERVAL_ENV, "").strip()
    interval = float(raw_interval) if raw_interval else DEFAULT_INTERVAL_SECONDS
    if interval <= 0.0:
        raise ValueError(
            f"{MEMORY_PROBE_INTERVAL_ENV} must be a positive number of seconds"
        )
    raw_top = os.environ.get(MEMORY_PROBE_TOP_ENV, "").strip()
    top = int(raw_top) if raw_top else DEFAULT_TOP_ALLOCATIONS
    return MemoryProbe(
        output_path,
        interval_seconds=interval,
        trace_allocations=_truthy(
            os.environ.get(MEMORY_PROBE_TRACEMALLOC_ENV)
        ),
        top_allocations=top,
    )


__all__ = [
    "DEFAULT_INTERVAL_SECONDS",
    "DEFAULT_TOP_ALLOCATIONS",
    "MEMORY_PROBE_ENV",
    "MEMORY_PROBE_INTERVAL_ENV",
    "MEMORY_PROBE_SCHEMA",
    "MEMORY_PROBE_TOP_ENV",
    "MEMORY_PROBE_TRACEMALLOC_ENV",
    "MemoryProbe",
    "probe_from_environment",
    "sample",
]
