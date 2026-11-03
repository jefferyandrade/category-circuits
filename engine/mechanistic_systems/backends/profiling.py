"""Low-overhead phase profiling with optional NVTX annotations."""

from __future__ import annotations

import math
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

try:  # Profiling must remain importable in CPU-only analysis environments.
    import torch
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]


@contextmanager
def nvtx_range(label: str) -> Iterator[None]:
    pushed = False
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.nvtx.range_push(str(label))
            pushed = True
        except Exception:
            pushed = False
    try:
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()


@dataclass
class PhaseProfiler:
    """Accumulate wall-clock phase timings suitable for a runtime receipt."""

    phase_time_s: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    phase_calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cuda_events_enabled: bool = field(
        default_factory=lambda: os.getenv("MECHANISTIC_PROFILE_CUDA_EVENTS", "0").strip()
        in {"1", "true", "yes", "on"}
    )
    _cuda_events: list[tuple[str, Any, Any]] = field(default_factory=list, repr=False)
    _cuda_event_cursor: int = field(default=0, repr=False)
    _cuda_phase_time_s: dict[str, float] = field(
        default_factory=lambda: defaultdict(float),
        repr=False,
    )
    _cuda_phase_calls: dict[str, int] = field(
        default_factory=lambda: defaultdict(int),
        repr=False,
    )
    # Per-run output context. These objects are never part of snapshot().
    phase_timing: Any | None = field(default=None, repr=False)
    run_product_writer: Any | None = field(default=None, repr=False)

    def record(self, name: str, elapsed_seconds: float, *, calls: int = 1) -> None:
        """Record an already-timed phase without changing scientific state."""

        label = str(name).strip()
        elapsed = float(elapsed_seconds)
        call_count = int(calls)
        if not label:
            raise ValueError("profile phase name must be nonempty")
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("profile phase elapsed_seconds must be finite and nonnegative")
        if call_count <= 0:
            raise ValueError("profile phase calls must be positive")
        self.phase_time_s[label] += elapsed
        self.phase_calls[label] += call_count

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        label = str(name).strip()
        if not label:
            raise ValueError("profile phase name must be nonempty")
        started = time.perf_counter()
        cuda_start = cuda_end = None
        if self.cuda_events_enabled and torch is not None and torch.cuda.is_available():
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        try:
            with nvtx_range(label):
                yield
        finally:
            if cuda_start is not None and cuda_end is not None:
                cuda_end.record()
                self._cuda_events.append((label, cuda_start, cuda_end))
            self.record(label, time.perf_counter() - started)

    def snapshot(self, *, wait: bool = True) -> dict[str, Any]:
        """Phase totals; ``wait=False`` reads only CUDA events already complete."""

        snapshot = {
            "phase_time_s": {
                key: float(value) for key, value in sorted(self.phase_time_s.items())
            },
            "phase_calls": {
                key: int(value) for key, value in sorted(self.phase_calls.items())
            },
        }
        # The look's control event owns its sole host wait. A later
        # receipt snapshot consumes phase events after device work finishes.
        from mechanistic_systems.inference.device_bce import DEVICE_LOOK_ACTIVE

        if (self._cuda_event_cursor < len(self._cuda_events)
                and not DEVICE_LOOK_ACTIVE.get()):
            if wait:
                torch.cuda.synchronize()
            for label, start, end in self._cuda_events[self._cuda_event_cursor :]:
                if not wait and not end.query():
                    break
                self._cuda_phase_time_s[label] += (
                    float(start.elapsed_time(end)) / 1000.0
                )
                self._cuda_phase_calls[label] += 1
                self._cuda_event_cursor += 1
        if self._cuda_phase_time_s:
            snapshot["cuda_phase_time_s"] = {
                key: float(value)
                for key, value in sorted(self._cuda_phase_time_s.items())
            }
            snapshot["cuda_phase_calls"] = {
                key: int(value)
                for key, value in sorted(self._cuda_phase_calls.items())
            }
        return snapshot

    def cuda_event_count(self) -> int:
        """Phase event pairs recorded so far (a cursor for ``cuda_phase_seconds``)."""

        return len(self._cuda_events)

    def cuda_phase_seconds(self, start: int, stop: int) -> dict[str, float]:
        """CUDA seconds per phase of the event pairs ``[start, stop)``.

        Callers read these events only after their execution has completed.
        """

        seconds: dict[str, float] = defaultdict(float)
        for label, begin, end in self._cuda_events[int(start) : int(stop)]:
            seconds[label] += float(begin.elapsed_time(end)) / 1000.0
        return dict(seconds)


__all__ = ["PhaseProfiler", "nvtx_range"]
