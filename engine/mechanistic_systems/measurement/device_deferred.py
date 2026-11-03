"""Device fault flags and event-timed receipts, settled off the critical path.

A readout queues its forward and its delivery checks and returns without a
host read.  Each check is a device flag:

* inside a device look (the M4 device statistics route) the flag joins the
  look's checks and is folded into the look's one status read;
* otherwise it is copied asynchronously into pinned host memory, stream
  ordered right after the check, and settled later: after a decision (the
  copy completed long before, so nothing waits) and at the end of a walk.

The detailed host report of a failed check (positions, sources, digests) is
built only when its flag is set; it is the raised error.  There are no
retries: a delivery fault ends the run.

Receipts that carry CUDA event times are finalized in order once their
events have completed, so the execution chain keeps its order; a blocking
drain finalizes the rest at export.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Callable

import torch


class DeviceFaultLedger:
    """Pending device fault flags with their error factories."""

    def __init__(self) -> None:
        self._pending: deque[tuple[Any, Any, Callable[[], BaseException]]] = deque()
        self._lock = threading.Lock()
        self.noted = 0
        self.settled = 0

    def note(
        self, flag: Any, error_factory: Callable[[], BaseException], *,
        fold_into_look: bool = True,
    ) -> None:
        """Queue a device flag (nonzero: the check failed)."""

        from mechanistic_systems.inference.device_bce import defer_device_check

        if fold_into_look and defer_device_check(flag, error_factory):
            return
        value = flag.reshape(()).to(torch.int32)
        event = None
        if value.device.type == "cuda":
            host = torch.empty((), dtype=torch.int32, pin_memory=True)
            host.copy_(value, non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(value.device))
            value = host
        with self._lock:
            self._pending.append((value, event, error_factory))
            self.noted += 1

    def settle(self, *, block: bool = False) -> None:
        """Raise the first failed check among the completed flags (all if ``block``)."""

        ready = []
        with self._lock:
            while self._pending:
                value, event, factory = self._pending[0]
                if event is not None and not block and not event.query():
                    break
                self._pending.popleft()
                ready.append((value, event, factory))
        for value, event, factory in ready:
            if block and event is not None:
                event.synchronize()
            self.settled += 1
            if int(value.item()):
                self.clear()
                raise factory()

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()

    def __len__(self) -> int:
        return len(self._pending)


class DeferredReceipts:
    """Receipt finalizers that wait for CUDA events, run in FIFO order."""

    def __init__(self) -> None:
        self._pending: deque[tuple[tuple[Any, ...], Callable[[], None]]] = deque()

    def add(self, events: tuple[Any, ...], finalize: Callable[[], None]) -> None:
        self._pending.append((tuple(event for event in events if event is not None), finalize))
        self.drain()

    def drain(self, *, block: bool = False) -> None:
        while self._pending:
            events, finalize = self._pending[0]
            if not block and not all(event.query() for event in events):
                return
            if block:
                for event in events:
                    event.synchronize()
            self._pending.popleft()
            finalize()

    def __len__(self) -> int:
        return len(self._pending)


# The process's delivery-check flags (one walk runs per process at a time).
DEVICE_FAULTS = DeviceFaultLedger()


def settle_device_faults(*, block: bool = False) -> None:
    DEVICE_FAULTS.settle(block=block)


__all__ = [
    "DEVICE_FAULTS",
    "DeferredReceipts",
    "DeviceFaultLedger",
    "settle_device_faults",
]
