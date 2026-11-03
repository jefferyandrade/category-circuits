"""Bounded host completion for the shared worker; GPU ownership stays local."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class DeferredItemProducts:
    future: Future
    destination: Path


class OrderedItemWriter:
    """Publish receipts in submission order and propagate earlier failures.

    The driver retires the preceding completion after each GPU item. There
    are at most two submitted host snapshots, with only one writer running.
    Callables contain host products and receipt metadata, never a runtime.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="item-products")
        self._previous: Future | None = None

    def submit(self, destination: Path, write: Callable[[], Mapping[str, Any]], *,
               abort: Callable[[], None] | None = None) -> DeferredItemProducts:
        preceding = self._previous

        def complete():
            try:
                if preceding is not None:
                    preceding.result()
                result = dict(write())
                return result, time.perf_counter()
            except BaseException:
                (destination / "gpu_done.json").unlink(missing_ok=True)
                if abort is not None:
                    abort()
                raise

        future = self._executor.submit(complete)
        self._previous = future
        return DeferredItemProducts(future, destination)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
