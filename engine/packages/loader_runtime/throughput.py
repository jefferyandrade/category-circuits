"""Shared throughput infrastructure: measured batching, pinned prefetch, residency.

None of these utilities can alter a computed number; they change when bytes
move, not what is computed. Production lanes adopt them explicitly
(batch-size adoption in a sealed lane remains parity-gated); the evidence
engines and their harnesses use them directly.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Iterator, Sequence

import torch


def measure_saturating_batch(
    forward: Callable[[torch.Tensor], object],
    *,
    device: torch.device | str,
    input_shape: tuple[int, ...] = (3, 224, 224),
    candidates: Sequence[int] = (64, 128, 256, 512, 1024),
    repeats: int = 3,
    knee_fraction: float = 0.9,
    fallback: int = 128,
) -> int:
    """The throughput-saturating batch size, measured, not configured.

    Sweeps candidate sizes with synchronized timed forwards and returns the
    smallest size achieving ``knee_fraction`` of the best observed
    throughput. On non-CUDA devices the fallback is returned untimed.
    """

    resolved = torch.device(device)
    if resolved.type != "cuda":
        return int(fallback)
    throughputs: dict[int, float] = {}
    with torch.no_grad():
        for size in candidates:
            probe = torch.randn(int(size), *input_shape, device=resolved)
            forward(probe[: min(int(size), 8)])
            torch.cuda.synchronize(resolved)
            started = time.time()
            for _ in range(int(repeats)):
                forward(probe)
            torch.cuda.synchronize(resolved)
            throughputs[int(size)] = (
                float(repeats) * float(size) / max(time.time() - started, 1e-9)
            )
            del probe
    best = max(throughputs.values())
    return int(
        min(size for size, rate in throughputs.items() if rate >= knee_fraction * best)
    )


def _pinnable(batch: "torch.Tensor") -> bool:
    """Pinning applies only to unpinned HOST tensors.

    Device-resident sources (e.g. a CUDA-resident uint8 bank under
    SCORING_PRELOAD_STORAGE_DEVICE=auto) need no host staging at all —
    attempting to pin them raises "Only dense CPU tensors can be
    pinned" (observed as a pilot failure).
    """

    return batch.device.type == "cpu" and not batch.is_pinned()


class PinnedPrefetcher:
    """Background prefetch of host batches into pinned memory.

    A single reader thread walks a (sealed, hence perfectly predictable)
    sequence of batch requests, materializes each host batch via the
    supplied gather function, copies it into a reusable pinned buffer, and
    queues it for consumption; the consumer issues genuinely asynchronous
    host-to-device copies from pinned memory. ``non_blocking`` from pageable
    mmap output is silently synchronous — this class is the fix.
    """

    def __init__(
        self,
        gather: Callable[[int], torch.Tensor | None],
        *,
        depth: int = 3,
        pin: bool = True,
    ) -> None:
        if depth < 1:
            raise ValueError("prefetch depth must be at least one")
        self._gather = gather
        self._pin = bool(pin) and torch.cuda.is_available()
        self._queue: queue.Queue = queue.Queue(maxsize=int(depth))
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        index = 0
        try:
            while not self._stop.is_set():
                batch = self._gather(index)
                if batch is None:
                    self._queue.put(None)
                    return
                if self._pin and _pinnable(batch):
                    pinned = torch.empty_like(batch, pin_memory=True)
                    pinned.copy_(batch)
                    batch = pinned
                self._queue.put(batch)
                index += 1
        except BaseException as error:  # surfaced on the consumer side
            self._error = error
            self._queue.put(None)

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            item = self._queue.get()
            if item is None:
                if self._error is not None:
                    raise self._error
                return
            yield item

    def close(self) -> None:
        self._stop.set()
        while not self._queue.empty():
            self._queue.get_nowait()


#: Upper bound, in seconds, on one ``PooledBackgroundIterator.close()``:
#: joining the feeder threads and settling in-flight stages together. A stage
#: still running at the bound is a fault (a stuck gather or copy), reported as
#: ``PipelineCloseTimeout`` rather than waited on forever.
POOLED_CLOSE_TIMEOUT_SECONDS = 120.0
_STOP_POLL_SECONDS = 0.05


class PipelineCloseTimeout(RuntimeError):
    """``PooledBackgroundIterator.close()`` could not settle within its bound."""


class PooledBackgroundIterator:
    """Stage independent items on a worker pool; emit strictly in order.

    ``stage(item)`` calls run concurrently on
    ``workers`` threads, but results are yielded in submission order —
    parallelize the STAGING, never the ORDER. In-flight work is bounded
    by ``workers + depth``; a stage exception re-raises on the consumer
    at that item's ordinal position. ``workers=1`` degenerates to the
    plain ordered pipeline.

    ``executor`` is a composition hook: a caller-owned executor gives the
    staging threads a lifetime longer than one iteration (the persistent
    panel producer keeps one pool per data plane instead of one per look).
    Ordering, in-flight bounding, and error propagation are identical; a
    caller-owned executor is never shut down here.

    Ownership on ``close()``. Every submitted future sits in ``_pending``
    until exactly one party claims it under ``_pending_lock``: the consumer
    (which then hands the result on) or ``close()`` (which awaits it and
    passes the result to ``on_discard``). An item the feeder pulled but never
    submitted belongs to the feeder, which passes it to ``on_discard_input``.
    So each staged payload is released once, by one owner, however the
    consumer thread and ``close()`` interleave.

    Settlement is tracked apart from ownership. Every submitted future also
    sits in ``_inflight`` until it has finished and been seen finished (by
    the consumer after its ``result()`` returns, or by ``close()``). A future
    the consumer has claimed but is still awaiting has left ``_pending`` but
    not ``_inflight``, so ``close()`` still waits for its stage.
    If the consumer's wait raises (a signal) before the stage has finished,
    the future stays in ``_inflight`` and goes back to ``_pending``, so
    ``close()`` waits for it and releases its output; if the stage had
    already finished, the consumer releases that output itself.
    """

    def __init__(
        self,
        items: Iterator,
        stage,
        *,
        workers: int = 1,
        depth: int = 2,
        executor=None,
        on_discard=None,
        on_discard_input=None,
    ) -> None:
        if workers < 1:
            raise ValueError("producer pool needs at least one worker")
        if depth < 1:
            raise ValueError("producer pool depth must be at least one")
        from concurrent.futures import ThreadPoolExecutor

        self._owns_executor = executor is None
        self._executor = (
            ThreadPoolExecutor(max_workers=int(workers))
            if executor is None
            else executor
        )
        self._futures: queue.Queue = queue.Queue()
        self._slots = threading.Semaphore(int(workers) + int(depth))
        self._sentinel = object()
        self._feed_error: BaseException | None = None
        self._items = items
        self._on_discard = on_discard
        self._on_discard_input = on_discard_input
        self._stop = threading.Event()
        self._pending: set = set()
        self._inflight: set = set()
        self._pending_lock = threading.Lock()
        self._close_lock = threading.Lock()
        # Set only once a close has settled every submitted stage. A close
        # that timed out leaves ``_closed`` False and records its timeout,
        # which every later close reports again.
        self._closed = False
        self._close_timeout: PipelineCloseTimeout | None = None
        self._inputs: dict = {}
        self._unsubmitted: list = []

        def _feed() -> None:
            try:
                for item in items:
                    while not self._stop.is_set():
                        if self._slots.acquire(timeout=_STOP_POLL_SECONDS):
                            break
                    if self._stop.is_set():
                        # Pulled but never submitted: this feeder is its
                        # only owner, so it alone returns the item.
                        self._unsubmitted.append(item)
                        break
                    future = self._executor.submit(stage, item)
                    with self._pending_lock:
                        self._pending.add(future)
                        self._inflight.add(future)
                        self._inputs[future] = item
                    future.add_done_callback(self._forget_input)
                    self._futures.put(future)
            except BaseException as error:
                self._feed_error = error
            finally:
                self._futures.put(self._sentinel)

        self._feeder = threading.Thread(target=_feed, daemon=True)
        self._feeder.start()

    def __iter__(self) -> Iterator:
        try:
            while not self._stop.is_set():
                try:
                    entry = self._futures.get(timeout=_STOP_POLL_SECONDS)
                except queue.Empty:
                    continue
                if entry is self._sentinel:
                    if self._feed_error is not None:
                        raise self._feed_error
                    return
                # Claim the future before awaiting it. A future ``close()``
                # has already claimed is released there; handing it on as
                # well would release its payload twice.
                with self._pending_lock:
                    owned = entry in self._pending
                    self._pending.discard(entry)
                if not owned:
                    return
                try:
                    result = entry.result()
                    # ``result()`` returned, so the stage has finished. Keep
                    # this ownership update under the interruption handler.
                    with self._pending_lock:
                        self._inflight.discard(entry)
                except BaseException as error:
                    # A stage error, or an interrupted wait or ownership
                    # update before the consumer handed on the output.
                    self._abandon(entry, error)
                    raise
                finally:
                    self._slots.release()
                yield result
        finally:
            if self._owns_executor:
                self._executor.shutdown(wait=False, cancel_futures=False)

    def _abandon(self, future, error: BaseException) -> None:
        """The consumer's wait on its claimed ``future`` raised ``error``.

        A future counts as settled only once it is done. An unfinished one
        stays in ``_inflight`` and returns to ``_pending``, so ``close()``
        waits for it and releases its output. A finished one with an output
        the consumer never handed on is released here, by its one owner.
        """

        with self._pending_lock:
            if not future.done():
                self._pending.add(future)
                return
            self._inflight.discard(future)
        if future.cancelled() or future.exception() is not None:
            return
        if self._on_discard is not None:
            try:
                self._on_discard(future.result())
            except BaseException as release_error:
                error.add_note(f"releasing the unserved output also failed: {release_error!r}")

    def _forget_input(self, future) -> None:
        with self._pending_lock:
            if not future.cancelled():
                self._inputs.pop(future, None)

    def _signal_stop(self) -> None:
        self._stop.set()
        # Wake a feeder waiting for a slot and a consumer waiting on the
        # queue; both also poll the flag. Slot accounting ends at stop.
        self._slots.release()
        self._futures.put(self._sentinel)

    def close(self, *, timeout: float | None = None, discard_result=None, discard_input=None) -> None:
        """Stop feeding and settle this iterator's work before freeing buffers.

        A shared executor stays alive. Its futures belonging to this pass are
        still awaited, including a future the consumer has claimed and is
        still awaiting or whose wait was interrupted, and a future cancelled
        in the executor's queue counts as settled, so an interrupted pass
        cannot write into the next
        pass's rings, and each unserved payload goes to ``on_discard`` exactly
        once. When the items come from another ``PooledBackgroundIterator``,
        this iterator's feeder (that iterator's consumer) is stopped and
        joined before the upstream iterator settles, so the two never race
        over the same queue.

        Nothing waits past ``timeout`` seconds (default
        ``POOLED_CLOSE_TIMEOUT_SECONDS``) in total, including the wait for a
        close already running on another thread. If a feeder or stage is
        still running then, the remaining steps still run and
        ``PipelineCloseTimeout`` is raised at the end; the payloads of stages
        that did not settle are left unreleased (never released twice).

        Repeated calls are serialized. Once a close has settled every stage,
        later calls return at once. After a close has timed out, including
        one that timed out waiting for another close, every later call raises
        ``PipelineCloseTimeout`` again (chained to the first), because a stage
        may still be running and able to write.
        """

        bound = POOLED_CLOSE_TIMEOUT_SECONDS if timeout is None else float(timeout)
        deadline = time.monotonic() + bound

        def remaining() -> float:
            return max(0.0, deadline - time.monotonic())

        if not self._close_lock.acquire(timeout=remaining()):
            timed_out = PipelineCloseTimeout(
                f"PooledBackgroundIterator.close() did not settle within {bound:g} s: "
                "another close() of this iterator was still settling its stages"
            )
            # Remembered like a settlement timeout, so every later close
            # raises again; the close lock is not retaken.
            with self._pending_lock:
                if self._close_timeout is None:
                    self._close_timeout = timed_out
            raise timed_out
        try:
            if self._close_timeout is not None:
                raise PipelineCloseTimeout(
                    "PooledBackgroundIterator.close() already timed out; "
                    "its unsettled stages may still be running"
                ) from self._close_timeout
            if self._closed:
                return
            if discard_result is not None:
                self._on_discard = discard_result
            if discard_input is not None:
                self._on_discard_input = discard_input
            try:
                release_error = self._settle(bound, remaining)
            except PipelineCloseTimeout as error:
                with self._pending_lock:
                    if self._close_timeout is None:
                        self._close_timeout = error
                raise
            self._closed = True
        finally:
            self._close_lock.release()
        if release_error is not None:
            raise release_error

    def _settle(self, bound: float, remaining) -> BaseException | None:
        """One close: stop, join, settle. Raises ``PipelineCloseTimeout``.

        Returns the first error raised while releasing payloads, if any;
        every stage has settled by then.
        """

        from concurrent.futures import TimeoutError as FutureTimeout

        faults: list[str] = []
        release_error: BaseException | None = None
        upstream = (
            self._items if isinstance(self._items, PooledBackgroundIterator) else None
        )
        self._signal_stop()
        if upstream is not None:
            # This iterator's feeder is upstream's consumer: stop both before
            # joining, so the feeder is not left waiting on upstream's queue.
            upstream._signal_stop()
        while True:
            try:
                self._feeder.join(remaining())
                break
            except (KeyboardInterrupt, InterruptedError) as error:
                if release_error is None:
                    release_error = error
                if not remaining():
                    break
        if self._feeder.is_alive():
            faults.append("feeder thread still running")
        for item in self._unsubmitted:
            if self._on_discard_input is not None:
                try:
                    self._on_discard_input(item)
                except BaseException as error:
                    if release_error is None:
                        release_error = error
        self._unsubmitted.clear()
        with self._pending_lock:
            queued = tuple((future, item) for future, item in self._inputs.items()
                           if future in self._pending)
        for future, item in queued:
            if future.cancel():
                with self._pending_lock:
                    self._inputs.pop(future, None)
                if self._on_discard_input is not None:
                    try:
                        self._on_discard_input(item)
                    except BaseException as error:
                        if release_error is None:
                            release_error = error
        if upstream is not None:
            try:
                upstream.close(timeout=remaining())
            except PipelineCloseTimeout as error:
                faults.append(f"upstream: {error}")
            except BaseException as error:
                release_error = error
        with self._pending_lock:
            unclaimed = tuple(self._pending)
            self._pending.clear()
            active = tuple(self._inflight)
        # Await every submitted stage still running, including one the
        # consumer has claimed: cancelling a stage would strand its input
        # payload's lease, whereas its normal stage returns it. Only the
        # unclaimed outputs are this close's to release.
        # Settled means done(). That includes a future cancelled in a
        # caller-owned executor's queue (shutdown with cancel_futures): it
        # never runs, and ``wait()`` alone would count it as unsettled.
        for future in active:
            while not future.done():
                try:
                    future.result(timeout=remaining())
                except FutureTimeout:
                    break
                except (KeyboardInterrupt, InterruptedError) as error:
                    if release_error is None:
                        release_error = error
                    if not remaining():
                        break
                except BaseException:
                    break
        settled = {future for future in active if future.done()}
        unsettled = [future for future in active if future not in settled]
        with self._pending_lock:
            self._inflight.difference_update(settled)
            # Outputs handed back by a consumer whose wait was interrupted
            # after the first snapshot are this close's to release too.
            unclaimed += tuple(self._pending)
            self._pending.clear()
        for future in unclaimed:
            if future not in settled:
                continue
            if future.cancelled() or future.exception() is not None:
                continue
            if self._on_discard is not None:
                try:
                    self._on_discard(future.result())
                except BaseException as error:
                    if release_error is None:
                        release_error = error
        if unsettled:
            faults.append(
                f"{len(unsettled)} in-flight stage(s) unsettled; "
                "the payloads of unclaimed ones were not released"
            )
        while True:
            try:
                self._futures.get_nowait()
            except queue.Empty:
                break
        if self._owns_executor:
            self._executor.shutdown(wait=not faults, cancel_futures=True)
        if faults:
            timed_out = PipelineCloseTimeout(
                f"PooledBackgroundIterator.close() did not settle within {bound:g} s: "
                + "; ".join(faults)
            )
            if release_error is not None:
                timed_out.add_note(f"a payload release also failed: {release_error!r}")
            raise timed_out
        return release_error


class BackgroundIterator:
    """Run a source iterator on a worker thread; consume in order.

    The source's per-item work (gather, host-to-device issue) happens on the
    worker while the consumer computes — the generic overlap primitive for
    pipelining first-touch loads under the same panel's forwards. Order is
    the source's order; source exceptions re-raise on the consumer.
    """

    def __init__(self, source: Iterator, *, depth: int = 2) -> None:
        if depth < 1:
            raise ValueError("background iteration depth must be at least one")
        self._queue: queue.Queue = queue.Queue(maxsize=int(depth))
        self._error: BaseException | None = None
        self._sentinel = object()

        def _run() -> None:
            try:
                for item in source:
                    self._queue.put(item)
            except BaseException as error:
                self._error = error
            finally:
                self._queue.put(self._sentinel)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def __iter__(self) -> Iterator:
        while True:
            item = self._queue.get()
            if item is self._sentinel:
                if self._error is not None:
                    raise self._error
                return
            yield item


class DeviceResidentBank:
    """A uint8 bank uploaded once and indexed on device thereafter.

    Each byte crosses the bus exactly once — the transfer floor. The bank
    stays uint8 on device; consumers dequantize after gathering.
    """

    def __init__(self, bank: torch.Tensor, device: torch.device | str) -> None:
        if bank.dtype != torch.uint8:
            raise ValueError("device residency expects a uint8 bank")
        self.device = torch.device(device)
        self._bank = bank.to(self.device, non_blocking=False)

    @property
    def rows(self) -> int:
        return int(self._bank.shape[0])

    def gather(self, indices: Sequence[int] | torch.Tensor) -> torch.Tensor:
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return self._bank.index_select(0, index_tensor)


__all__ = [
    "BackgroundIterator",
    "DeviceResidentBank",
    "PinnedPrefetcher",
    "PipelineCloseTimeout",
    "POOLED_CLOSE_TIMEOUT_SECONDS",
    "measure_saturating_batch",
]
