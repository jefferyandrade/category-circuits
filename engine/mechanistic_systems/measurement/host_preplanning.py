"""Private host descriptions for one decision ahead; no execution policy."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import copy
from dataclasses import dataclass, field, replace
import os
import threading
import time
from typing import Any, Mapping

from mechanistic_systems.measurement import iid_streams as _iid
from mechanistic_systems.measurement.packet_sha256 import digest_packet


def host_preplanning_enabled() -> bool:
    """Explicit CPU planner setting; the CUDA walk always plans ahead."""

    value = os.environ.get("IRG_HOST_PREPLANNING", "0")
    if value not in {"0", "1"}:
        raise ValueError("IRG_HOST_PREPLANNING must be 0 or 1")
    return value == "1"


HOST_PREPLANNING_SAVING_SECONDS = (8, 15)
HOST_PREPLANNING_SAVING_EVIDENCE = "modelled_after_calibration_and_mask_changes"


_DRAW_TIMINGS = threading.local()


def _sha256_packet(messages: tuple[bytes, ...]) -> tuple[bytes, ...]:
    """The stream's unchanged SHA-256 inputs, one native packet call."""

    started, cpu_started = time.perf_counter(), time.thread_time()
    output, native_wall, native_cpu = digest_packet(messages)
    _DRAW_TIMINGS.sha_wall = getattr(_DRAW_TIMINGS, "sha_wall", 0.0) + time.perf_counter() - started
    _DRAW_TIMINGS.sha_cpu = getattr(_DRAW_TIMINGS, "sha_cpu", 0.0) + time.thread_time() - cpu_started
    _DRAW_TIMINGS.sha_native_wall = getattr(_DRAW_TIMINGS, "sha_native_wall", 0.0) + native_wall
    _DRAW_TIMINGS.sha_native_cpu = getattr(_DRAW_TIMINGS, "sha_native_cpu", 0.0) + native_cpu
    return output


def _uniform_packet(specifications) -> tuple[int, ...]:
    """The existing SHA-256 rejection law, compressed a packet at a time."""

    specifications = tuple(specifications)
    if any(size <= 0 or index < 0 for _prefix, index, size in specifications):
        raise ValueError("IID draw index and population size are invalid")
    pending = tuple(range(len(specifications)))
    values = [None] * len(specifications)
    counter = 0
    while pending:
        messages = tuple(specifications[position][0] + b"%d\x00%d" % (specifications[position][1], counter) for position in pending)
        for position, digest in zip(pending, _sha256_packet(messages)):
            value = int.from_bytes(digest, "big")
            size = specifications[position][2]
            if value < _iid._RANGE_SIZE - (_iid._RANGE_SIZE % size):
                values[position] = value % size
        pending = tuple(position for position in pending if values[position] is None)
        counter += 1
    return tuple(values)


def _packet_draws(stream, ranges):
    """Private draw preparation with batched native arithmetic and unchanged ids."""

    design, role, stream_id = stream.design, stream.role, stream.stream_id
    sealed = not bool(design.sampling_with_replacement)
    uniform_head = b"\x00".join((
        _iid.BINARY_IID_STREAM_ALGORITHM.encode("ascii"),
        str(design.randomization_namespace).encode("utf-8"),
        str(role).encode("utf-8"), str(stream_id).encode("utf-8"),
    )) + b"\x00"
    descriptions, cursors, specifications = [], {}, []
    for stratum, start, stop in ranges:
        if start < 0 or stop < start:
            raise ValueError("IID draw range is invalid")
        try:
            population = design.source_context_ids_by_stratum[stratum]
        except KeyError as error:
            raise ValueError("IID stream draw uses an unknown stratum") from error
        prefix = uniform_head + str(stratum).encode("utf-8") + b"\x00"
        if sealed:
            cursor = design._cursor(role=role, stream_id=stream_id, stratum=stratum)
            if stop > len(population):
                raise ValueError("sealed stratum exhausted")
            earlier = cursors.get(stratum)
            cursors[stratum] = (cursor, prefix, max(stop, 0 if earlier is None else earlier[2]))
            descriptions.append((stratum, start, stop, population, cursor))
        else:
            descriptions.append((stratum, start, stop, population, len(specifications)))
            specifications.extend((prefix, index, len(population)) for index in range(start, stop))
    if sealed:
        for cursor, prefix, stop in cursors.values():
            specifications.extend((prefix, index, len(cursor.population) - index) for index in range(len(cursor.prefix), stop))
    uniforms = _uniform_packet(specifications)
    if sealed:
        offset = 0
        for cursor, _prefix, stop in cursors.values():
            for index in range(len(cursor.prefix), stop):
                swap = index + uniforms[offset]
                offset += 1
                here, there = cursor.displaced.get(index, index), cursor.displaced.get(swap, swap)
                cursor.displaced[swap] = here
                cursor.displaced.pop(index, None)
                cursor.prefix.append(there)
    messages, values = [], []
    stream_tail = ',"stream_hash":' + _iid._json_str(str(stream.content_hash)) + "}"
    for stratum, start, stop, population, cursor_or_offset in descriptions:
        source_indices = (cursor_or_offset.prefix[start:stop] if sealed
                          else uniforms[cursor_or_offset:cursor_or_offset + stop - start])
        tail = ',"stratum":' + _iid._json_str(str(stratum)) + stream_tail
        for index, source_index in zip(range(start, stop), source_indices):
            source = population[source_index]
            messages.append(('{"draw_index":%d,"source_context_id":%s%s'
                             % (index, _iid._json_str(source), tail)).encode("utf-8"))
            values.append((index, source, stratum))
    draw_prefix = "swor:" if sealed else "iid:"
    # The design and range checks already established the six field
    # invariants. Fill the same slots as the existing packet draw constructor.
    new = object.__new__
    set_id, set_source, set_stratum, set_index, set_role, set_stream = _iid._IID_DRAW_SLOT_SETTERS
    draws = []
    for (index, source, stratum), digest in zip(values, _sha256_packet(tuple(messages))):
        draw = new(_iid.IIDContextDraw)
        set_id(draw, draw_prefix + digest.hex())
        set_source(draw, source)
        set_stratum(draw, stratum)
        set_index(draw, index)
        set_role(draw, role)
        set_stream(draw, stream_id)
        draws.append(draw)
    return tuple(draws)


def _cursor_state(design: Any, role: str, stream_id: str, strata: tuple[str, ...]):
    cursors = getattr(design, "_cursors", {})
    return tuple(
        None if (cursor := cursors.get((role, stream_id, stratum))) is None
        else (tuple(cursor.prefix), tuple(sorted(cursor.displaced.items())))
        for stratum in strata
    )


def frozen_ranges(ranges: Any) -> tuple[tuple[str, int, int], ...]:
    """``(stratum, start, stop)`` triples from triples or ``{stratum: (start, stop)}``."""

    if isinstance(ranges, tuple):
        return ranges
    return tuple((str(h), int(a), int(b)) for h, (a, b) in ranges.items())


@dataclass(frozen=True)
class PreparedDrawRanges:
    """Draws and private Fisher--Yates cursors, never a live registration.

    Only the chosen branch can adopt these cursors, after checking the role,
    design identity, stream, ranges and original cursor state. With-replacement
    designs are stateless. Immutable population tuples can be shared; cursor
    lists and displacement dictionaries cannot.

    ``prepare`` touches the live design only through single-key
    dictionary reads and immutable fields, so it may run on the draw-ahead
    thread while the main thread works; every mutation is on the private copy.
    The stream receipt and the physical source ids are computed with the
    draws.  A packet is adopted at most once; adoption moves its cursors into
    the live design instead of copying them.
    """

    design: Any
    role: str
    stream_id: str
    ranges: tuple[tuple[str, int, int], ...]
    base_state: tuple[Any, ...]
    draws: tuple[Any, ...]
    receipt: Mapping[str, Any]
    sources: tuple[str, ...]
    preparation_wall_seconds: float = field(default=0.0, compare=False)
    preparation_thread_cpu_seconds: float = field(default=0.0, compare=False)
    sha_packet_wall_seconds: float = field(default=0.0, compare=False)
    sha_packet_thread_cpu_seconds: float = field(default=0.0, compare=False)
    sha_native_wall_seconds: float = field(default=0.0, compare=False)
    sha_native_thread_cpu_seconds: float = field(default=0.0, compare=False)

    @property
    def preparation_gil_cpu_upper_bound_seconds(self):
        """Preparation CPU excluding the measured native call without the GIL."""
        return max(self.preparation_thread_cpu_seconds - self.sha_native_thread_cpu_seconds, 0.0)

    @classmethod
    def prepare(cls, design, *, role, stream_id, ranges):
        started, cpu_started = time.perf_counter(), time.thread_time()
        sha_wall, sha_cpu = getattr(_DRAW_TIMINGS, "sha_wall", 0.0), getattr(_DRAW_TIMINGS, "sha_cpu", 0.0)
        native_wall, native_cpu = getattr(_DRAW_TIMINGS, "sha_native_wall", 0.0), getattr(_DRAW_TIMINGS, "sha_native_cpu", 0.0)
        role, stream_id = str(role), str(stream_id)
        frozen = frozen_ranges(ranges)
        strata = tuple(h for h, _, _ in frozen)
        base_state = _cursor_state(design, role, stream_id, strata)
        private = copy(design)
        object.__setattr__(private, "source_context_ids_by_stratum", dict(design.source_context_ids_by_stratum))
        object.__setattr__(private, "stratum_weights", dict(design.stratum_weights))
        # Copy only this stream's cursors, not the run's accumulated designs.
        # Single-key reads: the live dictionary is never iterated.
        if hasattr(design, "_cursors"):
            live = design._cursors
            private_cursors = {}
            for stratum in strata:
                key = (role, stream_id, stratum)
                cursor = live.get(key)
                if cursor is not None:
                    private_cursors[key] = replace(
                        cursor, prefix=list(cursor.prefix), displaced=dict(cursor.displaced)
                    )
            object.__setattr__(private, "_cursors", private_cursors)
        stream = private.stream(role=role, stream_id=stream_id)
        draws = _packet_draws(stream, frozen)
        receipt = stream.receipt(draws)
        sources = tuple(draw.source_context_id for draw in draws)
        return cls(private, role, stream_id, frozen, base_state, draws, receipt, sources,
            time.perf_counter() - started, time.thread_time() - cpu_started,
            getattr(_DRAW_TIMINGS, "sha_wall", 0.0) - sha_wall,
            getattr(_DRAW_TIMINGS, "sha_cpu", 0.0) - sha_cpu,
            getattr(_DRAW_TIMINGS, "sha_native_wall", 0.0) - native_wall,
            getattr(_DRAW_TIMINGS, "sha_native_cpu", 0.0) - native_cpu)

    def adopt(self, design, *, role, stream_id, ranges) -> tuple[Any, ...] | None:
        frozen = frozen_ranges(ranges)
        if (
            str(role) != self.role or str(stream_id) != self.stream_id
            or design.content_hash != self.design.content_hash or frozen != self.ranges
            or _cursor_state(design, self.role, self.stream_id, tuple(h for h, _, _ in frozen))
            != self.base_state
        ):
            return None
        if hasattr(design, "_cursors"):
            # Adoption runs on the main thread; the packet is spent afterwards.
            if self.__dict__.get("_adopted", False):
                return None
            object.__setattr__(self, "_adopted", True)
            design._cursors.update(self.design._cursors)
        return self.draws


_DRAW_AHEAD: tuple[int, ThreadPoolExecutor] | None = None
_DRAW_AHEAD_LOCK = threading.Lock()


class PendingDrawRanges:
    """A ``PreparedDrawRanges`` being computed on the draw-ahead thread.

    A branch's first-look draws depend on the branch only through its
    stream id, never on the current decision's outcome, so they are computed
    off the main thread while the current look runs.  The main thread waits
    only where it needs them (posting the branch's images, adopting the chosen
    branch). Preparation errors propagate at the first consumer; a taken
    branch does not repeat its preparation on the main thread.
    """

    __slots__ = ("role", "stream_id", "ranges", "_future")

    def __init__(self, future, role: str, stream_id: str, ranges) -> None:
        self._future = future
        self.role, self.stream_id = str(role), str(stream_id)
        self.ranges = frozen_ranges(ranges)

    def done(self) -> bool:
        return self._future.done()

    def cancel(self) -> None:
        """Drop an unchosen branch's draws if they have not started."""

        self._future.cancel()

    def result(self) -> PreparedDrawRanges:
        return self._future.result()

    @property
    def draws(self) -> tuple[Any, ...]:
        return self.result().draws

    @property
    def sources(self) -> tuple[str, ...]:
        return self.result().sources

    @property
    def receipt(self) -> Mapping[str, Any]:
        return self.result().receipt

    def adopt(self, design, *, role, stream_id, ranges) -> tuple[Any, ...] | None:
        return self.result().adopt(design, role=role, stream_id=stream_id, ranges=ranges)


def _draw_ahead_executor() -> ThreadPoolExecutor:
    """The one draw-ahead thread of this process (a forked child makes its own)."""

    global _DRAW_AHEAD
    with _DRAW_AHEAD_LOCK:
        if _DRAW_AHEAD is None or _DRAW_AHEAD[0] != os.getpid():
            _DRAW_AHEAD = (
                os.getpid(), ThreadPoolExecutor(max_workers=1, thread_name_prefix="irg-draw-ahead"),
            )
        return _DRAW_AHEAD[1]


def draw_ahead(design, *, role, stream_id, ranges) -> PendingDrawRanges:
    """Queue ``PreparedDrawRanges.prepare`` on the one draw-ahead thread."""

    frozen = frozen_ranges(ranges)
    future = _draw_ahead_executor().submit(
        PreparedDrawRanges.prepare, design, role=role, stream_id=stream_id, ranges=frozen,
    )
    return PendingDrawRanges(future, role, stream_id, frozen)


_BRANCH_AHEAD: tuple[int, ThreadPoolExecutor] | None = None
_BRANCH_AHEAD_LOCK = threading.Lock()


class PendingHostBranches:
    """Both outcome descriptions, independently of the main thread.

    Timings separate the worker's wall and thread CPU from the main thread's
    wait. Thread CPU includes NumPy and Python; it is an upper bound on CPU
    spent holding the interpreter lock, not a measurement of lock ownership.
    """

    def __init__(self, future, diagnostics):
        self._future = future
        self._diagnostics = diagnostics

    def result(self):
        started = time.perf_counter()
        pending = not self._future.done()
        try:
            return self._future.result()
        finally:
            diagnostics = self._diagnostics
            diagnostics["planning_commit_wait_seconds"] = diagnostics.get("planning_commit_wait_seconds", 0.0) + time.perf_counter() - started
            if pending:
                diagnostics["planning_commit_waits"] = diagnostics.get("planning_commit_waits", 0) + 1


def plan_host_branches(prepare, diagnostics) -> PendingHostBranches:
    """Queue one accept/reject preparation on the process's planner thread."""

    global _BRANCH_AHEAD
    with _BRANCH_AHEAD_LOCK:
        if _BRANCH_AHEAD is None or _BRANCH_AHEAD[0] != os.getpid():
            _BRANCH_AHEAD = (os.getpid(), ThreadPoolExecutor(max_workers=1, thread_name_prefix="irg-branch-ahead"))
        executor = _BRANCH_AHEAD[1]
    context = copy_context()

    def run():
        started, cpu_started = time.perf_counter(), time.thread_time()
        try:
            return context.run(prepare)
        except Exception:
            diagnostics["planning_failures"] += 1
            raise
        finally:
            diagnostics["planning_worker_seconds"] = diagnostics.get("planning_worker_seconds", 0.0) + time.perf_counter() - started
            diagnostics["planning_worker_thread_cpu_seconds"] = diagnostics.get("planning_worker_thread_cpu_seconds", 0.0) + time.thread_time() - cpu_started
            diagnostics["planning_worker_tasks"] = diagnostics.get("planning_worker_tasks", 0) + 1

    return PendingHostBranches(executor.submit(run), diagnostics)
