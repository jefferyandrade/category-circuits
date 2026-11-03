"""Per-category work held in memory by the shared-process worker.

The worker (``run_registered_resnet18_pack.py --worker-items``) runs every arm
of one category back to back in one process. While it does, it activates one
:class:`CategoryMemo`; the pilot's call sites consult :func:`active` and share
what is the same for every arm of the category: the census fields, dense rows,
training layouts, frozen banks, dense-only reductions and terminal products. The memo is dropped when
the category ends. With no active memo (a fresh-process arm) nothing is shared
and every call site runs exactly as before.

The worker also publishes a small process context (worker identity, item
order) that the pilot adds to each receipt.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, Mapping

import numpy as np


@dataclass
class CategoryMemo:
    """What one category's arms share. Entries are written only when complete."""

    category: str
    # P acquisitions by complete shared-field binding, including the run seed.
    # Consumer families and partition seeds never select an acquisition.
    census: dict[str, Any] = field(default_factory=dict)
    # The most recently acquired P census. Every other family evaluates its own
    # labels and order certificate on: {"family", "cache_key", "shared_key", "result", ...}.
    shared_census: dict[str, Any] | None = None
    # Dense frozen category banks by category, dense state and exact fit-panel
    # request digest, with the
    # dense terminal products once an arm has captured them.
    dense_banks: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    # Host-only category inputs and products. Device arrays belong to the item
    # provider and are released at the existing GPU completion boundary.
    dense_panels: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    # One line per shared or computed product, for the worker's own record.
    events: list[dict[str, Any]] = field(default_factory=list)

    def note(self, **event: Any) -> None:
        self.events.append(dict(event))

    def key(self, dense_state_digest: str, terminal_panel_digest: str) -> tuple[str, str, str]:
        """Bind shared work to the category, dense state and exact panel."""

        return self.category, str(dense_state_digest), str(terminal_panel_digest)


_ACTIVE: CategoryMemo | None = None
_PROCESS_CONTEXT: dict[str, Any] | None = None


def active() -> CategoryMemo | None:
    """The memo of the category the worker is running, or None."""

    return _ACTIVE


@contextmanager
def category(name: str) -> Iterator[CategoryMemo]:
    """Hold one category's memo for the arms run inside this block."""

    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("a category memo is already active")
    memo = CategoryMemo(category=str(name))
    _ACTIVE = memo
    try:
        yield memo
    finally:
        _ACTIVE = None
        memo.census.clear()
        memo.shared_census = None
        memo.dense_banks.clear()
        memo.dense_panels.clear()


@contextmanager
def arm_context(context: Mapping[str, Any]) -> Iterator[None]:
    """Publish the worker identity and item order for the arm's receipt."""

    global _PROCESS_CONTEXT
    _PROCESS_CONTEXT = dict(context)
    try:
        yield
    finally:
        _PROCESS_CONTEXT = None


def process_context() -> dict[str, Any] | None:
    return None if _PROCESS_CONTEXT is None else dict(_PROCESS_CONTEXT)


def freeze_arrays(value: Any) -> Any:
    """Mark every array in a nested product read-only; shared values stay intact.

    A consumer that wrote into a shared array would change a later arm's
    input; with the flag cleared that write raises instead.
    """

    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, Mapping):
        for item in value.values():
            freeze_arrays(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            freeze_arrays(item)
    return value


def container_copy(value: Any) -> Any:
    """Copy dicts, lists and tuples; share arrays and scalars."""

    if isinstance(value, dict):
        return {key: container_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [container_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(container_copy(item) for item in value)
    return value


def arrays_content_hash(value: Any) -> str:
    """SHA-256 over every array of a nested product, in key order.

    Each array is hashed on its own thread (hashlib releases the GIL for large
    buffers); the digest binds names, dtypes and shapes as well as the bytes.
    """

    records: list[tuple[str, np.ndarray]] = []

    def walk(prefix: str, item: Any) -> None:
        if isinstance(item, np.ndarray):
            records.append((prefix, item))
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                walk(f"{prefix}/{key}", item[key])
        elif isinstance(item, (tuple, list)):
            for index, entry in enumerate(item):
                walk(f"{prefix}[{index}]", entry)
        elif item is None or isinstance(item, (str, int, float, bool)):
            records.append((prefix, np.asarray(repr(item))))

    walk("", value)

    def digest(array: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(array)
        if contiguous.dtype.hasobject:
            raise TypeError("content hash requires numeric or string arrays")
        return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()

    with ThreadPoolExecutor(max_workers=max(1, min(8, len(records)))) as pool:
        digests = list(pool.map(digest, (array for _, array in records)))
    total = hashlib.sha256()
    for (name, array), item in zip(records, digests, strict=True):
        total.update(f"{name}|{array.dtype.str}|{array.shape}|{item}\n".encode())
    return total.hexdigest()


__all__ = [
    "CategoryMemo",
    "active",
    "arm_context",
    "arrays_content_hash",
    "category",
    "container_copy",
    "freeze_arrays",
    "process_context",
]
