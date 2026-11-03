"""Lazy complete Cartesian traversal by a separable proposal priority.

This is ordering only. A priority is neither finite work nor an acceptance
bound. Every yielded exchange still needs the caller's exact intervention
checks. Storage is linear in the input queues, never their Cartesian product.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator, Sequence

import numpy as np


def _coordinates(name: str, values: Sequence[int]) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in 'iu'):
        raise ValueError(f'{name} must be one-dimensional integer coordinates')
    if raw.size and (np.any(raw < 0) or np.any(raw > np.iinfo(np.int64).max)):
        raise ValueError(f'{name} must be nonnegative int64 coordinates')
    out = raw.astype(np.int64, copy=True)
    if np.unique(out).size != out.size:
        raise ValueError(f'{name} must be unique')
    return out


def _priorities(name: str, values: Sequence[float], count: int) -> np.ndarray:
    out = np.array(values, dtype=np.float64, copy=True)
    if out.shape != (count,) or not np.isfinite(out).all():
        raise ValueError(f'{name} must be finite and match its coordinates')
    return out


@dataclass(frozen=True)
class OrderedExchange:
    priority: float
    drop_priority: float
    add_priority: float
    drop_position: int
    add_route: int
    drop_route: int

    @property
    def key(self) -> tuple[float, float, float, int, int, int]:
        # Component priorities break rounded-sum ties before coordinates. This
        # makes each row monotone in either orientation of the Cartesian grid.
        return (self.priority, self.drop_priority, self.add_priority,
                self.drop_position, self.add_route, self.drop_route)


class OrderedExchangeStream(Iterator[OrderedExchange]):
    """Merge sorted Cartesian rows with a numeric heap on the shorter axis.

    Input coordinates and priorities are copied and sorted once. No route count
    truncation, pair materialization, filesystem scratch or per-pair visited set
    is used. The caller may stop after an actual accepted improvement. Complete
    exhaustion still takes O(D*A*log(min(D,A))) traversal work in the worst case.
    """

    def __init__(self, *, drop_routes: Sequence[int], drop_positions: Sequence[int],
                 drop_priorities: Sequence[float], add_routes: Sequence[int],
                 add_priorities: Sequence[float]):
        drops = _coordinates('drop_routes', drop_routes)
        positions = _coordinates('drop_positions', drop_positions)
        adds = _coordinates('add_routes', add_routes)
        if positions.size != drops.size:
            raise ValueError('drop_positions must match drop_routes')
        if np.intersect1d(drops, adds).size:
            raise ValueError('drop and add routes must be disjoint')
        dp = _priorities('drop_priorities', drop_priorities, drops.size)
        ap = _priorities('add_priorities', add_priorities, adds.size)
        di = np.lexsort((drops, positions, dp))
        ai = np.lexsort((adds, ap))
        self._drops, self._positions, self._dp = drops[di], positions[di], dp[di]
        self._adds, self._ap = adds[ai], ap[ai]
        self.drop_count, self.add_count = int(drops.size), int(adds.size)
        self.domain_size = self.drop_count * self.add_count
        self._drop_rows = self.drop_count <= self.add_count
        rows = min(self.drop_count, self.add_count) if self.domain_size else 0
        self._width = max(self.drop_count, self.add_count)
        self._heap = np.arange(rows, dtype=np.int64)
        self._cursor = np.zeros(rows, dtype=np.int64)
        self._size = rows
        self.visited = 0
        self.comparisons = 0
        self._closed = False
        # Retained numeric payload only; constructor sort/copy scratch, Python
        # objects, transient comparison tuples and allocator overhead are extra.
        self.retained_payload_bytes = sum(a.nbytes for a in (
            self._drops, self._positions, self._dp, self._adds, self._ap,
            self._heap, self._cursor))
        for pos in range(rows // 2 - 1, -1, -1):
            self._sift_down(pos)

    def _entry(self, row: int) -> OrderedExchange:
        col = int(self._cursor[row])
        d, a = (row, col) if self._drop_rows else (col, row)
        dp, ap = float(self._dp[d]), float(self._ap[a])
        priority = dp + ap
        # Finite operands can overflow to infinity. Infinity is an ordered
        # priority, not nonfinite work; component tie breaks remain monotone.
        if math.isnan(priority):
            raise ValueError('exchange priority is NaN')
        return OrderedExchange(priority, dp, ap, int(self._positions[d]),
                               int(self._adds[a]), int(self._drops[d]))

    def _less(self, left: int, right: int) -> bool:
        self.comparisons += 1
        return self._entry(int(self._heap[left])).key < self._entry(int(self._heap[right])).key

    def _sift_down(self, pos: int) -> None:
        while 2*pos+1 < self._size:
            child = 2*pos+1
            if child+1 < self._size and self._less(child+1, child):
                child += 1
            if not self._less(child, pos):
                return
            self._heap[pos], self._heap[child] = self._heap[child], self._heap[pos]
            pos = child

    def __iter__(self) -> OrderedExchangeStream:
        return self

    def __next__(self) -> OrderedExchange:
        if self._closed or self._size == 0:
            raise StopIteration
        row = int(self._heap[0])
        entry = self._entry(row)
        self._cursor[row] += 1
        if self._cursor[row] >= self._width:
            self._size -= 1
            if self._size:
                self._heap[0] = self._heap[self._size]
        self._sift_down(0)
        self.visited += 1
        return entry

    def close(self) -> None:
        self._closed = True
        self._size = 0
        self._drops = self._positions = self._dp = self._adds = self._ap = None
        self._heap = self._cursor = None
