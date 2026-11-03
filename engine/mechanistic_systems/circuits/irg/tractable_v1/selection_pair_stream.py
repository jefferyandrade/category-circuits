"""Bounded ranked-pair storage; no scientific selection authority.

Canonical spooling screens once and merges private sorted runs when necessary.
The page/cursor primitive remains an executable rescan reference. Storage bounds
cover numeric payload separately from interpreter, I/O buffering and process RSS.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
import struct
import tempfile

PairKey = tuple[float, float, int, int, int]
_DTYPE = np.dtype([('work', '<f8'), ('energy', '<f8'), ('position', '<i8'),
                   ('add', '<i8'), ('drop', '<i8')])


class RankedPairPage:
    def __init__(self, *, payload_bytes: int, maximum_pairs: int, after: PairKey | None = None):
        if type(payload_bytes) is not int or payload_bytes < _DTYPE.itemsize:
            raise ValueError('pair page payload must hold at least one numeric pair')
        if type(maximum_pairs) is not int or maximum_pairs <= 0:
            raise ValueError('maximum_pairs must be a positive integer')
        self.capacity = min(maximum_pairs, payload_bytes // _DTYPE.itemsize)
        self._maximum_pairs = maximum_pairs
        self._all_fit = self.capacity == maximum_pairs
        self._rows = np.empty(self.capacity, dtype=_DTYPE)
        self.after = after
        self.count = 0
        self.seen = 0
        self.remaining = 0
        self._ordered = False

    @property
    def payload_bytes(self) -> int:
        return self._rows.nbytes

    def _key(self, index: int) -> PairKey:
        row = self._rows[index]
        return (float(row['work']), float(row['energy']), int(row['position']),
                int(row['add']), int(row['drop']))

    def offer(self, key: PairKey) -> None:
        if self._ordered:
            raise ValueError('cannot modify an ordered pair page')
        self.seen += 1
        if self.seen > self._maximum_pairs:
            raise ValueError('pair producer exceeded its declared finite domain')
        if self.after is not None and key <= self.after:
            return
        self.remaining += 1
        if self._all_fit:
            self._rows[self.count] = key
            self.count += 1
            return
        if self.count < self.capacity:
            index = self.count
            self.count += 1
            while index:
                parent = (index - 1) // 2
                if self._key(parent) >= key:
                    break
                self._rows[index] = self._rows[parent]
                index = parent
            self._rows[index] = key
        elif key < self._key(0):
            index = 0
            while 2 * index + 1 < self.count:
                child = 2 * index + 1
                if child + 1 < self.count and self._key(child + 1) > self._key(child):
                    child += 1
                if self._key(child) <= key:
                    break
                self._rows[index] = self._rows[child]
                index = child
            self._rows[index] = key

    def keys(self):
        if not self._ordered:
            # Heapsort is in-place: no second capacity-sized numeric bank.
            self._rows[:self.count].sort(order=_DTYPE.names, kind='heapsort')
            self._ordered = True
        for index in range(self.count):
            yield self._key(index)

    def write_sorted(self, output):
        if not self._ordered:
            self._rows[:self.count].sort(order=_DTYPE.names, kind='heapsort')
            self._ordered = True
        self._rows[:self.count].tofile(output)


_RECORD = struct.Struct('<ddqqq')


def _read_key(stream):
    raw = stream.read(_RECORD.size)
    if not raw:
        return None
    if len(raw) != _RECORD.size:
        raise ValueError('truncated private pair run')
    return _RECORD.unpack(raw)


class RankedPairSpool:
    """One scan, bounded numeric storage and lazily created private sorted runs.

    Online binary merges retain at most logarithmically many run names. A merge
    needs two input records, one output record and fixed file buffering, never
    another whole numeric page. Scratch payload/traffic are reported separately.
    Call close on success, early termination and exceptions.
    """
    def __init__(self, *, payload_bytes: int, maximum_pairs: int):
        self._page = RankedPairPage(payload_bytes=payload_bytes, maximum_pairs=maximum_pairs)
        self._capacity = self._page.capacity
        self._payload_bytes = payload_bytes
        self._maximum_pairs = maximum_pairs
        self.payload_peak_bytes = self._page.payload_bytes
        self.seen = 0
        self.run_count = 0
        self.merge_count = 0
        self.bytes_written = 0
        self.bytes_read = 0
        self.scratch_peak_bytes = 0
        self._live_bytes = 0
        self._directory = None
        self._levels = []
        self._final = None
        self._serial = 0
        self._finished = False
        self._closed = False

    def _new_path(self):
        if self._directory is None:
            self._directory = tempfile.TemporaryDirectory(prefix='v1-pairs-')
        self._serial += 1
        return Path(self._directory.name) / f'run-{self._serial}.bin'

    def offer(self, key: PairKey):
        if self._finished or self._closed:
            raise ValueError('pair spool is no longer writable')
        if self.seen >= self._maximum_pairs:
            raise ValueError('pair producer exceeded its declared finite domain')
        if self._page.count == self._capacity:
            self._flush()
            self._page = RankedPairPage(payload_bytes=self._payload_bytes, maximum_pairs=self._capacity)
        self._page.offer(key)
        self.seen += 1

    def _merge(self, left, right):
        path = self._new_path()
        expected = left[1] + right[1]
        self.scratch_peak_bytes = max(self.scratch_peak_bytes, self._live_bytes + expected * _RECORD.size)
        count = 0
        with left[0].open('rb') as a, right[0].open('rb') as b, path.open('wb') as out:
            ka, kb = _read_key(a), _read_key(b)
            while ka is not None or kb is not None:
                if kb is None or (ka is not None and ka <= kb):
                    key = ka
                    ka = _read_key(a)
                else:
                    key = kb
                    kb = _read_key(b)
                out.write(_RECORD.pack(*key))
                count += 1
        if count != expected:
            raise ValueError('private pair run count changed')
        self.bytes_written += count * _RECORD.size
        self.bytes_read += count * _RECORD.size
        self.merge_count += 1
        left[0].unlink()
        right[0].unlink()
        # Output replaces equal input payload, so live bytes are unchanged.
        return path, count

    def _flush(self):
        path = self._new_path()
        count = self._page.count
        with path.open('wb') as output:
            self._page.write_sorted(output)
        self._page = None  # Release the bank before any merge or next page.
        self.run_count += 1
        self.bytes_written += count * _RECORD.size
        self._live_bytes += count * _RECORD.size
        self.scratch_peak_bytes = max(self.scratch_peak_bytes, self._live_bytes)
        run = path, count
        level = 0
        while level < len(self._levels) and self._levels[level] is not None:
            run = self._merge(self._levels[level], run)
            self._levels[level] = None
            level += 1
        if level == len(self._levels):
            self._levels.append(run)
        else:
            self._levels[level] = run

    def finish(self):
        if self._closed or self._finished:
            raise ValueError('pair spool cannot be finished again')
        if self._directory is not None:
            if self._page is not None and self._page.count:
                self._flush()
            for run in self._levels:
                if run is not None:
                    self._final = run if self._final is None else self._merge(self._final, run)
            self._levels.clear()
        self._finished = True

    def keys(self):
        if not self._finished or self._closed:
            raise ValueError('pair spool is not readable')
        if self._final is None:
            yield from self._page.keys()
        else:
            count = 0
            with self._final[0].open('rb') as source:
                while (key := _read_key(source)) is not None:
                    self.bytes_read += _RECORD.size
                    count += 1
                    yield key
            if count != self.seen:
                raise ValueError('private pair spool count changed')

    def close(self):
        self._page = None
        self._levels.clear()
        self._closed = True
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
