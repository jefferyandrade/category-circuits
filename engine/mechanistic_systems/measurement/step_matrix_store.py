"""Append-only on-disk store for per-decision step-category matrices.

WHY: the step-category accumulator held nine ``(steps, strata, width)`` float32
families as Python lists of per-decision slices, entirely in HOST RAM, for the
whole walk -- 4.93 GiB at the landed AlexNet shape (9 x 147 x 1000 x 1000 x 4 B).
Finalize then made it worse: ``np.stack`` allocated the joined plane while every
slice was still referenced, so both copies coexisted (~9.9 GiB peak).

Nothing re-reads a decision's slice during the walk -- it is complete the moment
the decision resolves -- so the slice can go to disk immediately. Steady-state
residency becomes one decision's slices (~36 MB) instead of the whole walk's.

THE DECISION COUNT IS NOT KNOWN IN ADVANCE (147 vs 187 across landed arms), so
this writes a real ``.npy`` whose header is RESERVED at a fixed width up front
and PATCHED with the true leading dimension at finalize. That keeps the offset
arithmetic exact and avoids the alternative's full-file copy: writing raw bytes
and prepending a header later would re-read and re-write all 5 GiB.

Writes are optionally handed to a single background thread with a bounded
queue. Each decision writes a DISJOINT, strictly appended slice, so there is no
ordering hazard between producer and writer beyond the queue's own FIFO order,
which is preserved. NumPy releases the GIL during the write, so the walk thread
overlaps it. Synchronous mode remains available and is what the tests compare
against.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Any, Iterable, Mapping

import numpy as np

# Reserve a generous fixed-width header so the leading dimension can grow from
# 0 to any realistic decision count without changing the header's byte length.
# numpy requires (magic + version + length field + header) to be 64-aligned.
_HEADER_RESERVE_BYTES = 512
_MAGIC = b"\x93NUMPY"


def _header_bytes(shape: tuple[int, ...], dtype: np.dtype) -> bytes:
    """Build a fixed-width .npy v1.0 header for ``shape``.

    The dict is space-padded to ``_HEADER_RESERVE_BYTES`` total so that
    rewriting it with a larger leading dimension cannot change its length.
    """

    descr = np.lib.format.dtype_to_descr(dtype)
    text = (
        "{'descr': %r, 'fortran_order': False, 'shape': %r, }"
        % (descr, tuple(int(value) for value in shape))
    )
    # 6 magic + 2 version + 2 length field = 10 bytes precede the header text.
    padding = _HEADER_RESERVE_BYTES - 10 - len(text) - 1
    if padding < 0:
        raise ValueError("step-matrix header does not fit the reserved width")
    body = (text + " " * padding + "\n").encode("latin1")
    length = len(body)
    if (10 + length) % 64:
        raise ValueError("step-matrix header is not 64-byte aligned")
    return _MAGIC + bytes([1, 0]) + int(length).to_bytes(2, "little") + body


class _MatrixFamilyFile:
    """One append-only ``.npy`` whose leading dimension is patched at close."""

    def __init__(self, path: str, *, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.path = str(path)
        self.slice_shape = tuple(int(value) for value in shape)
        self.dtype = np.dtype(dtype)
        self.count = 0
        self._handle = open(self.path, "wb")
        self._handle.write(_header_bytes((0, *self.slice_shape), self.dtype))

    def append(self, array: np.ndarray) -> None:
        values = np.asarray(array, dtype=self.dtype)
        if tuple(values.shape) != self.slice_shape:
            raise ValueError(
                f"step-matrix slice has shape {tuple(values.shape)}, "
                f"expected {self.slice_shape}"
            )
        # ascontiguousarray is a no-op for the C-contiguous slices the
        # accumulator produces, so this does not add a copy on the hot path.
        np.ascontiguousarray(values).tofile(self._handle)
        self.count += 1

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.flush()
        self._handle.seek(0)
        self._handle.write(
            _header_bytes((self.count, *self.slice_shape), self.dtype)
        )
        self._handle.flush()
        self._handle.close()
        self._handle = None


class StreamingMatrixStore:
    """Per-decision matrix families streamed to disk, optionally off-thread."""

    def __init__(
        self,
        directory: str,
        *,
        families: Iterable[str],
        slice_shape: tuple[int, ...],
        dtype: Any = np.float32,
        asynchronous: bool = True,
        queue_depth: int = 2,
    ) -> None:
        self.directory = str(directory)
        os.makedirs(self.directory, exist_ok=True)
        resolved = np.dtype(dtype)
        self._files = {
            str(name): _MatrixFamilyFile(
                os.path.join(self.directory, f"{name}.npy"),
                shape=slice_shape,
                dtype=resolved,
            )
            for name in families
        }
        self._closed = False
        self._error: BaseException | None = None
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        if asynchronous:
            self._queue = queue.Queue(maxsize=max(1, int(queue_depth)))
            self._thread = threading.Thread(
                target=self._drain, name="step-matrix-writer", daemon=True
            )
            self._thread.start()

    @property
    def family_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._files))

    def counts(self) -> dict[str, int]:
        return {name: item.count for name, item in self._files.items()}

    def path(self, family: str) -> str:
        return self._files[str(family)].path

    def _drain(self) -> None:
        assert self._queue is not None
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                family, array = item
                self._files[family].append(array)
            except BaseException as error:  # surfaced at the next append/close
                if self._error is None:
                    self._error = error
            finally:
                self._queue.task_done()

    def _reraise(self) -> None:
        if self._error is not None:
            error, self._error = self._error, None
            raise error

    def append(self, family: str, array: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("step-matrix store is closed")
        self._reraise()
        name = str(family)
        if name not in self._files:
            raise KeyError(f"unknown step-matrix family {name!r}")
        if self._queue is None:
            self._files[name].append(array)
            return
        # The producing buffer is handed off, so it must not be mutated
        # afterwards. The accumulator builds a fresh array per decision, but
        # copying a reference here (not the data) keeps that contract explicit.
        self._queue.put((name, array))

    def wait_idle(self) -> None:
        """Block until every queued slice has been written (or raised).

        The producer may hand the writer VIEWS into a host buffer it intends
        to reuse (the accumulator's single stacked D2H target).  Reuse is only
        safe once the writer has consumed every view, which this barrier
        guarantees: ``queue.join`` returns after the last ``task_done`` for
        the items enqueued so far.  Synchronous stores are always idle.  A
        writer-side error surfaces here exactly as it would at the next
        append.
        """

        if self._queue is not None:
            self._queue.join()
        self._reraise()

    def close(self) -> None:
        """Flush every pending write, patch headers, and stop the writer."""

        if self._closed:
            return
        if self._queue is not None:
            self._queue.put(None)
            if self._thread is not None:
                self._thread.join()
            self._thread = None
            self._queue = None
        self._closed = True
        for item in self._files.values():
            item.close()
        self._reraise()

    def __enter__(self) -> "StreamingMatrixStore":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def load(self, family: str, *, mmap: bool = True) -> np.ndarray:
        """Read a finalized family back, memory-mapped by default."""

        if not self._closed:
            raise RuntimeError("step-matrix store must be closed before loading")
        return np.load(
            self._files[str(family)].path,
            mmap_mode="r" if mmap else None,
        )


# A fixed zip timestamp. ``np.savez`` stamps each member with
# ``time.localtime()``, so its output is NOT byte-reproducible across a second
# boundary -- two calls can differ while the arrays are identical. Since the
# sidecar's sha256 is bound into the receipt, a container whose bytes depend on
# the wall clock makes that hash describe the moment of writing as much as the
# content. Pinning the stamp makes the digest a function of the DATA alone.
_FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

_HASH_CHUNK_BYTES = 8 << 20


def write_npz_streaming(
    path: str,
    *,
    arrays: Mapping[str, np.ndarray],
    member_files: Mapping[str, str],
) -> str:
    """Write an ``.npz`` without materializing the streamed members.

    ``np.savez`` requires every array in memory, and then only via a BytesIO
    whose ``getvalue()`` copies the whole archive again -- together the
    dominant term in a ~19.9 GiB finalize peak. An ``.npz`` is a zip of
    ``.npy`` members, so the per-family files already on disk are streamed in
    by the zip machinery while the small arrays are serialized normally.

    Members are written in ONE globally sorted order across both sources, so
    the archive does not depend on which members happened to be streamed.
    """

    import hashlib
    import zipfile

    class _HashingNonSeekableWriter:
        """Observe every final archive byte exactly once while writing."""

        def __init__(self, raw: Any) -> None:
            self.raw = raw
            self.digest = hashlib.sha256()
            self.position = 0

        def write(self, payload: Any) -> int:
            written = int(self.raw.write(payload))
            self.digest.update(memoryview(payload)[:written])
            self.position += written
            return written

        def tell(self) -> int:
            return self.position

        def seek(self, *_args: Any, **_kwargs: Any) -> int:
            raise OSError("step-matrix hash stream is intentionally non-seekable")

        def seekable(self) -> bool:
            return False

        def writable(self) -> bool:
            return True

        def flush(self) -> None:
            self.raw.flush()

    overlap = set(arrays) & set(member_files)
    if overlap:
        raise ValueError(
            f"step-matrix members are both in-memory and streamed: {sorted(overlap)}"
        )
    with open(path, "xb") as raw:
        stream = _HashingNonSeekableWriter(raw)
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted({*arrays, *member_files}):
                info = zipfile.ZipInfo(f"{name}.npy", date_time=_FIXED_ZIP_TIMESTAMP)
                if name in arrays:
                    with archive.open(info, "w", force_zip64=True) as handle:
                        np.lib.format.write_array(
                            handle, np.asarray(arrays[name]), allow_pickle=False
                        )
                    continue
                source = member_files[name]
                info.file_size = os.path.getsize(source)
                with open(source, "rb") as reader:
                    with archive.open(info, "w", force_zip64=True) as handle:
                        while True:
                            chunk = reader.read(_HASH_CHUNK_BYTES)
                            if not chunk:
                                break
                            handle.write(chunk)
        stream.flush()
        os.fsync(raw.fileno())
        return stream.digest.hexdigest()


def sha256_file(path: str) -> str:
    """Digest a file without reading it into memory."""

    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "StreamingMatrixStore",
    "sha256_file",
    "write_npz_streaming",
]
