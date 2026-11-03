"""Per-row digest table and a verified row reader for the certified uint8 bank.

Copying the whole 193 GB optimization bank to node-local disk in every
category, hashing it inline, and then serving rows by page faults through an
``mmap`` of the replica moves far more than a category consumes: only the rows
its sealed designs reach (about 70-190 GB of unique rows).

This module moves the integrity authority from "one whole-file SHA-256 at copy
time" to "one SHA-256 per row, checked every time the row is read":

* :func:`build_row_digest_table` reads the certified ``inputs.npy`` once (a
  campaign-level job, not per category), recomputes the WHOLE-FILE SHA-256 in
  file order and refuses unless it equals the certified digest, and records
  the SHA-256 of every row's bytes.  The table (N x 32 bytes, 41 MB for
  ImageNet train) and its receipt are therefore bound to exactly the certified
  bytes.
* :class:`VerifiedRowBank` is a read-only, array-like view of the certified
  file.  ``bank[ordinals]`` issues positional reads of exactly those rows
  (coalescing consecutive ordinals), verifies each row against the table, and
  returns an owned ``ndarray`` in the requested order.  A mismatch raises
  :class:`BankRowIntegrityError` before the bytes can reach any consumer.

No row, order, batch shape or value changes: the reader returns the bytes the
``mmap`` returned, for the rows the data plane asks for, when it asks.  The
data plane's own CLASS3 delivery digests still run downstream of it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROW_DIGEST_TABLE_SCHEMA = "irg_uint8_bank_row_digest_table_v1"
ROW_DIGEST_TABLE_FILENAME = "row_sha256.npy"
ROW_DIGEST_RECEIPT_FILENAME = "row_sha256.json"
_DIGEST_BYTES = 32


class BankRowIntegrityError(ValueError):
    """A row read from the certified bank differs from its recorded digest."""


def _npy_layout(path: Path) -> tuple[int, tuple[int, ...], np.dtype]:
    """(data offset, shape, dtype) of a C-ordered ``.npy`` file."""

    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported npy version {version!r}")
        offset = handle.tell()
    if fortran or dtype.hasobject or len(shape) < 1:
        raise ValueError("row digests require a C-ordered, non-object array")
    return int(offset), tuple(int(value) for value in shape), np.dtype(dtype)


def _row_bytes(shape: tuple[int, ...], dtype: np.dtype) -> int:
    return int(np.prod(shape[1:], dtype=np.int64)) * int(dtype.itemsize)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_row_digest_table(
    inputs_path: str | Path,
    *,
    expected_inputs_sha256: str,
    output_dir: str | Path,
    workers: int = 8,
    rows_per_block: int = 512,
) -> dict[str, Any]:
    """Hash the certified bank once: whole-file digest plus one per row.

    Refuses (and writes nothing) unless the recomputed whole-file SHA-256
    equals ``expected_inputs_sha256``.  Returns the written receipt.
    """

    source = Path(inputs_path)
    offset, shape, dtype = _npy_layout(source)
    row_bytes = _row_bytes(shape, dtype)
    row_count = int(shape[0])
    size = int(source.stat().st_size)
    if offset + row_count * row_bytes != size:
        raise ValueError("bank file size differs from its npy header")
    started = time.perf_counter()
    whole = hashlib.sha256()
    table = np.empty((row_count, _DIGEST_BYTES), dtype=np.uint8)
    block_rows = max(1, int(rows_per_block))
    descriptor = os.open(str(source), os.O_RDONLY)
    try:
        header = os.pread(descriptor, offset, 0)
        if len(header) != offset:
            raise OSError("short read of the bank header")
        whole.update(header)

        def read_block(first: int) -> tuple[int, bytearray]:
            count = min(block_rows, row_count - first)
            buffer = bytearray(count * row_bytes)
            view = memoryview(buffer)
            done = 0
            while done < len(buffer):
                got = os.preadv(descriptor, [view[done:]], offset + first * row_bytes + done)
                if got == 0:
                    raise OSError("short read while building row digests")
                done += got
            view = memoryview(buffer)
            for index in range(count):
                table[first + index] = np.frombuffer(
                    hashlib.sha256(
                        view[index * row_bytes : (index + 1) * row_bytes]
                    ).digest(),
                    dtype=np.uint8,
                )
            return first, buffer

        with ThreadPoolExecutor(
            max_workers=max(1, int(workers)), thread_name_prefix="bank-row-digests"
        ) as executor:
            firsts = list(range(0, row_count, block_rows))
            window = max(2, 2 * max(1, int(workers)))
            pending = [executor.submit(read_block, first) for first in firsts[:window]]
            submitted = len(pending)
            for index in range(len(firsts)):
                first, buffer = pending[index].result()
                if first != firsts[index]:  # pragma: no cover - invariant
                    raise RuntimeError("row digest blocks consumed out of order")
                whole.update(buffer)
                pending[index] = None  # type: ignore[call-overload]
                if submitted < len(firsts):
                    pending.append(executor.submit(read_block, firsts[submitted]))
                    submitted += 1
        if os.pread(descriptor, 1, size):
            raise OSError("bank file grew while row digests were built")
    finally:
        os.close(descriptor)
    inputs_sha256 = whole.hexdigest()
    if inputs_sha256 != str(expected_inputs_sha256):
        raise ValueError(
            "certified bank whole-file digest differs; row digests not written"
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    table_path = destination / ROW_DIGEST_TABLE_FILENAME
    partial = destination / f".{ROW_DIGEST_TABLE_FILENAME}.partial"
    with partial.open("wb") as handle:
        np.save(handle, table, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, table_path)
    table_sha256 = hashlib.sha256(table_path.read_bytes()).hexdigest()
    receipt: dict[str, Any] = {
        "schema": ROW_DIGEST_TABLE_SCHEMA,
        "digest_algorithm": "sha256_of_row_bytes",
        "inputs_sha256": inputs_sha256,
        "inputs_size_bytes": size,
        "data_offset": offset,
        "shape": list(shape),
        "dtype": dtype.str,
        "row_bytes": row_bytes,
        "row_count": row_count,
        "table_filename": ROW_DIGEST_TABLE_FILENAME,
        "table_sha256": table_sha256,
        "build_seconds": float(time.perf_counter() - started),
    }
    receipt["content_sha256"] = _canonical_sha256(receipt)
    receipt_path = destination / ROW_DIGEST_RECEIPT_FILENAME
    partial_receipt = destination / f".{ROW_DIGEST_RECEIPT_FILENAME}.partial"
    partial_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(partial_receipt, receipt_path)
    return receipt


def load_row_digest_table(
    table_dir: str | Path,
    *,
    inputs_path: str | Path,
    inputs_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a table bound to ``inputs_sha256`` and the file's npy layout."""

    directory = Path(table_dir)
    receipt = json.loads((directory / ROW_DIGEST_RECEIPT_FILENAME).read_text())
    claimed = receipt.pop("content_sha256", None)
    if claimed != _canonical_sha256(receipt):
        raise ValueError("row digest receipt content hash mismatch")
    receipt["content_sha256"] = claimed
    offset, shape, dtype = _npy_layout(Path(inputs_path))
    size = int(Path(inputs_path).stat().st_size)
    expected = {
        "schema": ROW_DIGEST_TABLE_SCHEMA,
        "digest_algorithm": "sha256_of_row_bytes",
        "inputs_sha256": str(inputs_sha256),
        "inputs_size_bytes": size,
        "data_offset": offset,
        "shape": list(shape),
        "dtype": dtype.str,
        "row_bytes": _row_bytes(shape, dtype),
        "row_count": int(shape[0]),
        "table_filename": ROW_DIGEST_TABLE_FILENAME,
    }
    mismatched = sorted(name for name, value in expected.items() if receipt.get(name) != value)
    if mismatched:
        raise ValueError("row digest receipt does not bind this bank: " + ",".join(mismatched))
    table_bytes = (directory / ROW_DIGEST_TABLE_FILENAME).read_bytes()
    if hashlib.sha256(table_bytes).hexdigest() != receipt.get("table_sha256"):
        raise ValueError("row digest table bytes differ from their receipt")
    # Parsed from the very bytes just hashed (one read, no second open).
    table = np.load(io.BytesIO(table_bytes), allow_pickle=False)
    if table.shape != (int(shape[0]), _DIGEST_BYTES) or table.dtype != np.uint8:
        raise ValueError("row digest table has the wrong shape")
    table.setflags(write=False)
    return table, receipt


class VerifiedRowBank:
    """Array-like, read-only view of a certified ``.npy`` bank, row-verified.

    Supports exactly what the bank adapters use: ``shape``/``dtype``/``ndim``/
    ``len`` and ``__getitem__`` with an int, a slice, or an integer index
    array (any order, duplicates allowed).  Every returned row is an owned
    copy whose SHA-256 has just been checked against the table.
    """

    def __init__(
        self,
        inputs_path: str | Path,
        *,
        digests: np.ndarray,
        receipt: Mapping[str, Any],
    ) -> None:
        self.path = Path(inputs_path)
        offset, shape, dtype = _npy_layout(self.path)
        if (
            int(receipt["data_offset"]) != offset
            or list(receipt["shape"]) != list(shape)
            or str(receipt["dtype"]) != dtype.str
        ):
            raise ValueError("verified row bank layout differs from its digest receipt")
        self._offset = offset
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)
        self._row_shape = shape[1:]
        self._row_bytes = _row_bytes(shape, dtype)
        self._digests = digests
        self._descriptor = os.open(str(self.path), os.O_RDONLY)
        self.receipt = dict(receipt)
        self.rows_read = 0
        self.bytes_read = 0
        self.read_calls = 0

    def __len__(self) -> int:
        return int(self.shape[0])

    def __array__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        # Fail closed: converting the reader itself would read every row of
        # a 193 GB bank.  Consumers index it (``bank[ordinals]``).
        raise TypeError("a verified row bank is read by index, never whole")

    def reshape(self, *args: Any, **kwargs: Any) -> Any:
        # The data plane's working-set page prefetch (off by default) needs an
        # mmap view; with row streaming there are no pages to warm.
        raise TypeError(
            "a verified row bank has no mmap view; working_set_prefetch and "
            "page warming are unavailable under row streaming"
        )

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def itemsize(self) -> int:
        return int(self.dtype.itemsize)

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    def close(self) -> None:
        descriptor, self._descriptor = self._descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown
        try:
            self.close()
        except Exception:
            pass

    def _ordinals(self, index: Any) -> tuple[np.ndarray, bool]:
        if isinstance(index, (int, np.integer)) and not isinstance(index, bool):
            ordinal = int(index)
            if ordinal < 0:
                ordinal += len(self)
            if not 0 <= ordinal < len(self):
                raise IndexError("verified row bank index out of range")
            return np.asarray([ordinal], dtype=np.int64), True
        if isinstance(index, slice):
            return np.arange(*index.indices(len(self)), dtype=np.int64), False
        ordinals = np.asarray(index)
        if ordinals.dtype == np.bool_ or ordinals.ndim != 1 or (
            ordinals.size and ordinals.dtype.kind not in "iu"
        ):
            raise TypeError("verified row bank supports int, slice or 1-D int indices")
        ordinals = ordinals.astype(np.int64, copy=True)
        ordinals[ordinals < 0] += len(self)
        if ordinals.size and (ordinals.min() < 0 or ordinals.max() >= len(self)):
            raise IndexError("verified row bank index out of range")
        return ordinals, False

    def __getitem__(self, index: Any) -> np.ndarray:
        ordinals, scalar = self._ordinals(index)
        out = np.empty((int(ordinals.size), *self._row_shape), dtype=self.dtype)
        flat = out.reshape(int(ordinals.size), self._row_bytes // self.itemsize)
        raw = memoryview(flat.view(np.uint8).reshape(-1))
        row_bytes = self._row_bytes
        position = 0
        count = int(ordinals.size)
        while position < count:
            # Coalesce a run of consecutive ordinals into one positional read.
            stop = position + 1
            while stop < count and ordinals[stop] == ordinals[stop - 1] + 1:
                stop += 1
            first = int(ordinals[position])
            view = raw[position * row_bytes : stop * row_bytes]
            done = 0
            while done < len(view):
                got = os.preadv(
                    self._descriptor, [view[done:]], self._offset + first * row_bytes + done
                )
                if got == 0:
                    raise BankRowIntegrityError("short read from the certified bank")
                done += got
            self.read_calls += 1
            position = stop
        for row in range(count):
            digest = hashlib.sha256(raw[row * row_bytes : (row + 1) * row_bytes]).digest()
            if digest != self._digests[int(ordinals[row])].tobytes():
                raise BankRowIntegrityError(
                    f"certified bank row {int(ordinals[row])} differs from its digest"
                )
        self.rows_read += count
        self.bytes_read += count * row_bytes
        return out[0] if scalar else out

    def counters(self) -> dict[str, int]:
        return {
            "rows_read": int(self.rows_read),
            "bytes_read": int(self.bytes_read),
            "read_calls": int(self.read_calls),
        }


def resolve_row_streaming_request(
    env: Mapping[str, str] | None = None,
) -> str:
    """``off`` (default), ``on`` (fall back to staging if unusable) or ``required``."""

    source = os.environ if env is None else env
    raw = str(source.get("IRG_BANK_ROW_STREAMING", "")).strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return "off"
    if raw in {"1", "true", "yes", "on"}:
        return "on"
    if raw == "required":
        return "required"
    raise ValueError(f"IRG_BANK_ROW_STREAMING must be off, on or required, got {raw!r}")


def row_digest_table_dir(
    inputs_path: str | Path,
    env: Mapping[str, str] | None = None,
) -> Path:
    source = os.environ if env is None else env
    override = str(source.get("IRG_BANK_ROW_DIGEST_DIR", "")).strip()
    return Path(override) if override else Path(inputs_path).parent / "row_sha256"


__all__ = [
    "BankRowIntegrityError",
    "ROW_DIGEST_RECEIPT_FILENAME",
    "ROW_DIGEST_TABLE_FILENAME",
    "ROW_DIGEST_TABLE_SCHEMA",
    "VerifiedRowBank",
    "build_row_digest_table",
    "load_row_digest_table",
    "resolve_row_streaming_request",
    "row_digest_table_dir",
]


if __name__ == "__main__":  # pragma: no cover - campaign-level builder CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--expected-inputs-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    print(
        json.dumps(
            build_row_digest_table(
                arguments.inputs,
                expected_inputs_sha256=arguments.expected_inputs_sha256,
                output_dir=arguments.output_dir,
                workers=arguments.workers,
            ),
            indent=2,
            sort_keys=True,
        )
    )
