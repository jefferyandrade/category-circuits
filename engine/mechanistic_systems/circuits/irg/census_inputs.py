"""Per-run inputs of the importance census: target logits, labels, u_i, a_i.

The size measure needs the dense target-class logit of every training image.
A class-major copy of the network's canonical dense readout table
(``values_class_major.npy``, classes x images, float32) lets a run read one
row (about 5 MB) instead of the row-major 5 GB table.  The row is checked
against its registered sha256 (``row_sha256`` in the copy's row-hash file),
and the image ids against the source table's ids file, so a run never uses a
column it has not verified.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .census_pps import class_weights, size_measure

_ID_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
_ID_LOCK = threading.Lock()


def _sha256_file(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                return h.hexdigest()
            h.update(chunk)


@dataclass(frozen=True)
class ClassMajorLogits:
    """A registered class-major dense-logit copy."""

    directory: Path
    receipt: dict[str, Any]
    row_sha256: tuple[str, ...]
    row_hash_file_sha256: str

    @classmethod
    def open(cls, directory: str | Path, row_hash_file: str | Path) -> "ClassMajorLogits":
        root = Path(directory)
        receipt = json.loads((root / "receipt.json").read_text())
        payload = Path(row_hash_file).read_bytes()
        rows = json.loads(payload)
        if rows.get("values_class_major_sha256") != receipt.get("values_class_major_sha256"):
            raise ValueError("census logits row hashes belong to another class-major copy")
        hashes = tuple(str(value) for value in rows["row_sha256"])
        if len(hashes) != int(receipt["shape_class_major"][0]):
            raise ValueError("census logits row-hash count differs from the class count")
        return cls(root, receipt, hashes, hashlib.sha256(payload).hexdigest())

    def identity(self) -> dict[str, Any]:
        return {
            "schema": self.receipt.get("schema"),
            "values_class_major_sha256": self.receipt.get("values_class_major_sha256"),
            "row_hash_file_sha256": self.row_hash_file_sha256,
            "source_payload_sha256": self.receipt.get("source_payload_sha256"),
            "source_context_axis_hash": self.receipt.get("source_context_axis_hash"),
            "source_model_state_hash": self.receipt.get("source_model_state_hash"),
            "shape_class_major": list(self.receipt.get("shape_class_major", [])),
        }

    def ids_and_labels(self) -> tuple[np.ndarray, np.ndarray]:
        path = Path(self.receipt["ids_file"])
        expected = self.receipt["source_payload_sha256"][path.name]
        key = (str(path), str(expected))
        with _ID_LOCK:
            cached = _ID_CACHE.get(key)
            if cached is None:
                if _sha256_file(path) != expected:
                    raise ValueError("census logits ids file differs from its registered sha256")
                with np.load(path) as handle:
                    cached = (np.asarray(handle["source_context_ids"]), np.asarray(handle["labels"], dtype=np.int64))
                _ID_CACHE[key] = cached
        return cached

    def target_row(self, target_class_id: int) -> np.ndarray:
        values = np.load(self.directory / "values_class_major.npy", mmap_mode="r")
        row = np.array(values[int(target_class_id)], dtype=np.float32, copy=True)
        if hashlib.sha256(row.tobytes()).hexdigest() != self.row_sha256[int(target_class_id)]:
            raise ValueError("census logits row differs from its registered sha256")
        return row


def row_hash_document(directory: str | Path) -> dict[str, Any]:
    """Row sha256s of a class-major copy (written once, beside the copy)."""

    root = Path(directory)
    receipt = json.loads((root / "receipt.json").read_text())
    values = np.load(root / "values_class_major.npy", mmap_mode="r")
    return {
        "schema": "irg_dense_logits_class_major_row_sha256_v1",
        "values_class_major_sha256": receipt["values_class_major_sha256"],
        "row_sha256": [hashlib.sha256(np.ascontiguousarray(values[i]).tobytes()).hexdigest()
                       for i in range(values.shape[0])],
    }


@dataclass(frozen=True)
class ImportanceInputs:
    on_target_ids: tuple[str, ...]
    on_target_a: np.ndarray
    on_target_u: np.ndarray
    off_target_ids: tuple[str, ...]
    off_target_a: np.ndarray
    off_target_u: np.ndarray
    off_target_labels: np.ndarray
    receipt: dict[str, Any]


def importance_inputs(
    *,
    context_ids: Sequence[str],
    labels: Sequence[int],
    target_logits_by_table_row: np.ndarray,
    table_ids: np.ndarray,
    table_labels: np.ndarray,
    target_class_id: int,
    calibration: Any,
    fixed_offsets: np.ndarray | None = None,
) -> ImportanceInputs:
    """Align logits and form image masses and residual proposal sizes.

    ``fixed_offsets`` is already aligned to ``context_ids`` by the resident
    table resolver, rather than to the class-major table's storage order.
    """

    from mechanistic_systems.measurement.run_products import pruning_functional

    if pruning_functional() == "substituted_softmax" and fixed_offsets is None:
        raise ValueError("substituted census proposals require resident dense table offsets")

    ids = np.asarray([str(value) for value in context_ids])
    run_labels = np.asarray(labels, dtype=np.int64)
    if ids.size != run_labels.size:
        raise ValueError("census axis ids and labels differ in length")
    table_ids = np.asarray(table_ids)
    if table_ids.shape == ids.shape and np.array_equal(table_ids, ids):
        positions = np.arange(ids.size)
        alignment = "identical_order"
    else:
        index = {str(value): i for i, value in enumerate(table_ids.tolist())}
        try:
            positions = np.fromiter((index[value] for value in ids.tolist()), dtype=np.int64, count=ids.size)
        except KeyError as error:
            raise ValueError("a census axis image is missing from the dense logit table") from error
        alignment = "mapped_by_id"
    if not np.array_equal(np.asarray(table_labels, dtype=np.int64)[positions], run_labels):
        raise ValueError("census axis labels differ from the dense logit table's labels")
    z = np.asarray(target_logits_by_table_row, dtype=np.float64)[positions]
    target = int(target_class_id)
    u = size_measure(
        z, run_labels, target_class_id=target, scale=float(calibration.scale),
        offset=float(calibration.offset), input_center=float(calibration.input_center),
        **({"fixed_offsets": fixed_offsets} if fixed_offsets is not None else {}),
    )
    a = class_weights(run_labels, target_class_id=target)
    on = run_labels == target
    off = ~on
    if not np.all(u[off] > 0.0):
        raise ValueError("an off-target image has a zero census size measure")
    return ImportanceInputs(
        on_target_ids=tuple(ids[on].tolist()), on_target_a=a[on], on_target_u=u[on],
        off_target_ids=tuple(ids[off].tolist()), off_target_a=a[off], off_target_u=u[off],
        off_target_labels=run_labels[off],
        receipt={
            "alignment": alignment, "on_target_count": int(on.sum()), "off_target_count": int(off.sum()),
            "off_target_classes": int(np.unique(run_labels[off]).size),
            "u_off_target_sum": float(u[off].sum()), "x_off_target_sum": float((a[off] * u[off]).sum()),
            "calibration": {"scale": float(calibration.scale), "offset": float(calibration.offset),
                            "input_center": float(calibration.input_center)},
            **({"pruning_functional": "substituted_softmax",
                "size_measure": "absolute_substituted_probability_residual",
                "fixed_offset_source": "resident_dense_readout_table"}
               if fixed_offsets is not None else {}),
        },
    )


__all__ = ["ClassMajorLogits", "ImportanceInputs", "importance_inputs", "row_hash_document"]
