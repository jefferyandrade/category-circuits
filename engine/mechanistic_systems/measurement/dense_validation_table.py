"""A sealed per-model dense validation table.

The dense mask's full-width float32 readouts on the validation rows are the
same for every category of one model: the model, the dense mask, the readout
projection and the registered numerics are shared, and the terminal
validation rows are selected per architecture class, not per query.  So one
shared table per model is allowed, hashed and sealed, read ONLY by the
post-lock validation leg and never by the search.

* ``build_dense_validation_table`` writes the rows once (an uncompressed NPZ)
  plus a sealed receipt: the payload's sha256, the identity it was built
  under, its row set and its label digest, and a seal over all of that.  A
  table is immutable; building over an existing one refuses.
* ``DenseValidationTable.open`` verifies the seal, the identity against the
  caller's expected identity, and the payload bytes (one verified read that
  hashes exactly the bytes the arrays come from).
* ``DenseValidationTable.rows_for`` is the ONLY read path.  It asserts the
  role (``POST_LOCK_VALIDATION_ROLE``) and that category validation is open on
  the reading provider, which happens only after the mask lock; the search
  runs before that point, so a search-time read fails closed.  Every read is
  logged in ``READ_LOG``.  A request the table does not cover returns
  ``None`` and the caller runs its own forward.

Precision: the rows are the registered float32 readouts (TF32 profile as
registered); nothing here computes on them.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mechanistic_systems.measurement.verified_once import read_verified_npz

TABLE_SCHEMA = "irg_v6_dense_validation_table_v1"
RECEIPT_SCHEMA = "irg_v6_dense_validation_table_receipt_v1"
IDENTITY_SCHEMA = "irg_v6_dense_validation_table_identity_v1"
POST_LOCK_VALIDATION_ROLE = "post_lock_terminal_validation"
TABLE_ENV = "IRG_V6_DENSE_VALIDATION_TABLE"
BUILD_ENV = "IRG_V6_DENSE_VALIDATION_TABLE_BUILD"
PAYLOAD_NAME = "dense_validation_table.npz"
RECEIPT_NAME = "dense_validation_table.receipt.json"
LOCK_NAME = ".dense_validation_table.lock"

# (role, rows) of every read in this process: tests and the receipt use it to
# show that only the post-lock validation leg ever read the table.
READ_LOG: list[dict[str, Any]] = []
_READ_LOG_LOCK = threading.Lock()


class DenseValidationTableError(RuntimeError):
    """The table is missing, altered, or does not match its identity."""


class DenseValidationRoleError(DenseValidationTableError):
    """A read outside the post-lock validation leg."""


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ids_sha256(context_ids: Sequence[str]) -> str:
    return _sha256(_canonical({"context_ids": [str(value) for value in context_ids]}))


def _array_sha256(array: np.ndarray) -> str:
    return _sha256(np.ascontiguousarray(array).tobytes())


@dataclass(frozen=True)
class DenseValidationTableIdentity:
    """Everything the dense validation readouts depend on."""

    model_state_hash: str
    dense_mask_hash: str
    full_readout_projection_hash: str
    readout_numerics: str
    runtime_execution_profile_hash: str
    validation_axis_hash: str
    readout_width: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": IDENTITY_SCHEMA,
            "model_state_hash": str(self.model_state_hash),
            "dense_mask_hash": str(self.dense_mask_hash),
            "full_readout_projection_hash": str(self.full_readout_projection_hash),
            "readout_numerics": str(self.readout_numerics),
            "runtime_execution_profile_hash": str(self.runtime_execution_profile_hash),
            "validation_axis_hash": str(self.validation_axis_hash),
            "readout_width": int(self.readout_width),
        }

    @property
    def content_hash(self) -> str:
        return _sha256(_canonical(self.to_dict()))


def _receipt_seal(body: Mapping[str, Any]) -> str:
    return _sha256(_canonical({key: body[key] for key in sorted(body) if key != "seal"}))


def build_dense_validation_table(
    directory: str | os.PathLike[str],
    *,
    identity: DenseValidationTableIdentity,
    context_ids: Sequence[str],
    rows: np.ndarray,
    labels: np.ndarray,
    builder: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write and seal a table; refuse if one exists.  Returns the receipt."""

    root = Path(directory)
    ordered = tuple(str(value) for value in context_ids)
    values = np.ascontiguousarray(np.asarray(rows))
    label_values = np.ascontiguousarray(np.asarray(labels, dtype=np.int64).reshape(-1))
    if (
        not ordered
        or len(set(ordered)) != len(ordered)
        or values.dtype != np.float32
        or values.shape != (len(ordered), int(identity.readout_width))
        or label_values.shape != (len(ordered),)
        or not bool(np.isfinite(values).all())
    ):
        raise DenseValidationTableError("dense validation table rows are invalid")
    root.mkdir(parents=True, exist_ok=True)
    payload_path = root / PAYLOAD_NAME
    receipt_path = root / RECEIPT_NAME
    with open(root / LOCK_NAME, "a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            if payload_path.exists() or receipt_path.exists():
                raise FileExistsError(f"a sealed dense validation table exists at {root}")
            partial = root / f".{PAYLOAD_NAME}.{os.getpid()}.partial"
            with open(partial, "wb") as handle:
                np.savez(
                    handle,
                    context_ids=np.asarray(ordered, dtype=np.str_),
                    rows=values,
                    labels=label_values,
                )
                handle.flush()
                os.fsync(handle.fileno())
            payload_sha256 = hashlib.sha256(partial.read_bytes()).hexdigest()
            body: dict[str, Any] = {
                "schema": RECEIPT_SCHEMA,
                "table_schema": TABLE_SCHEMA,
                "identity": identity.to_dict(),
                "identity_hash": identity.content_hash,
                "payload_file": PAYLOAD_NAME,
                "payload_sha256": payload_sha256,
                "payload_bytes": int(partial.stat().st_size),
                "row_count": len(ordered),
                "ordered_context_ids_sha256": _ids_sha256(ordered),
                "labels_sha256": _array_sha256(label_values),
                "rows_sha256": _array_sha256(values),
                "readers": [POST_LOCK_VALIDATION_ROLE],
                "builder": dict(builder or {}),
            }
            body["seal"] = _receipt_seal(body)
            os.replace(partial, payload_path)
            receipt_partial = root / f".{RECEIPT_NAME}.{os.getpid()}.partial"
            receipt_partial.write_text(
                json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(receipt_partial, receipt_path)
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    return body


class DenseValidationTable:
    """A verified, sealed table.  Read only through ``rows_for``."""

    def __init__(
        self,
        *,
        directory: Path,
        receipt: Mapping[str, Any],
        context_ids: tuple[str, ...],
        rows: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.directory = directory
        self.receipt = dict(receipt)
        self._context_ids = context_ids
        self._positions = {value: index for index, value in enumerate(context_ids)}
        self._rows = rows
        self._labels = labels
        self.reads = 0

    @classmethod
    def open(
        cls,
        directory: str | os.PathLike[str],
        *,
        expected_identity: DenseValidationTableIdentity,
    ) -> "DenseValidationTable":
        root = Path(directory)
        try:
            receipt = json.loads((root / RECEIPT_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise DenseValidationTableError(
                f"dense validation table receipt is unreadable at {root}"
            ) from error
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != RECEIPT_SCHEMA
            or receipt.get("seal") != _receipt_seal(receipt)
        ):
            raise DenseValidationTableError("dense validation table seal is broken")
        if (
            receipt.get("identity") != expected_identity.to_dict()
            or receipt.get("identity_hash") != expected_identity.content_hash
        ):
            raise DenseValidationTableError(
                "dense validation table was built for another model, mask, "
                "projection, numerics profile or validation axis"
            )
        if receipt.get("payload_file") != PAYLOAD_NAME:
            raise DenseValidationTableError("dense validation table payload is misnamed")
        payload = read_verified_npz(root / PAYLOAD_NAME)
        if payload.sha256 != receipt.get("payload_sha256"):
            raise DenseValidationTableError(
                "dense validation table payload differs from its sealed sha256"
            )
        try:
            context_ids = tuple(str(value) for value in payload["context_ids"].tolist())
            rows = np.asarray(payload["rows"])
            labels = np.asarray(payload["labels"])
        except KeyError as error:
            raise DenseValidationTableError("dense validation table members are missing") from error
        if (
            len(context_ids) != int(receipt.get("row_count", -1))
            or len(set(context_ids)) != len(context_ids)
            or _ids_sha256(context_ids) != receipt.get("ordered_context_ids_sha256")
            or rows.dtype != np.float32
            or rows.shape != (len(context_ids), int(expected_identity.readout_width))
            or labels.dtype != np.int64
            or labels.shape != (len(context_ids),)
            or _array_sha256(labels) != receipt.get("labels_sha256")
            or _array_sha256(rows) != receipt.get("rows_sha256")
            or not bool(np.isfinite(rows).all())
        ):
            raise DenseValidationTableError("dense validation table contents are invalid")
        rows.setflags(write=False)
        labels.setflags(write=False)
        return cls(
            directory=root,
            receipt=receipt,
            context_ids=context_ids,
            rows=rows,
            labels=labels,
        )

    def rows_for(
        self,
        ordered_context_ids: Sequence[str],
        *,
        role: str,
        validation_is_open: bool,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """``(rows, labels)`` in the requested order, or ``None`` if uncovered.

        Role assertion: only the post-lock terminal validation leg may
        read, and only once category validation is open on the reader.
        """

        if str(role) != POST_LOCK_VALIDATION_ROLE:
            raise DenseValidationRoleError(
                f"the dense validation table is read only by the post-lock "
                f"validation leg, not by {role!r}"
            )
        if not bool(validation_is_open):
            raise DenseValidationRoleError(
                "the dense validation table is read only after the mask lock "
                "(category validation is not open)"
            )
        try:
            index = np.fromiter(
                (self._positions[str(value)] for value in ordered_context_ids),
                dtype=np.int64,
            )
        except KeyError:
            return None
        if index.size == 0 or np.unique(index).size != index.size:
            return None
        self.reads += 1
        with _READ_LOG_LOCK:
            READ_LOG.append(
                {
                    "role": str(role),
                    "rows": int(index.size),
                    "identity_hash": str(self.receipt.get("identity_hash")),
                }
            )
        return self._rows[index], self._labels[index]

    def receipt_block(self) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "identity_hash": self.receipt.get("identity_hash"),
            "payload_sha256": self.receipt.get("payload_sha256"),
            "seal": self.receipt.get("seal"),
            "row_count": self.receipt.get("row_count"),
            "reads": int(self.reads),
            "reader_role": POST_LOCK_VALIDATION_ROLE,
        }


def configured_table_directory(env: Mapping[str, str] | None = None) -> Path | None:
    source = os.environ if env is None else env
    raw = str(source.get(TABLE_ENV, "")).strip()
    return Path(raw) if raw else None


def build_requested(env: Mapping[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    raw = str(source.get(BUILD_ENV, "0")).strip().lower()
    if raw not in {"0", "1", "false", "true", "no", "yes", "off", "on", ""}:
        raise ValueError(f"{BUILD_ENV} must be a boolean token")
    return raw in {"1", "true", "yes", "on"}


__all__ = [
    "BUILD_ENV",
    "DenseValidationRoleError",
    "DenseValidationTable",
    "DenseValidationTableError",
    "DenseValidationTableIdentity",
    "PAYLOAD_NAME",
    "POST_LOCK_VALIDATION_ROLE",
    "READ_LOG",
    "RECEIPT_NAME",
    "TABLE_ENV",
    "build_dense_validation_table",
    "build_requested",
    "configured_table_directory",
]
