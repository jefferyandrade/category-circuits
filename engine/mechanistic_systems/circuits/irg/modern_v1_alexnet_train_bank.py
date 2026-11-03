"""Path-free identities for the production AlexNet optimization bank.

The physical ImageNet-scale bank is deliberately staged into node-local
scratch at execution time.  Its scratch path, hostname, and copy timing are
therefore not scientific coordinates.  This module owns the small immutable
identity calculation shared by campaign preparation and runtime admission.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .contracts import content_hash


_SOURCE_IDENTITY_SCHEMA = "certified_uint8_bank_payload_sha256_v1"
_TRAIN_BANK_SOURCE_SCHEMA = "partition_free_v1_alexnet_train_bank_source_v1"
_TRAIN_BANK_SOURCE_PREFIX = "partition-free-v1-alexnet-train-bank-source-"


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _identity(value, name=name)
    if (
        len(result) != 64
        or result != result.lower()
        or any(character not in "0123456789abcdef" for character in result)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return result


def _positive_size(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return int(value)


def certified_uint8_bank_payload_identity_hash(
    *,
    inputs_sha256: str,
    labels_sha256: str,
    inputs_size_bytes: int,
    labels_size_bytes: int,
) -> str:
    """Reproduce the node-local staging replica key without a physical path."""

    payload: dict[str, Any] = {
        "schema_name": _SOURCE_IDENTITY_SCHEMA,
        "inputs_sha256": _sha256(inputs_sha256, name="inputs_sha256"),
        "labels_sha256": _sha256(labels_sha256, name="labels_sha256"),
        "inputs_size_bytes": _positive_size(
            inputs_size_bytes,
            name="inputs_size_bytes",
        ),
        "labels_size_bytes": _positive_size(
            labels_size_bytes,
            name="labels_size_bytes",
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def alexnet_train_bank_source_hash(
    *,
    population_hash: str,
    source_context_axis_hash: str,
    source_label_map_hash: str,
    replica_receipt_hash: str,
    source_identity_hash: str,
    replica_key: str | None = None,
) -> str:
    """Bind logical population and certified bytes, excluding scratch location."""

    source_identity = _sha256(
        source_identity_hash,
        name="source_identity_hash",
    )
    replica = source_identity if replica_key is None else _sha256(
        replica_key,
        name="replica_key",
    )
    if replica != source_identity:
        raise ValueError("replica_key must equal the certified source identity")
    return content_hash(
        {
            "schema": _TRAIN_BANK_SOURCE_SCHEMA,
            "population_hash": _identity(population_hash, name="population_hash"),
            "source_context_axis_hash": _identity(
                source_context_axis_hash,
                name="source_context_axis_hash",
            ),
            "source_label_map_hash": _identity(
                source_label_map_hash,
                name="source_label_map_hash",
            ),
            "replica_receipt_hash": _identity(
                replica_receipt_hash,
                name="replica_receipt_hash",
            ),
            "source_identity_hash": source_identity,
            "replica_key": replica,
            "node_local_staging_required": True,
            "physical_scratch_path_bound": False,
        },
        prefix=_TRAIN_BANK_SOURCE_PREFIX,
    )


__all__ = [
    "alexnet_train_bank_source_hash",
    "certified_uint8_bank_payload_identity_hash",
]
