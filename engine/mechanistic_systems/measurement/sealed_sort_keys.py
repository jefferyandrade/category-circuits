"""Content-seeded sort keys for calibration and terminal context orders.

The fixed-schema JSON is encoded directly before SHA-256 hashing. Context
and namespace identities define the scientific sampling order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from json.encoder import encode_basestring_ascii as _json_string

SORT_PERMUTATION_ALGORITHM = "sha256_sort_permutation_v1"
CLASS_PREFIX_ALGORITHM = "sha256_class_prefix_v2"


def _require_exact_str(values: Sequence[str]) -> None:
    for value in values:
        if type(value) is not str:
            raise TypeError("sealed sort keys require exact str context ids")


def sealed_order_sort_keys(
    context_ids: Sequence[str],
    *,
    namespace: str,
) -> list[str]:
    """``content_hash({"algorithm", "namespace", "context_id"})`` per row."""

    _require_exact_str(context_ids)
    head = f'{{"algorithm":"{SORT_PERMUTATION_ALGORITHM}","context_id":'
    tail = f',"namespace":{_json_string(str(namespace))}}}'
    sha256 = hashlib.sha256
    keys = [
        sha256((head + _json_string(context_id) + tail).encode("utf-8")).hexdigest()
        for context_id in context_ids
    ]
    return keys


def class_prefix_sort_keys(
    context_ids: Sequence[str],
    *,
    namespace: str,
    class_id: int,
) -> list[str]:
    """``content_hash({"algorithm", "namespace", "class_id", "context_id"})``."""

    _require_exact_str(context_ids)
    if type(class_id) is not int:
        raise TypeError("class-prefix sort keys require an exact int class id")
    head = (
        f'{{"algorithm":"{CLASS_PREFIX_ALGORITHM}","class_id":{int(class_id)!r},'
        '"context_id":'
    )
    tail = f',"namespace":{_json_string(str(namespace))}}}'
    sha256 = hashlib.sha256
    keys = [
        sha256((head + _json_string(context_id) + tail).encode("utf-8")).hexdigest()
        for context_id in context_ids
    ]
    return keys


def sorted_by_keys(values: Sequence[str], keys: Sequence[str]) -> tuple[str, ...]:
    """``tuple(sorted(values, key=...))`` given the precomputed keys.

    Python's sort is stable, so ties keep input order; sorting the
    ``(key, index)`` pairs reproduces exactly that tie rule.
    """

    if len(values) != len(keys):
        raise ValueError("sort keys must align with their values")
    order = sorted(range(len(values)), key=keys.__getitem__)
    return tuple(values[index] for index in order)


__all__ = [
    "CLASS_PREFIX_ALGORITHM",
    "SORT_PERMUTATION_ALGORITHM",
    "class_prefix_sort_keys",
    "sealed_order_sort_keys",
    "sorted_by_keys",
]
