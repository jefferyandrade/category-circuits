"""Bounded identity cache for large ordered scientific string axes.

Whole-route ResNet axes contain more than a million stable string IDs.  The
axis must be validated and cryptographically bound, but rebuilding a Python
``set`` and serializing every ID for every sufficient-statistic object turns a
constant scientific identity into a repeated O(R) control-plane cost.

This module validates each in-process tuple identity once, retains a bounded
strong-reference cache so object IDs cannot be reused, and exposes a
length-prefixed ordered hash.  Compact packed layouts carry that hash and the
axis length; their consumer must supply the already-declared physical axis.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import operator
from threading import RLock, Thread
from typing import Any, Iterator, Mapping, Sequence, overload


_MAX_CACHED_AXES = 8
_MAX_CACHED_CONSTRUCTED_ROUTE_AXES = 1
_CACHE_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class _StringAxisIdentity:
    axis: Sequence[str]
    ordered_digest: str
    canonical_json_digest: str | None = None


_AXIS_CACHE: OrderedDict[int, _StringAxisIdentity] = OrderedDict()
_CONSTRUCTED_ROUTE_AXIS_CACHE: OrderedDict[int, _StringAxisIdentity] = (
    OrderedDict()
)
_COMPACT_ROUTE_AXIS_CACHE: OrderedDict[int, "ConstructiveOrdinalRouteAxis"] = (
    OrderedDict()
)


# The two digests of the constructive ``route:<ordinal>`` axis are
# computed from vectorized byte records instead of five hashlib calls per
# ordinal (the ~17 s per process attributed to ``_global_route_axes``).  The
# hashed byte streams are the historical ones exactly:
#   ordered:   b"mechanistic-systems-ordered-string-axis-v1\0" + count(8, BE)
#              + for each ordinal: len(value)(8, BE) + value
#   canonical: b"[" + b'"route:0"' + for each later ordinal b',"route:N"' + b"]"
_ORDINAL_DIGEST_CHUNK = 1 << 20
_ORDINAL_DIGEST_VECTOR_MIN = 4096
_ORDINAL_DIGEST_OVERLAP_MIN = 1 << 20
_ROUTE_PREFIX = b"route:"


def _ordinal_route_axis_digests_scalar(count: int) -> tuple[str, str]:
    ordered = hashlib.sha256(b"mechanistic-systems-ordered-string-axis-v1\0")
    ordered.update(count.to_bytes(8, "big", signed=False))
    canonical = hashlib.sha256(b"[")
    for ordinal in range(count):
        encoded = f"route:{ordinal}".encode("ascii")
        ordered.update(len(encoded).to_bytes(8, "big", signed=False))
        ordered.update(encoded)
        if ordinal:
            canonical.update(b",")
        canonical.update(b'"')
        canonical.update(encoded)
        canonical.update(b'"')
    canonical.update(b"]")
    return ordered.hexdigest(), canonical.hexdigest()


def _ordinal_route_axis_digests(count: int) -> tuple[str, str]:
    """(ordered digest, canonical-JSON digest) of ``route:0 .. route:count-1``."""

    if count < _ORDINAL_DIGEST_VECTOR_MIN:
        return _ordinal_route_axis_digests_scalar(count)
    try:
        import numpy  # noqa: F401
    except Exception:  # pragma: no cover - numpy is a hard dependency
        return _ordinal_route_axis_digests_scalar(count)
    if count < _ORDINAL_DIGEST_OVERLAP_MIN:
        return (
            _ordinal_route_axis_stream_digest(count, ordered=True),
            _ordinal_route_axis_stream_digest(count, ordered=False),
        )
    # The two streams are independent; hash them on two threads (hashlib
    # and numpy's loops release the GIL).
    results: dict[bool, str] = {}
    failure: list[BaseException] = []

    def run(kind: bool) -> None:
        try:
            results[kind] = _ordinal_route_axis_stream_digest(count, ordered=kind)
        except BaseException as error:  # pragma: no cover - re-raised below
            failure.append(error)

    helper = Thread(target=run, args=(False,), name="ordinal-axis-canonical", daemon=True)
    helper.start()
    try:
        run(True)
    finally:
        helper.join()
    if failure:
        raise failure[0]
    return results[True], results[False]


def _ordinal_route_axis_stream_digest(count: int, *, ordered: bool) -> str:
    """One of the two digests, from vectorized byte records.

    The decimal digits of each chunk of consecutive ordinals are assembled
    from a table of the 10,000 four-digit suffixes and the (slowly changing)
    higher digits, then framed exactly as the scalar loop frames them.
    """

    import numpy as np

    if ordered:
        hasher = hashlib.sha256(b"mechanistic-systems-ordered-string-axis-v1\0")
        hasher.update(count.to_bytes(8, "big", signed=False))
    else:
        hasher = hashlib.sha256(b"[")
    prefix = np.frombuffer(_ROUTE_PREFIX, dtype=np.uint8)
    quoted_prefix = np.frombuffer(b',"' + _ROUTE_PREFIX, dtype=np.uint8)
    suffix_table = (
        (np.arange(10000, dtype=np.int64)[:, None] // np.array([1000, 100, 10, 1]))
        % 10
        + 48
    ).astype(np.uint8)
    block = 10000
    blocks_per_chunk = max(1, _ORDINAL_DIGEST_CHUNK // block)
    first = True
    width = 1
    low = 0
    while low < count:
        high = min(count, 10**width)
        record = len(_ROUTE_PREFIX) + width
        length_bytes = np.frombuffer(
            record.to_bytes(8, "big", signed=False), dtype=np.uint8
        )
        # Chunks of whole 10,000-ordinal blocks (``low`` is a multiple of
        # 10,000 once width >= 5); the last chunk is trimmed to ``high``.
        if width <= 4:
            spans = [(low, high)]
        else:
            spans = [
                (start, min(high, start + blocks_per_chunk * block))
                for start in range(low, high, blocks_per_chunk * block)
            ]
        for start, stop in spans:
            rows = stop - start
            if width <= 4:
                digits = suffix_table[start:stop, 4 - width :]
            else:
                first_block = start // block
                block_count = -(-(stop - start) // block)
                heads = np.arange(first_block, first_block + block_count, dtype=np.int64)
                head_width = width - 4
                head_digits = (
                    (heads[:, None] // (10 ** np.arange(head_width - 1, -1, -1, dtype=np.int64)))
                    % 10
                    + 48
                ).astype(np.uint8)
                digits = np.empty((block_count * block, width), dtype=np.uint8)
                digits[:, :head_width] = np.repeat(head_digits, block, axis=0)
                digits[:, head_width:] = np.tile(suffix_table, (block_count, 1))
                digits = digits[:rows]
            if ordered:
                ordered_rows = np.empty((rows, 8 + record), dtype=np.uint8)
                ordered_rows[:, :8] = length_bytes
                ordered_rows[:, 8 : 8 + len(_ROUTE_PREFIX)] = prefix
                ordered_rows[:, 8 + len(_ROUTE_PREFIX) :] = digits
                hasher.update(ordered_rows.reshape(-1))
            else:
                canonical_rows = np.empty(
                    (rows, len(quoted_prefix) + width + 1), dtype=np.uint8
                )
                canonical_rows[:, : len(quoted_prefix)] = quoted_prefix
                canonical_rows[:, len(quoted_prefix) : -1] = digits
                canonical_rows[:, -1] = ord('"')
                stream = canonical_rows.reshape(-1)
                hasher.update(stream[1:] if first else stream)
            first = False
        low = high
        width += 1
    if not ordered:
        hasher.update(b"]")
    return hasher.hexdigest()


class ConstructiveOrdinalRouteAxis(Sequence[str]):
    """O(1)-storage ``route:<ordinal>`` axis with exact historical hashes.

    A production AlexNet kernel-tap axis contains 56,994,496 routes.  Its
    identifier grammar is constructive, so storing tens of millions of
    independent Python strings provides no additional validation.  This
    immutable sequence generates identifiers on indexed access and computes
    the same ordered and canonical-JSON digests as the historical tuple,
    without retaining the generated strings.
    """

    __slots__ = ("_count", "_ordered_digest", "_canonical_json_digest")

    def __init__(self, route_count: int) -> None:
        try:
            count = operator.index(route_count)
        except TypeError as exc:
            raise ValueError(
                "constructive route-axis count must be an integer"
            ) from exc
        if isinstance(route_count, bool) or count <= 0:
            raise ValueError("constructive route-axis count must be a positive integer")
        self._count = int(count)
        self._ordered_digest: str | None = None
        self._canonical_json_digest: str | None = None

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[str, ...]: ...

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            return tuple(f"route:{ordinal}" for ordinal in range(start, stop, step))
        ordinal = operator.index(index)
        if ordinal < 0:
            ordinal += self._count
        if ordinal < 0 or ordinal >= self._count:
            raise IndexError("constructive route-axis index out of range")
        return f"route:{ordinal}"

    def __iter__(self) -> Iterator[str]:
        return (f"route:{ordinal}" for ordinal in range(self._count))

    def __contains__(self, value: object) -> bool:
        if type(value) is not str or not value.startswith("route:"):
            return False
        raw = value[6:]
        if not raw or (len(raw) > 1 and raw[0] == "0") or not raw.isascii():
            return False
        try:
            ordinal = int(raw)
        except ValueError:  # pragma: no cover - guarded by decimal check below
            return False
        return raw.isdecimal() and 0 <= ordinal < self._count

    def _compute_digests(self) -> tuple[str, str]:
        ordered = self._ordered_digest
        canonical = self._canonical_json_digest
        if ordered is not None and canonical is not None:
            return ordered, canonical
        ordered, canonical = _ordinal_route_axis_digests(self._count)
        self._ordered_digest = ordered
        self._canonical_json_digest = canonical
        return ordered, canonical

    @property
    def ordered_digest(self) -> str:
        return self._compute_digests()[0]

    @property
    def canonical_json_digest(self) -> str:
        return self._compute_digests()[1]

    def __repr__(self) -> str:
        return f"ConstructiveOrdinalRouteAxis(route_count={self._count})"


def compact_ordinal_route_axis(route_count: int) -> ConstructiveOrdinalRouteAxis:
    """Return a bounded-memory constructive ordinal route axis."""

    try:
        count = operator.index(route_count)
    except TypeError as exc:
        raise ValueError("constructive route-axis count must be an integer") from exc
    if isinstance(route_count, bool) or count <= 0:
        raise ValueError("constructive route-axis count must be a positive integer")
    with _CACHE_LOCK:
        cached = _COMPACT_ROUTE_AXIS_CACHE.get(int(count))
        if cached is not None:
            _COMPACT_ROUTE_AXIS_CACHE.move_to_end(int(count))
            return cached
        axis = ConstructiveOrdinalRouteAxis(int(count))
        _COMPACT_ROUTE_AXIS_CACHE[int(count)] = axis
        _COMPACT_ROUTE_AXIS_CACHE.move_to_end(int(count))
        while len(_COMPACT_ROUTE_AXIS_CACHE) > _MAX_CACHED_CONSTRUCTED_ROUTE_AXES:
            _COMPACT_ROUTE_AXIS_CACHE.popitem(last=False)
        return axis


def _cache_axis_identity(identity: _StringAxisIdentity) -> None:
    axis = identity.axis
    _AXIS_CACHE[id(axis)] = identity
    _AXIS_CACHE.move_to_end(id(axis))
    while len(_AXIS_CACHE) > _MAX_CACHED_AXES:
        _AXIS_CACHE.popitem(last=False)


def _ordered_axis_digest(axis: tuple[str, ...]) -> str:
    digest = hashlib.sha256(b"mechanistic-systems-ordered-string-axis-v1\0")
    digest.update(len(axis).to_bytes(8, "big", signed=False))
    for value in axis:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _canonical_json_axis_digest(axis: tuple[str, ...]) -> str:
    """Hash the exact historical canonical JSON array without materializing it."""

    digest = hashlib.sha256()
    digest.update(b"[")
    for index, value in enumerate(axis):
        if index:
            digest.update(b",")
        digest.update(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
    digest.update(b"]")
    return digest.hexdigest()


def constructive_ordinal_route_axis(route_count: int) -> tuple[str, ...]:
    """Build and register the internally defined ``route:<ordinal>`` axis.

    Uniqueness follows constructively from injectivity of the decimal ordinal
    encoding over ``range(route_count)``.  The factory therefore need not
    allocate an exhaustive ``set``.  Both the current ordered-axis digest and
    the historical ``content_hash(tuple)`` digest are computed while the tuple
    is materialized, without a second list or complete JSON string.
    """

    try:
        count = operator.index(route_count)
    except TypeError as exc:
        raise ValueError(
            "constructive route-axis count must be an integer"
        ) from exc
    if isinstance(route_count, bool):
        raise ValueError("constructive route-axis count must be an integer")
    if count <= 0:
        raise ValueError("constructive route-axis count must be positive")
    with _CACHE_LOCK:
        cached = _CONSTRUCTED_ROUTE_AXIS_CACHE.get(count)
        if cached is not None:
            _CONSTRUCTED_ROUTE_AXIS_CACHE.move_to_end(count)
            _cache_axis_identity(cached)
            return cached.axis

        # The identifiers are materialized by one comprehension
        # and both digests come from ``_ordinal_route_axis_digests`` (the same
        # byte streams the per-ordinal loop fed to SHA-256).  The factory owns
        # this ASCII-only grammar, so quoting it directly is byte-identical to
        # canonical_json(string).
        # The digests run on a helper thread while this thread builds the
        # identifiers (numpy's loops and hashlib release the GIL).
        digests: list[tuple[str, str]] = []
        failure: list[BaseException] = []

        def compute_digests() -> None:
            try:
                digests.append(_ordinal_route_axis_digests(count))
            except BaseException as error:  # pragma: no cover - re-raised below
                failure.append(error)

        helper = None
        if count >= _ORDINAL_DIGEST_OVERLAP_MIN:
            helper = Thread(
                target=compute_digests, name="ordinal-route-axis-digests", daemon=True
            )
            helper.start()
        try:
            axis = tuple([f"route:{ordinal}" for ordinal in range(count)])
        finally:
            if helper is not None:
                helper.join()
        if helper is None:
            compute_digests()
        if failure:
            raise failure[0]
        ordered_hex, canonical_hex = digests[0]
        identity = _StringAxisIdentity(
            axis=axis,
            ordered_digest=ordered_hex,
            canonical_json_digest=canonical_hex,
        )
        _cache_axis_identity(identity)
        _CONSTRUCTED_ROUTE_AXIS_CACHE[count] = identity
        _CONSTRUCTED_ROUTE_AXIS_CACHE.move_to_end(count)
        while (
            len(_CONSTRUCTED_ROUTE_AXIS_CACHE)
            > _MAX_CACHED_CONSTRUCTED_ROUTE_AXES
        ):
            _retired_count, retired = _CONSTRUCTED_ROUTE_AXIS_CACHE.popitem(
                last=False
            )
            registered = _AXIS_CACHE.get(id(retired.axis))
            if registered is retired:
                _AXIS_CACHE.pop(id(retired.axis), None)
        return axis


def canonical_unique_string_axis(
    values: Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> Sequence[str]:
    """Return one normalized, uniqueness-validated ordered string axis."""

    if isinstance(values, ConstructiveOrdinalRouteAxis):
        # Uniqueness follows from the injective route:<decimal ordinal>
        # grammar.  Preserve the lazy sequence rather than defeating it with
        # tuple/set materialization.
        return values
    if isinstance(values, tuple):
        with _CACHE_LOCK:
            cached = _AXIS_CACHE.get(id(values))
            if cached is not None and cached.axis is values:
                _AXIS_CACHE.move_to_end(id(values))
                return values
            constructed_match = next(
                (
                    (route_count, constructed)
                    for route_count, constructed in (
                        _CONSTRUCTED_ROUTE_AXIS_CACHE.items()
                    )
                    if constructed.axis is values
                ),
                None,
            )
            if constructed_match is not None:
                route_count, constructed = constructed_match
                _CONSTRUCTED_ROUTE_AXIS_CACHE.move_to_end(route_count)
                _cache_axis_identity(constructed)
                return values
        axis = (
            values
            if all(type(value) is str for value in values)
            else tuple(map(str, values))
        )
    else:
        axis = tuple(map(str, values))
    if (not axis and not allow_empty) or len(set(axis)) != len(axis):
        qualifier = "nonempty " if not allow_empty else ""
        raise ValueError(f"{name} must be a {qualifier}unique ordered string axis")
    axis_hash = _ordered_axis_digest(axis)
    with _CACHE_LOCK:
        _cache_axis_identity(
            _StringAxisIdentity(
                axis=axis,
                ordered_digest=axis_hash,
            )
        )
    return axis


def ordered_string_axis_hash(
    values: Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> str:
    """Return the cached exact ordered-axis digest."""

    if isinstance(values, ConstructiveOrdinalRouteAxis):
        if not values and not allow_empty:  # pragma: no cover - constructor forbids
            raise ValueError(f"{name} must be a nonempty unique ordered string axis")
        return values.ordered_digest

    axis = canonical_unique_string_axis(
        values,
        name=name,
        allow_empty=allow_empty,
    )
    with _CACHE_LOCK:
        cached = _AXIS_CACHE.get(id(axis))
        if (
            cached is None or cached.axis is not axis
        ):  # pragma: no cover - defensive
            raise RuntimeError("ordered string axis cache lost its validated identity")
        _AXIS_CACHE.move_to_end(id(axis))
        return cached.ordered_digest


def canonical_string_axis_content_hash(
    values: Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
    prefix: str = "",
) -> str:
    """Return the exact historical ``content_hash(tuple(axis))`` identity."""

    if isinstance(values, ConstructiveOrdinalRouteAxis):
        digest = values.canonical_json_digest
        return f"{prefix}{digest}" if prefix else digest

    axis = canonical_unique_string_axis(
        values,
        name=name,
        allow_empty=allow_empty,
    )
    with _CACHE_LOCK:
        cached = _AXIS_CACHE.get(id(axis))
        if cached is None or cached.axis is not axis:  # pragma: no cover
            raise RuntimeError("string axis cache lost its validated identity")
        if cached.canonical_json_digest is None:
            cached = _StringAxisIdentity(
                axis=axis,
                ordered_digest=cached.ordered_digest,
                canonical_json_digest=_canonical_json_axis_digest(axis),
            )
            _cache_axis_identity(cached)
        else:
            _AXIS_CACHE.move_to_end(id(axis))
        digest = cached.canonical_json_digest
    if digest is None:  # pragma: no cover - frozen identity invariant
        raise RuntimeError("string axis canonical digest is unavailable")
    return f"{prefix}{digest}" if prefix else digest


def compact_string_axis_layout(
    layout: Mapping[str, Any],
    *,
    axis: Sequence[str],
    axis_key: str,
    name: str,
    maximum_inline_count: int = 4096,
) -> dict[str, Any]:
    """Replace an inline packed axis with its exact count/hash reference."""

    canonical = canonical_unique_string_axis(axis, name=name)
    payload = dict(layout)
    if len(canonical) <= int(maximum_inline_count):
        return payload
    inline = payload.pop(axis_key, None)
    if inline is None:
        raise ValueError(f"packed {name} layout lacks {axis_key}")
    inline_axis = canonical_unique_string_axis(inline, name=name)
    if inline_axis is not canonical and inline_axis != canonical:
        raise ValueError(f"packed {name} differs from its declared axis")
    payload[f"{axis_key}_count"] = len(canonical)
    payload[f"{axis_key}_hash"] = ordered_string_axis_hash(
        canonical,
        name=name,
    )
    payload["axis_reference_schema"] = "ordered_string_axis_ref_v1"
    return payload


def resolve_packed_string_axis(
    layout: Mapping[str, Any],
    *,
    expected_axis: Sequence[str] | None,
    axis_key: str,
    name: str,
) -> Sequence[str]:
    """Resolve either a legacy inline axis or a compact exact-axis reference."""

    expected = (
        None
        if expected_axis is None
        else canonical_unique_string_axis(expected_axis, name=name)
    )
    inline = layout.get(axis_key)
    if inline is not None:
        resolved = canonical_unique_string_axis(inline, name=name)
        if expected is not None and resolved is not expected and resolved != expected:
            raise ValueError(f"packed {name} differs from its expected axis")
        return resolved if expected is None else expected
    if expected is None:
        raise ValueError(f"compact packed {name} requires its declared axis")
    if layout.get("axis_reference_schema") != "ordered_string_axis_ref_v1":
        raise ValueError(f"compact packed {name} lacks an axis-reference schema")
    if int(layout.get(f"{axis_key}_count", -1)) != len(expected):
        raise ValueError(f"compact packed {name} has the wrong axis length")
    expected_hash = ordered_string_axis_hash(expected, name=name)
    if str(layout.get(f"{axis_key}_hash", "")) != expected_hash:
        raise ValueError(f"compact packed {name} has the wrong ordered-axis hash")
    return expected
