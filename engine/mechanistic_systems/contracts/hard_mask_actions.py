"""Immutable complete hard-mask actions with compact route deltas.

A candidate is never identified only by density or by a scalar pruning step.
It carries a deployable complete packed mask plus a compact, canonical set of
physical route indices removed from its parent.  Human-readable route IDs are
inlined only for small actions; large prefix candidates therefore avoid both
quadratic Python-tuple storage and repeated copies of the full route-ID axis.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np

from mechanistic_systems.contracts.circuit_cost import CostVector
from mechanistic_systems.contracts.content_address import canonical_json, content_hash
from mechanistic_systems.contracts.response_geometry import (
    HistoricalResponseProfileName,
)


_BYTE_POPCOUNT = np.asarray(
    [int(value).bit_count() for value in range(256)], dtype=np.uint8
)
_INLINE_ROUTE_ID_LIMIT = 1024
_INLINE_INDEX_STORAGE_LIMIT_BYTES = 256 * 1024
_MASK_UPDATE_INDEX_CHUNK_SIZE = 1 << 20
_MASK_SCAN_BYTE_CHUNK_SIZE = 1 << 20
_IndexDTypeCode = Literal["u32", "u64"]


@dataclass(frozen=True)
class RemovedRouteIndexSet:
    """Compact canonical encoding of a sorted set of physical route indices.

    The encoding uses four bytes per route while the physical axis fits in
    uint32 and eight bytes otherwise.  This is deliberately content-addressed
    and representation-stable across platforms (little-endian on disk).
    """

    route_count: int
    count: int
    dtype_code: _IndexDTypeCode
    encoded: bytes

    def __post_init__(self) -> None:
        route_count = int(self.route_count)
        count = int(self.count)
        if route_count <= 0:
            raise ValueError("removed-route index set requires a positive route_count")
        if count < 0:
            raise ValueError("removed-route index count must be nonnegative")
        if self.dtype_code not in {"u32", "u64"}:
            raise ValueError("removed-route index dtype must be u32 or u64")
        itemsize = 4 if self.dtype_code == "u32" else 8
        if len(self.encoded) != count * itemsize:
            raise ValueError("removed-route index byte length does not match count")
        values = self.to_numpy()
        if values.size:
            if int(values[-1]) >= route_count:
                raise ValueError("removed-route index exceeds the physical route axis")
            if values.size > 1 and np.any(values[1:] <= values[:-1]):
                raise ValueError("removed-route indices must be strictly increasing")

    @classmethod
    def from_indices(
        cls,
        indices: Sequence[int] | np.ndarray,
        *,
        route_count: int,
    ) -> "RemovedRouteIndexSet":
        total = int(route_count)
        if total <= 0:
            raise ValueError("route_count must be positive")
        source = np.asarray(indices)
        raw = (
            source.reshape(-1)
            if source.dtype.kind in {"i", "u"}
            else np.asarray(indices, dtype=np.int64).reshape(-1)
        )
        if raw.size:
            if np.any(raw < 0) or np.any(raw >= total):
                raise ValueError(
                    "removed-route indices are outside the physical route axis"
                )
            if raw.size > 1 and not bool(np.all(raw[1:] > raw[:-1])):
                ordered = np.unique(raw)
                if ordered.size != raw.size:
                    raise ValueError("removed-route indices must be unique")
                # A canonical action is independent of caller ordering.
                raw = ordered
        code: _IndexDTypeCode = "u32" if total <= np.iinfo(np.uint32).max else "u64"
        dtype = np.dtype("<u4" if code == "u32" else "<u8")
        encoded = raw.astype(dtype, copy=False).tobytes(order="C")
        return cls(
            route_count=total, count=int(raw.size), dtype_code=code, encoded=encoded
        )

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype("<u4" if self.dtype_code == "u32" else "<u8")

    @property
    def storage_bytes(self) -> int:
        return len(self.encoded)

    @cached_property
    def index_hash(self) -> str:
        return content_hash(
            {
                "route_count": self.route_count,
                "count": self.count,
                "dtype_code": self.dtype_code,
                "encoded": self.encoded,
            },
            prefix="irg-removed-routes-",
        )

    @classmethod
    def index_hash_from_encoded_digest(
        cls,
        *,
        route_count: int,
        count: int,
        dtype_code: _IndexDTypeCode,
        encoded_sha256: str,
    ) -> str:
        """Reproduce ``index_hash`` from a streaming digest of encoded indices."""

        itemsize = 4 if dtype_code == "u32" else 8
        return content_hash(
            {
                "route_count": int(route_count),
                "count": int(count),
                "dtype_code": dtype_code,
                # Canonical bytes are represented by precisely this nested
                # digest/length payload in ``content_address._canonicalize``.
                "encoded": {
                    "bytes_sha256": str(encoded_sha256),
                    "length": int(count) * itemsize,
                },
            },
            prefix="irg-removed-routes-",
        )

    def to_numpy(self) -> np.ndarray:
        """Return a zero-copy, read-only native NumPy view."""

        return np.frombuffer(self.encoded, dtype=self.numpy_dtype, count=self.count)

    def to_tuple(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.to_numpy())

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[int]:
        return (int(value) for value in self.to_numpy())

    def to_dict(self) -> dict[str, object]:
        return {
            "route_count": self.route_count,
            "count": self.count,
            "dtype_code": self.dtype_code,
            "encoded_b64": base64.b64encode(self.encoded).decode("ascii"),
            "index_hash": self.index_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RemovedRouteIndexSet":
        value = cls(
            route_count=int(payload["route_count"]),
            count=int(payload["count"]),
            dtype_code=str(payload["dtype_code"]),  # type: ignore[arg-type]
            encoded=base64.b64decode(str(payload["encoded_b64"])),
        )
        expected = payload.get("index_hash")
        if expected is not None and str(expected) != value.index_hash:
            raise ValueError("serialized removed-route index hash mismatch")
        return value


@dataclass(frozen=True)
class HardMask:
    route_axis_hash: str
    route_count: int
    packed_bits: bytes

    def __post_init__(self) -> None:
        if not self.route_axis_hash:
            raise ValueError("hard mask requires a route_axis_hash")
        count = int(self.route_count)
        if count <= 0:
            raise ValueError("hard mask route_count must be positive")
        expected_bytes = (count + 7) // 8
        if len(self.packed_bits) != expected_bytes:
            raise ValueError("hard mask byte length does not match route_count")
        # Refuse nonzero padding bits so identical physical masks have exactly
        # one byte representation.
        padding = expected_bytes * 8 - count
        if padding:
            invalid_mask = (1 << padding) - 1
            if self.packed_bits[-1] & invalid_mask:
                raise ValueError("hard mask has nonzero padding bits")

    @classmethod
    def from_bool_array(
        cls, values: Sequence[bool] | np.ndarray, *, route_axis_hash: str
    ) -> "HardMask":
        # A bool ndarray skips the uint8 copy and the binary scan.
        # numpy's bool->uint8 cast maps every stored byte to 0/1 (nonzero ->
        # 1), so the v5 scan could never fire for bool input, and packbits
        # reads a bool array with the same nonzero -> 1 rule, so the packed
        # bytes are the ones the v5 copy produced.  Other inputs keep the v5
        # conversion; their scan becomes one max() pass (for uint8, some value
        # is neither 0 nor 1 iff the maximum exceeds 1).
        if isinstance(values, np.ndarray) and values.dtype == np.bool_:
            array = np.asarray(values).reshape(-1)
            if array.size == 0:
                raise ValueError("hard mask cannot be empty")
        else:
            array = np.asarray(values, dtype=np.uint8).reshape(-1)
            if array.size == 0:
                raise ValueError("hard mask cannot be empty")
            if int(array.max()) > 1:
                raise ValueError("hard mask values must be binary")
        packed = np.packbits(array, bitorder="big").tobytes()
        return cls(
            route_axis_hash=str(route_axis_hash),
            route_count=int(array.size),
            packed_bits=packed,
        )

    @classmethod
    def dense(cls, route_count: int, *, route_axis_hash: str) -> "HardMask":
        return cls.from_bool_array(
            np.ones(int(route_count), dtype=np.uint8), route_axis_hash=route_axis_hash
        )

    def to_bool_array(self) -> np.ndarray:
        return np.unpackbits(
            np.frombuffer(self.packed_bits, dtype=np.uint8),
            bitorder="big",
            count=self.route_count,
        ).astype(bool)

    @cached_property
    def mask_hash(self) -> str:
        return content_hash(
            {
                "route_axis_hash": self.route_axis_hash,
                "route_count": self.route_count,
                "packed_bits": self.packed_bits,
            },
            prefix="irg-mask-",
        )

    @classmethod
    def mask_hash_from_packed_digest(
        cls,
        *,
        route_axis_hash: str,
        route_count: int,
        packed_sha256: str,
    ) -> str:
        """Reproduce :attr:`mask_hash` from a streamed packed-byte digest.

        The helper deliberately does not define a second mask identity.  It
        supplies the exact canonicalized ``bytes`` payload used by
        :func:`content_hash`, allowing a route-scale producer to hash bounded
        packed-byte chunks without first retaining another complete
        :class:`HardMask`.  The caller remains responsible for streaming the
        canonical big-endian packed representation, including zero padding in
        the final byte.
        """

        axis_hash = str(route_axis_hash)
        if not axis_hash:
            raise ValueError("hard mask requires a route_axis_hash")
        count = int(route_count)
        if count <= 0:
            raise ValueError("hard mask route_count must be positive")
        digest = str(packed_sha256)
        if len(digest) != 64 or digest != digest.lower():
            raise ValueError("packed_sha256 must be a lowercase SHA-256 digest")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(
                "packed_sha256 must be a lowercase SHA-256 digest"
            ) from error
        return content_hash(
            {
                "route_axis_hash": axis_hash,
                "route_count": count,
                "packed_bits": {
                    "bytes_sha256": digest,
                    "length": (count + 7) // 8,
                },
            },
            prefix="irg-mask-",
        )

    @cached_property
    def retained_count(self) -> int:
        packed = np.frombuffer(self.packed_bits, dtype=np.uint8)
        return int(np.sum(_BYTE_POPCOUNT[packed], dtype=np.int64))

    def removed_indices_from(self, parent: "HardMask") -> tuple[int, ...]:
        """Return parent-retained routes removed by this mask without unpacking both masks."""

        return self.removed_index_set_from(parent).to_tuple()

    def removed_index_set_from(self, parent: "HardMask") -> RemovedRouteIndexSet:
        """Return the compact set of parent-retained routes removed by this mask."""

        chunks = tuple(self.iter_removed_index_chunks_from(parent))
        if not chunks:
            return RemovedRouteIndexSet.from_indices((), route_count=self.route_count)
        dtype = np.dtype(
            "<u4" if self.route_count <= np.iinfo(np.uint32).max else "<u8"
        )
        encoded = b"".join(
            np.asarray(chunk, dtype=dtype).tobytes(order="C") for chunk in chunks
        )
        return RemovedRouteIndexSet(
            route_count=self.route_count,
            count=sum(int(chunk.size) for chunk in chunks),
            dtype_code="u32" if dtype.itemsize == 4 else "u64",
            encoded=encoded,
        )

    def iter_removed_index_chunks_from(
        self,
        parent: "HardMask",
        *,
        packed_byte_chunk_size: int = _MASK_SCAN_BYTE_CHUNK_SIZE,
    ) -> Iterator[np.ndarray]:
        """Yield sorted packed-delta indices in bounded read-only NumPy chunks."""

        self.assert_same_axis(parent)
        chunk_size = int(packed_byte_chunk_size)
        if chunk_size <= 0:
            raise ValueError("packed_byte_chunk_size must be positive")
        child = np.frombuffer(self.packed_bits, dtype=np.uint8)
        parent_bytes = np.frombuffer(parent.packed_bits, dtype=np.uint8)
        dtype = np.dtype(
            "<u4" if self.route_count <= np.iinfo(np.uint32).max else "<u8"
        )
        for byte_start in range(0, int(child.size), chunk_size):
            byte_stop = min(int(child.size), byte_start + chunk_size)
            child_slab = child[byte_start:byte_stop]
            parent_slab = parent_bytes[byte_start:byte_stop]
            if np.any(np.bitwise_and(child_slab, np.bitwise_not(parent_slab))):
                raise ValueError("hard-mask action may not reactivate a route")
            removed_bytes = np.bitwise_and(parent_slab, np.bitwise_not(child_slab))
            relative_bytes = np.flatnonzero(removed_bytes)
            if not relative_bytes.size:
                continue
            bits = np.unpackbits(
                removed_bytes[relative_bytes],
                bitorder="big",
            ).reshape(-1, 8)
            rows, columns = np.nonzero(bits)
            indices = (byte_start + relative_bytes[rows]) * 8 + columns
            indices = indices[indices < self.route_count]
            if indices.size:
                encoded_view = np.asarray(indices, dtype=dtype)
                encoded_view.setflags(write=False)
                yield encoded_view

    def removed_index_count_and_hash_from(
        self,
        parent: "HardMask",
    ) -> tuple[int, str]:
        """Validate and content-address a mask delta without retaining its indices."""

        dtype_code: _IndexDTypeCode = (
            "u32" if self.route_count <= np.iinfo(np.uint32).max else "u64"
        )
        digest = hashlib.sha256()
        count = 0
        for chunk in self.iter_removed_index_chunks_from(parent):
            digest.update(memoryview(chunk).cast("B"))
            count += int(chunk.size)
        return (
            count,
            RemovedRouteIndexSet.index_hash_from_encoded_digest(
                route_count=self.route_count,
                count=count,
                dtype_code=dtype_code,
                encoded_sha256=digest.hexdigest(),
            ),
        )

    def with_removed_indices(
        self,
        removed_indices: Sequence[int] | np.ndarray | RemovedRouteIndexSet,
    ) -> "HardMask":
        """Create a child mask by clearing sparse route indices in packed form.

        This keeps complete-mask semantics while avoiding a full unpack/repack of
        the native route axis for every candidate.
        """

        if isinstance(removed_indices, RemovedRouteIndexSet):
            if removed_indices.route_count != self.route_count:
                raise ValueError(
                    "removed-route set belongs to a different physical route axis"
                )
            indices = removed_indices.to_numpy()
        else:
            source = np.asarray(removed_indices)
            indices = (
                source.reshape(-1)
                if source.dtype.kind in {"i", "u"}
                else np.asarray(removed_indices, dtype=np.int64).reshape(-1)
            )
            if indices.size and (
                np.any(indices < 0) or np.any(indices >= self.route_count)
            ):
                raise ValueError(
                    "removed-route indices are outside the physical route axis"
                )
        if indices.size == 0:
            return self
        packed = np.frombuffer(self.packed_bits, dtype=np.uint8).copy()
        # Build one packed clear mask.  ``np.bitwise_or.at`` is essential here:
        # several physical routes commonly share one packed byte, and ordinary
        # advanced-index assignment does not accumulate repeated indices.  The
        # bounded slabs avoid the former Python loop over millions of bytes and
        # avoid promoting a uint32 action axis to one route_count-sized int64
        # temporary.
        clear = np.zeros_like(packed)
        for start in range(0, int(indices.size), _MASK_UPDATE_INDEX_CHUNK_SIZE):
            slab = indices[start : start + _MASK_UPDATE_INDEX_CHUNK_SIZE]
            byte_indices = np.floor_divide(slab, 8)
            bit_offsets = np.bitwise_and(slab, 7).astype(np.uint8, copy=False)
            bit_masks = np.right_shift(np.uint8(0x80), bit_offsets)
            np.bitwise_or.at(clear, byte_indices, bit_masks)
        unique_count = int(np.sum(_BYTE_POPCOUNT[clear], dtype=np.int64))
        if unique_count != int(indices.size):
            raise ValueError("removed-route indices must be unique")
        if np.any(np.bitwise_and(packed, clear) != clear):
            raise ValueError("candidate attempts to remove an already-pruned route")
        np.bitwise_and(packed, np.bitwise_not(clear), out=packed)
        return HardMask(
            route_axis_hash=self.route_axis_hash,
            route_count=self.route_count,
            packed_bits=packed.tobytes(),
        )

    def assert_same_axis(self, other: "HardMask") -> None:
        if (
            self.route_axis_hash != other.route_axis_hash
            or self.route_count != other.route_count
        ):
            raise ValueError("hard masks lie on different physical route axes")

    def to_dict(self) -> dict[str, object]:
        return {
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "packed_bits_b64": base64.b64encode(self.packed_bits).decode("ascii"),
            "mask_hash": self.mask_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HardMask":
        mask = cls(
            route_axis_hash=str(payload["route_axis_hash"]),
            route_count=int(payload["route_count"]),
            packed_bits=base64.b64decode(str(payload["packed_bits_b64"])),
        )
        expected = payload.get("mask_hash")
        if expected is not None and str(expected) != mask.mask_hash:
            raise ValueError("serialized hard-mask hash mismatch")
        return mask


@dataclass(frozen=True)
class ActionCandidate:
    candidate_id: str
    parent_mask_hash: str
    removed_routes: RemovedRouteIndexSet | None
    removed_count_value: int
    removed_route_ids_inline: tuple[str, ...]
    resulting_mask: HardMask
    released_cost: CostVector
    generator_version: HistoricalResponseProfileName
    generator_parameters: Mapping[str, Any]
    legality_receipt: str
    parent_state_hash: str
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.candidate_id
            or not self.parent_mask_hash
            or not self.parent_state_hash
        ):
            raise ValueError("action candidate has an empty identity field")
        if not self.legality_receipt:
            raise ValueError("action candidate requires a legality receipt")
        if (
            int(self.removed_count_value) < 0
            or int(self.removed_count_value) > self.resulting_mask.route_count
        ):
            raise ValueError("candidate removed-route count is invalid")
        if self.removed_routes is not None:
            if self.removed_routes.route_count != self.resulting_mask.route_count:
                raise ValueError(
                    "candidate route delta does not match resulting hard-mask axis"
                )
            if self.removed_routes.count != int(self.removed_count_value):
                raise ValueError("candidate compact route delta count is inconsistent")
        if self.removed_route_ids_inline and len(self.removed_route_ids_inline) != int(
            self.removed_count_value
        ):
            raise ValueError(
                "inlined route IDs must cover the complete removed-route set"
            )
        if len(set(self.removed_route_ids_inline)) != len(
            self.removed_route_ids_inline
        ):
            raise ValueError("inlined removed route IDs must be unique")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("candidate tags must be unique")
        object.__setattr__(
            self, "generator_parameters", dict(self.generator_parameters)
        )
        canonical_json(self.generator_parameters)
        expected = self.compute_candidate_id()
        if self.candidate_id != expected:
            raise ValueError("action candidate content hash mismatch")

    @property
    def removed_count(self) -> int:
        return int(self.removed_count_value)

    @property
    def compact_removed_indices_available(self) -> bool:
        return self.removed_routes is not None

    def resolve_removed_indices(
        self, parent_mask: HardMask | None = None
    ) -> np.ndarray:
        """Return physical route indices, scanning packed masks only when necessary."""

        if self.removed_routes is not None:
            return self.removed_routes.to_numpy()
        if parent_mask is None:
            raise RuntimeError(
                "this large action omitted its explicit route-index set; "
                "provide the parent HardMask to resolve it"
            )
        if parent_mask.mask_hash != self.parent_mask_hash:
            raise ValueError("parent mask does not match the candidate parent identity")
        resolved = self.resulting_mask.removed_index_set_from(parent_mask)
        if resolved.count != self.removed_count:
            raise ValueError(
                "resolved hard-mask delta count differs from candidate receipt"
            )
        return resolved.to_numpy()

    @property
    def removed_indices_array(self) -> np.ndarray:
        """Compact zero-copy indices when retained; large actions require explicit resolution."""

        return self.resolve_removed_indices()

    @property
    def removed_route_indices(self) -> tuple[int, ...]:
        """Compatibility view; large actions require ``resolve_removed_indices(parent)``."""

        return tuple(int(value) for value in self.resolve_removed_indices())

    @property
    def removed_route_ids_complete(self) -> bool:
        return len(self.removed_route_ids_inline) == self.removed_count

    @property
    def removed_route_ids(self) -> tuple[str, ...]:
        """Human-readable IDs for small actions.

        Large actions intentionally do not retain thousands or millions of
        Python strings.  Call ``resolve_removed_route_ids`` with the canonical
        route axis when those names are genuinely needed.
        """

        if not self.removed_route_ids_complete:
            raise RuntimeError(
                "removed route IDs were not inlined for this large action; "
                "use resolve_removed_route_ids(canonical_route_ids)"
            )
        return self.removed_route_ids_inline

    def resolve_removed_route_ids(self, route_ids: Sequence[str]) -> tuple[str, ...]:
        if len(route_ids) != self.resulting_mask.route_count:
            raise ValueError(
                "route_ids do not cover the candidate's physical route axis"
            )
        return tuple(
            str(route_ids[int(index)]) for index in self.resolve_removed_indices()
        )

    @classmethod
    def _construct(
        cls,
        *,
        parent_mask_hash: str,
        removed_routes: RemovedRouteIndexSet | None,
        removed_count: int,
        removed_route_ids_inline: Sequence[str],
        resulting_mask: HardMask,
        released_cost: CostVector,
        generator_version: HistoricalResponseProfileName,
        generator_parameters: Mapping[str, Any],
        legality_receipt: str,
        parent_state_hash: str,
        tags: Sequence[str],
    ) -> "ActionCandidate":
        body = {
            "parent_mask_hash": str(parent_mask_hash),
            "removed_routes": removed_routes,
            "removed_count_value": int(removed_count),
            "removed_route_ids_inline": tuple(
                str(route_id) for route_id in removed_route_ids_inline
            ),
            "resulting_mask": resulting_mask,
            "released_cost": released_cost,
            "generator_version": generator_version,
            "generator_parameters": dict(generator_parameters),
            "legality_receipt": str(legality_receipt),
            "parent_state_hash": str(parent_state_hash),
            "tags": tuple(str(tag) for tag in tags),
        }
        provisional = cls.__new__(cls)
        for key, value in body.items():
            object.__setattr__(provisional, key, value)
        object.__setattr__(
            provisional, "candidate_id", provisional.compute_candidate_id()
        )
        provisional.__post_init__()
        return provisional

    @classmethod
    def create(
        cls,
        *,
        parent_mask: HardMask,
        resulting_mask: HardMask,
        removed_routes: RemovedRouteIndexSet | None = None,
        route_ids: Sequence[str],
        released_cost: CostVector,
        generator_version: HistoricalResponseProfileName,
        generator_parameters: Mapping[str, Any],
        legality_receipt: str,
        parent_state_hash: str,
        tags: Sequence[str] = (),
    ) -> "ActionCandidate":
        """Create from two complete masks.

        This general path scans the packed mask difference.  The common
        ``candidate_from_removed_indices`` path bypasses that scan because the
        route delta is already known.
        """

        parent_mask.assert_same_axis(resulting_mask)
        if len(route_ids) != parent_mask.route_count:
            raise ValueError("route_ids must cover the hard-mask route axis")
        if removed_routes is None:
            removed_routes = resulting_mask.removed_index_set_from(parent_mask)
        else:
            if removed_routes.route_count != resulting_mask.route_count:
                raise ValueError(
                    "candidate route delta does not match resulting hard-mask axis"
                )
            child = np.frombuffer(resulting_mask.packed_bits, dtype=np.uint8)
            parent = np.frombuffer(parent_mask.packed_bits, dtype=np.uint8)
            reactivated = np.bitwise_and(child, np.bitwise_not(parent))
            if np.any(reactivated):
                raise ValueError("hard-mask action may not reactivate a route")
            removed_bytes = np.bitwise_and(parent, np.bitwise_not(child))
            expected_count = int(np.sum(_BYTE_POPCOUNT[removed_bytes], dtype=np.int64))
            indices = removed_routes.to_numpy().astype(np.int64, copy=False)
            byte_indices = indices // 8
            bit_masks = (1 << (7 - (indices % 8))).astype(np.uint8)
            if removed_routes.count != expected_count or np.any(
                (removed_bytes[byte_indices] & bit_masks) == 0
            ):
                raise ValueError("candidate compact route delta is inconsistent")
        inline = (
            tuple(str(route_ids[int(index)]) for index in removed_routes.to_numpy())
            if removed_routes.count <= _INLINE_ROUTE_ID_LIMIT
            else ()
        )
        return cls._construct(
            parent_mask_hash=parent_mask.mask_hash,
            removed_routes=(
                removed_routes
                if removed_routes.storage_bytes <= _INLINE_INDEX_STORAGE_LIMIT_BYTES
                else None
            ),
            removed_count=removed_routes.count,
            removed_route_ids_inline=inline,
            resulting_mask=resulting_mask,
            released_cost=released_cost,
            generator_version=generator_version,
            generator_parameters=generator_parameters,
            legality_receipt=legality_receipt,
            parent_state_hash=parent_state_hash,
            tags=tags,
        )

    @classmethod
    def from_known_delta(
        cls,
        *,
        parent_mask: HardMask,
        resulting_mask: HardMask,
        removed_routes: RemovedRouteIndexSet,
        route_ids: Sequence[str],
        released_cost: CostVector,
        generator_version: HistoricalResponseProfileName,
        generator_parameters: Mapping[str, Any],
        legality_receipt: str,
        parent_state_hash: str,
        tags: Sequence[str] = (),
    ) -> "ActionCandidate":
        """``create`` for a delta that is exact by construction.

        The fields, ``candidate_id`` and every payload are those ``create``
        builds from the same arguments.  Only its full-mask reactivation and
        consistency scans are skipped: the caller derived ``resulting_mask``
        by clearing exactly ``removed_routes`` (all retained by the parent) in
        a copy of ``parent_mask``'s packed bytes, so the scans cannot fail.
        """

        parent_mask.assert_same_axis(resulting_mask)
        if len(route_ids) != parent_mask.route_count:
            raise ValueError("route_ids must cover the hard-mask route axis")
        if removed_routes.route_count != resulting_mask.route_count:
            raise ValueError(
                "candidate route delta does not match resulting hard-mask axis"
            )
        inline = (
            tuple(str(route_ids[int(index)]) for index in removed_routes.to_numpy())
            if removed_routes.count <= _INLINE_ROUTE_ID_LIMIT
            else ()
        )
        return cls._construct(
            parent_mask_hash=parent_mask.mask_hash,
            removed_routes=(
                removed_routes
                if removed_routes.storage_bytes <= _INLINE_INDEX_STORAGE_LIMIT_BYTES
                else None
            ),
            removed_count=removed_routes.count,
            removed_route_ids_inline=inline,
            resulting_mask=resulting_mask,
            released_cost=released_cost,
            generator_version=generator_version,
            generator_parameters=generator_parameters,
            legality_receipt=legality_receipt,
            parent_state_hash=parent_state_hash,
            tags=tags,
        )

    def compute_candidate_id(self) -> str:
        return content_hash(
            {
                "parent_mask_hash": self.parent_mask_hash,
                "removed_count": self.removed_count,
                "resulting_mask_hash": self.resulting_mask.mask_hash,
                "released_cost": self.released_cost,
                "generator_version": self.generator_version,
                "generator_parameters": self.generator_parameters,
                "legality_receipt": self.legality_receipt,
                "parent_state_hash": self.parent_state_hash,
                "tags": self.tags,
            },
            prefix="irg-candidate-",
        )

    def with_generator_receipt(
        self,
        *,
        generator_parameters: Mapping[str, Any],
        legality_receipt: str,
        tags: Sequence[str] | None = None,
    ) -> "ActionCandidate":
        """Rebind an unchanged physical action to an additional design receipt.

        Partition-local generation uses this after the version solver has run
        under a transient route filter.  The complete mask and compact route
        delta are preserved, while the candidate identity commits to the
        geometric partition that made its score comparisons lawful.
        """

        return self._construct(
            parent_mask_hash=self.parent_mask_hash,
            removed_routes=self.removed_routes,
            removed_count=self.removed_count,
            removed_route_ids_inline=self.removed_route_ids_inline,
            resulting_mask=self.resulting_mask,
            released_cost=self.released_cost,
            generator_version=self.generator_version,
            generator_parameters=dict(generator_parameters),
            legality_receipt=str(legality_receipt),
            parent_state_hash=self.parent_state_hash,
            tags=self.tags if tags is None else tuple(tags),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "parent_mask_hash": self.parent_mask_hash,
            "removed_routes": (
                self.removed_routes.to_dict()
                if self.removed_routes is not None
                else None
            ),
            "removed_count": self.removed_count,
            "compact_removed_indices_available": self.compact_removed_indices_available,
            "removed_route_ids_inline": list(self.removed_route_ids_inline),
            "removed_route_ids_complete": self.removed_route_ids_complete,
            "resulting_mask": self.resulting_mask.to_dict(),
            "released_cost": self.released_cost.to_dict(),
            "generator_version": self.generator_version,
            "generator_parameters": dict(self.generator_parameters),
            "legality_receipt": self.legality_receipt,
            "parent_state_hash": self.parent_state_hash,
            "tags": list(self.tags),
        }


def candidate_from_removed_indices(
    *,
    parent_mask: HardMask,
    route_ids: Sequence[str],
    removed_indices: Sequence[int] | np.ndarray | RemovedRouteIndexSet,
    released_cost: CostVector,
    generator_version: HistoricalResponseProfileName,
    generator_parameters: Mapping[str, Any],
    legality_receipt: str,
    parent_state_hash: str,
    tags: Sequence[str] = (),
) -> ActionCandidate:
    if len(route_ids) != parent_mask.route_count:
        raise ValueError("route_ids must cover the hard-mask route axis")
    compact = (
        removed_indices
        if isinstance(removed_indices, RemovedRouteIndexSet)
        else RemovedRouteIndexSet.from_indices(
            removed_indices, route_count=parent_mask.route_count
        )
    )
    if compact.route_count != parent_mask.route_count:
        raise ValueError("removed-route set belongs to a different physical route axis")
    resulting = parent_mask.with_removed_indices(compact)
    inline = (
        tuple(str(route_ids[int(index)]) for index in compact.to_numpy())
        if compact.count <= _INLINE_ROUTE_ID_LIMIT
        else ()
    )
    return ActionCandidate._construct(
        parent_mask_hash=parent_mask.mask_hash,
        removed_routes=(
            compact
            if compact.storage_bytes <= _INLINE_INDEX_STORAGE_LIMIT_BYTES
            else None
        ),
        removed_count=compact.count,
        removed_route_ids_inline=inline,
        resulting_mask=resulting,
        released_cost=released_cost,
        generator_version=generator_version,
        generator_parameters=generator_parameters,
        legality_receipt=legality_receipt,
        parent_state_hash=parent_state_hash,
        tags=tags,
    )


def candidate_materialization_limit(
    *,
    parent_mask: HardMask,
    configured_limit: int,
    memory_budget_bytes: int,
    minimum_candidates: int = 2,
) -> int:
    """Bound simultaneous complete-mask candidates by an explicit host-memory budget."""

    configured = int(configured_limit)
    budget = int(memory_budget_bytes)
    minimum = int(minimum_candidates)
    if configured <= 0 or budget <= 0 or minimum <= 0:
        raise ValueError("candidate materialization limits must be positive")
    # The complete packed child mask dominates at very large route counts.
    # Explicit route-index deltas are themselves capped and omitted beyond the
    # declared inline threshold, so this is a conservative per-candidate bound.
    per_candidate = (
        len(parent_mask.packed_bits) + _INLINE_INDEX_STORAGE_LIMIT_BYTES + 2048
    )
    affordable = budget // max(1, per_candidate)
    floor = min(configured, minimum)
    if affordable < floor:
        raise MemoryError(
            "candidate memory budget cannot retain the declared minimum number of complete masks"
        )
    return min(configured, int(affordable))


def deduplicate_candidates(
    candidates: Sequence[ActionCandidate],
) -> tuple[ActionCandidate, ...]:
    by_mask: dict[str, ActionCandidate] = {}
    for candidate in candidates:
        existing = by_mask.get(candidate.resulting_mask.mask_hash)
        if existing is None or candidate.candidate_id < existing.candidate_id:
            by_mask[candidate.resulting_mask.mask_hash] = candidate
    return tuple(sorted(by_mask.values(), key=lambda item: item.candidate_id))


__all__ = [
    "ActionCandidate",
    "HardMask",
    "RemovedRouteIndexSet",
    "candidate_from_removed_indices",
    "candidate_materialization_limit",
    "deduplicate_candidates",
]
