"""Deterministic scorer-native projection onto a frozen packed-v2 ladder.

The scorer emits signed positive-removal values ``-G*phi*c`` in raw target
logit units.  A content-addressed disjoint-shell CSR maps those values to a
selected-shell staging layout: each frozen action shell owns exactly the route
coordinates it removes, independent of the scorer's physical route windows.
Every shell is reduced through a deterministic padded binary tree, and a final
left-prefix forms the nested endpoints.  No floating atomic participates, no
host ``[context, route]`` matrix is materialized, and device memory scales with
the frozen ladder's final removal count rather than with irrelevant live routes.

The returned binary64 center is exact with respect to the *set of estimator
rows consumed*, not exact real arithmetic.  A conservative outward numerical
radius and absolute-input-sum bound accompany every center.  Projection alone
is not certification evidence: a post-provider closure must bind the scorer
receipt, all projection hashes, ledgers, and acquisition lineage.  Only that
closure hash is eligible for certifier records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from mechanistic_systems.circuits.irg.modern_v1_ladder_statistics import (
    V1FrozenCandidateLadder,
    V1FrozenLadderCertificationDesign,
    V1FrozenLadderProjectionRecord,
)
from mechanistic_systems.circuits.irg.modern_v1_directional_remainder import (
    V1DirectionalSecantDesign,
)
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.core.axis_identity import (
    ConstructiveOrdinalRouteAxis,
    ordered_string_axis_hash,
)
from mechanistic_systems.measurement.optimized_irg_providers import (
    _RouteChunkedLossOnlyConsumer,
)

try:  # pragma: no cover - import availability is environment dependent.
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

LADDER_PROJECTION_REQUEST_SCHEMA: Final = (
    "partition_free_v1_frozen_action_projection_request_v3"
)
LEGACY_DIRECTIONAL_LADDER_PROJECTION_REQUEST_SCHEMA: Final = (
    "partition_free_v1_directional_dual_chart_action_projection_request_v1"
)
DIRECTIONAL_LADDER_PROJECTION_REQUEST_SCHEMA: Final = (
    "partition_free_v1_directional_dual_chart_action_projection_request_v2"
)
LADDER_PROJECTION_CSR_SCHEMA: Final = (
    "partition_free_v1_frozen_action_projection_shell_csr_v3"
)
LADDER_PROJECTION_PACKET_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_frozen_action_projection_packet_receipt_v3"
)
LADDER_PROJECTION_DERIVATION_RECEIPT_SCHEMA: Final = (
    "partition_free_v1_frozen_action_projection_derivation_receipt_v3"
)
LADDER_PROJECTION_RESULT_SCHEMA: Final = (
    "partition_free_v1_frozen_action_projection_result_v3"
)
LADDER_PROJECTION_ACQUISITION_CLOSURE_SCHEMA: Final = (
    "partition_free_v1_frozen_action_projection_acquisition_closure_v3"
)
LADDER_PROJECTION_ACQUISITION_SCHEMA: Final = (
    "partition_free_v1_frozen_action_projection_acquisition_v3"
)
LEGACY_LADDER_PROJECTION_REPLAY_ADAPTER_SCHEMA: Final = (
    "partition_free_v1_legacy_tuple_ladder_projection_replay_adapter_v1"
)
LADDER_PROJECTION_ARITHMETIC_ID: Final = (
    "selected_shell_physical_index_gather_pairwise_endpoint_prefix_float64_v3"
)
LADDER_PROJECTION_COORDINATE: Final = "raw_target_logit"
LADDER_PROJECTION_COTANGENT: Final = "raw_target_logit_unit_cotangent"
LADDER_PROJECTION_ORIENTATION: Final = "positive_removal_negative_G_phi_c"
LADDER_PROJECTION_UNITS: Final = "raw_target_logit"
LADDER_PROJECTION_AUTHORITY: Final = (
    "estimator_projection_with_numerical_enclosure_only_"
    "no_fq_or_order_or_oracle_or_commit_authority"
)
LADDER_PROJECTION_MEMORY_SCOPE: Final = (
    "consumer_registered_tensors_excluding_scorer_source_allocator_"
    "framework_and_compiler_workspace_v2"
)

LOGICAL_TILE_SIZE: Final = 1 << 16
STAGE1_BLOCK_SIZE: Final = 1 << 8
_FLOAT64_BYTES: Final = 8
_UNIT_ROUNDOFF: Final = float(np.finfo(np.float64).eps / 2.0)
_NEGATIVE_ZERO_U64: Final = np.uint64(1 << 63)
_REQUEST_PREFIX: Final = "partition-free-v1-ladder-projection-request-"
_DIRECTIONAL_REQUEST_PREFIX: Final = (
    "partition-free-v1-directional-dual-chart-ladder-projection-request-"
)
_DIRECTIONAL_CAUSAL_REGISTRATION_PREFIX: Final = (
    "partition-free-v1-directional-d-c-causal-registration-"
)
_DIRECTIONAL_CAUSAL_DRAW_AXIS_PREFIX: Final = (
    "partition-free-v1-directional-causal-draw-axis-"
)
_CSR_PREFIX: Final = "partition-free-v1-ladder-projection-csr-"
_PACKET_PREFIX: Final = "partition-free-v1-ladder-projection-packet-"
_RECEIPT_PREFIX: Final = "partition-free-v1-ladder-projection-receipt-"
_CLOSURE_PREFIX: Final = "partition-free-v1-ladder-projection-closure-"
_MATRIX_PREFIX: Final = "partition-free-v1-ladder-projection-matrix-"
_SHELL_PREFIX: Final = "partition-free-v1-ladder-projection-shell-"
_WINDOW_PREFIX: Final = "partition-free-v1-ladder-projection-route-windows-"
_LEGACY_PREFIX: Final = "partition-free-v1-legacy-ladder-replay-"
_SCORER_RECEIPT_SCHEMA: Final = "partition_free_v1_raw_target_logit_stream_receipt_v1"
_SCORER_RECEIPT_PREFIX: Final = "partition-free-v1-raw-target-logit-stream-"


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be one nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be one positive integer")
    return result


def _identity_axis(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    return tuple(_identity(value, name=f"{name} item") for value in tuple(values))


def _directional_causal_draw_axis_hash(
    *,
    role: str,
    causal_registration_hash: str,
    context_ids: tuple[str, ...],
    draw_ids: tuple[str, ...],
) -> str:
    """Content-address one role's exact registered draw records.

    Context identity is deliberately only one coordinate of a draw record.
    The same ImageNet context may lawfully occur in independent D and C
    permutations; a raw draw identity may not be recycled across those roles.
    """

    return content_hash(
        {
            "schema": "partition_free_v1_directional_causal_draw_axis_v1",
            "role": role,
            "causal_registration_hash": causal_registration_hash,
            "records": tuple(zip(context_ids, draw_ids, strict=True)),
        },
        prefix=_DIRECTIONAL_CAUSAL_DRAW_AXIS_PREFIX,
    )


def projection_kernel_source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical_device(device: Any) -> Any:
    if torch is None:
        raise RuntimeError("frozen-action projection requires PyTorch")
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        try:
            return torch.device("cuda", int(torch.cuda.current_device()))
        except (AssertionError, RuntimeError):
            return result
    return result


def _device_identity(device: Any) -> str:
    result = _canonical_device(device)
    if result.type != "cuda":
        return str(result)
    index = (
        result.index if result.index is not None else int(torch.cuda.current_device())
    )
    try:
        name = torch.cuda.get_device_name(result)
    except (AssertionError, RuntimeError):  # pragma: no cover
        name = "cuda-device-name-unavailable"
    return f"cuda:{index}:{name}"


def _canonical_matrix(
    value: Any, *, shape: tuple[int, int], name: str
) -> tuple[np.ndarray, str]:
    array = np.array(value, dtype="<f8", order="C", copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one finite float64 matrix of shape {shape}")
    bits = array.reshape(-1).view(np.uint64)
    bits[bits == _NEGATIVE_ZERO_U64] = np.uint64(0)
    encoded = array.tobytes(order="C")
    result = np.frombuffer(encoded, dtype="<f8").reshape(shape)
    result.setflags(write=False)
    digest = content_hash(
        {
            "shape": shape,
            "dtype": "little_endian_float64",
            "encoded": {
                "bytes_sha256": hashlib.sha256(encoded).hexdigest(),
                "length": len(encoded),
            },
        },
        prefix=_MATRIX_PREFIX,
    )
    return result, digest


def _immutable_array(value: Any, *, dtype: str, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.dtype(dtype), order="C", copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one vector")
    result = np.frombuffer(array.tobytes(order="C"), dtype=np.dtype(dtype))
    result.setflags(write=False)
    return result


def _array_digest(array: np.ndarray) -> dict[str, object]:
    encoded = memoryview(np.ascontiguousarray(array)).cast("B")
    return {
        "dtype": array.dtype.str,
        "shape": tuple(int(value) for value in array.shape),
        "bytes_sha256": hashlib.sha256(encoded).hexdigest(),
        "length": encoded.nbytes,
    }


def _trailing_ones(value: int) -> int:
    result = 0
    while value & 1:
        result += 1
        value >>= 1
    return result


@dataclass(frozen=True, slots=True)
class V1PackedLadderShellCSR:
    """Content-addressed sorted ``(tile, shell)`` CSR for a packed ladder."""

    ladder_hash: str
    route_count: int
    physical_mask_axis_hash: str
    endpoint_action_hashes: tuple[str, ...]
    tile_segment_starts: np.ndarray = field(repr=False, compare=False)
    segment_tile_ids: np.ndarray = field(repr=False, compare=False)
    segment_shell_ids: np.ndarray = field(repr=False, compare=False)
    segment_member_starts: np.ndarray = field(repr=False, compare=False)
    member_tile_offsets: np.ndarray = field(repr=False, compare=False)
    segment_block_starts: np.ndarray = field(repr=False, compare=False)
    member_block_starts: np.ndarray = field(repr=False, compare=False)
    member_block_counts: np.ndarray = field(repr=False, compare=False)
    member_block_segment_ids: np.ndarray = field(repr=False, compare=False)
    tile_block_starts: np.ndarray = field(repr=False, compare=False)
    segment_stage2_merge_counts: np.ndarray = field(repr=False, compare=False)
    shell_segment_counts: np.ndarray = field(repr=False, compare=False)
    shell_segment_starts: np.ndarray = field(repr=False, compare=False)
    shell_segment_indices: np.ndarray = field(repr=False, compare=False)
    shell_hashes: tuple[str, ...]
    shell_removed_counts: tuple[int, ...]
    logical_tile_size: int = LOGICAL_TILE_SIZE
    schema: str = LADDER_PROJECTION_CSR_SCHEMA
    _csr_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ladder_hash", _identity(self.ladder_hash, name="ladder hash")
        )
        object.__setattr__(
            self,
            "physical_mask_axis_hash",
            _identity(self.physical_mask_axis_hash, name="physical mask-axis hash"),
        )
        route_count = _positive_int(self.route_count, name="route count")
        actions = _identity_axis(self.endpoint_action_hashes, name="CSR action axis")
        if not actions or len(actions) != len(set(actions)):
            raise ValueError("CSR requires unique endpoint identities")
        if self.logical_tile_size != LOGICAL_TILE_SIZE:
            raise ValueError("projection CSR logical tile size is not canonical")
        spec = {
            "tile_segment_starts": ("<u8", self.tile_segment_starts),
            "segment_tile_ids": ("<u8", self.segment_tile_ids),
            "segment_shell_ids": ("<u2", self.segment_shell_ids),
            "segment_member_starts": ("<u8", self.segment_member_starts),
            "member_tile_offsets": ("<u2", self.member_tile_offsets),
            "segment_block_starts": ("<u8", self.segment_block_starts),
            "member_block_starts": ("<u8", self.member_block_starts),
            "member_block_counts": ("<u2", self.member_block_counts),
            "member_block_segment_ids": ("<u8", self.member_block_segment_ids),
            "tile_block_starts": ("<u8", self.tile_block_starts),
            "segment_stage2_merge_counts": ("<u2", self.segment_stage2_merge_counts),
            "shell_segment_counts": ("<u8", self.shell_segment_counts),
            "shell_segment_starts": ("<u8", self.shell_segment_starts),
            "shell_segment_indices": ("<u8", self.shell_segment_indices),
        }
        for name, (dtype, value) in spec.items():
            object.__setattr__(
                self, name, _immutable_array(value, dtype=dtype, name=name)
            )
        tiles = (route_count + LOGICAL_TILE_SIZE - 1) // LOGICAL_TILE_SIZE
        segment_count = int(self.segment_tile_ids.size)
        block_count = int(self.member_block_starts.size)
        if (
            self.tile_segment_starts.size != tiles + 1
            or self.segment_member_starts.size != segment_count + 1
            or self.segment_block_starts.size != segment_count + 1
            or self.tile_block_starts.size != tiles + 1
            or self.segment_shell_ids.size != segment_count
            or self.segment_stage2_merge_counts.size != segment_count
            or self.member_block_counts.size != block_count
            or self.member_block_segment_ids.size != block_count
            or self.shell_segment_counts.size != len(actions)
            or self.shell_segment_starts.size != len(actions) + 1
            or self.shell_segment_indices.size != segment_count
            or int(self.tile_segment_starts[0]) != 0
            or int(self.tile_segment_starts[-1]) != segment_count
            or int(self.segment_member_starts[0]) != 0
            or int(self.segment_member_starts[-1]) != int(self.member_tile_offsets.size)
            or int(self.segment_block_starts[0]) != 0
            or int(self.segment_block_starts[-1]) != block_count
            or int(self.tile_block_starts[0]) != 0
            or int(self.tile_block_starts[-1]) != block_count
            or int(self.shell_segment_starts[0]) != 0
            or int(self.shell_segment_starts[-1]) != segment_count
        ):
            raise ValueError("projection CSR offsets do not close their axes")
        for axis in (
            self.tile_segment_starts,
            self.segment_member_starts,
            self.segment_block_starts,
            self.tile_block_starts,
            self.shell_segment_starts,
        ):
            if np.any(axis[1:] < axis[:-1]):
                raise ValueError("projection CSR offsets must be monotone")
        if segment_count <= 0 or self.member_tile_offsets.size <= 0:
            raise ValueError("projection CSR requires a nonempty final action")
        if np.any(self.segment_tile_ids >= tiles) or np.any(
            self.segment_shell_ids >= len(actions)
        ) or np.any(self.member_block_segment_ids >= segment_count):
            raise ValueError("projection CSR segment left its axis")
        pairs = np.stack(
            (self.segment_tile_ids, self.segment_shell_ids.astype("<u8")), axis=1
        )
        if len(pairs) > 1 and not np.all(
            (pairs[1:, 0] > pairs[:-1, 0])
            | ((pairs[1:, 0] == pairs[:-1, 0]) & (pairs[1:, 1] > pairs[:-1, 1]))
        ):
            raise ValueError("projection CSR segments are not sorted tile/shell pairs")
        seen = np.zeros(len(actions), dtype=np.int64)
        observed_counts = np.zeros(len(actions), dtype=np.int64)
        for tile in range(tiles):
            left, right = int(self.tile_segment_starts[tile]), int(
                self.tile_segment_starts[tile + 1]
            )
            if left < right and not np.all(self.segment_tile_ids[left:right] == tile):
                raise ValueError("projection CSR tile offsets disagree")
            if int(self.tile_block_starts[tile]) != int(
                self.segment_block_starts[left]
            ) or int(self.tile_block_starts[tile + 1]) != int(
                self.segment_block_starts[right]
            ):
                raise ValueError("projection CSR tile block offsets disagree")
        for segment in range(segment_count):
            left = int(self.segment_member_starts[segment])
            right = int(self.segment_member_starts[segment + 1])
            members = self.member_tile_offsets[left:right]
            if left >= right or (
                len(members) > 1 and np.any(members[1:] <= members[:-1])
            ):
                raise ValueError(
                    "projection CSR segment membership is not strictly sorted"
                )
            block_left = int(self.segment_block_starts[segment])
            block_right = int(self.segment_block_starts[segment + 1])
            if (
                block_right - block_left
                != (right - left + STAGE1_BLOCK_SIZE - 1) // STAGE1_BLOCK_SIZE
            ):
                raise ValueError("projection CSR stage-1 block count changed")
            for block in range(block_left, block_right):
                expected_start = left + (block - block_left) * STAGE1_BLOCK_SIZE
                expected_count = min(STAGE1_BLOCK_SIZE, right - expected_start)
                if (
                    int(self.member_block_starts[block]) != expected_start
                    or int(self.member_block_counts[block]) != expected_count
                    or int(self.member_block_segment_ids[block]) != segment
                ):
                    raise ValueError("projection CSR member block changed")
            shell = int(self.segment_shell_ids[segment])
            if int(self.segment_stage2_merge_counts[segment]) != _trailing_ones(
                int(seen[shell])
            ):
                raise ValueError("projection CSR stage-2 schedule changed")
            seen[shell] += 1
            observed_counts[shell] += right - left
        if not np.array_equal(seen.astype("<u8"), self.shell_segment_counts):
            raise ValueError("projection CSR shell segment counts changed")
        for shell in range(len(actions)):
            left = int(self.shell_segment_starts[shell])
            right = int(self.shell_segment_starts[shell + 1])
            segment_indices = self.shell_segment_indices[left:right]
            if int(right - left) != int(self.shell_segment_counts[shell]):
                raise ValueError("projection CSR shell segment count changed")
            if len(segment_indices) and (
                np.any(segment_indices >= segment_count)
                or np.any(self.segment_shell_ids[segment_indices] != shell)
                or (
                    len(segment_indices) > 1
                    and np.any(segment_indices[1:] <= segment_indices[:-1])
                )
            ):
                raise ValueError("projection CSR shell segment schedule changed")
        shell_hashes = _identity_axis(self.shell_hashes, name="CSR shell hash axis")
        shell_counts = tuple(
            _positive_int(value, name="CSR shell count")
            for value in self.shell_removed_counts
        )
        if (
            len(shell_hashes) != len(actions)
            or tuple(int(v) for v in observed_counts) != shell_counts
        ):
            raise ValueError("projection CSR shell identities are not aligned")
        if self.schema != LADDER_PROJECTION_CSR_SCHEMA:
            raise ValueError("projection CSR schema is invalid")
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "endpoint_action_hashes", actions)
        object.__setattr__(self, "shell_hashes", shell_hashes)
        object.__setattr__(self, "shell_removed_counts", shell_counts)
        self.assert_integrity()

    @property
    def tile_count(self) -> int:
        return int(self.tile_segment_starts.size - 1)

    @property
    def endpoint_count(self) -> int:
        return len(self.endpoint_action_hashes)

    @property
    def route_axis_hash(self) -> str:
        """Compatibility alias for the physical mask-axis hash only.

        This alias never identifies the ordered score-route axis.  New code
        must use :attr:`physical_mask_axis_hash` explicitly.
        """

        return self.physical_mask_axis_hash

    @property
    def maximum_stage2_levels(self) -> int:
        maximum = int(np.max(self.shell_segment_counts, initial=1))
        return max(1, int(math.ceil(math.log2(maximum + 1))))

    @property
    def host_storage_bytes(self) -> int:
        return sum(int(getattr(self, name).nbytes) for name in self._array_names())

    @staticmethod
    def _array_names() -> tuple[str, ...]:
        return (
            "tile_segment_starts",
            "segment_tile_ids",
            "segment_shell_ids",
            "segment_member_starts",
            "member_tile_offsets",
            "segment_block_starts",
            "member_block_starts",
            "member_block_counts",
            "member_block_segment_ids",
            "tile_block_starts",
            "segment_stage2_merge_counts",
            "shell_segment_counts",
            "shell_segment_starts",
            "shell_segment_indices",
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "ladder_hash": self.ladder_hash,
            "route_count": self.route_count,
            "physical_mask_axis_hash": self.physical_mask_axis_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "logical_tile_size": self.logical_tile_size,
            "stage1_block_size": STAGE1_BLOCK_SIZE,
            "shell_hashes": self.shell_hashes,
            "shell_removed_counts": self.shell_removed_counts,
            "arrays": {
                name: _array_digest(getattr(self, name)) for name in self._array_names()
            },
        }

    @property
    def csr_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_CSR_PREFIX)
        if self._csr_hash is not None and observed != self._csr_hash:
            raise ValueError("projection CSR was tampered after construction")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("projection CSR hash does not match")
        object.__setattr__(self, "_csr_hash", observed)
        return observed


def build_v1_packed_ladder_shell_csr(
    ladder: V1FrozenCandidateLadder,
) -> V1PackedLadderShellCSR:
    """Build the CSR once by two bounded scans of each disjoint packed shell."""

    if type(ladder) is not V1FrozenCandidateLadder:
        raise TypeError("projection CSR requires the exact frozen ladder type")
    ladder_hash = ladder.assert_integrity()
    if not ladder.is_packed:
        raise PermissionError(
            "production projection requires packed ladder schema v2; tuple-v1 is replay-only"
        )
    route_count = ladder.route_count
    endpoint_count = len(ladder.actions)
    tile_count = (route_count + LOGICAL_TILE_SIZE - 1) // LOGICAL_TILE_SIZE
    counts = np.zeros((tile_count, endpoint_count), dtype=np.int64)
    shell_hashes: list[str] = []
    shell_counts: list[int] = []
    index_dtype = np.dtype("<u4" if route_count <= np.iinfo(np.uint32).max else "<u8")
    cumulative = ladder.endpoint_removed_counts
    prior = 0
    for shell, action in enumerate(ladder.actions):
        digest = hashlib.sha256()
        observed = 0
        previous = -1
        for raw in ladder.iter_shell_index_chunks(endpoint_index=action.endpoint_index):
            indices = np.asarray(raw)
            if indices.ndim != 1 or indices.dtype.kind not in {"i", "u"}:
                raise ValueError("packed ladder emitted an invalid shell chunk")
            normalized = indices.astype(index_dtype, copy=False)
            if len(normalized):
                if (
                    int(normalized[0]) <= previous
                    or int(normalized[-1]) >= route_count
                    or (
                        len(normalized) > 1
                        and np.any(normalized[1:] <= normalized[:-1])
                    )
                ):
                    raise ValueError("packed shell indices are not sorted and unique")
                previous = int(normalized[-1])
                tiles, tile_counts = np.unique(
                    normalized.astype(np.uint64) // LOGICAL_TILE_SIZE,
                    return_counts=True,
                )
                counts[tiles.astype(np.int64), shell] += tile_counts
                digest.update(normalized.tobytes(order="C"))
                observed += int(normalized.size)
        expected = int(cumulative[shell]) - prior
        if observed != expected or observed <= 0:
            raise ValueError("packed shell count differs from its action identity")
        shell_counts.append(observed)
        shell_hashes.append(
            content_hash(
                {
                    "schema": "partition_free_v1_packed_disjoint_shell_v2",
                    "ladder_hash": ladder_hash,
                    "endpoint_index": action.endpoint_index,
                    "action_hash": action.action_hash,
                    "route_count": route_count,
                    "physical_mask_axis_hash": ladder.route_axis_hash,
                    "sorted_indices": {
                        "dtype": index_dtype.str,
                        "count": observed,
                        "bytes_sha256": digest.hexdigest(),
                    },
                },
                prefix=_SHELL_PREFIX,
            )
        )
        prior = int(cumulative[shell])

    pairs = np.argwhere(counts > 0)
    segment_tile_ids = pairs[:, 0].astype("<u8", copy=False)
    if endpoint_count > np.iinfo(np.uint16).max:
        raise ValueError("projection CSR shell axis exceeds canonical uint16 storage")
    segment_shell_ids = pairs[:, 1].astype("<u2", copy=False)
    segment_counts = counts[pairs[:, 0], pairs[:, 1]].astype(np.int64, copy=False)
    segment_member_starts = np.empty(len(segment_counts) + 1, dtype="<u8")
    segment_member_starts[0] = 0
    np.cumsum(segment_counts, out=segment_member_starts[1:])
    members = np.empty(int(segment_member_starts[-1]), dtype="<u2")
    lookup = np.full((tile_count, endpoint_count), -1, dtype=np.int64)
    lookup[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs), dtype=np.int64)
    cursors = segment_member_starts[:-1].astype(np.int64, copy=True)
    for shell, action in enumerate(ladder.actions):
        for raw in ladder.iter_shell_index_chunks(endpoint_index=action.endpoint_index):
            indices = np.asarray(raw, dtype=np.uint64)
            if not len(indices):
                continue
            tiles = indices // LOGICAL_TILE_SIZE
            boundaries = np.flatnonzero(tiles[1:] != tiles[:-1]) + 1
            starts = np.concatenate((np.asarray([0]), boundaries))
            stops = np.concatenate((boundaries, np.asarray([len(indices)])))
            for left, right in zip(starts, stops, strict=True):
                tile = int(tiles[int(left)])
                segment = int(lookup[tile, shell])
                count = int(right - left)
                destination = int(cursors[segment])
                members[destination : destination + count] = (
                    indices[int(left) : int(right)] % LOGICAL_TILE_SIZE
                ).astype("<u2", copy=False)
                cursors[segment] += count
    if not np.array_equal(cursors.astype("<u8"), segment_member_starts[1:]):
        raise RuntimeError("projection CSR did not close its membership")
    tile_segment_starts = np.zeros(tile_count + 1, dtype="<u8")
    np.cumsum(
        np.bincount(pairs[:, 0], minlength=tile_count), out=tile_segment_starts[1:]
    )
    blocks_per_segment = (segment_counts + STAGE1_BLOCK_SIZE - 1) // STAGE1_BLOCK_SIZE
    segment_block_starts = np.empty(len(segment_counts) + 1, dtype="<u8")
    segment_block_starts[0] = 0
    np.cumsum(blocks_per_segment, out=segment_block_starts[1:])
    block_count = int(segment_block_starts[-1])
    member_block_starts = np.empty(block_count, dtype="<u8")
    member_block_counts = np.empty(block_count, dtype="<u2")
    member_block_segment_ids = np.empty(block_count, dtype="<u8")
    for segment in range(len(segment_counts)):
        member_left = int(segment_member_starts[segment])
        member_right = int(segment_member_starts[segment + 1])
        block_left = int(segment_block_starts[segment])
        block_right = int(segment_block_starts[segment + 1])
        for block in range(block_left, block_right):
            start = member_left + (block - block_left) * STAGE1_BLOCK_SIZE
            member_block_starts[block] = start
            member_block_counts[block] = min(STAGE1_BLOCK_SIZE, member_right - start)
            member_block_segment_ids[block] = segment
    tile_block_starts = segment_block_starts[tile_segment_starts.astype(np.int64)]
    seen = np.zeros(endpoint_count, dtype=np.int64)
    merges = np.empty(len(segment_counts), dtype="<u2")
    for segment, shell_value in enumerate(segment_shell_ids):
        shell = int(shell_value)
        merges[segment] = _trailing_ones(int(seen[shell]))
        seen[shell] += 1
    shell_segment_starts = np.empty(endpoint_count + 1, dtype="<u8")
    shell_segment_starts[0] = 0
    np.cumsum(seen, out=shell_segment_starts[1:])
    shell_segment_indices = np.empty(len(segment_shell_ids), dtype="<u8")
    shell_cursors = shell_segment_starts[:-1].astype(np.int64, copy=True)
    for segment, shell_value in enumerate(segment_shell_ids):
        shell = int(shell_value)
        destination = int(shell_cursors[shell])
        shell_segment_indices[destination] = segment
        shell_cursors[shell] += 1
    if not np.array_equal(shell_cursors.astype("<u8"), shell_segment_starts[1:]):
        raise RuntimeError("projection CSR did not close its shell segment schedule")
    return V1PackedLadderShellCSR(
        ladder_hash=ladder_hash,
        route_count=route_count,
        physical_mask_axis_hash=ladder.route_axis_hash,
        endpoint_action_hashes=ladder.action_hashes,
        tile_segment_starts=tile_segment_starts,
        segment_tile_ids=segment_tile_ids,
        segment_shell_ids=segment_shell_ids,
        segment_member_starts=segment_member_starts,
        member_tile_offsets=members,
        segment_block_starts=segment_block_starts,
        member_block_starts=member_block_starts,
        member_block_counts=member_block_counts,
        member_block_segment_ids=member_block_segment_ids,
        tile_block_starts=tile_block_starts,
        segment_stage2_merge_counts=merges,
        shell_segment_counts=seen.astype("<u8", copy=False),
        shell_segment_starts=shell_segment_starts,
        shell_segment_indices=shell_segment_indices,
        shell_hashes=tuple(shell_hashes),
        shell_removed_counts=tuple(shell_counts),
    )


@dataclass(frozen=True, slots=True)
class V1LegacyTupleLadderProjectionReplayAdapter:
    """Inert identity adapter for historical tuple-v1 replay metadata."""

    design_hash: str
    ladder_hash: str
    endpoint_action_hashes: tuple[str, ...]
    legacy_payload_hash: str
    schema: str = LEGACY_LADDER_PROJECTION_REPLAY_ADAPTER_SCHEMA

    @classmethod
    def from_design(
        cls, design: V1FrozenLadderCertificationDesign
    ) -> "V1LegacyTupleLadderProjectionReplayAdapter":
        if type(design) is not V1FrozenLadderCertificationDesign:
            raise TypeError("legacy replay adapter requires the exact design type")
        design.assert_integrity()
        if design.ladder.is_packed:
            raise ValueError("packed-v2 does not require a legacy replay adapter")
        payload = {
            "schema": LEGACY_LADDER_PROJECTION_REPLAY_ADAPTER_SCHEMA,
            "design_hash": design.design_hash,
            "ladder_hash": design.ladder.ladder_hash,
            "endpoint_action_hashes": design.ladder.action_hashes,
            "runtime_projection_eligible": False,
        }
        return cls(
            design_hash=design.design_hash,
            ladder_hash=design.ladder.ladder_hash,
            endpoint_action_hashes=design.ladder.action_hashes,
            legacy_payload_hash=content_hash(payload, prefix=_LEGACY_PREFIX),
        )

    def __post_init__(self) -> None:
        for name in ("design_hash", "ladder_hash", "legacy_payload_hash"):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "endpoint_action_hashes",
            _identity_axis(self.endpoint_action_hashes, name="legacy action axis"),
        )
        if self.schema != LEGACY_LADDER_PROJECTION_REPLAY_ADAPTER_SCHEMA:
            raise ValueError("legacy projection replay adapter schema is invalid")

    def refuse_runtime_projection(self) -> None:
        raise PermissionError("legacy tuple ladders are provenance-only")


@dataclass(frozen=True, slots=True)
class V1FrozenActionProjectionRequest:
    """Frozen pre-outcome request with distinct physical and ordered axes.

    Packed masks and the selected-shell CSR use ``physical_mask_axis_hash``.
    Provider route IDs and streamed route windows use
    ``ordered_score_route_axis_hash``.  Equal hash strings are permitted but
    never inferred or required.
    """

    design: V1FrozenLadderCertificationDesign = field(repr=False, compare=False)
    expected_context_ids: tuple[str, ...]
    expected_draw_ids: tuple[str, ...]
    expected_stratum_ids: tuple[str, ...]
    sealed_order_start_indices_by_stratum: tuple[tuple[str, int], ...]
    raw_logit_stream_identity_hash: str
    destination_transport_hash: str
    scorer_request_identity_hash: str
    acquisition_lineage_hash: str
    ordered_score_route_axis_hash: str
    registered_route_chunk_size: int
    schema: str = LADDER_PROJECTION_REQUEST_SCHEMA
    _csr: V1PackedLadderShellCSR = field(init=False, repr=False, compare=False)
    _request_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.design) is not V1FrozenLadderCertificationDesign:
            raise TypeError(
                "projection request requires the exact certification design"
            )
        self.design.assert_integrity()
        if self.design.projection_kernel_source_hash != projection_kernel_source_hash():
            raise ValueError(
                "certification design names another projection kernel source"
            )
        if not self.design.ladder.is_packed:
            raise PermissionError(
                "production projection requires packed ladder schema v2"
            )
        csr = build_v1_packed_ladder_shell_csr(self.design.ladder)
        contexts = _identity_axis(
            self.expected_context_ids, name="expected context axis"
        )
        draws = _identity_axis(self.expected_draw_ids, name="expected draw axis")
        strata = _identity_axis(self.expected_stratum_ids, name="expected stratum axis")
        if not contexts or not (len(contexts) == len(draws) == len(strata)):
            raise ValueError("projection request ledgers must be aligned and nonempty")
        if len(set(contexts)) != len(contexts) or len(set(draws)) != len(draws):
            raise ValueError("projection contexts and draws must be fresh and unique")
        ladder = self.design.ladder
        if set(contexts).intersection(ladder.discovery_context_ids) or set(
            draws
        ).intersection(ladder.discovery_draw_ids):
            raise ValueError("projection evidence must be fresh from discovery")
        design_strata = self.design.stratum_names
        starts = tuple(
            (
                _identity(name, name="sealed start stratum"),
                _nonnegative_int(index, name="sealed start index"),
            )
            for name, index in self.sealed_order_start_indices_by_stratum
        )
        if tuple(name for name, _ in starts) != design_strata:
            raise ValueError("sealed start axis differs from certification strata")
        design_ledgers = dict(self.design.certification_context_ids_by_stratum)
        start_map = dict(starts)
        if any(label not in design_strata for label in strata):
            raise ValueError("projection request names an unregistered stratum")
        for stratum in design_strata:
            selected = tuple(
                context
                for context, label in zip(contexts, strata, strict=True)
                if label == stratum
            )
            start = start_map[stratum]
            if selected != design_ledgers[stratum][start : start + len(selected)]:
                raise ValueError("projection contexts violate sealed stratum order")
        for name in (
            "raw_logit_stream_identity_hash",
            "destination_transport_hash",
            "scorer_request_identity_hash",
            "acquisition_lineage_hash",
            "ordered_score_route_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "registered_route_chunk_size",
            _positive_int(
                self.registered_route_chunk_size, name="registered route chunk size"
            ),
        )
        object.__setattr__(self, "expected_context_ids", contexts)
        object.__setattr__(self, "expected_draw_ids", draws)
        object.__setattr__(self, "expected_stratum_ids", strata)
        object.__setattr__(self, "sealed_order_start_indices_by_stratum", starts)
        object.__setattr__(self, "_csr", csr)
        if self.schema != LADDER_PROJECTION_REQUEST_SCHEMA:
            raise ValueError("projection request schema is invalid")
        self.assert_integrity()

    @property
    def csr(self) -> V1PackedLadderShellCSR:
        return self._csr

    @property
    def design_hash(self) -> str:
        return self.design.design_hash

    @property
    def ladder_hash(self) -> str:
        return self.design.ladder.ladder_hash

    @property
    def current_mask_hash(self) -> str:
        return self.design.ladder.current_mask_hash

    @property
    def endpoint_action_hashes(self) -> tuple[str, ...]:
        return self.design.ladder.action_hashes

    @property
    def shell_hashes(self) -> tuple[str, ...]:
        return self.csr.shell_hashes

    @property
    def shell_removed_counts(self) -> tuple[int, ...]:
        return self.csr.shell_removed_counts

    @property
    def endpoint_removed_counts(self) -> tuple[int, ...]:
        return self.design.ladder.endpoint_removed_counts

    @property
    def route_count(self) -> int:
        return self.design.ladder.route_count

    @property
    def physical_mask_axis_hash(self) -> str:
        return self.design.ladder.route_axis_hash

    @property
    def route_axis_hash(self) -> str:
        """Compatibility alias for :attr:`physical_mask_axis_hash` only."""

        return self.physical_mask_axis_hash

    @property
    def endpoint_count(self) -> int:
        return len(self.endpoint_action_hashes)

    @property
    def context_count(self) -> int:
        return len(self.expected_context_ids)

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "ladder_hash": self.ladder_hash,
            "csr_hash": self.csr.csr_hash,
            "current_mask_hash": self.current_mask_hash,
            "route_count": self.route_count,
            "physical_mask_axis_hash": self.physical_mask_axis_hash,
            "ordered_score_route_axis_hash": self.ordered_score_route_axis_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "shell_hashes": self.shell_hashes,
            "shell_removed_counts": self.shell_removed_counts,
            "expected_context_ids": self.expected_context_ids,
            "expected_draw_ids": self.expected_draw_ids,
            "expected_stratum_ids": self.expected_stratum_ids,
            "sealed_order_start_indices_by_stratum": self.sealed_order_start_indices_by_stratum,
            "raw_logit_stream_identity_hash": self.raw_logit_stream_identity_hash,
            "destination_transport_hash": self.destination_transport_hash,
            "scorer_request_identity_hash": self.scorer_request_identity_hash,
            "acquisition_lineage_hash": self.acquisition_lineage_hash,
            "registered_route_chunk_size": self.registered_route_chunk_size,
            "projection_kernel_source_hash": self.design.projection_kernel_source_hash,
            "projection_arithmetic_id": LADDER_PROJECTION_ARITHMETIC_ID,
            "coordinate": LADDER_PROJECTION_COORDINATE,
            "cotangent": LADDER_PROJECTION_COTANGENT,
            "orientation": LADDER_PROJECTION_ORIENTATION,
            "units": LADDER_PROJECTION_UNITS,
            "authority": LADDER_PROJECTION_AUTHORITY,
        }

    @property
    def request_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.design.assert_integrity()
        if self.design.projection_kernel_source_hash != projection_kernel_source_hash():
            raise ValueError("projection source changed after request freeze")
        self.csr.assert_integrity()
        if (
            self.csr.ladder_hash != self.design.ladder.ladder_hash
            or self.csr.physical_mask_axis_hash != self.physical_mask_axis_hash
        ):
            raise ValueError("projection CSR differs from its frozen ladder")
        observed = content_hash(self._hash_payload(), prefix=_REQUEST_PREFIX)
        if self._request_hash is not None and observed != self._request_hash:
            raise ValueError("projection request was tampered after hashing")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("projection request hash does not match")
        object.__setattr__(self, "_request_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1DirectionalActionProjectionRequest:
    """One pre-C corrected-evidence or literal-J projection request.

    A matched pair of these requests names the same directional secant design,
    packed ladder, context/draw prefix, and same-capture/VJP receipt.  The pair
    differs only in ``projection_role`` and therefore in which registered row
    stream is projected.  It is projection evidence only; finite secants and
    the F/Q/R certifier remain separate authorities.
    """

    design: V1DirectionalSecantDesign = field(repr=False, compare=False)
    expected_context_ids: tuple[str, ...]
    expected_draw_ids: tuple[str, ...]
    expected_stratum_ids: tuple[str, ...]
    sealed_order_start_indices_by_stratum: tuple[tuple[str, int], ...]
    projection_role: str
    corrected_evidence_stream_identity_hash: str
    literal_jacobian_stream_identity_hash: str
    same_capture_and_vjp_registration_hash: str
    destination_transport_hash: str
    scorer_request_identity_hash: str
    acquisition_lineage_hash: str
    ordered_score_route_axis_hash: str
    registered_route_chunk_size: int
    projection_kernel_source_hash: str
    schema: str = DIRECTIONAL_LADDER_PROJECTION_REQUEST_SCHEMA
    _csr: V1PackedLadderShellCSR = field(init=False, repr=False, compare=False)
    _request_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.design) is not V1DirectionalSecantDesign:
            raise TypeError(
                "directional projection requires the exact secant design"
            )
        self.design.assert_integrity()
        if self.projection_kernel_source_hash != projection_kernel_source_hash():
            raise ValueError("directional projection names another kernel source")
        ladder = self.design.ladder
        if not ladder.is_packed:
            raise PermissionError("directional projection requires a packed ladder")
        csr = build_v1_packed_ladder_shell_csr(ladder)
        contexts = _identity_axis(
            self.expected_context_ids, name="directional projection context axis"
        )
        draws = _identity_axis(
            self.expected_draw_ids, name="directional projection draw axis"
        )
        strata = _identity_axis(
            self.expected_stratum_ids, name="directional projection stratum axis"
        )
        if not contexts or not (len(contexts) == len(draws) == len(strata)):
            raise ValueError(
                "directional projection ledgers must be aligned and nonempty"
            )
        if len(set(contexts)) != len(contexts) or len(set(draws)) != len(draws):
            raise ValueError(
                "directional projection contexts and draws must be fresh and unique"
            )
        design_axes = self.design.certification_context_ids_by_stratum
        design_strata = tuple(name for name, _ in design_axes)
        starts = tuple(
            (
                _identity(name, name="directional sealed-start stratum"),
                _nonnegative_int(index, name="directional sealed-start index"),
            )
            for name, index in self.sealed_order_start_indices_by_stratum
        )
        if tuple(name for name, _ in starts) != design_strata:
            raise ValueError("directional sealed-start axis differs from C strata")
        design_ledgers = dict(design_axes)
        start_map = dict(starts)
        if any(label not in design_strata for label in strata):
            raise ValueError("directional projection names an unregistered stratum")
        for stratum in design_strata:
            selected = tuple(
                context
                for context, label in zip(contexts, strata, strict=True)
                if label == stratum
            )
            start = start_map[stratum]
            if selected != design_ledgers[stratum][start : start + len(selected)]:
                raise ValueError(
                    "directional projection contexts violate sealed C order"
                )
        role = _identity(self.projection_role, name="directional projection role")
        if role not in {"corrected_evidence", "literal_jacobian"}:
            raise ValueError("unknown directional dual-chart projection role")
        evidence = _identity(
            self.corrected_evidence_stream_identity_hash,
            name="corrected-evidence stream identity",
        )
        jacobian = _identity(
            self.literal_jacobian_stream_identity_hash,
            name="literal-Jacobian stream identity",
        )
        registration = _identity(
            self.same_capture_and_vjp_registration_hash,
            name="same-capture/VJP registration hash",
        )
        # D and C are independent registered permutations of the same complete
        # train population, not disjoint context folds.  A context may therefore
        # recur across roles.  What may not recur is the exact registered draw
        # identity, nor may the D sampling-law, C stream, and within-C same-VJP
        # registrations collapse onto one untyped alias.
        causal_registration_sources = (
            ladder.sampling_law_hash,
            self.design.certification_stream_commitment_hash,
            registration,
        )
        if len(set(causal_registration_sources)) != len(
            causal_registration_sources
        ):
            raise ValueError(
                "directional D/C causal registrations are missing, forged, or collapsed"
            )
        if set(draws).intersection(ladder.discovery_draw_ids):
            raise ValueError(
                "directional C evidence reuses an exact registered D draw identity"
            )
        if evidence == jacobian:
            raise ValueError("directional dual-chart stream identities collapsed")
        if ladder.raw_logit_stream_source_hash != evidence:
            raise ValueError(
                "directional ladder lost its corrected-evidence D-stream identity"
            )
        for name in (
            "destination_transport_hash",
            "scorer_request_identity_hash",
            "acquisition_lineage_hash",
            "ordered_score_route_axis_hash",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(self, "projection_role", role)
        object.__setattr__(
            self, "corrected_evidence_stream_identity_hash", evidence
        )
        object.__setattr__(
            self, "literal_jacobian_stream_identity_hash", jacobian
        )
        object.__setattr__(
            self, "same_capture_and_vjp_registration_hash", registration
        )
        object.__setattr__(
            self,
            "registered_route_chunk_size",
            _positive_int(
                self.registered_route_chunk_size,
                name="directional registered route chunk size",
            ),
        )
        object.__setattr__(self, "expected_context_ids", contexts)
        object.__setattr__(self, "expected_draw_ids", draws)
        object.__setattr__(self, "expected_stratum_ids", strata)
        object.__setattr__(self, "sealed_order_start_indices_by_stratum", starts)
        object.__setattr__(self, "_csr", csr)
        if self.schema != DIRECTIONAL_LADDER_PROJECTION_REQUEST_SCHEMA:
            raise ValueError("directional projection request schema is invalid")
        self.assert_integrity()

    @property
    def raw_logit_stream_identity_hash(self) -> str:
        return (
            self.corrected_evidence_stream_identity_hash
            if self.projection_role == "corrected_evidence"
            else self.literal_jacobian_stream_identity_hash
        )

    @property
    def row_semantics(self) -> str:
        return (
            "corrected_aumann_shapley_positive_removal_-G_phi_c_v1"
            if self.projection_role == "corrected_evidence"
            else "literal_relaxed_gate_positive_removal_-G_relu_indicator_c_v1"
        )

    @property
    def csr(self) -> V1PackedLadderShellCSR:
        return self._csr

    @property
    def design_hash(self) -> str:
        return self.design.design_hash

    @property
    def ladder_hash(self) -> str:
        return self.design.ladder.ladder_hash

    @property
    def current_mask_hash(self) -> str:
        return self.design.ladder.current_mask_hash

    @property
    def endpoint_action_hashes(self) -> tuple[str, ...]:
        return self.design.ladder.action_hashes

    @property
    def shell_hashes(self) -> tuple[str, ...]:
        return self.csr.shell_hashes

    @property
    def shell_removed_counts(self) -> tuple[int, ...]:
        return self.csr.shell_removed_counts

    @property
    def endpoint_removed_counts(self) -> tuple[int, ...]:
        return self.design.ladder.endpoint_removed_counts

    @property
    def route_count(self) -> int:
        return self.design.ladder.route_count

    @property
    def physical_mask_axis_hash(self) -> str:
        return self.design.ladder.route_axis_hash

    @property
    def route_axis_hash(self) -> str:
        return self.physical_mask_axis_hash

    @property
    def endpoint_count(self) -> int:
        return len(self.endpoint_action_hashes)

    @property
    def context_count(self) -> int:
        return len(self.expected_context_ids)

    @property
    def d_c_causal_registration_hash(self) -> str:
        """Bind independent D/C role owners without imposing context disjointness."""

        ladder = self.design.ladder
        return content_hash(
            {
                "schema": (
                    "partition_free_v1_directional_d_c_causal_registration_v1"
                ),
                "joint_fqr_family_registration_hash": (
                    self.design.joint_fqr_family_registration_hash
                ),
                "discovery": {
                    "sampling_law_hash": ladder.sampling_law_hash,
                    "discovery_evidence_hash": ladder.discovery_evidence_hash,
                    "freeze_receipt_hash": ladder.freeze_receipt_hash,
                },
                "certification": {
                    "certification_stream_commitment_hash": (
                        self.design.certification_stream_commitment_hash
                    ),
                    "same_capture_and_vjp_registration_hash": (
                        self.same_capture_and_vjp_registration_hash
                    ),
                },
                "same_complete_train_population_context_overlap_lawful": True,
                "exact_draw_record_reuse_forbidden": True,
            },
            prefix=_DIRECTIONAL_CAUSAL_REGISTRATION_PREFIX,
        )

    @property
    def discovery_causal_draw_axis_hash(self) -> str:
        ladder = self.design.ladder
        return _directional_causal_draw_axis_hash(
            role="discovery",
            causal_registration_hash=self.d_c_causal_registration_hash,
            context_ids=ladder.discovery_context_ids,
            draw_ids=ladder.discovery_draw_ids,
        )

    @property
    def certification_causal_draw_axis_hash(self) -> str:
        return _directional_causal_draw_axis_hash(
            role="certification",
            causal_registration_hash=self.d_c_causal_registration_hash,
            context_ids=self.expected_context_ids,
            draw_ids=self.expected_draw_ids,
        )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "design_hash": self.design_hash,
            "ladder_hash": self.ladder_hash,
            "csr_hash": self.csr.csr_hash,
            "chart_identity_hash": self.design.chart_identity_hash,
            "joint_fqr_family_registration_hash": (
                self.design.joint_fqr_family_registration_hash
            ),
            "current_mask_hash": self.current_mask_hash,
            "route_count": self.route_count,
            "physical_mask_axis_hash": self.physical_mask_axis_hash,
            "ordered_score_route_axis_hash": self.ordered_score_route_axis_hash,
            "endpoint_action_hashes": self.endpoint_action_hashes,
            "shell_hashes": self.shell_hashes,
            "shell_removed_counts": self.shell_removed_counts,
            "expected_context_ids": self.expected_context_ids,
            "expected_draw_ids": self.expected_draw_ids,
            "expected_stratum_ids": self.expected_stratum_ids,
            "sealed_order_start_indices_by_stratum": (
                self.sealed_order_start_indices_by_stratum
            ),
            "projection_role": self.projection_role,
            "row_semantics": self.row_semantics,
            "raw_logit_stream_identity_hash": self.raw_logit_stream_identity_hash,
            "corrected_evidence_stream_identity_hash": (
                self.corrected_evidence_stream_identity_hash
            ),
            "literal_jacobian_stream_identity_hash": (
                self.literal_jacobian_stream_identity_hash
            ),
            "same_capture_and_vjp_registration_hash": (
                self.same_capture_and_vjp_registration_hash
            ),
            "d_c_causal_registration_hash": self.d_c_causal_registration_hash,
            "discovery_causal_draw_axis_hash": (
                self.discovery_causal_draw_axis_hash
            ),
            "certification_causal_draw_axis_hash": (
                self.certification_causal_draw_axis_hash
            ),
            "destination_transport_hash": self.destination_transport_hash,
            "scorer_request_identity_hash": self.scorer_request_identity_hash,
            "acquisition_lineage_hash": self.acquisition_lineage_hash,
            "registered_route_chunk_size": self.registered_route_chunk_size,
            "projection_kernel_source_hash": self.projection_kernel_source_hash,
            "projection_arithmetic_id": LADDER_PROJECTION_ARITHMETIC_ID,
            "coordinate": LADDER_PROJECTION_COORDINATE,
            "cotangent": LADDER_PROJECTION_COTANGENT,
            "units": LADDER_PROJECTION_UNITS,
            "same_capture_and_vjp_required": True,
            "same_complete_train_population_context_overlap_lawful": True,
            "exact_draw_record_reuse_forbidden": True,
            "paired_projection_only": True,
            "fqr_certification_authority": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
        }

    @property
    def request_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.design.assert_integrity()
        if self.projection_kernel_source_hash != projection_kernel_source_hash():
            raise ValueError("directional projection source changed after freeze")
        self.csr.assert_integrity()
        if (
            self.csr.ladder_hash != self.design.ladder.ladder_hash
            or self.csr.physical_mask_axis_hash != self.physical_mask_axis_hash
        ):
            raise ValueError("directional projection CSR differs from its ladder")
        observed = content_hash(
            self._hash_payload(), prefix=_DIRECTIONAL_REQUEST_PREFIX
        )
        if self._request_hash is not None and observed != self._request_hash:
            raise ValueError("directional projection request was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("directional projection request hash does not match")
        object.__setattr__(self, "_request_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1LadderProjectionPacketReceipt:
    """Execution receipt for one scorer context packet."""

    request_hash: str
    packet_index: int
    context_start: int
    context_stop: int
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    stratum_ids: tuple[str, ...]
    route_count: int
    physical_mask_axis_hash: str
    ordered_score_route_axis_hash: str
    route_windows: tuple[tuple[int, int], ...]
    route_window_ledger_hash: str
    signed_projection_matrix_hash: str
    observed_absolute_sum_matrix_hash: str
    numerical_backend: str
    peak_registered_device_bytes: int
    projection_host_materialization_bytes: int
    device_to_host_bytes: int
    device_identity: str
    schema: str = LADDER_PROJECTION_PACKET_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "request_hash",
            "physical_mask_axis_hash",
            "ordered_score_route_axis_hash",
            "route_window_ledger_hash",
            "signed_projection_matrix_hash",
            "observed_absolute_sum_matrix_hash",
            "numerical_backend",
            "device_identity",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        packet = _nonnegative_int(self.packet_index, name="packet index")
        start = _nonnegative_int(self.context_start, name="context start")
        stop = _positive_int(self.context_stop, name="context stop")
        route_count = _positive_int(self.route_count, name="route count")
        contexts = _identity_axis(self.context_ids, name="packet context axis")
        draws = _identity_axis(self.draw_ids, name="packet draw axis")
        strata = _identity_axis(self.stratum_ids, name="packet stratum axis")
        if (
            stop <= start
            or stop - start != len(contexts)
            or not (len(contexts) == len(draws) == len(strata))
        ):
            raise ValueError("packet context ledgers are not aligned")
        windows = tuple((int(left), int(right)) for left, right in self.route_windows)
        if (
            not windows
            or windows[0][0] != 0
            or windows[-1][1] != route_count
            or any(
                left < 0 or right <= left or (index and left != windows[index - 1][1])
                for index, (left, right) in enumerate(windows)
            )
        ):
            raise ValueError("packet route windows are not one contiguous axis")
        expected_window_hash = content_hash(
            {
                "request_hash": self.request_hash,
                "packet_index": packet,
                "route_count": route_count,
                "ordered_score_route_axis_hash": self.ordered_score_route_axis_hash,
                "route_windows": windows,
            },
            prefix=_WINDOW_PREFIX,
        )
        if expected_window_hash != self.route_window_ledger_hash:
            raise ValueError("packet route-window ledger hash does not replay")
        for name in (
            "peak_registered_device_bytes",
            "projection_host_materialization_bytes",
            "device_to_host_bytes",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name=name)
            )
        if self.schema != LADDER_PROJECTION_PACKET_RECEIPT_SCHEMA:
            raise ValueError("projection packet receipt schema is invalid")
        object.__setattr__(self, "packet_index", packet)
        object.__setattr__(self, "context_start", start)
        object.__setattr__(self, "context_stop", stop)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "stratum_ids", strata)
        object.__setattr__(self, "route_windows", windows)
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "request_hash": self.request_hash,
            "packet_index": self.packet_index,
            "context_start": self.context_start,
            "context_stop": self.context_stop,
            "context_ids": self.context_ids,
            "draw_ids": self.draw_ids,
            "stratum_ids": self.stratum_ids,
            "route_count": self.route_count,
            "physical_mask_axis_hash": self.physical_mask_axis_hash,
            "ordered_score_route_axis_hash": self.ordered_score_route_axis_hash,
            "route_windows": self.route_windows,
            "route_window_ledger_hash": self.route_window_ledger_hash,
            "signed_projection_matrix_hash": self.signed_projection_matrix_hash,
            "observed_absolute_sum_matrix_hash": self.observed_absolute_sum_matrix_hash,
            "numerical_backend": self.numerical_backend,
            "projection_arithmetic_id": LADDER_PROJECTION_ARITHMETIC_ID,
            "peak_registered_device_bytes": self.peak_registered_device_bytes,
            "projection_host_materialization_bytes": self.projection_host_materialization_bytes,
            "device_to_host_bytes": self.device_to_host_bytes,
            "device_identity": self.device_identity,
            "memory_scope": LADDER_PROJECTION_MEMORY_SCOPE,
            "authority": LADDER_PROJECTION_AUTHORITY,
        }

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_PACKET_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("projection packet receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("projection packet receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1LadderProjectionDerivationReceipt:
    """Complete projection receipt, still prior to provider closure."""

    request_hash: str
    design_hash: str
    ladder_hash: str
    csr_hash: str
    current_mask_hash: str
    route_count: int
    physical_mask_axis_hash: str
    ordered_score_route_axis_hash: str
    raw_logit_stream_source_hash: str
    raw_logit_stream_identity_hash: str
    destination_transport_hash: str
    scorer_request_identity_hash: str
    acquisition_lineage_hash: str
    context_ids: tuple[str, ...]
    draw_ids: tuple[str, ...]
    stratum_ids: tuple[str, ...]
    endpoint_action_hashes: tuple[str, ...]
    shell_hashes: tuple[str, ...]
    shell_removed_counts: tuple[int, ...]
    packet_receipt_hashes: tuple[str, ...]
    route_window_ledger_hashes: tuple[str, ...]
    signed_projection_matrix_hash: str
    observed_absolute_sum_matrix_hash: str
    absolute_input_sum_upper_matrix_hash: str
    numerical_radius_matrix_hash: str
    numerical_lower_matrix_hash: str
    numerical_upper_matrix_hash: str
    projection_kernel_source_hash: str
    numerical_backend: str
    peak_registered_device_bytes: int
    total_projection_host_materialization_bytes: int
    exact_device_to_host_bytes: int
    packet_count: int
    route_chunk_update_count: int
    estimator_rows_exactly_projected: bool = True
    real_arithmetic_exactness_claimed: bool = False
    numerical_enclosure_provided: bool = True
    scalar_projection_derivation_authority: bool = True
    fq_certification_authority: bool = False
    score_order_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = LADDER_PROJECTION_DERIVATION_RECEIPT_SCHEMA
    _receipt_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        identities = (
            "request_hash",
            "design_hash",
            "ladder_hash",
            "csr_hash",
            "current_mask_hash",
            "physical_mask_axis_hash",
            "ordered_score_route_axis_hash",
            "raw_logit_stream_source_hash",
            "raw_logit_stream_identity_hash",
            "destination_transport_hash",
            "scorer_request_identity_hash",
            "acquisition_lineage_hash",
            "signed_projection_matrix_hash",
            "observed_absolute_sum_matrix_hash",
            "absolute_input_sum_upper_matrix_hash",
            "numerical_radius_matrix_hash",
            "numerical_lower_matrix_hash",
            "numerical_upper_matrix_hash",
            "projection_kernel_source_hash",
            "numerical_backend",
        )
        for name in identities:
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        object.__setattr__(
            self, "route_count", _positive_int(self.route_count, name="route count")
        )
        object.__setattr__(
            self, "packet_count", _positive_int(self.packet_count, name="packet count")
        )
        object.__setattr__(
            self,
            "route_chunk_update_count",
            _positive_int(self.route_chunk_update_count, name="route update count"),
        )
        for name in (
            "peak_registered_device_bytes",
            "total_projection_host_materialization_bytes",
            "exact_device_to_host_bytes",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name=name)
            )
        contexts = _identity_axis(self.context_ids, name="derivation context axis")
        draws = _identity_axis(self.draw_ids, name="derivation draw axis")
        strata = _identity_axis(self.stratum_ids, name="derivation stratum axis")
        actions = _identity_axis(
            self.endpoint_action_hashes, name="derivation action axis"
        )
        shells = _identity_axis(self.shell_hashes, name="derivation shell axis")
        shell_counts = tuple(
            _positive_int(value, name="derivation shell count")
            for value in self.shell_removed_counts
        )
        packet_hashes = _identity_axis(
            self.packet_receipt_hashes, name="packet receipt axis"
        )
        window_hashes = _identity_axis(
            self.route_window_ledger_hashes, name="window receipt axis"
        )
        if not contexts or not (len(contexts) == len(draws) == len(strata)):
            raise ValueError("projection derivation ledgers are not aligned")
        if not actions or not (len(actions) == len(shells) == len(shell_counts)):
            raise ValueError("projection derivation action axes are not aligned")
        if (
            len(packet_hashes) != self.packet_count
            or len(window_hashes) != self.packet_count
        ):
            raise ValueError("projection derivation packet axes are not aligned")
        if not (
            self.estimator_rows_exactly_projected is True
            and self.real_arithmetic_exactness_claimed is False
            and self.numerical_enclosure_provided is True
            and self.scalar_projection_derivation_authority is True
        ) or any(
            (
                self.fq_certification_authority,
                self.score_order_authority,
                self.bce_oracle_authority,
                self.mask_commit_authority,
            )
        ):
            raise ValueError("projection derivation exceeded scalar authority")
        if self.schema != LADDER_PROJECTION_DERIVATION_RECEIPT_SCHEMA:
            raise ValueError("projection derivation receipt schema is invalid")
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "draw_ids", draws)
        object.__setattr__(self, "stratum_ids", strata)
        object.__setattr__(self, "endpoint_action_hashes", actions)
        object.__setattr__(self, "shell_hashes", shells)
        object.__setattr__(self, "shell_removed_counts", shell_counts)
        object.__setattr__(self, "packet_receipt_hashes", packet_hashes)
        object.__setattr__(self, "route_window_ledger_hashes", window_hashes)
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        names = (
            "schema",
            "request_hash",
            "design_hash",
            "ladder_hash",
            "csr_hash",
            "current_mask_hash",
            "route_count",
            "physical_mask_axis_hash",
            "ordered_score_route_axis_hash",
            "raw_logit_stream_source_hash",
            "raw_logit_stream_identity_hash",
            "destination_transport_hash",
            "scorer_request_identity_hash",
            "acquisition_lineage_hash",
            "context_ids",
            "draw_ids",
            "stratum_ids",
            "endpoint_action_hashes",
            "shell_hashes",
            "shell_removed_counts",
            "packet_receipt_hashes",
            "route_window_ledger_hashes",
            "signed_projection_matrix_hash",
            "observed_absolute_sum_matrix_hash",
            "absolute_input_sum_upper_matrix_hash",
            "numerical_radius_matrix_hash",
            "numerical_lower_matrix_hash",
            "numerical_upper_matrix_hash",
            "projection_kernel_source_hash",
            "numerical_backend",
            "peak_registered_device_bytes",
            "total_projection_host_materialization_bytes",
            "exact_device_to_host_bytes",
            "packet_count",
            "route_chunk_update_count",
            "estimator_rows_exactly_projected",
            "real_arithmetic_exactness_claimed",
            "numerical_enclosure_provided",
            "scalar_projection_derivation_authority",
            "fq_certification_authority",
            "score_order_authority",
            "bce_oracle_authority",
            "mask_commit_authority",
        )
        return {name: getattr(self, name) for name in names} | {
            "projection_arithmetic_id": LADDER_PROJECTION_ARITHMETIC_ID,
            "coordinate": LADDER_PROJECTION_COORDINATE,
            "cotangent": LADDER_PROJECTION_COTANGENT,
            "orientation": LADDER_PROJECTION_ORIENTATION,
            "units": LADDER_PROJECTION_UNITS,
            "memory_scope": LADDER_PROJECTION_MEMORY_SCOPE,
            "authority": LADDER_PROJECTION_AUTHORITY,
        }

    @property
    def route_axis_hash(self) -> str:
        """Compatibility alias for :attr:`physical_mask_axis_hash` only."""

        return self.physical_mask_axis_hash

    @property
    def receipt_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_RECEIPT_PREFIX)
        if self._receipt_hash is not None and observed != self._receipt_hash:
            raise ValueError("projection derivation receipt was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("projection derivation receipt hash does not match")
        object.__setattr__(self, "_receipt_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1LadderProjectionResult:
    receipt: V1LadderProjectionDerivationReceipt
    packet_receipts: tuple[V1LadderProjectionPacketReceipt, ...]
    signed_projection_matrix: np.ndarray = field(repr=False, compare=False)
    observed_absolute_sum_matrix: np.ndarray = field(repr=False, compare=False)
    absolute_input_sum_upper_matrix: np.ndarray = field(repr=False, compare=False)
    numerical_radius_matrix: np.ndarray = field(repr=False, compare=False)
    numerical_lower_matrix: np.ndarray = field(repr=False, compare=False)
    numerical_upper_matrix: np.ndarray = field(repr=False, compare=False)
    schema: str = LADDER_PROJECTION_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.receipt) is not V1LadderProjectionDerivationReceipt:
            raise TypeError("projection result requires its typed receipt")
        self.receipt.assert_integrity()
        packets = tuple(self.packet_receipts)
        if any(type(item) is not V1LadderProjectionPacketReceipt for item in packets):
            raise TypeError("projection packet receipts must be typed")
        if (
            tuple(item.receipt_hash for item in packets)
            != self.receipt.packet_receipt_hashes
            or any(
                item.physical_mask_axis_hash
                != self.receipt.physical_mask_axis_hash
                or item.ordered_score_route_axis_hash
                != self.receipt.ordered_score_route_axis_hash
                for item in packets
            )
        ):
            raise ValueError(
                "projection packet receipts or coordinate axes differ from derivation"
            )
        shape = (
            len(self.receipt.context_ids),
            len(self.receipt.endpoint_action_hashes),
        )
        specifications = (
            ("signed_projection_matrix", self.receipt.signed_projection_matrix_hash),
            (
                "observed_absolute_sum_matrix",
                self.receipt.observed_absolute_sum_matrix_hash,
            ),
            (
                "absolute_input_sum_upper_matrix",
                self.receipt.absolute_input_sum_upper_matrix_hash,
            ),
            ("numerical_radius_matrix", self.receipt.numerical_radius_matrix_hash),
            ("numerical_lower_matrix", self.receipt.numerical_lower_matrix_hash),
            ("numerical_upper_matrix", self.receipt.numerical_upper_matrix_hash),
        )
        matrices: dict[str, np.ndarray] = {}
        for name, expected in specifications:
            matrix, observed = _canonical_matrix(
                getattr(self, name), shape=shape, name=name
            )
            if observed != expected:
                raise ValueError(f"{name} differs from projection receipt")
            matrices[name] = matrix
        center = matrices["signed_projection_matrix"]
        observed_abs = matrices["observed_absolute_sum_matrix"]
        abs_upper = matrices["absolute_input_sum_upper_matrix"]
        radius = matrices["numerical_radius_matrix"]
        lower = matrices["numerical_lower_matrix"]
        upper = matrices["numerical_upper_matrix"]
        if (
            np.any(observed_abs < 0.0)
            or np.any(abs_upper < observed_abs)
            or np.any(abs_upper < np.abs(center))
            or np.any(radius < 0.0)
            or np.any(lower > center)
            or np.any(upper < center)
            or np.any(lower > upper)
        ):
            raise ValueError("projection numerical enclosure is inconsistent")
        if self.schema != LADDER_PROJECTION_RESULT_SCHEMA:
            raise ValueError("projection result schema is invalid")
        object.__setattr__(self, "packet_receipts", packets)
        for name, matrix in matrices.items():
            object.__setattr__(self, name, matrix)


@dataclass(frozen=True, slots=True)
class V1LadderProjectionAcquisitionClosure:
    """Typed joint closure over scorer execution and projection derivation."""

    request_hash: str
    design_hash: str
    ladder_hash: str
    projection_receipt_hash: str
    scorer_receipt_hash: str
    scorer_receipt_schema: str
    scorer_request_identity_hash: str
    acquisition_lineage_hash: str
    physical_mask_axis_hash: str
    ordered_score_route_axis_hash: str
    context_axis_hash: str
    draw_axis_hash: str
    stratum_axis_hash: str
    signed_projection_matrix_hash: str
    absolute_input_sum_upper_matrix_hash: str
    numerical_radius_matrix_hash: str
    numerical_lower_matrix_hash: str
    numerical_upper_matrix_hash: str
    estimator_rows_exactly_projected: bool = True
    numerical_enclosure_bound: bool = True
    certifier_record_proof_authority: bool = True
    fq_certification_authority: bool = False
    score_order_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    schema: str = LADDER_PROJECTION_ACQUISITION_CLOSURE_SCHEMA
    _closure_hash: str | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        names = (
            "request_hash",
            "design_hash",
            "ladder_hash",
            "projection_receipt_hash",
            "scorer_receipt_hash",
            "scorer_receipt_schema",
            "scorer_request_identity_hash",
            "acquisition_lineage_hash",
            "physical_mask_axis_hash",
            "ordered_score_route_axis_hash",
            "context_axis_hash",
            "draw_axis_hash",
            "stratum_axis_hash",
            "signed_projection_matrix_hash",
            "absolute_input_sum_upper_matrix_hash",
            "numerical_radius_matrix_hash",
            "numerical_lower_matrix_hash",
            "numerical_upper_matrix_hash",
        )
        for name in names:
            object.__setattr__(self, name, _identity(getattr(self, name), name=name))
        if not (
            self.estimator_rows_exactly_projected is True
            and self.numerical_enclosure_bound is True
            and self.certifier_record_proof_authority is True
        ) or any(
            (
                self.fq_certification_authority,
                self.score_order_authority,
                self.bce_oracle_authority,
                self.mask_commit_authority,
            )
        ):
            raise ValueError("projection acquisition closure exceeded its authority")
        if self.schema != LADDER_PROJECTION_ACQUISITION_CLOSURE_SCHEMA:
            raise ValueError("projection acquisition closure schema is invalid")
        self.assert_integrity()

    def _hash_payload(self) -> dict[str, object]:
        names = (
            "schema",
            "request_hash",
            "design_hash",
            "ladder_hash",
            "projection_receipt_hash",
            "scorer_receipt_hash",
            "scorer_receipt_schema",
            "scorer_request_identity_hash",
            "acquisition_lineage_hash",
            "physical_mask_axis_hash",
            "ordered_score_route_axis_hash",
            "context_axis_hash",
            "draw_axis_hash",
            "stratum_axis_hash",
            "signed_projection_matrix_hash",
            "absolute_input_sum_upper_matrix_hash",
            "numerical_radius_matrix_hash",
            "numerical_lower_matrix_hash",
            "numerical_upper_matrix_hash",
            "estimator_rows_exactly_projected",
            "numerical_enclosure_bound",
            "certifier_record_proof_authority",
            "fq_certification_authority",
            "score_order_authority",
            "bce_oracle_authority",
            "mask_commit_authority",
        )
        return {name: getattr(self, name) for name in names}

    @property
    def closure_hash(self) -> str:
        return self.assert_integrity()

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        observed = content_hash(self._hash_payload(), prefix=_CLOSURE_PREFIX)
        if self._closure_hash is not None and observed != self._closure_hash:
            raise ValueError("projection acquisition closure was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("projection acquisition closure hash does not match")
        object.__setattr__(self, "_closure_hash", observed)
        return observed


@dataclass(frozen=True, slots=True)
class V1LadderProjectionAcquisition:
    result: V1LadderProjectionResult
    closure: V1LadderProjectionAcquisitionClosure
    projection_records: tuple[V1FrozenLadderProjectionRecord, ...]
    schema: str = LADDER_PROJECTION_ACQUISITION_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.result) is not V1LadderProjectionResult
            or type(self.closure) is not V1LadderProjectionAcquisitionClosure
        ):
            raise TypeError("projection acquisition requires typed result and closure")
        closure_hash = self.closure.assert_integrity()
        receipt = self.result.receipt
        if (
            self.closure.projection_receipt_hash != receipt.receipt_hash
            or self.closure.physical_mask_axis_hash
            != receipt.physical_mask_axis_hash
            or self.closure.ordered_score_route_axis_hash
            != receipt.ordered_score_route_axis_hash
            or self.closure.signed_projection_matrix_hash
            != receipt.signed_projection_matrix_hash
            or self.closure.absolute_input_sum_upper_matrix_hash
            != receipt.absolute_input_sum_upper_matrix_hash
            or self.closure.numerical_radius_matrix_hash
            != receipt.numerical_radius_matrix_hash
            or self.closure.numerical_lower_matrix_hash
            != receipt.numerical_lower_matrix_hash
            or self.closure.numerical_upper_matrix_hash
            != receipt.numerical_upper_matrix_hash
        ):
            raise ValueError("projection closure differs from result")
        records = tuple(self.projection_records)
        if len(records) != len(receipt.context_ids):
            raise ValueError("projection record ledger has the wrong length")
        for row, record in enumerate(records):
            if type(record) is not V1FrozenLadderProjectionRecord:
                raise TypeError("projection records must use the certifier type")
            record.assert_integrity()
            if (
                record.design_hash != receipt.design_hash
                or record.ladder_hash != receipt.ladder_hash
                or record.context_id != receipt.context_ids[row]
                or record.draw_id != receipt.draw_ids[row]
                or record.stratum_id != receipt.stratum_ids[row]
                or record.projection_proof_hash != closure_hash
                or record.endpoint_action_hashes != receipt.endpoint_action_hashes
                or record.signed_projection_by_endpoint
                != tuple(
                    float(value) for value in self.result.signed_projection_matrix[row]
                )
                or record.numerical_radius_by_endpoint
                != tuple(
                    float(value) for value in self.result.numerical_radius_matrix[row]
                )
            ):
                raise ValueError("projection record is not bound to closure")
        if self.schema != LADDER_PROJECTION_ACQUISITION_SCHEMA:
            raise ValueError("projection acquisition schema is invalid")
        object.__setattr__(self, "projection_records", records)


@dataclass(frozen=True, slots=True)
class _ClosedPacket:
    receipt: V1LadderProjectionPacketReceipt
    center: np.ndarray = field(repr=False, compare=False)
    observed_abs: np.ndarray = field(repr=False, compare=False)


def _pairwise_reduce_axis1(values: Any) -> Any:
    """Reduce ``[B, N]`` through a deterministic padded binary tree."""

    if torch is None or not isinstance(values, torch.Tensor):
        raise TypeError("fixed reduction requires a torch tensor")
    if values.ndim != 2 or int(values.shape[1]) <= 0:
        raise ValueError("fixed reduction requires a nonempty route axis")
    width = 1 << (int(values.shape[1]) - 1).bit_length()
    work = torch.zeros(
        (int(values.shape[0]), width),
        dtype=torch.float64,
        device=values.device,
    )
    work[:, : int(values.shape[1])].copy_(values)
    while width > 1:
        work = work[:, :width:2] + work[:, 1:width:2]
        width //= 2
    return work[:, 0]


def _csr_shell_physical_indices(csr: V1PackedLadderShellCSR) -> tuple[np.ndarray, ...]:
    """Return sorted physical route indices for each disjoint ladder shell."""

    csr.assert_integrity()
    shells: list[np.ndarray] = []
    for shell in range(csr.endpoint_count):
        segment_left = int(csr.shell_segment_starts[shell])
        segment_right = int(csr.shell_segment_starts[shell + 1])
        physical_parts: list[np.ndarray] = []
        for raw_segment in csr.shell_segment_indices[segment_left:segment_right]:
            segment = int(raw_segment)
            tile = int(csr.segment_tile_ids[segment])
            member_left = int(csr.segment_member_starts[segment])
            member_right = int(csr.segment_member_starts[segment + 1])
            offsets = np.asarray(
                csr.member_tile_offsets[member_left:member_right],
                dtype=np.uint64,
            )
            physical_parts.append(offsets + np.uint64(tile * LOGICAL_TILE_SIZE))
        if not physical_parts:
            raise ValueError("projection shell has no physical route members")
        physical = np.concatenate(physical_parts).astype(np.int64, copy=False)
        if (
            physical.ndim != 1
            or int(physical.size) != int(csr.shell_removed_counts[shell])
            or int(physical[-1]) >= csr.route_count
            or (physical.size > 1 and np.any(physical[1:] <= physical[:-1]))
        ):
            raise ValueError("projection shell physical indices are not canonical")
        physical.setflags(write=False)
        shells.append(physical)
    return tuple(shells)


def _upward_nonnegative_add(left: float, right: float) -> float:
    """Return an outward binary64 upper bound for ``left + right``."""

    result = float(left) + float(right)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("projection enclosure addition left its finite cone")
    return math.nextafter(result, math.inf)


def _upward_nonnegative_multiply(left: float, right: float) -> float:
    """Return an outward binary64 upper bound for ``left * right``."""

    result = float(left) * float(right)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("projection enclosure product left its finite cone")
    return math.nextafter(result, math.inf)


def _upward_nonnegative_divide(numerator: float, denominator: float) -> float:
    """Return an outward binary64 upper bound for a positive quotient."""

    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("projection enclosure denominator must be finite and positive")
    result = float(numerator) / float(denominator)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("projection enclosure quotient left its finite cone")
    return math.nextafter(result, math.inf)


def _outward_enclosure(
    center: np.ndarray,
    observed_abs: np.ndarray,
    *,
    endpoint_removed_counts: tuple[int, ...],
    shell_segment_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enclose exact real sums of the emitted binary64 estimator values."""

    if center.shape != observed_abs.shape or center.shape[1] != len(
        endpoint_removed_counts
    ):
        raise ValueError("numerical enclosure axes are inconsistent")
    absolute_upper = np.empty_like(center)
    radius = np.empty_like(center)
    lower = np.empty_like(center)
    upper = np.empty_like(center)
    maximum_segments = int(np.max(shell_segment_counts, initial=1))
    stage2_depth = 2 * int(math.ceil(math.log2(maximum_segments + 1)))
    for endpoint, count in enumerate(endpoint_removed_counts):
        depth = 16 + stage2_depth + endpoint
        depth_roundoff = _upward_nonnegative_multiply(
            float(depth), _UNIT_ROUNDOFF
        )
        gamma_denominator = math.nextafter(1.0 - depth_roundoff, -math.inf)
        gamma = _upward_nonnegative_divide(
            depth_roundoff, gamma_denominator
        )
        eta_scale = _upward_nonnegative_multiply(4.0, float(count))
        eta = _upward_nonnegative_multiply(
            eta_scale, float(np.finfo(np.float64).tiny)
        )
        absolute_denominator = math.nextafter(1.0 - gamma, -math.inf)
        for row in range(center.shape[0]):
            observed = float(observed_abs[row, endpoint])
            abs_bound = _upward_nonnegative_divide(
                _upward_nonnegative_add(observed, eta),
                absolute_denominator,
            )
            error = _upward_nonnegative_add(
                _upward_nonnegative_multiply(gamma, abs_bound),
                eta,
            )
            abs_bound = max(
                abs_bound,
                _upward_nonnegative_add(
                    abs(float(center[row, endpoint])), error
                ),
            )
            low = math.nextafter(float(center[row, endpoint]) - error, -math.inf)
            high = math.nextafter(float(center[row, endpoint]) + error, math.inf)
            if not all(math.isfinite(value) for value in (abs_bound, error, low, high)):
                raise ValueError("projection numerical enclosure overflowed binary64")
            absolute_upper[row, endpoint] = abs_bound
            radius[row, endpoint] = error
            lower[row, endpoint] = low
            upper[row, endpoint] = high
    return absolute_upper, radius, lower, upper


class V1LadderProjectionGPUConsumer(_RouteChunkedLossOnlyConsumer):
    """Nominal packed-v2 sink for raw-logit positive-removal chunks."""

    positive_removal_stream_coordinate = LADDER_PROJECTION_COORDINATE

    def __init__(
        self,
        request: V1FrozenActionProjectionRequest | V1DirectionalActionProjectionRequest,
        *,
        route_ids: Sequence[str],
        device: Any = "cpu",
    ) -> None:
        if torch is None:
            raise RuntimeError("frozen-action projection requires PyTorch")
        if type(request) not in {
            V1FrozenActionProjectionRequest,
            V1DirectionalActionProjectionRequest,
        }:
            raise TypeError("projection consumer requires an exact request type")
        request.assert_integrity()
        if type(route_ids) not in {tuple, ConstructiveOrdinalRouteAxis}:
            raise TypeError(
                "projection route IDs must be an immutable tuple or compact "
                "ConstructiveOrdinalRouteAxis"
            )
        if (
            len(route_ids) != request.route_count
            or ordered_string_axis_hash(
                route_ids, name="frozen-action projection ordered score-route axis"
            )
            != request.ordered_score_route_axis_hash
        ):
            raise ValueError(
                "projection ordered score-route IDs differ from frozen request"
            )
        self.request = request
        # Preserve the validated constructive 57M-route axis.  Tuple expansion
        # would manufacture one Python string object per route before any
        # projection row is consumed.
        self.route_ids = route_ids
        self.device = _canonical_device(device)
        self.loss_pullback_hash = request.raw_logit_stream_identity_hash
        self.head_ids = ("positive_removal:raw_target_logit",)
        self.chunk_size = request.registered_route_chunk_size
        self._route_chunk_bound = False
        self._device_identity = _device_identity(self.device)
        self._numerical_backend = (
            "torch_cuda_selected_shell_fixed_tree_v2"
            if self.device.type == "cuda"
            else "torch_cpu_selected_shell_fixed_tree_v2"
        )
        self._next_context = 0
        self._packet_index = 0
        self._packet_context_start = 0
        self._packet_context_ids: tuple[str, ...] = ()
        self._packet_draw_ids: tuple[str, ...] = ()
        self._packet_stratum_ids: tuple[str, ...] = ()
        self._packet_shell_rows: tuple[Any, ...] = ()
        self._packet_state_center: Any | None = None
        self._packet_state_abs: Any | None = None
        self._packet_nonfinite_marker: Any | None = None
        self._packet_next_route = 0
        self._packet_windows: list[tuple[int, int]] = []
        self._packet_peak_registered_bytes = 0
        self._packet_base_registered_bytes = 0
        self._closed_packets: list[_ClosedPacket] = []
        self._packet_open = False
        self._failed = False
        self._failure_reason: str | None = None
        self._closed_result: V1LadderProjectionResult | None = None
        self._closed_acquisition: V1LadderProjectionAcquisition | None = None

        self._device_csr_bytes = 0
        self._shell_physical_indices = _csr_shell_physical_indices(request.csr)
        self._device_shell_indices: tuple[Any, ...] = tuple(
            torch.as_tensor(
                np.array(indices, dtype=np.int64, copy=True),
                dtype=torch.int64,
                device=self.device,
            )
            for indices in self._shell_physical_indices
        )
        self._device_csr_bytes += sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in self._device_shell_indices
        )
    @property
    def is_failed(self) -> bool:
        return self._failed

    @property
    def delivered_record_count(self) -> int:
        return (
            0
            if self._closed_acquisition is None
            else len(self._closed_acquisition.projection_records)
        )

    def bind_route_chunk_size(self, chunk_size: int) -> None:
        """Accept only an idempotent exact comparison with the sealed ABI."""

        try:
            self._assert_live()
            observed = _positive_int(chunk_size, name="provider route chunk size")
            if observed != self.request.registered_route_chunk_size:
                raise ValueError(
                    "provider route chunk size differs from frozen request"
                )
            if self._packet_open or self._next_context:
                raise RuntimeError("cannot bind route chunks after acquisition starts")
            self.chunk_size = observed
            self._route_chunk_bound = True
        except Exception as error:
            self._abort(error)

    def _clear_packet(self) -> None:
        self._packet_context_start = 0
        self._packet_context_ids = ()
        self._packet_draw_ids = ()
        self._packet_stratum_ids = ()
        self._packet_shell_rows = ()
        self._packet_state_center = None
        self._packet_state_abs = None
        self._packet_nonfinite_marker = None
        self._packet_next_route = 0
        self._packet_windows = []
        self._packet_peak_registered_bytes = 0
        self._packet_base_registered_bytes = 0
        self._packet_open = False

    def _abort(self, error: Exception) -> None:
        self._failed = True
        self._failure_reason = f"{type(error).__name__}: {error}"
        self._clear_packet()
        self._closed_packets.clear()
        self._closed_result = None
        self._closed_acquisition = None
        raise error

    def _assert_live(self, *, verify_request: bool = False) -> None:
        if self._failed:
            raise RuntimeError(
                "projection transaction is permanently refused after: "
                f"{self._failure_reason}"
            )
        if self._closed_result is not None:
            raise RuntimeError("projection transaction is already closed")
        if verify_request:
            self.request.assert_integrity()

    def _route_axis_matches(self, route_ids: Sequence[str]) -> bool:
        if route_ids is self.route_ids:
            return True
        return (
            len(route_ids) == self.request.route_count
            and ordered_string_axis_hash(
                route_ids, name="observed projection ordered score-route axis"
            )
            == self.request.ordered_score_route_axis_hash
        )

    def begin_packet(
        self,
        context_ids: Sequence[str],
        *,
        route_ids: Sequence[str],
        destination_transport_hash: str | None,
        loss_pullback_hash: str | None,
    ) -> None:
        try:
            self._assert_live(verify_request=self._packet_index == 0)
            if not self._route_chunk_bound:
                raise RuntimeError(
                    "projection readiness requires provider chunk binding before VJP"
                )
            if self._packet_open:
                raise RuntimeError("projection packet is already open")
            contexts = _identity_axis(context_ids, name="observed context axis")
            if not contexts:
                raise ValueError("projection packet cannot be empty")
            start = self._next_context
            stop = start + len(contexts)
            if contexts != self.request.expected_context_ids[start:stop]:
                raise ValueError("projection packet violates fresh-context order")
            if not self._route_axis_matches(route_ids):
                raise ValueError("projection packet route axis differs")
            if (
                str(destination_transport_hash or "")
                != self.request.destination_transport_hash
            ):
                raise ValueError("projection destination transport differs")
            if (
                str(loss_pullback_hash or "")
                != self.request.raw_logit_stream_identity_hash
            ):
                raise ValueError("projection raw-logit stream identity differs")
            batch = len(contexts)
            endpoints = self.request.endpoint_count
            shell_rows = tuple(
                torch.empty(
                    (batch, int(indices.size)),
                    dtype=torch.float64,
                    device=self.device,
                )
                for indices in self._shell_physical_indices
            )
            state_center = torch.zeros(
                (batch, endpoints), dtype=torch.float64, device=self.device
            )
            state_abs = torch.zeros_like(state_center)
            marker = torch.zeros((), dtype=torch.float64, device=self.device)
            base_bytes = (
                (
                    sum(int(value.numel()) for value in shell_rows)
                    + int(state_center.numel())
                    + int(state_abs.numel())
                    + 1
                )
                * _FLOAT64_BYTES
                + self._device_csr_bytes
            )
            self._packet_context_start = start
            self._packet_context_ids = contexts
            self._packet_draw_ids = self.request.expected_draw_ids[start:stop]
            self._packet_stratum_ids = self.request.expected_stratum_ids[start:stop]
            self._packet_shell_rows = shell_rows
            self._packet_state_center = state_center
            self._packet_state_abs = state_abs
            self._packet_nonfinite_marker = marker
            self._packet_base_registered_bytes = base_bytes
            self._packet_peak_registered_bytes = base_bytes
            self._packet_open = True
        except Exception as error:
            self._abort(error)

    def consume_positive_removal_chunk(self, start: int, stop: int, rows: Any) -> None:
        try:
            self._assert_live()
            begin = _nonnegative_int(start, name="route window start")
            end = _positive_int(stop, name="route window stop")
            span = end - begin
            contexts = len(self._packet_context_ids)
            if (
                not self._packet_open
                or not self._packet_shell_rows
                or self._packet_nonfinite_marker is None
                or begin != self._packet_next_route
                or span <= 0
                or end > self.request.route_count
                or span > self.request.registered_route_chunk_size
            ):
                raise ValueError(
                    "projection chunks must cover a bounded contiguous prefix"
                )
            if (
                not isinstance(rows, torch.Tensor)
                or tuple(rows.shape) != (contexts, span)
                or rows.dtype != torch.float64
                or rows.device != self.device
                or not rows.is_contiguous()
            ):
                raise ValueError(
                    "projection input must be contiguous float64 [context, route-window] "
                    "on the registered device"
                )
            finite = torch.isfinite(rows).all()
            self._packet_nonfinite_marker.add_(
                torch.where(
                    finite,
                    self._packet_nonfinite_marker.new_zeros(()),
                    self._packet_nonfinite_marker.new_full((), float("nan")),
                )
            )
            for shell, physical in enumerate(self._shell_physical_indices):
                left = int(np.searchsorted(physical, begin, side="left"))
                right = int(np.searchsorted(physical, end, side="left"))
                if left == right:
                    continue
                offsets = torch.as_tensor(
                    (physical[left:right] - begin).astype(np.int64, copy=False),
                    dtype=torch.int64,
                    device=self.device,
                )
                selected = torch.index_select(rows, 1, offsets)
                self._packet_shell_rows[shell][:, left:right].copy_(selected.detach())
                temporary_bytes = (
                    int(offsets.numel()) * int(offsets.element_size())
                    + int(selected.numel()) * int(selected.element_size())
                )
                self._packet_peak_registered_bytes = max(
                    self._packet_peak_registered_bytes,
                    self._packet_base_registered_bytes + temporary_bytes,
                )
            self._packet_windows.append((begin, end))
            self._packet_next_route = end
        except Exception as error:
            self._abort(error)

    def consume_chunk(self, start: int, stop: int, rows: Any) -> None:
        del start, stop, rows
        self._abort(
            PermissionError(
                "frozen-action projection accepts only direct signed raw-logit "
                "positive-removal rows"
            )
        )

    def consume_chunk_fused(
        self,
        start: int,
        stop: int,
        *,
        global_raw: Any,
        gate: Any,
        pullback: Any,
        affine_scale: float,
        score_epilogue: Any | None = None,
    ) -> None:
        del start, stop, global_raw, gate, pullback, affine_scale, score_epilogue
        self._abort(
            PermissionError(
                "projection refuses consumer-side pullback formation; scorer must emit "
                "-G*phi*c raw-logit chunks"
            )
        )

    def _final_endpoint_tensors(self) -> tuple[Any, Any]:
        assert torch is not None
        state_center = self._packet_state_center
        state_abs = self._packet_state_abs
        marker = self._packet_nonfinite_marker
        shell_rows = self._packet_shell_rows
        if (
            state_center is None
            or state_abs is None
            or marker is None
            or not shell_rows
        ):
            raise RuntimeError("projection finalizer has no packet state")
        if tuple(state_center.shape) != (
            len(self._packet_context_ids),
            self.request.endpoint_count,
        ) or state_abs.shape != state_center.shape:
            raise RuntimeError("selected-shell projection state changed shape")
        for shell, values in enumerate(shell_rows):
            state_center[:, shell].copy_(_pairwise_reduce_axis1(values))
            state_abs[:, shell].copy_(_pairwise_reduce_axis1(torch.abs(values)))
        endpoint_center = state_center.clone()
        endpoint_abs = state_abs.clone()
        endpoints = self.request.endpoint_count
        for endpoint in range(1, endpoints):
            endpoint_center[:, endpoint].add_(endpoint_center[:, endpoint - 1])
            endpoint_abs[:, endpoint].add_(endpoint_abs[:, endpoint - 1])
        endpoint_center[0, 0].add_(marker)
        return endpoint_center, endpoint_abs

    def end_packet(self) -> None:
        try:
            self._assert_live()
            if (
                not self._packet_open
                or self._packet_next_route != self.request.route_count
            ):
                raise RuntimeError("projection packet did not cover route axis exactly")
            contexts = len(self._packet_context_ids)
            endpoints = self.request.endpoint_count
            center_device, abs_device = self._final_endpoint_tensors()
            center_host = (
                center_device.detach()
                .to(device="cpu", dtype=torch.float64)
                .contiguous()
            )
            abs_host = (
                abs_device.detach().to(device="cpu", dtype=torch.float64).contiguous()
            )
            center, center_hash = _canonical_matrix(
                center_host.numpy(), shape=(contexts, endpoints), name="packet centers"
            )
            observed_abs, abs_hash = _canonical_matrix(
                abs_host.numpy(),
                shape=(contexts, endpoints),
                name="packet absolute sums",
            )
            if np.any(observed_abs < 0.0):
                raise ValueError("projection absolute sum became negative")
            windows = tuple(self._packet_windows)
            window_hash = content_hash(
                {
                    "request_hash": self.request.request_hash,
                    "packet_index": self._packet_index,
                    "route_count": self.request.route_count,
                    "ordered_score_route_axis_hash": (
                        self.request.ordered_score_route_axis_hash
                    ),
                    "route_windows": windows,
                },
                prefix=_WINDOW_PREFIX,
            )
            host_bytes = contexts * endpoints * _FLOAT64_BYTES * 2
            packet_receipt = V1LadderProjectionPacketReceipt(
                request_hash=self.request.request_hash,
                packet_index=self._packet_index,
                context_start=self._packet_context_start,
                context_stop=self._packet_context_start + contexts,
                context_ids=self._packet_context_ids,
                draw_ids=self._packet_draw_ids,
                stratum_ids=self._packet_stratum_ids,
                route_count=self.request.route_count,
                physical_mask_axis_hash=self.request.physical_mask_axis_hash,
                ordered_score_route_axis_hash=(
                    self.request.ordered_score_route_axis_hash
                ),
                route_windows=windows,
                route_window_ledger_hash=window_hash,
                signed_projection_matrix_hash=center_hash,
                observed_absolute_sum_matrix_hash=abs_hash,
                numerical_backend=self._numerical_backend,
                peak_registered_device_bytes=self._packet_peak_registered_bytes,
                projection_host_materialization_bytes=host_bytes,
                device_to_host_bytes=host_bytes if self.device.type == "cuda" else 0,
                device_identity=self._device_identity,
            )
            self._closed_packets.append(
                _ClosedPacket(packet_receipt, center, observed_abs)
            )
            self._next_context += contexts
            self._packet_index += 1
            self._clear_packet()
        except Exception as error:
            self._abort(error)

    def close(self) -> V1LadderProjectionResult:
        if self._closed_result is not None:
            return self._closed_result
        try:
            self._assert_live(verify_request=True)
            if self._packet_open or self._next_context != self.request.context_count:
                raise RuntimeError("cannot close a partial projection request")
            if not self._closed_packets:
                raise RuntimeError("projection request produced no packets")
            shape = (self.request.context_count, self.request.endpoint_count)
            center, center_hash = _canonical_matrix(
                np.concatenate([item.center for item in self._closed_packets], axis=0),
                shape=shape,
                name="complete projection centers",
            )
            observed_abs, observed_abs_hash = _canonical_matrix(
                np.concatenate(
                    [item.observed_abs for item in self._closed_packets], axis=0
                ),
                shape=shape,
                name="complete observed absolute sums",
            )
            abs_upper_raw, radius_raw, lower_raw, upper_raw = _outward_enclosure(
                center,
                observed_abs,
                endpoint_removed_counts=self.request.endpoint_removed_counts,
                shell_segment_counts=self.request.csr.shell_segment_counts,
            )
            abs_upper, abs_upper_hash = _canonical_matrix(
                abs_upper_raw, shape=shape, name="absolute input sum upper bounds"
            )
            radius, radius_hash = _canonical_matrix(
                radius_raw, shape=shape, name="projection numerical radii"
            )
            lower, lower_hash = _canonical_matrix(
                lower_raw, shape=shape, name="projection numerical lower bounds"
            )
            upper, upper_hash = _canonical_matrix(
                upper_raw, shape=shape, name="projection numerical upper bounds"
            )
            packets = tuple(item.receipt for item in self._closed_packets)
            receipt = V1LadderProjectionDerivationReceipt(
                request_hash=self.request.request_hash,
                design_hash=self.request.design_hash,
                ladder_hash=self.request.ladder_hash,
                csr_hash=self.request.csr.csr_hash,
                current_mask_hash=self.request.current_mask_hash,
                route_count=self.request.route_count,
                physical_mask_axis_hash=self.request.physical_mask_axis_hash,
                ordered_score_route_axis_hash=(
                    self.request.ordered_score_route_axis_hash
                ),
                raw_logit_stream_source_hash=self.request.design.ladder.raw_logit_stream_source_hash,
                raw_logit_stream_identity_hash=self.request.raw_logit_stream_identity_hash,
                destination_transport_hash=self.request.destination_transport_hash,
                scorer_request_identity_hash=self.request.scorer_request_identity_hash,
                acquisition_lineage_hash=self.request.acquisition_lineage_hash,
                context_ids=self.request.expected_context_ids,
                draw_ids=self.request.expected_draw_ids,
                stratum_ids=self.request.expected_stratum_ids,
                endpoint_action_hashes=self.request.endpoint_action_hashes,
                shell_hashes=self.request.shell_hashes,
                shell_removed_counts=self.request.shell_removed_counts,
                packet_receipt_hashes=tuple(item.receipt_hash for item in packets),
                route_window_ledger_hashes=tuple(
                    item.route_window_ledger_hash for item in packets
                ),
                signed_projection_matrix_hash=center_hash,
                observed_absolute_sum_matrix_hash=observed_abs_hash,
                absolute_input_sum_upper_matrix_hash=abs_upper_hash,
                numerical_radius_matrix_hash=radius_hash,
                numerical_lower_matrix_hash=lower_hash,
                numerical_upper_matrix_hash=upper_hash,
                projection_kernel_source_hash=projection_kernel_source_hash(),
                numerical_backend=self._numerical_backend,
                peak_registered_device_bytes=max(
                    item.peak_registered_device_bytes for item in packets
                ),
                total_projection_host_materialization_bytes=sum(
                    item.projection_host_materialization_bytes for item in packets
                ),
                exact_device_to_host_bytes=sum(
                    item.device_to_host_bytes for item in packets
                ),
                packet_count=len(packets),
                route_chunk_update_count=sum(
                    len(item.route_windows) for item in packets
                ),
            )
            result = V1LadderProjectionResult(
                receipt=receipt,
                packet_receipts=packets,
                signed_projection_matrix=center,
                observed_absolute_sum_matrix=observed_abs,
                absolute_input_sum_upper_matrix=abs_upper,
                numerical_radius_matrix=radius,
                numerical_lower_matrix=lower,
                numerical_upper_matrix=upper,
            )
            self._closed_packets.clear()
            self._closed_result = result
            return result
        except Exception as error:
            self._abort(error)
            raise AssertionError("unreachable")  # pragma: no cover

    @staticmethod
    def _find_projection_child_receipt(
        scorer_receipt: Mapping[str, object],
        *,
        projection_receipt_hash: str,
        stream_identity_hash: str,
    ) -> Mapping[str, object] | None:
        matches: list[Mapping[str, object]] = []

        def visit(value: Mapping[str, object]) -> None:
            if (
                value.get("frozen_action_projection_streaming") is True
                and value.get("projection_derivation_receipt_hash")
                == projection_receipt_hash
                and value.get("positive_removal_stream_identity_hash")
                == stream_identity_hash
            ):
                matches.append(value)
            for key in (
                "loss_only_stream_tee_child_receipts",
                "child_streaming_receipts",
                "dual_chart_evidence_receipt",
                "dual_chart_jacobian_receipt",
            ):
                children = value.get(key, ())
                if isinstance(children, Mapping):
                    visit(children)
                    continue
                if isinstance(children, Sequence) and not isinstance(
                    children, (str, bytes)
                ):
                    for child in children:
                        if isinstance(child, Mapping):
                            visit(child)

        visit(scorer_receipt)
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _find_dual_chart_receipt(
        scorer_receipt: Mapping[str, object],
        *,
        corrected_evidence_stream_identity_hash: str,
        literal_jacobian_stream_identity_hash: str,
    ) -> Mapping[str, object] | None:
        """Return the unique same-capture dual-stream receipt, if present."""

        matches: list[Mapping[str, object]] = []

        def visit(value: Mapping[str, object]) -> None:
            if (
                value.get("dual_chart_streaming") is True
                and value.get("corrected_evidence_stream_identity_hash")
                == corrected_evidence_stream_identity_hash
                and value.get("literal_jacobian_stream_identity_hash")
                == literal_jacobian_stream_identity_hash
            ):
                matches.append(value)
            for key in (
                "loss_only_stream_tee_child_receipts",
                "child_streaming_receipts",
                "dual_chart_evidence_receipt",
                "dual_chart_jacobian_receipt",
            ):
                children = value.get(key, ())
                if isinstance(children, Mapping):
                    visit(children)
                    continue
                if isinstance(children, Sequence) and not isinstance(
                    children, (str, bytes)
                ):
                    for child in children:
                        if isinstance(child, Mapping):
                            visit(child)

        visit(scorer_receipt)
        return matches[0] if len(matches) == 1 else None

    def close_acquisition(
        self, scorer_receipt: Mapping[str, object]
    ) -> V1LadderProjectionAcquisition:
        """Bind the returned scorer receipt before emitting certifier records."""

        if not isinstance(scorer_receipt, Mapping):
            raise TypeError("projection closure requires a scorer receipt")
        if self._closed_acquisition is not None:
            scorer_hash = _identity(
                scorer_receipt.get("receipt_hash"), name="scorer receipt hash"
            )
            scorer_payload = dict(scorer_receipt)
            scorer_payload.pop("receipt_hash", None)
            if (
                scorer_hash != self._closed_acquisition.closure.scorer_receipt_hash
                or content_hash(scorer_payload, prefix=_SCORER_RECEIPT_PREFIX)
                != scorer_hash
            ):
                raise ValueError(
                    "closed projection acquisition cannot bind another scorer receipt"
                )
            return self._closed_acquisition
        result = self.close()
        receipt = result.receipt
        scorer_hash = _identity(
            scorer_receipt.get("receipt_hash"), name="scorer receipt hash"
        )
        scorer_schema = _identity(
            scorer_receipt.get("schema"), name="scorer receipt schema"
        )
        scorer_payload = dict(scorer_receipt)
        scorer_payload.pop("receipt_hash", None)
        if (
            scorer_schema != _SCORER_RECEIPT_SCHEMA
            or content_hash(scorer_payload, prefix=_SCORER_RECEIPT_PREFIX)
            != scorer_hash
        ):
            raise ValueError("scorer receipt content hash does not replay")
        request_identity = str(
            scorer_receipt.get(
                "request_id", scorer_receipt.get("scorer_request_identity_hash", "")
            )
        )
        context_axis_hash = content_hash(
            self.request.expected_context_ids, prefix="irg-device-panel-contexts-"
        )
        child = self._find_projection_child_receipt(
            scorer_receipt,
            projection_receipt_hash=receipt.receipt_hash,
            stream_identity_hash=self.request.raw_logit_stream_identity_hash,
        )
        dual_chart = (
            None
            if type(self.request) is not V1DirectionalActionProjectionRequest
            else self._find_dual_chart_receipt(
                scorer_receipt,
                corrected_evidence_stream_identity_hash=(
                    self.request.corrected_evidence_stream_identity_hash
                ),
                literal_jacobian_stream_identity_hash=(
                    self.request.literal_jacobian_stream_identity_hash
                ),
            )
        )
        ordinary_stream_identity_matches = (
            scorer_receipt.get("positive_removal_stream_identity_hash")
            == self.request.raw_logit_stream_identity_hash
        )
        dual_stream_identity_matches = (
            dual_chart is not None
            and dual_chart.get("same_capture_and_vjp_required") is True
            and self.request.raw_logit_stream_identity_hash
            in {
                dual_chart.get("corrected_evidence_stream_identity_hash"),
                dual_chart.get("literal_jacobian_stream_identity_hash"),
            }
        )
        if (
            request_identity != self.request.scorer_request_identity_hash
            or int(scorer_receipt.get("context_count", -1))
            != self.request.context_count
            or int(scorer_receipt.get("route_count", -1)) != self.request.route_count
            or scorer_receipt.get("positive_removal_stream_coordinate")
            != LADDER_PROJECTION_COORDINATE
            or not (
                ordinary_stream_identity_matches
                or dual_stream_identity_matches
            )
            or scorer_receipt.get("context_ids_hash") != context_axis_hash
            or child is None
            or int(child.get("route_chunk_size", -1))
            != self.request.registered_route_chunk_size
            or child.get("physical_mask_axis_hash")
            != self.request.physical_mask_axis_hash
            or child.get("ordered_score_route_axis_hash")
            != self.request.ordered_score_route_axis_hash
            or child.get("projection_derivation_receipt_hash") != receipt.receipt_hash
        ):
            raise ValueError(
                "scorer receipt does not close frozen projection acquisition"
            )
        closure = V1LadderProjectionAcquisitionClosure(
            request_hash=self.request.request_hash,
            design_hash=self.request.design_hash,
            ladder_hash=self.request.ladder_hash,
            projection_receipt_hash=receipt.receipt_hash,
            scorer_receipt_hash=scorer_hash,
            scorer_receipt_schema=scorer_schema,
            scorer_request_identity_hash=self.request.scorer_request_identity_hash,
            acquisition_lineage_hash=self.request.acquisition_lineage_hash,
            physical_mask_axis_hash=self.request.physical_mask_axis_hash,
            ordered_score_route_axis_hash=(
                self.request.ordered_score_route_axis_hash
            ),
            context_axis_hash=context_axis_hash,
            draw_axis_hash=content_hash(
                self.request.expected_draw_ids,
                prefix="partition-free-v1-projection-draw-axis-",
            ),
            stratum_axis_hash=content_hash(
                self.request.expected_stratum_ids,
                prefix="partition-free-v1-projection-stratum-axis-",
            ),
            signed_projection_matrix_hash=receipt.signed_projection_matrix_hash,
            absolute_input_sum_upper_matrix_hash=receipt.absolute_input_sum_upper_matrix_hash,
            numerical_radius_matrix_hash=receipt.numerical_radius_matrix_hash,
            numerical_lower_matrix_hash=receipt.numerical_lower_matrix_hash,
            numerical_upper_matrix_hash=receipt.numerical_upper_matrix_hash,
        )
        proof_hash = closure.closure_hash
        records = tuple(
            V1FrozenLadderProjectionRecord(
                design_hash=self.request.design_hash,
                ladder_hash=self.request.ladder_hash,
                stratum_id=self.request.expected_stratum_ids[row],
                context_id=self.request.expected_context_ids[row],
                draw_id=self.request.expected_draw_ids[row],
                projection_proof_hash=proof_hash,
                endpoint_action_hashes=self.request.endpoint_action_hashes,
                signed_projection_by_endpoint=tuple(
                    float(value) for value in result.signed_projection_matrix[row]
                ),
                numerical_radius_by_endpoint=tuple(
                    float(value) for value in result.numerical_radius_matrix[row]
                ),
            )
            for row in range(self.request.context_count)
        )
        acquisition = V1LadderProjectionAcquisition(
            result=result, closure=closure, projection_records=records
        )
        self._closed_acquisition = acquisition
        return acquisition

    def route_streaming_receipt(self) -> Mapping[str, object]:
        if (
            self._closed_result is None
            and not self._failed
            and not self._packet_open
            and self._next_context == self.request.context_count
        ):
            self.close()
        if self._closed_result is None:
            if self._failed:
                raise RuntimeError("refused projection transaction has no receipt")
            raise RuntimeError("projection transaction must close before receipt")
        receipt = self._closed_result.receipt
        return MappingProxyType(
            {
                "route_chunked_streaming": True,
                "route_chunk_size": self.request.registered_route_chunk_size,
                "route_chunk_update_count": receipt.route_chunk_update_count,
                "direct_positive_removal_chunk_count": (
                    receipt.route_chunk_update_count
                ),
                "loss_only_route_streaming": True,
                "frozen_action_projection_streaming": True,
                "positive_removal_stream_coordinate": LADDER_PROJECTION_COORDINATE,
                "positive_removal_stream_identity_hash": self.request.raw_logit_stream_identity_hash,
                "projection_orientation": getattr(
                    self.request,
                    "row_semantics",
                    LADDER_PROJECTION_ORIENTATION,
                ),
                "projection_units": LADDER_PROJECTION_UNITS,
                "physical_mask_axis_hash": self.request.physical_mask_axis_hash,
                "ordered_score_route_axis_hash": (
                    self.request.ordered_score_route_axis_hash
                ),
                "projection_csr_hash": self.request.csr.csr_hash,
                "projection_derivation_receipt_hash": receipt.receipt_hash,
                "projection_kernel_source_hash": receipt.projection_kernel_source_hash,
                "projection_arithmetic_id": LADDER_PROJECTION_ARITHMETIC_ID,
                "numerical_backend": receipt.numerical_backend,
                "estimator_rows_exactly_projected": True,
                "real_arithmetic_exactness_claimed": False,
                "numerical_enclosure_provided": True,
                "peak_registered_device_bytes": receipt.peak_registered_device_bytes,
                "projection_device_to_host_bytes": receipt.exact_device_to_host_bytes,
                "host_context_route_rows_transferred": False,
                "host_context_route_row_transfer_count": 0,
                "host_context_route_row_transfer_bytes": 0,
                "full_tripartite_rows_materialized": False,
                "scalar_projection_derivation_authority": True,
                "certifier_records_emitted_before_provider_closure": False,
                "fq_certification_authority": False,
                "score_order_authority": False,
                "bce_oracle_authority": False,
                "mask_commit_authority": False,
            }
        )


__all__ = [
    "DIRECTIONAL_LADDER_PROJECTION_REQUEST_SCHEMA",
    "LEGACY_DIRECTIONAL_LADDER_PROJECTION_REQUEST_SCHEMA",
    "LADDER_PROJECTION_ARITHMETIC_ID",
    "LADDER_PROJECTION_AUTHORITY",
    "LADDER_PROJECTION_COORDINATE",
    "LADDER_PROJECTION_COTANGENT",
    "LADDER_PROJECTION_ORIENTATION",
    "LADDER_PROJECTION_UNITS",
    "LOGICAL_TILE_SIZE",
    "STAGE1_BLOCK_SIZE",
    "V1FrozenActionProjectionRequest",
    "V1DirectionalActionProjectionRequest",
    "V1LadderProjectionAcquisition",
    "V1LadderProjectionAcquisitionClosure",
    "V1LadderProjectionDerivationReceipt",
    "V1LadderProjectionGPUConsumer",
    "V1LadderProjectionPacketReceipt",
    "V1LadderProjectionResult",
    "V1LegacyTupleLadderProjectionReplayAdapter",
    "V1PackedLadderShellCSR",
    "build_v1_packed_ladder_shell_csr",
    "projection_kernel_source_hash",
]
