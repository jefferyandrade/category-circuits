"""Bounded streaming hash custody for the modern-v1 CUDA source theorem.

The relaxed-KKT owner lends the decision-level issuer route-scale key
intervals, but it deliberately does not own hard-mask identity.  This module
is the independent identity joint.  It classifies one live theorem-state
lease in bounded physical CUDA tiles, transfers only one bounded ``uint8``
membership slab at a time, and streams canonical little-endian physical route
indices into SHA-256.  No complete key, index, or membership axis is retained
on the host.

The hash placed in the legacy ``outcome_ladder_axis_hash`` field of
``V1DecisionLevelCudaShellHashCustody`` is explicitly a commitment to the
*complete shell-subset -> removed-index-hash mapping*.  It is not the exact
face/order outcome axis emitted later by the decision-level issuer.  The
issuer must derive and check that separate face-outcome axis before theorem
promotion.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import InitVar, dataclass, fields
from typing import Any, Final, Iterable, Literal, Mapping, NoReturn, Sequence

import numpy as np

from mechanistic_systems.contracts.hard_mask_actions import RemovedRouteIndexSet
from mechanistic_systems.measurement.gpu_runtime import cuda_reusable_memory_bytes

from .contracts import content_hash
from .modern_v1_decision_level_cuda_issuer import (
    DEFAULT_H200_HBM_CEILING_FRACTION,
    DEFAULT_MAXIMUM_SHELL_ROUTES,
    V1DecisionLevelCudaIssuerDesign,
    V1DecisionLevelCudaIssuerResult,
    V1DecisionLevelCudaIssuerRefusal,
    V1DecisionLevelCudaIssuerRefusalCode,
    V1DecisionLevelCudaKKTLeaseAdapter,
    V1DecisionLevelCudaShellHashCustody,
    assert_complete_v1_decision_level_cuda_route_scan,
    issue_v1_decision_level_cuda_certificate,
    plan_v1_decision_level_cuda_route_tiles,
)
from .modern_v1_literal_j_projection_inference import (
    V1LiteralJFixedDirectionProjectionReceipt,
)
from .modern_v1_relaxed_kkt import (
    V1RelaxedKKTAdaptiveRankBranchWitness,
    V1RelaxedKKTBranchResult,
    V1RelaxedKKTDecisionLevelTheoremStateLease,
    V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    V1RelaxedKKTDeviceShellExecutionBundle,
    V1RelaxedKKTGeometryUncertaintySet,
    V1RelaxedKKTPolicy,
    V1RelaxedKKTProblem,
)


MAXIMUM_HOST_LABEL_SLAB_BYTES: Final = 1 << 20
HASH_CUSTODY_TILE_WORKSPACE_BYTES_PER_ROUTE: Final = 320
HASH_CUSTODY_SHELL_WORKSPACE_BASE_BYTES: Final = 4096

PREFIX_LABEL: Final = 1
SUFFIX_LABEL: Final = 2
SHELL_LABEL_BASE: Final = 3

COMPLETE_SUBSET_MAPPING_COMMITMENT_KIND: Final = (
    "complete_shell_subset_to_removed_route_index_hash_mapping_"
    "not_face_outcome_axis_v1"
)
SCHEMA_SUBSET_HASH: Final = (
    "partition_free_v1_decision_level_cuda_shell_subset_hash_v1"
)
SCHEMA_STREAMED_HASH_FAMILY: Final = (
    "partition_free_v1_decision_level_cuda_streamed_hash_family_v1"
)
SCHEMA_HBM_FORECAST: Final = (
    "partition_free_v1_decision_level_cuda_hash_custody_hbm_forecast_v1"
)
SCHEMA_RECEIPT: Final = (
    "partition_free_v1_decision_level_cuda_hash_custody_receipt_v1"
)


def _refuse(
    code: V1DecisionLevelCudaIssuerRefusalCode,
    message: str,
    **payload: object,
) -> NoReturn:
    raise V1DecisionLevelCudaIssuerRefusal(code, message, **payload)


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return int(value)


def _nonnegative_int(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative int")
    return int(value)


def _nonempty(value: str, name: str) -> str:
    normalized = str(value)
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    return normalized


def _dtype_contract(route_count: int) -> tuple[Literal["u32", "u64"], np.dtype]:
    if route_count <= np.iinfo(np.uint32).max:
        return "u32", np.dtype("<u4")
    return "u64", np.dtype("<u8")


def assert_v1_decision_level_cuda_hash_custody_device(
    *,
    device: str,
    cuda_available: bool,
) -> None:
    """Refuse CPU classification and unavailable CUDA before any scan."""

    normalized = _nonempty(device, "device")
    if not normalized.startswith("cuda"):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.CPU_FALLBACK_FORBIDDEN,
            "decision-level hash custody requires CUDA key classification; "
            "CPU fallback is forbidden",
            device=normalized,
        )
    if type(cuda_available) is not bool:
        raise TypeError("cuda_available must be an exact bool")
    if not cuda_available:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.CUDA_UNAVAILABLE,
            "decision-level hash custody CUDA runtime is unavailable",
            device=normalized,
        )


@dataclass(frozen=True, slots=True)
class V1DecisionLevelCudaShellSubsetHash:
    """One exact hard removal identity for a bounded shell subset."""

    shell_offsets: tuple[int, ...]
    removed_count: int
    removed_route_index_hash: str
    schema: str = SCHEMA_SUBSET_HASH

    def __post_init__(self) -> None:
        offsets = tuple(
            _nonnegative_int(value, "shell offset") for value in self.shell_offsets
        )
        if offsets != tuple(sorted(set(offsets))):
            raise ValueError("shell offsets must be unique and sorted")
        object.__setattr__(self, "shell_offsets", offsets)
        object.__setattr__(
            self,
            "removed_count",
            _nonnegative_int(self.removed_count, "removed_count"),
        )
        object.__setattr__(
            self,
            "removed_route_index_hash",
            _nonempty(self.removed_route_index_hash, "removed_route_index_hash"),
        )
        if self.schema != SCHEMA_SUBSET_HASH:
            raise ValueError("unsupported shell-subset hash schema")


def compute_v1_decision_level_complete_subset_mapping_commitment(
    *,
    route_count: int,
    shell_route_index_hash: str,
    fixed_prefix_route_index_hash: str,
    fixed_prefix_count: int,
    subset_hashes: Sequence[V1DecisionLevelCudaShellSubsetHash],
) -> str:
    """Commit the complete subset mapping, never a face-outcome axis."""

    total = _positive_int(route_count, "route_count")
    entries = tuple(subset_hashes)
    if not entries or any(
        type(entry) is not V1DecisionLevelCudaShellSubsetHash for entry in entries
    ):
        raise TypeError("subset mapping requires exact subset-hash entries")
    ordered = tuple(sorted(entries, key=lambda entry: entry.shell_offsets))
    if len({entry.shell_offsets for entry in ordered}) != len(ordered):
        raise ValueError("subset mapping contains duplicate shell subsets")
    return content_hash(
        {
            "commitment_kind": COMPLETE_SUBSET_MAPPING_COMMITMENT_KIND,
            "route_count": total,
            "shell_route_index_hash": _nonempty(
                shell_route_index_hash,
                "shell_route_index_hash",
            ),
            "fixed_prefix_route_index_hash": _nonempty(
                fixed_prefix_route_index_hash,
                "fixed_prefix_route_index_hash",
            ),
            "fixed_prefix_count": _nonnegative_int(
                fixed_prefix_count,
                "fixed_prefix_count",
            ),
            "complete_subset_mapping": tuple(
                {
                    "shell_offsets": entry.shell_offsets,
                    "removed_count": entry.removed_count,
                    "removed_route_index_hash": entry.removed_route_index_hash,
                }
                for entry in ordered
            ),
        },
        prefix=(
            "partition-free-v1-decision-cuda-complete-shell-subset-"
            "mapping-commitment-"
        ),
    )


@dataclass(frozen=True, slots=True)
class V1DecisionLevelCudaStreamedHashFamily:
    """Array-free result of one complete bounded-label physical scan."""

    route_count: int
    shell_route_indices: tuple[int, ...]
    shell_route_index_hash: str
    fixed_prefix_route_index_hash: str
    fixed_prefix_count: int
    fixed_suffix_route_index_hash: str
    fixed_suffix_count: int
    eligible_route_count: int
    ignored_ineligible_route_count: int
    subset_hashes: tuple[V1DecisionLevelCudaShellSubsetHash, ...]
    registered_endpoint_shell_offsets: tuple[tuple[int, ...], ...]
    complete_subset_mapping_commitment: str
    scanned_route_count: int
    label_d2h_bytes_total: int
    maximum_label_slab_bytes_observed: int
    complete_physical_scan: bool = True
    route_scale_host_row_retention_used: bool = False
    full_key_or_index_d2h_used: bool = False
    commitment_kind: str = COMPLETE_SUBSET_MAPPING_COMMITMENT_KIND
    schema: str = SCHEMA_STREAMED_HASH_FAMILY

    def __post_init__(self) -> None:
        total = _positive_int(self.route_count, "route_count")
        shell = tuple(
            _nonnegative_int(value, "shell route index")
            for value in self.shell_route_indices
        )
        if (
            not shell
            or shell != tuple(sorted(set(shell)))
            or shell[-1] >= total
            or len(shell) > DEFAULT_MAXIMUM_SHELL_ROUTES
        ):
            raise ValueError("streamed hash family has an invalid bounded shell")
        object.__setattr__(self, "shell_route_indices", shell)
        for name in (
            "shell_route_index_hash",
            "fixed_prefix_route_index_hash",
            "fixed_suffix_route_index_hash",
            "complete_subset_mapping_commitment",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "fixed_prefix_count",
            "fixed_suffix_count",
            "eligible_route_count",
            "ignored_ineligible_route_count",
            "scanned_route_count",
            "label_d2h_bytes_total",
            "maximum_label_slab_bytes_observed",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        entries = tuple(self.subset_hashes)
        if any(type(entry) is not V1DecisionLevelCudaShellSubsetHash for entry in entries):
            raise TypeError("streamed hash family requires exact subset hashes")
        expected_subset_count = 1 << len(shell)
        if len(entries) != expected_subset_count:
            raise ValueError("streamed hash family is not the complete shell powerset")
        expected_offsets = {
            tuple(offset for offset in range(len(shell)) if mask & (1 << offset))
            for mask in range(expected_subset_count)
        }
        if {entry.shell_offsets for entry in entries} != expected_offsets:
            raise ValueError("streamed hash family omitted a shell subset")
        object.__setattr__(
            self,
            "subset_hashes",
            tuple(sorted(entries, key=lambda entry: entry.shell_offsets)),
        )
        endpoints = tuple(tuple(value) for value in self.registered_endpoint_shell_offsets)
        if any(
            offsets != tuple(sorted(set(offsets)))
            or any(offset < 0 or offset >= len(shell) for offset in offsets)
            for offsets in endpoints
        ):
            raise ValueError("registered endpoint shell offsets are invalid")
        if any(not set(left).issubset(right) for left, right in zip(endpoints, endpoints[1:])):
            raise ValueError("registered endpoint shell subsets are not nested")
        object.__setattr__(self, "registered_endpoint_shell_offsets", endpoints)
        if (
            self.scanned_route_count != total
            or self.label_d2h_bytes_total != total
            or self.eligible_route_count
            != self.fixed_prefix_count + self.fixed_suffix_count + len(shell)
            or self.ignored_ineligible_route_count
            != total - self.eligible_route_count
        ):
            raise ValueError("streamed hash-family census does not close")
        flags = (
            self.complete_physical_scan,
            self.route_scale_host_row_retention_used,
            self.full_key_or_index_d2h_used,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("streamed hash-family flags must be exact booleans")
        if flags != (True, False, False):
            raise PermissionError("streamed hash-family custody crossed its boundary")
        if self.commitment_kind != COMPLETE_SUBSET_MAPPING_COMMITMENT_KIND:
            raise ValueError("subset commitment was mislabeled as a face-outcome axis")
        expected_commitment = (
            compute_v1_decision_level_complete_subset_mapping_commitment(
                route_count=total,
                shell_route_index_hash=self.shell_route_index_hash,
                fixed_prefix_route_index_hash=self.fixed_prefix_route_index_hash,
                fixed_prefix_count=self.fixed_prefix_count,
                subset_hashes=self.subset_hashes,
            )
        )
        if self.complete_subset_mapping_commitment != expected_commitment:
            raise ValueError("complete shell-subset mapping commitment does not replay")
        if self.schema != SCHEMA_STREAMED_HASH_FAMILY:
            raise ValueError("unsupported streamed hash-family schema")

    @property
    def family_hash(self) -> str:
        return content_hash(
            {item.name: getattr(self, item.name) for item in fields(self)},
            prefix="partition-free-v1-decision-cuda-streamed-hash-family-",
        )

    @property
    def endpoint_hash_by_shell_offsets(self) -> Mapping[tuple[int, ...], str]:
        return {
            entry.shell_offsets: entry.removed_route_index_hash
            for entry in self.subset_hashes
        }


class _StreamingHashAccumulator:
    """One-pass bounded host digest state; never stores a previous slab."""

    __slots__ = (
        "_cursor",
        "_dtype",
        "_dtype_code",
        "_eligible_route_count",
        "_expected_eligible_route_count",
        "_expected_shell_hash",
        "_fixed_prefix_count",
        "_fixed_suffix_count",
        "_ignored_count",
        "_label_bytes",
        "_maximum_label_slab_bytes",
        "_maximum_observed",
        "_prefix_digest",
        "_route_count",
        "_shell_digest",
        "_shell_route_indices",
        "_shell_route_seen",
        "_subset_counts",
        "_subset_digests",
        "_suffix_digest",
    )

    def __init__(
        self,
        *,
        route_count: int,
        shell_route_indices: Sequence[int],
        expected_eligible_route_count: int,
        expected_shell_route_index_hash: str,
        maximum_label_slab_bytes: int,
    ) -> None:
        self._route_count = _positive_int(route_count, "route_count")
        shell = tuple(int(value) for value in shell_route_indices)
        if (
            not shell
            or shell != tuple(sorted(set(shell)))
            or shell[0] < 0
            or shell[-1] >= self._route_count
            or len(shell) > DEFAULT_MAXIMUM_SHELL_ROUTES
        ):
            raise ValueError("hash custody requires a sorted, unique, bounded shell")
        self._shell_route_indices = shell
        self._expected_eligible_route_count = _positive_int(
            expected_eligible_route_count,
            "expected_eligible_route_count",
        )
        if self._expected_eligible_route_count > self._route_count:
            raise ValueError("eligible route census exceeds the route axis")
        self._expected_shell_hash = _nonempty(
            expected_shell_route_index_hash,
            "expected_shell_route_index_hash",
        )
        self._maximum_label_slab_bytes = _positive_int(
            maximum_label_slab_bytes,
            "maximum_label_slab_bytes",
        )
        self._dtype_code, self._dtype = _dtype_contract(self._route_count)
        subset_count = 1 << len(shell)
        self._prefix_digest = hashlib.sha256()
        self._suffix_digest = hashlib.sha256()
        self._shell_digest = hashlib.sha256()
        self._subset_digests = tuple(hashlib.sha256() for _ in range(subset_count))
        self._subset_counts = [0 for _ in range(subset_count)]
        self._shell_route_seen: list[int | None] = [None for _ in shell]
        self._fixed_prefix_count = 0
        self._fixed_suffix_count = 0
        self._eligible_route_count = 0
        self._ignored_count = 0
        self._label_bytes = 0
        self._maximum_observed = 0
        self._cursor = 0

    def _encoded_indices(self, mask: np.ndarray, *, start: int) -> tuple[bytes, int]:
        positions = np.flatnonzero(mask)
        if not positions.size:
            return b"", 0
        physical = positions + int(start)
        encoded = physical.astype(self._dtype, copy=False).tobytes(order="C")
        return encoded, int(positions.size)

    def update(self, *, start: int, labels: np.ndarray) -> None:
        if type(start) is not int or start != self._cursor:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.ROUTE_SCAN_INCOMPLETE,
                "hash-custody membership slabs are not a contiguous physical scan",
                expected_start=self._cursor,
                observed_start=start,
            )
        slab = np.asarray(labels)
        if (
            slab.ndim != 1
            or slab.dtype != np.dtype(np.uint8)
            or not slab.flags.c_contiguous
            or slab.size <= 0
        ):
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.ROUTE_SCALE_D2H_FORBIDDEN,
                "hash-custody D2H membership slab is malformed or noncontiguous",
            )
        if slab.nbytes > self._maximum_label_slab_bytes:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.ROUTE_SCALE_D2H_FORBIDDEN,
                "hash-custody membership slab exceeds the registered D2H bound",
                slab_bytes=int(slab.nbytes),
                maximum_label_slab_bytes=self._maximum_label_slab_bytes,
            )
        stop = start + int(slab.size)
        if stop > self._route_count:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.ROUTE_SCAN_INCOMPLETE,
                "hash-custody membership slab exceeds the physical route axis",
            )
        maximum_label = SHELL_LABEL_BASE + len(self._shell_route_indices) - 1
        if slab.size and int(np.max(slab)) > maximum_label:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
                "hash-custody membership slab contains an unknown label",
            )

        prefix_mask = slab == PREFIX_LABEL
        suffix_mask = slab == SUFFIX_LABEL
        shell_mask = slab >= SHELL_LABEL_BASE
        classified_mask = prefix_mask | suffix_mask | shell_mask
        ignored = int(slab.size - np.count_nonzero(classified_mask))

        prefix_encoded, prefix_count = self._encoded_indices(prefix_mask, start=start)
        suffix_encoded, suffix_count = self._encoded_indices(suffix_mask, start=start)
        shell_encoded, shell_count = self._encoded_indices(shell_mask, start=start)
        self._prefix_digest.update(prefix_encoded)
        self._suffix_digest.update(suffix_encoded)
        self._shell_digest.update(shell_encoded)
        self._fixed_prefix_count += prefix_count
        self._fixed_suffix_count += suffix_count
        self._eligible_route_count += prefix_count + suffix_count + shell_count
        self._ignored_count += ignored

        for offset, expected_route in enumerate(self._shell_route_indices):
            positions = np.flatnonzero(slab == SHELL_LABEL_BASE + offset)
            if positions.size > 1:
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
                    "one shell offset appeared more than once in a membership slab",
                    shell_offset=offset,
                )
            if positions.size:
                observed = start + int(positions[0])
                if self._shell_route_seen[offset] is not None or observed != expected_route:
                    _refuse(
                        V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
                        "membership shell offset differs from the KKT-owned shell",
                        shell_offset=offset,
                        expected_route=expected_route,
                        observed_route=observed,
                    )
                self._shell_route_seen[offset] = observed

        # Every subset is retained because the later exact face/order issuer
        # can request any subset of B<=8.  Each temporary bool/index vector is
        # bounded by this one slab and is discarded before the next subset.
        for subset_mask, digest in enumerate(self._subset_digests):
            include = prefix_mask.copy()
            for offset in range(len(self._shell_route_indices)):
                if subset_mask & (1 << offset):
                    include |= slab == SHELL_LABEL_BASE + offset
            encoded, count = self._encoded_indices(include, start=start)
            digest.update(encoded)
            self._subset_counts[subset_mask] += count

        self._cursor = stop
        self._label_bytes += int(slab.nbytes)
        self._maximum_observed = max(self._maximum_observed, int(slab.nbytes))

    def finalize(
        self,
        *,
        registered_endpoint_counts: Sequence[int],
        registered_endpoint_hashes: Sequence[str],
    ) -> V1DecisionLevelCudaStreamedHashFamily:
        tiles = plan_v1_decision_level_cuda_route_tiles(
            route_count=self._route_count,
            route_tile_size=self._maximum_label_slab_bytes,
        )
        # The generic completion helper checks the intended complete physical
        # partition; the cursor independently checks the slabs actually seen.
        assert_complete_v1_decision_level_cuda_route_scan(
            route_count=self._route_count,
            tiles=tiles,
            scanned_route_count=self._cursor,
        )
        if self._cursor != self._route_count or self._label_bytes != self._route_count:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.ROUTE_SCAN_INCOMPLETE,
                "hash-custody digest did not consume exactly one label per route",
            )
        if tuple(self._shell_route_seen) != self._shell_route_indices:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
                "hash-custody physical scan did not observe the complete KKT shell",
                expected_shell=self._shell_route_indices,
                observed_shell=tuple(self._shell_route_seen),
            )
        if self._eligible_route_count != self._expected_eligible_route_count:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.ROUTE_SCAN_INCOMPLETE,
                "eligible-route census contains an unclassified or extra route",
                expected_eligible_route_count=self._expected_eligible_route_count,
                observed_eligible_route_count=self._eligible_route_count,
            )

        shell_hash = RemovedRouteIndexSet.index_hash_from_encoded_digest(
            route_count=self._route_count,
            count=len(self._shell_route_indices),
            dtype_code=self._dtype_code,
            encoded_sha256=self._shell_digest.hexdigest(),
        )
        if shell_hash != self._expected_shell_hash:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
                "streamed shell digest differs from KKT shell custody",
                expected_shell_route_index_hash=self._expected_shell_hash,
                observed_shell_route_index_hash=shell_hash,
            )
        prefix_hash = RemovedRouteIndexSet.index_hash_from_encoded_digest(
            route_count=self._route_count,
            count=self._fixed_prefix_count,
            dtype_code=self._dtype_code,
            encoded_sha256=self._prefix_digest.hexdigest(),
        )
        suffix_hash = RemovedRouteIndexSet.index_hash_from_encoded_digest(
            route_count=self._route_count,
            count=self._fixed_suffix_count,
            dtype_code=self._dtype_code,
            encoded_sha256=self._suffix_digest.hexdigest(),
        )
        entries: list[V1DecisionLevelCudaShellSubsetHash] = []
        for subset_mask, (digest, count) in enumerate(
            zip(self._subset_digests, self._subset_counts, strict=True)
        ):
            offsets = tuple(
                offset
                for offset in range(len(self._shell_route_indices))
                if subset_mask & (1 << offset)
            )
            expected_count = self._fixed_prefix_count + len(offsets)
            if count != expected_count:
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
                    "one shell-subset digest has an inconsistent route count",
                    shell_offsets=offsets,
                    expected_count=expected_count,
                    observed_count=count,
                )
            entries.append(
                V1DecisionLevelCudaShellSubsetHash(
                    shell_offsets=offsets,
                    removed_count=count,
                    removed_route_index_hash=(
                        RemovedRouteIndexSet.index_hash_from_encoded_digest(
                            route_count=self._route_count,
                            count=count,
                            dtype_code=self._dtype_code,
                            encoded_sha256=digest.hexdigest(),
                        )
                    ),
                )
            )

        counts = tuple(int(value) for value in registered_endpoint_counts)
        hashes = tuple(
            _nonempty(str(value), "registered endpoint hash")
            for value in registered_endpoint_hashes
        )
        if not counts or len(counts) != len(hashes):
            raise ValueError("registered endpoint count/hash axes must be nonempty")
        matched_offsets: list[tuple[int, ...]] = []
        for count, expected_hash in zip(counts, hashes, strict=True):
            if count <= 0 or count > self._route_count:
                raise ValueError("registered endpoint count is outside the route axis")
            matches = tuple(
                entry.shell_offsets
                for entry in entries
                if entry.removed_count == count
                and entry.removed_route_index_hash == expected_hash
            )
            if len(matches) != 1:
                _refuse(
                    V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
                    "a registered frozen endpoint does not match exactly one "
                    "fixed-prefix plus shell-subset digest",
                    endpoint_count=count,
                    expected_removed_route_index_hash=expected_hash,
                    matching_shell_subsets=matches,
                )
            matched_offsets.append(matches[0])
        if any(
            not set(left).issubset(right)
            for left, right in zip(matched_offsets, matched_offsets[1:])
        ):
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
                "registered frozen endpoints are not one nested shell trajectory",
                registered_endpoint_shell_offsets=tuple(matched_offsets),
            )

        commitment = compute_v1_decision_level_complete_subset_mapping_commitment(
            route_count=self._route_count,
            shell_route_index_hash=shell_hash,
            fixed_prefix_route_index_hash=prefix_hash,
            fixed_prefix_count=self._fixed_prefix_count,
            subset_hashes=tuple(entries),
        )
        return V1DecisionLevelCudaStreamedHashFamily(
            route_count=self._route_count,
            shell_route_indices=self._shell_route_indices,
            shell_route_index_hash=shell_hash,
            fixed_prefix_route_index_hash=prefix_hash,
            fixed_prefix_count=self._fixed_prefix_count,
            fixed_suffix_route_index_hash=suffix_hash,
            fixed_suffix_count=self._fixed_suffix_count,
            eligible_route_count=self._eligible_route_count,
            ignored_ineligible_route_count=self._ignored_count,
            subset_hashes=tuple(entries),
            registered_endpoint_shell_offsets=tuple(matched_offsets),
            complete_subset_mapping_commitment=commitment,
            scanned_route_count=self._cursor,
            label_d2h_bytes_total=self._label_bytes,
            maximum_label_slab_bytes_observed=self._maximum_observed,
        )


def stream_v1_decision_level_hash_family_from_membership_slabs(
    *,
    route_count: int,
    shell_route_indices: Sequence[int],
    expected_eligible_route_count: int,
    expected_shell_route_index_hash: str,
    membership_slabs: Iterable[tuple[int, np.ndarray]],
    registered_endpoint_counts: Sequence[int],
    registered_endpoint_hashes: Sequence[str],
    maximum_label_slab_bytes: int = MAXIMUM_HOST_LABEL_SLAB_BYTES,
) -> V1DecisionLevelCudaStreamedHashFamily:
    """Digest a one-pass stream of bounded physical ``uint8`` label slabs.

    This host routine is not a CPU key-classification fallback.  Its only
    accepted input is the bounded membership output of the CUDA classifier.
    It never receives route keys, scores, or a complete physical index axis.
    """

    accumulator = _StreamingHashAccumulator(
        route_count=route_count,
        shell_route_indices=shell_route_indices,
        expected_eligible_route_count=expected_eligible_route_count,
        expected_shell_route_index_hash=expected_shell_route_index_hash,
        maximum_label_slab_bytes=maximum_label_slab_bytes,
    )
    for item in membership_slabs:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("membership slabs must be exact (start, uint8 labels) tuples")
        start, labels = item
        accumulator.update(start=start, labels=labels)
    return accumulator.finalize(
        registered_endpoint_counts=registered_endpoint_counts,
        registered_endpoint_hashes=registered_endpoint_hashes,
    )


def build_v1_decision_level_cuda_shell_hash_custody_from_streamed_family(
    *,
    family: V1DecisionLevelCudaStreamedHashFamily,
    source_evidence_hash: str,
) -> V1DecisionLevelCudaShellHashCustody:
    """Detach one independently owned custody object from a streamed family."""

    if type(family) is not V1DecisionLevelCudaStreamedHashFamily:
        raise TypeError("family must be exact V1DecisionLevelCudaStreamedHashFamily")
    return V1DecisionLevelCudaShellHashCustody(
        source_evidence_hash=_nonempty(source_evidence_hash, "source_evidence_hash"),
        shell_route_index_hash=family.shell_route_index_hash,
        fixed_prefix_route_index_hash=family.fixed_prefix_route_index_hash,
        fixed_prefix_count=family.fixed_prefix_count,
        fixed_suffix_route_index_hash=family.fixed_suffix_route_index_hash,
        fixed_suffix_count=family.fixed_suffix_count,
        endpoint_hash_by_shell_offsets=family.endpoint_hash_by_shell_offsets,
        # IMPORTANT: this legacy-named field is a complete subset-mapping
        # commitment.  It is not the later exact face/order outcome axis.
        outcome_ladder_axis_hash=family.complete_subset_mapping_commitment,
    )


@dataclass(frozen=True, slots=True)
class V1DecisionLevelCudaHashCustodyHBMForecast:
    """Pre-allocation forecast for bounded membership classification."""

    route_count: int
    shell_count: int
    route_tile_size: int
    current_reserved_bytes: int
    tile_workspace_bytes: int
    bounded_shell_workspace_bytes: int
    forecast_incremental_bytes: int
    forecast_peak_reserved_bytes: int
    device_total_memory_bytes: int
    hbm_ceiling_bytes: int
    hbm_ceiling_fraction: float
    admitted: bool
    schema: str = SCHEMA_HBM_FORECAST
    _admission_memory: InitVar[tuple[int, int] | None] = None

    def __post_init__(self, _admission_memory: tuple[int, int] | None) -> None:
        for name in (
            "route_count",
            "shell_count",
            "route_tile_size",
            "current_reserved_bytes",
            "tile_workspace_bytes",
            "bounded_shell_workspace_bytes",
            "forecast_incremental_bytes",
            "forecast_peak_reserved_bytes",
            "device_total_memory_bytes",
            "hbm_ceiling_bytes",
        ):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                _positive_int(value, name)
                if name in {"route_count", "shell_count", "route_tile_size", "device_total_memory_bytes", "hbm_ceiling_bytes"}
                else _nonnegative_int(value, name),
            )
        fraction = float(self.hbm_ceiling_fraction)
        if not math.isfinite(fraction) or not (0.0 < fraction <= 1.0):
            raise ValueError("hbm_ceiling_fraction must be in (0, 1]")
        object.__setattr__(self, "hbm_ceiling_fraction", fraction)
        if self.forecast_incremental_bytes != (
            self.tile_workspace_bytes + self.bounded_shell_workspace_bytes
        ):
            raise ValueError("hash-custody incremental HBM accounting does not close")
        if self.forecast_peak_reserved_bytes != (
            self.current_reserved_bytes + self.forecast_incremental_bytes
        ):
            raise ValueError("hash-custody peak HBM accounting does not close")
        if self.hbm_ceiling_bytes != int(
            math.floor(self.device_total_memory_bytes * fraction)
        ):
            raise ValueError("hash-custody HBM ceiling is inconsistent")
        if _admission_memory is None:
            admitted = self.forecast_peak_reserved_bytes <= self.hbm_ceiling_bytes
        else:
            available, allocated = _admission_memory
            admitted = (
                self.forecast_incremental_bytes <= available
                and allocated + self.forecast_incremental_bytes <= self.hbm_ceiling_bytes
            )
        if self.admitted is not admitted:
            raise ValueError("hash-custody HBM admission differs from its forecast")
        if self.schema != SCHEMA_HBM_FORECAST:
            raise ValueError("unsupported hash-custody HBM forecast schema")

    @property
    def forecast_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-decision-cuda-hash-custody-hbm-",
        )


def estimate_v1_decision_level_cuda_hash_custody_hbm(
    *,
    route_count: int,
    shell_count: int,
    route_tile_size: int,
    device_total_memory_bytes: int,
    current_reserved_bytes: int = 0,
    hbm_ceiling_fraction: float = DEFAULT_H200_HBM_CEILING_FRACTION,
    _admission_memory: tuple[int, int] | None = None,
) -> V1DecisionLevelCudaHashCustodyHBMForecast:
    """Conservatively forecast all temporary CUDA classification storage."""

    total = _positive_int(route_count, "route_count")
    shell = _positive_int(shell_count, "shell_count")
    requested_tile = _positive_int(route_tile_size, "route_tile_size")
    tile = min(total, requested_tile, MAXIMUM_HOST_LABEL_SLAB_BYTES)
    device_total = _positive_int(
        device_total_memory_bytes,
        "device_total_memory_bytes",
    )
    reserved = _nonnegative_int(current_reserved_bytes, "current_reserved_bytes")
    fraction = float(hbm_ceiling_fraction)
    if not math.isfinite(fraction) or not (0.0 < fraction <= 1.0):
        raise ValueError("hbm_ceiling_fraction must be in (0, 1]")
    # The classifier uses route-length bool/int8 work vectors plus FP64
    # interval temporaries.  320 bytes/route deliberately upper-bounds the
    # simultaneously live eager-torch tensors in the implementation below.
    tile_workspace = tile * HASH_CUSTODY_TILE_WORKSPACE_BYTES_PER_ROUTE
    shell_workspace = (
        HASH_CUSTODY_SHELL_WORKSPACE_BASE_BYTES
        + shell * (2 * 3 * np.dtype(np.float64).itemsize + np.dtype(np.int64).itemsize)
    )
    incremental = tile_workspace + shell_workspace
    peak = reserved + incremental
    ceiling = int(math.floor(device_total * fraction))
    return V1DecisionLevelCudaHashCustodyHBMForecast(
        route_count=total,
        shell_count=shell,
        route_tile_size=tile,
        current_reserved_bytes=reserved,
        tile_workspace_bytes=tile_workspace,
        bounded_shell_workspace_bytes=shell_workspace,
        forecast_incremental_bytes=incremental,
        forecast_peak_reserved_bytes=peak,
        device_total_memory_bytes=device_total,
        hbm_ceiling_bytes=ceiling,
        hbm_ceiling_fraction=fraction,
        admitted=(
            peak <= ceiling
            if _admission_memory is None
            else incremental <= _admission_memory[0]
            and _admission_memory[1] + incremental <= ceiling
        ),
        _admission_memory=_admission_memory,
    )


def _forecast_hbm_from_device(
    *,
    torch: Any,
    device: Any,
    route_count: int,
    shell_count: int,
    design: V1DecisionLevelCudaIssuerDesign,
) -> V1DecisionLevelCudaHashCustodyHBMForecast:
    properties = torch.cuda.get_device_properties(device)
    return estimate_v1_decision_level_cuda_hash_custody_hbm(
        route_count=route_count,
        shell_count=shell_count,
        route_tile_size=design.route_tile_size,
        device_total_memory_bytes=int(properties.total_memory),
        current_reserved_bytes=int(torch.cuda.memory_reserved(device)),
        hbm_ceiling_fraction=design.hbm_ceiling_fraction,
        _admission_memory=(
            cuda_reusable_memory_bytes(device, framework=torch),
            int(torch.cuda.memory_allocated(device)),
        ),
    )


def _require_cuda_tensor(
    torch: Any,
    value: Any,
    *,
    name: str,
    dtype: Any,
    shape: tuple[int, ...],
    device: str,
) -> Any:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be one torch tensor")
    if (
        getattr(value.device, "type", None) != "cuda"
        or str(value.device) != device
        or value.dtype != dtype
        or tuple(int(item) for item in value.shape) != shape
        or not value.is_contiguous()
    ):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.TENSOR_CONTRACT,
            f"{name} left its exact contiguous CUDA tensor contract",
        )
    return value


def _lex_relation_to_one_shell_cuda(
    torch: Any,
    *,
    lower: Any,
    upper: Any,
    shell_lower: Any,
    shell_upper: Any,
) -> tuple[Any, Any]:
    """Vectorized per-route lex interval relation, matching issuer semantics."""

    rows = int(lower.shape[0])
    relation = torch.zeros((rows,), device=lower.device, dtype=torch.int8)
    margin = torch.zeros((rows,), device=lower.device, dtype=torch.float64)
    active = torch.ones((rows,), device=lower.device, dtype=torch.bool)
    for component in range(3):
        left_lo = lower[:, component]
        left_hi = upper[:, component]
        right_lo = shell_lower[component]
        right_hi = shell_upper[component]
        left_precedes = active & (left_hi < right_lo)
        right_precedes = active & (right_hi < left_lo)
        exact_equal = (
            active
            & (left_lo == left_hi)
            & (left_hi == right_lo)
            & (right_lo == right_hi)
        )
        ambiguous = active & ~(left_precedes | right_precedes | exact_equal)
        relation = torch.where(
            left_precedes,
            torch.full_like(relation, -1),
            relation,
        )
        relation = torch.where(
            right_precedes,
            torch.full_like(relation, 1),
            relation,
        )
        left_margin = torch.nextafter(
            right_lo - left_hi,
            torch.full_like(left_hi, -math.inf),
        )
        right_margin = torch.nextafter(
            left_lo - right_hi,
            torch.full_like(left_lo, -math.inf),
        )
        margin = torch.where(left_precedes, left_margin, margin)
        margin = torch.where(right_precedes, right_margin, margin)
        relation = torch.where(ambiguous, torch.zeros_like(relation), relation)
        active = exact_equal
    relation = torch.where(active, torch.full_like(relation, 2), relation)
    return relation, margin


def _membership_slab_cuda(
    torch: Any,
    *,
    lower: Any,
    upper: Any,
    resource: Any,
    corrected_work: Any,
    eligible: Any,
    shell_route_indices: tuple[int, ...],
    shell_lower: Any,
    shell_upper: Any,
    tile_start: int,
    numerical_key_radius: float,
) -> Any:
    """Classify one CUDA tile or refuse every ambiguous source boundary."""

    length = int(lower.shape[0])
    finite_ordered = (
        torch.isfinite(lower).all(dim=1)
        & torch.isfinite(upper).all(dim=1)
        & (lower <= upper).all(dim=1)
        & torch.isfinite(resource)
        & torch.isfinite(corrected_work)
        & (resource > 0.0)
    )
    if bool((eligible & ~finite_ordered).any().item()):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.TENSOR_CONTRACT,
            "eligible key/work/resource rows are nonfinite, reversed, or nonpositive",
        )
    shell_mask = torch.zeros((length,), device=lower.device, dtype=torch.bool)
    shell_offsets_in_tile: list[tuple[int, int]] = []
    tile_stop = tile_start + length
    for offset, route in enumerate(shell_route_indices):
        if tile_start <= route < tile_stop:
            local = route - tile_start
            shell_mask[local] = True
            shell_offsets_in_tile.append((local, offset))
    if bool((shell_mask & ~eligible).any().item()):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
            "KKT theorem shell contains an ineligible route",
        )
    outside = eligible & ~shell_mask
    ratio = corrected_work / resource
    ratio_enclosed = (ratio >= lower[:, 2]) & (ratio <= upper[:, 2])
    if bool((outside & ~ratio_enclosed).any().item()):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
            "canonical work/resource interval does not enclose the live source",
        )
    if bool((outside & (lower[:, 1] <= 0.0) & (upper[:, 1] >= 0.0)).any().item()):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
            "outside reduced-advantage interval touches the active boundary",
        )

    prefix = outside.clone()
    suffix = outside.clone()
    allowed_margin = 2.0 * float(numerical_key_radius)
    for shell_offset in range(len(shell_route_indices)):
        relation, margin = _lex_relation_to_one_shell_cuda(
            torch,
            lower=lower,
            upper=upper,
            shell_lower=shell_lower[shell_offset],
            shell_upper=shell_upper[shell_offset],
        )
        relation_allowed = (relation == -1) | (relation == 1)
        if bool((outside & (~relation_allowed | (margin <= allowed_margin))).any().item()):
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
                "outside key interval overlaps the shell or lacks a strict "
                "numerical margin",
                shell_offset=shell_offset,
            )
        prefix &= relation == -1
        suffix &= relation == 1
    if bool((outside & ~(prefix | suffix)).any().item()):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.DECISION_BOUNDARY_TOUCHED,
            "one outside route lies between active-shell keys",
        )

    labels = torch.zeros((length,), device=lower.device, dtype=torch.uint8)
    labels[prefix] = PREFIX_LABEL
    labels[suffix] = SUFFIX_LABEL
    for local, offset in shell_offsets_in_tile:
        labels[local] = SHELL_LABEL_BASE + offset
    return labels


@dataclass(frozen=True, slots=True)
class V1DecisionLevelCudaHashCustodyReceipt:
    """Array-free evidence for one independently streamed custody object."""

    problem_hash: str
    policy_hash: str
    branch_result_hash: str
    branch_receipt_hash: str
    source_rounding_hash: str
    endpoint_action_axis_hash: str
    geometry_uncertainty_hash: str
    source_evidence_hash: str
    theorem_state_request_hash: str
    theorem_state_axis_hash: str
    route_axis_hash: str
    route_count: int
    shell_route_index_hash: str
    shell_count: int
    streamed_hash_family_hash: str
    hbm_forecast_hash: str
    custody_hash: str
    complete_subset_mapping_commitment: str
    registered_endpoint_shell_offsets: tuple[tuple[int, ...], ...]
    label_d2h_bytes_total: int
    maximum_label_slab_bytes_observed: int
    subset_mapping_is_not_face_outcome_axis: bool = True
    complete_physical_scan: bool = True
    cpu_key_classification_used: bool = False
    route_scale_host_row_retention_used: bool = False
    full_key_or_index_d2h_used: bool = False
    schema: str = SCHEMA_RECEIPT

    def __post_init__(self) -> None:
        for name in (
            "problem_hash",
            "policy_hash",
            "branch_result_hash",
            "branch_receipt_hash",
            "source_rounding_hash",
            "endpoint_action_axis_hash",
            "geometry_uncertainty_hash",
            "source_evidence_hash",
            "theorem_state_request_hash",
            "theorem_state_axis_hash",
            "route_axis_hash",
            "shell_route_index_hash",
            "streamed_hash_family_hash",
            "hbm_forecast_hash",
            "custody_hash",
            "complete_subset_mapping_commitment",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(self, "route_count", _positive_int(self.route_count, "route_count"))
        object.__setattr__(self, "shell_count", _positive_int(self.shell_count, "shell_count"))
        object.__setattr__(
            self,
            "label_d2h_bytes_total",
            _positive_int(self.label_d2h_bytes_total, "label_d2h_bytes_total"),
        )
        object.__setattr__(
            self,
            "maximum_label_slab_bytes_observed",
            _positive_int(
                self.maximum_label_slab_bytes_observed,
                "maximum_label_slab_bytes_observed",
            ),
        )
        endpoints = tuple(tuple(value) for value in self.registered_endpoint_shell_offsets)
        object.__setattr__(self, "registered_endpoint_shell_offsets", endpoints)
        flags = (
            self.subset_mapping_is_not_face_outcome_axis,
            self.complete_physical_scan,
            self.cpu_key_classification_used,
            self.route_scale_host_row_retention_used,
            self.full_key_or_index_d2h_used,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("hash-custody receipt flags must be exact booleans")
        if flags != (True, True, False, False, False):
            raise PermissionError("hash-custody receipt crossed its production boundary")
        if self.label_d2h_bytes_total != self.route_count:
            raise ValueError("hash-custody receipt must transfer one uint8 label per route")
        if self.schema != SCHEMA_RECEIPT:
            raise ValueError("unsupported hash-custody receipt schema")

    @property
    def receipt_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-decision-cuda-hash-custody-receipt-",
        )


@dataclass(frozen=True, slots=True)
class V1DecisionLevelCudaHashCustodyIssuance:
    """Live custody plus its durable, array-free issuance receipt."""

    custody: V1DecisionLevelCudaShellHashCustody
    receipt: V1DecisionLevelCudaHashCustodyReceipt

    def __post_init__(self) -> None:
        if type(self.custody) is not V1DecisionLevelCudaShellHashCustody:
            raise TypeError("issuance requires exact shell hash custody")
        if type(self.receipt) is not V1DecisionLevelCudaHashCustodyReceipt:
            raise TypeError("issuance requires exact hash-custody receipt")
        if (
            self.custody.custody_hash != self.receipt.custody_hash
            or self.custody.outcome_ladder_axis_hash
            != self.receipt.complete_subset_mapping_commitment
        ):
            raise ValueError("hash custody and issuance receipt do not close")


@dataclass(frozen=True, slots=True)
class V1DecisionLevelCudaIntegratedIssuance:
    """Durable output of the complete KKT→hash→theorem source joint."""

    result: V1DecisionLevelCudaIssuerResult
    hash_custody_receipt: V1DecisionLevelCudaHashCustodyReceipt

    def __post_init__(self) -> None:
        if type(self.result) is not V1DecisionLevelCudaIssuerResult:
            raise TypeError("integrated issuance requires the exact issuer result")
        if type(self.hash_custody_receipt) is not (
            V1DecisionLevelCudaHashCustodyReceipt
        ):
            raise TypeError(
                "integrated issuance requires the exact hash-custody receipt"
            )
        binding = self.result.private_primal_binding
        receipt = self.hash_custody_receipt
        if (
            binding.hash_custody_hash != receipt.custody_hash
            or binding.source_evidence_hash != receipt.source_evidence_hash
            or binding.shell_route_index_hash != receipt.shell_route_index_hash
            or binding.problem_hash != receipt.problem_hash
            or binding.policy_hash != receipt.policy_hash
            or binding.branch_result_hash != receipt.branch_result_hash
        ):
            raise ValueError(
                "integrated CUDA issuer result and independent hash receipt "
                "do not close"
            )


def issue_v1_decision_level_cuda_certificate_from_kkt_custody(
    *,
    execution_bundle: V1RelaxedKKTDeviceShellExecutionBundle,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    branch_identity: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    literal_j_fixed_projection: V1LiteralJFixedDirectionProjectionReceipt,
    nominal_witness: V1RelaxedKKTAdaptiveRankBranchWitness,
    source_evidence_hash: str,
    design: V1DecisionLevelCudaIssuerDesign = V1DecisionLevelCudaIssuerDesign(),
) -> V1DecisionLevelCudaIntegratedIssuance:
    """Run the canonical noncircular source-theorem issuance transaction.

    The KKT bundle first lends its private FP64 theorem state.  Independent
    hash custody then performs the bounded CUDA membership scan over that
    live lease.  Only after the streamed mask identities close do we bind the
    literal-J private-primal receipt and ask the CUDA theorem issuer for its
    exact face/order receipts.  The dynamic leases are evicted on every exit;
    the returned objects are array-free.
    """

    with V1DecisionLevelCudaKKTLeaseAdapter(
        execution_bundle=execution_bundle,
        problem=problem,
        policy=policy,
        branch_result=branch_result,
        branch_identity=branch_identity,
        geometry_uncertainty=geometry_uncertainty,
        literal_j_fixed_projection=literal_j_fixed_projection,
        source_evidence_hash=_nonempty(
            source_evidence_hash,
            "source_evidence_hash",
        ),
        design=design,
    ) as adapter:
        receipt = adapter.hash_custody_receipt
        if type(receipt) is not V1DecisionLevelCudaHashCustodyReceipt:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
                "canonical KKT adaptation did not emit its exact streaming "
                "hash-custody receipt",
            )
        result = issue_v1_decision_level_cuda_certificate(
            problem=problem,
            policy=policy,
            branch_result=branch_result,
            branch_identity=branch_identity,
            geometry_uncertainty=geometry_uncertainty,
            nominal_witness=nominal_witness,
            source_lease=adapter.source_lease,
            private_primal_binding=adapter.private_primal_binding,
            design=design,
        )
    return V1DecisionLevelCudaIntegratedIssuance(
        result=result,
        hash_custody_receipt=receipt,
    )


def _assert_source_lineage(
    *,
    theorem_lease: V1RelaxedKKTDecisionLevelTheoremStateLease,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    branch_identity: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    source_evidence_hash: str,
) -> None:
    request = theorem_lease.request
    frozen = branch_identity.frozen_ladder
    if (
        theorem_lease.branch is not branch_result.branch
        or branch_identity.branch is not branch_result.branch
        or theorem_lease.route_axis_hash != problem.route_axis_hash
        or theorem_lease.route_count != problem.route_count
        or theorem_lease.rank != problem.rank
        or request.problem_hash != problem.problem_hash
        or request.policy_hash != policy.policy_hash
        or request.branch_result_hash != branch_result.result_hash
        or request.branch_receipt_hash != branch_result.receipt.receipt_hash
        or request.source_rounding_hash != branch_identity.source_rounding_hash
        or request.endpoint_action_axis_hash != branch_identity.endpoint_action_axis_hash
        or request.geometry_uncertainty_hash != geometry_uncertainty.uncertainty_hash
        or request.source_evidence_hash != source_evidence_hash
        or theorem_lease.source_evidence_hash != source_evidence_hash
        or frozen.route_axis_hash != problem.route_axis_hash
        or frozen.route_count != problem.route_count
        or geometry_uncertainty.problem_hash != problem.problem_hash
        or geometry_uncertainty.route_axis_hash != problem.route_axis_hash
        or geometry_uncertainty.source_geometry_hash != problem.source_geometry_hash
        or not geometry_uncertainty.source_derived_from_registered_geometry_receipts
        or branch_result.receipt.problem_hash != problem.problem_hash
        or branch_result.receipt.policy_hash != policy.policy_hash
    ):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.LITERAL_J_LINEAGE_MISMATCH,
            "hash-custody theorem lease, branch ladder, geometry, and source "
            "evidence do not bind one exact chart",
        )


def issue_v1_decision_level_cuda_shell_hash_custody(
    *,
    theorem_lease: V1RelaxedKKTDecisionLevelTheoremStateLease,
    problem: V1RelaxedKKTProblem,
    policy: V1RelaxedKKTPolicy,
    branch_result: V1RelaxedKKTBranchResult,
    branch_identity: V1RelaxedKKTDeviceShellBranchIdentityReceipt,
    geometry_uncertainty: V1RelaxedKKTGeometryUncertaintySet,
    source_evidence_hash: str,
    design: V1DecisionLevelCudaIssuerDesign,
) -> V1DecisionLevelCudaHashCustodyIssuance:
    """Issue independent hash custody from one live narrow KKT CUDA lease."""

    for value, expected, name in (
        (
            theorem_lease,
            V1RelaxedKKTDecisionLevelTheoremStateLease,
            "theorem_lease",
        ),
        (problem, V1RelaxedKKTProblem, "problem"),
        (policy, V1RelaxedKKTPolicy, "policy"),
        (branch_result, V1RelaxedKKTBranchResult, "branch_result"),
        (
            branch_identity,
            V1RelaxedKKTDeviceShellBranchIdentityReceipt,
            "branch_identity",
        ),
        (
            geometry_uncertainty,
            V1RelaxedKKTGeometryUncertaintySet,
            "geometry_uncertainty",
        ),
        (design, V1DecisionLevelCudaIssuerDesign, "design"),
    ):
        if type(value) is not expected:
            raise TypeError(f"{name} must be exact {expected.__name__}")
    evidence_hash = _nonempty(source_evidence_hash, "source_evidence_hash")
    _assert_source_lineage(
        theorem_lease=theorem_lease,
        problem=problem,
        policy=policy,
        branch_result=branch_result,
        branch_identity=branch_identity,
        geometry_uncertainty=geometry_uncertainty,
        source_evidence_hash=evidence_hash,
    )

    try:
        import torch
    except Exception as error:  # pragma: no cover - environment dependent.
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.CUDA_UNAVAILABLE,
            "PyTorch CUDA runtime is unavailable for hash custody",
            cause=type(error).__name__,
        )
    assert_v1_decision_level_cuda_hash_custody_device(
        device=theorem_lease.device,
        cuda_available=bool(torch.cuda.is_available()),
    )
    device = theorem_lease.device
    shell = tuple(theorem_lease.shell_route_indices)
    if (
        not shell
        or shell != tuple(sorted(set(shell)))
        or shell[0] < 0
        or shell[-1] >= problem.route_count
        or len(shell) > min(design.maximum_shell_routes, DEFAULT_MAXIMUM_SHELL_ROUTES)
    ):
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.SHELL_LIMIT_EXCEEDED,
            "the KKT theorem shell is malformed or exceeds the custody limit",
        )

    key_lower = _require_cuda_tensor(
        torch,
        theorem_lease.canonical_key_lower,
        name="canonical_key_lower",
        dtype=torch.float64,
        shape=(problem.route_count, 3),
        device=device,
    )
    key_upper = _require_cuda_tensor(
        torch,
        theorem_lease.canonical_key_upper,
        name="canonical_key_upper",
        dtype=torch.float64,
        shape=(problem.route_count, 3),
        device=device,
    )
    resource = _require_cuda_tensor(
        torch,
        theorem_lease.resource,
        name="resource",
        dtype=torch.float64,
        shape=(problem.route_count,),
        device=device,
    )
    corrected_work = _require_cuda_tensor(
        torch,
        theorem_lease.corrected_work,
        name="corrected_work",
        dtype=torch.float64,
        shape=(problem.route_count,),
        device=device,
    )
    eligible = _require_cuda_tensor(
        torch,
        theorem_lease.eligible_mask,
        name="eligible_mask",
        dtype=torch.bool,
        shape=(problem.route_count,),
        device=device,
    )
    _require_cuda_tensor(
        torch,
        theorem_lease.residual_diagonal,
        name="residual_diagonal",
        dtype=torch.float64,
        shape=(problem.route_count,),
        device=device,
    )
    _require_cuda_tensor(
        torch,
        theorem_lease.low_rank_factor,
        name="low_rank_factor",
        dtype=torch.float64,
        shape=(problem.rank, problem.route_count),
        device=device,
    )

    torch_device = torch.device(device)
    forecast = _forecast_hbm_from_device(
        torch=torch,
        device=torch_device,
        route_count=problem.route_count,
        shell_count=len(shell),
        design=design,
    )
    if not forecast.admitted:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.HBM_FORECAST_EXCEEDED,
            "hash-custody CUDA tile forecast exceeds the registered HBM ceiling",
            forecast_hash=forecast.forecast_hash,
            forecast_peak_reserved_bytes=forecast.forecast_peak_reserved_bytes,
            hbm_ceiling_bytes=forecast.hbm_ceiling_bytes,
        )

    shell_tensor = torch.tensor(shell, device=torch_device, dtype=torch.long)
    shell_lower = key_lower.index_select(0, shell_tensor)
    shell_upper = key_upper.index_select(0, shell_tensor)
    expected_shell_hash = RemovedRouteIndexSet.from_indices(
        np.asarray(shell, dtype=np.int64),
        route_count=problem.route_count,
    ).index_hash
    if expected_shell_hash != theorem_lease.shell_route_index_hash:
        _refuse(
            V1DecisionLevelCudaIssuerRefusalCode.HASH_CUSTODY_REQUIRED,
            "bounded theorem shell differs from its KKT-owned hash",
        )
    expected_eligible_count = int(eligible.sum(dtype=torch.int64).item())
    numerical_key_radius = max(
        geometry_uncertainty.solver_reduced_advantage_numerical_linf_radius,
        geometry_uncertainty.relaxed_gate_numerical_linf_radius,
        geometry_uncertainty.corrected_work_ratio_deterministic_linf_radius,
    )
    accumulator = _StreamingHashAccumulator(
        route_count=problem.route_count,
        shell_route_indices=shell,
        expected_eligible_route_count=expected_eligible_count,
        expected_shell_route_index_hash=expected_shell_hash,
        maximum_label_slab_bytes=forecast.route_tile_size,
    )
    tiles = plan_v1_decision_level_cuda_route_tiles(
        route_count=problem.route_count,
        route_tile_size=forecast.route_tile_size,
    )
    scanned = 0
    for tile in tiles:
        labels_cuda = _membership_slab_cuda(
            torch,
            lower=key_lower[tile.start : tile.stop],
            upper=key_upper[tile.start : tile.stop],
            resource=resource[tile.start : tile.stop],
            corrected_work=corrected_work[tile.start : tile.stop],
            eligible=eligible[tile.start : tile.stop],
            shell_route_indices=shell,
            shell_lower=shell_lower,
            shell_upper=shell_upper,
            tile_start=tile.start,
            numerical_key_radius=numerical_key_radius,
        )
        labels_host = labels_cuda.detach().cpu().numpy()
        if labels_host.dtype != np.dtype(np.uint8) or not labels_host.flags.c_contiguous:
            _refuse(
                V1DecisionLevelCudaIssuerRefusalCode.ROUTE_SCALE_D2H_FORBIDDEN,
                "CUDA membership copy did not produce one contiguous uint8 slab",
            )
        accumulator.update(start=tile.start, labels=labels_host)
        scanned += tile.length
        del labels_host, labels_cuda
    assert_complete_v1_decision_level_cuda_route_scan(
        route_count=problem.route_count,
        tiles=tiles,
        scanned_route_count=scanned,
    )

    frozen = branch_identity.frozen_ladder
    family = accumulator.finalize(
        registered_endpoint_counts=frozen.cumulative_counts,
        registered_endpoint_hashes=tuple(
            action.removed_route_index_hash for action in frozen.actions
        ),
    )
    custody = build_v1_decision_level_cuda_shell_hash_custody_from_streamed_family(
        family=family,
        source_evidence_hash=evidence_hash,
    )
    receipt = V1DecisionLevelCudaHashCustodyReceipt(
        problem_hash=problem.problem_hash,
        policy_hash=policy.policy_hash,
        branch_result_hash=branch_result.result_hash,
        branch_receipt_hash=branch_result.receipt.receipt_hash,
        source_rounding_hash=branch_identity.source_rounding_hash,
        endpoint_action_axis_hash=branch_identity.endpoint_action_axis_hash,
        geometry_uncertainty_hash=geometry_uncertainty.uncertainty_hash,
        source_evidence_hash=evidence_hash,
        theorem_state_request_hash=theorem_lease.request.request_hash,
        theorem_state_axis_hash=theorem_lease.theorem_state_axis_hash,
        route_axis_hash=problem.route_axis_hash,
        route_count=problem.route_count,
        shell_route_index_hash=family.shell_route_index_hash,
        shell_count=len(shell),
        streamed_hash_family_hash=family.family_hash,
        hbm_forecast_hash=forecast.forecast_hash,
        custody_hash=custody.custody_hash,
        complete_subset_mapping_commitment=(
            family.complete_subset_mapping_commitment
        ),
        registered_endpoint_shell_offsets=(
            family.registered_endpoint_shell_offsets
        ),
        label_d2h_bytes_total=family.label_d2h_bytes_total,
        maximum_label_slab_bytes_observed=(
            family.maximum_label_slab_bytes_observed
        ),
    )
    return V1DecisionLevelCudaHashCustodyIssuance(
        custody=custody,
        receipt=receipt,
    )


__all__ = [
    "COMPLETE_SUBSET_MAPPING_COMMITMENT_KIND",
    "MAXIMUM_HOST_LABEL_SLAB_BYTES",
    "PREFIX_LABEL",
    "SHELL_LABEL_BASE",
    "SUFFIX_LABEL",
    "V1DecisionLevelCudaHashCustodyHBMForecast",
    "V1DecisionLevelCudaHashCustodyIssuance",
    "V1DecisionLevelCudaHashCustodyReceipt",
    "V1DecisionLevelCudaIntegratedIssuance",
    "V1DecisionLevelCudaShellSubsetHash",
    "V1DecisionLevelCudaStreamedHashFamily",
    "assert_v1_decision_level_cuda_hash_custody_device",
    "build_v1_decision_level_cuda_shell_hash_custody_from_streamed_family",
    "compute_v1_decision_level_complete_subset_mapping_commitment",
    "estimate_v1_decision_level_cuda_hash_custody_hbm",
    "issue_v1_decision_level_cuda_certificate_from_kkt_custody",
    "issue_v1_decision_level_cuda_shell_hash_custody",
    "stream_v1_decision_level_hash_family_from_membership_slabs",
]
