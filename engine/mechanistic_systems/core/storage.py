"""Bit-packed storage for canonical circuit realizations."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from circuitflow.artifacts import (
    ArtifactManifest,
    ArtifactStore,
    ArtifactType,
    LifecycleState,
    PayloadRef,
    infer_lifecycle_state,
)
from contracts.hashing import stable_digest
from mechanistic_systems.contracts import CircuitPlan
from mechanistic_systems.storage import read_parquet_rows, write_parquet_rows

from .topology import RouteCoordinate

BLOCK_ROUTE_AXIS_SCHEMA = "mechanistic_block_route_axis_v1"


def _validate_packed_padding_bits(
    packed_masks: np.ndarray,
    *,
    route_count: int,
) -> None:
    trailing = int(route_count) % 8
    if trailing == 0:
        return
    disallowed_mask = np.uint8(0xFF ^ ((1 << trailing) - 1))
    if np.any(np.bitwise_and(packed_masks[:, -1], disallowed_mask)):
        raise ValueError("packed circuit payload has nonzero route-padding bits")


@dataclass(frozen=True)
class RouteBlockSpec:
    """Compact declaration of a contiguous block of stable route coordinates.

    Large CNN circuit masks contain millions of channel-pair routes. Persisting
    one string ID per route would dwarf the mask itself. A block spec retains
    the exact ordered coordinate rule in O(number of layers) space and lowers
    an individual route to ``RouteCoordinate`` only when a consumer requests
    it.
    """

    architecture_id: str
    topology_hash: str
    layer_id: str
    route_kind: str
    shape: tuple[int, ...]
    offset: int
    flattening: str = "c_order_target_then_source"

    @property
    def route_count(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def stop(self) -> int:
        return int(self.offset) + self.route_count

    def validate(self) -> None:
        if not self.architecture_id or not self.layer_id or not self.route_kind:
            raise ValueError("route block identity fields must be nonempty")
        if len(self.topology_hash) != 64:
            raise ValueError("route block topology hash must be a SHA-256 digest")
        if not self.shape or any(int(value) <= 0 for value in self.shape):
            raise ValueError("route block shapes must be positive")
        if self.route_kind == "conv_kernel_weight":
            # Kernel-weight routes are (target, source, kh, kw)
            # under their own versioned convention; taps are NEVER
            # flattened into the source axis (that would silently
            # redefine source_coordinate away from the input channel).
            if len(self.shape) != 4:
                raise ValueError(
                    "conv_kernel_weight route blocks require "
                    "(target, source, kh, kw) shapes"
                )
            if self.flattening != "c_order_target_source_taps_v1":
                raise ValueError(
                    "conv_kernel_weight blocks require the "
                    "c_order_target_source_taps_v1 flattening convention"
                )
        else:
            if len(self.shape) > 2:
                raise ValueError(
                    "route blocks require one- or two-dimensional "
                    "positive shapes"
                )
            if self.flattening != "c_order_target_then_source":
                raise ValueError(
                    "unsupported route block flattening convention"
                )
        if int(self.offset) < 0:
            raise ValueError("route block offset must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_name": BLOCK_ROUTE_AXIS_SCHEMA,
            "architecture_id": self.architecture_id,
            "topology_hash": self.topology_hash,
            "layer_id": self.layer_id,
            "route_kind": self.route_kind,
            "shape": list(self.shape),
            "offset": int(self.offset),
            "route_count": self.route_count,
            "flattening": self.flattening,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> RouteBlockSpec:
        shape = payload.get("shape", ())
        if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
            raise ValueError("route block shape must be a sequence")
        result = cls(
            architecture_id=str(payload.get("architecture_id", "")),
            topology_hash=str(payload.get("topology_hash", "")),
            layer_id=str(payload.get("layer_id", "")),
            route_kind=str(payload.get("route_kind", "")),
            shape=tuple(int(value) for value in shape),
            offset=int(payload.get("offset", 0)),
            flattening=str(payload.get("flattening", "c_order_target_then_source")),
        )
        result.validate()
        return result

    @property
    def block_id(self) -> str:
        identity = dict(self.to_dict())
        identity.pop("offset")
        return f"route-block:v1:{stable_digest(identity)}"

    def route_coordinate(self, flat_index: int) -> RouteCoordinate:
        """Lower one implicit block entry to its stable executable coordinate."""

        self.validate()
        index = int(flat_index)
        if index < 0 or index >= self.route_count:
            raise IndexError(index)
        if self.route_kind == "conv_kernel_weight":
            target_index, source_index, kh, kw = np.unravel_index(
                index, self.shape, order="C"
            )
            return RouteCoordinate(
                architecture_id=self.architecture_id,
                topology_hash=self.topology_hash,
                layer_id=self.layer_id,
                route_kind=self.route_kind,
                source_coordinate=(int(source_index),),
                target_coordinate=(int(target_index),),
                parameter_coordinate=(int(kh), int(kw)),
            )
        if self.route_kind == "residual_stream_gate":
            source = (index,)
            target = (0,)
        elif len(self.shape) == 1:
            source = target = (index,)
        else:
            target_index, source_index = np.unravel_index(index, self.shape, order="C")
            source = (int(source_index),)
            target = (int(target_index),)
        return RouteCoordinate(
            architecture_id=self.architecture_id,
            topology_hash=self.topology_hash,
            layer_id=self.layer_id,
            route_kind=self.route_kind,
            source_coordinate=source,
            target_coordinate=target,
        )


def ordered_block_route_axis_hash(blocks: Sequence[RouteBlockSpec]) -> str:
    normalized = tuple(blocks)
    if not normalized:
        raise ValueError("block route axis must be nonempty")
    expected_offset = 0
    for block in normalized:
        block.validate()
        if block.offset != expected_offset:
            raise ValueError("route blocks must form one ordered contiguous axis")
        expected_offset = block.stop
    return stable_digest(
        {
            "schema_name": BLOCK_ROUTE_AXIS_SCHEMA,
            "blocks": [block.to_dict() for block in normalized],
        }
    )


def rebase_route_blocks(blocks: Sequence[RouteBlockSpec]) -> tuple[RouteBlockSpec, ...]:
    """Return an ordered subset with a fresh contiguous implicit route axis."""

    offset = 0
    rebased: list[RouteBlockSpec] = []
    for block in blocks:
        block.validate()
        current = replace(block, offset=offset)
        current.validate()
        rebased.append(current)
        offset = current.stop
    result = tuple(rebased)
    ordered_block_route_axis_hash(result)
    return result


@dataclass(frozen=True)
class BlockBitPackedCircuitSet:
    """Bit-packed query masks over a compact, block-declared route axis."""

    query_ids: tuple[str, ...]
    route_blocks: tuple[RouteBlockSpec, ...]
    packed_masks: np.ndarray
    bitorder: str = "little"

    @property
    def total_route_count(self) -> int:
        return self.route_blocks[-1].stop if self.route_blocks else 0

    @property
    def route_axis_hash(self) -> str:
        return ordered_block_route_axis_hash(self.route_blocks)

    @classmethod
    def from_block_masks(
        cls,
        masks_by_query: Mapping[str, Mapping[str, np.ndarray]],
        *,
        route_blocks: Sequence[RouteBlockSpec],
    ) -> BlockBitPackedCircuitSet:
        blocks = tuple(route_blocks)
        ordered_block_route_axis_hash(blocks)
        query_ids = tuple(str(value) for value in masks_by_query)
        if not query_ids or len(set(query_ids)) != len(query_ids):
            raise ValueError("block circuit query IDs must be nonempty and unique")
        rows: list[np.ndarray] = []
        for query_id in query_ids:
            query_blocks = masks_by_query[query_id]
            expected_layers = {block.layer_id for block in blocks}
            if set(query_blocks) != expected_layers:
                raise ValueError(f"query {query_id!r} does not match the declared route blocks")
            flattened: list[np.ndarray] = []
            for block in blocks:
                values = np.asarray(query_blocks[block.layer_id], dtype=bool)
                if values.shape != block.shape:
                    raise ValueError(f"query {query_id!r} block {block.layer_id!r} has the wrong shape")
                flattened.append(values.reshape(-1, order="C"))
            rows.append(np.packbits(np.concatenate(flattened), bitorder="little"))
        result = cls(
            query_ids=query_ids,
            route_blocks=blocks,
            packed_masks=np.stack(rows, axis=0).astype(np.uint8, copy=False),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.bitorder != "little":
            raise ValueError("canonical circuit masks use little-endian bit packing")
        if not self.query_ids or len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("block circuit query IDs must be nonempty and unique")
        ordered_block_route_axis_hash(self.route_blocks)
        expected_columns = (self.total_route_count + 7) // 8
        if self.packed_masks.dtype != np.uint8 or self.packed_masks.shape != (
            len(self.query_ids),
            expected_columns,
        ):
            raise ValueError("packed block circuit payload has an invalid shape or dtype")
        _validate_packed_padding_bits(
            self.packed_masks,
            route_count=self.total_route_count,
        )

    def unpack(self) -> np.ndarray:
        self.validate()
        values = np.unpackbits(
            self.packed_masks,
            axis=1,
            count=self.total_route_count,
            bitorder=self.bitorder,
        )
        return values.astype(bool, copy=False)

    def unpack_block(self, query_id: str, layer_id: str) -> np.ndarray:
        self.validate()
        try:
            query_index = self.query_ids.index(str(query_id))
        except ValueError as error:
            raise KeyError(query_id) from error
        block = next((item for item in self.route_blocks if item.layer_id == str(layer_id)), None)
        if block is None:
            raise KeyError(layer_id)
        row = np.unpackbits(
            self.packed_masks[query_index],
            count=self.total_route_count,
            bitorder=self.bitorder,
        )
        return row[block.offset : block.stop].reshape(block.shape, order="C").astype(bool, copy=False)

    def unpack_query(self, query_id: str) -> dict[str, np.ndarray]:
        """Decode one query once and expose its ordered executable blocks."""

        self.validate()
        try:
            query_index = self.query_ids.index(str(query_id))
        except ValueError as error:
            raise KeyError(query_id) from error
        row = np.unpackbits(
            self.packed_masks[query_index],
            count=self.total_route_count,
            bitorder=self.bitorder,
        )
        return {
            block.layer_id: row[block.offset : block.stop]
            .reshape(block.shape, order="C")
            .astype(bool, copy=False)
            for block in self.route_blocks
        }

    def packed_block(self, layer_id: str) -> np.ndarray:
        """Return one block byte-aligned without unpacking the global axis."""

        self.validate()
        block = next((item for item in self.route_blocks if item.layer_id == str(layer_id)), None)
        if block is None:
            raise KeyError(layer_id)
        start_byte = block.offset // 8
        bit_shift = block.offset % 8
        output_bytes = (block.route_count + 7) // 8
        if bit_shift == 0 and block.route_count % 8 == 0:
            return self.packed_masks[
                :,
                start_byte : start_byte + output_bytes,
            ]
        required = output_bytes + (1 if bit_shift else 0)
        source = np.zeros((len(self.query_ids), required), dtype=np.uint8)
        available = self.packed_masks[:, start_byte : start_byte + required]
        source[:, : available.shape[1]] = available
        if bit_shift:
            result = np.bitwise_or(
                np.right_shift(source[:, :output_bytes], bit_shift),
                np.left_shift(source[:, 1 : output_bytes + 1], 8 - bit_shift),
            ).astype(np.uint8, copy=False)
        else:
            result = source[:, :output_bytes].copy()
        trailing = block.route_count % 8
        if trailing:
            result[:, -1] &= np.uint8((1 << trailing) - 1)
        return result

    def retained_route_counts(self) -> tuple[int, ...]:
        self.validate()
        return tuple(
            int(np.bitwise_count(row).sum(dtype=np.int64))
            for row in self.packed_masks
        )

    def write(self, directory: str | Path) -> tuple[Path, Path, Path]:
        self.validate()
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        payload_path = root / "circuit_masks.block_bitpack.npy"
        metadata_path = root / "circuit_masks.block_metadata.json"
        coordinate_path = root / "route_blocks.parquet"
        np.save(payload_path, self.packed_masks, allow_pickle=False)
        write_parquet_rows(
            coordinate_path,
            [
                {
                    "block_index": index,
                    "block_id": block.block_id,
                    "architecture_id": block.architecture_id,
                    "topology_hash": block.topology_hash,
                    "layer_id": block.layer_id,
                    "route_kind": block.route_kind,
                    "shape_json": json.dumps(list(block.shape), separators=(",", ":")),
                    "offset": block.offset,
                    "route_count": block.route_count,
                    "flattening": block.flattening,
                }
                for index, block in enumerate(self.route_blocks)
            ],
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_name": "mechanistic_block_bitpacked_circuit_v1",
                    "query_ids": list(self.query_ids),
                    "bitorder": self.bitorder,
                    "route_axis_hash": self.route_axis_hash,
                    "route_block_count": len(self.route_blocks),
                    "total_route_count": self.total_route_count,
                    "packed_shape": list(self.packed_masks.shape),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return payload_path, metadata_path, coordinate_path

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        mmap_mode: str | None = None,
    ) -> BlockBitPackedCircuitSet:
        root = Path(directory)
        metadata = json.loads((root / "circuit_masks.block_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_name") != "mechanistic_block_bitpacked_circuit_v1":
            raise ValueError("unsupported block-bitpacked circuit schema")
        rows = sorted(read_parquet_rows(root / "route_blocks.parquet"), key=lambda row: int(row["block_index"]))
        blocks = tuple(
            RouteBlockSpec(
                architecture_id=str(row["architecture_id"]),
                topology_hash=str(row["topology_hash"]),
                layer_id=str(row["layer_id"]),
                route_kind=str(row["route_kind"]),
                shape=tuple(int(value) for value in json.loads(str(row["shape_json"]))),
                offset=int(row["offset"]),
                flattening=str(row["flattening"]),
            )
            for row in rows
        )
        result = cls(
            query_ids=tuple(str(value) for value in metadata["query_ids"]),
            route_blocks=blocks,
            packed_masks=np.load(
                root / "circuit_masks.block_bitpack.npy",
                allow_pickle=False,
                mmap_mode=mmap_mode,
            ),
            bitorder=str(metadata.get("bitorder", "")),
        )
        result.validate()
        if result.route_axis_hash != str(metadata.get("route_axis_hash", "")):
            raise ValueError("block route axis hash does not match metadata")
        return result


@dataclass(frozen=True)
class BitPackedCircuitSet:
    query_ids: tuple[str, ...]
    route_ids: tuple[str, ...]
    packed_masks: np.ndarray
    bitorder: str = "little"

    @classmethod
    def from_masks(
        cls,
        masks: np.ndarray,
        *,
        query_ids: tuple[str, ...],
        route_ids: tuple[str, ...],
    ) -> BitPackedCircuitSet:
        array = np.asarray(masks, dtype=bool)
        if array.shape != (len(query_ids), len(route_ids)):
            raise ValueError("circuit mask shape does not match query and route axes")
        if len(set(query_ids)) != len(query_ids) or len(set(route_ids)) != len(route_ids):
            raise ValueError("circuit axes must contain unique IDs")
        return cls(
            query_ids=query_ids,
            route_ids=route_ids,
            packed_masks=np.packbits(array, axis=1, bitorder="little"),
        )

    def unpack(self) -> np.ndarray:
        values = np.unpackbits(
            np.asarray(self.packed_masks, dtype=np.uint8),
            axis=1,
            count=len(self.route_ids),
            bitorder=self.bitorder,
        )
        return values.astype(bool, copy=False)

    def retained_route_counts(self) -> tuple[int, ...]:
        self.validate()
        return tuple(
            int(np.bitwise_count(row).sum(dtype=np.int64))
            for row in self.packed_masks
        )

    def validate(self) -> None:
        if self.bitorder != "little":
            raise ValueError("canonical circuit masks use little-endian bit packing")
        expected_columns = (len(self.route_ids) + 7) // 8
        if self.packed_masks.dtype != np.uint8 or self.packed_masks.shape != (
            len(self.query_ids),
            expected_columns,
        ):
            raise ValueError("packed circuit payload has an invalid shape or dtype")
        _validate_packed_padding_bits(
            self.packed_masks,
            route_count=len(self.route_ids),
        )

    def write(self, directory: str | Path) -> tuple[Path, Path]:
        self.validate()
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        payload_path = root / "circuit_masks.bitpack.npy"
        metadata_path = root / "circuit_masks.metadata.json"
        np.save(payload_path, self.packed_masks, allow_pickle=False)
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_name": "mechanistic_circuit_bitpack_v1",
                    "query_ids": list(self.query_ids),
                    "route_ids": list(self.route_ids),
                    "bitorder": self.bitorder,
                    "unpacked_shape": [len(self.query_ids), len(self.route_ids)],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return payload_path, metadata_path

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        mmap_mode: str | None = None,
    ) -> BitPackedCircuitSet:
        root = Path(directory)
        metadata = json.loads((root / "circuit_masks.metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_name") != "mechanistic_circuit_bitpack_v1":
            raise ValueError("unsupported bit-packed circuit schema")
        result = cls(
            query_ids=tuple(str(value) for value in metadata["query_ids"]),
            route_ids=tuple(str(value) for value in metadata["route_ids"]),
            packed_masks=np.load(
                root / "circuit_masks.bitpack.npy",
                allow_pickle=False,
                mmap_mode=mmap_mode,
            ),
            bitorder=str(metadata.get("bitorder", "")),
        )
        result.validate()
        return result


def ordered_route_axis_hash(route_ids: tuple[str, ...]) -> str:
    return stable_digest({"ordered_route_ids": list(route_ids)})


def materialize_circuit_artifact(
    circuits: BitPackedCircuitSet | BlockBitPackedCircuitSet,
    plan: CircuitPlan,
    store: ArtifactStore,
    *,
    staging_dir: str | Path,
    preservation_by_query: Mapping[str, float] | None = None,
    frontier_rows: Sequence[Mapping[str, object]] | None = None,
    source_kind: str = "native",
    valid_for_claims: bool | None = None,
    additional_semantics: Mapping[str, object] | None = None,
    additional_parent_hashes: Sequence[str] = (),
    notes: Mapping[str, object] | None = None,
) -> tuple[ArtifactManifest, Path]:
    """Seal one classic point realization under the shared circuit contract."""

    resolved_valid_for_claims = (
        plan.status == "frozen"
        if valid_for_claims is None
        else bool(valid_for_claims)
    )
    if resolved_valid_for_claims and plan.status != "frozen":
        raise ValueError(
            "claim-valid circuit materialization requires a frozen CircuitPlan"
        )
    plan.validate(claim_bearing=resolved_valid_for_claims).require_valid()
    circuits.validate()
    block_storage = isinstance(circuits, BlockBitPackedCircuitSet)
    route_axis_hash = (
        circuits.route_axis_hash
        if block_storage
        else ordered_route_axis_hash(circuits.route_ids)
    )
    if route_axis_hash != plan.route_universe_hash:
        raise ValueError("circuit route axis does not match CircuitPlan route universe")
    extra_semantics = dict(additional_semantics or {})
    reserved_semantics = {
        "objective",
        "engine_id",
        "realization_kind",
        "mask_storage",
        "route_axis_hash",
        "route_block_count",
        "total_route_count",
        "frontier_row_count",
        "partition_policy",
        "matched_controls",
        "optimization_view_hash",
        "evaluation_view_hash",
        "response_distortion",
        "route_cost",
        "mask_budget",
    }
    overlap = sorted(reserved_semantics & set(extra_semantics))
    if overlap:
        raise ValueError(
            "additional circuit semantics cannot replace canonical fields: "
            + ", ".join(overlap)
        )
    preservation = {str(key): float(value) for key, value in (preservation_by_query or {}).items()}
    unknown = sorted(set(preservation) - set(circuits.query_ids))
    if unknown:
        raise ValueError("preservation table contains unknown queries: " + ", ".join(unknown))
    total_route_count = (
        circuits.total_route_count if block_storage else len(circuits.route_ids)
    )
    retained_counts = circuits.retained_route_counts()
    extra_parents = tuple(str(value) for value in additional_parent_hashes)
    if len(set(extra_parents)) != len(extra_parents):
        raise ValueError("additional circuit parents must be unique")
    if plan.measurement_artifact_hash in extra_parents:
        raise ValueError("measurement parent may not be repeated as an additional parent")
    if any(len(value) != 64 for value in extra_parents):
        raise ValueError("additional circuit parents require SHA-256 artifact hashes")
    rows = []
    for index, query_id in enumerate(circuits.query_ids):
        retained = retained_counts[index]
        rows.append(
            {
                "query_id": query_id,
                "retained_route_count": retained,
                "route_universe_count": total_route_count,
                "route_density": retained / max(total_route_count, 1),
                "target_preservation": preservation.get(query_id, float("nan")),
                "partition_policy": plan.partition_policy,
                "engine_id": plan.engine_id,
            }
        )
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".circuit-", dir=staging) as temporary:
        temporary_path = Path(temporary)
        written = circuits.write(temporary_path)
        if block_storage:
            mask_path, metadata_path, coordinate_path = written
            circuit_paths = (mask_path, metadata_path, coordinate_path)
        else:
            mask_path, metadata_path = written
            circuit_paths = (mask_path, metadata_path)
        summary_path = write_parquet_rows(temporary_path / "circuit_summary.parquet", rows)
        frontier_path = (
            write_parquet_rows(
                temporary_path / "rate_distortion_frontier.parquet",
                list(frontier_rows),
            )
            if frontier_rows
            else None
        )
        paths = (*circuit_paths, summary_path, *((frontier_path,) if frontier_path else ()))
        payloads = tuple(
            PayloadRef.from_file(
                name=path.name,
                path=path,
                uri=f"payloads/{path.name}",
                storage_format=(
                    "bitpacked_numpy"
                    if path == mask_path
                    else "json"
                    if path == metadata_path
                    else "parquet"
                ),
                media_type=(
                    "application/json"
                    if path == metadata_path
                    else "application/vnd.apache.parquet"
                    if path.suffix == ".parquet"
                    else "application/octet-stream"
                ),
            )
            for path in paths
        )
        manifest = ArtifactManifest(
            artifact_type=ArtifactType.CIRCUIT,
            study_hash=plan.study_hash,
            plan_hash=plan.content_hash,
            parent_hashes=(plan.measurement_artifact_hash, *extra_parents),
            axes={
                "query": circuits.query_ids,
                "route": (
                    tuple(block.block_id for block in circuits.route_blocks)
                    if block_storage
                    else circuits.route_ids
                ),
                **(
                    {"layer": tuple(block.layer_id for block in circuits.route_blocks)}
                    if block_storage
                    else {}
                ),
            },
            coordinate_schema_hash=route_axis_hash,
            semantics={
                "objective": "constrained_rate_distortion",
                "engine_id": plan.engine_id,
                "realization_kind": "classic_point_mask",
                "mask_storage": (
                    "implicit_route_blocks_little_endian_bitpacked"
                    if block_storage
                    else "explicit_route_axis_little_endian_bitpacked"
                ),
                "route_axis_hash": route_axis_hash,
                "route_block_count": len(circuits.route_blocks) if block_storage else 0,
                "total_route_count": total_route_count,
                "frontier_row_count": len(frontier_rows or ()),
                "partition_policy": plan.partition_policy,
                "matched_controls": list(plan.matched_controls),
                "optimization_view_hash": stable_digest(plan.optimization_view.to_dict()),
                "evaluation_view_hash": stable_digest(plan.evaluation_view.to_dict()),
                "response_distortion": dict(plan.response_distortion),
                "route_cost": dict(plan.route_cost),
                "mask_budget": dict(plan.mask_budget),
                **extra_semantics,
            },
            payloads=payloads,
            unit_ids=circuits.query_ids,
            source_kind=str(source_kind),
            valid_for_claims=resolved_valid_for_claims,
            notes=dict(notes or {}),
        )
        store.seal_contract(
            contract_kind="circuit_plan",
            content_hash=plan.content_hash,
            payload=plan.to_dict(),
        )
        bundle = store.write_bundle(
            manifest,
            payload_files={path.name: path for path in paths},
        )
    receipt = store.certify(manifest, validator_id="mechanistic_circuit.v1")
    if not receipt.passed:
        raise ValueError("circuit certification failed: " + "; ".join(receipt.issues))
    return manifest, bundle


def assemble_native_sampled_path_query_packets(
    *,
    plan: CircuitPlan,
    store: ArtifactStore,
    packet_artifact_hashes: Sequence[str],
    query_ids: Sequence[str],
    staging_dir: str | Path,
) -> tuple[ArtifactManifest, Path, object]:
    """Assemble independent claim-ready query packets into one circuit panel.

    Sampled-path queries are statistically independent but each query retains
    a sequential layer path. Running queries in separate work packets is
    therefore exact. This assembler verifies the complete scientific lineage,
    merges untouched-final-validation receipts, and emits the same ordinary
    multi-query ``CircuitArtifact`` consumed by execution and inference.
    """

    from mechanistic_systems.circuits.final_validation import NativeSampledPathFinalValidationReceipt

    expected_queries = tuple(str(value) for value in query_ids)
    packet_hashes = tuple(str(value) for value in packet_artifact_hashes)
    if not expected_queries or len(set(expected_queries)) != len(expected_queries):
        raise ValueError("query-packet assembly requires unique ordered query IDs")
    if len(packet_hashes) != len(expected_queries) or len(set(packet_hashes)) != len(
        packet_hashes
    ):
        raise ValueError("query-packet artifacts must map one-to-one to query IDs")
    if any(len(value) != 64 for value in packet_hashes):
        raise ValueError("query-packet artifacts require SHA-256 hashes")
    plan.validate(claim_bearing=True).require_valid()
    if plan.status != "frozen":
        raise ValueError("claim-ready query-packet assembly requires a frozen CircuitPlan")

    common_semantic_fields = (
        "search_algorithm",
        "algorithm_version",
        "sampled_path_profile_hash",
        "promotion_receipt_hash",
        "promotion_acceptance_mode",
        "exact_circuit_identity",
        "final_validation_policy",
        "group_reference_anchor_policy",
    )
    partition_provenance: tuple[str, str] | None = None
    first_manifest: ArtifactManifest | None = None
    first_circuits: BlockBitPackedCircuitSet | None = None
    packet_masks: list[np.ndarray] = []
    frontier_rows: list[Mapping[str, object]] = []
    validation_receipts: list[NativeSampledPathFinalValidationReceipt] = []
    validation_hashes: dict[str, str] = {}
    group_reference_hashes: dict[str, str] = {}
    preservation: dict[str, float] = {}
    sampled_path_group_count = 0
    group_reference_anchor_count = 0

    for query_id, artifact_hash in zip(expected_queries, packet_hashes, strict=True):
        manifest = store.load(artifact_hash, ArtifactType.CIRCUIT)
        certification_receipts = store.certification_receipts(artifact_hash)
        lifecycle = infer_lifecycle_state(
            manifest,
            certification_receipts,
            store.lifecycle_events(artifact_hash),
        )
        if lifecycle != LifecycleState.CLAIM_READY:
            raise ValueError(
                f"query packet {query_id!r} is not claim-ready: {lifecycle.value}"
            )
        if not manifest.valid_for_claims:
            raise ValueError(f"query packet {query_id!r} is not valid for claims")
        if manifest.plan_hash != plan.content_hash or manifest.study_hash != plan.study_hash:
            raise ValueError("query packet plan/study lineage differs from assembly plan")
        if manifest.parent_hashes[:1] != (plan.measurement_artifact_hash,):
            raise ValueError("query packet measurement parent differs from assembly plan")
        if manifest.source_kind not in {"native", "native_sampled_path_frontier_v1"}:
            raise ValueError("query packet is not a native sampled-path artifact")
        if manifest.semantics.get("algorithm_version") != "sampled_path_frontier_v1":
            raise ValueError("query packet algorithm differs from sampled_path_frontier_v1")
        if tuple(manifest.axes.get("query", ())) != (query_id,):
            raise ValueError("query packet artifact axis differs from requested order")
        if manifest.coordinate_schema_hash != plan.route_universe_hash:
            raise ValueError("query packet route universe differs from assembly plan")

        provider_id = str(
            manifest.semantics.get("partition_label_provider_id", "")
        )
        partition_hash = str(manifest.semantics.get("partition_label_hash", ""))
        if not provider_id or len(partition_hash) != 64:
            receipt_pairs = {
                (
                    str(receipt.details.get("partition_label_provider_id", "")),
                    str(receipt.details.get("partition_label_hash", "")),
                )
                for receipt in store.runtime_receipts(artifact_hash)
                if receipt.stage == "sampled_path_frontier_v1_native_panel"
                and receipt.details.get("partition_label_provider_id")
                and len(str(receipt.details.get("partition_label_hash", ""))) == 64
            }
            if len(receipt_pairs) != 1:
                raise ValueError(
                    "query packet lacks unique structural-partition provenance"
                )
            provider_id, partition_hash = next(iter(receipt_pairs))
        observed_partition_provenance = (provider_id, partition_hash)
        if partition_provenance is None:
            partition_provenance = observed_partition_provenance
        elif observed_partition_provenance != partition_provenance:
            raise ValueError("query packet structural-partition provenance differs")

        bundle = store.bundle_dir(manifest)
        circuits = BlockBitPackedCircuitSet.load(bundle / "payloads")
        if circuits.query_ids != (query_id,):
            raise ValueError("query packet mask axis differs from its manifest")
        if first_manifest is None:
            first_manifest = manifest
            first_circuits = circuits
        else:
            assert first_circuits is not None
            if circuits.route_blocks != first_circuits.route_blocks:
                raise ValueError("query packets do not share one ordered route-block axis")
            if tuple(manifest.axes.get("route", ())) != tuple(
                first_manifest.axes.get("route", ())
            ) or tuple(manifest.axes.get("layer", ())) != tuple(
                first_manifest.axes.get("layer", ())
            ):
                raise ValueError("query packet route/layer manifest axes differ")
            for field in common_semantic_fields:
                if manifest.semantics.get(field) != first_manifest.semantics.get(field):
                    raise ValueError(f"query packet semantic differs: {field}")
        packet_masks.append(circuits.packed_masks.copy())

        validation_hash = str(
            manifest.semantics.get("final_validation_receipt_hash", "")
        )
        if len(validation_hash) != 64:
            raise ValueError("query packet lacks an untouched-final-validation receipt")
        validation = NativeSampledPathFinalValidationReceipt.from_mapping(
            store.load_contract("sampled_path_final_validation", validation_hash)
        )
        validation.require_passed(
            plan_hash=plan.content_hash,
            profile_hash=str(manifest.semantics.get("sampled_path_profile_hash", "")),
            promotion_receipt_hash=str(
                manifest.semantics.get("promotion_receipt_hash", "")
            ),
            query_ids=(query_id,),
        )
        validation_receipts.append(validation)
        validation_hashes[query_id] = validation_hash
        preservation.update(validation.preservation_by_query)

        group_hash = str(manifest.semantics.get("group_reference_anchor_hash", ""))
        if len(group_hash) != 64:
            raise ValueError("query packet lacks committed group-reference evidence")
        group_reference_hashes[query_id] = group_hash
        sampled_path_group_count += int(
            manifest.notes.get("sampled_path_group_count", 0)
        )
        group_reference_anchor_count += int(
            manifest.notes.get("group_reference_anchor_count", 0)
        )

        frontier_path = bundle / "payloads" / "rate_distortion_frontier.parquet"
        if frontier_path.is_file():
            packet_frontier = read_parquet_rows(frontier_path)
            if any(str(row.get("query_id", "")) != query_id for row in packet_frontier):
                raise ValueError("query packet frontier contains another query")
            frontier_rows.extend(packet_frontier)

    assert first_manifest is not None and first_circuits is not None
    assert partition_provenance is not None
    first_validation = validation_receipts[0]
    parity_modes = {receipt.parity_configured for receipt in validation_receipts}
    if len(parity_modes) != 1:
        raise ValueError(
            "query-packet assembly refuses mixed legacy and terminal-parity "
            "validation receipts"
        )
    if parity_modes == {True}:
        raise ValueError(
            "legacy query-packet assembly cannot merge independently alpha-spent "
            "terminal audits; use one campaign-global glocal-v3 deployment "
            "certificate"
        )
    if any(
        (
            receipt.optimization_axis_hash,
            receipt.evaluation_axis_hash,
            receipt.calibration_policy,
            receipt.evaluation_policy,
        )
        != (
            first_validation.optimization_axis_hash,
            first_validation.evaluation_axis_hash,
            first_validation.calibration_policy,
            first_validation.evaluation_policy,
        )
        for receipt in validation_receipts[1:]
    ):
        raise ValueError("query packets do not share final-validation data/policy")
    merged_validation = NativeSampledPathFinalValidationReceipt(
        plan_hash=plan.content_hash,
        profile_hash=first_validation.profile_hash,
        promotion_receipt_hash=first_validation.promotion_receipt_hash,
        optimization_axis_hash=first_validation.optimization_axis_hash,
        evaluation_axis_hash=first_validation.evaluation_axis_hash,
        query_ids=expected_queries,
        rows=tuple(receipt.rows[0] for receipt in validation_receipts),
        issues=tuple(),
        calibration_policy=first_validation.calibration_policy,
        evaluation_policy=first_validation.evaluation_policy,
    )
    merged_validation.require_passed(query_ids=expected_queries)
    store.seal_contract(
        contract_kind="sampled_path_final_validation",
        content_hash=merged_validation.content_hash,
        payload=merged_validation.to_dict(),
    )

    circuits = BlockBitPackedCircuitSet(
        query_ids=expected_queries,
        route_blocks=first_circuits.route_blocks,
        packed_masks=np.concatenate(packet_masks, axis=0),
    )
    circuits.validate()
    assembled_group_reference_hash = stable_digest(
        {
            "schema_name": "mechanistic_query_packet_group_reference_assembly_v1",
            "policy": first_manifest.semantics["group_reference_anchor_policy"],
            "hashes_by_query": group_reference_hashes,
        }
    )
    manifest, bundle = materialize_circuit_artifact(
        circuits,
        plan,
        store,
        staging_dir=staging_dir,
        preservation_by_query=preservation,
        frontier_rows=frontier_rows,
        source_kind="native",
        valid_for_claims=True,
        additional_parent_hashes=packet_hashes,
        additional_semantics={
            **{
                field: first_manifest.semantics[field]
                for field in common_semantic_fields
                if field not in {"final_validation_policy"}
            },
            "final_validation_receipt_hash": merged_validation.content_hash,
            "final_validation_policy": merged_validation.evaluation_policy,
            "component_final_validation_receipt_hashes": validation_hashes,
            "group_reference_anchor_hash": assembled_group_reference_hash,
            "component_group_reference_anchor_hashes": group_reference_hashes,
            "query_packet_artifact_hashes": dict(zip(expected_queries, packet_hashes, strict=True)),
            "query_packet_assembly_schema": "mechanistic_circuit_query_packet_assembly_v1",
            "partition_label_provider_id": partition_provenance[0],
            "partition_label_hash": partition_provenance[1],
        },
        notes={
            "sampled_path_group_count": sampled_path_group_count,
            "algorithm_configuration_parity": True,
            "untouched_final_validation_passed": True,
            "group_reference_anchor_count": group_reference_anchor_count,
            "query_packet_count": len(packet_hashes),
        },
    )
    store.mark_analysis_ready(
        manifest, actor="mechanistic_systems.query_packet_assembly.v1"
    )
    store.mark_claim_ready(
        manifest, actor="mechanistic_systems.query_packet_assembly.v1"
    )
    return manifest, bundle, merged_validation


__all__ = [
    "BLOCK_ROUTE_AXIS_SCHEMA",
    "BitPackedCircuitSet",
    "BlockBitPackedCircuitSet",
    "RouteBlockSpec",
    "assemble_native_sampled_path_query_packets",
    "materialize_circuit_artifact",
    "ordered_block_route_axis_hash",
    "ordered_route_axis_hash",
]
