"""Chunked score-moment storage over implicit route blocks.

The route axis of a modern CNN contains millions of channel-pair coordinates.
One Parquet row per query-route pair is therefore not a production format. This
module stores numeric heads as layer-sharded Zarr arrays while retaining the
exact ordered coordinate rule in a compact ``RouteBlockSpec`` axis.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mechanistic_systems.core.storage import (
    RouteBlockSpec,
    ordered_block_route_axis_hash,
    rebase_route_blocks,
)

try:  # Optional for import smoke; required for production materialization.
    import zarr
    from numcodecs import Blosc
except Exception:  # pragma: no cover - environment dependent.
    zarr = None  # type: ignore[assignment]
    Blosc = None  # type: ignore[assignment]


BLOCK_MEASUREMENT_SCHEMA = "mechanistic_block_measurement_zarr_v1"
CANONICAL_ROUTE_MEASUREMENT_HEADS = ("route_scores", "route_abs_scores")


def _require_zarr() -> None:
    if zarr is None or Blosc is None:
        raise RuntimeError("block measurement storage requires zarr>=2.18,<3 and numcodecs")


def _string_tuple(values: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be an ordered sequence")
    result = tuple(str(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


def write_block_measurement_zarr(
    path: str | Path,
    *,
    query_ids: Sequence[str],
    route_blocks: Sequence[RouteBlockSpec],
    head_ids: Sequence[str],
    load_query: Callable[[str], Mapping[str, Mapping[str, np.ndarray]]],
    head_semantics: Mapping[str, str] | None = None,
    route_chunk: int = 262_144,
) -> Path:
    """Write exact aggregate score heads without expanding route IDs.

    ``load_query`` is called once per query and may release its source shard as
    soon as the returned arrays have been copied. Durable arrays are flattened
    only within each declared route block, preserving the original shape in
    block metadata.
    """

    _require_zarr()
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    queries = _string_tuple(tuple(query_ids), name="query_ids")
    heads = _string_tuple(tuple(head_ids), name="head_ids")
    blocks = tuple(route_blocks)
    route_axis_hash = ordered_block_route_axis_hash(blocks)
    if int(route_chunk) <= 0:
        raise ValueError("route_chunk must be positive")

    root = zarr.open_group(str(destination), mode="w")
    root.attrs.update(
        {
            "schema_name": BLOCK_MEASUREMENT_SCHEMA,
            "query_ids": list(queries),
            "head_ids": list(heads),
            "head_semantics": dict(head_semantics or {}),
            "route_axis_hash": route_axis_hash,
            "route_blocks": [block.to_dict() for block in blocks],
            "storage_dtype": "float32",
            "source_accumulation_policy": "producer_declared_exact_aggregate",
        }
    )
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    arrays: dict[tuple[str, str], Any] = {}
    block_group = root.require_group("blocks")
    for block_index, block in enumerate(blocks):
        group = block_group.require_group(f"block_{block_index:04d}")
        group.attrs.update(block.to_dict())
        for head_id in heads:
            arrays[(block.layer_id, head_id)] = group.create_dataset(
                head_id,
                shape=(len(queries), block.route_count),
                chunks=(1, min(int(route_chunk), block.route_count)),
                dtype=np.float32,
                compressor=compressor,
                overwrite=False,
            )

    expected_layers = {block.layer_id for block in blocks}
    for query_index, query_id in enumerate(queries):
        payload = load_query(query_id)
        if set(payload) != set(heads):
            raise ValueError(f"query {query_id!r} does not provide the declared measurement heads")
        for head_id in heads:
            layer_map = payload[head_id]
            if set(layer_map) != expected_layers:
                raise ValueError(
                    f"query {query_id!r} head {head_id!r} does not match the route-block axis"
                )
            for block in blocks:
                values = np.asarray(layer_map[block.layer_id])
                if values.dtype != np.float32:
                    raise ValueError(
                        f"query {query_id!r} head {head_id!r} layer {block.layer_id!r} "
                        f"must be exact float32, got {values.dtype}"
                    )
                if values.shape != block.shape or not np.isfinite(values).all():
                    raise ValueError(
                        f"query {query_id!r} head {head_id!r} layer {block.layer_id!r} "
                        "has incompatible shape or non-finite values"
                    )
                arrays[(block.layer_id, head_id)][query_index, :] = values.reshape(-1, order="C")
        del payload
    return destination


@dataclass(frozen=True)
class BlockMeasurementReader:
    path: Path

    def __post_init__(self) -> None:
        _require_zarr()
        root = zarr.open_group(str(self.path), mode="r")
        if str(root.attrs.get("schema_name", "")) != BLOCK_MEASUREMENT_SCHEMA:
            raise ValueError(f"unsupported block measurement schema at {self.path}")
        query_ids = _string_tuple(root.attrs["query_ids"], name="query_ids")
        head_ids = _string_tuple(root.attrs["head_ids"], name="head_ids")
        raw_blocks = root.attrs.get("route_blocks", ())
        blocks = tuple(RouteBlockSpec.from_mapping(dict(item)) for item in raw_blocks)
        route_hash = ordered_block_route_axis_hash(blocks)
        if route_hash != str(root.attrs.get("route_axis_hash", "")):
            raise ValueError("block measurement route-axis hash mismatch")
        for block_index, block in enumerate(blocks):
            group = root[f"blocks/block_{block_index:04d}"]
            for head_id in head_ids:
                if tuple(group[head_id].shape) != (len(query_ids), block.route_count):
                    raise ValueError(f"measurement block {block.layer_id!r} has an invalid shape")
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "head_ids", head_ids)
        object.__setattr__(self, "route_blocks", blocks)
        object.__setattr__(self, "route_axis_hash", route_hash)

    def read_block(
        self,
        layer_id: str,
        head_id: str,
        *,
        reshape: bool = True,
    ) -> np.ndarray:
        block_index = next(
            (index for index, block in enumerate(self.route_blocks) if block.layer_id == str(layer_id)),
            None,
        )
        if block_index is None:
            raise KeyError(layer_id)
        if str(head_id) not in self.head_ids:
            raise KeyError(head_id)
        block = self.route_blocks[block_index]
        values = np.asarray(
            self._root[f"blocks/block_{block_index:04d}/{head_id}"],
            dtype=np.float32,
        )
        return values.reshape((len(self.query_ids), *block.shape), order="C") if reshape else values

    def source_image_aggregate_block(
        self,
        layer_id: str,
        *,
        score_head: str,
    ) -> np.ndarray:
        """Expose aggregate route heads through the pruning reader protocol."""

        return self.read_block(layer_id, score_head, reshape=False)

    def projected_route_axis_hash(
        self,
        *,
        excluded_layer_ids: Sequence[str],
    ) -> str:
        """Hash a declared projection such as fixed-readout removal."""

        excluded = {str(value) for value in excluded_layer_ids}
        available = {block.layer_id for block in self.route_blocks}
        if not excluded or not excluded.issubset(available):
            raise ValueError("route-axis projection must exclude declared measured layers")
        selected = tuple(
            block for block in self.route_blocks if block.layer_id not in excluded
        )
        if not selected:
            raise ValueError("route-axis projection removed every measured layer")
        return ordered_block_route_axis_hash(rebase_route_blocks(selected))


__all__ = [
    "BLOCK_MEASUREMENT_SCHEMA",
    "CANONICAL_ROUTE_MEASUREMENT_HEADS",
    "BlockMeasurementReader",
    "write_block_measurement_zarr",
]
