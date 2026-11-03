"""Runtime-only readout projection capability.

This backend object reduces materialized readout columns without changing the
scientific query. It carries no pruning, intervention, or claim authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contracts.hashing import stable_digest

try:  # Optional for import smoke; required when applying a projection.
    import torch
except Exception:  # pragma: no cover - environment dependent.
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RuntimeReadoutProjectionSpec:
    """Runtime-only readout column projection."""

    projection_id: str
    column_ids: tuple[int, ...]
    retain_host_values: bool = False

    def __post_init__(self) -> None:
        if (
            not self.projection_id
            or not self.column_ids
            or len(set(self.column_ids)) != len(self.column_ids)
            or any(int(value) < 0 for value in self.column_ids)
        ):
            raise ValueError(
                "runtime readout projection requires unique nonnegative columns"
            )

    @classmethod
    def coordinate(
        cls,
        column_id: int,
        *,
        retain_host_values: bool = False,
    ) -> "RuntimeReadoutProjectionSpec":
        return cls(
            projection_id=f"runtime-readout-coordinate:{int(column_id)}",
            column_ids=(int(column_id),),
            retain_host_values=bool(retain_host_values),
        )

    @property
    def content_hash(self) -> str:
        return stable_digest(
            {
                "schema_name": "mechanistic_runtime_readout_projection_v1",
                "projection_id": self.projection_id,
                "column_ids": self.column_ids,
                "retain_host_values": bool(self.retain_host_values),
                "scientific_identity": "excluded",
            }
        )

    def apply(self, values: Any) -> Any:
        if torch is None:
            raise RuntimeError("runtime readout projection requires PyTorch")
        if values.ndim != 2:
            raise ValueError("runtime readout projection requires a matrix")
        if max(self.column_ids) >= int(values.shape[1]):
            raise ValueError("runtime readout projection exceeds readout width")
        return values.index_select(1, _column_index(self.column_ids, values.device))


# The column index tensor is built once per (columns, device)
# instead of on every call (a host tensor plus a blocking host-to-device copy
# per panel batch).  It is only ever read by ``index_select``, so the selected
# values are the same; the copy completes before the tensor is cached.
_COLUMN_INDEX_CACHE: dict[tuple[tuple[int, ...], Any], Any] = {}
_COLUMN_INDEX_CACHE_LIMIT = 256


def _column_index(column_ids: tuple[int, ...], device: Any) -> Any:
    key = (tuple(column_ids), device)
    index = _COLUMN_INDEX_CACHE.get(key)
    if index is None:
        index = torch.as_tensor(key[0], dtype=torch.long, device=device)
        if len(_COLUMN_INDEX_CACHE) >= _COLUMN_INDEX_CACHE_LIMIT:
            _COLUMN_INDEX_CACHE.clear()
        _COLUMN_INDEX_CACHE[key] = index
    return index


__all__ = ["RuntimeReadoutProjectionSpec"]
