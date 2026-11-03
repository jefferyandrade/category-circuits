from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from .features import FeatureBatch, FeatureSchema


@dataclass(frozen=True)
class FeatureDatasetManifestEntry:
    chunk: str
    rows: int
    schema_hash: str


class FeatureDatasetWriter:
    def __init__(
        self,
        root: str | Path,
        *,
        schema: FeatureSchema,
        layer: str,
        unit: str | None,
        format: str = "pt",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.layer = layer
        self.unit = unit
        self.format = format
        self._next_index = len(list(self.root.glob("chunk_*.*")))
        (self.root / "schema.json").write_text(
            json.dumps(schema.to_dict(), indent=2, sort_keys=True) + "\n"
        )

    def write(self, batch: FeatureBatch) -> Path:
        if batch.layer != self.layer:
            raise ValueError(f"Feature batch layer {batch.layer!r} does not match {self.layer!r}")
        suffix = ".parquet" if self.format == "parquet" else ".pt"
        path = self.root / f"chunk_{self._next_index:06d}{suffix}"
        self._next_index += 1
        payload = {
            "schema_hash": self.schema.schema_hash,
            "layer": batch.layer,
            "unit": batch.unit,
            "indices": batch.weight_indices.detach().to("cpu"),
            "values": {k: v.detach().to("cpu") for k, v in batch.values.items()},
            "counts": None if batch.counts is None else batch.counts.detach().to("cpu"),
        }
        if self.format == "parquet":
            self._write_parquet(path, payload)
        else:
            torch.save(payload, path)
        with (self.root / "manifest.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "chunk": path.name,
                        "rows": int(batch.weight_indices.numel()),
                        "schema_hash": self.schema.schema_hash,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        return path

    def _write_parquet(self, path: Path, payload: dict[str, object]) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        values = payload["values"]
        assert isinstance(values, dict)
        data = {
            "weight_index": payload["indices"].tolist(),  # type: ignore[union-attr]
            **{name: tensor.tolist() for name, tensor in values.items()},
        }
        if payload.get("counts") is not None:
            data["counts"] = payload["counts"].tolist()  # type: ignore[union-attr]
        table = pa.table(data)
        metadata = {
            b"schema_hash": str(payload["schema_hash"]).encode("utf-8"),
            b"layer": str(payload["layer"]).encode("utf-8"),
            b"unit": str(payload["unit"]).encode("utf-8"),
        }
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, path)


class FeatureDataset:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        schema_path = self.root / "schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"Feature dataset schema not found at {schema_path}")
        self.schema = FeatureSchema.from_dict(json.loads(schema_path.read_text()))

    def chunk_paths(self) -> list[Path]:
        return sorted([*self.root.glob("chunk_*.pt"), *self.root.glob("chunk_*.parquet")])

    def iter_batches(self) -> Iterator[FeatureBatch]:
        for path in self.chunk_paths():
            if path.suffix == ".parquet":
                yield self._read_parquet(path)
            else:
                payload = torch.load(path, map_location="cpu")
                yield FeatureBatch(
                    layer=str(payload["layer"]),
                    unit=payload.get("unit"),
                    weight_indices=payload["indices"],
                    values=dict(payload["values"]),
                    counts=payload.get("counts"),
                )

    def _read_parquet(self, path: Path) -> FeatureBatch:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        meta = table.schema.metadata or {}
        data = table.to_pydict()
        values = {
            name: torch.as_tensor(data[name], dtype=torch.float32)
            for name in self.schema.feature_names
        }
        counts = torch.as_tensor(data["counts"], dtype=torch.float32) if "counts" in data else None
        return FeatureBatch(
            layer=meta.get(b"layer", b"").decode("utf-8"),
            unit=None if meta.get(b"unit", b"None").decode("utf-8") == "None" else meta[b"unit"].decode("utf-8"),
            weight_indices=torch.as_tensor(data["weight_index"], dtype=torch.long),
            values=values,
            counts=counts,
        )


__all__ = ["FeatureDataset", "FeatureDatasetManifestEntry", "FeatureDatasetWriter"]
