from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .storage import FeatureDataset


@dataclass(frozen=True)
class EmbeddingConfig:
    feature_names: Sequence[str]
    target_dim: int = 16
    device: str = "cpu"
    batch_size: int = 262_144
    importance_feature: str | None = None
    reuse_stats: bool = True
    pca_method: str = "auto"
    pca_oversample: int = 4
    pca_power_iter: int = 2


@dataclass(frozen=True)
class EmbeddingSummary:
    count: int
    mean: torch.Tensor
    components: torch.Tensor
    explained_variance: torch.Tensor
    path: Path
    row_splits: tuple[int, ...]


def _stack_features(dataset: FeatureDataset, feature_names: Sequence[str]) -> tuple[torch.Tensor, list[int]]:
    chunks = []
    row_splits: list[int] = []
    for batch in dataset.iter_batches():
        columns = []
        for name in feature_names:
            if name not in batch.values:
                raise KeyError(f"Feature dataset is missing requested feature {name!r}")
            columns.append(batch.values[name].to(dtype=torch.float32).reshape(-1, 1))
        matrix = torch.cat(columns, dim=1)
        chunks.append(matrix)
        row_splits.append(int(matrix.shape[0]))
    if not chunks:
        raise ValueError(f"Feature dataset at {dataset.root} has no chunks")
    return torch.cat(chunks, dim=0), row_splits


def build_embedding(dataset: FeatureDataset, config: EmbeddingConfig, output_dir: str | Path) -> EmbeddingSummary:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "stats.pt"
    embedding_path = out / "embedding.pt"
    if config.reuse_stats and stats_path.exists() and embedding_path.exists():
        stats = torch.load(stats_path, map_location="cpu")
        return EmbeddingSummary(
            count=int(stats["count"]),
            mean=stats["mean"],
            components=stats["components"],
            explained_variance=stats["explained_variance"],
            path=out,
            row_splits=tuple(int(x) for x in stats.get("row_splits", ())),
        )

    matrix, row_splits = _stack_features(dataset, tuple(config.feature_names))
    count = int(matrix.shape[0])
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    dim = max(1, min(int(config.target_dim), centered.shape[1], centered.shape[0]))
    if centered.shape[0] == 1:
        components = torch.eye(centered.shape[1], dtype=centered.dtype)[:dim]
        embedding = centered[:, :dim]
        explained = torch.zeros(dim, dtype=centered.dtype)
    else:
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        components = vh[:dim].contiguous()
        embedding = centered @ components.T
        explained = (singular_values[:dim] ** 2) / max(count - 1, 1)
    torch.save(
        {
            "count": count,
            "mean": mean,
            "components": components,
            "explained_variance": explained,
            "row_splits": tuple(row_splits),
        },
        stats_path,
    )
    torch.save({"embedding": embedding, "row_splits": tuple(row_splits)}, embedding_path)
    return EmbeddingSummary(
        count=count,
        mean=mean,
        components=components,
        explained_variance=explained,
        path=out,
        row_splits=tuple(row_splits),
    )


__all__ = ["EmbeddingConfig", "EmbeddingSummary", "build_embedding"]
