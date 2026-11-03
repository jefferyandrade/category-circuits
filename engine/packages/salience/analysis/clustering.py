from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(frozen=True)
class ClusterConfig:
    num_clusters: int = 16
    max_iter: int = 50
    sample_size: int = 1_000_000
    sample_fraction: float | None = None
    min_sample_size: int = 100_000
    seed: int = 0
    device: str = "cpu"
    assignment_batch_size: int = 262_144
    prefetch_chunks: int = 2
    backend: str = "auto"


@dataclass(frozen=True)
class ClusterResult:
    centroids: torch.Tensor
    counts: torch.Tensor
    inertia: float
    assignment_paths: tuple[Path, ...]


def _initial_centroids(x: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    rng = random.Random(seed)
    indices = list(range(x.shape[0]))
    rng.shuffle(indices)
    if len(indices) < k:
        repeats = (k + len(indices) - 1) // max(len(indices), 1)
        indices = (indices * repeats)[:k]
    else:
        indices = indices[:k]
    return x[torch.as_tensor(indices, dtype=torch.long)].clone()


def cluster_layer(embedding_dir: str | Path, output_dir: str | Path, config: ClusterConfig) -> ClusterResult:
    if config.backend == "faiss" and not _faiss_available():
        raise RuntimeError("FAISS backend requested but faiss is unavailable")
    payload = torch.load(Path(embedding_dir) / "embedding.pt", map_location="cpu")
    x = payload["embedding"].to(dtype=torch.float32)
    row_splits = tuple(int(v) for v in payload.get("row_splits", (int(x.shape[0]),)))
    if x.ndim == 1:
        x = x[:, None]
    k = max(1, min(int(config.num_clusters), int(x.shape[0])))
    centroids = _initial_centroids(x, k, int(config.seed))
    assignments = torch.zeros(x.shape[0], dtype=torch.long)
    for _ in range(max(1, int(config.max_iter))):
        distances = torch.cdist(x, centroids)
        next_assignments = distances.argmin(dim=1)
        if torch.equal(next_assignments, assignments):
            break
        assignments = next_assignments
        for cid in range(k):
            mask = assignments == cid
            if mask.any():
                centroids[cid] = x[mask].mean(dim=0)
    final_distances = torch.cdist(x, centroids)
    assignments = final_distances.argmin(dim=1)
    inertia = float(final_distances[torch.arange(x.shape[0]), assignments].pow(2).sum().item())
    counts = torch.bincount(assignments, minlength=k)

    out = Path(output_dir)
    assign_dir = out / "assignments"
    assign_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    start = 0
    for idx, rows in enumerate(row_splits):
        path = assign_dir / f"assign_{idx:06d}.pt"
        torch.save(assignments[start : start + rows].cpu(), path)
        paths.append(path)
        start += rows
    torch.save(
        {
            "centroids": centroids.cpu(),
            "counts": counts.cpu(),
            "inertia": inertia,
            "assignment_paths": [str(path) for path in paths],
        },
        out / "clusters.pt",
    )
    return ClusterResult(
        centroids=centroids.cpu(),
        counts=counts.cpu(),
        inertia=inertia,
        assignment_paths=tuple(paths),
    )


__all__ = ["ClusterConfig", "ClusterResult", "_faiss_available", "cluster_layer"]
