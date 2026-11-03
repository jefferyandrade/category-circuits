"""ImageNet folder population, sealed permutations, and batch loading for the tractable lane.

Populations: the target stratum is the category's complete
ImageNet train folder; counts come from the actual listing. The complement
stratum is every other train image. This module creates one seeded permutation
per stratum. Role allocation is a controller responsibility; these permutations
do not themselves prove D/K/O independence. Validation is a terminal-only role.

Preprocessing is the canonical torchvision AlexNet evaluation transform:
resize 256, center crop 224, ImageNet mean/std normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from .alexnet_factors import IMAGENET_MEAN, IMAGENET_STD

_IMAGE_SUFFIXES = (".JPEG", ".jpeg", ".jpg", ".JPG", ".png", ".PNG")


@dataclass(frozen=True)
class ImageNetFolders:
    train_root: Path
    val_root: Path

    def class_dirs(self, split: str) -> list[str]:
        root = self.train_root if split == "train" else self.val_root
        return sorted(d for d in os.listdir(root) if d.startswith("n") and len(d) == 9 and d[1:].isdigit())

    def list_images(self, split: str, wnid: str) -> list[str]:
        root = self.train_root if split == "train" else self.val_root
        folder = root / wnid
        return sorted(str(folder / f) for f in os.listdir(folder) if f.endswith(_IMAGE_SUFFIXES))


@dataclass
class StratifiedPopulation:
    """Target vs complement train populations with sealed permutations."""

    target_wnid: str
    target_paths: list[str]
    complement_paths: list[str]
    seed: int
    target_permutation: np.ndarray
    complement_permutation: np.ndarray

    @classmethod
    def build(cls, folders: ImageNetFolders, target_wnid: str, *, seed: int,
              complement_cache: Path | None = None) -> "StratifiedPopulation":
        target = folders.list_images("train", target_wnid)
        # the cache holds the COMPLETE train listing; the target is filtered out per category
        if complement_cache is not None and complement_cache.exists():
            listing = json.loads(complement_cache.read_text())["paths"]
        else:
            listing = []
            for wnid in folders.class_dirs("train"):
                listing.extend(folders.list_images("train", wnid))
            if complement_cache is not None:
                complement_cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = complement_cache.with_suffix(".tmp")
                tmp.write_text(json.dumps({"schema": "tractable_v1_imagenet_train_listing_v1", "paths": listing}))
                tmp.replace(complement_cache)
        marker = "/" + target_wnid + "/"
        complement = [p for p in listing if marker not in p]
        rng_t = np.random.default_rng(np.random.SeedSequence([seed, 1, _stable_int(target_wnid)]))
        rng_c = np.random.default_rng(np.random.SeedSequence([seed, 2, _stable_int(target_wnid)]))
        return cls(target_wnid=target_wnid, target_paths=target, complement_paths=complement, seed=seed,
                   target_permutation=rng_t.permutation(len(target)),
                   complement_permutation=rng_c.permutation(len(complement)))

    @property
    def counts(self) -> tuple[int, int]:
        return len(self.target_paths), len(self.complement_paths)

    def sealed_paths(self, stratum: str, start: int, count: int) -> list[str]:
        """Contexts ``[start, start+count)`` of the sealed permutation of one stratum.

        Legacy compatibility: positions wrap modulo the stratum size. This
        repeats the same order; it is not a fresh permutation or inferential
        epoch. New oracle consumers must use strict ``sealed_indices`` slices.
        """

        if stratum == "target":
            perm, paths = self.target_permutation, self.target_paths
        elif stratum == "complement":
            perm, paths = self.complement_permutation, self.complement_paths
        else:
            raise ValueError(stratum)
        n = len(paths)
        idx = [int(perm[(start + i) % n]) for i in range(count)]
        return [paths[i] for i in idx]

    def _stratum_axis(self, stratum: str) -> tuple[np.ndarray, list[str]]:
        if stratum == "target":
            return self.target_permutation, self.target_paths
        if stratum == "complement":
            return self.complement_permutation, self.complement_paths
        raise ValueError("unknown stratum")

    def sealed_indices(self, stratum: str, start: int, count: int) -> np.ndarray:
        """Stable listing-index IDs in a strict, nonwrapping permutation slice.

        IDs are meaningful only together with stratum and the listing hash in
        ``identity()``. Returning IDs grants no sampling or inference authority.
        """
        if type(start) is not int or type(count) is not int or start < 0 or count < 0:
            raise ValueError("start and count must be nonnegative integers")
        permutation, paths = self._stratum_axis(stratum)
        if start + count > len(paths):
            raise ValueError("sealed slice exceeds the declared stratum; no wrap allowed")
        result = np.array(permutation[start:start + count], dtype=np.int64, copy=True)
        result.flags.writeable = False
        return result

    def paths_for_indices(self, stratum: str, indices: Sequence[int] | np.ndarray) -> list[str]:
        """Resolve explicit stable IDs; repeated IDs preserve requested draw order."""
        _, paths = self._stratum_axis(stratum)
        ids = np.asarray(indices)
        if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu"):
            raise ValueError("context IDs must be one-dimensional integer indices")
        if ids.size and (np.any(ids < 0) or np.any(ids >= len(paths))):
            raise ValueError("context ID outside the declared stratum")
        return [paths[int(i)] for i in ids]

    def identity(self) -> dict[str, object]:
        return dict(schema="tractable_v1_stratified_population_v1", target_wnid=self.target_wnid,
                    target_count=len(self.target_paths), complement_count=len(self.complement_paths), seed=self.seed,
                    target_listing_sha256=_sha256_lines(self.target_paths),
                    complement_listing_sha256=_sha256_lines(self.complement_paths))


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _sha256_lines(lines: Iterable[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode()); h.update(b"\n")
    return h.hexdigest()


class _PathDataset(torch.utils.data.Dataset):
    def __init__(self, paths: Sequence[str]) -> None:
        from torchvision import transforms
        self.paths = list(paths)
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        from PIL import Image
        with Image.open(self.paths[index]) as im:
            return self.transform(im.convert("RGB"))


def load_images(paths: Sequence[str], *, batch_size: int = 64, num_workers: int = 8,
                device: torch.device | str = "cpu") -> Iterable[torch.Tensor]:
    """Yield normalized ``[b, 3, 224, 224]`` batches in path order."""

    loader = torch.utils.data.DataLoader(_PathDataset(paths), batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=str(device).startswith("cuda"))
    for batch in loader:
        yield batch.to(device, non_blocking=True)


def load_images_tensor(paths: Sequence[str], **kwargs) -> torch.Tensor:
    return torch.cat(list(load_images(paths, **kwargs)), dim=0)
