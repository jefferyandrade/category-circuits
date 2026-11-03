"""
Replay helper utilities shared by salience frontier modules.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch


class TensorReplayProvider:
    """
    Minimal replay provider that keeps a detached copy of a batched tensor and
    serves slices on demand. Pipelines that can recompute activations cheaply
    can supply custom providers exposing the same ``batch_size`` attribute and
    ``get(start, end, device=..., dtype=...)`` method to avoid caching entirely.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        *,
        storage_device: Optional[torch.device] = None,
    ) -> None:
        detached = tensor.detach()
        target_device = storage_device or torch.device("cpu")
        if not isinstance(target_device, torch.device):
            target_device = torch.device(target_device)
        if detached.device != target_device:
            detached = detached.to(device=target_device, non_blocking=True)
        self._storage = detached.contiguous()
        self.batch_size = int(self._storage.size(0))

    def free(self) -> None:
        """Release the cached storage to help upstream layers reclaim memory."""
        self._storage = None
        self.batch_size = 0

    def get(
        self,
        start: int,
        end: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._storage is None:
            raise RuntimeError("Replay provider storage has been freed.")
        chunk = self._storage[start:end]
        if chunk.dtype != dtype or chunk.device != device:
            chunk = chunk.to(device=device, dtype=dtype, non_blocking=True)
        return chunk

    def get_indices(
        self,
        indices: Union[Sequence[int], torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._storage is None:
            raise RuntimeError("Replay provider storage has been freed.")
        if isinstance(indices, torch.Tensor):
            indices = indices.to(device="cpu", dtype=torch.long).tolist()
        chunk = self._storage[indices]
        if chunk.dtype != dtype or chunk.device != device:
            chunk = chunk.to(device=device, dtype=dtype, non_blocking=True)
        return chunk


def replay_batch_size(source: Union[torch.Tensor, "TensorReplayProvider"], fallback: int) -> int:
    """Inspect the replay source to determine the total number of samples."""
    return int(getattr(source, "batch_size", fallback))


def replay_feature_shape(source: Union[torch.Tensor, "TensorReplayProvider", object]) -> Optional[tuple[int, ...]]:
    """Return the non-batch tensor shape represented by a replay source, when known."""
    storage = getattr(source, "_storage", None)
    if torch.is_tensor(storage):
        return tuple(int(v) for v in storage.shape[1:])
    if torch.is_tensor(source):
        return tuple(int(v) for v in source.shape[1:])
    shape = getattr(source, "shape", None)
    if shape is None:
        shape = getattr(source, "tensor_shape", None)
    if shape is None:
        return None
    try:
        shape_tuple = tuple(int(v) for v in shape)
    except (TypeError, ValueError):
        return None
    return tuple(shape_tuple[1:])


def replay_matches_tensor(source: Union[torch.Tensor, "TensorReplayProvider", object], tensor: torch.Tensor) -> bool:
    """Return true only when a replay source exactly represents ``tensor``'s batch axes."""
    expected = getattr(source, "batch_size", None)
    if expected is not None and int(expected) != int(tensor.size(0)):
        return False
    feature_shape = replay_feature_shape(source)
    if feature_shape is None:
        return False
    return feature_shape == tuple(int(v) for v in tensor.shape[1:])


def fetch_replay_slice(
    source: Union[torch.Tensor, TensorReplayProvider],
    start: int,
    end: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Retrieve a slice either from a replay provider or directly from a tensor."""
    getter = getattr(source, "get", None)
    if callable(getter):
        return getter(start, end, device=device, dtype=dtype)
    return source[start:end].to(device=device, dtype=dtype, non_blocking=True)


def fetch_replay_indices(
    source: Union[torch.Tensor, TensorReplayProvider],
    indices: Union[torch.Tensor, Sequence[int]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Retrieve indexed samples either from a replay provider or directly from a tensor."""
    getter = getattr(source, "get_indices", None)
    if callable(getter):
        return getter(indices, device=device, dtype=dtype)
    return source[indices].to(device=device, dtype=dtype, non_blocking=True)


class ReplayTensorProxy:
    """
    Lightweight tensor-like wrapper that surfaces batched slices from a replay provider on demand.
    Only the operations required by the scoring routines are implemented (``size``, ``__len__``,
    and slicing with unit stride).
    """

    def __init__(
        self,
        provider: TensorReplayProvider,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._provider = provider
        self._size = int(getattr(provider, "batch_size"))
        self._device = device
        self._dtype = dtype

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return torch.Size([self._size])
        if dim == 0:
            return self._size
        raise ValueError("Replay proxy only supports size(0).")

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(self._size)
            if step != 1:
                raise ValueError("Replay proxy only supports contiguous slices.")
            return self._provider.get(start, stop, device=self._device, dtype=self._dtype)
        if isinstance(item, int):
            if item < 0:
                item += self._size
            if not 0 <= item < self._size:
                raise IndexError("Replay proxy index out of range.")
            return self._provider.get(item, item + 1, device=self._device, dtype=self._dtype).squeeze(0)
        raise TypeError(f"Unsupported index type {type(item)} for replay proxy.")

    def get(self, start: int, end: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._provider.get(start, end, device=device, dtype=dtype)

    def get_indices(
        self,
        indices: Union[Sequence[int], torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._provider.get_indices(indices, device=device, dtype=dtype)


__all__ = [
    "TensorReplayProvider",
    "ReplayTensorProxy",
    "replay_batch_size",
    "fetch_replay_slice",
    "fetch_replay_indices",
]
