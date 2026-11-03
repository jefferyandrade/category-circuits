"""Process-lifetime CUDA generator storage for captures in any autograd mode."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


_GENERATOR_GRAPHS: dict[int, torch.cuda.CUDAGraph] = {}
_GENERATOR_LOCK = threading.Lock()


def initialize_cuda_graph_state(device: torch.device | str) -> None:
    """Register the device's default generator before its first capture.

    PyTorch allocates the generator's shared graph seed/offset tensors on
    first registration. They must be normal tensors: capture_begin and
    replay update them even when the caller has left inference mode.
    Retaining this uncaptured graph keeps that registration alive for the
    worker's lifetime. Registration draws no random numbers and performs
    no capture or host synchronization. CPU workers do not touch CUDA.
    """
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    if index in _GENERATOR_GRAPHS:
        return
    with _GENERATOR_LOCK:
        if index in _GENERATOR_GRAPHS:
            return
        with torch.cuda.device(index), torch.inference_mode(False):
            graph = torch.cuda.CUDAGraph()
            graph.register_generator_state(torch.cuda.default_generators[index])
        _GENERATOR_GRAPHS[index] = graph
