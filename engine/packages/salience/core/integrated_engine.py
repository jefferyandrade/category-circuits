"""
Integrated Gradients engine that swaps activation derivatives for closed-form
multipliers during a single backward pass.
"""
from __future__ import annotations

import contextlib
import inspect
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

try:  # pragma: no cover - optional import for wrappers
    from .frontier_wrappers import HookedMultiheadAttention, HookedSwiGLU
except Exception:  # pragma: no cover - fallback for circular imports during docs
    HookedMultiheadAttention = None  # type: ignore[assignment]
    HookedSwiGLU = None  # type: ignore[assignment]


_SUPPORTED_ACTIVATIONS = (
    nn.ReLU,
    nn.SiLU,
    nn.GELU,
    nn.LeakyReLU,
    nn.ReLU6,
    nn.ELU,
)
_ATTENTION_TYPES = tuple(
    cls
    for cls in (
        getattr(nn, "MultiheadAttention", None),
        HookedMultiheadAttention,
    )
    if cls is not None
)


def _apply_module(module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return module(tensor)


def _mu_relu(z: torch.Tensor) -> torch.Tensor:
    return (z > 0).to(dtype=z.dtype)


def _mu_limit_at_zero(module: nn.Module, z: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.SiLU):
        return torch.full_like(z, 0.5)
    if isinstance(module, nn.GELU):
        return torch.full_like(z, 0.5)
    if isinstance(module, nn.LeakyReLU):
        slope = torch.tensor(module.negative_slope, dtype=z.dtype, device=z.device)
        return torch.where(z >= 0, torch.ones_like(z), slope)
    if isinstance(module, nn.ELU):
        return torch.ones_like(z)
    if isinstance(module, nn.ReLU):
        return torch.zeros_like(z)
    if isinstance(module, nn.ReLU6):
        return torch.zeros_like(z)
    return torch.zeros_like(z)


def _mu_from_outputs(module: nn.Module, z: torch.Tensor, cached_output: Optional[torch.Tensor]) -> torch.Tensor:
    z_detached = z.detach()
    if cached_output is None:
        out = _apply_module(module, z_detached)
    else:
        out = cached_output.detach()
    baseline = torch.zeros_like(out)
    diff = out - baseline
    eps = torch.tensor(1e-6, dtype=z_detached.dtype, device=z_detached.device)
    safe = torch.where(z_detached.abs() > eps, z_detached, torch.ones_like(z_detached))
    mu = diff / safe
    zero_mask = z_detached.abs() <= eps
    if zero_mask.any():
        mu = torch.where(zero_mask, _mu_limit_at_zero(module, z_detached), mu)
    return mu


class IGBackpropEngine:
    """Internal engine that installs IG-aware hooks on the post-frontier subgraph."""

    def __init__(
        self,
        module_items: Iterable[Tuple[str, nn.Module]],
        *,
        frontier_name: Optional[str] = None,
    ) -> None:
        self.frontier_name = frontier_name
        self.module_items: List[Tuple[str, nn.Module]] = list(module_items)
        self.module_lookup: Dict[str, nn.Module] = dict(self.module_items)
        self.modules: List[nn.Module] = [
            module for _, module in self.module_items if isinstance(module, _SUPPORTED_ACTIVATIONS)
        ]
        self._forward_handles: List = []
        self._forward_post_handles: List = []
        self._backward_handles: List = []
        self._output_handles: List = []
        self._tracked_outputs: Dict[str, torch.Tensor] = {}
        self.prev_map: Dict[str, List[str]] = {name: [] for name, _ in self.module_items}
        self._tensor_sources: Dict[int, str] = {}
        self._residual_modules: Set[str] = set()
        self._tracked_module_ids: Set[int] = set()

        found_frontier = frontier_name is None
        prev: Optional[str] = None
        for name, _ in self.module_items:
            if not found_frontier:
                if name == frontier_name:
                    found_frontier = True
                    prev = name
                continue
            if prev is not None and prev != name:
                parents = self.prev_map.setdefault(name, [])
                if prev not in parents:
                    parents.append(prev)
            prev = name

        if not found_frontier and frontier_name is not None:
            prev = None
            for name, _ in self.module_items:
                if prev is not None and prev != name:
                    parents = self.prev_map.setdefault(name, [])
                    if prev not in parents:
                        parents.append(prev)
                prev = name

        if HookedMultiheadAttention is not None:
            for name, module in self.module_items:
                if isinstance(module, HookedMultiheadAttention):
                    parents = self.prev_map.setdefault(name, [])
                    for candidate in (
                        f"{name}.original.out_proj",
                        f"{name}.out_proj",
                    ):
                        if candidate in self.prev_map and candidate not in parents:
                            parents.append(candidate)

    def register_forward_hooks(self) -> "IGBackpropEngine":
        def _forward_pre_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            z = inputs[0]
            setattr(module, "_ig_preact", z.detach())

        def _forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output):
            tensor = _extract_tensor(output)
            if tensor is None:
                return
            setattr(module, "_ig_output", tensor.detach())

        for module in self.modules:
            handle = module.register_forward_pre_hook(_forward_pre_hook, with_kwargs=False)
            self._forward_handles.append(handle)
            post_handle = module.register_forward_hook(_forward_hook, with_kwargs=False)
            self._forward_post_handles.append(post_handle)

        for name, module in self.module_items:
            if not self._should_track_module(module):
                continue
            module_id = id(module)
            if module_id in self._tracked_module_ids:
                continue
            if self._is_multi_input_module(module):
                self._residual_modules.add(name)
            handle = module.register_forward_hook(self._capture_output_hook(name), with_kwargs=False)
            self._output_handles.append(handle)
            self._tracked_module_ids.add(module_id)
        return self

    @contextlib.contextmanager
    def backward_hooks(self):
        def _backward_hook(module: nn.Module, grad_input, grad_output):
            if not grad_output or grad_output[0] is None:
                return grad_input
            grad_out = grad_output[0]
            z: Optional[torch.Tensor] = getattr(module, "_ig_preact", None)
            out: Optional[torch.Tensor] = getattr(module, "_ig_output", None)
            if z is None:
                return grad_input

            if isinstance(module, nn.ReLU):
                mu = _mu_relu(z)
            elif isinstance(module, _SUPPORTED_ACTIVATIONS):
                mu = _mu_from_outputs(module, z, out)
            else:
                return grad_input
            return (grad_out * mu,)

        try:
            for module in self.modules:
                handle = module.register_full_backward_hook(_backward_hook)
                self._backward_handles.append(handle)
            yield
        finally:
            while self._backward_handles:
                self._backward_handles.pop().remove()

    def cleanup(self) -> None:
        while self._forward_handles:
            self._forward_handles.pop().remove()
        while self._forward_post_handles:
            self._forward_post_handles.pop().remove()
        for _, module in self.module_items:
            if hasattr(module, "_ig_preact"):
                try:
                    delattr(module, "_ig_preact")
                except AttributeError:
                    pass
            if hasattr(module, "_ig_output"):
                try:
                    delattr(module, "_ig_output")
                except AttributeError:
                    pass
        while self._output_handles:
            self._output_handles.pop().remove()
        self._tracked_outputs.clear()
        self._tensor_sources.clear()
        self._residual_modules.clear()
        self._tracked_module_ids.clear()

    def _capture_output_hook(self, name: str):
        def _hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output):
            tensor = _extract_tensor(output)
            parents = self._resolve_parent_names(inputs)
            if parents:
                existing = self.prev_map.setdefault(name, [])
                for parent in parents:
                    if parent not in existing:
                        existing.append(parent)
                if len(existing) > 1:
                    self._residual_modules.add(name)
            elif name not in self.prev_map:
                self.prev_map[name] = []
            if tensor is None:
                return
            self._tensor_sources[id(tensor)] = name
            if not tensor.requires_grad:
                return
            tensor.retain_grad()
            self._tracked_outputs[name] = tensor

        return _hook

    def _should_track_module(self, module: nn.Module) -> bool:
        if getattr(module, "ig_capture_output", False):
            return True
        if _ATTENTION_TYPES and isinstance(module, _ATTENTION_TYPES):
            return True
        if HookedSwiGLU is not None and isinstance(module, HookedSwiGLU):
            return True
        if self._is_multi_input_module(module):
            return True
        return False

    def _resolve_parent_names(self, inputs: Tuple[torch.Tensor, ...]) -> List[str]:
        parents: List[str] = []
        for tensor in _iter_tensors(inputs):
            source = self._tensor_sources.get(id(tensor))
            if source is not None and source not in parents:
                parents.append(source)
        return parents

    @staticmethod
    def _is_multi_input_module(module: nn.Module) -> bool:
        forward = getattr(module, "forward", None)
        if forward is None:
            return False
        try:
            signature = inspect.signature(forward)
        except (TypeError, ValueError):
            return False

        required_positionals = 0
        for param in signature.parameters.values():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                return True
            if param.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                continue
            if param.default is inspect._empty:
                required_positionals += 1
        if required_positionals > 1:
            return True

        unbound_forward = getattr(type(module), "forward", None)
        if unbound_forward is None or unbound_forward is forward:
            return False
        try:
            signature = inspect.signature(unbound_forward)
        except (TypeError, ValueError):
            return False

        required_positionals = 0
        for idx, param in enumerate(signature.parameters.values()):
            if idx == 0 and param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                continue
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                return True
            if param.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                continue
            if param.default is inspect._empty:
                required_positionals += 1
        return required_positionals > 1

    def _propagate_residual(
        self,
        multipliers: Dict[str, torch.Tensor],
        name: str,
        module: Optional[nn.Module],
    ) -> None:
        value = multipliers.get(name)
        if value is None:
            return
        for parent in self.prev_map.get(name, []):
            multipliers.setdefault(parent, value)

    def pop_tracked_multipliers(self) -> Dict[str, torch.Tensor]:
        multipliers: Dict[str, torch.Tensor] = {}
        for name, tensor in list(self._tracked_outputs.items()):
            module = self.module_lookup.get(name)
            grad = tensor.grad
            tensor.grad = None
            tensor_id = id(tensor)
            handled = False
            value: Optional[torch.Tensor] = None
            if grad is not None:
                grad_local = grad.detach()
            else:
                grad_local = torch.zeros_like(tensor)

            try:
                if module is not None:
                    if HookedMultiheadAttention is not None and isinstance(module, HookedMultiheadAttention):
                        module.apply_local_ig(grad_local)
                        value = getattr(module, "saved_integrated_multipliers", None)
                        handled = value is not None
                    elif HookedSwiGLU is not None and isinstance(module, HookedSwiGLU):
                        module.apply_local_ig(grad_local)
                        value = getattr(module, "saved_integrated_multipliers", None)
                        handled = value is not None
                    else:
                        local = getattr(module, "apply_local_ig", None)
                        if callable(local):
                            local(grad_local)
                            value = getattr(module, "saved_integrated_multipliers", None)
                            if value is None and grad is not None:
                                value = grad.detach()
                            handled = value is not None
                        elif grad is not None and isinstance(module, _SUPPORTED_ACTIVATIONS):
                            z = getattr(module, "_ig_preact", None)
                            out = getattr(module, "_ig_output", None)
                            if z is not None:
                                if isinstance(module, nn.ReLU):
                                    mu = _mu_relu(z)
                                else:
                                    mu = _mu_from_outputs(module, z, out)
                                value = grad.detach() * mu
                                handled = True

                if not handled:
                    if grad is None:
                        continue
                    value = grad.detach()

                if value is None:
                    continue

                multipliers[name] = value
                if module is not None and hasattr(module, "saved_integrated_multipliers"):
                    module.saved_integrated_multipliers = value
                parents = self.prev_map.get(name, [])
                if parents:
                    if len(parents) > 1 or name in self._residual_modules:
                        self._propagate_residual(multipliers, name, module)
                    else:
                        parent = parents[0]
                        if parent not in multipliers:
                            multipliers[parent] = value
                clear = getattr(module, "clear_state", None)
                if callable(clear):
                    clear()
            finally:
                self._tensor_sources.pop(tensor_id, None)
        self._tracked_outputs.clear()
        return multipliers


class IntegratedEngine:
    """Convenience wrapper that manages hook lifecycle for Integrated Gradients."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        *,
        frontier_name: Optional[str] = None,
        module_items: Optional[Iterable[Tuple[str, nn.Module]]] = None,
    ) -> None:
        if module_items is None:
            if model is None or frontier_name is None:
                raise ValueError("model and frontier_name are required when module_items is not provided")
            module_items = collect_post_frontier_modules(model, frontier_name)
        self._engine = IGBackpropEngine(module_items, frontier_name=frontier_name)
        self._engine.register_forward_hooks()
        self._closed = False

    def backward_context(self):
        return self._engine.backward_hooks()

    def collect(self) -> Dict[str, torch.Tensor]:
        return self._engine.pop_tracked_multipliers()

    def cleanup(self) -> None:
        if not self._closed:
            self._engine.cleanup()
            self._closed = True

    def compute_multipliers(
        self,
        scalar: torch.Tensor,
        *,
        inputs: Optional[Iterable[torch.Tensor]] = None,
        retain_graph: bool = False,
        allow_unused: bool = True,
    ) -> Dict[str, torch.Tensor]:
        tensors: Optional[Tuple[torch.Tensor, ...]] = None
        if inputs is not None:
            if isinstance(inputs, (list, tuple)):
                tensors = tuple(inputs)
            else:
                tensors = (inputs,)

        with self.backward_context():
            if tensors is None:
                scalar.backward(retain_graph=retain_graph)
            else:
                torch.autograd.grad(
                    scalar,
                    tensors,
                    retain_graph=retain_graph,
                    allow_unused=allow_unused,
                )
        multipliers = self.collect()
        self.cleanup()
        return multipliers


def _extract_tensor(output) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = _extract_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def collect_post_frontier_modules(
    model: nn.Module, frontier_name: str
) -> List[Tuple[str, nn.Module]]:
    if isinstance(model, nn.DataParallel):
        model = model.module  # type: ignore[assignment]
    named_modules = list(model.named_modules())
    if not frontier_name:
        return named_modules

    module_dict = dict(named_modules)
    if frontier_name in module_dict:
        for idx, (name, _) in enumerate(named_modules):
            if name == frontier_name:
                return named_modules[idx:]

    prefix = f"{frontier_name}."
    for idx, (name, _) in enumerate(named_modules):
        if name.startswith(prefix):
            return named_modules[idx:]

    return named_modules


__all__ = ["IGBackpropEngine", "IntegratedEngine", "collect_post_frontier_modules"]
