"""AlexNet route axis, masked functional forward, and factored capture.

The seven prunable affine layers of torchvision AlexNet and their immediate
ReLUs (the frozen map of ``optimized_irg_providers``):

    features.0  -> features.1     conv  64 x (3*11*11)
    features.3  -> features.4     conv 192 x (64*5*5)
    features.6  -> features.7     conv 384 x (192*3*3)
    features.8  -> features.9     conv 256 x (384*3*3)
    features.10 -> features.11    conv 256 x (256*3*3)
    classifier.1-> classifier.2   fc  4096 x 9216
    classifier.4-> classifier.5   fc  4096 x 4096
    classifier.6 (readout)        fixed, never pruned

Routes are individual kernel weights (conv) and individual weights (fc); a
fiber is a destination unit (output channel or hidden unit).  The route axis is
the concatenation of the seven weight tensors flattened in ``[F, S]`` order
where ``S`` is the unfold order ``(C_in, kh, kw)``, so a flat route vector
``z`` indexes exactly the same coordinates as the weights.

For every context the capture returns, per layer:

    X  : unfolded input        [B, S, P]   (P = 1 for fc)
    a  : preactivation         [B, F, P]
    G  : dQuery / d relu(a)    [B, F, P]   post-activation cotangent
    Gt : G * phi(a, bias)      [B, F, P]   conserving (corrected AS) cotangent
    D  : G * 1[a > 0]          [B, F, P]   literal-Jacobian cotangent

so that the two registered row streams are exactly

    e_c[f, j] = -W[f, j] * sum_p Gt[c, f, p] X[c, j, p]      (corrected rows)
    J_c[f, j] = -W[f, j] * sum_p  D[c, f, p] X[c, j, p]      (literal-J rows)

evaluated as real numbers.  Masked
routes (m = 0) are ineligible and their rows are excluded by the mask factor.

AlexNet's ``avgpool`` is ``AdaptiveAvgPool2d((6, 6))``; at the canonical
224 x 224 input the conv stack already ends at 6 x 6, so the pool is the exact
identity and is replaced by it here (no nondeterministic backward kernel).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as tnf

from ..modern_v1_conserving_attribution import relu_path_open_fraction

ROUTE_LAYER_SPECS: tuple[tuple[str, str, str], ...] = (
    ("features.0", "features.1", "conv"),
    ("features.3", "features.4", "conv"),
    ("features.6", "features.7", "conv"),
    ("features.8", "features.9", "conv"),
    ("features.10", "features.11", "conv"),
    ("classifier.1", "classifier.2", "linear"),
    ("classifier.4", "classifier.5", "linear"),
)
READOUT_PATH = "classifier.6"
CANONICAL_INPUT_HW = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class LayerGeometry:
    """Static per-layer route geometry."""

    name: str
    relu_name: str
    kind: str
    fiber_count: int
    routes_per_fiber: int
    route_offset: int
    weight_shape: tuple[int, ...]
    conv_params: tuple[int, int, int] | None  # (kernel, stride, padding) for conv

    @property
    def route_count(self) -> int:
        return self.fiber_count * self.routes_per_fiber

    @property
    def route_slice(self) -> slice:
        return slice(self.route_offset, self.route_offset + self.route_count)


@dataclass(frozen=True)
class RouteAxis:
    """Concatenated route axis over the seven prunable AlexNet layers."""

    layers: tuple[LayerGeometry, ...]

    @property
    def route_count(self) -> int:
        return sum(layer.route_count for layer in self.layers)

    @property
    def fiber_count(self) -> int:
        return sum(layer.fiber_count for layer in self.layers)

    def fiber_offsets(self) -> tuple[int, ...]:
        out, acc = [], 0
        for layer in self.layers:
            out.append(acc)
            acc += layer.fiber_count
        return tuple(out)

    def split(self, flat: torch.Tensor) -> list[torch.Tensor]:
        """Split a flat route vector ``[N]`` or ``[N, L]`` into ``[F, S]`` views."""

        views = []
        for layer in self.layers:
            piece = flat[layer.route_slice]
            views.append(piece.reshape(layer.fiber_count, layer.routes_per_fiber, *piece.shape[1:]))
        return views

    def concat(self, pieces: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat([p.reshape(-1, *p.shape[2:]) for p in pieces], dim=0)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "tractable_v1_alexnet_route_axis_v1",
            "route_count": self.route_count,
            "fiber_count": self.fiber_count,
            "layers": [
                {
                    "name": l.name, "relu": l.relu_name, "kind": l.kind,
                    "fiber_count": l.fiber_count, "routes_per_fiber": l.routes_per_fiber,
                    "route_offset": l.route_offset, "weight_shape": list(l.weight_shape),
                    "conv_params": None if l.conv_params is None else list(l.conv_params),
                }
                for l in self.layers
            ],
        }


def _module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    node: Any = model
    for part in path.split("."):
        node = getattr(node, part) if not part.isdigit() else node[int(part)]
    return node


def build_route_axis(model: torch.nn.Module) -> RouteAxis:
    layers, offset = [], 0
    for name, relu_name, kind in ROUTE_LAYER_SPECS:
        module = _module(model, name)
        weight = module.weight
        if kind == "conv":
            if not isinstance(module, torch.nn.Conv2d) or module.groups != 1 or module.dilation != (1, 1):
                raise TypeError(f"{name} must be a plain Conv2d")
            k = int(module.kernel_size[0]); s = int(module.stride[0]); p = int(module.padding[0])
            if module.kernel_size != (k, k) or module.stride != (s, s) or module.padding != (p, p):
                raise TypeError(f"{name} must have square kernel/stride/padding")
            conv_params: tuple[int, int, int] | None = (k, s, p)
            fibers, per_fiber = int(weight.shape[0]), int(weight[0].numel())
        else:
            if not isinstance(module, torch.nn.Linear):
                raise TypeError(f"{name} must be Linear")
            conv_params = None
            fibers, per_fiber = int(weight.shape[0]), int(weight.shape[1])
        layers.append(LayerGeometry(name, relu_name, kind, fibers, per_fiber, offset,
                                    tuple(int(d) for d in weight.shape), conv_params))
        offset += fibers * per_fiber
    return RouteAxis(tuple(layers))


class MaskedAlexNet:
    """Functional AlexNet forward with a route mask and factored capture.

    ``weights`` are the frozen dense weights; ``mask`` is a flat 0/1 route
    vector.  The forward uses ``W * mask`` for the seven route layers and the
    fixed readout.  Only the canonical 224 x 224 input is admitted, where
    ``avgpool`` is the identity.
    """

    def __init__(self, model: torch.nn.Module, *, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.axis = build_route_axis(model)
        self.weights: list[torch.Tensor] = []
        self.biases: list[torch.Tensor] = []
        for layer in self.axis.layers:
            module = _module(model, layer.name)
            self.weights.append(module.weight.detach().to(self.device, dtype).clone())
            self.biases.append(module.bias.detach().to(self.device, dtype).clone())
        readout = _module(model, READOUT_PATH)
        self.readout_weight = readout.weight.detach().to(self.device, dtype).clone()
        self.readout_bias = readout.bias.detach().to(self.device, dtype).clone()
        self.pool_after = {"features.0", "features.3", "features.10"}  # MaxPool2d(3, 2)
        self.mask = torch.ones(self.axis.route_count, device=self.device, dtype=dtype)
        self._dense_biases = [b.clone() for b in self.biases]
        self._dense_readout_bias = self.readout_bias.clone()

    def set_biases(self, overrides: "dict[str, torch.Tensor] | None") -> None:
        """Override layer biases (v0 circuits ship ``updated_biases``); ``None`` restores the dense biases."""

        self.biases = [b.clone() for b in self._dense_biases]
        self.readout_bias = self._dense_readout_bias.clone()
        if overrides is None:
            return
        for i, layer in enumerate(self.axis.layers):
            if layer.name in overrides:
                self.biases[i] = torch.as_tensor(overrides[layer.name], device=self.device, dtype=self.dtype).reshape(self.biases[i].shape).clone()
        if READOUT_PATH in overrides:
            self.readout_bias = torch.as_tensor(overrides[READOUT_PATH], device=self.device, dtype=self.dtype).reshape(self.readout_bias.shape).clone()

    # -- mask handling --------------------------------------------------------
    def set_mask(self, mask: torch.Tensor | np.ndarray) -> None:
        self.mask = self._forward_mask(mask, require_binary=True).clone()

    def _forward_mask(self, mask: torch.Tensor | np.ndarray, *, require_binary: bool = False) -> torch.Tensor:
        """Validate an ephemeral dose without installing it as the hard state."""
        m = torch.as_tensor(mask, device=self.device)
        if m.ndim != 1 or m.numel() != self.axis.route_count:
            raise ValueError("forward mask must be a flat vector on the route axis")
        if not bool(torch.isfinite(m).all()) or not bool(((m >= 0) & (m <= 1)).all()):
            raise ValueError("forward mask must be finite and lie in [0, 1]")
        if require_binary and not bool(((m == 0) | (m == 1)).all()):
            raise ValueError("dose replay endpoints must be binary hard masks")
        return m.to(dtype=self.dtype)

    def masked_weights(self, *, mask_override: torch.Tensor | np.ndarray | None = None) -> list[torch.Tensor]:
        mask = self.mask if mask_override is None else self._forward_mask(mask_override)
        pieces = self.axis.split(mask)
        return [w * p.reshape(w.shape) for w, p in zip(self.weights, pieces, strict=True)]

    @property
    def density(self) -> float:
        return float(self.mask.sum().item()) / float(self.axis.route_count)

    # -- forward ---------------------------------------------------------------
    def forward(self, images: torch.Tensor, *, capture: bool = False,
                mask_override: torch.Tensor | np.ndarray | None = None,
                use_dense_biases: bool = False) -> dict[str, Any]:
        """Evaluate an optional fractional mask without changing mask or bias state.

        ``use_dense_biases`` selects the original loaded biases, including the
        fixed readout bias, for a matched dense referent. It does not install
        them. Persisted ``set_mask`` retains its binary contract.
        """

        if type(use_dense_biases) is not bool:
            raise TypeError("use_dense_biases must be bool")
        if images.ndim != 4 or tuple(images.shape[-2:]) != CANONICAL_INPUT_HW:
            raise ValueError("canonical AlexNet capture requires [B, 3, 224, 224] inputs")
        x = images.to(self.device, self.dtype)
        weights = self.masked_weights(mask_override=mask_override)
        biases = self._dense_biases if use_dense_biases else self.biases
        readout_bias = self._dense_readout_bias if use_dense_biases else self.readout_bias
        inputs, preacts, posts = [], [], []
        for layer, w, b in zip(self.axis.layers, weights, biases, strict=True):
            if layer.kind == "conv":
                k, s, p = layer.conv_params  # type: ignore[misc]
                a = tnf.conv2d(x, w, b, stride=s, padding=p)
            else:
                if x.ndim == 4:
                    if tuple(x.shape[-2:]) != (6, 6):
                        raise RuntimeError("conv stack did not end at 6x6; avgpool identity assumption violated")
                    x = torch.flatten(x, 1)
                a = tnf.linear(x, w, b)
            if capture:
                inputs.append(x)
                preacts.append(a)
                a = a.requires_grad_(True) if not a.requires_grad else a
            h = torch.relu(a)
            if capture:
                posts.append(h)
            if layer.name in self.pool_after:
                h = tnf.max_pool2d(h, kernel_size=3, stride=2)
            x = h
        logits = tnf.linear(x, self.readout_weight, readout_bias)
        out: dict[str, Any] = {"logits": logits}
        if capture:
            out.update(inputs=inputs, preacts=preacts, posts=posts)
        return out

    # -- factored capture ------------------------------------------------------
    def capture(self, images: torch.Tensor, target_class: int, *,
                factor_dtype: torch.dtype = torch.float64) -> "Factors":
        """Capture the factors of both registered row streams for a batch of contexts.

        Query: the raw pre-softmax target-class logit (canonical evidence).
        """

        with torch.enable_grad():
            out = self.forward(images, capture=True)
            logits = out["logits"]
            query = logits[:, int(target_class)].sum()
            grads = torch.autograd.grad(query, out["posts"], retain_graph=False, allow_unused=False)
        layers: list[LayerFactors] = []
        for layer, x, a, g, w, b in zip(self.axis.layers, out["inputs"], out["preacts"], grads,
                                         self.weights, self.biases, strict=True):
            a = a.detach(); g = g.detach()
            if layer.kind == "conv":
                k, s, p = layer.conv_params  # type: ignore[misc]
                X = tnf.unfold(x.detach(), kernel_size=k, stride=s, padding=p)  # [B, S, P]
                B_, F_ = a.shape[0], a.shape[1]
                a_flat = a.reshape(B_, F_, -1)
                g_flat = g.reshape(B_, F_, -1)
                bias = b.reshape(1, F_, 1)
            else:
                X = x.detach().unsqueeze(-1)  # [B, S, 1]
                a_flat = a.unsqueeze(-1)
                g_flat = g.unsqueeze(-1)
                bias = b.reshape(1, -1, 1)
            phi = relu_path_open_fraction(a_flat, bias.expand_as(a_flat).contiguous())
            layers.append(LayerFactors(
                geometry=layer,
                X=X.to(factor_dtype), a=a_flat.to(factor_dtype),
                G=g_flat.to(factor_dtype),
                Gt=(g_flat * phi).to(factor_dtype),
                D=(g_flat * (a_flat > 0).to(g_flat.dtype)).to(factor_dtype),
                W=w.reshape(layer.fiber_count, layer.routes_per_fiber).to(factor_dtype),
                mask=self.axis.split(self.mask)[len(layers)].reshape(
                    layer.fiber_count, layer.routes_per_fiber).to(factor_dtype),
            ))
        return Factors(layers=tuple(layers), logits=logits.detach(), target_class=int(target_class),
                       context_count=int(images.shape[0]))


@torch.no_grad()
def raw_target_logits(net: MaskedAlexNet, images: torch.Tensor, *, target_class: int,
                      batch_size: int = 256,
                      mask_override: torch.Tensor | np.ndarray | None = None,
                      use_dense_biases: bool = False) -> torch.Tensor:
    """Batched raw logits on the model device, with no persistent intervention."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(target_class) is not int or not 0 <= target_class < net.readout_bias.numel():
        raise ValueError("target_class is outside the readout axis")
    if images.ndim != 4 or images.shape[0] == 0:
        raise ValueError("raw target-logit replay requires a nonempty image batch")
    values = [net.forward(images[start:start + batch_size], mask_override=mask_override,
                          use_dense_biases=use_dense_biases)["logits"][:, target_class].detach()
              for start in range(0, images.shape[0], batch_size)]
    result = torch.cat(values)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("raw target-logit replay produced nonfinite values")
    return result


@dataclass(frozen=True)
class FunctionalDoseReplay:
    """Actual finite forward values; no whole-path or chart certificate."""
    parent_logits: torch.Tensor
    half_logits: torch.Tensor
    child_logits: torch.Tensor
    doses: tuple[float, float, float] = (0.0, 0.5, 1.0)


@torch.no_grad()
def functional_half_dose(net: MaskedAlexNet, images: torch.Tensor, *,
                         parent_mask: torch.Tensor | np.ndarray,
                         child_mask: torch.Tensor | np.ndarray,
                         target_class: int, batch_size: int = 256) -> FunctionalDoseReplay:
    """Replay ``parent - eta*(parent-child)`` at eta=0,1/2,1.

    The midpoint is an actual multilayer forward, never interpolated logits.
    Current biases are fixed across all three evaluations. One temporary
    route-sized midpoint mask and the existing forward workspace are used;
    this helper does not capture factors or allocate context-by-route rows.
    No mask or bias is installed, even if validation or a forward raises.
    """
    parent = net._forward_mask(parent_mask, require_binary=True)
    child = net._forward_mask(child_mask, require_binary=True)
    if not bool((child <= parent).all()):
        raise ValueError("dose replay child must be a removal-only subset of its parent")
    half = parent + child
    half.mul_(0.5)
    values = [raw_target_logits(net, images, target_class=target_class,
                               batch_size=batch_size, mask_override=mask)
              for mask in (parent, half, child)]
    return FunctionalDoseReplay(*values)


@dataclass
class LayerFactors:
    geometry: LayerGeometry
    X: torch.Tensor   # [B, S, P]
    a: torch.Tensor   # [B, F, P]
    G: torch.Tensor   # [B, F, P]
    Gt: torch.Tensor  # [B, F, P]
    D: torch.Tensor   # [B, F, P]
    W: torch.Tensor   # [F, S]
    mask: torch.Tensor  # [F, S]

    @property
    def eligible(self) -> torch.Tensor:
        return self.mask

    def dense_rows(self, stream: str, contexts: Sequence[int] | None = None) -> torch.Tensor:
        """Materialize rows ``[B, F, S]`` for tests and tiny layers only."""

        cot = {"corrected": self.Gt, "literal_j": self.D}[stream]
        if contexts is not None:
            cot, X = cot[list(contexts)], self.X[list(contexts)]
        else:
            X = self.X
        return -(self.W * self.mask).unsqueeze(0) * torch.einsum("bfp,bjp->bfj", cot, X)


@dataclass
class Factors:
    layers: tuple[LayerFactors, ...]
    logits: torch.Tensor          # [B, 1000] raw logits at the current mask
    target_class: int
    context_count: int

    @property
    def device(self) -> torch.device:
        return self.layers[0].X.device

    @property
    def dtype(self) -> torch.dtype:
        return self.layers[0].X.dtype

    def target_logits(self) -> torch.Tensor:
        return self.logits[:, self.target_class].to(self.dtype)

    def conservation_check(self) -> torch.Tensor:
        """Exact all-route conserving total per layer: sum_f Gt*(a-b) == G*(relu(a)-relu(b)).

        Returns the maximum absolute discrepancy per layer.
        """

        out = []
        for layer in self.layers:
            b = layer.a.new_zeros(layer.a.shape)  # bias broadcast recovered from a - W x below
            # b = a - sum_j W_fj X_j; recompute the drive to obtain the bias baseline exactly
            drive = torch.einsum("fj,bjp->bfp", layer.W * layer.mask, layer.X)
            b = layer.a - drive
            lhs = (layer.Gt * (layer.a - b)).sum(dim=(1, 2))
            rhs = (layer.G * (torch.relu(layer.a) - torch.relu(b))).sum(dim=(1, 2))
            out.append((lhs - rhs).abs().max())
        return torch.stack(out)


def capture_factor_batches(net: MaskedAlexNet, batches, *, context_count: int,
                           target_class: int, factor_dtype: torch.dtype = torch.float64,
                           max_factor_bytes: int | None = None) -> Factors:
    """Retain one factor bank plus one physical batch; never retain image batches.

    The optional limit covers the final factor/logit bank, not forward/BLAS
    workspace. A short, excess, empty or inconsistent stream refuses.
    """
    if type(context_count) is not int or context_count <= 0:
        raise ValueError("capture context_count must be positive")
    if max_factor_bytes is not None and (type(max_factor_bytes) is not int or max_factor_bytes <= 0):
        raise ValueError("max_factor_bytes must be positive")
    names = ("X", "a", "G", "Gt", "D")
    layers = None
    logits = None
    offset = 0
    iterator = iter(batches)
    try:
        for images in iterator:
            n = int(images.shape[0])
            if n <= 0 or offset + n > context_count:
                raise ValueError("capture batches exceed or violate registered context count")
            part = net.capture(images, target_class, factor_dtype=factor_dtype)
            del images
            if layers is None:
                bank_bytes = sum(context_count * getattr(l, name)[0].numel() * getattr(l, name).element_size()
                                 for l in part.layers for name in names)
                bank_bytes += context_count * part.logits[0].numel() * part.logits.element_size()
                bank_bytes += sum(t.numel() * t.element_size() for l in part.layers for t in (l.W, l.mask))
                if max_factor_bytes is not None and bank_bytes > max_factor_bytes:
                    raise ValueError("registered factor bank exceeds max_factor_bytes")
                if n == context_count:
                    # Still exhaust the stream to detect a forbidden extra batch.
                    layers, logits = part.layers, part.logits
                else:
                    layers = tuple(LayerFactors(
                        geometry=l.geometry, W=l.W, mask=l.mask,
                        **{name: torch.empty((context_count, *getattr(l, name).shape[1:]),
                                            device=getattr(l, name).device, dtype=getattr(l, name).dtype)
                           for name in names}) for l in part.layers)
                    logits = torch.empty((context_count, *part.logits.shape[1:]), device=part.logits.device, dtype=part.logits.dtype)
            if part.layers is not layers:
                for target, source in zip(layers, part.layers, strict=True):
                    if target.geometry != source.geometry:
                        raise ValueError("capture batch route geometry changed")
                    for name in names:
                        getattr(target, name)[offset:offset+n].copy_(getattr(source, name))
                logits[offset:offset+n].copy_(part.logits)
                del source
            offset += n
            del part
        if offset != context_count:
            raise ValueError("capture stream ended before registered context count")
        return Factors(layers=tuple(layers), logits=logits, target_class=target_class, context_count=context_count)
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def load_torchvision_alexnet(weights: str | None = "IMAGENET1K_V1") -> torch.nn.Module:
    import torchvision

    if weights is None:
        model = torchvision.models.alexnet(weights=None)
    else:
        model = torchvision.models.alexnet(weights=getattr(torchvision.models.AlexNet_Weights, weights))
    return model.eval()


def pack_mask(mask: torch.Tensor | np.ndarray) -> bytes:
    arr = np.asarray(mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask)
    return np.packbits(arr.astype(np.uint8).reshape(-1)).tobytes()


def unpack_mask(packed: bytes, route_count: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(packed, dtype=np.uint8))[:route_count].astype(np.float32)
