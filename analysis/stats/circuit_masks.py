"""The circuits' weight layout and the checked mask reader, with the checkpoints and layer names they use.

A circuit's mask (final_mask.packbits) holds one bit per prunable weight of the network, in the order of the
network's weight layout (analysis/common/architectures.py), which gives each weight layer's offset and size.
load_verified_mask reads a mask and refuses it (MaskRefused) unless its sha256 is the one its receipt records, its
length, padding, layout hash and weight count fit the layout, and every layer's kept count equals the receipt's (the
layer's size minus the weights removed by its accepted pruning steps). The receipts call weights routes.

Also here: the checkpoint files the circuits were extracted from and the weights' signs in them, the short layer names
(conv1..fc7, conv3_1..), each category's subgroup (block50, at most 50 categories), and a bootstrap over subgroups.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

NETWORK_FAMILY = {"resnet50v2": "resnet50_conv_kernel_weight", "resnet50v1": "resnet50_conv_kernel_weight", "alexnet": "alexnet_conv_kernel_weight", "vgg11": "vgg11_conv_kernel_weight",
                  "resnet18": "resnet18_conv_kernel_weight"}
CHECKPOINT_FILE = {"resnet50v2": "resnet50-11ad3fa6.pth", "resnet50v1": "resnet50-0676ba61.pth", "alexnet": "alexnet-owt-7be5be79.pth", "vgg11": "vgg11-8a719046.pth",
                   "resnet18": "resnet18-f37072fd.pth"}
SHORT = {"alexnet": {"features.stage1.conv": "conv1", "features.stage2.conv": "conv2", "features.stage3.conv": "conv3",
                     "features.stage4.conv": "conv4", "features.stage5.conv": "conv5", "readout.hidden1": "fc6",
                     "readout.hidden2": "fc7"},
         "vgg11": {"features.stage3.conv1": "conv3_1", "features.stage3.conv2": "conv3_2",
                   "features.stage4.conv1": "conv4_1", "features.stage4.conv2": "conv4_2",
                   "features.stage5.conv1": "conv5_1", "features.stage5.conv2": "conv5_2",
                   "readout.hidden1": "fc6", "readout.hidden2": "fc7"}}


class MaskRefused(ValueError):
    """A mask failed a check against its receipt or the weight layout."""


@dataclass(frozen=True)
class Layer:
    layer_id: str
    module_name: str
    kind: str                  # conv_kernel_weight | linear_weight | residual_stream_gate
    shape: tuple
    offset: int
    size: int


@dataclass(frozen=True)
class Layout:
    route_axis_hash: str
    route_count: int
    layers: tuple

    @property
    def packed_bytes(self) -> int:
        return (self.route_count + 7) // 8


def tree_layout(network: str) -> Layout:
    """The network's weight layout, from analysis/common/architectures.py (layer shapes from torchvision, checked
    against the engine's weight counts)."""
    from architectures import build_layout
    lay = build_layout(NETWORK_FAMILY[network])
    lay.validate()
    return Layout(lay.route_axis_hash, int(lay.route_count),
                  tuple(Layer(s.layer_id, s.module_name, s.kind, tuple(int(x) for x in s.shape), int(s.offset),
                              int(s.size)) for s in lay.layers))


def short_name(network: str, layer_id: str) -> str:
    """The paper's name of a layer (conv1..fc7 in AlexNet, conv3_1..fc7 in VGG-11); other layers keep their id."""
    return SHORT.get(network, {}).get(layer_id, layer_id)


def weight_signs(network: str, layout: Layout, checkpoint: Path) -> tuple[dict, str]:
    """For each weight layer, which weights are negative in the checkpoint (flattened in C order), and the checkpoint's
    sha256, which must start with the hash in its file name. A zero weight counts as positive, as in the engine."""
    import torch
    h = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    want = CHECKPOINT_FILE[network].split("-")[-1].split(".")[0]
    if not h.startswith(want):
        raise ValueError(f"{checkpoint}: sha256 {h[:8]} is not the {network} checkpoint ({want})")
    sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
    out = {}
    for s in layout.layers:
        if s.kind == "residual_stream_gate":
            continue
        w = sd[s.module_name + ".weight"].detach().cpu().numpy()
        if tuple(w.shape) != s.shape:
            raise ValueError(f"{s.layer_id}: checkpoint shape {w.shape} != layout {s.shape}")
        out[s.layer_id] = (w < 0).reshape(-1)
    return out, h


def _count(v, what: str) -> int:
    """A nonnegative integer count from a receipt field, or MaskRefused (format errors are refusals)."""
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0:
        raise MaskRefused(f"receipt {what} is not a nonnegative integer: {v!r}"[:200])
    return int(v)


def _obj(v, what: str) -> dict:
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise MaskRefused(f"receipt {what} is not an object")
    return v


def receipt_expectations(receipt: dict, layout: Layout) -> dict:
    """From the receipt: each layer's kept count (its size minus the weights removed by accepted pruning steps), the
    weights removed from each partition (sign group) and the partitions' sizes. A malformed receipt raises
    MaskRefused."""
    w = _obj(_obj(receipt, "").get("workflow"), "workflow")
    rows = w.get("decision_summaries")
    if rows is None:
        raise MaskRefused("receipt has no workflow.decision_summaries (per-layer retained counts)")
    if not isinstance(rows, list):
        raise MaskRefused("receipt workflow.decision_summaries is not a list")
    removed, removed_part = {}, {}
    for r in rows:
        r = _obj(r, "decision summary")
        if r.get("action") == "accept":
            lid, pid = r.get("layer_id"), str(r.get("partition_id"))
            if not isinstance(lid, str):
                raise MaskRefused(f"an accepted decision has no layer_id: {lid!r}"[:200])
            n = _count(r.get("removed_route_count"), f"removed_route_count ({lid})")
            removed[lid] = removed.get(lid, 0) + n
            removed_part.setdefault(lid, {})
            removed_part[lid][pid] = removed_part[lid].get(pid, 0) + n
    known = {s.layer_id for s in layout.layers}
    if set(removed) - known:
        raise MaskRefused(f"receipt removes routes in layers outside the layout: {sorted(set(removed) - known)}")
    scopes = {}
    hpm = _obj(_obj(w.get("hard_partition_measurement"), "hard_partition_measurement").get("prune_order_uncertainty"),
               "prune_order_uncertainty")
    srs = hpm.get("scope_receipts") or []
    if not isinstance(srs, list):
        raise MaskRefused("receipt scope_receipts is not a list")
    for s in srs:
        s = _obj(s, "scope receipt")
        if not isinstance(s.get("scope_id"), str):
            raise MaskRefused(f"a scope receipt has no scope_id: {s.get('scope_id')!r}"[:200])
        lid, _, p = s["scope_id"].partition(":partition:")
        scopes.setdefault(lid, {})[p] = _count(s.get("route_count"), f"scope route_count ({s['scope_id']})")
    return {"retained": {s.layer_id: s.size - removed.get(s.layer_id, 0) for s in layout.layers},
            "removed_by_partition": removed_part, "partition_sizes": scopes}


def product_sha(receipt: dict, name: str) -> str | None:
    """The receipt's recorded sha256 of a run product, or None when absent or malformed."""
    try:
        v = (((receipt.get("run_products") or {}).get("files") or {}).get(name) or {}).get("sha256")
    except AttributeError:
        return None
    return v if isinstance(v, str) and v else None


def load_verified_mask(mask_path: Path, receipt: dict, layout: Layout) -> np.ndarray:
    """The circuit's mask as one boolean per weight, or MaskRefused if it fails a check against its receipt (sha256,
    length, padding, layout hash, weight count, kept count of every layer)."""
    _obj(receipt, "")
    data = Path(mask_path).read_bytes()
    want = product_sha(receipt, "final_mask.packbits")
    if want is None:
        raise MaskRefused("receipt has no run_products sha256 for final_mask.packbits")
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise MaskRefused(f"sha256 {got[:12]} != receipt {want[:12]}")
    if len(data) != layout.packed_bytes:
        raise MaskRefused(f"{len(data)} bytes; the layout needs {layout.packed_bytes}")
    pad = layout.packed_bytes * 8 - layout.route_count
    if pad and data[-1] & ((1 << pad) - 1):
        raise MaskRefused("nonzero padding bits")
    for key in ("route_axis_hash", "route_count", "retained_route_count"):      # required, not optional
        if receipt.get(key) is None:
            raise MaskRefused(f"receipt has no {key}")
    if receipt["route_axis_hash"] != layout.route_axis_hash:
        raise MaskRefused(f"route axis {str(receipt['route_axis_hash'])[:12]} != layout {layout.route_axis_hash[:12]}")
    if _count(receipt["route_count"], "route_count") != layout.route_count:
        raise MaskRefused(f"route count {receipt['route_count']} != layout {layout.route_count}")
    rt = _count(receipt["retained_route_count"], "retained_route_count")
    bits = np.unpackbits(np.frombuffer(data, np.uint8), count=layout.route_count).view(bool)
    exp = receipt_expectations(receipt, layout)
    total = 0
    for s in layout.layers:
        n = int(np.count_nonzero(bits[s.offset:s.offset + s.size]))
        total += n
        if n != exp["retained"][s.layer_id]:
            raise MaskRefused(f"{s.layer_id}: popcount {n} != receipt retained {exp['retained'][s.layer_id]}")
    if total != rt:
        raise MaskRefused(f"total popcount {total} != retained_route_count {rt}")
    return bits


def load_blocks(taxonomy: Path) -> dict:
    """Each category's subgroup (block50) in the category tree table."""
    return {int(r["category"]): r["block50"] for r in csv.DictReader(open(taxonomy))}


def block_bootstrap(values, blocks, n_boot=2000, seed=0):
    """Mean over categories and its percentile interval (2,000 resamples), resampling whole subgroups (block50) with
    replacement; non-finite values are dropped first. Returns (mean, low, high, number of categories)."""
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    v, b = v[ok], np.asarray(blocks, dtype=object)[ok]
    if v.size == 0:
        return np.nan, np.nan, np.nan, 0
    labels, inv = np.unique(b.astype(str), return_inverse=True)
    G = labels.size
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, G, (n_boot, G))
    counts = np.bincount((np.arange(n_boot)[:, None] * G + picks).ravel(), minlength=n_boot * G).reshape(n_boot, G)
    W = counts[:, inv].astype(np.float64)
    draws = (W @ v) / W.sum(1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi), int(v.size)


def results_dir(root: Path) -> Path:
    """The folder that holds the circuits: ROOT/results if it exists, else ROOT."""
    root = Path(root)
    return root / "results" if (root / "results").is_dir() else root
