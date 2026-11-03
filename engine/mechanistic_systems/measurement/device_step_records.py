"""Accepted-step reductions and empirical-Bayes shrinkage on torch tensors.

Only reduced matrices and tracker surfaces survive an accepted step. They
stay on the device until the group's walk finishes, then leave in dtype
blocks for the bulk record writer. No operation below reads a device value
on the host.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def upload_metadata(values: Any, device: Any, *, dtype: Any) -> Any:
    """Stream-ordered metadata upload through pinned memory on CUDA."""

    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA metadata upload requires an available CUDA device")
    if torch.is_tensor(values):
        if values.device.type != "cpu":
            raise ValueError("metadata upload requires host values")
        host = values.to(dtype=dtype)
    else:
        host = torch.as_tensor(np.asarray(values), dtype=dtype, device="cpu")
    if target.type == "cuda":
        host = host.pin_memory()
    return host.to(device=target, non_blocking=True)


class DeviceClassReducer:
    """Float64 class means in fixed draw-order row windows.

    Indexed sums accumulate each row into its stratum. Fixed windows bound
    the row frame without constructing a strata-by-row one-hot matrix.
    Each row contributes only to its stratum, including NaN and infinity.
    """

    def __init__(
        self, labels: Any, class_ids: Sequence[int], *, width: int,
        covered: Any | None, row_budget_bytes: int,
    ) -> None:
        if width <= 0 or len(class_ids) == 0 or row_budget_bytes <= 0:
            raise ValueError("class reduction requires positive width, strata and row budget")
        self.labels = labels.reshape(-1)
        self.class_ids = upload_metadata(class_ids, labels.device, dtype=torch.int64)
        self.width = int(width)
        self.strata = len(class_ids)
        self.row_count = int(self.labels.numel())
        self.covered = covered
        if covered is not None and tuple(covered.shape) != (self.row_count,):
            raise ValueError("class reduction coverage differs from its row axis")
        sorted_classes, order = torch.sort(self.class_ids)
        positions = torch.searchsorted(sorted_classes, self.labels).clamp_max(self.strata - 1)
        self.ordinals = order[positions]
        from mechanistic_systems.measurement.device_deferred import DEVICE_FAULTS

        DEVICE_FAULTS.note(
            sorted_classes[positions].ne(self.labels).any(),
            lambda: ValueError("step category rows carry labels outside the declared class strata"),
        )
        rows_per_window = max(1, int(row_budget_bytes) // (8 * width))
        self.chunks = tuple(
            (start, min(start + rows_per_window, self.row_count))
            for start in range(0, self.row_count, rows_per_window)
        )
        self.counts = torch.zeros(self.strata, device=labels.device, dtype=torch.float64)
        self.covered_counts = torch.zeros_like(self.counts)
        for start, stop in self.chunks:
            ordinals = self.ordinals[start:stop]
            self.counts.index_add_(0, ordinals, torch.ones(stop - start, device=labels.device, dtype=torch.float64))
            if covered is not None:
                self.covered_counts.index_add_(0, ordinals, covered[start:stop].to(torch.float64))

    def means(self, frame: Any, *, masked: bool = False) -> Any:
        if tuple(frame.shape) != (self.row_count, self.width):
            raise ValueError("class reduction rows differ from the declared axes")
        if masked and self.covered is None:
            raise ValueError("masked class reduction requires row coverage")
        sums = torch.zeros(
            (self.strata, self.width), device=self.labels.device, dtype=torch.float64
        )
        for start, stop in self.chunks:
            rows = frame.window(start, stop)
            ordinals = self.ordinals[start:stop]
            if masked:
                rows = torch.where(self.covered[start:stop, None], rows, 0.0)
            sums.index_add_(0, ordinals, rows)
        counts = self.covered_counts if masked else self.counts
        return sums / counts[:, None]


def hierarchical_shrinkage_device(
    *, drift: Any, precision: Any, clusters: Any, cluster_count: int,
) -> dict[str, Any]:
    """The two DerSimonian-Laird passes and nested posteriors, in float64."""

    d, w = drift.to(torch.float64), precision.to(torch.float64)
    if d.ndim != 2 or d.shape != w.shape or clusters.shape != (d.shape[0],):
        raise ValueError("hierarchical shrinkage axes are misaligned")
    c_count = max(1, int(cluster_count))
    eligible = (clusters[:, None] >= 0) & (w > 0) & torch.isfinite(w) & torch.isfinite(d)
    we = torch.where(eligible, w, 0.0)
    de = torch.where(eligible, d, 0.0)
    # Each coordinate can have a different eligible set. Assign an
    # ineligible category to cluster zero with exactly zero weight.
    cl = torch.where(eligible, clusters[:, None], 0)

    def segment_sum(values: Any) -> Any:
        result = torch.zeros((c_count, d.shape[1]), dtype=d.dtype, device=d.device)
        return result.scatter_add_(0, cl, values)

    s1, s1d, s2 = segment_sum(we), segment_sum(we * de), segment_sum(we.square())
    members = segment_sum(eligible.to(d.dtype))
    occupied = s1 > 0
    safe_s1 = torch.where(occupied, s1, 1.0)
    mu_c_fe = s1d / safe_s1
    residual = de - torch.gather(mu_c_fe, 0, cl)
    q_total = segment_sum(we * residual.square()).sum(dim=0)
    df_total = torch.where(occupied, (members - 1).clamp_min(0), 0.0).sum(dim=0)
    denominator = (s1 - s2 / safe_s1).sum(dim=0)
    tau_cat = torch.where(
        denominator > 0, ((q_total - df_total) / torch.where(denominator > 0, denominator, 1.0)).clamp_min(0), 0.0,
    )
    w_star = we / (1 + tau_cat * we)
    pc = segment_sum(w_star)
    cluster_occupied = pc > 0
    gc = segment_sum(w_star * de) / torch.where(cluster_occupied, pc, 1.0)
    occupied_count = cluster_occupied.sum(dim=0)
    p_sum = pc.sum(dim=0)
    safe_p = torch.where(p_sum > 0, p_sum, 1.0)
    mu_fe = (pc * gc).sum(dim=0) / safe_p
    q = (pc * (gc - mu_fe).square()).sum(dim=0)
    dl_den = p_sum - pc.square().sum(dim=0) / safe_p
    tau_cl = torch.where(
        (occupied_count >= 2) & (dl_den > 0),
        ((q - (occupied_count - 1)) / torch.where(dl_den > 0, dl_den, 1.0)).clamp_min(0), 0.0,
    )
    wc = pc / (1 + tau_cl * pc)
    wc_sum = wc.sum(dim=0)
    mu = (wc * gc).sum(dim=0) / torch.where(wc_sum > 0, wc_sum, 1.0)
    bc = tau_cl * pc / (1 + tau_cl * pc)
    mc = bc * gc + (1 - bc) * mu
    bj = tau_cat * we / (1 + tau_cat * we)
    parent = torch.where(
        clusters[:, None] >= 0,
        mc[clusters.clamp_min(0)], mu,
    )
    posterior = bj * d + (1 - bj) * parent
    posterior = torch.where(torch.isfinite(posterior), posterior, mu)
    # With no eligible category the historical estimator retains each
    # finite drift, and returns zero for a non-finite one.
    posterior = torch.where(
        eligible.any(dim=0), posterior, torch.where(torch.isfinite(d), d, 0.0),
    )
    return {
        "posterior_drift": posterior, "own_data_weight": bj,
        "global_drift": mu, "tau2_category": tau_cat, "tau2_cluster": tau_cl,
        "cluster_raw_mean": gc, "cluster_posterior_mean": mc,
    }


def fetch_record(fields: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Fetch one completed record in dtype blocks, at the group's tail."""

    groups: dict[Any, list[tuple[str, Any]]] = defaultdict(list)
    for name, value in fields.items():
        groups[value.dtype].append((name, value))
    result = {}
    for entries in groups.values():
        packed = torch.cat([value.reshape(-1) for _, value in entries]).to(device="cpu").numpy()
        cursor = 0
        for name, value in entries:
            stop = cursor + int(value.numel())
            result[name] = packed[cursor:stop].reshape(tuple(value.shape))
            cursor = stop
    return result


__all__ = ["DeviceClassReducer", "fetch_record", "hierarchical_shrinkage_device", "upload_metadata"]
