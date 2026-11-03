"""
GPU-resident streaming statistics for frontier analysis.

This module maintains Welford-style accumulators directly on the GPU so the
frontier scoring pipeline can snapshot lightweight diagnostics (PFI, NFI, BGI,
CDS, GOI, GCI, etc.) without repeatedly moving large tensors back to the host.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

EPS = 1e-8


class GPUStats:
    """Streaming per-row statistics computed directly on the GPU."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device
        self.initialised = False
        self.channels: Optional[int] = None

        # Per-feature accumulators
        self.S1: Optional[torch.Tensor] = None
        self.Spos: Optional[torch.Tensor] = None
        self.Sneg: Optional[torch.Tensor] = None
        self.Npos: Optional[torch.Tensor] = None
        self.Ns: Optional[torch.Tensor] = None
        self.sum: Optional[torch.Tensor] = None
        self.sum_sq: Optional[torch.Tensor] = None
        self.outer_sum: Optional[torch.Tensor] = None

        # Target-aware accumulators (populated on demand)
        self._target_sum: Optional[torch.Tensor] = None
        self._target_sum_sq: Optional[torch.Tensor] = None
        self._target_abs_sum: Optional[torch.Tensor] = None
        self._cross_sum: Optional[torch.Tensor] = None
        self._diff_abs: Optional[torch.Tensor] = None
        self._diff_sq: Optional[torch.Tensor] = None
        self._target_present: bool = False

        # Book-keeping
        self.total_samples: float = 0.0

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure(self, channels: int, device: torch.device) -> None:
        if self.initialised:
            return
        self.device = device
        self.channels = channels

        zeros = torch.zeros(channels, dtype=torch.float32, device=device)
        matrix = torch.zeros(channels, channels, dtype=torch.float32, device=device)

        self.S1 = zeros.clone()
        self.Spos = zeros.clone()
        self.Sneg = zeros.clone()
        self.Npos = zeros.clone()
        self.Ns = zeros.clone()
        self.sum = zeros.clone()
        self.sum_sq = zeros.clone()
        self.outer_sum = matrix

        # Target accumulators are initialised lazily when targets are observed.
        self._target_sum = zeros.clone()
        self._target_sum_sq = zeros.clone()
        self._target_abs_sum = zeros.clone()
        self._cross_sum = zeros.clone()
        self._diff_abs = zeros.clone()
        self._diff_sq = zeros.clone()

        self.initialised = True

    def _expand_target(self, target: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        """Broadcast target tensors to match ``(rows, cols)``."""

        if target.ndim == 1:
            if target.numel() == rows:
                target = target.view(rows, 1)
            elif target.numel() == cols:
                target = target.view(1, cols).expand(rows, -1)
            else:
                raise ValueError(
                    "1D target must match either the batch size or the number of channels."
                )

        if target.ndim != 2:
            raise ValueError("Target tensor must be 1D or 2D.")

        if target.size(0) != rows:
            if target.size(0) == 1:
                target = target.expand(rows, target.size(1))
            else:
                raise ValueError(
                    f"Target batch dimension mismatch: expected {rows}, got {target.size(0)}."
                )

        if target.size(1) != cols:
            if target.size(1) == 1:
                target = target.expand(target.size(0), cols)
            else:
                raise ValueError(
                    f"Target feature dimension mismatch: expected {cols}, got {target.size(1)}."
                )

        return target

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset all accumulators."""

        if not self.initialised or self.channels is None or self.device is None:
            self.initialised = False
            self.channels = None
            self.device = None
            self.total_samples = 0.0
            return

        zeros = torch.zeros(self.channels, dtype=torch.float32, device=self.device)
        matrix = torch.zeros(self.channels, self.channels, dtype=torch.float32, device=self.device)

        for attr in ("S1", "Spos", "Sneg", "Npos", "Ns", "sum", "sum_sq"):
            setattr(self, attr, zeros.clone())
        self.outer_sum = matrix

        for attr in ("_target_sum", "_target_sum_sq", "_target_abs_sum", "_cross_sum", "_diff_abs", "_diff_sq"):
            setattr(self, attr, zeros.clone())

        self._target_present = False
        self.total_samples = 0.0

    def update(self, phi_rows: torch.Tensor, targets: Optional[torch.Tensor] = None) -> None:
        """Update the streaming statistics with a new batch of rows."""

        if phi_rows is None:
            return
        batch = phi_rows.detach().to(dtype=torch.float32)
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        rows, cols = batch.shape
        device = batch.device
        self._ensure(cols, device)

        assert (
            self.S1 is not None
            and self.Spos is not None
            and self.Sneg is not None
            and self.Npos is not None
            and self.Ns is not None
            and self.sum is not None
            and self.sum_sq is not None
            and self.outer_sum is not None
        )

        with torch.no_grad():
            abs_batch = batch.abs().sum(dim=0)
            pos_batch = torch.relu(batch).sum(dim=0)
            neg_batch = torch.relu(-batch).sum(dim=0)
            pos_counts = (batch > 0).to(torch.float32).sum(dim=0)

            self.S1.add_(abs_batch)
            self.Spos.add_(pos_batch)
            self.Sneg.add_(neg_batch)
            self.Npos.add_(pos_counts)
            self.Ns.add_(batch.new_full((cols,), float(rows)))
            self.sum.add_(batch.sum(dim=0))
            self.sum_sq.add_((batch * batch).sum(dim=0))
            self.outer_sum.add_(batch.t().matmul(batch))
            self.total_samples += float(rows)

            if targets is not None:
                target = targets.detach().to(dtype=torch.float32, device=device)
                target = self._expand_target(target, rows, cols)
                self._target_present = True

                assert (
                    self._target_sum is not None
                    and self._target_sum_sq is not None
                    and self._target_abs_sum is not None
                    and self._cross_sum is not None
                    and self._diff_abs is not None
                    and self._diff_sq is not None
                )

                self._target_sum.add_(target.sum(dim=0))
                self._target_sum_sq.add_((target * target).sum(dim=0))
                self._target_abs_sum.add_(target.abs().sum(dim=0))
                self._cross_sum.add_((batch * target).sum(dim=0))
                diff = batch - target
                self._diff_abs.add_(diff.abs().sum(dim=0))
                self._diff_sq.add_((diff * diff).sum(dim=0))

    def export_indices(self) -> Dict[str, torch.Tensor]:
        """Return inexpensive per-feature indices (PFI/NFI/BGI/etc.)."""

        if not self.initialised or self.S1 is None:
            return {}

        S1 = self.S1.detach()
        Spos = self.Spos.detach()
        Sneg = self.Sneg.detach()
        Ns = self.Ns.detach()
        Npos = self.Npos.detach()

        L1 = torch.where(Ns > 0, S1 / (Ns + EPS), torch.zeros_like(S1))
        PFI = torch.where(S1 > 0, Spos / (S1 + EPS), torch.zeros_like(S1))
        NFI = torch.where(S1 > 0, Sneg / (S1 + EPS), torch.zeros_like(S1))
        proportion_pos = torch.where(Ns > 0, Npos / (Ns + EPS), torch.zeros_like(Npos))
        BGI = 1.0 - 2.0 * (proportion_pos - 0.5).abs()

        indices: Dict[str, torch.Tensor] = {
            "L1": L1.cpu(),
            "PFI": PFI.cpu(),
            "NFI": NFI.cpu(),
            "BGI": BGI.cpu(),
            "Ns": Ns.cpu(),
        }

        if self._target_present and self._diff_abs is not None:
            # Contribution deviation score per feature (mean absolute deviation).
            cds = torch.where(Ns > 0, self._diff_abs / (Ns + EPS), torch.zeros_like(Ns))
            indices["CDS"] = cds.cpu()

        return indices

    def export_analysis(self) -> Dict[str, object]:
        """Return a CPU snapshot of all accumulated statistics."""

        if not self.initialised or self.S1 is None:
            return {}

        Ns = self.Ns.detach()
        mean_abs = torch.where(Ns > 0, self.S1 / (Ns + EPS), torch.zeros_like(self.S1))

        analysis: Dict[str, object] = {
            "sum": self.sum.detach().cpu(),
            "sum_sq": self.sum_sq.detach().cpu(),
            "outer_sum": self.outer_sum.detach().cpu(),
            "mean_abs": mean_abs.cpu(),
            "S1": self.S1.detach().cpu(),
            "Ns": Ns.cpu(),
            "count": torch.tensor(self.total_samples, dtype=torch.float32),
            "total_samples": float(self.total_samples),
        }

        if self._target_present:
            assert (
                self._target_sum is not None
                and self._target_sum_sq is not None
                and self._target_abs_sum is not None
                and self._cross_sum is not None
                and self._diff_abs is not None
                and self._diff_sq is not None
            )
            analysis.update(
                {
                    "target_sum": self._target_sum.detach().cpu(),
                    "target_sum_sq": self._target_sum_sq.detach().cpu(),
                    "target_abs_sum": self._target_abs_sum.detach().cpu(),
                    "cross_sum": self._cross_sum.detach().cpu(),
                    "diff_abs": self._diff_abs.detach().cpu(),
                    "diff_sq": self._diff_sq.detach().cpu(),
                }
            )

        analysis["global"] = self._compute_global_summary()
        return analysis

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _compute_global_summary(self) -> Dict[str, float]:
        """Aggregate scalar diagnostics for convenience."""

        if self.S1 is None or self.sum is None or self.sum_sq is None or self.Ns is None:
            return {}

        Ns_total = self.Ns.sum()
        if Ns_total <= 0:
            return {}

        sum_phi = self.sum.sum()
        sum_phi_sq = self.sum_sq.sum()
        abs_phi = self.S1.sum()

        mean_phi = sum_phi / (Ns_total + EPS)
        var_phi = (sum_phi_sq / (Ns_total + EPS)) - mean_phi ** 2
        var_phi = torch.clamp(var_phi, min=0.0)

        summary: Dict[str, float] = {
            "count": float(Ns_total.item()),
            "mean_phi": float(mean_phi.item()),
            "var_phi": float(var_phi.item()),
        }

        if not self._target_present or self._target_sum is None:
            return summary

        sum_target = self._target_sum.sum()
        sum_target_sq = self._target_sum_sq.sum()
        abs_target = self._target_abs_sum.sum()
        sum_cross = self._cross_sum.sum()
        diff_abs = self._diff_abs.sum()
        diff_sq = self._diff_sq.sum()

        mean_target = sum_target / (Ns_total + EPS)
        var_target = (sum_target_sq / (Ns_total + EPS)) - mean_target ** 2
        var_target = torch.clamp(var_target, min=0.0)
        cov = (sum_cross / (Ns_total + EPS)) - mean_phi * mean_target

        corr = cov / (torch.sqrt(var_phi * var_target) + EPS)
        slope = cov / (var_target + EPS)
        scale_ratio = torch.sqrt(var_phi + EPS) / (torch.sqrt(var_target + EPS))
        rmse = torch.sqrt(diff_sq / (Ns_total + EPS))
        mean_abs_diff = diff_abs / (Ns_total + EPS)
        comm = 1.0 - diff_abs / (abs_phi + abs_target + EPS)
        orientation = torch.abs(corr)

        summary.update(
            {
                "mean_target": float(mean_target.item()),
                "var_target": float(var_target.item()),
                "covariance": float(cov.item()),
                "corr": float(corr.clamp(-1.0, 1.0).item()),
                "scale_ratio": float(scale_ratio.item()),
                "slope": float(slope.item()),
                "rmse": float(rmse.item()),
                "mean_abs_diff": float(mean_abs_diff.item()),
                "CDS": float(mean_abs_diff.item()),
                "GOI": float(orientation.item()),  # Global orientation index
                "GCI": float(comm.clamp(min=-1.0, max=1.0).item()),  # Global commensurability index
            }
        )
        return summary


__all__ = ["GPUStats"]
