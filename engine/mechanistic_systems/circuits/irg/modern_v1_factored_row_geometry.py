"""Exact Gram geometry of the factored literal-J rows.

The frequent-directions sketch of the modern-v1 response geometry
(``modern_v1_stratified_fd_gpu.py``) approximates an object that the factored
rows make exactly computable: the ``n`` sampled rows span a space of dimension
at most ``n``, so the law-of-total-covariance stack

    s_x     = sqrt(w_s / n_s) (row(x) - m_s)      for every context x of stratum s,
    s_{n+s} = sqrt(w_s)       (m_s - m)            for every stratum s,
    m       = sum_s w_s m_s,

is a linear combination ``S = Aᵀ R`` of the ``n`` primitive rows, its Gram is
``Aᵀ K A`` with ``K`` the primitive Gram, and its thin SVD is
read from that ``(n+S) x (n+S)`` matrix.  The KKT quadratic budget
``Q(z) = ⟨m, z⟩² + ‖S z‖²`` is then an exact ``(n+S+1)``-term sum of bilinear
forms; the FD residual diagonal and spectral bound become floating-point
enclosures, and the ``(8, 16, 32)`` rank menu becomes the numerical rank of the
stack -- a property of the sample, never a registered numeral.

Two presentations are offered to the KKT consumers: the dense arrays the
existing ``V1RelaxedKKTProblem`` / Phase-A problems accept
(``low_rank_factor = [m; H]`` with ``H = Uᵀ S``, ``(1 + r) x N`` fp64,
forecast-gated because ``r`` may be ``n + S``), and a matrix-free operator
(``forward``/``adjoint``) with the same quadratic form and no route-scale
factor.  The frontier solver does not admit the operator.

Nothing here certifies F/Q/R, opens C/K/O, authorizes a mask commit, or
carries terminal, runtime, or Pareto authority.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, NoReturn

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .modern_v1_factored_row_capture import (
    DEFAULT_COMBINE_WORKSPACE_BYTES,
    V1FactoredRowPacket,
    V1FactoredRowRefusal,
    V1FactoredRowRefusalCode,
)
from .modern_v1_sampled_dual_chart_row_custody import V1SampledDualChartRowRole

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


FACTORED_ROW_EXACT_GRAM_GEOMETRY_SCHEMA: Final = (
    "partition_free_v1_factored_row_exact_gram_geometry_v2"
)
FACTORED_ROW_EXACT_GEOMETRY_KKT_DENSE_ADAPTER_SCHEMA: Final = (
    "partition_free_v1_factored_row_exact_geometry_kkt_dense_adapter_v2"
)
FACTORED_ROW_EXACT_GEOMETRY_LAW_ID: Final = (
    "exact_rank_le_n_gram_svd_zero_fd_residual_roundoff_enclosure_v1"
)
FACTORED_ROW_EXACT_GEOMETRY_STACK_ID: Final = (
    "law_of_total_covariance_stack_sqrt_ws_over_ns_centered_rows_plus_sqrt_ws_between_stratum_means_v1"
)
FACTORED_ROW_EXACT_GEOMETRY_AUTHORITY_SCOPE: Final = (
    "kkt_proposal_input_geometry_only_no_D_stability_C_K_O_mask_commit_terminal_or_"
    "pareto_authority_v1"
)
_PREFIX: Final = "partition-free-v1-factored-row-exact-gram-geometry-"
_ADAPTER_PREFIX: Final = "partition-free-v1-factored-row-exact-geometry-kkt-dense-adapter-"
# Same convention as the estate's Gram-FD PSD tolerance (device_route_moments.py).
_PSD_TOLERANCE_MULTIPLIER: Final = 64.0
_F64_BYTES: Final = 8


def _refuse(code: V1FactoredRowRefusalCode, message: str) -> NoReturn:
    raise V1FactoredRowRefusal(code, message)


def _torch() -> Any:
    if torch is None:  # pragma: no cover
        _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "exact geometry requires PyTorch")
    return torch


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, f"{name} must be a nonempty string")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        _refuse(V1FactoredRowRefusalCode.NONFINITE, f"{name} must be finite and nonnegative")
    return number


@dataclass(frozen=True)
class V1FactoredRowExactGeometryReceipt:
    """Array-free receipt of one exact Gram geometry (V2 family)."""

    physical_route_axis_hash: str
    route_id_axis_hash: str
    route_count: int
    role: str
    stratum_names: tuple[str, ...]
    stratum_weights: tuple[float, ...]
    stratum_context_counts: tuple[int, ...]
    context_count: int
    packet_receipt_hashes: tuple[str, ...]
    packet_stratum_labels: tuple[str, ...]
    gate_hash: str
    stacked_row_count: int
    retained_rank: int
    retained_eigenvalues: tuple[float, ...]
    discarded_tail_eigenvalue_sum: float
    residual_spectral_bound: float
    psd_tolerance: float
    gram_frobenius_norm: float
    mean_component_l2_norm: float
    stack_id: str = FACTORED_ROW_EXACT_GEOMETRY_STACK_ID
    law_id: str = FACTORED_ROW_EXACT_GEOMETRY_LAW_ID
    fd_sketch_used: bool = False
    fd_shrinkage: float = 0.0
    rank_menu_registered: bool = False
    route_scale_d2h_bytes: int = 0
    D_stability_authority: bool = False
    C_authority: bool = False
    calibration_authority: bool = False
    oracle_authority: bool = False
    mask_commit_authority: bool = False
    terminal_authority: bool = False
    pareto_authority: bool = False
    authority_scope: str = FACTORED_ROW_EXACT_GEOMETRY_AUTHORITY_SCOPE
    schema: str = FACTORED_ROW_EXACT_GRAM_GEOMETRY_SCHEMA
    _sealed_hash: str = field(init=False, default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("physical_route_axis_hash", "route_id_axis_hash", "role", "gate_hash"):
            _identity(getattr(self, name), name=name)
        if self.physical_route_axis_hash == self.route_id_axis_hash:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "physical and ordered route axes must remain distinct")
        if self.role not in {item.value for item in V1SampledDualChartRowRole}:
            _refuse(V1FactoredRowRefusalCode.ROLE_MISMATCH, "geometry role must be a dual-chart row role")
        names = tuple(_identity(v, name="stratum name") for v in self.stratum_names)
        weights = tuple(_finite_nonnegative(v, name="stratum weight") for v in self.stratum_weights)
        counts = tuple(int(v) for v in self.stratum_context_counts)
        if not names or len(names) != len(set(names)) or len(weights) != len(names) or len(counts) != len(names):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "stratum axes must align")
        if any(c <= 0 for c in counts) or sum(counts) != int(self.context_count):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "stratum context counts do not close the sample")
        if abs(sum(weights) - 1.0) > 1e-12:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "stratum weights must sum to one")
        object.__setattr__(self, "stratum_names", names)
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "stratum_context_counts", counts)
        hashes = tuple(_identity(v, name="packet receipt hash") for v in self.packet_receipt_hashes)
        labels = tuple(_identity(v, name="packet stratum label") for v in self.packet_stratum_labels)
        if not hashes or len(hashes) != len(labels) or len(set(hashes)) != len(hashes) or any(l not in names for l in labels):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "packet ledger must be unique and name registered strata")
        object.__setattr__(self, "packet_receipt_hashes", hashes)
        object.__setattr__(self, "packet_stratum_labels", labels)
        if int(self.route_count) <= 0 or int(self.context_count) <= 0:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "route and context counts must be positive")
        if int(self.stacked_row_count) != int(self.context_count) + len(names):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "stack must hold n + S rows")
        rank = int(self.retained_rank)
        eigen = tuple(_finite_nonnegative(v, name="retained eigenvalue") for v in self.retained_eigenvalues)
        if rank < 0 or rank > self.stacked_row_count or len(eigen) != rank or any(
            eigen[i] < eigen[i + 1] for i in range(len(eigen) - 1)
        ):
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "retained eigenvalues must be descending and match the rank")
        object.__setattr__(self, "retained_eigenvalues", eigen)
        for name in (
            "discarded_tail_eigenvalue_sum",
            "residual_spectral_bound",
            "psd_tolerance",
            "gram_frobenius_norm",
            "mean_component_l2_norm",
            "fd_shrinkage",
        ):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name=name))
        if self.residual_spectral_bound > self.psd_tolerance:
            _refuse(
                V1FactoredRowRefusalCode.STALE_IDENTITY,
                "exact geometry residual exceeds its roundoff enclosure; it is not exact",
            )
        if self.fd_sketch_used is not False or self.fd_shrinkage != 0.0 or self.rank_menu_registered is not False:
            _refuse(V1FactoredRowRefusalCode.AUTHORITY_VIOLATION, "exact geometry uses no FD sketch and no rank menu")
        if self.route_scale_d2h_bytes != 0:
            _refuse(V1FactoredRowRefusalCode.AUTHORITY_VIOLATION, "exact geometry performs no route-scale D2H")
        if self.schema != FACTORED_ROW_EXACT_GRAM_GEOMETRY_SCHEMA or self.law_id != FACTORED_ROW_EXACT_GEOMETRY_LAW_ID or self.stack_id != FACTORED_ROW_EXACT_GEOMETRY_STACK_ID or self.authority_scope != FACTORED_ROW_EXACT_GEOMETRY_AUTHORITY_SCOPE:
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "exact geometry receipt schema or law drifted")
        for name in (
            "D_stability_authority",
            "C_authority",
            "calibration_authority",
            "oracle_authority",
            "mask_commit_authority",
            "terminal_authority",
            "pareto_authority",
        ):
            if getattr(self, name) is not False:
                _refuse(V1FactoredRowRefusalCode.AUTHORITY_VIOLATION, f"exact geometry receipts carry no {name}")
        object.__setattr__(self, "_sealed_hash", self._compute_hash())

    def _hash_payload(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "physical_route_axis_hash",
                "route_id_axis_hash",
                "route_count",
                "role",
                "stratum_names",
                "stratum_weights",
                "stratum_context_counts",
                "context_count",
                "packet_receipt_hashes",
                "packet_stratum_labels",
                "gate_hash",
                "stacked_row_count",
                "retained_rank",
                "retained_eigenvalues",
                "discarded_tail_eigenvalue_sum",
                "residual_spectral_bound",
                "psd_tolerance",
                "gram_frobenius_norm",
                "mean_component_l2_norm",
                "stack_id",
                "law_id",
                "fd_sketch_used",
                "fd_shrinkage",
                "rank_menu_registered",
                "route_scale_d2h_bytes",
                "D_stability_authority",
                "C_authority",
                "calibration_authority",
                "oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_authority",
                "authority_scope",
                "schema",
            )
        }

    def _compute_hash(self) -> str:
        return content_hash(self._hash_payload(), prefix=_PREFIX)

    @property
    def receipt_hash(self) -> str:
        return self._sealed_hash

    def assert_integrity(self) -> str:
        if self._compute_hash() != self._sealed_hash:
            _refuse(V1FactoredRowRefusalCode.MUTATED_CUSTODY, "exact geometry receipt hash does not replay")
        return self._sealed_hash

    def to_json_payload(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload["receipt_hash"] = self.receipt_hash
        return payload


def _gate_hash(gate64: Any) -> str:
    return content_hash(
        np.packbits(gate64.detach().cpu().numpy() > 0.5, bitorder="big").tobytes(),
        prefix="partition-free-v1-factored-row-gate-",
    )


class V1FactoredRowExactGeometry:
    """Process-local exact geometry over live factored packets (nonserializable)."""

    __slots__ = (
        "_receipt",
        "_packets",
        "_role",
        "_gate64",
        "_context_weights",
        "_combination",
        "_eigenvectors",
        "_coefficients",
        "_mean",
        "_evicted",
    )

    def __init__(
        self,
        *,
        receipt: V1FactoredRowExactGeometryReceipt,
        packets: Sequence[V1FactoredRowPacket],
        role: V1SampledDualChartRowRole,
        gate64: Any,
        context_weights: Any,
        combination: Any,
        eigenvectors: Any,
        mean_component: Any,
    ) -> None:
        receipt.assert_integrity()
        self._receipt = receipt
        self._packets = tuple(packets)
        self._role = role
        self._gate64 = gate64
        self._context_weights = context_weights
        self._combination = combination
        self._eigenvectors = eigenvectors
        self._coefficients = combination @ eigenvectors
        self._mean = mean_component
        self._evicted = False

    def __reduce_ex__(self, _protocol: int) -> NoReturn:
        raise TypeError("exact geometry is nonserializable; persist the receipt only")

    def _assert_live(self) -> None:
        if self._evicted:
            _refuse(V1FactoredRowRefusalCode.EVICTED, "exact geometry was evicted")
        for packet in self._packets:
            packet.receipt  # raises EVICTED if a packet is gone

    @property
    def receipt(self) -> V1FactoredRowExactGeometryReceipt:
        return self._receipt

    @property
    def role(self) -> V1SampledDualChartRowRole:
        return self._role

    @property
    def retained_rank(self) -> int:
        return self._receipt.retained_rank

    @property
    def route_count(self) -> int:
        return self._receipt.route_count

    @property
    def mean_component(self) -> Any:
        """Route-scale fp64 mean row ``m = Σ_s w_s m_s`` (one vector)."""

        self._assert_live()
        return self._mean

    # -- bilinear primitives ----------------------------------------------------------

    def _primitive_inner(self, directions: Any) -> Any:
        """``⟨row(x), V_k⟩`` for every sampled context, fp64 ``[n, k]``."""

        parts = [packet.inner(self._role, directions, gate=self._gate64) for packet in self._packets]
        return _torch().cat(parts, dim=0)

    def combine_workspace_bytes(self, k: int) -> int:
        """Peak transient bytes of one ``combine_rows`` over ``k`` columns.

        The linear einsum intermediate ``[k_chunk, out, in]`` fp64 is bounded by
        ``DEFAULT_COMBINE_WORKSPACE_BYTES``; below that bound the whole ``[k, out, in]``
        block is the transient.
        """

        largest = 0
        for packet in self._packets:
            for layer in packet.layers:
                if layer.reducer_kind == "linear_weight":
                    out_units, in_units = tuple(int(v) for v in layer.weight.shape)
                    largest = max(largest, int(k) * out_units * in_units * _F64_BYTES)
        return min(int(DEFAULT_COMBINE_WORKSPACE_BYTES), largest)

    def _combine(self, coefficients: Any, *, out: Any | None = None) -> Any:
        """``Σ_x C[x, k] row(x)`` across packets, fp64 ``[k, N]``.

        Accumulates in place into one ``[k, N]`` tensor (``out`` when given,
        which must be zero) so the sum over packets never holds two
        route-scale tensors; each packet's linear einsum is rank-chunked by
        the packet's combine workspace.
        """

        torch_module = _torch()
        C = torch_module.as_tensor(coefficients, dtype=torch_module.float64, device=self._mean.device)
        if C.ndim == 1:
            C = C.unsqueeze(1)
        result = (
            torch_module.zeros((int(C.shape[1]), self.route_count), dtype=torch_module.float64, device=self._mean.device)
            if out is None
            else out
        )
        start = 0
        for packet in self._packets:
            stop = start + packet.context_count
            packet.combine_rows(self._role, C[start:stop], gate=self._gate64, out=result)
            start = stop
        return result

    def quadratic_form(self, z: Any) -> Any:
        """Exact ``Q(z) = ⟨m, z⟩² + ‖S z‖²`` for directions ``z`` (``[k, N]`` or ``[N]``), fp64 ``[k]``."""

        self._assert_live()
        torch_module = _torch()
        Z = torch_module.as_tensor(z, dtype=torch_module.float64, device=self._mean.device)
        if Z.ndim == 1:
            Z = Z.unsqueeze(0)
        p = self._primitive_inner(Z)  # [n, k]
        mean_part = (self._context_weights.unsqueeze(1) * p).sum(dim=0)  # [k]
        stacked = self._combination.T @ p  # [n+S, k]
        return mean_part**2 + (stacked**2).sum(dim=0)

    def forward(self, z: Any) -> Any:
        """Matrix-free ``[m; H] z`` : fp64 ``[1 + r, k]`` for directions ``[k, N]``."""

        self._assert_live()
        torch_module = _torch()
        Z = torch_module.as_tensor(z, dtype=torch_module.float64, device=self._mean.device)
        if Z.ndim == 1:
            Z = Z.unsqueeze(0)
        p = self._primitive_inner(Z)
        mean_part = (self._context_weights.unsqueeze(1) * p).sum(dim=0, keepdim=True)
        low = self._coefficients.T @ p  # [r, k]
        return torch_module.cat((mean_part, low), dim=0)

    def adjoint(self, y: Any) -> Any:
        """Matrix-free ``[m; H]ᵀ y`` : route-scale fp64 ``[N]`` for ``y`` of length ``1 + r``."""

        self._assert_live()
        torch_module = _torch()
        Y = torch_module.as_tensor(y, dtype=torch_module.float64, device=self._mean.device).reshape(-1)
        if int(Y.numel()) != 1 + self.retained_rank:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "adjoint input must have length 1 + retained rank")
        coefficients = Y[0] * self._context_weights + self._coefficients @ Y[1:]
        return self._combine(coefficients)[0]

    def dense_factor(self, *, budget_bytes: int) -> Any:
        """``[m; H]`` as a dense fp64 ``[1 + r, N]`` tensor, forecast-gated."""

        self._assert_live()
        torch_module = _torch()
        rows = 1 + self.retained_rank
        factor_bytes = rows * self.route_count * _F64_BYTES
        # Peak device custody while assembling: the factor itself plus the bounded
        # linear einsum workspace; the factor is filled in place (no second copy).
        workspace = self.combine_workspace_bytes(self.retained_rank)
        needed = factor_bytes + workspace
        budget = int(budget_bytes)
        if budget <= 0 or needed > budget:
            _refuse(
                V1FactoredRowRefusalCode.DENSE_FACTOR_NOT_ADMITTED,
                f"dense factor needs {needed} bytes ({rows} x {self.route_count} fp64 + "
                f"{workspace} workspace); budget {budget}",
            )
        factor = torch_module.zeros((rows, self.route_count), dtype=torch_module.float64, device=self._mean.device)
        factor[0].copy_(self._mean)
        if self.retained_rank > 0:
            self._combine(self._coefficients, out=factor[1:])
        return factor

    def residual_diagonal(self) -> Any:
        """``clamp(diag(SᵀS) - Σ_k H_k², 0)`` per route: a roundoff enclosure when ``r`` is full."""

        self._assert_live()
        torch_module = _torch()
        A = self._combination
        C = self._coefficients
        exact_diag = torch_module.zeros(self.route_count, dtype=torch_module.float64, device=self._mean.device)
        retained_diag = torch_module.zeros_like(exact_diag)
        # diag(SᵀS) = Σ_{a,b} (A Aᵀ)_{ab} row(a) ∘ row(b); diag(HᵀH) likewise with C Cᵀ.
        # Cross-packet pairs need both packets; assemble per (packet, packet) block.
        P_exact = A @ A.T
        P_kept = C @ C.T
        start_a = 0
        for packet_a in self._packets:
            stop_a = start_a + packet_a.context_count
            rows_a = None  # packet_a's own rows, materialized once per packet_a
            start_b = 0
            for packet_b in self._packets:
                stop_b = start_b + packet_b.context_count
                if packet_a is packet_b:
                    exact_diag += packet_a.pair_weighted_square_diagonal(
                        self._role, P_exact[start_a:stop_a, start_b:stop_b], gate=self._gate64
                    )
                    retained_diag += packet_a.pair_weighted_square_diagonal(
                        self._role, P_kept[start_a:stop_a, start_b:stop_b], gate=self._gate64
                    )
                else:
                    # cross-packet Hadamard pairs: Σ_{a∈A,b∈B} P_ab row(a)∘row(b) via combine_rows
                    if rows_a is None:
                        rows_a = packet_a.combine_rows(
                            self._role, torch_module.eye(packet_a.context_count, dtype=torch_module.float64, device=self._mean.device), gate=self._gate64
                        )
                    rows_b_exact = packet_b.combine_rows(
                        self._role, P_exact[start_a:stop_a, start_b:stop_b].T, gate=self._gate64
                    )  # [n_a, N]: Σ_b P_ab row(b)
                    exact_diag += (rows_a * rows_b_exact).sum(dim=0)
                    del rows_b_exact
                    rows_b_kept = packet_b.combine_rows(
                        self._role, P_kept[start_a:stop_a, start_b:stop_b].T, gate=self._gate64
                    )
                    retained_diag += (rows_a * rows_b_kept).sum(dim=0)
                    del rows_b_kept
                start_b = stop_b
            del rows_a
            start_a = stop_a
        return torch_module.clamp(exact_diag - retained_diag, min=0.0)

    def evict(self) -> None:
        self._evicted = True
        self._packets = ()
        self._mean = None


def compute_v1_factored_row_exact_geometry(
    *,
    packets: Sequence[V1FactoredRowPacket],
    packet_stratum_labels: Sequence[str],
    stratum_names: Sequence[str],
    stratum_weights: Sequence[float],
    role: V1SampledDualChartRowRole,
    gate: Any,
) -> V1FactoredRowExactGeometry:
    """Build the exact total-covariance geometry from live factored packets.

    Each packet is one same-stratum context packet (the legacy session's law).
    Per-context weights inside a stratum are uniform (``1 / n_s``), as in the
    legacy FD children fed with ``weights=None``; the design stratum weights
    combine strata by the law of total covariance.
    """

    torch_module = _torch()
    packet_tuple = tuple(packets)
    labels = tuple(packet_stratum_labels)
    names = tuple(stratum_names)
    weights = tuple(float(w) for w in stratum_weights)
    if not packet_tuple or any(type(p) is not V1FactoredRowPacket for p in packet_tuple):
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "exact geometry requires factored row packets")
    if len(labels) != len(packet_tuple):
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "one stratum label per packet is required")
    if len(names) != len(weights) or len(set(names)) != len(names):
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "stratum names and weights must align and be unique")
    if type(role) is not V1SampledDualChartRowRole:
        _refuse(V1FactoredRowRefusalCode.ROLE_MISMATCH, "geometry role must be exact")
    first = packet_tuple[0].receipt
    for packet in packet_tuple:
        receipt = packet.receipt
        if (
            receipt.physical_route_axis_hash != first.physical_route_axis_hash
            or receipt.route_id_axis_hash != first.route_id_axis_hash
            or receipt.route_count != first.route_count
            or receipt.chart_identity_hash != first.chart_identity_hash
            or receipt.current_mask_hash != first.current_mask_hash
            or receipt.corrected_a_stream_identity_hash != first.corrected_a_stream_identity_hash
            or receipt.literal_j_stream_identity_hash != first.literal_j_stream_identity_hash
        ):
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "packets belong to different charts or axes")
    if len({p.receipt.receipt_hash for p in packet_tuple}) != len(packet_tuple):
        _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "duplicate packet in the geometry")
    counts = {name: 0 for name in names}
    for label in labels:
        if label not in counts:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, f"packet stratum {label!r} is not registered")
    for packet, label in zip(packet_tuple, labels, strict=True):
        counts[label] += packet.context_count
    if any(counts[name] <= 0 for name in names):
        _refuse(V1FactoredRowRefusalCode.MISSING_INPUT, "every registered stratum needs at least one context")
    gate64 = packet_tuple[0].gate64(gate)
    device = gate64.device
    n = sum(p.context_count for p in packet_tuple)
    S = len(names)
    # per-context stratum index and weights w_s / n_s
    stratum_of_context = []
    for packet, label in zip(packet_tuple, labels, strict=True):
        stratum_of_context.extend([names.index(label)] * packet.context_count)
    stratum_index = torch_module.as_tensor(stratum_of_context, dtype=torch_module.int64, device=device)
    n_s = torch_module.as_tensor([counts[name] for name in names], dtype=torch_module.float64, device=device)
    w_s = torch_module.as_tensor(weights, dtype=torch_module.float64, device=device)
    context_weights = w_s[stratum_index] / n_s[stratum_index]  # coefficient of row(x) in m
    # Combination matrix A (n x (n+S)): stacked rows s = Aᵀ R.
    A = torch_module.zeros((n, n + S), dtype=torch_module.float64, device=device)
    for x in range(n):
        s = int(stratum_index[x])
        scale = math.sqrt(weights[s] / float(n_s[s]))
        A[x, x] += scale
        same = stratum_index == s
        A[same, x] -= scale / float(n_s[s])
    for s in range(S):
        same = stratum_index == s
        A[same, n + s] += math.sqrt(weights[s]) / float(n_s[s])
        A[:, n + s] -= math.sqrt(weights[s]) * context_weights
    # Primitive Gram K (n x n) across packets.
    K = torch_module.zeros((n, n), dtype=torch_module.float64, device=device)
    start_a = 0
    for packet_a in packet_tuple:
        stop_a = start_a + packet_a.context_count
        start_b = 0
        for packet_b in packet_tuple:
            stop_b = start_b + packet_b.context_count
            K[start_a:stop_a, start_b:stop_b] = packet_a.gram(role, None if packet_b is packet_a else packet_b, gate=gate64)
            start_b = stop_b
        start_a = stop_a
    K = 0.5 * (K + K.T)
    gram_stack = A.T @ K @ A
    gram_stack = 0.5 * (gram_stack + gram_stack.T)
    eigenvalues, vectors = torch_module.linalg.eigh(gram_stack)
    if not bool(torch_module.isfinite(eigenvalues).all().item()) or not bool(torch_module.isfinite(vectors).all().item()):
        _refuse(V1FactoredRowRefusalCode.NONFINITE, "stacked Gram eigendecomposition is nonfinite")
    lambda_max = float(eigenvalues.abs().max().item()) if eigenvalues.numel() else 0.0
    tolerance = (
        _PSD_TOLERANCE_MULTIPLIER
        * float(torch_module.finfo(torch_module.float64).eps)
        * max(1, n + S, int(first.route_count))
        * max(1.0, lambda_max)
    )
    if bool(torch_module.any(eigenvalues < -tolerance).item()):
        _refuse(V1FactoredRowRefusalCode.GRAM_NOT_PSD, "stacked row Gram is not PSD within its roundoff tolerance")
    eigenvalues = torch_module.clamp(torch_module.flip(eigenvalues, dims=(0,)), min=0.0)
    vectors = torch_module.flip(vectors, dims=(1,))
    # deterministic eigenvector signs (largest-magnitude entry positive)
    pivots = torch_module.argmax(vectors.abs(), dim=0)
    signs = torch_module.sign(vectors[pivots, torch_module.arange(vectors.shape[1], device=device)])
    signs = torch_module.where(signs == 0.0, torch_module.ones_like(signs), signs)
    vectors = vectors * signs
    retained = int(torch_module.count_nonzero(eigenvalues > tolerance).item())
    discarded = eigenvalues[retained:]
    residual_spectral_bound = float(discarded.max().item()) if discarded.numel() else 0.0
    tail_sum = float(discarded.sum().item()) if discarded.numel() else 0.0
    # H_k = Σ_j U[j,k] s_j with unit-norm eigenvectors: HᵀH = Σ_k λ_k u_k u_kᵀ ... the stacked
    # rows in the eigenbasis are U_kᵀ S; their Gram is diag(λ), so ‖H_k‖² = λ_k.  Rotation only.
    U = vectors[:, :retained].contiguous()
    mean = None
    start = 0
    for packet in packet_tuple:
        stop = start + packet.context_count
        block = packet.combine_rows(role, context_weights[start:stop], gate=gate64)[0]
        mean = block if mean is None else mean + block
        start = stop
    assert mean is not None
    receipt = V1FactoredRowExactGeometryReceipt(
        physical_route_axis_hash=first.physical_route_axis_hash,
        route_id_axis_hash=first.route_id_axis_hash,
        route_count=first.route_count,
        role=role.value,
        stratum_names=names,
        stratum_weights=weights,
        stratum_context_counts=tuple(counts[name] for name in names),
        context_count=n,
        packet_receipt_hashes=tuple(p.receipt.receipt_hash for p in packet_tuple),
        packet_stratum_labels=labels,
        gate_hash=_gate_hash(gate64),
        stacked_row_count=n + S,
        retained_rank=retained,
        retained_eigenvalues=tuple(float(v) for v in eigenvalues[:retained].tolist()),
        discarded_tail_eigenvalue_sum=tail_sum,
        residual_spectral_bound=residual_spectral_bound,
        psd_tolerance=tolerance,
        gram_frobenius_norm=float(torch_module.linalg.matrix_norm(gram_stack).item()),
        mean_component_l2_norm=float(torch_module.linalg.vector_norm(mean).item()),
    )
    return V1FactoredRowExactGeometry(
        receipt=receipt,
        packets=packet_tuple,
        role=role,
        gate64=gate64,
        context_weights=context_weights,
        combination=A,
        eigenvectors=U,
        mean_component=mean.contiguous(),
    )


@dataclass(frozen=True)
class V1FactoredExactGeometryKKTDenseAdapter:
    """Dense arrays in the shape ``V1RelaxedKKTProblem`` / Phase-A problems accept."""

    geometry_receipt_hash: str
    low_rank_factor: np.ndarray = field(repr=False, compare=False)
    residual_diagonal: np.ndarray = field(repr=False, compare=False)
    residual_spectral_bound: float = 0.0
    retained_rank: int = 0
    response_rows: int = 1
    dense_factor_bytes: int = 0
    budget_bytes: int = 0
    residual_is_roundoff_enclosure: bool = True
    schema: str = FACTORED_ROW_EXACT_GEOMETRY_KKT_DENSE_ADAPTER_SCHEMA

    def __post_init__(self) -> None:
        _identity(self.geometry_receipt_hash, name="geometry receipt hash")
        factor = np.ascontiguousarray(np.asarray(self.low_rank_factor, dtype=np.float64))
        residual = np.ascontiguousarray(np.asarray(self.residual_diagonal, dtype=np.float64))
        if factor.ndim != 2 or residual.ndim != 1 or factor.shape[1] != residual.shape[0]:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "KKT adapter arrays are misaligned")
        if not np.isfinite(factor).all() or not np.isfinite(residual).all() or bool(np.any(residual < 0.0)):
            _refuse(V1FactoredRowRefusalCode.NONFINITE, "KKT adapter arrays must be finite; residual nonnegative")
        rho = _finite_nonnegative(self.residual_spectral_bound, name="residual spectral bound")
        if residual.size and rho < float(residual.max()):
            # The relaxed-KKT invariant ``rho >= max residual_diagonal`` is kept
            # outward: snap up and record (both are roundoff-level here).
            rho = float(residual.max())
        factor.setflags(write=False)
        residual.setflags(write=False)
        object.__setattr__(self, "low_rank_factor", factor)
        object.__setattr__(self, "residual_diagonal", residual)
        object.__setattr__(self, "residual_spectral_bound", rho)
        if int(self.retained_rank) + int(self.response_rows) != factor.shape[0]:
            _refuse(V1FactoredRowRefusalCode.AXIS_MISMATCH, "factor rows must equal response rows plus retained rank")
        if int(self.dense_factor_bytes) != factor.nbytes:
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "dense factor byte count does not replay")
        if self.residual_is_roundoff_enclosure is not True or self.schema != FACTORED_ROW_EXACT_GEOMETRY_KKT_DENSE_ADAPTER_SCHEMA:
            _refuse(V1FactoredRowRefusalCode.STALE_IDENTITY, "KKT adapter schema drifted")

    @property
    def adapter_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "geometry_receipt_hash": self.geometry_receipt_hash,
                "factor_shape": list(self.low_rank_factor.shape),
                "retained_rank": self.retained_rank,
                "response_rows": self.response_rows,
                "residual_spectral_bound": self.residual_spectral_bound,
                "dense_factor_bytes": self.dense_factor_bytes,
                "budget_bytes": self.budget_bytes,
                "residual_is_roundoff_enclosure": self.residual_is_roundoff_enclosure,
                "route_scale_arrays_not_hashed": True,
            },
            prefix=_ADAPTER_PREFIX,
        )


def adapt_v1_factored_exact_geometry_to_kkt_arrays(
    geometry: V1FactoredRowExactGeometry,
    *,
    budget_bytes: int,
) -> V1FactoredExactGeometryKKTDenseAdapter:
    """Materialize ``low_rank_factor``, ``residual_diagonal``, ``residual_spectral_bound``.

    Refuses ``DENSE_FACTOR_NOT_ADMITTED`` when ``(1 + r) x N x 8`` exceeds the
    budget; the operator form (``forward``/``adjoint``) remains available.
    """

    if type(geometry) is not V1FactoredRowExactGeometry:
        _refuse(V1FactoredRowRefusalCode.INVALID_TYPE, "KKT adapter requires the exact geometry owner")
    factor = geometry.dense_factor(budget_bytes=budget_bytes)
    residual = geometry.residual_diagonal()
    receipt = geometry.receipt
    return V1FactoredExactGeometryKKTDenseAdapter(
        geometry_receipt_hash=receipt.receipt_hash,
        low_rank_factor=factor.detach().cpu().numpy(),
        residual_diagonal=residual.detach().cpu().numpy(),
        residual_spectral_bound=receipt.residual_spectral_bound,
        retained_rank=receipt.retained_rank,
        response_rows=1,
        dense_factor_bytes=int(factor.numel()) * _F64_BYTES,
        budget_bytes=int(budget_bytes),
    )


__all__ = [
    "FACTORED_ROW_EXACT_GEOMETRY_AUTHORITY_SCOPE",
    "FACTORED_ROW_EXACT_GEOMETRY_KKT_DENSE_ADAPTER_SCHEMA",
    "FACTORED_ROW_EXACT_GEOMETRY_LAW_ID",
    "FACTORED_ROW_EXACT_GEOMETRY_STACK_ID",
    "FACTORED_ROW_EXACT_GRAM_GEOMETRY_SCHEMA",
    "V1FactoredExactGeometryKKTDenseAdapter",
    "V1FactoredRowExactGeometry",
    "V1FactoredRowExactGeometryReceipt",
    "adapt_v1_factored_exact_geometry_to_kkt_arrays",
    "compute_v1_factored_row_exact_geometry",
]
