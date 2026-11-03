"""Nested-rank truncation views of one assembled ceiling FD state.

Row-backed D-to-C route: the rank-local
decision certificates for the exposed nested lattice below the ceiling are
*truncation views* of the one assembled ceiling FD state, never independently
accumulated rank-``r`` sketches.  For a nested rank ``r < ceiling`` the view's
geometry is the top-``r`` eigen-part of the ceiling state's centered factor
(rows rotated to the ceiling factor's own eigenbasis, which leaves the
quadratic form ``F^T F`` unchanged and only discards the trailing directions)
and its residual spectral bound is

    ceiling combined residual spectral bound + sum of the discarded retained
    eigenvalue squares,

which is exact for the truncation (the discarded PSD part has spectral norm at
most its trace) and outward for every action.  The residual diagonal bound is
the ceiling diagonal bound plus the exact diagonal of the discarded part.  At
``r == ceiling`` the view is the ceiling geometry itself (identity, byte for
byte).  This is the "nested ceiling-state view" the device session already
declares (``lower_rank_views_are_metadata_only``, ``CANONICAL_CEILING_ONLY``).

Every certificate derived on a truncation view receives a NEW sibling receipt
keyed by the certificate hash (``V1CeilingTruncationViewCertificateReceipt``)
that records the view provenance; no field is added to the landed certificate
family.

This module never re-executes D, never calls a measurement provider, and
carries no C/K/O authority.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from enum import Enum
import math
from typing import Any, Final, NoReturn

import numpy as np

from mechanistic_systems.geometry.trust import GlobalTrustGeometryState

from .contracts import content_hash
from .modern_v1_fqr_adaptive_rank import (
    V1FQRAdaptiveRankDecisionCertificate,
    V1FQREndogenousRankPolicyV2,
)
from .modern_v1_fqr_adaptive_rank_derivation import (
    V1FQRAdaptiveRankDerivedDecision,
    derive_v1_fqr_adaptive_rank_decision,
)
from .modern_v1_sampled_acquisition_adapters import V1SampledPreFreezeGeometrySource
from .modern_v1_sampled_kkt_bridge import V1SampledKKTBranchFamilyGraph
from .modern_v1_stratified_fd_gpu import V1StratifiedFDGeometryResult


CEILING_TRUNCATION_VIEW_GEOMETRY_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_ceiling_truncation_view_geometry_v1"
)
CEILING_TRUNCATION_VIEW_CERTIFICATE_SCHEMA: Final = (
    "partition_free_v1_sampled_v7_ceiling_truncation_view_certificate_v1"
)
CEILING_TRUNCATION_RESIDUAL_RULE: Final = (
    "view_residual_spectral_bound = ceiling_combined_residual_spectral_bound "
    "+ sum(discarded retained eigenvalue squares); exact for the truncation "
    "(discarded PSD part has spectral norm <= its trace); view residual "
    "diagonal = ceiling residual diagonal + exact discarded diagonal"
)
CEILING_TRUNCATION_VIEW_STATEMENT: Final = (
    "rank-local decision image below the ceiling is a truncation view of the "
    "ceiling sketch; its residual bound is exact for the truncation; it is "
    "not an independently accumulated rank-local sketch"
)
_PREFIX: Final = "partition-free-v1-sampled-v7-ceiling-truncation-view-"
_METADATA_VIEW_OF_RANK: Final = "ceiling_truncation_view_of_rank"
_METADATA_VIEW_RANK: Final = "ceiling_truncation_view_rank"
_METADATA_TAIL_TRACE: Final = "ceiling_truncation_tail_trace"
_METADATA_CEILING_GEOMETRY_HASH: Final = "ceiling_truncation_ceiling_geometry_hash"


class V1CeilingTruncationViewRefusalCode(str, Enum):
    INVALID_TYPE = "invalid_type"
    RANK_ABOVE_CEILING = "rank_above_ceiling"
    RETAINED_RANK_ABOVE_CEILING = "retained_rank_above_ceiling"
    RESIDUAL_DIAGONAL_ABSENT = "residual_diagonal_absent"
    NOT_A_TRUNCATION_VIEW = "not_a_truncation_view"
    RANK_NOT_ADMITTED = "rank_not_admitted"
    PROVIDER_REEXECUTION = "provider_reexecution"
    SOURCE_COORDINATE_MISMATCH = "source_coordinate_mismatch"


class V1CeilingTruncationViewRefusal(RuntimeError):
    def __init__(self, code: V1CeilingTruncationViewRefusalCode, message: str) -> None:
        if type(code) is not V1CeilingTruncationViewRefusalCode:
            raise TypeError("truncation-view refusal code must be exact")
        super().__init__(f"{code.value}: {message}")
        self.code = code
        self.message = message

    def to_json_payload(self) -> dict[str, object]:
        return {"code": self.code.value, "message": self.message}


def _refuse(code: V1CeilingTruncationViewRefusalCode, message: str) -> NoReturn:
    raise V1CeilingTruncationViewRefusal(code, message)


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        _refuse(
            V1CeilingTruncationViewRefusalCode.INVALID_TYPE,
            f"{name} must be one positive exact integer",
        )
    return value


# ---------------------------------------------------------------------------
# Spectrum of the ceiling state (witness input item 3)
# ---------------------------------------------------------------------------


def _eigen_rows(factor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalues desc, rotated rows) with rows @ rows.T diagonal.

    ``rows = U^T F`` for ``F F^T = U diag(lambda) U^T``; ``rows^T rows`` equals
    ``F^T F`` exactly over the reals, and ``||rows_i||^2 = lambda_i``.
    """

    rows = np.asarray(factor, dtype=np.float64)
    if rows.shape[0] == 0:
        return np.zeros(0, dtype=np.float64), rows
    gram = rows @ rows.T
    gram = 0.5 * (gram + gram.T)
    eigenvalues, vectors = np.linalg.eigh(gram)
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(vectors)):
        raise ValueError("ceiling factor Gram eigendecomposition is not finite")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    vectors = vectors[:, order]
    # Deterministic sign convention (largest-magnitude entry positive).
    pivots = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[pivots, np.arange(vectors.shape[1])])
    signs[signs == 0.0] = 1.0
    vectors = vectors * signs[None, :]
    rotated = vectors.T @ rows
    return eigenvalues, rotated


def derive_v1_ceiling_fd_eigenvalue_square_axis(
    geometry_result: V1StratifiedFDGeometryResult,
    *,
    ceiling_rank: int,
) -> tuple[float, ...]:
    """Retained eigenvalue squares of the assembled ceiling FD state.

    Eigenvalues of ``F F^T`` (``F`` the retained centered factor) in
    nonincreasing order, zero-padded to ``ceiling_rank`` entries when fewer
    rows were retained.  No provider call, no device allocation.
    """

    if type(geometry_result) is not V1StratifiedFDGeometryResult:
        _refuse(
            V1CeilingTruncationViewRefusalCode.INVALID_TYPE,
            "ceiling spectrum requires the exact stratified FD geometry result",
        )
    ceiling = _positive_int(ceiling_rank, name="ceiling rank")
    geometry_result.receipt.assert_integrity()
    if geometry_result.receipt.maximum_rank != ceiling:
        _refuse(
            V1CeilingTruncationViewRefusalCode.SOURCE_COORDINATE_MISMATCH,
            "assembled FD state maximum rank differs from the ceiling rank",
        )
    factor = np.asarray(geometry_result.geometry.centered_factor, dtype=np.float64)
    if factor.shape[0] > ceiling:
        _refuse(
            V1CeilingTruncationViewRefusalCode.RETAINED_RANK_ABOVE_CEILING,
            "assembled FD state retains more rows than the ceiling rank",
        )
    eigenvalues, _rows = _eigen_rows(factor)
    axis = [float(value) for value in eigenvalues]
    axis.extend(0.0 for _ in range(ceiling - len(axis)))
    if any(not math.isfinite(value) or value < 0.0 for value in axis):
        raise ValueError("ceiling eigenvalue squares must be finite and nonnegative")
    return tuple(axis)


# ---------------------------------------------------------------------------
# Truncation view of the geometry result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V1CeilingTruncationViewGeometryReceipt:
    """New family: how one nested-rank view was cut from the ceiling state."""

    ceiling_geometry_result_hash: str
    ceiling_geometry_hash: str
    ceiling_assembly_receipt_hash: str
    ceiling_rank: int
    view_rank: int
    view_retained_rank: int
    view_geometry_hash: str
    view_geometry_result_hash: str
    view_assembly_receipt_hash: str
    ceiling_residual_spectral_bound: float
    discarded_eigenvalue_square_sum: float
    view_residual_spectral_bound: float
    identity_view: bool
    ceiling_truncation_view: bool = True
    independently_accumulated_rank_local_sketch: bool = False
    provider_calls: int = 0
    residual_rule: str = CEILING_TRUNCATION_RESIDUAL_RULE
    statement: str = CEILING_TRUNCATION_VIEW_STATEMENT
    schema: str = CEILING_TRUNCATION_VIEW_GEOMETRY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "ceiling_geometry_result_hash",
            "ceiling_geometry_hash",
            "ceiling_assembly_receipt_hash",
            "view_geometry_hash",
            "view_geometry_result_hash",
            "view_assembly_receipt_hash",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be one nonempty string")
        ceiling = _positive_int(self.ceiling_rank, name="ceiling rank")
        rank = _positive_int(self.view_rank, name="view rank")
        if rank > ceiling:
            raise ValueError("view rank cannot exceed the ceiling rank")
        if type(self.view_retained_rank) is not int or not 0 <= self.view_retained_rank <= rank:
            raise ValueError("view retained rank must lie in [0, view rank]")
        for name in (
            "ceiling_residual_spectral_bound",
            "discarded_eigenvalue_square_sum",
            "view_residual_spectral_bound",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if not math.isclose(
            self.view_residual_spectral_bound,
            self.ceiling_residual_spectral_bound + self.discarded_eigenvalue_square_sum,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("view residual bound does not replay the truncation rule")
        if type(self.identity_view) is not bool:
            raise TypeError("identity_view must be an exact boolean")
        if self.identity_view != (rank == ceiling):
            raise ValueError("identity view exactly when the view rank is the ceiling")
        if self.identity_view and (
            self.discarded_eigenvalue_square_sum != 0.0
            or self.view_geometry_hash != self.ceiling_geometry_hash
        ):
            raise ValueError("identity view must be the ceiling geometry itself")
        if (
            self.ceiling_truncation_view is not True
            or self.independently_accumulated_rank_local_sketch is not False
            or self.provider_calls != 0
            or self.residual_rule != CEILING_TRUNCATION_RESIDUAL_RULE
            or self.statement != CEILING_TRUNCATION_VIEW_STATEMENT
            or self.schema != CEILING_TRUNCATION_VIEW_GEOMETRY_SCHEMA
        ):
            raise ValueError("truncation-view geometry receipt is invalid")

    @property
    def receipt_hash(self) -> str:
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        return content_hash(payload, prefix=_PREFIX + "geometry-")

    def to_json_payload(self) -> dict[str, object]:
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        payload["receipt_hash"] = self.receipt_hash
        return payload


def truncate_v1_stratified_fd_geometry_result_to_nested_rank(
    result: V1StratifiedFDGeometryResult,
    *,
    rank: int,
    ceiling_rank: int,
) -> tuple[V1StratifiedFDGeometryResult, V1CeilingTruncationViewGeometryReceipt]:
    """Cut one nested-rank truncation view from the assembled ceiling state."""

    if type(result) is not V1StratifiedFDGeometryResult:
        _refuse(
            V1CeilingTruncationViewRefusalCode.INVALID_TYPE,
            "truncation view requires the exact stratified FD geometry result",
        )
    view_rank = _positive_int(rank, name="view rank")
    ceiling = _positive_int(ceiling_rank, name="ceiling rank")
    if view_rank > ceiling:
        _refuse(
            V1CeilingTruncationViewRefusalCode.RANK_ABOVE_CEILING,
            f"nested rank {view_rank} exceeds the ceiling rank {ceiling}; no FD "
            "state above the ceiling exists on this session",
        )
    receipt = result.receipt
    receipt.assert_integrity()
    geometry = result.geometry
    if receipt.maximum_rank != ceiling:
        _refuse(
            V1CeilingTruncationViewRefusalCode.SOURCE_COORDINATE_MISMATCH,
            "assembled FD state maximum rank differs from the ceiling rank",
        )
    if geometry.residual_diagonal_bound is None:
        _refuse(
            V1CeilingTruncationViewRefusalCode.RESIDUAL_DIAGONAL_ABSENT,
            "ceiling geometry carries no residual diagonal bound",
        )
    factor = np.asarray(geometry.centered_factor, dtype=np.float64)
    if factor.shape[0] > ceiling:
        _refuse(
            V1CeilingTruncationViewRefusalCode.RETAINED_RANK_ABOVE_CEILING,
            "assembled FD state retains more rows than the ceiling rank",
        )
    ceiling_residual = float(receipt.combined_residual_spectral_bound)
    if view_rank == ceiling:
        view_receipt = V1CeilingTruncationViewGeometryReceipt(
            ceiling_geometry_result_hash=result.result_hash,
            ceiling_geometry_hash=geometry.content_hash,
            ceiling_assembly_receipt_hash=receipt.receipt_hash,
            ceiling_rank=ceiling,
            view_rank=view_rank,
            view_retained_rank=int(factor.shape[0]),
            view_geometry_hash=geometry.content_hash,
            view_geometry_result_hash=result.result_hash,
            view_assembly_receipt_hash=receipt.receipt_hash,
            ceiling_residual_spectral_bound=ceiling_residual,
            discarded_eigenvalue_square_sum=0.0,
            view_residual_spectral_bound=ceiling_residual,
            identity_view=True,
        )
        return result, view_receipt

    eigenvalues, rotated = _eigen_rows(factor)
    kept = rotated[:view_rank]
    discarded = rotated[view_rank:]
    tail_trace = float(math.fsum(float(value) for value in eigenvalues[view_rank:]))
    discarded_diagonal = (
        np.zeros(factor.shape[1], dtype=np.float64)
        if discarded.shape[0] == 0
        else np.sum(discarded * discarded, axis=0)
    )
    view_residual = ceiling_residual + tail_trace
    residual_diagonal = np.asarray(geometry.residual_diagonal_bound, dtype=np.float64)
    view_diagonal = np.ascontiguousarray(residual_diagonal + discarded_diagonal)
    view_factor = np.ascontiguousarray(kept)
    mean_host = np.ascontiguousarray(np.asarray(geometry.mean_component, dtype=np.float64))
    metadata = {
        **dict(geometry.metadata),
        _METADATA_VIEW_OF_RANK: ceiling,
        _METADATA_VIEW_RANK: view_rank,
        _METADATA_TAIL_TRACE: tail_trace,
        _METADATA_CEILING_GEOMETRY_HASH: geometry.content_hash,
        "recompression_residual_spectral_bound": float(
            receipt.recompression_residual_spectral_bound
        ),
    }
    view_geometry = GlobalTrustGeometryState(
        spec=geometry.spec,
        route_ids=geometry.route_ids,
        mean_component=mean_host,
        centered_factor=view_factor,
        residual_spectral_bound=view_residual,
        residual_diagonal_bound=view_diagonal,
        total_weight=geometry.total_weight,
        context_count=geometry.context_count,
        context_axis_hash=geometry.context_axis_hash,
        measured_mask_hash=geometry.measured_mask_hash,
        calibration_state_id=geometry.calibration_state_id,
        source_measurement_hashes=geometry.source_measurement_hashes,
        derivation_receipt=None,
        metadata=metadata,
    )
    view_assembly_receipt = replace(
        receipt,
        combined_geometry_hash=view_geometry.content_hash,
        retained_rank=int(view_factor.shape[0]),
        maximum_rank=view_rank,
        source_residual_spectral_bound=ceiling_residual,
        recompression_residual_spectral_bound=tail_trace,
        combined_residual_spectral_bound=view_residual,
        centered_factor_hash=content_hash(
            view_factor,
            prefix="partition-free-v1-stratified-fd-factor-",
        ),
        residual_diagonal_hash=content_hash(
            view_diagonal,
            prefix="partition-free-v1-stratified-fd-residual-diagonal-",
        ),
    )
    view_result = V1StratifiedFDGeometryResult(
        design=result.design,
        geometry=view_geometry,
        receipt=view_assembly_receipt,
        memory_forecast=result.memory_forecast,
    )
    view_receipt = V1CeilingTruncationViewGeometryReceipt(
        ceiling_geometry_result_hash=result.result_hash,
        ceiling_geometry_hash=geometry.content_hash,
        ceiling_assembly_receipt_hash=receipt.receipt_hash,
        ceiling_rank=ceiling,
        view_rank=view_rank,
        view_retained_rank=int(view_factor.shape[0]),
        view_geometry_hash=view_geometry.content_hash,
        view_geometry_result_hash=view_result.result_hash,
        view_assembly_receipt_hash=view_assembly_receipt.receipt_hash,
        ceiling_residual_spectral_bound=ceiling_residual,
        discarded_eigenvalue_square_sum=tail_trace,
        view_residual_spectral_bound=view_residual,
        identity_view=False,
    )
    return view_result, view_receipt


def truncate_v1_sampled_pre_freeze_geometry_source_to_nested_rank(
    source: V1SampledPreFreezeGeometrySource,
    *,
    rank: int,
    ceiling_rank: int,
) -> tuple[V1SampledPreFreezeGeometrySource, V1CeilingTruncationViewGeometryReceipt]:
    """Truncation view of a pre-freeze sampled geometry source (same admission)."""

    if type(source) is not V1SampledPreFreezeGeometrySource:
        _refuse(
            V1CeilingTruncationViewRefusalCode.INVALID_TYPE,
            "truncation view requires the exact pre-freeze sampled geometry source",
        )
    view_result, receipt = truncate_v1_stratified_fd_geometry_result_to_nested_rank(
        source.geometry_result,
        rank=rank,
        ceiling_rank=ceiling_rank,
    )
    if receipt.identity_view:
        return source, receipt
    return (
        V1SampledPreFreezeGeometrySource(
            admission=source.admission,
            geometry_result=view_result,
        ),
        receipt,
    )


def is_v1_ceiling_truncation_view_geometry(geometry: GlobalTrustGeometryState) -> bool:
    return (
        type(geometry) is GlobalTrustGeometryState
        and geometry.metadata.get(_METADATA_VIEW_OF_RANK) is not None
    )


# ---------------------------------------------------------------------------
# Certificate sibling receipt and the rank-local deriver
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class V1CeilingTruncationViewCertificateReceipt:
    """New sibling family keyed by the rank-local certificate hash."""

    certificate_hash: str
    rank: int
    ceiling_rank: int
    ceiling_fd_state_hash: str
    ceiling_geometry_hash: str
    view_geometry_receipt_hash: str
    view_geometry_hash: str
    graph_hash: str
    endogenous_rank_policy_hash: str
    ceiling_truncation_view: bool
    identity_view: bool
    statement: str = CEILING_TRUNCATION_VIEW_STATEMENT
    schema: str = CEILING_TRUNCATION_VIEW_CERTIFICATE_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "certificate_hash",
            "ceiling_fd_state_hash",
            "ceiling_geometry_hash",
            "view_geometry_receipt_hash",
            "view_geometry_hash",
            "graph_hash",
            "endogenous_rank_policy_hash",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be one nonempty string")
        rank = _positive_int(self.rank, name="certificate rank")
        ceiling = _positive_int(self.ceiling_rank, name="ceiling rank")
        if rank > ceiling:
            raise ValueError("certificate rank cannot exceed the ceiling rank")
        if type(self.ceiling_truncation_view) is not bool or type(self.identity_view) is not bool:
            raise TypeError("truncation-view flags must be exact booleans")
        if self.ceiling_truncation_view is not True:
            raise ValueError("every rank-local certificate on this lane is a ceiling view")
        if self.identity_view != (rank == ceiling):
            raise ValueError("identity view exactly at the ceiling rank")
        if (
            self.statement != CEILING_TRUNCATION_VIEW_STATEMENT
            or self.schema != CEILING_TRUNCATION_VIEW_CERTIFICATE_SCHEMA
        ):
            raise ValueError("truncation-view certificate receipt is invalid")

    @property
    def receipt_hash(self) -> str:
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        return content_hash(payload, prefix=_PREFIX + "certificate-")

    def to_json_payload(self) -> dict[str, object]:
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        payload["receipt_hash"] = self.receipt_hash
        return payload


GraphBuilder = Callable[[V1SampledPreFreezeGeometrySource], V1SampledKKTBranchFamilyGraph]


@dataclass(slots=True)
class V1CeilingTruncationRankLocalDeriver:
    """Rank-local certificate producer over nested views of one ceiling graph.

    ``__call__(rank)`` is the ``RankLocalDecisionDeriver`` the device-native
    witness consumes.  Rank ``ceiling`` derives on the ceiling graph itself;
    ranks below the ceiling truncate the ceiling graph's sampled geometry
    source, build one rank-local KKT graph on the same registered prefix
    through ``graph_builder`` (one rank-local KKT solve, no D re-execution),
    and derive there.  Ranks above the ceiling or outside the v2 admitted
    lattice return ``None`` so the witness refuses with its typed absence.

    ``provider_call_counter`` (optional) is read before and after every graph
    build; any change is a typed ``provider_reexecution`` refusal.  Live
    rank-local graphs are released after derivation; only compact
    certificates and receipts are retained.
    """

    ceiling_graph: V1SampledKKTBranchFamilyGraph = field(repr=False)
    endogenous_rank_policy: V1FQREndogenousRankPolicyV2
    numerical_enclosure_policy_hash: str
    graph_builder: GraphBuilder = field(repr=False)
    ceiling_fd_state_hash: str
    provider_call_counter: Callable[[], int] | None = field(default=None, repr=False)
    ceiling_rank: int = field(init=False)
    certificates_by_rank: dict[int, V1FQRAdaptiveRankDerivedDecision] = field(
        default_factory=dict, repr=False
    )
    geometry_receipts_by_rank: dict[int, V1CeilingTruncationViewGeometryReceipt] = field(
        default_factory=dict
    )
    certificate_receipts_by_hash: dict[str, V1CeilingTruncationViewCertificateReceipt] = (
        field(default_factory=dict)
    )
    graph_hashes_by_rank: dict[int, str] = field(default_factory=dict)
    provider_calls_observed: int = 0

    def __post_init__(self) -> None:
        if type(self.ceiling_graph) is not V1SampledKKTBranchFamilyGraph:
            _refuse(
                V1CeilingTruncationViewRefusalCode.INVALID_TYPE,
                "rank-local deriver requires the exact ceiling KKT graph",
            )
        if type(self.endogenous_rank_policy) is not V1FQREndogenousRankPolicyV2:
            _refuse(
                V1CeilingTruncationViewRefusalCode.INVALID_TYPE,
                "rank-local deriver requires the exact v2 endogenous-rank policy",
            )
        if type(self.ceiling_fd_state_hash) is not str or not self.ceiling_fd_state_hash:
            _refuse(
                V1CeilingTruncationViewRefusalCode.INVALID_TYPE,
                "rank-local deriver requires the ceiling FD state hash",
            )
        if not callable(self.graph_builder):
            _refuse(
                V1CeilingTruncationViewRefusalCode.INVALID_TYPE,
                "rank-local deriver requires one callable rank-local graph builder",
            )
        receipt = self.ceiling_graph.sampled_geometry_source.geometry_result.receipt
        self.ceiling_rank = int(receipt.maximum_rank)
        if (
            self.ceiling_graph.sampled_geometry_source.admission.receipt.fd_snapshot_hash
            != self.ceiling_fd_state_hash
        ):
            _refuse(
                V1CeilingTruncationViewRefusalCode.SOURCE_COORDINATE_MISMATCH,
                "ceiling graph was not assembled from the named ceiling FD state",
            )

    def _observe_provider(self, before: int | None) -> None:
        if self.provider_call_counter is None:
            return
        after = int(self.provider_call_counter())
        if before is not None and after != before:
            self.provider_calls_observed += after - before
            _refuse(
                V1CeilingTruncationViewRefusalCode.PROVIDER_REEXECUTION,
                "a rank-local truncation view triggered a measurement-provider call",
            )

    def __call__(self, rank: int) -> V1FQRAdaptiveRankDerivedDecision | None:
        view_rank = _positive_int(rank, name="requested nested rank")
        cached = self.certificates_by_rank.get(view_rank)
        if cached is not None:
            return cached
        if view_rank > self.ceiling_rank:
            return None
        if view_rank not in self.endogenous_rank_policy.admitted_rank_lattice:
            return None
        before = (
            None if self.provider_call_counter is None else int(self.provider_call_counter())
        )
        graph: V1SampledKKTBranchFamilyGraph
        if view_rank == self.ceiling_rank:
            graph = self.ceiling_graph
            _view_result, geometry_receipt = (
                truncate_v1_stratified_fd_geometry_result_to_nested_rank(
                    graph.sampled_geometry_source.geometry_result,
                    rank=view_rank,
                    ceiling_rank=self.ceiling_rank,
                )
            )
        else:
            view_source, geometry_receipt = (
                truncate_v1_sampled_pre_freeze_geometry_source_to_nested_rank(
                    self.ceiling_graph.sampled_geometry_source,
                    rank=view_rank,
                    ceiling_rank=self.ceiling_rank,
                )
            )
            graph = self.graph_builder(view_source)
            if type(graph) is not V1SampledKKTBranchFamilyGraph:
                _refuse(
                    V1CeilingTruncationViewRefusalCode.INVALID_TYPE,
                    "rank-local graph builder returned another graph type",
                )
            if graph.sampled_geometry_source is not view_source:
                _refuse(
                    V1CeilingTruncationViewRefusalCode.NOT_A_TRUNCATION_VIEW,
                    "rank-local graph was not built on the truncation-view source",
                )
        self._observe_provider(before)
        derived = derive_v1_fqr_adaptive_rank_decision(
            policy=self.endogenous_rank_policy,
            rank=view_rank,
            graph=graph,
            numerical_enclosure_policy_hash=self.numerical_enclosure_policy_hash,
        )
        certificate = derived.decision_certificate
        certificate_receipt = V1CeilingTruncationViewCertificateReceipt(
            certificate_hash=certificate.certificate_hash,
            rank=view_rank,
            ceiling_rank=self.ceiling_rank,
            ceiling_fd_state_hash=self.ceiling_fd_state_hash,
            ceiling_geometry_hash=geometry_receipt.ceiling_geometry_hash,
            view_geometry_receipt_hash=geometry_receipt.receipt_hash,
            view_geometry_hash=geometry_receipt.view_geometry_hash,
            graph_hash=graph.graph_hash,
            endogenous_rank_policy_hash=self.endogenous_rank_policy.policy_hash,
            ceiling_truncation_view=True,
            identity_view=geometry_receipt.identity_view,
        )
        self.certificates_by_rank[view_rank] = derived
        self.geometry_receipts_by_rank[view_rank] = geometry_receipt
        self.certificate_receipts_by_hash[certificate.certificate_hash] = certificate_receipt
        self.graph_hashes_by_rank[view_rank] = graph.graph_hash
        # Release the rank-local route-scale graph: only compact products stay.
        del graph
        return derived

    def certificate_receipt_for(
        self, certificate: V1FQRAdaptiveRankDecisionCertificate
    ) -> V1CeilingTruncationViewCertificateReceipt:
        return self.certificate_receipts_by_hash[certificate.certificate_hash]

    def summary(self) -> dict[str, Any]:
        return {
            "ceiling_rank": self.ceiling_rank,
            "ceiling_fd_state_hash": self.ceiling_fd_state_hash,
            "derived_ranks": tuple(sorted(self.certificates_by_rank)),
            "certificate_hashes_by_rank": tuple(
                (rank, value.decision_certificate.certificate_hash)
                for rank, value in sorted(self.certificates_by_rank.items())
            ),
            "geometry_receipt_hashes_by_rank": tuple(
                (rank, value.receipt_hash)
                for rank, value in sorted(self.geometry_receipts_by_rank.items())
            ),
            "certificate_receipt_hashes": tuple(
                (certificate_hash, value.receipt_hash)
                for certificate_hash, value in sorted(
                    self.certificate_receipts_by_hash.items()
                )
            ),
            "graph_hashes_by_rank": tuple(sorted(self.graph_hashes_by_rank.items())),
            "provider_calls_observed": self.provider_calls_observed,
            "statement": CEILING_TRUNCATION_VIEW_STATEMENT,
        }


__all__ = [
    "CEILING_TRUNCATION_RESIDUAL_RULE",
    "CEILING_TRUNCATION_VIEW_CERTIFICATE_SCHEMA",
    "CEILING_TRUNCATION_VIEW_GEOMETRY_SCHEMA",
    "CEILING_TRUNCATION_VIEW_STATEMENT",
    "V1CeilingTruncationRankLocalDeriver",
    "V1CeilingTruncationViewCertificateReceipt",
    "V1CeilingTruncationViewGeometryReceipt",
    "V1CeilingTruncationViewRefusal",
    "V1CeilingTruncationViewRefusalCode",
    "derive_v1_ceiling_fd_eigenvalue_square_axis",
    "is_v1_ceiling_truncation_view_geometry",
    "truncate_v1_sampled_pre_freeze_geometry_source_to_nested_rank",
    "truncate_v1_stratified_fd_geometry_result_to_nested_rank",
]
