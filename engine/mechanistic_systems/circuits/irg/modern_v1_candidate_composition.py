"""Finite-domain, proposal-only candidate composition for modern v1.

This module evaluates one registered chart surrogate exactly on an explicitly
declared finite tuple of binary removal sets.  "Exact" is deliberately narrow:
the numerical value of the registered surrogate and the optimum within the
declared domain are exact.  No result is an exact intervention damage, a
posterior statement, a scientific certificate, a globally optimal powerset
search, or authority to commit a mask.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Final, Literal

import numpy as np

from mechanistic_systems.core.axis_identity import (
    canonical_unique_string_axis,
    ordered_string_axis_hash,
)

from .candidates import HardMask, RemovedRouteIndexSet
from .contracts import GlobalTrustGeometrySpec, content_hash
from .modern_v1_core import (
    V1AnalyticExactGateCorrectionLineage,
    V1AnalyticExactGateCorrectionVerificationReceipt,
    V1FiniteActionClassProof,
    V1FrequentDirectionsLineage,
    V1InterventionalCorrectionLineage,
    V1InterventionalCorrectionVerificationReceipt,
    V1ProposalChart,
)
from .modern_v1_fd_adapter import (
    EFFECTIVE_FACTOR_DERIVATION_ID,
    EXACT_STREAMED_NORM_DERIVATION_ID,
    FD_SHRINK_DERIVATION_ID,
    V1FDAdapterResult,
)
from .modern_v1_intervention_adapter import (
    REMOVAL_INDICATOR_POSITIVE_DAMAGE_V1,
    V1AnalyticExactGateCorrectionResult,
    V1AtomCostState,
    V1InterventionalCorrectionResult,
)
from .trust_geometry import (
    GLOBAL_TRUST_GEOMETRY_STATE_SCHEMA,
    GlobalTrustGeometryDerivationReceipt,
    GlobalTrustGeometryState,
)

NO_HALF_SURROGATE_EQUATION_ID: Final = (
    "corrected_price_plus_fd_quadratic_plus_residual_diagonal_no_half_v1"
)
FD_SPECTRAL_UPPER_EQUATION_ID: Final = (
    "fd_quadratic_plus_cumulative_shrink_times_binary_l2_squared_v1"
)
EXACT_FINITE_DOMAIN_COMPOSER_ID: Final = "declared_finite_removal_domain_exact_v1"
FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID: Final = (
    "frozen_resource_descending_frontier_exact_v1"
)
FQ_AUTHORIZED_RESOURCE_DESCENDING_COMPOSER_ID: Final = (
    "fq_authorized_resource_descending_compact_v1"
)
_INLINE_ACTION_LIMIT: Final = 1024
_INLINE_ROUTE_ID_AXIS_LIMIT: Final = 4096
_ACTION_AGGREGATION_CHUNK_SIZE: Final = 1 << 18

DiagonalHardBoundMode = Literal[
    "proposal_ranking_only",
    "proved_chart_bound",
]

_DOMAIN_SCHEMA = "partition_free_v1_finite_removal_domain_v1"
_CONSTRAINTS_SCHEMA = "partition_free_v1_candidate_composition_constraints_v1"
_NUMERIC_STATE_SCHEMA = "partition_free_v1_candidate_numeric_state_v1"
_PROPOSAL_SCHEMA = "partition_free_v1_candidate_proposal_v1"
_RESULT_SCHEMA = "partition_free_v1_candidate_composition_result_v2"
_FQ_ACTION_NUMERIC_STATE_SCHEMA = (
    "partition_free_v1_fq_action_numeric_state_v1"
)
_FQ_PROPOSAL_SCHEMA = "partition_free_v1_fq_candidate_proposal_v1"
_FQ_RESULT_SCHEMA = "partition_free_v1_fq_candidate_composition_result_v1"
_FQ_COMPACT_DOMAIN_SCHEMA = "partition_free_v1_fq_compact_domain_identity_v1"
_FQ_CONSTRAINTS_SCHEMA = "partition_free_v1_fq_constraints_identity_v1"
_FQ_RESOURCE_GAIN_RECEIPT_SCHEMA = (
    "partition_free_v1_fq_exact_scalar_resource_gain_receipt_v1"
)
_FQ_ACTION_NUMERIC_STATE_TOKEN = object()
_FQ_PROPOSAL_TOKEN = object()
_FQ_RESULT_TOKEN = object()
_FQ_F_UNITS: Final = "absolute_raw_target_logit"
_FQ_Q_UNITS: Final = "squared_raw_target_logit"
_FQ_SQRT_Q_UNITS: Final = "absolute_raw_target_logit"
_NUMERIC_STATE_TOKEN = object()


def _nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be a nonnegative integer")
    number = int(value)
    if number < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return number


def _positive_int(value: object, label: str) -> int:
    number = _nonnegative_int(value, label)
    if number <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return number


def _bytes_vector(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    if nonnegative and bool(np.any(array < 0.0)):
        raise ValueError(f"{name} must be nonnegative")
    if array.flags.c_contiguous and _is_bytes_backed_readonly(array):
        return array
    payload = np.ascontiguousarray(array, dtype=np.float64).tobytes(order="C")
    return np.frombuffer(payload, dtype=np.float64, count=array.size)


def _bytes_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty matrix")
    payload = np.ascontiguousarray(array, dtype=np.float64).tobytes(order="C")
    return np.frombuffer(payload, dtype=np.float64, count=array.size).reshape(array.shape)


def _is_bytes_backed_readonly(array: np.ndarray) -> bool:
    if array.flags.writeable:
        return False
    owner: object = array
    seen: set[int] = set()
    while isinstance(owner, np.ndarray) and owner.base is not None:
        if id(owner) in seen:  # pragma: no cover - defensive cycle guard
            return False
        seen.add(id(owner))
        owner = owner.base
    if isinstance(owner, bytes):
        return True
    if isinstance(owner, memoryview):
        return owner.readonly
    return False


def _fresh_global_trust_state_hash(state: GlobalTrustGeometryState) -> str:
    payload = {
        "schema": state.schema,
        "spec": state.spec,
        "route_axis_hash": ordered_string_axis_hash(
            state.route_ids,
            name="modern-v1 candidate source route axis",
        ),
        "route_count": len(state.route_ids),
        "mean_component": state.mean_component,
        "centered_factor": state.centered_factor,
        "residual_spectral_bound": state.residual_spectral_bound,
        "residual_diagonal_bound": state.residual_diagonal_bound,
        "total_weight": state.total_weight,
        "context_count": state.context_count,
        "context_axis_hash": state.context_axis_hash,
        "measured_mask_hash": state.measured_mask_hash,
        "calibration_state_id": state.calibration_state_id,
        "source_measurement_hashes": state.source_measurement_hashes,
        "metadata": state.metadata,
    }
    if state.derivation_receipt is not None:
        payload["derivation_receipt"] = state.derivation_receipt
    return content_hash(payload, prefix="irg-global-trust-")


@dataclass(frozen=True, slots=True)
class V1RemovedRouteSetIdentity:
    """O(1) identity for a large exact route set recoverable from two masks."""

    route_count: int
    count: int
    index_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "route_count", _positive_int(self.route_count, "route_count"))
        object.__setattr__(self, "count", _positive_int(self.count, "count"))
        if self.count > self.route_count:
            raise ValueError("removed-route identity exceeds the physical route axis")
        object.__setattr__(self, "index_hash", _nonempty(self.index_hash, "index_hash"))

    @classmethod
    def from_removed_routes(
        cls,
        removed_routes: RemovedRouteIndexSet,
    ) -> "V1RemovedRouteSetIdentity":
        if type(removed_routes) is not RemovedRouteIndexSet:
            raise TypeError("removed-route identity requires RemovedRouteIndexSet")
        return cls(
            route_count=removed_routes.route_count,
            count=removed_routes.count,
            index_hash=removed_routes.index_hash,
        )


@dataclass(frozen=True, slots=True)
class V1RouteIdAxisIdentity:
    """Count/hash-only snapshot of a validated large ordered string axis."""

    count: int
    ordered_hash: str
    schema: str = "partition_free_v1_route_id_axis_identity_v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "count", _positive_int(self.count, "count"))
        object.__setattr__(
            self,
            "ordered_hash",
            _nonempty(self.ordered_hash, "ordered_hash"),
        )
        if self.schema != "partition_free_v1_route_id_axis_identity_v1":
            raise ValueError("unsupported route-ID-axis identity schema")

    def __len__(self) -> int:
        return self.count


def _selected_route_id_axis_hash(
    route_ids: tuple[str, ...],
    selected: np.ndarray,
) -> str:
    """Hash an ordered selected string axis without materializing its tuple."""

    flags = np.asarray(selected, dtype=bool).reshape(-1)
    if flags.shape != (len(route_ids),):
        raise ValueError("selected route-ID flags do not align with the route axis")
    count = int(np.count_nonzero(flags))
    if count <= 0:
        raise ValueError("selected route-ID axis must be nonempty")
    if count == len(route_ids):
        return ordered_string_axis_hash(
            route_ids,
            name="modern-v1 selected route-ID axis",
        )
    digest = hashlib.sha256(b"mechanistic-systems-ordered-string-axis-v1\0")
    digest.update(count.to_bytes(8, "big", signed=False))
    observed = 0
    for route_id, keep in zip(route_ids, flags, strict=True):
        if not bool(keep):
            continue
        encoded = route_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
        observed += 1
    if observed != count:  # pragma: no cover - defensive accounting invariant
        raise RuntimeError("selected route-ID hash count changed during traversal")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class V1PackedRemovalAction:
    """A large finite action stored as an exact packed child plus delta identity.

    A 57-million-route child occupies about 7 MiB, independent of the number
    removed.  The full sorted uint32 delta is reconstructed only while this one
    action is evaluated and is not retained by the domain or proposal.
    """

    resulting_mask: HardMask
    removed_count: int
    removed_route_index_hash: str

    def __post_init__(self) -> None:
        if type(self.resulting_mask) is not HardMask:
            raise TypeError("packed removal action requires an exact HardMask")
        count = _positive_int(self.removed_count, "removed_count")
        if count > self.resulting_mask.route_count:
            raise ValueError("packed removal action exceeds its route axis")
        object.__setattr__(self, "removed_count", count)
        object.__setattr__(
            self,
            "removed_route_index_hash",
            _nonempty(self.removed_route_index_hash, "removed_route_index_hash"),
        )

    @classmethod
    def from_removed_routes(
        cls,
        *,
        parent_mask: HardMask,
        removed_routes: RemovedRouteIndexSet,
    ) -> "V1PackedRemovalAction":
        if type(parent_mask) is not HardMask or type(removed_routes) is not RemovedRouteIndexSet:
            raise TypeError("packed removal action requires exact mask and route-set objects")
        if removed_routes.route_count != parent_mask.route_count:
            raise ValueError("packed removal action route axes differ")
        return cls(
            resulting_mask=parent_mask.with_removed_indices(removed_routes),
            removed_count=removed_routes.count,
            removed_route_index_hash=removed_routes.index_hash,
        )

    @classmethod
    def from_removed_indices(
        cls,
        *,
        parent_mask: HardMask,
        removed_indices: Sequence[int] | np.ndarray,
    ) -> "V1PackedRemovalAction":
        """Build directly from a ranking prefix without retaining/sorting it."""

        if type(parent_mask) is not HardMask:
            raise TypeError("packed removal action requires an exact parent mask")
        source = np.asarray(removed_indices)
        count = int(source.size)
        if count <= 0:
            raise ValueError("packed removal action requires a nonempty route delta")
        child = parent_mask.with_removed_indices(removed_indices)
        resolved_count, index_hash = child.removed_index_count_and_hash_from(parent_mask)
        if resolved_count != count:
            raise ValueError("packed removal action indices were not unique")
        return cls(
            resulting_mask=child,
            removed_count=resolved_count,
            removed_route_index_hash=index_hash,
        )

    @classmethod
    def from_resulting_mask(
        cls,
        *,
        parent_mask: HardMask,
        resulting_mask: HardMask,
    ) -> "V1PackedRemovalAction":
        """Build from a packed child by streaming the parent-to-child delta hash."""

        if type(parent_mask) is not HardMask or type(resulting_mask) is not HardMask:
            raise TypeError("packed removal action requires exact parent and child masks")
        parent_mask.assert_same_axis(resulting_mask)
        resolved_count, index_hash = resulting_mask.removed_index_count_and_hash_from(
            parent_mask
        )
        if resolved_count <= 0:
            raise ValueError("packed removal action requires a nonempty route delta")
        return cls(
            resulting_mask=resulting_mask,
            removed_count=resolved_count,
            removed_route_index_hash=index_hash,
        )

    @classmethod
    def action_hash_from_streamed_identity(
        cls,
        *,
        route_axis_hash: str,
        route_count: int,
        resulting_mask_packed_sha256: str,
        removed_count: int,
        removed_route_index_hash: str,
    ) -> str:
        """Reproduce :attr:`action_hash` without retaining a child mask.

        Route-scale device-shell construction already streams the canonical
        packed child bytes to SHA-256.  This helper injects that exact nested
        bytes representation into the historical dataclass payload; it does
        not define a second action identity.
        """

        axis = _nonempty(route_axis_hash, "route_axis_hash")
        routes = _positive_int(route_count, "route_count")
        count = _positive_int(removed_count, "removed_count")
        if count > routes:
            raise ValueError("packed removal action exceeds its route axis")
        removed_hash = _nonempty(
            removed_route_index_hash,
            "removed_route_index_hash",
        )
        # Reuse the HardMask validator so digest spelling, route count, and
        # packed length remain owned by the canonical mask contract.
        HardMask.mask_hash_from_packed_digest(
            route_axis_hash=axis,
            route_count=routes,
            packed_sha256=resulting_mask_packed_sha256,
        )
        return content_hash(
            {
                "resulting_mask": {
                    "route_axis_hash": axis,
                    "route_count": routes,
                    "packed_bits": {
                        "bytes_sha256": resulting_mask_packed_sha256,
                        "length": (routes + 7) // 8,
                    },
                },
                "removed_count": count,
                "removed_route_index_hash": removed_hash,
            },
            prefix="partition-free-v1-packed-removal-action-",
        )

    def __len__(self) -> int:
        return self.removed_count

    @property
    def action_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-packed-removal-action-")


@dataclass(frozen=True, slots=True)
class V1FiniteRemovalDomain:
    """Content-addressed finite action domain; never an implicit powerset."""

    route_axis_hash: str
    route_count: int
    removal_sets: tuple[
        tuple[int, ...] | RemovedRouteIndexSet | V1PackedRemovalAction,
        ...,
    ]
    declaration_receipt_hash: str
    action_class_proof: V1FiniteActionClassProof | None = None
    schema: str = _DOMAIN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "route_axis_hash",
            _nonempty(self.route_axis_hash, "route_axis_hash"),
        )
        route_count = _positive_int(self.route_count, "route_count")
        if self.schema != _DOMAIN_SCHEMA:
            raise ValueError("unsupported modern-v1 finite-removal-domain schema")
        normalized: list[
            tuple[int, ...] | RemovedRouteIndexSet | V1PackedRemovalAction
        ] = []
        action_hashes: list[str] = []
        for action_index, removal_set in enumerate(tuple(self.removal_sets)):
            if type(removal_set) is V1PackedRemovalAction:
                if (
                    removal_set.resulting_mask.route_count != route_count
                    or removal_set.resulting_mask.route_axis_hash != self.route_axis_hash
                ):
                    raise ValueError(
                        "packed removal action belongs to another physical route axis"
                    )
                normalized.append(removal_set)
                action_hashes.append(removal_set.removed_route_index_hash)
                continue
            if type(removal_set) is RemovedRouteIndexSet:
                if removal_set.route_count != route_count:
                    raise ValueError(
                        "compact removal set belongs to another physical route axis"
                    )
                if removal_set.count <= 0:
                    raise ValueError("finite removal-domain actions must be nonempty")
                # Preserve the historical tuple representation and content hash
                # for small declared actions.  Large actions remain compact.
                normalized_action: tuple[int, ...] | RemovedRouteIndexSet = (
                    removal_set.to_tuple()
                    if removal_set.count <= _INLINE_ACTION_LIMIT
                    else removal_set
                )
                action_hash = removal_set.index_hash
                normalized.append(normalized_action)
                action_hashes.append(action_hash)
                continue
            if isinstance(removal_set, (str, bytes)):
                raise TypeError("removal sets must contain integer indices")
            values = tuple(removal_set)
            if not values:
                raise ValueError("finite removal-domain actions must be nonempty")
            if any(isinstance(index, bool) or not isinstance(index, Integral) for index in values):
                raise TypeError(f"removal set {action_index} must contain exact integer indices")
            indices = tuple(int(index) for index in values)
            if len(set(indices)) != len(indices) or any(
                index < 0 or index >= route_count for index in indices
            ):
                raise ValueError("finite removal-domain actions require unique in-axis indices")
            ordered = tuple(sorted(indices))
            compact = RemovedRouteIndexSet.from_indices(
                ordered,
                route_count=route_count,
            )
            normalized.append(
                ordered if len(ordered) <= _INLINE_ACTION_LIMIT else compact
            )
            action_hashes.append(compact.index_hash)
        if not normalized:
            raise ValueError("finite removal domain must declare at least one action")
        declared = tuple(normalized)
        if len(set(action_hashes)) != len(action_hashes):
            raise ValueError("finite removal domain must not duplicate actions")
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "removal_sets", declared)
        if self.action_class_proof is not None and type(self.action_class_proof) is not (
            V1FiniteActionClassProof
        ):
            raise TypeError("action_class_proof must be an exact finite proof declaration")
        object.__setattr__(
            self,
            "declaration_receipt_hash",
            _nonempty(self.declaration_receipt_hash, "declaration_receipt_hash"),
        )

    @property
    def domain_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-removal-domain-")

    @property
    def action_class_hash(self) -> str | None:
        proof = self.action_class_proof
        return None if proof is None else proof.proof_hash

    @property
    def action_count(self) -> int:
        return len(self.removal_sets)

    @property
    def maximum_action_size(self) -> int:
        return max(len(action) for action in self.removal_sets)

    @property
    def has_compact_actions(self) -> bool:
        return any(
            type(action) in {RemovedRouteIndexSet, V1PackedRemovalAction}
            for action in self.removal_sets
        )

    @property
    def can_certify_decision(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class V1CandidateCompositionConstraints:
    """Frozen proposal-plane caps and their external source receipts."""

    maximum_removed_count: int
    minimum_resource_gain: float
    maximum_resource_gain: float
    maximum_proposal_damage: float
    maximum_fd_geometry_upper: float
    diagonal_hard_bound_mode: DiagonalHardBoundMode
    proposal_damage_cap_source_hash: str
    resource_cap_source_hash: str
    geometry_cap_source_hash: str
    constraints_source_receipt_hash: str
    schema: str = _CONSTRAINTS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _CONSTRAINTS_SCHEMA:
            raise ValueError("unsupported modern-v1 candidate-constraints schema")
        object.__setattr__(
            self,
            "maximum_removed_count",
            _positive_int(self.maximum_removed_count, "maximum_removed_count"),
        )
        for name in (
            "minimum_resource_gain",
            "maximum_resource_gain",
            "maximum_proposal_damage",
            "maximum_fd_geometry_upper",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )
        if self.maximum_resource_gain < self.minimum_resource_gain:
            raise ValueError("maximum resource gain cannot be below its minimum")
        if self.diagonal_hard_bound_mode not in {
            "proposal_ranking_only",
            "proved_chart_bound",
        }:
            raise ValueError("unsupported residual-diagonal hard-bound mode")
        for name in (
            "proposal_damage_cap_source_hash",
            "resource_cap_source_hash",
            "geometry_cap_source_hash",
            "constraints_source_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))

    @property
    def constraints_hash(self) -> str:
        return content_hash(
            self,
            prefix="partition-free-v1-candidate-constraints-",
        )

    @property
    def can_certify_decision(self) -> bool:
        return False


@dataclass(frozen=True, slots=True, init=False)
class V1CandidateNumericState:
    """Adapter-issued immutable numeric state for candidate evaluation."""

    route_ids: tuple[str, ...] | V1RouteIdAxisIdentity
    live_route_ids: tuple[str, ...] | V1RouteIdAxisIdentity
    parent_mask: HardMask
    effective_factor: np.ndarray
    residual_diagonal: np.ndarray
    corrected_prices: np.ndarray
    atom_costs: np.ndarray
    fd_shrink: float
    price_orientation_id: str
    surrogate_equation_id: str
    fd_spectral_upper_equation_id: str
    fresh_relinearization_required: bool
    chart_hash: str
    registration_hash: str
    source_geometry_state_hash: str
    fd_adapter_result_hash: str
    fd_lineage_hash: str
    fd_verification_receipt_hash: str
    correction_lineage_hash: str
    correction_result_receipt_hash: str
    correction_verification_receipt_hash: str
    atom_cost_state_hash: str
    route_axis_hash: str
    live_route_axis_hash: str
    parent_mask_hash: str
    effective_factor_hash: str
    source_exact_residual_diagonal_hash: str
    registered_residual_diagonal_hash: str
    corrected_price_hash: str
    atom_cost_hash: str
    fd_shrink_receipt_hash: str
    residual_diagonal_status: str
    residual_diagonal_receipt_hash: str | None
    residual_bound_proof_hash: str | None
    proved_action_class: V1FiniteActionClassProof | None
    referent_id: str
    context_mixture_hash: str
    calibration_identity: str
    schema: str = _NUMERIC_STATE_SCHEMA

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _NUMERIC_STATE_TOKEN:
            raise TypeError("V1CandidateNumericState can only be issued by its source validator")
        expected = tuple(name for name in self.__dataclass_fields__ if name != "schema")
        if set(values) != set(expected):
            missing = sorted(set(expected).difference(values))
            extra = sorted(set(values).difference(expected))
            raise TypeError(f"numeric-state fields differ: missing={missing}, extra={extra}")
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", _NUMERIC_STATE_SCHEMA)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.schema != _NUMERIC_STATE_SCHEMA:
            raise ValueError("unsupported modern-v1 candidate numeric-state schema")
        compact_routes = type(self.route_ids) is V1RouteIdAxisIdentity
        compact_live_routes = type(self.live_route_ids) is V1RouteIdAxisIdentity
        if compact_routes != compact_live_routes:
            raise ValueError("candidate route-ID axes mix inline and compact identities")
        if compact_routes:
            routes = self.route_ids
            live_routes = self.live_route_ids
            route_id_hash = routes.ordered_hash
            live_route_id_hash = live_routes.ordered_hash
        else:
            routes = canonical_unique_string_axis(
                self.route_ids,  # type: ignore[arg-type]
                name="modern-v1 candidate route axis",
            )
            live_routes = canonical_unique_string_axis(
                self.live_route_ids,  # type: ignore[arg-type]
                name="modern-v1 candidate live route axis",
            )
            route_id_hash = ordered_string_axis_hash(
                routes,
                name="modern-v1 candidate route axis",
            )
            live_route_id_hash = ordered_string_axis_hash(
                live_routes,
                name="modern-v1 candidate live route axis",
            )
        if type(self.parent_mask) is not HardMask:
            raise TypeError("candidate numeric state requires an exact HardMask")
        route_count = len(routes)
        if self.parent_mask.route_count != route_count:
            raise ValueError("candidate parent mask has the wrong route count")
        if (
            self.effective_factor.ndim != 2
            or self.effective_factor.shape[1] != route_count
            or self.residual_diagonal.shape != (route_count,)
            or self.corrected_prices.shape != (route_count,)
            or self.atom_costs.shape != (route_count,)
        ):
            raise ValueError("candidate numeric arrays do not share the full route axis")
        for array in (
            self.effective_factor,
            self.residual_diagonal,
            self.corrected_prices,
            self.atom_costs,
        ):
            if not _is_bytes_backed_readonly(array):
                raise ValueError("candidate numeric arrays must be bytes-backed snapshots")
        if self.price_orientation_id != REMOVAL_INDICATOR_POSITIVE_DAMAGE_V1:
            raise ValueError("candidate prices have an unsupported orientation")
        if self.surrogate_equation_id != NO_HALF_SURROGATE_EQUATION_ID:
            raise ValueError("candidate state changed the no-half surrogate equation")
        if self.fd_spectral_upper_equation_id != FD_SPECTRAL_UPPER_EQUATION_ID:
            raise ValueError("candidate state changed the FD spectral-upper equation")
        if type(self.fresh_relinearization_required) is not bool or not (
            self.fresh_relinearization_required
        ):
            raise ValueError("candidate state requires a fresh relinearization chart")
        if self.residual_diagonal_status not in {
            "absent",
            "proposal_only",
            "global_loewner_bound",
            "action_class_quadratic_bound",
        }:
            raise ValueError("candidate state has an unsupported residual-diagonal status")
        if self.proved_action_class is not None and type(self.proved_action_class) is not (
            V1FiniteActionClassProof
        ):
            raise TypeError("proved_action_class must be an exact finite proof declaration")
        for name in (
            "chart_hash",
            "registration_hash",
            "source_geometry_state_hash",
            "fd_adapter_result_hash",
            "fd_lineage_hash",
            "fd_verification_receipt_hash",
            "correction_lineage_hash",
            "correction_result_receipt_hash",
            "correction_verification_receipt_hash",
            "atom_cost_state_hash",
            "route_axis_hash",
            "live_route_axis_hash",
            "parent_mask_hash",
            "effective_factor_hash",
            "source_exact_residual_diagonal_hash",
            "registered_residual_diagonal_hash",
            "corrected_price_hash",
            "atom_cost_hash",
            "fd_shrink_receipt_hash",
            "referent_id",
            "context_mixture_hash",
            "calibration_identity",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "fd_shrink",
            _finite_nonnegative(self.fd_shrink, "fd_shrink"),
        )
        if self.parent_mask.mask_hash != self.parent_mask_hash:
            raise ValueError("candidate parent mask hash is not reproducible")
        if compact_routes:
            if len(live_routes) != self.parent_mask.retained_count:
                raise ValueError("candidate compact live route-ID count differs from the mask")
        else:
            live_flags = self.parent_mask.to_bool_array()
            expected_live = (
                routes
                if self.parent_mask.retained_count == route_count
                else tuple(
                    route
                    for route, live in zip(routes, live_flags, strict=True)
                    if live
                )
            )
            if live_routes != expected_live:
                raise ValueError("candidate live route IDs differ from the parent mask")
        if route_id_hash != self.route_axis_hash:
            raise ValueError("candidate route-ID-axis hash is not reproducible")
        if live_route_id_hash != self.live_route_axis_hash:
            raise ValueError("candidate live route-axis hash is not reproducible")
        hash_checks = (
            (
                content_hash(
                    self.effective_factor,
                    prefix="partition-free-v1-effective-fd-factor-",
                ),
                self.effective_factor_hash,
                "effective FD factor",
            ),
            (
                content_hash(
                    self.residual_diagonal,
                    prefix="partition-free-v1-registered-residual-diagonal-values-",
                ),
                self.registered_residual_diagonal_hash,
                "registered residual diagonal",
            ),
            (
                content_hash(
                    self.corrected_prices,
                    prefix="partition-free-v1-corrected-price-values-",
                ),
                self.corrected_price_hash,
                "corrected prices",
            ),
            (
                content_hash(
                    self.atom_costs,
                    prefix="partition-free-v1-atom-cost-values-",
                ),
                self.atom_cost_hash,
                "atom costs",
            ),
        )
        for observed, expected, label in hash_checks:
            if observed != expected:
                raise ValueError(f"candidate {label} hash is not reproducible")
        if self.residual_diagonal_status == "absent":
            if self.residual_diagonal_receipt_hash is not None or bool(
                np.any(self.residual_diagonal != 0.0)
            ):
                raise ValueError("absent candidate residual diagonal is contradictory")
        else:
            if self.residual_diagonal_receipt_hash != self.source_exact_residual_diagonal_hash:
                raise ValueError("candidate residual diagonal receipt differs from source")
        if (
            self.residual_diagonal_status
            in {"global_loewner_bound", "action_class_quadratic_bound"}
            and not self.residual_bound_proof_hash
        ):
            raise ValueError("candidate residual bound status lacks its proof hash")
        if self.residual_diagonal_status == "action_class_quadratic_bound":
            proof = self.proved_action_class
            if proof is None:
                raise ValueError("action-class residual bound lacks its finite proof declaration")
            if proof.proof_artifact_hash != self.residual_bound_proof_hash:
                raise ValueError("action-class declaration names another residual proof")
            if (
                proof.route_axis_hash != self.route_axis_hash
                or proof.route_count != route_count
                or proof.base_mask_hash != self.parent_mask_hash
                or proof.current_mask_hash != self.parent_mask_hash
            ):
                raise ValueError("action-class declaration differs from candidate chart identity")
        elif self.proved_action_class is not None:
            raise ValueError("non-action-class residual status cannot carry an action class")
        object.__setattr__(self, "route_ids", routes)
        object.__setattr__(self, "live_route_ids", live_routes)

    @property
    def state_hash(self) -> str:
        return content_hash(self, prefix="partition-free-v1-candidate-numeric-state-")

    @property
    def route_id_axis_hash(self) -> str:
        """Ordered human-readable/ordinal ID axis used by numeric producers."""

        return self.route_axis_hash

    @property
    def physical_route_axis_hash(self) -> str:
        """Executable block-layout axis carried by the exact hard mask."""

        return self.parent_mask.route_axis_hash

    @property
    def can_certify_decision(self) -> bool:
        return False

    def assert_can_certify_decision(self) -> None:
        raise PermissionError("candidate numeric state is proposal-only")


def _validate_fd_source(
    *,
    chart: V1ProposalChart,
    state: GlobalTrustGeometryState,
    fd_result: V1FDAdapterResult,
) -> tuple[np.ndarray, np.ndarray, str, str, str]:
    if type(state) is not GlobalTrustGeometryState:
        raise TypeError("candidate state requires an exact GlobalTrustGeometryState")
    if type(state.spec) is not GlobalTrustGeometrySpec:
        raise TypeError("candidate geometry state has a noncanonical spec")
    if state.schema != GLOBAL_TRUST_GEOMETRY_STATE_SCHEMA:
        raise ValueError("unsupported global trust geometry schema")
    if type(fd_result) is not V1FDAdapterResult:
        raise TypeError("candidate state requires an exact V1FDAdapterResult")
    if fd_result.schema != "partition_free_v1_fd_adapter_result_v1":
        raise ValueError("unsupported modern-v1 FD adapter-result schema")
    lineage = fd_result.lineage
    verification = fd_result.verification_receipt
    if type(lineage) is not V1FrequentDirectionsLineage:
        raise TypeError("FD adapter result contains a noncanonical lineage")
    if chart.fd_lineage.lineage_hash != lineage.lineage_hash:
        raise ValueError("candidate chart and FD adapter lineage hashes differ")
    if chart.fd_verification_receipt.receipt_hash != verification.receipt_hash:
        raise ValueError("candidate chart and FD verification receipts differ")
    if verification.lineage_hash != lineage.lineage_hash:
        raise ValueError("FD verification receipt names another lineage")
    if (
        verification.proposal_only is not True
        or verification.claim_execution_certified is not False
        or fd_result.can_certify_decision
        or lineage.can_certify_decision
    ):
        raise PermissionError("certifying FD state cannot enter proposal composition")

    receipt = state.derivation_receipt
    if type(receipt) is not GlobalTrustGeometryDerivationReceipt:
        raise ValueError("candidate geometry requires its accumulator derivation receipt")
    arrays = (
        state.mean_component,
        state.centered_factor,
        state.residual_diagonal_bound,
    )
    if state.residual_diagonal_bound is None or any(
        array is None or array.flags.writeable for array in arrays
    ):
        raise ValueError("candidate geometry source arrays must be immutable snapshots")
    source_hash = _fresh_global_trust_state_hash(state)
    if source_hash != state.content_hash:
        raise ValueError("global trust state changed after its hash was observed")
    if (
        verification.source_state_hash != source_hash
        or lineage.metadata.get("source_state_hash") != source_hash
    ):
        raise ValueError("FD adapter result is bound to another geometry state")
    if verification.accumulator_derivation_receipt_hash != receipt.receipt_hash:
        raise ValueError("FD verification receipt is bound to another accumulator receipt")

    route_hash = ordered_string_axis_hash(
        state.route_ids,
        name="modern-v1 candidate route axis",
    )
    if route_hash != lineage.route_axis_hash or route_hash != chart.route_axis_hash:
        raise ValueError("candidate geometry ordered route-axis hash mismatch")
    if state.measured_mask_hash != chart.base_mask_hash:
        raise ValueError("candidate geometry was measured at another chart base mask")
    if state.calibration_state_id != chart.calibration_identity:
        raise ValueError("candidate geometry calibration identity mismatch")

    mean = np.asarray(state.mean_component, dtype=np.float64)
    centered = np.asarray(state.centered_factor, dtype=np.float64)
    source_diagonal = np.asarray(
        state.residual_diagonal_bound,
        dtype=np.float64,
    ).reshape(-1)
    effective = _bytes_matrix(
        np.vstack((mean, centered)),
        name="candidate effective FD factor",
    )
    source_diagonal_snapshot = _bytes_vector(
        source_diagonal,
        name="candidate source residual diagonal",
        nonnegative=True,
    )
    factor_hash = content_hash(
        effective,
        prefix="partition-free-v1-effective-fd-factor-",
    )
    if factor_hash != lineage.factor_hash:
        raise ValueError("recomputed effective FD factor hash differs from its lineage")
    diagonal_hash = content_hash(
        source_diagonal_snapshot,
        prefix="partition-free-v1-fd-exact-residual-diagonal-",
    )
    if lineage.metadata.get("exact_residual_diagonal_hash") != diagonal_hash:
        raise ValueError("recomputed residual-diagonal hash differs from its lineage")
    fd_shrink = _finite_nonnegative(state.residual_spectral_bound, "fd_shrink")
    if fd_shrink != lineage.fd_shrink:
        raise ValueError("recomputed FD shrink differs from its lineage")

    mean_hash = content_hash(mean, prefix="partition-free-v1-fd-mean-")
    centered_hash = content_hash(
        centered,
        prefix="partition-free-v1-fd-centered-factor-",
    )
    source_ledger_hash = verification.source_ledger_hash
    factor_receipt_hash = content_hash(
        {
            "schema": "partition_free_v1_fd_factor_derivation_receipt_v1",
            "derivation_id": EFFECTIVE_FACTOR_DERIVATION_ID,
            "source_state_hash": source_hash,
            "source_ledger_hash": source_ledger_hash,
            "accumulator_derivation_receipt_hash": receipt.receipt_hash,
            "ordered_source_rows_hash": receipt.ordered_source_rows_hash,
            "ordered_route_axis_hash": route_hash,
            "mean_component_hash": mean_hash,
            "centered_factor_hash": centered_hash,
            "effective_factor_hash": factor_hash,
            "mean_component_rows": int(mean.shape[0]),
            "centered_fd_sketch_rank": int(centered.shape[0]),
            "effective_factor_rows": int(effective.shape[0]),
            "route_count": len(state.route_ids),
        },
        prefix="partition-free-v1-fd-factor-receipt-",
    )
    if (
        factor_receipt_hash != lineage.factor_receipt_hash
        or factor_receipt_hash != verification.factor_derivation_receipt_hash
    ):
        raise ValueError("recomputed FD factor receipt hash differs")
    shrink_receipt_hash = content_hash(
        {
            "schema": "partition_free_v1_fd_shrink_derivation_receipt_v1",
            "derivation_id": FD_SHRINK_DERIVATION_ID,
            "source_state_hash": source_hash,
            "source_ledger_hash": source_ledger_hash,
            "normalized_fd_shrink": fd_shrink,
            "producer_total_weight": float(state.total_weight),
            "unnormalized_cumulative_fd_shrink": (fd_shrink * float(state.total_weight)),
            "ordered_route_axis_hash": route_hash,
        },
        prefix="partition-free-v1-fd-shrink-receipt-",
    )
    if (
        shrink_receipt_hash != lineage.fd_shrink_receipt_hash
        or shrink_receipt_hash != verification.fd_shrink_derivation_receipt_hash
    ):
        raise ValueError("recomputed FD shrink receipt hash differs")

    mean_energy = float(np.sum(mean * mean, dtype=np.float64))
    centered_energy = float(np.sum(centered * centered, dtype=np.float64))
    residual_trace = float(np.sum(source_diagonal_snapshot, dtype=np.float64))
    exact_energy = math.fsum((mean_energy, centered_energy, residual_trace))
    exact_norm = math.sqrt(exact_energy)
    exact_norm_receipt_hash = content_hash(
        {
            "schema": "partition_free_v1_fd_exact_norm_derivation_receipt_v1",
            "derivation_id": EXACT_STREAMED_NORM_DERIVATION_ID,
            "source_state_hash": source_hash,
            "source_ledger_hash": source_ledger_hash,
            "mean_component_hash": mean_hash,
            "centered_factor_hash": centered_hash,
            "exact_residual_diagonal_hash": diagonal_hash,
            "mean_squared_frobenius": mean_energy,
            "centered_factor_squared_frobenius": centered_energy,
            "exact_residual_trace": residual_trace,
            "exact_streamed_squared_frobenius": exact_energy,
            "exact_streamed_frobenius_norm": exact_norm,
            "ordered_route_axis_hash": route_hash,
        },
        prefix="partition-free-v1-fd-exact-norm-receipt-",
    )
    if (
        exact_norm_receipt_hash != lineage.exact_streamed_norm_receipt_hash
        or exact_norm_receipt_hash != verification.exact_norm_derivation_receipt_hash
        or not math.isclose(
            exact_norm,
            lineage.exact_streamed_norm,
            rel_tol=1e-14,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("recomputed exact streamed-norm receipt differs")
    verified_algebra_hash = content_hash(
        {
            "schema": "partition_free_v1_fd_verified_algebra_v1",
            "source_state_hash": source_hash,
            "source_ledger_hash": source_ledger_hash,
            "accumulator_derivation_receipt_hash": receipt.receipt_hash,
            "lineage_hash": lineage.lineage_hash,
            "ordered_route_axis_hash": route_hash,
            "ordered_source_rows_hash": receipt.ordered_source_rows_hash,
            "stream_weight_hash": receipt.stream_weight_hash,
            "mean_component_hash": mean_hash,
            "centered_factor_hash": centered_hash,
            "exact_residual_diagonal_hash": diagonal_hash,
            "factor_hash": factor_hash,
            "fd_shrink": fd_shrink,
            "exact_streamed_norm": exact_norm,
            "factor_rows": int(effective.shape[0]),
            "route_count": len(state.route_ids),
        },
        prefix="partition-free-v1-fd-verified-algebra-",
    )
    if verified_algebra_hash != verification.verified_algebra_hash:
        raise ValueError("recomputed FD verified-algebra hash differs")
    return effective, source_diagonal_snapshot, source_hash, factor_hash, diagonal_hash


def _validate_correction_source(
    *,
    chart: V1ProposalChart,
    correction_result: (
        V1InterventionalCorrectionResult
        | V1AnalyticExactGateCorrectionResult
    ),
) -> tuple[np.ndarray, str, str]:
    if type(correction_result) is V1AnalyticExactGateCorrectionResult:
        lineage = correction_result.lineage
        verification = chart.correction_verification_receipt
        if (
            type(lineage) is not V1AnalyticExactGateCorrectionLineage
            or type(verification)
            is not V1AnalyticExactGateCorrectionVerificationReceipt
            or chart.correction_lineage.lineage_hash != lineage.lineage_hash
            or verification.lineage_hash != lineage.lineage_hash
            or verification.proposal_only is not True
            or verification.claim_execution_certified is not False
            or correction_result.can_certify_decision
            or correction_result.claim_execution_certified
            or lineage.can_certify_decision
        ):
            raise PermissionError("analytic exact-gate correction lacks proposal proof")
        for array in (
            correction_result.corrected_prices,
            correction_result.absolute_salience,
        ):
            if not _is_bytes_backed_readonly(array) or not np.isfinite(array).all():
                raise ValueError(
                    "analytic exact-gate arrays must be finite bytes-backed snapshots"
                )
        corrected = _bytes_vector(
            correction_result.corrected_prices,
            name="analytic exact-gate candidate prices",
        )
        corrected_hash = content_hash(
            corrected,
            prefix="partition-free-v1-corrected-price-values-",
        )
        absolute_hash = content_hash(
            correction_result.absolute_salience,
            prefix="partition-free-v1-exact-gate-absolute-salience-values-",
        )
        expected_result_hash = content_hash(
            {
                "schema": correction_result.schema,
                "lineage_hash": lineage.lineage_hash,
                "corrected_price_hash": corrected_hash,
                "absolute_salience_hash": absolute_hash,
                "analytic_identity_no_fitted_residual": True,
            },
            prefix="partition-free-v1-analytic-exact-gate-result-",
        )
        verified_algebra_hash = content_hash(
            {
                "schema": "partition_free_v1_analytic_exact_gate_algebra_check_v1",
                "lineage_hash": lineage.lineage_hash,
                "result_receipt_hash": correction_result.result_receipt_hash,
                "corrected_price_values_hash": corrected_hash,
                "absolute_salience_values_hash": absolute_hash,
                "source_attribution_hash": lineage.source_attribution_hash,
                "streaming_receipt_hash": lineage.streaming_receipt_hash,
                "raw_target_logit_unit_cotangent": True,
                "empirical_child_parent_mask_evaluation_count": 0,
            },
            prefix="partition-free-v1-analytic-exact-gate-algebra-check-",
        )
        if (
            corrected_hash != lineage.corrected_price_hash
            or absolute_hash != lineage.absolute_salience_hash
            or expected_result_hash != correction_result.result_receipt_hash
            or verification.result_receipt_hash
            != correction_result.result_receipt_hash
            or verification.corrected_price_values_hash != corrected_hash
            or verification.absolute_salience_values_hash != absolute_hash
            or verification.source_attribution_hash
            != lineage.source_attribution_hash
            or verification.streaming_receipt_hash
            != lineage.streaming_receipt_hash
            or verification.verified_algebra_hash != verified_algebra_hash
        ):
            raise ValueError("analytic exact-gate correction proof does not reproduce")
        return corrected, corrected_hash, correction_result.price_orientation_id
    if type(correction_result) is not V1InterventionalCorrectionResult:
        raise TypeError("candidate state requires an exact V1InterventionalCorrectionResult")
    if correction_result.schema != "partition_free_v1_interventional_correction_result_v1":
        raise ValueError("unsupported modern-v1 correction-result schema")
    lineage = correction_result.lineage
    if type(lineage) is not V1InterventionalCorrectionLineage:
        raise TypeError("correction result contains a noncanonical lineage")
    verification = chart.correction_verification_receipt
    if type(verification) is not V1InterventionalCorrectionVerificationReceipt:
        raise TypeError("candidate chart lacks a correction verification receipt")
    if chart.correction_lineage.lineage_hash != lineage.lineage_hash:
        raise ValueError("candidate chart and correction lineage hashes differ")
    if verification.lineage_hash != lineage.lineage_hash:
        raise ValueError("candidate correction verification names another lineage")
    if (
        verification.proposal_only is not True
        or verification.claim_execution_certified is not False
        or correction_result.can_certify_decision
        or correction_result.claim_execution_certified
        or lineage.can_certify_decision
    ):
        raise PermissionError("certifying correction results cannot enter proposals")
    for array in (
        correction_result.corrected_prices,
        correction_result.correction_coefficients,
        correction_result.design_matrix,
        correction_result.observed_action_residuals,
        correction_result.fitted_action_residuals,
    ):
        if not _is_bytes_backed_readonly(array):
            raise ValueError("correction result arrays must be bytes-backed snapshots")
        if not np.isfinite(array).all():
            raise ValueError("correction result arrays must remain finite")
    corrected = _bytes_vector(
        correction_result.corrected_prices,
        name="candidate corrected prices",
    )
    corrected_hash = content_hash(
        corrected,
        prefix="partition-free-v1-corrected-price-values-",
    )
    if corrected_hash != lineage.corrected_price_hash:
        raise ValueError("recomputed corrected-price hash differs from its lineage")
    fitted = np.asarray(correction_result.design_matrix) @ np.asarray(
        correction_result.correction_coefficients
    )
    if not np.allclose(
        fitted,
        correction_result.fitted_action_residuals,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("correction result violates its fitted-design algebra")
    result_receipt_hash = content_hash(
        {
            "lineage_hash": lineage.lineage_hash,
            "corrected_price_hash": corrected_hash,
            "correction_coefficients": correction_result.correction_coefficients,
            "observed_action_residuals": (correction_result.observed_action_residuals),
            "fitted_action_residuals": correction_result.fitted_action_residuals,
        },
        prefix="partition-free-v1-correction-result-",
    )
    if result_receipt_hash != correction_result.result_receipt_hash:
        raise ValueError("recomputed correction result receipt differs")
    orientation = correction_result.price_orientation_id
    tangent_state_hash = lineage.metadata.get("tangent_price_state_hash")
    if type(tangent_state_hash) is not str or not tangent_state_hash:
        raise ValueError("correction lineage lacks its tangent-price state hash")
    corrected_receipt_hash = content_hash(
        {
            "tangent_price_state_hash": tangent_state_hash,
            "feature_map_hash": lineage.feature_map_hash,
            "identification_receipt_hash": lineage.identification_receipt_hash,
            "correction_coefficients": correction_result.correction_coefficients,
            "corrected_price_hash": corrected_hash,
            "price_orientation_id": orientation,
        },
        prefix="partition-free-v1-corrected-price-receipt-",
    )
    if corrected_receipt_hash != lineage.corrected_price_receipt_hash:
        raise ValueError("recomputed corrected-price receipt differs from its lineage")

    value_hashes = {
        "corrected_price_values_hash": corrected_hash,
        "design_matrix_values_hash": content_hash(
            correction_result.design_matrix,
            prefix="partition-free-v1-correction-design-values-",
        ),
        "correction_coefficients_hash": content_hash(
            correction_result.correction_coefficients,
            prefix="partition-free-v1-correction-coefficients-",
        ),
        "observed_action_residuals_hash": content_hash(
            correction_result.observed_action_residuals,
            prefix="partition-free-v1-correction-observed-residuals-",
        ),
        "fitted_action_residuals_hash": content_hash(
            correction_result.fitted_action_residuals,
            prefix="partition-free-v1-correction-fitted-residuals-",
        ),
    }
    for name, expected in value_hashes.items():
        if getattr(verification, name) != expected:
            raise ValueError(f"correction verification hash mismatch for {name}")
    if verification.adapter_result_receipt_hash != result_receipt_hash:
        raise ValueError("correction verification names another adapter result")
    verified_algebra_hash = content_hash(
        {
            "schema": "partition_free_v1_correction_algebra_check_v1",
            "lineage_hash": lineage.lineage_hash,
            "adapter_result_receipt_hash": result_receipt_hash,
            **value_hashes,
            "fitted_equals_design_times_coefficients": True,
        },
        prefix="partition-free-v1-correction-algebra-check-",
    )
    if verified_algebra_hash != verification.verified_algebra_hash:
        raise ValueError("recomputed correction verified-algebra hash differs")
    return corrected, corrected_hash, orientation


def issue_v1_candidate_numeric_state(
    *,
    chart: V1ProposalChart,
    global_geometry_state: GlobalTrustGeometryState,
    fd_result: V1FDAdapterResult,
    correction_result: (
        V1InterventionalCorrectionResult
        | V1AnalyticExactGateCorrectionResult
    ),
    atom_costs: V1AtomCostState,
    current_mask: HardMask,
) -> V1CandidateNumericState:
    """Revalidate every producer and issue one immutable proposal snapshot."""

    if type(chart) is not V1ProposalChart:
        raise TypeError("candidate state requires an exact V1ProposalChart")
    if type(current_mask) is not HardMask:
        raise TypeError("candidate state requires an exact current HardMask")
    if type(atom_costs) is not V1AtomCostState:
        raise TypeError("candidate state requires an exact V1AtomCostState")
    if chart.can_certify_decision:
        raise PermissionError("a certifying chart cannot enter proposal composition")
    if chart.base_mask_hash != chart.current_mask_hash:
        raise ValueError(
            "candidate issuance requires a fresh relinearization chart at the current mask"
        )

    effective, source_diagonal, source_hash, factor_hash, source_diagonal_hash = (
        _validate_fd_source(
            chart=chart,
            state=global_geometry_state,
            fd_result=fd_result,
        )
    )
    corrected, corrected_hash, orientation = _validate_correction_source(
        chart=chart,
        correction_result=correction_result,
    )
    routes = canonical_unique_string_axis(
        global_geometry_state.route_ids,
        name="modern-v1 candidate route axis",
    )
    route_hash = ordered_string_axis_hash(
        routes,
        name="modern-v1 candidate route axis",
    )
    if (
        len(atom_costs.route_ids) != len(routes)
        or atom_costs.route_axis_hash != route_hash
        or correction_result.lineage.route_axis_hash != route_hash
    ):
        raise ValueError("candidate price/cost/geometry route axes differ")
    if current_mask.route_count != len(routes):
        raise ValueError("current hard mask route count differs from the numeric route axis")
    if current_mask.mask_hash != chart.current_mask_hash:
        raise ValueError("current hard mask differs from the chart parent identity")
    live_flags = current_mask.to_bool_array()
    live_count = current_mask.retained_count
    if live_count <= 0:
        raise ValueError("candidate composition requires at least one live route")
    live_hash = _selected_route_id_axis_hash(routes, live_flags)
    if live_hash != chart.live_route_axis_hash:
        raise ValueError("recomputed live route-axis hash differs from the chart")
    if len(routes) <= _INLINE_ROUTE_ID_AXIS_LIMIT:
        route_axis_state: tuple[str, ...] | V1RouteIdAxisIdentity = routes
        live_route_axis_state: tuple[str, ...] | V1RouteIdAxisIdentity = (
            routes
            if live_count == len(routes)
            else tuple(
                route
                for route, live in zip(routes, live_flags, strict=True)
                if bool(live)
            )
        )
    else:
        route_axis_state = V1RouteIdAxisIdentity(
            count=len(routes),
            ordered_hash=route_hash,
        )
        live_route_axis_state = V1RouteIdAxisIdentity(
            count=live_count,
            ordered_hash=live_hash,
        )

    if not _is_bytes_backed_readonly(atom_costs.costs):
        raise ValueError("atom costs must be a bytes-backed immutable snapshot")
    costs = _bytes_vector(
        atom_costs.costs,
        name="candidate atom costs",
        nonnegative=True,
    )
    if costs.shape != (len(routes),) or not bool(np.any(costs > 0.0)):
        raise ValueError("candidate atom costs must be aligned and not all zero")
    if not bool(np.any(costs[live_flags] > 0.0)):
        raise ValueError("all live candidate atom costs are zero")
    cost_hash = content_hash(
        costs,
        prefix="partition-free-v1-atom-cost-values-",
    )
    if (
        cost_hash != atom_costs.cost_hash
        or cost_hash != correction_result.lineage.atom_cost_hash
        or atom_costs.state_hash != correction_result.lineage.atom_cost_receipt_hash
    ):
        raise ValueError("recomputed atom-cost hashes differ from correction lineage")

    status = chart.residual_diagonal_status
    residual_receipt = chart.build_contract.residual_diagonal_receipt_hash
    if status == "absent":
        registered_diagonal = _bytes_vector(
            np.zeros(len(routes), dtype=np.float64),
            name="absent registered residual diagonal",
            nonnegative=True,
        )
    else:
        if residual_receipt != source_diagonal_hash:
            raise ValueError("chart residual-diagonal receipt does not bind the source values")
        registered_diagonal = _bytes_vector(
            source_diagonal,
            name="registered residual diagonal",
            nonnegative=True,
        )
    registered_diagonal_hash = content_hash(
        registered_diagonal,
        prefix="partition-free-v1-registered-residual-diagonal-values-",
    )
    chart_hash = chart.chart_hash
    return V1CandidateNumericState(
        _issuer_token=_NUMERIC_STATE_TOKEN,
        route_ids=route_axis_state,
        live_route_ids=live_route_axis_state,
        parent_mask=current_mask,
        effective_factor=effective,
        residual_diagonal=registered_diagonal,
        corrected_prices=corrected,
        atom_costs=costs,
        fd_shrink=float(global_geometry_state.residual_spectral_bound),
        price_orientation_id=orientation,
        surrogate_equation_id=NO_HALF_SURROGATE_EQUATION_ID,
        fd_spectral_upper_equation_id=FD_SPECTRAL_UPPER_EQUATION_ID,
        fresh_relinearization_required=True,
        chart_hash=chart_hash,
        registration_hash=chart.registration_hash,
        source_geometry_state_hash=source_hash,
        fd_adapter_result_hash=fd_result.result_hash,
        fd_lineage_hash=fd_result.lineage.lineage_hash,
        fd_verification_receipt_hash=fd_result.verification_receipt.receipt_hash,
        correction_lineage_hash=correction_result.lineage.lineage_hash,
        correction_result_receipt_hash=correction_result.result_receipt_hash,
        correction_verification_receipt_hash=(chart.correction_verification_receipt.receipt_hash),
        atom_cost_state_hash=atom_costs.state_hash,
        route_axis_hash=route_hash,
        live_route_axis_hash=live_hash,
        parent_mask_hash=current_mask.mask_hash,
        effective_factor_hash=factor_hash,
        source_exact_residual_diagonal_hash=source_diagonal_hash,
        registered_residual_diagonal_hash=registered_diagonal_hash,
        corrected_price_hash=corrected_hash,
        atom_cost_hash=cost_hash,
        fd_shrink_receipt_hash=fd_result.lineage.fd_shrink_receipt_hash,
        residual_diagonal_status=status,
        residual_diagonal_receipt_hash=residual_receipt,
        residual_bound_proof_hash=chart.build_contract.residual_bound_proof_hash,
        proved_action_class=(
            chart.build_contract.mixed_difference_envelope.action_class_proof
            if status == "action_class_quadratic_bound"
            and chart.build_contract.mixed_difference_envelope is not None
            else None
        ),
        referent_id=chart.referent_id,
        context_mixture_hash=chart.context_mixture_hash,
        calibration_identity=chart.calibration_identity,
    )


@dataclass(frozen=True, slots=True)
class V1CandidateProposal:
    """One complete-mask proposal with exact registered-surrogate values."""

    candidate_id: str
    numeric_state_hash: str
    domain_hash: str
    constraints_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    removed_routes: RemovedRouteIndexSet | V1RemovedRouteSetIdentity
    removed_route_ids: tuple[str, ...]
    resulting_mask: HardMask
    resource_gain: float
    corrected_linear_damage: float
    fd_quadratic: float
    residual_diagonal_damage: float
    predicted_surrogate_damage: float
    proposal_damage_constraint_value: float
    fd_spectral_upper: float
    diagonal_hard_bound_used: bool
    surrogate_equation_id: str
    fd_spectral_upper_equation_id: str
    evaluation_receipt_hash: str
    proposal_only: bool = True
    proposal_witness_only: bool = True
    fresh_relinearization_required: bool = True
    exact_registered_surrogate_evaluation: bool = True
    exact_intervention_damage_evaluated: bool = False
    claim_execution_certified: bool = False
    schema: str = _PROPOSAL_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "numeric_state_hash",
            "domain_hash",
            "constraints_hash",
            "parent_mask_hash",
            "route_axis_hash",
            "evaluation_receipt_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if type(self.removed_routes) not in {
            RemovedRouteIndexSet,
            V1RemovedRouteSetIdentity,
        }:
            raise TypeError("candidate proposal has an unsupported route-delta identity")
        if type(self.resulting_mask) is not HardMask:
            raise TypeError("candidate proposal requires an exact complete HardMask")
        removed_count = self.removed_routes.count
        if self.resulting_mask.route_axis_hash != self.route_axis_hash:
            raise ValueError("candidate proposal mask and removal-set axes differ")
        if self.removed_routes.route_count != self.resulting_mask.route_count:
            raise ValueError("candidate proposal compact delta identity is inconsistent")
        if (
            removed_count <= _INLINE_ACTION_LIMIT
            and type(self.removed_routes) is not RemovedRouteIndexSet
        ):
            raise ValueError("small candidate proposal must retain its exact route delta")
        if (
            removed_count > _INLINE_ACTION_LIMIT
            and type(self.removed_routes) is not V1RemovedRouteSetIdentity
        ):
            raise ValueError("large candidate proposal must not retain its full route delta")
        route_ids = tuple(self.removed_route_ids)
        if route_ids and (
            len(route_ids) != removed_count or any(not item for item in route_ids)
        ):
            raise ValueError(
                "inlined candidate route IDs must name the complete removed set"
            )
        if removed_count > _INLINE_ACTION_LIMIT and route_ids:
            raise ValueError(
                "large candidate proposal must not materialize removed route IDs"
            )
        object.__setattr__(self, "removed_route_ids", route_ids)
        for name in (
            "resource_gain",
            "corrected_linear_damage",
            "fd_quadratic",
            "residual_diagonal_damage",
            "predicted_surrogate_damage",
            "proposal_damage_constraint_value",
            "fd_spectral_upper",
        ):
            number = float(getattr(self, name))
            if not math.isfinite(number):
                raise ValueError(f"candidate {name} must be finite")
            object.__setattr__(self, name, number)
        if self.resource_gain < 0.0 or self.fd_quadratic < 0.0 or self.fd_spectral_upper < 0.0:
            raise ValueError("candidate resource and geometry values must be nonnegative")
        if type(self.diagonal_hard_bound_used) is not bool:
            raise TypeError("diagonal_hard_bound_used must be bool")
        if self.surrogate_equation_id != NO_HALF_SURROGATE_EQUATION_ID:
            raise ValueError("candidate proposal changed the no-half surrogate equation")
        if self.fd_spectral_upper_equation_id != FD_SPECTRAL_UPPER_EQUATION_ID:
            raise ValueError("candidate proposal changed the FD upper equation")
        if self.schema != _PROPOSAL_SCHEMA:
            raise ValueError("unsupported modern-v1 candidate-proposal schema")
        if (
            type(self.proposal_only) is not bool
            or not self.proposal_only
            or type(self.proposal_witness_only) is not bool
            or not self.proposal_witness_only
            or type(self.fresh_relinearization_required) is not bool
            or not self.fresh_relinearization_required
            or type(self.exact_registered_surrogate_evaluation) is not bool
            or not self.exact_registered_surrogate_evaluation
            or type(self.exact_intervention_damage_evaluated) is not bool
            or self.exact_intervention_damage_evaluated
            or type(self.claim_execution_certified) is not bool
            or self.claim_execution_certified
        ):
            raise ValueError("candidate proposal crossed its proposal-only firewall")
        expected_evaluation_receipt_hash = content_hash(
            _candidate_evaluation_payload(
                numeric_state_hash=self.numeric_state_hash,
                domain_hash=self.domain_hash,
                constraints_hash=self.constraints_hash,
                parent_mask_hash=self.parent_mask_hash,
                resulting_mask_hash=self.resulting_mask.mask_hash,
                removed_count=self.removed_count,
                removed_route_index_hash=self.removed_route_index_hash,
                removed_route_ids=self.removed_route_ids,
                resource_gain=self.resource_gain,
                corrected_linear_damage=self.corrected_linear_damage,
                fd_quadratic=self.fd_quadratic,
                residual_diagonal_damage=self.residual_diagonal_damage,
                predicted_surrogate_damage=self.predicted_surrogate_damage,
                proposal_damage_constraint_value=self.proposal_damage_constraint_value,
                fd_spectral_upper=self.fd_spectral_upper,
                diagonal_hard_bound_used=self.diagonal_hard_bound_used,
                surrogate_equation_id=self.surrogate_equation_id,
                fd_spectral_upper_equation_id=self.fd_spectral_upper_equation_id,
            ),
            prefix="partition-free-v1-candidate-evaluation-",
        )
        if self.evaluation_receipt_hash != expected_evaluation_receipt_hash:
            raise ValueError("candidate evaluation receipt hash is not reproducible")
        expected_candidate_id = content_hash(
            {
                "evaluation_receipt_hash": expected_evaluation_receipt_hash,
                "resulting_mask_hash": self.resulting_mask.mask_hash,
                "removed_route_index_hash": self.removed_route_index_hash,
            },
            prefix="partition-free-v1-candidate-",
        )
        if self.candidate_id != expected_candidate_id:
            raise ValueError("candidate proposal content hash mismatch")

    @property
    def can_certify_decision(self) -> bool:
        return False

    @property
    def can_commit_mask(self) -> bool:
        return False

    def assert_can_certify_decision(self) -> None:
        raise PermissionError("modern-v1 candidates are not scientific certificates")

    def assert_can_commit_mask(self) -> None:
        raise PermissionError("modern-v1 candidates cannot commit a hard mask")

    @property
    def removed_route_ids_complete(self) -> bool:
        return len(self.removed_route_ids) == self.removed_count

    @property
    def removed_count(self) -> int:
        return self.removed_routes.count

    @property
    def removed_route_index_hash(self) -> str:
        return self.removed_routes.index_hash

    @property
    def compact_removed_indices_available(self) -> bool:
        return type(self.removed_routes) is RemovedRouteIndexSet

    def resolve_removed_routes(self, parent_mask: HardMask) -> RemovedRouteIndexSet:
        """Resolve a large delta from exact packed masks and verify its identity."""

        if type(parent_mask) is not HardMask:
            raise TypeError("proposal route resolution requires an exact parent HardMask")
        if parent_mask.mask_hash != self.parent_mask_hash:
            raise ValueError("parent mask differs from the proposal parent identity")
        resolved = (
            self.removed_routes
            if type(self.removed_routes) is RemovedRouteIndexSet
            else self.resulting_mask.removed_index_set_from(parent_mask)
        )
        if (
            resolved.count != self.removed_count
            or resolved.index_hash != self.removed_route_index_hash
        ):
            raise ValueError("resolved proposal delta differs from its compact identity")
        return resolved

    def resolve_removed_route_ids(
        self,
        route_ids: Sequence[str],
        *,
        parent_mask: HardMask | None = None,
    ) -> tuple[str, ...]:
        """Resolve names on demand; large proposal receipts retain no strings."""

        if len(route_ids) != self.resulting_mask.route_count:
            raise ValueError("route_ids do not cover the proposal's physical route axis")
        if type(self.removed_routes) is V1RemovedRouteSetIdentity and parent_mask is None:
            raise RuntimeError("large proposal route names require the exact parent mask")
        removed = (
            self.removed_routes
            if type(self.removed_routes) is RemovedRouteIndexSet
            else self.resolve_removed_routes(parent_mask)  # type: ignore[arg-type]
        )
        return tuple(str(route_ids[int(index)]) for index in removed)


def _candidate_evaluation_payload(
    *,
    numeric_state_hash: str,
    domain_hash: str,
    constraints_hash: str,
    parent_mask_hash: str,
    resulting_mask_hash: str,
    removed_count: int,
    removed_route_index_hash: str,
    removed_route_ids: tuple[str, ...],
    resource_gain: float,
    corrected_linear_damage: float,
    fd_quadratic: float,
    residual_diagonal_damage: float,
    predicted_surrogate_damage: float,
    proposal_damage_constraint_value: float,
    fd_spectral_upper: float,
    diagonal_hard_bound_used: bool,
    surrogate_equation_id: str,
    fd_spectral_upper_equation_id: str,
) -> dict[str, object]:
    """Build a hash payload without expanding large physical route names."""

    compact = len(removed_route_ids) != removed_count
    payload: dict[str, object] = {
        "schema": (
            "partition_free_v1_candidate_evaluation_receipt_v2"
            if compact
            else "partition_free_v1_candidate_evaluation_receipt_v1"
        ),
        "numeric_state_hash": numeric_state_hash,
        "domain_hash": domain_hash,
        "constraints_hash": constraints_hash,
        "parent_mask_hash": parent_mask_hash,
        "resulting_mask_hash": resulting_mask_hash,
        "removed_route_index_hash": removed_route_index_hash,
        "resource_gain": resource_gain,
        "corrected_linear_damage": corrected_linear_damage,
        "fd_quadratic": fd_quadratic,
        "residual_diagonal_damage": residual_diagonal_damage,
        "predicted_surrogate_damage": predicted_surrogate_damage,
        "proposal_damage_constraint_value": proposal_damage_constraint_value,
        "fd_spectral_upper": fd_spectral_upper,
        "diagonal_hard_bound_used": diagonal_hard_bound_used,
        "surrogate_equation_id": surrogate_equation_id,
        "fd_spectral_upper_equation_id": fd_spectral_upper_equation_id,
        "exact_registered_surrogate_evaluation": True,
        "exact_intervention_damage_evaluated": False,
        "proposal_only": True,
        "proposal_witness_only": True,
        "fresh_relinearization_required": True,
    }
    if compact:
        payload.update(
            {
                "removed_route_count": removed_count,
                "removed_route_ids_inline": (),
                "removed_route_ids_complete": False,
            }
        )
    else:
        # This exact legacy field preserves all existing small-action hashes.
        payload["removed_route_ids"] = removed_route_ids
    return payload


def _proposal_order_key(proposal: V1CandidateProposal) -> tuple[object, ...]:
    resource = proposal.resource_gain
    if resource <= 0.0:
        damage_per_resource = math.inf
        predicted_per_resource = math.inf
    else:
        damage_per_resource = proposal.proposal_damage_constraint_value / resource
        predicted_per_resource = proposal.predicted_surrogate_damage / resource
    return (
        damage_per_resource,
        predicted_per_resource,
        proposal.proposal_damage_constraint_value,
        proposal.predicted_surrogate_damage,
        proposal.fd_spectral_upper,
        -proposal.resource_gain,
        proposal.removed_count,
        (
            proposal.removed_routes.to_tuple()  # type: ignore[union-attr]
            if proposal.removed_count <= _INLINE_ACTION_LIMIT
            else proposal.removed_route_index_hash
        ),
    )


def _selection_tie_break(
    proposals: Sequence[V1CandidateProposal],
) -> tuple[str, ...]:
    final_tie_break = (
        "removed_route_index_hash"
        if any(item.removed_count > _INLINE_ACTION_LIMIT for item in proposals)
        else "lexicographic_physical_indices"
    )
    return (
        "minimize_registered_damage_screen_per_resource",
        "minimize_predicted_surrogate_damage_per_resource",
        "minimize_registered_damage_screen",
        "minimize_registered_surrogate_damage",
        "minimize_fd_spectral_upper",
        "maximize_resource_gain",
        "minimize_cardinality",
        final_tie_break,
    )


def _selection_order_contract(
    composer_id: str,
    proposals: Sequence[V1CandidateProposal],
) -> tuple[str, ...]:
    if composer_id == EXACT_FINITE_DOMAIN_COMPOSER_ID:
        return _selection_tie_break(proposals)
    if composer_id == FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID:
        return (
            "frozen_selector_frontier_order",
            "nonincreasing_physical_resource_gain",
            "declared_domain_order_for_equal_resource",
        )
    raise ValueError("unsupported modern-v1 candidate composer")


def _evaluate_action(
    *,
    state: V1CandidateNumericState,
    numeric_state_hash: str,
    live_flags: np.ndarray,
    domain_hash: str,
    constraints: V1CandidateCompositionConstraints,
    constraints_hash: str,
    removal_indices: tuple[int, ...] | RemovedRouteIndexSet | V1PackedRemovalAction,
    legacy_surrogate_constraints_apply: bool = True,
) -> V1CandidateProposal | None:
    if type(legacy_surrogate_constraints_apply) is not bool:
        raise TypeError("legacy_surrogate_constraints_apply must be bool")
    removal_count = len(removal_indices)
    if removal_count > constraints.maximum_removed_count:
        return None
    if type(removal_indices) is V1PackedRemovalAction:
        resulting = removal_indices.resulting_mask
        resolved_count, removed_route_index_hash = (
            resulting.removed_index_count_and_hash_from(state.parent_mask)
        )
        if (
            resolved_count != removal_indices.removed_count
            or removed_route_index_hash
            != removal_indices.removed_route_index_hash
        ):
            raise ValueError("packed finite action differs from its parent-mask delta")
        removed = (
            resulting.removed_index_set_from(state.parent_mask)
            if removal_count <= _INLINE_ACTION_LIMIT
            else None
        )
    else:
        removed = (
            removal_indices
            if type(removal_indices) is RemovedRouteIndexSet
            else RemovedRouteIndexSet.from_indices(
                removal_indices,
                route_count=len(state.route_ids),
            )
        )
        removed_route_index_hash = removed.index_hash
        resulting = state.parent_mask.with_removed_indices(removed)

    # Preserve byte-for-byte historical arithmetic for small actions.  Large
    # composite candidates use bounded advanced-index slabs: no rank-by-action
    # matrix, route-ID tuple, or route_count-sized int64 index promotion is
    # materialized.
    if removal_count <= _INLINE_ACTION_LIMIT:
        if removed is None:  # pragma: no cover - guarded by the branch above
            raise RuntimeError("small packed action did not resolve its route delta")
        legacy_indices = removed.to_tuple()
        if any(not bool(live_flags[index]) for index in legacy_indices):
            raise ValueError("finite removal domain contains an already-pruned route")
        projection = np.sum(
            state.effective_factor[:, np.asarray(legacy_indices, dtype=np.int64)],
            axis=1,
            dtype=np.float64,
        )
        corrected_linear = math.fsum(
            float(state.corrected_prices[index]) for index in legacy_indices
        )
        diagonal_damage = math.fsum(
            float(state.residual_diagonal[index]) for index in legacy_indices
        )
        resource_gain = math.fsum(
            float(state.atom_costs[index]) for index in legacy_indices
        )
    else:
        projection_parts: list[np.ndarray] = []
        corrected_parts: list[float] = []
        diagonal_parts: list[float] = []
        resource_parts: list[float] = []
        if type(removal_indices) is V1PackedRemovalAction:
            slabs = resulting.iter_removed_index_chunks_from(
                state.parent_mask,
                packed_byte_chunk_size=_ACTION_AGGREGATION_CHUNK_SIZE // 8,
            )
        else:
            if removed is None:  # pragma: no cover - defensive invariant
                raise RuntimeError("large explicit action lost its compact delta")
            index_array = removed.to_numpy()
            slabs = (
                index_array[start : start + _ACTION_AGGREGATION_CHUNK_SIZE]
                for start in range(
                    0,
                    removal_count,
                    _ACTION_AGGREGATION_CHUNK_SIZE,
                )
            )
        for slab in slabs:
            if np.any(~live_flags[slab]):
                raise ValueError(
                    "finite removal domain contains an already-pruned route"
                )
            projection_parts.append(
                np.sum(
                    state.effective_factor[:, slab],
                    axis=1,
                    dtype=np.float64,
                )
            )
            corrected_parts.append(
                float(np.sum(state.corrected_prices[slab], dtype=np.float64))
            )
            diagonal_parts.append(
                float(np.sum(state.residual_diagonal[slab], dtype=np.float64))
            )
            resource_parts.append(
                float(np.sum(state.atom_costs[slab], dtype=np.float64))
            )
        projection = np.asarray(
            [
                math.fsum(float(part[row]) for part in projection_parts)
                for row in range(state.effective_factor.shape[0])
            ],
            dtype=np.float64,
        )
        corrected_linear = math.fsum(corrected_parts)
        diagonal_damage = math.fsum(diagonal_parts)
        resource_gain = math.fsum(resource_parts)

    fd_quadratic = float(np.dot(projection, projection))
    predicted_damage = math.fsum((corrected_linear, fd_quadratic, diagonal_damage))
    fd_upper = math.fsum((fd_quadratic, state.fd_shrink * float(removal_count)))
    values = (
        corrected_linear,
        fd_quadratic,
        diagonal_damage,
        predicted_damage,
        fd_upper,
        resource_gain,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("candidate surrogate evaluation produced a nonfinite value")
    use_diagonal = constraints.diagonal_hard_bound_mode == "proved_chart_bound"
    damage_constraint_value = (
        predicted_damage if use_diagonal else math.fsum((corrected_linear, fd_quadratic))
    )
    if (
        resource_gain < constraints.minimum_resource_gain
        or resource_gain > constraints.maximum_resource_gain
        or (
            legacy_surrogate_constraints_apply
            and (
                damage_constraint_value > constraints.maximum_proposal_damage
                or fd_upper > constraints.maximum_fd_geometry_upper
            )
        )
    ):
        return None

    removed_ids = (
        tuple(state.route_ids[index] for index in removed)  # type: ignore[union-attr]
        if removal_count <= _INLINE_ACTION_LIMIT
        and type(state.route_ids) is tuple
        else ()
    )
    evaluation_payload = _candidate_evaluation_payload(
        numeric_state_hash=numeric_state_hash,
        domain_hash=domain_hash,
        constraints_hash=constraints_hash,
        parent_mask_hash=state.parent_mask_hash,
        resulting_mask_hash=resulting.mask_hash,
        removed_count=removal_count,
        removed_route_index_hash=removed_route_index_hash,
        removed_route_ids=removed_ids,
        resource_gain=resource_gain,
        corrected_linear_damage=corrected_linear,
        fd_quadratic=fd_quadratic,
        residual_diagonal_damage=diagonal_damage,
        predicted_surrogate_damage=predicted_damage,
        proposal_damage_constraint_value=damage_constraint_value,
        fd_spectral_upper=fd_upper,
        diagonal_hard_bound_used=use_diagonal,
        surrogate_equation_id=NO_HALF_SURROGATE_EQUATION_ID,
        fd_spectral_upper_equation_id=FD_SPECTRAL_UPPER_EQUATION_ID,
    )
    evaluation_receipt_hash = content_hash(
        evaluation_payload,
        prefix="partition-free-v1-candidate-evaluation-",
    )
    candidate_id = content_hash(
        {
            "evaluation_receipt_hash": evaluation_receipt_hash,
            "resulting_mask_hash": resulting.mask_hash,
            "removed_route_index_hash": removed_route_index_hash,
        },
        prefix="partition-free-v1-candidate-",
    )
    return V1CandidateProposal(
        candidate_id=candidate_id,
        numeric_state_hash=numeric_state_hash,
        domain_hash=domain_hash,
        constraints_hash=constraints_hash,
        parent_mask_hash=state.parent_mask_hash,
        route_axis_hash=state.physical_route_axis_hash,
        removed_routes=(
            removed
            if removal_count <= _INLINE_ACTION_LIMIT
            else V1RemovedRouteSetIdentity(
                route_count=resulting.route_count,
                count=removal_count,
                index_hash=removed_route_index_hash,
            )
        ),
        removed_route_ids=removed_ids,
        resulting_mask=resulting,
        resource_gain=resource_gain,
        corrected_linear_damage=corrected_linear,
        fd_quadratic=fd_quadratic,
        residual_diagonal_damage=diagonal_damage,
        predicted_surrogate_damage=predicted_damage,
        proposal_damage_constraint_value=damage_constraint_value,
        fd_spectral_upper=fd_upper,
        diagonal_hard_bound_used=use_diagonal,
        surrogate_equation_id=NO_HALF_SURROGATE_EQUATION_ID,
        fd_spectral_upper_equation_id=FD_SPECTRAL_UPPER_EQUATION_ID,
        evaluation_receipt_hash=evaluation_receipt_hash,
    )


def _ordered_domain_candidate_mask_hashes(
    domain: V1FiniteRemovalDomain,
    parent_mask: HardMask,
) -> tuple[str, ...]:
    hashes: list[str] = []
    for action in domain.removal_sets:
        child = (
            action.resulting_mask
            if type(action) is V1PackedRemovalAction
            else parent_mask.with_removed_indices(action)
        )
        hashes.append(child.mask_hash)
    return tuple(hashes)


@dataclass(frozen=True, slots=True)
class V1CandidateCompositionResult:
    """Exhaustive result over one declared finite domain, never all subsets."""

    numeric_state: V1CandidateNumericState
    domain: V1FiniteRemovalDomain
    constraints: V1CandidateCompositionConstraints
    ranked_proposals: tuple[V1CandidateProposal, ...]
    evaluated_action_count: int
    eligible_action_count: int
    composer_id: str
    selection_receipt_hash: str
    declared_domain_exhausted: bool = True
    exact_within_declared_domain: bool = True
    claims_global_powerset_optimum: bool = False
    proposal_only: bool = True
    proposal_witness_only: bool = True
    fresh_relinearization_required: bool = True
    claim_execution_certified: bool = False
    schema: str = _RESULT_SCHEMA

    def __post_init__(self) -> None:
        if type(self.numeric_state) is not V1CandidateNumericState:
            raise TypeError("composition result requires an issued numeric state")
        if type(self.domain) is not V1FiniteRemovalDomain:
            raise TypeError("composition result requires a finite removal domain")
        if type(self.constraints) is not V1CandidateCompositionConstraints:
            raise TypeError("composition result requires typed constraints")
        proposals = tuple(self.ranked_proposals)
        if any(type(item) is not V1CandidateProposal for item in proposals):
            raise TypeError("composition result contains a nonproposal value")
        if self.composer_id == EXACT_FINITE_DOMAIN_COMPOSER_ID:
            if proposals != tuple(sorted(proposals, key=_proposal_order_key)):
                raise ValueError("candidate proposals are not in deterministic rank order")
        elif self.composer_id == FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID:
            if any(
                right.resource_gain > left.resource_gain + 1e-12
                for left, right in zip(proposals, proposals[1:], strict=False)
            ):
                raise ValueError(
                    "frozen frontier proposals are not in nonincreasing resource order"
                )
        else:
            raise ValueError("unsupported modern-v1 candidate composer")
        evaluated_count = _positive_int(
            self.evaluated_action_count,
            "evaluated_action_count",
        )
        eligible_count = _nonnegative_int(
            self.eligible_action_count,
            "eligible_action_count",
        )
        if evaluated_count != self.domain.action_count:
            raise ValueError("exact composition did not exhaust the declared finite domain")
        if eligible_count != len(proposals) or eligible_count > evaluated_count:
            raise ValueError("eligible candidate count differs from retained proposals")
        object.__setattr__(self, "evaluated_action_count", evaluated_count)
        object.__setattr__(self, "eligible_action_count", eligible_count)
        numeric_state_hash = self.numeric_state.state_hash
        domain_hash = self.domain.domain_hash
        constraints_hash = self.constraints.constraints_hash
        for proposal in proposals:
            if (
                proposal.numeric_state_hash != numeric_state_hash
                or proposal.domain_hash != domain_hash
                or proposal.constraints_hash != constraints_hash
            ):
                raise ValueError("candidate proposal is bound to another composition input")
            resolved_count, resolved_hash = (
                proposal.resulting_mask.removed_index_count_and_hash_from(
                self.numeric_state.parent_mask
                )
            )
            if (
                resolved_count != proposal.removed_count
                or resolved_hash != proposal.removed_route_index_hash
            ):
                raise ValueError("candidate complete mask differs from its removal receipt")
        if self.composer_id == FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID:
            retained_hashes = tuple(item.resulting_mask.mask_hash for item in proposals)
            retained_set = set(retained_hashes)
            declared_subsequence = tuple(
                value
                for value in _ordered_domain_candidate_mask_hashes(
                    self.domain,
                    self.numeric_state.parent_mask,
                )
                if value in retained_set
            )
            if retained_hashes != declared_subsequence:
                raise ValueError(
                    "frozen frontier proposals differ from declared domain order"
                )
        object.__setattr__(
            self,
            "selection_receipt_hash",
            _nonempty(self.selection_receipt_hash, "selection_receipt_hash"),
        )
        flags = (
            self.declared_domain_exhausted,
            self.exact_within_declared_domain,
            not self.claims_global_powerset_optimum,
            self.proposal_only,
            self.proposal_witness_only,
            self.fresh_relinearization_required,
            not self.claim_execution_certified,
        )
        if any(type(flag) is not bool or not flag for flag in flags):
            raise ValueError("composition result crossed its exact proposal-only firewall")
        if self.schema != _RESULT_SCHEMA:
            raise ValueError("unsupported modern-v1 composition-result schema")
        expected_selection_receipt_hash = content_hash(
            {
                "schema": "partition_free_v1_candidate_selection_receipt_v1",
                "composer_id": self.composer_id,
                "numeric_state_hash": numeric_state_hash,
                "domain_hash": domain_hash,
                "constraints_hash": constraints_hash,
                "evaluated_action_count": self.evaluated_action_count,
                "eligible_action_count": self.eligible_action_count,
                "ranked_candidate_ids": tuple(item.candidate_id for item in proposals),
                "tie_break": _selection_order_contract(self.composer_id, proposals),
                "declared_domain_exhausted": True,
                "exact_within_declared_domain": True,
                "claims_global_powerset_optimum": False,
                "proposal_only": True,
                "proposal_witness_only": True,
                "fresh_relinearization_required": True,
            },
            prefix="partition-free-v1-candidate-selection-",
        )
        if self.selection_receipt_hash != expected_selection_receipt_hash:
            raise ValueError("candidate selection receipt hash is not reproducible")
        object.__setattr__(self, "ranked_proposals", proposals)

    @property
    def selected_proposal(self) -> V1CandidateProposal:
        if not self.ranked_proposals:
            raise LookupError("candidate composition has no eligible proposal")
        return self.ranked_proposals[0]

    @property
    def result_hash(self) -> str:
        return content_hash(
            {
                "schema": self.schema,
                "numeric_state_hash": self.numeric_state.state_hash,
                "domain_hash": self.domain.domain_hash,
                "constraints_hash": self.constraints.constraints_hash,
                "ranked_candidate_ids": tuple(item.candidate_id for item in self.ranked_proposals),
                "evaluated_action_count": self.evaluated_action_count,
                "eligible_action_count": self.eligible_action_count,
                "composer_id": self.composer_id,
                "selection_receipt_hash": self.selection_receipt_hash,
                "declared_domain_exhausted": self.declared_domain_exhausted,
                "exact_within_declared_domain": self.exact_within_declared_domain,
                "claims_global_powerset_optimum": (self.claims_global_powerset_optimum),
                "proposal_only": self.proposal_only,
                "proposal_witness_only": self.proposal_witness_only,
                "fresh_relinearization_required": (self.fresh_relinearization_required),
                "claim_execution_certified": self.claim_execution_certified,
            },
            prefix="partition-free-v1-candidate-composition-",
        )

    @property
    def can_certify_decision(self) -> bool:
        return False

    @property
    def can_commit_mask(self) -> bool:
        return False

    def assert_can_certify_decision(self) -> None:
        raise PermissionError("candidate composition is not a scientific certificate")

    def assert_can_commit_mask(self) -> None:
        raise PermissionError("candidate composition cannot commit a hard mask")


def compose_v1_candidates_exact(
    *,
    numeric_state: V1CandidateNumericState,
    domain: V1FiniteRemovalDomain,
    constraints: V1CandidateCompositionConstraints,
) -> V1CandidateCompositionResult:
    """Exhaustively rank feasible actions in one explicit finite domain."""

    if type(numeric_state) is not V1CandidateNumericState:
        raise TypeError("exact composer requires an issued V1CandidateNumericState")
    if type(domain) is not V1FiniteRemovalDomain:
        raise TypeError("exact composer requires a V1FiniteRemovalDomain")
    if type(constraints) is not V1CandidateCompositionConstraints:
        raise TypeError("exact composer requires V1CandidateCompositionConstraints")
    if (
        domain.route_axis_hash != numeric_state.physical_route_axis_hash
        or domain.route_count != len(numeric_state.route_ids)
    ):
        raise ValueError("finite removal domain differs from the candidate route axis")
    if constraints.diagonal_hard_bound_mode == "proved_chart_bound":
        if (
            numeric_state.residual_diagonal_status
            not in {
                "global_loewner_bound",
                "action_class_quadratic_bound",
            }
            or not numeric_state.residual_bound_proof_hash
        ):
            raise ValueError(
                "residual diagonal cannot enter a hard bound without chart proof status"
            )
        if numeric_state.residual_diagonal_status == "action_class_quadratic_bound":
            proved_class = numeric_state.proved_action_class
            declared_class = domain.action_class_proof
            if proved_class is None or declared_class is None:
                raise ValueError("proved diagonal action class requires a typed finite declaration")
            # Never accept a caller-supplied or merely matching label.  Both
            # declarations must reproduce the same content address, bind the
            # current chart identity, and enumerate the exact same actions in
            # the exact declared order.  Membership is then rechecked for each
            # action before any diagonal value enters a hard-bound constraint.
            if declared_class.proof_hash != proved_class.proof_hash:
                raise ValueError("declared finite domain differs from the proved action class")
            if domain.has_compact_actions:
                raise ValueError(
                    "compact action domains require a compact-aware action-class proof"
                )
            if (
                declared_class.route_axis_hash != numeric_state.route_axis_hash
                or declared_class.route_count != len(numeric_state.route_ids)
                or declared_class.base_mask_hash != numeric_state.parent_mask_hash
                or declared_class.current_mask_hash != numeric_state.parent_mask_hash
                or declared_class.proof_artifact_hash != numeric_state.residual_bound_proof_hash
            ):
                raise ValueError("proved action-class identity is stale for this chart")
            if domain.removal_sets != declared_class.ordered_removal_sets:
                raise ValueError(
                    "declared finite domain order or membership differs from its action class"
                )
            if any(not declared_class.contains(action) for action in domain.removal_sets):
                raise ValueError("finite-domain action lies outside the proved action class")

    numeric_state_hash = numeric_state.state_hash
    domain_hash = domain.domain_hash
    constraints_hash = constraints.constraints_hash
    live_flags = numeric_state.parent_mask.to_bool_array()
    proposals: list[V1CandidateProposal] = []
    for removal_indices in domain.removal_sets:
        proposal = _evaluate_action(
            state=numeric_state,
            numeric_state_hash=numeric_state_hash,
            live_flags=live_flags,
            domain_hash=domain_hash,
            constraints=constraints,
            constraints_hash=constraints_hash,
            removal_indices=removal_indices,
        )
        if proposal is not None:
            proposals.append(proposal)
    ranked = tuple(sorted(proposals, key=_proposal_order_key))
    selection_receipt_hash = content_hash(
        {
            "schema": "partition_free_v1_candidate_selection_receipt_v1",
            "composer_id": EXACT_FINITE_DOMAIN_COMPOSER_ID,
            "numeric_state_hash": numeric_state_hash,
            "domain_hash": domain_hash,
            "constraints_hash": constraints_hash,
            "evaluated_action_count": domain.action_count,
            "eligible_action_count": len(ranked),
            "ranked_candidate_ids": tuple(item.candidate_id for item in ranked),
            "tie_break": _selection_tie_break(ranked),
            "declared_domain_exhausted": True,
            "exact_within_declared_domain": True,
            "claims_global_powerset_optimum": False,
            "proposal_only": True,
            "proposal_witness_only": True,
            "fresh_relinearization_required": True,
        },
        prefix="partition-free-v1-candidate-selection-",
    )
    return V1CandidateCompositionResult(
        numeric_state=numeric_state,
        domain=domain,
        constraints=constraints,
        ranked_proposals=ranked,
        evaluated_action_count=domain.action_count,
        eligible_action_count=len(ranked),
        composer_id=EXACT_FINITE_DOMAIN_COMPOSER_ID,
        selection_receipt_hash=selection_receipt_hash,
    )


def compose_v1_candidates_in_frozen_resource_descending_order(
    *,
    numeric_state: V1CandidateNumericState,
    domain: V1FiniteRemovalDomain,
    constraints: V1CandidateCompositionConstraints,
) -> V1CandidateCompositionResult:
    """Evaluate a frozen selector frontier without replacing its order.

    The declared domain must already be ordered by nonincreasing physical
    resource release.  The legacy signed-plus-quadratic surrogate is retained
    in each proposal as a diagnostic only; it neither ranks nor filters beyond
    the caller's explicit, normally nonbinding proposal constraints.  This
    allows the direct oracle to test the largest frozen candidate first and
    then move to smaller candidates after rejection.
    """

    if type(numeric_state) is not V1CandidateNumericState:
        raise TypeError("frozen-frontier composer requires an issued numeric state")
    if type(domain) is not V1FiniteRemovalDomain:
        raise TypeError("frozen-frontier composer requires a finite removal domain")
    if type(constraints) is not V1CandidateCompositionConstraints:
        raise TypeError("frozen-frontier composer requires typed constraints")
    if (
        domain.route_axis_hash != numeric_state.physical_route_axis_hash
        or domain.route_count != len(numeric_state.route_ids)
    ):
        raise ValueError("frozen frontier differs from the candidate route axis")
    if constraints.diagonal_hard_bound_mode != "proposal_ranking_only":
        raise ValueError(
            "frozen geometric frontier cannot use the legacy diagonal as a hard bound"
        )

    numeric_state_hash = numeric_state.state_hash
    domain_hash = domain.domain_hash
    constraints_hash = constraints.constraints_hash
    live_flags = numeric_state.parent_mask.to_bool_array()
    proposals: list[V1CandidateProposal] = []
    for removal_indices in domain.removal_sets:
        proposal = _evaluate_action(
            state=numeric_state,
            numeric_state_hash=numeric_state_hash,
            live_flags=live_flags,
            domain_hash=domain_hash,
            constraints=constraints,
            constraints_hash=constraints_hash,
            removal_indices=removal_indices,
            legacy_surrogate_constraints_apply=False,
        )
        if proposal is not None:
            proposals.append(proposal)
    ordered = tuple(proposals)
    if any(
        right.resource_gain > left.resource_gain + 1e-12
        for left, right in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError(
            "frozen selector frontier must be declared in nonincreasing resource order"
        )
    selection_receipt_hash = content_hash(
        {
            "schema": "partition_free_v1_candidate_selection_receipt_v1",
            "composer_id": FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID,
            "numeric_state_hash": numeric_state_hash,
            "domain_hash": domain_hash,
            "constraints_hash": constraints_hash,
            "evaluated_action_count": domain.action_count,
            "eligible_action_count": len(ordered),
            "ranked_candidate_ids": tuple(item.candidate_id for item in ordered),
            "tie_break": _selection_order_contract(
                FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID,
                ordered,
            ),
            "declared_domain_exhausted": True,
            "exact_within_declared_domain": True,
            "claims_global_powerset_optimum": False,
            "proposal_only": True,
            "proposal_witness_only": True,
            "fresh_relinearization_required": True,
        },
        prefix="partition-free-v1-candidate-selection-",
    )
    return V1CandidateCompositionResult(
        numeric_state=numeric_state,
        domain=domain,
        constraints=constraints,
        ranked_proposals=ordered,
        evaluated_action_count=domain.action_count,
        eligible_action_count=len(ordered),
        composer_id=FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID,
        selection_receipt_hash=selection_receipt_hash,
    )


def _fresh_hard_mask_hash(mask: HardMask, *, name: str) -> str:
    """Hash live packed bytes rather than trusting a cached mask property."""

    if type(mask) is not HardMask:
        raise TypeError(f"{name} must be an exact HardMask")
    route_count = _positive_int(mask.route_count, f"{name} route_count")
    route_axis_hash = _nonempty(mask.route_axis_hash, f"{name} route_axis_hash")
    packed = mask.packed_bits
    if type(packed) is not bytes or len(packed) != (route_count + 7) // 8:
        raise ValueError(f"{name} packed byte length differs from its route count")
    padding = len(packed) * 8 - route_count
    if padding and packed[-1] & ((1 << padding) - 1):
        raise ValueError(f"{name} contains nonzero padding bits")
    return content_hash(
        {
            "route_axis_hash": route_axis_hash,
            "route_count": route_count,
            "packed_bits": packed,
        },
        prefix="irg-mask-",
    )


def _fq_budget_ratio(value: float, budget: float) -> float | None:
    """Return a dimensionless utilization ratio, or None for a zero budget."""

    if budget == 0.0:
        return None
    return value / budget


def _fq_resource_gain_receipt_payload(
    *,
    endpoint_index: int,
    action_hash: str,
    atom_cost_source_hash: str,
    parent_mask_hash: str,
    resulting_mask_hash: str,
    route_axis_hash: str,
    route_count: int,
    removed_count: int,
    removed_route_index_hash: str,
    resource_gain: float,
) -> dict[str, object]:
    return {
        "schema": _FQ_RESOURCE_GAIN_RECEIPT_SCHEMA,
        "endpoint_index": endpoint_index,
        "action_hash": action_hash,
        "atom_cost_source_hash": atom_cost_source_hash,
        "parent_mask_hash": parent_mask_hash,
        "resulting_mask_hash": resulting_mask_hash,
        "route_axis_hash": route_axis_hash,
        "route_count": route_count,
        "removed_count": removed_count,
        "removed_route_index_hash": removed_route_index_hash,
        "exact_scalar_resource_gain": resource_gain,
        "full_route_cost_array_retained": False,
    }


@dataclass(frozen=True, slots=True, init=False)
class V1FQActionNumericState:
    """Compact numeric snapshot for one already-authorized F/Q endpoint.

    This is not the legacy full-route candidate numeric state.  It retains one
    exact packed child mask, scalar resource release, and independently
    certified F/Q confidence limits.  It deliberately has no route-ID axis,
    route-count-sized float array, registered surrogate, or intervention
    evaluation.
    """

    endpoint_index: int
    action_id: str
    action_hash: str
    parent_mask: HardMask
    packed_action: V1PackedRemovalAction
    route_axis_hash: str
    route_count: int
    parent_mask_hash: str
    resulting_mask_hash: str
    packed_action_hash: str
    removed_count: int
    removed_route_index_hash: str
    resource_gain: float
    atom_cost_source_hash: str
    resource_gain_receipt_hash: str
    f_lower: float
    f_upper: float
    q_lower: float
    q_upper: float
    sqrt_q_lower: float
    sqrt_q_upper: float
    f_units: str
    q_units: str
    sqrt_q_units: str
    fq_interval_schema: str
    fq_interval_hash: str
    fq_interval_full_census: bool
    fq_interval_population_census_complete: bool
    fq_interval_lower_confidence_authority: bool
    fq_interval_upper_confidence_authority: bool
    fq_interval_observation_semantics: str
    f_upper_budget: float
    q_upper_budget: float
    f_upper_budget_ratio: float | None
    q_upper_budget_ratio: float | None
    sqrt_q_upper_budget_ratio: float | None
    fq_budget_hash: str
    fq_certification_receipt_hash: str
    decisive_fq_commit_hash: str
    fq_queue_authorization_hash: str
    sampled_diagnostic_lineage_hash: str
    generation_index: int
    independently_certified_fq_interval: bool
    exact_scalar_resource_gain_from_bound_source: bool
    proposal_only: bool
    fq_monotonicity_assumed: bool
    exact_registered_surrogate_evaluation: bool
    exact_intervention_damage_evaluated: bool
    candidate_freeze_authority: bool
    bce_oracle_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    schema: str = _FQ_ACTION_NUMERIC_STATE_SCHEMA
    _state_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _FQ_ACTION_NUMERIC_STATE_TOKEN:
            raise TypeError(
                "V1FQActionNumericState can only be issued from exact F/Q authorization"
            )
        expected = tuple(
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        )
        if set(values) != set(expected):
            missing = sorted(set(expected).difference(values))
            extra = sorted(set(values).difference(expected))
            raise TypeError(
                f"F/Q action-state fields differ: missing={missing}, extra={extra}"
            )
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", _FQ_ACTION_NUMERIC_STATE_SCHEMA)
        object.__setattr__(self, "_state_hash", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        endpoint = _positive_int(self.endpoint_index, "endpoint_index")
        generation = _nonnegative_int(self.generation_index, "generation_index")
        if type(self.parent_mask) is not HardMask:
            raise TypeError("compact F/Q state requires an exact parent HardMask")
        if type(self.packed_action) is not V1PackedRemovalAction:
            raise TypeError("compact F/Q state requires an exact packed action")
        child = self.packed_action.resulting_mask
        parent_hash = _fresh_hard_mask_hash(
            self.parent_mask,
            name="compact F/Q parent mask",
        )
        child_hash = _fresh_hard_mask_hash(
            child,
            name="compact F/Q resulting mask",
        )
        self.parent_mask.assert_same_axis(child)
        removed_count, removed_hash = child.removed_index_count_and_hash_from(
            self.parent_mask
        )
        packed_hash = self.packed_action.action_hash
        route_count = _positive_int(self.route_count, "route_count")
        declared_removed_count = _positive_int(self.removed_count, "removed_count")
        if (
            route_count != self.parent_mask.route_count
            or self.route_axis_hash != self.parent_mask.route_axis_hash
            or self.parent_mask_hash != parent_hash
            or self.resulting_mask_hash != child_hash
            or self.packed_action_hash != packed_hash
            or declared_removed_count != removed_count
            or declared_removed_count != self.packed_action.removed_count
            or self.removed_route_index_hash != removed_hash
            or self.removed_route_index_hash
            != self.packed_action.removed_route_index_hash
        ):
            raise ValueError("compact F/Q action identity differs from its exact masks")

        for name in (
            "action_id",
            "action_hash",
            "route_axis_hash",
            "parent_mask_hash",
            "resulting_mask_hash",
            "packed_action_hash",
            "removed_route_index_hash",
            "atom_cost_source_hash",
            "resource_gain_receipt_hash",
            "fq_interval_schema",
            "fq_interval_hash",
            "fq_interval_observation_semantics",
            "fq_budget_hash",
            "fq_certification_receipt_hash",
            "decisive_fq_commit_hash",
            "fq_queue_authorization_hash",
            "sampled_diagnostic_lineage_hash",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        values = {
            name: _finite_nonnegative(getattr(self, name), name)
            for name in (
                "resource_gain",
                "f_lower",
                "f_upper",
                "q_lower",
                "q_upper",
                "sqrt_q_lower",
                "sqrt_q_upper",
                "f_upper_budget",
                "q_upper_budget",
            )
        }
        if (
            values["f_lower"] > values["f_upper"]
            or values["q_lower"] > values["q_upper"]
            or not math.isclose(
                values["sqrt_q_lower"],
                math.sqrt(values["q_lower"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                values["sqrt_q_upper"],
                math.sqrt(values["q_upper"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("compact F/Q intervals are inconsistent")
        if (
            values["f_upper"] > values["f_upper_budget"]
            or values["q_upper"] > values["q_upper_budget"]
        ):
            raise ValueError("compact F/Q endpoint exceeds its registered budget")
        if (self.f_units, self.q_units, self.sqrt_q_units) != (
            _FQ_F_UNITS,
            _FQ_Q_UNITS,
            _FQ_SQRT_Q_UNITS,
        ):
            raise ValueError("compact F/Q state changed its physical units")
        expected_f_ratio = _fq_budget_ratio(
            values["f_upper"], values["f_upper_budget"]
        )
        expected_q_ratio = _fq_budget_ratio(
            values["q_upper"], values["q_upper_budget"]
        )
        expected_sqrt_q_ratio = _fq_budget_ratio(
            values["sqrt_q_upper"], math.sqrt(values["q_upper_budget"])
        )
        for name, expected_ratio in (
            ("f_upper_budget_ratio", expected_f_ratio),
            ("q_upper_budget_ratio", expected_q_ratio),
            ("sqrt_q_upper_budget_ratio", expected_sqrt_q_ratio),
        ):
            observed = getattr(self, name)
            if expected_ratio is None:
                if observed is not None:
                    raise ValueError(f"{name} must be None for a zero budget")
            elif (
                observed is None
                or not math.isfinite(float(observed))
                or not math.isclose(
                    float(observed), expected_ratio, rel_tol=1e-15, abs_tol=0.0
                )
            ):
                raise ValueError(f"{name} differs from its dimensionless ratio")
            object.__setattr__(
                self,
                name,
                None if expected_ratio is None else float(expected_ratio),
            )

        interval_flags = (
            self.fq_interval_full_census,
            self.fq_interval_population_census_complete,
            self.fq_interval_lower_confidence_authority,
            self.fq_interval_upper_confidence_authority,
        )
        if any(type(flag) is not bool for flag in interval_flags):
            raise TypeError("compact F/Q interval authority flags must be booleans")
        firewall = (
            self.independently_certified_fq_interval,
            self.exact_scalar_resource_gain_from_bound_source,
            self.proposal_only,
            not self.fq_monotonicity_assumed,
            not self.exact_registered_surrogate_evaluation,
            not self.exact_intervention_damage_evaluated,
            not self.candidate_freeze_authority,
            not self.bce_oracle_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            not self.execution_authority,
        )
        if any(type(flag) is not bool or not flag for flag in firewall):
            raise ValueError("compact F/Q state crossed its proposal-only firewall")
        if not self.fq_interval_upper_confidence_authority:
            raise ValueError("compact F/Q admission requires an upper-authoritative interval")
        if self.schema != _FQ_ACTION_NUMERIC_STATE_SCHEMA:
            raise ValueError("unsupported compact F/Q action-state schema")

        expected_resource_receipt = content_hash(
            _fq_resource_gain_receipt_payload(
                endpoint_index=endpoint,
                action_hash=self.action_hash,
                atom_cost_source_hash=self.atom_cost_source_hash,
                parent_mask_hash=parent_hash,
                resulting_mask_hash=child_hash,
                route_axis_hash=self.route_axis_hash,
                route_count=route_count,
                removed_count=declared_removed_count,
                removed_route_index_hash=self.removed_route_index_hash,
                resource_gain=values["resource_gain"],
            ),
            prefix="partition-free-v1-fq-resource-gain-",
        )
        if self.resource_gain_receipt_hash != expected_resource_receipt:
            raise ValueError("compact F/Q resource-gain receipt is not reproducible")
        object.__setattr__(self, "endpoint_index", endpoint)
        object.__setattr__(self, "generation_index", generation)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "removed_count", declared_removed_count)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.assert_integrity()

    @property
    def resulting_mask(self) -> HardMask:
        return self.packed_action.resulting_mask

    @property
    def removed_routes(self) -> V1RemovedRouteSetIdentity:
        return V1RemovedRouteSetIdentity(
            route_count=self.route_count,
            count=self.removed_count,
            index_hash=self.removed_route_index_hash,
        )

    @property
    def state_hash(self) -> str:
        return self.assert_integrity()

    @property
    def numeric_state_hash(self) -> str:
        return self.state_hash

    @property
    def compact_removed_indices_available(self) -> bool:
        return False

    def _payload(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "schema",
                "endpoint_index",
                "action_id",
                "action_hash",
                "route_axis_hash",
                "route_count",
                "parent_mask_hash",
                "resulting_mask_hash",
                "packed_action_hash",
                "removed_count",
                "removed_route_index_hash",
                "resource_gain",
                "atom_cost_source_hash",
                "resource_gain_receipt_hash",
                "f_lower",
                "f_upper",
                "q_lower",
                "q_upper",
                "sqrt_q_lower",
                "sqrt_q_upper",
                "f_units",
                "q_units",
                "sqrt_q_units",
                "fq_interval_schema",
                "fq_interval_hash",
                "fq_interval_full_census",
                "fq_interval_population_census_complete",
                "fq_interval_lower_confidence_authority",
                "fq_interval_upper_confidence_authority",
                "fq_interval_observation_semantics",
                "f_upper_budget",
                "q_upper_budget",
                "f_upper_budget_ratio",
                "q_upper_budget_ratio",
                "sqrt_q_upper_budget_ratio",
                "fq_budget_hash",
                "fq_certification_receipt_hash",
                "decisive_fq_commit_hash",
                "fq_queue_authorization_hash",
                "sampled_diagnostic_lineage_hash",
                "generation_index",
                "independently_certified_fq_interval",
                "exact_scalar_resource_gain_from_bound_source",
                "proposal_only",
                "fq_monotonicity_assumed",
                "exact_registered_surrogate_evaluation",
                "exact_intervention_damage_evaluated",
                "candidate_freeze_authority",
                "bce_oracle_authority",
                "mask_commit_authority",
                "terminal_authority",
                "pareto_authority",
                "execution_authority",
            )
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        parent_hash = _fresh_hard_mask_hash(
            self.parent_mask,
            name="compact F/Q parent mask",
        )
        child_hash = _fresh_hard_mask_hash(
            self.resulting_mask,
            name="compact F/Q resulting mask",
        )
        count, delta_hash = self.resulting_mask.removed_index_count_and_hash_from(
            self.parent_mask
        )
        if (
            parent_hash != self.parent_mask_hash
            or child_hash != self.resulting_mask_hash
            or self.packed_action.action_hash != self.packed_action_hash
            or count != self.removed_count
            or delta_hash != self.removed_route_index_hash
        ):
            raise ValueError("compact F/Q state masks or delta were tampered")
        observed = content_hash(
            self._payload(),
            prefix="partition-free-v1-fq-action-numeric-state-",
        )
        if self._state_hash is not None and observed != self._state_hash:
            raise ValueError("compact F/Q action numeric state was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("compact F/Q action numeric-state hash does not match")
        object.__setattr__(self, "_state_hash", observed)
        return observed

    @property
    def can_certify_decision(self) -> bool:
        return False

    @property
    def can_commit_mask(self) -> bool:
        return False

    def assert_can_certify_decision(self) -> None:
        raise PermissionError("compact F/Q action states are not scientific certificates")

    def assert_can_commit_mask(self) -> None:
        raise PermissionError("compact F/Q action states cannot commit a hard mask")


def _fq_compact_domain_hash(states: Sequence[V1FQActionNumericState]) -> str:
    first = states[0]
    return content_hash(
        {
            "schema": _FQ_COMPACT_DOMAIN_SCHEMA,
            "fq_queue_authorization_hash": first.fq_queue_authorization_hash,
            "decisive_fq_commit_hash": first.decisive_fq_commit_hash,
            "parent_mask_hash": first.parent_mask_hash,
            "route_axis_hash": first.route_axis_hash,
            "route_count": first.route_count,
            "queue_endpoint_indices": tuple(item.endpoint_index for item in states),
            "queue_action_hashes": tuple(item.action_hash for item in states),
            "packed_action_hashes": tuple(item.packed_action_hash for item in states),
            "resulting_mask_hashes": tuple(item.resulting_mask_hash for item in states),
            "removed_counts": tuple(item.removed_count for item in states),
            "removed_route_index_hashes": tuple(
                item.removed_route_index_hash for item in states
            ),
            "arbitrary_certified_endpoint_subset_preserved": True,
            "fq_monotonicity_assumed": False,
            "filtering_performed": False,
            "reordering_performed": False,
            "claims_global_powerset_domain": False,
        },
        prefix="partition-free-v1-fq-compact-domain-",
    )


def _fq_constraints_hash(state: V1FQActionNumericState) -> str:
    return content_hash(
        {
            "schema": _FQ_CONSTRAINTS_SCHEMA,
            "fq_budget_hash": state.fq_budget_hash,
            "f_upper_budget": state.f_upper_budget,
            "q_upper_budget": state.q_upper_budget,
            "f_units": state.f_units,
            "q_units": state.q_units,
            "fq_queue_authorization_hash": state.fq_queue_authorization_hash,
            "decisive_fq_commit_hash": state.decisive_fq_commit_hash,
            "independently_certified_endpoint_intervals_required": True,
            "legacy_surrogate_constraints_present": False,
        },
        prefix="partition-free-v1-fq-constraints-",
    )


@dataclass(frozen=True, slots=True, init=False)
class V1FQCandidateProposal:
    """One packed F/Q-admitted proposal, with no surrogate relabeling."""

    numeric_state: V1FQActionNumericState
    domain_hash: str
    constraints_hash: str
    evaluation_receipt_hash: str
    candidate_id: str
    proposal_only: bool
    proposal_witness_only: bool
    fresh_relinearization_required: bool
    exact_registered_surrogate_evaluation: bool
    exact_intervention_damage_evaluated: bool
    fq_monotonicity_assumed: bool
    candidate_freeze_authority: bool
    bce_oracle_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    schema: str = _FQ_PROPOSAL_SCHEMA

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _FQ_PROPOSAL_TOKEN:
            raise TypeError("V1FQCandidateProposal can only be issued by its composer")
        expected = tuple(
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        )
        if set(values) != set(expected):
            raise TypeError("compact F/Q proposal fields differ from their schema")
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", _FQ_PROPOSAL_SCHEMA)
        self.__post_init__()

    def __post_init__(self) -> None:
        if type(self.numeric_state) is not V1FQActionNumericState:
            raise TypeError("compact F/Q proposal requires its issued numeric state")
        self.numeric_state.assert_integrity()
        for name in (
            "domain_hash",
            "constraints_hash",
            "evaluation_receipt_hash",
            "candidate_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        firewall = (
            self.proposal_only,
            self.proposal_witness_only,
            self.fresh_relinearization_required,
            not self.exact_registered_surrogate_evaluation,
            not self.exact_intervention_damage_evaluated,
            not self.fq_monotonicity_assumed,
            not self.candidate_freeze_authority,
            not self.bce_oracle_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            not self.execution_authority,
        )
        if any(type(flag) is not bool or not flag for flag in firewall):
            raise ValueError("compact F/Q proposal crossed its authority firewall")
        if self.schema != _FQ_PROPOSAL_SCHEMA:
            raise ValueError("unsupported compact F/Q proposal schema")
        expected_receipt = content_hash(
            self._evaluation_payload(),
            prefix="partition-free-v1-fq-candidate-evaluation-",
        )
        if self.evaluation_receipt_hash != expected_receipt:
            raise ValueError("compact F/Q evaluation receipt is not reproducible")
        expected_candidate = content_hash(
            {
                "evaluation_receipt_hash": expected_receipt,
                "resulting_mask_hash": self.resulting_mask_hash,
                "removed_route_index_hash": self.removed_route_index_hash,
                "fq_interval_hash": self.fq_interval_hash,
            },
            prefix="partition-free-v1-fq-candidate-",
        )
        if self.candidate_id != expected_candidate:
            raise ValueError("compact F/Q candidate content hash differs")

    def _evaluation_payload(self) -> dict[str, object]:
        return {
            "schema": "partition_free_v1_fq_candidate_evaluation_receipt_v1",
            "numeric_state_hash": self.numeric_state_hash,
            "domain_hash": self.domain_hash,
            "constraints_hash": self.constraints_hash,
            "endpoint_index": self.endpoint_index,
            "action_hash": self.action_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "resulting_mask_hash": self.resulting_mask_hash,
            "removed_count": self.removed_count,
            "removed_route_index_hash": self.removed_route_index_hash,
            "resource_gain": self.resource_gain,
            "resource_gain_receipt_hash": self.resource_gain_receipt_hash,
            "fq_interval_hash": self.fq_interval_hash,
            "f_interval": (self.f_lower, self.f_upper),
            "q_interval": (self.q_lower, self.q_upper),
            "sqrt_q_interval": (self.sqrt_q_lower, self.sqrt_q_upper),
            "f_units": self.f_units,
            "q_units": self.q_units,
            "sqrt_q_units": self.sqrt_q_units,
            "f_upper_budget": self.f_upper_budget,
            "q_upper_budget": self.q_upper_budget,
            "f_upper_budget_ratio": self.f_upper_budget_ratio,
            "q_upper_budget_ratio": self.q_upper_budget_ratio,
            "sqrt_q_upper_budget_ratio": self.sqrt_q_upper_budget_ratio,
            "fq_budget_hash": self.fq_budget_hash,
            "decisive_fq_commit_hash": self.decisive_fq_commit_hash,
            "fq_queue_authorization_hash": self.fq_queue_authorization_hash,
            "sampled_diagnostic_lineage_hash": (
                self.sampled_diagnostic_lineage_hash
            ),
            "independently_certified_fq_interval": True,
            "exact_registered_surrogate_evaluation": False,
            "exact_intervention_damage_evaluated": False,
            "proposal_only": True,
            "proposal_witness_only": True,
            "fresh_relinearization_required": True,
            "candidate_freeze_authority": False,
            "bce_oracle_authority": False,
            "mask_commit_authority": False,
            "terminal_authority": False,
            "pareto_authority": False,
            "execution_authority": False,
        }

    @property
    def proposal_id(self) -> str:
        return self.candidate_id

    @property
    def proposal_hash(self) -> str:
        return self.assert_integrity()

    @property
    def numeric_state_hash(self) -> str:
        return self.numeric_state.state_hash

    @property
    def endpoint_index(self) -> int:
        return self.numeric_state.endpoint_index

    @property
    def action_id(self) -> str:
        return self.numeric_state.action_id

    @property
    def action_hash(self) -> str:
        return self.numeric_state.action_hash

    @property
    def parent_mask(self) -> HardMask:
        return self.numeric_state.parent_mask

    @property
    def parent_mask_hash(self) -> str:
        return self.numeric_state.parent_mask_hash

    @property
    def resulting_mask(self) -> HardMask:
        return self.numeric_state.resulting_mask

    @property
    def resulting_mask_hash(self) -> str:
        return self.numeric_state.resulting_mask_hash

    @property
    def route_axis_hash(self) -> str:
        return self.numeric_state.route_axis_hash

    @property
    def removed_routes(self) -> V1RemovedRouteSetIdentity:
        return self.numeric_state.removed_routes

    @property
    def removed_count(self) -> int:
        return self.numeric_state.removed_count

    @property
    def removed_route_index_hash(self) -> str:
        return self.numeric_state.removed_route_index_hash

    @property
    def removed_route_ids(self) -> tuple[()]:
        return ()

    @property
    def compact_removed_indices_available(self) -> bool:
        return False

    @property
    def resource_gain(self) -> float:
        return self.numeric_state.resource_gain

    @property
    def atom_cost_source_hash(self) -> str:
        return self.numeric_state.atom_cost_source_hash

    @property
    def resource_gain_receipt_hash(self) -> str:
        return self.numeric_state.resource_gain_receipt_hash

    @property
    def fq_interval_hash(self) -> str:
        return self.numeric_state.fq_interval_hash

    @property
    def f_lower(self) -> float:
        return self.numeric_state.f_lower

    @property
    def f_upper(self) -> float:
        return self.numeric_state.f_upper

    @property
    def q_lower(self) -> float:
        return self.numeric_state.q_lower

    @property
    def q_upper(self) -> float:
        return self.numeric_state.q_upper

    @property
    def sqrt_q_lower(self) -> float:
        return self.numeric_state.sqrt_q_lower

    @property
    def sqrt_q_upper(self) -> float:
        return self.numeric_state.sqrt_q_upper

    @property
    def f_units(self) -> str:
        return self.numeric_state.f_units

    @property
    def q_units(self) -> str:
        return self.numeric_state.q_units

    @property
    def sqrt_q_units(self) -> str:
        return self.numeric_state.sqrt_q_units

    @property
    def f_upper_budget(self) -> float:
        return self.numeric_state.f_upper_budget

    @property
    def q_upper_budget(self) -> float:
        return self.numeric_state.q_upper_budget

    @property
    def f_upper_budget_ratio(self) -> float | None:
        return self.numeric_state.f_upper_budget_ratio

    @property
    def q_upper_budget_ratio(self) -> float | None:
        return self.numeric_state.q_upper_budget_ratio

    @property
    def sqrt_q_upper_budget_ratio(self) -> float | None:
        return self.numeric_state.sqrt_q_upper_budget_ratio

    @property
    def fq_budget_hash(self) -> str:
        return self.numeric_state.fq_budget_hash

    @property
    def decisive_fq_commit_hash(self) -> str:
        return self.numeric_state.decisive_fq_commit_hash

    @property
    def fq_queue_authorization_hash(self) -> str:
        return self.numeric_state.fq_queue_authorization_hash

    @property
    def sampled_diagnostic_lineage_hash(self) -> str:
        return self.numeric_state.sampled_diagnostic_lineage_hash

    def resolve_removed_routes(self, parent_mask: HardMask) -> RemovedRouteIndexSet:
        if type(parent_mask) is not HardMask:
            raise TypeError("compact F/Q delta resolution requires an exact parent mask")
        if _fresh_hard_mask_hash(parent_mask, name="F/Q proposal parent") != (
            self.parent_mask_hash
        ):
            raise ValueError("F/Q proposal parent differs from its bound identity")
        resolved = self.resulting_mask.removed_index_set_from(parent_mask)
        if (
            resolved.count != self.removed_count
            or resolved.index_hash != self.removed_route_index_hash
        ):
            raise ValueError("resolved F/Q proposal delta differs from its identity")
        return resolved

    @property
    def can_certify_decision(self) -> bool:
        return False

    @property
    def can_commit_mask(self) -> bool:
        return False

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        self.__post_init__()
        if expected_hash is not None and self.candidate_id != expected_hash:
            raise ValueError("compact F/Q proposal hash does not match")
        return self.candidate_id

    def assert_can_certify_decision(self) -> None:
        raise PermissionError("compact F/Q proposals are not scientific certificates")

    def assert_can_commit_mask(self) -> None:
        raise PermissionError("compact F/Q proposals cannot commit a hard mask")


def _issue_v1_fq_candidate_proposal(
    *,
    numeric_state: V1FQActionNumericState,
    domain_hash: str,
    constraints_hash: str,
) -> V1FQCandidateProposal:
    provisional = {
        "numeric_state": numeric_state,
        "domain_hash": domain_hash,
        "constraints_hash": constraints_hash,
        "proposal_only": True,
        "proposal_witness_only": True,
        "fresh_relinearization_required": True,
        "exact_registered_surrogate_evaluation": False,
        "exact_intervention_damage_evaluated": False,
        "fq_monotonicity_assumed": False,
        "candidate_freeze_authority": False,
        "bce_oracle_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
        "execution_authority": False,
    }
    shell = object.__new__(V1FQCandidateProposal)
    for name, value in provisional.items():
        object.__setattr__(shell, name, value)
    object.__setattr__(shell, "schema", _FQ_PROPOSAL_SCHEMA)
    receipt_hash = content_hash(
        shell._evaluation_payload(),
        prefix="partition-free-v1-fq-candidate-evaluation-",
    )
    candidate_id = content_hash(
        {
            "evaluation_receipt_hash": receipt_hash,
            "resulting_mask_hash": numeric_state.resulting_mask_hash,
            "removed_route_index_hash": numeric_state.removed_route_index_hash,
            "fq_interval_hash": numeric_state.fq_interval_hash,
        },
        prefix="partition-free-v1-fq-candidate-",
    )
    return V1FQCandidateProposal(
        _issuer_token=_FQ_PROPOSAL_TOKEN,
        **provisional,
        evaluation_receipt_hash=receipt_hash,
        candidate_id=candidate_id,
    )


@dataclass(frozen=True, slots=True, init=False)
class V1FQCandidateCompositionResult:
    """Exact materialization of one authorization-declared F/Q queue.

    ``ranked_proposals`` is provided only as a controller compatibility alias;
    the tuple is not ranked here.  It is the caller-authorized physical
    resource-descending queue, preserved without filtering or reordering.
    """

    action_numeric_states: tuple[V1FQActionNumericState, ...]
    proposals: tuple[V1FQCandidateProposal, ...]
    domain_hash: str
    constraints_hash: str
    fq_queue_authorization_hash: str
    decisive_fq_commit_hash: str
    fq_budget_hash: str
    fq_certification_receipt_hash: str
    sampled_diagnostic_lineage_hash: str
    atom_cost_source_hash: str
    parent_mask_hash: str
    route_axis_hash: str
    route_count: int
    queue_endpoint_indices: tuple[int, ...]
    queue_action_hashes: tuple[str, ...]
    queue_resulting_mask_hashes: tuple[str, ...]
    queue_resource_gain_receipt_hashes: tuple[str, ...]
    validated_action_count: int
    proposal_count: int
    composer_id: str
    selection_receipt_hash: str
    authorized_finite_queue_materialized_exactly: bool
    arbitrary_certified_endpoint_subset_preserved: bool
    resource_descending_order_verified: bool
    filtering_performed: bool
    reordering_performed: bool
    fq_monotonicity_assumed: bool
    claims_global_powerset_optimum: bool
    exact_registered_surrogate_evaluation: bool
    exact_intervention_damage_evaluated: bool
    proposal_only: bool
    proposal_witness_only: bool
    fresh_relinearization_required: bool
    candidate_freeze_authority: bool
    bce_oracle_authority: bool
    mask_commit_authority: bool
    terminal_authority: bool
    pareto_authority: bool
    execution_authority: bool
    schema: str = _FQ_RESULT_SCHEMA
    _result_hash: str | None = field(init=False, default=None, repr=False, compare=False)

    def __init__(self, *, _issuer_token: object, **values: object) -> None:
        if _issuer_token is not _FQ_RESULT_TOKEN:
            raise TypeError(
                "V1FQCandidateCompositionResult can only be issued by its composer"
            )
        expected = tuple(
            name
            for name, declaration in self.__dataclass_fields__.items()
            if declaration.init and name != "schema"
        )
        if set(values) != set(expected):
            raise TypeError("compact F/Q result fields differ from their schema")
        for name in expected:
            object.__setattr__(self, name, values[name])
        object.__setattr__(self, "schema", _FQ_RESULT_SCHEMA)
        object.__setattr__(self, "_result_hash", None)
        self.__post_init__()

    def __post_init__(self) -> None:
        states = tuple(self.action_numeric_states)
        proposals = tuple(self.proposals)
        if not states or any(type(item) is not V1FQActionNumericState for item in states):
            raise TypeError("compact F/Q result requires issued action numeric states")
        if len(states) != len(proposals) or any(
            type(item) is not V1FQCandidateProposal for item in proposals
        ):
            raise TypeError("compact F/Q result has an incomplete proposal axis")
        for state, proposal in zip(states, proposals, strict=True):
            state.assert_integrity()
            if proposal.numeric_state is not state:
                raise ValueError("compact F/Q proposal recontextualized its numeric state")
            if (
                proposal.domain_hash != self.domain_hash
                or proposal.constraints_hash != self.constraints_hash
            ):
                raise ValueError("compact F/Q proposal names another composition")
        first = states[0]
        shared = (
            ("fq_queue_authorization_hash", first.fq_queue_authorization_hash),
            ("decisive_fq_commit_hash", first.decisive_fq_commit_hash),
            ("fq_budget_hash", first.fq_budget_hash),
            ("fq_certification_receipt_hash", first.fq_certification_receipt_hash),
            ("sampled_diagnostic_lineage_hash", first.sampled_diagnostic_lineage_hash),
            ("atom_cost_source_hash", first.atom_cost_source_hash),
            ("parent_mask_hash", first.parent_mask_hash),
            ("route_axis_hash", first.route_axis_hash),
        )
        if any(getattr(self, name) != expected for name, expected in shared):
            raise ValueError("compact F/Q result changed a source identity")
        if self.route_count != first.route_count or any(
            item.fq_queue_authorization_hash != first.fq_queue_authorization_hash
            or item.decisive_fq_commit_hash != first.decisive_fq_commit_hash
            or item.fq_budget_hash != first.fq_budget_hash
            or item.fq_certification_receipt_hash
            != first.fq_certification_receipt_hash
            or item.sampled_diagnostic_lineage_hash
            != first.sampled_diagnostic_lineage_hash
            or item.atom_cost_source_hash != first.atom_cost_source_hash
            or item.parent_mask_hash != first.parent_mask_hash
            or item.route_axis_hash != first.route_axis_hash
            or item.route_count != first.route_count
            for item in states
        ):
            raise ValueError("compact F/Q states do not share one authorized coordinate")
        endpoint_axis = tuple(item.endpoint_index for item in states)
        action_axis = tuple(item.action_hash for item in states)
        mask_axis = tuple(item.resulting_mask_hash for item in states)
        resource_receipts = tuple(item.resource_gain_receipt_hash for item in states)
        if (
            self.queue_endpoint_indices != endpoint_axis
            or self.queue_action_hashes != action_axis
            or self.queue_resulting_mask_hashes != mask_axis
            or self.queue_resource_gain_receipt_hashes != resource_receipts
        ):
            raise ValueError("compact F/Q result silently changed its queue axis")
        if any(
            right.resource_gain > left.resource_gain
            for left, right in zip(states, states[1:], strict=False)
        ):
            raise ValueError("compact F/Q queue is not resource descending")
        if self.domain_hash != _fq_compact_domain_hash(states):
            raise ValueError("compact F/Q domain hash is not reproducible")
        if self.constraints_hash != _fq_constraints_hash(first):
            raise ValueError("compact F/Q constraints hash is not reproducible")
        validated = _positive_int(self.validated_action_count, "validated_action_count")
        count = _positive_int(self.proposal_count, "proposal_count")
        if validated != len(states) or count != len(proposals):
            raise ValueError("compact F/Q result filtered an authorized endpoint")
        if self.composer_id != FQ_AUTHORIZED_RESOURCE_DESCENDING_COMPOSER_ID:
            raise ValueError("unsupported compact F/Q composer")
        firewall = (
            self.authorized_finite_queue_materialized_exactly,
            self.arbitrary_certified_endpoint_subset_preserved,
            self.resource_descending_order_verified,
            not self.filtering_performed,
            not self.reordering_performed,
            not self.fq_monotonicity_assumed,
            not self.claims_global_powerset_optimum,
            not self.exact_registered_surrogate_evaluation,
            not self.exact_intervention_damage_evaluated,
            self.proposal_only,
            self.proposal_witness_only,
            self.fresh_relinearization_required,
            not self.candidate_freeze_authority,
            not self.bce_oracle_authority,
            not self.mask_commit_authority,
            not self.terminal_authority,
            not self.pareto_authority,
            not self.execution_authority,
        )
        if any(type(flag) is not bool or not flag for flag in firewall):
            raise ValueError("compact F/Q result crossed its proposal-only firewall")
        if self.schema != _FQ_RESULT_SCHEMA:
            raise ValueError("unsupported compact F/Q result schema")
        expected_receipt = content_hash(
            self._selection_payload(proposals),
            prefix="partition-free-v1-fq-candidate-selection-",
        )
        if self.selection_receipt_hash != expected_receipt:
            raise ValueError("compact F/Q selection receipt is not reproducible")
        object.__setattr__(self, "action_numeric_states", states)
        object.__setattr__(self, "proposals", proposals)
        object.__setattr__(self, "validated_action_count", validated)
        object.__setattr__(self, "proposal_count", count)
        self.assert_integrity()

    def _selection_payload(
        self, proposals: Sequence[V1FQCandidateProposal]
    ) -> dict[str, object]:
        return {
            "schema": "partition_free_v1_fq_candidate_selection_receipt_v1",
            "composer_id": self.composer_id,
            "domain_hash": self.domain_hash,
            "constraints_hash": self.constraints_hash,
            "fq_queue_authorization_hash": self.fq_queue_authorization_hash,
            "decisive_fq_commit_hash": self.decisive_fq_commit_hash,
            "fq_budget_hash": self.fq_budget_hash,
            "fq_certification_receipt_hash": self.fq_certification_receipt_hash,
            "sampled_diagnostic_lineage_hash": self.sampled_diagnostic_lineage_hash,
            "atom_cost_source_hash": self.atom_cost_source_hash,
            "queue_endpoint_indices": self.queue_endpoint_indices,
            "queue_action_hashes": self.queue_action_hashes,
            "queue_candidate_ids": tuple(item.candidate_id for item in proposals),
            "validated_authorized_action_count": self.validated_action_count,
            "proposal_count": self.proposal_count,
            "authorized_finite_queue_materialized_exactly": True,
            "arbitrary_certified_endpoint_subset_preserved": True,
            "resource_descending_order_verified": True,
            "filtering_performed": False,
            "reordering_performed": False,
            "fq_monotonicity_assumed": False,
            "claims_global_powerset_optimum": False,
            "exact_registered_surrogate_evaluation": False,
            "exact_intervention_damage_evaluated": False,
            "proposal_only": True,
            "proposal_witness_only": True,
            "fresh_relinearization_required": True,
        }

    @property
    def ranked_proposals(self) -> tuple[V1FQCandidateProposal, ...]:
        """Controller alias; this is the untouched authorization queue."""

        return self.proposals

    @property
    def evaluated_action_count(self) -> int:
        """Compatibility count of F/Q-validated, not surrogate-evaluated, actions."""

        return self.validated_action_count

    @property
    def eligible_action_count(self) -> int:
        return self.proposal_count

    @property
    def surrogate_evaluated_action_count(self) -> int:
        return 0

    @property
    def selected_proposal(self) -> V1FQCandidateProposal:
        return self.proposals[0]

    @property
    def result_hash(self) -> str:
        return self.assert_integrity()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "action_numeric_state_hashes": tuple(
                item.state_hash for item in self.action_numeric_states
            ),
            "candidate_ids": tuple(item.candidate_id for item in self.proposals),
            "domain_hash": self.domain_hash,
            "constraints_hash": self.constraints_hash,
            "fq_queue_authorization_hash": self.fq_queue_authorization_hash,
            "decisive_fq_commit_hash": self.decisive_fq_commit_hash,
            "fq_budget_hash": self.fq_budget_hash,
            "fq_certification_receipt_hash": self.fq_certification_receipt_hash,
            "sampled_diagnostic_lineage_hash": self.sampled_diagnostic_lineage_hash,
            "atom_cost_source_hash": self.atom_cost_source_hash,
            "parent_mask_hash": self.parent_mask_hash,
            "route_axis_hash": self.route_axis_hash,
            "route_count": self.route_count,
            "queue_endpoint_indices": self.queue_endpoint_indices,
            "queue_action_hashes": self.queue_action_hashes,
            "queue_resulting_mask_hashes": self.queue_resulting_mask_hashes,
            "queue_resource_gain_receipt_hashes": (
                self.queue_resource_gain_receipt_hashes
            ),
            "validated_action_count": self.validated_action_count,
            "proposal_count": self.proposal_count,
            "surrogate_evaluated_action_count": 0,
            "composer_id": self.composer_id,
            "selection_receipt_hash": self.selection_receipt_hash,
            "authorized_finite_queue_materialized_exactly": (
                self.authorized_finite_queue_materialized_exactly
            ),
            "arbitrary_certified_endpoint_subset_preserved": (
                self.arbitrary_certified_endpoint_subset_preserved
            ),
            "resource_descending_order_verified": (
                self.resource_descending_order_verified
            ),
            "filtering_performed": self.filtering_performed,
            "reordering_performed": self.reordering_performed,
            "fq_monotonicity_assumed": self.fq_monotonicity_assumed,
            "claims_global_powerset_optimum": self.claims_global_powerset_optimum,
            "exact_registered_surrogate_evaluation": (
                self.exact_registered_surrogate_evaluation
            ),
            "exact_intervention_damage_evaluated": (
                self.exact_intervention_damage_evaluated
            ),
            "proposal_only": self.proposal_only,
            "proposal_witness_only": self.proposal_witness_only,
            "fresh_relinearization_required": self.fresh_relinearization_required,
            "candidate_freeze_authority": self.candidate_freeze_authority,
            "bce_oracle_authority": self.bce_oracle_authority,
            "mask_commit_authority": self.mask_commit_authority,
            "terminal_authority": self.terminal_authority,
            "pareto_authority": self.pareto_authority,
            "execution_authority": self.execution_authority,
        }

    def assert_integrity(self, expected_hash: str | None = None) -> str:
        for state, proposal in zip(
            self.action_numeric_states, self.proposals, strict=True
        ):
            state.assert_integrity()
            if proposal.numeric_state is not state:
                raise ValueError("compact F/Q result proposal axis was tampered")
            proposal.assert_integrity()
        observed = content_hash(
            self._payload(),
            prefix="partition-free-v1-fq-candidate-composition-",
        )
        if self._result_hash is not None and observed != self._result_hash:
            raise ValueError("compact F/Q candidate composition was tampered")
        if expected_hash is not None and observed != expected_hash:
            raise ValueError("compact F/Q composition hash does not match")
        object.__setattr__(self, "_result_hash", observed)
        return observed

    @property
    def can_certify_decision(self) -> bool:
        return False

    @property
    def can_commit_mask(self) -> bool:
        return False

    def assert_can_certify_decision(self) -> None:
        raise PermissionError("compact F/Q composition is not a scientific certificate")

    def assert_can_commit_mask(self) -> None:
        raise PermissionError("compact F/Q composition cannot commit a hard mask")


def issue_v1_fq_action_numeric_states(
    *,
    authorization: object,
    sampled_diagnostic_lineage_hash: str,
    atom_cost_source_hash: str,
    exact_resource_gain_rows: Sequence[tuple[int, str, float]],
) -> tuple[V1FQActionNumericState, ...]:
    """Issue compact endpoint states from one exact decisive authorization.

    ``exact_resource_gain_rows`` must already be derived from the separately
    bound atom-cost source.  The row identity/order must exactly equal the
    authorization queue; this function never infers, sorts, or filters it.
    """

    # Local imports avoid the intentional dependency in the opposite
    # direction: ladder statistics and the F/Q bridge both use packed actions
    # defined in this module.
    from .modern_v1_fq_candidate_bridge import V1FQCandidateQueueAuthorization
    from .modern_v1_ladder_statistics import (
        V1FrozenLadderEndpointInterval,
    )

    if type(authorization) is not V1FQCandidateQueueAuthorization:
        raise TypeError("compact F/Q issuance requires the exact authorization type")
    authorization_hash = authorization.assert_integrity()
    sampled_hash = _nonempty(
        sampled_diagnostic_lineage_hash,
        "sampled_diagnostic_lineage_hash",
    )
    cost_source_hash = _nonempty(atom_cost_source_hash, "atom_cost_source_hash")
    rows: list[tuple[int, str, float]] = []
    for position, row in enumerate(tuple(exact_resource_gain_rows)):
        if isinstance(row, (str, bytes)):
            raise TypeError("exact resource-gain rows require three typed fields")
        values = tuple(row)
        if len(values) != 3:
            raise ValueError("exact resource-gain rows require three fields")
        endpoint = _positive_int(values[0], f"resource row {position} endpoint")
        action_hash = _nonempty(values[1], f"resource row {position} action_hash")
        gain = _finite_nonnegative(values[2], f"resource row {position} gain")
        rows.append((endpoint, action_hash, gain))
    expected_identity_axis = tuple(
        zip(
            authorization.queue_endpoint_indices_descending,
            authorization.queue_action_hashes_descending,
            strict=True,
        )
    )
    if tuple((endpoint, action) for endpoint, action, _gain in rows) != (
        expected_identity_axis
    ):
        raise ValueError(
            "resource-gain rows filtered, reordered, or relabeled the authorization queue"
        )
    if any(
        right[2] > left[2]
        for left, right in zip(rows, rows[1:], strict=False)
    ):
        raise ValueError("authorized F/Q queue is not resource descending")

    commit = authorization.final_commit
    commit_hash = commit.assert_integrity()
    receipt = commit.certification_receipt
    receipt_hash = receipt.assert_integrity()
    budget = authorization.budget
    budget_hash = budget.assert_integrity()
    interval_by_endpoint = {
        item.endpoint_index: item for item in receipt.endpoint_intervals
    }
    classification_by_endpoint = {
        item.endpoint_index: item for item in commit.assessment.classifications
    }
    states: list[V1FQActionNumericState] = []
    for endpoint, action_hash, resource_gain in rows:
        frozen = authorization.ladder.actions[endpoint - 1]
        frozen.assert_integrity(action_hash)
        packed = frozen.packed_removal_action
        if type(packed) is not V1PackedRemovalAction:
            raise TypeError("compact F/Q issuance requires exact packed actions")
        interval = interval_by_endpoint.get(endpoint)
        classification = classification_by_endpoint.get(endpoint)
        if type(interval) is not V1FrozenLadderEndpointInterval:
            raise TypeError("authorized F/Q endpoint lacks its exact interval")
        interval_hash = interval.assert_integrity()
        if (
            interval.action_hash != action_hash
            or classification is None
            or classification.action_hash != action_hash
            or classification.status != "certified_within_budgets"
            or interval.empty
            or not interval.upper_confidence_authority
            or interval.f_upper > budget.f_upper_budget
            or interval.q_upper > budget.q_upper_budget
        ):
            raise ValueError(
                "authorization endpoint is not independently certified within F/Q budgets"
            )
        parent = authorization.current_mask
        parent_hash = _fresh_hard_mask_hash(parent, name="authorized F/Q parent")
        child_hash = _fresh_hard_mask_hash(
            packed.resulting_mask,
            name="authorized F/Q resulting mask",
        )
        removed_count, removed_hash = (
            packed.resulting_mask.removed_index_count_and_hash_from(parent)
        )
        if (
            removed_count != packed.removed_count
            or removed_hash != packed.removed_route_index_hash
        ):
            raise ValueError("authorized packed action differs from its parent delta")
        resource_receipt = content_hash(
            _fq_resource_gain_receipt_payload(
                endpoint_index=endpoint,
                action_hash=action_hash,
                atom_cost_source_hash=cost_source_hash,
                parent_mask_hash=parent_hash,
                resulting_mask_hash=child_hash,
                route_axis_hash=parent.route_axis_hash,
                route_count=parent.route_count,
                removed_count=removed_count,
                removed_route_index_hash=removed_hash,
                resource_gain=resource_gain,
            ),
            prefix="partition-free-v1-fq-resource-gain-",
        )
        states.append(
            V1FQActionNumericState(
                _issuer_token=_FQ_ACTION_NUMERIC_STATE_TOKEN,
                endpoint_index=endpoint,
                action_id=frozen.action_id,
                action_hash=action_hash,
                parent_mask=parent,
                packed_action=packed,
                route_axis_hash=parent.route_axis_hash,
                route_count=parent.route_count,
                parent_mask_hash=parent_hash,
                resulting_mask_hash=child_hash,
                packed_action_hash=packed.action_hash,
                removed_count=removed_count,
                removed_route_index_hash=removed_hash,
                resource_gain=resource_gain,
                atom_cost_source_hash=cost_source_hash,
                resource_gain_receipt_hash=resource_receipt,
                f_lower=interval.f_lower,
                f_upper=interval.f_upper,
                q_lower=interval.q_lower,
                q_upper=interval.q_upper,
                sqrt_q_lower=interval.sqrt_q_lower,
                sqrt_q_upper=interval.sqrt_q_upper,
                f_units=interval.f_units,
                q_units=interval.q_units,
                sqrt_q_units=interval.sqrt_q_units,
                fq_interval_schema=interval.schema,
                fq_interval_hash=interval_hash,
                fq_interval_full_census=interval.full_census,
                fq_interval_population_census_complete=bool(
                    interval.population_census_complete
                ),
                fq_interval_lower_confidence_authority=(
                    interval.lower_confidence_authority
                ),
                fq_interval_upper_confidence_authority=(
                    interval.upper_confidence_authority
                ),
                fq_interval_observation_semantics=interval.observation_semantics,
                f_upper_budget=budget.f_upper_budget,
                q_upper_budget=budget.q_upper_budget,
                f_upper_budget_ratio=_fq_budget_ratio(
                    interval.f_upper,
                    budget.f_upper_budget,
                ),
                q_upper_budget_ratio=_fq_budget_ratio(
                    interval.q_upper,
                    budget.q_upper_budget,
                ),
                sqrt_q_upper_budget_ratio=_fq_budget_ratio(
                    interval.sqrt_q_upper,
                    math.sqrt(budget.q_upper_budget),
                ),
                fq_budget_hash=budget_hash,
                fq_certification_receipt_hash=receipt_hash,
                decisive_fq_commit_hash=commit_hash,
                fq_queue_authorization_hash=authorization_hash,
                sampled_diagnostic_lineage_hash=sampled_hash,
                generation_index=authorization.generation_index,
                independently_certified_fq_interval=True,
                exact_scalar_resource_gain_from_bound_source=True,
                proposal_only=True,
                fq_monotonicity_assumed=False,
                exact_registered_surrogate_evaluation=False,
                exact_intervention_damage_evaluated=False,
                candidate_freeze_authority=False,
                bce_oracle_authority=False,
                mask_commit_authority=False,
                terminal_authority=False,
                pareto_authority=False,
                execution_authority=False,
            )
        )
    if not states:
        raise ValueError("F/Q authorization contains no queue endpoint")
    return tuple(states)


def compose_v1_fq_candidates_in_authorized_resource_descending_order(
    *,
    authorization: object,
    action_numeric_states: Sequence[V1FQActionNumericState],
) -> V1FQCandidateCompositionResult:
    """Materialize exactly the authorization-declared compact F/Q queue."""

    from .modern_v1_fq_candidate_bridge import V1FQCandidateQueueAuthorization

    if type(authorization) is not V1FQCandidateQueueAuthorization:
        raise TypeError("compact F/Q composition requires the exact authorization")
    authorization_hash = authorization.assert_integrity()
    states = tuple(action_numeric_states)
    if not states or any(type(item) is not V1FQActionNumericState for item in states):
        raise TypeError("compact F/Q composition requires issued action states")
    for state in states:
        state.assert_integrity()
    observed_axis = tuple((item.endpoint_index, item.action_hash) for item in states)
    expected_axis = tuple(
        zip(
            authorization.queue_endpoint_indices_descending,
            authorization.queue_action_hashes_descending,
            strict=True,
        )
    )
    if observed_axis != expected_axis:
        raise ValueError(
            "compact F/Q states filtered or reordered the authorization queue"
        )
    if any(
        state.fq_queue_authorization_hash != authorization_hash
        or state.decisive_fq_commit_hash != authorization.commit_hash
        or state.fq_budget_hash != authorization.budget_hash
        or state.parent_mask_hash != authorization.current_mask.mask_hash
        or state.generation_index != authorization.generation_index
        for state in states
    ):
        raise ValueError("compact F/Q state belongs to another authorization")
    if any(
        right.resource_gain > left.resource_gain
        for left, right in zip(states, states[1:], strict=False)
    ):
        raise ValueError("compact F/Q queue is not in resource-descending order")

    domain_hash = _fq_compact_domain_hash(states)
    constraints_hash = _fq_constraints_hash(states[0])
    proposals = tuple(
        _issue_v1_fq_candidate_proposal(
            numeric_state=state,
            domain_hash=domain_hash,
            constraints_hash=constraints_hash,
        )
        for state in states
    )
    result_values: dict[str, object] = {
        "action_numeric_states": states,
        "proposals": proposals,
        "domain_hash": domain_hash,
        "constraints_hash": constraints_hash,
        "fq_queue_authorization_hash": authorization_hash,
        "decisive_fq_commit_hash": states[0].decisive_fq_commit_hash,
        "fq_budget_hash": states[0].fq_budget_hash,
        "fq_certification_receipt_hash": states[0].fq_certification_receipt_hash,
        "sampled_diagnostic_lineage_hash": states[0].sampled_diagnostic_lineage_hash,
        "atom_cost_source_hash": states[0].atom_cost_source_hash,
        "parent_mask_hash": states[0].parent_mask_hash,
        "route_axis_hash": states[0].route_axis_hash,
        "route_count": states[0].route_count,
        "queue_endpoint_indices": tuple(item.endpoint_index for item in states),
        "queue_action_hashes": tuple(item.action_hash for item in states),
        "queue_resulting_mask_hashes": tuple(
            item.resulting_mask_hash for item in states
        ),
        "queue_resource_gain_receipt_hashes": tuple(
            item.resource_gain_receipt_hash for item in states
        ),
        "validated_action_count": len(states),
        "proposal_count": len(proposals),
        "composer_id": FQ_AUTHORIZED_RESOURCE_DESCENDING_COMPOSER_ID,
        "authorized_finite_queue_materialized_exactly": True,
        "arbitrary_certified_endpoint_subset_preserved": True,
        "resource_descending_order_verified": True,
        "filtering_performed": False,
        "reordering_performed": False,
        "fq_monotonicity_assumed": False,
        "claims_global_powerset_optimum": False,
        "exact_registered_surrogate_evaluation": False,
        "exact_intervention_damage_evaluated": False,
        "proposal_only": True,
        "proposal_witness_only": True,
        "fresh_relinearization_required": True,
        "candidate_freeze_authority": False,
        "bce_oracle_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
        "execution_authority": False,
    }
    shell = object.__new__(V1FQCandidateCompositionResult)
    for name, value in result_values.items():
        object.__setattr__(shell, name, value)
    object.__setattr__(shell, "schema", _FQ_RESULT_SCHEMA)
    receipt_hash = content_hash(
        shell._selection_payload(proposals),
        prefix="partition-free-v1-fq-candidate-selection-",
    )
    return V1FQCandidateCompositionResult(
        _issuer_token=_FQ_RESULT_TOKEN,
        **result_values,
        selection_receipt_hash=receipt_hash,
    )


def issue_v1_fqr_action_numeric_states(
    *,
    authorization: object,
    sampled_diagnostic_lineage_hash: str,
    atom_cost_source_hash: str,
    exact_resource_gain_rows: Sequence[tuple[int, str, float]],
) -> tuple[V1FQActionNumericState, ...]:
    """Issue compact endpoint states from one canonical F/Q/R authorization.

    The compatibility carrier is the existing compact F/Q proposal state, but
    the authoritative identities come from the joint F/Q/R graph: corrected
    work bounds, literal-Jacobian response bounds, directional-remainder
    screening, and the no-independent-R budget derivation.
    """

    from .modern_v1_fqr_candidate_bridge import V1FQRCandidateQueueAuthorization

    if type(authorization) is not V1FQRCandidateQueueAuthorization:
        raise TypeError("compact F/Q/R issuance requires the exact authorization type")
    authorization_hash = authorization.assert_integrity()
    sampled_hash = _nonempty(
        sampled_diagnostic_lineage_hash,
        "sampled_diagnostic_lineage_hash",
    )
    cost_source_hash = _nonempty(atom_cost_source_hash, "atom_cost_source_hash")
    rows: list[tuple[int, str, float]] = []
    for position, row in enumerate(tuple(exact_resource_gain_rows)):
        if isinstance(row, (str, bytes)):
            raise TypeError("exact resource-gain rows require three typed fields")
        values = tuple(row)
        if len(values) != 3:
            raise ValueError("exact resource-gain rows require three fields")
        endpoint = _positive_int(values[0], f"resource row {position} endpoint")
        action_hash = _nonempty(values[1], f"resource row {position} action_hash")
        gain = _finite_nonnegative(values[2], f"resource row {position} gain")
        rows.append((endpoint, action_hash, gain))
    expected_identity_axis = tuple(
        zip(
            authorization.queue_endpoint_indices_descending,
            authorization.queue_action_hashes_descending,
            strict=True,
        )
    )
    if tuple((endpoint, action) for endpoint, action, _gain in rows) != (
        expected_identity_axis
    ):
        raise ValueError(
            "resource-gain rows filtered, reordered, or relabeled the F/Q/R queue"
        )
    if any(right[2] > left[2] for left, right in zip(rows, rows[1:], strict=False)):
        raise ValueError("authorized F/Q/R queue is not resource descending")

    receipt = authorization.receipt
    receipt_hash = receipt.assert_integrity()
    design = authorization.design
    design_hash = design.assert_integrity()
    decision_hash = authorization.decision.assert_integrity()
    bound_by_endpoint = {item.endpoint_index: item for item in receipt.fq_bounds}
    certified = {
        int(row[0]): row
        for row in authorization.decision.endpoint_outcomes
        if bool(row[11])
    }
    states: list[V1FQActionNumericState] = []
    for endpoint, action_hash, resource_gain in rows:
        frozen = authorization.ladder.actions[endpoint - 1]
        frozen.assert_integrity(action_hash)
        packed = frozen.packed_removal_action
        if type(packed) is not V1PackedRemovalAction:
            raise TypeError("compact F/Q/R issuance requires exact packed actions")
        bound = bound_by_endpoint.get(endpoint)
        outcome = certified.get(endpoint)
        if bound is None or outcome is None:
            raise ValueError("authorized F/Q/R endpoint lacks its certified bounds")
        bound_hash = bound.assert_integrity()
        if (
            bound.action_hash != action_hash
            or str(outcome[1]) != action_hash
            or bool(outcome[11]) is not True
            or bound.empty
            or bound.f_upper > design.tau_f
            or bound.q_upper > design.tau_q
        ):
            raise ValueError(
                "authorization endpoint is not independently certified within F/Q/R budgets"
            )
        parent = authorization.current_mask
        parent_hash = _fresh_hard_mask_hash(parent, name="authorized F/Q/R parent")
        child_hash = _fresh_hard_mask_hash(
            packed.resulting_mask,
            name="authorized F/Q/R resulting mask",
        )
        removed_count, removed_hash = (
            packed.resulting_mask.removed_index_count_and_hash_from(parent)
        )
        if (
            removed_count != packed.removed_count
            or removed_hash != packed.removed_route_index_hash
        ):
            raise ValueError("authorized F/Q/R packed action differs from its delta")
        resource_receipt = content_hash(
            _fq_resource_gain_receipt_payload(
                endpoint_index=endpoint,
                action_hash=action_hash,
                atom_cost_source_hash=cost_source_hash,
                parent_mask_hash=parent_hash,
                resulting_mask_hash=child_hash,
                route_axis_hash=parent.route_axis_hash,
                route_count=parent.route_count,
                removed_count=removed_count,
                removed_route_index_hash=removed_hash,
                resource_gain=resource_gain,
            ),
            prefix="partition-free-v1-fq-resource-gain-",
        )
        states.append(
            V1FQActionNumericState(
                _issuer_token=_FQ_ACTION_NUMERIC_STATE_TOKEN,
                endpoint_index=endpoint,
                action_id=frozen.action_id,
                action_hash=action_hash,
                parent_mask=parent,
                packed_action=packed,
                route_axis_hash=parent.route_axis_hash,
                route_count=parent.route_count,
                parent_mask_hash=parent_hash,
                resulting_mask_hash=child_hash,
                packed_action_hash=packed.action_hash,
                removed_count=removed_count,
                removed_route_index_hash=removed_hash,
                resource_gain=resource_gain,
                atom_cost_source_hash=cost_source_hash,
                resource_gain_receipt_hash=resource_receipt,
                f_lower=0.0,
                f_upper=bound.f_upper,
                q_lower=0.0,
                q_upper=bound.q_upper,
                sqrt_q_lower=0.0,
                sqrt_q_upper=math.sqrt(bound.q_upper),
                f_units=_FQ_F_UNITS,
                q_units=_FQ_Q_UNITS,
                sqrt_q_units=_FQ_SQRT_Q_UNITS,
                fq_interval_schema=bound.schema,
                fq_interval_hash=bound_hash,
                fq_interval_full_census=False,
                fq_interval_population_census_complete=False,
                fq_interval_lower_confidence_authority=False,
                fq_interval_upper_confidence_authority=True,
                fq_interval_observation_semantics=(
                    "joint_fqr_corrected_work_literal_jacobian_and_directional_"
                    "remainder_upper_screen_v1"
                ),
                f_upper_budget=design.tau_f,
                q_upper_budget=design.tau_q,
                f_upper_budget_ratio=_fq_budget_ratio(bound.f_upper, design.tau_f),
                q_upper_budget_ratio=_fq_budget_ratio(bound.q_upper, design.tau_q),
                sqrt_q_upper_budget_ratio=_fq_budget_ratio(
                    math.sqrt(bound.q_upper),
                    math.sqrt(design.tau_q),
                ),
                fq_budget_hash=design_hash,
                fq_certification_receipt_hash=receipt_hash,
                decisive_fq_commit_hash=decision_hash,
                fq_queue_authorization_hash=authorization_hash,
                sampled_diagnostic_lineage_hash=sampled_hash,
                generation_index=authorization.proposal_generation,
                independently_certified_fq_interval=True,
                exact_scalar_resource_gain_from_bound_source=True,
                proposal_only=True,
                fq_monotonicity_assumed=False,
                exact_registered_surrogate_evaluation=False,
                exact_intervention_damage_evaluated=False,
                candidate_freeze_authority=False,
                bce_oracle_authority=False,
                mask_commit_authority=False,
                terminal_authority=False,
                pareto_authority=False,
                execution_authority=False,
            )
        )
    if not states:
        raise ValueError("F/Q/R authorization contains no queue endpoint")
    return tuple(states)


def compose_v1_fqr_candidates_in_authorized_resource_descending_order(
    *,
    authorization: object,
    action_numeric_states: Sequence[V1FQActionNumericState],
) -> V1FQCandidateCompositionResult:
    """Materialize exactly the authorization-declared compact F/Q/R queue."""

    from .modern_v1_fqr_candidate_bridge import V1FQRCandidateQueueAuthorization

    if type(authorization) is not V1FQRCandidateQueueAuthorization:
        raise TypeError("compact F/Q/R composition requires the exact authorization")
    authorization_hash = authorization.assert_integrity()
    states = tuple(action_numeric_states)
    if not states or any(type(item) is not V1FQActionNumericState for item in states):
        raise TypeError("compact F/Q/R composition requires issued action states")
    for state in states:
        state.assert_integrity()
    observed_axis = tuple((item.endpoint_index, item.action_hash) for item in states)
    expected_axis = tuple(
        zip(
            authorization.queue_endpoint_indices_descending,
            authorization.queue_action_hashes_descending,
            strict=True,
        )
    )
    if observed_axis != expected_axis:
        raise ValueError(
            "compact F/Q/R states filtered or reordered the authorization queue"
        )
    decision_hash = authorization.decision.decision_hash
    design_hash = authorization.design.design_hash
    if any(
        state.fq_queue_authorization_hash != authorization_hash
        or state.decisive_fq_commit_hash != decision_hash
        or state.fq_budget_hash != design_hash
        or state.parent_mask_hash != authorization.current_mask.mask_hash
        or state.generation_index != authorization.proposal_generation
        for state in states
    ):
        raise ValueError("compact F/Q/R state belongs to another authorization")
    if any(right.resource_gain > left.resource_gain for left, right in zip(states, states[1:], strict=False)):
        raise ValueError("compact F/Q/R queue is not in resource-descending order")

    domain_hash = _fq_compact_domain_hash(states)
    constraints_hash = _fq_constraints_hash(states[0])
    proposals = tuple(
        _issue_v1_fq_candidate_proposal(
            numeric_state=state,
            domain_hash=domain_hash,
            constraints_hash=constraints_hash,
        )
        for state in states
    )
    result_values: dict[str, object] = {
        "action_numeric_states": states,
        "proposals": proposals,
        "domain_hash": domain_hash,
        "constraints_hash": constraints_hash,
        "fq_queue_authorization_hash": authorization_hash,
        "decisive_fq_commit_hash": states[0].decisive_fq_commit_hash,
        "fq_budget_hash": states[0].fq_budget_hash,
        "fq_certification_receipt_hash": states[0].fq_certification_receipt_hash,
        "sampled_diagnostic_lineage_hash": states[0].sampled_diagnostic_lineage_hash,
        "atom_cost_source_hash": states[0].atom_cost_source_hash,
        "parent_mask_hash": states[0].parent_mask_hash,
        "route_axis_hash": states[0].route_axis_hash,
        "route_count": states[0].route_count,
        "queue_endpoint_indices": tuple(item.endpoint_index for item in states),
        "queue_action_hashes": tuple(item.action_hash for item in states),
        "queue_resulting_mask_hashes": tuple(item.resulting_mask_hash for item in states),
        "queue_resource_gain_receipt_hashes": tuple(
            item.resource_gain_receipt_hash for item in states
        ),
        "validated_action_count": len(states),
        "proposal_count": len(proposals),
        "composer_id": FQ_AUTHORIZED_RESOURCE_DESCENDING_COMPOSER_ID,
        "authorized_finite_queue_materialized_exactly": True,
        "arbitrary_certified_endpoint_subset_preserved": True,
        "resource_descending_order_verified": True,
        "filtering_performed": False,
        "reordering_performed": False,
        "fq_monotonicity_assumed": False,
        "claims_global_powerset_optimum": False,
        "exact_registered_surrogate_evaluation": False,
        "exact_intervention_damage_evaluated": False,
        "proposal_only": True,
        "proposal_witness_only": True,
        "fresh_relinearization_required": True,
        "candidate_freeze_authority": False,
        "bce_oracle_authority": False,
        "mask_commit_authority": False,
        "terminal_authority": False,
        "pareto_authority": False,
        "execution_authority": False,
    }
    shell = object.__new__(V1FQCandidateCompositionResult)
    for name, value in result_values.items():
        object.__setattr__(shell, name, value)
    object.__setattr__(shell, "schema", _FQ_RESULT_SCHEMA)
    receipt_hash = content_hash(
        shell._selection_payload(proposals),
        prefix="partition-free-v1-fq-candidate-selection-",
    )
    return V1FQCandidateCompositionResult(
        _issuer_token=_FQ_RESULT_TOKEN,
        **result_values,
        selection_receipt_hash=receipt_hash,
    )


__all__ = [
    "EXACT_FINITE_DOMAIN_COMPOSER_ID",
    "FQ_AUTHORIZED_RESOURCE_DESCENDING_COMPOSER_ID",
    "FROZEN_RESOURCE_DESCENDING_FRONTIER_COMPOSER_ID",
    "FD_SPECTRAL_UPPER_EQUATION_ID",
    "NO_HALF_SURROGATE_EQUATION_ID",
    "V1CandidateCompositionConstraints",
    "V1CandidateCompositionResult",
    "V1CandidateNumericState",
    "V1CandidateProposal",
    "V1FQActionNumericState",
    "V1FQCandidateCompositionResult",
    "V1FQCandidateProposal",
    "V1FiniteRemovalDomain",
    "V1PackedRemovalAction",
    "V1RemovedRouteSetIdentity",
    "V1RouteIdAxisIdentity",
    "compose_v1_candidates_exact",
    "compose_v1_candidates_in_frozen_resource_descending_order",
    "compose_v1_fq_candidates_in_authorized_resource_descending_order",
    "compose_v1_fqr_candidates_in_authorized_resource_descending_order",
    "issue_v1_fq_action_numeric_states",
    "issue_v1_fqr_action_numeric_states",
    "issue_v1_candidate_numeric_state",
]
