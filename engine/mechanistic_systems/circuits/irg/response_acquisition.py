"""Frozen, adaptive whole-route response acquisition for IRG v1--v4.

The adapter in this module only chooses nested finite-population context
prefixes and records whether the downstream *discrete* action has stabilized.
Its covariance forecast is allocation guidance, never a safety authorization.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from mechanistic_systems.core.axis_identity import (
    canonical_unique_string_axis,
    ordered_string_axis_hash,
)
from .acquisition import (
    AcquisitionForecast,
    DecisionUncertaintySnapshot,
    UnifiedAcquisitionAdapter,
)
from .boundary_search import AdaptiveBoundarySearch
from .contracts import (
    ContextSelector,
    MeasurementOutcome,
    MeasurementProvenance,
    MeasurementRequest,
    content_hash,
)
from .costs import CostEstimate
from .precision import simultaneous_normal_critical_value
from .response_geometry import ResponseMomentTensor
from .trust_geometry import (
    GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY,
    GlobalTrustGeometryState,
)
from mechanistic_systems.measurement.in_process_response_summary import (
    claim_in_process_response_summary,
)
from mechanistic_systems.measurement.streaming_response_provider import (
    LocalAttributionMomentTensor,
    TripartitePullbackMomentTensor,
)


RESPONSE_ACQUISITION_STATE_CACHE_KEY = "irg_response_acquisition_state_v1"
SEALED_UNIFORM_PERMUTATION_ALGORITHM_V1 = "sha256_counter_fisher_yates_rejection_v1"
UNPARTITIONED_GEOMETRIC_ACTION_BLOCK_ID = "unpartitioned_compatibility_block"


# Sealed permutation receipts and response designs are immutable.  Replaying a
# million-context Fisher--Yates reveal or canonically serializing the same
# population on every nested-prefix request is therefore pure controller work,
# not additional verification.  Keep strong references beside object IDs so
# Python cannot recycle an ID into a false cache hit.  The small bound covers
# the two binary strata and active design while preventing campaign-scale
# retention.
_SEALED_IDENTITY_CACHE_LIMIT = 16
_SEALED_IDENTITY_CACHE_LOCK = RLock()
_VERIFIED_RECEIPTS: OrderedDict[int, SealedUniformPermutationReceipt] = OrderedDict()
_RECEIPT_CONTENT_HASHES: OrderedDict[
    int, tuple[SealedUniformPermutationReceipt, str]
] = OrderedDict()
_DESIGN_CONTENT_HASHES: OrderedDict[int, tuple[FrozenResponseContextDesign, str]] = (
    OrderedDict()
)


def _bounded_identity_put(
    cache: OrderedDict,
    owner: object,
    value: object | None = None,
) -> None:
    key = id(owner)
    cache[key] = owner if value is None else value
    cache.move_to_end(key)
    while len(cache) > _SEALED_IDENTITY_CACHE_LIMIT:
        cache.popitem(last=False)


def _mark_receipt_verified(receipt: SealedUniformPermutationReceipt) -> None:
    with _SEALED_IDENTITY_CACHE_LOCK:
        _bounded_identity_put(_VERIFIED_RECEIPTS, receipt)


def _receipt_is_verified(receipt: SealedUniformPermutationReceipt) -> bool:
    with _SEALED_IDENTITY_CACHE_LOCK:
        cached = _VERIFIED_RECEIPTS.get(id(receipt))
        if cached is receipt:
            _VERIFIED_RECEIPTS.move_to_end(id(receipt))
            return True
        return False


def _canonical_population(values: Sequence[str]) -> tuple[str, ...]:
    population = tuple(str(value) for value in values)
    if not population or len(set(population)) != len(population):
        raise ValueError(
            "sealed permutation requires a nonempty population of unique context IDs"
        )
    if population != tuple(sorted(population)):
        raise ValueError(
            "sealed permutation base population must use canonical ascending context-ID order"
        )
    return population


def _entropy_hex(value: str) -> str:
    entropy = str(value)
    if len(entropy) != 64 or any(
        character not in "0123456789abcdef" for character in entropy
    ):
        raise ValueError(
            "sealed permutation entropy must be a lowercase 256-bit hex value"
        )
    return entropy


def _uniform_index(
    *,
    population_hash: str,
    entropy_hex: str,
    step: int,
    span: int,
) -> int:
    """Draw one exact discrete-uniform index from a versioned SHA-256 stream."""

    if span <= 0:
        raise ValueError("sealed permutation draw span must be positive")
    range_size = 1 << 256
    acceptance_limit = range_size - (range_size % span)
    counter = 0
    while True:
        block = hashlib.sha256(
            b"\x00".join(
                (
                    SEALED_UNIFORM_PERMUTATION_ALGORITHM_V1.encode("ascii"),
                    population_hash.encode("ascii"),
                    bytes.fromhex(entropy_hex),
                    str(int(step)).encode("ascii"),
                    str(counter).encode("ascii"),
                )
            )
        ).digest()
        value = int.from_bytes(block, byteorder="big", signed=False)
        if value < acceptance_limit:
            return value % span
        counter += 1


def _generate_sealed_uniform_permutation(
    population: tuple[str, ...],
    *,
    entropy_hex: str,
) -> tuple[str, ...]:
    result = list(population)
    population_hash = content_hash(
        {
            "algorithm_id": SEALED_UNIFORM_PERMUTATION_ALGORITHM_V1,
            "canonical_population": population,
        },
        prefix="irg-sealed-permutation-population-",
    )
    for step in range(len(result) - 1, 0, -1):
        swap_index = _uniform_index(
            population_hash=population_hash,
            entropy_hex=entropy_hex,
            step=step,
            span=step + 1,
        )
        result[step], result[swap_index] = result[swap_index], result[step]
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SealedUniformPermutationReceipt:
    """Reproducible finite-population randomization, including its reveal.

    The canonical population is deliberately distinct from ``generated_order``.
    The latter is never trusted as a caller assertion: every adaptive consumer
    regenerates it from the frozen entropy and versioned rejection-sampled
    Fisher--Yates algorithm before using a non-census prefix.
    """

    canonical_population: tuple[str, ...]
    entropy_hex: str
    generated_order: tuple[str, ...]
    algorithm_id: str = SEALED_UNIFORM_PERMUTATION_ALGORITHM_V1
    schema: str = "irg_sealed_uniform_permutation_receipt_v1"

    def __post_init__(self) -> None:
        population = _canonical_population(self.canonical_population)
        entropy = _entropy_hex(self.entropy_hex)
        order = tuple(str(value) for value in self.generated_order)
        if (
            len(order) != len(population)
            or len(set(order)) != len(order)
            or set(order) != set(population)
        ):
            raise ValueError(
                "sealed permutation generated order must be an exact population permutation"
            )
        if self.algorithm_id != SEALED_UNIFORM_PERMUTATION_ALGORITHM_V1:
            raise ValueError("unsupported sealed permutation algorithm identity")
        if self.schema != "irg_sealed_uniform_permutation_receipt_v1":
            raise ValueError("unsupported sealed permutation receipt schema")
        object.__setattr__(self, "canonical_population", population)
        object.__setattr__(self, "entropy_hex", entropy)
        object.__setattr__(self, "generated_order", order)

    @classmethod
    def create(
        cls,
        population: Sequence[str],
        *,
        entropy_hex: str,
    ) -> "SealedUniformPermutationReceipt":
        canonical_population = tuple(sorted(str(value) for value in population))
        canonical_population = _canonical_population(canonical_population)
        entropy = _entropy_hex(entropy_hex)
        receipt = cls(
            canonical_population=canonical_population,
            entropy_hex=entropy,
            generated_order=_generate_sealed_uniform_permutation(
                canonical_population,
                entropy_hex=entropy,
            ),
        )
        # This order was produced, rather than asserted, by the exact
        # versioned generator above.  Do not regenerate the million-element
        # reveal when the enclosing stratum immediately verifies it.
        _mark_receipt_verified(receipt)
        return receipt

    def regenerate(self) -> tuple[str, ...]:
        return _generate_sealed_uniform_permutation(
            self.canonical_population,
            entropy_hex=self.entropy_hex,
        )

    def verify(self) -> None:
        if _receipt_is_verified(self):
            return
        if self.generated_order != self.regenerate():
            raise ValueError(
                "sealed permutation generated order does not match its frozen entropy"
            )
        _mark_receipt_verified(self)

    @property
    def content_hash(self) -> str:
        with _SEALED_IDENTITY_CACHE_LOCK:
            cached = _RECEIPT_CONTENT_HASHES.get(id(self))
            if cached is not None and cached[0] is self:
                _RECEIPT_CONTENT_HASHES.move_to_end(id(self))
                return cached[1]
        # Preserve the exact historical hash representation.  Only memoize
        # it; changing to a new compact digest would reinterpret old receipts.
        identity = content_hash(
            self,
            prefix="irg-sealed-uniform-permutation-",
        )
        with _SEALED_IDENTITY_CACHE_LOCK:
            _bounded_identity_put(
                _RECEIPT_CONTENT_HASHES,
                self,
                (self, identity),
            )
        return identity


@dataclass(frozen=True, slots=True)
class ResponseContextStratum:
    """A canonical finite stratum plus its verifiable randomized order."""

    stratum_id: str
    context_ids: tuple[str, ...]
    weight: float
    packet_target: int
    maximum_target: int
    expected_cost: CostEstimate = field(
        default_factory=lambda: CostEstimate(wall_seconds=1.0)
    )
    permutation_receipt: SealedUniformPermutationReceipt | None = None

    def __post_init__(self) -> None:
        contexts = _canonical_population(self.context_ids)
        if not self.stratum_id:
            raise ValueError("response context stratum requires a stable ID")
        if not math.isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise ValueError(
                "response context stratum weight must be finite and positive"
            )
        if int(self.packet_target) <= 0 or not 0 < int(self.maximum_target) <= len(
            contexts
        ):
            raise ValueError(
                "response context stratum packet/maximum targets are invalid"
            )
        receipt = self.permutation_receipt
        if not isinstance(receipt, SealedUniformPermutationReceipt):
            raise ValueError(
                "response context stratum requires a typed sealed permutation receipt"
            )
        if receipt.canonical_population != contexts:
            raise ValueError(
                "response context stratum and sealed permutation populations differ"
            )
        receipt.verify()
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "packet_target", int(self.packet_target))
        object.__setattr__(self, "maximum_target", int(self.maximum_target))

    @property
    def sampling_order(self) -> tuple[str, ...]:
        receipt = self.permutation_receipt
        if not isinstance(receipt, SealedUniformPermutationReceipt):
            raise RuntimeError("response stratum lost its sealed permutation receipt")
        return receipt.generated_order


@dataclass(frozen=True, slots=True)
class FrozenResponseContextDesign:
    """Sealed without-replacement stratified design and randomization receipt."""

    design_id: str
    strata: tuple[ResponseContextStratum, ...]
    confidence_level: float = 0.95
    stable_looks_required: int = 2
    unresolved_action: str = "full_census"
    schema: str = "irg_frozen_response_context_design_v2"

    def __post_init__(self) -> None:
        strata = tuple(self.strata)
        if not self.design_id or not self.schema:
            raise ValueError("response context design requires stable identity")
        if not strata or len({item.stratum_id for item in strata}) != len(strata):
            raise ValueError("response context design requires unique strata")
        context_ids = tuple(context for item in strata for context in item.context_ids)
        if len(set(context_ids)) != len(context_ids):
            raise ValueError("response context IDs must be unique across strata")
        if int(self.stable_looks_required) <= 0:
            raise ValueError("response context design stable looks must be positive")
        confidence = float(self.confidence_level)
        if not 0.5 < confidence < 1.0:
            raise ValueError(
                "response context design confidence_level must lie in (0.5,1)"
            )
        if self.unresolved_action not in {"full_census", "block"}:
            raise ValueError(
                "response context design unresolved action must be full_census or block"
            )
        object.__setattr__(self, "strata", strata)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(
            self, "stable_looks_required", int(self.stable_looks_required)
        )

    @property
    def content_hash(self) -> str:
        with _SEALED_IDENTITY_CACHE_LOCK:
            cached = _DESIGN_CONTENT_HASHES.get(id(self))
            if cached is not None and cached[0] is self:
                _DESIGN_CONTENT_HASHES.move_to_end(id(self))
                return cached[1]
        # As above, retain byte-compatible scientific identity while avoiding
        # repeated serialization of both full stratum populations.
        identity = content_hash(
            self,
            prefix="irg-response-context-design-",
        )
        with _SEALED_IDENTITY_CACHE_LOCK:
            _bounded_identity_put(
                _DESIGN_CONTENT_HASHES,
                self,
                (self, identity),
            )
        return identity

    @property
    def randomization_commitment(self) -> str:
        """Derived audit identity; never a caller-supplied randomization claim."""

        return content_hash(
            tuple(
                (
                    item.stratum_id,
                    item.permutation_receipt.content_hash,
                )
                for item in self.strata
                if item.permutation_receipt is not None
            ),
            prefix="irg-response-randomization-",
        )

    def stratum(self, stratum_id: str) -> ResponseContextStratum:
        for item in self.strata:
            if item.stratum_id == str(stratum_id):
                return item
        raise KeyError(f"unknown response context stratum {stratum_id!r}")


@dataclass
class ResponseAcquisitionState:
    """Mutable session receipt for cumulative prefixes and stable looks."""

    design_hash: str
    cumulative_prefixes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    outcome_hashes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    stratum_moments: dict[str, dict[str, ResponseMomentTensor]] = field(
        default_factory=dict
    )
    stratum_local_attribution_moments: dict[str, LocalAttributionMomentTensor] = field(
        default_factory=dict
    )
    stratum_tripartite_pullback_moments: dict[str, TripartitePullbackMomentTensor] = (
        field(default_factory=dict)
    )
    stratum_global_trust_geometries: dict[str, GlobalTrustGeometryState] = field(
        default_factory=dict
    )
    stratum_intrinsic_atlas_geometries: dict[str, GlobalTrustGeometryState] = field(
        default_factory=dict
    )
    prior_action_signature: str | None = None
    prior_decision_scope_hash: str | None = None
    stable_looks: int = 0
    dependency_hash: str | None = None
    owned_response_chart_hash: str | None = None
    awaiting_action_signature: bool = True
    exhausted_strata: set[str] = field(default_factory=set)
    unresolved_exhausted: bool = False
    reset_count: int = 0
    last_reset_reason: str | None = None
    chart_republication_count: int = 0
    schema: str = "irg_response_acquisition_state_v1"

    def __post_init__(self) -> None:
        if not self.design_hash or not self.schema or int(self.stable_looks) < 0:
            raise ValueError(
                "response acquisition state identity or stable looks are invalid"
            )
        self.cumulative_prefixes = {
            str(k): tuple(map(str, v)) for k, v in self.cumulative_prefixes.items()
        }
        self.outcome_hashes = {
            str(k): tuple(map(str, v)) for k, v in self.outcome_hashes.items()
        }
        self.stratum_moments = {
            str(stratum_id): {
                str(label): tensor for label, tensor in dict(tensors).items()
            }
            for stratum_id, tensors in self.stratum_moments.items()
        }
        if any(
            not isinstance(tensor, ResponseMomentTensor)
            for tensors in self.stratum_moments.values()
            for tensor in tensors.values()
        ):
            raise TypeError(
                "response acquisition stratum moments require typed tensors"
            )
        self.stratum_local_attribution_moments = {
            str(stratum_id): tensor
            for stratum_id, tensor in self.stratum_local_attribution_moments.items()
        }
        if any(
            not isinstance(tensor, LocalAttributionMomentTensor)
            for tensor in self.stratum_local_attribution_moments.values()
        ):
            raise TypeError("response acquisition local moments require typed tensors")
        self.stratum_tripartite_pullback_moments = {
            str(stratum_id): tensor
            for stratum_id, tensor in self.stratum_tripartite_pullback_moments.items()
        }
        if any(
            not isinstance(tensor, TripartitePullbackMomentTensor)
            for tensor in self.stratum_tripartite_pullback_moments.values()
        ):
            raise TypeError(
                "response acquisition tripartite moments require typed tensors"
            )
        self.stratum_global_trust_geometries = {
            str(stratum_id): geometry
            for stratum_id, geometry in self.stratum_global_trust_geometries.items()
        }
        if any(
            not isinstance(geometry, GlobalTrustGeometryState)
            for geometry in self.stratum_global_trust_geometries.values()
        ):
            raise TypeError(
                "response acquisition trust geometries require typed states"
            )
        self.stratum_intrinsic_atlas_geometries = {
            str(stratum_id): geometry
            for stratum_id, geometry in self.stratum_intrinsic_atlas_geometries.items()
        }
        if any(
            not isinstance(geometry, GlobalTrustGeometryState)
            for geometry in self.stratum_intrinsic_atlas_geometries.values()
        ):
            raise TypeError(
                "response acquisition intrinsic-atlas geometries require typed states"
            )
        self.exhausted_strata = set(map(str, self.exhausted_strata))
        self.stable_looks = int(self.stable_looks)

    @property
    def content_hash(self) -> str:
        return content_hash(self, prefix="irg-response-acquisition-state-")

    def has_initial_coverage(self, design: FrozenResponseContextDesign) -> bool:
        """Whether every frozen stratum has contributed its initial packet."""

        return all(
            len(self.cumulative_prefixes.get(stratum.stratum_id, ()))
            >= min(stratum.packet_target, stratum.maximum_target)
            for stratum in design.strata
        )

    def is_stable(self, design: FrozenResponseContextDesign) -> bool:
        """Whether the optional consecutive action-signature target was met."""

        return (
            self.has_initial_coverage(design)
            and self.stable_looks >= design.stable_looks_required
        )

    def has_full_census(self, design: FrozenResponseContextDesign) -> bool:
        """Whether every finite stratum population was observed exactly."""

        return all(
            len(self.cumulative_prefixes.get(stratum.stratum_id, ()))
            == len(stratum.context_ids)
            for stratum in design.strata
        )

    def is_resolved(self, design: FrozenResponseContextDesign) -> bool:
        """Whether response acquisition may stop by stability or exact census.

        This is only a response-geometry acquisition condition.  It is never
        an exact hard-mask safety authorization.
        """

        return self.has_full_census(design) or self.is_stable(design)

    def resolution(self, design: FrozenResponseContextDesign) -> str:
        if self.has_full_census(design):
            return "full_census_exact"
        if self.is_stable(design):
            return "stable_action_signature"
        return (
            "unresolved_noncensus_exhausted_"
            + (
                "full_census_required"
                if design.unresolved_action == "full_census"
                else "blocking"
            )
            if self.unresolved_exhausted
            else "unresolved"
        )

    def reset_for_state_transition(
        self,
        reason: str = "physical_state_transition",
    ) -> None:
        """Discard evidence that belongs to another mask/calibration chart."""

        self.reset_count += 1
        self.last_reset_reason = str(reason)
        self.cumulative_prefixes.clear()
        self.outcome_hashes.clear()
        self.stratum_moments.clear()
        self.stratum_local_attribution_moments.clear()
        self.stratum_tripartite_pullback_moments.clear()
        self.stratum_global_trust_geometries.clear()
        self.stratum_intrinsic_atlas_geometries.clear()
        self.prior_action_signature = None
        self.prior_decision_scope_hash = None
        self.stable_looks = 0
        self.exhausted_strata.clear()
        self.unresolved_exhausted = False
        self.awaiting_action_signature = True


def _hash_identity(value: object) -> str | None:
    identity = getattr(value, "content_hash", None)
    if callable(identity):
        identity = identity()
    return identity if isinstance(identity, str) and identity else None


def _response_chart_hash(session: object) -> str:
    state = getattr(session, "version_state", None)
    state = getattr(state, "intrinsic_state", state)
    # A production response chart is a deterministic projection of its sealed
    # measurement outcomes.  Hashing each nested tensor here re-serialized the
    # complete ordered route-ID axis after every panel.  The outcome hashes,
    # mask/calibration identities, and sample count are the exact compact chart
    # lineage; the enclosing session identity independently binds the route
    # axis.  Fixtures without physical lineage continue through the exhaustive
    # object-hash path below.
    source_outcome_hashes = tuple(map(str, getattr(state, "source_outcome_hashes", ())))
    if source_outcome_hashes:
        compatibility_overrides = {}
        for name in (
            "response_moments",
            "intrinsic_response_moments",
            "response_factor",
            "metric_factor",
            "interaction_metric",
            "jacobian",
        ):
            value = getattr(state, name, None)
            # Production moment tensors are deterministic projections of the
            # sealed outcome lineage. Compatibility fixtures and external
            # adapters may install a lightweight identity object directly;
            # retain that explicit override without forcing a route-scale
            # tensor hash.
            if value is not None and not isinstance(value, ResponseMomentTensor):
                compatibility_overrides[name] = _hash_identity(value)
        return content_hash(
            {
                "schema": "irg_response_chart_outcome_lineage_v1",
                "source_outcome_hashes": source_outcome_hashes,
                "sample_count": int(getattr(state, "sample_count", 0)),
                "measured_mask_hash": str(getattr(state, "measured_mask_hash", "")),
                "calibration_state_id": str(getattr(state, "calibration_state_id", "")),
                "compatibility_overrides": compatibility_overrides,
            },
            prefix="irg-response-chart-",
        )
    hashes = {}
    for name in (
        "response_moments",
        "intrinsic_response_moments",
        "response_factor",
        "metric_factor",
        "interaction_metric",
        "jacobian",
    ):
        identity = _hash_identity(getattr(state, name, None))
        if identity is not None:
            hashes[name] = identity
    return content_hash(hashes, prefix="irg-response-chart-")


def _unpack_response_moment_payload(
    *,
    outcome: MeasurementOutcome,
    expected_route_ids: tuple[str, ...],
) -> tuple[tuple[str, ResponseMomentTensor], ...]:
    """Decode and validate a capability-declared whole-route moment payload.

    A rectangular numeric array is not, by itself, a response measurement.
    The provider must supply the versioned packed-moment layout whose semantic
    axes and sufficient statistics are validated by ``ResponseMomentTensor``.
    A joint layout retains both fixed and nuisance-quotiented charts.
    """

    if str(outcome.metadata.get("kind", "")) != "response_moments":
        raise ValueError(
            "response acquisition outcome metadata must declare response_moments"
        )
    values = np.asarray(outcome.values, dtype=np.float64)
    layout = outcome.metadata.get("moment_layout")
    joint_layout = outcome.metadata.get("joint_moment_layout")
    if isinstance(layout, Mapping) and joint_layout is None:
        unpacked = (
            (
                "response_moments",
                ResponseMomentTensor.unpack(
                    values,
                    layout,
                    route_ids=expected_route_ids,
                ),
            ),
        )
    elif isinstance(joint_layout, Mapping) and layout is None:
        layout = joint_layout
        if (
            not isinstance(layout, Mapping)
            or layout.get("schema") != "fixed_and_intrinsic_response_moments_v1"
        ):
            raise ValueError(
                "response acquisition requires the fixed/intrinsic "
                "joint moment layout"
            )
        try:
            fixed_row_count = int(layout["fixed_row_count"])
            fixed_layout = layout["fixed"]
            intrinsic_layout = layout["intrinsic"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "response acquisition has an incomplete joint moment layout"
            ) from error
        if (
            fixed_row_count <= 0
            or fixed_row_count >= values.shape[0]
            or not isinstance(fixed_layout, Mapping)
            or not isinstance(intrinsic_layout, Mapping)
        ):
            raise ValueError(
                f"{version} response acquisition has invalid joint moment sections"
            )
        unpacked = (
            (
                "response_moments",
                ResponseMomentTensor.unpack(
                    values[:fixed_row_count],
                    fixed_layout,
                    route_ids=expected_route_ids,
                ),
            ),
            (
                "intrinsic_response_moments",
                ResponseMomentTensor.unpack(
                    values[fixed_row_count:],
                    intrinsic_layout,
                    route_ids=expected_route_ids,
                ),
            ),
        )
        fixed = unpacked[0][1]
        intrinsic = unpacked[1][1]
        if (
            fixed.channel_ids != intrinsic.channel_ids
            or fixed.context_axis_hash != intrinsic.context_axis_hash
            or fixed.sample_count != intrinsic.sample_count
        ):
            raise ValueError(
                f"{version} fixed and intrinsic response moments must share "
                "response, context, and sample axes"
            )
    else:
        raise ValueError(
            "response acquisition requires exactly one supported moment-layout capability"
        )

    context_count = len(outcome.context_ids)
    for label, tensor in unpacked:
        if tensor.route_ids != expected_route_ids:
            raise ValueError(
                f"{label} uses a different physical route axis from the retained state"
            )
        if tensor.sample_count != context_count:
            raise ValueError(
                f"{label} sample count must equal the cumulative context prefix"
            )
    return unpacked


def _unpack_local_attribution_payload(
    outcome: MeasurementOutcome,
    *,
    expected_route_ids: tuple[str, ...],
) -> LocalAttributionMomentTensor | None:
    layout = outcome.metadata.get("local_attribution_layout")
    if layout is None:
        return None
    if not isinstance(layout, Mapping):
        raise ValueError("local attribution layout must be a mapping")
    try:
        offset = int(layout["row_offset"])
        count = int(layout["row_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("local attribution layout lacks its row span") from error
    values = np.asarray(outcome.values, dtype=np.float64)
    if offset < 0 or count <= 0 or offset + count > values.shape[0]:
        raise ValueError("local attribution row span is outside the outcome")
    tensor = LocalAttributionMomentTensor.unpack(
        values[offset : offset + count],
        layout,
        route_ids=expected_route_ids,
    )
    if tensor.route_ids != expected_route_ids or tensor.sample_count != len(
        outcome.context_ids
    ):
        raise ValueError(
            "local attribution moments differ from the physical route/context axis"
        )
    return tensor


def _combine_stratified_local_attribution_moments(
    *,
    tensors_by_stratum: Mapping[str, LocalAttributionMomentTensor],
    design: FrozenResponseContextDesign,
    prefixes: Mapping[str, tuple[str, ...]],
) -> LocalAttributionMomentTensor:
    ordered = tuple(
        (stratum, tensors_by_stratum[stratum.stratum_id])
        for stratum in design.strata
        if stratum.stratum_id in tensors_by_stratum
    )
    if not ordered:
        raise ValueError("stratified local attribution requires typed moments")
    reference = ordered[0][1]
    for _stratum, tensor in ordered[1:]:
        if (
            tensor.route_ids != reference.route_ids
            or tensor.destination_transport_hash != reference.destination_transport_hash
        ):
            raise ValueError(
                "local attribution strata do not share a route/transport axis"
            )
    raw_weights = np.asarray(
        [stratum.weight for stratum, _ in ordered], dtype=np.float64
    )
    weights = raw_weights / float(raw_weights.sum())
    mean = sum(
        float(weight) * tensor.mean
        for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
    )
    second = sum(
        float(weight) * tensor.second_moment
        for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
    )
    mean_se = np.sqrt(
        sum(
            float(weight) ** 2 * tensor.mean_standard_error**2
            for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
        )
    )
    second_se = np.sqrt(
        sum(
            float(weight) ** 2 * tensor.second_moment_standard_error**2
            for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
        )
    )
    # Law of total covariance.  Each within-stratum sketched factor is
    # weighted by sqrt(w_h); one exact rank-one row per stratum represents
    # the between-stratum mean displacement.
    factor_rows: list[np.ndarray] = []
    for weight, (_stratum, tensor) in zip(weights, ordered, strict=True):
        root = math.sqrt(float(weight))
        factor_rows.append(root * tensor.covariance_factor)
        factor_rows.append(root * (tensor.mean - mean)[None, :])
    return LocalAttributionMomentTensor(
        route_ids=reference.route_ids,
        mean=mean,
        second_moment=second,
        mean_standard_error=mean_se,
        second_moment_standard_error=second_se,
        covariance_factor=np.concatenate(factor_rows, axis=0),
        sample_count=sum(tensor.sample_count for _stratum, tensor in ordered),
        context_axis_hash=content_hash(
            {
                "design_hash": design.content_hash,
                "strata": tuple(
                    (
                        stratum.stratum_id,
                        prefixes[stratum.stratum_id],
                        tensor.content_hash,
                        float(weight),
                    )
                    for weight, (stratum, tensor) in zip(weights, ordered, strict=True)
                ),
            },
            prefix="irg-stratified-local-attribution-context-axis-",
        ),
        destination_transport_hash=reference.destination_transport_hash,
        sketch_seed=reference.sketch_seed,
    )


def _unpack_tripartite_pullback_payload(
    outcome: MeasurementOutcome,
    *,
    expected_route_ids: tuple[str, ...],
) -> TripartitePullbackMomentTensor | None:
    layout = outcome.metadata.get("tripartite_pullback_layout")
    if layout is None:
        return None
    if not isinstance(layout, Mapping):
        raise ValueError("tripartite pullback layout must be a mapping")
    try:
        offset = int(layout["row_offset"])
        count = int(layout["row_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("tripartite pullback layout lacks its row span") from error
    values = np.asarray(outcome.values, dtype=np.float64)
    tensor = TripartitePullbackMomentTensor.unpack(
        values[offset : offset + count],
        layout,
        route_ids=expected_route_ids,
    )
    if tensor.route_ids != expected_route_ids or tensor.sample_count != len(
        outcome.context_ids
    ):
        raise ValueError(
            "tripartite pullback differs from the physical route/context axis"
        )
    return tensor


def _unpack_global_trust_payload(
    outcome: MeasurementOutcome,
    *,
    expected_route_ids: tuple[str, ...],
) -> GlobalTrustGeometryState | None:
    layout = outcome.metadata.get("global_trust_geometry_layout")
    if layout is None:
        return None
    if not isinstance(layout, Mapping):
        raise ValueError("global trust geometry layout must be a mapping")
    try:
        offset = int(layout["row_offset"])
        count = int(layout["row_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("global trust geometry layout lacks its row span") from error
    values = np.asarray(outcome.values, dtype=np.float64)
    if offset < 0 or count <= 0 or offset + count > values.shape[0]:
        raise ValueError("global trust geometry row span is outside the outcome")
    geometry = GlobalTrustGeometryState.unpack(
        values[offset : offset + count],
        layout,
        route_ids=expected_route_ids,
    )
    if geometry.route_ids != expected_route_ids or geometry.context_count != len(
        outcome.context_ids
    ):
        raise ValueError(
            "global trust geometry differs from the physical route/context axis"
        )
    return geometry


def _unpack_intrinsic_atlas_geometry_payload(
    outcome: MeasurementOutcome,
    *,
    expected_route_ids: tuple[str, ...],
) -> GlobalTrustGeometryState | None:
    layout = outcome.metadata.get("intrinsic_atlas_geometry_layout")
    if layout is None:
        return None
    if not isinstance(layout, Mapping):
        raise ValueError("intrinsic atlas geometry layout must be a mapping")
    try:
        offset = int(layout["row_offset"])
        count = int(layout["row_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("intrinsic atlas geometry lacks its row span") from error
    values = np.asarray(outcome.values, dtype=np.float64)
    if offset < 0 or count <= 0 or offset + count > values.shape[0]:
        raise ValueError("intrinsic atlas geometry row span is outside the outcome")
    geometry = GlobalTrustGeometryState.unpack(
        values[offset : offset + count],
        layout,
        route_ids=expected_route_ids,
    )
    if (
        geometry.route_ids != expected_route_ids
        or geometry.context_count != len(outcome.context_ids)
        or geometry.metadata.get("geometry_role")
        != "nuisance_quotiented_atlas_embedding"
    ):
        raise ValueError(
            "intrinsic atlas geometry differs from its route/context/quotient identity"
        )
    return geometry


def _combine_stratified_global_trust_geometries(
    *,
    geometries_by_stratum: Mapping[str, GlobalTrustGeometryState],
    design: FrozenResponseContextDesign,
    prefixes: Mapping[str, tuple[str, ...]],
) -> GlobalTrustGeometryState:
    """Combine normalized stratum geometries without discarding cross-route terms.

    The centered factor is the exact low-rank law-of-total-covariance
    projection of the retained factors and between-stratum means.  Omitted
    spectral envelopes add with the frozen risk weights.
    """

    ordered = tuple(
        (stratum, geometries_by_stratum[stratum.stratum_id])
        for stratum in design.strata
        if stratum.stratum_id in geometries_by_stratum
    )
    if not ordered:
        raise ValueError("stratified trust projection requires typed geometries")
    reference = ordered[0][1]
    for _stratum, geometry in ordered[1:]:
        if (
            geometry.route_ids != reference.route_ids
            or geometry.spec != reference.spec
            or geometry.measured_mask_hash != reference.measured_mask_hash
            or geometry.calibration_state_id != reference.calibration_state_id
            or geometry.mean_component.shape != reference.mean_component.shape
        ):
            raise ValueError(
                "trust strata do not share one route, state, calibration, and spec"
            )
    raw_weights = np.asarray(
        [stratum.weight for stratum, _geometry in ordered], dtype=np.float64
    )
    weights = raw_weights / float(raw_weights.sum())
    mean = sum(
        float(weight) * geometry.mean_component
        for weight, (_stratum, geometry) in zip(weights, ordered, strict=True)
    )
    factor_rows: list[np.ndarray] = []
    for weight, (_stratum, geometry) in zip(weights, ordered, strict=True):
        root = math.sqrt(float(weight))
        factor_rows.append(root * geometry.centered_factor)
        factor_rows.append(root * (geometry.mean_component - mean))
    factor = np.concatenate(factor_rows, axis=0)
    factor_before_recompression = factor
    # Concatenation can exceed the declared factor cap even though each
    # stratum was individually compressed. Recompress the finite factor
    # deterministically and carry the additional shrinkage into rho.
    maximum_rank = min(reference.spec.maximum_rank, len(reference.route_ids))
    extra_residual = 0.0
    if factor.shape[0] > maximum_rank:
        _left, singular, right = np.linalg.svd(factor, full_matrices=False)
        delta = (
            float(singular[maximum_rank] ** 2) if singular.size > maximum_rank else 0.0
        )
        retained = min(maximum_rank, singular.size)
        shrunk = np.sqrt(np.maximum(singular[:retained] ** 2 - delta, 0.0))
        factor = shrunk[:, None] * right[:retained]
        factor = factor[shrunk > 0.0]
        extra_residual = delta
    residual = extra_residual + sum(
        float(weight) * geometry.residual_spectral_bound
        for weight, (_stratum, geometry) in zip(weights, ordered, strict=True)
    )
    residual_diagonal = None
    if all(
        geometry.residual_diagonal_bound is not None for _stratum, geometry in ordered
    ):
        source_residual_diagonal = sum(
            float(weight) * geometry.residual_diagonal_bound
            for weight, (_stratum, geometry) in zip(
                weights,
                ordered,
                strict=True,
            )
        )
        recompression_diagonal = np.maximum(
            np.sum(factor_before_recompression**2, axis=0) - np.sum(factor**2, axis=0),
            0.0,
        )
        residual_diagonal = np.maximum(
            source_residual_diagonal + recompression_diagonal,
            0.0,
        )
    fold_counts = tuple(
        int(geometry.metadata.get("split_fold_validation_count", 0))
        for _stratum, geometry in ordered
    )
    split_fold_embeddings: list[np.ndarray] = []
    split_fold_residuals: list[float] = []
    if fold_counts and fold_counts[0] >= 2:
        if any(count != fold_counts[0] for count in fold_counts):
            raise ValueError(
                "stratified trust geometries have inconsistent split-fold counts"
            )
        mean_rows = int(reference.mean_component.shape[0])
        for fold_index in range(fold_counts[0]):
            fold_means: list[np.ndarray] = []
            fold_factors: list[np.ndarray] = []
            for _stratum, geometry in ordered:
                raw_embeddings = geometry.metadata.get("split_fold_embeddings", ())
                if len(raw_embeddings) != fold_counts[0]:
                    raise ValueError(
                        "stratified trust geometry lost a split-fold embedding"
                    )
                fold_embedding = np.asarray(
                    raw_embeddings[fold_index], dtype=np.float64
                )
                if (
                    fold_embedding.ndim != 2
                    or fold_embedding.shape[1] != len(reference.route_ids)
                    or fold_embedding.shape[0] < mean_rows
                ):
                    raise ValueError(
                        "split-fold embedding differs from the trust route axis"
                    )
                fold_means.append(fold_embedding[:mean_rows])
                fold_factors.append(fold_embedding[mean_rows:])
            fold_mean = sum(
                float(weight) * value
                for weight, value in zip(weights, fold_means, strict=True)
            )
            fold_rows: list[np.ndarray] = []
            for weight, value, fold_mean_h in zip(
                weights, fold_factors, fold_means, strict=True
            ):
                root = math.sqrt(float(weight))
                fold_rows.append(root * value)
                fold_rows.append(root * (fold_mean_h - fold_mean))
            fold_factor = np.concatenate(fold_rows, axis=0)
            fold_extra_residual = 0.0
            if fold_factor.shape[0] > maximum_rank:
                _left, singular, right = np.linalg.svd(fold_factor, full_matrices=False)
                delta = (
                    float(singular[maximum_rank] ** 2)
                    if singular.size > maximum_rank
                    else 0.0
                )
                retained = min(maximum_rank, singular.size)
                shrunk = np.sqrt(np.maximum(singular[:retained] ** 2 - delta, 0.0))
                fold_factor = shrunk[:, None] * right[:retained]
                fold_factor = fold_factor[shrunk > 0.0]
                fold_extra_residual = delta
            split_fold_embeddings.append(
                np.concatenate((fold_mean, fold_factor), axis=0)
            )
            split_fold_residuals.append(
                fold_extra_residual
                + sum(
                    float(weight)
                    * float(
                        geometry.metadata["split_fold_residual_spectral_bounds"][
                            fold_index
                        ]
                    )
                    for weight, (_stratum, geometry) in zip(
                        weights, ordered, strict=True
                    )
                )
            )
    atlas_embedding_key = (
        "intrinsic_atlas_split_fold_embeddings"
        if "intrinsic_atlas_split_fold_embeddings" in reference.metadata
        else "atlas_split_fold_global_embeddings"
    )
    atlas_count_key = (
        "intrinsic_atlas_split_fold_validation_count"
        if atlas_embedding_key.startswith("intrinsic_")
        else "atlas_split_fold_validation_count"
    )
    atlas_fold_counts = tuple(
        int(geometry.metadata.get(atlas_count_key, 0)) for _stratum, geometry in ordered
    )
    atlas_split_fold_embeddings: list[np.ndarray] = []
    if atlas_fold_counts and atlas_fold_counts[0] >= 2:
        if any(count != atlas_fold_counts[0] for count in atlas_fold_counts):
            raise ValueError(
                "stratified atlas embeddings have inconsistent split-fold counts"
            )
        atlas_mean_rows = int(reference.mean_component.shape[0])
        for fold_index in range(atlas_fold_counts[0]):
            fold_values: list[np.ndarray] = []
            for _stratum, geometry in ordered:
                raw = geometry.metadata.get(atlas_embedding_key, ())
                if len(raw) != atlas_fold_counts[0]:
                    raise ValueError(
                        "stratified geometry lost an atlas split-fold embedding"
                    )
                value = np.asarray(raw[fold_index], dtype=np.float64)
                if (
                    value.ndim != 2
                    or value.shape[1] != len(reference.route_ids)
                    or value.shape[0] < atlas_mean_rows
                ):
                    raise ValueError(
                        "atlas split-fold embedding differs from the route axis"
                    )
                fold_values.append(value)
            atlas_mean = sum(
                float(weight) * value[:atlas_mean_rows]
                for weight, value in zip(weights, fold_values, strict=True)
            )
            atlas_rows = [atlas_mean]
            for weight, value in zip(weights, fold_values, strict=True):
                root = math.sqrt(float(weight))
                atlas_rows.append(root * value[atlas_mean_rows:])
                atlas_rows.append(root * (value[:atlas_mean_rows] - atlas_mean))
            atlas_split_fold_embeddings.append(np.concatenate(atlas_rows, axis=0))
    return GlobalTrustGeometryState(
        spec=reference.spec,
        route_ids=reference.route_ids,
        mean_component=mean,
        centered_factor=factor,
        residual_spectral_bound=residual,
        residual_diagonal_bound=residual_diagonal,
        total_weight=1.0,
        context_count=sum(geometry.context_count for _stratum, geometry in ordered),
        context_axis_hash=content_hash(
            {
                "design_hash": design.content_hash,
                "strata": tuple(
                    (
                        stratum.stratum_id,
                        prefixes[stratum.stratum_id],
                        geometry.content_hash,
                        float(weight),
                    )
                    for weight, (stratum, geometry) in zip(
                        weights, ordered, strict=True
                    )
                ),
            },
            prefix="irg-stratified-global-trust-context-axis-",
        ),
        measured_mask_hash=reference.measured_mask_hash,
        calibration_state_id=reference.calibration_state_id,
        source_measurement_hashes=tuple(
            geometry.content_hash for _stratum, geometry in ordered
        ),
        metadata={
            "aggregation": "frozen_stratified_total_covariance_projection_v1",
            "response_acquisition_design_hash": design.content_hash,
            "normalized_stratum_weights": tuple(float(value) for value in weights),
            "between_stratum_mean_rows_retained": True,
            "cross_module_terms_retained": True,
            "post_projection_frequent_directions_residual": extra_residual,
            "residual_diagonal_projection": (
                "weighted_stratum_psd_remainders_plus_recompression_diagonal_v1"
                if residual_diagonal is not None
                else "unavailable_legacy_spectral_only"
            ),
            # A stratified rank-growth replay is complete only after every
            # stratum has been rebuilt at the requested rank.  Reporting the
            # reference stratum's rank could prematurely discharge the global
            # obligation after only the first replayed stratum.
            "requested_rank": min(
                int(geometry.metadata.get("requested_rank", geometry.spec.initial_rank))
                for _stratum, geometry in ordered
            ),
            "split_fold_embeddings": tuple(split_fold_embeddings),
            "split_fold_residual_spectral_bounds": tuple(split_fold_residuals),
            "split_fold_validation_count": len(split_fold_embeddings),
            "split_fold_assignment": (
                "within_stratum_round_robin_then_frozen_weight_projection_v1"
                if split_fold_embeddings
                else "not_available"
            ),
            "atlas_split_fold_global_embeddings": tuple(
                atlas_split_fold_embeddings
                if not atlas_embedding_key.startswith("intrinsic_")
                else ()
            ),
            "atlas_split_fold_validation_count": len(
                atlas_split_fold_embeddings
                if not atlas_embedding_key.startswith("intrinsic_")
                else ()
            ),
            "atlas_split_fold_shared_projection": (
                "stratified_total_covariance_of_shared_rademacher_fold_factors_v1"
                if atlas_split_fold_embeddings
                else "not_available"
            ),
            "intrinsic_atlas_split_fold_embeddings": tuple(
                atlas_split_fold_embeddings
                if atlas_embedding_key.startswith("intrinsic_")
                else ()
            ),
            "intrinsic_atlas_split_fold_validation_count": len(
                atlas_split_fold_embeddings
                if atlas_embedding_key.startswith("intrinsic_")
                else ()
            ),
            "intrinsic_atlas_split_fold_shared_projection": (
                "stratified_total_covariance_of_shared_rademacher_fold_factors_v1"
                if (
                    atlas_split_fold_embeddings
                    and atlas_embedding_key.startswith("intrinsic_")
                )
                else "not_available"
            ),
            **(
                {"geometry_role": reference.metadata["geometry_role"]}
                if "geometry_role" in reference.metadata
                else {}
            ),
            **(
                {"not_physical_trust_authority": True}
                if reference.metadata.get("not_physical_trust_authority")
                else {}
            ),
        },
    )


def _combine_stratified_tripartite_pullback_moments(
    *,
    tensors_by_stratum: Mapping[str, TripartitePullbackMomentTensor],
    design: FrozenResponseContextDesign,
    prefixes: Mapping[str, tuple[str, ...]],
) -> TripartitePullbackMomentTensor:
    ordered = tuple(
        (stratum, tensors_by_stratum[stratum.stratum_id])
        for stratum in design.strata
        if stratum.stratum_id in tensors_by_stratum
    )
    if not ordered:
        raise ValueError("stratified tripartite pullback requires typed moments")
    reference = ordered[0][1]
    for _stratum, tensor in ordered[1:]:
        if (
            tensor.head_ids != reference.head_ids
            or tensor.route_ids != reference.route_ids
            or tensor.destination_transport_hash != reference.destination_transport_hash
            or tensor.loss_pullback_hash != reference.loss_pullback_hash
        ):
            raise ValueError(
                "tripartite strata do not share one nested pullback identity"
            )
    raw_weights = np.asarray(
        [stratum.weight for stratum, _ in ordered], dtype=np.float64
    )
    weights = raw_weights / float(raw_weights.sum())
    mean = sum(
        float(weight) * tensor.mean
        for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
    )
    second = sum(
        float(weight) * tensor.route_head_second_moment
        for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
    )
    estimator = sum(
        float(weight) ** 2 * tensor.mean_estimator_covariance
        for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
    )
    second_se = np.sqrt(
        sum(
            float(weight) ** 2 * tensor.second_moment_standard_error**2
            for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
        )
    )
    activity_mean = (
        None
        if any(not tensor.activity_available for _stratum, tensor in ordered)
        else sum(
            float(weight) * tensor.view_activity_mean
            for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
        )
    )
    activity_second = (
        None
        if activity_mean is None
        else sum(
            float(weight) * tensor.view_activity_second_moment
            for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
        )
    )
    activity_se = (
        None
        if activity_mean is None
        else np.sqrt(
            sum(
                float(weight) ** 2 * tensor.view_activity_mean_standard_error**2
                for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
            )
        )
    )
    factor_rows: list[np.ndarray] = []
    for weight, (_stratum, tensor) in zip(weights, ordered, strict=True):
        root = math.sqrt(float(weight))
        factor_rows.append(root * tensor.covariance_factor)
        factor_rows.append(root * (tensor.mean - mean)[None, :, :])
    return TripartitePullbackMomentTensor(
        head_ids=reference.head_ids,
        route_ids=reference.route_ids,
        mean=mean,
        route_head_second_moment=second,
        mean_estimator_covariance=estimator,
        second_moment_standard_error=second_se,
        covariance_factor=np.concatenate(factor_rows, axis=0),
        sample_count=sum(tensor.sample_count for _stratum, tensor in ordered),
        context_axis_hash=content_hash(
            {
                "design_hash": design.content_hash,
                "strata": tuple(
                    (
                        stratum.stratum_id,
                        prefixes[stratum.stratum_id],
                        tensor.content_hash,
                        float(weight),
                    )
                    for weight, (stratum, tensor) in zip(weights, ordered, strict=True)
                ),
            },
            prefix="irg-stratified-tripartite-pullback-context-axis-",
        ),
        destination_transport_hash=reference.destination_transport_hash,
        loss_pullback_hash=reference.loss_pullback_hash,
        sketch_seed=reference.sketch_seed,
        view_activity_mean=activity_mean,
        view_activity_mean_standard_error=activity_se,
        view_activity_second_moment=activity_second,
    )


def _combine_stratified_response_moments(
    *,
    tensors_by_stratum: Mapping[str, ResponseMomentTensor],
    design: FrozenResponseContextDesign,
    prefixes: Mapping[str, tuple[str, ...]],
    label: str,
) -> ResponseMomentTensor:
    """Project typed stratum sufficient statistics into one frozen estimand.

    Stratum risk weights, rather than realized prefix sizes, define the
    scientific mixture. Estimator covariances and second-moment standard
    errors combine independently with squared normalized weights.
    """

    ordered = tuple(
        (stratum, tensors_by_stratum[stratum.stratum_id])
        for stratum in design.strata
        if stratum.stratum_id in tensors_by_stratum
    )
    if not ordered:
        raise ValueError("stratified response aggregation requires typed moments")
    reference = ordered[0][1]
    for _stratum, tensor in ordered[1:]:
        if (
            tensor.channel_ids != reference.channel_ids
            or tensor.route_ids != reference.route_ids
            or tensor.response_spec_hash != reference.response_spec_hash
            or tensor.metric_id != reference.metric_id
        ):
            raise ValueError(
                "response strata do not share one scientific response/route axis"
            )
    raw_weights = np.asarray(
        [stratum.weight for stratum, _tensor in ordered],
        dtype=np.float64,
    )
    weights = raw_weights / float(np.sum(raw_weights))
    mean = sum(
        float(weight) * tensor.mean
        for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
    )
    second = sum(
        float(weight) * tensor.second_moment_diagonal
        for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
    )
    estimator_covariance = sum(
        float(weight) ** 2 * tensor.mean_estimator_covariance_diagonal
        for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
    )
    second_standard_error = np.sqrt(
        sum(
            float(weight) ** 2 * tensor.second_moment_standard_error**2
            for weight, (_stratum, tensor) in zip(weights, ordered, strict=True)
        )
    )
    mean_standard_error = np.sqrt(
        np.maximum(
            np.diagonal(estimator_covariance, axis1=1, axis2=2).T,
            0.0,
        )
    )
    included_strata = tuple(stratum.stratum_id for stratum, _tensor in ordered)
    context_axis_hash = content_hash(
        {
            "design_hash": design.content_hash,
            "label": label,
            "strata": tuple(
                (
                    stratum.stratum_id,
                    prefixes[stratum.stratum_id],
                    tensor.content_hash,
                    float(weight),
                )
                for weight, (stratum, tensor) in zip(weights, ordered, strict=True)
            ),
        },
        prefix="irg-stratified-response-context-axis-",
    )
    return ResponseMomentTensor(
        channel_ids=reference.channel_ids,
        route_ids=reference.route_ids,
        mean=mean,
        second_moment_diagonal=second,
        mean_standard_error=mean_standard_error,
        second_moment_standard_error=second_standard_error,
        sample_count=sum(tensor.sample_count for _stratum, tensor in ordered),
        response_spec_hash=reference.response_spec_hash,
        metric_id=reference.metric_id,
        context_axis_hash=context_axis_hash,
        mean_estimator_covariance_diagonal=estimator_covariance,
        metadata={
            "aggregation": "frozen_stratified_risk_weight_projection_v1",
            "response_acquisition_design_hash": design.content_hash,
            "included_strata": included_strata,
            "normalized_stratum_weights": tuple(float(value) for value in weights),
            "stratum_tensor_hashes": tuple(
                tensor.content_hash for _stratum, tensor in ordered
            ),
            "projection_residual_bound": float(
                sum(
                    float(weight)
                    * tensor.projection_residual_bound_per_unit_route_coefficient
                    for weight, (_stratum, tensor) in zip(
                        weights,
                        ordered,
                        strict=True,
                    )
                )
            ),
        },
    )


def _stratified_response_update_outcome(
    *,
    version: str,
    design: FrozenResponseContextDesign,
    state: ResponseAcquisitionState,
    stratum: ResponseContextStratum,
    prefix: tuple[str, ...],
    unpacked: tuple[tuple[str, ResponseMomentTensor], ...],
    local_attribution: LocalAttributionMomentTensor | None,
    tripartite_pullback: TripartitePullbackMomentTensor | None,
    global_trust_geometry: GlobalTrustGeometryState | None,
    intrinsic_atlas_geometry: GlobalTrustGeometryState | None,
    physical_outcome: MeasurementOutcome,
    materialize_numeric_aggregate: bool = True,
) -> tuple[
    MeasurementOutcome | None,
    tuple[tuple[str, ResponseMomentTensor], ...],
    dict[str, dict[str, ResponseMomentTensor]],
    LocalAttributionMomentTensor | None,
    dict[str, LocalAttributionMomentTensor],
    TripartitePullbackMomentTensor | None,
    dict[str, TripartitePullbackMomentTensor],
    GlobalTrustGeometryState | None,
    dict[str, GlobalTrustGeometryState],
    GlobalTrustGeometryState | None,
    dict[str, GlobalTrustGeometryState],
]:
    """Build one atomic full-estimand update from staged stratum prefixes."""

    staged_moments = {
        stratum_id: dict(tensors)
        for stratum_id, tensors in state.stratum_moments.items()
    }
    staged_moments[stratum.stratum_id] = dict(unpacked)
    staged_local = dict(state.stratum_local_attribution_moments)
    if local_attribution is not None:
        staged_local[stratum.stratum_id] = local_attribution
    staged_tripartite = dict(state.stratum_tripartite_pullback_moments)
    if tripartite_pullback is not None:
        staged_tripartite[stratum.stratum_id] = tripartite_pullback
    staged_trust = dict(state.stratum_global_trust_geometries)
    if global_trust_geometry is not None:
        staged_trust[stratum.stratum_id] = global_trust_geometry
    staged_intrinsic_atlas = dict(state.stratum_intrinsic_atlas_geometries)
    if intrinsic_atlas_geometry is not None:
        staged_intrinsic_atlas[stratum.stratum_id] = intrinsic_atlas_geometry
    staged_prefixes = dict(state.cumulative_prefixes)
    staged_prefixes[stratum.stratum_id] = prefix
    initial_coverage_complete = all(
        len(staged_prefixes.get(item.stratum_id, ()))
        >= min(item.packet_target, item.maximum_target)
        for item in design.strata
    )
    if not initial_coverage_complete:
        # Physical measurement, resource, and lineage receipts are retained,
        # but no partial-stratum estimand may become the live scientific chart.
        return (
            None,
            unpacked,
            staged_moments,
            None,
            staged_local,
            None,
            staged_tripartite,
            None,
            staged_trust,
            None,
            staged_intrinsic_atlas,
        )
    if len(design.strata) == 1:
        return (
            physical_outcome,
            unpacked,
            staged_moments,
            local_attribution,
            staged_local,
            tripartite_pullback,
            staged_tripartite,
            global_trust_geometry,
            staged_trust,
            intrinsic_atlas_geometry,
            staged_intrinsic_atlas,
        )

    labels = tuple(label for label, _tensor in unpacked)
    if any(set(tensors) != set(labels) for tensors in staged_moments.values()):
        raise ValueError(
            "response strata do not expose the same version-owned moment charts"
        )
    combined = tuple(
        (
            label,
            _combine_stratified_response_moments(
                tensors_by_stratum={
                    stratum_id: tensors[label]
                    for stratum_id, tensors in staged_moments.items()
                },
                design=design,
                prefixes=staged_prefixes,
                label=label,
            ),
        )
        for label in labels
    )
    if staged_local and set(staged_local) != {
        item.stratum_id for item in design.strata
    }:
        raise ValueError(
            "local attribution must be present for every frozen response stratum"
        )
    combined_local = (
        None
        if not staged_local
        else _combine_stratified_local_attribution_moments(
            tensors_by_stratum=staged_local,
            design=design,
            prefixes=staged_prefixes,
        )
    )
    if staged_tripartite and set(staged_tripartite) != {
        item.stratum_id for item in design.strata
    }:
        raise ValueError(
            "tripartite pullback must be present for every response stratum"
        )
    combined_tripartite = (
        None
        if not staged_tripartite
        else _combine_stratified_tripartite_pullback_moments(
            tensors_by_stratum=staged_tripartite,
            design=design,
            prefixes=staged_prefixes,
        )
    )
    if staged_trust and set(staged_trust) != {
        item.stratum_id for item in design.strata
    }:
        raise ValueError(
            "global trust geometry must be present for every response stratum"
        )
    combined_trust = (
        None
        if not staged_trust
        else _combine_stratified_global_trust_geometries(
            geometries_by_stratum=staged_trust,
            design=design,
            prefixes=staged_prefixes,
        )
    )
    if staged_intrinsic_atlas and set(staged_intrinsic_atlas) != {
        item.stratum_id for item in design.strata
    }:
        raise ValueError(
            "intrinsic atlas geometry must be present for every response stratum"
        )
    combined_intrinsic_atlas = (
        None
        if not staged_intrinsic_atlas
        else _combine_stratified_global_trust_geometries(
            geometries_by_stratum=staged_intrinsic_atlas,
            design=design,
            prefixes=staged_prefixes,
        )
    )
    if not materialize_numeric_aggregate:
        context_ids = tuple(
            context_id
            for item in design.strata
            for context_id in staged_prefixes.get(item.stratum_id, ())
        )
        physical_hashes = tuple(
            outcome_hash
            for item in design.strata
            for outcome_hash in (
                state.outcome_hashes.get(item.stratum_id, ())
                + (
                    (physical_outcome.integrity_hash,)
                    if item.stratum_id == stratum.stratum_id
                    else ()
                )
            )
        )
        aggregate = MeasurementOutcome.create(
            request_id=physical_outcome.request_id,
            state_hash=physical_outcome.state_hash,
            context_ids=context_ids,
            candidate_ids=physical_outcome.candidate_ids,
            # The scientific numeric payload is retained by the typed objects
            # returned beside this lineage receipt. The sentinel prevents an
            # accidental empty-outcome identity without duplicating them.
            values=np.asarray([[float(len(context_ids))]], dtype=np.float64),
            calibration_state_id=physical_outcome.calibration_state_id,
            realized_cost=physical_outcome.realized_cost,
            cache_artifacts=physical_outcome.cache_artifacts,
            metadata={
                "kind": "response_moments",
                "mask_hash": physical_outcome.metadata.get("mask_hash"),
                "sample_count": len(context_ids),
                "stratified_aggregate": True,
                "typed_in_process_aggregate": True,
                "aggregation_schema": ("frozen_stratified_risk_weight_projection_v1"),
                "response_acquisition_design_hash": design.content_hash,
                "physical_outcome_hashes": physical_hashes,
                "component_schemas": {
                    "response": tuple(
                        (label, tensor.schema) for label, tensor in combined
                    ),
                    "local": (
                        None if combined_local is None else combined_local.schema
                    ),
                    "tripartite": (
                        None
                        if combined_tripartite is None
                        else combined_tripartite.schema
                    ),
                    "global_trust": (
                        None if combined_trust is None else combined_trust.schema
                    ),
                    "intrinsic_atlas": (
                        None
                        if combined_intrinsic_atlas is None
                        else combined_intrinsic_atlas.schema
                    ),
                },
                "stratum_prefixes": tuple(
                    (
                        item.stratum_id,
                        staged_prefixes.get(item.stratum_id, ()),
                    )
                    for item in design.strata
                    if item.stratum_id in staged_prefixes
                ),
            },
        )
        return (
            aggregate,
            combined,
            staged_moments,
            combined_local,
            staged_local,
            combined_tripartite,
            staged_tripartite,
            combined_trust,
            staged_trust,
            combined_intrinsic_atlas,
            staged_intrinsic_atlas,
        )
    packed = [tensor.pack() for _label, tensor in combined]
    if len(packed) == 1:
        values, layout = packed[0]
        layout_metadata: dict[str, object] = {"moment_layout": layout}
    else:
        fixed_values, fixed_layout = packed[0]
        intrinsic_values, intrinsic_layout = packed[1]
        values = np.concatenate((fixed_values, intrinsic_values), axis=0)
        layout_metadata = {
            "joint_moment_layout": {
                "schema": "fixed_and_intrinsic_response_moments_v1",
                "fixed_row_count": fixed_values.shape[0],
                "fixed": fixed_layout,
                "intrinsic": intrinsic_layout,
            }
        }
    if combined_local is not None:
        local_values, local_layout = combined_local.pack()
        local_offset = int(values.shape[0])
        values = np.concatenate((values, local_values), axis=0)
        layout_metadata["local_attribution_layout"] = {
            **local_layout,
            "row_offset": local_offset,
            "row_count": int(local_values.shape[0]),
        }
    if combined_tripartite is not None:
        pullback_values, pullback_layout = combined_tripartite.pack()
        pullback_offset = int(values.shape[0])
        values = np.concatenate((values, pullback_values), axis=0)
        layout_metadata["tripartite_pullback_layout"] = {
            **pullback_layout,
            "row_offset": pullback_offset,
            "row_count": int(pullback_values.shape[0]),
        }
    if combined_trust is not None:
        trust_values, trust_layout = combined_trust.pack()
        trust_offset = int(values.shape[0])
        values = np.concatenate((values, trust_values), axis=0)
        layout_metadata["global_trust_geometry_layout"] = {
            **trust_layout,
            "row_offset": trust_offset,
            "row_count": int(trust_values.shape[0]),
        }
    if combined_intrinsic_atlas is not None:
        intrinsic_values, intrinsic_layout = combined_intrinsic_atlas.pack()
        intrinsic_offset = int(values.shape[0])
        values = np.concatenate((values, intrinsic_values), axis=0)
        layout_metadata["intrinsic_atlas_geometry_layout"] = {
            **intrinsic_layout,
            "row_offset": intrinsic_offset,
            "row_count": int(intrinsic_values.shape[0]),
        }
    context_ids = tuple(
        context_id
        for item in design.strata
        for context_id in staged_prefixes.get(item.stratum_id, ())
    )
    physical_hashes = tuple(
        outcome_hash
        for item in design.strata
        for outcome_hash in (
            state.outcome_hashes.get(item.stratum_id, ())
            + (
                (physical_outcome.integrity_hash,)
                if item.stratum_id == stratum.stratum_id
                else ()
            )
        )
    )
    aggregate = MeasurementOutcome.create(
        request_id=physical_outcome.request_id,
        state_hash=physical_outcome.state_hash,
        context_ids=context_ids,
        candidate_ids=physical_outcome.candidate_ids,
        values=values,
        calibration_state_id=physical_outcome.calibration_state_id,
        realized_cost=physical_outcome.realized_cost,
        cache_artifacts=physical_outcome.cache_artifacts,
        metadata={
            **dict(physical_outcome.metadata),
            **layout_metadata,
            "stratified_aggregate": True,
            "response_acquisition_design_hash": design.content_hash,
            "physical_outcome_hashes": physical_hashes,
            "stratum_prefixes": tuple(
                (
                    item.stratum_id,
                    staged_prefixes.get(item.stratum_id, ()),
                )
                for item in design.strata
                if item.stratum_id in staged_prefixes
            ),
        },
    )
    return (
        aggregate,
        combined,
        staged_moments,
        combined_local,
        staged_local,
        combined_tripartite,
        staged_tripartite,
        combined_trust,
        staged_trust,
        combined_intrinsic_atlas,
        staged_intrinsic_atlas,
    )


def _require_retained_response_payload(
    *,
    version_state: object,
    unpacked: tuple[tuple[str, ResponseMomentTensor], ...],
    outcome_hash: str,
) -> None:
    """Require the scientific chart to retain the exact decoded measurement."""

    retained = getattr(version_state, "intrinsic_state", version_state)
    if retained is None:
        raise RuntimeError(
            "response acquisition scientific update lost its retained response chart"
        )
    for label, tensor in unpacked:
        retained_tensor = getattr(retained, label, None)
        if not isinstance(retained_tensor, ResponseMomentTensor):
            raise RuntimeError(
                "response acquisition scientific update did not retain the exact "
                f"typed {label}"
            )
        if (
            retained_tensor is not tensor
            and retained_tensor.content_hash != tensor.content_hash
        ):
            raise RuntimeError(
                "response acquisition scientific update retained response moments "
                "that differ from the physical outcome"
            )
    source_hashes = tuple(
        str(value) for value in getattr(retained, "source_outcome_hashes", ())
    )
    if outcome_hash not in source_hashes:
        raise RuntimeError(
            "response acquisition scientific update did not bind the physical "
            "outcome integrity hash into the retained chart"
        )


def response_acquisition_dependency_hash(session: object) -> str:
    """Physical identity whose change invalidates accumulated response rows.

    Hard functional partitions and action charts are deterministic consumers
    of this field, not physical inputs to it.  Including their hashes created
    a self-invalidating loop: publishing a better local/global sign chart
    discarded the very nested prefixes from which that chart was derived.
    """

    state = getattr(session, "version_state", None)
    intrinsic = getattr(state, "intrinsic_state", state)
    identity = getattr(session, "identity", None)
    return content_hash(
        {
            "mask_hash": getattr(
                getattr(session, "current_mask", None), "mask_hash", None
            ),
            "calibration_state_id": getattr(
                state,
                "calibration_state_id",
                getattr(intrinsic, "calibration_state_id", None),
            ),
            "response_constraint_hash": _hash_identity(
                getattr(identity, "response_constraint", None)
            ),
            "active_version": str(getattr(session, "active_version", "")),
        },
        prefix="irg-response-acquisition-dependency-",
    )


def _boundary_classification(
    entry: object,
    ceiling: float,
    *,
    critical_value: float,
) -> str:
    """Robust safety class; numerical equality is intentionally never decisive."""

    prediction = getattr(entry, "prediction", None)
    if prediction is None:
        return "unmeasured"
    if bool(getattr(entry, "measured", False)):
        center, error = float(entry.mean_quality_loss), max(
            float(entry.standard_error or 0.0), 0.0
        )
    else:
        center, error = float(prediction.predicted_quality_loss), max(
            float(prediction.standard_error), 0.0
        )
    margin = max(float(critical_value) * error, 1e-12)
    if center + margin < ceiling:
        return "safe"
    if center - margin > ceiling:
        return "unsafe"
    return "boundary"


def _primary_cost_coordinate(session: object) -> str:
    ledger = getattr(session, "frontier", None)
    coordinate = str(getattr(ledger, "primary_cost_coordinate", ""))
    if coordinate:
        return coordinate
    identity = getattr(session, "identity", None)
    cost = getattr(identity, "cost", None)
    coordinate = str(getattr(cost, "primary_coordinate", ""))
    return coordinate or "route_count"


def _action_semantic_key(
    session: object,
    candidate: object,
) -> str:
    """Identify one physical partition-local action without transient IDs.

    Candidate IDs intentionally bind generator and parent-state receipts.  A
    response-stability certificate instead concerns whether the same physical
    action remains decision-relevant after the response chart is refreshed.
    Its key therefore binds the comparability block, complete child mask,
    exact parent-to-child route delta, and deployable cost, but neither the
    candidate ID nor the mutable parent-state hash.
    """

    parent_mask = getattr(session, "current_mask", None)
    resulting_mask = getattr(candidate, "resulting_mask", None)
    parent_mask_hash = str(getattr(parent_mask, "mask_hash", ""))
    if (
        parent_mask is None
        or resulting_mask is None
        or not parent_mask_hash
        or str(getattr(candidate, "parent_mask_hash", "")) != parent_mask_hash
    ):
        raise RuntimeError(
            "response action semantics require a candidate on the current parent mask"
        )
    try:
        physical_delta = resulting_mask.removed_index_set_from(parent_mask)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError(
            "response action semantics could not resolve the exact physical "
            "parent-to-child route delta"
        ) from error
    removed_count = int(getattr(candidate, "removed_count", physical_delta.count))
    if physical_delta.count != removed_count:
        raise RuntimeError(
            "response action candidate's removed-route receipt disagrees with "
            "its complete physical masks"
        )
    retained_delta = getattr(candidate, "removed_routes", None)
    retained_delta_hash = getattr(retained_delta, "index_hash", None)
    if (
        isinstance(retained_delta_hash, str)
        and retained_delta_hash
        and retained_delta_hash != physical_delta.index_hash
    ):
        raise RuntimeError(
            "response action candidate's retained route indices disagree with "
            "its complete physical masks"
        )

    state = getattr(session, "version_state", None)
    intrinsic = getattr(state, "intrinsic_state", state)
    route_ids = tuple(str(value) for value in getattr(intrinsic, "route_ids", ()))
    resolved_route_ids: tuple[str, ...] | None = None
    if route_ids:
        if len(route_ids) != int(getattr(resulting_mask, "route_count", 0)):
            raise RuntimeError(
                "response action route-ID axis does not cover the complete hard mask"
            )
        resolved_route_ids = tuple(
            route_ids[int(index)] for index in physical_delta.to_numpy()
        )

    parameters = getattr(candidate, "generator_parameters", {})
    if not isinstance(parameters, Mapping):
        raise RuntimeError("response action generator parameters are not a mapping")
    raw_partition = parameters.get("geometric_partition_id")
    partition_id = (
        str(raw_partition)
        if raw_partition is not None and str(raw_partition)
        else UNPARTITIONED_GEOMETRIC_ACTION_BLOCK_ID
    )

    released_cost = getattr(candidate, "released_cost", None)
    coordinate = _primary_cost_coordinate(session)
    try:
        primary_cost = float(released_cost.coordinate(coordinate))
        released_cost_hash = content_hash(
            released_cost,
            prefix="irg-response-action-released-cost-",
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "response action semantics require the complete released-cost "
            "identity and its primary coordinate"
        ) from error
    if not math.isfinite(primary_cost):
        raise RuntimeError(
            "response action primary released-cost coordinate must be finite"
        )

    return content_hash(
        {
            "geometric_partition_id": partition_id,
            "resulting_mask_hash": str(getattr(resulting_mask, "mask_hash", "")),
            "physical_route_delta": {
                "route_axis_hash": str(getattr(resulting_mask, "route_axis_hash", "")),
                "removed_index_set_hash": physical_delta.index_hash,
                "removed_route_ids": resolved_route_ids,
            },
            "released_cost_hash": released_cost_hash,
            "primary_cost_coordinate": coordinate,
            "primary_released_cost": primary_cost,
        },
        prefix="irg-response-action-semantic-",
    )


def _practical_order_classes(
    session: object,
    entries: Sequence[object],
    semantic_keys: Mapping[str, str],
    *,
    critical_value: float,
) -> tuple[tuple[str, ...], ...]:
    indifference = max(
        float(
            getattr(getattr(session, "stopping", None), "practical_indifference", 0.0)
        ),
        0.0,
    )
    score_envelopes: dict[str, tuple[float, float]] = {}
    for entry in entries:
        prediction = getattr(entry, "prediction", None)
        if prediction is None:
            continue
        if bool(getattr(entry, "measured", False)):
            score, error = float(entry.mean_quality_loss), max(
                float(entry.standard_error or 0.0), 0.0
            )
        else:
            score, error = float(prediction.predicted_quality_loss), max(
                float(prediction.standard_error), 0.0
            )
        semantic_key = semantic_keys[str(entry.candidate.candidate_id)]
        lower, upper = score - error, score + error
        prior = score_envelopes.get(semantic_key)
        score_envelopes[semantic_key] = (
            lower if prior is None else min(lower, prior[0]),
            upper if prior is None else max(upper, prior[1]),
        )
    scored = sorted(
        (
            ((lower + upper) / 2.0, (upper - lower) / 2.0, semantic_key)
            for semantic_key, (lower, upper) in score_envelopes.items()
        ),
        key=lambda item: (item[0], item[2]),
    )
    classes: list[list[str]] = []
    anchors: list[tuple[float, float]] = []
    for score, error, semantic_key in scored:
        if not classes or abs(score - anchors[-1][0]) > indifference + float(
            critical_value
        ) * (error + anchors[-1][1]):
            classes.append([semantic_key])
            anchors.append((score, error))
        else:
            classes[-1].append(semantic_key)
    return tuple(tuple(sorted(group)) for group in classes)


def derive_response_action_signature(
    session: object,
    search: AdaptiveBoundarySearch | None = None,
    *,
    confidence_level: float = 0.95,
) -> str:
    """Hash simultaneous frontier classes, screen, and trust action.

    The response sample size targets invariance of this downstream action
    signature.  Candidate intervals use a Bonferroni critical value over the
    frozen active candidate family.  Stable signatures stop acquisition;
    unresolved designs fall back to their declared census/block action.
    """

    ledger = getattr(session, "frontier", None)
    current_mask_hash = getattr(getattr(session, "current_mask", None), "mask_hash", "")
    scoped_candidate_ids = (
        None if search is None else set(search.decision_relevant_candidate_ids())
    )
    entries = (
        []
        if ledger is None
        else [
            ledger.entry(candidate_id)
            for candidate_id in ledger.candidate_ids
            if (
                ledger.entry(candidate_id).candidate.parent_mask_hash
                == current_mask_hash
                and (
                    scoped_candidate_ids is None or candidate_id in scoped_candidate_ids
                )
            )
        ]
    )
    if (
        scoped_candidate_ids is not None
        and {str(entry.candidate.candidate_id) for entry in entries}
        != scoped_candidate_ids
    ):
        raise RuntimeError(
            "response decision scope references candidates outside the current "
            "parent frontier"
        )
    semantic_keys = {
        str(entry.candidate.candidate_id): _action_semantic_key(
            session,
            entry.candidate,
        )
        for entry in entries
    }
    ceiling = float(
        getattr(getattr(session, "stopping", None), "maximum_quality_loss", 0.0)
    )
    critical = simultaneous_normal_critical_value(
        confidence_level,
        family_size=max(1, len(entries)),
    )
    safety_rows = tuple(
        sorted(
            {
                (
                    semantic_keys[str(entry.candidate.candidate_id)],
                    _boundary_classification(
                        entry,
                        ceiling,
                        critical_value=critical,
                    ),
                    str(getattr(entry.status, "value", entry.status)),
                )
                for entry in entries
            }
        )
    )
    decision = getattr(session, "decision", None)
    top_set_ids = tuple(sorted(map(str, getattr(decision, "practical_top_set", set()))))
    unknown_top_set_ids = tuple(
        candidate_id
        for candidate_id in top_set_ids
        if candidate_id not in semantic_keys
    )
    if unknown_top_set_ids:
        raise RuntimeError(
            "response practical top set references candidates outside the "
            "active complete-mask decision scope"
        )
    practical_top_actions = tuple(
        sorted({semantic_keys[candidate_id] for candidate_id in top_set_ids})
    )
    scale = float(getattr(decision, "recommended_action_scale", 1.0))
    return content_hash(
        {
            "current_parent_mask_hash": str(current_mask_hash),
            "current_parent_action_safety_classes": safety_rows,
            "practical_tie_order_classes": _practical_order_classes(
                session,
                entries,
                semantic_keys,
                critical_value=critical,
            ),
            "pending_trust_action": str(
                getattr(decision, "pending_model_validity_action", "reuse")
            ),
            "trust_scale_class": (
                "full"
                if scale >= 1.0 - 1e-9
                else "shrunken" if scale > 1e-9 else "blocked"
            ),
            "practical_top_actions": practical_top_actions,
            "decision_scope_kind": (
                "current_parent_global_compatibility"
                if search is None
                else "adaptive_complete_mask_boundary"
            ),
            "confidence_level": float(confidence_level),
            "simultaneous_candidate_family_size": max(1, len(entries)),
            "simultaneous_critical_value": float(critical),
            "sample_size_target": (
                "active_frontier_screen_and_trust_action_invariance"
            ),
        },
        prefix="irg-response-action-signature-",
    )


def response_unified_acquisition_adapter(
    *,
    design: FrozenResponseContextDesign,
    code_hash: str,
    payload_factory: Callable[
        [object, ResponseContextStratum, tuple[str, ...]], Mapping[str, Any]
    ],
    outcome_consumer: Callable[[object, MeasurementRequest, object], None],
    version_update_callback: Callable[[object, MeasurementRequest, object], None],
    typed_version_update_callback: Callable[..., None] | None = None,
    session_admission: Callable[[object], bool],
    state_cache_key: str = RESPONSE_ACQUISITION_STATE_CACHE_KEY,
) -> UnifiedAcquisitionAdapter:
    """Create the v1--v4 whole-route sequential response adapter.

    The adapter owns the atomic initial stratified chart at ``response_state``
    and owns decision-specific prefix growth at ``boundary``.  This prevents
    candidate birth from falling back to a separate fixed-size response
    request before the sealed design becomes active.
    """

    if (
        not code_hash
        or not callable(payload_factory)
        or not callable(outcome_consumer)
        or not callable(session_admission)
    ):
        raise ValueError(
            "response acquisition adapter requires code and callable callbacks"
        )
    if not callable(version_update_callback):
        raise ValueError(
            "response acquisition requires a callable scientific version-state "
            "update; a no-op response adapter is not admissible"
        )
    if typed_version_update_callback is not None and not callable(
        typed_version_update_callback
    ):
        raise ValueError("typed response update callback must be callable")

    def semantic_version_state_hash(session: object) -> str:
        state = getattr(session, "version_state", None)
        semantic_hash = getattr(state, "state_hash", None)
        if semantic_hash is not None:
            return str(semantic_hash() if callable(semantic_hash) else semantic_hash)
        try:
            return content_hash(state, prefix="irg-response-version-state-")
        except TypeError:
            return content_hash(
                repr(state),
                prefix="irg-response-version-state-repr-",
            )

    def state_for(session: object) -> ResponseAcquisitionState:
        cache = getattr(session, "cache", None)
        if not isinstance(cache, dict):
            raise TypeError("response acquisition requires a mutable session cache")
        state = cache.get(state_cache_key)
        if state is None:
            state = ResponseAcquisitionState(design_hash=design.content_hash)
            cache[state_cache_key] = state
        if (
            not isinstance(state, ResponseAcquisitionState)
            or state.design_hash != design.content_hash
        ):
            raise RuntimeError(
                "response acquisition state belongs to another frozen design"
            )
        dependency = response_acquisition_dependency_hash(session)
        if state.dependency_hash is not None and state.dependency_hash != dependency:
            state.reset_for_state_transition("physical_dependency_changed")
        chart_hash = _response_chart_hash(session)
        if (
            state.owned_response_chart_hash is not None
            and state.owned_response_chart_hash != chart_hash
        ):
            # The raw sealed-prefix observations remain valid while the
            # physical dependency above is unchanged.  A trust refresh or
            # deterministic chart projection may clear/replace the published
            # response state, but it must not silently make the controller
            # rescore identical prefixes.  The adapter republishes its
            # sufficient statistics on the next nested update.
            state.chart_republication_count += 1
        state.dependency_hash = dependency
        state.owned_response_chart_hash = chart_hash
        return state

    def next_prefix(
        state: ResponseAcquisitionState, stratum: ResponseContextStratum
    ) -> tuple[str, ...]:
        prior = state.cumulative_prefixes.get(stratum.stratum_id, ())
        receipt = stratum.permutation_receipt
        if not isinstance(receipt, SealedUniformPermutationReceipt):
            raise RuntimeError("response stratum lost its sealed permutation receipt")
        receipt.verify()
        if prior != stratum.sampling_order[: len(prior)]:
            raise RuntimeError("response acquisition prefix is not design-nested")
        if state.unresolved_exhausted and design.unresolved_action == "full_census":
            return stratum.sampling_order
        return stratum.sampling_order[
            : min(stratum.maximum_target, len(prior) + stratum.packet_target)
        ]

    def requests(
        session: object, phase: str, search: object | None
    ) -> tuple[MeasurementRequest, ...]:
        if phase not in {"response_state", "boundary"} or not session_admission(
            session
        ):
            return ()
        if phase == "response_state" and search is not None:
            raise ValueError(
                "candidate-birth response acquisition may not bind a boundary search"
            )
        typed_search = (
            None
            if search is None
            else (search if isinstance(search, AdaptiveBoundarySearch) else None)
        )
        if search is not None and typed_search is None:
            raise TypeError(
                "boundary response acquisition requires AdaptiveBoundarySearch"
            )
        state = state_for(session)
        decision_scope_hash = (
            content_hash(
                {
                    "state_hash": str(getattr(session, "state_hash", "")),
                    "kind": "unscoped_compatibility",
                },
                prefix="irg-response-decision-scope-",
            )
            if typed_search is None
            else typed_search.decision_scope_hash
        )
        active_candidate_ids = (
            ()
            if typed_search is None
            else typed_search.decision_relevant_candidate_ids()
        )
        if typed_search is not None:
            decision = getattr(session, "decision", None)
            if decision is None or not hasattr(decision, "practical_top_set"):
                raise RuntimeError(
                    "scoped response acquisition requires mutable decision state"
                )
            # The active search bracket is authoritative for this look.  An
            # intersection with the prior set would silently omit a candidate
            # that has just become boundary-relevant while any old candidate
            # remains active.
            decision.practical_top_set = set(active_candidate_ids)
        if state.awaiting_action_signature:
            signature = derive_response_action_signature(
                session,
                typed_search,
                confidence_level=design.confidence_level,
            )
            if state.prior_action_signature is None:
                state.prior_action_signature = signature
                state.stable_looks = 0
            elif signature == state.prior_action_signature:
                state.stable_looks += 1
            else:
                state.prior_action_signature = signature
                state.stable_looks = 0
            state.awaiting_action_signature = False
        state.prior_decision_scope_hash = decision_scope_hash
        requested_rank_raw = getattr(session, "cache").get(
            GLOBAL_TRUST_REQUESTED_RANK_CACHE_KEY
        )
        requested_rank = None if requested_rank_raw is None else int(requested_rank_raw)
        rank_replay_strata = {
            stratum.stratum_id
            for stratum in design.strata
            if requested_rank is not None
            and state.has_initial_coverage(design)
            and (
                stratum.stratum_id not in state.stratum_global_trust_geometries
                or int(
                    state.stratum_global_trust_geometries[
                        stratum.stratum_id
                    ].metadata.get(
                        "requested_rank",
                        state.stratum_global_trust_geometries[
                            stratum.stratum_id
                        ].spec.initial_rank,
                    )
                )
                < requested_rank
            )
        }
        # Candidate birth needs one atomic packet from every nonempty frozen
        # stratum.  Decision-specific stability can only be evaluated after
        # those moments have produced an executable chart and candidates.
        if (
            phase == "response_state"
            and state.has_initial_coverage(design)
            and not rank_replay_strata
        ):
            return ()
        if state.is_resolved(design) and not rank_replay_strata:
            return ()
        if (
            state.unresolved_exhausted
            and design.unresolved_action == "block"
            and not rank_replay_strata
        ):
            return ()
        initial_coverage_required = not state.has_initial_coverage(design)
        full_census_fallback = (
            state.unresolved_exhausted and design.unresolved_action == "full_census"
        )
        result = []
        for stratum in design.strata:
            prior = state.cumulative_prefixes.get(stratum.stratum_id, ())
            rank_replay = stratum.stratum_id in rank_replay_strata
            if rank_replay:
                prefix = prior
                if not prefix:
                    raise RuntimeError(
                        "global trust rank replay lost its sealed stratum prefix"
                    )
            elif rank_replay_strata:
                continue
            target_limit = (
                min(stratum.packet_target, stratum.maximum_target)
                if initial_coverage_required
                else (
                    len(stratum.context_ids)
                    if full_census_fallback
                    else stratum.maximum_target
                )
            )
            if not rank_replay and len(prior) >= target_limit:
                if not initial_coverage_required:
                    state.exhausted_strata.add(stratum.stratum_id)
                continue
            if not rank_replay:
                prefix = next_prefix(state, stratum)
                if initial_coverage_required:
                    prefix = stratum.sampling_order[:target_limit]
            payload = dict(payload_factory(session, stratum, prefix))
            if "route_indices" in payload:
                raise ValueError(
                    "whole-route response acquisition may not narrow route_indices"
                )
            payload.update(
                {
                    "response_acquisition_design_hash": design.content_hash,
                    "response_acquisition_stratum_id": stratum.stratum_id,
                    "response_acquisition_prior_count": len(prior),
                    "response_acquisition_reset_count": state.reset_count,
                    "response_acquisition_last_reset_reason": (state.last_reset_reason),
                    "response_acquisition_chart_republication_count": (
                        state.chart_republication_count
                    ),
                    "response_acquisition_cumulative_prefix": prefix,
                    "response_acquisition_rank_replay": rank_replay,
                    "response_decision_scope_hash": decision_scope_hash,
                    "response_active_candidate_ids": active_candidate_ids,
                    "whole_physical_route_axis": True,
                    "response_stability_target": {
                        "kind": ("simultaneous_confidence_action_semantic_signature"),
                        "confidence_level": design.confidence_level,
                        "optional": not (
                            initial_coverage_required or full_census_fallback
                        ),
                        "stable_looks_required": design.stable_looks_required,
                        "maximum_target_is_full_stratum_population": (
                            stratum.maximum_target == len(stratum.context_ids)
                        ),
                        "unresolved_action": design.unresolved_action,
                        "full_census_fallback": full_census_fallback,
                    },
                }
            )
            if initial_coverage_required:
                payload.update(
                    {
                        "measurement_requirement_class": (
                            "candidate_birth_prerequisite"
                        ),
                        "measurement_obligation_id": (
                            "response-initial-stratified-coverage:"
                            f"{design.content_hash}"
                        ),
                        "initial_stratified_coverage_prerequisite": True,
                    }
                )
            elif rank_replay:
                payload.update(
                    {
                        "measurement_requirement_class": (
                            "chart_validity_prerequisite"
                        ),
                        "measurement_obligation_id": (
                            "global-trust-rank-replay:"
                            f"{design.content_hash}:{requested_rank}:"
                            f"{stratum.stratum_id}"
                        ),
                        "global_trust_rank_replay_prerequisite": True,
                    }
                )
            elif full_census_fallback:
                payload.update(
                    {
                        "measurement_requirement_class": (
                            "chart_validity_prerequisite"
                        ),
                        "measurement_obligation_id": (
                            "response-full-census:"
                            f"{design.content_hash}:{stratum.stratum_id}"
                        ),
                        "unresolved_response_stability_obligation": True,
                    }
                )
            result.append(
                MeasurementRequest.create(
                    kind="response_moments",
                    state_hash=str(getattr(session, "state_hash")),
                    version=getattr(session, "active_version"),
                    context_selector=ContextSelector(
                        selector_id=f"{design.design_id}:{stratum.stratum_id}:nested-prefix",
                        requested_count=len(prefix),
                        strata=(stratum.stratum_id,),
                        context_ids=prefix,
                        parameters={
                            "frozen_design_hash": design.content_hash,
                            "permutation_receipt_hash": (
                                stratum.permutation_receipt.content_hash
                            ),
                            "randomization_commitment": (
                                design.randomization_commitment
                            ),
                            "without_replacement": True,
                            "cumulative_nested_prefix": True,
                            "global_trust_rank_replay": rank_replay,
                            "full_census_fallback": full_census_fallback,
                        },
                    ),
                    expected_cost=stratum.expected_cost,
                    expected_decision_value=0.0,
                    reuse_keys=(design.content_hash, stratum.stratum_id),
                    provenance=MeasurementProvenance(
                        producer_id="irg_response_acquisition_adapter",
                        producer_version="1",
                        code_hash=code_hash,
                        notes={"whole_physical_route_axis": True},
                    ),
                    payload=payload,
                )
            )
        if (
            not rank_replay_strata
            and not result
            and all(
                len(state.cumulative_prefixes.get(item.stratum_id, ()))
                >= (
                    len(item.context_ids)
                    if full_census_fallback
                    else item.maximum_target
                )
                for item in design.strata
            )
        ):
            state.unresolved_exhausted = not state.is_resolved(design)
        return tuple(result)

    def forecast(
        session: object,
        request: MeasurementRequest,
        snapshot: DecisionUncertaintySnapshot,
        phase: str,
    ) -> AcquisitionForecast | None:
        del phase
        if (
            request.payload.get("response_acquisition_design_hash")
            != design.content_hash
        ):
            # Version-owned response_moments requests intentionally share the
            # public kind label.  They are forecast and consumed by the
            # canonical version path, not by this adaptive-prefix adapter.
            return None
        state = state_for(session)
        stratum = design.stratum(
            str(request.payload["response_acquisition_stratum_id"])
        )
        prior = state.cumulative_prefixes.get(stratum.stratum_id, ())
        added = len(request.context_selector.context_ids) - len(prior)
        rank_replay = bool(
            request.payload.get("response_acquisition_rank_replay", False)
        )
        if added <= 0 and not (rank_replay and added == 0):
            raise RuntimeError("response acquisition forecast has no new contexts")
        normalized_weight = stratum.weight / sum(item.weight for item in design.strata)
        retained = state.stratum_moments.get(stratum.stratum_id, {})
        empirical_variances = [
            float(np.sum(tensor.mean_estimator_covariance_diagonal))
            for tensor in retained.values()
        ]
        current = normalized_weight**2 * (
            max(
                max(empirical_variances),
                max(float(snapshot.response_variance), 0.0) / float(max(1, len(prior))),
            )
            if empirical_variances
            else max(float(snapshot.response_variance), 0.0)
        )
        baseline = max(1, len(prior))
        effective_work = len(prior) if rank_replay else added
        full_census_fallback = bool(
            request.payload.get("unresolved_response_stability_obligation", False)
        )
        initial_coverage_prerequisite = bool(
            request.payload.get(
                "initial_stratified_coverage_prerequisite",
                False,
            )
        )
        return AcquisitionForecast(
            request_id=request.request_id,
            state_hash=request.state_hash,
            forecast_model_id="stratified_whole_route_response_prefix_1_over_n_v1",
            uncertainty_source="frontier_response_geometry",
            current_variance=current,
            fantasy_posterior_variance=(
                current if rank_replay else current * baseline / float(baseline + added)
            ),
            current_sample_count=len(prior),
            fantasy_sample_count=len(prior) + added,
            rationale=(
                "rebuild the sealed stratum prefix at the required trust rank"
                if rank_replay
                else (
                    "complete the atomic initial frozen-stratum response estimand"
                    if initial_coverage_prerequisite
                    else (
                        "discharge unresolved response stability by exact stratum census"
                        if full_census_fallback
                        else "reduce active whole-route response variance on the next frozen stratum prefix"
                    )
                )
            ),
            metadata={
                "allocation_only": True,
                "stratum_id": stratum.stratum_id,
                "response_decision_scope_hash": request.payload[
                    "response_decision_scope_hash"
                ],
                "response_active_candidate_ids": request.payload[
                    "response_active_candidate_ids"
                ],
                "normalized_stratum_weight": normalized_weight,
                "incremental_context_count": added,
                "replayed_context_count": effective_work if rank_replay else 0,
                "global_trust_rank_replay_prerequisite": rank_replay,
                "expected_cost": stratum.expected_cost.to_dict(),
                "neyman_variance_reduction_proxy": (
                    0.0
                    if rank_replay
                    else current * added / float(max(1, baseline) * (baseline + added))
                ),
                "initial_stratified_coverage_prerequisite": (
                    initial_coverage_prerequisite
                ),
                "response_stability_target": {
                    "kind": "simultaneous_confidence_action_semantic_signature",
                    "confidence_level": design.confidence_level,
                    "optional": not (
                        initial_coverage_prerequisite or full_census_fallback
                    ),
                    "stable_looks_required": design.stable_looks_required,
                    "maximum_target_is_full_stratum_population": (
                        stratum.maximum_target == len(stratum.context_ids)
                    ),
                    "unresolved_action": design.unresolved_action,
                    "full_census_fallback": full_census_fallback,
                },
            },
        )

    def consume(session: object, request: MeasurementRequest, outcome: object) -> None:
        state = state_for(session)
        if (
            request.payload.get("response_acquisition_design_hash")
            != design.content_hash
            or "route_indices" in request.payload
            or not request.payload.get("whole_physical_route_axis")
        ):
            raise ValueError(
                "response acquisition request is not an owned whole-route request"
            )
        decision_scope_hash = str(
            request.payload.get("response_decision_scope_hash", "")
        )
        active_candidate_ids = tuple(
            map(str, request.payload.get("response_active_candidate_ids", ()))
        )
        if not decision_scope_hash or len(set(active_candidate_ids)) != len(
            active_candidate_ids
        ):
            raise ValueError(
                "response acquisition request lacks its frozen active-decision scope"
            )
        stratum = design.stratum(
            str(request.payload["response_acquisition_stratum_id"])
        )
        receipt = stratum.permutation_receipt
        if not isinstance(receipt, SealedUniformPermutationReceipt):
            raise RuntimeError("response stratum lost its sealed permutation receipt")
        parameters = request.context_selector.parameters
        if (
            parameters.get("frozen_design_hash") != design.content_hash
            or parameters.get("permutation_receipt_hash") != receipt.content_hash
            or parameters.get("randomization_commitment")
            != design.randomization_commitment
            or parameters.get("without_replacement") is not True
            or parameters.get("cumulative_nested_prefix") is not True
        ):
            raise ValueError(
                "response acquisition request does not carry the sealed permutation contract"
            )
        prefix = tuple(request.context_selector.context_ids)
        rank_replay = bool(
            request.payload.get("response_acquisition_rank_replay", False)
        )
        requested_rank = request.payload.get("global_trust_requested_rank")
        prior = state.cumulative_prefixes.get(stratum.stratum_id, ())
        current_geometry = state.stratum_global_trust_geometries.get(stratum.stratum_id)
        valid_rank_replay = bool(
            rank_replay
            and prefix == prior
            and requested_rank is not None
            and current_geometry is not None
            and int(
                current_geometry.metadata.get(
                    "requested_rank", current_geometry.spec.initial_rank
                )
            )
            < int(requested_rank)
        )
        if not valid_rank_replay and prefix != next_prefix(state, stratum):
            raise RuntimeError(
                "response acquisition request is not the next nested prefix"
            )
        if not isinstance(outcome, MeasurementOutcome):
            raise TypeError("response acquisition requires a typed MeasurementOutcome")
        if tuple(outcome.context_ids) != prefix:
            raise ValueError(
                "response acquisition outcome must cover its full cumulative prefix"
            )
        route_count = int(
            getattr(getattr(session, "current_mask", None), "route_count", 0)
        )
        values = outcome.values
        typed_summary = bool(
            outcome.metadata.get("in_process_typed_response_summary", False)
        )
        if (
            route_count <= 0
            or values.ndim != 2
            or (not typed_summary and values.shape[1] != route_count)
            or (typed_summary and tuple(values.shape) != (1, 1))
            or int(outcome.metadata.get("full_route_count", 0)) != route_count
            or int(outcome.metadata.get("returned_route_count", 0)) != route_count
        ):
            raise ValueError(
                "response acquisition outcome must retain packed statistics "
                "for the whole physical route axis"
            )
        before_state_hash = semantic_version_state_hash(session)
        before_partition_hash = _hash_identity(
            getattr(getattr(session, "identity", None), "geometric_partitioning", None)
        )
        before_state = getattr(session, "version_state", None)
        before_intrinsic = getattr(before_state, "intrinsic_state", before_state)
        version = str(getattr(session, "active_version", ""))
        expected_route_ids = canonical_unique_string_axis(
            getattr(before_intrinsic, "route_ids", ()),
            name="response acquisition route_ids",
        )
        if len(expected_route_ids) != route_count:
            raise RuntimeError(
                "response acquisition retained state lacks its whole physical route axis"
            )
        if typed_summary:
            expected_route_axis_hash = ordered_string_axis_hash(
                expected_route_ids,
                name="response acquisition route_ids",
            )
            if (
                str(
                    outcome.metadata.get(
                        "in_process_summary_route_axis_hash",
                        "",
                    )
                )
                != expected_route_axis_hash
            ):
                raise RuntimeError(
                    "typed response summary uses another ordered route axis"
                )
            summary = claim_in_process_response_summary(
                str(outcome.metadata.get("in_process_summary_token", "")),
                request_id=outcome.request_id,
                state_hash=outcome.state_hash,
                context_ids=outcome.context_ids,
                route_count=route_count,
                route_axis_hash=expected_route_axis_hash,
                component_hashes=outcome.metadata.get(
                    "in_process_summary_component_hashes"
                ),
            )
            unpacked = (("response_moments", summary.response_moments),)
            local_attribution = summary.local_attribution
            tripartite_pullback = summary.tripartite_pullback
            global_trust_geometry = summary.global_trust_geometry
            intrinsic_atlas_geometry = summary.intrinsic_atlas_geometry
        else:
            unpacked = _unpack_response_moment_payload(
                outcome=outcome,
                expected_route_ids=expected_route_ids,
            )
            local_attribution = _unpack_local_attribution_payload(
                outcome,
                expected_route_ids=expected_route_ids,
            )
            tripartite_pullback = _unpack_tripartite_pullback_payload(
                outcome,
                expected_route_ids=expected_route_ids,
            )
            global_trust_geometry = _unpack_global_trust_payload(
                outcome,
                expected_route_ids=expected_route_ids,
            )
            intrinsic_atlas_geometry = _unpack_intrinsic_atlas_geometry_payload(
                outcome,
                expected_route_ids=expected_route_ids,
            )
        use_typed_update = (
            typed_version_update_callback is not None
            and outcome.metadata.get("local_regime_summary_layout") is None
        )
        (
            version_update_outcome,
            retained_unpacked,
            staged_stratum_moments,
            _retained_local_attribution,
            staged_local_attribution_moments,
            _retained_tripartite_pullback,
            staged_tripartite_pullback_moments,
            _retained_global_trust_geometry,
            staged_global_trust_geometries,
            _retained_intrinsic_atlas_geometry,
            staged_intrinsic_atlas_geometries,
        ) = _stratified_response_update_outcome(
            version=version,
            design=design,
            state=state,
            stratum=stratum,
            prefix=prefix,
            unpacked=unpacked,
            local_attribution=local_attribution,
            tripartite_pullback=tripartite_pullback,
            global_trust_geometry=global_trust_geometry,
            intrinsic_atlas_geometry=intrinsic_atlas_geometry,
            physical_outcome=outcome,
            materialize_numeric_aggregate=not use_typed_update,
        )
        before_calibration = str(
            getattr(before_state, "calibration_state_id", "")
            or getattr(before_intrinsic, "calibration_state_id", "")
        )
        before_sample_count = int(getattr(before_intrinsic, "sample_count", 0))
        before_chart_hash = _response_chart_hash(session)
        if str(outcome.calibration_state_id) != before_calibration:
            raise RuntimeError(
                "response acquisition bypassed the frozen calibration state"
            )
        outcome_consumer(session, request, outcome)
        if semantic_version_state_hash(session) != before_state_hash:
            raise RuntimeError(
                "response acquisition physical outcome consumer may not mutate "
                "the scientific version state"
            )

        def retain_physical_stage(*, published_chart_hash: str) -> None:
            state.cumulative_prefixes[stratum.stratum_id] = prefix
            state.stratum_moments = staged_stratum_moments
            state.stratum_local_attribution_moments = staged_local_attribution_moments
            state.stratum_tripartite_pullback_moments = (
                staged_tripartite_pullback_moments
            )
            state.stratum_global_trust_geometries = staged_global_trust_geometries
            state.stratum_intrinsic_atlas_geometries = staged_intrinsic_atlas_geometries
            state.outcome_hashes[stratum.stratum_id] = state.outcome_hashes.get(
                stratum.stratum_id, ()
            ) + (outcome.integrity_hash,)
            if len(prefix) >= stratum.maximum_target:
                state.exhausted_strata.add(stratum.stratum_id)
            state.owned_response_chart_hash = published_chart_hash
            exhausted = all(
                len(state.cumulative_prefixes.get(item.stratum_id, ()))
                >= item.maximum_target
                for item in design.strata
            )
            state.unresolved_exhausted = exhausted and not state.is_resolved(design)
            getattr(session, "cache")[state_cache_key] = state

        if version_update_outcome is None:
            retain_physical_stage(published_chart_hash=before_chart_hash)
            # No action-signature look is earned until a full frozen-stratum
            # estimand has actually become the live response chart.
            state.awaiting_action_signature = False
            return

        if not use_typed_update:
            version_update_callback(session, request, version_update_outcome)
        else:
            assert typed_version_update_callback is not None
            typed_version_update_callback(
                session,
                request,
                version_update_outcome,
                retained_unpacked,
                _retained_local_attribution,
                _retained_tripartite_pullback,
                _retained_global_trust_geometry,
                _retained_intrinsic_atlas_geometry,
            )
        after_state = getattr(session, "version_state", None)
        after_intrinsic = getattr(after_state, "intrinsic_state", after_state)
        after_state_hash = semantic_version_state_hash(session)
        after_sample_count = int(getattr(after_intrinsic, "sample_count", 0))
        after_calibration = str(
            getattr(after_state, "calibration_state_id", "")
            or getattr(after_intrinsic, "calibration_state_id", "")
        )
        if after_state_hash == before_state_hash:
            raise RuntimeError(
                "response acquisition scientific update left the version state "
                "unchanged"
            )
        _require_retained_response_payload(
            version_state=after_state,
            unpacked=retained_unpacked,
            outcome_hash=version_update_outcome.integrity_hash,
        )
        if after_sample_count < before_sample_count:
            raise RuntimeError(
                "response acquisition scientific update regressed its sample count"
            )
        if (
            str(outcome.calibration_state_id) != before_calibration
            or after_calibration != before_calibration
        ):
            raise RuntimeError(
                "response acquisition changed or bypassed the frozen calibration "
                "state"
            )
        measured_mask_hash = str(getattr(after_intrinsic, "measured_mask_hash", ""))
        current_mask_hash = str(
            getattr(getattr(session, "current_mask", None), "mask_hash", "")
        )
        if measured_mask_hash and measured_mask_hash != current_mask_hash:
            raise RuntimeError(
                "response acquisition scientific update belongs to another hard mask"
            )
        dependency = response_acquisition_dependency_hash(session)
        if state.dependency_hash != dependency:
            after_partition = getattr(
                getattr(session, "identity", None),
                "geometric_partitioning",
                None,
            )
            after_partition_hash = _hash_identity(after_partition)
            local_partition_refresh = (
                _retained_local_attribution is not None
                and before_partition_hash != after_partition_hash
                and after_partition is not None
                and all(
                    item.metadata.get("source_geometry_hash")
                    == _retained_local_attribution.content_hash
                    and (
                        item.metadata.get("structural_sign_semantics")
                        == "local_attribution_not_weight_sign"
                        or (
                            item.metadata.get("partition_family")
                            == "randomized_2_partition"
                            and item.metadata.get("count_matched_to_sign_partition")
                            is True
                        )
                    )
                    for item in after_partition.partitions
                )
            )
            if not local_partition_refresh:
                state.reset_for_state_transition(
                    "scientific_update_changed_physical_dependency"
                )
            # A partition refresh derived solely from this same atomic
            # response estimand does not invalidate its own contexts.
            state.dependency_hash = dependency
        retain_physical_stage(published_chart_hash=_response_chart_hash(session))
        state.awaiting_action_signature = True

    def assert_ready(session: object) -> None:
        if not session_admission(session):
            raise ValueError(
                "response acquisition adapter rejected the session capability"
            )
        state_for(session)

    def blocking_reason(session: object, phase: str) -> str | None:
        if phase not in {"response_state", "boundary"} or not session_admission(
            session
        ):
            return None
        state = state_for(session)
        if not state.has_initial_coverage(design):
            return (
                "atomic initial frozen-stratum response coverage is incomplete; "
                "no candidate may be adjudicated or committed"
            )
        if state.unresolved_exhausted and design.unresolved_action == "block":
            return (
                "response action stability remains unresolved at the frozen "
                "noncensus horizon and the declared action is block"
            )
        return None

    return UnifiedAcquisitionAdapter(
        adapter_id="irg_adaptive_whole_route_response_acquisition_v1",
        supported_kinds=frozenset({"response_moments"}),
        request_factory=requests,
        outcome_consumer=consume,
        forecast_factory=forecast,
        readiness_checker=assert_ready,
        blocking_reason_checker=blocking_reason,
    )


__all__ = [
    "FrozenResponseContextDesign",
    "RESPONSE_ACQUISITION_STATE_CACHE_KEY",
    "ResponseAcquisitionState",
    "ResponseContextStratum",
    "SEALED_UNIFORM_PERMUTATION_ALGORITHM_V1",
    "SealedUniformPermutationReceipt",
    "UNPARTITIONED_GEOMETRIC_ACTION_BLOCK_ID",
    "derive_response_action_signature",
    "response_acquisition_dependency_hash",
    "response_unified_acquisition_adapter",
]
