"""Source-bound native destination-fiber geometry for modern v1.

Corrected evidence is additive only at a registered downstream gate.  For a
linear affine, that native gate is one output neuron.  A convolutional ReLU
gate is site-local, while a kernel route is shared over spatial sites.
Consequently, a channel-level route-addressable fiber is lawful only when an
external source receipt proves that signed sitewise cotangent contributions
are summed *before* magnitude/orientation.  It is a receipted spatial
contraction, not a claim that distinct spatial gates are identical.

This module turns those source facts into an O(number-of-cuts) receipt.  It
never materializes route-length cut or fiber identifier arrays.  Under the
registered C-order target-first layouts, route-to-fiber lookup is arithmetic
inside one binary-searched cut.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Final, Literal

from contracts.hashing import stable_digest
from mechanistic_systems.core.storage import RouteBlockSpec

from .modern_v1_neutral_runtime import DeclaredRouteAxis


NATIVE_DESTINATION_FIBER_AXIS_SCHEMA: Final = (
    "partition_free_v1_native_destination_fiber_axis_receipt_v1"
)
NATIVE_DESTINATION_FIBER_AXIS_CAPTURE_SCHEMA: Final = (
    "partition_free_v1_native_destination_fiber_axis_receipt_v2"
)
CAPTURE_STRUCTURAL_SCOPE: Final = "structural_only_requires_actual_capture_admission_v1"
PRIVATE_STRUCTURAL_SCOPE: Final = "private_cpu_meta_structural_reference_only_v1"
_CAPTURE_ISSUANCE_TOKEN: Final = object()

NATIVE_DESTINATION_FIBER_CUT_SCHEMA: Final = (
    "partition_free_v1_native_destination_fiber_cut_v1"
)
NATIVE_DESTINATION_ACTIVATION_PROOF_SCHEMA: Final = (
    "partition_free_v1_native_destination_activation_source_proof_v1"
)
SIGNED_SPATIAL_COTANGENT_REDUCER_PROOF_SCHEMA: Final = (
    "partition_free_v1_signed_spatial_cotangent_reducer_source_proof_v1"
)
NATIVE_DESTINATION_FIBER_COORDINATE_SCHEMA: Final = (
    "partition_free_v1_native_destination_fiber_coordinate_v1"
)

UNIFORM_CUT_COUNTING_FIBER_MEASURE_EQUATION: Final = (
    "F_corr_equals_E_context_sum_cut_uniform_probability_"
    "sum_native_destination_fiber_counting_measure_abs_"
    "sum_route_in_fiber_signed_corrected_evidence_v1"
)
WITHIN_CUT_MEASURE_ID: Final = (
    "native_destination_fiber_counting_total_variation_measure_v1"
)
NATIVE_FIBER_ROUTE_MAPPING_ID: Final = (
    "c_order_target_major_route_to_native_destination_fiber_v1"
)
LINEAR_NATIVE_GATE_ID: Final = "linear_output_neuron_gate_v1"
CONV_NATIVE_GATE_ID: Final = (
    "convolution_output_channel_route_addressable_spatial_contraction_v1"
)
SIGNED_SPATIAL_COTANGENT_REDUCER_EQUATION: Final = (
    "sum_spatial_site_signed_cotangent_times_route_drive_"
    "within_context_output_channel_before_magnitude_v1"
)
ACTIVATION_IMMEDIACY_PROOF_ID: Final = (
    "runtime_module_invocation_and_tensor_identity_single_occurrence_v1"
)

_SUPPORTED_ROUTE_KINDS: Final = frozenset(
    {"linear_weight", "conv_channel_pair", "conv_kernel_weight"}
)
_CONV_ROUTE_KINDS: Final = frozenset(
    {"conv_channel_pair", "conv_kernel_weight"}
)
_RECEIPT_PREFIX: Final = "partition-free-v1-native-fiber-axis-"
_ACTIVATION_PREFIX: Final = "partition-free-v1-native-activation-proof-"
_REDUCER_PREFIX: Final = "partition-free-v1-spatial-reducer-proof-"
_CUT_PREFIX: Final = "partition-free-v1-native-fiber-cut-"
_ISSUANCE_TOKEN: Final = object()


class V1NativeDestinationFiberAxisRefusalCode(str, Enum):
    """Typed fail-closed reasons for source-to-fiber lowering."""

    INVALID_TYPE = "invalid_type"
    HASH_MISMATCH = "hash_mismatch"
    AXIS_MISMATCH = "axis_mismatch"
    UNSUPPORTED_ROUTE_KIND = "unsupported_route_kind"
    ACTIVATION_UNPROVEN = "activation_unproven"
    SPATIAL_REDUCER_UNPROVEN = "spatial_reducer_unproven"
    SHARED_OCCURRENCE_AMBIGUITY = "shared_occurrence_ambiguity"
    INTEGRITY_FAILURE = "integrity_failure"


class V1NativeDestinationFiberAxisRefusal(ValueError):
    """A typed refusal instead of a guessed conservation geometry."""

    def __init__(
        self,
        code: V1NativeDestinationFiberAxisRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1NativeDestinationFiberAxisRefusalCode,
    message: str,
) -> None:
    raise V1NativeDestinationFiberAxisRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE,
            f"{name} must be one nonempty string",
        )
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _identity(value, name=name).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.HASH_MISMATCH,
            f"{name} must be one lowercase SHA-256 digest",
        )
    return result


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE,
            f"{name} must be one exact nonnegative integer",
        )
    result = int(value)
    if result < 0:
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
            f"{name} must be nonnegative",
        )
    return result


def _exact_positive_int(value: object, *, name: str) -> int:
    result = _exact_nonnegative_int(value, name=name)
    if result <= 0:
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
            f"{name} must be positive",
        )
    return result


def _route_block_content_hash(block: RouteBlockSpec) -> str:
    if type(block) is not RouteBlockSpec:
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE,
            "source route blocks must have exact RouteBlockSpec type",
        )
    block.validate()
    return stable_digest(block.to_dict())


@dataclass(frozen=True, slots=True)
class V1NativeDestinationActivationProof:
    """Source-derived affine-to-immediate-activation proof for one cut.

    The occurrence fields intentionally admit only a unique invocation.  A
    shared module called multiple times needs a future occurrence-addressed
    schema; silently pooling those calls would manufacture a fiber identity.
    """

    layer_id: str
    route_block_content_hash: str
    affine_module_path: str
    activation_kind: Literal["relu", "identity"]
    activation_module_path: str | None
    affine_occurrence_ordinal: int
    affine_occurrence_count: int
    immediate_consumer_tensor_identity_certified: bool
    execution_parity_receipt_hash: str
    activation_source_content_hash: str
    immediacy_proof_id: str = ACTIVATION_IMMEDIACY_PROOF_ID
    schema: str = NATIVE_DESTINATION_ACTIVATION_PROOF_SCHEMA

    def __post_init__(self) -> None:
        _identity(self.layer_id, name="activation-proof layer_id")
        _sha256(
            self.route_block_content_hash,
            name="activation-proof route-block content hash",
        )
        _identity(
            self.affine_module_path,
            name="activation-proof affine module path",
        )
        ordinal = _exact_nonnegative_int(
            self.affine_occurrence_ordinal,
            name="activation-proof affine occurrence ordinal",
        )
        count = _exact_positive_int(
            self.affine_occurrence_count,
            name="activation-proof affine occurrence count",
        )
        if count != 1 or ordinal != 0:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.SHARED_OCCURRENCE_AMBIGUITY,
                "shared or occurrence-indexed affine invocation lacks a "
                "registered native-fiber schema",
            )
        if self.activation_kind == "relu":
            path = _identity(
                self.activation_module_path,
                name="ReLU activation module path",
            )
            if path == self.affine_module_path:
                _refuse(
                    V1NativeDestinationFiberAxisRefusalCode.ACTIVATION_UNPROVEN,
                    "ReLU activation must be an explicit immediate consumer",
                )
        elif self.activation_kind == "identity":
            if self.activation_module_path is not None:
                _refuse(
                    V1NativeDestinationFiberAxisRefusalCode.ACTIVATION_UNPROVEN,
                    "identity activation cannot name a separate module",
                )
        else:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.ACTIVATION_UNPROVEN,
                "only explicit relu and identity activations are supported",
            )
        if self.immediate_consumer_tensor_identity_certified is not True:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.ACTIVATION_UNPROVEN,
                "native fibers require a passed immediate-consumer tensor "
                "identity proof",
            )
        _sha256(
            self.execution_parity_receipt_hash,
            name="activation-proof execution parity receipt hash",
        )
        _sha256(
            self.activation_source_content_hash,
            name="activation source content hash",
        )
        if self.immediacy_proof_id != ACTIVATION_IMMEDIACY_PROOF_ID:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.ACTIVATION_UNPROVEN,
                "unsupported activation immediacy proof",
            )
        if self.schema != NATIVE_DESTINATION_ACTIVATION_PROOF_SCHEMA:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE,
                "unsupported activation-proof schema",
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "layer_id": self.layer_id,
            "route_block_content_hash": self.route_block_content_hash,
            "affine_module_path": self.affine_module_path,
            "activation_kind": self.activation_kind,
            "activation_module_path": self.activation_module_path,
            "affine_occurrence_ordinal": self.affine_occurrence_ordinal,
            "affine_occurrence_count": self.affine_occurrence_count,
            "immediate_consumer_tensor_identity_certified": (
                self.immediate_consumer_tensor_identity_certified
            ),
            "execution_parity_receipt_hash": (
                self.execution_parity_receipt_hash
            ),
            "activation_source_content_hash": (
                self.activation_source_content_hash
            ),
            "immediacy_proof_id": self.immediacy_proof_id,
        }

    @property
    def proof_hash(self) -> str:
        return f"{_ACTIVATION_PREFIX}{stable_digest(self.to_payload())}"


@dataclass(frozen=True, slots=True)
class V1SignedSpatialCotangentReducerProof:
    """Registered reason a convolutional output-channel fiber is lawful."""

    layer_id: str
    route_block_content_hash: str
    affine_module_path: str
    route_kind: Literal["conv_channel_pair", "conv_kernel_weight"]
    route_parameter_shared_across_spatial_sites: bool
    signed_site_contributions_summed_before_magnitude: bool
    sitewise_gate_axis_materialized: bool
    reducer_implementation_hash: str
    spatial_geometry_content_hash: str
    execution_parity_receipt_hash: str
    reducer_equation: str = SIGNED_SPATIAL_COTANGENT_REDUCER_EQUATION
    schema: str = SIGNED_SPATIAL_COTANGENT_REDUCER_PROOF_SCHEMA

    def __post_init__(self) -> None:
        _identity(self.layer_id, name="spatial-reducer layer_id")
        _sha256(
            self.route_block_content_hash,
            name="spatial-reducer route-block content hash",
        )
        _identity(
            self.affine_module_path,
            name="spatial-reducer affine module path",
        )
        if self.route_kind not in _CONV_ROUTE_KINDS:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.SPATIAL_REDUCER_UNPROVEN,
                "spatial reduction proof is reserved for convolution routes",
            )
        if (
            self.route_parameter_shared_across_spatial_sites is not True
            or self.signed_site_contributions_summed_before_magnitude is not True
            or self.sitewise_gate_axis_materialized is not False
        ):
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.SPATIAL_REDUCER_UNPROVEN,
                "convolutional channel fibers require a spatially shared route "
                "parameter, signed sitewise summation before magnitude, and "
                "no fictitious route-addressable sitewise fiber axis",
            )
        _sha256(
            self.reducer_implementation_hash,
            name="spatial-reducer implementation hash",
        )
        _sha256(
            self.spatial_geometry_content_hash,
            name="spatial geometry content hash",
        )
        _sha256(
            self.execution_parity_receipt_hash,
            name="spatial-reducer execution parity receipt hash",
        )
        if self.reducer_equation != SIGNED_SPATIAL_COTANGENT_REDUCER_EQUATION:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.SPATIAL_REDUCER_UNPROVEN,
                "unsupported spatial cotangent reducer equation",
            )
        if self.schema != SIGNED_SPATIAL_COTANGENT_REDUCER_PROOF_SCHEMA:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE,
                "unsupported spatial-reducer proof schema",
            )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "layer_id": self.layer_id,
            "route_block_content_hash": self.route_block_content_hash,
            "affine_module_path": self.affine_module_path,
            "route_kind": self.route_kind,
            "route_parameter_shared_across_spatial_sites": (
                self.route_parameter_shared_across_spatial_sites
            ),
            "signed_site_contributions_summed_before_magnitude": (
                self.signed_site_contributions_summed_before_magnitude
            ),
            "sitewise_gate_axis_materialized": (
                self.sitewise_gate_axis_materialized
            ),
            "reducer_implementation_hash": self.reducer_implementation_hash,
            "spatial_geometry_content_hash": self.spatial_geometry_content_hash,
            "execution_parity_receipt_hash": (
                self.execution_parity_receipt_hash
            ),
            "reducer_equation": self.reducer_equation,
        }

    @property
    def proof_hash(self) -> str:
        return f"{_REDUCER_PREFIX}{stable_digest(self.to_payload())}"


def v1_native_activation_registry_hash(
    proofs: Sequence[V1NativeDestinationActivationProof],
) -> str:
    """Hash an ordered, exact activation-proof registry."""

    values = tuple(proofs)
    if not values or any(
        type(item) is not V1NativeDestinationActivationProof for item in values
    ):
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE,
            "activation registry requires exact typed proofs",
        )
    return stable_digest(
        {
            "schema": "native_destination_activation_registry_v1",
            "proof_hashes": [item.proof_hash for item in values],
        }
    )


def v1_signed_spatial_reducer_registry_hash(
    proofs: Sequence[V1SignedSpatialCotangentReducerProof],
) -> str:
    """Hash an ordered, exact convolutional spatial-reducer registry."""

    values = tuple(proofs)
    if any(type(item) is not V1SignedSpatialCotangentReducerProof for item in values):
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE,
            "spatial-reducer registry requires exact typed proofs",
        )
    return stable_digest(
        {
            "schema": "signed_spatial_cotangent_reducer_registry_v1",
            "proof_hashes": [item.proof_hash for item in values],
        }
    )


@dataclass(frozen=True, slots=True)
class V1NativeDestinationFiberCut:
    """One compact target-major cut record; storage is independent of routes."""

    cut_index: int
    cut_id: str
    layer_id: str
    route_kind: str
    route_block_id: str
    route_block_content_hash: str
    route_shape: tuple[int, ...]
    route_offset: int
    route_stop: int
    routes_per_fiber: int
    fiber_offset: int
    fiber_stop: int
    activation_kind: Literal["relu", "identity"]
    activation_proof_hash: str
    native_gate_id: str
    spatial_reducer_proof_hash: str | None
    cut_probability_numerator: int
    cut_probability_denominator: int
    within_cut_measure_id: str = WITHIN_CUT_MEASURE_ID
    route_mapping_id: str = NATIVE_FIBER_ROUTE_MAPPING_ID
    schema: str = NATIVE_DESTINATION_FIBER_CUT_SCHEMA
    _issuance_token: object = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUANCE_TOKEN:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE,
                "native-fiber cuts must be issued by the source-bound builder",
            )
        cut_index = _exact_nonnegative_int(self.cut_index, name="cut index")
        _identity(self.cut_id, name="cut ID")
        _identity(self.layer_id, name="cut layer ID")
        if self.route_kind not in _SUPPORTED_ROUTE_KINDS:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.UNSUPPORTED_ROUTE_KIND,
                f"unsupported affine route kind {self.route_kind!r}",
            )
        _identity(self.route_block_id, name="route-block ID")
        _sha256(
            self.route_block_content_hash,
            name="cut route-block content hash",
        )
        shape = tuple(
            _exact_positive_int(value, name="route-shape extent")
            for value in self.route_shape
        )
        if len(shape) < 2:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.UNSUPPORTED_ROUTE_KIND,
                "native destination fibers require an affine target/source axis",
            )
        route_offset = _exact_nonnegative_int(
            self.route_offset,
            name="cut route offset",
        )
        route_stop = _exact_positive_int(self.route_stop, name="cut route stop")
        routes_per_fiber = _exact_positive_int(
            self.routes_per_fiber,
            name="routes per fiber",
        )
        fiber_offset = _exact_nonnegative_int(
            self.fiber_offset,
            name="cut fiber offset",
        )
        fiber_stop = _exact_positive_int(self.fiber_stop, name="cut fiber stop")
        expected_routes_per_fiber = 1
        for extent in shape[1:]:
            expected_routes_per_fiber *= extent
        if (
            routes_per_fiber != expected_routes_per_fiber
            or route_stop - route_offset != shape[0] * routes_per_fiber
            or fiber_stop - fiber_offset != shape[0]
        ):
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
                "cut does not replay target-major route/fiber geometry",
            )
        if self.activation_kind not in {"relu", "identity"}:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.ACTIVATION_UNPROVEN,
                "cut activation is unsupported",
            )
        _identity(self.activation_proof_hash, name="activation proof hash")
        expected_gate = (
            CONV_NATIVE_GATE_ID
            if self.route_kind in _CONV_ROUTE_KINDS
            else LINEAR_NATIVE_GATE_ID
        )
        if self.native_gate_id != expected_gate:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
                "cut native-gate identity differs from its route kind",
            )
        if self.route_kind in _CONV_ROUTE_KINDS:
            _identity(
                self.spatial_reducer_proof_hash,
                name="convolutional spatial-reducer proof hash",
            )
        elif self.spatial_reducer_proof_hash is not None:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
                "linear cut cannot claim a spatial-reducer proof",
            )
        if (
            self.cut_probability_numerator != 1
            or _exact_positive_int(
                self.cut_probability_denominator,
                name="cut-probability denominator",
            )
            <= cut_index
        ):
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
                "cut measure must be one exact uniform probability atom",
            )
        if self.within_cut_measure_id != WITHIN_CUT_MEASURE_ID:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
                "within-cut measure must be counting/total variation",
            )
        if self.route_mapping_id != NATIVE_FIBER_ROUTE_MAPPING_ID:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
                "unsupported route-to-fiber mapping",
            )
        if self.schema != NATIVE_DESTINATION_FIBER_CUT_SCHEMA:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE,
                "unsupported native-fiber cut schema",
            )
        object.__setattr__(self, "cut_index", cut_index)
        object.__setattr__(self, "route_shape", shape)
        object.__setattr__(self, "route_offset", route_offset)
        object.__setattr__(self, "route_stop", route_stop)
        object.__setattr__(self, "routes_per_fiber", routes_per_fiber)
        object.__setattr__(self, "fiber_offset", fiber_offset)
        object.__setattr__(self, "fiber_stop", fiber_stop)

    @property
    def route_count(self) -> int:
        return self.route_stop - self.route_offset

    @property
    def fiber_count(self) -> int:
        return self.fiber_stop - self.fiber_offset

    @property
    def cut_probability(self) -> float:
        return self.cut_probability_numerator / self.cut_probability_denominator

    def route_span_for_fiber(self, fiber_index_within_cut: int) -> tuple[int, int]:
        index = _exact_nonnegative_int(
            fiber_index_within_cut,
            name="fiber index within cut",
        )
        if index >= self.fiber_count:
            raise IndexError("native destination fiber index out of range")
        start = self.route_offset + index * self.routes_per_fiber
        return start, start + self.routes_per_fiber

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "cut_index": self.cut_index,
            "cut_id": self.cut_id,
            "layer_id": self.layer_id,
            "route_kind": self.route_kind,
            "route_block_id": self.route_block_id,
            "route_block_content_hash": self.route_block_content_hash,
            "route_shape": self.route_shape,
            "route_offset": self.route_offset,
            "route_stop": self.route_stop,
            "routes_per_fiber": self.routes_per_fiber,
            "fiber_offset": self.fiber_offset,
            "fiber_stop": self.fiber_stop,
            "activation_kind": self.activation_kind,
            "activation_proof_hash": self.activation_proof_hash,
            "native_gate_id": self.native_gate_id,
            "spatial_reducer_proof_hash": self.spatial_reducer_proof_hash,
            "cut_probability_numerator": self.cut_probability_numerator,
            "cut_probability_denominator": self.cut_probability_denominator,
            "within_cut_measure_id": self.within_cut_measure_id,
            "route_mapping_id": self.route_mapping_id,
        }

    @property
    def content_hash(self) -> str:
        return f"{_CUT_PREFIX}{stable_digest(self.to_payload())}"


@dataclass(frozen=True, slots=True)
class V1NativeDestinationFiberCoordinate:
    """One lazily lowered route coordinate; never persisted route-scale."""

    route_ordinal: int
    cut_index: int
    cut_id: str
    layer_id: str
    fiber_index_within_cut: int
    global_fiber_index: int
    fiber_id: str
    schema: str = NATIVE_DESTINATION_FIBER_COORDINATE_SCHEMA

    def __post_init__(self) -> None:
        _exact_nonnegative_int(self.route_ordinal, name="route ordinal")
        _exact_nonnegative_int(self.cut_index, name="cut index")
        _identity(self.cut_id, name="cut ID")
        _identity(self.layer_id, name="layer ID")
        _exact_nonnegative_int(
            self.fiber_index_within_cut,
            name="fiber index within cut",
        )
        _exact_nonnegative_int(
            self.global_fiber_index,
            name="global fiber index",
        )
        _identity(self.fiber_id, name="fiber ID")
        if self.schema != NATIVE_DESTINATION_FIBER_COORDINATE_SCHEMA:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE,
                "unsupported native-fiber coordinate schema",
            )


@dataclass(frozen=True, slots=True)
class V1NativeDestinationFiberAxisReceipt:
    """Immutable O(cuts) authority for the native work-fiber measure."""

    architecture_id: str
    topology_hash: str
    source_architecture_schema_hash: str
    source_architecture_content_hash: str
    execution_parity_receipt_hash: str
    route_axis_schema: str
    route_axis_hash: str
    route_id_axis_hash: str
    route_axis_content_hash: str
    activation_registry_hash: str
    spatial_reducer_registry_hash: str
    route_count: int
    cut_count: int
    fiber_count: int
    cuts: tuple[V1NativeDestinationFiberCut, ...]
    measure_equation: str = UNIFORM_CUT_COUNTING_FIBER_MEASURE_EQUATION
    within_cut_measure_id: str = WITHIN_CUT_MEASURE_ID
    route_mapping_id: str = NATIVE_FIBER_ROUTE_MAPPING_ID
    native_fiber_merging_supported: bool = False
    schema: str = NATIVE_DESTINATION_FIBER_AXIS_SCHEMA
    execution_scope: str | None = None
    structural_source_binding_hash: str | None = None
    _issuance_token: object = field(
        default=None,
        repr=False,
        compare=False,
    )
    _route_stops: tuple[int, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _fiber_stops: tuple[int, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _receipt_hash: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._issuance_token is not _ISSUANCE_TOKEN:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE,
                "native-fiber axis receipts must be issued by the "
                "source-bound builder",
            )
        _identity(self.architecture_id, name="architecture ID")
        for name in (
            "topology_hash",
            "source_architecture_schema_hash",
            "source_architecture_content_hash",
            "execution_parity_receipt_hash",
            "route_axis_hash",
            "route_id_axis_hash",
            "route_axis_content_hash",
            "activation_registry_hash",
            "spatial_reducer_registry_hash",
        ):
            _sha256(getattr(self, name), name=name.replace("_", " "))
        _identity(self.route_axis_schema, name="route-axis schema")
        route_count = _exact_positive_int(self.route_count, name="route count")
        cut_count = _exact_positive_int(self.cut_count, name="cut count")
        fiber_count = _exact_positive_int(self.fiber_count, name="fiber count")
        cuts = tuple(self.cuts)
        if (
            len(cuts) != cut_count
            or any(type(item) is not V1NativeDestinationFiberCut for item in cuts)
        ):
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
                "cut records do not close the declared cut axis",
            )
        expected_route_offset = 0
        expected_fiber_offset = 0
        seen_cut_ids: set[str] = set()
        seen_layers: set[str] = set()
        for index, cut in enumerate(cuts):
            if (
                cut.cut_index != index
                or cut.route_offset != expected_route_offset
                or cut.fiber_offset != expected_fiber_offset
                or cut.cut_probability_numerator != 1
                or cut.cut_probability_denominator != cut_count
                or cut.cut_id in seen_cut_ids
                or cut.layer_id in seen_layers
            ):
                _refuse(
                    V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
                    "native-fiber cuts are not unique, contiguous, and "
                    "uniformly weighted",
                )
            seen_cut_ids.add(cut.cut_id)
            seen_layers.add(cut.layer_id)
            expected_route_offset = cut.route_stop
            expected_fiber_offset = cut.fiber_stop
        if expected_route_offset != route_count or expected_fiber_offset != fiber_count:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
                "native-fiber cut records do not close their compact axes",
            )
        if self.measure_equation != UNIFORM_CUT_COUNTING_FIBER_MEASURE_EQUATION:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
                "native-fiber measure equation is not canonical",
            )
        if self.within_cut_measure_id != WITHIN_CUT_MEASURE_ID:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
                "native-fiber measure must count fibers within each cut",
            )
        if self.route_mapping_id != NATIVE_FIBER_ROUTE_MAPPING_ID:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
                "unsupported native-fiber route mapping",
            )
        if self.native_fiber_merging_supported is not False:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
                "this receipt cannot merge distinct native destination gates",
            )
        if self.schema == NATIVE_DESTINATION_FIBER_AXIS_CAPTURE_SCHEMA:
            if self.execution_scope not in {CAPTURE_STRUCTURAL_SCOPE, PRIVATE_STRUCTURAL_SCOPE}:
                _refuse(V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE, "invalid structural execution scope")
            _sha256(self.structural_source_binding_hash, name="structural source binding hash")
        elif (self.schema != NATIVE_DESTINATION_FIBER_AXIS_SCHEMA
              or self.execution_scope is not None or self.structural_source_binding_hash is not None):
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE,
                "unsupported native-fiber axis schema",
            )
        object.__setattr__(self, "cuts", cuts)
        object.__setattr__(self, "route_count", route_count)
        object.__setattr__(self, "cut_count", cut_count)
        object.__setattr__(self, "fiber_count", fiber_count)
        object.__setattr__(
            self,
            "_route_stops",
            tuple(cut.route_stop for cut in cuts),
        )
        object.__setattr__(
            self,
            "_fiber_stops",
            tuple(cut.fiber_stop for cut in cuts),
        )
        object.__setattr__(self, "_receipt_hash", self._compute_receipt_hash())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "architecture_id": self.architecture_id,
            "topology_hash": self.topology_hash,
            "source_architecture_schema_hash": (
                self.source_architecture_schema_hash
            ),
            "source_architecture_content_hash": (
                self.source_architecture_content_hash
            ),
            "execution_parity_receipt_hash": (
                self.execution_parity_receipt_hash
            ),
            "route_axis_schema": self.route_axis_schema,
            "route_axis_hash": self.route_axis_hash,
            "route_id_axis_hash": self.route_id_axis_hash,
            "route_axis_content_hash": self.route_axis_content_hash,
            "activation_registry_hash": self.activation_registry_hash,
            "spatial_reducer_registry_hash": (
                self.spatial_reducer_registry_hash
            ),
            "route_count": self.route_count,
            "cut_count": self.cut_count,
            "fiber_count": self.fiber_count,
            "cuts": [cut.to_payload() for cut in self.cuts],
            "measure_equation": self.measure_equation,
            "within_cut_measure_id": self.within_cut_measure_id,
            "route_mapping_id": self.route_mapping_id,
            "native_fiber_merging_supported": (
                self.native_fiber_merging_supported
            ),
            **({
                "execution_scope": self.execution_scope,
                "structural_source_binding_hash": self.structural_source_binding_hash,
                "execution_parity_reference_status": "opaque_provenance_only",
            } if self.schema == NATIVE_DESTINATION_FIBER_AXIS_CAPTURE_SCHEMA else {}),
        }

    def _compute_receipt_hash(self) -> str:
        return f"{_RECEIPT_PREFIX}{stable_digest(self.to_payload())}"

    @property
    def receipt_hash(self) -> str:
        return self._receipt_hash

    @property
    def measure_hash(self) -> str:
        return stable_digest(
            {
                "schema": "native_destination_fiber_measure_v1",
                "axis_receipt_hash": self.receipt_hash,
                "measure_equation": self.measure_equation,
                "cut_probability": [1, self.cut_count],
                "within_cut_measure_id": self.within_cut_measure_id,
                "fiber_count_by_cut": [cut.fiber_count for cut in self.cuts],
            }
        )

    @property
    def total_fiber_measure(self) -> float:
        """Mass of counting/TV fibers after probability-averaging cuts."""

        return self.fiber_count / self.cut_count

    @property
    def compact_record_count(self) -> int:
        """Number of persistent variable-size records (never route count)."""

        return len(self.cuts)

    def assert_integrity(self) -> None:
        if self._receipt_hash != self._compute_receipt_hash():
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.INTEGRITY_FAILURE,
                "native-fiber receipt content differs from its issued hash",
            )

    def cut_for_route(self, route_ordinal: int) -> V1NativeDestinationFiberCut:
        route = _exact_nonnegative_int(route_ordinal, name="route ordinal")
        if route >= self.route_count:
            raise IndexError("native-fiber route ordinal out of range")
        return self.cuts[bisect_right(self._route_stops, route)]

    def cut_for_fiber(self, global_fiber_index: int) -> V1NativeDestinationFiberCut:
        fiber = _exact_nonnegative_int(
            global_fiber_index,
            name="global fiber index",
        )
        if fiber >= self.fiber_count:
            raise IndexError("native destination fiber index out of range")
        return self.cuts[bisect_right(self._fiber_stops, fiber)]

    def route_to_fiber(
        self,
        route_ordinal: int,
    ) -> V1NativeDestinationFiberCoordinate:
        route = _exact_nonnegative_int(route_ordinal, name="route ordinal")
        cut = self.cut_for_route(route)
        local_route = route - cut.route_offset
        local_fiber = local_route // cut.routes_per_fiber
        global_fiber = cut.fiber_offset + local_fiber
        return V1NativeDestinationFiberCoordinate(
            route_ordinal=route,
            cut_index=cut.cut_index,
            cut_id=cut.cut_id,
            layer_id=cut.layer_id,
            fiber_index_within_cut=local_fiber,
            global_fiber_index=global_fiber,
            fiber_id=(
                f"native-destination-fiber:v1:{cut.cut_index}:{local_fiber}"
            ),
        )

    def fiber_weight(self, global_fiber_index: int) -> float:
        self.cut_for_fiber(global_fiber_index)
        return 1.0 / self.cut_count


def build_v1_native_destination_fiber_axis(
    route_axis: DeclaredRouteAxis,
    *,
    source_route_blocks: Sequence[RouteBlockSpec],
    activation_proofs: Sequence[V1NativeDestinationActivationProof],
    spatial_reducer_proofs: Sequence[V1SignedSpatialCotangentReducerProof],
    source_architecture_schema_hash: str,
    source_architecture_content_hash: str,
    execution_parity_receipt_hash: str,
    activation_registry_hash: str,
    spatial_reducer_registry_hash: str,
    _structural_capture: tuple[str, str] | None = None,
    _capture_issuance_token: object = None,
) -> V1NativeDestinationFiberAxisReceipt:
    """Issue a compact axis only from exact, mutually consistent source facts."""

    if _structural_capture is not None and _capture_issuance_token is not _CAPTURE_ISSUANCE_TOKEN:
        _refuse(V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE, "structural capture axes are issuer-only")

    if type(route_axis) is not DeclaredRouteAxis:
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.INVALID_TYPE,
            "native-fiber lowering requires an exact DeclaredRouteAxis",
        )
    blocks = tuple(source_route_blocks)
    if (
        not blocks
        or any(type(block) is not RouteBlockSpec for block in blocks)
        or blocks != route_axis.route_blocks
    ):
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
            "source route blocks differ from the declared route axis",
        )
    for block in blocks:
        block.validate()
    architecture_ids = {block.architecture_id for block in blocks}
    topology_hashes = {block.topology_hash for block in blocks}
    if len(architecture_ids) != 1 or len(topology_hashes) != 1:
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.AXIS_MISMATCH,
            "source route blocks mix architecture or topology identities",
        )
    for block in blocks:
        if block.route_kind not in _SUPPORTED_ROUTE_KINDS:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.UNSUPPORTED_ROUTE_KIND,
                f"route kind {block.route_kind!r} has no native affine "
                "destination-fiber lowering",
            )
        if len(block.shape) < 2:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.UNSUPPORTED_ROUTE_KIND,
                "one-dimensional or residual gates need another fiber schema",
            )

    parity_hash = _sha256(
        execution_parity_receipt_hash,
        name="execution parity receipt hash",
    )
    architecture_schema_hash = _sha256(
        source_architecture_schema_hash,
        name="source architecture schema hash",
    )
    architecture_content_hash = _sha256(
        source_architecture_content_hash,
        name="source architecture content hash",
    )

    activations = tuple(activation_proofs)
    if (
        len(activations) != len(blocks)
        or any(type(item) is not V1NativeDestinationActivationProof for item in activations)
    ):
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.ACTIVATION_UNPROVEN,
            "activation proofs must cover every source route block exactly once",
        )
    if v1_native_activation_registry_hash(activations) != _sha256(
        activation_registry_hash,
        name="activation registry hash",
    ):
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.HASH_MISMATCH,
            "activation registry differs from its source hash",
        )
    activation_by_layer: dict[str, V1NativeDestinationActivationProof] = {}
    for proof in activations:
        if proof.layer_id in activation_by_layer:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.SHARED_OCCURRENCE_AMBIGUITY,
                "activation registry repeats a route layer",
            )
        activation_by_layer[proof.layer_id] = proof
    if len({item.affine_module_path for item in activations}) != len(activations):
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.SHARED_OCCURRENCE_AMBIGUITY,
            "multiple cuts share one affine module path without an "
            "occurrence-addressed schema",
        )

    reducers = tuple(spatial_reducer_proofs)
    expected_conv_count = sum(
        block.route_kind in _CONV_ROUTE_KINDS for block in blocks
    )
    if (
        len(reducers) != expected_conv_count
        or any(type(item) is not V1SignedSpatialCotangentReducerProof for item in reducers)
    ):
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.SPATIAL_REDUCER_UNPROVEN,
            "every convolutional cut requires one exact spatial-reducer proof",
        )
    if v1_signed_spatial_reducer_registry_hash(reducers) != _sha256(
        spatial_reducer_registry_hash,
        name="spatial-reducer registry hash",
    ):
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.HASH_MISMATCH,
            "spatial-reducer registry differs from its source hash",
        )
    reducer_by_layer: dict[str, V1SignedSpatialCotangentReducerProof] = {}
    for proof in reducers:
        if proof.layer_id in reducer_by_layer:
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.SPATIAL_REDUCER_UNPROVEN,
                "spatial-reducer registry repeats a route layer",
            )
        reducer_by_layer[proof.layer_id] = proof

    if set(activation_by_layer) != {block.layer_id for block in blocks}:
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.ACTIVATION_UNPROVEN,
            "activation proofs name a different route-layer axis",
        )
    if set(reducer_by_layer) != {
        block.layer_id for block in blocks if block.route_kind in _CONV_ROUTE_KINDS
    }:
        _refuse(
            V1NativeDestinationFiberAxisRefusalCode.SPATIAL_REDUCER_UNPROVEN,
            "spatial-reducer proofs name a different convolutional cut axis",
        )

    cut_count = len(blocks)
    fiber_offset = 0
    cuts: list[V1NativeDestinationFiberCut] = []
    for cut_index, block in enumerate(blocks):
        block_hash = _route_block_content_hash(block)
        activation = activation_by_layer[block.layer_id]
        if (
            activation.route_block_content_hash != block_hash
            or activation.execution_parity_receipt_hash != parity_hash
        ):
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.ACTIVATION_UNPROVEN,
                f"activation proof for {block.layer_id!r} is stale",
            )
        reducer = reducer_by_layer.get(block.layer_id)
        if block.route_kind in _CONV_ROUTE_KINDS:
            if (
                reducer is None
                or reducer.route_block_content_hash != block_hash
                or reducer.affine_module_path != activation.affine_module_path
                or reducer.route_kind != block.route_kind
                or reducer.execution_parity_receipt_hash != parity_hash
            ):
                _refuse(
                    V1NativeDestinationFiberAxisRefusalCode.SPATIAL_REDUCER_UNPROVEN,
                    f"spatial-reducer proof for {block.layer_id!r} is stale",
                )
        elif reducer is not None:  # pragma: no cover - excluded by exact set check
            _refuse(
                V1NativeDestinationFiberAxisRefusalCode.SPATIAL_REDUCER_UNPROVEN,
                "linear cut cannot carry a spatial-reducer proof",
            )
        routes_per_fiber = 1
        for extent in block.shape[1:]:
            routes_per_fiber *= int(extent)
        fiber_count = int(block.shape[0])
        cut = V1NativeDestinationFiberCut(
            cut_index=cut_index,
            cut_id=f"native-destination-cut:v1:{cut_index}:{block.layer_id}",
            layer_id=block.layer_id,
            route_kind=block.route_kind,
            route_block_id=block.block_id,
            route_block_content_hash=block_hash,
            route_shape=tuple(int(value) for value in block.shape),
            route_offset=int(block.offset),
            route_stop=int(block.stop),
            routes_per_fiber=routes_per_fiber,
            fiber_offset=fiber_offset,
            fiber_stop=fiber_offset + fiber_count,
            activation_kind=activation.activation_kind,
            activation_proof_hash=activation.proof_hash,
            native_gate_id=(
                CONV_NATIVE_GATE_ID
                if block.route_kind in _CONV_ROUTE_KINDS
                else LINEAR_NATIVE_GATE_ID
            ),
            spatial_reducer_proof_hash=(
                None if reducer is None else reducer.proof_hash
            ),
            cut_probability_numerator=1,
            cut_probability_denominator=cut_count,
            _issuance_token=_ISSUANCE_TOKEN,
        )
        cuts.append(cut)
        fiber_offset = cut.fiber_stop

    result = V1NativeDestinationFiberAxisReceipt(
        architecture_id=next(iter(architecture_ids)),
        topology_hash=next(iter(topology_hashes)),
        source_architecture_schema_hash=architecture_schema_hash,
        source_architecture_content_hash=architecture_content_hash,
        execution_parity_receipt_hash=parity_hash,
        route_axis_schema=route_axis.schema,
        route_axis_hash=route_axis.route_axis_hash,
        route_id_axis_hash=route_axis.route_id_axis_hash,
        route_axis_content_hash=route_axis.content_hash,
        activation_registry_hash=activation_registry_hash,
        spatial_reducer_registry_hash=spatial_reducer_registry_hash,
        route_count=route_axis.route_count,
        cut_count=cut_count,
        fiber_count=fiber_offset,
        cuts=tuple(cuts),
        schema=(NATIVE_DESTINATION_FIBER_AXIS_SCHEMA if _structural_capture is None
                else NATIVE_DESTINATION_FIBER_AXIS_CAPTURE_SCHEMA),
        execution_scope=None if _structural_capture is None else _structural_capture[0],
        structural_source_binding_hash=None if _structural_capture is None else _structural_capture[1],
        _issuance_token=_ISSUANCE_TOKEN,
    )
    result.assert_integrity()
    return result


__all__ = [
    "ACTIVATION_IMMEDIACY_PROOF_ID",
    "CONV_NATIVE_GATE_ID",
    "LINEAR_NATIVE_GATE_ID",
    "NATIVE_DESTINATION_ACTIVATION_PROOF_SCHEMA",
    "NATIVE_DESTINATION_FIBER_AXIS_SCHEMA",
    "NATIVE_DESTINATION_FIBER_COORDINATE_SCHEMA",
    "NATIVE_DESTINATION_FIBER_CUT_SCHEMA",
    "NATIVE_FIBER_ROUTE_MAPPING_ID",
    "SIGNED_SPATIAL_COTANGENT_REDUCER_EQUATION",
    "SIGNED_SPATIAL_COTANGENT_REDUCER_PROOF_SCHEMA",
    "UNIFORM_CUT_COUNTING_FIBER_MEASURE_EQUATION",
    "V1NativeDestinationActivationProof",
    "V1NativeDestinationFiberAxisReceipt",
    "V1NativeDestinationFiberAxisRefusal",
    "V1NativeDestinationFiberAxisRefusalCode",
    "V1NativeDestinationFiberCoordinate",
    "V1NativeDestinationFiberCut",
    "V1SignedSpatialCotangentReducerProof",
    "WITHIN_CUT_MEASURE_ID",
    "build_v1_native_destination_fiber_axis",
    "v1_native_activation_registry_hash",
    "v1_signed_spatial_reducer_registry_hash",
]
