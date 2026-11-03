"""Bind the sampled singleton corrected-work point to one relaxed-KKT problem.

This module owns one deliberately narrow identity seam.  It proves that the
current confidence-box point, the factory-issued sampled selector point, and
``V1RelaxedKKTProblem.corrected_work`` are the same ordered little-endian FP64
vector.  That vector is the routewise triangle envelope ``A_j = E_x |a_xj|``
used by the legacy linear work branch ``A^T z``.  It is not the full signed
action-work functional ``E_x |sum_j a_xj z_j|``.  The neutral position
certificate carries this ordered triangle vector onto the problem's physical
mask axis.

The route-scale CPU work here is admitted only as a bounded-memory binding
barrier: a read-only, fixed-maximum-span byte scan.  It never sorts, selects,
solves, rounds, constructs a route-scale key, or allocates whole-axis scratch.
Consequently the receipt grants singleton-envelope *identity* authority only.
It does not certify the point statistically, certify F/Q/R, identify a KKT
face, authorize signed action-work use, or authorize any later campaign stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
from numbers import Integral
from types import MappingProxyType
from typing import Final, NoReturn, cast

import numpy as np

from .contracts import content_hash
from .modern_v1_confidence_box_external_state import (
    CONFIDENCE_BOX_V2_EXTERNAL_METADATA_SCHEMA,
    V1ConfidenceBoxExternalDescriptor,
    V1ConfidenceBoxExternalView,
)
from .modern_v1_external_state import (
    V1ExternalStateDescriptor,
    V1ExternalStateInspectionReceipt,
    V1ExternalStateView,
)
from .modern_v1_proposal_statistics import (
    CORRECTED_ATTRIBUTION_CONFIDENCE_BOX_V2_SCHEMA,
)
from .modern_v1_relaxed_kkt import (
    RELAXED_KKT_SINGLETON_TRIANGLE_WORK_ID,
    V1RelaxedKKTProblem,
)
from .modern_v1_sampled_acquisition_adapters import (
    V1SampledPreFreezeGeometrySource,
)
from .modern_v1_sampled_geometric_provenance import (
    SAMPLED_CORRECTED_PATH_HASH_PREFIX,
    V1SampledProposalPositionCertificate,
    V1SampledSnapshotSelectorNumerics,
)


V1_KKT_CORRECTED_WORK_IDENTITY_SCHEMA: Final = (
    "partition_free_v1_kkt_corrected_work_identity_v1"
)
V1_KKT_CORRECTED_WORK_IDENTITY_AUTHORITY_SCOPE: Final = (
    "exact_confidence_point_sampled_numerics_kkt_corrected_work_identity_"
    "only_no_F_Q_R_J_face_C_K_O_oracle_commit_production_or_pareto_"
    "authority_v1"
)
V1_KKT_CORRECTED_WORK_IDENTITY_SEMANTICS: Final = (
    RELAXED_KKT_SINGLETON_TRIANGLE_WORK_ID
)
V1_KKT_CORRECTED_WORK_BINDING_SCAN_ALGORITHM: Final = (
    "readonly_le_f8_sha256_and_bitwise_equality_fixed_64MiB_max_span_"
    "no_whole_axis_scratch_no_selection_v1"
)

_RECEIPT_PREFIX: Final = "partition-free-v1-kkt-corrected-work-identity-"
_POINT_NAME: Final = "class_balanced_absolute_point"
_CURRENT_BOX_ARRAY_NAMES: Final = (
    "class_balanced_absolute_point",
    "current_class_balanced_absolute_lower",
    "current_class_balanced_absolute_upper",
    "cumulative_class_balanced_absolute_lower",
    "cumulative_class_balanced_absolute_upper",
)
_LE_F8: Final = np.dtype("<f8")
_MAX_SCAN_CHUNK_ELEMENTS: Final = 8 * 1024 * 1024
_DEFAULT_SCAN_CHUNK_ELEMENTS: Final = 1024 * 1024
_MAX_MEMORYVIEW_SPAN_BYTES: Final = _MAX_SCAN_CHUNK_ELEMENTS * _LE_F8.itemsize


class V1KKTCorrectedWorkIdentityRefusalCode(str, Enum):
    """Fail-closed reasons for the corrected-work identity barrier."""

    INVALID_TYPE = "invalid_type"
    INVALID_SCAN_DESIGN = "invalid_scan_design"
    EXTERNAL_VIEW_INTEGRITY = "external_view_integrity"
    SOURCE_CHAIN_MISMATCH = "source_chain_mismatch"
    STALE_COORDINATE = "stale_coordinate"
    AXIS_MISMATCH = "axis_mismatch"
    ARRAY_POLICY_MISMATCH = "array_policy_mismatch"
    POINT_HASH_MISMATCH = "point_hash_mismatch"
    POINT_VALUE_MISMATCH = "point_value_mismatch"


class V1KKTCorrectedWorkIdentityRefusal(ValueError):
    """Typed refusal from the authority-inert corrected-work binding seam."""

    def __init__(
        self,
        code: V1KKTCorrectedWorkIdentityRefusalCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


def _refuse(
    code: V1KKTCorrectedWorkIdentityRefusalCode,
    message: str,
) -> NoReturn:
    raise V1KKTCorrectedWorkIdentityRefusal(code, message)


def _identity(value: object, *, name: str) -> str:
    if type(value) is not str or not cast(str, value):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            f"{name} must be one nonempty string",
        )
    return cast(str, value)


def _exact_int(value: object, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            f"{name} must be one exact integer",
        )
    result = int(cast(Integral, value))
    if result < (1 if positive else 0):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            f"{name} is outside its registered range",
        )
    return result


def _sequence(value: object, *, name: str, length: int | None = None) -> Sequence[object]:
    if type(value) not in {list, tuple}:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            f"{name} must be one exact sequence",
        )
    result = cast(Sequence[object], value)
    if length is not None and len(result) != length:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            f"{name} has the wrong length",
        )
    return result


def _identity_pair(value: object, *, name: str) -> tuple[str, str]:
    result = _sequence(value, name=name, length=2)
    return (
        _identity(result[0], name=f"{name}[0]"),
        _identity(result[1], name=f"{name}[1]"),
    )


def _int_pair(value: object, *, name: str, positive: bool = False) -> tuple[int, int]:
    result = _sequence(value, name=name, length=2)
    return (
        _exact_int(result[0], name=f"{name}[0]", positive=positive),
        _exact_int(result[1], name=f"{name}[1]", positive=positive),
    )


def _identity_ledger_pair(
    value: object,
    *,
    name: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    result = _sequence(value, name=name, length=2)
    axes: list[tuple[str, ...]] = []
    for axis_index, raw_axis in enumerate(result):
        axis = _sequence(raw_axis, name=f"{name}[{axis_index}]")
        axes.append(
            tuple(
                _identity(item, name=f"{name}[{axis_index}] item")
                for item in axis
            )
        )
    return axes[0], axes[1]


def _class_balanced_weights(value: object) -> tuple[float, float]:
    result = _sequence(value, name="class-balanced stratum weights", length=2)
    if any(type(item) is not float for item in result):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            "class-balanced stratum weights must be exact JSON floats",
        )
    weights = (float(result[0]), float(result[1]))
    if weights != (0.5, 0.5):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.STALE_COORDINATE,
            "confidence point changed the registered class-balanced mixture",
        )
    return weights


def _require_pairs(
    pairs: Sequence[tuple[str, object, object]],
    *,
    code: V1KKTCorrectedWorkIdentityRefusalCode,
) -> None:
    for name, left, right in pairs:
        if left != right or type(left) is not type(right):
            _refuse(code, f"{name} differs across the registered identity chain")


def _validated_external_payload(
    view: V1ConfidenceBoxExternalView,
) -> tuple[Mapping[str, object], np.ndarray, str]:
    """Replay the scalar descriptor/view barrier and expose its current point."""

    descriptor = view.descriptor
    external_view = view.external_view
    if type(descriptor) is not V1ConfidenceBoxExternalDescriptor:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.INVALID_TYPE,
            "confidence-box view contains a noncanonical descriptor",
        )
    if type(external_view) is not V1ExternalStateView:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.INVALID_TYPE,
            "confidence-box view contains a noncanonical external view",
        )
    try:
        rebuilt = V1ConfidenceBoxExternalDescriptor.from_json_dict(
            descriptor.to_json_dict()
        )
    except Exception as error:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            f"confidence-box descriptor does not replay: {error}",
        )
    if rebuilt.descriptor_hash != descriptor.descriptor_hash:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            "confidence-box descriptor hash changed during replay",
        )
    if type(external_view.descriptor) is not V1ExternalStateDescriptor:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.INVALID_TYPE,
            "external view contains a noncanonical generic descriptor",
        )
    if (
        external_view.descriptor.descriptor_hash
        != descriptor.external_state.descriptor_hash
    ):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            "external view and typed descriptor name different publications",
        )
    inspection = external_view.inspection_receipt
    if type(inspection) is not V1ExternalStateInspectionReceipt:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.INVALID_TYPE,
            "external view lacks its exact inspection receipt",
        )
    try:
        inspection.__post_init__()
    except Exception as error:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            f"external inspection receipt does not replay: {error}",
        )
    expected_identity_hashes = tuple(
        identity.identity_hash for identity in descriptor.external_state.arrays
    )
    if (
        inspection.descriptor_hash != descriptor.external_state.descriptor_hash
        or inspection.manifest_hash != descriptor.external_state.manifest_hash
        or inspection.array_identity_hashes != expected_identity_hashes
    ):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            "external inspection barrier differs from the typed descriptor",
        )

    arrays = external_view.arrays
    if type(arrays) is not MappingProxyType:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.INVALID_TYPE,
            "external arrays require one read-only mapping",
        )
    expected_names = tuple(
        identity.logical_name for identity in descriptor.external_state.arrays
    )
    if set(arrays) != set(expected_names):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            "external array mapping differs from its descriptor manifest",
        )

    metadata = view.metadata
    if type(metadata) is not MappingProxyType or set(metadata) != {
        "schema",
        "current",
        "prior",
    }:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            "confidence-box metadata is not the exact verified mapping",
        )
    if metadata["schema"] != CONFIDENCE_BOX_V2_EXTERNAL_METADATA_SCHEMA:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            "confidence-box metadata schema changed after open",
        )
    current = metadata["current"]
    if type(current) is not dict or set(current) != {
        "receipt_hash",
        "payload_without_arrays",
        "array_names",
    }:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            "current confidence-box metadata wrapper is not exact",
        )
    current_map = cast(dict[str, object], current)
    if current_map["receipt_hash"] != descriptor.receipt_hash:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            "metadata names another confidence-box receipt",
        )
    names = _sequence(current_map["array_names"], name="current array names")
    if tuple(names) != tuple(sorted(_CURRENT_BOX_ARRAY_NAMES)):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            "current confidence-box array ledger is incomplete or reordered",
        )
    payload = current_map["payload_without_arrays"]
    if type(payload) is not dict:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            "current confidence-box scalar payload is not exact",
        )
    payload_map = cast(dict[str, object], payload)
    if payload_map.get("schema") != CORRECTED_ATTRIBUTION_CONFIDENCE_BOX_V2_SCHEMA:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            "current confidence-box receipt schema changed after open",
        )
    if payload_map.get("cumulative_intersection_nonempty") is not True:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            "current confidence-box cumulative intersection is not certified",
        )
    descriptor_pairs = (
        ("confidence-box look", descriptor.look_index, payload_map.get("look_index")),
        ("confidence-box route count", descriptor.route_count, payload_map.get("route_count")),
        (
            "confidence-box ordered axis",
            descriptor.route_axis_hash,
            payload_map.get("route_axis_hash"),
        ),
        (
            "confidence-box prior state",
            descriptor.prior_statistics_state_hash,
            payload_map.get("prior_statistics_state_hash"),
        ),
    )
    _require_pairs(
        descriptor_pairs,
        code=V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
    )

    try:
        point = arrays[_POINT_NAME]
    except (KeyError, ValueError) as error:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.EXTERNAL_VIEW_INTEGRITY,
            f"current confidence point is unavailable (possibly closed): {error}",
        )
    if not isinstance(point, np.ndarray):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.INVALID_TYPE,
            "current confidence point is not a NumPy array",
        )
    return payload_map, point, inspection.receipt_hash


def _validated_point_array(
    value: np.ndarray,
    *,
    route_count: int,
    name: str,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.INVALID_TYPE,
            f"{name} must be one NumPy array",
        )
    if (
        value.dtype.str != _LE_F8.str
        or value.shape != (route_count,)
        or not value.flags.c_contiguous
        or value.flags.writeable
    ):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.ARRAY_POLICY_MISMATCH,
            f"{name} must be an immutable C-contiguous exact <f8 vector",
        )
    return value


def _sampled_path_hash_from_raw_sha256(*, raw_sha256: str, route_count: int) -> str:
    """Replay ``content_hash(array, SAMPLED_...)`` from a bounded byte scan."""

    return content_hash(
        {
            "dtype": str(_LE_F8),
            "shape": [route_count],
            "sha256": raw_sha256,
        },
        prefix=SAMPLED_CORRECTED_PATH_HASH_PREFIX,
    )


def _scan_exact_points(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    chunk_elements: int,
) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """Hash and bit-compare three vectors without any numeric scratch array."""

    byte_views: list[memoryview] = []
    try:
        for value in arrays:
            byte_views.append(memoryview(value).cast("B"))
    except (BufferError, TypeError, ValueError) as error:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.ARRAY_POLICY_MISMATCH,
            f"point arrays do not expose a live read-only byte view: {error}",
        )
    byte_count = len(byte_views[0])
    if any(len(value) != byte_count for value in byte_views[1:]):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.POINT_VALUE_MISMATCH,
            "point byte lengths differ",
        )
    digests = (hashlib.sha256(), hashlib.sha256(), hashlib.sha256())
    chunk_bytes = chunk_elements * _LE_F8.itemsize
    for start in range(0, byte_count, chunk_bytes):
        stop = min(byte_count, start + chunk_bytes)
        chunks = tuple(value[start:stop] for value in byte_views)
        for digest, chunk in zip(digests, chunks, strict=True):
            digest.update(chunk)
        if chunks[0] != chunks[1] or chunks[0] != chunks[2]:
            _refuse(
                V1KKTCorrectedWorkIdentityRefusalCode.POINT_VALUE_MISMATCH,
                "confidence point, sampled numerics, and KKT work differ bitwise",
            )
    raw_hashes = cast(
        tuple[str, str, str],
        tuple(digest.hexdigest() for digest in digests),
    )
    route_count = byte_count // _LE_F8.itemsize
    sampled_hashes = cast(
        tuple[str, str, str],
        tuple(
            _sampled_path_hash_from_raw_sha256(
                raw_sha256=value,
                route_count=route_count,
            )
            for value in raw_hashes
        ),
    )
    return raw_hashes, sampled_hashes


@dataclass(frozen=True, slots=True)
class V1KKTCorrectedWorkIdentityReceipt:
    """Array-free identity proof for one sampled corrected-work KKT input."""

    confidence_box_external_descriptor_hash: str
    confidence_box_external_commit_hash: str
    confidence_box_external_inspection_receipt_hash: str
    confidence_box_receipt_hash: str
    confidence_box_statistics_state_hash: str
    confidence_box_design_hash: str
    trajectory_alpha_ledger_hash: str
    sampled_geometry_source_hash: str
    sampled_geometry_admission_receipt_hash: str
    sampled_geometry_assembly_receipt_hash: str
    sampled_snapshot_certificate_hash: str
    sampled_snapshot_numerics_hash: str
    proposal_position_certificate_hash: str
    physical_route_axis_declaration_hash: str
    declared_kkt_problem_hash: str
    kkt_problem_id: str
    kkt_source_geometry_hash: str
    campaign_contract_hash: str
    campaign_binding_hash: str
    model_identity_hash: str
    query_id: str
    query_contract_hash: str
    preprocessing_hash: str
    sampling_law_hash: str
    raw_logit_stream_source_hash: str
    current_mask_hash: str
    corrected_moments_state_hash: str
    corrected_path_cost_hash: str
    point_raw_sha256: str
    physical_route_axis_hash: str
    ordered_route_axis_hash: str
    route_count: int
    proposal_generation: int
    sampled_look_index: int
    point_byte_count: int
    binding_scan_algorithm: str = V1_KKT_CORRECTED_WORK_BINDING_SCAN_ALGORITHM
    binding_memoryview_span_bytes_upper_bound: int = _MAX_MEMORYVIEW_SPAN_BYTES
    whole_axis_numeric_scratch_bytes: int = 0
    full_axis_binding_scan_complete: bool = True
    exact_bitwise_point_equality_verified: bool = True
    immutable_little_endian_float64_verified: bool = True
    cpu_binding_scan_distinct_from_selection: bool = True
    route_selection_performed: bool = False
    route_ordering_performed: bool = False
    corrected_work_identity_authority: bool = True
    confidence_interval_authority: bool = False
    f_certification_authority: bool = False
    q_certification_authority: bool = False
    r_certification_authority: bool = False
    literal_j_operator_authority: bool = False
    face_stability_authority: bool = False
    c_stream_authority: bool = False
    k_calibration_authority: bool = False
    o_oracle_authority: bool = False
    bce_oracle_authority: bool = False
    mask_commit_authority: bool = False
    production_execution_authority: bool = False
    terminal_authority: bool = False
    pareto_claim_authority: bool = False
    corrected_work_semantics: str = V1_KKT_CORRECTED_WORK_IDENTITY_SEMANTICS
    authority_scope: str = V1_KKT_CORRECTED_WORK_IDENTITY_AUTHORITY_SCOPE
    schema: str = V1_KKT_CORRECTED_WORK_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "confidence_box_external_descriptor_hash",
            "confidence_box_external_commit_hash",
            "confidence_box_external_inspection_receipt_hash",
            "confidence_box_receipt_hash",
            "confidence_box_statistics_state_hash",
            "confidence_box_design_hash",
            "trajectory_alpha_ledger_hash",
            "sampled_geometry_source_hash",
            "sampled_geometry_admission_receipt_hash",
            "sampled_geometry_assembly_receipt_hash",
            "sampled_snapshot_certificate_hash",
            "sampled_snapshot_numerics_hash",
            "proposal_position_certificate_hash",
            "physical_route_axis_declaration_hash",
            "declared_kkt_problem_hash",
            "kkt_problem_id",
            "kkt_source_geometry_hash",
            "campaign_contract_hash",
            "campaign_binding_hash",
            "model_identity_hash",
            "query_id",
            "query_contract_hash",
            "preprocessing_hash",
            "sampling_law_hash",
            "raw_logit_stream_source_hash",
            "current_mask_hash",
            "corrected_moments_state_hash",
            "corrected_path_cost_hash",
            "point_raw_sha256",
            "physical_route_axis_hash",
            "ordered_route_axis_hash",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be one nonempty string")
        for name in (
            "route_count",
            "sampled_look_index",
            "point_byte_count",
            "binding_memoryview_span_bytes_upper_bound",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be one exact positive integer")
        if type(self.proposal_generation) is not int or self.proposal_generation < 0:
            raise ValueError("proposal_generation must be one exact nonnegative integer")
        if type(self.whole_axis_numeric_scratch_bytes) is not int:
            raise TypeError("whole_axis_numeric_scratch_bytes must be an exact integer")
        if self.point_byte_count != self.route_count * _LE_F8.itemsize:
            raise ValueError("point byte count differs from the exact <f8 route axis")
        if (
            self.binding_scan_algorithm
            != V1_KKT_CORRECTED_WORK_BINDING_SCAN_ALGORITHM
            or self.binding_memoryview_span_bytes_upper_bound
            != _MAX_MEMORYVIEW_SPAN_BYTES
            or self.whole_axis_numeric_scratch_bytes != 0
            or self.authority_scope
            != V1_KKT_CORRECTED_WORK_IDENTITY_AUTHORITY_SCOPE
            or self.corrected_work_semantics
            != V1_KKT_CORRECTED_WORK_IDENTITY_SEMANTICS
            or self.schema != V1_KKT_CORRECTED_WORK_IDENTITY_SCHEMA
        ):
            raise ValueError("corrected-work identity receipt changed its fixed scope")
        required_true = (
            self.full_axis_binding_scan_complete,
            self.exact_bitwise_point_equality_verified,
            self.immutable_little_endian_float64_verified,
            self.cpu_binding_scan_distinct_from_selection,
            self.corrected_work_identity_authority,
        )
        required_false = (
            self.route_selection_performed,
            self.route_ordering_performed,
            self.confidence_interval_authority,
            self.f_certification_authority,
            self.q_certification_authority,
            self.r_certification_authority,
            self.literal_j_operator_authority,
            self.face_stability_authority,
            self.c_stream_authority,
            self.k_calibration_authority,
            self.o_oracle_authority,
            self.bce_oracle_authority,
            self.mask_commit_authority,
            self.production_execution_authority,
            self.terminal_authority,
            self.pareto_claim_authority,
        )
        if any(type(value) is not bool for value in (*required_true, *required_false)):
            raise TypeError("corrected-work identity authority fields must be exact booleans")
        if not all(required_true) or any(required_false):
            raise PermissionError(
                "corrected-work identity receipt crossed its authority boundary"
            )

    @property
    def receipt_hash(self) -> str:
        return content_hash(self, prefix=_RECEIPT_PREFIX)


def bind_v1_kkt_corrected_work_identity(
    *,
    confidence_box_view: V1ConfidenceBoxExternalView,
    sampled_geometry_source: V1SampledPreFreezeGeometrySource,
    selector_numerics: V1SampledSnapshotSelectorNumerics,
    proposal_position_certificate: V1SampledProposalPositionCertificate,
    problem: V1RelaxedKKTProblem,
    scan_chunk_elements: int = _DEFAULT_SCAN_CHUNK_ELEMENTS,
) -> V1KKTCorrectedWorkIdentityReceipt:
    """Bind one exact sampled A-point to KKT without granting decision authority.

    ``scan_chunk_elements`` is an execution detail and is intentionally absent
    from the durable receipt.  Any admitted chunk size therefore produces the
    same receipt while the fixed maximum keeps the CPU scan memory-bounded at
    route scale.  This is an identity barrier, not a CPU selector fallback.
    """

    exact_types = (
        (
            confidence_box_view,
            V1ConfidenceBoxExternalView,
            "confidence_box_view",
        ),
        (
            sampled_geometry_source,
            V1SampledPreFreezeGeometrySource,
            "sampled_geometry_source",
        ),
        (
            selector_numerics,
            V1SampledSnapshotSelectorNumerics,
            "selector_numerics",
        ),
        (
            proposal_position_certificate,
            V1SampledProposalPositionCertificate,
            "proposal_position_certificate",
        ),
        (problem, V1RelaxedKKTProblem, "problem"),
    )
    for value, expected, name in exact_types:
        if type(value) is not expected:
            _refuse(
                V1KKTCorrectedWorkIdentityRefusalCode.INVALID_TYPE,
                f"{name} requires the exact {expected.__name__} type",
            )
    if (
        isinstance(scan_chunk_elements, bool)
        or not isinstance(scan_chunk_elements, Integral)
        or int(scan_chunk_elements) <= 0
        or int(scan_chunk_elements) > _MAX_SCAN_CHUNK_ELEMENTS
    ):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.INVALID_SCAN_DESIGN,
            "scan_chunk_elements must be a positive integer within the fixed 64 MiB cap",
        )
    chunk_elements = int(scan_chunk_elements)
    if problem.corrected_work_envelope_id != V1_KKT_CORRECTED_WORK_IDENTITY_SEMANTICS:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.SOURCE_CHAIN_MISMATCH,
            "KKT corrected-work identity can bind only the singleton triangle "
            "work branch; signed action work requires a signed-work receipt",
        )

    payload, confidence_point, inspection_hash = _validated_external_payload(
        confidence_box_view
    )
    source = sampled_geometry_source
    admission = source.admission
    admission_receipt = admission.receipt
    assembly_receipt = source.receipt
    moments = admission.moments
    numerics = selector_numerics
    certificate = numerics.certificate
    position = proposal_position_certificate
    try:
        moments_hash = moments.assert_integrity(
            admission_receipt.corrected_moments_state_hash
        )
        admission_receipt.__post_init__()
        assembly_receipt.__post_init__()
        certificate.__post_init__()
        position.__post_init__()
    except Exception as error:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.SOURCE_CHAIN_MISMATCH,
            f"sampled source/certificate graph does not replay: {error}",
        )

    source_pairs = (
        (
            "assembly admission receipt",
            assembly_receipt.admission_receipt_hash,
            admission_receipt.receipt_hash,
        ),
        (
            "assembly admission",
            assembly_receipt.admission_hash,
            admission.admission_hash,
        ),
        (
            "assembly campaign",
            assembly_receipt.campaign_contract_hash,
            admission_receipt.campaign_contract_hash,
        ),
        (
            "assembly binding",
            assembly_receipt.campaign_binding_hash,
            admission_receipt.campaign_binding_hash,
        ),
        (
            "assembly mask",
            assembly_receipt.current_mask_hash,
            admission_receipt.current_mask_hash,
        ),
        (
            "assembly physical axis",
            assembly_receipt.physical_route_axis_hash,
            admission_receipt.physical_route_axis_hash,
        ),
        (
            "assembly ordered axis",
            assembly_receipt.ordered_route_axis_hash,
            admission_receipt.ordered_route_axis_hash,
        ),
        (
            "assembly generation",
            assembly_receipt.proposal_generation,
            admission_receipt.proposal_generation,
        ),
        (
            "assembly raw stream",
            assembly_receipt.raw_logit_stream_source_hash,
            admission_receipt.raw_logit_stream_source_hash,
        ),
        (
            "assembly sample weights",
            assembly_receipt.sample_weight_hash,
            admission_receipt.sample_weight_hash,
        ),
        (
            "assembly corrected moments",
            assembly_receipt.corrected_moments_state_hash,
            moments_hash,
        ),
        (
            "assembly FD design",
            assembly_receipt.fd_design_hash,
            admission_receipt.fd_design_hash,
        ),
        (
            "assembly FD snapshot",
            assembly_receipt.fd_snapshot_hash,
            admission_receipt.fd_snapshot_hash,
        ),
        (
            "assembly geometry",
            assembly_receipt.geometry_state_hash,
            source.geometry_state.content_hash,
        ),
        (
            "assembly geometry result",
            assembly_receipt.geometry_result_hash,
            source.geometry_result.result_hash,
        ),
        (
            "certificate source",
            certificate.sampled_geometry_source_hash,
            source.source_hash,
        ),
        (
            "certificate admission",
            certificate.admission_receipt_hash,
            admission_receipt.receipt_hash,
        ),
        (
            "certificate assembly",
            certificate.assembly_receipt_hash,
            assembly_receipt.receipt_hash,
        ),
        (
            "certificate campaign",
            certificate.campaign_contract_hash,
            admission_receipt.campaign_contract_hash,
        ),
        (
            "certificate binding",
            certificate.campaign_binding_hash,
            admission_receipt.campaign_binding_hash,
        ),
        (
            "certificate model",
            certificate.model_identity_hash,
            admission_receipt.model_identity_hash,
        ),
        (
            "certificate query ID",
            certificate.query_id,
            admission.binding.contract.query_id,
        ),
        (
            "certificate query",
            certificate.query_contract_hash,
            admission_receipt.query_contract_hash,
        ),
        (
            "certificate preprocessing",
            certificate.preprocessing_hash,
            admission_receipt.preprocessing_hash,
        ),
        (
            "certificate sampling",
            certificate.sampling_law_hash,
            admission_receipt.sampling_law_hash,
        ),
        (
            "certificate mask",
            certificate.current_mask_hash,
            admission_receipt.current_mask_hash,
        ),
        (
            "certificate generation",
            certificate.proposal_generation,
            admission_receipt.proposal_generation,
        ),
        (
            "certificate physical axis",
            certificate.physical_route_axis_hash,
            admission_receipt.physical_route_axis_hash,
        ),
        (
            "certificate ordered axis",
            certificate.ordered_route_axis_hash,
            admission_receipt.ordered_route_axis_hash,
        ),
        (
            "certificate route count",
            certificate.route_count,
            admission_receipt.route_count,
        ),
        (
            "certificate look",
            certificate.sampled_look_index,
            admission_receipt.sampled_look_index,
        ),
        (
            "certificate calibration",
            certificate.calibration_identity,
            admission_receipt.calibration_identity,
        ),
        (
            "certificate raw stream",
            certificate.raw_logit_stream_source_hash,
            admission_receipt.raw_logit_stream_source_hash,
        ),
        (
            "certificate raw coordinate",
            certificate.raw_logit_coordinate,
            admission_receipt.raw_logit_coordinate,
        ),
        (
            "certificate context mixture",
            certificate.context_mixture_hash,
            admission_receipt.context_mixture_hash,
        ),
        (
            "certificate sample weights",
            certificate.sample_weight_hash,
            admission_receipt.sample_weight_hash,
        ),
        (
            "certificate measurements",
            certificate.source_measurement_axis_hash,
            admission_receipt.source_measurement_axis_hash,
        ),
        (
            "certificate corrected moments",
            certificate.corrected_moments_state_hash,
            moments_hash,
        ),
        (
            "certificate FD design",
            certificate.fd_design_hash,
            admission_receipt.fd_design_hash,
        ),
        (
            "certificate FD snapshot",
            certificate.fd_snapshot_hash,
            admission_receipt.fd_snapshot_hash,
        ),
        (
            "certificate FD numerics",
            certificate.snapshot_numerical_backend_hash,
            admission_receipt.snapshot_numerical_backend_hash,
        ),
        (
            "certificate geometry",
            certificate.geometry_state_hash,
            assembly_receipt.geometry_state_hash,
        ),
        (
            "certificate geometry result",
            certificate.geometry_result_hash,
            assembly_receipt.geometry_result_hash,
        ),
        (
            "certificate numerical assembly",
            certificate.numerical_assembly_receipt_hash,
            assembly_receipt.numerical_assembly_receipt_hash,
        ),
        (
            "certificate derivation absence",
            certificate.raw_row_derivation_absence_hash,
            assembly_receipt.raw_row_derivation_absence_hash,
        ),
        (
            "position selector certificate",
            position.sampled_snapshot_certificate_hash,
            certificate.certificate_hash,
        ),
        (
            "position source",
            position.sampled_geometry_source_hash,
            source.source_hash,
        ),
        (
            "position mask",
            position.current_mask_hash,
            admission_receipt.current_mask_hash,
        ),
        (
            "position physical axis",
            position.physical_route_axis_hash,
            admission_receipt.physical_route_axis_hash,
        ),
        (
            "position ordered axis",
            position.ordered_route_axis_hash,
            admission_receipt.ordered_route_axis_hash,
        ),
        (
            "position route-ID axis",
            position.route_id_axis_hash,
            admission_receipt.ordered_route_axis_hash,
        ),
        (
            "position route count",
            position.route_count,
            admission_receipt.route_count,
        ),
    )
    _require_pairs(
        source_pairs,
        code=V1KKTCorrectedWorkIdentityRefusalCode.SOURCE_CHAIN_MISMATCH,
    )

    confidence_coordinates = (
        (
            "confidence state",
            _identity(payload.get("state_hash"), name="confidence state"),
            moments_hash,
        ),
        (
            "confidence campaign",
            _identity(
                payload.get("campaign_contract_hash"), name="confidence campaign"
            ),
            admission_receipt.campaign_contract_hash,
        ),
        (
            "confidence model",
            _identity(payload.get("model_identity_hash"), name="confidence model"),
            admission_receipt.model_identity_hash,
        ),
        (
            "confidence query",
            _identity(payload.get("query_contract_hash"), name="confidence query"),
            admission_receipt.query_contract_hash,
        ),
        (
            "confidence preprocessing",
            _identity(
                payload.get("preprocessing_hash"),
                name="confidence preprocessing",
            ),
            admission_receipt.preprocessing_hash,
        ),
        (
            "confidence mask",
            _identity(payload.get("current_mask_hash"), name="confidence mask"),
            admission_receipt.current_mask_hash,
        ),
        (
            "confidence raw stream",
            _identity(
                payload.get("raw_logit_stream_source_hash"),
                name="confidence raw stream",
            ),
            admission_receipt.raw_logit_stream_source_hash,
        ),
        (
            "confidence sampling",
            _identity(
                payload.get("sampling_law_hash"), name="confidence sampling"
            ),
            admission_receipt.sampling_law_hash,
        ),
        (
            "confidence generation",
            _exact_int(
                payload.get("proposal_generation"), name="confidence generation"
            ),
            admission_receipt.proposal_generation,
        ),
        (
            "confidence route count",
            _exact_int(
                payload.get("route_count"),
                name="confidence route count",
                positive=True,
            ),
            admission_receipt.route_count,
        ),
        (
            "confidence ordered axis",
            _identity(
                payload.get("route_axis_hash"), name="confidence ordered axis"
            ),
            admission_receipt.ordered_route_axis_hash,
        ),
        (
            "confidence strata",
            _identity_pair(
                payload.get("stratum_names"), name="confidence strata"
            ),
            admission_receipt.stratum_names,
        ),
        (
            "confidence counts",
            _int_pair(payload.get("stratum_counts"), name="confidence counts"),
            admission_receipt.observed_stratum_counts,
        ),
        (
            "confidence populations",
            _int_pair(
                payload.get("population_stratum_counts"),
                name="confidence populations",
                positive=True,
            ),
            admission_receipt.population_stratum_counts,
        ),
        (
            "confidence orders",
            _identity_pair(
                payload.get("sealed_population_order_hashes"),
                name="confidence orders",
            ),
            admission_receipt.sealed_population_order_hashes,
        ),
        (
            "confidence contexts",
            _identity_ledger_pair(
                payload.get("observed_context_ids_by_stratum"),
                name="confidence contexts",
            ),
            moments.observed_context_ids_by_stratum,
        ),
        (
            "confidence draws",
            _identity_ledger_pair(
                payload.get("observed_draw_ids_by_stratum"),
                name="confidence draws",
            ),
            moments.observed_draw_ids_by_stratum,
        ),
        (
            "confidence weights",
            _class_balanced_weights(
                payload.get("class_balanced_stratum_weights")
            ),
            admission_receipt.stratum_weights,
        ),
        (
            "confidence look",
            _exact_int(
                payload.get("look_index"), name="confidence look", positive=True
            ),
            admission_receipt.sampled_look_index,
        ),
    )
    _require_pairs(
        confidence_coordinates,
        code=V1KKTCorrectedWorkIdentityRefusalCode.STALE_COORDINATE,
    )

    if certificate.physical_route_axis_hash == certificate.ordered_route_axis_hash:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.AXIS_MISMATCH,
            "sampled physical and ordered axes collapsed",
        )
    axis_pairs = (
        ("KKT physical route axis", problem.route_axis_hash, position.physical_route_axis_hash),
        ("KKT route count", problem.route_count, position.route_count),
    )
    _require_pairs(
        axis_pairs,
        code=V1KKTCorrectedWorkIdentityRefusalCode.AXIS_MISMATCH,
    )

    route_count = position.route_count
    confidence_point = _validated_point_array(
        confidence_point,
        route_count=route_count,
        name="confidence-box point",
    )
    sampled_point = _validated_point_array(
        numerics.corrected_path_cost,
        route_count=route_count,
        name="sampled selector corrected path",
    )
    problem_point = _validated_point_array(
        problem.corrected_work,
        route_count=route_count,
        name="KKT corrected work",
    )
    raw_hashes, sampled_hashes = _scan_exact_points(
        (confidence_point, sampled_point, problem_point),
        chunk_elements=chunk_elements,
    )
    if len(set(raw_hashes)) != 1 or len(set(sampled_hashes)) != 1:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.POINT_HASH_MISMATCH,
            "corrected-work byte or sampled-path hashes differ",
        )

    external_identities = {
        identity.logical_name: identity
        for identity in confidence_box_view.descriptor.external_state.arrays
    }
    point_identity = external_identities[_POINT_NAME]
    if (
        point_identity.dtype != _LE_F8.str
        or point_identity.shape != (route_count,)
        or point_identity.byte_count != route_count * _LE_F8.itemsize
        or point_identity.sha256 != raw_hashes[0]
        or point_identity.allow_nonfinite
    ):
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.POINT_HASH_MISMATCH,
            "confidence point differs from its immutable external identity",
        )
    if sampled_hashes[1] != certificate.corrected_path_cost_hash:
        _refuse(
            V1KKTCorrectedWorkIdentityRefusalCode.POINT_HASH_MISMATCH,
            "sampled corrected path differs from its factory-issued certificate",
        )

    descriptor = confidence_box_view.descriptor
    return V1KKTCorrectedWorkIdentityReceipt(
        confidence_box_external_descriptor_hash=descriptor.descriptor_hash,
        confidence_box_external_commit_hash=descriptor.commit_hash,
        confidence_box_external_inspection_receipt_hash=inspection_hash,
        confidence_box_receipt_hash=descriptor.receipt_hash,
        confidence_box_statistics_state_hash=_identity(
            payload.get("state_hash"), name="confidence state"
        ),
        confidence_box_design_hash=_identity(
            payload.get("design_hash"), name="confidence design"
        ),
        trajectory_alpha_ledger_hash=_identity(
            payload.get("trajectory_alpha_ledger_hash"),
            name="confidence trajectory alpha ledger",
        ),
        sampled_geometry_source_hash=source.source_hash,
        sampled_geometry_admission_receipt_hash=admission_receipt.receipt_hash,
        sampled_geometry_assembly_receipt_hash=assembly_receipt.receipt_hash,
        sampled_snapshot_certificate_hash=certificate.certificate_hash,
        sampled_snapshot_numerics_hash=numerics.numerics_hash,
        proposal_position_certificate_hash=position.certificate_hash,
        physical_route_axis_declaration_hash=(
            position.physical_route_axis_declaration_hash
        ),
        declared_kkt_problem_hash=problem.problem_hash,
        kkt_problem_id=problem.problem_id,
        kkt_source_geometry_hash=problem.source_geometry_hash,
        campaign_contract_hash=admission_receipt.campaign_contract_hash,
        campaign_binding_hash=admission_receipt.campaign_binding_hash,
        model_identity_hash=admission_receipt.model_identity_hash,
        query_id=certificate.query_id,
        query_contract_hash=admission_receipt.query_contract_hash,
        preprocessing_hash=admission_receipt.preprocessing_hash,
        sampling_law_hash=admission_receipt.sampling_law_hash,
        raw_logit_stream_source_hash=(
            admission_receipt.raw_logit_stream_source_hash
        ),
        current_mask_hash=admission_receipt.current_mask_hash,
        corrected_moments_state_hash=moments_hash,
        corrected_path_cost_hash=sampled_hashes[0],
        point_raw_sha256=raw_hashes[0],
        physical_route_axis_hash=position.physical_route_axis_hash,
        ordered_route_axis_hash=position.ordered_route_axis_hash,
        route_count=route_count,
        proposal_generation=admission_receipt.proposal_generation,
        sampled_look_index=admission_receipt.sampled_look_index,
        point_byte_count=route_count * _LE_F8.itemsize,
        corrected_work_semantics=V1_KKT_CORRECTED_WORK_IDENTITY_SEMANTICS,
    )


__all__ = [
    "V1_KKT_CORRECTED_WORK_BINDING_SCAN_ALGORITHM",
    "V1_KKT_CORRECTED_WORK_IDENTITY_AUTHORITY_SCOPE",
    "V1_KKT_CORRECTED_WORK_IDENTITY_SEMANTICS",
    "V1_KKT_CORRECTED_WORK_IDENTITY_SCHEMA",
    "V1KKTCorrectedWorkIdentityReceipt",
    "V1KKTCorrectedWorkIdentityRefusal",
    "V1KKTCorrectedWorkIdentityRefusalCode",
    "bind_v1_kkt_corrected_work_identity",
]
