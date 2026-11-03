"""Pure, authority-free compiler for the bounded scalar V2-R transport canary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
import hashlib
import itertools
import json
import math
import struct
from typing import ClassVar, Literal

import numpy as np


COST_KIND = "eligible_operator_edge_count_v1"
MAXIMUM_NONTERMINAL_BRANCHES = 8
MAXIMUM_CANONICAL_PHYSICAL_STATES = 256
BINARY64_POLICY_ID = "ieee754_binary64_rne_left_fold_v1"
SEMANTIC_ROLE_ONTOLOGY = ("supportive", "opposing", "context_bipolar", "negligible", "unresolved")
SUFFIX_OWNER_HASH_FIELDS = (
    "scalar_oracle_hash", "scalar_profile_hash", "calibration_fit_hash", "calibration_selection_face_hash",
    "positive_scale_face_hash", "identifiability_quotient_hash", "gauge_range_hash", "portable_reduction_receipt_hash",
)


class TransportCompileError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


def _fail(code: str, message: str) -> None:
    raise TransportCompileError(code, message)


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        _fail(code, message)


def _is_sha(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_sha(value: str, code: str) -> None:
    _require(_is_sha(value), code, "expected lowercase SHA-256 digest")


def _stable_digest(payload: object) -> str:
    encoded = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _receipt_envelope(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, float):
        _require(math.isfinite(value), "nonfinite_or_overflow", "nonfinite hash payload")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported canonical primitive {type(value)!r}")


class _Receipt:
    schema_name: ClassVar[str] = "documentation_only_not_hash_authority"
    schema_version: ClassVar[str] = "documentation_only_not_hash_authority"

    @property
    def content_hash(self) -> str:
        return _receipt_digest(self)


@dataclass(frozen=True)
class CanonicalRouteCoordinateReceipt(_Receipt):
    architecture_id: str; topology_hash: str
    layer_id: str; route_kind: str
    source_coordinate: tuple[int, ...]; target_coordinate: tuple[int, ...]
    parameter_coordinate: tuple[int, ...] = ()

    @property
    def route_id(self) -> str:
        _require_sha(self.topology_hash, "topology_route_schema_mismatch")
        _require(bool(self.architecture_id and self.layer_id and self.route_kind), "topology_route_schema_mismatch", "route coordinate identity is incomplete")
        coordinates = (*self.source_coordinate, *self.target_coordinate, *self.parameter_coordinate)
        _require(bool(self.source_coordinate and self.target_coordinate) and all(int(value) >= 0 for value in coordinates), "topology_route_schema_mismatch", "route endpoint coordinates are invalid")
        return f"route:v1:{_stable_digest(self.route_payload)}"

    @property
    def route_payload(self) -> dict[str, object]:
        return {
            "schema_version": "route_coordinate_v1",
            "architecture_id": self.architecture_id,
            "topology_hash": self.topology_hash,
            "layer_id": self.layer_id,
            "route_kind": self.route_kind,
            "source_coordinate": list(self.source_coordinate),
            "target_coordinate": list(self.target_coordinate),
            "parameter_coordinate": list(self.parameter_coordinate),
        }


@dataclass(frozen=True)
class EligibleRouteBlockReceipt(_Receipt):
    architecture_id: str; topology_hash: str
    layer_id: str; route_kind: str
    shape: tuple[int, ...]; eligible_offset: int
    full_block_index: int; production_binding_hash: str; annotation_hash: str
    first_coordinate: CanonicalRouteCoordinateReceipt; last_coordinate: CanonicalRouteCoordinateReceipt

    @property
    def route_count(self) -> int:
        return math.prod(self.shape)

    @property
    def stop(self) -> int:
        return self.eligible_offset + self.route_count


@dataclass(frozen=True)
class EligibleTopologyAnnotationReceipt(_Receipt):
    full_block_index: int; topology_rank: int
    production_binding_hash: str; annotation_hash: str
    layer_id: str; stage_id: str; block_id: str; operator_id: str


@dataclass(frozen=True)
class TopologyAnnotationReceipt(_Receipt):
    architecture_id: str; topology_hash: str
    full_binding_axis_hash: str; full_route_schema_hash: str; eligible_route_axis_hash: str
    full_to_eligible_block_index: tuple[int, ...]
    fixed_measurement_full_block_indices: tuple[int, ...]
    eligible_route_blocks: tuple[EligibleRouteBlockReceipt, ...]
    eligible_annotations: tuple[EligibleTopologyAnnotationReceipt, ...]
    full_binding_hashes: tuple[str, ...]; full_annotation_hashes: tuple[str, ...]
    fixed_measurement_exclusion_hash: str

    @property
    def eligible_route_count(self) -> int:
        return self.eligible_route_blocks[-1].stop if self.eligible_route_blocks else 0

    @property
    def eligible_route_blocks_hash(self) -> str:
        return _stable_digest({"schema_name": "mechanistic_transport_eligible_route_blocks_v1", "blocks": [_receipt_envelope(block) for block in self.eligible_route_blocks]})

    def validate(self) -> None:
        _validate_topology(self)


class RoleAssignmentKind(StrEnum):
    DISJOINT_EXHAUSTIVE = "disjoint_exhaustive"
    OVERLAPPING_EXHAUSTIVE = "overlapping_exhaustive"
    OVERLAPPING_PARTIAL = "overlapping_partial"


@dataclass(frozen=True)
class BlockPackedCanaryRoleOverlay(_Receipt):
    eligible_route_axis_hash: str; eligible_route_blocks_hash: str
    semantic_role_ids: tuple[str, ...]
    assignment_kind: RoleAssignmentKind
    packed_role_incidence_bytes: bytes
    packed_row_count: int; packed_byte_width: int
    bitorder: Literal["little"]
    measurement_artifact_hash: str; role_spec_hash: str; presentation_hash: str

    @classmethod
    def from_boolean(
        cls,
        *,
        eligible_route_axis_hash: str,
        eligible_route_blocks_hash: str,
        semantic_role_ids: Sequence[str],
        assignment_kind: RoleAssignmentKind,
        incidence: np.ndarray,
        measurement_artifact_hash: str,
        role_spec_hash: str,
        presentation_hash: str,
    ) -> BlockPackedCanaryRoleOverlay:
        raw = np.asarray(incidence)
        if raw.dtype != np.dtype(bool) or raw.ndim != 2:
            raise TypeError("role incidence must be a two-dimensional Boolean array")
        role_ids = tuple(str(value) for value in semantic_role_ids)
        _require(len(role_ids) == raw.shape[0] and len(set(role_ids)) == len(role_ids) and set(role_ids) <= set(SEMANTIC_ROLE_ONTOLOGY), "role_incidence_identity_mismatch", "role IDs are invalid")
        order = tuple(sorted(range(len(role_ids)), key=lambda index: SEMANTIC_ROLE_ONTOLOGY.index(role_ids[index])))
        role_ids, raw = tuple(role_ids[index] for index in order), raw[np.asarray(order)]
        packed = np.packbits(raw, axis=1, bitorder="little")
        result = cls(
            str(eligible_route_axis_hash), str(eligible_route_blocks_hash), role_ids,
            RoleAssignmentKind(assignment_kind), packed.tobytes(order="C"),
            int(packed.shape[0]), int(packed.shape[1]), "little",
            str(measurement_artifact_hash), str(role_spec_hash), str(presentation_hash),
        )
        _validate_role_overlay(result, route_count=raw.shape[1])
        return result

    def unpack(self, *, route_count: int | None = None) -> np.ndarray:
        values = np.frombuffer(self.packed_role_incidence_bytes, dtype=np.uint8).copy()
        values = values.reshape(self.packed_row_count, self.packed_byte_width)
        count = self.packed_byte_width * 8 if route_count is None else int(route_count)
        return np.unpackbits(values, axis=1, count=count, bitorder="little").astype(bool, copy=True)

    def contains(self, role_index: int, eligible_flat_index: int) -> bool:
        role, route = int(role_index), int(eligible_flat_index)
        _require(0 <= role < self.packed_row_count and 0 <= route < self.packed_byte_width * 8, "role_incidence_identity_mismatch", "selected role bit is out of range")
        offset = role * self.packed_byte_width + route // 8
        return bool((self.packed_role_incidence_bytes[offset] >> (route % 8)) & 1)

    @property
    def semantic_hash(self) -> str:
        return _stable_digest({
            "schema_name": "mechanistic_transport_semantic_role_overlay_v1", "eligible_route_axis_hash": self.eligible_route_axis_hash,
            "eligible_route_blocks_hash": self.eligible_route_blocks_hash, "semantic_role_ids": self.semantic_role_ids,
            "assignment_kind": self.assignment_kind.value, "packed_role_incidence_bytes": self.packed_role_incidence_bytes.hex(),
            "packed_row_count": self.packed_row_count, "packed_byte_width": self.packed_byte_width, "bitorder": self.bitorder,
            "measurement_artifact_hash": self.measurement_artifact_hash, "role_spec_hash": self.role_spec_hash,
        })


@dataclass(frozen=True)
class TechnicalCanaryUnitAssignment(_Receipt):
    grouping_spec_hash: str; geometry_search_unit_ids: tuple[str, ...]
    calibration_fitting_unit_ids: tuple[str, ...]; technical_adjudication_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("geometry_search_unit_ids", "calibration_fitting_unit_ids", "technical_adjudication_unit_ids"):
            object.__setattr__(self, name, tuple(sorted(str(value) for value in getattr(self, name))))
        _validate_units(self)

    @property
    def grouping_hash(self) -> str:
        _validate_units(self)
        return self.content_hash


@dataclass(frozen=True)
class CanaryCoverageFrameSpec(_Receipt):
    fixture_id: str; fixture_kind: Literal["residual_cnn", "vgg11_mixed_route"]


@dataclass(frozen=True)
class CanarySelectedAtom(_Receipt):
    physical_route_alias: str; route_id: str; eligible_flat_index: int
    layer_id: str; route_kind: str
    stage_id: str; block_id: str; operator_id: str


@dataclass(frozen=True)
class CanaryBranchSkeleton(_Receipt):
    branch_id: str; operator_id: str; semantic_role_id: str
    member_physical_route_aliases: tuple[str, ...]


@dataclass(frozen=True)
class CanaryCoverageSkeleton(_Receipt):
    frame_spec_hash: str; topology_receipt_hash: str
    semantic_role_overlay_hash: str; unit_assignment_hash: str
    selected_atoms: tuple[CanarySelectedAtom, ...]; branches: tuple[CanaryBranchSkeleton, ...]
    empty_intersection_hash: str; exclusion_hash: str


@dataclass(frozen=True)
class StructuralCompletionEntry(_Receipt):
    requested_physical_route_aliases: tuple[str, ...]
    completed_removed_eligible_indices: tuple[int, ...]
    completed_removed_physical_gate_ids: tuple[str, ...]; completion_lineage_hash: str


@dataclass(frozen=True)
class StructuralCompletionCatalog(_Receipt):
    catalog_kind: Literal["selected_singletons", "logical_suffix_universe"]
    topology_receipt_hash: str; eligible_route_axis_hash: str
    completion_policy_hash: str; dense_physical_state_hash: str
    entries: tuple[StructuralCompletionEntry, ...]
    topology_completion_certificate_hash: str


@dataclass(frozen=True)
class CompletedSingletonDisplacementReceipt(_Receipt):
    coverage_skeleton_hash: str; branch_skeleton_hashes: tuple[str, ...]
    selected_physical_route_alias: str; selected_route_id: str; eligible_flat_index: int
    topology_receipt_hash: str; eligible_route_axis_hash: str
    dense_physical_state_hash: str; completion_policy_hash: str
    completed_removed_eligible_indices: tuple[int, ...]
    completed_removed_physical_gate_ids: tuple[str, ...]
    completed_retained_state_hash: str; sparse_hard_projector_hash: str
    saved_cost: int; physical_cost_profile_hash: str


@dataclass(frozen=True)
class CompletedSingletonDisplacementBank(_Receipt):
    coverage_skeleton_hash: str; receipts: tuple[CompletedSingletonDisplacementReceipt, ...]


@dataclass(frozen=True)
class SuffixOrderEvidence(_Receipt):
    coverage_skeleton_hash: str; branch_skeleton_hash: str
    topology_receipt_hash: str; semantic_role_overlay_hash: str
    completed_singleton_bank_hash: str; branch_id: str
    ordered_member_aliases: tuple[str, ...]
    block_ids: tuple[str, str, str, str]
    block_effective_energy: tuple[tuple[float, ...], ...]
    technical_unit_assignment_hash: str; scalar_oracle_hash: str; scalar_profile_hash: str
    calibration_fit_hash: str; calibration_selection_face_hash: str; positive_scale_face_hash: str
    identifiability_quotient_hash: str; gauge_range_hash: str; portable_reduction_receipt_hash: str


@dataclass(frozen=True)
class SuffixOrderReceipt(_Receipt):
    evidence_hash: str; coverage_skeleton_hash: str
    branch_skeleton_hash: str; completed_singleton_bank_hash: str
    ordered_member_aliases: tuple[str, ...]
    negative_tolerance: float; absolute_tolerance: float; relative_tolerance: float
    retention_energy: tuple[float, ...]; compiler_owned_saved_cost: tuple[int, ...]
    priority: tuple[float, ...]; branch_scale: float; tie_width: float
    priority_bin_uint64: tuple[int, ...]
    ordered_physical_route_aliases: tuple[str, ...]
    binary64_policy_id: str


@dataclass(frozen=True)
class CanaryLogicalUniverseSkeleton(_Receipt):
    coverage_skeleton_hash: str; completed_singleton_bank_hash: str
    suffix_order_receipt_hashes: tuple[str, ...]
    branch_chains: tuple[tuple[str, tuple[str, ...]], ...]
    logical_state_count: int


@dataclass(frozen=True)
class CanaryPhysicalAlias(_Receipt):
    physical_state_hash: str; logical_state_hashes: tuple[str, ...]; retained_cost: int
    completed_removed_eligible_indices: tuple[int, ...]
    completed_removed_physical_gate_ids: tuple[str, ...]
    sparse_projector_hash: str


@dataclass(frozen=True)
class TechnicalCanaryCompileReceipt(_Receipt):
    logical_universe_hash: str; topology_receipt_hash: str
    semantic_role_overlay_hash: str; completed_singleton_bank_hash: str
    canonical_physical_aliases: tuple[CanaryPhysicalAlias, ...]
    physical_alias_hash: str; physical_cost_hash: str; sparse_projector_bank_hash: str
    completion_policy_hash: str; unit_assignment_hash: str
    failure_code: str = ""


def _receipt_domain_for_exact_type(receipt_type: type[object]) -> tuple[str, str]:
    if receipt_type is CanonicalRouteCoordinateReceipt:
        return "mechanistic_transport_route_coordinate_receipt", "1"
    if receipt_type is EligibleRouteBlockReceipt:
        return "mechanistic_transport_eligible_route_block_receipt", "1"
    if receipt_type is EligibleTopologyAnnotationReceipt:
        return "mechanistic_transport_eligible_topology_annotation_receipt", "1"
    if receipt_type is TopologyAnnotationReceipt:
        return "mechanistic_transport_topology_annotation_receipt", "1"
    if receipt_type is BlockPackedCanaryRoleOverlay:
        return "mechanistic_transport_role_overlay_receipt", "1"
    if receipt_type is TechnicalCanaryUnitAssignment:
        return "mechanistic_transport_unit_assignment_receipt", "1"
    if receipt_type is CanaryCoverageFrameSpec:
        return "mechanistic_transport_coverage_frame_spec", "1"
    if receipt_type is CanarySelectedAtom:
        return "mechanistic_transport_selected_atom_receipt", "1"
    if receipt_type is CanaryBranchSkeleton:
        return "mechanistic_transport_branch_skeleton_receipt", "1"
    if receipt_type is CanaryCoverageSkeleton:
        return "mechanistic_transport_coverage_skeleton_receipt", "1"
    if receipt_type is StructuralCompletionEntry:
        return "mechanistic_transport_structural_completion_entry", "1"
    if receipt_type is StructuralCompletionCatalog:
        return "mechanistic_transport_structural_completion_catalog", "1"
    if receipt_type is CompletedSingletonDisplacementReceipt:
        return "mechanistic_transport_completed_singleton_receipt", "1"
    if receipt_type is CompletedSingletonDisplacementBank:
        return "mechanistic_transport_completed_singleton_bank", "1"
    if receipt_type is SuffixOrderEvidence:
        return "mechanistic_transport_suffix_order_evidence", "1"
    if receipt_type is SuffixOrderReceipt:
        return "mechanistic_transport_suffix_order_receipt", "1"
    if receipt_type is CanaryLogicalUniverseSkeleton:
        return "mechanistic_transport_logical_universe_receipt", "1"
    if receipt_type is CanaryPhysicalAlias:
        return "mechanistic_transport_physical_alias_receipt", "1"
    if receipt_type is TechnicalCanaryCompileReceipt:
        return "mechanistic_transport_compile_receipt", "1"
    raise TypeError("unregistered exact receipt type")


def _canonical_receipt_payload(receipt: object) -> dict[str, object]:
    _receipt_domain_for_exact_type(type(receipt))
    return {field.name: _canonical(getattr(receipt, field.name)) for field in fields(receipt)}


def _receipt_envelope(receipt: object) -> dict[str, object]:
    schema_name, schema_version = _receipt_domain_for_exact_type(type(receipt))
    return {"schema_name": schema_name, "schema_version": schema_version, "payload": _canonical_receipt_payload(receipt)}


def _receipt_digest(receipt: object) -> str:
    _receipt_domain_for_exact_type(type(receipt))
    return _stable_digest(_receipt_envelope(receipt))


def _digest_for_exact_receipt_type(receipt_type: type[object], primitive_payload: Mapping[str, object]) -> str:
    schema_name, schema_version = _receipt_domain_for_exact_type(receipt_type)
    return _stable_digest({"schema_name": schema_name, "schema_version": schema_version, "payload": dict(primitive_payload)})


def _route_axis_hash(blocks: Sequence[EligibleRouteBlockReceipt]) -> str:
    payload = []
    expected_offset = 0
    for block in blocks:
        _require(block.eligible_offset == expected_offset, "eligible_axis_projection_mismatch", "eligible offsets are not contiguous")
        expected_offset = block.stop
        payload.append({
            "schema_name": "mechanistic_block_route_axis_v1", "architecture_id": block.architecture_id,
            "topology_hash": block.topology_hash, "layer_id": block.layer_id, "route_kind": block.route_kind,
            "shape": list(block.shape), "offset": block.eligible_offset, "route_count": block.route_count,
            "flattening": "c_order_target_then_source",
        })
    return _stable_digest({"schema_name": "mechanistic_block_route_axis_v1", "blocks": payload})


def _expected_endpoint(block: EligibleRouteBlockReceipt, index: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if block.route_kind == "residual_stream_gate":
        return (index,), (0,)
    if len(block.shape) == 1:
        return (index,), (index,)
    return (index % block.shape[1],), (index // block.shape[1],)


def _validate_topology(topology: TopologyAnnotationReceipt) -> None:
    for value in (topology.topology_hash, topology.full_binding_axis_hash, topology.full_route_schema_hash, topology.eligible_route_axis_hash, topology.fixed_measurement_exclusion_hash):
        _require_sha(value, "topology_route_schema_mismatch")
    full_count = len(topology.full_binding_hashes)
    _require(full_count > 0 and len(topology.full_annotation_hashes) == full_count and len(topology.full_to_eligible_block_index) == full_count, "missing_topology_annotation", "full binding and annotation axes differ")
    _require(all(_is_sha(value) for value in (*topology.full_binding_hashes, *topology.full_annotation_hashes)) and len(set(topology.full_binding_hashes)) == full_count and len(set(topology.full_annotation_hashes)) == full_count, "annotation_binding_mismatch", "full binding or annotation identities are invalid")
    projection = topology.full_to_eligible_block_index
    _require(all(type(value) is int and value >= -1 for value in projection) and tuple(value for value in projection if value >= 0) == tuple(range(len(topology.eligible_route_blocks))), "eligible_axis_projection_mismatch", "full-to-eligible projection is not one-to-one")
    fixed = tuple(index for index, value in enumerate(topology.full_to_eligible_block_index) if value == -1)
    _require(fixed == topology.fixed_measurement_full_block_indices, "eligible_axis_projection_mismatch", "fixed measurement projection differs")
    _require(len(topology.eligible_route_blocks) == len(topology.eligible_annotations), "missing_topology_annotation", "eligible annotation count differs")
    ranks = tuple(item.topology_rank for item in topology.eligible_annotations)
    _require(all(rank >= 0 for rank in ranks) and ranks == tuple(sorted(set(ranks))), "missing_topology_annotation", "topology ranks are invalid")
    _require(len({block.layer_id for block in topology.eligible_route_blocks}) == len(topology.eligible_route_blocks), "annotation_binding_mismatch", "eligible layer IDs are not unique")
    for eligible_index, (block, annotation) in enumerate(zip(topology.eligible_route_blocks, topology.eligible_annotations, strict=True)):
        _require(
            0 <= block.full_block_index < full_count and topology.full_to_eligible_block_index[block.full_block_index] == eligible_index,
            "eligible_axis_projection_mismatch",
            "full-to-eligible projection differs",
        )
        _require(
            annotation.full_block_index == block.full_block_index
            and annotation.production_binding_hash == block.production_binding_hash
            and annotation.annotation_hash == block.annotation_hash
            and topology.full_binding_hashes[block.full_block_index] == block.production_binding_hash
            and topology.full_annotation_hashes[block.full_block_index] == block.annotation_hash
            and annotation.annotation_hash == _stable_digest({"schema_name": "mechanistic_v2_route_topology_annotation_v1", "layer_id": annotation.layer_id, "stage_id": annotation.stage_id, "block_id": annotation.block_id, "operator_id": annotation.operator_id, "topology_rank": annotation.topology_rank})
            and annotation.layer_id == block.layer_id,
            "annotation_binding_mismatch",
            "eligible annotation and route block differ",
        )
        _require(
            block.architecture_id == topology.architecture_id
            and block.topology_hash == topology.topology_hash
            and bool(block.shape)
            and len(block.shape) <= 2
            and all(value > 0 for value in block.shape),
            "topology_route_schema_mismatch",
            "eligible block is malformed",
        )
        for flat_index, coordinate in ((0, block.first_coordinate), (block.route_count - 1, block.last_coordinate)):
            source, target = _expected_endpoint(block, flat_index)
            _require(
                coordinate.architecture_id == block.architecture_id
                and coordinate.topology_hash == block.topology_hash
                and coordinate.layer_id == block.layer_id
                and coordinate.route_kind == block.route_kind
                and coordinate.source_coordinate == source
                and coordinate.target_coordinate == target,
                "topology_route_schema_mismatch",
                "block endpoint coordinate differs",
            )
    _require(_route_axis_hash(topology.eligible_route_blocks) == topology.eligible_route_axis_hash, "topology_route_schema_mismatch", "eligible route-axis hash differs")


def _validate_role_overlay(overlay: BlockPackedCanaryRoleOverlay, *, route_count: int) -> None:
    for value in (overlay.eligible_route_axis_hash, overlay.eligible_route_blocks_hash, overlay.measurement_artifact_hash, overlay.role_spec_hash, overlay.presentation_hash):
        _require_sha(value, "role_incidence_identity_mismatch")
    _require(overlay.bitorder == "little" and overlay.packed_row_count == len(overlay.semantic_role_ids) and overlay.packed_byte_width == (int(route_count) + 7) // 8 and len(overlay.packed_role_incidence_bytes) == overlay.packed_row_count * overlay.packed_byte_width, "role_incidence_identity_mismatch", "packed role axes differ")
    _require(bool(overlay.semantic_role_ids) and overlay.semantic_role_ids == tuple(role for role in SEMANTIC_ROLE_ONTOLOGY if role in overlay.semantic_role_ids), "role_incidence_identity_mismatch", "role IDs are not in canonical ontology order")
    raw = np.frombuffer(overlay.packed_role_incidence_bytes, dtype=np.uint8).reshape(overlay.packed_row_count, overlay.packed_byte_width)
    valid = np.full(overlay.packed_byte_width, np.uint8(0xFF), dtype=np.uint8)
    if route_count % 8:
        mask = ((1 << (route_count % 8)) - 1)
        valid[-1] = np.uint8(mask)
        _require(bool(np.all((raw[:, -1] & np.uint8(0xFF ^ mask)) == 0)), "role_incidence_identity_mismatch", "role padding bits are nonzero")
    covered = np.bitwise_or.reduce(raw, axis=0)
    if overlay.assignment_kind is RoleAssignmentKind.DISJOINT_EXHAUSTIVE:
        seen = np.zeros(overlay.packed_byte_width, dtype=np.uint8)
        for row in raw:
            _require(not bool(np.any(seen & row)), "role_assignment_kind_mismatch", "disjoint roles overlap")
            seen |= row
        _require(bool(np.array_equal(seen, valid)), "role_assignment_kind_mismatch", "disjoint roles must cover once")
    elif overlay.assignment_kind is RoleAssignmentKind.OVERLAPPING_EXHAUSTIVE:
        _require(bool(np.array_equal(covered, valid)), "role_assignment_kind_mismatch", "overlapping roles must cover")
    else:
        _require("unresolved" in overlay.semantic_role_ids, "role_assignment_kind_mismatch", "partial roles require unresolved")
        unresolved = overlay.semantic_role_ids.index("unresolved")
        others = np.bitwise_or.reduce(np.delete(raw, unresolved, axis=0), axis=0) if len(raw) > 1 else np.zeros(overlay.packed_byte_width, dtype=np.uint8)
        expected = np.bitwise_not(others) & valid
        _require(bool(np.array_equal(raw[unresolved], expected)), "role_assignment_kind_mismatch", "unresolved is not the complement")


def _validate_units(units: TechnicalCanaryUnitAssignment) -> None:
    _require_sha(units.grouping_spec_hash, "grouped_unit_overlap")
    axes = (
        units.geometry_search_unit_ids,
        units.calibration_fitting_unit_ids,
        units.technical_adjudication_unit_ids,
    )
    _require(all(axis and axis == tuple(sorted(set(axis))) for axis in axes), "grouped_unit_overlap", "technical unit axes must be nonempty, unique, and canonical")
    sets = tuple(set(axis) for axis in axes)
    _require(not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "grouped_unit_overlap", "technical units overlap")


def physical_route_alias(eligible_route_axis_hash: str, eligible_flat_index: int) -> str:
    _require_sha(eligible_route_axis_hash, "eligible_axis_projection_mismatch")
    _require(int(eligible_flat_index) >= 0, "eligible_axis_projection_mismatch", "eligible index is negative")
    return f"physical-route:v2:{eligible_route_axis_hash}:{int(eligible_flat_index):016d}"


def _coordinate_for_flat_index(block: EligibleRouteBlockReceipt, flat_index: int) -> CanonicalRouteCoordinateReceipt:
    source, target = _expected_endpoint(block, flat_index)
    return CanonicalRouteCoordinateReceipt(
        architecture_id=block.architecture_id,
        topology_hash=block.topology_hash,
        layer_id=block.layer_id,
        route_kind=block.route_kind,
        source_coordinate=source,
        target_coordinate=target,
    )


def compile_transport_canary_skeleton(
    *,
    topology: TopologyAnnotationReceipt,
    full_role_overlay: BlockPackedCanaryRoleOverlay,
    technical_units: TechnicalCanaryUnitAssignment,
    frame: CanaryCoverageFrameSpec,
) -> CanaryCoverageSkeleton:
    _validate_topology(topology)
    _validate_units(technical_units)
    _require(bool(frame.fixture_id) and frame.fixture_kind in {"residual_cnn", "vgg11_mixed_route"}, "missing_required_operator_family", "coverage frame is invalid")
    _validate_role_overlay(full_role_overlay, route_count=topology.eligible_route_count)
    _require(
        full_role_overlay.eligible_route_axis_hash == topology.eligible_route_axis_hash
        and full_role_overlay.eligible_route_blocks_hash == topology.eligible_route_blocks_hash,
        "role_incidence_identity_mismatch",
        "role overlay is bound to another route axis",
    )
    by_operator: dict[str, list[tuple[EligibleTopologyAnnotationReceipt, EligibleRouteBlockReceipt]]] = {}
    for annotation, block in zip(topology.eligible_annotations, topology.eligible_route_blocks, strict=True):
        by_operator.setdefault(annotation.operator_id, []).append((annotation, block))
    operator_order = tuple(
        sorted(by_operator, key=lambda operator: min(row[0].topology_rank for row in by_operator[operator]))
    )
    if frame.fixture_kind == "residual_cnn":
        _require(len(operator_order) >= 3, "missing_required_operator_family", "residual frame requires three operators")
        selected_operators = tuple(dict.fromkeys((operator_order[0], operator_order[(len(operator_order) - 1) // 2], operator_order[-1])))
        _require(len(selected_operators) == 3, "missing_required_operator_family", "coverage anchors collapsed")
    else:
        conv = tuple(operator for operator in operator_order if any(row[1].route_kind == "conv_channel_pair" for row in by_operator[operator]))
        linear = tuple(operator for operator in operator_order if any(row[1].route_kind == "linear_weight" for row in by_operator[operator]))
        _require(bool(conv and linear), "missing_required_operator_family", "VGG frame requires Conv and Linear")
        selected_operators = tuple(dict.fromkeys((conv[0], conv[(len(conv) - 1) // 2], linear[0])))
        _require(len(selected_operators) == 3, "missing_required_operator_family", "VGG coverage anchors collapsed")
    selected_atoms: list[CanarySelectedAtom] = []
    for operator_id in selected_operators:
        for annotation, block in by_operator[operator_id]:
            for local_index in tuple(dict.fromkeys((0, block.route_count - 1))):
                eligible_index = block.eligible_offset + local_index
                coordinate = _coordinate_for_flat_index(block, local_index)
                selected_atoms.append(
                    CanarySelectedAtom(
                        physical_route_alias=physical_route_alias(topology.eligible_route_axis_hash, eligible_index),
                        route_id=coordinate.route_id,
                        eligible_flat_index=eligible_index,
                        layer_id=block.layer_id,
                        route_kind=block.route_kind,
                        stage_id=annotation.stage_id,
                        block_id=annotation.block_id,
                        operator_id=annotation.operator_id,
                    )
                )
    selected_atoms.sort(key=lambda atom: atom.eligible_flat_index)
    branches: list[CanaryBranchSkeleton] = []
    empty: list[tuple[str, str]] = []
    for operator_id in selected_operators:
        operator_atoms = tuple(atom for atom in selected_atoms if atom.operator_id == operator_id)
        for role_index, role_id in enumerate(full_role_overlay.semantic_role_ids):
            members = tuple(atom.physical_route_alias for atom in operator_atoms if full_role_overlay.contains(role_index, atom.eligible_flat_index))
            if not members:
                empty.append((operator_id, role_id))
                continue
            branch_id = f"canary-branch:v1:{_stable_digest({'operator_id': operator_id, 'semantic_role_id': role_id, 'members': members})}"
            branches.append(CanaryBranchSkeleton(branch_id, operator_id, role_id, members))
    _require(len(branches) <= MAXIMUM_NONTERMINAL_BRANCHES, "branch_bound_exceeded", "coverage frame exceeds branch bound")
    result = CanaryCoverageSkeleton(
        frame_spec_hash=frame.content_hash,
        topology_receipt_hash=topology.content_hash,
        semantic_role_overlay_hash=full_role_overlay.semantic_hash,
        unit_assignment_hash=technical_units.content_hash,
        selected_atoms=tuple(selected_atoms),
        branches=tuple(branches),
        empty_intersection_hash=_stable_digest({"schema_name": "mechanistic_transport_empty_intersections_v1", "pairs": empty}),
        exclusion_hash=_stable_digest({"schema_name": "mechanistic_transport_coverage_exclusions_v1", "eligible_route_axis_hash": topology.eligible_route_axis_hash, "selected_operators": selected_operators, "selected_indices": [atom.eligible_flat_index for atom in selected_atoms]}),
    )
    return result


def _validate_entry(entry: StructuralCompletionEntry, route_count: int) -> None:
    _require(entry.requested_physical_route_aliases == tuple(sorted(set(entry.requested_physical_route_aliases))), "structural_completion_catalog_mismatch", "requested aliases are not canonical")
    _require(entry.completed_removed_eligible_indices == tuple(sorted(set(entry.completed_removed_eligible_indices))) and all(0 <= value < route_count for value in entry.completed_removed_eligible_indices), "structural_completion_catalog_mismatch", "completed eligible indices are invalid")
    _require(entry.completed_removed_physical_gate_ids == tuple(sorted(set(entry.completed_removed_physical_gate_ids))), "structural_completion_catalog_mismatch", "completed gate IDs are not canonical")
    _require_sha(entry.completion_lineage_hash, "structural_completion_catalog_mismatch")


def _catalog_map(catalog: StructuralCompletionCatalog, *, route_count: int) -> dict[tuple[str, ...], StructuralCompletionEntry]:
    for value in (catalog.topology_receipt_hash, catalog.eligible_route_axis_hash, catalog.completion_policy_hash, catalog.dense_physical_state_hash, catalog.topology_completion_certificate_hash):
        _require_sha(value, "structural_completion_catalog_mismatch")
    result: dict[tuple[str, ...], StructuralCompletionEntry] = {}
    for entry in catalog.entries:
        _validate_entry(entry, route_count)
        _require(entry.requested_physical_route_aliases not in result, "structural_completion_catalog_mismatch", "duplicate or competing completion request")
        result[entry.requested_physical_route_aliases] = entry
    return result


def _physical_state_hash(topology_hash: str, axis_hash: str, dense_hash: str, policy_hash: str, entry: StructuralCompletionEntry) -> str:
    return _stable_digest({"schema": "mechanistic_canary_physical_state_v1", "topology_receipt_hash": topology_hash, "eligible_route_axis_hash": axis_hash, "dense_physical_state_hash": dense_hash, "completion_policy_hash": policy_hash, "removed_eligible_indices": entry.completed_removed_eligible_indices, "removed_physical_gate_ids": entry.completed_removed_physical_gate_ids})


def _projector_hash(state_hash: str, axis_hash: str, indices: tuple[int, ...]) -> str:
    return _stable_digest({"schema": "mechanistic_canary_sparse_projector_v1", "physical_state_hash": state_hash, "eligible_route_axis_hash": axis_hash, "removed_eligible_indices": indices})


def compile_completed_singleton_displacements(
    *,
    skeleton: CanaryCoverageSkeleton,
    topology: TopologyAnnotationReceipt,
    completion_catalog: StructuralCompletionCatalog,
) -> CompletedSingletonDisplacementBank:
    _validate_topology(topology)
    _require(skeleton.topology_receipt_hash == topology.content_hash, "structural_completion_catalog_mismatch", "skeleton topology is stale")
    _require(completion_catalog.catalog_kind == "selected_singletons" and completion_catalog.topology_receipt_hash == topology.content_hash and completion_catalog.eligible_route_axis_hash == topology.eligible_route_axis_hash, "structural_completion_catalog_mismatch", "singleton catalog identity differs")
    catalog = _catalog_map(completion_catalog, route_count=topology.eligible_route_count)
    required = {(atom.physical_route_alias,) for atom in skeleton.selected_atoms}
    _require(set(catalog) == required, "structural_completion_catalog_mismatch", "singleton catalog is missing or has extra requests")
    cost_hash = _stable_digest({"schema_name": "mechanistic_transport_uniform_cost_profile_v1", "cost_kind": COST_KIND, "eligible_route_axis_hash": topology.eligible_route_axis_hash, "route_count": topology.eligible_route_count})
    receipts = []
    for atom in skeleton.selected_atoms:
        entry = catalog[(atom.physical_route_alias,)]
        _require(atom.eligible_flat_index in entry.completed_removed_eligible_indices, "invalid_completed_singleton", "singleton completion omitted selected route")
        state_hash = _physical_state_hash(topology.content_hash, topology.eligible_route_axis_hash, completion_catalog.dense_physical_state_hash, completion_catalog.completion_policy_hash, entry)
        receipts.append(
            CompletedSingletonDisplacementReceipt(
                coverage_skeleton_hash=skeleton.content_hash,
                branch_skeleton_hashes=tuple(branch.content_hash for branch in skeleton.branches if atom.physical_route_alias in branch.member_physical_route_aliases),
                selected_physical_route_alias=atom.physical_route_alias,
                selected_route_id=atom.route_id,
                eligible_flat_index=atom.eligible_flat_index,
                topology_receipt_hash=topology.content_hash,
                eligible_route_axis_hash=topology.eligible_route_axis_hash,
                dense_physical_state_hash=completion_catalog.dense_physical_state_hash,
                completion_policy_hash=completion_catalog.completion_policy_hash,
                completed_removed_eligible_indices=entry.completed_removed_eligible_indices,
                completed_removed_physical_gate_ids=entry.completed_removed_physical_gate_ids,
                completed_retained_state_hash=state_hash,
                sparse_hard_projector_hash=_projector_hash(state_hash, topology.eligible_route_axis_hash, entry.completed_removed_eligible_indices),
                saved_cost=len(entry.completed_removed_eligible_indices),
                physical_cost_profile_hash=cost_hash,
            )
        )
    _require(all(receipt.saved_cost > 0 for receipt in receipts), "nonpositive_singleton_completed_cost", "singleton saved cost must be positive")
    return CompletedSingletonDisplacementBank(skeleton.content_hash, tuple(receipts))


def _f64(value: float) -> float:
    result = struct.unpack("!d", struct.pack("!d", float(value)))[0]
    if not math.isfinite(result):
        _fail("nonfinite_or_overflow", "binary64 value is nonfinite")
    return 0.0 if result == 0.0 else result


def portable_binary64_runtime_receipt_hash() -> str:
    ulp = math.ulp(1.0)
    separate = ((1.0 + 2.0**-27) * (1.0 - 2.0**-27)) - 1.0
    observations = {
        "binary64_bytes": struct.calcsize("d") == 8 and struct.pack("!d", 1.0).hex() == "3ff0000000000000",
        "halfway_even_lower": 1.0 + ulp / 2.0 == 1.0,
        "halfway_even_upper": 1.0 + 3.0 * ulp / 2.0 == 1.0 + 2.0 * ulp,
        "separate_multiply_add": separate == 0.0,
        "signed_zero_normalized": math.copysign(1.0, _f64(-0.0)) == 1.0,
    }
    _require(all(observations.values()), "nonportable_binary64_runtime", "runtime failed the source-owned binary64/RNE probe")
    return _stable_digest({"schema_name": "mechanistic_portable_binary64_runtime_receipt_v1", "policy_id": BINARY64_POLICY_ID, "observations": observations})


def freeze_suffix_order(
    evidence: SuffixOrderEvidence,
    *,
    skeleton: CanaryCoverageSkeleton,
    completed_singletons: CompletedSingletonDisplacementBank,
    expected_owner_hashes: Mapping[str, str],
    negative_tolerance: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> SuffixOrderReceipt:
    branch = next((value for value in skeleton.branches if value.content_hash == evidence.branch_skeleton_hash), None)
    _require(branch is not None and evidence.branch_id == branch.branch_id, "stale_suffix_evidence", "suffix branch is stale")
    lineage = (evidence.coverage_skeleton_hash, evidence.branch_skeleton_hash, evidence.topology_receipt_hash, evidence.semantic_role_overlay_hash, evidence.completed_singleton_bank_hash, evidence.technical_unit_assignment_hash, evidence.scalar_oracle_hash, evidence.scalar_profile_hash, evidence.calibration_fit_hash, evidence.calibration_selection_face_hash, evidence.positive_scale_face_hash, evidence.identifiability_quotient_hash, evidence.gauge_range_hash, evidence.portable_reduction_receipt_hash)
    _require(all(_is_sha(value) for value in lineage), "stale_suffix_evidence", "suffix evidence contains an invalid owner identity")
    _require(evidence.coverage_skeleton_hash == skeleton.content_hash and evidence.topology_receipt_hash == skeleton.topology_receipt_hash and evidence.semantic_role_overlay_hash == skeleton.semantic_role_overlay_hash and evidence.technical_unit_assignment_hash == skeleton.unit_assignment_hash and evidence.completed_singleton_bank_hash == completed_singletons.content_hash and completed_singletons.coverage_skeleton_hash == skeleton.content_hash, "stale_suffix_evidence", "suffix evidence lineage differs")
    _require(set(expected_owner_hashes) == set(SUFFIX_OWNER_HASH_FIELDS) and all(_is_sha(value) for value in expected_owner_hashes.values()) and all(getattr(evidence, field) == expected_owner_hashes[field] for field in SUFFIX_OWNER_HASH_FIELDS), "stale_suffix_evidence", "suffix owner identity differs")
    singleton_aliases = tuple(receipt.selected_physical_route_alias for receipt in completed_singletons.receipts)
    _require(len(singleton_aliases) == len(set(singleton_aliases)) and all(singleton_aliases.count(alias) == 1 for alias in branch.member_physical_route_aliases), "stale_suffix_evidence", "suffix branch lacks an exact singleton displacement")
    _require(evidence.portable_reduction_receipt_hash == portable_binary64_runtime_receipt_hash(), "nonportable_binary64_runtime", "suffix evidence portable-reduction identity differs")
    _require(evidence.ordered_member_aliases == branch.member_physical_route_aliases and len(evidence.block_ids) == 4 and len(set(evidence.block_ids)) == 4 and len(evidence.block_effective_energy) == 4, "invalid_scalar_suffix_evidence", "suffix member or block axis differs")
    tolerances = tuple(_f64(value) for value in (negative_tolerance, absolute_tolerance, relative_tolerance))
    _require(all(value >= 0.0 for value in tolerances), "invalid_scalar_suffix_evidence", "suffix tolerances are negative")
    count = len(evidence.ordered_member_aliases)
    _require(all(len(row) == count for row in evidence.block_effective_energy), "invalid_scalar_suffix_evidence", "effective-energy matrix shape differs")
    costs = {receipt.selected_physical_route_alias: receipt.saved_cost for receipt in completed_singletons.receipts}
    retention = []
    priorities = []
    saved = []
    for column, alias in enumerate(evidence.ordered_member_aliases):
        values = []
        for row in evidence.block_effective_energy:
            value = _f64(row[column])
            _require(value >= -tolerances[0], "invalid_scalar_suffix_evidence", "effective energy is negative")
            values.append(max(0.0, value))
        energy = _f64(max(values))
        cost = int(costs.get(alias, 0))
        _require(cost > 0, "nonpositive_singleton_completed_cost", "suffix member has no completed cost")
        retention.append(energy)
        saved.append(cost)
        priorities.append(_f64(energy / cost))
    branch_scale = _f64(max(1.0, *(abs(value) for value in priorities)))
    tie_width = _f64(_f64(tolerances[1]) + _f64(tolerances[2] * branch_scale))
    _require(tie_width > 0.0, "invalid_scalar_suffix_evidence", "tie width is not positive")
    bins = []
    for priority in priorities:
        value = math.floor(_f64(_f64(priority / tie_width) + 0.5))
        _require(0 <= value <= np.iinfo(np.uint64).max, "nonfinite_or_overflow", "priority bin exceeds uint64")
        bins.append(int(value))
    ordered = tuple(alias for _, alias in sorted(zip(bins, evidence.ordered_member_aliases, strict=True), key=lambda item: (item[0], item[1])))
    return SuffixOrderReceipt(
        evidence_hash=evidence.content_hash,
        coverage_skeleton_hash=skeleton.content_hash,
        branch_skeleton_hash=branch.content_hash,
        completed_singleton_bank_hash=completed_singletons.content_hash,
        ordered_member_aliases=evidence.ordered_member_aliases,
        negative_tolerance=tolerances[0],
        absolute_tolerance=tolerances[1],
        relative_tolerance=tolerances[2],
        retention_energy=tuple(retention),
        compiler_owned_saved_cost=tuple(saved),
        priority=tuple(priorities),
        branch_scale=branch_scale,
        tie_width=tie_width,
        priority_bin_uint64=tuple(bins),
        ordered_physical_route_aliases=ordered,
        binary64_policy_id=BINARY64_POLICY_ID,
    )


def compile_transport_canary_logical_universe(
    *,
    skeleton: CanaryCoverageSkeleton,
    suffix_order_receipts: tuple[SuffixOrderReceipt, ...],
    completed_singletons: CompletedSingletonDisplacementBank,
    technical_units: TechnicalCanaryUnitAssignment,
) -> CanaryLogicalUniverseSkeleton:
    _validate_units(technical_units)
    _require(skeleton.unit_assignment_hash == technical_units.content_hash and completed_singletons.coverage_skeleton_hash == skeleton.content_hash, "grouped_unit_overlap", "unit or singleton assignment is stale")
    singleton_aliases = tuple(receipt.selected_physical_route_alias for receipt in completed_singletons.receipts)
    _require(len(singleton_aliases) == len(set(singleton_aliases)) and all(singleton_aliases.count(alias) == 1 for branch in skeleton.branches for alias in branch.member_physical_route_aliases), "stale_suffix_evidence", "branch universe lacks exact singleton displacements")
    by_branch = {receipt.branch_skeleton_hash: receipt for receipt in suffix_order_receipts}
    _require(len(by_branch) == len(suffix_order_receipts) == len(skeleton.branches), "stale_suffix_evidence", "suffix receipt axis differs")
    chains = []
    hashes = []
    count = 1
    for branch in skeleton.branches:
        receipt = by_branch.get(branch.content_hash)
        _require(receipt is not None and receipt.coverage_skeleton_hash == skeleton.content_hash and receipt.completed_singleton_bank_hash == completed_singletons.content_hash and receipt.ordered_member_aliases == branch.member_physical_route_aliases and set(receipt.ordered_physical_route_aliases) == set(branch.member_physical_route_aliases) and len(receipt.ordered_physical_route_aliases) == len(branch.member_physical_route_aliases) and receipt.binary64_policy_id == BINARY64_POLICY_ID, "stale_suffix_evidence", "branch suffix receipt is stale")
        chains.append((branch.branch_id, receipt.ordered_physical_route_aliases))
        hashes.append(receipt.content_hash)
        count *= len(receipt.ordered_physical_route_aliases) + 1
    return CanaryLogicalUniverseSkeleton(skeleton.content_hash, completed_singletons.content_hash, tuple(hashes), tuple(chains), count)


def _logical_states(universe: CanaryLogicalUniverseSkeleton):
    ranges = tuple(range(len(chain) + 1) for _, chain in universe.branch_chains)
    for counts in itertools.product(*ranges):
        removed = tuple(sorted(set(alias for count, (_, chain) in zip(counts, universe.branch_chains, strict=True) for alias in chain[:count])))
        branch_counts = tuple((branch_id, count) for count, (branch_id, _) in zip(counts, universe.branch_chains, strict=True))
        logical_hash = _stable_digest({"schema_name": "mechanistic_transport_logical_state_v1", "branch_removed_counts": branch_counts, "requested_physical_route_aliases": removed})
        yield logical_hash, removed


def seal_transport_canary_frame(
    *,
    skeleton: CanaryCoverageSkeleton, logical_universe: CanaryLogicalUniverseSkeleton,
    topology: TopologyAnnotationReceipt,
    completed_singletons: CompletedSingletonDisplacementBank,
    completion_catalog: StructuralCompletionCatalog,
    technical_units: TechnicalCanaryUnitAssignment,
) -> TechnicalCanaryCompileReceipt:
    _validate_topology(topology)
    _validate_units(technical_units)
    _require(logical_universe.coverage_skeleton_hash == skeleton.content_hash == completed_singletons.coverage_skeleton_hash and logical_universe.completed_singleton_bank_hash == completed_singletons.content_hash and technical_units.content_hash == skeleton.unit_assignment_hash and completion_catalog.catalog_kind == "logical_suffix_universe" and completion_catalog.topology_receipt_hash == topology.content_hash and completion_catalog.eligible_route_axis_hash == topology.eligible_route_axis_hash, "structural_completion_catalog_mismatch", "physical-seal lineage differs")
    catalog = _catalog_map(completion_catalog, route_count=topology.eligible_route_count)
    logical_by_request: dict[tuple[str, ...], list[str]] = {}
    observed_count = 0
    for logical_hash, request in _logical_states(logical_universe):
        observed_count += 1
        logical_by_request.setdefault(request, []).append(logical_hash)
    _require(observed_count == logical_universe.logical_state_count and set(catalog) == set(logical_by_request), "structural_completion_catalog_mismatch", "logical universe catalog differs")
    singleton_by_request = {(receipt.selected_physical_route_alias,): receipt for receipt in completed_singletons.receipts}
    physical: dict[str, tuple[StructuralCompletionEntry, list[str], str]] = {}
    payload_to_hash: dict[tuple[tuple[int, ...], tuple[str, ...]], str] = {}
    for request, logical_hashes in logical_by_request.items():
        entry = catalog[request]
        if not request:
            _require(not entry.completed_removed_eligible_indices and not entry.completed_removed_physical_gate_ids, "structural_completion_catalog_mismatch", "empty completion is not the dense identity")
        requested_indices = {int(alias.rsplit(":", 1)[1]) for alias in request}
        _require(requested_indices.issubset(entry.completed_removed_eligible_indices), "structural_completion_catalog_mismatch", "completion dropped requested routes")
        state_hash = _physical_state_hash(topology.content_hash, topology.eligible_route_axis_hash, completion_catalog.dense_physical_state_hash, completion_catalog.completion_policy_hash, entry)
        projector = _projector_hash(state_hash, topology.eligible_route_axis_hash, entry.completed_removed_eligible_indices)
        if request in singleton_by_request:
            prior = singleton_by_request[request]
            _require(prior.completed_removed_eligible_indices == entry.completed_removed_eligible_indices and prior.completed_removed_physical_gate_ids == entry.completed_removed_physical_gate_ids and prior.completed_retained_state_hash == state_hash and prior.sparse_hard_projector_hash == projector, "structural_completion_catalog_mismatch", "singleton completion changed in whole universe")
        payload = (entry.completed_removed_eligible_indices, entry.completed_removed_physical_gate_ids)
        previous_hash = payload_to_hash.setdefault(payload, state_hash)
        _require(previous_hash == state_hash, "structural_completion_catalog_mismatch", "one physical payload has two hashes")
        if state_hash in physical:
            previous_entry, aliases, previous_projector = physical[state_hash]
            _require((previous_entry.completed_removed_eligible_indices, previous_entry.completed_removed_physical_gate_ids) == payload and previous_projector == projector, "structural_completion_catalog_mismatch", "one state hash has conflicting payload")
            aliases.extend(logical_hashes)
        else:
            physical[state_hash] = (entry, list(logical_hashes), projector)
    _require(len(physical) <= logical_universe.logical_state_count, "structural_completion_catalog_mismatch", "deterministic completion cardinality violated")
    _require(len(physical) <= MAXIMUM_CANONICAL_PHYSICAL_STATES, "physical_state_bound_exceeded", "canonical physical state bound exceeded")
    aliases = []
    for state_hash in sorted(physical):
        entry, logical_hashes, projector = physical[state_hash]
        aliases.append(
            CanaryPhysicalAlias(
                physical_state_hash=state_hash,
                logical_state_hashes=tuple(sorted(set(logical_hashes))),
                retained_cost=topology.eligible_route_count - len(entry.completed_removed_eligible_indices),
                completed_removed_eligible_indices=entry.completed_removed_eligible_indices,
                completed_removed_physical_gate_ids=entry.completed_removed_physical_gate_ids,
                sparse_projector_hash=projector,
            )
        )
    alias_tuple = tuple(aliases)
    return TechnicalCanaryCompileReceipt(
        logical_universe_hash=logical_universe.content_hash,
        topology_receipt_hash=topology.content_hash,
        semantic_role_overlay_hash=skeleton.semantic_role_overlay_hash,
        completed_singleton_bank_hash=completed_singletons.content_hash,
        canonical_physical_aliases=alias_tuple,
        physical_alias_hash=_stable_digest({"schema_name": "mechanistic_transport_physical_alias_axis_v1", "aliases": [_receipt_envelope(alias) for alias in alias_tuple]}),
        physical_cost_hash=_stable_digest({"schema_name": "mechanistic_transport_physical_cost_axis_v1", "cost_kind": COST_KIND, "costs": [alias.retained_cost for alias in alias_tuple]}),
        sparse_projector_bank_hash=_stable_digest({"schema_name": "mechanistic_transport_sparse_projector_bank_v1", "projectors": [alias.sparse_projector_hash for alias in alias_tuple]}),
        completion_policy_hash=completion_catalog.completion_policy_hash,
        unit_assignment_hash=skeleton.unit_assignment_hash,
    )
